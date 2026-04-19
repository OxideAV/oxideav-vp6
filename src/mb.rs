//! VP6 macroblock decode — MV parsing, coefficient parsing,
//! reconstruction (intra + inter MC + IDCT residual add).
//!
//! Ports large chunks of FFmpeg's `vp56.c` and `vp6.c`. The file
//! deliberately operates on simple row-major `Vec<u8>` buffers for the
//! Y/U/V planes instead of FFmpeg's `AVFrame` so that the code keeps
//! within `oxideav-core`'s `VideoFrame` model with no extra ceremony.

use crate::dsp;
use crate::models::Vp6Model;
use crate::range_coder::RangeCoder;
use crate::tables;

/// Per-block reference DC context kept in `left_block` / `above_blocks`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefDc {
    pub not_null_dc: bool,
    pub ref_frame: RefKind,
    pub dc_coeff: i16,
}

/// Which reference a block was coded against — tracked per block for
/// DC prediction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RefKind {
    #[default]
    None,
    Current,
    Previous,
    Golden,
}

impl From<tables::RefFrame> for RefKind {
    fn from(r: tables::RefFrame) -> Self {
        match r {
            tables::RefFrame::Current => RefKind::Current,
            tables::RefFrame::Previous => RefKind::Previous,
            tables::RefFrame::Golden => RefKind::Golden,
        }
    }
}

/// A motion vector in quarter-pel units for luma (eighth-pel for chroma).
/// VP6 tracks x/y as signed 16-bit to match FFmpeg's `VP56mv`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub x: i16,
    pub y: i16,
}

/// Cached per-MB metadata (type + shared MV). Matches `VP56Macroblock`.
#[derive(Clone, Copy, Debug)]
pub struct MacroblockInfo {
    pub mb_type: tables::Vp56Mb,
    pub mv: Mv,
}

impl Default for MacroblockInfo {
    fn default() -> Self {
        Self {
            mb_type: tables::Vp56Mb::Intra,
            mv: Mv::default(),
        }
    }
}

/// Per-block sub-pel-corrected coefficients + IDCT selector, populated
/// by [`parse_coeff`] then consumed by [`render_mb_intra`] /
/// [`render_mb_inter`].
#[derive(Clone, Debug)]
pub struct BlockScratch {
    pub block_coeff: [[i16; 64]; 6],
    pub idct_selector: [u8; 6],
    /// `true` when the current frame is a keyframe — drives the
    /// coeff-models branch that copies `def_prob` through on every
    /// non-updated node.
    pub keyframe: bool,
    /// Quantiser-derived scale factors (updated at header-parse time).
    pub dequant_dc: i32,
    pub dequant_ac: i32,
    /// VP3 loop-filter bounding-value table, keyed by `127 + delta`.
    pub bounding_values: [i32; 256],
    /// `prev_dc[plane][ref_frame]` — last decoded DC per plane/ref.
    pub prev_dc: [[i16; 3]; 3],
    /// Left-block DC/ref context, 4-wide.
    pub left_block: [RefDc; 4],
    /// Above-block DC/ref context, (4*mb_width + 6) wide.
    pub above_blocks: Vec<RefDc>,
    /// Into-`above_blocks` indices for blocks 0..=5 of the current MB.
    pub above_block_idx: [usize; 6],
    /// The MB-type picked by the decoder this call. Mirrors
    /// FFmpeg's `s->mb_type`.
    pub mb_type: tables::Vp56Mb,
    /// Per-block motion vectors (for 4V / chroma).
    pub mv: [Mv; 6],
    /// Per-MB MV candidate lookup from neighbours — used by delta MV.
    pub vector_candidate: [Mv; 2],
    pub vector_candidate_pos: i32,
}

impl BlockScratch {
    pub fn new(mb_width: usize) -> Self {
        Self {
            block_coeff: [[0; 64]; 6],
            idct_selector: [0; 6],
            keyframe: false,
            dequant_dc: 0,
            dequant_ac: 0,
            bounding_values: [0; 256],
            prev_dc: [[0; 3]; 3],
            left_block: [RefDc::default(); 4],
            above_blocks: vec![RefDc::default(); 4 * mb_width + 6],
            above_block_idx: [0; 6],
            mb_type: tables::Vp56Mb::Intra,
            mv: [Mv::default(); 6],
            vector_candidate: [Mv::default(); 2],
            vector_candidate_pos: 0,
        }
    }

    /// Re-size the above-blocks row when mb_width changes. Clears the
    /// DC-predictor state.
    pub fn reset_row(&mut self, mb_width: usize) {
        if self.above_blocks.len() != 4 * mb_width + 6 {
            self.above_blocks = vec![RefDc::default(); 4 * mb_width + 6];
        } else {
            for b in self.above_blocks.iter_mut() {
                *b = RefDc::default();
            }
        }
        // Matches FFmpeg's two "sentinels" that force the first chroma
        // DC to treat the first above-block as CURRENT.
        if 2 * mb_width + 2 < self.above_blocks.len() {
            self.above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < self.above_blocks.len() {
            self.above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        self.prev_dc = [[0; 3]; 3];
        self.prev_dc[1][ref_kind_index(RefKind::Current)] = 128;
        self.prev_dc[2][ref_kind_index(RefKind::Current)] = 128;
    }

    /// Set the initial per-row block state that FFmpeg re-primes at the
    /// top of each MB row.
    pub fn start_row(&mut self, mb_width: usize) {
        for b in &mut self.left_block {
            *b = RefDc::default();
        }
        self.above_block_idx[0] = 1;
        self.above_block_idx[1] = 2;
        self.above_block_idx[2] = 1;
        self.above_block_idx[3] = 2;
        self.above_block_idx[4] = 2 * mb_width + 2 + 1;
        self.above_block_idx[5] = 3 * mb_width + 4 + 1;
    }

    /// Advance the above-block / MV-context indices by one MB column.
    pub fn advance_column(&mut self) {
        for y in 0..4 {
            self.above_block_idx[y] += 2;
        }
        for uv in 4..6 {
            self.above_block_idx[uv] += 1;
        }
    }
}

#[inline]
fn ref_kind_index(r: RefKind) -> usize {
    match r {
        RefKind::Current => 0,
        RefKind::Previous => 1,
        RefKind::Golden => 2,
        RefKind::None => 0,
    }
}

/// Install quantiser-derived dequantisers into the scratch. Matches
/// `ff_vp56_init_dequant`.
pub fn init_dequant(scratch: &mut BlockScratch, qp: u8) {
    let q = qp as usize & 0x3F;
    scratch.dequant_dc = (tables::VP56_DC_DEQUANT[q] as i32) << 2;
    scratch.dequant_ac = (tables::VP56_AC_DEQUANT[q] as i32) << 2;
    dsp::set_bounding_values(
        &mut scratch.bounding_values,
        tables::VP56_FILTER_THRESHOLD[q],
    );
}

/// Parse one MB's coefficient block. Port of `vp6_parse_coeff`.
pub fn parse_coeff(model: &Vp6Model, scratch: &mut BlockScratch, rac: &mut RangeCoder<'_>) -> bool {
    if rac.is_end() {
        return false;
    }
    let permute = &tables::IDCT_SCANTABLE;
    for block in &mut scratch.block_coeff {
        for c in block.iter_mut() {
            *c = 0;
        }
    }

    for b in 0..6usize {
        let mut ct: usize = 1;
        let mut run: i32 = 1;
        let pt = if b > 3 { 1 } else { 0 };

        let ctx = scratch.left_block[tables::B6_TO_4[b] as usize].not_null_dc as usize
            + scratch.above_blocks[scratch.above_block_idx[b]].not_null_dc as usize;

        let mut model1_idx: ModelRef = ModelRef::Dccv(pt);
        let mut model2_idx: ModelRef = ModelRef::Dcct(pt, ctx);

        let mut coeff_idx: usize = 0;
        loop {
            let model2 = model_ref(model, model2_idx);
            let model1 = model_ref(model, model1_idx);
            let m2_0 = model2[0];
            let m2_1 = model2[1];

            let go_parse = (coeff_idx > 1 && ct == 0) || rac.get_prob(m2_0) != 0;

            if go_parse {
                let coeff: i32;
                if rac.get_prob(model2[2]) != 0 {
                    if rac.get_prob(model2[3]) != 0 {
                        let idx = rac.get_tree(tables::PC_TREE, model1) as usize;
                        let mut c = tables::VP56_COEFF_BIAS[idx + 5] as i32;
                        let bit_len = tables::VP56_COEFF_BIT_LENGTH[idx] as i32;
                        for i in (0..=bit_len).rev() {
                            c += (rac.get_prob(tables::VP56_COEFF_PARSE_TABLE[idx][i as usize])
                                as i32)
                                << i;
                        }
                        coeff = c;
                    } else if rac.get_prob(model2[4]) != 0 {
                        coeff = 3 + rac.get_prob(model1[5]) as i32;
                    } else {
                        coeff = 2;
                    }
                    ct = 2;
                } else {
                    ct = 1;
                    coeff = 1;
                }
                let sign = rac.get_bit() as i32;
                let mut coeff_signed = if sign != 0 { -coeff } else { coeff };
                if coeff_idx != 0 {
                    coeff_signed *= scratch.dequant_ac;
                }
                let idx = model.coeff_index_to_pos[coeff_idx] as usize;
                scratch.block_coeff[b][permute[idx] as usize] =
                    coeff_signed.clamp(-32768, 32767) as i16;
                run = 1;
            } else {
                ct = 0;
                if coeff_idx > 0 {
                    if rac.get_prob(m2_1) == 0 {
                        break;
                    }
                    let model3_idx = if coeff_idx >= 6 { 1 } else { 0 };
                    let model3 = &model.coeff_runv[model3_idx];
                    let rv = rac.get_tree(tables::VP6_PCR_TREE, model3);
                    run = if rv == 0 {
                        let mut r = 9i32;
                        for i in 0..6i32 {
                            r += (rac.get_prob(model3[(i + 8) as usize]) as i32) << i;
                        }
                        r
                    } else {
                        rv
                    };
                }
            }

            let nidx = coeff_idx as i32 + run;
            if nidx >= 64 {
                coeff_idx = 64;
                break;
            }
            coeff_idx = nidx as usize;
            let cg = tables::VP6_COEFF_GROUPS[coeff_idx] as usize;
            model1_idx = ModelRef::Ract(pt, ct, cg);
            model2_idx = ModelRef::Ract(pt, ct, cg);
        }

        let has_nonzero_dc = scratch.block_coeff[b][0] != 0;
        scratch.left_block[tables::B6_TO_4[b] as usize].not_null_dc = has_nonzero_dc;
        scratch.above_blocks[scratch.above_block_idx[b]].not_null_dc = has_nonzero_dc;
        scratch.idct_selector[b] = model.coeff_index_to_idct_selector[coeff_idx.min(63)];
    }
    true
}

/// Model slice selector — since we mutate `scratch` and borrow from
/// `model` concurrently, we use this enum plus the helper
/// [`model_ref`] to resolve references afresh each iteration.
#[derive(Clone, Copy, Debug)]
enum ModelRef {
    Dccv(usize),
    Dcct(usize, usize),
    Ract(usize, usize, usize),
}

fn model_ref(model: &Vp6Model, r: ModelRef) -> &[u8] {
    match r {
        ModelRef::Dccv(pt) => &model.coeff_dccv[pt],
        ModelRef::Dcct(pt, ctx) => &model.coeff_dcct[pt][ctx],
        ModelRef::Ract(pt, ct, cg) => &model.coeff_ract[pt][ct][cg],
    }
}

/// Apply DC prediction on each of the 6 blocks — adds the (above + left
/// avg) DC to the current block. Matches `vp56_add_predictors_dc`.
pub fn add_predictors_dc(scratch: &mut BlockScratch, ref_frame: RefKind) {
    let idx0 = tables::IDCT_SCANTABLE[0] as usize;
    for b in 0..6usize {
        let abi = scratch.above_block_idx[b];
        let lbi = tables::B6_TO_4[b] as usize;

        let ab = scratch.above_blocks[abi];
        let lb = scratch.left_block[lbi];

        let mut count = 0i32;
        let mut dc = 0i32;
        if ref_frame != RefKind::None && lb.ref_frame == ref_frame {
            dc += lb.dc_coeff as i32;
            count += 1;
        }
        if ref_frame != RefKind::None && ab.ref_frame == ref_frame {
            dc += ab.dc_coeff as i32;
            count += 1;
        }
        let dc = match count {
            0 => scratch.prev_dc[tables::B2P[b] as usize][ref_kind_index(ref_frame)] as i32,
            2 => dc / 2,
            _ => dc,
        };

        let new_dc = (scratch.block_coeff[b][idx0] as i32 + dc)
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        scratch.block_coeff[b][idx0] = new_dc;
        scratch.prev_dc[tables::B2P[b] as usize][ref_kind_index(ref_frame)] = new_dc;
        scratch.above_blocks[abi].dc_coeff = new_dc;
        scratch.above_blocks[abi].ref_frame = ref_frame;
        scratch.left_block[lbi].dc_coeff = new_dc;
        scratch.left_block[lbi].ref_frame = ref_frame;

        // Finally dequantise DC and promote back into the block.
        scratch.block_coeff[b][idx0] =
            (new_dc as i32 * scratch.dequant_dc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Emit the 6 reconstructed 8x8 tiles for an intra MB into the
/// Y/U/V planes at `(mb_row, mb_col)`. IDCTs the stored coefficients
/// with the "put" semantics.
pub fn render_mb_intra(
    scratch: &mut BlockScratch,
    y_plane: &mut [u8],
    y_stride: usize,
    u_plane: &mut [u8],
    u_stride: usize,
    v_plane: &mut [u8],
    v_stride: usize,
    mb_row: usize,
    mb_col: usize,
) {
    for b in 0..6usize {
        let (plane, base, stride) = match tables::B2P[b] {
            0 => {
                let (xoff, yoff) = luma_block_offset(b);
                let base = (mb_row * 16 + yoff) * y_stride + (mb_col * 16 + xoff);
                (PlaneRef::Y, base, y_stride)
            }
            1 => {
                let base = (mb_row * 8) * u_stride + mb_col * 8;
                (PlaneRef::U, base, u_stride)
            }
            2 => {
                let base = (mb_row * 8) * v_stride + mb_col * 8;
                (PlaneRef::V, base, v_stride)
            }
            _ => continue,
        };
        let mut block = scratch.block_coeff[b];
        let sel = scratch.idct_selector[b];
        match plane {
            PlaneRef::Y => idct_put_selector(&mut y_plane[base..], y_stride, &mut block, sel),
            PlaneRef::U => idct_put_selector(&mut u_plane[base..], u_stride, &mut block, sel),
            PlaneRef::V => idct_put_selector(&mut v_plane[base..], v_stride, &mut block, sel),
        }
        // scratch doesn't need the block kept.
        scratch.block_coeff[b] = block;
        let _ = stride; // stride equals the plane's own stride; used above.
    }
}

/// Rebuild an inter MB: copy-or-filter from `ref_frame` then add the
/// IDCT residual. Handles `VP56_MB_INTER_NOVEC_*` and the 7 other
/// inter variants (including 4V) via per-block MVs cached in `scratch`.
///
/// When `deblock_filtering` is set (per-frame flag, controlled by the
/// header bit `s->deblock_filtering`), we apply the VP3-style 4-tap
/// edge filter on the source-tile scratch across any 8x8 block boundary
/// that falls inside the MC region — this matches FFmpeg's
/// `vp56_deblock_filter` call in `vp56_mc`.
#[allow(clippy::too_many_arguments)]
pub fn render_mb_inter(
    scratch: &mut BlockScratch,
    y_plane: &mut [u8],
    y_stride: usize,
    u_plane: &mut [u8],
    u_stride: usize,
    v_plane: &mut [u8],
    v_stride: usize,
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    mb_row: usize,
    mb_col: usize,
    plane_w: [usize; 3],
    plane_h: [usize; 3],
    use_bicubic_luma: bool,
    deblock_filtering: bool,
) {
    // 12x12 source tile: 2 pixels of pad on every side (FFmpeg uses a
    // 12x16 scratch in `vp56_mc`; the extra width is alignment, not a
    // semantic need). The put-target 8x8 block sits at position (2, 2).
    //
    // The VP3-style loop filter reads `p[-2..+2]` across an edge, so we
    // need 2 pixels of context on each side — hence the pad width.
    const TILE: usize = 12;
    const TILE_PAD: usize = 2;

    for b in 0..6usize {
        let (plane, stride) = match tables::B2P[b] {
            0 => (PlaneRef::Y, y_stride),
            1 => (PlaneRef::U, u_stride),
            2 => (PlaneRef::V, v_stride),
            _ => continue,
        };

        let (px_xoff, px_yoff) = match plane {
            PlaneRef::Y => luma_block_offset(b),
            _ => (0, 0),
        };
        let pw = plane_w[plane as usize];
        let ph = plane_h[plane as usize];

        // Block origin in pixels (top-left of the 8x8 tile).
        let base_x = match plane {
            PlaneRef::Y => mb_col * 16 + px_xoff,
            _ => mb_col * 8,
        };
        let base_y = match plane {
            PlaneRef::Y => mb_row * 16 + px_yoff,
            _ => mb_row * 8,
        };

        let coord_div = tables::VP6_COORD_DIV[b] as i32;
        let mv = scratch.mv[b];
        let dx = mv.x as i32 / coord_div;
        let dy = mv.y as i32 / coord_div;

        // Sub-pel position (0..=7 in luma, 0..=7 in chroma — both after
        // the `*2` for luma inside `vp6_filter`).
        let mask = coord_div - 1;
        let x8 = (mv.x as i32 & mask) * if matches!(plane, PlaneRef::Y) { 2 } else { 1 };
        let y8 = (mv.y as i32 & mask) * if matches!(plane, PlaneRef::Y) { 2 } else { 1 };

        // Output tile in the current frame.
        let dst_base = base_y * stride + base_x;

        let ref_plane = match plane {
            PlaneRef::Y => ref_y,
            PlaneRef::U => ref_u,
            PlaneRef::V => ref_v,
        };
        // Materialise a 12x12 reference-tile scratch around `(base_x +
        // dx - TILE_PAD, base_y + dy - TILE_PAD)`. We clamp sample
        // positions so edge MVs stay well-defined.
        let mut temp = [0u8; TILE * TILE];
        for r in 0..TILE {
            for c in 0..TILE {
                let sx = (base_x as i32 + dx + c as i32 - TILE_PAD as i32).clamp(0, pw as i32 - 1);
                let sy = (base_y as i32 + dy + r as i32 - TILE_PAD as i32).clamp(0, ph as i32 - 1);
                temp[r * TILE + c] = ref_plane[sy as usize * stride + sx as usize];
            }
        }

        // VP3-style edge filter on the scratch, applied across the
        // nearest 8x8-aligned edge in the reference frame when the
        // integer-pel component of the MV straddles a block. We use
        // `rem_euclid` so negative MVs still map into [0, 7] the way
        // FFmpeg's signed `& 7` does on 2's-complement.
        //
        // Edge column = `TILE_PAD + (8 - dx_mod8)` = `10 - dx_mod8`.
        // Edge row = `TILE_PAD + (8 - dy_mod8)` = `10 - dy_mod8`.
        // The filter reads 2 pixels before + writes 2 in-place, hence
        // the 12-wide pad.
        if deblock_filtering {
            let dx_mod = dx.rem_euclid(8);
            let dy_mod = dy.rem_euclid(8);
            let bounding = scratch.bounding_values;
            if dx_mod != 0 {
                let col = 10 - dx_mod as usize;
                // Filter 12 rows, so we run top-to-bottom of the whole
                // 12-tall scratch. `first_pixel` is the row 0 column
                // index of the left neighbour of the edge.
                dsp::h_loop_filter_12(&mut temp, col, TILE, &bounding);
            }
            if dy_mod != 0 {
                let row = 10 - dy_mod as usize;
                dsp::v_loop_filter_12(&mut temp, row * TILE, TILE, &bounding);
            }
        }

        let src_base_in_temp = TILE_PAD * TILE + TILE_PAD;

        let want_bicubic = matches!(plane, PlaneRef::Y) && use_bicubic_luma;

        let dst_plane: &mut [u8] = match plane {
            PlaneRef::Y => y_plane,
            PlaneRef::U => u_plane,
            PlaneRef::V => v_plane,
        };

        if x8 == 0 && y8 == 0 {
            dsp::put_block8(dst_plane, dst_base, stride, &temp, src_base_in_temp, TILE);
        } else if want_bicubic && (x8 == 0 || y8 == 0) {
            let phase = if x8 != 0 { x8 } else { y8 } as usize;
            let delta = if x8 != 0 { 1 } else { TILE as i32 };
            let select = 16usize;
            let weights = &tables::VP6_BLOCK_COPY_FILTER[select][phase];
            dsp::filter_hv4_into(
                dst_plane,
                dst_base,
                stride,
                &temp,
                src_base_in_temp,
                TILE,
                delta,
                weights,
            );
        } else if want_bicubic {
            let select = 16usize;
            let h = &tables::VP6_BLOCK_COPY_FILTER[select][x8 as usize];
            let v = &tables::VP6_BLOCK_COPY_FILTER[select][y8 as usize];
            dsp::filter_diag4(
                dst_plane,
                dst_base,
                stride,
                &temp,
                src_base_in_temp,
                TILE,
                h,
                v,
            );
        } else {
            dsp::put_h264_chroma8(
                dst_plane,
                dst_base,
                stride,
                &temp,
                src_base_in_temp,
                TILE,
                x8,
                y8,
            );
        }

        // Residual add.
        let mut block = scratch.block_coeff[b];
        let sel = scratch.idct_selector[b];
        idct_add_selector(&mut dst_plane[dst_base..], stride, &mut block, sel);
        scratch.block_coeff[b] = block;
    }
}

#[derive(Clone, Copy, Debug)]
enum PlaneRef {
    Y = 0,
    U = 1,
    V = 2,
}

/// Luma block origin offset within the 16x16 MB, for blocks 0..=3.
#[inline]
fn luma_block_offset(b: usize) -> (usize, usize) {
    let x = if b == 1 || b == 3 { 8 } else { 0 };
    let y = if b == 2 || b == 3 { 8 } else { 0 };
    (x, y)
}

fn idct_put_selector(dst: &mut [u8], stride: usize, block: &mut [i16; 64], selector: u8) {
    if selector > 10 || selector == 1 {
        dsp::idct_put(dst, stride, block);
    } else {
        // 10-coefficient IDCT fast path — matches `ff_vp3dsp_idct10_put`.
        // We just fall through to the full IDCT for correctness
        // (numerically identical modulo truncation in the skipped
        // taps); the performance delta isn't worth the code duplication
        // for the initial port.
        dsp::idct_put(dst, stride, block);
    }
}

fn idct_add_selector(dst: &mut [u8], stride: usize, block: &mut [i16; 64], selector: u8) {
    if selector > 1 {
        // Both `idct10` and full-IDCT fall through to the same full
        // transform in this initial port — performance-only variants
        // can slot in later without touching callers.
        dsp::idct_add(dst, stride, block);
    } else {
        dsp::idct_dc_add(dst, stride, block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_matches_reference() {
        let mut scratch = BlockScratch::new(16);
        init_dequant(&mut scratch, 0);
        assert_eq!(scratch.dequant_dc, 47 << 2);
        assert_eq!(scratch.dequant_ac, 94 << 2);
        init_dequant(&mut scratch, 63);
        assert_eq!(scratch.dequant_dc, 2 << 2);
        assert_eq!(scratch.dequant_ac, 1 << 2);
    }

    #[test]
    fn reset_row_seeds_prev_dc_128() {
        let mut scratch = BlockScratch::new(29);
        scratch.reset_row(29);
        assert_eq!(scratch.prev_dc[1][0], 128);
        assert_eq!(scratch.prev_dc[2][0], 128);
        assert_eq!(scratch.above_blocks[2 * 29 + 2].ref_frame, RefKind::Current);
    }

    #[test]
    fn add_predictors_propagates_dc_across_mb() {
        let mut scratch = BlockScratch::new(16);
        scratch.reset_row(16);
        scratch.start_row(16);
        init_dequant(&mut scratch, 10);
        // Every block starts with DC=5 and `Current` ref frame. The
        // first luma block has no same-ref neighbours so picks up
        // `prev_dc[0][0] = 0` — subsequent luma blocks then pick up
        // the DC from the first via `left_block` / `above_blocks`
        // sharing, so `prev_dc` climbs by at least the raw DC each
        // step.
        for b in 0..6 {
            scratch.block_coeff[b][tables::IDCT_SCANTABLE[0] as usize] = 5;
        }
        add_predictors_dc(&mut scratch, RefKind::Current);
        assert!(scratch.prev_dc[0][0] >= 5);
    }
}

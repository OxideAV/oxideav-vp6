//! VP6F (VP6 for FLV) encoder.
//!
//! This is a keyframe-only encoder that emits both DC and AC coefficients
//! through the bool-coded coefficient path the decoder reads in
//! [`crate::mb::parse_coeff`]. Output is a standards-compliant VP6F
//! elementary stream that our own decoder — and ffmpeg's `vp6f` decoder
//! — round-trip.
//!
//! Scope:
//!
//! * Keyframes only (no P-frames).
//! * No motion vectors; no loop filter emission.
//! * Interlaced flag = 0, `sub_version = 0` (simple profile),
//!   `filter_header = 0`, bool-path coefficients (no Huffman).
//!
//! The forward DCT is a float-based DCT-II scaled so `block[u*8+v] =
//! F[u,v]` feeds through the decoder's two-stage IDCT back to the same
//! tile (within rounding). Natural raster layout — `u` vertical freq,
//! `v` horizontal freq — matches the VP6 spec (section 12.1 / 16).
//! Quantisation is a nearest-integer division by `dequant_ac` for AC
//! bins, matched to the decoder's multiply-back.

use oxideav_core::{Error, Result};

use crate::mb::RefKind;
use crate::models::Vp6Model;
use crate::range_coder::RangeEncoder;
use crate::tables;

/// Output of a single-frame encode: the concatenated VP6F elementary-
/// stream bytes (no FLV tag wrapping — callers prepend the 1-byte FLV
/// frame adjuster, typically `0x00`).
pub type EncodedFrame = Vec<u8>;

/// Per-block reference-DC context kept in `left_block` / `above_blocks`
/// on the encoder side. Mirrors [`crate::mb::RefDc`] but stays owned by
/// the encoder so we don't tangle encoder state into decoder types.
#[derive(Clone, Copy, Debug, Default)]
struct EncRefDc {
    not_null_dc: bool,
    ref_frame: RefKind,
    dc_coeff: i16,
}

/// VP6F encoder.
///
/// Each [`Vp6Encoder::encode_keyframe`] call emits a complete elementary-
/// stream frame. Internal state (model, reference frames) is reset at
/// the start of every keyframe so the encoder is "stateless" across
/// keyframes until inter-frame support lands.
#[derive(Debug)]
pub struct Vp6Encoder {
    /// Quantiser parameter (0 = highest quality, 63 = coarsest).
    pub qp: u8,
    /// Use the `sub_version = 0` "simple profile" layout.
    pub sub_version: u8,
}

impl Default for Vp6Encoder {
    fn default() -> Self {
        Self {
            qp: 32,
            sub_version: 0,
        }
    }
}

impl Vp6Encoder {
    /// New encoder with a given QP (clamped to 0..=63).
    pub fn new(qp: u8) -> Self {
        Self {
            qp: qp.min(63),
            sub_version: 0,
        }
    }

    /// Encode a single keyframe from row-major Y/U/V planes.
    ///
    /// * `y_plane` — `width * height` bytes.
    /// * `u_plane` / `v_plane` — `(width/2) * (height/2)` bytes.
    /// * `width`, `height` must be multiples of 16.
    pub fn encode_keyframe(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
    ) -> Result<EncodedFrame> {
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::invalid(
                "VP6 encode: width/height must be non-zero multiples of 16",
            ));
        }
        let mb_width = width / 16;
        let mb_height = height / 16;
        if mb_width > 255 || mb_height > 255 {
            return Err(Error::invalid(
                "VP6 encode: simple-profile MB dims capped at 255",
            ));
        }
        if y_plane.len() < width * height {
            return Err(Error::invalid("VP6 encode: y plane too small"));
        }
        let uv_stride = width / 2;
        let uv_h = height / 2;
        if u_plane.len() < uv_stride * uv_h || v_plane.len() < uv_stride * uv_h {
            return Err(Error::invalid("VP6 encode: chroma plane too small"));
        }

        // --- Fixed header -------------------------------------------------
        let mut out = Vec::<u8>::with_capacity(32 + mb_width * mb_height * 6);
        out.push((self.qp << 1) & 0x7E);
        out.push(self.sub_version << 3);
        out.push(0);
        out.push(2);
        out.push(mb_height as u8);
        out.push(mb_width as u8);
        out.push(mb_height as u8);
        out.push(mb_width as u8);

        // --- Bool-coded body ---------------------------------------------
        let mut enc = RangeEncoder::new();
        // 2 skip bits.
        enc.put_bits(2, 0);
        // use_huffman = 0.
        enc.put_bit(0);

        // --- Coefficient-model update pass -------------------------------
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);

        // DC value-update probabilities: emit 0 for every node. On the
        // decoder side `def_prob` is carried forward and (since key=true)
        // copied through into `coeff_dccv[pt][node]`.
        let def_prob_template = [0x80u8; 11];
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
                model.coeff_dccv[pt][node] = def_prob_template[node];
            }
        }

        // Coefficient-reorder update flag = 0 (keep defaults).
        enc.put_bit(0);

        // Run-value update probabilities — emit 0 for every node.
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }

        // AC coefficient update probabilities — emit 0 for every node.
        // The decoder carries `def_prob` through this nested loop.
        let mut def_prob = def_prob_template;
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                        model.coeff_ract[pt][ct][cg][node] = def_prob[node];
                    }
                }
            }
        }
        let _ = &mut def_prob;

        // Recompute the `coeff_dcct` linear combination.
        for pt in 0..2 {
            for ctx in 0..3 {
                for node in 0..5 {
                    let v = ((model.coeff_dccv[pt][node] as i32
                        * tables::VP6_DCCV_LC[ctx][node][0]
                        + 128)
                        >> 8)
                        + tables::VP6_DCCV_LC[ctx][node][1];
                    model.coeff_dcct[pt][ctx][node] = v.clamp(1, 255) as u8;
                }
            }
        }

        // --- Per-macroblock coefficient emission -------------------------
        let dequant_dc = (tables::VP56_DC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;
        let dequant_ac = (tables::VP56_AC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;

        let mut left_block: [EncRefDc; 4] = [EncRefDc::default(); 4];
        let mut above_blocks: Vec<EncRefDc> = vec![EncRefDc::default(); 4 * mb_width + 6];
        if 2 * mb_width + 2 < above_blocks.len() {
            above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < above_blocks.len() {
            above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        let mut prev_dc = [[0i16; 3]; 3];
        prev_dc[1][0] = 128;
        prev_dc[2][0] = 128;

        for mb_row in 0..mb_height {
            for b in &mut left_block {
                *b = EncRefDc::default();
            }
            let mut above_block_idx: [usize; 6] = [0; 6];
            above_block_idx[0] = 1;
            above_block_idx[1] = 2;
            above_block_idx[2] = 1;
            above_block_idx[3] = 2;
            above_block_idx[4] = 2 * mb_width + 2 + 1;
            above_block_idx[5] = 3 * mb_width + 4 + 1;

            for mb_col in 0..mb_width {
                for b in 0..6usize {
                    let pt = if b > 3 { 1usize } else { 0usize };
                    let plane_idx = tables::B2P[b] as usize;
                    let ctx = left_block[tables::B6_TO_4[b] as usize].not_null_dc as usize
                        + above_blocks[above_block_idx[b]].not_null_dc as usize;

                    // Gather the 8x8 pixel tile for this block.
                    let mut tile = [0i32; 64];
                    sample_block_tile(
                        b, mb_row, mb_col, y_plane, u_plane, v_plane, width, uv_stride, &mut tile,
                    );

                    // Forward DCT into 64 raw coefficients (natural order
                    // — `coefs[u*8+v]` is the frequency coefficient at
                    // (u, v), subtracted by 128 + scaled to match the
                    // decoder's IDCT expectation).
                    let mut coefs = [0i32; 64];
                    forward_dct8x8(&tile, &mut coefs);

                    // DC quantisation: `new_dc * dequant_dc` gives the
                    // block[0] value that the IDCT consumes.
                    let new_dc = div_nearest(coefs[0], dequant_dc).clamp(-32768, 32767) as i16;

                    // Compute the decoder's predictor_dc for this block.
                    let lb = left_block[tables::B6_TO_4[b] as usize];
                    let ab = above_blocks[above_block_idx[b]];
                    let mut count = 0i32;
                    let mut pdc = 0i32;
                    if lb.ref_frame == RefKind::Current {
                        pdc += lb.dc_coeff as i32;
                        count += 1;
                    }
                    if ab.ref_frame == RefKind::Current {
                        pdc += ab.dc_coeff as i32;
                        count += 1;
                    }
                    let predictor = match count {
                        0 => prev_dc[plane_idx][0] as i32,
                        2 => pdc / 2,
                        _ => pdc,
                    };
                    let coded_dc = (new_dc as i32 - predictor).clamp(-32768, 32767) as i16;

                    // Quantise AC coefficients (in coeff_idx order) — we
                    // walk coeff_idx = 1..=63, look up `pos =
                    // coeff_index_to_pos[coeff_idx]`, then the natural
                    // block index is `permute[pos]`. Quantise with
                    // `dequant_ac` and store as signed levels.
                    let mut ac_levels = [0i32; 64];
                    for coeff_idx in 1..64usize {
                        let pos = model.coeff_index_to_pos[coeff_idx] as usize;
                        let perm = tables::IDCT_SCANTABLE[pos] as usize;
                        // Clamp to the range the decoder's value-tree
                        // can represent (up to `2^11 + bias`).
                        ac_levels[coeff_idx] =
                            div_nearest(coefs[perm], dequant_ac).clamp(-2047, 2047);
                    }

                    // Emit the coefficient stream for this block. The
                    // decoder iterates coeff_idx from 0 upward, branching
                    // on parse-or-skip at each step. We mirror that state
                    // machine exactly.
                    let last_nz = find_last_nonzero(&ac_levels);
                    emit_block_coefs(&mut enc, &model, pt, ctx, coded_dc, &ac_levels, last_nz);

                    // Update DC-prediction context for subsequent blocks.
                    let has_nonzero_dc = coded_dc != 0;
                    let lb_idx = tables::B6_TO_4[b] as usize;
                    let new_dc_final = (coded_dc as i32 + predictor) as i16;
                    left_block[lb_idx] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Current,
                        dc_coeff: new_dc_final,
                    };
                    above_blocks[above_block_idx[b]] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Current,
                        dc_coeff: new_dc_final,
                    };
                    prev_dc[plane_idx][0] = new_dc_final;
                }

                for y in 0..4 {
                    above_block_idx[y] += 2;
                }
                for uv in 4..6 {
                    above_block_idx[uv] += 1;
                }
            }
        }

        out.extend_from_slice(&enc.finish());
        Ok(out)
    }
}

/// Sample the 8x8 block pixel tile from the right plane at block `b` of
/// macroblock `(mb_row, mb_col)`. Luma blocks 0..=3 pull from the Y
/// plane; blocks 4/5 pull from U/V respectively.
fn sample_block_tile(
    b: usize,
    mb_row: usize,
    mb_col: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    y_stride: usize,
    uv_stride: usize,
    out: &mut [i32; 64],
) {
    let (plane, stride, x0, y0) = match b {
        0 => (y, y_stride, mb_col * 16, mb_row * 16),
        1 => (y, y_stride, mb_col * 16 + 8, mb_row * 16),
        2 => (y, y_stride, mb_col * 16, mb_row * 16 + 8),
        3 => (y, y_stride, mb_col * 16 + 8, mb_row * 16 + 8),
        4 => (u, uv_stride, mb_col * 8, mb_row * 8),
        5 => (v, uv_stride, mb_col * 8, mb_row * 8),
        _ => unreachable!(),
    };
    for r in 0..8usize {
        for c in 0..8usize {
            out[r * 8 + c] = plane[(y0 + r) * stride + (x0 + c)] as i32;
        }
    }
}

/// Forward 2D DCT-II with scaling matched to the decoder's IDCT.
///
/// Standard orthonormal DCT-II puts the DC at `8 * (V - 128)` for a flat
/// V-tile; our decoder's IDCT expects `~32 * (V - 128)` (the full
/// two-stage chain multiplies by `C4S4/2^16 * C4S4/2^20 ≈ 2^-32.05`,
/// which inverts to a gain of ~32). We therefore apply the usual 2D
/// DCT-II formula without the `1/4` factor so the per-block scale is
/// exactly 4x the orthonormal normalisation.
///
/// Output `out[u*8+v] = F[u, v]`: raster layout with `u` as vertical
/// frequency and `v` as horizontal frequency — the exact convention
/// the VP6 spec uses for its `default_dequant_table` and IDCT stages.
fn forward_dct8x8(tile: &[i32; 64], out: &mut [i32; 64]) {
    // Precomputed cos((2k+1)*n*pi/16) for k,n in 0..8.
    let mut cos = [[0.0f64; 8]; 8];
    for k in 0..8 {
        for n in 0..8 {
            let angle = ((2 * k + 1) * n) as f64 * std::f64::consts::PI / 16.0;
            cos[k][n] = angle.cos();
        }
    }

    // Subtract 128 from pixels.
    let mut pm = [0.0f64; 64];
    for i in 0..64 {
        pm[i] = (tile[i] - 128) as f64;
    }

    let inv_sqrt2 = 1.0f64 / std::f64::consts::SQRT_2;

    // Row-wise 1D DCT: temp[r][u] = sum_c pm[r][c] * cos((2c+1)*u*pi/16).
    let mut temp = [0.0f64; 64];
    for r in 0..8 {
        for u in 0..8 {
            let mut s = 0.0f64;
            for c in 0..8 {
                s += pm[r * 8 + c] * cos[c][u];
            }
            temp[r * 8 + u] = s;
        }
    }

    // Column-wise 1D DCT, stored in natural 2D-DCT-II raster layout
    // `out[u*8+v] = F[u][v]` — `u` is the vertical frequency index
    // (iterated in the decoder's column IDCT pass) and `v` is the
    // horizontal frequency index (iterated in the row pass). This
    // matches the VP6 spec (section 12.1 + section 16): scan index 1
    // maps via `default_dequant_table` to raw position 1 = F[0,1] (the
    // first horizontal AC), scan index 2 to raw position 8 = F[1,0]
    // (the first vertical AC), and so on.
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0.0f64;
            for r in 0..8 {
                s += temp[r * 8 + v] * cos[r][u];
            }
            let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
            let cv = if v == 0 { inv_sqrt2 } else { 1.0 };
            out[u * 8 + v] = (s * cu * cv).round() as i32;
        }
    }
}

/// Find the highest `coeff_idx` in 1..64 that holds a non-zero AC level.
/// Returns 0 if every AC is zero.
fn find_last_nonzero(levels: &[i32; 64]) -> usize {
    for idx in (1..64usize).rev() {
        if levels[idx] != 0 {
            return idx;
        }
    }
    0
}

/// Emit a single 8x8 block's worth of coefficients, mirroring the
/// decoder's [`crate::mb::parse_coeff`] state machine. `coded_dc` is the
/// predictor-adjusted DC level; `ac_levels[1..=63]` carry the quantised
/// AC levels in coeff_idx order. `last_nz` is the highest coeff_idx
/// carrying a non-zero AC (or 0 if all AC are zero).
fn emit_block_coefs(
    enc: &mut RangeEncoder,
    model: &Vp6Model,
    pt: usize,
    ctx: usize,
    coded_dc: i16,
    ac_levels: &[i32; 64],
    last_nz: usize,
) {
    // Mirrors decoder's `ct`, `coeff_idx`, model1/model2 selection.
    let mut ct: usize = 1;
    let mut coeff_idx: usize = 0;

    // Start on (model1=Dccv, model2=Dcct). After coeff_idx advances,
    // both switch to Ract(pt, ct, cg) in lock-step with the decoder.
    loop {
        let (m2, m1) = if coeff_idx == 0 {
            (
                &model.coeff_dcct[pt][ctx] as &[u8],
                &model.coeff_dccv[pt] as &[u8],
            )
        } else {
            let cg = tables::VP6_COEFF_GROUPS[coeff_idx] as usize;
            (
                &model.coeff_ract[pt][ct][cg] as &[u8],
                &model.coeff_ract[pt][ct][cg] as &[u8],
            )
        };

        // Decide: parse or skip. At coeff_idx == 0 we always take the
        // m2_0 decision bit to match the decoder (the `coeff_idx > 1
        // && ct == 0` shortcut only fires later).
        let value_at_idx: i32 = if coeff_idx == 0 {
            coded_dc as i32
        } else {
            ac_levels[coeff_idx]
        };
        let want_parse = value_at_idx != 0;

        // Decoder's go_parse: forced-true when `coeff_idx > 1 && ct == 0`.
        // In that case it does NOT read the m2_0 bit.
        let forced_parse = coeff_idx > 1 && ct == 0;
        if !forced_parse {
            enc.put_prob(m2[0], if want_parse { 1 } else { 0 });
        }

        if want_parse {
            let abs_val = value_at_idx.unsigned_abs() as i32;
            let sign = if value_at_idx < 0 { 1u8 } else { 0u8 };
            encode_coeff_value(enc, m2, m1, abs_val, &mut ct);
            enc.put_bit(sign);
            // coeff_idx advances by 1 after a parse.
            coeff_idx += 1;
            if coeff_idx >= 64 {
                break;
            }
            if coeff_idx > last_nz {
                // No more non-zero AC values. But the decoder always
                // reads either an end-of-block or a run AFTER a parse
                // that didn't break the loop — we need to cleanly close
                // the block. Continue the loop so the next iteration
                // emits the "no parse" + EOB path.
            }
        } else {
            // Skip path: run of zeros starting here. `ct := 0` on the
            // decoder.
            ct = 0;
            if coeff_idx > 0 {
                // Decide end-of-block vs run. If there are no more
                // non-zero AC values in this block we emit EOB (m2_1=0).
                let want_eob = coeff_idx > last_nz;
                enc.put_prob(m2[1], if want_eob { 0 } else { 1 });
                if want_eob {
                    break;
                }
                // Emit a run length: the smallest `run` such that
                // `coeff_idx + run` is the next non-zero slot.
                let mut next_nz = coeff_idx + 1;
                while next_nz <= last_nz && ac_levels[next_nz] == 0 {
                    next_nz += 1;
                }
                // If somehow we overshot, fall back to last_nz+1 (safe
                // because emit_run only cares about the numeric run).
                let run = (next_nz - coeff_idx) as i32;
                let model3_idx = if coeff_idx >= 6 { 1 } else { 0 };
                let runv = &model.coeff_runv[model3_idx];
                emit_run(enc, runv, run);
                coeff_idx += run as usize;
                if coeff_idx >= 64 {
                    break;
                }
            } else {
                // coeff_idx == 0 and we took the skip. Advance by 1 and
                // continue — decoder does `nidx = coeff_idx + run` with
                // `run = 1` carried over from init.
                coeff_idx += 1;
                if coeff_idx >= 64 {
                    break;
                }
            }
        }
    }
}

/// Emit a non-zero coefficient value (absolute value ≥ 1) using the
/// decoder's tree shape. On return, `ct` is set to the `ct` value the
/// decoder would adopt (1 for value==1, 2 for value>1) — callers need
/// this to pick the right Ract model slice for the next iteration.
fn encode_coeff_value(enc: &mut RangeEncoder, m2: &[u8], m1: &[u8], abs_val: i32, ct: &mut usize) {
    if abs_val == 1 {
        enc.put_prob(m2[2], 0);
        *ct = 1;
        return;
    }
    enc.put_prob(m2[2], 1);
    if abs_val == 2 {
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 0);
        *ct = 2;
        return;
    }
    if abs_val == 3 || abs_val == 4 {
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 1);
        enc.put_prob(m1[5], (abs_val - 3) as u8);
        *ct = 2;
        return;
    }
    // Long path: m2[3] == 1 → PC_TREE + bias-category bits.
    enc.put_prob(m2[3], 1);
    let mut cat = 0usize;
    for i in 0..=5 {
        let lo = tables::VP56_COEFF_BIAS[i + 5] as i32;
        let hi = if i + 6 < tables::VP56_COEFF_BIAS.len() {
            tables::VP56_COEFF_BIAS[i + 6] as i32
        } else {
            i32::MAX
        };
        if abs_val >= lo && abs_val < hi {
            cat = i;
            break;
        }
    }
    match cat {
        0 => {
            enc.put_prob(m1[6], 0);
            enc.put_prob(m1[7], 0);
        }
        1 => {
            enc.put_prob(m1[6], 0);
            enc.put_prob(m1[7], 1);
        }
        2 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 0);
            enc.put_prob(m1[9], 0);
        }
        3 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 0);
            enc.put_prob(m1[9], 1);
        }
        4 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 1);
            enc.put_prob(m1[10], 0);
        }
        5 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 1);
            enc.put_prob(m1[10], 1);
        }
        _ => unreachable!(),
    }
    let bias = tables::VP56_COEFF_BIAS[cat + 5] as i32;
    let delta = abs_val - bias;
    let bit_len = tables::VP56_COEFF_BIT_LENGTH[cat] as i32;
    for i in (0..=bit_len).rev() {
        let bit = ((delta >> i) & 1) as u8;
        enc.put_prob(tables::VP56_COEFF_PARSE_TABLE[cat][i as usize], bit);
    }
    *ct = 2;
}

/// Emit a run length `run ≥ 1` through the [`crate::tables::VP6_PCR_TREE`]
/// encoding the decoder reads. `probs` is the active `coeff_runv[i]`
/// slice (14 entries). Runs 1..=8 are encoded via tree leaves; runs ≥ 9
/// emit the escape leaf and then 6 raw prob-bits of `run - 9`.
fn emit_run(enc: &mut RangeEncoder, probs: &[u8; 14], run: i32) {
    let run = run.clamp(1, 9 + 63);
    let r = run as u32;
    match r {
        1 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 0);
            enc.put_prob(probs[2], 0);
        }
        2 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 0);
            enc.put_prob(probs[2], 1);
        }
        3 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 1);
            enc.put_prob(probs[3], 0);
        }
        4 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 1);
            enc.put_prob(probs[3], 1);
        }
        5 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 0);
            enc.put_prob(probs[6], 0);
        }
        6 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 0);
            enc.put_prob(probs[6], 1);
        }
        7 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 1);
            enc.put_prob(probs[7], 0);
        }
        8 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 1);
            enc.put_prob(probs[7], 1);
        }
        _ => {
            // Escape leaf: prob indices (0,1), (4,1), then 6 bits of
            // `run - 9` LSB-first under probs[8..=13].
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 1);
            let extra = (r as i32 - 9).clamp(0, 63);
            for i in 0..6 {
                let bit = ((extra >> i) & 1) as u8;
                enc.put_prob(probs[8 + i], bit);
            }
        }
    }
}

/// Signed integer nearest-to-zero division (rounds toward nearest, ties
/// away from zero). Mirrors what the decoder implicitly expects when it
/// multiplies `level * dequant` back up.
fn div_nearest(num: i32, denom: i32) -> i32 {
    if denom == 0 {
        return 0;
    }
    let n = num as i64;
    let d = denom as i64;
    let q = if (n ^ d) >= 0 {
        (n + d / 2) / d
    } else {
        (n - d / 2) / d
    };
    q as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_nearest_rounds_away_from_zero_on_ties() {
        assert_eq!(div_nearest(5, 2), 3);
        assert_eq!(div_nearest(-5, 2), -3);
        assert_eq!(div_nearest(4, 2), 2);
        assert_eq!(div_nearest(1, 2), 1);
        assert_eq!(div_nearest(-1, 2), -1);
    }

    #[test]
    fn construct_and_encode_minimal() {
        let w = 16usize;
        let h = 16usize;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let mut enc = Vp6Encoder::new(32);
        let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
        assert_eq!(bytes[0], 64);
        assert_eq!(bytes[1], 0);
        assert!(bytes.len() > 8);
    }

    #[test]
    fn forward_dct_dc_only_flat_tile() {
        // Flat tile at pixel=128 → DC = 0 (after subtracting 128).
        let tile = [128i32; 64];
        let mut out = [0i32; 64];
        forward_dct8x8(&tile, &mut out);
        assert_eq!(out[0], 0);
        for i in 1..64 {
            assert_eq!(out[i], 0, "AC at {i} should be zero for flat tile");
        }
    }

    #[test]
    fn forward_dct_flat_tile_dc_scales_correctly() {
        // Flat tile at pixel=255 → expected DC near 32 * 127 = 4064.
        let tile = [255i32; 64];
        let mut out = [0i32; 64];
        forward_dct8x8(&tile, &mut out);
        let expected = 32 * 127;
        assert!(
            (out[0] - expected).abs() <= 2,
            "DC should be ~{expected}, got {}",
            out[0]
        );
    }
}

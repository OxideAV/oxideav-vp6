//! VP6 intra-frame (I-frame) **encoder** — the top-level dual of
//! [`crate::intra_frame::decode_intra_frame`].
//!
//! Given a 4:2:0 [`Frame`] of source pixels, this module produces a VP6
//! keyframe bitstream that [`crate::intra_frame::decode_intra_frame`]
//! reconstructs back to pixels at a quantiser-bounded PSNR floor. The
//! encode pipeline is the exact inverse of the decode pipeline, stage
//! for stage:
//!
//! | decode (§)                  | encode (this module)                |
//! | --------------------------- | ----------------------------------- |
//! | §17.1 `+128` level shift    | `−128` level shift                  |
//! | §16 inverse DCT             | §16-dual forward DCT (`forward_dct`)|
//! | §15 dequant (`× factor`)    | quantise (`round(coeff / factor)`)  |
//! | §14 DC predict (`pred+Δ`)   | DC predict, emit `Δ = coded−pred`   |
//! | §13 token decode            | §13 token encode (`token_encode`)   |
//! | §9 header parse             | §9 header emit                      |
//!
//! ## Frame profile
//!
//! The encoder emits the **simplest valid I-frame shape**: Simple
//! coding profile, `MultiStream == 0` (single BoolCoder partition),
//! VP6.0 version, the §12.1 default zig-zag scan, keyframe-baseline
//! probabilities (no §13 per-frame probability updates — every
//! probability stays at its §13.2/§13.3 keyframe value of 128). This is
//! the mirror of the single-partition arrangement
//! [`crate::intra_frame::decode_intra_frame`] targets.
//!
//! The Simple profile forces the `Buff2Offset` raw-prefix field
//! (`MultiStream || Simple`); with a single partition the encoder writes
//! `Buff2Offset == 0` (the decoder's intra path ignores it — the whole
//! coefficient stream lives in partition 1 immediately after the
//! header tail). Simple profile carries no prediction-filter or
//! loop-filter tail fields, and VP6.0 (not VP6.2) carries no
//! `PredictionFilterAlpha`, so the tail is exactly the IntraHeader
//! geometry + scaling + the trailing `UseHuffman` flag.
//!
//! ## Provenance
//!
//! Derived solely from the decode pipeline this module inverts (all
//! sequenced from `docs/video/vp6/vp6_format.pdf` §9–§17) plus the
//! in-tree errata. No external library code was consulted.

use crate::bool_coder::BoolEncoder;
use crate::coeff_prob_update::CoeffProbBanks;
use crate::dc_pred::{DcPredictionContext, Neighbour, ReferenceBucket};
use crate::dequant::DequantContext;
use crate::forward_dct::fdct_block;
use crate::frame_assembly::{Frame, BLOCK_DIM};
use crate::reconstruct::INTRA_DC_LEVEL_SHIFT;
use crate::scan::DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG;
use crate::token_encode::encode_block_coefficients;
use crate::tokens::AcPlane;
use crate::Error;

/// Quantise one 8x8 raster-order DCT-coefficient block into scan-order
/// quantised coefficients (the §15 dequant inverse).
///
/// `raster` holds the forward-DCT output in raster order. Each
/// coefficient is divided by its §15 factor (DC factor for raster
/// position 0, AC factor otherwise) with round-to-nearest (ties away
/// from zero), then placed at its scan position via the
/// raster→zig-zag permutation so the result is in the scan order the
/// token encoder and decoder share.
///
/// The returned array's `[0]` entry is the quantised DC (the "coded DC"
/// the §14 prediction and §13.2 context operate on); `[1..=63]` are the
/// quantised AC coefficients in scan order.
fn quantise_to_scan(raster: &[i32; 64], dequant: DequantContext) -> [i32; 64] {
    let mut scan = [0i32; 64];
    for (raster_pos, &coeff) in raster.iter().enumerate() {
        let factor = if raster_pos == 0 {
            dequant.dc_factor
        } else {
            dequant.ac_factor
        } as i32;
        let q = div_round_nearest(coeff, factor);
        let scan_pos = DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[raster_pos] as usize;
        scan[scan_pos] = q;
    }
    scan
}

/// Divide `n` by positive `d`, rounding to the nearest integer with ties
/// rounded away from zero. The §15 dequant is an exact `× factor`, so the
/// matching quantiser is a `/ factor` whose rounding minimises the
/// reconstruction error.
#[inline]
fn div_round_nearest(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// Extract one 8x8 block of pixels from a plane at the given pixel
/// origin, applying the §17.1 `−128` level shift (the encoder-side
/// inverse of the decoder's `+128`). Out-of-plane samples (only possible
/// for an over-hanging block on an odd-fragment frame, which the driver
/// never extracts) would read as 0; the driver guarantees in-bounds
/// extraction.
fn extract_levelshifted_block(
    samples: &[u8],
    plane_width: usize,
    top: usize,
    left: usize,
) -> [i32; 64] {
    let mut out = [0i32; 64];
    for r in 0..BLOCK_DIM {
        let row_base = (top + r) * plane_width + left;
        for c in 0..BLOCK_DIM {
            out[r * BLOCK_DIM + c] = samples[row_base + c] as i32 - INTRA_DC_LEVEL_SHIFT;
        }
    }
    out
}

/// A per-plane record of each encoded block's coded DC, indexed by
/// `(block_row, block_col)` — the encoder-side mirror of
/// [`crate::intra_frame`]'s `PlaneDcGrid`. Stores the **coded**
/// (quantised, pre-dequant) DC so the §14 prediction sees exactly the
/// values the decoder's grid will hold.
struct PlaneDcGrid {
    cols: usize,
    cells: Vec<Option<i32>>,
}

impl PlaneDcGrid {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            cells: vec![None; cols.saturating_mul(rows)],
        }
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    fn above(&self, row: usize, col: usize) -> Option<i32> {
        if row == 0 {
            return None;
        }
        self.cells[self.idx(row - 1, col)]
    }

    #[inline]
    fn left(&self, row: usize, col: usize) -> Option<i32> {
        if col == 0 {
            return None;
        }
        self.cells[self.idx(row, col - 1)]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, coded_dc: i32) {
        let i = self.idx(row, col);
        self.cells[i] = Some(coded_dc);
    }
}

/// Tokenise one intra block: forward-DCT the level-shifted pixels,
/// quantise, compute the §14 DC prediction delta. Returns `(scan,
/// coded_dc)` — the scan-order token values (DC delta at position 0)
/// plus the coded DC for the caller's grid update. Emission (arithmetic
/// or Huffman) happens separately.
fn tokenize_intra_block(
    pixels: &[i32; 64],
    dequant: DequantContext,
    dc_pred: &mut DcPredictionContext,
    left_dc: Option<i32>,
    above_dc: Option<i32>,
) -> ([i32; 64], i32) {
    // §16-dual forward DCT (raster order in, raster order out).
    let mut raster = [0i32; 64];
    fdct_block(pixels, &mut raster);

    // §15-inverse quantise into scan order.
    let mut scan = quantise_to_scan(&raster, dequant);

    // §14 DC prediction: the decoder reconstructs `coded_dc = predictor
    // + dc_delta`, so the encoder emits `dc_delta = coded_dc −
    // predictor`. `coded_dc` is the quantised DC (scan position 0).
    let reference = ReferenceBucket::Intra;
    let left_n = left_dc.map(|dc| Neighbour { dc, reference });
    let above_n = above_dc.map(|dc| Neighbour { dc, reference });
    let coded_dc = scan[0];
    let predictor = dc_pred.predict(reference, left_n, above_n);
    let dc_delta = coded_dc - predictor;
    dc_pred.set_last_dc(reference, coded_dc);

    // The DC token carries the prediction *delta*; AC carries the
    // quantised coefficients directly.
    scan[0] = dc_delta;
    (scan, coded_dc)
}

/// One tokenized block, ready for emission by whichever §5/§6 coder the
/// frame shape selects: the Table 28 plane, the §13.2 Table 26 DC
/// context (arithmetic emission only — Huffman ignores it), and the
/// scan-order values with the §14 DC delta at position 0.
pub(crate) struct TokenizedBlock {
    pub plane: AcPlane,
    pub dc_context: crate::tokens::DcContext,
    pub scan: [i32; 64],
}

/// Walk the frame in decode order (§13 page 58: four luma raster
/// blocks, then U, then V per macroblock) and tokenize every block —
/// the coder-independent front half of the intra encoder. The §14 DC
/// prediction contexts and per-plane coded-DC grids are threaded
/// exactly as the decoder threads them, so the recorded
/// [`TokenizedBlock::dc_context`] matches what the decoder derives.
fn tokenize_intra_frame(frame: &Frame, dequant: DequantContext) -> Vec<TokenizedBlock> {
    let h_fragments = frame.h_fragments;
    let v_fragments = frame.v_fragments;
    let mb_cols = frame.mb_cols();
    let mb_rows = frame.mb_rows();

    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

    let y_w = frame.y.width();
    let u_w = frame.u.width();
    let v_w = frame.v.width();

    let mut blocks = Vec::with_capacity(mb_rows * mb_cols * 6);
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- four luma blocks ---
            for &(dr, dc) in LUMA_OFFSETS.iter() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let pixels = extract_levelshifted_block(frame.y.samples(), y_w, br * 8, bc_col * 8);
                let dc_context = crate::tokens::DcContext::from_neighbours(
                    y_grid.left(br, bc_col).is_some_and(|d| d != 0),
                    y_grid.above(br, bc_col).is_some_and(|d| d != 0),
                );
                let (scan, coded_dc) = tokenize_intra_block(
                    &pixels,
                    dequant,
                    &mut y_dc_pred,
                    y_grid.left(br, bc_col),
                    y_grid.above(br, bc_col),
                );
                y_grid.set(br, bc_col, coded_dc);
                blocks.push(TokenizedBlock {
                    plane: AcPlane::Y,
                    dc_context,
                    scan,
                });
            }

            // --- U chroma block ---
            let u_pixels =
                extract_levelshifted_block(frame.u.samples(), u_w, mb_row * 8, mb_col * 8);
            let u_dc_context = crate::tokens::DcContext::from_neighbours(
                u_grid.left(mb_row, mb_col).is_some_and(|d| d != 0),
                u_grid.above(mb_row, mb_col).is_some_and(|d| d != 0),
            );
            let (u_scan, u_coded_dc) = tokenize_intra_block(
                &u_pixels,
                dequant,
                &mut u_dc_pred,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            );
            u_grid.set(mb_row, mb_col, u_coded_dc);
            blocks.push(TokenizedBlock {
                plane: AcPlane::UV,
                dc_context: u_dc_context,
                scan: u_scan,
            });

            // --- V chroma block ---
            let v_pixels =
                extract_levelshifted_block(frame.v.samples(), v_w, mb_row * 8, mb_col * 8);
            let v_dc_context = crate::tokens::DcContext::from_neighbours(
                v_grid.left(mb_row, mb_col).is_some_and(|d| d != 0),
                v_grid.above(mb_row, mb_col).is_some_and(|d| d != 0),
            );
            let (v_scan, v_coded_dc) = tokenize_intra_block(
                &v_pixels,
                dequant,
                &mut v_dc_pred,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            );
            v_grid.set(mb_row, mb_col, v_coded_dc);
            blocks.push(TokenizedBlock {
                plane: AcPlane::UV,
                dc_context: v_dc_context,
                scan: v_scan,
            });
        }
    }
    blocks
}

/// Emit an already-tokenized block sequence as §13 arithmetic tokens
/// into `enc`, selecting each block's DC row from its recorded Table 26
/// context — the emission back half shared by the single-stream and
/// MultiStream-BoolCoder intra encoders.
fn emit_blocks_arithmetic(
    enc: &mut BoolEncoder,
    blocks: &[TokenizedBlock],
    probs: &crate::intra_frame::IntraProbs,
) {
    for tb in blocks {
        let dc_node_probs = tb
            .dc_context
            .select_row(&probs.dc_contexts[tb.plane.index()]);
        encode_block_coefficients(
            enc,
            tb.plane,
            dc_node_probs,
            &probs.ac_probs,
            &probs.zrl_probs,
            &tb.scan,
        );
    }
}

/// Encode a full intra (I-)frame from a source [`Frame`] into a VP6
/// keyframe bitstream that
/// [`crate::intra_frame::decode_intra_frame`] reconstructs.
///
/// * `frame` — the 4:2:0 source pixels. Its luma block grid is
///   `h_fragments × v_fragments` (= `frame.h_fragments ×
///   frame.v_fragments`); chroma is one block per macroblock.
/// * `dct_q_mask` — the §9 6-bit quantiser index (0 = coarsest, 63 =
///   finest). Selects the §15 quantiser this encoder inverts.
///
/// The returned `Vec<u8>` is a complete frame payload: the §9 raw-bit
/// header prefix (byte-aligned), followed by a single BoolCoder
/// partition carrying the §9 header tail and then every block's §13
/// coefficient tokens, walked in the same macroblock/block order the
/// decoder uses.
///
/// # Errors
///
/// [`Error::NotImplemented`] if the frame's fragment dimensions exceed
/// the 8-bit `HFragments`/`VFragments` header fields (the §9 geometry is
/// an `b(8)` per axis, so each is capped at 255).
pub fn encode_intra_frame(frame: &Frame, dct_q_mask: u8) -> Result<Vec<u8>, Error> {
    encode_intra_frame_with_banks(frame, dct_q_mask, &CoeffProbBanks::keyframe())
}

/// Encode a keyframe whose §8 Figure 5 sub-stream carries the §13
/// coefficient-probability updates needed to reach `banks` from the
/// keyframe baseline, and whose coefficient tokens are then BoolCoder-coded
/// against those same updated `banks`.
///
/// This is the general form of [`encode_intra_frame`] (which passes
/// [`CoeffProbBanks::keyframe`], i.e. no updates). It proves the Figure-5
/// **content** path end-to-end: a downstream decoder seeds the keyframe
/// baselines, applies the decoded updates to recover `banks`, and decodes
/// the coefficient stream against the identical probabilities the encoder
/// used — so the pixels round-trip exactly through
/// [`crate::decode_frame::Vp6Decoder::decode_packet`].
///
/// The custom-scan half of `banks` (`band_assignment`) is **not** applied
/// to the coefficient ordering here: node-probability updates change only
/// the BoolCoder probabilities, not which coefficients are coded, so the
/// default zig-zag scan is retained and only representable
/// ([`crate::coeff_prob_update::node_prob_update_representable`]) node
/// probabilities may differ from baseline. Pass a `banks` with the default
/// `band_assignment`.
///
/// # Errors
///
/// [`Error::NotImplemented`] for an over-large frame geometry (the §9
/// `b(8)` per-axis fragment cap).
pub fn encode_intra_frame_with_banks(
    frame: &Frame,
    dct_q_mask: u8,
    banks: &CoeffProbBanks,
) -> Result<Vec<u8>, Error> {
    let h_fragments = frame.h_fragments;
    let v_fragments = frame.v_fragments;
    // §9 Table 2 transmits the geometry in 16-px MACROBLOCK units
    // (fixture-arbitrated erratum — see `Vp6HeaderTail`), so the coded
    // frame must be MB-aligned and each axis caps at 255 macroblocks.
    if h_fragments % 2 != 0 || v_fragments % 2 != 0 {
        return Err(Error::NotImplemented);
    }
    let (mb_cols, mb_rows) = (h_fragments / 2, v_fragments / 2);
    if mb_cols > u8::MAX as usize || mb_rows > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let dequant = DequantContext::new(dct_q_mask);

    // The §13 banks the coefficient tokens are coded against. These match
    // exactly what the decoder reconstructs by applying the Figure-5
    // updates emitted below to its keyframe baselines.
    let intra_probs = banks.to_intra_probs();

    // --- §9 raw-bit header prefix (byte-aligned) ---
    // Table 1 byte: FrameType(1)=0, DctQMask(6), MultiStream(1)=0.
    // Table 2 byte: Vp3VersionNo(5)=VP6.0(6), VpProfile(2)=Simple(0),
    //               Reserved(1)=0.
    // Buff2Offset(16) — present because Simple profile. Single
    // partition → 0.
    let mut header = oxideav_core::bits::BitWriter::with_capacity(8);
    header.write_u32(0, 1); // FrameType = 0 (keyframe)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(0, 1); // MultiStream = 0
    header.write_u32(6, 5); // Vp3VersionNo = 6 (VP6.0)
    header.write_u32(0, 2); // VpProfile = 0 (Simple)
    header.write_u32(0, 1); // Reserved
    header.write_u32(0, 16); // Buff2Offset
    let raw_prefix = header.finish();

    // --- §9 BoolCoder-coded header tail + §13 coefficient tokens ---
    let mut enc = BoolEncoder::new();

    // IntraHeader tail (Table 2), in the decoder's parse order:
    //   VFragments b(8), HFragments b(8), OutputVFragments b(8),
    //   OutputHFragments b(8), ScalingMode b(2).
    // No loop/prediction-filter fields (Simple profile), no
    // PredictionFilterAlpha (VP6.0). Then UseHuffman b(1) = 0
    // (BoolCoder second-half, matching the single-partition arithmetic
    // coefficient stream).
    enc.encode_b(mb_rows as u32, 8);
    enc.encode_b(mb_cols as u32, 8);
    enc.encode_b(mb_rows as u32, 8); // OutputVFragments = coded
    enc.encode_b(mb_cols as u32, 8); // OutputHFragments = coded
    enc.encode_b(0, 2); // ScalingMode = MAINTAIN_ASPECT_RATIO
    enc.encode_b1(0); // UseHuffman = 0

    // --- §8 Figure 5 coefficient-probability-update sub-stream ---
    // On a keyframe the per-MB info begins (Figure 2) with the Figure 5
    // coefficient prob-update pass — there is no §10 mode or §11.2 MV
    // tree on an I-frame (§10: "For I-frames each MB is implicitly
    // encoded in intra-mode so no signaling of mode is" needed). Emit the
    // updates that carry the keyframe baseline to `banks` (an empty pass
    // when `banks == CoeffProbBanks::keyframe()`); the decoder seeds the
    // baselines and applies these updates to recover the identical bank.
    crate::coeff_prob_update::encode_coefficient_prob_updates_full(&mut enc, banks);

    // Tokenize the frame (coder-independent front half), then emit the
    // §13 arithmetic tokens into the same partition-1 coder
    // (single-stream: coefficients interleave nothing on an I-frame, so
    // this is simply "all blocks after the Figure-5 pass").
    let blocks = tokenize_intra_frame(frame, dequant);
    emit_blocks_arithmetic(&mut enc, &blocks, &intra_probs);

    // Assemble: raw prefix then the BoolCoder partition.
    let mut out = raw_prefix;
    out.extend_from_slice(&enc.finish());
    Ok(out)
}

/// Encode a keyframe as a **two-partition** (`MultiStream == 1`) packet
/// (§6): partition 1 carries the §9 header tail + the §8 Figure-5
/// (no-update) pass, partition 2 the §13 DCT tokens — BoolCoder-coded
/// when `use_huffman` is `false`, raw-bit Huffman-coded (§7.2 /
/// §13.2.2 / §13.3.2) when `true`. The raw prefix's `Buff2Offset`
/// points at partition 2 (byte offset from the start of the packet, §9
/// Table 2).
///
/// Same Simple/VP6.0 shape and keyframe-baseline banks as
/// [`encode_intra_frame`]; the decoded pixels are bit-identical to the
/// single-stream encoding at the same quantiser (the §5/§6 partition
/// arrangement changes only the entropy transport, never the decoded
/// coefficients).
///
/// # Errors
///
/// [`Error::NotImplemented`] for an over-large frame geometry (the §9
/// `b(8)` per-axis fragment cap) or a partition-1 length that overflows
/// the 16-bit `Buff2Offset` field.
pub fn encode_intra_frame_multistream(
    frame: &Frame,
    dct_q_mask: u8,
    use_huffman: bool,
) -> Result<Vec<u8>, Error> {
    let h_fragments = frame.h_fragments;
    let v_fragments = frame.v_fragments;
    // §9 geometry is in macroblock units (see `encode_intra_frame_with_banks`).
    if h_fragments % 2 != 0 || v_fragments % 2 != 0 {
        return Err(Error::NotImplemented);
    }
    let (mb_cols, mb_rows) = (h_fragments / 2, v_fragments / 2);
    if mb_cols > u8::MAX as usize || mb_rows > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let dequant = DequantContext::new(dct_q_mask);
    let banks = CoeffProbBanks::keyframe();
    let intra_probs = banks.to_intra_probs();

    // --- Partition 1: §9 tail + §8 Figure-5 (no-update) pass ---
    let mut p1 = BoolEncoder::new();
    p1.encode_b(mb_rows as u32, 8);
    p1.encode_b(mb_cols as u32, 8);
    p1.encode_b(mb_rows as u32, 8); // OutputVFragments = coded
    p1.encode_b(mb_cols as u32, 8); // OutputHFragments = coded
    p1.encode_b(0, 2); // ScalingMode = MAINTAIN_ASPECT_RATIO
    p1.encode_b1(u8::from(use_huffman)); // UseHuffman
    crate::coeff_prob_update::encode_coefficient_prob_updates(&mut p1);
    let p1_bytes = p1.finish();

    // --- Partition 2: the §13 tokens under the selected coder ---
    let blocks = tokenize_intra_frame(frame, dequant);
    let p2_bytes = if use_huffman {
        let tables = crate::huff_coeff::HuffmanCoeffTables::from_banks(&banks);
        let pairs: Vec<(AcPlane, [i32; 64])> =
            blocks.iter().map(|tb| (tb.plane, tb.scan)).collect();
        let mut w = oxideav_core::bits::BitWriter::new();
        crate::huff_coeff::encode_frame_blocks_huffman(&mut w, &tables, &pairs);
        w.finish()
    } else {
        let mut p2 = BoolEncoder::new();
        emit_blocks_arithmetic(&mut p2, &blocks, &intra_probs);
        p2.finish()
    };

    // --- §9 raw prefix with MultiStream = 1 + the real Buff2Offset ---
    // The raw prefix is 4 bytes (Table 1 byte, Table 2 byte, 16-bit
    // Buff2Offset); partition 2 starts right after partition 1.
    let buff2_offset = 4usize + p1_bytes.len();
    let buff2_offset = u16::try_from(buff2_offset).map_err(|_| Error::NotImplemented)?;
    let mut header = oxideav_core::bits::BitWriter::with_capacity(8);
    header.write_u32(0, 1); // FrameType = 0 (keyframe)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(1, 1); // MultiStream = 1
    header.write_u32(6, 5); // Vp3VersionNo = 6 (VP6.0)
    header.write_u32(0, 2); // VpProfile = 0 (Simple)
    header.write_u32(0, 1); // Reserved
    header.write_u32(buff2_offset as u32, 16); // Buff2Offset
    let mut out = header.finish();
    out.extend_from_slice(&p1_bytes);
    out.extend_from_slice(&p2_bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::frame_header::{Vp6FrameHeader, Vp6HeaderTail};
    use crate::intra_frame::decode_intra_frame;
    use crate::scan::DEFAULT_SCAN_ORDER;

    /// Build a source frame whose luma/chroma planes carry a
    /// deterministic test pattern.
    fn pattern_frame(hf: usize, vf: usize) -> Frame {
        let mut frame = Frame::new(hf, vf);
        let yw = frame.y.width();
        let yh = frame.y.height();
        for r in 0..yh {
            for c in 0..yw {
                let v = ((r * 3 + c * 5) % 256) as u8;
                frame.y.samples_mut()[r * yw + c] = v;
            }
        }
        let uw = frame.u.width();
        let uh = frame.u.height();
        for r in 0..uh {
            for c in 0..uw {
                frame.u.samples_mut()[r * uw + c] = (128 + ((r + c) % 40) as i32 - 20) as u8;
                frame.v.samples_mut()[r * uw + c] = (128 - ((r * 2 + c) % 40) as i32 + 20) as u8;
            }
        }
        frame
    }

    /// Decode the encoder's output back to a frame, exercising the full
    /// header-parse + intra-decode path.
    fn round_trip(frame: &Frame, q: u8) -> Frame {
        let bytes = encode_intra_frame(frame, q).expect("encode");

        // Parse the raw-bit header prefix.
        let hdr = Vp6FrameHeader::parse(&bytes).expect("header prefix");
        assert!(hdr.is_keyframe);
        assert_eq!(hdr.dct_q_mask, q & 0x3F);

        // Parse the BoolCoder header tail, then decode the frame from the
        // same partition.
        let tail_bytes = &bytes[hdr.raw_prefix_len..];
        let mut bc = BoolCoder::new(tail_bytes).expect("bool coder");
        let tail =
            Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
                .expect("tail");
        // §9 geometry travels in macroblock units (fixture-arbitrated
        // erratum) — half the 8x8 luma block-grid dimensions.
        assert_eq!(tail.h_fragments, Some((frame.h_fragments / 2) as u8));
        assert_eq!(tail.v_fragments, Some((frame.v_fragments / 2) as u8));
        assert!(!tail.use_huffman);

        // The encoder now emits the §8 Figure 5 coefficient-prob-update
        // sub-stream right after the header tail; consume it (a no-update
        // pass that leaves the keyframe baselines untouched) before
        // decoding the coefficient stream.
        let mut banks = crate::coeff_prob_update::CoeffProbBanks::keyframe();
        let scan = crate::coeff_prob_update::decode_coefficient_prob_updates(&mut bc, &mut banks)
            .expect("figure-5 prefix");
        assert_eq!(
            scan, DEFAULT_SCAN_ORDER,
            "no-update keyframe keeps default scan"
        );
        let probs = banks.to_intra_probs();
        decode_intra_frame(
            &mut bc,
            frame.h_fragments,
            frame.v_fragments,
            hdr.dct_q_mask,
            &probs,
            &scan,
        )
        .expect("decode")
    }

    /// Compute the mean-squared error and PSNR between two same-sized
    /// sample planes.
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sse = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x as f64 - y as f64;
            sse += d * d;
        }
        let mse = sse / a.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    /// A flat mid-grey frame encodes and decodes back to (near) mid-grey
    /// — every block is pure-DC, so the round-trip is essentially exact.
    #[test]
    fn flat_frame_round_trips_high_psnr() {
        let mut frame = Frame::new(4, 4);
        for s in frame.y.samples_mut() {
            *s = 128;
        }
        for s in frame.u.samples_mut() {
            *s = 128;
        }
        for s in frame.v.samples_mut() {
            *s = 128;
        }
        let out = round_trip(&frame, 32);
        assert_eq!(out.y.width(), 32);
        // Flat 128 in → 128 out (DC delta 0, empty AC).
        for &s in out.y.samples() {
            assert_eq!(s, 128, "flat frame must round-trip exactly");
        }
    }

    /// A patterned frame round-trips above a quantiser-dependent PSNR
    /// floor. At the fine quantiser (q=48) the reconstruction is close;
    /// the floor is pinned conservatively to catch any structural break
    /// in the encode pipeline (mis-ordered blocks, wrong DC prediction,
    /// scan-order bug) which would collapse PSNR.
    #[test]
    fn patterned_frame_round_trips_above_floor() {
        let frame = pattern_frame(4, 4);
        let out = round_trip(&frame, 48);
        let y_psnr = psnr(frame.y.samples(), out.y.samples());
        let u_psnr = psnr(frame.u.samples(), out.u.samples());
        let v_psnr = psnr(frame.v.samples(), out.v.samples());
        assert!(
            y_psnr >= 28.0,
            "luma PSNR {y_psnr:.2} dB below floor — encode pipeline regression"
        );
        assert!(u_psnr >= 30.0, "U PSNR {u_psnr:.2} dB below floor");
        assert!(v_psnr >= 30.0, "V PSNR {v_psnr:.2} dB below floor");
    }

    /// A single-macroblock frame (hf = vf = 2) round-trips — exercises
    /// the smallest grid plus all six blocks of one macroblock.
    #[test]
    fn single_mb_frame_round_trips() {
        let frame = pattern_frame(2, 2);
        let out = round_trip(&frame, 40);
        assert_eq!(out.y.width(), 16);
        let y_psnr = psnr(frame.y.samples(), out.y.samples());
        assert!(y_psnr >= 26.0, "single-MB luma PSNR {y_psnr:.2} dB too low");
    }

    /// Finer quantisers yield higher PSNR than coarser ones on the same
    /// source — a monotonicity sanity check on the quantiser wiring.
    #[test]
    fn finer_quantiser_improves_psnr() {
        let frame = pattern_frame(4, 4);
        let coarse = round_trip(&frame, 8);
        let fine = round_trip(&frame, 56);
        let coarse_psnr = psnr(frame.y.samples(), coarse.y.samples());
        let fine_psnr = psnr(frame.y.samples(), fine.y.samples());
        assert!(
            fine_psnr > coarse_psnr,
            "finer quantiser ({fine_psnr:.2}) should beat coarser ({coarse_psnr:.2})"
        );
    }
}

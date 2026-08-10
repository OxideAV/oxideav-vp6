//! VP6 intra-frame (I-frame) per-macroblock decode loop (spec §6 / §10
//! / §13 / §14 / §15 / §16 / §17.1 / §2).
//!
//! This module is the **driver** that fuses the crate's already-built
//! per-block primitives into a single full-frame decode. For an I-frame
//! every macroblock is coded `CODE_INTRA` (§8: an I-frame "can be
//! completely decoded without reference to any previously decoded
//! frame"; §10 Table 4: `CODE_INTRA` "uses no prediction"), so the
//! driver needs no mode-decode or motion-vector stage — it walks the
//! macroblock grid in raster order and, for each macroblock's six
//! blocks (four luma then the two chroma, §13 page 58 "blocks are
//! encoded in raster order within a macroblock"), runs the full
//! coefficient → DC-prediction → dequant → IDCT → §17.1 reconstruct →
//! §2 raster-assembly chain.
//!
//! The decode reads a **single BoolCoder partition** (§6: "either one
//! or two further partitions"). The single-partition arrangement —
//! `MultiStream == 0`, both mode/MV info and DCT tokens in partition 1 —
//! is the simplest valid bitstream and the one this driver targets;
//! the two-partition (`MultiStream == 1`) and Huffman second-partition
//! variants are sequenced by a later driver layer that splits the
//! coefficient reads onto partition 2.
//!
//! ## Block grids and the §14 neighbour question
//!
//! §14 DC prediction and §13.2's Table 26 DC node-probability context
//! both reference the **block immediately to the left of and
//! immediately above** the current block in the *frame block grid* —
//! not in decode order. Because macroblocks are decoded MB-interleaved
//! (MB0's four luma blocks occupy block-grid cells `(0,0) (0,1) (1,0)
//! (1,1)`; MB1's occupy `(0,2) (0,3) (1,2) (1,3)`), the plane-raster
//! [`crate::tokens::DcZeroContextTracker`] — which assumes pure raster
//! decode order — cannot supply these neighbours. The driver therefore
//! maintains an explicit per-plane [`PlaneDcGrid`] of each decoded
//! block's coded DC value and zero-ness, indexed by `(block_row,
//! block_col)`, and resolves the left/above neighbours by direct grid
//! lookup. For an all-intra frame every block carries
//! [`ReferenceBucket::Intra`], so the §14 same-reference filter is a
//! no-op and only the edge-availability rule matters.
//!
//! All material here is sequenced from `docs/video/vp6/vp6_format.pdf`
//! (On2 Technologies, document version 1.02, August 2006) and the
//! clean-room errata at `docs/video/vp6/vp6-errata-and-clarifications.md`.
//! No external library code was consulted.

use crate::block_decode::{AcProbBank, ZeroRunProbBank, BLOCK_SIZE};
use crate::coeff_source::CoeffSource;
use crate::dc_pred::{DcPredictionContext, Neighbour, ReferenceBucket};
use crate::dequant::DequantContext;
use crate::frame_assembly::Frame;
use crate::idct::idct_block;
use crate::reconstruct::intra_block_to_pixels;
use crate::tokens::{
    dc_probs_to_node_contexts, AcPlane, DcContext, NUM_DC_CONTEXTS, NUM_PLANES, NUM_TREE_NODES,
};
use crate::Error;

/// A per-plane record of each decoded block's coded DC coefficient and
/// its zero-ness, indexed by `(block_row, block_col)` in the plane's
/// block grid.
///
/// The driver consults this grid for the two block-grid-neighbour
/// lookups the spec defines: §14 DC prediction (the left/above coded DC
/// values) and §13.2 Table 26 (whether the left/above coded DC was
/// non-zero). It is a flat row-major `Vec`, sized to the plane's block
/// dimensions, with every cell initially absent (`None`).
#[derive(Debug, Clone)]
struct PlaneDcGrid {
    cols: usize,
    /// `Some(coded_dc)` once block `(r, c)` has been decoded, else
    /// `None`. The stored value is the **coded** (pre-dequant) DC
    /// coefficient — the §14/§13.2 domain (see module docs).
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

    /// The coded DC of the block immediately above `(row, col)`, or
    /// `None` at the top edge / if not yet decoded.
    #[inline]
    fn above(&self, row: usize, col: usize) -> Option<i32> {
        if row == 0 {
            return None;
        }
        self.cells[self.idx(row - 1, col)]
    }

    /// The coded DC of the block immediately to the left of `(row,
    /// col)`, or `None` at the left edge / if not yet decoded.
    #[inline]
    fn left(&self, row: usize, col: usize) -> Option<i32> {
        if col == 0 {
            return None;
        }
        self.cells[self.idx(row, col - 1)]
    }

    /// Record block `(row, col)`'s coded DC.
    #[inline]
    fn set(&mut self, row: usize, col: usize, coded_dc: i32) {
        let i = self.idx(row, col);
        self.cells[i] = Some(coded_dc);
    }
}

/// The persistent per-frame probability banks an I-frame decode needs.
///
/// All four are seeded at keyframe baseline (§13.3 / §13.2: "At each key
/// frame every probability value in this array … is set to 128", and
/// the §13 Table-20-node baselines) by [`IntraProbs::keyframe`]. A
/// caller that has applied the per-frame §13 probability updates
/// (`prob_update`) supplies the mutated banks directly via the struct
/// fields.
#[derive(Debug, Clone)]
pub struct IntraProbs {
    /// `DcNodeContexts[plane][context][node]` — the §13.2 DC node
    /// probability bank, already expanded from the 11-entry per-plane
    /// DC probs via [`dc_probs_to_node_contexts`].
    pub dc_contexts: [[[u8; NUM_TREE_NODES]; NUM_DC_CONTEXTS]; NUM_PLANES],
    /// `AcProbs[plane][prec][band][node]` — the §13.3 AC bank.
    pub ac_probs: AcProbBank,
    /// `ZeroRunProbs[band][node]` — the §13.3.3 zero-run bank.
    pub zrl_probs: ZeroRunProbBank,
}

impl IntraProbs {
    /// Build the keyframe-baseline banks: the §13.2 DC node contexts
    /// expanded from [`crate::tokens::baseline_dc_probs`], the §13.3 AC
    /// bank from [`crate::tokens::baseline_ac_probs`], and the §13.3.3
    /// zero-run bank from [`crate::zrl::ZERO_RUN_PROB_DEFAULTS`].
    pub fn keyframe() -> Self {
        Self {
            dc_contexts: dc_probs_to_node_contexts(&crate::tokens::baseline_dc_probs()),
            ac_probs: crate::tokens::baseline_ac_probs(),
            zrl_probs: crate::zrl::ZERO_RUN_PROB_DEFAULTS,
        }
    }
}

/// Decode one block: entropy-decode its coefficients (from whichever
/// §6 partition/coder `src` wraps), apply §14 DC prediction, §15
/// dequantize, §16 IDCT, §17.1 reconstruct, returning the 8x8 pixel
/// block plus the coded DC for the caller's grid update.
///
/// Returns `(pixels, coded_dc)` where `coded_dc` is the reconstructed
/// **coded** (pre-dequant) DC the §14/§13.2 neighbour grids store.
#[allow(clippy::too_many_arguments)]
fn decode_intra_block(
    src: &mut CoeffSource<'_, '_>,
    plane: AcPlane,
    probs: &IntraProbs,
    dequant: DequantContext,
    scan_to_raster: &[u8; BLOCK_SIZE],
    dc_pred: &mut DcPredictionContext,
    left_dc: Option<i32>,
    above_dc: Option<i32>,
) -> Result<([u8; 64], i32), Error> {
    // §13.2 Table 26 DC node-probability context from the left/above
    // coded-DC zero-ness. An absent neighbour counts as zero (§13.2).
    let dc_context = DcContext::from_neighbours(
        left_dc.is_some_and(|d| d != 0),
        above_dc.is_some_and(|d| d != 0),
    );

    // Entropy decode: §13.2 DC delta + §13.3 AC loop (arithmetic or
    // Huffman per the frame's §5/§6 shape). coeffs are in scan order;
    // coeffs[0] is the DC *prediction error* (delta).
    let block = src.decode_block(plane, dc_context, probs)?;

    // §14 DC prediction: reconstructed coded DC = predictor + delta.
    // All intra → every block's reference bucket is Intra.
    let reference = ReferenceBucket::Intra;
    let left_n = left_dc.map(|dc| Neighbour { dc, reference });
    let above_n = above_dc.map(|dc| Neighbour { dc, reference });
    let dc_delta = block.coeffs[0];
    let predictor = dc_pred.predict(reference, left_n, above_n);
    let coded_dc = predictor.wrapping_add(dc_delta);
    dc_pred.set_last_dc(reference, coded_dc);

    // Splice the reconstructed coded DC back into the scan-order block
    // (scan position 0 is always the DC, §12.1) before §15 dequant.
    let mut scan_coeffs = block.coeffs;
    scan_coeffs[0] = coded_dc;

    // §15 dequant + scan→raster in one pass, then §16 IDCT, then §17.1.
    let mut raster =
        crate::block_decode::dequantize_to_raster(&scan_coeffs, scan_to_raster, dequant);
    idct_block(&mut raster);
    let pixels = intra_block_to_pixels(&raster);

    Ok((pixels, coded_dc))
}

/// Decode a full intra (I-)frame from a single BoolCoder partition,
/// returning the reconstructed 4:2:0 [`Frame`].
///
/// * `bc` — a [`crate::bool_coder::BoolCoder`] positioned at the start
///   of the first data partition (immediately after the §9 header tail;
///   the caller seeks past the header's `raw_prefix_len` and constructs
///   the coder over the partition bytes).
/// * `h_fragments` / `v_fragments` — the §9 luma block-grid dimensions
///   from the header tail. The macroblock grid is `ceil(hf/2)` ×
///   `ceil(vf/2)` (§2 4:2:0).
/// * `dct_q_mask` — the §9 6-bit quantizer index; selects the §15
///   dequant factors.
/// * `probs` — the per-frame probability banks (keyframe baseline via
///   [`IntraProbs::keyframe`], or post-update banks).
/// * `scan_to_raster` — the active §12 scan-position→raster permutation
///   ([`crate::scan::DEFAULT_SCAN_ORDER`] for the default zig-zag, or a
///   §12.2 custom scan).
///
/// The decode walks the macroblock grid in raster order; within each
/// macroblock the four luma blocks (raster TL, TR, BL, BR) are decoded
/// first, then the U block, then the V block (§13 page 58). Each
/// block's coefficients, §14 DC prediction, §15 dequant, §16 IDCT and
/// §17.1 reconstruction are threaded through the per-plane DC grids and
/// the reconstructed pixels are placed into the frame via §2 raster
/// assembly.
///
/// # Errors
///
/// [`Error::Truncated`] if the partition runs out of bytes mid-block.
pub fn decode_intra_frame(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    h_fragments: usize,
    v_fragments: usize,
    dct_q_mask: u8,
    probs: &IntraProbs,
    scan_to_raster: &[u8; BLOCK_SIZE],
) -> Result<Frame, Error> {
    let mut src = CoeffSource::Bool(bc);
    decode_intra_frame_from_source(
        &mut src,
        h_fragments,
        v_fragments,
        dct_q_mask,
        probs,
        scan_to_raster,
    )
}

/// Decode a full intra (I-)frame reading its §13 DCT tokens from an
/// arbitrary [`CoeffSource`] — the §6-general form of
/// [`decode_intra_frame`].
///
/// The three §5/§6 coefficient arrangements all reduce to this walk:
/// single-stream (tokens in the partition-1 BoolCoder — what
/// [`decode_intra_frame`] wraps), MultiStream BoolCoder (tokens in a
/// second coder at `Buff2Offset`), and MultiStream Huffman (raw-bit
/// tokens in partition 2). An I-frame's partition 1 carries nothing
/// per-macroblock (no §10 modes, no §11 MVs), so the per-MB walk reads
/// exclusively from `src`.
///
/// Parameters otherwise as [`decode_intra_frame`].
pub fn decode_intra_frame_from_source(
    src: &mut CoeffSource<'_, '_>,
    h_fragments: usize,
    v_fragments: usize,
    dct_q_mask: u8,
    probs: &IntraProbs,
    scan_to_raster: &[u8; BLOCK_SIZE],
) -> Result<Frame, Error> {
    let mut frame = Frame::new(h_fragments, v_fragments);
    let mb_cols = frame.mb_cols();
    let mb_rows = frame.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    // Per-plane coded-DC neighbour grids. Luma is the full hf×vf block
    // grid; each chroma plane is mb_cols×mb_rows (one block per MB).
    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);

    // §14: one DC-prediction context per plane, re-seeded at frame
    // start. The "last decoded DC" seed is plane-local: 0 for luma,
    // +128 (quantized-DC domain) for each chroma plane — the
    // fixture-arbitrated #277 part 2 correction of §14's unqualified
    // "set to zero".
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new_chroma();
    let mut v_dc_pred = DcPredictionContext::new_chroma();

    // The four luma block-grid offsets within a macroblock, in §13
    // page-58 raster order (TL, TR, BL, BR).
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- four luma blocks ---
            let mut luma_pixels = [[0u8; 64]; 4];
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                // Edge-overhang luma blocks (odd hf/vf) are not coded.
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let left_dc = y_grid.left(br, bc_col);
                let above_dc = y_grid.above(br, bc_col);
                let (pixels, coded_dc) = decode_intra_block(
                    src,
                    AcPlane::Y,
                    probs,
                    dequant,
                    scan_to_raster,
                    &mut y_dc_pred,
                    left_dc,
                    above_dc,
                )?;
                y_grid.set(br, bc_col, coded_dc);
                luma_pixels[k] = pixels;
            }

            // --- U chroma block ---
            let (u_pixels, u_coded_dc) = decode_intra_block(
                src,
                AcPlane::UV,
                probs,
                dequant,
                scan_to_raster,
                &mut u_dc_pred,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            )?;
            u_grid.set(mb_row, mb_col, u_coded_dc);

            // --- V chroma block ---
            let (v_pixels, v_coded_dc) = decode_intra_block(
                src,
                AcPlane::UV,
                probs,
                dequant,
                scan_to_raster,
                &mut v_dc_pred,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            )?;
            v_grid.set(mb_row, mb_col, v_coded_dc);

            // §2 raster assembly: place the six reconstructed blocks.
            frame
                .place_macroblock(mb_row, mb_col, &luma_pixels, &u_pixels, &v_pixels)
                .map_err(|_| Error::Truncated)?;
        }
    }

    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::{BoolCoder, BoolEncoder};
    use crate::scan::DEFAULT_SCAN_ORDER;

    /// Encode a single block that is *all zero* under the keyframe
    /// baselines: a DC ZERO leaf then an immediate AC EOB. We don't
    /// hand-encode token by token; instead we exercise the driver on an
    /// all-zero byte stream, which under the baselines decodes every
    /// block as empty (DC = 0, EOB at the first AC position — proven in
    /// `block_decode`'s `all_zero_stream_decodes_empty_block`). An empty
    /// block dequantizes/IDCTs/reconstructs to the §17.1 mid-grey 128
    /// in luma; chroma rides the §14 +128 quantized-DC seed (errata
    /// #277 part 2) and saturates instead.
    #[test]
    fn all_zero_stream_decodes_flat_planes() {
        // A 2x2-MB frame: hf = vf = 4 luma blocks.
        let bytes = [0u8; 256];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = IntraProbs::keyframe();
        let frame =
            decode_intra_frame(&mut bc, 4, 4, 32, &probs, &DEFAULT_SCAN_ORDER).expect("decode");

        // Every luma sample is 128 (mid-grey): an empty intra block is
        // §17.1's `0 + 128`.
        assert_eq!(frame.y.width(), 32);
        assert_eq!(frame.y.height(), 32);
        for &s in frame.y.samples() {
            assert_eq!(s, 128, "empty intra luma block → mid-grey 128");
        }
        // Chroma: the §14 frame-start seed is +128 in the quantized-DC
        // domain (errata #277 part 2), so an all-zero stream (DC delta
        // 0) reconstructs each chroma block at
        // `clip(idct(128 * dc_factor) + 128)` — saturated white, not
        // 128. Neutral chroma on the wire requires a coded DC delta of
        // −128 (which the in-tree encoder now emits).
        for &s in frame.u.samples() {
            assert_eq!(s, 255, "chroma seed 128 saturates on a zero delta");
        }
        for &s in frame.v.samples() {
            assert_eq!(s, 255);
        }
    }

    /// A 1x1-MB frame (hf = vf = 2) decodes its six blocks from the
    /// all-zero stream and produces a 16x16 luma / 8x8 chroma mid-grey
    /// frame without running out of bytes.
    #[test]
    fn single_mb_frame_decodes() {
        let bytes = [0u8; 64];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = IntraProbs::keyframe();
        let frame =
            decode_intra_frame(&mut bc, 2, 2, 0, &probs, &DEFAULT_SCAN_ORDER).expect("decode");
        assert_eq!(frame.y.width(), 16);
        assert_eq!(frame.y.height(), 16);
        assert_eq!(frame.u.width(), 8);
        assert_eq!(frame.v.height(), 8);
        assert!(frame.y.samples().iter().all(|&s| s == 128));
    }

    /// Odd luma-fragment dimensions: a frame with hf = vf = 3 has a 2x2
    /// MB grid but only a 3x3 luma block grid, so the bottom-right
    /// macroblock has its BR (and the right column / bottom row of
    /// neighbours) luma blocks overhanging the grid. The driver must
    /// skip the overhang luma blocks and still decode the in-grid ones
    /// plus all chroma. We assert it completes and the in-grid luma
    /// region is mid-grey.
    #[test]
    fn odd_fragment_dims_skip_overhang_luma() {
        // 2x2 MB grid (ceil(3/2)=2). MB count = 4. Each MB has up to 4
        // luma + 2 chroma blocks; overhang luma blocks are skipped.
        let bytes = [0u8; 256];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = IntraProbs::keyframe();
        let frame =
            decode_intra_frame(&mut bc, 3, 3, 16, &probs, &DEFAULT_SCAN_ORDER).expect("decode");
        assert_eq!(frame.y.width(), 24);
        assert_eq!(frame.y.height(), 24);
        // The 3x3 luma block grid is fully in-bounds; all 9 blocks were
        // decoded (the overhang positions of the bottom/right MBs simply
        // don't exist), so the whole plane is mid-grey.
        for &s in frame.y.samples() {
            assert_eq!(s, 128);
        }
    }

    /// The DC-prediction context is threaded across blocks: a block
    /// whose left and above neighbours both carry a non-zero coded DC
    /// selects the §13.2 `BothNonZero` node context. We can't easily
    /// assert the internal context choice through the public decode, but
    /// we *can* confirm the `PlaneDcGrid` neighbour lookups behave as the
    /// spec's edge rules require.
    #[test]
    fn plane_dc_grid_edge_neighbours() {
        let mut g = PlaneDcGrid::new(3, 3);
        // Top-left block: no left, no above.
        assert_eq!(g.left(0, 0), None);
        assert_eq!(g.above(0, 0), None);
        g.set(0, 0, 5);
        // Block (0,1): left is (0,0)=5, no above.
        assert_eq!(g.left(0, 1), Some(5));
        assert_eq!(g.above(0, 1), None);
        g.set(0, 1, 7);
        // Block (1,0): no left, above is (0,0)=5.
        assert_eq!(g.left(1, 0), None);
        assert_eq!(g.above(1, 0), Some(5));
        g.set(1, 0, 9);
        // Block (1,1): left (1,0)=9, above (0,1)=7.
        assert_eq!(g.left(1, 1), Some(9));
        assert_eq!(g.above(1, 1), Some(7));
    }

    /// Loop-wiring smoke test against a BoolEncoder-built stream. Each
    /// block under the keyframe baselines reads a DC token then AC
    /// tokens; an all-zero-bit encoded stream drives every per-block
    /// decision to the 0-branch (DC ZERO leaf → DC 0, immediate AC EOB),
    /// so the whole frame decodes empty and reconstructs to §17.1
    /// mid-grey. This confirms the BoolEncoder→BoolCoder→driver path is
    /// byte-consistent end to end (the per-block arithmetic itself is
    /// pinned by block_decode / idct / reconstruct unit tests).
    #[test]
    fn encoded_stream_round_trips_through_driver() {
        let mut enc = BoolEncoder::new();
        // All-zero bits at the per-bit probability the DC/AC trees use
        // keep every decision on the 0-branch.
        for _ in 0..1024u32 {
            enc.encode_bool(0, 128);
        }
        let bytes = enc.finish();
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = IntraProbs::keyframe();
        let frame =
            decode_intra_frame(&mut bc, 4, 4, 48, &probs, &DEFAULT_SCAN_ORDER).expect("decode");
        assert_eq!(frame.y.width(), 32);
        assert_eq!(frame.y.height(), 32);
        // Sanity: the frame is fully populated (no panic, correct dims).
        assert_eq!(frame.y.samples().len(), 32 * 32);
        assert_eq!(frame.u.samples().len(), 16 * 16);
        assert!(frame.y.samples().iter().all(|&s| s == 128));
    }
}

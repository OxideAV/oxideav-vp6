//! VP6 inter-frame (P-frame) **encoder** — the top-level dual of
//! [`crate::inter_frame::decode_inter_frame`] for the
//! `CODE_INTER_NO_MV` macroblock shape.
//!
//! Given a 4:2:0 source [`Frame`] and the previous-frame reconstruction,
//! this module produces the BoolCoder data partition that
//! [`crate::inter_frame::decode_inter_frame`] reconstructs back to pixels
//! at a quantiser-bounded PSNR floor. It is the inter-frame analogue of
//! [`crate::intra_encode::encode_intra_frame`].
//!
//! ## Frame shape
//!
//! The encoder emits the **simplest valid P-frame**: every macroblock is
//! `CODE_INTER_NO_MV` (§10) — zero motion vector, predicted from the
//! previous-frame reconstruction. This is the inter analogue of the
//! intra encoder's "every MB intra" shape: it exercises the full §10
//! mode walk, the §13 coefficient encode, the §14 DC prediction and the
//! §17.2 zero-MV inter recombination, without needing motion estimation
//! (a `MvSource::Zero` mode reads/writes no MV bits, §11) or the FourMV
//! path. A richer P-frame (with real motion search choosing among the
//! ten modes) is a strict superset that reuses this same per-block
//! residual-encode core.
//!
//! ## Per-block residual
//!
//! For a `CODE_INTER_NO_MV` block the decoder reconstructs
//! `pixels = clip(prediction + idct(dequant(coeffs)))`, where the
//! `prediction` is the co-located reference block (zero MV → whole-sample
//! identity, §11.4). The encoder therefore forms the **residual**
//! `source − prediction`, forward-DCTs it (§16-dual), quantises (§15
//! inverse), DC-predicts (§14) and emits the §13 token stream — exactly
//! the intra encode core but on the inter residual rather than the
//! level-shifted source. The `prediction` is produced by the *same*
//! [`crate::inter::predict_inter_block_subpel`] call the decoder uses, so
//! the two predictions are bit-identical by construction.
//!
//! ## Probabilities and references
//!
//! The encoder threads the same per-frame probability banks
//! ([`crate::inter_frame::InterProbs`]) and §11.3/§11.4 filter
//! configuration ([`crate::inter_frame::FilterConfig`]) the decoder
//! consumes, and predicts against the same
//! [`crate::inter_frame::BorderedRef`] previous-frame buffer — so a
//! caller that hands the encoder's partition plus the *identical*
//! probs/filter/refs to `decode_inter_frame` recovers the frame.
//!
//! ## Provenance
//!
//! Derived solely from the decode pipeline this module inverts
//! ([`crate::inter_frame`], sequenced from `docs/video/vp6/vp6_format.pdf`
//! §9–§17) plus the in-tree errata. No external library code was
//! consulted.

use crate::bool_coder::BoolEncoder;
use crate::dc_pred::{DcPredictionContext, Neighbour, ReferenceBucket};
use crate::dequant::DequantContext;
use crate::forward_dct::fdct_block;
use crate::frame_assembly::{Frame, BLOCK_DIM};
use crate::inter::{predict_inter_block_subpel, MvShift};
use crate::inter_frame::{BorderedRef, FilterConfig, InterProbs};
use crate::mode_encode::encode_mode_from_probs;
use crate::modes::{CodingMode, ModeAvailability};
use crate::mv_diff::select_diff_reference_mv_from_grid;
use crate::mv_encode::{encode_mv_pair, MAX_MV_MAGNITUDE};
use crate::near_mv::{resolve_near_mvs, MotionVector, NearMvResolution, NeighbourMv};
use crate::scan::DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG;
use crate::token_encode::encode_block_coefficients;
use crate::tokens::{AcPlane, DcContext};
use crate::Error;

/// Quantise one 8x8 raster-order residual-DCT block into scan-order
/// quantised coefficients — identical to the intra encoder's quantiser
/// (the §15 dequant inverse), but applied to the inter residual's
/// forward-DCT rather than the level-shifted source's.
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

/// Divide `n` by positive `d`, rounding to nearest (ties away from zero).
#[inline]
fn div_round_nearest(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// Per-plane coded-DC + reference grid mirroring the decoder's
/// `PlaneDcGrid`. Stores the coded (quantised, pre-dequant) DC and the
/// block's reference bucket so the §14 same-reference prediction and the
/// §13.2 Table-26 context see exactly the values the decoder's grid will.
struct PlaneDcGrid {
    cols: usize,
    cells: Vec<Option<(i32, ReferenceBucket)>>,
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
    fn above(&self, row: usize, col: usize) -> Option<(i32, ReferenceBucket)> {
        if row == 0 {
            return None;
        }
        self.cells[self.idx(row - 1, col)]
    }

    #[inline]
    fn left(&self, row: usize, col: usize) -> Option<(i32, ReferenceBucket)> {
        if col == 0 {
            return None;
        }
        self.cells[self.idx(row, col - 1)]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, coded_dc: i32, reference: ReferenceBucket) {
        let i = self.idx(row, col);
        self.cells[i] = Some((coded_dc, reference));
    }
}

/// Extract one 8x8 block of pixels at the given pixel origin as `i32`.
fn extract_block(samples: &[u8], plane_width: usize, top: usize, left: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    for r in 0..BLOCK_DIM {
        let row_base = (top + r) * plane_width + left;
        for c in 0..BLOCK_DIM {
            out[r * BLOCK_DIM + c] = samples[row_base + c] as i32;
        }
    }
    out
}

/// Encode one `CODE_INTER_NO_MV` block: form the residual against the
/// zero-MV prediction, forward-DCT, quantise, compute the §14 DC delta,
/// and emit the §13 token stream. Returns the coded DC for the grid.
#[allow(clippy::too_many_arguments)]
fn encode_inter_block(
    enc: &mut BoolEncoder,
    plane: AcPlane,
    source: &[i32; 64],
    prediction: &[u8; 64],
    dequant: DequantContext,
    dc_node_probs: &[u8; crate::tokens::NUM_TREE_NODES],
    probs: &InterProbs,
    dc_pred: &mut DcPredictionContext,
    reference: ReferenceBucket,
    left: Option<(i32, ReferenceBucket)>,
    above: Option<(i32, ReferenceBucket)>,
) -> i32 {
    // Inter residual = source − prediction (the §17.2 recombination
    // inverse: the decoder forms `clip(prediction + residual)`).
    let mut residual = [0i32; 64];
    for ((r, &s), &p) in residual
        .iter_mut()
        .zip(source.iter())
        .zip(prediction.iter())
    {
        *r = s - p as i32;
    }

    // §16-dual forward DCT, then §15-inverse quantise into scan order.
    let mut raster = [0i32; 64];
    fdct_block(&residual, &mut raster);
    let mut scan = quantise_to_scan(&raster, dequant);

    // §14 DC prediction against the block's actual reference bucket
    // (`InterLast` for previous-frame modes, `InterGolden` for the §10
    // Golden-Frame modes). The same-reference rule filters neighbours on
    // this bucket — the decoder records the identical bucket per block.
    let left_n = left.map(|(dc, r)| Neighbour { dc, reference: r });
    let above_n = above.map(|(dc, r)| Neighbour { dc, reference: r });
    let coded_dc = scan[0];
    let predictor = dc_pred.predict(reference, left_n, above_n);
    let dc_delta = coded_dc - predictor;
    dc_pred.set_last_dc(reference, coded_dc);

    scan[0] = dc_delta;
    encode_block_coefficients(
        enc,
        plane,
        dc_node_probs,
        &probs.coeffs.ac_probs,
        &probs.coeffs.zrl_probs,
        &scan,
    );

    coded_dc
}

/// Compute the luma prediction for a block at the given macroblock +
/// corner against the previous-frame `BorderedRef` with motion vector
/// `mv` — the exact prediction the decoder forms via
/// [`predict_inter_block_subpel`] at [`MvShift::Luma`]. A zero `mv` is the
/// `CODE_INTER_NO_MV` whole-sample identity copy.
fn predict_mv_luma(
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    corner: (i32, i32),
    mv: MotionVector,
    filter: &FilterConfig,
) -> [u8; 64] {
    let (buf, stride, origin) = prev.y_plane();
    let top = (mb_row * 16) as i32 + corner.0;
    let left = (mb_col * 16) as i32 + corner.1;
    let base = (origin as i32 + top * stride as i32 + left) as usize;
    let mut pred = [0u8; 64];
    predict_inter_block_subpel(
        buf,
        stride,
        base,
        mv.x as i32,
        mv.y as i32,
        MvShift::Luma,
        filter.policy,
        filter.loop_filter_qi,
        &mut pred,
    );
    pred
}

/// Compute the chroma prediction for a block with motion vector `mv` — the
/// exact prediction the decoder forms at [`MvShift::Chroma`] (⅛-pel). For
/// a single-vector MB the chroma MV is the MB MV itself (§11.4).
fn predict_mv_chroma(
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    is_v: bool,
    mv: MotionVector,
    filter: &FilterConfig,
) -> [u8; 64] {
    let (buf, stride, origin) = if is_v { prev.v_plane() } else { prev.u_plane() };
    let top = (mb_row * 8) as i32;
    let left = (mb_col * 8) as i32;
    let base = (origin as i32 + top * stride as i32 + left) as usize;
    let mut pred = [0u8; 64];
    predict_inter_block_subpel(
        buf,
        stride,
        base,
        mv.x as i32,
        mv.y as i32,
        MvShift::Chroma,
        filter.policy,
        filter.loop_filter_qi,
        &mut pred,
    );
    pred
}

/// Encode a full inter (P-)frame of the `CODE_INTER_NO_MV` shape into the
/// BoolCoder data partition [`crate::inter_frame::decode_inter_frame`]
/// consumes.
///
/// * `source` — the 4:2:0 source pixels for this P-frame.
/// * `prev` — the §11.5-bordered previous-frame reconstruction (the
///   reference every `CODE_INTER_NO_MV` MB predicts from).
/// * `dct_q_mask` — the §9 6-bit quantiser index (§15 dequant).
/// * `probs` — the per-frame mode/MV/coefficient banks; the encoder
///   reads `mode_probs` for the §10 mode emit and `coeffs` for the §13
///   token emit. The caller must hand the **same** banks to the decoder.
/// * `filter` — the §11.3/§11.4 filter configuration; threaded into the
///   zero-MV prediction so the encoder and decoder predictions match.
///
/// The returned `Vec<u8>` is the finished BoolCoder partition: feed it to
/// [`crate::bool_coder::BoolCoder::new`] and then `decode_inter_frame`
/// with the same `probs`/`filter`/`prev` (and any `golden`) to recover
/// the frame.
///
/// # Errors
///
/// [`Error::NotImplemented`] if the source's fragment dimensions exceed
/// the 8-bit header geometry fields.
pub fn encode_inter_frame(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    // Bare data partition: no header-tail prelude on the coder.
    encode_inter_frame_body(source, prev, dct_q_mask, probs, filter, |_| {})
}

/// Shared P-frame body: open a [`BoolEncoder`], run `prelude` against it
/// (used by [`encode_inter_frame_packet`] to emit the §9 InterHeader tail
/// bits in the same arithmetic stream), then encode every macroblock's
/// §10 mode + §13 coefficients. Returns the finished partition.
fn encode_inter_frame_body(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
    prelude: impl FnOnce(&mut BoolEncoder),
) -> Result<Vec<u8>, Error> {
    let h_fragments = source.h_fragments;
    let v_fragments = source.v_fragments;
    if h_fragments > u8::MAX as usize || v_fragments > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let mb_cols = source.mb_cols();
    let mb_rows = source.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    let mut enc = BoolEncoder::new();
    prelude(&mut enc);

    // Per-plane coded-DC grids + §14 prediction contexts, mirroring the
    // decoder exactly.
    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    const LUMA_CORNERS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];

    let y_w = source.y.width();
    let u_w = source.u.width();
    let v_w = source.v.width();

    // §10 mode-walk state: every MB is CODE_INTER_NO_MV. The decoder
    // resolves availability via `resolve_near_mvs`, which **skips zero
    // motion vectors** when building the Nearest/Near list (a zero MV is
    // not a usable predictor). Every MB in an all-`CODE_INTER_NO_MV`
    // frame carries a zero MV, so no neighbour ever supplies a Nearest or
    // Near vector: the decoder computes `ModeAvailability::Neither` for
    // **every** MB. The encoder therefore emits every mode against the
    // `Neither` probXmitted row, exactly the row the decoder selects.
    let availability = ModeAvailability::Neither;

    let mut last_mode = CodingMode::InterNoMv;
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- §10 mode emit (CODE_INTER_NO_MV) ---
            encode_mode_from_probs(
                &mut enc,
                CodingMode::InterNoMv,
                &probs.mode_probs,
                availability,
                last_mode,
            );
            last_mode = CodingMode::InterNoMv;
            // CODE_INTER_NO_MV is MvSource::Zero: no MV bits emitted.

            // --- four luma blocks ---
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let source_pixels = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
                let prediction = predict_mv_luma(
                    prev,
                    mb_row,
                    mb_col,
                    LUMA_CORNERS[k],
                    MotionVector::ZERO,
                    filter,
                );
                let dc_node_probs = DcContext::from_neighbours(
                    y_grid.left(br, bc_col).is_some_and(|(d, _)| d != 0),
                    y_grid.above(br, bc_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::Y.index()]);
                let coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::Y,
                    &source_pixels,
                    &prediction,
                    dequant,
                    dc_node_probs,
                    probs,
                    &mut y_dc_pred,
                    ReferenceBucket::InterLast,
                    y_grid.left(br, bc_col),
                    y_grid.above(br, bc_col),
                );
                y_grid.set(br, bc_col, coded_dc, ReferenceBucket::InterLast);
            }

            // --- U chroma block ---
            let u_source = extract_block(source.u.samples(), u_w, mb_row * 8, mb_col * 8);
            let u_pred = predict_mv_chroma(prev, mb_row, mb_col, false, MotionVector::ZERO, filter);
            let u_dc_node_probs = DcContext::from_neighbours(
                u_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                u_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let u_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &u_source,
                &u_pred,
                dequant,
                u_dc_node_probs,
                probs,
                &mut u_dc_pred,
                ReferenceBucket::InterLast,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            );
            u_grid.set(mb_row, mb_col, u_coded_dc, ReferenceBucket::InterLast);

            // --- V chroma block ---
            let v_source = extract_block(source.v.samples(), v_w, mb_row * 8, mb_col * 8);
            let v_pred = predict_mv_chroma(prev, mb_row, mb_col, true, MotionVector::ZERO, filter);
            let v_dc_node_probs = DcContext::from_neighbours(
                v_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                v_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let v_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &v_source,
                &v_pred,
                dequant,
                v_dc_node_probs,
                probs,
                &mut v_dc_pred,
                ReferenceBucket::InterLast,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            );
            v_grid.set(mb_row, mb_col, v_coded_dc, ReferenceBucket::InterLast);
        }
    }

    Ok(enc.finish())
}

// ===========================================================================
// Motion-estimated P-frame encoder
// ===========================================================================
//
// The zero-MV body above emits every MB as `CODE_INTER_NO_MV`. This section
// adds a real per-MB motion search that chooses between `CODE_INTER_NO_MV`
// (zero vector) and `CODE_INTER_PLUS_MV` (a §11.1-coded vector), emitting the
// chosen mode + (for a New-MV MB) the §11.1 delta against the §11
// differential reference, and the residual against the *chosen* prediction.
//
// Correctness invariant: the decoder reconstructs each MB by (a) resolving
// §10 availability from the MV grid built so far, (b) decoding the mode, (c)
// reconstructing the MV (zero for NoMv; differential-reference + delta for
// PlusMv), (d) motion-compensating with `predict_inter_block_subpel`, and (e)
// adding the IDCT residual. The encoder threads the *identical* `mv_grid`,
// `last_mode` and availability state and forms the residual against the
// *same* `predict_inter_block_subpel` prediction, so the bytes it emits feed
// straight back through `decode_inter_frame` to the same reconstruction the
// encoder predicted from (modulo the quantiser floor).

/// The half-extent (in whole luma samples) of the diamond/box motion search
/// around the zero/predicted vector. Kept small so the search stays cheap and
/// the candidate set bounded (the §11.5 border is 48 samples, comfortably
/// covering this range plus the sub-pixel filter reach).
const ME_SEARCH_RANGE: i32 = 8;

/// Sum of absolute differences between a source 16×16 luma region and its
/// motion-compensated prediction for a candidate `mv`, summed over the four
/// luma blocks. Lower is a better predictor. The prediction is formed with
/// the *same* [`predict_inter_block_subpel`] the decoder uses, so the SAD is
/// computed on the exact pixels the decoder will reconstruct from.
#[allow(clippy::too_many_arguments)]
fn luma_mb_sad(
    source: &Frame,
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    mv: MotionVector,
    filter: &FilterConfig,
    luma_corners: &[(i32, i32); 4],
    luma_offsets: &[(usize, usize); 4],
    best_so_far: u64,
) -> u64 {
    let y_w = source.y.width();
    let h_fragments = source.h_fragments;
    let v_fragments = source.v_fragments;
    let mut sad = 0u64;
    for (k, &(dr, dc)) in luma_offsets.iter().enumerate() {
        let br = mb_row * 2 + dr;
        let bc_col = mb_col * 2 + dc;
        if br >= v_fragments || bc_col >= h_fragments {
            continue;
        }
        let pred = predict_mv_luma(prev, mb_row, mb_col, luma_corners[k], mv, filter);
        let src = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
        for (&s, &p) in src.iter().zip(pred.iter()) {
            sad += (s - p as i32).unsigned_abs() as u64;
        }
        // Early-out: once this candidate is already worse than the best,
        // stop accumulating (block-by-block monotone lower bound).
        if sad >= best_so_far {
            return sad;
        }
    }
    sad
}

/// Sum of absolute differences between one source 8×8 luma block and its
/// motion-compensated prediction for a candidate `mv` — the single-block
/// (FourMV) analogue of [`luma_mb_sad`]. `corner` is the block's `(top, left)`
/// offset within the MB (one of the four [`LUMA_CORNERS`]).
#[allow(clippy::too_many_arguments)]
fn luma_block_sad(
    source: &Frame,
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    corner: (i32, i32),
    src_br: usize,
    src_bc: usize,
    mv: MotionVector,
    filter: &FilterConfig,
) -> u64 {
    let y_w = source.y.width();
    let pred = predict_mv_luma(prev, mb_row, mb_col, corner, mv, filter);
    let src = extract_block(source.y.samples(), y_w, src_br * 8, src_bc * 8);
    let mut sad = 0u64;
    for (&s, &p) in src.iter().zip(pred.iter()) {
        sad += (s - p as i32).unsigned_abs() as u64;
    }
    sad
}

/// Search the best whole-then-¼-pel MV for **one 8×8 luma block** (the FourMV
/// per-block search), minimising that block's 8×8 SAD against `prev`. Same
/// two-stage box-then-quarter-pel shape as [`search_luma_mv`] but scoped to a
/// single block at `corner` / source block `(src_br, src_bc)`.
#[allow(clippy::too_many_arguments)]
fn search_luma_block_mv(
    source: &Frame,
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    corner: (i32, i32),
    src_br: usize,
    src_bc: usize,
    filter: &FilterConfig,
) -> (MotionVector, u64) {
    let in_range = |c: i32| c.unsigned_abs() <= MAX_MV_MAGNITUDE;
    let sad_of = |mv: MotionVector| {
        luma_block_sad(
            source, prev, mb_row, mb_col, corner, src_br, src_bc, mv, filter,
        )
    };

    let mut best_mv = MotionVector::ZERO;
    let mut best_sad = sad_of(best_mv);
    for wy in -ME_SEARCH_RANGE..=ME_SEARCH_RANGE {
        for wx in -ME_SEARCH_RANGE..=ME_SEARCH_RANGE {
            if wx == 0 && wy == 0 {
                continue;
            }
            let (qx, qy) = (wx * 4, wy * 4);
            if !in_range(qx) || !in_range(qy) {
                continue;
            }
            let mv = MotionVector::new(qx as i16, qy as i16);
            let sad = sad_of(mv);
            if sad < best_sad {
                best_sad = sad;
                best_mv = mv;
            }
        }
    }

    let base = best_mv;
    for &(ddx, ddy) in &[
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] {
        let qx = base.x as i32 + ddx;
        let qy = base.y as i32 + ddy;
        if !in_range(qx) || !in_range(qy) {
            continue;
        }
        let mv = MotionVector::new(qx as i16, qy as i16);
        let sad = sad_of(mv);
        if sad < best_sad {
            best_sad = sad;
            best_mv = mv;
        }
    }

    (best_mv, best_sad)
}

/// Search for the best whole-then-quarter-pel luma MV for one macroblock
/// against the previous-frame reference, minimising the 16×16 luma SAD.
///
/// Two-stage search: (1) an integer-pel box search over
/// `±ME_SEARCH_RANGE` whole samples around `(0, 0)`, then (2) a ¼-pel
/// refinement over the eight quarter-pel neighbours of the best integer MV.
/// MVs are in ¼-pel units (luma `MvShift == 2`), so an integer step is `4`.
/// Returns the best MV and its SAD.
fn search_luma_mv(
    source: &Frame,
    prev: &BorderedRef,
    mb_row: usize,
    mb_col: usize,
    filter: &FilterConfig,
    luma_corners: &[(i32, i32); 4],
    luma_offsets: &[(usize, usize); 4],
) -> (MotionVector, u64) {
    // Clamp candidate components so |component| stays within the MV encoder's
    // representable range; the §11.5 border also bounds usable MVs.
    let in_range = |c: i32| c.unsigned_abs() <= MAX_MV_MAGNITUDE;

    let sad_of = |mv: MotionVector, best: u64| {
        luma_mb_sad(
            source,
            prev,
            mb_row,
            mb_col,
            mv,
            filter,
            luma_corners,
            luma_offsets,
            best,
        )
    };

    // --- Stage 1: integer-pel box search around (0, 0). ---
    let mut best_mv = MotionVector::ZERO;
    let mut best_sad = sad_of(best_mv, u64::MAX);
    for wy in -ME_SEARCH_RANGE..=ME_SEARCH_RANGE {
        for wx in -ME_SEARCH_RANGE..=ME_SEARCH_RANGE {
            if wx == 0 && wy == 0 {
                continue;
            }
            let (qx, qy) = (wx * 4, wy * 4);
            if !in_range(qx) || !in_range(qy) {
                continue;
            }
            let mv = MotionVector::new(qx as i16, qy as i16);
            let sad = sad_of(mv, best_sad);
            if sad < best_sad {
                best_sad = sad;
                best_mv = mv;
            }
        }
    }

    // --- Stage 2: ¼-pel refinement around the best integer MV. ---
    let base = best_mv;
    for &(ddx, ddy) in &[
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] {
        let qx = base.x as i32 + ddx;
        let qy = base.y as i32 + ddy;
        if !in_range(qx) || !in_range(qy) {
            continue;
        }
        let mv = MotionVector::new(qx as i16, qy as i16);
        let sad = sad_of(mv, best_sad);
        if sad < best_sad {
            best_sad = sad;
            best_mv = mv;
        }
    }

    (best_mv, best_sad)
}

/// Inputs to the per-MB mode decision: the search result, the §11
/// differential reference, the §10 Nearest/Near candidate MVs (when they
/// exist for this MB), and the 16×16 luma SAD of each candidate.
struct MbModeInputs {
    /// The best searched motion vector.
    best_mv: MotionVector,
    /// The 16×16 luma SAD of `best_mv`.
    best_sad: u64,
    /// The §11 differential reference MV (nearest same-reference above/left
    /// neighbour, else zero) — `best_mv − reference_mv` is the New-MV delta.
    reference_mv: MotionVector,
    /// The §10 Nearest neighbour MV, when one exists.
    nearest_mv: Option<MotionVector>,
    /// The §10 Near neighbour MV, when one exists.
    near_mv: Option<MotionVector>,
    /// The luma SAD of the zero motion vector.
    zero_sad: u64,
    /// The luma SAD of `nearest_mv` (when present).
    nearest_sad: Option<u64>,
    /// The luma SAD of `near_mv` (when present).
    near_sad: Option<u64>,
}

/// The chosen coding mode for one MB: the §10 mode, the reconstructed MB MV,
/// and (for `CODE_INTER_PLUS_MV` only) the §11.1 delta to emit.
struct MbModeDecision {
    mode: CodingMode,
    mb_mv: MotionVector,
    delta: Option<(i32, i32)>,
}

/// The best previous-frame (InterLast) coding choice for one MB, with the
/// 16×16 luma SAD that choice reconstructs at. Used by the Golden-aware
/// decision to compare against the best Golden-Frame choice.
struct RefChoice {
    decision: MbModeDecision,
    sad: u64,
}

/// Decide the best coding mode for one MB **against a single reference
/// frame** (previous-frame or Golden), weighing the §10 single-vector modes
/// by reconstruction SAD plus a bit-cost model. Returns the chosen mode and
/// the SAD it reconstructs at, so a caller comparing two reference frames
/// can pick the cheaper one.
///
/// The implicit-MV modes (zero-MV / Nearest / Near) read **no** MV bits — they
/// reuse the zero vector or a neighbour's vector — so they win when their SAD
/// is within [`ME_LAMBDA_SAD`] of the searched New-MV SAD. Among the implicit
/// modes the lowest-SAD one wins; the New-MV mode is taken only when it beats
/// every available implicit option by the margin **and** its delta is
/// representable. `is_golden` selects the Golden-Frame mode set
/// (`CODE_USING_GOLDEN` / `CODE_GOLD_NEAREST_MV` / `CODE_GOLD_NEAR_MV` /
/// `CODE_GOLDEN_MV`) over the previous-frame set; the SAD weighting is
/// identical.
///
/// The decoder reconstructs each implicit mode's MV from the same §10
/// Nearest/Near walk the caller supplied here, so the recorded `mb_mv`
/// matches the decoder's reconstruction exactly.
fn decide_ref_mode(inp: &MbModeInputs, is_golden: bool) -> RefChoice {
    // Mode quartet for this reference: (zero, nearest, near, new).
    let (m_zero, m_nearest, m_near, m_new) = if is_golden {
        (
            CodingMode::UsingGolden,
            CodingMode::GoldNearestMv,
            CodingMode::GoldNearMv,
            CodingMode::GoldenMv,
        )
    } else {
        (
            CodingMode::InterNoMv,
            CodingMode::InterNearestMv,
            CodingMode::InterNearMv,
            CodingMode::InterPlusMv,
        )
    };

    // Candidate implicit-MV options: (mode, mv, sad). Zero is always
    // available; Nearest/Near only when the §10 walk found them.
    let mut best_implicit = (m_zero, MotionVector::ZERO, inp.zero_sad);
    if let (Some(mv), Some(sad)) = (inp.nearest_mv, inp.nearest_sad) {
        if sad < best_implicit.2 {
            best_implicit = (m_nearest, mv, sad);
        }
    }
    if let (Some(mv), Some(sad)) = (inp.near_mv, inp.near_sad) {
        if sad < best_implicit.2 {
            best_implicit = (m_near, mv, sad);
        }
    }

    // A New-MV is worth its bits only if it beats the best implicit option by
    // the bit-cost margin and the delta is representable.
    let delta_x = inp.best_mv.x as i32 - inp.reference_mv.x as i32;
    let delta_y = inp.best_mv.y as i32 - inp.reference_mv.y as i32;
    let delta_representable =
        delta_x.unsigned_abs() <= MAX_MV_MAGNITUDE && delta_y.unsigned_abs() <= MAX_MV_MAGNITUDE;
    let new_is_distinct = inp.best_mv != best_implicit.1;

    if new_is_distinct && delta_representable && inp.best_sad + ME_LAMBDA_SAD < best_implicit.2 {
        RefChoice {
            decision: MbModeDecision {
                mode: m_new,
                mb_mv: inp.best_mv,
                delta: Some((delta_x, delta_y)),
            },
            sad: inp.best_sad,
        }
    } else {
        RefChoice {
            decision: MbModeDecision {
                mode: best_implicit.0,
                mb_mv: best_implicit.1,
                delta: None,
            },
            sad: best_implicit.2,
        }
    }
}

/// Decide one MB's coding mode against the previous-frame reference only
/// (no Golden-Frame consideration). The previous-frame-only ME encoder's
/// decision. See [`decide_ref_mode`] for the SAD/bit-cost weighting.
fn decide_mb_mode(inp: MbModeInputs) -> MbModeDecision {
    decide_ref_mode(&inp, false).decision
}

/// The reference frame a MB's coding mode predicts from, with the chosen
/// mode/MV/delta and the SAD it reconstructs at. The Golden-aware ME encoder's
/// per-MB choice.
struct MbRefDecision {
    decision: MbModeDecision,
    reference: ReferenceBucket,
}

/// Decide one MB's coding mode across **both** the previous-frame and the
/// Golden-Frame references, taking the cheaper reconstruction (with a
/// [`GOLDEN_SWITCH_PENALTY`] hysteresis so a marginal Golden win that costs a
/// reference switch doesn't flip-flop). `prev_in` is the previous-frame mode
/// inputs; `golden_in` is the Golden-Frame mode inputs (each built against its
/// own reference's prediction and §10/§11 neighbour state).
fn decide_mb_mode_golden(prev_in: MbModeInputs, golden_in: MbModeInputs) -> MbRefDecision {
    let prev_choice = decide_ref_mode(&prev_in, false);
    let golden_choice = decide_ref_mode(&golden_in, true);

    // Prefer previous-frame unless Golden beats it by the switch penalty (the
    // Golden mode set costs slightly more mode bits in the §10 tree, and a
    // reference switch loses the same-reference DC/MV prediction continuity).
    if golden_choice.sad + GOLDEN_SWITCH_PENALTY < prev_choice.sad {
        MbRefDecision {
            decision: golden_choice.decision,
            reference: ReferenceBucket::InterGolden,
        }
    } else {
        MbRefDecision {
            decision: prev_choice.decision,
            reference: ReferenceBucket::InterLast,
        }
    }
}

/// Encode a motion-estimated P-frame into the BoolCoder data partition
/// [`crate::inter_frame::decode_inter_frame`] consumes.
///
/// Each macroblock is coded as either `CODE_INTER_NO_MV` (zero MV) or
/// `CODE_INTER_PLUS_MV` (a §11.1-coded MV against the §11 differential
/// reference), whichever yields the lower 16×16 luma SAD by a margin (the
/// New-MV mode must beat zero-MV by more than [`ME_LAMBDA_SAD`] to justify
/// the extra MV bits). This is a strict superset of [`encode_inter_frame`]:
/// a frame with no motion reduces to the all-`CODE_INTER_NO_MV` shape.
///
/// * `source` — the 4:2:0 source pixels for this P-frame.
/// * `prev` — the §11.5-bordered previous-frame reconstruction.
/// * `dct_q_mask` — the §9 6-bit quantiser index (§15 dequant).
/// * `probs` — the per-frame mode/MV/coefficient banks; the caller must hand
///   the **same** banks to the decoder.
/// * `filter` — the §11.3/§11.4 filter configuration.
///
/// The returned `Vec<u8>` is the finished BoolCoder partition: feed it to
/// [`crate::bool_coder::BoolCoder::new`] then `decode_inter_frame` with the
/// same `probs`/`filter`/`prev` to recover the frame.
///
/// # Errors
///
/// [`Error::NotImplemented`] if the source's fragment dimensions exceed the
/// 8-bit header geometry fields.
pub fn encode_inter_frame_me(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    encode_inter_frame_me_body(source, prev, dct_q_mask, probs, filter, |_| {})
}

/// Shared motion-estimated P-frame body: open a [`BoolEncoder`], run
/// `prelude` against it (used by [`encode_inter_frame_me_packet`] to emit
/// the §9 InterHeader tail bits in the same arithmetic stream), then run
/// the per-MB motion search + §10 mode / §11.1 MV / §13 coefficient emit.
fn encode_inter_frame_me_body(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
    prelude: impl FnOnce(&mut BoolEncoder),
) -> Result<Vec<u8>, Error> {
    let h_fragments = source.h_fragments;
    let v_fragments = source.v_fragments;
    if h_fragments > u8::MAX as usize || v_fragments > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let mb_cols = source.mb_cols();
    let mb_rows = source.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    let mut enc = BoolEncoder::new();
    prelude(&mut enc);

    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    // §10/§11 MV neighbour grid, threaded exactly as the decoder builds it:
    // one representative `NeighbourMv` per MB, row-major. Drives both the §10
    // Nearest/Near availability and the §11 differential reference.
    let mut mv_grid: Vec<Option<NeighbourMv>> = vec![None; mb_cols.saturating_mul(mb_rows)];

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    const LUMA_CORNERS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];

    let y_w = source.y.width();
    let u_w = source.u.width();
    let v_w = source.v.width();

    let mut last_mode = CodingMode::InterNoMv;
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- §10 Nearest/Near resolution + availability from the MV grid
            // built so far (exactly the decoder's resolution: same-reference
            // InterLast, zero-MV skip). This single walk supplies both the
            // probXmitted availability row and the Nearest/Near candidate MVs
            // the implicit-MV modes reuse with no MV bits.
            let near = resolve_near_mvs(
                mb_row as i32,
                mb_col as i32,
                ReferenceBucket::InterLast,
                |r, c| grid_lookup(&mv_grid, mb_cols, mb_rows, r, c),
            );
            let availability = near.availability;

            // --- Motion search for the best new vector. ---
            let (best_mv, best_sad) = search_luma_mv(
                source,
                prev,
                mb_row,
                mb_col,
                filter,
                &LUMA_CORNERS,
                &LUMA_OFFSETS,
            );

            // The differential reference for a New-MV MB (nearest same-
            // reference above/left neighbour, else zero); the encoded delta is
            // `best_mv − reference_mv`.
            let reference_mv = select_diff_reference_mv_from_grid(
                &mv_grid,
                mb_cols,
                mb_row as i32,
                mb_col as i32,
                ReferenceBucket::InterLast,
            );

            let sad_of = |mv: MotionVector| {
                luma_mb_sad(
                    source,
                    prev,
                    mb_row,
                    mb_col,
                    mv,
                    filter,
                    &LUMA_CORNERS,
                    &LUMA_OFFSETS,
                    u64::MAX,
                )
            };

            // --- Mode decision: choose among Zero / Nearest / Near (no MV
            // bits) and PlusMv (delta bits, modelled by ME_LAMBDA_SAD). ---
            let decision = decide_mb_mode(MbModeInputs {
                best_mv,
                best_sad,
                reference_mv,
                nearest_mv: near.nearest_mv,
                near_mv: near.near_mv,
                zero_sad: sad_of(MotionVector::ZERO),
                nearest_sad: near.nearest_mv.map(sad_of),
                near_sad: near.near_mv.map(sad_of),
            });
            let mode = decision.mode;
            let mb_mv = decision.mb_mv;

            // --- §10 mode emit ---
            encode_mode_from_probs(&mut enc, mode, &probs.mode_probs, availability, last_mode);
            last_mode = mode;

            // --- §11.1 MV delta emit (New-MV only; the implicit-MV modes and
            // Zero read no MV bits). ---
            if let Some((dx, dy)) = decision.delta {
                encode_mv_pair(&mut enc, dx, dy, &probs.mv_probs);
            }

            // This MB contributes its reconstructed MV to the neighbour grid
            // (intra/zero-MV MBs still record their bucket; zero-MV MBs are
            // skipped by the Nearest/Near walk via the zero-MV predicate).
            mv_grid[mb_row * mb_cols + mb_col] =
                Some(NeighbourMv::new(mb_mv, ReferenceBucket::InterLast));

            // --- four luma blocks (residual against the chosen MV) ---
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let source_pixels = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
                let prediction =
                    predict_mv_luma(prev, mb_row, mb_col, LUMA_CORNERS[k], mb_mv, filter);
                let dc_node_probs = DcContext::from_neighbours(
                    y_grid.left(br, bc_col).is_some_and(|(d, _)| d != 0),
                    y_grid.above(br, bc_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::Y.index()]);
                let coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::Y,
                    &source_pixels,
                    &prediction,
                    dequant,
                    dc_node_probs,
                    probs,
                    &mut y_dc_pred,
                    ReferenceBucket::InterLast,
                    y_grid.left(br, bc_col),
                    y_grid.above(br, bc_col),
                );
                y_grid.set(br, bc_col, coded_dc, ReferenceBucket::InterLast);
            }

            // --- U chroma block (chroma MV == MB MV at ⅛-pel, §11.4) ---
            let u_source = extract_block(source.u.samples(), u_w, mb_row * 8, mb_col * 8);
            let u_pred = predict_mv_chroma(prev, mb_row, mb_col, false, mb_mv, filter);
            let u_dc_node_probs = DcContext::from_neighbours(
                u_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                u_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let u_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &u_source,
                &u_pred,
                dequant,
                u_dc_node_probs,
                probs,
                &mut u_dc_pred,
                ReferenceBucket::InterLast,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            );
            u_grid.set(mb_row, mb_col, u_coded_dc, ReferenceBucket::InterLast);

            // --- V chroma block ---
            let v_source = extract_block(source.v.samples(), v_w, mb_row * 8, mb_col * 8);
            let v_pred = predict_mv_chroma(prev, mb_row, mb_col, true, mb_mv, filter);
            let v_dc_node_probs = DcContext::from_neighbours(
                v_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                v_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let v_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &v_source,
                &v_pred,
                dequant,
                v_dc_node_probs,
                probs,
                &mut v_dc_pred,
                ReferenceBucket::InterLast,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            );
            v_grid.set(mb_row, mb_col, v_coded_dc, ReferenceBucket::InterLast);
        }
    }

    Ok(enc.finish())
}

/// Build the [`MbModeInputs`] for one MB against a single reference frame:
/// run the luma motion search, resolve the §10 Nearest/Near candidates and
/// the §11 differential reference against `ref_bucket`, and compute every
/// candidate's 16×16 luma SAD against `ref_frame`'s prediction. The Golden-
/// aware encoder calls this once per reference (previous-frame and Golden) and
/// compares the two via [`decide_mb_mode_golden`].
#[allow(clippy::too_many_arguments)]
fn mb_inputs_for_ref(
    source: &Frame,
    ref_frame: &BorderedRef,
    ref_bucket: ReferenceBucket,
    mb_row: usize,
    mb_col: usize,
    filter: &FilterConfig,
    mv_grid: &[Option<NeighbourMv>],
    mb_cols: usize,
    mb_rows: usize,
    luma_corners: &[(i32, i32); 4],
    luma_offsets: &[(usize, usize); 4],
) -> (MbModeInputs, NearMvResolution) {
    // §10 Nearest/Near walk filtered on this reference bucket: a Golden MB
    // reuses only a Golden neighbour's MV, a previous-frame MB only a
    // previous-frame neighbour's — exactly the decoder's same-reference walk.
    let near = resolve_near_mvs(mb_row as i32, mb_col as i32, ref_bucket, |r, c| {
        grid_lookup(mv_grid, mb_cols, mb_rows, r, c)
    });

    let (best_mv, best_sad) = search_luma_mv(
        source,
        ref_frame,
        mb_row,
        mb_col,
        filter,
        luma_corners,
        luma_offsets,
    );

    let reference_mv = select_diff_reference_mv_from_grid(
        mv_grid,
        mb_cols,
        mb_row as i32,
        mb_col as i32,
        ref_bucket,
    );

    let sad_of = |mv: MotionVector| {
        luma_mb_sad(
            source,
            ref_frame,
            mb_row,
            mb_col,
            mv,
            filter,
            luma_corners,
            luma_offsets,
            u64::MAX,
        )
    };

    let inputs = MbModeInputs {
        best_mv,
        best_sad,
        reference_mv,
        nearest_mv: near.nearest_mv,
        near_mv: near.near_mv,
        zero_sad: sad_of(MotionVector::ZERO),
        nearest_sad: near.nearest_mv.map(sad_of),
        near_sad: near.near_mv.map(sad_of),
    };
    (inputs, near)
}

/// Encode a **Golden-Frame-aware** motion-estimated P-frame into the
/// BoolCoder data partition [`crate::inter_frame::decode_inter_frame`]
/// consumes — a strict superset of [`encode_inter_frame_me`] that codes each
/// macroblock against **either** the previous-frame **or** the Golden-Frame
/// reference, whichever reconstructs more cheaply.
///
/// Per MB the encoder builds the §10 single-vector mode decision against both
/// references ([`mb_inputs_for_ref`] → [`decide_mb_mode_golden`]) and emits
/// the chosen reference's mode set: `CODE_INTER_NO_MV` / `CODE_INTER_NEAREST_MV`
/// / `CODE_INTER_NEAR_MV` / `CODE_INTER_PLUS_MV` for the previous frame, or
/// `CODE_USING_GOLDEN` / `CODE_GOLD_NEAREST_MV` / `CODE_GOLD_NEAR_MV` /
/// `CODE_GOLDEN_MV` for the Golden Frame. The §11 differential-MV reference,
/// the §10 Nearest/Near walk, and the §14 DC-prediction neighbour filter all
/// run on the chosen reference's bucket, exactly as the decoder reconstructs
/// them, so a mixed previous/golden frame round-trips.
///
/// On a frame whose content is better predicted by the previous frame
/// everywhere this reduces to [`encode_inter_frame_me`]; the Golden path only
/// fires when a MB's Golden reconstruction beats the previous-frame one by
/// [`GOLDEN_SWITCH_PENALTY`].
///
/// * `source` — the 4:2:0 source pixels for this P-frame.
/// * `prev` — the §11.5-bordered previous-frame reconstruction.
/// * `golden` — the §11.5-bordered Golden-Frame reconstruction.
/// * `dct_q_mask` — the §9 6-bit quantiser index (§15 dequant).
/// * `probs` / `filter` — the per-frame banks + filter config; the caller must
///   hand the **same** ones (and the same `prev`/`golden`) to the decoder.
///
/// # Errors
///
/// [`Error::NotImplemented`] if the source's fragment dimensions exceed the
/// 8-bit header geometry fields.
pub fn encode_inter_frame_me_golden(
    source: &Frame,
    prev: &BorderedRef,
    golden: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    encode_inter_frame_me_golden_body(source, prev, golden, dct_q_mask, probs, filter, |_| {})
}

/// Shared Golden-aware motion-estimated P-frame body (see
/// [`encode_inter_frame_me_golden`]). `prelude` emits any §9 header-tail bits
/// into the same arithmetic stream (used by the packet wrapper).
#[allow(clippy::too_many_arguments)]
fn encode_inter_frame_me_golden_body(
    source: &Frame,
    prev: &BorderedRef,
    golden: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
    prelude: impl FnOnce(&mut BoolEncoder),
) -> Result<Vec<u8>, Error> {
    let h_fragments = source.h_fragments;
    let v_fragments = source.v_fragments;
    if h_fragments > u8::MAX as usize || v_fragments > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let mb_cols = source.mb_cols();
    let mb_rows = source.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    let mut enc = BoolEncoder::new();
    prelude(&mut enc);

    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    let mut mv_grid: Vec<Option<NeighbourMv>> = vec![None; mb_cols.saturating_mul(mb_rows)];

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    const LUMA_CORNERS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];

    let y_w = source.y.width();
    let u_w = source.u.width();
    let v_w = source.v.width();

    let mut last_mode = CodingMode::InterNoMv;
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- Availability for the probXmitted mode row: resolved on the
            // previous-frame bucket exactly as the decoder does (the §10
            // availability question that indexes probXmitted is the
            // unconditional "is a previous-frame neighbour present", not the
            // per-reference Nearest/Near filter). ---
            let avail_near = resolve_near_mvs(
                mb_row as i32,
                mb_col as i32,
                ReferenceBucket::InterLast,
                |r, c| grid_lookup(&mv_grid, mb_cols, mb_rows, r, c),
            );
            let availability = avail_near.availability;

            // --- Per-reference mode inputs + the Golden-aware decision. ---
            let (prev_in, _) = mb_inputs_for_ref(
                source,
                prev,
                ReferenceBucket::InterLast,
                mb_row,
                mb_col,
                filter,
                &mv_grid,
                mb_cols,
                mb_rows,
                &LUMA_CORNERS,
                &LUMA_OFFSETS,
            );
            let (golden_in, _) = mb_inputs_for_ref(
                source,
                golden,
                ReferenceBucket::InterGolden,
                mb_row,
                mb_col,
                filter,
                &mv_grid,
                mb_cols,
                mb_rows,
                &LUMA_CORNERS,
                &LUMA_OFFSETS,
            );
            let chosen = decide_mb_mode_golden(prev_in, golden_in);
            let mode = chosen.decision.mode;
            let mb_mv = chosen.decision.mb_mv;
            let reference = chosen.reference;
            let ref_frame = match reference {
                ReferenceBucket::InterGolden => golden,
                _ => prev,
            };

            // --- §10 mode emit + §11.1 MV delta (New-MV modes only) ---
            encode_mode_from_probs(&mut enc, mode, &probs.mode_probs, availability, last_mode);
            last_mode = mode;
            if let Some((dx, dy)) = chosen.decision.delta {
                encode_mv_pair(&mut enc, dx, dy, &probs.mv_probs);
            }

            mv_grid[mb_row * mb_cols + mb_col] = Some(NeighbourMv::new(mb_mv, reference));

            // --- four luma blocks (residual against the chosen reference) ---
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let source_pixels = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
                let prediction =
                    predict_mv_luma(ref_frame, mb_row, mb_col, LUMA_CORNERS[k], mb_mv, filter);
                let dc_node_probs = DcContext::from_neighbours(
                    y_grid.left(br, bc_col).is_some_and(|(d, _)| d != 0),
                    y_grid.above(br, bc_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::Y.index()]);
                let coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::Y,
                    &source_pixels,
                    &prediction,
                    dequant,
                    dc_node_probs,
                    probs,
                    &mut y_dc_pred,
                    reference,
                    y_grid.left(br, bc_col),
                    y_grid.above(br, bc_col),
                );
                y_grid.set(br, bc_col, coded_dc, reference);
            }

            // --- U chroma block ---
            let u_source = extract_block(source.u.samples(), u_w, mb_row * 8, mb_col * 8);
            let u_pred = predict_mv_chroma(ref_frame, mb_row, mb_col, false, mb_mv, filter);
            let u_dc_node_probs = DcContext::from_neighbours(
                u_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                u_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let u_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &u_source,
                &u_pred,
                dequant,
                u_dc_node_probs,
                probs,
                &mut u_dc_pred,
                reference,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            );
            u_grid.set(mb_row, mb_col, u_coded_dc, reference);

            // --- V chroma block ---
            let v_source = extract_block(source.v.samples(), v_w, mb_row * 8, mb_col * 8);
            let v_pred = predict_mv_chroma(ref_frame, mb_row, mb_col, true, mb_mv, filter);
            let v_dc_node_probs = DcContext::from_neighbours(
                v_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                v_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
            )
            .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
            let v_coded_dc = encode_inter_block(
                &mut enc,
                AcPlane::UV,
                &v_source,
                &v_pred,
                dequant,
                v_dc_node_probs,
                probs,
                &mut v_dc_pred,
                reference,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            );
            v_grid.set(mb_row, mb_col, v_coded_dc, reference);
        }
    }

    Ok(enc.finish())
}

/// The SAD margin a `CODE_INTER_FOURMV` macroblock must beat the best
/// single-vector mode by — across the whole 16×16 luma sum of the four
/// per-block searches — before the FourMV encoder spends the extra Table 10
/// codeword + per-block MV bits on four independent vectors. FourMV emits up
/// to four §11.1 deltas where a single-vector MB emits at most one, so the
/// four-vector residual reduction must clear a wider margin than
/// [`ME_LAMBDA_SAD`] to pay for itself.
pub const FOURMV_SAD_MARGIN: u64 = 256;

/// Encode a P-frame that uses `CODE_INTER_FOURMV` macroblocks where four
/// independent per-block motion vectors beat the best single-vector mode by
/// [`FOURMV_SAD_MARGIN`] — a strict superset of [`encode_inter_frame_me`].
///
/// Per MB the encoder runs both the single-vector mode decision (the
/// [`encode_inter_frame_me`] path) **and** a four-block independent search
/// ([`search_luma_block_mv`] per 8×8 luma block). When the FourMV total luma
/// SAD beats the single-vector SAD by the margin **and** every per-block
/// delta is §11.1-representable, the MB is coded `CODE_INTER_FOURMV`:
/// [`crate::fourmv::encode_fourmv_macroblock`] emits the four Table 10 block
/// modes + the per-block deltas, each luma block's residual is formed against
/// its **own** reconstructed vector, and the two chroma blocks against the
/// §10-averaged chroma MV. Otherwise the MB falls back to the single-vector
/// decision.
///
/// A FourMV MB contributes **`None`** to the §10/§11 neighbour grid — exactly
/// as the decoder records it (the FourMV MB-representative-MV is a documented
/// §10 DOCS-GAP) — so the encoder and decoder neighbour contexts stay
/// identical and the frame round-trips.
///
/// * `source` / `prev` / `dct_q_mask` / `probs` / `filter` — as for
///   [`encode_inter_frame_me`].
///
/// # Errors
///
/// [`Error::NotImplemented`] for over-large frame geometry or an
/// unrepresentable FourMV delta (the latter cannot occur because the search is
/// range-clamped, but is surfaced rather than silently mis-coded).
pub fn encode_inter_frame_me_fourmv(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    encode_inter_frame_me_fourmv_body(source, prev, dct_q_mask, probs, filter, |_| {})
}

/// Shared FourMV-capable P-frame body (see [`encode_inter_frame_me_fourmv`]).
#[allow(clippy::too_many_arguments)]
fn encode_inter_frame_me_fourmv_body(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
    prelude: impl FnOnce(&mut BoolEncoder),
) -> Result<Vec<u8>, Error> {
    let h_fragments = source.h_fragments;
    let v_fragments = source.v_fragments;
    if h_fragments > u8::MAX as usize || v_fragments > u8::MAX as usize {
        return Err(Error::NotImplemented);
    }

    let dct_q_mask = dct_q_mask & 0x3F;
    let mb_cols = source.mb_cols();
    let mb_rows = source.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    let mut enc = BoolEncoder::new();
    prelude(&mut enc);

    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    let mut mv_grid: Vec<Option<NeighbourMv>> = vec![None; mb_cols.saturating_mul(mb_rows)];

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    const LUMA_CORNERS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];

    let y_w = source.y.width();
    let u_w = source.u.width();
    let v_w = source.v.width();

    let mut last_mode = CodingMode::InterNoMv;
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            let near = resolve_near_mvs(
                mb_row as i32,
                mb_col as i32,
                ReferenceBucket::InterLast,
                |r, c| grid_lookup(&mv_grid, mb_cols, mb_rows, r, c),
            );
            let availability = near.availability;

            // --- Single-vector decision (the encode_inter_frame_me path). ---
            let (best_mv, best_sad) = search_luma_mv(
                source,
                prev,
                mb_row,
                mb_col,
                filter,
                &LUMA_CORNERS,
                &LUMA_OFFSETS,
            );
            let reference_mv = select_diff_reference_mv_from_grid(
                &mv_grid,
                mb_cols,
                mb_row as i32,
                mb_col as i32,
                ReferenceBucket::InterLast,
            );
            let sad_of = |mv: MotionVector| {
                luma_mb_sad(
                    source,
                    prev,
                    mb_row,
                    mb_col,
                    mv,
                    filter,
                    &LUMA_CORNERS,
                    &LUMA_OFFSETS,
                    u64::MAX,
                )
            };
            let single = decide_mb_mode(MbModeInputs {
                best_mv,
                best_sad,
                reference_mv,
                nearest_mv: near.nearest_mv,
                near_mv: near.near_mv,
                zero_sad: sad_of(MotionVector::ZERO),
                nearest_sad: near.nearest_mv.map(sad_of),
                near_sad: near.near_mv.map(sad_of),
            });
            // The single-vector mode's reconstructed-SAD at its chosen MV.
            let single_sad = sad_of(single.mb_mv);

            // --- FourMV: independent per-block search. ---
            let mut block_mvs = [MotionVector::ZERO; 4];
            let mut fourmv_sad = 0u64;
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let (bmv, bsad) = search_luma_block_mv(
                    source,
                    prev,
                    mb_row,
                    mb_col,
                    LUMA_CORNERS[k],
                    br,
                    bc_col,
                    filter,
                );
                block_mvs[k] = bmv;
                fourmv_sad += bsad;
            }

            // Decide FourMV vs single-vector. FourMV must beat the single SAD by
            // the margin and all per-block deltas must be representable (probed
            // by attempting the emit on a scratch encoder is avoided — instead
            // the encode primitive returns NotImplemented and we fall back).
            let use_fourmv = fourmv_sad + FOURMV_SAD_MARGIN < single_sad
                && block_mvs.iter().any(|mv| !mv.is_zero());

            if use_fourmv {
                // §10 mode emit: CODE_INTER_FOURMV.
                encode_mode_from_probs(
                    &mut enc,
                    CodingMode::InterFourMv,
                    &probs.mode_probs,
                    availability,
                    last_mode,
                );
                last_mode = CodingMode::InterFourMv;

                let fmb = crate::fourmv::encode_fourmv_macroblock(
                    &mut enc,
                    mb_row as i32,
                    mb_col as i32,
                    &block_mvs,
                    &probs.mv_probs,
                    |r, c| grid_lookup(&mv_grid, mb_cols, mb_rows, r, c),
                )?;
                // FourMV contributes None to the neighbour grid (DOCS-GAP).
                mv_grid[mb_row * mb_cols + mb_col] = None;

                // Per-block luma residual against each reconstructed block MV.
                for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                    let br = mb_row * 2 + dr;
                    let bc_col = mb_col * 2 + dc;
                    if br >= v_fragments || bc_col >= h_fragments {
                        continue;
                    }
                    let source_pixels = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
                    let prediction = predict_mv_luma(
                        prev,
                        mb_row,
                        mb_col,
                        LUMA_CORNERS[k],
                        fmb.luma_mvs[k],
                        filter,
                    );
                    let dc_node_probs = DcContext::from_neighbours(
                        y_grid.left(br, bc_col).is_some_and(|(d, _)| d != 0),
                        y_grid.above(br, bc_col).is_some_and(|(d, _)| d != 0),
                    )
                    .select_row(&probs.coeffs.dc_contexts[AcPlane::Y.index()]);
                    let coded_dc = encode_inter_block(
                        &mut enc,
                        AcPlane::Y,
                        &source_pixels,
                        &prediction,
                        dequant,
                        dc_node_probs,
                        probs,
                        &mut y_dc_pred,
                        ReferenceBucket::InterLast,
                        y_grid.left(br, bc_col),
                        y_grid.above(br, bc_col),
                    );
                    y_grid.set(br, bc_col, coded_dc, ReferenceBucket::InterLast);
                }
                // Chroma residual against the §10-averaged chroma MV.
                let chroma_mv = fmb.chroma_mv;
                let u_source = extract_block(source.u.samples(), u_w, mb_row * 8, mb_col * 8);
                let u_pred = predict_mv_chroma(prev, mb_row, mb_col, false, chroma_mv, filter);
                let u_dc_node_probs = DcContext::from_neighbours(
                    u_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                    u_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
                let u_coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::UV,
                    &u_source,
                    &u_pred,
                    dequant,
                    u_dc_node_probs,
                    probs,
                    &mut u_dc_pred,
                    ReferenceBucket::InterLast,
                    u_grid.left(mb_row, mb_col),
                    u_grid.above(mb_row, mb_col),
                );
                u_grid.set(mb_row, mb_col, u_coded_dc, ReferenceBucket::InterLast);

                let v_source = extract_block(source.v.samples(), v_w, mb_row * 8, mb_col * 8);
                let v_pred = predict_mv_chroma(prev, mb_row, mb_col, true, chroma_mv, filter);
                let v_dc_node_probs = DcContext::from_neighbours(
                    v_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                    v_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
                let v_coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::UV,
                    &v_source,
                    &v_pred,
                    dequant,
                    v_dc_node_probs,
                    probs,
                    &mut v_dc_pred,
                    ReferenceBucket::InterLast,
                    v_grid.left(mb_row, mb_col),
                    v_grid.above(mb_row, mb_col),
                );
                v_grid.set(mb_row, mb_col, v_coded_dc, ReferenceBucket::InterLast);
            } else {
                // --- Single-vector fallback (mirrors encode_inter_frame_me). ---
                let mode = single.mode;
                let mb_mv = single.mb_mv;
                encode_mode_from_probs(&mut enc, mode, &probs.mode_probs, availability, last_mode);
                last_mode = mode;
                if let Some((dx, dy)) = single.delta {
                    encode_mv_pair(&mut enc, dx, dy, &probs.mv_probs);
                }
                mv_grid[mb_row * mb_cols + mb_col] =
                    Some(NeighbourMv::new(mb_mv, ReferenceBucket::InterLast));

                for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                    let br = mb_row * 2 + dr;
                    let bc_col = mb_col * 2 + dc;
                    if br >= v_fragments || bc_col >= h_fragments {
                        continue;
                    }
                    let source_pixels = extract_block(source.y.samples(), y_w, br * 8, bc_col * 8);
                    let prediction =
                        predict_mv_luma(prev, mb_row, mb_col, LUMA_CORNERS[k], mb_mv, filter);
                    let dc_node_probs = DcContext::from_neighbours(
                        y_grid.left(br, bc_col).is_some_and(|(d, _)| d != 0),
                        y_grid.above(br, bc_col).is_some_and(|(d, _)| d != 0),
                    )
                    .select_row(&probs.coeffs.dc_contexts[AcPlane::Y.index()]);
                    let coded_dc = encode_inter_block(
                        &mut enc,
                        AcPlane::Y,
                        &source_pixels,
                        &prediction,
                        dequant,
                        dc_node_probs,
                        probs,
                        &mut y_dc_pred,
                        ReferenceBucket::InterLast,
                        y_grid.left(br, bc_col),
                        y_grid.above(br, bc_col),
                    );
                    y_grid.set(br, bc_col, coded_dc, ReferenceBucket::InterLast);
                }

                let u_source = extract_block(source.u.samples(), u_w, mb_row * 8, mb_col * 8);
                let u_pred = predict_mv_chroma(prev, mb_row, mb_col, false, mb_mv, filter);
                let u_dc_node_probs = DcContext::from_neighbours(
                    u_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                    u_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
                let u_coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::UV,
                    &u_source,
                    &u_pred,
                    dequant,
                    u_dc_node_probs,
                    probs,
                    &mut u_dc_pred,
                    ReferenceBucket::InterLast,
                    u_grid.left(mb_row, mb_col),
                    u_grid.above(mb_row, mb_col),
                );
                u_grid.set(mb_row, mb_col, u_coded_dc, ReferenceBucket::InterLast);

                let v_source = extract_block(source.v.samples(), v_w, mb_row * 8, mb_col * 8);
                let v_pred = predict_mv_chroma(prev, mb_row, mb_col, true, mb_mv, filter);
                let v_dc_node_probs = DcContext::from_neighbours(
                    v_grid.left(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                    v_grid.above(mb_row, mb_col).is_some_and(|(d, _)| d != 0),
                )
                .select_row(&probs.coeffs.dc_contexts[AcPlane::UV.index()]);
                let v_coded_dc = encode_inter_block(
                    &mut enc,
                    AcPlane::UV,
                    &v_source,
                    &v_pred,
                    dequant,
                    v_dc_node_probs,
                    probs,
                    &mut v_dc_pred,
                    ReferenceBucket::InterLast,
                    v_grid.left(mb_row, mb_col),
                    v_grid.above(mb_row, mb_col),
                );
                v_grid.set(mb_row, mb_col, v_coded_dc, ReferenceBucket::InterLast);
            }
        }
    }

    Ok(enc.finish())
}

/// The §9-self-describing FourMV-capable P-frame packet (the FourMV dual of
/// [`encode_inter_frame_me_packet`]): the Table 1 raw prefix + Table 3 tail
/// (`RefreshGoldenFrame = 0` / `UseHuffman = 0`) prepended to the FourMV data
/// partition, so it decodes through the top-level [`crate::decode_frame::Vp6Decoder`].
///
/// # Errors
///
/// Propagates [`Error::NotImplemented`] from [`encode_inter_frame_me_fourmv`].
pub fn encode_inter_frame_me_fourmv_packet(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    let dct_q_mask = dct_q_mask & 0x3F;
    let mut header = oxideav_core::bits::BitWriter::with_capacity(1);
    header.write_u32(1, 1); // FrameType = 1 (inter)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(0, 1); // MultiStream = 0
    let raw_prefix = header.finish();

    let data = encode_inter_frame_me_fourmv_body(source, prev, dct_q_mask, probs, filter, |enc| {
        enc.encode_b1(0); // RefreshGoldenFrame = 0
        enc.encode_b1(0); // UseHuffman = 0
    })?;

    let mut out = raw_prefix;
    out.extend_from_slice(&data);
    Ok(out)
}

/// The SAD margin (a Lagrangian λ proxy) the New-MV mode must beat zero-MV
/// by to justify the extra §11.1 MV bits. A motion vector that reduces the
/// 16×16 luma SAD by fewer than this many absolute-difference units is not
/// worth the bits it costs, so the encoder keeps `CODE_INTER_NO_MV`.
pub const ME_LAMBDA_SAD: u64 = 64;

/// The SAD hysteresis the Golden-Frame reference must beat the previous-frame
/// reference by before the Golden-aware encoder switches a MB's reference
/// (see [`decide_mb_mode_golden`]). A reference switch loses the
/// same-reference §14 DC-prediction and §11 differential-MV continuity with
/// the surrounding previous-frame MBs and costs marginally more §10 mode-tree
/// bits, so a Golden win must clear this margin to be worth taking.
pub const GOLDEN_SWITCH_PENALTY: u64 = 128;

/// Grid accessor for the §10/§11 neighbour walks, mirroring the decoder's
/// `neighbour_lookup`: maps an absolute `(row, col)` MB position to its
/// representative neighbour MV, or `None` for off-frame / not-yet-coded.
#[inline]
fn grid_lookup(
    mv_grid: &[Option<NeighbourMv>],
    mb_cols: usize,
    mb_rows: usize,
    row: i32,
    col: i32,
) -> Option<NeighbourMv> {
    if row < 0 || col < 0 || row as usize >= mb_rows || col as usize >= mb_cols {
        return None;
    }
    mv_grid[row as usize * mb_cols + col as usize]
}

/// Encode a complete P-frame **packet** — the §9 InterHeader (raw-bit
/// prefix + BoolCoder-coded tail) followed by the data partition
/// [`encode_inter_frame`] produces — so the result decodes end-to-end
/// through the top-level [`crate::decode_frame::Vp6Decoder`].
///
/// This is the inter-frame dual of [`crate::intra_encode::encode_intra_frame`]'s
/// header emit: where `encode_inter_frame` returns only the BoolCoder
/// data partition the per-MB driver consumes directly, this wrapper
/// prepends the §9 InterHeader so the bytes are a self-describing packet.
///
/// It emits the **simplest valid P-frame shape**, matching the keyframe
/// encoder's Simple/VP6.0 profile:
///
/// * **Table 1 raw prefix** — `FrameType = 1` (inter), `DctQMask`,
///   `MultiStream = 0`. No `Buff2Offset` (the inter-frame header parse
///   stops after Table 1 for the single-partition arrangement).
/// * **InterHeader tail (Table 3)** — `RefreshGoldenFrame b(1) = 0`,
///   then `UseHuffman b(1) = 0`. Simple profile carries no loop-filter
///   fields; VP6.0 (not VP6.2) carries no prediction-filter fields or
///   `PredictionFilterAlpha`, so the tail is exactly those two flags.
///
/// `source` and `prev` are as for [`encode_inter_frame`]: the P-frame
/// source pixels and the §11.5-bordered previous-frame reconstruction.
/// `probs` and `filter` must be the keyframe-baseline banks
/// ([`InterProbs::keyframe`]) and the matching Simple-profile filter
/// config the decoder seeds.
///
/// # Errors
///
/// Propagates [`Error::NotImplemented`] from [`encode_inter_frame`] for
/// an over-large frame geometry.
pub fn encode_inter_frame_packet(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    let dct_q_mask = dct_q_mask & 0x3F;

    // --- §9 raw-bit header prefix (Table 1, byte-aligned) ---
    let mut header = oxideav_core::bits::BitWriter::with_capacity(1);
    header.write_u32(1, 1); // FrameType = 1 (inter)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(0, 1); // MultiStream = 0
    let raw_prefix = header.finish();

    // --- §9 BoolCoder-coded InterHeader tail (Table 3) + data ---
    // The header tail must share the *same* BoolCoder stream the decoder
    // reads (a BoolCoder partition is not byte-splittable), so the two
    // tail bits are emitted as a prelude on the data partition's coder:
    // RefreshGoldenFrame b(1) = 0, then (Simple/VP6.0: no loop/pred-filter
    // fields) UseHuffman b(1) = 0, immediately followed by the per-MB data.
    let data = encode_inter_frame_body(source, prev, dct_q_mask, probs, filter, |enc| {
        enc.encode_b1(0); // RefreshGoldenFrame = 0
        enc.encode_b1(0); // UseHuffman = 0
    })?;

    let mut out = raw_prefix;
    out.extend_from_slice(&data);
    Ok(out)
}

/// Encode a complete **motion-estimated** P-frame packet — the §9
/// InterHeader (raw-bit prefix + BoolCoder-coded tail) followed by the
/// motion-estimated data partition [`encode_inter_frame_me`] produces — so
/// the result decodes end-to-end through the top-level
/// [`crate::decode_frame::Vp6Decoder`].
///
/// Identical header shape to [`encode_inter_frame_packet`] (Simple/VP6.0:
/// Table 1 `FrameType = 1` / `DctQMask` / `MultiStream = 0`, then the
/// Table 3 tail `RefreshGoldenFrame = 0` / `UseHuffman = 0` riding the data
/// partition's BoolCoder), but the body is the motion-estimated encoder, so
/// the packet carries real `CODE_INTER_PLUS_MV` macroblocks where motion
/// search found them worthwhile.
///
/// # Errors
///
/// Propagates [`Error::NotImplemented`] from [`encode_inter_frame_me`] for
/// an over-large frame geometry.
pub fn encode_inter_frame_me_packet(
    source: &Frame,
    prev: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    let dct_q_mask = dct_q_mask & 0x3F;

    // --- §9 raw-bit header prefix (Table 1, byte-aligned) ---
    let mut header = oxideav_core::bits::BitWriter::with_capacity(1);
    header.write_u32(1, 1); // FrameType = 1 (inter)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(0, 1); // MultiStream = 0
    let raw_prefix = header.finish();

    // --- §9 BoolCoder-coded InterHeader tail (Table 3) + ME data ---
    let data = encode_inter_frame_me_body(source, prev, dct_q_mask, probs, filter, |enc| {
        enc.encode_b1(0); // RefreshGoldenFrame = 0
        enc.encode_b1(0); // UseHuffman = 0
    })?;

    let mut out = raw_prefix;
    out.extend_from_slice(&data);
    Ok(out)
}

/// Encode a complete **Golden-aware** motion-estimated P-frame packet — the
/// §9 InterHeader (raw-bit prefix + BoolCoder-coded tail) followed by the
/// Golden-aware data partition [`encode_inter_frame_me_golden`] produces — so
/// the result decodes end-to-end through the top-level
/// [`crate::decode_frame::Vp6Decoder`].
///
/// Identical header shape to [`encode_inter_frame_me_packet`] (Simple/VP6.0:
/// Table 1 `FrameType = 1` / `DctQMask` / `MultiStream = 0`, then the Table 3
/// tail `RefreshGoldenFrame = 0` / `UseHuffman = 0` riding the data
/// partition's BoolCoder). `RefreshGoldenFrame = 0` keeps the Golden Frame the
/// keyframe seeded, which is exactly the reference the body's `*_GOLD*` MBs
/// predict from; the decoder must thread the same `golden` buffer.
///
/// # Errors
///
/// Propagates [`Error::NotImplemented`] from [`encode_inter_frame_me_golden`]
/// for an over-large frame geometry.
pub fn encode_inter_frame_me_golden_packet(
    source: &Frame,
    prev: &BorderedRef,
    golden: &BorderedRef,
    dct_q_mask: u8,
    probs: &InterProbs,
    filter: &FilterConfig,
) -> Result<Vec<u8>, Error> {
    let dct_q_mask = dct_q_mask & 0x3F;

    // --- §9 raw-bit header prefix (Table 1, byte-aligned) ---
    let mut header = oxideav_core::bits::BitWriter::with_capacity(1);
    header.write_u32(1, 1); // FrameType = 1 (inter)
    header.write_u32(dct_q_mask as u32, 6);
    header.write_u32(0, 1); // MultiStream = 0
    let raw_prefix = header.finish();

    // --- §9 BoolCoder-coded InterHeader tail (Table 3) + Golden-aware data ---
    let data = encode_inter_frame_me_golden_body(
        source,
        prev,
        golden,
        dct_q_mask,
        probs,
        filter,
        |enc| {
            enc.encode_b1(0); // RefreshGoldenFrame = 0
            enc.encode_b1(0); // UseHuffman = 0
        },
    )?;

    let mut out = raw_prefix;
    out.extend_from_slice(&data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::inter::{FilterFamily, PredictionFilterPolicy};
    use crate::inter_frame::{decode_inter_frame, InterProbs};
    use crate::intra_frame::IntraProbs;
    use crate::scan::DEFAULT_SCAN_ORDER;

    fn keyframe_inter_probs() -> InterProbs {
        InterProbs::keyframe()
    }

    fn bilinear_filter() -> FilterConfig {
        FilterConfig {
            policy: PredictionFilterPolicy::Fixed(FilterFamily::Bilinear),
            loop_filter_qi: None,
        }
    }

    fn flat(hf: usize, vf: usize, value: u8) -> Frame {
        let mut f = Frame::new(hf, vf);
        for s in f.y.samples_mut() {
            *s = value;
        }
        for s in f.u.samples_mut() {
            *s = value;
        }
        for s in f.v.samples_mut() {
            *s = value;
        }
        f
    }

    fn pattern(hf: usize, vf: usize) -> Frame {
        let mut f = Frame::new(hf, vf);
        let yw = f.y.width();
        let yh = f.y.height();
        for r in 0..yh {
            for c in 0..yw {
                f.y.samples_mut()[r * yw + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        let uw = f.u.width();
        let uh = f.u.height();
        for r in 0..uh {
            for c in 0..uw {
                f.u.samples_mut()[r * uw + c] = (128 + ((r + c) % 40) as i32 - 20) as u8;
                f.v.samples_mut()[r * uw + c] = (128 - ((r * 2 + c) % 40) as i32 + 20) as u8;
            }
        }
        f
    }

    /// Decode the encoder's partition back to a frame against the same
    /// reference + probs + filter, exercising the full P-frame decode.
    fn round_trip(source: &Frame, prev: &Frame, q: u8) -> Frame {
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let prev_bordered = BorderedRef::new(prev);
        // Golden is unused for CODE_INTER_NO_MV (InterLast reference).
        let golden_bordered = BorderedRef::new(prev);

        let bytes = encode_inter_frame(source, &prev_bordered, q, &probs, &filter).expect("encode");
        let mut bc = BoolCoder::new(&bytes).expect("bool coder");
        decode_inter_frame(
            &mut bc,
            source.h_fragments,
            source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev_bordered,
            &golden_bordered,
        )
        .expect("decode")
    }

    /// Decode the **motion-estimated** encoder's partition back to a frame
    /// against the same reference + probs + filter.
    fn round_trip_me(source: &Frame, prev: &Frame, q: u8) -> Frame {
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let prev_bordered = BorderedRef::new(prev);
        let golden_bordered = BorderedRef::new(prev);

        let bytes =
            encode_inter_frame_me(source, &prev_bordered, q, &probs, &filter).expect("ME encode");
        let mut bc = BoolCoder::new(&bytes).expect("bool coder");
        decode_inter_frame(
            &mut bc,
            source.h_fragments,
            source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev_bordered,
            &golden_bordered,
        )
        .expect("decode")
    }

    /// A smooth gradient source — good motion-search material because a
    /// translated copy of it has a well-defined best MV (unlike the
    /// high-frequency `pattern`, where sub-pixel filtering blurs the match).
    fn gradient(hf: usize, vf: usize) -> Frame {
        let mut f = Frame::new(hf, vf);
        let yw = f.y.width();
        let yh = f.y.height();
        for r in 0..yh {
            for c in 0..yw {
                // A smooth bilinear ramp in [16, 235].
                let v = 16 + (r * 200 / yh.max(1) + c * 200 / yw.max(1)) / 2;
                f.y.samples_mut()[r * yw + c] = v.min(235) as u8;
            }
        }
        let uw = f.u.width();
        let uh = f.u.height();
        for r in 0..uh {
            for c in 0..uw {
                f.u.samples_mut()[r * uw + c] = (110 + (r + c) / 2) as u8;
                f.v.samples_mut()[r * uw + c] = (150 - (r + c) / 2) as u8;
            }
        }
        f
    }

    /// Translate one plane by `(dx, dy)` whole pixels with edge replication.
    fn translate_plane(
        src: &crate::frame_assembly::Plane,
        dst: &mut crate::frame_assembly::Plane,
        dx: i32,
        dy: i32,
    ) {
        let w = src.width() as i32;
        let h = src.height() as i32;
        for r in 0..h {
            for c in 0..w {
                let sr = (r + dy).clamp(0, h - 1);
                let sc = (c + dx).clamp(0, w - 1);
                let v = src.samples()[(sr * w + sc) as usize];
                dst.samples_mut()[(r * w + c) as usize] = v;
            }
        }
    }

    /// Translate a frame's luma+chroma by `(dx, dy)` whole pixels with edge
    /// replication, producing a "previous frame" the source moved away from.
    fn translate(src: &Frame, dx: i32, dy: i32) -> Frame {
        let mut out = Frame::new(src.h_fragments, src.v_fragments);
        translate_plane(&src.y, &mut out.y, dx, dy);
        // Chroma is half-resolution, so the chroma shift is half the luma one.
        translate_plane(&src.u, &mut out.u, dx / 2, dy / 2);
        translate_plane(&src.v, &mut out.v, dx / 2, dy / 2);
        out
    }

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

    /// When the source equals the reference frame, every residual is
    /// zero, so the P-frame round-trips **exactly** regardless of
    /// quantiser — the zero-MV inter copy reproduces the reference.
    #[test]
    fn unchanged_frame_round_trips_exactly() {
        let prev = pattern(4, 4);
        let source = prev.clone();
        let out = round_trip(&source, &prev, 16);
        assert_eq!(
            out.y.samples(),
            source.y.samples(),
            "unchanged luma must round-trip exactly via zero-MV inter copy"
        );
        assert_eq!(out.u.samples(), source.u.samples());
        assert_eq!(out.v.samples(), source.v.samples());
    }

    /// A flat source against a flat reference round-trips exactly (zero
    /// residual everywhere).
    #[test]
    fn flat_against_flat_round_trips_exactly() {
        let prev = flat(2, 2, 100);
        let source = flat(2, 2, 100);
        let out = round_trip(&source, &prev, 32);
        assert!(out.y.samples().iter().all(|&s| s == 100));
        assert!(out.u.samples().iter().all(|&s| s == 100));
        assert!(out.v.samples().iter().all(|&s| s == 100));
    }

    /// A source that differs from the reference round-trips above a
    /// quantiser-bounded PSNR floor — the residual is non-trivial and
    /// exercises the full §13 token + §14 DC-prediction encode path.
    #[test]
    fn changed_frame_round_trips_above_floor() {
        let prev = flat(4, 4, 128);
        let source = pattern(4, 4);
        let out = round_trip(&source, &prev, 48);
        let y = psnr(source.y.samples(), out.y.samples());
        let u = psnr(source.u.samples(), out.u.samples());
        let v = psnr(source.v.samples(), out.v.samples());
        assert!(y >= 28.0, "P-frame luma PSNR {y:.2} dB below floor");
        assert!(u >= 30.0, "P-frame U PSNR {u:.2} dB below floor");
        assert!(v >= 30.0, "P-frame V PSNR {v:.2} dB below floor");
    }

    /// A single-MB P-frame round-trips — smallest grid, all six blocks.
    #[test]
    fn single_mb_pframe_round_trips() {
        let prev = flat(2, 2, 120);
        let source = pattern(2, 2);
        let out = round_trip(&source, &prev, 40);
        assert_eq!(out.y.width(), 16);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(y >= 26.0, "single-MB P-frame luma PSNR {y:.2} dB too low");
    }

    /// Finer quantisers reconstruct a changed P-frame more faithfully
    /// than coarser ones — a quantiser-wiring monotonicity check.
    #[test]
    fn finer_quantiser_improves_pframe_psnr() {
        let prev = flat(4, 4, 128);
        let source = pattern(4, 4);
        let coarse = round_trip(&source, &prev, 8);
        let fine = round_trip(&source, &prev, 56);
        let coarse_psnr = psnr(source.y.samples(), coarse.y.samples());
        let fine_psnr = psnr(source.y.samples(), fine.y.samples());
        assert!(
            fine_psnr > coarse_psnr,
            "finer q ({fine_psnr:.2}) should beat coarser ({coarse_psnr:.2})"
        );
    }

    /// Full keyframe → P-frame GOP: encode an I-frame, decode it through
    /// the complete §9 header parse + intra-decode path, seed the §4
    /// `ReferenceFrames` from the reconstruction, encode a P-frame against
    /// that reconstruction, and decode it through
    /// `decode_inter_frame_with_refs`. This is the realistic two-frame
    /// pipeline a stream uses: the P-frame predicts from the *decoded*
    /// keyframe (not the source), so the encoder must subtract exactly the
    /// reconstruction the decoder will hold. The second frame round-trips
    /// above a quantiser-bounded floor against its own source.
    #[test]
    fn keyframe_then_pframe_gop_round_trips() {
        use crate::frame_header::{Vp6FrameHeader, Vp6HeaderTail};
        use crate::inter_frame::{decode_inter_frame_with_refs, ReferenceFrames};
        use crate::intra_frame::decode_intra_frame;

        let q = 40u8;

        // --- Frame 0: keyframe. Encode then decode it end-to-end. ---
        let key_source = pattern(4, 4);
        let key_bytes = crate::intra_encode::encode_intra_frame(&key_source, q).expect("I-encode");
        let hdr = Vp6FrameHeader::parse(&key_bytes).expect("header prefix");
        assert!(hdr.is_keyframe);
        let mut bc = BoolCoder::new(&key_bytes[hdr.raw_prefix_len..]).expect("bool coder");
        let _tail =
            Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
                .expect("tail");
        let key_recon = decode_intra_frame(
            &mut bc,
            key_source.h_fragments,
            key_source.v_fragments,
            hdr.dct_q_mask,
            &IntraProbs::keyframe(),
            &DEFAULT_SCAN_ORDER,
        )
        .expect("I-decode");

        // The reconstructed keyframe seeds both reference buffers (§4).
        let refs = ReferenceFrames::from_keyframe(key_recon.clone());

        // --- Frame 1: P-frame predicted from the *decoded* keyframe. ---
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let (prev_bordered, _golden) = refs.bordered();

        // A second source that differs from the keyframe reconstruction.
        let mut p_source = key_recon.clone();
        // Perturb the source so the residual is non-trivial.
        for (i, s) in p_source.y.samples_mut().iter_mut().enumerate() {
            *s = s.wrapping_add(((i % 7) as u8).wrapping_sub(3));
        }

        let p_bytes =
            encode_inter_frame(&p_source, &prev_bordered, q, &probs, &filter).expect("P-encode");
        let mut p_bc = BoolCoder::new(&p_bytes).expect("P bool coder");
        let p_recon = decode_inter_frame_with_refs(
            &mut p_bc,
            p_source.h_fragments,
            p_source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("P-decode");

        // The P-frame reconstruction tracks its source above a floor.
        let y = psnr(p_source.y.samples(), p_recon.y.samples());
        assert!(
            y >= 30.0,
            "GOP P-frame luma PSNR {y:.2} dB below floor (q={q})"
        );
        assert_eq!(p_recon.y.width(), 32);
    }

    /// A P-frame whose source equals the decoded keyframe round-trips
    /// **exactly** in a real GOP: zero residual against the reconstruction
    /// reproduces it bit-for-bit, independent of quantiser.
    #[test]
    fn gop_unchanged_pframe_is_exact() {
        use crate::frame_header::{Vp6FrameHeader, Vp6HeaderTail};
        use crate::inter_frame::{decode_inter_frame_with_refs, ReferenceFrames};
        use crate::intra_frame::decode_intra_frame;

        let q = 16u8;
        let key_source = pattern(4, 4);
        let key_bytes = crate::intra_encode::encode_intra_frame(&key_source, q).expect("I-encode");
        let hdr = Vp6FrameHeader::parse(&key_bytes).expect("header prefix");
        let mut bc = BoolCoder::new(&key_bytes[hdr.raw_prefix_len..]).expect("bool coder");
        let _ =
            Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
                .expect("tail");
        let key_recon = decode_intra_frame(
            &mut bc,
            key_source.h_fragments,
            key_source.v_fragments,
            hdr.dct_q_mask,
            &IntraProbs::keyframe(),
            &DEFAULT_SCAN_ORDER,
        )
        .expect("I-decode");

        let refs = ReferenceFrames::from_keyframe(key_recon.clone());
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let (prev_bordered, _g) = refs.bordered();

        // Source == decoded keyframe → every residual is zero.
        let p_bytes =
            encode_inter_frame(&key_recon, &prev_bordered, q, &probs, &filter).expect("P-encode");
        let mut p_bc = BoolCoder::new(&p_bytes).expect("P bool coder");
        let p_recon = decode_inter_frame_with_refs(
            &mut p_bc,
            key_recon.h_fragments,
            key_recon.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("P-decode");

        assert_eq!(
            p_recon.y.samples(),
            key_recon.y.samples(),
            "unchanged P-frame must reproduce the keyframe reconstruction exactly"
        );
        assert_eq!(p_recon.u.samples(), key_recon.u.samples());
        assert_eq!(p_recon.v.samples(), key_recon.v.samples());
    }

    /// A keyframe→P-frame GOP whose decode `FilterConfig` is built from the
    /// **decoded keyframe header** via `FilterConfig::from_header` (rather
    /// than a hardcoded family) round-trips an unchanged P-frame exactly.
    /// The encoder emits a Simple-profile keyframe, so the tail's
    /// prediction filter is `NotSignalled` ⇒ §11.4 bilinear with no loop
    /// filter — the header-derived config matches the hand-built one and
    /// the GOP reconstructs bit-for-bit.
    #[test]
    fn gop_filter_config_from_header_round_trips() {
        use crate::frame_header::{Vp6FrameHeader, Vp6HeaderTail};
        use crate::inter_frame::{decode_inter_frame_with_refs, ReferenceFrames};
        use crate::intra_frame::decode_intra_frame;

        let q = 24u8;
        let key_source = pattern(4, 4);
        let key_bytes = crate::intra_encode::encode_intra_frame(&key_source, q).expect("I-encode");
        let hdr = Vp6FrameHeader::parse(&key_bytes).expect("header prefix");
        let mut bc = BoolCoder::new(&key_bytes[hdr.raw_prefix_len..]).expect("bool coder");
        let tail =
            Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
                .expect("tail");

        // Build the operative filter config straight from the decoded
        // header tail — the wiring under test.
        let filter = FilterConfig::from_header(&tail, hdr.dct_q_mask);
        // The encoder's keyframe is Simple profile ⇒ no signalled filter.
        assert_eq!(
            filter.policy,
            PredictionFilterPolicy::Fixed(FilterFamily::Bilinear)
        );
        assert_eq!(filter.loop_filter_qi, None);

        let key_recon = decode_intra_frame(
            &mut bc,
            key_source.h_fragments,
            key_source.v_fragments,
            hdr.dct_q_mask,
            &IntraProbs::keyframe(),
            &DEFAULT_SCAN_ORDER,
        )
        .expect("I-decode");

        let refs = ReferenceFrames::from_keyframe(key_recon.clone());
        let probs = keyframe_inter_probs();
        let (prev_bordered, _g) = refs.bordered();

        let p_bytes =
            encode_inter_frame(&key_recon, &prev_bordered, q, &probs, &filter).expect("P-encode");
        let mut p_bc = BoolCoder::new(&p_bytes).expect("P bool coder");
        let p_recon = decode_inter_frame_with_refs(
            &mut p_bc,
            key_recon.h_fragments,
            key_recon.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("P-decode");

        assert_eq!(
            p_recon.y.samples(),
            key_recon.y.samples(),
            "header-derived FilterConfig must reconstruct the GOP exactly"
        );
        assert_eq!(p_recon.u.samples(), key_recon.u.samples());
        assert_eq!(p_recon.v.samples(), key_recon.v.samples());
    }

    // ===== Motion-estimated encoder (encode_inter_frame_me) =====

    /// The motion-estimated encoder reduces to the all-zero-MV shape when
    /// the source equals the reference: no MV beats zero-MV by the margin,
    /// so every MB stays `CODE_INTER_NO_MV` and the frame round-trips
    /// **exactly** (the ME path is a strict superset of the zero-MV path).
    #[test]
    fn me_unchanged_frame_round_trips_exactly() {
        let prev = gradient(4, 4);
        let source = prev.clone();
        let out = round_trip_me(&source, &prev, 16);
        assert_eq!(
            out.y.samples(),
            source.y.samples(),
            "ME unchanged luma must round-trip exactly"
        );
        assert_eq!(out.u.samples(), source.u.samples());
        assert_eq!(out.v.samples(), source.v.samples());
    }

    /// The ME encoder's bytes always round-trip through the decoder without
    /// error and the reconstruction tracks the source above a floor — the
    /// core correctness invariant (encoder MV-grid / availability / residual
    /// state matches the decoder's reconstruction).
    #[test]
    fn me_translated_source_round_trips_above_floor() {
        // The previous frame is the source shifted right+down by 3 luma px;
        // the encoder should find an MV near (+12, +12) ¼-pel that brings the
        // prediction close to the source, then code the small residual.
        let source = gradient(4, 4);
        let prev = translate(&source, 3, 3);
        let out = round_trip_me(&source, &prev, 32);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(
            y >= 28.0,
            "ME translated-source luma PSNR {y:.2} dB below floor"
        );
        assert_eq!(out.y.width(), 32);
    }

    /// Motion estimation *helps*: on a translated gradient the ME encoder
    /// (which can pick a non-zero MV) reconstructs the source at least as
    /// faithfully as the zero-MV-only encoder, and strictly better when the
    /// shift is large enough that a real MV reduces the residual. We compare
    /// the reconstruction PSNR of the two encoders at the same quantiser.
    #[test]
    fn me_beats_zero_mv_on_translation() {
        let source = gradient(6, 6);
        let prev = translate(&source, 4, 0);
        let q = 40;
        let me_out = round_trip_me(&source, &prev, q);
        let zero_out = round_trip(&source, &prev, q);
        let me_psnr = psnr(source.y.samples(), me_out.y.samples());
        let zero_psnr = psnr(source.y.samples(), zero_out.y.samples());
        assert!(
            me_psnr >= zero_psnr,
            "ME ({me_psnr:.2} dB) should be at least as good as zero-MV \
             ({zero_psnr:.2} dB) on a translated source"
        );
    }

    /// A single-MB ME P-frame round-trips — the smallest grid through the
    /// motion-search + New-MV emit + differential-reference path.
    #[test]
    fn me_single_mb_round_trips() {
        let source = gradient(2, 2);
        let prev = translate(&source, 2, 1);
        let out = round_trip_me(&source, &prev, 36);
        assert_eq!(out.y.width(), 16);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(y >= 26.0, "single-MB ME luma PSNR {y:.2} dB too low");
    }

    /// A multi-MB ME frame where adjacent MBs share a common motion exercises
    /// the §11 differential-reference path: the second MB in a row codes its
    /// MV relative to the left neighbour's reconstructed MV (a small delta),
    /// and the decoder must reconstruct the same absolute MV. The frame
    /// round-trips above a floor, proving the encoder's differential-MV
    /// emission matches the decoder's reconstruction.
    #[test]
    fn me_shared_motion_differential_round_trips() {
        let source = gradient(8, 4);
        let prev = translate(&source, 5, 2);
        let out = round_trip_me(&source, &prev, 28);
        let y = psnr(source.y.samples(), out.y.samples());
        let u = psnr(source.u.samples(), out.u.samples());
        let v = psnr(source.v.samples(), out.v.samples());
        assert!(y >= 28.0, "shared-motion luma PSNR {y:.2} dB below floor");
        assert!(u >= 28.0, "shared-motion U PSNR {u:.2} dB below floor");
        assert!(v >= 28.0, "shared-motion V PSNR {v:.2} dB below floor");
    }

    /// Full keyframe → ME-P-frame GOP: encode/decode an I-frame, seed the §4
    /// refs from the reconstruction, then encode a translated P-frame with
    /// the ME encoder and decode it through `decode_inter_frame_with_refs`.
    /// The realistic two-frame pipeline with real motion.
    #[test]
    fn me_keyframe_then_pframe_gop_round_trips() {
        use crate::frame_header::{Vp6FrameHeader, Vp6HeaderTail};
        use crate::inter_frame::{decode_inter_frame_with_refs, ReferenceFrames};
        use crate::intra_frame::decode_intra_frame;

        let q = 32u8;
        let key_source = gradient(4, 4);
        let key_bytes = crate::intra_encode::encode_intra_frame(&key_source, q).expect("I-encode");
        let hdr = Vp6FrameHeader::parse(&key_bytes).expect("header prefix");
        let mut bc = BoolCoder::new(&key_bytes[hdr.raw_prefix_len..]).expect("bool coder");
        let _tail =
            Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
                .expect("tail");
        let key_recon = decode_intra_frame(
            &mut bc,
            key_source.h_fragments,
            key_source.v_fragments,
            hdr.dct_q_mask,
            &IntraProbs::keyframe(),
            &DEFAULT_SCAN_ORDER,
        )
        .expect("I-decode");

        let refs = ReferenceFrames::from_keyframe(key_recon.clone());
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let (prev_bordered, _golden) = refs.bordered();

        // The P-frame source is the decoded keyframe shifted — so a real MV
        // near (-shift) brings the prediction back onto the source.
        let p_source = translate(&key_recon, 3, 2);

        let p_bytes = encode_inter_frame_me(&p_source, &prev_bordered, q, &probs, &filter)
            .expect("ME P-encode");
        let mut p_bc = BoolCoder::new(&p_bytes).expect("P bool coder");
        let p_recon = decode_inter_frame_with_refs(
            &mut p_bc,
            p_source.h_fragments,
            p_source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("P-decode");

        let y = psnr(p_source.y.samples(), p_recon.y.samples());
        assert!(y >= 28.0, "ME GOP P-frame luma PSNR {y:.2} dB below floor");
        assert_eq!(p_recon.y.width(), 32);
    }

    /// A larger uniformly-translated frame exercises the §10 implicit-MV
    /// (Nearest/Near) modes: once the first MB in a row codes a New-MV, its
    /// right/below neighbours share the *same* motion, so `decide_mb_mode`
    /// reuses the neighbour's vector via `CODE_INTER_NEAREST_MV` /
    /// `CODE_INTER_NEAR_MV` (no MV bits) rather than re-coding the delta. The
    /// frame must still round-trip above a floor — the encoder's implicit-MV
    /// reconstruction matches the decoder's Nearest/Near resolution exactly.
    #[test]
    fn me_uniform_motion_uses_implicit_modes_round_trips() {
        let source = gradient(10, 8);
        let prev = translate(&source, 6, 4);
        let out = round_trip_me(&source, &prev, 28);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(
            y >= 28.0,
            "uniform-motion implicit-mode luma PSNR {y:.2} dB below floor"
        );
        assert_eq!(out.y.width(), 80);
    }

    // ----- decide_mb_mode unit tests -----

    fn mv(x: i16, y: i16) -> MotionVector {
        MotionVector::new(x, y)
    }

    /// With no neighbour MVs and a clearly-better searched vector, the
    /// decision is `CODE_INTER_PLUS_MV` with the delta against the zero
    /// differential reference.
    #[test]
    fn decide_plus_mv_when_search_wins() {
        let d = decide_mb_mode(MbModeInputs {
            best_mv: mv(12, -8),
            best_sad: 100,
            reference_mv: MotionVector::ZERO,
            nearest_mv: None,
            near_mv: None,
            zero_sad: 100_000,
            nearest_sad: None,
            near_sad: None,
        });
        assert_eq!(d.mode, CodingMode::InterPlusMv);
        assert_eq!(d.mb_mv, mv(12, -8));
        assert_eq!(d.delta, Some((12, -8)));
    }

    /// When a Nearest neighbour's vector predicts within the bit-cost margin
    /// of the searched vector, the decision is the implicit
    /// `CODE_INTER_NEAREST_MV` (no delta), reusing the neighbour MV.
    #[test]
    fn decide_nearest_when_within_margin() {
        let d = decide_mb_mode(MbModeInputs {
            best_mv: mv(13, 5),
            best_sad: 980,
            reference_mv: MotionVector::ZERO,
            nearest_mv: Some(mv(12, 4)),
            near_mv: None,
            zero_sad: 50_000,
            // Nearest SAD only ME_LAMBDA_SAD-ish above best: not worth the bits.
            nearest_sad: Some(980 + ME_LAMBDA_SAD),
            near_sad: None,
        });
        assert_eq!(d.mode, CodingMode::InterNearestMv);
        assert_eq!(d.mb_mv, mv(12, 4));
        assert_eq!(d.delta, None);
    }

    /// Zero-MV wins when nothing beats it: no neighbours, and the search
    /// found nothing better than the co-located copy.
    #[test]
    fn decide_zero_when_nothing_better() {
        let d = decide_mb_mode(MbModeInputs {
            best_mv: mv(4, 0),
            best_sad: 1000,
            reference_mv: MotionVector::ZERO,
            nearest_mv: None,
            near_mv: None,
            zero_sad: 1000, // tie → no margin → keep zero
            nearest_sad: None,
            near_sad: None,
        });
        assert_eq!(d.mode, CodingMode::InterNoMv);
        assert_eq!(d.mb_mv, MotionVector::ZERO);
        assert_eq!(d.delta, None);
    }

    /// The cheapest implicit option wins among Zero/Nearest/Near when the
    /// search isn't worth its bits: a Near neighbour with the lowest SAD is
    /// chosen over Zero and Nearest.
    #[test]
    fn decide_picks_lowest_sad_implicit() {
        let d = decide_mb_mode(MbModeInputs {
            best_mv: mv(20, 20),
            best_sad: 900,
            reference_mv: MotionVector::ZERO,
            nearest_mv: Some(mv(2, 2)),
            near_mv: Some(mv(-3, 1)),
            zero_sad: 800,
            nearest_sad: Some(700),
            near_sad: Some(500), // lowest → Near wins
        });
        assert_eq!(d.mode, CodingMode::InterNearMv);
        assert_eq!(d.mb_mv, mv(-3, 1));
        assert_eq!(d.delta, None);
    }

    // ----- Golden-Frame encode-mode tests -----

    /// Decode the **Golden-aware** ME encoder's partition against distinct
    /// `prev` and `golden` references, exercising the §10 `*_GOLD*` modes.
    fn round_trip_me_golden(source: &Frame, prev: &Frame, golden: &Frame, q: u8) -> Frame {
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let prev_b = BorderedRef::new(prev);
        let golden_b = BorderedRef::new(golden);

        let bytes = encode_inter_frame_me_golden(source, &prev_b, &golden_b, q, &probs, &filter)
            .expect("golden ME encode");
        let mut bc = BoolCoder::new(&bytes).expect("bool coder");
        decode_inter_frame(
            &mut bc,
            source.h_fragments,
            source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev_b,
            &golden_b,
        )
        .expect("decode")
    }

    /// When the Golden Frame matches the source far better than the previous
    /// frame, the Golden-aware encoder switches every MB to `CODE_USING_GOLDEN`
    /// and the frame round-trips above a floor — proving the §10 Golden mode
    /// set encodes and decodes against the golden reference.
    #[test]
    fn golden_wins_when_prev_is_garbage_round_trips() {
        let source = gradient(6, 6);
        let golden = source.clone(); // golden == source: a perfect predictor
        let prev = flat(6, 6, 200); // previous frame is unrelated
        let out = round_trip_me_golden(&source, &prev, &golden, 24);
        let y = psnr(source.y.samples(), out.y.samples());
        let u = psnr(source.u.samples(), out.u.samples());
        let v = psnr(source.v.samples(), out.v.samples());
        assert!(
            y >= 30.0,
            "golden-predicted luma PSNR {y:.2} dB below floor"
        );
        assert!(
            u >= 30.0 && v >= 30.0,
            "golden chroma PSNR low: u={u:.2} v={v:.2}"
        );
        assert_eq!(out.y.width(), 48);
    }

    /// With identical `prev` and `golden`, the Golden-aware encoder reduces to
    /// the previous-frame path (the [`GOLDEN_SWITCH_PENALTY`] hysteresis keeps
    /// the previous-frame reference on every tie), so an unchanged frame still
    /// round-trips exactly.
    #[test]
    fn golden_unchanged_frame_round_trips_exactly() {
        let source = gradient(4, 4);
        let out = round_trip_me_golden(&source, &source, &source, 28);
        assert_eq!(
            source.y.samples(),
            out.y.samples(),
            "unchanged golden-aware frame must reconstruct exactly"
        );
        assert_eq!(source.u.samples(), out.u.samples());
        assert_eq!(source.v.samples(), out.v.samples());
    }

    /// A mixed frame: half the MBs are best predicted by the previous frame,
    /// half by the Golden Frame. The Golden-aware encoder must route each MB to
    /// the right reference and the whole frame round-trips above a floor —
    /// exercising a previous↔golden reference transition mid-frame (with the
    /// §14 DC / §11 differential-MV same-reference filtering crossing it).
    #[test]
    fn golden_mixed_reference_frame_round_trips() {
        // prev and golden differ; source matches golden on the left columns
        // and prev on the right columns.
        let golden = gradient(8, 6);
        let prev = {
            // A clearly different ramp so the two references are distinguishable.
            let mut f = Frame::new(8, 6);
            let yw = f.y.width();
            let yh = f.y.height();
            for r in 0..yh {
                for c in 0..yw {
                    f.y.samples_mut()[r * yw + c] = (235 - (r + c) % 200) as u8;
                }
            }
            for s in f.u.samples_mut() {
                *s = 100;
            }
            for s in f.v.samples_mut() {
                *s = 160;
            }
            f
        };
        let mut source = golden.clone();
        // Right half of the luma plane copies the prev frame (so those MBs
        // prefer the previous-frame reference).
        let yw = source.y.width();
        let yh = source.y.height();
        for r in 0..yh {
            for c in (yw / 2)..yw {
                source.y.samples_mut()[r * yw + c] = prev.y.samples()[r * yw + c];
            }
        }

        let out = round_trip_me_golden(&source, &prev, &golden, 20);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(y >= 28.0, "mixed-reference luma PSNR {y:.2} dB below floor");
        assert_eq!(out.y.width(), 64);
    }

    /// A full keyframe → Golden-aware P-frame GOP through the §4
    /// [`ReferenceFrames`]: the keyframe seeds the Golden Frame, the previous
    /// frame is then overwritten with garbage, and the Golden-aware P-frame
    /// (source == keyframe reconstruction) recovers it from the Golden
    /// reference. Proves the Golden mode set threads end-to-end through the
    /// real §4 reference bookkeeping.
    #[test]
    fn golden_gop_recovers_from_golden_reference() {
        use crate::inter_frame::{decode_inter_frame_with_refs, ReferenceFrames};

        let q = 24u8;
        let key = gradient(6, 6);
        // Use the source directly as the "decoded keyframe" stand-in for the
        // reference state (the golden recovery path is what's under test, not
        // the keyframe codec, which has its own GOP tests).
        let mut refs = ReferenceFrames::from_keyframe(key.clone());
        // Corrupt the previous-frame buffer so only the Golden Frame is a good
        // predictor; golden stays == key.
        refs.previous = flat(6, 6, 30);

        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let (prev_b, golden_b) = refs.bordered();

        let p_source = key.clone();
        let p_bytes =
            encode_inter_frame_me_golden(&p_source, &prev_b, &golden_b, q, &probs, &filter)
                .expect("golden ME P-encode");
        let mut p_bc = BoolCoder::new(&p_bytes).expect("P bool coder");
        let p_recon = decode_inter_frame_with_refs(
            &mut p_bc,
            p_source.h_fragments,
            p_source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("golden P-decode");

        let y = psnr(p_source.y.samples(), p_recon.y.samples());
        assert!(
            y >= 30.0,
            "golden-GOP P-frame luma PSNR {y:.2} dB below floor"
        );
    }

    /// `decide_mb_mode_golden` takes the Golden reference only when it beats
    /// the previous-frame SAD by the switch penalty; a marginal Golden win
    /// stays on the previous frame.
    #[test]
    fn golden_decision_respects_switch_penalty() {
        let prev_inputs = || MbModeInputs {
            best_mv: mv(0, 0),
            best_sad: 1000,
            reference_mv: MotionVector::ZERO,
            nearest_mv: None,
            near_mv: None,
            zero_sad: 1000,
            nearest_sad: None,
            near_sad: None,
        };
        // Golden barely better (within penalty) → keep previous frame.
        let marginal = decide_mb_mode_golden(
            prev_inputs(),
            MbModeInputs {
                zero_sad: 1000 - GOLDEN_SWITCH_PENALTY + 1,
                ..prev_inputs()
            },
        );
        assert_eq!(marginal.reference, ReferenceBucket::InterLast);
        assert_eq!(marginal.decision.mode, CodingMode::InterNoMv);

        // Golden clearly better (beyond penalty) → switch to Golden.
        let decisive = decide_mb_mode_golden(
            prev_inputs(),
            MbModeInputs {
                zero_sad: 1000 - GOLDEN_SWITCH_PENALTY - 50,
                ..prev_inputs()
            },
        );
        assert_eq!(decisive.reference, ReferenceBucket::InterGolden);
        assert_eq!(decisive.decision.mode, CodingMode::UsingGolden);
    }

    // ----- FourMV encode-mode tests -----

    /// Decode the **FourMV-capable** encoder's partition back to a frame
    /// against the same reference + probs + filter.
    fn round_trip_me_fourmv(source: &Frame, prev: &Frame, q: u8) -> Frame {
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let prev_b = BorderedRef::new(prev);
        let golden_b = BorderedRef::new(prev);

        let bytes = encode_inter_frame_me_fourmv(source, &prev_b, q, &probs, &filter)
            .expect("FourMV encode");
        let mut bc = BoolCoder::new(&bytes).expect("bool coder");
        decode_inter_frame(
            &mut bc,
            source.h_fragments,
            source.v_fragments,
            q,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev_b,
            &golden_b,
        )
        .expect("decode")
    }

    /// An unchanged frame reduces to all-`CODE_INTER_NO_MV` (no block has a
    /// non-zero best MV, so FourMV never fires) and round-trips **exactly**.
    #[test]
    fn fourmv_unchanged_frame_round_trips_exactly() {
        let source = gradient(4, 4);
        let out = round_trip_me_fourmv(&source, &source, 28);
        assert_eq!(source.y.samples(), out.y.samples());
        assert_eq!(source.u.samples(), out.u.samples());
        assert_eq!(source.v.samples(), out.v.samples());
    }

    /// A uniformly-translated frame is best served by a single MB vector, so
    /// the FourMV encoder falls back to the single-vector path and still
    /// round-trips above a floor (FourMV is a strict superset).
    #[test]
    fn fourmv_uniform_translation_round_trips() {
        let source = gradient(6, 6);
        let prev = translate(&source, 4, 3);
        let out = round_trip_me_fourmv(&source, &prev, 24);
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(
            y >= 28.0,
            "FourMV uniform-translation luma PSNR {y:.2} dB low"
        );
        assert_eq!(out.y.width(), 48);
    }

    /// A frame whose four luma quadrants move in **different** directions —
    /// the canonical FourMV case. The reference is a gradient; the source's
    /// four 8×8 luma quadrants of the single MB are each a differently-shifted
    /// copy of the reference, so independent per-block vectors beat any single
    /// MB vector. The frame must round-trip above a floor, exercising the
    /// `CODE_INTER_FOURMV` emit + per-block reconstruction end-to-end.
    #[test]
    fn fourmv_divergent_block_motion_round_trips() {
        // One macroblock (2×2 fragments).
        let golden = gradient(2, 2);
        // Build the previous frame as the gradient; the source's four quadrants
        // are the previous frame shifted in four different directions, so the
        // per-block search finds four distinct vectors.
        let prev = golden.clone();
        let shifts = [(2, 0), (-2, 0), (0, 2), (0, -2)];
        let mut source = Frame::new(2, 2);
        let yw = source.y.width();
        let yh = source.y.height();
        // Quadrant (block) layout: TL=(0,0), TR=(0,8), BL=(8,0), BR=(8,8).
        let corners = [(0usize, 0usize), (0, 8), (8, 0), (8, 8)];
        for (k, &(qr, qc)) in corners.iter().enumerate() {
            let (dx, dy) = shifts[k];
            for r in 0..8 {
                for c in 0..8 {
                    let sr = (qr + r) as i32;
                    let sc = (qc + c) as i32;
                    let pr = (sr + dy).clamp(0, yh as i32 - 1) as usize;
                    let pc = (sc + dx).clamp(0, yw as i32 - 1) as usize;
                    source.y.samples_mut()[(qr + r) * yw + (qc + c)] =
                        prev.y.samples()[pr * yw + pc];
                }
            }
        }
        // Chroma: copy the reference (the averaged chroma MV handles motion).
        source.u.samples_mut().copy_from_slice(prev.u.samples());
        source.v.samples_mut().copy_from_slice(prev.v.samples());

        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();
        let prev_b = BorderedRef::new(&prev);
        let golden_b = BorderedRef::new(&prev);
        let bytes = encode_inter_frame_me_fourmv(&source, &prev_b, 16, &probs, &filter)
            .expect("FourMV encode");
        let mut bc = BoolCoder::new(&bytes).expect("bool coder");
        let out = decode_inter_frame(
            &mut bc,
            source.h_fragments,
            source.v_fragments,
            16,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev_b,
            &golden_b,
        )
        .expect("decode");
        let y = psnr(source.y.samples(), out.y.samples());
        assert!(
            y >= 26.0,
            "FourMV divergent-motion luma PSNR {y:.2} dB below floor"
        );
        assert_eq!(out.y.width(), 16);
    }

    /// A multi-MB FourMV frame round-trips through the §9-self-describing
    /// packet path (`encode_inter_frame_me_fourmv_packet` →
    /// `decode_frame::Vp6Decoder`), proving the FourMV body threads the header
    /// tail correctly.
    #[test]
    fn fourmv_packet_round_trips_through_decoder() {
        use crate::decode_frame::Vp6Decoder;

        let q = 20u8;
        let key = gradient(4, 4);
        let probs = keyframe_inter_probs();
        let filter = bilinear_filter();

        // Seed the decoder with a keyframe and use its *decoded* reconstruction
        // as the previous-frame reference the FourMV packet is encoded against,
        // so the encoder and decoder predict from the same pixels.
        let key_packet = crate::intra_encode::encode_intra_frame(&key, q).expect("I-encode");
        let mut dec = Vp6Decoder::new();
        let key_recon = dec.decode_packet(&key_packet).expect("I-decode");
        let prev_b = BorderedRef::new(&key_recon);

        // Source: per-quadrant-shifted copy of the decoded keyframe so some MBs
        // go FourMV (the four 8×8 luma quadrants move in different directions).
        let mut source = key_recon.clone();
        let yw = source.y.width();
        let yh = source.y.height();
        for r in 0..yh {
            for c in 0..yw {
                let in_mb_r = (r / 8) % 2;
                let in_mb_c = (c / 8) % 2;
                let dy = if in_mb_r == 0 { 2 } else { -2 };
                let dx = if in_mb_c == 0 { 2 } else { -2 };
                let pr = (r as i32 + dy).clamp(0, yh as i32 - 1) as usize;
                let pc = (c as i32 + dx).clamp(0, yw as i32 - 1) as usize;
                source.y.samples_mut()[r * yw + c] = key_recon.y.samples()[pr * yw + pc];
            }
        }

        let packet = encode_inter_frame_me_fourmv_packet(&source, &prev_b, q, &probs, &filter)
            .expect("FourMV packet encode");
        let p_recon = dec.decode_packet(&packet).expect("FourMV P-decode");
        let y = psnr(source.y.samples(), p_recon.y.samples());
        assert!(y >= 22.0, "FourMV packet luma PSNR {y:.2} dB below floor");
    }
}

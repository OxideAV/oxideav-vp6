//! VP6 inter-frame (P-frame) per-macroblock decode loop (spec §6 / §8 /
//! §10 / §11 / §13 / §14 / §15 / §16 / §17.2–§17.4 / §2).
//!
//! This module is the **fused driver** that ties the crate's already-
//! built per-block and per-macroblock primitives into a single full
//! P-frame decode, the inter-frame analogue of
//! [`crate::intra_frame::decode_intra_frame`]. Where the intra driver
//! decodes every MB as `CODE_INTRA`, a P-frame's MBs carry one of the
//! ten §10 coding modes; the driver walks the MB grid in raster order
//! and, for each MB, decodes — in the §8 single-stream bitstream order
//! (Figure 2 / Figure 6 / Figure 7):
//!
//! 1. The MB **coding mode** (§10, [`crate::mode_decode`]) against the
//!    per-frame `probXmitted` bank, using the §10 Nearest/Near
//!    availability resolved from the MVs decoded so far.
//! 2. The MB **motion state** — for a single-vector mode the §11 MV
//!    ([`crate::mv_decode::reconstruct_macroblock_mv`]); for
//!    `CODE_INTER_FOURMV` the four per-block vectors
//!    ([`crate::fourmv::reconstruct_fourmv_macroblock`]); for
//!    `CODE_INTRA` no motion.
//! 3. The MB's six **block coefficients** (§13, Figure 7: four luma
//!    raster TL/TR/BL/BR, then U, then V), each with §14 DC prediction,
//!    §15 dequant and §16 IDCT.
//! 4. **Reconstruction** (§17): an intra MB recombines via §17.1 (`+128`
//!    level shift, [`crate::reconstruct`]); an inter MB motion-
//!    compensates each block against the previous-frame or golden-frame
//!    reconstruction ([`crate::inter::predict_inter_block_subpel`], which
//!    folds in §11.3 loop filtering and §11.4 sub-pixel interpolation)
//!    and recombines via §17.2/§17.3/§17.4.
//!
//! The §14 DC-prediction neighbour grid and the §10/§11 MV neighbour
//! grid are threaded across the walk so each MB sees its already-decoded
//! left/above/diagonal neighbours, with the §14 same-reference filter and
//! the §10 same-reference-and-non-zero filter both keyed on each block's
//! recorded [`ReferenceBucket`].
//!
//! ## Reference frames and the §11.5 border
//!
//! Inter prediction reads the previous-frame and golden-frame
//! reconstructions. The driver builds the §11.5 UMV-bordered form of each
//! reference plane once ([`crate::umv::build_extended_buffer`]) before the
//! MB walk; every block's motion-compensated fetch reads from the bordered
//! buffer, so unrestricted motion vectors stay well-defined.
//!
//! ## DOCS-GAP — FourMV macroblock neighbour representative
//!
//! §10 resolves Nearest/Near over "decoded macroblock neighbors", one MV
//! per neighbour MB, but never states which of a `CODE_INTER_FOURMV` MB's
//! four per-block luma vectors (or what combination) represents it in a
//! *later* MB's `NearMacroBlocks` list, nor which it contributes as the
//! §11 differential reference for an immediately-right/below `New` MB. The
//! driver reconstructs a FourMV MB's pixels correctly (each luma block
//! uses its own vector) but records the MB as contributing **no**
//! representative neighbour MV (the grid cell stays `None`), deferring the
//! choice rather than guessing. A stream that never places a Nearest/Near
//! or New MB spatially after a FourMV MB decodes identically regardless of
//! the resolution; the gap only affects that specific neighbour
//! interaction.
//!
//! All material here is sequenced from `docs/video/vp6/vp6_format.pdf`
//! (On2 Technologies, document version 1.02, August 2006) and the
//! clean-room errata at `docs/video/vp6/vp6-errata-and-clarifications.md`.
//! No external library code was consulted.

use crate::block_decode::{decode_block_coefficients, BLOCK_SIZE};
use crate::dc_pred::{DcPredictionContext, Neighbour, ReferenceBucket};
use crate::dequant::DequantContext;
use crate::frame_assembly::Frame;
use crate::idct::idct_block;
use crate::inter::{
    predict_inter_block_subpel, reconstruct_inter_block, MvShift, PredictionFilterPolicy,
};
use crate::intra_frame::IntraProbs;
use crate::mode_decode::decode_macroblock_modes;
use crate::modes::{
    CodingMode, ModeAvailability, MvSource, NUM_PROBABILITY_SITUATIONS, PROB_XMITTED_ROW_LEN,
};
use crate::mv_decode::{reconstruct_macroblock_mv, MvProbs, NUM_MV_AXES};
use crate::near_mv::{resolve_near_mvs, MotionVector, NeighbourMv};
use crate::reconstruct::intra_block_to_pixels;
use crate::tokens::{AcPlane, DcContext};
use crate::umv::build_extended_buffer;
use crate::Error;

/// A §11.5-UMV-bordered reference frame: the previous-frame or golden-
/// frame reconstruction with each plane extended by 48 samples on every
/// side so unrestricted motion vectors stay in bounds.
///
/// Built once per decode from a plain [`Frame`] via [`BorderedRef::new`].
#[derive(Debug, Clone)]
pub struct BorderedRef {
    y: Vec<u8>,
    y_stride: usize,
    y_origin: usize,
    u: Vec<u8>,
    u_stride: usize,
    u_origin: usize,
    v: Vec<u8>,
    v_stride: usize,
    v_origin: usize,
}

impl BorderedRef {
    /// Build the §11.5-bordered form of every plane of `frame`.
    pub fn new(frame: &Frame) -> Self {
        let (y, y_stride, y_origin) =
            build_extended_buffer(frame.y.samples(), frame.y.width(), frame.y.height());
        let (u, u_stride, u_origin) =
            build_extended_buffer(frame.u.samples(), frame.u.width(), frame.u.height());
        let (v, v_stride, v_origin) =
            build_extended_buffer(frame.v.samples(), frame.v.width(), frame.v.height());
        Self {
            y,
            y_stride,
            y_origin,
            u,
            u_stride,
            u_origin,
            v,
            v_stride,
            v_origin,
        }
    }

    /// The bordered luma plane: `(samples, stride, origin)`. `origin` is
    /// the flat-buffer index of the in-image top-left sample; `stride` is
    /// the bordered row length. Lets an encoder form the **same** zero-MV
    /// prediction the decoder does without duplicating the §11.5 border
    /// construction.
    #[inline]
    pub fn y_plane(&self) -> (&[u8], usize, usize) {
        (&self.y, self.y_stride, self.y_origin)
    }

    /// The bordered U-chroma plane: `(samples, stride, origin)`.
    #[inline]
    pub fn u_plane(&self) -> (&[u8], usize, usize) {
        (&self.u, self.u_stride, self.u_origin)
    }

    /// The bordered V-chroma plane: `(samples, stride, origin)`.
    #[inline]
    pub fn v_plane(&self) -> (&[u8], usize, usize) {
        (&self.v, self.v_stride, self.v_origin)
    }
}

/// The §4 reference-frame state a VP6 stream maintains across frames:
/// the previous-frame reconstruction and the Golden Frame (spec §4).
///
/// Per §4:
///
/// > The alternative prediction, or Golden Frame, is a frame buffer that
/// > by default holds the last decoded I-frame but it may be updated at
/// > any time. A flag in the frame header indicates to the decoder
/// > whether or not to update the Golden Frame buffer.
///
/// and
///
/// > This reference frame may either be the reconstruction of the
/// > immediately previous frame in the sequence or a stored previous
/// > frame known as the Golden Frame.
///
/// [`ReferenceFrames`] tracks both buffers and applies the §4 update
/// rules: every decoded frame becomes the new previous-frame
/// reconstruction; the Golden Frame is replaced when the frame is an
/// I-frame (which seeds it) or when the InterHeader's `RefreshGoldenFrame`
/// flag is set. A P-frame decode reads both buffers (the mode's
/// [`ReferenceBucket`] selects which); this holder threads them.
///
/// The stored buffers are plain [`Frame`]s; the §11.5 UMV border is
/// rebuilt per decode by [`ReferenceFrames::bordered`] (the spec's "copy
/// in its entirety including any UMV border" only matters because the
/// border is part of the buffer — rebuilding it from the unbordered
/// reconstruction is bit-identical, since the border is pure edge
/// replication of the same samples).
#[derive(Debug, Clone)]
pub struct ReferenceFrames {
    /// The immediately-previous decoded frame reconstruction.
    pub previous: Frame,
    /// The Golden Frame reconstruction.
    pub golden: Frame,
}

impl ReferenceFrames {
    /// Seed both buffers from a freshly-decoded I-frame. Per §4 the
    /// Golden Frame "by default holds the last decoded I-frame", and the
    /// previous-frame buffer is likewise this reconstruction.
    pub fn from_keyframe(keyframe: Frame) -> Self {
        Self {
            previous: keyframe.clone(),
            golden: keyframe,
        }
    }

    /// The coded block-grid geometry `(h_fragments, v_fragments)` of the
    /// reference frames — the dimensions an inter frame inherits since
    /// Table 3 carries no geometry of its own.
    pub fn coded_fragments(&self) -> (usize, usize) {
        (self.previous.h_fragments, self.previous.v_fragments)
    }

    /// Build the §11.5-bordered form of both reference buffers for one
    /// P-frame decode: `(previous, golden)`.
    pub fn bordered(&self) -> (BorderedRef, BorderedRef) {
        (
            BorderedRef::new(&self.previous),
            BorderedRef::new(&self.golden),
        )
    }

    /// Apply the §4 post-decode reference update for a just-decoded
    /// frame `decoded`.
    ///
    /// * `is_keyframe` — the frame was an I-frame, so it both becomes the
    ///   previous-frame reconstruction **and** replaces the Golden Frame
    ///   (§4: the Golden Frame defaults to "the last decoded I-frame").
    /// * `refresh_golden` — the InterHeader's `RefreshGoldenFrame` flag
    ///   for a P-frame; when set, the decoded P-frame replaces the Golden
    ///   Frame too (§4: "it may be updated at any time").
    ///
    /// In all cases the decoded frame becomes the new previous-frame
    /// reconstruction.
    pub fn update_after_decode(&mut self, decoded: Frame, is_keyframe: bool, refresh_golden: bool) {
        if is_keyframe || refresh_golden {
            self.golden = decoded.clone();
        }
        self.previous = decoded;
    }
}

/// The per-frame mode/MV probability banks a P-frame decode needs, the
/// inter-frame analogue of [`IntraProbs`].
///
/// All four banks (mode `probXmitted`, the two-axis MV probs, plus the
/// inherited coefficient banks carried in `coeffs`) are supplied already
/// in their post-update state — the caller has applied the §10
/// mode-probability updates, the §11.2 MV-probability updates and the
/// §13 coefficient-probability updates from the frame header before
/// constructing this struct.
#[derive(Debug, Clone)]
pub struct InterProbs {
    /// The §10 `probXmitted[3][20]` mode-decode bank (post-§10 update).
    pub mode_probs: [[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
    /// The §11 two-axis motion-vector probability bank (post-§11.2
    /// update).
    pub mv_probs: [MvProbs; NUM_MV_AXES],
    /// The §13 coefficient banks (DC node contexts, AC, zero-run) —
    /// reused verbatim from the intra path, post-§13 update.
    pub coeffs: IntraProbs,
}

impl InterProbs {
    /// Build the keyframe-baseline inter probability banks: the §10
    /// `probXmitted` mode bank at its [`crate::modes::VP6_BASELINE_XMITTED_PROBS`]
    /// defaults, the §11 two-axis MV bank at [`MvProbs::defaults`], and the
    /// §13 coefficient banks at the [`IntraProbs::keyframe`] baselines.
    ///
    /// This is the inter dual of [`IntraProbs::keyframe`]: the starting
    /// point a per-frame driver feeds into the §10/§11.2/§13 header
    /// probability-update sub-streams before dispatching a P-frame whose
    /// header signalled no updates (every probability stays at baseline).
    pub fn keyframe() -> Self {
        Self {
            mode_probs: crate::modes::VP6_BASELINE_XMITTED_PROBS,
            mv_probs: [
                MvProbs::defaults(crate::mv_decode::MV_AXIS_X),
                MvProbs::defaults(crate::mv_decode::MV_AXIS_Y),
            ],
            coeffs: IntraProbs::keyframe(),
        }
    }
}

/// The §11.3 / §11.4 prediction-filter configuration for a frame.
#[derive(Debug, Clone, Copy)]
pub struct FilterConfig {
    /// The §11.4 prediction-filter family policy (fixed or auto-select).
    pub policy: PredictionFilterPolicy,
    /// `Some(quantizer_index)` enables the §11.3 prediction loop filter;
    /// `None` disables it (Simple Profile or `UseLoopFilter == 0`). The
    /// index is the frame's `DctQMask` selecting the §11.3 limit value.
    pub loop_filter_qi: Option<usize>,
}

impl FilterConfig {
    /// Build the operative §11.3/§11.4 frame filter configuration from a
    /// decoded frame-header tail plus the frame's quantiser index.
    ///
    /// This is the bridge from the *signalled* header fields to the
    /// per-frame policy the per-block predictor consumes:
    ///
    /// * The §11.4 prediction-filter family policy comes from
    ///   [`crate::frame_header::PredictionFilter::resolve`] (MV-size and
    ///   variance thresholds, the bicubic alpha). On a Simple-profile or
    ///   non-VP6.2 frame whose selector was absent it resolves to fixed
    ///   bilinear, matching §11.4's "Simple Profile … Bilinear filtering
    ///   is used in all cases".
    /// * The §11.3 prediction loop filter is enabled (carrying the
    ///   frame's quantiser index) only when the tail's
    ///   [`LoopFilter`](crate::frame_header::LoopFilter) reports
    ///   `Enabled`; a `Disabled` / `NotSignalled` selector (Simple
    ///   profile, an IntraHeader, or `UseLoopFilter == 0`) leaves
    ///   `loop_filter_qi` `None`.
    ///
    /// `dct_q_mask` is the frame's `DctQMask` (Table 1), the index the
    /// §11.3 deblocking limit is selected with.
    pub fn from_header(tail: &crate::frame_header::Vp6HeaderTail, dct_q_mask: u8) -> Self {
        let policy = tail.prediction_filter.resolve(tail.prediction_filter_alpha);
        let loop_filter_qi = match tail.loop_filter {
            crate::frame_header::LoopFilter::Enabled { .. } => Some(dct_q_mask as usize),
            crate::frame_header::LoopFilter::Disabled
            | crate::frame_header::LoopFilter::NotSignalled => None,
        };
        FilterConfig {
            policy,
            loop_filter_qi,
        }
    }

    /// The Simple-profile / VP6.0 default filter configuration: §11.4 fixed
    /// bilinear interpolation with the §11.3 prediction loop filter disabled.
    ///
    /// This is the operative config the top-level decoder derives for the
    /// keyframe-baseline P-frame shape the in-tree encoders emit (Simple
    /// profile → bilinear; no `UseLoopFilter`), exposed as a named constructor
    /// so the encoder and the decode path share one source of truth instead of
    /// hand-building the `PredictionFilterPolicy::Fixed(FilterFamily::Bilinear)`
    /// literal.
    pub fn bilinear() -> Self {
        FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        }
    }
}

/// One block-grid cell of the §14 DC-prediction neighbour grid: the
/// reconstructed coded DC plus the block's reference bucket.
#[derive(Debug, Clone, Copy)]
struct DcCell {
    coded_dc: i32,
    reference: ReferenceBucket,
}

/// A per-plane record of each decoded block's coded DC + reference, the
/// inter-frame counterpart of the intra driver's `PlaneDcGrid`. Carries
/// the reference bucket so the §14 same-reference filter is meaningful
/// (a P-frame mixes intra, last-ref and golden-ref blocks).
#[derive(Debug, Clone)]
struct PlaneDcGrid {
    cols: usize,
    cells: Vec<Option<DcCell>>,
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
    fn above(&self, row: usize, col: usize) -> Option<DcCell> {
        if row == 0 {
            return None;
        }
        self.cells[self.idx(row - 1, col)]
    }

    #[inline]
    fn left(&self, row: usize, col: usize) -> Option<DcCell> {
        if col == 0 {
            return None;
        }
        self.cells[self.idx(row, col - 1)]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, cell: DcCell) {
        let i = self.idx(row, col);
        self.cells[i] = Some(cell);
    }
}

/// The decoded coefficients + DC bookkeeping for one block, ready for
/// reconstruction.
struct DecodedBlock {
    /// Scan-order coefficients with the §14-reconstructed coded DC
    /// already spliced into position 0.
    scan_coeffs: [i32; BLOCK_SIZE],
    /// The reconstructed coded DC (for the neighbour grid).
    coded_dc: i32,
}

/// Decode one block's coefficients and resolve its §14 DC prediction,
/// returning the scan-order coefficients (DC spliced) and the coded DC.
///
/// Shared by the intra and inter reconstruction paths in a P-frame: the
/// §13 coefficient decode and §14 DC prediction are identical; only the
/// final §17 recombination differs (caller's responsibility).
#[allow(clippy::too_many_arguments)]
fn decode_block(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    plane: AcPlane,
    reference: ReferenceBucket,
    probs: &IntraProbs,
    dc_pred: &mut DcPredictionContext,
    left: Option<DcCell>,
    above: Option<DcCell>,
) -> Result<DecodedBlock, Error> {
    // §13.2 Table 26 DC node-probability context from the left/above
    // coded-DC zero-ness (absent neighbour counts as zero, §13.2). The
    // same-reference filter does NOT gate the Table-26 context — that
    // gating is a §14-prediction concern, while Table 26 keys purely on
    // the spatially adjacent coded-DC zero-ness.
    let dc_context = DcContext::from_neighbours(
        left.is_some_and(|c| c.coded_dc != 0),
        above.is_some_and(|c| c.coded_dc != 0),
    );
    let dc_node_probs = dc_context.select_row(&probs.dc_contexts[plane.index()]);

    let block =
        decode_block_coefficients(bc, plane, dc_node_probs, &probs.ac_probs, &probs.zrl_probs)?;

    // §14 DC prediction: reconstructed coded DC = predictor + delta. The
    // same-reference rule filters out neighbours whose bucket differs.
    let left_n = left.map(|c| Neighbour {
        dc: c.coded_dc,
        reference: c.reference,
    });
    let above_n = above.map(|c| Neighbour {
        dc: c.coded_dc,
        reference: c.reference,
    });
    let dc_delta = block.coeffs[0];
    let predictor = dc_pred.predict(reference, left_n, above_n);
    let coded_dc = predictor.wrapping_add(dc_delta);
    dc_pred.set_last_dc(reference, coded_dc);

    let mut scan_coeffs = block.coeffs;
    scan_coeffs[0] = coded_dc;

    Ok(DecodedBlock {
        scan_coeffs,
        coded_dc,
    })
}

/// Dequantize + scan→raster + §16 IDCT, returning the raster residual.
fn residual_of(
    scan_coeffs: &[i32; BLOCK_SIZE],
    scan_to_raster: &[u8; BLOCK_SIZE],
    dequant: DequantContext,
) -> [i32; BLOCK_SIZE] {
    let mut raster =
        crate::block_decode::dequantize_to_raster(scan_coeffs, scan_to_raster, dequant);
    idct_block(&mut raster);
    raster
}

/// Decode a full inter (P-)frame from a single BoolCoder partition,
/// motion-compensating against the previous-frame and golden-frame
/// reconstructions and returning the reconstructed 4:2:0 [`Frame`].
///
/// * `bc` — a [`crate::bool_coder::BoolCoder`] positioned at the start of
///   the data partition (after the §9 header tail and any §10/§11.2/§13
///   probability-update sub-streams the caller has already consumed).
/// * `h_fragments` / `v_fragments` — the §9 luma block-grid dimensions
///   (inherited from the keyframe; an InterHeader carries no geometry).
/// * `dct_q_mask` — the §9 6-bit quantizer index (§15 dequant + §11.3
///   loop-filter limit selection).
/// * `probs` — the post-update mode/MV/coefficient banks.
/// * `scan_to_raster` — the active §12 scan permutation.
/// * `filter` — the §11.3/§11.4 prediction-filter configuration.
/// * `prev` / `golden` — the §11.5-bordered previous-frame and golden-
///   frame reconstructions ([`BorderedRef`]).
///
/// # Errors
///
/// [`Error::Truncated`] if the partition runs out mid-MB.
#[allow(clippy::too_many_arguments)]
pub fn decode_inter_frame(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    h_fragments: usize,
    v_fragments: usize,
    dct_q_mask: u8,
    probs: &InterProbs,
    scan_to_raster: &[u8; BLOCK_SIZE],
    filter: &FilterConfig,
    prev: &BorderedRef,
    golden: &BorderedRef,
) -> Result<Frame, Error> {
    let mut frame = Frame::new(h_fragments, v_fragments);
    let mb_cols = frame.mb_cols();
    let mb_rows = frame.mb_rows();
    let dequant = DequantContext::new(dct_q_mask);

    // §14 per-plane DC neighbour grids + prediction contexts.
    let mut y_grid = PlaneDcGrid::new(h_fragments, v_fragments);
    let mut u_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut v_grid = PlaneDcGrid::new(mb_cols, mb_rows);
    let mut y_dc_pred = DcPredictionContext::new();
    let mut u_dc_pred = DcPredictionContext::new();
    let mut v_dc_pred = DcPredictionContext::new();

    // §10/§11 MV neighbour grid: one representative NeighbourMv per MB,
    // row-major. A FourMV MB stays `None` (DOCS-GAP, see module docs); an
    // intra MB also stays `None` (it never qualifies as a same-reference
    // inter neighbour and carries no MV).
    let mut mv_grid: Vec<Option<NeighbourMv>> = vec![None; mb_cols.saturating_mul(mb_rows)];

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    const LUMA_CORNERS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];

    let mut last_mode = CodingMode::InterNoMv;

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            // --- Figure 6: mode, then motion (single-stream order) ---
            let availability = {
                // Resolve §10 availability from the MV grid built so far.
                // A grid lookup returns the representative neighbour MV.
                let res = resolve_near_mvs(
                    mb_row as i32,
                    mb_col as i32,
                    // Availability is computed per the mode being decoded;
                    // §10 resolves it against the *previous-frame* bucket
                    // for the unconditional "is a neighbour present"
                    // question used to index probXmitted. The walker
                    // filters on this reference, matching the spec's
                    // ProbabilitySituation derivation.
                    ReferenceBucket::InterLast,
                    |r, c| neighbour_lookup(&mv_grid, mb_cols, mb_rows, r, c),
                );
                res.availability
            };
            let mode = decode_one_mode(bc, &probs.mode_probs, availability, last_mode)?;
            last_mode = mode;

            // Resolve the MB's motion state and the representative
            // neighbour MV it contributes.
            let (mb_mv, four_mvs, representative) = resolve_motion(
                bc,
                mode,
                mb_row,
                mb_col,
                &probs.mv_probs,
                &mv_grid,
                mb_cols,
                mb_rows,
            )?;
            mv_grid[mb_row * mb_cols + mb_col] = representative;

            let reference = mode.reference_bucket();

            // --- Figure 7: six block coefficients + reconstruction ---
            let mut luma_pixels = [[128u8; 64]; 4];
            for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
                let br = mb_row * 2 + dr;
                let bc_col = mb_col * 2 + dc;
                if br >= v_fragments || bc_col >= h_fragments {
                    continue;
                }
                let left = y_grid.left(br, bc_col);
                let above = y_grid.above(br, bc_col);
                let decoded = decode_block(
                    bc,
                    AcPlane::Y,
                    reference,
                    &probs.coeffs,
                    &mut y_dc_pred,
                    left,
                    above,
                )?;
                y_grid.set(
                    br,
                    bc_col,
                    DcCell {
                        coded_dc: decoded.coded_dc,
                        reference,
                    },
                );
                let residual = residual_of(&decoded.scan_coeffs, scan_to_raster, dequant);
                luma_pixels[k] = reconstruct_luma_block(
                    mode,
                    reference,
                    mb_mv,
                    four_mvs.as_ref().map(|m| m[k]),
                    LUMA_CORNERS[k],
                    mb_row,
                    mb_col,
                    &residual,
                    filter,
                    prev,
                    golden,
                );
            }

            // --- U then V chroma blocks ---
            let chroma_mv = chroma_motion(mode, mb_mv, four_mvs.as_ref());
            let u_decoded = decode_block(
                bc,
                AcPlane::UV,
                reference,
                &probs.coeffs,
                &mut u_dc_pred,
                u_grid.left(mb_row, mb_col),
                u_grid.above(mb_row, mb_col),
            )?;
            u_grid.set(
                mb_row,
                mb_col,
                DcCell {
                    coded_dc: u_decoded.coded_dc,
                    reference,
                },
            );
            let u_residual = residual_of(&u_decoded.scan_coeffs, scan_to_raster, dequant);
            let u_pixels = reconstruct_chroma_block(
                mode,
                reference,
                chroma_mv,
                mb_row,
                mb_col,
                &u_residual,
                filter,
                prev,
                golden,
                ChromaPlane::U,
            );

            let v_decoded = decode_block(
                bc,
                AcPlane::UV,
                reference,
                &probs.coeffs,
                &mut v_dc_pred,
                v_grid.left(mb_row, mb_col),
                v_grid.above(mb_row, mb_col),
            )?;
            v_grid.set(
                mb_row,
                mb_col,
                DcCell {
                    coded_dc: v_decoded.coded_dc,
                    reference,
                },
            );
            let v_residual = residual_of(&v_decoded.scan_coeffs, scan_to_raster, dequant);
            let v_pixels = reconstruct_chroma_block(
                mode,
                reference,
                chroma_mv,
                mb_row,
                mb_col,
                &v_residual,
                filter,
                prev,
                golden,
                ChromaPlane::V,
            );

            frame
                .place_macroblock(mb_row, mb_col, &luma_pixels, &u_pixels, &v_pixels)
                .map_err(|_| Error::Truncated)?;
        }
    }

    Ok(frame)
}

/// Decode a P-frame against a [`ReferenceFrames`] state holder, building
/// the §11.5 borders internally.
///
/// Convenience wrapper over [`decode_inter_frame`] that takes the
/// [`ReferenceFrames`] (previous + golden) directly and constructs the
/// bordered buffers, so the caller threads only the reference *state*
/// across frames and doesn't manage [`BorderedRef`] lifetimes. The
/// caller still applies the §4 reference update afterward via
/// [`ReferenceFrames::update_after_decode`].
#[allow(clippy::too_many_arguments)]
pub fn decode_inter_frame_with_refs(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    h_fragments: usize,
    v_fragments: usize,
    dct_q_mask: u8,
    probs: &InterProbs,
    scan_to_raster: &[u8; BLOCK_SIZE],
    filter: &FilterConfig,
    refs: &ReferenceFrames,
) -> Result<Frame, Error> {
    let (prev, golden) = refs.bordered();
    decode_inter_frame(
        bc,
        h_fragments,
        v_fragments,
        dct_q_mask,
        probs,
        scan_to_raster,
        filter,
        &prev,
        &golden,
    )
}

/// Identify a chroma plane for the reconstruction helper.
#[derive(Debug, Clone, Copy)]
enum ChromaPlane {
    U,
    V,
}

/// Decode one MB coding mode (single-MB wrapper over
/// [`decode_macroblock_modes`] using a 1×1 grid with the supplied
/// availability and last-mode).
fn decode_one_mode(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    mode_probs: &[[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
    availability: ModeAvailability,
    last_mode: CodingMode,
) -> Result<CodingMode, Error> {
    let modes = decode_macroblock_modes(bc, mode_probs, 1, 1, last_mode, |_, _| availability)?;
    Ok(modes[0])
}

/// Resolve an MB's full motion state from its decoded mode, returning the
/// single MB-level MV (for single-vector + chroma derivation), the four
/// per-block luma MVs (for FourMV), and the representative neighbour MV
/// the MB contributes to the grid.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn resolve_motion(
    bc: &mut crate::bool_coder::BoolCoder<'_>,
    mode: CodingMode,
    mb_row: usize,
    mb_col: usize,
    mv_probs: &[MvProbs; NUM_MV_AXES],
    mv_grid: &[Option<NeighbourMv>],
    mb_cols: usize,
    mb_rows: usize,
) -> Result<(MotionVector, Option<[MotionVector; 4]>, Option<NeighbourMv>), Error> {
    match mode.mv_source() {
        MvSource::FourMv => {
            let fmb = crate::fourmv::reconstruct_fourmv_macroblock(
                bc,
                mb_row as i32,
                mb_col as i32,
                mv_probs,
                |r, c| neighbour_lookup(mv_grid, mb_cols, mb_rows, r, c),
            )?;
            // DOCS-GAP: no defined MB-representative for a FourMV MB; it
            // contributes `None` to the neighbour grid. Its own chroma MV
            // is still used for the two chroma blocks of *this* MB.
            Ok((fmb.chroma_mv, Some(fmb.luma_mvs), None))
        }
        _ => {
            let mb = reconstruct_macroblock_mv(
                bc,
                mode,
                mb_row as i32,
                mb_col as i32,
                mv_probs,
                |r, c| neighbour_lookup(mv_grid, mb_cols, mb_rows, r, c),
            )?;
            // An intra MB contributes no inter neighbour MV.
            let representative = if mode.is_intra() {
                None
            } else {
                Some(mb.as_neighbour())
            };
            Ok((mb.mv, None, representative))
        }
    }
}

/// Grid accessor for the §10/§11 neighbour walks: maps an absolute
/// `(row, col)` MB position to its representative neighbour MV, or `None`
/// for off-frame / not-yet-decoded / FourMV / intra cells.
#[inline]
fn neighbour_lookup(
    mv_grid: &[Option<NeighbourMv>],
    mb_cols: usize,
    mb_rows: usize,
    row: i32,
    col: i32,
) -> Option<NeighbourMv> {
    if row < 0 || col < 0 || row >= mb_rows as i32 || col >= mb_cols as i32 {
        return None;
    }
    mv_grid[row as usize * mb_cols + col as usize]
}

/// Derive the chroma motion vector for an MB. For single-vector modes the
/// chroma MV is the MB MV (re-interpreted at ⅛-pel by the §11.4
/// MvShift::Chroma decode); for FourMV the chroma MV is the §10 average of
/// the four luma vectors already computed in `four_mvs`'s owner.
fn chroma_motion(
    _mode: CodingMode,
    mb_mv: MotionVector,
    _four_mvs: Option<&[MotionVector; 4]>,
) -> MotionVector {
    // resolve_motion already folds the FourMV chroma average into mb_mv
    // (it returns fmb.chroma_mv as the MB MV), so mb_mv is the chroma MV
    // for every mode.
    mb_mv
}

/// Reconstruct one luma block: intra → §17.1 (`+128`); inter → motion-
/// compensate against the mode's reference frame + §17 recombine.
#[allow(clippy::too_many_arguments)]
fn reconstruct_luma_block(
    mode: CodingMode,
    reference: ReferenceBucket,
    mb_mv: MotionVector,
    block_mv: Option<MotionVector>,
    corner: (i32, i32),
    mb_row: usize,
    mb_col: usize,
    residual: &[i32; 64],
    filter: &FilterConfig,
    prev: &BorderedRef,
    golden: &BorderedRef,
) -> [u8; 64] {
    if mode.is_intra() {
        return intra_block_to_pixels(residual);
    }
    let mv = block_mv.unwrap_or(mb_mv);
    let r = match reference {
        ReferenceBucket::InterGolden => golden,
        _ => prev,
    };
    // Co-located corner of this luma block in the bordered buffer.
    let top = (mb_row * 16) as i32 + corner.0;
    let left = (mb_col * 16) as i32 + corner.1;
    let base = (r.y_origin as i32 + top * r.y_stride as i32 + left) as usize;
    let mut pred = [0u8; 64];
    predict_inter_block_subpel(
        &r.y,
        r.y_stride,
        base,
        mv.x as i32,
        mv.y as i32,
        MvShift::Luma,
        filter.policy,
        filter.loop_filter_qi,
        &mut pred,
    );
    let mut out = [0u8; 64];
    reconstruct_inter_block(&pred, residual, &mut out);
    out
}

/// Reconstruct one chroma block: intra → §17.1; inter → motion-compensate
/// against the mode's reference chroma plane + §17 recombine.
#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_block(
    mode: CodingMode,
    reference: ReferenceBucket,
    chroma_mv: MotionVector,
    mb_row: usize,
    mb_col: usize,
    residual: &[i32; 64],
    filter: &FilterConfig,
    prev: &BorderedRef,
    golden: &BorderedRef,
    plane: ChromaPlane,
) -> [u8; 64] {
    if mode.is_intra() {
        return intra_block_to_pixels(residual);
    }
    let r = match reference {
        ReferenceBucket::InterGolden => golden,
        _ => prev,
    };
    let (buf, stride, origin) = match plane {
        ChromaPlane::U => (&r.u, r.u_stride, r.u_origin),
        ChromaPlane::V => (&r.v, r.v_stride, r.v_origin),
    };
    let top = (mb_row * 8) as i32;
    let left = (mb_col * 8) as i32;
    let base = (origin as i32 + top * stride as i32 + left) as usize;
    let mut pred = [0u8; 64];
    predict_inter_block_subpel(
        buf,
        stride,
        base,
        chroma_mv.x as i32,
        chroma_mv.y as i32,
        MvShift::Chroma,
        filter.policy,
        filter.loop_filter_qi,
        &mut pred,
    );
    let mut out = [0u8; 64];
    reconstruct_inter_block(&pred, residual, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::frame_header::{LoopFilter, PredictionFilter, Vp6HeaderTail};
    use crate::inter::FilterFamily;
    use crate::scan::DEFAULT_SCAN_ORDER;

    /// A minimal Inter-frame `Vp6HeaderTail` with the given filter
    /// selectors; all geometry fields `None` (carried on the IntraHeader).
    fn inter_tail(
        loop_filter: LoopFilter,
        prediction_filter: PredictionFilter,
        prediction_filter_alpha: Option<u8>,
    ) -> Vp6HeaderTail {
        Vp6HeaderTail {
            v_fragments: None,
            h_fragments: None,
            output_v_fragments: None,
            output_h_fragments: None,
            scaling_mode: None,
            refresh_golden_frame: Some(false),
            loop_filter,
            prediction_filter,
            prediction_filter_alpha,
            use_huffman: false,
        }
    }

    /// `FilterConfig::from_header` resolves a fixed-bicubic Advanced-profile
    /// selector and an enabled loop filter into the operative policy +
    /// loop-filter quantiser index.
    #[test]
    fn filter_config_from_header_fixed_bicubic_loop_on() {
        let tail = inter_tail(
            LoopFilter::Enabled { selector: 0 },
            PredictionFilter::Fixed { bicubic: true },
            Some(9),
        );
        let cfg = FilterConfig::from_header(&tail, 42);
        assert_eq!(
            cfg.policy,
            PredictionFilterPolicy::Fixed(FilterFamily::Bicubic { alpha: 9 })
        );
        assert_eq!(cfg.loop_filter_qi, Some(42));
    }

    /// A `NotSignalled` selector (Simple profile / omitted) plus a
    /// `Disabled` loop filter resolves to fixed bilinear with no loop
    /// filtering — §11.4's Simple-profile-is-bilinear default.
    #[test]
    fn filter_config_from_header_not_signalled_is_bilinear_no_loop() {
        let tail = inter_tail(LoopFilter::Disabled, PredictionFilter::NotSignalled, None);
        let cfg = FilterConfig::from_header(&tail, 7);
        assert_eq!(
            cfg.policy,
            PredictionFilterPolicy::Fixed(FilterFamily::Bilinear)
        );
        assert_eq!(cfg.loop_filter_qi, None);
    }

    /// An auto-select selector resolves the §11.4 thresholds and, with the
    /// loop filter `NotSignalled`, leaves `loop_filter_qi` `None`.
    #[test]
    fn filter_config_from_header_auto_select_thresholds() {
        let tail = inter_tail(
            LoopFilter::NotSignalled,
            PredictionFilter::AutoSelect {
                var_thresh: 2,
                mv_size_thresh: 3,
            },
            Some(4),
        );
        let cfg = FilterConfig::from_header(&tail, 12);
        assert_eq!(
            cfg.policy,
            PredictionFilterPolicy::AutoSelect {
                // (1 << (3-1)) << 2 == 4 << 2 == 16.
                mv_size_thresh_qpel: 16,
                // 2 << 5 == 64.
                var_thresh: 64,
                alpha: 4,
            }
        );
        assert_eq!(cfg.loop_filter_qi, None);
    }

    fn keyframe_inter_probs() -> InterProbs {
        InterProbs::keyframe()
    }

    fn flat_ref(h_fragments: usize, v_fragments: usize, value: u8) -> Frame {
        let mut f = Frame::new(h_fragments, v_fragments);
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

    /// An all-zero bitstream over the keyframe baselines drives every
    /// per-MB decision to the 0-branch. Under VP6_BASELINE_XMITTED_PROBS
    /// the resulting mode is a single, deterministic value; whatever it
    /// is, the frame must decode to completion with the right dimensions
    /// and clamped pixels — a smoke test of the fused driver wiring.
    #[test]
    fn all_zero_stream_decodes_to_completion() {
        let bytes = [0u8; 512];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = keyframe_inter_probs();
        let prev = BorderedRef::new(&flat_ref(4, 4, 100));
        let golden = BorderedRef::new(&flat_ref(4, 4, 50));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let frame = decode_inter_frame(
            &mut bc,
            4,
            4,
            32,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev,
            &golden,
        )
        .expect("decode");
        assert_eq!(frame.y.width(), 32);
        assert_eq!(frame.y.height(), 32);
        assert_eq!(frame.u.width(), 16);
        // Every sample is a valid u8 (the type guarantees it); assert the
        // plane is fully populated.
        assert_eq!(frame.y.samples().len(), 32 * 32);
        assert_eq!(frame.u.samples().len(), 16 * 16);
    }

    /// A single-MB frame decodes its six blocks from the all-zero stream
    /// without running out of bytes.
    #[test]
    fn single_mb_inter_frame_decodes() {
        let bytes = [0u8; 256];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = keyframe_inter_probs();
        let prev = BorderedRef::new(&flat_ref(2, 2, 128));
        let golden = BorderedRef::new(&flat_ref(2, 2, 128));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let frame = decode_inter_frame(
            &mut bc,
            2,
            2,
            0,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &prev,
            &golden,
        )
        .expect("decode");
        assert_eq!(frame.y.width(), 16);
        assert_eq!(frame.y.height(), 16);
    }

    /// An inter `CODE_INTER_NO_MV` luma block with a zero MV and zero
    /// residual copies the co-located reference samples verbatim (§17.2
    /// zero-vector recombination with a zero error signal). Drives the
    /// `reconstruct_luma_block` helper directly so the inter-copy
    /// semantics are pinned independently of the bitstream mode walk.
    #[test]
    fn inter_zero_mv_copies_reference() {
        // A reference with a known per-sample pattern so a copy is
        // distinguishable from a flat fill.
        let mut f = Frame::new(2, 2);
        for (i, s) in f.y.samples_mut().iter_mut().enumerate() {
            *s = (i % 200) as u8;
        }
        let prev = BorderedRef::new(&f);
        let golden = BorderedRef::new(&flat_ref(2, 2, 0));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let zero = [0i32; 64];
        // The TL luma block (corner (0,0)) of MB (0,0) with a zero MV.
        let out = reconstruct_luma_block(
            CodingMode::InterNoMv,
            ReferenceBucket::InterLast,
            MotionVector::ZERO,
            None,
            (0, 0),
            0,
            0,
            &zero,
            &filter,
            &prev,
            &golden,
        );
        // It must equal the co-located 8×8 region of the reference plane.
        for r in 0..8 {
            for c in 0..8 {
                let expect = f.y.sample(r, c).unwrap();
                assert_eq!(out[r * 8 + c], expect, "zero-MV inter copy at ({r},{c})");
            }
        }
    }

    /// An inter golden-reference block reads from the golden frame, not
    /// the previous frame — `CODE_USING_GOLDEN` routes to `InterGolden`.
    #[test]
    fn golden_mode_reads_golden_frame() {
        let prev = BorderedRef::new(&flat_ref(2, 2, 10));
        let golden = BorderedRef::new(&flat_ref(2, 2, 200));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let zero = [0i32; 64];
        let out = reconstruct_luma_block(
            CodingMode::UsingGolden,
            ReferenceBucket::InterGolden,
            MotionVector::ZERO,
            None,
            (0, 0),
            0,
            0,
            &zero,
            &filter,
            &prev,
            &golden,
        );
        // Golden frame is flat 200, so every output sample is 200.
        assert!(out.iter().all(|&p| p == 200), "golden-ref copy reads 200");
    }

    /// An intra MB in a P-frame reconstructs via §17.1 (`+128` level
    /// shift) regardless of the reference frames — a zero-residual intra
    /// block is mid-grey 128, not a copy of the reference.
    #[test]
    fn intra_mb_in_pframe_is_mid_grey() {
        let prev = BorderedRef::new(&flat_ref(2, 2, 77));
        let golden = BorderedRef::new(&flat_ref(2, 2, 200));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let zero = [0i32; 64];
        let out = reconstruct_luma_block(
            CodingMode::Intra,
            ReferenceBucket::Intra,
            MotionVector::ZERO,
            None,
            (0, 0),
            0,
            0,
            &zero,
            &filter,
            &prev,
            &golden,
        );
        assert!(out.iter().all(|&p| p == 128), "zero-residual intra → 128");
    }

    /// A non-zero residual adds onto the inter prediction and clips, the
    /// §17.2/§17.3 recombination. A flat reference of 100 plus a uniform
    /// +20 residual produces 120.
    #[test]
    fn inter_residual_adds_to_prediction() {
        let prev = BorderedRef::new(&flat_ref(2, 2, 100));
        let golden = BorderedRef::new(&flat_ref(2, 2, 0));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let residual = [20i32; 64];
        let out = reconstruct_luma_block(
            CodingMode::InterNoMv,
            ReferenceBucket::InterLast,
            MotionVector::ZERO,
            None,
            (0, 0),
            0,
            0,
            &residual,
            &filter,
            &prev,
            &golden,
        );
        assert!(out.iter().all(|&p| p == 120), "100 + 20 = 120");
    }

    /// The §11.5 bordered reference round-trips a flat plane: every
    /// in-image and border sample equals the source constant.
    #[test]
    fn bordered_ref_flat_plane() {
        let f = flat_ref(2, 2, 77);
        let r = BorderedRef::new(&f);
        // The origin sample and a deep-border sample both read 77.
        assert_eq!(r.y[r.y_origin], 77);
        assert_eq!(r.y[0], 77);
    }

    // ---- §4 reference-frame bookkeeping ----------------------------

    /// `from_keyframe` seeds both the previous-frame and Golden buffers
    /// with the I-frame reconstruction (§4 default).
    #[test]
    fn keyframe_seeds_both_buffers() {
        let key = flat_ref(2, 2, 42);
        let refs = ReferenceFrames::from_keyframe(key);
        assert!(refs.previous.y.samples().iter().all(|&s| s == 42));
        assert!(refs.golden.y.samples().iter().all(|&s| s == 42));
    }

    /// A P-frame with `RefreshGoldenFrame == 0` updates only the
    /// previous-frame buffer; the Golden Frame is unchanged (§4).
    #[test]
    fn pframe_no_refresh_keeps_golden() {
        let mut refs = ReferenceFrames::from_keyframe(flat_ref(2, 2, 10));
        let decoded = flat_ref(2, 2, 99);
        refs.update_after_decode(decoded, false, false);
        assert!(refs.previous.y.samples().iter().all(|&s| s == 99));
        assert!(
            refs.golden.y.samples().iter().all(|&s| s == 10),
            "golden unchanged without refresh"
        );
    }

    /// A P-frame with `RefreshGoldenFrame == 1` replaces both buffers
    /// (§4: "it may be updated at any time").
    #[test]
    fn pframe_refresh_replaces_golden() {
        let mut refs = ReferenceFrames::from_keyframe(flat_ref(2, 2, 10));
        let decoded = flat_ref(2, 2, 200);
        refs.update_after_decode(decoded, false, true);
        assert!(refs.previous.y.samples().iter().all(|&s| s == 200));
        assert!(refs.golden.y.samples().iter().all(|&s| s == 200));
    }

    /// A subsequent I-frame re-seeds the Golden Frame even without the
    /// refresh flag (§4: Golden defaults to the last decoded I-frame).
    #[test]
    fn keyframe_reseeds_golden() {
        let mut refs = ReferenceFrames::from_keyframe(flat_ref(2, 2, 10));
        // A P-frame moves previous forward but not golden.
        refs.update_after_decode(flat_ref(2, 2, 50), false, false);
        // A new I-frame re-seeds golden.
        refs.update_after_decode(flat_ref(2, 2, 77), true, false);
        assert!(refs.previous.y.samples().iter().all(|&s| s == 77));
        assert!(refs.golden.y.samples().iter().all(|&s| s == 77));
    }

    /// The `_with_refs` wrapper decodes against a `ReferenceFrames` and
    /// produces the same dimensions as the explicit-border form.
    #[test]
    fn decode_with_refs_wrapper() {
        let bytes = [0u8; 256];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let probs = keyframe_inter_probs();
        let refs = ReferenceFrames::from_keyframe(flat_ref(2, 2, 128));
        let filter = FilterConfig {
            policy: PredictionFilterPolicy::Fixed(crate::inter::FilterFamily::Bilinear),
            loop_filter_qi: None,
        };
        let frame = decode_inter_frame_with_refs(
            &mut bc,
            2,
            2,
            0,
            &probs,
            &DEFAULT_SCAN_ORDER,
            &filter,
            &refs,
        )
        .expect("decode");
        assert_eq!(frame.y.width(), 16);
        assert_eq!(frame.y.height(), 16);
    }
}

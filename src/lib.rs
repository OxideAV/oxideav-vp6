//! # oxideav-vp6
//!
//! Pure-Rust On2 VP6 ("vp6f"/"VP60"/"VP61"/"VP62") video decoder for
//! the [`oxideav`](https://github.com/OxideAV/oxideav) framework.
//!
//! **Status:** clean-room rebuild in progress (orphan-rebuild scaffold
//! from 2026-05-18; rounds 1–2 below). The crate previously contained a
//! direct-port implementation that was retired under the workspace
//! clean-room provenance policy. The rebuild reads only:
//!
//! * `docs/video/vp6/vp6_format.pdf` — On2 Technologies' "VP6 Bitstream
//!   & Decoder Specification" (document version 1.02, August 2006).
//!
//! No third-party VP6 source has been consulted at any stage of the
//! rebuild.
//!
//! ## Round 1 surface
//!
//! * [`Vp6FrameHeader`] — parser for the raw-bit prefix of a VP6
//!   frame header (spec §9 Table 1 plus Table 2's four R(n) fields).
//!   Covers `FrameType`, `DctQMask`, `MultiStream`, `Vp3VersionNo`,
//!   `VpProfile`, the `Reserved` bit, and the conditional 16-bit
//!   `Buff2Offset`.
//! * [`Error`] — narrow `Truncated` / `NotImplemented` enum so callers
//!   can distinguish "I ran out of bytes" from "this code path isn't
//!   wired up yet".
//!
//! ## Round 2 surface
//!
//! * [`DequantContext`] — per-frame inverse-quantization context (spec
//!   §15). Resolves the DC and AC scalar quantizer factors from the
//!   header's `DctQMask` via the two 64-entry tables
//!   ([`DC_QUANTIZATION_TABLE`] / [`AC_QUANTIZATION_TABLE`]) and
//!   dequantizes a block's coefficients with a per-coefficient
//!   multiply. This layer is **independent of the BoolCoder** — the
//!   quantizer factor depends only on the already-parsed raw-bit
//!   `DctQMask` — so it advances past the round-1 surface without
//!   touching the contested §7.3 `Split` formula.
//!
//! What rounds 1–2 deliberately do **not** land:
//!
//! * Any field downstream of the BoolCoder switch in the frame header
//!   (`VFragments`/`HFragments`/scaling/filter selectors/`UseHuffman`),
//!   mode/MV decoding, and DCT-token decoding — every one of these is
//!   `b(n)`/`B(x)`/`T` BoolCoder-coded.
//! * The BoolCoder primitive itself (the `VP6_DecodeBool` bit decoder).
//!   Its initialization (`VP6_StartDecode`) and normalization are
//!   unambiguous, but the per-bit `Split` step is blocked.
//!
//! See the `DOCS-GAP` section below for the spec defect blocking that
//! step.
//!
//! ## DOCS-GAP: spec §7.3 BoolCoder Split formula
//!
//! `vp6_format.pdf` p. 15 defines the per-bit `Split` value as:
//!
//! ```text
//! Split = 1 + ( ((Range-1) * Probability) >> 7 )
//! ```
//!
//! with `Probability` in `1..=255` and `Range` in `0..=255` (post-
//! normalisation `Range >= 128`). With `Range = 255` and
//! `Probability = 128` (the value the spec's `b(x)` notation uses for
//! every fixed-prior raw bit) this evaluates to
//! `1 + ((254 * 128) >> 7) = 1 + 254 = 255`, i.e. `Split == Range`.
//!
//! The decode then compares `Value < (Split << 24)`. With `Split =
//! 255` that's `Value < 0xFF00_0000`, which is true for any
//! post-normalisation `Value` strictly less than `Range << 24 =
//! 0xFF00_0000`. The 0-branch is unconditionally taken; `Range` stays
//! at 255; no normalisation occurs; `Value` is unchanged; **the next
//! call returns the same state and the same 0-bit**, indefinitely.
//!
//! Empirically, all `b(n)` reads collapse to all-zeros, which can't
//! be what the encoder is producing for non-zero frame headers.
//!
//! Symmetric breakage at the other end of the range: `Probability =
//! 255` gives `Split = 1 + ((254 * 255) >> 7) = 507`, and
//! `Split << 24` overflows `u32`.
//!
//! Both symptoms vanish if `>> 7` is replaced by `>> 8`
//! (`Split = Range/2` at `Probability = 128`; `Split <= Range` for all
//! valid probabilities), which is the formula used in other binary
//! arithmetic coders in the VPx family.
//!
//! This is a spec **defect**, not a silence: §7.3 gives an explicit
//! formula, but that formula is provably self-contradictory against
//! the spec's own `b(x)` fixed-prob-128 raw-bit reads. We deliberately
//! **do not** "guess" the `>> 8` fix per the workspace "ask for docs,
//! don't fish" rule. The BoolCoder-coded layers (frame-header tail,
//! mode/MV decoding, DCT-token decoding) stay blocked on a docs patch
//! clarifying the Split formula — either confirming `>> 7` is correct
//! (and explaining the encoder-side mapping that makes it work) or
//! correcting it to `>> 8`.
//!
//! Round 2 worked **around** the block by landing the inverse-
//! quantization layer (spec §15, [`DequantContext`]), which is driven
//! solely by the raw-bit `DctQMask` and never calls `VP6_DecodeBool`.
//!
//! ## Round 3 surface
//!
//! * [`idct_block`] — the spec §16 inverse DCT transform. A separable,
//!   fixed-point integer IDCT (14-bit precision, seven Q16 cosine
//!   constants) that converts an 8x8 block of dequantized coefficients
//!   in raster order back to pixel / pixel-difference values via a row
//!   pass and a column pass. Like the §15 dequant layer it is
//!   **BoolCoder-independent** — it consumes the output of
//!   [`DequantContext::dequantize_block`] and never calls
//!   `VP6_DecodeBool` — so it advances past round 2 without touching
//!   the contested §7.3 `Split` formula.
//!
//! ## Round 4 surface
//!
//! * [`reconstruct_intra_block`] — the spec §17.1 intra block
//!   reconstruction step: per-sample `OutputValue = InputValue + 128`
//!   followed by an inclusive clip to `0..=255`. Inverts the
//!   encoder-side level shift that §17.1 documents (`prior to encoding
//!   the value 128 is subtracted from all data samples`). The natural
//!   successor to the §16 IDCT for the intra-coded path; like §15
//!   and §16 it is **BoolCoder-independent** so it advances the decoder
//!   past round 3 without touching the contested §7.3 `Split` formula.
//!   The remaining §17.2–§17.4 cases (zero MV, full-pixel MV, sub-pixel
//!   MV) combine the same clip with motion compensation against a
//!   reference reconstruction buffer; they are blocked on the BoolCoder
//!   for MV decoding upstream.
//!
//! ## Round 5 surface
//!
//! * [`interp`] — the spec §11.4 fractional-pixel motion-compensation
//!   interpolation filters: the bilinear 2-tap kernel ([`interp::bilinear_point`])
//!   and 4-tap bicubic kernel ([`interp::bicubic_point`]) with their full
//!   tap tables ([`interp::BILINEAR_LUMA_FILTERS`],
//!   [`interp::BILINEAR_CHROMA_FILTERS`], [`interp::BICUBIC_FILTER_SET`]),
//!   their separable two-pass 8x8 block applicators
//!   ([`interp::bilinear_block`] / [`interp::bicubic_block`]), and the
//!   §11.4 `Var16Point` prediction-block variance metric
//!   ([`interp::var_16_point`]) used by the Advanced-Profile filter
//!   selector. These produce the interpolated sub-pixel prediction
//!   samples that §17.4 reconstruction consumes. Given a reference buffer,
//!   stride and fractional phase the kernels are pure integer pixel
//!   arithmetic — they never call `VP6_DecodeBool`, so this stage (like
//!   §15, §16 and §17.1) advances the decoder without touching the
//!   contested §7.3 `Split` formula. The motion vector that *selects* the
//!   phase and source is BoolCoder-gated upstream; the filter-selection
//!   *size* threshold is also deferred on an undefined `MAX_MV_EXTENT`
//!   constant (see the [`interp`] module DOCS-GAP note).
//!
//! ## Round 6 surface
//!
//! * [`inter`] — the spec §17.2–§17.4 inter-block reconstruction stage.
//!   [`reconstruct_inter_block`] applies §17's shared recombination
//!   (`OutputValue = PredictionValue + PredictionError`, inclusive clip
//!   to `0..=255`) — one function for all three inter cases because the
//!   spec's recombination pseudocode is byte-identical across §17.2
//!   (zero vector), §17.3 (full-pixel vector) and §17.4 (fractional
//!   vector); the cases differ only in how the prediction block is
//!   sourced. [`fetch_prediction_block`] is the §17.2/§17.3 integer copy
//!   from a reference reconstruction buffer (zero MV = co-located,
//!   full-pixel MV = integer offset); §17.4 sources its prediction from
//!   the round-5 [`interp`] filters instead. [`MvShift`],
//!   [`whole_sample_aligned`], [`luma_frac`] and [`chroma_frac`]
//!   implement the §11.4 motion-vector decomposition (`MvX >> MvShift`
//!   whole part + low-bit fractional phase; `MvShift` is 2 for luma /
//!   3 for chroma). Like §15/§16/§17.1/§11.4 this stage reads **no
//!   BoolCoder bits** — given an already-decoded MV, a reference buffer
//!   and the IDCT residual, every step is pure integer pixel
//!   arithmetic — so it advances the decoder without touching the
//!   contested §7.3 `Split` formula. Note the inter path applies **no**
//!   `+128` intra level shift (§17.1 only): the prediction already
//!   carries the DC.
//!
//! ## Round 7 surface
//!
//! * [`loopfilter`] — the spec §11.3 prediction loop filter. Implements
//!   the 4-tap `(1, -3, 3, -1)` deblocking filter applied to prediction
//!   blocks that straddle 8x8 boundaries in the reference frame, with
//!   the quantizer-indexed `Bound()` soft-clip that preserves real
//!   reference-frame edges while smoothing block-boundary
//!   discontinuities. Surfaces the
//!   [`loopfilter::PREDICTION_LOOP_FILTER_LIMIT_VALUES`] 64-entry
//!   quantizer-indexed limit table, the
//!   [`loopfilter::boundary_x`] / [`loopfilter::boundary_y`] block-edge
//!   offset calculations (`(8 - (mV & 7)) & 7`), the
//!   [`loopfilter::bound`] soft-clip, the per-edge
//!   [`loopfilter::prediction_loop_filter_function`] applicator, and the
//!   [`loopfilter::filter_vertical_boundary`] /
//!   [`loopfilter::filter_horizontal_boundary`] 2-D wrappers. Per the
//!   spec the deringing variant is "not currently supported by the
//!   decoder" so only the deblocking filter is implemented; Simple
//!   Profile disables the loop filter entirely (caller-side gate). Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4 this stage reads **no BoolCoder
//!   bits** — given a whole-sample-aligned MV, a prediction buffer and
//!   the frame's `DctQMask`, every step is pure integer pixel arithmetic
//!   — so it advances the decoder without touching the contested §7.3
//!   `Split` formula.
//!
//! ## Round 8 surface
//!
//! * [`umv`] — the spec §11.5 Unrestricted Motion Vector (UMV) border
//!   extension. Surfaces the [`umv::UMV_BORDER_SIZE`] constant (48,
//!   per spec), [`umv::extended_stride`] / [`umv::extended_height`] /
//!   [`umv::origin_offset`] geometry helpers, the in-place
//!   [`umv::extend_border`] applicator, and the
//!   [`umv::build_extended_buffer`] convenience constructor that
//!   allocates a UMV-bordered buffer, copies the original image into
//!   the inner rectangle, and fills the borders. The extension is pure
//!   edge replication in the spec-mandated "first in x, then in y"
//!   order: every original-image row's 48 left-border samples take
//!   the row's leftmost-original-column value and the 48 right-border
//!   samples take the rightmost-original-column value; each of the 48
//!   top- and bottom-border rows is then a row-wide copy of the
//!   topmost / bottommost horizontally-extended row, which makes the
//!   four 48×48 corner quadrants uniform at the corresponding
//!   corner-pixel value of the original image. The result lets
//!   [`inter::fetch_prediction_block`] and the §11.4 interpolation
//!   filters fetch any sample position within `±48` of the image
//!   boundary as if the original image's edge samples extended
//!   indefinitely — the well-defined "clamp" semantics a UMV fetch
//!   needs. Like §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3 this stage
//!   reads **no BoolCoder bits** — it is pure pixel arithmetic on an
//!   already-reconstructed frame buffer — so it advances the decoder
//!   past round 7 without touching the contested §7.3 `Split`
//!   formula.
//!
//! ## Round 10 surface
//!
//! * [`modes`] — the spec §10 macroblock coding-mode static surface.
//!   Surfaces the ten Table 4 coding modes ([`modes::CodingMode`])
//!   and three Table 5 ProbabilitySituation indices
//!   ([`modes::ModeAvailability`]); the 12-entry [`modes::NEAR_MACROBLOCKS`]
//!   neighbour-offset table; the verbatim
//!   [`modes::VP6_BASELINE_XMITTED_PROBS`] `[3][20]` I-frame
//!   `probXmitted` initialiser; the verbatim [`modes::VP6_MODE_VQ`]
//!   `[3][16][20]` baseline-bank that `SetNewBaselineProbs` /
//!   `WhichVector` select from; and the pure-integer
//!   [`modes::build_mode_decision_tree`] (and per-node
//!   [`modes::mode_decision_tree_node_probability`]) transform that
//!   converts a `probXmitted[3][20]` table into the
//!   `ModeDecisionTree[3][10][9]` array §10's `VP6_DecodeMode`
//!   traversal consults. The companion
//!   [`modes::probability_mode_same`] /
//!   [`modes::build_probability_mode_same`] compute the root-node
//!   "Same As Last" probability the Figure 10 traversal's first read
//!   needs. The `VP6_DecodeMode` BoolCoder traversal itself stays
//!   deferred until the §7.3 DOCS-GAP is resolved — but every piece
//!   of static data and every pure-integer derivation it would
//!   consult is now landed. Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5/§12.1/§14 this
//!   module reads no BoolCoder bits.
//!
//! ## Round 12 surface
//!
//! * [`huffman`] — the spec §7.2 Huffman tree construction and
//!   traversal primitives (`HUFF_NODE`, `VP6_CreateHuffmanTree`,
//!   `VP6_HuffmanDecodeSymbol`). VP6 supports two entropy schemes (§7):
//!   the BoolCoder (§7.3) used in partition 1 for mode/MV decisions,
//!   and the Huffman coder (§7.2) used as an alternate DCT-token
//!   scheme when `UseHuffman` is set. The Huffman coder reads one
//!   whole raw bit per tree branch (`R(1)`; §3 nomenclature) rather
//!   than a sub-bit `B(prob)` BoolCoder bit, so it is **independent
//!   of the §7.3 `Split` formula DOCS-GAP**. Surfaces: [`huffman::HuffNode`]
//!   (the spec's `HUFF_NODE { Symbol, Prob, Left, Right }` with `-1`
//!   sentinels for internal-vs-leaf); [`huffman::create_huffman_tree`]
//!   (the verbatim §7.2.1 builder — `N-1` bottom-up merge rounds over
//!   a stable-sorted leaf list, root at index `2N-2`); [`huffman::decode_symbol`]
//!   (the verbatim §7.2 walk, parameterised over an external
//!   `FnMut() -> u8` raw-bit oracle so the byte-stream `R(1)` reader
//!   can land independently); plus [`huffman::tree_depth`] /
//!   [`huffman::codeword_for`] convenience helpers for inspecting
//!   the constructed tree. Like §15/§16/§17/§11/§12.1/§14/§10/§13
//!   this module reads **no BoolCoder bits** — every operation is
//!   pure integer arithmetic over the supplied probability vector —
//!   so it advances the decoder past round 11 without touching the
//!   contested §7.3 `Split` formula. The §13.3.3.2 AC zero-run
//!   probability conversion and the actual `R(1)` byte-stream reader
//!   stay deferred for later rounds.
//!
//! ## Round 9 surface
//!
//! * [`scan`] — the spec §12.1 default zig-zag scan order. Surfaces
//!   the 64-entry [`scan::DEFAULT_SCAN_ORDER`] table (Figure 14 /
//!   `default_dequant_table[64]`) that the decoder uses to convert
//!   tokens from zig-zag order back to raster order before §15
//!   inverse quantization and §16 inverse DCT, the inverse
//!   [`scan::DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG`] permutation for the
//!   encoder side, and the [`scan::zigzag_to_raster_block`] /
//!   [`scan::raster_to_zigzag_block`] block applicators. The §12.2
//!   per-frame *custom* scan-order updates are
//!   `ScanOrderUpdateFlag`-gated (BoolCoder-coded, see Table 17) and
//!   stay deferred until the §7.3 DOCS-GAP is resolved.
//! * [`dc_pred`] — the spec §14 DC coefficient prediction stage. A
//!   [`dc_pred::DcPredictionContext`] per plane holds the per-
//!   reference-frame "last decoded DC value" that §14 mandates, with
//!   `DcPredictionContext::new` / `reset_at_frame_start` applying the
//!   spec's per-frame zero seed ("At the beginning of each frame this
//!   last decoded DC value is set to zero for each prediction frame
//!   type"). [`dc_pred::DcPredictionContext::predict`] /
//!   [`dc_pred::DcPredictionContext::predict_and_record`] compute the
//!   §14 predictor for one block: the predictor table's four rows
//!   (neither neighbour → per-bucket last-DC seed; only left → L;
//!   only above → A; both → `(L + A + Sign(L+A)) / 2` truncated
//!   toward zero), with the same-reference-frame and intra-vs-inter
//!   neighbour-disqualification rules implemented inline.
//!   [`dc_pred::ReferenceBucket`] enumerates the three prediction
//!   frame types (Intra, InterLast, InterGolden) the spec
//!   distinguishes. Together with the §12.1 scan permutation this
//!   leaves the per-block DC reconstruction pipeline §15→§16→§17.1
//!   already-complete from prior rounds without depending on the §7.3
//!   `Split` formula: the `DcDelta` token whose value is added on top
//!   of the predictor is what the BoolCoder-gated §13.2 stage
//!   produces, but the predictor itself is BoolCoder-independent. Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5 this stage advances
//!   the decoder past round 8 without touching the contested §7.3
//!   `Split` formula.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

pub mod dc_pred;
pub mod dequant;
pub mod frame_header;
pub mod huffman;
pub mod idct;
pub mod inter;
pub mod interp;
pub mod loopfilter;
pub mod modes;
pub mod reconstruct;
pub mod scan;
pub mod tokens;
pub mod umv;

pub use dc_pred::{
    average_both_neighbours, sign as dc_sign, DcPredictionContext, Neighbour, ReferenceBucket,
};
pub use dequant::{DequantContext, AC_QUANTIZATION_TABLE, DC_QUANTIZATION_TABLE};
pub use frame_header::{CodingProfile, FrameType, Vp3Version, Vp6FrameHeader};
pub use huffman::{
    codeword_for, create_huffman_tree, decode_symbol, tree_depth, HuffNode, HuffmanError,
    INTERNAL_SYMBOL, NO_CHILD,
};
pub use idct::idct_block;
pub use inter::{
    chroma_frac, fetch_prediction_block, inter_block_to_pixels, luma_frac, reconstruct_inter_block,
    whole_sample_aligned, MvShift,
};
pub use interp::{
    bicubic_block, bicubic_point, bilinear_block, bilinear_point, var_16_point, BICUBIC_FILTER_SET,
    BICUBIC_VP61_INDEX, BILINEAR_CHROMA_FILTERS, BILINEAR_LUMA_FILTERS,
};
pub use loopfilter::{
    bound, boundary_x, boundary_y, filter_horizontal_boundary, filter_vertical_boundary,
    prediction_loop_filter_function, PREDICTION_LOOP_FILTER_LIMIT_VALUES,
};
pub use modes::{
    build_mode_decision_tree, build_probability_mode_same, mode_decision_tree_node_probability,
    probability_mode_same, CodingMode, ModeAvailability, ModeDecisionTree, ModeDecisionTreeRow,
    NEAR_MACROBLOCKS, NUM_CODING_MODES, NUM_MODE_DECISION_NODES, NUM_MODE_VQ_VECTORS,
    NUM_PROBABILITY_SITUATIONS, PROB_XMITTED_ROW_LEN, VP6_BASELINE_XMITTED_PROBS, VP6_MODE_VQ,
};
pub use reconstruct::{intra_block_to_pixels, reconstruct_intra_block};
pub use scan::{
    raster_to_zigzag_block, zigzag_to_raster_block, DEFAULT_SCAN_ORDER,
    DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG,
};
pub use tokens::{
    baseline_ac_probs, baseline_dc_probs, dc_probs_to_node_contexts,
    dct_token_bool_tree_to_huff_probs, DctToken, TreeNode, AC_UPDATE_PROBS, DC_NODE_EQS,
    NUM_AC_BANDS, NUM_AC_PREC_CONTEXTS, NUM_DCT_TOKENS, NUM_DC_CONTEXTS, NUM_DC_NODE_EQS,
    NUM_PLANES, NUM_TREE_NODES, VP6_DC_UPDATE_PROBS,
};
pub use umv::{
    build_extended_buffer, extend_border, extended_height, extended_stride, origin_offset,
    UMV_BORDER_SIZE,
};

/// Crate-local error type.
///
/// Currently narrow because the round-1 surface only does one thing
/// (parse the raw-bit frame-header prefix); future rounds will expand
/// this as the decoder/encoder come online.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input ran out before all required syntactic elements were
    /// consumed.
    Truncated,
    /// The requested code path isn't wired up in this round.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Truncated => f.write_str("oxideav-vp6: truncated VP6 bitstream"),
            Error::NotImplemented => {
                f.write_str("oxideav-vp6: requested code path not implemented in this round")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Codec registration entry-point.
///
/// The round-1 scaffold doesn't register a `Decoder` or `Encoder` —
/// the full decoder isn't operational yet. Once the BoolCoder
/// DOCS-GAP is closed and the remainder of the frame header parses,
/// the decoder shell will be registered here.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vp6", register);

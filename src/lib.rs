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

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

pub mod dequant;
pub mod frame_header;
pub mod idct;
pub mod inter;
pub mod interp;
pub mod loopfilter;
pub mod reconstruct;

pub use dequant::{DequantContext, AC_QUANTIZATION_TABLE, DC_QUANTIZATION_TABLE};
pub use frame_header::{CodingProfile, FrameType, Vp3Version, Vp6FrameHeader};
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
pub use reconstruct::{intra_block_to_pixels, reconstruct_intra_block};

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

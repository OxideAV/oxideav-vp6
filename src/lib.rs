//! # oxideav-vp6
//!
//! Pure-Rust On2 VP6 ("vp6f"/"VP60"/"VP61"/"VP62") video decoder for
//! the [`oxideav`](https://github.com/OxideAV/oxideav) framework.
//!
//! **Status:** clean-room rebuild in progress (orphan-rebuild scaffold
//! from 2026-05-18; round 1 below). The crate previously contained a
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
//! What this round lands:
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
//! What this round deliberately does **not** land:
//!
//! * Any field downstream of the BoolCoder switch in the frame header
//!   (`VFragments`/`HFragments`/scaling/filter selectors/`UseHuffman`).
//! * The BoolCoder primitive itself.
//!
//! See the `DOCS-GAP` section below for the spec ambiguity blocking
//! that step.
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
//! We deliberately **do not** "guess" the fix per the workspace
//! "ask for docs, don't fish" rule. Round 2 is blocked on a docs
//! patch clarifying the Split formula — either confirming `>> 7` is
//! correct (and explaining the encoder-side mapping that makes it
//! work) or correcting it to `>> 8`.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

pub mod frame_header;

pub use frame_header::{CodingProfile, FrameType, Vp3Version, Vp6FrameHeader};

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

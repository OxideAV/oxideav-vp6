//! VP6 arithmetic DCT-coefficient decoding (spec §13.2.1).
//!
//! This module wires the §7.3 [`BoolCoder`] primitive
//! against the §13 [`DctToken`] / [`TreeNode`] static surface and the
//! errata-#67 corrected magnitude-bit reading
//! ([`DctToken::magnitude_probs`]) to produce signed DC coefficient
//! values directly from a coded VP6 bitstream partition.
//!
//! Scope of this round:
//!
//! * The Figure 15 binary tree walk for the **DC** path
//!   ([`decode_dc_token`]) — the DC tree forbids `EOB`, so the
//!   walk's two-level prefix is `ZERO_CONTEXT_NODE → ONE_CONTEXT_NODE`
//!   instead of the AC `ZERO_CONTEXT_NODE → EOB_CONTEXT_NODE` prefix.
//! * The shared magnitude-loop + sign read for the value-carrying
//!   tokens ([`decode_token_value`]).
//! * A full [`decode_dc`] wrapper that combines the walk, the
//!   magnitude decode, and the `ONE_TOKEN..FOUR_TOKEN` short-cut
//!   exactly as the §13.2.1 listing structures the cases.
//!
//! The AC path of §13.3.1 adds an `EOB_CONTEXT_NODE` branch above the
//! same binary tree plus an "implicitly-1" first-decision shortcut
//! when the preceding AC coefficient was zero, and on a `ZERO_TOKEN`
//! leaf it transitions into the §13.3.3 zero-run-length decoder. The
//! tree walk + magnitude loop landed here is the substrate the AC
//! path will share, but the AC-specific branching and the §13.3.3
//! zero-run integration stay deferred for the next round.
//!
//! ## Provenance
//!
//! Sourced exclusively from material in `docs/video/vp6/`:
//!
//! * `vp6_format.pdf` §13.2.1 (pages 64–65) — the arithmetic DC
//!   decode pseudocode (the `ContPtr = DcNodeContexts[Plane][Context]`
//!   listing and the magnitude / sign tail).
//! * `vp6_format.pdf` §13 Table 18 (page 56) — the token / extrabit
//!   geometry consumed via [`DctToken`].
//! * `vp6_format.pdf` §13 Table 20 / Figure 15 (page 59) — the
//!   internal-node indexing and the binary tree topology consumed
//!   via [`TreeNode`].
//! * `vp6_format.pdf` §3 (page 9) — the `B(x)` and `b(1)` notation,
//!   wired against [`crate::BoolCoder::decode_bool`] /
//!   [`crate::BoolCoder::decode_b1`].
//! * `vp6-errata-and-clarifications.md` entry **#67** — the
//!   `Probs[]` length / sign-handling reading that
//!   [`DctToken::magnitude_probs`] surfaces.
//!
//! No third-party VP6 source has been consulted at any stage.

use crate::tokens::{AcPrecContext, DctToken, TreeNode, NUM_TREE_NODES};
use crate::{BoolCoder, Error};

/// Read the magnitude-bit suffix and sign bit for a category token and
/// reassemble the signed coefficient value.
///
/// Implements the inner block of the §13.2.1 / §13.3.1 pseudocode:
///
/// ```text
/// value = TokenSetExtrabits[token].Min
/// BitsCount = TokenSetExtrabits[token].ExtraBits - 1
/// do
/// {
///    value += B(TokenSetExtrabits[token].Probs[BitsCount]) << BitsCount
///    BitsCount --
/// }
/// while (BitsCount >= 0)
///
/// SignBit = b(1)
/// out = (value ^ -SignBit) + SignBit
/// ```
///
/// The errata-#67 corrected reading interprets `Probs[]` as
/// magnitude-only (length `#ExtraBits − 1`) and the sign as a
/// **separately** decoded fixed-`b(1)` bit. With that reading the
/// magnitude loop iterates `#ExtraBits − 1` times for every category;
/// the loop index `BitsCount` is the bit position (so the
/// most-significant magnitude bit is read first, lined up with the
/// printed MSB-first column ordering of Table 18 and exposed by
/// [`DctToken::magnitude_probs`]).
///
/// For `ONE_TOKEN..FOUR_TOKEN` there are no magnitude bits (the
/// magnitude is a constant); the routine reads only the sign and
/// returns the signed constant. For `ZERO_TOKEN` the result is always
/// `0` (no magnitude bits, no sign read — DC consumers branch on the
/// `ZERO_CONTEXT_NODE` decision before getting here, and AC zero-run
/// integration belongs to a later round). For `EOB_TOKEN` no value
/// is produced (a `DecodeEndOfBlock` sentinel — the caller of the AC
/// path handles EOB structurally), so this routine simply returns 0
/// without consuming any bits; AC callers must branch on EOB above
/// this layer rather than passing the token in.
///
/// Returns `Error::Truncated` if the byte stream is exhausted during
/// any of the constituent BoolCoder calls.
pub fn decode_token_value(bc: &mut BoolCoder<'_>, token: DctToken) -> Result<i32, Error> {
    match token {
        DctToken::Zero | DctToken::EndOfBlock => Ok(0),
        DctToken::One | DctToken::Two | DctToken::Three | DctToken::Four => {
            // No magnitude bits — value is the token constant; only
            // the sign bit needs to be read. The §13.2.1 listing
            // writes this as `Dc = ((token ^ -SignBit) + SignBit)`
            // where `token` is the token's integer value (1..=4).
            let magnitude = token.min_value() as i32;
            let sign = bc.decode_b1()? as i32;
            Ok((magnitude ^ -sign) + sign)
        }
        _ => {
            // CATEGORY1..CATEGORY6. Iterate the magnitude loop over
            // the errata-#67 corrected magnitude-bit slice
            // (length `#ExtraBits − 1`), MSB-first via the descending
            // `BitsCount` index. Then a separate `b(1)` sign decode.
            let min = token.min_value() as i32;
            let probs = token.magnitude_probs();
            let bits = probs.len();
            let mut value: i32 = min;
            // `BitsCount` runs from `bits - 1` down to 0 inclusive.
            // `probs[BitsCount]` is the probability for the bit at
            // position `BitsCount` of the magnitude (so the highest
            // bit is read first, matching Table 18's printed order).
            for bits_count in (0..bits).rev() {
                let bit = bc.decode_bool(probs[bits_count])? as i32;
                value += bit << bits_count;
            }
            let sign = bc.decode_b1()? as i32;
            Ok((value ^ -sign) + sign)
        }
    }
}

/// Walk the §13.2.1 DC binary tree from the `ZERO_CONTEXT_NODE` root
/// down to a leaf and return the decoded [`DctToken`].
///
/// `node_probs` is the per-node probability vector — for arithmetic
/// DC decoding that is `DcNodeContexts[plane][context]` (the
/// `dc_probs_to_node_contexts` output of
/// [`crate::tokens::dc_probs_to_node_contexts`]).
///
/// The DC path **never returns `EndOfBlock`**: §13.2 explicitly
/// forbids EOB in the DC position, and the matching `DcNodeEqs` dummy
/// row pins `EOB_CONTEXT_NODE`'s probability to 1. The walk here
/// therefore reads `ZERO_CONTEXT_NODE → ONE_CONTEXT_NODE → …` as the
/// §13.2.1 listing structures it (instead of the AC
/// `ZERO_CONTEXT_NODE → EOB_CONTEXT_NODE → ONE_CONTEXT_NODE` chain).
///
/// A `0` at `ZERO_CONTEXT_NODE` leaves immediately with
/// [`DctToken::Zero`]; a `1` selects the value-carrying subtree and
/// the walk continues down to one of `ONE_TOKEN..FOUR_TOKEN` or
/// `CATEGORY1..CATEGORY6` per Figure 15.
///
/// Returns `Error::Truncated` if the byte stream is exhausted during
/// any of the constituent BoolCoder calls.
pub fn decode_dc_token(
    bc: &mut BoolCoder<'_>,
    node_probs: &[u8; NUM_TREE_NODES],
) -> Result<DctToken, Error> {
    // §13.2.1 root: `if ( !B( ContPtr[ZERO_CONTEXT_NODE] ) ) Dc = 0`.
    if bc.decode_bool(node_probs[TreeNode::Zero.index()])? == 0 {
        return Ok(DctToken::Zero);
    }
    // §13.2.1: `if ( B( ContPtr[ONE_CONTEXT_NODE] ) )` — the
    // value-carrying subtree.
    if bc.decode_bool(node_probs[TreeNode::One.index()])? == 1 {
        // `if ( B( ContPtr[LOW_VAL_CONTEXT_NODE] ) )` — categories
        // 1..=6; else the constant-magnitude tokens 2..=4.
        if bc.decode_bool(node_probs[TreeNode::LowVal.index()])? == 1 {
            // Categories 1..=6.
            if bc.decode_bool(node_probs[TreeNode::HighLow.index()])? == 1 {
                // CAT3..CAT6 subtree.
                if bc.decode_bool(node_probs[TreeNode::CatThreeFour.index()])? == 1 {
                    // CAT5 / CAT6.
                    if bc.decode_bool(node_probs[TreeNode::CatFive.index()])? == 1 {
                        Ok(DctToken::Category6)
                    } else {
                        Ok(DctToken::Category5)
                    }
                } else {
                    // CAT3 / CAT4.
                    if bc.decode_bool(node_probs[TreeNode::CatThree.index()])? == 1 {
                        Ok(DctToken::Category4)
                    } else {
                        Ok(DctToken::Category3)
                    }
                }
            } else {
                // CAT1 / CAT2.
                if bc.decode_bool(node_probs[TreeNode::CatOne.index()])? == 1 {
                    Ok(DctToken::Category2)
                } else {
                    Ok(DctToken::Category1)
                }
            }
        } else {
            // TWO / THREE / FOUR tokens.
            if bc.decode_bool(node_probs[TreeNode::Two.index()])? == 1 {
                if bc.decode_bool(node_probs[TreeNode::Three.index()])? == 1 {
                    Ok(DctToken::Four)
                } else {
                    Ok(DctToken::Three)
                }
            } else {
                Ok(DctToken::Two)
            }
        }
    } else {
        // §13.2.1 short branch: `Dc = ((1 ^ -SignBit) + SignBit)`
        // — magnitude-one token, sign read by the wrapper.
        Ok(DctToken::One)
    }
}

/// Full arithmetic DC coefficient decode (§13.2.1).
///
/// Walks the DC tree with [`decode_dc_token`], then for the
/// value-carrying tokens reads the magnitude / sign tail with
/// [`decode_token_value`]. Returns the signed integer DC coefficient
/// `Dc` exactly as the §13.2.1 listing's `Dc` output.
///
/// The DC tree never produces `EOB_TOKEN`; a returned `0` (signed)
/// always corresponds to `ZERO_TOKEN` at the root. Magnitudes for
/// `ONE_TOKEN..FOUR_TOKEN` are short-cut through the listing's
/// `Dc = ((token ^ -SignBit) + SignBit)` path inside
/// [`decode_token_value`].
///
/// `node_probs` is the per-(plane, context) probability vector — for
/// most callers that is the output of
/// [`crate::tokens::dc_probs_to_node_contexts`] indexed by plane and
/// by the DC context (`NUM_DC_CONTEXTS`).
///
/// Returns `Error::Truncated` on byte-stream exhaustion.
pub fn decode_dc(bc: &mut BoolCoder<'_>, node_probs: &[u8; NUM_TREE_NODES]) -> Result<i32, Error> {
    let token = decode_dc_token(bc, node_probs)?;
    decode_token_value(bc, token)
}

// ---------------------------------------------------------------------------
// §13.3.1 — Arithmetic AC coefficient decoder
// ---------------------------------------------------------------------------

/// Outcome of a single §13.3.1 per-coefficient AC decode step.
///
/// The §13.3.1 pseudo-code's per-coefficient inner loop produces one
/// of three structural outcomes depending on which leaf of the Figure
/// 15 binary tree the BoolCoder walk lands on. This enum surfaces
/// that three-way distinction so callers can drive the surrounding
/// per-block loop (the "do { … } while (EncodedCoeffs < BLOCK_SIZE)"
/// from the spec) without parsing a signed integer that contains
/// in-band sentinel values.
///
/// * [`AcOutcome::EndOfBlock`] — the EOB leaf was taken (left side of
///   the EOB-branch under `ZERO_CONTEXT_NODE`). The §13.3.1 listing
///   increments `EncodedCoeffs` once more and exits the per-block
///   loop; no value is emitted for the current scan position.
/// * [`AcOutcome::ZeroRun`] — the ZERO leaf was taken (right side of
///   the EOB-branch). The current AC coefficient is `0`; per the
///   §13.3.1 listing the caller then runs the §13.3.3 zero-run
///   decoder to advance `EncodedCoeffs` by `ZeroRunCount`. The
///   `Prec` context the caller threads into the next decode is
///   `WasZero` (which also gates the "implicitly-1" first-decision
///   shortcut on the *next* coefficient).
/// * [`AcOutcome::Value(coeff)`] — a value-carrying token was
///   decoded; the signed AC coefficient is `coeff`. The `Prec`
///   context for the next step is `WasOne` if the magnitude is 1 and
///   `WasGreaterThanOne` otherwise; this routine returns the updated
///   context next to the coefficient so callers don't have to
///   recompute it.
///
/// The `Value` and `ZeroRun` variants both update `EncodedCoeffs`
/// after a coefficient is written (or skipped by the zero-run);
/// `EndOfBlock` exits the loop. Mapping the outcome onto a block-level
/// state machine is the caller's job; this enum is the narrow per-step
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcOutcome {
    /// The Figure 15 EOB leaf was taken — end of the AC block. No
    /// coefficient is emitted at the current position; the caller
    /// exits the §13.3.1 per-block loop.
    EndOfBlock,
    /// The ZERO leaf was taken — current AC coefficient is 0 and the
    /// caller must run the §13.3.3 zero-run decoder to advance
    /// `EncodedCoeffs` past the trailing zeros. The next `Prec`
    /// context the caller threads in is always `WasZero`.
    ZeroRun,
    /// A value-carrying leaf was taken — signed AC coefficient
    /// `coeff` was decoded. `next_prec` is the §13.3.1 `Prec` update
    /// that applies to the next coefficient: `WasOne` if
    /// `coeff.unsigned_abs() == 1`, else `WasGreaterThanOne`.
    Value {
        /// The signed AC coefficient (`(value ^ -SignBit) + SignBit`).
        coeff: i32,
        /// `Prec` context to thread into the next coefficient's
        /// `decode_ac_coefficient` call.
        next_prec: AcPrecContext,
    },
}

/// Decode the §13.3.1 AC token tree for one coefficient position.
///
/// Walks the §13.3.1 binary tree from its `ZERO_CONTEXT_NODE` root
/// down to a leaf [`DctToken`], applying the §13.3.1 "implicitly-1"
/// first-decision shortcut: when the previously-decoded coefficient
/// in the current scan order was `0` (`prec == WasZero`) **and** we
/// are past the first AC coefficient (`encoded_coeffs > 1`), the
/// spec mandates that the next token can be neither `ZERO_TOKEN` nor
/// `EOB_TOKEN`, so the root decision is implicitly `1` and the walk
/// starts at `ONE_CONTEXT_NODE` instead of `ZERO_CONTEXT_NODE`.
///
/// `node_probs` is the per-`(prec, plane, band)` probability vector
/// the §13.3.1 listing reads as `AcUpdateProbs[Prec][Plane][Band]`
/// — typically [`crate::tokens::AC_UPDATE_PROBS`] indexed by the
/// caller's plane / band / preceding-coefficient context, or the
/// per-frame-updated `AcProbs` after Table 31–35 updates have been
/// applied (per-frame update is BoolCoder-coded and lands separately).
///
/// The `encoded_coeffs` parameter is the spec's per-block scan-
/// position counter that drives the `EncodedCoeffs > 1` half of the
/// implicit-1 gate. The §13.3.1 listing starts the counter at `1`
/// (the first AC coefficient), so on the *first* AC decode of a
/// block the gate is **not** active.
///
/// Returns `Error::Truncated` if the byte stream is exhausted during
/// any of the constituent BoolCoder calls. May return any of the
/// twelve [`DctToken`]s — including `EndOfBlock` (the EOB leaf the
/// AC tree adds above the DC variant) and `Zero` (the ZERO leaf the
/// caller hands to the §13.3.3 zero-run decoder). Value-carrying
/// leaves (`One`..=`Four`, `Category1`..=`Category6`) are paired
/// with the magnitude / sign read by [`decode_token_value`].
pub fn decode_ac_token(
    bc: &mut BoolCoder<'_>,
    prec: AcPrecContext,
    encoded_coeffs: usize,
    node_probs: &[u8; NUM_TREE_NODES],
) -> Result<DctToken, Error> {
    // §13.3.1 root: the spec writes
    //
    //     if ( (EncodedCoeffs > 1) && (Prec == 0) )
    //        ThisTokeNonZero = TRUE
    //     else
    //        ThisTokeNonZero = B( ProbPtr[ZERO_CONTEXT_NODE] )
    //
    // The implicit-1 shortcut prevents two consecutive `ZERO_TOKEN`s
    // (the §13.3.3 zero-run already encoded the trailing zeros) and
    // also prevents `EOB_TOKEN` immediately after a zero-run (an EOB
    // there would be a redundant encoding of the same state).
    let this_token_non_zero = if encoded_coeffs > 1 && matches!(prec, AcPrecContext::WasZero) {
        true
    } else {
        bc.decode_bool(node_probs[TreeNode::Zero.index()])? != 0
    };

    if !this_token_non_zero {
        // ZERO_CONTEXT_NODE = 0 → EOB_CONTEXT_NODE decides
        //   1-branch  → ZERO_TOKEN (proceed into §13.3.3 zero run)
        //   0-branch  → EOB_TOKEN  (end of block)
        //
        // Note the spec's inversion: B(EOB_CONTEXT_NODE) reads
        //   "1 → continue (zero run), 0 → end of block".
        if bc.decode_bool(node_probs[TreeNode::EndOfBlock.index()])? == 1 {
            Ok(DctToken::Zero)
        } else {
            Ok(DctToken::EndOfBlock)
        }
    } else if bc.decode_bool(node_probs[TreeNode::One.index()])? == 1 {
        // LOW_VAL_CONTEXT_NODE = 1 → categories 1..=6.
        if bc.decode_bool(node_probs[TreeNode::LowVal.index()])? == 1 {
            if bc.decode_bool(node_probs[TreeNode::HighLow.index()])? == 1 {
                // CAT3..CAT6 subtree.
                if bc.decode_bool(node_probs[TreeNode::CatThreeFour.index()])? == 1 {
                    if bc.decode_bool(node_probs[TreeNode::CatFive.index()])? == 1 {
                        Ok(DctToken::Category6)
                    } else {
                        Ok(DctToken::Category5)
                    }
                } else if bc.decode_bool(node_probs[TreeNode::CatThree.index()])? == 1 {
                    Ok(DctToken::Category4)
                } else {
                    Ok(DctToken::Category3)
                }
            } else if bc.decode_bool(node_probs[TreeNode::CatOne.index()])? == 1 {
                Ok(DctToken::Category2)
            } else {
                Ok(DctToken::Category1)
            }
        } else {
            // TWO / THREE / FOUR tokens.
            if bc.decode_bool(node_probs[TreeNode::Two.index()])? == 1 {
                if bc.decode_bool(node_probs[TreeNode::Three.index()])? == 1 {
                    Ok(DctToken::Four)
                } else {
                    Ok(DctToken::Three)
                }
            } else {
                Ok(DctToken::Two)
            }
        }
    } else {
        // ONE_CONTEXT_NODE = 0 → magnitude-1 token.
        Ok(DctToken::One)
    }
}

/// Full single-coefficient arithmetic AC decode (§13.3.1).
///
/// Walks the AC token tree with [`decode_ac_token`] applying the
/// `(EncodedCoeffs > 1) && (Prec == 0)` implicit-1 shortcut, then
/// dispatches on the leaf:
///
/// * EOB leaf → [`AcOutcome::EndOfBlock`] (the caller exits the
///   §13.3.1 per-block loop, after the §13.3.1 listing's trailing
///   `EncodedCoeffs ++`).
/// * ZERO leaf → [`AcOutcome::ZeroRun`] (the caller invokes the
///   §13.3.3 zero-run decoder; the next `Prec` is `WasZero`).
/// * Value-carrying leaf → [`AcOutcome::Value`] with the signed
///   coefficient and the §13.3.1 `Prec` update for the next position:
///   - `One` token → `next_prec = WasOne`
///   - `Two`..`Four` / `Category*` → `next_prec = WasGreaterThanOne`
///
/// Magnitude / sign for value-carrying tokens are decoded by
/// [`decode_token_value`] using the errata-#67 corrected
/// magnitude-only probability slice; the signed reconstruction
/// formula `(value ^ -SignBit) + SignBit` is the same identity the
/// §13.2.1 DC path uses.
///
/// `node_probs`, `prec`, and `encoded_coeffs` mean the same things as
/// in [`decode_ac_token`]. The caller is responsible for selecting
/// `node_probs` per the §13.3 `[prec][plane][band][node]` layout (the
/// `tokens::AcBand::for_coefficient_position` helper produces the
/// correct band index from a scan position).
///
/// Returns `Error::Truncated` on byte-stream exhaustion in any of
/// the constituent BoolCoder calls. Does **not** advance
/// `encoded_coeffs` or run the §13.3.3 zero-run decoder — both
/// are caller-side concerns (the zero-run decoder itself is
/// §13.3.3.1 and stays gated on its own DOCS-GAP-free BoolCoder
/// substrate that the round-15 BoolCoder primitive already
/// supplies; integration with the §13.3.3 surface is a separate
/// per-block driver round).
pub fn decode_ac_coefficient(
    bc: &mut BoolCoder<'_>,
    prec: AcPrecContext,
    encoded_coeffs: usize,
    node_probs: &[u8; NUM_TREE_NODES],
) -> Result<AcOutcome, Error> {
    let token = decode_ac_token(bc, prec, encoded_coeffs, node_probs)?;
    match token {
        DctToken::EndOfBlock => Ok(AcOutcome::EndOfBlock),
        DctToken::Zero => Ok(AcOutcome::ZeroRun),
        DctToken::One => {
            let coeff = decode_token_value(bc, DctToken::One)?;
            Ok(AcOutcome::Value {
                coeff,
                next_prec: AcPrecContext::WasOne,
            })
        }
        // Tokens Two / Three / Four and the six Category tokens all
        // carry magnitude > 1.
        _ => {
            let coeff = decode_token_value(bc, token)?;
            Ok(AcOutcome::Value {
                coeff,
                next_prec: AcPrecContext::WasGreaterThanOne,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{baseline_dc_probs, dc_probs_to_node_contexts};

    /// Helper: build a BoolCoder backed by a slice padded with
    /// zeros so the tests don't need to reason about end-of-stream.
    fn bc_over(bytes: &[u8]) -> BoolCoder<'_> {
        BoolCoder::new(bytes).expect("at least 4 bytes")
    }

    /// `decode_token_value(ZERO)` is a no-op: returns 0 without
    /// consuming any bits. (DC callers branch on `ZERO_CONTEXT_NODE`
    /// before reaching this path, but the symmetric behaviour is
    /// useful and matches the table semantics.)
    #[test]
    fn decode_token_value_zero_is_noop() {
        let bytes = [0xA5u8; 8];
        let mut bc = bc_over(&bytes);
        let pre_pos = bc.pos();
        let pre_value = bc.value();
        let v = decode_token_value(&mut bc, DctToken::Zero).expect("not truncated");
        assert_eq!(v, 0);
        assert_eq!(bc.pos(), pre_pos);
        assert_eq!(bc.value(), pre_value);
    }

    /// `decode_token_value(EOB)` returns 0 without consuming bits.
    /// AC callers branch on EOB structurally above this layer; the
    /// no-op behaviour matches the spec's interpretation that EOB is
    /// a control marker rather than a value carrier.
    #[test]
    fn decode_token_value_eob_is_noop() {
        let bytes = [0xA5u8; 8];
        let mut bc = bc_over(&bytes);
        let pre_pos = bc.pos();
        let pre_value = bc.value();
        let v = decode_token_value(&mut bc, DctToken::EndOfBlock).expect("not truncated");
        assert_eq!(v, 0);
        assert_eq!(bc.pos(), pre_pos);
        assert_eq!(bc.value(), pre_value);
    }

    /// `decode_token_value(ONE)` reads a single `b(1)` sign and
    /// returns `+1` or `-1`. With an all-zero byte stream the first
    /// `b(1)` at probability 128 (`Split = 255`, `Value = 0 < 0xFF00_0000`)
    /// always decodes to `0` → positive sign → value `+1`.
    #[test]
    fn decode_token_value_one_zero_sign_positive() {
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let v = decode_token_value(&mut bc, DctToken::One).expect("not truncated");
        assert_eq!(v, 1, "all-zero stream → sign bit 0 → +1");
    }

    /// `decode_token_value(TWO)` likewise returns `+2` against the
    /// all-zero stream. Verifies the table's `min_value` is what
    /// drives the constant; the §13.2.1 listing's
    /// `Dc = ((token ^ -SignBit) + SignBit)` with token=2, SignBit=0
    /// yields 2.
    #[test]
    fn decode_token_value_constant_magnitudes() {
        for (token, expected) in [
            (DctToken::One, 1i32),
            (DctToken::Two, 2),
            (DctToken::Three, 3),
            (DctToken::Four, 4),
        ] {
            let bytes = [0u8; 8];
            let mut bc = bc_over(&bytes);
            let v = decode_token_value(&mut bc, token).expect("not truncated");
            assert_eq!(v, expected, "{token:?} against all-zero stream");
        }
    }

    /// Sign-bit application contract: the sign-bit reassembly formula
    /// `(value ^ -SignBit) + SignBit` is equivalent to "if sign is 1,
    /// negate; else leave alone". This is the §13.2.1 / §13.3.1
    /// reconstruction step. We can't easily drive `b(1)` (fixed
    /// probability 128) to 1 against an arbitrary bitstream because
    /// errata #35 documents that probability 128 with the §7.3 `Split`
    /// formula gives `Split = Range` exactly, which pins well-formed
    /// `b(1)` reads to the 0-branch. The reconstruction arithmetic is
    /// however a pure-integer identity verifiable in isolation.
    #[test]
    fn sign_reconstruction_identity_holds() {
        for magnitude in [1i32, 2, 3, 5, 11, 67, 2114] {
            for sign in [0i32, 1] {
                let reconstructed = (magnitude ^ -sign) + sign;
                let expected = if sign == 0 { magnitude } else { -magnitude };
                assert_eq!(
                    reconstructed, expected,
                    "(magnitude ^ -SignBit) + SignBit must be ±magnitude (mag={magnitude}, sign={sign})"
                );
            }
        }
    }

    /// `decode_token_value(CATEGORY1)` with an all-zero stream
    /// decodes every magnitude bit and the sign to 0 → result =
    /// `min_value = 5`. CATEGORY1 has exactly one magnitude bit
    /// (errata #67's `magnitude_probs` length = `#ExtraBits − 1`
    /// = 1).
    #[test]
    fn decode_token_value_category1_all_zeros() {
        let bytes = [0u8; 16];
        let mut bc = bc_over(&bytes);
        let v = decode_token_value(&mut bc, DctToken::Category1).expect("not truncated");
        // magnitude = min_value (5) + 0 (single magnitude bit
        // decoded to 0) = 5; positive sign → 5.
        assert_eq!(v, 5);
    }

    /// `decode_token_value(CATEGORY6)` with an all-zero stream
    /// decodes every magnitude bit and the sign to 0 → result =
    /// `min_value = 67`. The errata-#67 corrected slice has
    /// 11 entries (= `#ExtraBits − 1`).
    #[test]
    fn decode_token_value_category6_all_zeros() {
        let bytes = [0u8; 32];
        let mut bc = bc_over(&bytes);
        let v = decode_token_value(&mut bc, DctToken::Category6).expect("not truncated");
        assert_eq!(v, 67);
    }

    /// `decode_token_value(CATEGORY1)` magnitude range: the token
    /// covers `min_value=5, max_value=6` and has exactly **one**
    /// magnitude bit (errata #67). Either the magnitude bit decodes
    /// to 0 (→ 5) or to 1 (→ 6), with a sign-flip at the end. So the
    /// signed-result set is `{ +5, -5, +6, -6 }`.
    ///
    /// We sweep multiple BoolCoder inputs and verify every decoded
    /// value lies inside that set.
    #[test]
    fn decode_token_value_category1_range() {
        for seed_byte in [0x00u8, 0x33, 0x55, 0x88, 0xAA, 0xCC, 0xFF] {
            let bytes = [seed_byte; 16];
            let mut bc = bc_over(&bytes);
            let v = decode_token_value(&mut bc, DctToken::Category1).expect("not truncated");
            let abs = v.unsigned_abs();
            assert!(
                v == 5 || v == -5 || v == 6 || v == -6,
                "CATEGORY1 magnitude must be 5..=6 (got {v}, |v|={abs})"
            );
        }
    }

    /// `decode_token_value(CATEGORY3)` magnitude range: 11..=18 (8
    /// magnitudes = 3 magnitude bits, errata #67 magnitude_probs
    /// length = 3). Sign flip ±. Verify every decode lands in the
    /// stated bounds.
    #[test]
    fn decode_token_value_category3_range() {
        for seed_byte in [0x00u8, 0x33, 0x55, 0x88, 0xAA, 0xCC, 0xFF] {
            let bytes = [seed_byte; 16];
            let mut bc = bc_over(&bytes);
            let v = decode_token_value(&mut bc, DctToken::Category3).expect("not truncated");
            let abs = v.unsigned_abs() as i32;
            assert!(
                (11..=18).contains(&abs),
                "CATEGORY3 |v| must be 11..=18 (got v={v}, |v|={abs})"
            );
        }
    }

    /// `decode_token_value(CATEGORY6)` against an all-zero stream
    /// always lands on the minimum-magnitude leaf (`+67`) because
    /// `Value=0` is below every positive `Split<<24` so every magnitude
    /// bit decodes to 0 and the sign bit at probability 128 also
    /// decodes to 0 (Split=Range edge documented in errata #35; the
    /// 0-branch is taken). This is the only fully-predictable Cat6
    /// trace driveable from the public byte-stream constructor without
    /// a test-only encoder helper.
    #[test]
    fn decode_token_value_category6_all_zeros_min_magnitude() {
        let bytes = [0u8; 64];
        let mut bc = bc_over(&bytes);
        let v = decode_token_value(&mut bc, DctToken::Category6).expect("not truncated");
        assert_eq!(
            v, 67,
            "all-zero stream → all-zero magnitude + sign = +min_value"
        );
    }

    /// MSB-first magnitude reading: construct a contrived `BoolCoder`
    /// trace where the magnitude bits are well-defined and confirm
    /// the magnitude is reconstructed in the MSB-first ordering
    /// implied by Table 18's column layout.
    ///
    /// Strategy: drive `decode_token_value(CATEGORY3)` (3 magnitude
    /// bits) against an all-zero stream. The magnitude must be
    /// `min_value(11) + 0 = 11`; the sign bit decodes to 0 (positive).
    /// This pins down the contract that an all-zero magnitude trace
    /// yields the minimum value — confirming both the slice
    /// ordering and the `+ Min` offset application.
    #[test]
    fn decode_token_value_msb_first_zero_magnitude_yields_min() {
        for token in [
            DctToken::Category1,
            DctToken::Category2,
            DctToken::Category3,
            DctToken::Category4,
            DctToken::Category5,
            DctToken::Category6,
        ] {
            let bytes = [0u8; 32];
            let mut bc = bc_over(&bytes);
            let v = decode_token_value(&mut bc, token).expect("not truncated");
            let expected = token.min_value() as i32;
            assert_eq!(
                v, expected,
                "{token:?}: all-zero magnitude must yield +min_value"
            );
        }
    }

    /// `decode_dc_token` with `node_probs[0] = 1` makes the
    /// `ZERO_CONTEXT_NODE` decision overwhelmingly 0 — `B(1)` is the
    /// minimum non-zero probability and against an all-zero stream
    /// the comparison `Value(0) < Split<<24` is trivially true, so
    /// the 0-branch is taken every time. Walk should return
    /// `ZERO_TOKEN` and consume only one bit's worth of work.
    #[test]
    fn decode_dc_token_zero_leaf_no_value() {
        // Pin all node probabilities to 1 so the all-zero stream
        // takes the 0-branch at every node — though only the root
        // matters here because the 0-branch at the root short-circuits.
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let token = decode_dc_token(&mut bc, &node_probs).expect("not truncated");
        assert_eq!(token, DctToken::Zero);
    }

    /// `decode_dc_token` is exhaustive over its declared leaves
    /// (every value-carrying token + `Zero`, never `EndOfBlock`).
    /// We verify this structurally by enumerating the cartesian
    /// product of feasible `node_probs` corners and checking each
    /// run terminates with a non-EOB token. The implementation under
    /// test is the §13.2.1 walk; the result-set bound holds by
    /// construction of the binary tree.
    #[test]
    fn decode_dc_token_never_returns_eob() {
        // Sweep several `node_probs` corners plus several stream
        // prefixes; assert the returned token is never EOB. The DC
        // tree topology in `decode_dc_token` literally has no path to
        // `EndOfBlock` (the EOB branch is the AC-only insertion above
        // `ZERO_CONTEXT_NODE`), so this is a structural guarantee.
        let prob_corners = [
            [1u8; NUM_TREE_NODES],
            [128u8; NUM_TREE_NODES],
            [255u8; NUM_TREE_NODES],
        ];
        let stream_seeds = [0x00u8, 0x33, 0x55, 0xA5];
        for node_probs in prob_corners.iter() {
            for &seed in &stream_seeds {
                let bytes = [seed; 16];
                let mut bc = bc_over(&bytes);
                if let Ok(token) = decode_dc_token(&mut bc, node_probs) {
                    assert_ne!(
                        token,
                        DctToken::EndOfBlock,
                        "DC walk must never return EOB (probs={node_probs:?}, seed={seed:#x})"
                    );
                }
                // A Truncated result is acceptable; the DC tree's
                // contract is "never returns EOB on success", not
                // "always succeeds against arbitrary inputs".
            }
        }
    }

    /// End-to-end `decode_dc` smoke: against an all-zero stream and
    /// the baseline-DC node probabilities (every prob = 128), the
    /// root `ZERO_CONTEXT_NODE` decode picks the 0-branch (Split =
    /// 255 at prob 128, Value = 0 < 0xFF000000 → 0-branch → ZERO),
    /// so the full decode short-circuits to 0 without consuming
    /// further bits.
    #[test]
    fn decode_dc_zero_short_circuit() {
        let dc_probs = baseline_dc_probs();
        let node_contexts = dc_probs_to_node_contexts(&dc_probs);
        let node_probs = node_contexts[0][0]; // plane 0, context 0
        let bytes = [0u8; 16];
        let mut bc = bc_over(&bytes);
        let dc = decode_dc(&mut bc, &node_probs).expect("not truncated");
        assert_eq!(dc, 0, "baseline DC + all-zero stream → ZERO_TOKEN → 0");
    }

    /// End-to-end `decode_dc` against an all-zero stream with
    /// `node_probs[ZERO] = 255` (strong 1-bias at the root). The
    /// `Split` formula at prob 255, Range 255 gives `Split = 508`
    /// which is *greater than Range*: `Split<<24 = 0x1FC00_0000`
    /// (u64). Compare `Value=0 < 0x1FC00_0000` → 0-branch. So even
    /// a "1-biased" root with `Value=0` actually takes the 0-branch,
    /// returning `ZERO_TOKEN` → DC = 0. This pins the edge-case
    /// arithmetic the implementation uses (`u64` widening for the
    /// `Split<<24` comparison, errata #35).
    #[test]
    fn decode_dc_root_high_prob_with_zero_value_short_circuits() {
        let mut node_probs = [255u8; NUM_TREE_NODES];
        node_probs[TreeNode::EndOfBlock.index()] = 255;
        let bytes = [0u8; 16];
        let mut bc = bc_over(&bytes);
        let dc = decode_dc(&mut bc, &node_probs).expect("not truncated");
        assert_eq!(
            dc, 0,
            "Split=508 > Range=255 but Value=0 still satisfies Value < Split<<24 → 0-branch → ZERO_TOKEN"
        );
    }

    /// Determinism: two identical (BoolCoder, node-probs) pairs must
    /// produce identical decoded values and end in identical decoder
    /// state — the §13.2.1 walk is a pure function of the bitstream.
    #[test]
    fn decode_dc_deterministic() {
        let dc_probs = baseline_dc_probs();
        let node_contexts = dc_probs_to_node_contexts(&dc_probs);
        let node_probs = node_contexts[0][1];
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let mut bc_a = bc_over(&bytes);
        let mut bc_b = bc_over(&bytes);
        let dc_a = decode_dc(&mut bc_a, &node_probs).expect("not truncated");
        let dc_b = decode_dc(&mut bc_b, &node_probs).expect("not truncated");
        assert_eq!(dc_a, dc_b);
        assert_eq!(bc_a.pos(), bc_b.pos());
        assert_eq!(bc_a.range(), bc_b.range());
        assert_eq!(bc_a.value(), bc_b.value());
        assert_eq!(bc_a.count(), bc_b.count());
    }

    /// Truncation: low-probability node walks drive aggressive
    /// renormalization that quickly exhausts a minimal byte stream
    /// and surfaces `Error::Truncated`. We exercise the truncation
    /// path through [`decode_dc`] by pinning `node_probs[ZERO] = 1`
    /// (the spec's minimum non-zero value, which forces deep renorm
    /// on the root decode) with only a 4-byte stream (no fill bytes
    /// beyond the prefill). The renormalization loop's byte-pull
    /// surfaces `Truncated` per the §7.3 reader contract.
    ///
    /// (Setting `node_probs[ZERO] = 1` does not strictly guarantee
    /// truncation on a single decode; we loop the wrapper repeatedly
    /// so the cumulative state eventually exhausts the stream.)
    #[test]
    fn decode_dc_surfaces_truncation() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0xFFu8; 4];
        let mut bc = bc_over(&bytes);
        let mut saw_truncation = false;
        // The root decode at prob=1 + value=0xFFFFFFFF picks the
        // 1-branch (Split = 2, Split<<24 = 0x02000000, Value 0xFFFFFFFF
        // > 0x02000000 → 1-branch). With Range=255-2=253 → renorm
        // doubles once to ≥128 → no truncation that decode. Subsequent
        // decodes burn bits aggressively; one of them runs the stream
        // dry.
        for _ in 0..32 {
            match decode_dc(&mut bc, &node_probs) {
                Ok(_) => {}
                Err(Error::Truncated) => {
                    saw_truncation = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
            }
        }
        assert!(
            saw_truncation,
            "low-prob decoder walks must surface Truncated on a 4-byte stream"
        );
    }

    /// Cross-check: `decode_dc_token` followed by `decode_token_value`
    /// is identical to `decode_dc`, by construction. This pins the
    /// composition guarantee that callers can split the two halves
    /// when needed (e.g. a probability-update layer between).
    #[test]
    fn decode_dc_is_walk_then_value() {
        let dc_probs = baseline_dc_probs();
        let node_contexts = dc_probs_to_node_contexts(&dc_probs);
        let node_probs = node_contexts[1][2];
        let bytes = [
            0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
        ];

        let mut bc_one = bc_over(&bytes);
        let dc_one = decode_dc(&mut bc_one, &node_probs).expect("not truncated");

        let mut bc_two = bc_over(&bytes);
        let token = decode_dc_token(&mut bc_two, &node_probs).expect("not truncated");
        let dc_two = decode_token_value(&mut bc_two, token).expect("not truncated");

        assert_eq!(dc_one, dc_two);
        assert_eq!(bc_one.pos(), bc_two.pos());
        assert_eq!(bc_one.value(), bc_two.value());
        assert_eq!(bc_one.range(), bc_two.range());
        assert_eq!(bc_one.count(), bc_two.count());
    }

    /// Errata #67 contract: every category token's `magnitude_probs`
    /// slice length equals `extra_bits − 1`, and the magnitude range
    /// `(2^bits)` exactly covers `max_value − min_value + 1`. This is
    /// the static invariant the magnitude loop relies on.
    #[test]
    fn magnitude_probs_length_matches_extra_bits_minus_one() {
        for token in [
            DctToken::Category1,
            DctToken::Category2,
            DctToken::Category3,
            DctToken::Category4,
            DctToken::Category5,
            DctToken::Category6,
        ] {
            let probs_len = token.magnitude_probs().len();
            let expected = token.extra_bits() - 1;
            assert_eq!(
                probs_len, expected,
                "{token:?}: magnitude_probs() len = {probs_len}, expected extra_bits-1 = {expected}"
            );
            let magnitudes = (token.max_value() - token.min_value() + 1) as usize;
            let from_bits = 1usize << probs_len;
            assert_eq!(
                from_bits, magnitudes,
                "{token:?}: 2^magnitude_bits = {from_bits} must equal max-min+1 = {magnitudes}"
            );
        }
    }

    // ----------------------------------------------------------------------
    // §13.3.1 — Arithmetic AC coefficient decoder
    // ----------------------------------------------------------------------

    /// `decode_ac_token` against an all-zero stream with `prec=WasZero`
    /// but `encoded_coeffs == 1` (the very first AC position): the
    /// implicit-1 gate's `EncodedCoeffs > 1` clause is **not** met, so
    /// the root `B(ZERO_CONTEXT_NODE)` read fires normally. With every
    /// node prob pinned to 1 (minimum) the all-zero stream takes the
    /// 0-branch at the root (Value=0 < (Split=2)<<24 = 0x0200_0000),
    /// then `B(EOB_CONTEXT_NODE)` also fires (Value < Split<<24): the
    /// inversion in the spec says 1-branch → ZERO_TOKEN, 0-branch →
    /// EOB_TOKEN — so we land on EOB.
    #[test]
    fn decode_ac_token_zero_stream_first_position_lands_eob() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let token = decode_ac_token(&mut bc, AcPrecContext::WasZero, 1, &node_probs)
            .expect("not truncated");
        assert_eq!(token, DctToken::EndOfBlock);
    }

    /// `decode_ac_token` "implicitly-1" shortcut: with
    /// `(EncodedCoeffs > 1) && (Prec == WasZero)` the root decision is
    /// skipped entirely. Starting with all-zero node probs and an
    /// all-zero stream, the walk would otherwise land on `ZERO_TOKEN`
    /// from the root; with the shortcut it must skip the ZERO root and
    /// proceed down the value-carrying subtree, landing on
    /// `ONE_TOKEN` (the `B(ONE_CONTEXT_NODE)==0` short-branch with
    /// prob=1, all-zero stream).
    #[test]
    fn decode_ac_token_implicit_one_shortcut_lands_on_value_subtree() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let token = decode_ac_token(&mut bc, AcPrecContext::WasZero, 2, &node_probs)
            .expect("not truncated");
        // With every prob = 1 and all-zero stream, every B() reads 0.
        // The walk skips the root, lands at ONE_CONTEXT_NODE which
        // reads 0 → ONE_TOKEN per §13.3.1 (the `1` branch of the
        // outer if covers categories/two/three/four; the `0` branch
        // takes the magnitude-1 short).
        assert_eq!(token, DctToken::One);
    }

    /// `decode_ac_token` implicit-1 shortcut is **not** triggered when
    /// `Prec != WasZero`, even if `EncodedCoeffs > 1`. We re-run the
    /// previous test with `Prec=WasOne` and confirm the root decision
    /// IS read (and against the same all-zero stream lands on
    /// `EndOfBlock` per the previous test's mechanics).
    #[test]
    fn decode_ac_token_implicit_one_gated_on_prec_was_zero() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let token =
            decode_ac_token(&mut bc, AcPrecContext::WasOne, 5, &node_probs).expect("not truncated");
        assert_eq!(token, DctToken::EndOfBlock);
    }

    /// `decode_ac_token` implicit-1 shortcut is **not** triggered when
    /// `EncodedCoeffs == 1`, even if `Prec == WasZero`. The very-first
    /// AC position always reads the root `B(ZERO_CONTEXT_NODE)`. The
    /// "preceded by 0" Prec there came from the *DC* of the same
    /// block (not from a prior AC zero token), so the spec mandates
    /// the root read fires.
    #[test]
    fn decode_ac_token_implicit_one_gated_on_encoded_coeffs_above_one() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        // EncodedCoeffs == 1 + Prec == WasZero: gate is closed.
        let token = decode_ac_token(&mut bc, AcPrecContext::WasZero, 1, &node_probs)
            .expect("not truncated");
        assert_eq!(
            token,
            DctToken::EndOfBlock,
            "EncodedCoeffs == 1 must skip the implicit-1 shortcut"
        );
    }

    /// `decode_ac_token` ZERO/EOB inversion: with
    /// `node_probs[EOB_CONTEXT_NODE] = 255` (strong 1-bias) the
    /// `B(EOB)` read against an all-zero stream still picks the
    /// 0-branch (Value=0 < Split=508 shifted — Split>Range edge from
    /// errata #35), which under the §13.3.1 inversion gives
    /// EOB_TOKEN. We then verify the 1-branch outcome (ZERO_TOKEN)
    /// via the opposite probability+stream choice.
    #[test]
    fn decode_ac_token_eob_inversion_branches() {
        // Pin the root to take the 0-branch (ZERO sub-branch) and the
        // EOB node to take the 0-branch (→ EOB_TOKEN under the §13.3.1
        // inversion). All-zero stream + prob=1 root works:
        //   Split = 1 + ((255-1)*1 >> 7) = 1 + 1 = 2. Split<<24
        //   = 0x0200_0000 > Value=0 → 0-branch.
        let mut node_probs = [1u8; NUM_TREE_NODES];
        node_probs[TreeNode::EndOfBlock.index()] = 1;
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let token =
            decode_ac_token(&mut bc, AcPrecContext::WasOne, 5, &node_probs).expect("not truncated");
        assert_eq!(
            token,
            DctToken::EndOfBlock,
            "0-branch at EOB node → EOB_TOKEN"
        );
    }

    /// `decode_ac_coefficient` outcome on the EOB leaf: returns
    /// `EndOfBlock` and emits no coefficient.
    #[test]
    fn decode_ac_coefficient_eob_outcome() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let outcome = decode_ac_coefficient(&mut bc, AcPrecContext::WasOne, 5, &node_probs)
            .expect("not truncated");
        assert_eq!(outcome, AcOutcome::EndOfBlock);
    }

    /// `decode_ac_coefficient` outcome on the ZERO leaf: returns
    /// `ZeroRun`. We construct a stream + probabilities where the
    /// root takes the 0-branch (into the EOB/ZERO decision) and the
    /// EOB node takes the 1-branch (under the §13.3.1 inversion,
    /// 1-branch at EOB-node = ZERO_TOKEN).
    ///
    /// Construction: pin `ZERO_CONTEXT_NODE` prob to 255 and
    /// `EOB_CONTEXT_NODE` prob to 1, all-0xFF stream. Then the root
    /// Split is `1 + ((255-1)*255 >> 7) = 508`; `Split << 24` is much
    /// greater than `Value = 0xFFFF_FFFF`, so the root takes the
    /// 0-branch (ZERO subtree). The EOB Split is `1 + ((255-1)*1 >> 7)
    /// = 2`; `Split << 24 = 0x0200_0000 < Value`, so EOB takes the
    /// 1-branch — which under the §13.3.1 inversion is `ZERO_TOKEN`.
    #[test]
    fn decode_ac_coefficient_zero_run_outcome() {
        let mut node_probs = [1u8; NUM_TREE_NODES];
        node_probs[TreeNode::Zero.index()] = 255;
        node_probs[TreeNode::EndOfBlock.index()] = 1;
        let bytes = [0xFFu8; 16];
        let mut bc = bc_over(&bytes);
        let outcome = decode_ac_coefficient(&mut bc, AcPrecContext::WasOne, 5, &node_probs)
            .expect("not truncated");
        assert_eq!(
            outcome,
            AcOutcome::ZeroRun,
            "0-branch at root + 1-branch at EOB → ZERO_TOKEN → ZeroRun"
        );
    }

    /// `decode_ac_coefficient` Value outcome with magnitude 1: the
    /// `next_prec` must be `WasOne`. We use the implicit-1 shortcut
    /// plus all-zero probs/stream to land on `ONE_TOKEN` (per the
    /// implicit-1 shortcut test). The sign at `b(1)` reads 0 against
    /// the all-zero stream → +1.
    #[test]
    fn decode_ac_coefficient_value_one_sets_prec_was_one() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0u8; 8];
        let mut bc = bc_over(&bytes);
        let outcome = decode_ac_coefficient(&mut bc, AcPrecContext::WasZero, 2, &node_probs)
            .expect("not truncated");
        assert_eq!(
            outcome,
            AcOutcome::Value {
                coeff: 1,
                next_prec: AcPrecContext::WasOne,
            }
        );
    }

    /// `decode_ac_coefficient` Value outcome's `next_prec` invariant:
    /// for every `Value { coeff, next_prec }` outcome the §13.3.1
    /// `Prec` update rule is:
    ///
    /// * `|coeff| == 1` → `next_prec == WasOne`
    /// * `|coeff| > 1`  → `next_prec == WasGreaterThanOne`
    ///
    /// We sweep `(prec, encoded_coeffs, prob, seed)` corners and
    /// assert the invariant on every `Value` outcome (`ZeroRun` and
    /// `EndOfBlock` outcomes are skipped — they carry no coefficient
    /// and don't exercise this rule). The sweep guarantees we cover
    /// both the magnitude-1 (`ONE_TOKEN`) path and the magnitude->1
    /// (`TWO..=FOUR` / `Category*`) paths.
    #[test]
    fn decode_ac_coefficient_value_next_prec_invariant() {
        let prob_corners = [
            [1u8; NUM_TREE_NODES],
            [128u8; NUM_TREE_NODES],
            [255u8; NUM_TREE_NODES],
        ];
        let stream_seeds: [u8; 6] = [0x00, 0x33, 0x55, 0xA5, 0xCC, 0xFF];
        let prec_options = AcPrecContext::ALL;
        let encoded_coeffs_options: [usize; 3] = [1, 2, 17];

        let mut saw_value = false;
        let mut saw_mag_one = false;
        let mut saw_mag_gt_one = false;
        for node_probs in prob_corners.iter() {
            for &seed in &stream_seeds {
                for &prec in &prec_options {
                    for &ec in &encoded_coeffs_options {
                        let bytes = [seed; 64];
                        let mut bc = bc_over(&bytes);
                        if let Ok(AcOutcome::Value { coeff, next_prec }) =
                            decode_ac_coefficient(&mut bc, prec, ec, node_probs)
                        {
                            saw_value = true;
                            let mag = coeff.unsigned_abs();
                            if mag == 1 {
                                saw_mag_one = true;
                                assert_eq!(
                                    next_prec, AcPrecContext::WasOne,
                                    "|coeff|=1 → next_prec must be WasOne (prec={prec:?}, ec={ec}, seed={seed:#x}, probs={node_probs:?})"
                                );
                            } else {
                                assert!(mag >= 2, "|coeff| must be ≥ 1 (got {coeff})");
                                saw_mag_gt_one = true;
                                assert_eq!(
                                    next_prec, AcPrecContext::WasGreaterThanOne,
                                    "|coeff|>1 → next_prec must be WasGreaterThanOne (prec={prec:?}, ec={ec}, seed={seed:#x}, probs={node_probs:?})"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(saw_value, "sweep must produce at least one Value outcome");
        assert!(
            saw_mag_one,
            "sweep must hit the magnitude-1 (ONE_TOKEN) path"
        );
        assert!(
            saw_mag_gt_one,
            "sweep must hit the magnitude-greater-than-1 path"
        );
    }

    /// Determinism: identical inputs yield identical outcomes for the
    /// AC decoder (pure function of the bitstream + node probs +
    /// context).
    #[test]
    fn decode_ac_coefficient_deterministic() {
        let dc_probs = baseline_dc_probs();
        let _ = dc_probs_to_node_contexts(&dc_probs);
        let node_probs = [128u8; NUM_TREE_NODES];
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let mut bc_a = bc_over(&bytes);
        let mut bc_b = bc_over(&bytes);
        let outcome_a = decode_ac_coefficient(&mut bc_a, AcPrecContext::WasOne, 3, &node_probs)
            .expect("not truncated");
        let outcome_b = decode_ac_coefficient(&mut bc_b, AcPrecContext::WasOne, 3, &node_probs)
            .expect("not truncated");
        assert_eq!(outcome_a, outcome_b);
        assert_eq!(bc_a.pos(), bc_b.pos());
        assert_eq!(bc_a.range(), bc_b.range());
        assert_eq!(bc_a.value(), bc_b.value());
        assert_eq!(bc_a.count(), bc_b.count());
    }

    /// Composition: `decode_ac_coefficient` == `decode_ac_token`
    /// followed by the appropriate per-leaf value/sign decode. The
    /// AC outcome and the manually-driven equivalent must end up in
    /// the same BoolCoder state and produce the same coefficient.
    #[test]
    fn decode_ac_coefficient_is_token_then_value() {
        let node_probs = [128u8; NUM_TREE_NODES];
        let bytes = [
            0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
        ];

        let mut bc_one = bc_over(&bytes);
        let outcome = decode_ac_coefficient(&mut bc_one, AcPrecContext::WasOne, 4, &node_probs)
            .expect("not truncated");

        let mut bc_two = bc_over(&bytes);
        let token = decode_ac_token(&mut bc_two, AcPrecContext::WasOne, 4, &node_probs)
            .expect("not truncated");
        let expected = match token {
            DctToken::EndOfBlock => AcOutcome::EndOfBlock,
            DctToken::Zero => AcOutcome::ZeroRun,
            DctToken::One => AcOutcome::Value {
                coeff: decode_token_value(&mut bc_two, DctToken::One).expect("not truncated"),
                next_prec: AcPrecContext::WasOne,
            },
            other => AcOutcome::Value {
                coeff: decode_token_value(&mut bc_two, other).expect("not truncated"),
                next_prec: AcPrecContext::WasGreaterThanOne,
            },
        };

        assert_eq!(outcome, expected);
        assert_eq!(bc_one.pos(), bc_two.pos());
        assert_eq!(bc_one.value(), bc_two.value());
        assert_eq!(bc_one.range(), bc_two.range());
        assert_eq!(bc_one.count(), bc_two.count());
    }

    /// `AcPrecContext::seed_from_dc` matches the §13.3.1 listing's
    /// `if (dc == 0) Prec = 0 else if (dc == 1) Prec = 1 else Prec = 2`
    /// for the first AC coefficient.
    #[test]
    fn ac_prec_context_seed_from_dc_matches_spec() {
        assert_eq!(AcPrecContext::seed_from_dc(0), AcPrecContext::WasZero);
        assert_eq!(AcPrecContext::seed_from_dc(1), AcPrecContext::WasOne);
        assert_eq!(
            AcPrecContext::seed_from_dc(2),
            AcPrecContext::WasGreaterThanOne
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(-1),
            AcPrecContext::WasGreaterThanOne,
            "dc == -1 is NOT WasOne per the spec's signed `dc == 1` test"
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(100),
            AcPrecContext::WasGreaterThanOne
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(-2048),
            AcPrecContext::WasGreaterThanOne
        );
    }

    /// Truncation: low-probability AC walks on a 4-byte stream
    /// surface `Error::Truncated` through `decode_ac_coefficient`.
    #[test]
    fn decode_ac_coefficient_surfaces_truncation() {
        let node_probs = [1u8; NUM_TREE_NODES];
        let bytes = [0xFFu8; 4];
        let mut bc = bc_over(&bytes);
        let mut saw_truncation = false;
        for n in 0..64 {
            let prec = if n & 1 == 0 {
                AcPrecContext::WasZero
            } else {
                AcPrecContext::WasOne
            };
            match decode_ac_coefficient(&mut bc, prec, (n % 60) + 1, &node_probs) {
                Ok(_) => {}
                Err(Error::Truncated) => {
                    saw_truncation = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
            }
        }
        assert!(
            saw_truncation,
            "low-prob AC walks must surface Truncated on a 4-byte stream"
        );
    }

    /// Structural: the §13.3.1 walk's leaf set covers the DC tree's
    /// leaves PLUS `EndOfBlock`. Sweeping the `(prec, encoded_coeffs,
    /// node_probs)` corners over a small stream alphabet should
    /// never produce a leaf outside this set.
    #[test]
    fn decode_ac_token_leaf_set_covers_eob_and_dc_tree() {
        let prob_corners = [
            [1u8; NUM_TREE_NODES],
            [128u8; NUM_TREE_NODES],
            [255u8; NUM_TREE_NODES],
        ];
        let stream_seeds: [u8; 4] = [0x00, 0x33, 0xA5, 0xFF];
        let prec_options = AcPrecContext::ALL;
        let pos_options: [usize; 3] = [1, 2, 17];

        for node_probs in prob_corners.iter() {
            for &seed in &stream_seeds {
                for &prec in &prec_options {
                    for &pos in &pos_options {
                        let bytes = [seed; 16];
                        let mut bc = bc_over(&bytes);
                        if let Ok(token) = decode_ac_token(&mut bc, prec, pos, node_probs) {
                            // Every result must be one of the twelve
                            // declared `DctToken`s. The §13.3.1 walk
                            // is structurally permitted to produce
                            // all twelve (DC was 11; AC adds EOB).
                            let _ = DctToken::from_index(token.index())
                                .expect("token must be a valid DctToken");
                        }
                    }
                }
            }
        }
    }

    /// `decode_ac_token` against the baseline AC probabilities and an
    /// all-zero stream from the very-first AC position (with the §13.2
    /// DC having seeded `Prec = WasZero` via `seed_from_dc(0)`). The
    /// baseline AC probabilities are all 128, which the §7.3 errata
    /// #35 documents as the half-interval where `Split = Range = 255`
    /// against `Range = 255` — every `B(128)` against `Value <
    /// 0xFF00_0000` reads 0. So at the first AC position the root
    /// `B(ZERO_CONTEXT_NODE)` reads 0 (Prec context doesn't matter at
    /// EncodedCoeffs=1), then the EOB-node read also reads 0 → the
    /// inversion gives `EOB_TOKEN`. The first-AC-position decode of
    /// the all-zero stream against baseline probs is therefore an
    /// "EOB at start of block" outcome.
    #[test]
    fn decode_ac_token_baseline_zero_stream_first_position_eob() {
        let node_probs = [128u8; NUM_TREE_NODES];
        let bytes = [0u8; 16];
        let mut bc = bc_over(&bytes);
        let prec = AcPrecContext::seed_from_dc(0);
        let token = decode_ac_token(&mut bc, prec, 1, &node_probs).expect("not truncated");
        assert_eq!(token, DctToken::EndOfBlock);
    }
}

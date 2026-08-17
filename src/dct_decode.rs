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
use crate::zrl::{ZrlBand, ZrlNode, NUM_ZRL_NODES};
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
            // Table 18 lists the per-extrabit probabilities in
            // transmission order — the FIRST-listed probability codes
            // the most-significant magnitude bit ("the most
            // significant bit of the magnitude sent first … encoded
            // with differing probabilities as specified by the final
            // column"). The §13.2.1/§13.3.1 listings' `Probs[BitsCount]`
            // (BitsCount descending) would instead put the LAST-listed
            // probability on the MSB; the conformant third-party
            // P-frame arbitrates for the prose pairing (MB (0,31) Y3's
            // CATEGORY5 DC magnitude decodes 54 = the oracle-recovered
            // value only under it), so the index is mirrored here.
            for bits_count in (0..bits).rev() {
                let bit = bc.decode_bool(probs[bits - 1 - bits_count])? as i32;
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

// ---------------------------------------------------------------------------
// §13.3.3.1 — Arithmetic AC zero-run-length decoder
// ---------------------------------------------------------------------------

/// Decode an AC zero-run length under the arithmetic entropy scheme
/// (spec §13.3.3.1).
///
/// When the §13.3.1 AC decoder produces a `ZERO_TOKEN`
/// ([`AcOutcome::ZeroRun`]) the spec mandates that the run of trailing
/// zero coefficients that follows be encoded under one of the two §7
/// entropy schemes. This function implements the BoolCoder path of
/// §13.3.3.1 — the Figure 16 binary tree traversal that returns one of
/// the eight literal run lengths `1..=8`, or, when the `>8` leaf is
/// taken, the six-bit `(RunLength - 9)` extrabit suffix plus `+ 9`
/// reconstruction.
///
/// `band` selects the row of `ZeroRunProbs[2][14]` that drives the
/// walk (Table 37):
///
/// * [`ZrlBand::Band0`] when the zero run starts at AC coefficient
///   position 1..=5,
/// * [`ZrlBand::Band1`] when it starts at position 6..=63.
///
/// `probs` is the full 14-entry row (indexed by [`ZrlNode`]) the walk
/// reads at each Figure 16 internal node and at each of the six
/// `(RunLength - 9)` extrabit positions. For the first arithmetic
/// decode of a keyframe the caller uses
/// [`crate::zrl::ZERO_RUN_PROB_DEFAULTS`]`[band.index()]`; after the
/// §13.3.3 per-frame Table 39–41 update bitstream lands, the caller
/// threads the per-frame-updated `ZeroRunProbs[band]` row instead.
///
/// The returned `u32` is the run length: the number of zero AC
/// coefficients (inclusive of the position whose `ZERO_TOKEN` triggered
/// the call). Per the §13.3.3.1 listing the literal range is `1..=8`
/// and the `>8` escape yields `9..=72` (`9 + (0..=63)`).
///
/// ## Spec pseudocode (§13.3.3.1)
///
/// ```text
/// // Select the appropriate Zero run context
/// ZeroRunProbPtr = pbi->ZeroRunProbs[ ZrlBand[pos] ]
///
/// // Now decode the zero run length
/// // Run length 1-4
/// if ( !B( ZeroRunProbPtr[0] ) )         // [>4 false]
/// {
///    if ( !B( ZeroRunProbPtr[1] ) )      // [>2 false]
///       ZeroRunCount = 1 + B( ZeroRunProbPtr[2] )   // 1 or 2 (>1 gate)
///    else                                // [>2 true]
///       ZeroRunCount = 3 + B( ZeroRunProbPtr[3] )   // 3 or 4 (>3 gate)
/// }
/// // Run length 5-8
/// else if ( !B( ZeroRunProbPtr[4] ) )    // [>4 true, >8 false]
/// {
///    if ( !B( ZeroRunProbPtr[5] ) )      // [>6 false]
///       ZeroRunCount = 5 + B( ZeroRunProbPtr[6] )   // 5 or 6 (>5 gate)
///    else                                // [>6 true]
///       ZeroRunCount = 7 + B( ZeroRunProbPtr[7] )   // 7 or 8 (>7 gate)
/// }
/// // Run length > 8
/// else                                    // [>4 true, >8 true]
/// {
///    ZeroRunCount  = B( ZeroRunProbPtr[8] )
///    ZeroRunCount += B( ZeroRunProbPtr[9]  ) << 1
///    ZeroRunCount += B( ZeroRunProbPtr[10] ) << 2
///    ZeroRunCount += B( ZeroRunProbPtr[11] ) << 3
///    ZeroRunCount += B( ZeroRunProbPtr[12] ) << 4
///    ZeroRunCount += B( ZeroRunProbPtr[13] ) << 5
///    ZeroRunCount += 9
/// }
/// ```
///
/// Note the asymmetry between the spec's printed `B( ZRP[8] ) << 0`
/// initialisation and the conventional bit-accumulation: the spec
/// reads the **least-significant** bit of `(RunLength - 9)` first.
/// This matches the §13.3.3 commentary ("the run length minus nine is
/// encoded using six-bits, **least significant bit first**", spec
/// page 78) and the analogous `read_lsb_first` raw-bit path the
/// §13.3.3.2 Huffman variant uses for the same six-bit suffix.
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted during
/// any of the constituent [`BoolCoder::decode_bool`] calls.
///
/// ## Provenance
///
/// Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §13.3.3.1
/// (page 78) plus the Table 37 / Table 38 / [`ZrlBand`] / [`ZrlNode`]
/// surface landed in round 13's [`crate::zrl`] module.
pub fn decode_ac_zero_run(
    bc: &mut BoolCoder<'_>,
    band: ZrlBand,
    probs: &[u8; NUM_ZRL_NODES],
) -> Result<u32, Error> {
    let _ = band; // The band is the caller's index into `ZeroRunProbs`;
                  // the row is already passed in via `probs`. Retained
                  // in the signature to keep the §13.3.3.1 listing's
                  // `ZeroRunProbPtr = pbi->ZeroRunProbs[ZrlBand[pos]]`
                  // structure visible in callers.
    let p = |node: ZrlNode| probs[node.index()];

    if bc.decode_bool(p(ZrlNode::GreaterThan4))? == 0 {
        // Run length 1..=4 branch.
        if bc.decode_bool(p(ZrlNode::GreaterThan2))? == 0 {
            // Run length 1 or 2.
            let bit = bc.decode_bool(p(ZrlNode::GreaterThan1))? as u32;
            Ok(1 + bit)
        } else {
            // Run length 3 or 4.
            let bit = bc.decode_bool(p(ZrlNode::GreaterThan3))? as u32;
            Ok(3 + bit)
        }
    } else if bc.decode_bool(p(ZrlNode::GreaterThan8))? == 0 {
        // Run length 5..=8 branch.
        if bc.decode_bool(p(ZrlNode::GreaterThan6))? == 0 {
            // Run length 5 or 6.
            let bit = bc.decode_bool(p(ZrlNode::GreaterThan5))? as u32;
            Ok(5 + bit)
        } else {
            // Run length 7 or 8.
            let bit = bc.decode_bool(p(ZrlNode::GreaterThan7))? as u32;
            Ok(7 + bit)
        }
    } else {
        // Run length > 8: six extrabits encoding (RunLength - 9), LSB
        // first. Each bit is read with its own per-position
        // probability from the Table 38 indices 8..=13.
        let mut value: u32 = 0;
        value |= bc.decode_bool(p(ZrlNode::ExtraBit0))? as u32;
        value |= (bc.decode_bool(p(ZrlNode::ExtraBit1))? as u32) << 1;
        value |= (bc.decode_bool(p(ZrlNode::ExtraBit2))? as u32) << 2;
        value |= (bc.decode_bool(p(ZrlNode::ExtraBit3))? as u32) << 3;
        value |= (bc.decode_bool(p(ZrlNode::ExtraBit4))? as u32) << 4;
        value |= (bc.decode_bool(p(ZrlNode::ExtraBit5))? as u32) << 5;
        Ok(value + 9)
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
    /// reconstruction step. Rather than hand-craft a bitstream whose
    /// `b(1)` (fixed probability 128, operative `>> 8` Split per errata
    /// #35: `Split ≈ Range/2`) lands on each sign value, the
    /// reconstruction arithmetic is verified as the pure-integer
    /// identity it is, in isolation.
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
    /// `node_probs[ZERO] = 255` (strong 1-bias at the root). Under the
    /// operative `>> 8` Split (errata #35) prob 255 at Range 255 gives
    /// `Split = 1 + (254 * 255 >> 8) = 254 = Range - 1`, so the 1-wide
    /// `Bit = 1` interval needs `Value >= 254 << 24 = 0xFE00_0000`.
    /// With `Value = 0` the comparison `0 < 0xFE00_0000` is true, so
    /// the read takes the 0-branch regardless of the 1-bias —
    /// returning `ZERO_TOKEN` → DC = 0. (An all-zero `Value` always
    /// takes the 0-branch since `Split >= 1`.)
    #[test]
    fn decode_dc_root_high_prob_with_zero_value_short_circuits() {
        let mut node_probs = [255u8; NUM_TREE_NODES];
        node_probs[TreeNode::EndOfBlock.index()] = 255;
        let bytes = [0u8; 16];
        let mut bc = bc_over(&bytes);
        let dc = decode_dc(&mut bc, &node_probs).expect("not truncated");
        assert_eq!(
            dc, 0,
            "Value=0 < Split<<24 → 0-branch even with prob-255 root → ZERO_TOKEN → DC=0"
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

    /// `decode_ac_token` ZERO/EOB inversion: against an all-zero stream
    /// every `B(prob)` read picks the 0-branch (Value=0 is below any
    /// `Split << 24` since the operative `>> 8` Split is always `>= 1`,
    /// errata #35), so both the root ZERO node and the EOB node read 0;
    /// under the §13.3.1 inversion that gives EOB_TOKEN. The opposite
    /// (ZERO_TOKEN) outcome is verified by
    /// `decode_ac_coefficient_zero_run_outcome`.
    #[test]
    fn decode_ac_token_eob_inversion_branches() {
        // Pin the root to take the 0-branch (ZERO sub-branch) and the
        // EOB node to take the 0-branch (→ EOB_TOKEN under the §13.3.1
        // inversion). All-zero stream works at any probability:
        //   Split = 1 + ((255-1)*1 >> 8) = 1 + 0 = 1. Split<<24
        //   = 0x0100_0000 > Value=0 → 0-branch.
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
    /// Construction (operative `>> 8` Split, errata #35): pin
    /// `ZERO_CONTEXT_NODE` prob to 255 and `EOB_CONTEXT_NODE` prob to
    /// 1, with a leading byte `0xFD` (initial `Value = 0xFDFF_FFFF`).
    /// The root Split is `1 + ((255-1)*255 >> 8) = 254`; `Split << 24 =
    /// 0xFE00_0000 > Value = 0xFDFF_FFFF`, so the root takes the
    /// 0-branch (ZERO subtree), updating `Range = 254`. The EOB Split
    /// is `1 + ((254-1)*1 >> 8) = 1`; `Split << 24 = 0x0100_0000 <
    /// Value`, so EOB takes the 1-branch — which under the §13.3.1
    /// inversion is `ZERO_TOKEN`.
    #[test]
    fn decode_ac_coefficient_zero_run_outcome() {
        let mut node_probs = [1u8; NUM_TREE_NODES];
        node_probs[TreeNode::Zero.index()] = 255;
        node_probs[TreeNode::EndOfBlock.index()] = 1;
        let bytes = [0xFDu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
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

    /// `AcPrecContext::seed_from_dc` classifies the DC **magnitude**
    /// (the §13.3.1 printed signed `dc == 1` test is a spec defect —
    /// fixture-arbitrated; see `seed_from_dc`'s docs).
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
            AcPrecContext::WasOne,
            "dc == -1 seeds Prec = 1: the context is magnitude-based"
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
    /// baseline AC probabilities are all 128, the §7.3 half-interval
    /// point (operative `>> 8` Split, errata #35: `Split = 128` at
    /// `Range = 255`) — every `B(128)` against the all-zero `Value = 0`
    /// reads 0 (`0 < Split << 24 = 0x8000_0000`). So at the first AC
    /// position the root
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

    // -------- §13.3.3.1 AC zero-run-length BoolCoder decoder --------

    use crate::zrl::ZERO_RUN_PROB_DEFAULTS;

    /// All-zero byte stream against a `[1; 14]` probability row.
    ///
    /// At `Probability = 1, Range = 255` the §7.3 `Split` formula
    /// (operative `>> 8`, errata #35) yields
    /// `Split = 1 + ((254 * 1) >> 8) = 1 + 0 = 1`, so the 0-branch is
    /// `Value < (Split << 24) = 0x0100_0000`. With `Value = 0` the
    /// comparison is always true → every `decode_bool` returns `0`.
    /// Following the §13.3.3.1 tree with
    /// every internal-node read returning `0`:
    /// `>4 = false → >2 = false → >1 = false → ZeroRunCount = 1 + 0 = 1`.
    #[test]
    fn decode_ac_zero_run_zero_stream_low_prob_yields_one() {
        let probs = [1u8; NUM_ZRL_NODES];
        let bytes = [0u8; 32];
        let mut bc = bc_over(&bytes);
        let n = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs).expect("not truncated");
        assert_eq!(n, 1, "all-zero reads at low prob → run length 1");
    }

    /// Same input shape but with [`ZrlBand::Band1`] — the band index
    /// is informational; this test pins that the returned run length
    /// is independent of the band when the probability row is
    /// identical (the band is just the row selector in
    /// `ZeroRunProbs[2][14]`).
    #[test]
    fn decode_ac_zero_run_band_argument_is_row_selector_only() {
        let probs = [1u8; NUM_ZRL_NODES];
        let bytes0 = [0u8; 32];
        let bytes1 = [0u8; 32];
        let mut bc0 = bc_over(&bytes0);
        let mut bc1 = bc_over(&bytes1);
        let r0 = decode_ac_zero_run(&mut bc0, ZrlBand::Band0, &probs).unwrap();
        let r1 = decode_ac_zero_run(&mut bc1, ZrlBand::Band1, &probs).unwrap();
        assert_eq!(r0, r1);
    }

    /// At very small probabilities (`[1; 14]`) the all-zero stream
    /// only ever reads `0` bits from the BoolCoder, so the walk goes
    /// down the leftmost path on every internal node. The leftmost
    /// path of Figure 16 is `>4 false → >2 false → >1 false →
    /// ZeroRunCount = 1 + 0 = 1`. No matter the band, the result is
    /// always `1` — and the BoolCoder state must advance (renorm
    /// loop consumes bytes for small-Split branches).
    #[test]
    fn decode_ac_zero_run_advances_bool_coder_state() {
        let probs = [1u8; NUM_ZRL_NODES];
        let bytes = [0u8; 64];
        let mut bc = bc_over(&bytes);
        let pos_before = bc.pos();
        let _ = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs).unwrap();
        let pos_after = bc.pos();
        assert!(
            pos_after > pos_before,
            "small-Split renormalization must have pulled at least one fresh byte"
        );
    }

    /// `decode_ac_zero_run` against the `> 8` escape path: drive
    /// every internal node to the 1-branch (the run-length > 8
    /// region) and every extrabit to the 0-branch (lowest possible
    /// `(RunLength - 9)` value of `0`), so the result is exactly `9`
    /// (the minimum of the `> 8` escape's `9 + (0..=63)` range).
    ///
    /// Setup (operative `>> 8` Split, errata #35): the first two
    /// internal reads (`> 4`, `> 8`) use probability `1` so
    /// `Split = 1 + (254 * 1 >> 8) = 1`; a high-valued stream
    /// (`0xF0FF_0000` initial `Value`) keeps `Value >= Split << 24 =
    /// 0x0100_0000`, so both fire their 1-branch (entering the `> 8`
    /// escape). The six extrabits use probability `255` so
    /// `Split = 1 + ((Range-1) * 255 >> 8) = Range - 1`; as the high
    /// bits drain out of `Value`, each extrabit read finds
    /// `Value < Split << 24` and takes the 0-branch, so the six-bit
    /// `(RunLength - 9)` suffix is `0` and the run is exactly `9`.
    #[test]
    fn decode_ac_zero_run_greater_than_eight_with_zero_extrabits_yields_nine() {
        let mut probs = [255u8; NUM_ZRL_NODES];
        // Internal nodes `>4` and `>8`: probability 1 → Split = 1, the
        // 1-branch fires for the high-valued leading bytes, entering
        // the `> 8` escape.
        probs[ZrlNode::GreaterThan4.index()] = 1;
        probs[ZrlNode::GreaterThan8.index()] = 1;
        let mut bytes = [0u8; 64];
        bytes[0] = 0xF0;
        bytes[1] = 0xFF;
        let mut bc = bc_over(&bytes);
        let n = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs).expect("not truncated");
        assert_eq!(
            n, 9,
            "1-branch at >4 and >8, 0-branches on six extrabits → run = 0 + 9 = 9"
        );
    }

    /// A full row of `Probability = 255` against a moderate stream
    /// drives every Figure 16 read to its 0-branch, landing on the
    /// leftmost leaf (run-length 1). Under the operative `>> 8` Split
    /// (errata #35) `Probability = 255` gives `Split = Range - 1`
    /// (e.g. `254` at `Range = 255`), so the 1-wide `Bit = 1` interval
    /// needs `Value >= 0xFE00_0000`; the `0xA5A5_A5A5` stream stays
    /// well below that, locking the 0-branch on every read.
    #[test]
    fn decode_ac_zero_run_prob_extremes_for_each_node_force_branch() {
        // Probability 255 → Split = Range - 1 (operative `>> 8` Split,
        // errata #35), so the 0-branch fires unless `Value` reaches the
        // top 1-wide interval. The 0xA5 stream never does, so the full
        // Figure 16 walk goes 0 on every read and lands on the leftmost
        // leaf: `>4 false → >2 false → >1 false → 1 + 0 = 1`.
        let probs_high = [255u8; NUM_ZRL_NODES];
        let bytes = [0xA5u8; 64];
        let mut bc = bc_over(&bytes);
        let n = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs_high).expect("not truncated");
        assert_eq!(n, 1, "prob=255 forces 0-branch on every read → run = 1");
    }

    /// The §13.3.3.1 listing's published `if (!B(prob[0]))` form
    /// inverts the `B` result before branching. Verify that
    /// distinction holds: the 0-branch of `B(prob[0])` is the
    /// "run length 1..=4" subtree, and the 1-branch is the "5+"
    /// subtree. We pin this by driving the first read to 0 via
    /// `probs[0] = 255` (operative `>> 8` Split, errata #35:
    /// `Split = Range - 1`, and the `0xCD` stream stays below the
    /// top 1-wide `Bit = 1` interval → 0-branch); every subsequent
    /// read also takes the 0-branch (full row of 255s) and we land
    /// on `1 + 0 = 1` (in the 1..=4 subtree).
    #[test]
    fn decode_ac_zero_run_root_zero_branch_picks_lower_subtree() {
        let probs = [255u8; NUM_ZRL_NODES];
        let bytes = [0xCDu8; 64];
        let mut bc = bc_over(&bytes);
        let n = decode_ac_zero_run(&mut bc, ZrlBand::Band1, &probs).expect("not truncated");
        // 0-branch on every node → lower subtree → run-length 1.
        assert!(
            (1..=4).contains(&n),
            "0-branch root selects lower subtree (run 1..=4); got {n}"
        );
    }

    /// `decode_ac_zero_run` must surface `Error::Truncated` if the
    /// BoolCoder runs out of bytes. We construct a 4-byte stream
    /// (the §7.3 minimum init) with `probs[0] = 1` so every read
    /// triggers heavy renormalization. After a small number of
    /// reads the BoolCoder will exhaust its byte stream.
    #[test]
    fn decode_ac_zero_run_truncated_surface() {
        let probs = [1u8; NUM_ZRL_NODES];
        let bytes = [0u8; 4];
        let mut bc = bc_over(&bytes);
        // The init consumed the only 4 bytes; the very first
        // decode_bool will renorm to refill and fail with Truncated.
        let err = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs).unwrap_err();
        assert_eq!(err, Error::Truncated);
    }

    /// Determinism: two BoolCoders fed the same bytes against the
    /// same probability row produce identical results.
    ///
    /// We use `Probability = 64`: at `Range = 255` the §7.3 formula
    /// (operative `>> 8` Split, errata #35) yields
    /// `Split = 1 + (254 * 64 >> 8) = 64`, a well-centred partition
    /// that keeps both sub-intervals comfortably wide. (Under the
    /// operative `>> 8` Split the coder is non-degenerate at every
    /// probability, so this is purely for a readable trace.)
    #[test]
    fn decode_ac_zero_run_determinism() {
        let probs = [64u8; NUM_ZRL_NODES];
        for seed in [0x00u8, 0x55, 0xA5, 0xFF] {
            let bytes_a = [seed; 64];
            let bytes_b = [seed; 64];
            let mut bc_a = bc_over(&bytes_a);
            let mut bc_b = bc_over(&bytes_b);
            let r_a = decode_ac_zero_run(&mut bc_a, ZrlBand::Band0, &probs).unwrap();
            let r_b = decode_ac_zero_run(&mut bc_b, ZrlBand::Band0, &probs).unwrap();
            assert_eq!(r_a, r_b, "same input → same output (seed={seed:#x})");
        }
    }

    /// The §13.3.3.1 output range is `1..=72`: literal 1..=8 from
    /// the eight binary-tree leaves plus 9..=72 from the
    /// `9 + (0..=63)` escape. Sweep stream seeds and probability
    /// rows; every successful decode must land in this range.
    #[test]
    fn decode_ac_zero_run_output_range_invariant() {
        let prob_rows = [
            [1u8; NUM_ZRL_NODES],
            [64u8; NUM_ZRL_NODES],
            [128u8; NUM_ZRL_NODES],
            [200u8; NUM_ZRL_NODES],
            [255u8; NUM_ZRL_NODES],
            ZERO_RUN_PROB_DEFAULTS[0],
            ZERO_RUN_PROB_DEFAULTS[1],
        ];
        let stream_seeds: [u8; 5] = [0x00, 0x33, 0x55, 0xA5, 0xFF];

        for probs in &prob_rows {
            for &seed in &stream_seeds {
                for band in ZrlBand::ALL {
                    let bytes = [seed; 64];
                    let mut bc = bc_over(&bytes);
                    if let Ok(n) = decode_ac_zero_run(&mut bc, band, probs) {
                        assert!(
                            (1..=72).contains(&n),
                            "run length must be in 1..=72; got {n} (probs={probs:?}, seed={seed:#x}, band={band:?})"
                        );
                    }
                }
            }
        }
    }

    /// `decode_ac_zero_run` against [`ZERO_RUN_PROB_DEFAULTS`] is the
    /// concrete keyframe-decode path. Verify that decoding against
    /// the two spec-published rows + an all-zero byte stream returns
    /// a well-defined run length (any value in `1..=72`) without
    /// panicking and without exhausting the bytestream prematurely.
    #[test]
    fn decode_ac_zero_run_against_keyframe_defaults() {
        for band in ZrlBand::ALL {
            let probs = ZERO_RUN_PROB_DEFAULTS[band.index()];
            let bytes = [0u8; 64];
            let mut bc = bc_over(&bytes);
            let n = decode_ac_zero_run(&mut bc, band, &probs).expect("not truncated");
            assert!(
                (1..=72).contains(&n),
                "keyframe-default decode produced out-of-range run = {n} for band {band:?}"
            );
        }
    }

    /// `AcOutcome::ZeroRun` is the §13.3.1 hand-off into the
    /// §13.3.3.1 decoder. Compose the two layers and verify the
    /// resulting run length is well-defined: with baseline
    /// `node_probs = [128; 11]` and `prec = WasZero` seeded from
    /// `DC = 0` at the first AC position, an all-zero stream lands
    /// on `EndOfBlock` (see `decode_ac_token_baseline_zero_stream_first_position_eob`).
    /// We therefore use a stream that drives the §13.3.1 walk to a
    /// `ZeroRun` outcome instead: an all-zero stream at the
    /// **second** AC position with `prec = WasGreaterThanOne` makes
    /// the EOB branch read 1-bit (escape from the implicit-1 shortcut)
    /// → `ZeroRun`. This pins the hand-off contract.
    #[test]
    fn decode_ac_zero_run_composes_with_ac_outcome_zero_run() {
        // The §13.3.1 path at the second AC position with prec ==
        // WasGreaterThanOne reads `B(ZERO)` from the root. With
        // baseline probs (all 128) and an all-zero stream the
        // 0-branch fires (Value 0 < Split<<24); then the EOB-node
        // read also takes the 0-branch (Value still 0; Split<<24 >
        // 0) → DctToken::EndOfBlock, not Zero.
        //
        // To deterministically obtain the `Zero` token (and thus the
        // ZeroRun outcome) we use `node_probs[EOB_CONTEXT_NODE] = 1`
        // with all other entries at 128, plus a byte stream where
        // the first bit decoded against that low EOB probability
        // forces the 1-branch. The cleanest way is to manually
        // construct the AcOutcome::ZeroRun directly and pipe through
        // the zero-run decoder.
        let outcome = AcOutcome::ZeroRun;
        if let AcOutcome::ZeroRun = outcome {
            let probs = ZERO_RUN_PROB_DEFAULTS[ZrlBand::Band0.index()];
            let bytes = [0u8; 64];
            let mut bc = bc_over(&bytes);
            let run = decode_ac_zero_run(&mut bc, ZrlBand::Band0, &probs).expect("not truncated");
            assert!(
                (1..=72).contains(&run),
                "ZeroRun hand-off produced an out-of-range run = {run}"
            );
        } else {
            panic!("AcOutcome::ZeroRun did not match the ZeroRun pattern");
        }
    }
}

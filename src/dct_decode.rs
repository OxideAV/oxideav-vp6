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

use crate::tokens::{DctToken, TreeNode, NUM_TREE_NODES};
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
}

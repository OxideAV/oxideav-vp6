//! VP6 arithmetic DCT-token **encoder** — the bit-for-bit dual of the
//! §13.2.1 / §13.3.1 / §13.3.3.1 decoders in [`crate::dct_decode`].
//!
//! Each function here emits, via the §7.3 [`crate::bool_coder::BoolEncoder`],
//! exactly the bit sequence the matching decoder consumes — same node
//! probabilities, same tree branch order, same magnitude/sign layout —
//! so an encode followed by a decode recovers the original coefficient
//! block identically. The encoder is the structural mirror image of the
//! decoder: where the decoder reads `B(prob)` and branches, the encoder
//! knows the branch it wants and emits the bit that drives the decoder
//! there.
//!
//! ## Token classification
//!
//! A signed coefficient maps to a [`DctToken`] by magnitude (§13 Table
//! 18 `Min`/`Max` ranges): magnitude 0 → `ZERO_TOKEN`, 1..=4 →
//! `ONE`..`FOUR`, and 5..=2114 into the six `CATEGORY` tokens. The
//! magnitude bits and sign are then emitted exactly as
//! [`crate::dct_decode::decode_token_value`] reads them.
//!
//! ## Provenance
//!
//! Derived solely from the §13 decode listings in
//! `docs/video/vp6/vp6_format.pdf` (the trees this module inverts) plus
//! the in-tree errata at `docs/video/vp6/vp6-errata-and-clarifications.md`
//! (#67, the magnitude-only probability slice). No external library code
//! was consulted.

use crate::block_decode::{AcProbBank, ZeroRunProbBank, BLOCK_SIZE};
use crate::bool_coder::BoolEncoder;
use crate::tokens::{AcBand, AcPlane, AcPrecContext, DctToken, TreeNode, NUM_TREE_NODES};
use crate::zrl::{ZrlBand, ZrlNode, NUM_ZRL_NODES};

/// Classify a signed coefficient magnitude into the value-carrying
/// [`DctToken`] whose `Min..=Max` range (§13 Table 18) contains it.
///
/// `magnitude` is the absolute value of the coefficient (`0` returns
/// `ZERO_TOKEN`). Panics only on a magnitude past `CATEGORY6`'s `2114`
/// ceiling, which a conformant §15-quantised coefficient never reaches.
pub fn token_for_magnitude(magnitude: u16) -> DctToken {
    match magnitude {
        0 => DctToken::Zero,
        1 => DctToken::One,
        2 => DctToken::Two,
        3 => DctToken::Three,
        4 => DctToken::Four,
        5..=6 => DctToken::Category1,
        7..=10 => DctToken::Category2,
        11..=18 => DctToken::Category3,
        19..=34 => DctToken::Category4,
        35..=66 => DctToken::Category5,
        67..=2114 => DctToken::Category6,
        other => panic!("coefficient magnitude {other} exceeds CATEGORY6 max 2114"),
    }
}

/// Emit the magnitude bits and sign for a value-carrying token — the
/// dual of [`crate::dct_decode::decode_token_value`].
///
/// `coeff` is the signed coefficient; `token` is its classification
/// from [`token_for_magnitude`]. For `ONE`..`FOUR` only the sign bit is
/// emitted (the magnitude is the token constant). For the `CATEGORY`
/// tokens the magnitude offset `(|coeff| - Min)` is emitted MSB-first
/// over the errata-#67 magnitude-only probability slice, then the sign.
///
/// `ZERO_TOKEN` / `EOB_TOKEN` carry no value and emit nothing.
pub fn encode_token_value(enc: &mut BoolEncoder, token: DctToken, coeff: i32) {
    match token {
        DctToken::Zero | DctToken::EndOfBlock => {}
        DctToken::One | DctToken::Two | DctToken::Three | DctToken::Four => {
            // Magnitude is the token constant; only the sign is coded.
            // Decoder reconstructs `(magnitude ^ -sign) + sign`, so the
            // sign bit is `1` iff `coeff < 0`.
            let sign = (coeff < 0) as u8;
            enc.encode_b1(sign);
        }
        _ => {
            // CATEGORY1..CATEGORY6. The decoder accumulates
            // `value = min; for bits_count in (0..bits).rev()
            //   value += B(probs[bits - 1 - bits_count]) << bits_count`
            // — `value - min` is sent MSB-first, and the Table 18
            // probability list is in transmission order (first-listed
            // prob codes the MSB; fixture-arbitrated round 447), so the
            // encoder mirrors the same mirrored index.
            let min = token.min_value() as i32;
            let probs = token.magnitude_probs();
            let bits = probs.len();
            let magnitude = coeff.unsigned_abs() as i32;
            let offset = magnitude - min;
            debug_assert!(
                offset >= 0 && offset < (1 << bits),
                "magnitude {magnitude} out of {token} range (min {min}, {bits} bits)"
            );
            for bits_count in (0..bits).rev() {
                let bit = ((offset >> bits_count) & 1) as u8;
                enc.encode_bool(bit, probs[bits - 1 - bits_count]);
            }
            let sign = (coeff < 0) as u8;
            enc.encode_b1(sign);
        }
    }
}

/// Walk the §13.2.1 DC tree to a leaf and emit the branch bits — the
/// dual of [`crate::dct_decode::decode_dc_token`].
///
/// `node_probs` is the per-(plane, context) probability vector
/// (`DcNodeContexts[plane][context]`), identical to the decoder's. The
/// DC tree never produces `EOB`, so `token` must be a value-carrying
/// token or `ZERO_TOKEN`.
pub fn encode_dc_token(enc: &mut BoolEncoder, node_probs: &[u8; NUM_TREE_NODES], token: DctToken) {
    let p = |n: TreeNode| node_probs[n.index()];

    // Root: 0 → ZERO_TOKEN, 1 → value-carrying subtree.
    if matches!(token, DctToken::Zero) {
        enc.encode_bool(0, p(TreeNode::Zero));
        return;
    }
    enc.encode_bool(1, p(TreeNode::Zero));

    // ONE_CONTEXT_NODE: 0 → ONE_TOKEN, 1 → larger.
    if matches!(token, DctToken::One) {
        enc.encode_bool(0, p(TreeNode::One));
        return;
    }
    enc.encode_bool(1, p(TreeNode::One));

    encode_value_subtree(enc, node_probs, token);
}

/// Walk the §13.3.1 AC tree to a leaf and emit the branch bits — the
/// dual of [`crate::dct_decode::decode_ac_token`], including the
/// `(EncodedCoeffs > 1) && (Prec == WasZero)` implicit-1 shortcut.
///
/// When the shortcut fires the decoder does **not** read the
/// `ZERO_CONTEXT_NODE` bit (it is implicitly `1`), so this encoder must
/// likewise **not** emit it. `token` is the desired leaf: `EndOfBlock`,
/// `Zero` (a zero-run trigger), or a value-carrying token.
pub fn encode_ac_token(
    enc: &mut BoolEncoder,
    prec: AcPrecContext,
    encoded_coeffs: usize,
    node_probs: &[u8; NUM_TREE_NODES],
    token: DctToken,
) {
    let p = |n: TreeNode| node_probs[n.index()];

    let implicit_one = encoded_coeffs > 1 && matches!(prec, AcPrecContext::WasZero);

    let this_token_non_zero = !matches!(token, DctToken::Zero | DctToken::EndOfBlock);

    if !implicit_one {
        // Emit the ZERO_CONTEXT_NODE bit: 1 ⇒ value, 0 ⇒ ZERO/EOB pair.
        enc.encode_bool(this_token_non_zero as u8, p(TreeNode::Zero));
    } else {
        // Decoder forced `this_token_non_zero = true`; a Zero/EOB token
        // is unrepresentable here. The block-level encoder guarantees it
        // never asks for one after a zero run.
        debug_assert!(
            this_token_non_zero,
            "implicit-1 AC position cannot encode {token}"
        );
    }

    if !this_token_non_zero {
        // ZERO_CONTEXT_NODE == 0 → EOB_CONTEXT_NODE decides:
        //   1 → ZERO_TOKEN (zero run follows), 0 → EOB_TOKEN.
        let eob_bit = matches!(token, DctToken::Zero) as u8;
        enc.encode_bool(eob_bit, p(TreeNode::EndOfBlock));
        return;
    }

    // ONE_CONTEXT_NODE: 0 → ONE_TOKEN, 1 → larger.
    if matches!(token, DctToken::One) {
        enc.encode_bool(0, p(TreeNode::One));
        return;
    }
    enc.encode_bool(1, p(TreeNode::One));

    encode_value_subtree(enc, node_probs, token);
}

/// Emit the shared value-token subtree below `ONE_CONTEXT_NODE` (the
/// `LOW_VAL`/category/`TWO`..`FOUR` fan-out, identical in the DC and AC
/// trees). `token` must be one of `Two`..`Four` or `Category1`..
/// `Category6`.
fn encode_value_subtree(enc: &mut BoolEncoder, node_probs: &[u8; NUM_TREE_NODES], token: DctToken) {
    let p = |n: TreeNode| node_probs[n.index()];

    let is_category = matches!(
        token,
        DctToken::Category1
            | DctToken::Category2
            | DctToken::Category3
            | DctToken::Category4
            | DctToken::Category5
            | DctToken::Category6
    );

    // LOW_VAL_CONTEXT_NODE: 1 → categories 1..=6, 0 → TWO/THREE/FOUR.
    enc.encode_bool(is_category as u8, p(TreeNode::LowVal));

    if !is_category {
        // TWO_CONTEXT_NODE: 0 → TWO, 1 → THREE/FOUR.
        match token {
            DctToken::Two => enc.encode_bool(0, p(TreeNode::Two)),
            DctToken::Three => {
                enc.encode_bool(1, p(TreeNode::Two));
                enc.encode_bool(0, p(TreeNode::Three));
            }
            DctToken::Four => {
                enc.encode_bool(1, p(TreeNode::Two));
                enc.encode_bool(1, p(TreeNode::Three));
            }
            _ => unreachable!("non-category, non-2/3/4 token in value subtree"),
        }
        return;
    }

    // Categories. HIGH_LOW_CONTEXT_NODE: 1 → CAT3..CAT6, 0 → CAT1/CAT2.
    match token {
        DctToken::Category1 => {
            enc.encode_bool(0, p(TreeNode::HighLow));
            enc.encode_bool(0, p(TreeNode::CatOne));
        }
        DctToken::Category2 => {
            enc.encode_bool(0, p(TreeNode::HighLow));
            enc.encode_bool(1, p(TreeNode::CatOne));
        }
        DctToken::Category3 => {
            enc.encode_bool(1, p(TreeNode::HighLow));
            enc.encode_bool(0, p(TreeNode::CatThreeFour));
            enc.encode_bool(0, p(TreeNode::CatThree));
        }
        DctToken::Category4 => {
            enc.encode_bool(1, p(TreeNode::HighLow));
            enc.encode_bool(0, p(TreeNode::CatThreeFour));
            enc.encode_bool(1, p(TreeNode::CatThree));
        }
        DctToken::Category5 => {
            enc.encode_bool(1, p(TreeNode::HighLow));
            enc.encode_bool(1, p(TreeNode::CatThreeFour));
            enc.encode_bool(0, p(TreeNode::CatFive));
        }
        DctToken::Category6 => {
            enc.encode_bool(1, p(TreeNode::HighLow));
            enc.encode_bool(1, p(TreeNode::CatThreeFour));
            enc.encode_bool(1, p(TreeNode::CatFive));
        }
        _ => unreachable!("non-category token in category subtree"),
    }
}

/// Full single-coefficient arithmetic DC encode — the dual of
/// [`crate::dct_decode::decode_dc`]. Classifies `coeff`, walks the DC
/// tree, then emits the magnitude/sign.
pub fn encode_dc(enc: &mut BoolEncoder, node_probs: &[u8; NUM_TREE_NODES], coeff: i32) {
    let token = token_for_magnitude(coeff.unsigned_abs() as u16);
    encode_dc_token(enc, node_probs, token);
    encode_token_value(enc, token, coeff);
}

/// Emit the §13.3.3.1 AC zero-run length — the dual of
/// [`crate::dct_decode::decode_ac_zero_run`].
///
/// `run` is the run length (`1..=72`, inclusive of the position whose
/// `ZERO_TOKEN` triggered it). For `1..=8` the Figure 16 literal tree is
/// walked; for `>= 9` the `>8` escape is taken and `(run - 9)` is
/// emitted as six extrabits LSB-first under their Table 38
/// probabilities. `probs` is the full 14-entry `ZeroRunProbs[band]` row.
pub fn encode_ac_zero_run(
    enc: &mut BoolEncoder,
    band: ZrlBand,
    probs: &[u8; NUM_ZRL_NODES],
    run: u32,
) {
    let _ = band; // Row already supplied via `probs`, matching decoder.
    let p = |n: ZrlNode| probs[n.index()];
    debug_assert!((1..=72).contains(&run), "zero run {run} out of range");

    if run <= 4 {
        enc.encode_bool(0, p(ZrlNode::GreaterThan4));
        if run <= 2 {
            enc.encode_bool(0, p(ZrlNode::GreaterThan2));
            // run 1 → bit 0, run 2 → bit 1.
            enc.encode_bool((run - 1) as u8, p(ZrlNode::GreaterThan1));
        } else {
            enc.encode_bool(1, p(ZrlNode::GreaterThan2));
            // run 3 → bit 0, run 4 → bit 1.
            enc.encode_bool((run - 3) as u8, p(ZrlNode::GreaterThan3));
        }
    } else if run <= 8 {
        enc.encode_bool(1, p(ZrlNode::GreaterThan4));
        enc.encode_bool(0, p(ZrlNode::GreaterThan8));
        if run <= 6 {
            enc.encode_bool(0, p(ZrlNode::GreaterThan6));
            enc.encode_bool((run - 5) as u8, p(ZrlNode::GreaterThan5));
        } else {
            enc.encode_bool(1, p(ZrlNode::GreaterThan6));
            enc.encode_bool((run - 7) as u8, p(ZrlNode::GreaterThan7));
        }
    } else {
        enc.encode_bool(1, p(ZrlNode::GreaterThan4));
        enc.encode_bool(1, p(ZrlNode::GreaterThan8));
        let value = run - 9;
        enc.encode_bool((value & 1) as u8, p(ZrlNode::ExtraBit0));
        enc.encode_bool(((value >> 1) & 1) as u8, p(ZrlNode::ExtraBit1));
        enc.encode_bool(((value >> 2) & 1) as u8, p(ZrlNode::ExtraBit2));
        enc.encode_bool(((value >> 3) & 1) as u8, p(ZrlNode::ExtraBit3));
        enc.encode_bool(((value >> 4) & 1) as u8, p(ZrlNode::ExtraBit4));
        enc.encode_bool(((value >> 5) & 1) as u8, p(ZrlNode::ExtraBit5));
    }
}

/// Encode one full 8x8 block of scan-order DCT coefficients — the
/// bit-for-bit dual of [`crate::block_decode::decode_block_coefficients`].
///
/// `coeffs` is the block in **scan order**: `coeffs[0]` the §13.2.1 DC,
/// `coeffs[1..=63]` the AC coefficients at scan positions `1..=63`. The
/// function emits, via `enc`:
///
/// 1. The §13.2.1 DC token + magnitude/sign ([`encode_dc`]).
/// 2. The §13.3.1 AC loop: each non-zero AC coefficient as a value
///    token (with its preceding zero run, if any, coded by the
///    §13.3.3.1 zero-run encoder), trailing zeros terminated by an EOB
///    token — exactly mirroring the decoder's `EncodedCoeffs` walk,
///    including the `Prec` context evolution and the implicit-1 shortcut
///    after a zero run.
///
/// `plane`, `dc_node_probs`, `ac_probs`, `zrl_probs` are the same
/// probability surfaces the decoder uses, so the emitted stream decodes
/// back to `coeffs` identically.
///
/// ## Zero-run / EOB choreography
///
/// The decoder's AC loop, at each scan position, either reads a value,
/// an EOB (block ends), or a ZERO token that triggers a §13.3.3.1 zero
/// run (advancing `EncodedCoeffs` by the inclusive run length). The
/// encoder inverts this by scanning for the last non-zero AC position:
///
/// * positions up to and including the last non-zero are coded as value
///   tokens, with any gap of `k` consecutive zeros before a value coded
///   as a single ZERO token whose run length is `k` (the run is
///   inclusive of the first zero position, so a gap of `k` zeros is run
///   length `k`);
/// * after the last non-zero, an EOB token terminates the block (unless
///   the last non-zero is at scan position 63, in which case the block
///   is naturally full and no EOB is coded — matching the decoder's
///   `EncodedCoeffs >= BLOCK_SIZE` loop exit).
pub fn encode_block_coefficients(
    enc: &mut BoolEncoder,
    plane: AcPlane,
    dc_node_probs: &[u8; NUM_TREE_NODES],
    ac_probs: &AcProbBank,
    zrl_probs: &ZeroRunProbBank,
    coeffs: &[i32; BLOCK_SIZE],
) {
    // §13.2.1 DC.
    let dc = coeffs[0];
    encode_dc(enc, dc_node_probs, dc);

    // §13.3.1 AC loop state, mirroring the decoder.
    let mut prec = AcPrecContext::seed_from_dc(dc);
    let mut encoded_coeffs: usize = 1;

    // Find the last non-zero AC scan position (1..=63), if any.
    let last_nonzero = (1..BLOCK_SIZE).rev().find(|&i| coeffs[i] != 0);

    let Some(last) = last_nonzero else {
        // No AC energy: the decoder's first AC step must take the EOB
        // leaf. `encoded_coeffs == 1`, so the implicit-1 shortcut is
        // inactive (it needs `> 1`); the ZERO/EOB bits are emitted.
        let band = AcBand::for_coefficient_position(1).expect("position 1 maps to a band");
        let node_probs = &ac_probs[plane.index()][prec.index()][band.index()];
        encode_ac_token(enc, prec, encoded_coeffs, node_probs, DctToken::EndOfBlock);
        return;
    };

    while encoded_coeffs <= last {
        let pos = encoded_coeffs;
        if coeffs[pos] == 0 {
            // Accumulate a zero run up to (but not including) the next
            // non-zero coefficient. The run is inclusive of `pos`.
            let mut run = 0usize;
            while coeffs[encoded_coeffs] == 0 {
                run += 1;
                encoded_coeffs += 1;
            }

            // The ZERO token is emitted at scan position `pos` under the
            // current `prec` (the decoder reads the ZERO leaf there).
            let band = AcBand::for_coefficient_position(pos).expect("scan pos maps to a band");
            let node_probs = &ac_probs[plane.index()][prec.index()][band.index()];
            encode_ac_token(enc, prec, pos, node_probs, DctToken::Zero);

            // §13.3.3.1 run length (inclusive), band from the triggering
            // position.
            let zrl_band =
                ZrlBand::for_coefficient_position(pos).expect("scan pos maps to a ZRL band");
            encode_ac_zero_run(enc, zrl_band, &zrl_probs[zrl_band.index()], run as u32);

            // Decoder: `Prec = WasZero` after a zero run. `encoded_coeffs`
            // already advanced past the run to the next non-zero.
            prec = AcPrecContext::WasZero;
        } else {
            // A value coefficient at `pos`.
            let coeff = coeffs[pos];
            let band = AcBand::for_coefficient_position(pos).expect("scan pos maps to a band");
            let node_probs = &ac_probs[plane.index()][prec.index()][band.index()];
            let token = token_for_magnitude(coeff.unsigned_abs() as u16);
            encode_ac_token(enc, prec, pos, node_probs, token);
            encode_token_value(enc, token, coeff);

            // §13.3.1 Prec update: WasOne for magnitude 1, else
            // WasGreaterThanOne.
            prec = if coeff.unsigned_abs() == 1 {
                AcPrecContext::WasOne
            } else {
                AcPrecContext::WasGreaterThanOne
            };
            encoded_coeffs += 1;
        }
    }

    // After the last non-zero coefficient, terminate the block with EOB
    // unless the block is naturally full (last non-zero at position 63 →
    // `encoded_coeffs == 64`, the decoder's loop exits without an EOB).
    if encoded_coeffs < BLOCK_SIZE {
        let band =
            AcBand::for_coefficient_position(encoded_coeffs).expect("scan pos maps to a band");
        let node_probs = &ac_probs[plane.index()][prec.index()][band.index()];
        encode_ac_token(enc, prec, encoded_coeffs, node_probs, DctToken::EndOfBlock);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_decode::decode_block_coefficients;
    use crate::bool_coder::BoolCoder;
    use crate::dct_decode::{decode_ac_zero_run, decode_dc, decode_token_value};
    use crate::tokens::baseline_ac_probs;
    use crate::tokens::{baseline_dc_probs, dc_probs_to_node_contexts};
    use crate::zrl::ZERO_RUN_PROB_DEFAULTS as ZRL_DEFAULTS;
    use crate::zrl::ZERO_RUN_PROB_DEFAULTS;

    fn dc_node_probs() -> [u8; NUM_TREE_NODES] {
        dc_probs_to_node_contexts(&baseline_dc_probs())[0][0]
    }

    /// Token classification covers every Table 18 boundary.
    #[test]
    fn token_classification_boundaries() {
        assert_eq!(token_for_magnitude(0), DctToken::Zero);
        assert_eq!(token_for_magnitude(1), DctToken::One);
        assert_eq!(token_for_magnitude(4), DctToken::Four);
        assert_eq!(token_for_magnitude(5), DctToken::Category1);
        assert_eq!(token_for_magnitude(6), DctToken::Category1);
        assert_eq!(token_for_magnitude(7), DctToken::Category2);
        assert_eq!(token_for_magnitude(10), DctToken::Category2);
        assert_eq!(token_for_magnitude(11), DctToken::Category3);
        assert_eq!(token_for_magnitude(18), DctToken::Category3);
        assert_eq!(token_for_magnitude(19), DctToken::Category4);
        assert_eq!(token_for_magnitude(35), DctToken::Category5);
        assert_eq!(token_for_magnitude(67), DctToken::Category6);
        assert_eq!(token_for_magnitude(2114), DctToken::Category6);
    }

    /// Every DC coefficient across the full signed range round-trips:
    /// encode_dc → decode_dc recovers the exact value.
    #[test]
    fn dc_round_trips_full_range() {
        let probs = dc_node_probs();
        let values = [
            0, 1, -1, 2, -2, 4, -4, 5, -6, 7, -10, 11, -18, 19, -34, 35, -66, 67, -100, 500, -2114,
            2114,
        ];
        for &v in &values {
            let mut enc = BoolEncoder::new();
            encode_dc(&mut enc, &probs, v);
            let bytes = enc.finish();
            let mut bc = BoolCoder::new(&bytes).unwrap();
            let got = decode_dc(&mut bc, &probs).unwrap();
            assert_eq!(got, v, "DC {v} did not round-trip (got {got})");
        }
    }

    /// Token-value (magnitude + sign) round-trips for category tokens
    /// across their whole magnitude range.
    #[test]
    fn token_value_round_trips_categories() {
        for &mag in &[5u16, 6, 7, 10, 11, 18, 19, 34, 35, 66, 67, 1000, 2114] {
            for sign in [1i32, -1] {
                let coeff = sign * mag as i32;
                let token = token_for_magnitude(mag);
                let mut enc = BoolEncoder::new();
                encode_token_value(&mut enc, token, coeff);
                let bytes = enc.finish();
                let mut bc = BoolCoder::new(&bytes).unwrap();
                let got = decode_token_value(&mut bc, token).unwrap();
                assert_eq!(got, coeff, "token {token} value {coeff} round-trip");
            }
        }
    }

    /// Full block round-trip: a hand-built scan-order coefficient block
    /// (DC + scattered AC + zero runs) encodes and decodes back exactly.
    fn assert_block_round_trips(coeffs: [i32; BLOCK_SIZE], plane: AcPlane) {
        let dc_probs = dc_node_probs();
        let ac = baseline_ac_probs();
        let mut enc = BoolEncoder::new();
        encode_block_coefficients(&mut enc, plane, &dc_probs, &ac, &ZRL_DEFAULTS, &coeffs);
        let bytes = enc.finish();
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_coefficients(&mut bc, plane, &dc_probs, &ac, &ZRL_DEFAULTS)
            .expect("decode");
        assert_eq!(
            block.coeffs, coeffs,
            "block coefficients did not round-trip"
        );
    }

    #[test]
    fn empty_block_round_trips() {
        assert_block_round_trips([0i32; BLOCK_SIZE], AcPlane::Y);
    }

    #[test]
    fn dc_only_block_round_trips() {
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = 17;
        assert_block_round_trips(c, AcPlane::Y);
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = -200;
        assert_block_round_trips(c, AcPlane::UV);
    }

    #[test]
    fn scattered_ac_block_round_trips() {
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = 5;
        c[1] = -3;
        c[2] = 1;
        c[7] = 12; // gap of 4 zeros (positions 3..=6) before this
        c[20] = -1;
        c[40] = 80; // category6
        assert_block_round_trips(c, AcPlane::Y);
    }

    #[test]
    fn leading_zero_run_block_round_trips() {
        // DC then a long zero run before the first AC value.
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = 1;
        c[30] = 4; // 29 leading AC zeros (positions 1..=29)
        assert_block_round_trips(c, AcPlane::UV);
    }

    #[test]
    fn full_block_to_position_63_round_trips() {
        // Last non-zero at scan position 63 → no EOB coded.
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = 2;
        for (i, slot) in c.iter_mut().enumerate().skip(1) {
            *slot = if i % 2 == 0 { 1 } else { -1 };
        }
        c[63] = 3;
        assert_block_round_trips(c, AcPlane::Y);
    }

    #[test]
    fn zero_dc_with_ac_round_trips() {
        // DC == 0 seeds Prec = WasZero; the first AC step still emits the
        // ZERO/value bits because encoded_coeffs == 1 (implicit-1 needs
        // > 1).
        let mut c = [0i32; BLOCK_SIZE];
        c[0] = 0;
        c[1] = 7;
        c[5] = -2;
        assert_block_round_trips(c, AcPlane::Y);
    }

    /// Zero-run lengths round-trip for both bands across the literal and
    /// escape ranges.
    #[test]
    fn zero_run_round_trips() {
        for band in [ZrlBand::Band0, ZrlBand::Band1] {
            let probs = &ZERO_RUN_PROB_DEFAULTS[band.index()];
            for run in [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 72] {
                let mut enc = BoolEncoder::new();
                encode_ac_zero_run(&mut enc, band, probs, run);
                let bytes = enc.finish();
                let mut bc = BoolCoder::new(&bytes).unwrap();
                let got = decode_ac_zero_run(&mut bc, band, probs).unwrap();
                assert_eq!(got, run, "band {band:?} run {run} round-trip (got {got})");
            }
        }
    }
}

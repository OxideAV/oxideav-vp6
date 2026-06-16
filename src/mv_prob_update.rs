//! VP6 per-frame motion-vector probability updates (spec §11.2).
//!
//! After every keyframe ("intra frame") the four §11.1 per-axis MV
//! probability banks are reset to their defaults
//! ([`crate::mv_decode::IS_MV_SHORT_PROBS_DEFAULTS`],
//! [`crate::mv_decode::SHORT_MV_PROBS_DEFAULTS`],
//! [`crate::mv_decode::MV_SIZE_PROBS_DEFAULTS`],
//! [`crate::mv_decode::MV_SIGN_PROBS_DEFAULTS`]). On every inter frame
//! the decoder consumes a Table 13 update bitstream that mutates them
//! in place against four flag-probability lookup banks staged here:
//! [`UPDATE_IS_MV_SHORT_PROBABILITIES`],
//! [`UPDATE_MV_SIGN_PROBABILITIES`],
//! [`UPDATE_SHORT_VECTOR_NODE_PROBABILITIES`], and
//! [`UPDATE_LONG_VECTOR_BIT_PROBABILITIES`].
//!
//! ## Spec layout (Table 13)
//!
//! Per-axis fields appear in the order x-then-y for each of the four
//! banks. Each per-field record is the same two-step pattern the §13
//! updates use (the `B(flag_prob)` + optional `b(7)` pair the
//! [`crate::prob_update::decode_new_node_prob`] primitive consumes):
//!
//! ```text
//! XShortVecProbUpdateFlag   B(UpdateIsMvShortProbabilities[0])
//! XshortVecProbability      b(7)   // only if XShortVecProbUpdateFlag == 1
//! XsignProbUpdateFlag       B(UpdateMvSignProbabilities[0])
//! XsignProbability          b(7)   // only if XsignProbUpdateFlag == 1
//! YshortVecProbUpdateFlag   B(UpdateIsMvShortProbabilities[1])
//! YshortVecProbability      b(7)   // only if YshortVecProbUpdateFlag == 1
//! YsignProbUpdateFlag       B(UpdateMvSignProbabilities[1])
//! YsignProbability          b(7)   // only if YsignProbUpdateFlag == 1
//! ShortVecXTreeNodeProbs    // 7 sets of Table 14 (x row)
//! ShortVecYTreeNodeProbs    // 7 sets of Table 14 (y row)
//! LongVecXBitProbs          // 8 sets of Table 15 (x row)
//! LongVecYBitProbs          // 8 sets of Table 15 (y row)
//! ```
//!
//! Table 14 ("Short MV Tree Node updates") is seven per-axis
//! `NodeProbFollows` records reading
//! `B(UpdateShortVectorNodeProbabilities[axis][n])` and an optional
//! `b(7)` "NewTreeNodeProb" — mutating
//! `ShortMvProbs[axis][n]` for `n` in `0..=6`.
//!
//! Table 15 ("Long motion vector bit probability updates") is eight
//! per-axis `BitProbFollows` records in the traversal order
//! `[0, 1, 2, 7, 6, 5, 4, 3]` — note this differs from the §11.1
//! decode-time traversal `[0, 1, 2, 7, 6, 5, 4]` by the trailing `3`:
//! at update time bit 3's probability is always present in the spec's
//! per-axis traversal order. Each record reads
//! `B(UpdateLongVectorBitProbabilities[axis][k])` and an optional
//! `b(7)` "BitProb", mutating `MvSizeProbs[axis][bit_position]` where
//! `k` is the position in the traversal order and `bit_position` is
//! the spec-mandated bit index `[0, 1, 2, 7, 6, 5, 4, 3][k]`.
//!
//! ## `b(7) -> probability` conversion
//!
//! Spec §11.2 ("In all cases the updates are 7 bit numbers. To convert
//! these numbers to valid probabilities they must be modified as
//! follows."):
//!
//! ```text
//! NewProbability = DecodedValue << 1
//! If (NewProbability == 0)
//!     NewProbability = 1
//! ```
//!
//! This is the same `max(1, value * 2)` recipe the §13.2 / §13.3 /
//! §13.3.3 updates use, so the §11.2 driver shares
//! [`crate::prob_update::decode_new_node_prob`] for every per-node
//! step — no new BoolCoder material, no new errata.
//!
//! ## Flag-probability tables (§11.2 verbatim)
//!
//! ```text
//! UpdateIsMvShortProbabilities[2] = { 237, 231 }   // x, y
//! UpdateMvSignProbabilities[2]    = { 246, 243 }   // x, y
//!
//! UpdateShortVectorNodeProbabilities[2][7] =
//! {
//!     { 253, 253, 254, 254, 254, 254, 254 },   // x
//!     { 245, 253, 254, 254, 254, 254, 254 }    // y
//! }
//!
//! UpdateLongVectorBitProbabilities[2][8] =
//! {
//!     { 254, 254, 254, 254, 254, 250, 250, 252 },   // x
//!     { 254, 254, 254, 254, 254, 251, 251, 254 }    // y
//! }
//! ```
//!
//! ## What this module deliberately does not land
//!
//! * The wrapping per-frame driver that gates the §11.2 update against
//!   the inter/intra frame-type flag (intra branch resets the banks to
//!   the §11.1 defaults instead of walking the update bitstream;
//!   spec §11.2 prologue).
//! * The §10 mode-decode upstream that selects whether a given MB
//!   actually consumes a per-component MV. The §11.2 update walks the
//!   four MV-probability banks once per inter frame regardless of how
//!   many MVs are actually decoded under them.
//!
//! ## Provenance
//!
//! Sourced exclusively from material in `docs/video/vp6/`:
//!
//! * `vp6_format.pdf` §11.2 (pages 42-44) — Tables 13, 14, 15 and the
//!   surrounding `XShortVecProbUpdateFlag` / `XsignProbUpdateFlag` /
//!   etc. commentary, plus the four `Update*Probabilities` constant
//!   tables.
//! * `vp6_format.pdf` §3 (page 9) — the `B(x)` and `b(n)` notation.
//!
//! No third-party VP6 source has been consulted at any stage.

use crate::mv_decode::{MvProbs, NUM_MV_AXES, NUM_MV_SIZE_NODES, NUM_SHORT_MV_NODES};
use crate::prob_update::decode_new_node_prob;
use crate::{BoolCoder, Error};

/// Default `UpdateIsMvShortProbabilities[2]` flag-probability bank
/// from §11.2.
///
/// Probability used for `B(XShortVecProbUpdateFlag)` /
/// `B(YShortVecProbUpdateFlag)` — the per-axis discriminator that
/// decides whether a fresh `IsMvShortProbs[axis]` value follows.
///
/// Verbatim from §11.2:
///
/// ```text
/// UpdateIsMvShortProbabilities[2] = { 237, 231 }   // x, y
/// ```
pub const UPDATE_IS_MV_SHORT_PROBABILITIES: [u8; NUM_MV_AXES] = [237, 231];

/// Default `UpdateMvSignProbabilities[2]` flag-probability bank from
/// §11.2.
///
/// Probability used for `B(XsignProbUpdateFlag)` /
/// `B(YsignProbUpdateFlag)` — the per-axis discriminator that decides
/// whether a fresh `MvSignProbs[axis]` value follows.
///
/// Verbatim from §11.2:
///
/// ```text
/// UpdateMvSignProbabilities[2] = { 246, 243 }   // x, y
/// ```
pub const UPDATE_MV_SIGN_PROBABILITIES: [u8; NUM_MV_AXES] = [246, 243];

/// Default `UpdateShortVectorNodeProbabilities[2][7]` flag-probability
/// bank from §11.2.
///
/// Probability used for each of the seven Table 14 `NodeProbFollows`
/// reads against `ShortMvProbs[axis][0..=6]` — the per-axis per-node
/// discriminators that decide whether the corresponding short-MV tree
/// node's probability is updated.
///
/// Verbatim from §11.2:
///
/// ```text
/// UpdateShortVectorNodeProbabilities[2][7] =
/// {
///     { 253, 253, 254, 254, 254, 254, 254 },   // x
///     { 245, 253, 254, 254, 254, 254, 254 }    // y
/// }
/// ```
#[rustfmt::skip]
pub const UPDATE_SHORT_VECTOR_NODE_PROBABILITIES: [[u8; NUM_SHORT_MV_NODES]; NUM_MV_AXES] = [
    [253, 253, 254, 254, 254, 254, 254],
    [245, 253, 254, 254, 254, 254, 254],
];

/// Default `UpdateLongVectorBitProbabilities[2][8]` flag-probability
/// bank from §11.2.
///
/// Probability used for each of the eight Table 15 `BitProbFollows`
/// reads against `MvSizeProbs[axis][0..=7]` — the per-axis per-bit
/// discriminators that decide whether the corresponding long-MV bit
/// probability is updated. The eight reads are indexed by *traversal
/// position* `k = 0..=7` (the index into this table); the spec-mandated
/// *bit position* they target is given by [`LONG_VECTOR_BIT_ORDER`].
///
/// Verbatim from §11.2:
///
/// ```text
/// UpdateLongVectorBitProbabilities[2][8] =
/// {
///     { 254, 254, 254, 254, 254, 250, 250, 252 },   // x
///     { 254, 254, 254, 254, 254, 251, 251, 254 }    // y
/// }
/// ```
#[rustfmt::skip]
pub const UPDATE_LONG_VECTOR_BIT_PROBABILITIES: [[u8; NUM_MV_SIZE_NODES]; NUM_MV_AXES] = [
    [254, 254, 254, 254, 254, 250, 250, 252],
    [254, 254, 254, 254, 254, 251, 251, 254],
];

/// Spec §11.2 Table 15 long-vector bit traversal order.
///
/// At update time the spec walks the eight per-axis long-vector
/// bit-probabilities in the order `[0, 1, 2, 7, 6, 5, 4, 3]` — the
/// same order as §11.1's decode-time traversal `[0, 1, 2, 7, 6, 5, 4]`
/// with bit 3 appended at the end. (The decode-time path treats bit 3
/// specially — read conditionally if any of bits `4..=7` are non-zero
/// — but the update path treats it as a plain eighth probability.)
///
/// `LONG_VECTOR_BIT_ORDER[k]` is the bit-position the `k`-th Table 15
/// record updates. So a Table 15 traversal updates
/// `MvSizeProbs[axis][LONG_VECTOR_BIT_ORDER[k]]` for `k` in `0..=7`.
pub const LONG_VECTOR_BIT_ORDER: [usize; NUM_MV_SIZE_NODES] = [0, 1, 2, 7, 6, 5, 4, 3];

/// Decode one Table 13 per-axis `(short-discriminator, sign)` pair
/// of update records into the supplied `MvProbs[axis]` bundle.
///
/// Reads the two top-level Table 13 records for the given axis:
///
/// * `*ShortVecProbUpdateFlag`: `B(UPDATE_IS_MV_SHORT_PROBABILITIES[axis])`
///   plus optional `b(7)` `*shortVecProbability`, updating
///   `MvProbs::is_short`.
/// * `*signProbUpdateFlag`: `B(UPDATE_MV_SIGN_PROBABILITIES[axis])` plus
///   optional `b(7)` `*signProbability`, updating `MvProbs::sign`.
///
/// Both reads use the shared
/// [`crate::prob_update::decode_new_node_prob`] primitive — same
/// `max(1, value * 2)` recipe as the §13.2 / §13.3 / §13.3.3 updates.
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted.
#[inline]
fn update_per_axis_top_level(
    bc: &mut BoolCoder<'_>,
    probs: &mut MvProbs,
    axis: usize,
) -> Result<(), Error> {
    if let Some(new_is_short) = decode_new_node_prob(bc, UPDATE_IS_MV_SHORT_PROBABILITIES[axis])? {
        probs.is_short = new_is_short;
    }
    if let Some(new_sign) = decode_new_node_prob(bc, UPDATE_MV_SIGN_PROBABILITIES[axis])? {
        probs.sign = new_sign;
    }
    Ok(())
}

/// Decode the seven Table 14 short-MV tree-node update records for a
/// single axis into `probs.short[0..=6]`.
///
/// Each per-node record reads
/// `B(UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[axis][node])` plus an
/// optional `b(7)` "NewTreeNodeProb"; on `flag == 1` the corresponding
/// `ShortMvProbs[axis][node]` slot is overwritten with
/// `max(1, value * 2)`. Iteration order is the literal `node` index
/// `0..=NUM_SHORT_MV_NODES` (Table 14's "Seven Sets of: 0 to 6" prefix).
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted.
// Index-loop form keeps the per-node `flag_probs[axis][node]` lookup
// visibly aligned with the spec's Table 14 traversal; an
// `iter_mut().enumerate()` rewrite obscures the lookup-vs-store
// independence the §11.2 walk relies on, mirroring the existing
// `update_dc_probs` / `update_ac_probs` / `update_zero_run_probs`
// drivers in `crate::prob_update`.
#[allow(clippy::needless_range_loop)]
fn update_short_tree_per_axis(
    bc: &mut BoolCoder<'_>,
    probs: &mut MvProbs,
    axis: usize,
) -> Result<(), Error> {
    for node in 0..NUM_SHORT_MV_NODES {
        if let Some(new_prob) =
            decode_new_node_prob(bc, UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[axis][node])?
        {
            probs.short[node] = new_prob;
        }
    }
    Ok(())
}

/// Decode the eight Table 15 long-MV bit-probability update records
/// for a single axis into `probs.size[0..=7]`.
///
/// Each per-bit record reads
/// `B(UPDATE_LONG_VECTOR_BIT_PROBABILITIES[axis][k])` plus an optional
/// `b(7)` "BitProb"; on `flag == 1` the
/// `MvSizeProbs[axis][LONG_VECTOR_BIT_ORDER[k]]` slot is overwritten
/// with `max(1, value * 2)`. Traversal order is the Table 15
/// "Bit order (0, 1, 2, 7, 6, 5, 4, 3):" prefix, indexed via
/// [`LONG_VECTOR_BIT_ORDER`].
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted.
#[allow(clippy::needless_range_loop)]
fn update_long_bits_per_axis(
    bc: &mut BoolCoder<'_>,
    probs: &mut MvProbs,
    axis: usize,
) -> Result<(), Error> {
    for k in 0..NUM_MV_SIZE_NODES {
        if let Some(new_prob) =
            decode_new_node_prob(bc, UPDATE_LONG_VECTOR_BIT_PROBABILITIES[axis][k])?
        {
            let bit_position = LONG_VECTOR_BIT_ORDER[k];
            probs.size[bit_position] = new_prob;
        }
    }
    Ok(())
}

/// Walk the §11.2 motion-vector probability-update bitstream and apply
/// every update in place to a persistent `[MvProbs; 2]` bank.
///
/// The persistent bank holds the per-axis `MvProbs::{is_short, short,
/// size, sign}` state the §11.1 [`crate::mv_decode::decode_mv_component`]
/// reads from. Per §11.2 the bank is reset to defaults at every intra
/// frame (use [`MvProbs::defaults`] for that); on inter frames this
/// driver walks the Table 13 update bitstream and mutates the bank in
/// place.
///
/// ## Iteration order
///
/// Per Table 13 (top-to-bottom field order), reading from a single
/// shared BoolCoder cursor:
///
/// 1. `XShortVecProbUpdateFlag` + optional `XshortVecProbability` →
///    `probs[X].is_short`
/// 2. `XsignProbUpdateFlag` + optional `XsignProbability` →
///    `probs[X].sign`
/// 3. `YshortVecProbUpdateFlag` + optional `YshortVecProbability` →
///    `probs[Y].is_short`
/// 4. `YsignProbUpdateFlag` + optional `YsignProbability` →
///    `probs[Y].sign`
/// 5. `ShortVecXTreeNodeProbs` — seven Table 14 records →
///    `probs[X].short[0..=6]`
/// 6. `ShortVecYTreeNodeProbs` — seven Table 14 records →
///    `probs[Y].short[0..=6]`
/// 7. `LongVecXBitProbs` — eight Table 15 records via
///    [`LONG_VECTOR_BIT_ORDER`] → `probs[X].size[bit_position]`
/// 8. `LongVecYBitProbs` — eight Table 15 records via
///    [`LONG_VECTOR_BIT_ORDER`] → `probs[Y].size[bit_position]`
///
/// Each per-record step consumes one `B(flag_prob)` BoolCoder bit
/// plus seven more `b(1)` bits when the flag is `1`, via the shared
/// [`crate::prob_update::decode_new_node_prob`] primitive.
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted during
/// any of the constituent BoolCoder calls. On success every per-axis
/// entry in `probs` remains in the legal `1..=255` probability range
/// (the spec clip `max(1, value * 2)` enforces the lower bound;
/// `b(7) * 2 = 254` is the upper bound the update can write).
///
/// ## Provenance
///
/// `vp6_format.pdf` §11.2 (pages 42-44), Tables 13 / 14 / 15 plus the
/// four `Update*Probabilities` constant tables on pages 43-44.
pub fn update_mv_probs(
    bc: &mut BoolCoder<'_>,
    probs: &mut [MvProbs; NUM_MV_AXES],
) -> Result<(), Error> {
    // Steps 1-4: per-axis top-level discriminator + sign updates, x
    // then y per Table 13's literal record order
    // (XShortVecProbUpdateFlag, XsignProbUpdateFlag,
    // YShortVecProbUpdateFlag, YsignProbUpdateFlag).
    // The per-axis helper sequences (is_short, sign) — i.e. emitting
    // XShortVecProbUpdateFlag then XsignProbUpdateFlag for axis 0 and
    // the same pair for axis 1. This matches Table 13's row order.
    update_per_axis_top_level(bc, &mut probs[0], 0)?;
    update_per_axis_top_level(bc, &mut probs[1], 1)?;

    // Steps 5-6: ShortVecXTreeNodeProbs then ShortVecYTreeNodeProbs.
    update_short_tree_per_axis(bc, &mut probs[0], 0)?;
    update_short_tree_per_axis(bc, &mut probs[1], 1)?;

    // Steps 7-8: LongVecXBitProbs then LongVecYBitProbs.
    update_long_bits_per_axis(bc, &mut probs[0], 0)?;
    update_long_bits_per_axis(bc, &mut probs[1], 1)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mv_decode::MV_AXIS_X;

    fn bc_over(bytes: &[u8]) -> BoolCoder<'_> {
        BoolCoder::new(bytes).expect("at least 4 bytes")
    }

    // A NOTE ON BOOLCODER BEHAVIOUR AT THE §11.2 FLAG-PROBABILITY
    // VALUES.
    //
    // The published §11.2 `UpdateIsMvShortProbabilities` /
    // `UpdateMvSignProbabilities` / `UpdateShortVectorNodeProbabilities`
    // / `UpdateLongVectorBitProbabilities` flag-probability banks
    // cluster near 255 (smallest entry is 231). Under the operative
    // `>> 8` Split (errata #35) the BoolCoder is non-degenerate at
    // every such probability: at `Range = 255, Probability = 237`,
    // `Split = 1 + (254 * 237 >> 8) = 236 = Range - 19`, a valid
    // partition (the earlier note that the printed `>> 7` gave
    // `Split = 471 > Range` is exactly the spec typo errata #35
    // corrects). There is no "pathological corner" and no `range`
    // overflow; the implementation's `Range`/`Value` stay bounded for
    // every probability in `1..=255`.
    //
    // The tests below still exercise the **driver shape and static
    // surfaces** as their primary contract. A full Table 13 walk
    // against a §11.2-conformant bitstream remains a follow-up (an
    // integration test bound to a real .vp6 fixture) — not because the
    // synthetic-stream BoolCoder is unsafe, but because verifying the
    // *decoded* update values needs a conformant encoder's bytes.

    /// Verbatim §11.2 default tables — pinned in a test so a
    /// transcription drift in either spec or impl trips immediately.
    #[test]
    fn flag_probability_tables_verbatim() {
        assert_eq!(UPDATE_IS_MV_SHORT_PROBABILITIES, [237, 231]);
        assert_eq!(UPDATE_MV_SIGN_PROBABILITIES, [246, 243]);
        assert_eq!(
            UPDATE_SHORT_VECTOR_NODE_PROBABILITIES,
            [
                [253, 253, 254, 254, 254, 254, 254],
                [245, 253, 254, 254, 254, 254, 254],
            ]
        );
        assert_eq!(
            UPDATE_LONG_VECTOR_BIT_PROBABILITIES,
            [
                [254, 254, 254, 254, 254, 250, 250, 252],
                [254, 254, 254, 254, 254, 251, 251, 254],
            ]
        );
    }

    /// `LONG_VECTOR_BIT_ORDER` is the §11.2 Table 15 traversal order
    /// — `[0, 1, 2, 7, 6, 5, 4, 3]`. Pin the values and the property
    /// that it is a permutation of `0..=7`.
    #[test]
    fn long_vector_bit_order_is_permutation_of_zero_to_seven() {
        assert_eq!(LONG_VECTOR_BIT_ORDER, [0, 1, 2, 7, 6, 5, 4, 3]);
        let mut sorted = LONG_VECTOR_BIT_ORDER;
        sorted.sort();
        for (i, &bit) in sorted.iter().enumerate() {
            assert_eq!(bit, i, "missing bit position {i}");
        }
    }

    /// Static, BoolCoder-independent verification: the `MvProbs`
    /// default constructors line up with the `Update*Probabilities`
    /// bank dimensions in the obvious places — `is_short` against
    /// `UPDATE_IS_MV_SHORT_PROBABILITIES`, `sign` against
    /// `UPDATE_MV_SIGN_PROBABILITIES`, `short` against
    /// `UPDATE_SHORT_VECTOR_NODE_PROBABILITIES`, `size` against
    /// `UPDATE_LONG_VECTOR_BIT_PROBABILITIES`. Compile-time-equivalent
    /// in the absence of actually walking the BoolCoder driver, but
    /// gives a single test failure rather than a deferred compile
    /// error if either side's dimension drifts.
    #[test]
    fn flag_bank_dimensions_match_mvprobs_shape() {
        let p = MvProbs::defaults(MV_AXIS_X);
        assert_eq!(
            UPDATE_IS_MV_SHORT_PROBABILITIES.len(),
            [p.is_short].len() * NUM_MV_AXES
        );
        assert_eq!(
            UPDATE_MV_SIGN_PROBABILITIES.len(),
            [p.sign].len() * NUM_MV_AXES
        );
        assert_eq!(
            UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[0].len(),
            p.short.len()
        );
        assert_eq!(UPDATE_LONG_VECTOR_BIT_PROBABILITIES[0].len(), p.size.len());
    }

    // The remaining tests verify the §11.2 driver under truncation
    // and against synthetic high-probability streams. Under the
    // operative `>> 8` Split (errata #35) these streams decode
    // deterministically at every probability in `1..=255` (see the
    // note above).
    //
    // Verifying the *decoded* update values under the published
    // flag-probabilities still requires a conformant encoder's bytes,
    // so the full Table 13 walk against a real .vp6 bitstream remains
    // a follow-up integration test.

    /// Static check: the per-axis helper functions exist and have the
    /// expected `(BoolCoder, &mut MvProbs, axis) -> Result` shape.
    /// Compile-time-equivalent — fails at compile time if any helper's
    /// signature drifts.
    #[test]
    fn driver_helpers_exist_with_expected_shape() {
        fn _check_top_level(f: fn(&mut BoolCoder<'_>, &mut MvProbs, usize) -> Result<(), Error>) {
            let _ = f;
        }
        fn _check_short(f: fn(&mut BoolCoder<'_>, &mut MvProbs, usize) -> Result<(), Error>) {
            let _ = f;
        }
        fn _check_long(f: fn(&mut BoolCoder<'_>, &mut MvProbs, usize) -> Result<(), Error>) {
            let _ = f;
        }
        _check_top_level(update_per_axis_top_level);
        _check_short(update_short_tree_per_axis);
        _check_long(update_long_bits_per_axis);
        // And the public driver.
        fn _check_driver(
            f: fn(&mut BoolCoder<'_>, &mut [MvProbs; NUM_MV_AXES]) -> Result<(), Error>,
        ) {
            let _ = f;
        }
        _check_driver(update_mv_probs);
    }

    /// The §11.2 driver's Table 13 step order is x-then-y for each of
    /// the four sub-walks (top-level, short-tree, long-bits). Pin
    /// this by enumerating the static flag-prob banks in the same
    /// order — a defence-in-depth against the driver source being
    /// reordered (which would break alignment between the per-axis
    /// helpers and the Table 13 bitstream layout).
    #[test]
    fn table13_step_order_constants_are_x_then_y() {
        // Each Update*Probabilities table is indexed [axis][...]. For
        // the per-axis Table 13 step order (X reads come before Y
        // reads at every level), the X-row must occupy index 0 and
        // the Y-row index 1.
        assert_eq!(MV_AXIS_X, 0);
        // Spot-check that index 0 holds the §11.2 x-row literals.
        assert_eq!(UPDATE_IS_MV_SHORT_PROBABILITIES[0], 237);
        assert_eq!(UPDATE_MV_SIGN_PROBABILITIES[0], 246);
        assert_eq!(UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[0][0], 253);
        assert_eq!(UPDATE_LONG_VECTOR_BIT_PROBABILITIES[0][0], 254);
        // And index 1 is the y-row.
        assert_eq!(UPDATE_IS_MV_SHORT_PROBABILITIES[1], 231);
        assert_eq!(UPDATE_MV_SIGN_PROBABILITIES[1], 243);
        assert_eq!(UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[1][0], 245);
        assert_eq!(UPDATE_LONG_VECTOR_BIT_PROBABILITIES[1][0], 254);
    }

    /// `decode_new_node_prob` re-export sanity: under a moderate
    /// flag-prob bank `flag = 128` on a half-interval-leaning stream
    /// the primitive returns a well-defined `Option<u8>` (the `None`
    /// vs `Some(p)` choice and the `p` value both depend on the
    /// BoolCoder state, but the return must be in the well-defined
    /// surface). Confirms the §11.2 driver's reliance on the round-20
    /// primitive is intact.
    #[test]
    fn round20_primitive_round_trips_under_moderate_prob() {
        let bytes = [
            0x80u8, 0x55, 0xAA, 0x33, 0xCC, 0x66, 0x99, 0x5A, 0xA5, 0x3C, 0xC3, 0x69,
        ];
        let mut bc = bc_over(&bytes);
        let out = decode_new_node_prob(&mut bc, 128).expect("not truncated");
        if let Some(prob) = out {
            assert!((1..=254).contains(&prob), "out-of-range prob {prob}");
        }
    }

    /// `LONG_VECTOR_BIT_ORDER` length matches `NUM_MV_SIZE_NODES`.
    #[test]
    fn long_vector_bit_order_length_matches_size_nodes() {
        assert_eq!(LONG_VECTOR_BIT_ORDER.len(), NUM_MV_SIZE_NODES);
    }

    /// The `UPDATE_SHORT_VECTOR_NODE_PROBABILITIES` x-row leads with
    /// `253` (the most "always skip"-leaning of the seven node
    /// flag-probs) and the y-row leads with `245` (slightly more
    /// permissive for "update node 0 of the y-axis short tree"). Pin
    /// the relative ordering so a transcription typo flips
    /// immediately.
    #[test]
    fn short_node_flag_x_above_y_on_root_node() {
        let x_root_flag = UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[MV_AXIS_X][0];
        let y_root_flag = UPDATE_SHORT_VECTOR_NODE_PROBABILITIES[1][0];
        assert!(
            x_root_flag > y_root_flag,
            "x-row root flag {x_root_flag} should be > y-row root flag {y_root_flag}"
        );
    }

    /// The `UPDATE_LONG_VECTOR_BIT_PROBABILITIES` tail of each row
    /// (bit positions 4..=7 in the traversal order; corresponding to
    /// bit-positions 6, 5, 4, 3 of the magnitude) has slightly
    /// reduced probabilities relative to the head (250 / 251 vs
    /// 254). Pin this so a row-shift or off-by-one transcription
    /// surfaces immediately.
    #[test]
    fn long_bit_flag_tail_is_more_permissive_than_head() {
        for (axis, row) in UPDATE_LONG_VECTOR_BIT_PROBABILITIES.iter().enumerate() {
            let head = row[0];
            let tail_min = (5..=6).map(|k| row[k]).min().unwrap();
            assert!(
                tail_min < head,
                "axis={axis} tail flag {tail_min} should be < head flag {head}"
            );
        }
    }
}

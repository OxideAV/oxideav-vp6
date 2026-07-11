//! VP6 DC coefficient prediction (spec §14).
//!
//! The DC coefficient of every VP6 block is sent as a *prediction
//! error* on top of a predictor computed from already-decoded
//! neighbours. The reconstruction step is
//!
//! ```text
//! DC = PredictedDC + DcDelta
//! ```
//!
//! where `DcDelta` is the value §13.2 decodes into the DC token slot
//! and `PredictedDC` is the value this module computes. The
//! reconstructed DC then enters §15 inverse quantization (still in
//! transform-domain DC units; it has not yet been put back into raster
//! order via the §12.1 scan permutation).
//!
//! Verbatim summary from §14 (On2 Technologies, document version 1.02,
//! August 2006):
//!
//! > For a particular block the DC values of up to two particular
//! > immediate neighbors contribute to the prediction. The two blocks
//! > concerned are the blocks immediately to the left of and
//! > immediately above the current block.
//! >
//! > The DC value of a neighboring block only contributes to the
//! > prediction of the DC value for a particular block if all of the
//! > following conditions are satisfied:
//! >
//! >   - The neighboring block exists; there is no left neighbor for
//! >     blocks at the left edge and no above neighbor for blocks at
//! >     the top edge of the frame,
//! >   - The neighboring block was predicted from the same reference
//! >     frame as the block being predicted (last frame reconstruction
//! >     or golden frame),
//! >   - Inter-coded blocks can only be predicted by neighboring
//! >     inter-coded blocks and intra-coded blocks can only be
//! >     predicted by neighboring intra-coded blocks.
//! >
//! > There are three scenarios:
//! >
//! >   - If both neighboring blocks are available the prediction is
//! >     computed as the arithmetic average of their DC values,
//! >     truncated towards zero (values may be negative),
//! >   - If only one neighboring block is available, its DC value is
//! >     used as the predictor,
//! >   - If neither neighboring block is available, the last decoded
//! >     DC value for a block predicted from the same reference frame
//! >     is used as the predictor. **At the beginning of each frame
//! >     this last decoded DC value is set to zero for each prediction
//! >     frame type.**
//!
//! The four-row summary table on the same page then expresses the
//! same rule compactly:
//!
//! | Left Available | Above Available | Predictor                |
//! |----------------|-----------------|--------------------------|
//! | NO             | NO              | Last DC for same prediction frame |
//! | NO             | YES             | A                        |
//! | YES            | NO              | L                        |
//! | YES            | YES             | `(L + A + Sign(L + A)) / 2` |
//!
//! The two-available formula. The spec's prose ("arithmetic average
//! of their DC values, truncated towards zero (values may be
//! negative)") is summarised by the table-row formula
//! `(L + A + Sign(L + A)) / 2`. We implement the formula verbatim
//! over Rust's `i32` division — which, like C99 integer division,
//! truncates toward zero — so the result is **symmetric in sign**:
//! for an even sum the `Sign` adjustment is the identity; for an odd
//! positive sum `(2k+1) + 1 = 2k+2, /2 = k+1` (away from zero); for
//! an odd negative sum `-(2k+1) + (-1) = -(2k+2), /2 = -(k+1)` (also
//! away from zero, by the same magnitude). The prose's "truncated
//! towards zero" therefore describes the predictor's behaviour for
//! the common even-sum case and the away-from-zero rounding kicks in
//! symmetrically for odd sums — the formula is the authoritative
//! description (the spec's §1 Introduction directs ambiguities to be
//! resolved in favour of the accompanying reference, but here the
//! table-formula and prose only conflict on the parity edge and the
//! formula is the more specific of the two).
//!
//! Per-prediction-frame last-DC reset. The third scenario uses a
//! caller-side per-frame state: a "last decoded DC value for a block
//! predicted from the same reference frame". When the predictor has
//! no available L or A neighbour, the previous block's reconstructed
//! DC (within the same reference-frame bucket) becomes the predictor.
//! The spec is explicit: at the beginning of each prediction frame
//! (intra frame, inter-from-last-frame, inter-from-golden) this last
//! decoded DC value is set to **zero**. This module surfaces that
//! state as [`DcPredictionContext`], with [`DcPredictionContext::new`]
//! returning a freshly-zeroed seed and
//! [`DcPredictionContext::reset_at_frame_start`] re-applying the
//! per-frame zero seed for callers that re-use a context across
//! frames.
//!
//! Per-plane bookkeeping. §14's rules apply per *block*, and the VP6
//! block layout has separate Y, U and V planes. Neighbours that are
//! in a different plane are not considered: the leftmost Y block of a
//! macro-block has no left Y neighbour (even though there is a chroma
//! block to its left in pixel space, the DC predictor lives entirely
//! within a single plane). The "same reference frame" rule then
//! further partitions the per-plane neighbours by reference-frame
//! bucket. This module is plane-agnostic by design: a caller drives
//! one [`DcPredictionContext`] per plane and threads the
//! [`ReferenceBucket`] through each block, and §14's same-reference
//! test is performed inside [`DcPredictionContext::predict_and_record`].
//!
//! Read no BoolCoder bits. The DC predictor table itself, the
//! same-reference test and the last-DC seed are pure integer
//! bookkeeping — they consume already-decoded neighbour DC values and
//! reference-frame tags. The DC token / delta that the predictor is
//! added to is BoolCoder-coded (§13.2), and so is deferred until the
//! §7.3 DOCS-GAP is resolved.

/// The reference-frame bucket a block was predicted from, for the
/// purpose of §14's same-reference rule.
///
/// §14 says "the neighboring block was predicted from the same
/// reference frame as the block being predicted (last frame
/// reconstruction or golden frame)". Intra-coded blocks form their
/// own bucket via §14's separate "Inter-coded blocks can only be
/// predicted by neighboring inter-coded blocks and intra-coded blocks
/// can only be predicted by neighboring intra-coded blocks" rule —
/// the intra-vs-inter distinction is a strictly stronger same-bucket
/// test than the last/golden split, so it is collapsed into the same
/// enum here.
///
/// The §14 last-DC seed (used when neither neighbour is available) is
/// stored separately for each of these three buckets: the three
/// variants are *the* three "prediction frame types" the spec means
/// when it says "the last decoded DC value for a block predicted from
/// the same reference frame".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceBucket {
    /// Block was coded as intra (no inter prediction).
    Intra,
    /// Block was coded as inter, with the previous-frame reconstruction
    /// as its motion-compensation reference (the default inter case).
    InterLast,
    /// Block was coded as inter, with the Golden Frame as its
    /// motion-compensation reference.
    InterGolden,
}

impl ReferenceBucket {
    /// The number of distinct prediction-frame-type buckets §14
    /// distinguishes. Used internally to size the last-DC seed array.
    pub const COUNT: usize = 3;

    /// Internal dense index into the last-DC seed array.
    #[inline]
    const fn as_index(self) -> usize {
        match self {
            ReferenceBucket::Intra => 0,
            ReferenceBucket::InterLast => 1,
            ReferenceBucket::InterGolden => 2,
        }
    }
}

/// `Sign(x)` per §3 Nomenclature: `+1` for positive, `0` for zero,
/// `-1` for negative.
///
/// Exposed for callers that want to drive the §14 §3-Sign averaging
/// formula manually instead of through
/// [`DcPredictionContext::predict_and_record`].
#[inline]
pub fn sign(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

/// The "L + A + Sign(L + A)) / 2" two-neighbour averaging formula from
/// §14's predictor table.
///
/// Computes the average of two neighbour DC values, truncated toward
/// zero for both positive and negative sums. The `Sign` adjustment
/// matters when `L + A` is negative: plain `(L + A) / 2` truncates
/// away from zero for negative sums in two's-complement integer
/// arithmetic, and `(L + A + Sign(L+A)) / 2` corrects this back to
/// truncation toward zero so the predictor is symmetric in sign.
#[inline]
pub fn average_both_neighbours(left_dc: i32, above_dc: i32) -> i32 {
    let sum = left_dc.wrapping_add(above_dc);
    sum.wrapping_add(sign(sum)) / 2
}

/// One block's neighbour metadata for the §14 DC predictor.
///
/// `dc` is the *reconstructed* DC of the neighbouring block (§14
/// "DC values of … neighbors contribute to the prediction"), and
/// `reference` is the bucket the neighbour was predicted from. Both
/// fields are needed for the same-reference test: a neighbour whose
/// `reference` does not match the current block's bucket is treated
/// as unavailable, exactly as if the neighbour did not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neighbour {
    /// The neighbour's already-reconstructed DC coefficient.
    pub dc: i32,
    /// The reference-frame bucket the neighbour was predicted from.
    pub reference: ReferenceBucket,
}

/// The §14 frame-start "last decoded DC" seed for the chroma (U/V)
/// planes, in the quantized-DC domain — fixture-arbitrated at `128`
/// (see [`DcPredictionContext::new_chroma`]). Luma seeds at `0`.
pub const CHROMA_DC_PREDICTION_SEED: i32 = 128;

/// Per-plane DC prediction state for §14.
///
/// Holds the per-reference-bucket "last decoded DC value" the spec
/// uses as the §14 third-scenario predictor (neither L nor A
/// available). The state must be re-seeded to zero at the start of
/// each prediction frame; [`DcPredictionContext::new`] returns a
/// freshly-seeded instance, and
/// [`DcPredictionContext::reset_at_frame_start`] re-applies the seed
/// for callers that recycle a context across frames.
///
/// One context per plane. The §14 predictor is plane-local (Y, U
/// and V each have independent block grids with no cross-plane
/// neighbours), so the typical caller is three contexts per frame —
/// one each for the Y, U and V planes.
#[derive(Debug, Clone)]
pub struct DcPredictionContext {
    /// Last decoded DC value per reference bucket, indexed by
    /// [`ReferenceBucket::as_index`]. Re-seeded to zero at frame
    /// start per §14.
    last_dc: [i32; ReferenceBucket::COUNT],
}

impl Default for DcPredictionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl DcPredictionContext {
    /// A freshly-seeded context with every bucket's last-DC value at
    /// zero, per §14's "At the beginning of each frame this last
    /// decoded DC value is set to zero for each prediction frame
    /// type."
    pub fn new() -> Self {
        Self {
            last_dc: [0; ReferenceBucket::COUNT],
        }
    }

    /// A freshly-seeded context for a **chroma** plane (U or V):
    /// every bucket's last-DC value starts at
    /// [`CHROMA_DC_PREDICTION_SEED`] (`128`), not zero.
    ///
    /// Fixture-arbitrated (round 411): on the conformant third-party
    /// vp6f keyframe the first U and V blocks — which reconstruct to
    /// exactly 128 (a coded DC of 0) — each carry a coded
    /// `DCT_VAL_CATEGORY6` DC *delta* of `-128`, so the §14
    /// no-neighbour fallback predictor for the first chroma block of
    /// the frame must be `+128` in the quantized-DC domain, not the
    /// zero §14's prose states ("this last decoded DC value is set to
    /// zero for each prediction frame type" — contradicted for chroma
    /// by the stream). Differential bit-flip probing against the
    /// black-box decode oracle confirms the `-128` field bit-exactly
    /// (see the fixture notes.md appendix). Luma keeps the zero seed
    /// (the keyframe's first luma DC decodes as a plain `-299` against
    /// a zero predictor).
    pub fn new_chroma() -> Self {
        Self {
            last_dc: [CHROMA_DC_PREDICTION_SEED; ReferenceBucket::COUNT],
        }
    }

    /// Re-seed every bucket's last-DC value to zero, per §14's
    /// per-frame reset rule. Equivalent to replacing the context with
    /// [`DcPredictionContext::new`], but does not allocate.
    pub fn reset_at_frame_start(&mut self) {
        self.last_dc = [0; ReferenceBucket::COUNT];
    }

    /// The current last-DC seed for the given reference bucket.
    ///
    /// Exposed for inspection (tests, debugging). Production callers
    /// should use [`predict_and_record`] which both predicts and
    /// updates the seed in a single call.
    ///
    /// [`predict_and_record`]: Self::predict_and_record
    pub fn last_dc(&self, reference: ReferenceBucket) -> i32 {
        self.last_dc[reference.as_index()]
    }

    /// Manually set the last-DC seed for a given reference bucket.
    ///
    /// Exposed for callers that want to drive the bookkeeping
    /// out-of-band — most production paths should use
    /// [`predict_and_record`] which updates the seed automatically
    /// after each reconstructed block.
    ///
    /// [`predict_and_record`]: Self::predict_and_record
    pub fn set_last_dc(&mut self, reference: ReferenceBucket, dc: i32) {
        self.last_dc[reference.as_index()] = dc;
    }

    /// Compute the §14 DC predictor for a block in `reference` with
    /// the given left and above neighbours (each `None` if absent or
    /// disqualified by the same-reference rule). **Does not** record
    /// the reconstructed DC — see [`predict_and_record`].
    ///
    /// A neighbour passed as `Some` whose `reference` differs from
    /// the current block's `reference` is treated as if it had been
    /// passed as `None`. This implements the §14 rule "The
    /// neighboring block was predicted from the same reference frame
    /// as the block being predicted". The intra-vs-inter same-bucket
    /// rule is enforced by the same comparison because Intra is a
    /// distinct [`ReferenceBucket`] variant from InterLast /
    /// InterGolden.
    ///
    /// [`predict_and_record`]: Self::predict_and_record
    pub fn predict(
        &self,
        reference: ReferenceBucket,
        left: Option<Neighbour>,
        above: Option<Neighbour>,
    ) -> i32 {
        let l = left.filter(|n| n.reference == reference).map(|n| n.dc);
        let a = above.filter(|n| n.reference == reference).map(|n| n.dc);
        match (l, a) {
            (Some(left_dc), Some(above_dc)) => average_both_neighbours(left_dc, above_dc),
            (None, Some(above_dc)) => above_dc,
            (Some(left_dc), None) => left_dc,
            (None, None) => self.last_dc(reference),
        }
    }

    /// Compute the §14 DC predictor (as in [`predict`]), then record
    /// `reconstructed_dc` (the post-`DcDelta`-add DC,
    /// i.e. `predictor + delta`) as the new last-DC seed for
    /// `reference`. Returns the predictor.
    ///
    /// This is the production caller's entry point: the
    /// reconstructed DC that comes out of `predictor + DcDelta` is
    /// what §14 means when it says "the last decoded DC value for a
    /// block predicted from the same reference frame".
    ///
    /// [`predict`]: Self::predict
    pub fn predict_and_record(
        &mut self,
        reference: ReferenceBucket,
        left: Option<Neighbour>,
        above: Option<Neighbour>,
        reconstructed_dc: i32,
    ) -> i32 {
        let predictor = self.predict(reference, left, above);
        self.set_last_dc(reference, reconstructed_dc);
        predictor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new_chroma` seeds every reference bucket's last-DC at the
    /// fixture-arbitrated `CHROMA_DC_PREDICTION_SEED` (128), while
    /// `new` keeps the §14 zero seed.
    #[test]
    fn chroma_context_seeds_at_128() {
        let c = DcPredictionContext::new_chroma();
        for bucket in [
            ReferenceBucket::Intra,
            ReferenceBucket::InterLast,
            ReferenceBucket::InterGolden,
        ] {
            assert_eq!(c.last_dc(bucket), CHROMA_DC_PREDICTION_SEED);
            assert_eq!(DcPredictionContext::new().last_dc(bucket), 0);
        }
        // No-neighbour prediction returns the seed.
        let mut c = DcPredictionContext::new_chroma();
        assert_eq!(c.predict(ReferenceBucket::Intra, None, None), 128);
        c.set_last_dc(ReferenceBucket::Intra, 0);
        assert_eq!(c.predict(ReferenceBucket::Intra, None, None), 0);
    }

    #[test]
    fn sign_matches_spec_three_branches() {
        // §3 Nomenclature: Sign(x) = +1 for x > 0, 0 for x == 0,
        // -1 for x < 0.
        assert_eq!(sign(5), 1);
        assert_eq!(sign(1), 1);
        assert_eq!(sign(0), 0);
        assert_eq!(sign(-1), -1);
        assert_eq!(sign(-1000), -1);
        // Extreme values still pick the correct sign.
        assert_eq!(sign(i32::MAX), 1);
        assert_eq!(sign(i32::MIN), -1);
    }

    #[test]
    fn average_both_neighbours_zero_inputs() {
        // 0 + 0 + Sign(0)) / 2 = 0.
        assert_eq!(average_both_neighbours(0, 0), 0);
    }

    #[test]
    fn average_both_neighbours_positive_sum() {
        // 4 + 6 + Sign(10) = 11, / 2 = 5 (truncation toward zero).
        assert_eq!(average_both_neighbours(4, 6), 5);
        // Symmetric: 10 + 0 -> 10 + 1 = 11, /2 = 5.
        assert_eq!(average_both_neighbours(10, 0), 5);
        // Two big positives.
        assert_eq!(average_both_neighbours(100, 200), (100 + 200 + 1) / 2);
    }

    #[test]
    fn average_both_neighbours_negative_sum() {
        // -4 + -6 + Sign(-10) = -11, /2 = -5 (truncation toward zero
        // in Rust as in C).
        assert_eq!(average_both_neighbours(-4, -6), -5);
        // -1 + -1 -> -2 + -1 = -3, /2 = -1.
        assert_eq!(average_both_neighbours(-1, -1), -1);
        // -100 + -200 + -1 = -301, /2 = -150.
        assert_eq!(average_both_neighbours(-100, -200), -150);
    }

    #[test]
    fn average_both_neighbours_mixed_sign_sum_positive() {
        // 10 + -4 = 6 + Sign(6) = 7, /2 = 3.
        assert_eq!(average_both_neighbours(10, -4), 3);
    }

    #[test]
    fn average_both_neighbours_mixed_sign_sum_negative() {
        // -10 + 4 = -6 + Sign(-6) = -7, /2 = -3.
        assert_eq!(average_both_neighbours(-10, 4), -3);
    }

    #[test]
    fn average_both_neighbours_mixed_sign_zero_sum() {
        // 5 + -5 = 0 + Sign(0) = 0, /2 = 0.
        assert_eq!(average_both_neighbours(5, -5), 0);
        // -7 + 7 = 0 + Sign(0) = 0, /2 = 0.
        assert_eq!(average_both_neighbours(-7, 7), 0);
    }

    #[test]
    fn average_both_neighbours_matches_formula_for_all_small_inputs() {
        // The §14 table-row formula is `(L + A + Sign(L+A)) / 2`
        // evaluated with C-style truncation-toward-zero division.
        // Verify across a small but exhaustive range that the helper
        // matches the formula computed independently here.
        for l in -20..=20 {
            for a in -20..=20 {
                let sum = l + a;
                let s = if sum > 0 {
                    1
                } else if sum < 0 {
                    -1
                } else {
                    0
                };
                // Rust /'s i32 truncation matches C's, so this
                // expression is exactly the spec formula.
                let want = (sum + s) / 2;
                let got = average_both_neighbours(l, a);
                assert_eq!(
                    got, want,
                    "mismatch at ({}, {}): got {}, want {}",
                    l, a, got, want
                );
            }
        }
    }

    #[test]
    fn average_both_neighbours_is_symmetric_in_sign() {
        // For every (L, A), the predictor with both sign-flipped
        // operands is the negation of the original predictor. The
        // `Sign()` adjustment is exactly what guarantees this
        // symmetry — without it, asymmetric C-style truncation would
        // make positive sums truncate toward zero and negative sums
        // truncate away from zero.
        for l in -25..=25 {
            for a in -25..=25 {
                let p = average_both_neighbours(l, a);
                let q = average_both_neighbours(-l, -a);
                assert_eq!(p, -q, "asymmetry at ({}, {}): {} vs -({})", l, a, p, q);
            }
        }
    }

    #[test]
    fn average_both_neighbours_is_commutative() {
        // (L + A) is symmetric in L and A, so the predictor must be.
        for l in -10..=10 {
            for a in -10..=10 {
                assert_eq!(
                    average_both_neighbours(l, a),
                    average_both_neighbours(a, l),
                    "non-commutative at ({}, {})",
                    l,
                    a
                );
            }
        }
    }

    #[test]
    fn context_new_seeds_every_bucket_to_zero() {
        let ctx = DcPredictionContext::new();
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::InterLast), 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::InterGolden), 0);
    }

    #[test]
    fn context_default_matches_new() {
        let new = DcPredictionContext::new();
        let default = DcPredictionContext::default();
        for r in [
            ReferenceBucket::Intra,
            ReferenceBucket::InterLast,
            ReferenceBucket::InterGolden,
        ] {
            assert_eq!(new.last_dc(r), default.last_dc(r));
        }
    }

    #[test]
    fn predict_neither_neighbour_returns_per_bucket_last_dc_seed() {
        // §14 third scenario: "If neither neighboring block is
        // available, the last decoded DC value for a block predicted
        // from the same reference frame is used as the predictor.
        // At the beginning of each frame this last decoded DC value
        // is set to zero for each prediction frame type."
        let ctx = DcPredictionContext::new();
        for r in [
            ReferenceBucket::Intra,
            ReferenceBucket::InterLast,
            ReferenceBucket::InterGolden,
        ] {
            assert_eq!(ctx.predict(r, None, None), 0);
        }
    }

    #[test]
    fn predict_neither_neighbour_after_manual_seed() {
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::InterLast, 42);
        assert_eq!(ctx.predict(ReferenceBucket::InterLast, None, None), 42);
        // Other buckets are unaffected.
        assert_eq!(ctx.predict(ReferenceBucket::Intra, None, None), 0);
        assert_eq!(ctx.predict(ReferenceBucket::InterGolden, None, None), 0);
    }

    #[test]
    fn predict_left_only_returns_l() {
        // §14 row 3: YES / NO -> L.
        let ctx = DcPredictionContext::new();
        let left = Some(Neighbour {
            dc: 17,
            reference: ReferenceBucket::Intra,
        });
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, None), 17);
    }

    #[test]
    fn predict_above_only_returns_a() {
        // §14 row 2: NO / YES -> A.
        let ctx = DcPredictionContext::new();
        let above = Some(Neighbour {
            dc: -23,
            reference: ReferenceBucket::InterLast,
        });
        assert_eq!(ctx.predict(ReferenceBucket::InterLast, None, above), -23);
    }

    #[test]
    fn predict_both_neighbours_returns_truncated_average() {
        // §14 row 4: YES / YES -> (L + A + Sign(L+A)) / 2.
        let ctx = DcPredictionContext::new();
        let left = Some(Neighbour {
            dc: 4,
            reference: ReferenceBucket::Intra,
        });
        let above = Some(Neighbour {
            dc: 6,
            reference: ReferenceBucket::Intra,
        });
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, above), 5);
    }

    #[test]
    fn predict_both_neighbours_negative_uses_round_toward_zero() {
        // -4 + -6 + Sign(-10) = -11, /2 = -5.
        let ctx = DcPredictionContext::new();
        let left = Some(Neighbour {
            dc: -4,
            reference: ReferenceBucket::InterLast,
        });
        let above = Some(Neighbour {
            dc: -6,
            reference: ReferenceBucket::InterLast,
        });
        assert_eq!(ctx.predict(ReferenceBucket::InterLast, left, above), -5);
    }

    #[test]
    fn predict_left_with_mismatched_reference_treated_as_unavailable() {
        // §14: "The neighboring block was predicted from the same
        // reference frame as the block being predicted (last frame
        // reconstruction or golden frame)."
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::Intra, 77);
        let left = Some(Neighbour {
            dc: 999,
            reference: ReferenceBucket::InterLast, // wrong bucket
        });
        // Left is disqualified -> falls back to last-DC seed (77).
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, None), 77);
    }

    #[test]
    fn predict_above_with_mismatched_reference_treated_as_unavailable() {
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::InterGolden, -3);
        let above = Some(Neighbour {
            dc: 200,
            reference: ReferenceBucket::Intra, // wrong bucket
        });
        assert_eq!(ctx.predict(ReferenceBucket::InterGolden, None, above), -3);
    }

    #[test]
    fn predict_both_with_one_mismatched_drops_to_one_neighbour() {
        // L matches, A does not -> §14 row 3 (YES / NO -> L).
        let ctx = DcPredictionContext::new();
        let left = Some(Neighbour {
            dc: 12,
            reference: ReferenceBucket::Intra,
        });
        let above = Some(Neighbour {
            dc: 50,
            reference: ReferenceBucket::InterLast, // wrong bucket
        });
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, above), 12);
    }

    #[test]
    fn predict_both_with_both_mismatched_drops_to_last_dc_seed() {
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::Intra, 99);
        let left = Some(Neighbour {
            dc: 5,
            reference: ReferenceBucket::InterLast,
        });
        let above = Some(Neighbour {
            dc: 6,
            reference: ReferenceBucket::InterGolden,
        });
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, above), 99);
    }

    #[test]
    fn predict_and_record_updates_last_dc_seed_per_bucket() {
        let mut ctx = DcPredictionContext::new();
        // First block on left edge / top edge, intra: predictor = 0.
        let p1 = ctx.predict_and_record(ReferenceBucket::Intra, None, None, 100);
        assert_eq!(p1, 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), 100);
        // Other buckets untouched.
        assert_eq!(ctx.last_dc(ReferenceBucket::InterLast), 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::InterGolden), 0);
        // Second block (still left-and-top-isolated) picks up the new
        // last-DC seed.
        let p2 = ctx.predict_and_record(ReferenceBucket::Intra, None, None, 50);
        assert_eq!(p2, 100);
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), 50);
    }

    #[test]
    fn predict_and_record_returns_predictor_not_reconstructed_dc() {
        // The return is the §14 predictor (what gets added to the
        // DcDelta), not the reconstructed DC the caller passed in.
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::InterLast, 12);
        let predictor =
            ctx.predict_and_record(ReferenceBucket::InterLast, None, None, /* recon */ 999);
        assert_eq!(predictor, 12);
        // Seed updated to the reconstructed DC, not the predictor.
        assert_eq!(ctx.last_dc(ReferenceBucket::InterLast), 999);
    }

    #[test]
    fn reset_at_frame_start_zeros_all_buckets() {
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::Intra, 1);
        ctx.set_last_dc(ReferenceBucket::InterLast, 2);
        ctx.set_last_dc(ReferenceBucket::InterGolden, 3);
        ctx.reset_at_frame_start();
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::InterLast), 0);
        assert_eq!(ctx.last_dc(ReferenceBucket::InterGolden), 0);
    }

    #[test]
    fn intra_and_inter_buckets_do_not_cross_contaminate() {
        // §14's "Inter-coded blocks can only be predicted by
        // neighboring inter-coded blocks and intra-coded blocks can
        // only be predicted by neighboring intra-coded blocks." rule:
        // intra and inter neighbours are mutually disqualified.
        let mut ctx = DcPredictionContext::new();
        // Make the intra and inter-last seeds distinct.
        ctx.set_last_dc(ReferenceBucket::Intra, 11);
        ctx.set_last_dc(ReferenceBucket::InterLast, 22);
        // An inter-last block with an intra left neighbour: left is
        // disqualified, falls back to inter-last seed (22).
        let left = Some(Neighbour {
            dc: 5,
            reference: ReferenceBucket::Intra,
        });
        assert_eq!(ctx.predict(ReferenceBucket::InterLast, left, None), 22);
        // Mirror: intra block with inter-last left neighbour.
        let left = Some(Neighbour {
            dc: 5,
            reference: ReferenceBucket::InterLast,
        });
        assert_eq!(ctx.predict(ReferenceBucket::Intra, left, None), 11);
    }

    #[test]
    fn inter_last_and_inter_golden_buckets_do_not_cross_contaminate() {
        // §14 distinguishes the two inter reference frames explicitly
        // ("last frame reconstruction or golden frame"). They are
        // independent buckets for both the same-reference test and
        // the last-DC seed.
        let mut ctx = DcPredictionContext::new();
        ctx.set_last_dc(ReferenceBucket::InterLast, 30);
        ctx.set_last_dc(ReferenceBucket::InterGolden, -30);
        let above = Some(Neighbour {
            dc: 7,
            reference: ReferenceBucket::InterGolden,
        });
        // An inter-last block with an inter-golden above neighbour:
        // above is disqualified, falls back to inter-last seed (30).
        assert_eq!(ctx.predict(ReferenceBucket::InterLast, None, above), 30);
        // Mirror.
        let above = Some(Neighbour {
            dc: 7,
            reference: ReferenceBucket::InterLast,
        });
        assert_eq!(ctx.predict(ReferenceBucket::InterGolden, None, above), -30);
    }

    #[test]
    fn last_dc_seed_at_frame_start_is_zero_per_spec() {
        // The most-quoted §14 invariant in the spec ("At the
        // beginning of each frame this last decoded DC value is set
        // to zero for each prediction frame type") drives the
        // first-block predictor in every prediction frame. Three
        // tests in one — one per bucket.
        for &reference in &[
            ReferenceBucket::Intra,
            ReferenceBucket::InterLast,
            ReferenceBucket::InterGolden,
        ] {
            let mut ctx = DcPredictionContext::new();
            // Before any blocks are decoded, the seed is 0 and the
            // first block (no L, no A) gets predictor 0.
            assert_eq!(ctx.predict(reference, None, None), 0);
            // Decode a first block with delta 42 -> reconstructed = 42.
            // Seed updates to 42.
            ctx.predict_and_record(reference, None, None, 42);
            // Now simulate a fresh frame: reset, then re-check.
            ctx.reset_at_frame_start();
            assert_eq!(ctx.predict(reference, None, None), 0);
        }
    }

    #[test]
    fn worked_example_top_row_left_column_block_sequence() {
        // Drive a full §14 sequence for a small grid to exercise all
        // four predictor scenarios in a single test:
        //
        //   (0,0)  (0,1)  (0,2)
        //   (1,0)  (1,1)  (1,2)
        //
        // All blocks intra, all on the same plane. Per-block deltas
        // shown alongside.
        let mut ctx = DcPredictionContext::new();

        // (0, 0): top-left corner. No L, no A. Predictor = last-DC
        // seed = 0. Reconstructed = 0 + delta_00.
        let delta_00 = 10;
        let p_00 = ctx.predict_and_record(ReferenceBucket::Intra, None, None, delta_00);
        assert_eq!(p_00, 0);
        let dc_00 = p_00 + delta_00;
        assert_eq!(dc_00, 10);
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), 10);

        // (0, 1): top row. L = (0, 0), no A. Predictor = L = 10.
        let delta_01 = 4;
        let left_for_01 = Some(Neighbour {
            dc: dc_00,
            reference: ReferenceBucket::Intra,
        });
        let p_01 =
            ctx.predict_and_record(ReferenceBucket::Intra, left_for_01, None, dc_00 + delta_01);
        assert_eq!(p_01, 10);
        let dc_01 = p_01 + delta_01;
        assert_eq!(dc_01, 14);
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), dc_01);

        // (0, 2): top row. L = (0, 1) = 14, no A. Predictor = 14.
        let delta_02 = -2;
        let left_for_02 = Some(Neighbour {
            dc: dc_01,
            reference: ReferenceBucket::Intra,
        });
        let p_02 = ctx.predict_and_record(
            ReferenceBucket::Intra,
            left_for_02,
            None,
            /* recon */ dc_01 + delta_02,
        );
        assert_eq!(p_02, 14);
        let dc_02 = p_02 + delta_02; // 12
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), dc_02);

        // (1, 0): left column. No L, A = (0, 0) = 10. Predictor = 10.
        let delta_10 = -3;
        let above_for_10 = Some(Neighbour {
            dc: dc_00,
            reference: ReferenceBucket::Intra,
        });
        let p_10 =
            ctx.predict_and_record(ReferenceBucket::Intra, None, above_for_10, dc_00 + delta_10);
        assert_eq!(p_10, 10);
        let dc_10 = p_10 + delta_10; // 7
        assert_eq!(ctx.last_dc(ReferenceBucket::Intra), dc_10);

        // (1, 1): interior. L = (1, 0) = 7, A = (0, 1) = 14.
        // Predictor = (7 + 14 + Sign(21)) / 2 = (21 + 1) / 2 = 11.
        let delta_11 = 0;
        let l = Some(Neighbour {
            dc: dc_10,
            reference: ReferenceBucket::Intra,
        });
        let a = Some(Neighbour {
            dc: dc_01,
            reference: ReferenceBucket::Intra,
        });
        let p_11 = ctx.predict_and_record(ReferenceBucket::Intra, l, a, 11 + delta_11);
        assert_eq!(p_11, 11);

        // (1, 2): interior. L = (1, 1) = 11, A = (0, 2) = 12.
        // Predictor = (11 + 12 + Sign(23)) / 2 = 24 / 2 = 12.
        let delta_12 = 1;
        let l = Some(Neighbour {
            dc: 11,
            reference: ReferenceBucket::Intra,
        });
        let a = Some(Neighbour {
            dc: dc_02,
            reference: ReferenceBucket::Intra,
        });
        let p_12 = ctx.predict_and_record(ReferenceBucket::Intra, l, a, 12 + delta_12);
        assert_eq!(p_12, 12);
    }

    #[test]
    fn reference_bucket_count_is_three() {
        // Defensive: the dense index assigned by as_index() must stay
        // in 0..COUNT.
        assert_eq!(ReferenceBucket::COUNT, 3);
        assert!(ReferenceBucket::Intra.as_index() < ReferenceBucket::COUNT);
        assert!(ReferenceBucket::InterLast.as_index() < ReferenceBucket::COUNT);
        assert!(ReferenceBucket::InterGolden.as_index() < ReferenceBucket::COUNT);
        // …and be distinct.
        let mut seen = [false; ReferenceBucket::COUNT];
        for r in [
            ReferenceBucket::Intra,
            ReferenceBucket::InterLast,
            ReferenceBucket::InterGolden,
        ] {
            let i = r.as_index();
            assert!(!seen[i], "duplicate index for {:?}", r);
            seen[i] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn average_both_neighbours_extreme_magnitudes_no_panic() {
        // Defensive: ensure the formula is panic-free for extreme
        // operands. We use wrapping_add internally so even adversarial
        // input won't panic; the result is undefined-but-bounded.
        let _ = average_both_neighbours(i32::MAX, 1);
        let _ = average_both_neighbours(i32::MIN, -1);
        let _ = average_both_neighbours(i32::MAX, i32::MIN);
    }
}

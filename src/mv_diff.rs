//! VP6 differential motion-vector reconstruction (spec §11 intro).
//!
//! The §11 intro paragraph (page 38) says:
//!
//! > New motion vectors are coded differentially with respect to the
//! > motion vector of the **nearest** MacroBlock that uses the same
//! > reference frame (either the previous frame reconstruction or the
//! > Golden frame), if such a MacroBlock exists and it is either
//! > immediately to the left of or immediately above the current
//! > MacroBlock. Otherwise, new motion vectors are coded absolutely
//! > (this can be thought of as differential coded with respect to the
//! > vector (0,0)).
//!
//! That paragraph governs three §10 coding modes — `CODE_INTER_PLUS_MV`,
//! `CODE_GOLDEN_MV`, and (per-block) `CODE_INTER_FOURMV` — which carry a
//! fresh §11.1-decoded delta component-pair. This module is the
//! BoolCoder-independent piece that combines that delta with the
//! correct reference MV to produce the final reconstructed vector.
//!
//! ## Two narrow concerns
//!
//! 1. **Reference-MV selection.** The spec ties the differential
//!    reference to the §10 definition of **nearest** (bold-faced in the
//!    spec, referring back to §10's "first non (0, 0) same-reference
//!    neighbour" predicate) but adds a geographic restriction: the
//!    "nearest" candidate counts as the differential reference *only* if
//!    it sits at offset `(-1, 0)` (the macroblock immediately above) or
//!    `(0, -1)` (the macroblock immediately to the left). These are
//!    exactly the first two entries of [`crate::modes::NEAR_MACROBLOCKS`].
//!    [`select_diff_reference_mv`] walks those two offsets in spec
//!    [`crate::modes::NEAR_MACROBLOCKS`] order, applies the same two §10
//!    predicates the [`crate::near_mv`] walker uses
//!    (`mv != (0, 0)` AND `reference == self.reference`), and returns
//!    the first qualifying neighbour's MV, or `(0, 0)` if neither
//!    qualifies.
//!
//! 2. **Delta application.** [`reconstruct_diff_mv`] is the
//!    per-component addition `final = reference + delta`, clamped to
//!    the §11 component bound of `[-127, 127]` ("The maximum magnitude
//!    of a MV component is 31 ¾ whole pixels (127 in units of ¼
//!    pixel)"). A conformant stream never produces a sum outside that
//!    bound; the clamp keeps a corrupt or desynchronised one from
//!    stepping past the §11.5 48-sample UMV border during §17
//!    reconstruction.
//!
//! Composing them, [`reconstruct_new_mv`] is the one-shot the §10
//! `CODE_INTER_PLUS_MV` / `CODE_GOLDEN_MV` paths consume per MB:
//! select reference, add delta, return final.
//!
//! ## What this module does **not** land
//!
//! * The §11.1 delta decode itself — that landed in round 21 as
//!   [`crate::mv_decode::decode_mv_pair`]. This module's input is a
//!   freshly-decoded delta [`MotionVector`].
//! * The §10 mode-decode that selects whether the current MB uses
//!   `CODE_INTER_PLUS_MV` / `CODE_GOLDEN_MV` / `CODE_INTER_FOURMV` or
//!   one of the seven implicit-MV modes — still a DOCS-GAP candidate
//!   around the `B(Stats[0])` / `B(Stats[2])` else-branch indentation
//!   on page 36.
//! * The §10 [`crate::near_mv`] full 12-neighbour walker — that
//!   resolves the §10 *Nearest / Near* alternative-MV pair the
//!   implicit-MV modes consume; the differential reference here is a
//!   strictly narrower 2-neighbour walk because of the "left or above
//!   only" geographic constraint the §11 intro adds.
//! * The §10 `CODE_INTER_FOURMV` per-block 2-bit codeword (Table 10)
//!   that picks one of the four implicit modes for each of the four
//!   luma blocks within a four-MV-coded MB — landed in round 23 as
//!   [`crate::fourmv::decode_fourmv_block_mode`].
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §11 intro
//! (On2 Technologies, document version 1.02, August 2006, pages 37-38).
//! No third-party VP6 implementation has been consulted at any stage.

use crate::dc_pred::ReferenceBucket;
use crate::near_mv::{MotionVector, NeighbourMv};

/// The two §11-restricted neighbour offsets, in spec traversal order.
///
/// `(-1, 0)` is the macroblock immediately above the current MB;
/// `(0, -1)` is the macroblock immediately to its left. These are
/// exactly the first two entries of
/// [`crate::modes::NEAR_MACROBLOCKS`]; the constant is re-exposed here
/// (and the [`above_left_offsets_match_near_macroblocks`] test pins it
/// against that table) so the geographic constraint of the §11 intro
/// stays locatable in code search.
///
/// [`above_left_offsets_match_near_macroblocks`]: tests::above_left_offsets_match_near_macroblocks
pub const DIFF_REFERENCE_OFFSETS: [(i8, i8); 2] = [(-1, 0), (0, -1)];

/// Walk the two §11-restricted neighbour offsets (above, left) against
/// a caller-supplied accessor and return the first qualifying
/// neighbour's MV, or [`MotionVector::ZERO`] if neither neighbour
/// qualifies.
///
/// A neighbour qualifies when it satisfies the two §10 predicates the
/// §11 intro inherits via the bold-faced **nearest** reference:
///
/// 1. `neighbour.reference == reference` (the "uses the same reference
///    frame" rule), and
/// 2. `!neighbour.mv.is_zero()` (the §10 "non (0, 0)" rule that the
///    bold-faced **nearest** terminology carries forward).
///
/// The walker visits the above neighbour first, then the left
/// neighbour; it returns the first qualifying MV. When neither
/// qualifies — because the accessor returns `None` for both positions,
/// because both neighbours code against a different reference frame,
/// or because both neighbours have `(0, 0)` MVs — the walker falls
/// back to `(0, 0)`, which per the §11 intro is the "coded absolutely"
/// equivalent of a zero differential reference.
///
/// # Parameters
///
/// * `row`, `col` — the current MB's `(row, col)` position. Combined
///   with each [`DIFF_REFERENCE_OFFSETS`] entry to form the absolute
///   neighbour position passed to `neighbour_at`. Signed `i32` to
///   accommodate off-frame `(row, col)` at the top-left corner.
/// * `reference` — the current MB's target prediction-frame bucket.
///   Used to filter neighbours by the same-reference rule.
/// * `neighbour_at` — closure that maps an absolute `(row, col)`
///   position to the neighbour's MV + reference, or `None` for
///   off-frame / not-yet-decoded positions.
#[inline]
pub fn select_diff_reference_mv<F>(
    row: i32,
    col: i32,
    reference: ReferenceBucket,
    mut neighbour_at: F,
) -> MotionVector
where
    F: FnMut(i32, i32) -> Option<NeighbourMv>,
{
    for (dr, dc) in DIFF_REFERENCE_OFFSETS {
        let r = row.wrapping_add(dr as i32);
        let c = col.wrapping_add(dc as i32);
        let Some(neigh) = neighbour_at(r, c) else {
            continue;
        };
        if neigh.reference != reference {
            continue;
        }
        if neigh.mv.is_zero() {
            continue;
        }
        return neigh.mv;
    }
    MotionVector::ZERO
}

/// Dense-grid convenience wrapper for [`select_diff_reference_mv`].
///
/// Backs the walker with a flat `&[Option<NeighbourMv>]` slice indexed
/// row-major as `grid[row * grid_width + col]`. Out-of-bounds
/// `(row, col)` access (negative coordinates, or `col >= grid_width`,
/// or `row * grid_width + col >= grid.len()`) returns `None` exactly
/// as if the underlying accessor had returned `None`, which folds into
/// the `(0, 0)` differential-reference fallback.
pub fn select_diff_reference_mv_from_grid(
    grid: &[Option<NeighbourMv>],
    grid_width: usize,
    row: i32,
    col: i32,
    reference: ReferenceBucket,
) -> MotionVector {
    select_diff_reference_mv(row, col, reference, |r, c| {
        if r < 0 || c < 0 {
            return None;
        }
        let (r, c) = (r as usize, c as usize);
        if c >= grid_width {
            return None;
        }
        let idx = r.checked_mul(grid_width).and_then(|x| x.checked_add(c))?;
        grid.get(idx).copied().flatten()
    })
}

/// Apply a §11.1-decoded delta to a differential reference MV, clamping
/// the sum to the §11 component bound.
///
/// Per-component addition `final = reference + delta`, then a clamp of
/// each component to `[-127, 127]` — §11's "The maximum magnitude of a
/// MV component is 31 ¾ whole pixels (127 in units of ¼ pixel)". The
/// §11.1 decoder produces delta components in `[-127, 127]` and the
/// reference is itself a §11-bounded MV, so the raw sum lives in
/// `[-254, 254]`; a *conformant* stream never lets it leave the §11
/// bound, but a corrupt or desynchronised one can, and an unclamped
/// out-of-bound vector would step past the §11.5 48-sample UMV border
/// during §17 reconstruction (an out-of-buffer fetch). The clamp keeps
/// every §17 fetch inside the bordered buffer for any input.
#[inline]
pub const fn reconstruct_diff_mv(reference: MotionVector, delta: MotionVector) -> MotionVector {
    const MV_COMPONENT_BOUND: i16 = 127;
    const fn clamp_component(v: i16) -> i16 {
        if v > MV_COMPONENT_BOUND {
            MV_COMPONENT_BOUND
        } else if v < -MV_COMPONENT_BOUND {
            -MV_COMPONENT_BOUND
        } else {
            v
        }
    }
    MotionVector::new(
        clamp_component(reference.x + delta.x),
        clamp_component(reference.y + delta.y),
    )
}

/// One-shot wrapper that selects the §11 differential reference MV
/// against the supplied grid accessor and applies the §11.1-decoded
/// delta in a single call.
///
/// Composes [`select_diff_reference_mv`] and [`reconstruct_diff_mv`]
/// in the spec-mandated order: reference selection runs first (using
/// only the supplied `(row, col, reference)` and the neighbour grid),
/// then the delta is added to the result. Returns the final
/// reconstructed motion vector ready for §17 motion compensation.
#[inline]
pub fn reconstruct_new_mv<F>(
    row: i32,
    col: i32,
    reference: ReferenceBucket,
    delta: MotionVector,
    neighbour_at: F,
) -> MotionVector
where
    F: FnMut(i32, i32) -> Option<NeighbourMv>,
{
    let reference_mv = select_diff_reference_mv(row, col, reference, neighbour_at);
    reconstruct_diff_mv(reference_mv, delta)
}

/// Dense-grid convenience wrapper for [`reconstruct_new_mv`]. Picks
/// the differential reference from a row-major slice grid and applies
/// `delta` in one call. Out-of-bounds slots are treated as `None` per
/// [`select_diff_reference_mv_from_grid`].
pub fn reconstruct_new_mv_from_grid(
    grid: &[Option<NeighbourMv>],
    grid_width: usize,
    row: i32,
    col: i32,
    reference: ReferenceBucket,
    delta: MotionVector,
) -> MotionVector {
    let reference_mv = select_diff_reference_mv_from_grid(grid, grid_width, row, col, reference);
    reconstruct_diff_mv(reference_mv, delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::NEAR_MACROBLOCKS;

    fn mv(x: i16, y: i16) -> MotionVector {
        MotionVector::new(x, y)
    }

    fn neigh(x: i16, y: i16, r: ReferenceBucket) -> NeighbourMv {
        NeighbourMv::new(mv(x, y), r)
    }

    /// The two §11-restricted offsets match the first two entries of
    /// the §10 [`NEAR_MACROBLOCKS`] table in the same order. If the
    /// §10 table is ever reordered, this assertion makes the §11
    /// geographic constraint trip immediately rather than silently
    /// drift.
    #[test]
    fn above_left_offsets_match_near_macroblocks() {
        assert_eq!(DIFF_REFERENCE_OFFSETS[0], NEAR_MACROBLOCKS[0]); // above
        assert_eq!(DIFF_REFERENCE_OFFSETS[1], NEAR_MACROBLOCKS[1]); // left
        assert_eq!(DIFF_REFERENCE_OFFSETS, [(-1, 0), (0, -1)]);
    }

    /// No neighbour anywhere → `(0, 0)` reference (the "coded
    /// absolutely" branch of the §11 intro).
    #[test]
    fn empty_neighbourhood_returns_zero_reference() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |_, _| None);
        assert_eq!(r, MotionVector::ZERO);
    }

    /// Single qualifying neighbour at the above slot → its MV is the
    /// differential reference.
    #[test]
    fn above_qualifying_neighbour_becomes_reference() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| {
            if (r, c) == (4, 5) {
                Some(neigh(10, -4, ReferenceBucket::InterLast))
            } else {
                None
            }
        });
        assert_eq!(r, mv(10, -4));
    }

    /// Single qualifying neighbour at the left slot → its MV is the
    /// differential reference.
    #[test]
    fn left_qualifying_neighbour_becomes_reference() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| {
            if (r, c) == (5, 4) {
                Some(neigh(-3, 7, ReferenceBucket::InterLast))
            } else {
                None
            }
        });
        assert_eq!(r, mv(-3, 7));
    }

    /// Both neighbours qualify → the **above** neighbour wins (it
    /// comes first in the spec traversal order, matching the first
    /// entry of [`NEAR_MACROBLOCKS`]).
    #[test]
    fn above_neighbour_wins_over_left_when_both_qualify() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(20, -10, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(-5, 5, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(r, mv(20, -10));
    }

    /// Above neighbour codes against a different reference frame; the
    /// walker skips it and falls through to the left neighbour. This
    /// pins the same-reference rule.
    #[test]
    fn different_reference_above_falls_through_to_left() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(99, 99, ReferenceBucket::InterGolden)),
            (5, 4) => Some(neigh(8, -8, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(r, mv(8, -8));
    }

    /// Above neighbour exists and matches reference but has `(0, 0)`
    /// MV — the §10 **nearest** definition requires non-(0, 0), so the
    /// walker falls through to the left neighbour.
    #[test]
    fn zero_mv_above_falls_through_to_left() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(0, 0, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(3, 4, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(r, mv(3, 4));
    }

    /// Both neighbours fail to qualify (one different-reference, one
    /// zero-MV) → the differential reference falls back to `(0, 0)`,
    /// which per the §11 intro is the "coded absolutely" branch.
    #[test]
    fn neither_qualifies_returns_zero() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(11, 22, ReferenceBucket::InterGolden)),
            (5, 4) => Some(neigh(0, 0, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(r, MotionVector::ZERO);
    }

    /// The §10 12-neighbour walker would discover a Nearest neighbour
    /// at offset `(-1, -1)` (the third [`NEAR_MACROBLOCKS`] entry).
    /// The §11 differential reference walker explicitly does NOT look
    /// past the above / left positions, so a qualifying neighbour at
    /// `(-1, -1)` is ignored and the reference falls back to `(0, 0)`.
    /// This pins the "left or above only" geographic constraint that
    /// distinguishes this module from the §10 walker.
    #[test]
    fn upper_left_diagonal_neighbour_is_ignored() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 4) => Some(neigh(50, -50, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(r, MotionVector::ZERO);
    }

    /// Top-left-corner MB at `(0, 0)`: both above `(-1, 0)` and left
    /// `(0, -1)` are off-frame; the accessor returns `None` for both
    /// and the walker falls back to `(0, 0)`.
    #[test]
    fn top_left_corner_falls_back_to_zero() {
        let r = select_diff_reference_mv(0, 0, ReferenceBucket::InterLast, |_, _| None);
        assert_eq!(r, MotionVector::ZERO);
    }

    /// Top-row MB at `(0, 5)`: above `(-1, 5)` is off-frame, only
    /// the left neighbour `(0, 4)` is visible. A qualifying left
    /// neighbour is picked.
    #[test]
    fn top_row_uses_left_only() {
        let r = select_diff_reference_mv(0, 5, ReferenceBucket::InterLast, |r, c| {
            if (r, c) == (0, 4) {
                Some(neigh(15, -7, ReferenceBucket::InterLast))
            } else {
                None
            }
        });
        assert_eq!(r, mv(15, -7));
    }

    /// Left-column MB at `(5, 0)`: left `(5, -1)` is off-frame, only
    /// the above neighbour `(4, 0)` is visible. A qualifying above
    /// neighbour is picked.
    #[test]
    fn left_column_uses_above_only() {
        let r = select_diff_reference_mv(5, 0, ReferenceBucket::InterLast, |r, c| {
            if (r, c) == (4, 0) {
                Some(neigh(-9, 11, ReferenceBucket::InterLast))
            } else {
                None
            }
        });
        assert_eq!(r, mv(-9, 11));
    }

    /// Pure addition: `reference + delta = final` per component.
    #[test]
    fn reconstruct_diff_mv_is_per_component_addition() {
        assert_eq!(reconstruct_diff_mv(mv(10, -5), mv(3, 8)), mv(13, 3));
        assert_eq!(reconstruct_diff_mv(mv(-12, 4), mv(-3, -1)), mv(-15, 3));
        assert_eq!(
            reconstruct_diff_mv(MotionVector::ZERO, mv(7, -2)),
            mv(7, -2)
        );
        assert_eq!(
            reconstruct_diff_mv(mv(15, -8), MotionVector::ZERO),
            mv(15, -8)
        );
        assert_eq!(
            reconstruct_diff_mv(MotionVector::ZERO, MotionVector::ZERO),
            MotionVector::ZERO
        );
    }

    /// At the §11.1 magnitude cap (`±127`) the sum is clamped back to
    /// the §11 component bound, so a corrupt stream whose
    /// reference+delta sum leaves the legal range cannot step past the
    /// §11.5 UMV border during reconstruction.
    #[test]
    fn maximum_magnitude_sum_stays_in_range() {
        let max_pos = mv(127, 127);
        let max_neg = mv(-127, -127);
        assert_eq!(reconstruct_diff_mv(max_pos, max_pos), mv(127, 127));
        assert_eq!(reconstruct_diff_mv(max_neg, max_neg), mv(-127, -127));
        assert_eq!(reconstruct_diff_mv(max_pos, max_neg), MotionVector::ZERO);
        assert_eq!(reconstruct_diff_mv(max_neg, max_pos), MotionVector::ZERO);
        // In-bound sums pass through unchanged.
        assert_eq!(
            reconstruct_diff_mv(mv(100, -100), mv(27, -27)),
            mv(127, -127)
        );
        assert_eq!(reconstruct_diff_mv(mv(3, -4), mv(10, 8)), mv(13, 4));
    }

    /// One-shot wrapper composes reference selection and delta
    /// application in the spec-mandated order: above neighbour wins,
    /// then delta is added.
    #[test]
    fn reconstruct_new_mv_composes_reference_and_delta() {
        let result =
            reconstruct_new_mv(5, 5, ReferenceBucket::InterLast, mv(3, -4), |r, c| {
                match (r, c) {
                    (4, 5) => Some(neigh(10, 8, ReferenceBucket::InterLast)),
                    _ => None,
                }
            });
        // reference (10, 8) + delta (3, -4) = (13, 4)
        assert_eq!(result, mv(13, 4));
    }

    /// Absolute-coded fallback: when no qualifying neighbour exists,
    /// the differential reference is `(0, 0)` and the result is just
    /// the delta. Pins the "coded absolutely" branch end-to-end.
    #[test]
    fn reconstruct_new_mv_falls_back_to_delta_for_absolute_coding() {
        let result = reconstruct_new_mv(
            5,
            5,
            ReferenceBucket::InterLast,
            mv(-7, 12),
            |_, _| None, // no neighbours at all
        );
        assert_eq!(result, mv(-7, 12));
    }

    /// Grid-wrapper variant: 6×6 row-major grid with a qualifying
    /// above neighbour. Wrapper produces the same result as the
    /// closure-driven walker would.
    #[test]
    fn grid_wrapper_resolves_above_neighbour() {
        const W: usize = 6;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        grid[2 * W + 3] = Some(neigh(20, -10, ReferenceBucket::InterLast));
        let r = select_diff_reference_mv_from_grid(&grid, W, 3, 3, ReferenceBucket::InterLast);
        assert_eq!(r, mv(20, -10));
    }

    /// Grid-wrapper top-left corner: both above and left positions
    /// are off-frame; reference falls back to `(0, 0)`.
    #[test]
    fn grid_wrapper_top_left_corner_falls_back_to_zero() {
        const W: usize = 6;
        let grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        let r = select_diff_reference_mv_from_grid(&grid, W, 0, 0, ReferenceBucket::InterLast);
        assert_eq!(r, MotionVector::ZERO);
    }

    /// Grid-wrapper diagonal-only neighbour: a qualifying neighbour
    /// at `(-1, -1)` is ignored (geographic-constraint regression
    /// test through the grid path).
    #[test]
    fn grid_wrapper_diagonal_neighbour_ignored() {
        const W: usize = 6;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        // (2, 2) = offset (-1, -1) from current MB (3, 3)
        grid[2 * W + 2] = Some(neigh(50, -50, ReferenceBucket::InterLast));
        let r = select_diff_reference_mv_from_grid(&grid, W, 3, 3, ReferenceBucket::InterLast);
        assert_eq!(r, MotionVector::ZERO);
    }

    /// Grid-wrapper composition: `reconstruct_new_mv_from_grid`
    /// composes reference + delta against a grid.
    #[test]
    fn grid_wrapper_reconstructs_new_mv() {
        const W: usize = 6;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        // Left neighbour qualifies.
        grid[3 * W + 2] = Some(neigh(-4, 6, ReferenceBucket::InterGolden));
        let result =
            reconstruct_new_mv_from_grid(&grid, W, 3, 3, ReferenceBucket::InterGolden, mv(2, -1));
        // reference (-4, 6) + delta (2, -1) = (-2, 5)
        assert_eq!(result, mv(-2, 5));
    }

    /// Reference-bucket isolation: an `InterLast`-coded above
    /// neighbour does not match a current MB coded against
    /// `InterGolden`. Pins that the same-reference rule is enforced
    /// independently for each reference bucket variant.
    #[test]
    fn inter_last_above_does_not_match_inter_golden_current() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterGolden, |r, c| match (r, c) {
            (4, 5) => Some(neigh(99, 99, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(2, 2, ReferenceBucket::InterGolden)),
            _ => None,
        });
        assert_eq!(r, mv(2, 2));
    }

    /// `Intra`-referenced neighbours never qualify for an inter-coded
    /// current MB. The differential reference falls back to `(0, 0)`
    /// regardless of how many intra neighbours surround the MB.
    #[test]
    fn intra_neighbours_do_not_qualify_for_inter_current() {
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(10, 10, ReferenceBucket::Intra)),
            (5, 4) => Some(neigh(-20, -20, ReferenceBucket::Intra)),
            _ => None,
        });
        assert_eq!(r, MotionVector::ZERO);
    }

    /// The walker stops at the first qualifying neighbour even when a
    /// second qualifying neighbour exists at the next slot. Visitor-
    /// counting test.
    #[test]
    fn walker_short_circuits_at_first_qualifying_neighbour() {
        let mut visits: Vec<(i32, i32)> = Vec::new();
        let r = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| {
            visits.push((r, c));
            match (r, c) {
                (4, 5) => Some(neigh(1, 2, ReferenceBucket::InterLast)),
                (5, 4) => Some(neigh(3, 4, ReferenceBucket::InterLast)),
                _ => None,
            }
        });
        // Walker only visits the above neighbour, then stops.
        assert_eq!(visits, vec![(4, 5)]);
        assert_eq!(r, mv(1, 2));
    }

    /// The walker visits both positions when the first one is
    /// `None` (off-frame, not-yet-decoded). Visitor-counting test for
    /// the fall-through path.
    #[test]
    fn walker_visits_both_positions_when_above_absent() {
        let mut visits: Vec<(i32, i32)> = Vec::new();
        let _ = select_diff_reference_mv(5, 5, ReferenceBucket::InterLast, |r, c| {
            visits.push((r, c));
            None::<NeighbourMv>
        });
        assert_eq!(visits, vec![(4, 5), (5, 4)]);
    }
}

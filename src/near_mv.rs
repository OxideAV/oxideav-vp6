//! VP6 Nearest / Near alternative-MV neighbour walker (spec §10).
//!
//! Three of the §10 coding modes — `CODE_INTER_NEAREST_MV`,
//! `CODE_INTER_NEAR_MV`, `CODE_GOLD_NEAREST_MV`,
//! `CODE_GOLD_NEAR_MV` — reuse a motion vector inherited from one of
//! the spatially-nearest already-decoded macroblock neighbours instead
//! of carrying a freshly-decoded delta. The same Nearest / Near
//! availability also drives the §10 [`ModeAvailability`] row index
//! (Table 5) that selects which `probXmitted[3][20]` / `VP6_ModeVq` /
//! `ModeDecisionTree` row the §10 mode-decode traversal consults.
//!
//! Per §10 (page 28), Nearest and Near are
//!
//! > the first 2 non (0,0) MVs encountered when traversing, in order, a
//! > list of the twelve spatially nearest decoded macroblock neighbors
//! > (the list is described by offsets from the present macroblock
//! > defined in the array NearMacroblocks below), that are encoded with
//! > reference to the same prediction frame as the current block.
//!
//! and
//!
//! > If no such blocks exist in the list then Nearest and Near MVs are
//! > undefined.
//!
//! The 12 neighbour offsets are already staged at
//! [`crate::modes::NEAR_MACROBLOCKS`]; this module's walker is the
//! straight-line traversal that consumes them.
//!
//! ## Surfaces
//!
//! * [`MotionVector`] — a `(x, y)` pair in spec ¼-pixel units (signed
//!   16-bit; range `-127..=127` per §11.1's "magnitude `<= 127`" cap),
//!   plus a [`MotionVector::is_zero`] discriminator that pins the
//!   "non (0,0)" predicate to a single place.
//! * [`NeighbourMv`] — a typed neighbour record
//!   (`{mv, reference}`) the walker filters on. `reference` is reused
//!   from [`crate::dc_pred::ReferenceBucket`] for the same-reference
//!   gating §10 mandates ("encoded with reference to the same
//!   prediction frame as the current block").
//! * [`NearMvResolution`] — the walker's output: optional
//!   `nearest_mv`, optional `near_mv`, plus the derived
//!   [`crate::modes::ModeAvailability`].
//! * [`resolve_near_mvs`] — single-MB walker. Traverses
//!   `NEAR_MACROBLOCKS` in spec order against a caller-supplied
//!   `FnMut(row: i32, col: i32) -> Option<NeighbourMv>` accessor that
//!   yields the neighbour at the offset (or `None` if the position is
//!   outside the frame / has not yet been decoded). The accessor is a
//!   closure so callers stay free to back the MV grid with whatever
//!   storage they prefer (`Vec<Vec<…>>`, hashmap, sparse grid, etc.)
//!   without this module imposing a layout.
//! * [`resolve_near_mvs_from_grid`] — convenience wrapper for the
//!   common dense-grid case: takes a flat slice indexed
//!   `row * grid_width + col` plus the current MB's `(row, col)` and
//!   reference bucket, and runs the walker.
//!
//! ## What this round lands
//!
//! This is the §10 alt-MV walker — the BoolCoder-independent neighbour
//! traversal that resolves `(nearest_mv, near_mv, availability)` from
//! the surrounding already-decoded MB grid. Like
//! §15/§16/§17/§11.3-§11.5/§12.1/§14/§10's static probability surface,
//! this stage reads no BoolCoder bits — every step is pure integer
//! arithmetic over the supplied neighbour-MV grid — so it advances the
//! decoder past round 23 without re-entering any of the previously-
//! resolved BoolCoder gates.
//!
//! ## What this round deliberately does NOT land
//!
//! * The §10 `VP6_DecodeMode` Figure-10 BoolCoder traversal itself —
//!   the `ModeDecisionTree` lookup the resolved availability would
//!   index into. (Static probability surface landed in round 10; the
//!   per-bit walk is gated on the round-21 DOCS-GAP about the
//!   `B(Stats[0])` / `B(Stats[2])` else-branch indentation.)
//! * The §11 differential-MV reconstruction that combines a
//!   round-21-decoded delta with the resolved Nearest MV when the
//!   chosen mode is `CODE_INTER_PLUS_MV` / `CODE_GOLDEN_MV`. (Spec
//!   §11 intro page 37: "New motion vectors are coded differentially
//!   with respect to the motion vector of the nearest MacroBlock that
//!   uses the same reference frame …, if such a MacroBlock exists and
//!   it is either immediately to the left of or immediately above the
//!   current MacroBlock. Otherwise, new motion vectors are coded
//!   absolutely." — distinct logical unit because of the "left or
//!   above only" constraint, which is stricter than this module's
//!   12-neighbour traversal.)
//!
//! ## Provenance
//!
//! Sourced from `docs/video/vp6/vp6_format.pdf` §10 (On2 Technologies,
//! document version 1.02, August 2006), pages 27-28, and the
//! [`crate::modes::NEAR_MACROBLOCKS`] table already landed in round 10.

use crate::dc_pred::ReferenceBucket;
use crate::modes::{ModeAvailability, NEAR_MACROBLOCKS};

/// A `(x, y)` motion vector in spec ¼-pixel units.
///
/// VP6 motion vector components are signed and bounded by §11.1:
/// "the maximum magnitude of a MV component is 31 ¾ whole pixels
/// (127 in units of ¼ pixel)". This module's walker treats vectors as
/// opaque payloads keyed by [`MotionVector::is_zero`]; no arithmetic
/// is performed on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MotionVector {
    /// Horizontal component, ¼-pixel units. Positive moves right.
    pub x: i16,
    /// Vertical component, ¼-pixel units. Positive moves down.
    pub y: i16,
}

impl MotionVector {
    /// The all-zero MV. Equivalent to `MotionVector { x: 0, y: 0 }`.
    pub const ZERO: Self = Self { x: 0, y: 0 };

    /// Construct a `MotionVector` from its two ¼-pixel components.
    #[inline]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// True iff the vector is `(0, 0)`.
    ///
    /// Pins the spec's "non (0, 0)" predicate ("the first 2 non (0,0)
    /// MVs encountered …") to one place so the walker stays
    /// translation-symmetric in `(x, y)`.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.x == 0 && self.y == 0
    }
}

/// One macroblock-grid neighbour's metadata for the §10 Nearest/Near
/// walker.
///
/// `mv` is the neighbour's already-decoded motion vector; `reference`
/// is the prediction-frame bucket the neighbour was coded against.
/// Both fields are needed because the §10 walker filters on
/// `mv != (0, 0)` AND `reference == self.reference` simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourMv {
    /// The neighbour's already-decoded motion vector.
    pub mv: MotionVector,
    /// The reference-frame bucket the neighbour was coded against.
    ///
    /// Reused from [`ReferenceBucket`] so the same-reference gating
    /// the §10 walker shares with §14 DC prediction lives in one
    /// canonical enum.
    pub reference: ReferenceBucket,
}

impl NeighbourMv {
    /// Construct a [`NeighbourMv`] from its components.
    #[inline]
    pub const fn new(mv: MotionVector, reference: ReferenceBucket) -> Self {
        Self { mv, reference }
    }
}

/// Output of [`resolve_near_mvs`].
///
/// Carries the resolved Nearest and Near motion vectors (each `None`
/// when the corresponding §10 traversal step did not find a qualifying
/// neighbour) plus the derived [`ModeAvailability`] row index.
///
/// The two `Option`s respect the spec's bookkeeping order: `Near` is
/// always `None` when `Nearest` is `None` (Near is "the second" non
/// (0, 0) same-reference neighbour, so it can only be discovered after
/// Nearest is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NearMvResolution {
    /// The first non-(0, 0) MV from a same-reference neighbour in
    /// spec [`NEAR_MACROBLOCKS`] order; `None` if no such neighbour
    /// exists.
    pub nearest_mv: Option<MotionVector>,
    /// The second non-(0, 0) MV from a same-reference neighbour in
    /// spec [`NEAR_MACROBLOCKS`] order, distinct from
    /// [`Self::nearest_mv`] in traversal position; `None` if no second
    /// qualifying neighbour exists. Always `None` when
    /// [`Self::nearest_mv`] is `None`.
    pub near_mv: Option<MotionVector>,
    /// The §10 Table 5 `ProbabilitySituation` row index implied by the
    /// two `Option`s, ready to drop into the §10 mode-decode
    /// `probXmitted` / `ModeDecisionTree` row selection.
    pub availability: ModeAvailability,
}

impl NearMvResolution {
    /// The "no qualifying neighbour" outcome:
    /// `(None, None, ModeAvailability::Neither)`.
    ///
    /// Returned by the walker when the 12-neighbour traversal finds
    /// no same-reference neighbour with a non-(0, 0) MV.
    pub const NONE: Self = Self {
        nearest_mv: None,
        near_mv: None,
        availability: ModeAvailability::Neither,
    };

    /// True iff at least Nearest is resolved.
    ///
    /// Equivalent to `self.availability != ModeAvailability::Neither`.
    #[inline]
    pub const fn has_nearest(&self) -> bool {
        self.nearest_mv.is_some()
    }

    /// True iff both Nearest and Near are resolved.
    ///
    /// Equivalent to `self.availability == ModeAvailability::NearestAndNear`.
    #[inline]
    pub const fn has_near(&self) -> bool {
        self.near_mv.is_some()
    }
}

/// Run the §10 [`NEAR_MACROBLOCKS`] walker against a caller-supplied
/// accessor and resolve the Nearest / Near motion vectors plus the
/// implied [`ModeAvailability`].
///
/// Each step of the traversal calls `neighbour_at(row + dr, col + dc)`
/// where `(dr, dc)` is the next entry of `NEAR_MACROBLOCKS` in spec
/// order. The accessor returns `Some(NeighbourMv)` if the neighbour at
/// the absolute `(row, col)` position has already been decoded **and**
/// the caller wishes to expose it to the walker (callers can also use
/// the accessor as a per-MB visibility filter: return `None` for any
/// position outside the frame, before the current decoding raster
/// pointer, or otherwise unavailable).
///
/// The walker only retains neighbours that pass both §10 predicates:
///
/// 1. the neighbour's `reference` matches the supplied `reference`
///    (the "encoded with reference to the same prediction frame as
///    the current block" rule), and
/// 2. the neighbour's `mv` is not `(0, 0)` (the "non (0, 0)" rule).
///
/// Of the qualifying neighbours, the first becomes `nearest_mv`; the
/// second becomes `near_mv`. The walker stops as soon as both are
/// found (or after all 12 offsets have been visited, whichever comes
/// first). All other neighbours along the way — including `(0, 0)`-MV
/// or different-reference ones — are passed over silently per spec.
///
/// # Parameters
///
/// * `row`, `col` — the current MB's `(row, col)` position. Combined
///   with each [`NEAR_MACROBLOCKS`] offset to form the absolute
///   neighbour position passed to `neighbour_at`. Signed `i32` to
///   accommodate offsets `(-2, -2)` near the frame's top-left corner;
///   the accessor sees negative coordinates for off-frame positions
///   and is expected to return `None`.
/// * `reference` — the current MB's target prediction-frame bucket.
///   Used to filter neighbours by the same-reference rule.
/// * `neighbour_at` — accessor that maps an absolute `(row, col)`
///   position to the neighbour's MV + reference, or `None`.
#[inline]
pub fn resolve_near_mvs<F>(
    row: i32,
    col: i32,
    reference: ReferenceBucket,
    mut neighbour_at: F,
) -> NearMvResolution
where
    F: FnMut(i32, i32) -> Option<NeighbourMv>,
{
    let mut nearest: Option<MotionVector> = None;
    let mut near: Option<MotionVector> = None;
    for (dr, dc) in NEAR_MACROBLOCKS {
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
        if nearest.is_none() {
            nearest = Some(neigh.mv);
        } else {
            near = Some(neigh.mv);
            break;
        }
    }
    let availability = ModeAvailability::from_neighbours(nearest.is_some(), near.is_some());
    NearMvResolution {
        nearest_mv: nearest,
        near_mv: near,
        availability,
    }
}

/// Dense-grid convenience wrapper for [`resolve_near_mvs`].
///
/// Backs the walker with a flat `&[Option<NeighbourMv>]` slice indexed
/// row-major as `grid[row * grid_width + col]`. Out-of-bounds
/// `(row, col)` access (negative coordinates, or `col >= grid_width`,
/// or `row * grid_width + col >= grid.len()`) returns `None` exactly
/// as if the underlying accessor had returned `None`.
///
/// `grid_width` is taken explicitly so callers can store grids in
/// row-major rectangles wider than the frame (e.g. for padding /
/// border bookkeeping); the walker only consults `(row, col)`
/// positions read out of the slice and never inspects the slice's
/// shape directly.
///
/// # Panics
///
/// Does not panic for any out-of-range access; all index arithmetic
/// is bounded by the explicit `(row, col)` and `grid_width` checks.
pub fn resolve_near_mvs_from_grid(
    grid: &[Option<NeighbourMv>],
    grid_width: usize,
    row: i32,
    col: i32,
    reference: ReferenceBucket,
) -> NearMvResolution {
    resolve_near_mvs(row, col, reference, |r, c| {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i16, y: i16) -> MotionVector {
        MotionVector::new(x, y)
    }

    fn neigh(x: i16, y: i16, r: ReferenceBucket) -> NeighbourMv {
        NeighbourMv::new(mv(x, y), r)
    }

    /// All twelve [`NEAR_MACROBLOCKS`] offsets pinned in spec order.
    /// Re-asserted here so a future re-ordering of the constant trips
    /// this module's tests immediately (the walker's behaviour depends
    /// on the traversal order: the *first* qualifying neighbour
    /// becomes `nearest_mv`).
    #[test]
    fn near_macroblocks_order_pins_spec_listing() {
        assert_eq!(
            NEAR_MACROBLOCKS,
            [
                (-1, 0),
                (0, -1),
                (-1, -1),
                (-1, 1),
                (-2, 0),
                (0, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
                (-2, -2),
                (-2, 2),
            ]
        );
    }

    #[test]
    fn motion_vector_zero_pins_zero_predicate() {
        assert!(MotionVector::ZERO.is_zero());
        assert!(mv(0, 0).is_zero());
        assert!(!mv(1, 0).is_zero());
        assert!(!mv(0, 1).is_zero());
        assert!(!mv(-1, 0).is_zero());
        assert!(!mv(0, -1).is_zero());
        assert!(!mv(-127, -127).is_zero());
        assert!(!mv(127, 127).is_zero());
    }

    /// No neighbour anywhere → `NONE` outcome.
    #[test]
    fn empty_grid_resolves_to_none() {
        let result = resolve_near_mvs(0, 0, ReferenceBucket::InterLast, |_, _| None);
        assert_eq!(result, NearMvResolution::NONE);
        assert_eq!(result.availability, ModeAvailability::Neither);
        assert!(!result.has_nearest());
        assert!(!result.has_near());
    }

    /// Single qualifying neighbour at offset `(-1, 0)` (the first
    /// [`NEAR_MACROBLOCKS`] entry) → `Nearest` resolves, `Near`
    /// stays unresolved, availability is `NearestOnly`.
    #[test]
    fn single_above_neighbour_resolves_nearest_only() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| {
            if (r, c) == (4, 5) {
                Some(neigh(8, -12, ReferenceBucket::InterLast))
            } else {
                None
            }
        });
        assert_eq!(result.nearest_mv, Some(mv(8, -12)));
        assert_eq!(result.near_mv, None);
        assert_eq!(result.availability, ModeAvailability::NearestOnly);
        assert!(result.has_nearest());
        assert!(!result.has_near());
    }

    /// Two qualifying neighbours at the first two [`NEAR_MACROBLOCKS`]
    /// offsets (`(-1, 0)` and `(0, -1)`) → both resolve in spec order,
    /// availability is `NearestAndNear`.
    #[test]
    fn two_neighbours_resolve_in_spec_order() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(4, -8, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(-12, 20, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(result.nearest_mv, Some(mv(4, -8)));
        assert_eq!(result.near_mv, Some(mv(-12, 20)));
        assert_eq!(result.availability, ModeAvailability::NearestAndNear);
        assert!(result.has_nearest());
        assert!(result.has_near());
    }

    /// Different-reference neighbours are skipped. The walker is
    /// targeted at `InterGolden`; an `InterLast` neighbour at the
    /// first offset gets passed over and the `InterGolden` neighbour
    /// at the third offset becomes Nearest.
    #[test]
    fn different_reference_neighbours_skipped() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterGolden, |r, c| match (r, c) {
            (4, 5) => Some(neigh(1, 1, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(2, 2, ReferenceBucket::Intra)),
            (4, 4) => Some(neigh(7, -5, ReferenceBucket::InterGolden)),
            _ => None,
        });
        assert_eq!(result.nearest_mv, Some(mv(7, -5)));
        assert_eq!(result.near_mv, None);
        assert_eq!(result.availability, ModeAvailability::NearestOnly);
    }

    /// Same-reference but `(0, 0)`-MV neighbours are skipped (spec's
    /// "non (0, 0)" rule).
    #[test]
    fn zero_mv_neighbours_skipped() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(0, 0, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(0, 0, ReferenceBucket::InterLast)),
            (4, 4) => Some(neigh(3, 4, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(result.nearest_mv, Some(mv(3, 4)));
        assert_eq!(result.near_mv, None);
        assert_eq!(result.availability, ModeAvailability::NearestOnly);
    }

    /// The walker stops at the second qualifying neighbour even when
    /// further qualifying neighbours exist downstream in the
    /// traversal. Visitor-counting test.
    #[test]
    fn walker_short_circuits_after_second_hit() {
        let mut visits: Vec<(i32, i32)> = Vec::new();
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| {
            visits.push((r, c));
            match (r, c) {
                (4, 5) => Some(neigh(1, 0, ReferenceBucket::InterLast)),
                (5, 4) => Some(neigh(0, 1, ReferenceBucket::InterLast)),
                // These would be qualifying neighbours if reached.
                (4, 4) => Some(neigh(2, 2, ReferenceBucket::InterLast)),
                (4, 6) => Some(neigh(3, 3, ReferenceBucket::InterLast)),
                _ => None,
            }
        });
        // Walker visits the first two `NEAR_MACROBLOCKS` offsets and
        // then short-circuits (no third visit recorded).
        assert_eq!(visits, vec![(4, 5), (5, 4)]);
        assert_eq!(result.nearest_mv, Some(mv(1, 0)));
        assert_eq!(result.near_mv, Some(mv(0, 1)));
    }

    /// Negative `(row, col)` positions near the top-left corner are
    /// reported to the accessor; an accessor that returns `None` for
    /// them produces a `Neither` outcome.
    #[test]
    fn top_left_corner_off_frame_positions_reported_to_accessor() {
        let mut saw_negative = false;
        let _ = resolve_near_mvs(0, 0, ReferenceBucket::InterLast, |r, c| {
            if r < 0 || c < 0 {
                saw_negative = true;
            }
            None
        });
        assert!(saw_negative);
    }

    /// Dense-grid wrapper finds neighbours in row-major slice with
    /// the spec-ordered priorities preserved.
    #[test]
    fn grid_wrapper_resolves_nearest_and_near() {
        // 6×6 macroblock grid, current MB at (3, 3). Place a
        // qualifying neighbour at the above slot (offset (-1, 0)) and
        // the left slot (offset (0, -1)).
        const W: usize = 6;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        grid[2 * W + 3] = Some(neigh(10, -4, ReferenceBucket::InterLast));
        grid[3 * W + 2] = Some(neigh(-3, 7, ReferenceBucket::InterLast));
        let result = resolve_near_mvs_from_grid(&grid, W, 3, 3, ReferenceBucket::InterLast);
        assert_eq!(result.nearest_mv, Some(mv(10, -4)));
        assert_eq!(result.near_mv, Some(mv(-3, 7)));
        assert_eq!(result.availability, ModeAvailability::NearestAndNear);
    }

    /// Grid-wrapper handles top-left-corner MBs by silently treating
    /// negative-row / negative-col positions as `None`.
    #[test]
    fn grid_wrapper_handles_top_left_corner() {
        // 6×6 grid, current MB at (0, 0). All 12 NEAR_MACROBLOCKS
        // offsets have at least one negative coordinate, so every
        // lookup returns `None`.
        const W: usize = 6;
        let grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        let result = resolve_near_mvs_from_grid(&grid, W, 0, 0, ReferenceBucket::InterLast);
        assert_eq!(result, NearMvResolution::NONE);
    }

    /// Grid-wrapper clamps out-of-bounds positive `(row, col)` to
    /// `None` (no panic, no spurious resolution).
    #[test]
    fn grid_wrapper_handles_bottom_right_corner() {
        // 4×4 grid, current MB at (3, 3). Offsets (-1, 2) and (-2, 2)
        // and the right-of-row positions land outside `grid_width`;
        // the wrapper must report `None` for those without panicking.
        const W: usize = 4;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        // Put a qualifying neighbour at (-1, 0) = (2, 3) so we can
        // verify the wrapper still finds it.
        grid[2 * W + 3] = Some(neigh(5, 5, ReferenceBucket::InterLast));
        let result = resolve_near_mvs_from_grid(&grid, W, 3, 3, ReferenceBucket::InterLast);
        assert_eq!(result.nearest_mv, Some(mv(5, 5)));
        assert_eq!(result.near_mv, None);
        assert_eq!(result.availability, ModeAvailability::NearestOnly);
    }

    /// Grid-wrapper treats different-reference neighbours as absent
    /// per the same-reference rule.
    #[test]
    fn grid_wrapper_filters_on_reference() {
        const W: usize = 6;
        let mut grid: Vec<Option<NeighbourMv>> = vec![None; W * W];
        // Above neighbour codes against Golden but current MB codes
        // against InterLast — should be skipped.
        grid[2 * W + 3] = Some(neigh(1, 1, ReferenceBucket::InterGolden));
        // Left neighbour matches reference and qualifies.
        grid[3 * W + 2] = Some(neigh(2, 2, ReferenceBucket::InterLast));
        let result = resolve_near_mvs_from_grid(&grid, W, 3, 3, ReferenceBucket::InterLast);
        assert_eq!(result.nearest_mv, Some(mv(2, 2)));
        assert_eq!(result.near_mv, None);
    }

    /// `NearMvResolution::NONE` matches the empty-grid output.
    #[test]
    fn none_constant_matches_walker_output() {
        let from_walker = resolve_near_mvs(0, 0, ReferenceBucket::Intra, |_, _| None);
        assert_eq!(from_walker, NearMvResolution::NONE);
        assert_eq!(NearMvResolution::NONE.nearest_mv, None);
        assert_eq!(NearMvResolution::NONE.near_mv, None);
        assert_eq!(
            NearMvResolution::NONE.availability,
            ModeAvailability::Neither
        );
    }

    /// The walker's `availability` field is exactly what
    /// [`ModeAvailability::from_neighbours`] would return for the
    /// `(nearest.is_some(), near.is_some())` pair — the §10 walker
    /// computes the same Table 5 row index the
    /// [`crate::modes::ModeAvailability`] enum already pins.
    #[test]
    fn availability_matches_from_neighbours() {
        // Two-neighbour case → NearestAndNear.
        let r = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(1, 1, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(2, 2, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(
            r.availability,
            ModeAvailability::from_neighbours(r.has_nearest(), r.has_near())
        );

        // One-neighbour case → NearestOnly.
        let r = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(1, 1, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(
            r.availability,
            ModeAvailability::from_neighbours(r.has_nearest(), r.has_near())
        );

        // Zero-neighbour case → Neither.
        let r = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |_, _| None);
        assert_eq!(
            r.availability,
            ModeAvailability::from_neighbours(r.has_nearest(), r.has_near())
        );
    }

    /// `Intra` reference: the walker still runs; only neighbours
    /// flagged as `Intra` qualify. (In practice §10's NEAREST/NEAR
    /// modes don't apply to intra-coded MBs, but the walker is
    /// reference-symmetric and exercising the Intra branch
    /// documents that.)
    #[test]
    fn intra_reference_filters_to_intra_neighbours_only() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::Intra, |r, c| match (r, c) {
            (4, 5) => Some(neigh(1, 1, ReferenceBucket::InterLast)),
            (5, 4) => Some(neigh(2, 2, ReferenceBucket::Intra)),
            _ => None,
        });
        assert_eq!(result.nearest_mv, Some(mv(2, 2)));
        assert_eq!(result.near_mv, None);
    }

    /// Maximum-magnitude MV components (the §11.1 `±127` cap) are
    /// non-zero and qualify for the walker — pinning that the
    /// `i16` storage handles the spec's full range.
    #[test]
    fn maximum_magnitude_mv_qualifies() {
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| match (r, c) {
            (4, 5) => Some(neigh(127, -127, ReferenceBucket::InterLast)),
            _ => None,
        });
        assert_eq!(result.nearest_mv, Some(mv(127, -127)));
    }

    /// All-12-qualify case: a same-reference non-zero MV at every
    /// `NEAR_MACROBLOCKS` offset. The walker picks the first two and
    /// stops. Pins both that the traversal terminates and that the
    /// two it picks are the spec-first two.
    #[test]
    fn all_twelve_qualify_picks_first_two() {
        let mut grid_map = std::collections::HashMap::<(i32, i32), NeighbourMv>::new();
        for (i, (dr, dc)) in NEAR_MACROBLOCKS.iter().enumerate() {
            let absolute = (5 + (*dr as i32), 5 + (*dc as i32));
            // Use the index+1 as the MV magnitude so the test can
            // verify the spec-first one was picked.
            let m = (i as i16) + 1;
            grid_map.insert(absolute, neigh(m, m, ReferenceBucket::InterLast));
        }
        let result = resolve_near_mvs(5, 5, ReferenceBucket::InterLast, |r, c| {
            grid_map.get(&(r, c)).copied()
        });
        // First offset is (-1, 0) → (4, 5), magnitude 1.
        assert_eq!(result.nearest_mv, Some(mv(1, 1)));
        // Second offset is (0, -1) → (5, 4), magnitude 2.
        assert_eq!(result.near_mv, Some(mv(2, 2)));
        assert_eq!(result.availability, ModeAvailability::NearestAndNear);
    }
}

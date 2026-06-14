//! VP6 macroblock coding-mode static surface (spec §10).
//!
//! VP6 P-frame macroblocks carry one of ten **coding modes** (spec
//! Table 4) that tells the decoder how to source the MB's prediction:
//! intra-coded, inter-coded against the previous reconstruction or
//! against the Golden Frame, with a fresh motion vector or one
//! inherited from a "Nearest" or "Near" neighbour, optionally split
//! into four per-Y-block vectors. I-frame MBs are implicitly intra and
//! transmit no mode.
//!
//! This module surfaces the **BoolCoder-independent** half of §10:
//!
//! * [`CodingMode`] — the ten Table 4 modes plus the spec's canonical
//!   `0..=9` integer indexing the decoder uses throughout (e.g. as the
//!   `lastmode` argument of `ModeDecisionTree`, as the second index of
//!   `VP6_BASELINE_XMITTED_PROBS`).
//! * [`ModeAvailability`] — the three Table 5 "ProbabilitySituation"
//!   indices that distinguish Nearest&Near-both-exist /
//!   Nearest-only-exists / neither-exists neighbour configurations.
//! * [`NEAR_MACROBLOCKS`] — the 12 (row, column) macroblock-unit
//!   offsets §10 mandates for traversing the spatially-nearest decoded
//!   MB neighbours when resolving Nearest and Near motion vectors.
//! * [`VP6_BASELINE_XMITTED_PROBS`] — the verbatim `[3][20]` baseline
//!   set §10's Table 6 ProbabilitySituation table initialises
//!   `probXmitted` to at every I-frame.
//! * [`VP6_MODE_VQ`] — the verbatim `[3][16][20]` quantised mode-prob
//!   baseline bank (16 vectors per availability) the §10
//!   `SetNewBaselineProbs` flag selects from.
//! * [`build_mode_decision_tree`] — the pure-integer transform that
//!   converts a `probXmitted[3][20]` table into the
//!   `ModeDecisionTree[3][10][9]` array of per-node BoolCoder
//!   probabilities used by the §10 `VP6_DecodeMode` traversal.
//! * [`mode_decision_tree_node_probability`] — single-node helper that
//!   computes one entry of the `ModeDecisionTree` without materialising
//!   the full `[3][10][9]` array (useful for spot checks).
//! * [`probability_mode_same`] — the §10 `probModeSame` companion
//!   probability the root of the decision tree consults to decide
//!   whether the MB inherits the previous MB's mode.
//!
//! ## What this module does NOT land
//!
//! The §10 decoder *traversal* (`VP6_DecodeMode`) itself reads
//! `B(probModeSame[type][lastmode])`, then walks the `Figure 10`
//! decision tree by repeated `B(Stats[…])` reads. Each `B(x)` is a
//! BoolCoder bit and so depends on the §7.3 `Split` formula, which is
//! blocked by a DOCS-GAP (see the crate-root docs `## DOCS-GAP`
//! section). The mode-probability update bitstream from §10's Table 7
//! / Table 8 (`SetNewBaselineProbs`, `WhichVector`,
//! `VectorUpdatesPresentFlag`, `ModeProbUpdateVector`) is similarly
//! BoolCoder-gated and stays deferred.
//!
//! What we *do* land is everything that does not call the BoolCoder:
//! the static tables, the enum surface, and the pure-integer
//! `probXmitted → ModeDecisionTree` conversion. With the conversion
//! in place, the only piece of §10 still pending the §7.3 fix is the
//! 11 BoolCoder reads of the per-MB traversal itself.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §10 (On2
//! Technologies, document version 1.02, August 2006). No third-party
//! VP6 implementation has been consulted.

use core::fmt;

/// Number of macroblock coding modes the spec defines (§10 Table 4).
///
/// The §10 prose states "VP6 defines ten possible coding modes." The
/// constant pins it for callers iterating over a fixed-size buffer.
pub const NUM_CODING_MODES: usize = 10;

/// Number of `ProbabilitySituation` rows §10 Table 5 distinguishes.
///
/// Three rows: Nearest&Near both exist, Nearest only, neither exists.
/// Used as the first dimension of `probXmitted`, `VP6_ModeVq`, and
/// `ModeDecisionTree`.
pub const NUM_PROBABILITY_SITUATIONS: usize = 3;

/// Number of probability entries per `probXmitted` row.
///
/// §10 Table 6 specifies twenty per row (two entries — `same-as-prior`
/// and `different-from-prior` — for each of the ten coding modes).
pub const PROB_XMITTED_ROW_LEN: usize = 20;

/// Number of `VP6_ModeVq` baseline vectors per ProbabilitySituation
/// (§10: "one of 16 pre-defined sets").
pub const NUM_MODE_VQ_VECTORS: usize = 16;

/// Number of internal decision-tree nodes per `(type, lastmode)`
/// stats row (the `9` of `ModeDecisionTree[3][10][9]`, Figure 10's
/// nodes 0..=8).
pub const NUM_MODE_DECISION_NODES: usize = 9;

/// VP6 macroblock coding modes (spec §10 Table 4).
///
/// The discriminant matches the canonical 0..=9 index the spec uses
/// when indexing arrays by mode (e.g. as `lastmode` in
/// `ModeDecisionTree[type][lastmode][node]`). The order in the
/// declaration follows Table 4's row order verbatim.
///
/// I-frame MBs are implicitly `CodeIntra` — the I-frame path
/// transmits no mode element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CodingMode {
    /// `CODE_INTER_NO_MV`. Inter-coded against the previous-frame
    /// reconstruction with the fixed (0, 0) motion vector. Index 0.
    InterNoMv = 0,
    /// `CODE_INTRA`. Intra-coded; the MB carries no prediction. Index 1.
    Intra = 1,
    /// `CODE_INTER_PLUS_MV`. Inter-coded against the previous-frame
    /// reconstruction with a newly-decoded motion vector. Index 2.
    InterPlusMv = 2,
    /// `CODE_INTER_NEAREST_MV`. Inter-coded against the previous-frame
    /// reconstruction reusing the same MV as the `Nearest` neighbour.
    /// Index 3.
    InterNearestMv = 3,
    /// `CODE_INTER_NEAR_MV`. Inter-coded against the previous-frame
    /// reconstruction reusing the same MV as the `Near` neighbour.
    /// Index 4.
    InterNearMv = 4,
    /// `CODE_USING_GOLDEN`. Inter-coded against the Golden Frame with
    /// the fixed (0, 0) motion vector. Index 5.
    UsingGolden = 5,
    /// `CODE_GOLDEN_MV`. Inter-coded against the Golden Frame with a
    /// newly-decoded motion vector. Index 6.
    GoldenMv = 6,
    /// `CODE_INTER_FOURMV`. Inter-coded against the previous-frame
    /// reconstruction with **four** per-Y-block motion vectors (the
    /// two chroma MVs are derived as the rounding-away-from-zero
    /// average of the four luma MVs per §10). Index 7.
    InterFourMv = 7,
    /// `CODE_GOLD_NEAREST_MV`. Inter-coded against the Golden Frame
    /// reusing the same MV as the `Nearest` neighbour that referenced
    /// Golden. Index 8.
    GoldNearestMv = 8,
    /// `CODE_GOLD_NEAR_MV`. Inter-coded against the Golden Frame
    /// reusing the same MV as the `Near` neighbour that referenced
    /// Golden. Index 9.
    GoldNearMv = 9,
}

impl CodingMode {
    /// Canonical 0..=9 spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`CodingMode::index`]: build a `CodingMode` from
    /// the spec's 0..=9 integer. Returns `None` for out-of-range
    /// values.
    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::InterNoMv),
            1 => Some(Self::Intra),
            2 => Some(Self::InterPlusMv),
            3 => Some(Self::InterNearestMv),
            4 => Some(Self::InterNearMv),
            5 => Some(Self::UsingGolden),
            6 => Some(Self::GoldenMv),
            7 => Some(Self::InterFourMv),
            8 => Some(Self::GoldNearestMv),
            9 => Some(Self::GoldNearMv),
            _ => None,
        }
    }

    /// All ten coding modes in canonical Table 4 order.
    pub const ALL: [Self; NUM_CODING_MODES] = [
        Self::InterNoMv,
        Self::Intra,
        Self::InterPlusMv,
        Self::InterNearestMv,
        Self::InterNearMv,
        Self::UsingGolden,
        Self::GoldenMv,
        Self::InterFourMv,
        Self::GoldNearestMv,
        Self::GoldNearMv,
    ];

    /// True if the mode predicts from the Golden Frame (the four
    /// `Golden`-tagged modes), false if it predicts from the previous
    /// reconstruction or carries no prediction (intra).
    ///
    /// Useful for routing prediction fetches: see
    /// [`crate::inter::fetch_prediction_block`].
    #[inline]
    pub const fn uses_golden(self) -> bool {
        matches!(
            self,
            Self::UsingGolden | Self::GoldenMv | Self::GoldNearestMv | Self::GoldNearMv
        )
    }

    /// True for the intra mode (`CODE_INTRA`), false otherwise.
    ///
    /// Intra MBs use the §17.1 (`+128` level shift + clip) intra
    /// reconstruction path; all other modes use the §17.2–§17.4 inter
    /// path.
    #[inline]
    pub const fn is_intra(self) -> bool {
        matches!(self, Self::Intra)
    }

    /// True if the mode requires a freshly-decoded MV (`*_PLUS_MV` /
    /// `*_MV` variants), false if the MV is implicit (zero) or
    /// inherited from a neighbour (`*_NEAREST_MV` / `*_NEAR_MV`).
    #[inline]
    pub const fn carries_new_mv(self) -> bool {
        matches!(self, Self::InterPlusMv | Self::GoldenMv | Self::InterFourMv)
    }
}

impl fmt::Display for CodingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InterNoMv => "CODE_INTER_NO_MV",
            Self::Intra => "CODE_INTRA",
            Self::InterPlusMv => "CODE_INTER_PLUS_MV",
            Self::InterNearestMv => "CODE_INTER_NEAREST_MV",
            Self::InterNearMv => "CODE_INTER_NEAR_MV",
            Self::UsingGolden => "CODE_USING_GOLDEN",
            Self::GoldenMv => "CODE_GOLDEN_MV",
            Self::InterFourMv => "CODE_INTER_FOURMV",
            Self::GoldNearestMv => "CODE_GOLD_NEAREST_MV",
            Self::GoldNearMv => "CODE_GOLD_NEAR_MV",
        })
    }
}

/// VP6 ProbabilitySituation (spec §10 Table 5).
///
/// Selects which row of `probXmitted` / `VP6_ModeVq` /
/// `ModeDecisionTree` applies to the current MB. The choice depends
/// on which of the Nearest / Near motion vectors are defined for the
/// MB; see §10's `NearMacroblocks[12]` traversal and the
/// `nearest_mv_exists` / `near_mv_exists` parameters of
/// [`ModeAvailability::from_neighbours`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ModeAvailability {
    /// Index 0: Nearest **and** Near MVs both exist for this MB.
    NearestAndNear = 0,
    /// Index 1: Nearest exists, Near does not.
    NearestOnly = 1,
    /// Index 2: Neither Nearest nor Near exists.
    Neither = 2,
}

impl ModeAvailability {
    /// Spec-canonical 0..=2 index used in array subscripts.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`ModeAvailability::index`]. Returns `None` for
    /// indices outside 0..=2.
    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::NearestAndNear),
            1 => Some(Self::NearestOnly),
            2 => Some(Self::Neither),
            _ => None,
        }
    }

    /// Decide the situation index from the per-MB Nearest/Near
    /// availability flags.
    ///
    /// Per §10's `NearMacroblocks[12]` traversal: a MV exists when at
    /// least one of the 12 spatially-nearest decoded MB neighbours
    /// (a) has a non-(0,0) MV (Nearest is "the first" such; Near is
    /// "the second") and (b) is encoded with reference to the same
    /// prediction frame as the current MB. The two boolean inputs to
    /// this function are the spec's bookkeeping result.
    #[inline]
    pub const fn from_neighbours(nearest_mv_exists: bool, near_mv_exists: bool) -> Self {
        match (nearest_mv_exists, near_mv_exists) {
            (true, true) => Self::NearestAndNear,
            (true, false) => Self::NearestOnly,
            // The spec specifies "Nearest exists" as the precondition
            // for "Near exists" (Near is the "second" non-(0,0) MV in
            // the traversal). The (false, true) combination is
            // therefore degenerate; we fold it to `Neither` rather
            // than allow a meaningless availability state.
            (false, _) => Self::Neither,
        }
    }

    /// All three availabilities in spec order.
    pub const ALL: [Self; NUM_PROBABILITY_SITUATIONS] =
        [Self::NearestAndNear, Self::NearestOnly, Self::Neither];
}

impl fmt::Display for ModeAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NearestAndNear => "Nearest & Near both exist",
            Self::NearestOnly => "Nearest only",
            Self::Neither => "Neither Nearest nor Near",
        })
    }
}

/// Spatially-nearest MB neighbour offsets used for Nearest/Near MV
/// resolution (spec §10 `NearMacroblocks[12]`).
///
/// Each entry is `(row_offset, col_offset)` in macroblock units
/// **relative to the current MB**. The spec traverses the 12
/// neighbours in this order; the first non-(0,0) MV from a neighbour
/// coded against the same prediction frame becomes `NearestMv`, the
/// second becomes `NearMv`. Verbatim from §10:
///
/// ```text
/// NearMacroblocks[12] =
/// {
///     { -1,  0 },
///     {  0, -1 },
///     { -1, -1 },
///     { -1,  1 },
///     { -2,  0 },
///     {  0, -2 },
///     { -1, -2 },
///     { -2, -1 },
///     { -2,  1 },
///     { -1,  2 },
///     { -2, -2 },
///     { -2,  2 }
/// }
/// ```
pub const NEAR_MACROBLOCKS: [(i8, i8); 12] = [
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
];

/// Baseline `probXmitted[3][20]` initialiser §10 mandates at every
/// I-frame (`VP6_BaselineXmittedProbs`).
///
/// At each I-frame, `probXmitted` (the dynamic mode-probability table
/// the decoder maintains) is reset to this baseline. For P-frames the
/// table persists from the previous decoded frame, optionally
/// modified by Table 7 / Table 8 `Mode Probability Updates` bitstream
/// entries (BoolCoder-gated and deferred).
///
/// First dimension: `ModeAvailability` (0 = both exist, 1 = nearest
/// only, 2 = neither). Second dimension: per-mode `(same-as-prior,
/// different-from-prior)` pairs in §10 Table 6 order.
///
/// Verbatim from §10:
///
/// ```text
/// VP6_BaselineXmittedProbs[3][20] =
/// {
///     {  42, 69,  2,  1,  7,  1, 42, 44, 22,  6,
///         3,  1,  2,  0,  5,  1,  1,  0,  0,  0 },
///     {   8,229,  1,  1,  8,  0,  0,  0,  0,  0,
///         2,  1,  1,  0,  0,  0,  1,  1,  0,  0 },
///     {  35,122,  1,  1,  6,  1, 34, 46,  0,  0,
///         2,  1,  1,  0,  1,  0,  1,  1,  0,  0 }
/// }
/// ```
#[rustfmt::skip]
pub const VP6_BASELINE_XMITTED_PROBS: [[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS] = [
    [ 42,  69,   2,   1,   7,   1,  42,  44,  22,   6,
       3,   1,   2,   0,   5,   1,   1,   0,   0,   0 ],
    [  8, 229,   1,   1,   8,   0,   0,   0,   0,   0,
       2,   1,   1,   0,   0,   0,   1,   1,   0,   0 ],
    [ 35, 122,   1,   1,   6,   1,  34,  46,   0,   0,
       2,   1,   1,   0,   1,   0,   1,   1,   0,   0 ],
];

/// Quantised mode-probability baseline bank `VP6_ModeVq[3][16][20]`
/// the §10 `SetNewBaselineProbs` flag selects from.
///
/// Three rows (one per `ModeAvailability`), each holding sixteen
/// 20-element probability vectors. When `SetNewBaselineProbs` is set,
/// the next four-bit `WhichVector` field picks the vector copied into
/// `probXmitted` for the matching availability row.
///
/// Verbatim from §10:
///
/// ```text
/// VP6_ModeVq[3][16][20] = { /* see spec; transcribed below */ }
/// ```
#[rustfmt::skip]
pub const VP6_MODE_VQ: [[[u8; PROB_XMITTED_ROW_LEN]; NUM_MODE_VQ_VECTORS]; NUM_PROBABILITY_SITUATIONS] = [
    // ModeAvailability = NearestAndNear (situation 0)
    [
        [   9,  15,  32,  25,   7,  19,   9,  21,   1,  12,  14,  12,   3,  18,  14,  23,   3,  10,   0,   4 ],
        [  48,  39,   1,   2,  11,  27,  29,  44,   7,  27,   1,   4,   0,   3,   1,   6,   1,   2,   0,   0 ],
        [  21,  32,   1,   2,   4,  10,  32,  43,   6,  23,   2,   3,   1,  19,   1,   6,  12,  21,   0,   7 ],
        [  69,  83,   0,   0,   0,   2,  10,  29,   3,  12,   0,   1,   0,   3,   0,   3,   2,   2,   0,   0 ],
        [  11,  20,   1,   4,  18,  36,  43,  48,  13,  35,   0,   2,   0,   5,   3,  12,   1,   2,   0,   0 ],
        [  70,  44,   0,   1,   2,  10,  37,  46,   8,  26,   0,   2,   0,   2,   0,   2,   0,   1,   0,   0 ],
        [   8,  15,   0,   1,   8,  21,  74,  53,  22,  42,   0,   1,   0,   2,   0,   3,   1,   2,   0,   0 ],
        [ 141,  42,   0,   0,   1,   4,  11,  24,   1,  11,   0,   1,   0,   1,   0,   2,   0,   0,   0,   0 ],
        [   8,  19,   4,  10,  24,  45,  21,  37,   9,  29,   0,   3,   1,   7,  11,  25,   0,   2,   0,   1 ],
        [  46,  42,   0,   1,   2,  10,  54,  51,  10,  30,   0,   2,   0,   2,   0,   1,   0,   1,   0,   0 ],
        [  28,  32,   0,   0,   3,  10,  75,  51,  14,  33,   0,   1,   0,   2,   0,   1,   1,   2,   0,   0 ],
        [ 100,  46,   0,   1,   3,   9,  21,  37,   5,  20,   0,   1,   0,   2,   1,   2,   0,   1,   0,   0 ],
        [  27,  29,   0,   1,   9,  25,  53,  51,  12,  34,   0,   1,   0,   3,   1,   5,   0,   2,   0,   0 ],
        [  80,  38,   0,   0,   1,   4,  69,  33,   5,  16,   0,   1,   0,   1,   0,   0,   0,   1,   0,   0 ],
        [  16,  20,   0,   0,   2,   8, 104,  49,  15,  33,   0,   1,   0,   1,   0,   1,   1,   1,   0,   0 ],
        [ 194,  16,   0,   0,   1,   1,   1,   9,   1,   3,   0,   0,   0,   1,   0,   1,   0,   0,   0,   0 ],
    ],
    // ModeAvailability = NearestOnly (situation 1)
    [
        [  41,  22,   1,   0,   1,  31,   0,   0,   0,   0,   0,   1,   1,   7,   0,   1,  98,  25,   4,  10 ],
        [ 123,  37,   6,   4,   1,  27,   0,   0,   0,   0,   5,   8,   1,   7,   0,   1,  12,  10,   0,   2 ],
        [  26,  14,  14,  12,   0,  24,   0,   0,   0,   0,  55,  17,   1,   9,   0,  36,   5,   7,   1,   3 ],
        [ 209,   5,   0,   0,   0,  27,   0,   0,   0,   0,   0,   1,   0,   1,   0,   1,   0,   0,   0,   0 ],
        [   2,   5,   4,   5,   0, 121,   0,   0,   0,   0,   0,   3,   2,   4,   1,   4,   2,   2,   0,   1 ],
        [ 175,   5,   0,   1,   0,  48,   0,   0,   0,   0,   0,   2,   0,   1,   0,   2,   0,   1,   0,   0 ],
        [  83,   5,   2,   3,   0, 102,   0,   0,   0,   0,   1,   3,   0,   2,   0,   1,   0,   0,   0,   0 ],
        [ 233,   6,   0,   0,   0,   8,   0,   0,   0,   0,   0,   1,   0,   1,   0,   0,   0,   1,   0,   0 ],
        [  34,  16, 112,  21,   1,  28,   0,   0,   0,   0,   6,   8,   1,   7,   0,   3,   2,   5,   0,   2 ],
        [ 159,  35,   2,   2,   0,  25,   0,   0,   0,   0,   3,   6,   0,   5,   0,   1,   4,   4,   0,   1 ],
        [  75,  39,   5,   7,   2,  48,   0,   0,   0,   0,   3,  11,   2,  16,   1,   4,   7,  10,   0,   2 ],
        [ 212,  21,   0,   1,   0,   9,   0,   0,   0,   0,   1,   2,   0,   2,   0,   0,   2,   2,   0,   0 ],
        [   4,   2,   0,   0,   0, 172,   0,   0,   0,   0,   0,   1,   0,   2,   0,   0,   2,   0,   0,   0 ],
        [ 187,  22,   1,   1,   0,  17,   0,   0,   0,   0,   3,   6,   0,   4,   0,   1,   4,   4,   0,   1 ],
        [ 133,   6,   1,   2,   1,  70,   0,   0,   0,   0,   0,   2,   0,   4,   0,   3,   1,   1,   0,   0 ],
        [ 251,   1,   0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    ],
    // ModeAvailability = Neither (situation 2)
    [
        [   2,   3,   2,   3,   0,   2,   0,   2,   0,   0,  11,   4,   1,   4,   0,   2,   3,   2,   0,   4 ],
        [  49,  46,   3,   4,   7,  31,  42,  41,   0,   0,   2,   6,   1,   7,   1,   4,   2,   4,   0,   1 ],
        [  26,  25,   1,   1,   2,  10,  67,  39,   0,   0,   1,   1,   0,  14,   0,   2,  31,  26,   1,   6 ],
        [ 103,  46,   1,   2,   2,  10,  33,  42,   0,   0,   1,   4,   0,   3,   0,   1,   1,   3,   0,   0 ],
        [  14,  31,   9,  13,  14,  54,  22,  29,   0,   0,   2,   6,   4,  18,   6,  13,   1,   5,   0,   1 ],
        [  85,  39,   0,   0,   1,   9,  69,  40,   0,   0,   0,   1,   0,   3,   0,   1,   2,   3,   0,   0 ],
        [  31,  28,   0,   0,   3,  14, 130,  34,   0,   0,   0,   1,   0,   3,   0,   1,   3,   3,   0,   1 ],
        [ 171,  25,   0,   0,   1,   5,  25,  21,   0,   0,   0,   1,   0,   1,   0,   0,   0,   0,   0,   0 ],
        [  17,  21,  68,  29,   6,  15,  13,  22,   0,   0,   6,  12,   3,  14,   4,  10,   1,   7,   0,   3 ],
        [  51,  39,   0,   1,   2,  12,  91,  44,   0,   0,   0,   2,   0,   3,   0,   1,   2,   3,   0,   1 ],
        [  81,  25,   0,   0,   2,   9, 106,  26,   0,   0,   0,   1,   0,   1,   0,   1,   1,   1,   0,   0 ],
        [ 140,  37,   0,   1,   1,   8,  24,  33,   0,   0,   1,   2,   0,   2,   0,   1,   1,   2,   0,   0 ],
        [  14,  23,   1,   3,  11,  53,  90,  31,   0,   0,   0,   3,   1,   5,   2,   6,   1,   2,   0,   0 ],
        [ 123,  29,   0,   0,   1,   7,  57,  30,   0,   0,   0,   1,   0,   1,   0,   1,   0,   1,   0,   0 ],
        [  13,  14,   0,   0,   4,  20, 175,  20,   0,   0,   0,   1,   0,   1,   0,   1,   1,   1,   0,   0 ],
        [ 202,  23,   0,   0,   1,   3,   2,   9,   0,   0,   0,   1,   0,   1,   0,   1,   0,   0,   0,   0 ],
    ],
];

/// Probability that the current MB's mode is **the same as** the
/// previously coded MB's mode (spec §10).
///
/// Computed from `probXmitted[k][i]` (the "same-as-prior" half of
/// each mode-pair) via the verbatim §10 formula
///
/// ```text
/// probModeSame[k][i] = 255 - 255 * probXmitted[k][i*2]
///                          / (1 + probXmitted[k][i*2+1] + probXmitted[k][i*2])
/// ```
///
/// Note `probXmitted[k][i*2]` is **the same-as-prior count** (the
/// even-indexed entry of the pair) and `probXmitted[k][i*2+1]` is
/// the different-from-prior count (the odd-indexed one).
///
/// The result is the BoolCoder node probability the §10
/// `VP6_DecodeMode` traversal consults at the decision tree's root
/// (the `Same As Last` shortcut in Figure 10).
///
/// `availability` selects the row; `last_mode` selects the column
/// (the `i` index, the previous MB's mode).
///
/// # Panics
///
/// `last_mode.index() < NUM_CODING_MODES` is guaranteed by the enum
/// surface, so no runtime check is needed. Callers passing raw indices
/// must bound them to `0..NUM_CODING_MODES`.
#[inline]
pub fn probability_mode_same(
    prob_xmitted: &[[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
    availability: ModeAvailability,
    last_mode: CodingMode,
) -> u8 {
    let k = availability.index();
    let i = last_mode.index();
    // Spec formula: 255 - 255 * probXmitted[k][i*2] /
    //                       (1 + probXmitted[k][i*2+1] + probXmitted[k][i*2])
    let p_same = prob_xmitted[k][i * 2] as u32;
    let p_diff = prob_xmitted[k][i * 2 + 1] as u32;
    let denom = 1 + p_diff + p_same; // never zero (1+)
    let num = 255u32 * p_same;
    let q = num / denom; // truncating integer division per spec pseudocode
    (255 - q) as u8
}

/// One row of the per-node `ModeDecisionTree`, indexed by the 9
/// internal-node positions of Figure 10.
pub type ModeDecisionTreeRow = [u8; NUM_MODE_DECISION_NODES];

/// Full `ModeDecisionTree[3][10][9]` — per-`(availability, last_mode)`
/// per-node BoolCoder probabilities for the §10 mode-decoding
/// decision tree (Figure 10).
pub type ModeDecisionTree = [[ModeDecisionTreeRow; NUM_CODING_MODES]; NUM_PROBABILITY_SITUATIONS];

/// Compute one node's BoolCoder probability for the §10 mode-decoding
/// decision tree (Figure 10), given the current `probXmitted` table.
///
/// Per the spec the per-tree row is derived from a per-mode "weight"
/// array `C[10]` synthesised from `probXmitted[k][i]` as
///
/// ```text
/// for ( j = 0; j < 10; j++ )
///     if ( j == i )    C[j] = 0
///     else             C[j] = 100 * probXmitted[k][j*2+1]
/// ```
///
/// (so the "weight" for mode `j` when transitioning *away from* the
/// previous mode `i` is the scaled "different-from-prior" entry).
/// The node probabilities follow the binary-tree formula
///
/// ```text
/// ModeDecisionTree[k][i][n] = 1 + 255 * (sum of C[] in left subtree)
///                                   / (1 + sum of C[] across both subtrees)
/// ```
///
/// per Figure 10's node-by-node accounting (see spec §10 for the
/// nine explicit cases, also transcribed inline below).
///
/// # Panics
///
/// Indices outside `availability.index() in 0..3`,
/// `last_mode.index() in 0..10`, `node in 0..9` cannot occur via the
/// enum surface; raw-index callers must bound their inputs.
pub fn mode_decision_tree_node_probability(
    prob_xmitted: &[[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
    availability: ModeAvailability,
    last_mode: CodingMode,
    node: usize,
) -> u8 {
    assert!(
        node < NUM_MODE_DECISION_NODES,
        "node index {} out of range",
        node
    );
    let k = availability.index();
    let i = last_mode.index();

    // Build the per-mode weight array C[10]. Mode j == i contributes
    // zero (the "Same As Last" shortcut), other modes contribute
    // 100 * probXmitted[k][j*2+1].
    let mut c = [0u32; NUM_CODING_MODES];
    let mut total = 0u32;
    for (j, slot) in c.iter_mut().enumerate() {
        if j != i {
            *slot = 100 * prob_xmitted[k][j * 2 + 1] as u32;
            total += *slot;
        }
    }

    // Per Figure 10:
    //
    //   Node 0: branch between {0,1,2,3,4} and {5,6,7,8,9}
    //                          left  = NO_MV + PLUS_MV + NEAREST_MV + NEAR_MV (4 inter-prev)
    //                                 (and INTRA at the boundary — see below)
    //   Node 1: within left subtree, branch between {NO_MV, PLUS_MV, NEAREST_MV, NEAR_MV}
    //   Node 2: within right subtree, branch between {INTRA, FOURMV} and golden modes
    //   Node 3: split {NO_MV} vs {PLUS_MV} on left
    //   Node 4: split {NEAREST_MV} vs {NEAR_MV} on left
    //   Node 5: split {INTRA} vs {FOURMV}
    //   Node 6: branch between {USING_GOLDEN, GOLDEN_MV} and {GOLD_NEAREST_MV, GOLD_NEAR_MV}
    //   Node 7: split {USING_GOLDEN} vs {GOLDEN_MV}
    //   Node 8: split {GOLD_NEAREST_MV} vs {GOLD_NEAR_MV}
    //
    // The numerators below come straight from the spec listing.
    let (left_sum, branch_sum) = match node {
        0 => {
            let left = c[CodingMode::InterNoMv.index()]
                + c[CodingMode::InterPlusMv.index()]
                + c[CodingMode::InterNearestMv.index()]
                + c[CodingMode::InterNearMv.index()];
            (left, total)
        }
        1 => {
            let left = c[CodingMode::InterNoMv.index()] + c[CodingMode::InterPlusMv.index()];
            let branch = c[CodingMode::InterNoMv.index()]
                + c[CodingMode::InterPlusMv.index()]
                + c[CodingMode::InterNearestMv.index()]
                + c[CodingMode::InterNearMv.index()];
            (left, branch)
        }
        2 => {
            let left = c[CodingMode::Intra.index()] + c[CodingMode::InterFourMv.index()];
            let branch = c[CodingMode::Intra.index()]
                + c[CodingMode::InterFourMv.index()]
                + c[CodingMode::UsingGolden.index()]
                + c[CodingMode::GoldenMv.index()]
                + c[CodingMode::GoldNearestMv.index()]
                + c[CodingMode::GoldNearMv.index()];
            (left, branch)
        }
        3 => {
            let left = c[CodingMode::InterNoMv.index()];
            let branch = c[CodingMode::InterNoMv.index()] + c[CodingMode::InterPlusMv.index()];
            (left, branch)
        }
        4 => {
            let left = c[CodingMode::InterNearestMv.index()];
            let branch = c[CodingMode::InterNearestMv.index()] + c[CodingMode::InterNearMv.index()];
            (left, branch)
        }
        5 => {
            let left = c[CodingMode::Intra.index()];
            let branch = c[CodingMode::Intra.index()] + c[CodingMode::InterFourMv.index()];
            (left, branch)
        }
        6 => {
            let left = c[CodingMode::UsingGolden.index()] + c[CodingMode::GoldenMv.index()];
            let branch = c[CodingMode::UsingGolden.index()]
                + c[CodingMode::GoldenMv.index()]
                + c[CodingMode::GoldNearestMv.index()]
                + c[CodingMode::GoldNearMv.index()];
            (left, branch)
        }
        7 => {
            let left = c[CodingMode::UsingGolden.index()];
            let branch = c[CodingMode::UsingGolden.index()] + c[CodingMode::GoldenMv.index()];
            (left, branch)
        }
        8 => {
            let left = c[CodingMode::GoldNearestMv.index()];
            let branch = c[CodingMode::GoldNearestMv.index()] + c[CodingMode::GoldNearMv.index()];
            (left, branch)
        }
        _ => unreachable!(),
    };

    // Per spec: ModeDecisionTree[k][i][n] = 1 + 255 * left / (1 + branch)
    let denom = 1 + branch_sum;
    let num = 255u32 * left_sum;
    let p = 1 + num / denom; // truncating int division per spec pseudocode
                             // Clip to 1..=255 (BoolCoder probabilities live in this range per
                             // §7). The +1 floor handles the lower bound; cap defensively for
                             // the upper.
    p.min(255) as u8
}

/// Build the full `ModeDecisionTree[3][10][9]` array from the
/// current `probXmitted` table per spec §10.
///
/// The output is consumed by the §10 `VP6_DecodeMode` BoolCoder
/// traversal: at each node it reads `B(ModeDecisionTree[type][lastmode]
/// [node])` to choose between left and right subtrees. The traversal
/// itself depends on the §7.3 BoolCoder and stays deferred until the
/// DOCS-GAP is resolved; this function (and the per-node helper above)
/// produce the static probability table the traversal would consult.
pub fn build_mode_decision_tree(
    prob_xmitted: &[[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
) -> ModeDecisionTree {
    let mut out: ModeDecisionTree =
        [[[0u8; NUM_MODE_DECISION_NODES]; NUM_CODING_MODES]; NUM_PROBABILITY_SITUATIONS];
    for (k, availability) in ModeAvailability::ALL.iter().enumerate() {
        for (i, last_mode) in CodingMode::ALL.iter().enumerate() {
            for (node, slot) in out[k][i].iter_mut().enumerate() {
                *slot = mode_decision_tree_node_probability(
                    prob_xmitted,
                    *availability,
                    *last_mode,
                    node,
                );
            }
        }
    }
    out
}

/// Build the `probModeSame[3][10]` companion array from the current
/// `probXmitted` table per spec §10.
///
/// Used by §10's `VP6_DecodeMode` BoolCoder traversal as the root-node
/// "Same As Last" probability. Each entry is the value
/// [`probability_mode_same`] returns for the matching
/// `(availability, last_mode)` pair.
pub fn build_probability_mode_same(
    prob_xmitted: &[[u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS],
) -> [[u8; NUM_CODING_MODES]; NUM_PROBABILITY_SITUATIONS] {
    let mut out = [[0u8; NUM_CODING_MODES]; NUM_PROBABILITY_SITUATIONS];
    for (k, availability) in ModeAvailability::ALL.iter().enumerate() {
        for (i, last_mode) in CodingMode::ALL.iter().enumerate() {
            out[k][i] = probability_mode_same(prob_xmitted, *availability, *last_mode);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- enum surface --------

    #[test]
    fn coding_mode_indices_match_spec_table_4_order() {
        assert_eq!(CodingMode::InterNoMv.index(), 0);
        assert_eq!(CodingMode::Intra.index(), 1);
        assert_eq!(CodingMode::InterPlusMv.index(), 2);
        assert_eq!(CodingMode::InterNearestMv.index(), 3);
        assert_eq!(CodingMode::InterNearMv.index(), 4);
        assert_eq!(CodingMode::UsingGolden.index(), 5);
        assert_eq!(CodingMode::GoldenMv.index(), 6);
        assert_eq!(CodingMode::InterFourMv.index(), 7);
        assert_eq!(CodingMode::GoldNearestMv.index(), 8);
        assert_eq!(CodingMode::GoldNearMv.index(), 9);
    }

    #[test]
    fn coding_mode_from_index_round_trip() {
        for (i, mode) in CodingMode::ALL.iter().enumerate() {
            assert_eq!(CodingMode::from_index(i), Some(*mode));
            assert_eq!(mode.index(), i);
        }
        assert_eq!(CodingMode::from_index(10), None);
        assert_eq!(CodingMode::from_index(usize::MAX), None);
    }

    #[test]
    fn coding_mode_all_length_matches_constant() {
        assert_eq!(CodingMode::ALL.len(), NUM_CODING_MODES);
    }

    #[test]
    fn uses_golden_partitions_modes_correctly() {
        // Golden-Frame predicting modes: USING_GOLDEN, GOLDEN_MV,
        // GOLD_NEAREST_MV, GOLD_NEAR_MV.
        let goldens = [
            CodingMode::UsingGolden,
            CodingMode::GoldenMv,
            CodingMode::GoldNearestMv,
            CodingMode::GoldNearMv,
        ];
        for mode in CodingMode::ALL.iter() {
            assert_eq!(mode.uses_golden(), goldens.contains(mode), "{}", mode);
        }
    }

    #[test]
    fn is_intra_only_for_intra_mode() {
        for mode in CodingMode::ALL.iter() {
            assert_eq!(mode.is_intra(), *mode == CodingMode::Intra, "{}", mode);
        }
    }

    #[test]
    fn carries_new_mv_only_for_plus_mv_variants() {
        let new_mv = [
            CodingMode::InterPlusMv,
            CodingMode::GoldenMv,
            CodingMode::InterFourMv,
        ];
        for mode in CodingMode::ALL.iter() {
            assert_eq!(mode.carries_new_mv(), new_mv.contains(mode), "{}", mode);
        }
    }

    #[test]
    fn coding_mode_display_strings_match_spec_names() {
        assert_eq!(format!("{}", CodingMode::InterNoMv), "CODE_INTER_NO_MV");
        assert_eq!(format!("{}", CodingMode::Intra), "CODE_INTRA");
        assert_eq!(format!("{}", CodingMode::GoldNearMv), "CODE_GOLD_NEAR_MV");
    }

    #[test]
    fn mode_availability_indices_match_spec_table_5() {
        assert_eq!(ModeAvailability::NearestAndNear.index(), 0);
        assert_eq!(ModeAvailability::NearestOnly.index(), 1);
        assert_eq!(ModeAvailability::Neither.index(), 2);
    }

    #[test]
    fn mode_availability_from_index_round_trip() {
        for (i, av) in ModeAvailability::ALL.iter().enumerate() {
            assert_eq!(ModeAvailability::from_index(i), Some(*av));
        }
        assert_eq!(ModeAvailability::from_index(3), None);
    }

    #[test]
    fn mode_availability_from_neighbours_matches_truth_table() {
        assert_eq!(
            ModeAvailability::from_neighbours(true, true),
            ModeAvailability::NearestAndNear
        );
        assert_eq!(
            ModeAvailability::from_neighbours(true, false),
            ModeAvailability::NearestOnly
        );
        // Near-without-nearest is degenerate; folds to Neither.
        assert_eq!(
            ModeAvailability::from_neighbours(false, true),
            ModeAvailability::Neither
        );
        assert_eq!(
            ModeAvailability::from_neighbours(false, false),
            ModeAvailability::Neither
        );
    }

    // -------- NEAR_MACROBLOCKS --------

    #[test]
    fn near_macroblocks_has_twelve_entries() {
        assert_eq!(NEAR_MACROBLOCKS.len(), 12);
    }

    #[test]
    fn near_macroblocks_spec_verbatim_first_entries() {
        // Spec §10: first four entries are the 4-connected and one
        // diagonal neighbour at distance 1.
        assert_eq!(NEAR_MACROBLOCKS[0], (-1, 0)); // above
        assert_eq!(NEAR_MACROBLOCKS[1], (0, -1)); // left
        assert_eq!(NEAR_MACROBLOCKS[2], (-1, -1)); // above-left
        assert_eq!(NEAR_MACROBLOCKS[3], (-1, 1)); // above-right
    }

    #[test]
    fn near_macroblocks_spec_verbatim_distance_two_entries() {
        // Indices 4..12 are the eight distance-2 neighbours in spec
        // traversal order.
        assert_eq!(NEAR_MACROBLOCKS[4], (-2, 0));
        assert_eq!(NEAR_MACROBLOCKS[5], (0, -2));
        assert_eq!(NEAR_MACROBLOCKS[6], (-1, -2));
        assert_eq!(NEAR_MACROBLOCKS[7], (-2, -1));
        assert_eq!(NEAR_MACROBLOCKS[8], (-2, 1));
        assert_eq!(NEAR_MACROBLOCKS[9], (-1, 2));
        assert_eq!(NEAR_MACROBLOCKS[10], (-2, -2));
        assert_eq!(NEAR_MACROBLOCKS[11], (-2, 2));
    }

    #[test]
    fn near_macroblocks_all_unique() {
        let mut seen = std::collections::HashSet::new();
        for &offset in &NEAR_MACROBLOCKS {
            assert!(seen.insert(offset), "duplicate offset {:?}", offset);
        }
    }

    #[test]
    fn near_macroblocks_never_below_current() {
        // The spec only ever traverses already-decoded neighbours. For
        // raster-order decoding that means row offset must be <= 0,
        // and where row offset == 0 the column offset must be < 0.
        for (idx, &(dr, dc)) in NEAR_MACROBLOCKS.iter().enumerate() {
            assert!(dr <= 0, "entry {} has dr={} (below current)", idx, dr);
            if dr == 0 {
                assert!(dc < 0, "entry {} has dc={} on same row", idx, dc);
            }
        }
    }

    // -------- VP6_BASELINE_XMITTED_PROBS --------

    #[test]
    fn baseline_xmitted_probs_shape_matches_spec() {
        assert_eq!(VP6_BASELINE_XMITTED_PROBS.len(), NUM_PROBABILITY_SITUATIONS);
        for row in &VP6_BASELINE_XMITTED_PROBS {
            assert_eq!(row.len(), PROB_XMITTED_ROW_LEN);
        }
    }

    #[test]
    fn baseline_xmitted_probs_situation_0_spec_spot_values() {
        // §10 listing for ModeAvailability=0: {42, 69, 2, 1, 7, 1, 42, 44, 22, 6, 3, 1, 2, 0, 5, 1, 1, 0, 0, 0}
        let row = &VP6_BASELINE_XMITTED_PROBS[0];
        assert_eq!(row[0], 42);
        assert_eq!(row[1], 69);
        assert_eq!(row[6], 42);
        assert_eq!(row[7], 44);
        assert_eq!(row[8], 22);
        assert_eq!(row[9], 6);
        assert_eq!(row[19], 0);
    }

    #[test]
    fn baseline_xmitted_probs_situation_1_spec_spot_values() {
        // {8, 229, 1, 1, 8, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0}
        let row = &VP6_BASELINE_XMITTED_PROBS[1];
        assert_eq!(row[0], 8);
        assert_eq!(row[1], 229);
        // Golden-mode pairs (indices 10..) are all very small or zero.
        assert_eq!(row[6], 0);
        assert_eq!(row[7], 0);
        assert_eq!(row[16], 1);
        assert_eq!(row[17], 1);
    }

    #[test]
    fn baseline_xmitted_probs_situation_2_spec_spot_values() {
        // {35, 122, 1, 1, 6, 1, 34, 46, 0, 0, 2, 1, 1, 0, 1, 0, 1, 1, 0, 0}
        let row = &VP6_BASELINE_XMITTED_PROBS[2];
        assert_eq!(row[0], 35);
        assert_eq!(row[1], 122);
        assert_eq!(row[6], 34);
        assert_eq!(row[7], 46);
    }

    #[test]
    fn baseline_xmitted_probs_entries_are_byte_sized() {
        // u8 storage enforces "no entry exceeds 255"; this test pins
        // the implicit invariant for the audit trail by walking every
        // cell and inspecting its width.
        for row in &VP6_BASELINE_XMITTED_PROBS {
            assert_eq!(row.len(), PROB_XMITTED_ROW_LEN);
            for &v in row.iter() {
                let _: u8 = v; // type-level guarantee the entries fit in a byte.
            }
        }
    }

    // -------- VP6_MODE_VQ --------

    #[test]
    fn mode_vq_shape_matches_spec() {
        assert_eq!(VP6_MODE_VQ.len(), NUM_PROBABILITY_SITUATIONS);
        for block in &VP6_MODE_VQ {
            assert_eq!(block.len(), NUM_MODE_VQ_VECTORS);
            for vec in block {
                assert_eq!(vec.len(), PROB_XMITTED_ROW_LEN);
            }
        }
    }

    #[test]
    fn mode_vq_situation_0_first_vector_spec_spot_values() {
        // First vector of {[3][16][20]} block 0:
        // {9, 15, 32, 25, 7, 19, 9, 21, 1, 12, 14, 12, 3, 18, 14, 23, 3, 10, 0, 4}
        let v = &VP6_MODE_VQ[0][0];
        assert_eq!(v[0], 9);
        assert_eq!(v[1], 15);
        assert_eq!(v[2], 32);
        assert_eq!(v[19], 4);
    }

    #[test]
    fn mode_vq_situation_0_last_vector_spec_spot_values() {
        // Last vector of block 0:
        // {194, 16, 0, 0, 1, 1, 1, 9, 1, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0}
        let v = &VP6_MODE_VQ[0][15];
        assert_eq!(v[0], 194);
        assert_eq!(v[1], 16);
        assert_eq!(v[7], 9);
    }

    #[test]
    fn mode_vq_situation_1_first_vector_spec_spot_values() {
        // {41, 22, 1, 0, 1, 31, 0, 0, 0, 0, 0, 1, 1, 7, 0, 1, 98, 25, 4, 10}
        let v = &VP6_MODE_VQ[1][0];
        assert_eq!(v[0], 41);
        assert_eq!(v[1], 22);
        assert_eq!(v[5], 31);
        assert_eq!(v[16], 98);
        assert_eq!(v[19], 10);
    }

    #[test]
    fn mode_vq_situation_2_first_vector_spec_spot_values() {
        // {2, 3, 2, 3, 0, 2, 0, 2, 0, 0, 11, 4, 1, 4, 0, 2, 3, 2, 0, 4}
        let v = &VP6_MODE_VQ[2][0];
        assert_eq!(v[0], 2);
        assert_eq!(v[10], 11);
        assert_eq!(v[19], 4);
    }

    #[test]
    fn mode_vq_situation_2_last_vector_spec_spot_values() {
        // {202, 23, 0, 0, 1, 3, 2, 9, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0}
        let v = &VP6_MODE_VQ[2][15];
        assert_eq!(v[0], 202);
        assert_eq!(v[1], 23);
        assert_eq!(v[5], 3);
        assert_eq!(v[7], 9);
    }

    // -------- probModeSame --------

    #[test]
    fn probability_mode_same_with_zero_inputs_returns_255() {
        // probXmitted all-zeroes: numerator 255*0 == 0, denom 1+0+0 == 1,
        // result = 255 - 0 == 255.
        let zeros = [[0u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS];
        for av in ModeAvailability::ALL.iter() {
            for mode in CodingMode::ALL.iter() {
                assert_eq!(probability_mode_same(&zeros, *av, *mode), 255);
            }
        }
    }

    #[test]
    fn probability_mode_same_baseline_for_intra_inter() {
        // Situation 0, last_mode = Intra (index 1):
        // probXmitted[0][2] == 2, probXmitted[0][3] == 1
        // p_same = 255 - 255 * 2 / (1 + 1 + 2) = 255 - 255*2/4 = 255 - 127 = 128.
        let p = probability_mode_same(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::Intra,
        );
        assert_eq!(p, 128, "computed = 255 - 255*2/4 = 128");
    }

    #[test]
    fn probability_mode_same_baseline_situation_1_inter_no_mv() {
        // Situation 1, last_mode = InterNoMv (index 0):
        // probXmitted[1][0] = 8, probXmitted[1][1] = 229.
        // p_same = 255 - 255 * 8 / (1 + 229 + 8) = 255 - 2040/238 = 255 - 8 = 247.
        let p = probability_mode_same(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestOnly,
            CodingMode::InterNoMv,
        );
        assert_eq!(p, 247);
    }

    #[test]
    fn build_probability_mode_same_matches_per_element_helper() {
        let table = build_probability_mode_same(&VP6_BASELINE_XMITTED_PROBS);
        for (k, av) in ModeAvailability::ALL.iter().enumerate() {
            for (i, mode) in CodingMode::ALL.iter().enumerate() {
                assert_eq!(
                    table[k][i],
                    probability_mode_same(&VP6_BASELINE_XMITTED_PROBS, *av, *mode),
                    "(av={av}, last={mode})"
                );
            }
        }
    }

    // -------- ModeDecisionTree --------

    #[test]
    fn decision_tree_full_build_has_correct_shape() {
        let tree = build_mode_decision_tree(&VP6_BASELINE_XMITTED_PROBS);
        assert_eq!(tree.len(), NUM_PROBABILITY_SITUATIONS);
        for plane in &tree {
            assert_eq!(plane.len(), NUM_CODING_MODES);
            for row in plane {
                assert_eq!(row.len(), NUM_MODE_DECISION_NODES);
            }
        }
    }

    #[test]
    fn decision_tree_zero_input_collapses_to_floor_probability() {
        // All-zero probXmitted: every C[j] is zero, total is zero,
        // and every node's left_sum/branch_sum is 0/(1+0) = 0,
        // giving 1 + 0 = 1.
        let zeros = [[0u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS];
        let tree = build_mode_decision_tree(&zeros);
        for plane in &tree {
            for row in plane {
                for &p in row {
                    assert_eq!(p, 1);
                }
            }
        }
    }

    #[test]
    fn decision_tree_probabilities_within_valid_boolcoder_range() {
        // §7.3: 1 <= Node Probability <= 255. The +1 floor guarantees
        // the lower bound; the .min(255) cap protects the upper.
        let tree = build_mode_decision_tree(&VP6_BASELINE_XMITTED_PROBS);
        for plane in &tree {
            for row in plane {
                for &p in row {
                    assert!(p >= 1, "probability {} below floor", p);
                }
            }
        }
    }

    #[test]
    fn decision_tree_per_node_helper_matches_full_build() {
        let tree = build_mode_decision_tree(&VP6_BASELINE_XMITTED_PROBS);
        for (k, av) in ModeAvailability::ALL.iter().enumerate() {
            for (i, mode) in CodingMode::ALL.iter().enumerate() {
                for (node, &built) in tree[k][i].iter().enumerate() {
                    let got = mode_decision_tree_node_probability(
                        &VP6_BASELINE_XMITTED_PROBS,
                        *av,
                        *mode,
                        node,
                    );
                    assert_eq!(built, got, "(k={k}, i={i}, node={node})");
                }
            }
        }
    }

    #[test]
    fn decision_tree_node_5_intra_vs_fourmv_baseline() {
        // Node 5 splits {INTRA} vs {FOURMV}. With baseline ProbXmitted
        // situation 0:
        //   C[INTRA] = 100 * probXmitted[0][3] = 100 * 1 = 100  (when i != 1)
        //   C[FOURMV]= 100 * probXmitted[0][15] = 100 * 1 = 100 (when i != 7)
        // Pick a last_mode that excludes neither (lastmode = NEAR_MV = 4).
        // left_sum  = 100, branch_sum = 200
        // p = 1 + 255*100/(1+200) = 1 + 25500/201 = 1 + 126 = 127.
        let p = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::InterNearMv,
            5,
        );
        assert_eq!(p, 127);
    }

    #[test]
    fn decision_tree_node_0_root_split_baseline() {
        // Node 0 splits {INTER_NO_MV, INTER_PLUS_MV, INTER_NEAREST_MV,
        // INTER_NEAR_MV} (the 4 prev-frame inter modes) vs the rest.
        // With lastmode = 0 (CODE_INTER_NO_MV), the INTER_NO_MV slot
        // in C[] is zeroed. With baseline ProbXmitted situation 0:
        //   C[INTER_NO_MV]      = 0      (i == 0, dropped)
        //   C[INTRA]            = 100 * probXmitted[0][3]  = 100*1   = 100
        //   C[INTER_PLUS_MV]    = 100 * probXmitted[0][5]  = 100*1   = 100
        //   C[INTER_NEAREST_MV] = 100 * probXmitted[0][7]  = 100*44  = 4400
        //   C[INTER_NEAR_MV]    = 100 * probXmitted[0][9]  = 100*6   = 600
        //   C[USING_GOLDEN]     = 100 * probXmitted[0][11] = 100*1   = 100
        //   C[GOLDEN_MV]        = 100 * probXmitted[0][13] = 100*0   = 0
        //   C[FOURMV]           = 100 * probXmitted[0][15] = 100*1   = 100
        //   C[GOLD_NEAREST_MV]  = 100 * probXmitted[0][17] = 100*0   = 0
        //   C[GOLD_NEAR_MV]     = 100 * probXmitted[0][19] = 100*0   = 0
        // left_sum  (4 prev-frame inter modes including the zeroed
        // INTER_NO_MV) = 0 + 100 + 4400 + 600 = 5100
        // total = 5100 + 100 + 100 + 0 + 100 + 0 + 0 = 5400
        // p = 1 + 255 * 5100 / (1 + 5400) = 1 + 1300500/5401
        //   = 1 + 240 = 241.
        let p = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::InterNoMv,
            0,
        );
        assert_eq!(p, 241);
    }

    #[test]
    fn decision_tree_node_3_no_mv_vs_plus_mv_baseline() {
        // Node 3: {NO_MV} vs {PLUS_MV}. With situation 0, lastmode = INTRA (1):
        //   C[INTER_NO_MV]   = 100 * probXmitted[0][1] = 100 * 69 = 6900
        //   C[INTER_PLUS_MV] = 100 * probXmitted[0][5] = 100 * 1  = 100
        //   left = 6900, branch = 7000
        //   p = 1 + 255*6900/(1+7000) = 1 + 1759500/7001 = 1 + 251 = 252.
        let p = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::Intra,
            3,
        );
        assert_eq!(p, 252);
    }

    #[test]
    fn decision_tree_node_7_using_golden_vs_golden_mv_baseline_situation_1() {
        // Node 7: {USING_GOLDEN} vs {GOLDEN_MV}. With situation 1,
        // lastmode = INTRA (1):
        //   probXmitted[1][11]=1 (GOLDEN-pair-half = USING_GOLDEN diff)
        //   probXmitted[1][13]=0 (GOLDEN_MV diff)
        //   C[USING_GOLDEN] = 100 * 1 = 100
        //   C[GOLDEN_MV]    = 100 * 0 = 0
        //   left=100, branch=100
        //   p = 1 + 255*100/(1+100) = 1 + 25500/101 = 1 + 252 = 253.
        let p = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestOnly,
            CodingMode::Intra,
            7,
        );
        assert_eq!(p, 253);
    }

    #[test]
    #[should_panic]
    fn decision_tree_node_index_out_of_range_panics() {
        let _ = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::Intra,
            9, // valid range 0..9
        );
    }

    #[test]
    fn decision_tree_lastmode_zeros_the_same_branch_in_weights() {
        // Per spec: when j == i, C[j] is zeroed. This means the weight
        // for "same as last" is dropped from every tree-row sum that
        // would have included it. Sanity-check via two computations
        // with different `lastmode` and the same tree node.
        //
        // For node 5 (INTRA vs FOURMV), changing lastmode FROM intra
        // (C[INTRA]=0, since i==1) TO four_mv (C[FOURMV]=0, since i==7)
        // swaps which weight is zeroed. With baseline situation 0:
        //   * lastmode=Intra:  C[INTRA]=0,        C[FOURMV]=100 → p = 1 + 255*0/(1+100) = 1
        //   * lastmode=FourMv: C[INTRA]=100*69/.. — actually probXmitted[0][3]=1, so C[INTRA]=100;
        //                       C[FOURMV]=0 → p = 1 + 255*100/(1+100) = 253
        let p_when_lastmode_intra = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::Intra,
            5,
        );
        assert_eq!(p_when_lastmode_intra, 1);

        let p_when_lastmode_fourmv = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::NearestAndNear,
            CodingMode::InterFourMv,
            5,
        );
        assert_eq!(p_when_lastmode_fourmv, 253);
    }

    #[test]
    fn decision_tree_node_8_gold_nearest_vs_gold_near_baseline_situation_2() {
        // Node 8: {GOLD_NEAREST_MV} vs {GOLD_NEAR_MV}. Situation 2,
        // lastmode = INTER_NO_MV (0):
        //   probXmitted[2][17] = 1 (GOLD_NEAREST_MV diff)
        //   probXmitted[2][19] = 0 (GOLD_NEAR_MV diff)
        //   C[GOLD_NEAREST_MV] = 100; C[GOLD_NEAR_MV] = 0
        //   left=100, branch=100
        //   p = 1 + 255*100/(1+100) = 1 + 252 = 253.
        let p = mode_decision_tree_node_probability(
            &VP6_BASELINE_XMITTED_PROBS,
            ModeAvailability::Neither,
            CodingMode::InterNoMv,
            8,
        );
        assert_eq!(p, 253);
    }

    #[test]
    fn decision_tree_mode_vq_seed_round_trips_through_build() {
        // Feeding any VP6_ModeVq[k] vector into build_mode_decision_tree
        // (replicated across all 3 availability rows) must produce a
        // valid tree (every probability >= 1; the type u8 caps the
        // upper bound at 255 already).
        for plane in VP6_MODE_VQ.iter() {
            for vec in plane.iter() {
                let mut probs = [[0u8; PROB_XMITTED_ROW_LEN]; NUM_PROBABILITY_SITUATIONS];
                for row in probs.iter_mut() {
                    *row = *vec;
                }
                let tree = build_mode_decision_tree(&probs);
                for plane in &tree {
                    for row in plane {
                        for &p in row {
                            assert!(p >= 1);
                        }
                    }
                }
            }
        }
    }
}

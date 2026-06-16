//! VP6 custom scan order (spec §12.2).
//!
//! In addition to the §12.1 default zig-zag scan order, VP6 supports
//! per-frame *custom* scan orders. §12.2: "The use of custom scan
//! orders is an encoder decision and is signaled to the decoder using
//! the ScanOrderUpdateFlag (see Table 17)."
//!
//! Custom scan orders are expressed as a *band assignment*: §12.2
//! splits the 63 AC positions (numbered 1 to 63) of the modified scan
//! order into the 16 bands of Table 16, and "to specify a custom scan
//! order, each AC coefficient (in zig zag order) is assigned to one of
//! the above bands. Within each band the coefficients are then sorted
//! into ascending order based upon the original zig-zag scan order."
//! §12.2 fixes the identifier space: "All references below to specific
//! AC coefficients refer to their position in the standard zig-zag
//! scan order as shown in Figure 14. For example AC2 would refer to
//! the second AC coefficient in zig-zag order that corresponds to
//! coefficient 8 in the original raster order."
//!
//! Per-frame lifetime (§12.2):
//!
//! * "If ScanOrderUpdateFlag indicates that there is no custom
//!   scan-order for a frame, the scan order must be reset to the
//!   default."
//! * "For intra-coded frames the scan order is first set to the
//!   appropriate default. This default is then updated using delta
//!   information encoded in the bitstream. For inter-coded frames
//!   deltas are applied to the custom scan order used in the previous
//!   frame rather than to the one of the default scan orders."
//! * "In all scan orders the first DCT coefficient is always the DC
//!   coefficient."
//!
//! Bitstream placement: "Custom scan order updates are read as part of
//! the functional block 'Coefficient Probability Updates' (see Figure
//! 2-Figure 5)" — Figure 5 orders the block as scan-order updates,
//! then the §13.3.3 zero-run probability updates, then the §13.3 AC
//! probability updates.
//!
//! Table 17 gives the wire format: `ScanOrderUpdateFlag` is a `b(1)`
//! (one fixed-probability-128 BoolCoder bit, §3 nomenclature),
//! followed — only when the flag is `1` — by 63 sets of
//! `CoeffBandUpdateFlag` (`B(x)` against the per-coefficient
//! [`COEFF_BAND_UPDATE_FLAG_PROBS`] bank) and, when that flag is `1`,
//! `NewCoeffBand` (`b(4)`, "the new band for the coefficient").
//!
//! This is the **seventh** BoolCoder-consuming layer (after rounds
//! 16's §13.2.1 DC, 17's §13.3.1 AC, 19's §13.3.3.1 ZRL, 20's
//! §13.2/§13.3/§13.3.3 probability updates, 21's §11.1 MV component
//! decoder and 22's §11.2 MV probability updates); it composes only
//! the round-15 [`BoolCoder`] primitives (`decode_b1` / `decode_bool`
//! / `decode_b`) over the verbatim §12.2 tables.

use crate::bool_coder::BoolCoder;
use crate::scan::DEFAULT_SCAN_ORDER;
use crate::Error;

/// Number of custom scan order bands (§12.2 Table 16: band numbers
/// `0..=15`).
///
/// The Table 17 `NewCoeffBand` field is a `b(4)` — a four-bit value
/// `0..=15` — so every decodable band specifier addresses a valid
/// Table 16 band.
pub const NUM_SCAN_BANDS: usize = 16;

/// Number of AC coefficient positions a scan order arranges (§12.2:
/// "The 63 AC positions (numbered 1 to 63)").
pub const NUM_AC_POSITIONS: usize = 63;

/// The §12.2 Table 16 custom scan order bands, as inclusive
/// `(first_position, last_position)` ranges of the modified scan
/// order, indexed by band number `0..=15`.
///
/// Verbatim from Table 16 ("Custom scan order bands"): band 0 holds
/// position 1; band 1 holds positions 2 to 4; … band 15 holds
/// positions 58 to 63. The 16 ranges tile the AC position space
/// `1..=63` contiguously (pinned by the
/// `band_ranges_tile_ac_positions` test).
pub const CUSTOM_SCAN_BAND_RANGES: [(u8, u8); NUM_SCAN_BANDS] = [
    (1, 1),   // band 0
    (2, 4),   // band 1
    (5, 10),  // band 2
    (11, 12), // band 3
    (13, 15), // band 4
    (16, 19), // band 5
    (20, 21), // band 6
    (22, 26), // band 7
    (27, 28), // band 8
    (29, 34), // band 9
    (35, 36), // band 10
    (37, 42), // band 11
    (43, 48), // band 12
    (49, 53), // band 13
    (54, 57), // band 14
    (58, 63), // band 15
];

/// The §12.2 `CoeffBandUpdateFlagProbs[64]` table: "the probabilities
/// used for decoding CoeffBandUpdateFlag for each of the AC
/// coefficients in standard zig-zag order."
///
/// Verbatim from the §12.2 listing. The spec prints the first entry
/// as `NA` and explains: "The first entry in the table is a dummy
/// entry for the DC coefficient. This always appears at the start of
/// the scan order and is never updated in the bitstream." The dummy
/// is stored here as `0` — a value outside the legal `1..=255` node
/// probability range (§7) — and is never read by
/// [`decode_coeff_band_updates`], whose walk starts at coefficient 1.
pub const COEFF_BAND_UPDATE_FLAG_PROBS: [u8; 64] = [
    0, 132, 132, 159, 153, 151, 161, 170, //
    164, 162, 136, 110, 103, 114, 129, 118, //
    124, 125, 132, 136, 114, 110, 142, 135, //
    134, 123, 143, 126, 153, 183, 166, 161, //
    171, 180, 179, 164, 203, 218, 225, 217, //
    215, 206, 203, 217, 229, 241, 248, 243, //
    253, 255, 253, 255, 255, 255, 255, 255, //
    255, 255, 255, 255, 255, 255, 255, 255, //
];

/// Per-coefficient band assignment: `assignment[c]` is the Table 16
/// band (`0..=15`) the zig-zag AC coefficient `c` (`1..=63`) belongs
/// to. Entry `0` is a dummy for the DC coefficient, which §12.2 keeps
/// outside the band system ("In all scan orders the first DCT
/// coefficient is always the DC coefficient").
pub type BandAssignment = [u8; 64];

/// The default band assignment, derived from §12.1 + §12.2.
///
/// §12.2: "For intra-coded frames the scan order is first set to the
/// appropriate default." Under the §12.1 default scan order the
/// coefficient decoded at modified-scan position `p` *is* zig-zag
/// coefficient `p` (the default order is the identity permutation in
/// zig-zag space), so the default band of zig-zag coefficient `c` is
/// the Table 16 band whose position range contains `c`. Entry `0` is
/// the DC dummy (set to `0`, never consulted).
pub const DEFAULT_BAND_ASSIGNMENT: BandAssignment = {
    let mut assignment = [0u8; 64];
    let mut band = 0;
    while band < NUM_SCAN_BANDS {
        let (first, last) = CUSTOM_SCAN_BAND_RANGES[band];
        let mut pos = first;
        while pos <= last {
            assignment[pos as usize] = band as u8;
            pos += 1;
        }
        band += 1;
    }
    assignment
};

/// Decode the §12.2 Table 17 scan order update record.
///
/// Reads the `b(1)` `ScanOrderUpdateFlag` ("Indicates whether or not
/// a set of scan-order updates follow: (1) yes (0) no"):
///
/// * Flag `0` — no custom scan order for this frame. Per §12.2 ("the
///   scan order must be reset to the default") `assignment` is reset
///   to [`DEFAULT_BAND_ASSIGNMENT`]. Returns `Ok(false)`.
/// * Flag `1` — delegates to [`decode_coeff_band_updates`], applying
///   the 63 per-coefficient update sets to `assignment` in place.
///   Returns `Ok(true)`.
///
/// `flag_probs` is the per-coefficient `CoeffBandUpdateFlag` node
/// probability bank — [`COEFF_BAND_UPDATE_FLAG_PROBS`] for the
/// published §12.2 listing. (Parameterised like the
/// [`crate::prob_update`] drivers so the walk can be exercised under
/// chosen flag banks; under the operative `>> 8` Split (errata #35)
/// the BoolCoder is non-degenerate for every probability, so any bank
/// in `1..=255` is well-defined.)
///
/// Intra-vs-inter seeding is the caller's responsibility: §12.2 has
/// intra frames reset `assignment` to [`DEFAULT_BAND_ASSIGNMENT`]
/// *before* the deltas apply, while inter frames apply deltas "to the
/// custom scan order used in the previous frame".
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted
/// mid-walk.
pub fn decode_scan_order_update(
    bc: &mut BoolCoder<'_>,
    flag_probs: &[u8; 64],
    assignment: &mut BandAssignment,
) -> Result<bool, Error> {
    if bc.decode_b1()? == 0 {
        *assignment = DEFAULT_BAND_ASSIGNMENT;
        return Ok(false);
    }
    decode_coeff_band_updates(bc, flag_probs, assignment)?;
    Ok(true)
}

/// Decode the 63 per-coefficient update sets of §12.2 Table 17 (the
/// body that follows a set `ScanOrderUpdateFlag`).
///
/// For each AC coefficient `c` in `1..=63` (standard zig-zag order,
/// the order the §12.2 `CoeffBandUpdateFlagProbs` listing is indexed
/// in): read `CoeffBandUpdateFlag` as `B(flag_probs[c])` ("A flag
/// indicating whether or not a coefficient's band has been updated:
/// (1) yes (0) no"); when set, read `NewCoeffBand` as `b(4)` ("The
/// new band for the coefficient") and store it into `assignment[c]`.
/// Pass [`COEFF_BAND_UPDATE_FLAG_PROBS`] as `flag_probs` for the
/// published §12.2 bank.
///
/// The DC entry (`assignment[0]`, like `flag_probs[0]`) is never
/// touched: "The first entry in the table is a dummy entry for the DC
/// coefficient. This always appears at the start of the scan order
/// and is never updated in the bitstream."
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted
/// mid-walk.
pub fn decode_coeff_band_updates(
    bc: &mut BoolCoder<'_>,
    flag_probs: &[u8; 64],
    assignment: &mut BandAssignment,
) -> Result<(), Error> {
    for coeff in 1..=NUM_AC_POSITIONS {
        if bc.decode_bool(flag_probs[coeff])? == 1 {
            // b(4) yields 0..=15, exactly the Table 16 band space.
            assignment[coeff] = bc.decode_b(4)? as u8;
        }
    }
    Ok(())
}

/// Build the modified scan order from a band assignment (§12.2).
///
/// Returns `scan` such that `scan[p]` is the zig-zag coefficient
/// index (`0..=63`) decoded at modified-scan position `p`:
///
/// * Position 0 is always the DC coefficient (§12.2: "In all scan
///   orders the first DCT coefficient is always the DC coefficient").
/// * The AC positions `1..=63` are filled band by band in ascending
///   band order, and "within each band the coefficients are then
///   sorted into ascending order based upon the original zig-zag scan
///   order" — i.e. ascending zig-zag coefficient index.
///
/// §12.2's worked example pins the construction: "if AC7 and AC21 are
/// labeled as belonging to band 3, then AC7 will be assigned position
/// 11 and AC21 position 12 in the modified scan order" (exercised by
/// the `spec_worked_example_ac7_ac21_band3` test).
///
/// The output is always a permutation of `0..=63`: every coefficient
/// carries exactly one band, so it is emitted exactly once.
pub fn build_custom_scan_order(assignment: &BandAssignment) -> [u8; 64] {
    let mut scan = [0u8; 64];
    let mut pos = 1;
    for band in 0..NUM_SCAN_BANDS as u8 {
        for (coeff, &assigned) in assignment.iter().enumerate().skip(1) {
            if assigned == band {
                scan[pos] = coeff as u8;
                pos += 1;
            }
        }
    }
    debug_assert_eq!(pos, 64);
    scan
}

/// Compose a modified scan order with the §12.1 zig-zag-to-raster
/// table.
///
/// §12.2 identifies coefficients "by their position in the standard
/// zig-zag scan order", while §15 inverse quantization and §16
/// inverse DCT consume raster positions. Given the
/// [`build_custom_scan_order`] output, returns `raster` such that
/// `raster[p] = DEFAULT_SCAN_ORDER[scan[p]]` — the raster position of
/// the coefficient decoded at modified-scan position `p`, the direct
/// §12 "re-arranged back to raster order before inverse quantization
/// and IDCT" permutation.
pub fn custom_scan_order_to_raster(scan: &[u8; 64]) -> [u8; 64] {
    let mut raster = [0u8; 64];
    for (out, &zigzag) in raster.iter_mut().zip(scan.iter()) {
        *out = DEFAULT_SCAN_ORDER[zigzag as usize];
    }
    raster
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::DEFAULT_SCAN_ORDER;

    /// Table 16's sixteen ranges tile the AC position space `1..=63`
    /// contiguously: each band starts where the previous ended + 1,
    /// band 0 starts at position 1, band 15 ends at position 63.
    #[test]
    fn band_ranges_tile_ac_positions() {
        assert_eq!(CUSTOM_SCAN_BAND_RANGES[0].0, 1);
        assert_eq!(CUSTOM_SCAN_BAND_RANGES[NUM_SCAN_BANDS - 1].1, 63);
        for band in 1..NUM_SCAN_BANDS {
            assert_eq!(
                CUSTOM_SCAN_BAND_RANGES[band].0,
                CUSTOM_SCAN_BAND_RANGES[band - 1].1 + 1,
                "band {band} must start right after band {}",
                band - 1
            );
        }
        for &(first, last) in &CUSTOM_SCAN_BAND_RANGES {
            assert!(first <= last);
        }
        // 63 positions total.
        let total: u32 = CUSTOM_SCAN_BAND_RANGES
            .iter()
            .map(|&(first, last)| u32::from(last - first + 1))
            .sum();
        assert_eq!(total, NUM_AC_POSITIONS as u32);
    }

    /// Verbatim spot checks of the §12.2 `CoeffBandUpdateFlagProbs`
    /// listing (transcription guard), plus the DC dummy entry.
    #[test]
    fn coeff_band_update_flag_probs_verbatim() {
        // Spec prints `NA` for the DC dummy; stored as the
        // out-of-range poison value 0 and never read.
        assert_eq!(COEFF_BAND_UPDATE_FLAG_PROBS[0], 0);
        // Row 1 of the listing: NA, 132, 132, 159, 153, 151, 161, 170.
        assert_eq!(
            &COEFF_BAND_UPDATE_FLAG_PROBS[1..8],
            &[132, 132, 159, 153, 151, 161, 170]
        );
        // Row 2 starts: 164, 162, 136, 110, 103, ...
        assert_eq!(
            &COEFF_BAND_UPDATE_FLAG_PROBS[8..13],
            &[164, 162, 136, 110, 103]
        );
        // Row 5 of the listing: 171, 180, 179, 164, 203, 218, 225, 217.
        assert_eq!(
            &COEFF_BAND_UPDATE_FLAG_PROBS[32..40],
            &[171, 180, 179, 164, 203, 218, 225, 217]
        );
        // Tail of the table saturates at 255.
        assert!(COEFF_BAND_UPDATE_FLAG_PROBS[51..].iter().all(|&p| p == 255));
        // Every non-dummy entry is a legal §7 node probability.
        assert!(COEFF_BAND_UPDATE_FLAG_PROBS[1..]
            .iter()
            .all(|&p| (1..=255).contains(&p)));
    }

    /// The default band assignment places zig-zag coefficient `c`
    /// (`1..=63`) in the Table 16 band whose position range contains
    /// `c` (the §12.1 default order is the identity in zig-zag space).
    #[test]
    fn default_band_assignment_matches_table_16() {
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[0], 0); // DC dummy
        for (band, &(first, last)) in CUSTOM_SCAN_BAND_RANGES.iter().enumerate() {
            for pos in first..=last {
                assert_eq!(
                    DEFAULT_BAND_ASSIGNMENT[pos as usize], band as u8,
                    "coefficient {pos} must default to band {band}"
                );
            }
        }
        // Spot checks against Table 16 rows.
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[1], 0);
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[4], 1);
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[10], 2);
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[12], 3);
        assert_eq!(DEFAULT_BAND_ASSIGNMENT[63], 15);
        // Defaults are monotonically non-decreasing in zig-zag order.
        for c in 2..64 {
            assert!(DEFAULT_BAND_ASSIGNMENT[c] >= DEFAULT_BAND_ASSIGNMENT[c - 1]);
        }
    }

    /// The default band assignment rebuilds to the identity
    /// permutation in zig-zag space — i.e. the §12.1 default scan
    /// order itself.
    #[test]
    fn default_assignment_builds_identity_scan() {
        let scan = build_custom_scan_order(&DEFAULT_BAND_ASSIGNMENT);
        for (pos, &coeff) in scan.iter().enumerate() {
            assert_eq!(coeff as usize, pos);
        }
        // Composed with §12.1 it reproduces the default
        // zigzag-to-raster table exactly.
        assert_eq!(custom_scan_order_to_raster(&scan), DEFAULT_SCAN_ORDER);
    }

    /// §12.2's worked example: "if AC7 and AC21 are labeled as
    /// belonging to band 3, then AC7 will be assigned position 11 and
    /// AC21 position 12 in the modified scan order." (The example
    /// presumes bands 0..=2 retain their default ten coefficients, so
    /// the assignment here backfills AC11 into band 2 — replacing the
    /// departing AC7 — and moves the default band-3 members AC11/AC12
    /// out of the way.)
    #[test]
    fn spec_worked_example_ac7_ac21_band3() {
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        assignment[7] = 3; // AC7 → band 3 (default band 2)
        assignment[21] = 3; // AC21 → band 3 (default band 6)
        assignment[11] = 2; // keep bands 0..=2 at ten coefficients
        assignment[12] = 4; // move the other default band-3 member out
        let scan = build_custom_scan_order(&assignment);
        assert_eq!(scan[11], 7, "AC7 must take modified position 11");
        assert_eq!(scan[12], 21, "AC21 must take modified position 12");
    }

    /// Within a band, coefficients are sorted "into ascending order
    /// based upon the original zig-zag scan order".
    #[test]
    fn within_band_ascending_zigzag_order() {
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        // Throw three widely-spaced coefficients into band 0.
        assignment[63] = 0;
        assignment[2] = 0;
        assignment[30] = 0;
        let scan = build_custom_scan_order(&assignment);
        // Band 0 now holds {1 (default), 2, 30, 63}, ascending.
        assert_eq!(&scan[1..5], &[1, 2, 30, 63]);
    }

    /// Any band assignment rebuilds to a permutation of `0..=63` with
    /// the DC pinned at position 0.
    #[test]
    fn build_is_always_a_permutation() {
        let assignments: [BandAssignment; 3] = [
            [0; 64],                                              // everything in band 0
            [15; 64],                                             // everything in band 15
            core::array::from_fn(|c| (c % NUM_SCAN_BANDS) as u8), // striped
        ];
        for assignment in &assignments {
            let scan = build_custom_scan_order(assignment);
            assert_eq!(scan[0], 0, "DC always first (§12.2)");
            let mut seen = [false; 64];
            for &coeff in &scan {
                assert!(!seen[coeff as usize], "coefficient emitted twice");
                seen[coeff as usize] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    /// `custom_scan_order_to_raster` composes position-for-position
    /// with the §12.1 table.
    #[test]
    fn raster_composition_per_position() {
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        assignment[5] = 9;
        let scan = build_custom_scan_order(&assignment);
        let raster = custom_scan_order_to_raster(&scan);
        for pos in 0..64 {
            assert_eq!(raster[pos], DEFAULT_SCAN_ORDER[scan[pos] as usize]);
        }
        // Still a permutation of the raster positions.
        let mut seen = [false; 64];
        for &r in &raster {
            assert!(!seen[r as usize]);
            seen[r as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// An all-zero byte stream decodes `ScanOrderUpdateFlag = 0`,
    /// which must reset a previously-customised assignment to the
    /// default (§12.2: "the scan order must be reset to the default").
    #[test]
    fn flag_zero_resets_to_default() {
        let bytes = [0u8; 16];
        let mut bc = BoolCoder::new(&bytes).expect("init");
        let mut assignment = [7u8; 64]; // previous frame's custom state
        let updated =
            decode_scan_order_update(&mut bc, &COEFF_BAND_UPDATE_FLAG_PROBS, &mut assignment)
                .expect("decode");
        assert!(!updated);
        assert_eq!(assignment, DEFAULT_BAND_ASSIGNMENT);
    }

    /// An all-zero byte stream decodes every `CoeffBandUpdateFlag`
    /// to 0 under a moderate flag bank, so the 63-coefficient walk
    /// leaves the assignment untouched (§12.2 Table 17:
    /// `NewCoeffBand` is "present only if both ScanOrderUpdateFlag
    /// and CoeffBandUpdateFlag are 1").
    ///
    /// The moderate `[128; 64]` flag bank gives a deterministic,
    /// readable branch trace (same rationale as the
    /// [`crate::prob_update`] driver tests); under the operative
    /// `>> 8` Split (errata #35) the BoolCoder is non-degenerate for
    /// every probability, so the published
    /// [`COEFF_BAND_UPDATE_FLAG_PROBS`] bank — whose tail saturates at
    /// 255 — is equally well-defined and is exercised under realistic
    /// VP6 bitstreams once the per-frame driver round lands.
    #[test]
    fn all_zero_stream_walk_is_a_no_op() {
        let bytes = [0u8; 64];
        let mut bc = BoolCoder::new(&bytes).expect("init");
        let flag_probs = [128u8; 64];
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        assignment[20] = 11; // pre-existing custom entry must survive
        let before = assignment;
        decode_coeff_band_updates(&mut bc, &flag_probs, &mut assignment).expect("walk");
        assert_eq!(assignment, before);
    }

    /// A low flag probability forces the 1-branch on every
    /// `CoeffBandUpdateFlag` (operative `>> 8` Split, errata #35:
    /// `flag_prob = 1` gives `Split = 1`, so any stream whose running
    /// value keeps its top byte at `>= 1` decodes 63 consecutive set
    /// flags), exercising the `NewCoeffBand` `b(4)` path for all 63
    /// coefficients. Over an all-`0xFF` stream every flag fires and
    /// every `b(4)` reads a band in `0..=15`, so the post-walk
    /// assignment rewrites all 63 AC positions (the DC dummy at index
    /// 0 stays untouched) and is no longer the default.
    #[test]
    fn walk_applies_updates_under_forced_flags() {
        let bytes = [0xFFu8; 64];
        let mut bc = BoolCoder::new(&bytes).expect("init");
        let flag_probs = [1u8; 64];
        // Sentinel value outside the `0..=15` band space so we can
        // confirm every AC position was actually rewritten by a
        // `b(4)` read (which can only produce `0..=15`).
        let mut sentinel = DEFAULT_BAND_ASSIGNMENT;
        for s in sentinel.iter_mut().skip(1) {
            *s = 0xFF;
        }
        let mut assignment = sentinel;
        decode_coeff_band_updates(&mut bc, &flag_probs, &mut assignment).expect("walk");
        assert_eq!(assignment[0], sentinel[0], "DC dummy never written");
        for (c, &band) in assignment.iter().enumerate().skip(1) {
            assert!(
                band < NUM_SCAN_BANDS as u8,
                "coefficient {c} must be rewritten to a valid band (got {band})"
            );
        }
        assert_ne!(
            assignment, sentinel,
            "forced flags must rewrite at least one AC position"
        );
    }

    /// The walk only ever writes Table 16 band numbers: `NewCoeffBand`
    /// is a `b(4)` read, so every updated entry stays in `0..=15`.
    /// Swept across several moderate seed streams under the forced
    /// `[1; 64]` flag bank.
    #[test]
    fn walk_output_band_range_invariant() {
        for seed in [0x3Cu8, 0x77, 0xA1, 0xD0] {
            let bytes: Vec<u8> = (0..96)
                .map(|i| seed ^ (i as u8).wrapping_mul(0x35))
                .collect();
            let mut bc = BoolCoder::new(&bytes).expect("init");
            let flag_probs = [1u8; 64];
            let mut assignment = DEFAULT_BAND_ASSIGNMENT;
            decode_coeff_band_updates(&mut bc, &flag_probs, &mut assignment).expect("walk");
            assert_eq!(assignment[0], 0, "DC dummy never written");
            for (c, &band) in assignment.iter().enumerate().skip(1) {
                assert!(
                    band < NUM_SCAN_BANDS as u8,
                    "coefficient {c} got out-of-range band {band}"
                );
            }
        }
    }

    /// Determinism: the same bytes + flag bank walk to the same
    /// assignment across two independent BoolCoder runs.
    #[test]
    fn walk_is_deterministic() {
        let bytes = [
            0x80u8, 0x55, 0xAA, 0x33, 0xCC, 0x66, 0x99, 0x5A, 0xA5, 0x3C, 0xC3, 0x69,
        ]
        .repeat(8);
        let run = |input: &[u8]| {
            let mut bc = BoolCoder::new(input).expect("init");
            let flag_probs = [1u8; 64];
            let mut assignment = DEFAULT_BAND_ASSIGNMENT;
            decode_coeff_band_updates(&mut bc, &flag_probs, &mut assignment).expect("walk");
            assignment
        };
        assert_eq!(run(&bytes), run(&bytes));
    }

    /// Truncation surface: a 4-byte stream whose renormalization
    /// pulls exhaust mid-walk surfaces `Error::Truncated` cleanly.
    /// (`flag_prob = 64` gives `Split ~ 128`, so each decode halves
    /// `Range` and renormalizes — consuming the stream.)
    #[test]
    fn truncated_stream_errors() {
        let bytes = [0x80u8; 4];
        let mut bc = BoolCoder::new(&bytes).expect("init");
        let flag_probs = [64u8; 64];
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        let result = decode_coeff_band_updates(&mut bc, &flag_probs, &mut assignment);
        assert_eq!(result, Err(Error::Truncated));
    }
}

//! VP6 default zig-zag scan order (spec §12.1).
//!
//! The VP6 entropy stage codes the 64 coefficients of an 8×8 transformed
//! block in a *zig-zag* order chosen so non-zero coefficients tend to
//! group near the start of the stream. Before the §15 inverse
//! quantization and §16 inverse DCT can run, the decoder must
//! re-arrange the 64 coefficients back to raster order. §12.1 calls the
//! re-arrangement table `default_dequant_table[64]` and specifies its
//! exact contents.
//!
//! Verbatim from §12.1 (On2 Technologies, document version 1.02,
//! August 2006):
//!
//! > The default scan order is the standard zig-zag order shown in
//! > Figure 14.
//! >
//! > The decoder uses the following table to convert back to raster
//! > order before applying the inverse quantizer and IDCT.
//! >
//! > ```text
//! > default_dequant_table[64] =
//! > {
//! >       0,  1,  8, 16,  9,  2,  3, 10,
//! >      17, 24, 32, 25, 18, 11,  4,  5,
//! >      12, 19, 26, 33, 40, 48, 41, 34,
//! >      27, 20, 13,  6,  7, 14, 21, 28,
//! >      35, 42, 49, 56, 57, 50, 43, 36,
//! >      29, 22, 15, 23, 30, 37, 44, 51,
//! >      58, 59, 52, 45, 38, 31, 39, 46,
//! >      53, 60, 61, 54, 47, 55, 62, 63
//! > }
//! > ```
//!
//! Indexing convention. The §12.1 prose ("the 64 coefficients of an
//! 8x8 transformed block in raster order such that coefficients 0 and
//! 63 are the DC and highest order AC coefficients, respectively, then
//! the scan re-ordering is specified by a 64 element array which gives
//! the new ordering") fixes the meaning of `default_dequant_table[i]`:
//! at zig-zag position `i` (i.e. the `i`-th coefficient that arrives
//! out of the entropy decoder, with `i = 0` being the DC), the
//! corresponding raster position in the 8×8 block is
//! `default_dequant_table[i]`. The table is therefore the
//! `zigzag-to-raster` permutation. Its inverse, the
//! `raster-to-zigzag` permutation, is computed once at module
//! initialisation and exposed as
//! [`DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG`] for callers that need to
//! drive a forward DCT in zig-zag order.
//!
//! Spec invariants this module enforces:
//!
//! * `DEFAULT_SCAN_ORDER[0] == 0` — the DC coefficient is always
//!   coded first (§12.1: "In all scan orders the first DCT coefficient
//!   is always the DC coefficient"; the same invariant restricts to
//!   the default scan order specifically).
//! * `DEFAULT_SCAN_ORDER[63] == 63` — the highest-frequency raster
//!   position is coded last by the standard zig-zag (Figure 14).
//! * `DEFAULT_SCAN_ORDER` is a *permutation* of `0..64` — every raster
//!   position is hit exactly once.
//!
//! Read no BoolCoder bits. The scan table itself is constant; the
//! per-frame `ScanOrderUpdateFlag` (§12.2 custom scan order, Table 17)
//! and the band-by-band updates it gates are BoolCoder-coded (`b(1)` /
//! `B(x)` / `b(4)`), and so are deferred until the §7.3 DOCS-GAP is
//! resolved. The default order is the one used whenever the per-frame
//! flag is `0`, which is the entire intra-frame path until the flag
//! flips, and is the seed for inter-frame custom-scan delta updates.

/// The default zig-zag scan order from spec §12.1, Figure 14 /
/// `default_dequant_table[64]`.
///
/// `DEFAULT_SCAN_ORDER[i]` is the raster position (`0..=63`) of the
/// `i`-th coefficient in zig-zag order, where `i = 0` is the DC
/// coefficient and `i = 63` is the highest-frequency AC coefficient.
///
/// To convert a block of 64 entropy-decoded coefficients (in zig-zag
/// order) back to raster order before §15 inverse quantization and
/// §16 inverse DCT, the decoder writes the `i`-th decoded coefficient
/// to raster position `DEFAULT_SCAN_ORDER[i]`. See
/// [`zigzag_to_raster_block`].
pub const DEFAULT_SCAN_ORDER: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, //
    17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, //
    27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, //
    29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63, //
];

/// The raster-to-zigzag inverse of [`DEFAULT_SCAN_ORDER`], computed at
/// `const` time from the §12.1 default table by inverting the
/// permutation.
///
/// `DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[r]` is the zig-zag position
/// (`0..=63`) of the coefficient at raster position `r`. It is the
/// permutation a forward DCT / encoder uses to emit raster-order
/// transform coefficients in the §12.1 zig-zag order.
///
/// The inverse-permutation property is exercised by the
/// `default_scan_inverse_is_consistent` unit test.
pub const DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG: [u8; 64] = {
    let mut inv = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        inv[DEFAULT_SCAN_ORDER[i] as usize] = i as u8;
        i += 1;
    }
    inv
};

/// Reorder a block of 64 coefficients from zig-zag order to raster
/// order via the §12.1 default scan.
///
/// The §13 entropy stage emits coefficients in zig-zag order: the
/// first decoded coefficient is the DC, the next 63 are the AC
/// coefficients in §12.1 Figure 14 zig-zag order. Before §15 inverse
/// quantization (which is defined per raster position via the
/// `DcQuantizationTable[0]` / `AcQuantizationTable[1..64]` split) and
/// §16 inverse DCT (which expects raster-order input), this function
/// permutes them.
///
/// For each zig-zag index `i`, writes `zigzag[i]` to raster position
/// `DEFAULT_SCAN_ORDER[i]` in `raster`.
pub fn zigzag_to_raster_block(zigzag: &[i32; 64], raster: &mut [i32; 64]) {
    for i in 0..64 {
        raster[DEFAULT_SCAN_ORDER[i] as usize] = zigzag[i];
    }
}

/// Reorder a block of 64 coefficients from raster order to zig-zag
/// order via the §12.1 default scan.
///
/// The encoder-side counterpart of [`zigzag_to_raster_block`]. For
/// each raster index `r`, writes `raster[r]` to zig-zag position
/// `DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[r]` in `zigzag`.
pub fn raster_to_zigzag_block(raster: &[i32; 64], zigzag: &mut [i32; 64]) {
    for r in 0..64 {
        zigzag[DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[r] as usize] = raster[r];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scan_order_table_length_is_64() {
        assert_eq!(DEFAULT_SCAN_ORDER.len(), 64);
    }

    #[test]
    fn default_scan_order_first_is_dc() {
        // §12.1 / §12.2: "In all scan orders the first DCT coefficient
        // is always the DC coefficient." Raster position 0 == DC.
        assert_eq!(DEFAULT_SCAN_ORDER[0], 0);
    }

    #[test]
    fn default_scan_order_last_is_high_frequency() {
        // Figure 14's standard zig-zag ends at raster position 63
        // (the highest-frequency coefficient, bottom-right corner).
        assert_eq!(DEFAULT_SCAN_ORDER[63], 63);
    }

    #[test]
    fn default_scan_order_is_a_permutation_of_0_to_63() {
        let mut seen = [false; 64];
        for &raster_pos in &DEFAULT_SCAN_ORDER {
            let r = raster_pos as usize;
            assert!(r < 64, "raster pos {} out of range", r);
            assert!(!seen[r], "raster pos {} appears twice", r);
            seen[r] = true;
        }
        assert!(seen.iter().all(|b| *b), "permutation incomplete");
    }

    #[test]
    fn default_scan_order_spot_values_match_spec() {
        // Spot-check several entries directly from spec §12.1 listing
        // to guard against transcription typos.
        assert_eq!(DEFAULT_SCAN_ORDER[1], 1); // 2nd zig-zag is raster col 1 of row 0
        assert_eq!(DEFAULT_SCAN_ORDER[2], 8); // 3rd zig-zag is raster col 0 of row 1
        assert_eq!(DEFAULT_SCAN_ORDER[3], 16); // 4th zig-zag is raster col 0 of row 2
        assert_eq!(DEFAULT_SCAN_ORDER[4], 9);
        assert_eq!(DEFAULT_SCAN_ORDER[5], 2);
        // End of first 8-entry row in the spec listing.
        assert_eq!(DEFAULT_SCAN_ORDER[7], 10);
        // Second row endpoints.
        assert_eq!(DEFAULT_SCAN_ORDER[8], 17);
        assert_eq!(DEFAULT_SCAN_ORDER[15], 5);
        // Mid table.
        assert_eq!(DEFAULT_SCAN_ORDER[31], 28);
        assert_eq!(DEFAULT_SCAN_ORDER[32], 35);
        // Final row.
        assert_eq!(DEFAULT_SCAN_ORDER[62], 62);
    }

    #[test]
    fn default_scan_order_diagonal_traversal_is_zigzag() {
        // The first four anti-diagonals of an 8x8 in raster row*8 + col
        // form, for each diagonal d (d in 0..7), the raster positions
        // (row, col) with row + col == d. Zig-zag visits these
        // diagonals in alternating directions: diagonal 0 (raster
        // {0}), then diagonal 1 from top-right ({1, 8}), then
        // diagonal 2 from bottom-left ({16, 9, 2}), etc.
        //
        // We don't enumerate all 15 diagonals — that's effectively
        // copy-pasting the spec table — but we *do* check that the
        // first three traversal segments match what Figure 14
        // describes.
        //
        // Diagonal 0: just raster 0.
        assert_eq!(DEFAULT_SCAN_ORDER[0], 0);
        // Diagonal 1: raster 1, then raster 8 (top-right -> bottom-left
        // sweep, written in zig-zag's bottom-up direction).
        assert_eq!(DEFAULT_SCAN_ORDER[1], 1);
        assert_eq!(DEFAULT_SCAN_ORDER[2], 8);
        // Diagonal 2: raster 16, raster 9, raster 2 (the next sweep
        // runs the other way).
        assert_eq!(DEFAULT_SCAN_ORDER[3], 16);
        assert_eq!(DEFAULT_SCAN_ORDER[4], 9);
        assert_eq!(DEFAULT_SCAN_ORDER[5], 2);
    }

    #[test]
    fn default_scan_inverse_table_length_is_64() {
        assert_eq!(DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG.len(), 64);
    }

    #[test]
    fn default_scan_inverse_is_consistent() {
        // The two tables must be inverse permutations of each other:
        // inv[scan[i]] == i for all i, and scan[inv[r]] == r for all r.
        for (i, &raster_pos) in DEFAULT_SCAN_ORDER.iter().enumerate() {
            let r = raster_pos as usize;
            assert_eq!(
                DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[r] as usize, i,
                "inverse table mismatch at zig-zag pos {} -> raster {}",
                i, r
            );
        }
        for (r, &zz_pos) in DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG.iter().enumerate() {
            let i = zz_pos as usize;
            assert_eq!(
                DEFAULT_SCAN_ORDER[i] as usize, r,
                "forward table mismatch at raster pos {} -> zig-zag {}",
                r, i
            );
        }
    }

    #[test]
    fn default_scan_inverse_is_also_a_permutation() {
        let mut seen = [false; 64];
        for &zz in &DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG {
            let i = zz as usize;
            assert!(i < 64);
            assert!(!seen[i], "zig-zag pos {} appears twice", i);
            seen[i] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn zigzag_to_raster_block_identity_when_table_seeded() {
        // Seed a zig-zag-ordered block with each coefficient equal to
        // its zig-zag position, then permute. After permutation, the
        // raster-position-`r` slot must hold the zig-zag position `i`
        // such that DEFAULT_SCAN_ORDER[i] == r, i.e. the
        // raster-to-zigzag inverse.
        let mut zz = [0i32; 64];
        for (i, slot) in zz.iter_mut().enumerate() {
            *slot = i as i32;
        }
        let mut raster = [0i32; 64];
        zigzag_to_raster_block(&zz, &mut raster);
        for (r, (&got, &want)) in raster
            .iter()
            .zip(DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG.iter())
            .enumerate()
        {
            assert_eq!(got, want as i32, "raster[{}] mismatch", r);
        }
    }

    #[test]
    fn raster_to_zigzag_block_identity_when_table_seeded() {
        // Symmetric forward-DCT-side check.
        let mut raster = [0i32; 64];
        for (r, slot) in raster.iter_mut().enumerate() {
            *slot = r as i32;
        }
        let mut zz = [0i32; 64];
        raster_to_zigzag_block(&raster, &mut zz);
        for (i, (&got, &want)) in zz.iter().zip(DEFAULT_SCAN_ORDER.iter()).enumerate() {
            assert_eq!(got, want as i32, "zig-zag[{}] mismatch", i);
        }
    }

    #[test]
    fn zigzag_then_raster_round_trip_preserves_block() {
        // Seed a raster-order block with non-trivial values, permute
        // raster -> zig-zag -> raster, and assert the original is
        // recovered. This is the only legitimate sanity check for the
        // pair: the two operations are inverse permutations.
        let mut raster_in = [0i32; 64];
        for (r, slot) in raster_in.iter_mut().enumerate() {
            // Use a value spread (mix of positive, negative, zero) so
            // a misaligned permutation can't accidentally look right.
            *slot = (r as i32 * 7 - 200).wrapping_mul(if r % 3 == 0 { -1 } else { 1 });
        }
        let mut zz = [0i32; 64];
        let mut raster_out = [0i32; 64];
        raster_to_zigzag_block(&raster_in, &mut zz);
        zigzag_to_raster_block(&zz, &mut raster_out);
        assert_eq!(raster_in, raster_out);
    }

    #[test]
    fn raster_then_zigzag_round_trip_preserves_block() {
        let mut zz_in = [0i32; 64];
        for (i, slot) in zz_in.iter_mut().enumerate() {
            *slot = (i as i32 * 13 + 5).wrapping_mul(if i & 1 == 0 { 1 } else { -1 });
        }
        let mut raster = [0i32; 64];
        let mut zz_out = [0i32; 64];
        zigzag_to_raster_block(&zz_in, &mut raster);
        raster_to_zigzag_block(&raster, &mut zz_out);
        assert_eq!(zz_in, zz_out);
    }

    #[test]
    fn zigzag_to_raster_dc_only_block_lands_at_raster_zero() {
        // §12.1 invariant: DC (zig-zag index 0) -> raster index 0.
        // After permutation, only raster[0] is non-zero.
        let mut zz = [0i32; 64];
        zz[0] = 1234;
        let mut raster = [0i32; 64];
        zigzag_to_raster_block(&zz, &mut raster);
        assert_eq!(raster[0], 1234);
        for (r, &v) in raster.iter().enumerate().skip(1) {
            assert_eq!(v, 0, "raster[{}] should be 0", r);
        }
    }

    #[test]
    fn zigzag_to_raster_last_coefficient_lands_at_raster_63() {
        // Figure 14 endpoint: zig-zag index 63 -> raster index 63
        // (the highest-frequency 7,7 position).
        let mut zz = [0i32; 64];
        zz[63] = -42;
        let mut raster = [0i32; 64];
        zigzag_to_raster_block(&zz, &mut raster);
        assert_eq!(raster[63], -42);
        for (r, &v) in raster.iter().enumerate().take(63) {
            assert_eq!(v, 0, "raster[{}] should be 0", r);
        }
    }
}

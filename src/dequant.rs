//! VP6 inverse quantization — per-frame dequant context (spec §15).
//!
//! VP6 quantizes an 8x8 block's 64 DCT coefficients with two uniform
//! scalar quantizers: one for the DC coefficient (index 0) and one
//! shared by the 63 AC coefficients (indices 1..=63). Reversing the
//! quantizer is a single integer multiply per coefficient by a value
//! looked up from a 64-entry table, indexed by the frame-level
//! `DctQMask` (spec §9 Table 1, 6 bits, 0..=63).
//!
//! This layer is **independent of the BoolCoder** (spec §7.3): the
//! quantizer factor depends only on `DctQMask`, which is a raw-bit
//! field already parsed by [`crate::frame_header`]. It therefore makes
//! progress without touching the contested §7.3 `Split` formula (see
//! the crate-root `DOCS-GAP` block).
//!
//! All values in this file are transcribed verbatim from
//! `docs/video/vp6/vp6_format.pdf` §15 (On2 Technologies, document
//! version 1.02, August 2006). No external library code was consulted.
//!
//! ## Spec note: pseudocode loop bound vs prose
//!
//! The §15 dequantization pseudocode reads:
//!
//! ```text
//! CoeffData[0] *= DcQuantizationTable[DctQMask]
//! for(i=1;i<63;i++)
//!     CoeffData[i] *= AcQuantizationTable[DctQMask]
//! ```
//!
//! The literal loop bound `i < 63` covers AC indices 1..=62 only,
//! leaving coefficient 63 un-dequantized. That contradicts the §15
//! prose, which states the quantizer is reversed by "performing
//! integer multiplication on each of its **64** coefficients" with
//! "**1** for the DC coefficient, and **1** for all **63** of the AC
//! coefficients" — i.e. all 64 entries are dequantized (1 DC + 63 AC,
//! the AC set being indices 1..=63 inclusive). The prose is internally
//! consistent (64 = 1 + 63) and the pseudocode bound is a clear
//! off-by-one against it; an 8x8 block has 64 coefficients and a
//! decoder that left index 63 at its quantized magnitude would produce
//! a wrong reconstruction. [`dequantize_block`] follows the prose and
//! dequantizes all 64 coefficients (DC at 0, AC at 1..=63).

/// DC dequantization factors indexed by `DctQMask` (spec §15
/// `DcQuantizationTable[64]`).
///
/// `DctQMask == 63` is the most accurate (smallest factor) quantizer
/// level; `DctQMask == 0` the least accurate (largest factor).
pub const DC_QUANTIZATION_TABLE: [u16; 64] = [
    188, 188, 188, 188, 180, 172, 172, 172, //
    172, 172, 168, 164, 164, 160, 160, 160, //
    160, 140, 140, 140, 140, 132, 132, 132, //
    132, 128, 128, 128, 108, 108, 104, 104, //
    100, 100, 96, 96, 92, 92, 76, 76, //
    76, 76, 72, 72, 68, 64, 64, 64, //
    64, 64, 60, 44, 44, 44, 40, 40, //
    36, 32, 28, 20, 12, 12, 8, 8, //
];

/// AC dequantization factors indexed by `DctQMask` (spec §15
/// `ACQuantizationTable[64]`).
pub const AC_QUANTIZATION_TABLE: [u16; 64] = [
    376, 368, 360, 352, 344, 328, 312, 296, //
    280, 264, 248, 232, 216, 212, 208, 204, //
    200, 196, 192, 188, 184, 180, 176, 172, //
    168, 160, 156, 148, 144, 140, 136, 132, //
    128, 124, 120, 116, 112, 108, 104, 100, //
    96, 92, 88, 84, 80, 76, 72, 68, //
    64, 60, 56, 52, 48, 44, 40, 36, //
    32, 28, 24, 20, 16, 12, 8, 4, //
];

/// Per-frame inverse-quantization context.
///
/// Resolved once per frame from the frame header's `DctQMask`, then
/// applied to every block's coefficient array. Holds the two scalar
/// quantizer factors (DC and AC) for the frame's quantizer level so
/// the per-block dequantize step is a single lookup-free multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DequantContext {
    /// `DcQuantizationTable[DctQMask]` — the DC coefficient multiplier.
    pub dc_factor: u16,
    /// `ACQuantizationTable[DctQMask]` — the shared AC multiplier.
    pub ac_factor: u16,
    /// The `DctQMask` this context was resolved from (0..=63).
    pub dct_q_mask: u8,
}

impl DequantContext {
    /// Build the dequant context for a frame whose header carried the
    /// 6-bit `dct_q_mask` (spec §9 Table 1).
    ///
    /// `dct_q_mask` is masked to its low 6 bits so an out-of-range
    /// value can never index past the 64-entry tables; the parser
    /// already constrains it to `0..=63` via `R(6)`, so the mask is
    /// defence-in-depth rather than a semantic transform.
    pub fn new(dct_q_mask: u8) -> Self {
        let idx = (dct_q_mask & 0x3F) as usize;
        Self {
            dc_factor: DC_QUANTIZATION_TABLE[idx],
            ac_factor: AC_QUANTIZATION_TABLE[idx],
            dct_q_mask: dct_q_mask & 0x3F,
        }
    }

    /// Inverse-quantize one coefficient at zig-zag/raster position
    /// `index` (0 = DC, 1..=63 = AC).
    ///
    /// Returns `coeff * factor`, where `factor` is the DC multiplier
    /// for `index == 0` and the AC multiplier otherwise. The multiply
    /// is performed in `i32` — quantized coefficients have a range of
    /// twelve bits plus sign (spec §13: -2048..=2047) and the largest
    /// table factor is 376, so the product fits comfortably in `i32`.
    ///
    /// `index >= 64` is treated as an AC coefficient; callers operating
    /// on a single 8x8 block pass `0..=63`.
    pub fn dequantize_coeff(self, index: usize, coeff: i32) -> i32 {
        let factor = if index == 0 {
            self.dc_factor
        } else {
            self.ac_factor
        } as i32;
        coeff * factor
    }

    /// Inverse-quantize a full 8x8 block of 64 coefficients in place.
    ///
    /// `coeffs[0]` is multiplied by the DC factor; `coeffs[1..=63]` by
    /// the AC factor. See the module-level "Spec note" for why all 64
    /// entries are processed despite the §15 pseudocode's `i < 63`
    /// loop bound.
    pub fn dequantize_block(self, coeffs: &mut [i32; 64]) {
        coeffs[0] *= self.dc_factor as i32;
        for c in &mut coeffs[1..] {
            *c *= self.ac_factor as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_64_entries() {
        assert_eq!(DC_QUANTIZATION_TABLE.len(), 64);
        assert_eq!(AC_QUANTIZATION_TABLE.len(), 64);
    }

    /// Spec endpoints: `DctQMask == 0` is the coarsest quantizer
    /// (largest factors), `DctQMask == 63` the finest (smallest).
    #[test]
    fn table_endpoints_match_spec() {
        assert_eq!(DC_QUANTIZATION_TABLE[0], 188);
        assert_eq!(DC_QUANTIZATION_TABLE[63], 8);
        assert_eq!(AC_QUANTIZATION_TABLE[0], 376);
        assert_eq!(AC_QUANTIZATION_TABLE[63], 4);
    }

    /// Tables are (weakly) monotonically non-increasing — the spec
    /// describes "roughly linear changes in data rate" with index 63
    /// "the most accurate quantizer level". This guards against a
    /// transcription transposition that would break monotonicity.
    #[test]
    fn tables_are_non_increasing() {
        for w in DC_QUANTIZATION_TABLE.windows(2) {
            assert!(w[0] >= w[1], "DC table not non-increasing at {w:?}");
        }
        for w in AC_QUANTIZATION_TABLE.windows(2) {
            assert!(w[0] >= w[1], "AC table not non-increasing at {w:?}");
        }
    }

    #[test]
    fn context_resolves_factors() {
        let ctx = DequantContext::new(0);
        assert_eq!(ctx.dc_factor, 188);
        assert_eq!(ctx.ac_factor, 376);
        assert_eq!(ctx.dct_q_mask, 0);

        let ctx = DequantContext::new(63);
        assert_eq!(ctx.dc_factor, 8);
        assert_eq!(ctx.ac_factor, 4);
        assert_eq!(ctx.dct_q_mask, 63);

        // A mid-table value (index 25): DC 128, AC 160.
        let ctx = DequantContext::new(25);
        assert_eq!(ctx.dc_factor, 128);
        assert_eq!(ctx.ac_factor, 160);
    }

    #[test]
    fn out_of_range_mask_is_clamped_to_low_six_bits() {
        // 0xC0 = 0b1100_0000 -> low 6 bits == 0, same as DctQMask 0.
        let ctx = DequantContext::new(0xC0);
        assert_eq!(ctx.dct_q_mask, 0);
        assert_eq!(ctx.dc_factor, 188);
        // 0xFF -> low 6 bits == 63.
        let ctx = DequantContext::new(0xFF);
        assert_eq!(ctx.dct_q_mask, 63);
        assert_eq!(ctx.ac_factor, 4);
    }

    #[test]
    fn dequantize_coeff_picks_dc_for_index_zero() {
        let ctx = DequantContext::new(0); // DC 188, AC 376
        assert_eq!(ctx.dequantize_coeff(0, 3), 3 * 188);
        assert_eq!(ctx.dequantize_coeff(1, 3), 3 * 376);
        assert_eq!(ctx.dequantize_coeff(63, 3), 3 * 376);
    }

    #[test]
    fn dequantize_coeff_handles_negative_and_extremes() {
        let ctx = DequantContext::new(0); // DC 188, AC 376
        assert_eq!(ctx.dequantize_coeff(0, -2048), -2048 * 188);
        // Largest-magnitude AC product fits i32 comfortably.
        assert_eq!(ctx.dequantize_coeff(5, 2047), 2047 * 376);
        assert_eq!(ctx.dequantize_coeff(5, -2048), -2048 * 376);
    }

    /// All 64 coefficients are dequantized: index 0 by DC, indices
    /// 1..=63 by AC. Index 63 in particular MUST be touched (the §15
    /// pseudocode's `i < 63` loop bound would have left it untouched).
    #[test]
    fn dequantize_block_covers_all_64_including_index_63() {
        let ctx = DequantContext::new(0); // DC 188, AC 376
        let mut block = [1i32; 64];
        ctx.dequantize_block(&mut block);
        assert_eq!(block[0], 188, "DC coefficient");
        for (i, &v) in block.iter().enumerate().skip(1) {
            assert_eq!(v, 376, "AC coefficient at index {i}");
        }
        // The off-by-one guard: index 63 is dequantized, not left as 1.
        assert_eq!(block[63], 376);
    }

    #[test]
    fn dequantize_block_preserves_sign() {
        let ctx = DequantContext::new(63); // DC 8, AC 4
        let mut block = [0i32; 64];
        block[0] = -5;
        block[1] = 7;
        block[63] = -3;
        ctx.dequantize_block(&mut block);
        assert_eq!(block[0], -40); // -5 * 8
        assert_eq!(block[1], 28); // 7 * 4
        assert_eq!(block[63], -12); // -3 * 4
    }
}

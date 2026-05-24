//! VP6 frame reconstruction — inter-coded blocks (spec §17.2–§17.4).
//!
//! Frame reconstruction (§17) recombines the IDCT output (the
//! *prediction error* signal) with a *prediction* signal to rebuild the
//! reconstructed image. §17 enumerates four cases:
//!
//! 1. Intra coded blocks (no prediction signal). — landed in
//!    [`crate::reconstruct`] (§17.1).
//! 2. Inter blocks with a zero `(0,0)` motion vector (§17.2).
//! 3. Inter blocks with a full-pixel-aligned motion vector (§17.3).
//! 4. Inter blocks with a sub-pixel motion vector in `x` and/or `y`
//!    (§17.4).
//!
//! This module lands cases (2)–(4). The key observation from §17 is that
//! all three inter cases share **one** recombination formula:
//!
//! > ```text
//! > For each sample in the block
//! > {
//! >    OutputValue = PredictionValue + PredictionError
//! >
//! >    // Clip to the range 0-255
//! >    If ( OutputValue < 0 )
//! >       OutputValue = 0
//! >    Else If ( OutputValue > 255 )
//! >       OutputValue = 255
//! > }
//! > ```
//!
//! (verbatim §17.2/§17.3/§17.4). The three cases differ only in how the
//! 8x8 **prediction block** is *sourced*:
//!
//! * §17.2 zero vector — "each sample is predicted by the sample at the
//!   same position in either the previous frame reconstruction … or the
//!   golden frame reconstruction. No filtering is carried out." A
//!   straight copy of the co-located 8x8 region.
//! * §17.3 full-pixel-aligned vector — "combining the prediction error
//!   signal with sample values from a set of points … that are offset by
//!   the given x and y." A straight copy of the 8x8 region offset by the
//!   integer `(x, y)`.
//! * §17.4 fractional-pixel vector — the prediction block is the
//!   interpolated output of the §11.4 filters (see [`crate::interp`]).
//!
//! So this module provides:
//!
//! * [`reconstruct_inter_block`] — the shared §17.2/§17.3/§17.4
//!   recombination: add an already-sourced 8x8 prediction block to the
//!   IDCT residual and clip to `0..=255`. This single function serves
//!   all three inter cases because the spec's recombination pseudocode is
//!   identical across them.
//! * [`fetch_prediction_block`] — the §17.2/§17.3 integer-offset
//!   prediction fetch: copy an 8x8 region from a reference reconstruction
//!   buffer. Zero MV (§17.2) is the `dx == dy == 0` special case; a
//!   full-pixel MV (§17.3) is the general integer offset.
//! * [`whole_sample_aligned`] / [`MvShift`] / [`luma_frac`] /
//!   [`chroma_frac`] — the §11.4 motion-vector decomposition that splits
//!   an MV component into its whole-sample-aligned part (`MvX >> MvShift`)
//!   and its fractional phase (the low `MvShift` bits). `MvShift` is 2 for
//!   luma (¼-sample precision) and 3 for chroma (⅛-sample precision).
//!
//! Like the §15 dequant, §16 IDCT, §17.1 intra reconstruction and §11.4
//! interpolation layers, this stage reads **no BoolCoder bits**: given an
//! already-decoded motion vector, a reference buffer and the IDCT
//! residual, every operation here is pure integer pixel arithmetic. It
//! therefore advances the decoder without touching the contested §7.3
//! `Split` formula (see the crate-root `DOCS-GAP`). The motion vector
//! that drives the fetch/interpolation phase is decoded upstream, behind
//! the BoolCoder.
//!
//! All material in this file is transcribed verbatim from
//! `docs/video/vp6/vp6_format.pdf` §11.4 and §17 (On2 Technologies,
//! document version 1.02, August 2006). No external library code was
//! consulted.

use crate::reconstruct::{PIXEL_MAX, PIXEL_MIN};

/// Motion-vector fractional-precision shift (spec §11.4).
///
/// > `// Mvshift is 2 for luma blocks and 3 for chroma blocks`
///
/// VP6 motion vectors are expressed in ¼-pixel units for luma and
/// ⅛-pixel units for chroma. The whole-sample-aligned component of an MV
/// is `MvComponent >> MvShift`; the fractional phase is the low
/// `MvShift` bits (`MvComponent & ((1 << MvShift) - 1)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvShift {
    /// Luma blocks: ¼-sample precision, `MvShift == 2`.
    Luma,
    /// Chroma blocks: ⅛-sample precision, `MvShift == 3`.
    Chroma,
}

impl MvShift {
    /// The numeric shift amount: `2` for luma, `3` for chroma (§11.4).
    #[inline]
    pub const fn bits(self) -> u32 {
        match self {
            MvShift::Luma => 2,
            MvShift::Chroma => 3,
        }
    }

    /// The number of distinct fractional phases at this precision:
    /// `1 << bits` — `4` for luma, `8` for chroma. Matches the
    /// [`crate::interp::BILINEAR_LUMA_FILTERS`] (4 rows) and
    /// [`crate::interp::BILINEAR_CHROMA_FILTERS`] (8 rows) dimensions.
    #[inline]
    pub const fn phase_count(self) -> u32 {
        1 << self.bits()
    }

    /// The fractional-phase mask: `(1 << bits) - 1` — `0b11` for luma,
    /// `0b111` for chroma.
    #[inline]
    pub const fn frac_mask(self) -> i32 {
        (1 << self.bits()) - 1
    }
}

/// Whole-sample-aligned component of a motion-vector component (spec
/// §11.4):
///
/// > ```text
/// > WholeSampleAlignedX = (MvX >> MvShift)
/// > WholeSampleAlignedY = (MvY >> MvShift)
/// > ```
///
/// The shift is arithmetic so that negative motion vectors floor toward
/// negative infinity, keeping the whole/fractional split consistent
/// across the sign boundary: e.g. for luma (`MvShift = 2`) an MV
/// component of `-1` (¼-pixel units) decomposes to whole part `-1` and
/// fractional phase `3` (¾ of the way to the *next* whole sample), the
/// standard two's-complement floor-division identity the spec's `>>`
/// relies on.
#[inline]
pub fn whole_sample_aligned(mv_component: i32, shift: MvShift) -> i32 {
    mv_component >> shift.bits()
}

/// Fractional phase of a luma motion-vector component (spec §11.4): the
/// low two bits, an index `0..=3` into [`crate::interp::BILINEAR_LUMA_FILTERS`]
/// (and the per-alpha rows of [`crate::interp::BICUBIC_FILTER_SET`] which
/// also index by ⅛ but where luma uses the even phases).
///
/// `MvX & 0b11` — phase 0 is whole-sample aligned, phases 1/2/3 are
/// ¼/½/¾.
#[inline]
pub fn luma_frac(mv_component: i32) -> usize {
    (mv_component & MvShift::Luma.frac_mask()) as usize
}

/// Fractional phase of a chroma motion-vector component (spec §11.4): the
/// low three bits, an index `0..=7` into
/// [`crate::interp::BILINEAR_CHROMA_FILTERS`] (and the per-alpha rows of
/// [`crate::interp::BICUBIC_FILTER_SET`]).
///
/// `MvX & 0b111` — phase 0 is whole-sample aligned, phases 1..=7 are the
/// ⅛-sample offsets.
#[inline]
pub fn chroma_frac(mv_component: i32) -> usize {
    (mv_component & MvShift::Chroma.frac_mask()) as usize
}

/// Fetch the 8x8 prediction block for a zero (§17.2) or full-pixel
/// aligned (§17.3) motion vector — a straight copy of the reference
/// reconstruction buffer.
///
/// `ref_buf` is the reference reconstruction buffer (previous-frame or
/// golden-frame, with its §11.5 UMV border applied so the offset stays in
/// bounds). `base_pos` is the buffer index of the block's top-left sample
/// as if the MV were zero (the co-located position). `(dx, dy)` is the
/// *whole-sample* offset in samples: `(0, 0)` for §17.2's zero vector,
/// the integer `(WholeSampleAlignedX, WholeSampleAlignedY)` for §17.3's
/// full-pixel vector. `ref_stride` is the reference buffer line length.
/// The 64 prediction samples are written to `pred` in raster order.
///
/// §17.2: "each sample is predicted by the sample at the same position"
/// (the `dx == dy == 0` call). §17.3: "sample values from a set of points
/// … offset by the given x and y" (the general call). No filtering is
/// applied — that is §17.4's job (see [`crate::interp`]).
///
/// The source index for output sample `(r, c)` is
/// `base_pos + (dy + r) * ref_stride + (dx + c)`; the caller guarantees
/// the resulting index range is in bounds via the §11.5 border.
pub fn fetch_prediction_block(
    ref_buf: &[u8],
    base_pos: usize,
    ref_stride: usize,
    dx: i32,
    dy: i32,
    pred: &mut [u8; 64],
) {
    for r in 0..8i32 {
        // Signed row/col arithmetic, then re-base onto the unsigned
        // buffer index. The caller's UMV border keeps this non-negative
        // and in range.
        let src_row = (base_pos as i32) + (dy + r) * (ref_stride as i32);
        for c in 0..8i32 {
            let src = (src_row + dx + c) as usize;
            pred[(r * 8 + c) as usize] = ref_buf[src];
        }
    }
}

/// Recombine a prediction block with the IDCT residual to reconstruct an
/// 8x8 inter-coded block (spec §17.2 / §17.3 / §17.4).
///
/// `pred` is the 8x8 prediction signal in raster order — the output of
/// [`fetch_prediction_block`] (§17.2/§17.3) or of the §11.4 interpolation
/// filters ([`crate::interp::bilinear_block`] /
/// [`crate::interp::bicubic_block`]) for §17.4. `residual` is the 64
/// post-IDCT prediction-error values in raster order (the output of
/// [`crate::idct::idct_block`] on a dequantized inter block). On return
/// `pixels` holds the 64 reconstructed pixel values, each clipped to
/// `0..=255`.
///
/// Per the §17.2/§17.3/§17.4 pseudocode (identical across all three):
///
/// 1. `OutputValue = PredictionValue + PredictionError`
/// 2. Clip `OutputValue` to `0..=255` (inclusive at both ends).
/// 3. Cast to `u8`.
///
/// The addition is performed in signed `i32`: the residual is routinely
/// negative, and `pred + residual` can both underflow below 0 and
/// overflow above 255 — exactly the two cases §17's `If OutputValue < 0`
/// / `Else If OutputValue > 255` clamp handles.
///
/// Note: unlike the §17.1 intra path there is **no `+ 128` level shift**
/// here. The level shift only undoes the encoder's intra pre-DCT
/// subtraction; inter blocks code a difference against a prediction that
/// already carries the DC, so §17.2–§17.4's formula adds the prediction
/// directly with no constant offset.
pub fn reconstruct_inter_block(pred: &[u8; 64], residual: &[i32; 64], pixels: &mut [u8; 64]) {
    for ((out, &p), &e) in pixels.iter_mut().zip(pred.iter()).zip(residual.iter()) {
        // Step 1: OutputValue = PredictionValue + PredictionError.
        let v = (p as i32) + e;
        // Step 2: inclusive clip to 0..=255 (§17's `< 0` / `> 255`).
        let clipped = v.clamp(PIXEL_MIN, PIXEL_MAX);
        // Step 3: clip guarantees 0..=255, so the cast is lossless.
        *out = clipped as u8;
    }
}

/// In-place convenience: recombine and return a fresh 64-element pixel
/// array. Mirrors [`crate::reconstruct::intra_block_to_pixels`] for the
/// inter path.
pub fn inter_block_to_pixels(pred: &[u8; 64], residual: &[i32; 64]) -> [u8; 64] {
    let mut pixels = [0u8; 64];
    reconstruct_inter_block(pred, residual, &mut pixels);
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MvShift / decomposition -----------------------------------

    /// `MvShift::bits` is 2 for luma, 3 for chroma (§11.4 comment).
    #[test]
    fn mv_shift_bits_match_spec() {
        assert_eq!(MvShift::Luma.bits(), 2);
        assert_eq!(MvShift::Chroma.bits(), 3);
    }

    /// Phase counts mirror the §11.4.1 filter-table row counts:
    /// 4 for luma (`BilinearLumaFilters[4][2]`), 8 for chroma
    /// (`BilinearChromaFilters[8][2]`).
    #[test]
    fn phase_counts_match_filter_tables() {
        assert_eq!(MvShift::Luma.phase_count(), 4);
        assert_eq!(MvShift::Chroma.phase_count(), 8);
        assert_eq!(
            MvShift::Luma.phase_count() as usize,
            crate::interp::BILINEAR_LUMA_FILTERS.len()
        );
        assert_eq!(
            MvShift::Chroma.phase_count() as usize,
            crate::interp::BILINEAR_CHROMA_FILTERS.len()
        );
    }

    /// Fractional masks are `(1<<bits)-1`: `0b11` luma, `0b111` chroma.
    #[test]
    fn frac_masks_are_low_bits() {
        assert_eq!(MvShift::Luma.frac_mask(), 0b11);
        assert_eq!(MvShift::Chroma.frac_mask(), 0b111);
    }

    /// `whole_sample_aligned` is `MvX >> MvShift` (§11.4). A luma MV of
    /// 9 (¼-pixel units) is 2 whole samples + ¼ phase 1.
    #[test]
    fn whole_sample_split_positive_luma() {
        // 9 = 0b1001 -> whole 9>>2 = 2, frac 9&3 = 1.
        assert_eq!(whole_sample_aligned(9, MvShift::Luma), 2);
        assert_eq!(luma_frac(9), 1);
        // Reassemble: whole*4 + frac == original.
        assert_eq!(
            whole_sample_aligned(9, MvShift::Luma) * 4 + luma_frac(9) as i32,
            9
        );
    }

    /// A chroma MV of 13 (⅛-pixel units) is 1 whole sample + ⅝ phase 5.
    #[test]
    fn whole_sample_split_positive_chroma() {
        // 13 = 0b1101 -> whole 13>>3 = 1, frac 13&7 = 5.
        assert_eq!(whole_sample_aligned(13, MvShift::Chroma), 1);
        assert_eq!(chroma_frac(13), 5);
        assert_eq!(
            whole_sample_aligned(13, MvShift::Chroma) * 8 + chroma_frac(13) as i32,
            13
        );
    }

    /// Negative MVs floor toward -inf (arithmetic `>>`): luma -1 splits
    /// to whole -1 + frac 3 (¾ toward the next whole sample), and the
    /// two-piece identity `whole*4 + frac` reconstructs the original.
    #[test]
    fn whole_sample_split_negative_luma() {
        assert_eq!(whole_sample_aligned(-1, MvShift::Luma), -1);
        assert_eq!(luma_frac(-1), 3);
        assert_eq!(
            whole_sample_aligned(-1, MvShift::Luma) * 4 + luma_frac(-1) as i32,
            -1
        );
        // -8 (¼-pixel units) is exactly -2 whole samples, phase 0.
        assert_eq!(whole_sample_aligned(-8, MvShift::Luma), -2);
        assert_eq!(luma_frac(-8), 0);
    }

    /// Whole-pixel-aligned MVs have phase 0 (the filter identity).
    #[test]
    fn whole_pixel_mvs_have_zero_phase() {
        for whole in -3..=3 {
            assert_eq!(luma_frac(whole * 4), 0);
            assert_eq!(chroma_frac(whole * 8), 0);
        }
    }

    // ---- fetch_prediction_block (§17.2 / §17.3) --------------------

    /// §17.2 zero vector: `(dx, dy) = (0, 0)` copies the co-located 8x8
    /// region verbatim.
    #[test]
    fn fetch_zero_vector_copies_colocated() {
        let stride = 12usize;
        let mut buf = [0u8; 12 * 12];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let base = 2 * stride + 2;
        let mut pred = [0u8; 64];
        fetch_prediction_block(&buf, base, stride, 0, 0, &mut pred);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(pred[r * 8 + c], buf[base + r * stride + c]);
            }
        }
    }

    /// §17.3 full-pixel vector: a positive `(dx, dy)` integer offset
    /// shifts the copied region by exactly that many whole samples.
    #[test]
    fn fetch_full_pixel_positive_offset() {
        let stride = 16usize;
        let mut buf = [0u8; 16 * 16];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i * 3) % 251) as u8;
        }
        let base = 4 * stride + 4;
        let (dx, dy) = (2i32, 3i32);
        let mut pred = [0u8; 64];
        fetch_prediction_block(&buf, base, stride, dx, dy, &mut pred);
        for r in 0..8i32 {
            for c in 0..8i32 {
                let src = (base as i32 + (dy + r) * stride as i32 + dx + c) as usize;
                assert_eq!(pred[(r * 8 + c) as usize], buf[src]);
            }
        }
    }

    /// §17.3 full-pixel vector: a negative offset works too (the source
    /// region moves up/left), exercising the signed-arithmetic path.
    #[test]
    fn fetch_full_pixel_negative_offset() {
        let stride = 16usize;
        let mut buf = [0u8; 16 * 16];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i * 5) % 251) as u8;
        }
        let base = 5 * stride + 5;
        let (dx, dy) = (-2i32, -1i32);
        let mut pred = [0u8; 64];
        fetch_prediction_block(&buf, base, stride, dx, dy, &mut pred);
        for r in 0..8i32 {
            for c in 0..8i32 {
                let src = (base as i32 + (dy + r) * stride as i32 + dx + c) as usize;
                assert_eq!(pred[(r * 8 + c) as usize], buf[src]);
            }
        }
    }

    /// The whole-sample offset fed to `fetch_prediction_block` for a
    /// full-pixel MV is `whole_sample_aligned(mv)`. A luma MV of `(8, 12)`
    /// in ¼-pixel units (== whole pixel `(2, 3)`) fetches the region
    /// offset by `(2, 3)`.
    #[test]
    fn full_pixel_mv_drives_whole_sample_offset() {
        let stride = 20usize;
        let mut buf = [0u8; 20 * 20];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let base = 5 * stride + 5;
        let (mvx, mvy) = (8i32, 12i32); // ¼-pixel units, both whole-pixel.
                                        // Phase must be zero for this to be a §17.3 (not §17.4) case.
        assert_eq!(luma_frac(mvx), 0);
        assert_eq!(luma_frac(mvy), 0);
        let dx = whole_sample_aligned(mvx, MvShift::Luma);
        let dy = whole_sample_aligned(mvy, MvShift::Luma);
        assert_eq!((dx, dy), (2, 3));
        let mut pred = [0u8; 64];
        fetch_prediction_block(&buf, base, stride, dx, dy, &mut pred);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(pred[r * 8 + c], buf[base + (3 + r) * stride + (2 + c)]);
            }
        }
    }

    // ---- reconstruct_inter_block (§17.2 / §17.3 / §17.4) -----------

    /// Zero residual: the reconstruction equals the prediction exactly
    /// (the common case for a perfectly-predicted block — no residual
    /// coded). `OutputValue = PredictionValue + 0`.
    #[test]
    fn zero_residual_returns_prediction() {
        let mut pred = [0u8; 64];
        for (i, v) in pred.iter_mut().enumerate() {
            *v = (i * 4 % 256) as u8;
        }
        let residual = [0i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, pred);
    }

    /// A positive residual adds to the prediction without clipping when
    /// the sum stays in range: `100 + 20 = 120`.
    #[test]
    fn positive_residual_adds() {
        let pred = [100u8; 64];
        let residual = [20i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [120u8; 64]);
    }

    /// A negative residual subtracts from the prediction: `100 + (-30) =
    /// 70`.
    #[test]
    fn negative_residual_subtracts() {
        let pred = [100u8; 64];
        let residual = [-30i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [70u8; 64]);
    }

    /// Overflow above 255 clips to 255 (§17's `If OutputValue > 255`):
    /// `200 + 100 = 300 -> 255`.
    #[test]
    fn overflow_clips_to_255() {
        let pred = [200u8; 64];
        let residual = [100i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [255u8; 64]);
    }

    /// Underflow below 0 clips to 0 (§17's `If OutputValue < 0`):
    /// `50 + (-100) = -50 -> 0`.
    #[test]
    fn underflow_clips_to_zero() {
        let pred = [50u8; 64];
        let residual = [-100i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [0u8; 64]);
    }

    /// Boundary values: exactly 255 and exactly 0 are NOT clipped (the
    /// spec's clip is `< 0` / `> 255`, strict at the endpoints).
    #[test]
    fn exact_boundaries_pass_through() {
        let mut pred = [0u8; 64];
        let mut residual = [0i32; 64];
        pred[0] = 200;
        residual[0] = 55; // 255 exactly
        pred[1] = 10;
        residual[1] = -10; // 0 exactly
        pred[2] = 200;
        residual[2] = 56; // 256 -> 255
        pred[3] = 10;
        residual[3] = -11; // -1 -> 0
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels[0], 255);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 255);
        assert_eq!(pixels[3], 0);
    }

    /// Samples are independent — there's no inter-sample state in §17's
    /// recombination. A single perturbed sample must not bleed into its
    /// neighbours.
    #[test]
    fn samples_are_independent() {
        let pred = [128u8; 64];
        let mut residual = [0i32; 64];
        residual[10] = 40;
        residual[20] = -40;
        let pixels = inter_block_to_pixels(&pred, &residual);
        for (i, &p) in pixels.iter().enumerate() {
            let expected = match i {
                10 => 168u8,
                20 => 88u8,
                _ => 128u8,
            };
            assert_eq!(p, expected, "sample {i} interfered with a neighbour");
        }
    }

    /// The dual-buffer form and the wrapper agree.
    #[test]
    fn wrapper_matches_dual_buffer_form() {
        let mut pred = [0u8; 64];
        let mut residual = [0i32; 64];
        for i in 0..64 {
            pred[i] = (i * 3 % 256) as u8;
            residual[i] = (i as i32) - 30;
        }
        let by_wrapper = inter_block_to_pixels(&pred, &residual);
        let mut by_dual = [0u8; 64];
        reconstruct_inter_block(&pred, &residual, &mut by_dual);
        assert_eq!(by_wrapper, by_dual);
    }

    /// The inter recombination does NOT apply the §17.1 `+128` intra
    /// level shift. A zero prediction + zero residual reconstructs to 0,
    /// not 128 (the intra path would give 128). This pins the
    /// intra/inter distinction.
    #[test]
    fn no_intra_level_shift() {
        let pred = [0u8; 64];
        let residual = [0i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [0u8; 64], "inter path must not add the +128 shift");
    }

    // ---- end-to-end §17.2 / §17.3 integration ----------------------

    /// §17.2 end-to-end: fetch the zero-vector prediction, add a residual,
    /// reconstruct. Verifies the fetch + recombine chain composes.
    #[test]
    fn zero_vector_end_to_end() {
        let stride = 12usize;
        let mut buf = [0u8; 12 * 12];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i % 200) as u8;
        }
        let base = 2 * stride + 2;
        let mut pred = [0u8; 64];
        fetch_prediction_block(&buf, base, stride, 0, 0, &mut pred);
        let residual = [5i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        for r in 0..8 {
            for c in 0..8 {
                let expected = (buf[base + r * stride + c] as i32 + 5).clamp(0, 255) as u8;
                assert_eq!(pixels[r * 8 + c], expected);
            }
        }
    }

    /// §17.4 integration: feed a §11.4 interpolated prediction block
    /// (bilinear) into `reconstruct_inter_block`. Confirms the §11.4 →
    /// §17.4 hand-off type-checks and composes; for a flat source the
    /// interpolated prediction is flat, so the recombination is
    /// `flat + residual` clipped.
    #[test]
    fn fractional_vector_uses_interp_output() {
        use crate::interp::{bilinear_block, BILINEAR_LUMA_FILTERS};

        let stride = 16usize;
        let src = [120u8; 16 * 16];
        let mut pred = [0u8; 64];
        // ¼ phase in x, ½ phase in y — a genuine two-pass interpolation.
        bilinear_block(
            &src,
            stride + 1,
            stride,
            BILINEAR_LUMA_FILTERS[1],
            BILINEAR_LUMA_FILTERS[2],
            &mut pred,
        );
        // Flat source -> flat interpolated prediction at the source value.
        assert_eq!(pred, [120u8; 64]);
        let residual = [10i32; 64];
        let pixels = inter_block_to_pixels(&pred, &residual);
        assert_eq!(pixels, [130u8; 64]);
    }
}

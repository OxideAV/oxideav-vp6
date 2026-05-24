//! VP6 frame reconstruction — intra-coded blocks (spec §17.1).
//!
//! Frame reconstruction is the §17 stage that recombines the IDCT
//! output (the prediction error signal, or — for intra blocks — the
//! transformed image samples themselves) with the prediction signal
//! into the final image. §17 enumerates four cases:
//!
//! 1. Intra coded blocks (no prediction signal present).
//! 2. Inter blocks with a zero (0,0) motion vector.
//! 3. Inter blocks with a full-pixel-aligned motion vector.
//! 4. Inter blocks with a sub-pixel motion vector in `x` and/or `y`.
//!
//! This round lands case (1), **intra block reconstruction**, which is
//! the simplest of the four and the natural successor to the §16 IDCT.
//! The remaining three cases combine §17's clipping with motion
//! compensation against a reference reconstruction buffer (§17.2–§17.4)
//! and depend on mode/MV decoding upstream, which is gated on the
//! §7.3 BoolCoder `Split` defect (see the crate-root `DOCS-GAP`).
//!
//! Like the §15 dequant and §16 IDCT layers, this stage reads **no
//! BoolCoder bits** — it operates on a finished post-IDCT 8x8 block —
//! and therefore advances the decoder past round 3 without touching
//! the contested §7.3 `Split` formula.
//!
//! ## §17.1 in full
//!
//! > Intra coded frames and blocks are coded without any reference to
//! > any previous frame. However, prior to encoding the value 128 is
//! > subtracted from all data samples, so this needs to be added back
//! > in as part of the reconstruction process.
//! >
//! > Intra block reconstruction can be summarized by the following
//! > pseudo code:
//! >
//! > ```text
//! > For each sample in the block
//! > {
//! >    OutputValue = InputValue + 128
//! >
//! >    // Clip to the range 0-255
//! >    If ( OutputValue < 0 )
//! >       OutputValue = 0
//! >    Else If ( OutputValue > 255 )
//! >       OutputValue = 255
//! > }
//! > ```
//!
//! Two observations from the spec text:
//!
//! * The encoder-side level shift is `sample - 128` before the forward
//!   DCT; this stage undoes it with `+ 128` after the inverse DCT.
//!   §17.1's own clarification ("the value 128 is subtracted from all
//!   data samples").
//! * The clip is to the unsigned 8-bit pixel range `0..=255`. The
//!   pseudocode's `< 0` / `> 255` comparisons are inclusive at the
//!   endpoints — values that already land in range are passed through
//!   unchanged.
//!
//! The shift is performed in signed-`i32` arithmetic to avoid the
//! intermediate-overflow surprises that a `u8 + 128` would produce for
//! the negative inputs §17.1's clip explicitly contemplates (`If
//! OutputValue < 0`).
//!
//! All material in this file is transcribed verbatim from
//! `docs/video/vp6/vp6_format.pdf` §17.1 (On2 Technologies, document
//! version 1.02, August 2006). No external library code was consulted.

/// The intra-block level shift constant, from §17.1's `+ 128`.
///
/// Exposed as a `pub const` because the encoder side (`sample - 128`,
/// per §17.1's prose) will need exactly the same constant, and a
/// single source of truth keeps the two paths from drifting.
pub const INTRA_DC_LEVEL_SHIFT: i32 = 128;

/// The unsigned-8-bit pixel range, inclusive at both endpoints, that
/// §17.1's clip target.
pub const PIXEL_MIN: i32 = 0;
/// Upper inclusive bound for the §17.1 clip (255).
pub const PIXEL_MAX: i32 = 255;

/// Reconstruct one 8x8 intra-coded block per spec §17.1.
///
/// `block` holds the 64 post-IDCT sample values in raster order — i.e.
/// the output of [`crate::idct::idct_block`] applied to the
/// [`crate::dequant::DequantContext::dequantize_block`] output of an
/// intra block's coefficients. On return, `pixels` holds the 64
/// reconstructed pixel values in raster order, each in `0..=255`.
///
/// The transformation per sample is exactly §17.1's pseudocode:
///
/// 1. `OutputValue = InputValue + 128`
/// 2. Clamp `OutputValue` to `0..=255`.
/// 3. Cast to `u8`.
///
/// The level shift uses signed `i32` arithmetic; the IDCT output is
/// allowed to be negative (and routinely is for AC-dominant blocks),
/// and a `u8 + 128` would wrap rather than clip the negative tail.
///
/// This function does **not** clamp the IDCT output before the level
/// shift; the spec's clip is after the shift. A pre-shift value of
/// `-200`, for example, becomes `-72` then is clipped to `0` — the
/// same end result as clipping the post-shift value alone, but only
/// because the clip and shift are applied in §17.1's order.
pub fn reconstruct_intra_block(block: &[i32; 64], pixels: &mut [u8; 64]) {
    for (out, &v) in pixels.iter_mut().zip(block.iter()) {
        // Step 1: undo the encoder-side level shift.
        let shifted = v + INTRA_DC_LEVEL_SHIFT;
        // Step 2: clip to the 8-bit pixel range, inclusive at both
        // ends. `.clamp` is the inclusive-clip primitive; §17.1's
        // `If x < 0 { 0 } else if x > 255 { 255 } else { x }`
        // is exactly its semantics.
        let clipped = shifted.clamp(PIXEL_MIN, PIXEL_MAX);
        // Step 3: clip guarantees `0..=255`, so the cast is lossless.
        *out = clipped as u8;
    }
}

/// In-place variant of [`reconstruct_intra_block`] returning a fresh
/// 64-element pixel array.
///
/// Convenience for callers that don't need to retain the pre-shift
/// IDCT values; for the common decode path the dual-buffer
/// [`reconstruct_intra_block`] is cheaper because the IDCT block can
/// be re-used in place for the next block while pixels are emitted to
/// the destination plane.
pub fn intra_block_to_pixels(block: &[i32; 64]) -> [u8; 64] {
    let mut pixels = [0u8; 64];
    reconstruct_intra_block(block, &mut pixels);
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The level-shift constant is `128` as the spec defines.
    #[test]
    fn level_shift_constant_is_128() {
        assert_eq!(INTRA_DC_LEVEL_SHIFT, 128);
    }

    /// The clip range is `0..=255` inclusive at both ends.
    #[test]
    fn pixel_range_is_unsigned_8_bit() {
        assert_eq!(PIXEL_MIN, 0);
        assert_eq!(PIXEL_MAX, 255);
    }

    /// An all-zero post-IDCT block becomes an all-`128` mid-grey
    /// image: `0 + 128 = 128`, which is already in `0..=255`, so the
    /// clip is a no-op. This is the post-IDCT result for an intra
    /// block whose every dequantized coefficient was zero — i.e. a
    /// flat mid-grey block, which is exactly what such a block should
    /// reconstruct to.
    #[test]
    fn all_zero_block_reconstructs_to_mid_grey() {
        let block = [0i32; 64];
        let pixels = intra_block_to_pixels(&block);
        assert_eq!(pixels, [128u8; 64]);
    }

    /// In-range positive values pass through the shift with no
    /// clipping: `12 + 128 = 140`, which lies in `0..=255`.
    #[test]
    fn positive_in_range_input_is_just_shifted() {
        let mut block = [0i32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = i as i32; // 0..=63, all `+ 128` stays in range.
        }
        let pixels = intra_block_to_pixels(&block);
        for (i, &p) in pixels.iter().enumerate() {
            assert_eq!(p, (i as u8) + 128);
        }
    }

    /// In-range negative values pass through the shift with no
    /// clipping: `-50 + 128 = 78`, which lies in `0..=255`.
    #[test]
    fn negative_in_range_input_is_just_shifted() {
        let block = [-50i32; 64];
        let pixels = intra_block_to_pixels(&block);
        assert_eq!(pixels, [78u8; 64]);
    }

    /// Inputs below `-128` clip to `0` after the shift: e.g.
    /// `-200 + 128 = -72`, which §17.1's `If OutputValue < 0` branch
    /// pins to `0`.
    #[test]
    fn far_negative_input_clips_to_zero() {
        let block = [-200i32; 64];
        let pixels = intra_block_to_pixels(&block);
        assert_eq!(pixels, [0u8; 64]);
    }

    /// Inputs above `127` clip to `255` after the shift: e.g.
    /// `300 + 128 = 428`, which §17.1's `If OutputValue > 255` branch
    /// pins to `255`.
    #[test]
    fn far_positive_input_clips_to_255() {
        let block = [300i32; 64];
        let pixels = intra_block_to_pixels(&block);
        assert_eq!(pixels, [255u8; 64]);
    }

    /// Boundary check: a sample of exactly `-128` produces `0` (the
    /// inclusive lower bound — §17.1's clip is `< 0`, so `-128 + 128
    /// = 0` is NOT clipped; it just happens to coincide with the
    /// lower bound). And a sample of exactly `127` produces `255` (no
    /// clip; `127 + 128 = 255`).
    #[test]
    fn exact_boundary_values_pass_through() {
        let mut block = [0i32; 64];
        block[0] = -128;
        block[1] = 127;
        block[2] = -129; // clips to 0
        block[3] = 128; // clips to 255
        let pixels = intra_block_to_pixels(&block);
        assert_eq!(pixels[0], 0, "-128 + 128 = 0 (no clip)");
        assert_eq!(pixels[1], 255, "127 + 128 = 255 (no clip)");
        assert_eq!(pixels[2], 0, "-129 + 128 = -1 clipped to 0");
        assert_eq!(pixels[3], 255, "128 + 128 = 256 clipped to 255");
    }

    /// Every sample is processed independently — there's no
    /// inter-sample state in §17.1. Verifying this catches an
    /// accidental Σ-style implementation that would mix neighbours.
    #[test]
    fn samples_are_independent() {
        let mut block = [0i32; 64];
        block[10] = 50;
        block[20] = -50;
        let pixels = intra_block_to_pixels(&block);
        for (i, &p) in pixels.iter().enumerate() {
            let expected = match i {
                10 => 178u8, // 50 + 128
                20 => 78u8,  // -50 + 128
                _ => 128u8,
            };
            assert_eq!(p, expected, "sample {i} interfered with neighbours");
        }
    }

    /// The dual-buffer form and the convenience wrapper return the
    /// same bytes. Sanity check that the wrapper isn't reordering.
    #[test]
    fn wrapper_matches_dual_buffer_form() {
        let mut block = [0i32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i as i32) * 7) - 200; // mix of negatives + positives
        }
        let by_wrapper = intra_block_to_pixels(&block);
        let mut by_dual = [0u8; 64];
        reconstruct_intra_block(&block, &mut by_dual);
        assert_eq!(by_wrapper, by_dual);
    }

    /// Stress integration: feed the §17.1 stage what an actual
    /// upstream pipeline produces — a flat IDCT output from a
    /// DC-only post-dequant block. For a DC value `d`, the §16 IDCT
    /// produces a flat block of `((xC4S4 * ((xC4S4 * d) >> 16)) >> 16)
    /// >> 4`. Whatever §16 yields, §17.1 shifts-and-clips it; we
    /// just check the result is uniform (since the §16 output is
    /// uniform) and lands somewhere in `0..=255`.
    #[test]
    fn integrates_with_idct_dc_only_output() {
        use crate::idct::idct_block;

        let mut block = [0i32; 64];
        block[0] = 200; // pre-IDCT DC coefficient
        idct_block(&mut block);
        // Sanity: §16 IDCT of a DC-only block is uniform.
        let v0 = block[0];
        for &b in &block {
            assert_eq!(b, v0, "IDCT of DC-only block should be flat");
        }

        let pixels = intra_block_to_pixels(&block);
        let p0 = pixels[0];
        for &p in &pixels {
            assert_eq!(p, p0, "§17.1 output should remain uniform");
        }
    }
}

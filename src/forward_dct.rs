//! VP6 forward DCT transform — the encoder-side dual of the §16 inverse
//! DCT ([`crate::idct::idct_block`]).
//!
//! The §16 inverse transform is a fixed-point integer separable IDCT.
//! Its normalisation is exact and observable: a pure-DC raster block of
//! value `d` (only `block[0]` non-zero) inverse-transforms to a flat
//! block whose every sample is `d / 32` (the two `>> 16` cosine
//! descales contribute the `1/√2 · 1/√2 = 1/2` of `xC4S4²/2³²`, and the
//! column pass's trailing `>> 4` contributes the `1/16` — together
//! `1/32`). More generally the §16 transform realises a scaled
//! orthonormal 8-point DCT-III (the inverse of DCT-II) per axis with an
//! overall `1/32` gain spread across the two passes.
//!
//! The encoder must therefore produce coefficients such that, after §15
//! dequantisation and §16 inversion, the original (level-shifted) pixel
//! block is recovered to within the quantiser's rounding floor. This
//! module computes the matching **forward** transform: a separable
//! orthonormal 8-point DCT-II per axis with the inverse gain (`×32`
//! overall, i.e. the reciprocal of the IDCT's `1/32`), evaluated in
//! `f64` and rounded to the nearest integer.
//!
//! Because the §16 inverse is a *fixed-point* transform while this
//! forward transform is computed in floating point, the composition
//! `idct(quantise(fdct(x)))` is **not** bit-exact — it is accurate to
//! within a fraction of a quantiser step, which is exactly the regime an
//! encoder targets (a PSNR floor, not a bit-exact identity). The forward
//! transform is mathematically the analytic inverse of the orthonormal
//! transform the §16 integer IDCT approximates, so the round-trip error
//! is bounded by the §16 fixed-point truncation plus the §15 quantiser
//! step, not by any transform-shape mismatch.
//!
//! All structure here is derived solely from the observable §16
//! normalisation in `docs/video/vp6/vp6_format.pdf` (the `d / 32`
//! pure-DC gain and the separable two-pass shape). No external library
//! code was consulted.

use core::f64::consts::PI;

/// The overall forward gain that inverts the §16 IDCT's `1/32` pure-DC
/// gain. Split as `√32` per axis would be irrational; instead the
/// per-axis orthonormal DCT-II already carries its own `√(2/8)` scaling
/// and the residual factor needed to match the §16 normalisation is
/// applied once at the end as a single multiply (see [`fdct_block`]).
const IDCT_DC_GAIN: f64 = 32.0;

/// Per-axis orthonormal DCT-II scale factor `α(u)`: `√(1/8)` for `u = 0`
/// and `√(2/8)` otherwise. Squaring this across the two separable axes
/// yields the standard `1/4 · c(u) · c(v)` 2D DCT weighting.
#[inline]
fn alpha(u: usize) -> f64 {
    if u == 0 {
        (1.0_f64 / 8.0).sqrt()
    } else {
        (2.0_f64 / 8.0).sqrt()
    }
}

/// Forward-DCT one 8x8 block of (level-shifted) pixel-domain samples
/// into 64 raster-order DCT coefficients.
///
/// `input[row * 8 + col]` is the level-shifted spatial sample (the
/// encoder subtracts the §17.1 `128` level shift before calling this, so
/// inputs are roughly `-128..=127`). The output is written in **raster
/// order** (`out[row * 8 + col]`, `out[0]` the DC) — the same ordering
/// [`crate::idct::idct_block`] consumes, so the encoder maps to
/// scan order (§12) only when packing tokens.
///
/// The transform is the separable orthonormal DCT-II
///
/// ```text
/// F(u,v) = α(u) α(v) Σ_x Σ_y  f(x,y)
///          cos((2x+1)uπ/16) cos((2y+1)vπ/16)
/// ```
///
/// scaled by [`IDCT_DC_GAIN`] so that the §16 inverse's `1/32` pure-DC
/// gain is exactly undone (a flat input of value `s` yields `out[0] = s
/// · 32 · α(0)² · 8 · 8`-normalised DC that, after `idct`, returns `s`).
/// Each coefficient is rounded to the nearest integer (ties away from
/// zero) so the quantiser sees an unbiased estimate of the analytic
/// coefficient.
pub fn fdct_block(input: &[i32; 64], out: &mut [i32; 64]) {
    // Precompute the cosine basis: cos_basis[k][n] = cos((2n+1)kπ/16).
    // Small enough (64 entries) to compute inline; a const table would
    // need float-const support, so we build it per call. The block
    // transform itself dominates the cost.
    let mut cos = [[0.0_f64; 8]; 8];
    for (k, row) in cos.iter_mut().enumerate() {
        for (n, c) in row.iter_mut().enumerate() {
            *c = ((2 * n + 1) as f64 * k as f64 * PI / 16.0).cos();
        }
    }

    // Separable pass 1: transform each row (horizontal DCT) into a
    // temporary float buffer.
    let mut tmp = [[0.0_f64; 8]; 8];
    for r in 0..8 {
        for u in 0..8 {
            let mut acc = 0.0_f64;
            for x in 0..8 {
                acc += input[r * 8 + x] as f64 * cos[u][x];
            }
            tmp[r][u] = acc * alpha(u);
        }
    }

    // Separable pass 2: transform each column (vertical DCT) of the
    // row-transformed buffer, apply the §16 inverse-gain calibration,
    // and round to nearest.
    //
    // The `IDCT_DC_GAIN` factor undoes the §16 IDCT's overall `1/32`
    // gain. Per axis the orthonormal `α` already supplies `√(1/8)` /
    // `√(2/8)`; the product of the two axes' `α` for the DC term is
    // `1/8`, and the spatial sum of a flat block is `64 s`, giving a raw
    // DC of `8 s` before the gain — the `× 32` then yields the
    // coefficient whose §16 inverse is `s` (since `8 s · 32 · (xC4S4 /
    // 2^16)² / 16 = s` to the fixed-point floor).
    for v in 0..8 {
        for c in 0..8 {
            let mut acc = 0.0_f64;
            for y in 0..8 {
                acc += tmp[y][c] * cos[v][y];
            }
            let coeff = acc * alpha(v) * (IDCT_DC_GAIN / 8.0);
            out[v * 8 + c] = round_half_away(coeff);
        }
    }
}

/// Round to nearest integer, ties away from zero. Matches the unbiased
/// rounding a quantiser expects from a forward transform.
#[inline]
fn round_half_away(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5).floor() as i32
    } else {
        (x - 0.5).ceil() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::idct_block;

    /// A flat (constant) input block transforms to a pure-DC coefficient
    /// block: only `out[0]` is non-zero, all AC coefficients are zero.
    #[test]
    fn flat_input_is_pure_dc() {
        let input = [40i32; 64];
        let mut out = [0i32; 64];
        fdct_block(&input, &mut out);
        assert_ne!(out[0], 0, "flat block must have a non-zero DC");
        for (i, &c) in out.iter().enumerate().skip(1) {
            assert_eq!(c, 0, "AC coefficient {i} of a flat block must be 0");
        }
    }

    /// The forward transform inverts the §16 IDCT's pure-DC `1/32` gain:
    /// a flat input of value `s` produces a DC coefficient whose §16
    /// inverse returns the flat block `s` (to the fixed-point floor).
    #[test]
    fn dc_gain_matches_idct() {
        for s in [-100i32, -32, -1, 0, 1, 16, 50, 100] {
            let input = [s; 64];
            let mut coeffs = [0i32; 64];
            fdct_block(&input, &mut coeffs);
            // Inverse-transform the (un-quantised) coefficients.
            let mut recon = coeffs;
            idct_block(&mut recon);
            // Every reconstructed sample should be within 1 of `s`
            // (fixed-point truncation floor).
            for &r in recon.iter() {
                assert!(
                    (r - s).abs() <= 1,
                    "flat round-trip s={s} recon={r} exceeds floor"
                );
            }
        }
    }

    /// Full round-trip on an arbitrary (non-flat) block: fdct then idct
    /// recovers the input to within a small fixed-point floor on every
    /// sample. This is the un-quantised transform-pair accuracy; the
    /// quantiser adds its own (larger) step on top in the real encoder.
    #[test]
    fn unquantised_round_trip_is_accurate() {
        // A deterministic gradient + ripple pattern spanning the range.
        let mut input = [0i32; 64];
        for r in 0..8 {
            for c in 0..8 {
                let v = (r as i32 - 4) * 20 + (c as i32 - 4) * 10;
                input[r * 8 + c] = v;
            }
        }
        let mut coeffs = [0i32; 64];
        fdct_block(&input, &mut coeffs);
        let mut recon = coeffs;
        idct_block(&mut recon);
        let mut max_err = 0i32;
        for i in 0..64 {
            max_err = max_err.max((recon[i] - input[i]).abs());
        }
        // The integer IDCT + nearest-rounded float FDCT compose to a
        // small per-sample error; pin a conservative bound.
        assert!(max_err <= 3, "max round-trip error {max_err} too large");
    }

    /// The transform is separable and orthogonal enough that a single
    /// off-DC basis input produces predominantly one coefficient. A
    /// horizontal cosine of frequency 1 should put most of its energy in
    /// `out[1]`.
    #[test]
    fn horizontal_basis_concentrates_energy() {
        let mut input = [0i32; 64];
        for r in 0..8 {
            for c in 0..8 {
                // cos((2c+1)·1·π/16) scaled — the u=1 horizontal basis.
                let basis = ((2 * c + 1) as f64 * PI / 16.0).cos();
                input[r * 8 + c] = (basis * 60.0).round() as i32;
            }
        }
        let mut out = [0i32; 64];
        fdct_block(&input, &mut out);
        // out[1] (u=1, v=0) should dominate the AC spectrum.
        let target = out[1].abs();
        for (i, &c) in out.iter().enumerate() {
            if i != 0 && i != 1 {
                assert!(
                    c.abs() <= target,
                    "coefficient {i}={c} exceeds the expected dominant out[1]={target}"
                );
            }
        }
    }
}

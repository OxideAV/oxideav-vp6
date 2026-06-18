//! VP6 inverse DCT transform (spec §16).
//!
//! VP6 reconstructs an 8x8 block's pixel (or pixel-difference) values
//! from its 64 dequantized coefficients with a separable, fixed-point
//! integer inverse discrete cosine transform of 14-bit precision. The
//! transform is applied in two passes — a horizontal (row) pass over
//! the eight rows followed by a vertical (column) pass over the eight
//! columns — exactly as the §16 pseudocode prescribes.
//!
//! This layer is **independent of the BoolCoder** (spec §7.3): it
//! consumes a block of already-dequantized coefficients (the output of
//! [`crate::dequant`], spec §15) that are in **raster order**, and
//! produces a block of pixel/pixel-difference values. It never calls
//! `VP6_DecodeBool`, so — like the dequant layer — it advances the
//! decoder without touching the contested §7.3 `Split` formula (see the
//! crate-root `DOCS-GAP` block).
//!
//! All constants and the butterfly structure in this file are
//! transcribed verbatim from `docs/video/vp6/vp6_format.pdf` §16 (On2
//! Technologies, document version 1.02, August 2006). No external
//! library code was consulted.
//!
//! ## Input ordering (spec §16, opening paragraph)
//!
//! > Inversing the DCT requires that the coefficients have been placed
//! > back in raster order (not zig-zag or custom scan order as
//! > described in Section 12).
//!
//! [`idct_block`] therefore assumes its input is already in raster
//! order. Un-scanning (reversing the §12 zig-zag / custom scan) is the
//! caller's responsibility and happens before this stage; it is not
//! performed here.
//!
//! ## Fixed-point structure (spec §16 pseudocode)
//!
//! The transform uses seven fixed-point cosine constants (`xC1S7` …
//! `xC7S1`), each a Q16 representation of `cos(k·π/16)`. Every product
//! `(C * coeff) >> 16` performs the multiply-then-descale that keeps
//! the intermediate values bounded. The multiply itself is evaluated
//! in 64-bit: for a conformant §15-dequantized coefficient (magnitude
//! up to ≈ 2114 × 376) the product `C * coeff` exceeds `i32::MAX`, so
//! the pre-`>> 16` product must be 64-bit even though the descaled
//! result fits `i32`. The first (row) pass writes its
//! results without a final shift; the second (column) pass applies a
//! `>> 4` descale to each output, which is the transform's overall
//! normalisation. This matches a standard separable IDCT: no descale
//! between the passes, a single `>> 4` after the second.
//!
//! ## Spec note: a column-pass OCR artefact
//!
//! In the §16 column-pass pseudocode the `_Bd` line is rendered as
//!
//! ```text
//! _Bd = ((xC4S4 * (_B - _D)>>16)
//! ```
//!
//! with an unbalanced parenthesis (one `(` too few to close before the
//! `>>16`). The row pass gives the same quantity correctly as
//!
//! ```text
//! _Bd = ((xC4S4 * (_B - _D))>>16)
//! ```
//!
//! i.e. the shift applies to the whole product. The two passes are
//! structurally identical butterflies — the column pass differs only in
//! its `*8` strides and the trailing `>> 4` — so the missing paren is a
//! transcription artefact in the document, not a semantic difference.
//! [`idct_block`] uses the balanced form `((xC4S4 * (_B - _D)) >> 16)`
//! in both passes.

/// Q16 cosine constants (spec §16): `xCkS(8-k) ≈ round(cos(k·π/16) ·
/// 2^16)`.
const XC1S7: i32 = 64277;
const XC2S6: i32 = 60547;
const XC3S5: i32 = 54491;
const XC4S4: i32 = 46341;
const XC5S3: i32 = 36410;
const XC6S2: i32 = 25080;
const XC7S1: i32 = 12785;

/// One eight-point IDCT butterfly (spec §16 pseudocode, shared body of
/// both passes).
///
/// `ip` are the eight inputs `ip[0..=7]`; the returned array is the
/// eight outputs `op[0..=7]` **before** any final descale. The row pass
/// uses these outputs directly; the column pass shifts each by `>> 4`.
#[inline]
fn butterfly(ip: [i32; 8]) -> [i32; 8] {
    // Each `(C * coeff) >> 16` multiply is evaluated in `i64`. The
    // cosine constants reach 64277 and a dequantized §15 coefficient
    // reaches ≈ 2114 × 376 ≈ 795 k, so the product `C * coeff` reaches
    // ≈ 5.1e10 — well past `i32::MAX` (≈ 2.1e9). The spec's §16
    // pseudocode writes the multiply and the `>> 16` descale as one
    // expression with no declared width; the only width that holds the
    // pre-shift product for conformant input is 64-bit. After the
    // `>> 16` descale every term is back in the coefficient's own scale
    // and the butterfly sums fit `i32`, so the function's `i32`
    // input/output contract is preserved — only the intermediate
    // products widen.
    #[inline]
    fn m(c: i32, v: i32) -> i32 {
        ((c as i64 * v as i64) >> 16) as i32
    }
    let a = m(XC1S7, ip[1]) + m(XC7S1, ip[7]);
    let b = m(XC7S1, ip[1]) - m(XC1S7, ip[7]);
    let c = m(XC3S5, ip[3]) + m(XC5S3, ip[5]);
    let d = m(XC3S5, ip[5]) - m(XC5S3, ip[3]);

    let ad = m(XC4S4, a - c);
    let bd = m(XC4S4, b - d);
    let cd = a + c;
    let dd = b + d;
    let e = m(XC4S4, ip[0] + ip[4]);
    let f = m(XC4S4, ip[0] - ip[4]);
    let g = m(XC2S6, ip[2]) + m(XC6S2, ip[6]);
    let h = m(XC6S2, ip[2]) - m(XC2S6, ip[6]);
    let ed = e - g;
    let gd = e + g;
    let add = f + ad;
    let bdd = bd - h;
    let fd = f - ad;
    let hd = bd + h;

    [
        gd + cd,  // op[0]
        add + hd, // op[1]
        add - hd, // op[2]
        ed + dd,  // op[3]
        ed - dd,  // op[4]
        fd + bdd, // op[5]
        fd - bdd, // op[6]
        gd - cd,  // op[7]
    ]
}

/// Inverse-DCT one 8x8 block in place (spec §16).
///
/// `block` is the block's 64 dequantized coefficients in **raster
/// order** (`block[row * 8 + col]`). On return it holds the
/// reconstructed pixel (intra) or pixel-difference (inter) values, also
/// in raster order, as produced by the §16 two-pass fixed-point IDCT.
///
/// Steps (spec §16):
///
/// 1. **Row pass** — for each of the eight rows, replace
///    `block[row*8 + 0..=7]` with the butterfly of that row (no final
///    descale).
/// 2. **Column pass** — for each of the eight columns, replace
///    `block[0..=7 *8 + col]` with the butterfly of that column, each
///    output descaled by `>> 4`.
///
/// The values are intermediate transform outputs, not yet clamped to a
/// pixel range; clamping (and the intra `+128` level shift) belong to
/// the §17 frame-reconstruction stage, not the transform.
pub fn idct_block(block: &mut [i32; 64]) {
    // Pass 1: rows. `ip = SourceData; op = IntermediateData` with
    // `ip += 8 / op += 8` each iteration — i.e. an independent
    // eight-point transform of each row, written back in place.
    for row in 0..8 {
        let base = row * 8;
        let ip = [
            block[base],
            block[base + 1],
            block[base + 2],
            block[base + 3],
            block[base + 4],
            block[base + 5],
            block[base + 6],
            block[base + 7],
        ];
        let op = butterfly(ip);
        block[base..base + 8].copy_from_slice(&op);
    }

    // Pass 2: columns. `ip = IntermediateData; op = OutputData` with
    // `ip[k*8]` strides and `ip++ / op++` each iteration — i.e. an
    // eight-point transform of each column, with every output descaled
    // by `>> 4`.
    for col in 0..8 {
        let ip = [
            block[col],
            block[col + 8],
            block[col + 16],
            block[col + 24],
            block[col + 32],
            block[col + 40],
            block[col + 48],
            block[col + 56],
        ];
        let op = butterfly(ip);
        for k in 0..8 {
            block[col + k * 8] = op[k] >> 4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seven cosine constants are transcribed exactly from the §16
    /// `#define` list. This guards against a digit transposition.
    #[test]
    fn cosine_constants_match_spec() {
        assert_eq!(XC1S7, 64277);
        assert_eq!(XC2S6, 60547);
        assert_eq!(XC3S5, 54491);
        assert_eq!(XC4S4, 46341);
        assert_eq!(XC5S3, 36410);
        assert_eq!(XC6S2, 25080);
        assert_eq!(XC7S1, 12785);
    }

    /// An all-zero coefficient block transforms to an all-zero output:
    /// every butterfly term is `0`, both passes preserve zero, and the
    /// final `>> 4` of `0` is `0`.
    #[test]
    fn all_zero_block_stays_zero() {
        let mut block = [0i32; 64];
        idct_block(&mut block);
        assert_eq!(block, [0i32; 64]);
    }

    /// A DC-only block (only `block[0]` non-zero) produces a flat
    /// (constant) output across all 64 positions.
    ///
    /// Derivation from the §16 butterfly with only `ip[0]` non-zero:
    /// `_A=_B=_C=_D=0`, so `_Ad=_Bd=_Cd=_Dd=0`, `_G=_H=0`. With
    /// `ip[4]=0`, `_E = _F = (xC4S4 * ip[0]) >> 16`. Then `op[k]` for
    /// every `k` equals `_E` (each output is `_E ± 0`). So the row pass
    /// fills row 0 with `_E` and leaves the rest zero; the column pass
    /// then sees, in every column, only `ip[0] = _E` non-zero and
    /// fills the whole column with `((xC4S4 * _E) >> 16) >> 4`.
    #[test]
    fn dc_only_block_is_flat() {
        let dc: i32 = 100;
        let mut block = [0i32; 64];
        block[0] = dc;
        idct_block(&mut block);

        // Recompute the expected uniform value the same way the
        // transform does, term by term.
        let e_pass1 = (XC4S4 * dc) >> 16;
        let expected = ((XC4S4 * e_pass1) >> 16) >> 4;

        for (i, &v) in block.iter().enumerate() {
            assert_eq!(v, expected, "position {i} not flat");
        }
    }

    /// A negative DC coefficient yields a uniformly negative (or zero,
    /// per descaling) flat block — sign is preserved through the
    /// transform's arithmetic-shift descales.
    #[test]
    fn negative_dc_is_flat_and_signed() {
        let dc: i32 = -512;
        let mut block = [0i32; 64];
        block[0] = dc;
        idct_block(&mut block);

        let e_pass1 = (XC4S4 * dc) >> 16;
        let expected = ((XC4S4 * e_pass1) >> 16) >> 4;
        assert!(expected < 0, "negative DC should descale negative");
        for &v in block.iter() {
            assert_eq!(v, expected);
        }
    }

    /// The transform is linear in its input: transforming `2*X` gives
    /// the same result as adding the transforms of `X` and `X`, up to
    /// the per-term truncating shifts. Rather than assert exact
    /// linearity (truncation breaks it), this checks the weaker but
    /// transcription-sensitive property that doubling a single
    /// coefficient changes the output (i.e. that coefficient actually
    /// participates in the butterfly).
    #[test]
    fn every_input_coefficient_participates() {
        for k in 0..64 {
            let mut a = [0i32; 64];
            a[k] = 1024; // large enough to survive the >>16 / >>4 descales
            idct_block(&mut a);
            assert_ne!(
                a, [0i32; 64],
                "coefficient {k} produced an all-zero block — it is \
                 not wired into the butterfly"
            );
        }
    }

    /// Single off-DC coefficient blocks are not flat: an AC basis
    /// function varies across the block. This distinguishes a correct
    /// separable IDCT from a degenerate one that ignores AC terms.
    #[test]
    fn ac_only_block_is_not_flat() {
        // block[1] is the first horizontal AC coefficient.
        let mut block = [0i32; 64];
        block[1] = 2048;
        idct_block(&mut block);
        let first = block[0];
        assert!(
            block.iter().any(|&v| v != first),
            "an AC-only block must vary across positions"
        );
    }

    /// Symmetry sanity: a purely horizontal AC coefficient (row-0
    /// frequency, here `block[1]`) yields a block whose columns are
    /// identical (no vertical variation), because the vertical pass
    /// only ever sees the DC of each column. Each row is a horizontal
    /// cosine; every row is the same.
    #[test]
    fn horizontal_ac_has_no_vertical_variation() {
        let mut block = [0i32; 64];
        block[1] = 2048;
        idct_block(&mut block);
        for col in 0..8 {
            let top = block[col];
            for row in 1..8 {
                assert_eq!(
                    block[row * 8 + col],
                    top,
                    "column {col} varies vertically for a horizontal-AC \
                     input"
                );
            }
        }
    }

    /// A maximally-large conformant dequantized coefficient does not
    /// overflow. The largest coded coefficient is Table 18 CATEGORY6's
    /// 2114, and the largest §15 factor is 376, so a single dequantized
    /// coefficient can reach ≈ 795 k. Filling a whole block with that
    /// value and transforming it must not panic — the multiply is done
    /// in `i64`. (A pre-fix `i32` multiply panicked with "attempt to
    /// multiply with overflow" here.)
    #[test]
    fn max_conformant_coefficient_does_not_overflow() {
        let max_dequant: i32 = 2114 * 376;
        let mut block = [max_dequant; 64];
        idct_block(&mut block); // must not panic
                                // And a single max DC coefficient (the worst single-term case).
        let mut dc_block = [0i32; 64];
        dc_block[0] = max_dequant;
        idct_block(&mut dc_block);
    }

    /// Mirror of the previous test: a purely vertical AC coefficient
    /// (`block[8]`, the first vertical frequency) yields a block whose
    /// rows are identical (no horizontal variation).
    #[test]
    fn vertical_ac_has_no_horizontal_variation() {
        let mut block = [0i32; 64];
        block[8] = 2048;
        idct_block(&mut block);
        for row in 0..8 {
            let left = block[row * 8];
            for col in 1..8 {
                assert_eq!(
                    block[row * 8 + col],
                    left,
                    "row {row} varies horizontally for a vertical-AC \
                     input"
                );
            }
        }
    }
}

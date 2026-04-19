//! VP6 DSP primitives — IDCT, loop filter, sub-pel filters.
//!
//! Direct Rust ports of FFmpeg's `libavcodec/vp3dsp.c` (IDCT + loop
//! filter) and `vp6dsp.c` (`vp6_filter_diag4`). The chroma-filter path
//! used for non-diagonal inter-MC cases mirrors `h264chroma_template.c`
//! with `put_h264_chroma_pixels_tab[0]` — an 8x8 block put, bilinear
//! in x and y.

const X_C1S7: i32 = 64277;
const X_C2S6: i32 = 60547;
const X_C3S5: i32 = 54491;
const X_C4S4: i32 = 46341;
const X_C5S3: i32 = 36410;
const X_C6S2: i32 = 25080;
const X_C7S1: i32 = 12785;

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Full 8-column IDCT. Operates on a 64-coefficient block in zigzag-
/// scanned order (caller is expected to have deposited coefficients
/// through the scan permutation before calling).
///
/// `mode` controls whether the output is "put" (overwrite dst — the
/// intra path) or "add" (residual to be added to predicted pixels —
/// the inter path), matching FFmpeg's `idct(dst, stride, input, type)`.
fn idct_full(dst: &mut [u8], stride: usize, input: &mut [i16; 64], mode: IdctMode) {
    // Inverse DCT on the rows now (== columns in FFmpeg's layout).
    for col in 0..8 {
        let p = |i: usize| input[i * 8 + col] as i32;
        let has_ac = (1..8).any(|r| p(r) != 0);
        let has_any = has_ac || p(0) != 0;
        if has_any {
            let a = mul(X_C1S7, p(1)) + mul(X_C7S1, p(7));
            let b = mul(X_C7S1, p(1)) - mul(X_C1S7, p(7));
            let c = mul(X_C3S5, p(3)) + mul(X_C5S3, p(5));
            let d = mul(X_C3S5, p(5)) - mul(X_C5S3, p(3));

            let ad = mul(X_C4S4, a - c);
            let bd = mul(X_C4S4, b - d);
            let cd = a + c;
            let dd = b + d;

            let e = mul(X_C4S4, p(0) + p(4));
            let f = mul(X_C4S4, p(0) - p(4));

            let g = mul(X_C2S6, p(2)) + mul(X_C6S2, p(6));
            let h = mul(X_C6S2, p(2)) - mul(X_C2S6, p(6));

            let ed = e - g;
            let gd = e + g;
            let add_ = f + ad;
            let bdd = bd - h;
            let fd = f - ad;
            let hd = bd + h;

            input[0 * 8 + col] = (gd + cd) as i16;
            input[7 * 8 + col] = (gd - cd) as i16;
            input[1 * 8 + col] = (add_ + hd) as i16;
            input[2 * 8 + col] = (add_ - hd) as i16;
            input[3 * 8 + col] = (ed + dd) as i16;
            input[4 * 8 + col] = (ed - dd) as i16;
            input[5 * 8 + col] = (fd + bdd) as i16;
            input[6 * 8 + col] = (fd - bdd) as i16;
        }
    }

    // IDCT on rows.
    for row in 0..8 {
        let offset = row * 8;
        let p = |i: usize| input[offset + i] as i32;
        let has_ac = (1..8).any(|c| p(c) != 0);
        let dst_row = row * stride;

        if has_ac {
            let a = mul(X_C1S7, p(1)) + mul(X_C7S1, p(7));
            let b = mul(X_C7S1, p(1)) - mul(X_C1S7, p(7));
            let c = mul(X_C3S5, p(3)) + mul(X_C5S3, p(5));
            let d = mul(X_C3S5, p(5)) - mul(X_C5S3, p(3));

            let ad = mul(X_C4S4, a - c);
            let bd = mul(X_C4S4, b - d);
            let cd = a + c;
            let dd = b + d;

            let mut e = mul(X_C4S4, p(0) + p(4)) + 8;
            let mut f = mul(X_C4S4, p(0) - p(4)) + 8;
            if matches!(mode, IdctMode::Put) {
                e += 16 * 128;
                f += 16 * 128;
            }

            let g = mul(X_C2S6, p(2)) + mul(X_C6S2, p(6));
            let h = mul(X_C6S2, p(2)) - mul(X_C2S6, p(6));

            let ed = e - g;
            let gd = e + g;
            let add_ = f + ad;
            let bdd = bd - h;
            let fd = f - ad;
            let hd = bd + h;

            write_row(
                dst,
                dst_row,
                stride,
                mode,
                [
                    (gd + cd) >> 4,
                    (add_ + hd) >> 4,
                    (add_ - hd) >> 4,
                    (ed + dd) >> 4,
                    (ed - dd) >> 4,
                    (fd + bdd) >> 4,
                    (fd - bdd) >> 4,
                    (gd - cd) >> 4,
                ],
            );
        } else {
            match mode {
                IdctMode::Put => {
                    let v = clip_u8(128 + ((X_C4S4 * p(0) + (8 << 16)) >> 20));
                    for c in 0..8 {
                        dst[dst_row + c] = v;
                    }
                }
                IdctMode::Add => {
                    if p(0) != 0 {
                        let v = (X_C4S4 * p(0) + (8 << 16)) >> 20;
                        for c in 0..8 {
                            dst[dst_row + c] = clip_u8(dst[dst_row + c] as i32 + v);
                        }
                    }
                }
            }
        }
    }

    // FFmpeg zeroes the block after IDCT; do likewise so re-entry is
    // clean.
    for v in input.iter_mut() {
        *v = 0;
    }
}

#[inline]
fn mul(a: i32, b: i32) -> i32 {
    // Emulates FFmpeg's `M(a,b) = ((SUINT)(a) * (b)) >> 16`:
    // unsigned multiply in 32-bit, arithmetic right shift of 16.
    a.wrapping_mul(b) >> 16
}

#[inline]
fn write_row(dst: &mut [u8], base: usize, _stride: usize, mode: IdctMode, vals: [i32; 8]) {
    match mode {
        IdctMode::Put => {
            for (c, v) in vals.iter().enumerate() {
                dst[base + c] = clip_u8(*v);
            }
        }
        IdctMode::Add => {
            for (c, v) in vals.iter().enumerate() {
                dst[base + c] = clip_u8(dst[base + c] as i32 + *v);
            }
        }
    }
}

/// "Put" flavour of the IDCT: overwrite the 8x8 `dst` tile with the
/// reconstructed pixels (intra path).
pub fn idct_put(dst: &mut [u8], stride: usize, block: &mut [i16; 64]) {
    idct_full(dst, stride, block, IdctMode::Put);
}

/// "Add" flavour of the IDCT: treat the block as a residual and add it
/// to the pixels already in `dst` (inter path).
pub fn idct_add(dst: &mut [u8], stride: usize, block: &mut [i16; 64]) {
    idct_full(dst, stride, block, IdctMode::Add);
}

/// DC-only add path — when the 8x8 block carries only a DC term, we can
/// skip the full IDCT and broadcast `(block[0] + 15) >> 5` across the
/// tile. Matches `vp3_idct_dc_add_c`.
pub fn idct_dc_add(dst: &mut [u8], stride: usize, block: &mut [i16; 64]) {
    let dc = (block[0] as i32 + 15) >> 5;
    for r in 0..8 {
        for c in 0..8 {
            dst[r * stride + c] = clip_u8(dst[r * stride + c] as i32 + dc);
        }
    }
    block[0] = 0;
}

#[derive(Clone, Copy, Debug)]
enum IdctMode {
    Put,
    Add,
}

// -- loop filter ----------------------------------------------------------

/// Compute the VP3-style loop-filter bounding values. Populates
/// `bounding` so that `bounding[127 + x]` gives the filter output for
/// delta `x` in `[-128, 127]`. Matches `ff_vp3dsp_set_bounding_values`.
pub fn set_bounding_values(bounding: &mut [i32; 256], filter_limit: u8) {
    for v in bounding.iter_mut() {
        *v = 0;
    }
    let fl = filter_limit as i32;
    let base = 127isize;
    for x in 0..fl {
        bounding[(base - x as isize) as usize] = -x;
        bounding[(base + x as isize) as usize] = x;
    }
    let mut value = fl;
    let mut x = fl;
    while x < 128 && value > 0 {
        bounding[(base + x as isize) as usize] = value;
        bounding[(base - x as isize) as usize] = -value;
        x += 1;
        value -= 1;
    }
}

/// Horizontal VP3 loop filter across a 12-pixel vertical edge starting
/// at `first_pixel`. `stride` is the row stride in `data`.
pub fn h_loop_filter_12(data: &mut [u8], first_pixel: usize, stride: usize, bounding: &[i32; 256]) {
    for i in 0..12 {
        let p = first_pixel + i * stride;
        let fv =
            (data[p - 2] as i32 - data[p + 1] as i32) + (data[p] as i32 - data[p - 1] as i32) * 3;
        let fv = bounding[(127 + ((fv + 4) >> 3)).clamp(0, 255) as usize];
        data[p - 1] = clip_u8(data[p - 1] as i32 + fv);
        data[p] = clip_u8(data[p] as i32 - fv);
    }
}

/// Vertical VP3 loop filter across a 12-pixel horizontal edge.
pub fn v_loop_filter_12(data: &mut [u8], first_pixel: usize, stride: usize, bounding: &[i32; 256]) {
    for i in 0..12 {
        let p = first_pixel + i;
        let two_above = p.wrapping_sub(2 * stride);
        let one_above = p.wrapping_sub(stride);
        let fv = (data[two_above] as i32 - data[p + stride] as i32)
            + (data[p] as i32 - data[one_above] as i32) * 3;
        let fv = bounding[(127 + ((fv + 4) >> 3)).clamp(0, 255) as usize];
        data[one_above] = clip_u8(data[one_above] as i32 + fv);
        data[p] = clip_u8(data[p] as i32 - fv);
    }
}

// -- sub-pel filter primitives -------------------------------------------

/// 4-tap 1D filter, 8x8 block (`vp6_filter_hv4` in FFmpeg). `delta` is
/// the element stride between taps (1 for horizontal, `stride` for
/// vertical).
pub fn filter_hv4_into(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: usize,
    delta: i32,
    w: &[i16; 4],
) {
    for y in 0..8usize {
        for x in 0..8usize {
            let sx = (src_base + y * src_stride) as isize + x as isize;
            let s0 = src[(sx - delta as isize) as usize] as i32;
            let s1 = src[sx as usize] as i32;
            let s2 = src[(sx + delta as isize) as usize] as i32;
            let s3 = src[(sx + 2 * delta as isize) as usize] as i32;
            let v =
                (s0 * w[0] as i32 + s1 * w[1] as i32 + s2 * w[2] as i32 + s3 * w[3] as i32 + 64)
                    >> 7;
            dst[dst_base + y * dst_stride + x] = clip_u8(v);
        }
    }
}

/// 8x8 `put` path via the 2D 4-tap separable bicubic filter — matches
/// `vp6_filter_diag4_c`. `src_base` points at the top-left of the
/// target 8x8 tile in `src` (the filter reads 1 row above and 2 rows
/// below).
pub fn filter_diag4(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: usize,
    h_weights: &[i16; 4],
    v_weights: &[i16; 4],
) {
    let mut tmp = [0i32; 8 * 11];
    let src_top = src_base - src_stride; // one row above
    for y in 0..11usize {
        for x in 0..8usize {
            let pos = (src_top + y * src_stride) as isize + x as isize;
            let s0 = src[(pos - 1) as usize] as i32;
            let s1 = src[pos as usize] as i32;
            let s2 = src[(pos + 1) as usize] as i32;
            let s3 = src[(pos + 2) as usize] as i32;
            let v = (s0 * h_weights[0] as i32
                + s1 * h_weights[1] as i32
                + s2 * h_weights[2] as i32
                + s3 * h_weights[3] as i32
                + 64)
                >> 7;
            tmp[y * 8 + x] = v.clamp(0, 255);
        }
    }
    for y in 0..8usize {
        for x in 0..8usize {
            let i = (y + 1) * 8 + x;
            let v = (tmp[i - 8] * v_weights[0] as i32
                + tmp[i] * v_weights[1] as i32
                + tmp[i + 8] * v_weights[2] as i32
                + tmp[i + 16] * v_weights[3] as i32
                + 64)
                >> 7;
            dst[dst_base + y * dst_stride + x] = clip_u8(v);
        }
    }
}

/// 8x8 "put" H.264 chroma-filter. `x` and `y` are sub-pel offsets in
/// [0, 8); when either is zero we fall into the simpler 1D bilinear
/// path. Matches `put_h264_chroma_mc8` with `op_put`.
pub fn put_h264_chroma8(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: usize,
    x: i32,
    y: i32,
) {
    let a = (8 - x) * (8 - y);
    let b = x * (8 - y);
    let c = (8 - x) * y;
    let d = x * y;

    for row in 0..8usize {
        for col in 0..8usize {
            let sbase = src_base + row * src_stride + col;
            let v = if d != 0 {
                a * src[sbase] as i32
                    + b * src[sbase + 1] as i32
                    + c * src[sbase + src_stride] as i32
                    + d * src[sbase + src_stride + 1] as i32
            } else if b + c != 0 {
                let e = b + c;
                let step = if c != 0 { src_stride } else { 1 };
                a * src[sbase] as i32 + e * src[sbase + step] as i32
            } else {
                a * src[sbase] as i32
            };
            dst[dst_base + row * dst_stride + col] = clip_u8((v + 32) >> 6);
        }
    }
}

/// 8x8 block copy from `src` (at `src_base`, `src_stride`) into `dst`
/// (at `dst_base`, `dst_stride`). Used when the MV is integer-pel.
pub fn put_block8(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: usize,
) {
    for row in 0..8 {
        let d = dst_base + row * dst_stride;
        let s = src_base + row * src_stride;
        dst[d..d + 8].copy_from_slice(&src[s..s + 8]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_put_dc_only_produces_flat_block() {
        // DC-only block (linear index 0) → uniform pixels slightly
        // above mid-grey (the transform spreads the DC through both
        // the column and row passes).
        let mut block = [0i16; 64];
        block[0] = 200;
        let mut dst = [0u8; 64];
        idct_put(&mut dst, 8, &mut block);
        let v = dst[0];
        for &p in &dst {
            assert_eq!(p, v);
        }
        // Value should be above mid-grey.
        assert!(v > 128, "v={v}");
        // And shouldn't saturate.
        assert!(v < 255);
    }

    #[test]
    fn idct_add_round_trips_zero() {
        let mut block = [0i16; 64];
        let mut dst = [42u8; 64];
        idct_add(&mut dst, 8, &mut block);
        for &p in &dst {
            assert_eq!(p, 42);
        }
    }

    #[test]
    fn idct_dc_add_broadcasts() {
        let mut block = [0i16; 64];
        block[0] = 32; // (32 + 15) >> 5 = 1
        let mut dst = [50u8; 64];
        idct_dc_add(&mut dst, 8, &mut block);
        for &p in &dst {
            assert_eq!(p, 51);
        }
    }

    #[test]
    fn loop_filter_identity_zero_limit() {
        let mut bounding = [0i32; 256];
        set_bounding_values(&mut bounding, 0);
        for v in bounding.iter() {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn loop_filter_positive_limit_bell_curve() {
        let mut bounding = [0i32; 256];
        set_bounding_values(&mut bounding, 8);
        // At x = 0, value is 0.
        assert_eq!(bounding[127], 0);
        // At x = 1, value is 1 (pre-saturation).
        assert_eq!(bounding[128], 1);
        // Far out, value returns to 0.
        assert_eq!(bounding[127 + 32], 0);
    }

    #[test]
    fn put_block8_copies_exactly() {
        let src: Vec<u8> = (0u16..16 * 16).map(|v| v as u8).collect();
        let mut dst = [0u8; 16 * 16];
        put_block8(&mut dst, 0, 16, &src, 0, 16);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 16 + c], src[r * 16 + c]);
            }
        }
    }

    #[test]
    fn chroma8_integer_pel_matches_put_block8() {
        let src: Vec<u8> = (0u16..32 * 32).map(|v| v as u8).collect();
        let mut dst_a = [0u8; 32 * 32];
        let mut dst_b = [0u8; 32 * 32];
        // chroma (0, 0) should be an 8x8 block copy.
        put_h264_chroma8(&mut dst_a, 0, 32, &src, 0, 32, 0, 0);
        put_block8(&mut dst_b, 0, 32, &src, 0, 32);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    dst_a[r * 32 + c],
                    dst_b[r * 32 + c],
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn filter_hv4_bypass_is_identity() {
        // Filter [0, 128, 0, 0] / 128 with +64 rounding => identity.
        let src: Vec<u8> = (0u16..16 * 16).map(|v| v as u8).collect();
        let mut dst = [0u8; 16 * 16];
        let w = [0i16, 128, 0, 0];
        filter_hv4_into(&mut dst, 0, 16, &src, 2, 16, 1, &w);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 16 + c], src[2 + r * 16 + c]);
            }
        }
    }

    #[test]
    fn h_loop_filter_smooths_sharp_edge() {
        // Build a 12-row tall, 12-col wide scratch with a sharp vertical
        // step at col 6: cols 0..6 = 100, cols 6..12 = 200. The filter
        // should soften that edge at cols 5,6 (one each side).
        let mut data = [0u8; 12 * 12];
        for r in 0..12 {
            for c in 0..12 {
                data[r * 12 + c] = if c < 6 { 100 } else { 200 };
            }
        }
        let mut bounding = [0i32; 256];
        set_bounding_values(&mut bounding, 30);
        h_loop_filter_12(&mut data, 6, 12, &bounding);
        // After filtering: col 5 should rise, col 6 should fall; the
        // rest of the row is untouched.
        for r in 0..12 {
            assert!(
                data[r * 12 + 5] > 100,
                "row {r} col 5 should have risen (is {})",
                data[r * 12 + 5]
            );
            assert!(
                data[r * 12 + 6] < 200,
                "row {r} col 6 should have fallen (is {})",
                data[r * 12 + 6]
            );
            assert_eq!(data[r * 12 + 4], 100, "row {r} col 4 should be untouched");
            assert_eq!(data[r * 12 + 7], 200, "row {r} col 7 should be untouched");
        }
    }

    #[test]
    fn v_loop_filter_smooths_sharp_edge() {
        // Horizontal edge: rows 0..6 = 50, rows 6..12 = 220. Filter
        // across row 6 smooths it.
        let mut data = [0u8; 12 * 12];
        for r in 0..12 {
            for c in 0..12 {
                data[r * 12 + c] = if r < 6 { 50 } else { 220 };
            }
        }
        let mut bounding = [0i32; 256];
        set_bounding_values(&mut bounding, 30);
        v_loop_filter_12(&mut data, 6 * 12, 12, &bounding);
        for c in 0..12 {
            assert!(data[5 * 12 + c] > 50, "col {c} row 5 should have risen");
            assert!(data[6 * 12 + c] < 220, "col {c} row 6 should have fallen");
        }
    }
}

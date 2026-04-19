//! Synthetic test for the VP3-style loop filter in `render_mb_inter`.
//!
//! Builds a reference plane with a sharp block-edged step at column 8
//! in an otherwise uniform 32x16 area. Runs two inter-MB MC passes: one
//! with deblock-filtering disabled, one enabled. Confirms the filtered
//! output is measurably smoother across the edge by checking that the
//! mean absolute horizontal gradient at the former step is lower when
//! the filter is on.
//!
//! We exercise `render_mb_inter` directly rather than driving the whole
//! decoder, because synthesising a VP6 bitstream that hits every branch
//! is nontrivial. The scratch state is the same `BlockScratch` the
//! decoder builds, so this is a direct test of the MC-path filter
//! wiring.

use oxideav_vp6::{dsp, mb};

fn mean_abs_gradient(plane: &[u8], stride: usize, col: usize, rows: usize) -> f64 {
    let mut acc = 0u64;
    for r in 0..rows {
        let a = plane[r * stride + col - 1] as i32;
        let b = plane[r * stride + col] as i32;
        acc += (a - b).unsigned_abs() as u64;
    }
    acc as f64 / rows as f64
}

#[test]
fn inter_mc_deblock_softens_reference_block_edge() {
    // 32x32 planes (Y) — two MBs side by side. We set up an MB-sized
    // reference pattern with a block-aligned edge down the middle.
    let w = 32usize;
    let h = 32usize;
    let mut ref_y = vec![0u8; w * h];
    let uv_w = w / 2;
    let uv_h = h / 2;
    let ref_u = vec![128u8; uv_w * uv_h];
    let ref_v = vec![128u8; uv_w * uv_h];
    for r in 0..h {
        for c in 0..w {
            // Sharp step at column 16 (= 2 blocks worth).
            ref_y[r * w + c] = if c < 16 { 60 } else { 200 };
        }
    }

    // Build a scratch state as the decoder would: plane layout, no
    // coefficients, zero-MV residual.
    let mut scratch_off = mb::BlockScratch::new(2);
    scratch_off.reset_row(2);
    scratch_off.start_row(2);
    mb::init_dequant(&mut scratch_off, 0); // QP 0 → filter_limit 14
                                           // Override with an aggressive filter limit so the 140-magnitude
                                           // step gets meaningfully softened in this test pattern.
    dsp::set_bounding_values(&mut scratch_off.bounding_values, 80);
    // Force all 6 blocks to an INTER_NOVEC_PF-like MV=(2 quarter-pel,
    // 0) — that produces a 1-pixel integer-pel shift horizontally (MV
    // / coord_div = 2/4 = 0 integer pels for luma; mv.x & 3 = 2, so a
    // half-pel subpel phase). This puts the 8x8 read window on top of
    // the sharp edge — exactly what the loop filter should soften.
    for b in 0..6 {
        scratch_off.mv[b] = mb::Mv { x: 8, y: 0 }; // integer-pel dx = 2
    }
    // Zero residual.
    for b in 0..6 {
        for c in 0..64 {
            scratch_off.block_coeff[b][c] = 0;
        }
        scratch_off.idct_selector[b] = 12; // > 1 so add-path takes idct_add
    }
    let mut scratch_on = scratch_off.clone();

    // Render target: col-1 MB to cover the right-half block that
    // straddles the ref-plane edge. mb_col = 1 puts the MC source at
    // x = 16 (+ 2 integer offset from dx), reading straight across the
    // discontinuity in `ref_y`.
    let y_stride = w;
    let uv_stride = uv_w;
    let mut y_off = vec![0u8; y_stride * h];
    let mut u_off = vec![128u8; uv_stride * uv_h];
    let mut v_off = vec![128u8; uv_stride * uv_h];
    let mut y_on = vec![0u8; y_stride * h];
    let mut u_on = vec![128u8; uv_stride * uv_h];
    let mut v_on = vec![128u8; uv_stride * uv_h];

    let plane_w = [w, uv_w, uv_w];
    let plane_h = [h, uv_h, uv_h];
    // The per-MB output writes its 16x16 tile starting at mb_col*16 =
    // 0 or 16. We render MB 0 (left) — its block 1 (right half of the
    // MB, x=8..16 in the output) reads from ref_y around x=8..16 which
    // is fine on the off-side, but with dx=2 the source window shifts
    // to 10..18 — straddling the step at col 16. That's where the loop
    // filter should mask the seam.
    mb::render_mb_inter(
        &mut scratch_off,
        &mut y_off,
        y_stride,
        &mut u_off,
        uv_stride,
        &mut v_off,
        uv_stride,
        &ref_y,
        &ref_u,
        &ref_v,
        0,
        0,
        plane_w,
        plane_h,
        true,  // use_bicubic_luma
        false, // deblock off
    );
    mb::render_mb_inter(
        &mut scratch_on,
        &mut y_on,
        y_stride,
        &mut u_on,
        uv_stride,
        &mut v_on,
        uv_stride,
        &ref_y,
        &ref_u,
        &ref_v,
        0,
        0,
        plane_w,
        plane_h,
        true,
        true, // deblock on
    );

    // The block boundary in the *output* tile corresponds to the
    // former block edge in the reference at ref_x = 16. With dx=2, the
    // source tile starts at ref_x=2, so the ref edge sits at output
    // col 14 (= 16 - 2). The loop filter acts at columns 13 and 14.
    let grad_col = 14;
    let g_off = mean_abs_gradient(&y_off, y_stride, grad_col, 16);
    let g_on = mean_abs_gradient(&y_on, y_stride, grad_col, 16);
    eprintln!("gradient @ col {grad_col}: off={g_off:.2}, on={g_on:.2}");
    assert!(
        g_on < g_off,
        "loop filter should reduce gradient (off={g_off:.2}, on={g_on:.2})"
    );
}

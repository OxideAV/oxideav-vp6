//! VP6 Unrestricted Motion Vector (UMV) reconstruction-buffer borders
//! (spec §11.5).
//!
//! VP6 motion vectors are allowed to point at prediction blocks that
//! extend beyond the borders of the decoded image — the spec calls these
//! "unrestricted motion vectors" (§11.5). To make such fetches well
//! defined, the reconstruction buffer that the §17.2/§17.3 prediction
//! fetch ([`crate::inter::fetch_prediction_block`]) and the §11.4
//! fractional-pixel interpolation
//! ([`crate::interp::bilinear_block`] / [`crate::interp::bicubic_block`])
//! read against is **extended by 48 sample points in all four directions**
//! before any inter block is reconstructed.
//!
//! The extension is a pure pixel-duplication of the original image's edge
//! samples — every sample in the left border row column comes from the
//! original column-0 value of the same row, every sample in the top
//! border row from the original row-0 value of the same column, and so
//! on. Verbatim from §11.5 (On2 Technologies, document version 1.02,
//! August 2006):
//!
//! > To support this feature and the playback scaling features of the
//! > codec (see Section 2) the reconstruction buffers are extended by 48
//! > sample points in all directions:-
//! >
//! > The buffers are extended by duplicating the edge values 48 times.
//! > This is done first in x (horizontally) and then in y (vertically).
//!
//! The order matters at the four corner quadrants. Because the horizontal
//! pass runs first, the **left** border rows of the original image hold
//! the leftmost-original-column value 48 times along that row, and the
//! **right** border rows hold the rightmost-original-column value 48
//! times. The subsequent vertical pass then copies the *entire* width of
//! these horizontally-extended row buffers — original middle + left
//! border + right border — into the top 48 and bottom 48 rows of the
//! framed buffer. The result is that all four corner quadrants are
//! filled with the corresponding corner-pixel value of the original
//! image: the top-left 48×48 quadrant is uniform at the
//! `(row=0, col=0)` original value, the top-right at `(0, width-1)`, and
//! so on. This is exactly the "duplicate edge values" behaviour
//! described in §11.5 (Figure 13 shows the same arrangement).
//!
//! Because the extension is built by edge replication, any read from a
//! sample position inside the `±BORDER_SIZE`-wide extended frame is
//! equivalent to clamping the read position to the original image's
//! valid range — exactly the well-defined "clamp" semantics a UMV
//! fetch needs.
//!
//! The 48-sample size is what §11.5 specifies. A VP6 motion-vector
//! component can address fractional positions across up to
//! 48 = 12 macro-blocks of ¼-pixel range either side of the source
//! position, plus a small additional safety margin for the §11.4
//! 4-tap bicubic interpolator (which reads two samples either side of
//! the integer sample position). The two together fit inside the
//! 48-sample border the spec mandates.
//!
//! This stage reads **no BoolCoder bits** — it is pure edge-replication
//! pixel arithmetic on an already-reconstructed frame buffer — so it
//! advances the decoder past round 7 without touching the contested
//! §7.3 `Split` formula. The UMV border is built *once* per
//! reconstructed frame, after frame reconstruction is otherwise complete
//! and before that frame is consumed as a reference by a later frame's
//! inter prediction.

/// Width, in samples, of the §11.5 UMV border on each side of the
/// reconstructed frame.
///
/// The spec mandates 48 samples of border in every direction
/// (left, right, top, bottom). Verbatim from §11.5:
///
/// > the reconstruction buffers are extended by 48 sample points in
/// > all directions:-
pub const UMV_BORDER_SIZE: usize = 48;

/// Compute the stride (line length, in samples) of a UMV-bordered buffer
/// holding an image of `width` columns plus a [`UMV_BORDER_SIZE`]-sample
/// left border and a [`UMV_BORDER_SIZE`]-sample right border.
///
/// The returned value is `width + 2 * UMV_BORDER_SIZE`. The caller
/// allocates a buffer of `extended_stride(width) * extended_height(height)`
/// samples to hold the image and its UMV border.
pub const fn extended_stride(width: usize) -> usize {
    width + 2 * UMV_BORDER_SIZE
}

/// Compute the row count of a UMV-bordered buffer holding an image of
/// `height` rows plus a [`UMV_BORDER_SIZE`]-sample top border and a
/// [`UMV_BORDER_SIZE`]-sample bottom border.
///
/// The returned value is `height + 2 * UMV_BORDER_SIZE`.
pub const fn extended_height(height: usize) -> usize {
    height + 2 * UMV_BORDER_SIZE
}

/// Compute the linear buffer index of the original-image origin
/// (`row=0, col=0`) inside a UMV-bordered buffer with the given
/// `extended_stride`.
///
/// The origin sits [`UMV_BORDER_SIZE`] rows down and
/// [`UMV_BORDER_SIZE`] columns in from the buffer's `(0, 0)` corner.
/// Callers pass this as the `base_pos` argument to
/// [`crate::inter::fetch_prediction_block`].
pub const fn origin_offset(extended_stride: usize) -> usize {
    UMV_BORDER_SIZE * extended_stride + UMV_BORDER_SIZE
}

/// Extend a reconstructed frame in place by [`UMV_BORDER_SIZE`] samples
/// in each direction, per spec §11.5.
///
/// `buf` is a buffer of `extended_stride(width) * extended_height(height)`
/// samples that already holds the original `width * height` image at
/// the inner rectangle starting at [`origin_offset`]. The function fills
/// in the four border strips (left, right, top, bottom) and the four
/// corner quadrants by edge replication.
///
/// `width` and `height` are the dimensions of the original image (the
/// inner rectangle), not of the extended buffer.
///
/// The extension is performed in the spec-mandated order: horizontal
/// first (every original row's left/right borders), then vertical (top
/// and bottom border rows copied from the topmost / bottommost
/// horizontally-extended row).
///
/// # Panics
///
/// Panics if `buf.len()` is smaller than the extended buffer size, or
/// if `width == 0` / `height == 0` (a zero-width or zero-height image
/// has no edge to replicate).
pub fn extend_border(buf: &mut [u8], width: usize, height: usize) {
    assert!(width > 0, "extend_border: width must be > 0");
    assert!(height > 0, "extend_border: height must be > 0");

    let stride = extended_stride(width);
    let total_rows = extended_height(height);
    let needed = stride * total_rows;
    assert!(
        buf.len() >= needed,
        "extend_border: buffer too small ({} < {} = stride {} * rows {})",
        buf.len(),
        needed,
        stride,
        total_rows
    );

    // --- Horizontal pass: spec §11.5 "first in x (horizontally)". ---
    //
    // For each original-image row (there are `height` of them, occupying
    // rows `UMV_BORDER_SIZE..UMV_BORDER_SIZE + height` of the extended
    // buffer), copy the leftmost-original-column sample into all
    // `UMV_BORDER_SIZE` left-border columns of the same row, and the
    // rightmost-original-column sample into all `UMV_BORDER_SIZE`
    // right-border columns of the same row.
    for r in 0..height {
        let row_start = (UMV_BORDER_SIZE + r) * stride;
        let left_edge = buf[row_start + UMV_BORDER_SIZE];
        let right_edge = buf[row_start + UMV_BORDER_SIZE + width - 1];
        for c in 0..UMV_BORDER_SIZE {
            buf[row_start + c] = left_edge;
            buf[row_start + UMV_BORDER_SIZE + width + c] = right_edge;
        }
    }

    // --- Vertical pass: spec §11.5 "then in y (vertically)". ---
    //
    // For each top-border row (rows `0..UMV_BORDER_SIZE` of the extended
    // buffer), copy the *entire* width of the topmost original-image row
    // (which has already been horizontally-extended by the pass above —
    // its left and right borders are filled with the original
    // (row=0, col=0) and (row=0, col=width-1) values respectively). Same
    // for the bottom border.
    //
    // This is what makes the four corner quadrants uniform at the
    // corresponding corner sample of the original image, per §11.5's
    // ordering ("first in x, then in y").
    let top_src = UMV_BORDER_SIZE * stride;
    let bottom_src = (UMV_BORDER_SIZE + height - 1) * stride;
    for r in 0..UMV_BORDER_SIZE {
        let top_dst = r * stride;
        let bottom_dst = (UMV_BORDER_SIZE + height + r) * stride;
        // Copy whole-row span (left border + original samples + right
        // border) into the border row.
        for c in 0..stride {
            buf[top_dst + c] = buf[top_src + c];
            buf[bottom_dst + c] = buf[bottom_src + c];
        }
    }
}

/// Allocate a fresh UMV-bordered buffer and copy `image` (a `width *
/// height` plane in raster order) into the inner rectangle, then fill
/// in the §11.5 borders by edge replication.
///
/// Returns `(buf, stride, origin)` where `buf` is the extended buffer,
/// `stride` is the line length in samples, and `origin` is the buffer
/// index of the original `(row=0, col=0)` sample (i.e. the base position
/// callers pass to [`crate::inter::fetch_prediction_block`] for a
/// zero motion vector).
///
/// # Panics
///
/// Panics if `image.len()` is not exactly `width * height`, or if
/// `width == 0` / `height == 0`.
pub fn build_extended_buffer(image: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    assert!(width > 0, "build_extended_buffer: width must be > 0");
    assert!(height > 0, "build_extended_buffer: height must be > 0");
    assert_eq!(
        image.len(),
        width * height,
        "build_extended_buffer: image.len() {} != width * height {}",
        image.len(),
        width * height,
    );

    let stride = extended_stride(width);
    let rows = extended_height(height);
    let mut buf = vec![0u8; stride * rows];

    // Copy the original image into the inner rectangle.
    for r in 0..height {
        let src_row = r * width;
        let dst_row = (UMV_BORDER_SIZE + r) * stride + UMV_BORDER_SIZE;
        buf[dst_row..dst_row + width].copy_from_slice(&image[src_row..src_row + width]);
    }

    extend_border(&mut buf, width, height);

    (buf, stride, origin_offset(stride))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_size_matches_spec() {
        // §11.5: "extended by 48 sample points in all directions".
        assert_eq!(UMV_BORDER_SIZE, 48);
    }

    #[test]
    fn extended_stride_adds_border_each_side() {
        // 2 × 48 + width.
        assert_eq!(extended_stride(1), 97);
        assert_eq!(extended_stride(16), 112);
        assert_eq!(extended_stride(320), 416);
        assert_eq!(extended_stride(1920), 2016);
    }

    #[test]
    fn extended_height_adds_border_each_side() {
        assert_eq!(extended_height(1), 97);
        assert_eq!(extended_height(16), 112);
        assert_eq!(extended_height(240), 336);
        assert_eq!(extended_height(1080), 1176);
    }

    #[test]
    fn origin_offset_is_top_left_inner_corner() {
        // The origin row is UMV_BORDER_SIZE down; within that row, the
        // first inner column is UMV_BORDER_SIZE in.
        let stride = extended_stride(16);
        assert_eq!(origin_offset(stride), 48 * stride + 48);
        let stride = extended_stride(320);
        assert_eq!(origin_offset(stride), 48 * stride + 48);
    }

    #[test]
    fn build_returns_consistent_geometry() {
        let image = vec![100u8; 16 * 8];
        let (buf, stride, origin) = build_extended_buffer(&image, 16, 8);
        assert_eq!(stride, extended_stride(16));
        assert_eq!(buf.len(), extended_stride(16) * extended_height(8));
        assert_eq!(origin, origin_offset(stride));
    }

    #[test]
    fn build_preserves_inner_image() {
        // Walking pattern so each cell has a unique value.
        let width = 8usize;
        let height = 4usize;
        let image: Vec<u8> = (0..(width * height))
            .map(|i| (i as u8).wrapping_add(7))
            .collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        for r in 0..height {
            for c in 0..width {
                assert_eq!(
                    buf[origin + r * stride + c],
                    image[r * width + c],
                    "inner image preserved at row {r}, col {c}"
                );
            }
        }
    }

    #[test]
    fn left_border_row_replicates_left_edge() {
        // Different value per original-image row so we can tell rows
        // apart in the left-border check.
        let width = 4usize;
        let height = 6usize;
        let mut image = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                image[r * width + c] = (10 + r as u8) * 10 + c as u8;
            }
        }
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        for r in 0..height {
            let left_edge_value = image[r * width];
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin + r * stride - UMV_BORDER_SIZE + c;
                assert_eq!(
                    buf[idx], left_edge_value,
                    "left border row {r} col {c} replicates leftmost original col"
                );
            }
        }
    }

    #[test]
    fn right_border_row_replicates_right_edge() {
        let width = 4usize;
        let height = 6usize;
        let mut image = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                image[r * width + c] = (20 + r as u8) * 10 + c as u8;
            }
        }
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        for r in 0..height {
            let right_edge_value = image[r * width + width - 1];
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin + r * stride + width + c;
                assert_eq!(
                    buf[idx], right_edge_value,
                    "right border row {r} col {c} replicates rightmost original col"
                );
            }
        }
    }

    #[test]
    fn top_border_replicates_top_row() {
        // Different value per column so we can check per-column
        // replication across the 48 top-border rows.
        let width = 5usize;
        let height = 3usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 30 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        // Top border occupies rows 0..UMV_BORDER_SIZE of the extended
        // buffer; column c (original-image coordinates) sits at
        // `origin - UMV_BORDER_SIZE * stride + c` for the topmost border
        // row, and so on rising back toward `origin` for the bottommost
        // border row.
        for r in 0..UMV_BORDER_SIZE {
            let top_row_idx = origin - (UMV_BORDER_SIZE - r) * stride;
            for c in 0..width {
                assert_eq!(
                    buf[top_row_idx + c],
                    image[c],
                    "top border row {r} col {c} replicates top row of original",
                );
            }
        }
    }

    #[test]
    fn bottom_border_replicates_bottom_row() {
        let width = 5usize;
        let height = 3usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 40 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        // The last original-image row is at row UMV_BORDER_SIZE + height
        // - 1 of the extended buffer; the bottom border is the
        // UMV_BORDER_SIZE rows immediately below it.
        for r in 0..UMV_BORDER_SIZE {
            let bottom_row_idx = origin + (height + r) * stride;
            let last_row_start = (height - 1) * width;
            for c in 0..width {
                assert_eq!(
                    buf[bottom_row_idx + c],
                    image[last_row_start + c],
                    "bottom border row {r} col {c} replicates bottom row of original",
                );
            }
        }
    }

    #[test]
    fn top_left_corner_quadrant_uniform_at_top_left_pixel() {
        // §11.5's "first in x, then in y" ordering means the top
        // border rows copy whole rows from the topmost
        // *horizontally-extended* row — that row's left border (the 48
        // samples to the left of the original image) is uniform at the
        // (0, 0) value. So the entire top-left 48×48 quadrant of the
        // extended buffer must hold the (0, 0) value.
        let width = 6usize;
        let height = 6usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 50 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        let top_left_value = image[0];

        for r in 0..UMV_BORDER_SIZE {
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin - UMV_BORDER_SIZE * stride - UMV_BORDER_SIZE + r * stride + c;
                assert_eq!(
                    buf[idx], top_left_value,
                    "top-left quadrant uniform at (0,0) image value (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn top_right_corner_quadrant_uniform_at_top_right_pixel() {
        let width = 6usize;
        let height = 6usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 60 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        let top_right_value = image[width - 1];

        for r in 0..UMV_BORDER_SIZE {
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin - UMV_BORDER_SIZE * stride + width + r * stride + c;
                assert_eq!(
                    buf[idx], top_right_value,
                    "top-right quadrant uniform at (0, width-1) image value (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn bottom_left_corner_quadrant_uniform_at_bottom_left_pixel() {
        let width = 6usize;
        let height = 6usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 70 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        let bottom_left_value = image[(height - 1) * width];

        for r in 0..UMV_BORDER_SIZE {
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin + (height + r) * stride - UMV_BORDER_SIZE + c;
                assert_eq!(
                    buf[idx], bottom_left_value,
                    "bottom-left quadrant uniform at (height-1, 0) image value (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn bottom_right_corner_quadrant_uniform_at_bottom_right_pixel() {
        let width = 6usize;
        let height = 6usize;
        let image: Vec<u8> = (0..(width * height)).map(|i| 80 + i as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        let bottom_right_value = image[height * width - 1];

        for r in 0..UMV_BORDER_SIZE {
            for c in 0..UMV_BORDER_SIZE {
                let idx = origin + (height + r) * stride + width + c;
                assert_eq!(
                    buf[idx], bottom_right_value,
                    "bottom-right quadrant uniform at (height-1, width-1) value (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn extend_border_idempotent_on_already_extended_buffer() {
        // Running extend_border twice should produce the same buffer:
        // edge replication on top of the already-replicated borders is
        // a no-op because the "edge value" at column UMV_BORDER_SIZE
        // is the same on the second pass as the first.
        let width = 8usize;
        let height = 5usize;
        let image: Vec<u8> = (0..(width * height))
            .map(|i| (i as u8).wrapping_mul(3))
            .collect();
        let (mut buf, _stride, _origin) = build_extended_buffer(&image, width, height);
        let first = buf.clone();
        extend_border(&mut buf, width, height);
        assert_eq!(buf, first, "second extend_border call is a no-op");
    }

    #[test]
    fn extend_border_one_by_one_image_uniform_buffer() {
        // A 1×1 image: the single sample is the value for every cell of
        // the entire extended buffer.
        let value = 173u8;
        let (buf, stride, _origin) = build_extended_buffer(&[value], 1, 1);
        let rows = extended_height(1);
        assert_eq!(buf.len(), stride * rows);
        for &b in buf.iter() {
            assert_eq!(
                b, value,
                "every sample of a 1×1 extended image is the single value"
            );
        }
    }

    #[test]
    fn extend_border_one_row_image_replicates_row_to_all_rows() {
        let width = 8usize;
        let image: Vec<u8> = (0..width as u8).map(|c| 100 + c).collect();
        let (buf, stride, _origin) = build_extended_buffer(&image, width, 1);
        let rows = extended_height(1);

        for r in 0..rows {
            // Each row's original-image columns must replicate the
            // single source row exactly.
            let row_origin = r * stride + UMV_BORDER_SIZE;
            for c in 0..width {
                assert_eq!(buf[row_origin + c], image[c]);
            }
        }
    }

    #[test]
    fn extend_border_one_column_image_replicates_column_to_all_columns() {
        let height = 6usize;
        let image: Vec<u8> = (0..height as u8).map(|r| 200 + r).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, 1, height);

        for r in 0..height {
            let row_value = image[r];
            for c in 0..stride {
                assert_eq!(
                    buf[origin + r * stride - UMV_BORDER_SIZE + c],
                    row_value,
                    "every column of single-column row {r} is the row's single value (c={c})"
                );
            }
        }
    }

    #[test]
    fn extend_border_inner_image_unmodified() {
        // The inner rectangle must be byte-identical to the source image
        // after extend_border runs. (Verifying we never overwrite our
        // own original samples while replicating edges.)
        let width = 12usize;
        let height = 7usize;
        let image: Vec<u8> = (0..(width * height) as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        for r in 0..height {
            for c in 0..width {
                assert_eq!(
                    buf[origin + r * stride + c],
                    image[r * width + c],
                    "inner image untouched at (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn fetch_prediction_compatible_with_zero_mv_at_origin() {
        // Smoke-integrate with crate::inter: a zero-MV fetch from the
        // UMV-bordered buffer at `origin` must reproduce the inner
        // image's top-left 8×8 block exactly.
        let width = 16usize;
        let height = 8usize;
        let image: Vec<u8> = (0..(width * height) as u8).collect();
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        let mut pred = [0u8; 64];
        crate::inter::fetch_prediction_block(&buf, origin, stride, 0, 0, &mut pred);

        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    pred[r * 8 + c],
                    image[r * width + c],
                    "zero-MV fetch reproduces inner image top-left block (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    fn fetch_prediction_into_left_border_uses_clamped_edge() {
        // A negative-x MV that pulls the entire prediction block into
        // the left UMV border: every sample of the prediction block
        // should be the leftmost-original-column value of the
        // corresponding original row, because the border is edge-
        // replication.
        let width = 16usize;
        let height = 8usize;
        let mut image = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                image[r * width + c] = (r as u8) * 16 + c as u8;
            }
        }
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        // Pull the prediction block 8 samples left of origin: the entire
        // block is in the left border zone.
        let mut pred = [0u8; 64];
        crate::inter::fetch_prediction_block(&buf, origin, stride, -8, 0, &mut pred);

        for r in 0..8 {
            let left_edge = image[r * width];
            for c in 0..8 {
                assert_eq!(
                    pred[r * 8 + c],
                    left_edge,
                    "prediction in left border reads leftmost original column for row {r}",
                );
            }
        }
    }

    #[test]
    fn fetch_prediction_into_top_border_uses_clamped_edge() {
        // Same idea, but with a negative-y MV that pulls the entire
        // prediction block into the top UMV border. Every prediction
        // sample should be the topmost-original-row value of the
        // corresponding original column.
        let width = 16usize;
        let height = 8usize;
        let mut image = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                image[r * width + c] = (r as u8) * 16 + c as u8;
            }
        }
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);

        // Pull the prediction block 8 rows above origin: entirely in
        // the top border.
        let mut pred = [0u8; 64];
        crate::inter::fetch_prediction_block(&buf, origin, stride, 0, -8, &mut pred);

        for r in 0..8 {
            for c in 0..8 {
                let top_edge = image[c];
                assert_eq!(
                    pred[r * 8 + c],
                    top_edge,
                    "prediction in top border reads topmost original row for col {c}",
                );
            }
        }
    }

    #[test]
    fn fetch_prediction_at_max_border_extent_in_bounds() {
        // The UMV border supports MV magnitudes up to UMV_BORDER_SIZE
        // samples either side; a fetch at exactly that limit must
        // remain in-bounds (no panic / no UB).
        let width = 16usize;
        let height = 8usize;
        let image = vec![55u8; width * height];
        let (buf, stride, origin) = build_extended_buffer(&image, width, height);
        let mut pred = [0u8; 64];

        // dx = -UMV_BORDER_SIZE, dy = -UMV_BORDER_SIZE: top-left
        // corner quadrant.
        crate::inter::fetch_prediction_block(
            &buf,
            origin,
            stride,
            -(UMV_BORDER_SIZE as i32),
            -(UMV_BORDER_SIZE as i32),
            &mut pred,
        );
        assert!(pred.iter().all(|&b| b == 55));

        // dx = width, dy = height: bottom-right corner quadrant.
        crate::inter::fetch_prediction_block(
            &buf,
            origin,
            stride,
            width as i32,
            height as i32,
            &mut pred,
        );
        assert!(pred.iter().all(|&b| b == 55));
    }

    #[test]
    fn buffer_size_matches_extended_dimensions() {
        // For each of a handful of inner-image dimensions, the produced
        // buffer must be exactly extended_stride * extended_height.
        for &(w, h) in &[(1, 1), (8, 8), (16, 16), (320, 240), (640, 480)] {
            let image = vec![0u8; w * h];
            let (buf, stride, _origin) = build_extended_buffer(&image, w, h);
            assert_eq!(stride, extended_stride(w));
            assert_eq!(buf.len(), extended_stride(w) * extended_height(h));
        }
    }

    #[test]
    #[should_panic(expected = "width must be > 0")]
    fn extend_border_rejects_zero_width() {
        extend_border(&mut [0u8; 100], 0, 8);
    }

    #[test]
    #[should_panic(expected = "height must be > 0")]
    fn extend_border_rejects_zero_height() {
        extend_border(&mut [0u8; 100], 8, 0);
    }

    #[test]
    #[should_panic(expected = "buffer too small")]
    fn extend_border_rejects_small_buffer() {
        // 1×1 image needs at least extended_stride(1) * extended_height(1)
        // = 97 * 97 = 9409 samples; pass less to provoke the panic.
        let mut small = vec![0u8; 100];
        extend_border(&mut small, 1, 1);
    }

    #[test]
    #[should_panic(expected = "image.len()")]
    fn build_rejects_mismatched_image_length() {
        let image = vec![0u8; 10];
        let _ = build_extended_buffer(&image, 4, 4);
    }
}

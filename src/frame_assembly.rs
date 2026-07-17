//! VP6 frame assembly — placing reconstructed 8x8 blocks into image
//! planes (spec §2, §13, §17).
//!
//! The prior rounds landed the complete *per-block* pixel pipeline:
//! §13 entropy decode → §12 scan-to-raster → §15 dequant → §16
//! [`crate::idct_block`] → §17 reconstruction
//! ([`crate::reconstruct_intra_block`] / [`crate::inter`]). Each of
//! those produces a single 8x8 block of reconstructed pixel values.
//! This module is the **frame-assembly** stage that those blocks feed:
//! it owns the per-plane raster image buffers and writes each finished
//! 8x8 block into its correct pixel position inside the plane, so that
//! the per-block decoder output accumulates into a full decoded image.
//!
//! ## Image geometry (spec §2, §9)
//!
//! VP6 is a YUV 4:2:0, macroblock-based codec: "*YUV 4:2:0 image
//! format*" and "*Macro-block (MB) based coding (MB is 16x16 luma plus
//! two 8x8 chroma)*" (§2, page 9). A macroblock therefore carries:
//!
//! * a **16x16** luma (Y) region — four 8x8 luma blocks, and
//! * one **8x8** U chroma block and one **8x8** V chroma block
//!   (4:2:0 sub-sampling halves each chroma dimension relative to
//!   luma, so a 16x16 luma MB maps to 8x8 in each chroma plane).
//!
//! The luma plane's block grid is `HFragments` blocks wide by
//! `VFragments` blocks tall: "*VFragments. The vertical coded height of
//! the frame in 8x8 block units. If image is 240 pixels high,
//! VFragments will be 30*" and "*HFragments … If image is 320 pixels
//! wide, HFragments will be 40*" (§9, page 24). So the luma plane is
//! `HFragments * 8` pixels wide and `VFragments * 8` pixels tall.
//!
//! Because each macroblock contributes exactly one 8x8 block to each
//! chroma plane, the chroma block grid is the **macroblock grid** —
//! `ceil(HFragments / 2)` blocks wide by `ceil(VFragments / 2)` blocks
//! tall — and each chroma plane is that many 8-pixel blocks in each
//! dimension. This module derives the chroma geometry from the
//! macroblock count directly (one chroma block per MB per plane), which
//! is exact regardless of whether `HFragments` / `VFragments` are even
//! or odd.
//!
//! ## Block ordering (spec §13, page 58)
//!
//! "*Blocks are encoded in raster order within a macroblock, and then
//! macroblocks are encoded in raster order within a video image.*" The
//! same page draws the luma block numbering across two horizontally
//! adjacent macroblocks as
//!
//! ```text
//! 0   1     4   5
//! 2   3     6   7
//! ```
//!
//! i.e. within a single macroblock the four luma blocks are in 2x2
//! raster order — top-left `0`, top-right `1`, bottom-left `2`,
//! bottom-right `3` — and the next macroblock to the right continues
//! with `4..=7`. [`MB_LUMA_BLOCK_OFFSETS`] encodes that 2x2 layout as
//! (block-row, block-col) offsets within the macroblock's luma block
//! sub-grid.
//!
//! ## What this stage does and does not do
//!
//! This module is **BoolCoder-independent**: it consumes already-
//! reconstructed 8x8 pixel blocks (the §17 output) and copies them into
//! a plane buffer. Every operation is pure integer index arithmetic and
//! a memcpy-shaped 8x8 write; it never calls `VP6_DecodeBool`, so —
//! like §15/§16/§17/§11/§12/§14 — it advances the decoder without
//! touching the contested §7.3 `Split` formula.
//!
//! It does **not** perform the §11.5 Unrestricted-Motion-Vector border
//! extension (that lives in [`crate::umv`] and is applied to a
//! *reference* plane before inter prediction reads from it); the plane
//! buffers here are the unbordered reconstruction images. Wiring the
//! per-macroblock decode loop (mode decode → MV decode → per-block
//! coefficient decode → reconstruct → assemble) is the per-frame
//! driver's job and is gated upstream on the §7.3 BoolCoder and §10
//! mode-traversal DOCS-GAPs; this round lands the assembly primitive
//! that driver will call once those gaps are resolved.
//!
//! All material in this file is sourced exclusively from
//! `docs/video/vp6/vp6_format.pdf` (On2 Technologies, document version
//! 1.02, August 2006), §2 (page 9), §9 (page 24) and §13 (page 58). No
//! third-party VP6 source has been consulted at any stage.

/// The side length, in pixels, of one DCT block (spec §16: "*8x8 DCT
/// transform*", §2). Each reconstructed block is `BLOCK_DIM x
/// BLOCK_DIM`.
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const BLOCK_DIM: usize = 8;

/// The side length, in pixels, of a macroblock's luma region (spec §2:
/// "*MB is 16x16 luma*").
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const MB_LUMA_DIM: usize = 16;

/// The number of 8x8 luma blocks per macroblock (the 2x2 luma
/// sub-grid: `16 / 8 = 2` per side).
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const MB_LUMA_BLOCKS: usize = 4;

/// The (block-row, block-col) offsets of the four luma blocks within a
/// macroblock's 2x2 luma block sub-grid, in the spec §13 (page 58)
/// raster order `0=TL, 1=TR, 2=BL, 3=BR`:
///
/// ```text
/// 0  1
/// 2  3
/// ```
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const MB_LUMA_BLOCK_OFFSETS: [(usize, usize); MB_LUMA_BLOCKS] =
    [(0, 0), (0, 1), (1, 0), (1, 1)];

/// Errors the frame-assembly stage can surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyError {
    /// A block placement would write outside the plane's pixel
    /// rectangle (block coordinate beyond the plane's block grid).
    OutOfBounds,
}

impl core::fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AssemblyError::OutOfBounds => {
                f.write_str("oxideav-vp6: block placement outside plane bounds")
            }
        }
    }
}

impl std::error::Error for AssemblyError {}

/// A single raster-order image plane (luma or one chroma channel).
///
/// Stores `width * height` `u8` samples row-major. The `stride` equals
/// `width` (no per-row padding) — VP6 reconstruction planes are dense;
/// the §11.5 UMV border (if needed for inter prediction) is applied
/// separately by [`crate::umv::build_extended_buffer`] on a *copy* of
/// the reference plane, leaving this reconstruction buffer unbordered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Plane {
    /// Allocate a `width` x `height` plane, all samples zeroed.
    ///
    /// `width` and `height` are in pixels and need not be multiples of
    /// [`BLOCK_DIM`]; block writes that would overflow the rectangle
    /// are rejected by [`Plane::place_block`] rather than silently
    /// clipped, so a caller assembling a non-block-multiple plane is
    /// responsible for only placing blocks that fit.
    pub fn new(width: usize, height: usize) -> Self {
        Plane {
            width,
            height,
            data: vec![0u8; width * height],
        }
    }

    /// Allocate a plane sized to hold exactly `block_cols` x
    /// `block_rows` 8x8 blocks (so `width = block_cols * 8`,
    /// `height = block_rows * 8`).
    pub fn with_block_grid(block_cols: usize, block_rows: usize) -> Self {
        Plane::new(block_cols * BLOCK_DIM, block_rows * BLOCK_DIM)
    }

    /// Plane width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Plane height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read-only view of the raster-order sample buffer.
    pub fn samples(&self) -> &[u8] {
        &self.data
    }

    /// Mutable view of the raster-order sample buffer.
    pub fn samples_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Read one sample at pixel `(row, col)`. Returns `None` if the
    /// coordinate is outside the plane.
    pub fn sample(&self, row: usize, col: usize) -> Option<u8> {
        if row >= self.height || col >= self.width {
            return None;
        }
        Some(self.data[row * self.width + col])
    }

    /// Write a reconstructed 8x8 `block` (raster order, the §17
    /// reconstruction output) into the plane at *pixel* origin
    /// `(top, left)`.
    ///
    /// `block[r * 8 + c]` is written to plane pixel
    /// `(top + r, left + c)`. Returns [`AssemblyError::OutOfBounds`] if
    /// the 8x8 region would extend past the plane's right or bottom
    /// edge — no partial write is performed in that case.
    pub fn place_block_at_pixel(
        &mut self,
        top: usize,
        left: usize,
        block: &[u8; BLOCK_DIM * BLOCK_DIM],
    ) -> Result<(), AssemblyError> {
        if top + BLOCK_DIM > self.height || left + BLOCK_DIM > self.width {
            return Err(AssemblyError::OutOfBounds);
        }
        for r in 0..BLOCK_DIM {
            let dst = (top + r) * self.width + left;
            let src = r * BLOCK_DIM;
            self.data[dst..dst + BLOCK_DIM].copy_from_slice(&block[src..src + BLOCK_DIM]);
        }
        Ok(())
    }

    /// Write a reconstructed 8x8 `block` at *block-grid* coordinate
    /// `(block_row, block_col)` — i.e. pixel origin
    /// `(block_row * 8, block_col * 8)`. This is the natural addressing
    /// for the §13 raster-order block stream.
    pub fn place_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block: &[u8; BLOCK_DIM * BLOCK_DIM],
    ) -> Result<(), AssemblyError> {
        self.place_block_at_pixel(block_row * BLOCK_DIM, block_col * BLOCK_DIM, block)
    }
}

/// A full decoded VP6 frame: the three YUV 4:2:0 planes plus the
/// macroblock grid geometry that drives assembly.
///
/// The luma plane is `HFragments * 8` x `VFragments * 8` pixels (§9).
/// The chroma planes are sized to the macroblock grid: each macroblock
/// contributes one 8x8 block to each of U and V, so the chroma planes
/// are `mb_cols * 8` x `mb_rows * 8` pixels, where
/// `mb_cols = ceil(HFragments / 2)` and `mb_rows = ceil(VFragments / 2)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Luma (Y) plane.
    pub y: Plane,
    /// U chroma plane.
    pub u: Plane,
    /// V chroma plane.
    pub v: Plane,
    /// Luma block-grid width (`HFragments`).
    pub h_fragments: usize,
    /// Luma block-grid height (`VFragments`).
    pub v_fragments: usize,
}

impl Frame {
    /// Allocate a frame for a `h_fragments` x `v_fragments` luma block
    /// grid (the §9 `HFragments` / `VFragments`).
    ///
    /// Sizes the luma plane to `h_fragments * 8` x `v_fragments * 8`
    /// and each chroma plane to the macroblock grid (`mb_cols * 8` x
    /// `mb_rows * 8`), per the §2 4:2:0 geometry.
    pub fn new(h_fragments: usize, v_fragments: usize) -> Self {
        let mb_cols = mb_cols_for(h_fragments);
        let mb_rows = mb_rows_for(v_fragments);
        Frame {
            y: Plane::with_block_grid(h_fragments, v_fragments),
            u: Plane::with_block_grid(mb_cols, mb_rows),
            v: Plane::with_block_grid(mb_cols, mb_rows),
            h_fragments,
            v_fragments,
        }
    }

    /// The number of macroblocks across the frame (`ceil(HFragments / 2)`).
    pub fn mb_cols(&self) -> usize {
        mb_cols_for(self.h_fragments)
    }

    /// The number of macroblocks down the frame (`ceil(VFragments / 2)`).
    pub fn mb_rows(&self) -> usize {
        mb_rows_for(self.v_fragments)
    }

    /// Place the four reconstructed luma blocks of the macroblock at
    /// grid position `(mb_row, mb_col)` into the luma plane.
    ///
    /// `luma_blocks` is in the §13 (page 58) in-MB raster order
    /// (`0=TL, 1=TR, 2=BL, 3=BR`, per [`MB_LUMA_BLOCK_OFFSETS`]). Each
    /// block lands at luma block-grid coordinate
    /// `(mb_row * 2 + dr, mb_col * 2 + dc)` for its `(dr, dc)` offset.
    ///
    /// Blocks whose 2x2-offset position falls outside the luma block
    /// grid (possible only when `HFragments` / `VFragments` are odd, so
    /// the bottom / right macroblock row/column has fewer than 2 luma
    /// blocks in that dimension) are skipped — the spec codes a partial
    /// macroblock at the frame edge, and only the in-grid luma blocks
    /// exist. Returns [`AssemblyError::OutOfBounds`] only if a block
    /// that *is* inside the grid still fails to fit (which cannot happen
    /// for a correctly-sized plane).
    pub fn place_macroblock_luma(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        luma_blocks: &[[u8; BLOCK_DIM * BLOCK_DIM]; MB_LUMA_BLOCKS],
    ) -> Result<(), AssemblyError> {
        let h_blocks = self.h_fragments;
        let v_blocks = self.v_fragments;
        for (block, &(dr, dc)) in luma_blocks.iter().zip(MB_LUMA_BLOCK_OFFSETS.iter()) {
            let br = mb_row * 2 + dr;
            let bc = mb_col * 2 + dc;
            // Skip edge-overhang luma blocks that the partial-MB grid
            // does not actually contain.
            if br >= v_blocks || bc >= h_blocks {
                continue;
            }
            self.y.place_block(br, bc, block)?;
        }
        Ok(())
    }

    /// Place the U and V chroma blocks of the macroblock at grid
    /// position `(mb_row, mb_col)` into the chroma planes.
    ///
    /// Each macroblock owns exactly one 8x8 block in each chroma plane,
    /// at chroma block-grid coordinate `(mb_row, mb_col)`.
    pub fn place_macroblock_chroma(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        u_block: &[u8; BLOCK_DIM * BLOCK_DIM],
        v_block: &[u8; BLOCK_DIM * BLOCK_DIM],
    ) -> Result<(), AssemblyError> {
        self.u.place_block(mb_row, mb_col, u_block)?;
        self.v.place_block(mb_row, mb_col, v_block)?;
        Ok(())
    }

    /// Place a full macroblock — its four luma blocks plus its U and V
    /// chroma blocks — in one call. Combines [`place_macroblock_luma`]
    /// and [`place_macroblock_chroma`].
    ///
    /// [`place_macroblock_luma`]: Frame::place_macroblock_luma
    /// [`place_macroblock_chroma`]: Frame::place_macroblock_chroma
    pub fn place_macroblock(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        luma_blocks: &[[u8; BLOCK_DIM * BLOCK_DIM]; MB_LUMA_BLOCKS],
        u_block: &[u8; BLOCK_DIM * BLOCK_DIM],
        v_block: &[u8; BLOCK_DIM * BLOCK_DIM],
    ) -> Result<(), AssemblyError> {
        self.place_macroblock_luma(mb_row, mb_col, luma_blocks)?;
        self.place_macroblock_chroma(mb_row, mb_col, u_block, v_block)?;
        Ok(())
    }
}

/// Number of macroblock columns for a luma block-grid width of
/// `h_fragments` blocks: `ceil(h_fragments / 2)` (each MB spans two
/// luma blocks horizontally).
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const fn mb_cols_for(h_fragments: usize) -> usize {
    h_fragments.div_ceil(2)
}

/// Number of macroblock rows for a luma block-grid height of
/// `v_fragments` blocks: `ceil(v_fragments / 2)`.
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const fn mb_rows_for(v_fragments: usize) -> usize {
    v_fragments.div_ceil(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct, recognisable 8x8 block whose every sample encodes
    /// its `(row, col)` so a placement test can verify orientation.
    fn marker_block(tag: u8) -> [u8; BLOCK_DIM * BLOCK_DIM] {
        let mut b = [0u8; BLOCK_DIM * BLOCK_DIM];
        for (i, s) in b.iter_mut().enumerate() {
            // tag in the high nibble, raster index in the low 6 bits.
            *s = (tag.wrapping_shl(6)) | (i as u8 & 0x3F);
        }
        b
    }

    #[test]
    fn constants_match_spec() {
        assert_eq!(BLOCK_DIM, 8, "§16 8x8 DCT");
        assert_eq!(MB_LUMA_DIM, 16, "§2 16x16 luma MB");
        assert_eq!(MB_LUMA_BLOCKS, 4, "16/8 squared = 4 luma blocks");
        // §13 page-58 2x2 raster order: TL, TR, BL, BR.
        assert_eq!(
            MB_LUMA_BLOCK_OFFSETS,
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            "luma block 2x2 raster order"
        );
    }

    #[test]
    fn plane_geometry_from_block_grid() {
        let p = Plane::with_block_grid(40, 30); // §9 worked example: 320x240
        assert_eq!(p.width(), 320);
        assert_eq!(p.height(), 240);
        assert_eq!(p.samples().len(), 320 * 240);
    }

    #[test]
    fn place_block_writes_exact_8x8_region() {
        let mut p = Plane::with_block_grid(4, 4); // 32x32
        let block = marker_block(1);
        p.place_block(1, 2, &block).expect("in bounds");
        // The block occupies pixel rows 8..16, cols 16..24.
        for r in 0..BLOCK_DIM {
            for c in 0..BLOCK_DIM {
                assert_eq!(
                    p.sample(8 + r, 16 + c).unwrap(),
                    block[r * BLOCK_DIM + c],
                    "pixel ({}, {}) mismatch",
                    8 + r,
                    16 + c
                );
            }
        }
        // Everything outside the written block stays zero.
        let mut nonzero_outside = 0;
        for r in 0..32 {
            for c in 0..32 {
                let inside = (8..16).contains(&r) && (16..24).contains(&c);
                if !inside && p.sample(r, c).unwrap() != 0 {
                    nonzero_outside += 1;
                }
            }
        }
        assert_eq!(nonzero_outside, 0, "placement leaked outside its block");
    }

    #[test]
    fn place_block_preserves_raster_orientation() {
        // A block whose samples are their own raster index must land so
        // that plane pixel (top+r, left+c) == block[r*8 + c]. This pins
        // that we never transpose rows/cols.
        let mut p = Plane::with_block_grid(2, 2);
        let mut block = [0u8; BLOCK_DIM * BLOCK_DIM];
        for (i, s) in block.iter_mut().enumerate() {
            *s = i as u8;
        }
        p.place_block_at_pixel(0, 0, &block).unwrap();
        for r in 0..BLOCK_DIM {
            for c in 0..BLOCK_DIM {
                assert_eq!(p.sample(r, c).unwrap(), (r * BLOCK_DIM + c) as u8);
            }
        }
    }

    #[test]
    fn place_block_rejects_out_of_bounds() {
        let mut p = Plane::with_block_grid(2, 2); // 16x16, blocks 0..2 each axis
        let block = marker_block(1);
        assert_eq!(p.place_block(0, 0, &block), Ok(()));
        assert_eq!(p.place_block(1, 1, &block), Ok(()));
        // Block column 2 starts at pixel 16 == width → overflow.
        assert_eq!(p.place_block(0, 2, &block), Err(AssemblyError::OutOfBounds));
        assert_eq!(p.place_block(2, 0, &block), Err(AssemblyError::OutOfBounds));
    }

    #[test]
    fn out_of_bounds_leaves_plane_untouched() {
        let mut p = Plane::with_block_grid(2, 2);
        let block = marker_block(3);
        let before = p.samples().to_vec();
        assert_eq!(p.place_block(5, 5, &block), Err(AssemblyError::OutOfBounds));
        assert_eq!(p.samples(), &before[..], "no partial write on overflow");
    }

    #[test]
    fn frame_geometry_4_2_0_even_fragments() {
        // 320x240: HFragments=40, VFragments=30, both even.
        let f = Frame::new(40, 30);
        assert_eq!((f.y.width(), f.y.height()), (320, 240));
        assert_eq!(f.mb_cols(), 20);
        assert_eq!(f.mb_rows(), 15);
        // Chroma is half each dimension → 160x120.
        assert_eq!((f.u.width(), f.u.height()), (160, 120));
        assert_eq!((f.v.width(), f.v.height()), (160, 120));
    }

    #[test]
    fn frame_geometry_odd_fragments_round_up_mbs() {
        // Odd fragment counts: the MB grid rounds up, so the chroma
        // plane covers the partial edge MB.
        let f = Frame::new(5, 3);
        assert_eq!((f.y.width(), f.y.height()), (40, 24));
        assert_eq!(f.mb_cols(), 3, "ceil(5/2)");
        assert_eq!(f.mb_rows(), 2, "ceil(3/2)");
        assert_eq!((f.u.width(), f.u.height()), (24, 16));
    }

    #[test]
    fn place_macroblock_luma_lands_2x2_in_order() {
        let f_blocks = 4; // 2 MBs wide
        let v_blocks = 4; // 2 MBs tall
        let mut f = Frame::new(f_blocks, v_blocks);
        let blocks = [
            marker_block(0), // TL
            marker_block(1), // TR
            marker_block(2), // BL
            marker_block(3), // BR
        ];
        // Place into MB (1, 1): luma block-grid base (2, 2).
        f.place_macroblock_luma(1, 1, &blocks).unwrap();
        // TL → luma block (2,2); TR → (2,3); BL → (3,2); BR → (3,3).
        let expect = [
            ((2, 2), &blocks[0]),
            ((2, 3), &blocks[1]),
            ((3, 2), &blocks[2]),
            ((3, 3), &blocks[3]),
        ];
        for ((br, bc), blk) in expect {
            let (top, left) = (br * BLOCK_DIM, bc * BLOCK_DIM);
            for r in 0..BLOCK_DIM {
                for c in 0..BLOCK_DIM {
                    assert_eq!(
                        f.y.sample(top + r, left + c).unwrap(),
                        blk[r * BLOCK_DIM + c],
                        "luma block ({br},{bc}) misplaced"
                    );
                }
            }
        }
    }

    #[test]
    fn place_macroblock_chroma_one_block_per_plane() {
        let mut f = Frame::new(4, 4); // 2x2 MBs → 2x2 chroma block grid
        let u = marker_block(1);
        let v = marker_block(2);
        f.place_macroblock_chroma(1, 0, &u, &v).unwrap();
        // U block at chroma block (1,0) → pixel origin (8,0).
        for r in 0..BLOCK_DIM {
            for c in 0..BLOCK_DIM {
                assert_eq!(f.u.sample(8 + r, c).unwrap(), u[r * BLOCK_DIM + c]);
                assert_eq!(f.v.sample(8 + r, c).unwrap(), v[r * BLOCK_DIM + c]);
            }
        }
    }

    #[test]
    fn place_full_macroblock_fills_all_six_blocks() {
        let mut f = Frame::new(2, 2); // exactly one MB
        let luma = [
            marker_block(0),
            marker_block(1),
            marker_block(2),
            marker_block(3),
        ];
        let u = marker_block(1);
        let v = marker_block(2);
        f.place_macroblock(0, 0, &luma, &u, &v).unwrap();
        // All 16x16 luma pixels are now from one of the four blocks
        // (each marker_block has at least one nonzero sample), and both
        // 8x8 chroma blocks are filled.
        // Spot-check the four luma quadrant origins.
        assert_eq!(f.y.sample(0, 0).unwrap(), luma[0][0]); // TL
        assert_eq!(f.y.sample(0, 8).unwrap(), luma[1][0]); // TR
        assert_eq!(f.y.sample(8, 0).unwrap(), luma[2][0]); // BL
        assert_eq!(f.y.sample(8, 8).unwrap(), luma[3][0]); // BR
        assert_eq!(f.u.sample(0, 0).unwrap(), u[0]);
        assert_eq!(f.v.sample(0, 0).unwrap(), v[0]);
    }

    #[test]
    fn odd_fragment_edge_mb_skips_overhang_luma_blocks() {
        // 3x3 luma blocks = ceil 2x2 MB grid. The bottom-right MB (1,1)
        // covers luma blocks (2..4, 2..4), but only (2,2) exists; the
        // TR/BL/BR offsets fall off the 3x3 grid and must be skipped
        // without error.
        let mut f = Frame::new(3, 3);
        let blocks = [
            marker_block(1), // TL → (2,2): in grid
            marker_block(2), // TR → (2,3): off grid
            marker_block(3), // BL → (3,2): off grid
            marker_block(0), // BR → (3,3): off grid
        ];
        f.place_macroblock_luma(1, 1, &blocks)
            .expect("overhang blocks are skipped, not errors");
        // Only the in-grid TL block landed.
        let (top, left) = (2 * BLOCK_DIM, 2 * BLOCK_DIM);
        for r in 0..BLOCK_DIM {
            for c in 0..BLOCK_DIM {
                assert_eq!(
                    f.y.sample(top + r, left + c).unwrap(),
                    blocks[0][r * BLOCK_DIM + c]
                );
            }
        }
    }

    /// End-to-end: drive the real per-block pipeline (dequant → IDCT →
    /// §17.1 intra reconstruct) for a flat DC-only block, then assemble
    /// the result into a frame plane. Confirms the assembly stage
    /// accepts the genuine reconstruction output and lands actual
    /// decoded pixels — not just synthetic markers.
    #[test]
    fn assembles_real_reconstructed_block() {
        use crate::idct::idct_block;
        use crate::reconstruct::intra_block_to_pixels;

        // A DC-only dequantized block reconstructs to a flat mid-ish
        // grey via the §16 IDCT + §17.1 level shift.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 200;
        idct_block(&mut coeffs);
        let pixels = intra_block_to_pixels(&coeffs);
        let expected = pixels[0];
        // §16 IDCT of a DC-only block is flat, so §17.1 output is flat.
        assert!(pixels.iter().all(|&p| p == expected));

        let mut f = Frame::new(2, 2);
        f.place_macroblock_luma(0, 0, &[pixels, pixels, pixels, pixels])
            .unwrap();
        // The entire 16x16 luma region now holds the reconstructed
        // pixel value — actual decoded output, not zero.
        for r in 0..MB_LUMA_DIM {
            for c in 0..MB_LUMA_DIM {
                assert_eq!(f.y.sample(r, c).unwrap(), expected);
            }
        }
        assert_ne!(expected, 0, "DC=200 must reconstruct to a real pixel");
    }
}

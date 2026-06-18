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
//! * [`fetch_prediction_block_clamped`] — the §11.5 edge-clamped form
//!   of the integer prediction fetch. The spec's "duplicate edge values
//!   48 times" rule (§11.5) makes any read from the bordered buffer
//!   equivalent to clamping the read position into the original image's
//!   valid range; this entry point implements that equivalence directly
//!   without materialising the §11.5 border buffer, and remains
//!   well-defined for motion vectors whose magnitude exceeds the
//!   48-sample border the spec mandates (which would index out of bounds
//!   in the bordered path).
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

/// Fetch the 8x8 prediction block for an integer (§17.2/§17.3) motion
/// vector against an **unbordered** reference image, applying the §11.5
/// "duplicate edge values" semantic as edge-clamping of the read
/// position.
///
/// `image` is a flat `width * height` reference reconstruction with **no**
/// §11.5 UMV border applied — i.e. the raw decoder output of the previous
/// or golden frame, sample `(r, c)` at index `r * width + c`. `(top, left)`
/// is the position of the block's top-left sample if the MV were zero
/// (the co-located corner), expressed as integer `(row, col)` in the
/// original image's grid. `(dx, dy)` is the whole-sample motion-vector
/// offset (`(WholeSampleAlignedX, WholeSampleAlignedY)` from §11.4),
/// either component of which is free to push the read past the original
/// image edge. The 64 output samples are written to `pred` in raster
/// order.
///
/// # The §11.5 derivation
///
/// §11.5 defines the unrestricted-motion-vector behaviour by extending
/// every reconstruction buffer by 48 sample points in each direction:
///
/// > The buffers are extended by duplicating the edge values 48 times.
///
/// [`crate::umv::extend_border`] implements that extension verbatim, and
/// [`fetch_prediction_block`] reads from a buffer that already has it
/// applied. The crate-level documentation on
/// [`crate::umv`] records the equivalence the spec sets up:
///
/// > because the extension is built by edge replication, any read from a
/// > sample position inside the `±BORDER_SIZE`-wide extended frame is
/// > equivalent to clamping the read position to the original image's
/// > valid range.
///
/// This function takes the equivalence as primitive: rather than
/// allocate the bordered buffer and copy `(image.len() + border)` bytes
/// of edge-replicated padding per reference frame, it clamps each per-
/// sample source position into the original image's `[0, width)` x
/// `[0, height)` rectangle on the read side. For any MV whose source
/// rectangle lies entirely inside the original image, the output is
/// **bit-identical** to a [`fetch_prediction_block`] call against the
/// §11.5-bordered version of the same image (the in-range read paths
/// agree). For MVs whose source rectangle hangs off an edge, the
/// clamped reads produce the same edge-duplicated samples that the §11.5
/// border-buffer would have served back, regardless of how far past the
/// edge the MV points — including MVs that exceed the 48-sample border
/// the spec mandates (which would index out of bounds in the bordered
/// reader).
///
/// In short, this entry point implements §11.5's "well-defined clamp"
/// semantic directly, without materializing the border. Callers can
/// reach it from the §17.2 zero MV (`dx == dy == 0`, which just becomes
/// a co-located copy) or the §17.3 full-pixel MV (the general integer
/// offset) without preallocating the §11.5-bordered buffer; the result
/// is identical bit-for-bit.
///
/// # Panics
///
/// Panics if `width == 0` or `height == 0` (a degenerate image has no
/// edge to replicate), or if `image.len() < width * height` (truncated
/// buffer; the clamped read would otherwise dereference uninitialised
/// memory).
// Eight parameters is one over the clippy default, but each carries a
// distinct §11.5 / §17 meaning (image + dims + co-located corner +
// MV + output buffer) and bundling any pair would obscure the spec
// mapping. The neighbour `fetch_prediction_block` is the same shape
// minus dims.
#[allow(clippy::too_many_arguments)]
pub fn fetch_prediction_block_clamped(
    image: &[u8],
    width: usize,
    height: usize,
    top: i32,
    left: i32,
    dx: i32,
    dy: i32,
    pred: &mut [u8; 64],
) {
    assert!(
        width > 0,
        "fetch_prediction_block_clamped: width must be > 0"
    );
    assert!(
        height > 0,
        "fetch_prediction_block_clamped: height must be > 0"
    );
    assert!(
        image.len() >= width.checked_mul(height).expect("width * height overflow"),
        "fetch_prediction_block_clamped: image too small ({} < {} = {} * {})",
        image.len(),
        width * height,
        width,
        height
    );

    // The §11.5 edge-clamp upper bounds. The valid sample range in the
    // original image is `[0, width-1]` x `[0, height-1]`.
    let max_col = (width - 1) as i32;
    let max_row = (height - 1) as i32;

    for r in 0..8i32 {
        // Source row before clamping: co-located row + MV-y + block row.
        let unclamped_row = top + dy + r;
        // Clamp into [0, max_row]. The two-sided clamp is the §11.5
        // edge-duplication semantic: row < 0 reads top-row samples;
        // row > max_row reads bottom-row samples.
        let src_row = unclamped_row.clamp(0, max_row) as usize;
        let row_base = src_row * width;
        for c in 0..8i32 {
            let unclamped_col = left + dx + c;
            let src_col = unclamped_col.clamp(0, max_col) as usize;
            pred[(r * 8 + c) as usize] = image[row_base + src_col];
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

/// One reference plane for inter motion compensation: a flat,
/// **unbordered** `width × height` reconstruction (the previous-frame or
/// golden-frame luma or chroma plane). §11.5's UMV border is applied on
/// the read side by [`fetch_prediction_block_clamped`], so the plane
/// itself carries no padding.
#[derive(Debug, Clone, Copy)]
pub struct RefPlane<'a> {
    /// The flat plane samples, `sample(r, c) = data[r * width + c]`.
    pub data: &'a [u8],
    /// Plane width in samples.
    pub width: usize,
    /// Plane height in samples.
    pub height: usize,
}

/// The six reconstructed 8×8 pixel blocks of one motion-compensated
/// macroblock: four luma (raster TL, TR, BL, BR) plus U and V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconstructedMacroblock {
    /// The four luma blocks, raster order (0=TL, 1=TR, 2=BL, 3=BR).
    pub luma: [[u8; 64]; 4],
    /// The U chroma block.
    pub u: [u8; 64],
    /// The V chroma block.
    pub v: [u8; 64],
}

/// Reconstruct one **integer-motion-vector** inter macroblock by
/// motion-compensating each of its six blocks against the reference
/// planes and recombining with the per-block IDCT residual
/// (§17.2 / §17.3).
///
/// This is the integer-MV (whole-pixel-aligned, §17.2 zero and §17.3
/// full-pixel) MB-level glue between MV resolution and §2 raster
/// assembly: it drives the per-block prediction fetch + §17 recombine
/// across the four luma blocks and the two chroma blocks of one
/// macroblock, handling the §11.4 luma-vs-chroma motion-vector shift and
/// the 4:2:0 plane geometry. Sub-pixel motion (§17.4) routes through the
/// [`crate::interp`] bilinear/bicubic filters and is **not** handled
/// here — a fractional MV component is treated by its whole-sample part
/// only, so callers needing sub-pixel accuracy filter the fetched
/// prediction themselves. FourMV macroblocks (per-block luma vectors)
/// are likewise a separate per-block path.
///
/// * `mb_row` / `mb_col` — the macroblock's grid position. The luma
///   plane co-located corner is `(mb_row * 16, mb_col * 16)`; each
///   chroma plane's is `(mb_row * 8, mb_col * 8)` (§2 4:2:0).
/// * `mv` — the MB motion vector in ¼-pel luma units. The luma blocks
///   use `mv >> 2` whole-sample offsets ([`MvShift::Luma`]); the chroma
///   blocks reinterpret the same vector at ⅛-sample precision
///   (`mv >> 3`, [`MvShift::Chroma`]) per §11.4.
/// * `luma_residual` — the four luma blocks' post-IDCT residuals, raster
///   order; `u_residual` / `v_residual` — the chroma residuals.
/// * `ref_y` / `ref_u` / `ref_v` — the reference reconstruction planes.
///
/// Each block's prediction is fetched with
/// [`fetch_prediction_block_clamped`] (the §11.5 read-side edge clamp)
/// and recombined via [`reconstruct_inter_block`].
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_macroblock(
    mb_row: usize,
    mb_col: usize,
    mv_x: i32,
    mv_y: i32,
    luma_residual: &[[i32; 64]; 4],
    u_residual: &[i32; 64],
    v_residual: &[i32; 64],
    ref_y: RefPlane<'_>,
    ref_u: RefPlane<'_>,
    ref_v: RefPlane<'_>,
) -> ReconstructedMacroblock {
    // §11.4 whole-sample offsets per plane.
    let luma_dx = whole_sample_aligned(mv_x, MvShift::Luma);
    let luma_dy = whole_sample_aligned(mv_y, MvShift::Luma);
    let chroma_dx = whole_sample_aligned(mv_x, MvShift::Chroma);
    let chroma_dy = whole_sample_aligned(mv_y, MvShift::Chroma);

    // The four luma block co-located corners within the luma plane.
    const LUMA_OFFSETS: [(i32, i32); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    let luma_top = (mb_row * 16) as i32;
    let luma_left = (mb_col * 16) as i32;

    let mut luma = [[0u8; 64]; 4];
    for (k, &(dr, dc)) in LUMA_OFFSETS.iter().enumerate() {
        let mut pred = [0u8; 64];
        fetch_prediction_block_clamped(
            ref_y.data,
            ref_y.width,
            ref_y.height,
            luma_top + dr,
            luma_left + dc,
            luma_dx,
            luma_dy,
            &mut pred,
        );
        reconstruct_inter_block(&pred, &luma_residual[k], &mut luma[k]);
    }

    // One chroma block per plane at the MB's chroma corner.
    let chroma_top = (mb_row * 8) as i32;
    let chroma_left = (mb_col * 8) as i32;

    let mut u = [0u8; 64];
    let mut pred_u = [0u8; 64];
    fetch_prediction_block_clamped(
        ref_u.data,
        ref_u.width,
        ref_u.height,
        chroma_top,
        chroma_left,
        chroma_dx,
        chroma_dy,
        &mut pred_u,
    );
    reconstruct_inter_block(&pred_u, u_residual, &mut u);

    let mut v = [0u8; 64];
    let mut pred_v = [0u8; 64];
    fetch_prediction_block_clamped(
        ref_v.data,
        ref_v.width,
        ref_v.height,
        chroma_top,
        chroma_left,
        chroma_dx,
        chroma_dy,
        &mut pred_v,
    );
    reconstruct_inter_block(&pred_v, v_residual, &mut v);

    ReconstructedMacroblock { luma, u, v }
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

    // ---- fetch_prediction_block_clamped (§11.5 edge-clamp form) ----

    /// Helper: build a `width * height` test image whose sample value at
    /// `(r, c)` is a deterministic non-trivial function of position, so
    /// the tests can recover *which* sample the read picked up.
    fn ramp_image(width: usize, height: usize) -> Vec<u8> {
        let mut v = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                // Distinct values across both axes; wrap into u8.
                v[r * width + c] = ((r * 7 + c * 3 + 5) & 0xff) as u8;
            }
        }
        v
    }

    /// A zero MV in-range — the clamped fetch reduces to a co-located
    /// 8x8 copy out of the image (no clamping triggers).
    #[test]
    fn clamped_zero_mv_copies_colocated_unclamped() {
        let (w, h) = (32, 32);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Position the block at (top=4, left=4), entirely inside the image.
        fetch_prediction_block_clamped(&img, w, h, 4, 4, 0, 0, &mut pred);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    pred[r * 8 + c],
                    img[(4 + r) * w + (4 + c)],
                    "row {r}, col {c}: zero-MV in-range read should match co-located sample"
                );
            }
        }
    }

    /// A full-pixel positive MV in-range — same equivalence as
    /// [`fetch_full_pixel_positive_offset`] for the unbordered API.
    #[test]
    fn clamped_full_pixel_positive_offset_in_range() {
        let (w, h) = (32, 32);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (4, 4), MV = (+3, +5): all 8x8 reads land inside the image.
        fetch_prediction_block_clamped(&img, w, h, 4, 4, 3, 5, &mut pred);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(pred[r * 8 + c], img[(4 + 5 + r) * w + (4 + 3 + c)]);
            }
        }
    }

    /// MV pushes the read past the **left** edge. The clamped fetch must
    /// substitute the leftmost-column value for every out-of-range
    /// column — the §11.5 edge-duplication semantic.
    #[test]
    fn clamped_left_edge_replicates_leftmost_column() {
        let (w, h) = (32, 32);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (top=4, left=4), MV = (-10, 0): cols 4-10 .. 4-3 .. 1
        // are negative for the first 6 sample columns of the block —
        // clamped to col 0 — and cols (-2..=1) for the last two columns.
        fetch_prediction_block_clamped(&img, w, h, 4, 4, -10, 0, &mut pred);
        for r in 0..8 {
            let src_row = 4 + r;
            for c in 0..8 {
                let src_col_unclamped = 4i32 + (-10) + c as i32;
                let expected_col = src_col_unclamped.max(0) as usize;
                assert_eq!(
                    pred[r * 8 + c],
                    img[src_row * w + expected_col],
                    "row {r}, col {c}: src col {src_col_unclamped} should clamp to {expected_col}"
                );
            }
        }
    }

    /// MV pushes the read past the **right** edge. Must replicate the
    /// rightmost-column value (the §11.5 symmetric case of the left
    /// test).
    #[test]
    fn clamped_right_edge_replicates_rightmost_column() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (top=4, left=12), MV = (+10, 0): cols 12+10 .. 12+17
        // are > 15 from the first sample column onward — clamped to col 15.
        fetch_prediction_block_clamped(&img, w, h, 4, 12, 10, 0, &mut pred);
        for r in 0..8 {
            let src_row = 4 + r;
            for c in 0..8 {
                let src_col_unclamped = 12i32 + 10 + c as i32;
                let expected_col = src_col_unclamped.min((w - 1) as i32) as usize;
                assert_eq!(pred[r * 8 + c], img[src_row * w + expected_col]);
            }
        }
    }

    /// MV pushes the read past the **top** edge. Must replicate the
    /// top-row value.
    #[test]
    fn clamped_top_edge_replicates_top_row() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (top=4, left=4), MV = (0, -10): rows 4-10 .. 4-3 are
        // negative — clamped to row 0.
        fetch_prediction_block_clamped(&img, w, h, 4, 4, 0, -10, &mut pred);
        for r in 0..8 {
            let src_row_unclamped = 4i32 + (-10) + r as i32;
            let expected_row = src_row_unclamped.max(0) as usize;
            for c in 0..8 {
                assert_eq!(pred[r * 8 + c], img[expected_row * w + (4 + c)]);
            }
        }
    }

    /// MV pushes the read past the **bottom** edge. Must replicate the
    /// bottom-row value.
    #[test]
    fn clamped_bottom_edge_replicates_bottom_row() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (top=12, left=4), MV = (0, +10): rows 12+10..12+17
        // are > 15 from the start — clamped to row 15.
        fetch_prediction_block_clamped(&img, w, h, 12, 4, 0, 10, &mut pred);
        for r in 0..8 {
            let src_row_unclamped = 12i32 + 10 + r as i32;
            let expected_row = src_row_unclamped.min((h - 1) as i32) as usize;
            for c in 0..8 {
                assert_eq!(pred[r * 8 + c], img[expected_row * w + (4 + c)]);
            }
        }
    }

    /// MV pushes the read past **both** a horizontal and a vertical
    /// edge — the §11.5 corner case (the four 48x48 corner quadrants
    /// the `umv::extend_border` test suite covers via the bordered
    /// path). The clamped read should serve up the single corner pixel
    /// for every output sample.
    #[test]
    fn clamped_top_left_corner_returns_corner_pixel() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // Block at (0, 0), MV = (-100, -100): every read is far above
        // and far left of the image — clamps to (0, 0) for every sample.
        fetch_prediction_block_clamped(&img, w, h, 0, 0, -100, -100, &mut pred);
        let corner = img[0];
        assert_eq!(pred, [corner; 64]);
    }

    /// The reverse corner: MV pushes past bottom-right. Every sample
    /// should be the `(h-1, w-1)` corner pixel.
    #[test]
    fn clamped_bottom_right_corner_returns_corner_pixel() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        fetch_prediction_block_clamped(
            &img,
            w,
            h,
            (h - 1) as i32,
            (w - 1) as i32,
            100,
            100,
            &mut pred,
        );
        let corner = img[(h - 1) * w + (w - 1)];
        assert_eq!(pred, [corner; 64]);
    }

    /// **The equivalence property.** For an in-range MV against an
    /// image with no §11.5 border applied, the clamped fetch must
    /// produce bit-identical output to `fetch_prediction_block` against
    /// the §11.5-bordered version of the same image. This is the spec's
    /// "edge replication == read clamp" identity, exercised concretely.
    #[test]
    fn clamped_matches_bordered_fetch_for_in_range_mv() {
        let (w, h) = (32, 32);
        let img = ramp_image(w, h);
        // Build the §11.5-bordered version through the umv module.
        let (bordered, ext_stride, _ext_height) = crate::umv::build_extended_buffer(&img, w, h);
        let origin = crate::umv::origin_offset(ext_stride);

        // Sweep a variety of (top, left, dx, dy) combinations that stay
        // entirely inside the image — both readers must agree byte-for-byte.
        let cases = [
            (0, 0, 0, 0),
            (4, 4, 0, 0),
            (8, 8, 3, 5),
            (10, 10, -3, -5),
            (16, 16, 7, 7),
            (20, 4, -1, 2),
        ];
        for (top, left, dx, dy) in cases {
            let mut p_clamped = [0u8; 64];
            fetch_prediction_block_clamped(&img, w, h, top, left, dx, dy, &mut p_clamped);
            let mut p_bordered = [0u8; 64];
            // Translate (top, left) into the bordered buffer's base_pos.
            let base_pos = origin + (top as usize) * ext_stride + (left as usize);
            fetch_prediction_block(&bordered, base_pos, ext_stride, dx, dy, &mut p_bordered);
            assert_eq!(
                p_clamped, p_bordered,
                "in-range case (top={top}, left={left}, dx={dx}, dy={dy}): clamped fetch != bordered fetch"
            );
        }
    }

    /// **The MV-beyond-border equivalence.** For an MV that stays
    /// inside the 48-sample §11.5 border (so the bordered fetch is
    /// still in-bounds) but pushes past the *original* image edge, the
    /// clamped fetch must still agree with the bordered fetch. This
    /// exercises the spec's edge-replication arithmetic on both sides.
    #[test]
    fn clamped_matches_bordered_fetch_for_edge_overhang() {
        let (w, h) = (32, 32);
        let img = ramp_image(w, h);
        let (bordered, ext_stride, _) = crate::umv::build_extended_buffer(&img, w, h);
        let origin = crate::umv::origin_offset(ext_stride);

        // Each case pushes the 8x8 read at least partially off the
        // image but stays well within the 48-sample border so the
        // bordered path is in bounds.
        let cases = [
            (0, 0, -8, 0),    // past left edge
            (0, 0, 0, -8),    // past top edge
            (0, 0, -8, -8),   // top-left corner
            (24, 24, 8, 0),   // past right edge
            (24, 24, 0, 8),   // past bottom edge
            (24, 24, 8, 8),   // bottom-right corner
            (0, 16, -20, -4), // left + top, partly overlapping the image
        ];
        for (top, left, dx, dy) in cases {
            let mut p_clamped = [0u8; 64];
            fetch_prediction_block_clamped(&img, w, h, top, left, dx, dy, &mut p_clamped);
            let mut p_bordered = [0u8; 64];
            let base_pos = origin + (top as usize) * ext_stride + (left as usize);
            fetch_prediction_block(&bordered, base_pos, ext_stride, dx, dy, &mut p_bordered);
            assert_eq!(
                p_clamped, p_bordered,
                "edge-overhang case (top={top}, left={left}, dx={dx}, dy={dy}): clamped fetch != bordered fetch"
            );
        }
    }

    /// MV exceeds the 48-sample §11.5 border — the bordered fetch
    /// would index out of bounds, but the clamped fetch is well-defined
    /// and returns edge-replicated samples for every position.
    #[test]
    fn clamped_well_defined_beyond_umv_border() {
        let (w, h) = (16, 16);
        let img = ramp_image(w, h);
        let mut pred = [0u8; 64];
        // MV = (-200, -200) — 200 samples past the top-left corner,
        // well beyond the 48-sample §11.5 border.
        fetch_prediction_block_clamped(&img, w, h, 0, 0, -200, -200, &mut pred);
        // Every sample should clamp to (0, 0).
        assert_eq!(pred, [img[0]; 64]);
    }

    /// Per-sample independence: each output sample comes from a
    /// distinct clamped source, never spilling between rows or columns.
    #[test]
    fn clamped_per_sample_independence() {
        let (w, h) = (32, 32);
        let mut img = vec![0u8; w * h];
        // Single bright sample at (10, 12); the rest is 0. Verify the
        // clamped fetch picks it up at exactly one output position when
        // the block is positioned to read it.
        img[10 * w + 12] = 200;
        let mut pred = [0u8; 64];
        // Block at (top=8, left=10), MV = (0, 0): reads rows 8..=15, cols 10..=17.
        // (10, 12) maps to output (r=10-8, c=12-10) = (2, 2).
        fetch_prediction_block_clamped(&img, w, h, 8, 10, 0, 0, &mut pred);
        for r in 0..8 {
            for c in 0..8 {
                let expected = if r == 2 && c == 2 { 200 } else { 0 };
                assert_eq!(pred[r * 8 + c], expected, "spurious value at ({r}, {c})");
            }
        }
    }

    /// Degenerate image dimensions panic with a clear message rather
    /// than producing nonsense or out-of-bounds reads.
    #[test]
    #[should_panic(expected = "width must be > 0")]
    fn clamped_zero_width_panics() {
        let mut pred = [0u8; 64];
        fetch_prediction_block_clamped(&[], 0, 16, 0, 0, 0, 0, &mut pred);
    }

    #[test]
    #[should_panic(expected = "height must be > 0")]
    fn clamped_zero_height_panics() {
        let mut pred = [0u8; 64];
        fetch_prediction_block_clamped(&[], 16, 0, 0, 0, 0, 0, &mut pred);
    }

    #[test]
    #[should_panic(expected = "image too small")]
    fn clamped_truncated_image_panics() {
        // Caller claims 16x16 but only supplies 10 bytes.
        let img = vec![0u8; 10];
        let mut pred = [0u8; 64];
        fetch_prediction_block_clamped(&img, 16, 16, 0, 0, 0, 0, &mut pred);
    }

    // -------- reconstruct_inter_macroblock --------

    /// A zero-MV macroblock against a uniform reference plus a zero
    /// residual reproduces the reference exactly: every reconstructed
    /// block is the co-located reference content (here a flat value).
    #[test]
    fn inter_mb_zero_mv_zero_residual_copies_reference() {
        // A 16x16 luma plane filled with 200, 8x8 chroma planes with 50.
        let ref_y = vec![200u8; 16 * 16];
        let ref_u = vec![50u8; 8 * 8];
        let ref_v = vec![60u8; 8 * 8];
        let zero = [0i32; 64];
        let mb = reconstruct_inter_macroblock(
            0,
            0,
            0,
            0,
            &[zero; 4],
            &zero,
            &zero,
            RefPlane {
                data: &ref_y,
                width: 16,
                height: 16,
            },
            RefPlane {
                data: &ref_u,
                width: 8,
                height: 8,
            },
            RefPlane {
                data: &ref_v,
                width: 8,
                height: 8,
            },
        );
        for block in mb.luma {
            assert!(block.iter().all(|&p| p == 200));
        }
        assert!(mb.u.iter().all(|&p| p == 50));
        assert!(mb.v.iter().all(|&p| p == 60));
    }

    /// A constant residual offsets the copied prediction by that amount,
    /// clipped to 0..=255 (§17.2 recombine).
    #[test]
    fn inter_mb_residual_offsets_prediction() {
        let ref_y = vec![100u8; 16 * 16];
        let ref_u = vec![100u8; 8 * 8];
        let ref_v = vec![100u8; 8 * 8];
        let plus10 = [10i32; 64];
        let zero = [0i32; 64];
        let mb = reconstruct_inter_macroblock(
            0,
            0,
            0,
            0,
            &[plus10; 4],
            &zero,
            &zero,
            RefPlane {
                data: &ref_y,
                width: 16,
                height: 16,
            },
            RefPlane {
                data: &ref_u,
                width: 8,
                height: 8,
            },
            RefPlane {
                data: &ref_v,
                width: 8,
                height: 8,
            },
        );
        for block in mb.luma {
            assert!(block.iter().all(|&p| p == 110), "100 + 10 residual");
        }
        // Chroma had a zero residual → unchanged.
        assert!(mb.u.iter().all(|&p| p == 100));
    }

    /// A full-pixel (¼-pel multiple of 4) horizontal MV shifts the luma
    /// prediction source by whole samples. Build a reference whose
    /// columns ramp 0,1,2,… so a known dx produces a known shifted read.
    #[test]
    fn inter_mb_full_pixel_mv_shifts_source() {
        // 32x16 luma plane: sample(r,c) = c (column ramp, 0..=31).
        let width = 32usize;
        let height = 16usize;
        let mut ref_y = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                ref_y[r * width + c] = c as u8;
            }
        }
        let ref_c = vec![0u8; 16 * 8];
        let zero = [0i32; 64];
        // MV x = 4 quarter-pels = +1 whole sample (luma >> 2). MB at
        // (row 0, col 0) so luma corner is (0, 0); the TL block reads
        // columns 1..=8 → values 1..=8 across each row.
        let mb = reconstruct_inter_macroblock(
            0,
            0,
            4,
            0,
            &[zero; 4],
            &zero,
            &zero,
            RefPlane {
                data: &ref_y,
                width,
                height,
            },
            RefPlane {
                data: &ref_c,
                width: 16,
                height: 8,
            },
            RefPlane {
                data: &ref_c,
                width: 16,
                height: 8,
            },
        );
        // TL luma block row 0: source columns 1..=8 → 1,2,..,8.
        for c in 0..8 {
            assert_eq!(mb.luma[0][c], (c as u8) + 1, "shifted column read");
        }
    }

    /// The §11.5 read-side clamp: an MV that points off the left edge
    /// duplicates the edge column rather than reading out of bounds.
    #[test]
    fn inter_mb_negative_mv_clamps_to_edge() {
        let width = 16usize;
        let height = 16usize;
        let mut ref_y = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                ref_y[r * width + c] = c as u8;
            }
        }
        let ref_c = vec![0u8; 8 * 8];
        let zero = [0i32; 64];
        // MV x = -40 quarter-pels = -10 whole samples; the TL block at
        // corner (0,0) reads columns -10..=-3, all clamped to column 0
        // (value 0).
        let mb = reconstruct_inter_macroblock(
            0,
            0,
            -40,
            0,
            &[zero; 4],
            &zero,
            &zero,
            RefPlane {
                data: &ref_y,
                width,
                height,
            },
            RefPlane {
                data: &ref_c,
                width: 8,
                height: 8,
            },
            RefPlane {
                data: &ref_c,
                width: 8,
                height: 8,
            },
        );
        assert!(
            mb.luma[0].iter().all(|&p| p == 0),
            "off-left-edge read clamps to column 0"
        );
    }
}

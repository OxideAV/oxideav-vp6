//! VP6 output scaling — the §9 `ScalingMode` / `Output*Fragments` surface
//! (spec §9 Table 2, page 24) **and its application**: the post-decode
//! resample/placement that turns a coded-resolution reconstruction into
//! the output-resolution frame the header signals.
//!
//! ## What the staged spec fixes
//!
//! A VP6 frame is decoded at its **coded** resolution and "*may be encoded
//! at a different resolution to the eventual size that it is presented on
//! output from the decoder*" (§9). §2 lists "*Scaling on output after
//! decode*" as a codec feature, and §11.5 notes the reconstruction-buffer
//! borders exist "*to support this feature and the playback scaling
//! features of the codec*" — i.e. scaling is a **post-reconstruction
//! presentation step**. The §4 reference buffers a following frame
//! predicts from are the *coded-resolution* reconstructions; the scaled
//! output never re-enters the prediction loop, so no choice made here can
//! affect the bit-exactness of the decode itself.
//!
//! The header carries the output geometry as `OutputHFragments` /
//! `OutputVFragments` plus a two-bit `ScalingMode` naming one of four
//! strategies ("*MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER*",
//! §9 page 24, in that listing order). Per the fixture-arbitrated erratum
//! #338 the four `*Fragments` fields are transmitted in **16-px
//! macroblock units** (the printed "8x8 block units" description and its
//! worked examples are wrong — the conformant 864x480 stream transmits
//! 54x30, and asserts the same units for the `Output*` pair).
//!
//! ## What the staged spec leaves open (and how this module fills it)
//!
//! The spec names the four modes but prints **no per-mode pixel-mapping
//! algorithm** and **no resampling kernel**. Because scaling is
//! presentation-only (above), those choices are decoder-discretionary;
//! this module makes them explicitly and documents each:
//!
//! * **Mode geometry (name-implied):** `SCALE_TO_FIT` stretches each axis
//!   independently to fill the output rectangle; `MAINTAIN_ASPECT_RATIO`
//!   performs the largest aspect-preserving fit inside the output
//!   rectangle (degenerating to the full-rectangle stretch when the
//!   aspect ratios already match — the plain "coded at quarter size,
//!   present at full size" case), centred, with the remainder filled
//!   neutral; `CENTER` places the coded image unscaled and centred
//!   (padding or cropping each axis as the output is larger or smaller).
//!   `OTHER` has no spec-supplied semantics at all and is applied as the
//!   identity (the coded frame is returned unchanged) — a docs-gap.
//! * **Resampling kernel (implementation-defined):** a separable 2-tap
//!   linear interpolator on the centre-aligned sample grid, fixed-point
//!   (Q8 per axis), edge-clamped (echoing the §11.5 edge-duplication
//!   convention for out-of-frame reads). It is exact for equal sizes and
//!   on constant regions. The vendor's own kernel is not described by the
//!   staged documents; matching it bit-for-bit would need a scaled
//!   fixture + oracle, but is *not* required for bitstream conformance.
//! * **Rounding ties (implementation-defined):** aspect-fit rectangle
//!   dimensions round to the nearest even pixel count and placement
//!   offsets round down to even, so every chroma rectangle is exactly
//!   half its luma rectangle; letterbox/pad areas fill with Y = 0,
//!   U = V = 128 (neutral).
//!
//! All field semantics are sourced from `docs/video/vp6/vp6_format.pdf`
//! §9 / §2 / §11.5 and the staged erratum #338; no external library code
//! was consulted.

use crate::frame_assembly::{Frame, Plane};

/// Pixels per 8x8 fragment edge — the §2 transform block size.
pub const FRAGMENT_DIM: u32 = 8;

/// Pixels per macroblock edge (§2: a macroblock is 16x16 luma). The §9
/// `*Fragments` header fields count these (erratum #338).
pub const MACROBLOCK_DIM: u32 = 16;

/// The §9 `ScalingMode` field — how the decoded frame is scaled to the
/// output resolution (Table 2, `ScalingMode b(2)`).
///
/// The spec lists the four modes in the order
/// "*MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER*" (§9, page 24)
/// for the two-bit field. That listing order is the only ordering the
/// document supplies, so the discriminants follow it (`0..=3`).
///
/// The per-mode placement geometry applied by this crate is the
/// name-implied one (see the module docs); the staged doc itself names
/// the modes without an algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScalingMode {
    /// `0` — `MAINTAIN_ASPECT_RATIO`. First-listed §9 mode: the largest
    /// aspect-preserving fit inside the output rectangle, centred.
    MaintainAspectRatio = 0,
    /// `1` — `SCALE_TO_FIT`. Second-listed §9 mode: stretch each axis
    /// independently to fill the output rectangle.
    ScaleToFit = 1,
    /// `2` — `CENTER`. Third-listed §9 mode: place the coded image
    /// unscaled, centred in the output rectangle.
    Center = 2,
    /// `3` — `OTHER`. Fourth-listed §9 mode; semantics unspecified by
    /// the staged doc (applied as the identity — docs-gap).
    Other = 3,
}

impl ScalingMode {
    /// Map a decoded two-bit `ScalingMode` field (`0..=3`) to its typed
    /// mode, following the §9 listing order. Returns `None` for any value
    /// outside `0..=3` (a `b(2)` read can only produce `0..=3`, so a
    /// larger value indicates a caller bug rather than malformed input).
    pub fn from_b2(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::MaintainAspectRatio),
            1 => Some(Self::ScaleToFit),
            2 => Some(Self::Center),
            3 => Some(Self::Other),
            _ => None,
        }
    }

    /// The canonical `0..=3` index — the inverse of [`ScalingMode::from_b2`].
    pub fn index(self) -> u8 {
        self as u8
    }
}

/// A frame geometry in the §9 header's transmitted units — **16-px
/// macroblocks** per axis (erratum #338; the printed "8x8 block units"
/// description is wrong) — shared by the coded (`HFragments` /
/// `VFragments`) and output (`OutputHFragments` / `OutputVFragments`)
/// descriptions.
///
/// The luma plane is `mb_cols * 16` pixels wide and `mb_rows * 16` tall;
/// the 4:2:0 chroma planes are `mb_cols * 8` by `mb_rows * 8` (§2: one
/// 8x8 chroma block per macroblock). The 8x8 luma block grid the §13/§14
/// walks use is `2 * mb_cols` by `2 * mb_rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameGeometry {
    /// Horizontal macroblock count — the wire `HFragments` (coded) or
    /// `OutputHFragments` (output) field value.
    pub mb_cols: u32,
    /// Vertical macroblock count — the wire `VFragments` (coded) or
    /// `OutputVFragments` (output) field value.
    pub mb_rows: u32,
}

impl FrameGeometry {
    /// Construct a geometry from a `(mb_cols, mb_rows)` macroblock pair.
    pub fn new(mb_cols: u32, mb_rows: u32) -> Self {
        Self { mb_cols, mb_rows }
    }

    /// Construct from the raw §9 wire fields (`HFragments`-style column
    /// count, `VFragments`-style row count) — macroblock units per
    /// erratum #338.
    pub fn from_wire(h_fragments_field: u8, v_fragments_field: u8) -> Self {
        Self::new(h_fragments_field as u32, v_fragments_field as u32)
    }

    /// The geometry of an assembled [`Frame`], whose
    /// `h_fragments`/`v_fragments` are in this crate's 8x8 luma-block
    /// units (each macroblock spans 2x2 luma blocks; odd block counts
    /// round up to the covering macroblock grid).
    pub fn of_frame(frame: &Frame) -> Self {
        Self::new(
            (frame.h_fragments as u32).div_ceil(2),
            (frame.v_fragments as u32).div_ceil(2),
        )
    }

    /// The luma width in pixels: `mb_cols * 16`.
    pub fn luma_width(self) -> u32 {
        self.mb_cols * MACROBLOCK_DIM
    }

    /// The luma height in pixels: `mb_rows * 16`.
    pub fn luma_height(self) -> u32 {
        self.mb_rows * MACROBLOCK_DIM
    }

    /// The 4:2:0 chroma width in pixels: `mb_cols * 8`.
    pub fn chroma_width(self) -> u32 {
        self.mb_cols * FRAGMENT_DIM
    }

    /// The 4:2:0 chroma height in pixels: `mb_rows * 8`.
    pub fn chroma_height(self) -> u32 {
        self.mb_rows * FRAGMENT_DIM
    }

    /// The 8x8 luma block-grid width: `2 * mb_cols` (the `h_fragments`
    /// this crate's frame drivers consume).
    pub fn block_cols(self) -> u32 {
        2 * self.mb_cols
    }

    /// The 8x8 luma block-grid height: `2 * mb_rows`.
    pub fn block_rows(self) -> u32 {
        2 * self.mb_rows
    }

    /// True when either axis is zero macroblocks — a degenerate geometry
    /// no real bitstream describes (treated as "no scaling" throughout).
    pub fn is_degenerate(self) -> bool {
        self.mb_cols == 0 || self.mb_rows == 0
    }
}

/// The §9 output-scaling description: the desired output geometry plus the
/// [`ScalingMode`] that maps the coded frame onto it.
///
/// Pairs `OutputHFragments` / `OutputVFragments` (Table 2, macroblock
/// units per erratum #338) with the `ScalingMode` field. The coded
/// geometry that feeds the scale lives separately;
/// [`OutputScaling::is_identity`] reports whether a given coded geometry
/// needs any scaling at all, and [`OutputScaling::plan`] derives the
/// placement geometry [`apply_output_scaling`] executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputScaling {
    /// The output geometry (`OutputHFragments` / `OutputVFragments`),
    /// macroblock units.
    pub output: FrameGeometry,
    /// How to scale the coded frame onto the output rectangle.
    pub mode: ScalingMode,
}

impl OutputScaling {
    /// Construct from an output geometry and a scaling mode.
    pub fn new(output: FrameGeometry, mode: ScalingMode) -> Self {
        Self { output, mode }
    }

    /// True when the supplied coded geometry already matches the output
    /// geometry, so no resampling is required regardless of `mode`.
    ///
    /// (When coded == output the choice of mode is moot: every mode maps
    /// an N-macroblock image onto an N-macroblock output as the
    /// identity.)
    pub fn is_identity(self, coded: FrameGeometry) -> bool {
        coded == self.output
    }

    /// Derive the placement geometry for scaling a `coded`-geometry frame
    /// onto this output description. All rectangle dimensions and offsets
    /// are **luma pixels** and are even, so the corresponding chroma
    /// rectangle is exactly the luma one halved.
    ///
    /// Degenerate geometries (zero macroblocks on either axis, coded or
    /// output) plan as [`ScalingPlan::Identity`] — no real bitstream
    /// describes them, and the identity is the safe non-panicking answer.
    pub fn plan(self, coded: FrameGeometry) -> ScalingPlan {
        if coded.is_degenerate() || self.output.is_degenerate() || self.is_identity(coded) {
            return ScalingPlan::Identity;
        }
        let (cw, ch) = (coded.luma_width() as u64, coded.luma_height() as u64);
        let (ow, oh) = (
            self.output.luma_width() as u64,
            self.output.luma_height() as u64,
        );
        match self.mode {
            ScalingMode::ScaleToFit => ScalingPlan::Stretch,
            ScalingMode::MaintainAspectRatio => {
                if cw * oh == ch * ow {
                    // Aspect ratios already match: the aspect-preserving
                    // fit *is* the full-rectangle stretch.
                    return ScalingPlan::Stretch;
                }
                // Largest aspect-preserving fit inside (ow, oh):
                // width-limited when ow/cw <= oh/ch.
                let (fw, fh) = if ow * ch <= oh * cw {
                    (ow, round_even(ch * ow, cw).clamp(2, oh))
                } else {
                    (round_even(cw * oh, ch).clamp(2, ow), oh)
                };
                ScalingPlan::AspectFit {
                    luma_width: fw as u32,
                    luma_height: fh as u32,
                    luma_left: (((ow - fw) / 2) & !1) as u32,
                    luma_top: (((oh - fh) / 2) & !1) as u32,
                }
            }
            ScalingMode::Center => {
                let copy_w = cw.min(ow);
                let copy_h = ch.min(oh);
                ScalingPlan::Center {
                    luma_copy_width: copy_w as u32,
                    luma_copy_height: copy_h as u32,
                    luma_src_left: (((cw - copy_w) / 2) & !1) as u32,
                    luma_src_top: (((ch - copy_h) / 2) & !1) as u32,
                    luma_dst_left: (((ow - copy_w) / 2) & !1) as u32,
                    luma_dst_top: (((oh - copy_h) / 2) & !1) as u32,
                }
            }
            ScalingMode::Other => ScalingPlan::Unspecified,
        }
    }
}

/// `round(num / den)` snapped to the nearest **even** value
/// (`round(num / (2·den)) * 2`) — keeps aspect-fit luma dimensions even
/// so the chroma rectangle is exactly half.
fn round_even(num: u64, den: u64) -> u64 {
    ((num + den) / (2 * den)) * 2
}

/// The placement geometry [`OutputScaling::plan`] derives for one
/// coded→output mapping. All dimensions/offsets are luma pixels (even by
/// construction); chroma uses exactly half of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingPlan {
    /// Coded and output geometry match (or one is degenerate): emit the
    /// decoded frame unchanged.
    Identity,
    /// `SCALE_TO_FIT` (or an aspect-matching `MAINTAIN_ASPECT_RATIO`):
    /// resample each axis independently onto the full output rectangle.
    Stretch,
    /// `MAINTAIN_ASPECT_RATIO` with differing aspect ratios: resample
    /// onto a `luma_width x luma_height` rectangle placed at
    /// `(luma_left, luma_top)` in the output canvas; the remainder is
    /// neutral fill.
    AspectFit {
        /// Fitted rectangle width (luma pixels, even).
        luma_width: u32,
        /// Fitted rectangle height (luma pixels, even).
        luma_height: u32,
        /// Left placement offset in the output canvas (even).
        luma_left: u32,
        /// Top placement offset in the output canvas (even).
        luma_top: u32,
    },
    /// `CENTER`: copy an unscaled `luma_copy_width x luma_copy_height`
    /// window from the coded frame at `(luma_src_left, luma_src_top)`
    /// to `(luma_dst_left, luma_dst_top)` in the output canvas (the
    /// window is the full coded frame when the output is larger — pad —
    /// and a centred crop when it is smaller).
    Center {
        /// Copied window width (luma pixels, even).
        luma_copy_width: u32,
        /// Copied window height (luma pixels, even).
        luma_copy_height: u32,
        /// Window origin in the coded frame (even).
        luma_src_left: u32,
        /// Window origin in the coded frame (even).
        luma_src_top: u32,
        /// Window origin in the output canvas (even).
        luma_dst_left: u32,
        /// Window origin in the output canvas (even).
        luma_dst_top: u32,
    },
    /// `OTHER`: no spec-supplied semantics (docs-gap) — applied as the
    /// identity.
    Unspecified,
}

/// Resample a `src_w x src_h` plane to `dst_w x dst_h` with the
/// implementation-defined kernel documented at module level: separable
/// 2-tap linear interpolation on the centre-aligned sample grid
/// (`src_pos = (dst_pos + 0.5) * src/dst - 0.5`), Q8 fixed-point weights,
/// edge-clamped taps, round-to-nearest output.
///
/// Properties (pinned by tests): equal sizes copy exactly; constant
/// inputs are preserved exactly at any size; a linear ramp upsamples to
/// within 1 LSB of the ideal linear interpolant.
///
/// `src.len()` must be at least `src_w * src_h`; any zero dimension
/// yields an all-zero output of the requested size.
pub fn resample_plane(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }
    let x_taps = axis_taps(src_w, dst_w);
    let y_taps = axis_taps(src_h, dst_h);
    for (dy, &(y0, y1, fy)) in y_taps.iter().enumerate() {
        let row0 = y0 * src_w;
        let row1 = y1 * src_w;
        let wy1 = fy;
        let wy0 = 256 - fy;
        for (dx, &(x0, x1, fx)) in x_taps.iter().enumerate() {
            let wx1 = fx;
            let wx0 = 256 - fx;
            let a = src[row0 + x0] as u32;
            let b = src[row0 + x1] as u32;
            let c = src[row1 + x0] as u32;
            let d = src[row1 + x1] as u32;
            let top = a * wx0 + b * wx1;
            let bot = c * wx0 + d * wx1;
            let v = (top * wy0 + bot * wy1 + (1 << 15)) >> 16;
            out[dy * dst_w + dx] = v.min(255) as u8;
        }
    }
    out
}

/// Per-destination-index `(tap0, tap1, Q8 fraction)` for one axis of the
/// centre-aligned linear resample, edge-clamped at both ends.
fn axis_taps(src_len: usize, dst_len: usize) -> Vec<(usize, usize, u32)> {
    let mut taps = Vec::with_capacity(dst_len);
    for d in 0..dst_len {
        // Q8 centre-aligned source position:
        // pos = (d + 0.5) * src/dst - 0.5, scaled by 256.
        let pos = ((2 * d as i64 + 1) * src_len as i64 * 256) / (2 * dst_len as i64) - 128;
        if pos <= 0 {
            taps.push((0, 0, 0));
            continue;
        }
        let i0 = (pos >> 8) as usize;
        if i0 >= src_len - 1 {
            taps.push((src_len - 1, src_len - 1, 0));
            continue;
        }
        taps.push((i0, i0 + 1, (pos & 255) as u32));
    }
    taps
}

/// Resample a whole 4:2:0 [`Frame`] to the destination geometry
/// (luma to `dst.luma_*`, each chroma plane to `dst.chroma_*`) with
/// [`resample_plane`].
///
/// This is both the decoder-side full-rectangle upscale
/// ([`ScalingPlan::Stretch`]) and the encoder-side **downsample** used to
/// code a frame at reduced resolution before signalling the original
/// size through the §9 `Output*Fragments` / `ScalingMode` fields (at
/// 2:1 the centre-aligned kernel degenerates to the 2x2 box average).
pub fn resample_frame(frame: &Frame, dst: FrameGeometry) -> Frame {
    let mut out = new_fill_frame(dst);
    let (lw, lh) = (dst.luma_width() as usize, dst.luma_height() as usize);
    let (cw, ch) = (dst.chroma_width() as usize, dst.chroma_height() as usize);
    out.y.samples_mut().copy_from_slice(&resample_plane(
        frame.y.samples(),
        frame.y.width(),
        frame.y.height(),
        lw,
        lh,
    ));
    out.u.samples_mut().copy_from_slice(&resample_plane(
        frame.u.samples(),
        frame.u.width(),
        frame.u.height(),
        cw,
        ch,
    ));
    out.v.samples_mut().copy_from_slice(&resample_plane(
        frame.v.samples(),
        frame.v.width(),
        frame.v.height(),
        cw,
        ch,
    ));
    out
}

/// Allocate an output-geometry [`Frame`] pre-filled with the neutral
/// letterbox colour (Y = 0, U = V = 128).
fn new_fill_frame(geom: FrameGeometry) -> Frame {
    let mut f = Frame::new(geom.block_cols() as usize, geom.block_rows() as usize);
    f.u.samples_mut().fill(128);
    f.v.samples_mut().fill(128);
    f
}

/// Copy a `w x h` pixel window from `src` at `src_origin = (left, top)`
/// into `dst` at `dst_origin = (left, top)`. All coordinates must be in
/// range.
fn blit_window(
    dst: &mut Plane,
    dst_origin: (usize, usize),
    src: &Plane,
    src_origin: (usize, usize),
    w: usize,
    h: usize,
) {
    let (dst_left, dst_top) = dst_origin;
    let (src_left, src_top) = src_origin;
    let dst_w = dst.width();
    let src_w = src.width();
    for r in 0..h {
        let s = (src_top + r) * src_w + src_left;
        let d = (dst_top + r) * dst_w + dst_left;
        let row = src.samples()[s..s + w].to_vec();
        dst.samples_mut()[d..d + w].copy_from_slice(&row);
    }
}

/// Write `data` (a tightly-packed `w x h` buffer) into `dst` at
/// `(left, top)`.
fn blit_buffer(dst: &mut Plane, left: usize, top: usize, data: &[u8], w: usize, h: usize) {
    let dst_w = dst.width();
    for r in 0..h {
        let d = (top + r) * dst_w + left;
        dst.samples_mut()[d..d + w].copy_from_slice(&data[r * w..r * w + w]);
    }
}

/// Apply the §9 output scaling to a decoded coded-resolution frame,
/// producing the output-geometry frame the header signals.
///
/// Executes the [`OutputScaling::plan`] placement (see the module docs
/// for exactly which parts are spec-fixed, name-implied, and
/// implementation-defined). Identity plans — including `OTHER`
/// (unspecified — docs-gap) and degenerate geometries — return the frame
/// unchanged.
///
/// `frame` must be macroblock-aligned (even `h_fragments` /
/// `v_fragments` — the only shape real §9 bitstreams describe per
/// erratum #338, and the only shape this crate's drivers produce); a
/// non-MB-aligned frame is returned unchanged.
pub fn apply_output_scaling(frame: &Frame, scaling: OutputScaling) -> Frame {
    if frame.h_fragments % 2 != 0 || frame.v_fragments % 2 != 0 {
        return frame.clone();
    }
    let coded = FrameGeometry::of_frame(frame);
    match scaling.plan(coded) {
        ScalingPlan::Identity | ScalingPlan::Unspecified => frame.clone(),
        ScalingPlan::Stretch => resample_frame(frame, scaling.output),
        ScalingPlan::AspectFit {
            luma_width,
            luma_height,
            luma_left,
            luma_top,
        } => {
            let mut out = new_fill_frame(scaling.output);
            let (lw, lh) = (luma_width as usize, luma_height as usize);
            let (ll, lt) = (luma_left as usize, luma_top as usize);
            let y = resample_plane(frame.y.samples(), frame.y.width(), frame.y.height(), lw, lh);
            blit_buffer(&mut out.y, ll, lt, &y, lw, lh);
            let (cw, ch) = (lw / 2, lh / 2);
            let (cl, ct) = (ll / 2, lt / 2);
            let u = resample_plane(frame.u.samples(), frame.u.width(), frame.u.height(), cw, ch);
            blit_buffer(&mut out.u, cl, ct, &u, cw, ch);
            let v = resample_plane(frame.v.samples(), frame.v.width(), frame.v.height(), cw, ch);
            blit_buffer(&mut out.v, cl, ct, &v, cw, ch);
            out
        }
        ScalingPlan::Center {
            luma_copy_width,
            luma_copy_height,
            luma_src_left,
            luma_src_top,
            luma_dst_left,
            luma_dst_top,
        } => {
            let mut out = new_fill_frame(scaling.output);
            let (w, h) = (luma_copy_width as usize, luma_copy_height as usize);
            let dst_luma = (luma_dst_left as usize, luma_dst_top as usize);
            let src_luma = (luma_src_left as usize, luma_src_top as usize);
            blit_window(&mut out.y, dst_luma, &frame.y, src_luma, w, h);
            let (cw, ch) = (w / 2, h / 2);
            let dst_chroma = (dst_luma.0 / 2, dst_luma.1 / 2);
            let src_chroma = (src_luma.0 / 2, src_luma.1 / 2);
            blit_window(&mut out.u, dst_chroma, &frame.u, src_chroma, cw, ch);
            blit_window(&mut out.v, dst_chroma, &frame.v, src_chroma, cw, ch);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_constants() {
        // §2 / §16 transform block size; §2 macroblock size.
        assert_eq!(FRAGMENT_DIM, 8);
        assert_eq!(MACROBLOCK_DIM, 16);
    }

    #[test]
    fn scaling_mode_listing_order_discriminants() {
        // §9 lists "MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER".
        assert_eq!(ScalingMode::MaintainAspectRatio.index(), 0);
        assert_eq!(ScalingMode::ScaleToFit.index(), 1);
        assert_eq!(ScalingMode::Center.index(), 2);
        assert_eq!(ScalingMode::Other.index(), 3);
    }

    #[test]
    fn scaling_mode_from_b2_round_trip() {
        for v in 0u8..=3 {
            let mode = ScalingMode::from_b2(v).expect("0..=3 are all valid");
            assert_eq!(mode.index(), v);
        }
    }

    #[test]
    fn scaling_mode_from_b2_rejects_out_of_range() {
        for v in 4u8..=255 {
            assert_eq!(ScalingMode::from_b2(v), None);
        }
    }

    #[test]
    fn fixture_geometry_macroblock_units() {
        // Erratum #338: the conformant 864x480 stream transmits
        // HFragments = 54, VFragments = 30 — macroblock counts.
        let g = FrameGeometry::from_wire(54, 30);
        assert_eq!(g.luma_width(), 864);
        assert_eq!(g.luma_height(), 480);
        assert_eq!(g.chroma_width(), 432);
        assert_eq!(g.chroma_height(), 240);
        assert_eq!(g.block_cols(), 108);
        assert_eq!(g.block_rows(), 60);
    }

    #[test]
    fn of_frame_recovers_mb_geometry() {
        let f = Frame::new(8, 6); // 8x6 luma blocks = 4x3 macroblocks
        assert_eq!(FrameGeometry::of_frame(&f), FrameGeometry::new(4, 3));
    }

    #[test]
    fn output_scaling_identity_when_coded_matches_output() {
        let coded = FrameGeometry::new(40, 30);
        let out = OutputScaling::new(coded, ScalingMode::ScaleToFit);
        assert!(out.is_identity(coded));
        assert_eq!(out.plan(coded), ScalingPlan::Identity);
    }

    #[test]
    fn plan_stretch_for_scale_to_fit() {
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::ScaleToFit);
        assert_eq!(out.plan(coded), ScalingPlan::Stretch);
    }

    #[test]
    fn plan_aspect_match_degenerates_to_stretch() {
        // 2x2 -> 4x4 preserves aspect: MAINTAIN_ASPECT_RATIO is the full
        // stretch (the plain quarter-res -> full-res presentation case).
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::MaintainAspectRatio);
        assert_eq!(out.plan(coded), ScalingPlan::Stretch);
    }

    #[test]
    fn plan_aspect_fit_letterboxes_wide_output() {
        // 2x2 MB (32x32) into 4x2 MB (64x32): height-limited, fitted
        // 32x32 centred horizontally at x = 16.
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(4, 2), ScalingMode::MaintainAspectRatio);
        assert_eq!(
            out.plan(coded),
            ScalingPlan::AspectFit {
                luma_width: 32,
                luma_height: 32,
                luma_left: 16,
                luma_top: 0,
            }
        );
    }

    #[test]
    fn plan_aspect_fit_letterboxes_tall_output() {
        // 4x2 MB (64x32) into 2x2 MB (32x32): width-limited, fitted
        // 32x16 centred vertically at y = 8.
        let coded = FrameGeometry::new(4, 2);
        let out = OutputScaling::new(FrameGeometry::new(2, 2), ScalingMode::MaintainAspectRatio);
        assert_eq!(
            out.plan(coded),
            ScalingPlan::AspectFit {
                luma_width: 32,
                luma_height: 16,
                luma_left: 0,
                luma_top: 8,
            }
        );
    }

    #[test]
    fn plan_center_pads_and_crops() {
        // Pad: 2x2 MB (32x32) centred in 4x4 MB (64x64).
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::Center);
        assert_eq!(
            out.plan(coded),
            ScalingPlan::Center {
                luma_copy_width: 32,
                luma_copy_height: 32,
                luma_src_left: 0,
                luma_src_top: 0,
                luma_dst_left: 16,
                luma_dst_top: 16,
            }
        );
        // Crop: 4x4 MB (64x64) centred into 2x2 MB (32x32).
        let coded = FrameGeometry::new(4, 4);
        let out = OutputScaling::new(FrameGeometry::new(2, 2), ScalingMode::Center);
        assert_eq!(
            out.plan(coded),
            ScalingPlan::Center {
                luma_copy_width: 32,
                luma_copy_height: 32,
                luma_src_left: 16,
                luma_src_top: 16,
                luma_dst_left: 0,
                luma_dst_top: 0,
            }
        );
    }

    #[test]
    fn plan_other_is_unspecified() {
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::Other);
        assert_eq!(out.plan(coded), ScalingPlan::Unspecified);
    }

    #[test]
    fn plan_degenerate_geometry_is_identity() {
        let coded = FrameGeometry::new(2, 2);
        let out = OutputScaling::new(FrameGeometry::new(0, 4), ScalingMode::ScaleToFit);
        assert_eq!(out.plan(coded), ScalingPlan::Identity);
        let out = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::ScaleToFit);
        assert_eq!(out.plan(FrameGeometry::new(0, 0)), ScalingPlan::Identity);
    }

    #[test]
    fn resample_equal_size_is_exact_copy() {
        let src: Vec<u8> = (0..64u32).map(|v| (v * 3 % 251) as u8).collect();
        let out = resample_plane(&src, 8, 8, 8, 8);
        assert_eq!(out, src);
    }

    #[test]
    fn resample_preserves_constants() {
        for &v in &[0u8, 1, 127, 128, 254, 255] {
            let src = vec![v; 12 * 4];
            for &(dw, dh) in &[(24usize, 8usize), (6, 2), (5, 7), (48, 16)] {
                let out = resample_plane(&src, 12, 4, dw, dh);
                assert!(
                    out.iter().all(|&s| s == v),
                    "constant {v} not preserved at {dw}x{dh}"
                );
            }
        }
    }

    #[test]
    fn resample_upscale_ramp_tracks_linear_interpolant() {
        // A horizontal ramp y = 4x upsampled 2x: the centre-aligned
        // linear kernel must land within 1 LSB of the ideal linear
        // interpolant at every output sample.
        let src: Vec<u8> = (0..32u32).map(|x| (4 * x) as u8).collect();
        let out = resample_plane(&src, 32, 1, 64, 1);
        for (d, &got) in out.iter().enumerate() {
            let pos = (d as f64 + 0.5) * 32.0 / 64.0 - 0.5;
            let ideal = 4.0 * pos.clamp(0.0, 31.0);
            assert!(
                (got as f64 - ideal).abs() <= 1.0,
                "sample {d}: got {got}, ideal {ideal}"
            );
        }
    }

    #[test]
    fn resample_downscale_2to1_is_box_average() {
        // At exactly 2:1 the centre-aligned taps fall midway between
        // sample pairs: each output is the rounded 2-sample average per
        // axis (the 2x2 box in 2D).
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let out = resample_plane(&src, 8, 1, 4, 1);
        assert_eq!(out, vec![15, 35, 55, 75]);
    }

    #[test]
    fn resample_frame_stretch_dimensions_and_content() {
        // A flat frame stretches to a flat frame of the output size.
        let mut f = Frame::new(4, 4); // 2x2 MB, luma 32x32
        f.y.samples_mut().fill(77);
        f.u.samples_mut().fill(100);
        f.v.samples_mut().fill(200);
        let out = resample_frame(&f, FrameGeometry::new(4, 3));
        assert_eq!(out.y.width(), 64);
        assert_eq!(out.y.height(), 48);
        assert_eq!(out.u.width(), 32);
        assert_eq!(out.u.height(), 24);
        assert!(out.y.samples().iter().all(|&s| s == 77));
        assert!(out.u.samples().iter().all(|&s| s == 100));
        assert!(out.v.samples().iter().all(|&s| s == 200));
    }

    #[test]
    fn apply_identity_returns_frame_unchanged() {
        let mut f = Frame::new(4, 4);
        f.y.samples_mut().fill(9);
        let s = OutputScaling::new(FrameGeometry::new(2, 2), ScalingMode::ScaleToFit);
        assert_eq!(apply_output_scaling(&f, s), f);
        // OTHER: unspecified — identity even when geometry differs.
        let s = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::Other);
        assert_eq!(apply_output_scaling(&f, s), f);
    }

    #[test]
    fn apply_center_pads_with_neutral_fill() {
        let mut f = Frame::new(4, 4); // 2x2 MB, luma 32x32
        f.y.samples_mut().fill(50);
        f.u.samples_mut().fill(90);
        f.v.samples_mut().fill(160);
        let s = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::Center);
        let out = apply_output_scaling(&f, s);
        assert_eq!(out.y.width(), 64);
        // Centre pixel is source luma; corner is fill.
        assert_eq!(out.y.sample(32, 32), Some(50));
        assert_eq!(out.y.sample(0, 0), Some(0));
        assert_eq!(out.u.sample(16, 16), Some(90));
        assert_eq!(out.u.sample(0, 0), Some(128));
        assert_eq!(out.v.sample(16, 16), Some(160));
        assert_eq!(out.v.sample(0, 0), Some(128));
    }

    #[test]
    fn apply_center_crop_takes_centred_window() {
        // 4x4 MB (64x64) luma holding a coordinate pattern, cropped to
        // 2x2 MB (32x32): output (r, c) == source (r + 16, c + 16).
        let mut f = Frame::new(8, 8);
        for r in 0..64 {
            for c in 0..64 {
                let w = f.y.width();
                f.y.samples_mut()[r * w + c] = ((r * 3 + c) % 251) as u8;
            }
        }
        let s = OutputScaling::new(FrameGeometry::new(2, 2), ScalingMode::Center);
        let out = apply_output_scaling(&f, s);
        assert_eq!(out.y.width(), 32);
        for r in 0..32 {
            for c in 0..32 {
                assert_eq!(out.y.sample(r, c), f.y.sample(r + 16, c + 16));
            }
        }
    }

    #[test]
    fn apply_aspect_fit_places_scaled_image_with_letterbox() {
        // Flat 2x2 MB frame into a 4x2 MB output: fitted 32x32 at
        // x = 16, neutral fill either side.
        let mut f = Frame::new(4, 4);
        f.y.samples_mut().fill(200);
        f.u.samples_mut().fill(64);
        f.v.samples_mut().fill(192);
        let s = OutputScaling::new(FrameGeometry::new(4, 2), ScalingMode::MaintainAspectRatio);
        let out = apply_output_scaling(&f, s);
        assert_eq!(out.y.width(), 64);
        assert_eq!(out.y.height(), 32);
        // Inside the fitted rectangle.
        assert_eq!(out.y.sample(16, 32), Some(200));
        assert_eq!(out.u.sample(8, 16), Some(64));
        assert_eq!(out.v.sample(8, 16), Some(192));
        // Letterbox columns.
        assert_eq!(out.y.sample(16, 4), Some(0));
        assert_eq!(out.y.sample(16, 60), Some(0));
        assert_eq!(out.u.sample(8, 2), Some(128));
        assert_eq!(out.v.sample(8, 30), Some(128));
    }

    #[test]
    fn apply_stretch_upscales_to_output_geometry() {
        // Gradient 2x2 MB frame stretched to 4x4 MB: dimensions double
        // and the upscale stays close to the ideal linear interpolant.
        let mut f = Frame::new(4, 4);
        let w = f.y.width();
        for r in 0..32 {
            for c in 0..32 {
                f.y.samples_mut()[r * w + c] = (4 * c) as u8;
            }
        }
        f.u.samples_mut().fill(128);
        f.v.samples_mut().fill(128);
        let s = OutputScaling::new(FrameGeometry::new(4, 4), ScalingMode::ScaleToFit);
        let out = apply_output_scaling(&f, s);
        assert_eq!(out.y.width(), 64);
        assert_eq!(out.y.height(), 64);
        for c in 0..64usize {
            let pos = (c as f64 + 0.5) * 0.5 - 0.5;
            let ideal = 4.0 * pos.clamp(0.0, 31.0);
            let got = out.y.sample(32, c).unwrap() as f64;
            assert!(
                (got - ideal).abs() <= 1.0,
                "col {c}: got {got}, ideal {ideal}"
            );
        }
        assert!(out.u.samples().iter().all(|&s| s == 128));
    }
}

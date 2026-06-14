//! VP6 output scaling — the §9 `ScalingMode` / `Output*Fragments` static
//! surface (spec §9 Table 2, page 24).
//!
//! A VP6 frame is decoded at its **coded** resolution — `HFragments`
//! columns by `VFragments` rows of 8x8 blocks (§9: "*VFragments. The
//! vertical coded height of the frame in 8x8 block units. If image is 240
//! pixels high, VFragments will be 30*"; "*HFragments. The horizontal
//! coded width … If image is 320 pixels wide, HFragments will be 40*").
//! After decoding, the frame "*may be encoded at a different resolution to
//! the eventual size that it is presented on output from the decoder*"
//! (§9). The header carries the **output** geometry as
//! `OutputHFragments` / `OutputVFragments` (also in 8x8-block units) plus
//! a two-bit `ScalingMode` selecting one of four named scaling
//! strategies.
//!
//! This module surfaces the **BoolCoder-independent** parts of that
//! feature: the four named `ScalingMode` values, the coded/output
//! fragment-count → pixel-dimension derivations, and the relationship
//! between them. Every value here is pure integer arithmetic over already
//! decoded fragment counts; nothing reads a BoolCoder bit, so — like
//! §15/§16/§17/§11/§12/§14 — it advances the decoder without touching the
//! contested §7.3 `Split` formula. The `Output*Fragments` and
//! `ScalingMode` fields are themselves `b(8)` / `b(2)` BoolCoder-coded in
//! Table 2's tail; this module describes their *meaning* once decoded,
//! leaving the BoolCoder read to the (still-blocked) §9 header-tail
//! parser.
//!
//! ## DOCS-GAP — per-mode placement geometry
//!
//! The staged `vp6_format.pdf` names the four scaling modes
//! ("*MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER*", §9 page 24)
//! and states that output geometry may differ from coded geometry, but it
//! does **not** specify the per-mode pixel-mapping algorithm: how a
//! smaller coded image is positioned within (CENTER) or stretched to
//! (SCALE_TO_FIT) the output rectangle, what aspect-preserving fit
//! MAINTAIN_ASPECT_RATIO performs, or what OTHER signals. §2 lists
//! "*Scaling on output after decode*" only as a feature bullet. The
//! actual resampling/placement math is therefore out of scope here and is
//! reported as a docs-gap candidate; this module lands the typed mode
//! surface and the dimension derivations the math would consume.
//!
//! All field semantics are sourced verbatim from
//! `docs/video/vp6/vp6_format.pdf` §9 / §2; no external library code was
//! consulted.

/// Pixels per 8x8 fragment edge — the §2 transform block size.
///
/// A fragment count `n` corresponds to `n * FRAGMENT_DIM` pixels along
/// that axis.
pub const FRAGMENT_DIM: u32 = 8;

/// The §9 `ScalingMode` field — how the decoded frame is scaled to the
/// output resolution (Table 2, `ScalingMode b(2)`).
///
/// The spec lists the four modes in the order
/// "*MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER*" (§9, page 24)
/// for the two-bit field. That listing order is the only ordering the
/// document supplies, so the discriminants follow it (`0..=3`).
///
/// The per-mode pixel-placement algorithm is **not specified** in the
/// staged doc (see the module-level DOCS-GAP note); this enum captures
/// only the mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScalingMode {
    /// `0` — `MAINTAIN_ASPECT_RATIO`. First-listed §9 mode.
    MaintainAspectRatio = 0,
    /// `1` — `SCALE_TO_FIT`. Second-listed §9 mode.
    ScaleToFit = 1,
    /// `2` — `CENTER`. Third-listed §9 mode.
    Center = 2,
    /// `3` — `OTHER`. Fourth-listed §9 mode.
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

/// A frame geometry expressed in 8x8-fragment units along each axis,
/// shared by both the coded ([`FrameGeometry::coded`]) and output
/// ([`FrameGeometry::output`]) descriptions.
///
/// The luma plane is `h_fragments * 8` pixels wide and
/// `v_fragments * 8` pixels tall (§9 worked example: a 320x240 image has
/// `HFragments = 40`, `VFragments = 30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameGeometry {
    /// Horizontal fragment count — `HFragments` (coded) or
    /// `OutputHFragments` (output).
    pub h_fragments: u32,
    /// Vertical fragment count — `VFragments` (coded) or
    /// `OutputVFragments` (output).
    pub v_fragments: u32,
}

impl FrameGeometry {
    /// Construct a geometry from a `(h_fragments, v_fragments)` pair.
    pub fn new(h_fragments: u32, v_fragments: u32) -> Self {
        Self {
            h_fragments,
            v_fragments,
        }
    }

    /// The coded luma width in pixels: `h_fragments * 8` (§9).
    pub fn luma_width(self) -> u32 {
        self.h_fragments * FRAGMENT_DIM
    }

    /// The coded luma height in pixels: `v_fragments * 8` (§9).
    pub fn luma_height(self) -> u32 {
        self.v_fragments * FRAGMENT_DIM
    }

    /// The horizontal macroblock count covering this geometry:
    /// `ceil(h_fragments / 2)` (§2 4:2:0 — each 16x16 macroblock spans a
    /// 2x2 block of luma fragments).
    pub fn mb_cols(self) -> u32 {
        self.h_fragments.div_ceil(2)
    }

    /// The vertical macroblock count covering this geometry:
    /// `ceil(v_fragments / 2)` (§2 4:2:0).
    pub fn mb_rows(self) -> u32 {
        self.v_fragments.div_ceil(2)
    }
}

/// The §9 output-scaling description: the desired output geometry plus the
/// [`ScalingMode`] that maps the coded frame onto it.
///
/// Pairs `OutputHFragments` / `OutputVFragments` (Table 2) with the
/// `ScalingMode` field. The coded geometry that feeds the scale lives
/// separately ([`FrameGeometry::coded`]); [`OutputScaling::is_identity`]
/// reports whether a given coded geometry needs any scaling at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputScaling {
    /// The output geometry in fragment units (`OutputHFragments` /
    /// `OutputVFragments`).
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
    /// an N-fragment image onto an N-fragment output as the identity.)
    pub fn is_identity(self, coded: FrameGeometry) -> bool {
        coded == self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_dim_is_eight() {
        // §2 / §16 transform block size.
        assert_eq!(FRAGMENT_DIM, 8);
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
        // A b(2) read only ever produces 0..=3; anything larger is a
        // caller bug, surfaced as None rather than silently mapped.
        for v in 4u8..=255 {
            assert_eq!(ScalingMode::from_b2(v), None);
        }
    }

    #[test]
    fn coded_geometry_320x240_worked_example() {
        // §9 worked example: a 320x240 image has HFragments 40,
        // VFragments 30.
        let g = FrameGeometry::new(40, 30);
        assert_eq!(g.luma_width(), 320);
        assert_eq!(g.luma_height(), 240);
    }

    #[test]
    fn luma_dimensions_are_fragments_times_eight() {
        for &(h, v) in &[(1u32, 1u32), (40, 30), (44, 36), (255, 255)] {
            let g = FrameGeometry::new(h, v);
            assert_eq!(g.luma_width(), h * 8);
            assert_eq!(g.luma_height(), v * 8);
        }
    }

    #[test]
    fn mb_grid_is_fragment_count_round_up() {
        // §2 4:2:0: each macroblock spans a 2x2 block of luma fragments.
        // Even fragment counts halve exactly; odd ones round up (the
        // partial edge macroblock the frame_assembly module handles).
        assert_eq!(FrameGeometry::new(40, 30).mb_cols(), 20);
        assert_eq!(FrameGeometry::new(40, 30).mb_rows(), 15);
        // Odd fragment counts round up.
        assert_eq!(FrameGeometry::new(41, 31).mb_cols(), 21);
        assert_eq!(FrameGeometry::new(41, 31).mb_rows(), 16);
        // 1 fragment still needs 1 macroblock.
        assert_eq!(FrameGeometry::new(1, 1).mb_cols(), 1);
        assert_eq!(FrameGeometry::new(1, 1).mb_rows(), 1);
    }

    #[test]
    fn output_scaling_identity_when_coded_matches_output() {
        let coded = FrameGeometry::new(40, 30);
        let out = OutputScaling::new(coded, ScalingMode::ScaleToFit);
        assert!(out.is_identity(coded));
    }

    #[test]
    fn output_scaling_non_identity_when_dimensions_differ() {
        let coded = FrameGeometry::new(20, 15);
        let out = OutputScaling::new(FrameGeometry::new(40, 30), ScalingMode::ScaleToFit);
        assert!(!out.is_identity(coded));
        // The mode does not affect the identity test — only the geometry
        // comparison does.
        let out_center = OutputScaling::new(FrameGeometry::new(40, 30), ScalingMode::Center);
        assert!(!out_center.is_identity(coded));
    }

    #[test]
    fn output_scaling_identity_independent_of_mode() {
        // When coded == output, every mode reports identity (the choice
        // of mode is moot for an exact-size output).
        let geom = FrameGeometry::new(44, 36);
        for v in 0u8..=3 {
            let mode = ScalingMode::from_b2(v).unwrap();
            let out = OutputScaling::new(geom, mode);
            assert!(out.is_identity(geom));
        }
    }

    #[test]
    fn output_geometry_pixel_dimensions() {
        // Output fragments map to pixels the same way coded fragments do.
        let out = OutputScaling::new(FrameGeometry::new(80, 60), ScalingMode::MaintainAspectRatio);
        assert_eq!(out.output.luma_width(), 640);
        assert_eq!(out.output.luma_height(), 480);
    }
}

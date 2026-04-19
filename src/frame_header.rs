//! VP6 frame-header parse.
//!
//! Based on FFmpeg's `libavcodec/vp6.c::vp6_parse_header`. The first
//! byte of a VP6 frame is a set of raw-bit fields; on a keyframe the
//! next bytes carry the frame size in 16x16 macroblocks. After that
//! comes the bool-coder-encoded part of the header (the "Picture
//! Coding Extension" payload — filter mode, bicubic-filter flag,
//! golden-refresh flag, ...).
//!
//! Only the fields needed to allocate the YUV buffers and to run the
//! range coder through the rest of the picture layer are extracted
//! here. Coefficient-partition offsets and macroblock decode live
//! elsewhere (or — for this initial scaffold — return
//! `Error::Unsupported`).

use oxideav_core::{Error, Result};

use crate::range_coder::RangeCoder;

/// Frame kind: keyframe (I) or interframe (P).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Key,
    Inter,
}

/// Parsed VP6 frame header.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub kind: FrameKind,
    /// Low-bit quantisation index.
    pub qp: u8,
    /// Spec calls this "separated coeff" — when set, coefficient data
    /// lives in a second partition whose byte offset is given by
    /// `coeff_offset`.
    pub separated_coeff: bool,
    /// Byte offset of the coefficient partition inside the frame
    /// buffer, measured from the start of the frame. Only set when
    /// `separated_coeff` is true — otherwise 0.
    pub coeff_offset: u32,
    /// Profile byte — copied through for completeness.
    pub vp_profile: u8,
    /// VP6 "simple profile" (version 0) vs "advanced profile" (1..).
    pub simple_profile: bool,
    /// True when the bicubic luma filter is selected (vs bilinear).
    pub use_bicubic: bool,
    /// Width in 16x16 macroblocks (only valid on keyframes; inter
    /// frames inherit from the last keyframe).
    pub mb_width: u16,
    /// Height in 16x16 macroblocks.
    pub mb_height: u16,
    /// Display width — equal to mb_width*16 minus any cropping.
    pub display_width: u16,
    /// Display height — equal to mb_height*16 minus any cropping.
    pub display_height: u16,
    /// Index into the range coder's input where bool-coded data starts.
    pub range_coder_offset: usize,
    /// True when `golden_refresh` bit is set in this inter frame
    /// (golden reference gets refreshed to the new reconstruction).
    pub refresh_golden: bool,
    /// True when the frame carries an interlaced flag.
    pub interlaced: bool,
}

impl FrameHeader {
    /// Parse a VP6 frame header from `buf`.
    ///
    /// Layout recap (FFmpeg `vp6_parse_header`):
    ///
    /// ```text
    /// byte 0  bit 7       frame-mode (0 = keyframe, 1 = inter)
    ///         bits 6..3   qp
    ///         bit 2       separated-coeff
    ///         bits 1..0   unused in simple profile
    /// byte 1  bits 7..5   reserved
    ///         bit 4       simple-profile? (when keyframe)
    /// ...
    /// ```
    ///
    /// The exact field order depends on the profile (simple vs
    /// advanced) and on whether the frame is a keyframe. We decode
    /// enough to start the range coder and surface mb_width /
    /// mb_height; anything else is deferred to the per-frame decode
    /// path.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 4 {
            return Err(Error::invalid("VP6: frame too short for header"));
        }
        let b0 = buf[0];
        let frame_mode = (b0 >> 7) & 0x01;
        let qp = (b0 >> 1) & 0x3F; // 6-bit quantiser
        let separated_coeff = (b0 & 0x01) != 0;
        let kind = if frame_mode == 0 {
            FrameKind::Key
        } else {
            FrameKind::Inter
        };

        let b1 = buf[1];
        let vp_profile = (b1 >> 3) & 0x07;
        // Profile 0 ("simple") skips the interlace / chroma-scale bits.
        let simple_profile = vp_profile == 0;

        let mut pos = 2usize;
        let mut coeff_offset = 0u32;
        if separated_coeff {
            if buf.len() < pos + 2 {
                return Err(Error::invalid("VP6: missing coeff offset"));
            }
            coeff_offset = ((buf[pos] as u32) << 8) | buf[pos + 1] as u32;
            pos += 2;
        }

        let mut mb_width = 0u16;
        let mut mb_height = 0u16;
        let mut display_width = 0u16;
        let mut display_height = 0u16;
        let mut interlaced = false;

        if matches!(kind, FrameKind::Key) {
            // Keyframe-only header bytes:
            //   [pos]     MB height
            //   [pos+1]   MB width
            //   [pos+2]   display height (units of 16)
            //   [pos+3]   display width (units of 16)
            if buf.len() < pos + 4 {
                return Err(Error::invalid("VP6: missing keyframe dims"));
            }
            mb_height = buf[pos] as u16;
            mb_width = buf[pos + 1] as u16;
            display_height = buf[pos + 2] as u16;
            display_width = buf[pos + 3] as u16;
            pos += 4;
            // Reject implausibly-zero frames — indicates a truncated or
            // misidentified stream rather than a valid decode candidate.
            if mb_width == 0 || mb_height == 0 {
                return Err(Error::invalid("VP6: zero frame dimensions"));
            }
            if !simple_profile {
                // Advanced profile keyframe carries a single "filter
                // header" byte here with flags we don't need yet —
                // skip it.
                if buf.len() < pos + 1 {
                    return Err(Error::invalid("VP6: truncated advanced keyframe header"));
                }
                let adv_flags = buf[pos];
                pos += 1;
                interlaced = (adv_flags & 0x80) != 0;
            }
        }

        if buf.len() <= pos + 2 {
            return Err(Error::invalid("VP6: missing range-coder payload"));
        }

        // Construct the range coder and pull flags that live in the
        // bool-coded section of the picture header. We consume the
        // flags we need to match FFmpeg's framing, then stop — the
        // macroblock-decode path will re-create the bool coder later
        // once full decode is implemented.
        let mut rac = RangeCoder::new(&buf[pos..])?;
        let refresh_golden = if matches!(kind, FrameKind::Inter) {
            rac.get_prob(128) != 0
        } else {
            // Keyframe refreshes golden by definition.
            true
        };
        // Bicubic flag: VP6 picks a 4-tap bicubic filter vs a bilinear
        // chroma filter. It lives in the bool-coded header regardless
        // of frame kind.
        let use_bicubic = rac.get_prob(128) != 0;

        Ok(FrameHeader {
            kind,
            qp,
            separated_coeff,
            coeff_offset,
            vp_profile,
            simple_profile,
            use_bicubic,
            mb_width,
            mb_height,
            display_width,
            display_height,
            range_coder_offset: pos,
            refresh_golden,
            interlaced,
        })
    }

    pub fn frame_width(&self) -> u32 {
        self.mb_width as u32 * 16
    }

    pub fn frame_height(&self) -> u32 {
        self.mb_height as u32 * 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_keyframe_bytes() -> Vec<u8> {
        // Hand-rolled minimal VP6 keyframe header:
        //   byte0 = 0 (key) + qp=10 in bits 6..1, separated_coeff=0
        //   byte1 = 0, vp_profile=0 (simple)
        //   byte2 = mb_height=12 (192 px)
        //   byte3 = mb_width=16 (256 px)
        //   byte4 = display_height=12
        //   byte5 = display_width=16
        //   byte6..  = range coder priming bytes
        let mut out = Vec::new();
        let qp = 10u8;
        let b0 = (qp << 1) & 0x7F; // frame_mode=0 (key)
        out.push(b0);
        let vp_profile = 0u8;
        let b1 = (vp_profile & 0x07) << 3;
        out.push(b1);
        out.push(12); // mb height
        out.push(16); // mb width
        out.push(12); // display height
        out.push(16); // display width
                      // Range coder bytes — 0xFF 0xFF is a benign prime (splits at midpoint).
        out.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00]);
        out
    }

    #[test]
    fn parses_minimal_keyframe() {
        let bytes = synth_keyframe_bytes();
        let h = FrameHeader::parse(&bytes).unwrap();
        assert_eq!(h.kind, FrameKind::Key);
        assert_eq!(h.qp, 10);
        assert_eq!(h.mb_width, 16);
        assert_eq!(h.mb_height, 12);
        assert_eq!(h.frame_width(), 256);
        assert_eq!(h.frame_height(), 192);
        assert!(h.simple_profile);
        assert!(h.refresh_golden);
    }

    #[test]
    fn rejects_truncated() {
        let bytes = [0, 0];
        assert!(FrameHeader::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_zero_dims() {
        let mut bytes = synth_keyframe_bytes();
        bytes[2] = 0;
        bytes[3] = 0;
        assert!(FrameHeader::parse(&bytes).is_err());
    }
}

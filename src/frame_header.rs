//! VP6 frame-header parse.
//!
//! Ports `libavcodec/vp6.c::vp6_parse_header`. The first byte is a
//! fixed field layout (frame kind, QP, separated-coeff flag); on a
//! keyframe the next bytes carry MB-aligned dimensions, then the bool
//! coder starts. On inter frames the bool coder starts immediately
//! after the (optional) coeff-offset field.

use oxideav_core::{Error, Result};

/// Frame kind: keyframe (I) or interframe (P).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Key,
    Inter,
}

/// A parsed VP6 frame header, along with the byte offset where the
/// bool-coded picture header begins (range-coder priming point).
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub kind: FrameKind,
    /// 6-bit quantiser.
    pub qp: u8,
    /// When set, the coefficient partition starts at a separate byte
    /// offset inside the frame buffer — see `coeff_offset_bytes`.
    pub separated_coeff: bool,
    /// Byte offset into the frame buffer where the coefficient
    /// partition starts, measured from `range_coder_offset`. Always 0
    /// when there's a single partition.
    pub coeff_offset_bytes: i32,
    /// Sub-version (upper 5 bits of byte 1, only meaningful on keyframes).
    /// Preserved across frames so inter frames can check against `sub_version > 7`.
    pub sub_version: u8,
    /// True for "simple profile" (sub_version == 0).
    pub simple_profile: bool,
    /// `buf[1] & 0x06` — present when the filter-info section is present
    /// after the picture header.
    pub filter_header: u8,
    /// Width in 16x16 macroblocks (only meaningful on keyframes;
    /// callers should cache the last keyframe value for inter frames).
    pub mb_width: u16,
    pub mb_height: u16,
    /// Display dims in MB units (crop is applied by the container).
    pub display_mb_width: u16,
    pub display_mb_height: u16,
    /// Offset into the packet where the bool-coded picture header starts.
    pub range_coder_offset: usize,
    /// True when the frame carries the "interlaced" advanced-profile flag.
    pub interlaced: bool,
}

impl FrameHeader {
    /// Parse the fixed-layout part of the header. Consumers then
    /// construct a [`RangeCoder`](crate::range_coder::RangeCoder) from
    /// `&buf[range_coder_offset..]` to read the bool-coded part.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 2 {
            return Err(Error::invalid("VP6: frame too short for header"));
        }
        let b0 = buf[0];
        let frame_mode_bit = (b0 & 0x80) != 0;
        let kind = if frame_mode_bit {
            FrameKind::Inter
        } else {
            FrameKind::Key
        };
        let qp = (b0 >> 1) & 0x3F;
        let separated_coeff = (b0 & 0x01) != 0;

        let mb_width;
        let mb_height;
        let display_mb_width;
        let display_mb_height;
        let interlaced;
        let sub_version;
        let simple_profile;
        let filter_header;
        let coeff_offset_bytes;
        let range_coder_offset;

        if matches!(kind, FrameKind::Key) {
            sub_version = buf[1] >> 3;
            if sub_version > 8 {
                return Err(Error::invalid("VP6: unsupported sub-version"));
            }
            filter_header = buf[1] & 0x06;
            interlaced = (buf[1] & 1) != 0;
            simple_profile = sub_version == 0;

            // Slide window if separated-coeff or filter-header absent.
            let mut cursor = 2usize;
            if separated_coeff || filter_header == 0 {
                if buf.len() < cursor + 2 {
                    return Err(Error::invalid("VP6: missing keyframe coeff offset"));
                }
                coeff_offset_bytes = (((buf[cursor] as i32) << 8) | (buf[cursor + 1] as i32)) - 2;
                cursor += 2;
            } else {
                coeff_offset_bytes = 0;
            }

            if buf.len() < cursor + 4 {
                return Err(Error::invalid("VP6: missing keyframe dims"));
            }
            mb_height = buf[cursor] as u16;
            mb_width = buf[cursor + 1] as u16;
            display_mb_height = buf[cursor + 2] as u16;
            display_mb_width = buf[cursor + 3] as u16;

            if mb_width == 0 || mb_height == 0 {
                return Err(Error::invalid("VP6: zero frame dimensions"));
            }
            range_coder_offset = cursor + 4;
        } else {
            return Err(Error::invalid(
                "VP6: inter frame requires parse_inter (needs cached filter_header)",
            ));
        }

        if buf.len() < range_coder_offset + 1 {
            return Err(Error::invalid("VP6: missing range-coder payload"));
        }

        Ok(FrameHeader {
            kind,
            qp,
            separated_coeff,
            coeff_offset_bytes,
            sub_version,
            simple_profile,
            filter_header,
            mb_width,
            mb_height,
            display_mb_width,
            display_mb_height,
            range_coder_offset,
            interlaced,
        })
    }

    /// Parse an inter-frame header. Callers must supply the
    /// `filter_header` and `sub_version` cached from the previous
    /// keyframe (FFmpeg keeps these in the `VP56Context`).
    pub fn parse_inter(buf: &[u8], filter_header: u8, sub_version: u8) -> Result<Self> {
        if buf.len() < 2 {
            return Err(Error::invalid("VP6: inter frame too short"));
        }
        let b0 = buf[0];
        if (b0 & 0x80) == 0 {
            return Err(Error::invalid("VP6: parse_inter called on keyframe"));
        }
        let qp = (b0 >> 1) & 0x3F;
        let separated_coeff = (b0 & 0x01) != 0;

        let mut cursor = 1usize;
        let coeff_offset_bytes;
        if separated_coeff || filter_header == 0 {
            if buf.len() < cursor + 2 {
                return Err(Error::invalid("VP6: missing inter coeff offset"));
            }
            coeff_offset_bytes = (((buf[cursor] as i32) << 8) | (buf[cursor + 1] as i32)) - 2;
            cursor += 2;
        } else {
            coeff_offset_bytes = 0;
        }
        let range_coder_offset = cursor;
        if buf.len() < range_coder_offset + 1 {
            return Err(Error::invalid("VP6: missing range-coder payload"));
        }

        Ok(FrameHeader {
            kind: FrameKind::Inter,
            qp,
            separated_coeff,
            coeff_offset_bytes,
            sub_version,
            simple_profile: sub_version == 0,
            filter_header,
            mb_width: 0,
            mb_height: 0,
            display_mb_width: 0,
            display_mb_height: 0,
            range_coder_offset,
            interlaced: false,
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
        let mut out = Vec::new();
        let qp = 10u8;
        let b0 = (qp << 1) & 0x7F; // frame_mode=0 (key), sep_coeff=0
        out.push(b0);
        // sub_version=0 (simple_profile), filter_header=0, interlaced=0
        out.push(0);
        // With filter_header==0, the coeff-offset field IS present (2 bytes).
        out.push(0);
        out.push(2); // coeff offset so adjusted = 0
        out.push(12); // mb_height
        out.push(16); // mb_width
        out.push(12); // display_mb_height
        out.push(16); // display_mb_width
                      // Range-coder seed: 3 non-zero bytes so decoding doesn't immediately
                      // signal end-of-stream.
        out.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00]);
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
        assert_eq!(h.sub_version, 0);
    }

    #[test]
    fn rejects_truncated() {
        let bytes = [0, 0];
        assert!(FrameHeader::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_zero_dims() {
        let mut bytes = synth_keyframe_bytes();
        bytes[4] = 0;
        bytes[5] = 0;
        assert!(FrameHeader::parse(&bytes).is_err());
    }
}

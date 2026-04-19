//! VP6 decoder skeleton.
//!
//! Wires up the [`oxideav_codec::Decoder`] trait against the range
//! coder + frame-header parser. Produces a keyframe-sized zero-filled
//! `VideoFrame` so downstream pipeline code has something plausible to
//! display until full macroblock / coefficient decode is implemented.
//! Interframes currently error out with `Error::Unsupported`.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::frame_header::{FrameHeader, FrameKind};

/// VP6 flavour. Tracks whether we're decoding the FLV-only
/// `vp6f` (no alpha) or `vp6a` (alpha plane prefixed). Today only
/// `vp6f` decode proceeds — `vp6a` packets surface as
/// `Error::Unsupported`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp6Variant {
    /// FLV's vp6f.
    Flv,
    /// FLV's vp6a — VP6 plus a full-resolution alpha plane.
    FlvAlpha,
}

impl Vp6Variant {
    pub fn from_codec_id(id: &CodecId) -> Self {
        match id.as_str() {
            "vp6a" => Self::FlvAlpha,
            _ => Self::Flv,
        }
    }
}

/// Streaming VP6 decoder.
pub struct Vp6Decoder {
    codec_id: CodecId,
    variant: Vp6Variant,
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    width: Option<u32>,
    height: Option<u32>,
    /// Last-decoded luma plane (retained across frames so inter
    /// reconstruction can reference it once wired up).
    last_luma: Option<Vec<u8>>,
}

impl std::fmt::Debug for Vp6Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp6Decoder")
            .field("codec_id", &self.codec_id)
            .field("variant", &self.variant)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("queued", &self.queued.len())
            .finish()
    }
}

impl Vp6Decoder {
    pub fn new(params: CodecParameters) -> Self {
        let variant = Vp6Variant::from_codec_id(&params.codec_id);
        Self {
            codec_id: params.codec_id,
            variant,
            queued: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            width: params.width,
            height: params.height,
            last_luma: None,
        }
    }

    /// Parse + decode a single packet's bytes. Keyframes emit a
    /// zero-valued YUV frame sized from the frame header; interframes
    /// / alpha variants return `Unsupported` for now so callers can
    /// gracefully skip them.
    fn decode_bytes(&mut self, data: &[u8]) -> Result<VideoFrame> {
        if matches!(self.variant, Vp6Variant::FlvAlpha) {
            return Err(Error::unsupported(
                "VP6: vp6a (alpha plane) decode not yet implemented",
            ));
        }
        let header = FrameHeader::parse(data)?;
        if matches!(header.kind, FrameKind::Inter) {
            return Err(Error::unsupported(
                "VP6: interframe decode not yet implemented",
            ));
        }
        let width = header.frame_width();
        let height = header.frame_height();
        self.width = Some(width);
        self.height = Some(height);

        // Allocate Yuv420P planes. Full VP6 decode isn't ported yet —
        // we return a mid-grey luma (Y=128) + neutral chroma
        // (U=V=128) so downstream code can at least see a non-empty
        // buffer and the dimensions are visible for diagnostics /
        // tests.
        let stride_y = width as usize;
        let stride_uv = (width as usize).div_ceil(2);
        let h_uv = (height as usize).div_ceil(2);
        let mut y_plane = vec![128u8; stride_y * height as usize];
        let u_plane = vec![128u8; stride_uv * h_uv];
        let v_plane = vec![128u8; stride_uv * h_uv];

        // Stash the plane for reference-picture continuity — real
        // decode will re-use this once the MB path lands.
        self.last_luma = Some(y_plane.clone());
        let _ = header.qp; // reserved

        let planes = vec![
            VideoPlane {
                stride: stride_y,
                data: std::mem::take(&mut y_plane),
            },
            VideoPlane {
                stride: stride_uv,
                data: u_plane,
            },
            VideoPlane {
                stride: stride_uv,
                data: v_plane,
            },
        ];
        let frame = VideoFrame {
            format: PixelFormat::Yuv420P,
            width,
            height,
            pts: self.pending_pts,
            time_base: self.pending_tb,
            planes,
        };
        Ok(frame)
    }
}

impl Decoder for Vp6Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        // FLV wraps VP6 frames with a 1-byte "adjustment" prefix:
        //   bits 4..7 = horizontal adjustment (pixels)
        //   bits 0..3 = vertical adjustment (pixels)
        // The byte is discarded by the codec — consumers that care
        // about the crop offset would read it from the container.
        let data = if packet.data.is_empty() {
            packet.data.as_slice()
        } else {
            &packet.data[1..]
        };
        let frame = self.decode_bytes(data)?;
        self.queued.push_back(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.queued.pop_front() {
            Some(v) => Ok(Frame::Video(v)),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.queued.clear();
        self.pending_pts = None;
        self.last_luma = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_keyframe_bytes() -> Vec<u8> {
        // Mirrors the helper in frame_header::tests — 256x192 keyframe.
        let mut out = Vec::new();
        let qp = 10u8;
        let b0 = (qp << 1) & 0x7F;
        out.push(b0);
        let vp_profile = 0u8;
        let b1 = (vp_profile & 0x07) << 3;
        out.push(b1);
        out.push(12); // mb_h
        out.push(16); // mb_w
        out.push(12); // display_h
        out.push(16); // display_w
        out.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00]);
        out
    }

    #[test]
    fn keyframe_yields_frame() {
        let mut params = CodecParameters::video(CodecId::new("vp6f"));
        params.width = Some(256);
        params.height = Some(192);
        let mut dec = Vp6Decoder::new(params);
        // send_packet expects the FLV 1-byte prefix — prepend 0.
        let mut data = vec![0u8];
        data.extend_from_slice(&synth_keyframe_bytes());
        let pkt = Packet::new(0, TimeBase::new(1, 1000), data);
        dec.send_packet(&pkt).unwrap();
        let f = dec.receive_frame().unwrap();
        match f {
            Frame::Video(v) => {
                assert_eq!(v.format, PixelFormat::Yuv420P);
                assert_eq!(v.width, 256);
                assert_eq!(v.height, 192);
                assert_eq!(v.planes.len(), 3);
                assert_eq!(v.planes[0].data.len(), 256 * 192);
                assert_eq!(v.planes[1].data.len(), 128 * 96);
                assert_eq!(v.planes[2].data.len(), 128 * 96);
            }
            _ => panic!("expected video frame"),
        }
    }

    #[test]
    fn reset_clears_queue() {
        let mut params = CodecParameters::video(CodecId::new("vp6f"));
        params.width = Some(256);
        params.height = Some(192);
        let mut dec = Vp6Decoder::new(params);
        let mut data = vec![0u8];
        data.extend_from_slice(&synth_keyframe_bytes());
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1000), data))
            .unwrap();
        dec.reset().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    #[test]
    fn alpha_variant_errors() {
        let params = CodecParameters::video(CodecId::new("vp6a"));
        let mut dec = Vp6Decoder::new(params);
        let mut data = vec![0u8];
        data.extend_from_slice(&synth_keyframe_bytes());
        let pkt = Packet::new(0, TimeBase::new(1, 1000), data);
        assert!(matches!(dec.send_packet(&pkt), Err(Error::Unsupported(_))));
    }
}

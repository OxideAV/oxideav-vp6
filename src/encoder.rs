//! `oxideav-core` encoder registration — wires the VP6 keyframe + P-frame
//! encoders into the framework's [`oxideav_core::Encoder`] trait.
//!
//! [`Vp6CodecEncoder`] is a GOP-aware adapter: the first `send_frame` (and
//! every `keyframe_interval`-th frame after) emits a §9 keyframe packet via
//! [`crate::intra_encode::encode_intra_frame`]; every other frame emits a
//! motion-estimated P-frame packet via
//! [`crate::inter_encode::encode_inter_frame_me_packet`] against the
//! **decoded** previous frame.
//!
//! The previous-frame reference is maintained by decoding the encoder's own
//! output through an internal [`crate::decode_frame::Vp6Decoder`], so the
//! reference the encoder predicts from is byte-identical to the one a
//! downstream decoder reconstructs — the round-trip is closed by construction.
//!
//! ## Geometry
//!
//! VP6 is §2 fragment-based: the luma plane is `HFragments·8 × VFragments·8`,
//! so the encoder derives `h_fragments = width / 8`, `v_fragments = height / 8`
//! and requires both dimensions to be multiples of 8 (the only shape the
//! in-tree fragment-grid encoders emit). 4:2:0 chroma is one block per
//! macroblock.
//!
//! ## Provenance
//!
//! Framework plumbing over this crate's own spec-sourced encode pipeline. No
//! external library code was consulted.

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error as CoreError, Frame, MediaType, Packet, PixelFormat,
    Result, TimeBase, VideoFrame,
};

use crate::decode_frame::Vp6Decoder;
use crate::decoder::VP6_CODEC_ID;
use crate::frame_assembly::Frame as Vp6Frame;
use crate::inter_encode::encode_inter_frame_me_packet;
use crate::inter_frame::{BorderedRef, InterProbs};
use crate::intra_encode::encode_intra_frame;

/// The default keyframe interval (GOP length): a keyframe every N frames.
pub const DEFAULT_KEYFRAME_INTERVAL: u32 = 12;

/// The default §9 `DctQMask` quantiser index the encoder uses when the caller
/// doesn't override it. A mid-range index (finer than the I-frame floor, well
/// short of the lossless tail).
pub const DEFAULT_DCT_Q_MASK: u8 = 32;

/// `oxideav_core::Encoder` adapter over the VP6 keyframe + P-frame encoders.
///
/// One `send_frame` produces one `Packet` (queued for `receive_packet`):
/// keyframe at the start of every GOP, motion-estimated P-frame otherwise.
pub struct Vp6CodecEncoder {
    output_params: CodecParameters,
    h_fragments: usize,
    v_fragments: usize,
    /// §9 downsampled-encode state: when set, every source frame is
    /// resampled down to `coded` before encoding and the keyframe header
    /// signals `signal` (the display geometry + `ScalingMode`), so a
    /// decoder presents the GOP back at the original size (§2 "scaling
    /// on output after decode"). `None` ⇒ code at the display size.
    downscale: Option<(crate::scaling::FrameGeometry, crate::scaling::OutputScaling)>,
    dct_q_mask: u8,
    keyframe_interval: u32,
    time_base: TimeBase,
    /// Frame counter since the last keyframe (0 ⇒ next frame is a keyframe).
    frames_since_keyframe: u32,
    /// Internal decoder threading the §4 reference state; its last decoded
    /// frame is the previous-frame reference the next P-frame predicts from.
    decoder: Vp6Decoder,
    /// The most recent reconstructed frame (the P-frame reference). `None`
    /// until the first keyframe is emitted.
    reference: Option<Vp6Frame>,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl std::fmt::Debug for Vp6CodecEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp6CodecEncoder")
            .field("h_fragments", &self.h_fragments)
            .field("v_fragments", &self.v_fragments)
            .field("dct_q_mask", &self.dct_q_mask)
            .field("keyframe_interval", &self.keyframe_interval)
            .field("frames_since_keyframe", &self.frames_since_keyframe)
            .field("has_reference", &self.reference.is_some())
            .field("pending", &self.pending.len())
            .field("eof", &self.eof)
            .finish()
    }
}

impl Vp6CodecEncoder {
    /// Build a VP6 encoder for `width`×`height` (both multiples of 8) at the
    /// given `dct_q_mask` and `keyframe_interval`.
    ///
    /// # Errors
    ///
    /// [`CoreError::invalid`] if the dimensions aren't positive multiples of 8
    /// or exceed the §9 8-bit fragment-count fields (`HFragments`/`VFragments`
    /// are 8-bit, so the max is `255·8 = 2040` pixels per axis).
    pub fn new(
        params: &CodecParameters,
        width: u32,
        height: u32,
        dct_q_mask: u8,
        keyframe_interval: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(CoreError::invalid(
                "vp6 encoder: width and height must be positive",
            ));
        }
        if width % 8 != 0 || height % 8 != 0 {
            return Err(CoreError::invalid(
                "vp6 encoder: width and height must be multiples of 8",
            ));
        }
        let h_fragments = (width / 8) as usize;
        let v_fragments = (height / 8) as usize;
        if h_fragments > u8::MAX as usize || v_fragments > u8::MAX as usize {
            return Err(CoreError::invalid(
                "vp6 encoder: fragment count exceeds the §9 8-bit field (max 2040 px/axis)",
            ));
        }
        let pixel_format = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        if pixel_format != PixelFormat::Yuv420P {
            return Err(CoreError::unsupported(
                "vp6 encoder: only PixelFormat::Yuv420P is supported",
            ));
        }

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(VP6_CODEC_ID);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pixel_format);

        let time_base = params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));

        Ok(Self {
            output_params,
            h_fragments,
            v_fragments,
            downscale: None,
            dct_q_mask: dct_q_mask & 0x3F,
            keyframe_interval: keyframe_interval.max(1),
            time_base,
            frames_since_keyframe: 0,
            decoder: Vp6Decoder::new(),
            reference: None,
            pending: VecDeque::new(),
            eof: false,
        })
    }

    /// Enable the §9 **downsampled-encode** path: code every frame at
    /// `1/factor` of the display resolution (each axis) and signal the
    /// display geometry through the keyframe's `OutputVFragments` /
    /// `OutputHFragments` / `ScalingMode` fields
    /// ([`crate::intra_encode::encode_intra_frame_scaled`]), so a
    /// downstream decoder reconstructs at the coded size and upscales
    /// back on output ([`crate::decode_frame::Vp6Decoder::
    /// decode_packet_scaled`] / the registered [`crate::decoder::
    /// Vp6CodecDecoder`]). Inter frames carry no geometry (§9 Table 3),
    /// so the whole GOP is coded at the reduced size under the one
    /// keyframe signal.
    ///
    /// `factor == 1` disables downsampling. The signalled mode is
    /// `SCALE_TO_FIT` (the aspect ratio is preserved by construction, so
    /// the fit is the full-rectangle stretch).
    ///
    /// # Errors
    ///
    /// [`CoreError::invalid`] if the display dimensions are not
    /// macroblock-aligned (multiples of 16) or the macroblock grid is
    /// not divisible by `factor` (each coded axis must stay a whole,
    /// non-zero macroblock count).
    pub fn with_downscale(mut self, factor: u8) -> Result<Self> {
        if factor <= 1 {
            self.downscale = None;
            return Ok(self);
        }
        let f = factor as usize;
        if self.h_fragments % 2 != 0 || self.v_fragments % 2 != 0 {
            return Err(CoreError::invalid(
                "vp6 encoder: downscale needs macroblock-aligned dimensions (multiples of 16)",
            ));
        }
        let (mb_cols, mb_rows) = (self.h_fragments / 2, self.v_fragments / 2);
        if mb_cols % f != 0 || mb_rows % f != 0 || mb_cols / f == 0 || mb_rows / f == 0 {
            return Err(CoreError::invalid(
                "vp6 encoder: macroblock grid must divide evenly by the downscale factor",
            ));
        }
        let coded = crate::scaling::FrameGeometry::new((mb_cols / f) as u32, (mb_rows / f) as u32);
        let signal = crate::scaling::OutputScaling::new(
            crate::scaling::FrameGeometry::new(mb_cols as u32, mb_rows as u32),
            crate::scaling::ScalingMode::ScaleToFit,
        );
        self.downscale = Some((coded, signal));
        Ok(self)
    }

    /// Convert a framework [`VideoFrame`] (three I420 planes) into this crate's
    /// fragment-grid [`Vp6Frame`], repacking each plane to the crate's tight
    /// per-plane stride.
    fn to_vp6_frame(&self, v: &VideoFrame) -> Result<Vp6Frame> {
        if v.planes.len() < 3 {
            return Err(CoreError::invalid(
                "vp6 encoder: VideoFrame must carry 3 planes (Y/U/V)",
            ));
        }
        let mut frame = Vp6Frame::new(self.h_fragments, self.v_fragments);
        let (yw, yh) = (frame.y.width(), frame.y.height());
        let (uw, uh) = (frame.u.width(), frame.u.height());
        let (vw, vh) = (frame.v.width(), frame.v.height());
        copy_plane(&v.planes[0], frame.y.samples_mut(), yw, yh)?;
        copy_plane(&v.planes[1], frame.u.samples_mut(), uw, uh)?;
        copy_plane(&v.planes[2], frame.v.samples_mut(), vw, vh)?;
        Ok(frame)
    }

    /// Encode one source frame into a self-describing VP6 packet, advancing the
    /// §4 reference state via the internal decoder.
    fn encode_one(&mut self, v: &VideoFrame) -> Result<Vec<u8>> {
        let display = self.to_vp6_frame(v)?;
        // Downsampled-encode path: shrink the source to the coded
        // geometry; the keyframe header signals the display geometry so
        // the decoder upscales on output. The internal reference loop
        // stays at the coded resolution (scaling never re-enters the
        // §4 prediction loop).
        let source = match self.downscale {
            Some((coded, _)) => crate::scaling::resample_frame(&display, coded),
            None => display,
        };
        let is_keyframe = self.frames_since_keyframe == 0 || self.reference.is_none();

        let bytes = if is_keyframe {
            match self.downscale {
                Some((_, signal)) => {
                    crate::intra_encode::encode_intra_frame_scaled(&source, self.dct_q_mask, signal)
                        .map_err(CoreError::from)?
                }
                None => encode_intra_frame(&source, self.dct_q_mask).map_err(CoreError::from)?,
            }
        } else {
            let prev = self
                .reference
                .as_ref()
                .expect("non-keyframe path has a reference");
            let prev_bordered = BorderedRef::new(prev);
            let probs = InterProbs::keyframe();
            let filter = crate::inter_frame::FilterConfig::bilinear();
            encode_inter_frame_me_packet(&source, &prev_bordered, self.dct_q_mask, &probs, &filter)
                .map_err(CoreError::from)?
        };

        // Decode the encoder's own output to advance the reference exactly as a
        // downstream decoder would reconstruct it.
        let recon = self
            .decoder
            .decode_packet(&bytes)
            .map_err(CoreError::from)?;
        self.reference = Some(recon);

        self.frames_since_keyframe += 1;
        if self.frames_since_keyframe >= self.keyframe_interval {
            self.frames_since_keyframe = 0;
        }
        Ok(bytes)
    }
}

impl Encoder for Vp6CodecEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(CoreError::invalid("vp6 encoder: video frames only")),
        };
        let is_keyframe = self.frames_since_keyframe == 0 || self.reference.is_none();
        let bytes = self.encode_one(v)?;
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            Ok(p)
        } else if self.eof {
            Err(CoreError::Eof)
        } else {
            Err(CoreError::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Copy a framework plane into a tightly-packed `width × height` destination,
/// honouring the source stride and validating the buffer is long enough.
fn copy_plane(
    plane: &oxideav_core::frame::VideoPlane,
    dst: &mut [u8],
    width: usize,
    height: usize,
) -> Result<()> {
    if plane.stride < width
        || plane.data.len() < plane.stride * height
        || dst.len() < width * height
    {
        return Err(CoreError::invalid(
            "vp6 encoder: plane buffer shorter than declared dimensions",
        ));
    }
    for r in 0..height {
        let src_off = r * plane.stride;
        let dst_off = r * width;
        dst[dst_off..dst_off + width].copy_from_slice(&plane.data[src_off..src_off + width]);
    }
    Ok(())
}

/// `make_encoder` factory: build a [`Vp6CodecEncoder`] from `params` with the
/// default quantiser + keyframe interval. `params` must declare positive
/// `width`/`height` that are multiples of 8.
///
/// # Errors
///
/// [`CoreError::invalid`] on missing/invalid dimensions (see
/// [`Vp6CodecEncoder::new`]).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_q(params, DEFAULT_DCT_Q_MASK)
}

/// `make_encoder` with an explicit §9 `DctQMask` quantiser index (`0..=63`,
/// **higher = finer/larger**).
///
/// # Errors
///
/// [`CoreError::invalid`] on missing/invalid dimensions.
pub fn make_encoder_with_q(params: &CodecParameters, dct_q_mask: u8) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| CoreError::invalid("vp6 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| CoreError::invalid("vp6 encoder: missing height"))?;
    Ok(Box::new(Vp6CodecEncoder::new(
        params,
        width,
        height,
        dct_q_mask,
        DEFAULT_KEYFRAME_INTERVAL,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Vp6CodecDecoder;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::Decoder;

    fn params(w: u32, h: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(VP6_CODEC_ID));
        p.width = Some(w);
        p.height = Some(h);
        p
    }

    /// Build a 3-plane I420 `VideoFrame` of a `w×h` gradient (chroma at the
    /// macroblock-block-grid width VP6 uses, one block per MB).
    fn gradient_video_frame(w: usize, h: usize) -> VideoFrame {
        let hf = w / 8;
        let vf = h / 8;
        let mb_cols = hf.div_ceil(2);
        let mb_rows = vf.div_ceil(2);
        let cw = mb_cols * 8;
        let ch = mb_rows * 8;

        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = (16 + (r + c) % 200) as u8;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut vv = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (110 + (r + c) / 2 % 30) as u8;
                vv[r * cw + c] = (150 - (r + c) / 2 % 30) as u8;
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: u,
                },
                VideoPlane {
                    stride: cw,
                    data: vv,
                },
            ],
        }
    }

    #[test]
    fn rejects_non_multiple_of_8_dimensions() {
        let err = Vp6CodecEncoder::new(&params(30, 32), 30, 32, 32, 12);
        assert!(err.is_err());
    }

    #[test]
    fn output_params_carry_dimensions() {
        let enc = make_encoder(&params(32, 32)).expect("encoder");
        let op = enc.output_params();
        assert_eq!(op.width, Some(32));
        assert_eq!(op.height, Some(32));
        assert_eq!(op.codec_id.as_str(), VP6_CODEC_ID);
    }

    /// A single keyframe encodes through the `Encoder` trait and decodes back
    /// through the `Decoder` trait, producing a 3-plane frame of the right
    /// dimensions.
    #[test]
    fn keyframe_round_trips_through_traits() {
        let mut enc = make_encoder(&params(32, 32)).expect("encoder");
        let vf = gradient_video_frame(32, 32);
        enc.send_frame(&Frame::Video(vf)).expect("send");
        let pkt = enc.receive_packet().expect("packet");
        assert!(pkt.flags.keyframe, "first frame must be a keyframe");

        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        dec.send_packet(&pkt).expect("send packet");
        let frame = dec.receive_frame().expect("decode");
        match frame {
            Frame::Video(out) => {
                assert_eq!(out.planes.len(), 3);
                assert_eq!(out.planes[0].data.len(), 32 * 32);
            }
            other => panic!("expected video, got {other:?}"),
        }
    }

    /// A keyframe → P-frame GOP: the second frame is a P-frame (not a
    /// keyframe) and both decode end-to-end through the decoder trait. The
    /// encoder's internal reference advances so the P-frame predicts from the
    /// decoded keyframe.
    #[test]
    fn gop_emits_keyframe_then_pframe() {
        let mut enc = make_encoder_with_q(&params(32, 32), 24).expect("encoder");
        let f0 = gradient_video_frame(32, 32);
        // A slightly shifted second frame.
        let mut f1 = gradient_video_frame(32, 32);
        // Perturb f1's luma so the P-frame carries a real residual.
        for (i, s) in f1.planes[0].data.iter_mut().enumerate() {
            *s = s.wrapping_add((i % 3) as u8);
        }

        enc.send_frame(&Frame::Video(f0)).expect("send f0");
        let k = enc.receive_packet().expect("keyframe");
        assert!(k.flags.keyframe);

        enc.send_frame(&Frame::Video(f1)).expect("send f1");
        let p = enc.receive_packet().expect("pframe");
        assert!(!p.flags.keyframe, "second frame must be a P-frame");

        // Both decode through one decoder instance (GOP order).
        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        dec.send_packet(&k).expect("send k");
        let _ = dec.receive_frame().expect("decode k");
        dec.send_packet(&p).expect("send p");
        let out = dec.receive_frame().expect("decode p");
        match out {
            Frame::Video(vf) => assert_eq!(vf.planes[0].data.len(), 32 * 32),
            other => panic!("expected video, got {other:?}"),
        }
    }

    /// The encoder resolves through the registry under the canonical id (the
    /// `register_codecs` wiring exposes `.encoder(make_encoder)`).
    #[test]
    fn encoder_resolves_through_registry() {
        use oxideav_core::CodecRegistry;
        let mut reg = CodecRegistry::new();
        crate::decoder::register_codecs(&mut reg);
        let enc = reg
            .first_encoder(&params(32, 32))
            .expect("registry encoder");
        assert_eq!(enc.codec_id().as_str(), VP6_CODEC_ID);
    }

    /// `receive_packet` with nothing queued asks for more input, then reports
    /// EOF after `flush`.
    #[test]
    fn receive_without_send_needs_more_then_eof() {
        let mut enc = make_encoder(&params(16, 16)).expect("encoder");
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
        enc.flush().expect("flush");
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
    }

    /// Build a smooth (wrap-free) I420 gradient — content that survives
    /// a downsample/upsample cycle with little loss, for the scaled-GOP
    /// PSNR gate.
    fn smooth_video_frame(w: usize, h: usize, shift: u8) -> VideoFrame {
        let cw = w / 2;
        let ch = h / 2;
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = (32 + (r + c) / 2) as u8 + shift;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut vv = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (96 + (r + c) / 2) as u8;
                vv[r * cw + c] = (160 - ((r + c) / 2) as i32) as u8;
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: u,
                },
                VideoPlane {
                    stride: cw,
                    data: vv,
                },
            ],
        }
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum();
        let mse = sse / a.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    /// The §9 downsampled-encode arc through the **trait surfaces**: a
    /// 64x64 GOP coded at 32x32 (`with_downscale(2)`), whose keyframe
    /// signals the display geometry via `Output*Fragments` /
    /// `ScalingMode`, decodes through the registered decoder back at
    /// 64x64 (the decoder applies the §9 upscale on output) and tracks
    /// the original full-resolution source above a down/up-plus-
    /// quantiser floor. The P-frame inherits the keyframe's scaling.
    #[test]
    fn downscaled_gop_round_trips_at_display_size() {
        let mut enc = Vp6CodecEncoder::new(&params(64, 64), 64, 64, 56, 12)
            .expect("encoder")
            .with_downscale(2)
            .expect("downscale config");

        let f0 = smooth_video_frame(64, 64, 0);
        let f1 = smooth_video_frame(64, 64, 2);
        let src0 = f0.planes[0].data.clone();
        let src1 = f1.planes[0].data.clone();

        enc.send_frame(&Frame::Video(f0)).expect("send f0");
        let k = enc.receive_packet().expect("keyframe");
        assert!(k.flags.keyframe);
        enc.send_frame(&Frame::Video(f1)).expect("send f1");
        let p = enc.receive_packet().expect("pframe");
        assert!(!p.flags.keyframe, "second frame must be a P-frame");

        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        dec.send_packet(&k).expect("send k");
        let out_k = match dec.receive_frame().expect("decode k") {
            Frame::Video(v) => v,
            other => panic!("expected video, got {other:?}"),
        };
        dec.send_packet(&p).expect("send p");
        let out_p = match dec.receive_frame().expect("decode p") {
            Frame::Video(v) => v,
            other => panic!("expected video, got {other:?}"),
        };

        // Both frames emit at the *display* geometry (the §9 upscale is
        // applied by the decoder), not the 32x32 coded geometry.
        assert_eq!(out_k.planes[0].data.len(), 64 * 64);
        assert_eq!(out_p.planes[0].data.len(), 64 * 64);
        assert_eq!(out_k.planes[1].data.len(), 32 * 32);

        let yk = psnr(&src0, &out_k.planes[0].data);
        let yp = psnr(&src1, &out_p.planes[0].data);
        assert!(yk >= 30.0, "keyframe luma PSNR {yk:.2} dB below floor");
        assert!(yp >= 30.0, "P-frame luma PSNR {yp:.2} dB below floor");
    }

    /// `with_downscale` validates the geometry: the macroblock grid must
    /// divide evenly by the factor, and factor 1 disables the path.
    #[test]
    fn downscale_config_validation() {
        // 48x48 = 3x3 macroblocks: not divisible by 2.
        let enc = Vp6CodecEncoder::new(&params(48, 48), 48, 48, 32, 12).expect("encoder");
        assert!(enc.with_downscale(2).is_err());
        // 24x24 is not macroblock-aligned.
        let enc = Vp6CodecEncoder::new(&params(24, 24), 24, 24, 32, 12).expect("encoder");
        assert!(enc.with_downscale(2).is_err());
        // Factor 1 is a no-op and always valid.
        let enc = Vp6CodecEncoder::new(&params(48, 48), 48, 48, 32, 12).expect("encoder");
        assert!(enc.with_downscale(1).is_ok());
    }
}

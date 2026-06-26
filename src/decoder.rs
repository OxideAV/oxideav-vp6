//! `oxideav-core` codec registration — wires the top-level
//! [`crate::decode_frame::Vp6Decoder`] into the framework's
//! [`oxideav_core::Decoder`] trait + [`oxideav_core::CodecRegistry`].
//!
//! The registry path mirrors the `oxideav-vp8` shape: a packet-queueing
//! [`Vp6CodecDecoder`] that drains one [`Packet`] per `send_packet` and
//! produces one [`Frame::Video`] per `receive_frame`, threading the §4
//! reference frames + §9 cross-frame state internally via the
//! [`Vp6Decoder`] state machine. The codec registers under the canonical
//! id `"vp6"` with the container FourCC / Matroska tags VP6 carries
//! (`VP60` / `VP61` / `VP62` / `vp6f` / `V_VP6`).
//!
//! ## Coverage
//!
//! The driver decodes the frame shape the in-tree encoders produce
//! (Simple/VP6.0, single-partition, BoolCoder coefficients, no
//! probability-update sub-streams). Frames signalling `MultiStream`,
//! `UseHuffman`, or in-header probability updates surface as
//! [`oxideav_core::Error`]`::invalid` until a conformant `.vp6` fixture
//! pins the Figure-5 update-substream ordering (see the `decode_frame`
//! module docs). An inter frame fed before any keyframe is likewise a
//! stream-mis-feed error, not a missing feature.
//!
//! ## Provenance
//!
//! Pure framework plumbing over this crate's own already-spec-sourced
//! decode pipeline. No external library code was consulted.

use std::collections::VecDeque;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame, Packet, Result, RuntimeContext, VideoFrame,
};

use crate::decode_frame::Vp6Decoder;
use crate::frame_assembly::Frame as Vp6Frame;
use crate::Error as Vp6Error;

/// Canonical codec id under which this decoder registers — matches the
/// `oxideav-vp8` / `oxideav-vp9` naming convention.
pub const VP6_CODEC_ID: &str = "vp6";

impl From<Vp6Error> for CoreError {
    fn from(value: Vp6Error) -> Self {
        match value {
            Vp6Error::Truncated => CoreError::invalid(value.to_string()),
            // "Not implemented in this round" maps to `unsupported`: the
            // bitstream is well-formed, the decoder just doesn't cover
            // this shape yet (MultiStream / Huffman / pre-keyframe inter).
            Vp6Error::NotImplemented => CoreError::unsupported(value.to_string()),
        }
    }
}

impl From<&Vp6Frame> for VideoFrame {
    fn from(f: &Vp6Frame) -> Self {
        // §2 4:2:0: the luma plane stride is the luma width; each chroma
        // plane is the macroblock-grid width. The plane `samples()` are
        // already row-major at those widths.
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: f.y.width(),
                    data: f.y.samples().to_vec(),
                },
                VideoPlane {
                    stride: f.u.width(),
                    data: f.u.samples().to_vec(),
                },
                VideoPlane {
                    stride: f.v.width(),
                    data: f.v.samples().to_vec(),
                },
            ],
        }
    }
}

/// `oxideav_core::Decoder` impl driving the top-level [`Vp6Decoder`].
///
/// Each `send_packet` queues one compressed VP6 frame; the next
/// `receive_frame` pops it and feeds it to [`Vp6Decoder::decode_packet`],
/// which threads the §4 reference frames + §9 cross-frame profile/version
/// across calls. The first packet fed must be a keyframe.
pub struct Vp6CodecDecoder {
    codec_id: CodecId,
    pending: VecDeque<Packet>,
    eof: bool,
    state: Vp6Decoder,
}

impl Vp6CodecDecoder {
    /// Build a fresh decoder bound to `codec_id`, with empty carry-state.
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            pending: VecDeque::new(),
            eof: false,
            state: Vp6Decoder::new(),
        }
    }
}

impl std::fmt::Debug for Vp6CodecDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp6CodecDecoder")
            .field("codec_id", &self.codec_id)
            .field("pending", &self.pending.len())
            .field("eof", &self.eof)
            .field("has_reference", &self.state.has_reference())
            .finish()
    }
}

impl Decoder for Vp6CodecDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending.push_back(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.pop_front() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let decoded = self.state.decode_packet(&pkt.data)?;
        let mut vf: VideoFrame = (&decoded).into();
        vf.pts = pkt.pts;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.pending.clear();
        self.eof = false;
        self.state.reset();
        Ok(())
    }
}

/// `make_decoder` factory plugged into the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp6CodecDecoder::new(params.codec_id.clone())))
}

/// Register the VP6 codec into `reg` under the canonical [`VP6_CODEC_ID`],
/// with the container tags VP6 commonly carries: the On2 FourCCs
/// `VP60` / `VP61` / `VP62`, the Flash `vp6f` FourCC, and the Matroska
/// `V_VP6` id.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video(VP6_CODEC_ID)
        .with_lossy(true)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(VP6_CODEC_ID))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(crate::encoder::make_encoder)
            .tags([
                CodecTag::fourcc(b"VP60"),
                CodecTag::fourcc(b"VP61"),
                CodecTag::fourcc(b"VP62"),
                CodecTag::fourcc(b"vp6f"),
                CodecTag::matroska("V_VP6"),
            ]),
    );
}

/// Unified entry point — installs the VP6 codec into a [`RuntimeContext`].
/// Called by `oxideav-meta` via the [`oxideav_core::register!`] hook in
/// `lib.rs`.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_encode::encode_intra_frame;
    use oxideav_core::{CodecId, Packet, TimeBase};

    fn pattern_frame(hf: usize, vf: usize) -> Vp6Frame {
        let mut frame = Vp6Frame::new(hf, vf);
        let yw = frame.y.width();
        let yh = frame.y.height();
        for r in 0..yh {
            for c in 0..yw {
                frame.y.samples_mut()[r * yw + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        let uw = frame.u.width();
        let uh = frame.u.height();
        for r in 0..uh {
            for c in 0..uw {
                frame.u.samples_mut()[r * uw + c] = (128 + ((r + c) % 40) as i32 - 20) as u8;
                frame.v.samples_mut()[r * uw + c] = (128 - ((r * 2 + c) % 40) as i32 + 20) as u8;
            }
        }
        frame
    }

    /// A keyframe packet decodes end-to-end through the registered
    /// `Decoder` trait surface (`send_packet` → `receive_frame`),
    /// producing a 3-plane 4:2:0 `VideoFrame`.
    #[test]
    fn decoder_trait_decodes_keyframe_packet() {
        let src = pattern_frame(4, 4);
        let bytes = encode_intra_frame(&src, 48).expect("encode");

        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        let pkt = Packet::new(0, TimeBase::new(1, 1000), bytes);
        dec.send_packet(&pkt).expect("send");
        let frame = dec.receive_frame().expect("decode");

        match frame {
            Frame::Video(vf) => {
                assert_eq!(vf.planes.len(), 3, "4:2:0 → Y/U/V");
                assert_eq!(vf.planes[0].stride, 32, "luma stride = 4 frags * 8");
                assert_eq!(vf.planes[0].data.len(), 32 * 32);
            }
            other => panic!("expected video frame, got {other:?}"),
        }
    }

    /// With no queued packet and no EOF, the decoder asks for more input.
    #[test]
    fn decoder_needs_more_when_empty() {
        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
        dec.flush().expect("flush");
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    /// The codec registers under "vp6" and resolves back via the registry.
    #[test]
    fn registers_under_canonical_id() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::video(CodecId::new(VP6_CODEC_ID));
        let dec = reg.first_decoder(&params).expect("make decoder");
        assert_eq!(dec.codec_id().as_str(), VP6_CODEC_ID);
    }

    /// `reset` clears carry-state so a fresh keyframe can be decoded.
    #[test]
    fn reset_clears_state() {
        let src = pattern_frame(2, 2);
        let bytes = encode_intra_frame(&src, 40).expect("encode");
        let mut dec = Vp6CodecDecoder::new(CodecId::new(VP6_CODEC_ID));
        let pkt = Packet::new(0, TimeBase::new(1, 1000), bytes);
        dec.send_packet(&pkt).expect("send");
        dec.receive_frame().expect("decode");
        dec.reset().expect("reset");
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
    }
}

//! Pure-Rust VP6 video decoder (FFmpeg-port).
//!
//! See the crate README for implementation status. The authoritative
//! reference for the bitstream is FFmpeg's `libavcodec/vp56.c` +
//! `libavcodec/vp6.c` — VP6 has no public written spec.

#![deny(missing_debug_implementations)]
// Bitstream ports keep mathematical row/col layouts legible: allow the
// "0 * 8 + col" style inside idct / MC kernels.
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
// Port kernels are inherently long-parameter — tolerate the lint.
#![allow(clippy::too_many_arguments)]
// Loop variables that double as array indices are the norm in DSP code.
#![allow(clippy::needless_range_loop)]

pub mod decoder;
pub mod dsp;
pub mod encoder;
pub mod frame_header;
pub mod mb;
pub mod models;
pub mod range_coder;
pub mod tables;

use oxideav_codec::{CodecInfo, CodecRegistry, Decoder, DecoderFactory};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};

pub use decoder::{Vp6Decoder, Vp6Variant};
pub use encoder::Vp6Encoder;
pub use frame_header::{FrameHeader, FrameKind};
pub use range_coder::{RangeCoder, RangeEncoder};
pub use tables::Vp56Mb;

/// Stable codec-id strings.
pub const CODEC_ID_VP6F: &str = "vp6f";
pub const CODEC_ID_VP6A: &str = "vp6a";

/// Decoder factory — see the [`oxideav_codec`] registry for the
/// integration details. Accepts both `vp6f` and `vp6a`; the `vp6a`
/// path currently errors out at `send_packet` because the alpha-plane
/// decode isn't wired up yet.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp6Decoder::new(params.clone())))
}

/// Factory value, suitable for `CodecInfo::decoder(...)` when
/// constructing a registry entry.
pub const DECODER_FACTORY: DecoderFactory = make_decoder;

/// Short-hand `CodecId` constructor for `vp6f`.
pub fn vp6f_codec_id() -> CodecId {
    CodecId::new(CODEC_ID_VP6F)
}

/// Short-hand `CodecId` constructor for `vp6a`.
pub fn vp6a_codec_id() -> CodecId {
    CodecId::new(CODEC_ID_VP6A)
}

/// Register the VP6 decoder(s) with a codec registry.
///
/// Two codec ids are registered:
/// * `vp6f` — Flash Video codec-id 4, plain YUV 4:2:0.
/// * `vp6a` — Flash Video codec-id 5, YUVA 4:2:0:4 with an alpha plane.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("vp6_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16383, 16383);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP6F))
            .capabilities(caps.clone())
            .decoder(make_decoder)
            .tags([
                CodecTag::fourcc(b"VP60"),
                CodecTag::fourcc(b"VP61"),
                CodecTag::fourcc(b"VP62"),
            ]),
    );
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP6A))
            .capabilities(caps)
            .decoder(make_decoder)
            .tags([CodecTag::fourcc(b"VP6A")]),
    );
}

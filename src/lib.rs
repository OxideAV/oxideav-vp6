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
pub mod huffman;
pub mod mb;
pub mod models;
pub mod range_coder;
pub mod tables;

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, DecoderFactory, RuntimeContext};

pub use decoder::{Vp6Decoder, Vp6Variant};
pub use encoder::Vp6Encoder;
pub use frame_header::{FrameHeader, FrameKind};
pub use range_coder::{RangeCoder, RangeEncoder};
pub use tables::Vp56Mb;

/// Stable codec-id strings.
pub const CODEC_ID_VP6F: &str = "vp6f";
pub const CODEC_ID_VP6A: &str = "vp6a";

/// Decoder factory — see the [`oxideav_codec`] registry for the
/// integration details. Accepts both `vp6f` and `vp6a`. Honours
/// [`CodecParameters::limits`] so server callers that pass a tightened
/// `CodecParameters` actually get a tightened decoder (header-parse
/// pixel cap + arena-pool size + per-arena byte cap).
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp6Decoder::with_limits(
        params.codec_id.clone(),
        *params.limits(),
    )))
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
pub fn register_codecs(reg: &mut CodecRegistry) {
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

/// Unified registration entry point: install the VP6 codec factories
/// into the codec sub-registry of a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that need
/// only the codec sub-registry can keep using [`register_codecs`].
///
/// Also auto-registered into [`oxideav_core::REGISTRARS`] via the
/// [`oxideav_core::register!`] macro below so consumers calling
/// [`oxideav_core::RuntimeContext::with_all_features`] pick VP6 up
/// without any explicit umbrella plumbing.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("vp6", register);

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_VP6F));
        let dec = ctx
            .codecs
            .make_decoder(&params)
            .expect("vp6 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_VP6F);
    }
}

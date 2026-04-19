# oxideav-vp6

Pure-Rust On2 **VP6** video decoder for oxideav. Zero C dependencies,
no FFI, no `*-sys` crates.

Covers the `vp6f` flavour used inside FLV (Flash Video). `vp6a`
(VP6 with an alpha plane) is out of scope for the initial release —
the FLV container passes the alpha-offset byte through as `extradata`
and this decoder returns `Error::Unsupported` for it.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-vp6 = "0.0"
```

## Status

VP6 has no public specification — the authoritative reference is
FFmpeg's reverse-engineered `libavcodec/vp56.c` + `libavcodec/vp6.c`.

Implemented today:

- **Range coder** (the "VP56 bool coder"): 8-bit range / bottom-up
  renormalisation + raw bit reads. Round-trip tested against a
  reference encoder in-crate.
- **Frame header parse**: version / profile / offset-to-coefficient-
  partition, keyframe width/height in macroblocks, quantiser, filter
  header (deblock-on/off, bicubic flag), golden-refresh.
- **Keyframe plane allocation**: YUV 4:2:0 frame buffers sized to
  the MB-aligned width/height.

Not yet implemented (returns `Error::Unsupported`):

- Macroblock type decode + DCT coefficient decode (intra path).
- Integer 8x8 IDCT + reconstruction.
- Inter prediction / motion compensation (4-tap sub-pel filter).
- Loop filter.
- Inter frame decode (P-frames).
- `vp6a` (alpha plane).

## Quick use

```rust
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};
use oxideav_codec::Decoder;

let params = CodecParameters::video(CodecId::new("vp6f"));
let mut dec = oxideav_vp6::Vp6Decoder::new(params);
let pkt = Packet::new(0, TimeBase::new(1, 1000), vec![/* coded frame */]);
dec.send_packet(&pkt)?;
let _frame = dec.receive_frame();
# Ok::<(), oxideav_core::Error>(())
```

## License

MIT — see [LICENSE](LICENSE).

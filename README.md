# oxideav-vp6

Pure-Rust On2 **VP6** video decoder for oxideav. Zero C dependencies,
no FFI, no `*-sys` crates.

Covers both FLV flavours:
* **vp6f** — the Flash Video codec-id-4 stream, YUV 4:2:0.
* **vp6a** — codec-id-5 with an additional full-resolution alpha plane;
  output as `Yuva420P`.

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
FFmpeg's reverse-engineered `libavcodec/vp56.c` + `libavcodec/vp6.c`
(+ `vp3dsp.c` for the IDCT / loop filter + `vp6dsp.c` for the 2D
bicubic interpolator + `vpx_rac.h` for the bool coder).

### Implemented

- **Range coder** ("VP56 bool coder"): FFmpeg-equivalent 16-bit
  renormalisation, 24-bit seed, both `get_prob(p)` and `get_bit()`
  (equiprobable) paths. Walks static [`Vp56Tree`] probability trees
  and round-trips against an in-crate encoder in unit tests.
- **Frame header parse**: keyframe (with dimensions) and inter-frame
  headers, including the "separated-coeff" offset field, sub-version
  gating for the extra filter-info block, and the interlaced-profile
  flag.
- **Model state** (`Vp6Model`): default init, MB-type-stats re-training,
  vector model updates, coefficient model updates, coefficient-reorder
  scan retuning, linear-combination rebuild of `coeff_dcct` from
  `coeff_dccv`.
- **Macroblock-type decode tree**: ports `vp56_parse_mb_type_models`
  + `vp56_parse_mb_type` over the MB-type context tree.
- **Motion-vector decode**: all 10 VP56 MB types including `INTER_4V`
  (per-8x8 MVs, averaged chroma MV), `INTER_DELTA_*`
  (`vp6_parse_vector_adjustment` with the predicted-delta tree and
  the full-delta bit layout), and the 12-position MV candidate
  predictor walk. The chosen MV is stashed back into the persistent
  `macroblocks[]` table so future neighbours see it — the missing
  piece of that write-back was the cause of inter-frame drift
  through the 0.0.1 series.
- **Coefficient decode — range path**: full port of `vp6_parse_coeff`,
  including zero-run categories, the 6-category long tree, the
  `coeff_dcct` / `coeff_ract` / `coeff_runv` trees, and the reorder
  permutation.
- **Integer 8×8 IDCT**: port of `vp3dsp.c::idct` (put + add paths),
  with the DC-only fast path (`vp3_idct_dc_add_c`).
- **Intra prediction**: DC prediction from above/left 8x8 block
  boundaries (`vp56_add_predictors_dc`), using the reference-frame
  DC context kept in `left_block[]` / `above_blocks[]`.
- **Inter motion compensation**: reference-frame MC with the 4-tap
  VP6 bicubic filter (`vp6_block_copy_filter` + `vp6_filter_diag4`)
  for luma and the H.264-style bilinear chroma filter
  (`put_h264_chroma_mc8`). Handles integer-pel, half-pel, quarter-pel
  phases. Edge pixels are mirror-clamped into a scratch tile instead
  of FFmpeg's `emulated_edge_mc` — functionally equivalent for
  interior-dominated streams.
- **VP3-style deblock loop filter**: per-block edge filter
  (`ff_vp3dsp_{h,v}_loop_filter_12`) applied to the 12x12 MC scratch
  tile before sub-pel filtering, gated by the `deblock_filtering`
  bit in the picture header (keyframe default = on, matching
  FFmpeg's `ff_vp56_init_context`). Only the VP6 variant is wired;
  VP5's separate `vp56dsp.edge_filter_{hor,ver}` path is irrelevant
  to FLV/vp6 streams.
- **`vp6a` alpha plane**: two-stream decode driven by the 3-byte
  BE24 alpha offset at the head of each packet. The primary context
  decodes YUV as usual; a second context decodes the alpha partition
  as a monochrome VP6 stream and the luma samples surface as the
  `A` plane of a `Yuva420P` output.
- **Reference-frame management**: tracks `prev_frame` and
  `golden_frame` planes inside the decoder. Keyframes overwrite both;
  inter frames refresh golden when the golden-frame flag is set.

### Not yet implemented (deferrals)

- **Huffman coefficient path** (`vp6_parse_coeff_huffman`): the older
  optional path some VP6 streams use. When the frame header signals
  `use_huffman = 1`, the decoder returns `Error::Unsupported`. Out of
  scope for the first release; the main FLV sample and most real
  streams use the range-coder path.
- **Interlaced profile**: parsed but not exercised end-to-end.

### Test coverage

The crate ships 30 library unit tests plus 4 integration tests:

- **Unit tests** for the range coder round-trip, the IDCT (DC-only flat
  block, add-zero identity), the loop filter bounding-values table and
  edge-smoothing (`h_loop_filter_smooths_sharp_edge`,
  `v_loop_filter_smooths_sharp_edge`), the H.264 chroma MC
  integer-pel fast path, model defaults, and the MB-type enum layout.
- `tests/keyframe_from_flv.rs` — walks the
  `asian-commercials-are-weird.flv` sample (skipped if absent;
  override path via `OXIDEAV_FLV_SAMPLE=...`), decodes the first
  VP6F keyframe, then runs 20 consecutive frames through the inter
  decode path and asserts all 20 decode cleanly.
- `tests/loop_filter_delta.rs` — synthetic test that renders an inter
  MB over a block-aligned reference edge with deblock-filtering off
  vs. on, asserts the filter reduces the output gradient across the
  former edge.
- `tests/vp6a_roundtrip.rs` — synthesises a vp6a packet by wrapping a
  real VP6F keyframe in the 3-byte alpha-offset prefix + duplicating
  it into the alpha partition, decodes it, and verifies a 4-plane
  YUVA frame with non-zero alpha pixels. No real vp6a FLV fixture is
  shipped in the tree; if you have one, set `OXIDEAV_FLV_SAMPLE`.

## Quick use

```rust
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};
use oxideav_codec::Decoder;

let params = CodecParameters::video(CodecId::new("vp6f"));
let mut dec = oxideav_vp6::Vp6Decoder::new(params);
let pkt = Packet::new(0u32, TimeBase::new(1, 1000), vec![/* coded frame */]);
dec.send_packet(&pkt)?;
let _frame = dec.receive_frame();
# Ok::<(), oxideav_core::Error>(())
```

## License

MIT — see [LICENSE](LICENSE).

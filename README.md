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

The authoritative reference is the On2 VP6 *Bitstream & Decoder
Specification* (vendored in this workspace at
`docs/video/vp6/vp6_format.pdf`); historically work also cross-checked
FFmpeg's reverse-engineered `libavcodec/vp56.c` + `libavcodec/vp6.c`
(+ `vp3dsp.c` for the IDCT / loop filter + `vp6dsp.c` for the 2D
bicubic interpolator + `vpx_rac.h` for the bool coder), but encoder
audit rounds (r19+) lean on the spec directly. r22 lands the spec
page 28 Table 5 ctx mapping for `vector_predictors` (`ctx = 2 -
nb_pred`) on both decoder + encoder sides. **r23 unblocks ffmpeg
inter-frame interop**: encoder default `sub_version` is now 6
(`Vp3VersionNo = VP6.0` per spec Table 2), was 0 — the spec-forbidden
zero was lenient on ffmpeg's keyframe path but mis-routed the inter
parser, surfacing as the long-running "Invalid data" inter error.
**r24 wires real inter residual coefficients**: `encode_inter_frame`
now materialises the integer-pel MC prediction, computes the per-pixel
residual against the source, runs the forward DCT (residual mode — no
`-128` bias), quantises DC + AC, runs the `RefKind::Previous` DC
predictor mirror, and emits the result through the same
`emit_block_coefs` state machine that already drove the keyframe
path. Internal-decoder PSNR on a flat-baseline + per-MB brightness-
shift fixture jumps from ~19 dB (MC-only) to ~43 dB (with residual).
ffmpeg-side residual interop still diverges (the inter decode lands on
the MC-only baseline, suggesting a per-MB coefficient model state
mismatch downstream of the keyframe-time `0x80` defaults) — left for
r25+. **r25 adds quarter-pel sub-pel motion estimation.**
`motion_search` now seeds the integer-pel SAD winner, then evaluates
every quarter-pel offset in a `±3 qpel` window around it via the same
H.264-chroma-style bilinear filter the decoder uses
(`put_h264_chroma8` mirror) plus a Lagrangian rate cost on the MV
bits. The MC tile sampling path (`sample_mc_tile`) likewise grew a
sub-pel branch so the residual computation matches the decoder
exactly. On translating-stripe / translating-disk fixtures with a
0.5-pel shift, internal-decoder Y PSNR climbs from ~19-29 dB
(integer-pel MC alone) to 35-37 dB (qpel MC + DCT residual). ffmpeg
cross-decodes the qpel-MV inter packet cleanly (Y PSNR ~32 dB on the
stripes fixture). **r26 wires golden-frame refresh + per-MB
golden-vs-previous selection** via the new
`Vp6Encoder::encode_inter_frame_with_golden(prev_*, golden_*, new_*,
…)` method. Cadence-driven by `golden_refresh_period` (default 30):
when fired, the picture-header `golden_frame_flag` bit is set so the
decoder snaps the just-decoded frame into its `golden_frame` slot.
Per MB the encoder runs `motion_search` against both refs and picks
the lower Lagrangian cost (SAD + λ * mv_bits), emitting one of
{`InterNoVecPf`, `InterDeltaPf`, `InterNoVecGf`, `InterDeltaGf`}
accordingly. On a 5-frame A,B,A,B,A periodic loop (32x32, QP 12)
pinning golden to the keyframe brings the inter-frame total wire size
from 378 bytes to 282 bytes (~25% smaller) vs refreshing every frame
— the loop-back A frames pick golden for every MB and emit
near-zero residual. ffmpeg cross-decodes the key + golden-refresh
inter pair cleanly.

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
- **Encoder golden-frame refresh** (r26+):
  `Vp6Encoder::encode_inter_frame_with_golden(prev_*, golden_*,
  new_*, …)` accepts both refs and per-MB picks whichever beats the
  other on a Lagrangian SAD cost. Cadence-driven by the public
  `golden_refresh_period: u32` (default 30); when fired the
  picture-header `golden_frame_flag` bit is set so the decoder snaps
  the just-decoded frame into its `golden_frame` slot. MB types are
  one of `{InterNoVecPf, InterDeltaPf, InterNoVecGf, InterDeltaGf}`
  with separate prev / golden MV-candidate state, and the per-block
  DC predictor mirror tracks `RefKind::Golden` alongside
  `RefKind::Previous`. ffmpeg cross-decodes the refresh-flag inter
  packet cleanly. Reduces wire size on periodic-structure content
  (animation loop, slideshow) — see CHANGELOG / fixture results.

### Not yet implemented (deferrals)

- **Huffman coefficient path** (`vp6_parse_coeff_huffman`): the older
  optional path some VP6 streams use. When the frame header signals
  `use_huffman = 1`, the decoder returns `Error::Unsupported`. Out of
  scope for the first release; the main FLV sample and most real
  streams use the range-coder path.
- **Interlaced profile**: parsed but not exercised end-to-end.

### Test coverage

The crate ships 41 library unit tests plus 36 integration tests
across 7 files (77 tests total):

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
- `tests/ffmpeg_interop.rs` — external-ffmpeg interop guards. Skipped
  silently when `ffmpeg` isn't on `PATH`. As of r23 every guard
  asserts ffmpeg accepts both packets in a 2-tag (key + inter) stream
  (`n == 2`), covering the keyframe path
  (`ffmpeg_accepts_keyframe`), the skip-frame inter path
  (`ffmpeg_decodes_keyframe_in_two_tag_stream`), the motion-search
  inter path (`r21_inter_frame_ffmpeg_decode_state`), and the
  spec-legal `Vp3VersionNo` byte
  (`keyframe_vp3_version_no_is_spec_legal`).
- `tests/dump_inter.rs` — opt-in `VP6_DUMP_INTER=1` diagnostic that
  writes a 2-tag FLV to `/tmp/oxideav_vp6_dump.flv` for ffmpeg-side
  manual inspection, plus a `inter_buff2_offset_is_spec_compliant`
  guard that pins the spec layout of the partition-offset field.
- `tests/encoder_roundtrip.rs::r24_inter_residual_psnr_floor` (new in
  r24) — encodes a flat keyframe + per-MB brightness-shift inter
  through `encode_inter_frame`, decodes the 2-tag stream through the
  in-tree decoder, and asserts the inter-frame Y PSNR clears 30 dB
  AND beats the MC-only baseline by ≥5 dB. Fails immediately on a
  regression that drops the residual coefficient path back to the
  pre-r24 zero-block shortcut.
- `tests/encoder_roundtrip.rs::r25_qpel_translating_stripes_psnr_clears_35db`
  / `r25_qpel_translating_disk_psnr_clears_35db` (new in r25) — pin
  the quarter-pel ME path against a 0.5-pel sub-pel translation of a
  smooth low-frequency stripe / Gaussian-disk fixture. Both assert
  internal-decoder Y PSNR ≥ 35 dB; the integer-only baseline (MC
  alone, no qpel) is ~19-29 dB so the qpel ME contribution is
  unmistakable. `r25_ffmpeg_decodes_qpel_inter_frame` cross-decodes
  the stripes packet through ffmpeg's vp6f decoder and asserts ≥ 20
  dB Y PSNR, confirming the qpel MV bits parse cleanly.
- `tests/encoder_roundtrip.rs::golden_refresh_*` (new in r26) — five
  guards on the new `encode_inter_frame_with_golden` API:
  `golden_refresh_cadence_fires_on_period` pins the counter
  semantics; `golden_refresh_disabled_at_period_zero` covers the
  `period = 0` disabled branch; `golden_refresh_loop_back_uses_
  golden_reference` walks an A→B→A loop and asserts our decoder
  reconstructs frame 2 at ≥ 25 dB Y PSNR (45 dB observed) vs an 8.6
  dB skip-from-prev baseline; `golden_refresh_reduces_bytes_on_
  periodic_loop` encodes a 5-frame A,B,A,B,A loop twice (chasing-
  golden vs pinned-golden) and pins the pinned-golden total wire
  size to ≤ 110% of chasing-golden (282 vs 378 bytes observed);
  `ffmpeg_decodes_inter_with_golden_refresh_flag` cross-decodes a
  key + golden-refresh inter pair through ffmpeg's vp6f decoder
  (must accept both packets).

## Quick use

```rust
use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_codec::Decoder;

let mut dec = oxideav_vp6::Vp6Decoder::new(CodecId::new("vp6f"));
let pkt = Packet::new(0u32, TimeBase::new(1, 1000), vec![/* coded frame */]);
dec.send_packet(&pkt)?;
let _frame = dec.receive_frame();
# Ok::<(), oxideav_core::Error>(())
```

For server / sandbox callers, use `Vp6Decoder::with_limits(codec_id, limits)`
to thread an explicit `oxideav_core::DecoderLimits` through. The decoder
honours `max_pixels_per_frame` (header-parse pixel cap),
`max_arenas_in_flight` (arena-pool size; natural backpressure when full),
and `max_alloc_bytes_per_frame` (per-arena byte cap, clamped to a VP6
ceiling of 8 MiB). The `Decoder::receive_arena_frame` override returns
true zero-copy frames whose plane bytes live inside the leased arena
buffer.

## License

MIT — see [LICENSE](LICENSE).

# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 7 (2026-05-25).** The orphan-rebuild
scaffold from 2026-05-18 is being replaced incrementally by parsers
sourced exclusively from
[On2 Technologies' VP6 Bitstream & Decoder Specification](https://github.com/OxideAV/oxideav/blob/master/docs/video/vp6/vp6_format.pdf)
(document version 1.02, August 2006). No third-party VP6
implementation is consulted at any stage.

### What round 1 lands

- `Vp6FrameHeader::parse` — frame-header raw-bit prefix:
  - Table 1 (`FrameType`, `DctQMask`, `MultiStream`)
  - Table 2 R(n) fields (`Vp3VersionNo`, `VpProfile`, `Reserved`,
    conditional `Buff2Offset`)
- Typed `CodingProfile` (Simple / Advanced / Reserved) and
  `Vp3Version` (VP6.0 / 6.1 / 6.2 / Other) enums.
- `Error::Truncated` for short-input failures.

### What round 2 lands

- `DequantContext` — per-frame inverse-quantization context (spec
  §15). Resolves the DC and AC scalar quantizer factors from the
  header's `DctQMask` via the two 64-entry tables
  (`DC_QUANTIZATION_TABLE` / `AC_QUANTIZATION_TABLE`), then
  dequantizes a block with `dequantize_block` / `dequantize_coeff`.
- This layer is **BoolCoder-independent** (it reads only the raw-bit
  `DctQMask`), so it advances past round 1 without touching the
  blocked §7.3 `Split` formula.

### What round 3 lands

- `idct_block` — the spec §16 inverse DCT transform: a separable,
  fixed-point integer IDCT (14-bit precision, seven Q16 cosine
  constants) that turns an 8x8 block of dequantized coefficients in
  raster order back to pixel / pixel-difference values via a row pass
  and a column pass.
- This stage is the natural successor to dequant (§15 → §16): it
  consumes `DequantContext::dequantize_block`'s output and, like that
  layer, reads **no BoolCoder bits**, so it lands without touching the
  blocked §7.3 `Split` formula. Clamping and the intra `+128` level
  shift belong to §17 frame reconstruction, a later round.

### What round 4 lands

- `reconstruct_intra_block` / `intra_block_to_pixels` — the spec §17.1
  intra-block reconstruction step. For each of the 64 post-IDCT samples
  in raster order: `OutputValue = InputValue + 128`, then inclusive
  clip to `0..=255`. Inverts the encoder-side level shift that §17.1
  documents ("prior to encoding the value 128 is subtracted from all
  data samples").
- This is the natural successor to the §16 IDCT for the intra-coded
  path. Like the §15 dequant and §16 IDCT layers it reads **no
  BoolCoder bits** — it operates on a finished post-IDCT 8x8 block — so
  it advances the decoder past round 3 without touching the contested
  §7.3 `Split` formula.
- The remaining §17.2–§17.4 cases (zero MV, full-pixel MV, sub-pixel MV)
  combine the same clip with motion compensation against a reference
  reconstruction buffer; they are blocked on the BoolCoder for MV
  decoding upstream.

### What round 5 lands

- `interp` — the spec §11.4 fractional-pixel motion-compensation
  interpolation filters. The bilinear 2-tap kernel (`bilinear_point`) and
  4-tap bicubic kernel (`bicubic_point`), their full tap tables
  (`BILINEAR_LUMA_FILTERS` `[4][2]`, `BILINEAR_CHROMA_FILTERS` `[8][2]`,
  `BICUBIC_FILTER_SET` `[17][8][4]`), their separable two-pass 8x8 block
  applicators (`bilinear_block` / `bicubic_block`), and the §11.4
  `Var16Point` prediction-block variance metric (`var_16_point`) used by
  the Advanced-Profile filter selector. Each tap set sums to 128 and the
  kernels descale by `(Σ + 64) >> 7`; the bicubic kernel clips its
  output to `0..=255` per §11.4.2.
- These produce the interpolated sub-pixel prediction samples that §17.4
  reconstruction consumes. Given a reference buffer, stride and fractional
  phase the kernels are pure integer pixel arithmetic and read **no
  BoolCoder bits**, so — like §15 dequant, §16 IDCT and §17.1 intra
  reconstruction — this stage advances the decoder without touching the
  contested §7.3 `Split` formula. The motion vector that *selects* the
  filter phase and source position is BoolCoder-gated upstream.
- **DOCS-GAP (selector only):** §11.4's Advanced-Profile filter-size
  selector reads `FilterMvSizeThresh = ((MAX_MV_EXTENT >> 1) + 1) << 2`,
  but `MAX_MV_EXTENT` is never assigned a numeric value anywhere in the
  document. The per-point kernels, tap tables and the variance half of
  the selector are fully specified and landed; the size-threshold
  selector is deferred until the constant is supplied.

### What round 6 lands

- `reconstruct_inter_block` / `inter_block_to_pixels` — the spec
  §17.2–§17.4 inter-block reconstruction step. For each of the 64
  samples: `OutputValue = PredictionValue + PredictionError`, then
  inclusive clip to `0..=255`. One function for all three inter cases:
  §17.2 (zero motion vector), §17.3 (full-pixel-aligned motion vector)
  and §17.4 (fractional-pixel motion vector) share the byte-identical
  recombination pseudocode — they differ only in how the 8x8 prediction
  block is *sourced*. No `+128` intra level shift here: the prediction
  already carries the DC.
- `fetch_prediction_block` — the §17.2/§17.3 integer-offset prediction
  fetch. A straight copy of an 8x8 region from a reference reconstruction
  buffer at an integer `(dx, dy)` offset: `(0, 0)` for the §17.2 zero
  vector (co-located), the integer whole-sample offset for §17.3 (full
  pixel). §17.4 instead sources its prediction from the round-5 §11.4
  filters (`bilinear_block` / `bicubic_block`).
- `MvShift::{Luma, Chroma}` plus `whole_sample_aligned`, `luma_frac` and
  `chroma_frac` — the §11.4 motion-vector decomposition:
  `WholeSampleAligned = MvComponent >> MvShift` (arithmetic shift floors
  negatives) plus the low-`MvShift`-bit fractional phase. `MvShift` is 2
  for luma (¼-pixel precision, 4 phases) and 3 for chroma (⅛-pixel
  precision, 8 phases), matching the `BILINEAR_LUMA_FILTERS[4]` /
  `BILINEAR_CHROMA_FILTERS[8]` row counts.
- This stage is the natural successor to §17.1 (intra) and §11.4
  (interpolation): it composes them into the full inter recombination.
  Like the §15 dequant, §16 IDCT, §17.1 intra and §11.4 interpolation
  layers it reads **no BoolCoder bits** — given an already-decoded MV, a
  reference buffer and the IDCT residual, every step is pure integer
  pixel arithmetic — so it advances the decoder past round 5 without
  touching the contested §7.3 `Split` formula. The motion vector that
  drives the fetch/interpolation phase is decoded upstream, behind the
  BoolCoder.

### What round 7 lands

- `loopfilter` — the spec §11.3 prediction loop filter. VP6 has no
  traditional in-loop deblocking filter on the reconstruction buffer;
  instead, when a non-zero motion vector produces a prediction block
  that straddles an 8x8 boundary in the reference frame, the samples
  on either side of that boundary in the **prediction** signal are
  deblocked into a temporary buffer before §11.4 fractional-pixel
  interpolation runs.
- `PREDICTION_LOOP_FILTER_LIMIT_VALUES[64]` — the quantizer-indexed
  `FLimit` table (`[0]=30`, `[63]=1`, monotonically non-increasing),
  selected by the frame's raw-bit `DctQMask`.
- `boundary_x` / `boundary_y` — the §11.3 `(8 - (mV & 7)) & 7`
  block-edge offset calculation that locates the straddling boundary
  inside the prediction block from the whole-sample-aligned MV
  components (`mV >> MvShift`, already provided by round 6's
  `inter::whole_sample_aligned`).
- `bound(FLimit, FiltVal)` — the §11.3 soft-clip: linear passthrough
  in `|FiltVal| < FLimit`, symmetric taper toward `0` across
  `[FLimit, 2*FLimit)`, hard zero at `|FiltVal| >= 2*FLimit`. The taper
  is what preserves real reference-frame edges (large cross-boundary
  gradient → zeroed filter response) while smoothing block-boundary
  discontinuities (small gradient relative to the limit → linear
  smoothing).
- `prediction_loop_filter_function(buf, boundary_offset, step, pitch,
  points, current_quantizer_index)` — the per-edge applicator
  implementing the §11.3 `(1, -3, 3, -1)` 4-tap filter with `+ 4 ) >> 3`
  rounding-and-descale, the `Bound()` soft-clip and the per-sample
  `Clamp0To255` writes on `Src[-Step]` and `Src[0]`.
- `filter_vertical_boundary` / `filter_horizontal_boundary` — 2D
  wrappers that select `step=1, pitch=stride` (vertical) or
  `step=stride, pitch=1` (horizontal) and sweep the 8-sample edge.
- Per the spec, only the deblocking filter is implemented; the
  deringing variant carries the spec's own "not currently supported by
  the decoder (see Table 3)" rider.
- Like §15/§16/§17.1/§11.4/§17.2–§17.4 this stage reads **no BoolCoder
  bits** — given a whole-sample-aligned MV, a prediction buffer and
  the frame's `DctQMask`, every step is pure integer pixel arithmetic
  — so it advances the decoder past round 6 without touching the
  contested §7.3 `Split` formula. `UseLoopFilter` itself is a raw-bit
  frame-header field; the per-profile gate (disabled in Simple Profile,
  read from the header in Advanced) is a caller-side concern.

### What round 8 lands

- `umv` — the spec §11.5 Unrestricted Motion Vector (UMV) border
  extension. VP6 permits motion vectors that address prediction blocks
  beyond the borders of the decoded image; before any inter block is
  reconstructed against a reference frame, that reference frame's
  reconstruction buffer is extended by 48 sample points in all four
  directions, with the borders filled by pure edge replication. The
  result is that an out-of-image fetch reads the original image's
  nearest edge sample — the well-defined "clamp" semantics
  `inter::fetch_prediction_block` and the §11.4 interpolation filters
  expect (and that `inter`'s round-6 commentary explicitly defers to
  the §11.5 border).
- `UMV_BORDER_SIZE` — the 48-sample constant the spec mandates
  ("the reconstruction buffers are extended by 48 sample points in all
  directions").
- `extended_stride(width)` / `extended_height(height)` /
  `origin_offset(stride)` — geometry helpers for the extended buffer
  layout: `stride = width + 2*48`, `rows = height + 2*48`, and the
  original-image origin sits at `48 * stride + 48` in the linear
  buffer.
- `extend_border(buf, width, height)` — the in-place §11.5 applicator.
  Performs the extension in the spec-mandated order: first horizontal
  (every original-image row's left and right 48-sample borders take
  the row's leftmost / rightmost original-column value), then vertical
  (each top / bottom border row is a row-wide copy of the topmost /
  bottommost horizontally-extended row). The "first in x, then in y"
  ordering is what makes the four 48×48 corner quadrants uniform at
  the corresponding corner-pixel value of the original image, per the
  spec's Figure 13.
- `build_extended_buffer(image, width, height)` — convenience that
  allocates a `Vec<u8>` of the right size, copies the raster-order
  `image` plane into the inner rectangle, runs `extend_border`, and
  returns `(buf, stride, origin)` ready to hand to
  `inter::fetch_prediction_block`.
- Like §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3 this stage reads **no
  BoolCoder bits** — it is pure edge-replication pixel arithmetic on
  an already-reconstructed frame buffer — so it advances the decoder
  past round 7 without touching the contested §7.3 `Split` formula.

### What rounds 1–8 do NOT land

- Anything downstream of the BoolCoder switch in the frame header
  (`VFragments`, `HFragments`, scaling, filter selectors,
  `UseHuffman`), plus mode/MV decoding and DCT-token decoding — every
  one of these is `b(n)`/`B(x)`/`T` BoolCoder-coded. Blocked on a
  DOCS-GAP against spec §7.3 — the
  `Split = 1 + (((Range-1) * Probability) >> 7)` formula collapses
  the prob-128 (`b(n)`) decoder path to always-0 (the `Bit=1` branch
  yields `Range == 0`, a dead coder) and overflows `u32` when
  `Probability > 128`. The fix is either a confirmation that `>> 7`
  is correct alongside an encoder-side mapping explanation, or a
  correction to `>> 8` (matching the VPx-family arithmetic coder
  pattern). See the crate-root docs for the full report.

## License

MIT — see [LICENSE](./LICENSE).

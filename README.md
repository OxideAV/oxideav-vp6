# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 5 (2026-05-24).** The orphan-rebuild
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

### What rounds 1–5 do NOT land

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

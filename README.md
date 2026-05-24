# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 3 (2026-05-24).** The orphan-rebuild
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

### What rounds 1–3 do NOT land

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

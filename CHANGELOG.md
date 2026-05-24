# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (clean-room round 4, 2026-05-24)

- `reconstruct_intra_block` / `intra_block_to_pixels` — the spec §17.1
  intra-block reconstruction step. For each of the 64 post-IDCT samples
  in raster order: `OutputValue = InputValue + 128`, then inclusive
  clip to `0..=255` (per §17.1's `If OutputValue < 0 { 0 } else if
  OutputValue > 255 { 255 }`). Inverts the encoder-side level shift
  §17.1 documents (the encoder subtracts 128 from every sample before
  the forward DCT). Transcribed verbatim from
  `docs/video/vp6/vp6_format.pdf` §17.1. The natural successor to the
  §16 IDCT for the intra path; like §15 and §16 it reads no BoolCoder
  bits, so it advances past round 3 without touching the contested
  §7.3 `Split` formula.
- `INTRA_DC_LEVEL_SHIFT` / `PIXEL_MIN` / `PIXEL_MAX` public constants
  so the matching encoder-side path can share the single source of
  truth for `128` and `0..=255`.
- 10 unit tests over the §17.1 stage: level-shift constant value,
  pixel-range constants, all-zero block reconstructs to flat mid-grey
  (128), in-range positive inputs pass through with no clip, in-range
  negatives pass through with no clip, far-negative inputs clip to 0,
  far-positive inputs clip to 255, inclusive-boundary behaviour at
  `-128` / `-129` / `127` / `128`, per-sample independence (no
  inter-sample state), wrapper-vs-dual-buffer parity, and an
  integration test that drives the §17.1 stage from a real `idct_block`
  output (DC-only flat block stays flat after reconstruction).

### Added (clean-room round 3, 2026-05-24)

- `idct_block` — the spec §16 inverse DCT transform: a separable,
  fixed-point integer IDCT (14-bit precision; seven Q16 cosine
  constants `xC1S7`…`xC7S1`) that converts an 8x8 block of dequantized
  coefficients in raster order back to pixel / pixel-difference values
  via a row pass followed by a column pass (the column pass applies the
  transform's `>> 4` output descale). Transcribed verbatim from
  `docs/video/vp6/vp6_format.pdf` §16. Like the §15 dequant layer this
  stage reads no BoolCoder bits — it consumes the output of
  `DequantContext::dequantize_block` — so it advances the decoder
  without depending on the §7.3 `Split` defect.
- 8 unit tests over the IDCT: exact cosine-constant transcription,
  all-zero passthrough, DC-only flatness (with the expected uniform
  value recomputed term-by-term), negative-DC sign preservation, that
  every one of the 64 input coefficients participates in the
  butterfly, that an AC-only block is non-flat, and that purely
  horizontal / vertical AC inputs produce no vertical / horizontal
  variation respectively (separability check).

### Spec note (clean-room round 3)

- The §16 column-pass pseudocode renders the `_Bd` assignment with an
  unbalanced parenthesis (`_Bd = ((xC4S4 * (_B - _D)>>16)` — one `(`
  short of closing the product before `>>16`). The row pass gives the
  same quantity correctly as `((xC4S4 * (_B - _D))>>16)`, and the two
  passes are structurally identical butterflies, so this is a document
  transcription artefact, not a semantic difference; `idct_block` uses
  the balanced form in both passes.

### Added (clean-room round 2, 2026-05-23)

- `DequantContext` — per-frame inverse-quantization context (spec
  §15). Resolves the DC and AC scalar quantizer factors from the
  frame header's `DctQMask` (already parsed by round 1) via the two
  64-entry tables `DC_QUANTIZATION_TABLE` / `AC_QUANTIZATION_TABLE`,
  and dequantizes a block of coefficients via `dequantize_block` /
  `dequantize_coeff`. Transcribed verbatim from
  `docs/video/vp6/vp6_format.pdf` §15. This layer reads only the
  raw-bit `DctQMask` and never calls the BoolCoder, so it advances
  the decoder without depending on the §7.3 `Split` defect.
- 9 unit tests over the dequant tables and context: table sizes,
  spec endpoints, monotonicity, factor resolution, 6-bit mask
  clamping, DC-vs-AC selection, sign/extreme-magnitude handling, and
  full-block coverage of all 64 coefficients (the test for index 63
  guards against the §15 pseudocode's `i < 63` off-by-one).

### Spec note (clean-room round 2)

- §15's dequantization pseudocode `for(i=1;i<63;i++)` leaves
  coefficient 63 un-dequantized, contradicting the §15 prose ("each
  of its 64 coefficients", "all 63 of the AC coefficients"). The
  prose is internally consistent (64 = 1 DC + 63 AC); the pseudocode
  bound is a clear off-by-one. We follow the prose and dequantize all
  64 coefficients.

### Added (clean-room round 1, 2026-05-21)

- `Vp6FrameHeader::parse` — raw-bit prefix parser covering spec §9
  Table 1 (`FrameType`, `DctQMask`, `MultiStream`) plus Table 2's
  four R(n) fields (`Vp3VersionNo`, `VpProfile`, `Reserved`,
  conditional `Buff2Offset`). Sourced exclusively from
  `docs/video/vp6/vp6_format.pdf` (On2 Technologies, document
  version 1.02, August 2006).
- `CodingProfile`, `Vp3Version`, `FrameType` typed enums surfacing
  the spec's declared encodings (Simple/Advanced, VP6.0/6.1/6.2)
  alongside `Reserved(u8)` / `Other(u8)` escape hatches for
  out-of-spec values so policy can live in the caller.
- `Error::Truncated` variant for under-length input.
- 8 unit tests over hand-encoded byte sequences covering all four
  branches of the Buff2Offset gate plus truncated-input paths.

### Pending

- `b(n)` (BoolCoder) decoding for the rest of the frame header —
  `VFragments`, `HFragments`, `OutputVFragments`, `OutputHFragments`,
  `ScalingMode`, prediction-filter selectors, `UseHuffman` — blocked
  on a DOCS-GAP against spec §7.3's `Split = 1 + (((Range-1) *
  Probability) >> 7)` formula. See the crate-root docs for the
  detailed report; the formula as written collapses prob-128
  decisions to all-zeros and overflows for probability > 128.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

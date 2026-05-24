# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (clean-room round 8, 2026-05-25)

- `umv` module — the spec §11.5 Unrestricted Motion Vector (UMV)
  border extension. VP6 permits motion vectors that address prediction
  blocks beyond the borders of the decoded image; before any inter
  block is reconstructed against a reference frame, that reference
  frame's reconstruction buffer is extended by 48 sample points in all
  four directions, with the borders filled by edge replication.
  Surfaces:
  - `UMV_BORDER_SIZE` — the 48-sample constant the spec mandates
    ("the reconstruction buffers are extended by 48 sample points in
    all directions").
  - `extended_stride(width)` / `extended_height(height)` /
    `origin_offset(stride)` — geometry helpers for the extended buffer
    layout (`stride = width + 2 * 48`, `rows = height + 2 * 48`, and
    the original-image origin sits at `48 * stride + 48` in the
    linear buffer).
  - `extend_border(buf, width, height)` — the in-place §11.5
    applicator. Performs the extension in the spec-mandated order:
    first horizontal (every original-image row's left and right
    48-sample borders take the row's leftmost / rightmost
    original-column value), then vertical (each top / bottom border
    row is a row-wide copy of the topmost / bottommost
    horizontally-extended row). The "first in x, then in y" ordering
    is what makes the four 48×48 corner quadrants uniform at the
    corresponding corner-pixel value of the original image, per the
    spec's Figure 13.
  - `build_extended_buffer(image, width, height)` — convenience that
    allocates a `Vec<u8>` of the right size, copies the raster-order
    `image` plane into the inner rectangle, runs `extend_border`, and
    returns `(buf, stride, origin)` ready to hand to
    `inter::fetch_prediction_block`.
- Transcribed verbatim from `docs/video/vp6/vp6_format.pdf` §11.5
  (On2 Technologies, document version 1.02, August 2006). Like
  §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3 it reads no BoolCoder bits,
  so it advances past round 7 without touching the contested §7.3
  `Split` formula.
- 26 unit tests over the §11.5 stage: the 48-sample border constant;
  `extended_stride` / `extended_height` for a range of input sizes
  including `1×1`, `16×16`, common SD/HD frame sizes (`320×240`,
  `640×480`, `1920×1080`); `origin_offset` at the top-left inner
  corner of two strides; `build_extended_buffer` geometry
  consistency, inner-image preservation, and length matches across
  five common frame sizes; left-border and right-border row
  replication with per-row distinct edge values; top-border and
  bottom-border row replication; all four 48×48 corner quadrants
  uniform at the corresponding corner pixel of the original image
  (the "first in x, then in y" ordering test); `extend_border`
  idempotency when called twice; degenerate `1×1` (whole extended
  buffer uniform), `8×1` (every extended-buffer row reproduces the
  source row), and `1×6` (every extended-buffer column reproduces
  the source column) shapes; the inner image left untouched after
  border extension; and four `should_panic` tests covering input
  validation (`width = 0`, `height = 0`, buffer too small,
  `image.len()` mismatch). Plus four integration tests with
  `crate::inter::fetch_prediction_block`: zero-MV fetch at the
  origin reproduces the inner image's top-left 8×8 block;
  negative-x MV fetch into the left border reads the leftmost
  original column for each row; negative-y MV fetch into the top
  border reads the topmost original row for each column; and a
  combined `±UMV_BORDER_SIZE` MV magnitude check at both the
  top-left and bottom-right corner quadrants demonstrates that
  fetches at the maximum border extent remain in-bounds.

### Added (clean-room round 7, 2026-05-25)

- `loopfilter` module — the spec §11.3 prediction loop filter.
  Implements the 4-tap `(1, -3, 3, -1)` deblocking filter that VP6
  applies to prediction blocks straddling 8x8 boundaries in the
  reference frame (instead of an in-loop reconstruction-buffer
  filter). Surfaces:
  - `PREDICTION_LOOP_FILTER_LIMIT_VALUES[64]` — the
    quantizer-indexed `FLimit` table (`[0]=30`, `[63]=1`, monotonically
    non-increasing), indexed by the frame's raw-bit `DctQMask`.
  - `boundary_x` / `boundary_y` — the `(8 - (mV & 7)) & 7` block-edge
    offset calculation that locates the straddling boundary inside
    the prediction block from the whole-sample-aligned MV components
    (`mV >> MvShift`, provided by round 6's
    `inter::whole_sample_aligned`).
  - `bound(FLimit, FiltVal)` — the soft-clip: linear passthrough in
    `|FiltVal| < FLimit`, symmetric taper across
    `[FLimit, 2*FLimit)`, hard zero at `|FiltVal| >= 2*FLimit`. The
    taper preserves real reference-frame edges and smooths
    quantization-induced block-boundary discontinuities.
  - `prediction_loop_filter_function` — the per-edge applicator
    implementing the §11.3 `(1, -3, 3, -1)` filter with `+ 4 ) >> 3`
    round-and-descale, the `Bound()` soft-clip, and the
    `Clamp0To255` writes on the two boundary-adjacent samples.
  - `filter_vertical_boundary` / `filter_horizontal_boundary` —
    2-D wrappers that select `step=1, pitch=stride` (vertical) or
    `step=stride, pitch=1` (horizontal) and sweep the 8-sample
    edge.
- Per the spec, only the deblocking variant is implemented; the
  deringing variant carries the spec's own "not currently supported
  by the decoder (see Table 3)" rider.
- Transcribed verbatim from `docs/video/vp6/vp6_format.pdf` §11.3
  (On2 Technologies, document version 1.02, August 2006). Like
  §15/§16/§17.1/§11.4/§17.2–§17.4 it reads no BoolCoder bits, so it
  advances past round 6 without touching the contested §7.3 `Split`
  formula. `UseLoopFilter` is a raw-bit frame-header field; the
  per-profile gate (disabled in Simple Profile, read from the header
  in Advanced) is a caller-side concern.
- 26 unit tests over the §11.3 stage: the 64-entry limit table
  (length, endpoints, monotonicity, six mid-table spot values);
  `boundary_x` / `boundary_y` for aligned-MV (zero), non-aligned
  positive, sign-mirror identity, and an exhaustive `in 0..=7`
  range sweep across `-64..=64`; `abs` and `clamp_0_to_255`
  pseudocode equivalence; the `Bound` soft-clip in all four
  spec branches (zero-in/zero-out, saturation at `±2·FLimit`,
  small-positive passthrough, small-negative passthrough,
  taper-band linearity, sign symmetry); the applicator's
  flat-input no-change, small-step smoothing, large-step edge
  preservation, high-quantizer preserves-more-than-low; the
  `Clamp0To255` clip-path invariant under a 5⁴ pixel sweep and a
  wider 256·5³ sweep; multi-row vertical-boundary and
  multi-column horizontal-boundary sweeps with closed-form
  expected values; the "must not mutate reference in place"
  caller-temp-copy pattern; and an integration test combining
  `inter::whole_sample_aligned` with `boundary_x` to verify the
  8x8-aligned-MV → no-boundary identity and the
  one-sample-past-boundary case.

### Added (clean-room round 6, 2026-05-24)

- `inter` module — the spec §17.2–§17.4 inter-block reconstruction
  stage. `reconstruct_inter_block` / `inter_block_to_pixels` apply
  §17's shared recombination formula
  `OutputValue = PredictionValue + PredictionError` followed by an
  inclusive clip to `0..=255`. One function for all three inter cases:
  §17.2 (zero MV), §17.3 (full-pixel MV) and §17.4 (fractional MV)
  share byte-identical recombination pseudocode and differ only in how
  the 8x8 prediction block is *sourced*. No `+128` intra level shift
  applies — the prediction already carries the DC.
- `fetch_prediction_block` — the §17.2/§17.3 integer-offset prediction
  fetch: a straight copy of an 8x8 region from a reference
  reconstruction buffer at an integer `(dx, dy)` offset (`(0, 0)` for
  §17.2's zero vector, the integer whole-sample MV for §17.3). §17.4
  sources its prediction from the round-5 §11.4 filters
  (`bilinear_block` / `bicubic_block`) instead.
- `MvShift::{Luma, Chroma}` plus `whole_sample_aligned`, `luma_frac`
  and `chroma_frac` — the §11.4 motion-vector decomposition:
  `WholeSampleAligned = MvComponent >> MvShift` (arithmetic shift, so
  negative MVs floor toward `-inf`) and the low-`MvShift`-bit
  fractional phase. `MvShift` is 2 for luma (¼-pixel precision, 4
  phases) and 3 for chroma (⅛-pixel precision, 8 phases) per §11.4's
  `// Mvshift is 2 for luma blocks and 3 for chroma blocks` comment.
  Phase counts mirror the `BILINEAR_LUMA_FILTERS[4]` /
  `BILINEAR_CHROMA_FILTERS[8]` row counts. Transcribed verbatim from
  `docs/video/vp6/vp6_format.pdf` §11.4 and §17.
- BoolCoder-independent like §15/§16/§17.1/§11.4, so it advances past
  round 5 without touching the contested §7.3 `Split` formula. The
  motion vector that drives the fetch/interpolation phase is decoded
  upstream, behind the BoolCoder.
- 22 unit tests over the §17.2–§17.4 stage: `MvShift::bits` /
  `phase_count` / `frac_mask` spec values, whole/fractional MV split for
  positive luma, positive chroma, negative luma (arithmetic shift floor)
  and exact-whole-pixel MVs; `fetch_prediction_block` for zero vector
  (co-located copy), positive integer offset, negative integer offset
  and round-trip through `whole_sample_aligned`; `reconstruct_inter_block`
  for zero residual (prediction passes through), positive residual,
  negative residual, overflow clip to 255, underflow clip to 0,
  inclusive-boundary behaviour at 0/255, per-sample independence,
  wrapper-vs-dual-buffer parity, the **no-intra-level-shift** invariant
  (a pinned distinction against §17.1); plus end-to-end §17.2 (fetch +
  recombine) and §17.4 (§11.4 bilinear → recombine) integration tests.

### Added (clean-room round 5, 2026-05-24)

- `interp` module — the spec §11.4 fractional-pixel motion-compensation
  interpolation filters. Bilinear 2-tap kernel (`bilinear_point`) and
  4-tap bicubic kernel (`bicubic_point`); the full tap tables
  `BILINEAR_LUMA_FILTERS` `[4][2]`, `BILINEAR_CHROMA_FILTERS` `[8][2]`,
  `BICUBIC_FILTER_SET` `[17][8][4]` (`BICUBIC_VP61_INDEX = 16` selects
  the VP6.1 coefficient set); separable two-pass 8x8 block applicators
  `bilinear_block` / `bicubic_block`; and the §11.4 `Var16Point`
  prediction-block variance metric `var_16_point` used by the
  Advanced-Profile filter selector. Each tap set sums to 128; the
  per-point descale is `(Σ taps + 64) >> 7` and the bicubic kernel clips
  to `0..=255` per §11.4.2. Transcribed verbatim from
  `docs/video/vp6/vp6_format.pdf` §11.4. Produces the interpolated
  sub-pixel prediction samples that §17.4 reconstruction consumes;
  BoolCoder-independent like §15/§16/§17.1 so it advances past round 4
  without touching the contested §7.3 `Split` formula.
- 20 unit tests over the §11.4 stage: tap-sum invariants, phase-0
  identity, table dimensions, bilinear/bicubic flat-source and identity
  kernels, bicubic overshoot clipping, separable block applicators
  (whole-pixel copy, flat-source all-phases, horizontal-matches-point),
  and the `Var16Point` variance metric (flat-is-zero, two-level
  closed-form, even-row/even-column sampling).

### DOCS-GAP (round 5)

- §11.4's Advanced-Profile filter-*size* selector reads
  `FilterMvSizeThresh = ((MAX_MV_EXTENT >> 1) + 1) << 2` but
  `MAX_MV_EXTENT` is never assigned a numeric value in the document.
  The per-point kernels, tap tables and the variance half of the
  selector are fully specified and landed; the size-threshold selector
  is deferred until the constant is supplied.

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

# oxideav-vp6

Pure-Rust On2 VP6 (`vp6f` / `VP60` / `VP61` / `VP62`) video codec for
the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Clean-room rebuild in progress.** The crate is a stage-by-stage
re-implementation against On2 Technologies' "VP6 Bitstream & Decoder
Specification" (document version 1.02, August 2006), staged at
`docs/video/vp6/vp6_format.pdf` plus the clean-room errata at
`docs/video/vp6/vp6-errata-and-clarifications.md`. No third-party VP6
source has been consulted at any stage.

Almost every decode primitive is implemented and unit-tested. The crate
exposes a **full intra-frame (I-frame) decoder** (`decode_intra_frame`)
**and** a **full inter-frame (P-frame) decoder** (`decode_inter_frame` /
`decode_inter_frame_with_refs`), both driving an entire frame end-to-end
to output pixels — the P-frame path fuses the §10 mode walk, §11 motion
resolution (single-vector + FourMV), §13 coefficient decode, §14 DC
prediction and §17.2/§17.3/§17.4 reconstruction (including the §11.3
prediction loop filter and §11.4 sub-pixel filter-family dispatch), with
§4 golden-frame bookkeeping (`ReferenceFrames`).

The crate now **also encodes both a VP6 keyframe and a P-frame**.
`encode_intra_frame` takes a 4:2:0 source `Frame` and produces a
single-partition I-frame bitstream that `decode_intra_frame`
reconstructs back to pixels at a quantiser-bounded PSNR floor (a 32×32
patterned frame at q=48 round-trips at ~44 dB luma / ~45 dB chroma; flat
frames are exact). `encode_inter_frame` produces the BoolCoder data
partition for the simplest valid P-frame — every macroblock
`CODE_INTER_NO_MV` (zero MV, predicted from the previous-frame
reconstruction) — that `decode_inter_frame` reconstructs: an unchanged
frame round-trips **exactly** via the zero-MV inter copy, a changed
frame above a quantiser-bounded floor. Both encode paths are the
stage-for-stage inverse of the decode pipeline: (I) −128 level shift, or
(P) residual = source − zero-MV prediction → §16-dual forward DCT →
§15-inverse quantise → §14 DC delta → §13 token emit (+ §10 mode emit
for P-frames via `mode_encode`).

A full **keyframe → P-frame GOP** round-trips end-to-end: encode an
I-frame, decode it, seed the §4 `ReferenceFrames`, encode a P-frame
against the *decoded* keyframe, and decode it via
`decode_inter_frame_with_refs` — an unchanged P-frame reproduces the
keyframe reconstruction bit-for-bit.

What remains is the top-level per-frame assembly (§9 header →
§10/§11.2/§13 probability-update sub-streams → frame dispatch),
**registering a `Decoder`/`Encoder` with `oxideav-core`** (`register()`
is currently a no-op), the P-frame encoder's §9 InterHeader emit (the
current `encode_inter_frame` returns the BoolCoder partition the
decoder's per-MB driver consumes directly), and the encoder's motion
estimation (the richer non-`CODE_INTER_NO_MV` modes) / per-frame
probability-update / rate-control surfaces.

### Implemented stages

- **Frame header — both coders** (`frame_header`) — the §9 Table 1 / 2
  `R(n)` raw-bit prefix (reporting `raw_prefix_len`, the BoolCoder
  partition offset) **and** the §9 Table 2 / 3 BoolCoder-coded `b(n)`
  tail (`Vp6HeaderTail`): `VFragments` / `HFragments` /
  `OutputVFragments` / `OutputHFragments` / `ScalingMode` (IntraHeader),
  `RefreshGoldenFrame` + the Advanced-profile `UseLoopFilter` /
  `LoopFilterSelector` (InterHeader), the `AutoSelectPMFlag`-gated
  prediction-filter selectors with their VP6.2 InterHeader gating, the
  VP6.2 `PredictionFilterAlpha`, and the trailing `UseHuffman` flag.
- **Per-block reconstruction** — inverse quantization (`dequant`, §15),
  inverse DCT (`idct_block`, §16), intra and inter block recombination
  (§17.1–§17.4), fractional-pixel interpolation filters (`interp`,
  §11.4), the prediction loop filter (`loopfilter`, §11.3), and the
  Unrestricted-Motion-Vector border extension (`umv`, §11.5).
- **Static entropy surfaces** — the zig-zag scan + DC predictor
  (`scan` / `dc_pred`, §12.1 / §14), coding-mode tables (`modes`, §10),
  DCT-token tables (`tokens`, §13), the Huffman tree builder
  (`huffman`, §7.2), the AC zero-run-length tables (`zrl`, §13.3.3),
  and the raw-bit reader (`raw_bits`, §3).
- **BoolCoder primitive — decoder + encoder** — the §7.3 binary
  arithmetic decoder (`bool_coder::BoolCoder`) and its exact inverse,
  `bool_coder::BoolEncoder` (derived solely from the §7.3 decode
  equations; encode→decode round-trips bit-for-bit), the §13.2.1 DC and
  §13.3.1 AC arithmetic decoders
  (`dct_decode`), the §13.3.3.1 zero-run traversal, the per-frame
  probability-update bitstreams (`prob_update` / `mv_prob_update` /
  `mode_prob_update` / `scan_update`), the §11.1 motion-vector
  component decoder (`mv_decode`), the §11 differential MV
  reconstruction (`mv_diff`), the §10 `CODE_INTER_FOURMV` block-mode
  signaling (`fourmv`) and chroma-MV derivation, the §10
  Nearest/Near neighbour walker (`near_mv`), the §10 `VP6_DecodeMode`
  macroblock-mode traversal plus the §10/§13 frame-level
  raster-order macroblock mode-decode pass
  (`mode_decode::decode_macroblock_modes` — threads `last_mode` and
  per-MB `ModeAvailability` across the MB grid; the first stage of the
  per-MB driver), the §10/§11 per-MB **motion-vector resolution**
  (`mv_decode::reconstruct_macroblock_mv` — dispatches a decoded
  [`CodingMode`] on its `MvSource` class: `Zero`/`Intra` read no bits,
  `New` reads a §11.1 delta and adds it to the §11 differential
  reference, `Nearest`/`Near` reuse the §10 neighbour walk; returns the
  final `MotionVector` + `ReferenceBucket` as a `MacroblockMv` ready to
  feed the next MB's neighbour grid via `as_neighbour`), the §10 Table 4
  mode→reference and mode→MV-source classifiers
  (`CodingMode::reference_bucket` / `mv_source`), and the §9
  output-scaling surface (`scaling`).
- **Frame assembly** (`frame_assembly`) — block-to-plane raster
  placement of reconstructed 8×8 blocks into a YUV 4:2:0 image.

### Full intra-frame decode loop

- **`intra_frame::decode_intra_frame`** drives a complete keyframe
  end-to-end to output pixels: it walks the macroblock grid in raster
  order and, for each MB's six blocks (four luma raster TL/TR/BL/BR,
  then U, then V — §13 page 58), runs the full §13 coefficient decode →
  §14 DC prediction → §15 dequant → §16 IDCT → §17.1 reconstruct → §2
  raster-assembly chain, threading the §14/§13.2 left/above block-grid
  neighbours through explicit per-plane coded-DC grids (the
  MB-interleaved decode order makes the plane-raster
  `DcZeroContextTracker` unusable for this). Reads a single BoolCoder
  partition (`MultiStream == 0`). `IntraProbs::keyframe()` seeds the
  per-frame DC/AC/zero-run banks at the §13 baselines. Driving this loop
  surfaced and fixed a latent §16 IDCT `i32` overflow: a conformant
  dequantized coefficient times the Q16 cosine constant exceeds
  `i32::MAX`, so the multiply now widens to `i64` before the `>> 16`
  descale.

### Full intra-frame encode loop — round-trips to pixels

- **`intra_encode::encode_intra_frame`** is the top-level keyframe
  **encoder**, the stage-for-stage inverse of `decode_intra_frame`.
  Per block it applies the §17.1 `−128` level shift, the §16-dual
  forward DCT (`forward_dct::fdct_block`), the §15-inverse quantiser
  (`round(coeff / factor)` into scan order), the §14 DC-prediction delta
  (`Δ = coded_dc − predictor`), and the §13 token emit
  (`token_encode::encode_block_coefficients`) — threaded through the same
  per-plane coded-DC grids and `DcPredictionContext` the decoder uses, in
  the identical MB/block walk order and with identical §13.2 Table-26 DC
  context selection. It emits the simplest valid I-frame shape: Simple
  profile, `MultiStream == 0` (single BoolCoder partition), VP6.0,
  default zig-zag scan, keyframe-baseline probabilities. The §9 raw-bit
  prefix goes through `oxideav_core::bits::BitWriter`; the header tail +
  coefficient tokens through the §7.3 `BoolEncoder`.
- **`forward_dct::fdct_block`** is the §16-dual forward transform: a
  separable orthonormal 8-point DCT-II per axis, scaled to invert the
  §16 integer IDCT's observable `1/32` pure-DC gain, evaluated in `f64`
  and rounded to nearest. `idct(fdct(x))` recovers the input to ≤3 LSB
  per sample (the un-quantised transform-pair floor).
- **`token_encode`** is the bit-for-bit inverse of the `dct_decode`
  token trees: the §13.2.1 DC tree walk, the §13.3.1 AC tree walk (incl.
  the `EncodedCoeffs>1 && Prec==WasZero` implicit-1 shortcut), the
  magnitude/sign emit, the §13.3.3.1 zero-run emit, and the per-block
  `encode_block_coefficients` mirroring the decoder's `EncodedCoeffs`
  loop (Prec evolution, inclusive zero-run choreography, EOB-vs-natural-
  full-block termination). Round-trip tests pin every DC value across the
  full signed range, all category magnitudes/signs, both zero-run bands,
  and full coefficient blocks (empty, DC-only, scattered AC, leading zero
  run, last-nonzero-at-63, zero-DC-with-AC).

### Full P-frame encode loop — round-trips to pixels

- **`inter_encode::encode_inter_frame`** is the top-level **P-frame
  encoder**, the inter-frame dual of `encode_intra_frame`, producing the
  BoolCoder data partition `decode_inter_frame` consumes. It emits the
  simplest valid P-frame: every macroblock `CODE_INTER_NO_MV` (§10) —
  zero motion vector, predicted from the previous-frame reconstruction.
  Per block it forms the inter residual (`source − zero-MV prediction`,
  using the *same* `predict_inter_block_subpel` call the decoder uses so
  predictions are bit-identical), then the same §16-dual forward DCT →
  §15-inverse quantise → §14 DC delta → §13 token emit core as the intra
  encoder, threaded through the same per-plane coded-DC grids. The §10
  mode emit goes through `mode_encode`. Because the §10 Nearest/Near walk
  skips zero MVs, an all-`CODE_INTER_NO_MV` frame has `ModeAvailability::
  Neither` for every MB, so the encoder transmits each mode against the
  same probXmitted row the decoder selects. `BorderedRef::{y,u,v}_plane`
  expose the §11.5-bordered reference so the encoder forms the prediction
  without duplicating the border construction. End-to-end tests feed the
  partition to `decode_inter_frame`: unchanged frames are exact, changed
  frames clear a quantiser-bounded PSNR floor, and a full keyframe →
  P-frame GOP (encode/decode I → seed §4 refs → encode/decode P against
  the decoded keyframe) round-trips.
- **`mode_encode`** is the bit-for-bit inverse of `mode_decode`: the §10
  Figure 10 `VP6_DecodeMode` tree, emitting the root "same as last" bit
  and the node-path bits that drive the decoder's nine-node descent to a
  mode's leaf. `encode_mode` / `encode_mode_descend` /
  `encode_mode_from_probs` mirror the three decode entry points; the
  same-as-last fast path takes the minimal one-bit encoding when a mode
  repeats `last_mode`. Round-trip tests pin every
  `(mode, availability, last_mode)` triple against the decoder.

### Inter (P-frame) path — decodes end-to-end to pixels

- **`inter_frame::decode_inter_frame`** is the **fused P-frame driver**.
  It walks the macroblock grid in §8 single-stream bitstream order and,
  per MB, decodes the §10 coding mode (against the per-frame `probXmitted`
  bank + §10 Nearest/Near availability from the MV grid built so far), the
  §11 motion state (single-vector via
  `mv_decode::reconstruct_macroblock_mv`, FourMV via
  `fourmv::reconstruct_fourmv_macroblock`, or none for intra), the six §13
  block coefficients with §14 DC prediction, and §17 reconstruction —
  intra MBs via §17.1 (`+128`), inter MBs by motion-compensating against
  the previous/golden `BorderedRef` and recombining (§17.2/§17.3/§17.4).
  The §14 DC neighbour grid carries each block's reference bucket so the
  same-reference filter is meaningful in a P-frame mixing intra/last/
  golden blocks. `decode_inter_frame_with_refs` threads a
  `ReferenceFrames` directly.
- **`inter::predict_inter_block_subpel`** is the §17.4 fractional-pixel
  block predictor: §11.4 whole/fractional MV decomposition, the §11.3
  prediction loop filter at the straddled `BoundaryX`/`BoundaryY` edges
  (to a separate working window per §11.3), then §11.4 bilinear/bicubic
  interpolation. `PredictionFilterPolicy` resolves the §11.4
  `AutoSelectPMFlag` decision (fixed family, or per-block auto-select from
  the MV-size and `Var16Point` variance thresholds).
- **`inter_frame::ReferenceFrames`** is the §4 golden-frame bookkeeping:
  previous-frame + Golden Frame buffers with the §4 update rules (seed
  Golden from an I-frame; refresh it on `RefreshGoldenFrame`).
- **`inter::reconstruct_inter_macroblock`** remains the §17.2/§17.3
  integer-MV MB-level glue (per-block §11.5-clamped fetch + §17
  recombine).

### Blocked / remaining

- **Top-level `Decoder` registration** — the per-frame assembly chain
  (§9 header parse → §10 mode-prob updates → §11.2 MV-prob updates → §13
  Figure-5 coefficient-prob updates / §12 scan updates → intra/P-frame
  dispatch) and the registered `Decoder` shell over it. Each stage exists
  (`frame_header`, `mode_prob_update`, `mv_prob_update`, `prob_update`,
  `scan_update`, `decode_intra_frame`, `decode_inter_frame`); the
  remaining work is sequencing them in the exact Figure-1/Figure-5
  bitstream order, which wants a conformant `.vp6` fixture to validate the
  parse order against.
- **DOCS-GAP — FourMV MB neighbour representative.** §10 defines the
  Nearest/Near walk over "decoded macroblock neighbors" (one MV per
  neighbour MB), but never states which of a `CODE_INTER_FOURMV` MB's
  four per-block vectors (or what combination) represents it in a
  *later* MB's `NearMacroBlocks` list, nor which it contributes as the
  §11 differential reference for an immediately-right/below `New` MB.
  `reconstruct_fourmv_macroblock` exposes all four vectors and defers
  the choice rather than guess.
- **High-bit-depth / scaling resampling math** and **sample-exact
  validation against a conformant `.vp6` bitstream** — the latter
  needs an encoder-produced fixture.

## License

MIT — see [LICENSE](./LICENSE).

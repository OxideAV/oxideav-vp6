# oxideav-vp6

[![CI](https://github.com/OxideAV/oxideav-vp6/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-vp6/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-vp6.svg)](https://crates.io/crates/oxideav-vp6) [![docs.rs](https://docs.rs/oxideav-vp6/badge.svg)](https://docs.rs/oxideav-vp6) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust On2 VP6 (`vp6f` / `VP60` / `VP61` / `VP62`) video codec for
the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Clean-room rebuild in progress.** The crate is a stage-by-stage
re-implementation against On2 Technologies' "VP6 Bitstream & Decoder
Specification" (document version 1.02, August 2006), staged at
`docs/video/vp6/vp6_format.pdf` plus the clean-room errata at
`docs/video/vp6/vp6-errata-and-clarifications.md`. No third-party VP6
source has been consulted at any stage.

**Round 447 — P-frame arithmetic-path beachhead + a new Table 18
erratum.** Verify-first: the round-2 Extractor-03 staging (docs commit
2026-08-10) was already consumed by round 439, so the round attacked
the open P-frame blocker by pixel arbitration (recovering each
macroblock's true motion and integer coefficients from the oracle by
inverting the reconstruction pipeline against the bit-exact keyframe).
It landed one conclusive printed-spec correction: **Table 18's
extra-bit probability list is in transmission order — the first-listed
probability codes the most-significant magnitude bit** — where the
§13.2.1/§13.3.1 listings' `B(Probs[BitsCount])` (descending) would
pair the last-listed probability with the MSB. Arbitrated by the
P-frame's first content macroblock (a CATEGORY5 DC whose magnitude
bits decode to the oracle-recovered 54 only under the corrected
pairing, with every following token then exact); invisible to all
earlier gates because the keyframe's Huffman transport reads extras as
raw bits. Decoder and encoder both carry the pairing. Two new CI
gates: the P-frame partition-2 arithmetic token stream decodes
coefficient-exact through the static prefix plus the first content
macroblock, and macroblocks (0,0)..=(0,30) of the P-frame reconstruct
bit-exactly through the two-pass MultiStream driver (pass 1 walking
all 1620 MBs). The remaining blocker is sharpened: the §10/§11.1 wire
diverges at the first transmitted MV (pixel truth ~(1,24) ¼-pel vs the
printed reading's (-5,-73); pass 1 over-runs the 473-byte partition-1
budget by 180 bytes), first at the first `NearestOnly`-availability
macroblock — see the CHANGELOG for the candidate readings and the
docs asks.

**Round 442 — §9 output scaling applied, plus the downsampled-encode
path.** The `scaling` module grows from the typed field surface into the
executed post-decode stage: `FrameGeometry` is reworked to the operative
macroblock units (erratum #338), `OutputScaling::plan` derives the
per-mode placement (`SCALE_TO_FIT` stretch; `MAINTAIN_ASPECT_RATIO`
centred aspect-preserving fit, degenerating to the stretch when aspects
match; `CENTER` pad/crop; `OTHER` stays unspecified by the staged doc —
docs-gap — and applies as identity), and `apply_output_scaling` executes
it with a documented implementation-defined resampler (separable 2-tap
centre-aligned linear, Q8 fixed point, edge-clamped — scaling is a §2
"on output after decode" presentation step that never re-enters the §4
prediction loop, so the kernel is not conformance-bearing).
`Vp6Decoder::decode_packet_scaled` (and the registered decoder) emit
frames at the signalled output geometry, carrying the keyframe's scaling
across the GOP; `encode_intra_frame_scaled` (+ the MultiStream dual) and
`Vp6CodecEncoder::with_downscale` land the encoder half — code the GOP
at reduced resolution, signal the display geometry, decoder upscales
back. Round-trips pinned: flat scaled keyframe exact; 64x64 gradient
coded at 32x32 reconstructs ≥35 dB vs the full-res source; the fixture's
identity scaling (Output == coded 54x30, mode 0) decodes bit-identical
through the scaled path. The round also statically corroborated the r439
errata against the staged docs: the §14 toward-zero average **agrees
with the spec's own prose** (only the summary-table `Sign` formula is
defective); the magnitude-based §13.3.1 `Prec` seed is the internally
consistent reading (both printed mid-block updates are
magnitude-domain) and matches the extraction record; the chroma +128
seed's **Intra-only bucket scoping remains uncorroborated** by the
staged errata (ask filed).

**Round 439 — the conformant third-party keyframe decodes
pixel-exactly, end-to-end.** The full-frame Huffman blocker is closed:
`Vp6Decoder::decode_packet` reconstructs the fixture keyframe's entire
854x480 display region — all 9720 blocks, every Y/U/V sample —
bit-identical to the black-box decode oracle (CI gate
`keyframe_decodes_pixel_exact`). The staged extraction record
(`docs/video/vp6/provenance/03-extractor-binary-huffman.md` + tables
04–06 + the extended errata) supplied the corrected §7.2.1 tree
construction (insert-before-equals tie-break), the keyframe
carry-forward probability fill (`#277 part 7` — clear update flags
write a shared running vector, which also subsumes the earlier
"chroma DC tree from the luma bank" finding), and the closed
§13.2.2/§13.3.3.2 run conventions (`#193`). Landing them surfaced and
fixture-arbitrated three further printed-spec defects: the §13.3.1
`Prec` seed classifies the DC **magnitude** (a DC of −1 seeds
`Prec = 1`), the §14 two-neighbour DC average truncates **toward
zero** (the printed `+Sign(L+A)` rounds odd sums away from zero and
fails both sign directions), and the §14 chroma +128 quantized-DC seed
applies to the **Intra bucket only** (seeding the inter buckets
desynchronises the fixture P-frame's Table-26 DC contexts). The r390
"DC-fold" tree variant is removed — under the corrected banks and
tie-break the printed 12-leaf §13.1 DC mapping is the operative one.
The two fixture **P-frames** (MultiStream arithmetic path) decode
their static prefix sample-exactly but diverge at the first content
macroblock; their §10/§11 wire semantics need a behavioural P-frame
trace (ask filed — the extraction record explicitly leaves P-frames
un-established).

**Round 390 — first third-party conformance fixture.** A conformant
external vp6f stream (Huffman/MultiStream, 854x480, I+P+P, black-box
decode-oracle YUV) lives at
`tests/fixtures/vp6f-huffman-i-then-p-854x480/` with CI gates
(`tests/conformance_vp6f.rs`). It arbitrated printed-spec errata, fixed
and pinned: §9 Table 2 geometry is transmitted in **macroblock units**
(not the printed 8x8-block units); partition 1's BoolCoder legitimately
reads past `Buff2Offset` (tight partition sizing + 32-bit look-ahead).
Round 411 added the §16 IDCT rounding arbitration (per-multiply
`>> 16` exactly as printed + a final `(x + 8) >> 4` rounding add the
listing omits, pinned by all 555 non-uniform oracle luma blocks) and
the differential bit-flip probe map that first exposed the chroma DC
seed and bank findings round 439 wired in.

Almost every decode primitive is implemented and unit-tested. The crate
exposes a **full intra-frame (I-frame) decoder** (`decode_intra_frame`)
**and** a **full inter-frame (P-frame) decoder** (`decode_inter_frame` /
`decode_inter_frame_with_refs`), both driving an entire frame end-to-end
to output pixels — the P-frame path fuses the §10 mode walk, §11 motion
resolution (single-vector + FourMV), §13 coefficient decode, §14 DC
prediction and §17.2/§17.3/§17.4 reconstruction (including the §11.3
prediction loop filter and §11.4 sub-pixel filter-family dispatch), with
§4 golden-frame bookkeeping (`ReferenceFrames`).

The crate now **also encodes both a VP6 keyframe and a P-frame, the
latter with real motion estimation**. `encode_intra_frame` takes a 4:2:0
source `Frame` and produces a single-partition I-frame bitstream that
`decode_intra_frame` reconstructs back to pixels at a quantiser-bounded
PSNR floor (a 32×32 patterned frame at q=48 round-trips at ~44 dB luma /
~45 dB chroma; flat frames are exact). For P-frames there are now two
encoders: `encode_inter_frame` produces the simplest valid all-zero-MV
shape (every macroblock `CODE_INTER_NO_MV`), and
**`encode_inter_frame_me`** performs a per-macroblock **motion search**,
emitting `CODE_INTER_PLUS_MV` (a real §11.1-coded motion vector against
the §11 differential reference) wherever the search beats zero-MV by the
bit-cost margin. Both decode back through `decode_inter_frame`: an
unchanged frame round-trips **exactly** via the zero-MV inter copy, a
translated frame reconstructs above a quantiser-bounded floor with the ME
encoder tracking the motion. All encode paths are the stage-for-stage
inverse of the decode pipeline: (I) −128 level shift, or (P) residual =
source − (zero-or-searched-MV) prediction → §16-dual forward DCT →
§15-inverse quantise → §14 DC delta → §13 token emit (+ §10 mode emit via
`mode_encode` and, for `CODE_INTER_PLUS_MV`, §11.1 MV-delta emit via
`mv_encode`). Both P-frame encoders also emit self-describing packets
(`encode_inter_frame_packet` / `encode_inter_frame_me_packet`) that flow
through the top-level `decode_frame::Vp6Decoder`.

The crate now exposes a **top-level per-frame assembly** and a
**registered `oxideav-core` `Decoder`**. `decode_frame::Vp6Decoder` is a
stateful driver that sequences the §9 header prefix parse → BoolCoder
construction → §9 header-tail parse → keyframe/inter dispatch, threading
the §9 cross-frame profile/version (Table 3 omits both — inherited from
the most recent I-frame) and the §4 `ReferenceFrames` across
`decode_packet` calls. `decoder::Vp6CodecDecoder` wraps it in the
framework `Decoder` trait (`send_packet`/`receive_frame` → 3-plane 4:2:0
`VideoFrame`); `register()` installs it under id `"vp6"` with the On2 /
Flash / Matroska container tags (`VP60` / `VP61` / `VP62` / `vp6f` /
`V_VP6`). A full **keyframe → P-frame GOP** round-trips end-to-end
through a single `Vp6Decoder`: a keyframe packet seeds the §4 refs +
carries the profile/version, then an unchanged inter frame predicted from
the decoded keyframe reproduces it bit-for-bit. The P-frame encoder now
emits a self-describing packet too: `encode_inter_frame_packet` prepends
the §9 InterHeader (Table 1 prefix + Table 3 BoolCoder tail) to the data
partition so the round-trip flows through `decode_packet`.

The top-level driver now sequences the **§8 Figure 1 / Figure 5
probability-update sub-streams** in spec order. `coeff_prob_update`
fuses the four §13/§12.2 passes into the single Figure-5 order the
bitstream map fixes (DC node updates → scan-update bit + custom-scan
body → ZRL updates → AC updates); `decode_keyframe` consumes the
Figure-5 pass (a keyframe carries no §10/§11.2 tree — §10: I-frame MBs
are implicitly intra), and `decode_interframe` consumes the full inter
prefix §10 *Mode Probability Updates* → §11.2 *Mv Tree* → Figure 5
*Coefficient Probability Updates* before the per-MB walk. The in-tree
encoders emit the matching prefix (`emit_inter_pre_data_substreams` +
the no-update / `_full` / `_from` Figure-5 emitters), so a keyframe
carrying **real** §13 node-probability updates round-trips end-to-end
through `decode_packet`, and a **P-frame can re-train** the banks
mid-GOP (`encode_inter_frame_packet_with_banks`). The §10/§11.2/§13
banks (+ the §12.2 band assignment) are **persistent** across frames per
spec — reset at each I-frame, mutated by every frame's update
sub-streams, carried into the next frame. The two-partition
**MultiStream** (§6) arrangement and the **Huffman** (§7.2/§13.2.2/§13.3.2)
second-partition coefficient coder are both implemented, decode and
encode: `CoeffSource`/`CoeffSink` route the §13 tokens through any of
the three §5/§6 transports (single-stream partition-1 BoolCoder,
partition-2 BoolCoder at `Buff2Offset`, partition-2 raw-bit Huffman),
every keyframe and P-frame shape (zero-MV / ME / Golden / FourMV) has a
multistream packet emitter in both flavours, and equivalence tests pin
bit-identical pixels across transports. The encoder is
now **registered** as an `oxideav-core` `Encoder` (`encoder::Vp6CodecEncoder`
/ `make_encoder`, a GOP-aware keyframe + motion-estimated-P-frame adapter),
and its **motion estimation** (single-vector `CODE_INTER_PLUS_MV` with a
two-stage box-then-¼-pel luma search and §11 differential-MV emission), its
**Nearest/Near implicit-MV modes**, its **Golden-frame encode modes**, its
**FourMV encode mode** (four independent per-block vectors), and a
**rate-control quantiser selector** (`rate_control`) **all exist**
(`encode_inter_frame_me` / `encode_inter_frame_me_golden` /
`encode_inter_frame_me_fourmv` / `rate_control::select_quantiser_for_budget`).

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
  output-scaling surface **and application** (`scaling` — macroblock-unit
  geometry per erratum #338, per-mode placement plans, and the
  post-decode resample/placement `apply_output_scaling` executes).
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
- **`inter_encode::encode_inter_frame_me`** is the **motion-estimated
  P-frame encoder** — a strict superset of `encode_inter_frame` that codes
  each MB as one of `CODE_INTER_NO_MV` / `CODE_INTER_NEAREST_MV` /
  `CODE_INTER_NEAR_MV` (implicit, no MV bits) or `CODE_INTER_PLUS_MV` (an
  explicit §11.1-coded MV). Per MB it runs a two-stage luma motion search
  (`search_luma_mv`): an integer-pel box search over `±ME_SEARCH_RANGE`
  whole samples around `(0,0)` then a ¼-pel refinement, minimising the
  16×16 luma SAD (`luma_mb_sad`) computed against the *same*
  `predict_inter_block_subpel` prediction the decoder forms. The mode
  decision (`decide_mb_mode`) weighs all options by reconstruction SAD plus
  a bit-cost model: the cheapest implicit option (zero, or a §10
  Nearest/Near neighbour vector) wins unless the searched
  `CODE_INTER_PLUS_MV` beats it by more than `ME_LAMBDA_SAD` (a Lagrangian
  λ proxy for the MV bit-cost) and its §11 differential delta is
  representable. The explicit delta is `best_mv − differential_reference`
  (the nearest same-reference above/left neighbour via
  `select_diff_reference_mv_from_grid`, else zero), emitted with
  `mv_encode::encode_mv_pair`. The Nearest/Near candidates come from the
  same `resolve_near_mvs` walk that supplies the §10 availability, and the
  encoder threads the **identical** §10/§11 `mv_grid` / `last_mode` the
  decoder builds, so each MB's reconstructed MV, mode-context and residual
  match the decoder exactly. Luma residual is formed against the chosen MV;
  chroma against the MB MV at ⅛-pel (§11.4). Round-trip tests against
  `decode_inter_frame` pin the unchanged→exact reduction, translated-source
  reconstruction above a floor, ME ≥ zero-MV on translation, the single-MB
  path, the shared-motion differential-reference path, a uniform-motion
  frame exercising the implicit Nearest/Near modes, and a full keyframe →
  ME-P GOP; `decide_mb_mode` unit tests pin the selection logic.
- **`mv_encode`** is the bit-for-bit inverse of `mv_decode`'s §11.1
  per-component decode: `encode_mv_component` emits the
  `B(IsMvShortProbs)` short/long discriminator (short for `|c| ≤ 7`, long
  for `8..=255`), the Figure 11 short tree or the `[0,1,2,7,6,5,4]`-order
  long bit-stream (respecting the implicit-bit-3 rule), then the
  `B(MvSignProbs)` sign. `encode_mv_pair` is the `(dx, dy)` dual of
  `decode_mv_pair`. Round-trip tests pin every short component, the full
  `0..=255` magnitude range, and mixed pairs against the decoder.
- **`inter_encode::encode_inter_frame_me_golden`** is the **Golden-Frame-aware
  motion-estimated P-frame encoder** — a strict superset of
  `encode_inter_frame_me` that codes each macroblock against **either** the
  previous-frame **or** the Golden-Frame reference. Per MB it builds the §10
  single-vector mode decision twice via `mb_inputs_for_ref` — once against
  `prev` (filtered on `ReferenceBucket::InterLast`), once against `golden`
  (filtered on `ReferenceBucket::InterGolden`) — and `decide_mb_mode_golden`
  takes the cheaper reconstruction, with a `GOLDEN_SWITCH_PENALTY` SAD
  hysteresis so a marginal Golden win that loses the same-reference §14 DC / §11
  differential-MV continuity (and costs marginally more §10 mode-tree bits)
  doesn't flip the reference. The chosen reference's mode set is emitted
  (`CODE_INTER_*` for the previous frame, `CODE_USING_GOLDEN` /
  `CODE_GOLD_NEAREST_MV` / `CODE_GOLD_NEAR_MV` / `CODE_GOLDEN_MV` for the
  Golden Frame), and `encode_inter_block` threads each block's actual reference
  bucket into the §14 DC prediction + per-plane coded-DC grids so a mixed
  previous/golden frame's same-reference DC filter matches the decoder. The §10
  probXmitted availability row is resolved on the previous-frame bucket exactly
  as the decoder indexes it. Round-trip tests against `decode_inter_frame` pin:
  golden-wins (unrelated `prev` → every MB `CODE_USING_GOLDEN`, ≥30 dB luma +
  chroma); the identical-references reduction (unchanged frame **exact**); a
  mixed previous↔golden frame; a full keyframe → Golden-aware P-frame GOP
  recovering the source from the Golden reference through the §4
  `ReferenceFrames`; and a `decide_mb_mode_golden` switch-penalty boundary test.
- **`inter_encode::encode_inter_frame_me_fourmv`** is the **FourMV P-frame
  encoder** — a strict superset of `encode_inter_frame_me` that codes a
  macroblock `CODE_INTER_FOURMV` (four independent per-Y-block vectors) when
  four `search_luma_block_mv` per-block searches beat the best single-vector
  mode by `FOURMV_SAD_MARGIN`. `fourmv::encode_fourmv_macroblock` (the inverse
  of `reconstruct_fourmv_macroblock`) chooses each block's Table 10 mode by
  matching the target MV against the decoder's reconstructable candidates
  (zero → `CODE_INTER_NO_MV`; §10 Nearest/Near match → no-MV-bits implicit;
  else `CODE_INTER_PLUS_MV` against the §11 differential reference), emits the
  four Table 10 codewords + per-block deltas, and returns the reconstructed
  per-block vectors so each luma block's residual is formed against its own
  vector and the two chroma blocks against the §10-averaged chroma MV. A FourMV
  MB contributes `None` to the neighbour grid exactly as the decoder records it
  (the FourMV MB-representative-MV §10 DOCS-GAP is **sidestepped** — the
  round-trip is correct without resolving it). Round-trip tests pin the
  divergent-block-motion case (`CODE_INTER_FOURMV` fires), the unchanged-frame
  exact reduction, the uniform-translation single-vector fallback, and a
  multi-MB FourMV packet through the top-level `Vp6Decoder`.
- **`inter_encode::encode_inter_frame_packet` /
  `encode_inter_frame_me_packet` / `encode_inter_frame_me_golden_packet` /
  `encode_inter_frame_me_fourmv_packet`** prepend the §9 InterHeader (Table 1
  prefix + Table 3 BoolCoder tail) to the zero-MV / motion-estimated /
  Golden-aware / FourMV data partition so each encoded P-frame is a
  self-describing packet `decode_frame::Vp6Decoder::decode_packet` consumes
  end-to-end.
- **`rate_control`** is the per-frame quantiser selector. The encoders take a
  fixed §9 `DctQMask`; `rate_control` solves the inverse — pick the index that
  hits a bit budget. It exploits the §15-table monotonicity (the dequant factor
  decreases with the index, so encoded size is weakly monotonically
  non-decreasing in `DctQMask`) to binary-search the `0..=63` space against a
  caller `encode(q) -> Vec<u8>` closure: `select_quantiser_for_budget` returns
  the finest index whose output fits a hard cap, `select_quantiser_for_target_size`
  the index closest to a target size, each returning a `QuantiserChoice { q,
  size, bytes }` (the chosen index + its already-encoded partition). A
  real-encoder integration test confirms the in-tree encoder's output is
  genuinely monotone in `q` and the budget search fits with the next-finer index
  overflowing.

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
  interpolation. The §11.3 `BoundaryX`/`BoundaryY` offsets are derived from
  the spec's own **round-toward-zero** whole-pixel MV reduction
  (`loopfilter::boundary_whole_pixel`, `mVx = (mx>0)?(mx>>shift):-((-mx)>>shift)`),
  which is distinct from §11.4's arithmetic-shift floor used for the source
  position and variance window — the two diverge for negative MVs whose
  magnitude is not a multiple of `2^MvShift`, and using the floor would
  filter a boundary §11.3 leaves unfiltered. `PredictionFilterPolicy` is the *operative* §11.4
  `AutoSelectPMFlag` decision (fixed family, or per-block auto-select from
  the MV-size and `Var16Point` variance thresholds), built from the
  signalled header fields by **`frame_header::PredictionFilter::resolve`**:
  it converts the MV-size threshold to ¼-pixel units (`(1 << (thresh-1))
  << 2`, or the no-restriction branch) and applies `FilterVarThresh =
  PredictionFilterVarThresh << 5`. The §11.4 variance read uses
  **`interp::var_16_point_clamped`**, which edge-clamps each sampled
  position so an unrestricted (or out-of-spec long) MV whose
  whole-sample-aligned window runs past a buffer edge replicates the §11.5
  edge sample instead of indexing out of bounds (bit-identical to the
  unclamped form for any in-range window).
  **`inter_frame::FilterConfig::from_header`** then assembles the full
  per-frame §11.3/§11.4 configuration from a decoded `Vp6HeaderTail` + the
  frame's `DctQMask` (resolved family policy + the §11.3 loop-filter
  quantiser index when `UseLoopFilter == 1`).
- **`inter_frame::ReferenceFrames`** is the §4 golden-frame bookkeeping:
  previous-frame + Golden Frame buffers with the §4 update rules (seed
  Golden from an I-frame; refresh it on `RefreshGoldenFrame`).
- **`inter::reconstruct_inter_macroblock`** remains the §17.2/§17.3
  integer-MV MB-level glue (per-block §11.5-clamped fetch + §17
  recombine).

### Top-level per-frame assembly + `Decoder` registration

- **`decode_frame::Vp6Decoder`** is the stateful per-frame driver:
  `decode_packet` parses the §9 header prefix (`Vp6FrameHeader::parse`),
  builds the BoolCoder over the partition, parses the §9 header tail
  (`Vp6HeaderTail::parse_with`), and dispatches keyframe →
  `decode_intra_frame` / inter → `decode_inter_frame_with_refs`. It
  threads the §9 cross-frame profile/version (Table 3 omits both) and the
  §4 `ReferenceFrames` between calls, applying the §4/`RefreshGoldenFrame`
  buffer update after every frame. Targets the no-probability-update /
  single-partition / BoolCoder-coefficient shape (`MultiStream` /
  `UseHuffman` / pre-keyframe-inter surface as errors). A keyframe →
  P-frame GOP round-trips end-to-end through one instance.
- **`decoder::Vp6CodecDecoder` + `register`** wrap `Vp6Decoder` in the
  `oxideav_core::Decoder` trait (packet-queue → 3-plane 4:2:0
  `VideoFrame`) and install it under id `"vp6"` with the On2 / Flash /
  Matroska container tags. `register()` is no longer a no-op.
- **`encoder::Vp6CodecEncoder` + `make_encoder`** wrap the keyframe +
  motion-estimated-P-frame encoders in the `oxideav_core::Encoder` trait, now
  registered alongside the decoder (`register_codecs` wires
  `.encoder(make_encoder)`, so `CodecRegistry::first_encoder` resolves a VP6
  encoder under `"vp6"`). A GOP-aware adapter: a keyframe packet
  (`encode_intra_frame`) at the start of every GOP, a motion-estimated P-frame
  packet (`encode_inter_frame_me_packet`) otherwise, against the **decoded**
  previous frame — the reference is maintained by decoding the encoder's own
  output through an internal `Vp6Decoder`, so the round-trip is closed by
  construction. Tests round-trip a keyframe and a keyframe → P-frame GOP
  **through the `Encoder` → `Decoder` trait surfaces**.
- **`inter_encode::encode_inter_frame_packet`** is the §9 InterHeader
  emit that turns the data-partition-only `encode_inter_frame` into a
  self-describing P-frame packet `decode_packet` consumes.

### Blocked / remaining

- **In-header probability-update sequencing — NOW WIRED.** The §8
  Figure 1 / Figure 5 ordering is fully specified in `vp6_format.pdf` §8
  (Figure 1's bullet list + Figure 5's within-pass order on p. 20), so no
  fixture was needed to pin it down. `coeff_prob_update` fuses the four
  §13/§12.2 passes in Figure-5 order; `decode_keyframe` consumes the
  Figure-5 pass and `decode_interframe` consumes §10 → §11.2 → Figure 5
  before the per-MB walk. The encoders emit the symmetric prefix
  (`emit_inter_pre_data_substreams`, `encode_coefficient_prob_updates` /
  `_with_scan` / `_full`, `encode_no_mode_prob_updates`,
  `encode_no_mv_prob_updates`), and a keyframe carrying real §13
  node-probability updates round-trips end-to-end through `decode_packet`.
- **Cross-frame bank persistence — LANDED (round 384).** `Vp6Decoder`
  carries the §13 coefficient banks (+ §12.2 band assignment), the §10
  `probXmitted` bank and the §11.2 MV bank across frames per the spec's
  persistence rules (reset at I-frames; P-frames start from the previous
  frame's post-update values; §12.2 inter deltas apply to the previous
  custom assignment). A P-frame can carry **real** Figure-5 updates
  (`encode_coefficient_prob_updates_from` /
  `encode_inter_frame_packet_with_banks`) and the re-trained banks
  persist into following frames.
- **MultiStream (§6) + Huffman (§7) second partition — LANDED (round
  384), whole-frame validated (round 439).** `decode_packet` splits at
  `Buff2Offset` and dispatches on `MultiStream`/`UseHuffman`; keyframes
  run the §6-general `decode_intra_frame_from_source`, inter frames the
  Figure 3/4 two-pass `decode_inter_frame_multistream` (all MB
  prediction info from partition 1, then all coefficients from
  partition 2). `huff_coeff` implements the full
  §13.1/§13.2.2/§13.3.2/§13.3.3.2/§13.4 Huffman coefficient coder
  (tree derivation from the carry-forward-filled §13 banks, cross-block
  DC/AC1 run state, EOB/DC0 block runs) with a frame-level bit-exact
  encoder. The three printed-spec inconsistencies it disambiguated in
  r384 (the §13.3.3.2 ZRL symbol↔run off-by-one, the long-escape base,
  the §13.2.2 DC-run store missing its `− 1`) are now **measured and
  closed** by the staged errata (`#193 parts 1+2`) and exercised by the
  whole-keyframe pixel-exact gate.
- **FourMV MB neighbour representative — DOCS-GAP CLOSED (errata #155,
  round 384).** The representative a `CODE_INTER_FOURMV` MB contributes
  to later MBs' §10 Nearest/Near scans and §11 differential references
  is its §10 **chroma-derived average** (four Y vectors, rounded away
  from zero). Decoder and encoder both record it in the neighbour grid.
- **P-frame §10/§11 wire semantics — the remaining conformance
  blocker (sharpened round 447).** The arithmetic-path **coefficient
  transport is now pinned** (partition-2 tokens decode
  coefficient-exact through the first content macroblock; the static
  31-MB prefix reconstructs bit-exactly — both CI-gated), and the
  round's pixel arbitration recovered the true motion field around the
  divergence: the first transmitted MV at MB (0,31) is ~(1, 24) ¼-pel
  while the printed §11.1 reading decodes (-5, -73), and pass 1
  consumes 653 bytes against a 473-byte partition-1 budget. The first
  §10 divergence lands exactly at the first macroblock whose
  availability is not `Neither`. Open questions (all statically
  extractable from the staged vendor decoder builds): the operative
  §11.1 short/long discriminator polarity + long-magnitude bit order
  (a flipped-polarity, LSB-first candidate decodes the pixel-true MV
  at (0,31) but fails deeper), the §10 mode-tree behaviour on the
  VQ-updated nearest-exists `probXmitted` rows, the Nearest/Near reuse
  semantics of the golden modes, and the §13.2 Table 26 / §14
  bookkeeping interlock with per-MB reference buckets on P-frames. The
  staged extraction record explicitly leaves P-frames un-established.
- **§9 output scaling — LANDED (round 442).** Decode-side application
  (`decode_packet_scaled` + the registered decoder emitting the
  signalled output geometry) and the encoder-side downsampled-encode
  path (`encode_intra_frame_scaled` /
  `Vp6CodecEncoder::with_downscale`) both exist and round-trip.
  Remaining docs-gaps, both presentation-only (not conformance-
  bearing): the `OTHER` scaling mode's semantics are not defined by the
  staged doc (applied as identity), and the vendor's own resampling
  kernel is undescribed — matching it bit-for-bit would need a scaled
  fixture (coded ≠ output geometry) with a decode oracle.

## License

MIT — see [LICENSE](./LICENSE).

# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.9](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.8...v0.0.9) - 2026-08-23

### Fixed

- fixture notes: round 447 P-frame pixel-arbitration appendix

### Other

- record the §11.1 wire under-determination root cause (letterbox)
- P-frame MB (0,31) inter reconstruction gate + §11.1 wire characterisation
- round 447 - P-frame beachhead, Table 18 erratum, sharpened wire asks
- P-frame arithmetic-path gates + tolerant two-pass diagnostics
- Table 18 extra-bit probabilities pair MSB-first (fixture-arbitrated)
- coded_geometry accessor + CENTER/aspect-fit end-to-end driver gates
- apply §9 output scaling end-to-end + downsampled-encode path
- multistream keyframe encoder with retrained banks (the fixture keyframe's shape)
- round 439 — whole-keyframe pixel-exact decode; P-frame wire semantics are the remaining conformance blocker
- vp6f keyframe decodes pixel-exact end-to-end: §7.2.1 tie-break, keyframe carry-forward, §13/§14 corrections
- clamp reference+delta sums to the §11 ±127 component bound
- Hide internal plumbing behind #[doc(hidden)]; keep stable decode/encode surface
- dc_pred + fixture notes: differential bit-flip probe map and the chroma DC tree/seed findings
- §16 descales are `>> 16` as printed plus a final `(x + 8) >> 4` rounding add (fixture erratum)

### Added (clean-room round 450, 2026-08-24) — **P-frame inter reconstruction gate + §11.1 MV-wire characterisation**

- **The first P-frame's first content macroblock, MB (0,31), now
  reconstructs pixel-exactly under its oracle-recovered motion** — the
  new `pframe_mb31_inter_reconstruction_pixel_exact` gate. Driving the
  §13 coefficient pass (`decode_inter_frame_multistream_pass2`) with the
  per-MB motion supplied externally (static prefix zero-motion inter, MB
  (0,31) `CODE_INTER_PLUS_MV` with the arbitrated `(-1, 24)` ¼-pel
  vector), MB (0,31)'s sixteen luma columns **and** its U and V chroma
  blocks match the decode oracle bit-for-bit. This is the first gate to
  exercise §11.4 fractional-pixel motion compensation (¼-pel luma, ⅛-pel
  chroma), the §15/§16 dequant+IDCT residual path and the §17
  motion-compensated recombination against **real vendor-encoded inter
  coefficients** — machinery the intra-only keyframe gate never touches.
  The partition-2 stream stays aligned through the macroblock because its
  token decode is bit-exact (the round-447 pinned-token gate); this gate
  adds the reconstruction half.
- **The remaining P-frame blocker is localised to the §10/§11.1
  mode/motion-vector wire, and the mode layer is cleared.** Round 450
  probed the fixture's first P-frame by whole-partition synthesis
  (re-encoding a semantically-identical partition 1 through the crate's
  own encoders, then appending scripted BoolCoder symbols, and letting
  the black-box oracle judge each hypothesis by whole-frame
  re-synchronisation) plus single-bit-flip ownership mapping of the real
  partition 1. Established, all against the decode oracle only: the §10
  mode-decode grammar (root "same-as-last" bit, Figure-10 tree topology,
  the `ModeAvailability` neighbour walk) reproduces the vendor wire for
  the whole frame when scripted; MB (0,31)'s coding mode is
  `CODE_INTER_PLUS_MV` and its motion is uniquely `(-1, 24)`; the
  **§11.1 motion-vector component grammar is the sole open defect** — the
  crate reads `(-5, -73)` where the wire carries `(-1, 24)`. Two
  discriminated symptoms: the first-decoded component is the *vertical*
  motion (24), not the horizontal as the spec's "the X component is
  decoded first" implies; and the printed §11.1 short-vector tree decodes
  the macroblock's horizontal magnitude to 2 where the reconstruction
  admits only |x| ≤ 1. The root reason this fixture cannot close the
  grammar: its leading ~30 macroblocks sit in the 854→864 letterbox and
  the vendor stream codes several of them with real modes and motion
  vectors that motion-compensate black-into-black and so reconstruct
  identically for *any* vector — leaving the letterbox mode/MV wire
  pixel-unconstrained while it still consumes coder position, so a full
  crate-verified-mode sweep over the §11.1 component grammar scored by
  whole-frame reconstruction finds no assignment that even reaches MB
  (0,31) in sync. This is a genuine under-determination of the wire from
  the fixture's pixels, not a search-depth limit. Closing the grammar
  needs the staged decoder extraction — `docs/video/vp6/provenance/03`
  explicitly leaves P-frames un-established. The measured facts and the
  discriminating datums are recorded in the fixture's `notes.md`
  (Appendix C).

### Fixed (fixture-arbitrated spec erratum, round 447, 2026-08-17) — **§13 Table 18 extra-bit probability pairing**

- **The Table 18 arithmetic extra-bit probability list is in
  transmission order: the first-listed probability codes the
  most-significant magnitude bit.** The §13.2.1/§13.3.1 listings read
  the magnitude MSB-first via `B(Probs[BitsCount])` with `BitsCount`
  descending, which — against a `Probs[]` stored in the printed Table
  18 order — would pair the *last*-listed probability with the MSB.
  Table 18's own prose fixes the pairing the other way ("the most
  significant bit of the magnitude sent first … encoded with differing
  probabilities as specified by the final column"), and the conformant
  third-party fixture arbitrates for the prose: the first P-frame's
  first content macroblock opens its bottom-right luma block with a
  CATEGORY5 DC whose five magnitude bits decode to the
  oracle-recovered delta 54 (= 35 + 0b10011) only under the MSB-first
  pairing, after which **every remaining token of the block decodes
  exactly** (17 scan positions, pinned by the new
  `pframe_first_content_mb_tokens_decode_exact` gate); the listing's
  pairing decodes 59 and desynchronises the partition.
  `decode_token_value` now mirrors the index
  (`probs[bits - 1 - bits_count]`), and `token_encode` emits with the
  same pairing so encode→decode round-trips stay bit-exact. This
  defect was invisible to every earlier gate: the keyframe's §7.2
  Huffman transport reads category extra-bits as **raw bits** (no
  probabilities), the in-tree encoder round-trips were symmetric, and
  the pairing only shows when the per-bit probabilities differ (the
  ONE..FOUR tokens and CATEGORY1 have at most one magnitude bit).

### Added (clean-room round 447, 2026-08-17) — **P-frame arithmetic-path conformance beachhead**

- **Verify-first:** the round-2 Extractor-03 staging
  (`docs/video/vp6/provenance/03-extractor-binary-huffman.md`, docs
  commit 2026-08-10) was already fully consumed by round 439 (the
  whole-keyframe pixel-exact gate); this round pivoted to the open
  P-frame blocker it explicitly leaves un-established.
- **Two new fixture gates** (`tests/conformance_vp6f.rs`):
  - `pframe_first_content_mb_tokens_decode_exact` — the P-frame's
    partition-2 arithmetic token stream decodes coefficient-exact
    through the 189-block static prefix and the first content
    macroblock's three coefficient-carrying blocks (oracle-recovered
    expected values), pinning the §13.2.1/§13.3.1 arithmetic path —
    contexts, category magnitude bits, zero runs — against
    vendor-encoded wire data for the first time.
  - `pframe_static_prefix_reconstructs_pixel_exact` — the §9
    InterHeader + §10/§11.2/Figure-5 update prefix + the full 1620-MB
    pass-1 walk complete without error, and macroblocks (0,0)..=(0,30)
    reconstruct bit-exactly against the oracle through the two-pass
    MultiStream driver.
- **Diagnostic surfaces** (doc-hidden, powering the gates and future
  conformance work): `decode_inter_frame_multistream_traced` (tolerant
  two-pass decode returning per-MB §10/§11 prediction records +
  first-error positions), `decode_inter_frame_multistream_pass2`
  (coefficient pass with caller-supplied prediction info,
  `ExternalMbMotion`), and `Clone` on `BoolCoder` (snapshot/restore
  for wire probing).

### Known remaining (round 447) — P-frame §10/§11 wire + §13.2/§14 bookkeeping

Pixel-arbitration this round pinned the P-frame divergence far more
precisely than round 439's "diverges at the first content macroblock":

- The true motion at MBs (0,31)/(0,32) is ~(0..1, +24..26) ¼-pel
  (oracle-arbitrated, exact integer-coefficient solutions); the §11.1
  reading decodes (-5, -73) there, and pass 1 over-runs the partition-1
  budget (653 bytes consumed vs `Buff2Offset` − prefix = 473), so the
  §10/§11.1 wire reading is wrong beyond the zero-motion prefix. A
  candidate §11.1 reading (short/long discriminator polarity flipped +
  long magnitude as plain LSB-first bits 0..7 with no implicit bit 3)
  decodes the pixel-true (1,24) at MB (0,31) but fails deeper; it is
  recorded here, not landed.
- The first §10 divergence sits exactly at the first macroblock whose
  availability leaves `Neither` (MB (0,32), `NearestOnly`) — the
  updated `probXmitted` rows for the nearest-exists situations and/or
  the Nearest/Near reuse semantics of the golden modes are the open
  question (the mode row for that situation carries VQ-updated
  weights whose Nearest/Near entries are zero, which contradicts a
  plain reuse reading).
- §13.2 Table 26 DC contexts and the §14 chain on the arithmetic
  P-frame path interlock with per-MB reference buckets in a way the
  fixture alone did not fully discriminate this round (constraint
  solving reconciled the first ~160 MBs under several partial rules
  before diverging); the crate keeps its existing readings. All of
  these are answerable by static extraction from the staged vendor
  decoder builds; precise asks are listed in the round report.

### Added (clean-room round 442, 2026-08-13) — **§9 output scaling applied + downsampled-encode path**

- **The §9 output scaling is now applied, not just parsed.** The
  `scaling` module grows from the typed field surface into the full
  post-decode resample/placement stage:
  - `FrameGeometry` is reworked to the operative **macroblock units**
    (erratum #338 — the printed "8x8 block units" description is wrong
    for all four `*Fragments` fields): `mb_cols`/`mb_rows` with
    `luma_*`/`chroma_*`/`block_*` pixel and grid derivations, plus
    `from_wire` / `of_frame` constructors. `MACROBLOCK_DIM` joins
    `FRAGMENT_DIM`.
  - `OutputScaling::plan` derives a typed `ScalingPlan` per §9 mode.
    What is spec-fixed vs name-implied vs implementation-defined is
    documented exhaustively in the module docs: scaling is a §2
    "on output after decode" **presentation step** (§11.5 ties the
    UMV borders to "the playback scaling features") — the §4 reference
    buffers stay at the coded resolution, so no resampling choice can
    affect bitstream conformance. `SCALE_TO_FIT` stretches to the
    output rectangle; `MAINTAIN_ASPECT_RATIO` performs the largest
    centred aspect-preserving fit (degenerating to the full stretch
    when aspects match — the plain quarter-res→full-res case);
    `CENTER` places unscaled, centred (pad or crop per axis);
    `OTHER` remains unspecified by the staged doc (docs-gap) and is
    applied as the identity.
  - `resample_plane` / `resample_frame` — the implementation-defined
    kernel, documented as such: separable 2-tap linear interpolation on
    the centre-aligned grid, Q8 fixed point, edge-clamped, exact at
    equal sizes and on constants; at 2:1 it degenerates to the 2x2 box
    average (the encoder-side downsample).
  - `apply_output_scaling` executes the plan (neutral Y=0/U=V=128
    letterbox fill; even-rounded rectangles so chroma is exactly half).
- **Decoder wiring:** `Vp6HeaderTail::coded_geometry` /
  `output_scaling` expose the typed §9 fields; `Vp6Decoder` carries the
  keyframe's output-scaling state across the GOP (Table 3 does not
  re-transmit it), and `decode_packet_scaled` / `scale_to_output` /
  `output_scaling` emit frames at the signalled output geometry. The
  registered `Vp6CodecDecoder` now decodes through
  `decode_packet_scaled`, so the framework surface presents the
  signalled output size (identity for streams whose output matches the
  coded geometry — pinned bit-identical on the conformance fixture by
  the new `keyframe_output_scaling_is_identity` gate; the fixture
  transmits Output == coded 54x30, mode 0).
- **Encoder-side scaling signalling + downsampled-encode path:**
  `encode_intra_frame_scaled` / `encode_intra_frame_multistream_scaled`
  emit a keyframe whose IntraHeader carries a caller-chosen
  `Output*Fragments` + `ScalingMode`; `Vp6CodecEncoder::with_downscale
  (factor)` codes the whole GOP at `1/factor` resolution per axis
  (source resampled down before encoding, internal reference loop at
  the coded size) while signalling the display geometry, so a
  downstream decoder reconstructs at the coded size and upscales back
  on output. P-frames need no variant — Table 3 carries no geometry,
  so they inherit the keyframe's signal. Round-trips pinned end-to-end:
  flat scaled keyframe exact through `decode_packet_scaled`; a 64x64
  gradient coded at 32x32 reconstructs ≥35 dB against the original
  full-resolution source; the scaled GOP's unchanged P-frame reproduces
  the scaled keyframe bit-for-bit; both MultiStream transports match
  the single-stream scaled decode; the trait-surface downscaled GOP
  decodes at the display size ≥30 dB.

### Corroborated (round 442) — r439 errata statically re-checked against the staged docs

- **§14 two-neighbour average, truncate toward zero — AGREES with the
  spec prose.** The staged PDF's §14 prose itself says "the arithmetic
  average of their DC values, **truncated towards zero** (values may be
  negative)"; only the summary-table row `(L + A + Sign(L+A)) / 2`
  conflicts (it rounds odd sums away from zero). The r439 correction is
  a resolution of a printed internal contradiction in favour of the
  prose sentence, not a deviation from the document.
- **§13.3.1 magnitude-based `Prec` seed — internally consistent
  reading; the printed seed comparison is the outlier.** Both printed
  *mid-block* `Prec` updates are magnitude-domain (§13.3.1 arithmetic:
  `Prec = 1` on the ±1 token / `Prec = 2` on the greater-than-one
  branch, set before the sign is applied; §13.3.2 Huffman:
  `Prec = (value > 1) ? 2 : 1` on the pre-sign `value`), and the staged
  extraction record pins the vendor decoder's index term as
  `1 + (previous magnitude > 1)`. Only the DC-based seed is printed
  signed (`dc == 1`); the magnitude seed aligns it with every other
  `Prec` write in the document. The staged errata doc has no
  seed-specific entry yet.
- **§14 chroma +128 seed — the +128 and its quantized-DC units are
  corroborated (errata `#277 part 2` + docs Round 1); the
  Intra-bucket-only scoping is NOT yet corroborated.** The staged
  errata state the chroma exception without bucket qualification; the
  Intra-only scoping rests solely on the r439 P-frame-prefix
  arbitration. Ask filed: statically confirm from the vendor decoder's
  frame-setup whether the chroma "last DC" registers seed 128 into the
  intra bank only or into all buckets.

### Added (clean-room round 439, 2026-08-11) — **whole-keyframe pixel-exact decode**

- **The conformant third-party vp6f Huffman keyframe now decodes
  pixel-exactly, end-to-end, through the top-level `Vp6Decoder`** — all
  9720 blocks; every luma, U and V sample of the 854x480 display region
  matches the black-box decode oracle bit-for-bit (new CI gate
  `keyframe_decodes_pixel_exact`). This closes the crate's long-standing
  full-frame Huffman blocker: the trace ask filed against the docs
  collaborator was answered by the staged extraction record
  (`docs/video/vp6/provenance/03-extractor-binary-huffman.md`, tables
  04–06, and the substantially extended errata), and this round lands
  every corrected reading it pins plus three further fixture-arbitrated
  corrections of its own:
  - **§7.2.1 Huffman tree construction — operative tie-break** (errata
    `#277 part 3, closed`). `create_huffman_tree` now maintains an
    ascending-weight list with insert-before-first-greater-or-equal
    semantics (equal-weight symbols end in descending index order;
    merged nodes precede their equals; first head = left/bit-0 child),
    replacing the stable-ascending-sort reading of the printed text.
  - **Keyframe carry-forward probability fill** (errata `#277 part 7`).
    New `KeyframeNodeCarry` + `update_dc_probs_keyframe` /
    `update_ac_probs_keyframe` / `decode_coefficient_prob_updates_keyframe`:
    on a key frame a clear DC/AC update flag writes the shared 11-slot
    running vector's value into the bank (the vector is seeded to 128
    once and never reset between the DC and AC walks), so every DC/AC
    entry is written; ZRL keeps literal semantics. The keyframe
    Figure-5 emitter (`encode_coefficient_prob_updates_full`) mirrors
    the rule. The fixture keyframe's re-derived banks match the staged
    corrected table (`tables/05`) exactly; the earlier "chroma DC tree
    is built from the luma bank" finding is explained by the rule (the
    chroma row inherits the vector) and the special-case copy is gone.
  - **§13.2.2 DC Huffman trees use the printed 12-leaf §13.1 mapping**
    (EOB included; an EOB codeword in the DC position is a bitstream
    error). The r390 "fold node 0's left branch into ZERO_TOKEN"
    variant was a compensating misreading fitted against the literal
    (pre-carry-forward) banks and is removed
    (`dct_token_bool_tree_to_huff_probs_dc` deleted).
  - **§13.3.1 `Prec` seed is magnitude-based** (new erratum, fixture-
    arbitrated): `AcPrecContext::seed_from_dc(-1)` seeds `Prec = 1`,
    not the printed signed `dc == 1` reading's `Prec = 2` — the block
    after the staged `tables/03` datum block carries DC −1 followed by
    an AC1 EOB that only decodes under the `Prec = 1` tree. Matches
    the extraction record's mid-block `1 + (magnitude > 1)` form.
  - **§14 two-neighbour DC average truncates toward zero** (new
    erratum, fixture-arbitrated in both sign directions): the operative
    average is `(L + A) / 2` toward zero; the printed
    `(L + A + Sign(L+A)) / 2` rounds odd sums away from zero and fails
    the fixture on both a negative (`(-299,-298) → -298`) and a
    positive (`(15,0) → 7`) odd-sum pair.
  - **§14 chroma DC seed wired into all drivers** (r411 finding, now
    operative): chroma planes seed the "last decoded DC" register at
    +128 in the quantized-DC domain — for the **Intra bucket only**
    (new fixture arbitration: seeding the inter buckets at 128
    desynchronises the fixture P-frame's Table-26 DC contexts; with
    inter buckets at zero its static prefix parses sample-exactly).
    Both decoders and all encoders thread `new_chroma()` so round-trips
    stay exact.
  - §13.2.2 DC-run store (`run − 1`) and §13.3.3.2 zero-run advance
    (`symbol + 1`, escape `9 + R(6)`) — previously in-tree readings,
    now confirmed closed by errata `#193 parts 1+2` and exercised by
    the whole-frame gate.

- **`encode_intra_frame_multistream_with_banks`** — the encode-side
  dual of the fixture keyframe's shape: a two-partition keyframe whose
  Figure-5 sub-stream carries the carry-forward-encoded retraining to
  reach an arbitrary representable bank set, with the partition-2
  tokens (BoolCoder or §7.2 Huffman) coded against those retrained
  banks. Round-tripped through `Vp6Decoder::decode_packet` under both
  transports, including pure carry-inheritance of the chroma DC row.

### Changed (round 439)

- `reconstruct_diff_mv` clamps the reference+delta sum to the §11
  component bound (±127 ¼-pel). A conformant stream never exceeds it;
  the clamp keeps corrupt or desynchronised input from stepping past
  the §11.5 UMV border during reconstruction (previously an
  out-of-bounds panic was reachable).
- The vp6f conformance suite's leading-macroblock prefix gate is
  superseded by the whole-keyframe gate; the Figure-5 parse gate now
  runs the keyframe carry-forward pass and pins the carry-filled
  `DcProbs` rows.

### Known remaining (round 439)

- The fixture's two **P-frames** do not yet decode pixel-exactly: their
  static prefix (first 31 macroblock columns) parses and reconstructs
  sample-exactly under the corrected readings, but the parse diverges
  at the first content macroblock — the §10 mode-tree probabilities /
  §11.1 MV wire details / arithmetic-path content residuals are not
  discriminable from this fixture alone (several single-knob
  alternative readings all fail at the same macroblock). The staged
  extraction record explicitly leaves P-frames un-established; a
  behavioural P-frame trace ask has been filed with the round report.

### Changed

- Marked the crate's internal plumbing (`#[doc(hidden)]`): the BoolCoder,
  DCT/IDCT, MV, entropy-table and reconstruction modules are exposed only
  for tests/fuzz, so they are hidden from the public API. The stable
  surface is the registered decode/encode drivers (`Vp6Decoder`,
  `decoder`/`encoder` registration, `make_decoder`/`make_encoder`,
  `register`) plus the frame/error types in their signatures (`Frame`,
  `Plane`, `AssemblyError`, `ReferenceFrames`, `BorderedRef`, `Error`).
  This stops cargo-semver-checks from treating internal churn as public
  API. No behavioural or signature change.

### Fixed (fixture-arbitrated spec errata, round 411, 2026-07-11)

- **§16 IDCT descale rounding — corrects the round-390 "toward zero"
  note, which was an under-determined misdiagnosis.** The operative
  descales are: per-multiply `>> 16` **exactly as printed** (arithmetic
  shift, toward -inf on negatives) and a final column-pass
  `(x + 8) >> 4` — a rounding add the printed listing omits. Arbitrated
  by AC-carrying oracle blocks: every one of the fixture keyframe's 555
  non-uniform luma display blocks admits an integer coefficient block
  reconstructing its oracle pixels exactly under this combination, and
  under **no other** {floor, toward-zero, round-nearest} multiply/final
  pairing (each leaves most content blocks with an irreducible residual
  for every integer coefficient assignment). Flat DC-only blocks — the
  only evidence round 390 had — reconstruct identically under several
  roundings and cannot distinguish them. New CI gate:
  `keyframe_content_blocks_reconstruct_pixel_exact` pins three oracle
  blocks' exact reconstruction.

### Added (clean-room round 411, 2026-07-11)

- **Differential bit-flip conformance probing + chroma DC findings.**
  Single-bit corruptions of the fixture keyframe's second partition,
  decoded through the black-box oracle, map the true bit→block
  ownership of the opening tokens (recorded in the fixture `notes.md`
  appendix). Two §13.2.2/§14 findings validated in an experimental
  decode walk (whole 31-MB uniform prefix + the first content block's
  DC now parse pixel-exactly): the chroma DC Huffman tree is built from
  the **luma** node-probability bank, and the §14 frame-start "last
  decoded DC" seed for the chroma planes is **+128** (quantized-DC
  domain), not the zero §14's prose states.
  `DcPredictionContext::new_chroma` + `CHROMA_DC_PREDICTION_SEED` land
  the seed API; wiring both findings into the shared drivers is
  deferred until a pre-existing arithmetic-path encoder fidelity bug
  (worst-case sample errors ~189 hiding under loose PSNR floors) is
  fixed, so the encoder⇄decoder round-trips stay exact. The remaining
  Huffman-path divergence is isolated to the §13.3.2 AC Huffman trees
  (first mismatch: true CATEGORY2 read as FIVE at the first content
  block's AC 1).

### Added (clean-room round 390, 2026-07-06)

- **First third-party conformance fixture + gates.** A conformant
  external vp6f stream (Huffman/MultiStream, 854x480 display / 864x480
  coded, 1 I + 2 P frames, black-box decode-oracle YUV) is pinned at
  `tests/fixtures/vp6f-huffman-i-then-p-854x480/` with five CI gates
  (`tests/conformance_vp6f.rs`): FLV framing, §9 geometry, Figure-5
  update-substream parse across the partition boundary, and
  leading-macroblock pixel-exact decode of the keyframe's Huffman
  coefficient partition against the oracle.

### Fixed (fixture-arbitrated spec errata, round 390)

- **§9 Table 2 geometry is transmitted in macroblock units.** The
  printed spec calls `VFragments`/`HFragments` "8x8 block units" with
  worked examples (240 px → 30), but the conformant stream transmits
  54x30 for its 864x480 coded frame — 16-px macroblock counts. The
  decoder now sizes frames as `2 * transmitted` luma blocks per axis
  and the keyframe encoders emit macroblock counts (and reject
  non-MB-aligned block grids, which the wire format cannot express).
- **Partition 1's BoolCoder spans past `Buff2Offset`.** The §7.3
  pseudo-code advances `Pos` with no end-of-partition check; the real
  encoder sizes partition 1 tightly and the coder's 32-bit look-ahead
  legitimately renormalizes into the first partition-2 byte (the
  fixture's Figure-5 pass ends exactly one byte past the boundary).
  `Buff2Offset` only positions the second reader.
- **§13.2.2 DC Huffman trees fold node 0's left branch into
  `ZERO_TOKEN`.** EOB is forbidden in the DC position (§13.2.1), so the
  DC BoolCoder tree skips the EOB/0 decision and the Huffman conversion
  credits the whole `NodeProb[0]` to ZERO (the generic §13.1 split
  describes the AC alphabet). Without the fold the fixture's chroma
  zero-DC runs decode as the forbidden EOB-in-DC. Corroborated by the
  §13.2 `DcNodeEqs` "UNUSED DUMMY" row that pins the EOB node at
  probability 1.
- **§16 IDCT descales round toward zero.** The printed `>> 16` / `>> 4`
  descales round toward -inf on negative intermediates; the oracle's
  flat region (DC -299 at `DctQMask` 60) reconstructs to luma 16 only
  with truncating division (`>>` lands on 15). Both descales now use
  `/`.

### Added (clean-room round 384, 2026-07-03)

- **Golden-aware + FourMV MultiStream packets.** The `CoeffSink` routing
  is threaded through the remaining two P-frame bodies, so
  `encode_inter_frame_me_golden_multistream_packet` and
  `encode_inter_frame_me_fourmv_multistream_packet` emit the §6
  two-partition arrangement (BoolCoder or Huffman partition 2) for the
  full mode surface — every in-tree P-frame shape now exists in all
  three §5/§6 transports. Equivalence tests pin bit-identical decode vs
  the single-stream Golden and FourMV packets for both transports.

- **Inter-frame §13 re-training — real Figure-5 updates on P-frames.**
  `coeff_prob_update::encode_coefficient_prob_updates_from(current,
  target)` is the general inter-frame Figure-5 emitter (updates relative
  to the banks the previous frame left behind, including §12.2 band
  deltas relative to the previous custom assignment);
  `encode_coefficient_prob_updates_full` is now the `current = baseline`
  special case. `inter_encode::encode_inter_frame_packet_with_banks`
  emits a P-frame packet whose Figure-5 pass re-trains the §13 banks and
  whose tokens are coded against the re-trained values. A mid-GOP test
  drives I → P1-with-updates → P2-no-updates and pins exact
  reconstruction of both (P2 decodes only if the decoder persisted P1's
  mutated banks).

- **Cross-frame probability-bank persistence.** `Vp6Decoder` now carries
  the §13.2/§13.3/§13.3.3 coefficient banks (+ the §12.2 band assignment
  inside them), the §10 `probXmitted` bank, and the §11.2 two-axis MV
  bank **across frames**, per the spec's persistence rules: each I-frame
  resets every bank to its defaults; every frame's update sub-streams
  mutate the banks in place; each P-frame starts from the values the
  previous frame left behind ("For P-frames probXmitted values persist
  from the previously decoded frame" / "updates are applied in respect of
  the probability values used in the previous frame" / "persists from a
  keyframe (I Frame) to each subsequent interframe" / §12.2 inter deltas
  apply to the previous frame's custom scan). Tests pin a
  keyframe-with-real-Figure-5-updates GOP whose P-frames are coded
  against the *persisted* updated banks (exact reconstruction — baseline
  reseeding would desynchronise the stream), continuation across a second
  P-frame, and the reset at a new baseline keyframe.

- **FourMV neighbour representative — DOCS-GAP closed (errata #155).**
  The staged errata now disambiguates which vector a `CODE_INTER_FOURMV`
  macroblock contributes to a later MB's §10 Nearest/Near scan and §11
  differential reference: the **§10 chroma-derived average** of its four
  Y-block vectors (rounded away from zero) — the only MB-level vector the
  spec defines for a FourMV MB. The decoder's `resolve_motion` now records
  `fmb.chroma_mv` in the neighbour grid (previously `None`, deferring the
  choice) and the FourMV encoder mirrors it, so a FourMV MB can seed a
  following MB's implicit-MV modes. A lockstep test drives a FourMV MB
  whose average vector the surrounding MBs' motion matches, so their
  mode/MV coding runs against the representative on both sides.

- **MultiStream (§6) two-partition inter frames — decode + encode, zero-MV
  and motion-estimated, BoolCoder and Huffman.** The Figure 3 / Figure 4
  arrangement splits a P-frame's per-MB data: partition 1 carries **every**
  macroblock's §10 mode + §11 motion first, then partition 2 carries every
  block's §13 tokens. `inter_frame::decode_inter_frame_multistream[_with_refs]`
  is the two-pass driver (the fused single-stream walk is refactored onto
  the same shared per-MB steps: `decode_mb_prediction` threading
  `mv_grid`/`last_mode`, `decode_mb_blocks` threading the §14 `DcState` and
  reading from a `CoeffSource`); `Vp6Decoder::decode_packet` dispatches
  inter frames on `MultiStream`/`UseHuffman` exactly as keyframes. On the
  encoder, `tokenize_inter_block` splits tokenisation from emission and a
  `CoeffSink` (shared partition-1 coder / separate partition-2 coder /
  Huffman block collector) routes the zero-MV and ME bodies' tokens, so
  `encode_inter_frame_multistream_packet` and
  `encode_inter_frame_me_multistream_packet` emit both flavours without
  reordering the per-MB walk. Tests pin: an unchanged frame reproduces
  exactly through both transports, and a translated-source **ME**
  multistream P-frame decodes bit-identical to the single-stream ME packet
  (the §10/§11 decisions are transport-independent).

- **MultiStream (§6) two-partition keyframes — decode + encode, BoolCoder
  and Huffman.** `coeff_source::CoeffSource` abstracts the three §5/§6
  coefficient transports (single-stream partition-1 BoolCoder /
  partition-2 BoolCoder at `Buff2Offset` / partition-2 §7.2 raw-bit
  Huffman) behind one `decode_block`, and
  `intra_frame::decode_intra_frame_from_source` is the §6-general form of
  the keyframe driver (the §8 Figure-5 pass always rides partition 1 —
  in the Huffman arrangement partition 2 is raw bits and cannot carry
  BoolCoder-coded updates). `Vp6Decoder::decode_packet` now splits the
  packet at `Buff2Offset` and dispatches on `MultiStream`/`UseHuffman`;
  `intra_encode::encode_intra_frame_multistream` emits the two-partition
  packet (partition 1: tail + no-update Figure 5; partition 2: arithmetic
  tokens via the new tokenize/emit split, or `huff_coeff` raw-bit
  tokens), with the intra encoder's walk refactored into a
  coder-independent `tokenize_intra_frame` front half. Tests pin: the
  MultiStream-BoolCoder and MultiStream-Huffman keyframes decode
  bit-identical to the single-stream encoding at the same quantiser, a
  Huffman keyframe seeds a GOP whose P-frame decodes against it, and a
  corrupted `Buff2Offset` (past-end / inside-prefix) errors cleanly.

- **`huff_coeff` — the §13 Huffman coefficient coder, decode + encode.**
  When `UseHuffman == 1` the second partition codes DCT tokens as raw
  bits (§7.2) over trees derived from the same §13 probability banks the
  arithmetic path uses. `HuffmanCoeffTables::from_banks` builds the
  §13.2.2 per-plane DC trees (§13.1 `DCTTokenBoolTreeToHuffProbs` over
  the *raw un-contexted* `DcProbs[2][11]`), the §13.3.2
  `AcHuffTree[2][3][4]` over the Table 36 four-band split, and the
  §13.3.3.2 `ZeroHuffTree[2]`, with §7-mandated clamping of zero leaf
  products to 1. `decode_block_coefficients_huffman` implements the
  §13.2.2 DC + §13.3.2 AC block decode with the cross-block
  `CurrentDcRunLen` / `CurrentAc1RunLen` plane state
  (`HuffmanRunState`), the §13.3.3.2 zero runs, and the §13.4 EOB/DC0
  block-run decode (`decode_eob_or_dc0_run`, value space 1..=74);
  `encode_frame_blocks_huffman` is its frame-level bit-for-bit inverse,
  sizing the same-plane block runs by lookahead. Three spec-internal
  inconsistencies (ZRL symbol↔run off-by-one that would loop forever as
  printed, the `8 + R(6)` vs `9 + R(6)` long-escape base, and the
  missing `− 1` on the DC run store) are disambiguated against the
  §13.3.3.1 arithmetic value space and documented in the module docs.
  19 round-trip tests cover every token category, both ZRL bands, the
  long escape, cross-block and cross-plane runs, the >74 run split, the
  Prec-context transitions, updated (non-baseline) banks, and the
  clamped zero-probability leaf.

- **Inter-frame `Buff2Offset` conformance (§9 Table 3).** The InterHeader
  opens with `Buff2Offset R(16)`, present "If (MultiStream == 1) ||
  (SIMPLE_PROFILE == 1)" — but the profile half of the gate needs the
  `VpProfile` carried from the most recent I-frame (Table 3 omits it).
  `Vp6FrameHeader::parse_with_profile` threads that cross-frame state in
  (the packet-local `MultiStream` half needs no state and now works
  through plain `parse` too); `Vp6Decoder::decode_packet` supplies its
  carried profile. All four inter packet emitters now write the field
  (0 with a single partition), matching the intra encoder's
  Simple-profile shape — previously a Simple-profile P-frame packet
  omitted a field the spec marks always-present for that profile.

### Added (clean-room round 377, 2026-06-28)

- **`coeff_prob_update` — §8 Figure 5 coefficient-probability-update
  driver.** A new module that fuses the four already-implemented per-frame
  passes (§13.2 DC node, §12.2 scan, §13.3.3 ZRL, §13.3 AC) into the single
  spec-mandated order the §8 *Bitstream Map* (Figure 5, p. 20) fixes.
  `decode_coefficient_prob_updates` runs the sequence over one `BoolCoder`,
  mutating a `CoeffProbBanks` and returning the active §12.2 raster scan
  order; `encode_coefficient_prob_updates` / `_with_scan` / `_full` are its
  bit-for-bit inverses (minimal no-update prefix, custom-scan-only, and the
  general full-update form emitting real `NewNodeProbValue` records).
  `node_prob_update_representable` exposes the §13 `max(1, value*2)` target
  space (`{1} ∪ even`).
- **Top-level §8 Figure 1 / Figure 5 ordering wired into `decode_frame`.**
  `decode_keyframe` now consumes the Figure-5 coefficient-prob-update pass
  (a keyframe carries no §10/§11.2 tree — §10: I-frame MBs are implicitly
  intra); `decode_interframe` consumes the full inter prefix §10 *Mode
  Probability Updates* → §11.2 *Mv Tree* → Figure 5 before the per-MB walk.
  The intra encoder + every inter packet encoder emit the symmetric prefix
  (`emit_inter_pre_data_substreams`, `encode_no_mode_prob_updates`,
  `encode_no_mv_prob_updates`). The §8 ordering is fully spec-pinned, so no
  conformant `.vp6` fixture was needed (retiring the prior "needs a fixture"
  caveat for the ordering question).
- **`encode_intra_frame_with_banks` — real coeff-prob-update keyframe
  round-trip.** The general form of `encode_intra_frame`: emits the Figure-5
  updates carrying the baseline to an arbitrary representable
  `CoeffProbBanks` and codes the §13 tokens against those banks.
  `keyframe_with_coeff_prob_updates_round_trips` drives a keyframe whose
  prefix sets real DC/AC/ZRL `NewNodeProbValue` records through the
  top-level `decode_packet` and asserts the pixels reconstruct identically
  to the baseline-bank keyframe (node probs change only the entropy coding).

### Added (clean-room round 373, 2026-06-26)

- **`encode_inter_frame_me_golden_packet_refresh` + `Vp6Decoder::references`** —
  the §4 Golden-Frame **refresh** path for the Golden-aware encoder. The new
  packet variant emits an explicit `RefreshGoldenFrame` flag in the Table 3
  tail; when `true`, the decoder replaces its Golden Frame with the decoded
  P-frame after the frame (§4), letting a GOP periodically advance the Golden
  reference instead of pinning it to the keyframe.
  `encode_inter_frame_me_golden_packet` is now a thin wrapper passing
  `false`. `Vp6Decoder::references()` exposes the §4 reference state (previous +
  Golden buffers) so a Golden-aware encoder reads the **decoded** Golden Frame.
  A decode-side test drives keyframe → `RefreshGoldenFrame=1` P-frame and
  asserts the decoder's Golden buffer advances from the keyframe to the P-frame
  reconstruction.

- **`encoder` — `oxideav-core` `Encoder` registration** (`Vp6CodecEncoder`,
  `make_encoder` / `make_encoder_with_q`), closing the last named encoder
  "lack" (the `Encoder` shell was previously unregistered). The codec's
  `register_codecs` now wires `.encoder(make_encoder)` alongside the decoder, so
  `CodecRegistry::first_encoder` resolves a VP6 encoder under id `"vp6"`.
  `Vp6CodecEncoder` is a GOP-aware adapter: the first frame (and every
  `keyframe_interval`-th after) emits a §9 keyframe packet via
  `encode_intra_frame`; every other frame a motion-estimated P-frame packet via
  `encode_inter_frame_me_packet` against the **decoded** previous frame. The
  reference is maintained by decoding the encoder's own output through an
  internal `Vp6Decoder`, so the pixels the encoder predicts from are
  byte-identical to a downstream decoder's reconstruction — the round-trip is
  closed by construction. Geometry: §2 fragment-based (`h_fragments = width/8`),
  width/height must be multiples of 8 within the §9 8-bit fragment fields;
  4:2:0 only. `FilterConfig::bilinear()` is a new named constructor for the
  Simple/VP6.0 default (fixed bilinear, no loop filter) the encoder and decode
  path now share. Six tests: dimension validation, `output_params` propagation,
  a keyframe round-trip **through the `Encoder` → `Decoder` trait surfaces**, a
  keyframe → P-frame GOP (second frame is a P-frame, both decode in GOP order),
  registry resolution, and the NeedMore→Eof drain contract.

- **`rate_control` — per-frame quantiser selection (rate control)**, landing
  the encoder's third named "lack". The VP6 encoders take a fixed §9 `DctQMask`
  index; `rate_control` solves the inverse problem — pick the index that hits a
  bit budget / target frame size. It exploits the §15-table monotonicity (the
  dequant factor **decreases** with the index, so encoded partition size is
  weakly **monotonically non-decreasing** in `DctQMask`) to binary-search the
  `0..=63` index space against a caller-supplied `encode(q) -> Vec<u8>` closure:
  - `select_quantiser_for_budget(budget_bytes, encode)` — the **finest** index
    whose real output fits a hard byte cap (best quality under the cap),
    falling back to `MIN_Q` when even the coarsest index overflows.
  - `select_quantiser_for_target_size(target_bytes, encode)` — the index whose
    output size is **closest** to a target (over- or under-shoot), the
    nearest-size building block for a constant-bitrate driver.
  - `QuantiserChoice { q, size, bytes }` returns the chosen index **and** its
    already-encoded partition so the caller doesn't re-encode; `MIN_Q` / `MAX_Q`
    pin the `0..=63` bounds.
  Eleven tests: the budget/target selection logic against a synthetic monotone
  encoder (finest-that-fits, exact-boundary, min-q fallback, huge-budget →
  max-q, nearest-even-if-over, exact hit), a brute-force cross-check of the
  binary search across step shapes + budgets, error propagation, and a
  **real-encoder integration** test that confirms the in-tree intra encoder's
  output size is genuinely monotone in `q` and that the budget search picks a q
  whose real output fits the cap with the next-finer index overflowing it.
  `select_quantiser_for_budget` / `select_quantiser_for_target_size` /
  `QuantiserChoice` / `MIN_Q` / `MAX_Q` are re-exported from the crate root.
  An encoder-side policy layer over the §9 `DctQMask` field; no third-party VP6
  source consulted.

- **`inter_encode::encode_inter_frame_me_fourmv` — the FourMV P-frame
  encoder**, landing the encoder's second named "lack" (FourMV encode modes).
  A strict superset of `encode_inter_frame_me` that codes a macroblock
  `CODE_INTER_FOURMV` — four independent per-Y-block motion vectors — when that
  beats the best single-vector mode by `FOURMV_SAD_MARGIN` (`256`). Per MB:
  - **Per-block search** (`search_luma_block_mv` / `luma_block_sad`): each 8×8
    luma block runs its own box-then-¼-pel search; the four block SADs sum to
    the FourMV total. The single-vector decision (`decide_mb_mode`) runs in
    parallel, and the MB takes FourMV only when the four-vector total clears the
    margin **and** at least one block vector is non-zero.
  - **`fourmv::encode_fourmv_macroblock`** — the bit-for-bit inverse of
    `reconstruct_fourmv_macroblock`: it chooses each block's Table 10 mode by
    matching the target against the reconstructable candidates the decoder
    produces (zero → `CODE_INTER_NO_MV`; MB-level §10 Nearest/Near match →
    `CODE_INTER_NEAREST_MV` / `CODE_INTER_NEAR_MV`, no MV bits; else
    `CODE_INTER_PLUS_MV` with the §11.1 delta against the §11 differential
    reference), emits the four Table 10 two-bit codewords (raster order) then
    the per-block deltas, and returns the reconstructed `FourMvMacroblock` so
    the caller forms each luma block's residual against its **own**
    reconstructed vector and the two chroma blocks against the §10-averaged
    chroma MV. `encode_fourmv_block_mode` is the single-codeword inverse of
    `decode_fourmv_block_mode`.
  - A FourMV MB contributes **`None`** to the §10/§11 neighbour grid — exactly
    as the decoder records it (the FourMV MB-representative-MV is the documented
    §10 DOCS-GAP) — so the encoder and decoder neighbour contexts stay
    identical; the round-trip is correct **without** resolving that gap.
  - `encode_inter_frame_me_fourmv_packet` is the §9-self-describing dual.
  Six tests: an unchanged-frame exact reduction; a uniform-translation
  single-vector fallback; the canonical **divergent-block-motion** round-trip
  (the four luma quadrants move in four directions → `CODE_INTER_FOURMV` fires
  and reconstructs above a floor); a multi-MB FourMV packet through the
  top-level `Vp6Decoder`; the `encode_fourmv_macroblock` ↔
  `reconstruct_fourmv_macroblock` primitive round-trip; and the out-of-set mode
  rejection. All memory-bounded (per-block search is `O(range² · 64)` SAD adds).
  `encode_inter_frame_me_fourmv`, `encode_inter_frame_me_fourmv_packet` and
  `FOURMV_SAD_MARGIN` are re-exported from the crate root. Derived solely from
  the §10 Table 10 / §11 decode pipeline it inverts; no third-party VP6 source
  consulted.

- **`inter_encode::encode_inter_frame_me_golden` — the Golden-Frame-aware
  motion-estimated P-frame encoder**, a strict superset of
  `encode_inter_frame_me` that codes each macroblock against **either** the
  previous-frame **or** the Golden-Frame reference, whichever reconstructs more
  cheaply. This lands the encoder's first of three named "lacks" (Golden-frame
  encode modes). Per MB:
  - **Per-reference mode inputs** (`mb_inputs_for_ref`): the luma motion search,
    the §10 Nearest/Near candidate walk, the §11 differential reference, and
    every candidate's 16×16 luma SAD are all computed twice — once against
    `prev` filtered on `ReferenceBucket::InterLast`, once against `golden`
    filtered on `ReferenceBucket::InterGolden` — so each reference's
    Nearest/Near reuse and differential-MV reconstruction match what the decoder
    resolves for the corresponding `*_GOLD*` / previous-frame mode.
  - **Reference decision** (`decide_mb_mode_golden`): the cheaper-reconstruction
    reference wins, with a `GOLDEN_SWITCH_PENALTY` (`128`) SAD hysteresis so a
    marginal Golden win that loses the same-reference §14 DC / §11 differential-MV
    continuity (and costs marginally more §10 mode-tree bits) doesn't flip the
    reference. The previous-frame set emits `CODE_INTER_NO_MV` /
    `CODE_INTER_NEAREST_MV` / `CODE_INTER_NEAR_MV` / `CODE_INTER_PLUS_MV`; the
    Golden set emits `CODE_USING_GOLDEN` / `CODE_GOLD_NEAREST_MV` /
    `CODE_GOLD_NEAR_MV` / `CODE_GOLDEN_MV`.
  - **Reference-aware block emit**: `encode_inter_block` now threads the block's
    actual `ReferenceBucket` (was hardcoded `InterLast`) into the §14 DC
    prediction and the per-plane coded-DC grids, so a mixed previous/golden
    frame's same-reference DC filter is correct. The §10 probXmitted
    availability row is still resolved on the previous-frame bucket, exactly as
    the decoder indexes it.
  - `encode_inter_frame_me_golden_packet` is the §9-self-describing dual
    (Table 1 prefix + Table 3 tail `RefreshGoldenFrame = 0` / `UseHuffman = 0`),
    keeping the keyframe-seeded Golden Frame the body's `*_GOLD*` MBs predict
    from, so a Golden-aware P-frame decodes end-to-end through the top-level
    `decode_frame::Vp6Decoder`.
  Six round-trip / decision tests against `decode_inter_frame`: golden-wins
  (`prev` is unrelated → every MB switches to `CODE_USING_GOLDEN`, ≥30 dB luma
  + chroma); the identical-references reduction (unchanged frame is **exact**);
  a mixed previous↔golden frame crossing a reference transition mid-frame; a
  full keyframe → Golden-aware P-frame GOP recovering the source from the Golden
  reference through the §4 `ReferenceFrames`; and a `decide_mb_mode_golden`
  unit test pinning the switch-penalty boundary. All memory-bounded (small
  grids; the search is `O(range² · 256)` SAD adds per reference per MB).
  `encode_inter_frame_me_golden`, `encode_inter_frame_me_golden_packet` and
  `GOLDEN_SWITCH_PENALTY` are re-exported from the crate root. Derived solely
  from the decode pipeline it inverts (§10 Table 4 / §11 / §14 / §17.2); no
  third-party VP6 source consulted.

### Added (clean-room round 366, 2026-06-25)

- **`inter_encode::encode_inter_frame_me` — the motion-estimated P-frame
  encoder.** The encoder now performs a real per-macroblock motion search
  and emits `CODE_INTER_PLUS_MV` (a §11.1-coded motion vector) where it
  pays off, not just the `CODE_INTER_NO_MV` (zero-MV) shape
  `encode_inter_frame` produces. Per MB:
  - **Motion search** (`search_luma_mv`): a two-stage box-then-¼-pel
    search around `(0,0)` minimising the 16×16 luma SAD (`luma_mb_sad`),
    computed against the *same* `predict_inter_block_subpel` prediction
    the decoder forms — so the cost reflects the exact reconstruction
    pixels. Stage 1 is an integer-pel box over `±ME_SEARCH_RANGE`; stage 2
    refines over the eight ¼-pel neighbours of the best integer MV.
  - **Mode decision** (`decide_mb_mode`): weighs the §10 single-vector
    modes by reconstruction SAD plus a bit-cost model. The implicit-MV
    modes — `CODE_INTER_NO_MV` (zero), `CODE_INTER_NEAREST_MV` and
    `CODE_INTER_NEAR_MV` (reuse a §10 neighbour's vector) — read **no** MV
    bits, so the cheapest available implicit option wins unless an explicit
    `CODE_INTER_PLUS_MV` beats it by more than `ME_LAMBDA_SAD` (a
    Lagrangian λ proxy for the MV bit-cost) *and* its §11 differential
    delta is representable. The Nearest/Near candidate vectors come from
    the same `resolve_near_mvs` walk that supplies the §10 availability, so
    the encoder's implicit-mode reconstruction matches the decoder's
    Nearest/Near resolution exactly. The path is a strict superset of
    `encode_inter_frame`: a motionless frame reduces to all-zero-MV and
    round-trips exactly.
  - **Differential MV emit**: the encoded delta is
    `best_mv − differential_reference` (the nearest same-reference
    above/left neighbour via `select_diff_reference_mv_from_grid`, else
    zero), emitted with the round-366 `encode_mv_pair`. The encoder
    threads the **identical** §10/§11 `mv_grid`, `last_mode` and §10
    Nearest/Near availability (`resolve_near_mvs`) the decoder builds, so
    each MB's reconstructed MV, mode-context and residual match the
    decoder's reconstruction exactly.
  - The luma residual is formed against the chosen MV's prediction and the
    chroma residual against the single-vector chroma MV (the MB MV at
    ⅛-pel, §11.4) — the `predict_mv_luma`/`predict_mv_chroma` helpers
    (generalised from the old zero-MV-only helpers) call the decoder's
    `predict_inter_block_subpel`, guaranteeing bit-identical predictions.
  - `ME_LAMBDA_SAD` (`64`) and `ME_SEARCH_RANGE` (`8` whole samples)
    tune the cost margin and search extent.
  Round-trip tests against the decoder pin: the unchanged-frame exact
  round-trip (ME reduces to zero-MV); a translated-source reconstruction
  above a PSNR floor; ME at-least-matching the zero-MV encoder on a
  translated gradient; the single-MB path; the multi-MB shared-motion
  **differential-reference** path; a larger uniform-motion frame
  exercising the **implicit Nearest/Near modes**; and a full keyframe → ME
  P-frame GOP through `decode_inter_frame_with_refs`. Four `decide_mb_mode`
  unit tests pin the mode-selection logic (search-wins → PlusMv;
  within-margin → Nearest; nothing-better → Zero; lowest-SAD implicit →
  Near). All memory-bounded (small grids; the search is `O(range² · 256)`
  SAD adds per MB). `encode_inter_frame_packet` is now re-exported from the
  crate root. Derived solely from the decode pipeline it inverts; no
  third-party VP6 source consulted.
- **`inter_encode::encode_inter_frame_me_packet`** — the §9-self-describing
  motion-estimated P-frame packet (the ME dual of
  `encode_inter_frame_packet`): the Table 1 raw prefix + Table 3 BoolCoder
  tail (`RefreshGoldenFrame = 0` / `UseHuffman = 0`) prepended to the ME
  data partition, so a motion-estimated P-frame decodes end-to-end through
  the top-level `decode_frame::Vp6Decoder`. `encode_inter_frame_me` is
  refactored to delegate to a shared `encode_inter_frame_me_body` taking a
  header-tail prelude closure (mirroring the zero-MV `encode_inter_frame`
  split). A new `decode_frame` test drives a full keyframe → translated
  ME-P-frame GOP through one `Vp6Decoder` (against its
  `InterProbs::keyframe()` / header-derived `FilterConfig`, no caller-side
  filter wiring) above a quantiser-bounded floor.

- **`mv_encode` — the §11.1 motion-vector component encoder**, the
  bit-for-bit inverse of [`mv_decode::decode_mv_component`]. The
  foundational primitive the motion-estimated P-frame encoder needs to
  emit a real (non-zero) MV. Surfaces:
  - `encode_mv_component(enc, component, probs)` — emits the
    `B(IsMvShortProbs)` short/long discriminator (short for
    `|component| <= 7`, long for `8..=255`), the magnitude path, then the
    `B(MvSignProbs)` sign bit (`0` for non-negative including zero, `1`
    for negative — a zero still emits a sign per §11.1 and negates to a
    zero). The short path mirrors the Figure 11 tree; the long path emits
    bits in the decoder's `[0,1,2,7,6,5,4]` order and respects the
    implicit-bit-3 rule (bit 3 is transmitted only when the high nibble is
    non-zero, so magnitudes `8..=15` reconstruct from their low three bits
    as `(m & 7) | 8 == m`).
  - `encode_mv_pair(enc, dx, dy, probs)` — the `(dx, dy)` dual of
    [`mv_decode::decode_mv_pair`] (X then Y).
  - `MAX_MV_MAGNITUDE` (`0xFF`) — the bit-arithmetic ceiling.
  Five round-trip tests pin every short component (`-7..=7`),
  representative long magnitudes (both signs, both axes), the full
  unsigned `0..=255` range, the zero case, and a mixed-pair sweep — each
  encoded then decoded back through `decode_mv_component`/`decode_mv_pair`
  to the exact input. Memory-bounded (each working set is a few bytes).
  Derived solely from the §11.1 decode functions it inverts plus the §7.3
  `BoolEncoder`; no third-party VP6 source consulted.

### Added (clean-room round 363, 2026-06-23)

- **`oxideav-core` `Decoder` registration** (`decoder::Vp6CodecDecoder`,
  `register`): the framework `Decoder` trait over `decode_frame::Vp6Decoder`
  — `send_packet` queues a compressed VP6 frame, `receive_frame` decodes it
  into a 3-plane 4:2:0 `VideoFrame`. `register()` (previously a no-op)
  installs the codec under id `"vp6"` with the On2 / Flash / Matroska
  container tags (`VP60` / `VP61` / `VP62` / `vp6f` / `V_VP6`). The crate's
  `Error` now maps into `oxideav_core::Error` (`Truncated` → invalid,
  `NotImplemented` → unsupported).
- **Top-level per-frame assembly** (`decode_frame::Vp6Decoder`): a stateful
  decoder that sequences the §9 header prefix parse (`Vp6FrameHeader::parse`)
  → BoolCoder construction over the partition → §9 header-tail parse
  (`Vp6HeaderTail::parse_with`) → keyframe/inter dispatch
  (`decode_intra_frame` / `decode_inter_frame_with_refs`), threading the §9
  cross-frame profile/version (Table 3 omits both — inherited from the most
  recent I-frame) and the §4 `ReferenceFrames` across `decode_packet` calls.
  Updates the §4 previous-frame + Golden Frame buffers per the
  `RefreshGoldenFrame` rules after every frame. Targets the no-probability-
  update / single-partition / BoolCoder-coefficient frame shape the in-tree
  encoders produce (the §10/§11.2/§13 update sub-streams' exact Figure-5
  ordering still awaits a conformant `.vp6` fixture; `MultiStream`/`UseHuffman`
  surface as `NotImplemented`). A keyframe → P-frame GOP now round-trips
  end-to-end through one `Vp6Decoder` instance.
- **P-frame packet encoder** (`inter_encode::encode_inter_frame_packet`): the
  §9 InterHeader emit (Table 1 raw prefix `FrameType=1`/`DctQMask`/
  `MultiStream=0` + the Table 3 BoolCoder tail `RefreshGoldenFrame`/
  `UseHuffman`) prepended to the existing data partition, so an encoded
  P-frame is a self-describing packet `Vp6Decoder::decode_packet` consumes.
  `encode_inter_frame` is refactored to delegate to a shared
  `encode_inter_frame_body` taking a header-tail prelude closure (the two
  tail bits ride the data partition's BoolCoder, which is not byte-splittable).
- **`InterProbs::keyframe()`** and **`ReferenceFrames::coded_fragments()`**
  constructors/accessors: the inter dual of `IntraProbs::keyframe()` (the §10
  baseline `probXmitted` + §11 default MV banks + §13 baseline coefficient
  banks) and the reference-frame geometry an inter frame inherits. Two
  duplicated test helpers collapse onto the new constructor.
- **Multi-frame GOP coverage** through `Vp6Decoder`: a three-frame (I, P, P)
  GOP test proves the §4 previous-frame buffer advances after every
  `decode_packet` (P2 predicts from the decoded P1, not a stale keyframe),
  and a content-changed P-frame test exercises the residual path (non-zero
  coefficients on the zero-MV prediction) through the top-level driver above
  a quantiser-bounded PSNR floor.

### Fixed (clean-room round 359, 2026-06-22)

- §11.3 prediction loop filter: the `BoundaryX` / `BoundaryY` straddled-edge
  offsets are now derived from the spec's own **round-toward-zero**
  whole-pixel reduction (`mVx = (mx>0) ? (mx>>shift) : -((-mx)>>shift)`) via
  the new `loopfilter::boundary_whole_pixel`, instead of reusing §11.4's
  arithmetic-shift floor (`MvX >> MvShift`). The two agree for non-negative
  MV components but diverge for negative ones whose magnitude is not a
  multiple of `2^MvShift` — e.g. a luma MV component of `-1` floors to `-1`
  (yielding `BoundaryX == 1`) but truncates to `0` (yielding `BoundaryX ==
  0`, no straddled boundary). The previous code filtered a boundary §11.3
  leaves unfiltered for such negative MVs, corrupting the prediction signal;
  the loop-filtered sub-pixel MC path (`predict_inter_block_subpel`) now
  matches §11.3 exactly across the full signed MV range.

### Added (clean-room round 356, 2026-06-21)

- `PredictionFilter::resolve` — the §11.4 bridge from the *signalled*
  Advanced-profile filter selector (the decoded header `PredictionFilter`)
  to the *operative* `PredictionFilterPolicy` the per-block fractional-pixel
  predictor consumes. Applies the three header→runtime conversions:
  (1) MV-size threshold to ¼-pixel units — `(1 << (thresh-1)) << 2`, or the
  "no restriction" branch `((MAX_MV_EXTENT >> 1) + 1) << 2` when the field
  is zero; (2) `FilterVarThresh = PredictionFilterVarThresh << 5` (the §11.4
  formula as printed names `MvSizeThresh`, but the surrounding prose, the
  zero/non-zero gate field, and the `FilterVarThresh` result name are all
  the *variance* threshold — using `MvSizeThresh` would force a zero
  MV-size field to make `FilterVarThresh == 0` regardless of the variance
  field, contradicting the prose; the internally-consistent reading shifts
  `PredictionFilterVarThresh`); (3) bicubic alpha index — the VP6.2
  `PredictionFilterAlpha` when present, else the VP6.1 default index 16.
  `Fixed` / `NotSignalled` selectors resolve to the corresponding fixed
  family (Simple profile / omitted selector → bilinear per §11.4).
- `interp::var_16_point_clamped` — the §11.5 out-of-range edge form of the
  §11.4 16-point variance metric. Each sampled position is edge-clamped
  into the reference buffer's valid range, replicating the §11.5
  edge-extension sample instead of indexing out of bounds when an
  unrestricted (or out-of-spec long) motion vector places the
  whole-sample-aligned variance window past a buffer edge. Bit-identical to
  `var_16_point` for any window fully inside the buffer.
  `PredictionFilterPolicy::select` now takes a signed (`i64`) variance
  position and reads through the clamped form, so the Advanced-profile
  filter selection never panics on an edge/long-MV block.
- `inter_frame::FilterConfig::from_header` — the per-frame bridge from a
  decoded `Vp6HeaderTail` (+ the frame's `DctQMask`) to the operative
  §11.3/§11.4 `FilterConfig`: it resolves the §11.4 family policy via
  `PredictionFilter::resolve` and enables the §11.3 prediction loop filter
  (carrying the quantiser index) only when the tail's `LoopFilter` reports
  `Enabled`. Lets the GOP decode path build its filter configuration from
  the real header instead of a hardcoded family. `Vp6HeaderTail`,
  `LoopFilter` and `PredictionFilter` are now re-exported from the crate
  root so the builder is reachable.
- `gop_filter_config_from_header_round_trips` — a keyframe→P-frame GOP test
  whose decode `FilterConfig` is built from the *decoded keyframe header*
  via `FilterConfig::from_header` (not a hardcoded family). It asserts the
  Simple-profile keyframe resolves to §11.4 bilinear with no loop filter
  and that an unchanged P-frame decoded with the header-derived config
  reconstructs the keyframe bit-for-bit — an end-to-end check of the new
  resolve→from_header wiring.

### Added (clean-room round 350, 2026-06-20)

- `mode_encode` — the bit-for-bit inverse of `mode_decode` (§10 Figure 10
  `VP6_DecodeMode`): `encode_mode` / `encode_mode_descend` /
  `encode_mode_from_probs` emit the root "same as last" bit and the
  node-path bits that drive the decoder's nine-node descent to a mode's
  leaf. The same-as-last fast path takes the minimal one-bit encoding when
  a mode repeats `last_mode`. Round-trip tests pin every
  `(mode, availability, last_mode)` triple against the decoder.
- `inter_encode::encode_inter_frame` — the top-level P-frame encoder, the
  inter-frame dual of `encode_intra_frame`. Emits the simplest valid
  P-frame (every macroblock `CODE_INTER_NO_MV`: zero MV, predicted from
  the previous-frame reconstruction) as the BoolCoder data partition
  `decode_inter_frame` consumes. Per block: residual = source − zero-MV
  prediction (the same `predict_inter_block_subpel` call the decoder uses)
  → §16-dual forward DCT → §15-inverse quantise → §14 DC delta → §13 token
  emit; §10 mode emit via `mode_encode`. The §10 Nearest/Near walk skips
  zero MVs, so availability is `Neither` for every MB. End-to-end tests
  feed the partition to `decode_inter_frame`: unchanged frames round-trip
  exactly, changed frames clear a PSNR floor, finer quantisers improve
  PSNR, and a full keyframe → P-frame GOP (encode/decode I → seed §4
  refs → encode/decode P against the decoded keyframe) round-trips with an
  unchanged P-frame reproducing the keyframe reconstruction bit-for-bit.
- `inter_frame::BorderedRef::{y,u,v}_plane` — accessors exposing the
  §11.5-bordered reference planes `(samples, stride, origin)` so the
  encoder forms the zero-MV prediction without duplicating the border
  construction.

### Added (clean-room round 347, 2026-06-20)

- `forward_dct::fdct_block` — the §16-dual forward DCT for the encoder.
  A separable orthonormal 8-point DCT-II per axis, scaled to invert the
  §16 integer IDCT's observable `1/32` pure-DC gain, in `f64` rounded to
  nearest. `idct(fdct(x))` recovers the input to ≤3 LSB per sample.
- `token_encode` — the bit-for-bit inverse of the `dct_decode` token
  trees: DC tree walk, AC tree walk (incl. the `EncodedCoeffs>1 &&
  Prec==WasZero` implicit-1 shortcut), magnitude/sign emit, §13.3.3.1
  zero-run emit, and the per-block `encode_block_coefficients` mirroring
  the decoder's `EncodedCoeffs` loop (Prec evolution, inclusive zero-run
  choreography, EOB-vs-natural-full-block termination). Round-trips every
  DC value, all category magnitudes/signs, both zero-run bands, and full
  coefficient blocks.
- `intra_encode::encode_intra_frame` — the top-level I-frame encoder, the
  stage-for-stage inverse of `decode_intra_frame` (−128 level shift →
  §16-dual forward DCT → §15-inverse quantise → §14 DC delta → §13 token
  emit → §9 header emit). Emits the simplest valid I-frame shape (Simple
  profile, single partition, VP6.0, default scan, keyframe-baseline
  probs). Round-trip tests drive `encode_intra_frame → Vp6FrameHeader::
  parse → Vp6HeaderTail::parse_with → decode_intra_frame` and measure
  PSNR: flat frames exact, a 32×32 patterned frame at q=48 hits ~44 dB
  luma / ~45 dB chroma, quantiser-monotonicity holds.

### Added (clean-room round 343, 2026-06-20)

- `inter_frame` — `ReferenceFrames` + `decode_inter_frame_with_refs`:
  the §4 golden-frame bookkeeping. `ReferenceFrames` tracks the
  previous-frame and Golden Frame reconstructions and applies the §4
  update rules (every decoded frame becomes the new previous-frame
  reconstruction; the Golden Frame is replaced on an I-frame — which
  seeds it — or when the InterHeader's `RefreshGoldenFrame` flag is
  set). `decode_inter_frame_with_refs` decodes a P-frame against a
  `ReferenceFrames`, building the §11.5 borders internally so the caller
  threads only the reference state across frames.
- `inter_frame` — `decode_inter_frame` + `InterProbs` + `BorderedRef` +
  `FilterConfig`: the **fused inter (P-frame) per-macroblock decode
  loop**, the inter-frame analogue of `decode_intra_frame`. Walks the MB
  grid in §8 single-stream order, decoding per MB the §10 coding mode
  (against the per-frame `probXmitted` bank + §10 Nearest/Near
  availability resolved from the MV grid built so far), the §11 motion
  state (single-vector via `reconstruct_macroblock_mv`, FourMV via
  `reconstruct_fourmv_macroblock`, or none for intra), the six §13 block
  coefficients with §14 DC prediction, and §17 reconstruction (intra →
  §17.1 `+128`; inter → `predict_inter_block_subpel` motion compensation
  against the previous-frame/golden-frame `BorderedRef` + §17.2/§17.3/
  §17.4 recombine). Threads the §14 DC neighbour grid (now carrying each
  block's `ReferenceBucket` for the same-reference filter) and the
  §10/§11 MV neighbour grid across the walk. P-frames now decode
  end-to-end to real pixels. FourMV MBs reconstruct their pixels
  correctly but contribute no neighbour-representative MV (documented
  §10 DOCS-GAP).
- `inter` — `predict_inter_block_subpel` + `PredictionFilterPolicy` +
  `FilterFamily`: the full §17.4 fractional-pixel motion-compensation
  block predictor. Decomposes the MV into its §11.4 whole-sample-aligned
  part and fractional phase, applies the §11.3 prediction loop filter at
  the straddled `BoundaryX`/`BoundaryY` 8×8-grid edges (when enabled and
  the MV is non-zero, writing to a separate working window per §11.3),
  and interpolates to the fractional phase with the §11.4 bilinear or
  bicubic filter family. `PredictionFilterPolicy` resolves the §11.4
  `AutoSelectPMFlag` decision — `Fixed` (one family per frame) or
  `AutoSelect` (per-block from the §11.4 MV-size threshold and
  `Var16Point` prediction-block variance threshold).

### Added (clean-room round 339, 2026-06-19)

- `intra_frame` — `decode_intra_frame` + `IntraProbs`: the full
  intra-frame (I-frame) per-macroblock decode loop. Walks the MB grid in
  raster order and, for each MB's six blocks (four luma raster
  TL/TR/BL/BR, then U, then V — §13 page 58), runs the §13 coefficient
  decode → §14 DC prediction → §15 dequant → §16 IDCT → §17.1
  reconstruct → §2 raster-assembly chain, threading the §14/§13.2
  left/above block-grid neighbours through explicit per-plane coded-DC
  grids. Single BoolCoder partition (`MultiStream == 0`).
- `fourmv` — `reconstruct_fourmv_macroblock` + `FourMvMacroblock`:
  resolves a `CODE_INTER_FOURMV` macroblock's four per-block coding
  modes (Table 10) and motion vectors (NoMv → `(0,0)`; PlusMv → §11.1
  delta + §11 differential reference; Nearest/Near → MB-level §10 walk),
  plus the derived chroma MV. Defers the FourMV MB-representative-MV
  choice (a documented §10 DOCS-GAP).
- `inter` — `reconstruct_inter_macroblock` + `RefPlane` +
  `ReconstructedMacroblock`: §17.2/§17.3 integer-MV macroblock
  reconstruction, driving the per-block §11.5-clamped prediction fetch +
  §17 recombine across one MB's six blocks with the §11.4 luma/chroma
  MV-shift split.

### Fixed (round 339)

- `idct` — the §16 inverse-DCT multiply now widens to `i64` before the
  `>> 16` descale. A conformant §15-dequantized coefficient (≈ 2114 ×
  376) times the Q16 cosine constant 64277 overflowed the previous
  `i32` product; value-identical for non-overflowing inputs.

### Added (clean-room round 336, 2026-06-18)

- `mv_decode` — `reconstruct_macroblock_mv`, the §10/§11 per-MB
  motion-vector resolution stage that sequences behind the round-331
  mode-decode pass. Given one MB's decoded `CodingMode`, its `(row,
  col)`, the per-axis `[MvProbs; 2]` bank and a neighbour-grid accessor,
  it dispatches on the mode's `MvSource`: `Intra`/`Zero`
  (`CODE_INTRA` / `CODE_INTER_NO_MV` / `CODE_USING_GOLDEN`) resolve to
  `(0,0)` reading **no** BoolCoder bits; `New` (`CODE_INTER_PLUS_MV` /
  `CODE_GOLDEN_MV`) reads a §11.1 `(dx,dy)` delta with `decode_mv_pair`,
  selects the §11 differential reference MV
  (`mv_diff::select_diff_reference_mv`), and adds them
  (`reconstruct_diff_mv`); `Nearest`/`Near` reuse the §10 neighbour
  walk (`near_mv::resolve_near_mvs`), falling back to `(0,0)` when the
  Nearest/Near vector is undefined. Returns a `MacroblockMv`
  (`MotionVector` + `ReferenceBucket`) with an `as_neighbour` view that
  feeds the next MB's neighbour grid. `CODE_INTER_FOURMV` returns
  `Error::NotImplemented` (its MB-representative neighbour vector is a
  DOCS-GAP). Sourced exclusively from `docs/video/vp6/vp6_format.pdf`
  §10 (Table 4) / §11 (intro + §11.1) and the staged errata #35.
- `modes` — `CodingMode::reference_bucket` (Table 4 mode →
  `ReferenceBucket`: intra / Golden / previous-frame) and
  `CodingMode::mv_source` + the `MvSource` enum (mode → one of
  `Zero` / `New` / `Nearest` / `Near` / `FourMv` / `Intra`), the
  classifiers the per-MB MV resolver dispatches on.

### Added (clean-room round 331, 2026-06-18)

- `mode_decode` — `decode_macroblock_modes`, the §10 frame-level
  macroblock mode-decode pass and first stage of the per-MB decode
  driver. Walks the `mb_cols × mb_rows` macroblock grid in spec raster
  order (§13), threading `last_mode` (feeds the root
  `B(probModeSame[type][lastmode])` "same as last" decision and the
  `ModeDecisionTree[type][lastmode]` row selection, advancing to each
  MB's just-decoded mode) and resolving each MB's §10 Table 5
  `ModeAvailability` via a caller-supplied closure (the spec couples
  the Nearest/Near reference-frame filter to the mode being decoded,
  so availability resolution — typically backed by
  `near_mv::resolve_near_mvs_from_grid` over the MV grid built so far —
  is delegated to the caller). The first-MB `last_mode` seed is a
  parameter so no unstated constant is baked in. Returns one
  `CodingMode` per MB in raster order; empty grids read no BoolCoder
  bits. Sourced exclusively from `docs/video/vp6/vp6_format.pdf`
  §10/§13 and the staged errata #35.

### Added (clean-room round 36, 2026-06-16)

- `bool_coder` — `BoolEncoder`, the binary arithmetic **encoder** that
  is the exact algebraic inverse of the §7.3 `VP6_DecodeBool` decoder.
  The §7.3 spec is a decoder specification (no encoder pseudocode), but
  a binary arithmetic encoder is uniquely determined by the decoder it
  must feed: `BoolEncoder` is derived solely from the in-tree §7.3
  decode equations, mirroring the same `Split = 1 + (((Range-1) *
  Probability) >> 8)` interval subdivision (operative `>> 8` per errata
  #35), and emitting a byte stream that `BoolCoder` reconstructs
  bit-for-bit. Surface:
  - `BoolEncoder::new` / `Default` — fresh encoder (`Range = 255`).
  - `encode_bool(bit, probability)` — dual of `decode_bool`: selects the
    chosen sub-interval (`Bit = 0` keeps `low`, `range = Split`; `Bit =
    1` does `low += Split << 24`, `range -= Split`) and renormalizes at
    bit granularity to mirror the decoder's `Range *= 2; Value *= 2`.
    Renormalization carries out of the 32-bit window ripple correctly by
    storing committed output as an explicit bit list and resolving a
    carry as `+1` from the tail toward the head.
  - `encode_b1(bit)` / `encode_b(value, n)` — duals of `decode_b1` /
    `decode_b`, fixed-probability-128, MSB-first, `n` capped at 32.
  - `finish()` — drains the 32-bit window and packs the committed bits
    MSB-first into bytes, zero-padding the final partial byte and
    padding to the decoder's 4-byte `VP6_StartDecode` minimum.
  Eight encode→`BoolCoder`-decode round-trip tests pin the inverse
  relationship across single bits, fixed and pseudo-random
  bit/probability sequences, `b(n)` widths, a 120-symbol high-probability
  carry-propagation stress, an empty encode, and an interleaved
  `bool`/`b(n)` syntax mix. All working sets are bounded (≤ a few hundred
  bits) so the encoder's `O(output_len)` footprint stays trivial, well
  under the test memory cap.

### Changed (clean-room round 35, 2026-06-16)

- `bool_coder` — corrected the §7.3 `VP6_DecodeBool` `Split` formula to
  the operative `>> 8` shift, following the rewritten errata #35
  (Issue #115): the spec PDF prints `Split = 1 + (((Range-1) *
  Probability) >> 7)`, but that shift is a transcription typo. Under the
  printed `>> 7` the coder is degenerate — at `Probability = 128` it
  gives `Split = Range` (an empty `Bit = 1` interval) and at
  `Probability = 255` it gives `Split > Range` (a negative interval).
  The operative `>> 8` keeps `1 ≤ Split ≤ Range − 1` for every
  `Probability ∈ [1,255]`, `Range ∈ [128,255]`, and makes probability
  128 the equiprobable half-interval point. Added
  `split_bounded_for_all_probabilities_and_ranges`, a grid test that
  pins the `1 ≤ Split ≤ Range − 1` invariant so a regression back to
  `>> 7` fails immediately.
- Threaded the correction through every BoolCoder-consuming module's
  documentation and tests (`frame_header`, `dct_decode`, `mv_decode`,
  `mode_decode`, `prob_update`, `mv_prob_update`, `mode_prob_update`,
  `scan_update`, `block_decode`, `fourmv`, crate-root `lib.rs`): the
  earlier "`>> 7` is correct / probability 128 is `Split = Range` /
  `Split = 507 > Range` self-correcting corner" commentary was based on
  the now-superseded errata reading and has been replaced with the
  `>> 8` analysis. Three tests whose expected decoded outcomes depended
  on the old `>> 7` Split values were re-derived against `>> 8` with
  fresh hand-traced fixtures (`dct_decode::decode_ac_coefficient_zero_run_outcome`,
  `dct_decode::decode_ac_zero_run_greater_than_eight_with_zero_extrabits_yields_nine`,
  `scan_update::walk_applies_updates_under_forced_flags`).

### Added (clean-room round 34, 2026-06-15)

- `frame_header` — the §9 BoolCoder-coded `b(n)` frame-header tail
  (`Vp6HeaderTail`), the second half of the frame header that the
  earlier rounds deferred at the BoolCoder boundary. The §7.3 `Split`
  degeneracy that prose-blocked this tail is resolved by the staged
  errata #35 (the operative shift is `>> 8`; probability 128 is the
  half-interval point — see the round-35 "Changed" note above for the
  correction history), so the fixed-probability `b(n)` fields decode
  cleanly through the existing `BoolCoder`.
  - `Vp6FrameHeader::raw_prefix_len` (field + accessor) — the
    byte-aligned offset at which the BoolCoder partition begins (1 byte
    Inter, 2 bytes Intra-no-Buff2Offset, 4 bytes Intra-with-Buff2Offset).
  - `Vp6HeaderTail::parse(tail_bytes, is_keyframe, profile, version)`
    and `parse_with(&mut BoolCoder, …)` — decode the Table 2
    (IntraHeader) geometry/scaling fields (`VFragments`, `HFragments`,
    `OutputVFragments`, `OutputHFragments`, `ScalingMode`), the Table 3
    (InterHeader) `RefreshGoldenFrame` + Advanced loop-filter selectors
    (`UseLoopFilter`, `LoopFilterSelector`), the `AutoSelectPMFlag`-gated
    prediction-filter selectors (with the VP6.2-only gating that applies
    to InterHeaders), the VP6.2 `PredictionFilterAlpha b(4)`, and the
    trailing Table 1 `UseHuffman b(1)`. `parse_with` leaves the borrowed
    coder positioned for the §10 mode data that follows in partition 1.
  - `PredictionFilter` / `LoopFilter` enums modelling the
    Advanced-profile selectors verbatim from Tables 2/3.
  - Seven capture-then-verify unit tests (no range encoder; 16-byte
    fixed partition) covering Intra/Simple, Intra/Advanced,
    Inter/Advanced VP6.0 and VP6.2 field orderings, `parse_with`
    coder-position continuity, truncation, and an end-to-end
    prefix-then-tail parse.

### Added (clean-room round 33, 2026-06-15)

- `mode_decode` module — the §10 `VP6_DecodeMode` macroblock
  coding-mode traversal (Figure 10), the **ninth BoolCoder-consuming
  layer** and the resolution of the DOCS-GAP candidate carried forward
  from round 21 (the page-36 pseudo-code's `B(Stats[0])` /
  `B(Stats[2])` else-branch indentation ambiguity). Resolved entirely
  from the staged spec: Figure 10 (page 34) plus the
  `ModeDecisionTree[k][i][n]` node-mass equations (page 35) pin the
  nine-node binary-tree topology unambiguously, and the BoolCoder
  polarity (the node probability's numerator is always the
  `B(node) == 0` left-subtree mass) fixes which bit follows which
  child; tracing the literal page-36 pseudo-code under that polarity
  reproduces the same leaves, confirming the indentation was a
  typesetting artifact.
  - `decode_mode(bc, prob_mode_same, last_mode, &stats)` — the full
    traversal: root `B(probModeSame)` "same as last" bit (repeat
    `last_mode` on 1, descend on 0) then the Figure 10 tree walk.
  - `descend_mode_tree(bc, &stats)` — the nine-node descent in
    isolation, for callers that have already consumed the root bit.
  - `decode_mode_from_probs(bc, &prob_xmitted, availability,
    last_mode)` — convenience deriving `probModeSame` and the nine
    `ModeDecisionTree` node probabilities from a live
    `probXmitted[3][20]` table (the bank round 32's `mode_prob_update`
    mutates per frame) via round 10's `modes` helpers before walking.
  - Composes only round-15's `BoolCoder::decode_bool` over round 10's
    static `modes` surface — no new spec material beyond §10 (Figure
    10, the node-mass equations, the `VP6_DecodeMode` pseudo-code), no
    new errata, no third-party VP6 source consulted.
- 12 new unit tests pinning: the root "same as last" repeat; the
  descent on a root miss; the two extreme leaves (all-0 →
  `InterNoMv`, all-1 → `GoldNearMv`); the node-0 partition (inter vs
  Golden subtree); the `decode_mode` vs `decode_mode_from_probs`
  agreement with byte-exact BoolCoder-state match across all
  `availability × last_mode` pairs against the baseline `probXmitted`;
  determinism; the canonical-mode output invariant; the truncation
  surface; and the same-as-last single-bit short-circuit.
- Test count: 574 → 586. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean.

### Added (clean-room round 32, 2026-06-15)

- `mode_prob_update` module — the §10 per-frame **mode-probability-update
  bitstream** (Tables 7 / 8 / 9 and the Figure 9 magnitude tree), the
  deferred "Mode Probability Updates bitstream" round 10 flagged. The
  eighth BoolCoder-consuming layer; composes only round-15's
  `BoolCoder::decode_bool` / `decode_b` over the verbatim §10 static
  data in `modes`.
  - `update_mode_probs` — the full driver, walking the three
    `ModeAvailability` situations in spec order and mutating the
    persistent `probXmitted[3][20]` table in place.
  - `update_mode_probs_for_situation` — one situation's Table 7 / Table 8
    walk: `SetNewBaselineProbs B(174)` → optional `WhichVector b(4)`
    copying `VP6_MODE_VQ[situation][which]` into the row;
    `VectorUpdatesPresentFlag B(254)` → optional 20 Table 9 records.
  - `decode_mode_prob_update_value` — one Table 9 record:
    `UpdateFlag B(205)`, then on update `Sign B(128)` + Figure 9
    `Difference`, applied to the current value.
  - `decode_mode_prob_difference` — the verbatim Figure 9 magnitude-tree
    decode (`B(171)` / `B(83)` / `B(199)` / `B(140)` / `B(125)` /
    `B(104)` / `b(7)` escape), returning the already-signed delta.
  - `apply_prob_difference` — `0..=255` clamp (probXmitted entries are
    counts the §10 tree builder reads through `1 + sum` denominators, so
    `0` is valid — distinct from directly-read `B(prob)` node
    probabilities, which forbid `0`).
  - Flag/bit constants `SET_NEW_BASELINE_PROBS_FLAG` (174),
    `VECTOR_UPDATES_PRESENT_FLAG` (254), `UPDATE_FLAG_PROB` (205),
    `SIGN_PROB` (128), `WHICH_VECTOR_BITS` (4), `LONG_DIFFERENCE_BITS`
    (7), `FIGURE9_NODE_PROBS`.
  - **Spec inconsistency noted (resolved by dimension):** Table 9's
    figure region labels the record list "Ten Sets of:", but the §10
    prose states twice that `ModeProbUpdateVector` is "20 sets of
    probability updates" and `probXmitted[3][20]`'s second dimension is
    20 (ten modes × {same-as-prior, different-from-prior}). The 20-count
    is the consistent reading; the module walks 20.
- **Transcription fix:** corrected six `VP6_MODE_VQ` baseline-bank
  vectors in `modes.rs` that had single-element shifts / trailing-value
  errors against the §10 spec listing (situation 0 vectors 1 & 10;
  situation 1 vectors 3 & 7; situation 2 vectors 7 & 15). All 48
  vectors now match the spec verbatim. This bank is copied into
  `probXmitted` by the new `SetNewBaselineProbs` path, so the
  correctness matters for the layer landing this round.
- 16 new unit tests pinning: the flag/bit constants vs spec; the
  `0..=255` clamp (low + high saturation, zero pass-through); Figure 9
  all-zero-stream fall-through to `sign * 24`, sign negation,
  multiple-of-4 structural invariant, determinism, truncation surface;
  the per-value flag-clear no-op (one bit consumed); the per-situation
  no-baseline-no-update identity; the 20-record-length pin; the full
  driver's all-zero identity, range/completion, determinism (with
  BoolCoder position match) and the situation-indexed baseline-bank
  shape. The high-probability flag-set / Figure-9 1-branch paths are
  exercised only against a real conformant bitstream — the same
  synthetic-stream limitation rounds 22 / 26 / 27 documented for
  high-prob (`> 128`) reads under the round-15 BoolCoder (errata #35).
- Test count: 558 → 574. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean. No new spec material
  read beyond §10 (Tables 5/6/7/8/9, Figure 9) of the staged
  `vp6_format.pdf`; no errata change required.

### Added (clean-room round 31, 2026-06-14)

- `scaling` module — the §9 `ScalingMode` / `Output*Fragments` static
  output-scaling surface (Table 2, page 24). BoolCoder-independent: pure
  integer arithmetic over already-decoded fragment counts.
  - `ScalingMode` — the four §9 named modes (`MaintainAspectRatio` (0),
    `ScaleToFit` (1), `Center` (2), `Other` (3)) on the spec's listing
    order, with `from_b2(value)` (decoded `b(2)` → mode, `None` outside
    `0..=3`) and `index()` round-trip.
  - `FrameGeometry` — `(h_fragments, v_fragments)` → pixel dimensions
    (`luma_width`/`luma_height` = `fragments * 8`, per the §9 worked
    example: 320x240 → HFragments 40 / VFragments 30) plus the §2 4:2:0
    macroblock-grid round-up (`mb_cols`/`mb_rows` = `ceil(fragments / 2)`).
  - `OutputScaling` — the desired output `FrameGeometry` paired with a
    `ScalingMode`, with `is_identity(coded)` reporting whether a coded
    geometry needs any resampling.
  - `FRAGMENT_DIM` (8) — the §2/§16 transform-block edge.
  - **DOCS-GAP** documented on the module: the staged `vp6_format.pdf`
    names the four modes but does not specify the per-mode pixel-placement
    algorithm (how CENTER positions / SCALE_TO_FIT stretches / etc.); the
    resampling math is reported as a docs-gap candidate. The typed mode
    surface and the dimension derivations the math would consume land here.
- 11 new unit tests pinning: `FRAGMENT_DIM`; the §9 listing-order
  discriminants; `from_b2` round-trip + out-of-range rejection
  (`4..=255` → `None`); the 320x240 worked example; `luma_*` =
  `fragments * 8`; the 4:2:0 mb-grid round-up (even exact, odd up); and
  `OutputScaling::is_identity` (coded == output, mode-independent).
- Test count: 547 → 558. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean. No new spec material
  read beyond §9 (Table 2) / §2 of the staged `vp6_format.pdf`; no errata
  change required.

### Added (clean-room round 30, 2026-06-14)

- `tokens::DcContext` — the §13.2 Table 26 DC node context (the
  left/above predicted-DC zero-vs-non-zero situation): `BothZero` (0),
  `OneNonZero` (1), `BothNonZero` (2). Provides `index` / `from_index`
  / `from_neighbours(left_non_zero, above_non_zero)` (the Table 26
  partition) / `select_row(&DcNodeContexts[plane])` (the §13.2.1
  `ContPtr = DcNodeContexts[Plane][Context]` indexing) and a `Display`.
  This is the deferred "§13.2 DC-coefficient token-context wiring"
  flagged by round 28: the decoder reads from
  `DcNodeContexts[plane][context]`, not `DcProbs` directly, and this
  resolves *which* of the three precomputed
  `dc_probs_to_node_contexts` rows the §13.2.1 DC tree walk consults.
  An absent neighbour (frame left/top edge) counts as zero-DC per
  §13.2, so a corner block decodes with `BothZero`.
- `tokens::DcZeroContextTracker` — a per-plane raster bookkeeping
  companion. As blocks are decoded left→right / top→bottom it tracks
  the running left-neighbour non-zero flag plus one above-neighbour
  flag per column; `context_for()` returns the current block's
  `DcContext` and `record(non_zero)` advances the raster position
  (wrapping rows, resetting the left flag at each row start). Pure
  integer/boolean bookkeeping — reads no BoolCoder bits.
- `block_decode::decode_block_coefficients_ctx` — the
  context-resolving convenience over `decode_block_coefficients`:
  takes the per-plane `DcNodeContexts[plane][3][11]` bank plus a
  `DcContext` and selects the `[context]` row before invoking the
  base decoder, so a driver threading `DcZeroContextTracker` calls it
  directly rather than pre-resolving the row.
- 13 new unit tests (534 → 547): Table 26 index/round-trip/partition
  pins; `select_row` dimension check; tracker raster behaviour
  (corner both-zero, first-row left-only, row wrap resetting left,
  both-non-zero, single-column plane, zero-cols panic); and
  `decode_block_coefficients_ctx` byte-exact equality with manual row
  selection across plane × context × seed plus a context-distinction
  test. `cargo fmt --check` and `cargo clippy --all-targets --no-deps
  -- -D warnings` clean.

### Added (clean-room round 29, 2026-06-13)

- `frame_assembly` module — the §2/§13/§17 block-to-plane frame
  assembly stage, the natural successor to the per-block
  reconstruction pipeline (rounds 2–6/28: §13 entropy → §12 scan →
  §15 dequant → §16 `idct_block` → §17 reconstruct). Owns the
  per-plane raster image buffers and writes each finished
  reconstructed 8x8 block into its correct pixel position, so the
  per-block decoder output accumulates into a full decoded YUV 4:2:0
  image. Surfaces: `Plane` (a dense raster `u8` plane with
  `place_block` / `place_block_at_pixel` bounds-checked 8x8 writes,
  `with_block_grid` block-sized allocation, and `sample` /
  `samples` / `samples_mut` accessors); `Frame` (the three Y/U/V
  planes plus the §9 `HFragments` / `VFragments` geometry, with
  `place_macroblock_luma` mapping the four in-MB luma blocks to their
  §13-page-58 2x2 raster positions `0=TL, 1=TR, 2=BL, 3=BR`,
  `place_macroblock_chroma` placing one U + one V block per
  macroblock, and `place_macroblock` doing all six);
  `MB_LUMA_BLOCK_OFFSETS` (the verbatim 2x2 luma offset table);
  `mb_cols_for` / `mb_rows_for` (the §2 4:2:0 chroma-grid derivation,
  one chroma block per macroblock, `ceil(fragments / 2)` MBs);
  `BLOCK_DIM` / `MB_LUMA_DIM` / `MB_LUMA_BLOCKS` constants; and
  `AssemblyError::OutOfBounds`. Odd-`HFragments` / odd-`VFragments`
  partial edge macroblocks skip their off-grid overhang luma blocks
  without error. BoolCoder-independent — pure integer index
  arithmetic plus raster 8x8 writes over already-reconstructed §17
  pixels — so it advances the decoder without touching the contested
  §7.3 `Split` formula. The §11.5 UMV border stays in `umv` (applied
  to a reference-plane copy); these reconstruction planes are
  unbordered. Sourced exclusively from `vp6_format.pdf` §2 (page 9),
  §9 (page 24) and §13 (page 58). 13 new unit tests pinning the
  constant transcription, plane geometry from the §9 worked example
  (320x240 → HFragments 40 / VFragments 30), exact 8x8 write region +
  raster orientation, out-of-bounds rejection leaving the plane
  untouched, even/odd 4:2:0 frame geometry, the macroblock 2x2 luma
  ordering, one-chroma-block-per-MB placement, the full six-block
  macroblock, the odd-fragment overhang skip, and an end-to-end test
  driving the real `idct_block` + `intra_block_to_pixels` pipeline
  into a frame plane. Test count: 521 → 534.

### Added (clean-room round 28, 2026-06-12)

- `block_decode` module — the §13 per-block coefficient
  reconstruction driver, the explicit prior-round deferral now that
  every constituent primitive exists. `decode_block_coefficients`
  composes the §13.2.1 DC decode (seeding `CoeffData[0]` and the
  `Prec` context) with the §13.3.1 per-block AC do-while
  (per-iteration `[Prec][Band]` probability re-selection, implicit-1
  shortcut, per-leaf `Prec` updates, §13.3.3.1 zero-run transition,
  EOB exit) into `BlockCoeffs { coeffs[64], coeff_count }`;
  `dequantize_to_raster` fuses the §12 scan-position-to-raster
  permutation (default or custom) with the §15 DC/AC dequantizer;
  `decode_block_to_raster` is the one-shot composition feeding the
  §16 `idct_block` directly. Documents the §13.3.1 listing's
  `AcUpdateProbs[Prec][Plane][Band]` naming slip (the decoding bank
  is `AcProbs[plane][prec][band][node]` per the §13.3 prose) and the
  EOB `EncodedCoeffs++` / post-loop `--` exit choreography. Adds the
  `AcProbBank` / `ZeroRunProbBank` type aliases and the `BLOCK_SIZE`
  constant. 13 new unit tests (508 → 521) including two
  hand-computed §7.3 traces, an independent spec-listing replay
  pinning coefficients + count + final byte position across a
  stream × bank × plane grid, and fused-vs-two-step scan/dequant
  equality.

### Added (clean-room round 27, 2026-06-11)

- `fourmv::derive_fourmv_chroma_mv` / `fourmv::average_four_away_from_zero`
  — the spec §10 chroma motion vector for a `CODE_INTER_FOURMV`
  macroblock ("the motion vector for the two chroma blocks is computed
  by averaging the four Y vectors (rounding away from zero)", page 28).
  Per-component averaging with directed away-from-zero rounding
  (`sign(sum) * ceil(|sum| / 4)`), the round-23 explicit deferral.
  BoolCoder-independent; 10 new unit tests (498 → 508).

### Documented (clean-room round 27, 2026-06-11)

- README "What round 27 does NOT land" records the §9 frame-header
  BoolCoder-tail finding: errata #35's `>> 7` conclusion is internally
  inconsistent with its own rationale (at probability 128 the printed
  formula yields `Split = Range`, an empty `Bit = 1` interval, making
  every `b(n)` header field undecodable); the §9 tail stays blocked
  pending a docs correction.

### Added (clean-room round 26, 2026-06-10)

- `scan_update` module — the spec §12.2 per-frame custom scan
  order, the seventh BoolCoder-consuming layer (after rounds 16's
  §13.2.1 DC, 17's §13.3.1 AC, 19's §13.3.3.1 ZRL token, 20's
  §13.2/§13.3/§13.3.3 probability updates, 21's §11.1 MV component
  decoder and 22's §11.2 MV probability updates). Resolves the
  round-9 deferral of the Table 17 `ScanOrderUpdateFlag` /
  `CoeffBandUpdateFlag` / `NewCoeffBand` fields. Surfaces:
  - `CUSTOM_SCAN_BAND_RANGES[16]` — the Table 16 sixteen-band
    partition of the AC positions `1..=63` (band 0 = position 1,
    band 1 = positions 2–4, … band 15 = positions 58–63).
  - `COEFF_BAND_UPDATE_FLAG_PROBS[64]` — the verbatim §12.2
    `CoeffBandUpdateFlagProbs` bank; the spec's `NA` first entry
    ("a dummy entry for the DC coefficient … never updated in the
    bitstream") stored as the out-of-range poison value `0` and
    never read.
  - `BandAssignment` / `DEFAULT_BAND_ASSIGNMENT` — per-coefficient
    band state plus the §12.1-derived default (the default scan
    order is the identity in zig-zag space, so coefficient `c`
    defaults to the Table 16 band whose position range contains
    `c`).
  - `decode_scan_order_update(bc, &flag_probs, &mut assignment)` —
    the Table 17 record: `b(1)` `ScanOrderUpdateFlag`; on 0 the
    assignment resets to the default per §12.2 ("the scan order
    must be reset to the default"); on 1 dispatches into the
    63-coefficient walk. Intra-vs-inter seeding (§12.2: intra
    resets to the default before the deltas apply, inter carries
    the previous frame's assignment) documented and left to the
    per-frame driver.
  - `decode_coeff_band_updates(bc, &flag_probs, &mut assignment)`
    — the 63-set walk: `B(flag_probs[c])` `CoeffBandUpdateFlag`
    per AC coefficient in standard zig-zag order, plus a
    conditional `b(4)` `NewCoeffBand` (`0..=15`, exactly the
    Table 16 band space). Flag-prob bank parameterised like the
    round-20 `prob_update` drivers so synthetic-stream tests stay
    inside the round-15 BoolCoder's self-correcting envelope
    (errata #35 `Split > Range` commentary); the published bank's
    255-saturated tail is exercised under a real-bitstream
    integration once the per-frame driver round lands.
  - `build_custom_scan_order(&assignment)` — the §12.2 rebuild:
    bands in ascending order, coefficients within a band "sorted
    into ascending order based upon the original zig-zag scan
    order", DC pinned at position 0 ("In all scan orders the
    first DCT coefficient is always the DC coefficient").
  - `custom_scan_order_to_raster(&scan)` — composition with the
    §12.1 `DEFAULT_SCAN_ORDER` so §15/§16 consumers get raster
    positions directly.
- 14 new unit tests pinning: the Table 16 contiguous tiling of
  `1..=63`; the verbatim `CoeffBandUpdateFlagProbs` rows + DC
  dummy + legal-probability invariant; the default banding vs
  Table 16 with monotonicity; the default assignment rebuilding to
  the identity scan (and composing to the §12.1 table exactly);
  the §12.2 AC7/AC21-to-band-3 worked example (positions 11/12);
  the within-band ascending-zig-zag ordering; the
  permutation-for-any-assignment invariant (all-band-0,
  all-band-15, striped); the per-position raster composition; the
  flag-0 reset-to-default; the all-zero-stream no-op walk; the
  forced-flag (`flag_prob = 1`) walk rewriting all 63 bands
  through the `b(4)` path; the band `< 16` range invariant across
  seed streams; determinism across two independent runs; and the
  truncation surface on a 4-byte stream.
- Test count: 484 → 498. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean.

### Added (clean-room round 25, 2026-06-09)

- `mv_diff` module — the §11 intro's differential motion-vector
  reconstruction (BoolCoder-independent). Combines a
  round-21-decoded delta with a same-reference above/left
  neighbour MV — the §11 intro's "immediately to the left of or
  immediately above" constraint, strictly narrower than the §10
  12-neighbour walk — or with `(0, 0)` when neither qualifies
  (the "coded absolutely" branch). Surfaces:
  `DIFF_REFERENCE_OFFSETS` (the two-entry `(-1, 0)` / `(0, -1)`
  table pinned against the first two `modes::NEAR_MACROBLOCKS`
  entries); `select_diff_reference_mv` /
  `select_diff_reference_mv_from_grid` (two-neighbour walker
  applying the §10 predicates `mv != (0, 0)` + matching
  `dc_pred::ReferenceBucket`); `reconstruct_diff_mv` (the
  per-component sum `final = reference + delta`, no-wrap since
  §11.1 caps each input component at ±127); and
  `reconstruct_new_mv` / `reconstruct_new_mv_from_grid` (one-shot
  compositions).
- 24 new unit tests pinning: spec offset order,
  single-above/single-left qualification, above-wins precedence,
  different-reference and zero-MV fall-throughs,
  diagonal-neighbour exclusion, top-row / left-column /
  top-left-corner boundary cases, sum-no-wrap at the ±127 cap,
  walker short-circuit at first hit, and grid-wrapper equivalence
  with the closure-driven walker.
- Test count: 460 → 484. (Round 25 landed as commit `9eb340e`
  without its README/CHANGELOG entries; this entry is the round-26
  catch-up.)

### Added (clean-room round 24, 2026-06-06)

- `near_mv` module — the spec §10 Nearest / Near alternative-MV
  neighbour walker (the BoolCoder-independent piece of the §10
  mode-decode pipeline). Resolves the §10 alternative-MV pair plus
  the implied `modes::ModeAvailability` row index (Table 5) from
  the surrounding already-decoded macroblock grid by walking
  `modes::NEAR_MACROBLOCKS` in spec order and applying the two
  §10 predicates (`mv != (0, 0)` and matching
  `dc_pred::ReferenceBucket`) at each step. The first qualifying
  neighbour becomes `nearest_mv`, the second becomes `near_mv`;
  the walker short-circuits at the second hit. Surfaces:
  - `MotionVector` — typed `(x, y)` ¼-pixel motion vector
    (`i16` per §11.1 `±127` magnitude cap) with
    `MotionVector::ZERO` and `MotionVector::is_zero()` so the
    §10 "non (0, 0)" predicate lives in one place.
  - `NeighbourMv` — `{mv, reference}` neighbour metadata. The
    `reference` field reuses `dc_pred::ReferenceBucket` so the
    same-reference gate the §10 walker shares with §14 DC
    prediction stays in one enum.
  - `NearMvResolution` — walker output:
    `{ nearest_mv, near_mv, availability }` plus the
    `NearMvResolution::NONE` sentinel for the
    no-qualifying-neighbour case and `has_nearest()` /
    `has_near()` shortcuts.
  - `resolve_near_mvs(row, col, reference, neighbour_at)` —
    closure-driven walker. `neighbour_at: FnMut(i32, i32) ->
    Option<NeighbourMv>` so consumers keep their preferred
    MV-grid storage; out-of-frame `(row, col)` positions are
    reported with negative coordinates.
  - `resolve_near_mvs_from_grid(grid, grid_width, row, col,
    reference)` — dense-grid wrapper. Backs the walker with a
    flat `&[Option<NeighbourMv>]` slice indexed
    `row * grid_width + col`; out-of-bounds access returns
    `None` without panic.
- 18 new unit tests pinning: `NEAR_MACROBLOCKS` spec order
  (re-asserted locally so a future reorder trips this module's
  tests); `MotionVector::is_zero` across `(0, 0)` /
  `(±1, 0)` / `(0, ±1)` / `(±127, ±127)`; empty-grid →
  `NearMvResolution::NONE`; single-above-neighbour →
  `NearestOnly`; two-neighbours in spec order →
  `NearestAndNear`; different-reference skip; `(0, 0)`-MV
  skip; short-circuit-at-second-hit (visitor counting);
  top-left-corner negative-coordinate reporting; dense-grid
  wrapper resolution; bottom-right and top-left corner bounds
  safety; reference filtering through the wrapper; the
  `NearMvResolution::NONE` constant matches walker output; the
  walker's `availability` field matches
  `ModeAvailability::from_neighbours` for all three
  availability cases; `Intra` reference filtering; ±127
  maximum-magnitude MV qualification; all-12-qualify case
  picks the first two `NEAR_MACROBLOCKS` entries.
- Test count: 442 → 460. `cargo fmt --check` and `cargo
  clippy --all-targets --no-deps -- -D warnings` clean.

### Added (clean-room round 23, 2026-06-05)

- `fourmv` module — the spec §10 / Table 10 per-Y-block
  coding-mode signaling for `CodingMode::InterFourMv`
  macroblocks. When the MB-level §10 mode decision lands on
  `CODE_INTER_FOURMV`, each of the four 8x8 luma blocks
  transmits a two-bit codeword over the round-15 BoolCoder at
  probability 128 per bit; the codeword indexes a reduced
  four-mode set `{InterNoMv, InterPlusMv, InterNearestMv,
  InterNearMv}` (Table 10, page 37). Surfaces:
  - `FOURMV_BLOCK_MODES` — the four-entry Table 10 lookup, in
    canonical codeword-value order `[InterNoMv, InterPlusMv,
    InterNearestMv, InterNearMv]`.
  - `NUM_LUMA_BLOCKS_PER_MB` (`4`),
    `NUM_FOURMV_BLOCK_MODES` (`4`) — shape constants pinning
    the four-blocks-per-MB and the reduced-set width.
  - `decode_fourmv_block_mode(bc)` — single-block decoder. One
    `BoolCoder::decode_b(2)` read (two fixed-probability-128
    bits MSB-first) plus the lookup.
  - `decode_fourmv_block_modes(bc)` — four-block raster-order
    walker (block 0 = top-left, block 1 = top-right, block 2 =
    bottom-left, block 3 = bottom-right). Eight BoolCoder bits
    per MB total. Returns `[CodingMode; 4]`.
- 10 new unit tests pinning: the Table 10 lookup cover; the
  shape constants; `NUM_FOURMV_BLOCK_MODES` matches the lookup
  length; the all-zero stream decoding to `InterNoMv`
  (codeword `00`); the four-block walker on the all-zero
  stream producing four `InterNoMv` decodes; the four-block
  walker vs four serial per-block calls producing
  byte-identical BoolCoder state at the end (`pos`, `range`,
  `value`, `count` all equal); determinism across two
  independent BoolCoder runs; the reduced-set-membership
  invariant across a sweep of representative seed streams;
  the truncation surface on a 4-byte buffer; and per-block
  reduced-set-membership on three single-block seeds.
- Test count: 432 → 442. `cargo fmt` + `cargo clippy
  --all-targets -D warnings` clean. The §10 `VP6_DecodeMode`
  Figure-10 traversal — and its DOCS-GAP candidate around the
  `B(Stats[0])` / `B(Stats[2])` else-branches carried forward
  from round 21 — remains separate from this Table 10 read:
  every per-block bit here is fixed-probability-128, so the
  read is a closed lookup with no branching dependence on the
  pseudo-code's indentation.

### Added (clean-room round 22, 2026-06-04)

- `mv_prob_update` module — the spec §11.2 per-frame motion-vector
  probability-update bitstream, the sixth BoolCoder-consuming
  layer (after rounds 16's §13.2.1 DC, 17's §13.3.1 AC, 19's
  §13.3.3.1 ZRL token, 20's §13.2/§13.3/§13.3.3 per-frame
  probability updates, and 21's §11.1 MV component decoder).
  Walks the Table 13 update bitstream against four
  flag-probability lookup banks, mutating the persistent
  `[MvProbs; 2]` bank in place via the shared
  `prob_update::decode_new_node_prob` primitive (same
  `B(flag_prob)` + optional `b(7)` `NewProbability =
  max(1, value * 2)` recipe the §13.2 / §13.3 / §13.3.3 updates
  use). Surfaces:
  - `update_mv_probs(bc, &mut [MvProbs; 2])` — the Table 13
    walker, eight steps in the spec-mandated order
    (X top-level / Y top-level / X short-tree / Y short-tree /
    X long-bits / Y long-bits, with each "top-level" step being a
    `(short-discriminator, sign)` pair).
  - `UPDATE_IS_MV_SHORT_PROBABILITIES` (`{237, 231}`),
    `UPDATE_MV_SIGN_PROBABILITIES` (`{246, 243}`),
    `UPDATE_SHORT_VECTOR_NODE_PROBABILITIES` (`[[253, 253, 254,
    254, 254, 254, 254], [245, 253, 254, 254, 254, 254, 254]]`),
    `UPDATE_LONG_VECTOR_BIT_PROBABILITIES` (`[[254, 254, 254,
    254, 254, 250, 250, 252], [254, 254, 254, 254, 254, 251,
    251, 254]]`) — verbatim §11.2 `Update*Probabilities`
    initialisers from page 43-44.
  - `LONG_VECTOR_BIT_ORDER` — the eight-entry traversal-to-bit
    permutation `[0, 1, 2, 7, 6, 5, 4, 3]` Table 15 walks. Note
    this differs from §11.1's decode-time traversal
    `[0, 1, 2, 7, 6, 5, 4]` by the trailing `3`: at update time
    bit 3's probability is always present in the per-axis
    traversal order.
- 9 new unit tests pinning: the verbatim flag-probability tables
  (transcription guard); the `LONG_VECTOR_BIT_ORDER` permutation
  vs `0..=7`; the `LONG_VECTOR_BIT_ORDER` length matches
  `NUM_MV_SIZE_NODES`; the flag-bank dimensions vs `MvProbs`
  shape; the driver helpers' function signatures (compile-time
  shape check); the Table 13 X-before-Y step-order constants;
  the round-20 `decode_new_node_prob` primitive round-trip at
  moderate flag-prob `128`; the X-row vs Y-row root-node
  flag-prob asymmetry on the short-tree; the long-vector
  flag-prob tail-vs-head ordering.

Total test count: 423 → 432 (9 new, all green). Composes only
round-15 `BoolCoder` over the round-20
`prob_update::decode_new_node_prob` primitive — no new spec
material, no new errata. The published §11.2 flag-prob banks
cluster at the top of the `1..=255` range; full-driver coverage
under stream-driven Table 13 walks is deferred to an integration
test bound to a real .vp6 inter-frame fixture (the round-15
BoolCoder's behaviour at `Probability > 128` on synthetic
high-prob inputs sits outside the round-15 self-correcting
envelope).

### Deferred (round 22 follow-ups)

- Integration test: full Table 13 walk against a conformant
  inter-frame .vp6 fixture. Static surface + driver shape are
  validated here; sample-exact bit-walk validation needs a real
  bitstream.
- Intra-vs-inter gating of the §11.2 walk (intra frames reset
  the bank to §11.1 defaults instead of walking; lives in the
  upstream per-frame driver alongside the §9 BoolCoder-tail).
- BoolCoder round-15 primitive behavior at `Probability > 128`
  on synthetic streams — observed to drive the implementation's
  internal `range` accumulator above 255 when repeated 0-branch
  reads cumulate. Spec §7.3 keeps `Range` as a u8 (0-255). The
  errata #35 commentary documents the case as "statistically
  pathological"; integration coverage against a real bitstream
  is the path to confirming the implementation is conformant on
  spec-shaped input.
- §10 `VP6_DecodeMode` mode-decoder traversal (literal
  pseudo-code's indentation is ambiguous around the
  `B(Stats[0])` / `B(Stats[2])` else-branches — DOCS-GAP
  candidate carried forward from round 21).
- §10 `CODE_INTER_FOURMV` per-block 2-bit codeword.
- §11 differential MV reconstruction.

### Added (clean-room round 21, 2026-06-04)

- `mv_decode` module — the spec §11.1 per-component motion-vector
  arithmetic decoder, the fifth BoolCoder-consuming layer (after
  rounds 16's §13.2.1 DC, 17's §13.3.1 AC, 19's §13.3.3.1 ZRL
  token and 20's per-frame probability-update layers). Decodes one
  signed motion-vector component from a `BoolCoder` byte stream by
  composing four per-axis probability banks: `IsMvShortProbs[axis]`
  (short-vs-long discriminator), `ShortMvProbs[axis][0..=6]`
  (Figure 11 three-bit short-tree), `MvSizeProbs[axis][0..=7]`
  (seven-bit traversal in `[0, 1, 2, 7, 6, 5, 4]` order plus a
  conditional bit-3 read), and `MvSignProbs[axis]` (sign bit
  with negation). Returns a signed `i32` in the §11.1-staged
  decoder range `-255..=255`. Surfaces:
  - `decode_short_mv_magnitude(bc, &short)` — Figure 11 short-MV
    tree walk; magnitude `0..=7`.
  - `decode_long_mv_magnitude(bc, &size)` — seven-bit traversal
    plus the conditional `B(size[3])` bit-3 read; magnitude
    `8..=255`.
  - `decode_mv_component(bc, &probs)` — per-component wrapper
    composing magnitude + sign.
  - `decode_mv_pair(bc, &[probs_x, probs_y])` — full `(x, y)`
    pair, x first then y per the §11.1 outer loop.
  - `MvProbs` per-axis bundle with `defaults(axis)` constructor.
  - `IS_MV_SHORT_PROBS_DEFAULTS` (`{162, 164}`),
    `SHORT_MV_PROBS_DEFAULTS` (`{225, 146, 172, 147, 214, 39,
    156}` / `{204, 170, 119, 235, 140, 230, 228}`),
    `MV_SIZE_PROBS_DEFAULTS` (`{247, 210, 135, 68, 138, 220,
    239, 246}` / `{244, 184, 201, 44, 173, 221, 239, 253}`),
    `MV_SIGN_PROBS_DEFAULTS` (`{128, 128}`) — verbatim §11.1
    `Default_*` initialisers.
- 18 new unit tests pinning: the short-tree zero-path
  short-circuit; the short-tree all-1-path max-magnitude (3-bit
  walk → 7); the short-magnitude `0..=7` range invariant across
  both default-axis rows and four byte streams; the BoolCoder
  bit-advance bound through the short walk; the long-MV all-zero
  "implicit-bit-3" path yielding `0x08`; the long-MV all-ones
  path yielding `0xFF`; the long-MV `>= 8` lower bound; the
  long-MV high-bits branch reading bit 3; per-component positive
  + negative + zero-stream + ones-stream composite decodes
  against varied prob banks; the signed range invariant across
  the two §11.1 default-vector rows and three streams;
  determinism (same bytes + probs → same output) across two
  independent runs; pair-decoder x-axis independence (x with
  default probs, y with varied probs → x agrees across runs);
  truncation on a 4-byte buffer that exhausts during the
  per-component traversal; default-probs-against-zero-stream
  produces a well-defined signed result.

Total test count: 405 → 423 (18 new, all green). No spec gap
encountered; no errata change required. Composes only round-15
`BoolCoder::decode_bool` over the verbatim §11.1 default
probability tables — no new spec material, no third-party VP6
source consulted.

### Deferred (round 21 follow-ups)

- §11.2 per-frame MV-probability update bitstream — same shape
  as the §13.2 / §13.3 / §13.3.3 updates already landed in
  `prob_update`; lands as a thin wrapper over the existing
  `decode_new_node_prob` primitive once §11.2's flag-prob tables
  are transcribed.
- §10 `VP6_DecodeMode` mode-decoder traversal — literal
  pseudo-code's indentation is ambiguous around the
  `B(Stats[0])` / `B(Stats[2])` else-branches; reported as a
  DOCS-GAP candidate (separate from the round-13
  `zrl::ZrlNode` 9th-leaf candidate and the round-15-resolved
  §7.3 `Split` formula entry #35).
- §10 `CODE_INTER_FOURMV` per-block 2-bit codeword (Table 10);
  trivial on the existing BoolCoder substrate but distinct
  logical unit.
- §11 differential MV reconstruction — combining a decoded
  delta with the same-reference neighbour MV (or absolute
  fallback). Lives in the §10 caller alongside the
  `modes::NEAR_MACROBLOCKS` traversal.

### Added (clean-room round 20, 2026-06-03)

- `prob_update` module — the per-frame BoolCoder-coded
  probability-update bitstream the §13.2 DC, §13.3 AC and §13.3.3 ZRL
  token decoders all consume to mutate their persistent probability
  banks at every frame. The fourth BoolCoder-consuming layer (after
  rounds 16's §13.2.1 DC, 17's §13.3.1 AC and 19's §13.3.3.1 ZRL
  token decoders). Three update bitstreams share the same per-node
  shape (Tables 24, 35 and 41 are all the same two-field record:
  `B(flag_prob)` `NewNodeProbFlag` followed by a conditional `b(7)`
  `NewNodeProbValue`) and the same disambiguated reading of
  "½ of the new probability value" (§13.2 Table 24 commentary):
  `new_prob = max(1, NewNodeProbValue * 2)`. The 7-bit raw read puts
  `NewNodeProbValue` in `0..=127`; doubling gives `0..=254`; the spec
  clip converts the `0` case to `1`. Surfaces:
  - `prob_update::decode_new_node_prob(bc, flag_prob)` — the per-node
    step, returning `Ok(None)` on skip or `Ok(Some(prob))` on update.
  - `prob_update::update_dc_probs(bc, &mut dc_probs, &flag_probs)` —
    the §13.2 driver, walking Tables 22 / 23 / 24 as
    `for plane in 0..2 { for node in 0..11 { ... } }`.
  - `prob_update::update_ac_probs(bc, &mut ac_probs, &flag_probs)` —
    the §13.3 driver, walking Tables 31 / 32 / 33 / 34 / 35 as
    `for prec in 0..3 { for plane in 0..2 { for band in 0..6 { for
    node in 0..11 { ... } } } }`. Notes the spec's two different
    dimension orderings: `AcProbs[plane][prec][band][node]` (the
    per-token bank §13.3.1 reads) vs `AcUpdateProbs[prec][plane]
    [band][node]` (the flag-prob bank this driver walks), and writes
    the spec walk order into the `AcProbs`-shaped target.
  - `prob_update::update_zero_run_probs(bc, &mut zero_run_probs,
    &flag_probs)` — the §13.3.3 driver, walking Tables 39 / 40 / 41
    as `for band in 0..2 { for node in 0..14 { ... } }`.
- 16 new unit tests pinning: the `flag_prob = 255` shortcut to
  `None` (errata-#35 `Split == Range` 0-branch); the flag-set path
  reading seven `b(7)` raw-bit tail; the `0 → 1` clip on the spec's
  "½ of the new probability value" formula; the `None` return as a
  no-op on the persistent bank; the `(1..=255)` range invariant
  swept across `flag_prob ∈ {1, 64, 128, 192, 254}` and three
  representative byte streams; the three drivers' deterministic
  reproduction across two independent runs; the three drivers'
  range invariant across the full `[2 * 11]` (DC), `[2][3][6][11]`
  (AC) and `[2][14]` (ZRL) bank entries; the truncation surface on
  a 4-byte buffer that exhausts during the DC walk; the AC walk's
  byte-budget consumption against the worst-case `8 *
  396 = 3168` BoolCoder-bit estimate; and the direct formula
  invariants sweep over all 128 `b(7)` values (clip-to-1, parity
  preservation under non-clip).

Total test count: 389 → 405 (16 new, all green). No spec gap
encountered; no errata change required. Composes only round-15
`BoolCoder::decode_bool` / `decode_b` and the staged §13 lookup
tables (`VP6_DC_UPDATE_PROBS`, `AC_UPDATE_PROBS`,
`ZRL_UPDATE_PROBS`) — no new spec material, no third-party VP6
source consulted.

### Added (clean-room round 19, 2026-06-03)

- `dct_decode::decode_ac_zero_run(bc, band, &probs)` — the spec
  §13.3.3.1 BoolCoder zero-run-length traversal, the third
  BoolCoder-consuming layer (after §13.2.1 DC and §13.3.1 AC). The
  immediate consumer of the [`AcOutcome::ZeroRun`] hand-off that
  round-17's §13.3.1 AC decoder surfaces. Walks Figure 16's binary
  tree reading one `B(prob)` BoolCoder bit at each of the eight
  internal nodes (`>4`, `>2`, `>1`, `>3`, `>8`, `>6`, `>5`, `>7`)
  in the Table 38 / round-13 `ZrlNode` ordering. On the `>8` escape
  branch reads six additional `B(prob)` extrabits as the LSB-first
  encoding of `(RunLength - 9)` and reconstructs
  `RunLength = value + 9`. Returns the run length as a `u32` in the
  full spec output range `1..=72` (literal `1..=8` from the eight
  binary-tree leaves plus `9..=72` from the `9 + (0..=63)` escape).
  Composes round-15 `BoolCoder::decode_bool` over the round-13
  [`zrl`] static surface; no new spec material, no new errata, no
  third-party VP6 source consulted.
- 11 new unit tests pinning: the low-probability all-zero-stream
  leftmost-leaf result (`run = 1`); the band-argument's
  row-selector semantics (output independent of `band` when `probs`
  is the same); the BoolCoder state advance under renormalization;
  the §7.3 errata-#35 "`Split > Range` collapses to 0-branch"
  shortcut forcing the leftmost-leaf result at `probs = [255; 14]`;
  the `>8` escape with zero extrabits yielding the minimum-escape
  `run = 9`; the root 0-branch picking the lower (`1..=4`) subtree;
  the truncation surface on a 4-byte stream that exhausts during
  the first renormalization; determinism across a four-seed sweep;
  the `1..=72` output-range invariant across the canonical
  keyframe probability rows + five stream seeds + both bands; a
  decode against `ZERO_RUN_PROB_DEFAULTS` at the all-zero stream
  lands a well-defined run length per band; and the
  `AcOutcome::ZeroRun` hand-off contract pinned by composing the
  §13.3.1 outcome with `decode_ac_zero_run` at the keyframe
  defaults.

Total test count: 378 → 389 (all green).

### Added (clean-room round 18, 2026-06-03)

- `inter::fetch_prediction_block_clamped(image, width, height, top, left,
  dx, dy, &mut pred)` — the §11.5-derived **edge-clamped** integer-MC
  fetch. The spec defines §11.5 "Unrestricted Motion Vectors" via a
  48-sample buffer extension built by "duplicating the edge values 48
  times", which is mathematically equivalent to clamping the read
  position into the original image's `[0, width)` x `[0, height)`
  rectangle (the equivalence already recorded in the `umv` module
  docs). This entry point implements that equivalence directly: it
  reads from the unbordered reference image, clamping each per-sample
  `(row, col)` source before the dereference. For any MV inside the
  48-sample §11.5 border the output is bit-identical to a
  `fetch_prediction_block` call against the §11.5-bordered version of
  the same image (verified by the
  `clamped_matches_bordered_fetch_for_in_range_mv` and
  `clamped_matches_bordered_fetch_for_edge_overhang` equivalence
  tests, which sweep both fully-inside-image cases and the four
  per-edge / four corner-quadrant overhang cases). For MVs whose
  magnitude **exceeds** the 48-sample border the spec mandates (where
  the bordered fetch would index out of bounds), the clamped fetch
  remains well-defined and continues to serve up the corresponding
  corner / edge-row / edge-column pixel of the original image — the
  `clamped_well_defined_beyond_umv_border` test exercises a
  `-200` sample MV against a 16x16 image and verifies every output
  sample reads the `(0, 0)` corner.
- 15 new unit tests covering all four edge-overhang directions
  (left / right / top / bottom), the four corner quadrants, the
  per-sample independence property (one bright sample in an
  otherwise-zero image lands at exactly the right output position),
  the in-range zero-MV and positive-MV reductions to a plain
  co-located / offset copy, and the three degenerate panic cases
  (zero width, zero height, truncated image buffer).

Total test count: 363 → 378 (all green).

### Added (clean-room round 17, 2026-06-01)

- `dct_decode::decode_ac_token(bc, prec, encoded_coeffs, &node_probs)`
  / `dct_decode::decode_ac_coefficient(bc, prec, encoded_coeffs,
  &node_probs)` — the spec §13.3.1 per-coefficient arithmetic AC
  decoder. The walk differs from the §13.2.1 DC variant on two
  structural counts:
  - **`EOB_CONTEXT_NODE` branch.** A 0-bit at the
    `ZERO_CONTEXT_NODE` root no longer short-circuits to
    `ZERO_TOKEN`; it enters a `B(EOB_CONTEXT_NODE)` decision whose
    0-branch is `EOB_TOKEN` (end-of-block) and whose 1-branch is
    `ZERO_TOKEN` (hand off to the §13.3.3 zero-run decoder).
  - **Implicitly-1 first-decision shortcut.** When the previous AC
    token was `ZERO_TOKEN` (`prec == WasZero`) and we are past the
    first AC coefficient (`encoded_coeffs > 1`), the §13.3.1
    pseudo-code mandates the next token can be neither `ZERO_TOKEN`
    nor `EOB_TOKEN`, so the root decision is implicitly `1`. The
    gate is correctly closed at `encoded_coeffs == 1` (the
    very-first AC position, whose `Prec` was seeded from the
    §13.2-decoded DC of the same block).
- `dct_decode::AcOutcome` — the three-way per-coefficient result:
  `EndOfBlock` (exit the per-block loop), `ZeroRun` (current coeff
  is 0, caller invokes §13.3.3 zero-run decoder), or
  `Value { coeff, next_prec }` (signed AC coefficient + §13.3.1
  `Prec` update for the next position: `WasOne` if `|coeff| == 1`,
  `WasGreaterThanOne` otherwise).
- Static surface in `tokens`:
  - `AcBand` — Table 30 six AC bands (coefficient 1; 2–4; 5–10;
    11–21; 22–36; 37–63) with the `for_coefficient_position(usize)
    -> Option<AcBand>` lookup that returns the §13.3.1
    `AcProbBand[encodedCoeffs]` band index for any AC scan position
    `1..=63`. Companion `index` / `from_index` / `ALL`.
  - `AcPlane` — Table 28 (Y / UV) with the standard `index` /
    `from_index` / `ALL`.
  - `AcPrecContext` — Table 29 (`WasZero` / `WasOne` /
    `WasGreaterThanOne`) plus `seed_from_dc(dc: i32) -> Self` that
    implements the §13.3.1 first-AC seeding rule.
- 18 new unit tests in `dct_decode`: the implicit-1 shortcut's
  positive/negative cases on each conjunct
  (`encoded_coeffs > 1`, `prec == WasZero`); the EOB/ZERO inversion
  branches at the EOB-node; the `EndOfBlock` / `ZeroRun` / `Value`
  outcome variants; the `next_prec` update invariant via a property
  sweep that hits both magnitude-1 and magnitude->1 paths;
  determinism; the `decode_ac_coefficient = decode_ac_token + value`
  composition; `seed_from_dc` exact spec-matching against the
  `0 / 1 / -1 / 2 / 2114 / -2114` corners; the truncation surface
  on a 4-byte stream; a structural leaf-set sweep proving the AC
  walk's leaves are all valid `DctToken` values. Plus 9 new unit
  tests in `tokens` covering the Table 28/29/30 enum surfaces
  (round-trip, Table-30 partition cover of `1..=63` with
  per-band-count verification, display names).
- Resolves the §13.2.1 round-16 close-out item "§13.3.1 AC
  branching and §13.3.3 zero-run integration deferred to a later
  round" for the §13.3.1 half. The §13.3.3 zero-run integration
  remains follow-on (the `AcOutcome::ZeroRun` variant surfaces the
  hand-off point; the §13.3.3.1 BoolCoder traversal is its own
  logical unit on top of the round-15 BoolCoder primitive).

### Added (clean-room round 16, 2026-06-01)

- `dct_decode` module — the first BoolCoder-consuming layer: the
  spec §13.2.1 arithmetic DC coefficient decoder. Surfaces:
  - `decode_dc_token(bc, &node_probs)` — the Figure 15 binary-tree
    walk down to a `DctToken` leaf (DC variant, never returns EOB).
  - `decode_token_value(bc, token)` — the magnitude-loop + sign
    decode shared between §13.2.1 and §13.3.1 (reads `#ExtraBits − 1`
    magnitude bits MSB-first per errata #67, then a separate
    fixed-prob-128 `b(1)` sign, then reconstructs the signed value via
    `(value ^ -SignBit) + SignBit`).
  - `decode_dc(bc, &node_probs)` — full §13.2.1 wrapper returning
    the signed DC coefficient.
- `DctToken::magnitude_probs()` — the errata-#67 corrected
  magnitude-only probability slice (length `#ExtraBits − 1`).
  Legacy `extra_bit_probs()` accessor preserved verbatim with its
  docstring updated to point at the new accessor for the
  magnitude-loop traversal.
- 19 new unit tests pinning the §13.2.1 walk's behaviour: the
  zero-token short-circuit; per-token magnitude-bit count vs value
  range; MSB-first magnitude reading via all-zero-stream traces
  (every category returns +min_value); the constant-magnitude
  tokens' +1..+4 short-cuts; determinism + composition guarantees
  (`decode_dc = decode_dc_token + decode_token_value`); the
  truncation surface on a 4-byte stream + low-prob walks; the
  sign-reconstruction identity; and the structural "DC walk never
  returns EOB" property over a sweep of `node_probs` corners.

## [0.0.7](https://github.com/OxideAV/oxideav-vp6/releases/tag/v0.0.7) - 2026-05-30

### Other

- round 15: §7.3 BoolCoder primitive (decode_bool + b(n) + init) per errata #35
- round 14: §3 R(n) raw-bit byte-stream reader + Huffman driver
- round 13: §13.3.3 AC zero-run static surface + §13.3.3.2 Huffman conversion
- round 12: §7.2 Huffman tree construction + traversal
- §13 DCT-token static surface (BoolCoder-independent half)
- round 10: §10 mode-decoding static surface (Table 4/5, NEAR_MACROBLOCKS, baseline+VQ probs, ModeDecisionTree builder)
- round 9: §12.1 default zig-zag scan order + §14 DC prediction
- round 8: §11.5 Unrestricted Motion Vector borders
- round 7: §11.3 prediction loop filter
- round 6 — §17.2/§17.3/§17.4 inter-block reconstruction + §11.4 MV decomposition
- §11.4 fractional-pixel interpolation filters (round 5)
- §17.1 intra block reconstruction (round 4)
- inverse DCT transform (spec §16)
- round 2: inverse-quantization layer (spec §15)
- round 1: frame-header raw-bit prefix parser
- orphan rebuild: clean-room scaffold post 2026-05-18 audit

### Added (clean-room round 15, 2026-05-30)

- `bool_coder` module — the spec §7.3 binary arithmetic decoder
  (`BoolCoder`). Surfaces:
  - `BoolCoder::new(bytes)` — `VP6_StartDecode`: 4-byte big-endian
    prefill of `Value`, `Range = 255`, `Count = 8`, `Pos = 4`.
    Returns `Error::Truncated` if `bytes.len() < 4`.
  - `BoolCoder::decode_bool(probability) -> Result<u8, Error>` —
    the §7.3 `VP6_DecodeBool` per-bit step
    (`Split = 1 + ( ((Range-1) * Probability) >> 7 )`, branch on
    `Value < (Split << 24)`, update `Range`/`Value`, then run the
    renormalization loop pulling fresh bytes via `Pos`). The §3
    `B(x)` primitive.
  - `BoolCoder::decode_b1() -> Result<u8, Error>` — single
    fixed-probability-128 bit (§3 `b(1)`).
  - `BoolCoder::decode_b(n) -> Result<u32, Error>` — `n`-bit
    fixed-probability-128 raw read (§3 `b(n)`), MSB-first so the
    bit order matches §3 `R(n)`. Saturates at `n = 32`.
  - `range`, `value`, `count`, `pos` diagnostic accessors.
- Resolves the prior DOCS-GAP about the §7.3 `Split` formula. The
  newly-staged clean-room errata
  `docs/video/vp6/vp6-errata-and-clarifications.md` entry **#35**
  confirms `>> 7` (divide by 128) is correct and intentional
  precisely because it makes `Probability = 128` the half-interval
  point — exactly what binary-arithmetic-coder semantics for §3
  `b(x)` require. The crate-root DOCS-GAP block is removed and
  replaced with a round-15 "§7.3 BoolCoder primitive" summary.
- 13 new tests exercising: §7.3 `VP6_StartDecode` initial state
  (big-endian prefill, `Range = 255`, `Count = 8`, `Pos = 4`);
  `Error::Truncated` rejection of streams shorter than 4 bytes; the
  all-zero-stream zero-bit invariant across multiple probabilities;
  the errata #35 canonical half-interval Split value
  (`Probability = 128, Range = 255 → Split = 255`); the small-Split
  renormalization shift (`Probability = 1, Range = 255 → Split = 2`);
  a fully-traced renormalization byte-pull sequence (Range / Value /
  Count / Pos at each step); `decode_b` MSB-first accumulation
  matching repeated `decode_b1` calls; the `decode_b(0)` no-op;
  `decode_b` saturation at 32 bits; `Truncated` propagation from
  both single-bit and multi-bit reads; and bit-stream-input
  determinism.

### Added (clean-room round 14, 2026-05-29)

- `raw_bits` module — the spec §3 `R(x)` raw-bit byte-stream reader,
  the byte-stream substrate underneath the §7.2 Huffman coder (and,
  once the §7.3 DOCS-GAP is closed, underneath the BoolCoder's `R(8)`
  refill reads as well). Surfaces:
  - `RawBitReader<'a>` — a thin wrapper around
    `oxideav_core::bits::BitReader` exposing the standard `R(n)`
    MSB-first byte-stream convention used by §9 Tables 1/2 (the same
    convention `frame_header::Vp6FrameHeader::parse` already consumes).
    Constructors `RawBitReader::new(bytes)` and
    `RawBitReader::with_byte_offset(bytes, byte_offset)` (the latter
    for partition 2 reads where the caller has `Buff2Offset` from
    §9). Bookkeeping accessors: `bits_remaining`, `bit_position`,
    `byte_position`, `is_byte_aligned`, `is_empty`. Position control:
    `align_to_byte` for the *"the next field starts at the next byte
    boundary"* phrasing some §9 entries use.
  - `RawBitReader::read_bit() -> Result<u8, RawBitError>` — read one
    raw bit (`R(1)`), MSB-first, returning the §7.2 walker's expected
    `0`/`1`.
  - `RawBitReader::read(n) -> Result<u32, RawBitError>` — read `n`
    raw bits (`R(n)`, `0 <= n <= 32`) as an unsigned MSB-first integer.
    Matches the §9 Tables 1/2 convention.
  - `RawBitReader::read_lsb_first(n)` — the explicit *least-significant
    bit first* variant for the one place in the spec that overrides
    the MSB-first byte-stream convention by name: §13.3.3.1 (page 78),
    *"the run length minus nine is encoded using six-bits, least
    significant bit first."* The `R(6)` escape suffix the §13.3.3 AC
    zero-run path reads (in both the BoolCoder and Huffman entropy
    schemes — the spec's demonstration pseudo-code is
    `if (ZrlToken < 8) … else 8 + R(6)`) consumes this.
  - `RawBitReader::read_huffman_symbol(&mut self, tree)` — convenience
    that wires the byte-stream `R(1)` source straight into the §7.2
    `huffman::decode_symbol` walk. The Huffman path of §13 token
    decoding (when the frame header's `UseHuffman == 1`) and the
    §13.3.3.2 AC zero-run Huffman walker both consume this directly;
    both landed parameterised over an `FnMut() -> u8` oracle in
    rounds 12 and 13 precisely so the byte-stream reader could land
    independently here.
  - `RawBitError::{OutOfBits, TooManyBits}` — narrow error type. VP6
    partitions are bounded byte buffers per §6; reading past the end
    is a malformed-input condition the decoder surfaces cleanly.
- The reader implements `Clone + Copy` (it owns nothing but a borrowed
  slice and a position) so a parser can checkpoint-and-restore by
  assignment — useful for partition probes that look ahead without
  committing to a read. The `Debug` impl is hand-written because the
  underlying `BitReader` doesn't derive it.
- Unit tests (22 new): MSB-first packing of `R(n)`, the §9 Table 1
  byte-0 layout (`FrameType R(1) | DctQMask R(6) | MultiStream R(1)`)
  matched against `frame_header::Vp6FrameHeader::parse`'s convention,
  `read_lsb_first` against an exhaustive `0..=63` round-trip, the
  §13.3.3.1 escape worked example (`run_length = 17` → 6-bit payload
  decimal `8`, packed `0b0001_0000`), end-to-end §7.2 Huffman decoding
  through the byte stream (cross-checked against the closure form of
  `decode_symbol`), out-of-bits truncation, the `n > 32` rejection,
  the `n == 0` no-op, byte alignment after partial-byte reads, and the
  `Copy` checkpoint pattern.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2 this module reads **no
  BoolCoder bits** — every operation is plain byte-stream bit
  arithmetic — so it advances the decoder past round 13 without
  touching the contested §7.3 `Split` formula. With this round the
  Huffman path of the §13 DCT-token decoder and the §13.3.3.2 zero-run
  Huffman decoder both have a complete end-to-end data path (modulo
  the §13.3.3.2 9th-leaf semantics docs-gap noted in round 13's `zrl`
  report).

### Added (clean-room round 13, 2026-05-29)

- `zrl` module — the spec §13.3.3 AC zero-run-length static surface
  (the BoolCoder-independent half of zero-run decoding). When the
  §13 token decoder produces a `ZERO_TOKEN` in the AC position, a
  zero-run length follows that says how many consecutive AC
  coefficients are zero. The run length can be coded with either of
  the two §7 entropy schemes — the BoolCoder path of §13.3.3.1
  reads `B(prob)` BoolCoder bits across Figure 16 plus six
  `(RunLength - 9)` extrabits and stays deferred behind the §7.3
  DOCS-GAP, while the Huffman path of §13.3.3.2 converts the same
  Figure 16 node probabilities into a 9-entry Huffman probability
  set and traverses the resulting Huffman tree with raw `R(1)` bits
  — so it is **independent of the §7.3 `Split` formula DOCS-GAP**.
  Surfaces:
  - `ZrlBand` — Table 37 zero-coefficient-starting-band indices
    (Band0 = coefficient positions 1–5, Band1 = 6–63) with the
    spec's canonical 0/1 indexing, an `ALL` array, round-trip
    `index` / `from_index` accessors, and a
    `for_coefficient_position` helper that partitions the AC
    coefficient range into the two bands.
  - `ZrlNode` — the fourteen Table 38 node indices. The first
    eight (`0..=7`) name the eight internal nodes of the Figure 16
    binary tree (`>4`, `>2`, `>1`, `>3`, `>8`, `>6`, `>5`, `>7`) in
    the spec's canonical order; the remaining six (`8..=13`) name
    the bit positions of the `(RunLength - 9)` six-bit suffix the
    BoolCoder path reads when the run is greater than 8, with each
    extrabit's shift exposed via `extrabit_shift`. `is_tree_node`
    partitions the fourteen names into the two halves.
  - `ZERO_RUN_PROB_DEFAULTS[2][14]` — the verbatim
    `ZeroRunProbDefaults` keyframe initialiser ("At each key frame
    every probability value in this array of AC Probabilities is
    set to the multidimensional array ZeroRunProbDefaults").
  - `ZRL_UPDATE_PROBS[2][14]` — the verbatim `ZrlUpdateProbs`
    per-node `NewNodeProbFlag` update-flag probability bank (the
    Table 41 BoolCoder reads themselves stay deferred).
  - `zrl_bool_tree_to_huff_probs` — the verbatim §13.3.3.2
    `ZRLBoolTreeToHuffProbs` transform that converts an 8-entry
    node-probability vector into the 9-entry Huffman probability
    set the Huffman tree builder consumes (one chain factor per
    Figure 16 internal-node branch; `>> 8` truncation per the
    spec's listing; intermediate values held in `u32` so the final
    narrowing to `u8` is lossless).
  - `build_zrl_huffman_tree` — composes the §13.3.3.2 pseudo-code
    pair `ZRLBoolTreeToHuffCodes` + `VP6_BuildHuffTree` for one
    band. Runs `zrl_bool_tree_to_huff_probs` and then invokes the
    §7.2 `create_huffman_tree` primitive (already landed in round
    12) to build a `2N - 1 = 17`-node `HuffNode` tree the §7.2
    walker can traverse. Zero converted probabilities are floored
    to `1` so the builder's "probability 0 is forbidden" check
    doesn't reject the structural tree shape under chain-factor
    underflow.
- Tests (27): `ZrlBand` / `ZrlNode` enum-index round-trips +
  Table 37 / Table 38 ordering invariants; the
  `for_coefficient_position` partition of AC coefficients 1–63;
  `extrabit_shift` against Table 38's `>> 0..=5` shifts;
  `ZERO_RUN_PROB_DEFAULTS` and `ZRL_UPDATE_PROBS` verbatim values
  per row; `zrl_bool_tree_to_huff_probs` output size; the
  within-internal-node pair-equality invariant under uniform
  `[128; 8]` inputs; the asymmetric-tree-depth geometry invariant
  (the depth-1 `>8` leaf carries the most mass and the four
  depth-2 left-half leaves outweigh the four depth-3 right-lower
  leaves); root-extreme zeroing of the opposite subtree;
  conversion well-formedness on both keyframe-default rows;
  `build_zrl_huffman_tree` topology invariants (`17`-node total,
  9 leaves, 8 internal nodes, root at index `2N - 2 = 16`); every
  canonical symbol 0..=8 reachable from the root via the §7.2
  walker; round-trip against both keyframe-default rows; canonical
  Huffman invariant on skewed inputs (the dominant symbol's
  codeword is no longer than the rare symbol's).
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2 this stage reads no
  BoolCoder bits — every operation is pure integer arithmetic over
  the supplied probability vector — so it advances the decoder past
  round 12 without touching the contested §7.3 `Split` formula. The
  §13.3.3.1 BoolCoder Figure 16 traversal + six-bit extrabit reads
  remain deferred; the §13.3.3.2 9th-leaf literal-vs-escape
  semantics (whether `ZrlToken == 8` means a literal run of 8 or
  the `>8` escape) are reported as a docs-gap candidate (see the
  module's "What this module does NOT land" section).

### Added (clean-room round 12, 2026-05-26)

- `huffman` module — the spec §7.2 Huffman tree construction and
  traversal primitives (`HUFF_NODE`, `VP6_CreateHuffmanTree`,
  `VP6_HuffmanDecodeSymbol`). VP6 supports two entropy schemes (§7):
  the BoolCoder (§7.3) used in partition 1 for mode/MV decisions
  and the Huffman coder (§7.2) used as an alternate DCT-token
  scheme when the frame header's `UseHuffman` flag is set. The
  Huffman coder reads one whole raw bit per tree branch (`R(1)`;
  §3 nomenclature) rather than a sub-bit `B(prob)` BoolCoder bit,
  so this stage is **independent of the §7.3 `Split` formula
  DOCS-GAP**. Surfaces:
  - `HuffNode` — the spec's `HUFF_NODE { Symbol, Prob, Left, Right }`
    struct (page 13) with `-1` sentinels for the internal-node and
    no-child markers; the `INTERNAL_SYMBOL` / `NO_CHILD` constants
    name the spec's convention.
  - `create_huffman_tree` — the verbatim §7.2.1 builder. `N-1`
    bottom-up merge rounds over a stable-sorted leaf list (the
    `slice::sort_by_key` invariant matches §7.2.1's repeated
    "*maintaining relative order of nodes having equal
    probability*" requirement). Returns a `Vec<HuffNode>` of length
    exactly `2N-1` with the root at index `2N-2`. Rejects zero
    probabilities (§7 forbids `0`) and `N < 2` inputs via the new
    `HuffmanError` enum.
  - `decode_symbol` — the verbatim §7.2 `VP6_HuffmanDecodeSymbol`
    walk. Parameterised over an external `FnMut() -> u8` raw-bit
    oracle so the byte-stream `R(1)` reader can land independently;
    per §7 *"0 indicates left, 1 indicates right"*.
  - `tree_depth` / `codeword_for` — pure-walker helpers that recover
    a symbol's codeword length and bit pattern, used by the
    round-trip and shape-invariant tests.
- Tests (15): empty / mismatched-length / zero-prob input
  validation; the two-symbol degenerate tree's exact geometry
  (`SortList[0..2]` = leaves, `SortList[2]` = root with both
  children); the `2N-1` length / `N-1` internal-node-count /
  leaf-symbol-set invariants across `N = 2..=12`; the spec's
  stable-sort invariant under equal probabilities; round-trip
  decode of every leaf in the §13 keyframe-baseline Huffman tree
  (driven by `dct_token_bool_tree_to_huff_probs` on all-128 node
  probabilities); plus shape checks (balanced inputs give
  uniform-depth trees; skewed inputs give dominant symbols shorter
  codewords than rare symbols).
- Like §15/§16/§17/§11/§12.1/§14/§10/§13 this stage reads no
  BoolCoder bits — every operation is pure integer arithmetic over
  the supplied probability vector — so it advances the decoder
  past round 11 without touching the contested §7.3 `Split`
  formula. The §13.3.3.2 AC zero-run probability conversion and
  the actual `R(1)` byte-stream reader stay deferred for later
  rounds.

### Added (clean-room round 11, 2026-05-25)

- `tokens` module — the spec §13 DCT-coefficient token static
  surface (the BoolCoder-independent half of coefficient decoding).
  Reads no BoolCoder bits; the §13 `VP6_DecodeToken` traversal and
  per-frame probability-update bitstream that consume the surface
  stay deferred behind the §7.3 `Split` DOCS-GAP. Surfaces:
  - `DctToken` — the twelve Table 18 tokens (`ZERO_TOKEN`,
    `ONE_TOKEN`..`FOUR_TOKEN`, `DCT_VAL_CATEGORY1`..`6`,
    `DCT_EOB_TOKEN`) as a `#[repr(u8)]` enum on the canonical
    0..=11 index, with per-token `min_value` / `max_value` /
    `extra_bits` (Table 18 "# of extrabits, incl. sign") and the
    verbatim `extra_bit_probs` "Arithmetic Encoding the Extra
    Bits" column.
  - `TreeNode` — the eleven Table 20 coding-tree node names on the
    canonical 0..=10 index (the index into a node probability
    vector consulted by the Figure 15 traversal).
  - `baseline_dc_probs` / `baseline_ac_probs` — the all-128
    keyframe initialisers for `DcProbs[2][11]` and
    `AcProbs[2][3][6][11]` (§13.2 / §13.3).
  - `VP6_DC_UPDATE_PROBS[2][11]` — the verbatim `VP6_DcUpdateProbs`
    per-node update-flag probability bank (§13.2).
  - `AC_UPDATE_PROBS[3][2][6][11]` — the verbatim `AcUpdateProbs`
    per-node update-flag probability bank (§13.3).
  - `DC_NODE_EQS[5][3][2]` — the verbatim `DcNodeEqs` slope/constant
    linear-equation table (Table 27), including the EOB dummy row
    that forces the EOB node to 1.
  - `dc_probs_to_node_contexts` — the pure-integer §13.2 conversion
    expanding a `DcProbs[2][11]` bank into the
    `DcNodeContexts[2][3][11]` per-context trees the §13.2.1
    arithmetic DC decoder consults (linear equation on nodes
    0..5 clipped to 1..=255, pass-through on nodes 5..11).
  - `dct_token_bool_tree_to_huff_probs` — the verbatim §13.1
    `DCTTokenBoolTreeToHuffProbs` transform converting an 11-entry
    node-probability vector into the 12-entry Huffman probability
    set the §13.2.2 / §13.3.2 Huffman decoders use.
- Re-exports of all of the above plus the supporting count
  constants (`NUM_DCT_TOKENS`, `NUM_TREE_NODES`, `NUM_PLANES`,
  `NUM_DC_CONTEXTS`, `NUM_AC_PREC_CONTEXTS`, `NUM_AC_BANDS`,
  `NUM_DC_NODE_EQS`) from the crate root.
- 25 unit tests pinning the Table 18/19/20/27 values, the all-128
  baselines, both update-flag banks, the DC node-context conversion
  (dummy-EOB-to-1, node 5..11 pass-through, hand-computed baseline
  expansion, 1..=255 clipping), and the §13.1 Huffman-prob
  transform (hand-computed against the all-128 listing).

### Added (clean-room round 10, 2026-05-25)

- `modes` module — the spec §10 macroblock coding-mode static
  surface. Reads no BoolCoder bits; the §10 `VP6_DecodeMode`
  traversal that consumes the surface stays deferred behind the
  §7.3 `Split` DOCS-GAP. Surfaces:
  - `CodingMode` enum — the ten Table 4 coding modes
    (`CODE_INTER_NO_MV`, `CODE_INTRA`, `CODE_INTER_PLUS_MV`,
    `CODE_INTER_NEAREST_MV`, `CODE_INTER_NEAR_MV`,
    `CODE_USING_GOLDEN`, `CODE_GOLDEN_MV`, `CODE_INTER_FOURMV`,
    `CODE_GOLD_NEAREST_MV`, `CODE_GOLD_NEAR_MV`) as a `#[repr(u8)]`
    enum whose discriminants match the spec's canonical 0..=9
    indexing throughout. Convenience predicates `is_intra`,
    `uses_golden`, `carries_new_mv` cover the three partitions §17
    reconstruction and §11 motion-vector paths route on.
  - `ModeAvailability::{NearestAndNear, NearestOnly, Neither}` —
    the three Table 5 ProbabilitySituation indices, with a
    `from_neighbours(nearest_exists, near_exists)` constructor
    mirroring the §10 traversal result.
  - `NEAR_MACROBLOCKS[12]` — the verbatim 12-entry (row, col)
    MB-unit neighbour offset table §10 traverses for
    Nearest/Near MV resolution.
  - `VP6_BASELINE_XMITTED_PROBS[3][20]` — the verbatim
    `VP6_BaselineXmittedProbs` I-frame `probXmitted` initialiser.
  - `VP6_MODE_VQ[3][16][20]` — the verbatim `VP6_ModeVq` 960-entry
    baseline-bank that `SetNewBaselineProbs` / `WhichVector` select
    from.
  - `mode_decision_tree_node_probability` /
    `build_mode_decision_tree` — the pure-integer transform that
    converts a `probXmitted[3][20]` table into the
    `ModeDecisionTree[3][10][9]` array §10's `VP6_DecodeMode`
    traversal consults at each of Figure 10's nine internal nodes.
  - `probability_mode_same` / `build_probability_mode_same` — the
    §10 `probModeSame` companion the decision-tree root reads to
    decide whether the MB inherits the previous MB's mode.
- Transcribed verbatim from `docs/video/vp6/vp6_format.pdf` §10
  (On2 Technologies, document version 1.02, August 2006). Like
  §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5/§12.1/§14 the module
  reads no BoolCoder bits and advances the decoder past round 9
  without touching the contested §7.3 `Split` formula.
- 42 unit tests over the §10 stage:
  - **CodingMode (7 tests):** Table 4 order pinned by enum
    discriminant; `from_index` round-trip across 0..=9 and rejection
    of out-of-range indices; `ALL` length matches `NUM_CODING_MODES`;
    `uses_golden` partitions the 4 golden modes; `is_intra` flags
    only `CODE_INTRA`; `carries_new_mv` flags the three
    fresh-MV-carrying modes; `Display` strings match the spec's
    canonical `CODE_*` names.
  - **ModeAvailability (3 tests):** Table 5 indices pinned; `from_index`
    round-trip; `from_neighbours` truth-table covering all four
    `(nearest, near)` input combinations including the (false, true)
    degenerate case folded to `Neither`.
  - **NEAR_MACROBLOCKS (5 tests):** 12-entry length; verbatim
    spot-checks of the four distance-1 and eight distance-2 entries;
    every offset unique; every offset is on a previously-decoded
    macroblock (raster-order causality: `dr <= 0`, and `dc < 0` when
    `dr == 0`).
  - **VP6_BASELINE_XMITTED_PROBS (5 tests):** `[3][20]` shape;
    per-situation verbatim spot-checks against the spec listing;
    byte-width invariant on every entry.
  - **VP6_MODE_VQ (6 tests):** `[3][16][20]` shape; first / last
    vector spot-checks against the spec listing for each of the
    three ProbabilitySituation rows.
  - **probModeSame (4 tests):** all-zero input returns 255 across
    all 30 `(av, last_mode)` pairs; baseline situation-0 + Intra
    last yields 128 (verifying `255 - 255*2/4 = 128`); baseline
    situation-1 + InterNoMv last yields 247 (verifying
    `255 - 255*8/238 = 247`); `build_probability_mode_same`
    bulk-build matches the per-element helper across all 30 cells.
  - **ModeDecisionTree (12 tests):** `[3][10][9]` build shape;
    all-zero input collapses every node probability to the floor
    value 1; every probability stays >= 1 across all 270 cells of
    the baseline tree; `build_mode_decision_tree` matches the
    per-node helper across all 270 cells; per-node spec-derived
    closed-form expected values for nodes 0, 3, 5, 7 and 8 at
    several different `(availability, lastmode)` selections, each
    walked through C[] weights and the `1 + 255*left/(1+branch)`
    formula in the test commentary; the lastmode-zeroing rule
    swapping which mode's weight is dropped from the `C[j]` table;
    out-of-range node index panics; iterating every one of the
    `VP6_ModeVq` 48 vectors as a `probXmitted` seed produces a
    valid tree (every probability >= 1).



- `scan` module — the spec §12.1 default zig-zag scan order. Reads no
  BoolCoder bits and is the natural predecessor to §15 inverse
  quantization and §16 inverse DCT in the per-block decode pipeline.
  Surfaces:
  - `DEFAULT_SCAN_ORDER[64]` — the verbatim `default_dequant_table[64]`
    from §12.1 / Figure 14 mapping zig-zag positions back to raster
    positions for the 8×8 block.
  - `DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[64]` — the const-evaluated
    inverse permutation for the encoder side.
  - `zigzag_to_raster_block` / `raster_to_zigzag_block` — block
    applicators that drive the permutation across all 64 coefficients.
- `dc_pred` module — the spec §14 DC coefficient prediction stage.
  Reads no BoolCoder bits; the DC delta the predictor is added to is
  what §13.2 BoolCoder-decodes, but the predictor itself is pure
  integer bookkeeping over already-decoded neighbour DC values.
  Surfaces:
  - `DcPredictionContext` — per-plane state holding the
    per-reference-bucket "last decoded DC value". `new` /
    `reset_at_frame_start` apply the spec's per-frame zero seed ("At
    the beginning of each frame this last decoded DC value is set to
    zero for each prediction frame type").
  - `predict` / `predict_and_record` — the §14 predictor for one
    block, implementing the four-row predictor table (`L`, `A`,
    `(L + A + Sign(L+A)) / 2`, last-DC seed) plus the
    same-reference-frame and intra-vs-inter neighbour-disqualification
    rules.
  - `ReferenceBucket::{Intra, InterLast, InterGolden}` — the three
    distinct "prediction frame types" §14 distinguishes (the §4
    "intra / previous-frame inter / golden-frame inter" trichotomy
    collapsed into one enum since both rules — same-reference and
    intra-vs-inter — partition the same bucket space).
  - `average_both_neighbours` / `dc_sign` — direct helpers exposing
    the §14 §3-`Sign` averaging formula and §3 `Sign()` so callers
    can drive the predictor manually.
- Transcribed verbatim from `docs/video/vp6/vp6_format.pdf` §12.1
  (default zig-zag) and §14 (DC prediction), On2 Technologies,
  document version 1.02, August 2006. Like
  §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5, both stages advance
  the decoder past round 8 without touching the contested §7.3
  `Split` formula.
- 46 unit tests over the §12.1 + §14 stages:
  - **scan (15 tests):** table length; DC-first invariant; high-freq
    last invariant; permutation completeness (every raster position
    hit exactly once); 11 spot-value checks against the spec listing;
    Figure 14's first-three-diagonals zig-zag traversal pattern; the
    inverse table's length, mutual-inverse property, and
    permutation-completeness; both block applicators' identity-under-
    seeded-input semantics; raster→zig-zag→raster and
    zig-zag→raster→zig-zag round-trips on non-trivial inputs; DC-only
    block lands at raster 0; highest-frequency-only block lands at
    raster 63.
  - **dc_pred (31 tests):** §3 `Sign()` three-branch + extreme-value
    behaviour; the two-neighbour averaging formula across all four
    sign permutations of `(L, A)` plus zero-sum and mixed-sign cases;
    exhaustive 41×41-grid match against the formula computed
    independently; sign-symmetry of the predictor; context
    zero-seeding at `new()`/`default()`; the §14 four predictor rows
    (neither / left-only / above-only / both) for positive and
    negative neighbour DC; the same-reference-frame rule for both
    left- and above-mismatched neighbours and both-mismatched; the
    intra-vs-inter and inter-last-vs-inter-golden bucket isolation
    rules; the last-DC seed update / read / per-frame reset; a
    cross-block worked example over a 2×3 intra grid exercising all
    four predictor scenarios in one test; defensive extreme-input
    panic-freedom.

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

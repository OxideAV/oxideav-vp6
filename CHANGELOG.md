# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (clean-room round 343, 2026-06-20)

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

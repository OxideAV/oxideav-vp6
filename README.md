# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 19 (2026-06-03).** The orphan-rebuild
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

### What round 9 lands

- `scan` — the spec §12.1 default zig-zag scan order. Surfaces:
  - `DEFAULT_SCAN_ORDER[64]` — the verbatim
    `default_dequant_table[64]` from §12.1 / Figure 14: at zig-zag
    position `i` the corresponding raster position is
    `DEFAULT_SCAN_ORDER[i]`. The decoder uses this to convert
    entropy-stage coefficients (which arrive in zig-zag order) back
    to raster order before §15 inverse quantization and §16 inverse
    DCT.
  - `DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG[64]` — the const-time
    inverse permutation, for the encoder side.
  - `zigzag_to_raster_block` / `raster_to_zigzag_block` — block
    applicators that drive the permutation across all 64 coefficients
    of an 8×8 block.
- Spec invariants enforced by the test suite: `DEFAULT_SCAN_ORDER[0]
  == 0` (DC always first), `DEFAULT_SCAN_ORDER[63] == 63`
  (highest-frequency last), table is a permutation of `0..64`, and
  the inverse table is its true inverse.
- `dc_pred` — the spec §14 DC coefficient prediction stage. Surfaces:
  - `DcPredictionContext` — per-plane state holding the
    per-reference-bucket "last decoded DC value" the spec mandates.
    `DcPredictionContext::new` returns a freshly-zeroed seed and
    `reset_at_frame_start` re-applies it per §14's "At the
    beginning of each frame this last decoded DC value is set to
    zero for each prediction frame type."
  - `predict` / `predict_and_record` — compute the §14 predictor for
    one block (the four-row predictor table: neither neighbour →
    per-bucket last-DC seed; only left → L; only above → A; both →
    `(L + A + Sign(L + A)) / 2`), and (for `predict_and_record`)
    record the post-`DcDelta` reconstructed DC as the new last-DC
    seed.
  - `ReferenceBucket::{Intra, InterLast, InterGolden}` — the three
    "prediction frame types" §14 distinguishes; cross-bucket
    neighbours are disqualified per the spec's same-reference-frame
    and intra-vs-inter rules.
  - `average_both_neighbours` / `dc_sign` — direct helpers for
    callers wanting to drive the §14 §3-`Sign` formula manually.
- Like §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5 these stages read
  **no BoolCoder bits** — the scan permutation is constant and the
  DC predictor is pure integer bookkeeping over already-decoded
  neighbour DC values and reference tags — so they advance the
  decoder past round 8 without touching the contested §7.3 `Split`
  formula. Together with rounds 2–6 they make the per-block
  reconstruction pipeline §14→§15→§16→§17.1 complete given a
  caller-supplied `DcDelta` (the §13.2 token that supplies the delta
  is itself BoolCoder-gated and remains deferred until the §7.3
  DOCS-GAP is resolved).
- The §12.2 per-frame *custom* scan-order updates and their
  `ScanOrderUpdateFlag` / `CoeffBandUpdateFlag` / `NewCoeffBand`
  fields (Table 17) are BoolCoder-coded and remain deferred.

### What round 10 lands

- `modes` — the spec §10 macroblock coding-mode static surface.
  Surfaces:
  - `CodingMode` — the ten Table 4 coding modes (`CODE_INTER_NO_MV`,
    `CODE_INTRA`, `CODE_INTER_PLUS_MV`, `CODE_INTER_NEAREST_MV`,
    `CODE_INTER_NEAR_MV`, `CODE_USING_GOLDEN`, `CODE_GOLDEN_MV`,
    `CODE_INTER_FOURMV`, `CODE_GOLD_NEAREST_MV`, `CODE_GOLD_NEAR_MV`)
    as a `#[repr(u8)]` enum whose discriminants match the canonical
    spec 0..=9 indexing throughout (`probXmitted`, `VP6_ModeVq`,
    `ModeDecisionTree`). Convenience predicates `is_intra`,
    `uses_golden`, `carries_new_mv` cover the three partitions the
    §17 reconstruction and §11 motion-vector paths route on.
  - `ModeAvailability::{NearestAndNear, NearestOnly, Neither}` — the
    three Table 5 "ProbabilitySituation" indices that gate which
    probability row applies, plus a `from_neighbours(nearest_exists,
    near_exists)` constructor mirroring the §10 traversal result.
  - `NEAR_MACROBLOCKS[12]` — the verbatim 12 (row, column) MB-unit
    neighbour offsets §10 traverses to resolve Nearest/Near MVs.
  - `VP6_BASELINE_XMITTED_PROBS[3][20]` — the verbatim
    `VP6_BaselineXmittedProbs` I-frame `probXmitted` initialiser.
  - `VP6_MODE_VQ[3][16][20]` — the verbatim `VP6_ModeVq` baseline
    bank `SetNewBaselineProbs` / `WhichVector` select from (960
    probability entries total).
  - `mode_decision_tree_node_probability` / `build_mode_decision_tree`
    — the pure-integer transform that converts a `probXmitted[3][20]`
    table into the `ModeDecisionTree[3][10][9]` array §10's
    `VP6_DecodeMode` traversal consults at each Figure 10 node.
  - `probability_mode_same` / `build_probability_mode_same` — the
    §10 `probModeSame` companion the decision-tree root reads to
    decide whether the MB inherits the previous MB's mode.
- The §10 `VP6_DecodeMode` traversal itself reads eleven BoolCoder
  bits (`B(probModeSame)` at the root, then `B(Stats[…])` for nodes
  0..=8 plus the per-node walk) and stays deferred behind the §7.3
  `Split` DOCS-GAP; every piece of *static data* and every
  *pure-integer derivation* the traversal would consult is now
  landed. The §10 Mode Probability Updates bitstream (Table 7/8/9)
  is similarly BoolCoder-gated and stays deferred.
- Like §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5/§12.1/§14 this
  stage reads **no BoolCoder bits** — every transform is pure
  integer arithmetic over already-known tables — so it advances the
  decoder past round 9 without touching the contested §7.3 `Split`
  formula.

### What round 11 lands

- `tokens` — the spec §13 DCT-coefficient token static surface (the
  BoolCoder-independent half of coefficient decoding). Surfaces:
  - `DctToken` — the twelve Table 18 tokens (`ZERO_TOKEN`,
    `ONE_TOKEN`..`FOUR_TOKEN`, `DCT_VAL_CATEGORY1`..`6`,
    `DCT_EOB_TOKEN`) as a `#[repr(u8)]` enum on the canonical 0..=11
    index, with each token's `(min, max, extra_bits)` geometry and the
    verbatim `extra_bit_probs` "Arithmetic Encoding the Extra Bits"
    column.
  - `TreeNode` — the eleven Table 20 coding-tree node names on the
    canonical 0..=10 index (the index into a node probability vector).
  - `baseline_dc_probs` / `baseline_ac_probs` — the all-128 keyframe
    initialisers for `DcProbs[2][11]` and `AcProbs[2][3][6][11]`.
  - `VP6_DC_UPDATE_PROBS[2][11]` / `AC_UPDATE_PROBS[3][2][6][11]` — the
    verbatim per-node update-flag probability banks (§13.2 / §13.3).
  - `DC_NODE_EQS[5][3][2]` — the verbatim `DcNodeEqs` slope/constant
    linear-equation table (Table 27), with the EOB dummy row.
  - `dc_probs_to_node_contexts` — the pure-integer §13.2 conversion
    expanding `DcProbs[2][11]` into the `DcNodeContexts[2][3][11]`
    per-context trees the §13.2.1 arithmetic DC decoder consults
    (linear equation on nodes 0..5, pass-through on 5..11, clipped to
    1..=255).
  - `dct_token_bool_tree_to_huff_probs` — the verbatim §13.1
    `DCTTokenBoolTreeToHuffProbs` transform converting an 11-entry
    node-probability vector into the 12-entry Huffman probability set
    the §13.2.2 / §13.3.2 Huffman decoders use.
- The §13 `VP6_DecodeToken` traversal (per-node `B(prob)` reads, the
  per-token extrabit `B(...)` loop, and the §13.3.3 AC zero-run reads)
  plus the §13.2/§13.3 per-frame probability-update bitstream stay
  deferred behind the §7.3 `Split` DOCS-GAP — every static table and
  pure-integer derivation the traversal would consult is now landed.
- Spec observation (deferred traversal, not this surface): Table 18
  lists `DCT_VAL_CATEGORY6` with 12 extra bits but only 11 arithmetic
  probabilities; the §13.2.1 magnitude loop
  (`BitsCount = ExtraBits - 1; … Probs[BitsCount]`) would index one
  past that 11-entry array before the separate `SignBit = b(1)`. The
  accessor reports both Table 18 columns verbatim and leaves the
  off-by-one for the BoolCoder-gated traversal to resolve.
- Like §10/§15/§16/§17 this stage reads **no BoolCoder bits** — every
  transform is pure integer arithmetic over already-known tables — so
  it advances the decoder past round 10 without touching the contested
  §7.3 `Split` formula.

### What round 12 lands

- `huffman` — the spec §7.2 Huffman tree construction and traversal
  primitives. VP6 supports **two** entropy schemes (§7): the
  BoolCoder (§7.3) used in partition 1 for mode/MV decisions, and
  the Huffman coder (§7.2) used as an alternate DCT-token scheme
  when the frame header's `UseHuffman` flag is set. The Huffman
  coder reads one whole raw bit per tree branch (`R(1)`; §3
  nomenclature) rather than a sub-bit `B(prob)` BoolCoder bit, so
  this stage is **independent of the §7.3 `Split` formula DOCS-GAP**.
- Surfaces:
  - `HuffNode` — the spec's `HUFF_NODE { Symbol, Prob, Left, Right }`
    struct with the `-1` sentinels for internal-vs-leaf (page 13);
    `INTERNAL_SYMBOL` (`-1`) and `NO_CHILD` (`-1`) constants
    document the spec's marker convention.
  - `create_huffman_tree` — the verbatim §7.2.1 `VP6_CreateHuffmanTree`
    builder. `N-1` bottom-up merge rounds over a stable-sorted leaf
    list; the returned `Vec<HuffNode>` has length `2N-1` with the
    root at index `2N-2`, matching the spec's *"Huffman tree root
    node is at position 2\*N-2 in SortList"* terminating comment.
    Rejects zero probabilities (§7: *"the value 0 is explicitly
    forbidden, so the valid range is 1 ≤ Node Probability ≤ 255"*)
    and `N < 2` inputs.
  - `decode_symbol` — the verbatim §7.2 `VP6_HuffmanDecodeSymbol`
    walk, parameterised over an external `FnMut() -> u8` raw-bit
    oracle so the actual byte-stream `R(1)` reader can land
    independently. Per §7 *"0 indicates left, 1 indicates right"*.
  - `tree_depth` / `codeword_for` — convenience walkers used in the
    test suite to verify shape invariants (skewed inputs give
    dominant symbols shorter codewords; balanced inputs give
    uniform-depth trees) and to drive the round-trip test.
- Stability invariant: §7.2.1 twice asks for "*ascending probability
  order maintaining relative order of nodes having equal probability*"
  — a *stable* sort, which is what `slice::sort_by_key` provides.
  Equal-probability inputs preserve their original relative ordering
  in the leaf zone, so encoder and decoder agree on the resulting
  tree shape from the probability vector alone.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13 this stage reads **no
  BoolCoder bits** — every step is pure integer arithmetic over the
  supplied probability vector — so it advances the decoder past
  round 11 without touching the contested §7.3 `Split` formula.
- The §13.3.3.2 AC zero-run probability conversion (a separate
  transform that feeds another Huffman tree) and the actual `R(1)`
  byte-stream reader are deferred for later rounds.

### What round 13 lands

- `zrl` — the spec §13.3.3 AC zero-run-length static surface (the
  BoolCoder-independent half of zero-run decoding). When the §13
  token decoder produces a `ZERO_TOKEN` in the AC position, a
  zero-run length follows. The run length can be coded with either
  of the two §7 entropy schemes; the BoolCoder path of §13.3.3.1
  stays deferred behind the §7.3 DOCS-GAP, while the Huffman path
  of §13.3.3.2 reads only raw `R(1)` bits over a tree this module
  builds, so it is **independent of the §7.3 `Split` formula
  DOCS-GAP**.
- Surfaces:
  - `ZrlBand` — Table 37 zero-coefficient-starting-band indices
    (`Band0` for AC coefficient positions 1–5; `Band1` for 6–63),
    on the spec's canonical `0..=1` indexing, with a
    `for_coefficient_position` helper that returns the band a
    given AC coefficient lives in.
  - `ZrlNode` — the fourteen Table 38 node indices. The first
    eight (`0..=7`) name the eight internal nodes of the Figure 16
    binary tree (`>4`, `>2`, `>1`, `>3`, `>8`, `>6`, `>5`, `>7`)
    in the spec's canonical order; the remaining six (`8..=13`)
    name the bit positions of the `(RunLength - 9)` six-bit
    suffix the BoolCoder path reads when the run is greater than
    8, with each extrabit's `(RunLength - 9) >> n & 1` shift
    exposed via `extrabit_shift`.
  - `ZERO_RUN_PROB_DEFAULTS[2][14]` — the verbatim
    `ZeroRunProbDefaults` keyframe initialiser.
  - `ZRL_UPDATE_PROBS[2][14]` — the verbatim `ZrlUpdateProbs`
    per-node `NewNodeProbFlag` update-flag probability bank used
    by the Table 41 BoolCoder reads (the reads themselves stay
    deferred).
  - `zrl_bool_tree_to_huff_probs` — the verbatim §13.3.3.2
    `ZRLBoolTreeToHuffProbs` transform that converts an 8-entry
    node-probability vector into a 9-entry Huffman probability
    set (one chain factor per Figure 16 internal-node branch;
    `>> 8` truncation per the spec's listing). Pure integer
    arithmetic, no BoolCoder.
  - `build_zrl_huffman_tree` — composes the §13.3.3.2 pseudo-code
    pair `ZRLBoolTreeToHuffCodes` + `VP6_BuildHuffTree` for one
    band. Runs `zrl_bool_tree_to_huff_probs` and then invokes
    `create_huffman_tree` (the round-12 §7.2 primitive) to build
    a `2N - 1 = 17`-node `HuffNode` tree the round-12 `decode_symbol`
    walker can traverse against a byte-stream `R(1)` reader.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2 this stage reads
  **no BoolCoder bits** — every operation is pure integer
  arithmetic over the supplied probability vector — so it advances
  the decoder past round 12 without touching the contested §7.3
  `Split` formula.
- **DOCS-GAP candidate (literal-vs-escape semantics):** the
  Figure 16 tree drawing carries two leaves labelled `8`, and the
  §13.3.3 demonstration code's `if (ZrlToken<8) EncodedCoeffs +=
  ZrlToken else EncodedCoeffs += 8 + R(6)` does not name which of
  the two Figure 16 leaves emits which `ZrlToken` value — i.e.
  whether the leaf at canonical index 7 carries a literal run of 8
  or is the `>8` escape that triggers the 6-extrabit `R(6)` read.
  The static surface itself is unambiguous (the conversion
  outputs 9 probabilities, one per leaf-codeword) so it lands; the
  literal-vs-escape disambiguation is reported as a docs-gap
  candidate for the orchestrator to commission.

### What round 14 lands

- `raw_bits` — the spec §3 `R(x)` raw-bit byte-stream reader, the
  byte-stream substrate underneath the §7.2 Huffman coder (and, once
  the §7.3 DOCS-GAP is closed, underneath the BoolCoder's `R(8)`
  refill reads as well). Surfaces:
  - `RawBitReader<'a>` — thin wrapper around
    `oxideav_core::bits::BitReader` exposing the §9 Tables 1/2
    MSB-first `R(n)` convention (the same one
    `frame_header::Vp6FrameHeader::parse` already consumes).
    Constructors `new` and `with_byte_offset` (for partition-2 reads
    where the caller has `Buff2Offset` from §9); bookkeeping accessors
    `bit_position`, `byte_position`, `bits_remaining`, `is_empty`,
    `is_byte_aligned`; alignment helper `align_to_byte` for the
    *"the next field starts at the next byte boundary"* phrasing some
    §9 entries use.
  - `read_bit` / `read(n)` — the standard MSB-first `R(1)` / `R(n)`
    reads. `read(n)` accepts `0..=32` bits per call (the largest
    single `R(n)` field in the spec is `R(16)` for `Buff2Offset`;
    §13.3.3 uses at most `R(6)`).
  - `read_lsb_first(n)` — the explicit *least-significant bit first*
    variant for the one place the spec inverts that ordering by name:
    §13.3.3.1 (page 78), *"the run length minus nine is encoded using
    six-bits, least significant bit first."* The `R(6)` escape suffix
    the §13.3.3 AC zero-run path reads (in both the BoolCoder and the
    Huffman entropy schemes — the spec's demonstration pseudo-code is
    `if (ZrlToken < 8) … else 8 + R(6)`) consumes this.
  - `read_huffman_symbol(&mut self, tree)` — convenience that drives
    the §7.2 `huffman::decode_symbol` walk against the byte stream
    directly, so callers using the §13 token Huffman path or the
    §13.3.3.2 zero-run Huffman tree don't have to assemble the
    `R(1)` closure themselves. Both walkers landed parameterised over
    an `FnMut() -> u8` oracle in rounds 12 and 13 precisely so the
    byte-stream reader could land independently here.
  - `RawBitError::{OutOfBits, TooManyBits}` — narrow error type. VP6
    partitions are bounded byte buffers per §6; reading past the end
    is a malformed-input condition the decoder surfaces cleanly.
- The reader implements `Clone + Copy` (it owns nothing but a borrowed
  slice and a position) so a parser can checkpoint and restore by
  assignment — useful for partition probes that look ahead without
  committing.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2 this module reads **no
  BoolCoder bits** — every operation is plain byte-stream bit
  arithmetic — so it advances the decoder past round 13 without
  touching the contested §7.3 `Split` formula. With this round the
  Huffman path of the §13 DCT-token decoder and the §13.3.3.2
  zero-run Huffman decoder both have a complete end-to-end data path
  (modulo the §13.3.3.2 9th-leaf semantics docs-gap noted in round
  13's `zrl` report).

### What round 15 lands

- `bool_coder` module / `BoolCoder` struct — the spec §7.3 binary
  arithmetic decoder, landing the primitive every remaining
  BoolCoder-coded layer (frame-header tail, §10 mode decoding,
  §11 motion-vector decoding, §13 DCT-token decoding, §13.3.3.1
  AC zero-run-length decoding) depends on. Surfaces:
  - `BoolCoder::new(bytes)` — `VP6_StartDecode`: 4-byte big-endian
    prefill of `Value`, `Range = 255`, `Count = 8`, `Pos = 4`.
    `Error::Truncated` for `bytes.len() < 4`.
  - `BoolCoder::decode_bool(probability) -> Result<u8, Error>` — the
    spec §7.3 `VP6_DecodeBool` per-bit step
    (`Split = 1 + ( ((Range-1) * Probability) >> 7 )`, branch on
    `Value < (Split << 24)`, update `Range`/`Value`, then run the
    renormalization loop pulling fresh bytes via `Pos`). The §3
    `B(x)` primitive every §10/§11/§13 tree walk consumes.
  - `BoolCoder::decode_b1() -> Result<u8, Error>` — single
    fixed-probability-128 bit (§3 `b(1)`).
  - `BoolCoder::decode_b(n) -> Result<u32, Error>` — `n`-bit
    fixed-probability-128 raw read (§3 `b(n)`), accumulated
    most-significant-bit first so the bit ordering matches §3
    `R(n)`.
  - Diagnostic accessors `range` / `value` / `count` / `pos` for
    test introspection.
- The previous DOCS-GAP block (rounds 1–14) flagged what looked like
  a self-contradiction in the §7.3 `Split` formula: at
  `Probability = 128, Range = 255` the formula evaluates to
  `Split = 255 = Range`, which makes the 0-branch unconditional and
  collapses every `b(n)` read to zero. The newly-staged clean-room
  errata `docs/video/vp6/vp6-errata-and-clarifications.md` entry
  **#35** resolves the gap: the `>> 7` is correct and intentional
  precisely because it makes probability 128 the half-interval
  point, exactly what a binary arithmetic coder's fixed-probability
  `b(x)` reads require. The `Split = 255` edge case is the natural
  half-interval boundary, not a defect — when `Value < 0xFF00_0000`
  (i.e. its top byte is below `0xFF`) the bit decodes to 0; otherwise
  the 1-branch fires. The formula is bit-exact as printed; only the
  unsigned-integer evaluation order ("multiply → shift-by-7 → add-1")
  needed pinning down.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2/§3 this module reads
  only the staged spec PDF and the staged clean-room errata. No
  third-party VP6 implementation has been consulted.

### What round 15 does NOT land

- The BoolCoder-gated payload parsers themselves: the §10
  `VP6_DecodeMode` mode-tree walk, the §11 motion-vector decoder,
  the §13 DCT-token tree walk, and the §13.3.3.1 AC zero-run
  BoolCoder traversal. These are the immediate beneficiaries of
  the b(n) primitive and the natural targets for subsequent rounds;
  every one of them depends on the BoolCoder landed here.
- Errata #67's §13 Table 18 `DCT_VAL_CATEGORY6` off-by-one
  resolution. That entry is separate to #35 and lands as part of
  a future §13.2 / §13.3 round.

### What round 16 lands

- `dct_decode` module — the first BoolCoder-consuming layer, the
  spec §13.2.1 arithmetic DC coefficient decoder. Surfaces:
  - `decode_dc_token(bc, &node_probs)` — the Figure 15 binary-tree
    walk down to a [`DctToken`] leaf, **DC variant**: the §13.2.1
    listing's `if (!B(ZERO_CONTEXT_NODE))` root + the value subtree
    descent through `ONE_CONTEXT_NODE`, `LOW_VAL_CONTEXT_NODE`,
    `HIGH_LOW_CONTEXT_NODE`, `CAT_THREEFOUR_CONTEXT_NODE`,
    `CAT_FIVE_CONTEXT_NODE`, `CAT_THREE_CONTEXT_NODE`,
    `CAT_ONE_CONTEXT_NODE`, `TWO_CONTEXT_NODE`,
    `THREE_CONTEXT_NODE`. Never returns `DctToken::EndOfBlock` (the
    DC tree forbids EOB, per §13.2 + the `DcNodeEqs` dummy row).
  - `decode_token_value(bc, token)` — the magnitude-loop + sign
    decode shared between §13.2.1 and §13.3.1. Reads `#ExtraBits − 1`
    magnitude bits (the errata-#67 corrected count) MSB-first using
    `DctToken::magnitude_probs`, then a separate fixed-probability-128
    sign bit, then reconstructs the signed coefficient via the
    `(value ^ -SignBit) + SignBit` identity. Short-cuts the
    `ONE_TOKEN..FOUR_TOKEN` constant-magnitude cases (no magnitude
    bits, sign only).
  - `decode_dc(bc, &node_probs)` — the full §13.2.1 wrapper:
    [`decode_dc_token`] followed by [`decode_token_value`], returning
    the signed `Dc` per the listing's final `Dc = ((value ^ -SignBit)
    + SignBit)`.
- `DctToken::magnitude_probs()` — the errata-#67 corrected
  magnitude-only probability slice (`#ExtraBits − 1` entries,
  trailing sign-prior `B(128)` stripped from `CATEGORY1..CATEGORY5`
  and from `ONE..FOUR`; `CATEGORY6`'s as-printed 11-entry slice is
  already magnitude-only). The legacy `extra_bit_probs` accessor is
  preserved verbatim for callers that need the as-printed Table 18
  column, with its docstring updated to point at the new accessor
  for the magnitude-loop traversal.
- 19 new unit tests pinning: the §13.2.1 zero-token short-circuit;
  the per-token magnitude-bit count vs the value range
  (`2^magnitude_bits = max - min + 1`); the MSB-first magnitude
  reading (all-zero stream → +min_value for every category); the
  constant-magnitude tokens' `+1..+4` output against the all-zero
  stream; the determinism + `decode_dc = walk + value` composition
  guarantees; the truncation surface on a 4-byte stream; the
  sign-reconstruction identity; and the structural "DC walk never
  returns EOB" property over a sweep of `node_probs` corners.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2/§3/§7.3, this module
  reads only the staged spec PDF and the staged clean-room errata.
  No third-party VP6 implementation has been consulted.

### What round 16 does NOT land

- The §13.3.1 AC arithmetic decoder. The tree walk + magnitude loop
  landed here is the substrate the AC path shares, but AC adds an
  `EOB_CONTEXT_NODE` branch above the same tree plus the
  "implicitly-1" first-decision shortcut when the preceding AC
  coefficient was zero, and on a `ZERO_TOKEN` leaf transitions into
  the §13.3.3 zero-run-length decoder. Wiring those AC-specific
  branches + the §13.3.3 zero-run integration is the immediate
  next-round target.
- The §13.2 / §13.3 per-frame probability **update** bitstream
  (Tables 22–24 / 31–35). That layer drives the per-frame
  `DcProbs` / `AcProbs` adaptation via `B(NewNodeProbFlag)` reads
  and a conditional `b(7)` `NewNodeProbValue`, both BoolCoder-coded;
  it stays deferred behind the round-15 BoolCoder primitive's
  reach.

### What round 17 lands

- `decode_ac_token(bc, prec, encoded_coeffs, &node_probs)` /
  `decode_ac_coefficient(bc, prec, encoded_coeffs, &node_probs)` —
  the spec §13.3.1 per-coefficient AC arithmetic decoder, the second
  BoolCoder-consuming layer. The AC tree differs from the §13.2.1 DC
  tree on two structural counts:
  - `EOB_CONTEXT_NODE` branch above the `ZERO_CONTEXT_NODE` root. A
    0-bit at the root no longer short-circuits to `ZERO_TOKEN`; it
    enters a `B(EOB_CONTEXT_NODE)` sub-decision whose 0-branch is
    `EOB_TOKEN` (end-of-block) and whose 1-branch is `ZERO_TOKEN`
    (proceed into the §13.3.3 zero-run decoder).
  - The "implicitly-1" first-decision shortcut. When the previously-
    decoded AC coefficient in the current scan order was the
    `ZERO_TOKEN` (`prec == WasZero`) **and** we are past the very
    first AC coefficient (`encoded_coeffs > 1`), the §13.3.1
    pseudo-code mandates the next token can be neither `ZERO_TOKEN`
    nor `EOB_TOKEN`, so the root decision is implicitly `1` and the
    walk starts at `ONE_CONTEXT_NODE`. (At the *first* AC position
    the `Prec` context came from the §13.2-decoded DC of the same
    block, not from a prior AC zero, so the shortcut is gated to
    fire only at `encoded_coeffs > 1`.)
- `AcOutcome` — the three-way per-step result the §13.3.1 pseudo-code
  distinguishes:
  - `AcOutcome::EndOfBlock` — exit the per-block loop, no coefficient
    emitted.
  - `AcOutcome::ZeroRun` — current AC coefficient is 0; caller invokes
    the §13.3.3 zero-run decoder to advance the scan position; the
    next `Prec` context is `WasZero` (which also gates the implicit-1
    shortcut on the *next* coefficient).
  - `AcOutcome::Value { coeff, next_prec }` — signed AC coefficient
    `coeff` was decoded; `next_prec` is the §13.3.1 update rule
    (`WasOne` if `|coeff| == 1`, `WasGreaterThanOne` otherwise).
- Static surface in `tokens`: `AcBand` (Table 30, six AC bands with
  `for_coefficient_position(usize) -> Option<AcBand>` returning the
  §13.3.1 `AcProbBand[encodedCoeffs]` band index for any AC scan
  position `1..=63`), `AcPlane` (Table 28, Y / UV), `AcPrecContext`
  (Table 29, `WasZero` / `WasOne` / `WasGreaterThanOne` with a
  `seed_from_dc(dc: i32) -> Self` constructor that implements the
  §13.3.1 first-AC seeding `if (dc == 0) Prec = 0; else if (dc == 1)
  Prec = 1; else Prec = 2`).
- The signed reconstruction reuses the round-16 `decode_token_value`
  magnitude-loop + sign kernel verbatim — the §13.2.1 and §13.3.1
  per-token magnitude/sign reads are identical (the errata-#67
  corrected magnitude-only slice and the separate fixed-prob-128
  sign bit), so no duplicated arithmetic.
- 18 new unit tests pinning: the implicit-1 shortcut's
  `(EncodedCoeffs > 1) && (Prec == WasZero)` gate (positive +
  negatives against both conjuncts); the EOB/ZERO inversion at the
  EOB-node (`B(EOB_CONTEXT_NODE) == 0 → EOB_TOKEN`,
  `== 1 → ZERO_TOKEN`); the `next_prec` update rule (sweep over
  `(prec, encoded_coeffs, node_probs, stream)` corners hitting both
  magnitude-1 and magnitude->1 paths); the §13.3.1 first-AC seeding
  `seed_from_dc` against `0 / 1 / 2 / −1 / 2114 / −2114`;
  determinism; `decode_ac_coefficient` = `decode_ac_token` +
  per-leaf value/sign; the truncation surface; and a structural
  leaf-set check across all twelve possible `DctToken`s. Plus
  Table 28/29/30 enum-surface tests in `tokens` (`AcBand` partition
  cover of `1..=63`, plane round-trip, prec round-trip).
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2/§3/§7.3/§13.2.1, this
  layer reads only the staged spec PDF and the staged clean-room
  errata clarifications. No third-party VP6 implementation has been
  consulted.

### What round 17 does NOT land

- The §13.3.3.1 BoolCoder zero-run-length traversal of Figure 16
  itself. The static probability data + the §13.3.3.2 Huffman path
  are already in `zrl`; the BoolCoder traversal would walk the
  Figure 16 tree reading `B(ZeroRunProbs[band][node])` per node and
  on the "run > 8" branch read six `B(...)` extrabits. The per-bit
  primitive is in place (round 15 `BoolCoder::decode_bool`) so this
  is a clean follow-on, but it's a separate logical unit. The
  `AcOutcome::ZeroRun` variant surfaces the hand-off point.
- The surrounding per-block driver loop. The §13.3.1 pseudo-code's
  `EncodedCoeffs ++` envelope, scan-order updates, and the
  `do { … } while (EncodedCoeffs < 64)` wrapper that ties this
  per-coefficient routine into the spec's per-block lifetime are a
  caller-side driver concern; `AcPrecContext::seed_from_dc` is
  exposed so the DC → first-AC handoff is wired correctly when the
  driver lands.
- The §13.3 per-frame `AcProbs` update bitstream (Tables 31–35).
  Same shape as the §13.2 DC update — `B(NewNodeProbFlag)` plus a
  conditional `b(7)` `NewNodeProbValue` — and uses the same
  BoolCoder substrate, but it is its own per-frame ingestion stage.

### What round 18 lands

- `inter::fetch_prediction_block_clamped(image, width, height, top,
  left, dx, dy, &mut pred)` — the **edge-clamped** integer-MC fetch.
  §11.5 "Unrestricted Motion Vectors" defines the buffer extension
  as "duplicating the edge values 48 times", which is mathematically
  equivalent to clamping the read position into the original image's
  `[0, width)` x `[0, height)` rectangle (an equivalence the `umv`
  module already records). The new entry point implements that
  equivalence directly: instead of reading from a 48-sample-bordered
  buffer, it reads from the unbordered reference image and clamps each
  per-sample `(row, col)` source position before the dereference.
- The §11.5 equivalence is exercised concretely as a property test:
  for any MV that stays inside the 48-sample §11.5 border the new
  fetch produces bit-identical output to `fetch_prediction_block` on
  the §11.5-bordered version of the same image — across both
  fully-inside reads and the four per-edge / four corner overhang
  cases. Beyond the 48-sample border (where the bordered fetch would
  index out of bounds) the clamped fetch remains well-defined and
  continues to serve up the appropriate corner / edge-row /
  edge-column pixel.
- Test count: **378** (15 new, all green). No spec-gap newly
  encountered; no errata change required.

### What round 18 does NOT land

- Any change to the existing `inter::fetch_prediction_block` /
  `umv::build_extended_buffer` path. Both remain in place; the
  bordered buffer is still needed by the §11.4 fractional-pixel
  interpolation (which reads two samples either side of the integer
  sample position and is therefore most naturally served by a
  pre-extended buffer). The clamped fetch is an opt-in
  memory-efficient alternative for the integer §17.2 / §17.3 path.
- A fractional-pixel edge-clamped variant. The §11.4
  bilinear / bicubic interpolation already runs against the
  §11.5-bordered buffer through `crate::interp`; lifting it onto the
  edge-clamp form would mean writing a clamped variant that handles
  the 2- and 4-tap filter reach explicitly, which is a separate
  logical unit.
- Any motion-vector decode path. The MV that drives the fetch is
  parsed upstream and behind the BoolCoder (§10 / §11); this round
  works strictly on the integer-MC consumer side, identical in scope
  to the existing `fetch_prediction_block` it sits alongside.

### What round 19 lands

- `dct_decode::decode_ac_zero_run(bc, band, &probs)` — the spec
  §13.3.3.1 BoolCoder zero-run-length traversal of Figure 16, the
  immediate consumer of the [`AcOutcome::ZeroRun`] hand-off the
  round-17 §13.3.1 AC decoder surfaces. Walks the Figure 16 binary
  tree reading a `B(prob)` BoolCoder bit at each of the eight
  internal nodes (the `>4`, `>2`, `>1`, `>3`, `>8`, `>6`, `>5`, `>7`
  decisions in the Table 38 / round-13 `ZrlNode` ordering); on the
  `>8` escape branch reads six additional `B(prob)` extrabits as the
  LSB-first encoding of `(RunLength - 9)` and reconstructs
  `RunLength = value + 9`. Returns the run length as a `u32` in the
  spec's full output range `1..=72` (literal `1..=8` from the eight
  binary-tree leaves plus `9..=72` from the `9 + (0..=63)` escape).
- This is the third BoolCoder-consuming layer (after the round-16
  §13.2.1 DC decoder and the round-17 §13.3.1 AC decoder); like its
  siblings it composes only round-15's [`BoolCoder::decode_bool`]
  primitive over the round-13 [`zrl`] static surface
  (`ZrlBand` / `ZrlNode` / `ZERO_RUN_PROB_DEFAULTS` /
  `ZRL_UPDATE_PROBS`) — no new spec material, no new errata. The
  §13.3.3.1 listing on page 78 (the `if (!B(ZRP[0])) … else …`
  cascade plus the six `B(ZRP[8..=13])` extrabit reads) maps
  branch-for-branch onto the implementation.
- 11 new unit tests pinning: the low-probability all-zero-stream
  leftmost-leaf result (`run = 1`); the band argument's row-selector
  semantics (output independent of `band` when `probs` is the same);
  the BoolCoder state advance under renormalization; the §7.3
  errata-#35 "`Split > Range` collapses to the 0-branch" shortcut
  forcing the leftmost-leaf result at `probs = [255; 14]`; the `>8`
  escape with zero extrabits yielding the minimum-escape `run = 9`;
  the root 0-branch picking the lower (`1..=4`) subtree; the
  truncation surface on a 4-byte stream that exhausts during the
  first renormalization; determinism (same bytes + same probs →
  same output) across a four-seed sweep; the `1..=72` output-range
  invariant across the canonical keyframe probability rows and five
  stream seeds and both bands; a decode against
  [`ZERO_RUN_PROB_DEFAULTS`] at the all-zero stream lands a
  well-defined run length per band; and the
  [`AcOutcome::ZeroRun`] hand-off contract pinned by composing the
  §13.3.1 outcome variant with `decode_ac_zero_run` at the keyframe
  defaults.
- Test count: **378 → 389** (11 new, all green). No spec-gap newly
  encountered; no errata change required. The §13.3.3.2 9th-leaf
  literal-vs-escape semantics docs-gap candidate (Figure 16 carries
  two leaves labelled `8` — see round 13's [`zrl`] report) is
  **unrelated** to this round's BoolCoder path: §13.3.3.1 reads its
  own discrete `>8`-internal-node decision and the six
  `(RunLength - 9)` extrabits, so the literal/escape distinction is
  unambiguous in the BoolCoder variant.

### What round 19 does NOT land

- The §13.3.3.2 Huffman zero-run path's actual walk against
  [`zrl::build_zrl_huffman_tree`]. The 9-entry Huffman probability
  set, the tree builder, and the §7.2 [`huffman::decode_symbol`]
  walker against a [`raw_bits::RawBitReader`] source are all
  landed (rounds 12–14), but the integration — symbol indexing
  decision + the `(RunLength - 9)` six-bit raw-suffix path —
  belongs to a separate driver round that also resolves the
  literal-vs-escape docs-gap candidate for the 9th leaf.
- The §13.3.3 per-frame Table 39–41 `ZeroRunProbs` update
  bitstream. Same shape as the §13.2 DC update and the §13.3 AC
  update: `B(NewNodeProbFlag)` plus a conditional `b(7)`
  `NewNodeProbValue`. Lands as a separate per-frame ingestion stage
  using the same round-15 BoolCoder substrate.
- The surrounding per-block driver loop that ties §13.3.1's
  `AcOutcome::ZeroRun` to `decode_ac_zero_run` and threads the run
  length back into the `EncodedCoeffs += ZeroRunCount` advance plus
  the §12.1 scan-order traversal. Caller-side concern; the
  [`AcOutcome::ZeroRun`] surface from round 17 and the run-length
  output from this round make the wiring straightforward when the
  driver lands.

## License

MIT — see [LICENSE](./LICENSE).

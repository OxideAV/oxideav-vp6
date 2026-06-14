# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 32 (2026-06-15).** The orphan-rebuild
scaffold from 2026-05-18 is being replaced incrementally by parsers
sourced from
[On2 Technologies' VP6 Bitstream & Decoder Specification](https://github.com/OxideAV/oxideav/blob/master/docs/video/vp6/vp6_format.pdf)
(document version 1.02, August 2006).

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

### What round 20 lands

- `prob_update` module — the per-frame BoolCoder-coded
  probability-update bitstream the §13.2 DC, §13.3 AC and §13.3.3
  ZRL token decoders all consume to mutate their persistent
  probability banks at every frame. The fourth BoolCoder-consuming
  layer (after rounds 16's §13.2.1 DC, 17's §13.3.1 AC and 19's
  §13.3.3.1 ZRL token decoders). The three update bitstreams share
  the same per-node shape (Tables 24, 35 and 41 are all the same
  two-field record: `B(flag_prob)` `NewNodeProbFlag` followed by a
  conditional `b(7)` `NewNodeProbValue`) and the same disambiguated
  reading of "½ of the new probability value" (§13.2 Table 24
  commentary): `new_prob = max(1, NewNodeProbValue * 2)`.
- `prob_update::decode_new_node_prob(bc, flag_prob)` — the per-node
  step, returning `Ok(None)` on `NewNodeProbFlag == 0` (skip) or
  `Ok(Some(prob))` on `NewNodeProbFlag == 1` (update).
- `prob_update::update_dc_probs(bc, &mut dc_probs, &flag_probs)` —
  the §13.2 driver, walking Tables 22 / 23 / 24 in the
  `[plane][node]` traversal order. Wires against
  [`tokens::VP6_DC_UPDATE_PROBS`] as the flag-prob bank and
  mutates a `DcProbs[2][11]`-shaped persistent bank in place.
- `prob_update::update_ac_probs(bc, &mut ac_probs, &flag_probs)` —
  the §13.3 driver, walking Tables 31 / 32 / 33 / 34 / 35 in the
  `[prec][plane][band][node]` traversal order. Notes the spec's
  two different dimension orderings: `AcProbs[plane][prec][band]
  [node]` (the per-token bank §13.3.1 reads from, matching
  [`tokens::baseline_ac_probs`]) vs `AcUpdateProbs[prec][plane]
  [band][node]` (the flag-prob bank this driver walks, matching
  [`tokens::AC_UPDATE_PROBS`]), and writes the spec walk order
  into the `AcProbs`-shaped target via outer-two-index remap.
- `prob_update::update_zero_run_probs(bc, &mut zero_run_probs,
  &flag_probs)` — the §13.3.3 driver, walking Tables 39 / 40 / 41
  in the `[band][node]` traversal order. Wires against
  [`zrl::ZRL_UPDATE_PROBS`] as the flag-prob bank and mutates a
  `ZeroRunProbs[2][14]`-shaped persistent bank in place.
- Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2/§3/§7.3/§13.2.1/
  §13.3.1/§13.3.3.1 this module composes only round-15's
  [`BoolCoder::decode_bool`] / [`BoolCoder::decode_b`] primitives —
  no new spec material, no new errata.
- 16 new unit tests pinning: the `flag_prob = 255` shortcut to
  `None` (errata-#35 `Split == Range` 0-branch); the flag-set
  path reading seven `b(7)` raw-bit tail; the `0 → 1` clip on the
  spec's "½ of the new probability value" formula; the `None`
  return as a no-op on the persistent bank; the `(1..=255)`
  range invariant swept across `flag_prob ∈ {1, 64, 128, 192,
  254}` and three representative byte streams; the three drivers'
  deterministic reproduction across two independent runs; the
  three drivers' range invariant across the full `[2 * 11]` (DC),
  `[2][3][6][11]` (AC) and `[2][14]` (ZRL) bank entries; the
  truncation surface on a 4-byte buffer that exhausts during the
  DC walk; the AC walk's byte-budget consumption against the
  worst-case `8 * 396 = 3168` BoolCoder-bit estimate; and the
  direct formula invariants sweep over all 128 `b(7)` values
  (clip-to-1, parity preservation under non-clip).
- Test count: **389 → 405** (16 new, all green). No spec-gap
  newly encountered; no errata change required.

### What round 20 does NOT land

- The per-frame driver that interleaves DC update → DC decode →
  AC update → AC decode → ZRL update → ZRL decode in their
  spec-mandated order; each per-frame ingest stage has its own
  per-bitstream-position constraints (see §9 frame header and
  §13's "Updates to this Baseline set of probabilities are made on
  each frame" commentary) that belong to a wiring round, not a
  per-stage round.
- The keyframe-vs-interframe gating of the update bitstream by the
  §9 frame header flags. Deferred behind the §9 BoolCoder-tail
  parser; the round-1 raw-bit prefix only reaches the
  pre-BoolCoder switch.
- The §13.3.3.2 Huffman zero-run path's literal-vs-escape
  9th-leaf integration. Still gated by the docs-gap candidate from
  round 13's [`zrl`] report.

### What round 21 lands

- `mv_decode` module — the spec §11.1 per-component motion-vector
  arithmetic decoder, the **fifth BoolCoder-consuming layer** (after
  rounds 16's §13.2.1 DC, 17's §13.3.1 AC, 19's §13.3.3.1 ZRL token
  and 20's per-frame probability-update layers). §11.1 describes how
  one signed component of a "new" motion vector is decoded from the
  bitstream by repeated BoolCoder reads against four per-axis
  probability banks:
  - `IsMvShortProbs[axis]` — the short-vs-long discriminator
    (single `B(...)` read).
  - `ShortMvProbs[axis][0..=6]` — the Figure 11 short-MV binary
    tree (3 BoolCoder reads producing magnitude `0..=7`).
  - `MvSizeProbs[axis][0..=7]` — the long-MV bit-position
    probabilities (seven BoolCoder reads in traversal order
    `[0, 1, 2, 7, 6, 5, 4]`, then conditionally bit 3 if any of
    bits `4..=7` are non-zero; otherwise bit 3 is implicit `1`).
  - `MvSignProbs[axis]` — the sign bit (final `B(...)` read with
    negation).
- Surfaces:
  - `decode_short_mv_magnitude(bc, &short)` — Figure 11 short-MV
    tree walk, magnitude `0..=7`.
  - `decode_long_mv_magnitude(bc, &size)` — the seven-bit traversal
    plus the conditional `B(size[3])` bit-3 read, magnitude
    `8..=255` (decoder formula bounds; §11.1's `<= 127` cap is an
    encoder-side constraint per "a long vector is defined as a
    vector with a length that is … less than or equal to 127").
  - `decode_mv_component(bc, &probs)` — the per-component wrapper:
    `B(IsVectorShort)` discriminator, short/long magnitude path,
    `B(MvSignProbs)` sign read with negation. Returns signed `i32`.
  - `decode_mv_pair(bc, &[probs_x, probs_y])` — full `(x, y)` pair,
    x first then y per §11.1's outer `for i = 0..=1` loop.
  - `MvProbs` — per-axis bundle (`is_short`, `short`, `size`,
    `sign`) plus `defaults(axis)` constructor.
  - `IS_MV_SHORT_PROBS_DEFAULTS` / `SHORT_MV_PROBS_DEFAULTS` /
    `MV_SIZE_PROBS_DEFAULTS` / `MV_SIGN_PROBS_DEFAULTS` — the
    verbatim §11.1 `Default_*` initialisers.
- 18 new unit tests pinning: the short-tree zero-path
  short-circuit; the short-tree all-1-path max-magnitude path
  (3-bit-deep walk yielding 7); the short magnitude range
  `0..=7` invariant across both default axis rows and four byte
  streams; the BoolCoder bit-advance bound through the short
  walk; the long-MV all-zero "implicit-bit-3" path yielding
  `0x08`; the long-MV all-ones path yielding `0xFF`; the
  long-MV lower-bound at the §11.1 `>= 8` floor; the long-MV
  read of bit 3 against the high-bits branch; per-component
  positive + negative + zero-stream + ones-stream
  composite decodes; the signed range invariant across
  default-vector probs; determinism (same bytes + probs →
  same output); pair-decoder axis independence (x with one
  prob set, y with another → x agrees across runs); truncation
  on a 4-byte buffer; default-against-zero-stream produces a
  well-defined signed result.
- Test count: **405 → 423** (18 new, all green). No spec gap
  newly encountered; no errata change required. Composes only
  round-15's [`BoolCoder::decode_bool`] over the verbatim §11.1
  default probability tables — no new spec material, no new
  errata, no third-party VP6 source consulted.

### What round 21 does NOT land

- The §11.2 per-frame MV-probability update bitstream. Same
  two-field shape as the §13.2 / §13.3 / §13.3.3 updates already
  landed in [`prob_update`] (`B(flag_prob)` plus a conditional
  `b(7)` `NewNodeProbValue`); the §11.2 driver is a thin wrapper
  over the existing `decode_new_node_prob` primitive once §11.2's
  flag-prob tables are transcribed (separate per-codec wiring
  round).
- The §10 mode-decode itself, which signals whether an MV is
  present for the current MB. The literal §10 `VP6_DecodeMode`
  pseudo-code's indentation is ambiguous around the
  `B(Stats[0]) == 0` branch (the inner `mode = CODE_INTRA;` is
  followed by `if (B(Stats[5])) ... else if (B(Stats[1])) ...
  else if (B(Stats[3])) ...` whose indentation could be parsed
  as either nested inside the `else` of `B(Stats[2])` or as the
  outer `else` of `B(Stats[0])`). The Figure 10 tree shape
  (left subtree: Inter modes + Intra + 4MV; right subtree:
  Golden modes) implies a specific traversal that the literal
  pseudo-code does not unambiguously match. The static surface
  (`probXmitted`, `ModeDecisionTree`, `probModeSame`) is in
  place from round 10; the mode-decoder traversal stays deferred
  pending a docs-gap clarification report.
- **DOCS-GAP candidate:** §10 `VP6_DecodeMode` literal
  pseudo-code (page 36) — the indentation of the `B(Stats[0])`
  / `B(Stats[2])` else-branches and the placement of
  `mode = CODE_INTRA;` create three plausible readings. A
  spec-faithful trace (one byte sequence walked through both
  Figure 10 and the literal pseudo-code) would pin which
  reading is intended.
- The §10 `CODE_INTER_FOURMV` per-block 2-bit codeword (Table
  10, two fixed-probability-128 bits selecting from
  `{CODE_INTER_NO_MV, CODE_INTER_PLUS_MV, CODE_INTER_NEAREST_MV,
  CODE_INTER_NEAR_MV}`). Lands cleanly on the existing
  `BoolCoder::decode_b1` primitive but is a separate logical
  unit — deferred to a per-MB driver wiring round alongside
  the resolved §10 mode-decoder.
- The §11 differential MV reconstruction (new vector = decoded
  delta + same-reference neighbour MV, or absolute when no
  qualifying neighbour exists). This module decodes one delta
  component; the neighbour-MV resolution traversal lives in the
  §10 caller (the [`modes::NEAR_MACROBLOCKS`] offsets landed in
  round 10).

### What round 22 lands

- `mv_prob_update` module — the spec §11.2 per-frame motion-vector
  probability-update bitstream, the **sixth BoolCoder-consuming
  layer** (after rounds 16's §13.2.1 DC, 17's §13.3.1 AC, 19's
  §13.3.3.1 ZRL token, 20's §13.2/§13.3/§13.3.3 per-frame
  probability updates, and 21's §11.1 MV component decoder).
  Walks the Table 13 update bitstream against four
  flag-probability lookup banks, mutating the persistent
  `[MvProbs; 2]` bank in place via the shared
  `prob_update::decode_new_node_prob` primitive — same
  `B(flag_prob)` + optional `b(7)` `NewProbability =
  max(1, value * 2)` recipe the §13 updates use. Surfaces:
  - `update_mv_probs(bc, &mut [MvProbs; 2])` — the Table 13
    walker. Iteration order, top-to-bottom per Table 13: X
    top-level (short-discriminator + sign), Y top-level, X
    short-tree (7 nodes), Y short-tree (7 nodes), X long-bits
    (8 bit positions in `LONG_VECTOR_BIT_ORDER`), Y long-bits.
  - `UPDATE_IS_MV_SHORT_PROBABILITIES` (`{237, 231}`),
    `UPDATE_MV_SIGN_PROBABILITIES` (`{246, 243}`),
    `UPDATE_SHORT_VECTOR_NODE_PROBABILITIES`
    (`[[253,253,254,…], [245,253,254,…]]`),
    `UPDATE_LONG_VECTOR_BIT_PROBABILITIES`
    (`[[254,…,250,250,252], [254,…,251,251,254]]`) — verbatim
    §11.2 `Update*Probabilities` initialisers (page 43-44).
  - `LONG_VECTOR_BIT_ORDER` — the eight-entry traversal-to-bit
    permutation `[0, 1, 2, 7, 6, 5, 4, 3]` Table 15 walks.
    Differs from §11.1's decode-time traversal
    `[0, 1, 2, 7, 6, 5, 4]` by the trailing `3`: at update
    time bit 3's probability is always present.
- 9 new unit tests pinning the verbatim flag-probability
  tables; the `LONG_VECTOR_BIT_ORDER` permutation vs `0..=7`;
  the order's length-matches-`NUM_MV_SIZE_NODES`; the
  flag-bank dimensions vs `MvProbs` shape; the helper
  functions' signatures (compile-time shape check); Table 13's
  X-before-Y step-order constants; the round-20
  `decode_new_node_prob` primitive round-trip at moderate
  flag-prob `128`; the X-row vs Y-row root-node flag-prob
  asymmetry on the short-tree; and the long-vector flag-prob
  tail-vs-head ordering.

### What round 23 lands

- `fourmv` module — the spec §10 / Table 10 per-Y-block
  coding-mode signaling for `CODE_INTER_FOURMV` macroblocks.
  Once the MB-level §10 mode decision lands on
  `CODE_INTER_FOURMV`, each of the four 8x8 luma blocks
  transmits a two-bit codeword over the round-15 BoolCoder at
  fixed probability 128 per bit, selecting from a reduced
  four-mode set `{InterNoMv, InterPlusMv, InterNearestMv,
  InterNearMv}` indexed by codeword value `00..=11` (Table 10,
  page 37). Surfaces:
  - `FOURMV_BLOCK_MODES` — the four-entry Table 10 lookup, in
    canonical codeword-value order
    `[InterNoMv, InterPlusMv, InterNearestMv, InterNearMv]`.
  - `NUM_LUMA_BLOCKS_PER_MB` (`4`),
    `NUM_FOURMV_BLOCK_MODES` (`4`) — shape constants.
  - `decode_fourmv_block_mode(bc)` — single-block decoder. One
    `BoolCoder::decode_b(2)` read (two fixed-probability-128
    bits MSB-first) plus the lookup.
  - `decode_fourmv_block_modes(bc)` — four-block raster-order
    walker (block 0 = top-left, 1 = top-right, 2 =
    bottom-left, 3 = bottom-right). Eight BoolCoder bits per
    MB total. Returns `[CodingMode; 4]`.
- 10 new unit tests pinning Table 10 cover, the shape
  constants, the all-zero stream → `InterNoMv` decode, the
  walker vs four serial calls producing byte-identical
  BoolCoder state, determinism, the reduced-set-membership
  invariant across seeded streams, the truncation surface on a
  4-byte buffer, and per-block reduced-set membership on
  single-block seeds.
- Test count: 432 → 442. `cargo fmt` and `cargo clippy
  --all-targets -D warnings` clean.

### What round 23 does NOT land

- The §10 `VP6_DecodeMode` MB-level mode-tree traversal — the
  decision that the MB *is* `CODE_INTER_FOURMV` is the gated
  piece (DOCS-GAP candidate carried forward from round 21
  around the `B(Stats[0])` / `B(Stats[2])` else-branch
  indentation).
- The §11.4 per-block MV-component decode for the four blocks
  selected as `CODE_INTER_PLUS_MV` by this module (each such
  block carries its own explicitly-coded MV; the round-21
  `decode_mv_component` primitive handles the per-component
  arithmetic but the per-block wiring into the InterFourMv
  driver remains a downstream piece).
- The chroma-block averaging-of-four-Y-vectors rounding rule
  for the InterFourMv MB's two 8x8 chroma blocks (spec §10
  prose; lives with the MB-level reconstruction driver).

### What round 24 lands

- `near_mv` module — the spec §10 Nearest / Near alternative-MV
  neighbour walker, the BoolCoder-independent piece of the §10
  mode-decode pipeline. Resolves the §10 alternative-MV pair plus
  the implied `ModeAvailability` row index (Table 5) from the
  surrounding already-decoded macroblock grid. Walks
  `modes::NEAR_MACROBLOCKS` in spec order applying the two §10
  predicates at each step (`mv != (0, 0)` and matching
  `ReferenceBucket`); the first qualifying neighbour becomes
  `nearest_mv`, the second becomes `near_mv`; the walker
  short-circuits at the second hit. Surfaces:
  - `MotionVector` — typed `(x, y)` motion-vector pair in spec
    ¼-pixel units (signed `i16`, range `±127` per §11.1's
    magnitude cap), with `MotionVector::ZERO` and
    `MotionVector::is_zero()` so the §10 "non (0, 0)" predicate
    lives in one place.
  - `NeighbourMv` — one neighbour's `{mv, reference}` metadata,
    reusing `dc_pred::ReferenceBucket` so the same-reference
    gating the §10 walker shares with §14 DC prediction stays in
    one canonical enum.
  - `NearMvResolution` — walker output:
    `{ nearest_mv: Option<MotionVector>, near_mv: Option<MotionVector>,
       availability: ModeAvailability }` plus the
    `NearMvResolution::NONE` sentinel for the no-qualifying-neighbour
    case and `has_nearest()` / `has_near()` shortcuts.
  - `resolve_near_mvs(row, col, reference, neighbour_at)` —
    closure-driven walker. `neighbour_at: FnMut(i32, i32) ->
    Option<NeighbourMv>` so callers keep their preferred MV-grid
    storage; out-of-frame `(row, col)` positions are reported
    with negative coordinates.
  - `resolve_near_mvs_from_grid(grid, grid_width, row, col, reference)`
    — dense-grid convenience wrapper. Backs the walker with a flat
    `&[Option<NeighbourMv>]` slice indexed `row * grid_width + col`;
    out-of-bounds access returns `None` without panic.
- 18 new unit tests pinning: `NEAR_MACROBLOCKS` spec order
  (re-asserted locally so a future reorder trips the walker
  tests); `MotionVector::is_zero` across the `(0, 0)` /
  `(±1, 0)` / `(0, ±1)` / `(±127, ±127)` boundary; empty-grid →
  `NearMvResolution::NONE`; single-above-neighbour →
  `NearestOnly`; two-neighbours in spec order →
  `NearestAndNear`; different-reference skip; `(0, 0)`-MV
  skip; short-circuit at the second hit (visitor-counting);
  top-left-corner negative-coordinate reporting; dense-grid
  wrapper resolution; bottom-right and top-left corner
  bounds safety; reference filtering through the wrapper;
  `NONE` constant matches walker output; the
  `availability` field matches
  `ModeAvailability::from_neighbours`; `Intra` reference
  filtering; maximum-magnitude (±127) MV qualification;
  twelve-qualify-all → first-two picked.
- Test count: 442 → 460. `cargo fmt` and `cargo clippy
  --all-targets -D warnings` clean.

### What round 24 does NOT land

- The §10 `VP6_DecodeMode` Figure-10 BoolCoder traversal itself
  (the `ModeDecisionTree` lookup the resolved availability would
  index). Static probability surface landed in round 10; the
  per-bit walk is gated on the round-21 DOCS-GAP candidate
  about the `B(Stats[0])` / `B(Stats[2])` else-branch
  indentation.
- The §11 differential-MV reconstruction that combines a
  round-21-decoded delta with the resolved Nearest MV when the
  chosen mode is `CODE_INTER_PLUS_MV` / `CODE_GOLDEN_MV`. The
  §11 intro paragraph carries an extra constraint ("immediately
  to the left of or immediately above") stricter than the
  12-neighbour traversal landed here, so the differential layer
  is its own logical unit.
- A driver wiring `near_mv` into the per-MB decode loop —
  needs the §10 mode traversal landed first to know which
  `ModeAvailability` row to feed the §10 probability surface.

### What round 22 does NOT land

- Sample-exact stream-driven walk against a real .vp6
  inter-frame fixture. The published §11.2 flag-prob banks
  cluster at the top of `1..=255`; the round-15 BoolCoder's
  behaviour at `Probability > 128` on synthetic high-prob
  streams drives the implementation's internal `range`
  accumulator outside the round-15 self-correcting envelope
  documented in errata #35. Sample-exact validation needs an
  integration test bound to a conformant .vp6 bitstream
  (encoder-produced, not synthetic).
- Intra-vs-inter gating of the §11.2 walk (intra frames reset
  the bank to §11.1 defaults instead of walking the update
  bitstream; lives in the upstream per-frame driver alongside
  the §9 BoolCoder-tail).
- The §10 mode-decode traversal (DOCS-GAP candidate carried
  forward from round 21 — the `B(Stats[0])` / `B(Stats[2])`
  else-branch indentation ambiguity).
- The §11 differential MV reconstruction (combining a decoded
  delta with the same-reference neighbour MV).

### What round 25 lands

- `mv_diff` module — the §11 intro's differential motion-vector
  reconstruction (BoolCoder-independent). Combines a
  round-21-decoded delta with a same-reference **above/left**
  neighbour MV — the §11 intro's "immediately to the left of or
  immediately above" constraint, strictly narrower than the §10
  12-neighbour walk — or with `(0, 0)` when neither qualifies
  (the "coded absolutely" branch). Surfaces:
  `DIFF_REFERENCE_OFFSETS` (the two-entry `(-1, 0)` / `(0, -1)`
  table pinned against the first two `modes::NEAR_MACROBLOCKS`
  entries); `select_diff_reference_mv` /
  `select_diff_reference_mv_from_grid` (the two-neighbour walker
  applying the §10 predicates `mv != (0, 0)` + matching
  `dc_pred::ReferenceBucket`); `reconstruct_diff_mv` (the
  per-component sum `final = reference + delta`); and
  `reconstruct_new_mv` / `reconstruct_new_mv_from_grid`
  (one-shot compositions).
- Test count: 460 → 484 (24 new, all green).

### What round 26 lands

- `scan_update` module — the spec §12.2 per-frame custom scan
  order, the seventh BoolCoder-consuming layer. Resolves the
  round-9 deferral ("the §12.2 per-frame custom scan-order
  updates and their `ScanOrderUpdateFlag` /
  `CoeffBandUpdateFlag` / `NewCoeffBand` fields (Table 17) are
  BoolCoder-coded and remain deferred") now that the round-15
  BoolCoder primitive is in place. Surfaces:
  - `CUSTOM_SCAN_BAND_RANGES[16]` — the Table 16 sixteen-band
    partition of the AC positions `1..=63`.
  - `COEFF_BAND_UPDATE_FLAG_PROBS[64]` — the verbatim §12.2
    `CoeffBandUpdateFlagProbs` bank (the spec's `NA` DC dummy
    stored as the out-of-range poison value `0`, never read).
  - `DEFAULT_BAND_ASSIGNMENT` — the §12.1-derived per-coefficient
    default banding (the default scan order is the identity in
    zig-zag space, so coefficient `c` defaults to the Table 16
    band whose position range contains `c`).
  - `decode_scan_order_update` — the Table 17 record: `b(1)`
    `ScanOrderUpdateFlag`; on 0 the assignment resets to the
    default per §12.2's "the scan order must be reset to the
    default"; on 1 dispatches into the per-coefficient walk.
  - `decode_coeff_band_updates` — the 63-set walk:
    `B(flag_probs[c])` `CoeffBandUpdateFlag` per AC coefficient
    in standard zig-zag order plus a conditional `b(4)`
    `NewCoeffBand`. Flag-prob bank parameterised like the
    round-20 `prob_update` drivers (synthetic-stream tests use
    moderate banks inside the round-15 BoolCoder's
    self-correcting envelope; the published bank's 255-saturated
    tail is exercised once a real-bitstream integration lands).
  - `build_custom_scan_order` — the §12.2 rebuild: bands in
    ascending order, coefficients within a band "sorted into
    ascending order based upon the original zig-zag scan order",
    DC pinned at position 0; pinned against the §12.2 worked
    example ("if AC7 and AC21 are labeled as belonging to band
    3, then AC7 will be assigned position 11 and AC21 position
    12").
  - `custom_scan_order_to_raster` — composition with the §12.1
    `DEFAULT_SCAN_ORDER` so the §15 inverse quantizer and §16
    IDCT consumers get raster positions directly.
- Intra-vs-inter seeding (§12.2: intra frames reset to the
  default before the deltas apply; inter frames carry the
  previous frame's assignment) is documented on the decoder and
  left to the per-frame driver, alongside the Figure 5
  "Coefficient Probability Updates" sequencing (scan-order
  updates, then §13.3.3 ZRL updates, then §13.3 AC updates).
- Test count: 484 → 498 (14 new, all green).

### What round 27 lands

- `derive_fourmv_chroma_mv` / `average_four_away_from_zero` — the
  spec §10 chroma motion vector for a `CODE_INTER_FOURMV`
  macroblock, the round-23 explicit deferral ("the chroma-block
  averaging-of-four-Y-vectors rounding rule"). §10 prose (page
  28): "the motion vector for the two chroma blocks is computed
  by averaging the four Y vectors (rounding away from zero)."
  BoolCoder-independent. Each component is averaged
  independently; "rounding away from zero" is implemented as the
  *directed* rounding mode (every non-integer quotient moves to
  the next integer of larger magnitude, `sign(sum) *
  ceil(|sum| / 4)`), matching the spec's parallel use of the
  opposite directed mode in §14 ("the arithmetic average of
  their DC values, **truncated towards zero**"). A
  nearest-with-ties-away reading would require the word "half"
  the prose does not contain; the doc comment records the
  distinction and the `sum = 1 → +1` case that separates the two
  readings. Inputs are the four **resolved** per-Y-block vectors
  (post-Table-10 mode application) in ¼-pel luma units; the
  output is in the same units and feeds the §11.4 fractional
  fetch at 1/8 chroma-sample precision via `MvShift::Chroma`,
  exactly as for single-MV macroblocks.
- 10 new unit tests pinning: exact quotients untouched (±508
  boundary included); the directed-vs-nearest distinguishing
  cases (`sum = 1, 5` and negative mirrors); odd symmetry
  `f(-s) == -f(s)` over the full conformant range ±508; the
  signed-ceiling definition cross-check + the §11.1-derived
  ±127 output bound over the full range; identical-vectors
  identity; per-component independence; exact cancellation →
  zero; permutation invariance; the corner sweep (all 256
  ±127-corner combinations) plus a seeded interior sweep
  respecting the component bound; and a mixed-magnitude worked
  example (x: 7 → 2, y: −9 → −3).
- Test count: 498 → 508. `cargo fmt` and `cargo clippy
  --all-targets -D warnings` clean.

### What round 27 does NOT land

- **The §9 frame-header BoolCoder-tail parser — investigated as
  this round's primary target and found still blocked, now with
  a sharper diagnosis.** Errata #35 (the §7.3 `Split`
  clarification staged 2026-06) is **internally inconsistent**:
  its summary table concludes the shift amount is `>> 7`
  ("divide by 128; prob 128 = half interval. **Not** `>> 8`"),
  but every quantitative property its own rationale relies on
  holds only under `>> 8`:
  - At `Probability = 128`, `Split = 1 + ((Range-1)*128 >> 7)
    = 1 + (Range-1) = Range` — the **full** interval, not the
    half interval the errata asserts (`"= 1 + (Range-1) ≈
    Range/2"` is arithmetically false as printed). The `Bit = 1`
    interval `Range - Split` is **empty**, so a `b(n)` field can
    never carry a 1-bit: decoding a 1 requires the top byte of
    `Value` to be `0xFF` and then sets `Range = 0`, which the
    renormalization loop can never recover (`0 * 2 = 0`).
    Header fields like `VFragments = 30` are therefore
    undecodable under `>> 7` — and unencodable, so no conformant
    encoder can have produced such a stream.
  - At `Probability = 255`, `Split = 507 > Range` — a negative
    `Bit = 1` interval, violating the errata's own "keeps
    `Split` strictly less than `Range`, preserving a non-empty
    `Bit = 1` interval" claim and the spec's `Range` ∈ 0–255
    attribute bound.
  - The errata's counter-claim that `>> 8` "would make
    Probability = 128 yield only a quarter-range split" is also
    false: `1 + ((Range-1)*128 >> 8) = 1 + (Range-1)/2 ≈
    Range/2`, exactly the even split.
  Every numbered property the errata's rationale demands
  (half-interval at 128, near-full at 255, both branches
  non-empty, `Range ≤ 255` invariant) is satisfied by `>> 8`
  and violated by `>> 7`. The spec PDF's literal `>> 7` (page
  15) appears to be the original typo, and the errata's
  conclusion rationalises it while its mathematics argue for
  `>> 8`. Resolving this requires a docs correction (or a
  worked byte-trace pinning the intended arithmetic); the
  round-15 `BoolCoder` implements the errata's literal `>> 7`
  and all seven BoolCoder-consuming layers inherit the
  degeneracy at `Probability >= 128` — none of the existing
  synthetic-stream tests are sensitive to it, but sample-exact
  decoding of any real `.vp6` bitstream will be.
- The §10 `VP6_DecodeMode` MB-level traversal — DOCS-GAP
  candidate carried forward from round 21 (the `B(Stats[0])` /
  `B(Stats[2])` else-branch indentation), unaddressed by the
  staged errata.
- The per-MB driver that wires `derive_fourmv_chroma_mv` to the
  four resolved block vectors (needs the §10 traversal plus the
  §11.4 per-block MV decode wiring).

### What round 28 lands

- `block_decode` module — the §13 **per-block coefficient
  reconstruction driver**, composing the per-coefficient entropy
  primitives (rounds 16/17/19) with the §12 scan orders and the §15
  dequantizer:
  - `decode_block_coefficients` — the §13.2.1 DC decode seeding
    `CoeffData[0]` and the `Prec` context (`dc == 0 / 1 / else`),
    then the §13.3.1 per-block do-while: per-iteration
    `[Prec][Band]` probability-row re-selection, the
    `(EncodedCoeffs > 1) && (Prec == 0)` implicit-1 shortcut,
    per-leaf `Prec` updates (`Prec = 0` on the ZERO leaf, 1 / 2 on
    value leaves), the ZERO-leaf transition into the §13.3.3.1
    zero-run decoder (band per `ZrlBand[EncodedCoeffs]`, run
    inclusive of the triggering position, saturating past the block
    end), and the EOB exit. Returns `BlockCoeffs { coeffs[64],
    coeff_count }` with the invariant `coeffs[coeff_count..] == 0`.
  - `dequantize_to_raster` / `BlockCoeffs::dequantize_to_raster` —
    the §12 scan-position-to-raster permutation (default §12.1
    zig-zag or a §12.2 custom order via
    `custom_scan_order_to_raster`) fused with the §15 DC/AC scalar
    dequantizer in one pass.
  - `decode_block_to_raster` — the one-shot composition; its
    `DequantizedBlock::raster` output feeds the §16 `idct_block`
    directly.
  - Two spec readings documented on the module: the §13.3.1
    listing's `AcUpdateProbs[Prec][Plane][Band]` lookup is a naming
    slip for the persistent `AcProbs[plane][prec][band][node]`
    decoding bank (the §13.3.1 prose "stored in ACProbs act as the
    binary decoding node probabilities" + the §13.3 dimension
    tables resolve it); and the listing's EOB `EncodedCoeffs++` /
    post-loop `EncodedCoeffs--` choreography nets to the
    coefficient count on the EOB path while the post-loop value is
    never consumed by any later listing, so the driver defines the
    unambiguous `coeff_count` semantics instead.
- 13 new unit tests pinning: the all-zero-stream empty block (EOB at
  the first AC, count 1); two hand-computed §7.3 traces (DC-only
  block with forced magnitude-1 DC then EOB; the DC-seeded `Prec`
  selecting the first AC probability row); a forced
  zero-run → implicit-1 value → EOB walk exercising every outcome
  arm in one block; driver-vs-listing replay equality (an
  independent do-while replay on the public primitives) across a
  seed-stream × bank × plane grid including the final BoolCoder
  byte position; structural invariants over arbitrary seed streams
  (count bounds, zero tail, Table 18 magnitude bound); determinism;
  the truncation surface on a 4-byte stream under a
  CATEGORY6-forcing bank; fused-vs-two-step dequant equality against
  `zigzag_to_raster_block` + `dequantize_block`; the DC/AC factor
  split; custom-scan routing through the §12.2 AC7/AC21 worked
  example; one-shot-vs-staged equality with identical bit
  consumption; and the empty block IDCT-ing to all-zero differences.
- Test count: 508 → 521. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean.

### What round 28 does NOT land

- The per-frame / per-macroblock driver that selects each block's
  `DcNodeContexts[plane][context]` row from the §14 DC-prediction
  neighbour state and threads `BlockCoeffs` through prediction +
  IDCT + §17 reconstruction — blocked on the §10 `VP6_DecodeMode`
  MB traversal (DOCS-GAP carried forward) and the §9 frame-header
  BoolCoder tail (errata #35 internal inconsistency, see round 27).
- The §13.2 DC-coefficient *token-context* wiring (Table 26's
  left/above zero-DC context selection) — pure composition over
  `dc_pred`, a natural next-round target alongside this driver.

### What round 29 lands

- `frame_assembly` module — the §2/§13/§17 **block-to-plane frame
  assembly** stage. The per-block reconstruction pipeline was already
  complete in prior rounds (§13 entropy → §12 scan → §15 dequant →
  §16 `idct_block` → §17 `reconstruct_intra_block` / `inter`), each
  producing a single reconstructed 8x8 block. This stage owns the
  per-plane raster image buffers and writes each finished block into
  its correct pixel position, accumulating the per-block decoder
  output into a full decoded YUV 4:2:0 image — the "frame assembly"
  the codec needs to turn blocks into actual output pixels.
  - `Plane` — a dense raster `u8` image plane (luma or one chroma
    channel). `place_block(block_row, block_col, &block)` writes a
    reconstructed 8x8 block at block-grid coordinates;
    `place_block_at_pixel(top, left, &block)` writes at an arbitrary
    pixel origin; both reject out-of-bounds placements
    (`AssemblyError::OutOfBounds`) without a partial write.
    `with_block_grid(cols, rows)` allocates a block-sized plane;
    `sample` / `samples` / `samples_mut` read back the result.
  - `Frame` — the three Y/U/V planes plus the §9 `HFragments` /
    `VFragments` geometry. The luma plane is `HFragments * 8` x
    `VFragments * 8` pixels (§9 worked example: a 320x240 image has
    HFragments 40 / VFragments 30); the chroma planes are sized to
    the macroblock grid (`mb_cols * 8` x `mb_rows * 8`), since each
    macroblock contributes exactly one 8x8 block to each chroma plane
    under §2's 4:2:0 sub-sampling.
  - `Frame::place_macroblock_luma` maps the four in-macroblock luma
    blocks to their §13 (page 58) 2x2 raster positions — `0=TL,
    1=TR, 2=BL, 3=BR` (`MB_LUMA_BLOCK_OFFSETS`) — into the luma
    plane; `place_macroblock_chroma` places the one U + one V block;
    `place_macroblock` does all six in one call. Partial edge
    macroblocks (when `HFragments` / `VFragments` are odd) skip their
    off-grid overhang luma blocks without error.
  - `mb_cols_for` / `mb_rows_for` — the §2 4:2:0 chroma-grid
    derivation (`ceil(fragments / 2)` macroblocks per dimension).
  - `BLOCK_DIM` (8), `MB_LUMA_DIM` (16), `MB_LUMA_BLOCKS` (4)
    constants, transcribed from §2 / §16.
- BoolCoder-independent: every operation is pure integer index
  arithmetic plus a raster 8x8 write over already-reconstructed §17
  pixel blocks. Like §15/§16/§17/§11/§12/§14 it reads **no BoolCoder
  bits**, so it advances the decoder without touching the contested
  §7.3 `Split` formula. The §11.5 Unrestricted-Motion-Vector border
  extension stays in `umv` (applied to a *reference*-plane copy
  before inter prediction reads it); these reconstruction planes are
  unbordered.
- 13 new unit tests pinning: the constant transcription against §2 /
  §13 / §16; plane geometry from the §9 320x240 worked example; the
  exact 8x8 write region with raster orientation preserved (no
  row/col transpose) and no leakage outside the block; out-of-bounds
  rejection leaving the plane untouched; even/odd 4:2:0 frame
  geometry (mb-grid round-up); the macroblock 2x2 luma ordering;
  one-chroma-block-per-MB placement; the full six-block macroblock;
  the odd-fragment edge-MB overhang skip; and an **end-to-end** test
  that drives the real `idct_block` + `intra_block_to_pixels`
  pipeline for a DC-only block and assembles the genuine
  reconstructed pixels into a frame plane (not synthetic markers) —
  confirming the assembly stage lands actual decoded output.
- Test count: 521 → 534. `cargo fmt --check` and `cargo clippy
  --all-targets --no-deps -- -D warnings` clean.

### What round 29 does NOT land

- The per-frame / per-macroblock **driver** that walks the macroblock
  grid in raster order (mode decode → MV decode → per-block
  coefficient decode → reconstruct → assemble), threading the §14 DC
  prediction state, the reference frame buffers and the §11.5 UMV
  border. It is gated upstream on the §7.3 BoolCoder degeneracy
  (errata #35 internal inconsistency, see round 27) and the §10
  `VP6_DecodeMode` traversal DOCS-GAP (round 21); this round lands
  the assembly primitive that driver will call once those gaps are
  resolved.
- Output scaling (§9 `ScalingMode` / `OutputHFragments` /
  `OutputVFragments`) — the decoded frame may be presented at a
  different resolution than its coded `HFragments` / `VFragments`;
  the post-decode scale is a separate output-stage concern. The
  assembled `Frame` here is at coded resolution.
- The §11.5 UMV-bordered *reference* buffer construction for inter
  prediction (already available via `umv::build_extended_buffer`);
  this round assembles the unbordered reconstruction planes only.

### What round 30 lands

- `tokens::DcContext` — the **§13.2 Table 26 DC node context**: the
  deferred "DC-coefficient token-context wiring (Table 26's left/above
  zero-DC context selection)" round 28 explicitly flagged as a natural
  next-round target. The §13.2.1 arithmetic DC decoder does not read
  from `DcProbs` directly; it reads from
  `DcNodeContexts[plane][context]`, where `context` is selected per
  block from whether the immediately adjacent **left** and **above**
  blocks' predicted DCs were zero or non-zero. Table 26 enumerates the
  three situations and `DcContext` names them on the canonical 0..=2
  index:
  - `BothZero` (0) — left's predicted DC was 0 **and** above's was 0.
  - `OneNonZero` (1) — exactly one of left / above is non-zero.
  - `BothNonZero` (2) — both are non-zero.
  - `index` / `from_index` round-trip; `from_neighbours(left_non_zero,
    above_non_zero)` (the Table 26 partition); `select_row(
    &DcNodeContexts[plane])` (the §13.2.1
    `ContPtr = DcNodeContexts[Plane][Context]` indexing returning the
    11-entry node-probability row the round-16 `decode_dc` consumes);
    and a `Display`. A **missing** neighbour (the frame's left edge has
    no left block, the top edge no above block) counts as a **zero-DC**
    neighbour per §13.2's "have 0 or non 0 dc values" note, so the
    top-left corner block decodes with `BothZero`.
- `tokens::DcZeroContextTracker` — the per-plane raster bookkeeping
  companion that supplies the Table 26 context without the caller
  re-deriving neighbour positions. As a plane's blocks are decoded in
  raster order (left→right, top→bottom) it holds the running
  left-neighbour non-zero flag plus one above-neighbour flag per
  column. `context_for()` returns the current block's `DcContext`;
  `record(non_zero)` records the block's own predicted-DC non-zero
  state and advances the raster position, wrapping at the row end
  (resetting the left flag — the new row's first block has no left
  neighbour) with the just-completed row's flags becoming the above
  row.
- `block_decode::decode_block_coefficients_ctx` — the
  context-resolving convenience over round-28's
  `decode_block_coefficients`. Instead of a pre-resolved
  `DcNodeContexts[plane][context]` row (which the §13.2.1 caller had to
  select by hand), it takes the per-plane `DcNodeContexts[plane][3][11]`
  bank plus a `DcContext` and performs the `[context]` selection
  internally before invoking the base decoder — so a driver threading
  a `DcZeroContextTracker` per plane calls it directly. Verified
  byte-exact against the manual pre-resolved path.
- Like §15/§16/§17/§11/§12/§14, this context selection is pure integer
  index arithmetic over already-decoded neighbour DC state — it reads
  **no BoolCoder bits**, picking *which* probability row the §13.2.1
  tree walk consults, not consuming any. It advances the decoder past
  round 29 without touching the contested §7.3 `Split` formula.
- Test count: 534 → 547 (13 new, all green). `cargo fmt --check` and
  `cargo clippy --all-targets --no-deps -- -D warnings` clean. No new
  spec material read beyond §13.2 (Tables 25/26/27) of the staged
  `vp6_format.pdf`; no errata change required.

### What round 30 does NOT land

- The per-frame / per-macroblock **driver** that walks the macroblock
  grid, threads the §14 DC-prediction state into the
  `DcZeroContextTracker`'s `record` calls, and decodes each block's
  coefficients with its resolved context — still gated upstream on the
  §7.3 BoolCoder degeneracy (errata #35 internal inconsistency, round
  27) and the §10 `VP6_DecodeMode` MB-mode traversal DOCS-GAP (round
  21). This round lands the Table 26 context primitive that driver
  will call; the driver itself remains the immediate blocked target.
- The §14 same-reference partition of the zero-DC context. §13.2
  specifies the Table 26 test on the raw neighbour DC without further
  qualifying it by reference frame (unlike the §14 *predictor*, which
  disqualifies cross-reference neighbours). `DcZeroContextTracker`
  therefore tracks the raw neighbour non-zero state; a driver that
  needs per-reference partitioning can run one tracker per
  plane × bucket.
- Output scaling (§9 `ScalingMode` / `OutputHFragments` /
  `OutputVFragments`) — carried forward from round 29 as a separate
  output-stage concern.

### What round 31 lands

- `scaling` module — the spec §9 `ScalingMode` / `Output*Fragments`
  static output-scaling surface (Table 2, page 24), the deferred
  "Output scaling (§9 `ScalingMode` / `OutputHFragments` /
  `OutputVFragments`)" item rounds 29 and 30 carried forward as a
  separate output-stage concern. After a frame is decoded at its
  **coded** resolution (`HFragments` x `VFragments` 8x8 blocks), §9
  says it "*may be encoded at a different resolution to the eventual
  size that it is presented on output*"; the header carries the output
  geometry plus a two-bit `ScalingMode`. Surfaces:
  - `ScalingMode` — the four §9 named modes
    (`MaintainAspectRatio` (0), `ScaleToFit` (1), `Center` (2),
    `Other` (3)) on the spec's listing order
    ("*MAINTAIN_ASPECT_RATIO, SCALE_TO_FIT, CENTER, OTHER*"), with
    `from_b2(value)` (a decoded `b(2)` field → mode, `None` for any
    value outside `0..=3`) and an `index()` round-trip.
  - `FrameGeometry` — a `(h_fragments, v_fragments)` pair (shared by
    the coded and output descriptions) → pixel dimensions
    (`luma_width` / `luma_height` = `fragments * 8`, pinned against
    the §9 worked example: a 320x240 image has HFragments 40 /
    VFragments 30) plus the §2 4:2:0 macroblock-grid round-up
    (`mb_cols` / `mb_rows` = `ceil(fragments / 2)`, matching the
    `frame_assembly` chroma-grid derivation).
  - `OutputScaling` — the desired output `FrameGeometry` paired with a
    `ScalingMode`, plus `is_identity(coded)` reporting whether a given
    coded geometry already matches the output (no resampling needed,
    mode-independent).
  - `FRAGMENT_DIM` (8) — the §2 / §16 transform-block edge constant.
- Like §15/§16/§17/§11/§12/§14 this stage reads **no BoolCoder bits** —
  every value is pure integer arithmetic over already-decoded fragment
  counts — so it advances the decoder past round 30 without touching
  the contested §7.3 `Split` formula. The `Output*Fragments` /
  `ScalingMode` fields are themselves `b(8)` / `b(2)` BoolCoder-coded
  in Table 2's tail; this module describes their *meaning* once
  decoded, leaving the read to the (still-blocked) §9 header-tail
  parser.
- **DOCS-GAP candidate (per-mode placement geometry):** the staged
  `vp6_format.pdf` names the four scaling modes and states that output
  geometry may differ from coded geometry, but it does **not** specify
  the per-mode pixel-mapping algorithm — how a smaller coded image is
  positioned within (CENTER) or stretched to (SCALE_TO_FIT) the output
  rectangle, what aspect-preserving fit MAINTAIN_ASPECT_RATIO performs,
  or what OTHER signals. §2 lists "*Scaling on output after decode*"
  only as a feature bullet. The actual resampling/placement math is
  therefore out of scope; the typed mode surface and the dimension
  derivations that math would consume land here. The §9 listing supplies
  the only ordering for the `b(2)` field, so the discriminants follow it
  (`0..=3`).
- Test count: 547 → 558 (11 new, all green). `cargo fmt --check` and
  `cargo clippy --all-targets --no-deps -- -D warnings` clean. No new
  spec material read beyond §9 (Table 2) / §2 of the staged
  `vp6_format.pdf`; no errata change required.

### What round 31 does NOT land

- The per-mode pixel-resampling/placement algorithm for the four
  `ScalingMode` values (DOCS-GAP candidate above — the math is
  unspecified in the staged doc).
- The §9 frame-header BoolCoder-tail parser that reads the
  `Output*Fragments` / `ScalingMode` `b(8)` / `b(2)` fields — still
  gated upstream on the §7.3 BoolCoder degeneracy (errata #35 internal
  inconsistency, round 27).
- The per-frame / per-macroblock **driver** — still gated on the §7.3
  BoolCoder degeneracy and the §10 `VP6_DecodeMode` MB-mode traversal
  DOCS-GAP (round 21).

### What round 32 lands

- `mode_prob_update` module — the spec §10 per-frame
  **mode-probability-update bitstream** (Tables 7 / 8 / 9 and the
  Figure 9 magnitude tree), the deferred "Mode Probability Updates
  bitstream" round 10 flagged as BoolCoder-gated. The **eighth
  BoolCoder-consuming layer** (after rounds 16's §13.2.1 DC, 17's
  §13.3.1 AC, 19's §13.3.3.1 ZRL token, 20's §13.2/§13.3/§13.3.3
  probability updates, 21's §11.1 MV component decode, 22's §11.2 MV
  probability update and 26's §12.2 custom scan-order update). The §10
  mode decoder reads from a persistent `probXmitted[3][20]` table; at
  every I-frame it resets to `modes::VP6_BASELINE_XMITTED_PROBS`, for
  P-frames it persists — and in **both** cases the frame header carries
  this optional update bitstream that mutates the table before the
  per-MB mode decode. Surfaces:
  - `update_mode_probs` — the full driver, walking the three
    `ModeAvailability` situations in `ModeAvailability::ALL` order
    (NearestAndNear, NearestOnly, Neither) and mutating the persistent
    `probXmitted[3][20]` table in place.
  - `update_mode_probs_for_situation` — one situation's Table 7 / Table 8
    walk: `SetNewBaselineProbs B(174)` → on `1`, `WhichVector b(4)`
    copies `modes::VP6_MODE_VQ[situation][which]` into the
    `probXmitted` row; `VectorUpdatesPresentFlag B(254)` → on `1`,
    twenty Table 9 per-value records applied left-to-right.
  - `decode_mode_prob_update_value` — one Table 9 record:
    `UpdateFlag B(205)` (0-branch returns the value unchanged, one bit
    consumed), then `Sign B(128)` + the Figure 9 `Difference`, applied
    to the current value.
  - `decode_mode_prob_difference` — the verbatim §10 Figure 9
    magnitude-tree decode. The pseudo-code maps branch-for-branch:
    `if B(171) return (sign*4)*(1+B(83))`; else `if !B(199)` the
    small-difference subtree `B(140)→12 / B(125)→16 / B(104)→20 /
    fall-through 24`, else the `b(7)` escape `sign * diff * 4`. Returns
    the already-signed delta.
  - `apply_prob_difference` — `0..=255` clamp. `probXmitted` entries are
    counts the §10 decision-tree builder consumes through
    `1 + probXmitted[...]` denominators, so a value of `0` is valid
    (the baseline banks contain many zeros) — distinct from the
    directly-read `B(prob)` node probabilities elsewhere in §13, which
    forbid `0`.
  - Flag/bit constants `SET_NEW_BASELINE_PROBS_FLAG` (174),
    `VECTOR_UPDATES_PRESENT_FLAG` (254), `UPDATE_FLAG_PROB` (205),
    `SIGN_PROB` (128), `WHICH_VECTOR_BITS` (4), `LONG_DIFFERENCE_BITS`
    (7), `FIGURE9_NODE_PROBS`.
- **Spec inconsistency noted (resolved by dimension):** the Table 9
  figure region labels the record list "*Ten Sets of:*", but the §10
  prose states **twice** that `ModeProbUpdateVector` is "*20 sets of
  probability updates*" and the `probXmitted[3][20]` second dimension is
  20 (ten modes, each with a same-as-prior and a different-from-prior
  probability — Table 6). The 20-count is the consistent reading
  (prose + array dimension agree); the "Ten Sets of" label is a spec
  slip. This module walks 20 records per situation, matching
  `PROB_XMITTED_ROW_LEN`. Same shape of resolution-by-dimension as
  round 28's `AcUpdateProbs` naming slip.
- **Transcription fix (same commit):** six `modes::VP6_MODE_VQ`
  baseline-bank vectors had single-element shifts / trailing-value
  errors against the §10 listing (situation 0 vectors 1 & 10;
  situation 1 vectors 3 & 7; situation 2 vectors 7 & 15). All 48
  vectors now match the spec verbatim. This bank is the one
  `SetNewBaselineProbs` copies into `probXmitted`, so its correctness
  is load-bearing for the layer landing this round.
- Like the prior BoolCoder-consuming layers, the **high-probability**
  flag reads (`B(174)` / `B(254)` / `B(205)`) and the Figure 9 1-branch
  leaves (12 / 16 / 20 / the `4*(1+B(83))` and `b(7)` escape) are
  exercised only against a real conformant `.vp6` bitstream: on
  synthetic streams those `> 128` reads take their 0-branch under the
  round-15 BoolCoder (errata #35 degeneracy, round 27 diagnosis), so the
  synthetic-stream-reachable Figure 9 leaf is the `sign * 24`
  fall-through. The same limitation rounds 22 / 26 / 27 documented.
- Test count: 558 → 574 (16 new, all green). `cargo fmt --check` and
  `cargo clippy --all-targets --no-deps -- -D warnings` clean. No new
  spec material read beyond §10 (Tables 5/6/7/8/9, Figure 9) of the
  staged `vp6_format.pdf`; no errata change required.

### What round 32 does NOT land

- The §10 `VP6_DecodeMode` MB-mode traversal itself — DOCS-GAP
  candidate carried forward from round 21 (the `B(Stats[0])` /
  `B(Stats[2])` else-branch indentation ambiguity), unaddressed by the
  staged errata. This round prepares the `probXmitted` table that
  traversal consults; the traversal stays deferred.
- Sample-exact stream-driven validation of the high-prob flag paths /
  Figure 9 1-branches — needs a real encoder-produced `.vp6`
  inter-frame fixture (the synthetic-stream limitation above).
- The per-frame driver that calls `update_mode_probs` at the right
  point in the frame header (after the §9 BoolCoder tail) and the
  intra-vs-inter `probXmitted` seeding — still gated on the §7.3
  BoolCoder degeneracy (errata #35 internal inconsistency, round 27).

## License

MIT — see [LICENSE](./LICENSE).

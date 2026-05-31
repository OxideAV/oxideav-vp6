# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

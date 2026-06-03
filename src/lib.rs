//! # oxideav-vp6
//!
//! Pure-Rust On2 VP6 ("vp6f"/"VP60"/"VP61"/"VP62") video decoder for
//! the [`oxideav`](https://github.com/OxideAV/oxideav) framework.
//!
//! **Status:** clean-room rebuild in progress (orphan-rebuild scaffold
//! from 2026-05-18; rounds 1–2 below). The crate previously contained a
//! direct-port implementation that was retired under the workspace
//! clean-room provenance policy. The rebuild reads only:
//!
//! * `docs/video/vp6/vp6_format.pdf` — On2 Technologies' "VP6 Bitstream
//!   & Decoder Specification" (document version 1.02, August 2006).
//!
//! No third-party VP6 source has been consulted at any stage of the
//! rebuild.
//!
//! ## Round 1 surface
//!
//! * [`Vp6FrameHeader`] — parser for the raw-bit prefix of a VP6
//!   frame header (spec §9 Table 1 plus Table 2's four R(n) fields).
//!   Covers `FrameType`, `DctQMask`, `MultiStream`, `Vp3VersionNo`,
//!   `VpProfile`, the `Reserved` bit, and the conditional 16-bit
//!   `Buff2Offset`.
//! * [`Error`] — narrow `Truncated` / `NotImplemented` enum so callers
//!   can distinguish "I ran out of bytes" from "this code path isn't
//!   wired up yet".
//!
//! ## Round 2 surface
//!
//! * [`DequantContext`] — per-frame inverse-quantization context (spec
//!   §15). Resolves the DC and AC scalar quantizer factors from the
//!   header's `DctQMask` via the two 64-entry tables
//!   ([`DC_QUANTIZATION_TABLE`] / [`AC_QUANTIZATION_TABLE`]) and
//!   dequantizes a block's coefficients with a per-coefficient
//!   multiply. This layer is **independent of the BoolCoder** — the
//!   quantizer factor depends only on the already-parsed raw-bit
//!   `DctQMask` — so it advances past the round-1 surface without
//!   touching the contested §7.3 `Split` formula.
//!
//! What rounds 1–2 deliberately do **not** land:
//!
//! * Any field downstream of the BoolCoder switch in the frame header
//!   (`VFragments`/`HFragments`/scaling/filter selectors/`UseHuffman`),
//!   mode/MV decoding, and DCT-token decoding — every one of these is
//!   `b(n)`/`B(x)`/`T` BoolCoder-coded.
//! * The BoolCoder primitive itself (the `VP6_DecodeBool` bit decoder).
//!   Its initialization (`VP6_StartDecode`) and normalization are
//!   unambiguous, but the per-bit `Split` step is blocked.
//!
//! See the round-15 surface below for the BoolCoder primitive that
//! closes the previous DOCS-GAP about the §7.3 `Split` formula.
//!
//! Round 2 worked **around** the (then-) blocked BoolCoder by landing
//! the inverse-quantization layer (spec §15, [`DequantContext`]),
//! which is driven solely by the raw-bit `DctQMask` and never calls
//! `VP6_DecodeBool`.
//!
//! ## Round 3 surface
//!
//! * [`idct_block`] — the spec §16 inverse DCT transform. A separable,
//!   fixed-point integer IDCT (14-bit precision, seven Q16 cosine
//!   constants) that converts an 8x8 block of dequantized coefficients
//!   in raster order back to pixel / pixel-difference values via a row
//!   pass and a column pass. Like the §15 dequant layer it is
//!   **BoolCoder-independent** — it consumes the output of
//!   [`DequantContext::dequantize_block`] and never calls
//!   `VP6_DecodeBool` — so it advances past round 2 without touching
//!   the contested §7.3 `Split` formula.
//!
//! ## Round 4 surface
//!
//! * [`reconstruct_intra_block`] — the spec §17.1 intra block
//!   reconstruction step: per-sample `OutputValue = InputValue + 128`
//!   followed by an inclusive clip to `0..=255`. Inverts the
//!   encoder-side level shift that §17.1 documents (`prior to encoding
//!   the value 128 is subtracted from all data samples`). The natural
//!   successor to the §16 IDCT for the intra-coded path; like §15
//!   and §16 it is **BoolCoder-independent** so it advances the decoder
//!   past round 3 without touching the contested §7.3 `Split` formula.
//!   The remaining §17.2–§17.4 cases (zero MV, full-pixel MV, sub-pixel
//!   MV) combine the same clip with motion compensation against a
//!   reference reconstruction buffer; they are blocked on the BoolCoder
//!   for MV decoding upstream.
//!
//! ## Round 5 surface
//!
//! * [`interp`] — the spec §11.4 fractional-pixel motion-compensation
//!   interpolation filters: the bilinear 2-tap kernel ([`interp::bilinear_point`])
//!   and 4-tap bicubic kernel ([`interp::bicubic_point`]) with their full
//!   tap tables ([`interp::BILINEAR_LUMA_FILTERS`],
//!   [`interp::BILINEAR_CHROMA_FILTERS`], [`interp::BICUBIC_FILTER_SET`]),
//!   their separable two-pass 8x8 block applicators
//!   ([`interp::bilinear_block`] / [`interp::bicubic_block`]), and the
//!   §11.4 `Var16Point` prediction-block variance metric
//!   ([`interp::var_16_point`]) used by the Advanced-Profile filter
//!   selector. These produce the interpolated sub-pixel prediction
//!   samples that §17.4 reconstruction consumes. Given a reference buffer,
//!   stride and fractional phase the kernels are pure integer pixel
//!   arithmetic — they never call `VP6_DecodeBool`, so this stage (like
//!   §15, §16 and §17.1) advances the decoder without touching the
//!   contested §7.3 `Split` formula. The motion vector that *selects* the
//!   phase and source is BoolCoder-gated upstream; the filter-selection
//!   *size* threshold is also deferred on an undefined `MAX_MV_EXTENT`
//!   constant (see the [`interp`] module DOCS-GAP note).
//!
//! ## Round 6 surface
//!
//! * [`inter`] — the spec §17.2–§17.4 inter-block reconstruction stage.
//!   [`reconstruct_inter_block`] applies §17's shared recombination
//!   (`OutputValue = PredictionValue + PredictionError`, inclusive clip
//!   to `0..=255`) — one function for all three inter cases because the
//!   spec's recombination pseudocode is byte-identical across §17.2
//!   (zero vector), §17.3 (full-pixel vector) and §17.4 (fractional
//!   vector); the cases differ only in how the prediction block is
//!   sourced. [`fetch_prediction_block`] is the §17.2/§17.3 integer copy
//!   from a reference reconstruction buffer (zero MV = co-located,
//!   full-pixel MV = integer offset); §17.4 sources its prediction from
//!   the round-5 [`interp`] filters instead. [`MvShift`],
//!   [`whole_sample_aligned`], [`luma_frac`] and [`chroma_frac`]
//!   implement the §11.4 motion-vector decomposition (`MvX >> MvShift`
//!   whole part + low-bit fractional phase; `MvShift` is 2 for luma /
//!   3 for chroma). Like §15/§16/§17.1/§11.4 this stage reads **no
//!   BoolCoder bits** — given an already-decoded MV, a reference buffer
//!   and the IDCT residual, every step is pure integer pixel
//!   arithmetic — so it advances the decoder without touching the
//!   contested §7.3 `Split` formula. Note the inter path applies **no**
//!   `+128` intra level shift (§17.1 only): the prediction already
//!   carries the DC.
//!
//! ## Round 7 surface
//!
//! * [`loopfilter`] — the spec §11.3 prediction loop filter. Implements
//!   the 4-tap `(1, -3, 3, -1)` deblocking filter applied to prediction
//!   blocks that straddle 8x8 boundaries in the reference frame, with
//!   the quantizer-indexed `Bound()` soft-clip that preserves real
//!   reference-frame edges while smoothing block-boundary
//!   discontinuities. Surfaces the
//!   [`loopfilter::PREDICTION_LOOP_FILTER_LIMIT_VALUES`] 64-entry
//!   quantizer-indexed limit table, the
//!   [`loopfilter::boundary_x`] / [`loopfilter::boundary_y`] block-edge
//!   offset calculations (`(8 - (mV & 7)) & 7`), the
//!   [`loopfilter::bound`] soft-clip, the per-edge
//!   [`loopfilter::prediction_loop_filter_function`] applicator, and the
//!   [`loopfilter::filter_vertical_boundary`] /
//!   [`loopfilter::filter_horizontal_boundary`] 2-D wrappers. Per the
//!   spec the deringing variant is "not currently supported by the
//!   decoder" so only the deblocking filter is implemented; Simple
//!   Profile disables the loop filter entirely (caller-side gate). Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4 this stage reads **no BoolCoder
//!   bits** — given a whole-sample-aligned MV, a prediction buffer and
//!   the frame's `DctQMask`, every step is pure integer pixel arithmetic
//!   — so it advances the decoder without touching the contested §7.3
//!   `Split` formula.
//!
//! ## Round 8 surface
//!
//! * [`umv`] — the spec §11.5 Unrestricted Motion Vector (UMV) border
//!   extension. Surfaces the [`umv::UMV_BORDER_SIZE`] constant (48,
//!   per spec), [`umv::extended_stride`] / [`umv::extended_height`] /
//!   [`umv::origin_offset`] geometry helpers, the in-place
//!   [`umv::extend_border`] applicator, and the
//!   [`umv::build_extended_buffer`] convenience constructor that
//!   allocates a UMV-bordered buffer, copies the original image into
//!   the inner rectangle, and fills the borders. The extension is pure
//!   edge replication in the spec-mandated "first in x, then in y"
//!   order: every original-image row's 48 left-border samples take
//!   the row's leftmost-original-column value and the 48 right-border
//!   samples take the rightmost-original-column value; each of the 48
//!   top- and bottom-border rows is then a row-wide copy of the
//!   topmost / bottommost horizontally-extended row, which makes the
//!   four 48×48 corner quadrants uniform at the corresponding
//!   corner-pixel value of the original image. The result lets
//!   [`inter::fetch_prediction_block`] and the §11.4 interpolation
//!   filters fetch any sample position within `±48` of the image
//!   boundary as if the original image's edge samples extended
//!   indefinitely — the well-defined "clamp" semantics a UMV fetch
//!   needs. Like §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3 this stage
//!   reads **no BoolCoder bits** — it is pure pixel arithmetic on an
//!   already-reconstructed frame buffer — so it advances the decoder
//!   past round 7 without touching the contested §7.3 `Split`
//!   formula.
//!
//! ## Round 10 surface
//!
//! * [`modes`] — the spec §10 macroblock coding-mode static surface.
//!   Surfaces the ten Table 4 coding modes ([`modes::CodingMode`])
//!   and three Table 5 ProbabilitySituation indices
//!   ([`modes::ModeAvailability`]); the 12-entry [`modes::NEAR_MACROBLOCKS`]
//!   neighbour-offset table; the verbatim
//!   [`modes::VP6_BASELINE_XMITTED_PROBS`] `[3][20]` I-frame
//!   `probXmitted` initialiser; the verbatim [`modes::VP6_MODE_VQ`]
//!   `[3][16][20]` baseline-bank that `SetNewBaselineProbs` /
//!   `WhichVector` select from; and the pure-integer
//!   [`modes::build_mode_decision_tree`] (and per-node
//!   [`modes::mode_decision_tree_node_probability`]) transform that
//!   converts a `probXmitted[3][20]` table into the
//!   `ModeDecisionTree[3][10][9]` array §10's `VP6_DecodeMode`
//!   traversal consults. The companion
//!   [`modes::probability_mode_same`] /
//!   [`modes::build_probability_mode_same`] compute the root-node
//!   "Same As Last" probability the Figure 10 traversal's first read
//!   needs. The `VP6_DecodeMode` BoolCoder traversal itself stays
//!   deferred until the §7.3 DOCS-GAP is resolved — but every piece
//!   of static data and every pure-integer derivation it would
//!   consult is now landed. Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5/§12.1/§14 this
//!   module reads no BoolCoder bits.
//!
//! ## Round 13 surface
//!
//! * [`zrl`] — the spec §13.3.3 AC zero-run-length static surface
//!   (the BoolCoder-independent half of zero-run decoding).
//!   Surfaces: the [`zrl::ZrlBand`] / [`zrl::ZrlNode`] Table 37 /
//!   Table 38 enums on the spec's canonical `0..=1` / `0..=13`
//!   indexing; the verbatim
//!   [`zrl::ZERO_RUN_PROB_DEFAULTS`] `[2][14]` keyframe initialiser
//!   for `ZeroRunProbs[2][14]`; the verbatim
//!   [`zrl::ZRL_UPDATE_PROBS`] `[2][14]` per-node
//!   `NewNodeProbFlag` update-flag probability bank (Table 41); the
//!   pure-integer [`zrl::zrl_bool_tree_to_huff_probs`] transform
//!   (the §13.3.3.2 `ZRLBoolTreeToHuffProbs` listing); and the
//!   [`zrl::build_zrl_huffman_tree`] composer that runs the
//!   conversion and then builds the §7.2 `2N - 1 = 17`-node
//!   `HuffNode` tree the §7.2 walker traverses against a raw-bit
//!   byte-stream reader. The §13.3.3.1 BoolCoder Figure 16
//!   traversal + six-bit extrabit reads stay deferred behind the
//!   §7.3 `Split` DOCS-GAP; the §13.3.3.2 Huffman path is fully
//!   landed at the static-surface level, with the run-length
//!   semantics for the 9th leaf (literal vs `> 8` escape)
//!   identified as a docs-gap candidate. Like §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2
//!   this module reads **no BoolCoder bits** — every operation is
//!   pure integer arithmetic over the supplied probability vector —
//!   so it advances the decoder past round 12 without touching the
//!   contested §7.3 `Split` formula.
//!
//! ## Round 12 surface
//!
//! * [`huffman`] — the spec §7.2 Huffman tree construction and
//!   traversal primitives (`HUFF_NODE`, `VP6_CreateHuffmanTree`,
//!   `VP6_HuffmanDecodeSymbol`). VP6 supports two entropy schemes (§7):
//!   the BoolCoder (§7.3) used in partition 1 for mode/MV decisions,
//!   and the Huffman coder (§7.2) used as an alternate DCT-token
//!   scheme when `UseHuffman` is set. The Huffman coder reads one
//!   whole raw bit per tree branch (`R(1)`; §3 nomenclature) rather
//!   than a sub-bit `B(prob)` BoolCoder bit, so it is **independent
//!   of the §7.3 `Split` formula DOCS-GAP**. Surfaces: [`huffman::HuffNode`]
//!   (the spec's `HUFF_NODE { Symbol, Prob, Left, Right }` with `-1`
//!   sentinels for internal-vs-leaf); [`huffman::create_huffman_tree`]
//!   (the verbatim §7.2.1 builder — `N-1` bottom-up merge rounds over
//!   a stable-sorted leaf list, root at index `2N-2`); [`huffman::decode_symbol`]
//!   (the verbatim §7.2 walk, parameterised over an external
//!   `FnMut() -> u8` raw-bit oracle so the byte-stream `R(1)` reader
//!   can land independently); plus [`huffman::tree_depth`] /
//!   [`huffman::codeword_for`] convenience helpers for inspecting
//!   the constructed tree. Like §15/§16/§17/§11/§12.1/§14/§10/§13
//!   this module reads **no BoolCoder bits** — every operation is
//!   pure integer arithmetic over the supplied probability vector —
//!   so it advances the decoder past round 11 without touching the
//!   contested §7.3 `Split` formula. The §13.3.3.2 AC zero-run
//!   probability conversion and the actual `R(1)` byte-stream reader
//!   stay deferred for later rounds.
//!
//! ## Round 9 surface
//!
//! * [`scan`] — the spec §12.1 default zig-zag scan order. Surfaces
//!   the 64-entry [`scan::DEFAULT_SCAN_ORDER`] table (Figure 14 /
//!   `default_dequant_table[64]`) that the decoder uses to convert
//!   tokens from zig-zag order back to raster order before §15
//!   inverse quantization and §16 inverse DCT, the inverse
//!   [`scan::DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG`] permutation for the
//!   encoder side, and the [`scan::zigzag_to_raster_block`] /
//!   [`scan::raster_to_zigzag_block`] block applicators. The §12.2
//!   per-frame *custom* scan-order updates are
//!   `ScanOrderUpdateFlag`-gated (BoolCoder-coded, see Table 17) and
//!   stay deferred until the §7.3 DOCS-GAP is resolved.
//! * [`dc_pred`] — the spec §14 DC coefficient prediction stage. A
//!   [`dc_pred::DcPredictionContext`] per plane holds the per-
//!   reference-frame "last decoded DC value" that §14 mandates, with
//!   `DcPredictionContext::new` / `reset_at_frame_start` applying the
//!   spec's per-frame zero seed ("At the beginning of each frame this
//!   last decoded DC value is set to zero for each prediction frame
//!   type"). [`dc_pred::DcPredictionContext::predict`] /
//!   [`dc_pred::DcPredictionContext::predict_and_record`] compute the
//!   §14 predictor for one block: the predictor table's four rows
//!   (neither neighbour → per-bucket last-DC seed; only left → L;
//!   only above → A; both → `(L + A + Sign(L+A)) / 2` truncated
//!   toward zero), with the same-reference-frame and intra-vs-inter
//!   neighbour-disqualification rules implemented inline.
//!   [`dc_pred::ReferenceBucket`] enumerates the three prediction
//!   frame types (Intra, InterLast, InterGolden) the spec
//!   distinguishes. Together with the §12.1 scan permutation this
//!   leaves the per-block DC reconstruction pipeline §15→§16→§17.1
//!   already-complete from prior rounds without depending on the §7.3
//!   `Split` formula: the `DcDelta` token whose value is added on top
//!   of the predictor is what the BoolCoder-gated §13.2 stage
//!   produces, but the predictor itself is BoolCoder-independent. Like
//!   §15/§16/§17.1/§11.4/§17.2–§17.4/§11.3/§11.5 this stage advances
//!   the decoder past round 8 without touching the contested §7.3
//!   `Split` formula.
//!
//! ## Round 14 surface
//!
//! * [`raw_bits`] — the spec §3 `R(x)` raw-bit byte-stream reader.
//!   Surfaces [`raw_bits::RawBitReader`] with `read_bit` / `read(n)`
//!   for the standard MSB-first `R(n)` convention used by §9
//!   Tables 1/2 (and consumed today by
//!   [`frame_header::Vp6FrameHeader::parse`]), plus an explicit
//!   [`raw_bits::RawBitReader::read_lsb_first`] variant for the one
//!   place the spec inverts that ordering: §13.3.3.1 (page 78), *"the
//!   run length minus nine is encoded using six-bits, least
//!   significant bit first."* Also surfaces
//!   [`raw_bits::RawBitReader::read_huffman_symbol`], a convenience
//!   that drives the §7.2 [`huffman::decode_symbol`] walk against the
//!   byte stream so callers using the §13 / §13.3.3.2 Huffman paths
//!   don't have to assemble the `R(1)` closure themselves. This is
//!   the byte-stream substrate the round-12 §7.2 Huffman walker and
//!   the round-13 §13.3.3.2 Huffman zero-run tree both consume; both
//!   landed parameterised over an `FnMut() -> u8` source precisely so
//!   the byte-stream reader could land independently. With this round
//!   the Huffman path of the §13 token decoder and the §13.3.3.2
//!   zero-run decoder both have a complete end-to-end data path
//!   (modulo the §13.3.3.2 9th-leaf semantics docs-gap noted in
//!   round 13's `zrl` report). Like
//!   §15/§16/§17/§11/§12.1/§14/§10/§13/§7.2 this module reads **no
//!   BoolCoder bits** — every operation is bit-level byte-stream
//!   arithmetic — so it advances the decoder past round 13 without
//!   touching the contested §7.3 `Split` formula.
//!
//! ## Round 15 surface — §7.3 BoolCoder primitive
//!
//! * [`bool_coder`] / [`BoolCoder`] — the spec §7.3 binary arithmetic
//!   decoder: `VP6_StartDecode` (the four-byte big-endian prefill of
//!   `Value`, `Range = 255`, `Count = 8`, `Pos = 4`), `VP6_DecodeBool`
//!   (the `Split = 1 + ( ((Range-1) * Probability) >> 7 )` per-bit
//!   step), and the renormalization loop that doubles `Range`/`Value`
//!   while `Range < 128` and pulls fresh bytes when `Count` reaches
//!   zero. Surfaces three call shapes for the §10/§11/§13 consumers:
//!   [`BoolCoder::decode_bool`] (single bit at arbitrary node
//!   probability; the §3 `B(x)` primitive and what the §10 mode-tree
//!   walk, §11 motion-vector decoder, and §13 DCT-token tree-walks
//!   will consume), [`BoolCoder::decode_b1`] (single fixed-probability-
//!   128 bit; the §3 `b(1)` primitive), and [`BoolCoder::decode_b`]
//!   (multi-bit fixed-probability-128 raw read; the §3 `b(n)`
//!   primitive accumulating MSB-first so the bit ordering matches
//!   §3 `R(n)`).
//!
//!   The earlier DOCS-GAP block in the round-1/2 commentary documented
//!   what looked like a self-contradiction in the §7.3 `Split` formula:
//!   at `Probability = 128, Range = 255` the formula evaluates to
//!   `Split = 255 = Range`, which makes the 0-branch unconditional and
//!   collapses every `b(n)` read to zero. The newly-staged
//!   `docs/video/vp6/vp6-errata-and-clarifications.md` entry **#35**
//!   resolves the gap by clean-room analysis: the `>> 7` (divide by
//!   128) is correct and intentional precisely because it makes
//!   probability 128 the half-interval point — exactly what the
//!   spec's `b(x)` notation requires. The `Split == Range` edge case
//!   at `Probability = 128, Range = 255` is statistically the
//!   half-interval boundary; it does collapse to the 0-branch when
//!   `Value < 0xFF00_0000`, which is the correct half-probability-128
//!   semantics for a `Value` whose top byte is below `0xFF`. The
//!   formula is bit-exact as printed; only the spec's evaluation
//!   order ("multiply → shift-by-7 → add-1", unsigned integer) needed
//!   pinning down.
//!
//!   The §7.3 BoolCoder primitive is what every remaining BoolCoder-
//!   coded layer in the codec (frame-header tail, §10 mode decoding,
//!   §11 motion-vector decoding, §13 DCT-token decoding, §13.3.3.1
//!   AC zero-run-length decoding) depends on; landing it unblocks
//!   all of those for later rounds. Like §15/§16/§17/§11.3-§11.5/
//!   §12.1/§14/§10/§13/§7.2, this module reads only the staged spec
//!   PDF and the staged clean-room errata clarifications. No
//!   third-party VP6 implementation was consulted.
//!
//! ## Round 16 surface — §13.2.1 arithmetic DC decoder
//!
//! * [`dct_decode`] — the first BoolCoder-consuming layer: the §13.2.1
//!   arithmetic DC decoder. Surfaces [`decode_dc_token`] (the Figure 15
//!   binary-tree walk down to a leaf [`DctToken`], DC variant — never
//!   returns EOB), [`decode_token_value`] (the magnitude-loop + sign
//!   read for the value-carrying tokens, driven by the
//!   errata-#67-corrected `magnitude_probs` slice with length
//!   `#ExtraBits − 1`), and [`decode_dc`] (the §13.2.1 full wrapper
//!   that combines them and returns the signed DC coefficient).
//!
//!   This is the round-15 BoolCoder primitive's first concrete
//!   consumer — it confirms the §7.3 errata-#35 disambiguation in
//!   end-to-end traversal against the §13 static surfaces landed in
//!   rounds 10–14. The §13.3.1 AC path adds an `EOB_CONTEXT_NODE`
//!   branch above the same binary tree (plus the "implicitly-1"
//!   first-decision shortcut when the previous AC coefficient was
//!   zero) and on a `ZERO_TOKEN` leaf transitions into the §13.3.3
//!   zero-run-length decoder; the tree-walk substrate that the AC
//!   path will share is the same [`decode_token_value`] /
//!   [`decode_dc_token`] kernel landed here, with the AC-specific
//!   branching and §13.3.3 zero-run integration deferred to a later
//!   round. The newly-staged errata #67 also corrects every prior
//!   round's `magnitude_probs` reading: the prior `extra_bit_probs`
//!   accessor stays available for callers that want the as-printed
//!   Table 18 columns, but the magnitude-loop traversal uses the new
//!   accessor.
//!
//! ## Round 17 surface — §13.3.1 arithmetic AC decoder
//!
//! * [`dct_decode::decode_ac_token`] / [`dct_decode::decode_ac_coefficient`]
//!   — the §13.3.1 per-coefficient AC arithmetic decoder, the second
//!   BoolCoder-consuming layer. The AC tree differs from the §13.2.1
//!   DC tree on two counts: it adds an `EOB_CONTEXT_NODE` branch
//!   above the `ZERO_CONTEXT_NODE` root so a left turn distinguishes
//!   end-of-block from a zero coefficient (the DC variant forbids
//!   EOB by spec), and it implements the §13.3.1 "implicitly-1"
//!   first-decision shortcut for the case
//!   `(EncodedCoeffs > 1) && (Prec == 0)` — when the previously-
//!   decoded AC coefficient in the current scan order was a
//!   ZERO_TOKEN, the next coefficient is structurally forbidden from
//!   being either `ZERO_TOKEN` or `EOB_TOKEN`, so the root decision
//!   is implicitly `1` and the walk starts at `ONE_CONTEXT_NODE`.
//!   [`AcOutcome`] surfaces the three-way per-step result the spec's
//!   pseudo-code distinguishes: `EndOfBlock` (exits the per-block
//!   loop), `ZeroRun` (hands off to §13.3.3's zero-run decoder), or
//!   `Value { coeff, next_prec }` (carries a signed AC coefficient
//!   and the §13.3.1 `Prec` update for the next position — `WasOne`
//!   for magnitude 1, `WasGreaterThanOne` otherwise). The signed
//!   reconstruction reuses the round-16 [`decode_token_value`]
//!   magnitude-loop + sign kernel; the magnitude-only probability
//!   slice (errata #67) and the §13.2.1/§13.3.1 sign read are
//!   identical between the two paths.
//!
//! * Companion static surface in [`tokens`]: [`AcBand`] (Table 30,
//!   six AC bands with `for_coefficient_position(usize) -> Option<AcBand>`
//!   that returns the §13.3.1 `AcProbBand[encodedCoeffs]` band index
//!   the walk reads), [`AcPlane`] (Table 28, Y / UV), and
//!   [`AcPrecContext`] (Table 29, `WasZero` / `WasOne` /
//!   `WasGreaterThanOne` plus a `seed_from_dc(dc: i32)` constructor
//!   for the very-first-AC case). These let callers select the
//!   correct row of [`AC_UPDATE_PROBS`] without hand-coded literal
//!   indices.
//!
//! What round 17 deliberately does **not** land: the §13.3.3.1
//! BoolCoder Figure 16 zero-run-length traversal itself (the
//! static surface is already in [`zrl`], but the per-bit
//! `B(prob)` walk + the six `(RunLength - 9)` extrabit reads are a
//! separate logical unit — `decode_ac_coefficient` surfaces a
//! `ZeroRun` outcome the caller routes into that decoder once
//! landed), the per-frame Table 31–35 AC probability update
//! bitstream, and the surrounding per-block driver that ties
//! coefficient positions to scan-order updates of `Prec` (the
//! `EncodedCoeffs ++` / `do { … } while (EncodedCoeffs < 64)`
//! envelope from §13.3.1). The first AC coefficient seeding of
//! `Prec` from the §13.2-decoded DC value is exposed via
//! [`AcPrecContext::seed_from_dc`] so a future driver round can wire
//! the DC → first-AC handoff without re-reading the spec.
//!
//! ## Round 18 surface — §11.5-derived edge-clamped MC fetch
//!
//! [`inter::fetch_prediction_block_clamped`] — the **edge-clamped**
//! form of the §17.2/§17.3 integer-offset prediction fetch. The §11.5
//! "Unrestricted Motion Vectors" buffer extension is defined as
//! "duplicating the edge values 48 times", which (as the
//! [`umv`] module-level documentation already records) is
//! mathematically equivalent to clamping the read position into the
//! original image's valid `[0, width)` x `[0, height)` rectangle.
//! This entry point implements the equivalence directly: it reads
//! straight out of the unbordered reference image, clamping each
//! per-sample `(row, col)` source position before the dereference,
//! and produces the same output the bordered
//! [`inter::fetch_prediction_block`] path would have produced for any
//! MV inside the §11.5 border — bit-identical on in-range reads,
//! identical edge-replicated values on the overhang reads — without
//! needing to allocate and fill the bordered buffer. Beyond the
//! 48-sample border the spec mandates, where the bordered fetch
//! would index out of bounds, the clamped fetch remains well-defined
//! and continues to serve up edge-replicated corner / edge pixels.
//!
//! What round 18 deliberately does **not** land: there is no spec
//! mandate that the decoder use one form or the other; the bordered
//! path remains available for callers that already build the §11.5
//! buffer (e.g. for the §11.4 fractional-pixel interpolation, which
//! reads two samples either side of the integer sample position and
//! is therefore naturally served by a pre-extended buffer). The
//! clamped fetch is a memory-efficient alternative for the integer
//! (§17.2 / §17.3) path that doesn't require a per-frame border
//! materialisation.
//!
//! ## Round 19 surface — §13.3.3.1 arithmetic AC zero-run-length decoder
//!
//! * [`decode_ac_zero_run`] — the §13.3.3.1 BoolCoder zero-run-length
//!   traversal of Figure 16. The immediate consumer of the round-17
//!   [`AcOutcome::ZeroRun`] hand-off: when the §13.3.1 AC decoder
//!   produces a `ZERO_TOKEN`, the spec mandates the run of trailing
//!   zero coefficients that follows is encoded under one of the two
//!   §7 entropy schemes; this routine implements the BoolCoder path.
//!   Walks the Figure 16 binary tree reading a `B(prob)` bit at each
//!   of the eight internal nodes (`>4`, `>2`, `>1`, `>3`, `>8`, `>6`,
//!   `>5`, `>7`) in the Table 38 / round-13 [`zrl::ZrlNode`] ordering;
//!   on the `>8` escape reads six `B(prob)` extrabits as the LSB-first
//!   encoding of `(RunLength - 9)` and reconstructs
//!   `RunLength = value + 9`. Returns the run length as `u32` in the
//!   spec's full output range `1..=72` (literal `1..=8` plus the
//!   `9..=72` escape interval).
//!
//!   This is the third BoolCoder-consuming layer (after the round-16
//!   §13.2.1 DC decoder and round-17 §13.3.1 AC decoder). Composes
//!   round-15 [`BoolCoder::decode_bool`] over the round-13
//!   [`zrl`] static surface ([`zrl::ZrlBand`] / [`zrl::ZrlNode`] /
//!   [`zrl::ZERO_RUN_PROB_DEFAULTS`] / [`zrl::ZRL_UPDATE_PROBS`]) — no
//!   new spec material, no new errata. The §13.3.3.1 listing on
//!   page 78 (the `if (!B(ZRP[0])) … else …` cascade plus the six
//!   `B(ZRP[8..=13])` extrabit reads) maps branch-for-branch onto
//!   the implementation.
//!
//! What round 19 deliberately does **not** land: the §13.3.3.2
//! Huffman zero-run path's actual walk against
//! [`zrl::build_zrl_huffman_tree`] (the static surfaces are landed
//! in rounds 12–14, but the integration with the literal-vs-escape
//! 9th-leaf docs-gap candidate is its own logical unit); the
//! §13.3.3 per-frame Table 39–41 `ZeroRunProbs` update bitstream
//! (`B(NewNodeProbFlag)` + conditional `b(7)` `NewNodeProbValue` —
//! same shape as §13.2 / §13.3); and the surrounding per-block
//! driver loop that ties [`AcOutcome::ZeroRun`] to
//! [`decode_ac_zero_run`] and threads the run length back into
//! `EncodedCoeffs += ZeroRunCount` plus the §12.1 scan-order
//! traversal.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

pub mod bool_coder;
pub mod dc_pred;
pub mod dct_decode;
pub mod dequant;
pub mod frame_header;
pub mod huffman;
pub mod idct;
pub mod inter;
pub mod interp;
pub mod loopfilter;
pub mod modes;
pub mod raw_bits;
pub mod reconstruct;
pub mod scan;
pub mod tokens;
pub mod umv;
pub mod zrl;

pub use bool_coder::BoolCoder;
pub use dc_pred::{
    average_both_neighbours, sign as dc_sign, DcPredictionContext, Neighbour, ReferenceBucket,
};
pub use dct_decode::{
    decode_ac_coefficient, decode_ac_token, decode_ac_zero_run, decode_dc, decode_dc_token,
    decode_token_value, AcOutcome,
};
pub use dequant::{DequantContext, AC_QUANTIZATION_TABLE, DC_QUANTIZATION_TABLE};
pub use frame_header::{CodingProfile, FrameType, Vp3Version, Vp6FrameHeader};
pub use huffman::{
    codeword_for, create_huffman_tree, decode_symbol, tree_depth, HuffNode, HuffmanError,
    INTERNAL_SYMBOL, NO_CHILD,
};
pub use idct::idct_block;
pub use inter::{
    chroma_frac, fetch_prediction_block, fetch_prediction_block_clamped, inter_block_to_pixels,
    luma_frac, reconstruct_inter_block, whole_sample_aligned, MvShift,
};
pub use interp::{
    bicubic_block, bicubic_point, bilinear_block, bilinear_point, var_16_point, BICUBIC_FILTER_SET,
    BICUBIC_VP61_INDEX, BILINEAR_CHROMA_FILTERS, BILINEAR_LUMA_FILTERS,
};
pub use loopfilter::{
    bound, boundary_x, boundary_y, filter_horizontal_boundary, filter_vertical_boundary,
    prediction_loop_filter_function, PREDICTION_LOOP_FILTER_LIMIT_VALUES,
};
pub use modes::{
    build_mode_decision_tree, build_probability_mode_same, mode_decision_tree_node_probability,
    probability_mode_same, CodingMode, ModeAvailability, ModeDecisionTree, ModeDecisionTreeRow,
    NEAR_MACROBLOCKS, NUM_CODING_MODES, NUM_MODE_DECISION_NODES, NUM_MODE_VQ_VECTORS,
    NUM_PROBABILITY_SITUATIONS, PROB_XMITTED_ROW_LEN, VP6_BASELINE_XMITTED_PROBS, VP6_MODE_VQ,
};
pub use raw_bits::{RawBitError, RawBitReader};
pub use reconstruct::{intra_block_to_pixels, reconstruct_intra_block};
pub use scan::{
    raster_to_zigzag_block, zigzag_to_raster_block, DEFAULT_SCAN_ORDER,
    DEFAULT_SCAN_ORDER_RASTER_TO_ZIGZAG,
};
pub use tokens::{
    baseline_ac_probs, baseline_dc_probs, dc_probs_to_node_contexts,
    dct_token_bool_tree_to_huff_probs, AcBand, AcPlane, AcPrecContext, DctToken, TreeNode,
    AC_UPDATE_PROBS, DC_NODE_EQS, NUM_AC_BANDS, NUM_AC_PREC_CONTEXTS, NUM_DCT_TOKENS,
    NUM_DC_CONTEXTS, NUM_DC_NODE_EQS, NUM_PLANES, NUM_TREE_NODES, VP6_DC_UPDATE_PROBS,
};
pub use umv::{
    build_extended_buffer, extend_border, extended_height, extended_stride, origin_offset,
    UMV_BORDER_SIZE,
};
pub use zrl::{
    build_zrl_huffman_tree, zrl_bool_tree_to_huff_probs, ZrlBand, ZrlNode, NUM_ZRL_BANDS,
    NUM_ZRL_HUFF_PROBS, NUM_ZRL_NODES, ZERO_RUN_PROB_DEFAULTS, ZRL_UPDATE_PROBS,
};

/// Crate-local error type.
///
/// Currently narrow because the round-1 surface only does one thing
/// (parse the raw-bit frame-header prefix); future rounds will expand
/// this as the decoder/encoder come online.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input ran out before all required syntactic elements were
    /// consumed.
    Truncated,
    /// The requested code path isn't wired up in this round.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Truncated => f.write_str("oxideav-vp6: truncated VP6 bitstream"),
            Error::NotImplemented => {
                f.write_str("oxideav-vp6: requested code path not implemented in this round")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Codec registration entry-point.
///
/// The round-1 scaffold doesn't register a `Decoder` or `Encoder` —
/// the full decoder isn't operational yet. Once the BoolCoder
/// DOCS-GAP is closed and the remainder of the frame header parses,
/// the decoder shell will be registered here.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vp6", register);

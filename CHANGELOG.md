# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.8](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.7...v0.0.8) - 2026-05-17

### Other

- vp6 r73: SATD-based qpel ME refinement (encoder)

### Added

- **r73 — SATD-based qpel ME refinement (encoder).** Quarter-pel diamond
  motion estimation in `motion_search` and `motion_search_8x8` now scores
  candidate sub-pel offsets via SATD (Sum of Absolute Transformed
  Differences using a 4×4 Hadamard kernel per tile) by default, instead
  of plain SAD. SATD better predicts the post-DCT bit cost of the
  residual because it captures frequency-domain energy distribution: a
  residual whose pixel-domain SAD is low but whose AC bins are strongly
  excited (e.g. a smooth shift mispredicted by one qpel step) gets a
  higher SATD, pushing the diamond toward the candidate whose residual
  is genuinely sparse in the transform domain. Integer-pel search stays
  SAD-based (the integer-pel surface is multimodal and SAD's full-window
  scan is cheap); SATD's value is concentrated in the qpel refinement
  where the SAD surface is nearly flat. New public field
  `Vp6Encoder::allow_satd_me: bool` (default `true`) gates the path;
  setting to `false` recovers pre-r73 SAD-only diamond behaviour. Lambda
  is scaled by `SATD_LAMBDA_SCALE = 4` so the cost ratio between
  distortion and MV-bit rate stays the same under SATD as under SAD.
  Wire format unchanged — only the chosen sub-pel MV may differ. Wired
  into `encode_inter_frame`, `encode_inter_frame_with_golden`, and
  `encode_inter_frame_huffman` via the shared ME helpers. New tests:
  * `r73_allow_satd_me_default_and_disable` — pins the public field
    default + disable contract.
  * `r73_satd_qpel_internal_psnr_clears_45db_on_flat` — flat-content
    skip-path still recovers near-losslessly under SATD.
  * `r73_satd_qpel_no_regression_on_r25_stripes_fixture` — clears the
    r25 35 dB floor on the translating-stripes fixture.
  * `r73_satd_qpel_improves_or_matches_psnr_on_textured_motion` —
    pins SATD-on within 0.05 dB of SAD-off on a textured-motion fixture,
    confirming the metric swap doesn't hurt at fixture scale.
  * `r73_satd_disable_decodes_cleanly_on_smooth_motion` — verifies the
    SAD-only path is still reachable and produces decodable output.
  Plus 6 unit tests pinning the Hadamard kernel (`r73_hadamard4x4_satd_*`
  + `r73_satd8x8_qpel_zero_on_self` + `r73_satd16x16_qpel_zero_on_flat_match`).

## [0.0.7](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.6...v0.0.7) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- vp6 r39: PID controller + diamond qpel ME + trellis-style AC quantisation
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- rustfmt pass on tests/encoder_roundtrip.rs (r31 new tests)
- rustfmt pass after r31 (line-length wraps in encoder.rs)
- vp6 r31: scene-change golden refresh + Huffman inter encode + bool/Huffman RDO
- r30 — PI controller + DCT-count intra cost + golden-aware intra-in-inter (encoder)
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-vp6/pull/502))

### Added

- **r39 — PID controller + iterative diamond qpel ME + trellis-style AC
  quantisation (encoder).** Three encoder-only refinements:

  1. **PID controller.** `BitrateControl` grows `kd: f32` (default `0.15`)
     + `prev_error: f32` fields. `update_qp_after_frame` adds a
     derivative-on-error term: `qp_delta = round((kp * err + ki *
     integral + kd * derivative) * 8)`. The derivative is taken against
     the EMA-smoothed error so single noisy frames don't kick the
     controller into a spurious correction. Reduces overshoot during
     bitrate transients vs PI-only. Setting `kd = 0.0` recovers pre-r39
     PI behaviour exactly. Pinned by `r39_pid_kd_zero_matches_pi_exactly`
     (byte-for-byte QP path equivalence with PI when kd=0) and
     `r39_pid_controller_reduces_overshoot_vs_pi_only`.
  2. **Iterative diamond qpel ME.** `motion_search` and
     `motion_search_8x8` swap their pre-r39 ±3 qpel exhaustive box
     (49-position search) for an iterative 8-conn diamond pattern with
     up to 6 iterations and ±6 qpel bounds from the integer winner.
     Each iteration evaluates the 8-conn neighbours of the current best;
     if a neighbour strictly improves the Lagrangian cost it becomes the
     new centre. Probe budget per MB ≤ 8 × 6 = 48, comparable to the
     pre-r39 box, but the effective catch radius is doubled (6 qpel vs
     3 qpel). Stops early when no neighbour beats the current best —
     typically 2-3 iterations on smooth motion content. Pinned by
     `r39_diamond_qpel_me_internal_psnr_clears_45db` (flat skip-path)
     and `r39_diamond_qpel_me_no_regression_on_r25_stripes_fixture`
     (existing r25 stripes fixture clears the 35 dB floor at 35.20 dB).
  3. **Trellis-style AC quantisation.** New public field
     `Vp6Encoder::allow_trellis: bool` (default `true`) gates a per-block
     per-coef RD pass on the inter-frame residual AC stream. For each
     non-zero level, considers driving the level toward zero by 1 LSB;
     chooses the drop when the per-coef rate saving (bool-tree depth
     proxy) outweighs the squared-quantisation-error increase scaled by
     a QP-derived λ. Bit-exact byte-output-equivalent to plain
     `div_nearest` quantise when no win is found. Wire format unchanged
     — bool-decoder reads identical state machine. Wired into
     `encode_inter_frame`, `encode_inter_frame_with_golden`, and the
     Huffman inter path. Conservative sweet spot: levels of `±1` get
     pruned only when the raw coef sat just over the half-step
     threshold, so quality drops are bounded. Pinned by
     `r39_trellis_shrinks_bitstream_at_minimal_psnr_loss` (≤ size, ≤
     0.5 dB Y PSNR drop on 64×64 natural-content fixture; observed 1
     byte saved at 0.02 dB drop).

- **r31 — Scene-change golden refresh + Huffman inter encode + bool/Huffman RDO
  (encoder).** Three encoder-only improvements:

  1. **Scene-change-driven golden refresh.** `Vp6Encoder` gains two new
     public fields: `scene_change_threshold: f32` (default `2.0`) and
     internal `sad_ema: f32`. `encode_inter_frame_with_golden` now computes
     the per-pixel normalised luma SAD of the new frame against the previous
     frame. The SAD is compared against an EMA-smoothed running mean (α=0.1).
     When `frame_sad_pp > scene_change_threshold × sad_ema`, a scene cut is
     declared and the `golden_frame_flag` bit is set regardless of the
     cadence counter — giving the decoder a fresh reference immediately on
     scene cuts. The EMA is seeded on the first inter frame (with a minimum
     floor of 1.0 so static-first-frame content doesn't permanently suppress
     the detector). Setting `scene_change_threshold = 0.0` disables
     scene-change detection entirely (cadence-only behaviour). New tests:
     * `r31_scene_change_triggers_golden_refresh` — 3-frame sequence
       (stripes key → stripes inter → checkerboard inter) asserts the
       cadence counter resets to 1 on the scene-cut frame.
     * `r31_scene_change_detection_disabled_at_threshold_zero` — same
       sequence with threshold=0 asserts the counter reaches 2 (no refresh).

  2. **Huffman inter encode.** New encoder method
     `Vp6Encoder::encode_inter_frame_huffman(prev_*, new_*, ...)` emits a
     P-frame whose coefficient partition (partition 2) is Huffman-coded
     (`UseHuffman = 1`) while partition 1 (mode info + MVs) remains
     bool-coded. The implementation mirrors `encode_inter_frame` for the ME
     + mode-decision + MV-emission pass; coefficients are gathered into a
     per-block table (same quantise/DC-predictor path) then emitted via
     `encode_block_huffman` using trees built from the post-keyframe `0x80`
     coefficient model baseline. On a 32×32 shifting-stripes fixture (QP 16,
     search 8) the Huffman inter is 89 B vs 85 B for the bool path (1.05×)
     — the tree overhead dominates on small frames; larger frames benefit
     more. Our decoder and ffmpeg's vp6f decoder both consume the output
     cleanly (tested by `r31_huffman_inter_roundtrip_own_decoder` and
     `r31_ffmpeg_decodes_huffman_inter_frame`).

  3. **Bool/Huffman RDO for inter frames.** New encoder method
     `Vp6Encoder::encode_inter_frame_rdo(prev_*, new_*, ...)` runs both
     `encode_inter_frame` (bool) and `encode_inter_frame_huffman` (Huffman)
     and returns whichever output is smaller. Since both paths encode the
     same DCT levels (same distortion), this is a pure size-based selection
     with no PSNR cost. The RDO output is guaranteed to be ≤ the bool-only
     output (verified by `r31_rdo_inter_not_larger_than_bool_inter`).
     Round-trip PSNR ≥ 32 dB via our decoder on the shifting-stripes
     fixture (`r31_rdo_inter_roundtrip_own_decoder`). Note: this runs the
     full ME + quantise pass twice, so it is ~2× slower than either single
     path; suitable for offline / high-quality encodes.

- **r30 — PI controller + DCT-count intra cost + golden-aware
  intra-in-inter (encoder).** Three encoder-only refinements layered on
  r29's bitrate control + intra-in-inter RDO:

  1. **PI controller (was P-only).** `BitrateControl` grows three
     fields: `ki: f32` (default `0.05`), `integral: f32` (internal
     state, accumulator), and `integral_clamp: f32` (default `5.0`,
     anti-windup bound). `update_qp_after_frame` now computes
     `qp_delta = round((kp * error_ratio + ki * integral) * 8)` after
     accumulating `integral := clamp(integral + error_ratio,
     ±integral_clamp)`. Two anti-windup mechanisms:
     * The accumulator is hard-clamped to `[-integral_clamp,
       +integral_clamp]` so a long stretch of saturated-QP operation
       can't bank an arbitrarily large correction.
     * **Saturated-actuator back-leak**: when the new `qp` lands at
       `qp_max` AND the integral is still pushing further up (or
       symmetrically `qp_min` and pushing down), the most recent
       integral accumulation is undone — the integral can't grow during
       stretches where the actuator is pinned.

     Setting `ki = 0.0` recovers pre-r30 P-only behaviour exactly. The
     integral term eliminates the steady-state bitrate offset a P-only
     controller leaves on content whose intrinsic complexity needs a
     QP set-point different from the seed. On a noisy 32x32 fixture
     (target 1067 bytes/frame, seed QP 20), the PI controller pushes
     QP from 20 → 44 in 6 frames vs the P-only path's 20 → 36 — ~22%
     faster convergence. New tests:
     * `r30_pi_controller_ki_zero_matches_p_only` — PI with `ki=0`
       keeps QP trajectory and frame-byte size identical to the
       default-PI path while QPs match.
     * `r30_pi_controller_integral_accumulates_steady_state` — over
       6 over-target frames the integral accumulates positively and
       respects the clamp.
     * `r30_pi_controller_antiwindup_caps_integral` — pin
       `qp_min = qp_max`, run 20 frames, integral never exceeds clamp.

  2. **DCT-count intra cost.** New helper `mb_intra_dct_count_proxy`
     forward-DCTs each of the 4 luma 8×8 blocks of the MB at the
     current `qp`, quantises by `dequant_ac`, and counts surviving
     non-zero AC coefficients. The intra-vs-inter cost in
     `encode_inter_frame` (and now in `encode_inter_frame_with_golden`)
     becomes `Σ |pixel - mean| + λ * (DCT-survivor-count * 4 + 6)` —
     the SAD-against-mean predictability proxy plus a per-token bit
     budget term (~4 bool-coded bits per surviving AC token). Closer to
     actual encode cost than SAD-against-mean alone — high-frequency
     MBs no longer slip through the cheap-intra cracks; flat-but-noisy
     MBs no longer get mis-classified as expensive-intra. Verified by
     `r30_dct_count_intra_cost_no_regression_on_smooth_motion`: on
     well-MC-compensated horizontal-shift content the intra-on encode
     is byte-identical to the intra-off baseline (RDO correctly rejects
     intra everywhere).

  3. **Golden-aware intra-in-inter.** `encode_inter_frame_with_golden`
     now also evaluates `Vp56Mb::Intra` as a per-MB candidate (it
     previously had no intra branch at all — the r29 work only landed
     in the single-ref `encode_inter_frame`). "Golden-aware" means the
     intra-cost has to beat the BEST inter (golden vs prev) — intra
     fires only when both refs are unrelated to the new content. The
     per-MB residual encoding becomes a 3-way split (Intra /
     Inter-Prev / Inter-Golden) sharing the same emission shape;
     ref-kind drives the `RefKind`-matched DC-predictor neighbour
     check (Intra → `Current`, Inter-Prev → `Previous`, Inter-Golden
     → `Golden`). Vector-candidate position state for the chosen ref
     is only advanced when an inter mode is picked — Intra MBs don't
     consume an MV-candidate slot. On a scene-change against
     matching-content refs (both prev + golden = vertical stripes,
     new = checkerboard, QP 16), the with-intra encode is 235 B vs
     306 B for intra-off — ~23% smaller, 30.5 dB Y PSNR via our own
     decoder. ffmpeg's vp6f decoder cross-decodes the resulting
     bitstream cleanly. New tests:
     * `r30_golden_aware_intra_in_inter_fires_on_scene_change` —
       wire-size + Y PSNR sanity vs intra-off baseline.
     * `r30_golden_aware_intra_byte_identical_on_smooth_motion` —
       smooth-motion content stays byte-identical (RDO rejects intra
       on every MB).
     * `r30_ffmpeg_decodes_golden_aware_intra_in_inter` — opt-in
       ffmpeg vp6f cross-decode of the key + golden-aware-with-intra
       inter pair.

## [0.0.6](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.5...v0.0.6) - 2026-05-05

### Other

- r29 — bitrate control + Intra-in-inter RDO (encoder)

## [0.0.5](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- commit module + roundtrip tests previously left untracked
- vp6 r28: Huffman coefficient path (encoder + decoder)
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- r27 — INTER_FOURMV (per-8×8 motion vectors, encoder)
- r26 — golden-frame refresh + per-MB golden-vs-previous selection (encoder)
- r25 — quarter-pel sub-pel motion estimation (encoder)
- port to DoS-limits framework (true zero-copy arena receive)
- r24 — inter residual coefficient encoding
- r23 — Vp3VersionNo spec fix unblocks ffmpeg inter interop
- r22 — vector_predictors ctx mapping spec fix + coeff shortcut audit
- r21 — DEF_MB_TYPES_STATS pair-order spec fix
- r20 — Buff2Offset spec compliance + ffmpeg interop scaffolding
- round 19 — partial inter-frame audit (ffmpeg still rejects)
- round 18 — diagnostic dump for inter-frame ffmpeg interop
- round 17 — encoder MV emission + integer-pel ME
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

### Added

- **r29 — bitrate control + Intra-in-inter RDO (encoder).**
  Two new encoder behaviours, both opt-in / off-by-default-impact-only:

  1. **Bitrate-targeting feedback loop.** New public type
     `oxideav_vp6::encoder::BitrateControl` and field
     `Vp6Encoder::bitrate: Option<BitrateControl>`. Convenience method
     `Vp6Encoder::set_bitrate_target(bps, fps)` derives a
     `target_bytes_per_frame = ceil(bps / (fps * 8))`. After each
     `encode_*` call, callers invoke
     `Vp6Encoder::update_qp_after_frame(bytes_emitted)` and the
     controller adapts `qp` for the next frame using a proportional-
     with-EMA shape:
     * `ema_bytes := alpha * bytes_emitted + (1 - alpha) * ema_bytes`
       (default `alpha = 0.3` for moderate smoothing);
     * `error_ratio := (ema_bytes - target) / target`;
     * `qp_delta := round(kp * error_ratio * 8)` (default `kp = 0.5`);
     * `qp := clamp(qp + qp_delta, qp_min, qp_max)` (defaults `[4, 60]`).
     The wire format is unchanged — only byte-0 of each subsequent frame
     reflects the new QP. The controller is a no-op when
     `bitrate.is_none()` (preserves pre-r29 fixed-QP behaviour for
     callers that don't opt in).
     New tests: `r29_bitrate_control_tracks_target` (overshoot →
     controller pushes QP from 4 → 60),
     `r29_bitrate_control_lowers_qp_when_undertarget` (undershoot →
     controller pushes QP from 50 → 22),
     `r29_bitrate_control_inactive_when_no_target`,
     `r29_bitrate_control_target_zero_clears`,
     `r29_bitrate_control_field_defaults`.

  2. **Intra-in-inter RDO.** New public field
     `Vp6Encoder::allow_intra_in_inter: bool` (default `true`).
     `encode_inter_frame` now considers `Vp56Mb::Intra` as a per-MB
     candidate alongside `InterNoVecPf` / `InterDeltaPf` / `Inter4V`,
     gated by a Lagrangian RDO comparison. The intra-cost proxy is
     `Σ |pixel - mean|` (cheap monotonic-in-difficulty signal) plus
     a baseline 6-bit PMBT-tree depth charge. Intra fires on revealed-
     content / scene-change MBs where the inter SAD dramatically
     exceeds the intra signal.

     Wire emission piggy-backs on the existing `encode_pmbt_tree`
     walk — `Vp56Mb::Intra` is a reachable leaf from every prev_type
     in the spec's PMBT tree (`tables::PMBT_TREE`). Per-block residual
     encoding diverges by mode: Intra MBs use `forward_dct8x8` (with
     -128 bias, like the keyframe path) and the `RefKind::Current` DC
     predictor (mirroring `add_predictors_dc(scratch, RefKind::Current)`
     gated by `tables::REFERENCE_FRAME[Intra] = Current`); inter MBs
     keep the residual-mode DCT + `RefKind::Previous` predictor from
     r24. The per-block DC neighbour state (`enc_left_block`,
     `enc_above_blocks`, `enc_prev_dc[plane][ref_idx]`) is updated
     with the chosen MB's `ref_frame` so subsequent MBs see the
     matching predictor regardless of mode mix.

     Setting `allow_intra_in_inter = false` reproduces pre-r29
     behaviour. On a scene-change fixture (keyframe stripes →
     inter checkerboard) the intra-on encode is 235 bytes vs 306 bytes
     for intra-off — a 23% reduction with cleaner reconstruction
     (30.5 dB Y PSNR vs MC-only). On smooth-motion content the two
     paths emit byte-identical wire streams (RDO never picks intra).
     New tests: `r29_intra_in_inter_fires_on_scene_change` (≤ 105% of
     intra-off baseline + decoder reconstruction sanity),
     `r29_intra_in_inter_byte_identical_on_smooth_motion` (≤ +4 byte
     wobble vs intra-off baseline on smooth-motion content).

- **r28 — Huffman coefficient path (encoder + decoder).**
  Implements the optional VP6 second-data-partition Huffman coding
  scheme described in spec sections 7.2 (Huffman Decoder), 13.1 (DCT
  Token Huffman Tree), 13.2.2 (Huffman Decoding DC Values), 13.3.2
  (Decoding Huffman AC Coefficients), 13.3.3.2 (Decoding Huffman AC
  Zero Runs), and 13.4 (Decoding Huffman EOB and DC 0 Runs). New
  module `oxideav_vp6::huffman` exports:
  * `BitReader` / `BitWriter` — MSB-first raw bit IO matching the
    spec's `R(n)` operator in the Huffman path.
  * `HuffTree` — tree built per `VP6_CreateHuffmanTree` (spec page
    14) with pre-computed `(codeword, length)` per symbol so encode
    is O(1) per token. Decode walks the tree with `read_bit()`.
  * `dct_token_bool_tree_to_huff_probs` /
    `zrl_bool_tree_to_huff_probs` — the spec's
    `DCTTokenBoolTreeToHuffProbs` / `ZRLBoolTreeToHuffProbs` exact
    ports.
  * `HuffmanTreeSet` — DC trees [Y, UV], AC trees [prec][plane][band],
    ZRL trees [band] — built from current model state.
  * `encode_block_huffman` / `decode_block_huffman` — per-block
    coefficient emission + parse with the cross-block DC-zero and
    AC1-EOB run state in `HuffmanFrameState`.
  * `encode_eob_run` / `decode_eob_run` — spec page 81 raw-bit run
    encoding (`1 + R(2)` … `11 + R(6)`).
  Encoder side: new `Vp6Encoder::encode_keyframe_huffman(...)` method
  emits a keyframe with `MultiStream = 1` + `UseHuffman = 1`, header
  + bool-coded partition 1 (model updates) + Huffman partition 2.
  Decoder side: the previous `Error::Unsupported` for `use_huffman ==
  1` is replaced; the decoder builds `HuffmanTreeSet` after the
  bool-coded picture-header model updates, then routes per-MB
  coefficient decode through a new `parse_coeff_huffman` helper that
  mirrors `mb::parse_coeff` but reads Huffman tokens instead. New
  test file `tests/huffman_roundtrip.rs` exercises:
  * `huffman_keyframe_wire_markers` — pins MultiStream / Vp3VersionNo
    / Buff2Offset header bytes.
  * `huffman_keyframe_flat_gray_roundtrip` — flat Y=128 keyframe
    roundtrip via in-tree decoder.
  * `huffman_keyframe_constant_color_roundtrip` — non-128 flat color
    exercises non-zero DC tokens.
  * `huffman_keyframe_gradient_roundtrip` — vertical gradient
    exercises every Huffman token category.
  * `ffmpeg_decodes_huffman_keyframe` — opt-in ffmpeg vp6f
    cross-decode of the Huffman keyframe.
  10 huffman-module unit tests pin tree construction, bit-IO
  roundtrip, EOB-run / zero-run roundtrip, and per-block encode +
  decode roundtrips for empty / DC-only / general / cross-block-run
  blocks.

- **r27 — INTER_FOURMV (per-8×8 motion vectors, encoder).**
  `Vp6Encoder::encode_inter_frame` now considers `Vp56Mb::Inter4V`
  (the spec's "FOURMV" mb_type) as a candidate alongside the single-MV
  `InterDeltaPf` / `InterNoVecPf` path. Per-MB pipeline:
  1. Whole-MB qpel motion search (existing r25 behaviour) → single MV.
  2. **NEW**: per-8×8 luma motion search (`motion_search_8x8`) seeded
     from the whole-MB integer winner with a tight `±2` pel window plus
     a `±3 qpel` refine — same shape as the whole-MB qpel refine, but
     scoped to the block.
  3. RDO on `cost = SAD + λ * bits`. SAD is summed across the 4 luma
     8×8 blocks (single-MV reuses one MV everywhere; FOURMV uses each
     block's own MV). `bits` is a coarse proxy: single-MV ≈ MV-delta
     cost for one component pair; FOURMV ≈ 8 raw type bits + 4 ×
     (MV-delta cost) for non-zero blocks. FOURMV fires only when (a)
     per-block MVs diverge from the whole-MB MV by ≥ 2 qpel in either
     component AND (b) the FOURMV cost strictly beats the single-MV
     cost.
  4. Wire emission mirrors `decode_4mv` in `decoder.rs`: 4 × 2 raw
     bits for block-type tags (raw 0 = NoVec, raw 1 = Delta — never
     candidate-cycle types 2/3 since the encoder doesn't pre-mirror
     per-block candidate state), THEN per-block deltas in order.
     Critical: tags emitted before deltas, otherwise the decoder
     desynchronises (per-block deltas reach `parse_vector_adjustment`
     mid-tag-read).
  5. Per-block residual encoding picks up the per-block MV automatically
     via `block_mv_full[b]` threaded into `sample_mc_tile`. Chroma MVs
     are derived as the round-shifted average of the 4 luma MVs (mirror
     of the decoder's `RSHIFT(sum, 2)` chroma derive).
  6. The MV stored back into `mb_info[]` for downstream MV-candidate
     lookup is `block_mvs[3]` (decoder's `decode_mv` Inter4V branch
     uses `scratch.mv[3]` for the same purpose).

  Public API surface: new field `Vp6Encoder::allow_fourmv: bool`
  (default `true`). Setting to `false` reproduces pre-r27 single-MV
  behaviour for A/B testing and regression pinning. Honoured by
  `encode_inter_frame` only — the golden-aware path
  (`encode_inter_frame_with_golden`) stays single-MV regardless because
  `tables::REFERENCE_FRAME[Inter4V] = Previous`, so FOURMV has no role
  on a golden-ref MB.

  On a 32×32 fixture where each 8×8 quadrant of every MB has a
  distinct optimal motion (`build_diverging_block_motion`), the FOURMV
  encode is 292 bytes vs 430 bytes for the single-MV encode — a 32%
  reduction. ffmpeg's vp6f decoder accepts the FOURMV bitstream
  cleanly. The pre-r27 `inter_frame_horizontal_shift_uses_mv` fixture
  (no per-block divergence) records the same PSNR as before; the
  single-MV path is unchanged on smooth-motion content.

- `tests/encoder_roundtrip.rs::r27_fourmv_inter_smaller_than_single_mv_on_diverging_blocks`:
  pins the wire-size win at ≤ 95% of the single-MV baseline on the
  diverging-blocks fixture (we observe ~68% in practice). Also pins
  Y PSNR through our own decoder ≥ 20 dB and the fact that FOURMV-on
  vs FOURMV-off keyframe encodes are byte-for-byte identical (only the
  inter payload moves).
- `tests/encoder_roundtrip.rs::r27_ffmpeg_decodes_fourmv_inter_frame`:
  pins ffmpeg cross-decode of the FOURMV bitstream layer. Mirrors
  `r25_ffmpeg_decodes_qpel_inter_frame` shape (mux key + inter into a
  2-tag FLV stream, run ffmpeg's vp6f decoder, assert it produces 2
  frames of YUV).

- **r26 — golden-frame refresh (encoder).**
  New `Vp6Encoder::encode_inter_frame_with_golden(prev_*, golden_*,
  new_*, w, h, search)` accepts both a previous-frame reference and a
  golden-frame reference. Two added behaviours over
  `encode_inter_frame`:
  1. **Cadence-driven golden refresh.** Public field
     `golden_refresh_period: u32` (default 30) drives a counter
     `inter_frames_since_golden` (reset to 0 by `encode_keyframe`,
     incremented by every `encode_inter_frame*` call). When
     `should_refresh_golden()` fires (`counter >= period`), the
     picture-header `golden_frame_flag` bit is set to 1 — the decoder
     snaps the just-decoded reconstruction into its `golden_frame`
     slot (`decoder.rs:422`). The counter resets to 1 on a refresh
     frame (matching the keyframe path's "next inter is 1 since
     golden" semantics).
  2. **Per-MB golden-vs-previous selection.** Per MB the encoder runs
     `motion_search` against both `prev_*` and `golden_*`, then picks
     the lower Lagrangian cost (SAD + λ * mv_bits). The MB type maps
     to one of `{InterNoVecPf, InterDeltaPf, InterNoVecGf,
     InterDeltaGf}` accordingly. Golden-ref MBs use `RefKind::Golden`
     for their DC predictor state (mirroring
     `mb::add_predictors_dc(scratch, RefKind::Golden)`) and contribute
     to the golden-ref MV-candidate pool the decoder walks for
     subsequent golden-ref MBs (separate `vector_candidate_pos_gf`
     state).
  Spec refs: page 28 (picture-header layout, golden_frame_flag bit),
  `tables.rs::REFERENCE_FRAME` (`InterNoVecGf`, `InterDeltaGf`, …
  → `RefFrame::Golden`).
  On the new `golden_refresh_loop_back_uses_golden_reference` fixture
  (A→B→A loop), our decoder reconstructs frame 2 at 45 dB Y PSNR via
  the golden ref vs an 8.6 dB skip-from-prev baseline (a 36 dB win).
  On `golden_refresh_reduces_bytes_on_periodic_loop` (5-frame
  A,B,A,B,A loop, QP 12), pinning golden to the keyframe brings the
  total inter-frame wire size from 378 bytes to 282 bytes (~25%
  smaller) vs refreshing every frame. ffmpeg's vp6f decoder accepts
  the key + golden-refresh inter pair cleanly
  (`ffmpeg_decodes_inter_with_golden_refresh_flag`).
- `tests/encoder_roundtrip.rs::golden_refresh_cadence_fires_on_period`,
  `golden_refresh_disabled_at_period_zero`,
  `golden_refresh_loop_back_uses_golden_reference`,
  `golden_refresh_reduces_bytes_on_periodic_loop`, and
  `ffmpeg_decodes_inter_with_golden_refresh_flag`. Pin the cadence
  semantics, the `period = 0` disabled branch, the golden-ref decode
  path, the bitrate-reduction property on periodic content, and
  ffmpeg cross-decode of the refresh-flag picture-header layout.

- **r25 — quarter-pel sub-pel motion estimation (encoder).**
  `Vp6Encoder::encode_inter_frame` now picks quarter-pel-accurate MVs.
  `motion_search` runs the existing integer-pel SAD search to seed
  `(int_dx, int_dy)`, then evaluates every qpel offset in a `±3 qpel`
  window around the integer winner via the H.264-chroma-style bilinear
  filter the decoder uses (`mb::render_mb_inter` `use_bicubic_luma ==
  false` branch). Each qpel candidate's cost is `SAD(MC) + λ *
  mv_bits` with `λ` proportional to QP — so sub-pel wins are taken
  only when they measurably beat the integer winner including the
  extra MV-bit cost. The MC-tile sampler (`sample_mc_tile`) likewise
  grew a sub-pel branch (`bilinear_luma_sample`) so the residual
  computation matches the decoder exactly when the chosen MV has
  sub-pel components. Spec ref: `vp6_format.pdf` §17.2 (Half / Quarter
  Pixel Aligned Vectors). Internal-decoder Y PSNR on the new
  translating-stripes / translating-disk fixtures (0.5-pel sub-pel
  shift, smooth low-frequency content) climbs from ~19-29 dB
  (integer-pel MC alone) to 35-37 dB (qpel MC + DCT residual). ffmpeg
  cross-decodes the qpel-MV inter packet cleanly (~32 dB Y on the
  stripes fixture) — no regression in
  `r21_inter_frame_ffmpeg_decode_state`.
- `tests/encoder_roundtrip.rs::r25_qpel_translating_stripes_psnr_clears_35db`,
  `r25_qpel_translating_disk_psnr_clears_35db`, and
  `r25_ffmpeg_decodes_qpel_inter_frame`. The first two pin the qpel
  PSNR floor at ≥ 35 dB Y on smooth low-frequency translation
  fixtures; both report the integer-only baseline alongside for
  visibility. The third confirms ffmpeg's vp6f decoder accepts the
  qpel MV bits without error and reconstructs ≥ 20 dB Y.

- **r24 — inter residual coefficient encoding.** `encode_inter_frame`
  now emits real DCT residual through the same `emit_block_coefs`
  state machine the keyframe path uses. Per MB:
  1. integer-pel motion search picks the best MV (unchanged from r23);
  2. `sample_mc_tile` materialises the MC prediction tile (mirror of
     `mb::render_mb_inter`'s integer-pel branch);
  3. per-pixel residual = `original - mc_pred`;
  4. `forward_dct8x8_residual` (new — same scaling as the keyframe
     `forward_dct8x8` but without the `-128` pixel bias since the
     residual is already centred on zero) → 64 frequency-domain
     coefficients;
  5. quantise DC + AC against `dequant_dc` / `dequant_ac` (`<< 2`
     scaled);
  6. apply the `RefKind::Previous` DC predictor (mirroring
     `mb::add_predictors_dc(scratch, RefKind::Previous)`) and emit
     `coded_dc = new_dc - predictor`;
  7. update the per-block DC mirror (`enc_left_block`,
     `enc_above_blocks`, `enc_prev_dc[plane][1]`) with the
     reconstruction the decoder will land on, so subsequent MBs see
     the same predictor.
  Internal-decoder Y PSNR on the new
  `r24_inter_residual_psnr_floor` fixture (flat keyframe + per-MB
  brightness shift) jumps from ~19 dB (MC-only baseline, the pre-r24
  ceiling) to ~43 dB (with residual). The existing
  `inter_frame_horizontal_shift_uses_mv` (where MC alone covers most
  of the change) records 40+ dB unchanged — the residual encoder
  doesn't hurt MV-friendly content. ffmpeg-side residual interop
  remains pending: ffmpeg accepts the bitstream end-to-end (no decode
  errors, both packets `n == 2`) but produces the MC-only baseline,
  suggesting a per-MB coefficient model state divergence downstream
  of the keyframe-time `0x80` defaults — left for r25+.
- `tests/encoder_roundtrip.rs::r24_inter_residual_psnr_floor`: pins
  the residual encoding floor at ≥30 dB Y PSNR AND ≥5 dB above the
  MC-only baseline. A regression that re-introduces the pre-r24
  3-bool zero-block shortcut trips the test immediately because the
  brightness-shift fixture is unrepresentable by MC alone.

### Fixed

- **r23 — ffmpeg inter-frame interop UNBLOCKED.** `Vp6Encoder::new` /
  `Vp6Encoder::default` now seed `sub_version = 6` (VP6.0 / Simple
  Profile) instead of the pre-r23 `0`. VP6 spec §9 / Table 2 defines
  byte 1 of the keyframe header as `Vp3VersionNo[5b] | VpProfile[2b] |
  Reserved[1b]` with `Vp3VersionNo` REQUIRED to hold 6, 7, or 8 (the
  spec page 25 description: "The decoder should check this field to
  ensure that it can decode the bitstream"). The previous value of 0
  was silently accepted by ffmpeg on the keyframe path but routed the
  inter parser through a Vp6.<keyframe-only> code path that mishandled
  subsequent frames — producing the long-standing "Invalid data found
  when processing input" inter-frame error. Wire change: byte 1 of the
  keyframe header is now `0x30` (was `0x00`); no other byte moves.
  The fix unblocks `tests/ffmpeg_interop.rs::r21_inter_frame_*` and
  `ffmpeg_decodes_keyframe_in_two_tag_stream`, both of which now
  strictly assert ffmpeg decodes both packets (`n == 2`). The
  `decode_first_20_frames` regression remains green because the
  decoder's `sub_version` gates (`> 7` / `< 8` / `> 6` in
  `rebuild_coeff_tables`) all behave identically for `sub_version = 6`
  as they did for `sub_version = 0`.
- `vector_predictors` (decoder) and `enc_vector_predictors` (encoder)
  now return the spec page 28 Table 5 mapping
  `(0 cands -> ctx 2, 1 cand -> ctx 1, 2+ cands -> ctx 0)`, i.e.
  `ctx = 2 - nb_pred`, instead of the legacy `nb_pred + 1` form.
  The skip-frame encoder's hard-coded `ctx = 1` was changed to
  `ctx = 2` to match (all neighbours OOB / zero-MV → spec ctx 2,
  "Neither Nearest nor Near MVs exists for this macroblock"). The
  pre-r22 codec was internally consistent but `mb_type[ctx][...]`
  picked the wrong row for a spec-following decoder. (r22 audit.)
- Audit of the per-MB block coefficient state machine confirmed the
  3-bit "all zero" shortcut path matches the decoder's `parse_coeff`
  exit conditions: at `coeff_idx = 0` the decoder reads `m2_0` only
  (DC has no EOB token by spec); at `coeff_idx = 1` with `ct = 0`
  the shortcut `coeff_idx > 1 && ct == 0` is false (1 is not
  strictly greater than 1), so the decoder reads `m2_0` then `m2_1`
  (EOB) — exactly the encoder's three emissions. `VP6_COEFF_GROUPS[1]
  = 0` so `cg = 0` for the AC pair, matching the encoder's index
  choices. No code change needed; the path is spec-correct as-is.
- `DEF_MB_TYPES_STATS` pair order now matches VP6 spec page 30
  `VP6_BaselineXmittedProbs[3][20]` — pairs flatten as
  `(probSame_t, probDiff_t)` per spec page 29 Table 6, not the
  previously-reversed `(probDiff, probSame)`. With this layout
  `Vp6Model::rebuild_mb_type_probs` reproduces spec page 35's
  `probModeSame` formula directly, so `mb_type[ctx][prev][0]` carries
  the spec's switch-rate semantics. The pre-fix table was internally
  consistent but disagreed with the already-spec-compliant
  `PRE_DEF_MB_TYPE_STATS` (`VP6_ModeVq` page 32), which would have
  produced visible breakage on a SetNewBaselineProbs reset. (r20
  audit.) ffmpeg-side acceptance of the inter packet still pending —
  see `src/encoder.rs` for the residual r21 suspect list (per-MB
  coefficient state machine + `vector_predictors` ctx mapping).
- VP6 Buff2Offset (spec Tables 2 & 3) emitted/parsed without the
  legacy +/-2 fudge so the on-wire value matches the literal frame-
  buffer byte offset to partition 2. Inter packets now have a spec-
  compliant partition layout. (r19 audit.)

### Added

- `tests/ffmpeg_interop.rs::keyframe_vp3_version_no_is_spec_legal`:
  pins byte 1 of the keyframe header to `Vp3VersionNo ∈ 6..=8`,
  `VpProfile == 0` (Simple), `Reserved == 0`. A regression that
  re-introduces the pre-r23 `sub_version = 0` default (which broke
  ffmpeg inter-frame interop while passing every internal
  round-trip test) trips this guard immediately. (r23 audit.)
- `decoder::tests::vector_predictors_ctx_mapping_matches_spec`: pins
  the spec page 28 Table 5 mapping for `vector_predictors`. A
  regression to `nb_pred + 1` (the pre-r22 form) trips the test
  because a top-left MB with all-OOB neighbours would yield ctx=1
  instead of the spec-required ctx=2.
- `tables::tests::def_mb_types_stats_matches_spec_baseline`: pins the
  `DEF_MB_TYPES_STATS` rows against VP6 spec page 30
  `VP6_BaselineXmittedProbs` so accidental pair-order reverts surface
  immediately.
- `tests/ffmpeg_interop.rs::r21_inter_frame_ffmpeg_decode_state`:
  records the ffmpeg-cross-decode contract for an inter frame
  produced by `encode_inter_frame` (motion search). Currently green
  at "1 frame decoded, 1 decode error" — fails red the moment ffmpeg
  starts accepting the inter so the assertion can be tightened to
  `n == 2`.
- `tests/ffmpeg_interop.rs`: external-ffmpeg interop guards
  (`ffmpeg_accepts_keyframe`, `ffmpeg_decodes_keyframe_in_two_tag_stream`).
  Skipped silently when ffmpeg isn't on PATH.
- `tests/dump_inter.rs::inter_buff2_offset_is_spec_compliant`: catches
  regressions of the Buff2Offset field semantics.

## [0.0.4](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.3](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.2...v0.0.3) - 2026-04-24

### Other

- add P-frame skip-encoder scaffold (InterNoVecPf, zero residual)
- fix IDCT axis transpose — scan -> raster uses default_dequant_table
- add AC coefficient encoding + zig-zag/run-length emission
- initial VP6F encoder scaffold (DC-only keyframes)
- add register() function for aggregator wiring

## [0.0.2](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.1...v0.0.2) - 2026-04-19

### Other

- fix inter drift, wire loop filter, add vp6a alpha path
- Merge wt/vp6-impl: VP6 keyframe decode + partial inter
- polish + README — remove unsafe MC path, document status
- fix inter-frame header parsing
- port range coder, tables, IDCT, MB decode

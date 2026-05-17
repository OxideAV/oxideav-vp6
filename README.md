# oxideav-vp6

Pure-Rust On2 **VP6** video decoder for oxideav. Zero C dependencies,
no FFI, no `*-sys` crates.

Covers both FLV flavours:
* **vp6f** — the Flash Video codec-id-4 stream, YUV 4:2:0.
* **vp6a** — codec-id-5 with an additional full-resolution alpha plane;
  output as `Yuva420P`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-vp6 = "0.0"
```

## Status

The authoritative reference is the On2 VP6 *Bitstream & Decoder
Specification* (vendored in this workspace at
`docs/video/vp6/vp6_format.pdf`); historically work also cross-checked
FFmpeg's reverse-engineered `libavcodec/vp56.c` + `libavcodec/vp6.c`
(+ `vp3dsp.c` for the IDCT / loop filter + `vp6dsp.c` for the 2D
bicubic interpolator + `vpx_rac.h` for the bool coder), but encoder
audit rounds (r19+) lean on the spec directly. **r27 wires INTER_FOURMV
on the encoder side**: `encode_inter_frame` now considers
`Vp56Mb::Inter4V` per MB. After the existing whole-MB qpel motion
search, a per-8×8 search runs (seeded from the integer winner with a
tight `±2` pel window + `±3` qpel refine) and a Lagrangian RDO step
picks FOURMV over `InterDeltaPf` when (a) at least one block's MV
diverges from the whole-MB MV by ≥ 2 qpel and (b) the FOURMV cost
(luma SAD across 4 blocks + λ × proxy bits) strictly beats the
single-MV cost. Wire emission mirrors `decode_4mv` in `decoder.rs`:
4 × 2 raw type bits FIRST (raw 0 = NoVec, raw 1 = Delta), then
per-block deltas. Per-block residual encoding picks up each block's MV
automatically; chroma MV is the round-shifted average of the 4 luma MVs
(mirror of the decoder's `RSHIFT(sum, 2)` chroma derive). New public
field `Vp6Encoder::allow_fourmv: bool` (default `true`) gates the
branch; setting to `false` reproduces pre-r27 single-MV behaviour. On a
32×32 diverging-blocks fixture (each 8×8 quadrant of every MB has a
distinct optimal MV), the FOURMV encode is 292 bytes vs 430 for
single-MV (~32% smaller); ffmpeg's vp6f decoder accepts the FOURMV
bitstream cleanly. r22 lands the spec
page 28 Table 5 ctx mapping for `vector_predictors` (`ctx = 2 -
nb_pred`) on both decoder + encoder sides. **r23 unblocks ffmpeg
inter-frame interop**: encoder default `sub_version` is now 6
(`Vp3VersionNo = VP6.0` per spec Table 2), was 0 — the spec-forbidden
zero was lenient on ffmpeg's keyframe path but mis-routed the inter
parser, surfacing as the long-running "Invalid data" inter error.
**r24 wires real inter residual coefficients**: `encode_inter_frame`
now materialises the integer-pel MC prediction, computes the per-pixel
residual against the source, runs the forward DCT (residual mode — no
`-128` bias), quantises DC + AC, runs the `RefKind::Previous` DC
predictor mirror, and emits the result through the same
`emit_block_coefs` state machine that already drove the keyframe
path. Internal-decoder PSNR on a flat-baseline + per-MB brightness-
shift fixture jumps from ~19 dB (MC-only) to ~43 dB (with residual).
ffmpeg-side residual interop still diverges (the inter decode lands on
the MC-only baseline, suggesting a per-MB coefficient model state
mismatch downstream of the keyframe-time `0x80` defaults) — left for
r25+. **r25 adds quarter-pel sub-pel motion estimation.**
`motion_search` now seeds the integer-pel SAD winner, then evaluates
every quarter-pel offset in a `±3 qpel` window around it via the same
H.264-chroma-style bilinear filter the decoder uses
(`put_h264_chroma8` mirror) plus a Lagrangian rate cost on the MV
bits. The MC tile sampling path (`sample_mc_tile`) likewise grew a
sub-pel branch so the residual computation matches the decoder
exactly. On translating-stripe / translating-disk fixtures with a
0.5-pel shift, internal-decoder Y PSNR climbs from ~19-29 dB
(integer-pel MC alone) to 35-37 dB (qpel MC + DCT residual). ffmpeg
cross-decodes the qpel-MV inter packet cleanly (Y PSNR ~32 dB on the
stripes fixture). **r26 wires golden-frame refresh + per-MB
golden-vs-previous selection** via the new
`Vp6Encoder::encode_inter_frame_with_golden(prev_*, golden_*, new_*,
…)` method. Cadence-driven by `golden_refresh_period` (default 30):
when fired, the picture-header `golden_frame_flag` bit is set so the
decoder snaps the just-decoded frame into its `golden_frame` slot.
Per MB the encoder runs `motion_search` against both refs and picks
the lower Lagrangian cost (SAD + λ * mv_bits), emitting one of
{`InterNoVecPf`, `InterDeltaPf`, `InterNoVecGf`, `InterDeltaGf`}
accordingly. On a 5-frame A,B,A,B,A periodic loop (32x32, QP 12)
pinning golden to the keyframe brings the inter-frame total wire size
from 378 bytes to 282 bytes (~25% smaller) vs refreshing every frame
— the loop-back A frames pick golden for every MB and emit
near-zero residual. ffmpeg cross-decodes the key + golden-refresh
inter pair cleanly. **r29 adds bitrate-targeting feedback control + Intra-in-inter
RDO on the encoder side.** New optional `Vp6Encoder::bitrate:
Option<BitrateControl>` field carries a proportional-with-EMA rate
controller; convenience method `Vp6Encoder::set_bitrate_target(bps,
fps)` derives `target_bytes_per_frame` and seeds the controller, and
`Vp6Encoder::update_qp_after_frame(bytes_emitted)` is called by the
caller after each `encode_*` to nudge `qp` toward the target within
configurable `[qp_min, qp_max]` bounds (defaults `[4, 60]`). The
controller is a pure no-op when `bitrate.is_none()` so pre-r29 fixed-QP
callers are unaffected. Verified by `r29_bitrate_control_tracks_target`
(seed QP 4 + 200-byte target → controller pushes QP to 60) and
`r29_bitrate_control_lowers_qp_when_undertarget` (seed QP 50 + 33K-byte
target → controller pulls QP to 22). Independent of bitrate control,
new public field `Vp6Encoder::allow_intra_in_inter: bool` (default
`true`) gates an intra-vs-inter RDO comparison per MB inside
`encode_inter_frame`. `Vp56Mb::Intra` is a reachable leaf from every
prev_type in the spec's PMBT tree; the encoder's existing
`encode_pmbt_tree` walk handles wire emission, and the per-block
residual loop now branches on `Intra` to use `forward_dct8x8` (with
-128 bias) + `RefKind::Current` DC predictor (mirror of
`add_predictors_dc(scratch, RefKind::Current)`) instead of the inter
path's residual-mode DCT + `RefKind::Previous` predictor. Intra fires
on revealed-content / scene-change MBs where the cheap intra-cost
proxy (per-MB SAD against the MB mean, +6 baseline PMBT-depth bits)
strictly beats the inter Lagrangian cost. On a scene-change fixture
(keyframe stripes → inter checkerboard) the intra-on encode is 235 B
vs 306 B for intra-off — 23% smaller, 30.5 dB Y PSNR via our decoder
(no MC-only baseline). On smooth-motion content the wire output is
byte-identical with `allow_intra_in_inter = true` vs `false` — the
heuristic correctly identifies that inter compensates fully.
**r31 adds scene-change-driven golden refresh, Huffman inter encode, and
bool/Huffman RDO.** Three encoder-only improvements:

1. `Vp6Encoder::scene_change_threshold` (default `2.0`): when set > 0,
   `encode_inter_frame_with_golden` computes the per-pixel luma SAD of the
   new frame against the previous frame and compares it against an EMA
   running mean (α=0.1). A SAD spike above `threshold × ema_sad` triggers a
   golden refresh immediately, independent of the cadence counter. Set to
   `0.0` to disable (cadence-only). On a scene-change fixture (stripes →
   checkerboard) the counter resets correctly at the cut frame.
2. `Vp6Encoder::encode_inter_frame_huffman(...)` — same ME + mode-decision
   pipeline as `encode_inter_frame` but coefficient partition 2 is Huffman-
   coded (`UseHuffman = 1`). Our decoder and ffmpeg both consume the output.
   On a 32×32 shifting-stripes fixture (QP 16) Huffman is 89 B vs 85 B
   bool (1.05× — tree overhead dominates on small frames).
3. `Vp6Encoder::encode_inter_frame_rdo(...)` — runs both bool and Huffman
   paths and returns the smaller output. Guaranteed ≤ bool-only. ~2× slower
   (two full ME passes); suitable for offline encodes.

**r30 promotes the rate controller to PI + adds DCT-count
intra cost + lifts intra-in-inter into the golden-aware path.** Three
encoder-only refinements layered on r29:

1. `BitrateControl` grows `ki: f32` (default `0.05`) + `integral: f32`
   + `integral_clamp: f32` (default `5.0`). `update_qp_after_frame` now
   computes `qp_delta = round((kp * err + ki * integral) * 8)` after an
   anti-windup-clamped integral accumulation; the integral term
   eliminates the steady-state bitrate offset a P-only controller leaves
   on content whose intrinsic complexity needs a QP set-point different
   from the seed. Saturated-actuator back-leak prevents runaway. Setting
   `ki = 0.0` recovers pre-r30 P-only behaviour exactly. On a
   noisy-content fixture the PI controller pushes QP from 20 → 44 in 6
   frames vs P-only's 20 → 36 (22% faster convergence).
2. The intra-vs-inter cost in `encode_inter_frame` now adds a DCT-count
   term (`mb_intra_dct_count_proxy`) that quantises each 8×8 luma block
   of the MB at the current `qp` and counts surviving non-zero AC
   coefficients. Cost = `Σ |pixel - mean| + λ * (count * 4 + 6)`,
   matching the per-token bit budget of the bool-coded coefficient
   tree (~4 bits per surviving AC token). Closer to actual encode cost
   than SAD-against-mean alone — high-frequency MBs no longer slip
   through the cheap-intra cracks; flat-but-noisy MBs no longer get
   mis-classified as expensive-intra.
3. `encode_inter_frame_with_golden` now also evaluates intra against
   the BEST inter (golden vs prev) — the "golden-aware" qualifier means
   intra fires only when both refs are unrelated to the new content.
   Per-MB residual encoding picks up a 3-way split (Intra / Inter-Prev
   / Inter-Golden) sharing the same emission shape; ref-kind drives
   the DC-predictor neighbour-match. On a scene-change against
   matching-content refs (both prev + golden = stripes, new =
   checkerboard) the with-intra encode is 235 B vs 306 B for intra-off
   (~23% smaller), 30.5 dB Y PSNR; ffmpeg's vp6f decoder cross-decodes
   cleanly.
**r28 adds the Huffman coefficient path on both
encoder and decoder sides.** The frame-header `UseHuffman` bit (spec
page 23 Table 1) selects between the existing bool-coded coefficient
partition and a new raw-bit Huffman partition described in spec
sections 7.2 + 13.1 + 13.2.2 + 13.3.2 + 13.3.3.2 + 13.4. Trees are
built per `VP6_CreateHuffmanTree` from probability vectors derived
via the spec's `DCTTokenBoolTreeToHuffProbs` /
`ZRLBoolTreeToHuffProbs` conversions; the DCT-token tree (12 leaves),
AC-band trees (prec×plane×band = 24 trees) and ZRL trees (2) are all
rebuilt per frame from the post-update bool-coded model state.
Cross-block DC-zero and AC1-EOB runs use the spec page 81
`1 + R(2)` raw-bit encoding. New encoder method
`Vp6Encoder::encode_keyframe_huffman(...)` writes a keyframe whose
partition 2 is Huffman-coded; the decoder's previous
`Error::Unsupported` for `use_huffman = 1` is replaced by the new
parse path. Both sides share the bool-coded DC-predictor /
dequantiser plumbing (`mb::add_predictors_dc`), so the
reconstruction model is byte-identical to the bool path — only the
coefficient-bitstream emission changes. Round-trips through our own
decoder; ffmpeg's vp6f decoder accepts the Huffman keyframe
(`tests/huffman_roundtrip.rs::ffmpeg_decodes_huffman_keyframe`).
**r73 lands SATD-based qpel motion-estimation refinement.**
`motion_search` and `motion_search_8x8` now score quarter-pel candidates
via SATD (Sum of Absolute Transformed Differences, computed by a
classical H.264-style 4×4 Hadamard kernel applied to tiled 4×4 sub-
blocks of the 8×8 residual) instead of plain SAD. SATD better predicts
post-DCT bit cost than SAD because it captures frequency-domain energy
distribution — a residual whose pixel-domain SAD is low but whose AC
bins are strongly excited gets a higher SATD, steering the diamond
toward candidates whose residual is genuinely sparse in the transform
domain. Integer-pel search stays SAD-based (full-window scan, cheap,
multimodal surface); SATD is applied only during qpel refinement where
the SAD surface is near-flat and the residual energy distribution is
what differentiates candidates. New public field
`Vp6Encoder::allow_satd_me: bool` (default `true`) gates the path; set
to `false` to recover pre-r73 SAD-only diamond behaviour for A/B
testing. Lambda is scaled by 4× under SATD so the cost ratio between
distortion and MV-bit rate stays comparable. Wire format is unchanged
— only the chosen sub-pel MV may differ. Wired into
`encode_inter_frame`, `encode_inter_frame_with_golden`, and
`encode_inter_frame_huffman` via the shared ME helpers. Pinned by
`r73_satd_qpel_internal_psnr_clears_45db_on_flat`,
`r73_satd_qpel_no_regression_on_r25_stripes_fixture` (35.20 dB,
identical to pre-r73 SAD baseline), and
`r73_satd_qpel_improves_or_matches_psnr_on_textured_motion`.
**r39 lands a PID controller + iterative diamond qpel ME + trellis-style
AC quantisation.** Three independent encoder-only refinements, each
gated by an existing or new public field for A/B testing:

1. `BitrateControl` grows `kd: f32` (default `0.15`) + `prev_error: f32`
   fields. `update_qp_after_frame` now adds a derivative-on-error term:
   `qp_delta = round((kp * err + ki * integral + kd * derivative) * 8)`
   where `derivative = error_ratio - prev_error_ratio` against the
   EMA-smoothed error. Setting `kd = 0.0` recovers pre-r39 PI-only
   behaviour exactly. Pinned by
   `r39_pid_kd_zero_matches_pi_exactly` (byte-for-byte QP path
   equivalence with PI when kd=0) and
   `r39_pid_controller_reduces_overshoot_vs_pi_only`.
2. `motion_search` and `motion_search_8x8` swap their pre-r39 ±3 qpel
   box (49-position exhaustive search around the integer winner) for
   an iterative 8-conn diamond pattern with 6 iterations and ±6 qpel
   bounds. Each iteration evaluates the 8-conn neighbours of the
   current best candidate; if a neighbour strictly improves the
   Lagrangian cost, it becomes the new centre. Probe count is bounded
   by 8 × 6 = 48 (comparable to the pre-r39 box) but the effective
   radius is 6 qpel — double the pre-r39 catch radius for sub-pel
   winners that lived past the box edge. Pinned by
   `r39_diamond_qpel_me_internal_psnr_clears_45db` (flat-content
   skip-path is exactly recoverable, ∞ dB PSNR) and
   `r39_diamond_qpel_me_no_regression_on_r25_stripes_fixture` (existing
   r25 stripes fixture clears the 35 dB floor at 35.20 dB).
3. New public field `Vp6Encoder::allow_trellis: bool` (default `true`)
   gates a per-block trellis-style AC quantisation pass that, for each
   non-zero AC level, considers driving the level toward zero by 1 LSB
   when the resulting drop in rate (per-coef bool-tree depth proxy)
   outweighs the squared-quantisation-error increase scaled by a
   QP-derived Lagrangian λ. Bit-exact byte-output-equivalent to plain
   `div_nearest` quantise when the per-coef RD never finds a win;
   shaves 1-3% of inter-frame bytes on natural-content fixtures with
   AC coefs near the quantiser threshold. Wired into
   `encode_inter_frame`, `encode_inter_frame_with_golden`, and the
   Huffman inter path. Pinned by
   `r39_trellis_shrinks_bitstream_at_minimal_psnr_loss` (≤ size, ≤
   0.5 dB PSNR drop on a 64×64 natural fixture; observed 1 byte saving
   at 0.02 dB drop). Set `allow_trellis = false` to recover plain
   `div_nearest` quantise behaviour exactly. The trellis is
   bool-decoder-compatible — wire format is unchanged, only the per-coef
   level decisions differ.

### Implemented

- **Range coder** ("VP56 bool coder"): FFmpeg-equivalent 16-bit
  renormalisation, 24-bit seed, both `get_prob(p)` and `get_bit()`
  (equiprobable) paths. Walks static [`Vp56Tree`] probability trees
  and round-trips against an in-crate encoder in unit tests.
- **Frame header parse**: keyframe (with dimensions) and inter-frame
  headers, including the "separated-coeff" offset field, sub-version
  gating for the extra filter-info block, and the interlaced-profile
  flag.
- **Model state** (`Vp6Model`): default init, MB-type-stats re-training,
  vector model updates, coefficient model updates, coefficient-reorder
  scan retuning, linear-combination rebuild of `coeff_dcct` from
  `coeff_dccv`.
- **Macroblock-type decode tree**: ports `vp56_parse_mb_type_models`
  + `vp56_parse_mb_type` over the MB-type context tree.
- **Motion-vector decode**: all 10 VP56 MB types including `INTER_4V`
  (per-8x8 MVs, averaged chroma MV), `INTER_DELTA_*`
  (`vp6_parse_vector_adjustment` with the predicted-delta tree and
  the full-delta bit layout), and the 12-position MV candidate
  predictor walk. The chosen MV is stashed back into the persistent
  `macroblocks[]` table so future neighbours see it — the missing
  piece of that write-back was the cause of inter-frame drift
  through the 0.0.1 series.
- **Coefficient decode — range path**: full port of `vp6_parse_coeff`,
  including zero-run categories, the 6-category long tree, the
  `coeff_dcct` / `coeff_ract` / `coeff_runv` trees, and the reorder
  permutation.
- **Integer 8×8 IDCT**: port of `vp3dsp.c::idct` (put + add paths),
  with the DC-only fast path (`vp3_idct_dc_add_c`).
- **Intra prediction**: DC prediction from above/left 8x8 block
  boundaries (`vp56_add_predictors_dc`), using the reference-frame
  DC context kept in `left_block[]` / `above_blocks[]`.
- **Inter motion compensation**: reference-frame MC with the 4-tap
  VP6 bicubic filter (`vp6_block_copy_filter` + `vp6_filter_diag4`)
  for luma and the H.264-style bilinear chroma filter
  (`put_h264_chroma_mc8`). Handles integer-pel, half-pel, quarter-pel
  phases. Edge pixels are mirror-clamped into a scratch tile instead
  of FFmpeg's `emulated_edge_mc` — functionally equivalent for
  interior-dominated streams.
- **VP3-style deblock loop filter**: per-block edge filter
  (`ff_vp3dsp_{h,v}_loop_filter_12`) applied to the 12x12 MC scratch
  tile before sub-pel filtering, gated by the `deblock_filtering`
  bit in the picture header (keyframe default = on, matching
  FFmpeg's `ff_vp56_init_context`). Only the VP6 variant is wired;
  VP5's separate `vp56dsp.edge_filter_{hor,ver}` path is irrelevant
  to FLV/vp6 streams.
- **`vp6a` alpha plane**: two-stream decode driven by the 3-byte
  BE24 alpha offset at the head of each packet. The primary context
  decodes YUV as usual; a second context decodes the alpha partition
  as a monochrome VP6 stream and the luma samples surface as the
  `A` plane of a `Yuva420P` output.
- **Reference-frame management**: tracks `prev_frame` and
  `golden_frame` planes inside the decoder. Keyframes overwrite both;
  inter frames refresh golden when the golden-frame flag is set.
- **Encoder INTER_FOURMV / per-8×8 motion vectors** (r27+):
  `Vp6Encoder::encode_inter_frame` considers `Vp56Mb::Inter4V` as a
  third candidate alongside `InterNoVecPf` / `InterDeltaPf`. Per-MB
  pipeline: whole-MB qpel ME → per-8×8 qpel ME (`motion_search_8x8`,
  seeded from the whole-MB integer winner) → Lagrangian RDO between
  single-MV and FOURMV costs. FOURMV fires when (a) at least one
  block's MV diverges from the whole-MB MV by ≥ 2 qpel AND (b) the
  FOURMV cost strictly beats single-MV. Wire format mirrors
  `decode_4mv`: 4 × 2 raw type bits (raw 0 = NoVec, raw 1 = Delta) all
  emitted before any delta, then per-block MV deltas in order. Chroma
  MVs derived as the round-shifted average of the 4 luma MVs. Public
  `allow_fourmv: bool` (default `true`) gates the branch; turning it
  off reproduces pre-r27 single-MV behaviour for A/B comparison. On
  the diverging-blocks fixture (each 8×8 quadrant within every MB has
  a distinct optimal MV) the FOURMV encode is 32% smaller than the
  single-MV encode at QP 12; ffmpeg's vp6f decoder cross-decodes the
  FOURMV bitstream cleanly.
- **Encoder golden-frame refresh** (r26+):
  `Vp6Encoder::encode_inter_frame_with_golden(prev_*, golden_*,
  new_*, …)` accepts both refs and per-MB picks whichever beats the
  other on a Lagrangian SAD cost. Cadence-driven by the public
  `golden_refresh_period: u32` (default 30); when fired the
  picture-header `golden_frame_flag` bit is set so the decoder snaps
  the just-decoded frame into its `golden_frame` slot. MB types are
  one of `{InterNoVecPf, InterDeltaPf, InterNoVecGf, InterDeltaGf}`
  with separate prev / golden MV-candidate state, and the per-block
  DC predictor mirror tracks `RefKind::Golden` alongside
  `RefKind::Previous`. ffmpeg cross-decodes the refresh-flag inter
  packet cleanly. Reduces wire size on periodic-structure content
  (animation loop, slideshow) — see CHANGELOG / fixture results.

- **Encoder bitrate-targeting PID controller** (r29+, PI in r30, PID in r39).
  `oxideav_vp6::encoder::BitrateControl` carries `target_bytes_per_frame`
  + `qp_min` / `qp_max` bounds + EMA + proportional + integral +
  derivative gains (`kp`, `ki`, `kd`) + anti-windup `integral_clamp`.
  `set_bitrate_target(bps, fps)` initialises the controller;
  `update_qp_after_frame(bytes)` applies the per-frame nudge
  `qp_delta = round((kp * err + ki * integral + kd * derivative) * 8)`
  after EMA-smoothed error, clamped integral accumulation, and
  per-frame derivative against the last-frame error. Saturated-actuator
  back-leak prevents integral wind-up. Set `ki = 0.0, kd = 0.0` to
  recover P-only; set `kd = 0.0` alone to recover pre-r39 PI behaviour.
  Pure no-op when `bitrate.is_none()`.

- **Encoder trellis-style AC quantisation** (r39+). Per-block per-coef
  RD pass on the inter-frame residual AC stream. For each non-zero
  level, considers driving toward zero by 1 LSB; chooses the drop when
  the rate saving (per-coef bool-tree depth proxy) outweighs the
  squared-quantisation-error increase scaled by a QP-derived λ. Public
  field `Vp6Encoder::allow_trellis: bool` (default `true`). Wire-format
  unchanged — bool-decoder reads identical state machine. Conservative
  sweet spot: levels of `±1` get pruned only when the raw coef sat
  just over the half-step threshold. Wired into `encode_inter_frame`,
  `encode_inter_frame_with_golden`, and the Huffman inter path. Set
  `false` to recover plain `div_nearest` quantise.

- **Encoder SATD-based qpel ME refinement** (r73+). `motion_search`
  and `motion_search_8x8` evaluate quarter-pel candidates inside the
  iterative diamond via SATD (Sum of Absolute Transformed Differences
  with a 4×4 Hadamard kernel applied to tiled 4×4 sub-blocks of the
  residual) instead of plain SAD. SATD captures frequency-domain energy
  distribution and so better predicts post-DCT bit cost than pixel-
  domain SAD on sub-pel-mispredicted residuals. Integer-pel search
  stays SAD-based for full-window speed; SATD applies only during qpel
  refinement. Public field `Vp6Encoder::allow_satd_me: bool` (default
  `true`) gates the metric; setting to `false` recovers pre-r73 SAD-
  only diamond behaviour exactly. Wire format unchanged. Wired into
  `encode_inter_frame`, `encode_inter_frame_with_golden`, and
  `encode_inter_frame_huffman`.

- **Encoder iterative diamond qpel ME** (r25 → r39). `motion_search`
  and `motion_search_8x8` swap their pre-r39 ±3 qpel box (49-position
  exhaustive search) for an iterative 8-conn diamond pattern bounded
  to ±6 qpel from the integer winner with up to 6 iterations per MB
  (probe budget ~8 × 6 = 48). Each iteration evaluates the 8-conn
  neighbours of the current best candidate; if a neighbour strictly
  improves the Lagrangian cost, it becomes the new centre and search
  continues. Doubles the effective qpel catch-radius without growing
  the per-MB probe budget. Stops early when no neighbour beats the
  current best — typically 2-3 iterations on smooth motion content.

- **Scene-change golden refresh** (r31+). `encode_inter_frame_with_golden`
  detects SAD spikes via `scene_change_threshold` (default `2.0`) and
  `sad_ema` (EMA α=0.1). A spike triggers a forced golden refresh regardless
  of the cadence counter. Disable with `scene_change_threshold = 0.0`.

- **Huffman inter encode** (r31+). `encode_inter_frame_huffman(...)` emits
  P-frames with Huffman-coded coefficient partition (`UseHuffman = 1`) and
  bool-coded mode/MV partition. Compatible with our decoder and ffmpeg's
  vp6f decoder.

- **Bool/Huffman inter RDO** (r31+). `encode_inter_frame_rdo(...)` runs both
  bool and Huffman encode paths and returns the smaller output. Guaranteed ≤
  bool-only at the cost of ~2× encode time. Suitable for offline encodes.

- **Encoder Intra-in-inter RDO** (r29+, golden-aware in r30).
  Both `encode_inter_frame` and `encode_inter_frame_with_golden`
  consider `Vp56Mb::Intra` as a per-MB candidate alongside the inter
  modes. Cost = `Σ |pixel - mean| + λ * (DCT-survivor-count * 4 + 6)`
  — the SAD-against-mean predictability proxy plus an
  `mb_intra_dct_count_proxy` term that quantises each 8×8 luma block
  at the current `qp` and counts surviving non-zero AC coefficients,
  matching the per-token bit budget of the bool-coded coefficient tree
  much more closely than SAD alone. The golden-aware path compares
  intra against the BEST inter (golden vs prev) so intra fires only
  when both refs are unrelated to the new content. Spec-correct wire
  emission via the existing PMBT-tree walk; per-block residual
  encoding switches between `forward_dct8x8` (Intra, with -128 bias)
  and `forward_dct8x8_residual` (Inter), with the matching DC
  predictor (`RefKind::Current` vs `RefKind::Previous` /
  `RefKind::Golden`). Public field `Vp6Encoder::allow_intra_in_inter:
  bool` (default `true`) gates the branch — set to `false` for
  pre-r29 inter-only behaviour.

- **Huffman coefficient path** (r28+, encoder + decoder). When a VP6F
  stream sets `UseHuffman = 1` in the frame header (per VP6 spec page
  23 Table 1) the second data partition (DCT coefficients) is read /
  written as a raw MSB-first bitstream of Huffman codewords instead of
  the range-coder bool path. Trees are constructed per spec section 7.2
  (`VP6_CreateHuffmanTree`) from the bool-coded probability vectors via
  the spec's `DCTTokenBoolTreeToHuffProbs` /
  `ZRLBoolTreeToHuffProbs` conversions (sections 13.1 and 13.3.3.2).
  The DC token path uses the `DCT_EOB` / `ZERO` / `ONE` …
  `DCT_VAL_CATEGORY6` token set (spec page 57 Table 18); cross-block
  DC-zero and AC1-EOB runs follow the spec page 81 `1 + R(2)` raw-bit
  encoding. Encoder side: `Vp6Encoder::encode_keyframe_huffman(...)`
  emits a keyframe whose partition 2 is Huffman-coded; the existing
  `encode_keyframe(...)` continues to emit the bool path. Decoder side:
  the `UseHuffman` bit in the picture header is now respected — both
  paths share the same DC predictor / dequantiser plumbing
  (`mb::add_predictors_dc`).

### Not yet implemented (deferrals)

- **Interlaced profile**: parsed but not exercised end-to-end.

### Test coverage

The crate ships 58 library unit tests plus 74 integration tests
across 8 files (132 tests total):

- **Unit tests** for the range coder round-trip, the IDCT (DC-only flat
  block, add-zero identity), the loop filter bounding-values table and
  edge-smoothing (`h_loop_filter_smooths_sharp_edge`,
  `v_loop_filter_smooths_sharp_edge`), the H.264 chroma MC
  integer-pel fast path, model defaults, and the MB-type enum layout.
- `tests/keyframe_from_flv.rs` — walks the
  `asian-commercials-are-weird.flv` sample (skipped if absent;
  override path via `OXIDEAV_FLV_SAMPLE=...`), decodes the first
  VP6F keyframe, then runs 20 consecutive frames through the inter
  decode path and asserts all 20 decode cleanly.
- `tests/loop_filter_delta.rs` — synthetic test that renders an inter
  MB over a block-aligned reference edge with deblock-filtering off
  vs. on, asserts the filter reduces the output gradient across the
  former edge.
- `tests/vp6a_roundtrip.rs` — synthesises a vp6a packet by wrapping a
  real VP6F keyframe in the 3-byte alpha-offset prefix + duplicating
  it into the alpha partition, decodes it, and verifies a 4-plane
  YUVA frame with non-zero alpha pixels. No real vp6a FLV fixture is
  shipped in the tree; if you have one, set `OXIDEAV_FLV_SAMPLE`.
- `tests/ffmpeg_interop.rs` — external-ffmpeg interop guards. Skipped
  silently when `ffmpeg` isn't on `PATH`. As of r23 every guard
  asserts ffmpeg accepts both packets in a 2-tag (key + inter) stream
  (`n == 2`), covering the keyframe path
  (`ffmpeg_accepts_keyframe`), the skip-frame inter path
  (`ffmpeg_decodes_keyframe_in_two_tag_stream`), the motion-search
  inter path (`r21_inter_frame_ffmpeg_decode_state`), and the
  spec-legal `Vp3VersionNo` byte
  (`keyframe_vp3_version_no_is_spec_legal`).
- `tests/dump_inter.rs` — opt-in `VP6_DUMP_INTER=1` diagnostic that
  writes a 2-tag FLV to `/tmp/oxideav_vp6_dump.flv` for ffmpeg-side
  manual inspection, plus a `inter_buff2_offset_is_spec_compliant`
  guard that pins the spec layout of the partition-offset field.
- `tests/encoder_roundtrip.rs::r24_inter_residual_psnr_floor` (new in
  r24) — encodes a flat keyframe + per-MB brightness-shift inter
  through `encode_inter_frame`, decodes the 2-tag stream through the
  in-tree decoder, and asserts the inter-frame Y PSNR clears 30 dB
  AND beats the MC-only baseline by ≥5 dB. Fails immediately on a
  regression that drops the residual coefficient path back to the
  pre-r24 zero-block shortcut.
- `tests/encoder_roundtrip.rs::r25_qpel_translating_stripes_psnr_clears_35db`
  / `r25_qpel_translating_disk_psnr_clears_35db` (new in r25) — pin
  the quarter-pel ME path against a 0.5-pel sub-pel translation of a
  smooth low-frequency stripe / Gaussian-disk fixture. Both assert
  internal-decoder Y PSNR ≥ 35 dB; the integer-only baseline (MC
  alone, no qpel) is ~19-29 dB so the qpel ME contribution is
  unmistakable. `r25_ffmpeg_decodes_qpel_inter_frame` cross-decodes
  the stripes packet through ffmpeg's vp6f decoder and asserts ≥ 20
  dB Y PSNR, confirming the qpel MV bits parse cleanly.
- `tests/encoder_roundtrip.rs::r27_fourmv_inter_smaller_than_single_mv_on_diverging_blocks`
  / `r27_ffmpeg_decodes_fourmv_inter_frame` (new in r27) — pin the
  FOURMV path against a 32×32 diverging-blocks fixture (each 8×8
  quadrant of every MB has a distinct optimal MV). The first asserts
  the FOURMV-on encode is ≤ 95% of the FOURMV-off encode (we observe
  ~68% in practice — 292 vs 430 bytes), keyframes are byte-identical
  across `allow_fourmv` values, and own-decoder Y PSNR ≥ 20 dB. The
  second cross-decodes the key + FOURMV inter pair through ffmpeg's
  vp6f decoder and asserts both packets parse cleanly.
- `tests/encoder_roundtrip.rs::golden_refresh_*` (new in r26) — five
  guards on the new `encode_inter_frame_with_golden` API:
  `golden_refresh_cadence_fires_on_period` pins the counter
  semantics; `golden_refresh_disabled_at_period_zero` covers the
  `period = 0` disabled branch; `golden_refresh_loop_back_uses_
  golden_reference` walks an A→B→A loop and asserts our decoder
  reconstructs frame 2 at ≥ 25 dB Y PSNR (45 dB observed) vs an 8.6
  dB skip-from-prev baseline; `golden_refresh_reduces_bytes_on_
  periodic_loop` encodes a 5-frame A,B,A,B,A loop twice (chasing-
  golden vs pinned-golden) and pins the pinned-golden total wire
  size to ≤ 110% of chasing-golden (282 vs 378 bytes observed);
  `ffmpeg_decodes_inter_with_golden_refresh_flag` cross-decodes a
  key + golden-refresh inter pair through ffmpeg's vp6f decoder
  (must accept both packets).

## Quick use

```rust
use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_codec::Decoder;

let mut dec = oxideav_vp6::Vp6Decoder::new(CodecId::new("vp6f"));
let pkt = Packet::new(0u32, TimeBase::new(1, 1000), vec![/* coded frame */]);
dec.send_packet(&pkt)?;
let _frame = dec.receive_frame();
# Ok::<(), oxideav_core::Error>(())
```

For server / sandbox callers, use `Vp6Decoder::with_limits(codec_id, limits)`
to thread an explicit `oxideav_core::DecoderLimits` through. The decoder
honours `max_pixels_per_frame` (header-parse pixel cap),
`max_arenas_in_flight` (arena-pool size; natural backpressure when full),
and `max_alloc_bytes_per_frame` (per-arena byte cap, clamped to a VP6
ceiling of 8 MiB). The `Decoder::receive_arena_frame` override returns
true zero-copy frames whose plane bytes live inside the leased arena
buffer.

## License

MIT — see [LICENSE](LICENSE).

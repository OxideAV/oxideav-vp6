# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

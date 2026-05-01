# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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

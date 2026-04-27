# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.5](https://github.com/OxideAV/oxideav-vp6/compare/v0.0.4...v0.0.5) - 2026-04-27

### Other

- r20 — Buff2Offset spec compliance + ffmpeg interop scaffolding
- round 19 — partial inter-frame audit (ffmpeg still rejects)
- round 18 — diagnostic dump for inter-frame ffmpeg interop
- round 17 — encoder MV emission + integer-pel ME
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

### Fixed

- VP6 Buff2Offset (spec Tables 2 & 3) emitted/parsed without the
  legacy +/-2 fudge so the on-wire value matches the literal frame-
  buffer byte offset to partition 2. Inter packets now have a spec-
  compliant partition layout. (r19 audit; ffmpeg still rejects the
  inter packet body — see `src/encoder.rs` for the residual suspect
  list.)

### Added

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

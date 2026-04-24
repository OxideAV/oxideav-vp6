# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

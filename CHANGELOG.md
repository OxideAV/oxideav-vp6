# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

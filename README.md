# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 1 (2026-05-21).** The orphan-rebuild
scaffold from 2026-05-18 is being replaced incrementally by parsers
sourced exclusively from
[On2 Technologies' VP6 Bitstream & Decoder Specification](https://github.com/OxideAV/oxideav/blob/master/docs/video/vp6/vp6_format.pdf)
(document version 1.02, August 2006). No third-party VP6
implementation is consulted at any stage.

### What round 1 lands

- `Vp6FrameHeader::parse` — frame-header raw-bit prefix:
  - Table 1 (`FrameType`, `DctQMask`, `MultiStream`)
  - Table 2 R(n) fields (`Vp3VersionNo`, `VpProfile`, `Reserved`,
    conditional `Buff2Offset`)
- Typed `CodingProfile` (Simple / Advanced / Reserved) and
  `Vp3Version` (VP6.0 / 6.1 / 6.2 / Other) enums.
- `Error::Truncated` for short-input failures.

### What round 1 does NOT land

- Anything downstream of the BoolCoder switch in the frame header
  (`VFragments`, `HFragments`, scaling, filter selectors,
  `UseHuffman`). Blocked on a DOCS-GAP against spec §7.3 — the
  `Split = 1 + (((Range-1) * Probability) >> 7)` formula collapses
  the prob-128 (`b(n)`) decoder path to always-0 and overflows when
  `Probability > 128`. The fix is either a confirmation that `>> 7`
  is correct alongside an encoder-side mapping explanation, or a
  correction to `>> 8` (matching the VPx-family arithmetic coder
  pattern). See the crate-root docs for the full report.

## License

MIT — see [LICENSE](./LICENSE).

# oxideav-vp6

Pure-Rust On2 VP6 (`vp6f` / `VP60` / `VP61` / `VP62`) video codec for
the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Clean-room rebuild in progress.** The crate is a stage-by-stage
re-implementation against On2 Technologies' "VP6 Bitstream & Decoder
Specification" (document version 1.02, August 2006), staged at
`docs/video/vp6/vp6_format.pdf` plus the clean-room errata at
`docs/video/vp6/vp6-errata-and-clarifications.md`. No third-party VP6
source has been consulted at any stage.

Almost every decode primitive is implemented and unit-tested, but the
crate does **not yet expose a full frame decoder** and does **not
register a `Decoder` with `oxideav-core`** — `register()` is currently
a no-op. It is a primitive library, not a wired codec.

### Implemented stages

- **Frame-header raw-bit prefix** (`frame_header`) — the §9 Table 1 / 2
  `R(n)` fields up to the BoolCoder tail.
- **Per-block reconstruction** — inverse quantization (`dequant`, §15),
  inverse DCT (`idct_block`, §16), intra and inter block recombination
  (§17.1–§17.4), fractional-pixel interpolation filters (`interp`,
  §11.4), the prediction loop filter (`loopfilter`, §11.3), and the
  Unrestricted-Motion-Vector border extension (`umv`, §11.5).
- **Static entropy surfaces** — the zig-zag scan + DC predictor
  (`scan` / `dc_pred`, §12.1 / §14), coding-mode tables (`modes`, §10),
  DCT-token tables (`tokens`, §13), the Huffman tree builder
  (`huffman`, §7.2), the AC zero-run-length tables (`zrl`, §13.3.3),
  and the raw-bit reader (`raw_bits`, §3).
- **BoolCoder-consuming layers** — the §7.3 binary arithmetic decoder
  (`bool_coder`), the §13.2.1 DC and §13.3.1 AC arithmetic decoders
  (`dct_decode`), the §13.3.3.1 zero-run traversal, the per-frame
  probability-update bitstreams (`prob_update` / `mv_prob_update` /
  `mode_prob_update` / `scan_update`), the §11.1 motion-vector
  component decoder (`mv_decode`), the §11 differential MV
  reconstruction (`mv_diff`), the §10 `CODE_INTER_FOURMV` block-mode
  signaling (`fourmv`) and chroma-MV derivation, the §10
  Nearest/Near neighbour walker (`near_mv`), the §10 `VP6_DecodeMode`
  macroblock-mode traversal (`mode_decode`), and the §9 output-scaling
  surface (`scaling`).
- **Frame assembly** (`frame_assembly`) — block-to-plane raster
  placement of reconstructed 8×8 blocks into a YUV 4:2:0 image.

### Blocked

- **Full P-frame / I-frame decode loop.** The per-MB driver that walks
  the macroblock grid (mode decode → MV decode → coefficient decode →
  reconstruct → assemble) and the `Decoder` registration are blocked on
  the §9 frame-header BoolCoder tail. That tail is gated on a
  **documented inconsistency in the staged §7.3 `Split` errata
  (#35)**: the spec PDF prints `Split = 1 + (((Range-1)*Probability)
  >> 7)`, but every quantitative property the errata's own rationale
  relies on (half interval at probability 128, non-empty branches,
  `Range <= 255`) holds only under `>> 8`; under the literal `>> 7`
  the `Bit = 1` interval is empty at probability 128, making
  fixed-probability `b(n)` header fields undecodable. The round-15
  `BoolCoder` implements the literal `>> 7`; resolving the degeneracy
  needs a docs correction or a worked byte-trace, after which the
  remaining driver wiring is mechanical.
- **High-bit-depth / scaling resampling math** and **sample-exact
  validation against a conformant `.vp6` bitstream** — the latter
  needs an encoder-produced fixture (synthetic high-probability streams
  hit the same §7.3 degeneracy).

## License

MIT — see [LICENSE](./LICENSE).

# vp6f-huffman-i-then-p-854x480: Flash VP6 (Huffman 2nd partition), I-frame + 2 P-frames

Third-party conformance fixture. The `.flv`/`.vp6` input is a **black-box-produced
VP6 bitstream** (data, not derived from any decoder source); FFmpeg's `vp6f`
decoder is used only as the **decode oracle** to produce `expected.yuv`. Same
pattern as the other `video/*/fixtures` oracles. FFmpeg's VP6 decoder source was
**not** consulted.

## Provenance

| | |
| --- | --- |
| Source file | `backcountry_bombshells_4min_HD_1500_96.flv` |
| Source URL | https://samples.ffmpeg.org/V-codecs/VP6/huffman/backcountry_bombshells_4min_HD_1500_96.flv (mirror: samples.mplayerhq.hu) |
| Full-source MD5 | `a10455decdb94f4e41343752dd22860a` (published in the repo `md5sum`) |
| Full-source size | 51,556,671 bytes |
| Container | FLV; video stream only was kept |
| Codec (`ffprobe`) | `vp6f` — On2 VP6, Flash variant |
| Display resolution | 854x480, `yuv420p` (8-bit 4:2:0) |
| Coded resolution | 864x480 (16-aligned; see framing below) |
| Frame rate | ~24 fps (source ~1500 kbps HD clip) |

This is the canonical **Huffman-coded** VP6 sample from the FFmpeg samples pool
(the `huffman/` sub-directory). It is a high-motion action clip, so keyframes are
frequent and P-frames carry real motion — good for exercising the Huffman token
paths and inter mode decoding.

## What this fixture is

The first GOP-worth of frames starting at the keyframe at source PTS ≈ 1.998 s,
trimmed to **3 frames**: one I-frame followed by two P-frames.

| Idx | Type | VP6 bitstream length | FLV tag `byte0` | FLV tag `byte1` |
| --- | --- | --- | --- | --- |
| 0 | I-frame (keyframe) | 14,591 B | `0x14` (ftype=1, codecid=4=VP6) | `0xa0` (adj_h=10, adj_v=0) |
| 1 | P-frame (inter)   | 6,687 B  | `0x24` (ftype=2, codecid=4)     | `0xa0` |
| 2 | P-frame (inter)   | 6,946 B  | `0x24` (ftype=2, codecid=4)     | `0xa0` |

Two P-frames (not one) so that the 2nd P-frame's spatial neighbours include
already-decoded P-MBs carrying **real motion vectors** — the single-P case would
leave every neighbour tracing back to the intra keyframe (MV = 0), giving no
exercise of the Nearest/Near MV prediction scan.

## FLV → raw VP6 framing (exactly what was stripped)

Each FLV `VIDEODATA` tag body (`§E.4.3` of the FLV spec) is:

```
byte 0 : (FrameType << 4) | CodecID     ; CodecID 4 = VP6, high nibble 1=key 2=inter
byte 1 : VP6 dimension-adjustment       ; high nibble = horiz crop px, low nibble = vert crop px
byte 2..: raw VP6 elementary bitstream  ; the frame header (§9) starts here
```

For every frame here `byte1 = 0xa0` → **10 px cropped from the right, 0 from the
bottom**: coded 864x480 minus 10 = **854x480 display**. (854 is not a multiple of
8/16, so it cannot be a VP6 fragment count; the crop is what yields it.)

FFmpeg's FLV demuxer strips **both** bytes and hands `byte 2..` (the pure VP6
bitstream) to the decoder, signalling the 10 px crop out-of-band — verified: the
demuxed keyframe packet size is 14,591 = tag size (14,593) − 2.

## Frame-header bits decoded from `input.vp6` (in-repo spec §9, Table 1)

The header opens with raw `R()` bits (no BoolCoder needed). `input.vp6` begins
`79 30 00 e1 …`; `0x79` = `0111 1001`:

- `FrameType` `R(1)` = **0** → I-frame ✓
- `DctQMask`  `R(6)` = `111100` = **60** (near-max quantiser quality)
- `MultiStream` `R(1)` = **1** → **two partitions present** (prerequisite for a
  Huffman 2nd partition)

`MultiStream=1` + the `huffman/` provenance establish this as the **Huffman**
entropy path (`UseHuffman=1`; that flag is BoolCoder-`b(1)` coded and is resolved
by the crate's own header parse). This is exactly the path §13.2.2 / §13.3.3.2 /
§13.4 govern — a BoolCoder-only stream would not exercise them.

## §13 / §10 items this fixture can exercise / arbitrate

| Item | How the fixture hits it |
| --- | --- |
| **§13.2.2 Huffman Decoding DC Values** | Every block of all 3 Huffman-coded frames decodes its DC via the §13.2.2 path. Disambiguates the DC Huffman tree / value reconstruction against the oracle. |
| **§13.3.3.2 Decoding Huffman AC Zero Runs** | Real AC content (14.6 KB keyframe, ~6.8 KB P-frames) produces many AC zero-runs; the Huffman run-length branch (§13.3.3.2, distinct from the arithmetic §13.3.3.1) is exercised. |
| **§13.4 Decoding Huffman EOB and DC-0 Runs** | The Huffman EOB-run / DC-0-run tables (Figure 17) are driven by the multi-block frames. |
| **§10 Mode Decoding** | The 2 P-frames decode inter MB modes; the keyframe→P boundary lets a decoder observe mode-probability behaviour across a keyframe. |
| **§10 keyframe mode-prob contradiction (arbitration)** | A keyframe immediately precedes P-frames, so whichever way the crate initialises/reset s the keyframe mode-prob context, divergence from the oracle shows on P-frame 0's first MBs. Comparing `expected.yuv` frame 1 pixel-exactly arbitrates the contradiction. |
| **#155 FourMV Nearest/Near representative MV** (errata) | Any FourMV MB in P-frame 1 that becomes a neighbour in P-frame 2's Nearest/Near scan tests the "chroma-average MV" reading. |
| **VP6 dimension adjustment / non-16-aligned crop** | 864→854 crop via the FLV `0xa0` byte. |

## Decode oracle command (produces `expected.yuv`)

```
ffmpeg -i input.flv -f rawvideo -pix_fmt yuv420p expected.yuv
```

`expected.yuv` is the **display** output (854x480, cropped). A crate that decodes
the raw `input.vp6` keyframe through its decoder core produces the **coded**
864x480 frame; drop the rightmost 10 columns (adj_h=10) to match `expected.yuv`
frame 0. Decoding via `input.flv` (FLV demuxer applies the crop) matches
`expected.yuv` directly and is the recommended path.

## Reproduction

```
# 1. fetch full source (MD5 a10455decdb94f4e41343752dd22860a)
curl -O https://samples.ffmpeg.org/V-codecs/VP6/huffman/backcountry_bombshells_4min_HD_1500_96.flv
# 2. trim 3 frames from the keyframe at ~1.998 s, video only, no re-encode
ffmpeg -ss 1.998 -t 0.10 -i backcountry_bombshells_4min_HD_1500_96.flv \
       -map 0:v -c:v copy input.flv
# 3. raw keyframe bitstream = FLV video tag 0 body, bytes 2.. (strip byte0+byte1)
#    -> input.vp6   (see FLV framing above)
# 4. oracle decode
ffmpeg -i input.flv -f rawvideo -pix_fmt yuv420p expected.yuv
```

## Artifacts

| File | Size | Notes |
| --- | --- | --- |
| `input.flv` | 28,523 B | Trimmed FLV, video-only, 3 frames (I + P + P). Self-contained; standard container. |
| `input.vp6` | 14,591 B | Raw VP6 keyframe elementary bitstream (FLV tag 0 body, bytes 2..). Decoder-core input; frame header §9 starts at byte 0. |
| `expected.yuv` | 1,844,640 B | 3 × 854×480 `yuv420p` = 3 × 614,880. Oracle decode of `input.flv`. |

SHA-256:

```
input.flv     161eddeb7f2318f08d938286b66be0128b7b53ba859f353a32b2995cb5e6a436
input.vp6     2192efea0c66d6e4ea285c8b41160df494bea962b8d658e4d21c1a83799bf737
expected.yuv  3e24da31aa790e085c265dd596223af3d2f66862843ec7e77072a00d8f079cbb
```

## Spec cross-references (in-repo `vp6_format.pdf`, On2, 2006-08-17)

- §9 Frame Header (Table 1 / Table 2 IntraHeader) — `FrameType`, `DctQMask`,
  `MultiStream`, `UseHuffman`.
- §6 Data Partitioning — two-partition transport; `Buff2Offset`.
- §7.2 Huffman Decoder.
- §10 Mode Decoding — inter mode probabilities; keyframe mode-prob question.
- §13.2.2 Huffman Decoding DC Values.
- §13.3.3.2 Decoding Huffman AC Zero Runs.
- §13.4 Decoding Huffman EOB and DC-0 Runs.

---

## Round-411 investigation appendix — whole-frame Huffman conformance

All findings below were derived from this fixture only: the staged spec
(`docs/video/vp6/vp6_format.pdf`), the oracle `expected.yuv`, and
**differential bit-flip probing** of the black-box decode oracle (flip a
single bit of the keyframe's second partition inside `input.flv`, decode
with the oracle binary, diff the YUV output — the first block whose
pixels change identifies which block owns that bit). The oracle pairing
was re-verified this round: a fresh `ffmpeg -i input.flv -f rawvideo
-pix_fmt yuv420p` decode is byte-identical to `expected.yuv`.

### 1. §16 IDCT descale rounding (LANDED — corrects the r390 note)

Every §16 rounding combination was tested by searching, per non-uniform
oracle display block, for an integer quantized-coefficient block that
reconstructs the oracle pixels exactly (dequant at `DctQMask == 60`:
DC 12 / AC 16, then §16 IDCT, then §17.1 `+128` + clamp):

| multiply descale | final descale | luma blocks solving exactly |
|---|---|---|
| `>> 16` (floor, as printed) | `(x + 8) >> 4` | **555 / 555** |
| any other {floor, toward-zero, nearest} combination | | 0–33 / 555 |

The r390 "round toward zero" reading was under-determined: flat DC-only
blocks (the only evidence r390 had) reconstruct to 16 under several
roundings. Gated by `keyframe_content_blocks_reconstruct_pixel_exact`.

### 2. Bit→block ownership map of the keyframe's opening (bit-flip probe)

Bit positions are relative to the start of partition 2 (byte 225 of
`input.vp6`). "structural" = a flipped bit makes the oracle decoder
refuse/short the frame (token codewords, run fields).

| bits | owner |
|---|---|
| 0..3 | Y(0,0) DC codeword (`1100`, CATEGORY6 in the retrained luma DC tree) |
| 4..14 | Y(0,0) DC `R(11)` magnitude = 232 (DC −299) |
| 15 | Y(0,0) DC sign |
| 16..18 | Y(0,0) AC-EOB codeword (`111`, prec-2 band-0) |
| 19..27 | §13.4 EOB run = 74 |
| 28..29 | Y(0,1) DC ZERO codeword (`00`) |
| 30..38 | §13.4 DC-zero run = 74 |
| 39..42 | **U(0,0) DC codeword `1100` = CATEGORY6 in the *luma* tree shape** |
| 43..53 | U(0,0) DC `R(11)` magnitude = 61 (value −(67+61) = **−128**) |
| 54 | U(0,0) DC sign (flip ⇒ +96 px on U(0,0) = 2×128 DC quanta — confirms \|DC\| = 128) |
| 55..58 | U(0,0) AC-EOB codeword |
| 59..67 | §13.4 EOB run = 74 |
| 68..71 | **V(0,0) DC codeword `1100`** |
| 72..83 | V(0,0) DC magnitude/sign — identical −128 field |
| 84..115 | structural (V AC-EOB + run, further §13.4 runs covering the uniform prefix) |
| 116..121 | Y(0,62) DC ONE token (+1 — a "hidden" DC delta rendering identically to 16) |
| 121..128 | Y(0,63) DC = CATEGORY4, magnitude bits 5, `+24` delta — decodes exactly against predictor −298 |
| 129.. | Y(0,63) AC tokens (still misdecoding under our derived AC trees — open) |

Per-bit flip differentials on U(0,0) read out an exact binary-weighted
magnitude ladder (256/128/64/32/16/8/4/2/1 DC quanta at 0.375 px per
quantum), pinning the CATEGORY6 `R(11)` field layout.

### 3. Chroma DC findings (validated in an experimental decode walk)

* **Chroma DC Huffman tree = the luma-bank tree.** The chroma DC reads
  use codewords matching the tree built from the *retrained plane-0
  (luma)* DC node probabilities — not the untouched all-128 chroma bank
  (this stream's Figure-5 pass updates only the luma DC nodes). The
  §13.2.2 `DcHuffTree[2]` derivation for the chroma plane therefore
  does not consume `DCProbs[1]` the way its prose suggests. Open: the
  precise mechanism (shared tree vs a different Table 25 indexing vs
  chroma-bank update semantics).
* **Chroma DC prediction seeds at +128.** U(0,0)/V(0,0) each carry a
  coded DC delta of −128 yet reconstruct to exactly 128 (coded DC 0):
  the §14 "last decoded DC" frame-start seed for the chroma planes is
  `+128` in the quantized-DC domain, not the zero §14's prose states.
  (`DcPredictionContext::new_chroma` / `CHROMA_DC_PREDICTION_SEED`
  landed; not yet wired into the shared drivers — see below.)
* With both applied experimentally, the whole uniform prefix of the
  keyframe (the first 31 macroblocks, including a hidden `+1` DC at
  Y(0,62)) parses and reconstructs pixel-exactly, and the first content
  block's DC token decodes exactly (`+24` at bits 121..128). The parse
  then diverges at Y(0,63)'s **AC** tokens: the §13.3.2 AC Huffman
  trees our §13.1 conversion derives do not match the stream (first
  mismatch: true CATEGORY2(−7) read as FIVE under
  `AcHuffTree[Y][prec2][band0]`). Known-plaintext solving of the AC
  codewords is the next step.

### 4. Why the driver wiring is deferred

Wiring `new_chroma` + the luma-bank chroma tree into the shared
decode/encode drivers currently breaks three arithmetic-path round-trip
tests — investigation shows those failures expose a **pre-existing**
encoder/decoder fidelity bug on the arithmetic path (a fully-reverted
tree round-trips a 32×32 gradient with worst-case sample errors of ~189
while still clearing the suite's PSNR floors). That bug must be fixed
first so the seed change can land with the round-trips staying exact.

### Reproduction (bit-flip probe)

```
# flip bit N of partition 2 inside the keyframe tag of input.flv,
# decode frame 0, diff against expected.yuv frame 0
python3 - <<PY
flv = bytearray(open('input.flv','rb').read())
vp6 = open('input.vp6','rb').read()
off = bytes(flv).find(vp6) + 225   # partition 2
N = 43                             # bit to flip
flv[off + N//8] ^= 1 << (7 - N%8)
open('mut.flv','wb').write(flv)
PY
ffmpeg -y -i mut.flv -frames:v 1 -f rawvideo -pix_fmt yuv420p mut.yuv
cmp mut.yuv expected.yuv
```

## Appendix B — round 447 P-frame pixel-arbitration data

The round-447 campaign recovered ground truth for the first P-frame
(frame 1) by inverting the reconstruction pipeline against the
bit-exact decoded keyframe: for a candidate (prediction, motion
vector), the residual `oracle − prediction` was forward-transformed,
quantised at `DctQMask == 60` and accepted only when the rounded
integer coefficients reproduce the oracle samples exactly through the
in-tree §15/§16/§17 chain. Facts established (all display-region,
¼-pel units):

* **MB (0,31)** — single-MV inter, MV in `{(-1..1, 24..26)}` (interior
  ambiguity from locally smooth content; all candidates share the
  visible prediction), ~30 non-zero quantised coefficients. Its
  bottom-right luma block (Y3) opens with a CATEGORY5 DC of delta 54
  followed by 16 AC coefficients — the datum that arbitrates the
  Table 18 extra-bit probability pairing (see the round-447 CHANGELOG
  entry and the `pframe_first_content_mb_tokens_decode_exact` gate).
* **MB (0,32)** — single-MV inter, MV in `{(-1..1, 24..25)}`, 13
  non-zero coefficients; its §13 tokens decode exactly under the
  corrected pairing, with the wire mode most plausibly a
  Nearest-class reuse of (0,31)'s vector.
* Other pixel-pinned motion samples: (1,7) ≈ (0, −24..−25);
  (2,6)/(2,51) ≈ (−28..−29, 0..1); (2,43) ≈ (−1..0, 24); (3,9) =
  (0, −24) uniquely; (2,7) is FourMV-shaped (per-block solutions
  include (0, −24) on its top-right luma block with the others near
  zero or far-field ≈ (x, −54)). A full-frame map (zero-MV inter for
  ~1431 MBs, ~108 searched single-MV, ~51 FourMV-shaped, ~30 intra)
  was derived and is reproducible from the fixture alone.

**Wire-reading exclusions (partition 1).** Decoding the §9 InterHeader
+ §10/§11.2/Figure-5 prefix and then walking all 1620 macroblocks'
prediction data consumes 653 bytes under the crate's printed-spec
reading, against a partition-1 budget of `Buff2Offset − prefix = 473`
bytes — so the §10/§11.1 wire reading is wrong beyond the zero-motion
prefix (first divergence: the §11.1 decode at MB (0,31) yields
(−5, −73) where pixels require ≈ (1, 24); the first mode-level
divergence sits at MB (0,32), the frame's first
`NearestOnly`-availability macroblock). An exhaustive screen of
24 576 reading pairs — 384 §10 variants (same-as-last polarity ×
weight source × four `probModeSame` forms × six availability-row
permutations × branch polarity × own-weight zeroing) × 64 §11.1
variants (component order, sign position, short/long flag polarity,
long-magnitude bit order, implicit-bit-3 handling, sign-on-zero) —
found **no** combination that both lands in the byte-budget window
and decodes the pixel-true motion at MB (0,31): the 188
budget-window hits all fail the pixel screen at MB 31, and the
combinations that decode (1, 24) at MB 31 overrun the budget. The
operative reading therefore differs outside this space (candidates:
the Nearest/Near walk semantics feeding availability and implicit-MV
reuse, the golden modes' reuse source, the §10 VQ/update grammar, or
the §11.2 MV-probability update), and needs the staged decoder
extraction to settle — the extraction record explicitly leaves
P-frames un-established.

**§13.2/§14 bookkeeping observations (partition 2).** With
pixel-derived prediction info substituted for pass 1, the partition-2
arithmetic token stream was constraint-solved under several Table-26
context readings. No single tested rule (neighbour reconstructed-DC
zero-ness — the crate's current reading; transmitted-delta zero-ness;
reference-gated variants; mixed within-MB/completed-MB forms)
reconciles the whole frame: each reading decodes exactly for hundreds
of macroblocks and then requires a contradictory choice at a specific
block, with per-macroblock reference buckets (previous vs golden —
pixel-identical on the first P-frame, but §14-distinct) as the
coupling unknown. The discriminating blocks are recorded in the
round-447 report; the question is directly answerable by static
extraction of the vendor decoder's DC-context selection and §14
bookkeeping.

## Appendix C — round 450 P-frame §10/§11.1 wire characterisation

Round 450 attacked the open P-frame blocker with two black-box methods
over the decode oracle (no third-party decoder source read; the oracle
YUV is decoded *output data*):

1. **Partition-1 single-bit-flip ownership.** Each bit of the first
   P-frame's partition 1 was flipped in `input.flv`, the frame decoded
   with the oracle binary, and the first macroblock whose pixels changed
   recorded. This maps the mode/MV wire onto the MB grid and gives a
   staircase of exact wire boundaries (the first bit whose flip perturbs
   MB *m* or later), which any candidate reading must reproduce.
2. **Whole-partition synthesis.** A semantically-identical partition 1
   is re-encoded through the crate's own encoders (header tail, the
   `VP6_ModeVq[1][13]` baseline reset the frame actually carries, no MV
   updates, the coefficient-probability updates re-emitted current→
   target), then an arbitrary script of BoolCoder `(probability, bit)`
   symbols is appended and the original partition 2 spliced back on.
   Feeding the result through the oracle and scoring by whole-frame
   re-synchronisation (how many later, static macroblocks still match a
   pure-static baseline decode) turns the oracle into a yes/no judge of
   any hypothesised wire fragment: a scripted symbol sequence that
   *exactly matches the vendor grammar's node probabilities* is read back
   unchanged and keeps full sync, while any mismatch drifts the decode.

### Established (all oracle-verified)

* **The §10 mode-decode grammar is correct as the crate reads it.** A
  scripted mode layer — root `B(probModeSame)` "same as last" bit, the
  Figure-10 nine-node tree, the `ModeAvailability` neighbour walk, and
  the `VP6_BaselineXmittedProbs` / `VP6_ModeVq` banks — re-synchronises
  the *entire* frame (1580/1580 non-prefix macroblocks) when driven
  through the synthesis harness. The mode wire is not the blocker.
* **MB (0,31) is `CODE_INTER_PLUS_MV`, motion `(-1, 24)` ¼-pel.** The
  mode decodes correctly from the real wire (the crate already reports
  `InterPlusMv` there); the motion is pinned by pixel reconstruction
  (unique horizontal, `|x| ≤ 1`, with the round-447 vertical band
  `24..26` narrowed to 24 by the synthesis cross-checks) and is the
  vector under which the new `pframe_mb31_inter_reconstruction_pixel_exact`
  gate reconstructs the macroblock bit-exactly.

### The open defect — §11.1 motion-vector component grammar

The crate decodes MB (0,31)'s motion as `(-5, -73)`; the wire carries
`(-1, 24)`. Two discriminated symptoms:

* **Component order / axis.** The first-decoded MV component (read with
  the `IsMvShortProbs[0]`/`ShortMvProbs[0]` bank) yields **24**, which is
  the *vertical* motion — not the horizontal that §11.1's "When decoding
  a motion vector the X component is decoded first" implies. The crate
  assembles the first-decoded component as the x field, which transposes
  the reconstructed vector.
* **Short-vector magnitude.** From the coder state after the first
  component, the second component's bits drive the printed Figure-11
  short tree along `>3 = 0, >1 = 1, >2 = 0`, which the §11.1 pseudo-code
  evaluates to magnitude **2**. The reconstruction admits only a
  horizontal magnitude `≤ 1` at this macroblock, so the printed tree
  over-reads by one. A search over short-tree node-probability
  assignments recovers magnitude 1 (and MB (0,31)'s exact `(-1, 24)`)
  under a specific remapping, but that single remapping does not extend
  cleanly past the first row of content macroblocks, so the full
  component grammar is **not** closed here.

### DOCS-GAP — Extractor round 3 ask

Settling the §11.1 wire from the spec plus this one fixture is
under-determined; it is the `provenance/03` "P-frames un-established"
item. A clean-room static extraction of the vendor decoder's motion-
vector decode should record, as it did for the Huffman residual path:
the exact order and axis of the two component decodes and how each maps
to the reconstructed vector; the operative short-vector tree node
structure and its `ShortMvProbs` node-to-index mapping (the printed
Figure-11 reading over-reads by one on the arbitration datum); the
long-vector bit order and the implicit-bit-3 rule as the decoder applies
them; the §11 differential-reference selection (which neighbour supplies
the predictor, and when a new vector is coded absolutely); and the §10
`ModeAvailability` neighbour-walk length and same-reference gating that
feeds `probXmitted` row selection. The mode grammar above can be treated
as confirmed; the extraction need only settle §11.1 and the §10
availability walk.

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
</content>
</invoke>

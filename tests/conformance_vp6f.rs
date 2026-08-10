//! Third-party conformance gates — vp6f Huffman I+P fixture.
//!
//! `tests/fixtures/vp6f-huffman-i-then-p-854x480/` holds the first GOP
//! (1 I-frame + 2 P-frames) of a conformant third-party Flash VP6
//! (`vp6f`) stream — the keyframe on the **Huffman** entropy path
//! (`MultiStream == 1`, `UseHuffman == 1`), the two P-frames on the
//! MultiStream **arithmetic** path — 854x480 display / 864x480 coded,
//! together with the black-box decode-oracle output (`expected.yuv`,
//! yuv420p). See the fixture's `notes.md` for full provenance. This is
//! the crate's first real-encoder stream; the gates below pin the
//! fixture-arbitrated readings so they cannot regress:
//!
//! 1. §9 Table 2 geometry is transmitted in **macroblock units** (the
//!    printed "8x8 block units" prose is an erratum).
//! 2. Partition 1's BoolCoder reads past `Buff2Offset` (the §7.3
//!    pseudo-code has no end-of-partition check; the real encoder
//!    sizes partition 1 tightly and the coder's 32-bit look-ahead
//!    legitimately renormalizes into the first partition-2 byte).
//! 3. The §8 Figure-5 sub-stream of a real keyframe carries live DC
//!    node updates, a custom §12.2 scan order, and AC updates — filled
//!    into the banks under the **keyframe carry-forward rule** (staged
//!    errata `#277 part 7`): a clear DC/AC update flag writes the
//!    shared 11-slot running vector's value, so every DC/AC entry is
//!    written and the untouched chroma DC bank comes out a copy of the
//!    retrained luma bank.
//! 4. The §16 IDCT descale rounding (round-411 correction): the
//!    per-multiply `>> 16` descales are arithmetic shifts exactly as
//!    printed, and the final column-pass descale is `(x + 8) >> 4` — a
//!    rounding add the printed listing omits. Arbitrated by AC-carrying
//!    oracle blocks (`keyframe_content_blocks_reconstruct_pixel_exact`).
//! 5. **The whole keyframe decodes pixel-exactly** through the
//!    top-level `Vp6Decoder` (`keyframe_decodes_pixel_exact`): all
//!    9720 blocks — every luma, U and V sample of the 854x480 display
//!    region — match the black-box oracle bit-for-bit. This gate
//!    closes the former full-frame Huffman blocker and jointly pins
//!    the round-439 corrections: the operative §7.2.1 tree
//!    construction (errata `#277 part 3, closed`), the keyframe
//!    carry-forward banks (`#277 part 7`), the printed 12-leaf §13.1
//!    DC mapping (superseding the earlier node-0 fold, which was a
//!    compensating misreading fitted against the literal banks), the
//!    §13.3.1 magnitude-based `Prec` seed, the §13.2.2/§13.3.3.2 run
//!    conventions (`#193 parts 1+2, closed`), the §14 toward-zero
//!    two-neighbour average, and the §14 chroma DC seed (+128 in the
//!    quantized-DC domain, Intra bucket).
//!
//! The two P-frames do **not** yet decode pixel-exactly: their §10
//! mode / §11 MV / arithmetic-coefficient wire semantics diverge at the
//! first content macroblock (the static prefix decodes exactly). The
//! staged extraction record (`provenance/03-extractor-binary-huffman.md`)
//! explicitly leaves P-frames un-established; closing them needs a
//! behavioural P-frame trace.

use oxideav_vp6::bool_coder::BoolCoder;
use oxideav_vp6::coeff_prob_update::{decode_coefficient_prob_updates_keyframe, CoeffProbBanks};
use oxideav_vp6::decode_frame::Vp6Decoder;
use oxideav_vp6::dequant::DequantContext;
use oxideav_vp6::frame_header::{CodingProfile, Vp3Version, Vp6FrameHeader, Vp6HeaderTail};
use oxideav_vp6::scan_update::DEFAULT_BAND_ASSIGNMENT;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vp6f-huffman-i-then-p-854x480"
);

/// Display geometry (after the container's 10-px right crop).
const DISPLAY_W: usize = 854;
const DISPLAY_H: usize = 480;
const YUV_FRAME_LEN: usize = DISPLAY_W * DISPLAY_H + 2 * (DISPLAY_W / 2) * (DISPLAY_H / 2);

fn load(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURE_DIR}/{name}")).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

/// Fixture integrity: sizes match the notes' inventory.
#[test]
fn fixture_inventory() {
    assert_eq!(load("input.vp6").len(), 14_591);
    assert_eq!(load("input.flv").len(), 28_523);
    assert_eq!(load("expected.yuv").len(), 3 * YUV_FRAME_LEN);
}

/// Minimal FLV tag walk over `input.flv`: three VP6 (`CodecID == 4`)
/// video tags (I + P + P), each with the VP6 dimension-adjust byte
/// signalling the 10-px right crop, and the keyframe body (bytes `2..`)
/// byte-identical to `input.vp6`.
#[test]
fn flv_transport_framing() {
    let flv = load("input.flv");
    assert_eq!(&flv[0..3], b"FLV", "FLV signature");
    let data_offset = u32::from_be_bytes([flv[5], flv[6], flv[7], flv[8]]) as usize;
    let mut pos = data_offset + 4; // skip PreviousTagSize0
    let mut video_bodies: Vec<(u8, u8, Vec<u8>)> = Vec::new();
    while pos + 11 <= flv.len() {
        let tag_type = flv[pos];
        let data_size = ((flv[pos + 1] as usize) << 16)
            | ((flv[pos + 2] as usize) << 8)
            | flv[pos + 3] as usize;
        let body = &flv[pos + 11..pos + 11 + data_size];
        if tag_type == 9 {
            video_bodies.push((body[0], body[1], body[2..].to_vec()));
        }
        pos += 11 + data_size + 4;
    }
    assert_eq!(video_bodies.len(), 3, "I + P + P");
    // byte0 = (FrameType << 4) | CodecID; byte1 = (adj_h << 4) | adj_v.
    assert_eq!(video_bodies[0].0, 0x14, "keyframe, CodecID 4 = VP6");
    assert_eq!(video_bodies[1].0, 0x24, "inter frame");
    assert_eq!(video_bodies[2].0, 0x24, "inter frame");
    for (_, adj, _) in &video_bodies {
        assert_eq!(*adj, 0xa0, "10 px right crop, 0 px bottom");
    }
    assert_eq!(
        video_bodies[0].2,
        load("input.vp6"),
        "keyframe body == raw elementary fixture"
    );
    // Every VP6 payload begins with a parseable §9 header; the two
    // inter frames must open as FrameType == Inter (Table 1) and see
    // the keyframe-inherited profile for their Buff2Offset gate.
    let key_hdr = Vp6FrameHeader::parse(&video_bodies[0].2).expect("keyframe header");
    for (_, _, body) in &video_bodies[1..] {
        let hdr = Vp6FrameHeader::parse_with_profile(body, key_hdr.profile).expect("inter header");
        assert!(!hdr.is_keyframe);
    }
}

/// §9 header of the real keyframe: the Table 2 geometry fields carry
/// **macroblock** counts (54x30 for the 864x480 coded frame) — the
/// fixture-arbitrated unit erratum — alongside the Huffman
/// multi-stream signalling this fixture was chosen for.
#[test]
fn keyframe_header_geometry_is_in_macroblock_units() {
    let raw = load("input.vp6");
    let hdr = Vp6FrameHeader::parse(&raw).expect("§9 raw prefix");
    assert!(hdr.is_keyframe);
    assert_eq!(hdr.dct_q_mask, 60);
    assert!(hdr.multi_stream, "two partitions (Huffman prerequisite)");
    assert_eq!(hdr.profile, Some(CodingProfile::Simple));
    assert_eq!(hdr.version, Some(Vp3Version::Vp60));
    assert_eq!(hdr.buff2_offset, Some(225));

    let mut bc = BoolCoder::new(&raw[hdr.raw_prefix_len..]).expect("partition 1");
    let tail = Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
        .expect("§9 tail");
    // 864x480 coded = 54x30 macroblocks = 108x60 8x8 luma blocks.
    assert_eq!(tail.h_fragments, Some(54), "MB cols (NOT 8x8-block cols)");
    assert_eq!(tail.v_fragments, Some(30), "MB rows");
    assert_eq!(tail.output_h_fragments, Some(54));
    assert_eq!(tail.output_v_fragments, Some(30));
    assert!(tail.use_huffman, "Huffman second partition");
}

/// The §8 Figure-5 probability-update sub-stream of the real keyframe
/// parses to completion under the keyframe carry-forward rule — pinning
/// that partition 1's BoolCoder spans past `Buff2Offset` (the pass
/// legitimately renormalizes one byte into partition 2), that the
/// stream carries live updates (retrained luma DC nodes, a custom
/// §12.2 scan order, AC updates), and that the carry-forward fills the
/// untouched chroma DC bank with a copy of the retrained luma bank
/// (errata `#277 part 7` — the staged corrected-banks table's
/// `DcProbs` rows, re-derived here from the bitstream).
#[test]
fn keyframe_prob_updates_parse_and_span_partition_boundary() {
    let raw = load("input.vp6");
    let hdr = Vp6FrameHeader::parse(&raw).unwrap();
    let p1_end = hdr.buff2_offset.unwrap() as usize - hdr.raw_prefix_len;

    // Sliced at Buff2Offset the Figure-5 pass runs out of bytes...
    let mut sliced = BoolCoder::new(&raw[hdr.raw_prefix_len..hdr.buff2_offset.unwrap() as usize])
        .expect("partition 1 slice");
    let _ = Vp6HeaderTail::parse_with(
        &mut sliced,
        true,
        hdr.profile.unwrap(),
        hdr.version.unwrap(),
    )
    .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    assert!(
        decode_coefficient_prob_updates_keyframe(&mut sliced, &mut banks).is_err(),
        "tightly-sized partition 1 exhausts mid-pass when sliced at Buff2Offset"
    );

    // ...while the full-span coder (what `decode_packet` builds)
    // completes exactly one byte past the boundary.
    let mut bc = BoolCoder::new(&raw[hdr.raw_prefix_len..]).unwrap();
    let _ = Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
        .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    decode_coefficient_prob_updates_keyframe(&mut bc, &mut banks).expect("Figure-5 pass");
    assert_eq!(
        bc.pos(),
        p1_end + 1,
        "the pass needs exactly one look-ahead byte past Buff2Offset"
    );

    // Live content under the carry-forward: the luma DC row carries
    // this frame's six retrained nodes plus carried 128s, and the
    // chroma row — which receives no updates of its own in this frame
    // — inherits the running vector, coming out an exact copy of luma.
    let expected_dc = [52u8, 1, 30, 36, 128, 128, 50, 128, 128, 128, 180];
    assert_eq!(banks.dc_probs[0], expected_dc, "luma DC row (carry-filled)");
    assert_eq!(
        banks.dc_probs[1], banks.dc_probs[0],
        "chroma DC row inherits the shared carry vector"
    );
    assert_ne!(
        banks.band_assignment, DEFAULT_BAND_ASSIGNMENT,
        "custom §12.2 scan order"
    );
    assert_ne!(
        banks.ac_probs,
        CoeffProbBanks::keyframe().ac_probs,
        "AC nodes retrained"
    );
}

/// §16 IDCT descale rounding, arbitrated by AC-carrying oracle blocks.
///
/// Flat DC-only blocks cannot distinguish the §16 descale roundings
/// (a flat black block reconstructs to luma 16 under several of
/// them), but blocks carrying AC coefficients can. For three of the
/// oracle keyframe's non-uniform display blocks, the quantized
/// coefficient sets below reproduce the oracle pixels **exactly**
/// under the operative §16 rounding — per-multiply descale `>> 16`
/// exactly as printed (arithmetic shift) and final descale
/// `(x + 8) >> 4` (a rounding add the printed listing omits) — and
/// under no other combination of {floor, truncate-toward-zero,
/// round-nearest} multiply/final descales. (Exhaustively checked
/// against the oracle: **all 555** non-uniform luma display blocks
/// admit exact integer coefficient solutions only under this
/// combination; the previous toward-zero reading left every one of
/// them with an irreducible residual.)
///
/// The coefficient sets were recovered from the oracle by inverting
/// the §15/§16 pipeline (forward transform + quantization at
/// `DctQMask == 60`: DC factor 12, AC factor 16) and verifying the
/// reconstruction is bit-exact; raster-order `(position, value)`
/// pairs.
#[test]
fn keyframe_content_blocks_reconstruct_pixel_exact() {
    let expected = load("expected.yuv");
    let oy = &expected[..DISPLAY_W * DISPLAY_H];
    let dequant = DequantContext::new(60);

    // (block_row, block_col, raster-order nonzero quantized coeffs)
    type Case = (usize, usize, &'static [(usize, i32)]);
    let cases: [Case; 3] = [
        (
            0,
            63,
            &[
                (0, -274),
                (1, -7),
                (2, -3),
                (8, 17),
                (9, -6),
                (10, -2),
                (16, 4),
                (17, -1),
            ],
        ),
        (
            0,
            64,
            &[
                (0, -292),
                (1, 5),
                (2, 2),
                (3, 1),
                (8, 5),
                (9, 5),
                (10, 2),
                (16, 1),
                (17, 1),
            ],
        ),
        (
            4,
            88,
            &[
                (0, -258),
                (1, -5),
                (2, -9),
                (5, -1),
                (6, -1),
                (8, -13),
                (9, 2),
                (10, 3),
                (16, -8),
                (17, 1),
                (18, 2),
                (24, 3),
                (26, -1),
                (40, -1),
            ],
        ),
    ];

    for (br, bc, coeffs) in cases {
        let mut raster = [0i32; 64];
        for &(pos, q) in coeffs {
            let factor = if pos == 0 {
                dequant.dc_factor
            } else {
                dequant.ac_factor
            } as i32;
            raster[pos] = q * factor;
        }
        oxideav_vp6::idct::idct_block(&mut raster);
        let pix = oxideav_vp6::reconstruct::intra_block_to_pixels(&raster);
        for r in 0..8 {
            for c in 0..8 {
                let (x, y) = (bc * 8 + c, br * 8 + r);
                assert_eq!(
                    pix[r * 8 + c],
                    oy[y * DISPLAY_W + x],
                    "block ({br},{bc}) sample ({x},{y})"
                );
            }
        }
    }
}

/// **The whole keyframe decodes pixel-exactly** through the top-level
/// `Vp6Decoder`: every one of the 854x480 display luma samples and
/// every 427x240 U and V sample matches the black-box decode oracle
/// bit-for-bit.
///
/// This is the crate's strongest single gate. Because the frame is one
/// serial Huffman parse with no resynchronisation points, any error in
/// the §9 header, the keyframe carry-forward Figure-5 fill, the §7.2.1
/// tree construction, the §13.1 leaf mapping, the Table-36 band map,
/// the §13.2.2/§13.3.2/§13.4 cross-block run bookkeeping, the §13.3.1
/// `Prec` seeding, the §14 DC prediction (toward-zero average + chroma
/// +128 seed), the §12.2 custom scan, the §15 dequant or the §16 IDCT
/// destroys the agreement within a few macroblocks.
#[test]
fn keyframe_decodes_pixel_exact() {
    let raw = load("input.vp6");
    let expected = load("expected.yuv");
    let oy = &expected[..DISPLAY_W * DISPLAY_H];
    let ou =
        &expected[DISPLAY_W * DISPLAY_H..DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)];
    let ov = &expected[DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)..YUV_FRAME_LEN];

    let mut dec = Vp6Decoder::new();
    let frame = dec.decode_packet(&raw).expect("whole-keyframe decode");

    // 864x480 coded; the display region is the left 854 columns (the
    // container's 10-px right crop).
    assert_eq!(frame.y.width(), 864);
    assert_eq!(frame.y.height(), 480);

    let yw = frame.y.width();
    for y in 0..DISPLAY_H {
        for x in 0..DISPLAY_W {
            assert_eq!(
                frame.y.samples()[y * yw + x],
                oy[y * DISPLAY_W + x],
                "luma mismatch at ({x},{y})"
            );
        }
    }
    let cw = frame.u.width();
    for y in 0..DISPLAY_H / 2 {
        for x in 0..DISPLAY_W / 2 {
            assert_eq!(
                frame.u.samples()[y * cw + x],
                ou[y * (DISPLAY_W / 2) + x],
                "U mismatch at ({x},{y})"
            );
            assert_eq!(
                frame.v.samples()[y * cw + x],
                ov[y * (DISPLAY_W / 2) + x],
                "V mismatch at ({x},{y})"
            );
        }
    }
}

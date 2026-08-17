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

/// The three VP6 video-tag bodies (I + P + P) of `input.flv`, with the
/// two FLV per-frame prefix bytes stripped (see `flv_transport_framing`).
fn flv_video_bodies(flv: &[u8]) -> Vec<Vec<u8>> {
    let data_offset = u32::from_be_bytes([flv[5], flv[6], flv[7], flv[8]]) as usize;
    let mut pos = data_offset + 4;
    let mut out = Vec::new();
    while pos + 11 <= flv.len() {
        let tag_type = flv[pos];
        let data_size = ((flv[pos + 1] as usize) << 16)
            | ((flv[pos + 2] as usize) << 8)
            | flv[pos + 3] as usize;
        if tag_type == 9 {
            out.push(flv[pos + 11 + 2..pos + 11 + data_size].to_vec());
        }
        pos += 11 + data_size + 4;
    }
    out
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

/// §9 output scaling on the real stream: the keyframe transmits
/// `Output*Fragments == *Fragments` (54x30) with `ScalingMode == 0`
/// (`MAINTAIN_ASPECT_RATIO`, staged `tables/01`) — an **identity**
/// scaling description — so the scaled decode entry point must emit
/// bit-identical pixels at the identical coded geometry. Pins that
/// wiring the §9 scaling application into the driver cannot perturb a
/// conformant stream whose output geometry matches its coded geometry.
#[test]
fn keyframe_output_scaling_is_identity() {
    use oxideav_vp6::scaling::{FrameGeometry, OutputScaling, ScalingMode};

    let raw = load("input.vp6");
    let hdr = Vp6FrameHeader::parse(&raw).expect("§9 raw prefix");
    let mut bc = BoolCoder::new(&raw[hdr.raw_prefix_len..]).expect("partition 1");
    let tail = Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
        .expect("§9 tail");

    // Typed surface: 54x30 MB output, mode 0, identity vs the coded
    // geometry (erratum #338 macroblock units on both field pairs).
    let coded = tail.coded_geometry().expect("keyframe coded geometry");
    assert_eq!(coded, FrameGeometry::new(54, 30));
    let scaling = tail.output_scaling().expect("keyframe scaling fields");
    assert_eq!(
        scaling,
        OutputScaling::new(FrameGeometry::new(54, 30), ScalingMode::MaintainAspectRatio)
    );
    assert!(scaling.is_identity(coded));

    // The scaled decode is bit-identical to the unscaled decode.
    let mut dec_plain = Vp6Decoder::new();
    let plain = dec_plain.decode_packet(&raw).expect("decode");
    let mut dec_scaled = Vp6Decoder::new();
    let scaled = dec_scaled
        .decode_packet_scaled(&raw)
        .expect("decode scaled");
    assert_eq!(dec_scaled.output_scaling(), Some(scaling));
    assert_eq!(scaled.y.width(), 864);
    assert_eq!(scaled.y.samples(), plain.y.samples());
    assert_eq!(scaled.u.samples(), plain.u.samples());
    assert_eq!(scaled.v.samples(), plain.v.samples());
}

/// **P-frame partition-2 arithmetic tokens — the first content
/// macroblock decodes coefficient-exact** (round 447).
///
/// The fixture's first P-frame codes its coefficients on the
/// MultiStream **arithmetic** path (partition 2 BoolCoder at
/// `Buff2Offset`). Its letterboxed static prefix tokenises 189
/// consecutive all-zero blocks (every §13.2 Table 26 context is
/// both-zero there under any reading, since every decoded DC is 0),
/// then macroblock (0,31) carries the frame's first content. The
/// expected coefficient sets below were recovered from the oracle
/// frame (`expected.yuv` frame 1) by inverting the §15/§16
/// reconstruction pipeline against the bit-exact decoded keyframe
/// (integer-exact solutions, unique at the frame quantiser), so the
/// gate pins the *arithmetic* §13.2.1/§13.3.1 token path against
/// vendor-encoded wire data for the first time (the keyframe gate
/// exercises only the §7.2 Huffman transport, which reads its
/// category extra-bits as raw bits).
///
/// This arbitrated a new printed-spec defect (round 447): §13's
/// Table 18 lists each category's extra-bit probabilities in
/// **transmission order** (the first-listed probability codes the
/// most-significant magnitude bit — "the most significant bit of the
/// magnitude sent first … encoded with differing probabilities as
/// specified by the final column"), while the §13.2.1/§13.3.1
/// pseudo-code's `B(Probs[BitsCount])` with `BitsCount` descending
/// would pair the *last*-listed probability with the MSB. The
/// MSB-first pairing is operative: macroblock (0,31)'s bottom-right
/// luma block opens with a CATEGORY5 DC (delta magnitude 54 = 35 +
/// 0b10011) whose five magnitude bits decode to the oracle-recovered
/// value only under it, and every following AC token then lands
/// exactly; under the listing's pairing the same bits decode 59 and
/// the block (and frame) desynchronises.
#[test]
fn pframe_first_content_mb_tokens_decode_exact() {
    use oxideav_vp6::block_decode::decode_block_coefficients_ctx;
    use oxideav_vp6::coeff_prob_update::decode_coefficient_prob_updates;
    use oxideav_vp6::mode_prob_update::update_mode_probs;
    use oxideav_vp6::mv_prob_update::update_mv_probs;
    use oxideav_vp6::tokens::{AcPlane, DcContext};

    let flv = load("input.flv");
    let bodies = flv_video_bodies(&flv);

    // Keyframe: derive the persistent post-Figure-5 banks.
    let khdr = Vp6FrameHeader::parse(&bodies[0]).unwrap();
    let mut kbc = BoolCoder::new(&bodies[0][khdr.raw_prefix_len..]).unwrap();
    let _ = Vp6HeaderTail::parse_with(&mut kbc, true, khdr.profile.unwrap(), khdr.version.unwrap())
        .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    decode_coefficient_prob_updates_keyframe(&mut kbc, &mut banks).unwrap();

    // P-frame 1: run the §10/§11.2/Figure-5 update prefix to obtain the
    // frame's operative coefficient banks.
    let body = &bodies[1];
    let hdr = Vp6FrameHeader::parse_with_profile(body, khdr.profile).unwrap();
    assert!(hdr.multi_stream && !hdr.is_keyframe);
    let buff2 = hdr.buff2_offset.unwrap() as usize;
    let mut bc = BoolCoder::new(&body[hdr.raw_prefix_len..]).unwrap();
    let tail =
        Vp6HeaderTail::parse_with(&mut bc, false, khdr.profile.unwrap(), khdr.version.unwrap())
            .unwrap();
    assert!(!tail.use_huffman, "P-frames ride the arithmetic path");
    let mut mode_probs = oxideav_vp6::modes::VP6_BASELINE_XMITTED_PROBS;
    update_mode_probs(&mut bc, &mut mode_probs).unwrap();
    let mut mv_probs = [
        oxideav_vp6::mv_decode::MvProbs::defaults(oxideav_vp6::mv_decode::MV_AXIS_X),
        oxideav_vp6::mv_decode::MvProbs::defaults(oxideav_vp6::mv_decode::MV_AXIS_Y),
    ];
    update_mv_probs(&mut bc, &mut mv_probs).unwrap();
    let _scan = decode_coefficient_prob_updates(&mut bc, &mut banks).unwrap();
    let probs = banks.to_intra_probs();

    // Partition 2: 31 letterboxed MBs (186 all-zero blocks) + MB (0,31)'s
    // Y0 (still empty), then the three content blocks.
    let mut p2 = BoolCoder::new(&body[buff2..]).unwrap();
    let decode = |bc: &mut BoolCoder, plane: AcPlane, ctx: DcContext| -> Vec<(usize, i32)> {
        let b = decode_block_coefficients_ctx(
            bc,
            plane,
            &probs.dc_contexts,
            ctx,
            &probs.ac_probs,
            &probs.zrl_probs,
        )
        .expect("partition-2 block");
        b.coeffs
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect()
    };
    for blk in 0..31 * 6 + 1 {
        let plane = if blk % 6 < 4 { AcPlane::Y } else { AcPlane::UV };
        let nz = decode(&mut p2, plane, DcContext::BothZero);
        assert!(nz.is_empty(), "static-prefix block {blk} must be all-zero");
    }
    // MB (0,31) Y1/Y2 (left/above coded DCs still zero) then Y3 (both
    // neighbours' coded DCs non-zero). Values are scan-order (position,
    // value) pairs in DC-delta form, oracle-recovered.
    assert_eq!(
        decode(&mut p2, AcPlane::Y, DcContext::BothZero),
        vec![(0, 1), (1, -2), (3, 1), (9, -1), (10, 1)],
        "MB (0,31) Y1"
    );
    assert_eq!(
        decode(&mut p2, AcPlane::Y, DcContext::BothZero),
        vec![
            (0, 2),
            (1, -1),
            (2, -2),
            (3, -1),
            (4, 1),
            (5, 2),
            (6, -1),
            (8, 1)
        ],
        "MB (0,31) Y2"
    );
    assert_eq!(
        decode(&mut p2, AcPlane::Y, DcContext::BothNonZero),
        vec![
            (0, 54),
            (1, -10),
            (2, -6),
            (3, -8),
            (4, 1),
            (5, -8),
            (6, -1),
            (7, 2),
            (8, 1),
            (9, 3),
            (10, -2),
            (11, 1),
            (12, 1),
            (15, 1),
            (20, -1),
            (25, -1),
            (26, -1)
        ],
        "MB (0,31) Y3 — the CATEGORY5 DC that arbitrates the Table 18 \
         extra-bit probability pairing"
    );
}

/// **P-frame static prefix reconstructs pixel-exactly through the
/// two-pass MultiStream driver** (round 447).
///
/// The first P-frame's §10 mode / §11 MV wire is still un-established
/// past the first transmitted motion vector (macroblock (0,31); the
/// staged extraction record leaves P-frames un-established), so the
/// full frame cannot yet be gated. What *is* pinned: the §9
/// InterHeader parse, the §10/§11.2/Figure-5 update prefix, the pass-1
/// walk across all 1620 macroblocks, and — for the leading 31
/// macroblocks, whose §10 modes are zero-motion and whose §13 blocks
/// are all-zero — bit-exact reconstruction against the decode oracle.
#[test]
fn pframe_static_prefix_reconstructs_pixel_exact() {
    use oxideav_vp6::coeff_prob_update::decode_coefficient_prob_updates;
    use oxideav_vp6::coeff_source::CoeffSource;
    use oxideav_vp6::inter_frame::{
        decode_inter_frame_multistream_traced, FilterConfig, InterProbs, ReferenceFrames,
    };
    use oxideav_vp6::mode_prob_update::update_mode_probs;
    use oxideav_vp6::mv_prob_update::update_mv_probs;

    let flv = load("input.flv");
    let expected = load("expected.yuv");
    let bodies = flv_video_bodies(&flv);

    // Keyframe (bit-exact per `keyframe_decodes_pixel_exact`) seeds the
    // §4 references and the persistent banks.
    let mut dec = Vp6Decoder::new();
    let f0 = dec.decode_packet(&bodies[0]).expect("keyframe");
    let khdr = Vp6FrameHeader::parse(&bodies[0]).unwrap();
    let mut kbc = BoolCoder::new(&bodies[0][khdr.raw_prefix_len..]).unwrap();
    let _ = Vp6HeaderTail::parse_with(&mut kbc, true, khdr.profile.unwrap(), khdr.version.unwrap())
        .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    decode_coefficient_prob_updates_keyframe(&mut kbc, &mut banks).unwrap();
    let refs = ReferenceFrames::from_keyframe(f0);

    let body = &bodies[1];
    let hdr = Vp6FrameHeader::parse_with_profile(body, khdr.profile).unwrap();
    let buff2 = hdr.buff2_offset.unwrap() as usize;
    let mut bc = BoolCoder::new(&body[hdr.raw_prefix_len..]).unwrap();
    let tail =
        Vp6HeaderTail::parse_with(&mut bc, false, khdr.profile.unwrap(), khdr.version.unwrap())
            .expect("§9 InterHeader tail");
    let mut mode_probs = oxideav_vp6::modes::VP6_BASELINE_XMITTED_PROBS;
    update_mode_probs(&mut bc, &mut mode_probs).unwrap();
    let mut mv_probs = [
        oxideav_vp6::mv_decode::MvProbs::defaults(oxideav_vp6::mv_decode::MV_AXIS_X),
        oxideav_vp6::mv_decode::MvProbs::defaults(oxideav_vp6::mv_decode::MV_AXIS_Y),
    ];
    update_mv_probs(&mut bc, &mut mv_probs).unwrap();
    let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).unwrap();
    let probs = InterProbs {
        mode_probs,
        mv_probs,
        coeffs: banks.to_intra_probs(),
    };
    let filter = FilterConfig::from_header(&tail, hdr.dct_q_mask);
    let (prev, golden) = refs.bordered();
    let mut bc2 = BoolCoder::new(&body[buff2..]).unwrap();
    let mut src = CoeffSource::Bool(&mut bc2);
    let (hf, vf) = refs.coded_fragments();
    let trace = decode_inter_frame_multistream_traced(
        &mut bc,
        &mut src,
        hf,
        vf,
        hdr.dct_q_mask,
        &probs,
        &scan,
        &filter,
        &prev,
        &golden,
    );

    // Pass 1 walks the whole MB grid without erroring.
    assert!(trace.prediction_error.is_none(), "pass-1 walk completes");
    assert_eq!(trace.prediction.len(), 54 * 30);
    // The leading 31 macroblocks decode zero-motion modes...
    for m in trace.prediction.iter().take(31) {
        assert!(
            m.mb_mv.is_zero() && m.four_mvs.is_none(),
            "static prefix MBs carry no motion"
        );
    }
    // ...and reconstruct bit-exactly: luma rows 0..16 x cols 0..496,
    // chroma rows 0..8 x cols 0..248 (MBs (0,0)..=(0,30)).
    let o1 = &expected[YUV_FRAME_LEN..2 * YUV_FRAME_LEN];
    let oy = &o1[..DISPLAY_W * DISPLAY_H];
    let ou = &o1[DISPLAY_W * DISPLAY_H..DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)];
    let ov = &o1[DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)..];
    let yw = trace.frame.y.width();
    for y in 0..16 {
        for x in 0..496 {
            assert_eq!(
                trace.frame.y.samples()[y * yw + x],
                oy[y * DISPLAY_W + x],
                "P-frame static-prefix luma ({x},{y})"
            );
        }
    }
    let cw = trace.frame.u.width();
    for y in 0..8 {
        for x in 0..248 {
            assert_eq!(
                trace.frame.u.samples()[y * cw + x],
                ou[y * (DISPLAY_W / 2) + x],
                "P-frame static-prefix U ({x},{y})"
            );
            assert_eq!(
                trace.frame.v.samples()[y * cw + x],
                ov[y * (DISPLAY_W / 2) + x],
                "P-frame static-prefix V ({x},{y})"
            );
        }
    }
}

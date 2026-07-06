//! Third-party conformance gates — vp6f Huffman I+P fixture.
//!
//! `tests/fixtures/vp6f-huffman-i-then-p-854x480/` holds the first GOP
//! (1 I-frame + 2 P-frames) of a conformant third-party Flash VP6
//! (`vp6f`) stream on the **Huffman** entropy path (`MultiStream == 1`,
//! `UseHuffman == 1`), 854x480 display / 864x480 coded, together with
//! the black-box decode-oracle output (`expected.yuv`, yuv420p). See
//! the fixture's `notes.md` for full provenance. This is the crate's
//! first real-encoder stream, and it arbitrates several readings the
//! printed spec leaves wrong or open; each test below pins one of the
//! fixture-arbitrated behaviours so they cannot regress:
//!
//! 1. §9 Table 2 geometry is transmitted in **macroblock units** (the
//!    printed "8x8 block units" prose is an erratum).
//! 2. Partition 1's BoolCoder reads past `Buff2Offset` (the §7.3
//!    pseudo-code has no end-of-partition check; the real encoder
//!    sizes partition 1 tightly and the coder's 32-bit look-ahead
//!    legitimately renormalizes into the first partition-2 byte).
//! 3. The §8 Figure-5 sub-stream of a real keyframe carries live DC
//!    node updates, a custom §12.2 scan order, and AC updates.
//! 4. The §13.2.2 DC Huffman tree folds the node-0 left branch wholly
//!    into `ZERO_TOKEN` (EOB is forbidden in the DC position).
//! 5. The §16 IDCT descales round toward zero (truncating division),
//!    not toward -inf (the printed `>>` shifts).
//!
//! Full three-frame pixel conformance is the outstanding goal: the
//! keyframe currently decodes pixel-exactly through the leading
//! macroblocks (pinned below) but a §13 Huffman-side desync further in
//! is still under investigation, so the whole-frame gate is not yet
//! landed.

use oxideav_vp6::bool_coder::BoolCoder;
use oxideav_vp6::coeff_prob_update::{decode_coefficient_prob_updates, CoeffProbBanks};
use oxideav_vp6::coeff_source::CoeffSource;
use oxideav_vp6::dc_pred::{DcPredictionContext, Neighbour, ReferenceBucket};
use oxideav_vp6::dequant::DequantContext;
use oxideav_vp6::frame_header::{CodingProfile, Vp3Version, Vp6FrameHeader, Vp6HeaderTail};
use oxideav_vp6::huff_coeff::HuffmanCoeffTables;
use oxideav_vp6::scan_update::DEFAULT_BAND_ASSIGNMENT;
use oxideav_vp6::tokens::{AcPlane, DcContext};

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
/// parses to completion — pinning that partition 1's BoolCoder spans
/// past `Buff2Offset` (the pass legitimately renormalizes one byte into
/// partition 2) and that the stream carries live updates: retrained
/// luma DC nodes, a custom §12.2 scan order, and AC updates.
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
        decode_coefficient_prob_updates(&mut sliced, &mut banks).is_err(),
        "tightly-sized partition 1 exhausts mid-pass when sliced at Buff2Offset"
    );

    // ...while the full-span coder (what `decode_packet` now builds)
    // completes exactly one byte past the boundary.
    let mut bc = BoolCoder::new(&raw[hdr.raw_prefix_len..]).unwrap();
    let _ = Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
        .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    decode_coefficient_prob_updates(&mut bc, &mut banks).expect("Figure-5 pass");
    assert_eq!(
        bc.pos(),
        p1_end + 1,
        "the pass needs exactly one look-ahead byte past Buff2Offset"
    );

    // Live content: luma DC nodes retrained, chroma left at defaults,
    // a custom scan order, and retrained AC nodes.
    assert_ne!(banks.dc_probs[0], [128u8; 11], "luma DC nodes retrained");
    assert_eq!(banks.dc_probs[1], [128u8; 11], "chroma DC stays default");
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

/// Decode the leading macroblocks of the keyframe's Huffman coefficient
/// partition to pixels and compare against the decode oracle — the
/// first real-stream pixel-exactness this crate has had.
///
/// This pins, in one pass: the §13.2.2 DC-fold Huffman tree (block 0's
/// `DCT_VAL_CATEGORY6` = -299 and the chroma zero-DC runs are only
/// decodable under the fold), the §13.4 run decoding, the §15 dequant
/// at `DctQMask == 60`, the §14 DC prediction chain, and the §16
/// truncating (toward-zero) IDCT descales: `-299 * 12` reconstructs to
/// the oracle's luma 16 only with truncation (`>>` gives 15).
#[test]
fn keyframe_leading_macroblocks_decode_pixel_exact() {
    let raw = load("input.vp6");
    let expected = load("expected.yuv");
    let oy = &expected[..DISPLAY_W * DISPLAY_H];
    let ou =
        &expected[DISPLAY_W * DISPLAY_H..DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)];
    let ov = &expected[DISPLAY_W * DISPLAY_H + (DISPLAY_W / 2) * (DISPLAY_H / 2)..YUV_FRAME_LEN];

    let hdr = Vp6FrameHeader::parse(&raw).unwrap();
    let mut bc = BoolCoder::new(&raw[hdr.raw_prefix_len..]).unwrap();
    let _ = Vp6HeaderTail::parse_with(&mut bc, true, hdr.profile.unwrap(), hdr.version.unwrap())
        .unwrap();
    let mut banks = CoeffProbBanks::keyframe();
    let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).unwrap();
    let probs = banks.to_intra_probs();
    let tables = HuffmanCoeffTables::from_banks(&banks);
    let mut src = CoeffSource::huffman(&raw[hdr.buff2_offset.unwrap() as usize..], &tables);
    let dequant = DequantContext::new(hdr.dct_q_mask);

    // §14 per-plane DC prediction state over the walked prefix.
    struct PlaneState {
        dc: Vec<Option<i32>>,
        cols: usize,
        pred: DcPredictionContext,
    }
    impl PlaneState {
        fn new(cols: usize, rows: usize) -> Self {
            Self {
                dc: vec![None; cols * rows],
                cols,
                pred: DcPredictionContext::new(),
            }
        }
    }
    let decode_block = |src: &mut CoeffSource<'_, '_>,
                        plane: AcPlane,
                        st: &mut PlaneState,
                        r: usize,
                        c: usize|
     -> [u8; 64] {
        let left = if c == 0 {
            None
        } else {
            st.dc[r * st.cols + c - 1]
        };
        let above = if r == 0 {
            None
        } else {
            st.dc[(r - 1) * st.cols + c]
        };
        let dc_context =
            DcContext::from_neighbours(left.is_some_and(|d| d != 0), above.is_some_and(|d| d != 0));
        let block = src.decode_block(plane, dc_context, &probs).expect("block");
        let reference = ReferenceBucket::Intra;
        let ln = left.map(|dc| Neighbour { dc, reference });
        let an = above.map(|dc| Neighbour { dc, reference });
        let predictor = st.pred.predict(reference, ln, an);
        let coded_dc = predictor.wrapping_add(block.coeffs[0]);
        st.pred.set_last_dc(reference, coded_dc);
        st.dc[r * st.cols + c] = Some(coded_dc);
        let mut sc = block.coeffs;
        sc[0] = coded_dc;
        let mut raster = oxideav_vp6::block_decode::dequantize_to_raster(&sc, &scan, dequant);
        oxideav_vp6::idct::idct_block(&mut raster);
        oxideav_vp6::reconstruct::intra_block_to_pixels(&raster)
    };

    // Walk the first 12 macroblocks of row 0 (x < 192 — well inside the
    // display width) and require every reconstructed sample to match
    // the oracle exactly.
    const N_MBS: usize = 12;
    let mut ys = PlaneState::new(108, 60);
    let mut us = PlaneState::new(54, 30);
    let mut vs = PlaneState::new(54, 30);
    let mut checked = 0usize;
    for mb_col in 0..N_MBS {
        for (k, (dr, dc)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
            .iter()
            .enumerate()
        {
            let (br, bcol) = (*dr, mb_col * 2 + dc);
            let pix = decode_block(&mut src, AcPlane::Y, &mut ys, br, bcol);
            for r in 0..8 {
                for c in 0..8 {
                    let (x, y) = (bcol * 8 + c, br * 8 + r);
                    assert_eq!(
                        pix[r * 8 + c],
                        oy[y * DISPLAY_W + x],
                        "luma mismatch at ({x},{y}), mb {mb_col} block {k}"
                    );
                    checked += 1;
                }
            }
        }
        for (oracle, st, tag) in [(ou, &mut us, "U"), (ov, &mut vs, "V")] {
            let pix = decode_block(&mut src, AcPlane::UV, st, 0, mb_col);
            for r in 0..8 {
                for c in 0..8 {
                    let (x, y) = (mb_col * 8 + c, r);
                    assert_eq!(
                        pix[r * 8 + c],
                        oracle[y * (DISPLAY_W / 2) + x],
                        "{tag} mismatch at ({x},{y}), mb {mb_col}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, N_MBS * 6 * 64, "all samples compared");
}

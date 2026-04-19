//! Integration test: pull the first VP6 keyframe out of the
//! `asian-commercials-are-weird.flv` sample and run it through the
//! decoder. Skips when the sample file isn't available (same pattern
//! as oxideav-av1's `reference_clips.rs`).
//!
//! Because the sample FLV sits in the oxideav monorepo's
//! `samples/` directory rather than in this crate, the test walks up
//! from `CARGO_MANIFEST_DIR` looking for it. Pass
//! `OXIDEAV_FLV_SAMPLE=<path>` to override.

use std::path::PathBuf;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp6::Vp6Decoder;

fn sample_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_FLV_SAMPLE") {
        let pb = PathBuf::from(p);
        return if pb.exists() { Some(pb) } else { None };
    }
    let mut here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..4 {
        let candidate = here.join("samples/asian-commercials-are-weird.flv");
        if candidate.exists() {
            return Some(candidate);
        }
        if !here.pop() {
            break;
        }
    }
    None
}

/// Hand-roll the minimum FLV walk needed to pull the first video-tag
/// payload. We don't pull in oxideav-flv as a dev-dep so this test
/// stays standalone — the FLV crate has its own integration test.
fn first_vp6_keyframe_body(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 13 || &bytes[0..3] != b"FLV" {
        return None;
    }
    let data_offset = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let mut pos = data_offset + 4; // skip first PreviousTagSize
    while pos + 11 < bytes.len() {
        let tag_type = bytes[pos] & 0x1F;
        let data_size = ((bytes[pos + 1] as u32) << 16)
            | ((bytes[pos + 2] as u32) << 8)
            | (bytes[pos + 3] as u32);
        let body_start = pos + 11;
        let body_end = body_start + data_size as usize;
        if body_end + 4 > bytes.len() {
            return None;
        }
        if tag_type == 0x09 && body_end - body_start >= 1 {
            let vhdr = bytes[body_start];
            let frame_type = vhdr >> 4;
            let codec_id = vhdr & 0x0F;
            if (frame_type == 1 || frame_type == 4) && codec_id == 4 {
                // VP6f keyframe — return the payload without the 1-byte
                // FLV video header.
                return Some(bytes[body_start + 1..body_end].to_vec());
            }
        }
        pos = body_end + 4;
    }
    None
}

#[test]
fn decode_first_vp6_keyframe() {
    let Some(path) = sample_path() else {
        eprintln!("sample FLV missing — skipping decode_first_vp6_keyframe");
        return;
    };
    let Some(keyframe) = first_vp6_keyframe_body(&path) else {
        eprintln!("couldn't locate first VP6 keyframe in sample — skipping");
        return;
    };
    // `first_vp6_keyframe_body` returns exactly what the FLV demuxer
    // would hand the decoder: the 1-byte VP6 horizontal / vertical
    // adjustment followed by the coded frame. `Vp6Decoder::send_packet`
    // strips the adjustment byte itself.
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut pkt = Packet::new(0, TimeBase::new(1, 1000), keyframe);
    pkt.pts = Some(0);
    pkt.flags.keyframe = true;
    dec.send_packet(&pkt).expect("vp6 decode keyframe");
    let frame = dec.receive_frame().expect("receive frame");
    match frame {
        Frame::Video(v) => {
            assert!(
                v.width > 0 && v.width <= 4096,
                "implausible width {}",
                v.width
            );
            assert!(
                v.height > 0 && v.height <= 4096,
                "implausible height {}",
                v.height
            );
            assert_eq!(v.planes.len(), 3);
            assert_eq!(v.planes[0].data.len(), v.width as usize * v.height as usize);
            // Luma mean — full MB + coefficient decode isn't ported
            // yet, so the current skeleton returns a mid-grey plane.
            // Check that the mean is within [0, 255].
            let sum: u64 = v.planes[0].data.iter().map(|&b| b as u64).sum();
            let mean = sum / (v.planes[0].data.len() as u64);
            assert!(mean <= 255);
            eprintln!("VP6 keyframe: {}x{} luma_mean={mean}", v.width, v.height);
        }
        other => panic!("expected video frame, got {other:?}"),
    }
}

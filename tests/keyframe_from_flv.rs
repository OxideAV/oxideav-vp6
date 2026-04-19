//! Integration test: decode the first VP6 keyframe — and the first
//! handful of inter frames — from the `asian-commercials-are-weird.flv`
//! sample. Skips when the sample file isn't available (pattern mirrors
//! oxideav-av1's `reference_clips.rs`).
//!
//! We hand-roll a minimal FLV walker instead of pulling the
//! `oxideav-flv` crate in as a dev-dep to keep this test standalone.

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
    for _ in 0..6 {
        let candidate = here.join("samples/asian-commercials-are-weird.flv");
        if candidate.exists() {
            return Some(candidate);
        }
        let alt = here.join("oxideav/samples/asian-commercials-are-weird.flv");
        if alt.exists() {
            return Some(alt);
        }
        if !here.pop() {
            break;
        }
    }
    None
}

/// Walk the FLV, collecting the first `n` VP6F video-tag payloads
/// (each payload is the FLV "byte after the one-byte vidoe header"
/// prefix, ready to feed directly to `Vp6Decoder::send_packet`).
fn collect_vp6_tags(path: &std::path::Path, n: usize) -> Vec<(bool, Vec<u8>)> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    if bytes.len() < 13 || &bytes[0..3] != b"FLV" {
        return Vec::new();
    }
    let data_offset = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let mut pos = data_offset + 4; // skip first PreviousTagSize
    let mut out = Vec::new();
    while pos + 11 < bytes.len() && out.len() < n {
        let tag_type = bytes[pos] & 0x1F;
        let data_size = ((bytes[pos + 1] as u32) << 16)
            | ((bytes[pos + 2] as u32) << 8)
            | (bytes[pos + 3] as u32);
        let body_start = pos + 11;
        let body_end = body_start + data_size as usize;
        if body_end + 4 > bytes.len() {
            break;
        }
        if tag_type == 0x09 && body_end - body_start >= 1 {
            let vhdr = bytes[body_start];
            let frame_type = vhdr >> 4;
            let codec_id = vhdr & 0x0F;
            if codec_id == 4 {
                let is_key = frame_type == 1;
                out.push((is_key, bytes[body_start + 1..body_end].to_vec()));
            }
        }
        pos = body_end + 4;
    }
    out
}

#[test]
fn decode_first_vp6_keyframe() {
    let Some(path) = sample_path() else {
        eprintln!("sample FLV missing — skipping decode_first_vp6_keyframe");
        return;
    };
    let tags = collect_vp6_tags(&path, 1);
    let Some((true, ref keyframe)) = tags.first().cloned() else {
        eprintln!("first VP6 tag in sample isn't a keyframe — skipping");
        return;
    };
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut pkt = Packet::new(0, TimeBase::new(1, 1000), keyframe.clone());
    pkt.pts = Some(0);
    pkt.flags.keyframe = true;
    dec.send_packet(&pkt).expect("vp6 decode keyframe");
    let frame = dec.receive_frame().expect("receive frame");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.width, 464, "expected 464x352 frame");
            assert_eq!(v.height, 352);
            assert_eq!(v.planes.len(), 3);
            assert_eq!(v.planes[0].data.len(), v.width as usize * v.height as usize);

            // Luma statistics: mean, distinct values.
            let y = &v.planes[0].data;
            let sum: u64 = y.iter().map(|&b| b as u64).sum();
            let mean = sum / (y.len() as u64);
            let mut seen = [false; 256];
            for &b in y {
                seen[b as usize] = true;
            }
            let distinct = seen.iter().filter(|b| **b).count();
            // Stable fingerprint (FNV-1a 64) of the first 64 luma bytes
            // — surfaces in test output so regressions show up.
            let mut hash: u64 = 0xcbf29ce484222325;
            for &b in y.iter().take(64) {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            eprintln!(
                "VP6 keyframe: {}x{} luma_mean={mean} luma_distinct={distinct} fnv64_y64=0x{hash:016x}",
                v.width, v.height
            );
            // The target scene is a bright commercial opening card;
            // mean should sit in a plausible mid-to-bright range.
            assert!((32..=240).contains(&mean), "implausible luma mean {mean}");
            // Real decoded content — not a flat plane.
            assert!(distinct >= 10, "luma only has {distinct} distinct values");
        }
        other => panic!("expected video frame, got {other:?}"),
    }
}

#[test]
fn decode_first_20_frames() {
    let Some(path) = sample_path() else {
        eprintln!("sample FLV missing — skipping decode_first_20_frames");
        return;
    };
    let tags = collect_vp6_tags(&path, 20);
    if tags.len() < 2 {
        eprintln!("sample FLV has <2 VP6 tags — skipping");
        return;
    }
    let want = tags.len();

    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut decoded = 0usize;
    let mut means = Vec::new();
    for (idx, (is_key, payload)) in tags.iter().enumerate() {
        let mut pkt = Packet::new(0u32, TimeBase::new(1, 1000), payload.clone());
        pkt.pts = Some(idx as i64);
        pkt.flags.keyframe = *is_key;
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("send_packet #{idx}: {e:?}");
                continue;
            }
        }
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                let y = &v.planes[0].data;
                let sum: u64 = y.iter().map(|&b| b as u64).sum();
                let mean = sum / (y.len() as u64);
                means.push(mean);
                decoded += 1;
            }
            Ok(other) => panic!("unexpected frame {other:?}"),
            Err(e) => {
                eprintln!("receive_frame #{idx}: {e:?}");
            }
        }
    }
    eprintln!("decoded {decoded}/{want} frames; luma means = {means:?}");
    assert_eq!(
        decoded, want,
        "all {want} frames should decode (got {decoded})"
    );
}

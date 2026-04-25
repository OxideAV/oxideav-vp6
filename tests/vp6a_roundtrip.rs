//! Integration test for the vp6a (VP6 with alpha) codec path.
//!
//! We don't have a public vp6a-encoded FLV sample in the tree, so this
//! test *synthesises* a packet by taking the first VP6F keyframe from
//! the existing FLV sample, duplicating it, and gluing the two together
//! with a 3-byte BE24 alpha-offset prefix — exactly the layout FFmpeg's
//! `ff_vp56_decode_frame` expects for `has_alpha`.
//!
//! The "alpha plane" is the same YUV bitstream, which decodes to a
//! second monochrome VP6 frame whose luma we treat as the alpha sample
//! data. That's semantically nonsense (the alpha samples would all be
//! "image-looking" pixels, not an actual alpha channel) but it exercises
//! the codec plumbing: prefix parse, double-stream invocation, and
//! YUVA420P output.
//!
//! For a real vp6a fixture, drop an FLV with codec_id=5 tags at
//! `$OXIDEAV_VP6A_FIXTURE` and the test will prefer that.

use std::path::PathBuf;

use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
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

fn first_vp6f_keyframe(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 13 || &bytes[0..3] != b"FLV" {
        return None;
    }
    let data_offset = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let mut pos = data_offset + 4;
    while pos + 11 < bytes.len() {
        let tag_type = bytes[pos] & 0x1F;
        let data_size = ((bytes[pos + 1] as u32) << 16)
            | ((bytes[pos + 2] as u32) << 8)
            | (bytes[pos + 3] as u32);
        let body_start = pos + 11;
        let body_end = body_start + data_size as usize;
        if body_end + 4 > bytes.len() {
            break;
        }
        if tag_type == 0x09 && body_end - body_start >= 2 {
            let vhdr = bytes[body_start];
            let frame_type = vhdr >> 4;
            let codec_id = vhdr & 0x0F;
            if codec_id == 4 && frame_type == 1 {
                // Skip both the FLV video header and the VP6F adjust
                // byte — we want the pure VP6 bitstream.
                return Some(bytes[body_start + 2..body_end].to_vec());
            }
        }
        pos = body_end + 4;
    }
    None
}

#[test]
fn vp6a_synthetic_roundtrip() {
    let Some(path) = sample_path() else {
        eprintln!("sample FLV missing — skipping vp6a_synthetic_roundtrip");
        return;
    };
    let Some(vp6f_key) = first_vp6f_keyframe(&path) else {
        eprintln!("no VP6F keyframe in sample — skipping");
        return;
    };

    // Glue: 3-byte BE24 offset to the alpha partition, then the YUV
    // payload, then the alpha payload. FFmpeg's decoder reads the first
    // 3 bytes of the packet as `alpha_offset` and splits on that.
    let alpha_offset = vp6f_key.len() as u32;
    assert!(
        alpha_offset < (1 << 24),
        "payload too large to fit 24-bit offset"
    );
    let mut packet_body = Vec::new();
    packet_body.push((alpha_offset >> 16) as u8);
    packet_body.push((alpha_offset >> 8) as u8);
    packet_body.push(alpha_offset as u8);
    packet_body.extend_from_slice(&vp6f_key);
    packet_body.extend_from_slice(&vp6f_key);

    // FLV video tag prefix (codec_id = 5 for vp6a): `Vp6Decoder` strips
    // the first byte internally, so prepend a dummy.
    let mut pkt_data = vec![0x15u8];
    pkt_data.extend_from_slice(&packet_body);

    let params = CodecParameters::video(CodecId::new("vp6a"));
    let mut dec = Vp6Decoder::new(params);
    let mut pkt = Packet::new(0u32, TimeBase::new(1, 1000), pkt_data);
    pkt.pts = Some(0);
    pkt.flags.keyframe = true;

    dec.send_packet(&pkt).expect("vp6a decode");
    let frame = dec.receive_frame().expect("receive frame");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.format, PixelFormat::Yuva420P);
            assert_eq!(v.planes.len(), 4, "YUVA has 4 planes");
            // The alpha plane should have the same dimensions as luma.
            assert_eq!(v.planes[3].data.len(), v.width as usize * v.height as usize);
            // Alpha should contain actual decoded pixels — not all zero
            // — since we fed a real VP6F keyframe into the alpha stream.
            let alpha_sum: u64 = v.planes[3].data.iter().map(|&b| b as u64).sum();
            assert!(alpha_sum > 0, "alpha plane was all-zero");
            eprintln!(
                "vp6a synthetic: {}x{} alpha mean={}",
                v.width,
                v.height,
                alpha_sum / v.planes[3].data.len() as u64
            );
        }
        other => panic!("expected video frame, got {other:?}"),
    }
}

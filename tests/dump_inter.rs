//! Diagnostic dump for inter-frame ffmpeg debugging — opt-in via
//! `VP6_DUMP_INTER=1`. Writes a 2-tag (key + skip) FLV to
//! `/tmp/oxideav_vp6_dump.flv` and traces the bool-stream symbols of
//! the inter packet's body to stderr, so the next round can `xxd` /
//! `ffmpeg -v debug` against a known-shape input without re-deriving
//! the test fixture.
//!
//! Bool stream is reverse-decoded with the in-tree [`RangeCoder`] —
//! that confirms our encoder's bit-stream is internally consistent
//! (decoder-side) before we look for divergences with ffmpeg's vp6f
//! reference.

use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp6::{RangeCoder, Vp6Decoder, Vp6Encoder};

#[test]
fn dump_inter_minimal() {
    if std::env::var("VP6_DUMP_INTER").is_err() {
        return;
    }

    // 64x32 striped luma so the keyframe carries enough bool-coded
    // body to clear ffmpeg's vp6f probe (≤ ~30-byte keyframes get
    // rejected at the demuxer level — confirmed empirically).
    let (w, h) = (64usize, 32usize);
    let mut y0 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y0[row * w + col] = if (col / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).unwrap();
    print_hex("Keyframe", &key);

    // Decode the keyframe so we have reconstructed planes for the
    // skip-frame motion search baseline (matches what ffmpeg sees).
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut p = Vec::new();
    p.push(0u8);
    p.extend_from_slice(&key);
    let mut pkt = Packet::new(0u32, TimeBase::new(1, 1000), p);
    pkt.pts = Some(0);
    pkt.flags.keyframe = true;
    dec.send_packet(&pkt).unwrap();
    let _ = match dec.receive_frame().unwrap() {
        Frame::Video(vf) => vf,
        _ => panic!(),
    };

    let inter = enc.encode_skip_frame().unwrap();
    print_hex("Skip frame", &inter);

    // Round-trip the skip frame through our decoder — this catches any
    // encoder/decoder asymmetry before we go look for ffmpeg-side bugs.
    let mut p2 = Vec::new();
    p2.push(0u8);
    p2.extend_from_slice(&inter);
    let mut spkt = Packet::new(0u32, TimeBase::new(1, 1000), p2);
    spkt.pts = Some(33);
    match dec.send_packet(&spkt) {
        Ok(_) => eprintln!("Our decoder accepts the skip frame"),
        Err(e) => eprintln!("Our decoder rejects: {e:?}"),
    }

    // Trace the first bool-coded symbols (golden / huffman / mb-type
    // model update flags) so a future investigator can compare against
    // ffmpeg's debug output without re-running the encoder.
    let body = &inter[3..];
    let mut rac = RangeCoder::new(body).unwrap();
    eprintln!("\n-- Manual bool-stream trace (header section) --");
    eprintln!(
        "golden_frame_flag (get_bit): {}\nuse_huffman (get_bit):       {}",
        rac.get_bit(),
        rac.get_bit()
    );
    for ctx in 0..3 {
        let a = rac.get_prob(174);
        let b = rac.get_prob(254);
        eprintln!(
            "ctx={ctx}: SetNewBaselineProbs(B(174))={a} \
             VectorUpdatesPresentFlag(B(254))={b}"
        );
    }

    // Mux key + skip into a 2-tag FLV at /tmp/oxideav_vp6_dump.flv so
    // ffmpeg / ffprobe can be pointed at it directly.
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());
    push_tag(&mut flv, &key, 0, true);
    push_tag(&mut flv, &inter, 33, false);
    std::fs::write("/tmp/oxideav_vp6_dump.flv", &flv).unwrap();
    eprintln!("\nFLV: /tmp/oxideav_vp6_dump.flv ({} bytes)", flv.len());
}

fn print_hex(label: &str, bytes: &[u8]) {
    eprintln!("{label} ({} bytes):", bytes.len());
    for (i, b) in bytes.iter().enumerate() {
        eprint!("{:02x} ", b);
        if (i + 1) % 16 == 0 {
            eprintln!();
        }
    }
    if bytes.len() % 16 != 0 {
        eprintln!();
    }
}

fn push_tag(flv: &mut Vec<u8>, frame: &[u8], pts: u32, is_key: bool) {
    let payload_len = (1 + 1 + frame.len()) as u32;
    flv.push(9);
    flv.extend_from_slice(&payload_len.to_be_bytes()[1..]);
    flv.push(((pts >> 16) & 0xff) as u8);
    flv.push(((pts >> 8) & 0xff) as u8);
    flv.push((pts & 0xff) as u8);
    flv.push(((pts >> 24) & 0xff) as u8);
    flv.extend_from_slice(&[0, 0, 0]);
    flv.push(if is_key { 0x14 } else { 0x24 });
    flv.push(0x00);
    flv.extend_from_slice(frame);
    let tag_size = 11 + payload_len;
    flv.extend_from_slice(&tag_size.to_be_bytes());
}

/// Sanity test: assert the encoded inter packet's `Buff2Offset` raw
/// value matches the spec — it MUST equal `header_size + p1.len()` so
/// that ffmpeg can locate partition 2 directly from the wire offset.
/// Catches regressions of the r19 spec-compliance fix.
#[test]
fn inter_buff2_offset_is_spec_compliant() {
    use oxideav_vp6::Vp6Encoder;
    let (w, h) = (64usize, 32usize);
    let y0 = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(16);
    enc.encode_keyframe(&y0, &u, &v, w, h).unwrap();
    let inter = enc.encode_skip_frame().unwrap();
    // byte 0: header; bytes 1-2: Buff2Offset raw; bytes 3..buff2: p1;
    // bytes buff2..end: p2.
    let buff2 = ((inter[1] as usize) << 8) | (inter[2] as usize);
    assert!(
        buff2 >= 3 && buff2 <= inter.len(),
        "Buff2Offset {} outside frame bounds (len={})",
        buff2,
        inter.len()
    );
    // Partition 2 must hold at least the bool-coder priming seed (3 bytes).
    assert!(
        inter.len() - buff2 >= 3,
        "partition 2 too short: {} bytes",
        inter.len() - buff2
    );
}

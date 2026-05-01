//! ffmpeg-side interop check for the inter-frame encoder. Builds a
//! minimal 2-tag FLV (key + skip OR key + motion-search inter) from
//! our encoder, hands it to an external `ffmpeg` process, and asserts
//! ffmpeg accepts every packet (i.e., decodes 2 frames, no decode
//! errors).
//!
//! Skipped silently when `ffmpeg` isn't on PATH so CI without it stays
//! green. As of r23 both `ffmpeg_decodes_keyframe_in_two_tag_stream`
//! and `r21_inter_frame_ffmpeg_decode_state` assert `n == 2`; the
//! breakthrough was setting `Vp3VersionNo = 6` on the keyframe header
//! (was emitting 0, which is forbidden by spec — see Table 2).

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_vp6::Vp6Encoder;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build a minimal FLV with a 64x32 keyframe + skip frame and hand it
/// to ffmpeg. Returns the number of frames ffmpeg successfully
/// decoded, parsed out of its `frame=N` line.
fn ffmpeg_decode_count(flv: &[u8]) -> u32 {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-i",
            "pipe:0",
            "-c:v",
            "rawvideo",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(flv).expect("write flv to ffmpeg stdin");
    }
    let out = child.wait_with_output().expect("ffmpeg wait");
    // ffmpeg always returns 0 even when individual packets fail to
    // decode — count "frame=N" instead. ffmpeg emits the stats line as
    // `frame=    1` (variable whitespace between `=` and the digits),
    // so we collapse `frame=` + one or more spaces + digits.
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), stderr);
    let mut last = 0u32;
    for line in combined.lines() {
        if let Some(after) = line.split("frame=").nth(1) {
            let digits: String = after
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                last = n;
            }
        }
    }
    last
}

fn build_flv(key: &[u8], inter: &[u8]) -> Vec<u8> {
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01); // video only
    flv.extend_from_slice(&9u32.to_be_bytes()); // header size
    flv.extend_from_slice(&0u32.to_be_bytes()); // PreviousTagSize0

    let push_tag = |flv: &mut Vec<u8>, frame: &[u8], pts: u32, is_key: bool| {
        let payload_len = (1 + 1 + frame.len()) as u32;
        flv.push(9); // video tag
        flv.push(((payload_len >> 16) & 0xff) as u8);
        flv.push(((payload_len >> 8) & 0xff) as u8);
        flv.push((payload_len & 0xff) as u8);
        flv.push(((pts >> 16) & 0xff) as u8);
        flv.push(((pts >> 8) & 0xff) as u8);
        flv.push((pts & 0xff) as u8);
        flv.push(((pts >> 24) & 0xff) as u8);
        flv.extend_from_slice(&[0, 0, 0]);
        flv.push(if is_key { 0x14 } else { 0x24 }); // FrameType + CodecId
        flv.push(0x00); // FLV adjuster
        flv.extend_from_slice(frame);
        let tag_size = 11 + payload_len;
        flv.extend_from_slice(&tag_size.to_be_bytes());
    };

    push_tag(&mut flv, key, 0, true);
    push_tag(&mut flv, inter, 33, false);
    flv
}

/// Asserts ffmpeg decodes our keyframe — guards against a regression in
/// the keyframe path. Uses a 2-keyframe FLV (back-to-back) so ffmpeg
/// reliably flushes the first decode without needing extra explicit
/// flush handling on stdin pipes. Skipped when ffmpeg isn't on PATH.
#[test]
fn ffmpeg_accepts_keyframe() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping ffmpeg_accepts_keyframe");
        return;
    }
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
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("encode key");

    // Two back-to-back keyframes give ffmpeg an unambiguous "next frame"
    // boundary so the first decoded frame surfaces in `frame=N`.
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());
    let push_tag = |flv: &mut Vec<u8>, frame: &[u8], pts: u32| {
        let payload_len = (1 + 1 + frame.len()) as u32;
        flv.push(9);
        flv.push(((payload_len >> 16) & 0xff) as u8);
        flv.push(((payload_len >> 8) & 0xff) as u8);
        flv.push((payload_len & 0xff) as u8);
        flv.push(((pts >> 16) & 0xff) as u8);
        flv.push(((pts >> 8) & 0xff) as u8);
        flv.push((pts & 0xff) as u8);
        flv.push(((pts >> 24) & 0xff) as u8);
        flv.extend_from_slice(&[0, 0, 0]);
        flv.push(0x14);
        flv.push(0x00);
        flv.extend_from_slice(frame);
        flv.extend_from_slice(&(11 + payload_len).to_be_bytes());
    };
    push_tag(&mut flv, &key, 0);
    push_tag(&mut flv, &key, 33);

    let n = ffmpeg_decode_count(&flv);
    assert!(
        n >= 2,
        "ffmpeg should decode both keyframes (got {n} frames)"
    );
}

/// Tracks the inter-frame interop state. As of r23 (`Vp3VersionNo = 6`
/// fix on the keyframe header), ffmpeg accepts both the keyframe AND
/// the skip inter packet — `n == 2`. Pre-r23 this was `n == 1` (key
/// accepted, inter rejected with "Invalid data found"). The
/// per-frame upper-bound + per-frame lower-bound now both pin to 2
/// so a regression in either direction (key OR inter) trips the test.
/// Skipped when ffmpeg isn't on PATH.
#[test]
fn ffmpeg_decodes_keyframe_in_two_tag_stream() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping ffmpeg_decodes_keyframe_in_two_tag_stream");
        return;
    }
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
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("encode key");
    let inter = enc.encode_skip_frame().expect("encode skip");

    let flv = build_flv(&key, &inter);
    let n = ffmpeg_decode_count(&flv);
    assert_eq!(
        n, 2,
        "ffmpeg should decode both keyframe + skip inter (got {n})"
    );
}

/// r21 + r23 milestone — the motion-search inter packet must
/// round-trip through ffmpeg's vp6f decoder. Pre-r23 ffmpeg rejected
/// our inter packet ("Invalid data found when processing input") even
/// though it accepted the keyframe; r23 traced this to byte 1 of the
/// keyframe header (`Vp3VersionNo`, R(5)) being emitted as 0 instead
/// of the spec-required 6/7/8. ffmpeg's keyframe path accepted the
/// invalid version (silently routing through a Vp6.<keyframe-only>
/// path) but the inter parser then mis-routed. Setting the encoder's
/// default `sub_version` to 6 (VP6.0 / Simple Profile) fixes both
/// paths — see the r23 audit notes in `src/encoder.rs`.
///
/// This is structurally similar to
/// `ffmpeg_decodes_keyframe_in_two_tag_stream` above but uses
/// `encode_inter_frame` (motion search) rather than `encode_skip_frame`
/// — both inter paths share the same picture-header model code, so a
/// model-level fix unblocks both tests at once. Skipped when ffmpeg
/// isn't on PATH.
#[test]
fn r21_inter_frame_ffmpeg_decode_state() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping r21_inter_frame_ffmpeg_decode_state");
        return;
    }
    let (w, h) = (64usize, 32usize);
    let mut y0 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y0[row * w + col] = if (col / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
    // Synthesise a "second frame" with a tiny horizontal shift so the
    // motion-search path picks a non-zero MV for at least some MBs —
    // this exercises the InterDeltaPf branch the skip frame doesn't.
    let mut y1 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let src = (col + 4).min(w - 1);
            y1[row * w + col] = if (src / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("encode key");
    // encode_inter_frame argument order: (prev_y, prev_u, prev_v,
    // new_y, new_u, new_v, w, h, search). The previous-frame planes
    // are the reference (y0); y1 is the new frame to encode.
    let inter = enc
        .encode_inter_frame(&y0, &u, &v, &y1, &u, &v, w, h, 4)
        .expect("encode inter");

    let flv = build_flv(&key, &inter);
    let n = ffmpeg_decode_count(&flv);
    // r23: both keyframe + motion-search inter must decode cleanly.
    // Strict equality so a regression in either direction is caught.
    assert_eq!(
        n, 2,
        "ffmpeg must accept both keyframe + motion-search inter (got {n})"
    );
}

/// r23 spec-compliance pin — VP6 spec §9 / Table 2 specifies that the
/// keyframe header's `Vp3VersionNo` field (R(5), bits 7..3 of byte 1)
/// must hold the value 6 (VP6.0), 7 (VP6.1), or 8 (VP6.2). The value
/// 0 is forbidden, even though some lenient decoders (including
/// ffmpeg's keyframe decode path pre-r23) would silently accept it.
/// This guard pins the on-wire byte to the spec-legal range so any
/// regression of `Vp6Encoder::default().sub_version` to 0 surfaces
/// immediately — without it the inter packet's ffmpeg interop quietly
/// breaks.
#[test]
fn keyframe_vp3_version_no_is_spec_legal() {
    let (w, h) = (16usize, 16usize);
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(32);
    let key = enc.encode_keyframe(&y, &u, &v, w, h).expect("encode key");
    let vp3_version_no = key[1] >> 3;
    assert!(
        (6..=8).contains(&vp3_version_no),
        "Vp3VersionNo must be 6/7/8 per VP6 spec Table 2 (got {vp3_version_no}, byte1=0x{:02x})",
        key[1]
    );
    // VpProfile (bits 2..1) should be 0 (Simple) for our scaffold.
    let vp_profile = (key[1] >> 1) & 0x3;
    assert_eq!(
        vp_profile, 0,
        "VpProfile must be 0 (Simple) for the scaffold (got {vp_profile})"
    );
    // Reserved bit 0 must be 0 per spec.
    assert_eq!(key[1] & 1, 0, "Reserved bit (byte 1, bit 0) must be 0");
}

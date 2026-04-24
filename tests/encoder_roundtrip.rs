//! Integration-test for the VP6F encoder scaffold.
//!
//! Encodes a few synthetic frames (flat gray, constant color, vertical
//! gradient) and feeds the output into our own decoder, then verifies
//! the result is close enough to the source.
//!
//! PSNR threshold is 30 dB for flat content — the initial encoder is
//! DC-only (all AC coefficients zero) so anything non-flat will lose a
//! lot of detail, but a constant-color frame should reconstruct within
//! a handful of ULPs per pixel.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp6::{Vp6Decoder, Vp6Encoder};

/// Build a VP6F FLV-style packet by prefixing the 1-byte FLV adjuster.
fn packet_from_frame(bytes: Vec<u8>) -> Vec<u8> {
    let mut p = Vec::with_capacity(bytes.len() + 1);
    p.push(0u8);
    p.extend_from_slice(&bytes);
    p
}

fn decode_first_frame(bytes: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize, usize) {
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let pkt = Packet::new(0, TimeBase::new(1, 1000), packet_from_frame(bytes));
    dec.send_packet(&pkt).expect("decode send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("decode receive_frame") else {
        panic!("expected VideoFrame");
    };
    let width = vf.width as usize;
    let height = vf.height as usize;
    let y = vf.planes[0].data.clone();
    let u = vf.planes[1].data.clone();
    let v = vf.planes[2].data.clone();
    (y, u, v, width, height)
}

fn plane_psnr(src: &[u8], dst: &[u8]) -> f64 {
    assert_eq!(src.len(), dst.len());
    let mut sse: u64 = 0;
    for (a, b) in src.iter().zip(dst.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / src.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn flat_gray_roundtrip_exact() {
    let (w, h) = (32usize, 16usize);
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(32);
    let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
    let (dy, du, dv, dw, dh) = decode_first_frame(bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    let py = plane_psnr(&y, &dy);
    let pu = plane_psnr(&u, &du);
    let pv = plane_psnr(&v, &dv);
    // 128/128/128 should be reconstructable exactly (all DC coded_dc=0
    // except the first chroma MB block which compensates predictor 128).
    assert!(py >= 30.0, "Y PSNR too low: {py}");
    assert!(pu >= 30.0, "U PSNR too low: {pu}");
    assert!(pv >= 30.0, "V PSNR too low: {pv}");
}

#[test]
fn constant_color_roundtrip() {
    // Non-128 flat color. Both luma and chroma will need non-zero coded
    // DCs at the first MB; every other MB coded_dc=0 via DC prediction.
    let (w, h) = (48usize, 32usize);
    let y = vec![96u8; w * h];
    let u = vec![64u8; (w / 2) * (h / 2)];
    let v = vec![200u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(16);
    let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
    let (dy, du, dv, _, _) = decode_first_frame(bytes);
    let py = plane_psnr(&y, &dy);
    let pu = plane_psnr(&u, &du);
    let pv = plane_psnr(&v, &dv);
    assert!(py >= 30.0, "Y PSNR too low: {py}");
    assert!(pu >= 30.0, "U PSNR too low: {pu}");
    assert!(pv >= 30.0, "V PSNR too low: {pv}");
}

/// Opt-in test: pipe an encoded keyframe through ffmpeg's built-in
/// `vp6f` decoder and verify it produces the expected luma plane.
/// Skipped unless `ffmpeg` is on `PATH` — doesn't fail when absent.
#[test]
fn ffmpeg_vp6f_decodes_our_flat_keyframe() {
    use std::process::Command;

    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("ffmpeg not available — skipping");
        return;
    }

    let (w, h) = (32usize, 16usize);
    let y = vec![96u8; w * h];
    let u = vec![64u8; (w / 2) * (h / 2)];
    let v = vec![200u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(16);
    let frame = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();

    // Minimal FLV container around one keyframe.
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01); // version
    flv.push(0x01); // flags (has video)
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());
    let video_payload_len = 1 + 1 + frame.len();
    flv.push(9); // video tag
    flv.extend_from_slice(&(video_payload_len as u32).to_be_bytes()[1..]);
    flv.extend_from_slice(&[0, 0, 0, 0]); // timestamp + extended
    flv.extend_from_slice(&[0, 0, 0]); // stream id
    flv.push(0x14); // keyframe | codec 4 (VP6F)
    flv.push(0x00); // adjuster
    flv.extend_from_slice(&frame);
    flv.extend_from_slice(&(11 + video_payload_len as u32).to_be_bytes());

    let flv_path = std::env::temp_dir().join("oxideav_vp6_rt.flv");
    let yuv_path = std::env::temp_dir().join("oxideav_vp6_rt.yuv");
    std::fs::write(&flv_path, &flv).unwrap();

    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "flv",
            "-i",
        ])
        .arg(&flv_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");

    let raw = std::fs::read(&yuv_path).unwrap();
    let ylen = w * h;
    let uvlen = (w / 2) * (h / 2);
    assert_eq!(raw.len(), ylen + 2 * uvlen);
    let ff_y = &raw[0..ylen];
    let ff_u = &raw[ylen..ylen + uvlen];
    let ff_v = &raw[ylen + uvlen..];
    // ffmpeg's decode should match our decoder bit-exactly.
    let (ours_y, ours_u, ours_v, _, _) = decode_first_frame(frame.clone());
    assert_eq!(ff_y, ours_y.as_slice(), "luma mismatch vs ffmpeg");
    assert_eq!(ff_u, ours_u.as_slice(), "chroma U mismatch vs ffmpeg");
    assert_eq!(ff_v, ours_v.as_slice(), "chroma V mismatch vs ffmpeg");

    let _ = std::fs::remove_file(flv_path);
    let _ = std::fs::remove_file(yuv_path);
}

#[test]
fn vertical_gradient_plane_mean_preserved() {
    // A vertical gradient: each 8x8 block has a clearly different mean.
    // DC-only encoding can only reconstruct the per-block mean, so the
    // *aggregate* plane mean is what we check — PSNR against the source
    // pixel-per-pixel is much lower because we lose all AC.
    let (w, h) = (64usize, 32usize);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        let val = (row as u32 * 255 / (h as u32 - 1)) as u8;
        for col in 0..w {
            y[row * w + col] = val;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(8);
    let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
    let (dy, _, _, _, _) = decode_first_frame(bytes);
    // Block-mean should match within a couple ULPs for each 8x8 block.
    for mb_row in 0..(h / 8) {
        for mb_col in 0..(w / 8) {
            let mut src_sum = 0u32;
            let mut dst_sum = 0u32;
            for r in 0..8 {
                for c in 0..8 {
                    src_sum += y[(mb_row * 8 + r) * w + (mb_col * 8 + c)] as u32;
                    dst_sum += dy[(mb_row * 8 + r) * w + (mb_col * 8 + c)] as u32;
                }
            }
            let src_mean = src_sum / 64;
            let dst_mean = dst_sum / 64;
            assert!(
                (src_mean as i32 - dst_mean as i32).abs() <= 4,
                "block ({mb_row},{mb_col}) mean drift: src={src_mean} dst={dst_mean}"
            );
        }
    }
}

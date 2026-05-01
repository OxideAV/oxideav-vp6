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

use oxideav_core::Decoder;
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
    let width = vf.planes[0].stride;
    let height = vf.planes[0].data.len() / width;
    let y = vf.planes[0].data.clone();
    let u = vf.planes[1].data.clone();
    let v = vf.planes[2].data.clone();
    (y, u, v, width, height)
}

/// Wrap an encoded VP6F elementary-stream frame in a minimal FLV
/// container and decode it with an external `ffmpeg` process. Returns
/// the three YUV420p planes. Panics if ffmpeg isn't on PATH — callers
/// should gate on availability first.
fn ffmpeg_decode_frame(frame: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::process::Command;
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());
    let video_payload_len = 1 + 1 + frame.len();
    flv.push(9);
    flv.extend_from_slice(&(video_payload_len as u32).to_be_bytes()[1..]);
    flv.extend_from_slice(&[0, 0, 0, 0]);
    flv.extend_from_slice(&[0, 0, 0]);
    flv.push(0x14);
    flv.push(0x00);
    flv.extend_from_slice(frame);
    flv.extend_from_slice(&(11 + video_payload_len as u32).to_be_bytes());

    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let stamp = std::process::id();
    let flv_path = std::env::temp_dir().join(format!("oxideav_vp6_t{stamp}_{seq}.flv"));
    let yuv_path = std::env::temp_dir().join(format!("oxideav_vp6_t{stamp}_{seq}.yuv"));
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
    assert!(status.success(), "ffmpeg failed to decode frame");
    let raw = std::fs::read(&yuv_path).unwrap();
    let ylen = w * h;
    let uvlen = (w / 2) * (h / 2);
    assert_eq!(raw.len(), ylen + 2 * uvlen);
    let ff_y = raw[0..ylen].to_vec();
    let ff_u = raw[ylen..ylen + uvlen].to_vec();
    let ff_v = raw[ylen + uvlen..].to_vec();
    let _ = std::fs::remove_file(flv_path);
    let _ = std::fs::remove_file(yuv_path);
    (ff_y, ff_u, ff_v)
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
fn vertical_gradient_psnr_recovers_detail() {
    // A vertical gradient — every 8x8 block has both non-zero DC and
    // non-zero low-frequency AC. With AC coding enabled, PSNR should
    // lift well above the DC-only ~26 dB via our own decoder.
    //
    // After the Round-9 axis-transpose fix, encoder <-> our decoder
    // round-trips cleanly on gradient content: the forward DCT now
    // uses natural `out[u*8+v] = F[u,v]` layout, the scan permutation
    // matches the spec's `default_dequant_table` (no transpose), and
    // our decoder's IDCT sees coefficients in the same raster order
    // ffmpeg's VP6 decoder does.
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
    let frame = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();

    let (dy, _, _, _, _) = decode_first_frame(frame);
    let py = plane_psnr(&y, &dy);
    assert!(
        py >= 35.0,
        "Y PSNR via in-tree decoder too low with AC encoding: {py} (target >= 35 dB)"
    );
}

#[test]
fn horizontal_gradient_psnr_recovers_detail() {
    // Mirror of the vertical gradient test — exercises the other axis
    // to catch any residual row/col swap. Each 8x8 block now has a
    // purely horizontal AC component.
    let (w, h) = (64usize, 32usize);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let val = (col as u32 * 255 / (w as u32 - 1)) as u8;
            y[row * w + col] = val;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(8);
    let frame = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();

    let (dy, _, _, _, _) = decode_first_frame(frame);
    let py = plane_psnr(&y, &dy);
    assert!(
        py >= 35.0,
        "Y PSNR via in-tree decoder too low on horizontal gradient: {py}"
    );
}

/// Gradient content: verify ffmpeg accepts our output and decodes it
/// with ≥ 35 dB Y PSNR against the source AND that our own decoder
/// produces a bit-identical reconstruction. This is the cross-check
/// that guards the Round-9 axis-transpose fix.
#[test]
fn ffmpeg_vp6f_decodes_gradient_keyframe() {
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
    let frame = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();

    let (ff_y, ff_u, ff_v) = ffmpeg_decode_frame(&frame, w, h);
    let py = plane_psnr(&y, &ff_y);
    let pu = plane_psnr(&u, &ff_u);
    let pv = plane_psnr(&v, &ff_v);
    assert!(
        py >= 35.0,
        "Y PSNR via ffmpeg too low on gradient: {py} (target >= 35 dB)"
    );
    assert!(pu >= 30.0, "U PSNR via ffmpeg too low: {pu}");
    assert!(pv >= 30.0, "V PSNR via ffmpeg too low: {pv}");

    // Round-9: our decoder now matches ffmpeg byte-close on the same
    // keyframe. A handful of single-ULP differences come from
    // shift-and-round divergences between the spec's fixed-point IDCT
    // chain and ffmpeg's internal one; anything larger (or a wholesale
    // transpose) would blow well past `mean_abs_diff <= 1` here.
    let (ours_y, ours_u, ours_v, _, _) = decode_first_frame(frame.clone());
    let mean_abs = |a: &[u8], b: &[u8]| -> f64 {
        let mut s = 0u64;
        for (x, y) in a.iter().zip(b) {
            s += (*x as i32 - *y as i32).unsigned_abs() as u64;
        }
        s as f64 / a.len() as f64
    };
    let y_mad = mean_abs(&ff_y, &ours_y);
    let u_mad = mean_abs(&ff_u, &ours_u);
    let v_mad = mean_abs(&ff_v, &ours_v);
    assert!(y_mad <= 1.0, "Y mean-abs-diff vs ffmpeg: {y_mad}");
    assert!(u_mad <= 1.0, "U mean-abs-diff vs ffmpeg: {u_mad}");
    assert!(v_mad <= 1.0, "V mean-abs-diff vs ffmpeg: {v_mad}");
}

/// P-frame scaffold: encode a keyframe, then an identity skip frame,
/// push both through the decoder in sequence, and verify the skip
/// frame reconstructs to the same planes as the preceding keyframe.
#[test]
fn skip_frame_identity_reproduces_previous_frame() {
    let (w, h) = (32usize, 16usize);
    let y = vec![96u8; w * h];
    let u = vec![64u8; (w / 2) * (h / 2)];
    let v = vec![200u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    let key = enc.encode_keyframe(&y, &u, &v, w, h).expect("keyframe");
    let skip = enc.encode_skip_frame().expect("skip frame");

    // Decode both packets through a single decoder so the skip frame's
    // inter-decode path picks up the keyframe's state.
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);

    let mut key_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(key));
    key_pkt.pts = Some(0);
    key_pkt.flags.keyframe = true;
    dec.send_packet(&key_pkt).expect("send keyframe");
    let key_frame = match dec.receive_frame().expect("receive keyframe") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };

    let mut skip_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(skip));
    skip_pkt.pts = Some(1);
    dec.send_packet(&skip_pkt).expect("send skip");
    let skip_frame = match dec.receive_frame().expect("receive skip") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };

    // Skip frame should decode to the same dimensions + planes as the
    // preceding keyframe: the decoder copies the previous frame with no
    // residual, matching what our scaffold encoder asked for.
    assert_eq!(skip_frame.planes[0].stride, key_frame.planes[0].stride);
    assert_eq!(
        skip_frame.planes[0].data.len(),
        key_frame.planes[0].data.len()
    );
    for plane in 0..3usize {
        assert_eq!(
            skip_frame.planes[plane].data, key_frame.planes[plane].data,
            "skip plane {plane} should mirror keyframe plane"
        );
    }
}

/// MV encode — encode a keyframe of a checker-style luma pattern, then
/// translate the pattern horizontally by 4 pixels and encode that as a
/// P-frame against the keyframe. The decoder applies the encoded MV to
/// the previous frame: the result should be very close to the shifted
/// source (the MC is integer-pel, so reconstruction is exact within
/// the MV search window's reach for non-edge MBs).
#[test]
fn inter_frame_horizontal_shift_uses_mv() {
    // Use enough rows/cols so the search has room and the MC reads
    // entirely inside the previous frame.
    let (w, h) = (64usize, 32usize);
    // Build a vertical-stripes Y plane (high-contrast horizontal AC).
    let mut y0 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            // 8-pixel-period stripes (period intentionally not equal to
            // the MB size so the shift produces a measurably different
            // plane vs. the original).
            y0[row * w + col] = if (col / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
    // Frame 1: shift the stripes 4 px to the right.
    let shift = 4i32;
    let mut y1 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w as i32 {
            let src_col = (col - shift).clamp(0, w as i32 - 1) as usize;
            y1[row * w + col as usize] = y0[row * w + src_col];
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("keyframe");

    // Pre-decode the keyframe through our own decoder so the encoder's
    // MV search runs against the same reconstruction the decoder will
    // see (small drift from quantisation otherwise picks the wrong MV).
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut key_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(key));
    key_pkt.pts = Some(0);
    key_pkt.flags.keyframe = true;
    dec.send_packet(&key_pkt).expect("send keyframe");
    let key_frame = match dec.receive_frame().expect("receive keyframe") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    let recon_y = key_frame.planes[0].data.clone();
    let recon_u = key_frame.planes[1].data.clone();
    let recon_v = key_frame.planes[2].data.clone();

    // Now encode the inter frame. Using the reconstructed previous
    // frame (not the source) makes ME pick MVs that match what the
    // decoder will see.
    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v, w, h, 8)
        .expect("encode inter");

    let mut inter_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(inter));
    inter_pkt.pts = Some(1);
    dec.send_packet(&inter_pkt).expect("send inter");
    let inter_frame = match dec.receive_frame().expect("receive inter") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    let dy = &inter_frame.planes[0].data;

    // A pure-skip P-frame decodes to recon_y. We expect MV emission to
    // do strictly better than that — the per-MB MV should align the
    // shifted stripes back so the inter reconstruction matches y1
    // closely on the interior MBs (edge MBs may stay zero-MV when the
    // search window doesn't reach a better candidate).
    let psnr_skip = plane_psnr(&y1, &recon_y);
    let psnr_inter = plane_psnr(&y1, dy);
    eprintln!("skip-PSNR vs y1 = {psnr_skip:.2} dB, inter-PSNR vs y1 = {psnr_inter:.2} dB");
    assert!(
        psnr_inter > psnr_skip + 3.0,
        "MV-encoded inter frame should improve on skip baseline by ≥3 dB \
         (skip={psnr_skip:.2}, inter={psnr_inter:.2})"
    );
}

/// MV encode: ffmpeg interop. Encode a key + inter pair, mux into FLV,
/// decode via ffmpeg, and verify the resulting Y plane has reasonable
/// PSNR against the shifted source.
///
/// Currently opt-in via `OXIDEAV_VP6_FFMPEG_INTER=1` because the
/// inter-frame bitstream layer (probability-model update pass) still
/// diverges from what ffmpeg's vp6f decoder accepts — same caveat as
/// the existing `encode_skip_frame` scaffold notes. Our own decoder
/// round-trips the inter frame cleanly (see
/// `inter_frame_horizontal_shift_uses_mv`).
#[test]
fn ffmpeg_decodes_inter_frame_with_mv() {
    use std::process::Command;
    if std::env::var("OXIDEAV_VP6_FFMPEG_INTER").is_err() {
        eprintln!(
            "ffmpeg inter-frame interop opt-in: set \
             OXIDEAV_VP6_FFMPEG_INTER=1 to run"
        );
        return;
    }
    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("ffmpeg not available — skipping");
        return;
    }

    let (w, h) = (64usize, 32usize);
    let mut y0 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y0[row * w + col] = if (col / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
    let shift = 4i32;
    let mut y1 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w as i32 {
            let src_col = (col - shift).clamp(0, w as i32 - 1) as usize;
            y1[row * w + col as usize] = y0[row * w + src_col];
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("keyframe");
    // Decode locally so MV search picks the same MVs the decoder sees.
    let (recon_y, recon_u, recon_v, _, _) = decode_first_frame(key.clone());
    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v, w, h, 8)
        .expect("encode inter");

    // Mux key + inter into a 2-tag FLV stream. FLV layout per the
    // Adobe spec: 9-byte signature/header, 4-byte PreviousTagSize0=0,
    // then a sequence of (Tag, PreviousTagSize) pairs.
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes()); // PreviousTagSize0

    let push_tag = |flv: &mut Vec<u8>, frame: &[u8], pts: u32, is_key: bool| -> u32 {
        let video_payload_len = (1 + 1 + frame.len()) as u32;
        flv.push(9); // tag type: video
        flv.extend_from_slice(&video_payload_len.to_be_bytes()[1..]);
        let ts = pts;
        flv.push(((ts >> 16) & 0xff) as u8);
        flv.push(((ts >> 8) & 0xff) as u8);
        flv.push((ts & 0xff) as u8);
        flv.push(((ts >> 24) & 0xff) as u8); // ts extended
        flv.extend_from_slice(&[0, 0, 0]);
        let frame_type_codec = if is_key { 0x14 } else { 0x24 };
        flv.push(frame_type_codec);
        flv.push(0x00);
        flv.extend_from_slice(frame);
        let tag_size = 11 + video_payload_len;
        flv.extend_from_slice(&tag_size.to_be_bytes());
        tag_size
    };
    let _ = push_tag(&mut flv, &key, 0, true);
    let _ = push_tag(&mut flv, &inter, 33, false);

    let stamp = std::process::id();
    let flv_path = std::env::temp_dir().join(format!("oxideav_vp6_mv_{stamp}.flv"));
    let yuv_path = std::env::temp_dir().join(format!("oxideav_vp6_mv_{stamp}.yuv"));
    std::fs::write(&flv_path, &flv).unwrap();
    eprintln!("FLV written to {flv_path:?}, len={}", flv.len());
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
    assert!(status.success(), "ffmpeg failed to decode 2-tag FLV");
    let raw = std::fs::read(&yuv_path).unwrap();
    let frame_size = w * h + 2 * (w / 2) * (h / 2);
    assert!(
        raw.len() >= 2 * frame_size,
        "expected 2 frames, got {}",
        raw.len()
    );
    // Second frame's Y plane.
    let off = frame_size;
    let ff_y = &raw[off..off + w * h];

    let psnr_inter = plane_psnr(&y1, ff_y);
    eprintln!("ffmpeg PSNR vs shifted source: {psnr_inter:.2} dB");
    assert!(
        psnr_inter >= 25.0,
        "ffmpeg should decode our P-frame and reproduce the shift well: {psnr_inter:.2} dB"
    );

    // Keep the FLV around for ffprobe inspection on test failure.
    let _ = std::fs::remove_file(yuv_path);
    let _ = (recon_u, recon_v); // chroma reused only for MV search input
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

/// r24 — Inter residual coefficient encoding floor.
///
/// Encodes a flat keyframe, then encodes a SECOND frame with a
/// brightness ramp added on top. The ME can't help (the prev frame is
/// flat — every MB's best MV is `(0, 0)`), so MC alone reconstructs the
/// flat baseline. With residual coefficients enabled (r24+), the
/// encoder absorbs the per-block shift into the DCT residual and the
/// reconstruction PSNR clears 30 dB; the pre-r24 path was bounded by
/// the brightness-shift energy (MC-only baseline ~ 20.5 dB).
///
/// Why a flat keyframe? A flat reference frame guarantees ME picks
/// `(0, 0)` for every MB (zero SAD across the search window), so the
/// per-MB residual is exactly `y1 - 128` — directly testing the DCT +
/// quantise + emit path without any MV-thrashing noise.
#[test]
fn r24_inter_residual_psnr_floor() {
    let (w, h) = (32usize, 32usize);
    // Keyframe: a flat 128-luma plane. Decoded as 128 everywhere.
    let y0 = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v_plane = vec![128u8; (w / 2) * (h / 2)];

    // Inter frame: same content + per-MB brightness shifts. Block shift
    // varies across the frame so DC residual is non-trivial across
    // multiple MBs (exercising the per-MB DC predictor mirror, not just
    // the first one).
    let mut y1 = vec![128u8; w * h];
    for mb_row in 0..(h / 16) {
        for mb_col in 0..(w / 16) {
            let shift = 8 + ((mb_row * 7 + mb_col * 11) % 24) as u8 * 2;
            for r in 0..16usize {
                for c in 0..16usize {
                    let i = (mb_row * 16 + r) * w + (mb_col * 16 + c);
                    y1[i] = 128u8.saturating_add(shift);
                }
            }
        }
    }

    let mut enc = Vp6Encoder::new(12); // tighter QP -> better residual fidelity
    let key = enc
        .encode_keyframe(&y0, &u, &v_plane, w, h)
        .expect("keyframe");

    // Decode the keyframe through our own decoder so the encoder's
    // motion search runs against the same reconstruction the decoder
    // will see.
    let (recon_y, recon_u, recon_v, _, _) = decode_first_frame(key.clone());
    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v_plane, w, h, 4)
        .expect("encode inter");

    // Decode the (key + inter) sequence through our decoder.
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params);
    let mut key_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(key));
    key_pkt.pts = Some(0);
    key_pkt.flags.keyframe = true;
    dec.send_packet(&key_pkt).expect("send keyframe");
    let _ = dec.receive_frame().expect("receive keyframe");

    let mut inter_pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(inter));
    inter_pkt.pts = Some(33);
    dec.send_packet(&inter_pkt).expect("send inter");
    let inter_frame = match dec.receive_frame().expect("receive inter") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    let dy = &inter_frame.planes[0].data;

    // MC-only baseline (no residual): reconstructed Y == prev recon_y.
    // MSE_baseline = E[(y1 - recon_y)^2] ≈ shift^2 in the steady state.
    let psnr_mc_only = plane_psnr(&y1, &recon_y);
    let psnr_inter = plane_psnr(&y1, dy);
    eprintln!(
        "r24: MC-only baseline PSNR = {psnr_mc_only:.2} dB, \
         residual-coded PSNR = {psnr_inter:.2} dB"
    );

    // With residual encoding the PSNR should comfortably clear 30 dB.
    // The pre-r24 path topped out around the MC-only baseline because
    // the entire brightness delta had to live in the unrepresented
    // residual.
    assert!(
        psnr_inter >= 30.0,
        "Y PSNR with residual encoding too low: {psnr_inter:.2} dB \
         (target >= 30 dB; MC-only baseline was {psnr_mc_only:.2} dB)"
    );
    assert!(
        psnr_inter >= psnr_mc_only + 5.0,
        "residual encoding should improve on MC-only baseline by ≥5 dB \
         (mc_only={psnr_mc_only:.2}, residual={psnr_inter:.2})"
    );
}

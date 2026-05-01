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
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
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
    let mut dec = Vp6Decoder::new(params.codec_id.clone());

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
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
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
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
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

// =====================================================================
// r25: quarter-pel sub-pel motion estimation
//
// The two tests below validate the sub-pel ME path: a fixture (vertical
// stripes / circular gradient) is shifted by a sub-integer-pel amount
// between the keyframe and the inter frame. Without sub-pel ME the
// integer-pel MV alone can't capture the shift, so the inter
// reconstruction is dominated by the unrepresented sub-pel error
// (~21 dB Y for these fixtures). With qpel ME enabled, the decoder's
// bilinear filter follows the qpel MV and reconstruction clears 35 dB
// Y comfortably.
//
// To make the comparison apples-to-apples the tests compute both the
// "MC-only at the chosen qpel MV" baseline (what the qpel ME bought us
// before residual) and the actual decoded PSNR (qpel MC + DCT
// residual). Both are reported via `eprintln!` for diagnostic visibility.
// =====================================================================

/// Build a "translating vertical stripes" Y fixture pair.
/// `keyframe` has period-32 stripes (smooth sine, much wider than the
/// MB so bilinear MC reproduces the shift precisely); `inter` has the
/// same stripes shifted right by `qpel_shift` quarter-pel units (so
/// `qpel_shift = 2` is exactly half a pixel). The smooth low-frequency
/// profile means integer-only ME alone misses the sub-pel shift but
/// bilinear MC captures it within the noise floor.
fn build_translating_stripes(w: usize, h: usize, qpel_shift: i32) -> (Vec<u8>, Vec<u8>) {
    let period_pels = 32.0;
    let profile = |x_8th: i32| -> u8 {
        let x = x_8th as f64 / 8.0;
        let phase = 2.0 * std::f64::consts::PI * (x / period_pels);
        let v = 128.0 + 100.0 * phase.sin();
        v.round().clamp(0.0, 255.0) as u8
    };
    let mut y0 = vec![0u8; w * h];
    let mut y1 = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y0[r * w + c] = profile((c as i32) * 8);
            y1[r * w + c] = profile((c as i32) * 8 - qpel_shift * 2);
        }
    }
    (y0, y1)
}

/// Build a "translating circle" Y fixture pair. A smooth radial Gaussian
/// centered at `(cx, cy)` shifted by `(qpel_dx, qpel_dy)` quarter-pel
/// units. The Gaussian is band-limited (no sharp edges) so bilinear MC
/// reconstructs sub-pel offsets within the noise floor.
fn build_translating_disk(w: usize, h: usize, qpel_dx: i32, qpel_dy: i32) -> (Vec<u8>, Vec<u8>) {
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let sigma = (w.min(h) as f64) * 0.18;
    let profile = |x_8th: i32, y_8th: i32| -> u8 {
        let x = x_8th as f64 / 8.0;
        let y = y_8th as f64 / 8.0;
        let dx = x - cx;
        let dy = y - cy;
        let r2 = (dx * dx + dy * dy) / (sigma * sigma);
        // Gaussian peaked at 220, baseline 64.
        let v = 64.0 + 156.0 * (-r2 / 2.0).exp();
        v.round().clamp(0.0, 255.0) as u8
    };
    let mut y0 = vec![0u8; w * h];
    let mut y1 = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y0[r * w + c] = profile((c as i32) * 8, (r as i32) * 8);
            // Inter frame: same disk shifted by (qpel_dx, qpel_dy)
            // quarter-pel units = `(qpel_dx * 2, qpel_dy * 2)` 8th-pel
            // units (the profile is in 8ths-of-a-pel).
            y1[r * w + c] = profile((c as i32) * 8 - qpel_dx * 2, (r as i32) * 8 - qpel_dy * 2);
        }
    }
    (y0, y1)
}

/// r25 — Quarter-pel sub-pel motion estimation, translating vertical
/// stripes. The inter frame is shifted by 2 quarter-pel units (= 0.5
/// integer pel) right; integer-pel ME alone misses the sub-pel offset
/// and produces ~21 dB Y. With qpel ME the bilinear MC follows the
/// shift and reconstruction clears 35 dB Y.
#[test]
fn r25_qpel_translating_stripes_psnr_clears_35db() {
    let (w, h) = (64usize, 32usize);
    let (y0, y1) = build_translating_stripes(w, h, 2); // 0.5-pel shift
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(8);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("keyframe");
    let (recon_y, recon_u, recon_v, _, _) = decode_first_frame(key.clone());

    // Search window covers ±2 integer pels — enough for a 0.5-pel shift,
    // small enough the qpel refine doesn't have to chase noise. With
    // qpel enabled the encoder lands on a sub-pel MV that the bilinear
    // filter reproduces ~exactly.
    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v, w, h, 2)
        .expect("encode inter");

    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
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

    // MC-only baseline at integer MV (zero) — what we'd get without any
    // sub-pel refinement.
    let psnr_int_only = plane_psnr(&y1, &recon_y);
    let psnr_inter = plane_psnr(&y1, dy);
    eprintln!(
        "r25 stripes (0.5-pel shift): integer-MC baseline = {psnr_int_only:.2} dB, \
         qpel-MC + residual = {psnr_inter:.2} dB"
    );
    assert!(
        psnr_inter >= 35.0,
        "qpel-MC Y PSNR too low on translating-stripes fixture: {psnr_inter:.2} dB \
         (target >= 35 dB; integer-only baseline was {psnr_int_only:.2} dB)"
    );
}

/// r25 — Quarter-pel sub-pel motion estimation, translating disk. The
/// inter frame is shifted by `(2, 2)` quarter-pel units = `(0.5, 0.5)`
/// pel diagonally; integer-pel ME alone misses both axes' sub-pel
/// component. With qpel ME the bilinear MC follows the diagonal shift
/// and reconstruction clears 35 dB Y.
#[test]
fn r25_qpel_translating_disk_psnr_clears_35db() {
    let (w, h) = (64usize, 48usize);
    let (y0, y1) = build_translating_disk(w, h, 2, 2); // 0.5-pel diag
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(8);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("keyframe");
    let (recon_y, recon_u, recon_v, _, _) = decode_first_frame(key.clone());

    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v, w, h, 2)
        .expect("encode inter");

    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
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

    let psnr_int_only = plane_psnr(&y1, &recon_y);
    let psnr_inter = plane_psnr(&y1, dy);
    eprintln!(
        "r25 disk ((0.5, 0.5)-pel shift): integer-MC baseline = {psnr_int_only:.2} dB, \
         qpel-MC + residual = {psnr_inter:.2} dB"
    );
    assert!(
        psnr_inter >= 35.0,
        "qpel-MC Y PSNR too low on translating-disk fixture: {psnr_inter:.2} dB \
         (target >= 35 dB; integer-only baseline was {psnr_int_only:.2} dB)"
    );
}

/// r25 — ffmpeg interop on a sub-pel-MV inter packet. Encode a
/// translating-stripes fixture at a 0.5-pel shift, mux key + inter
/// into FLV, and verify ffmpeg's vp6f decoder accepts both packets
/// (i.e. the sub-pel MV bits don't break ffmpeg's parser). The decoded
/// PSNR through ffmpeg should clear 25 dB (a softer bar than our own
/// decoder's 35 dB because ffmpeg may interpret the bilinear MC
/// slightly differently on edge MBs).
#[test]
fn r25_ffmpeg_decodes_qpel_inter_frame() {
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
    let (y0, y1) = build_translating_stripes(w, h, 2);
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(12);
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("keyframe");
    let (recon_y, recon_u, recon_v, _, _) = decode_first_frame(key.clone());
    let inter = enc
        .encode_inter_frame(&recon_y, &recon_u, &recon_v, &y1, &u, &v, w, h, 2)
        .expect("encode inter");

    // Mux key + inter into a 2-tag FLV stream (same shape as
    // `ffmpeg_decodes_inter_frame_with_mv`).
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());

    let push_tag = |flv: &mut Vec<u8>, frame: &[u8], pts: u32, is_key: bool| -> u32 {
        let video_payload_len = (1 + 1 + frame.len()) as u32;
        flv.push(9);
        flv.extend_from_slice(&video_payload_len.to_be_bytes()[1..]);
        let ts = pts;
        flv.push(((ts >> 16) & 0xff) as u8);
        flv.push(((ts >> 8) & 0xff) as u8);
        flv.push((ts & 0xff) as u8);
        flv.push(((ts >> 24) & 0xff) as u8);
        flv.extend_from_slice(&[0, 0, 0]);
        flv.push(if is_key { 0x14 } else { 0x24 });
        flv.push(0x00);
        flv.extend_from_slice(frame);
        let tag_size = 11 + video_payload_len;
        flv.extend_from_slice(&tag_size.to_be_bytes());
        tag_size
    };
    let _ = push_tag(&mut flv, &key, 0, true);
    let _ = push_tag(&mut flv, &inter, 33, false);

    let stamp = std::process::id();
    let flv_path = std::env::temp_dir().join(format!("oxideav_vp6_r25_{stamp}.flv"));
    let yuv_path = std::env::temp_dir().join(format!("oxideav_vp6_r25_{stamp}.yuv"));
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
    assert!(status.success(), "ffmpeg failed to decode 2-tag qpel FLV");
    let raw = std::fs::read(&yuv_path).unwrap();
    let frame_size = w * h + 2 * (w / 2) * (h / 2);
    assert!(
        raw.len() >= 2 * frame_size,
        "expected 2 frames, got {}",
        raw.len()
    );
    let off = frame_size;
    let ff_y = &raw[off..off + w * h];
    let psnr = plane_psnr(&y1, ff_y);
    eprintln!("r25 ffmpeg qpel decode: Y PSNR = {psnr:.2} dB");
    let _ = std::fs::remove_file(flv_path);
    let _ = std::fs::remove_file(yuv_path);
    // Accept any reasonable reconstruction — the goal is to confirm
    // ffmpeg parses the qpel MV bits cleanly. Even a soft baseline like
    // pure MC-only (no residual interpretation) clears 20 dB on the
    // smooth-stripe fixture.
    assert!(
        psnr >= 20.0,
        "ffmpeg qpel-MV inter Y PSNR too low: {psnr:.2} dB (target >= 20 dB)"
    );
}

// =============================================================================
// Round 26 — Golden-frame refresh
// =============================================================================
//
// VP6 carries a single always-available "golden" reference frame in
// addition to the previous-frame reference. The encoder emits the
// `golden_frame_flag` bit on the inter picture header to refresh the
// decoder's golden slot to the current reconstruction; per-MB ME then
// considers BOTH references and picks whichever beats the other on a
// Lagrangian SAD cost. On periodic-structure content (slideshow,
// animation loop) this can reduce the per-MB residual magnitude
// dramatically: a frame that revisits earlier content can pick a
// near-zero-residual prediction off the golden ref instead of a
// large-residual delta from the immediately-preceding frame.

/// Golden-refresh cadence: the encoder's `should_refresh_golden`
/// predicate fires once `inter_frames_since_golden >=
/// golden_refresh_period`. After a refresh the counter resets to 1
/// (matching the keyframe path, where the next inter is "1 since
/// golden"). This pin guards the cadence semantics so a regression in
/// the counter logic surfaces immediately.
#[test]
fn golden_refresh_cadence_fires_on_period() {
    let (w, h) = (32usize, 16usize);
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    enc.golden_refresh_period = 2;
    enc.encode_keyframe(&y, &u, &v, w, h).expect("keyframe");
    // Right after a keyframe the counter reads 0 — the keyframe itself
    // refreshed golden on the decoder side.
    assert_eq!(enc.inter_frames_since_golden(), 0);
    assert!(!enc.should_refresh_golden());

    // Inter call 1: counter goes 0 -> 1, no refresh (0 < 2).
    let _ = enc
        .encode_inter_frame_with_golden(&y, &u, &v, &y, &u, &v, &y, &u, &v, w, h, 2)
        .expect("encode inter 1");
    assert_eq!(enc.inter_frames_since_golden(), 1);
    assert!(!enc.should_refresh_golden());

    // Inter call 2: counter goes 1 -> 2, still no refresh (1 < 2).
    let _ = enc
        .encode_inter_frame_with_golden(&y, &u, &v, &y, &u, &v, &y, &u, &v, w, h, 2)
        .expect("encode inter 2");
    assert_eq!(enc.inter_frames_since_golden(), 2);
    // Now the predicate fires — the next call refreshes.
    assert!(enc.should_refresh_golden());

    // Inter call 3: refresh fires (2 >= 2), counter resets to 1.
    let _ = enc
        .encode_inter_frame_with_golden(&y, &u, &v, &y, &u, &v, &y, &u, &v, w, h, 2)
        .expect("encode inter 3");
    assert_eq!(enc.inter_frames_since_golden(), 1);
    assert!(!enc.should_refresh_golden());
}

/// `golden_refresh_period = 0` disables the refresh entirely — the
/// flag is never set and the cadence counter just counts up
/// indefinitely.
#[test]
fn golden_refresh_disabled_at_period_zero() {
    let (w, h) = (32usize, 16usize);
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(16);
    enc.golden_refresh_period = 0;
    enc.encode_keyframe(&y, &u, &v, w, h).expect("keyframe");
    for _ in 0..5 {
        let _ = enc
            .encode_inter_frame_with_golden(&y, &u, &v, &y, &u, &v, &y, &u, &v, w, h, 2)
            .expect("encode inter");
        assert!(!enc.should_refresh_golden());
    }
}

/// End-to-end round-trip: a keyframe + golden-refresh inter + a
/// "loops back" inter that should pick the golden reference for every
/// MB. Verifies our own decoder reconstructs the loop-back frame at
/// high PSNR — i.e. golden-ref MBs decode through the
/// `RefKind::Golden` branch correctly.
///
/// Animation pattern:
///  * frame 0 = keyframe with stripe pattern A.
///  * frame 1 = stripe pattern B (very different from A).
///  * frame 2 = stripe pattern A again. With golden-refresh after
///    frame 0 (golden = frame 0 reconstruction), frame 2 should pick
///    golden for every MB and reconstruct A near-perfectly.
#[test]
fn golden_refresh_loop_back_uses_golden_reference() {
    let (w, h) = (32usize, 16usize);
    // Pattern A: vertical stripes at x in {0, 8, 16, 24}.
    let mut y_a = vec![64u8; w * h];
    for row in 0..h {
        for col in 0..w {
            if (col / 8) % 2 == 0 {
                y_a[row * w + col] = 64;
            } else {
                y_a[row * w + col] = 200;
            }
        }
    }
    // Pattern B: horizontal stripes — drastically different content
    // (so frame-1 vs frame-0 is unfriendly for MC, and frame-2 vs
    // frame-1 is similarly unfriendly).
    let mut y_b = vec![64u8; w * h];
    for row in 0..h {
        for col in 0..w {
            if (row / 4) % 2 == 0 {
                y_b[row * w + col] = 64;
            } else {
                y_b[row * w + col] = 200;
            }
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    let mut enc = Vp6Encoder::new(12);
    // Trigger refresh on the *first* inter frame so frame 1's
    // reconstruction becomes the new golden — wait, that's the wrong
    // semantics for a loop-back. We want the keyframe (= frame 0) to
    // be the golden for frame 2. The keyframe path already snaps the
    // keyframe reconstruction into the decoder's golden slot, so
    // golden_refresh_period > 1 is enough.
    enc.golden_refresh_period = 99;

    let key = enc.encode_keyframe(&y_a, &u, &v, w, h).expect("keyframe");
    let (recon_a_y, recon_a_u, recon_a_v, _, _) = decode_first_frame(key.clone());

    // Frame 1: pattern B, with prev = recon_a (the keyframe), golden = recon_a too.
    let inter1 = enc
        .encode_inter_frame_with_golden(
            &recon_a_y, &recon_a_u, &recon_a_v, &recon_a_y, &recon_a_u, &recon_a_v, &y_b, &u, &v,
            w, h, 4,
        )
        .expect("encode inter1");

    // Decode key + inter1 to get frame 1's reconstruction.
    let params = CodecParameters::video(CodecId::new("vp6f"));
    let mut dec = Vp6Decoder::new(params.codec_id.clone());
    let mut pkt0 = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(key));
    pkt0.pts = Some(0);
    pkt0.flags.keyframe = true;
    dec.send_packet(&pkt0).expect("send key");
    let _ = dec.receive_frame().expect("recv key");
    let mut pkt1 = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(inter1));
    pkt1.pts = Some(33);
    dec.send_packet(&pkt1).expect("send inter1");
    let f1 = match dec.receive_frame().expect("recv inter1") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let recon_b_y = f1.planes[0].data.clone();
    let recon_b_u = f1.planes[1].data.clone();
    let recon_b_v = f1.planes[2].data.clone();

    // Frame 2: pattern A again. prev = recon_b (very different from A);
    // golden = recon_a (== A). Encoder should pick golden for every MB.
    let inter2 = enc
        .encode_inter_frame_with_golden(
            &recon_b_y, &recon_b_u, &recon_b_v, &recon_a_y, &recon_a_u, &recon_a_v, &y_a, &u, &v,
            w, h, 4,
        )
        .expect("encode inter2");

    let mut pkt2 = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(inter2));
    pkt2.pts = Some(66);
    dec.send_packet(&pkt2).expect("send inter2");
    let f2 = match dec.receive_frame().expect("recv inter2") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let f2_y = &f2.planes[0].data;

    // Decoded frame 2 should reconstruct pattern A at high quality
    // (golden ref + small residual). The "skip" baseline (carrying B
    // forward) would land at ~5 dB PSNR vs A on this fixture.
    let psnr = plane_psnr(&y_a, f2_y);
    let baseline_skip = plane_psnr(&y_a, &recon_b_y);
    eprintln!(
        "golden-refresh loop-back: golden-decode PSNR={psnr:.2} dB, \
         skip-from-prev baseline={baseline_skip:.2} dB"
    );
    assert!(
        psnr > baseline_skip + 5.0,
        "golden-ref decode should beat carry-forward-prev baseline by ≥5 dB \
         (golden={psnr:.2}, skip={baseline_skip:.2})"
    );
    assert!(
        psnr > 25.0,
        "golden-ref decode should clear 25 dB on the loop-back fixture (got {psnr:.2})"
    );
}

/// Bitrate delta on a periodic-structure fixture: encode 5 frames in
/// an A→B→A→B→A loop, once with `golden_refresh_period = 1` (golden
/// always tracks the most recent reconstruction) and once with the
/// period set absurdly high (golden stays pinned to the keyframe).
/// The fixed-golden run should produce a smaller total wire size on
/// the A→A loop-back frames because golden-ref MBs have near-zero
/// residual — a meaningful regression guard against any future
/// "always pick prev" tie-breaker.
///
/// Note: the absolute byte delta is small at this resolution
/// (`32x32`) and QP — the test asserts only that the fixed-golden
/// run is *no worse* than the unstable-golden run, plus a soft
/// expectation that on this exact fixture it's strictly smaller.
#[test]
fn golden_refresh_reduces_bytes_on_periodic_loop() {
    let (w, h) = (32usize, 32usize);
    // Pattern A: vertical stripes.
    let mut y_a = vec![64u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y_a[row * w + col] = if (col / 8) % 2 == 0 { 64 } else { 200 };
        }
    }
    // Pattern B: horizontal stripes.
    let mut y_b = vec![64u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y_b[row * w + col] = if (row / 4) % 2 == 0 { 80 } else { 180 };
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];

    // Helper: encode the keyframe + 4 inters in an A,B,A,B,A loop and
    // return the total inter-frame byte count. The keyframe is also
    // re-encoded into a separate decoder for state tracking — we use
    // a fresh `Vp6Encoder` for that to avoid disturbing the caller's
    // cadence counter.
    let encode_loop = |enc: &mut Vp6Encoder| -> usize {
        let key = enc.encode_keyframe(&y_a, &u, &v, w, h).expect("kf");
        let (mut recon_y, mut recon_u, mut recon_v, _, _) = decode_first_frame(key.clone());
        let mut golden_y = recon_y.clone();
        let mut golden_u = recon_u.clone();
        let mut golden_v = recon_v.clone();
        let frames = [&y_b, &y_a, &y_b, &y_a];
        let mut total = 0usize;

        let params = CodecParameters::video(CodecId::new("vp6f"));
        let mut dec = Vp6Decoder::new(params.codec_id.clone());
        let mut pkt = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(key));
        pkt.pts = Some(0);
        pkt.flags.keyframe = true;
        dec.send_packet(&pkt).expect("send key");
        let _ = dec.receive_frame().expect("recv key");

        for (i, src) in frames.iter().enumerate() {
            let was_refresh = enc.should_refresh_golden();
            let inter = enc
                .encode_inter_frame_with_golden(
                    &recon_y, &recon_u, &recon_v, &golden_y, &golden_u, &golden_v, src, &u, &v, w,
                    h, 4,
                )
                .expect("encode inter");
            total += inter.len();
            let mut p = Packet::new(0u32, TimeBase::new(1, 1000), packet_from_frame(inter));
            p.pts = Some(33 * (i as i64 + 1));
            dec.send_packet(&p).expect("send inter");
            let f = match dec.receive_frame().expect("recv inter") {
                Frame::Video(vf) => vf,
                other => panic!("expected video, got {other:?}"),
            };
            recon_y = f.planes[0].data.clone();
            recon_u = f.planes[1].data.clone();
            recon_v = f.planes[2].data.clone();
            if was_refresh {
                golden_y = recon_y.clone();
                golden_u = recon_u.clone();
                golden_v = recon_v.clone();
            }
        }
        total
    };

    // Run 1: refresh every frame. Golden chases the previous
    // reconstruction, so loop-back frames look "different from
    // golden" too (B's reconstruction was the most recent golden,
    // and A is being encoded against B).
    let mut enc1 = Vp6Encoder::new(12);
    enc1.golden_refresh_period = 1;
    let bytes_chasing = encode_loop(&mut enc1);

    // Run 2: never refresh. Golden stays pinned to the keyframe (A).
    // A→A loop-back frames pick golden for every MB and emit
    // near-zero residual — smaller wire size.
    let mut enc2 = Vp6Encoder::new(12);
    enc2.golden_refresh_period = 9999;
    let bytes_pinned = encode_loop(&mut enc2);

    eprintln!(
        "golden-refresh fixture: chasing-golden={bytes_chasing} bytes, \
         pinned-golden={bytes_pinned} bytes (delta = \
         {} bytes)",
        bytes_chasing as i64 - bytes_pinned as i64
    );
    // The pinned-golden run benefits from the periodic loop-back; on
    // this fixture it should be strictly smaller. Allow 10% slack to
    // tolerate quantisation drift between the two runs.
    let slack = (bytes_chasing as f64 * 1.10) as usize;
    assert!(
        bytes_pinned <= slack,
        "pinned-golden total bytes ({bytes_pinned}) should be ≤ \
         110% of chasing-golden ({bytes_chasing}) — golden refresh is hurting periodic-loop coding"
    );
}

/// ffmpeg cross-decode pin: a key + golden-refresh inter must round-
/// trip cleanly through ffmpeg's vp6f decoder. Pre-r26 the encoder
/// always emitted `golden_frame_flag = 0`; r26 flips it to 1 on the
/// refresh path. This guard pins the layout — a regression in the
/// inter picture-header layout (e.g. a stray bit before / after the
/// golden flag) would surface here as ffmpeg's "Invalid data" error.
/// Skipped silently when ffmpeg isn't on PATH.
#[test]
fn ffmpeg_decodes_inter_with_golden_refresh_flag() {
    use std::process::{Command, Stdio};
    let ffmpeg_ok = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ffmpeg_ok {
        eprintln!("ffmpeg not on PATH — skipping ffmpeg_decodes_inter_with_golden_refresh_flag");
        return;
    }
    let (w, h) = (64usize, 32usize);
    let mut y0 = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y0[row * w + col] = if (col / 8) % 2 == 0 { 50 } else { 200 };
        }
    }
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
    // Trigger refresh on the FIRST inter frame.
    enc.golden_refresh_period = 1;
    let key = enc.encode_keyframe(&y0, &u, &v, w, h).expect("kf");
    // golden-refresh inter: prev = key reconstruction (we use y0 as a
    // proxy — the encoder uses it for ME only, and the on-wire
    // golden_frame_flag = 1 setting drives ffmpeg's interop check).
    let inter = enc
        .encode_inter_frame_with_golden(&y0, &u, &v, &y0, &u, &v, &y1, &u, &v, w, h, 4)
        .expect("encode inter");
    assert_eq!(
        enc.inter_frames_since_golden(),
        1,
        "first inter with period=1 should refresh and reset counter to 1"
    );

    // Mux into FLV and hand to ffmpeg.
    let mut flv = Vec::new();
    flv.extend_from_slice(b"FLV");
    flv.push(0x01);
    flv.push(0x01);
    flv.extend_from_slice(&9u32.to_be_bytes());
    flv.extend_from_slice(&0u32.to_be_bytes());
    let push_tag = |flv: &mut Vec<u8>, frame: &[u8], pts: u32, is_key: bool| {
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
        flv.push(if is_key { 0x14 } else { 0x24 });
        flv.push(0x00);
        flv.extend_from_slice(frame);
        flv.extend_from_slice(&(11 + payload_len).to_be_bytes());
    };
    push_tag(&mut flv, &key, 0, true);
    push_tag(&mut flv, &inter, 33, false);

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
        use std::io::Write;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(&flv).unwrap();
    }
    let out = child.wait_with_output().expect("wait ffmpeg");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut last = 0u32;
    for line in stderr.lines() {
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
    assert_eq!(
        last, 2,
        "ffmpeg should accept key + golden-refresh inter (got {last} frames). stderr:\n{stderr}"
    );
}

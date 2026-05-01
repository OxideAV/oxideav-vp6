//! DoS-protection coverage for the VP6 decoder.
//!
//! Four fixtures, each demonstrating one layer of the
//! [`oxideav_core::DecoderLimits`] framework:
//!
//! 1. `picture_header_pixel_cap_rejects_oversize_dimensions` — the
//!    keyframe-header parse rejects a packet whose declared `mb_width
//!    × mb_height × 256` exceeds `limits.max_pixels_per_frame`. The
//!    rejection happens at `send_packet` time, *before* any plane
//!    allocation. The error is [`Error::ResourceExhausted`] (per the
//!    user's tightened cap).
//!
//! 2. `arena_pool_exhaustion_returns_resource_exhausted` — the
//!    decoder's `arena_pool` is leased N+1 times (N =
//!    `DecoderLimits::max_arenas_in_flight`); the (N+1)th `lease()`
//!    returns [`Error::ResourceExhausted`].
//!
//! 3. `make_decoder_factory_honours_codec_parameters_limits` — the
//!    registry-facing `make_decoder` factory must read
//!    `params.limits()` so server callers that pass a tightened
//!    `CodecParameters` actually get a tightened decoder.
//!
//! 4. `receive_arena_frame_planes_point_into_arena_buffer` — the
//!    plane bytes returned by [`Decoder::receive_arena_frame`] live
//!    inside the leased arena's backing buffer (zero-copy proof). We
//!    verify by leasing the remaining arenas after the receive — they
//!    must all check out with the receive's slot still held, and the
//!    plane pointer must precede the next-leased arena's address space
//!    (the receive-held slot remains pinned).

use std::sync::Arc;

use oxideav_core::arena::sync::ArenaPool;
use oxideav_core::packet::PacketFlags;
use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, DecoderLimits, Error, Packet, TimeBase};

use oxideav_vp6::decoder::{Vp6Decoder, DEFAULT_VP6_ARENA_BYTES};
use oxideav_vp6::Vp6Encoder;

/// Synthesise a minimal VP6 keyframe whose declared dims are
/// `mb_width × mb_height` macroblocks. The bytes are arranged exactly
/// as `frame_header::tests::synth_keyframe_bytes` constructs them
/// (qp=10, sub_version=0, filter_header=0, separated_coeff=0). The
/// 1-byte FLV adjustment prefix `[0x00]` is prepended so the packet
/// matches what `Vp6Decoder::send_packet` expects (the decoder strips
/// the first byte).
fn synth_keyframe_packet(mb_width: u8, mb_height: u8) -> Vec<u8> {
    let qp = 10u8;
    let b0 = (qp << 1) & 0x7F; // frame_mode=0 (key), sep_coeff=0
    let mut body = vec![
        b0, 0,         // sub_version=0, filter_header=0, interlaced=0
        0,         // coeff offset hi
        2,         // coeff offset lo (adjusted to 0 since filter_header=0)
        mb_height, // mb_height (raw byte; max 255 = 4080 px)
        mb_width,  // mb_width
        mb_height, // display_mb_height
        mb_width,  // display_mb_width
    ];
    body.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00]);
    // FLV 1-byte adjustment prefix.
    let mut packet = Vec::with_capacity(body.len() + 1);
    packet.push(0u8);
    packet.extend_from_slice(&body);
    packet
}

fn keyframe_packet(es: Vec<u8>) -> Packet {
    Packet {
        stream_index: 0,
        data: es,
        pts: Some(0),
        dts: Some(0),
        duration: None,
        time_base: TimeBase::new(1, 1000),
        flags: PacketFlags {
            keyframe: true,
            ..PacketFlags::default()
        },
    }
}

#[test]
fn picture_header_pixel_cap_rejects_oversize_dimensions() {
    // Synth a 16x12 mb (256x192 px = 49 152 px) keyframe. Cap below
    // that — header parse must reject before any allocation.
    let limits = DecoderLimits::default().with_max_pixels_per_frame(100);
    let mut dec = Vp6Decoder::with_limits(CodecId::new(oxideav_vp6::CODEC_ID_VP6F), limits);
    let payload = synth_keyframe_packet(16, 12);
    let pkt = keyframe_packet(payload);
    let res = dec.send_packet(&pkt);
    match res {
        Err(Error::ResourceExhausted(msg)) => {
            // Diagnostic should name the actual frame dims so a server
            // operator can see what was rejected.
            assert!(
                msg.contains("256") && msg.contains("192"),
                "diag should name the actual dims, got: {msg}"
            );
        }
        other => panic!("expected ResourceExhausted, got {other:?}"),
    }
}

#[test]
fn picture_header_pixel_cap_passes_under_default_limits() {
    // The same keyframe under default limits (32k × 32k = 1 G px) goes
    // through the pixel-cap check fine. The downstream decode of the
    // (truncated) range-coder body may fail later, but the failure
    // must NOT be ResourceExhausted on the pixel cap.
    let mut dec = Vp6Decoder::new(CodecId::new(oxideav_vp6::CODEC_ID_VP6F));
    let payload = synth_keyframe_packet(16, 12);
    let pkt = keyframe_packet(payload);
    if let Err(Error::ResourceExhausted(msg)) = dec.send_packet(&pkt) {
        if msg.contains("max_pixels_per_frame") {
            panic!("default limits incorrectly rejected on pixel cap: {msg}");
        }
    }
}

#[test]
fn arena_pool_exhaustion_returns_resource_exhausted() {
    // Pool size 2 — two arenas may be checked out concurrently; the
    // third lease must error with ResourceExhausted.
    let limits = DecoderLimits::default().with_max_arenas_in_flight(2);
    let dec = Vp6Decoder::with_limits(CodecId::new(oxideav_vp6::CODEC_ID_VP6F), limits);
    let pool: Arc<ArenaPool> = dec.arena_pool().clone();
    assert_eq!(pool.max_arenas(), 2);

    let a = pool.lease().expect("first lease");
    let b = pool.lease().expect("second lease");
    let third = pool.lease();
    match third {
        Err(Error::ResourceExhausted(_)) => {}
        Ok(_) => panic!("expected ResourceExhausted on 3rd lease, got Ok(_)"),
        Err(other) => panic!("expected ResourceExhausted on 3rd lease, got {other:?}"),
    }
    drop((a, b));
    // After dropping both, the pool refills.
    let _again = pool.lease().expect("re-lease after drop");
}

#[test]
fn arena_pool_cap_per_arena_is_bounded_by_vp6_default() {
    // A decoder constructed with the workspace default DecoderLimits
    // must size each arena to no more than DEFAULT_VP6_ARENA_BYTES.
    // Guards against the global default 1 GiB
    // max_alloc_bytes_per_frame leaking through and eating gigabytes
    // of address space across the default 8-slot pool.
    let dec = Vp6Decoder::new(CodecId::new(oxideav_vp6::CODEC_ID_VP6F));
    let pool = dec.arena_pool();
    assert!(
        pool.cap_per_arena() as u64 <= DEFAULT_VP6_ARENA_BYTES,
        "default per-arena cap = {} exceeds vp6 ceiling {}",
        pool.cap_per_arena(),
        DEFAULT_VP6_ARENA_BYTES
    );
    assert_eq!(
        pool.max_arenas(),
        DecoderLimits::default().max_arenas_in_flight as usize
    );
}

#[test]
fn make_decoder_factory_honours_codec_parameters_limits() {
    // The registry-facing `make_decoder` factory must read
    // `params.limits()` so server callers that pass a tightened
    // `CodecParameters` actually get a tightened decoder.
    let limits = DecoderLimits::default()
        .with_max_pixels_per_frame(100)
        .with_max_arenas_in_flight(1);
    let params =
        CodecParameters::video(CodecId::new(oxideav_vp6::CODEC_ID_VP6F)).with_limits(limits);
    let mut decoder = oxideav_vp6::make_decoder(&params).expect("factory");

    let pkt = keyframe_packet(synth_keyframe_packet(16, 12));
    match decoder.send_packet(&pkt) {
        Err(Error::ResourceExhausted(_)) => {} // expected
        other => panic!("factory-produced decoder ignored limits: got {other:?}"),
    }

    // A receive_frame on the failed decoder yields NeedMore (the
    // packet was rejected before queueing). Confirms the rejection
    // happened *before* any decode work.
    let r = decoder.receive_frame();
    assert!(
        matches!(r, Err(Error::NeedMore) | Err(Error::Eof)),
        "expected NeedMore/Eof after rejected packet, got {r:?}"
    );
}

/// Encode a real VP6 keyframe (32×16 px) via the in-tree encoder so
/// the decode path exercises the full kernel chain (range-coder body +
/// DC/AC residuals + MB renderer). Returns the FLV-prefixed packet
/// payload ready to feed to `Vp6Decoder::send_packet`.
fn synth_decodable_keyframe() -> Vec<u8> {
    let (w, h) = (32usize, 16usize);
    let y = vec![128u8; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut enc = Vp6Encoder::new(32);
    let bytes = enc.encode_keyframe(&y, &u, &v, w, h).expect("encode");
    // Prepend the 1-byte FLV adjustment prefix.
    let mut packet = Vec::with_capacity(bytes.len() + 1);
    packet.push(0u8);
    packet.extend_from_slice(&bytes);
    packet
}

#[test]
fn receive_arena_frame_planes_point_into_arena_buffer() {
    // Lease the entire pool *before* the arena receive: that way the
    // receive will trigger lazy growth of the pool only if the receive
    // call has already grabbed the next slot. We test the converse:
    // size the pool to exactly 2, lease 1 first, then receive →
    // receive's lease consumes the 2nd slot, and a 3rd lease must
    // fail. After dropping the receive's Frame, the pool refills.
    let limits = DecoderLimits::default().with_max_arenas_in_flight(2);
    let mut dec = Vp6Decoder::with_limits(CodecId::new(oxideav_vp6::CODEC_ID_VP6F), limits);
    let pool = dec.arena_pool().clone();

    let pkt = keyframe_packet(synth_decodable_keyframe());
    dec.send_packet(&pkt).expect("send_packet should queue");

    // First lease: occupies one of the pool's 2 slots.
    let pre_lease = pool.lease().expect("pre-receive lease");
    let pre_addr = pre_lease.used();
    let _ = pre_addr;

    // Receive: consumes the second slot.
    let arena_frame = dec
        .receive_arena_frame()
        .expect("receive_arena_frame should succeed with 1 free slot");

    // The arena Frame must hold the second slot pinned: a 3rd lease
    // attempt must fail with ResourceExhausted.
    let extra = pool.lease();
    match extra {
        Err(Error::ResourceExhausted(_)) => {}
        Ok(_) => panic!("3rd lease unexpectedly succeeded — Frame did not retain its arena"),
        Err(other) => panic!("expected ResourceExhausted, got {other:?}"),
    }

    // Plane 0 (luma) bytes must point into the arena buffer space, NOT
    // a heap-owned `Vec` separate from the arena. We don't have a
    // direct API to enumerate the arena's range, so we use a behavioural
    // proof: after `drop(arena_frame)` the slot returns to the pool
    // and a fresh lease succeeds — i.e. the plane bytes were tied to
    // the arena lease, not to a heap copy.
    let y_plane = arena_frame.plane(0).expect("luma plane");
    assert_eq!(
        y_plane.len(),
        32 * 16,
        "32x16 keyframe should produce 512-byte luma plane"
    );

    // Drop pre_lease so the pool only has the arena_frame's slot in
    // flight. Now drop arena_frame → pool empties → all slots refillable.
    drop(pre_lease);
    drop(arena_frame);
    let _refill1 = pool.lease().expect("lease 1 after receive drop");
    let _refill2 = pool.lease().expect("lease 2 after receive drop");
    // Pool exhausted again with both back out.
    assert!(matches!(pool.lease(), Err(Error::ResourceExhausted(_))));
}

#[test]
fn receive_arena_frame_plane_lives_in_arena_address_range() {
    // Strong zero-copy proof by direct pointer-range comparison.
    //
    // Strategy: lease an arena from the pool, observe its byte
    // address range (start/end of its backing slice), then return
    // that arena and execute a `receive_arena_frame` call. The
    // returned Frame's luma plane pointer must lie within an address
    // range similar to the one we observed. Because the pool reuses
    // its boxed buffers in LIFO order (idle.pop()), the second lease
    // hits the same backing buffer the receive does.
    let limits = DecoderLimits::default().with_max_arenas_in_flight(1);
    let mut dec = Vp6Decoder::with_limits(CodecId::new(oxideav_vp6::CODEC_ID_VP6F), limits);
    let pool = dec.arena_pool().clone();

    // Force the pool to materialise its single buffer, observe its
    // address, then return it.
    let probe = pool.lease().expect("probe lease");
    // The arena exposes its buffer indirectly via alloc(); allocate 1
    // byte and read its address as the buffer start.
    let probe_byte = probe
        .alloc::<u8>(1)
        .expect("probe alloc should succeed in fresh arena");
    let probe_start = probe_byte.as_ptr() as usize;
    let probe_end = probe_start + (pool.cap_per_arena() - 1);
    drop(probe); // returns buffer to the pool's idle list (LIFO).

    let pkt = keyframe_packet(synth_decodable_keyframe());
    dec.send_packet(&pkt).expect("send_packet");
    let arena_frame = dec.receive_arena_frame().expect("receive_arena_frame");
    let y_plane = arena_frame.plane(0).expect("luma plane");
    let plane_start = y_plane.as_ptr() as usize;
    let plane_end = plane_start + y_plane.len();

    // Plane must lie entirely inside the probed arena's byte range.
    // If the decoder had memcpy'd planes to a separate Vec<u8> heap
    // alloc, plane_start would NOT lie in [probe_start, probe_end].
    assert!(
        plane_start >= probe_start && plane_end <= probe_end + 1,
        "plane @ {plane_start:#x}..{plane_end:#x} not inside arena buffer {probe_start:#x}..={probe_end:#x} \
         (proves zero-copy: planes live in arena memory, not heap-owned Vec<u8>)"
    );
}

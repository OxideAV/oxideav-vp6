//! VP6 motion-vector component **encoder** — the bit-for-bit inverse of
//! [`crate::mv_decode::decode_mv_component`] (spec §11.1).
//!
//! The decoder reconstructs one signed MV component by a
//! `B(IsMvShortProbs)` discriminator, then either the Figure 11 short tree
//! (`decode_short_mv_magnitude`, magnitude `0..=7`) or the long bit-stream
//! walk (`decode_long_mv_magnitude`, magnitude `8..=127`), then a
//! `B(MvSignProbs)` sign read. This module performs the exact inverse:
//! given the signed component to transmit and the same per-axis
//! [`crate::mv_decode::MvProbs`] bank the decoder will read, it emits the
//! discriminator, the magnitude bits, and the sign bit so that
//! [`crate::mv_decode::decode_mv_component`] recovers the component
//! bit-for-bit.
//!
//! ## Short vs long choice
//!
//! The decoder's discriminator selects the short tree on
//! `B(IsMvShortProbs) == 1` and the long bit-stream on `0`. The magnitude
//! ranges are disjoint and exhaustive — short covers `0..=7`, long covers
//! `8..=255` (§11.1 documents the typical `8..=127` range, but the
//! bit-arithmetic admits up to `0xFF`) — so the choice is forced by
//! `|component|`: a magnitude in `0..=7` *must* go short (the long path
//! cannot represent `1..=7`, because its implicit-bit-3 rule sets bit 3
//! whenever the high nibble is clear, so the smallest long magnitude is
//! `0x08`), and a magnitude in `8..=255` *must* go long. This module emits
//! the discriminator accordingly.
//!
//! ## Sign
//!
//! Per §11.1 a zero magnitude still emits a sign bit (`B(MvSignProbs)`),
//! and the decoder negates a zero to a zero — so the encoder emits sign
//! `0` for a non-negative component (including zero) and `1` for a
//! negative one. The `(dx, dy)` pair encoder emits X then Y, mirroring
//! [`crate::mv_decode::decode_mv_pair`].
//!
//! ## Provenance
//!
//! Derived solely from the §11.1 decode functions this module inverts
//! ([`crate::mv_decode`], sequenced from `docs/video/vp6/vp6_format.pdf`
//! §11.1) plus the §7.3 [`crate::bool_coder::BoolEncoder`] (itself derived
//! from the §7.3 decode equations per errata #35). No third-party VP6
//! implementation was consulted.

use crate::bool_coder::BoolEncoder;
use crate::mv_decode::{MvProbs, MV_AXIS_X, MV_AXIS_Y, NUM_MV_AXES};

/// The largest magnitude a single MV component can carry (§11.1: the long
/// bit-stream is 8 bits, so the value tops out at `0xFF = 255`, but the
/// implicit bit-3 rule keeps `8..=127` for the typical range; the decoder
/// can in principle reach `255`). The encoder accepts `0..=255` and routes
/// `8..=255` through the long path.
pub const MAX_MV_MAGNITUDE: u32 = 0xFF;

/// Emit the Figure 11 short-MV magnitude tree, the exact inverse of
/// [`crate::mv_decode::decode_short_mv_magnitude`].
///
/// `magnitude` must be in `0..=7`. The decoder's short tree partitions
/// `0..=7` as:
///
/// ```text
/// B(short[0]) == 1 :  4 + (B(short[4])==1 ? 2 + B(short[6]) : B(short[5]))   -> 4..=7
/// B(short[0]) == 0 :  B(short[1])==1 ? 2 + B(short[3]) : B(short[2])         -> 0..=3
/// ```
///
/// The encoder walks the same nodes, emitting the bit that steers the
/// decoder toward `magnitude`'s leaf and reading the same `short[node]`
/// probability the decoder reads.
fn encode_short_mv_magnitude(enc: &mut BoolEncoder, magnitude: u32, short: &[u8; 7]) {
    debug_assert!(magnitude <= 7, "short magnitude must be 0..=7");
    if magnitude >= 4 {
        // node0 = 1: the 4..=7 subtree.
        enc.encode_bool(1, short[0]);
        let low = magnitude - 4; // 0..=3
        if low >= 2 {
            // node4 = 1: 2 + B(short[6])
            enc.encode_bool(1, short[4]);
            enc.encode_bool((low - 2) as u8, short[6]);
        } else {
            // node4 = 0: B(short[5])
            enc.encode_bool(0, short[4]);
            enc.encode_bool(low as u8, short[5]);
        }
    } else {
        // node0 = 0: the 0..=3 subtree.
        enc.encode_bool(0, short[0]);
        if magnitude >= 2 {
            // node1 = 1: 2 + B(short[3])
            enc.encode_bool(1, short[1]);
            enc.encode_bool((magnitude - 2) as u8, short[3]);
        } else {
            // node1 = 0: Vector = B(short[2])  (0 or 1)
            enc.encode_bool(0, short[1]);
            enc.encode_bool(magnitude as u8, short[2]);
        }
    }
}

/// Emit the long-MV bit-stream, the exact inverse of
/// [`crate::mv_decode::decode_long_mv_magnitude`].
///
/// `magnitude` must be in `8..=255`. The §11.1 implicit-bit-3 rule means
/// bit 3 is *not* transmitted when the high nibble (`magnitude & 0xF0`) is
/// zero — the decoder forces it to 1, so any magnitude in `8..=15`
/// reconstructs from its transmitted low three bits as
/// `(magnitude & 0x07) | 0x08 == magnitude` (every such magnitude already
/// has bit 3 set). When the high nibble is non-zero bit 3 *is* transmitted.
/// The decoder reads bits in the order `[0,1,2,7,6,5,4]` then conditionally
/// bit 3; the encoder emits the same bits in the same order.
fn encode_long_mv_magnitude(enc: &mut BoolEncoder, magnitude: u32, size: &[u8; 8]) {
    debug_assert!(
        (8..=0xFF).contains(&magnitude),
        "long magnitude must be 8..=255"
    );
    debug_assert!(
        (magnitude & 0xF0) != 0 || (magnitude & 0x08) != 0,
        "a zero-high-nibble long magnitude must have bit 3 set (8..=15)"
    );
    let bit = |n: u32| ((magnitude >> n) & 1) as u8;
    enc.encode_bool(bit(0), size[0]);
    enc.encode_bool(bit(1), size[1]);
    enc.encode_bool(bit(2), size[2]);
    enc.encode_bool(bit(7), size[7]);
    enc.encode_bool(bit(6), size[6]);
    enc.encode_bool(bit(5), size[5]);
    enc.encode_bool(bit(4), size[4]);

    // Bit 3 is implicit (set, not transmitted) when none of the high-order
    // bits are present; otherwise it is transmitted.
    if (magnitude & 0xF0) != 0 {
        enc.encode_bool(bit(3), size[3]);
    }
}

/// Encode one signed motion-vector component, the bit-for-bit inverse of
/// [`crate::mv_decode::decode_mv_component`].
///
/// Emits the `B(IsMvShortProbs)` discriminator (short for
/// `|component| <= 7`, long for `8..=255`), the magnitude path, then the
/// `B(MvSignProbs)` sign bit (`0` for non-negative — including zero — and
/// `1` for negative).
///
/// # Panics (debug)
///
/// If `|component|` exceeds [`MAX_MV_MAGNITUDE`].
pub fn encode_mv_component(enc: &mut BoolEncoder, component: i32, probs: &MvProbs) {
    let magnitude = component.unsigned_abs();
    debug_assert!(
        magnitude <= MAX_MV_MAGNITUDE,
        "MV component magnitude out of range"
    );

    if magnitude <= 7 {
        enc.encode_bool(1, probs.is_short);
        encode_short_mv_magnitude(enc, magnitude, &probs.short);
    } else {
        enc.encode_bool(0, probs.is_short);
        encode_long_mv_magnitude(enc, magnitude, &probs.size);
    }

    let sign_bit = if component < 0 { 1 } else { 0 };
    enc.encode_bool(sign_bit, probs.sign);
}

/// Encode a full motion-vector delta `(dx, dy)` pair, the inverse of
/// [`crate::mv_decode::decode_mv_pair`] (X component first, then Y).
pub fn encode_mv_pair(enc: &mut BoolEncoder, dx: i32, dy: i32, probs: &[MvProbs; NUM_MV_AXES]) {
    encode_mv_component(enc, dx, &probs[MV_AXIS_X]);
    encode_mv_component(enc, dy, &probs[MV_AXIS_Y]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;
    use crate::mv_decode::{decode_mv_component, decode_mv_pair};

    fn axis_probs(axis: usize) -> MvProbs {
        MvProbs::defaults(axis)
    }

    /// Round-trip every short-range component (`-7..=7`) through both axes.
    #[test]
    fn short_components_round_trip() {
        for axis in 0..NUM_MV_AXES {
            let probs = axis_probs(axis);
            for component in -7..=7 {
                let mut enc = BoolEncoder::new();
                encode_mv_component(&mut enc, component, &probs);
                let bytes = enc.finish();
                let mut bc = BoolCoder::new(&bytes).unwrap();
                let got = decode_mv_component(&mut bc, &probs).unwrap();
                assert_eq!(got, component, "axis {axis} short component {component}");
            }
        }
    }

    /// Round-trip representative long-range magnitudes (`8..=127`) and the
    /// boundary `0x08`, both signs, both axes.
    #[test]
    fn long_components_round_trip() {
        for axis in 0..NUM_MV_AXES {
            let probs = axis_probs(axis);
            for mag in [8i32, 9, 15, 16, 31, 64, 100, 127] {
                for &component in &[mag, -mag] {
                    let mut enc = BoolEncoder::new();
                    encode_mv_component(&mut enc, component, &probs);
                    let bytes = enc.finish();
                    let mut bc = BoolCoder::new(&bytes).unwrap();
                    let got = decode_mv_component(&mut bc, &probs).unwrap();
                    assert_eq!(got, component, "axis {axis} long component {component}");
                }
            }
        }
    }

    /// The full unsigned magnitude range `0..=255` round-trips (with a
    /// non-negative sign), covering high-nibble bit-3-transmitted paths.
    #[test]
    fn full_magnitude_range_round_trips() {
        let probs = axis_probs(0);
        for mag in 0..=255i32 {
            let mut enc = BoolEncoder::new();
            encode_mv_component(&mut enc, mag, &probs);
            let bytes = enc.finish();
            let mut bc = BoolCoder::new(&bytes).unwrap();
            let got = decode_mv_component(&mut bc, &probs).unwrap();
            assert_eq!(got, mag, "magnitude {mag}");
        }
    }

    /// Zero encodes through the short path and round-trips to zero
    /// (a sign bit is still emitted and a zero negates to a zero).
    #[test]
    fn zero_round_trips() {
        let probs = axis_probs(0);
        let mut enc = BoolEncoder::new();
        encode_mv_component(&mut enc, 0, &probs);
        let bytes = enc.finish();
        let mut bc = BoolCoder::new(&bytes).unwrap();
        assert_eq!(decode_mv_component(&mut bc, &probs).unwrap(), 0);
    }

    /// The `(dx, dy)` pair encoder round-trips both components and leaves
    /// the decoder's stream position consistent (full sweep of small +
    /// large mixed pairs).
    #[test]
    fn pair_round_trips() {
        let probs = [MvProbs::defaults(0), MvProbs::defaults(1)];
        for &(dx, dy) in &[(0, 0), (3, -5), (-7, 7), (16, -100), (-127, 64), (8, -8)] {
            let mut enc = BoolEncoder::new();
            encode_mv_pair(&mut enc, dx, dy, &probs);
            let bytes = enc.finish();
            let mut bc = BoolCoder::new(&bytes).unwrap();
            let got = decode_mv_pair(&mut bc, &probs).unwrap();
            assert_eq!(got, (dx, dy), "pair ({dx}, {dy})");
        }
    }
}

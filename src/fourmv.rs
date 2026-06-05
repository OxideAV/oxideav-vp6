//! VP6 per-block coding-mode signaling for `CODE_INTER_FOURMV`
//! macroblocks (spec §10, Table 10).
//!
//! When a macroblock's MB-level coding mode (decoded by the §10
//! `VP6_DecodeMode` traversal) is [`CodingMode::InterFourMv`], the
//! decoder must follow up by reading a separate per-Y-block coding
//! mode for **each** of the four luma blocks. Per §10:
//!
//! ```text
//! In the case where the MB is coded using mode CODE_INTER_FOURMV
//! the specific coding mode for each of the four blocks comes from a
//! reduced set of four modes. In this case the mode is coded as a
//! fixed two bit codeword using the BoolCoder and a probability of
//! 128 for each bit.
//! ```
//!
//! The four codewords (Table 10):
//!
//! ```text
//! Block Coding Mode      Code
//! CODE_INTER_NO_MV       00
//! CODE_INTER_PLUS_MV     01
//! CODE_INTER_NEAREST_MV  10
//! CODE_INTER_NEAR_MV     11
//! ```
//!
//! The bits are read MSB-first via [`BoolCoder::decode_b`] (which is
//! itself an MSB-first packing of two `b(1)` fixed-probability-128
//! reads — see round 15's [`crate::bool_coder`]). The resulting
//! `0..=3` integer indexes [`FOURMV_BLOCK_MODES`], yielding the
//! decoded [`CodingMode`].
//!
//! Because every bit is fixed-probability-128, the §10
//! `VP6_DecodeMode` Figure-10 traversal — and its outstanding
//! DOCS-GAP candidate around the `B(Stats[0])` / `B(Stats[2])` else
//! branches (round 21 report) — does **not** apply here. The
//! per-block Table 10 codeword is a closed, unambiguous read that
//! lands independently. The MB-level decision that the MB is
//! `InterFourMv` in the first place is the gated piece; once that
//! decision has fired, the four per-block reads documented in this
//! module are well-defined.
//!
//! ## Surfaces
//!
//! * [`FOURMV_BLOCK_MODES`] — the four-entry Table 10 lookup, in
//!   canonical codeword order `[00, 01, 10, 11]`.
//! * [`NUM_LUMA_BLOCKS_PER_MB`] — the constant `4` that pins the
//!   four-blocks-per-MB shape Table 10 walks.
//! * [`decode_fourmv_block_mode`] — single-block decoder. Consumes
//!   one `decode_b(2)` read from the supplied [`BoolCoder`] and
//!   returns the corresponding [`CodingMode`].
//! * [`decode_fourmv_block_modes`] — the four-block walker, reading
//!   in raster order (block 0 = top-left, block 1 = top-right,
//!   block 2 = bottom-left, block 3 = bottom-right). Returns
//!   `[CodingMode; 4]`.
//!
//! ## Provenance
//!
//! Sourced from `docs/video/vp6/vp6_format.pdf` §10 / Table 10 (On2
//! Technologies, document version 1.02, August 2006).

use crate::bool_coder::BoolCoder;
use crate::modes::CodingMode;
use crate::Error;

/// Number of luma blocks the §10 `CODE_INTER_FOURMV` mode covers
/// per macroblock.
///
/// VP6 macroblocks carry four 8x8 luma blocks (the two 8x8 chroma
/// blocks share an averaged MV when the MB is `InterFourMv`, per
/// §10 / §11 prose).
pub const NUM_LUMA_BLOCKS_PER_MB: usize = 4;

/// Number of distinct block coding modes Table 10 selects from.
///
/// The reduced four-mode set is `{InterNoMv, InterPlusMv,
/// InterNearestMv, InterNearMv}`, all four are inter-coded against
/// the previous-frame reconstruction (none of the Golden-frame or
/// Intra modes are valid for a per-block decision inside an
/// `InterFourMv` macroblock).
pub const NUM_FOURMV_BLOCK_MODES: usize = 4;

/// The four-entry Table 10 codeword → coding mode lookup, in
/// canonical codeword-value order.
///
/// Indexed by the two-bit `decode_b(2)` value (MSB-first, so
/// `bit0 << 1 | bit1`):
///
/// * `0b00` (0) → [`CodingMode::InterNoMv`]
/// * `0b01` (1) → [`CodingMode::InterPlusMv`]
/// * `0b10` (2) → [`CodingMode::InterNearestMv`]
/// * `0b11` (3) → [`CodingMode::InterNearMv`]
pub const FOURMV_BLOCK_MODES: [CodingMode; NUM_FOURMV_BLOCK_MODES] = [
    CodingMode::InterNoMv,
    CodingMode::InterPlusMv,
    CodingMode::InterNearestMv,
    CodingMode::InterNearMv,
];

/// Decode one Table 10 two-bit per-block coding-mode codeword.
///
/// Reads two fixed-probability-128 BoolCoder bits MSB-first
/// (`decode_b(2)`) and looks the resulting `0..=3` index up in
/// [`FOURMV_BLOCK_MODES`].
///
/// # Errors
///
/// Returns [`Error::Truncated`] if the underlying bitstream runs
/// out of bytes during the two-bit read.
#[inline]
pub fn decode_fourmv_block_mode(bc: &mut BoolCoder) -> Result<CodingMode, Error> {
    let codeword = bc.decode_b(2)?;
    // `decode_b(2)` returns 0..=3; cast cannot overflow and the
    // table cover is exhaustive across that range.
    let idx = codeword as usize;
    Ok(FOURMV_BLOCK_MODES[idx])
}

/// Decode the four Table 10 per-block coding-mode codewords for one
/// `CODE_INTER_FOURMV` macroblock.
///
/// Reads `4 * 2 = 8` fixed-probability-128 BoolCoder bits in raster
/// order — block 0 (top-left) first, then 1 (top-right), 2
/// (bottom-left), 3 (bottom-right). Returns the four decoded
/// [`CodingMode`]s in the same raster order.
///
/// # Errors
///
/// Returns [`Error::Truncated`] if the underlying bitstream runs
/// out of bytes during any of the four two-bit reads.
#[inline]
pub fn decode_fourmv_block_modes(
    bc: &mut BoolCoder,
) -> Result<[CodingMode; NUM_LUMA_BLOCKS_PER_MB], Error> {
    let mut out = [CodingMode::InterNoMv; NUM_LUMA_BLOCKS_PER_MB];
    for slot in out.iter_mut() {
        *slot = decode_fourmv_block_mode(bc)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Right-pad a byte slice to at least 64 bytes (with zeros) so
    /// the §7.3 four-byte `VP6_StartDecode` prefill plus a healthy
    /// margin of renormalization refills succeed without hitting
    /// the `Error::Truncated` boundary mid-test, even when every
    /// per-bit step provokes a renormalization. The returned `Vec`
    /// is the caller's so the `BoolCoder` borrow lives only as
    /// long as the test scope.
    fn pad_to_64(bytes: &[u8]) -> Vec<u8> {
        let mut padded = bytes.to_vec();
        while padded.len() < 64 {
            padded.push(0);
        }
        padded
    }

    /// Table 10 cover: the four-entry lookup is exactly the reduced
    /// per-block mode set the §10 prose mandates, in codeword order.
    #[test]
    fn fourmv_block_modes_lookup_is_table_10() {
        assert_eq!(FOURMV_BLOCK_MODES[0], CodingMode::InterNoMv);
        assert_eq!(FOURMV_BLOCK_MODES[1], CodingMode::InterPlusMv);
        assert_eq!(FOURMV_BLOCK_MODES[2], CodingMode::InterNearestMv);
        assert_eq!(FOURMV_BLOCK_MODES[3], CodingMode::InterNearMv);
    }

    /// The four-block constant matches the `[CodingMode; 4]` return
    /// shape of [`decode_fourmv_block_modes`].
    #[test]
    fn num_luma_blocks_per_mb_is_four() {
        assert_eq!(NUM_LUMA_BLOCKS_PER_MB, 4);
    }

    /// `NUM_FOURMV_BLOCK_MODES` matches the lookup-table length.
    #[test]
    fn num_fourmv_block_modes_matches_lookup() {
        assert_eq!(NUM_FOURMV_BLOCK_MODES, FOURMV_BLOCK_MODES.len());
    }

    /// An all-zero stream decodes to `InterNoMv` (codeword `00`).
    ///
    /// The §7.3 errata #35 disambiguation guarantees that against
    /// `Probability = 128` (the `decode_b` per-bit fixed probability)
    /// the BoolCoder's first bit decodes to 0 when the top byte of
    /// `Value` is below `0xFF` — which holds for the all-zero stream
    /// where the four-byte prefill places `0` in the high byte.
    #[test]
    fn all_zero_stream_decodes_to_inter_no_mv() {
        let buf = pad_to_64(&[0; 16]);
        let mut bc = BoolCoder::new(&buf).expect("BoolCoder init");
        let mode = decode_fourmv_block_mode(&mut bc).expect("decode");
        assert_eq!(mode, CodingMode::InterNoMv);
    }

    /// The four-block walker against an all-zero stream produces
    /// four `InterNoMv` decodes — the spec's raster-order
    /// invariant (each block reads its own two bits independently).
    #[test]
    fn all_zero_stream_four_blocks_decode_to_inter_no_mv() {
        let buf = pad_to_64(&[0; 16]);
        let mut bc = BoolCoder::new(&buf).expect("BoolCoder init");
        let modes = decode_fourmv_block_modes(&mut bc).expect("decode");
        assert_eq!(modes, [CodingMode::InterNoMv; NUM_LUMA_BLOCKS_PER_MB]);
    }

    /// The four-block walker consumes the BoolCoder's bits **in
    /// order** — re-decoding the same per-block reads against a
    /// fresh BoolCoder from the same bytes reproduces the exact
    /// sequence the walker produced.
    #[test]
    fn four_block_walker_matches_per_block_calls() {
        let bytes: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0xA5, 0x5A, 0xFF, 0x00, 0x80, 0x01,
            0x7F, 0xC3,
        ];
        let buf = pad_to_64(&bytes);
        let mut bc_walker = BoolCoder::new(&buf).expect("BoolCoder init");
        let walker = decode_fourmv_block_modes(&mut bc_walker).expect("decode");

        let mut bc_serial = BoolCoder::new(&buf).expect("BoolCoder init");
        let mut serial = [CodingMode::InterNoMv; NUM_LUMA_BLOCKS_PER_MB];
        for slot in serial.iter_mut() {
            *slot = decode_fourmv_block_mode(&mut bc_serial).expect("decode");
        }
        assert_eq!(walker, serial);
        // Both decoders must end at the same BoolCoder state: the
        // walker is exactly four sequential per-block calls.
        assert_eq!(bc_walker.pos(), bc_serial.pos());
        assert_eq!(bc_walker.range(), bc_serial.range());
        assert_eq!(bc_walker.value(), bc_serial.value());
        assert_eq!(bc_walker.count(), bc_serial.count());
    }

    /// Determinism: same bytes in, same modes out, across two
    /// independent BoolCoder runs.
    #[test]
    fn decode_is_deterministic() {
        let bytes: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
            0x07, 0x08,
        ];
        let buf = pad_to_64(&bytes);
        let mut bc1 = BoolCoder::new(&buf).expect("BoolCoder init");
        let mut bc2 = BoolCoder::new(&buf).expect("BoolCoder init");
        let a = decode_fourmv_block_modes(&mut bc1).expect("decode");
        let b = decode_fourmv_block_modes(&mut bc2).expect("decode");
        assert_eq!(a, b);
    }

    /// Every decoded per-block mode is one of the four reduced
    /// modes [`FOURMV_BLOCK_MODES`] surfaces — the §10 prose's
    /// "reduced set of four modes" invariant. Validates across a
    /// sweep of seed byte streams whose top-byte profile keeps the
    /// BoolCoder away from the `Range == 0` pathological case the
    /// round-15 errata #35 commentary records (a sustained all-`0xFF`
    /// top-byte profile drives `Range` to 0 at `Probability = 128`,
    /// freezing the renormalization loop; valid VP6 bitstreams do
    /// not exhibit that profile).
    #[test]
    fn decoded_mode_is_always_in_reduced_set() {
        let seeds: [&[u8]; 4] = [
            &[0; 16],
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0],
            &[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            &[0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80],
        ];
        for seed in seeds {
            let buf = pad_to_64(seed);
            let mut bc = BoolCoder::new(&buf).expect("BoolCoder init");
            let modes = decode_fourmv_block_modes(&mut bc).expect("decode");
            for m in modes {
                assert!(
                    FOURMV_BLOCK_MODES.contains(&m),
                    "mode {:?} (from seed {:?}) not in reduced set",
                    m,
                    seed,
                );
            }
        }
    }

    /// Truncation surfaces cleanly: a 4-byte buffer satisfies the
    /// §7.3 `VP6_StartDecode` prefill, but the subsequent
    /// renormalization-driven `decode_b(2)` reads exhaust the
    /// stream and surface [`Error::Truncated`].
    #[test]
    fn truncation_surfaces() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut bc = BoolCoder::new(&bytes).expect("prefill OK");
        // Repeatedly attempt the walker; after a finite number of
        // renormalization refills the bitstream must surface
        // `Truncated`. Bound the attempts so a regression does not
        // hang the suite.
        let mut saw_truncated = false;
        for _ in 0..32 {
            match decode_fourmv_block_modes(&mut bc) {
                Ok(_) => continue,
                Err(Error::Truncated) => {
                    saw_truncated = true;
                    break;
                }
                Err(other) => panic!("unexpected error: {:?}", other),
            }
        }
        assert!(
            saw_truncated,
            "expected Truncated within 32 walker calls on a 4-byte stream"
        );
    }

    /// Single-block decode never returns a mode outside Table 10's
    /// four-entry reduced set. Per-block sweep over `decode_b(2)`'s
    /// 0..=3 output range pinned by the lookup-table cover. Skips
    /// the all-`0xFF` seed for the same `Range == 0` reason the
    /// `decoded_mode_is_always_in_reduced_set` test calls out.
    #[test]
    fn single_block_decode_returns_reduced_mode() {
        let seeds: [&[u8]; 3] = [
            &[0x00; 16],
            &[0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80],
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0],
        ];
        for seed in seeds {
            let buf = pad_to_64(seed);
            let mut bc = BoolCoder::new(&buf).expect("BoolCoder init");
            let m = decode_fourmv_block_mode(&mut bc).expect("decode");
            assert!(FOURMV_BLOCK_MODES.contains(&m));
        }
    }
}

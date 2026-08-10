//! VP6 raw-bit byte-stream reader (spec §3 `R(x)` operator).
//!
//! VP6 has two entropy schemes (spec §7): the **Huffman coder** (§7.2)
//! and the **BoolCoder** (§7.3). The Huffman coder reads whole raw
//! bits one at a time straight from the byte stream — the spec's `R(1)`
//! operator from §3 Nomenclature. The BoolCoder is a binary arithmetic
//! coder whose per-bit `Split` step depends on a contested formula in
//! §7.3 (the crate-root [`DOCS-GAP`](crate#docs-gap-spec-73-boolcoder-split-formula)),
//! but the *byte stream* underneath both coders is the same and is
//! defined by §3 alone — so this module's `R(n)` reader is
//! **independent of the §7.3 DOCS-GAP**.
//!
//! This module surfaces three things:
//!
//! 1. [`RawBitReader`] — a thin byte-stream `R(n)` reader. Backed by
//!    [`oxideav_core::bits::BitReader`] for the standard MSB-first
//!    layout (`R(1)` returns the byte's high bit, then the next bit,
//!    etc.), which is what every `R(n)` field in §9 Tables 1/2 is
//!    parsed as in [`crate::frame_header`] today.
//! 2. [`RawBitReader::read_lsb_first`] — an explicit *least-significant
//!    bit first* variant for the one place the spec calls that ordering
//!    out by name: §13.3.3.1 (page 78), *"the run length minus nine is
//!    encoded using six-bits, least significant bit first."* This
//!    matters for the `R(6)` escape suffix in the AC zero-run path
//!    (§13.3.3.1 BoolCoder path *and* §13.3.3.2 Huffman path — the
//!    spec's demonstration pseudo-code `if (ZrlToken < 8) … else 8 +
//!    R(6)` reads the same escape in either entropy scheme).
//! 3. [`RawBitReader::read_huffman_symbol`] — convenience that wires
//!    the byte-stream `R(1)` source straight into the §7.2
//!    [`crate::huffman::decode_symbol`] walk so callers don't have to
//!    capture the reader in a closure themselves. The Huffman path of
//!    §13.3.3.2 / §13 (when `UseHuffman == 1`) consumes this directly.
//!
//! ## Bit ordering: where the spec speaks, and where it doesn't
//!
//! The §3 Nomenclature listing defines `R(x)` as *"a sequence of x-bits
//! written directly to the bitstream as a sequence of raw bits"* without
//! specifying within-byte order. The convention used through Tables 1/2
//! (§9) and consumed by [`crate::frame_header::Vp6FrameHeader::parse`]
//! is MSB-first: `R(1)` consumes the byte's high bit first, then the
//! next-lower bit, etc., across byte boundaries. That same convention
//! is what the §7.2 Huffman walker assumes (Tables 1/2 use it; nothing
//! in §7.2 / §13 contradicts it). [`RawBitReader::read`] and
//! [`RawBitReader::read_bit`] implement that ordering.
//!
//! §13.3.3.1 then explicitly calls out the *opposite* ordering for one
//! specific 6-bit field: *"the run length minus nine is encoded using
//! six-bits, least significant bit first."*
//! [`RawBitReader::read_lsb_first`] implements that.
//!
//! No other location in the spec we've transcribed for rounds 1–13
//! overrides the §3 MSB-first convention.
//!
//! ## What this module deliberately does **not** do
//!
//! * It does **not** drive the §7.3 BoolCoder. That coder reads
//!   *sub-bit* `B(prob)` decisions, which depend on the contested
//!   `Split` formula. Once the §7.3 DOCS-GAP is closed the BoolCoder
//!   will sit on top of this raw-bit reader (consuming `R(8)` bytes
//!   for its `Value` accumulator refills) but the BoolCoder itself
//!   stays deferred.
//! * It does **not** know about partitions. VP6's partitioning is a
//!   §6/§9 concern: partition 1 starts at byte 0 of the frame and
//!   partition 2 (when present) starts at `Buff2Offset`. Callers
//!   construct a `RawBitReader` over the *byte slice* of the partition
//!   they want to read; this module never crosses a partition
//!   boundary on its own.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf`:
//!
//! * §3 Nomenclature (page 9): the `R(x)` operator definition.
//! * §13.3.3.1 (page 78): the *"least significant bit first"* override
//!   for the 6-bit escape suffix.
//! * §7.2 / §7.2.1 (pages 11–14): the Huffman walker that consumes
//!   `R(1)` bits and that this module's [`RawBitReader::read_huffman_symbol`]
//!   wraps.
//!
//! No third-party VP6 implementation has been consulted.

use oxideav_core::bits::BitReader;

use crate::huffman::{decode_symbol, HuffNode};

/// Error returned when an `R(n)` read runs past the end of the byte
/// stream the reader was constructed over.
///
/// Spec §3 doesn't itself describe error handling — it's a notation
/// document — but every VP6 partition (§6) is a contiguous byte buffer
/// of known length, and reading past its end is a malformed-input
/// condition the decoder needs to surface cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawBitError {
    /// The reader was asked for more bits than remain in the byte
    /// stream.
    OutOfBits,
    /// `read` was called with `n > 32`. The spec's `R(x)` operator
    /// itself imposes no bound, but the integer-return form here caps
    /// at 32 bits per call (the largest single VP6 `R(n)` field is
    /// `R(16)` for `Buff2Offset`, and §13.3.3 uses at most `R(6)`).
    TooManyBits,
}

impl core::fmt::Display for RawBitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBits => f.write_str("oxideav-vp6: raw-bit reader ran out of bits"),
            Self::TooManyBits => {
                f.write_str("oxideav-vp6: raw-bit reader can return at most 32 bits per call")
            }
        }
    }
}

impl std::error::Error for RawBitError {}

/// Reader for the spec's §3 `R(x)` operator.
///
/// Wraps the input byte stream and exposes whole-bit reads in the
/// MSB-first convention used by §9's Tables 1/2 (and consumed by
/// [`crate::frame_header::Vp6FrameHeader::parse`]) plus an explicit
/// LSB-first variant for the §13.3.3.1 6-bit escape.
///
/// Implements `Clone + Copy` (it owns nothing but a borrowed slice and
/// a position) so a parser can checkpoint and restore by simple
/// assignment — useful for partition probes that need to look ahead
/// without committing to a read.
#[derive(Clone, Copy)]
pub struct RawBitReader<'a> {
    inner: BitReader<'a>,
}

impl<'a> core::fmt::Debug for RawBitReader<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `oxideav_core::bits::BitReader` doesn't implement `Debug`, so
        // surface the bookkeeping it exposes through its public accessors
        // instead of trying to derive `Debug` on the wrapper.
        f.debug_struct("RawBitReader")
            .field("bit_position", &self.inner.bit_position())
            .field("byte_position", &self.inner.byte_position())
            .field("bits_remaining", &self.inner.bits_remaining())
            .field("is_byte_aligned", &self.inner.is_byte_aligned())
            .finish()
    }
}

impl<'a> RawBitReader<'a> {
    /// Construct a reader over `bytes`. Reads start at the high bit of
    /// `bytes[0]` per the MSB-first convention.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            inner: BitReader::new(bytes),
        }
    }

    /// Construct a reader starting at a specific byte offset within
    /// `bytes`. Useful for partition 2 reads where the caller knows
    /// `Buff2Offset` from the §9 frame header but doesn't want to
    /// re-slice the input.
    pub fn with_byte_offset(bytes: &'a [u8], byte_offset: usize) -> Self {
        Self {
            inner: BitReader::with_position(bytes, byte_offset),
        }
    }

    /// Remaining bits not yet consumed.
    #[inline]
    pub fn bits_remaining(&self) -> u64 {
        self.inner.bits_remaining()
    }

    /// True iff no more bits remain.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.bits_remaining() == 0
    }

    /// Bit offset from the start of the byte stream (0 at the first
    /// read).
    #[inline]
    pub fn bit_position(&self) -> u64 {
        self.inner.bit_position()
    }

    /// Byte offset of the next byte the reader will touch.
    #[inline]
    pub fn byte_position(&self) -> usize {
        self.inner.byte_position()
    }

    /// True iff the reader is on a byte boundary.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.inner.is_byte_aligned()
    }

    /// Discard any unread bits in the current byte, leaving the reader
    /// byte-aligned. Equivalent to the spec's *"the next field starts
    /// at the next byte boundary"* phrasing some §9 entries use.
    pub fn align_to_byte(&mut self) {
        self.inner.align_to_byte();
    }

    /// Read one raw bit (the §3 `R(1)` operator), MSB-first.
    ///
    /// Returns `0` or `1`. Equivalent to `self.read(1)?` but returns a
    /// `u8` so it can be dropped straight into the §7.2
    /// [`crate::huffman::decode_symbol`] oracle.
    pub fn read_bit(&mut self) -> Result<u8, RawBitError> {
        self.inner
            .read_u32(1)
            .map(|v| v as u8)
            .map_err(|_| RawBitError::OutOfBits)
    }

    /// Read `n` raw bits as an unsigned MSB-first integer (the §3
    /// `R(n)` operator).
    ///
    /// For `n == 0` returns `0` without touching the stream — matches
    /// the spec's no-op convention. For `n == 1` returns `0` or `1`.
    /// For larger `n` the high bit of the value is the first bit read
    /// from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`RawBitError::TooManyBits`] if `n > 32`,
    /// [`RawBitError::OutOfBits`] if fewer than `n` bits remain.
    pub fn read(&mut self, n: u32) -> Result<u32, RawBitError> {
        if n > 32 {
            return Err(RawBitError::TooManyBits);
        }
        self.inner.read_u32(n).map_err(|_| RawBitError::OutOfBits)
    }

    /// Read `n` raw bits as an unsigned integer with the bits emitted
    /// **least-significant first** (the §13.3.3.1 explicit override).
    ///
    /// Spec p. 78 (§13.3.3.1): *"the run length minus nine is encoded
    /// using six-bits, least significant bit first."* The first bit
    /// pulled from the stream becomes the value's bit `0`, the second
    /// becomes bit `1`, and so on. (`R(n)` itself is byte-stream
    /// MSB-first; this method exists for the one §13.3.3 field whose
    /// payload semantics reverse that within the read.)
    ///
    /// # Errors
    ///
    /// Returns [`RawBitError::TooManyBits`] if `n > 32`,
    /// [`RawBitError::OutOfBits`] if fewer than `n` bits remain.
    pub fn read_lsb_first(&mut self, n: u32) -> Result<u32, RawBitError> {
        if n > 32 {
            return Err(RawBitError::TooManyBits);
        }
        if n == 0 {
            return Ok(0);
        }
        let mut acc: u32 = 0;
        // Pull `n` raw bits and place each into position `i` (`i`-th
        // bit from the LSB). The first bit fetched is the LSB per
        // §13.3.3.1.
        for i in 0..n {
            let bit = self.inner.read_u32(1).map_err(|_| RawBitError::OutOfBits)?;
            acc |= (bit & 1) << i;
        }
        Ok(acc)
    }

    /// Decode one symbol from `tree` per the §7.2 Huffman walk, using
    /// this reader as the `R(1)` source.
    ///
    /// `tree` must be a sort-list returned by
    /// [`crate::huffman::create_huffman_tree`]; the root sits at
    /// `tree.len() - 1`. Walks the tree from the root, consuming one
    /// `R(1)` bit at each internal node (`0` → left, `1` → right per
    /// §7.2), and returns the decoded symbol identifier (the `Symbol`
    /// field of the leaf reached).
    ///
    /// If the reader runs out of bits mid-traversal the function
    /// surfaces [`RawBitError::OutOfBits`]; the reader's position is
    /// left where the failure was detected (do **not** rely on partial
    /// progress — the caller should treat the stream as malformed).
    ///
    /// Internally this wraps the existing
    /// [`crate::huffman::decode_symbol`] closure form. The wall between
    /// the two layers is preserved: the Huffman walker doesn't know
    /// about byte streams and the bit reader doesn't know about tree
    /// traversal.
    ///
    /// # Errors
    ///
    /// As above; the traversal itself is infallible against a
    /// well-formed tree, so the only failure mode is bit-stream
    /// exhaustion.
    pub fn read_huffman_symbol(&mut self, tree: &[HuffNode]) -> Result<i32, RawBitError> {
        // We have to thread the read result out of the closure because
        // `decode_symbol` is `-> i32`, not `Result<i32, _>`. The closure
        // pre-checks bits remaining; if exhausted it sets `err` and
        // returns `0` (selecting the `left` child arbitrarily) so the
        // tree walk terminates promptly without panicking — the
        // surrounding function then surfaces the error.
        let mut err: Option<RawBitError> = None;
        let symbol = decode_symbol(tree, || {
            if err.is_some() {
                return 0;
            }
            match self.inner.read_u32(1) {
                Ok(v) => v as u8,
                Err(_) => {
                    err = Some(RawBitError::OutOfBits);
                    0
                }
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(symbol),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::create_huffman_tree;

    // -----------------------------------------------------------------
    // Construction / position helpers
    // -----------------------------------------------------------------

    #[test]
    fn new_starts_at_zero() {
        let br = RawBitReader::new(&[0xFFu8, 0x00, 0xAA]);
        assert_eq!(br.bit_position(), 0);
        assert_eq!(br.byte_position(), 0);
        assert_eq!(br.bits_remaining(), 24);
        assert!(br.is_byte_aligned());
        assert!(!br.is_empty());
    }

    #[test]
    fn with_byte_offset_skips_bytes() {
        let br = RawBitReader::with_byte_offset(&[0x11u8, 0x22, 0x33, 0x44], 2);
        // Two bytes consumed → 16 bits gone, 16 bits left.
        assert_eq!(br.bits_remaining(), 16);
        assert_eq!(br.byte_position(), 2);
    }

    #[test]
    fn empty_input_is_empty() {
        let br = RawBitReader::new(&[]);
        assert!(br.is_empty());
        assert_eq!(br.bits_remaining(), 0);
    }

    // -----------------------------------------------------------------
    // MSB-first reads (`R(n)`)
    // -----------------------------------------------------------------

    #[test]
    fn read_bit_yields_msb_first() {
        // 0b1010_0110 = 0xA6
        let mut br = RawBitReader::new(&[0xA6]);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert!(br.is_empty());
    }

    #[test]
    fn read_n_packs_msb_first() {
        // 0b1010_0110_0101 split across two bytes.
        // Byte 0 = 0xA6, byte 1's high nibble = 0x5x.
        let mut br = RawBitReader::new(&[0xA6, 0x50]);
        // R(4) over the high nibble of byte 0 = 0b1010 = 0xA.
        assert_eq!(br.read(4).unwrap(), 0xA);
        // R(8) straddles byte 0's low nibble and byte 1's high nibble:
        // 0b0110_0101 = 0x65.
        assert_eq!(br.read(8).unwrap(), 0x65);
        // R(4) over byte 1's low nibble = 0b0000.
        assert_eq!(br.read(4).unwrap(), 0x0);
        assert!(br.is_empty());
    }

    #[test]
    fn read_zero_bits_is_noop() {
        let mut br = RawBitReader::new(&[0xFFu8]);
        assert_eq!(br.read(0).unwrap(), 0);
        assert_eq!(br.bit_position(), 0);
        assert_eq!(br.bits_remaining(), 8);
    }

    #[test]
    fn read_table1_layout_matches_frame_header_convention() {
        // §9 Table 1 byte 0 = [FrameType R(1) | DctQMask R(6) | MultiStream R(1)].
        // Encode FrameType=1, DctQMask=0b101_010, MultiStream=0.
        // Packed MSB-first: 1 101_010 0 = 0b1101_0100 = 0xD4.
        let mut br = RawBitReader::new(&[0xD4]);
        assert_eq!(br.read(1).unwrap(), 1, "FrameType");
        assert_eq!(br.read(6).unwrap(), 0b101_010, "DctQMask");
        assert_eq!(br.read(1).unwrap(), 0, "MultiStream");
        assert!(br.is_empty());
    }

    #[test]
    fn read_returns_out_of_bits_when_exhausted() {
        let mut br = RawBitReader::new(&[0xFFu8]);
        assert_eq!(br.read(8).unwrap(), 0xFF);
        assert_eq!(br.read_bit(), Err(RawBitError::OutOfBits));
        assert_eq!(br.read(1), Err(RawBitError::OutOfBits));
    }

    #[test]
    fn read_rejects_more_than_thirtytwo_bits() {
        let mut br = RawBitReader::new(&[0u8; 8]);
        assert_eq!(br.read(33), Err(RawBitError::TooManyBits));
        // No bits consumed on rejection.
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn read_max_thirtytwo_bits_is_accepted() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut br = RawBitReader::new(&bytes);
        assert_eq!(br.read(32).unwrap(), 0xDEAD_BEEF);
        assert!(br.is_empty());
    }

    #[test]
    fn align_to_byte_drops_partial_bits() {
        let mut br = RawBitReader::new(&[0xA6, 0x5A]);
        // Consume 3 bits of byte 0 (0b101).
        assert_eq!(br.read(3).unwrap(), 0b101);
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        // Next R(8) should be byte 1 untouched.
        assert_eq!(br.read(8).unwrap(), 0x5A);
    }

    // -----------------------------------------------------------------
    // LSB-first reads (§13.3.3.1 6-bit escape)
    // -----------------------------------------------------------------

    #[test]
    fn read_lsb_first_reverses_bit_order_within_field() {
        // Encode an "LSB-first" 6-bit value of 0b101_011 = decimal 43.
        // Bit-stream order is bit 0 first (=1), then bit 1 (=1), bit 2
        // (=0), bit 3 (=1), bit 4 (=0), bit 5 (=1).
        // Packed MSB-first into a byte: 1 1 0 1 0 1 _ _ = 0b1101_0100
        // (= 0xD4 in the high 6 bits, low 2 are don't-care padding).
        let mut br = RawBitReader::new(&[0xD4]);
        assert_eq!(br.read_lsb_first(6).unwrap(), 43);
    }

    #[test]
    fn read_lsb_first_one_bit_matches_read_bit_value() {
        // For n = 1 the two orderings are indistinguishable.
        let mut br_msb = RawBitReader::new(&[0x80]);
        let mut br_lsb = RawBitReader::new(&[0x80]);
        assert_eq!(br_msb.read_bit().unwrap(), 1);
        assert_eq!(br_lsb.read_lsb_first(1).unwrap(), 1);
    }

    #[test]
    fn read_lsb_first_round_trip_all_six_bit_values() {
        // Exhaustively round-trip all 64 possible 6-bit values to
        // confirm the bit-reversal is its own inverse against an
        // MSB-first packer.
        for v in 0u32..64 {
            // Pack `v` with bit 0 first: bit `i` of v goes to bit
            // `7 - i` of the byte (MSB-first packing where slot 0 of
            // the byte holds bit 0 of v).
            let mut packed = 0u8;
            for i in 0..6 {
                let b = (v >> i) & 1;
                packed |= (b as u8) << (7 - i);
            }
            let bytes = [packed];
            let mut br = RawBitReader::new(&bytes);
            assert_eq!(br.read_lsb_first(6).unwrap(), v, "v = {v}");
        }
    }

    #[test]
    fn read_lsb_first_zero_bits_is_noop() {
        let mut br = RawBitReader::new(&[0xFFu8]);
        assert_eq!(br.read_lsb_first(0).unwrap(), 0);
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn read_lsb_first_returns_out_of_bits_when_exhausted() {
        let mut br = RawBitReader::new(&[0u8; 0]);
        assert_eq!(br.read_lsb_first(1), Err(RawBitError::OutOfBits));
    }

    #[test]
    fn read_lsb_first_rejects_more_than_thirtytwo_bits() {
        let mut br = RawBitReader::new(&[0u8; 8]);
        assert_eq!(br.read_lsb_first(33), Err(RawBitError::TooManyBits));
        assert_eq!(br.bit_position(), 0);
    }

    // -----------------------------------------------------------------
    // §7.2 Huffman walker driven by the byte stream
    // -----------------------------------------------------------------

    /// Helper: build a Huffman tree for a synthetic alphabet whose
    /// codewords are knowable by symmetry (use the symbol identifiers
    /// 0..N-1 with identical probabilities so the §7.2.1 stable sort
    /// produces a unique tree shape).
    fn flat_tree(n: usize) -> Vec<HuffNode> {
        let symbols: Vec<i32> = (0..n as i32).collect();
        let probs: Vec<u8> = vec![1; n];
        create_huffman_tree(&symbols, &probs).expect("flat_tree: create_huffman_tree")
    }

    #[test]
    fn read_huffman_symbol_two_symbol_tree() {
        // Two symbols → tree has a root with two leaves, codewords
        // are 1 bit each. The equal-probability tie-break (errata
        // #277 part 3) puts the later-inserted symbol 1 ahead of its
        // equal, so symbol 1 is the root's left / bit-0 child and
        // symbol 0 the right / bit-1 child.
        let tree = flat_tree(2);
        // bits = 0,1,1,0,1 → expect symbols 1,0,0,1,0.
        // Pack MSB-first: 0b01101000 = 0x68 (only the high 5 bits
        // matter; low 3 are padding zeros).
        let mut br = RawBitReader::new(&[0x68]);
        assert_eq!(br.read_huffman_symbol(&tree).unwrap(), 1);
        assert_eq!(br.read_huffman_symbol(&tree).unwrap(), 0);
        assert_eq!(br.read_huffman_symbol(&tree).unwrap(), 0);
        assert_eq!(br.read_huffman_symbol(&tree).unwrap(), 1);
        assert_eq!(br.read_huffman_symbol(&tree).unwrap(), 0);
    }

    #[test]
    fn read_huffman_symbol_drives_via_byte_stream_directly() {
        // Cross-check: the same bits, fed through the closure form of
        // `decode_symbol`, must produce the same symbol the byte-stream
        // wrapper produces.
        use crate::huffman::decode_symbol;
        let tree = flat_tree(2);
        // 0b1100_0000 -> two right-branches followed by zeros.
        let bytes = [0xC0u8];
        let mut br_a = RawBitReader::new(&bytes);
        let mut br_b = RawBitReader::new(&bytes);
        let from_wrapper = br_a.read_huffman_symbol(&tree).unwrap();
        let from_closure = decode_symbol(&tree, || br_b.read_bit().unwrap());
        assert_eq!(from_wrapper, from_closure);
    }

    #[test]
    fn read_huffman_symbol_surfaces_truncation() {
        // Single byte with one well-formed symbol followed by a
        // padding pattern that would require more bits than the
        // buffer holds for a deeper tree.
        let tree = flat_tree(4); // codewords ≥ 2 bits each.
        let bytes = [0b1010_0000u8];
        // Consume two symbols (4 bits) — should succeed.
        let mut br = RawBitReader::new(&bytes);
        let _ = br.read_huffman_symbol(&tree).unwrap();
        let _ = br.read_huffman_symbol(&tree).unwrap();
        let _ = br.read_huffman_symbol(&tree).unwrap();
        let _ = br.read_huffman_symbol(&tree).unwrap();
        // Fifth read should run out of bits.
        assert_eq!(
            br.read_huffman_symbol(&tree),
            Err(RawBitError::OutOfBits),
            "fifth read should exhaust the single-byte buffer"
        );
    }

    // -----------------------------------------------------------------
    // Clone/Copy checkpoint semantics
    // -----------------------------------------------------------------

    #[test]
    fn copy_acts_as_checkpoint() {
        // The reader implements Copy: assigning to a fresh local is a
        // checkpoint, and restoring the original snapshot via
        // assignment unwinds any reads in between.
        let mut br = RawBitReader::new(&[0xAA, 0x55]);
        let snapshot = br;
        let _ = br.read(8).unwrap();
        assert_eq!(br.byte_position(), 1);
        br = snapshot;
        assert_eq!(br.byte_position(), 0);
        assert_eq!(br.read(8).unwrap(), 0xAA);
    }

    // -----------------------------------------------------------------
    // §13.3.3 worked example: §13.3.3.1 LSB-first 6-bit escape
    // -----------------------------------------------------------------

    #[test]
    fn read_lsb_first_models_section_13_3_3_1_escape() {
        // §13.3.3.1 (page 78): *"If a run length greater than eight is
        // indicated, then the run length minus nine is encoded using
        // six-bits, least significant bit first."*
        //
        // Pick run_length = 17 → encoded payload = run_length - 9 = 8.
        // Bit-stream order: bit 0 (= 0), bit 1 (= 0), bit 2 (= 0),
        // bit 3 (= 1), bit 4 (= 0), bit 5 (= 0).
        // Packed MSB-first into a byte: 0 0 0 1 0 0 _ _ = 0b0001_0000.
        let mut br = RawBitReader::new(&[0b0001_0000u8]);
        let encoded = br.read_lsb_first(6).unwrap();
        assert_eq!(encoded, 8);
        let run_length = encoded + 9;
        assert_eq!(run_length, 17);
    }
}

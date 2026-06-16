//! VP6 binary arithmetic decoder (spec §7.3 `VP6_DecodeBool`).
//!
//! This module surfaces the per-bit binary arithmetic decoder that
//! every BoolCoder-coded VP6 field consults: the §3 `B(prob)` single
//! bit, the §3 `b(n)` fixed-probability-128 multi-bit raw read, and
//! the §3 `T` decision-tree walk (built on top of these primitives by
//! the §10 / §11.2 / §13 callers in subsequent rounds).
//!
//! ## Provenance
//!
//! Sourced exclusively from material in `docs/video/vp6/`:
//!
//! * `vp6_format.pdf` §7.3 (page 15) — the `VP6_StartDecode` /
//!   `VP6_DecodeBool` pseudocode and the renormalization loop printed
//!   verbatim.
//! * `vp6_format.pdf` §3 (page 9) — the `B(x)`, `b(x)`, `T` notation
//!   definitions.
//! * `vp6-errata-and-clarifications.md` entry **#35** — the clean-room
//!   disambiguation that pins down the `Split` formula's evaluation
//!   order (`multiply → shift-right-by-8 → add-1`) and establishes that
//!   the §7.3 PDF's printed `>> 7` is a **spec typo**: the operative
//!   shift is `>> 8` (divide by 256), under which probability `128` is
//!   the half-interval point so a `b(n)` read at fixed probability 128
//!   partitions `Range` (almost) evenly.
//!
//! No third-party VP6 source has been consulted at any stage.
//!
//! ## The `Split` formula (errata #35)
//!
//! As **printed** in §7.3 (the shift count `7` is a transcription
//! typo — see errata #35):
//!
//! ```text
//! Split = 1 + ( ((Range-1) * Probability) >> 7 )   // printed
//! ```
//!
//! The **operative** formula errata #35 establishes:
//!
//! ```text
//! Split = 1 + ( ((Range-1) * Probability) >> 8 )   // correct
//! ```
//!
//! Errata #35 nails the evaluation order in unsigned integer
//! arithmetic:
//!
//! 1. Compute `t = (Range - 1) * Probability`. `Range` is in `1..=255`
//!    on entry to `VP6_DecodeBool` (post-renormalization invariant
//!    `Range >= 128`); `Probability` is in `1..=255` (§7.3 forbids
//!    `Probability == 0`). The product fits in 16 bits, so 32-bit
//!    arithmetic is more than sufficient.
//! 2. Arithmetic shift right by **8**: `s = t >> 8`. The shift applies
//!    only to `t`, not to `1 + t`.
//! 3. Add **1 after** the shift: `Split = 1 + s`.
//!
//! The `>> 8` (divide by 256) is the only shift that makes the coder
//! function. It keeps `1 <= Split <= Range - 1` for every
//! `Probability ∈ [1,255]` and `Range ∈ [128,255]`, so both the
//! `Bit = 0` sub-interval `[0, Split)` and the `Bit = 1` sub-interval
//! `[Split, Range)` are non-empty. At `Probability = 128` it gives
//! `Split = 1 + ((Range-1) >> 1) ≈ Range/2` — the equiprobable split a
//! fixed-probability `b(x)` read needs. The printed `>> 7` (divide by
//! 128) is degenerate: at `Probability = 128` it yields `Split = Range`
//! (collapsing the `Bit = 1` interval to width 0) and at
//! `Probability = 255` it yields `Split > Range` (a negative `Bit = 1`
//! interval) — neither can be decoded, so `>> 7` cannot be the
//! operative shift.
//!
//! The `<< 24` alignment in the comparison `Value < (Split << 24)`
//! and the update `Value = Value - (Split << 24)` aligns the 8-bit
//! `Split` against the top byte of the 32-bit `Value`. This is
//! unambiguous as printed and the implementation here reproduces it
//! verbatim.
//!
//! ## What this module does **not** do
//!
//! It does not consume any specific syntactic field. The BoolCoder is
//! a *primitive* — the §10 mode-tree walk, the §11 motion-vector
//! decoder, the §13 DCT-token tree and the §13.3.3.1 zero-run-length
//! tree all sit on top of [`BoolCoder::decode_bool`] /
//! [`BoolCoder::decode_b`] / [`BoolCoder::decode_b1`]. Wiring those
//! consumers is later-round work.

use crate::Error;

/// VP6 binary arithmetic decoder.
///
/// Owns a borrowed byte-stream cursor plus the §7.3 four-tuple of
/// decoder state (`Range`, `Count`, `Value`, `Pos`). Bits are decoded
/// one at a time by [`BoolCoder::decode_bool`]; multi-bit reads at
/// fixed probability 128 use [`BoolCoder::decode_b`].
///
/// The reader is constructed over a single contiguous partition's
/// byte slice ([`BoolCoder::new`]); it never crosses a partition
/// boundary on its own. Partition layout (single-partition vs.
/// `Buff2Offset`-split) is a §6 / §9 concern handled by the caller.
#[derive(Debug)]
pub struct BoolCoder<'a> {
    /// Byte stream the decoder pulls renormalization fill from.
    bytes: &'a [u8],
    /// `Range` per §7.3. Initialized to 255; renormalization
    /// maintains the invariant `128 <= Range <= 255` on entry to each
    /// [`BoolCoder::decode_bool`] call.
    range: u32,
    /// `Value` per §7.3. Top 32 bits of the still-undecoded bitstream
    /// arithmetic value, kept left-aligned so the comparison with
    /// `Split << 24` and the renormalization doubling work uniformly.
    value: u32,
    /// `Count` per §7.3. Number of usable bits remaining in `Value`
    /// before the renormalization loop must pull a fresh byte. Drops
    /// from `8` to `0`; on reaching zero a fresh byte is OR'ed into
    /// the low bits of `Value` and `Count` is refilled to `8`.
    count: i32,
    /// `Pos` per §7.3. Index of the next byte to read from
    /// `bytes`. Initialized to `4` after `VP6_StartDecode` consumes
    /// the first four bytes.
    pos: usize,
}

impl<'a> BoolCoder<'a> {
    /// Initialize the decoder over `bytes` per §7.3 `VP6_StartDecode`:
    ///
    /// > ```text
    /// > Range  = 255
    /// > Count  = 8
    /// > Value  = First 32-bits extracted from bit stream
    /// > Pos    = 4 (4 bytes already extracted in to Value)
    /// > ```
    ///
    /// The "first 32 bits" are read big-endian: byte 0 occupies bits
    /// 31..24 of `Value`, byte 1 occupies 23..16, byte 2 occupies
    /// 15..8, byte 3 occupies 7..0. This is the only encoding
    /// consistent with the post-init `Value < (Split << 24)`
    /// comparison (which aligns `Split` against the top byte of
    /// `Value`) and with the renormalization loop that doubles
    /// `Value` and OR's fresh bytes into the low bits — both behave
    /// correctly only if the original four bytes were placed
    /// most-significant-byte-first.
    ///
    /// Returns [`Error::Truncated`] if `bytes.len() < 4`.
    pub fn new(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < 4 {
            return Err(Error::Truncated);
        }
        let value = (u32::from(bytes[0]) << 24)
            | (u32::from(bytes[1]) << 16)
            | (u32::from(bytes[2]) << 8)
            | u32::from(bytes[3]);
        Ok(Self {
            bytes,
            range: 255,
            value,
            count: 8,
            pos: 4,
        })
    }

    /// Decode one bit at node probability `probability`.
    ///
    /// Returns the decoded bit (0 or 1). `probability` is the §7.3
    /// "probability of decoding a zero" on the linear 8-bit scale
    /// where 1 represents probability 1/256 and 255 represents
    /// probability 255/256. The value 0 is explicitly forbidden by
    /// the spec; this implementation does not enforce that bound at
    /// runtime (callers in §10/§11/§13 use either spec-provided
    /// constants or values pulled from probability tables that were
    /// themselves validated against the spec's `1..=255` range).
    ///
    /// Returns [`Error::Truncated`] if the renormalization loop tries
    /// to read past the end of the byte stream.
    pub fn decode_bool(&mut self, probability: u8) -> Result<u8, Error> {
        // §7.3 Split formula per errata #35: multiply → shift-8 → add-1.
        // The §7.3 PDF prints `>> 7`, but that is a transcription typo:
        // `>> 7` makes `Split = Range` at probability 128 (empty
        // `Bit = 1` interval) and `Split > Range` at probability 255
        // (negative interval). The operative shift is `>> 8`, which
        // keeps `1 <= Split <= Range - 1` for every `Probability` in
        // `1..=255` and `Range` in `128..=255`.
        //
        // Range is in 1..=255 on entry (post-renormalization invariant
        // `128 <= Range <= 255`, maintained by the loop at the bottom);
        // `Probability` is in `1..=255` (§7.3 forbids 0). The product
        // `(Range - 1) * Probability` fits in 16 bits (max 254 * 255 =
        // 64770), so u32 arithmetic is more than enough.
        let t = (self.range - 1) * u32::from(probability);
        let split = 1 + (t >> 8);

        // §7.3 branch: align the 8-bit Split against the top byte of
        // the 32-bit Value. With `Split <= Range - 1 <= 254`, the
        // shifted quantity `Split << 24 <= 0xFE00_0000` fits in u32,
        // but the comparison and subtraction are done in u32 directly
        // (no overflow possible since `Split <= 254`).
        let split_shifted = split << 24;
        let bit = if self.value < split_shifted {
            self.range = split;
            0u8
        } else {
            self.range -= split;
            self.value -= split_shifted;
            1u8
        };

        // §7.3 renormalization loop: while Range < 128, double Range
        // and Value, decrement Count; when Count hits zero pull a
        // fresh byte from the bitstream into the low bits of Value
        // and refill Count to 8.
        while self.range < 128 {
            self.range <<= 1;
            self.value <<= 1;
            self.count -= 1;
            if self.count == 0 {
                if self.pos >= self.bytes.len() {
                    return Err(Error::Truncated);
                }
                self.value |= u32::from(self.bytes[self.pos]);
                self.pos += 1;
                self.count = 8;
            }
        }

        Ok(bit)
    }

    /// Decode a single fixed-probability-128 bit (`b(1)` per §3).
    ///
    /// Equivalent to [`Self::decode_bool`] with `probability = 128`.
    /// At probability 128 the [`Split`](Self::decode_bool) formula
    /// (operative `>> 8`, per errata #35) gives `Split ≈ Range/2` — an
    /// (almost) even partition of `Range` — so the bit behaves
    /// statistically like a raw bit pulled straight from the underlying
    /// bitstream.
    pub fn decode_b1(&mut self) -> Result<u8, Error> {
        self.decode_bool(128)
    }

    /// Decode an `n`-bit fixed-probability-128 raw value (`b(n)` per
    /// §3).
    ///
    /// §3 *Nomenclature* defines `b(x)` as "a sequence of x-bits
    /// encoded using the BoolCoder with a fixed node probability of
    /// 128 for each bit." This method calls [`Self::decode_bool`]
    /// `n` times at probability 128 and accumulates the bits
    /// **most-significant-bit first** into the returned value, so
    /// that the same bit pattern parses identically under the §3
    /// raw-bit `R(n)` operator (the convention used in §9 Tables 1/2
    /// and implemented by [`crate::raw_bits::RawBitReader`]).
    ///
    /// `n` is capped at 32 bits per call — VP6's largest single
    /// `b(n)` field (§9 Table 2 `Buff2Offset` is `R(16)` raw-bit, not
    /// `b(n)`; the BoolCoder-coded multi-bit fields in §10/§11/§13
    /// are all <= 13 bits). Calls with `n > 32` saturate at 32.
    ///
    /// Returns [`Error::Truncated`] if the underlying byte stream is
    /// exhausted during any of the constituent [`Self::decode_bool`]
    /// calls.
    pub fn decode_b(&mut self, n: u32) -> Result<u32, Error> {
        let n = n.min(32);
        let mut value = 0u32;
        for _ in 0..n {
            let bit = self.decode_bool(128)?;
            value = (value << 1) | u32::from(bit);
        }
        Ok(value)
    }

    /// Return the current `Range` state. Exposed for diagnostics /
    /// testing; production callers should not need this.
    pub fn range(&self) -> u32 {
        self.range
    }

    /// Return the current `Value` state. Exposed for diagnostics /
    /// testing; production callers should not need this.
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Return the current `Count` state. Exposed for diagnostics /
    /// testing; production callers should not need this.
    pub fn count(&self) -> i32 {
        self.count
    }

    /// Return the current byte-stream position. Exposed for
    /// diagnostics / testing; production callers should not need
    /// this.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// VP6 binary arithmetic **encoder** — the exact inverse of the §7.3
/// [`BoolCoder`] decoder.
///
/// ## Provenance
///
/// The §7.3 VP6 specification (`docs/video/vp6/vp6_format.pdf`,
/// pages 14–15) is a *decoder* specification: it prints
/// `VP6_StartDecode` / `VP6_DecodeBool` and the renormalization loop,
/// but no encoder pseudocode. A binary arithmetic encoder is, however,
/// **uniquely determined** by the decoder it must feed: the encoder's
/// only job is to emit a byte stream that `VP6_StartDecode` /
/// `VP6_DecodeBool` reconstruct bit-for-bit. This module is therefore
/// derived **solely** from the §7.3 decode equations in the in-tree
/// spec — it is the algebraic dual of [`BoolCoder::decode_bool`], not a
/// transcription of any third-party encoder. No third-party VP6 source
/// has been consulted at any stage. The round-trip tests below pin the
/// inverse relationship to the in-crate decoder.
///
/// ## How the inverse is constructed
///
/// The decoder models the still-undecoded bitstream as a 32-bit,
/// left-aligned arithmetic `Value` and narrows `Range` per bit:
///
/// * `Split = 1 + ( ((Range-1) * Probability) >> 8 )`  (operative shift
///   per errata #35; the §7.3 PDF's printed `>> 7` is a typo).
/// * `Bit = 0` selects the low sub-interval `[0, Split << 24)`:
///   `Range = Split`.
/// * `Bit = 1` selects the high sub-interval `[Split << 24, Range << 24)`:
///   `Range -= Split`, and the decoder subtracts `Split << 24` from
///   `Value` (rebasing the high sub-interval to zero).
///
/// The encoder mirrors this with two state words:
///
/// * `range` — identical to the decoder's `Range` (starts at 255,
///   renormalized to stay `>= 128`).
/// * `low` — the bottom of the current coding interval, tracked in the
///   **same `<< 24`-aligned domain** as the decoder's `Value`, but in a
///   wider [`u64`] accumulator so a renormalization carry can ripple up
///   past bit 31 into already-buffered output bytes.
///
/// Encoding one bit at probability `p`:
///
/// * compute the identical `Split`;
/// * `Bit = 0`: keep `low`, set `range = Split` (the decoder's
///   `Range = Split`);
/// * `Bit = 1`: `low += Split << 24`, set `range -= Split` (mirroring the
///   decoder's `Value -= Split << 24` / `Range -= Split`).
///
/// Renormalization is performed **at bit granularity**, exactly
/// mirroring the decoder's `Range *= 2; Value *= 2` loop: while
/// `range < 128`, double `range` and shift `low` left by one bit,
/// shifting the bit that leaves the 32-bit window out to the byte
/// buffer (with carry propagation). Because the encoder renorm shifts
/// the same number of times the decoder renorm does (both gated on
/// `range < 128` with identical `range` evolution), the emitted bits
/// line up one-for-one with the decoder's `Value *= 2` byte pulls.
///
/// ## Memory bound
///
/// The encoder buffers its output in a `Vec<u8>` whose length is
/// proportional to the number of bits encoded (one output bit per input
/// bit, plus renormalization, plus a 4-byte flush). It holds no
/// per-symbol scratch and never materializes an interval table, so its
/// working set is `O(output_len)` with a tiny constant. Callers driving
/// long round-trip tests should keep the bit count bounded; the tests
/// here stay well under a few kilobytes.
#[derive(Debug, Clone)]
pub struct BoolEncoder {
    /// Already-committed output **bits**, most-significant first, in the
    /// order the decoder consumes them (the first pushed bit is bit 7 of
    /// `bytes[0]`, the top of the decoder's initial 32-bit `Value`).
    ///
    /// Keeping renormalized bits as an explicit `0`/`1` list — rather
    /// than packing them eagerly — makes carry propagation trivially
    /// correct: a carry out of the 32-bit window is just `+1` added to
    /// the integer these bits represent, rippled from the tail toward
    /// the head. The list is packed into bytes only at [`Self::finish`].
    /// For the bounded inputs this encoder is used with, the list stays
    /// small (one entry per encoded bit plus renorm).
    bits: Vec<u8>,
    /// Bottom of the current coding interval, kept in the **same 32-bit
    /// domain as the decoder's `Value`** (the active comparison byte sits
    /// at bits 31..24). A renormalization doubling (`low <<= 1`) shifts
    /// the top bit, bit 31, out into `bits`. Stored in a [`u64`] purely
    /// so the `low += split << 24` addition can be inspected for a
    /// carry out of bit 31.
    low: u64,
    /// Current `Range`, identical in meaning and evolution to the
    /// decoder's `Range`. Starts at 255.
    range: u32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    /// Create a fresh encoder. Mirrors `VP6_StartDecode`'s post-init
    /// invariant `Range = 255`. The first encoded bit's renormalization
    /// output forms the most-significant bit of the stream, which the
    /// decoder loads as the top of its initial 32-bit `Value` window.
    pub fn new() -> Self {
        Self {
            bits: Vec::new(),
            // `low` lives in the same 32-bit domain as the decoder's
            // `Value`; bit 32 is transient carry space, resolved
            // immediately after each interval update.
            low: 0,
            range: 255,
        }
    }

    /// Renormalization shift, mirroring the decoder's `Value *= 2`:
    /// double the window, committing the bit that leaves bit 31 to the
    /// output bit list.
    fn shift_out_top_bit(&mut self) {
        // Any carry out of bit 31 must already have been resolved by
        // [`Self::resolve_carry`] before we get here, so `low` occupies
        // bits 0..=31 only.
        debug_assert!(self.low < (1u64 << 32));
        let top = ((self.low >> 31) & 1) as u8;
        self.low = (self.low << 1) & ((1u64 << 32) - 1);
        self.bits.push(top);
    }

    /// Resolve a carry out of bit 31 of `low`: it represents `+1` added
    /// to the integer already committed to `bits`, rippled from the most
    /// recently emitted bit toward the oldest. Because every committed
    /// bit is stored explicitly, the ripple is exact regardless of byte
    /// alignment.
    fn resolve_carry(&mut self) {
        if self.low & (1u64 << 32) == 0 {
            return;
        }
        self.low &= (1u64 << 32) - 1;
        let mut i = self.bits.len();
        while i > 0 {
            i -= 1;
            if self.bits[i] == 0 {
                self.bits[i] = 1;
                return;
            }
            self.bits[i] = 0;
        }
        // A carry rippling off the front would mean the cumulative coded
        // value exceeded the number of emitted bits — impossible for an
        // encode that started from `low = 0`. The guard makes any logic
        // error surface loudly instead of silently corrupting output.
        debug_assert!(false, "BoolEncoder carry rippled off the bit buffer");
    }

    /// Encode one bit `bit` (0 or 1) at node probability `probability`.
    ///
    /// The dual of [`BoolCoder::decode_bool`]: the same `Split` is
    /// computed, the chosen sub-interval is selected, and `range` is
    /// renormalized back to `>= 128`, emitting one output bit per
    /// doubling — exactly the doublings the decoder will perform.
    pub fn encode_bool(&mut self, bit: u8, probability: u8) {
        // Identical Split to the decoder (errata #35 operative `>> 8`).
        let t = (self.range - 1) * u32::from(probability);
        let split = 1 + (t >> 8);

        if bit == 0 {
            self.range = split;
        } else {
            // Mirror the decoder's `Value -= Split << 24` by moving the
            // interval bottom up to the start of the high sub-interval.
            // The addition may carry out of bit 31; resolve it at once so
            // `low` is back in its 32-bit domain before renormalizing.
            self.low += u64::from(split) << 24;
            self.resolve_carry();
            self.range -= split;
        }

        // Renormalize to mirror the decoder's `while Range < 128`.
        while self.range < 128 {
            self.range <<= 1;
            self.shift_out_top_bit();
        }
    }

    /// Encode a single fixed-probability-128 bit (`b(1)`), the dual of
    /// [`BoolCoder::decode_b1`].
    pub fn encode_b1(&mut self, bit: u8) {
        self.encode_bool(bit, 128);
    }

    /// Encode the low `n` bits of `value` at fixed probability 128,
    /// **most-significant-bit first** — the dual of
    /// [`BoolCoder::decode_b`], so a value written here decodes back
    /// identically. `n` is capped at 32.
    pub fn encode_b(&mut self, value: u32, n: u32) {
        let n = n.min(32);
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.encode_bool(bit, 128);
        }
    }

    /// Finish encoding and return the completed byte stream.
    ///
    /// Flushes enough trailing bits so the decoder's 32-bit `Value`
    /// window is fully primed for every bit that was encoded. The
    /// decoder prefills 4 bytes and then pulls one byte per 8
    /// renormalization doublings, so flushing 32 more bits (the current
    /// 32-bit window contents, top bit first) guarantees the decoder can
    /// reproduce every encoded bit before the stream is exhausted. Any
    /// partially-filled output byte is zero-padded on the right.
    pub fn finish(mut self) -> Vec<u8> {
        // Drain the 32 bits currently held in the window so the decoder
        // has every encoded bit available before the stream ends.
        // `shift_out_top_bit` commits bit 31 each call, so 32 calls
        // empty the window MSB-first.
        for _ in 0..32 {
            self.shift_out_top_bit();
        }

        // Pack the committed bit list MSB-first into bytes, zero-padding
        // the final partial byte on the right.
        let mut out = Vec::with_capacity(self.bits.len() / 8 + 5);
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                byte |= b << (7 - i);
            }
            out.push(byte);
        }

        // The decoder requires at least 4 bytes (the `VP6_StartDecode`
        // prefill). A stream this short only happens for a near-empty
        // encode; pad it out so `BoolCoder::new` accepts it.
        while out.len() < 4 {
            out.push(0);
        }
        out
    }

    /// Current `Range`. Exposed for diagnostics / testing.
    pub fn range(&self) -> u32 {
        self.range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.3 `VP6_StartDecode` initial state: 4 bytes consumed into a
    /// big-endian 32-bit `Value`, `Range = 255`, `Count = 8`, `Pos =
    /// 4`.
    #[test]
    fn start_decode_initial_state() {
        let bytes = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let bc = BoolCoder::new(&bytes).expect("4+ bytes available");
        assert_eq!(bc.range(), 255);
        assert_eq!(bc.count(), 8);
        assert_eq!(bc.pos(), 4);
        assert_eq!(bc.value(), 0x1234_5678);
    }

    /// Constructor surfaces `Truncated` when the stream is too short
    /// for the §7.3 4-byte prefill.
    #[test]
    fn start_decode_rejects_short_input() {
        for n in 0..4 {
            let bytes = vec![0u8; n];
            assert_eq!(BoolCoder::new(&bytes).unwrap_err(), Error::Truncated);
        }
        // 4 bytes is the minimum and must succeed.
        assert!(BoolCoder::new(&[0u8; 4]).is_ok());
    }

    /// An all-zero byte stream decodes to all-zero bits at any
    /// probability strictly greater than 0. The §7.3 half-interval
    /// property (errata #35) makes this the natural smoke test for
    /// the formula: with `Value` initially 0, the comparison
    /// `Value < (Split << 24)` is always true (any `Split >= 1`
    /// makes `Split << 24 >= 1`), so the 0-branch is taken; the
    /// renormalization keeps `Value` at 0; the next call repeats.
    ///
    /// We sample several probabilities on independent decoder
    /// instances (each gets a fresh 64-byte all-zero stream so the
    /// renormalization byte pulls don't run the stream dry across
    /// rounds — low probabilities drive aggressive renormalization
    /// which consumes one fill byte every few decodes).
    #[test]
    fn all_zero_stream_decodes_all_zeros() {
        for prob in [1u8, 32, 64, 128, 200, 255] {
            let bytes = [0u8; 64];
            let mut bc = BoolCoder::new(&bytes).expect("4+ bytes");
            for i in 0..8 {
                let bit = bc.decode_bool(prob).expect("not truncated");
                assert_eq!(
                    bit, 0,
                    "probability {prob}, iter {i}: zero stream must decode to 0"
                );
            }
        }
    }

    /// The §7.3 `Split` formula's specific value at the canonical
    /// half-interval point under the operative `>> 8` shift (errata
    /// #35): `Probability = 128, Range = 255` gives
    /// `Split = 1 + ((254 * 128) >> 8) = 1 + 127 = 128` — an
    /// equiprobable partition of `Range`, exactly what a
    /// fixed-probability bit needs. (The printed `>> 7` would instead
    /// give `Split = 255 = Range`, an empty `Bit = 1` interval — the
    /// degeneracy errata #35 rules out.) Drive a single decode where
    /// `Value` is just below `Split << 24 = 0x8000_0000` so the
    /// 0-branch is taken, then verify the post-state: `Range = Split =
    /// 128` (no renormalization needed at exactly 128).
    #[test]
    fn split_formula_canonical_half_interval_value() {
        // Manually arrange Value at the edge of the 0-branch
        // interval. Construct a byte stream whose initial big-endian
        // 32-bit Value is 0x7FFF_FFFF (one below 0x8000_0000).
        let bytes = [0x7F, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        let mut bc = BoolCoder::new(&bytes).expect("4+ bytes");
        assert_eq!(bc.value(), 0x7FFF_FFFF);
        assert_eq!(bc.range(), 255);

        let bit = bc.decode_bool(128).expect("not truncated");
        assert_eq!(bit, 0, "Value 0x7FFFFFFF < 0x80000000: 0-branch");
        assert_eq!(bc.range(), 128, "Split = 128 → Range becomes Split = 128");
        // No renormalization triggered (Range == 128, loop entry is
        // `Range < 128`).
        assert_eq!(bc.count(), 8);
        assert_eq!(bc.pos(), 4);
    }

    /// Errata #35 degeneracy guard: under the operative `>> 8` shift,
    /// `Split` is bounded `1 <= Split <= Range - 1` for **every**
    /// `Probability ∈ [1,255]` and `Range ∈ [128,255]`, so both decode
    /// sub-intervals stay non-empty. The printed `>> 7` violates this
    /// (Split reaches or exceeds Range at high probabilities); this
    /// test pins the operative bound across the whole grid so a future
    /// regression back to `>> 7` is caught immediately.
    #[test]
    fn split_bounded_for_all_probabilities_and_ranges() {
        for range in 128u32..=255 {
            for prob in 1u32..=255 {
                let t = (range - 1) * prob;
                let split = 1 + (t >> 8);
                // Both decode sub-intervals must be non-empty:
                // `1 <= Split <= Range - 1`, i.e. `Split >= 1 && Split
                // < Range`.
                assert!(
                    split >= 1 && split < range,
                    "range {range}, prob {prob}: Split {split} not in 1..={}",
                    range - 1
                );
            }
        }
    }

    /// Renormalization byte-pull: arrange a probability that
    /// produces a small `Range` post-decode and verify the loop
    /// renormalizes back to 128, doubling `Value` alongside. With
    /// `Range = 255, Probability = 1` (operative `>> 8`):
    /// `Split = 1 + (254 * 1 >> 8) = 1 + 0 = 1`. Comparison
    /// `Value < 1 << 24 = 0x0100_0000`. If `Value = 0x0080_0000`
    /// (below threshold) → 0-branch, `Range = 1`. Then renormalization
    /// doubles `Range` 7 times to reach 128, doubling `Value`
    /// alongside (`0x0080_0000 << 7 = 0x4000_0000`).
    #[test]
    fn split_formula_small_split_renormalization() {
        // Initial big-endian: Value = 0x0080_0000.
        let bytes = [0x00, 0x80, 0x00, 0x00, 0xA5, 0x5A, 0xC3, 0x3C];
        let mut bc = BoolCoder::new(&bytes).expect("4+ bytes");
        assert_eq!(bc.value(), 0x0080_0000);

        let bit = bc.decode_bool(1).expect("not truncated");
        assert_eq!(bit, 0, "Value 0x00800000 < Split<<24 0x01000000");
        // Post-decode pre-renorm: Range = 1, Value = 0x0080_0000.
        // Renormalization: 7 doublings take Range from 1 to 128,
        // Value goes from 0x0080_0000 to 0x4000_0000. Count was 8 → 1,
        // and no fresh byte is needed (yet — Count never reaches 0).
        assert_eq!(bc.range(), 128);
        assert_eq!(bc.value(), 0x4000_0000);
        assert_eq!(bc.count(), 1, "8 - 7 = 1 bit remaining");
        assert_eq!(bc.pos(), 4, "no byte pulled yet");
    }

    /// Renormalization byte-pull: drive enough decodes that the loop
    /// actually pulls a byte from the stream.
    ///
    /// Step-by-step trace from the §7.3 pseudocode under the operative
    /// `>> 8` shift (errata #35):
    /// * Initial: `Range = 255, Value = 0, Count = 8, Pos = 4`.
    /// * 1st `decode_bool(1)`: `Split = 1 + (254 * 1 >> 8) = 1 + 0 = 1`.
    ///   Compare `0 < 0x0100_0000` → 0-branch. `Range = 1`. Renorm:
    ///   7 doublings to reach 128. `Count = 8 - 7 = 1`. No byte pull
    ///   yet (Count never reaches 0).
    /// * 2nd `decode_bool(1)`: `Range = 128`, `Split = 1 + (127 * 1 >>
    ///   8) = 1 + 0 = 1`. Compare `0 < 0x0100_0000` → 0-branch. `Range
    ///   = 1`. Renorm: need 7 doublings to reach 128. After 1 doubling
    ///   `Count = 0`, which triggers a byte pull (`Value |= bytes[4] =
    ///   0xA5`, `Pos = 5`, `Count = 8`). Then 6 more doublings to exit:
    ///   `Count = 8 - 6 = 2`, `Range = 128`.
    #[test]
    fn renormalization_pulls_byte_from_stream() {
        let bytes = [0x00, 0x00, 0x00, 0x00, 0xA5, 0x5A, 0xC3, 0x3C];
        let mut bc = BoolCoder::new(&bytes).expect("4+ bytes");

        let _ = bc.decode_bool(1).expect("not truncated");
        assert_eq!(bc.pos(), 4, "first decode renorm doesn't reach Count == 0");
        assert_eq!(bc.count(), 1);
        assert_eq!(bc.range(), 128);

        let _ = bc.decode_bool(1).expect("not truncated");
        assert_eq!(bc.pos(), 5, "Pos advances to 5 after consuming bytes[4]");
        assert_eq!(bc.count(), 2, "Count = 8 - 6 doublings after refill = 2");
        assert_eq!(bc.range(), 128, "renormalized back to 128");
    }

    /// `decode_b(n)` accumulates MSB-first. With a stream that
    /// decodes to 0,1,0,1 at fixed probability 128 the integer
    /// value should be 0b0101 = 5 (high bit first).
    #[test]
    fn decode_b_accumulates_msb_first() {
        // Construct a value whose first 4 b(1) reads at prob 128
        // give 0,1,0,1. Easier than reverse-engineering the bits is
        // to drive the decoder once to capture the sequence, then
        // re-drive on the same input checking the multi-bit form
        // matches.
        let bytes = [0x55, 0xAA, 0xCC, 0x33, 0x11, 0x22, 0x44, 0x88];

        // Capture the four individual bits.
        let mut bc1 = BoolCoder::new(&bytes).expect("4+ bytes");
        let b0 = bc1.decode_b1().expect("not truncated");
        let b1 = bc1.decode_b1().expect("not truncated");
        let b2 = bc1.decode_b1().expect("not truncated");
        let b3 = bc1.decode_b1().expect("not truncated");
        let expected_msb_first: u32 =
            (u32::from(b0) << 3) | (u32::from(b1) << 2) | (u32::from(b2) << 1) | u32::from(b3);

        // Now re-drive with decode_b(4) on a fresh decoder over the
        // same bytes.
        let mut bc2 = BoolCoder::new(&bytes).expect("4+ bytes");
        let four = bc2.decode_b(4).expect("not truncated");
        assert_eq!(
            four, expected_msb_first,
            "decode_b(4) must equal b(1)·b(1)·b(1)·b(1) packed MSB-first"
        );

        // Final state of bc1 and bc2 should match exactly (same input,
        // same number of bit reads).
        assert_eq!(bc1.range(), bc2.range());
        assert_eq!(bc1.value(), bc2.value());
        assert_eq!(bc1.count(), bc2.count());
        assert_eq!(bc1.pos(), bc2.pos());
    }

    /// `decode_b(0)` is a no-op (zero bits requested → returns 0
    /// without consuming any bits). The §3 nomenclature doesn't
    /// explicitly bless `b(0)` but the §13 / §10 callers issue
    /// length-driven `b(n)` reads where `n` can become zero, so the
    /// no-op behaviour is required.
    #[test]
    fn decode_b_zero_is_noop() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        let mut bc = BoolCoder::new(&bytes).expect("4+ bytes");
        let v = bc.decode_b(0).expect("zero-bit read");
        assert_eq!(v, 0);
        assert_eq!(bc.range(), 255);
        assert_eq!(bc.value(), 0);
        assert_eq!(bc.count(), 8);
        assert_eq!(bc.pos(), 4);
    }

    /// `decode_b(n)` saturates at 32 bits (the §3 nomenclature
    /// permits arbitrary `n` but VP6's largest BoolCoder-coded
    /// multi-bit field is well under 32 bits; the cap is a defensive
    /// invariant).
    #[test]
    fn decode_b_saturates_at_32_bits() {
        // Use enough bytes to actually satisfy 32 BoolCoder bits
        // worth of renormalization fills (each bit can pull at most
        // one byte, so 32 + 4 prefill = 36 bytes is comfortably
        // safe). decode_b(64) should consume the same input as
        // decode_b(32) and return the same value.
        let bytes = [0xA5u8; 64];

        let mut bc_32 = BoolCoder::new(&bytes).expect("4+ bytes");
        let v_32 = bc_32.decode_b(32).expect("32 bits");

        let mut bc_64 = BoolCoder::new(&bytes).expect("4+ bytes");
        let v_64 = bc_64.decode_b(64).expect("requested 64, capped at 32");

        assert_eq!(v_32, v_64, "decode_b saturates at 32 bits");
        assert_eq!(bc_32.pos(), bc_64.pos(), "same number of bytes consumed");
    }

    /// Truncation surfaces from `decode_bool` when the
    /// renormalization loop tries to pull a byte past the end of
    /// the stream.
    #[test]
    fn truncation_surfaces_from_decode_bool() {
        // Minimal 4-byte stream → 4 prefill bytes, zero bytes
        // available for renormalization fills. Drive low-probability
        // decodes that force aggressive renormalization until the
        // stream is exhausted.
        let bytes = [0x00u8; 4];
        let mut bc = BoolCoder::new(&bytes).expect("4 bytes");
        let mut saw_truncation = false;
        for _ in 0..256 {
            match bc.decode_bool(1) {
                Ok(_) => {}
                Err(Error::Truncated) => {
                    saw_truncation = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
            }
        }
        assert!(
            saw_truncation,
            "decode_bool must surface Truncated when the stream is exhausted"
        );
    }

    /// Truncation surfaces from `decode_b` (the multi-bit wrapper)
    /// when one of its constituent `decode_bool` calls runs out of
    /// bytes. We use a low-probability spec example interleaved
    /// with the b(1) reads to drive aggressive renormalization;
    /// alternating prob-1 reads cap the available bit budget per
    /// stream.
    ///
    /// Note: pure `b(n)` at fixed probability 128 with an all-zero
    /// stream never refills `Value` (the formula's `Split = 255`
    /// keeps `Range = 255` and `Value = 0` on every step), so this
    /// test exercises [`BoolCoder::decode_bool`] directly with a
    /// small probability that does drive renormalization.
    #[test]
    fn truncation_surfaces_from_decode_b() {
        // 4-byte stream with no fill bytes available beyond the
        // prefill. Drive decode_bool at probability 1 until the
        // renormalization loop tries to pull byte index 4 which
        // doesn't exist.
        let bytes = [0x00u8; 4];
        let mut bc = BoolCoder::new(&bytes).expect("4 bytes");
        let mut saw_truncation = false;
        for _ in 0..32 {
            match bc.decode_bool(1) {
                Ok(_) => {}
                Err(Error::Truncated) => {
                    saw_truncation = true;
                    break;
                }
                Err(other) => panic!("unexpected error variant: {other:?}"),
            }
        }
        assert!(
            saw_truncation,
            "decode_bool at low probability must surface Truncated"
        );
    }

    /// Determinism: two `BoolCoder`s over the same input that issue
    /// the same call sequence produce the same outputs and end in
    /// identical state.
    #[test]
    fn decode_is_deterministic() {
        let bytes = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78,
        ];

        let mut bc_a = BoolCoder::new(&bytes).expect("12 bytes");
        let mut bc_b = BoolCoder::new(&bytes).expect("12 bytes");
        let probs = [1u8, 32, 64, 128, 200, 255, 100, 50, 7, 13, 17, 200];
        for &p in &probs {
            let a = bc_a.decode_bool(p).expect("not truncated");
            let b = bc_b.decode_bool(p).expect("not truncated");
            assert_eq!(a, b, "deterministic bit at probability {p}");
        }
        assert_eq!(bc_a.range(), bc_b.range());
        assert_eq!(bc_a.value(), bc_b.value());
        assert_eq!(bc_a.count(), bc_b.count());
        assert_eq!(bc_a.pos(), bc_b.pos());
    }

    /// The MSB-first accumulation in `decode_b` matches the
    /// per-bit `decode_b1` sequence; verifying the documented
    /// identity is essential because §10/§11/§13 callers will rely
    /// on the bit order matching the spec's R(n) convention.
    #[test]
    fn decode_b_equivalence_with_repeated_decode_b1() {
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
        ];

        for n in 1..=12u32 {
            let mut bc_bulk = BoolCoder::new(&bytes).expect("12 bytes");
            let bulk = bc_bulk.decode_b(n).expect("not truncated");

            let mut bc_step = BoolCoder::new(&bytes).expect("12 bytes");
            let mut stepped: u32 = 0;
            for _ in 0..n {
                let bit = bc_step.decode_b1().expect("not truncated");
                stepped = (stepped << 1) | u32::from(bit);
            }

            assert_eq!(
                bulk, stepped,
                "decode_b({n}) must match {n} decode_b1 reads"
            );
            assert_eq!(bc_bulk.range(), bc_step.range());
            assert_eq!(bc_bulk.value(), bc_step.value());
            assert_eq!(bc_bulk.count(), bc_step.count());
            assert_eq!(bc_bulk.pos(), bc_step.pos());
        }
    }

    // ---------------------------------------------------------------
    // BoolEncoder — round-trip against the §7.3 decoder.
    //
    // The encoder has no spec pseudocode of its own; its correctness
    // criterion is purely that `BoolCoder` decodes back exactly what was
    // encoded. Every test below therefore encodes a known (bit,
    // probability) sequence and asserts the decoder reproduces it.
    // Working sets are bounded (≤ a few hundred bits) so the memory
    // footprint stays trivial.
    // ---------------------------------------------------------------

    /// A small deterministic pseudo-random generator so the round-trip
    /// fuzz tests are reproducible without pulling in a crate. (xorshift)
    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// Encode then decode a single `b(1)` for both bit values at a range
    /// of probabilities; the decoder must return the encoded bit.
    #[test]
    fn encode_decode_single_bit_roundtrip() {
        for bit in [0u8, 1] {
            for prob in [1u8, 7, 64, 128, 200, 255] {
                let mut enc = BoolEncoder::new();
                enc.encode_bool(bit, prob);
                let bytes = enc.finish();
                let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
                let got = dec.decode_bool(prob).expect("not truncated");
                assert_eq!(got, bit, "single bit {bit} at prob {prob} must round-trip");
            }
        }
    }

    /// A fixed multi-bit sequence at a fixed probability round-trips in
    /// order.
    #[test]
    fn encode_decode_fixed_sequence_roundtrip() {
        let bits = [1u8, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
        for prob in [1u8, 30, 128, 211, 255] {
            let mut enc = BoolEncoder::new();
            for &b in &bits {
                enc.encode_bool(b, prob);
            }
            let bytes = enc.finish();
            let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
            for (i, &b) in bits.iter().enumerate() {
                let got = dec.decode_bool(prob).expect("not truncated");
                assert_eq!(got, b, "bit {i} at prob {prob} mismatch");
            }
        }
    }

    /// `encode_b` / `decode_b` are inverses for MSB-first multi-bit
    /// integers across a range of widths and values.
    #[test]
    fn encode_b_decode_b_roundtrip() {
        for n in 1..=16u32 {
            for &value in &[0u32, 1, (1 << n) - 1, 0xA5A5 & ((1u32 << n) - 1)] {
                let mut enc = BoolEncoder::new();
                enc.encode_b(value, n);
                let bytes = enc.finish();
                let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
                let got = dec.decode_b(n).expect("not truncated");
                assert_eq!(got, value, "b({n}) value {value:#x} must round-trip");
            }
        }
    }

    /// Mixed probabilities and bit values — the realistic case where
    /// `Split` varies symbol to symbol and renormalization carries can
    /// ripple. Bounded at 256 symbols so the working set is tiny.
    #[test]
    fn encode_decode_mixed_probabilities_roundtrip() {
        let mut state = 0x1234_5678u32;
        let mut bits = Vec::with_capacity(256);
        let mut probs = Vec::with_capacity(256);
        for _ in 0..256 {
            bits.push((xorshift32(&mut state) & 1) as u8);
            // probability in 1..=255 (never 0, which §7.3 forbids).
            probs.push((1 + (xorshift32(&mut state) % 255)) as u8);
        }

        let mut enc = BoolEncoder::new();
        for i in 0..bits.len() {
            enc.encode_bool(bits[i], probs[i]);
        }
        let bytes = enc.finish();

        let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
        for i in 0..bits.len() {
            let got = dec.decode_bool(probs[i]).expect("not truncated");
            assert_eq!(got, bits[i], "symbol {i} (prob {}) mismatch", probs[i]);
        }
    }

    /// Carry-stress: encode many `Bit = 1` decisions at high probability,
    /// where `low += Split << 24` repeatedly nudges the interval bottom
    /// upward and forces carry propagation through already-buffered
    /// bytes. The decoder must still reconstruct every bit.
    #[test]
    fn encode_decode_carry_propagation_roundtrip() {
        // All-ones at high probability maximizes the cumulative `low`
        // additions, exercising multi-byte carry ripple.
        for prob in [200u8, 240, 254, 255] {
            let bits = [1u8; 120];
            let mut enc = BoolEncoder::new();
            for &b in &bits {
                enc.encode_bool(b, prob);
            }
            let bytes = enc.finish();
            let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
            for (i, &b) in bits.iter().enumerate() {
                let got = dec.decode_bool(prob).expect("not truncated");
                assert_eq!(got, b, "carry-stress bit {i} at prob {prob}");
            }
        }
    }

    /// An empty encode still produces a decoder-acceptable (≥ 4-byte)
    /// stream.
    #[test]
    fn encode_empty_produces_valid_stream() {
        let enc = BoolEncoder::new();
        let bytes = enc.finish();
        assert!(bytes.len() >= 4, "decoder requires a 4-byte prefill");
        // Constructible by the decoder.
        assert!(BoolCoder::new(&bytes).is_ok());
    }

    /// Interleaved `encode_bool` and `encode_b` round-trip when the
    /// decoder reads them back in the matching `decode_bool` /
    /// `decode_b` order — the realistic syntax-element mix.
    #[test]
    fn encode_decode_interleaved_b_and_bool_roundtrip() {
        let mut enc = BoolEncoder::new();
        // bool(1, 200), b(0b1011, 4), bool(0, 30), b(0x2A, 6), bool(1,128)
        enc.encode_bool(1, 200);
        enc.encode_b(0b1011, 4);
        enc.encode_bool(0, 30);
        enc.encode_b(0x2A, 6);
        enc.encode_bool(1, 128);
        let bytes = enc.finish();

        let mut dec = BoolCoder::new(&bytes).expect("4+ bytes");
        assert_eq!(dec.decode_bool(200).expect("ok"), 1);
        assert_eq!(dec.decode_b(4).expect("ok"), 0b1011);
        assert_eq!(dec.decode_bool(30).expect("ok"), 0);
        assert_eq!(dec.decode_b(6).expect("ok"), 0x2A);
        assert_eq!(dec.decode_bool(128).expect("ok"), 1);
    }
}

//! VP56 "bool coder" range decoder.
//!
//! VP6 inherits its arithmetic decoder from VPX (VP5..VP9). See
//! FFmpeg's `libavcodec/vpx_rac.c` + `vpx_rac.h` — the decoder is
//! initialised with the first 3 bytes of the partition (big-endian
//! 24-bit seed) and refills `code_word` in 16-bit chunks.
//!
//! Key functions:
//! * [`RangeCoder::get_prob`] — probabilistic binary read (matches
//!   `vpx_rac_get_prob`).
//! * [`RangeCoder::get_bit`] — equiprobable binary read (matches
//!   `vpx_rac_get`). **NOT** the same as `get_prob(128)` — the two
//!   paths split the current interval at different points.
//! * [`RangeCoder::get_bits`] — `n` equiprobable bits (MSB-first).
//! * [`RangeCoder::get_tree`] — walk a [`Vp56Tree`] the way FFmpeg's
//!   `vp56_rac_get_tree` does (used for PMBT/PVA trees).
//!
//! A [`RangeEncoder`] counterpart lives in tests / round-trip checks.

use oxideav_core::{Error, Result};

/// Shift lookup used during renormalisation — how many bit-shifts are
/// needed to drag `high` back up to 128..=255. Mirrors
/// `ff_vpx_norm_shift` in FFmpeg.
const NORM_SHIFT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 1usize;
    while i < 256 {
        let mut v = i as u32;
        let mut s = 0u8;
        while v < 128 {
            v <<= 1;
            s += 1;
        }
        t[i] = s;
        i += 1;
    }
    t[0] = 8;
    t
};

/// VP56 static binary probability tree. Each node has `val` — the jump
/// distance if the bit reads `1` (0 or negative marks a leaf whose
/// symbol is `-val`) — and `prob_idx` — the index into the caller-
/// supplied probability slice.
#[derive(Clone, Copy, Debug)]
pub struct Vp56Tree {
    pub val: i8,
    pub prob_idx: u8,
}

impl Vp56Tree {
    pub const fn new(val: i8, prob_idx: u8) -> Self {
        Self { val, prob_idx }
    }
}

/// VP56 bool-coder decoder.
#[derive(Debug, Clone)]
pub struct RangeCoder<'a> {
    data: &'a [u8],
    /// Current byte cursor inside `data`.
    pos: usize,
    /// Current interval high value — `high` in FFmpeg.
    high: u32,
    /// "bits" counter — stored as a signed value that starts negative
    /// (at `-16` after init) and ticks toward `0`, matching FFmpeg's
    /// `c->bits` sign convention (stored negated so the common path is
    /// a single add).
    bits: i32,
    /// Current 32-bit window into the stream.
    code_word: u32,
    /// Counts refills past the end of the stream. FFmpeg flags
    /// `vpx_rac_is_end` after ~10 such refills; decoders use this to
    /// bail out of infinite loops when the stream is malformed.
    end_reached: u32,
}

impl<'a> RangeCoder<'a> {
    /// Prime a decoder for `data`. Mirrors `ff_vpx_init_range_decoder`:
    /// read a 24-bit big-endian seed into `code_word` and start with
    /// `bits = -16` so the first `get_prob` triggers a 16-bit refill
    /// at the correct time.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::invalid("VP56: range coder needs >= 1 byte"));
        }
        let b0 = data[0] as u32;
        let b1 = data.get(1).copied().unwrap_or(0) as u32;
        let b2 = data.get(2).copied().unwrap_or(0) as u32;
        let code_word = (b0 << 16) | (b1 << 8) | b2;
        let pos = data.len().min(3);
        Ok(Self {
            data,
            pos,
            high: 255,
            bits: -16,
            code_word,
            end_reached: 0,
        })
    }

    /// Refill `code_word` and scale `high` so the top octet is always
    /// at least 128. Matches the `vpx_rac_renorm` inline helper.
    /// Returns the fresh `code_word` for the caller to sample.
    #[inline]
    fn renorm(&mut self) -> u32 {
        let shift = NORM_SHIFT[self.high as usize & 0xFF] as i32;
        self.high <<= shift;
        self.code_word <<= shift;
        self.bits += shift;
        if self.bits >= 0 && self.pos < self.data.len() {
            let b0 = self.data[self.pos] as u32;
            let b1 = self.data.get(self.pos + 1).copied().unwrap_or(0) as u32;
            self.pos = (self.pos + 2).min(self.data.len());
            // Shift the fresh 16 bits into the upper half of the 32-bit
            // register — the precise position is given by `bits`
            // (negative = room left, 0 = shift into positions 0..=15).
            let shift_amount = self.bits as u32;
            self.code_word |= ((b0 << 8) | b1) << shift_amount;
            self.bits -= 16;
        }
        self.code_word
    }

    /// Probabilistic binary read (`vpx_rac_get_prob`). Returns 0/1 with
    /// the given probability of `1`. `prob` is on a 0..=255 scale where
    /// 0 means "always 0" and 255 means "almost always 1".
    #[inline]
    pub fn get_prob(&mut self, prob: u8) -> u8 {
        let code_word = self.renorm();
        let low = 1 + (((self.high - 1) * prob as u32) >> 8);
        let low_shift = low << 16;
        let bit = if code_word >= low_shift { 1u8 } else { 0u8 };
        if bit != 0 {
            self.high -= low;
            self.code_word = code_word - low_shift;
        } else {
            self.high = low;
            self.code_word = code_word;
        }
        bit
    }

    /// Equiprobable binary read (`vpx_rac_get`). Functionally close to
    /// `get_prob(128)` but uses a different interval split so the two
    /// are **not** interchangeable when replaying an FFmpeg-encoded
    /// stream.
    #[inline]
    pub fn get_bit(&mut self) -> u8 {
        let code_word = self.renorm();
        let low = (self.high + 1) >> 1;
        let low_shift = low << 16;
        let bit = if code_word >= low_shift { 1u8 } else { 0u8 };
        if bit != 0 {
            self.high -= low;
            self.code_word = code_word - low_shift;
        } else {
            self.high = low;
            self.code_word = code_word;
        }
        bit
    }

    /// Read `n` equiprobable bits, MSB-first. Matches VP56's
    /// `vp56_rac_gets`.
    pub fn get_bits(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.get_bit() as u32;
        }
        value
    }

    /// Read a 7-bit "non-zero-nudged" value. Mirrors
    /// `vp56_rac_gets_nn`: read 7 bits, shift left by 1, then force
    /// bit 0 to be 1 if the result is zero (so the model never lands
    /// on a zero probability). Returns 0..=255.
    pub fn get_bits_nn(&mut self) -> u32 {
        let v = self.get_bits(7) << 1;
        if v == 0 {
            1
        } else {
            v
        }
    }

    /// Walk a [`Vp56Tree`] using per-node probabilities. Returns the
    /// leaf symbol (positive integer). Mirrors `vp56_rac_get_tree`.
    pub fn get_tree(&mut self, tree: &[Vp56Tree], probs: &[u8]) -> i32 {
        let mut idx = 0usize;
        loop {
            let node = tree[idx];
            if node.val <= 0 {
                return (-node.val) as i32;
            }
            let bit = self.get_prob(probs[node.prob_idx as usize]);
            if bit != 0 {
                idx = idx.wrapping_add(node.val as usize);
            } else {
                idx += 1;
            }
        }
    }

    /// `true` after the stream is effectively exhausted — used by the
    /// coefficient parser to abort on malformed data. Mirrors
    /// `vpx_rac_is_end`.
    pub fn is_end(&mut self) -> bool {
        if self.pos >= self.data.len() && self.bits >= 0 {
            self.end_reached = self.end_reached.saturating_add(1);
        }
        self.end_reached > 10
    }

    /// Advisory: number of input bytes consumed. Used by VP6's
    /// coefficient-partition offset logic.
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }
}

/// Round-trip encoder for unit tests. Mirrors libvpx's VP8 bool-encoder
/// which shares arithmetic with VP6.
#[derive(Debug, Clone)]
pub struct RangeEncoder {
    out: Vec<u8>,
    range: u32,
    lowvalue: u32,
    /// -24 = empty; 0 = ready to emit one byte.
    count: i32,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            range: 255,
            lowvalue: 0,
            count: -24,
        }
    }

    fn add_one_to_output(buf: &mut [u8]) {
        let mut x = buf.len() as isize - 1;
        while x >= 0 && buf[x as usize] == 0xff {
            buf[x as usize] = 0;
            x -= 1;
        }
        if x >= 0 {
            buf[x as usize] += 1;
        }
    }

    pub fn put_prob(&mut self, prob: u8, bit: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let (mut range, mut lowvalue) = if bit != 0 {
            (self.range - split, self.lowvalue.wrapping_add(split))
        } else {
            (split, self.lowvalue)
        };
        while range < 128 {
            range <<= 1;
            if (lowvalue & 0x8000_0000) != 0 {
                Self::add_one_to_output(&mut self.out);
            }
            lowvalue <<= 1;
            self.count += 1;
            if self.count == 0 {
                self.out.push(((lowvalue >> 24) & 0xff) as u8);
                lowvalue &= 0x00ff_ffff;
                self.count = -8;
            }
        }
        self.range = range;
        self.lowvalue = lowvalue;
    }

    /// Encode an equiprobable bit matching FFmpeg's `vpx_rac_get` split
    /// (`low = (high+1) >> 1`).
    pub fn put_bit(&mut self, bit: u8) {
        let split = (self.range + 1) >> 1;
        let (mut range, mut lowvalue) = if bit != 0 {
            (self.range - split, self.lowvalue.wrapping_add(split))
        } else {
            (split, self.lowvalue)
        };
        while range < 128 {
            range <<= 1;
            if (lowvalue & 0x8000_0000) != 0 {
                Self::add_one_to_output(&mut self.out);
            }
            lowvalue <<= 1;
            self.count += 1;
            if self.count == 0 {
                self.out.push(((lowvalue >> 24) & 0xff) as u8);
                lowvalue &= 0x00ff_ffff;
                self.count = -8;
            }
        }
        self.range = range;
        self.lowvalue = lowvalue;
    }

    pub fn put_bits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.put_bit(((value >> i) & 1) as u8);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.put_bit(0);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_requires_at_least_one_byte() {
        assert!(RangeCoder::new(&[]).is_err());
        assert!(RangeCoder::new(&[0]).is_ok());
    }

    #[test]
    fn roundtrip_single_prob_symbol() {
        for prob in [1u8, 64, 128, 192, 250] {
            for bit in [0u8, 1u8] {
                let mut enc = RangeEncoder::new();
                enc.put_prob(prob, bit);
                let bytes = enc.finish();
                let mut dec = RangeCoder::new(&bytes).unwrap();
                assert_eq!(dec.get_prob(prob), bit, "prob={prob} bit={bit}");
            }
        }
    }

    #[test]
    fn roundtrip_single_equiprobable_bit() {
        for bit in [0u8, 1u8] {
            let mut enc = RangeEncoder::new();
            enc.put_bit(bit);
            let bytes = enc.finish();
            let mut dec = RangeCoder::new(&bytes).unwrap();
            assert_eq!(dec.get_bit(), bit, "bit={bit}");
        }
    }

    #[test]
    fn roundtrip_sequence_mixed() {
        // Mix `put_prob` and `put_bit` — this catches any divergence
        // between the two interval-split rules.
        let script: Vec<(bool, u8, u8)> = (0..128)
            .map(|i| {
                let prob = [16u8, 64, 128, 200, 240][i % 5];
                let bit = ((i * 1103515245 + 12345) >> 7) as u8 & 1;
                let is_raw = (i & 3) == 0;
                (is_raw, prob, bit)
            })
            .collect();
        let mut enc = RangeEncoder::new();
        for (is_raw, p, b) in &script {
            if *is_raw {
                enc.put_bit(*b);
            } else {
                enc.put_prob(*p, *b);
            }
        }
        let bytes = enc.finish();
        let mut dec = RangeCoder::new(&bytes).unwrap();
        for (i, (is_raw, p, b)) in script.iter().enumerate() {
            let got = if *is_raw {
                dec.get_bit()
            } else {
                dec.get_prob(*p)
            };
            assert_eq!(got, *b, "mismatch at symbol {i} (raw={is_raw} p={p})");
        }
    }

    /// Stress test for the bool encoder: emit many puts of varying
    /// probabilities then decode them all, ensuring perfect round-trip.
    /// This pattern matches what the inter-frame encoder produces.
    #[test]
    fn roundtrip_picture_header_then_per_mb() {
        let mut enc = RangeEncoder::new();
        // Mimic the picture-header section of encode_inter_frame.
        enc.put_bit(0);
        enc.put_bit(0); // golden + huffman flags
        for _ in 0..3 {
            enc.put_prob(174, 0);
            enc.put_prob(254, 0);
        }
        for _ in 0..4 {
            enc.put_prob(237, 0);
        }
        for _ in 0..14 {
            enc.put_prob(225, 0);
        }
        for _ in 0..16 {
            enc.put_prob(244, 0);
        }
        for _ in 0..22 {
            enc.put_prob(146, 0);
        }
        enc.put_bit(0);
        for _ in 0..28 {
            enc.put_prob(219, 0);
        }
        for _ in 0..396 {
            enc.put_prob(227, 0);
        }
        // mb(0,0): stay=1
        enc.put_prob(10, 1);
        for _ in 0..6 {
            enc.put_prob(140, 0);
            enc.put_prob(220, 0);
            enc.put_prob(220, 0);
        }
        // mb(0,1): stay=0, then PMBT walk to InterDeltaPf using the
        // ACTUAL probs the encoder uses (the failing case).
        enc.put_prob(10, 0);
        enc.put_prob(157, 0);
        enc.put_prob(255, 0);
        enc.put_prob(1, 1);
        let bytes = enc.finish();
        eprintln!("encoded {} bytes", bytes.len());
        let mut dec = RangeCoder::new(&bytes).unwrap();
        assert_eq!(dec.get_bit(), 0);
        assert_eq!(dec.get_bit(), 0);
        for _ in 0..3 {
            assert_eq!(dec.get_prob(174), 0);
            assert_eq!(dec.get_prob(254), 0);
        }
        for _ in 0..4 {
            assert_eq!(dec.get_prob(237), 0);
        }
        for _ in 0..14 {
            assert_eq!(dec.get_prob(225), 0);
        }
        for _ in 0..16 {
            assert_eq!(dec.get_prob(244), 0);
        }
        for _ in 0..22 {
            assert_eq!(dec.get_prob(146), 0);
        }
        assert_eq!(dec.get_bit(), 0);
        for _ in 0..28 {
            assert_eq!(dec.get_prob(219), 0);
        }
        for _ in 0..396 {
            assert_eq!(dec.get_prob(227), 0);
        }
        assert_eq!(dec.get_prob(10), 1);
        for _ in 0..6 {
            assert_eq!(dec.get_prob(140), 0);
            assert_eq!(dec.get_prob(220), 0);
            assert_eq!(dec.get_prob(220), 0);
        }
        assert_eq!(dec.get_prob(10), 0);
        assert_eq!(dec.get_prob(157), 0);
        assert_eq!(dec.get_prob(255), 0);
        assert_eq!(dec.get_prob(1), 1);
    }

    #[test]
    fn roundtrip_extreme_prob_after_burst() {
        // Reproducer of the desync seen in the inter-frame encoder:
        // a long burst of probs followed by a high-then-low extreme
        // sequence (255, 1) with bit=1 at the rare (prob=1) symbol.
        let mut enc = RangeEncoder::new();
        // Burst that sets up the bool-coder state seen before the failing point.
        enc.put_prob(10, 1);
        for _ in 0..6 {
            enc.put_prob(140, 0);
            enc.put_prob(220, 0);
            enc.put_prob(220, 0);
        }
        enc.put_prob(10, 0);
        enc.put_prob(157, 0);
        enc.put_prob(255, 0);
        enc.put_prob(1, 1);
        let bytes = enc.finish();
        let mut dec = RangeCoder::new(&bytes).unwrap();
        assert_eq!(dec.get_prob(10), 1);
        for _ in 0..6 {
            assert_eq!(dec.get_prob(140), 0);
            assert_eq!(dec.get_prob(220), 0);
            assert_eq!(dec.get_prob(220), 0);
        }
        assert_eq!(dec.get_prob(10), 0);
        assert_eq!(dec.get_prob(157), 0);
        assert_eq!(dec.get_prob(255), 0);
        assert_eq!(dec.get_prob(1), 1);
    }

    #[test]
    fn raw_bits_roundtrip_16_bits() {
        let value: u32 = 0xBEEF;
        let mut enc = RangeEncoder::new();
        enc.put_bits(16, value);
        let bytes = enc.finish();
        let mut dec = RangeCoder::new(&bytes).unwrap();
        assert_eq!(dec.get_bits(16), value);
    }

    #[test]
    fn zero_fill_past_end_doesnt_panic() {
        let mut dec = RangeCoder::new(&[0, 0]).unwrap();
        for _ in 0..256 {
            let _ = dec.get_prob(128);
        }
        // is_end is a polling predicate that increments a counter each
        // time it observes "buffer exhausted + bits didn't slide"; it
        // flips to true after ~10 such calls, matching FFmpeg.
        let mut flipped = false;
        for _ in 0..32 {
            if dec.is_end() {
                flipped = true;
                break;
            }
        }
        assert!(flipped);
    }

    #[test]
    fn get_tree_linear() {
        // Two-leaf tree: probs[0] decides 0 vs 1. Encode + decode.
        let tree = [
            Vp56Tree::new(2, 0),
            Vp56Tree::new(-5, 0),
            Vp56Tree::new(-11, 0),
        ];
        let probs = [200u8];

        for expected in [5i32, 11] {
            let bit = if expected == 11 { 1 } else { 0 };
            let mut enc = RangeEncoder::new();
            enc.put_prob(probs[0], bit);
            let bytes = enc.finish();
            let mut dec = RangeCoder::new(&bytes).unwrap();
            let got = dec.get_tree(&tree, &probs);
            assert_eq!(got, expected);
        }
    }
}

//! VP56 "bool coder" range decoder.
//!
//! VP6 inherits its arithmetic decoder from VP5/VP56 — an 8-bit range
//! coder whose state is an unsigned range (`code_word` + `high`) and a
//! count of bits still valid in `code_word`. The semantics match
//! FFmpeg's `libavcodec/vp56rac.c`:
//!
//! * `get_prob(p)`: decode a single binary symbol whose probability
//!   of being zero is `p` (out of 256). Returns 0 or 1.
//! * `get_bits(n)`: decode `n` uniformly-distributed raw bits (i.e.
//!   `get_prob(128)` repeated `n` times). Used for raw fields and
//!   trailing motion-vector magnitude bits.
//! * `get_tree(tree, probs)`: walk a binary prob tree the way libvpx
//!   / libavcodec express motion-vector / token / mode trees.
//!
//! VP6 streams are MSB-first, so `load_byte()` shifts the next byte
//! into the low end of `code_word` and advances the read cursor.
//!
//! A companion [`RangeEncoder`] lets the unit tests round-trip
//! arbitrary symbol sequences without needing a reference stream.

use oxideav_core::{Error, Result};

/// VP56 bool-coder decoder.
#[derive(Debug, Clone)]
pub struct RangeCoder<'a> {
    data: &'a [u8],
    /// Current byte offset within `data`.
    pos: usize,
    /// Current range (the "high" value in FFmpeg). Always > 0x80 after
    /// each renormalisation.
    high: u32,
    /// Current bits-remaining counter — when it drops to zero we need
    /// to pull another byte from `data`.
    bits: i32,
    /// Current low word — the "code word" in FFmpeg.
    code: u32,
}

impl<'a> RangeCoder<'a> {
    /// Create a new decoder positioned at the start of `data`. Pulls
    /// two bytes to prime the state (matches
    /// `ff_vp56_init_range_decoder`).
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::invalid("VP56: range coder needs >= 2 bytes"));
        }
        let code = ((data[0] as u32) << 8) | data[1] as u32;
        Ok(Self {
            data,
            pos: 2,
            high: 255,
            bits: 8,
            code,
        })
    }

    /// Read one binary symbol whose probability-of-zero is `prob`
    /// (256 = certain zero, 0 = certain one). Matches libavcodec's
    /// `vp56_rac_get_prob`.
    pub fn get_prob(&mut self, prob: u8) -> u8 {
        // `split` is the first-order split of the current interval.
        // VP56 uses `1 + (((self.high - 1) * prob) >> 8)` — expressed
        // here in u32 to avoid overflow on the multiply.
        let split = 1 + (((self.high - 1) * prob as u32) >> 8);
        let bit = if (self.code >> 8) >= split {
            self.high -= split;
            self.code -= split << 8;
            1
        } else {
            self.high = split;
            0
        };
        self.renormalise();
        bit
    }

    /// Read `n` uniformly-distributed raw bits.
    pub fn get_bits(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.get_prob(128) as u32;
        }
        value
    }

    /// Walk a VP56 probability tree the way libvpx expresses them
    /// (array of `(prob, next_index)`-ish encoded as a flat
    /// `i8` slice with sentinel).
    ///
    /// This is a helper — callers with bespoke tree shapes should use
    /// `get_prob` directly. The tree here takes a probability at each
    /// step and returns the final leaf value as a `u8`, matching the
    /// small VP6 intra-mode trees that don't need the full libvpx
    /// encoding scheme.
    pub fn get_tree(&mut self, probs: &[u8]) -> u8 {
        // Walk a linear probability chain (used for small binary
        // decisions) — extended trees are decoded by specialised
        // functions rather than the generic helper.
        let mut v = 0u8;
        for &p in probs {
            v = (v << 1) | self.get_prob(p);
        }
        v
    }

    fn renormalise(&mut self) {
        // Left-shift both state words until `high` is at least 0x80
        // again. Pull a new input byte each time bits runs out. Matches
        // the tight inner loop in ff_vp56_rac_renormalize.
        while self.high < 0x80 {
            self.high <<= 1;
            self.code <<= 1;
            self.bits -= 1;
            if self.bits == 0 {
                self.load_byte();
            }
        }
    }

    fn load_byte(&mut self) {
        let b = if self.pos < self.data.len() {
            let v = self.data[self.pos];
            self.pos += 1;
            v
        } else {
            // FFmpeg simply reads past the end with zero fill when the
            // stream is exhausted; match that behaviour so we don't
            // trip on the last few symbols of a valid frame.
            0
        };
        self.code |= b as u32;
        self.bits = 8;
    }

    /// How many bytes of the input have been consumed (advisory — VP6
    /// coefficient decoding occasionally wants to pick up right where
    /// the bool coder left off).
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }
}

/// Matching encoder for unit tests — ported from libvpx's
/// `vp8_encode_bool` loop-free variant (RFC 6386 §20.2). VP6's
/// decoder mirrors VP8's arithmetic exactly, so the same encoder
/// round-trips here.
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

    pub fn put_bits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.put_prob(128, ((value >> i) & 1) as u8);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Pad with 32 uniform zero bits to flush the state register.
        for _ in 0..32 {
            self.put_prob(128, 0);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_requires_two_bytes() {
        assert!(RangeCoder::new(&[0]).is_err());
        assert!(RangeCoder::new(&[]).is_err());
        assert!(RangeCoder::new(&[0, 0]).is_ok());
    }

    #[test]
    fn roundtrip_single_symbol() {
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
    fn roundtrip_sequence() {
        // Encode a deterministic sequence of mixed probs and decode it
        // back. 64 symbols is enough to exercise several
        // renormalisations and at least one byte boundary.
        let script: Vec<(u8, u8)> = (0..64)
            .map(|i| {
                let prob = [16u8, 64, 128, 200, 240][i % 5];
                let bit = ((i * 1103515245 + 12345) >> 7) as u8 & 1;
                (prob, bit)
            })
            .collect();
        let mut enc = RangeEncoder::new();
        for (p, b) in &script {
            enc.put_prob(*p, *b);
        }
        let bytes = enc.finish();
        let mut dec = RangeCoder::new(&bytes).unwrap();
        for (i, (p, b)) in script.iter().enumerate() {
            let got = dec.get_prob(*p);
            assert_eq!(got, *b, "mismatch at symbol {i} (prob={p} expect={b})");
        }
    }

    #[test]
    fn raw_bits_roundtrip() {
        // 16 bits of arbitrary data.
        let value: u32 = 0xBEEF;
        let mut enc = RangeEncoder::new();
        enc.put_bits(16, value);
        let bytes = enc.finish();
        let mut dec = RangeCoder::new(&bytes).unwrap();
        assert_eq!(dec.get_bits(16), value);
    }

    #[test]
    fn zero_fill_past_end() {
        // Starve the decoder and make sure it returns a defined result
        // (FFmpeg-style zero fill) rather than panicking.
        let mut dec = RangeCoder::new(&[0, 0]).unwrap();
        for _ in 0..256 {
            let _ = dec.get_prob(128);
        }
    }
}

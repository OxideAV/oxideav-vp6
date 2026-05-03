//! VP6 Huffman coefficient path.
//!
//! Spec reference: VP6 *Bitstream & Decoder Specification* sections 7.2
//! (Huffman Decoder), 13.1 (DCT Token Huffman Tree), 13.2.2 (Huffman
//! Decoding DC Values), 13.3.2 (Decoding Huffman AC Coefficients),
//! 13.3.3.2 (Decoding Huffman AC Zero Runs), and 13.4 (Decoding Huffman
//! EOB and DC 0 Runs). Vendored at `docs/video/vp6/vp6_format.pdf`.
//!
//! Important spec facts encoded here:
//!
//! * The `UseHuffman` bit (frame header bit b(1) — see Table 1) only
//!   applies to the **second** data partition (the coefficient
//!   partition). Partition 1 (mode-update + MV header) is *always*
//!   bool-coded; we still use [`crate::range_coder::RangeCoder`] for it.
//!
//! * The `R(n)` extra-bits operator appearing in the Huffman pseudocode
//!   on spec pages 65, 66, 74, 81 is a **raw** `n`-bit MSB-first read in
//!   the Huffman path (see spec page 57: *"In Huffman encodings these
//!   bits are just pumped on to the bitstream"*). That contrasts with
//!   the arithmetic-coder path where `R(n)` actually means "decode `n`
//!   probabilistically-equiprobable bits via the bool coder". The
//!   [`BitReader`] / [`BitWriter`] types in this module implement the
//!   raw-bits convention.
//!
//! * Per spec page 16 Figure 1 + page 22 Figure 7, when the Huffman
//!   bit is set the bitstream is laid out as
//!   `header || partition1(bool) || partition2(huffman)` and the
//!   `Buff2Offset` field in the frame header points at the start of
//!   partition 2. Partition 2 is a raw bitstream — no bool-coder
//!   priming; just MSB-first bytes.
//!
//! * Tree construction uses the algorithm in spec page 14
//!   (`VP6_CreateHuffmanTree`): at each step pick the two least-probable
//!   nodes, merge them; the codeword is a left/right walk from the
//!   root with `0` for left and `1` for right. We compute codeword +
//!   length tables directly from the result so encode is O(1) per
//!   symbol.

use oxideav_core::{Error, Result};

use crate::models::Vp6Model;

/// Number of token symbols. Matches spec page 57 Table 18 — one entry
/// per token (`ZERO_TOKEN` through `DCT_EOB_TOKEN`).
pub const TOKEN_COUNT: usize = 12;

pub const ZERO_TOKEN: u8 = 0;
pub const ONE_TOKEN: u8 = 1;
pub const TWO_TOKEN: u8 = 2;
pub const THREE_TOKEN: u8 = 3;
pub const FOUR_TOKEN: u8 = 4;
pub const DCT_VAL_CATEGORY1: u8 = 5;
pub const DCT_VAL_CATEGORY2: u8 = 6;
pub const DCT_VAL_CATEGORY3: u8 = 7;
pub const DCT_VAL_CATEGORY4: u8 = 8;
pub const DCT_VAL_CATEGORY5: u8 = 9;
pub const DCT_VAL_CATEGORY6: u8 = 10;
pub const DCT_EOB_TOKEN: u8 = 11;

/// Per-token raw extra-bit count (sign bit included).
/// Mirrors spec Table 18 column "#Extra Bits (incl. sign)".
pub const TOKEN_EXTRA_BITS: [u8; TOKEN_COUNT] = [0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 12, 0];
/// Per-token absolute-value floor — the smallest |coefficient| this
/// token can encode. Spec Table 18 column "Min".
pub const TOKEN_VALUE_MIN: [i32; TOKEN_COUNT] = [0, 1, 2, 3, 4, 5, 7, 11, 19, 35, 67, 0];
/// Per-token absolute-value ceiling. Spec Table 18 column "Max" + 1
/// (so the half-open `[min, max+1)` range matches the extra-bits count).
pub const TOKEN_VALUE_RANGE: [i32; TOKEN_COUNT] = [1, 2, 3, 4, 5, 7, 11, 19, 35, 67, 2115, 0];

/// AC band index by encoded-coefficient position. Spec Table 36
/// "AC Huffman Prob Band Index": position 1 → band 0, positions 2-4 →
/// band 1, positions 5-10 → band 2, positions 11-63 → band 3.
pub const AC_HUFF_BAND: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 1usize;
    while i <= 63 {
        t[i] = if i == 1 {
            0
        } else if i <= 4 {
            1
        } else if i <= 10 {
            2
        } else {
            3
        };
        i += 1;
    }
    t
};

/// ZRL band by current encoded-coeff position. Spec Table 37: positions
/// 1-5 → band 0, positions 6-63 → band 1.
pub const ZRL_BAND: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0usize;
    while i < 64 {
        t[i] = if i <= 5 { 0 } else { 1 };
        i += 1;
    }
    t
};

// =====================================================================
// Bit IO
// =====================================================================

/// MSB-first raw-bit reader. Used for the Huffman partition (which is a
/// plain raw bitstream — see spec page 57 + module docs).
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Next bit to read inside `data[pos]`, counted from MSB (bit 7
    /// down to bit 0).
    bit: u8,
    /// True after reading past the end — every subsequent read returns 0
    /// so a malformed Huffman stream can't loop forever (decoder still
    /// detects the EOB / position == 63 termination).
    end_reached: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit: 7,
            end_reached: false,
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        if self.pos >= self.data.len() {
            self.end_reached = true;
            return 0;
        }
        let v = ((self.data[self.pos] >> self.bit) & 1) as u32;
        if self.bit == 0 {
            self.bit = 7;
            self.pos += 1;
        } else {
            self.bit -= 1;
        }
        v
    }

    /// Read `n` raw bits, MSB-first. Mirrors the spec's `R(n)` operator
    /// in the Huffman path.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit();
        }
        value
    }

    pub fn end_reached(&self) -> bool {
        self.end_reached
    }
}

/// MSB-first raw-bit writer. Pads the trailing partial byte with zeros
/// on [`Self::finish`].
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    out: Vec<u8>,
    /// Bits collected in the current byte, in the high bits.
    cur: u8,
    /// Number of bits already in `cur` (0..=7). `0` means `cur` is
    /// fresh / empty.
    cur_bits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn write_bit(&mut self, bit: u32) {
        self.cur |= ((bit & 1) as u8) << (7 - self.cur_bits);
        self.cur_bits += 1;
        if self.cur_bits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    /// Write `n` bits of `value`, MSB-first. The lower `n` bits of
    /// `value` are emitted; higher bits are ignored.
    #[inline]
    pub fn write_bits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Finalise the bitstream. Pads the trailing partial byte with
    /// zeros (which never decode to a Huffman leaf — the decoder
    /// terminates per-block on EOB or coeff position == 63 well before
    /// the padding is reached).
    pub fn finish(mut self) -> Vec<u8> {
        if self.cur_bits != 0 {
            self.out.push(self.cur);
        }
        self.out
    }

    pub fn bytes_written(&self) -> usize {
        self.out.len() + if self.cur_bits != 0 { 1 } else { 0 }
    }
}

// =====================================================================
// Huffman tree (spec page 12-14)
// =====================================================================

/// Huffman tree node — leaf if `symbol >= 0`, internal otherwise.
/// Mirrors the spec's `HUFF_NODE` struct on page 13.
#[derive(Clone, Copy, Debug)]
struct HuffNode {
    /// `>= 0` for a leaf (the decoded symbol); `-1` for an internal
    /// node.
    symbol: i32,
    prob: u32,
    left: i32,
    right: i32,
}

/// A built Huffman tree. The tree is stored as a flat array of nodes;
/// the root is the last entry. Symbol → (codeword, length) is also
/// pre-computed for the encode side.
#[derive(Debug, Clone)]
pub struct HuffTree {
    nodes: Vec<HuffNode>,
    root: usize,
    /// `codes[symbol]` — packed `(codeword, length)`. `length == 0`
    /// means the symbol is unreachable in this tree (probability was
    /// effectively zero — should never be queried by a well-formed
    /// encoder).
    codes: Vec<(u32, u8)>,
    n_symbols: usize,
}

impl HuffTree {
    /// Build a Huffman tree from `n` symbol probabilities. Mirrors the
    /// spec's `VP6_CreateHuffmanTree` pseudocode on page 14:
    ///
    /// 1. Seed `n` leaf nodes with their probabilities.
    /// 2. Sort ascending (stable by index for tie-break).
    /// 3. Repeatedly merge the two least-probable into a new internal
    ///    node, then re-sort the tail.
    pub fn build(probs: &[u32]) -> Self {
        let n = probs.len();
        assert!(n >= 1, "Huffman tree requires at least one symbol");
        let mut nodes: Vec<HuffNode> = (0..n)
            .map(|i| HuffNode {
                symbol: i as i32,
                prob: probs[i].max(1),
                left: -1,
                right: -1,
            })
            .collect();
        // `order` carries the sort-list indices into `nodes`. We sort
        // it stably by `(prob, original_index)` to mirror the spec's
        // "maintaining the relative order of symbols having equal
        // probabilities" requirement.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (nodes[i].prob, i));

        // Single-symbol corner case: produce a 1-bit tree so the encode
        // side has a definite codeword. Decode-side never queries the
        // tree because the spec keeps tokens that can't appear out of
        // the prob set.
        if n == 1 {
            let mut codes = vec![(0u32, 0u8); n];
            codes[0] = (0, 1);
            return Self {
                nodes,
                root: 0,
                codes,
                n_symbols: n,
            };
        }

        // Per spec page 14: for `i in 0..N-1`, take SortList[2*i] and
        // SortList[2*i+1] as the two least-probable, merge into a new
        // node at position N+i, then re-insert maintaining order.
        for _ in 0..(n - 1) {
            let l_idx = order.remove(0);
            let r_idx = order.remove(0);
            let new_idx = nodes.len();
            let merged_prob = nodes[l_idx].prob.saturating_add(nodes[r_idx].prob);
            nodes.push(HuffNode {
                symbol: -1,
                prob: merged_prob,
                left: l_idx as i32,
                right: r_idx as i32,
            });
            // Re-insert, stable by (prob, new_idx) — new_idx is monotonic
            // so newly-inserted nodes always sort *after* equally-probable
            // earlier nodes (matches spec's "relative order of nodes
            // having equal probability" requirement).
            let pos = order.partition_point(|&i| (nodes[i].prob, i) < (merged_prob, new_idx));
            order.insert(pos, new_idx);
        }

        let root = order[0];

        // Pre-compute codeword + length per symbol via DFS from root.
        let mut codes = vec![(0u32, 0u8); n];
        // `stack` carries `(node_idx, codeword, length)`.
        let mut stack: Vec<(usize, u32, u8)> = Vec::with_capacity(n);
        stack.push((root, 0, 0));
        while let Some((idx, code, len)) = stack.pop() {
            let node = nodes[idx];
            if node.symbol >= 0 {
                let s = node.symbol as usize;
                // Single-symbol tree: length stays at 0; bump to 1 so
                // the encoder emits a deterministic bit. Multi-symbol
                // trees never have a 0-length leaf (each leaf is at
                // depth >= 1).
                let final_len = if len == 0 { 1 } else { len };
                codes[s] = (code, final_len);
            } else {
                // Right sub-tree first so left is popped (and visited)
                // first — pure tidiness; doesn't change correctness.
                if node.right >= 0 {
                    stack.push((node.right as usize, (code << 1) | 1, len + 1));
                }
                if node.left >= 0 {
                    stack.push((node.left as usize, code << 1, len + 1));
                }
            }
        }

        Self {
            nodes,
            root,
            codes,
            n_symbols: n,
        }
    }

    /// Walk the tree from the root, reading one bit per internal node,
    /// until a leaf is reached. Returns the leaf symbol. `0` is
    /// substituted on unexpected stream end — the per-block decode loop
    /// then terminates via the `coeff == 63` / EOB check.
    pub fn decode_symbol(&self, br: &mut BitReader<'_>) -> u32 {
        let mut idx = self.root;
        // Hard cap on tree depth to defend against malformed input —
        // the deepest legal tree has `n_symbols - 1` internal nodes,
        // i.e. depth ≤ `n_symbols`. Tokens have 12 symbols → max 12
        // internal-node steps.
        for _ in 0..(self.n_symbols + 4) {
            let node = self.nodes[idx];
            if node.symbol >= 0 {
                return node.symbol as u32;
            }
            let bit = br.read_bit();
            let next = if bit == 0 { node.left } else { node.right };
            if next < 0 {
                return 0;
            }
            idx = next as usize;
            if br.end_reached() {
                return 0;
            }
        }
        0
    }

    /// Encode a symbol into `bw`. Panics on an unreachable symbol —
    /// callers should only emit symbols that were present in the prob
    /// set used to build the tree.
    pub fn encode_symbol(&self, bw: &mut BitWriter, symbol: u32) {
        let (code, len) = self.codes[symbol as usize];
        debug_assert!(
            len > 0,
            "Huffman encode of unreachable symbol {symbol} (length=0)"
        );
        bw.write_bits(len as u32, code);
    }
}

// =====================================================================
// Probability conversion: bool-coder tree → Huffman leaf probabilities
// =====================================================================

/// Convert the 11-entry DC/AC bool-tree node-probability vector into
/// the 12-entry Huffman leaf-probability vector. Direct port of
/// `DCTTokenBoolTreeToHuffProbs` (spec pages 60-61).
///
/// The shift-by-8 normalisation is done implicitly here by leaving the
/// values in 16-bit-ish ranges; the tree builder only cares about
/// relative ordering so the `>> 8` divisions in the spec pseudocode
/// don't change the resulting tree shape (we keep them so the layout
/// matches verbatim and is easy to cross-reference against the spec).
pub fn dct_token_bool_tree_to_huff_probs(node_prob: &[u8; 11]) -> [u32; 12] {
    let np: [u32; 11] = std::array::from_fn(|i| node_prob[i] as u32);
    let mut hp = [0u32; 12];
    hp[DCT_EOB_TOKEN as usize] = (np[0] * np[1]) >> 8;
    hp[ZERO_TOKEN as usize] = (np[0] * (255 - np[1])) >> 8;

    let prob = 255 - np[0];
    hp[ONE_TOKEN as usize] = (prob * np[2]) >> 8;

    let prob_b = (prob * (255 - np[2])) >> 8;
    let prob1 = (prob_b * np[3]) >> 8;
    hp[TWO_TOKEN as usize] = (prob1 * np[4]) >> 8;

    let prob1b = (prob1 * (255 - np[4])) >> 8;
    hp[THREE_TOKEN as usize] = (prob1b * np[5]) >> 8;
    hp[FOUR_TOKEN as usize] = (prob1b * (255 - np[5])) >> 8;

    let prob_c = (prob_b * (255 - np[3])) >> 8;
    let prob1c = (prob_c * np[6]) >> 8;
    hp[DCT_VAL_CATEGORY1 as usize] = (prob1c * np[7]) >> 8;
    hp[DCT_VAL_CATEGORY2 as usize] = (prob1c * (255 - np[7])) >> 8;

    let prob_d = (prob_c * (255 - np[6])) >> 8;
    let prob1d = (prob_d * np[8]) >> 8;
    hp[DCT_VAL_CATEGORY3 as usize] = (prob1d * np[9]) >> 8;
    hp[DCT_VAL_CATEGORY4 as usize] = (prob1d * (255 - np[9])) >> 8;

    let prob_e = (prob_d * (255 - np[8])) >> 8;
    hp[DCT_VAL_CATEGORY5 as usize] = (prob_e * np[10]) >> 8;
    hp[DCT_VAL_CATEGORY6 as usize] = (prob_e * (255 - np[10])) >> 8;

    hp
}

/// Convert the 14-entry ZRL bool-tree node-probability vector into the
/// 9-entry Huffman leaf-probability vector. Direct port of
/// `ZRLBoolTreeToHuffProbs` (spec page 79). The 9 leaves are:
///
/// * `[0..=7]` — runs 1..=8.
/// * `[8]` — escape (run > 8 → followed by raw 6-bit `run - 9`).
pub fn zrl_bool_tree_to_huff_probs(node_prob: &[u8; 14]) -> [u32; 9] {
    let np: [u32; 14] = std::array::from_fn(|i| node_prob[i] as u32);
    let mut hp = [0u32; 9];

    let prob_a = (np[0] * np[1]) >> 8;
    hp[0] = (prob_a * np[2]) >> 8;
    hp[1] = (prob_a * (255 - np[2])) >> 8;

    let prob_b = (np[0] * (255 - np[1])) >> 8;
    hp[2] = (prob_b * np[3]) >> 8;
    hp[3] = (prob_b * (255 - np[3])) >> 8;

    let prob_c = ((255 - np[0]) * np[4]) >> 8;
    let prob_c1 = (prob_c * np[5]) >> 8;
    hp[4] = (prob_c1 * np[6]) >> 8;
    hp[5] = (prob_c1 * (255 - np[6])) >> 8;

    let prob_c2 = (prob_c * (255 - np[5])) >> 8;
    hp[6] = (prob_c2 * np[7]) >> 8;
    hp[7] = (prob_c2 * (255 - np[7])) >> 8;

    hp[8] = ((255 - np[0]) * (255 - np[4])) >> 8;
    hp
}

// =====================================================================
// Tree set per frame
// =====================================================================

/// All Huffman trees needed to decode one frame's coefficients. Built
/// once per frame after the bool-coded probability-update pass.
#[derive(Debug)]
pub struct HuffmanTreeSet {
    /// `[plane=Y/UV]` — DC token tree (12 leaves).
    pub dc: [HuffTree; 2],
    /// `[prec][plane][band]` — AC token tree (12 leaves). `prec` is the
    /// previous-coefficient context (0 = was 0, 1 = was 1, 2 = was > 1);
    /// `band` is `AC_HUFF_BAND[encoded_coeffs]` (0..=3).
    pub ac: [[[HuffTree; 4]; 2]; 3],
    /// `[band]` — ZRL tree (9 leaves; runs 1..=8 + escape).
    pub zrl: [HuffTree; 2],
}

impl HuffmanTreeSet {
    /// Build the tree set for the current model state. Must be called
    /// once per frame (after the bool-coded picture-header probability
    /// updates).
    pub fn build(model: &Vp6Model) -> Self {
        let dc_y_probs = dct_token_bool_tree_to_huff_probs(&model.coeff_dccv[0]);
        let dc_uv_probs = dct_token_bool_tree_to_huff_probs(&model.coeff_dccv[1]);
        let dc = [HuffTree::build(&dc_y_probs), HuffTree::build(&dc_uv_probs)];

        // AC: per spec page 73, only the first 4 bands of `AcProbs`
        // (out of 6) feed into the Huffman trees; the remaining 2 bands
        // (positions 22+) are not used by the Huffman path, so for
        // `band ∈ 0..=3` we use `model.coeff_ract[plane][prec][band]`.
        let ac = std::array::from_fn(|prec| {
            std::array::from_fn(|plane| {
                std::array::from_fn(|band| {
                    let probs =
                        dct_token_bool_tree_to_huff_probs(&model.coeff_ract[plane][prec][band]);
                    HuffTree::build(&probs)
                })
            })
        });

        let zrl_b0 = zrl_bool_tree_to_huff_probs(&model.coeff_runv[0]);
        let zrl_b1 = zrl_bool_tree_to_huff_probs(&model.coeff_runv[1]);
        let zrl = [HuffTree::build(&zrl_b0), HuffTree::build(&zrl_b1)];

        Self { dc, ac, zrl }
    }
}

// =====================================================================
// Per-MB encode + decode
// =====================================================================

/// Per-frame Huffman-coding state. Tracks the cross-block "DC zero run"
/// and "EOB run" counters that span macroblocks within the same plane
/// (spec Section 13.2.2 + 13.4 + Table 19). Reset at the start of every
/// frame.
#[derive(Debug, Default, Clone)]
pub struct HuffmanFrameState {
    /// `[plane]` — number of subsequent blocks that share a DC == 0.
    /// When the DC token of a block is `ZERO_TOKEN`, the decoder reads
    /// a run length and skips the DC of that many subsequent blocks
    /// within the same plane.
    pub dc_run_len: [u32; 2],
    /// `[plane]` — number of subsequent blocks whose every AC
    /// coefficient is 0 (`EOB` at the first AC position of a block ≡
    /// "all AC zero, plus next `N` blocks have all-AC-zero too").
    pub ac1_eob_run_len: [u32; 2],
}

/// Plane index for the cross-block run counters. Y = 0, U = 1, V = 1.
#[inline]
pub fn block_plane_index(b: usize) -> usize {
    if b < 4 {
        0
    } else {
        1
    }
}

/// Encode the AC-1 EOB / DC-0 run length per spec page 81. Uses raw
/// bits per the pseudocode (`R(n)` = raw n-bit MSB-first read in the
/// Huffman path).
///
/// Wire layout (decode side):
/// ```text
/// EOBRunCount = 1 + R(2)
/// if (EOBRunCount == 3) EOBRunCount += R(2)
/// else if (EOBRunCount == 4) {
///     if (R(1)) EOBRunCount = 11 + R(6)
///     else      EOBRunCount = 7 + R(2)
/// }
/// ```
///
/// Encoder is the inverse: pick the smallest representation that can
/// reach `count` and emit the matching raw bits.
pub fn encode_eob_run(bw: &mut BitWriter, count: u32) {
    let count = count.max(1);
    if count <= 2 {
        // Encoded as `1 + R(2)` with R(2) ∈ {0, 1}, value ∈ {1, 2}.
        bw.write_bits(2, count - 1);
    } else if count <= 6 {
        // R(2) == 2 → value starts at 3, plus R(2) ∈ 0..=3.
        bw.write_bits(2, 2);
        bw.write_bits(2, count - 3);
    } else if count <= 10 {
        // R(2) == 3 (initial value 4), R(1) == 0 → 7 + R(2).
        bw.write_bits(2, 3);
        bw.write_bit(0);
        bw.write_bits(2, count - 7);
    } else {
        // R(2) == 3, R(1) == 1 → 11 + R(6); cap at 11 + 63 = 74.
        let count = count.min(11 + 63);
        bw.write_bits(2, 3);
        bw.write_bit(1);
        bw.write_bits(6, count - 11);
    }
}

/// Decode the AC-1 EOB / DC-0 run length. Mirror of
/// [`encode_eob_run`]. Returns the number of subsequent blocks (within
/// the same plane) that the caller should treat as "fully encoded by
/// this run" — the *current* block is always counted as the first of
/// the run (per spec page 81: `EOBRunCount = 1 + R(2)`).
pub fn decode_eob_run(br: &mut BitReader<'_>) -> u32 {
    let mut run = 1 + br.read_bits(2);
    if run == 3 {
        run += br.read_bits(2);
    } else if run == 4 {
        if br.read_bit() != 0 {
            run = 11 + br.read_bits(6);
        } else {
            run = 7 + br.read_bits(2);
        }
    }
    run
}

/// Encode an AC zero-run length using the supplied ZRL Huffman tree
/// (the one that matches the *current* coefficient position's band).
/// Spec page 80: tokens 0..=7 cover runs 1..=8 directly; token 8 is the
/// escape — decoder then reads 6 raw bits and adds 8.
pub fn encode_zero_run(bw: &mut BitWriter, zrl_tree: &HuffTree, run: u32) {
    let run = run.max(1);
    if run <= 8 {
        zrl_tree.encode_symbol(bw, run - 1);
    } else {
        let extra = (run - 8 - 1).min(63);
        zrl_tree.encode_symbol(bw, 8);
        bw.write_bits(6, extra);
    }
}

/// Decode an AC zero-run length using the supplied ZRL tree.
pub fn decode_zero_run(br: &mut BitReader<'_>, zrl_tree: &HuffTree) -> u32 {
    let token = zrl_tree.decode_symbol(br);
    if token < 8 {
        token + 1
    } else {
        // The spec page 80 pseudocode reads `8 + R(6)` for the escape
        // — gives runs 8..=71. We encode the same way (so escape covers
        // 9..=72 since the non-escape branch handles 1..=8); a stream
        // with a "run == 8" preference will use the cheaper non-escape
        // branch.
        8 + br.read_bits(6) + 1
    }
}

/// Encode a single token's value extra-bits + sign bit. Per spec page
/// 57 (Table 18): tokens `ONE_TOKEN`..=`FOUR_TOKEN` carry a sign bit
/// only; `DCT_VAL_CATEGORY*` tokens carry `(extra_bits - 1)` value bits
/// MSB-first then a sign bit; `ZERO_TOKEN` / `DCT_EOB_TOKEN` carry no
/// extra bits at the value level (their extra bits are spent on
/// run-length signalling instead).
pub fn encode_value_extra(bw: &mut BitWriter, token: u8, abs_val: i32, sign_bit: u32) {
    match token {
        ONE_TOKEN | TWO_TOKEN | THREE_TOKEN | FOUR_TOKEN => {
            bw.write_bit(sign_bit);
        }
        DCT_VAL_CATEGORY1 | DCT_VAL_CATEGORY2 | DCT_VAL_CATEGORY3 | DCT_VAL_CATEGORY4
        | DCT_VAL_CATEGORY5 | DCT_VAL_CATEGORY6 => {
            let bias = TOKEN_VALUE_MIN[token as usize];
            let extra = (abs_val - bias).max(0) as u32;
            let value_bits = TOKEN_EXTRA_BITS[token as usize] as u32 - 1;
            bw.write_bits(value_bits, extra);
            bw.write_bit(sign_bit);
        }
        _ => {} // ZERO_TOKEN, DCT_EOB_TOKEN
    }
}

/// Decode a single token's value + sign bits. Returns `(absolute_value,
/// sign_bit)`. For `ZERO_TOKEN` / `DCT_EOB_TOKEN` returns `(0, 0)` and
/// callers handle the run-length implications separately.
pub fn decode_value_extra(br: &mut BitReader<'_>, token: u8) -> (i32, u32) {
    match token {
        ONE_TOKEN | TWO_TOKEN | THREE_TOKEN | FOUR_TOKEN => {
            let abs_val = TOKEN_VALUE_MIN[token as usize];
            let sign = br.read_bit();
            (abs_val, sign)
        }
        DCT_VAL_CATEGORY1 | DCT_VAL_CATEGORY2 | DCT_VAL_CATEGORY3 | DCT_VAL_CATEGORY4
        | DCT_VAL_CATEGORY5 | DCT_VAL_CATEGORY6 => {
            let bias = TOKEN_VALUE_MIN[token as usize];
            let value_bits = TOKEN_EXTRA_BITS[token as usize] as u32 - 1;
            let extra = br.read_bits(value_bits) as i32;
            let sign = br.read_bit();
            (bias + extra, sign)
        }
        _ => (0, 0),
    }
}

/// Pick the smallest token that can encode an absolute coefficient
/// value of `abs_val`. Returns `ZERO_TOKEN` for zero, `DCT_EOB_TOKEN`
/// is not picked here (callers decide EOB vs run separately).
pub fn token_for_abs_value(abs_val: i32) -> u8 {
    let v = abs_val.max(0);
    if v == 0 {
        return ZERO_TOKEN;
    }
    // Walk tokens 1..=10 in min-value order; pick the first whose
    // [min, range) contains `v`.
    for t in 1..=10u8 {
        let lo = TOKEN_VALUE_MIN[t as usize];
        let hi = TOKEN_VALUE_RANGE[t as usize];
        if v >= lo && v < hi {
            return t;
        }
    }
    // Saturate at category 6 — values >= 2115 are clipped to the max
    // representable.
    DCT_VAL_CATEGORY6
}

/// Encode one block's worth of Huffman coefficients into `bw`.
///
/// `coeffs[0]` is the predictor-adjusted DC level; `coeffs[1..=63]` are
/// the quantised AC levels in encoded-coefficient order (i.e. the same
/// `coeff_idx` ordering the bool-coded path uses).
///
/// `dc_zero_run_pending` and `ac1_eob_run_pending` (mutable references)
/// are the cross-block counters held in [`HuffmanFrameState`]; on entry
/// they're inspected and on exit they're updated with any new runs the
/// encoder issued for this block. The function returns whether DC and
/// AC1 were "consumed" by an in-progress run (in which case the
/// corresponding token isn't emitted at all — see spec Table 19).
///
/// This function **does not** decide when to *start* a new run — that's
/// the caller's responsibility (typically the per-MB driver scans
/// ahead, decides "the next K Y blocks are all-DC-zero", emits the
/// run on the first block, then calls into here with `dc_run_pending =
/// K` for blocks 2..=K+1).
pub fn encode_block_huffman(
    bw: &mut BitWriter,
    trees: &HuffmanTreeSet,
    plane: usize,
    coeffs: &[i32; 64],
    dc_zero_run_pending: &mut u32,
    ac1_eob_run_pending: &mut u32,
    queued_dc_run: u32,
    queued_ac1_eob_run: u32,
) {
    // ---- DC --------------------------------------------------------
    let dc_consumed = if *dc_zero_run_pending > 0 {
        *dc_zero_run_pending -= 1;
        true
    } else {
        false
    };
    if !dc_consumed {
        let dc = coeffs[0];
        let abs_dc = dc.unsigned_abs() as i32;
        let token = if dc == 0 {
            ZERO_TOKEN
        } else {
            token_for_abs_value(abs_dc)
        };
        let plane_idx = plane.min(1);
        trees.dc[plane_idx].encode_symbol(bw, token as u32);
        if token == ZERO_TOKEN {
            // Emit a DC-zero run length covering this block + the
            // upcoming `queued_dc_run` blocks.
            encode_eob_run(bw, queued_dc_run + 1);
            // Caller pre-computed the run; consume the queued portion.
            *dc_zero_run_pending = queued_dc_run;
        } else {
            let sign_bit = if dc < 0 { 1 } else { 0 };
            encode_value_extra(bw, token, abs_dc, sign_bit);
        }
    }

    // ---- AC --------------------------------------------------------
    let ac1_consumed = if *ac1_eob_run_pending > 0 {
        *ac1_eob_run_pending -= 1;
        true
    } else {
        false
    };
    if ac1_consumed {
        return;
    }

    // Determine `last_nz` AC position so we know where to emit EOB.
    let mut last_nz = 0usize;
    for i in (1..64).rev() {
        if coeffs[i] != 0 {
            last_nz = i;
            break;
        }
    }

    if last_nz == 0 {
        // All AC zero. Per spec Table 19: when Huffman encoded, an EOB
        // at the *first* AC position is followed by a run length.
        let plane_idx = plane.min(1);
        let band = AC_HUFF_BAND[1] as usize;
        // `prec` for the first AC token: per spec page 74 pseudocode
        // `if dc==0 prec=0 else if dc==1 prec=1 else prec=2`.
        let prec = match coeffs[0].unsigned_abs() {
            0 => 0usize,
            1 => 1usize,
            _ => 2usize,
        };
        trees.ac[prec][plane_idx][band].encode_symbol(bw, DCT_EOB_TOKEN as u32);
        encode_eob_run(bw, queued_ac1_eob_run + 1);
        *ac1_eob_run_pending = queued_ac1_eob_run;
        return;
    }

    // General AC walk: emit zero-runs + tokens, terminate with EOB
    // unless the last_nz lands at coefficient 63 (in which case
    // position == 63 implicitly terminates per Figure 8 / spec page 22).
    let mut encoded_coeffs = 1usize;
    let mut prev_value: i32 = coeffs[0]; // DC value drives the *first* AC's prec.
    while encoded_coeffs <= 63 {
        // Find next non-zero AC starting at `encoded_coeffs`.
        let mut next_nz = encoded_coeffs;
        while next_nz <= 63 && coeffs[next_nz] == 0 {
            next_nz += 1;
        }
        let plane_idx = plane.min(1);
        let prec = match prev_value.unsigned_abs() {
            0 => 0usize,
            1 => 1usize,
            _ => 2usize,
        };

        if next_nz > 63 {
            // No more non-zero AC after this point. Emit EOB unless
            // the previous emit was already at position 63 (in which
            // case the loop should have exited above).
            let band = AC_HUFF_BAND[encoded_coeffs] as usize;
            trees.ac[prec][plane_idx][band].encode_symbol(bw, DCT_EOB_TOKEN as u32);
            // Mid-block EOB: per spec Table 19 "Anywhere but the first
            // AC Coefficient when Huffman Encoded → just terminate the
            // block, no run length". So no extra bits.
            return;
        }

        // Emit any zero run that precedes the non-zero.
        if next_nz > encoded_coeffs {
            let run = (next_nz - encoded_coeffs) as u32;
            let band = ZRL_BAND[encoded_coeffs] as usize;
            // Token "ZERO_TOKEN" + AC zero-run length on the matching
            // band tree.
            let band_ac = AC_HUFF_BAND[encoded_coeffs] as usize;
            trees.ac[prec][plane_idx][band_ac].encode_symbol(bw, ZERO_TOKEN as u32);
            encode_zero_run(bw, &trees.zrl[band], run);
            encoded_coeffs = next_nz;
            prev_value = 0;
            // Re-loop: now `encoded_coeffs` points at the non-zero,
            // emit it on the next iteration.
            continue;
        }

        // Emit the non-zero token.
        let v = coeffs[encoded_coeffs];
        let abs_v = v.unsigned_abs() as i32;
        let token = token_for_abs_value(abs_v);
        let band_ac = AC_HUFF_BAND[encoded_coeffs] as usize;
        trees.ac[prec][plane_idx][band_ac].encode_symbol(bw, token as u32);
        let sign_bit = if v < 0 { 1 } else { 0 };
        encode_value_extra(bw, token, abs_v, sign_bit);
        prev_value = v;
        encoded_coeffs += 1;
        if encoded_coeffs > 63 {
            break;
        }
        if encoded_coeffs > last_nz {
            // Wrap up — next iteration will emit EOB.
            // (Loop continues; the `next_nz > 63` branch above fires.)
        }
    }
}

/// Decode one block's worth of Huffman coefficients from `br` into
/// `out[1..=63]` (AC) and `out[0]` (DC). The DC value is the
/// predictor-adjusted level (caller adds the predictor + dequantises).
///
/// `dc_zero_run_pending` / `ac1_eob_run_pending` are inspected and
/// updated to track cross-block runs (mirror of [`encode_block_huffman`]).
///
/// Returns `Err(Error::invalid)` only on detectably-malformed streams
/// (token table out of range, etc.); transient bit-end is absorbed by
/// [`BitReader::read_bit`] returning 0, which terminates the per-block
/// walk via the EOB / position-63 conditions.
pub fn decode_block_huffman(
    br: &mut BitReader<'_>,
    trees: &HuffmanTreeSet,
    plane: usize,
    out: &mut [i32; 64],
    dc_zero_run_pending: &mut u32,
    ac1_eob_run_pending: &mut u32,
) -> Result<()> {
    for c in out.iter_mut() {
        *c = 0;
    }

    let plane_idx = plane.min(1);

    // ---- DC --------------------------------------------------------
    let mut dc_value: i32 = 0;
    if *dc_zero_run_pending > 0 {
        *dc_zero_run_pending -= 1;
    } else {
        let token = trees.dc[plane_idx].decode_symbol(br) as u8;
        if token >= TOKEN_COUNT as u8 {
            return Err(Error::invalid("VP6 huffman: DC token out of range"));
        }
        if token == DCT_EOB_TOKEN {
            return Err(Error::invalid("VP6 huffman: EOB at DC (forbidden)"));
        }
        if token == ZERO_TOKEN {
            // Followed by a DC-zero run length: this block + N more
            // blocks have DC = 0.
            let run = decode_eob_run(br);
            *dc_zero_run_pending = run.saturating_sub(1);
        } else {
            let (abs_v, sign) = decode_value_extra(br, token);
            dc_value = if sign != 0 { -abs_v } else { abs_v };
        }
    }
    out[0] = dc_value;

    // ---- AC --------------------------------------------------------
    if *ac1_eob_run_pending > 0 {
        *ac1_eob_run_pending -= 1;
        return Ok(());
    }

    let mut encoded_coeffs = 1usize;
    let mut prev_value = dc_value;
    while encoded_coeffs <= 63 {
        let band = AC_HUFF_BAND[encoded_coeffs] as usize;
        let prec = match prev_value.unsigned_abs() {
            0 => 0usize,
            1 => 1usize,
            _ => 2usize,
        };
        let token = trees.ac[prec][plane_idx][band].decode_symbol(br) as u8;
        if token >= TOKEN_COUNT as u8 {
            return Err(Error::invalid("VP6 huffman: AC token out of range"));
        }
        if token == DCT_EOB_TOKEN {
            if encoded_coeffs == 1 {
                let run = decode_eob_run(br);
                *ac1_eob_run_pending = run.saturating_sub(1);
            }
            // Either way, this block's AC are all zero from here on.
            return Ok(());
        }
        if token == ZERO_TOKEN {
            let zband = ZRL_BAND[encoded_coeffs] as usize;
            let run = decode_zero_run(br, &trees.zrl[zband]);
            encoded_coeffs += run as usize;
            prev_value = 0;
            if br.end_reached() {
                return Ok(());
            }
            continue;
        }
        let (abs_v, sign) = decode_value_extra(br, token);
        let v = if sign != 0 { -abs_v } else { abs_v };
        if encoded_coeffs <= 63 {
            out[encoded_coeffs] = v;
        }
        prev_value = v;
        encoded_coeffs += 1;
        if br.end_reached() {
            return Ok(());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_io_roundtrip_msb_first() {
        let mut bw = BitWriter::new();
        bw.write_bits(3, 0b101);
        bw.write_bits(5, 0b11010);
        bw.write_bit(1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(3), 0b101);
        assert_eq!(br.read_bits(5), 0b11010);
        assert_eq!(br.read_bit(), 1);
    }

    #[test]
    fn huff_tree_uniform_probs_roundtrip() {
        // Uniform-ish probabilities → balanced tree; every symbol is
        // reachable.
        let probs = vec![16u32; 12];
        let tree = HuffTree::build(&probs);
        for s in 0..12u32 {
            let mut bw = BitWriter::new();
            tree.encode_symbol(&mut bw, s);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let dec = tree.decode_symbol(&mut br);
            assert_eq!(dec, s, "uniform-probs symbol {s} round-trip");
        }
    }

    #[test]
    fn huff_tree_skewed_probs_roundtrip_all_symbols() {
        let mut probs = [1u32; 12];
        probs[0] = 1000;
        probs[1] = 100;
        probs[5] = 50;
        probs[11] = 5;
        let tree = HuffTree::build(&probs);
        for s in 0..12u32 {
            let mut bw = BitWriter::new();
            tree.encode_symbol(&mut bw, s);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let dec = tree.decode_symbol(&mut br);
            assert_eq!(dec, s);
        }
        // The most-probable symbol should have the shortest codeword.
        let (_, len_p0) = tree.codes[0];
        let (_, len_p11) = tree.codes[11];
        assert!(len_p0 <= len_p11);
    }

    #[test]
    fn eob_run_roundtrip_full_range() {
        for count in 1u32..=74 {
            let mut bw = BitWriter::new();
            encode_eob_run(&mut bw, count);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = decode_eob_run(&mut br);
            assert_eq!(got, count, "eob-run roundtrip count={count}");
        }
    }

    #[test]
    fn zero_run_roundtrip() {
        let probs = [16u32; 9];
        let tree = HuffTree::build(&probs);
        for run in 1u32..=72 {
            let mut bw = BitWriter::new();
            encode_zero_run(&mut bw, &tree, run);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = decode_zero_run(&mut br, &tree);
            assert_eq!(got, run, "zero-run roundtrip run={run}");
        }
    }

    #[test]
    fn token_for_abs_value_categories() {
        assert_eq!(token_for_abs_value(0), ZERO_TOKEN);
        assert_eq!(token_for_abs_value(1), ONE_TOKEN);
        assert_eq!(token_for_abs_value(4), FOUR_TOKEN);
        assert_eq!(token_for_abs_value(5), DCT_VAL_CATEGORY1);
        assert_eq!(token_for_abs_value(6), DCT_VAL_CATEGORY1);
        assert_eq!(token_for_abs_value(7), DCT_VAL_CATEGORY2);
        assert_eq!(token_for_abs_value(11), DCT_VAL_CATEGORY3);
        assert_eq!(token_for_abs_value(67), DCT_VAL_CATEGORY6);
        assert_eq!(token_for_abs_value(2114), DCT_VAL_CATEGORY6);
    }

    #[test]
    fn block_roundtrip_zero_block() {
        let model = Vp6Model {
            coeff_dccv: [[128u8; 11]; 2],
            coeff_ract: [[[[128u8; 11]; 6]; 3]; 2],
            coeff_runv: [[128u8; 14]; 2],
            ..Vp6Model::default()
        };
        let trees = HuffmanTreeSet::build(&model);
        let coeffs = [0i32; 64];
        let mut bw = BitWriter::new();
        let mut dc_run = 0u32;
        let mut eob_run = 0u32;
        encode_block_huffman(&mut bw, &trees, 0, &coeffs, &mut dc_run, &mut eob_run, 0, 0);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut out = [0i32; 64];
        let mut d_dc = 0u32;
        let mut d_eob = 0u32;
        decode_block_huffman(&mut br, &trees, 0, &mut out, &mut d_dc, &mut d_eob).unwrap();
        assert_eq!(out, coeffs);
    }

    #[test]
    fn block_roundtrip_dc_only() {
        let model = Vp6Model {
            coeff_dccv: [[128u8; 11]; 2],
            coeff_ract: [[[[128u8; 11]; 6]; 3]; 2],
            coeff_runv: [[128u8; 14]; 2],
            ..Vp6Model::default()
        };
        let trees = HuffmanTreeSet::build(&model);
        let mut coeffs = [0i32; 64];
        coeffs[0] = -42;
        let mut bw = BitWriter::new();
        let mut dc_run = 0u32;
        let mut eob_run = 0u32;
        encode_block_huffman(&mut bw, &trees, 0, &coeffs, &mut dc_run, &mut eob_run, 0, 0);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut out = [0i32; 64];
        let mut d_dc = 0u32;
        let mut d_eob = 0u32;
        decode_block_huffman(&mut br, &trees, 0, &mut out, &mut d_dc, &mut d_eob).unwrap();
        assert_eq!(out, coeffs);
    }

    #[test]
    fn block_roundtrip_general() {
        let model = Vp6Model {
            coeff_dccv: [[128u8; 11]; 2],
            coeff_ract: [[[[128u8; 11]; 6]; 3]; 2],
            coeff_runv: [[128u8; 14]; 2],
            ..Vp6Model::default()
        };
        let trees = HuffmanTreeSet::build(&model);
        let mut coeffs = [0i32; 64];
        coeffs[0] = 17;
        coeffs[1] = -3;
        coeffs[2] = 1;
        coeffs[5] = 2;
        coeffs[20] = -8;
        coeffs[40] = 1;
        coeffs[63] = -100;
        let mut bw = BitWriter::new();
        let mut dc_run = 0u32;
        let mut eob_run = 0u32;
        encode_block_huffman(&mut bw, &trees, 1, &coeffs, &mut dc_run, &mut eob_run, 0, 0);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut out = [0i32; 64];
        let mut d_dc = 0u32;
        let mut d_eob = 0u32;
        decode_block_huffman(&mut br, &trees, 1, &mut out, &mut d_dc, &mut d_eob).unwrap();
        assert_eq!(out, coeffs);
    }

    #[test]
    fn dc_zero_run_crosses_blocks() {
        let model = Vp6Model {
            coeff_dccv: [[128u8; 11]; 2],
            coeff_ract: [[[[128u8; 11]; 6]; 3]; 2],
            coeff_runv: [[128u8; 14]; 2],
            ..Vp6Model::default()
        };
        let trees = HuffmanTreeSet::build(&model);

        let coeffs = [0i32; 64];
        let mut bw = BitWriter::new();
        let mut dc_run = 0u32;
        let mut eob_run = 0u32;
        // First block emits a DC-zero run covering itself + 3 more
        // blocks; subsequent encode calls see `dc_run_pending > 0` and
        // skip emitting any DC.
        let queued = 3u32;
        encode_block_huffman(
            &mut bw,
            &trees,
            0,
            &coeffs,
            &mut dc_run,
            &mut eob_run,
            queued,
            0,
        );
        for _ in 0..queued {
            encode_block_huffman(&mut bw, &trees, 0, &coeffs, &mut dc_run, &mut eob_run, 0, 0);
        }
        // After the run is done, the next block emits a fresh DC.
        let mut nonzero_dc = [0i32; 64];
        nonzero_dc[0] = 5;
        encode_block_huffman(
            &mut bw,
            &trees,
            0,
            &nonzero_dc,
            &mut dc_run,
            &mut eob_run,
            0,
            0,
        );
        let bytes = bw.finish();

        // Decode mirror.
        let mut br = BitReader::new(&bytes);
        let mut d_dc = 0u32;
        let mut d_eob = 0u32;
        for i in 0..(queued as usize + 1) {
            let mut out = [0i32; 64];
            decode_block_huffman(&mut br, &trees, 0, &mut out, &mut d_dc, &mut d_eob).unwrap();
            assert_eq!(out[0], 0, "block {i} DC zero");
        }
        let mut last = [0i32; 64];
        decode_block_huffman(&mut br, &trees, 0, &mut last, &mut d_dc, &mut d_eob).unwrap();
        assert_eq!(last[0], 5);
    }
}

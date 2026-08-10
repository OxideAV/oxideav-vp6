//! VP6 Huffman tree construction and traversal (spec §7.2).
//!
//! VP6 supports **two** entropy-coding strategies (spec §7): the
//! BoolCoder (§7.3) — a binary arithmetic coder used in the first
//! data-partition for mode/MV decisions, and the **Huffman coder**
//! (§7.2) — a conventional whole-bit binary-tree decoder used as an
//! alternate scheme for DCT-token decoding when the frame header's
//! `UseHuffman` flag is set (§5 / §6).
//!
//! Unlike the BoolCoder, the Huffman coder reads one *whole* bit per
//! tree branch (0 → left, 1 → right) directly from the bitstream — the
//! spec's `R(1)` raw-bit operator (§3 Nomenclature). It does **not**
//! call `VP6_DecodeBool`, so it is **independent of the §7.3 `Split`
//! formula DOCS-GAP** documented in the crate-root docs.
//!
//! This module surfaces the spec's §7.2 Huffman primitives:
//!
//! * [`HuffNode`] — the spec's `HUFF_NODE { Symbol, Prob, Left, Right }`
//!   struct, with `Symbol == -1` denoting an internal (merged) node and
//!   `(Left, Right) == (-1, -1)` denoting a leaf, per §7.2.1.
//! * [`create_huffman_tree`] — the §7.2.1 `VP6_CreateHuffmanTree`
//!   algorithm under the **operative tie-break** the staged errata pins
//!   (see below). Given `N` symbol identifiers and their probabilities
//!   (`1..=255`), it returns a `[HuffNode; 2N-1]` node list with the
//!   root at index `2*N - 2`.
//! * [`decode_symbol`] — the verbatim §7.2 `VP6_HuffmanDecodeSymbol`
//!   walk. Driven by an externally-supplied raw-bit oracle (the caller
//!   provides the `R(1)` source), so this module ships the *traversal
//!   step* without coupling to any specific bitstream reader. The
//!   crate-level `R(1)` source for VP6 is a single byte-stream bit
//!   reader (§3); plugging it into this traversal needs no §7.3 work.
//! * [`tree_depth`] / [`codeword_for`] — convenience helpers that walk
//!   the constructed tree to compute a symbol's codeword length and
//!   bit pattern by tracing root-to-leaf. Useful for round-tripping
//!   and for inspecting the tie-break-sensitive tree shape the §13.1 /
//!   §13.3.3.2 conversions rely on.
//!
//! ## The operative §7.2.1 construction (errata `#277 part 3, closed`)
//!
//! The printed §7.2.1 listing fixes ties only *relatively* ("maintaining
//! the relative order of symbols having equal probabilities") and never
//! states the initial symbol order, so two conforming readings of the
//! text produce two different codebooks whenever probabilities tie —
//! which they do heavily at the §13 keyframe defaults. The staged
//! errata (`docs/video/vp6/vp6-errata-and-clarifications.md`, "#277
//! (part 3, closed)", arbitrated against the conformant fixture datum
//! in `docs/video/vp6/tables/03-first-content-block.csv`) pins the
//! operative construction:
//!
//! 1. A single list of nodes is maintained in **ascending** weight
//!    order.
//! 2. Symbols are offered in **index order** (`S[i] = i`, the Table 18
//!    token order for the §13 alphabets), each inserted **before the
//!    first node whose weight is greater than or equal to its own** —
//!    so a group of equal-weight symbols ends up in **descending
//!    symbol-index order**, not insertion order.
//! 3. Each merge round takes the two head nodes (the two smallest
//!    weights): the **first** becomes the **left** (bit-0) child, the
//!    **second** the right (bit-1) child. The merged node re-enters the
//!    list by the same insert-before-greater-or-equal rule, so it also
//!    precedes equal-weight nodes.
//!
//! This is *not* the stable ascending sort the printed wording
//! suggests; the stable-sort reading assigns `DCT_VAL_CATEGORY1` and
//! `DCT_VAL_CATEGORY2` (tied at the §13 defaults) the opposite
//! codewords and fails the fixture (the errata's measured
//! `DCT_VAL_CATEGORY2 = 010` requirement).
//!
//! ## What this module does NOT land
//!
//! The §13.1 conversion `DCTTokenBoolTreeToHuffProbs` (already landed
//! in [`crate::tokens::dct_token_bool_tree_to_huff_probs`]) produces
//! the 12-entry symbol-probability vector this builder consumes for
//! the §13 DCT-token Huffman tree. The §13.3.3.2 conversion that
//! covers the AC zero-run Huffman tree is a separate transform and
//! is not in scope for this round. The actual `R(1)` byte-stream
//! reader is not in scope either — this module exposes the traversal
//! step parameterised over any `FnMut() -> u8` source so the bit
//! reader can land independently.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §7.2 (On2
//! Technologies, document version 1.02, August 2006) and the staged
//! clean-room errata `docs/video/vp6/vp6-errata-and-clarifications.md`
//! ("#277 (part 3, closed)"), which settles the tie-break, merge-order
//! and clamp questions the printed §7.2.1 listing leaves free. No
//! third-party VP6 implementation has been consulted.

use core::fmt;

/// Sentinel value the spec uses in [`HuffNode::symbol`] to mean
/// "this node is an internal / merged node, not a leaf."
///
/// §7.2.1: *"A leaf node is represented by a node where Symbol is
/// **not** set to -1 and the Left and Right child indices are both
/// set to -1."*
pub const INTERNAL_SYMBOL: i32 = -1;

/// Sentinel value the spec uses in [`HuffNode::left`] / [`HuffNode::right`]
/// to mean "this side has no child" (i.e. the node is a leaf).
///
/// §7.2.1: *"A leaf node is represented by a node where Symbol is
/// **not** set to -1 and the Left and Right child indices are both
/// set to **-1**."*
pub const NO_CHILD: i32 = -1;

/// One node of the §7.2 Huffman `SortList[2N-1]` array.
///
/// Direct transcription of the spec's `HUFF_NODE` struct (page 13):
///
/// ```text
/// HUFF_NODE
/// {
///     Symbol     // Decoded Symbol for leaf node, -1 for internal node
///     Prob       // Huffman node probability
///     Left       // Index of Left Child in the sort list
///     Right      // Index of Right Child in the sort list
/// }
/// ```
///
/// `Prob` is held as `u32` so the sum-of-children accumulation in
/// `create_huffman_tree` never overflows (the maximum total summed
/// probability across all `N` symbols is bounded by `N * 255`, which
/// fits trivially in `u32` for any plausible `N`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HuffNode {
    /// Symbol identifier for a leaf; [`INTERNAL_SYMBOL`] (`-1`) for
    /// internal nodes. The spec's `Symbol` field.
    pub symbol: i32,
    /// Node probability: leaf-symbol probability for a leaf; sum of
    /// children's probabilities for an internal node. The spec's
    /// `Prob` field.
    pub prob: u32,
    /// Sort-list index of the left child, or [`NO_CHILD`] (`-1`)
    /// for a leaf. The spec's `Left` field. Left == "0-branch" of
    /// the binary tree per §7 ("0 indicates left").
    pub left: i32,
    /// Sort-list index of the right child, or [`NO_CHILD`] (`-1`)
    /// for a leaf. The spec's `Right` field. Right == "1-branch"
    /// per §7 ("1 indicates right").
    pub right: i32,
}

impl HuffNode {
    /// `true` iff this is a leaf (i.e. carries a decoded symbol).
    ///
    /// Per §7.2.1 a leaf is identified by `Symbol != -1` and
    /// `Left == Right == -1`; the equivalent inverse (an internal
    /// node always has `Symbol == -1`) is what the §7.2
    /// `VP6_HuffmanDecodeSymbol` `while` condition tests.
    #[inline]
    pub const fn is_leaf(self) -> bool {
        self.symbol != INTERNAL_SYMBOL
    }

    /// Construct an unmerged leaf node for symbol `s` with probability
    /// `p`. Per §7.2.1 leaf nodes have `Left = Right = -1`.
    #[inline]
    pub const fn leaf(s: i32, p: u32) -> Self {
        Self {
            symbol: s,
            prob: p,
            left: NO_CHILD,
            right: NO_CHILD,
        }
    }
}

/// Build a Huffman tree from `N` symbols and their probabilities, per
/// the §7.2.1 `VP6_CreateHuffmanTree` algorithm under the operative
/// tie-break the staged errata pins ("#277 (part 3, closed)").
///
/// `symbols[i]` is the spec's `S[i]` (the symbol identifier — for the
/// VP6 DCT tree this is a [`crate::tokens::DctToken`] index 0..=11);
/// `probs[i]` is the spec's `P[i]` (the per-symbol leaf probability,
/// in the §7 valid range `1..=255`). The two slices must be the same
/// length, which is `N`. `N >= 2` (a one-symbol tree is degenerate;
/// at least one merge round is needed to produce a root).
///
/// The construction maintains a single ascending-weight list. Symbols
/// enter in slice order, each inserted **before the first node whose
/// weight is greater than or equal to its own** (so equal-weight
/// symbols end in reverse insertion order). Each of the `N - 1` merge
/// rounds removes the two head nodes — the first becomes the left
/// (bit-0) child, the second the right (bit-1) child — and re-inserts
/// the merged node by the same before-greater-or-equal rule.
///
/// The returned `Vec<HuffNode>` has length `2 * N - 1` exactly: `N`
/// leaves (in input slice order) followed by `N - 1` internal merge
/// nodes in merge order, with the **root at index `2 * N - 2`** as the
/// spec's terminating comment states (*"Huffman tree root node is at
/// position 2\*N-2 in SortList"*).
///
/// The build is pure integer arithmetic and reads no bits from any
/// bitstream.
///
/// # Errors
///
/// Returns [`HuffmanError::TooFewSymbols`] if `symbols.len() < 2`,
/// [`HuffmanError::LengthMismatch`] if `symbols` and `probs` differ
/// in length, [`HuffmanError::InvalidProbability`] if any probability
/// is zero (§7: *"the value 0 is explicitly forbidden, so the valid
/// range is `1 <= Node Probability <= 255`"*; the §13 callers clamp
/// converted zero leaf-weights to 1 before calling, per the same
/// errata entry's clamp rule).
pub fn create_huffman_tree(symbols: &[i32], probs: &[u8]) -> Result<Vec<HuffNode>, HuffmanError> {
    if symbols.len() != probs.len() {
        return Err(HuffmanError::LengthMismatch);
    }
    let n = symbols.len();
    if n < 2 {
        return Err(HuffmanError::TooFewSymbols);
    }
    for &p in probs {
        if p == 0 {
            return Err(HuffmanError::InvalidProbability);
        }
    }

    // Node arena: leaves 0..N in input order, then the N-1 merge nodes
    // appended in merge order (root last, at 2N-2).
    let mut arena: Vec<HuffNode> = Vec::with_capacity(2 * n - 1);
    for (&s, &p) in symbols.iter().zip(probs.iter()) {
        arena.push(HuffNode::leaf(s, p as u32));
    }

    // The ascending-weight working list, holding arena indices.
    // Insertion rule (errata "#277 (part 3, closed)"): a new node goes
    // immediately before the first node whose weight is greater than
    // OR EQUAL to its own — it jumps ahead of its equals.
    let mut list: Vec<usize> = Vec::with_capacity(n);
    let insert = |list: &mut Vec<usize>, arena: &[HuffNode], idx: usize| {
        let w = arena[idx].prob;
        let pos = list
            .iter()
            .position(|&e| arena[e].prob >= w)
            .unwrap_or(list.len());
        list.insert(pos, idx);
    };
    for i in 0..n {
        insert(&mut list, &arena, i);
    }

    // N-1 merge rounds: the two head nodes (smallest weights) merge;
    // the first (smaller) is the left / bit-0 child, the second the
    // right / bit-1 child. The merged node re-enters the list by the
    // same before-greater-or-equal rule.
    for _ in 0..(n - 1) {
        let l = list.remove(0);
        let r = list.remove(0);
        let merged = HuffNode {
            symbol: INTERNAL_SYMBOL,
            prob: arena[l].prob + arena[r].prob,
            left: l as i32,
            right: r as i32,
        };
        let idx = arena.len();
        arena.push(merged);
        insert(&mut list, &arena, idx);
    }
    debug_assert_eq!(list.len(), 1);
    debug_assert_eq!(list[0], 2 * n - 2, "root is the final merge node");

    Ok(arena)
}

/// Decode one symbol from a constructed Huffman tree, per the §7.2
/// `VP6_HuffmanDecodeSymbol` listing.
///
/// `tree` is the sort-list returned by [`create_huffman_tree`]: a
/// `[HuffNode; 2N-1]` with the root at index `2 * N - 2 = tree.len() - 1`.
/// `n` is the spec's `N` (the symbol count); equivalently
/// `n == (tree.len() + 1) / 2`. The traversal starts at the root and
/// at each internal node consults `r1()` for the next raw bit: a
/// `0` selects [`HuffNode::left`], a `1` selects [`HuffNode::right`]
/// (§7.2: *"0 indicates left, 1 indicates right"*).
///
/// `r1` is the spec's `R(1)` source — a closure returning the next
/// raw bit from the bitstream as `0` or `1`. The Huffman coder is
/// orthogonal to the §7.3 BoolCoder; any raw-bit reader will do. The
/// closure is invoked exactly `tree_depth(tree, symbol)` times per
/// call.
///
/// Returns the decoded symbol identifier (the spec's `DecodedSymbol`),
/// i.e. the `Symbol` field of the leaf the walk arrives at.
///
/// # Panics
///
/// Panics if `tree` is empty or malformed (no leaf reachable). A
/// well-formed tree from [`create_huffman_tree`] cannot trigger this.
pub fn decode_symbol<F: FnMut() -> u8>(tree: &[HuffNode], mut r1: F) -> i32 {
    assert!(
        !tree.is_empty(),
        "decode_symbol: empty Huffman tree (build with create_huffman_tree first)"
    );
    // §7.2 listing: `NextNode = 2*N-2 // Root node`. With a sort-list
    // produced by `create_huffman_tree` that is exactly the last index.
    let mut next_node = tree.len() - 1;
    while tree[next_node].symbol == INTERNAL_SYMBOL {
        let bit = r1();
        let node = &tree[next_node];
        let child = if bit == 0 { node.left } else { node.right };
        debug_assert!(
            child >= 0,
            "decode_symbol: internal node {next_node} has -1 child on bit {bit}"
        );
        next_node = child as usize;
    }
    tree[next_node].symbol
}

/// Length of the codeword for `symbol` in this tree (number of edges
/// from the root to the leaf carrying that symbol).
///
/// Returns `None` if no leaf with that symbol exists. Useful for
/// inspecting whether the §7.2.1 stable sort produced the
/// tree-shape the encoder also produces from the same probabilities.
pub fn tree_depth(tree: &[HuffNode], symbol: i32) -> Option<usize> {
    codeword_for(tree, symbol).map(|(_pattern, len)| len)
}

/// Codeword for `symbol`: the bit-pattern and its length, with the
/// MSB being the first bit emitted (i.e. the root-most edge taken).
///
/// Returns `None` if no leaf with that symbol exists.
///
/// The pattern is packed in the low `len` bits of the `u32`; for a
/// codeword of length `L`, bit `L-1` is the first bit consumed during
/// decode and bit `0` is the last. With `len == 0` (the degenerate
/// one-symbol tree) the pattern is `0`. The maximum length supported
/// is 32 bits (any tree built from a Huffman alphabet with at most
/// `N <= u32::MAX/2 + 1` symbols), which is well beyond what any §13
/// VP6 alphabet (≤ 12 tokens) can ever produce.
pub fn codeword_for(tree: &[HuffNode], symbol: i32) -> Option<(u32, usize)> {
    if tree.is_empty() {
        return None;
    }
    let root = tree.len() - 1;
    let mut buf = [0u8; 32];
    walk(tree, root, symbol, &mut buf, 0).map(|len| {
        let mut pattern = 0u32;
        for &bit in &buf[..len] {
            pattern = (pattern << 1) | bit as u32;
        }
        (pattern, len)
    })
}

fn walk(
    tree: &[HuffNode],
    node: usize,
    target: i32,
    buf: &mut [u8; 32],
    depth: usize,
) -> Option<usize> {
    let n = tree[node];
    if n.is_leaf() {
        if n.symbol == target {
            return Some(depth);
        }
        return None;
    }
    if depth >= buf.len() {
        // Tree too deep for the 32-bit accumulator. With any plausible
        // VP6 alphabet this is unreachable, but bail rather than
        // overflow.
        return None;
    }
    if n.left >= 0 {
        buf[depth] = 0;
        if let Some(d) = walk(tree, n.left as usize, target, buf, depth + 1) {
            return Some(d);
        }
    }
    if n.right >= 0 {
        buf[depth] = 1;
        if let Some(d) = walk(tree, n.right as usize, target, buf, depth + 1) {
            return Some(d);
        }
    }
    None
}

/// Errors that can be returned by [`create_huffman_tree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanError {
    /// Fewer than two symbols were supplied. A meaningful Huffman tree
    /// needs at least one internal node, which needs at least two
    /// children.
    TooFewSymbols,
    /// `symbols.len() != probs.len()` — the spec defines `S[N]` and
    /// `P[N]` as parallel arrays.
    LengthMismatch,
    /// A probability of zero was supplied. §7: *"the value 0 is
    /// explicitly forbidden, so the valid range is
    /// `1 <= Node Probability <= 255`."*
    InvalidProbability,
}

impl fmt::Display for HuffmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSymbols => f.write_str("Huffman tree needs at least 2 symbols"),
            Self::LengthMismatch => {
                f.write_str("symbols and probabilities must be the same length")
            }
            Self::InvalidProbability => {
                f.write_str("Huffman node probability 0 is forbidden by spec §7 (valid: 1..=255)")
            }
        }
    }
}

impl std::error::Error for HuffmanError {}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- HuffNode surface --------

    #[test]
    fn leaf_constructor_matches_spec_sentinel_pattern() {
        let n = HuffNode::leaf(7, 42);
        assert_eq!(n.symbol, 7);
        assert_eq!(n.prob, 42);
        assert_eq!(n.left, NO_CHILD);
        assert_eq!(n.right, NO_CHILD);
        assert!(n.is_leaf());
    }

    #[test]
    fn internal_node_is_not_leaf() {
        let n = HuffNode {
            symbol: INTERNAL_SYMBOL,
            prob: 100,
            left: 0,
            right: 1,
        };
        assert!(!n.is_leaf());
    }

    // -------- create_huffman_tree input validation --------

    #[test]
    fn create_rejects_zero_probability() {
        let err = create_huffman_tree(&[0, 1, 2], &[1, 0, 1]).unwrap_err();
        assert_eq!(err, HuffmanError::InvalidProbability);
    }

    #[test]
    fn create_rejects_mismatched_lengths() {
        let err = create_huffman_tree(&[0, 1], &[1, 2, 3]).unwrap_err();
        assert_eq!(err, HuffmanError::LengthMismatch);
    }

    #[test]
    fn create_rejects_fewer_than_two_symbols() {
        let err = create_huffman_tree(&[7], &[1]).unwrap_err();
        assert_eq!(err, HuffmanError::TooFewSymbols);
        let err = create_huffman_tree(&[], &[]).unwrap_err();
        assert_eq!(err, HuffmanError::TooFewSymbols);
    }

    // -------- create_huffman_tree shape invariants --------

    #[test]
    fn create_two_symbol_tree_has_expected_geometry() {
        // N = 2 → node list of length 2N-1 = 3, root at index 2. The
        // two symbols tie, so the later-inserted one (arena index 1)
        // jumps ahead of its equal and becomes the left / bit-0 child.
        let tree = create_huffman_tree(&[10, 20], &[1, 1]).unwrap();
        assert_eq!(tree.len(), 3);
        // Root is the last entry.
        let root = tree.last().unwrap();
        assert!(!root.is_leaf());
        assert_eq!(root.left, 1, "later equal jumps ahead → left child");
        assert_eq!(root.right, 0);
        assert_eq!(root.prob, 2);
        // Children are leaves.
        assert!(tree[0].is_leaf());
        assert!(tree[1].is_leaf());
    }

    #[test]
    fn create_produces_exact_2n_minus_1_nodes() {
        for n in 2..=12usize {
            let symbols: Vec<i32> = (0..n as i32).collect();
            let probs: Vec<u8> = (0..n).map(|i| ((i + 1) * 7 % 200 + 1) as u8).collect();
            let tree = create_huffman_tree(&symbols, &probs).unwrap();
            assert_eq!(tree.len(), 2 * n - 1, "N={n}");
            // Root invariants.
            let root = tree.last().unwrap();
            assert!(!root.is_leaf(), "N={n}: root must be internal");
            // Internal-node count == N - 1.
            let internal = tree.iter().filter(|n| !n.is_leaf()).count();
            assert_eq!(internal, n - 1, "N={n}: internal-node count");
            // Leaf count == N and every input symbol appears exactly once.
            let mut seen: Vec<i32> = tree
                .iter()
                .filter(|n| n.is_leaf())
                .map(|n| n.symbol)
                .collect();
            seen.sort_unstable();
            let mut expected = symbols.clone();
            expected.sort_unstable();
            assert_eq!(seen, expected, "N={n}: leaf symbols");
        }
    }

    #[test]
    fn root_probability_equals_sum_of_input_probabilities() {
        let probs: [u8; 6] = [3, 11, 7, 19, 5, 23];
        let symbols: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let tree = create_huffman_tree(&symbols, &probs).unwrap();
        let expected_sum: u32 = probs.iter().map(|&p| p as u32).sum();
        assert_eq!(tree.last().unwrap().prob, expected_sum);
    }

    // -------- decode_symbol round-trip --------

    #[test]
    fn decode_symbol_round_trips_two_symbol_tree() {
        let tree = create_huffman_tree(&[100, 200], &[1, 1]).unwrap();
        // The two symbols tie: the later-inserted (200) jumps ahead
        // and becomes the left / bit-0 child (errata #277 part 3).
        let mut bits0 = [0u8].iter().copied();
        assert_eq!(decode_symbol(&tree, || bits0.next().unwrap()), 200);
        let mut bits1 = [1u8].iter().copied();
        assert_eq!(decode_symbol(&tree, || bits1.next().unwrap()), 100);
    }

    #[test]
    fn decode_symbol_round_trips_every_leaf() {
        // Use the §13 DCT-token alphabet's keyframe-baseline shape:
        // 11 baseline probabilities derived from the all-128 BoolCoder
        // node probabilities via DCTTokenBoolTreeToHuffProbs. Picked
        // because it exercises a deeper, irregular tree.
        let baseline_node_probs = [128u8; 11];
        let huff_probs = crate::tokens::dct_token_bool_tree_to_huff_probs(&baseline_node_probs);
        // Filter out any zero leaf-probabilities — §7 forbids zero, and
        // the §13.1 transform can in principle emit zero for unreachable
        // tokens. The all-128 baseline doesn't actually produce any zeros
        // (every right-shift keeps every term at 1+), but be defensive.
        let mut symbols: Vec<i32> = Vec::new();
        let mut probs: Vec<u8> = Vec::new();
        for (i, p) in huff_probs.iter().enumerate() {
            if *p > 0 {
                symbols.push(i as i32);
                probs.push(*p);
            }
        }
        assert!(symbols.len() >= 2);

        let tree = create_huffman_tree(&symbols, &probs).unwrap();
        for &s in &symbols {
            let (pattern, len) = codeword_for(&tree, s).expect("symbol present");
            // Drive decode with the recovered codeword's bits, MSB first.
            let mut remaining = len;
            let mut acc = pattern;
            let bits = core::iter::from_fn(|| {
                if remaining == 0 {
                    None
                } else {
                    remaining -= 1;
                    let bit = ((acc >> remaining) & 1) as u8;
                    acc &= (1u32 << remaining).wrapping_sub(1);
                    Some(bit)
                }
            });
            let mut bits = bits;
            let decoded = decode_symbol(&tree, || bits.next().unwrap());
            assert_eq!(decoded, s, "round-trip for symbol {s}");
        }
    }

    // -------- operative §7.2.1 tie-break (errata #277 part 3, closed) --------

    #[test]
    fn equal_probabilities_merge_in_descending_insertion_order() {
        // Three symbols, all equal probability. Each new equal jumps
        // ahead of the ones already in the list, so the first merge
        // consumes the LAST two inserted symbols (30 as left, 20 as
        // right), and the earliest-inserted symbol survives to pair
        // with the merged node — getting the shortest codeword.
        let tree = create_huffman_tree(&[10, 20, 30], &[5, 5, 5]).unwrap();
        assert_eq!(codeword_for(&tree, 10), Some((0b0, 1)));
        assert_eq!(codeword_for(&tree, 30), Some((0b10, 2)));
        assert_eq!(codeword_for(&tree, 20), Some((0b11, 2)));
        // The arena keeps leaves in input order regardless.
        assert_eq!(tree[0].symbol, 10);
        assert_eq!(tree[1].symbol, 20);
        assert_eq!(tree[2].symbol, 30);
    }

    #[test]
    fn merged_node_precedes_equal_weight_leaves() {
        // Weights [1, 1, 2, 2]: symbols 0 and 1 merge to weight 2,
        // which re-enters BEFORE the two weight-2 leaves. The second
        // merge therefore pairs the merged node with the weight-2
        // leaf ahead of it (symbol 3 — it jumped ahead of symbol 2),
        // leaving symbol 2 the sole depth-1 leaf. (Under the
        // after-equals placement the two weight-2 leaves merge with
        // each other instead and the tree is a balanced depth-2 one;
        // this pins the errata's before-equals rule.)
        let tree = create_huffman_tree(&[0, 1, 2, 3], &[1, 1, 2, 2]).unwrap();
        assert_eq!(tree_depth(&tree, 2), Some(1));
        assert_eq!(tree_depth(&tree, 3), Some(2));
        assert_eq!(tree_depth(&tree, 0), Some(3));
        assert_eq!(tree_depth(&tree, 1), Some(3));
    }

    // -------- tree_depth / codeword_for --------

    #[test]
    fn balanced_four_symbol_tree_yields_uniform_depth_two() {
        // Four equal-probability symbols build a perfectly balanced
        // depth-2 tree: round 1 merges (0+1), round 2 merges (2+3) (the
        // next two lowest probabilities after the first merge increases
        // SortList[3].prob), round 3 merges those two internal nodes.
        // The exact shape depends on the spec's stable sort; what is
        // invariant is the depth of every leaf == 2.
        let tree = create_huffman_tree(&[0, 1, 2, 3], &[1, 1, 1, 1]).unwrap();
        for s in 0..4 {
            assert_eq!(tree_depth(&tree, s), Some(2), "symbol {s}");
        }
    }

    #[test]
    fn skewed_probabilities_produce_shorter_code_for_more_likely_symbol() {
        // Symbol 0 dominates: its leaf should sit closest to the root.
        let tree = create_huffman_tree(&[0, 1, 2, 3, 4], &[200, 10, 10, 10, 10]).unwrap();
        let dominant_depth = tree_depth(&tree, 0).unwrap();
        for s in 1..5 {
            let d = tree_depth(&tree, s).unwrap();
            assert!(
                d >= dominant_depth,
                "rare symbol {s} (depth {d}) should be at least as deep as dominant 0 (depth {dominant_depth})"
            );
        }
    }

    #[test]
    fn codeword_for_returns_none_for_unknown_symbol() {
        let tree = create_huffman_tree(&[5, 6], &[1, 1]).unwrap();
        assert_eq!(codeword_for(&tree, 99), None);
        assert_eq!(tree_depth(&tree, 99), None);
    }

    // -------- HuffmanError surface --------

    #[test]
    fn huffman_error_display_messages() {
        assert_eq!(
            HuffmanError::TooFewSymbols.to_string(),
            "Huffman tree needs at least 2 symbols"
        );
        assert_eq!(
            HuffmanError::LengthMismatch.to_string(),
            "symbols and probabilities must be the same length"
        );
        assert!(HuffmanError::InvalidProbability
            .to_string()
            .contains("1..=255"));
    }
}

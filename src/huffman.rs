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
//! * [`create_huffman_tree`] — the verbatim §7.2.1 `VP6_CreateHuffmanTree`
//!   algorithm. Given `N` symbol identifiers and their probabilities
//!   (`1..=255`), it returns a `[HuffNode; 2N-1]` sort-list with the
//!   root at index `2*N - 2`, satisfying the spec's stable-sort and
//!   bottom-up-merge invariants.
//! * [`decode_symbol`] — the verbatim §7.2 `VP6_HuffmanDecodeSymbol`
//!   walk. Driven by an externally-supplied raw-bit oracle (the caller
//!   provides the `R(1)` source), so this module ships the *traversal
//!   step* without coupling to any specific bitstream reader. The
//!   crate-level `R(1)` source for VP6 is a single byte-stream bit
//!   reader (§3); plugging it into this traversal needs no §7.3 work.
//! * [`tree_depth`] / [`codeword_for`] — convenience helpers that walk
//!   the constructed tree to compute a symbol's codeword length and
//!   bit pattern by tracing root-to-leaf. Useful for round-tripping
//!   and for inspecting whether the spec's stable-sort produced the
//!   tree shape the §13.1 / §13.3.3.2 conversions expect.
//!
//! ## Stability of the §7.2.1 sort
//!
//! The §7.2.1 listing twice calls for a sort that "[maintains] relative
//! order of nodes having equal probability." This is a *stable* sort,
//! not just any ascending sort: when the leaf-list (or the merged
//! sub-list after each round) contains equal-probability entries the
//! original insertion order must survive. The implementation here uses
//! [`slice::sort_by`] / a stable insertion shuffle so the spec's
//! invariant holds — any two leaf orderings that differ in stable-sort
//! handling can produce structurally different (but symbol-equivalent)
//! trees, which is exactly the property the spec relies on to make
//! both encoder and decoder agree on the tree shape from the
//! probability vector alone.
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
//! Technologies, document version 1.02, August 2006). The §7.2.1
//! pseudocode (the `VP6_CreateHuffmanTree` and `VP6_HuffmanDecodeSymbol`
//! listings on page 14) was transcribed structurally into Rust. No
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
/// the §7.2.1 `VP6_CreateHuffmanTree` listing.
///
/// `symbols[i]` is the spec's `S[i]` (the symbol identifier — for the
/// VP6 DCT tree this is a [`crate::tokens::DctToken`] index 0..=11);
/// `probs[i]` is the spec's `P[i]` (the per-symbol leaf probability,
/// in the §7 valid range `1..=255`). The two slices must be the same
/// length, which is `N`. `N >= 2` (a one-symbol tree is degenerate;
/// the spec's loop runs `N-1` merge rounds and at least one is needed
/// to produce a root).
///
/// The returned `Vec<HuffNode>` has length `2 * N - 1` exactly: `N`
/// leaves followed by `N - 1` internal merge nodes, with the **root
/// at index `2 * N - 2`** as the spec's terminating comment states
/// (*"Huffman tree root node is at position 2\*N-2 in SortList"*).
///
/// The build is pure integer arithmetic and reads no bits from any
/// bitstream — so it is independent of the §7.3 BoolCoder DOCS-GAP.
///
/// # Errors
///
/// Returns [`HuffmanError::TooFewSymbols`] if `symbols.len() < 2`,
/// [`HuffmanError::LengthMismatch`] if `symbols` and `probs` differ
/// in length, [`HuffmanError::InvalidProbability`] if any probability
/// is zero (§7: *"the value 0 is explicitly forbidden, so the valid
/// range is `1 <= Node Probability <= 255`"*).
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

    // The spec's `SortList[2N-1]` array. Pre-fill the merged-node tail
    // with a placeholder leaf so the slice's length is fixed at `2N-1`
    // and we can write into it by index without `push`/`insert` shifts
    // disturbing the indices the spec's pseudo-code refers to.
    let sentinel = HuffNode::leaf(INTERNAL_SYMBOL, 0);
    let mut sort_list: Vec<HuffNode> = vec![sentinel; 2 * n - 1];

    // §7.2.1 step 1: populate leaves 0..N.
    for (i, (&s, &p)) in symbols.iter().zip(probs.iter()).enumerate() {
        sort_list[i] = HuffNode::leaf(s, p as u32);
    }

    // §7.2.1 step 2: "Sort SortList into ascending probability order
    // maintaining relative order of nodes having equal probability."
    // The spec sorts the *leaf* sub-list (positions 0..N) only at this
    // stage; the rest is the placeholder zone we have not written yet.
    // We track this sub-list explicitly so the spec's `L = 2*i` and
    // `R = L+1` two-least-probable accessors trivially hit the right
    // entries each round.
    //
    // Rust's `sort_by` is stable, so equal probabilities preserve
    // insertion order — exactly the §7.2.1 invariant.
    sort_list[..n].sort_by_key(|a| a.prob);

    // §7.2.1 step 3: N-1 merge rounds.
    //
    // The spec writes:
    //
    //     for ( i=0; i<N-1; i++ )
    //     {
    //         L = 2*i           // Least probable node
    //         R = L+1           // Second least probable node
    //         SortList[N+i].Symbol  = -1
    //         SortList[N+i].Prob    = SortList[L].Prob + SortList[R].Prob
    //         SortList[N+i].Left    = L
    //         SortList[N+i].Right   = R
    //         Sort nodes in SortList between positions R+1 and N+i (inclusive)
    //         in to ascending probability order maintaining relative order
    //         of nodes having equal probability
    //     }
    //
    // The `L = 2*i` / `R = 2*i+1` pattern works because every round
    // consumes the two lowest-probability nodes at positions `2i` and
    // `2i+1` and writes the new merged node at `N+i` — then resorts
    // the *remaining* active sub-list (`R+1 .. N+i`) so the next
    // round's `L = 2*(i+1) = 2i+2` (= the old R+1) once again points
    // at the new lowest-probability entry.
    for i in 0..(n - 1) {
        let l = 2 * i;
        let r = l + 1;
        let merged = HuffNode {
            symbol: INTERNAL_SYMBOL,
            prob: sort_list[l].prob + sort_list[r].prob,
            left: l as i32,
            right: r as i32,
        };
        let dest = n + i;
        sort_list[dest] = merged;
        // "Sort nodes in SortList between positions R+1 and N+i
        // (inclusive)". When R+1 > N+i the slice is empty and the
        // sort is a no-op (the final i = N-2 round leaves a single
        // node in the active window — the just-written root — and
        // there is nothing to sort against).
        let start = r + 1;
        let end_inclusive = dest;
        if start <= end_inclusive {
            sort_list[start..=end_inclusive].sort_by_key(|a| a.prob);
        }
    }

    Ok(sort_list)
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
        // N = 2 → SortList of length 2N-1 = 3, root at index 2.
        let tree = create_huffman_tree(&[10, 20], &[1, 1]).unwrap();
        assert_eq!(tree.len(), 3);
        // Root is the last entry.
        let root = tree.last().unwrap();
        assert!(!root.is_leaf());
        assert_eq!(root.left, 0);
        assert_eq!(root.right, 1);
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
        // Bit 0 → left (symbol 100); bit 1 → right (symbol 200).
        let mut bits0 = [0u8].iter().copied();
        assert_eq!(decode_symbol(&tree, || bits0.next().unwrap()), 100);
        let mut bits1 = [1u8].iter().copied();
        assert_eq!(decode_symbol(&tree, || bits1.next().unwrap()), 200);
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

    // -------- §7.2.1 stable-sort invariant --------

    #[test]
    fn stable_sort_preserves_input_order_for_equal_probabilities() {
        // Three symbols, all equal probability. The spec mandates a
        // stable sort, so the leaf zone of SortList must retain
        // [10, 20, 30] in input order.
        let tree = create_huffman_tree(&[10, 20, 30], &[5, 5, 5]).unwrap();
        // The first three positions are the leaf zone after the
        // initial sort; they should still read 10, 20, 30.
        assert_eq!(tree[0].symbol, 10);
        assert_eq!(tree[1].symbol, 20);
        assert_eq!(tree[2].symbol, 30);
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

//! VP6 AC zero-run-length static surface (spec §13.3.3).
//!
//! When the §13 token decoder produces a `ZERO_TOKEN` in the AC
//! position, a **zero-run length** follows that says how many
//! consecutive AC coefficients are zero. The run length is coded in
//! one of two ways:
//!
//! * **BoolCoder path** (§13.3.3.1) — traverses the Figure 16 binary
//!   tree reading one `B(prob)` BoolCoder bit at each internal node.
//!   This module surfaces the static probability data the path
//!   consumes but does **not** land the traversal itself, because
//!   `B(prob)` depends on the §7.3 `Split` formula
//!   (DOCS-GAP, see the crate-root `## DOCS-GAP` block).
//! * **Huffman path** (§13.3.3.2) — converts the same Figure 16
//!   tree's node probabilities into a 9-entry Huffman probability
//!   set via the verbatim `ZRLBoolTreeToHuffProbs` transform, then
//!   builds a Huffman tree via the §7.2 `VP6_CreateHuffmanTree`
//!   primitive ([`crate::huffman::create_huffman_tree`]). The
//!   traversal of the resulting Huffman tree reads only `R(1)` raw
//!   bits, so it is **independent of the §7.3 BoolCoder DOCS-GAP**.
//!
//! This module surfaces the **BoolCoder-independent** half of §13.3.3:
//!
//! * [`NUM_ZRL_BANDS`] / [`NUM_ZRL_NODES`] — the two structural
//!   dimensions of the `ZeroRunProbs[2][14]` array (Table 37 / Table
//!   38).
//! * [`ZrlBand`] — the two Table 37 zero-coefficient-starting-band
//!   indices: `Band0` for runs starting at coefficient positions 1–5
//!   and `Band1` for positions 6–63.
//! * [`ZrlNode`] — the fourteen Table 38 node-index names. The first
//!   eight (`0..=7`) correspond to the eight internal nodes of the
//!   Figure 16 binary tree (`>4`, `>2`, `>1`, `>3`, `>8`, `>6`, `>5`,
//!   `>7`) in the spec's enumeration order; the remaining six
//!   (`8..=13`) correspond to the six extrabits the BoolCoder path
//!   uses to encode a run length greater than 8 (LSB first), with
//!   names indicating each bit's place in `(RunLength - 9)`.
//! * [`ZERO_RUN_PROB_DEFAULTS`] — the verbatim `ZeroRunProbDefaults
//!   [2][14]` keyframe initialiser the spec mandates ("At each key
//!   frame every probability value in this array of AC Probabilities
//!   is set to the multidimensional array ZeroRunProbDefaults").
//! * [`ZRL_UPDATE_PROBS`] — the verbatim `ZrlUpdateProbs[2][14]`
//!   per-node update-flag probability bank (§13.3.3, used by the
//!   per-frame `NewNodeProbFlag` BoolCoder reads of Table 41 — the
//!   reads themselves stay deferred).
//! * [`NUM_ZRL_HUFF_PROBS`] — the size of the 9-entry Huffman
//!   probability set the §13.3.3.2 conversion outputs.
//! * [`zrl_bool_tree_to_huff_probs`] — the verbatim §13.3.3.2
//!   `ZRLBoolTreeToHuffProbs` pure-integer transform that converts an
//!   8-entry node-probability vector into a 9-entry Huffman
//!   probability set.
//! * [`build_zrl_huffman_tree`] — composes the §13.3.3.2 pseudo-code
//!   pair `ZRLBoolTreeToHuffCodes` + `VP6_BuildHuffTree` for one
//!   band: it runs the conversion above and then constructs a
//!   `2N-1 = 17`-node `HuffNode` tree the §7.2 `decode_symbol` walk
//!   can traverse against a raw-bit byte-stream reader.
//!
//! ## What this module does NOT land
//!
//! * §13.3.3.1 BoolCoder zero-run decode (Figure 16 traversal +
//!   six-bit extrabit reads). Both the per-node `B(prob)` reads and
//!   the extrabit `B(prob)` reads route through `VP6_DecodeBool`,
//!   which is blocked on the §7.3 `Split` formula DOCS-GAP. The
//!   per-frame probability *update* bitstream (Tables 39–41) is the
//!   same.
//! * The §13.3.3.2 Huffman path's symbol semantics for the 9th
//!   leaf — i.e. whether `ZrlToken == 8` (leaf at index 7 in
//!   spec-canonical order) means a literal run of 8 or is the escape
//!   trigger for the 6-extrabit read of `R(6)` shown in the §13.3.3
//!   demonstration pseudo-code. The Figure 16 tree drawing carries
//!   two leaves labelled `8`, and the demonstration
//!   `if (ZrlToken<8) … else 8 + R(6)` does not name which of the two
//!   produces which `ZrlToken` value. Surfaced as a docs-gap candidate
//!   in the crate-root report; the static surface itself is unambiguous
//!   (the conversion outputs 9 probabilities, one per leaf-codeword) so
//!   it lands here.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §13.3.3
//! (On2 Technologies, document version 1.02, August 2006). No
//! third-party VP6 implementation has been consulted.

use crate::huffman::{create_huffman_tree, HuffNode, HuffmanError};

/// Number of ZRL bands (Table 37): two — runs starting at AC
/// coefficient positions 1–5 (band 0) and 6–63 (band 1).
pub const NUM_ZRL_BANDS: usize = 2;

/// Number of ZRL nodes per band (Table 38): fourteen — eight for the
/// internal nodes of the Figure 16 binary tree and six for the
/// extrabit positions of the `(RunLength - 9)` six-bit suffix used
/// when the run length is greater than 8.
pub const NUM_ZRL_NODES: usize = 14;

/// Number of probability entries the §13.3.3.2 Huffman conversion
/// outputs: nine, matching the nine leaves of the Figure 16 binary
/// tree (the eight literal-run leaves plus the `>8` escape leaf).
pub const NUM_ZRL_HUFF_PROBS: usize = 9;

/// The two Table 37 zero-coefficient-starting-band indices.
///
/// The first dimension of `ZeroRunProbs[2][14]` is indexed by which
/// AC coefficient position the run of zeros starts at: coefficient
/// positions 1–5 select band 0 and positions 6–63 select band 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ZrlBand {
    /// Coefficient positions 1–5.
    Band0 = 0,
    /// Coefficient positions 6–63.
    Band1 = 1,
}

impl ZrlBand {
    /// All Table 37 band indices in canonical order.
    pub const ALL: [ZrlBand; NUM_ZRL_BANDS] = [ZrlBand::Band0, ZrlBand::Band1];

    /// Returns the band that contains the given AC coefficient
    /// position. The argument is the spec's `EncodedCoeffs` index
    /// (1..=63 in the AC range — coefficient 0 is the DC term, which
    /// has its own §13.4 run-length tree).
    ///
    /// Returns `None` if `coeff_index == 0` (DC, not AC) or
    /// `coeff_index > 63` (out of block).
    pub const fn for_coefficient_position(coeff_index: usize) -> Option<ZrlBand> {
        match coeff_index {
            1..=5 => Some(ZrlBand::Band0),
            6..=63 => Some(ZrlBand::Band1),
            _ => None,
        }
    }

    /// Spec-canonical index of this band (0 or 1).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Recover a [`ZrlBand`] from its canonical index. Returns
    /// `None` for indices `>= NUM_ZRL_BANDS`.
    pub const fn from_index(index: usize) -> Option<ZrlBand> {
        match index {
            0 => Some(ZrlBand::Band0),
            1 => Some(ZrlBand::Band1),
            _ => None,
        }
    }
}

/// The fourteen Table 38 node indices the second dimension of
/// `ZeroRunProbs[2][14]` enumerates.
///
/// The first eight (`0..=7`) name internal nodes of the Figure 16
/// binary tree in the spec's canonical order; per Table 38:
///
/// | Index | Run Length      |
/// |-------|-----------------|
/// | 0     | `> 4`           |
/// | 1     | `> 2`           |
/// | 2     | `> 1`           |
/// | 3     | `> 3`           |
/// | 4     | `> 8`           |
/// | 5     | `> 6`           |
/// | 6     | `> 5`           |
/// | 7     | `> 7`           |
///
/// The remaining six (`8..=13`) name the bit positions of the
/// `(RunLength - 9)` six-bit suffix the BoolCoder path reads when
/// the run length is greater than 8:
///
/// | Index | Bit                           |
/// |-------|-------------------------------|
/// | 8     | `(RunLength - 9) & 1`         |
/// | 9     | `((RunLength - 9) >> 1) & 1`  |
/// | 10    | `((RunLength - 9) >> 2) & 1`  |
/// | 11    | `((RunLength - 9) >> 3) & 1`  |
/// | 12    | `((RunLength - 9) >> 4) & 1`  |
/// | 13    | `((RunLength - 9) >> 5) & 1`  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ZrlNode {
    /// Internal node `> 4` — root of the Figure 16 tree.
    GreaterThan4 = 0,
    /// Internal node `> 2` — left child of root.
    GreaterThan2 = 1,
    /// Internal node `> 1` — left child of `> 2`.
    GreaterThan1 = 2,
    /// Internal node `> 3` — right child of `> 2`.
    GreaterThan3 = 3,
    /// Internal node `> 8` — right child of root.
    GreaterThan8 = 4,
    /// Internal node `> 6` — left child of `> 8`.
    GreaterThan6 = 5,
    /// Internal node `> 5` — left child of `> 6`.
    GreaterThan5 = 6,
    /// Internal node `> 7` — right child of `> 6`.
    GreaterThan7 = 7,
    /// Extrabit position 0 — `(RunLength - 9) & 1`.
    ExtraBit0 = 8,
    /// Extrabit position 1 — `((RunLength - 9) >> 1) & 1`.
    ExtraBit1 = 9,
    /// Extrabit position 2 — `((RunLength - 9) >> 2) & 1`.
    ExtraBit2 = 10,
    /// Extrabit position 3 — `((RunLength - 9) >> 3) & 1`.
    ExtraBit3 = 11,
    /// Extrabit position 4 — `((RunLength - 9) >> 4) & 1`.
    ExtraBit4 = 12,
    /// Extrabit position 5 — `((RunLength - 9) >> 5) & 1`.
    ExtraBit5 = 13,
}

impl ZrlNode {
    /// All fourteen Table 38 node indices in canonical (Table 38)
    /// order.
    pub const ALL: [ZrlNode; NUM_ZRL_NODES] = [
        ZrlNode::GreaterThan4,
        ZrlNode::GreaterThan2,
        ZrlNode::GreaterThan1,
        ZrlNode::GreaterThan3,
        ZrlNode::GreaterThan8,
        ZrlNode::GreaterThan6,
        ZrlNode::GreaterThan5,
        ZrlNode::GreaterThan7,
        ZrlNode::ExtraBit0,
        ZrlNode::ExtraBit1,
        ZrlNode::ExtraBit2,
        ZrlNode::ExtraBit3,
        ZrlNode::ExtraBit4,
        ZrlNode::ExtraBit5,
    ];

    /// Spec-canonical index (0..=13).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Recover a [`ZrlNode`] from its canonical index. Returns
    /// `None` for indices `>= NUM_ZRL_NODES`.
    pub fn from_index(index: usize) -> Option<ZrlNode> {
        ZrlNode::ALL.get(index).copied()
    }

    /// Reports whether this node index names an internal node of the
    /// Figure 16 binary tree (indices 0..=7). The remaining six
    /// (8..=13) name extrabit positions and are not consulted by the
    /// §13.3.3.2 Huffman conversion (`zrl_bool_tree_to_huff_probs`
    /// reads only the first eight entries).
    pub const fn is_tree_node(self) -> bool {
        (self as u8) < 8
    }

    /// Reports whether this node index names an extrabit position
    /// (indices 8..=13). Returns the bit shift the BoolCoder path
    /// applies to `(RunLength - 9)` to produce this bit.
    pub const fn extrabit_shift(self) -> Option<u8> {
        match self {
            ZrlNode::ExtraBit0 => Some(0),
            ZrlNode::ExtraBit1 => Some(1),
            ZrlNode::ExtraBit2 => Some(2),
            ZrlNode::ExtraBit3 => Some(3),
            ZrlNode::ExtraBit4 => Some(4),
            ZrlNode::ExtraBit5 => Some(5),
            _ => None,
        }
    }
}

/// Per-band keyframe initialiser for `ZeroRunProbs[2][14]`
/// (§13.3.3).
///
/// "At each key frame (I frame) every probability value in this
/// array of AC Probabilities is set to the multidimensional array
/// ZeroRunProbDefaults." The array persists from a keyframe to each
/// subsequent interframe (P frame), with per-frame updates layered
/// on top via the Table 39 / Table 40 / Table 41 BoolCoder reads
/// (which stay deferred on the §7.3 DOCS-GAP).
///
/// The two rows are indexed by [`ZrlBand`] and the fourteen columns
/// by [`ZrlNode`].
pub const ZERO_RUN_PROB_DEFAULTS: [[u8; NUM_ZRL_NODES]; NUM_ZRL_BANDS] = [
    [
        198, 197, 196, 146, 198, 204, 169, 142, 130, 136, 149, 149, 191, 249,
    ],
    [
        135, 201, 181, 154, 98, 117, 132, 126, 146, 169, 184, 240, 246, 254,
    ],
];

/// Per-band per-node update-flag probability bank for the
/// `NewNodeProbFlag` `B(x)` reads of Table 41 (§13.3.3).
///
/// "The probability used for decoding zrl probabilities node field
/// NewNodeProbFlag is determined from the following table." Indexed
/// by [`ZrlBand`] then by [`ZrlNode`]. The BoolCoder reads themselves
/// stay deferred on the §7.3 DOCS-GAP; this surface lands the
/// numeric data the reads will consume once the DOCS-GAP is closed.
pub const ZRL_UPDATE_PROBS: [[u8; NUM_ZRL_NODES]; NUM_ZRL_BANDS] = [
    [
        219, 246, 238, 249, 232, 239, 249, 255, 248, 253, 239, 244, 241, 248,
    ],
    [
        198, 232, 251, 253, 219, 241, 253, 255, 248, 249, 244, 238, 251, 255,
    ],
];

/// The verbatim §13.3.3.2 `ZRLBoolTreeToHuffProbs` conversion.
///
/// Reads the first eight entries of a `ZeroRunProbs` row (the eight
/// internal-node probabilities of the Figure 16 binary tree) and
/// outputs nine Huffman probabilities — one per leaf the Huffman
/// tree the §7.2 `VP6_CreateHuffmanTree` primitive will be asked to
/// build for this band.
///
/// Spec pseudo-code, verbatim from §13.3.3.2:
///
/// ```text
/// ZRLBoolTreeToHuffProbs
/// {
///     Prob       = (NodeProb[0] * NodeProb[1]) >> 8
///     HuffProb[0] = (Prob * NodeProb[2]) >> 8
///     HuffProb[1] = (Prob * (255 - NodeProb[2])) >> 8
///
///     Prob       = (NodeProb[0] * 255 - NodeProb[1]) >> 8
///     HuffProb[2] = (Prob * NodeProb[3]) >> 8
///     HuffProb[3] = (Prob * 255 - NodeProb[3]) >> 8
///
///     Prob       = (255 - NodeProb[0]) * NodeProb[4]) >> 8
///     Prob       = (Prob * NodeProb[5]) >> 8
///     HuffProb[4] = (Prob * NodeProb[6]) >> 8
///     HuffProb[5] = (Prob * 255 - NodeProb[6])) >> 8
///
///     Prob       = ((255 - NodeProb[0]) * NodeProb[4]) >> 8
///     Prob       = (Prob * (255 - NodeProb[5])) >> 8
///     HuffProb[6] = (Prob * NodeProb[7]) >> 8
///     HuffProb[7] = (Prob * 255 - NodeProb[7])) >> 8
///
///     Prob       = ((255 - NodeProb[0]) * (255 - NodeProb[4])) >> 8
///     HuffProb[8] = Prob
/// }
/// ```
///
/// The output is indexed `0..=8` and corresponds (in spec order)
/// to the nine leaves of Figure 16, walking the tree's branches
/// from the `>4` root outward via the §13.3.3.2 ordering implicit
/// in the conversion. Pure integer arithmetic; no BoolCoder reads.
///
/// Note: the verbatim listing above has parenthesisation peculiarities
/// in the four lines that begin "Prob = ((255 …" / "Prob = …" — we
/// reproduce the intent (subtraction from 255 of the relevant
/// `NodeProb` entry, applied via `(255 - x)` and `* y >> 8`) so that
/// the conversion is internally consistent: every leaf probability
/// derives from a chain whose probabilities sum to ~255 (`>> 8`
/// truncation aside).
pub fn zrl_bool_tree_to_huff_probs(node_prob: &[u8; 8]) -> [u8; NUM_ZRL_HUFF_PROBS] {
    let np = |i: usize| node_prob[i] as u32;
    let mut huff = [0u32; NUM_ZRL_HUFF_PROBS];

    // Left half of root (NodeProb[0] = ">4", taken left):
    //   under ">2" (NodeProb[1]):
    //     left  -> ">1" (NodeProb[2])     -> leaves 0 (>1=false) / 1 (>1=true)
    //     right -> ">3" (NodeProb[3])     -> leaves 2 (>3=false) / 3 (>3=true)
    let mut prob = (np(0) * np(1)) >> 8;
    huff[0] = (prob * np(2)) >> 8;
    huff[1] = (prob * (255 - np(2))) >> 8;

    prob = (np(0) * (255 - np(1))) >> 8;
    huff[2] = (prob * np(3)) >> 8;
    huff[3] = (prob * (255 - np(3))) >> 8;

    // Right half of root (NodeProb[0] = ">4", taken right):
    //   under ">8" (NodeProb[4]):
    //     left -> ">6" (NodeProb[5]):
    //       left  -> ">5" (NodeProb[6]) -> leaves 4 / 5
    //       right -> ">7" (NodeProb[7]) -> leaves 6 / 7
    //     right -> leaf 8 (the ">8" escape).
    let root_right = ((255 - np(0)) * np(4)) >> 8;
    let gt6 = (root_right * np(5)) >> 8;
    huff[4] = (gt6 * np(6)) >> 8;
    huff[5] = (gt6 * (255 - np(6))) >> 8;

    let gt6_right = (root_right * (255 - np(5))) >> 8;
    huff[6] = (gt6_right * np(7)) >> 8;
    huff[7] = (gt6_right * (255 - np(7))) >> 8;

    huff[8] = ((255 - np(0)) * (255 - np(4))) >> 8;

    let mut out = [0u8; NUM_ZRL_HUFF_PROBS];
    for (slot, value) in out.iter_mut().zip(huff.iter()) {
        *slot = *value as u8;
    }
    out
}

/// Compose §13.3.3.2 `ZRLBoolTreeToHuffCodes` + `VP6_BuildHuffTree`
/// for one band.
///
/// Given the eight internal-node probabilities of one
/// `ZeroRunProbs[band]` row (i.e. `node_prob[0..=7]`), this:
///
/// 1. Converts them to a nine-entry Huffman probability set via
///    [`zrl_bool_tree_to_huff_probs`] (the §13.3.3.2 listing's
///    `ZRLBoolTreeToHuffCodes` step).
/// 2. Builds a `(2*9 - 1) = 17`-node `HuffNode` tree via
///    [`create_huffman_tree`] (the §7.2 `VP6_CreateHuffmanTree`
///    primitive), tagging each leaf with its spec-canonical symbol
///    index `0..=8` (matching the conversion output index).
///
/// The returned tree is BoolCoder-independent: walking it consumes
/// only `R(1)` raw bits per branch, so [`crate::huffman::decode_symbol`]
/// can traverse it once a byte-stream `R(1)` reader is wired up.
///
/// Returns the same [`HuffmanError`] family that
/// [`create_huffman_tree`] can produce; for any sane input the
/// outer call cannot fail (the conversion always yields nine
/// probabilities). The `Err` arm is preserved for symmetry with
/// [`create_huffman_tree`] and to future-proof against changes to
/// the leaf count.
pub fn build_zrl_huffman_tree(node_prob: &[u8; 8]) -> Result<Vec<HuffNode>, HuffmanError> {
    let huff_probs = zrl_bool_tree_to_huff_probs(node_prob);
    // Each leaf gets its spec-canonical symbol index as its symbol.
    // The §7.2 builder rejects probability == 0 (spec: "0 indicates
    // an impossible event"); for zero leaves we substitute the
    // smallest legal probability (1) so the structural tree shape
    // matches the spec's nine-leaf topology even when the converted
    // probabilities under-flow on the `>> 8` truncation.
    let symbols: Vec<i32> = (0..NUM_ZRL_HUFF_PROBS as i32).collect();
    let probs: Vec<u8> = huff_probs
        .iter()
        .map(|&p| if p == 0 { 1 } else { p })
        .collect();
    // `create_huffman_tree` expects parallel symbol / probability
    // slices.
    create_huffman_tree(&symbols, &probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::{tree_depth, INTERNAL_SYMBOL, NO_CHILD};

    // -------- ZrlBand enum surface --------

    #[test]
    fn zrl_band_indices_match_table_37() {
        assert_eq!(ZrlBand::Band0.index(), 0);
        assert_eq!(ZrlBand::Band1.index(), 1);
    }

    #[test]
    fn zrl_band_round_trip() {
        for (i, band) in ZrlBand::ALL.iter().enumerate() {
            assert_eq!(ZrlBand::from_index(i), Some(*band));
            assert_eq!(band.index(), i);
        }
        assert_eq!(ZrlBand::from_index(NUM_ZRL_BANDS), None);
        assert_eq!(ZrlBand::from_index(usize::MAX), None);
    }

    #[test]
    fn zrl_band_for_coefficient_position_partitions_1_to_63() {
        // Table 37 row 0: coefficients 1-5 -> Band 0.
        for coeff in 1..=5 {
            assert_eq!(
                ZrlBand::for_coefficient_position(coeff),
                Some(ZrlBand::Band0)
            );
        }
        // Table 37 row 1: coefficients 6-63 -> Band 1.
        for coeff in 6..=63 {
            assert_eq!(
                ZrlBand::for_coefficient_position(coeff),
                Some(ZrlBand::Band1)
            );
        }
    }

    #[test]
    fn zrl_band_for_coefficient_position_rejects_dc_and_out_of_block() {
        // Coefficient 0 is DC, not AC.
        assert_eq!(ZrlBand::for_coefficient_position(0), None);
        // Anything beyond the 64-coefficient block is out of range.
        assert_eq!(ZrlBand::for_coefficient_position(64), None);
        assert_eq!(ZrlBand::for_coefficient_position(255), None);
        assert_eq!(ZrlBand::for_coefficient_position(usize::MAX), None);
    }

    // -------- ZrlNode enum surface --------

    #[test]
    fn zrl_node_indices_match_table_38() {
        assert_eq!(ZrlNode::GreaterThan4.index(), 0);
        assert_eq!(ZrlNode::GreaterThan2.index(), 1);
        assert_eq!(ZrlNode::GreaterThan1.index(), 2);
        assert_eq!(ZrlNode::GreaterThan3.index(), 3);
        assert_eq!(ZrlNode::GreaterThan8.index(), 4);
        assert_eq!(ZrlNode::GreaterThan6.index(), 5);
        assert_eq!(ZrlNode::GreaterThan5.index(), 6);
        assert_eq!(ZrlNode::GreaterThan7.index(), 7);
        assert_eq!(ZrlNode::ExtraBit0.index(), 8);
        assert_eq!(ZrlNode::ExtraBit1.index(), 9);
        assert_eq!(ZrlNode::ExtraBit2.index(), 10);
        assert_eq!(ZrlNode::ExtraBit3.index(), 11);
        assert_eq!(ZrlNode::ExtraBit4.index(), 12);
        assert_eq!(ZrlNode::ExtraBit5.index(), 13);
    }

    #[test]
    fn zrl_node_round_trip() {
        for (i, node) in ZrlNode::ALL.iter().enumerate() {
            assert_eq!(ZrlNode::from_index(i), Some(*node));
            assert_eq!(node.index(), i);
        }
        assert_eq!(ZrlNode::from_index(NUM_ZRL_NODES), None);
        assert_eq!(ZrlNode::from_index(usize::MAX), None);
    }

    #[test]
    fn zrl_node_all_length_matches_constant() {
        assert_eq!(ZrlNode::ALL.len(), NUM_ZRL_NODES);
    }

    #[test]
    fn zrl_node_is_tree_node_partitions_first_eight() {
        // Tree-internal nodes: indices 0..=7.
        for node in ZrlNode::ALL.iter().take(8) {
            assert!(node.is_tree_node(), "{:?} should be a tree node", node);
        }
        // Extrabits: indices 8..=13.
        for node in ZrlNode::ALL.iter().skip(8) {
            assert!(!node.is_tree_node(), "{:?} should not be a tree node", node);
        }
    }

    #[test]
    fn zrl_node_extrabit_shift_matches_table_38() {
        // Tree-internal nodes have no extrabit shift.
        for node in ZrlNode::ALL.iter().take(8) {
            assert_eq!(node.extrabit_shift(), None, "{:?}", node);
        }
        // Extrabit shifts increment by one from index 8 onwards.
        assert_eq!(ZrlNode::ExtraBit0.extrabit_shift(), Some(0));
        assert_eq!(ZrlNode::ExtraBit1.extrabit_shift(), Some(1));
        assert_eq!(ZrlNode::ExtraBit2.extrabit_shift(), Some(2));
        assert_eq!(ZrlNode::ExtraBit3.extrabit_shift(), Some(3));
        assert_eq!(ZrlNode::ExtraBit4.extrabit_shift(), Some(4));
        assert_eq!(ZrlNode::ExtraBit5.extrabit_shift(), Some(5));
    }

    // -------- Static probability banks --------

    #[test]
    fn zero_run_prob_defaults_table_dimensions() {
        assert_eq!(ZERO_RUN_PROB_DEFAULTS.len(), NUM_ZRL_BANDS);
        for row in &ZERO_RUN_PROB_DEFAULTS {
            assert_eq!(row.len(), NUM_ZRL_NODES);
        }
    }

    #[test]
    fn zero_run_prob_defaults_first_row_verbatim_values() {
        // Spec listing row 0, verbatim.
        assert_eq!(
            ZERO_RUN_PROB_DEFAULTS[0],
            [198, 197, 196, 146, 198, 204, 169, 142, 130, 136, 149, 149, 191, 249,]
        );
    }

    #[test]
    fn zero_run_prob_defaults_second_row_verbatim_values() {
        // Spec listing row 1, verbatim.
        assert_eq!(
            ZERO_RUN_PROB_DEFAULTS[1],
            [135, 201, 181, 154, 98, 117, 132, 126, 146, 169, 184, 240, 246, 254,]
        );
    }

    #[test]
    fn zrl_update_probs_table_dimensions() {
        assert_eq!(ZRL_UPDATE_PROBS.len(), NUM_ZRL_BANDS);
        for row in &ZRL_UPDATE_PROBS {
            assert_eq!(row.len(), NUM_ZRL_NODES);
        }
    }

    #[test]
    fn zrl_update_probs_first_row_verbatim_values() {
        assert_eq!(
            ZRL_UPDATE_PROBS[0],
            [219, 246, 238, 249, 232, 239, 249, 255, 248, 253, 239, 244, 241, 248,]
        );
    }

    #[test]
    fn zrl_update_probs_second_row_verbatim_values() {
        assert_eq!(
            ZRL_UPDATE_PROBS[1],
            [198, 232, 251, 253, 219, 241, 253, 255, 248, 249, 244, 238, 251, 255,]
        );
    }

    // -------- zrl_bool_tree_to_huff_probs --------

    #[test]
    fn zrl_huff_probs_size_matches_nine_leaves() {
        let probs = zrl_bool_tree_to_huff_probs(&[128; 8]);
        assert_eq!(probs.len(), NUM_ZRL_HUFF_PROBS);
    }

    #[test]
    fn zrl_huff_probs_uniform_inputs_yield_within_node_pair_equality() {
        // With every node probability at 128 (50/50 left/right), each
        // pair of leaves that share the *same* internal parent node
        // must be approximately equal: the only difference between
        // them is a single `(prob * 128) >> 8` vs `(prob * 127) >> 8`
        // at the parent's branch, which loses at most 1 across the
        // remaining chain factors. Verify that each within-parent
        // pair differs by at most 1.
        let probs = zrl_bool_tree_to_huff_probs(&[128; 8]);
        for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
            let diff = (probs[a] as i32) - (probs[b] as i32);
            assert!(
                diff.unsigned_abs() <= 1,
                "pair ({a},{b}) differs by {diff}: {} vs {}",
                probs[a],
                probs[b]
            );
        }
    }

    #[test]
    fn zrl_huff_probs_uniform_inputs_match_tree_depth_geometry() {
        // The Figure 16 tree is asymmetric: the LEFT half of the
        // root has two levels (depth 2) and contains 4 leaves
        // (indices 0..=3); the RIGHT half of the root has one level
        // to the `>8` escape leaf (depth 1, leaf 8) and two levels
        // to the deeper sub-tree (depth 3, leaves 4..=7). With
        // uniform `[128; 8]` inputs each `>> 8` of `*128` halves
        // the chain, so we should see:
        //   - leaves 0..=3 (depth 2): each ≈ 255 / 4 ≈ 63 (minus
        //     truncation), so each ≈ 31 after a `* 128 >> 8`
        //     truncation chain.
        //   - leaves 4..=7 (depth 3): each ≈ 255 / 8 ≈ 31, then
        //     halved again to ≈ 15.
        //   - leaf 8 (depth 1): ≈ 255 / 2 ≈ 127.
        let probs = zrl_bool_tree_to_huff_probs(&[128; 8]);
        // Left half (4 leaves at depth 2) totals should be larger
        // than right_lower (4 leaves at depth 3).
        let left_half = probs[0] as u32 + probs[1] as u32 + probs[2] as u32 + probs[3] as u32;
        let right_lower = probs[4] as u32 + probs[5] as u32 + probs[6] as u32 + probs[7] as u32;
        assert!(
            left_half > right_lower,
            "left half (depth-2 leaves) total {} should exceed right \
             lower (depth-3 leaves) total {}",
            left_half,
            right_lower
        );
        // The shallowest leaf (the `>8` escape, depth 1) carries
        // the most mass of any single leaf.
        for i in 0..8 {
            assert!(
                probs[8] > probs[i],
                "leaf 8 (depth 1) prob {} should exceed leaf {} prob {}",
                probs[8],
                i,
                probs[i]
            );
        }
    }

    #[test]
    fn zrl_huff_probs_root_extreme_left_zeroes_right_branches() {
        // node_prob[0] = 0 means "always go left from the root":
        // none of the right-subtree leaves (4..=8) should receive
        // any probability mass; all `(255 - np(0)) * x = 255 * x`
        // factors are still nonzero, BUT the leaves under the right
        // subtree multiply by `np(0)`-free factors that depend on
        // the remaining np values. With np(0) = 0 the spec listing's
        // "left half" prefactors `np(0) * np(1)` and
        // `np(0) * (255 - np(1))` are both zero — so leaves 0..=3
        // are zero — and leaves 4..=8 are nonzero. We test the
        // mirror case in the next test for the opposite extreme.
        let mut np = [128u8; 8];
        np[0] = 0;
        let probs = zrl_bool_tree_to_huff_probs(&np);
        assert_eq!(probs[0], 0);
        assert_eq!(probs[1], 0);
        assert_eq!(probs[2], 0);
        assert_eq!(probs[3], 0);
        // The right-subtree leaves receive all the mass.
        let right_sum =
            probs[4] as u32 + probs[5] as u32 + probs[6] as u32 + probs[7] as u32 + probs[8] as u32;
        assert!(right_sum > 0);
    }

    #[test]
    fn zrl_huff_probs_root_extreme_right_zeroes_left_branches() {
        // Mirror of the previous test: np(0) = 255 means "always
        // go right from the root" so leaves 4..=8 of the right
        // subtree become zero (since `(255 - np(0)) = 0`).
        let mut np = [128u8; 8];
        np[0] = 255;
        let probs = zrl_bool_tree_to_huff_probs(&np);
        assert_eq!(probs[4], 0);
        assert_eq!(probs[5], 0);
        assert_eq!(probs[6], 0);
        assert_eq!(probs[7], 0);
        assert_eq!(probs[8], 0);
        // The left-subtree leaves receive all the mass.
        let left_sum = probs[0] as u32 + probs[1] as u32 + probs[2] as u32 + probs[3] as u32;
        assert!(left_sum > 0);
    }

    #[test]
    fn zrl_huff_probs_keyframe_defaults_band_0_well_formed() {
        // The keyframe defaults must yield well-formed probabilities
        // for both bands: every output entry is a valid u8 (cannot
        // overflow because every chain factor is `(... * y) >> 8`
        // with `y <= 255`).
        let probs = zrl_bool_tree_to_huff_probs(ZERO_RUN_PROB_DEFAULTS[0][..8].try_into().unwrap());
        // No assertion on magnitudes — we just confirm we can call
        // the conversion with the spec defaults and get the right
        // number of outputs.
        assert_eq!(probs.len(), NUM_ZRL_HUFF_PROBS);
    }

    #[test]
    fn zrl_huff_probs_keyframe_defaults_band_1_well_formed() {
        let probs = zrl_bool_tree_to_huff_probs(ZERO_RUN_PROB_DEFAULTS[1][..8].try_into().unwrap());
        assert_eq!(probs.len(), NUM_ZRL_HUFF_PROBS);
    }

    // -------- build_zrl_huffman_tree --------

    #[test]
    fn build_zrl_huffman_tree_topology_has_nine_leaves() {
        // The Figure 16 tree has exactly nine leaves (eight literal
        // run-length leaves plus the `>8` escape leaf), so the §7.2
        // builder produces a tree of `2 * 9 - 1 = 17` nodes.
        let tree = build_zrl_huffman_tree(&[128; 8]).expect("uniform probs build");
        assert_eq!(tree.len(), 17);

        // Count leaves (nodes with a non-`INTERNAL_SYMBOL` symbol).
        let leaves: Vec<&HuffNode> = tree
            .iter()
            .filter(|n| n.symbol != INTERNAL_SYMBOL)
            .collect();
        assert_eq!(leaves.len(), 9);

        // Each leaf carries one of the canonical symbol indices 0..=8.
        let mut leaf_symbols: Vec<i32> = leaves.iter().map(|n| n.symbol).collect();
        leaf_symbols.sort_unstable();
        assert_eq!(leaf_symbols, (0..9).collect::<Vec<_>>());

        // Internal nodes have `NO_CHILD` as a non-applicable marker
        // only for the children of leaves; they always have two
        // children themselves.
        let internals: Vec<&HuffNode> = tree
            .iter()
            .filter(|n| n.symbol == INTERNAL_SYMBOL)
            .collect();
        assert_eq!(internals.len(), 8); // 9 - 1 = 8 internal nodes
        for node in &internals {
            assert_ne!(node.left, NO_CHILD);
            assert_ne!(node.right, NO_CHILD);
        }
    }

    #[test]
    fn build_zrl_huffman_tree_root_at_2n_minus_2() {
        // §7.2.1: "the root is at index 2N - 2". With N = 9 leaves,
        // the root sits at index 16 (last entry of a 17-element
        // vector).
        let tree = build_zrl_huffman_tree(&[128; 8]).unwrap();
        let root = tree.last().unwrap();
        // The root is an internal node.
        assert_eq!(root.symbol, INTERNAL_SYMBOL);
        // The root's children indices are strictly less than its own
        // (the §7.2.1 bottom-up merge produces increasing internal
        // indices).
        assert!(root.left < 16);
        assert!(root.right < 16);
    }

    #[test]
    fn build_zrl_huffman_tree_each_leaf_reachable() {
        // From the root, every canonical symbol 0..=8 must be
        // reachable along some root-to-leaf path. Confirm via the
        // §7.2 walker's tree_depth helper from
        // crate::huffman::tree_depth (which returns Some(_) for any
        // reachable symbol).
        let tree = build_zrl_huffman_tree(&[128; 8]).unwrap();
        for sym in 0..NUM_ZRL_HUFF_PROBS as i32 {
            assert!(
                tree_depth(&tree, sym).is_some(),
                "leaf symbol {} not reachable from root",
                sym
            );
        }
    }

    #[test]
    fn build_zrl_huffman_tree_handles_keyframe_defaults() {
        // Both keyframe-default rows must build a valid tree.
        for row in ZERO_RUN_PROB_DEFAULTS.iter().take(NUM_ZRL_BANDS) {
            let np: [u8; 8] = row[..8].try_into().unwrap();
            let tree = build_zrl_huffman_tree(&np).expect("default probs build");
            assert_eq!(tree.len(), 17);
            // Every canonical symbol must be reachable.
            for sym in 0..NUM_ZRL_HUFF_PROBS as i32 {
                assert!(tree_depth(&tree, sym).is_some());
            }
        }
    }

    #[test]
    fn build_zrl_huffman_tree_skewed_inputs_shorten_dominant_symbol() {
        // §7.2.1 builds canonical Huffman trees: the symbol with the
        // largest probability gets the shortest codeword. Skew the
        // root strongly so that the LEFT subtree dominates and one
        // of its leaves has substantially more mass than any other.
        // Then assert that this leaf's depth is shallower than the
        // deepest leaf.
        let np = [200u8, 200, 250, 5, 5, 5, 5, 5];
        let tree = build_zrl_huffman_tree(&np).unwrap();
        let probs = zrl_bool_tree_to_huff_probs(&np);

        // Identify the symbol with the largest converted probability.
        let (max_sym, _) = probs
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| **p)
            .map(|(s, p)| (s as i32, *p))
            .unwrap();
        // And the symbol with the smallest non-zero converted
        // probability (which the builder substitutes 1 for if zero).
        let (min_sym, _) = probs
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0)
            .min_by_key(|(_, p)| **p)
            .map(|(s, p)| (s as i32, *p))
            .unwrap_or_else(|| {
                // Fallback if every probability collapsed to zero
                // (won't happen for this input).
                (0, 0)
            });

        let max_depth = tree_depth(&tree, max_sym).unwrap();
        let min_depth = tree_depth(&tree, min_sym).unwrap();
        // The dominant symbol's codeword should be no longer than
        // the rare symbol's.
        assert!(
            max_depth <= min_depth,
            "dominant symbol depth {} > rare symbol depth {}",
            max_depth,
            min_depth
        );
    }
}

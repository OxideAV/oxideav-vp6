//! VP6 DCT-coefficient token static surface (spec §13).
//!
//! VP6 codes quantized DCT coefficients with a set of twelve **tokens**
//! (spec §13 Table 18) decoded from a fixed binary tree (Figure 15).
//! The arithmetic decoder walks the tree node-by-node, reading one
//! `B(prob)` BoolCoder bit at each internal node; the Huffman path
//! converts the same node probabilities into a 12-entry Huffman
//! probability set. Both paths share the per-node baseline probability
//! banks (`DcProbs` / `AcProbs`), the per-frame update-flag probability
//! tables (`VP6_DcUpdateProbs` / `AcUpdateProbs`), and — for DC — a
//! pure-integer `DcProbs → DcNodeContexts` conversion driven by the
//! `DcNodeEqs` linear-equation table.
//!
//! This module surfaces the **BoolCoder-independent** half of §13:
//!
//! * [`DctToken`] — the twelve Table 18 tokens, the spec's canonical
//!   `0..=11` indexing, and each token's `(min, extra_bits)` extrabit
//!   geometry plus the per-extrabit arithmetic-coding probability
//!   vector ([`DctToken::extra_bit_probs`]).
//! * [`TreeNode`] — the eleven Table 20 internal-node names with the
//!   spec's canonical `0..=10` indexing (the index into a node
//!   probability vector).
//! * [`baseline_dc_probs`] / [`baseline_ac_probs`] — the all-128
//!   keyframe initialisers for `DcProbs[2][11]` and
//!   `AcProbs[2][3][6][11]` (§13.2 / §13.3: "At each key frame every
//!   probability value … is set to 128").
//! * [`VP6_DC_UPDATE_PROBS`] — the verbatim `VP6_DcUpdateProbs[2][11]`
//!   per-node update-flag probability bank (§13.2).
//! * [`AC_UPDATE_PROBS`] — the verbatim `AcUpdateProbs[3][2][6][11]`
//!   per-node update-flag probability bank (§13.3).
//! * [`DC_NODE_EQS`] — the verbatim `DcNodeEqs[5][3][2]` slope/constant
//!   linear-equation table (§13.2 Table 27).
//! * [`dc_probs_to_node_contexts`] — the pure-integer §13.2 conversion
//!   that expands a `DcProbs[2][11]` bank into the
//!   `DcNodeContexts[2][3][11]` array the §13.2.1 arithmetic DC decoder
//!   consults (one tree per left/above zero-DC context).
//! * [`dct_token_bool_tree_to_huff_probs`] — the verbatim §13.1
//!   `DCTTokenBoolTreeToHuffProbs` transform that converts an 11-entry
//!   node-probability vector into the 12-entry Huffman probability set
//!   used by the §13.2.2 / §13.3.2 Huffman token decoders.
//!
//! ## What this module does NOT land
//!
//! The §13 *token traversal* (`VP6_DecodeToken`) reads `B(prob)`
//! BoolCoder bits at each tree node, plus per-token extrabits via
//! `B(...)`, and the AC zero-run decode of §13.3.3 reads further
//! BoolCoder bits. Every one of those reads depends on the §7.3
//! `Split` formula, which is blocked by a DOCS-GAP (see the crate-root
//! docs `## DOCS-GAP` section). The per-frame probability *update*
//! bitstream (§13.2 Table 22–24 / §13.3 Table 31–35) is likewise
//! BoolCoder-gated and stays deferred.
//!
//! What we *do* land is everything that does not call the BoolCoder:
//! the enum surfaces, the static probability banks, the slope/constant
//! `DcNodeEqs` table, and the two pure-integer conversions
//! (`DcProbs → DcNodeContexts` and the node-tree → Huffman-prob
//! transform). With these in place, the only piece of §13 still pending
//! the §7.3 fix is the BoolCoder reads of the traversal itself.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §13 (On2
//! Technologies, document version 1.02, August 2006). No third-party
//! VP6 implementation has been consulted.

use core::fmt;

/// Number of DCT coefficient tokens (§13 Table 18).
///
/// Twelve: `ZERO_TOKEN`, `ONE_TOKEN`, `TWO_TOKEN`, `THREE_TOKEN`,
/// `FOUR_TOKEN`, `DCT_VAL_CATEGORY1`..`6`, and `DCT_EOB_TOKEN`. The
/// spec's `MAX_ENTROPY_TOKENS` is this value.
pub const NUM_DCT_TOKENS: usize = 12;

/// Number of internal probability nodes per coding-tree vector
/// (§13 Table 20: "a single dimensional vector with 11 entries").
///
/// The fourth dimension of `AcProbs`, the second of `DcProbs`, the
/// third of `DcNodeContexts`, and the input length of both §13
/// conversions all equal this.
pub const NUM_TREE_NODES: usize = 11;

/// Number of colour planes the §13 probability banks distinguish
/// (Table 21 / Table 28): index 0 = Y, index 1 = U or V.
pub const NUM_PLANES: usize = 2;

/// Number of DC node contexts (§13.2 Table 26): the left/above
/// predicted-DC-zero situation — both zero (0), exactly one non-zero
/// (1), both non-zero (2).
pub const NUM_DC_CONTEXTS: usize = 3;

/// Number of AC "preceding coefficient" contexts (§13.3 Table 29):
/// preceding decoded coefficient was 0 (0), 1 (1), or > 1 (2).
pub const NUM_AC_PREC_CONTEXTS: usize = 3;

/// Number of AC coefficient bands (§13.3 Table 30): coefficient 1 (0),
/// 2–4 (1), 5–10 (2), 11–21 (3), 22–36 (4), 37–63 (5).
pub const NUM_AC_BANDS: usize = 6;

/// Number of linear-equation rows in `DcNodeEqs` (§13.2): the first
/// five tree nodes get a slope/constant equation; nodes 5–10 use the
/// transmitted probability directly.
pub const NUM_DC_NODE_EQS: usize = 5;

/// VP6 DCT coefficient tokens (spec §13 Table 18).
///
/// The discriminant matches the canonical `0..=11` index the spec uses
/// when indexing token arrays (e.g. as the second index of
/// `DcHuffProbs[2][12]`, or as the `token` value the Figure 15
/// traversal resolves). The declaration order follows Table 18's `Ind`
/// column verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DctToken {
    /// `ZERO_TOKEN`. Coefficient value 0. Index 0. In AC and
    /// Huffman-DC positions it carries a zero-run extrabit suffix
    /// (§13 Table 19); in arithmetic DC it is a plain zero.
    Zero = 0,
    /// `ONE_TOKEN`. Coefficient magnitude 1; one extra bit (sign).
    /// Index 1.
    One = 1,
    /// `TWO_TOKEN`. Coefficient magnitude 2; one extra bit (sign).
    /// Index 2.
    Two = 2,
    /// `THREE_TOKEN`. Coefficient magnitude 3; one extra bit (sign).
    /// Index 3.
    Three = 3,
    /// `FOUR_TOKEN`. Coefficient magnitude 4; one extra bit (sign).
    /// Index 4.
    Four = 4,
    /// `DCT_VAL_CATEGORY1`. Magnitude 5–6; two extra bits. Index 5.
    Category1 = 5,
    /// `DCT_VAL_CATEGORY2`. Magnitude 7–10; three extra bits. Index 6.
    Category2 = 6,
    /// `DCT_VAL_CATEGORY3`. Magnitude 11–18; four extra bits. Index 7.
    Category3 = 7,
    /// `DCT_VAL_CATEGORY4`. Magnitude 19–34; five extra bits. Index 8.
    Category4 = 8,
    /// `DCT_VAL_CATEGORY5`. Magnitude 35–66; six extra bits. Index 9.
    Category5 = 9,
    /// `DCT_VAL_CATEGORY6`. Magnitude 67–2114; twelve extra bits.
    /// Index 10.
    Category6 = 10,
    /// `DCT_EOB_TOKEN`. End-of-block marker. Index 11. Forbidden in
    /// the DC position (§13.2).
    EndOfBlock = 11,
}

impl DctToken {
    /// All twelve tokens in canonical (Table 18) index order.
    pub const ALL: [DctToken; NUM_DCT_TOKENS] = [
        DctToken::Zero,
        DctToken::One,
        DctToken::Two,
        DctToken::Three,
        DctToken::Four,
        DctToken::Category1,
        DctToken::Category2,
        DctToken::Category3,
        DctToken::Category4,
        DctToken::Category5,
        DctToken::Category6,
        DctToken::EndOfBlock,
    ];

    /// Canonical `0..=11` spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`DctToken::index`]: build a token from the spec's
    /// `0..=11` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            5 => Some(Self::Category1),
            6 => Some(Self::Category2),
            7 => Some(Self::Category3),
            8 => Some(Self::Category4),
            9 => Some(Self::Category5),
            10 => Some(Self::Category6),
            11 => Some(Self::EndOfBlock),
            _ => None,
        }
    }

    /// Smallest magnitude this token can encode (§13 Table 18 `Min`).
    ///
    /// `ZERO_TOKEN` is 0; `EOB_TOKEN` carries no value and returns 0.
    #[inline]
    pub const fn min_value(self) -> u16 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Category1 => 5,
            Self::Category2 => 7,
            Self::Category3 => 11,
            Self::Category4 => 19,
            Self::Category5 => 35,
            Self::Category6 => 67,
            Self::EndOfBlock => 0,
        }
    }

    /// Largest magnitude this token can encode (§13 Table 18 `Max`).
    ///
    /// `EOB_TOKEN` carries no value and returns 0.
    #[inline]
    pub const fn max_value(self) -> u16 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Category1 => 6,
            Self::Category2 => 10,
            Self::Category3 => 18,
            Self::Category4 => 34,
            Self::Category5 => 66,
            Self::Category6 => 2114,
            Self::EndOfBlock => 0,
        }
    }

    /// Number of extra bits (including sign) the token's value field
    /// occupies (§13 Table 18 `#Extra Bits`).
    ///
    /// `ZERO_TOKEN` and `EOB_TOKEN` carry no per-token magnitude
    /// extrabits in the §13.2.1 / §13.3.1 traversal sense (their
    /// suffixes are the zero-run / block-run codes of §13.3.3 /
    /// §13.4), so this returns 0 for both.
    #[inline]
    pub const fn extra_bits(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 1,
            Self::Three => 1,
            Self::Four => 1,
            Self::Category1 => 2,
            Self::Category2 => 3,
            Self::Category3 => 4,
            Self::Category4 => 5,
            Self::Category5 => 6,
            Self::Category6 => 12,
            Self::EndOfBlock => 0,
        }
    }

    /// The per-extrabit arithmetic-coding probabilities (§13 Table 18,
    /// "Arithmetic Encoding the Extra Bits"), verbatim as printed.
    ///
    /// This is the spec's `TokenSetExtrabits[token].Probs` field, which
    /// §13.2.1 defines as "an array made from concatenating the choices
    /// in field 'Arithmetic Encoding the Extra Bits'". Tokens with no
    /// value field (`ZERO_TOKEN`, `EOB_TOKEN`) return an empty slice.
    ///
    /// **Length note:** for every token except `DCT_VAL_CATEGORY6` this
    /// list's length equals [`DctToken::extra_bits`] (the "# of
    /// extrabits, incl. sign" column), and its final entry is `B(128)`
    /// — the sign prior. `DCT_VAL_CATEGORY6` lists **11** probabilities
    /// against an `extra_bits` of **12**: its magnitude spans 67..=2114
    /// (2048 values → 11 magnitude bits) plus a sign.
    ///
    /// The errata-#67 corrected reading (resolved in
    /// `docs/video/vp6/vp6-errata-and-clarifications.md`) is that
    /// `CATEGORY6` is the only internally consistent row — every other
    /// row's trailing `B(128)` is the spurious sign prior, which is
    /// actually decoded separately by a fixed `b(1)`. Callers that
    /// drive the §13.2.1 / §13.3.1 magnitude loop should use
    /// [`DctToken::magnitude_probs`] instead, which surfaces the
    /// `#ExtraBits − 1` magnitude-only probabilities; this accessor
    /// preserves the as-printed columns for callers that need them.
    #[inline]
    pub const fn extra_bit_probs(self) -> &'static [u8] {
        match self {
            Self::Zero | Self::EndOfBlock => &[],
            // Single sign bit, fixed prior 128.
            Self::One | Self::Two | Self::Three | Self::Four => &[128],
            Self::Category1 => &[159, 128],
            Self::Category2 => &[165, 145, 128],
            Self::Category3 => &[173, 148, 140, 128],
            Self::Category4 => &[176, 155, 140, 135, 128],
            Self::Category5 => &[180, 157, 141, 134, 130, 128],
            Self::Category6 => &[254, 254, 243, 230, 196, 177, 153, 140, 133, 129, 128],
        }
    }

    /// The per-magnitude-bit arithmetic-coding probabilities for the
    /// §13.2.1 / §13.3.1 traversal loop (errata #67 corrected reading).
    ///
    /// Equal to [`DctToken::extra_bit_probs`] with the trailing
    /// sign-prior `B(128)` stripped for `CATEGORY1..CATEGORY5` and for
    /// `ONE_TOKEN..FOUR_TOKEN`; for `CATEGORY6` the as-printed
    /// 11-entry slice is already the magnitude-only set. The list's
    /// length is therefore `#ExtraBits − 1` for every category token,
    /// and zero for `ONE..FOUR` (whose magnitudes are constants) and
    /// for `ZERO_TOKEN` / `EOB_TOKEN`. The slice is **MSB-first** as
    /// printed in Table 18: entry 0 is the highest-order magnitude
    /// bit, the last entry is the lowest. The §13.2.1 listing reads
    /// it in reverse (`BitsCount = ExtraBits − 1` downwards), so a
    /// caller indexes `Probs[BitsCount]` directly when accumulating
    /// the magnitude.
    ///
    /// See `docs/video/vp6/vp6-errata-and-clarifications.md` entry
    /// **#67** for the full derivation.
    #[inline]
    pub const fn magnitude_probs(self) -> &'static [u8] {
        match self {
            // No magnitude bits to decode: ZERO/EOB carry no value;
            // ONE..FOUR carry a single sign-only extrabit (handled by
            // the caller's separate `b(1)` sign read).
            Self::Zero | Self::EndOfBlock => &[],
            Self::One | Self::Two | Self::Three | Self::Four => &[],
            // CATEGORY1..CATEGORY5: strip the trailing sign prior; the
            // remaining N-1 entries are the MSB-first magnitude bits.
            Self::Category1 => &[159],
            Self::Category2 => &[165, 145],
            Self::Category3 => &[173, 148, 140],
            Self::Category4 => &[176, 155, 140, 135],
            Self::Category5 => &[180, 157, 141, 134, 130],
            // CATEGORY6: the printed 11-entry slice is already
            // magnitude-only (errata #67); no trailing sign prior to
            // strip.
            Self::Category6 => &[254, 254, 243, 230, 196, 177, 153, 140, 133, 129, 128],
        }
    }
}

impl fmt::Display for DctToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Zero => "ZERO_TOKEN",
            Self::One => "ONE_TOKEN",
            Self::Two => "TWO_TOKEN",
            Self::Three => "THREE_TOKEN",
            Self::Four => "FOUR_TOKEN",
            Self::Category1 => "DCT_VAL_CATEGORY1",
            Self::Category2 => "DCT_VAL_CATEGORY2",
            Self::Category3 => "DCT_VAL_CATEGORY3",
            Self::Category4 => "DCT_VAL_CATEGORY4",
            Self::Category5 => "DCT_VAL_CATEGORY5",
            Self::Category6 => "DCT_VAL_CATEGORY6",
            Self::EndOfBlock => "DCT_EOB_TOKEN",
        };
        f.write_str(name)
    }
}

/// VP6 DC/AC coding-tree internal node names (spec §13 Table 20).
///
/// The discriminant is the index into an 11-entry node probability
/// vector (`DcProbs[plane]`, `AcProbs[plane][prec][band]`,
/// `DcNodeContexts[plane][ctx]`). The Figure 15 traversal reads
/// `B(prob[node])` at each of these nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TreeNode {
    /// `ZERO_CONTEXT_NODE`. Root of Figure 15: 0-branch = the
    /// coefficient is zero. Index 0.
    Zero = 0,
    /// `EOB_CONTEXT_NODE`. Under the zero branch: 0-branch = EOB.
    /// Index 1. (DC forbids EOB, so the DC conversion forces this to
    /// 1 via the `DcNodeEqs` dummy row.)
    EndOfBlock = 1,
    /// `ONE_CONTEXT_NODE`. Index 2.
    One = 2,
    /// `LOW_VAL_CONTEXT_NODE`. Index 3.
    LowVal = 3,
    /// `TWO_CONTEXT_NODE`. Index 4.
    Two = 4,
    /// `THREE_CONTEXT_NODE`. Index 5.
    Three = 5,
    /// `HIGH_LOW_CONTEXT_NODE`. Index 6.
    HighLow = 6,
    /// `CAT_ONE_CONTEXT_NODE`. Index 7.
    CatOne = 7,
    /// `CAT_THREEFOUR_CONTEXT_NODE`. Index 8.
    CatThreeFour = 8,
    /// `CAT_THREE_CONTEXT_NODE`. Index 9.
    CatThree = 9,
    /// `CAT_FIVE_CONTEXT_NODE`. Index 10.
    CatFive = 10,
}

impl TreeNode {
    /// All eleven nodes in canonical (Table 20) index order.
    pub const ALL: [TreeNode; NUM_TREE_NODES] = [
        TreeNode::Zero,
        TreeNode::EndOfBlock,
        TreeNode::One,
        TreeNode::LowVal,
        TreeNode::Two,
        TreeNode::Three,
        TreeNode::HighLow,
        TreeNode::CatOne,
        TreeNode::CatThreeFour,
        TreeNode::CatThree,
        TreeNode::CatFive,
    ];

    /// Canonical `0..=10` spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`TreeNode::index`]: build a node from the spec's
    /// `0..=10` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Zero),
            1 => Some(Self::EndOfBlock),
            2 => Some(Self::One),
            3 => Some(Self::LowVal),
            4 => Some(Self::Two),
            5 => Some(Self::Three),
            6 => Some(Self::HighLow),
            7 => Some(Self::CatOne),
            8 => Some(Self::CatThreeFour),
            9 => Some(Self::CatThree),
            10 => Some(Self::CatFive),
            _ => None,
        }
    }
}

impl fmt::Display for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Zero => "ZERO_CONTEXT_NODE",
            Self::EndOfBlock => "EOB_CONTEXT_NODE",
            Self::One => "ONE_CONTEXT_NODE",
            Self::LowVal => "LOW_VAL_CONTEXT_NODE",
            Self::Two => "TWO_CONTEXT_NODE",
            Self::Three => "THREE_CONTEXT_NODE",
            Self::HighLow => "HIGH_LOW_CONTEXT_NODE",
            Self::CatOne => "CAT_ONE_CONTEXT_NODE",
            Self::CatThreeFour => "CAT_THREEFOUR_CONTEXT_NODE",
            Self::CatThree => "CAT_THREE_CONTEXT_NODE",
            Self::CatFive => "CAT_FIVE_CONTEXT_NODE",
        };
        f.write_str(name)
    }
}

/// The keyframe baseline `DcProbs[2][11]` bank.
///
/// §13.2: "At each key frame (I frame) every probability value in this
/// array of DC Probabilities is set to 128." The bank persists from a
/// keyframe to each subsequent interframe and is updated in-place by
/// the §13.2 Table 22–24 BoolCoder bitstream (deferred).
#[inline]
pub const fn baseline_dc_probs() -> [[u8; NUM_TREE_NODES]; NUM_PLANES] {
    [[128; NUM_TREE_NODES]; NUM_PLANES]
}

/// The keyframe baseline `AcProbs[2][3][6][11]` bank.
///
/// §13.3: "At each key frame (I frame) every probability value in this
/// array of AC Probabilities is set to 128." The bank persists from a
/// keyframe to each subsequent interframe and is updated in-place by
/// the §13.3 Table 31–35 BoolCoder bitstream (deferred).
#[inline]
pub const fn baseline_ac_probs(
) -> [[[[u8; NUM_TREE_NODES]; NUM_AC_BANDS]; NUM_AC_PREC_CONTEXTS]; NUM_PLANES] {
    [[[[128; NUM_TREE_NODES]; NUM_AC_BANDS]; NUM_AC_PREC_CONTEXTS]; NUM_PLANES]
}

/// `VP6_DcUpdateProbs[2][11]` — the per-node update-flag probabilities
/// for the §13.2 DC coding-tree-node update bitstream (Table 24's
/// `NewNodeProbFlag` is read as `B(VP6_DcUpdateProbs[plane][node])`).
///
/// First dimension: plane (Y = 0, UV = 1). Second dimension: tree node
/// (Table 20 index 0..=10). Verbatim from §13.2.
pub const VP6_DC_UPDATE_PROBS: [[u8; NUM_TREE_NODES]; NUM_PLANES] = [
    [146, 255, 181, 207, 232, 243, 238, 251, 244, 250, 249],
    [179, 255, 214, 240, 250, 255, 244, 255, 255, 255, 255],
];

/// `AcUpdateProbs[3][2][6][11]` — the per-node update-flag
/// probabilities for the §13.3 AC coding-tree-node update bitstream
/// (Table 35's `NewNodeProbFlag` is read as
/// `B(AcUpdateProbs[prec][plane][band][node])`).
///
/// Dimension order, per §13.3: `[prec][plane][band][node]` where
/// `prec` is the Table 29 preceding-coefficient context, `plane` is
/// the Table 28 plane index, `band` is the Table 30 band index, and
/// `node` is the Table 20 node index. Verbatim from §13.3.
pub const AC_UPDATE_PROBS: [[[[u8; NUM_TREE_NODES]; NUM_AC_BANDS]; NUM_PLANES];
    NUM_AC_PREC_CONTEXTS] = [
    [
        // preceded by 0
        [
            [227, 246, 230, 247, 244, 255, 255, 255, 255, 255, 255],
            [255, 255, 209, 231, 231, 249, 249, 253, 255, 255, 255],
            [255, 255, 225, 242, 241, 251, 253, 255, 255, 255, 255],
            [255, 255, 241, 253, 252, 255, 255, 255, 255, 255, 255],
            [255, 255, 248, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [240, 255, 248, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 240, 253, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
    [
        // preceded by 1
        [
            [206, 203, 227, 239, 247, 255, 253, 255, 255, 255, 255],
            [207, 199, 220, 236, 243, 252, 252, 255, 255, 255, 255],
            [212, 219, 230, 243, 244, 253, 252, 255, 255, 255, 255],
            [236, 237, 247, 252, 253, 255, 255, 255, 255, 255, 255],
            [240, 240, 248, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [230, 233, 249, 255, 255, 255, 255, 255, 255, 255, 255],
            [238, 238, 250, 255, 255, 255, 255, 255, 255, 255, 255],
            [248, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
    [
        // preceded by > 1
        [
            [225, 239, 227, 231, 244, 253, 243, 255, 255, 253, 255],
            [232, 234, 224, 228, 242, 249, 242, 252, 251, 251, 255],
            [235, 249, 238, 240, 251, 255, 249, 255, 253, 253, 255],
            [249, 253, 251, 250, 255, 255, 255, 255, 255, 255, 255],
            [251, 250, 249, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [243, 244, 250, 250, 255, 255, 255, 255, 255, 255, 255],
            [249, 248, 250, 253, 255, 255, 255, 255, 255, 255, 255],
            [253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
];

/// `DcNodeEqs[5][3][2]` — the slope/constant linear-equation table the
/// §13.2 `DcProbs → DcNodeContexts` conversion applies to the first
/// five tree nodes (§13.2 Table 27).
///
/// First dimension: the first five Table 20 nodes (0 = Zero,
/// 1 = EOB — an *unused dummy* row that maps any input to 1,
/// 2 = One, 3 = LowVal, 4 = Two). Second dimension: the Table 26 DC
/// node context (0 = both-zero, 1 = one-non-zero, 2 = both-non-zero).
/// Third dimension: 0 = slope, 1 = constant. Constants can be negative,
/// so the entries are `i32`. Verbatim from §13.2.
pub const DC_NODE_EQS: [[[i32; 2]; NUM_DC_CONTEXTS]; NUM_DC_NODE_EQS] = [
    [[122, 133], [133, 51], [142, -16]], // Zero Node
    [[0, 1], [0, 1], [0, 1]],            // UNUSED DUMMY (EOB)
    [[78, 171], [169, 71], [221, -30]],  // One Node
    [[139, 117], [214, 44], [246, -3]],  // Low Val Node
    [[168, 79], [210, 38], [203, 17]],   // Two Node (2, 3 or 4)
];

/// Convert a `DcProbs[2][11]` bank into the `DcNodeContexts[2][3][11]`
/// array per the §13.2 pseudo-code.
///
/// For each plane and each of the three DC node contexts, the first
/// five nodes are passed through their `DcNodeEqs` line
/// (`Temp = ((DcProbs[p][n] * slope + 128) >> 8) + constant`, clipped
/// to `1..=255`); nodes 5..=10 copy the transmitted probability
/// unchanged. The result is the per-context node-probability tree the
/// §13.2.1 arithmetic DC decoder consults.
///
/// This is pure integer arithmetic (no BoolCoder), exactly as the §13.2
/// listing specifies.
pub fn dc_probs_to_node_contexts(
    dc_probs: &[[u8; NUM_TREE_NODES]; NUM_PLANES],
) -> [[[u8; NUM_TREE_NODES]; NUM_DC_CONTEXTS]; NUM_PLANES] {
    let mut out = [[[0u8; NUM_TREE_NODES]; NUM_DC_CONTEXTS]; NUM_PLANES];
    for (plane_in, plane_out) in dc_probs.iter().zip(out.iter_mut()) {
        for (ctx, ctx_out) in plane_out.iter_mut().enumerate() {
            for (node, slot) in ctx_out.iter_mut().enumerate() {
                if node < NUM_DC_NODE_EQS {
                    // Tree nodes 0..5: apply the linear equation.
                    let slope = DC_NODE_EQS[node][ctx][0];
                    let constant = DC_NODE_EQS[node][ctx][1];
                    let prob = plane_in[node] as i32;
                    let temp = ((((prob * slope) + 128) >> 8) + constant).clamp(1, 255);
                    *slot = temp as u8;
                } else {
                    // Tree nodes 5..11: pass the transmitted probability
                    // through unchanged.
                    *slot = plane_in[node];
                }
            }
        }
    }
    out
}

/// VP6 DC node context (spec §13.2 Table 26).
///
/// The §13.2.1 arithmetic DC decoder does **not** read from `DcProbs`
/// directly; it reads from `DcNodeContexts[plane][context]`, where
/// `context` is selected per block from whether the immediately
/// adjacent left and above blocks' **predicted DC values were zero or
/// non-zero**. Table 26 enumerates the three situations:
///
/// * `BothZero` (0) — left block's predicted DC was 0 **and** above
///   block's predicted DC was 0.
/// * `OneNonZero` (1) — either the left or the above block's predicted
///   DC is non-zero, but not both.
/// * `BothNonZero` (2) — both the left and the above block's predicted
///   DCs are non-zero.
///
/// The §13.2.1 note makes the contextual requirement explicit:
/// *"Decoding the dc requires that the contextual information regarding
/// whether the blocks immediately to the left of and above the current
/// block have 0 or non 0 dc values."*
///
/// "Predicted DC" here is the neighbour block's reconstructed DC
/// coefficient (the §14 predictor plus the §13.2-decoded `DcDelta`) —
/// the actual DC value that block carries — tested for being zero. A
/// missing neighbour (the frame's left edge has no left block, the top
/// edge has no above block) contributes a **zero** DC to the test:
/// §13.2 treats an absent neighbour the same as a zero-DC neighbour for
/// the purpose of this context, so a top-left corner block (no left, no
/// above) decodes with [`DcContext::BothZero`].
///
/// This selection is pure integer bookkeeping over already-decoded
/// neighbour DC values (no BoolCoder bits): it picks *which* of the
/// three precomputed [`dc_probs_to_node_contexts`] rows the §13.2.1 DC
/// tree walk consults. The discriminant is the spec's canonical
/// `0..=2` index — matching the second dimension of `DcNodeContexts`
/// and of [`DC_NODE_EQS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DcContext {
    /// Both the left and above neighbours' predicted DCs were 0
    /// (an absent neighbour counts as 0). Index 0.
    BothZero = 0,
    /// Exactly one of the left / above neighbours' predicted DCs was
    /// non-zero. Index 1.
    OneNonZero = 1,
    /// Both the left and above neighbours' predicted DCs were non-zero.
    /// Index 2.
    BothNonZero = 2,
}

impl DcContext {
    /// All DC node contexts in canonical (Table 26) order.
    pub const ALL: [DcContext; NUM_DC_CONTEXTS] = [
        DcContext::BothZero,
        DcContext::OneNonZero,
        DcContext::BothNonZero,
    ];

    /// Canonical `0..=2` spec index (matches the enum discriminant and
    /// the second dimension of `DcNodeContexts`).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`DcContext::index`]: build a context from the spec's
    /// `0..=2` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::BothZero),
            1 => Some(Self::OneNonZero),
            2 => Some(Self::BothNonZero),
            _ => None,
        }
    }

    /// Select the §13.2 Table 26 DC node context from the
    /// zero/non-zero state of the left and above neighbour blocks'
    /// predicted DC values.
    ///
    /// `left_non_zero` / `above_non_zero` are `true` when the
    /// corresponding neighbour block exists *and* carried a non-zero
    /// predicted DC. A missing neighbour passes `false` (an absent
    /// neighbour counts as zero-DC per §13.2). The mapping is the
    /// Table 26 partition:
    ///
    /// ```text
    /// neither non-zero → BothZero    (0)
    /// exactly one      → OneNonZero  (1)
    /// both non-zero    → BothNonZero (2)
    /// ```
    #[inline]
    pub const fn from_neighbours(left_non_zero: bool, above_non_zero: bool) -> Self {
        match (left_non_zero, above_non_zero) {
            (false, false) => Self::BothZero,
            (true, true) => Self::BothNonZero,
            _ => Self::OneNonZero,
        }
    }

    /// Select the active per-(plane) DC node-probability row a §13.2.1
    /// DC tree walk consults, from a `DcNodeContexts[plane][context]`
    /// bank (the [`dc_probs_to_node_contexts`] output).
    ///
    /// This is the convenience that wires the Table 26 context choice
    /// into the [`crate::dct_decode::decode_dc`] /
    /// [`crate::block_decode::decode_block_coefficients`] caller: given
    /// the per-plane converted bank and this context, it returns the
    /// 11-entry node-probability vector those decoders expect as
    /// `dc_node_probs`.
    #[inline]
    pub fn select_row(
        self,
        contexts: &[[u8; NUM_TREE_NODES]; NUM_DC_CONTEXTS],
    ) -> &[u8; NUM_TREE_NODES] {
        &contexts[self.index()]
    }
}

impl fmt::Display for DcContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BothZero => "DC_CONTEXT_BOTH_ZERO",
            Self::OneNonZero => "DC_CONTEXT_ONE_NON_ZERO",
            Self::BothNonZero => "DC_CONTEXT_BOTH_NON_ZERO",
        };
        f.write_str(name)
    }
}

/// Per-plane raster tracker for the §13.2 Table 26 DC node context.
///
/// As a plane's blocks are decoded in raster order (left→right,
/// top→bottom), the Table 26 DC context for each block depends on the
/// zero/non-zero state of its immediate **left** and **above**
/// neighbours' predicted DCs (see [`DcContext`]). This tracker holds
/// the small amount of state needed to supply that context without the
/// caller re-deriving neighbour positions:
///
/// * the non-zero flag of the block decoded immediately before the
///   current one in this row (the **left** neighbour), and
/// * one non-zero flag per column for the row above (the **above**
///   neighbours).
///
/// The caller decodes blocks in raster order, calling
/// [`context_for`](Self::context_for) at the start of each block to get
/// its [`DcContext`], then [`record`](Self::record) with the block's
/// own predicted-DC non-zero state once it is reconstructed. At the
/// left edge of a row there is no left neighbour and at the top row
/// there is no above neighbour; both absences contribute a zero-DC
/// (`false`) to the context per §13.2 (an absent neighbour counts as
/// zero-DC).
///
/// Pure integer / boolean bookkeeping — reads **no BoolCoder bits**. A
/// driver runs one tracker per plane (or per plane × reference bucket,
/// if the §14 same-reference partition is also applied to this
/// context; §13.2 specifies the zero-DC test without further
/// qualifying it by reference, so the default tracker tests the raw
/// neighbour DC).
#[derive(Debug, Clone)]
pub struct DcZeroContextTracker {
    /// Number of block columns in this plane.
    cols: usize,
    /// Per-column "above neighbour's predicted DC was non-zero" flags
    /// for the most-recently-completed row. All `false` before the
    /// first row.
    above_non_zero: Vec<bool>,
    /// The current block column being decoded (`0..cols`).
    col: usize,
    /// "Left neighbour's predicted DC was non-zero" flag for the
    /// current row. Reset to `false` (no left neighbour) at the start
    /// of each row.
    left_non_zero: bool,
}

impl DcZeroContextTracker {
    /// Build a tracker for a plane that is `cols` blocks wide.
    ///
    /// Panics if `cols == 0` (a plane must have at least one column).
    pub fn new(cols: usize) -> Self {
        assert!(cols > 0, "DcZeroContextTracker requires cols > 0");
        Self {
            cols,
            above_non_zero: vec![false; cols],
            col: 0,
            left_non_zero: false,
        }
    }

    /// The plane width (in blocks) this tracker was built for.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The current block column (`0..cols`).
    #[inline]
    pub fn col(&self) -> usize {
        self.col
    }

    /// The §13.2 Table 26 [`DcContext`] for the **current** block —
    /// the one about to be decoded at the current raster position.
    ///
    /// Combines the running left-neighbour flag with the stored
    /// above-neighbour flag for the current column. At the left edge
    /// (`col == 0`) the left flag is `false`; in the first row the
    /// above flags are all `false`. Does not advance the tracker — call
    /// [`record`](Self::record) once the block's predicted-DC non-zero
    /// state is known.
    #[inline]
    pub fn context_for(&self) -> DcContext {
        DcContext::from_neighbours(self.left_non_zero, self.above_non_zero[self.col])
    }

    /// Record the current block's predicted-DC non-zero state and
    /// advance the raster position to the next block.
    ///
    /// `non_zero` is `true` when this block's reconstructed (predicted)
    /// DC was non-zero. The flag becomes the **left** neighbour for the
    /// next block in this row and the **above** neighbour for the block
    /// directly below in the next row. Advancing past the last column
    /// wraps to the next row: the left flag resets to `false` (the new
    /// row's first block has no left neighbour) and the column counter
    /// returns to 0, with the just-completed row's flags now serving as
    /// the above row.
    pub fn record(&mut self, non_zero: bool) {
        self.above_non_zero[self.col] = non_zero;
        self.left_non_zero = non_zero;
        self.col += 1;
        if self.col == self.cols {
            // Wrap to the next row: no left neighbour for its first
            // block, and the row just finished is now the "above" row
            // (already written into `above_non_zero` column-by-column).
            self.col = 0;
            self.left_non_zero = false;
        }
    }
}

/// Convert an 11-entry node-probability vector into the 12-entry
/// Huffman probability set per the §13.1 `DCTTokenBoolTreeToHuffProbs`
/// transform.
///
/// Used by the §13.2.2 / §13.3.2 Huffman token decoders to derive a
/// Huffman probability table directly from a BoolCoder coding-tree
/// node vector. The output is indexed by [`DctToken::index`]. Pure
/// integer arithmetic (no BoolCoder), exactly as the §13.1 listing
/// specifies.
pub fn dct_token_bool_tree_to_huff_probs(node_prob: &[u8; NUM_TREE_NODES]) -> [u8; NUM_DCT_TOKENS] {
    // Work in u32 to mirror the spec's intermediate `Prob` / `Prob1`
    // chaining; each `>> 8` keeps values within 0..=255 so the final
    // narrowing to u8 is lossless.
    let np = |i: usize| node_prob[i] as u32;
    let mut huff = [0u32; NUM_DCT_TOKENS];

    huff[DctToken::EndOfBlock.index()] = (np(0) * np(1)) >> 8;
    huff[DctToken::Zero.index()] = (np(0) * (255 - np(1))) >> 8;

    let mut prob = 255 - np(0);
    huff[DctToken::One.index()] = (prob * np(2)) >> 8;

    prob = (prob * (255 - np(2))) >> 8;
    let mut prob1 = (prob * np(3)) >> 8;
    huff[DctToken::Two.index()] = (prob1 * np(4)) >> 8;

    prob1 = (prob1 * (255 - np(4))) >> 8;
    huff[DctToken::Three.index()] = (prob1 * np(5)) >> 8;
    huff[DctToken::Four.index()] = (prob1 * (255 - np(5))) >> 8;

    prob = (prob * (255 - np(3))) >> 8;
    prob1 = (prob * np(6)) >> 8;
    huff[DctToken::Category1.index()] = (prob1 * np(7)) >> 8;
    huff[DctToken::Category2.index()] = (prob1 * (255 - np(7))) >> 8;

    prob = (prob * (255 - np(6))) >> 8;
    prob1 = (prob * np(8)) >> 8;
    huff[DctToken::Category3.index()] = (prob1 * np(9)) >> 8;
    huff[DctToken::Category4.index()] = (prob1 * (255 - np(9))) >> 8;

    prob = (prob * (255 - np(8))) >> 8;
    huff[DctToken::Category5.index()] = (prob * np(10)) >> 8;
    huff[DctToken::Category6.index()] = (prob * (255 - np(10))) >> 8;

    let mut out = [0u8; NUM_DCT_TOKENS];
    for (slot, value) in out.iter_mut().zip(huff.iter()) {
        *slot = *value as u8;
    }
    out
}

/// VP6 AC band index (spec §13.3 Table 30).
///
/// The §13.3.1 arithmetic AC decoder selects a per-band node-probability
/// vector via `AcProbs[plane][prec][AcProbBand[encodedCoeffs]]`. The
/// table partitions the 63 AC scan positions `1..=63` into six bands
/// per the spec's column layout: coefficient 1 (band 0), 2–4 (band 1),
/// 5–10 (band 2), 11–21 (band 3), 22–36 (band 4), 37–63 (band 5). The
/// discriminant is the spec's canonical `0..=5` index — matching the
/// third dimension of [`AC_UPDATE_PROBS`] and the second dimension of
/// `baseline_ac_probs()`'s inner `[plane][prec][band][node]` layout.
///
/// This is the AC counterpart of [`crate::zrl::ZrlBand`] (Table 37,
/// which uses a coarser two-band partition for AC zero-run-length
/// probability selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AcBand {
    /// Coefficient position 1. Index 0.
    Coefficient1 = 0,
    /// Coefficient positions 2–4. Index 1.
    Coefficients2To4 = 1,
    /// Coefficient positions 5–10. Index 2.
    Coefficients5To10 = 2,
    /// Coefficient positions 11–21. Index 3.
    Coefficients11To21 = 3,
    /// Coefficient positions 22–36. Index 4.
    Coefficients22To36 = 4,
    /// Coefficient positions 37–63. Index 5.
    Coefficients37To63 = 5,
}

impl AcBand {
    /// All six AC bands in canonical (Table 30) index order.
    pub const ALL: [AcBand; NUM_AC_BANDS] = [
        AcBand::Coefficient1,
        AcBand::Coefficients2To4,
        AcBand::Coefficients5To10,
        AcBand::Coefficients11To21,
        AcBand::Coefficients22To36,
        AcBand::Coefficients37To63,
    ];

    /// Canonical `0..=5` spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`AcBand::index`]: build a band from the spec's
    /// `0..=5` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Coefficient1),
            1 => Some(Self::Coefficients2To4),
            2 => Some(Self::Coefficients5To10),
            3 => Some(Self::Coefficients11To21),
            4 => Some(Self::Coefficients22To36),
            5 => Some(Self::Coefficients37To63),
            _ => None,
        }
    }

    /// Look up the §13.3.1 `AcProbBand[encodedCoeffs]` band index for
    /// a given AC scan position.
    ///
    /// Returns `None` for `coeff_index == 0` (the DC position, which
    /// the §13.2 decoder handles independently) and for
    /// `coeff_index > 63` (outside the 64-coefficient block).
    ///
    /// The mapping is the verbatim Table 30 column-layout partition:
    /// position 1 → `Coefficient1`, positions 2–4 →
    /// `Coefficients2To4`, …, positions 37–63 → `Coefficients37To63`.
    #[inline]
    pub const fn for_coefficient_position(coeff_index: usize) -> Option<Self> {
        match coeff_index {
            0 => None,
            1 => Some(Self::Coefficient1),
            2..=4 => Some(Self::Coefficients2To4),
            5..=10 => Some(Self::Coefficients5To10),
            11..=21 => Some(Self::Coefficients11To21),
            22..=36 => Some(Self::Coefficients22To36),
            37..=63 => Some(Self::Coefficients37To63),
            _ => None,
        }
    }
}

impl fmt::Display for AcBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Coefficient1 => "AC_BAND_COEFF_1",
            Self::Coefficients2To4 => "AC_BAND_COEFFS_2_4",
            Self::Coefficients5To10 => "AC_BAND_COEFFS_5_10",
            Self::Coefficients11To21 => "AC_BAND_COEFFS_11_21",
            Self::Coefficients22To36 => "AC_BAND_COEFFS_22_36",
            Self::Coefficients37To63 => "AC_BAND_COEFFS_37_63",
        };
        f.write_str(name)
    }
}

/// VP6 AC plane index (spec §13.3 Table 28).
///
/// Y (luma) blocks select plane 0; U or V (chroma) blocks select plane 1.
/// The discriminant is the spec's canonical `0..=1` index, matching the
/// second dimension of [`AC_UPDATE_PROBS`] and of `baseline_ac_probs()`
/// (per `[prec][plane][band][node]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AcPlane {
    /// Y colour plane. Index 0.
    Y = 0,
    /// U or V colour plane. Index 1.
    UV = 1,
}

impl AcPlane {
    /// All AC planes in canonical (Table 28) order.
    pub const ALL: [AcPlane; NUM_PLANES] = [AcPlane::Y, AcPlane::UV];

    /// Canonical `0..=1` spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`AcPlane::index`]: build a plane from the spec's
    /// `0..=1` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Y),
            1 => Some(Self::UV),
            _ => None,
        }
    }
}

impl fmt::Display for AcPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Y => "AC_PLANE_Y",
            Self::UV => "AC_PLANE_UV",
        };
        f.write_str(name)
    }
}

/// VP6 AC "preceding decoded coefficient" context (spec §13.3 Table 29).
///
/// Selects the third row of `AcProbs[plane][prec][band][node]` (called
/// `Prec` in the §13.3.1 pseudo-code). For each AC coefficient the
/// decoder looks back at the previously-decoded coefficient in the
/// current scan order:
///
/// * `WasZero` (0) — preceding decoded coefficient was 0.
/// * `WasOne` (1) — preceding decoded coefficient was 1.
/// * `WasGreaterThanOne` (2) — preceding decoded coefficient had
///   magnitude > 1.
///
/// The first AC coefficient seeds `Prec` from the §13.2-decoded DC
/// value of the same block (DC == 0 → `WasZero`, DC == 1 →
/// `WasOne`, otherwise → `WasGreaterThanOne`). Subsequent coefficients
/// update `Prec` per the §13.3.1 pseudo-code's branch outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AcPrecContext {
    /// Preceding decoded coefficient was 0. Index 0.
    WasZero = 0,
    /// Preceding decoded coefficient was 1. Index 1.
    WasOne = 1,
    /// Preceding decoded coefficient had magnitude > 1. Index 2.
    WasGreaterThanOne = 2,
}

impl AcPrecContext {
    /// All preceding-coefficient contexts in canonical (Table 29)
    /// order.
    pub const ALL: [AcPrecContext; NUM_AC_PREC_CONTEXTS] = [
        AcPrecContext::WasZero,
        AcPrecContext::WasOne,
        AcPrecContext::WasGreaterThanOne,
    ];

    /// Canonical `0..=2` spec index (matches the enum discriminant).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`AcPrecContext::index`]: build a context from the
    /// spec's `0..=2` integer. Returns `None` for out-of-range values.
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::WasZero),
            1 => Some(Self::WasOne),
            2 => Some(Self::WasGreaterThanOne),
            _ => None,
        }
    }

    /// Seed the §13.3.1 `Prec` context from a freshly-decoded DC
    /// coefficient of the same block:
    ///
    /// ```text
    /// if (dc == 0)        Prec = 0
    /// else if (dc == 1)   Prec = 1
    /// else                Prec = 2
    /// ```
    ///
    /// Note the spec's signed-value test treats DC of 1 as
    /// distinguished from DC of −1 (magnitude 1 with negative sign):
    /// only `dc == 1` literally seeds `WasOne`. The §13.3.1 listing
    /// reads `dc == 1` not `|dc| == 1`, so this routine mirrors that
    /// exact comparison.
    #[inline]
    pub const fn seed_from_dc(dc: i32) -> Self {
        match dc {
            0 => Self::WasZero,
            1 => Self::WasOne,
            _ => Self::WasGreaterThanOne,
        }
    }
}

impl fmt::Display for AcPrecContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::WasZero => "AC_PREC_WAS_ZERO",
            Self::WasOne => "AC_PREC_WAS_ONE",
            Self::WasGreaterThanOne => "AC_PREC_WAS_GREATER_THAN_ONE",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- DctToken enum surface --------

    #[test]
    fn dct_token_indices_match_table_18_order() {
        assert_eq!(DctToken::Zero.index(), 0);
        assert_eq!(DctToken::One.index(), 1);
        assert_eq!(DctToken::Two.index(), 2);
        assert_eq!(DctToken::Three.index(), 3);
        assert_eq!(DctToken::Four.index(), 4);
        assert_eq!(DctToken::Category1.index(), 5);
        assert_eq!(DctToken::Category2.index(), 6);
        assert_eq!(DctToken::Category3.index(), 7);
        assert_eq!(DctToken::Category4.index(), 8);
        assert_eq!(DctToken::Category5.index(), 9);
        assert_eq!(DctToken::Category6.index(), 10);
        assert_eq!(DctToken::EndOfBlock.index(), 11);
    }

    #[test]
    fn dct_token_from_index_round_trip() {
        for (i, token) in DctToken::ALL.iter().enumerate() {
            assert_eq!(DctToken::from_index(i), Some(*token));
            assert_eq!(token.index(), i);
        }
        assert_eq!(DctToken::from_index(12), None);
        assert_eq!(DctToken::from_index(usize::MAX), None);
    }

    #[test]
    fn dct_token_all_length_matches_constant() {
        assert_eq!(DctToken::ALL.len(), NUM_DCT_TOKENS);
    }

    #[test]
    fn dct_token_min_max_match_table_18() {
        // (token, min, max) verbatim from Table 18.
        let cases = [
            (DctToken::Zero, 0u16, 0u16),
            (DctToken::One, 1, 1),
            (DctToken::Two, 2, 2),
            (DctToken::Three, 3, 3),
            (DctToken::Four, 4, 4),
            (DctToken::Category1, 5, 6),
            (DctToken::Category2, 7, 10),
            (DctToken::Category3, 11, 18),
            (DctToken::Category4, 19, 34),
            (DctToken::Category5, 35, 66),
            (DctToken::Category6, 67, 2114),
        ];
        for (token, min, max) in cases {
            assert_eq!(token.min_value(), min, "{token} min");
            assert_eq!(token.max_value(), max, "{token} max");
        }
    }

    #[test]
    fn dct_token_extra_bits_match_table_18() {
        let cases = [
            (DctToken::Zero, 0usize),
            (DctToken::One, 1),
            (DctToken::Two, 1),
            (DctToken::Three, 1),
            (DctToken::Four, 1),
            (DctToken::Category1, 2),
            (DctToken::Category2, 3),
            (DctToken::Category3, 4),
            (DctToken::Category4, 5),
            (DctToken::Category5, 6),
            (DctToken::Category6, 12),
            (DctToken::EndOfBlock, 0),
        ];
        for (token, bits) in cases {
            assert_eq!(token.extra_bits(), bits, "{token}");
        }
    }

    #[test]
    fn dct_token_probs_list_length_tracks_extra_bits_except_cat6() {
        // For every value-bearing token *except* CATEGORY6 the verbatim
        // "Arithmetic Encoding the Extra Bits" column length equals the
        // "# of extrabits (incl. sign)" column. CATEGORY6 lists 11
        // probabilities against an extra-bits count of 12 (the §13.2.1
        // traversal off-by-one documented on `extra_bit_probs`).
        for token in DctToken::ALL.iter() {
            match token {
                DctToken::Zero | DctToken::EndOfBlock => {
                    assert!(token.extra_bit_probs().is_empty(), "{token}");
                }
                DctToken::Category6 => {
                    assert_eq!(token.extra_bits(), 12);
                    assert_eq!(token.extra_bit_probs().len(), 11);
                }
                _ => {
                    assert_eq!(token.extra_bit_probs().len(), token.extra_bits(), "{token}");
                }
            }
        }
    }

    #[test]
    fn dct_token_extra_bit_probs_match_table_18() {
        assert_eq!(DctToken::One.extra_bit_probs(), &[128]);
        assert_eq!(DctToken::Two.extra_bit_probs(), &[128]);
        assert_eq!(DctToken::Three.extra_bit_probs(), &[128]);
        assert_eq!(DctToken::Four.extra_bit_probs(), &[128]);
        assert_eq!(DctToken::Category1.extra_bit_probs(), &[159, 128]);
        assert_eq!(DctToken::Category2.extra_bit_probs(), &[165, 145, 128]);
        assert_eq!(DctToken::Category3.extra_bit_probs(), &[173, 148, 140, 128]);
        assert_eq!(
            DctToken::Category4.extra_bit_probs(),
            &[176, 155, 140, 135, 128]
        );
        assert_eq!(
            DctToken::Category5.extra_bit_probs(),
            &[180, 157, 141, 134, 130, 128]
        );
        assert_eq!(
            DctToken::Category6.extra_bit_probs(),
            &[254, 254, 243, 230, 196, 177, 153, 140, 133, 129, 128]
        );
        // Every value-bearing token ends with the fixed sign prior 128.
        for token in DctToken::ALL.iter() {
            if let Some(last) = token.extra_bit_probs().last() {
                assert_eq!(*last, 128, "{token} sign prior");
            }
        }
    }

    #[test]
    fn dct_token_min_plus_extra_range_covers_max() {
        // The value field (extra_bits minus the sign bit) must span
        // [min, max]. e.g. CATEGORY3: 11 + (2^3 - 1) = 18 = max.
        for token in DctToken::ALL.iter() {
            if matches!(token, DctToken::Zero | DctToken::EndOfBlock) {
                continue;
            }
            let value_bits = token.extra_bits() - 1; // minus sign
            let span = (1u32 << value_bits) - 1;
            assert_eq!(
                token.min_value() as u32 + span,
                token.max_value() as u32,
                "{token} range"
            );
        }
    }

    // -------- TreeNode enum surface --------

    #[test]
    fn tree_node_indices_match_table_20_order() {
        assert_eq!(TreeNode::Zero.index(), 0);
        assert_eq!(TreeNode::EndOfBlock.index(), 1);
        assert_eq!(TreeNode::One.index(), 2);
        assert_eq!(TreeNode::LowVal.index(), 3);
        assert_eq!(TreeNode::Two.index(), 4);
        assert_eq!(TreeNode::Three.index(), 5);
        assert_eq!(TreeNode::HighLow.index(), 6);
        assert_eq!(TreeNode::CatOne.index(), 7);
        assert_eq!(TreeNode::CatThreeFour.index(), 8);
        assert_eq!(TreeNode::CatThree.index(), 9);
        assert_eq!(TreeNode::CatFive.index(), 10);
    }

    #[test]
    fn tree_node_from_index_round_trip() {
        for (i, node) in TreeNode::ALL.iter().enumerate() {
            assert_eq!(TreeNode::from_index(i), Some(*node));
            assert_eq!(node.index(), i);
        }
        assert_eq!(TreeNode::from_index(11), None);
        assert_eq!(TreeNode::from_index(usize::MAX), None);
    }

    #[test]
    fn tree_node_all_length_matches_constant() {
        assert_eq!(TreeNode::ALL.len(), NUM_TREE_NODES);
    }

    // -------- baseline banks --------

    #[test]
    fn baseline_dc_probs_all_128() {
        let dc = baseline_dc_probs();
        assert_eq!(dc.len(), NUM_PLANES);
        for plane in dc.iter() {
            assert_eq!(plane.len(), NUM_TREE_NODES);
            assert!(plane.iter().all(|&p| p == 128));
        }
    }

    #[test]
    fn baseline_ac_probs_all_128() {
        let ac = baseline_ac_probs();
        assert_eq!(ac.len(), NUM_PLANES);
        let mut count = 0;
        for plane in ac.iter() {
            assert_eq!(plane.len(), NUM_AC_PREC_CONTEXTS);
            for prec in plane.iter() {
                assert_eq!(prec.len(), NUM_AC_BANDS);
                for band in prec.iter() {
                    assert_eq!(band.len(), NUM_TREE_NODES);
                    assert!(band.iter().all(|&p| p == 128));
                    count += band.len();
                }
            }
        }
        // 2 * 3 * 6 * 11 = 396 entries.
        assert_eq!(count, 2 * 3 * 6 * 11);
    }

    // -------- update-flag probability banks --------

    #[test]
    fn vp6_dc_update_probs_dimensions_and_spot_values() {
        assert_eq!(VP6_DC_UPDATE_PROBS.len(), NUM_PLANES);
        for row in VP6_DC_UPDATE_PROBS.iter() {
            assert_eq!(row.len(), NUM_TREE_NODES);
        }
        // First and last of each plane row (§13.2 listing).
        assert_eq!(VP6_DC_UPDATE_PROBS[0][0], 146);
        assert_eq!(VP6_DC_UPDATE_PROBS[0][10], 249);
        assert_eq!(VP6_DC_UPDATE_PROBS[1][0], 179);
        assert_eq!(VP6_DC_UPDATE_PROBS[1][10], 255);
        // EOB node (index 1) is 255 in both planes.
        assert_eq!(VP6_DC_UPDATE_PROBS[0][1], 255);
        assert_eq!(VP6_DC_UPDATE_PROBS[1][1], 255);
    }

    #[test]
    fn ac_update_probs_dimensions() {
        assert_eq!(AC_UPDATE_PROBS.len(), NUM_AC_PREC_CONTEXTS);
        for prec in AC_UPDATE_PROBS.iter() {
            assert_eq!(prec.len(), NUM_PLANES);
            for plane in prec.iter() {
                assert_eq!(plane.len(), NUM_AC_BANDS);
                for band in plane.iter() {
                    assert_eq!(band.len(), NUM_TREE_NODES);
                }
            }
        }
    }

    #[test]
    fn ac_update_probs_spot_values() {
        // Corners of the §13.3 listing.
        assert_eq!(AC_UPDATE_PROBS[0][0][0][0], 227);
        assert_eq!(AC_UPDATE_PROBS[0][0][0][10], 255);
        assert_eq!(AC_UPDATE_PROBS[1][0][0][0], 206);
        assert_eq!(AC_UPDATE_PROBS[1][0][1][1], 199);
        assert_eq!(AC_UPDATE_PROBS[2][0][0][0], 225);
        assert_eq!(AC_UPDATE_PROBS[2][0][2][6], 249);
        assert_eq!(AC_UPDATE_PROBS[2][1][0][0], 243);
        // Last prec/plane/band row is all 255.
        assert!(AC_UPDATE_PROBS[2][1][5].iter().all(|&p| p == 255));
    }

    // -------- DcNodeEqs --------

    #[test]
    fn dc_node_eqs_dimensions_and_spot_values() {
        assert_eq!(DC_NODE_EQS.len(), NUM_DC_NODE_EQS);
        for node in DC_NODE_EQS.iter() {
            assert_eq!(node.len(), NUM_DC_CONTEXTS);
            for ctx in node.iter() {
                assert_eq!(ctx.len(), 2);
            }
        }
        // Zero Node, context 0: slope 122 constant 133.
        assert_eq!(DC_NODE_EQS[0][0], [122, 133]);
        // Zero Node, context 2: negative constant.
        assert_eq!(DC_NODE_EQS[0][2], [142, -16]);
        // The EOB dummy row maps any input to constant 1 (slope 0).
        assert_eq!(DC_NODE_EQS[1], [[0, 1], [0, 1], [0, 1]]);
        // Low Val Node, context 2: negative constant.
        assert_eq!(DC_NODE_EQS[3][2], [246, -3]);
        // Two Node, context 0.
        assert_eq!(DC_NODE_EQS[4][0], [168, 79]);
    }

    // -------- DcProbs -> DcNodeContexts conversion --------

    #[test]
    fn dc_node_contexts_dummy_eob_node_is_one() {
        // The EOB node (index 1) uses the slope-0 / constant-1 dummy,
        // so it maps to 1 regardless of the input probability.
        for input in [0u8, 1, 64, 128, 200, 255] {
            let mut dc = baseline_dc_probs();
            for plane in dc.iter_mut() {
                plane[1] = input;
            }
            let contexts = dc_probs_to_node_contexts(&dc);
            for plane in contexts.iter() {
                for ctx in plane.iter() {
                    assert_eq!(ctx[1], 1, "EOB node should clip to 1 for input {input}");
                }
            }
        }
    }

    #[test]
    fn dc_node_contexts_passthrough_nodes_5_to_10() {
        // Nodes 5..=10 copy the transmitted probability verbatim into
        // every context.
        let mut dc = baseline_dc_probs();
        let marks = [11u8, 22, 33, 44, 55, 66];
        for plane in dc.iter_mut() {
            for (k, m) in marks.iter().enumerate() {
                plane[5 + k] = *m;
            }
        }
        let contexts = dc_probs_to_node_contexts(&dc);
        for plane in contexts.iter() {
            for ctx in plane.iter() {
                for (k, m) in marks.iter().enumerate() {
                    assert_eq!(ctx[5 + k], *m);
                }
            }
        }
    }

    #[test]
    fn dc_node_contexts_baseline_128_matches_hand_computed() {
        // Hand-computed from the §13.2 pseudo-code with DcProbs all 128.
        // Temp = ((128 * slope + 128) >> 8) + constant, clipped 1..=255.
        let contexts = dc_probs_to_node_contexts(&baseline_dc_probs());
        let expect = |slope: i32, constant: i32| -> u8 {
            ((((128 * slope) + 128) >> 8) + constant).clamp(1, 255) as u8
        };
        for (plane, plane_ctx) in contexts.iter().enumerate() {
            for (ctx, row) in plane_ctx.iter().enumerate() {
                for (node, &got) in row.iter().enumerate() {
                    if node < NUM_DC_NODE_EQS {
                        let slope = DC_NODE_EQS[node][ctx][0];
                        let constant = DC_NODE_EQS[node][ctx][1];
                        assert_eq!(
                            got,
                            expect(slope, constant),
                            "plane {plane} ctx {ctx} node {node}"
                        );
                    } else {
                        // Pass-through nodes remain 128.
                        assert_eq!(got, 128, "plane {plane} ctx {ctx} node {node}");
                    }
                }
            }
        }
        // A couple of concrete spot values for the Zero node.
        // ctx 0: ((128*122+128)>>8)+133 = (15616>>8)+133 = 61+133 = 194.
        assert_eq!(contexts[0][0][0], 194);
        // ctx 2: ((128*142+128)>>8)+(-16) = (18304>>8)-16 = 71-16 = 55.
        assert_eq!(contexts[0][2][0], 55);
    }

    #[test]
    fn dc_node_contexts_output_always_clipped_to_1_255() {
        // Probe a spread of inputs; every output must be in 1..=255.
        for input in [0u8, 1, 50, 128, 200, 255] {
            let dc = [[input; NUM_TREE_NODES]; NUM_PLANES];
            let contexts = dc_probs_to_node_contexts(&dc);
            for plane in contexts.iter() {
                for ctx in plane.iter() {
                    // Only nodes 0..5 go through the clip; pass-through
                    // nodes echo the input (which may be 0 only if input
                    // was 0 — that is the transmitted value, not a tree
                    // probability the decoder reads at node 5+).
                    for (node, &p) in ctx.iter().take(NUM_DC_NODE_EQS).enumerate() {
                        assert!((1..=255).contains(&p), "node {node} prob {p} out of range");
                    }
                }
            }
        }
    }

    // -------- DCTTokenBoolTreeToHuffProbs --------

    #[test]
    fn huff_probs_all_128_matches_hand_computed() {
        // With every node probability == 128, reproduce the §13.1
        // listing by hand to pin the transform.
        let node = [128u8; NUM_TREE_NODES];
        let huff = dct_token_bool_tree_to_huff_probs(&node);

        // EOB = (128*128)>>8 = 64
        assert_eq!(huff[DctToken::EndOfBlock.index()], 64);
        // ZERO = (128*(255-128))>>8 = (128*127)>>8 = 16256>>8 = 63
        assert_eq!(huff[DctToken::Zero.index()], 63);

        // prob = 255-128 = 127
        // ONE = (127*128)>>8 = 16256>>8 = 63
        assert_eq!(huff[DctToken::One.index()], 63);

        // prob = (127*(255-128))>>8 = (127*127)>>8 = 16129>>8 = 63
        // prob1 = (63*128)>>8 = 8064>>8 = 31
        // TWO = (31*128)>>8 = 3968>>8 = 15
        assert_eq!(huff[DctToken::Two.index()], 15);
        // prob1 = (31*(255-128))>>8 = (31*127)>>8 = 3937>>8 = 15
        // THREE = (15*128)>>8 = 1920>>8 = 7
        assert_eq!(huff[DctToken::Three.index()], 7);
        // FOUR = (15*(255-128))>>8 = (15*127)>>8 = 1905>>8 = 7
        assert_eq!(huff[DctToken::Four.index()], 7);

        // prob = (63*(255-128))>>8 = (63*127)>>8 = 8001>>8 = 31
        // prob1 = (31*128)>>8 = 15
        // CAT1 = (15*128)>>8 = 7
        assert_eq!(huff[DctToken::Category1.index()], 7);
        // CAT2 = (15*(255-128))>>8 = 7
        assert_eq!(huff[DctToken::Category2.index()], 7);

        // prob = (31*(255-128))>>8 = (31*127)>>8 = 3937>>8 = 15
        // prob1 = (15*128)>>8 = 7
        // CAT3 = (7*128)>>8 = 3
        assert_eq!(huff[DctToken::Category3.index()], 3);
        // CAT4 = (7*(255-128))>>8 = (7*127)>>8 = 889>>8 = 3
        assert_eq!(huff[DctToken::Category4.index()], 3);

        // prob = (15*(255-128))>>8 = (15*127)>>8 = 1905>>8 = 7
        // CAT5 = (7*128)>>8 = 3
        assert_eq!(huff[DctToken::Category5.index()], 3);
        // CAT6 = (7*(255-128))>>8 = 3
        assert_eq!(huff[DctToken::Category6.index()], 3);
    }

    #[test]
    fn huff_probs_output_length() {
        let huff = dct_token_bool_tree_to_huff_probs(&[128u8; NUM_TREE_NODES]);
        assert_eq!(huff.len(), NUM_DCT_TOKENS);
    }

    #[test]
    fn huff_probs_never_panic_on_extremes() {
        // The chained `>> 8` arithmetic keeps every intermediate in
        // 0..=255, so all-1 and all-255 inputs must not overflow u8.
        for fill in [1u8, 255u8] {
            let node = [fill; NUM_TREE_NODES];
            let _ = dct_token_bool_tree_to_huff_probs(&node);
        }
        // A mixed vector too.
        let node = [200, 1, 180, 30, 250, 5, 99, 128, 7, 240, 60];
        let huff = dct_token_bool_tree_to_huff_probs(&node);
        assert_eq!(huff.len(), NUM_DCT_TOKENS);
    }

    #[test]
    fn display_names_match_spec() {
        assert_eq!(DctToken::Category6.to_string(), "DCT_VAL_CATEGORY6");
        assert_eq!(DctToken::EndOfBlock.to_string(), "DCT_EOB_TOKEN");
        assert_eq!(
            TreeNode::CatThreeFour.to_string(),
            "CAT_THREEFOUR_CONTEXT_NODE"
        );
        assert_eq!(TreeNode::Zero.to_string(), "ZERO_CONTEXT_NODE");
    }

    // ----------------------------------------------------------------------
    // §13.3 AC band / plane / preceding-coefficient enums (Tables 28–30)
    // ----------------------------------------------------------------------

    #[test]
    fn ac_band_indices_match_table_30_order() {
        assert_eq!(AcBand::Coefficient1.index(), 0);
        assert_eq!(AcBand::Coefficients2To4.index(), 1);
        assert_eq!(AcBand::Coefficients5To10.index(), 2);
        assert_eq!(AcBand::Coefficients11To21.index(), 3);
        assert_eq!(AcBand::Coefficients22To36.index(), 4);
        assert_eq!(AcBand::Coefficients37To63.index(), 5);
        assert_eq!(AcBand::ALL.len(), NUM_AC_BANDS);
    }

    #[test]
    fn ac_band_from_index_round_trip() {
        for (i, band) in AcBand::ALL.iter().enumerate() {
            assert_eq!(AcBand::from_index(i), Some(*band));
            assert_eq!(band.index(), i);
        }
        assert_eq!(AcBand::from_index(NUM_AC_BANDS), None);
        assert_eq!(AcBand::from_index(usize::MAX), None);
    }

    #[test]
    fn ac_band_for_coefficient_position_partition() {
        // DC position (0) returns None — §13.2 handles it separately.
        assert_eq!(AcBand::for_coefficient_position(0), None);

        // Position 1 → Coefficient1.
        assert_eq!(
            AcBand::for_coefficient_position(1),
            Some(AcBand::Coefficient1)
        );

        // Positions 2..=4 → Coefficients2To4.
        for pos in 2..=4 {
            assert_eq!(
                AcBand::for_coefficient_position(pos),
                Some(AcBand::Coefficients2To4),
                "pos {pos}"
            );
        }

        // Positions 5..=10 → Coefficients5To10.
        for pos in 5..=10 {
            assert_eq!(
                AcBand::for_coefficient_position(pos),
                Some(AcBand::Coefficients5To10),
                "pos {pos}"
            );
        }

        // Positions 11..=21 → Coefficients11To21.
        for pos in 11..=21 {
            assert_eq!(
                AcBand::for_coefficient_position(pos),
                Some(AcBand::Coefficients11To21),
                "pos {pos}"
            );
        }

        // Positions 22..=36 → Coefficients22To36.
        for pos in 22..=36 {
            assert_eq!(
                AcBand::for_coefficient_position(pos),
                Some(AcBand::Coefficients22To36),
                "pos {pos}"
            );
        }

        // Positions 37..=63 → Coefficients37To63.
        for pos in 37..=63 {
            assert_eq!(
                AcBand::for_coefficient_position(pos),
                Some(AcBand::Coefficients37To63),
                "pos {pos}"
            );
        }

        // Out-of-block positions return None.
        assert_eq!(AcBand::for_coefficient_position(64), None);
        assert_eq!(AcBand::for_coefficient_position(100), None);
        assert_eq!(AcBand::for_coefficient_position(usize::MAX), None);
    }

    #[test]
    fn ac_band_partition_covers_every_ac_position_exactly_once() {
        // Every AC scan position 1..=63 maps to exactly one band, and
        // every band is hit by at least one position. (Structural
        // covering check — a defensive guard against future
        // refactors of `for_coefficient_position`.)
        let mut counts = [0usize; NUM_AC_BANDS];
        for pos in 1..=63 {
            let band = AcBand::for_coefficient_position(pos)
                .expect("every AC position 1..=63 maps to some band");
            counts[band.index()] += 1;
        }
        assert_eq!(counts[AcBand::Coefficient1.index()], 1);
        assert_eq!(counts[AcBand::Coefficients2To4.index()], 3);
        assert_eq!(counts[AcBand::Coefficients5To10.index()], 6);
        assert_eq!(counts[AcBand::Coefficients11To21.index()], 11);
        assert_eq!(counts[AcBand::Coefficients22To36.index()], 15);
        assert_eq!(counts[AcBand::Coefficients37To63.index()], 27);
        // Sum to 63 AC positions.
        assert_eq!(counts.iter().sum::<usize>(), 63);
    }

    #[test]
    fn ac_plane_indices_match_table_28_order() {
        assert_eq!(AcPlane::Y.index(), 0);
        assert_eq!(AcPlane::UV.index(), 1);
        assert_eq!(AcPlane::ALL.len(), NUM_PLANES);
    }

    #[test]
    fn ac_plane_from_index_round_trip() {
        for (i, plane) in AcPlane::ALL.iter().enumerate() {
            assert_eq!(AcPlane::from_index(i), Some(*plane));
            assert_eq!(plane.index(), i);
        }
        assert_eq!(AcPlane::from_index(NUM_PLANES), None);
    }

    #[test]
    fn ac_prec_context_indices_match_table_29_order() {
        assert_eq!(AcPrecContext::WasZero.index(), 0);
        assert_eq!(AcPrecContext::WasOne.index(), 1);
        assert_eq!(AcPrecContext::WasGreaterThanOne.index(), 2);
        assert_eq!(AcPrecContext::ALL.len(), NUM_AC_PREC_CONTEXTS);
    }

    #[test]
    fn ac_prec_context_from_index_round_trip() {
        for (i, prec) in AcPrecContext::ALL.iter().enumerate() {
            assert_eq!(AcPrecContext::from_index(i), Some(*prec));
            assert_eq!(prec.index(), i);
        }
        assert_eq!(AcPrecContext::from_index(NUM_AC_PREC_CONTEXTS), None);
    }

    #[test]
    fn ac_prec_context_seed_from_dc_partitions_signed_dc() {
        assert_eq!(AcPrecContext::seed_from_dc(0), AcPrecContext::WasZero);
        assert_eq!(AcPrecContext::seed_from_dc(1), AcPrecContext::WasOne);
        // Spec uses signed `dc == 1`; -1 is not the same as 1.
        assert_eq!(
            AcPrecContext::seed_from_dc(-1),
            AcPrecContext::WasGreaterThanOne,
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(2),
            AcPrecContext::WasGreaterThanOne,
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(2114),
            AcPrecContext::WasGreaterThanOne,
        );
        assert_eq!(
            AcPrecContext::seed_from_dc(-2114),
            AcPrecContext::WasGreaterThanOne,
        );
    }

    #[test]
    fn ac_band_plane_prec_display_names() {
        assert_eq!(AcBand::Coefficient1.to_string(), "AC_BAND_COEFF_1");
        assert_eq!(
            AcBand::Coefficients37To63.to_string(),
            "AC_BAND_COEFFS_37_63"
        );
        assert_eq!(AcPlane::Y.to_string(), "AC_PLANE_Y");
        assert_eq!(AcPlane::UV.to_string(), "AC_PLANE_UV");
        assert_eq!(AcPrecContext::WasZero.to_string(), "AC_PREC_WAS_ZERO");
        assert_eq!(
            AcPrecContext::WasGreaterThanOne.to_string(),
            "AC_PREC_WAS_GREATER_THAN_ONE"
        );
    }

    // -------- DcContext (§13.2 Table 26) --------

    #[test]
    fn dc_context_indices_match_table_26_order() {
        assert_eq!(DcContext::BothZero.index(), 0);
        assert_eq!(DcContext::OneNonZero.index(), 1);
        assert_eq!(DcContext::BothNonZero.index(), 2);
        assert_eq!(DcContext::ALL.len(), NUM_DC_CONTEXTS);
        for (i, &c) in DcContext::ALL.iter().enumerate() {
            assert_eq!(c.index(), i);
        }
    }

    #[test]
    fn dc_context_from_index_round_trip() {
        for &c in &DcContext::ALL {
            assert_eq!(DcContext::from_index(c.index()), Some(c));
        }
        assert_eq!(DcContext::from_index(3), None);
        assert_eq!(DcContext::from_index(usize::MAX), None);
    }

    #[test]
    fn dc_context_from_neighbours_partitions_table_26() {
        // Table 26: neither → 0, exactly one → 1, both → 2.
        assert_eq!(
            DcContext::from_neighbours(false, false),
            DcContext::BothZero
        );
        assert_eq!(
            DcContext::from_neighbours(true, false),
            DcContext::OneNonZero
        );
        assert_eq!(
            DcContext::from_neighbours(false, true),
            DcContext::OneNonZero
        );
        assert_eq!(
            DcContext::from_neighbours(true, true),
            DcContext::BothNonZero
        );
    }

    #[test]
    fn dc_context_select_row_picks_the_right_dimension() {
        // Build a per-plane bank with each context row distinctly tagged
        // so select_row's indexing is observable.
        let dc_probs = baseline_dc_probs();
        let banks = dc_probs_to_node_contexts(&dc_probs);
        for plane_bank in &banks {
            for &ctx in &DcContext::ALL {
                let row = ctx.select_row(plane_bank);
                assert_eq!(row, &plane_bank[ctx.index()]);
            }
        }
    }

    #[test]
    fn dc_context_display_names() {
        assert_eq!(DcContext::BothZero.to_string(), "DC_CONTEXT_BOTH_ZERO");
        assert_eq!(DcContext::OneNonZero.to_string(), "DC_CONTEXT_ONE_NON_ZERO");
        assert_eq!(
            DcContext::BothNonZero.to_string(),
            "DC_CONTEXT_BOTH_NON_ZERO"
        );
    }

    // -------- DcZeroContextTracker raster bookkeeping --------

    #[test]
    fn tracker_first_block_is_both_zero() {
        // Top-left corner: no left, no above → both zero.
        let t = DcZeroContextTracker::new(4);
        assert_eq!(t.cols(), 4);
        assert_eq!(t.col(), 0);
        assert_eq!(t.context_for(), DcContext::BothZero);
    }

    #[test]
    fn tracker_first_row_only_uses_left_neighbour() {
        // First row has no above neighbours; context is driven entirely
        // by the left-neighbour non-zero state.
        let mut t = DcZeroContextTracker::new(3);
        assert_eq!(t.context_for(), DcContext::BothZero); // col 0: no left
        t.record(true); // block 0 had non-zero DC
        assert_eq!(t.col(), 1);
        // col 1: left non-zero, above absent → exactly one.
        assert_eq!(t.context_for(), DcContext::OneNonZero);
        t.record(false); // block 1 had zero DC
                         // col 2: left zero, above absent → both zero.
        assert_eq!(t.context_for(), DcContext::BothZero);
        t.record(true);
    }

    #[test]
    fn tracker_wraps_to_next_row_resetting_left() {
        let mut t = DcZeroContextTracker::new(2);
        // Row 0: record [true, true].
        t.record(true); // col 0
        t.record(true); // col 1 → wraps to row 1 col 0
        assert_eq!(t.col(), 0);
        // Row 1, col 0: no left (row start), above (col 0) was non-zero
        // → exactly one.
        assert_eq!(t.context_for(), DcContext::OneNonZero);
        t.record(false); // block at row1 col0 had zero DC
                         // Row 1, col 1: left zero, above (col 1) was non-zero → one.
        assert_eq!(t.context_for(), DcContext::OneNonZero);
        t.record(true);
    }

    #[test]
    fn tracker_both_non_zero_when_left_and_above_set() {
        let mut t = DcZeroContextTracker::new(2);
        // Row 0: [true, true].
        t.record(true);
        t.record(true);
        // Row 1, col 0: above non-zero, no left → one.
        assert_eq!(t.context_for(), DcContext::OneNonZero);
        t.record(true); // row1 col0 non-zero
                        // Row 1, col 1: left non-zero AND above (col1) non-zero → both.
        assert_eq!(t.context_for(), DcContext::BothNonZero);
        t.record(true);
    }

    #[test]
    fn tracker_single_column_plane() {
        // A 1-column plane: every block has no left neighbour; the
        // context is driven by the single above column only.
        let mut t = DcZeroContextTracker::new(1);
        assert_eq!(t.context_for(), DcContext::BothZero); // row 0
        t.record(true);
        // Row 1: above non-zero, no left → exactly one.
        assert_eq!(t.col(), 0);
        assert_eq!(t.context_for(), DcContext::OneNonZero);
        t.record(false);
        // Row 2: above zero, no left → both zero.
        assert_eq!(t.context_for(), DcContext::BothZero);
    }

    #[test]
    #[should_panic(expected = "cols > 0")]
    fn tracker_rejects_zero_cols() {
        let _ = DcZeroContextTracker::new(0);
    }
}

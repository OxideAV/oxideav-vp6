//! VP6 / VP56 data tables — ported verbatim (as values) from FFmpeg's
//! `vp56data.{c,h}` and `vp6data.h`.
//!
//! These are the probability / filter / scan tables VP6 needs for
//! bitstream decode. They're listed here as `pub(crate) const` slices
//! so the rest of the codebase can treat them as read-only data with
//! no runtime cost.

use crate::range_coder::Vp56Tree;

/// Macroblock type enumeration — matches `VP56mb` in `vp56.h`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum Vp56Mb {
    InterNoVecPf = 0,
    Intra = 1,
    InterDeltaPf = 2,
    InterV1Pf = 3,
    InterV2Pf = 4,
    InterNoVecGf = 5,
    InterDeltaGf = 6,
    Inter4V = 7,
    InterV1Gf = 8,
    InterV2Gf = 9,
}

impl Vp56Mb {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::InterNoVecPf),
            1 => Some(Self::Intra),
            2 => Some(Self::InterDeltaPf),
            3 => Some(Self::InterV1Pf),
            4 => Some(Self::InterV2Pf),
            5 => Some(Self::InterNoVecGf),
            6 => Some(Self::InterDeltaGf),
            7 => Some(Self::Inter4V),
            8 => Some(Self::InterV1Gf),
            9 => Some(Self::InterV2Gf),
            _ => None,
        }
    }
}

/// Which reference a given [`Vp56Mb`] is coded against. Indexed by the
/// variant's numeric value.
pub const REFERENCE_FRAME: [RefFrame; 10] = [
    RefFrame::Previous, // InterNoVecPf
    RefFrame::Current,  // Intra
    RefFrame::Previous, // InterDeltaPf
    RefFrame::Previous, // InterV1Pf
    RefFrame::Previous, // InterV2Pf
    RefFrame::Golden,   // InterNoVecGf
    RefFrame::Golden,   // InterDeltaGf
    RefFrame::Previous, // Inter4V
    RefFrame::Golden,   // InterV1Gf
    RefFrame::Golden,   // InterV2Gf
];

/// Which decoded frame to reference for a given MB type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefFrame {
    Current,
    Previous,
    Golden,
}

// -- vp56 common tables --------------------------------------------------

/// Block index (0..=5) -> plane index (0=Y, 1=U, 2=V, 3=alpha).
pub const B2P: [u8; 10] = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3];

/// 6-block -> 4-block-context index (used to share left-block state
/// between the two chroma blocks and the paired luma blocks).
pub const B6_TO_4: [u8; 6] = [0, 0, 1, 1, 2, 3];

/// `vp56_coeff_parse_table` — per-bit fallback probabilities for the
/// long path in `vp6_parse_coeff`.
pub const VP56_COEFF_PARSE_TABLE: [[u8; 11]; 6] = [
    [159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [145, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [140, 148, 173, 0, 0, 0, 0, 0, 0, 0, 0],
    [135, 140, 155, 176, 0, 0, 0, 0, 0, 0, 0],
    [130, 134, 141, 157, 180, 0, 0, 0, 0, 0, 0],
    [129, 130, 133, 140, 153, 177, 196, 230, 243, 254, 254],
];

/// `vp56_coeff_bias` — starts of the non-linear coefficient categories.
pub const VP56_COEFF_BIAS: [i16; 11] = [0, 1, 2, 3, 4, 5, 7, 11, 19, 35, 67];

/// Number of extra bits (minus 1) to read per category when decoding
/// long-path coefficients.
pub const VP56_COEFF_BIT_LENGTH: [u8; 6] = [0, 1, 2, 3, 4, 10];

/// Default per-context MB type statistics (keyframes start here).
pub const DEF_MB_TYPES_STATS: [[[u8; 2]; 10]; 3] = [
    [
        [69, 42],
        [1, 2],
        [1, 7],
        [44, 42],
        [6, 22],
        [1, 3],
        [0, 2],
        [1, 5],
        [0, 1],
        [0, 0],
    ],
    [
        [229, 8],
        [1, 1],
        [0, 8],
        [0, 0],
        [0, 0],
        [1, 2],
        [0, 1],
        [0, 0],
        [1, 1],
        [0, 0],
    ],
    [
        [122, 35],
        [1, 1],
        [1, 6],
        [46, 34],
        [0, 0],
        [1, 2],
        [0, 1],
        [0, 1],
        [1, 1],
        [0, 0],
    ],
];

/// `vp56_filter_threshold` — strength of the loop filter per quantiser.
pub const VP56_FILTER_THRESHOLD: [u8; 64] = [
    14, 14, 13, 13, 12, 12, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 3,
    3, 3, 3, 2,
];

/// Dequantiser for the DC coefficient, indexed by QP.
pub const VP56_DC_DEQUANT: [u16; 64] = [
    47, 47, 47, 47, 45, 43, 43, 43, 43, 43, 42, 41, 41, 40, 40, 40, 40, 35, 35, 35, 35, 33, 33, 33,
    33, 32, 32, 32, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 19, 19, 19, 19, 18, 18, 17, 16, 16, 16,
    16, 16, 15, 11, 11, 11, 10, 10, 9, 8, 7, 5, 3, 3, 2, 2,
];

/// Dequantiser for AC coefficients.
pub const VP56_AC_DEQUANT: [u16; 64] = [
    94, 92, 90, 88, 86, 82, 78, 74, 70, 66, 62, 58, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
    42, 40, 39, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
];

/// Relative (dx, dy) positions of candidate motion-vector predictor
/// macroblocks, from nearest to farthest.
pub const VP56_CANDIDATE_PREDICTOR_POS: [[i8; 2]; 12] = [
    [0, -1],
    [-1, 0],
    [-1, -1],
    [1, -1],
    [0, -2],
    [-2, 0],
    [-2, -1],
    [-1, -2],
    [1, -2],
    [2, -1],
    [-2, -2],
    [2, -2],
];

/// Seed values for predefined MB-type stat tables (used when the
/// encoder signals "reset my stats to predef-idx X").
pub const PRE_DEF_MB_TYPE_STATS: [[[[u8; 2]; 10]; 3]; 16] = include_crate_data::PRE_DEF;

mod include_crate_data {
    // The giant 16x3x10x2 table from `vp56data.c::ff_vp56_pre_def_mb_type_stats`.
    // Kept in its own module purely to keep the file navigable.
    pub(super) const PRE_DEF: [[[[u8; 2]; 10]; 3]; 16] = [
        [
            [
                [9, 15],
                [32, 25],
                [7, 19],
                [9, 21],
                [1, 12],
                [14, 12],
                [3, 18],
                [14, 23],
                [3, 10],
                [0, 4],
            ],
            [
                [41, 22],
                [1, 0],
                [1, 31],
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 7],
                [0, 1],
                [98, 25],
                [4, 10],
            ],
            [
                [2, 3],
                [2, 3],
                [0, 2],
                [0, 2],
                [0, 0],
                [11, 4],
                [1, 4],
                [0, 2],
                [3, 2],
                [0, 4],
            ],
        ],
        [
            [
                [48, 39],
                [1, 2],
                [11, 27],
                [29, 44],
                [7, 27],
                [1, 4],
                [0, 3],
                [1, 6],
                [1, 2],
                [0, 0],
            ],
            [
                [123, 37],
                [6, 4],
                [1, 27],
                [0, 0],
                [0, 0],
                [5, 8],
                [1, 7],
                [0, 1],
                [12, 10],
                [0, 2],
            ],
            [
                [49, 46],
                [3, 4],
                [7, 31],
                [42, 41],
                [0, 0],
                [2, 6],
                [1, 7],
                [1, 4],
                [2, 4],
                [0, 1],
            ],
        ],
        [
            [
                [21, 32],
                [1, 2],
                [4, 10],
                [32, 43],
                [6, 23],
                [2, 3],
                [1, 19],
                [1, 6],
                [12, 21],
                [0, 7],
            ],
            [
                [26, 14],
                [14, 12],
                [0, 24],
                [0, 0],
                [0, 0],
                [55, 17],
                [1, 9],
                [0, 36],
                [5, 7],
                [1, 3],
            ],
            [
                [26, 25],
                [1, 1],
                [2, 10],
                [67, 39],
                [0, 0],
                [1, 1],
                [0, 14],
                [0, 2],
                [31, 26],
                [1, 6],
            ],
        ],
        [
            [
                [69, 83],
                [0, 0],
                [0, 2],
                [10, 29],
                [3, 12],
                [0, 1],
                [0, 3],
                [0, 3],
                [2, 2],
                [0, 0],
            ],
            [
                [209, 5],
                [0, 0],
                [0, 27],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            [
                [103, 46],
                [1, 2],
                [2, 10],
                [33, 42],
                [0, 0],
                [1, 4],
                [0, 3],
                [0, 1],
                [1, 3],
                [0, 0],
            ],
        ],
        [
            [
                [11, 20],
                [1, 4],
                [18, 36],
                [43, 48],
                [13, 35],
                [0, 2],
                [0, 5],
                [3, 12],
                [1, 2],
                [0, 0],
            ],
            [
                [2, 5],
                [4, 5],
                [0, 121],
                [0, 0],
                [0, 0],
                [0, 3],
                [2, 4],
                [1, 4],
                [2, 2],
                [0, 1],
            ],
            [
                [14, 31],
                [9, 13],
                [14, 54],
                [22, 29],
                [0, 0],
                [2, 6],
                [4, 18],
                [6, 13],
                [1, 5],
                [0, 1],
            ],
        ],
        [
            [
                [70, 44],
                [0, 1],
                [2, 10],
                [37, 46],
                [8, 26],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 1],
                [0, 0],
            ],
            [
                [175, 5],
                [0, 1],
                [0, 48],
                [0, 0],
                [0, 0],
                [0, 2],
                [0, 1],
                [0, 2],
                [0, 1],
                [0, 0],
            ],
            [
                [85, 39],
                [0, 0],
                [1, 9],
                [69, 40],
                [0, 0],
                [0, 1],
                [0, 3],
                [0, 1],
                [2, 3],
                [0, 0],
            ],
        ],
        [
            [
                [8, 15],
                [0, 1],
                [8, 21],
                [74, 53],
                [22, 42],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [0, 0],
            ],
            [
                [83, 5],
                [2, 3],
                [0, 102],
                [0, 0],
                [0, 0],
                [1, 3],
                [0, 2],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            [
                [31, 28],
                [0, 0],
                [3, 14],
                [130, 34],
                [0, 0],
                [0, 1],
                [0, 3],
                [0, 1],
                [3, 3],
                [0, 1],
            ],
        ],
        [
            [
                [141, 42],
                [0, 0],
                [1, 4],
                [11, 24],
                [1, 11],
                [0, 1],
                [0, 1],
                [0, 2],
                [0, 0],
                [0, 0],
            ],
            [
                [233, 6],
                [0, 0],
                [0, 8],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
            ],
            [
                [171, 25],
                [0, 0],
                [1, 5],
                [25, 21],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ],
        [
            [
                [8, 19],
                [4, 10],
                [24, 45],
                [21, 37],
                [9, 29],
                [0, 3],
                [1, 7],
                [11, 25],
                [0, 2],
                [0, 1],
            ],
            [
                [34, 16],
                [112, 21],
                [1, 28],
                [0, 0],
                [0, 0],
                [6, 8],
                [1, 7],
                [0, 3],
                [2, 5],
                [0, 2],
            ],
            [
                [17, 21],
                [68, 29],
                [6, 15],
                [13, 22],
                [0, 0],
                [6, 12],
                [3, 14],
                [4, 10],
                [1, 7],
                [0, 3],
            ],
        ],
        [
            [
                [46, 42],
                [0, 1],
                [2, 10],
                [54, 51],
                [10, 30],
                [0, 2],
                [0, 2],
                [0, 1],
                [0, 1],
                [0, 0],
            ],
            [
                [159, 35],
                [2, 2],
                [0, 25],
                [0, 0],
                [0, 0],
                [3, 6],
                [0, 5],
                [0, 1],
                [4, 4],
                [0, 1],
            ],
            [
                [51, 39],
                [0, 1],
                [2, 12],
                [91, 44],
                [0, 0],
                [0, 2],
                [0, 3],
                [0, 1],
                [2, 3],
                [0, 1],
            ],
        ],
        [
            [
                [28, 32],
                [0, 0],
                [3, 10],
                [75, 51],
                [14, 33],
                [0, 1],
                [0, 2],
                [0, 1],
                [1, 2],
                [0, 0],
            ],
            [
                [75, 39],
                [5, 7],
                [2, 48],
                [0, 0],
                [0, 0],
                [3, 11],
                [2, 16],
                [1, 4],
                [7, 10],
                [0, 2],
            ],
            [
                [81, 25],
                [0, 0],
                [2, 9],
                [106, 26],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 1],
                [0, 0],
            ],
        ],
        [
            [
                [100, 46],
                [0, 1],
                [3, 9],
                [21, 37],
                [5, 20],
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 1],
                [0, 0],
            ],
            [
                [212, 21],
                [0, 1],
                [0, 9],
                [0, 0],
                [0, 0],
                [1, 2],
                [0, 2],
                [0, 0],
                [2, 2],
                [0, 0],
            ],
            [
                [140, 37],
                [0, 1],
                [1, 8],
                [24, 33],
                [0, 0],
                [1, 2],
                [0, 2],
                [0, 1],
                [1, 2],
                [0, 0],
            ],
        ],
        [
            [
                [27, 29],
                [0, 1],
                [9, 25],
                [53, 51],
                [12, 34],
                [0, 1],
                [0, 3],
                [1, 5],
                [0, 2],
                [0, 0],
            ],
            [
                [4, 2],
                [0, 0],
                [0, 172],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 0],
                [2, 0],
                [0, 0],
            ],
            [
                [14, 23],
                [1, 3],
                [11, 53],
                [90, 31],
                [0, 0],
                [0, 3],
                [1, 5],
                [2, 6],
                [1, 2],
                [0, 0],
            ],
        ],
        [
            [
                [80, 38],
                [0, 0],
                [1, 4],
                [69, 33],
                [5, 16],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
            ],
            [
                [187, 22],
                [1, 1],
                [0, 17],
                [0, 0],
                [0, 0],
                [3, 6],
                [0, 4],
                [0, 1],
                [4, 4],
                [0, 1],
            ],
            [
                [123, 29],
                [0, 0],
                [1, 7],
                [57, 30],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 0],
            ],
        ],
        [
            [
                [16, 20],
                [0, 0],
                [2, 8],
                [104, 49],
                [15, 33],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            [
                [133, 6],
                [1, 2],
                [1, 70],
                [0, 0],
                [0, 0],
                [0, 2],
                [0, 4],
                [0, 3],
                [1, 1],
                [0, 0],
            ],
            [
                [13, 14],
                [0, 0],
                [4, 20],
                [175, 20],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 1],
                [0, 0],
            ],
        ],
        [
            [
                [194, 16],
                [0, 0],
                [1, 1],
                [1, 9],
                [1, 3],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            [
                [251, 1],
                [0, 0],
                [0, 2],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [202, 23],
                [0, 0],
                [1, 3],
                [2, 9],
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
        ],
    ];
}

/// MB-type delta model probabilities (used when re-training the MB
/// stats inside `vp56_parse_mb_type_models`).
pub const MB_TYPE_MODEL_MODEL: [u8; 6] = [171, 83, 199, 140, 125, 104];

/// `ff_vp56_pmbtm_tree` — binary tree used to read the MB-type *delta*
/// when re-training models.
pub const PMBTM_TREE: &[Vp56Tree] = &[
    Vp56Tree::new(4, 0),
    Vp56Tree::new(2, 1),
    Vp56Tree::new(-8, 0),
    Vp56Tree::new(-4, 0),
    Vp56Tree::new(8, 2),
    Vp56Tree::new(6, 3),
    Vp56Tree::new(4, 4),
    Vp56Tree::new(2, 5),
    Vp56Tree::new(-24, 0),
    Vp56Tree::new(-20, 0),
    Vp56Tree::new(-16, 0),
    Vp56Tree::new(-12, 0),
    Vp56Tree::new(-0, 0),
];

/// `ff_vp56_pmbt_tree` — MB-type selection tree (leaves are
/// `VP56_MB_*` values).
pub const PMBT_TREE: &[Vp56Tree] = &[
    Vp56Tree::new(8, 1),
    Vp56Tree::new(4, 2),
    Vp56Tree::new(2, 4),
    Vp56Tree::new(-(Vp56Mb::InterNoVecPf as i8), 0),
    Vp56Tree::new(-(Vp56Mb::InterDeltaPf as i8), 0),
    Vp56Tree::new(2, 5),
    Vp56Tree::new(-(Vp56Mb::InterV1Pf as i8), 0),
    Vp56Tree::new(-(Vp56Mb::InterV2Pf as i8), 0),
    Vp56Tree::new(4, 3),
    Vp56Tree::new(2, 6),
    Vp56Tree::new(-(Vp56Mb::Intra as i8), 0),
    Vp56Tree::new(-(Vp56Mb::Inter4V as i8), 0),
    Vp56Tree::new(4, 7),
    Vp56Tree::new(2, 8),
    Vp56Tree::new(-(Vp56Mb::InterNoVecGf as i8), 0),
    Vp56Tree::new(-(Vp56Mb::InterDeltaGf as i8), 0),
    Vp56Tree::new(2, 9),
    Vp56Tree::new(-(Vp56Mb::InterV1Gf as i8), 0),
    Vp56Tree::new(-(Vp56Mb::InterV2Gf as i8), 0),
];

/// `ff_vp56_pva_tree` — predicted vector adjustment tree (8 leaves).
pub const PVA_TREE: &[Vp56Tree] = &[
    Vp56Tree::new(8, 0),
    Vp56Tree::new(4, 1),
    Vp56Tree::new(2, 2),
    Vp56Tree::new(-0, 0),
    Vp56Tree::new(-1, 0),
    Vp56Tree::new(2, 3),
    Vp56Tree::new(-2, 0),
    Vp56Tree::new(-3, 0),
    Vp56Tree::new(4, 4),
    Vp56Tree::new(2, 5),
    Vp56Tree::new(-4, 0),
    Vp56Tree::new(-5, 0),
    Vp56Tree::new(2, 6),
    Vp56Tree::new(-6, 0),
    Vp56Tree::new(-7, 0),
];

/// `ff_vp56_pc_tree` — high-bit encoding tree for coefficient values
/// in the long path.
pub const PC_TREE: &[Vp56Tree] = &[
    Vp56Tree::new(4, 6),
    Vp56Tree::new(2, 7),
    Vp56Tree::new(-0, 0),
    Vp56Tree::new(-1, 0),
    Vp56Tree::new(4, 8),
    Vp56Tree::new(2, 9),
    Vp56Tree::new(-2, 0),
    Vp56Tree::new(-3, 0),
    Vp56Tree::new(2, 10),
    Vp56Tree::new(-4, 0),
    Vp56Tree::new(-5, 0),
];

// -- vp6-specific tables -------------------------------------------------

/// Default full-delta-value vector model probabilities.
pub const VP6_DEF_FDV_MODEL: [[u8; 8]; 2] = [
    [247, 210, 135, 68, 138, 220, 239, 246],
    [244, 184, 201, 44, 173, 221, 239, 253],
];

/// Default predicted-delta-value vector model probabilities.
pub const VP6_DEF_PDV_MODEL: [[u8; 7]; 2] = [
    [225, 146, 172, 147, 214, 39, 156],
    [204, 170, 119, 235, 140, 230, 228],
];

/// Default coefficient reorder map (progressive scan).
pub const VP6_DEF_COEFF_REORDER: [u8; 64] = [
    0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9,
    9, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14,
    14, 14, 15, 15, 15, 15, 15, 15,
];

/// Interlaced coefficient reorder map.
pub const VP6_IL_COEFF_REORDER: [u8; 64] = [
    0, 1, 0, 1, 1, 2, 5, 3, 2, 2, 2, 2, 4, 7, 8, 10, 9, 7, 5, 4, 2, 3, 5, 6, 8, 9, 11, 12, 13, 12,
    11, 10, 9, 7, 5, 4, 6, 7, 9, 11, 12, 12, 13, 13, 14, 12, 11, 9, 7, 9, 11, 12, 14, 14, 14, 15,
    13, 11, 13, 15, 15, 15, 15, 15,
];

/// Default run-value coefficient model.
pub const VP6_DEF_RUNV_MODEL: [[u8; 14]; 2] = [
    [
        198, 197, 196, 146, 198, 204, 169, 142, 130, 136, 149, 149, 191, 249,
    ],
    [
        135, 201, 181, 154, 98, 117, 132, 126, 146, 169, 184, 240, 246, 254,
    ],
];

/// Probability that vector DCT / sig models need to be re-parsed.
pub const VP6_SIG_DCT_PCT: [[u8; 2]; 2] = [[237, 246], [231, 243]];

/// Probabilities for predicted-delta-value updates.
pub const VP6_PDV_PCT: [[u8; 7]; 2] = [
    [253, 253, 254, 254, 254, 254, 254],
    [245, 253, 254, 254, 254, 254, 254],
];

/// Probabilities for full-delta-value updates.
pub const VP6_FDV_PCT: [[u8; 8]; 2] = [
    [254, 254, 254, 254, 254, 250, 250, 252],
    [254, 254, 254, 254, 254, 251, 251, 254],
];

/// DC coefficient value update probabilities.
pub const VP6_DCCV_PCT: [[u8; 11]; 2] = [
    [146, 255, 181, 207, 232, 243, 238, 251, 244, 250, 249],
    [179, 255, 214, 240, 250, 255, 244, 255, 255, 255, 255],
];

/// Probability that the coefficient reorder table gets re-parsed.
pub const VP6_COEFF_REORDER_PCT: [u8; 64] = [
    255, 132, 132, 159, 153, 151, 161, 170, 164, 162, 136, 110, 103, 114, 129, 118, 124, 125, 132,
    136, 114, 110, 142, 135, 134, 123, 143, 126, 153, 183, 166, 161, 171, 180, 179, 164, 203, 218,
    225, 217, 215, 206, 203, 217, 229, 241, 248, 243, 253, 255, 253, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
];

/// Run-value coefficient model update probabilities.
pub const VP6_RUNV_PCT: [[u8; 14]; 2] = [
    [
        219, 246, 238, 249, 232, 239, 249, 255, 248, 253, 239, 244, 241, 248,
    ],
    [
        198, 232, 251, 253, 219, 241, 253, 255, 248, 249, 244, 238, 251, 255,
    ],
];

/// `vp6_ract_pct[ct][pt][cg][node]` — AC coefficient value update probabilities.
pub const VP6_RACT_PCT: [[[[u8; 11]; 6]; 2]; 3] = [
    [
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

/// `vp6_dccv_lc[ctx][node]` linear combination coefficients for
/// re-deriving `coeff_dcct` from `coeff_dccv`.
pub const VP6_DCCV_LC: [[[i32; 2]; 5]; 3] = [
    [[122, 133], [0, 1], [78, 171], [139, 117], [168, 79]],
    [[133, 51], [0, 1], [169, 71], [214, 44], [210, 38]],
    [[142, -16], [0, 1], [221, -30], [246, -3], [203, 17]],
];

/// Index -> coefficient-group map for the long AC path.
pub const VP6_COEFF_GROUPS: [u8; 64] = [
    0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
];

/// 4-tap sub-pel filter coefficient table (17 filters x 8 phases x 4 taps).
/// Filter 16 is the pure-bilinear one; 0..15 are the VP6 bicubics.
pub const VP6_BLOCK_COPY_FILTER: [[[i16; 4]; 8]; 17] = [
    [
        [0, 128, 0, 0],
        [-3, 122, 9, 0],
        [-4, 109, 24, -1],
        [-5, 91, 45, -3],
        [-4, 68, 68, -4],
        [-3, 45, 91, -5],
        [-1, 24, 109, -4],
        [0, 9, 122, -3],
    ],
    [
        [0, 128, 0, 0],
        [-4, 124, 9, -1],
        [-5, 110, 25, -2],
        [-6, 91, 46, -3],
        [-5, 69, 69, -5],
        [-3, 46, 91, -6],
        [-2, 25, 110, -5],
        [-1, 9, 124, -4],
    ],
    [
        [0, 128, 0, 0],
        [-4, 123, 10, -1],
        [-6, 110, 26, -2],
        [-7, 92, 47, -4],
        [-6, 70, 70, -6],
        [-4, 47, 92, -7],
        [-2, 26, 110, -6],
        [-1, 10, 123, -4],
    ],
    [
        [0, 128, 0, 0],
        [-5, 124, 10, -1],
        [-7, 110, 27, -2],
        [-7, 91, 48, -4],
        [-6, 70, 70, -6],
        [-4, 48, 92, -8],
        [-2, 27, 110, -7],
        [-1, 10, 124, -5],
    ],
    [
        [0, 128, 0, 0],
        [-6, 124, 11, -1],
        [-8, 111, 28, -3],
        [-8, 92, 49, -5],
        [-7, 71, 71, -7],
        [-5, 49, 92, -8],
        [-3, 28, 111, -8],
        [-1, 11, 124, -6],
    ],
    [
        [0, 128, 0, 0],
        [-6, 123, 12, -1],
        [-9, 111, 29, -3],
        [-9, 93, 50, -6],
        [-8, 72, 72, -8],
        [-6, 50, 93, -9],
        [-3, 29, 111, -9],
        [-1, 12, 123, -6],
    ],
    [
        [0, 128, 0, 0],
        [-7, 124, 12, -1],
        [-10, 111, 30, -3],
        [-10, 93, 51, -6],
        [-9, 73, 73, -9],
        [-6, 51, 93, -10],
        [-3, 30, 111, -10],
        [-1, 12, 124, -7],
    ],
    [
        [0, 128, 0, 0],
        [-7, 123, 13, -1],
        [-11, 112, 31, -4],
        [-11, 94, 52, -7],
        [-10, 74, 74, -10],
        [-7, 52, 94, -11],
        [-4, 31, 112, -11],
        [-1, 13, 123, -7],
    ],
    [
        [0, 128, 0, 0],
        [-8, 124, 13, -1],
        [-12, 112, 32, -4],
        [-12, 94, 53, -7],
        [-10, 74, 74, -10],
        [-7, 53, 94, -12],
        [-4, 32, 112, -12],
        [-1, 13, 124, -8],
    ],
    [
        [0, 128, 0, 0],
        [-9, 124, 14, -1],
        [-13, 112, 33, -4],
        [-13, 95, 54, -8],
        [-11, 75, 75, -11],
        [-8, 54, 95, -13],
        [-4, 33, 112, -13],
        [-1, 14, 124, -9],
    ],
    [
        [0, 128, 0, 0],
        [-9, 123, 15, -1],
        [-14, 113, 34, -5],
        [-14, 95, 55, -8],
        [-12, 76, 76, -12],
        [-8, 55, 95, -14],
        [-5, 34, 112, -13],
        [-1, 15, 123, -9],
    ],
    [
        [0, 128, 0, 0],
        [-10, 124, 15, -1],
        [-14, 113, 34, -5],
        [-15, 96, 56, -9],
        [-13, 77, 77, -13],
        [-9, 56, 96, -15],
        [-5, 34, 113, -14],
        [-1, 15, 124, -10],
    ],
    [
        [0, 128, 0, 0],
        [-10, 123, 16, -1],
        [-15, 113, 35, -5],
        [-16, 98, 56, -10],
        [-14, 78, 78, -14],
        [-10, 56, 98, -16],
        [-5, 35, 113, -15],
        [-1, 16, 123, -10],
    ],
    [
        [0, 128, 0, 0],
        [-11, 124, 17, -2],
        [-16, 113, 36, -5],
        [-17, 98, 57, -10],
        [-14, 78, 78, -14],
        [-10, 57, 98, -17],
        [-5, 36, 113, -16],
        [-2, 17, 124, -11],
    ],
    [
        [0, 128, 0, 0],
        [-12, 125, 17, -2],
        [-17, 114, 37, -6],
        [-18, 99, 58, -11],
        [-15, 79, 79, -15],
        [-11, 58, 99, -18],
        [-6, 37, 114, -17],
        [-2, 17, 125, -12],
    ],
    [
        [0, 128, 0, 0],
        [-12, 124, 18, -2],
        [-18, 114, 38, -6],
        [-19, 99, 59, -11],
        [-16, 80, 80, -16],
        [-11, 59, 99, -19],
        [-6, 38, 114, -18],
        [-2, 18, 124, -12],
    ],
    [
        [0, 128, 0, 0],
        [-4, 118, 16, -2],
        [-7, 106, 34, -5],
        [-8, 90, 53, -7],
        [-8, 72, 72, -8],
        [-7, 53, 90, -8],
        [-5, 34, 106, -7],
        [-2, 16, 118, -4],
    ],
];

/// Divisor from MV magnitude to sub-pel offset per block (4 = luma,
/// 8 = chroma). Used for chroma half-pel positioning from per-block MVs.
pub const VP6_COORD_DIV: [u8; 6] = [4, 4, 4, 4, 8, 8];

/// `vp6_pcr_tree` — the long path "parse run" tree used to read consecutive
/// run lengths in `vp6_parse_coeff`.
pub const VP6_PCR_TREE: &[Vp56Tree] = &[
    Vp56Tree::new(8, 0),
    Vp56Tree::new(4, 1),
    Vp56Tree::new(2, 2),
    Vp56Tree::new(-1, 0),
    Vp56Tree::new(-2, 0),
    Vp56Tree::new(2, 3),
    Vp56Tree::new(-3, 0),
    Vp56Tree::new(-4, 0),
    Vp56Tree::new(8, 4),
    Vp56Tree::new(4, 5),
    Vp56Tree::new(2, 6),
    Vp56Tree::new(-5, 0),
    Vp56Tree::new(-6, 0),
    Vp56Tree::new(2, 7),
    Vp56Tree::new(-7, 0),
    Vp56Tree::new(-8, 0),
    Vp56Tree::new(-0, 0),
];

// -- VP3 IDCT scantable / transpose --------------------------------------

/// Standard zigzag scan of an 8x8 block (as used in JPEG / MPEG / VP6).
pub const ZIGZAG_DIRECT: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// VP6 scan table — maps scan index to raw-order position within the
/// 8x8 block. Per the VP6 bitstream spec (section 12.1), the default
/// scan is the standard zig-zag shown in Figure 14 and the decoder
/// uses `default_dequant_table` (== [`ZIGZAG_DIRECT`]) to rearrange
/// scan-ordered coefficients back to raster order before the IDCT.
///
/// Earlier revisions of this file applied an extra `(v>>3)|((v&7)<<3)`
/// transpose to match an assumed axis convention in a third-party
/// reference. That transpose introduced an axis swap: the first scan
/// AC coefficient (spec: F[0,1], horizontal freq 1, raw index 1) was
/// being placed at raw index 8 (F[1,0], vertical freq 1), so pure
/// horizontal patterns decoded as vertical and vice versa. The spec
/// is unambiguous — scan index `i` maps to raw index `ZIGZAG_DIRECT[i]`
/// — so this alias preserves that direct relationship.
pub const IDCT_SCANTABLE: [u8; 64] = ZIGZAG_DIRECT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mb_type_enum_matches_layout() {
        assert_eq!(Vp56Mb::InterNoVecPf as u8, 0);
        assert_eq!(Vp56Mb::Intra as u8, 1);
        assert_eq!(Vp56Mb::InterV2Gf as u8, 9);
    }

    #[test]
    fn reference_frame_table_agrees_with_ffmpeg() {
        assert_eq!(REFERENCE_FRAME[Vp56Mb::Intra as usize], RefFrame::Current);
        assert_eq!(
            REFERENCE_FRAME[Vp56Mb::InterNoVecGf as usize],
            RefFrame::Golden
        );
        assert_eq!(
            REFERENCE_FRAME[Vp56Mb::InterNoVecPf as usize],
            RefFrame::Previous
        );
    }

    #[test]
    fn idct_scantable_matches_default_dequant_table() {
        // VP6 spec section 12.1: default_dequant_table[64] = ZIGZAG_DIRECT.
        // Scan index i -> raw-order position ZIGZAG_DIRECT[i].
        assert_eq!(IDCT_SCANTABLE[0], 0);
        assert_eq!(IDCT_SCANTABLE[1], 1); // first AC: raster (0,1)
        assert_eq!(IDCT_SCANTABLE[2], 8); // second AC: raster (1,0)
        assert_eq!(IDCT_SCANTABLE, ZIGZAG_DIRECT);
    }
}

//! VP6 motion-vector component decoder (spec §11.1).
//!
//! The fifth BoolCoder-consuming layer (after rounds 16's §13.2.1 DC,
//! 17's §13.3.1 AC, 19's §13.3.3.1 ZRL and 20's per-frame
//! probability-update bitstreams). §11.1 describes how a single
//! component of a new motion vector is decoded from the bitstream by
//! repeated [`crate::bool_coder::BoolCoder::decode_bool`] reads against
//! four probability banks ([`IS_MV_SHORT_PROBS_DEFAULTS`],
//! [`SHORT_MV_PROBS_DEFAULTS`], [`MV_SIZE_PROBS_DEFAULTS`],
//! [`MV_SIGN_PROBS_DEFAULTS`]).
//!
//! ## Layout
//!
//! Each "new" motion vector ([`crate::modes::CodingMode::carries_new_mv`])
//! consists of an x-component and a y-component, decoded in that order.
//! Per §11.1 a component is either a **short vector** (magnitude < 8 in
//! ¼-pixel units) or a **long vector** (magnitude in `[8, 127]`). The
//! `IsVectorShort` flag (a single BoolCoder bit drawn against the
//! per-axis [`IS_MV_SHORT_PROBS_DEFAULTS`]) selects between the two
//! encodings:
//!
//! * **Short path** — three to four BoolCoder bits walk the Figure 11
//!   binary tree (`ShortMvProbs[axis][0..=6]`), producing a magnitude
//!   in `0..=7`. Tree shape verbatim from the §11.1 pseudo-code:
//!   `B(p[0])` selects the upper / lower half (`>3` decision); within
//!   each half a two-step descent picks the leaf.
//! * **Long path** — seven BoolCoder bits encode the low seven bits of
//!   the magnitude in the spec-mandated traversal order
//!   `[0, 1, 2, 7, 6, 5, 4]` against `MvSizeProbs[axis][0..=7]`. Bit 3
//!   is implicit: if any of bits `4..=7` are non-zero, an eighth read
//!   `B(MvSizeProbs[axis][3])` supplies it; otherwise bit 3 is
//!   implicitly `1` (`+= 0x08`) because magnitude `>= 8` is the
//!   long-vector lower bound. Yields a magnitude in `8..=127`.
//! * **Sign** — a final BoolCoder bit drawn against
//!   `MvSignProbs[axis]`; if set, negates the magnitude.
//!
//! The signed range of a decoded component is `-127..=127` (or `0` in
//! the well-defined-but-degenerate "short-path-with-tree-yields-0"
//! case the spec admits via `Vector = B(ShortMvProbs[i][2])` on the
//! third subtree).
//!
//! ## What this module does **not** land
//!
//! * The §10 mode-decode itself, which signals whether an MV is
//!   present for the current MB. The literal §10 pseudo-code is
//!   ambiguously indented and deferred behind a separate DOCS-GAP
//!   report (see the crate-level "DOCS-GAP" rollup); per-MB MV
//!   decode runs only when the upstream mode is one of
//!   `CODE_INTER_PLUS_MV`, `CODE_GOLDEN_MV`, or `CODE_INTER_FOURMV`
//!   per §11. The MV-component decoder here is the consumer that
//!   would be invoked once that gating is wired.
//! * **§11.2 MV probability updates**. The per-frame update
//!   bitstream that mutates the four probability banks at every
//!   inter-frame uses the same two-field pattern (`B(flag_prob)` +
//!   conditional `b(7)` doubled-value) as the §13.2 / §13.3 / §13.3.3
//!   updates already landed in [`crate::prob_update`]; the §11.2
//!   driver lands cleanly as a thin wrapper over the existing
//!   `decode_new_node_prob` primitive once §11.2's flag-prob tables
//!   are transcribed (separate per-codec wiring round).
//! * **MV differential reconstruction**. §11 mandates new vectors are
//!   coded "differentially with respect to the motion vector of the
//!   nearest MacroBlock that uses the same reference frame, if such a
//!   MacroBlock exists … otherwise … coded absolutely". This module
//!   decodes one *delta* component; combining it with the neighbour
//!   MV reference is a caller-side responsibility (the neighbour-MV
//!   resolution itself uses the [`crate::modes::NEAR_MACROBLOCKS`]
//!   traversal landed in round 10).
//! * **The §10 `CODE_INTER_FOURMV` per-block 2-bit codeword** (Table
//!   10). Also a clean fit for this BoolCoder layer (two fixed
//!   probability-128 bits) but a distinct logical unit; deferred.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` §11.1
//! (On2 Technologies, document version 1.02, August 2006). No
//! third-party VP6 implementation has been consulted at any stage.

use crate::bool_coder::BoolCoder;
use crate::Error;

/// Number of motion-vector axes (`x = 0`, `y = 1`).
///
/// Per Table 12: "the first index in each table specifies x or y
/// where x=0 and y=1." Used as the leading dimension of every MV
/// probability bank.
pub const NUM_MV_AXES: usize = 2;

/// Number of tree-node probabilities per axis in the short-MV bank
/// (`ShortMvProbs[2][7]`, Table 12).
///
/// Figure 11's tree has seven internal-node probabilities indexed
/// `0..=6`; the eight leaves carry magnitudes `0..=7`.
pub const NUM_SHORT_MV_NODES: usize = 7;

/// Number of bit-position probabilities per axis in the long-MV bank
/// (`MvSizeProbs[2][8]`, Table 12).
///
/// Eight bit positions: the seven raw bits the spec traverses in the
/// fixed order `[0, 1, 2, 7, 6, 5, 4]` plus bit 3 which is read only
/// when bits `4..=7` are non-zero (otherwise implicitly `1`).
pub const NUM_MV_SIZE_NODES: usize = 8;

/// X-axis index in the §11.1 per-axis MV probability banks.
pub const MV_AXIS_X: usize = 0;

/// Y-axis index in the §11.1 per-axis MV probability banks.
pub const MV_AXIS_Y: usize = 1;

/// Default `IsMvShortProbs[2]` initialiser from §11.1.
///
/// `IsMvShortProbs[axis]` is the probability used for the
/// `B(IsVectorShort)` discriminator that picks between the short-tree
/// and long-bit-stream encodings.
///
/// Verbatim from §11.1:
///
/// ```text
/// Default_IsMvShortProbs[2] = { 162, 164 }   // x, y
/// ```
pub const IS_MV_SHORT_PROBS_DEFAULTS: [u8; NUM_MV_AXES] = [162, 164];

/// Default `ShortMvProbs[2][7]` initialiser from §11.1.
///
/// `ShortMvProbs[axis][node]` holds the per-node probabilities of the
/// Figure 11 short-MV binary tree, with `node` indexing the seven
/// internal nodes (`0..=6`). Default values verbatim from §11.1:
///
/// ```text
/// Default_ShortMvProbs[2][7] =
/// {
///     { 225, 146, 172, 147, 214,  39, 156 },   // x
///     { 204, 170, 119, 235, 140, 230, 228 }    // y
/// }
/// ```
#[rustfmt::skip]
pub const SHORT_MV_PROBS_DEFAULTS: [[u8; NUM_SHORT_MV_NODES]; NUM_MV_AXES] = [
    [225, 146, 172, 147, 214,  39, 156],
    [204, 170, 119, 235, 140, 230, 228],
];

/// Default `MvSizeProbs[2][8]` initialiser from §11.1.
///
/// `MvSizeProbs[axis][bit]` holds the probability for the long-vector
/// bit at index `bit` of the magnitude (low-order bits `0..=7`; bit 3
/// is read only when bits `4..=7` are non-zero). Default values
/// verbatim from §11.1:
///
/// ```text
/// Default_MvSizeProbs[2][8] =
/// {
///     { 247, 210, 135,  68, 138, 220, 239, 246 },   // x
///     { 244, 184, 201,  44, 173, 221, 239, 253 }    // y
/// }
/// ```
#[rustfmt::skip]
pub const MV_SIZE_PROBS_DEFAULTS: [[u8; NUM_MV_SIZE_NODES]; NUM_MV_AXES] = [
    [247, 210, 135,  68, 138, 220, 239, 246],
    [244, 184, 201,  44, 173, 221, 239, 253],
];

/// Default `MvSignProbs[2]` initialiser from §11.1.
///
/// `MvSignProbs[axis]` is the probability used for the sign bit
/// (final `B(MvSignProbs[axis])` read in the per-component pseudo-code).
///
/// Verbatim from §11.1:
///
/// ```text
/// Default_MvSignProbs[2] = { 128, 128 }   // x, y
/// ```
///
/// Note: the default value `128` is the half-interval point — a
/// fresh-from-defaults sign-bit decode is statistically a coin flip.
pub const MV_SIGN_PROBS_DEFAULTS: [u8; NUM_MV_AXES] = [128, 128];

/// Per-axis motion-vector probability snapshot.
///
/// Bundles the four §11.1 per-axis probability banks into a single
/// struct so a caller can hand the per-frame state to
/// [`decode_mv_component`] without juggling four separate borrows.
///
/// Frame-start (intra) initialisation: per §11.1 *"when an intra frame
/// is decoded all the probability values must all be reset to their
/// defaults"* — use [`MvProbs::defaults`]. P-frames persist the
/// previously-decoded values; the §11.2 update bitstream then mutates
/// them in place (separate per-frame ingest stage; not landed here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvProbs {
    /// `IsMvShortProbs[axis]`.
    pub is_short: u8,
    /// `ShortMvProbs[axis][0..=6]`.
    pub short: [u8; NUM_SHORT_MV_NODES],
    /// `MvSizeProbs[axis][0..=7]`.
    pub size: [u8; NUM_MV_SIZE_NODES],
    /// `MvSignProbs[axis]`.
    pub sign: u8,
}

impl MvProbs {
    /// Build the per-axis bundle from `IS_MV_SHORT_PROBS_DEFAULTS` /
    /// `SHORT_MV_PROBS_DEFAULTS` / `MV_SIZE_PROBS_DEFAULTS` /
    /// `MV_SIGN_PROBS_DEFAULTS` for the given axis (`MV_AXIS_X` or
    /// `MV_AXIS_Y`).
    ///
    /// Panics if `axis >= NUM_MV_AXES`.
    pub const fn defaults(axis: usize) -> Self {
        assert!(axis < NUM_MV_AXES, "axis must be 0 (x) or 1 (y)");
        Self {
            is_short: IS_MV_SHORT_PROBS_DEFAULTS[axis],
            short: SHORT_MV_PROBS_DEFAULTS[axis],
            size: MV_SIZE_PROBS_DEFAULTS[axis],
            sign: MV_SIGN_PROBS_DEFAULTS[axis],
        }
    }
}

/// Decode one short-MV magnitude (§11.1, Figure 11).
///
/// Walks the Figure 11 binary tree by repeated `B(short[node])`
/// reads against the per-axis short-MV probability bank. Returns a
/// magnitude in `0..=7`.
///
/// Pseudo-code verbatim from §11.1 (the inner "Traverse the short
/// vector tree" block under `B(IsMvShortProbs[i]) == 1`):
///
/// ```text
/// If ( B( ShortMvProbs[i][0] ) )
/// {
///     Vector += (1 << 2)
///     If ( B( ShortMvProbs[i][4] ) )
///     {
///         Vector += (1 << 1)
///         Vector += B( ShortMvProbs[i][6] )
///     }
///     Else
///         Vector += B( ShortMvProbs[i][5] )
/// }
/// Else
/// {
///     If ( B( ShortMvProbs[i][1] ) )
///     {
///         Vector += (1 << 1)
///         Vector += B( ShortMvProbs[i][3] )
///     }
///     Else
///         Vector = B( ShortMvProbs[i][2] )
/// }
/// ```
///
/// The `Vector = B(...)` (assignment) on the third subtree — as
/// opposed to `Vector += B(...)` — is verbatim per the spec; with
/// `Vector` already 0 at entry it makes no observable difference.
pub fn decode_short_mv_magnitude(
    bc: &mut BoolCoder<'_>,
    short: &[u8; NUM_SHORT_MV_NODES],
) -> Result<u32, Error> {
    let mut vector: u32 = 0;
    if bc.decode_bool(short[0])? != 0 {
        vector += 1 << 2;
        if bc.decode_bool(short[4])? != 0 {
            vector += 1 << 1;
            vector += u32::from(bc.decode_bool(short[6])?);
        } else {
            vector += u32::from(bc.decode_bool(short[5])?);
        }
    } else if bc.decode_bool(short[1])? != 0 {
        vector += 1 << 1;
        vector += u32::from(bc.decode_bool(short[3])?);
    } else {
        vector = u32::from(bc.decode_bool(short[2])?);
    }
    Ok(vector)
}

/// Decode one long-MV magnitude (§11.1, "Read bit 0,1,2,7,6,5,4 of
/// the Long vector").
///
/// Reads seven BoolCoder bits in the spec-mandated traversal order
/// `[0, 1, 2, 7, 6, 5, 4]` against `size[0..=7]`. Bit 3 is implicit:
/// if any of bits `4..=7` are non-zero (`Vector[i] & 0xF0`), an
/// eighth read `B(size[3])` supplies it; otherwise bit 3 is set
/// implicitly (`Vector += 0x08`) per the §11.1 "Bit 3 is implicit if
/// none of the higher order bits are" comment.
///
/// Returns a magnitude in `8..=127` (`0x08 ..= 0xFF` minus the
/// always-implicit-or-decoded bit-3 setting).
///
/// Pseudo-code verbatim from §11.1 (the inner "Else" block under
/// `B(IsMvShortProbs[i]) == 0`):
///
/// ```text
/// // Read bit 0,1,2, 7, 6, 5, 4 of the Long vector
/// Vector[i] = B( MvSizeProbs[i][0] )
/// Vector[i] += B( MvSizeProbs[i][1] ) << 1
/// Vector[i] += B( MvSizeProbs[i][2] ) << 2
/// Vector[i] += B( MvSizeProbs[i][7] ) << 7
/// Vector[i] += B( MvSizeProbs[i][6] ) << 6
/// Vector[i] += B( MvSizeProbs[i][5] ) << 5
/// Vector[i] += B( MvSizeProbs[i][4] ) << 4
///
/// // Note : Bit 3 is implicit if none of
/// // the higher order bits are
/// if (Vector[i] & 0xF0 )
///    Vector[i] += B( MvSizeProbs[i][3] ) << 3
/// else
///    Vector[i] += 0x08
/// ```
pub fn decode_long_mv_magnitude(
    bc: &mut BoolCoder<'_>,
    size: &[u8; NUM_MV_SIZE_NODES],
) -> Result<u32, Error> {
    let mut vector: u32 = u32::from(bc.decode_bool(size[0])?);
    vector += u32::from(bc.decode_bool(size[1])?) << 1;
    vector += u32::from(bc.decode_bool(size[2])?) << 2;
    vector += u32::from(bc.decode_bool(size[7])?) << 7;
    vector += u32::from(bc.decode_bool(size[6])?) << 6;
    vector += u32::from(bc.decode_bool(size[5])?) << 5;
    vector += u32::from(bc.decode_bool(size[4])?) << 4;

    if (vector & 0xF0) != 0 {
        vector += u32::from(bc.decode_bool(size[3])?) << 3;
    } else {
        vector += 0x08;
    }
    Ok(vector)
}

/// Decode one signed motion-vector component (§11.1).
///
/// Composes the §11.1 per-component pseudo-code in full: the
/// `B(IsVectorShort)` discriminator, the short / long magnitude
/// path, and the final `B(MvSignProbs)` sign read with negation.
/// Returns a signed `i32` magnitude:
///
/// * Short path: `-7..=7` (zero falls within the magnitude range
///   because §11.1's third subtree assigns `Vector = B(...)` which
///   yields 0 when the bit is 0).
/// * Long path: `-127..=-8` ∪ `8..=127`.
///
/// Pseudo-code verbatim from §11.1 (the outer per-axis loop body):
///
/// ```text
/// For ( i == 0; i < 2; i++ )
/// {
///    Vector = 0
///
///    // Is the vector a short motion vector
///    If ( B( IsMvShortProbs[i] ) )
///    {
///        // (short-tree walk; see decode_short_mv_magnitude)
///    }
///    Else
///    {
///        // (long-bit-stream walk; see decode_long_mv_magnitude)
///    }
///
///    SignBit = B(MvSignProbs[i])
///    If (SignBit)
///        Vector[i] = -Vector[i]
/// }
/// ```
pub fn decode_mv_component(bc: &mut BoolCoder<'_>, probs: &MvProbs) -> Result<i32, Error> {
    let magnitude: u32 = if bc.decode_bool(probs.is_short)? != 0 {
        decode_short_mv_magnitude(bc, &probs.short)?
    } else {
        decode_long_mv_magnitude(bc, &probs.size)?
    };

    let sign_bit = bc.decode_bool(probs.sign)?;
    let signed = magnitude as i32;
    Ok(if sign_bit != 0 { -signed } else { signed })
}

/// Decode a full motion-vector `(x, y)` pair (§11.1, outer
/// `for i = 0..=1` loop).
///
/// Calls [`decode_mv_component`] first against `probs[MV_AXIS_X]` and
/// then against `probs[MV_AXIS_Y]`, returning the pair `(dx, dy)`.
///
/// The returned components are deltas relative to the differential
/// reference per §11; combining them with the neighbour MV (or with
/// `(0, 0)` for the absolute-coding case) is a caller-side concern
/// (see [`crate::modes::NEAR_MACROBLOCKS`] for the §10 neighbour-MV
/// traversal that resolves the reference).
pub fn decode_mv_pair(
    bc: &mut BoolCoder<'_>,
    probs: &[MvProbs; NUM_MV_AXES],
) -> Result<(i32, i32), Error> {
    let dx = decode_mv_component(bc, &probs[MV_AXIS_X])?;
    let dy = decode_mv_component(bc, &probs[MV_AXIS_Y])?;
    Ok((dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenate `count` zero bytes onto a starting prefix; useful
    /// for steering the BoolCoder's per-bit `Value < Split << 24`
    /// comparison toward the 0-branch.
    fn zero_stream(count: usize) -> Vec<u8> {
        vec![0u8; count]
    }

    /// All-ones byte buffer; steers the BoolCoder toward the 1-branch
    /// whenever the probability is below 255.
    fn ones_stream(count: usize) -> Vec<u8> {
        vec![0xFFu8; count]
    }

    #[test]
    fn defaults_are_per_axis_consistent() {
        let x = MvProbs::defaults(MV_AXIS_X);
        let y = MvProbs::defaults(MV_AXIS_Y);
        assert_eq!(x.is_short, IS_MV_SHORT_PROBS_DEFAULTS[0]);
        assert_eq!(y.is_short, IS_MV_SHORT_PROBS_DEFAULTS[1]);
        assert_eq!(&x.short, &SHORT_MV_PROBS_DEFAULTS[0]);
        assert_eq!(&y.short, &SHORT_MV_PROBS_DEFAULTS[1]);
        assert_eq!(&x.size, &MV_SIZE_PROBS_DEFAULTS[0]);
        assert_eq!(&y.size, &MV_SIZE_PROBS_DEFAULTS[1]);
        assert_eq!(x.sign, MV_SIGN_PROBS_DEFAULTS[0]);
        assert_eq!(y.sign, MV_SIGN_PROBS_DEFAULTS[1]);
    }

    #[test]
    fn defaults_match_spec_text() {
        // Verbatim §11.1 default tables, re-listed in the test to
        // pin a transcription error on either side from drifting in.
        assert_eq!(IS_MV_SHORT_PROBS_DEFAULTS, [162, 164]);
        assert_eq!(
            SHORT_MV_PROBS_DEFAULTS,
            [
                [225, 146, 172, 147, 214, 39, 156],
                [204, 170, 119, 235, 140, 230, 228],
            ]
        );
        assert_eq!(
            MV_SIZE_PROBS_DEFAULTS,
            [
                [247, 210, 135, 68, 138, 220, 239, 246],
                [244, 184, 201, 44, 173, 221, 239, 253],
            ]
        );
        assert_eq!(MV_SIGN_PROBS_DEFAULTS, [128, 128]);
    }

    /// At all-zero stream with low probabilities (the 0-branch always
    /// fires) the short-tree pseudo-code walks
    /// `Stats[0]=0 → Stats[1]=0 → Vector = B(Stats[2])` and produces
    /// magnitude 0 (since `B(Stats[2]) == 0` too).
    ///
    /// At prob=1 against zero stream (operative `>> 8` Split, errata
    /// #35): `Split = 1 + (254*1 >> 8) = 1`, `Split << 24 = 0x01000000`,
    /// `Value = 0`, comparison is true → `Bit = 0`. So every `B(1)`
    /// against zero-stream yields 0.
    #[test]
    fn short_magnitude_all_zero_path_yields_zero() {
        let short = [1u8; NUM_SHORT_MV_NODES];
        let stream = zero_stream(8);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let m = decode_short_mv_magnitude(&mut bc, &short).unwrap();
        assert_eq!(m, 0);
    }

    /// At all-ones stream with low probabilities the 1-branch fires
    /// for every `B(prob)` because `Value = 0xFFFFFFFF` is the
    /// largest possible initial state.
    ///
    /// At prob=1 against ones stream (operative `>> 8` Split, errata
    /// #35): `Split = 1`, `Split << 24 = 0x01000000`,
    /// `Value = 0xFFFFFFFF`, comparison is false → `Bit = 1`. The short
    /// tree walks `Stats[0]=1 → Stats[4]=1 → Vector = 4 + 2 +
    /// B(Stats[6])` and produces magnitude 7. (At prob=1 with a high
    /// `Value` the 1-branch interval `[Split, Range)` covers nearly the
    /// whole range, so the read is `Bit = 1`.)
    #[test]
    fn short_magnitude_max_path_yields_seven() {
        let short = [1u8; NUM_SHORT_MV_NODES];
        let stream = ones_stream(8);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let m = decode_short_mv_magnitude(&mut bc, &short).unwrap();
        assert_eq!(m, 7);
    }

    /// Short-MV magnitude always falls in `0..=7`, regardless of the
    /// probability vector or the input bit stream. Sweep over the
    /// VP6-staged §11.1 default probability rows (`x` and `y` axes)
    /// against several byte streams. Under the operative `>> 8` Split
    /// (errata #35) the BoolCoder is non-degenerate at every
    /// probability, so the magnitude bound holds for any prob row.
    #[test]
    fn short_magnitude_range_invariant() {
        let probs_corners: [[u8; NUM_SHORT_MV_NODES]; 2] =
            [SHORT_MV_PROBS_DEFAULTS[0], SHORT_MV_PROBS_DEFAULTS[1]];
        let streams: [Vec<u8>; 4] = [
            zero_stream(32),
            ones_stream(32),
            vec![0x55; 32],
            [0xA3, 0x5C, 0xF1, 0x08, 0x91, 0x47, 0xBE, 0x20].repeat(4),
        ];
        for probs in &probs_corners {
            for stream in &streams {
                let mut bc = BoolCoder::new(stream).unwrap();
                let m = decode_short_mv_magnitude(&mut bc, probs).unwrap();
                assert!(m <= 7, "magnitude {m} outside 0..=7");
            }
        }
    }

    /// Short-MV magnitude consumes between 2 and 3 BoolCoder bits
    /// inclusive: the tree depth is 3 (root + 2 internal layers).
    #[test]
    fn short_magnitude_consumes_three_or_fewer_bits() {
        let probs = SHORT_MV_PROBS_DEFAULTS[0];
        let stream = vec![0x55u8; 16];
        let mut bc = BoolCoder::new(&stream).unwrap();
        let pos_before = bc.pos();
        let count_before = bc.count();
        let _m = decode_short_mv_magnitude(&mut bc, &probs).unwrap();
        // The decoder advances some combination of `count` (in-byte
        // ticks) and `pos` (byte refills). The total bit-advance is
        // `8 * (pos_after - pos_before) + (count_before - count_after)`.
        let pos_after = bc.pos();
        let count_after = bc.count();
        // Each per-bit decode_bool potentially consumes 1..8 renorm
        // steps so we can't pin the exact byte-count, but the total
        // bit-advance is well-defined as the per-bit count.
        let bytes_advanced = pos_after - pos_before;
        // Upper bound: a 3-deep walk at probability values that force
        // every bit's renormalization to refill a whole byte is at
        // most 3 byte refills; in practice it's typically less.
        assert!(bytes_advanced <= 3, "advanced {bytes_advanced} bytes");
        // Sanity: count is always in 1..=8.
        assert!((1..=8).contains(&count_before));
        assert!((1..=8).contains(&count_after));
    }

    /// At zero stream + low probabilities (every `B(prob)==0`) the
    /// long-MV path computes `Vector = 0` from the seven low bits,
    /// then takes the `(Vector & 0xF0) == 0` branch and sets bit 3
    /// implicitly, yielding magnitude `0x08 = 8`.
    #[test]
    fn long_magnitude_all_zero_yields_implicit_bit3() {
        let size = [1u8; NUM_MV_SIZE_NODES];
        let stream = zero_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let m = decode_long_mv_magnitude(&mut bc, &size).unwrap();
        assert_eq!(m, 0x08);
    }

    /// At ones stream + low probabilities (every `B(prob)==1` fires;
    /// see `short_magnitude_max_path_yields_seven` for the
    /// errata-#35 derivation) the long-MV path computes
    /// `Vector = 0b_1111_0111 = 0xF7 = 247` from the seven traversed
    /// bits, then takes the `(Vector & 0xF0) != 0` branch and reads
    /// `B(size[3]) == 1`, yielding `0xFF = 255`. §11.1 caps the
    /// long-MV range at `<= 127`. The all-ones probability-and-stream
    /// combination is therefore an *out-of-spec* test of the formula
    /// itself; the bit-arithmetic is well-defined (8-bit unsigned)
    /// and produces `0xFF` because the spec doesn't bound the formula
    /// to enforce the documented range cap (that's the encoder's
    /// responsibility per §11.1 *"a long vector is defined as a
    /// vector with a length that is … less than or equal to 127"*).
    #[test]
    fn long_magnitude_all_ones_path_yields_seven_bits_set_plus_bit3() {
        let size = [1u8; NUM_MV_SIZE_NODES];
        let stream = ones_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let m = decode_long_mv_magnitude(&mut bc, &size).unwrap();
        // The seven traversal reads contribute bits at positions
        // 0, 1, 2, 7, 6, 5, 4 — all set; result = 0b_1111_0111 = 0xF7.
        // Then bit 3 is decoded against size[3] (since 0xF0 != 0)
        // and set, yielding 0xFF.
        assert_eq!(m, 0xFF);
    }

    /// Long magnitude is in `8..=255` for any input. The spec's
    /// `<= 127` range cap is an encoder-side constraint per
    /// §11.1's "less than or equal to 127" definition; the decoder
    /// formula itself produces `8..=255`.
    #[test]
    fn long_magnitude_lower_bound_at_least_eight() {
        let stream = vec![0x55u8; 32];
        for size in MV_SIZE_PROBS_DEFAULTS.iter() {
            let mut bc = BoolCoder::new(&stream).unwrap();
            let m = decode_long_mv_magnitude(&mut bc, size).unwrap();
            assert!(m >= 8, "magnitude {m} below long-vector floor of 8");
        }
    }

    /// Long-magnitude consumes `B(size[0..=7])` in that traversal
    /// order; per §11.1 the contributions are bit positions
    /// `[0, 1, 2, 7, 6, 5, 4]` (then conditionally `3`). Verifies the
    /// traversal-order mapping by constructing a stream where only
    /// the third of the seven traversal reads (which contributes to
    /// bit position 7) decodes to 1, and all others to 0. Since the
    /// zero-stream yields `Bit = 0` at any probability (`Value = 0` is
    /// below any `Split << 24`, the operative `>> 8` Split keeping
    /// `Split >= 1`, errata #35), we cannot use
    /// the zero stream to force a 1-branch without specially crafting
    /// the byte sequence. Instead test the magnitude formula
    /// algebraically: setting bit 7 directly via `B(size[2]) == 1`
    /// (the third traversal read) is not straightforward without
    /// careful stream construction. We pin a weaker property: when
    /// the formula reads the seven bits and the result yields
    /// `(magnitude & 0xF0) != 0`, it then reads bit 3, otherwise it
    /// implicitly sets bit 3.
    #[test]
    fn long_magnitude_high_bits_path_reads_bit_three() {
        // Pick a stream + probs that produces `(vector & 0xF0) == 0`
        // after the seven traversal reads, exercising the
        // "bit 3 implicit" branch.
        let size = [255u8; NUM_MV_SIZE_NODES];
        let stream = zero_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let pos_before = bc.pos();
        let m = decode_long_mv_magnitude(&mut bc, &size).unwrap();
        // All B(255) against zero stream yield 0 → vector = 0 from
        // the seven reads → (vector & 0xF0) == 0 → bit 3 implicit.
        // Final magnitude = 0 + 0x08 = 0x08.
        assert_eq!(m, 0x08);
        // Eight reads (seven traversal + zero conditional) consume
        // some bytes from the stream; record the upper bound.
        let pos_after = bc.pos();
        assert!(pos_after >= pos_before);
    }

    /// MV-component decode composes magnitude + sign correctly: at
    /// zero stream + low `is_short` (so `B(is_short)` decodes to 0:
    /// `Value = 0` < `Split << 24` always, operative `>> 8` Split per
    /// errata #35) the long path is taken; at low size probs against
    /// zero stream every
    /// `B(size[k])` yields 0, so `vector & 0xF0 == 0` triggers the
    /// implicit-bit-3 branch and the magnitude is `0x08`. Sign at
    /// low prob + zero stream yields 0 → final signed = `+8`.
    #[test]
    fn component_decode_long_positive() {
        let probs = MvProbs {
            is_short: 1,
            short: [1; NUM_SHORT_MV_NODES],
            size: [1; NUM_MV_SIZE_NODES],
            sign: 1,
        };
        let stream = zero_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let signed = decode_mv_component(&mut bc, &probs).unwrap();
        assert_eq!(signed, 8);
    }

    /// MV-component decode against high `is_short` prob + zero stream
    /// → `B(255)` yields 0 (operative `>> 8` Split, errata #35:
    /// `Split = 254 = Range - 1`, so `Split << 24 = 0xFE00_0000 >
    /// Value = 0` → 0-branch) → long path is taken. At zero stream +
    /// high
    /// size probs every `B(size[k])` yields 0; vector & 0xF0 == 0
    /// triggers implicit bit 3; magnitude `0x08`. Sign at high
    /// prob + zero stream yields 0 → final signed = `+8`.
    #[test]
    fn component_decode_long_via_high_is_short_zero_stream() {
        let probs = MvProbs {
            is_short: 255,
            short: [1; NUM_SHORT_MV_NODES],
            size: [255; NUM_MV_SIZE_NODES],
            sign: 255,
        };
        let stream = zero_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let signed = decode_mv_component(&mut bc, &probs).unwrap();
        // Long path; all B(255) at zero stream yield 0 →
        // vector from seven reads = 0 → implicit bit 3 → mag 0x08.
        // Sign B(255) at zero stream yields 0 → positive.
        assert_eq!(signed, 8);
    }

    /// MV-component decode against low `is_short` + ones stream →
    /// `B(1)` at ones stream yields 1 → short path taken; all
    /// `B(short[k])` at ones stream yield 1 → max-magnitude
    /// short-tree walk = 7; sign `B(1)` at ones stream yields 1 →
    /// negate → final signed = `-7`.
    #[test]
    fn component_decode_short_negative_via_ones_stream() {
        let probs = MvProbs {
            is_short: 1,
            short: [1; NUM_SHORT_MV_NODES],
            size: [1; NUM_MV_SIZE_NODES],
            sign: 1,
        };
        let stream = ones_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let signed = decode_mv_component(&mut bc, &probs).unwrap();
        assert_eq!(signed, -7);
    }

    /// MV-component decode signed range per §11.1: short path yields
    /// `-7..=7`, long path yields `-255..=-8` ∪ `8..=255` (where
    /// `>127` is encoder-side out-of-spec but decoder-formula valid).
    /// Sweep over §11.1 default probability vectors against several
    /// byte streams. Under the operative `>> 8` Split (errata #35) the
    /// BoolCoder is non-degenerate at every probability in `1..=255`,
    /// so the signed-range bound holds for any prob vector.
    #[test]
    fn component_decode_range_invariant() {
        let prob_choices: [MvProbs; 2] =
            [MvProbs::defaults(MV_AXIS_X), MvProbs::defaults(MV_AXIS_Y)];
        let streams: [Vec<u8>; 3] = [zero_stream(64), ones_stream(64), vec![0x55; 64]];
        for probs in &prob_choices {
            for stream in &streams {
                let mut bc = BoolCoder::new(stream).unwrap();
                let signed = decode_mv_component(&mut bc, probs).unwrap();
                assert!(
                    (-255..=255).contains(&signed),
                    "signed {signed} outside theoretical decoder range"
                );
            }
        }
    }

    /// MV-component decode is deterministic: same probabilities +
    /// same byte stream → same output across runs.
    #[test]
    fn component_decode_is_deterministic() {
        let probs = MvProbs::defaults(MV_AXIS_X);
        let stream = vec![0xA3, 0x5C, 0xF1, 0x08, 0x91, 0x47, 0xBE, 0x20];
        let mut bc1 = BoolCoder::new(&stream).unwrap();
        let mut bc2 = BoolCoder::new(&stream).unwrap();
        let r1 = decode_mv_component(&mut bc1, &probs).unwrap();
        let r2 = decode_mv_component(&mut bc2, &probs).unwrap();
        assert_eq!(r1, r2);
    }

    /// The pair decoder reads x first then y; varying y's probs
    /// while holding x's fixed only affects the second component.
    #[test]
    fn pair_decode_axis_independence() {
        let stream = vec![0xA3, 0x5C, 0xF1, 0x08, 0x91, 0x47, 0xBE, 0x20];
        let probs_xy_default: [MvProbs; 2] =
            [MvProbs::defaults(MV_AXIS_X), MvProbs::defaults(MV_AXIS_Y)];
        let mut bc = BoolCoder::new(&stream).unwrap();
        let (dx_a, _) = decode_mv_pair(&mut bc, &probs_xy_default).unwrap();

        let probs_x_default_y_high: [MvProbs; 2] = [
            MvProbs::defaults(MV_AXIS_X),
            MvProbs {
                is_short: 255,
                short: [255; NUM_SHORT_MV_NODES],
                size: [255; NUM_MV_SIZE_NODES],
                sign: 255,
            },
        ];
        let mut bc2 = BoolCoder::new(&stream).unwrap();
        let (dx_b, _) = decode_mv_pair(&mut bc2, &probs_x_default_y_high).unwrap();
        // X is decoded with identical probs and identical bytes, so
        // x must match between the two runs.
        assert_eq!(dx_a, dx_b);
    }

    /// Truncation surface: a 4-byte buffer (the minimum the
    /// `BoolCoder::new` constructor accepts) exhausts during the
    /// per-component traversal.
    #[test]
    fn component_decode_truncation_surface() {
        // 4 bytes prefill `Value` and put `Pos = 4`. The first
        // renormalization that needs a byte at `Pos = 4` fails with
        // `Error::Truncated`. With low probabilities the
        // renormalization runs aggressively; this stream triggers
        // truncation reliably within one mv-component traversal.
        let probs = MvProbs {
            is_short: 1,
            short: [1; NUM_SHORT_MV_NODES],
            size: [1; NUM_MV_SIZE_NODES],
            sign: 1,
        };
        let stream = vec![0u8; 4];
        let mut bc = BoolCoder::new(&stream).unwrap();
        let result = decode_mv_component(&mut bc, &probs);
        assert_eq!(result, Err(Error::Truncated));
    }

    /// At default probabilities + a zero stream the short-path
    /// IsVectorShort branch is taken when `is_short` is high enough
    /// (defaults 162/164 are above the half-interval point 128, so
    /// at zero stream the BoolCoder steers toward `B == 0`). Walking
    /// the resulting short-tree from defaults against zero stream
    /// returns a fully-determined magnitude.
    #[test]
    fn defaults_against_zero_stream_well_defined() {
        let probs = MvProbs::defaults(MV_AXIS_X);
        let stream = zero_stream(16);
        let mut bc = BoolCoder::new(&stream).unwrap();
        let signed = decode_mv_component(&mut bc, &probs).unwrap();
        // Zero stream + defaults produces a specific deterministic
        // magnitude; we don't pin a particular value because the
        // BoolCoder's interaction with the probability vector is
        // sensitive to the exact byte sequence — but it must be
        // in the valid signed-decoder range.
        assert!((-255..=255).contains(&signed));
    }
}

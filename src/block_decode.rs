//! VP6 per-block DCT coefficient reconstruction (spec §13.2.1 +
//! §13.3.1 + §13.3.3.1 composed with §12 scan order and §15 inverse
//! quantization).
//!
//! This module is the per-block *driver* that earlier rounds left as
//! an explicit deferral: it composes the already-landed
//! per-coefficient primitives into the spec's "decode the AC
//! coefficients of a block" loop and the follow-on scan/dequant
//! mapping, producing an 8x8 raster-order coefficient block ready for
//! the §16 [`crate::idct_block`] inverse transform.
//!
//! The composition is exactly the §13.3.1 listing:
//!
//! ```text
//! Set CoeffData to 64 0's
//! if (dc == 0)       Prec = 0
//! else if (dc == 1)  Prec = 1
//! else               Prec = 2
//! EncodedCoeffs = 1
//! do
//! {
//!    ProbPtr = AcUpdateProbs[Prec][Plane][ACProbBand[EncodedCoeffs]]
//!    ... token tree walk (decode_ac_coefficient) ...
//!    EOB leaf   → EncodedCoeffs++ ; break
//!    ZERO leaf  → Prec = 0 ; EncodedCoeffs += ZeroRunCount
//!    value leaf → CoeffData[EncodedCoeffs] = signed value ;
//!                 Prec update ; EncodedCoeffs++
//! } while (EncodedCoeffs < BLOCK_SIZE)
//! ```
//!
//! preceded by the §13.2.1 DC decode (`CoeffData[0]`) and followed by
//! the §12 scan-to-raster permutation and the §15 dequantizer.
//!
//! ## Probability-bank layout note
//!
//! The §13.3.1 listing names its lookup `AcUpdateProbs[Prec][Plane]
//! [Band]`, but the §13.3.1 prose is explicit that "the set of
//! probabilities that correspond to these 4 pieces of context stored
//! in ACProbs act as the binary decoding node probabilities" — i.e.
//! the *decoding* bank is the persistent `AcProbs` array whose §13.3
//! dimension order is `[plane][prec][band][node]` (the layout
//! [`crate::tokens::baseline_ac_probs`] seeds and
//! [`crate::prob_update::update_ac_probs`] mutates). The listing's
//! identifier is a naming slip for the unrelated Table 35 update-flag
//! bank ([`crate::tokens::AC_UPDATE_PROBS`], dimension order
//! `[prec][plane][band][node]`), which drives the per-frame
//! *probability update* bitstream, not coefficient decoding. This
//! driver therefore takes the `[plane][prec][band][node]` `AcProbs`
//! bank and performs the listing's per-iteration `[Prec][Band]`
//! re-selection inside it.
//!
//! ## `EncodedCoeffs` exit choreography
//!
//! The printed listing increments `EncodedCoeffs` once more on the
//! EOB branch before `break`, then decrements once after the loop
//! (`EncodedCoeffs--` ahead of the `Finished:` label). On the EOB
//! path the two cancel, leaving the count of scan positions that
//! carry decoded data. On the natural `EncodedCoeffs >= BLOCK_SIZE`
//! exit the literal decrement would yield 63 for a fully-populated
//! block — but no later listing in the document ever consumes the
//! post-loop value, so this driver defines the unambiguous
//! [`BlockCoeffs::coeff_count`] invariant instead: the number of scan
//! positions covered by the decode, with everything from
//! `coeff_count` onward guaranteed zero.
//!
//! ## Provenance
//!
//! Sourced exclusively from material in `docs/video/vp6/`:
//!
//! * `vp6_format.pdf` §13.2.1 (pages 64–65) — the DC decode that
//!   seeds `CoeffData[0]` and the `Prec` context.
//! * `vp6_format.pdf` §13.3.1 (pages 70–72) — the per-block AC loop
//!   pseudocode reproduced above.
//! * `vp6_format.pdf` §13.3.3 / §13.3.3.1 (page 78) — the zero-run
//!   decode invoked on the ZERO leaf and its
//!   `ZeroRunProbs[ZrlBand[pos]]` band selection.
//! * `vp6_format.pdf` §12.1 / §12.2 (pages 53–55) — the
//!   scan-position-to-raster permutation consumed by
//!   [`dequantize_to_raster`].
//! * `vp6_format.pdf` §15 (page 82) — the DC / AC scalar
//!   dequantization applied per coefficient.
//! * `vp6-errata-and-clarifications.md` entries #35 (BoolCoder
//!   `Split`) and #67 (magnitude-bit `Probs[]` lengths), inherited
//!   through the round-16/17/19 primitives.
//!
//! No third-party VP6 source has been consulted at any stage.

use crate::dct_decode::{decode_ac_coefficient, decode_ac_zero_run, decode_dc, AcOutcome};
use crate::dequant::DequantContext;
use crate::tokens::{
    AcBand, AcPlane, AcPrecContext, NUM_AC_BANDS, NUM_AC_PREC_CONTEXTS, NUM_PLANES, NUM_TREE_NODES,
};
use crate::zrl::{ZrlBand, NUM_ZRL_BANDS, NUM_ZRL_NODES};
use crate::{BoolCoder, Error};

/// The number of DCT coefficients in an 8x8 block — the §13.3.1
/// loop bound `BLOCK_SIZE`.
pub const BLOCK_SIZE: usize = 64;

/// The persistent §13.3 `AcProbs[2][3][6][11]` decoding bank,
/// dimension order `[plane][prec][band][node]` (§13.3 prose: first
/// dimension plane per Table 28, second the Table 29 preceding-
/// coefficient context, third the Table 30 band, fourth the Table 20
/// node). Seeded by [`crate::tokens::baseline_ac_probs`] at each
/// keyframe and mutated per-frame by
/// [`crate::prob_update::update_ac_probs`].
pub type AcProbBank = [[[[u8; NUM_TREE_NODES]; NUM_AC_BANDS]; NUM_AC_PREC_CONTEXTS]; NUM_PLANES];

/// The persistent §13.3.3 `ZeroRunProbs[2][14]` bank, indexed by
/// [`ZrlBand`] then [`crate::zrl::ZrlNode`]. Seeded by
/// [`crate::zrl::ZERO_RUN_PROB_DEFAULTS`] and mutated per-frame by
/// [`crate::prob_update::update_zero_run_probs`].
pub type ZeroRunProbBank = [[u8; NUM_ZRL_NODES]; NUM_ZRL_BANDS];

/// A fully entropy-decoded 8x8 coefficient block in **scan order**
/// (the §13.3.1 `CoeffData[64]` array).
///
/// `coeffs[0]` is the §13.2.1 DC coefficient; `coeffs[1..]` are the
/// AC coefficients at scan positions `1..=63` of whichever scan order
/// (§12.1 default or §12.2 custom) the frame is using — the entropy
/// stage itself is scan-order-agnostic, the order only matters when
/// mapping to raster positions via [`dequantize_to_raster`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCoeffs {
    /// The 64 decoded coefficients in scan order (`CoeffData`).
    pub coeffs: [i32; BLOCK_SIZE],
    /// Number of scan positions covered by the decode: on the EOB
    /// exit, the EOB scan position itself (no coefficient is emitted
    /// there); on the natural full-block exit, [`BLOCK_SIZE`].
    /// Invariant: `coeffs[coeff_count..]` are all zero. See the
    /// module-level "`EncodedCoeffs` exit choreography" note for how
    /// this maps onto the listing's `++`/`--` pair.
    pub coeff_count: usize,
}

impl BlockCoeffs {
    /// Map this scan-order block to raster order and §15-dequantize
    /// it in one pass. See [`dequantize_to_raster`].
    pub fn dequantize_to_raster(
        &self,
        scan_to_raster: &[u8; BLOCK_SIZE],
        dequant: DequantContext,
    ) -> [i32; BLOCK_SIZE] {
        dequantize_to_raster(&self.coeffs, scan_to_raster, dequant)
    }
}

/// An 8x8 coefficient block mapped to raster order and dequantized —
/// the direct input of the §16 [`crate::idct_block`] transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DequantizedBlock {
    /// The 64 dequantized coefficients in raster order.
    pub raster: [i32; BLOCK_SIZE],
    /// The [`BlockCoeffs::coeff_count`] of the underlying entropy
    /// decode (scan positions covered before the block ended).
    pub coeff_count: usize,
}

/// Decode one full 8x8 block of DCT coefficients under the arithmetic
/// entropy scheme: §13.2.1 DC, then the §13.3.1 per-block AC loop
/// with §13.3.3.1 zero runs.
///
/// * `plane` — the Table 28 plane of the block (`Y` or `UV`); selects
///   the first dimension of `ac_probs`.
/// * `dc_node_probs` — the already-context-resolved
///   `DcNodeContexts[Plane][Context]` row for this block (the §14 DC
///   prediction direction context and the
///   [`crate::tokens::dc_probs_to_node_contexts`] conversion are the
///   caller's responsibility, exactly as for [`decode_dc`]).
/// * `ac_probs` — the persistent §13.3 `AcProbs` decoding bank,
///   `[plane][prec][band][node]` (see [`AcProbBank`] and the
///   module-level layout note).
/// * `zrl_probs` — the persistent §13.3.3 `ZeroRunProbs[2][14]` bank
///   ([`ZeroRunProbBank`]).
///
/// Per the §13.3.1 listing, the probability row is re-selected on
/// every iteration from the live `Prec` context and the Table 30 band
/// of the current scan position; the implicit-1 shortcut
/// (`EncodedCoeffs > 1 && Prec == 0`), the per-leaf `Prec` updates,
/// and the ZERO-leaf transition into the §13.3.3.1 zero-run decoder
/// (band per `ZrlBand[EncodedCoeffs]`, run length inclusive of the
/// triggering position) are all applied by the loop. A zero run may
/// carry `EncodedCoeffs` past the end of the block (`9 + b(6)` runs
/// reach 72); the excess positions simply don't exist and
/// `coeff_count` saturates at [`BLOCK_SIZE`].
///
/// Returns [`Error::Truncated`] if the byte stream is exhausted in
/// any constituent BoolCoder call.
pub fn decode_block_coefficients(
    bc: &mut BoolCoder<'_>,
    plane: AcPlane,
    dc_node_probs: &[u8; NUM_TREE_NODES],
    ac_probs: &AcProbBank,
    zrl_probs: &ZeroRunProbBank,
) -> Result<BlockCoeffs, Error> {
    // "Set CoeffData to 64 0's"
    let mut coeffs = [0i32; BLOCK_SIZE];

    // §13.2.1 — the DC coefficient seeds CoeffData[0]...
    let dc = decode_dc(bc, dc_node_probs)?;
    coeffs[0] = dc;

    // ...and the §13.3.1 Prec context ("the decoded value of the DC
    // coefficient is used as contextual information for the first AC
    // coefficient").
    let mut prec = AcPrecContext::seed_from_dc(dc);
    let mut encoded_coeffs: usize = 1;

    // The listing is a do-while with EncodedCoeffs starting at 1;
    // since 1 < BLOCK_SIZE the first iteration is unconditional, so a
    // top-tested loop is equivalent.
    while encoded_coeffs < BLOCK_SIZE {
        // ProbPtr = AcProbs[Plane][Prec][ACProbBand[EncodedCoeffs]]
        // (see the module-level layout note for the listing's
        // `AcUpdateProbs[Prec][Plane][Band]` naming slip).
        let band = AcBand::for_coefficient_position(encoded_coeffs)
            .expect("scan positions 1..=63 always map to a Table 30 band");
        let node_probs = &ac_probs[plane.index()][prec.index()][band.index()];

        match decode_ac_coefficient(bc, prec, encoded_coeffs, node_probs)? {
            AcOutcome::EndOfBlock => {
                // Listing: `EncodedCoeffs++ ; break`, cancelled by the
                // post-loop `EncodedCoeffs--` — net: the count stays
                // at the EOB scan position.
                break;
            }
            AcOutcome::ZeroRun => {
                // Listing: `Prec = 0` then `EncodedCoeffs +=
                // ZeroRunCount` (the run is inclusive of the position
                // whose ZERO leaf triggered it — no separate `++`).
                prec = AcPrecContext::WasZero;
                let zrl_band = ZrlBand::for_coefficient_position(encoded_coeffs)
                    .expect("scan positions 1..=63 always map to a Table 37 band");
                let run = decode_ac_zero_run(bc, zrl_band, &zrl_probs[zrl_band.index()])?;
                encoded_coeffs += run as usize;
            }
            AcOutcome::Value { coeff, next_prec } => {
                // Listing: `CoeffData[EncodedCoeffs] = (value ^
                // -SignBit) + SignBit` (done inside the primitive),
                // the per-leaf Prec update, then `EncodedCoeffs++`.
                coeffs[encoded_coeffs] = coeff;
                prec = next_prec;
                encoded_coeffs += 1;
            }
        }
    }

    Ok(BlockCoeffs {
        coeffs,
        coeff_count: encoded_coeffs.min(BLOCK_SIZE),
    })
}

/// Map a scan-order coefficient block to raster order and apply the
/// §15 inverse quantizer in a single pass.
///
/// `scan_to_raster` is the active scan-position-to-raster permutation:
/// [`crate::scan::DEFAULT_SCAN_ORDER`] when the frame uses the §12.1
/// default zig-zag, or the output of
/// [`crate::scan_update::custom_scan_order_to_raster`] when a §12.2
/// custom scan order is in effect. Both pin the DC coefficient at
/// position 0 ("In all scan orders the first DCT coefficient is
/// always the DC coefficient"), so scan position 0 always lands on
/// raster position 0 and the §15 DC-vs-AC factor split is identical
/// whether viewed in scan or raster indexing.
///
/// Each raster entry is `dequant.dequantize_coeff(i, scan_coeffs[i])`
/// placed at `scan_to_raster[i]` — coefficient `i = 0` gets the DC
/// factor, `1..=63` the AC factor (§15). Table entries are masked to
/// `0..=63` as defence-in-depth against a malformed permutation
/// (mirroring [`DequantContext::new`]'s `DctQMask` mask); a valid
/// scan table is a permutation of `0..64`, so the mask is inert on
/// conformant input.
pub fn dequantize_to_raster(
    scan_coeffs: &[i32; BLOCK_SIZE],
    scan_to_raster: &[u8; BLOCK_SIZE],
    dequant: DequantContext,
) -> [i32; BLOCK_SIZE] {
    let mut raster = [0i32; BLOCK_SIZE];
    for (i, (&coeff, &pos)) in scan_coeffs.iter().zip(scan_to_raster.iter()).enumerate() {
        raster[(pos & 0x3F) as usize] = dequant.dequantize_coeff(i, coeff);
    }
    raster
}

/// One-shot per-block coefficient reconstruction: entropy decode
/// ([`decode_block_coefficients`]), scan-to-raster mapping and §15
/// dequantization ([`dequantize_to_raster`]) composed. The returned
/// [`DequantizedBlock::raster`] feeds the §16 [`crate::idct_block`]
/// transform directly.
///
/// Parameters are those of the two constituent stages; see their
/// documentation.
pub fn decode_block_to_raster(
    bc: &mut BoolCoder<'_>,
    plane: AcPlane,
    dc_node_probs: &[u8; NUM_TREE_NODES],
    ac_probs: &AcProbBank,
    zrl_probs: &ZeroRunProbBank,
    scan_to_raster: &[u8; BLOCK_SIZE],
    dequant: DequantContext,
) -> Result<DequantizedBlock, Error> {
    let block = decode_block_coefficients(bc, plane, dc_node_probs, ac_probs, zrl_probs)?;
    Ok(DequantizedBlock {
        raster: block.dequantize_to_raster(scan_to_raster, dequant),
        coeff_count: block.coeff_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::{zigzag_to_raster_block, DEFAULT_SCAN_ORDER};
    use crate::scan_update::{build_custom_scan_order, custom_scan_order_to_raster};
    use crate::tokens::{
        baseline_ac_probs, baseline_dc_probs, dc_probs_to_node_contexts, TreeNode,
    };
    use crate::zrl::ZERO_RUN_PROB_DEFAULTS;

    /// Largest representable coefficient magnitude (Table 18
    /// CATEGORY6 Max).
    const MAX_COEFF_MAGNITUDE: i32 = 2114;

    fn baseline_dc_node_probs() -> [u8; NUM_TREE_NODES] {
        dc_probs_to_node_contexts(&baseline_dc_probs())[0][0]
    }

    /// An independent replay of the §13.3.1 listing built directly on
    /// the per-coefficient primitives, written in the listing's
    /// do-while shape (explicit `Prec` variable, per-iteration
    /// `ProbPtr` re-selection, EOB `++`/post-loop `--` pair). Used to
    /// pin the driver's context threading bit-for-bit, including the
    /// final BoolCoder byte position.
    #[allow(clippy::type_complexity)]
    fn spec_replay(
        bytes: &[u8],
        plane: AcPlane,
        dc_node_probs: &[u8; NUM_TREE_NODES],
        ac_probs: &AcProbBank,
        zrl_probs: &ZeroRunProbBank,
    ) -> Result<([i32; BLOCK_SIZE], usize, usize), Error> {
        let mut bc = BoolCoder::new(bytes)?;
        let mut coeff_data = [0i32; BLOCK_SIZE];
        coeff_data[0] = decode_dc(&mut bc, dc_node_probs)?;
        let mut prec = if coeff_data[0] == 0 {
            AcPrecContext::WasZero
        } else if coeff_data[0] == 1 {
            AcPrecContext::WasOne
        } else {
            AcPrecContext::WasGreaterThanOne
        };
        let mut encoded_coeffs: usize = 1;
        loop {
            let band = AcBand::for_coefficient_position(encoded_coeffs).unwrap();
            let probs = &ac_probs[plane.index()][prec.index()][band.index()];
            match decode_ac_coefficient(&mut bc, prec, encoded_coeffs, probs)? {
                AcOutcome::EndOfBlock => {
                    encoded_coeffs += 1; // listing: ++ then break...
                    encoded_coeffs -= 1; // ...cancelled by post-loop --
                    break;
                }
                AcOutcome::ZeroRun => {
                    prec = AcPrecContext::WasZero;
                    let zrl_band = ZrlBand::for_coefficient_position(encoded_coeffs).unwrap();
                    let run = decode_ac_zero_run(&mut bc, zrl_band, &zrl_probs[zrl_band.index()])?;
                    encoded_coeffs += run as usize;
                }
                AcOutcome::Value { coeff, next_prec } => {
                    coeff_data[encoded_coeffs] = coeff;
                    prec = next_prec;
                    encoded_coeffs += 1;
                }
            }
            if encoded_coeffs >= BLOCK_SIZE {
                break;
            }
        }
        Ok((coeff_data, encoded_coeffs.min(BLOCK_SIZE), bc.pos()))
    }

    /// All-zero stream against the keyframe baselines: every
    /// BoolCoder decision takes the 0-branch, so the DC root lands on
    /// ZERO (DC = 0), the first AC walk takes `ZERO_CONTEXT_NODE = 0`
    /// then `EOB_CONTEXT_NODE = 0` → EOB at scan position 1. The
    /// block is empty and `coeff_count` is 1 (only the DC position is
    /// covered).
    #[test]
    fn all_zero_stream_decodes_empty_block() {
        let bytes = [0u8; 16];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_coefficients(
            &mut bc,
            AcPlane::Y,
            &baseline_dc_node_probs(),
            &baseline_ac_probs(),
            &ZERO_RUN_PROB_DEFAULTS,
        )
        .expect("not truncated");
        assert_eq!(block.coeffs, [0i32; BLOCK_SIZE]);
        assert_eq!(block.coeff_count, 1, "EOB at the first AC position");
    }

    /// DC-then-immediate-EOB trace, hand-computed from the §7.3
    /// arithmetic. Stream `0x40 0x40 0x40 0x40 …`; DC node probs with
    /// `ZERO = 1` (Split = 2, Value 0x40404040 ≥ 0x02000000 → 1 →
    /// non-zero), `ONE = 255` (Split = 501 > Range → 0-branch →
    /// magnitude-1 leaf), sign `b(1)` 0 (top byte stays below Range)
    /// → DC = +1, seeding `Prec = WasOne`. AC row `[Y][WasOne][band0]`
    /// has `ZERO = 255` (Split ≈ 2·Range → 0) and `EOB = 255`
    /// (likewise 0) → EOB leaf immediately. Net: DC-only block.
    #[test]
    fn dc_only_block_eob_at_first_ac() {
        let bytes = [0x40u8; 16];
        let mut dc_node_probs = [128u8; NUM_TREE_NODES];
        dc_node_probs[TreeNode::Zero.index()] = 1; // → 1-branch
        dc_node_probs[TreeNode::One.index()] = 255; // → 0-branch (mag 1)
        let mut ac_probs = baseline_ac_probs();
        for row in &mut ac_probs[0][AcPrecContext::WasOne.index()] {
            row[TreeNode::Zero.index()] = 255;
            row[TreeNode::EndOfBlock.index()] = 255;
        }
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_coefficients(
            &mut bc,
            AcPlane::Y,
            &dc_node_probs,
            &ac_probs,
            &ZERO_RUN_PROB_DEFAULTS,
        )
        .expect("not truncated");
        assert_eq!(block.coeffs[0], 1, "DC magnitude-1 leaf, positive sign");
        assert_eq!(block.coeff_count, 1, "EOB at scan position 1");
        assert!(block.coeffs[1..].iter().all(|&c| c == 0));
    }

    /// The DC-seeded `Prec` context must select the probability row
    /// for the first AC coefficient. The `[Y][WasOne]` rows force the
    /// value path (`ZERO = 1` → non-zero, `ONE = 255` → magnitude-1
    /// leaf) while the `[Y][WasZero]` rows force EOB (`ZERO = 255`,
    /// `EOB = 255`). With DC = +1 (same trace as above) the driver
    /// must use the `WasOne` row and emit a non-zero first AC; using
    /// the `WasZero` row instead would terminate the block empty.
    #[test]
    fn dc_seeded_prec_selects_first_ac_row() {
        let bytes = [0x40u8; 64];
        let mut dc_node_probs = [128u8; NUM_TREE_NODES];
        dc_node_probs[TreeNode::Zero.index()] = 1;
        dc_node_probs[TreeNode::One.index()] = 255;
        let mut ac_probs = baseline_ac_probs();
        // WasZero rows: EOB immediately.
        for row in &mut ac_probs[0][AcPrecContext::WasZero.index()] {
            row[TreeNode::Zero.index()] = 255;
            row[TreeNode::EndOfBlock.index()] = 255;
        }
        // WasOne rows: magnitude-1 value leaf.
        for row in &mut ac_probs[0][AcPrecContext::WasOne.index()] {
            row[TreeNode::Zero.index()] = 1;
            row[TreeNode::One.index()] = 255;
        }
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_coefficients(
            &mut bc,
            AcPlane::Y,
            &dc_node_probs,
            &ac_probs,
            &ZERO_RUN_PROB_DEFAULTS,
        )
        .expect("not truncated");
        assert_eq!(block.coeffs[0], 1);
        assert!(
            block.coeff_count > 1,
            "WasOne row must be selected after DC = 1 (got count {})",
            block.coeff_count
        );
        assert_eq!(
            block.coeffs[1].unsigned_abs(),
            1,
            "first AC must be the magnitude-1 leaf the WasOne row forces"
        );
    }

    /// Zero-run integration: rows tuned so the `WasGreaterThanOne`
    /// context (seeded by a large DC) lands on the ZERO leaf
    /// (`ZERO = 255` → 0-branch, `EOB = 1` → 1-branch → zero run) and
    /// the ZRL bank forces the literal run length 1 (all tree nodes
    /// 255 → every gate 0). After the run, `Prec = WasZero` makes the
    /// implicit-1 shortcut skip the root decision and the `WasZero`
    /// row forces a magnitude-1 value (`ONE = 255` → 0-branch), whose
    /// `WasOne` successor row then forces EOB. Exercises every
    /// AcOutcome arm in one block.
    #[test]
    fn zero_run_then_value_then_eob() {
        let bytes = [0x40u8; 64];
        let mut dc_node_probs = [128u8; NUM_TREE_NODES];
        dc_node_probs[TreeNode::Zero.index()] = 1; // non-zero DC
        dc_node_probs[TreeNode::One.index()] = 1; // 1-branch → LowVal subtree
        dc_node_probs[TreeNode::LowVal.index()] = 255; // 0-branch → TWO/THREE/FOUR
        dc_node_probs[TreeNode::Two.index()] = 255; // 0-branch → TWO_TOKEN (DC = ±2)
        let mut ac_probs = baseline_ac_probs();
        for row in &mut ac_probs[0][AcPrecContext::WasGreaterThanOne.index()] {
            row[TreeNode::Zero.index()] = 255; // 0-branch
            row[TreeNode::EndOfBlock.index()] = 1; // 1-branch → zero run
        }
        for row in &mut ac_probs[0][AcPrecContext::WasZero.index()] {
            row[TreeNode::One.index()] = 255; // magnitude 1
        }
        for row in &mut ac_probs[0][AcPrecContext::WasOne.index()] {
            row[TreeNode::Zero.index()] = 255; // ZERO → 0
            row[TreeNode::EndOfBlock.index()] = 255; // EOB
        }
        let zrl_probs = [[255u8; NUM_ZRL_NODES]; NUM_ZRL_BANDS];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block =
            decode_block_coefficients(&mut bc, AcPlane::Y, &dc_node_probs, &ac_probs, &zrl_probs)
                .expect("not truncated");
        assert_eq!(block.coeffs[0].unsigned_abs(), 2, "TWO_TOKEN DC");
        assert_eq!(block.coeffs[1], 0, "zero-run position");
        assert_eq!(
            block.coeffs[2].unsigned_abs(),
            1,
            "implicit-1 value after the run"
        );
        assert_eq!(block.coeff_count, 3, "EOB at scan position 3");
        assert!(block.coeffs[3..].iter().all(|&c| c == 0));
    }

    /// Driver-vs-listing replay equality over a (seed-stream ×
    /// probability-bank) grid: coefficients, count and the final
    /// BoolCoder byte position must all match the independent
    /// do-while replay, for both planes.
    #[test]
    fn driver_matches_spec_replay_across_streams_and_banks() {
        let seeds: [u8; 6] = [0x00, 0x33, 0x40, 0x55, 0x77, 0x5A];
        let mut banks: Vec<AcProbBank> = vec![baseline_ac_probs()];
        // A bank biased toward value leaves at moderate probabilities
        // (both BoolCoder branches well-conditioned).
        let mut value_bank = baseline_ac_probs();
        for plane in value_bank.iter_mut() {
            for prec in plane.iter_mut() {
                for row in prec.iter_mut() {
                    row[TreeNode::Zero.index()] = 64;
                    row[TreeNode::EndOfBlock.index()] = 64;
                    row[TreeNode::One.index()] = 200;
                }
            }
        }
        banks.push(value_bank);
        let dc_node_probs = baseline_dc_node_probs();
        for bank in &banks {
            for &seed in &seeds {
                for plane in [AcPlane::Y, AcPlane::UV] {
                    let bytes = vec![seed; 96];
                    let replay =
                        spec_replay(&bytes, plane, &dc_node_probs, bank, &ZERO_RUN_PROB_DEFAULTS);
                    let mut bc = BoolCoder::new(&bytes).unwrap();
                    let driven = decode_block_coefficients(
                        &mut bc,
                        plane,
                        &dc_node_probs,
                        bank,
                        &ZERO_RUN_PROB_DEFAULTS,
                    );
                    match (replay, driven) {
                        (Ok((coeffs, count, pos)), Ok(block)) => {
                            assert_eq!(block.coeffs, coeffs, "seed {seed:#04x} plane {plane}");
                            assert_eq!(block.coeff_count, count, "seed {seed:#04x} plane {plane}");
                            assert_eq!(bc.pos(), pos, "seed {seed:#04x} plane {plane}");
                        }
                        (Err(e1), Err(e2)) => assert_eq!(e1, e2),
                        (r, d) => {
                            panic!("replay/driver divergence seed {seed:#04x}: {r:?} vs {d:?}")
                        }
                    }
                }
            }
        }
    }

    /// Structural invariants over arbitrary seed streams against the
    /// baselines: when the decode succeeds, `coeff_count` is in
    /// `1..=64`, everything from `coeff_count` onward is zero, and
    /// every coefficient respects the Table 18 magnitude bound.
    /// `Truncated` is an acceptable outcome for arbitrary input.
    #[test]
    fn invariants_hold_across_seed_streams() {
        let dc_node_probs = baseline_dc_node_probs();
        let ac_probs = baseline_ac_probs();
        for seed in [0x00u8, 0x11, 0x29, 0x33, 0x40, 0x55, 0x77, 0x88, 0xA5, 0xC3] {
            let bytes = vec![seed; 128];
            let mut bc = BoolCoder::new(&bytes).unwrap();
            if let Ok(block) = decode_block_coefficients(
                &mut bc,
                AcPlane::Y,
                &dc_node_probs,
                &ac_probs,
                &ZERO_RUN_PROB_DEFAULTS,
            ) {
                assert!(
                    (1..=BLOCK_SIZE).contains(&block.coeff_count),
                    "seed {seed:#04x}: count {} out of range",
                    block.coeff_count
                );
                assert!(
                    block.coeffs[block.coeff_count..].iter().all(|&c| c == 0),
                    "seed {seed:#04x}: tail beyond coeff_count must be zero"
                );
                assert!(
                    block.coeffs.iter().all(|&c| c.abs() <= MAX_COEFF_MAGNITUDE),
                    "seed {seed:#04x}: coefficient out of Table 18 range"
                );
            }
        }
    }

    /// Two independent runs over the same bytes and banks produce
    /// identical blocks (pure function of the stream).
    #[test]
    fn decode_is_deterministic() {
        let bytes = [0x5Au8, 0xA5, 0x3C, 0xC3, 0x69, 0x96, 0x0F, 0xF0].repeat(12);
        let dc_node_probs = baseline_dc_node_probs();
        let ac_probs = baseline_ac_probs();
        let run = || {
            let mut bc = BoolCoder::new(&bytes).unwrap();
            decode_block_coefficients(
                &mut bc,
                AcPlane::UV,
                &dc_node_probs,
                &ac_probs,
                &ZERO_RUN_PROB_DEFAULTS,
            )
        };
        assert_eq!(run(), run());
    }

    /// A 4-byte stream exhausts under a read-heavy bank: the decode
    /// must surface `Error::Truncated` rather than panic.
    #[test]
    fn truncated_stream_surfaces_error() {
        let bytes = [0x40u8, 0x40, 0x40, 0x40];
        // Every tree-node probability pinned to 1 biases each walk
        // deep into the CATEGORY6 leaf (1-branches while the top byte
        // of Value exceeds the tiny Split), and each CATEGORY6 leaf
        // costs 11 magnitude bits plus a sign — the renormalization
        // loop has to pull bytes long before 63 coefficients land.
        let dc_node_probs = [1u8; NUM_TREE_NODES];
        let mut ac_probs = baseline_ac_probs();
        for plane in ac_probs.iter_mut() {
            for prec in plane.iter_mut() {
                for row in prec.iter_mut() {
                    *row = [1u8; NUM_TREE_NODES];
                }
            }
        }
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let r = decode_block_coefficients(
            &mut bc,
            AcPlane::Y,
            &dc_node_probs,
            &ac_probs,
            &ZERO_RUN_PROB_DEFAULTS,
        );
        assert_eq!(r, Err(Error::Truncated));
    }

    // ---- scan + dequant composition ----

    /// Against the §12.1 default scan, `dequantize_to_raster` must
    /// equal the existing two-step composition
    /// `zigzag_to_raster_block` → `DequantContext::dequantize_block`.
    #[test]
    fn dequantize_to_raster_matches_two_step_default_scan() {
        let mut scan_coeffs = [0i32; BLOCK_SIZE];
        for (i, c) in scan_coeffs.iter_mut().enumerate() {
            *c = (i as i32 - 31) * 7; // mixed signs, distinct values
        }
        for mask in [0u8, 17, 63] {
            let dq = DequantContext::new(mask);
            let fused = dequantize_to_raster(&scan_coeffs, &DEFAULT_SCAN_ORDER, dq);
            let mut two_step = [0i32; BLOCK_SIZE];
            zigzag_to_raster_block(&scan_coeffs, &mut two_step);
            dq.dequantize_block(&mut two_step);
            assert_eq!(fused, two_step, "DctQMask {mask}");
        }
    }

    /// DC vs AC factor split: scan position 0 takes the DC factor,
    /// every other position the AC factor, regardless of where the
    /// permutation sends them.
    #[test]
    fn dequantize_factor_split() {
        let mut scan_coeffs = [1i32; BLOCK_SIZE];
        scan_coeffs[0] = 1;
        let dq = DequantContext::new(0); // coarsest: distinct DC/AC factors
        let raster = dequantize_to_raster(&scan_coeffs, &DEFAULT_SCAN_ORDER, dq);
        assert_eq!(raster[0], i32::from(dq.dc_factor));
        for (r, &v) in raster.iter().enumerate().skip(1) {
            assert_eq!(v, i32::from(dq.ac_factor), "raster position {r}");
        }
    }

    /// A §12.2 custom scan order routes coefficients to the raster
    /// positions its permutation dictates: rebuild a modified band
    /// assignment (the spec's AC7/AC21-to-band-3 worked example),
    /// compose to raster, and verify a marker coefficient placed at
    /// modified scan position 11 lands at the raster home of zig-zag
    /// coefficient 7.
    #[test]
    fn dequantize_to_raster_follows_custom_scan() {
        let mut assignment = crate::scan_update::DEFAULT_BAND_ASSIGNMENT;
        assignment[7] = 3;
        assignment[21] = 3;
        assignment[11] = 2;
        assignment[12] = 4;
        let scan = build_custom_scan_order(&assignment);
        let scan_to_raster = custom_scan_order_to_raster(&scan);
        assert_eq!(scan[11], 7, "spec worked example precondition");

        let mut scan_coeffs = [0i32; BLOCK_SIZE];
        scan_coeffs[0] = 5; // DC
        scan_coeffs[11] = 9; // marker at modified position 11
        let dq = DequantContext::new(63); // finest quantizer
        let raster = dequantize_to_raster(&scan_coeffs, &scan_to_raster, dq);
        assert_eq!(
            raster[DEFAULT_SCAN_ORDER[7] as usize],
            9 * i32::from(dq.ac_factor),
            "marker must land at zig-zag coefficient 7's raster home"
        );
        assert_eq!(raster[0], 5 * i32::from(dq.dc_factor));
    }

    /// `BlockCoeffs::dequantize_to_raster` and the free function
    /// agree; the one-shot `decode_block_to_raster` equals the
    /// two-stage composition on the same stream.
    #[test]
    fn one_shot_equals_two_stage() {
        let bytes = [0x40u8; 96];
        let dc_node_probs = baseline_dc_node_probs();
        let ac_probs = baseline_ac_probs();
        let dq = DequantContext::new(21);

        let mut bc1 = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_coefficients(
            &mut bc1,
            AcPlane::Y,
            &dc_node_probs,
            &ac_probs,
            &ZERO_RUN_PROB_DEFAULTS,
        )
        .expect("not truncated");
        let staged = block.dequantize_to_raster(&DEFAULT_SCAN_ORDER, dq);
        assert_eq!(
            staged,
            dequantize_to_raster(&block.coeffs, &DEFAULT_SCAN_ORDER, dq)
        );

        let mut bc2 = BoolCoder::new(&bytes).unwrap();
        let one_shot = decode_block_to_raster(
            &mut bc2,
            AcPlane::Y,
            &dc_node_probs,
            &ac_probs,
            &ZERO_RUN_PROB_DEFAULTS,
            &DEFAULT_SCAN_ORDER,
            dq,
        )
        .expect("not truncated");
        assert_eq!(one_shot.raster, staged);
        assert_eq!(one_shot.coeff_count, block.coeff_count);
        assert_eq!(bc1.pos(), bc2.pos(), "both paths consume the same bits");
    }

    /// The empty (all-zero-stream) block round-trips through the full
    /// reconstruction tail: dequantize → IDCT yields an all-zero
    /// pixel-difference block.
    #[test]
    fn empty_block_idcts_to_zero() {
        let bytes = [0u8; 16];
        let mut bc = BoolCoder::new(&bytes).unwrap();
        let block = decode_block_to_raster(
            &mut bc,
            AcPlane::Y,
            &baseline_dc_node_probs(),
            &baseline_ac_probs(),
            &ZERO_RUN_PROB_DEFAULTS,
            &DEFAULT_SCAN_ORDER,
            DequantContext::new(0),
        )
        .expect("not truncated");
        let mut pixels = block.raster;
        crate::idct_block(&mut pixels);
        assert_eq!(pixels, [0i32; BLOCK_SIZE]);
    }
}

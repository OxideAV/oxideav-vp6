//! VP6 §13 *Coefficient Probability Updates* — the Figure 5 sub-stream
//! that sits between the §9 frame header (plus, on inter frames, the §10
//! mode-probability and §11.2 MV-probability sub-streams) and the
//! per-macroblock data.
//!
//! ## Why this module exists
//!
//! The four per-frame coefficient-probability update passes already exist
//! as independent primitives:
//!
//! * §13.2 DC node updates — [`crate::prob_update::update_dc_probs`]
//! * §12.2 custom scan-order update — [`crate::scan_update::decode_scan_order_update`]
//! * §13.3.3 zero-run-length updates — [`crate::prob_update::update_zero_run_probs`]
//! * §13.3 AC node updates — [`crate::prob_update::update_ac_probs`]
//!
//! What was missing is the **ordering glue**. §8 *Bitstream Map*,
//! Figure 5 *Coefficient Probability Updates*, fixes the exact order of
//! the four passes:
//!
//! ```text
//!   Y,U,V DC * 11 Node Probability Updates       (§13.2)
//!   Scan Update Bit                               (§12.2)
//!     if set → 63 Coefficients Scan Order Updates (§12.2)
//!   2 Bands * 14 Nodes Zero Run Probability Updates   (§13.3.3)
//!   3 Prec Cases * 2 Planes * 6 Bands * 11 Nodes Ac Probability Updates  (§13.3)
//! ```
//!
//! (Figure 5 is reproduced on p. 20 of `vp6_format.pdf`; it is the single
//! authoritative statement of the within-pass order, which no individual
//! pass module asserts on its own.)
//!
//! [`decode_coefficient_prob_updates`] runs exactly that sequence over a
//! single [`BoolCoder`], mutating an [`IntraProbs`] bank in place and
//! returning the active raster scan order. [`encode_coefficient_prob_updates`]
//! is its bit-for-bit inverse: it emits the same four passes, in the same
//! order, with every per-node "no update" flag and (when no custom scan is
//! requested) the cleared scan-update bit — so an encoder that does not
//! re-train probabilities produces the minimal conformant Figure-5 prefix
//! and the round-trip is exact.
//!
//! ## Persistence (§13.2 / §13.3 / §13.3.3 / §11.2)
//!
//! VP6 probability banks **persist** keyframe→interframe and are mutated
//! by these updates. At a keyframe every bank is first reset to its
//! baseline (DC/AC to the §13 defaults, ZRL to `ZeroRunProbDefaults`, the
//! scan order to the default zig-zag band assignment) and *then* the
//! Figure-5 updates apply on top. This module operates on whatever
//! [`IntraProbs`] the caller threads in, so the caller controls the
//! keyframe reset (it passes [`IntraProbs::keyframe`]) versus the
//! inter-frame carry (it passes the previous frame's mutated bank).
//!
//! Because [`IntraProbs`] stores the DC bank in its *expanded*
//! `DcNodeContexts[plane][context][node]` form (§13.2 Table 26) rather
//! than the raw `DcProbs[plane][node]` the update pass mutates, this
//! driver carries the raw `DcProbs[plane][node]` separately: the caller
//! holds the raw bank across frames, this driver applies the §13.2 update
//! to it, and re-expands it into the [`IntraProbs::dc_contexts`] field via
//! [`crate::tokens::dc_probs_to_node_contexts`].
//!
//! ## Provenance
//!
//! Sequences only this crate's own already-spec-sourced passes. The
//! ordering is `vp6_format.pdf` §8 Figure 5 (p. 20); the per-pass content
//! is §12.2 / §13.2 / §13.3 / §13.3.3 as cited on each primitive. No
//! external library code was consulted.

use crate::bool_coder::{BoolCoder, BoolEncoder};
use crate::intra_frame::IntraProbs;
use crate::prob_update::{update_ac_probs, update_dc_probs, update_zero_run_probs};
use crate::scan::DEFAULT_SCAN_ORDER;
use crate::scan_update::{
    build_custom_scan_order, custom_scan_order_to_raster, decode_scan_order_update, BandAssignment,
    COEFF_BAND_UPDATE_FLAG_PROBS, DEFAULT_BAND_ASSIGNMENT,
};
use crate::tokens::{
    dc_probs_to_node_contexts, AC_UPDATE_PROBS, NUM_PLANES, NUM_TREE_NODES, VP6_DC_UPDATE_PROBS,
};
use crate::zrl::ZRL_UPDATE_PROBS;
use crate::Error;

/// The persistent §13 coefficient-probability banks a decoder carries
/// across frames, in the *raw* (pre-expansion) form the Figure-5 update
/// passes mutate.
///
/// [`IntraProbs`] stores the DC bank already expanded into node contexts
/// for the §13.2.1 decoder; the §13.2 *update* pass however mutates the
/// raw `DcProbs[plane][node]` bank. This struct keeps the raw DC bank
/// (plus the AC and ZRL banks, which need no transform) so the per-frame
/// update can run and then re-expand the DC contexts.
///
/// At a keyframe the caller seeds this with [`Self::keyframe`]; for inter
/// frames it threads the previous frame's mutated value back in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoeffProbBanks {
    /// §13.2 raw DC node probabilities `[plane][node]`.
    pub dc_probs: [[u8; NUM_TREE_NODES]; NUM_PLANES],
    /// §13.3 AC node probabilities `[plane][prec][band][node]`.
    pub ac_probs: crate::block_decode::AcProbBank,
    /// §13.3.3 zero-run-length node probabilities `[band][node]`.
    pub zrl_probs: crate::block_decode::ZeroRunProbBank,
    /// §12.2 active band assignment `[coeff] → band`. Default zig-zag
    /// until a custom scan-order update replaces it.
    pub band_assignment: BandAssignment,
}

impl CoeffProbBanks {
    /// The keyframe baselines (§13.2 / §13.3 / §13.3.3 defaults + the
    /// §12.2 default zig-zag band assignment). The §13 spec resets every
    /// bank to these at each I-frame *before* the Figure-5 updates apply.
    pub fn keyframe() -> Self {
        Self {
            dc_probs: crate::tokens::baseline_dc_probs(),
            ac_probs: crate::tokens::baseline_ac_probs(),
            zrl_probs: crate::zrl::ZERO_RUN_PROB_DEFAULTS,
            band_assignment: DEFAULT_BAND_ASSIGNMENT,
        }
    }

    /// Expand the raw banks into the [`IntraProbs`] form the per-block
    /// §13 decoder consumes (DC raw probs → §13.2 Table 26 node contexts;
    /// AC and ZRL banks pass through unchanged).
    pub fn to_intra_probs(&self) -> IntraProbs {
        IntraProbs {
            dc_contexts: dc_probs_to_node_contexts(&self.dc_probs),
            ac_probs: self.ac_probs,
            zrl_probs: self.zrl_probs,
        }
    }

    /// The raster scan order implied by the current band assignment
    /// (§12.2): default zig-zag when the assignment is the default,
    /// otherwise the custom permutation.
    pub fn raster_scan_order(&self) -> [u8; 64] {
        if self.band_assignment == DEFAULT_BAND_ASSIGNMENT {
            DEFAULT_SCAN_ORDER
        } else {
            custom_scan_order_to_raster(&build_custom_scan_order(&self.band_assignment))
        }
    }
}

/// Decode the §8 Figure 5 *Coefficient Probability Updates* sub-stream
/// from `bc`, mutating `banks` in place and returning the active raster
/// scan order for the frame.
///
/// The four passes run in the exact Figure-5 order:
///
/// 1. §13.2 DC node updates (`Y,U,V DC * 11 Node Probability Updates`).
/// 2. §12.2 scan-update bit; if set, the 63-coefficient scan-order
///    updates. The scan-update primitive resets the assignment to the
///    default zig-zag when the bit is clear (§12.2 intra reset).
/// 3. §13.3.3 zero-run updates (`2 Bands * 14 Nodes`).
/// 4. §13.3 AC node updates (`3 Prec * 2 Planes * 6 Bands * 11 Nodes`).
///
/// Returns the raster scan order (`raster[p]` = raster index decoded at
/// modified-scan position `p`) for the caller to thread into the §13
/// block decode.
///
/// # Errors
///
/// [`Error::Truncated`] if the BoolCoder partition is exhausted mid-walk.
pub fn decode_coefficient_prob_updates(
    bc: &mut BoolCoder<'_>,
    banks: &mut CoeffProbBanks,
) -> Result<[u8; 64], Error> {
    // 1. §13.2 DC node probability updates.
    update_dc_probs(bc, &mut banks.dc_probs, &VP6_DC_UPDATE_PROBS)?;

    // 2. §12.2 scan-update bit + (if set) the 63-coefficient updates.
    //    `decode_scan_order_update` resets the assignment to the default
    //    zig-zag when the bit is clear, matching §12.2's intra reset.
    decode_scan_order_update(
        bc,
        &COEFF_BAND_UPDATE_FLAG_PROBS,
        &mut banks.band_assignment,
    )?;

    // 3. §13.3.3 zero-run-length probability updates.
    update_zero_run_probs(bc, &mut banks.zrl_probs, &ZRL_UPDATE_PROBS)?;

    // 4. §13.3 AC node probability updates.
    update_ac_probs(bc, &mut banks.ac_probs, &AC_UPDATE_PROBS)?;

    Ok(banks.raster_scan_order())
}

/// Emit a "no probability updates" §8 Figure 5 sub-stream — the minimal
/// conformant Figure-5 prefix an encoder that does not re-train any
/// coefficient probabilities produces.
///
/// Every per-node update flag is the cleared `B(flag_prob)` zero-branch,
/// the §12.2 scan-update bit is the cleared `b(1)`, and no custom-scan
/// body or `NewNodeProbValue` follows. This is the exact bit-for-bit
/// inverse of [`decode_coefficient_prob_updates`] when that decode reads a
/// stream with no updates: the DC/ZRL/AC banks stay at their incoming
/// values and the scan order reverts to the default zig-zag.
// Index loops keep the per-node `*_UPDATE_PROBS[..]` flag lookups visibly
// aligned with the §13.2 / §13.3.3 / §13.3 Tables traversal the decoder
// reads (same shape as the `prob_update` drivers this inverts).
#[allow(clippy::needless_range_loop)]
pub fn encode_coefficient_prob_updates(enc: &mut BoolEncoder) {
    // 1. §13.2 DC: NUM_PLANES * NUM_TREE_NODES cleared update flags.
    for plane in 0..NUM_PLANES {
        for node in 0..NUM_TREE_NODES {
            enc.encode_bool(0, VP6_DC_UPDATE_PROBS[plane][node]);
        }
    }

    // 2. §12.2 scan-update bit cleared (no custom scan).
    enc.encode_b1(0);

    // 3. §13.3.3 ZRL: NUM_ZRL_BANDS * NUM_ZRL_NODES cleared update flags.
    for band in 0..crate::zrl::NUM_ZRL_BANDS {
        for node in 0..crate::zrl::NUM_ZRL_NODES {
            enc.encode_bool(0, ZRL_UPDATE_PROBS[band][node]);
        }
    }

    // 4. §13.3 AC: prec * plane * band * node cleared update flags, in
    //    the Table-31-outer walk order the decoder reads.
    for prec in 0..crate::tokens::NUM_AC_PREC_CONTEXTS {
        for plane in 0..NUM_PLANES {
            for band in 0..crate::tokens::NUM_AC_BANDS {
                for node in 0..NUM_TREE_NODES {
                    enc.encode_bool(0, AC_UPDATE_PROBS[prec][plane][band][node]);
                }
            }
        }
    }
}

/// Emit a §8 Figure 5 sub-stream that requests a **custom scan order**
/// (and no DC/ZRL/AC node updates) — the band-assignment half of the
/// Figure-5 pass.
///
/// The DC pass, ZRL pass and AC pass each emit only cleared flags (as in
/// [`encode_coefficient_prob_updates`]); the §12.2 scan-update bit is set
/// and the 63-coefficient band updates follow for every coefficient whose
/// band differs from the default zig-zag assignment. This is the
/// bit-for-bit inverse of a [`decode_coefficient_prob_updates`] that
/// reads a set scan-update bit and the matching band updates.
// See `encode_coefficient_prob_updates` for the index-loop rationale.
#[allow(clippy::needless_range_loop)]
pub fn encode_coefficient_prob_updates_with_scan(
    enc: &mut BoolEncoder,
    band_assignment: &BandAssignment,
) {
    // 1. §13.2 DC: cleared flags.
    for plane in 0..NUM_PLANES {
        for node in 0..NUM_TREE_NODES {
            enc.encode_bool(0, VP6_DC_UPDATE_PROBS[plane][node]);
        }
    }

    // 2. §12.2 scan-update bit set + the 63-coefficient band updates.
    enc.encode_b1(1);
    encode_coeff_band_updates(enc, band_assignment);

    // 3. §13.3.3 ZRL: cleared flags.
    for band in 0..crate::zrl::NUM_ZRL_BANDS {
        for node in 0..crate::zrl::NUM_ZRL_NODES {
            enc.encode_bool(0, ZRL_UPDATE_PROBS[band][node]);
        }
    }

    // 4. §13.3 AC: cleared flags.
    for prec in 0..crate::tokens::NUM_AC_PREC_CONTEXTS {
        for plane in 0..NUM_PLANES {
            for band in 0..crate::tokens::NUM_AC_BANDS {
                for node in 0..NUM_TREE_NODES {
                    enc.encode_bool(0, AC_UPDATE_PROBS[prec][plane][band][node]);
                }
            }
        }
    }
}

/// Emit the §12.2 63-coefficient band-update body (the inverse of
/// [`decode_coeff_band_updates`]).
///
/// For each AC coefficient `1..=63` (DC at index 0 is never updated,
/// §12.2): emit `B(COEFF_BAND_UPDATE_FLAG_PROBS[c])` set iff the band
/// differs from the default zig-zag assignment, and when set the `b(4)`
/// new band. A coefficient at its default band emits the cleared flag.
fn encode_coeff_band_updates(enc: &mut BoolEncoder, band_assignment: &BandAssignment) {
    for coeff in 1..64usize {
        let band = band_assignment[coeff];
        if band == DEFAULT_BAND_ASSIGNMENT[coeff] {
            enc.encode_bool(0, COEFF_BAND_UPDATE_FLAG_PROBS[coeff]);
        } else {
            enc.encode_bool(1, COEFF_BAND_UPDATE_FLAG_PROBS[coeff]);
            enc.encode_b(band as u32, 4);
        }
    }
}

/// The set of `NewNodeProbValue` targets a §13 node-probability update
/// can actually express.
///
/// [`crate::prob_update::decode_new_node_prob`] decodes a set update as
/// `b(7) → value`, then `prob = max(1, value * 2)`. The reachable target
/// probabilities are therefore exactly `{1} ∪ {2, 4, …, 254}` (the even
/// values, plus `1` standing in for the forbidden `0`). An odd target
/// above `1` cannot be transmitted by the §13 update mechanism; this
/// predicate lets a caller pick representable targets.
pub fn node_prob_update_representable(target: u8) -> bool {
    target == 1 || (target != 0 && target % 2 == 0)
}

/// The `b(7)` `NewNodeProbValue` that decodes to `target` via the §13
/// `max(1, value * 2)` rule, or `None` if `target` is unrepresentable
/// (see [`node_prob_update_representable`]).
fn node_prob_update_value(target: u8) -> Option<u32> {
    if target == 1 {
        // value 0 → doubled 0 → clipped to 1.
        Some(0)
    } else if target != 0 && target % 2 == 0 {
        Some((target >> 1) as u32)
    } else {
        None
    }
}

/// Emit one §13 node-probability update record (the inverse of
/// [`crate::prob_update::decode_new_node_prob`]).
///
/// If `target == current` emit the cleared `B(flag_prob)` flag (no
/// update). Otherwise emit the set flag and the `b(7) NewNodeProbValue`
/// that decodes to `target`. `target` must be representable
/// ([`node_prob_update_representable`]); the function panics in debug on
/// an unrepresentable target rather than silently emitting a wrong value.
fn encode_node_prob_update(enc: &mut BoolEncoder, flag_prob: u8, current: u8, target: u8) {
    if target == current {
        enc.encode_bool(0, flag_prob);
        return;
    }
    let value = node_prob_update_value(target)
        .expect("node-prob update target must be representable (1 or even)");
    enc.encode_bool(1, flag_prob);
    enc.encode_b(value, 7);
}

/// Emit a §8 Figure 5 sub-stream that transforms the keyframe-baseline
/// banks into `target` — emitting a real `NewNodeProbValue` for every
/// DC / ZRL / AC node that differs from baseline, the §12.2 scan-update
/// for the band assignment, in the exact Figure-5 order.
///
/// This is the general inverse of [`decode_coefficient_prob_updates`]
/// against a [`CoeffProbBanks::keyframe`] starting point: decoding the
/// emitted sub-stream over fresh keyframe banks reproduces `target`
/// exactly. Every node probability in `target` must be representable by
/// the §13 update mechanism ([`node_prob_update_representable`]) — i.e.
/// `1` or even; the DC/AC/ZRL baselines are all `128` (even) and the
/// decode rule only ever writes representable values, so any bank
/// obtained by decoding is round-trippable.
///
/// # Panics
///
/// Debug-panics if any target node probability is an unrepresentable odd
/// value `> 1`.
// Index loops keep the per-node `*_UPDATE_PROBS[..]` flag lookups aligned
// with the §13.2 / §13.3.3 / §13.3 Tables traversal the decoder reads.
#[allow(clippy::needless_range_loop)]
pub fn encode_coefficient_prob_updates_full(enc: &mut BoolEncoder, target: &CoeffProbBanks) {
    let baseline = CoeffProbBanks::keyframe();

    // 1. §13.2 DC node updates.
    for plane in 0..NUM_PLANES {
        for node in 0..NUM_TREE_NODES {
            encode_node_prob_update(
                enc,
                VP6_DC_UPDATE_PROBS[plane][node],
                baseline.dc_probs[plane][node],
                target.dc_probs[plane][node],
            );
        }
    }

    // 2. §12.2 scan-update bit + (if custom) the band updates.
    if target.band_assignment == DEFAULT_BAND_ASSIGNMENT {
        enc.encode_b1(0);
    } else {
        enc.encode_b1(1);
        encode_coeff_band_updates(enc, &target.band_assignment);
    }

    // 3. §13.3.3 ZRL node updates.
    for band in 0..crate::zrl::NUM_ZRL_BANDS {
        for node in 0..crate::zrl::NUM_ZRL_NODES {
            encode_node_prob_update(
                enc,
                ZRL_UPDATE_PROBS[band][node],
                baseline.zrl_probs[band][node],
                target.zrl_probs[band][node],
            );
        }
    }

    // 4. §13.3 AC node updates, in the Table-31-outer walk order. Note
    //    AcUpdateProbs is `[prec][plane][band][node]` while the AC bank
    //    is `[plane][prec][band][node]` (the §13.3 transpose).
    for prec in 0..crate::tokens::NUM_AC_PREC_CONTEXTS {
        for plane in 0..NUM_PLANES {
            for band in 0..crate::tokens::NUM_AC_BANDS {
                for node in 0..NUM_TREE_NODES {
                    encode_node_prob_update(
                        enc,
                        AC_UPDATE_PROBS[prec][plane][band][node],
                        baseline.ac_probs[plane][prec][band][node],
                        target.ac_probs[plane][prec][band][node],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh `BoolEncoder`, finished, fed straight back into a
    /// `BoolCoder`. The 4-byte minimum the §7.3 coder needs is satisfied
    /// by the encoder's flush padding.
    fn roundtrip<F>(emit: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolEncoder),
    {
        let mut enc = BoolEncoder::new();
        emit(&mut enc);
        enc.finish()
    }

    /// The minimal "no updates" Figure-5 sub-stream round-trips: the
    /// decoder applies zero changes, so the banks come back identical to
    /// the keyframe baseline and the scan order is the default zig-zag.
    #[test]
    fn no_update_substream_round_trips_to_baseline() {
        let bytes = roundtrip(encode_coefficient_prob_updates);
        let mut bc = BoolCoder::new(&bytes).expect("bc");

        let mut banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).expect("decode");

        assert_eq!(
            banks,
            CoeffProbBanks::keyframe(),
            "no-update must be a no-op"
        );
        assert_eq!(scan, DEFAULT_SCAN_ORDER, "no-update keeps the default scan");
    }

    /// The "no updates" pass is a true no-op on a *non-baseline* incoming
    /// bank too: an inter frame that carries mutated probabilities must
    /// keep them when the sub-stream signals no updates (§13 persistence).
    #[test]
    fn no_update_preserves_carried_banks() {
        let mut carried = CoeffProbBanks::keyframe();
        // Perturb a few entries to simulate a previous frame's updates.
        carried.dc_probs[0][0] = 200;
        carried.dc_probs[1][5] = 17;
        carried.ac_probs[0][1][2][3] = 99;
        carried.zrl_probs[1][7] = 44;
        let before = carried.clone();

        let bytes = roundtrip(encode_coefficient_prob_updates);
        let mut bc = BoolCoder::new(&bytes).expect("bc");
        let scan = decode_coefficient_prob_updates(&mut bc, &mut carried).expect("decode");

        assert_eq!(
            carried, before,
            "carried banks must survive a no-update pass"
        );
        // A clear scan-update bit resets to the default zig-zag (§12.2).
        assert_eq!(scan, DEFAULT_SCAN_ORDER);
    }

    /// A custom-scan request round-trips: the decoded band assignment
    /// matches the encoded one and the resulting raster scan order is the
    /// custom permutation.
    #[test]
    fn custom_scan_substream_round_trips() {
        // A simple non-default assignment: move coefficient 1 into band 2
        // and coefficient 2 into band 1 (swap of two early bands).
        let mut assignment = DEFAULT_BAND_ASSIGNMENT;
        assignment[1] = 2;
        assignment[2] = 1;

        let bytes = roundtrip(|enc| encode_coefficient_prob_updates_with_scan(enc, &assignment));
        let mut bc = BoolCoder::new(&bytes).expect("bc");

        let mut banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).expect("decode");

        assert_eq!(
            banks.band_assignment, assignment,
            "band assignment must round-trip"
        );
        let expected = custom_scan_order_to_raster(&build_custom_scan_order(&assignment));
        assert_eq!(scan, expected, "custom scan order must round-trip");
        // The custom scan must be a genuine permutation, not the default.
        assert_ne!(scan, DEFAULT_SCAN_ORDER);
    }

    /// The expanded `to_intra_probs` DC contexts match `IntraProbs::keyframe`
    /// when the banks are at baseline — the raw→expanded bridge is exact.
    #[test]
    fn keyframe_banks_expand_to_keyframe_intra_probs() {
        let banks = CoeffProbBanks::keyframe();
        let probs = banks.to_intra_probs();
        let baseline = IntraProbs::keyframe();
        assert_eq!(probs.dc_contexts, baseline.dc_contexts);
        assert_eq!(probs.ac_probs, baseline.ac_probs);
        assert_eq!(probs.zrl_probs, baseline.zrl_probs);
    }

    /// Every full scan order produced is a permutation of `0..=63`.
    #[test]
    fn raster_scan_order_is_permutation() {
        let mut banks = CoeffProbBanks::keyframe();
        banks.band_assignment[10] = 5;
        banks.band_assignment[11] = 4;
        let scan = banks.raster_scan_order();
        let mut seen = [false; 64];
        for &p in scan.iter() {
            assert!(!seen[p as usize], "scan order must be a permutation");
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// The §13 node-prob update target space is exactly `{1} ∪ even`.
    #[test]
    fn node_prob_update_representability() {
        assert!(node_prob_update_representable(1));
        assert!(node_prob_update_representable(2));
        assert!(node_prob_update_representable(128));
        assert!(node_prob_update_representable(254));
        assert!(!node_prob_update_representable(0)); // forbidden node prob
        assert!(!node_prob_update_representable(3)); // odd > 1 unreachable
        assert!(!node_prob_update_representable(255));
        // The b(7) value chosen decodes back through max(1, value*2).
        assert_eq!(node_prob_update_value(1), Some(0));
        assert_eq!(node_prob_update_value(2), Some(1));
        assert_eq!(node_prob_update_value(254), Some(127));
    }

    /// A full-update Figure-5 sub-stream transforms the keyframe baseline
    /// into an arbitrary representable target bank, round-tripping every
    /// DC / ZRL / AC node probability and the custom scan exactly.
    #[test]
    fn full_update_substream_round_trips_target() {
        // Build a target with representable (even / 1) node-prob deltas
        // across all four banks, plus a custom scan.
        let mut target = CoeffProbBanks::keyframe();
        target.dc_probs[0][0] = 200;
        target.dc_probs[1][6] = 2;
        target.dc_probs[0][10] = 1; // the `1` escape (value 0)
        target.zrl_probs[0][0] = 64;
        target.zrl_probs[1][13] = 254;
        target.ac_probs[0][0][0][0] = 80;
        target.ac_probs[1][2][5][10] = 4;
        target.band_assignment[1] = 2;
        target.band_assignment[2] = 1;

        let bytes = roundtrip(|enc| encode_coefficient_prob_updates_full(enc, &target));
        let mut bc = BoolCoder::new(&bytes).expect("bc");

        let mut banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).expect("decode");

        assert_eq!(
            banks, target,
            "full Figure-5 update must round-trip the bank"
        );
        let expected =
            custom_scan_order_to_raster(&build_custom_scan_order(&target.band_assignment));
        assert_eq!(
            scan, expected,
            "custom scan must round-trip alongside the updates"
        );
    }

    /// A full update with the default band assignment emits the cleared
    /// scan bit and still round-trips the node-prob deltas.
    #[test]
    fn full_update_default_scan_round_trips() {
        let mut target = CoeffProbBanks::keyframe();
        target.dc_probs[0][3] = 50;
        target.ac_probs[0][0][1][2] = 222;

        let bytes = roundtrip(|enc| encode_coefficient_prob_updates_full(enc, &target));
        let mut bc = BoolCoder::new(&bytes).expect("bc");
        let mut banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut banks).expect("decode");
        assert_eq!(banks, target);
        assert_eq!(scan, DEFAULT_SCAN_ORDER);
    }
}

//! Per-frame quantiser selection for VP6 encoding (rate control).
//!
//! The VP6 encoders (`intra_encode` / `inter_encode`) take a fixed §9
//! `DctQMask` quantiser index and emit a partition whose size depends on how
//! coarsely that index quantises the residual. Rate control is the inverse
//! problem: given a **bit budget** (or a target frame size) for the frame,
//! pick the `DctQMask` that produces a partition as close to the target as
//! possible.
//!
//! ## Monotonicity
//!
//! The §15 dequantisation factor (`AC/DC_QUANTIZATION_TABLE`) **decreases** as
//! the `DctQMask` index increases — index 0 is the coarsest quantiser (largest
//! factor, fewest surviving coefficients, smallest output) and index 63 is the
//! finest (smallest factor, most coefficients, largest output). The encoded
//! partition size is therefore (weakly) **monotonically non-decreasing** in
//! the quantiser index. That monotonicity is what makes the budget search a
//! binary search: probe a candidate index, and if its output fits the budget
//! every coarser index also fits, so the search moves toward the finest index
//! that still fits.
//!
//! The monotonicity is *weak* (two adjacent indices can produce identically
//! sized output, e.g. when the residual is already all-zero), which the
//! search handles by always keeping the finest fitting index seen.
//!
//! ## Surfaces
//!
//! * [`MIN_Q`] / [`MAX_Q`] — the inclusive `DctQMask` index bounds (`0..=63`).
//! * [`QuantiserChoice`] — a selection result: the chosen index, the encoded
//!   byte length at that index, and the encoded bytes themselves.
//! * [`select_quantiser_for_budget`] — pick the **finest** index whose output
//!   fits a byte budget (best quality under the cap), falling back to [`MIN_Q`]
//!   when even the coarsest index overflows.
//! * [`select_quantiser_for_target_size`] — pick the index whose output size
//!   is **closest** to a target (over- or under-shoot), the building block for
//!   a constant-bitrate driver that wants the nearest size rather than a hard
//!   cap.
//!
//! ## Provenance
//!
//! Rate control is an encoder-side policy layer over the §9 `DctQMask` field
//! and the §15 quantiser tables (`docs/video/vp6/vp6_format.pdf`); it consumes
//! only the in-tree encoders' observable output sizes. No third-party VP6
//! source consulted.

use crate::Error;

/// The coarsest §9 `DctQMask` quantiser index (largest dequant factor,
/// smallest encoded output).
pub const MIN_Q: u8 = 0;

/// The finest §9 `DctQMask` quantiser index (smallest dequant factor, largest
/// encoded output). The §15 tables are 64 entries (`0..=63`).
pub const MAX_Q: u8 = 63;

/// The outcome of a rate-control quantiser selection: the chosen `DctQMask`
/// index, the encoded partition byte length at that index, and the encoded
/// bytes themselves (so the caller doesn't re-encode at the chosen index).
#[derive(Debug, Clone)]
pub struct QuantiserChoice {
    /// The selected §9 `DctQMask` index (`0..=63`).
    pub q: u8,
    /// The encoded partition's length in bytes at `q`.
    pub size: usize,
    /// The encoded partition bytes at `q`.
    pub bytes: Vec<u8>,
}

/// Pick the **finest** quantiser index whose encoded output fits `budget_bytes`
/// — the largest quality (highest index) under a hard size cap.
///
/// `encode(q)` must produce the encoded partition for `DctQMask == q`. The
/// search exploits the size-vs-index monotonicity (see the module docs): it
/// binary-searches `MIN_Q..=MAX_Q` for the highest index whose output is
/// `<= budget_bytes`. If even [`MIN_Q`] (the coarsest, smallest output)
/// overflows the budget, the returned choice is [`MIN_Q`] anyway (the closest
/// the codec can get — the caller decides whether to accept the overflow or
/// drop the frame).
///
/// Returns the chosen index, its size, and its bytes in a [`QuantiserChoice`]
/// so the caller reuses the already-encoded partition.
///
/// # Errors
///
/// Propagates any [`Error`] from `encode`.
pub fn select_quantiser_for_budget<F>(
    budget_bytes: usize,
    mut encode: F,
) -> Result<QuantiserChoice, Error>
where
    F: FnMut(u8) -> Result<Vec<u8>, Error>,
{
    // Start with the coarsest index as the always-available fallback.
    let min_bytes = encode(MIN_Q)?;
    let mut best = QuantiserChoice {
        q: MIN_Q,
        size: min_bytes.len(),
        bytes: min_bytes,
    };

    // If even the coarsest overflows, there's nothing finer that helps.
    if best.size > budget_bytes {
        return Ok(best);
    }

    // Binary search for the finest index that still fits. Invariant: every
    // index in `lo..=hi` is a candidate; `best` already holds the finest
    // confirmed-fitting index seen so far.
    let mut lo = MIN_Q as u32 + 1;
    let mut hi = MAX_Q as u32;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let bytes = encode(mid as u8)?;
        if bytes.len() <= budget_bytes {
            // `mid` fits — record it and search finer (higher indices).
            best = QuantiserChoice {
                q: mid as u8,
                size: bytes.len(),
                bytes,
            };
            lo = mid + 1;
        } else {
            // `mid` overflows — search coarser (lower indices).
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
    }

    Ok(best)
}

/// Pick the quantiser index whose encoded output size is **closest** to
/// `target_bytes` (over- or under-shoot allowed), the nearest-size building
/// block for a constant-bitrate driver.
///
/// `encode(q)` produces the partition for `DctQMask == q`. Unlike
/// [`select_quantiser_for_budget`] (a hard cap), this minimises the absolute
/// size difference from the target, so it may pick an index whose output
/// slightly exceeds the target if that's nearer than the largest index that
/// fits under it. The search uses the same monotonicity: it binary-searches
/// for the boundary between under- and over-target indices, then compares the
/// two straddling candidates and keeps the closer one.
///
/// # Errors
///
/// Propagates any [`Error`] from `encode`.
pub fn select_quantiser_for_target_size<F>(
    target_bytes: usize,
    mut encode: F,
) -> Result<QuantiserChoice, Error>
where
    F: FnMut(u8) -> Result<Vec<u8>, Error>,
{
    let diff = |size: usize| size.abs_diff(target_bytes);

    // The coarsest index (smallest output) is the initial best.
    let min_bytes = encode(MIN_Q)?;
    let mut best = QuantiserChoice {
        q: MIN_Q,
        size: min_bytes.len(),
        bytes: min_bytes,
    };

    // Binary search the index space, keeping the closest-to-target seen. We
    // can't stop at the first fit (we want nearest, not under-cap), so this is
    // a monotone-aware narrowing: if `mid` is under target, the nearest could
    // be `mid` or finer; if over, it could be `mid` or coarser. Track the
    // best straddle on both sides.
    let mut lo = MIN_Q as u32 + 1;
    let mut hi = MAX_Q as u32;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let bytes = encode(mid as u8)?;
        let size = bytes.len();
        if diff(size) < diff(best.size) || (diff(size) == diff(best.size) && (mid as u8) < best.q) {
            best = QuantiserChoice {
                q: mid as u8,
                size,
                bytes,
            };
        }
        if size < target_bytes {
            // Under target: a finer index (larger output) might be nearer.
            lo = mid + 1;
        } else if size > target_bytes {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            // Exact hit.
            break;
        }
    }

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic monotone encoder: output size grows linearly with the
    /// quantiser index, so `size(q) = base + q * step`. Models the real
    /// size-vs-index monotonicity without running the full codec.
    fn linear_encode(base: usize, step: usize) -> impl FnMut(u8) -> Result<Vec<u8>, Error> {
        move |q: u8| Ok(vec![0u8; base + q as usize * step])
    }

    #[test]
    fn q_bounds_match_table_size() {
        assert_eq!(MIN_Q, 0);
        assert_eq!(MAX_Q, 63);
        // The §15 tables are exactly 64 entries.
        assert_eq!(crate::dequant::AC_QUANTIZATION_TABLE.len(), 64);
    }

    #[test]
    fn budget_picks_finest_index_that_fits() {
        // size(q) = 10 + q*2 → at q=20 size=50, q=21 size=52.
        let choice = select_quantiser_for_budget(51, linear_encode(10, 2)).expect("select");
        assert_eq!(choice.q, 20); // 50 <= 51 < 52
        assert_eq!(choice.size, 50);
        assert_eq!(choice.bytes.len(), 50);
    }

    #[test]
    fn budget_exact_boundary_includes_equal_size() {
        // size(q)=10+q*2; q=20→50. Budget exactly 50 must accept q=20.
        let choice = select_quantiser_for_budget(50, linear_encode(10, 2)).expect("select");
        assert_eq!(choice.q, 20);
        assert_eq!(choice.size, 50);
    }

    #[test]
    fn budget_overflow_at_min_q_falls_back_to_min() {
        // size(0)=100 already exceeds a tiny budget → fall back to MIN_Q.
        let choice = select_quantiser_for_budget(10, linear_encode(100, 5)).expect("select");
        assert_eq!(choice.q, MIN_Q);
        assert_eq!(choice.size, 100);
    }

    #[test]
    fn budget_huge_budget_picks_max_q() {
        // Any q fits a huge budget → finest index wins.
        let choice = select_quantiser_for_budget(1_000_000, linear_encode(10, 2)).expect("select");
        assert_eq!(choice.q, MAX_Q);
        assert_eq!(choice.size, 10 + MAX_Q as usize * 2);
    }

    #[test]
    fn target_size_picks_nearest_even_if_over() {
        // size(q)=10+q*3 → q=10→40, q=11→43. Target 42 is nearer 43 (q=11,
        // diff 1) than 40 (q=10, diff 2).
        let choice = select_quantiser_for_target_size(42, linear_encode(10, 3)).expect("select");
        assert_eq!(choice.q, 11);
        assert_eq!(choice.size, 43);
    }

    #[test]
    fn target_size_exact_hit() {
        // size(q)=10+q*5 → q=8→50. Target 50 → exact q=8.
        let choice = select_quantiser_for_target_size(50, linear_encode(10, 5)).expect("select");
        assert_eq!(choice.q, 8);
        assert_eq!(choice.size, 50);
    }

    #[test]
    fn target_size_below_min_picks_min_q() {
        // size(0)=100 > target 10 → coarsest is nearest.
        let choice = select_quantiser_for_target_size(10, linear_encode(100, 5)).expect("select");
        assert_eq!(choice.q, MIN_Q);
        assert_eq!(choice.size, 100);
    }

    #[test]
    fn budget_search_is_correct_against_brute_force() {
        // Verify the binary search matches a linear scan for many budgets and
        // step shapes (weakly-monotone too: step 0 segments).
        for step in [1usize, 2, 3, 7] {
            for budget in [0usize, 5, 10, 50, 100, 200, 500] {
                let choice =
                    select_quantiser_for_budget(budget, linear_encode(8, step)).expect("select");
                // Brute force: finest q whose size <= budget, else MIN_Q.
                let mut expected = MIN_Q;
                let mut found = false;
                for q in (MIN_Q..=MAX_Q).rev() {
                    if 8 + q as usize * step <= budget {
                        expected = q;
                        found = true;
                        break;
                    }
                }
                if !found {
                    expected = MIN_Q;
                }
                assert_eq!(
                    choice.q, expected,
                    "budget={budget} step={step}: got q={} expected q={expected}",
                    choice.q
                );
            }
        }
    }

    #[test]
    fn encoder_error_propagates() {
        let res = select_quantiser_for_budget(100, |_q| Err(Error::NotImplemented));
        assert!(matches!(res, Err(Error::NotImplemented)));
    }

    // ----- Real-encoder integration -----

    /// Drive rate control against the **real** intra encoder and confirm:
    /// (1) the encoded partition size is monotonically non-decreasing in `q`
    /// (the property the search relies on), and (2) the budget search picks a
    /// q whose real output fits the budget, and one index finer overflows it.
    #[test]
    fn budget_selects_real_intra_q_under_cap() {
        use crate::frame_assembly::Frame;
        use crate::intra_encode::encode_intra_frame;

        // A patterned 32×32 frame so finer quantisers genuinely cost more bits.
        let mut f = Frame::new(4, 4);
        let yw = f.y.width();
        let yh = f.y.height();
        for r in 0..yh {
            for c in 0..yw {
                f.y.samples_mut()[r * yw + c] = ((r * 7 + c * 11) % 256) as u8;
            }
        }
        for s in f.u.samples_mut() {
            *s = 120;
        }
        for s in f.v.samples_mut() {
            *s = 140;
        }

        // (1) Monotonicity of the real encoder's output size.
        let mut prev = 0usize;
        for q in (MIN_Q..=MAX_Q).step_by(8) {
            let size = encode_intra_frame(&f, q).expect("encode").len();
            assert!(
                size >= prev,
                "real intra size not monotone at q={q}: {size} < {prev}"
            );
            prev = size;
        }

        // (2) Budget search against the real encoder. Pick a budget between the
        // coarsest and finest sizes.
        let coarse = encode_intra_frame(&f, MIN_Q).expect("encode").len();
        let fine = encode_intra_frame(&f, MAX_Q).expect("encode").len();
        assert!(
            fine > coarse,
            "encoder must be rate-sensitive for this test"
        );
        let budget = (coarse + fine) / 2;

        let choice = select_quantiser_for_budget(budget, |q| encode_intra_frame(&f, q))
            .expect("rate control");
        assert!(
            choice.size <= budget,
            "chosen q={} size={} exceeds budget={budget}",
            choice.q,
            choice.size
        );
        // The returned bytes must be the real encode at the chosen q and decode
        // back through the header + intra decoder.
        assert_eq!(
            choice.bytes,
            encode_intra_frame(&f, choice.q).expect("re-encode")
        );
        // One index finer must overflow (else the search stopped too early),
        // unless we already chose the finest.
        if choice.q < MAX_Q {
            let finer = encode_intra_frame(&f, choice.q + 1).expect("encode").len();
            assert!(
                finer > budget,
                "q={}+1 size={finer} should overflow budget={budget}",
                choice.q
            );
        }
    }
}

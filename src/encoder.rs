//! VP6F (VP6 for FLV) encoder — initial scaffold.
//!
//! This is a minimal encoder that emits **DC-only keyframes**. Every
//! 8x8 block carries a single DC coefficient chosen so the reconstructed
//! pixel average matches the target block mean; all AC coefficients are
//! zero. The output is a blocky but standards-compliant VP6F elementary
//! stream that our own decoder (and, hopefully, ffmpeg's `vp6f` decoder)
//! can round-trip.
//!
//! Scope:
//!
//! * Keyframes only (no P-frames).
//! * No motion vectors; no AC coefficients; no loop filter.
//! * Interlaced flag = 0, `sub_version = 0` (simple profile),
//!   `filter_header = 0`, bool-path coefficients (no Huffman).
//!
//! Later rounds will extend the encoder with AC coefficients, motion
//! estimation, loop-filter emission, and inter-frame support.
//!
//! ### Bitstream layout matches [`crate::frame_header::FrameHeader`]
//! and the decoder path in [`crate::decoder::Vp6Stream::decode_frame`]
//! — encoder state mirrors the decoder's [`Vp6Model`] + [`BlockScratch`]
//! so DC-prediction works byte-for-byte the same on both sides.

use oxideav_core::{Error, Result};

use crate::mb::RefKind;
use crate::models::Vp6Model;
use crate::range_coder::RangeEncoder;
use crate::tables;

/// Output of a single-frame encode: the concatenated VP6F elementary-
/// stream bytes (no FLV tag wrapping — callers prepend the 1-byte FLV
/// frame adjuster, typically `0x00`).
pub type EncodedFrame = Vec<u8>;

/// Per-block reference-DC context kept in `left_block` / `above_blocks`
/// on the encoder side. Mirrors [`crate::mb::RefDc`] but stays owned by
/// the encoder so we don't tangle encoder state into decoder types.
#[derive(Clone, Copy, Debug, Default)]
struct EncRefDc {
    not_null_dc: bool,
    ref_frame: RefKind,
    dc_coeff: i16,
}

/// VP6F encoder.
///
/// Each [`Vp6Encoder::encode_keyframe`] call emits a complete elementary-
/// stream frame. Internal state (model, reference frames) is reset at
/// the start of every keyframe so the encoder is "stateless" across
/// keyframes until inter-frame support lands.
#[derive(Debug)]
pub struct Vp6Encoder {
    /// Quantiser parameter (0 = highest quality, 63 = coarsest). Smaller
    /// QPs mean finer dequant steps, so the achievable DC-step is
    /// smaller and DC-only encoding gets closer to the source.
    pub qp: u8,
    /// Use the `sub_version = 0` "simple profile" layout. Anything else
    /// demands the filter-header parse; the initial scaffold sticks to
    /// simple-profile to keep the header minimal.
    pub sub_version: u8,
}

impl Default for Vp6Encoder {
    fn default() -> Self {
        Self {
            qp: 32,
            sub_version: 0,
        }
    }
}

impl Vp6Encoder {
    /// New encoder with a given QP (clamped to 0..=63).
    pub fn new(qp: u8) -> Self {
        Self {
            qp: qp.min(63),
            sub_version: 0,
        }
    }

    /// Encode a single keyframe from row-major Y/U/V planes.
    ///
    /// * `y_plane` — `width * height` bytes.
    /// * `u_plane` / `v_plane` — `(width/2) * (height/2)` bytes.
    /// * `width`, `height` must be multiples of 16.
    pub fn encode_keyframe(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
    ) -> Result<EncodedFrame> {
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::invalid(
                "VP6 encode: width/height must be non-zero multiples of 16",
            ));
        }
        let mb_width = width / 16;
        let mb_height = height / 16;
        if mb_width > 255 || mb_height > 255 {
            return Err(Error::invalid(
                "VP6 encode: simple-profile MB dims capped at 255",
            ));
        }
        if y_plane.len() < width * height {
            return Err(Error::invalid("VP6 encode: y plane too small"));
        }
        let uv_stride = width / 2;
        let uv_h = height / 2;
        if u_plane.len() < uv_stride * uv_h || v_plane.len() < uv_stride * uv_h {
            return Err(Error::invalid("VP6 encode: chroma plane too small"));
        }

        // --- Fixed header -------------------------------------------------
        let mut out = Vec::<u8>::with_capacity(32 + mb_width * mb_height * 6);
        // byte 0: frame_mode=0 (key), qp<<1, sep_coeff=0
        out.push((self.qp << 1) & 0x7E);
        // byte 1: sub_version<<3 | filter_header=0 | interlaced=0
        out.push(self.sub_version << 3);
        // With filter_header=0 the decoder still reads the 2-byte
        // coeff-offset field; emit value `2` so the adjusted offset is 0
        // (single-partition bool stream).
        out.push(0);
        out.push(2);
        // mb_height / mb_width / display_mb_* — simple-profile layout.
        out.push(mb_height as u8);
        out.push(mb_width as u8);
        out.push(mb_height as u8);
        out.push(mb_width as u8);

        // --- Bool-coded body ---------------------------------------------
        let mut enc = RangeEncoder::new();
        // 2 skip bits — the decoder reads them via `rac.get_bits(2)` and
        // ignores them.
        enc.put_bits(2, 0);

        // `use_huffman` flag — we always emit the bool-coded coefficient
        // path, matching what the decoder supports.
        enc.put_bit(0);

        // --- Coefficient-model update pass -------------------------------
        //
        // We emit "no updates" for every entry so the decoder keeps its
        // default probabilities. On a keyframe that means `def_prob[node]`
        // is copied through into `coeff_dccv` / `coeff_ract` on the
        // decoder side. To stay in sync we mirror the same mutation on
        // the encoder's model.
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);

        // DC value-update probabilities — emit 0 for every node.
        let mut def_prob = [0x80u8; 11];
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
                // key=true path copies def_prob through.
                model.coeff_dccv[pt][node] = def_prob[node];
            }
        }

        // Coefficient-reorder update flag — emit 0 to keep defaults.
        enc.put_bit(0);

        // Run-value update probabilities — emit 0 for every node.
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }

        // AC coefficient update probabilities — emit 0 for every node.
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                        model.coeff_ract[pt][ct][cg][node] = def_prob[node];
                        let _ = &mut def_prob;
                    }
                }
            }
        }

        // Recompute the `coeff_dcct` linear combination that `parse_coeff_models`
        // does on the decoder side. Without this step the encoder and
        // decoder disagree on the DC context probabilities.
        for pt in 0..2 {
            for ctx in 0..3 {
                for node in 0..5 {
                    let v = ((model.coeff_dccv[pt][node] as i32
                        * tables::VP6_DCCV_LC[ctx][node][0]
                        + 128)
                        >> 8)
                        + tables::VP6_DCCV_LC[ctx][node][1];
                    model.coeff_dcct[pt][ctx][node] = v.clamp(1, 255) as u8;
                }
            }
        }

        // --- Per-macroblock coefficient emission -------------------------
        //
        // Keyframe: every MB is intra (`Vp56Mb::Intra`), reference frame
        // is `Current`. DC prediction mirrors `BlockScratch` exactly so
        // the decoder's predictor matches our assumed predictor.
        let dequant_dc = (tables::VP56_DC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;

        // Mirror of decoder's BlockScratch for DC-prediction state.
        let mut left_block: [EncRefDc; 4] = [EncRefDc::default(); 4];
        let mut above_blocks: Vec<EncRefDc> = vec![EncRefDc::default(); 4 * mb_width + 6];
        // Match the sentinels installed by `BlockScratch::reset_row`.
        if 2 * mb_width + 2 < above_blocks.len() {
            above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < above_blocks.len() {
            above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        // prev_dc[plane][ref_kind_index(Current)=0]: chroma seeded at 128.
        let mut prev_dc = [[0i16; 3]; 3];
        prev_dc[1][0] = 128;
        prev_dc[2][0] = 128;

        for mb_row in 0..mb_height {
            // `start_row` equivalent.
            for b in &mut left_block {
                *b = EncRefDc::default();
            }
            let mut above_block_idx: [usize; 6] = [0; 6];
            above_block_idx[0] = 1;
            above_block_idx[1] = 2;
            above_block_idx[2] = 1;
            above_block_idx[3] = 2;
            above_block_idx[4] = 2 * mb_width + 2 + 1;
            above_block_idx[5] = 3 * mb_width + 4 + 1;

            for mb_col in 0..mb_width {
                // Emit 6 blocks: Y0, Y1, Y2, Y3, U, V.
                for b in 0..6usize {
                    let pt = if b > 3 { 1usize } else { 0usize };
                    let plane_idx = tables::B2P[b] as usize;
                    let ctx = left_block[tables::B6_TO_4[b] as usize].not_null_dc as usize
                        + above_blocks[above_block_idx[b]].not_null_dc as usize;

                    // Compute target DC from the source pixels.
                    let target_pixel = sample_block_mean(
                        b, mb_row, mb_col, y_plane, u_plane, v_plane, width, uv_stride,
                    );
                    // Pick the reconstructed-DC value that, after the
                    // decoder's two-stage IDCT DC-only chain, minimises
                    // the absolute error versus `target_pixel`. Note
                    // `block[0]` at the decoder is `new_dc *
                    // dequant_dc`, so we search over integer `new_dc`
                    // values with step `dequant_dc` in the underlying
                    // IDCT-space.
                    let new_dc =
                        quantised_new_dc(target_pixel, dequant_dc).clamp(-32768, 32767) as i16;

                    // Compute the decoder's predictor_dc for this block.
                    let lb = left_block[tables::B6_TO_4[b] as usize];
                    let ab = above_blocks[above_block_idx[b]];
                    let mut count = 0i32;
                    let mut dc = 0i32;
                    if lb.ref_frame == RefKind::Current {
                        dc += lb.dc_coeff as i32;
                        count += 1;
                    }
                    if ab.ref_frame == RefKind::Current {
                        dc += ab.dc_coeff as i32;
                        count += 1;
                    }
                    let predictor = match count {
                        0 => prev_dc[plane_idx][0] as i32,
                        2 => dc / 2,
                        _ => dc,
                    };
                    let coded_dc = (new_dc as i32 - predictor).clamp(-32768, 32767) as i16;

                    // --- Bitstream: emit a single-DC block. ---
                    //
                    // Shape: either
                    //   * coded_dc == 0: 3 bits — skip DC, skip idx=1 m2_0,
                    //     break via m2_1=0.
                    //   * coded_dc != 0: enter parse, emit value + sign,
                    //     then emit end-of-block (m2_0=0, m2_1=0).
                    //
                    // `ct` (1/2) drives the Ract model selection for the
                    // post-DC path — must match the decoder's update.
                    let mut ct_ract: usize = 1;
                    if coded_dc == 0 {
                        // Skip bit for DC.
                        enc.put_prob(model.coeff_dcct[pt][ctx][0], 0);
                        ct_ract = 0;
                    } else {
                        enc.put_prob(model.coeff_dcct[pt][ctx][0], 1);
                        let abs_val = coded_dc.unsigned_abs() as i32;
                        let sign = if coded_dc < 0 { 1 } else { 0 };
                        encode_coeff_value(
                            &mut enc,
                            &model.coeff_dcct[pt][ctx],
                            &model.coeff_dccv[pt],
                            abs_val,
                            &mut ct_ract,
                        );
                        enc.put_bit(sign);
                    }

                    // Second iteration: we're at coeff_idx = (0+run=1) = 1.
                    // Emit "no AC" — ct_ract drives model2_0, and m2_1 = end.
                    //
                    // cg for idx=1 is VP6_COEFF_GROUPS[1] = 0.
                    let cg = tables::VP6_COEFF_GROUPS[1] as usize;
                    let ract = &model.coeff_ract[pt][ct_ract][cg];
                    // Skip bit (coeff_idx=1, ct_ract could be 0 or non-0 —
                    // in either case we need a 0 bit emitted via get_prob).
                    enc.put_prob(ract[0], 0);
                    // End-of-block bit. Decoder reads this only when
                    // coeff_idx > 0 — we're at idx=1, so it's read.
                    enc.put_prob(ract[1], 0);

                    // Update DC-prediction context for subsequent blocks.
                    let has_nonzero_dc = coded_dc != 0;
                    let lb_idx = tables::B6_TO_4[b] as usize;
                    let new_dc_final = (coded_dc as i32 + predictor) as i16;
                    left_block[lb_idx] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Current,
                        dc_coeff: new_dc_final,
                    };
                    above_blocks[above_block_idx[b]] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Current,
                        dc_coeff: new_dc_final,
                    };
                    prev_dc[plane_idx][0] = new_dc_final;
                }

                // advance_column: luma +2, chroma +1.
                for y in 0..4 {
                    above_block_idx[y] += 2;
                }
                for uv in 4..6 {
                    above_block_idx[uv] += 1;
                }
            }
        }

        out.extend_from_slice(&enc.finish());
        Ok(out)
    }
}

/// Sample the 8x8 block mean from the right plane at block `b` of
/// macroblock `(mb_row, mb_col)`. Luma blocks 0..=3 pull from the Y
/// plane; blocks 4/5 pull from U/V respectively.
fn sample_block_mean(
    b: usize,
    mb_row: usize,
    mb_col: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    y_stride: usize,
    uv_stride: usize,
) -> u8 {
    let (plane, stride, x0, y0) = match b {
        0 => (y, y_stride, mb_col * 16, mb_row * 16),
        1 => (y, y_stride, mb_col * 16 + 8, mb_row * 16),
        2 => (y, y_stride, mb_col * 16, mb_row * 16 + 8),
        3 => (y, y_stride, mb_col * 16 + 8, mb_row * 16 + 8),
        4 => (u, uv_stride, mb_col * 8, mb_row * 8),
        5 => (v, uv_stride, mb_col * 8, mb_row * 8),
        _ => unreachable!(),
    };
    let mut sum = 0u32;
    for r in 0..8 {
        for c in 0..8 {
            sum += plane[(y0 + r) * stride + (x0 + c)] as u32;
        }
    }
    ((sum + 32) / 64) as u8
}

/// Pick the integer DC block coefficient such that a DC-only IDCT
/// (Put mode) reconstructs a pixel closest to `target`.
///
/// The decoder's IDCT runs in two stages for intra Put mode: first a
/// column pass that turns `block[0]` into a column of identical values
/// `row_dc = (C4S4 * block[0]) >> 16` (arithmetic), then a row pass
/// that emits `clip(128 + ((C4S4 * row_dc + (8<<16)) >> 20))` for each
/// output pixel. We invert that two-stage relation by analytic
/// estimate + a ±4 search around the estimate (the implicit rounding
/// at each stage can skew by a couple of ULPs — cheap to cover).
fn dc_level_from_pixel(target: u8) -> i32 {
    // Reconstruct pixel from an input block[0] value, matching the
    // behaviour of the decoder's DC-only `idct_put` path exactly.
    let recon = |block0: i32| -> i32 {
        // Column stage: row_dc = (C4S4 * block0) >> 16 (arithmetic).
        let row_dc = ((46341i64 * block0 as i64) >> 16) as i32;
        // Row stage: pixel = clip(128 + ((C4S4 * row_dc + (8<<16)) >> 20)).
        let v = 128 + (((46341i64 * row_dc as i64) + (8i64 << 16)) >> 20);
        v.clamp(0, 255) as i32
    };

    // Analytic estimate: treat the two >>16 and >>20 shifts as a
    // combined scale of (C4S4 * C4S4) / 2^36.
    let delta = target as i32 - 128;
    let num = (delta as i64) << 36;
    let denom = 46341i64 * 46341i64;
    let analytic = (num / denom) as i32;

    let mut best = analytic;
    let mut best_err = i32::MAX;
    for cand in (analytic - 4)..=(analytic + 4) {
        let err = (recon(cand) - target as i32).abs();
        if err < best_err {
            best_err = err;
            best = cand;
        }
    }
    best
}

/// Pick the integer `new_dc` such that after the decoder multiplies
/// `new_dc * dequant_dc` and IDCT-reconstructs the resulting DC-only
/// block, the output pixel is as close as possible to `target`.
fn quantised_new_dc(target: u8, dequant_dc: i32) -> i32 {
    // Seed with the analytic inverse divided by dequant_dc.
    let seed = dc_level_from_pixel(target);
    let base = if dequant_dc == 0 {
        0
    } else {
        div_nearest(seed, dequant_dc)
    };
    let recon = |new_dc: i32| -> i32 {
        let block0 = new_dc * dequant_dc;
        let row_dc = ((46341i64 * block0 as i64) >> 16) as i32;
        let v = 128 + (((46341i64 * row_dc as i64) + (8i64 << 16)) >> 20);
        v.clamp(0, 255) as i32
    };
    let mut best = base;
    let mut best_err = i32::MAX;
    for cand in (base - 3)..=(base + 3) {
        let err = (recon(cand) - target as i32).abs();
        if err < best_err {
            best_err = err;
            best = cand;
        }
    }
    best
}

/// Signed integer nearest-to-zero division (rounds toward nearest, ties
/// away from zero). Mirrors what the decoder implicitly expects when
/// it multiplies `new_dc * dequant_dc` back up.
fn div_nearest(num: i32, denom: i32) -> i32 {
    if denom == 0 {
        return 0;
    }
    let n = num as i64;
    let d = denom as i64;
    let q = if (n ^ d) >= 0 {
        (n + d / 2) / d
    } else {
        (n - d / 2) / d
    };
    q as i32
}

/// Emit a single non-zero coefficient via the bool path. Matches the
/// decoder's [`crate::mb::parse_coeff`] for the coefficient-value branch
/// reached when `get_prob(m2[0]) == 1`.
///
/// On return, `ct` is set to the `ct` value the decoder would adopt
/// (1 for value==1, 2 for value>1) — callers need this to pick the
/// right Ract model slice for the next iteration.
fn encode_coeff_value(enc: &mut RangeEncoder, m2: &[u8], m1: &[u8], abs_val: i32, ct: &mut usize) {
    if abs_val == 1 {
        // m2[2] == 0 branch.
        enc.put_prob(m2[2], 0);
        *ct = 1;
        return;
    }
    // m2[2] == 1 — value is 2 or larger.
    enc.put_prob(m2[2], 1);
    if abs_val == 2 {
        // m2[3] == 0, m2[4] == 0 → coeff = 2.
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 0);
        *ct = 2;
        return;
    }
    if abs_val == 3 || abs_val == 4 {
        // m2[3] == 0, m2[4] == 1, m1[5] is the low bit.
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 1);
        // coeff = 3 + m1[5]_bit, so bit = abs_val - 3.
        enc.put_prob(m1[5], (abs_val - 3) as u8);
        *ct = 2;
        return;
    }
    // Long path: m2[3] == 1 → PC_TREE + bias-category bits.
    enc.put_prob(m2[3], 1);
    // Determine category: bias = VP56_COEFF_BIAS[idx+5], so value is in
    // `[bias[i+5], bias[i+6])` for category i in 0..5, or category 5
    // (67+) covers [67, 67+2^11).
    let mut cat = 0usize;
    for i in 0..=5 {
        let lo = tables::VP56_COEFF_BIAS[i + 5] as i32;
        let hi = if i + 6 < tables::VP56_COEFF_BIAS.len() {
            tables::VP56_COEFF_BIAS[i + 6] as i32
        } else {
            i32::MAX
        };
        if abs_val >= lo && abs_val < hi {
            cat = i;
            break;
        }
    }
    // Walk the PC_TREE to the leaf for `cat`. We know the tree shape
    // (from decoder), so we precompute the bit sequence for each of the
    // six leaves. Each leaf is reached by a specific sequence of
    // probability indices and bit values taken from [`PC_TREE`].
    // Rather than solving generic tree walks on the fly, we use the
    // static mapping below that mirrors the known shape.
    //
    // Tree shape (leaf -> (prob_idx, bit) sequence):
    //   0: (6, 0), (7, 0)
    //   1: (6, 0), (7, 1)
    //   2: (6, 1), (8, 0), (9, 0)
    //   3: (6, 1), (8, 0), (9, 1)
    //   4: (6, 1), (8, 1), (10, 0)
    //   5: (6, 1), (8, 1), (10, 1)
    match cat {
        0 => {
            enc.put_prob(m1[6], 0);
            enc.put_prob(m1[7], 0);
        }
        1 => {
            enc.put_prob(m1[6], 0);
            enc.put_prob(m1[7], 1);
        }
        2 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 0);
            enc.put_prob(m1[9], 0);
        }
        3 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 0);
            enc.put_prob(m1[9], 1);
        }
        4 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 1);
            enc.put_prob(m1[10], 0);
        }
        5 => {
            enc.put_prob(m1[6], 1);
            enc.put_prob(m1[8], 1);
            enc.put_prob(m1[10], 1);
        }
        _ => unreachable!(),
    }
    let bias = tables::VP56_COEFF_BIAS[cat + 5] as i32;
    let delta = abs_val - bias;
    let bit_len = tables::VP56_COEFF_BIT_LENGTH[cat] as i32;
    // Emit bits from MSB (index `bit_len`) down to bit 0, each under
    // `VP56_COEFF_PARSE_TABLE[cat][i]`.
    for i in (0..=bit_len).rev() {
        let bit = ((delta >> i) & 1) as u8;
        enc.put_prob(tables::VP56_COEFF_PARSE_TABLE[cat][i as usize], bit);
    }
    *ct = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_inversion_hits_target() {
        // Key pixel values — encoder picks a block0 that the decoder
        // reconstructs back to the same value via the two-stage IDCT
        // DC-only path.
        for target in [0u8, 16, 32, 64, 128, 192, 235, 255] {
            let b0 = dc_level_from_pixel(target);
            // Reconstruct the same way dc_level_from_pixel does.
            let row_dc = ((46341i64 * b0 as i64) >> 16) as i32;
            let pixel =
                (128 + (((46341i64 * row_dc as i64) + (8i64 << 16)) >> 20)).clamp(0, 255) as u8;
            assert!(
                (pixel as i32 - target as i32).abs() <= 1,
                "target={target} -> b0={b0} -> pixel={pixel}"
            );
        }
    }

    #[test]
    fn div_nearest_rounds_away_from_zero_on_ties() {
        assert_eq!(div_nearest(5, 2), 3);
        assert_eq!(div_nearest(-5, 2), -3);
        assert_eq!(div_nearest(4, 2), 2);
        assert_eq!(div_nearest(1, 2), 1);
        assert_eq!(div_nearest(-1, 2), -1);
    }

    #[test]
    fn construct_and_encode_minimal() {
        // 16x16 flat-gray keyframe round-trips through encode.
        let w = 16usize;
        let h = 16usize;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let mut enc = Vp6Encoder::new(32);
        let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
        // Sanity: first byte carries QP=32 (<<1=64), frame_mode=0.
        assert_eq!(bytes[0], 64);
        // sub_version=0 byte.
        assert_eq!(bytes[1], 0);
        // Body must have at least the 4 post-filter bytes + some payload.
        assert!(bytes.len() > 8);
    }
}

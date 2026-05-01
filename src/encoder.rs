//! VP6F (VP6 for FLV) encoder.
//!
//! Keyframes emit both DC and AC coefficients through the bool-coded
//! coefficient path the decoder reads in [`crate::mb::parse_coeff`];
//! the output is a standards-compliant VP6F elementary stream that
//! our own decoder — and ffmpeg's `vp6f` decoder — round-trip.
//!
//! P-frames are emitted via [`Vp6Encoder::encode_skip_frame`] (identity
//! copy of the previous frame) or [`Vp6Encoder::encode_inter_frame`]
//! (integer-pel motion-vector search + `InterDeltaPf` mb-type emission;
//! no coded residual). As of r23 ffmpeg's `vp6f` decoder accepts both
//! inter paths end-to-end (`tests/ffmpeg_interop.rs::*` all pin
//! `n == 2`).
//!
//! Round-19 audit (recorded so the next round doesn't repeat steps):
//!
//! * `tests/dump_inter.rs` (run with `VP6_DUMP_INTER=1`) writes a 2-tag
//!   FLV (key + skip) to `/tmp/oxideav_vp6_dump.flv`, hex-dumps both
//!   packets, traces the first ~10 bool-coded symbols of the skip
//!   packet through our decoder, and confirms our decoder accepts.
//! * **Bug fixed**: `Buff2Offset` (VP6 spec Tables 2 & 3, R(16) raw
//!   field) was being emitted with a `+2` fudge factor on the encoder
//!   side and parsed with a matching `-2` on the decoder side. The
//!   spec defines the value as the literal frame-buffer byte offset of
//!   partition 2, no fudge. Both encoder and decoder now use that
//!   convention; the keyframe path translates raw → rac-relative by
//!   subtracting `range_coder_offset` (the bool-coder priming point),
//!   matching the spec exactly. Inter frames now expose a
//!   `tests/dump_inter.rs::inter_buff2_offset_is_spec_compliant` guard.
//! * Reproduced in r18 and confirmed in r19 post-fix: ffmpeg accepts
//!   the FLV container + the keyframe but still rejects the inter
//!   packet ("Invalid data found when processing input"). The Buff2
//!   field is now spec-compliant (verified by manually parsing the dump
//!   in /tmp/oxideav_vp6_dump.flv), so the residual divergence is past
//!   the partition-layout layer.
//!
//! Round-20 audit (delta — completed; ffmpeg interop still pending):
//!
//! * **Bug fixed**: `DEF_MB_TYPES_STATS` pair order. Per spec page 30
//!   `VP6_BaselineXmittedProbs[3][20]` and page 29 Table 6 (second-
//!   dimension semantics), each row flattens as
//!   `(probSame_t, probDiff_t)` for `t` in `0..10`. Our table stored
//!   the pairs swapped; `rebuild_mb_type_probs` (which mirrors spec
//!   page 35's `probModeSame` formula `255 - 255 * pX[k][i*2] / (1 +
//!   pX[k][i*2] + pX[k][i*2+1])`) was therefore computing values with
//!   stay-rate semantics where the spec yields switch-rate semantics.
//!   `PRE_DEF_MB_TYPE_STATS` was already in spec order so the two
//!   tables also disagreed with each other on a SetNewBaselineProbs
//!   reset; the swap brings DEF in line with both spec page 30 and
//!   the existing PRE_DEF layout. Pinned by
//!   `tables::tests::def_mb_types_stats_matches_spec_baseline`.
//! * **Behaviour**: own-codec round-trips still pass (the encoder +
//!   decoder both consume the new layout consistently). FFmpeg
//!   continues to reject the inter packet (1 frame decoded, 1 decode
//!   error); the spec-compliance fix is necessary but not sufficient
//!   on its own. The `tests/keyframe_from_flv.rs::decode_first_20_frames`
//!   guard still passes after the swap because the embedded
//!   FFmpeg-encoded sample issues SetNewBaselineProbs updates that
//!   overwrite the baseline before any per-MB decode reads it, so the
//!   layout disagreement at the baseline never surfaces in that test.
//! * r22 update — the `vector_predictors` ctx mapping was switched to
//!   the spec page 28 Table 5 form (`ctx = 2 - nb_pred`) on both
//!   decoder + encoder sides. The skip-frame encoder's hard-coded
//!   `ctx = 1` was changed to `ctx = 2` to match the new decoder
//!   return (`nb_pred = 0` for all neighbours OOB / zero-MV).
//!
//!   Audit of the per-MB block coefficient state machine confirmed
//!   the 3-bit "all zero" shortcut path matches `parse_coeff`'s exit
//!   conditions: at `coeff_idx = 0` the decoder reads `m2_0` only
//!   (DC has no EOB token by spec); at `coeff_idx = 1` with `ct = 0`
//!   the shortcut `coeff_idx > 1 && ct == 0` is **false** (1 is not
//!   strictly greater than 1), so the decoder reads `m2_0` then
//!   `m2_1` (EOB) — exactly the encoder's three emissions.
//!   `VP6_COEFF_GROUPS[1] = 0` so `cg = 0` for the AC pair, matching
//!   the encoder's index choices.
//!
//!   ffmpeg outcome (r22): the inter packet remains rejected. r21
//!   verified each candidate fix in isolation didn't move the needle;
//!   r22 landed both together (ctx mapping fix + coeff shortcut audit)
//!   and ffmpeg still reports "Invalid data found when processing
//!   input" — so the residual issue lives somewhere else (candidates
//!   for r23: keyframe `Vp3VersionNo = 0` may be lenient in ffmpeg's
//!   keyframe path but mis-route the inter parser; the per-MB
//!   coefficient model state ffmpeg expects after the keyframe may
//!   diverge from the `0x80` baseline our encoder pins).
//!
//! Round-23 audit (delta — INTEROP UNBLOCKED):
//!
//! * **Bug fixed**: `Vp6Encoder::default().sub_version` was 0, which
//!   the encoder serialised into byte 1 of the keyframe header as
//!   `0 << 3 = 0x00`. VP6 spec §9 / Table 2 (`IntraHeader`) defines
//!   bits 7..3 of byte 1 as `Vp3VersionNo` R(5), required to hold the
//!   value 6, 7, or 8 (VP6.0 / VP6.1 / VP6.2). The value 0 is
//!   forbidden. ffmpeg's keyframe-decode path silently accepted the
//!   illegal 0 (so our keyframe round-tripped pre-r23) but routed the
//!   inter parser through a code path that mishandled subsequent
//!   frames, surfacing as the long-running "Invalid data found when
//!   processing input" inter-frame error. The fix is one byte:
//!   default `sub_version` is now 6 (VP6.0 / Simple Profile), so byte
//!   1 reads `6 << 3 = 0x30`. Pinned by
//!   `tests/ffmpeg_interop.rs::keyframe_vp3_version_no_is_spec_legal`.
//! * **Behaviour**: `tests/ffmpeg_interop.rs` flips green —
//!   `r21_inter_frame_ffmpeg_decode_state` and
//!   `ffmpeg_decodes_keyframe_in_two_tag_stream` both now strictly
//!   assert `n == 2` (both keyframe + inter accepted by ffmpeg). The
//!   `decode_first_20_frames` regression on the embedded FLV sample
//!   continues to pass — the decoder's `sub_version` gates
//!   (`> 7` / `< 8`) all behave the same for `sub_version = 6` as
//!   they did for `sub_version = 0`. The on-wire byte count is
//!   unchanged; only the value of byte 1 moves from `0x00` to `0x30`.
//! * Spec cross-reference: page 23 Table 1 (Frame Header) lists the
//!   8-byte keyframe-prelude shape; page 24 Table 2 (IntraHeader)
//!   defines the 5-bit `Vp3VersionNo` and 2-bit `VpProfile` fields
//!   that share byte 1; page 25 explicitly states "The decoder
//!   should check this field to ensure that it can decode the
//!   bitstream", confirming the value is gated, not advisory.
//!
//! Round-24 audit (delta — inter residual coefficient encoding):
//!
//! * **Feature added**: `encode_inter_frame` emits real DCT residual
//!   per block. Previously every block emitted the 3-bool "all zero"
//!   shortcut, so reconstruction was the MC prediction alone (a hard
//!   PSNR ceiling on any content the integer-pel MV couldn't capture).
//!   The new path mirrors the decoder's `parse_coeff` +
//!   `add_predictors_dc(.., RefKind::Previous)` chain on the encoder
//!   side: build the integer-pel MC tile (`sample_mc_tile`), compute
//!   the pixel residual, forward DCT (`forward_dct8x8_residual` —
//!   same scaling as the keyframe DCT but without the `-128` bias
//!   since the residual is already centred on zero), quantise, run
//!   the DC predictor, and emit through the same `emit_block_coefs`
//!   state machine the keyframe path uses. Per-block state
//!   (`enc_left_block`, `enc_above_blocks`,
//!   `enc_prev_dc[plane][Previous]`) is tracked alongside so
//!   subsequent MBs see exactly the same predictor the decoder will
//!   compute.
//! * **Behaviour**: in-tree decoder Y PSNR on the new
//!   `r24_inter_residual_psnr_floor` fixture (flat keyframe + per-MB
//!   brightness shift) jumps from ~19 dB (MC-only baseline — the
//!   pre-r24 ceiling) to ~43 dB (with residual). The
//!   `inter_frame_horizontal_shift_uses_mv` fixture (MC-friendly
//!   content where residual ≈ 0) records 40+ dB unchanged.
//! * **ffmpeg cross-decode**: ffmpeg accepts both packets in the
//!   key + inter stream (`r21_inter_frame_ffmpeg_decode_state` still
//!   passes `n == 2`). Cross-decoded residual content however lands
//!   on the MC-only baseline, suggesting the per-MB coefficient model
//!   state ffmpeg expects after the keyframe diverges from the
//!   `0x80` baseline this encoder pins. r25+ work.
//!
//! Round-25 audit (delta — quarter-pel sub-pel motion estimation):
//!
//! * **Feature added**: `motion_search` now picks quarter-pel
//!   accurate MVs via a two-stage search. Stage 1 runs the existing
//!   integer-pel SAD search to seed `(int_dx, int_dy)`. Stage 2
//!   evaluates every quarter-pel offset in a `±3 qpel` window around
//!   the integer winner via the same H.264-chroma-style bilinear
//!   filter the decoder uses (`mb::render_mb_inter` `use_bicubic_luma
//!   == false` branch — see `bilinear_luma_sample`). Each qpel
//!   candidate's cost is `SAD(MC) + λ * mv_bits` with `λ` proportional
//!   to QP, so sub-pel wins are taken only when they measurably beat
//!   the integer winner including the extra MV-bit cost. The
//!   MC-tile sampler (`sample_mc_tile`) likewise grew a sub-pel branch
//!   so the residual computation matches the decoder exactly when the
//!   chosen MV has sub-pel components. Spec ref: `vp6_format.pdf`
//!   §17.2 (Half / Quarter Pixel Aligned Vectors).
//! * **Behaviour**: internal-decoder Y PSNR on the new
//!   `r25_qpel_translating_stripes_psnr_clears_35db` /
//!   `r25_qpel_translating_disk_psnr_clears_35db` fixtures (0.5-pel
//!   sub-pel shift, smooth low-frequency content) climbs from ~19-29
//!   dB (integer-only MC baseline) to 35-37 dB (qpel MC + DCT
//!   residual). The pre-r25
//!   `inter_frame_horizontal_shift_uses_mv` fixture (4-pel integer
//!   shift, no sub-pel component) records the same PSNR as before.
//!   `r24_inter_residual_psnr_floor` (zero-MV brightness ramp)
//!   continues to pass at the same level.
//! * **ffmpeg cross-decode**: ffmpeg's vp6f decoder accepts the
//!   qpel-MV inter packet (`r25_ffmpeg_decodes_qpel_inter_frame` —
//!   reconstructs ~32 dB Y on the stripes fixture). The existing
//!   `r21_inter_frame_ffmpeg_decode_state` interop guard still passes
//!   `n == 2`.
//!
//! Scope:
//!
//! * Sub-version 0 (simple profile), `filter_header = 0`, interlaced=0.
//! * Bool-path coefficients (no Huffman).
//! * Inter frames: quarter-pel MVs (r25+), mb_type ∈ {InterNoVecPf,
//!   InterDeltaPf}, real DCT residual encoded per block (r24+).
//!
//! The forward DCT is a float-based DCT-II scaled so `block[u*8+v] =
//! F[u,v]` feeds through the decoder's two-stage IDCT back to the same
//! tile (within rounding). Natural raster layout — `u` vertical freq,
//! `v` horizontal freq — matches the VP6 spec (section 12.1 / 16).
//! Quantisation is a nearest-integer division by `dequant_ac` for AC
//! bins, matched to the decoder's multiply-back.

use oxideav_core::{Error, Result};

use crate::mb::{Mv, RefKind};
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
/// stream frame and resets internal state. Subsequent inter-frame APIs:
///
/// * [`Vp6Encoder::encode_skip_frame`] — every MB coded as `InterNoVecPf`
///   (copy previous frame, no residual). Useful as a minimum-viable
///   identity inter, and as the fallback when no motion search is wired.
/// * [`Vp6Encoder::encode_inter_frame`] — integer-pel SAD motion search
///   per MB; emits `InterDeltaPf` (with delta-from-candidate MV) where
///   motion is found and `InterNoVecPf` elsewhere. No residual coded —
///   reconstruction is the MC prediction alone.
///
/// Both inter paths round-trip cleanly through our own decoder.
/// ffmpeg-side acceptance of inter frames is still pending — see the
/// per-method docs.
#[derive(Debug)]
pub struct Vp6Encoder {
    /// Quantiser parameter (0 = highest quality, 63 = coarsest).
    pub qp: u8,
    /// Use the `sub_version = 0` "simple profile" layout.
    pub sub_version: u8,
    /// Period (in inter frames since the last golden refresh) at which
    /// [`Vp6Encoder::encode_inter_frame_with_golden`] flips the
    /// `golden_frame_flag` bit on the inter picture header. The decoder
    /// then snaps the just-decoded reconstruction into its
    /// `golden_frame` slot, so subsequent inter MBs that pick a golden
    /// reference (`InterNoVecGf` / `InterDeltaGf` etc.) reference the
    /// refreshed plane rather than the keyframe-time golden.
    ///
    /// `0` disables golden-refresh entirely (the encoder never sets the
    /// flag and never emits golden-ref MBs). `30` is the default —
    /// roughly one refresh per second at 30 fps, picked to give
    /// slideshow / animation-loop content a "good" golden cadence
    /// without paying the keyframe cost (a golden-refresh frame is
    /// still a P-frame with bool-coded residual).
    pub golden_refresh_period: u32,
    /// Dimensions (in MBs) from the last encoded keyframe. Needed so
    /// [`Vp6Encoder::encode_skip_frame`] can produce a well-formed
    /// inter-frame header without requiring callers to re-supply them.
    mb_width: u16,
    mb_height: u16,
    /// True once a keyframe has been successfully emitted — gates
    /// `encode_skip_frame` so callers can't emit an inter before a
    /// keyframe is established.
    have_keyframe: bool,
    /// Inter frames emitted since the last golden refresh (or since
    /// the keyframe — keyframes implicitly refresh the decoder's
    /// golden slot per `decoder.rs` line 422). Incremented on every
    /// `encode_inter_frame*` call; reset to 0 when
    /// [`Self::should_refresh_golden`] returns true and the encoder
    /// emits the `golden_frame_flag` bit.
    inter_frames_since_golden: u32,
}

impl Default for Vp6Encoder {
    fn default() -> Self {
        Self {
            qp: 32,
            // VP6 spec Table 2: `Vp3VersionNo` is R(5) and is required to
            // hold the values 6 (VP6.0), 7 (VP6.1) or 8 (VP6.2). The
            // decoder explicitly checks this — the value 0 (the pre-r23
            // default) is forbidden by spec. Round-trip-wise our own
            // decoder accepts 0..=8, but ffmpeg's decoder routes the
            // inter-frame parser through paths gated on `sub_version`
            // (VP6.0 vs VP6.2 sub-pel filter selection) where a 0 lands
            // on a Vp6.<keyframe-only> code path that mishandles
            // subsequent inter frames. Default to 6 (VP6.0 / Simple
            // Profile) so the keyframe header advertises a valid
            // version. (See r23 audit notes in this module's head
            // comment.)
            sub_version: 6,
            golden_refresh_period: 30,
            mb_width: 0,
            mb_height: 0,
            have_keyframe: false,
            inter_frames_since_golden: 0,
        }
    }
}

impl Vp6Encoder {
    /// New encoder with a given QP (clamped to 0..=63). Emits VP6.0
    /// (`Vp3VersionNo = 6`) headers; see [`Self::default`] for why this
    /// matters for ffmpeg-side interop.
    pub fn new(qp: u8) -> Self {
        Self {
            qp: qp.min(63),
            sub_version: 6,
            golden_refresh_period: 30,
            mb_width: 0,
            mb_height: 0,
            have_keyframe: false,
            inter_frames_since_golden: 0,
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
        out.push((self.qp << 1) & 0x7E);
        out.push(self.sub_version << 3);
        out.push(0);
        out.push(2);
        out.push(mb_height as u8);
        out.push(mb_width as u8);
        out.push(mb_height as u8);
        out.push(mb_width as u8);

        // --- Bool-coded body ---------------------------------------------
        let mut enc = RangeEncoder::new();
        // 2 skip bits.
        enc.put_bits(2, 0);
        // use_huffman = 0.
        enc.put_bit(0);

        // --- Coefficient-model update pass -------------------------------
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);

        // DC value-update probabilities: emit 0 for every node. On the
        // decoder side `def_prob` is carried forward and (since key=true)
        // copied through into `coeff_dccv[pt][node]`.
        let def_prob_template = [0x80u8; 11];
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
                model.coeff_dccv[pt][node] = def_prob_template[node];
            }
        }

        // Coefficient-reorder update flag = 0 (keep defaults).
        enc.put_bit(0);

        // Run-value update probabilities — emit 0 for every node.
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }

        // AC coefficient update probabilities — emit 0 for every node.
        // The decoder carries `def_prob` through this nested loop.
        let mut def_prob = def_prob_template;
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                        model.coeff_ract[pt][ct][cg][node] = def_prob[node];
                    }
                }
            }
        }
        let _ = &mut def_prob;

        // Recompute the `coeff_dcct` linear combination.
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
        let dequant_dc = (tables::VP56_DC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;
        let dequant_ac = (tables::VP56_AC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;

        let mut left_block: [EncRefDc; 4] = [EncRefDc::default(); 4];
        let mut above_blocks: Vec<EncRefDc> = vec![EncRefDc::default(); 4 * mb_width + 6];
        if 2 * mb_width + 2 < above_blocks.len() {
            above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < above_blocks.len() {
            above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        let mut prev_dc = [[0i16; 3]; 3];
        prev_dc[1][0] = 128;
        prev_dc[2][0] = 128;

        for mb_row in 0..mb_height {
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
                for b in 0..6usize {
                    let pt = if b > 3 { 1usize } else { 0usize };
                    let plane_idx = tables::B2P[b] as usize;
                    let ctx = left_block[tables::B6_TO_4[b] as usize].not_null_dc as usize
                        + above_blocks[above_block_idx[b]].not_null_dc as usize;

                    // Gather the 8x8 pixel tile for this block.
                    let mut tile = [0i32; 64];
                    sample_block_tile(
                        b, mb_row, mb_col, y_plane, u_plane, v_plane, width, uv_stride, &mut tile,
                    );

                    // Forward DCT into 64 raw coefficients (natural order
                    // — `coefs[u*8+v]` is the frequency coefficient at
                    // (u, v), subtracted by 128 + scaled to match the
                    // decoder's IDCT expectation).
                    let mut coefs = [0i32; 64];
                    forward_dct8x8(&tile, &mut coefs);

                    // DC quantisation: `new_dc * dequant_dc` gives the
                    // block[0] value that the IDCT consumes.
                    let new_dc = div_nearest(coefs[0], dequant_dc).clamp(-32768, 32767) as i16;

                    // Compute the decoder's predictor_dc for this block.
                    let lb = left_block[tables::B6_TO_4[b] as usize];
                    let ab = above_blocks[above_block_idx[b]];
                    let mut count = 0i32;
                    let mut pdc = 0i32;
                    if lb.ref_frame == RefKind::Current {
                        pdc += lb.dc_coeff as i32;
                        count += 1;
                    }
                    if ab.ref_frame == RefKind::Current {
                        pdc += ab.dc_coeff as i32;
                        count += 1;
                    }
                    let predictor = match count {
                        0 => prev_dc[plane_idx][0] as i32,
                        2 => pdc / 2,
                        _ => pdc,
                    };
                    let coded_dc = (new_dc as i32 - predictor).clamp(-32768, 32767) as i16;

                    // Quantise AC coefficients (in coeff_idx order) — we
                    // walk coeff_idx = 1..=63, look up `pos =
                    // coeff_index_to_pos[coeff_idx]`, then the natural
                    // block index is `permute[pos]`. Quantise with
                    // `dequant_ac` and store as signed levels.
                    let mut ac_levels = [0i32; 64];
                    for coeff_idx in 1..64usize {
                        let pos = model.coeff_index_to_pos[coeff_idx] as usize;
                        let perm = tables::IDCT_SCANTABLE[pos] as usize;
                        // Clamp to the range the decoder's value-tree
                        // can represent (up to `2^11 + bias`).
                        ac_levels[coeff_idx] =
                            div_nearest(coefs[perm], dequant_ac).clamp(-2047, 2047);
                    }

                    // Emit the coefficient stream for this block. The
                    // decoder iterates coeff_idx from 0 upward, branching
                    // on parse-or-skip at each step. We mirror that state
                    // machine exactly.
                    let last_nz = find_last_nonzero(&ac_levels);
                    emit_block_coefs(&mut enc, &model, pt, ctx, coded_dc, &ac_levels, last_nz);

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

                for y in 0..4 {
                    above_block_idx[y] += 2;
                }
                for uv in 4..6 {
                    above_block_idx[uv] += 1;
                }
            }
        }

        out.extend_from_slice(&enc.finish());
        self.mb_width = mb_width as u16;
        self.mb_height = mb_height as u16;
        self.have_keyframe = true;
        // The decoder snaps the keyframe reconstruction into its
        // `golden_frame` slot (see decoder.rs:422), so the cadence
        // counter resets here too — the next inter frame is "0 inters
        // since golden refresh" by definition.
        self.inter_frames_since_golden = 0;
        Ok(out)
    }

    /// Emit a minimal P-frame: every macroblock is `InterNoVecPf`
    /// (reference = previous frame, zero motion vector) with no coded
    /// residual. Our in-tree decoder reconstructs by copying the
    /// previous frame 1:1, so this is a valid "identity" inter frame.
    ///
    /// Scope:
    /// * Bitstream layout: inter-frame header, bool-coded picture header
    ///   with zero updates to every prob model (mb-type, vector, and
    ///   inter coefficient), one `stay on prev_type` bit per MB, and
    ///   three zero-EOB bits per block (all coefficients zero).
    /// * Compatibility: verified end-to-end with our decoder; the
    ///   ffmpeg reference decoder currently rejects the output (the
    ///   prob-encoding of the mb-type model update pass and/or some
    ///   subsequent flag diverges from ffmpeg's expectations). The
    ///   mismatch is limited to inter-frame probabilities, not the
    ///   keyframe path which rides through cleanly.
    /// * Next steps: fixing the ffmpeg-side acceptance, wiring up real
    ///   MV decisions via `VP56_PVA_TREE` / `vector_fdv`, and emitting
    ///   actual residual coefficients through the inter-key coeff-model
    ///   branch. All three can build on top of this scaffold without
    ///   reshaping the bitstream surface.
    ///
    /// Returns an error if no keyframe has been encoded yet, since the
    /// decoder needs to have parsed the keyframe-side of the model /
    /// dimensions before it can consume an inter frame.
    pub fn encode_skip_frame(&mut self) -> Result<EncodedFrame> {
        if !self.have_keyframe {
            return Err(Error::invalid(
                "VP6 encode: skip frame requires a preceding keyframe",
            ));
        }
        let mb_width = self.mb_width as usize;
        let mb_height = self.mb_height as usize;

        // --- Fixed header (inter layout) ---------------------------------
        // byte 0: frame_mode=1 (top bit), 6-bit QP, MultiStream/separated_coeff=1.
        //
        // VP6 spec Tables 1 & 3: byte 0 carries FrameType R(1) +
        // DctQMask R(6) + MultiStream R(1). When MultiStream==1 OR
        // SIMPLE_PROFILE==1, Buff2Offset R(16) follows directly as 2
        // raw bytes giving the frame-buffer offset of partition 2.
        // Our keyframes use sub_version=0 (SIMPLE_PROFILE) with
        // filter_header=0, so the inter parse on the decoder side
        // always reads Buff2Offset — emit it.
        let mut out = Vec::<u8>::with_capacity(16 + mb_width * mb_height * 2);
        out.push(0x80 | ((self.qp << 1) & 0x7E) | 0x01); // MultiStream=1
                                                         // Buff2Offset placeholder; overwritten once partition 1 is sized.
        out.push(0);
        out.push(0);
        let buff2_hi_idx = out.len() - 2;
        let buff2_lo_idx = out.len() - 1;

        // --- Partition 1: bool-coded picture header ----------------------
        let mut enc = RangeEncoder::new();
        // golden_frame_flag = 0. No filter-info path because our
        // keyframes use filter_header = 0 + sub_version = 0.
        enc.put_bit(0);
        // use_huffman = 0 (always bool-coded path here).
        enc.put_bit(0);

        // --- MB-type probability model update pass (parse_mb_type_models).
        // For every context and prob flag we emit 0 so the decoder
        // retains its default DEF_MB_TYPES_STATS; after rebuild those
        // give prob(mb_type[ctx][InterNoVecPf][0]) = a usable value we
        // then feed "stay on prev_type" bits through.
        for _ctx in 0..3 {
            enc.put_prob(174, 0); // no per-context stats reset
            enc.put_prob(254, 0); // no per-entry delta updates
        }

        // --- Vector-model update pass (parse_vector_models). Zero for
        // every flag => defaults retained.
        for comp in 0..2 {
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][0], 0);
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][1], 0);
        }
        for comp in 0..2 {
            for node in 0..7 {
                enc.put_prob(tables::VP6_PDV_PCT[comp][node], 0);
            }
        }
        for comp in 0..2 {
            for node in 0..8 {
                enc.put_prob(tables::VP6_FDV_PCT[comp][node], 0);
            }
        }

        // --- Coefficient-model update pass (parse_coeff_models, key=false).
        // Zero for every flag. On `key=false` the decoder does NOT copy
        // `def_prob` through for skipped nodes — the table carries over
        // whatever the keyframe established, which is what we want.
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
            }
        }
        // Coefficient-reorder update flag = 0.
        enc.put_bit(0);
        // Run-value update probabilities.
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }
        // AC coefficient update probabilities.
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                    }
                }
            }
        }

        // --- Per-MB: keep mb_type = InterNoVecPf for every MB, emit
        // zero-residual for all 6 blocks.
        //
        // `parse_mb_type` reads `rac.get_prob(mb_type_model[ctx][prev][0])`
        // as a "stay on prev_type" prob: bit=1 means "yes, same as
        // prev_type". We want InterNoVecPf every time, so we shadow the
        // decoder's model-rebuild sequence on the encoder side too, and
        // feed bit=1 through the correct prob.
        //
        // Critically, the decoder's `coeff_dccv` / `coeff_ract` carry
        // their `0x80` defaults from the keyframe (we set them in
        // `encode_keyframe`), and the inter-frame `parse_coeff_models`
        // does NOT copy `def_prob` through on `key=false` — so they
        // stay at `0x80`. Mirror that here, then rebuild `coeff_dcct`
        // through the same linear combination the decoder uses, so the
        // per-MB zero-block bits we emit use the same probabilities the
        // decoder reads back with.
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);
        for pt in 0..2 {
            for node in 0..11 {
                model.coeff_dccv[pt][node] = 0x80;
            }
        }
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        model.coeff_ract[pt][ct][cg][node] = 0x80;
                    }
                }
            }
        }
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
        // parse_mb_type_models emitted no updates => stats untouched =>
        // mb_type[] matches what rebuild_mb_type_probs produces from
        // DEF_MB_TYPES_STATS.
        model.rebuild_mb_type_probs();

        // Partition 2: per-MB DCT coefficients only.
        let mut enc2 = RangeEncoder::new();
        for _mb_row in 0..mb_height {
            for _mb_col in 0..mb_width {
                // -- Partition 1: prediction info (MB-type only — every
                // MB stays on InterNoVecPf so no MV bits). All neighbour
                // candidate slots are out-of-bounds / zero-MV, so the
                // decoder's `vector_predictors` returns nb_pred=0 →
                // ctx = 2 - nb_pred = 2 (spec page 28 Table 5: "Neither
                // Nearest nor Near MVs exists for this macroblock").
                let ctx: usize = 2;
                let prev_type = tables::Vp56Mb::InterNoVecPf as usize;
                let prob = model.mb_type[ctx][prev_type][0];
                enc.put_prob(prob, 1); // stay on InterNoVecPf

                // -- Partition 2: 6 blocks per MB. Each block: 3 bool
                // bits for "all zero" (m2_0 at idx=0, m2_0 at idx=1,
                // m2_1 at idx=1 = EOB).
                for b in 0..6usize {
                    let pt = if b > 3 { 1 } else { 0 };
                    // ctx for DC = 0 (both left + above not_null_dc = 0
                    // in a skip frame).
                    let dc_ctx = 0usize;
                    // idx=0: skip path. m2_0 = coeff_dcct[pt][0][0].
                    enc2.put_prob(model.coeff_dcct[pt][dc_ctx][0], 0);
                    // idx=1: still within go_parse = false shortcut
                    // boundary (`coeff_idx > 1 && ct == 0` requires
                    // idx>=2); emit m2_0 = coeff_ract[pt][0][cg=0][0].
                    enc2.put_prob(model.coeff_ract[pt][0][0][0], 0);
                    // idx=1, coeff_idx>0: EOB bit = coeff_ract[...][1].
                    enc2.put_prob(model.coeff_ract[pt][0][0][1], 0);
                }
            }
        }

        // Mux: header (3 bytes) + partition1 + partition2.
        //
        // Buff2Offset is the spec-defined raw byte offset from the start
        // of the frame buffer to the start of partition 2 (VP6 spec §9 /
        // Table 3). Partition 2 starts at `header_size (3) + p1.len()`,
        // so the wire value is exactly that number — no fudge factor.
        let p1 = enc.finish();
        let p2 = enc2.finish();
        out.extend_from_slice(&p1);
        let buff2 = ((3 + p1.len()) as u32).min(0xFFFF);
        out[buff2_hi_idx] = ((buff2 >> 8) & 0xFF) as u8;
        out[buff2_lo_idx] = (buff2 & 0xFF) as u8;
        out.extend_from_slice(&p2);
        // Skip frames carry no golden_frame_flag (the bit was emitted
        // as 0 above). Bump the cadence counter so a downstream
        // `encode_inter_frame_with_golden` call still measures the
        // refresh interval correctly.
        self.inter_frames_since_golden = self.inter_frames_since_golden.saturating_add(1);
        Ok(out)
    }

    /// Emit a P-frame with quarter-pel-accurate motion vectors against
    /// the supplied previous reconstruction (`prev_*` planes). Per-MB a
    /// two-stage ME picks a luma quarter-pel offset:
    ///
    /// 1. integer-pel SAD search in `[-search, search]` (in pixels) —
    ///    same shape as r23/r24;
    /// 2. quarter-pel refine around the integer winner via the
    ///    H.264-chroma-style bilinear filter the decoder uses
    ///    (`mb::render_mb_inter`'s `use_bicubic_luma == false` branch),
    ///    with a Lagrangian tie-breaker `SAD + λ * mv_bits` so we don't
    ///    pay extra MV bits for noise-level wins.
    ///
    /// MBs whose best MV is `(0, 0)` are emitted as `InterNoVecPf`;
    /// otherwise as `InterDeltaPf` with the MV delta from the candidate
    /// predictor (mirroring the decoder's `vector_predictors` +
    /// `parse_vector_adjustment` walk).
    ///
    /// Spec references in `vp6_format.pdf`:
    /// * Section 10 (Mode Decoding) — MB-type tree (Figure 10) and
    ///   `probXmitted` model. We keep the model defaults intact (no
    ///   updates emitted in the picture header).
    /// * Section 11 (Motion Vectors) — short / long vector encoding
    ///   trees, mirrored on the encoder side via [`encode_mv_component`].
    /// * Section 11.2 (MV Probability Updates) — we emit "no update"
    ///   for every flag so the default probabilities continue to apply.
    /// * Section 17.2 (Half / Quarter Pixel Aligned Vectors) — sub-pel
    ///   phase = `mv.x & 3` mapped to 8-step phases via `* 2`; bilinear
    ///   filter taps a 2x2 window around the integer base.
    ///
    /// Round-24:
    /// * Inter residual coefficients emitted (DCT + quantise + DC
    ///   predictor + per-block coeff state mirror) — see
    ///   `r24_inter_residual_psnr_floor`.
    ///
    /// Round-25 (this revision):
    /// * **Quarter-pel sub-pel ME.** `motion_search` returns qpel
    ///   `(qdx, qdy)` directly; `sample_mc_tile` mirrors the decoder's
    ///   bilinear branch (`bilinear_luma_sample`) so the residual
    ///   computation matches what the decoder will compute. PSNR on
    ///   sub-pel-translation fixtures climbs from ~19-29 dB
    ///   (integer-only) to 35-37 dB (qpel). ffmpeg cross-decodes the
    ///   qpel-MV inter packet cleanly.
    ///
    /// Limitations (still pending r26+):
    /// * Single-MV-per-MB only (mb_type ∈ {InterNoVecPf, InterDeltaPf}).
    ///   No 4V / golden modes.
    /// * `search` window is `±search` integer pels in both axes (qpel
    ///   refine extends to `±search + 1` integer pels via the bilinear
    ///   taps); MV magnitude is clamped so the long-vector encoding
    ///   fits the spec's 7-bit absolute range (≤ 127 quarter-pel
    ///   units, i.e. ≤ 31 integer pels).
    pub fn encode_inter_frame(
        &mut self,
        prev_y: &[u8],
        prev_u: &[u8],
        prev_v: &[u8],
        new_y: &[u8],
        new_u: &[u8],
        new_v: &[u8],
        width: usize,
        height: usize,
        search: i32,
    ) -> Result<EncodedFrame> {
        if !self.have_keyframe {
            return Err(Error::invalid(
                "VP6 encode: inter frame requires a preceding keyframe",
            ));
        }
        let mb_width = self.mb_width as usize;
        let mb_height = self.mb_height as usize;
        if width != mb_width * 16 || height != mb_height * 16 {
            return Err(Error::invalid(
                "VP6 encode: inter-frame dims must match the preceding keyframe",
            ));
        }
        let uv_stride = width / 2;
        let uv_h = height / 2;
        let need_y = width * height;
        let need_uv = uv_stride * uv_h;
        if prev_y.len() < need_y
            || new_y.len() < need_y
            || prev_u.len() < need_uv
            || prev_v.len() < need_uv
            || new_u.len() < need_uv
            || new_v.len() < need_uv
        {
            return Err(Error::invalid("VP6 encode: plane buffers too small"));
        }

        // --- Fixed inter header (MultiStream=1, two partitions) ---------
        // VP6 spec §5: SIMPLE_PROFILE encoders emit DCT tokens in a
        // separate partition. ffmpeg's vp6f decoder enforces this on
        // inter frames; see comment in `encode_skip_frame`.
        let mut out = Vec::<u8>::with_capacity(16 + mb_width * mb_height * 8);
        out.push(0x80 | ((self.qp << 1) & 0x7E) | 0x01);
        out.push(0);
        out.push(0);
        let buff2_hi_idx = out.len() - 2;
        let buff2_lo_idx = out.len() - 1;

        let mut enc = RangeEncoder::new();
        // golden_frame_flag = 0; use_huffman = 0.
        enc.put_bit(0);
        enc.put_bit(0);

        // --- All probability-model "update" pass blocks ------------------
        // We carry the keyframe-time defaults forward by emitting "no
        // update" for every flag, matching the decoder's parse path.
        for _ctx in 0..3 {
            enc.put_prob(174, 0);
            enc.put_prob(254, 0);
        }
        for comp in 0..2 {
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][0], 0);
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][1], 0);
        }
        for comp in 0..2 {
            for node in 0..7 {
                enc.put_prob(tables::VP6_PDV_PCT[comp][node], 0);
            }
        }
        for comp in 0..2 {
            for node in 0..8 {
                enc.put_prob(tables::VP6_FDV_PCT[comp][node], 0);
            }
        }
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
            }
        }
        enc.put_bit(0);
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                    }
                }
            }
        }

        // --- Build the model + per-MB cache the same way the decoder does.
        //
        // Critical: the decoder's `model.coeff_dccv` / `coeff_ract` carry
        // state from the preceding keyframe (we set them to the
        // `def_prob_template = 0x80` baseline in `encode_keyframe`). On
        // inter frames `parse_coeff_models(..., key=false)` does NOT
        // copy `def_prob` through for unchanged nodes — the table just
        // retains its keyframe-time value. We mirror that here so that
        // `coeff_dcct` (and any per-MB coeff probs we read) match what
        // the decoder is using.
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);
        for pt in 0..2 {
            for node in 0..11 {
                model.coeff_dccv[pt][node] = 0x80;
            }
        }
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        model.coeff_ract[pt][ct][cg][node] = 0x80;
                    }
                }
            }
        }
        // Rebuild the linear-combination `coeff_dcct` table from
        // `coeff_dccv` exactly as `parse_coeff_models` does at the end.
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
        model.rebuild_mb_type_probs();

        // Mirror MacroblockInfo[] for predictor lookup on the encoder side.
        let mut mb_info: Vec<EncMbInfo> = vec![EncMbInfo::default(); mb_width * mb_height];

        // The decoder threads `vector_candidate_pos` through `parse_*`
        // calls; default-initialised to 0 in `BlockScratch::default`.
        let mut vector_candidate_pos: i32 = 0;
        // `prev_type` for the MB-type tree starts at InterNoVecPf, just
        // as the decoder primes `s->mb_type` before the per-MB loop.
        let mut prev_type = tables::Vp56Mb::InterNoVecPf;

        // -- Per-block DC-predictor mirror (matches `BlockScratch`'s
        // `left_block` / `above_blocks` / `prev_dc` for inter frames).
        // The decoder applies `add_predictors_dc(scratch, RefKind::Previous)`
        // after `parse_coeff` for every InterDelta/InterNoVec MB, so the
        // encoder must compute the same predictor (and update the same
        // running state) before quantising `coded_dc = new_dc - predictor`.
        let mut enc_left_block: [EncRefDc; 4] = [EncRefDc::default(); 4];
        let mut enc_above_blocks: Vec<EncRefDc> = vec![EncRefDc::default(); 4 * mb_width + 6];
        if 2 * mb_width + 2 < enc_above_blocks.len() {
            enc_above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < enc_above_blocks.len() {
            enc_above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        // `prev_dc[plane][ref_kind_index]` — luma=0, U=1, V=2; ref index
        // matches `mb::ref_kind_index` (Current=0, Previous=1, Golden=2).
        // Decoder seeds `prev_dc[1][Current]=128`, `prev_dc[2][Current]=128`
        // — Previous defaults stay at 0.
        let mut enc_prev_dc = [[0i16; 3]; 3];
        enc_prev_dc[1][0] = 128;
        enc_prev_dc[2][0] = 128;

        let dequant_dc = (tables::VP56_DC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;
        let dequant_ac = (tables::VP56_AC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;

        // Partition 2: per-MB DCT coefficients only.
        let mut enc2 = RangeEncoder::new();

        for mb_row in 0..mb_height {
            // Reset per-row left-block context (decoder's `start_row`).
            for b in &mut enc_left_block {
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
                // -- 1. Motion-search this MB against the previous frame.
                // `motion_search` returns quarter-pel `(qdx, qdy)` —
                // integer search seeded, then qpel-refined around the
                // integer winner with bilinear MC mirroring the decoder.
                let (q_dx, q_dy) = motion_search(
                    new_y, prev_y, width, height, mb_row, mb_col, search, self.qp,
                );

                // -- 2. Predictor + candidate state, mirroring the decoder.
                let (ctx_val, candidate0, candidate1, new_pos) = enc_vector_predictors(
                    &mb_info,
                    mb_width,
                    mb_height,
                    mb_row,
                    mb_col,
                    tables::RefFrame::Previous,
                    vector_candidate_pos,
                );
                vector_candidate_pos = new_pos;
                // `enc_vector_predictors` already returns the spec page
                // 28 Table 5 ctx (`2 - nb_pred`, mirroring the decoder's
                // `vector_predictors`), so we use it directly as the
                // MB-type context (0..=2).
                let ctx = ctx_val.clamp(0, 2) as usize;

                // -- 3. Choose mb_type. Picking InterNoVecPf when the best
                // MV is zero keeps the bitstream tight (no MV bits) and
                // matches the decoder's expectation for skip MBs.
                let want_mv = Mv {
                    x: q_dx as i16, // already in quarter-pel luma units
                    y: q_dy as i16,
                };
                let new_type = if want_mv.x == 0 && want_mv.y == 0 {
                    tables::Vp56Mb::InterNoVecPf
                } else {
                    tables::Vp56Mb::InterDeltaPf
                };

                // -- 4. Emit MB-type into partition 1.
                let stay_prob = model.mb_type[ctx][prev_type as usize][0];
                if new_type == prev_type {
                    enc.put_prob(stay_prob, 1);
                } else {
                    enc.put_prob(stay_prob, 0);
                    encode_pmbt_tree(&mut enc, &model.mb_type[ctx][prev_type as usize], new_type);
                }

                // -- 5. Emit MV delta into partition 1 if this MB has one.
                let stored_mv = if new_type == tables::Vp56Mb::InterDeltaPf {
                    let base = if vector_candidate_pos < 2 {
                        candidate0
                    } else {
                        Mv::default()
                    };
                    let delta_x = want_mv.x as i32 - base.x as i32;
                    let delta_y = want_mv.y as i32 - base.y as i32;
                    encode_mv_component(&mut enc, &model, 0, delta_x);
                    encode_mv_component(&mut enc, &model, 1, delta_y);
                    want_mv
                } else {
                    Mv::default()
                };
                let _ = candidate1;

                // -- 6. Update per-MB cache for downstream predictors.
                mb_info[mb_row * mb_width + mb_col] = EncMbInfo {
                    mb_type: new_type,
                    mv: stored_mv,
                };
                prev_type = new_type;

                // -- 7. Per-block residual encoding into partition 2.
                //
                // For each of the 6 blocks: build the integer-pel MC
                // prediction tile, compute the pixel residual, forward
                // DCT (residual mode — no `-128` subtraction), quantise,
                // run the RefKind::Previous DC predictor, and emit via
                // the same `emit_block_coefs` state machine the keyframe
                // path uses.
                for b in 0..6usize {
                    let pt = if b > 3 { 1 } else { 0 };
                    let plane_idx = tables::B2P[b] as usize;

                    // Coefficient context = sum of `not_null_dc` flags
                    // from the (left, above) DC neighbours. Same lookup
                    // as the keyframe path / decoder.
                    let coeff_ctx = enc_left_block[tables::B6_TO_4[b] as usize].not_null_dc
                        as usize
                        + enc_above_blocks[above_block_idx[b]].not_null_dc as usize;

                    // -- (a) Sample original pixels for this 8x8 block.
                    let mut orig_tile = [0i32; 64];
                    sample_block_tile(
                        b,
                        mb_row,
                        mb_col,
                        new_y,
                        new_u,
                        new_v,
                        width,
                        uv_stride,
                        &mut orig_tile,
                    );

                    // -- (b) Materialise the integer-pel MC prediction.
                    let mut mc_tile = [0i32; 64];
                    sample_mc_tile(
                        b,
                        mb_row,
                        mb_col,
                        prev_y,
                        prev_u,
                        prev_v,
                        width,
                        uv_stride,
                        height,
                        uv_h,
                        stored_mv,
                        &mut mc_tile,
                    );

                    // -- (c) Pixel residual.
                    let mut residual = [0i32; 64];
                    for i in 0..64 {
                        residual[i] = orig_tile[i] - mc_tile[i];
                    }

                    // -- (d) Forward DCT in residual mode (no -128 bias).
                    let mut coefs = [0i32; 64];
                    forward_dct8x8_residual(&residual, &mut coefs);

                    // -- (e) Quantise DC.
                    let new_dc = div_nearest(coefs[0], dequant_dc).clamp(-32768, 32767) as i16;

                    // -- (f) Compute the decoder's predictor_dc for this
                    // block under RefKind::Previous.
                    let lb = enc_left_block[tables::B6_TO_4[b] as usize];
                    let ab = enc_above_blocks[above_block_idx[b]];
                    let mut count = 0i32;
                    let mut pdc = 0i32;
                    if lb.ref_frame == RefKind::Previous {
                        pdc += lb.dc_coeff as i32;
                        count += 1;
                    }
                    if ab.ref_frame == RefKind::Previous {
                        pdc += ab.dc_coeff as i32;
                        count += 1;
                    }
                    let predictor: i32 = match count {
                        0 => enc_prev_dc[plane_idx][1] as i32, // 1 = Previous
                        2 => pdc / 2,
                        _ => pdc,
                    };
                    let coded_dc = (new_dc as i32 - predictor).clamp(-32768, 32767) as i16;

                    // -- (g) Quantise AC coefficients in coeff_idx order.
                    let mut ac_levels = [0i32; 64];
                    for coeff_idx in 1..64usize {
                        let pos = model.coeff_index_to_pos[coeff_idx] as usize;
                        let perm = tables::IDCT_SCANTABLE[pos] as usize;
                        ac_levels[coeff_idx] =
                            div_nearest(coefs[perm], dequant_ac).clamp(-2047, 2047);
                    }

                    let last_nz = find_last_nonzero(&ac_levels);

                    // -- (h) Emit the block coefficient stream.
                    emit_block_coefs(
                        &mut enc2, &model, pt, coeff_ctx, coded_dc, &ac_levels, last_nz,
                    );

                    // -- (i) Update per-block DC context with the
                    // reconstruction the decoder will land on. The
                    // decoder's `add_predictors_dc` stores `new_dc =
                    // coded_dc + predictor` back into both
                    // `left_block.dc_coeff` and `above_blocks[].dc_coeff`,
                    // and into `prev_dc[plane][Previous]`.
                    let new_dc_final = (coded_dc as i32 + predictor) as i16;
                    let has_nonzero_dc = coded_dc != 0;
                    let lb_idx = tables::B6_TO_4[b] as usize;
                    enc_left_block[lb_idx] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Previous,
                        dc_coeff: new_dc_final,
                    };
                    enc_above_blocks[above_block_idx[b]] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: RefKind::Previous,
                        dc_coeff: new_dc_final,
                    };
                    enc_prev_dc[plane_idx][1] = new_dc_final;
                }

                // Advance per-MB above-block indices the same way
                // `BlockScratch::advance_column` does.
                for y in 0..4 {
                    above_block_idx[y] += 2;
                }
                for uv in 4..6 {
                    above_block_idx[uv] += 1;
                }
            }
        }

        let p1 = enc.finish();
        let p2 = enc2.finish();
        out.extend_from_slice(&p1);
        // Spec-defined raw byte offset to partition 2 — see comment in
        // `encode_skip_frame` for the rationale.
        let buff2 = ((3 + p1.len()) as u32).min(0xFFFF);
        out[buff2_hi_idx] = ((buff2 >> 8) & 0xFF) as u8;
        out[buff2_lo_idx] = (buff2 & 0xFF) as u8;
        out.extend_from_slice(&p2);
        // Plain inter frames don't refresh golden — the
        // `golden_frame_flag` bit in the picture header is emitted as 0
        // above. Bump the cadence counter so a downstream
        // `encode_inter_frame_with_golden` call still measures the
        // refresh interval correctly.
        self.inter_frames_since_golden = self.inter_frames_since_golden.saturating_add(1);
        Ok(out)
    }

    /// Returns true when the next inter frame should set the
    /// `golden_frame_flag` bit on the picture header (spec §10 / page
    /// 28: inter-frame bool 1 immediately after the partition header).
    /// The flag tells the decoder to overwrite its `golden_frame` slot
    /// with the just-decoded reconstruction, so subsequent inter MBs
    /// that pick a golden reference (`InterNoVecGf` / `InterDeltaGf`
    /// /…) reference this updated plane rather than the keyframe-time
    /// golden.
    ///
    /// Trigger: `golden_refresh_period > 0 &&
    /// inter_frames_since_golden >= golden_refresh_period`.
    pub fn should_refresh_golden(&self) -> bool {
        self.golden_refresh_period > 0
            && self.inter_frames_since_golden >= self.golden_refresh_period
    }

    /// Inter-frame encoder with explicit `golden_*` reference planes.
    ///
    /// Mirror of [`Self::encode_inter_frame`] with two added behaviours:
    ///
    /// 1. **Golden-frame refresh.** When [`Self::should_refresh_golden`]
    ///    fires (cadence-driven by `golden_refresh_period`), the
    ///    picture-header `golden_frame_flag` bit is set to 1. The
    ///    decoder snaps the just-decoded frame into its `golden_frame`
    ///    slot — see `decoder.rs:422` — so the next call's
    ///    `golden_*` planes should reflect that updated reference. The
    ///    cadence counter resets at the same time.
    /// 2. **Per-MB golden-vs-previous selection.** For every MB the
    ///    encoder runs a `motion_search` against both `prev_*` and
    ///    `golden_*`, then picks the lower Lagrangian cost (SAD + λ *
    ///    mv_bits). The MB type is then one of {`InterNoVecPf`,
    ///    `InterDeltaPf`, `InterNoVecGf`, `InterDeltaGf`} accordingly.
    ///    Golden-ref MBs use `RefKind::Golden` for their DC predictor
    ///    state and contribute to the golden-ref MV-candidate pool the
    ///    decoder walks for subsequent golden-ref MBs.
    ///
    /// Picking the same `prev_*` for both arguments and a refresh
    /// period of 0 makes this method behave equivalently to
    /// [`Self::encode_inter_frame`] (zero MBs will pick golden because
    /// the SADs tie and the prev branch wins on the strict-less
    /// comparison; the flag stays 0).
    ///
    /// Spec refs:
    /// * Page 28 — picture-header layout, `golden_frame_flag` bit.
    /// * `REFERENCE_FRAME` table in `tables.rs`: `InterNoVecGf`,
    ///   `InterDeltaGf`, `InterV1Gf`, `InterV2Gf` all map to
    ///   `RefFrame::Golden`. Only `InterNoVecGf` + `InterDeltaGf` are
    ///   emitted by this encoder (single-MV-per-MB scope, same as
    ///   [`Self::encode_inter_frame`]).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_inter_frame_with_golden(
        &mut self,
        prev_y: &[u8],
        prev_u: &[u8],
        prev_v: &[u8],
        golden_y: &[u8],
        golden_u: &[u8],
        golden_v: &[u8],
        new_y: &[u8],
        new_u: &[u8],
        new_v: &[u8],
        width: usize,
        height: usize,
        search: i32,
    ) -> Result<EncodedFrame> {
        if !self.have_keyframe {
            return Err(Error::invalid(
                "VP6 encode: inter frame requires a preceding keyframe",
            ));
        }
        let mb_width = self.mb_width as usize;
        let mb_height = self.mb_height as usize;
        if width != mb_width * 16 || height != mb_height * 16 {
            return Err(Error::invalid(
                "VP6 encode: inter-frame dims must match the preceding keyframe",
            ));
        }
        let uv_stride = width / 2;
        let uv_h = height / 2;
        let need_y = width * height;
        let need_uv = uv_stride * uv_h;
        if prev_y.len() < need_y
            || new_y.len() < need_y
            || golden_y.len() < need_y
            || prev_u.len() < need_uv
            || prev_v.len() < need_uv
            || golden_u.len() < need_uv
            || golden_v.len() < need_uv
            || new_u.len() < need_uv
            || new_v.len() < need_uv
        {
            return Err(Error::invalid("VP6 encode: plane buffers too small"));
        }

        // Decide whether to flip the golden_frame_flag this frame.
        let refresh_golden = self.should_refresh_golden();

        // --- Fixed inter header (MultiStream=1, two partitions) ---------
        let mut out = Vec::<u8>::with_capacity(16 + mb_width * mb_height * 8);
        out.push(0x80 | ((self.qp << 1) & 0x7E) | 0x01);
        out.push(0);
        out.push(0);
        let buff2_hi_idx = out.len() - 2;
        let buff2_lo_idx = out.len() - 1;

        let mut enc = RangeEncoder::new();
        // golden_frame_flag — set when the cadence triggers. The
        // decoder reads `rac.get_bit()` at decoder.rs:218; when the bit
        // is 1, decoder.rs:422 overwrites the `golden_frame` slot with
        // the just-decoded reconstruction.
        enc.put_bit(if refresh_golden { 1 } else { 0 });
        // use_huffman = 0.
        enc.put_bit(0);

        // --- All probability-model "update" pass blocks (no updates) ----
        // Same shape as `encode_inter_frame`.
        for _ctx in 0..3 {
            enc.put_prob(174, 0);
            enc.put_prob(254, 0);
        }
        for comp in 0..2 {
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][0], 0);
            enc.put_prob(tables::VP6_SIG_DCT_PCT[comp][1], 0);
        }
        for comp in 0..2 {
            for node in 0..7 {
                enc.put_prob(tables::VP6_PDV_PCT[comp][node], 0);
            }
        }
        for comp in 0..2 {
            for node in 0..8 {
                enc.put_prob(tables::VP6_FDV_PCT[comp][node], 0);
            }
        }
        for pt in 0..2 {
            for node in 0..11 {
                enc.put_prob(tables::VP6_DCCV_PCT[pt][node], 0);
            }
        }
        enc.put_bit(0);
        for cg in 0..2 {
            for node in 0..14 {
                enc.put_prob(tables::VP6_RUNV_PCT[cg][node], 0);
            }
        }
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        enc.put_prob(tables::VP6_RACT_PCT[ct][pt][cg][node], 0);
                    }
                }
            }
        }

        // --- Build model + per-MB cache exactly as the decoder does. ----
        let mut model = Vp6Model::default();
        model.reset_defaults(false, self.sub_version);
        for pt in 0..2 {
            for node in 0..11 {
                model.coeff_dccv[pt][node] = 0x80;
            }
        }
        for ct in 0..3 {
            for pt in 0..2 {
                for cg in 0..6 {
                    for node in 0..11 {
                        model.coeff_ract[pt][ct][cg][node] = 0x80;
                    }
                }
            }
        }
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
        model.rebuild_mb_type_probs();

        let mut mb_info: Vec<EncMbInfo> = vec![EncMbInfo::default(); mb_width * mb_height];

        let mut vector_candidate_pos_pf: i32 = 0;
        let mut vector_candidate_pos_gf: i32 = 0;
        let mut prev_type = tables::Vp56Mb::InterNoVecPf;

        let mut enc_left_block: [EncRefDc; 4] = [EncRefDc::default(); 4];
        let mut enc_above_blocks: Vec<EncRefDc> = vec![EncRefDc::default(); 4 * mb_width + 6];
        if 2 * mb_width + 2 < enc_above_blocks.len() {
            enc_above_blocks[2 * mb_width + 2].ref_frame = RefKind::Current;
        }
        if 3 * mb_width + 4 < enc_above_blocks.len() {
            enc_above_blocks[3 * mb_width + 4].ref_frame = RefKind::Current;
        }
        let mut enc_prev_dc = [[0i16; 3]; 3];
        enc_prev_dc[1][0] = 128;
        enc_prev_dc[2][0] = 128;

        let dequant_dc = (tables::VP56_DC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;
        let dequant_ac = (tables::VP56_AC_DEQUANT[self.qp as usize & 0x3F] as i32) << 2;

        // Partition 2: per-MB DCT coefficients only.
        let mut enc2 = RangeEncoder::new();

        for mb_row in 0..mb_height {
            for b in &mut enc_left_block {
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
                // -- 1. Motion-search this MB against BOTH refs.
                let (q_dx_pf, q_dy_pf) = motion_search(
                    new_y, prev_y, width, height, mb_row, mb_col, search, self.qp,
                );
                let (q_dx_gf, q_dy_gf) = motion_search(
                    new_y, golden_y, width, height, mb_row, mb_col, search, self.qp,
                );

                // -- 2. Lagrangian cost for each ref (final SAD at the
                // chosen qpel offset + MV bits scaled by λ). The
                // motion_search return already minimises this internally
                // per ref; here we re-evaluate the final SAD so we can
                // compare across refs apples-to-apples.
                let mb_x = (mb_col * 16) as i32;
                let mb_y = (mb_row * 16) as i32;
                let lambda = self.qp as u64;
                let pf_sad =
                    sad16x16_qpel(new_y, prev_y, width, height, mb_x, mb_y, q_dx_pf, q_dy_pf);
                let pf_bits = mv_bit_cost_estimate(q_dx_pf, q_dy_pf);
                let pf_cost = pf_sad.saturating_add(lambda.saturating_mul(pf_bits));
                let gf_sad =
                    sad16x16_qpel(new_y, golden_y, width, height, mb_x, mb_y, q_dx_gf, q_dy_gf);
                let gf_bits = mv_bit_cost_estimate(q_dx_gf, q_dy_gf);
                let gf_cost = gf_sad.saturating_add(lambda.saturating_mul(gf_bits));

                // Pick golden when it strictly beats prev — ties go to
                // prev so we never pay the MB-type-tree extra bits for a
                // noise-level "improvement". The PMBT tree puts
                // `InterNoVecPf` at depth 1 and `InterNoVecGf` at depth
                // ~3, so prev is intrinsically cheaper to encode.
                let use_golden = gf_cost < pf_cost;
                let (q_dx, q_dy) = if use_golden {
                    (q_dx_gf, q_dy_gf)
                } else {
                    (q_dx_pf, q_dy_pf)
                };
                let ref_kind = if use_golden {
                    RefKind::Golden
                } else {
                    RefKind::Previous
                };

                // -- 3. MV-candidate predictor for the chosen ref.
                let want_ref_frame = if use_golden {
                    tables::RefFrame::Golden
                } else {
                    tables::RefFrame::Previous
                };
                let initial_pos = if use_golden {
                    vector_candidate_pos_gf
                } else {
                    vector_candidate_pos_pf
                };
                let (ctx_val, candidate0, candidate1, new_pos) = enc_vector_predictors(
                    &mb_info,
                    mb_width,
                    mb_height,
                    mb_row,
                    mb_col,
                    want_ref_frame,
                    initial_pos,
                );
                if use_golden {
                    vector_candidate_pos_gf = new_pos;
                } else {
                    vector_candidate_pos_pf = new_pos;
                }
                let ctx = ctx_val.clamp(0, 2) as usize;

                // -- 4. Choose mb_type by ref kind + zero/non-zero MV.
                let want_mv = Mv {
                    x: q_dx as i16,
                    y: q_dy as i16,
                };
                let new_type = match (use_golden, want_mv.x == 0 && want_mv.y == 0) {
                    (false, true) => tables::Vp56Mb::InterNoVecPf,
                    (false, false) => tables::Vp56Mb::InterDeltaPf,
                    (true, true) => tables::Vp56Mb::InterNoVecGf,
                    (true, false) => tables::Vp56Mb::InterDeltaGf,
                };

                // -- 5. Emit MB-type into partition 1 via PMBT tree walk.
                let stay_prob = model.mb_type[ctx][prev_type as usize][0];
                if new_type == prev_type {
                    enc.put_prob(stay_prob, 1);
                } else {
                    enc.put_prob(stay_prob, 0);
                    encode_pmbt_tree(&mut enc, &model.mb_type[ctx][prev_type as usize], new_type);
                }

                // -- 6. Emit MV delta into partition 1 if this MB has one.
                let stored_mv = if matches!(
                    new_type,
                    tables::Vp56Mb::InterDeltaPf | tables::Vp56Mb::InterDeltaGf
                ) {
                    let candidate_pos = if use_golden {
                        vector_candidate_pos_gf
                    } else {
                        vector_candidate_pos_pf
                    };
                    let base = if candidate_pos < 2 {
                        candidate0
                    } else {
                        Mv::default()
                    };
                    let delta_x = want_mv.x as i32 - base.x as i32;
                    let delta_y = want_mv.y as i32 - base.y as i32;
                    encode_mv_component(&mut enc, &model, 0, delta_x);
                    encode_mv_component(&mut enc, &model, 1, delta_y);
                    want_mv
                } else {
                    Mv::default()
                };
                let _ = candidate1;

                // -- 7. Update per-MB cache.
                mb_info[mb_row * mb_width + mb_col] = EncMbInfo {
                    mb_type: new_type,
                    mv: stored_mv,
                };
                prev_type = new_type;

                // -- 8. Residual encoding into partition 2.
                let ref_dc_idx = match ref_kind {
                    RefKind::Previous => 1usize,
                    RefKind::Golden => 2usize,
                    _ => 1usize,
                };
                for b in 0..6usize {
                    let pt = if b > 3 { 1 } else { 0 };
                    let plane_idx = tables::B2P[b] as usize;
                    let coeff_ctx = enc_left_block[tables::B6_TO_4[b] as usize].not_null_dc
                        as usize
                        + enc_above_blocks[above_block_idx[b]].not_null_dc as usize;

                    let mut orig_tile = [0i32; 64];
                    sample_block_tile(
                        b,
                        mb_row,
                        mb_col,
                        new_y,
                        new_u,
                        new_v,
                        width,
                        uv_stride,
                        &mut orig_tile,
                    );

                    // Sample from prev or golden depending on ref.
                    let (rp_y, rp_u, rp_v): (&[u8], &[u8], &[u8]) = if use_golden {
                        (golden_y, golden_u, golden_v)
                    } else {
                        (prev_y, prev_u, prev_v)
                    };
                    let mut mc_tile = [0i32; 64];
                    sample_mc_tile(
                        b,
                        mb_row,
                        mb_col,
                        rp_y,
                        rp_u,
                        rp_v,
                        width,
                        uv_stride,
                        height,
                        uv_h,
                        stored_mv,
                        &mut mc_tile,
                    );

                    let mut residual = [0i32; 64];
                    for i in 0..64 {
                        residual[i] = orig_tile[i] - mc_tile[i];
                    }

                    let mut coefs = [0i32; 64];
                    forward_dct8x8_residual(&residual, &mut coefs);

                    let new_dc = div_nearest(coefs[0], dequant_dc).clamp(-32768, 32767) as i16;

                    // DC predictor — neighbour contributes only when it
                    // matches the current MB's ref kind.
                    let lb = enc_left_block[tables::B6_TO_4[b] as usize];
                    let ab = enc_above_blocks[above_block_idx[b]];
                    let mut count = 0i32;
                    let mut pdc = 0i32;
                    if lb.ref_frame == ref_kind {
                        pdc += lb.dc_coeff as i32;
                        count += 1;
                    }
                    if ab.ref_frame == ref_kind {
                        pdc += ab.dc_coeff as i32;
                        count += 1;
                    }
                    let predictor: i32 = match count {
                        0 => enc_prev_dc[plane_idx][ref_dc_idx] as i32,
                        2 => pdc / 2,
                        _ => pdc,
                    };
                    let coded_dc = (new_dc as i32 - predictor).clamp(-32768, 32767) as i16;

                    let mut ac_levels = [0i32; 64];
                    for coeff_idx in 1..64usize {
                        let pos = model.coeff_index_to_pos[coeff_idx] as usize;
                        let perm = tables::IDCT_SCANTABLE[pos] as usize;
                        ac_levels[coeff_idx] =
                            div_nearest(coefs[perm], dequant_ac).clamp(-2047, 2047);
                    }
                    let last_nz = find_last_nonzero(&ac_levels);

                    emit_block_coefs(
                        &mut enc2, &model, pt, coeff_ctx, coded_dc, &ac_levels, last_nz,
                    );

                    let new_dc_final = (coded_dc as i32 + predictor) as i16;
                    let has_nonzero_dc = coded_dc != 0;
                    let lb_idx = tables::B6_TO_4[b] as usize;
                    enc_left_block[lb_idx] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: ref_kind,
                        dc_coeff: new_dc_final,
                    };
                    enc_above_blocks[above_block_idx[b]] = EncRefDc {
                        not_null_dc: has_nonzero_dc,
                        ref_frame: ref_kind,
                        dc_coeff: new_dc_final,
                    };
                    enc_prev_dc[plane_idx][ref_dc_idx] = new_dc_final;
                }

                for y in 0..4 {
                    above_block_idx[y] += 2;
                }
                for uv in 4..6 {
                    above_block_idx[uv] += 1;
                }
            }
        }

        let p1 = enc.finish();
        let p2 = enc2.finish();
        out.extend_from_slice(&p1);
        let buff2 = ((3 + p1.len()) as u32).min(0xFFFF);
        out[buff2_hi_idx] = ((buff2 >> 8) & 0xFF) as u8;
        out[buff2_lo_idx] = (buff2 & 0xFF) as u8;
        out.extend_from_slice(&p2);

        // Cadence bookkeeping. A refresh frame counts as the "0th" inter
        // since the new golden, so the next frame is "1 since golden" —
        // mirror the keyframe path's behaviour.
        if refresh_golden {
            self.inter_frames_since_golden = 1;
        } else {
            self.inter_frames_since_golden = self.inter_frames_since_golden.saturating_add(1);
        }
        Ok(out)
    }

    /// Current encoder dims in macroblocks — exposed for tests + diagnostics.
    pub fn dims_mb(&self) -> (u16, u16) {
        (self.mb_width, self.mb_height)
    }

    /// Inter-frames-since-last-golden-refresh counter — exposed for
    /// tests + diagnostics. Reset to 0 by `encode_keyframe` and to 1 by
    /// the refresh path of `encode_inter_frame_with_golden`.
    pub fn inter_frames_since_golden(&self) -> u32 {
        self.inter_frames_since_golden
    }
}

/// Encoder-side mirror of [`crate::mb::MacroblockInfo`]. Kept private
/// so the encoder doesn't leak its scratch into the public API.
#[derive(Clone, Copy, Debug)]
struct EncMbInfo {
    mb_type: tables::Vp56Mb,
    mv: Mv,
}

impl Default for EncMbInfo {
    fn default() -> Self {
        Self {
            mb_type: tables::Vp56Mb::Intra,
            mv: Mv::default(),
        }
    }
}

/// Integer-pel SAD search around `(0, 0)` within `±search` luma pixels,
/// followed by a quarter-pel refine around the integer best. Returns
/// `(qdx, qdy)` — the **quarter-pel** offset (luma units) that
/// minimises a Lagrangian cost (SAD plus MV-bit cost) between the
/// current MB and the corresponding candidate MB in `prev`.
///
/// Quarter-pel search algorithm (r25):
///
/// 1. Run the integer-pel SAD search (existing behaviour) to seed
///    `(int_dx, int_dy)`.
/// 2. Around the integer winner, evaluate every quarter-pel offset
///    `(qx, qy) ∈ {-3..=3}^2` (a 7x7 grid covering the qpel positions
///    immediately around the integer winner without overlapping the
///    next integer cell). Each candidate's cost is
///    `SAD(MC(qpel)) + λ * mv_bits` where `MC(qpel)` is computed with
///    the decoder's bilinear sub-pel filter (mirror of
///    `mb::render_mb_inter`'s `put_h264_chroma8` path) and `λ` is a
///    QP-dependent weighting tuned so the rate cost only matters when
///    SAD is already tied (we want sub-pel precision when it
///    measurably helps, but don't pay rate for noise-level wins).
/// 3. Return the best qpel offset.
///
/// Integer search positions are clamped so the candidate MB stays
/// fully inside the previous frame; the qpel refine clamps the qpel
/// search bounds to the same window so we never reach past the integer
/// search budget by more than ±1 integer pel (the bilinear filter taps
/// pull from the next integer cell).
///
/// Spec ref: `vp6_format.pdf` §17.2 (Half / Quarter Pixel Aligned
/// Vectors) — luma MV `mv.x` in quarter-pel units, integer part is
/// `mv.x / 4`, sub-pel phase `mv.x & 3` (mapped to 8-step phases via
/// `* 2` in the decoder's `render_mb_inter`).
fn motion_search(
    cur: &[u8],
    prev: &[u8],
    width: usize,
    height: usize,
    mb_row: usize,
    mb_col: usize,
    search: i32,
    qp: u8,
) -> (i32, i32) {
    let mb_x = (mb_col * 16) as i32;
    let mb_y = (mb_row * 16) as i32;
    let max_dx = ((width as i32) - 16 - mb_x).min(search);
    let max_dy = ((height as i32) - 16 - mb_y).min(search);
    let min_dx = (-mb_x).max(-search);
    let min_dy = (-mb_y).max(-search);

    // -- Stage 1: integer-pel search, full window.
    let mut best_int_dx = 0i32;
    let mut best_int_dy = 0i32;
    let mut best_int_sad = sad16x16(cur, prev, width, mb_x, mb_y, 0, 0);
    for dy in min_dy..=max_dy {
        for dx in min_dx..=max_dx {
            if dx == 0 && dy == 0 {
                continue;
            }
            let s = sad16x16(cur, prev, width, mb_x, mb_y, dx, dy);
            if s < best_int_sad {
                best_int_sad = s;
                best_int_dx = dx;
                best_int_dy = dy;
            }
        }
    }

    // -- Stage 2: quarter-pel refine around the integer winner. The
    // bilinear filter taps reach to `+1` integer pel in both axes, so
    // clamp the qpel search so `(int + 1)` stays inside `[min, max]`.
    // We also need 1 pel of left/top headroom for the integer base.
    let pw = width as i32;
    let ph = height as i32;
    let min_q_x = (-mb_x * 4).max(-search * 4 - 4);
    let max_q_x = ((pw - 17 - mb_x) * 4).min(search * 4 + 4);
    let min_q_y = (-mb_y * 4).max(-search * 4 - 4);
    let max_q_y = ((ph - 17 - mb_y) * 4).min(search * 4 + 4);

    // Lagrangian λ: roughly QP-proportional, tuned empirically so a
    // sub-pel win of <1 SAD/pixel (i.e. mostly noise) doesn't outweigh
    // the extra MV bits. QP 0..=63 maps to λ ~ 0..=63.
    let lambda = qp as u64;

    let int_qx = best_int_dx * 4;
    let int_qy = best_int_dy * 4;
    let mut best_qx = int_qx;
    let mut best_qy = int_qy;
    // Seed the cost from the integer winner — the qpel search has to
    // beat it including its MV-bit cost.
    let int_bits = mv_bit_cost_estimate(int_qx, int_qy);
    let mut best_cost = best_int_sad.saturating_add(lambda.saturating_mul(int_bits));

    for dqy in -3..=3i32 {
        for dqx in -3..=3i32 {
            if dqx == 0 && dqy == 0 {
                continue;
            }
            let qx = int_qx + dqx;
            let qy = int_qy + dqy;
            if qx < min_q_x || qx > max_q_x || qy < min_q_y || qy > max_q_y {
                continue;
            }
            let sad = sad16x16_qpel(cur, prev, width, height, mb_x, mb_y, qx, qy);
            let bits = mv_bit_cost_estimate(qx, qy);
            let cost = sad.saturating_add(lambda.saturating_mul(bits));
            if cost < best_cost {
                best_cost = cost;
                best_qx = qx;
                best_qy = qy;
            }
        }
    }

    (best_qx, best_qy)
}

/// Sum of absolute differences over a 16×16 luma MB at integer-pel
/// offset `(dx, dy)` against the previous frame.
fn sad16x16(cur: &[u8], prev: &[u8], stride: usize, mb_x: i32, mb_y: i32, dx: i32, dy: i32) -> u64 {
    let mut acc = 0u64;
    for r in 0..16i32 {
        let cy = (mb_y + r) as usize;
        let py = (mb_y + r + dy) as usize;
        for c in 0..16i32 {
            let cx = (mb_x + c) as usize;
            let px = (mb_x + c + dx) as usize;
            let a = cur[cy * stride + cx] as i32;
            let b = prev[py * stride + px] as i32;
            acc += (a - b).unsigned_abs() as u64;
        }
    }
    acc
}

/// SAD over a 16×16 luma MB against a **quarter-pel** MV `(qx, qy)`
/// (luma quarter-pel units). Materialises the bilinear MC tile via
/// `bilinear_luma_sample` per pixel and accumulates absolute
/// differences. Sample positions are clamped to `[0, w-1] × [0, h-1]`,
/// matching the decoder's edge-clamp inside its 12x12 reference
/// scratch.
#[allow(clippy::too_many_arguments)]
fn sad16x16_qpel(
    cur: &[u8],
    prev: &[u8],
    stride: usize,
    height: usize,
    mb_x: i32,
    mb_y: i32,
    qx: i32,
    qy: i32,
) -> u64 {
    // VP6 luma sub-pel: integer = qx/4 (truncated toward zero), phase
    // bits = qx & 3 mapped to 8-step phases via `* 2`. Mirror of
    // `mb::render_mb_inter` for plane = Y.
    let dx = qx / 4;
    let dy = qy / 4;
    let x8 = (qx & 3) * 2;
    let y8 = (qy & 3) * 2;
    let mut acc = 0u64;
    for r in 0..16i32 {
        let cy = (mb_y + r) as usize;
        for c in 0..16i32 {
            let cx = (mb_x + c) as usize;
            let mc =
                bilinear_luma_sample(prev, stride, height, mb_x + dx + c, mb_y + dy + r, x8, y8);
            let a = cur[cy * stride + cx] as i32;
            acc += (a - mc).unsigned_abs() as u64;
        }
    }
    acc
}

/// Bilinear luma sample at `(base_x + frac_x/8, base_y + frac_y/8)`
/// against `prev`. Mirrors the integer-pel-then-bilinear path the
/// decoder uses (`put_h264_chroma8` with sub-pel offsets `x8`, `y8` in
/// `[0, 8)`). Sample positions are clamped to plane bounds.
#[inline]
fn bilinear_luma_sample(
    prev: &[u8],
    stride: usize,
    height: usize,
    base_x: i32,
    base_y: i32,
    frac_x: i32,
    frac_y: i32,
) -> i32 {
    let pw = stride as i32;
    let ph = height as i32;
    let sx0 = base_x.clamp(0, pw - 1);
    let sy0 = base_y.clamp(0, ph - 1);
    let sx1 = (base_x + 1).clamp(0, pw - 1);
    let sy1 = (base_y + 1).clamp(0, ph - 1);
    let p00 = prev[sy0 as usize * stride + sx0 as usize] as i32;
    let p01 = prev[sy0 as usize * stride + sx1 as usize] as i32;
    let p10 = prev[sy1 as usize * stride + sx0 as usize] as i32;
    let p11 = prev[sy1 as usize * stride + sx1 as usize] as i32;
    // Bilinear weights from `dsp::put_h264_chroma8`: `a, b, c, d` in
    // [0, 64], rounding by `(v + 32) >> 6`.
    let a = (8 - frac_x) * (8 - frac_y);
    let b = frac_x * (8 - frac_y);
    let c = (8 - frac_x) * frac_y;
    let d = frac_x * frac_y;
    let v = a * p00 + b * p01 + c * p10 + d * p11;
    (v + 32) >> 6
}

/// Rough estimate of the bit cost of encoding a (qpel) MV component
/// pair `(qx, qy)`. We use `bits ≈ ceil(log2(|qx| + |qy| + 1)) + 2`
/// as a crude monotonic proxy for the decoder's MV-coding tree depth
/// (PVA short tree = 1..3 bool bits + sign; FDV long tree = ~9 bool
/// bits + sign). This isn't an exact bit count but is good enough for
/// a Lagrangian tie-breaker between qpel candidates of comparable SAD.
#[inline]
fn mv_bit_cost_estimate(qx: i32, qy: i32) -> u64 {
    let mag = qx.unsigned_abs() as u64 + qy.unsigned_abs() as u64;
    // Cheap log2: 32 - leading_zeros of (mag | 1).
    let log2 = 64u64 - (mag | 1).leading_zeros() as u64;
    log2 + 2
}

/// Mirror of the decoder's `vector_predictors`. Returns
/// `(ctx, candidate0, candidate1, new_pos)` where `ctx` is the spec
/// page 28 Table 5 mapping `(0 cands -> 2, 1 cand -> 1, 2+ cands -> 0)`,
/// suitable for indexing `model.mb_type[ctx][...]` directly.
fn enc_vector_predictors(
    mb_info: &[EncMbInfo],
    mb_width: usize,
    mb_height: usize,
    row: usize,
    col: usize,
    ref_frame: tables::RefFrame,
    initial_pos: i32,
) -> (i32, Mv, Mv, i32) {
    let mut nb = 0i32;
    let mut cand = [Mv::default(); 2];
    let mut new_pos = initial_pos;
    for (pos, d) in tables::VP56_CANDIDATE_PREDICTOR_POS.iter().enumerate() {
        let nc = col as i32 + d[0] as i32;
        let nr = row as i32 + d[1] as i32;
        if nc < 0 || nc >= mb_width as i32 || nr < 0 || nr >= mb_height as i32 {
            continue;
        }
        let mb = mb_info[nr as usize * mb_width + nc as usize];
        if tables::REFERENCE_FRAME[mb.mb_type as usize] != ref_frame {
            continue;
        }
        if (mb.mv.x == cand[0].x && mb.mv.y == cand[0].y) || (mb.mv.x == 0 && mb.mv.y == 0) {
            continue;
        }
        cand[nb as usize] = mb.mv;
        nb += 1;
        if nb > 1 {
            // Third hit observed: matches spec "Nearest & Near MVs both
            // exist" (ctx 0). Mirror the decoder's clamping.
            nb = 2;
            break;
        }
        new_pos = pos as i32;
    }
    // Spec page 28 Table 5: 0 cands -> ctx 2, 1 cand -> ctx 1, 2+ -> ctx 0.
    (2 - nb, cand[0], cand[1], new_pos)
}

/// Encode a single MB-type leaf via the [`crate::tables::PMBT_TREE`].
/// Mirrors the decoder's `rac.get_tree(PMBT_TREE, mb_type_model)` walk.
fn encode_pmbt_tree(enc: &mut RangeEncoder, probs: &[u8; 10], target: tables::Vp56Mb) {
    // Walk the static PMBT tree to find the bit sequence that lands on
    // `target` (the leaf whose value matches `target as i32`).
    let target_val = target as i32;
    let mut bits: Vec<(u8, u8)> = Vec::with_capacity(8);
    if !walk_tree(tables::PMBT_TREE, 0, target_val, probs, &mut bits) {
        // Should never happen — PMBT tree covers all 10 MB types.
        return;
    }
    for (prob, bit) in bits {
        enc.put_prob(prob, bit);
    }
}

/// Recursive tree walk: append the (`prob`, `bit`) pairs that reach the
/// leaf with value `target_val`. Returns `true` on success.
fn walk_tree(
    tree: &[crate::range_coder::Vp56Tree],
    idx: usize,
    target_val: i32,
    probs: &[u8],
    out: &mut Vec<(u8, u8)>,
) -> bool {
    let node = tree[idx];
    if node.val <= 0 {
        return (-node.val) as i32 == target_val;
    }
    let prob = probs[node.prob_idx as usize];
    // Try bit = 0 (advance to idx+1).
    out.push((prob, 0));
    if walk_tree(tree, idx + 1, target_val, probs, out) {
        return true;
    }
    out.pop();
    // Try bit = 1 (advance by `node.val`).
    out.push((prob, 1));
    let next = idx.wrapping_add(node.val as usize);
    if walk_tree(tree, next, target_val, probs, out) {
        return true;
    }
    out.pop();
    false
}

/// Encode one MV-component delta exactly as the decoder reads it in
/// `parse_vector_adjustment`. `comp` selects the x (0) or y (1) model
/// slice. `delta` is the signed quarter-pel offset.
fn encode_mv_component(enc: &mut RangeEncoder, model: &Vp6Model, comp: usize, delta: i32) {
    let abs_d = delta.unsigned_abs() as i32;
    let abs_d = abs_d.min(127); // spec: max magnitude 127 quarter-pel units

    if abs_d < 8 {
        // Short path: vector_dct probe = 0, then walk PVA_TREE.
        enc.put_prob(model.vector_dct[comp], 0);
        encode_pva_tree(enc, &model.vector_pdv[comp], abs_d);
    } else {
        // Long path: vector_dct probe = 1, then 8 magnitude bits with
        // bit 3 implicit when bits 4..7 are all zero.
        enc.put_prob(model.vector_dct[comp], 1);
        // Bits 0, 1, 2.
        enc.put_prob(model.vector_fdv[comp][0], ((abs_d >> 0) & 1) as u8);
        enc.put_prob(model.vector_fdv[comp][1], ((abs_d >> 1) & 1) as u8);
        enc.put_prob(model.vector_fdv[comp][2], ((abs_d >> 2) & 1) as u8);
        // Bits 7, 6, 5, 4 (in that order).
        enc.put_prob(model.vector_fdv[comp][7], ((abs_d >> 7) & 1) as u8);
        enc.put_prob(model.vector_fdv[comp][6], ((abs_d >> 6) & 1) as u8);
        enc.put_prob(model.vector_fdv[comp][5], ((abs_d >> 5) & 1) as u8);
        enc.put_prob(model.vector_fdv[comp][4], ((abs_d >> 4) & 1) as u8);
        // Bit 3: if bits 4..7 are non-zero, send it; otherwise it's
        // implicit (decoder forces it to 1, so abs_d's bit 3 must be 1
        // to be representable — guaranteed by abs_d ≥ 8).
        if abs_d & 0xF0 != 0 {
            enc.put_prob(model.vector_fdv[comp][3], ((abs_d >> 3) & 1) as u8);
        }
    }

    // Sign bit: only emitted when delta != 0.
    if abs_d != 0 {
        let sign_bit = if delta < 0 { 1u8 } else { 0u8 };
        enc.put_prob(model.vector_sig[comp], sign_bit);
    }
}

/// Walk the [`crate::tables::PVA_TREE`] to emit a 0..=7 short-MV value.
fn encode_pva_tree(enc: &mut RangeEncoder, probs: &[u8; 7], target: i32) {
    let mut bits: Vec<(u8, u8)> = Vec::with_capacity(4);
    if walk_tree(tables::PVA_TREE, 0, target, probs, &mut bits) {
        for (p, b) in bits {
            enc.put_prob(p, b);
        }
    }
}

/// Sample the 8x8 block pixel tile from the right plane at block `b` of
/// macroblock `(mb_row, mb_col)`. Luma blocks 0..=3 pull from the Y
/// plane; blocks 4/5 pull from U/V respectively.
fn sample_block_tile(
    b: usize,
    mb_row: usize,
    mb_col: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    y_stride: usize,
    uv_stride: usize,
    out: &mut [i32; 64],
) {
    let (plane, stride, x0, y0) = match b {
        0 => (y, y_stride, mb_col * 16, mb_row * 16),
        1 => (y, y_stride, mb_col * 16 + 8, mb_row * 16),
        2 => (y, y_stride, mb_col * 16, mb_row * 16 + 8),
        3 => (y, y_stride, mb_col * 16 + 8, mb_row * 16 + 8),
        4 => (u, uv_stride, mb_col * 8, mb_row * 8),
        5 => (v, uv_stride, mb_col * 8, mb_row * 8),
        _ => unreachable!(),
    };
    for r in 0..8usize {
        for c in 0..8usize {
            out[r * 8 + c] = plane[(y0 + r) * stride + (x0 + c)] as i32;
        }
    }
}

/// Forward 2D DCT-II with scaling matched to the decoder's IDCT.
///
/// Standard orthonormal DCT-II puts the DC at `8 * (V - 128)` for a flat
/// V-tile; our decoder's IDCT expects `~32 * (V - 128)` (the full
/// two-stage chain multiplies by `C4S4/2^16 * C4S4/2^20 ≈ 2^-32.05`,
/// which inverts to a gain of ~32). We therefore apply the usual 2D
/// DCT-II formula without the `1/4` factor so the per-block scale is
/// exactly 4x the orthonormal normalisation.
///
/// Output `out[u*8+v] = F[u, v]`: raster layout with `u` as vertical
/// frequency and `v` as horizontal frequency — the exact convention
/// the VP6 spec uses for its `default_dequant_table` and IDCT stages.
fn forward_dct8x8(tile: &[i32; 64], out: &mut [i32; 64]) {
    // Precomputed cos((2k+1)*n*pi/16) for k,n in 0..8.
    let mut cos = [[0.0f64; 8]; 8];
    for k in 0..8 {
        for n in 0..8 {
            let angle = ((2 * k + 1) * n) as f64 * std::f64::consts::PI / 16.0;
            cos[k][n] = angle.cos();
        }
    }

    // Subtract 128 from pixels.
    let mut pm = [0.0f64; 64];
    for i in 0..64 {
        pm[i] = (tile[i] - 128) as f64;
    }

    let inv_sqrt2 = 1.0f64 / std::f64::consts::SQRT_2;

    // Row-wise 1D DCT: temp[r][u] = sum_c pm[r][c] * cos((2c+1)*u*pi/16).
    let mut temp = [0.0f64; 64];
    for r in 0..8 {
        for u in 0..8 {
            let mut s = 0.0f64;
            for c in 0..8 {
                s += pm[r * 8 + c] * cos[c][u];
            }
            temp[r * 8 + u] = s;
        }
    }

    // Column-wise 1D DCT, stored in natural 2D-DCT-II raster layout
    // `out[u*8+v] = F[u][v]` — `u` is the vertical frequency index
    // (iterated in the decoder's column IDCT pass) and `v` is the
    // horizontal frequency index (iterated in the row pass). This
    // matches the VP6 spec (section 12.1 + section 16): scan index 1
    // maps via `default_dequant_table` to raw position 1 = F[0,1] (the
    // first horizontal AC), scan index 2 to raw position 8 = F[1,0]
    // (the first vertical AC), and so on.
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0.0f64;
            for r in 0..8 {
                s += temp[r * 8 + v] * cos[r][u];
            }
            let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
            let cv = if v == 0 { inv_sqrt2 } else { 1.0 };
            out[u * 8 + v] = (s * cu * cv).round() as i32;
        }
    }
}

/// Forward DCT for residual encoding — same shape as
/// [`forward_dct8x8`] but without the `-128` pixel bias. The MC
/// prediction is already centred near the original pixel range, so the
/// per-pixel residual is a small signed value around 0; subtracting 128
/// (as the keyframe path does to undo the encoder's `+128` reconstruction
/// bias) would skew the DCT input by an unrelated constant.
fn forward_dct8x8_residual(tile: &[i32; 64], out: &mut [i32; 64]) {
    let mut cos = [[0.0f64; 8]; 8];
    for k in 0..8 {
        for n in 0..8 {
            let angle = ((2 * k + 1) * n) as f64 * std::f64::consts::PI / 16.0;
            cos[k][n] = angle.cos();
        }
    }

    let mut pm = [0.0f64; 64];
    for i in 0..64 {
        pm[i] = tile[i] as f64;
    }

    let inv_sqrt2 = 1.0f64 / std::f64::consts::SQRT_2;

    let mut temp = [0.0f64; 64];
    for r in 0..8 {
        for u in 0..8 {
            let mut s = 0.0f64;
            for c in 0..8 {
                s += pm[r * 8 + c] * cos[c][u];
            }
            temp[r * 8 + u] = s;
        }
    }

    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0.0f64;
            for r in 0..8 {
                s += temp[r * 8 + v] * cos[r][u];
            }
            let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
            let cv = if v == 0 { inv_sqrt2 } else { 1.0 };
            out[u * 8 + v] = (s * cu * cv).round() as i32;
        }
    }
}

/// Sample an 8x8 MC prediction tile from the previous frame, mirroring
/// the bilinear branch of `mb::render_mb_inter`. For block `b`:
/// * Luma blocks 0..=3: pull from `prev_y` at `(base_x + dx, base_y +
///   dy)` where `dx = mv.x / 4`, `dy = mv.y / 4` (integer-pel from
///   quarter-pel units). Sub-pel phase = `(mv.x & 3) * 2` (0..=6,
///   step 2) — when non-zero we apply the H.264-chroma bilinear filter
///   the decoder uses (`use_bicubic_luma == false` branch in
///   `render_mb_inter`).
/// * Chroma blocks 4/5: pull from `prev_u` / `prev_v` at `(base_x + dx,
///   base_y + dy)` where `dx = mv.x / 8`, `dy = mv.y / 8`. Sub-pel
///   phase = `mv.x & 7` (0..=7, step 1).
///
/// Sample positions are clamped to the plane's `[0, w-1] x [0, h-1]`,
/// matching the decoder's edge-clamp inside the 12x12 reference scratch.
#[allow(clippy::too_many_arguments)]
fn sample_mc_tile(
    b: usize,
    mb_row: usize,
    mb_col: usize,
    prev_y: &[u8],
    prev_u: &[u8],
    prev_v: &[u8],
    y_stride: usize,
    uv_stride: usize,
    y_h: usize,
    uv_h: usize,
    mv: Mv,
    out: &mut [i32; 64],
) {
    let coord_div = tables::VP6_COORD_DIV[b] as i32;
    let dx = mv.x as i32 / coord_div;
    let dy = mv.y as i32 / coord_div;
    let mask = coord_div - 1;
    let is_luma = b < 4;
    let phase_scale = if is_luma { 2 } else { 1 };
    let x8 = (mv.x as i32 & mask) * phase_scale;
    let y8 = (mv.y as i32 & mask) * phase_scale;

    let (plane, stride, plane_h, base_x, base_y): (&[u8], usize, usize, i32, i32) = match b {
        0 => (
            prev_y,
            y_stride,
            y_h,
            (mb_col * 16) as i32,
            (mb_row * 16) as i32,
        ),
        1 => (
            prev_y,
            y_stride,
            y_h,
            (mb_col * 16 + 8) as i32,
            (mb_row * 16) as i32,
        ),
        2 => (
            prev_y,
            y_stride,
            y_h,
            (mb_col * 16) as i32,
            (mb_row * 16 + 8) as i32,
        ),
        3 => (
            prev_y,
            y_stride,
            y_h,
            (mb_col * 16 + 8) as i32,
            (mb_row * 16 + 8) as i32,
        ),
        4 => (
            prev_u,
            uv_stride,
            uv_h,
            (mb_col * 8) as i32,
            (mb_row * 8) as i32,
        ),
        5 => (
            prev_v,
            uv_stride,
            uv_h,
            (mb_col * 8) as i32,
            (mb_row * 8) as i32,
        ),
        _ => unreachable!(),
    };

    let pw = stride as i32;
    let ph = plane_h as i32;
    if x8 == 0 && y8 == 0 {
        // Integer-pel path — direct copy with edge clamp.
        for r in 0..8usize {
            for c in 0..8usize {
                let sx = (base_x + dx + c as i32).clamp(0, pw - 1);
                let sy = (base_y + dy + r as i32).clamp(0, ph - 1);
                out[r * 8 + c] = plane[sy as usize * stride + sx as usize] as i32;
            }
        }
    } else {
        // Sub-pel path — bilinear filter mirroring `put_h264_chroma8`.
        // Sample positions are clamped to plane bounds, matching the
        // decoder's 12x12 reference-tile clamp inside `render_mb_inter`.
        for r in 0..8usize {
            for c in 0..8usize {
                let sx = base_x + dx + c as i32;
                let sy = base_y + dy + r as i32;
                out[r * 8 + c] = bilinear_luma_sample(plane, stride, plane_h, sx, sy, x8, y8);
            }
        }
    }
}

/// Find the highest `coeff_idx` in 1..64 that holds a non-zero AC level.
/// Returns 0 if every AC is zero.
fn find_last_nonzero(levels: &[i32; 64]) -> usize {
    for idx in (1..64usize).rev() {
        if levels[idx] != 0 {
            return idx;
        }
    }
    0
}

/// Emit a single 8x8 block's worth of coefficients, mirroring the
/// decoder's [`crate::mb::parse_coeff`] state machine. `coded_dc` is the
/// predictor-adjusted DC level; `ac_levels[1..=63]` carry the quantised
/// AC levels in coeff_idx order. `last_nz` is the highest coeff_idx
/// carrying a non-zero AC (or 0 if all AC are zero).
fn emit_block_coefs(
    enc: &mut RangeEncoder,
    model: &Vp6Model,
    pt: usize,
    ctx: usize,
    coded_dc: i16,
    ac_levels: &[i32; 64],
    last_nz: usize,
) {
    // Mirrors decoder's `ct`, `coeff_idx`, model1/model2 selection.
    let mut ct: usize = 1;
    let mut coeff_idx: usize = 0;

    // Start on (model1=Dccv, model2=Dcct). After coeff_idx advances,
    // both switch to Ract(pt, ct, cg) in lock-step with the decoder.
    loop {
        let (m2, m1) = if coeff_idx == 0 {
            (
                &model.coeff_dcct[pt][ctx] as &[u8],
                &model.coeff_dccv[pt] as &[u8],
            )
        } else {
            let cg = tables::VP6_COEFF_GROUPS[coeff_idx] as usize;
            (
                &model.coeff_ract[pt][ct][cg] as &[u8],
                &model.coeff_ract[pt][ct][cg] as &[u8],
            )
        };

        // Decide: parse or skip. At coeff_idx == 0 we always take the
        // m2_0 decision bit to match the decoder (the `coeff_idx > 1
        // && ct == 0` shortcut only fires later).
        let value_at_idx: i32 = if coeff_idx == 0 {
            coded_dc as i32
        } else {
            ac_levels[coeff_idx]
        };
        let want_parse = value_at_idx != 0;

        // Decoder's go_parse: forced-true when `coeff_idx > 1 && ct == 0`.
        // In that case it does NOT read the m2_0 bit.
        let forced_parse = coeff_idx > 1 && ct == 0;
        if !forced_parse {
            enc.put_prob(m2[0], if want_parse { 1 } else { 0 });
        }

        if want_parse {
            let abs_val = value_at_idx.unsigned_abs() as i32;
            let sign = if value_at_idx < 0 { 1u8 } else { 0u8 };
            encode_coeff_value(enc, m2, m1, abs_val, &mut ct);
            enc.put_bit(sign);
            // coeff_idx advances by 1 after a parse.
            coeff_idx += 1;
            if coeff_idx >= 64 {
                break;
            }
            if coeff_idx > last_nz {
                // No more non-zero AC values. But the decoder always
                // reads either an end-of-block or a run AFTER a parse
                // that didn't break the loop — we need to cleanly close
                // the block. Continue the loop so the next iteration
                // emits the "no parse" + EOB path.
            }
        } else {
            // Skip path: run of zeros starting here. `ct := 0` on the
            // decoder.
            ct = 0;
            if coeff_idx > 0 {
                // Decide end-of-block vs run. If there are no more
                // non-zero AC values in this block we emit EOB (m2_1=0).
                let want_eob = coeff_idx > last_nz;
                enc.put_prob(m2[1], if want_eob { 0 } else { 1 });
                if want_eob {
                    break;
                }
                // Emit a run length: the smallest `run` such that
                // `coeff_idx + run` is the next non-zero slot.
                let mut next_nz = coeff_idx + 1;
                while next_nz <= last_nz && ac_levels[next_nz] == 0 {
                    next_nz += 1;
                }
                // If somehow we overshot, fall back to last_nz+1 (safe
                // because emit_run only cares about the numeric run).
                let run = (next_nz - coeff_idx) as i32;
                let model3_idx = if coeff_idx >= 6 { 1 } else { 0 };
                let runv = &model.coeff_runv[model3_idx];
                emit_run(enc, runv, run);
                coeff_idx += run as usize;
                if coeff_idx >= 64 {
                    break;
                }
            } else {
                // coeff_idx == 0 and we took the skip. Advance by 1 and
                // continue — decoder does `nidx = coeff_idx + run` with
                // `run = 1` carried over from init.
                coeff_idx += 1;
                if coeff_idx >= 64 {
                    break;
                }
            }
        }
    }
}

/// Emit a non-zero coefficient value (absolute value ≥ 1) using the
/// decoder's tree shape. On return, `ct` is set to the `ct` value the
/// decoder would adopt (1 for value==1, 2 for value>1) — callers need
/// this to pick the right Ract model slice for the next iteration.
fn encode_coeff_value(enc: &mut RangeEncoder, m2: &[u8], m1: &[u8], abs_val: i32, ct: &mut usize) {
    if abs_val == 1 {
        enc.put_prob(m2[2], 0);
        *ct = 1;
        return;
    }
    enc.put_prob(m2[2], 1);
    if abs_val == 2 {
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 0);
        *ct = 2;
        return;
    }
    if abs_val == 3 || abs_val == 4 {
        enc.put_prob(m2[3], 0);
        enc.put_prob(m2[4], 1);
        enc.put_prob(m1[5], (abs_val - 3) as u8);
        *ct = 2;
        return;
    }
    // Long path: m2[3] == 1 → PC_TREE + bias-category bits.
    enc.put_prob(m2[3], 1);
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
    for i in (0..=bit_len).rev() {
        let bit = ((delta >> i) & 1) as u8;
        enc.put_prob(tables::VP56_COEFF_PARSE_TABLE[cat][i as usize], bit);
    }
    *ct = 2;
}

/// Emit a run length `run ≥ 1` through the [`crate::tables::VP6_PCR_TREE`]
/// encoding the decoder reads. `probs` is the active `coeff_runv[i]`
/// slice (14 entries). Runs 1..=8 are encoded via tree leaves; runs ≥ 9
/// emit the escape leaf and then 6 raw prob-bits of `run - 9`.
fn emit_run(enc: &mut RangeEncoder, probs: &[u8; 14], run: i32) {
    let run = run.clamp(1, 9 + 63);
    let r = run as u32;
    match r {
        1 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 0);
            enc.put_prob(probs[2], 0);
        }
        2 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 0);
            enc.put_prob(probs[2], 1);
        }
        3 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 1);
            enc.put_prob(probs[3], 0);
        }
        4 => {
            enc.put_prob(probs[0], 0);
            enc.put_prob(probs[1], 1);
            enc.put_prob(probs[3], 1);
        }
        5 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 0);
            enc.put_prob(probs[6], 0);
        }
        6 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 0);
            enc.put_prob(probs[6], 1);
        }
        7 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 1);
            enc.put_prob(probs[7], 0);
        }
        8 => {
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 0);
            enc.put_prob(probs[5], 1);
            enc.put_prob(probs[7], 1);
        }
        _ => {
            // Escape leaf: prob indices (0,1), (4,1), then 6 bits of
            // `run - 9` LSB-first under probs[8..=13].
            enc.put_prob(probs[0], 1);
            enc.put_prob(probs[4], 1);
            let extra = (r as i32 - 9).clamp(0, 63);
            for i in 0..6 {
                let bit = ((extra >> i) & 1) as u8;
                enc.put_prob(probs[8 + i], bit);
            }
        }
    }
}

/// Signed integer nearest-to-zero division (rounds toward nearest, ties
/// away from zero). Mirrors what the decoder implicitly expects when it
/// multiplies `level * dequant` back up.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let w = 16usize;
        let h = 16usize;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let mut enc = Vp6Encoder::new(32);
        let bytes = enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
        assert_eq!(bytes[0], 64);
        // r23: byte 1 carries Vp3VersionNo (R5) << 3. Default encoder
        // emits VP6.0 (`Vp3VersionNo = 6`) so byte 1 = 6 << 3 = 0x30 (48).
        assert_eq!(bytes[1], 48, "Vp3VersionNo (bits 7..3) must be 6 (VP6.0)");
        assert!(bytes.len() > 8);
    }

    #[test]
    fn skip_frame_requires_preceding_keyframe() {
        let mut enc = Vp6Encoder::new(32);
        let res = enc.encode_skip_frame();
        assert!(res.is_err(), "skip before keyframe should error");
    }

    #[test]
    fn skip_frame_after_keyframe_has_inter_mode_bit() {
        let w = 16usize;
        let h = 16usize;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let mut enc = Vp6Encoder::new(32);
        enc.encode_keyframe(&y, &u, &v, w, h).unwrap();
        let skip = enc.encode_skip_frame().unwrap();
        // Top bit of byte 0 is frame_mode — 1 means inter.
        assert_eq!(skip[0] & 0x80, 0x80);
        assert_eq!(enc.dims_mb(), (1, 1));
    }

    #[test]
    fn pmbt_tree_walk_matches_get_tree() {
        // Cross-check that our recursive `walk_tree` produces a bit
        // sequence that the decoder's `get_tree` reads back as the same
        // MB type. Round-trip every leaf value the PMBT tree emits.
        use crate::range_coder::RangeCoder;
        for target in [
            tables::Vp56Mb::InterNoVecPf,
            tables::Vp56Mb::Intra,
            tables::Vp56Mb::InterDeltaPf,
            tables::Vp56Mb::InterV1Pf,
            tables::Vp56Mb::InterV2Pf,
            tables::Vp56Mb::InterNoVecGf,
            tables::Vp56Mb::InterDeltaGf,
            tables::Vp56Mb::Inter4V,
            tables::Vp56Mb::InterV1Gf,
            tables::Vp56Mb::InterV2Gf,
        ] {
            // Use the actual mb_type[1][0] probabilities the inter
            // encoder hands `walk_tree`.
            let probs: [u8; 10] = [10, 157, 255, 51, 1, 1, 253, 191, 170, 253];
            let mut enc = RangeEncoder::new();
            encode_pmbt_tree(&mut enc, &probs, target);
            let bytes = enc.finish();
            let mut dec = RangeCoder::new(&bytes).unwrap();
            let v = dec.get_tree(tables::PMBT_TREE, &probs);
            assert_eq!(v, target as i32, "round-trip failed for target {target:?}");
        }
    }

    #[test]
    fn forward_dct_dc_only_flat_tile() {
        // Flat tile at pixel=128 → DC = 0 (after subtracting 128).
        let tile = [128i32; 64];
        let mut out = [0i32; 64];
        forward_dct8x8(&tile, &mut out);
        assert_eq!(out[0], 0);
        for i in 1..64 {
            assert_eq!(out[i], 0, "AC at {i} should be zero for flat tile");
        }
    }

    #[test]
    fn forward_dct_flat_tile_dc_scales_correctly() {
        // Flat tile at pixel=255 → expected DC near 32 * 127 = 4064.
        let tile = [255i32; 64];
        let mut out = [0i32; 64];
        forward_dct8x8(&tile, &mut out);
        let expected = 32 * 127;
        assert!(
            (out[0] - expected).abs() <= 2,
            "DC should be ~{expected}, got {}",
            out[0]
        );
    }
}

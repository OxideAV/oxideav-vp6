//! Top-level per-frame assembly — the §9 → §10/§11.2/§13 → frame-dispatch
//! glue that sequences the already-implemented stages into a single
//! "compressed frame bytes in, [`Frame`] out" entry point.
//!
//! Every primitive this module orchestrates already exists:
//!
//! | stage                         | implemented in            |
//! | ----------------------------- | ------------------------- |
//! | §9 raw-bit header prefix      | [`Vp6FrameHeader::parse`] |
//! | §9 BoolCoder header tail      | [`Vp6HeaderTail::parse_with`] |
//! | §11.3/§11.4 filter config     | [`FilterConfig::from_header`] |
//! | I-frame coefficient decode    | [`decode_intra_frame`]    |
//! | P-frame fused driver          | [`decode_inter_frame_with_refs`] |
//! | §4 golden-frame bookkeeping   | [`ReferenceFrames`]       |
//!
//! What this module adds is the **sequencing**: parsing the header off
//! the front of a packet, building the BoolCoder over the partition,
//! threading the §9 cross-frame state (the most-recent I-frame's profile
//! / version, which Table 3 does not re-transmit) and the §4 reference
//! buffers across frames, and dispatching keyframe vs inter on the parsed
//! `FrameType`.
//!
//! ## Cross-frame state
//!
//! [`Vp6Decoder`] holds the small amount of state VP6 carries between
//! frames:
//!
//! * **Profile + version** — Table 3 (InterHeader) omits `VpProfile` and
//!   `Vp3VersionNo`; the decoder uses the values from the most recent
//!   I-frame. [`Vp6Decoder::decode_packet`] records them on every
//!   keyframe and threads them into the inter-frame tail parse.
//! * **Reference frames** — the §4 previous-frame + Golden Frame buffers
//!   ([`ReferenceFrames`]), seeded from the first keyframe and updated
//!   after every decoded frame per the §4 / `RefreshGoldenFrame` rules.
//!
//! ## Probability sub-streams (§8 Figure 1 / Figure 5)
//!
//! The §8 *Bitstream Map* fixes the order of the pre-data sub-streams that
//! sit between the §9 header and the per-macroblock data (all in the same
//! BoolCoder partition when `MultiStream == 0`):
//!
//! * **Keyframe** (Figure 1 → Figure 2): the per-MB info opens directly
//!   with the §8 Figure 5 *Coefficient Probability Updates* pass. There is
//!   no §10 mode or §11.2 MV tree on an I-frame (§10: I-frame MBs are
//!   implicitly intra, so no mode signaling).
//! * **Inter frame** (Figure 1): §10 *Mode Probability Updates* → §11.2
//!   *Mv Tree* → §8 Figure 5 *Coefficient Probability Updates* → per-MB
//!   data.
//!
//! This module sequences exactly that order. [`decode_keyframe`] consumes
//! the Figure-5 pass ([`decode_coefficient_prob_updates`]) and
//! [`decode_interframe`] consumes all three passes
//! ([`crate::mode_prob_update::update_mode_probs`] →
//! [`crate::mv_prob_update::update_mv_probs`] →
//! [`decode_coefficient_prob_updates`]) before dispatching. Each bank
//! starts from its baseline and the (typically empty) update pass mutates
//! it in place; the resulting banks + active §12.2 scan order thread into
//! the per-MB decode. The in-tree encoders emit the matching prefix, so a
//! keyframe carrying real §13 node-probability updates round-trips
//! end-to-end (see `keyframe_with_coeff_prob_updates_round_trips`).
//!
//! Cross-frame **persistence** of the §10 / §11.2 / §13 banks (carrying a
//! P-frame's mutated banks into the next frame instead of reseeding the
//! baseline) is a follow-up; the in-tree encoder emits only no-update
//! prefixes on inter frames, for which per-frame baseline reseeding is
//! bit-exact.
//!
//! ## Provenance
//!
//! Sequences only this crate's own already-spec-sourced stages
//! (`docs/video/vp6/vp6_format.pdf` §4 / §9–§17 plus the in-tree errata).
//! No external library code was consulted.

use crate::bool_coder::BoolCoder;
use crate::coeff_prob_update::{decode_coefficient_prob_updates, CoeffProbBanks};
use crate::coeff_source::CoeffSource;
use crate::frame_assembly::Frame;
use crate::frame_header::{CodingProfile, Vp3Version, Vp6FrameHeader, Vp6HeaderTail};
use crate::huff_coeff::HuffmanCoeffTables;
use crate::inter_frame::{decode_inter_frame_with_refs, FilterConfig, InterProbs, ReferenceFrames};
use crate::intra_frame::decode_intra_frame_from_source;
use crate::Error;

/// A stateful VP6 frame decoder: feed it whole compressed frames in
/// stream order and it produces reconstructed [`Frame`]s, carrying the
/// §4 reference buffers and the §9 cross-frame profile/version between
/// calls.
///
/// The first frame fed **must** be a keyframe (an inter frame before any
/// keyframe has no reference to predict from, and no profile/version to
/// inherit — [`Error::NotImplemented`] is returned in that case).
#[derive(Debug, Default)]
pub struct Vp6Decoder {
    /// §4 previous-frame + Golden Frame buffers. `None` until the first
    /// keyframe seeds them.
    refs: Option<ReferenceFrames>,
    /// Profile carried from the most recent I-frame (Table 3 omits it).
    profile: Option<CodingProfile>,
    /// Version carried from the most recent I-frame (Table 3 omits it).
    version: Option<Vp3Version>,
}

impl Vp6Decoder {
    /// Construct a decoder with no carried state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode one whole compressed VP6 frame (the bytes of a single
    /// packet, starting at the §9 header prefix) into a reconstructed
    /// 4:2:0 [`Frame`], updating the carried §4 reference state.
    ///
    /// # Errors
    ///
    /// * [`Error::Truncated`] if the packet runs out of bytes mid-parse.
    /// * [`Error::NotImplemented`] if:
    ///   * the first frame is not a keyframe (no reference / profile to
    ///     inherit);
    ///   * a frame signals `MultiStream == 1` (the two-partition split,
    ///     §6) or `UseHuffman == 1` (the Huffman second-partition coder,
    ///     §7) — paths not yet wired through this single-partition driver;
    ///   * a keyframe carries an unsupported profile/version combination.
    ///
    /// The §8 Figure 1 / Figure 5 probability-update sub-streams **are**
    /// consumed in spec order (keyframe: Figure 5; inter: §10 → §11.2 →
    /// Figure 5), so a frame carrying real coefficient / mode / MV
    /// probability updates decodes correctly.
    pub fn decode_packet(&mut self, bytes: &[u8]) -> Result<Frame, Error> {
        // Thread the profile carried from the most recent I-frame so an
        // Inter frame's Table 3 `Buff2Offset` presence gate
        // (`MultiStream || SIMPLE_PROFILE`) is evaluated correctly.
        let header = Vp6FrameHeader::parse_with_profile(bytes, self.profile)?;

        // §6: partition 1 (the §9 BoolCoder tail + mode/MV data) begins
        // immediately after the byte-aligned raw prefix. With
        // `MultiStream == 1` the §13 DCT tokens live in partition 2 at
        // `Buff2Offset` (a byte offset from the start of the compressed
        // frame buffer); with a single partition everything follows in
        // partition 1 and any transmitted `Buff2Offset` is inert.
        let (tail_bytes, partition2) = if header.multi_stream {
            let off = header.buff2_offset.ok_or(Error::Truncated)? as usize;
            if off <= header.raw_prefix_len || off > bytes.len() {
                return Err(Error::Truncated);
            }
            (
                bytes
                    .get(header.raw_prefix_len..off)
                    .ok_or(Error::Truncated)?,
                Some(bytes.get(off..).ok_or(Error::Truncated)?),
            )
        } else {
            (
                bytes.get(header.raw_prefix_len..).ok_or(Error::Truncated)?,
                None,
            )
        };

        if header.is_keyframe {
            self.decode_keyframe(&header, tail_bytes, partition2)
        } else {
            self.decode_interframe(&header, tail_bytes, partition2)
        }
    }

    /// Drop all carried state so the next [`Self::decode_packet`] starts
    /// from a fresh keyframe (e.g. after a container seek).
    pub fn reset(&mut self) {
        self.refs = None;
        self.profile = None;
        self.version = None;
    }

    /// True once a keyframe has seeded the §4 reference buffers — i.e.
    /// the decoder can accept an inter frame.
    pub fn has_reference(&self) -> bool {
        self.refs.is_some()
    }

    /// The §4 reference-frame state (previous-frame + Golden Frame buffers),
    /// or `None` until the first keyframe seeds them. Lets a Golden-aware
    /// encoder built on this decoder read the **decoded** Golden Frame to
    /// predict its `*_GOLD*` macroblocks against the exact pixels a downstream
    /// decoder holds.
    pub fn references(&self) -> Option<&crate::inter_frame::ReferenceFrames> {
        self.refs.as_ref()
    }

    fn decode_keyframe(
        &mut self,
        header: &Vp6FrameHeader,
        tail_bytes: &[u8],
        partition2: Option<&[u8]>,
    ) -> Result<Frame, Error> {
        // Profile / version are transmitted on the keyframe (Table 2).
        let profile = header.profile.ok_or(Error::Truncated)?;
        let version = header.version.ok_or(Error::Truncated)?;

        let mut bc = BoolCoder::new(tail_bytes)?;
        let tail = Vp6HeaderTail::parse_with(&mut bc, true, profile, version)?;

        // §5/§6: the Huffman coder only exists as a second-partition
        // transport; UseHuffman with a single partition is inconsistent.
        if tail.use_huffman && partition2.is_none() {
            return Err(Error::NotImplemented);
        }

        let h_fragments = tail.h_fragments.ok_or(Error::Truncated)? as usize;
        let v_fragments = tail.v_fragments.ok_or(Error::Truncated)? as usize;

        // §8 Figure 5 coefficient-probability-update sub-stream. On a
        // keyframe the per-MB info (Figure 2) opens directly with this
        // pass — there is no §10 mode or §11.2 MV tree on an I-frame
        // (§10: I-frame MBs are implicitly intra, no mode signaling). The
        // banks are reset to the §13 keyframe baselines, then the
        // (typically empty) Figure-5 updates apply on top. The pass also
        // yields the active §12.2 scan order. The pass always rides
        // partition 1's coder (Figures 2/3/4 all open with it; in the
        // Huffman arrangement partition 2 is raw bits and cannot carry
        // BoolCoder-coded updates).
        let mut banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut banks)?;
        let probs = banks.to_intra_probs();

        // §6 coefficient transport dispatch: single-stream reads the
        // tokens from the same partition-1 coder; MultiStream builds a
        // second source over partition 2 (BoolCoder or §7.2 Huffman per
        // UseHuffman).
        let frame = match partition2 {
            None => {
                let mut src = CoeffSource::Bool(&mut bc);
                decode_intra_frame_from_source(
                    &mut src,
                    h_fragments,
                    v_fragments,
                    header.dct_q_mask,
                    &probs,
                    &scan,
                )?
            }
            Some(p2) if tail.use_huffman => {
                let tables = HuffmanCoeffTables::from_banks(&banks);
                let mut src = CoeffSource::huffman(p2, &tables);
                decode_intra_frame_from_source(
                    &mut src,
                    h_fragments,
                    v_fragments,
                    header.dct_q_mask,
                    &probs,
                    &scan,
                )?
            }
            Some(p2) => {
                let mut bc2 = BoolCoder::new(p2)?;
                let mut src = CoeffSource::Bool(&mut bc2);
                decode_intra_frame_from_source(
                    &mut src,
                    h_fragments,
                    v_fragments,
                    header.dct_q_mask,
                    &probs,
                    &scan,
                )?
            }
        };

        // §4: a keyframe (re)seeds the previous-frame buffer and the
        // Golden Frame. Carry the profile/version for the following
        // inter frames.
        self.refs = Some(ReferenceFrames::from_keyframe(frame.clone()));
        self.profile = Some(profile);
        self.version = Some(version);

        Ok(frame)
    }

    fn decode_interframe(
        &mut self,
        header: &Vp6FrameHeader,
        tail_bytes: &[u8],
        partition2: Option<&[u8]>,
    ) -> Result<Frame, Error> {
        // Table 3 omits profile/version — inherit the most-recent
        // keyframe's. No keyframe yet ⇒ nothing to predict from.
        let profile = self.profile.ok_or(Error::NotImplemented)?;
        let version = self.version.ok_or(Error::NotImplemented)?;
        let refs = self.refs.as_ref().ok_or(Error::NotImplemented)?;

        // Two-partition inter frames are sequenced in a follow-up (the
        // per-MB prediction/coefficient split of Figures 3/4).
        if partition2.is_some() {
            return Err(Error::NotImplemented);
        }

        let mut bc = BoolCoder::new(tail_bytes)?;
        let tail = Vp6HeaderTail::parse_with(&mut bc, false, profile, version)?;

        if tail.use_huffman {
            return Err(Error::NotImplemented);
        }

        // Inter frames carry no geometry (Table 3) — reuse the reference
        // frame's dimensions, which the §4 buffers preserve.
        let (h_fragments, v_fragments) = refs.coded_fragments();

        let filter = FilterConfig::from_header(&tail, header.dct_q_mask);

        // §8 Figure 1 pre-data sub-streams, in the exact bitstream-map
        // order: §10 Mode Probability Updates → §11.2 MV Tree → §8
        // Figure 5 Coefficient Probability Updates → per-MB data. Each
        // bank starts from its baseline and the (typically empty) update
        // pass mutates it in place. (Cross-frame persistence of these
        // banks is a follow-up; the in-tree encoder emits only no-update
        // prefixes, for which per-frame baseline reseeding is exact.)
        let mut mode_probs = crate::modes::VP6_BASELINE_XMITTED_PROBS;
        let mut mv_probs = [
            crate::mv_decode::MvProbs::defaults(crate::mv_decode::MV_AXIS_X),
            crate::mv_decode::MvProbs::defaults(crate::mv_decode::MV_AXIS_Y),
        ];
        crate::mode_prob_update::update_mode_probs(&mut bc, &mut mode_probs)?;
        crate::mv_prob_update::update_mv_probs(&mut bc, &mut mv_probs)?;
        let mut coeff_banks = CoeffProbBanks::keyframe();
        let scan = decode_coefficient_prob_updates(&mut bc, &mut coeff_banks)?;

        let probs = InterProbs {
            mode_probs,
            mv_probs,
            coeffs: coeff_banks.to_intra_probs(),
        };

        let frame = decode_inter_frame_with_refs(
            &mut bc,
            h_fragments,
            v_fragments,
            header.dct_q_mask,
            &probs,
            &scan,
            &filter,
            refs,
        )?;

        // §4 update: the decoded frame becomes the new previous-frame
        // buffer; it refreshes the Golden Frame iff RefreshGoldenFrame.
        let refresh_golden = tail.refresh_golden_frame.unwrap_or(false);
        // Safe: `refs` was Some above, so `self.refs` is Some.
        if let Some(r) = self.refs.as_mut() {
            r.update_after_decode(frame.clone(), false, refresh_golden);
        }

        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_assembly::Frame;
    use crate::inter::{FilterFamily, PredictionFilterPolicy};
    use crate::inter_encode::{encode_inter_frame_me_packet, encode_inter_frame_packet};
    use crate::inter_frame::{BorderedRef, FilterConfig};
    use crate::intra_encode::encode_intra_frame;

    /// The Simple/VP6.0 inter-frame filter config the top-level decoder
    /// builds for an InterHeader with no prediction/loop-filter fields:
    /// fixed bilinear, loop filter off. The P-frame encoder must form its
    /// predictions with the same config so they are bit-identical.
    fn simple_inter_filter() -> FilterConfig {
        FilterConfig {
            policy: PredictionFilterPolicy::Fixed(FilterFamily::Bilinear),
            loop_filter_qi: None,
        }
    }

    fn pattern_frame(hf: usize, vf: usize) -> Frame {
        let mut frame = Frame::new(hf, vf);
        let yw = frame.y.width();
        let yh = frame.y.height();
        for r in 0..yh {
            for c in 0..yw {
                frame.y.samples_mut()[r * yw + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        let uw = frame.u.width();
        let uh = frame.u.height();
        for r in 0..uh {
            for c in 0..uw {
                frame.u.samples_mut()[r * uw + c] = (128 + ((r + c) % 40) as i32 - 20) as u8;
                frame.v.samples_mut()[r * uw + c] = (128 - ((r * 2 + c) % 40) as i32 + 20) as u8;
            }
        }
        frame
    }

    fn flat_frame(hf: usize, vf: usize, value: u8) -> Frame {
        let mut frame = Frame::new(hf, vf);
        for s in frame.y.samples_mut() {
            *s = value;
        }
        for s in frame.u.samples_mut() {
            *s = value;
        }
        for s in frame.v.samples_mut() {
            *s = value;
        }
        frame
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sse = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x as f64 - y as f64;
            sse += d * d;
        }
        let mse = sse / a.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    /// A keyframe encoded by `encode_intra_frame` decodes end-to-end via
    /// the top-level `decode_packet` — the header prefix + tail parse and
    /// the intra dispatch are sequenced internally (no hand-positioning of
    /// the BoolCoder by the caller).
    #[test]
    fn keyframe_round_trips_through_decode_packet() {
        let src = pattern_frame(4, 4);
        let bytes = encode_intra_frame(&src, 48).expect("encode");

        let mut dec = Vp6Decoder::new();
        let out = dec.decode_packet(&bytes).expect("decode");

        assert_eq!(out.y.width(), 32);
        assert!(dec.has_reference(), "keyframe must seed §4 refs");
        let y = psnr(src.y.samples(), out.y.samples());
        assert!(y >= 28.0, "luma PSNR {y:.2} dB below floor");
    }

    /// A keyframe whose §8 Figure 5 sub-stream carries **real** §13
    /// coefficient-probability updates round-trips through `decode_packet`:
    /// the encoder codes its tokens against the updated banks and emits the
    /// matching Figure-5 updates, the decoder seeds the keyframe baselines
    /// and applies the decoded updates to recover the identical banks, and
    /// the pixels reconstruct to the same floor as the no-update keyframe.
    /// This exercises the Figure-5 *content* path (set NewNodeProbValue
    /// records), not just the all-cleared no-update prefix.
    #[test]
    fn keyframe_with_coeff_prob_updates_round_trips() {
        use crate::coeff_prob_update::CoeffProbBanks;
        use crate::intra_encode::encode_intra_frame_with_banks;

        let src = pattern_frame(4, 4);

        // A non-baseline (but representable: even / 1) target bank across
        // all three §13 node-prob families.
        let mut banks = CoeffProbBanks::keyframe();
        banks.dc_probs[0][0] = 200;
        banks.dc_probs[1][4] = 64;
        banks.ac_probs[0][0][0][0] = 100;
        banks.ac_probs[1][2][3][5] = 220;
        banks.zrl_probs[0][2] = 80;
        banks.zrl_probs[1][9] = 2;

        let bytes = encode_intra_frame_with_banks(&src, 48, &banks).expect("encode");

        let mut dec = Vp6Decoder::new();
        let out = dec.decode_packet(&bytes).expect("decode");

        assert_eq!(out.y.width(), 32);
        assert!(dec.has_reference(), "keyframe must seed §4 refs");
        let y = psnr(src.y.samples(), out.y.samples());
        assert!(
            y >= 28.0,
            "luma PSNR {y:.2} dB below floor — Figure-5 content path regression"
        );

        // The updated-bank keyframe must reconstruct *identically* to the
        // baseline-bank keyframe at the same q: the §13 node probabilities
        // change only the entropy coding, never the decoded coefficients,
        // so the pixels are bit-for-bit the same.
        let baseline_bytes = encode_intra_frame(&src, 48).expect("encode baseline");
        let mut dec2 = Vp6Decoder::new();
        let baseline_out = dec2
            .decode_packet(&baseline_bytes)
            .expect("decode baseline");
        assert_eq!(
            out.y.samples(),
            baseline_out.y.samples(),
            "node-prob updates must not change decoded pixels"
        );
        assert_eq!(out.u.samples(), baseline_out.u.samples());
        assert_eq!(out.v.samples(), baseline_out.v.samples());
    }

    /// A flat keyframe round-trips exactly through the top-level driver.
    #[test]
    fn flat_keyframe_exact_through_decode_packet() {
        let src = flat_frame(4, 4, 128);
        let bytes = encode_intra_frame(&src, 32).expect("encode");
        let mut dec = Vp6Decoder::new();
        let out = dec.decode_packet(&bytes).expect("decode");
        for &s in out.y.samples() {
            assert_eq!(s, 128, "flat keyframe must round-trip exactly");
        }
    }

    /// A full keyframe → P-frame GOP decodes through one `Vp6Decoder`
    /// instance: the keyframe seeds the §4 refs + carries the
    /// profile/version, then an unchanged inter frame predicted from the
    /// decoded keyframe reproduces it exactly.
    #[test]
    fn keyframe_then_unchanged_pframe_gop() {
        let src = pattern_frame(4, 4);
        let q = 40;

        let mut dec = Vp6Decoder::new();
        // Keyframe.
        let kf_bytes = encode_intra_frame(&src, q).expect("encode I");
        let kf_out = dec.decode_packet(&kf_bytes).expect("decode I");

        // P-frame: encode the *decoded* keyframe (unchanged content)
        // against itself, which the all-zero-MV inter encoder reproduces
        // exactly. The prediction is formed from the §11.5-bordered
        // decoded keyframe with the same Simple/VP6.0 filter the decoder
        // seeds.
        let prev = BorderedRef::new(&kf_out);
        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();
        let pf_bytes =
            encode_inter_frame_packet(&kf_out, &prev, q, &probs, &filter).expect("encode P");
        let pf_out = dec.decode_packet(&pf_bytes).expect("decode P");

        assert_eq!(pf_out.y.samples(), kf_out.y.samples());
        assert_eq!(pf_out.u.samples(), kf_out.u.samples());
        assert_eq!(pf_out.v.samples(), kf_out.v.samples());
    }

    /// A three-frame GOP (I, P, P) decodes through one `Vp6Decoder`: each
    /// P-frame predicts from the **previous decoded frame**, so the §4
    /// previous-frame buffer must advance after every `decode_packet`. An
    /// all-zero-MV chain of unchanged content reproduces the keyframe at
    /// every step — proving the reference buffer is the just-decoded
    /// frame, not a stale keyframe copy.
    #[test]
    fn three_frame_gop_advances_reference_chain() {
        let src = pattern_frame(4, 4);
        let q = 40;
        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();

        let mut dec = Vp6Decoder::new();
        let kf_out = dec
            .decode_packet(&encode_intra_frame(&src, q).expect("encode I"))
            .expect("decode I");

        // P1 predicts from the decoded keyframe.
        let p1_bytes =
            encode_inter_frame_packet(&kf_out, &BorderedRef::new(&kf_out), q, &probs, &filter)
                .expect("encode P1");
        let p1_out = dec.decode_packet(&p1_bytes).expect("decode P1");
        assert_eq!(p1_out.y.samples(), kf_out.y.samples());

        // P2 predicts from the decoded P1 (the advanced previous-frame
        // buffer). Encode against P1's reconstruction, decode against the
        // decoder's carried state.
        let p2_bytes =
            encode_inter_frame_packet(&p1_out, &BorderedRef::new(&p1_out), q, &probs, &filter)
                .expect("encode P2");
        let p2_out = dec.decode_packet(&p2_bytes).expect("decode P2");
        assert_eq!(
            p2_out.y.samples(),
            p1_out.y.samples(),
            "P2 must predict from the decoded P1, not a stale keyframe"
        );
        assert_eq!(p2_out.u.samples(), p1_out.u.samples());
        assert_eq!(p2_out.v.samples(), p1_out.v.samples());
    }

    /// A content-changed P-frame (different source than the reference)
    /// decodes through `decode_packet` above a quantiser-bounded PSNR
    /// floor — the residual path (non-zero coefficients on the zero-MV
    /// prediction) flows through the top-level driver.
    #[test]
    fn changed_pframe_clears_floor_through_decode_packet() {
        let kf_src = flat_frame(4, 4, 120);
        let q = 32;
        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();

        let mut dec = Vp6Decoder::new();
        let kf_out = dec
            .decode_packet(&encode_intra_frame(&kf_src, q).expect("encode I"))
            .expect("decode I");

        // Changed P-frame source: a gradient over the flat reference.
        let mut p_src = kf_out.clone();
        let yw = p_src.y.width();
        let yh = p_src.y.height();
        for r in 0..yh {
            for c in 0..yw {
                p_src.y.samples_mut()[r * yw + c] =
                    (120 + ((r + c) % 24) as i32 - 12).clamp(0, 255) as u8;
            }
        }

        let p_bytes =
            encode_inter_frame_packet(&p_src, &BorderedRef::new(&kf_out), q, &probs, &filter)
                .expect("encode P");
        let p_out = dec.decode_packet(&p_bytes).expect("decode P");

        let y = psnr(p_src.y.samples(), p_out.y.samples());
        assert!(
            y >= 30.0,
            "changed P-frame luma PSNR {y:.2} dB below floor — residual path regression"
        );
    }

    /// A keyframe → **motion-estimated** P-frame GOP decodes end-to-end
    /// through one `Vp6Decoder` via `encode_inter_frame_me_packet`: the
    /// keyframe seeds the §4 refs + profile/version, then a translated
    /// P-frame (whose source moved relative to the decoded keyframe) decodes
    /// above a quantiser-bounded floor — the motion-estimated MVs and the
    /// §11 differential-reference emission flow through the top-level driver
    /// against the decoder's `InterProbs::keyframe()` / header-derived
    /// `FilterConfig`, with no caller-side filter wiring.
    #[test]
    fn keyframe_then_me_pframe_gop_through_decode_packet() {
        let src = pattern_frame(6, 6);
        let q = 32;

        let mut dec = Vp6Decoder::new();
        let kf_out = dec
            .decode_packet(&encode_intra_frame(&src, q).expect("encode I"))
            .expect("decode I");

        // The P-frame source is the decoded keyframe shifted right+down by a
        // few luma pixels with edge replication — so a real MV brings the
        // prediction back onto the source.
        let mut p_src = Frame::new(kf_out.h_fragments, kf_out.v_fragments);
        let shift = 3i32;
        for (sp, dp) in [
            (&kf_out.y, &mut p_src.y),
            (&kf_out.u, &mut p_src.u),
            (&kf_out.v, &mut p_src.v),
        ] {
            let w = sp.width() as i32;
            let h = sp.height() as i32;
            // Chroma uses half the luma shift (½-resolution planes).
            let s = if sp.width() == kf_out.y.width() {
                shift
            } else {
                shift / 2
            };
            for r in 0..h {
                for c in 0..w {
                    let sr = (r + s).clamp(0, h - 1);
                    let sc = (c + s).clamp(0, w - 1);
                    dp.samples_mut()[(r * w + c) as usize] = sp.samples()[(sr * w + sc) as usize];
                }
            }
        }

        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();
        let pf_bytes =
            encode_inter_frame_me_packet(&p_src, &BorderedRef::new(&kf_out), q, &probs, &filter)
                .expect("encode ME P");
        let pf_out = dec.decode_packet(&pf_bytes).expect("decode ME P");

        let y = psnr(p_src.y.samples(), pf_out.y.samples());
        assert!(
            y >= 28.0,
            "ME P-frame luma PSNR {y:.2} dB below floor through decode_packet"
        );
        assert_eq!(pf_out.y.width(), 48);
    }

    /// A two-partition (`MultiStream == 1`) BoolCoder keyframe decodes
    /// through `decode_packet` to pixels **bit-identical** to the
    /// single-stream encoding at the same quantiser: the §6 partition
    /// arrangement changes only the entropy transport.
    #[test]
    fn multistream_bool_keyframe_round_trips() {
        use crate::intra_encode::encode_intra_frame_multistream;

        let src = pattern_frame(4, 4);
        let q = 48;

        let ms_bytes = encode_intra_frame_multistream(&src, q, false).expect("encode ms");
        let hdr = crate::frame_header::Vp6FrameHeader::parse(&ms_bytes).expect("header");
        assert!(hdr.is_keyframe);
        assert!(hdr.multi_stream);
        let off = hdr.buff2_offset.expect("Buff2Offset present") as usize;
        assert!(off > hdr.raw_prefix_len && off < ms_bytes.len());

        let mut dec = Vp6Decoder::new();
        let ms_out = dec.decode_packet(&ms_bytes).expect("decode ms");
        assert!(dec.has_reference(), "multistream keyframe must seed refs");

        let ss_bytes = encode_intra_frame(&src, q).expect("encode ss");
        let mut dec2 = Vp6Decoder::new();
        let ss_out = dec2.decode_packet(&ss_bytes).expect("decode ss");
        assert_eq!(ms_out.y.samples(), ss_out.y.samples());
        assert_eq!(ms_out.u.samples(), ss_out.u.samples());
        assert_eq!(ms_out.v.samples(), ss_out.v.samples());
    }

    /// A two-partition **Huffman** keyframe (`UseHuffman == 1`) decodes
    /// through `decode_packet` to the same pixels as the single-stream
    /// arithmetic encoding — the §7.2/§13.2.2/§13.3.2 raw-bit coefficient
    /// transport carries identical coefficients.
    #[test]
    fn multistream_huffman_keyframe_round_trips() {
        use crate::intra_encode::encode_intra_frame_multistream;

        let src = pattern_frame(4, 4);
        let q = 48;

        let huff_bytes = encode_intra_frame_multistream(&src, q, true).expect("encode huff");
        let mut dec = Vp6Decoder::new();
        let huff_out = dec.decode_packet(&huff_bytes).expect("decode huff");

        let ss_bytes = encode_intra_frame(&src, q).expect("encode ss");
        let mut dec2 = Vp6Decoder::new();
        let ss_out = dec2.decode_packet(&ss_bytes).expect("decode ss");
        assert_eq!(huff_out.y.samples(), ss_out.y.samples());
        assert_eq!(huff_out.u.samples(), ss_out.u.samples());
        assert_eq!(huff_out.v.samples(), ss_out.v.samples());
    }

    /// A multistream keyframe seeds the §4 refs + profile/version, so a
    /// following single-stream P-frame decodes against it — the carried
    /// state is partition-arrangement-agnostic.
    #[test]
    fn multistream_keyframe_then_pframe_gop() {
        use crate::intra_encode::encode_intra_frame_multistream;

        let src = pattern_frame(4, 4);
        let q = 40;
        let mut dec = Vp6Decoder::new();
        let kf_out = dec
            .decode_packet(&encode_intra_frame_multistream(&src, q, true).expect("encode I"))
            .expect("decode I");

        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();
        let pf_bytes =
            encode_inter_frame_packet(&kf_out, &BorderedRef::new(&kf_out), q, &probs, &filter)
                .expect("encode P");
        let pf_out = dec.decode_packet(&pf_bytes).expect("decode P");
        assert_eq!(pf_out.y.samples(), kf_out.y.samples());
    }

    /// A multistream keyframe whose Buff2Offset points outside the packet
    /// (or into the raw prefix) surfaces `Truncated`, not a panic.
    #[test]
    fn multistream_bad_buff2offset_errors() {
        use crate::intra_encode::encode_intra_frame_multistream;

        let src = flat_frame(2, 2, 90);
        let good = encode_intra_frame_multistream(&src, 32, false).expect("encode");

        // Corrupt the 16-bit Buff2Offset (bytes 2..4 of the intra raw
        // prefix) to point past the packet end.
        let mut past_end = good.clone();
        past_end[2] = 0xFF;
        past_end[3] = 0xFF;
        let mut dec = Vp6Decoder::new();
        assert!(matches!(
            dec.decode_packet(&past_end),
            Err(Error::Truncated)
        ));

        // ...and to point inside the raw prefix itself.
        let mut inside_prefix = good;
        inside_prefix[2] = 0;
        inside_prefix[3] = 2;
        assert!(matches!(
            dec.decode_packet(&inside_prefix),
            Err(Error::Truncated)
        ));
    }

    /// An inter frame before any keyframe has no reference / profile —
    /// the driver reports `NotImplemented` rather than guessing.
    #[test]
    fn pframe_before_keyframe_errors() {
        let src = pattern_frame(2, 2);
        let prev = BorderedRef::new(&src);
        let probs = InterProbs::keyframe();
        let filter = simple_inter_filter();
        let pf_bytes =
            encode_inter_frame_packet(&src, &prev, 40, &probs, &filter).expect("encode P");
        let mut dec = Vp6Decoder::new();
        assert!(matches!(
            dec.decode_packet(&pf_bytes),
            Err(Error::NotImplemented)
        ));
    }

    /// `reset` drops the carried reference state.
    #[test]
    fn reset_clears_reference() {
        let src = flat_frame(2, 2, 100);
        let bytes = encode_intra_frame(&src, 32).expect("encode");
        let mut dec = Vp6Decoder::new();
        dec.decode_packet(&bytes).expect("decode");
        assert!(dec.has_reference());
        dec.reset();
        assert!(!dec.has_reference());
    }
}

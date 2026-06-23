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
//! ## Probability sub-streams
//!
//! A conformant header is followed by the §10 mode-probability,
//! §11.2 MV-probability and §13 coefficient-probability update
//! sub-streams *before* the per-macroblock data, all in the same
//! BoolCoder partition (when `MultiStream == 0`). Those update passes are
//! implemented (`mode_prob_update`, `mv_prob_update`, `prob_update`,
//! `scan_update`) but their exact Figure-5 ordering relative to the
//! header tail needs a conformant `.vp6` fixture to pin down (see the
//! README "Blocked / remaining" note). This module therefore decodes
//! frames whose header signals **no probability updates** — the shape the
//! in-tree `intra_encode` / `inter_encode` encoders produce — by seeding
//! the keyframe-baseline banks ([`IntraProbs::keyframe`] /
//! [`InterProbs::keyframe`]) and dispatching directly. When the fixture
//! lands, the update passes slot in between the tail parse and the
//! dispatch with no change to this module's public surface.
//!
//! ## Provenance
//!
//! Sequences only this crate's own already-spec-sourced stages
//! (`docs/video/vp6/vp6_format.pdf` §4 / §9–§17 plus the in-tree errata).
//! No external library code was consulted.

use crate::bool_coder::BoolCoder;
use crate::frame_assembly::Frame;
use crate::frame_header::{CodingProfile, Vp3Version, Vp6FrameHeader, Vp6HeaderTail};
use crate::inter_frame::{decode_inter_frame_with_refs, FilterConfig, InterProbs, ReferenceFrames};
use crate::intra_frame::{decode_intra_frame, IntraProbs};
use crate::scan::DEFAULT_SCAN_ORDER;
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
    ///   * a frame signals `MultiStream == 1`, `UseHuffman == 1`, or a
    ///     non-default scan / probability update — paths whose exact
    ///     bitstream sequencing awaits a conformant fixture;
    ///   * a keyframe carries an unsupported profile/version combination.
    pub fn decode_packet(&mut self, bytes: &[u8]) -> Result<Frame, Error> {
        let header = Vp6FrameHeader::parse(bytes)?;

        // The §9 BoolCoder tail + the data partition begin immediately
        // after the byte-aligned raw prefix (MultiStream == 0; the
        // single-partition arrangement).
        if header.multi_stream {
            // Two-partition (MultiStream) sequencing needs the
            // Buff2Offset split + a conformant fixture to validate.
            return Err(Error::NotImplemented);
        }
        let tail_bytes = bytes.get(header.raw_prefix_len..).ok_or(Error::Truncated)?;

        if header.is_keyframe {
            self.decode_keyframe(&header, tail_bytes)
        } else {
            self.decode_interframe(&header, tail_bytes)
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

    fn decode_keyframe(
        &mut self,
        header: &Vp6FrameHeader,
        tail_bytes: &[u8],
    ) -> Result<Frame, Error> {
        // Profile / version are transmitted on the keyframe (Table 2).
        let profile = header.profile.ok_or(Error::Truncated)?;
        let version = header.version.ok_or(Error::Truncated)?;

        let mut bc = BoolCoder::new(tail_bytes)?;
        let tail = Vp6HeaderTail::parse_with(&mut bc, true, profile, version)?;

        // No §13 probability-update sub-stream is consumed (see module
        // docs): the keyframe banks are the §13 baselines. A conformant
        // header that signalled UseHuffman uses a different coefficient
        // coder than this BoolCoder dispatch.
        if tail.use_huffman {
            return Err(Error::NotImplemented);
        }

        let h_fragments = tail.h_fragments.ok_or(Error::Truncated)? as usize;
        let v_fragments = tail.v_fragments.ok_or(Error::Truncated)? as usize;

        let probs = IntraProbs::keyframe();
        let frame = decode_intra_frame(
            &mut bc,
            h_fragments,
            v_fragments,
            header.dct_q_mask,
            &probs,
            &DEFAULT_SCAN_ORDER,
        )?;

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
    ) -> Result<Frame, Error> {
        // Table 3 omits profile/version — inherit the most-recent
        // keyframe's. No keyframe yet ⇒ nothing to predict from.
        let profile = self.profile.ok_or(Error::NotImplemented)?;
        let version = self.version.ok_or(Error::NotImplemented)?;
        let refs = self.refs.as_ref().ok_or(Error::NotImplemented)?;

        let mut bc = BoolCoder::new(tail_bytes)?;
        let tail = Vp6HeaderTail::parse_with(&mut bc, false, profile, version)?;

        if tail.use_huffman {
            return Err(Error::NotImplemented);
        }

        // Inter frames carry no geometry (Table 3) — reuse the reference
        // frame's dimensions, which the §4 buffers preserve.
        let (h_fragments, v_fragments) = refs.coded_fragments();

        let filter = FilterConfig::from_header(&tail, header.dct_q_mask);
        let probs = InterProbs::keyframe();

        let frame = decode_inter_frame_with_refs(
            &mut bc,
            h_fragments,
            v_fragments,
            header.dct_q_mask,
            &probs,
            &DEFAULT_SCAN_ORDER,
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
    use crate::inter_encode::encode_inter_frame_packet;
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

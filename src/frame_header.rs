//! VP6 frame header — raw-bit prefix parser (spec §9 Table 1 / Table 2).
//!
//! The VP6 frame header is split between two coders:
//!
//! 1. A short raw-bit prefix (spec uses `R(n)` to denote raw bits)
//!    that always starts the partition. Table 1's `FrameType`,
//!    `DctQMask`, `MultiStream` (all on byte 0) and Table 2's
//!    `Vp3VersionNo`, `VpProfile`, `Reserved`, `Buff2Offset` (next
//!    3 raw bytes on I-frames when the conditional triggers) live in
//!    this prefix.
//! 2. A longer BoolCoder-encoded tail (`b(n)` notation) — `VFragments`,
//!    `HFragments`, scaling, filter selectors, `UseHuffman`.
//!
//! This module implements **both** parts:
//!
//! * [`Vp6FrameHeader::parse`] reads the byte-aligned raw-bit prefix
//!   (part 1) and reports how many bytes it consumed via
//!   [`Vp6FrameHeader::raw_prefix_len`], which is where the BoolCoder
//!   partition begins.
//! * [`Vp6HeaderTail::parse`] reads the BoolCoder-coded `b(n)` tail
//!   (part 2): `VFragments`, `HFragments`, `OutputVFragments`,
//!   `OutputHFragments`, `ScalingMode`, the Advanced-profile prediction
//!   and loop-filter selectors, and the trailing `UseHuffman` flag.
//!
//! The §7.3 BoolCoder `Split` formula `Split = 1 + (((Range-1) *
//! Probability) >> 7)` is resolved by the clean-room errata #35 in
//! `docs/video/vp6/vp6-errata-and-clarifications.md`: the `>> 7`
//! (divide by 128) is correct and intentional — probability 128 is the
//! half-interval point, exactly what the fixed-probability `b(n)`
//! reads require. The `b(n)` tail therefore decodes cleanly through the
//! existing [`crate::bool_coder::BoolCoder`].
//!
//! All field semantics in this file are sourced verbatim from
//! `docs/video/vp6/vp6_format.pdf` §9 (Tables 1/2/3) and the staged
//! errata; no external library code was consulted.

use oxideav_core::bits::BitReader;

use crate::Error;

/// Frame type signalled by the first bit of the partition (Table 1,
/// `FrameType` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// `FrameType == 0` — intra-coded (random-access) frame.
    Intra,
    /// `FrameType == 1` — inter-coded (P-) frame, predicted from the
    /// previous reconstruction or the Golden Frame.
    Inter,
}

/// Coding profile selected by `VpProfile` in the IntraHeader (Table 2).
///
/// Only two of the four 2-bit encodings are defined; the others are
/// reserved (spec: "0 Simple, 3 Advanced (1 and 2 undefined)").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodingProfile {
    /// `VpProfile == 0`. Loop-filter and bi-cubic prediction filter
    /// are both disabled; second partition (if present) carries DCT
    /// tokens.
    Simple,
    /// `VpProfile == 3`. Loop-filter optional, bi-cubic prediction
    /// filter optional, dynamic filter switching available.
    Advanced,
    /// `VpProfile == 1 || 2`. Reserved — the parser surfaces this so
    /// callers can decide whether to surface as a hard error.
    Reserved(u8),
}

impl CodingProfile {
    /// True if this profile is the "Simple" profile (matters for the
    /// `Buff2Offset` presence rule in Table 2's footnote).
    pub fn is_simple(self) -> bool {
        matches!(self, CodingProfile::Simple)
    }
}

/// VPx version reported in the IntraHeader (Table 2 `Vp3VersionNo`).
///
/// Spec wording: "The values 6, 7, and 8 represent VP6.0, VP6.1, and
/// VP6.2 bitstreams, respectively. The decoder should check this
/// field to ensure that it can decode the bitstream."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp3Version {
    /// `Vp3VersionNo == 6` — VP6.0.
    Vp60,
    /// `Vp3VersionNo == 7` — VP6.1.
    Vp61,
    /// `Vp3VersionNo == 8` — VP6.2. Spec calls out several fields
    /// that only exist on this version (e.g. `PredictionFilterAlpha`).
    Vp62,
    /// Any other 5-bit value the encoder emitted. The decoder is
    /// expected to reject these; we surface the raw value so the
    /// caller can decide what to do.
    Other(u8),
}

impl Vp3Version {
    fn from_raw(raw: u8) -> Self {
        match raw {
            6 => Self::Vp60,
            7 => Self::Vp61,
            8 => Self::Vp62,
            other => Self::Other(other),
        }
    }
}

/// Raw-bit prefix of the VP6 frame header.
///
/// Holds every field in Table 1 plus the four raw-bit IntraHeader
/// fields (`Vp3VersionNo`, `VpProfile`, `Reserved`, `Buff2Offset`).
/// Inter (P-) frames carry no R(n) fields beyond Table 1 itself — the
/// InterHeader's only R-coded field is `Buff2Offset`, and it's gated
/// by the same MultiStream / SIMPLE_PROFILE rule as the IntraHeader's,
/// but on P-frames `VpProfile` isn't transmitted in-band (Table 3
/// omits it), so the caller is expected to carry the profile state
/// from the most recent I-frame in order to know whether Buff2Offset
/// is present.
///
/// This struct represents the *prefix* that's structurally parseable
/// without BoolCoder. Fields that live downstream of the BoolCoder
/// switch (scaling, filter selectors, `UseHuffman`) are not present
/// here; they will be added once the BoolCoder spec gap is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp6FrameHeader {
    /// True iff `FrameType == 0` (I-frame).
    pub is_keyframe: bool,
    /// Six-bit `DctQMask` quantiser index, 0..=63 (Table 1).
    pub dct_q_mask: u8,
    /// True iff `MultiStream == 1` — the frame carries two partitions
    /// rather than one.
    pub multi_stream: bool,
    /// `Vp3VersionNo`, only present on I-frames (Table 2). `None` for
    /// P-frames since the InterHeader (Table 3) doesn't re-transmit
    /// it; callers should carry the most-recent I-frame value.
    pub version: Option<Vp3Version>,
    /// `VpProfile`, only present on I-frames (Table 2). Same caveat
    /// as `version` regarding P-frames.
    pub profile: Option<CodingProfile>,
    /// `Reserved` bit from Table 2 (intra) — must be 0 per spec but
    /// "must be consumed during decoding". Surfaced verbatim so the
    /// caller can decide whether to harden against non-conforming
    /// encoders.
    pub reserved_bit: Option<u8>,
    /// `Buff2Offset` — 16-bit byte offset to the second partition,
    /// present only when `MultiStream == 1 || profile == Simple`.
    /// `None` if the field wasn't transmitted (no second partition
    /// signalled, Advanced profile).
    pub buff2_offset: Option<u16>,
    /// Number of bytes consumed by the byte-aligned raw-bit prefix
    /// (Table 1 + the IntraHeader raw fields when present). The
    /// BoolCoder-coded `b(n)` tail begins at this byte offset into the
    /// frame payload — feed `&bytes[raw_prefix_len..]` to
    /// [`Vp6HeaderTail::parse`].
    ///
    /// The raw prefix is always byte-aligned: Table 1 occupies one full
    /// byte (`1 + 6 + 1`), the IntraHeader raw fields occupy a second
    /// full byte (`5 + 2 + 1`), and `Buff2Offset` is a whole 16-bit
    /// (2-byte) field. So this is 1 byte for an Inter frame, 2 bytes for
    /// an Intra frame without `Buff2Offset`, and 4 bytes for an Intra
    /// frame with it.
    pub raw_prefix_len: usize,
}

impl Vp6FrameHeader {
    /// Parse the raw-bit prefix of a VP6 frame header from `bytes`.
    ///
    /// `bytes` must point at the start of the compressed frame payload
    /// (i.e. the first byte of partition 1). The function reads at
    /// most three bytes on Inter frames (just Table 1) and at most
    /// five bytes on Intra frames (Table 1 + Table 2 raw prefix).
    ///
    /// Returns [`Error::Truncated`] if the slice is too short to
    /// satisfy the field set the FrameType/Profile combination
    /// requires.
    ///
    /// Notes:
    ///
    /// * Bit packing across the wire is MSB-first per the spec's
    ///   convention for R(n). `BitReader` from `oxideav_core::bits`
    ///   already implements that order.
    /// * `Buff2Offset`'s presence on Inter frames depends on the
    ///   profile of the most recent I-frame (Simple profile always
    ///   transmits Buff2Offset). Because that profile isn't in the
    ///   InterHeader's wire format, this parser cannot determine
    ///   Inter-frame `Buff2Offset` from a single packet — callers
    ///   that need it must thread the profile in. For round 1 we
    ///   surface `buff2_offset = None` on Inter frames and document
    ///   the cross-frame dependency.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let mut br = BitReader::new(bytes);

        // Table 1, byte 0:
        //   FrameType   R(1)
        //   DctQMask    R(6)
        //   MultiStream R(1)
        let frame_type_raw = br.read_u32(1).map_err(|_| Error::Truncated)?;
        let dct_q_mask = br.read_u32(6).map_err(|_| Error::Truncated)? as u8;
        let multi_stream_raw = br.read_u32(1).map_err(|_| Error::Truncated)?;

        let is_keyframe = frame_type_raw == 0;
        let multi_stream = multi_stream_raw == 1;

        if !is_keyframe {
            // Inter frames have no further R(n) prefix: Table 3
            // starts with `Buff2Offset R(16)` only when the
            // (MultiStream || SIMPLE_PROFILE) gate fires — and
            // SIMPLE_PROFILE is established by the most recent
            // I-frame, not the current packet. Punt to None and let
            // a higher layer thread the state in once the BoolCoder
            // gap is resolved.
            return Ok(Self {
                is_keyframe,
                dct_q_mask,
                multi_stream,
                version: None,
                profile: None,
                reserved_bit: None,
                buff2_offset: None,
                raw_prefix_len: br.byte_position(),
            });
        }

        // Table 2 raw prefix (Intra only):
        //   Vp3VersionNo R(5)
        //   VpProfile    R(2)
        //   Reserved     R(1)
        //   Buff2Offset  R(16)   -- if (MultiStream==1) || (SIMPLE_PROFILE==1)
        let vp3_version_raw = br.read_u32(5).map_err(|_| Error::Truncated)? as u8;
        let vp_profile_raw = br.read_u32(2).map_err(|_| Error::Truncated)? as u8;
        let reserved = br.read_u32(1).map_err(|_| Error::Truncated)? as u8;

        let profile = match vp_profile_raw {
            0 => CodingProfile::Simple,
            3 => CodingProfile::Advanced,
            other => CodingProfile::Reserved(other),
        };
        let version = Vp3Version::from_raw(vp3_version_raw);

        let buff2_offset = if multi_stream || profile.is_simple() {
            let v = br.read_u32(16).map_err(|_| Error::Truncated)? as u16;
            Some(v)
        } else {
            None
        };

        Ok(Self {
            is_keyframe,
            dct_q_mask,
            multi_stream,
            version: Some(version),
            profile: Some(profile),
            reserved_bit: Some(reserved),
            buff2_offset,
            raw_prefix_len: br.byte_position(),
        })
    }

    /// Byte length of the raw-bit prefix — equivalently, the offset at
    /// which the BoolCoder-coded `b(n)` tail begins.
    ///
    /// Convenience accessor mirroring the [`Self::raw_prefix_len`]
    /// field, for callers that prefer a method.
    pub fn raw_prefix_len(&self) -> usize {
        self.raw_prefix_len
    }
}

/// The prediction-filter selection signalled in the §9 header tail
/// (Tables 2/3 `AutoSelectPMFlag` / `BiCubicOrBiLinearFiltFlag`).
///
/// Two filter families exist (§11.4): a bi-linear and a bi-cubic
/// sub-pixel interpolation filter. The header chooses between a *fixed*
/// selection (one family used for the whole frame) and *auto-select*
/// (the decoder picks per-block from the variance / MV-size
/// thresholds). The variants below capture both the fixed choice and
/// the auto-select thresholds verbatim from the bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionFilter {
    /// `AutoSelectPMFlag == 0`: a single fixed filter family is used for
    /// the whole frame. `bicubic` is the decoded
    /// `BiCubicOrBiLinearFiltFlag` (`true` ⇒ bi-cubic, `false` ⇒
    /// bi-linear).
    Fixed {
        /// `BiCubicOrBiLinearFiltFlag`: `true` ⇒ bi-cubic, `false` ⇒
        /// bi-linear.
        bicubic: bool,
    },
    /// `AutoSelectPMFlag == 1`: the decoder auto-selects bi-cubic vs
    /// bi-linear per block using the two thresholds.
    AutoSelect {
        /// `PredictionFilterVarThresh` b(5) — variance at or above which
        /// the bi-cubic filter is used (`0` ⇒ always bi-cubic).
        var_thresh: u8,
        /// `PredictionFilterMvSizeThresh` b(3) — largest MV component
        /// (whole pixels) for bi-cubic use is `1 << (thresh - 1)`.
        mv_size_thresh: u8,
    },
    /// The Advanced-profile prediction-filter selector was not present
    /// in this header. Simple profile never transmits it; in that case
    /// the spec mandates bi-linear with no dynamic switching (§5).
    NotSignalled,
}

/// Loop-filter selection from the §9 InterHeader tail (Table 3
/// `UseLoopFilter` / `LoopFilterSelector`).
///
/// Present only on Advanced-profile frames. The IntraHeader (Table 2)
/// carries no loop-filter fields, so an I-frame always reports
/// [`LoopFilter::NotSignalled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopFilter {
    /// `UseLoopFilter == 0`: the prediction loop-filter is disabled for
    /// this frame.
    Disabled,
    /// `UseLoopFilter == 1`: the loop-filter is enabled. `selector` is
    /// the decoded `LoopFilterSelector` (spec mandates `0` — basic
    /// de-blocking; the `1` de-ringing variant is reserved and not
    /// defined by the decoder spec).
    Enabled {
        /// `LoopFilterSelector`: `0` basic de-blocking, `1` de-ringing
        /// (reserved per spec note).
        selector: u8,
    },
    /// The loop-filter selector was not present (Simple profile, or an
    /// IntraHeader which carries no loop-filter fields).
    NotSignalled,
}

/// The BoolCoder-coded `b(n)` tail of a VP6 frame header (spec §9
/// Table 2 IntraHeader / Table 3 InterHeader, plus the trailing
/// Table 1 `UseHuffman` flag).
///
/// This is the second half of the frame header — everything past the
/// byte-aligned raw-bit prefix parsed by [`Vp6FrameHeader::parse`]. It
/// is consumed with the §7.3 [`crate::bool_coder::BoolCoder`] at fixed
/// node probability 128 (the §3 `b(n)` operator), which the errata #35
/// disambiguation confirms is the correct half-interval behaviour.
///
/// Field presence follows the conditionals printed in Tables 2/3:
///
/// * `Output*Fragments` / `ScalingMode` are present **only** on the
///   IntraHeader (an Inter frame inherits the coded/scaled geometry of
///   the keyframe it predicts from, so it carries none of those).
/// * `RefreshGoldenFrame` is present **only** on the InterHeader.
/// * The prediction-filter and loop-filter selectors are
///   Advanced-profile only; the VP6.2-gated variants additionally
///   require `Vp3VersionNo == 8`.
/// * `PredictionFilterAlpha` is present only on VP6.2 bitstreams.
/// * `UseHuffman` is the final `b(1)` and is always present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp6HeaderTail {
    /// `VFragments` b(8) — rows of 8×8 blocks in the un-scaled coded
    /// image (`None` on an Inter frame, which carries no geometry).
    pub v_fragments: Option<u8>,
    /// `HFragments` b(8) — cols of 8×8 blocks in the un-scaled coded
    /// image (`None` on an Inter frame).
    pub h_fragments: Option<u8>,
    /// `OutputVFragments` b(8) — rows of 8×8 blocks in the scaled
    /// output image (`None` on an Inter frame).
    pub output_v_fragments: Option<u8>,
    /// `OutputHFragments` b(8) — cols of 8×8 blocks in the scaled
    /// output image (`None` on an Inter frame).
    pub output_h_fragments: Option<u8>,
    /// `ScalingMode` b(2) — how the coded frame is mapped to the output
    /// resolution (`None` on an Inter frame).
    pub scaling_mode: Option<crate::scaling::ScalingMode>,
    /// `RefreshGoldenFrame` b(1) — whether this decoded frame becomes
    /// the new Golden Frame. `None` on an Intra frame (Table 2 carries
    /// no such field).
    pub refresh_golden_frame: Option<bool>,
    /// Loop-filter selection (Table 3, Advanced profile only).
    pub loop_filter: LoopFilter,
    /// Prediction-filter selection (Tables 2/3, Advanced profile only).
    pub prediction_filter: PredictionFilter,
    /// `PredictionFilterAlpha` b(4) — index into the bi-cubic filter
    /// coefficient set. Present only on VP6.2 (`Vp3VersionNo == 8`)
    /// bitstreams.
    pub prediction_filter_alpha: Option<u8>,
    /// `UseHuffman` b(1) — `false` ⇒ second partition uses the
    /// BoolCoder, `true` ⇒ it uses the Huffman coder. Always present.
    pub use_huffman: bool,
}

impl Vp6HeaderTail {
    /// Parse the BoolCoder-coded `b(n)` tail of a VP6 frame header.
    ///
    /// `tail_bytes` must point at the start of the BoolCoder partition,
    /// i.e. `&frame[header.raw_prefix_len()..]`. `is_keyframe`,
    /// `profile`, and `version` come from the already-parsed
    /// [`Vp6FrameHeader`] and gate the per-field conditionals exactly as
    /// printed in Tables 2/3 (and Table 1 for `UseHuffman`).
    ///
    /// On an Inter frame `version` and `profile` are not transmitted in
    /// the InterHeader; the caller is expected to thread in the state
    /// carried from the most recent I-frame (the same cross-frame
    /// dependency [`Vp6FrameHeader::parse`] documents for
    /// `Buff2Offset`).
    ///
    /// Returns [`Error::Truncated`] if the BoolCoder runs out of bytes,
    /// or [`Error::NotImplemented`] if a `ScalingMode` outside `0..=3`
    /// is decoded (impossible from a 2-bit field, but kept explicit so
    /// the typed [`crate::scaling::ScalingMode`] mapping stays total).
    pub fn parse(
        tail_bytes: &[u8],
        is_keyframe: bool,
        profile: CodingProfile,
        version: Vp3Version,
    ) -> Result<Self, Error> {
        let mut bc = crate::bool_coder::BoolCoder::new(tail_bytes)?;
        Self::parse_with(&mut bc, is_keyframe, profile, version)
    }

    /// Parse the `b(n)` tail from an already-initialised
    /// [`crate::bool_coder::BoolCoder`].
    ///
    /// Identical to [`Self::parse`] but operates on a borrowed coder so
    /// a future per-frame driver can continue consuming the same
    /// BoolCoder partition (the §10 mode data immediately follows the
    /// header tail in partition 1 when `MultiStream == 0`). The coder is
    /// left positioned just past `UseHuffman`.
    pub fn parse_with(
        bc: &mut crate::bool_coder::BoolCoder<'_>,
        is_keyframe: bool,
        profile: CodingProfile,
        version: Vp3Version,
    ) -> Result<Self, Error> {
        let advanced = matches!(profile, CodingProfile::Advanced);
        let is_vp62 = matches!(version, Vp3Version::Vp62);

        // Table 2 (IntraHeader) carries the geometry + scaling fields;
        // Table 3 (InterHeader) omits them entirely.
        let (v_fragments, h_fragments, output_v_fragments, output_h_fragments, scaling_mode) =
            if is_keyframe {
                let vf = bc.decode_b(8)? as u8;
                let hf = bc.decode_b(8)? as u8;
                let ovf = bc.decode_b(8)? as u8;
                let ohf = bc.decode_b(8)? as u8;
                let sm_raw = bc.decode_b(2)? as u8;
                let sm =
                    crate::scaling::ScalingMode::from_b2(sm_raw).ok_or(Error::NotImplemented)?;
                (Some(vf), Some(hf), Some(ovf), Some(ohf), Some(sm))
            } else {
                (None, None, None, None, None)
            };

        // Table 3 (InterHeader) carries RefreshGoldenFrame here, ahead
        // of the loop-filter fields. Table 2 has no such field.
        let refresh_golden_frame = if is_keyframe {
            None
        } else {
            Some(bc.decode_b1()? != 0)
        };

        // Loop-filter selectors (Table 3, Advanced profile only). The
        // IntraHeader has no loop-filter fields at all.
        let loop_filter = if !is_keyframe && advanced {
            if bc.decode_b1()? != 0 {
                let selector = bc.decode_b1()?;
                LoopFilter::Enabled { selector }
            } else {
                LoopFilter::Disabled
            }
        } else {
            LoopFilter::NotSignalled
        };

        // Prediction-filter selectors. On the IntraHeader (Table 2) the
        // AutoSelectPMFlag / threshold / fixed-flag fields are present
        // for any Advanced-profile frame. On the InterHeader (Table 3)
        // the same fields are additionally gated on VP6.2
        // (Vp3VersionNo == 8).
        let pf_present = if is_keyframe {
            advanced
        } else {
            advanced && is_vp62
        };
        let prediction_filter = if pf_present {
            if bc.decode_b1()? != 0 {
                // AutoSelectPMFlag == 1
                let var_thresh = bc.decode_b(5)? as u8;
                let mv_size_thresh = bc.decode_b(3)? as u8;
                PredictionFilter::AutoSelect {
                    var_thresh,
                    mv_size_thresh,
                }
            } else {
                // AutoSelectPMFlag == 0
                let bicubic = bc.decode_b1()? != 0;
                PredictionFilter::Fixed { bicubic }
            }
        } else {
            PredictionFilter::NotSignalled
        };

        // PredictionFilterAlpha b(4) — VP6.2 only (both tables).
        let prediction_filter_alpha = if is_vp62 {
            Some(bc.decode_b(4)? as u8)
        } else {
            None
        };

        // Table 1 trailing field: UseHuffman b(1), always present.
        let use_huffman = bc.decode_b1()? != 0;

        Ok(Self {
            v_fragments,
            h_fragments,
            output_v_fragments,
            output_h_fragments,
            scaling_mode,
            refresh_golden_frame,
            loop_filter,
            prediction_filter,
            prediction_filter_alpha,
            use_huffman,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-encode a minimal Intra frame, Simple profile, MultiStream=0
    /// (Buff2Offset still present because Simple). DctQMask = 0x2A
    /// (binary 101010).
    ///
    /// Byte 0 bits (MSB first): 0 101010 0  -> 0x54
    /// Byte 1 bits: Vp3VersionNo(5)=8 (binary 01000), VpProfile(2)=0,
    ///              Reserved(1)=0
    ///   => 01000_00_0 -> 0x40
    /// Bytes 2..4: Buff2Offset (R16) = 0x1234 -> 0x12 0x34
    /// (No further bits parsed by this module — BoolCoder land starts
    /// after.)
    #[test]
    fn parses_intra_simple_with_buff2offset() {
        let bytes = [0x54, 0x40, 0x12, 0x34, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert!(hdr.is_keyframe);
        assert_eq!(hdr.dct_q_mask, 0x2A);
        assert!(!hdr.multi_stream);
        assert_eq!(hdr.version, Some(Vp3Version::Vp62));
        assert_eq!(hdr.profile, Some(CodingProfile::Simple));
        assert_eq!(hdr.reserved_bit, Some(0));
        assert_eq!(hdr.buff2_offset, Some(0x1234));
    }

    /// Intra, Advanced profile, MultiStream=0 -> Buff2Offset NOT
    /// transmitted (the gate `MultiStream || Simple` is false).
    ///
    /// Byte 0: 0 101010 0 -> 0x54
    /// Byte 1: Vp3VersionNo=6 (00110), VpProfile=3 (11), Reserved=0
    ///   => 00110_11_0 -> 0x36
    #[test]
    fn parses_intra_advanced_without_buff2offset() {
        let bytes = [0x54, 0x36, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert!(hdr.is_keyframe);
        assert_eq!(hdr.version, Some(Vp3Version::Vp60));
        assert_eq!(hdr.profile, Some(CodingProfile::Advanced));
        assert_eq!(hdr.buff2_offset, None);
    }

    /// Intra, Advanced profile but MultiStream=1 -> Buff2Offset IS
    /// transmitted (gate triggers via MultiStream branch).
    ///
    /// Byte 0: 0 000000 1 -> 0x01
    /// Byte 1: Vp3VersionNo=7 (00111), VpProfile=3 (11), Reserved=0
    ///   => 00111_11_0 -> 0x3E
    /// Bytes 2..4: Buff2Offset = 0xBEEF
    #[test]
    fn parses_intra_advanced_multistream_with_buff2offset() {
        let bytes = [0x01, 0x3E, 0xBE, 0xEF, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert!(hdr.is_keyframe);
        assert_eq!(hdr.dct_q_mask, 0);
        assert!(hdr.multi_stream);
        assert_eq!(hdr.version, Some(Vp3Version::Vp61));
        assert_eq!(hdr.profile, Some(CodingProfile::Advanced));
        assert_eq!(hdr.buff2_offset, Some(0xBEEF));
    }

    /// Inter frame: only Table 1 is parsed in round 1 since the
    /// InterHeader's Buff2Offset gate depends on the profile of the
    /// most recent I-frame, which this single-frame API doesn't have.
    ///
    /// Byte 0: 1 111111 1 -> 0xFF (FrameType=1, DctQMask=63,
    ///          MultiStream=1)
    #[test]
    fn parses_inter_table_one_only() {
        let bytes = [0xFF, 0x00, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert!(!hdr.is_keyframe);
        assert_eq!(hdr.dct_q_mask, 63);
        assert!(hdr.multi_stream);
        // Inter-frame parser doesn't reach into the IntraHeader fields.
        assert!(hdr.version.is_none());
        assert!(hdr.profile.is_none());
        assert!(hdr.reserved_bit.is_none());
        assert!(hdr.buff2_offset.is_none());
    }

    /// Reserved VpProfile encodings (1 or 2) round-trip through the
    /// `Reserved(u8)` variant so callers can decide policy.
    ///
    /// Byte 0: 0 000000 0 -> 0x00 (Intra, q=0, MultiStream=0)
    /// Byte 1: Vp3VersionNo=8 (01000), VpProfile=1 (01), Reserved=0
    ///   => 01000_01_0 -> 0x42
    /// (No Buff2Offset because !MultiStream && profile != Simple.)
    #[test]
    fn surfaces_reserved_profile_encoding() {
        let bytes = [0x00, 0x42, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert_eq!(hdr.profile, Some(CodingProfile::Reserved(1)));
        assert_eq!(hdr.buff2_offset, None);
    }

    /// `Vp3Version::Other(_)` round-trips for unrecognised version
    /// codes — the spec says the decoder *should* reject these but
    /// the parser surfaces them so policy can live higher up.
    #[test]
    fn surfaces_unknown_version_code() {
        // Byte 0: 0 000000 0 -> 0x00
        // Byte 1: Vp3VersionNo=5 (00101), VpProfile=0 (00), Reserved=0
        //   => 00101_00_0 -> 0x28
        // Buff2Offset bytes 2..4 (Simple profile triggers gate).
        let bytes = [0x00, 0x28, 0x00, 0x00, 0x00];
        let hdr = Vp6FrameHeader::parse(&bytes).unwrap();
        assert_eq!(hdr.version, Some(Vp3Version::Other(5)));
    }

    /// Truncated input — far too short to even contain Table 1.
    #[test]
    fn truncated_input_returns_error() {
        assert!(matches!(Vp6FrameHeader::parse(&[]), Err(Error::Truncated)));
    }

    /// Truncated *during* the IntraHeader Buff2Offset read.
    #[test]
    fn truncated_during_buff2offset_returns_error() {
        // Intra, Simple, MultiStream=0 -> Buff2Offset present, needs
        // bytes 2..4. Provide only bytes 0..3 (one byte short of
        // completing the 16-bit Buff2Offset).
        let bytes = [0x00, 0x40, 0x00];
        assert!(matches!(
            Vp6FrameHeader::parse(&bytes),
            Err(Error::Truncated)
        ));
    }

    /// `raw_prefix_len` reports the byte offset at which the BoolCoder
    /// tail begins, for each prefix shape.
    #[test]
    fn raw_prefix_len_per_shape() {
        // Inter: only Table 1 (1 byte).
        let inter = Vp6FrameHeader::parse(&[0xFF, 0x00, 0x00]).unwrap();
        assert_eq!(inter.raw_prefix_len, 1);
        assert_eq!(inter.raw_prefix_len(), 1);

        // Intra, Advanced, no MultiStream -> no Buff2Offset (2 bytes).
        let intra_adv = Vp6FrameHeader::parse(&[0x54, 0x36, 0x00]).unwrap();
        assert_eq!(intra_adv.raw_prefix_len, 2);

        // Intra, Simple -> Buff2Offset present (4 bytes).
        let intra_simple = Vp6FrameHeader::parse(&[0x54, 0x40, 0x12, 0x34, 0x00]).unwrap();
        assert_eq!(intra_simple.raw_prefix_len, 4);
    }
}

#[cfg(test)]
mod tail_tests {
    use super::*;
    use crate::bool_coder::BoolCoder;

    // A fixed, tightly-bounded BoolCoder partition used across the tail
    // tests. We never *encode* — instead each test independently drives
    // a fresh `BoolCoder` over these exact bytes to capture the decoded
    // `b(n)` field values, then asserts `Vp6HeaderTail::parse` over the
    // same bytes reproduces them. This is the same capture-then-verify
    // pattern the `bool_coder` unit tests use; no range encoder is built
    // and the buffer is 16 bytes.
    const TAIL: [u8; 16] = [
        0x9C, 0x2B, 0x4F, 0xE1, 0x07, 0xD3, 0x6A, 0x55, 0x10, 0xFE, 0x21, 0x8B, 0x44, 0xC0, 0x3D,
        0x77,
    ];

    /// Intra / Simple / VP6.0: the tail is exactly
    /// `VFragments b(8)`, `HFragments b(8)`, `OutputVFragments b(8)`,
    /// `OutputHFragments b(8)`, `ScalingMode b(2)`, then `UseHuffman
    /// b(1)` — no prediction/loop-filter fields (Simple profile) and no
    /// alpha (not VP6.2).
    #[test]
    fn intra_simple_vp60_field_order() {
        // Capture the reference field values directly from a raw coder.
        let mut ref_bc = BoolCoder::new(&TAIL).unwrap();
        let vf = ref_bc.decode_b(8).unwrap() as u8;
        let hf = ref_bc.decode_b(8).unwrap() as u8;
        let ovf = ref_bc.decode_b(8).unwrap() as u8;
        let ohf = ref_bc.decode_b(8).unwrap() as u8;
        let sm = ref_bc.decode_b(2).unwrap() as u8;
        let huff = ref_bc.decode_b1().unwrap() != 0;

        let tail = Vp6HeaderTail::parse(
            &TAIL,
            /* is_keyframe */ true,
            CodingProfile::Simple,
            Vp3Version::Vp60,
        )
        .unwrap();

        assert_eq!(tail.v_fragments, Some(vf));
        assert_eq!(tail.h_fragments, Some(hf));
        assert_eq!(tail.output_v_fragments, Some(ovf));
        assert_eq!(tail.output_h_fragments, Some(ohf));
        assert_eq!(
            tail.scaling_mode,
            Some(crate::scaling::ScalingMode::from_b2(sm).unwrap())
        );
        // Simple profile: no prediction/loop-filter fields.
        assert_eq!(tail.prediction_filter, PredictionFilter::NotSignalled);
        assert_eq!(tail.loop_filter, LoopFilter::NotSignalled);
        assert_eq!(tail.prediction_filter_alpha, None);
        assert_eq!(tail.refresh_golden_frame, None);
        assert_eq!(tail.use_huffman, huff);
    }

    /// Intra / Advanced / VP6.0: after the five geometry/scaling fields
    /// the Advanced prediction-filter selector appears
    /// (`AutoSelectPMFlag` then either thresholds or the fixed flag),
    /// then `UseHuffman`. No loop-filter on an IntraHeader, no alpha
    /// (not VP6.2).
    #[test]
    fn intra_advanced_vp60_reads_prediction_filter() {
        let mut ref_bc = BoolCoder::new(&TAIL).unwrap();
        let _vf = ref_bc.decode_b(8).unwrap();
        let _hf = ref_bc.decode_b(8).unwrap();
        let _ovf = ref_bc.decode_b(8).unwrap();
        let _ohf = ref_bc.decode_b(8).unwrap();
        let _sm = ref_bc.decode_b(2).unwrap();
        let expected_pf = if ref_bc.decode_b1().unwrap() != 0 {
            let var_thresh = ref_bc.decode_b(5).unwrap() as u8;
            let mv_size_thresh = ref_bc.decode_b(3).unwrap() as u8;
            PredictionFilter::AutoSelect {
                var_thresh,
                mv_size_thresh,
            }
        } else {
            PredictionFilter::Fixed {
                bicubic: ref_bc.decode_b1().unwrap() != 0,
            }
        };
        let expected_huff = ref_bc.decode_b1().unwrap() != 0;

        let tail =
            Vp6HeaderTail::parse(&TAIL, true, CodingProfile::Advanced, Vp3Version::Vp60).unwrap();

        assert_eq!(tail.prediction_filter, expected_pf);
        // IntraHeader carries no loop-filter / golden-frame fields.
        assert_eq!(tail.loop_filter, LoopFilter::NotSignalled);
        assert_eq!(tail.refresh_golden_frame, None);
        assert_eq!(tail.prediction_filter_alpha, None);
        assert_eq!(tail.use_huffman, expected_huff);
    }

    /// Inter / Advanced / VP6.0: the InterHeader carries no geometry,
    /// begins with `RefreshGoldenFrame b(1)`, then the Advanced
    /// loop-filter fields. The prediction-filter selector is gated on
    /// VP6.2 for InterHeaders, so on VP6.0 it is NOT present — straight
    /// to `UseHuffman` after the loop filter.
    #[test]
    fn inter_advanced_vp60_reads_golden_and_loop_filter() {
        let mut ref_bc = BoolCoder::new(&TAIL).unwrap();
        let expected_golden = ref_bc.decode_b1().unwrap() != 0;
        let expected_lf = if ref_bc.decode_b1().unwrap() != 0 {
            LoopFilter::Enabled {
                selector: ref_bc.decode_b1().unwrap(),
            }
        } else {
            LoopFilter::Disabled
        };
        // No prediction filter on VP6.0 InterHeader, no alpha.
        let expected_huff = ref_bc.decode_b1().unwrap() != 0;

        let tail = Vp6HeaderTail::parse(
            &TAIL,
            /* is_keyframe */ false,
            CodingProfile::Advanced,
            Vp3Version::Vp60,
        )
        .unwrap();

        assert_eq!(tail.refresh_golden_frame, Some(expected_golden));
        assert_eq!(tail.loop_filter, expected_lf);
        // No geometry fields on an InterHeader.
        assert_eq!(tail.v_fragments, None);
        assert_eq!(tail.scaling_mode, None);
        // VP6.0 InterHeader: prediction filter gated on VP6.2 → absent.
        assert_eq!(tail.prediction_filter, PredictionFilter::NotSignalled);
        assert_eq!(tail.prediction_filter_alpha, None);
        assert_eq!(tail.use_huffman, expected_huff);
    }

    /// Inter / Advanced / VP6.2: the full InterHeader tail —
    /// `RefreshGoldenFrame`, loop-filter, the VP6.2-gated
    /// prediction-filter selector, `PredictionFilterAlpha b(4)`, then
    /// `UseHuffman`.
    #[test]
    fn inter_advanced_vp62_reads_full_tail() {
        let mut ref_bc = BoolCoder::new(&TAIL).unwrap();
        let expected_golden = ref_bc.decode_b1().unwrap() != 0;
        let expected_lf = if ref_bc.decode_b1().unwrap() != 0 {
            LoopFilter::Enabled {
                selector: ref_bc.decode_b1().unwrap(),
            }
        } else {
            LoopFilter::Disabled
        };
        let expected_pf = if ref_bc.decode_b1().unwrap() != 0 {
            PredictionFilter::AutoSelect {
                var_thresh: ref_bc.decode_b(5).unwrap() as u8,
                mv_size_thresh: ref_bc.decode_b(3).unwrap() as u8,
            }
        } else {
            PredictionFilter::Fixed {
                bicubic: ref_bc.decode_b1().unwrap() != 0,
            }
        };
        let expected_alpha = ref_bc.decode_b(4).unwrap() as u8;
        let expected_huff = ref_bc.decode_b1().unwrap() != 0;

        let tail =
            Vp6HeaderTail::parse(&TAIL, false, CodingProfile::Advanced, Vp3Version::Vp62).unwrap();

        assert_eq!(tail.refresh_golden_frame, Some(expected_golden));
        assert_eq!(tail.loop_filter, expected_lf);
        assert_eq!(tail.prediction_filter, expected_pf);
        assert_eq!(tail.prediction_filter_alpha, Some(expected_alpha));
        assert_eq!(tail.use_huffman, expected_huff);
    }

    /// `parse_with` over a borrowed coder leaves it positioned for the
    /// next consumer (the §10 mode data). After parsing an Intra/Simple
    /// tail the coder state must match an independent coder driven over
    /// the identical field sequence.
    #[test]
    fn parse_with_leaves_coder_positioned() {
        // Independent reference coder: same field sequence as
        // Intra/Simple/VP6.0 (4×b(8), b(2), b(1)).
        let mut ref_bc = BoolCoder::new(&TAIL).unwrap();
        let _ = ref_bc.decode_b(8).unwrap();
        let _ = ref_bc.decode_b(8).unwrap();
        let _ = ref_bc.decode_b(8).unwrap();
        let _ = ref_bc.decode_b(8).unwrap();
        let _ = ref_bc.decode_b(2).unwrap();
        let _ = ref_bc.decode_b1().unwrap();

        let mut bc = BoolCoder::new(&TAIL).unwrap();
        let _ = Vp6HeaderTail::parse_with(&mut bc, true, CodingProfile::Simple, Vp3Version::Vp60)
            .unwrap();

        assert_eq!(bc.range(), ref_bc.range());
        assert_eq!(bc.value(), ref_bc.value());
        assert_eq!(bc.count(), ref_bc.count());
        assert_eq!(bc.pos(), ref_bc.pos());
    }

    /// Truncation: a partition shorter than the BoolCoder's 4-byte init
    /// surfaces `Truncated`.
    #[test]
    fn tail_truncated_input() {
        assert_eq!(
            Vp6HeaderTail::parse(&[0x00, 0x00], true, CodingProfile::Simple, Vp3Version::Vp60),
            Err(Error::Truncated)
        );
    }

    /// End-to-end: parse the raw prefix, then feed the BoolCoder tail
    /// from `raw_prefix_len` onward. The combined parse must succeed and
    /// the geometry must be present on the keyframe.
    #[test]
    fn end_to_end_prefix_then_tail() {
        // Intra, Simple, MultiStream=0: byte0=0x54 (I, q=0x2A, MS=0),
        // byte1=0x00 (Vp3VersionNo=0... use Simple anyway), Buff2Offset
        // bytes 2..4. We only need a valid 4-byte prefix; the tail
        // bytes follow.
        let mut frame = vec![0x54u8, 0x00, 0x00, 0x00];
        frame.extend_from_slice(&TAIL);

        let hdr = Vp6FrameHeader::parse(&frame).unwrap();
        assert!(hdr.is_keyframe);
        assert_eq!(hdr.raw_prefix_len, 4);

        let tail = Vp6HeaderTail::parse(
            &frame[hdr.raw_prefix_len..],
            hdr.is_keyframe,
            CodingProfile::Simple,
            Vp3Version::Vp60,
        )
        .unwrap();
        assert!(tail.v_fragments.is_some());
        assert!(tail.scaling_mode.is_some());
    }
}

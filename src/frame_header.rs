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
//! This module implements (1) only. The BoolCoder primitive in spec
//! §7.3 has a formula ambiguity (see `docs/video/vp6/vp6_format.pdf`
//! page 15) — `Split = 1 + ( ((Range-1) * Probability) >> 7 )`
//! produces `Split = Range` when `Probability = 128`, which collapses
//! the 50/50 case to always-decode-0 and so cannot reproduce the
//! `b(n)` fixed-prob-128 bitstream the spec also calls out. Until that
//! ambiguity is resolved by a docs update we deliberately stop the
//! parse at the BoolCoder boundary rather than commit a guess.
//!
//! See the crate-root `DOCS-GAP` block for the gap report being filed
//! against the spec.
//!
//! All field semantics in this file are sourced verbatim from
//! `docs/video/vp6/vp6_format.pdf` §9; no external library code was
//! consulted.

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
}

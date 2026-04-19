//! Streaming VP6 decoder, implementing
//! [`oxideav_codec::Decoder`](oxideav_codec::Decoder).
//!
//! This module ties together the pieces in `range_coder`, `models`,
//! `mb`, and `dsp` to produce a decoded `VideoFrame` per input packet.
//!
//! See FFmpeg's `vp56.c::ff_vp56_decode_frame` + `vp56_decode_mbs` for
//! the reference control flow.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::frame_header::{FrameHeader, FrameKind};
use crate::mb::{self, BlockScratch, MacroblockInfo, Mv, RefKind};
use crate::models::{self, Vp6Model};
use crate::range_coder::RangeCoder;
use crate::tables;

/// VP6 flavour. Tracks whether we're decoding the FLV-only `vp6f` (no
/// alpha) or `vp6a` (alpha plane prefixed). Today only `vp6f` decode
/// proceeds — `vp6a` packets surface as `Error::Unsupported`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp6Variant {
    Flv,
    FlvAlpha,
}

impl Vp6Variant {
    pub fn from_codec_id(id: &CodecId) -> Self {
        match id.as_str() {
            "vp6a" => Self::FlvAlpha,
            _ => Self::Flv,
        }
    }
}

/// Streaming VP6 decoder.
pub struct Vp6Decoder {
    codec_id: CodecId,
    variant: Vp6Variant,
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    width: Option<u32>,
    height: Option<u32>,

    // Persistent state across frames.
    mb_width: usize,
    mb_height: usize,
    sub_version: u8,
    filter_header: u8,
    interlaced: bool,
    model: Vp6Model,
    scratch: BlockScratch,
    prev_frame: Option<RefPlanes>,
    golden_frame: Option<RefPlanes>,
    macroblocks: Vec<MacroblockInfo>,
    /// Set to `true` once we've decoded the first keyframe and know
    /// `mb_width`/`mb_height` + sub_version etc.
    initialised: bool,
}

/// Lightweight reference-frame plane holder.
#[derive(Clone, Debug)]
pub(crate) struct RefPlanes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl std::fmt::Debug for Vp6Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp6Decoder")
            .field("codec_id", &self.codec_id)
            .field("variant", &self.variant)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("queued", &self.queued.len())
            .finish()
    }
}

impl Vp6Decoder {
    pub fn new(params: CodecParameters) -> Self {
        let variant = Vp6Variant::from_codec_id(&params.codec_id);
        Self {
            codec_id: params.codec_id,
            variant,
            queued: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            width: params.width,
            height: params.height,
            mb_width: 0,
            mb_height: 0,
            sub_version: 0,
            filter_header: 0,
            interlaced: false,
            model: Vp6Model::default(),
            scratch: BlockScratch::new(1),
            prev_frame: None,
            golden_frame: None,
            macroblocks: Vec::new(),
            initialised: false,
        }
    }

    fn on_keyframe(&mut self, header: &FrameHeader) {
        self.mb_width = header.mb_width as usize;
        self.mb_height = header.mb_height as usize;
        self.sub_version = header.sub_version;
        self.filter_header = header.filter_header;
        self.interlaced = header.interlaced;
        self.width = Some(header.frame_width());
        self.height = Some(header.frame_height());
        self.scratch = BlockScratch::new(self.mb_width);
        self.model = Vp6Model::default();
        self.model.reset_defaults(self.interlaced);
        self.model.rebuild_coeff_tables(self.sub_version);
        self.macroblocks = vec![
            MacroblockInfo {
                mb_type: tables::Vp56Mb::Intra,
                mv: Mv::default(),
            };
            self.mb_width * self.mb_height
        ];
        self.initialised = true;
    }

    fn decode_bytes(&mut self, data: &[u8]) -> Result<VideoFrame> {
        if matches!(self.variant, Vp6Variant::FlvAlpha) {
            return Err(Error::unsupported(
                "VP6: vp6a (alpha plane) decode not yet implemented",
            ));
        }

        let header = FrameHeader::parse(data)?;
        if matches!(header.kind, FrameKind::Key) {
            self.on_keyframe(&header);
        } else if !self.initialised {
            return Err(Error::invalid("VP6: inter frame before keyframe"));
        }

        mb::init_dequant(&mut self.scratch, header.qp);
        self.scratch.keyframe = matches!(header.kind, FrameKind::Key);

        // The bool coder starts at `range_coder_offset`. When there's a
        // separated coefficient partition the secondary bool coder
        // starts `coeff_offset_bytes` past that point.
        let rac_start = header.range_coder_offset;
        let rac_buf = &data[rac_start..];
        let mut rac = RangeCoder::new(rac_buf)?;
        let mut header_filter_info = false;

        // Rest of the picture header.
        let mut golden_frame_flag = false;
        if matches!(header.kind, FrameKind::Key) {
            // Skip 2 bits (`vp56_rac_gets(c, 2)`).
            let _ = rac.get_bits(2);
            header_filter_info = header.filter_header != 0;
        } else {
            golden_frame_flag = rac.get_bit() != 0;
            if self.filter_header != 0 {
                let deblock_filtering = rac.get_bit() != 0;
                if deblock_filtering {
                    let _ = rac.get_bit();
                }
                if self.sub_version > 7 {
                    header_filter_info = rac.get_bit() != 0;
                }
            }
        }

        // Filter-info selection (carried through picture header).
        let vrt_shift: u32 = if matches!(header.kind, FrameKind::Key) && self.sub_version < 8 {
            5
        } else {
            0
        };
        let mut filter_mode = 0u8;
        let mut filter_selection = 16u8;
        if header_filter_info {
            if rac.get_bit() != 0 {
                filter_mode = 2;
                let _sample_variance_threshold = (rac.get_bits(5) as i32) << vrt_shift;
                let _max_vector_length = 2i32 << rac.get_bits(3);
            } else if rac.get_bit() != 0 {
                filter_mode = 1;
            }
            if self.sub_version > 7 {
                filter_selection = rac.get_bits(4) as u8;
            }
        }

        let use_huffman = rac.get_bit() != 0;
        if use_huffman {
            return Err(Error::unsupported(
                "VP6: huffman coefficient path not yet implemented",
            ));
        }

        // If coeff_offset is non-zero there's a second bool coder for
        // the coefficient partition. We build it but don't drive it
        // separately — FFmpeg's `vp6_parse_coeff` is a single function
        // that pulls from `s->ccp`. We implement that here by handing
        // in the right range coder.
        let coeff_offset_bytes = header.coeff_offset_bytes.max(0) as usize;

        // Keyframe / inter picture headers.
        if !matches!(header.kind, FrameKind::Key) {
            models::parse_mb_type_models(&mut self.model, &mut rac);
            models::parse_vector_models(&mut self.model, &mut rac);
        }
        models::parse_coeff_models(
            &mut self.model,
            &mut rac,
            self.sub_version,
            matches!(header.kind, FrameKind::Key),
        );

        if self.interlaced {
            let _il_prob = rac.get_bits(8);
        }

        let _ = filter_mode;
        let _ = filter_selection;
        let _ = golden_frame_flag;

        // Set up a secondary range coder for the coefficient partition
        // if present. VP6 splits coeff/mb data in some streams.
        let mut rac2_storage: Option<RangeCoder<'_>> = if coeff_offset_bytes > 0 {
            let coeff_start = rac_start + coeff_offset_bytes;
            if coeff_start >= data.len() {
                return Err(Error::invalid("VP6: coeff partition past end"));
            }
            Some(RangeCoder::new(&data[coeff_start..])?)
        } else {
            None
        };

        // Allocate output planes.
        let width = self.mb_width * 16;
        let height = self.mb_height * 16;
        let y_stride = width;
        let uv_stride = width / 2;
        let uv_h = height / 2;
        let mut y_plane = vec![0u8; y_stride * height];
        let mut u_plane = vec![128u8; uv_stride * uv_h];
        let mut v_plane = vec![128u8; uv_stride * uv_h];

        let plane_w = [width, width / 2, width / 2];
        let plane_h = [height, height / 2, height / 2];

        // Primary-pass DC-predictor / MV predictor state.
        self.scratch.reset_row(self.mb_width);

        let use_bicubic_luma = filter_mode != 0;

        for mb_row in 0..self.mb_height {
            self.scratch.start_row(self.mb_width);
            for mb_col in 0..self.mb_width {
                // MB-type + MV decode.
                let mb_type = if matches!(header.kind, FrameKind::Key) {
                    tables::Vp56Mb::Intra
                } else {
                    decode_mv(
                        &mut self.model,
                        &mut self.scratch,
                        &mut rac,
                        &self.macroblocks,
                        self.mb_width,
                        mb_row,
                        mb_col,
                    )
                };
                self.scratch.mb_type = mb_type;
                self.macroblocks[mb_row * self.mb_width + mb_col].mb_type = mb_type;

                // Coefficients.
                let coeff_rac: &mut RangeCoder<'_> = rac2_storage.as_mut().unwrap_or(&mut rac);
                if !mb::parse_coeff(&self.model, &mut self.scratch, coeff_rac) {
                    return Err(Error::invalid("VP6: coeff stream ended prematurely"));
                }

                // Prediction + reconstruction.
                let ref_kind: RefKind = tables::REFERENCE_FRAME[mb_type as usize].into();
                mb::add_predictors_dc(&mut self.scratch, ref_kind);

                match mb_type {
                    tables::Vp56Mb::Intra => {
                        mb::render_mb_intra(
                            &mut self.scratch,
                            &mut y_plane,
                            y_stride,
                            &mut u_plane,
                            uv_stride,
                            &mut v_plane,
                            uv_stride,
                            mb_row,
                            mb_col,
                        );
                    }
                    _ => {
                        let ref_planes = match ref_kind {
                            RefKind::Previous => self.prev_frame.as_ref(),
                            RefKind::Golden => self.golden_frame.as_ref(),
                            _ => None,
                        };
                        if let Some(rp) = ref_planes {
                            mb::render_mb_inter(
                                &mut self.scratch,
                                &mut y_plane,
                                y_stride,
                                &mut u_plane,
                                uv_stride,
                                &mut v_plane,
                                uv_stride,
                                &rp.y,
                                &rp.u,
                                &rp.v,
                                mb_row,
                                mb_col,
                                plane_w,
                                plane_h,
                                use_bicubic_luma,
                            );
                        } else {
                            // No valid reference — fall back to intra
                            // reconstruction (matches FFmpeg's conceal
                            // path).
                            mb::render_mb_intra(
                                &mut self.scratch,
                                &mut y_plane,
                                y_stride,
                                &mut u_plane,
                                uv_stride,
                                &mut v_plane,
                                uv_stride,
                                mb_row,
                                mb_col,
                            );
                        }
                    }
                }

                self.scratch.advance_column();
            }
        }

        // Build the output frame and update reference pools.
        let output = RefPlanes {
            y: y_plane.clone(),
            u: u_plane.clone(),
            v: v_plane.clone(),
        };
        if matches!(header.kind, FrameKind::Key) || golden_frame_flag {
            self.golden_frame = Some(output.clone());
        }
        self.prev_frame = Some(output);

        let frame = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: width as u32,
            height: height as u32,
            pts: self.pending_pts,
            time_base: self.pending_tb,
            planes: vec![
                VideoPlane {
                    stride: y_stride,
                    data: y_plane,
                },
                VideoPlane {
                    stride: uv_stride,
                    data: u_plane,
                },
                VideoPlane {
                    stride: uv_stride,
                    data: v_plane,
                },
            ],
        };
        Ok(frame)
    }
}

/// Port of `vp56_decode_mv` / `vp6_parse_vector_adjustment` + the MB
/// candidate predictor walk. Returns the decoded MB type and updates
/// `scratch.mv[..]` + `scratch.vector_candidate[..]`.
fn decode_mv(
    model: &mut Vp6Model,
    scratch: &mut BlockScratch,
    rac: &mut RangeCoder<'_>,
    macroblocks: &[MacroblockInfo],
    mb_width: usize,
    row: usize,
    col: usize,
) -> tables::Vp56Mb {
    let ctx = vector_predictors(
        scratch,
        macroblocks,
        mb_width,
        row,
        col,
        tables::RefFrame::Previous,
    );
    let mb_type = parse_mb_type(rac, model, scratch.mb_type, ctx as usize);

    match mb_type {
        tables::Vp56Mb::InterV1Pf => {
            let mv = scratch.vector_candidate[0];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::InterV2Pf => {
            let mv = scratch.vector_candidate[1];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::InterV1Gf => {
            vector_predictors(
                scratch,
                macroblocks,
                mb_width,
                row,
                col,
                tables::RefFrame::Golden,
            );
            let mv = scratch.vector_candidate[0];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::InterV2Gf => {
            vector_predictors(
                scratch,
                macroblocks,
                mb_width,
                row,
                col,
                tables::RefFrame::Golden,
            );
            let mv = scratch.vector_candidate[1];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::InterDeltaPf => {
            let mv = parse_vector_adjustment(rac, model, scratch);
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::InterDeltaGf => {
            vector_predictors(
                scratch,
                macroblocks,
                mb_width,
                row,
                col,
                tables::RefFrame::Golden,
            );
            let mv = parse_vector_adjustment(rac, model, scratch);
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
        }
        tables::Vp56Mb::Inter4V => {
            decode_4mv(rac, model, scratch);
        }
        _ => {
            for b in 0..6 {
                scratch.mv[b] = Mv::default();
            }
        }
    }

    mb_type
}

fn parse_mb_type(
    rac: &mut RangeCoder<'_>,
    model: &Vp6Model,
    prev_type: tables::Vp56Mb,
    ctx: usize,
) -> tables::Vp56Mb {
    let mb_type_model = &model.mb_type[ctx][prev_type as usize];
    if rac.get_prob(mb_type_model[0]) != 0 {
        prev_type
    } else {
        let v = rac.get_tree(tables::PMBT_TREE, mb_type_model);
        tables::Vp56Mb::from_u8(v as u8).unwrap_or(tables::Vp56Mb::InterNoVecPf)
    }
}

/// Port of `vp6_parse_vector_adjustment` — reads a per-component delta
/// and optionally applies its sign. Uses `vector_candidate[0]` as the
/// starting point when the candidate-position counter is <2.
fn parse_vector_adjustment(
    rac: &mut RangeCoder<'_>,
    model: &Vp6Model,
    scratch: &mut BlockScratch,
) -> Mv {
    let mut vect = Mv::default();
    if scratch.vector_candidate_pos < 2 {
        vect = scratch.vector_candidate[0];
    }
    for comp in 0..2usize {
        let mut delta = 0i32;
        if rac.get_prob(model.vector_dct[comp]) != 0 {
            const PROB_ORDER: [u8; 7] = [0, 1, 2, 7, 6, 5, 4];
            for &j in &PROB_ORDER {
                delta |= (rac.get_prob(model.vector_fdv[comp][j as usize]) as i32) << j;
            }
            if delta & 0xF0 != 0 {
                delta |= (rac.get_prob(model.vector_fdv[comp][3]) as i32) << 3;
            } else {
                delta |= 8;
            }
        } else {
            delta = rac.get_tree(tables::PVA_TREE, &model.vector_pdv[comp]);
        }

        if delta != 0 && rac.get_prob(model.vector_sig[comp]) != 0 {
            delta = -delta;
        }

        if comp == 0 {
            vect.x = vect.x.saturating_add(delta as i16);
        } else {
            vect.y = vect.y.saturating_add(delta as i16);
        }
    }
    vect
}

/// Port of `vp56_decode_4mv`.
fn decode_4mv(rac: &mut RangeCoder<'_>, model: &Vp6Model, scratch: &mut BlockScratch) {
    let mut types = [0u8; 4];
    for t in types.iter_mut() {
        let v = rac.get_bits(2) as u8;
        *t = if v != 0 { v + 1 } else { 0 };
    }
    let mut sum = (0i32, 0i32);
    for (b, &tt) in types.iter().enumerate() {
        let mv = match tt {
            0 => Mv::default(), // INTER_NOVEC_PF
            2 => parse_vector_adjustment(rac, model, scratch),
            3 => scratch.vector_candidate[0],
            4 => scratch.vector_candidate[1],
            _ => Mv::default(),
        };
        scratch.mv[b] = mv;
        sum.0 += mv.x as i32;
        sum.1 += mv.y as i32;
    }
    // Chroma MVs = shifted average luma MV. The round-shift from
    // FFmpeg (`RSHIFT(x, 2)`) is "(x + (1<<(n-1))) >> n" adjusted for
    // sign; we mirror that.
    let shifted = |v: i32| -> i16 {
        let r = if v >= 0 {
            (v + 2) >> 2
        } else {
            -(((-v) + 2) >> 2)
        };
        r as i16
    };
    scratch.mv[4] = Mv {
        x: shifted(sum.0),
        y: shifted(sum.1),
    };
    scratch.mv[5] = scratch.mv[4];
}

/// Port of `vp56_get_vectors_predictors`. Populates
/// `scratch.vector_candidate[0..=1]` from neighbouring-MB MVs that
/// match `ref_frame`. Returns the count — caller forwards this as the
/// `ctx` into the MB-type model.
fn vector_predictors(
    scratch: &mut BlockScratch,
    macroblocks: &[MacroblockInfo],
    mb_width: usize,
    row: usize,
    col: usize,
    ref_frame: tables::RefFrame,
) -> i32 {
    let mut nb_pred = 0i32;
    let mut candidates = [Mv::default(); 2];
    scratch.vector_candidate_pos = 0;

    for (pos, d) in tables::VP56_CANDIDATE_PREDICTOR_POS.iter().enumerate() {
        let nc = col as i32 + d[0] as i32;
        let nr = row as i32 + d[1] as i32;
        if nc < 0 || nc >= mb_width as i32 || nr < 0 || nr >= (macroblocks.len() / mb_width) as i32
        {
            continue;
        }
        let mb = macroblocks[nr as usize * mb_width + nc as usize];
        if tables::REFERENCE_FRAME[mb.mb_type as usize] != ref_frame {
            continue;
        }
        if (mb.mv.x == candidates[0].x && mb.mv.y == candidates[0].y)
            || (mb.mv.x == 0 && mb.mv.y == 0)
        {
            continue;
        }
        candidates[nb_pred as usize] = mb.mv;
        nb_pred += 1;
        if nb_pred > 1 {
            nb_pred = -1;
            break;
        }
        scratch.vector_candidate_pos = pos as i32;
    }

    scratch.vector_candidate[0] = candidates[0];
    scratch.vector_candidate[1] = candidates[1];
    nb_pred + 1
}

impl Decoder for Vp6Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        // FLV wraps VP6 frames with a 1-byte adjustment prefix; strip
        // it before decoding.
        let data = if packet.data.is_empty() {
            packet.data.as_slice()
        } else {
            &packet.data[1..]
        };
        let frame = self.decode_bytes(data)?;
        self.queued.push_back(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.queued.pop_front() {
            Some(v) => Ok(Frame::Video(v)),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.queued.clear();
        self.pending_pts = None;
        self.prev_frame = None;
        self.golden_frame = None;
        self.initialised = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_variant_errors() {
        let params = CodecParameters::video(CodecId::new("vp6a"));
        let mut dec = Vp6Decoder::new(params);
        let mut data = vec![0u8];
        data.extend_from_slice(&[0u8, 0, 0, 2, 12, 16, 12, 16, 0xFF, 0xFF, 0xFF, 0, 0]);
        let pkt = Packet::new(0, TimeBase::new(1, 1000), data);
        assert!(matches!(dec.send_packet(&pkt), Err(Error::Unsupported(_))));
    }

    #[test]
    fn inter_before_keyframe_errors() {
        let params = CodecParameters::video(CodecId::new("vp6f"));
        let mut dec = Vp6Decoder::new(params);
        // Inter frame (bit 7 set), sub_coeff=0, QP=10.
        let mut data = vec![0u8];
        let qp = 10u8;
        data.push((1 << 7) | ((qp << 1) & 0x7E));
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0]);
        let pkt = Packet::new(0, TimeBase::new(1, 1000), data);
        let res = dec.send_packet(&pkt);
        assert!(matches!(res, Err(Error::InvalidData(_))), "{res:?}");
    }
}

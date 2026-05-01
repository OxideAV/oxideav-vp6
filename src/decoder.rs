//! Streaming VP6 decoder, implementing
//! [`oxideav_core::Decoder`](oxideav_core::Decoder).
//!
//! This module ties together the pieces in `range_coder`, `models`,
//! `mb`, and `dsp` to produce a decoded `VideoFrame` per input packet.
//!
//! See FFmpeg's `vp56.c::ff_vp56_decode_frame` + `vp56_decode_mbs` for
//! the reference control flow.

use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::frame_header::{FrameHeader, FrameKind};
use crate::mb::{self, BlockScratch, MacroblockInfo, Mv, RefKind};
use crate::models::{self, Vp6Model};
use crate::range_coder::RangeCoder;
use crate::tables;

/// VP6 flavour. Tracks whether we're decoding the FLV-only `vp6f` (no
/// alpha) or `vp6a` (alpha plane prefixed).
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

/// Lightweight reference-frame plane holder. For the YUV stream all
/// three planes are populated; for the alpha stream only `y` is used
/// (the alpha data rides in the luma slot of a monochrome VP6 decode).
#[derive(Clone, Debug)]
pub(crate) struct RefPlanes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Per-stream VP6 decode state. An alpha-enabled packet carries two
/// independent streams (YUV + alpha); we keep their state in separate
/// [`Vp6Stream`] instances, matching FFmpeg's `s->alpha_context`.
#[derive(Debug)]
pub(crate) struct Vp6Stream {
    pub mb_width: usize,
    pub mb_height: usize,
    pub sub_version: u8,
    pub filter_header: u8,
    pub interlaced: bool,
    pub deblock_filtering: bool,
    pub model: Vp6Model,
    pub scratch: BlockScratch,
    pub prev_frame: Option<RefPlanes>,
    pub golden_frame: Option<RefPlanes>,
    pub macroblocks: Vec<MacroblockInfo>,
    pub initialised: bool,
}

impl Default for Vp6Stream {
    fn default() -> Self {
        Self {
            mb_width: 0,
            mb_height: 0,
            sub_version: 0,
            filter_header: 0,
            interlaced: false,
            deblock_filtering: true,
            model: Vp6Model::default(),
            scratch: BlockScratch::new(1),
            prev_frame: None,
            golden_frame: None,
            macroblocks: Vec::new(),
            initialised: false,
        }
    }
}

/// Output planes plus (width, height) from a single frame decode.
type DecodedPlanes = (Vec<u8>, Vec<u8>, Vec<u8>, usize, usize);

impl Vp6Stream {
    fn on_keyframe(&mut self, header: &FrameHeader) {
        self.mb_width = header.mb_width as usize;
        self.mb_height = header.mb_height as usize;
        self.sub_version = header.sub_version;
        self.filter_header = header.filter_header;
        self.interlaced = header.interlaced;
        self.scratch = BlockScratch::new(self.mb_width);
        self.model = Vp6Model::default();
        self.model.reset_defaults(self.interlaced, self.sub_version);
        self.macroblocks = vec![
            MacroblockInfo {
                mb_type: tables::Vp56Mb::Intra,
                mv: Mv::default(),
            };
            self.mb_width * self.mb_height
        ];
        self.initialised = true;
    }

    /// Decode a single VP6 frame's worth of bytes into three YUV planes
    /// (luma stride = `width`, chroma stride = `width/2`). The alpha
    /// stream calls this same routine — the `y` plane is the carrier
    /// for the alpha sample data (chroma bands are produced but unused).
    fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedPlanes> {
        let is_key = !data.is_empty() && (data[0] & 0x80) == 0;
        let header = if is_key {
            FrameHeader::parse(data)?
        } else if !self.initialised {
            return Err(Error::invalid("VP6: inter frame before keyframe"));
        } else {
            FrameHeader::parse_inter(data, self.filter_header, self.sub_version)?
        };
        if matches!(header.kind, FrameKind::Key) {
            self.on_keyframe(&header);
        }

        mb::init_dequant(&mut self.scratch, header.qp);
        self.scratch.keyframe = matches!(header.kind, FrameKind::Key);

        let rac_start = header.range_coder_offset;
        let rac_buf = &data[rac_start..];
        let mut rac = RangeCoder::new(rac_buf)?;
        let mut header_filter_info = false;

        let mut golden_frame_flag = false;
        if matches!(header.kind, FrameKind::Key) {
            let _ = rac.get_bits(2);
            header_filter_info = header.filter_header != 0;
        } else {
            golden_frame_flag = rac.get_bit() != 0;
            if self.filter_header != 0 {
                self.deblock_filtering = rac.get_bit() != 0;
                if self.deblock_filtering {
                    let _ = rac.get_bit();
                }
                if self.sub_version > 7 {
                    header_filter_info = rac.get_bit() != 0;
                }
            }
        }

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

        let coeff_offset_bytes = header.coeff_offset_bytes.max(0) as usize;

        if !matches!(header.kind, FrameKind::Key) {
            models::parse_mb_type_models(&mut self.model, &mut rac);
            models::parse_vector_models(&mut self.model, &mut rac);
            self.scratch.mb_type = tables::Vp56Mb::InterNoVecPf;
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

        let _ = filter_selection;
        let _ = golden_frame_flag;

        let mut rac2_storage: Option<RangeCoder<'_>> = if coeff_offset_bytes > 0 {
            let coeff_start = rac_start + coeff_offset_bytes;
            if coeff_start >= data.len() {
                return Err(Error::invalid("VP6: coeff partition past end"));
            }
            Some(RangeCoder::new(&data[coeff_start..])?)
        } else {
            None
        };

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

        self.scratch.reset_row(self.mb_width);

        let use_bicubic_luma = filter_mode != 0;

        for mb_row in 0..self.mb_height {
            self.scratch.start_row(self.mb_width);
            for mb_col in 0..self.mb_width {
                let (mb_type, stored_mv) = if matches!(header.kind, FrameKind::Key) {
                    (tables::Vp56Mb::Intra, Mv::default())
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
                let cell = &mut self.macroblocks[mb_row * self.mb_width + mb_col];
                cell.mb_type = mb_type;
                cell.mv = stored_mv;

                let coeff_rac: &mut RangeCoder<'_> = rac2_storage.as_mut().unwrap_or(&mut rac);
                if !mb::parse_coeff(&self.model, &mut self.scratch, coeff_rac) {
                    return Err(Error::invalid(format!(
                        "VP6: coeff stream ended prematurely at mb ({mb_row},{mb_col})"
                    )));
                }

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
                                self.deblock_filtering,
                            );
                        } else {
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

        let output = RefPlanes {
            y: y_plane.clone(),
            u: u_plane.clone(),
            v: v_plane.clone(),
        };
        if matches!(header.kind, FrameKind::Key) || golden_frame_flag {
            self.golden_frame = Some(output.clone());
        }
        self.prev_frame = Some(output);

        Ok((y_plane, u_plane, v_plane, width, height))
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

    /// Primary YUV decode context.
    stream: Vp6Stream,
    /// Secondary decode context used by `vp6a` for the alpha plane.
    alpha_stream: Vp6Stream,
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
            stream: Vp6Stream::default(),
            alpha_stream: Vp6Stream::default(),
        }
    }

    fn decode_bytes(&mut self, data: &[u8]) -> Result<VideoFrame> {
        // vp6a prefix: 3-byte BE24 offset to the alpha partition. The
        // bytes after the prefix up to that offset are the primary YUV
        // bitstream; the remainder is a second, independently-coded
        // VP6 bitstream carrying the alpha plane.
        let (yuv_data, alpha_data) = if matches!(self.variant, Vp6Variant::FlvAlpha) {
            if data.len() < 3 {
                return Err(Error::invalid(
                    "VP6 vp6a: packet too short for alpha offset",
                ));
            }
            let alpha_offset =
                (((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32)) as usize;
            let rest = &data[3..];
            if alpha_offset > rest.len() {
                return Err(Error::invalid("VP6 vp6a: alpha offset past end"));
            }
            (&rest[..alpha_offset], Some(&rest[alpha_offset..]))
        } else {
            (data, None)
        };

        let (y_plane, u_plane, v_plane, width, height) = self.stream.decode_frame(yuv_data)?;
        self.width = Some(width as u32);
        self.height = Some(height as u32);
        let y_stride = width;
        let uv_stride = width / 2;

        if let Some(alpha_bytes) = alpha_data {
            // The alpha stream is a standalone monochrome VP6. Its luma
            // plane is the alpha sample data; chroma is discarded.
            let (alpha_plane, _, _, aw, ah) = self.alpha_stream.decode_frame(alpha_bytes)?;
            if aw != width || ah != height {
                return Err(Error::invalid(
                    "VP6 vp6a: alpha dimensions disagree with YUV",
                ));
            }
            let frame = VideoFrame {
                pts: self.pending_pts,
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
                    VideoPlane {
                        stride: y_stride,
                        data: alpha_plane,
                    },
                ],
            };
            Ok(frame)
        } else {
            let frame = VideoFrame {
                pts: self.pending_pts,
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
}

/// Port of `vp56_decode_mv` / `vp6_parse_vector_adjustment` + the MB
/// candidate predictor walk. Returns the decoded MB type and the MV to
/// stash into the persistent `macroblocks[]` slot.
fn decode_mv(
    model: &mut Vp6Model,
    scratch: &mut BlockScratch,
    rac: &mut RangeCoder<'_>,
    macroblocks: &[MacroblockInfo],
    mb_width: usize,
    row: usize,
    col: usize,
) -> (tables::Vp56Mb, Mv) {
    let ctx = vector_predictors(
        scratch,
        macroblocks,
        mb_width,
        row,
        col,
        tables::RefFrame::Previous,
    );
    let mb_type = parse_mb_type(rac, model, scratch.mb_type, ctx as usize);

    // The MV stored back into `macroblocks[row*mb_width + col].mv` drives
    // future MV-candidate lookups from neighbouring MBs. FFmpeg's
    // `vp56_decode_mv` assigns `*mv` into that slot at function exit
    // (and the 4V branch stores `s->mv[3]` explicitly). Mirror that here.
    let stored_mv = match mb_type {
        tables::Vp56Mb::InterV1Pf => {
            let mv = scratch.vector_candidate[0];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
            mv
        }
        tables::Vp56Mb::InterV2Pf => {
            let mv = scratch.vector_candidate[1];
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
            mv
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
            mv
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
            mv
        }
        tables::Vp56Mb::InterDeltaPf => {
            let mv = parse_vector_adjustment(rac, model, scratch);
            for b in 0..6 {
                scratch.mv[b] = mv;
            }
            mv
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
            mv
        }
        tables::Vp56Mb::Inter4V => {
            decode_4mv(rac, model, scratch);
            // FFmpeg stores block 3's MV for the whole-MB predictor.
            scratch.mv[3]
        }
        _ => {
            for b in 0..6 {
                scratch.mv[b] = Mv::default();
            }
            Mv::default()
        }
    };

    (mb_type, stored_mv)
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
            vect.x = vect.x.wrapping_add(delta as i16);
        } else {
            vect.y = vect.y.wrapping_add(delta as i16);
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
/// match `ref_frame`. Returns the MB-type ctx (0..=2) per spec page 28
/// Table 5: `(both Nearest+Near exist => ctx 0, only Nearest => ctx 1,
/// neither => ctx 2)`.  In terms of `nb_pred` the canonical mapping is
/// `ctx = 2 - nb_pred`.
///
/// `nb_pred` is the count of distinct non-(0,0) candidate MVs found
/// while walking [`tables::VP56_CANDIDATE_PREDICTOR_POS`]; on the third
/// hit we early-exit (treated as "both exist") so the value passed to
/// the subtraction is clamped to `0..=2` already.
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
            // Third hit observed: this matches spec "Nearest & Near MVs
            // both exist" (ctx = 0). Encode that here as `nb_pred = 2`
            // so the `2 - nb_pred` mapping below produces ctx 0.
            nb_pred = 2;
            break;
        }
        scratch.vector_candidate_pos = pos as i32;
    }

    scratch.vector_candidate[0] = candidates[0];
    scratch.vector_candidate[1] = candidates[1];
    // Spec page 28 Table 5: 0 cands -> ctx 2, 1 cand -> ctx 1, 2+ -> ctx 0.
    2 - nb_pred
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
        self.stream = Vp6Stream::default();
        self.alpha_stream = Vp6Stream::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// r22: pin `vector_predictors` to spec page 28 Table 5 mapping —
    /// `(0 cands -> ctx 2, 1 cand -> ctx 1, 2+ cands -> ctx 0)`.
    /// A regression of `nb_pred + 1` (the pre-r22 value) would surface
    /// as ctx=1 for a top-left MB whose neighbours are all OOB / zero-MV,
    /// disagreeing with ffmpeg's `mb_type[ctx][...]` indexing.
    #[test]
    fn vector_predictors_ctx_mapping_matches_spec() {
        // Top-left MB in an empty 4x2 grid: every candidate position
        // walks off-frame, so nb_pred stays 0 and the spec ctx is 2.
        let mb_width = 4usize;
        let mb_height = 2usize;
        let mut scratch = mb::BlockScratch::new(mb_width);
        let macroblocks = vec![MacroblockInfo::default(); mb_width * mb_height];
        let ctx = vector_predictors(
            &mut scratch,
            &macroblocks,
            mb_width,
            0,
            0,
            tables::RefFrame::Previous,
        );
        assert_eq!(ctx, 2, "nb_pred=0 must map to ctx 2 (spec p.28 Table 5)");

        // Seed a single non-(0,0) Previous-ref neighbour at (-1,0) so
        // exactly one distinct candidate is found → ctx 1.
        let mut macroblocks = vec![MacroblockInfo::default(); mb_width * mb_height];
        macroblocks[0 * mb_width + 0] = MacroblockInfo {
            mb_type: tables::Vp56Mb::InterDeltaPf,
            mv: Mv { x: 4, y: 0 },
        };
        let ctx = vector_predictors(
            &mut scratch,
            &macroblocks,
            mb_width,
            1,
            0,
            tables::RefFrame::Previous,
        );
        assert_eq!(ctx, 1, "nb_pred=1 must map to ctx 1 (spec p.28 Table 5)");
    }

    #[test]
    fn alpha_packet_too_short_for_offset_prefix() {
        // Without enough bytes for the 3-byte alpha offset prefix, the
        // decoder reports InvalidData (not Unsupported).
        let params = CodecParameters::video(CodecId::new("vp6a"));
        let mut dec = Vp6Decoder::new(params);
        // 1-byte flv prefix + 2 body bytes → after strip, len=2, short
        // for alpha prefix.
        let pkt = Packet::new(0, TimeBase::new(1, 1000), vec![0u8, 0, 0]);
        let res = dec.send_packet(&pkt);
        assert!(matches!(res, Err(Error::InvalidData(_))), "{res:?}");
    }
}

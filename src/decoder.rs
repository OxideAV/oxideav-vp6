//! Streaming VP6 decoder, implementing
//! [`oxideav_core::Decoder`](oxideav_core::Decoder).
//!
//! This module ties together the pieces in `range_coder`, `models`,
//! `mb`, and `dsp` to produce a decoded `VideoFrame` per input packet.
//!
//! See FFmpeg's `vp56.c::ff_vp56_decode_frame` + `vp56_decode_mbs` for
//! the reference control flow.
//!
//! ## DoS-protection — round 24+
//!
//! The decoder honours [`oxideav_core::DecoderLimits`]:
//!
//! - **Header-parse pixel cap.** [`Vp6Decoder::with_limits`] threads
//!   [`DecoderLimits::max_pixels_per_frame`] into the keyframe path;
//!   any keyframe whose declared `mb_width × mb_height × 256` exceeds
//!   the cap is rejected with [`Error::ResourceExhausted`] *before*
//!   any plane is allocated.
//!
//! - **Arena-backed planes (true zero-copy).** The decoder owns an
//!   `Arc<arena::sync::ArenaPool>` sized at construction. The
//!   per-decode pipeline is **lazy**: `send_packet` only queues raw
//!   bytes; the actual pixel decode runs when `receive_frame` /
//!   `receive_arena_frame` is called. The latter leases one arena from
//!   the pool, allocates Y/U/V (and optional alpha) directly inside
//!   it, and runs every MB-render kernel against those arena slices —
//!   no intermediate `Vec<u8>` and no memcpy at the API boundary. The
//!   reference frames kept for inter-MC are still heap-owned (one
//!   internal memcpy per decode); they live longer than any single
//!   arena lease.

use std::collections::VecDeque;
use std::sync::Arc;

use oxideav_core::arena::sync::{ArenaPool, FrameHeader, FrameInner};
use oxideav_core::format::PixelFormat;
use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, DecoderLimits, Error, Frame, Packet, Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::frame_header::{FrameHeader as Vp6FrameHeader, FrameKind};
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

/// Per-decode geometry. Filled in by [`Vp6Stream::peek_dims`] before any
/// arena allocation happens, so the arena byte count is known up front.
#[derive(Clone, Copy, Debug)]
struct DecodeDims {
    /// Width in pixels (luma). Always `mb_width * 16`.
    width: usize,
    /// Height in pixels (luma). Always `mb_height * 16`.
    height: usize,
    /// Luma plane size in bytes.
    y_bytes: usize,
    /// Chroma (Cb / Cr) plane size in bytes.
    c_bytes: usize,
}

impl Vp6Stream {
    fn on_keyframe(&mut self, header: &Vp6FrameHeader) {
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

    /// Peek the dims this packet will produce **without** advancing any
    /// stream state. Used by the arena path so it can compute the byte
    /// count it needs from the pool before leasing.
    fn peek_dims(&self, data: &[u8]) -> Result<DecodeDims> {
        let is_key = !data.is_empty() && (data[0] & 0x80) == 0;
        if is_key {
            let hdr = Vp6FrameHeader::parse(data)?;
            let width = hdr.mb_width as usize * 16;
            let height = hdr.mb_height as usize * 16;
            Ok(DecodeDims {
                width,
                height,
                y_bytes: width * height,
                c_bytes: (width / 2) * (height / 2),
            })
        } else if !self.initialised {
            Err(Error::invalid("VP6: inter frame before keyframe"))
        } else {
            let width = self.mb_width * 16;
            let height = self.mb_height * 16;
            Ok(DecodeDims {
                width,
                height,
                y_bytes: width * height,
                c_bytes: (width / 2) * (height / 2),
            })
        }
    }

    /// Decode a single VP6 frame's worth of bytes directly into the
    /// caller-supplied YUV plane slices (luma stride = `width`, chroma
    /// stride = `width/2`). The alpha stream calls this same routine —
    /// the `y_plane` slot carries the alpha sample data (chroma bands
    /// are produced but unused).
    ///
    /// `y_plane`, `u_plane`, `v_plane` MUST have lengths
    /// `width*height`, `(width/2)*(height/2)`, `(width/2)*(height/2)`
    /// respectively, where (width, height) = the dims this packet
    /// implies (querable via [`Self::peek_dims`]).
    fn decode_frame_into(
        &mut self,
        data: &[u8],
        y_plane: &mut [u8],
        u_plane: &mut [u8],
        v_plane: &mut [u8],
    ) -> Result<DecodeDims> {
        let is_key = !data.is_empty() && (data[0] & 0x80) == 0;
        let header = if is_key {
            Vp6FrameHeader::parse(data)?
        } else if !self.initialised {
            return Err(Error::invalid("VP6: inter frame before keyframe"));
        } else {
            Vp6FrameHeader::parse_inter(data, self.filter_header, self.sub_version)?
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

        // Caller-supplied buffers must have been sized to the dims we
        // implied via `peek_dims`. Sanity-check; this catches a
        // mis-wired arena allocation rather than corrupting memory.
        debug_assert_eq!(y_plane.len(), y_stride * height);
        debug_assert_eq!(u_plane.len(), uv_stride * uv_h);
        debug_assert_eq!(v_plane.len(), uv_stride * uv_h);

        // Fresh start: VP6 paints every MB so seeding with 0 (Y) / 128
        // (chroma) matches the prior `vec![]`-initialised behaviour.
        // The arena buffer comes back uninitialised from the pool's
        // bump allocator, so we must wipe it before any conditional
        // skipped-MB path could read pre-existing bytes.
        for px in y_plane.iter_mut() {
            *px = 0;
        }
        for px in u_plane.iter_mut() {
            *px = 128;
        }
        for px in v_plane.iter_mut() {
            *px = 128;
        }

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
                            y_plane,
                            y_stride,
                            u_plane,
                            uv_stride,
                            v_plane,
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
                                y_plane,
                                y_stride,
                                u_plane,
                                uv_stride,
                                v_plane,
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
                                y_plane,
                                y_stride,
                                u_plane,
                                uv_stride,
                                v_plane,
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

        // Update reference frames. The arena buffer that backs the
        // output planes goes back to the pool when the caller drops
        // the returned Frame, so we must memcpy here into a heap-owned
        // RefPlanes that survives across packets.
        let output = RefPlanes {
            y: y_plane.to_vec(),
            u: u_plane.to_vec(),
            v: v_plane.to_vec(),
        };
        if matches!(header.kind, FrameKind::Key) || golden_frame_flag {
            self.golden_frame = Some(output.clone());
        }
        self.prev_frame = Some(output);

        Ok(DecodeDims {
            width,
            height,
            y_bytes: y_stride * height,
            c_bytes: uv_stride * uv_h,
        })
    }
}

/// Per-codec ceiling for arena-backed plane allocations. VP6 in the
/// wild rarely tops 1280×720 (HD); 8 MiB is enough for one I420 plane
/// triple at that size (1280×720×1.5 = ~1.4 MiB) plus generous
/// headroom for a 4-plane vp6a frame at up to ~1920×1080. Keeps
/// `max_arenas_in_flight × cap_per_arena` sane even when the caller
/// hands in a huge `max_alloc_bytes_per_frame`.
pub const DEFAULT_VP6_ARENA_BYTES: u64 = 8 * 1024 * 1024;

/// Streaming VP6 decoder.
///
/// Construct via [`Vp6Decoder::new`] (default DoS limits) or
/// [`Vp6Decoder::with_limits`] (explicit caps for server / sandbox
/// callers). The `Decoder` trait's [`receive_arena_frame`] override
/// returns true zero-copy frames whose plane bytes live inside the
/// pool's arena buffer — see the module docs for the lazy-decode
/// pipeline.
///
/// [`receive_arena_frame`]: Decoder::receive_arena_frame
pub struct Vp6Decoder {
    codec_id: CodecId,
    variant: Vp6Variant,
    /// Raw packet bytes awaiting decode. The decoder is **lazy**:
    /// `send_packet` only enqueues; the actual decode runs from
    /// `receive_frame` / `receive_arena_frame`. This keeps the arena
    /// pool short-lived (one slot held only while a Frame clone exists)
    /// and lets `receive_arena_frame` write pixels straight into the
    /// arena with no intermediate `Vec<u8>`.
    pending: VecDeque<(Vec<u8>, Option<i64>, TimeBase)>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    width: Option<u32>,
    height: Option<u32>,

    /// Primary YUV decode context.
    stream: Vp6Stream,
    /// Secondary decode context used by `vp6a` for the alpha plane.
    alpha_stream: Vp6Stream,

    /// DoS-protection caps threaded from the caller's
    /// [`DecoderLimits`]. The header-parse path consults
    /// `max_pixels_per_frame`; the pool below is sized from
    /// `max_arenas_in_flight × min(max_alloc_bytes_per_frame,
    /// DEFAULT_VP6_ARENA_BYTES)`.
    limits: DecoderLimits,
    /// Bounded buffer pool for arena-backed frames. When every slot
    /// is checked out, [`ArenaPool::lease`] returns
    /// [`Error::ResourceExhausted`] — natural backpressure for the
    /// `receive_arena_frame` path.
    arena_pool: Arc<ArenaPool>,
}

impl std::fmt::Debug for Vp6Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp6Decoder")
            .field("codec_id", &self.codec_id)
            .field("variant", &self.variant)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pending", &self.pending.len())
            .field("limits", &self.limits)
            .finish()
    }
}

impl Vp6Decoder {
    /// Construct a decoder for the given codec id with default
    /// [`DecoderLimits`]. Shorthand for
    /// [`Vp6Decoder::with_limits(codec_id, DecoderLimits::default())`].
    pub fn new(codec_id: CodecId) -> Self {
        Self::with_limits(codec_id, DecoderLimits::default())
    }

    /// Construct a decoder with explicit DoS-protection caps. The
    /// per-arena byte cap is `min(limits.max_alloc_bytes_per_frame,
    /// DEFAULT_VP6_ARENA_BYTES)` — the caller's global cap is
    /// honoured but never exceeds the VP6-specific ceiling, so the
    /// pool's resident memory stays bounded even with a generous
    /// global default.
    pub fn with_limits(codec_id: CodecId, limits: DecoderLimits) -> Self {
        let variant = Vp6Variant::from_codec_id(&codec_id);
        let cap_per_arena = (limits
            .max_alloc_bytes_per_frame
            .min(DEFAULT_VP6_ARENA_BYTES)) as usize;
        let pool = ArenaPool::with_alloc_count_cap(
            limits.max_arenas_in_flight as usize,
            cap_per_arena,
            limits.max_alloc_count_per_frame,
        );
        Self {
            codec_id,
            variant,
            pending: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            width: None,
            height: None,
            stream: Vp6Stream::default(),
            alpha_stream: Vp6Stream::default(),
            limits,
            arena_pool: pool,
        }
    }

    /// Borrow the in-flight DoS limits for this decoder.
    pub fn limits(&self) -> &DecoderLimits {
        &self.limits
    }

    /// Borrow the arena pool. Tests probe pool state directly (e.g.
    /// lease N+1 arenas to exercise [`Error::ResourceExhausted`]).
    pub fn arena_pool(&self) -> &Arc<ArenaPool> {
        &self.arena_pool
    }

    /// Split a packet payload into `(yuv_bytes, alpha_bytes_opt)` per
    /// the vp6a 3-byte BE24 alpha-offset prefix. For vp6f the input
    /// passes through untouched.
    fn split_alpha<'a>(&self, data: &'a [u8]) -> Result<(&'a [u8], Option<&'a [u8]>)> {
        if matches!(self.variant, Vp6Variant::FlvAlpha) {
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
            Ok((&rest[..alpha_offset], Some(&rest[alpha_offset..])))
        } else {
            Ok((data, None))
        }
    }

    /// Parse just enough of the next pending packet's header to compute
    /// the byte count its planes will need, and reject it early when
    /// the declared dims exceed [`DecoderLimits::max_pixels_per_frame`].
    fn check_pending_dims(&self, data: &[u8]) -> Result<(DecodeDims, Option<DecodeDims>)> {
        let (yuv_bytes, alpha_bytes) = self.split_alpha(data)?;
        let yuv_dims = self.stream.peek_dims(yuv_bytes)?;
        let pixels = (yuv_dims.width as u64).saturating_mul(yuv_dims.height as u64);
        if pixels > self.limits.max_pixels_per_frame {
            return Err(Error::resource_exhausted(format!(
                "VP6 frame {}x{} exceeds DecoderLimits.max_pixels_per_frame={}",
                yuv_dims.width, yuv_dims.height, self.limits.max_pixels_per_frame
            )));
        }
        let alpha_dims = if let Some(ab) = alpha_bytes {
            let ad = self.alpha_stream.peek_dims(ab)?;
            // Alpha dims must agree with YUV dims; surface the mismatch
            // as InvalidData rather than ResourceExhausted.
            if ad.width != yuv_dims.width || ad.height != yuv_dims.height {
                return Err(Error::invalid(
                    "VP6 vp6a: alpha dimensions disagree with YUV",
                ));
            }
            Some(ad)
        } else {
            None
        };
        Ok((yuv_dims, alpha_dims))
    }

    /// Decode the next pending packet straight into the supplied arena
    /// and return a `(arena::sync::Frame, pts)` pair. The arena is
    /// leased internally; failure to lease surfaces as
    /// [`Error::ResourceExhausted`].
    fn decode_into_arena(&mut self) -> Result<oxideav_core::arena::sync::Frame> {
        let (data, pts, _tb) = self
            .pending
            .pop_front()
            .ok_or_else(|| Error::Other("VP6: no pending packet".into()))?;
        // Header-parse + DoS check before leasing anything.
        let (yuv_dims, alpha_dims) = self.check_pending_dims(&data)?;
        // Total bytes we'll need from the arena, before lease.
        let plane_bytes = yuv_dims
            .y_bytes
            .checked_add(yuv_dims.c_bytes)
            .and_then(|n| n.checked_add(yuv_dims.c_bytes))
            .ok_or_else(|| Error::resource_exhausted("VP6 frame size overflow".to_string()))?;
        let total = if let Some(ad) = alpha_dims {
            plane_bytes
                .checked_add(ad.y_bytes)
                .ok_or_else(|| Error::resource_exhausted("VP6+alpha size overflow".to_string()))?
        } else {
            plane_bytes
        };
        if total > self.arena_pool.cap_per_arena() {
            return Err(Error::resource_exhausted(format!(
                "VP6 frame {}x{} needs {} arena bytes (cap {})",
                yuv_dims.width,
                yuv_dims.height,
                total,
                self.arena_pool.cap_per_arena()
            )));
        }

        let arena = self.arena_pool.lease()?;

        // Allocate the planes from the arena, recording their offsets
        // for the FrameInner plane table. The bump allocator hands out
        // contiguous regions; the offsets are exactly y_bytes, y+c,
        // y+2c (and optionally y+2c+y for alpha).
        let y_off = arena.used();
        let y_buf = arena.alloc::<u8>(yuv_dims.y_bytes)?;
        // SAFETY: we need to re-borrow these into independent &mut
        // slices simultaneously (one Y, one U, one V) for the kernel
        // call. Each `arena.alloc` returns a disjoint region, so the
        // pointers are non-overlapping. We split via raw pointers and
        // reconstitute mutable slices over distinct ranges.
        let y_ptr = y_buf.as_mut_ptr();
        let y_len = y_buf.len();

        let u_off = arena.used();
        let u_buf = arena.alloc::<u8>(yuv_dims.c_bytes)?;
        let u_ptr = u_buf.as_mut_ptr();
        let u_len = u_buf.len();

        let v_off = arena.used();
        let v_buf = arena.alloc::<u8>(yuv_dims.c_bytes)?;
        let v_ptr = v_buf.as_mut_ptr();
        let v_len = v_buf.len();

        // Optional alpha plane (vp6a). Allocated up front so the offset
        // is stable; the bytes are filled by a second decode_frame_into
        // below.
        let (alpha_off, alpha_len, alpha_ptr) = if let Some(ad) = alpha_dims {
            let off = arena.used();
            let buf = arena.alloc::<u8>(ad.y_bytes)?;
            let ptr = buf.as_mut_ptr();
            (Some(off), Some(buf.len()), Some(ptr))
        } else {
            (None, None, None)
        };

        // Re-split data now that the arena has the buffers it needs.
        // `split_alpha` already validated this earlier; re-using the
        // same parse here is cheap and keeps the lifetimes simple.
        let (yuv_bytes, alpha_bytes) = self.split_alpha(&data)?;

        // SAFETY: the three pointers come from disjoint
        // arena.alloc<u8>() calls; their backing regions never overlap.
        // We borrow as mutable slices for the duration of the kernel
        // call only (no other alias outstanding).
        let dims = unsafe {
            let y_slice = std::slice::from_raw_parts_mut(y_ptr, y_len);
            let u_slice = std::slice::from_raw_parts_mut(u_ptr, u_len);
            let v_slice = std::slice::from_raw_parts_mut(v_ptr, v_len);
            self.stream
                .decode_frame_into(yuv_bytes, y_slice, u_slice, v_slice)?
        };
        self.width = Some(dims.width as u32);
        self.height = Some(dims.height as u32);

        // Alpha plane (vp6a only). We treat its luma plane as the
        // alpha sample data and ignore the chroma bands; the
        // alpha_stream's chroma scratch goes into a small temp Vec
        // here (chroma is only used internally for the alpha decode's
        // MC, not surfaced to the caller).
        if let (Some(ab), Some(_off), Some(alen), Some(aptr), Some(ad)) =
            (alpha_bytes, alpha_off, alpha_len, alpha_ptr, alpha_dims)
        {
            // Scratch chroma planes for the alpha-stream decode. These
            // never reach the caller (the alpha plane is the alpha
            // stream's *luma*); allocate from the heap rather than the
            // arena so we don't waste arena capacity on bytes the
            // Frame won't expose.
            let mut alpha_u_scratch = vec![128u8; ad.c_bytes];
            let mut alpha_v_scratch = vec![128u8; ad.c_bytes];
            // SAFETY: alpha_ptr came from a disjoint arena.alloc<u8>()
            // call (the only other arena allocs were Y / U / V above).
            let alpha_dims_actual = unsafe {
                let alpha_slice = std::slice::from_raw_parts_mut(aptr, alen);
                self.alpha_stream.decode_frame_into(
                    ab,
                    alpha_slice,
                    &mut alpha_u_scratch,
                    &mut alpha_v_scratch,
                )?
            };
            if alpha_dims_actual.width != dims.width || alpha_dims_actual.height != dims.height {
                return Err(Error::invalid(
                    "VP6 vp6a: alpha dimensions disagree with YUV",
                ));
            }
        }

        // Build the FrameHeader (pixel format depends on alpha
        // presence) and freeze into a Frame.
        let pixel_format = if alpha_off.is_some() {
            PixelFormat::Yuva420P
        } else {
            PixelFormat::Yuv420P
        };
        let header = FrameHeader::new(dims.width as u32, dims.height as u32, pixel_format, pts);

        let mut planes: Vec<(usize, usize)> = Vec::with_capacity(4);
        planes.push((y_off, y_len));
        planes.push((u_off, u_len));
        planes.push((v_off, v_len));
        if let (Some(off), Some(len)) = (alpha_off, alpha_len) {
            planes.push((off, len));
        }
        FrameInner::new(arena, &planes, header)
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
        // it before queueing.
        let data = if packet.data.is_empty() {
            packet.data.clone()
        } else {
            packet.data[1..].to_vec()
        };
        // DoS check fires before queueing so a malformed-size packet is
        // rejected at send_packet rather than sitting in the queue
        // until receive_*. We don't lease anything here — just parse
        // the header bytes that name the dims.
        let _ = self.check_pending_dims(&data)?;
        self.pending.push_back((data, packet.pts, packet.time_base));
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.pending.is_empty() {
            return Err(Error::NeedMore);
        }
        // Decode the next pending packet through the arena pipeline,
        // then materialise a heap-owned VideoFrame from it. The arena
        // lease is released when `arena_frame` is dropped at the end
        // of this scope — the legacy `receive_frame` path therefore
        // never holds a pool slot across calls.
        let arena_frame = self.decode_into_arena()?;
        let video = arena_frame_to_video_frame(&arena_frame, self.pending_pts);
        drop(arena_frame);
        Ok(Frame::Video(video))
    }

    fn receive_arena_frame(&mut self) -> Result<oxideav_core::arena::sync::Frame> {
        if self.pending.is_empty() {
            return Err(Error::NeedMore);
        }
        // True zero-copy: `decode_into_arena` writes pixels straight
        // into the arena buffer; we hand the resulting Frame back
        // unchanged. The pool slot stays checked out until the caller
        // drops the last `Arc` clone of the returned Frame.
        self.decode_into_arena()
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.pending.clear();
        self.pending_pts = None;
        self.stream = Vp6Stream::default();
        self.alpha_stream = Vp6Stream::default();
        Ok(())
    }
}

/// Materialise a stride-packed `VideoFrame` from an arena Frame. Walks
/// the plane table and copies each plane via `to_vec()` — used by the
/// legacy [`Decoder::receive_frame`] path to surface heap-owned bytes
/// across `Send` boundaries that don't want the arena lifetime tied to
/// the consumer's frame lifetime.
fn arena_frame_to_video_frame(
    af: &oxideav_core::arena::sync::Frame,
    pts: Option<i64>,
) -> VideoFrame {
    let hdr = af.header();
    let w = hdr.width as usize;
    let cw = w.div_ceil(2);
    let mut planes = Vec::with_capacity(af.plane_count());
    // Plane 0: luma (full width).
    if let Some(p) = af.plane(0) {
        planes.push(VideoPlane {
            stride: w,
            data: p.to_vec(),
        });
    }
    // Planes 1, 2: chroma (half width).
    for i in 1..af.plane_count().min(3) {
        if let Some(p) = af.plane(i) {
            planes.push(VideoPlane {
                stride: cw,
                data: p.to_vec(),
            });
        }
    }
    // Plane 3 (optional): alpha (full width again).
    if af.plane_count() >= 4 {
        if let Some(p) = af.plane(3) {
            planes.push(VideoPlane {
                stride: w,
                data: p.to_vec(),
            });
        }
    }
    VideoFrame { pts, planes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inter_before_keyframe_errors() {
        let mut dec = Vp6Decoder::new(CodecId::new("vp6f"));
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
        let mut dec = Vp6Decoder::new(CodecId::new("vp6a"));
        // 1-byte flv prefix + 2 body bytes → after strip, len=2, short
        // for alpha prefix.
        let pkt = Packet::new(0, TimeBase::new(1, 1000), vec![0u8, 0, 0]);
        let res = dec.send_packet(&pkt);
        assert!(matches!(res, Err(Error::InvalidData(_))), "{res:?}");
    }
}

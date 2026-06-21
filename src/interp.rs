//! VP6 fractional-pixel motion-compensation interpolation (spec §11.4).
//!
//! VP6 supports fractional-pixel motion compensation: a motion vector
//! may point between integer sample positions, in which case the
//! prediction samples are produced by *interpolating* the reference
//! reconstruction buffer. The spec interpolates to ¼-sample precision in
//! luma and ⅛-sample precision in chroma (§11.4). Two filter families
//! are defined:
//!
//! * **Bilinear** — 2-tap filters (§11.4.1). The only family permitted
//!   in Simple Profile; also selectable in Advanced Profile.
//! * **Bicubic** — 4-tap filters (§11.4.2). VP6.1 / VP6.2 only,
//!   selectable per-block in Advanced Profile.
//!
//! This stage is **independent of the BoolCoder** (spec §7.3): given a
//! reference-buffer pointer, a stride and a fractional offset, the filter
//! kernels are pure integer pixel arithmetic. They never call
//! `VP6_DecodeBool`. Like the §15 dequant, §16 IDCT and §17.1 intra
//! reconstruction layers, this module therefore advances the decoder
//! without touching the contested §7.3 `Split` formula (see the
//! crate-root `DOCS-GAP` block). What *is* BoolCoder-gated is the motion
//! vector that selects which filter phase to apply and where to source
//! the samples — but that is upstream of this module, which only does the
//! arithmetic once the phase and source are known.
//!
//! All tap tables and the convolution / separable-pass structure in this
//! file are transcribed verbatim from `docs/video/vp6/vp6_format.pdf`
//! §11.4 (On2 Technologies, document version 1.02, August 2006). No
//! external library code was consulted.
//!
//! ## Fixed-point structure (spec §11.4)
//!
//! Every filter tap set sums to 128, so each filtered point is
//! `(Σ SrcPtr[k]·Filter[k] + 64) >> 7` — a multiply-accumulate, a
//! rounding bias of 64 (half of `1<<7`), and a 7-bit descale. The `+64`
//! makes the `>> 7` round-to-nearest rather than truncate. The bicubic
//! kernel clips its output to `0..=255` per the §11.4.2 pseudocode; the
//! bilinear §11.4.1 pseudocode shows the multiply-accumulate-descale
//! without an explicit per-point clip (its taps are both non-negative so
//! a single pass cannot leave `0..=255` for `0..=255` inputs — see
//! [`bilinear_point`]).
//!
//! ## Separable two-pass application (spec §11.4)
//!
//! > In cases where the motion vector has a fractional component in both
//! > x and y, an intermediate result is calculated by applying the filter
//! > in the x direction (horizontally). This intermediate result used as
//! > input to a second pass which filters in the y direction (vertically)
//! > to produce the final 2-d filtered output.
//!
//! The `PixelStep` argument selects the pass direction: `1` for the
//! horizontal pass, the buffer stride for the vertical pass.
//!
//! ## DOCS-GAP (residual): filter *selection* threshold (`MAX_MV_EXTENT`)
//!
//! §11.4's Advanced-Profile filter-selection logic computes a motion
//! vector size threshold whose "no restriction" branch reads
//! `FilterMvSizeThresh = ((MAX_MV_EXTENT >> 1) + 1) << 2`. `MAX_MV_EXTENT`
//! is never assigned a numeric value in §11.4 itself; the §11.1 long-vector
//! magnitude cap (the 11-bit `dx`/`dy` limit, 2047) supplies the only
//! self-consistent value, which
//! [`crate::inter::PredictionFilterPolicy::NO_MV_RESTRICTION_QPEL`] uses.
//! With that, the full §11.4 selector — MV-size threshold conversion, the
//! `FilterVarThresh = VarThresh << 5` formula (see
//! [`crate::frame_header::PredictionFilter::resolve`]) and the 16-point
//! variance metric ([`var_16_point`] / [`var_16_point_clamped`]) — is
//! implemented; the only residual ambiguity is the spec never spelling out
//! `MAX_MV_EXTENT` numerically.

/// Bilinear luma filter taps (spec §11.4.1 `BilinearLumaFilters[4][2]`).
///
/// Indexed by ¼-sample phase `0..=3`. Each row is `{Filter[0],
/// Filter[1]}` and sums to 128. Phase 0 is the full-sample-aligned
/// identity `{128, 0}`.
pub const BILINEAR_LUMA_FILTERS: [[i32; 2]; 4] = [
    [128, 0], // full sample aligned
    [96, 32], // 1/4
    [64, 64], // 1/2
    [32, 96], // 3/4
];

/// Bilinear chroma filter taps (spec §11.4.1
/// `BilinearChromaFilters[8][2]`).
///
/// Indexed by ⅛-sample phase `0..=7`. Each row is `{Filter[0],
/// Filter[1]}` and sums to 128. Phase 0 is the full-sample-aligned
/// identity `{128, 0}`.
pub const BILINEAR_CHROMA_FILTERS: [[i32; 2]; 8] = [
    [128, 0],  // full sample aligned
    [112, 16], // 1/8
    [96, 32],  // 1/4
    [80, 48],  // 3/8
    [64, 64],  // 1/2
    [48, 80],  // 5/8
    [32, 96],  // 3/4
    [16, 112], // 7/8
];

/// Bicubic filter set (spec §11.4.2 `BicubicFilterSet[17][8][4]`).
///
/// The outer index `0..=15` selects an `alpha` value from `-0.25` to
/// `-1.00` in steps of `-0.05` (VP6.2 bitstreams; the encoder signals the
/// index via `PredictionFilterAlpha`, see §9 Table 2). The 17th entry
/// (index 16) holds the fixed coefficients used for VP6.1 bitstreams.
///
/// The middle index `0..=7` is the ⅛-sample phase. The inner four are
/// the 4-tap kernel `{Filter[0], Filter[1], Filter[2], Filter[3]}`,
/// summing to 128. Phase 0 of every alpha is the full-sample-aligned
/// identity `{0, 128, 0, 0}`.
pub const BICUBIC_FILTER_SET: [[[i32; 4]; 8]; 17] = [
    // alpha ~= -0.25
    [
        [0, 128, 0, 0],
        [-3, 122, 9, 0],
        [-4, 109, 24, -1],
        [-5, 91, 45, -3],
        [-4, 68, 68, -4],
        [-3, 45, 91, -5],
        [-1, 24, 109, -4],
        [0, 9, 122, -3],
    ],
    // alpha ~= -0.30
    [
        [0, 128, 0, 0],
        [-4, 124, 9, -1],
        [-5, 110, 25, -2],
        [-6, 91, 46, -3],
        [-5, 69, 69, -5],
        [-3, 46, 91, -6],
        [-2, 25, 110, -5],
        [-1, 9, 124, -4],
    ],
    // alpha ~= -0.35
    [
        [0, 128, 0, 0],
        [-4, 123, 10, -1],
        [-6, 110, 26, -2],
        [-7, 92, 47, -4],
        [-6, 70, 70, -6],
        [-4, 47, 92, -7],
        [-2, 26, 110, -6],
        [-1, 10, 123, -4],
    ],
    // alpha ~= -0.40
    [
        [0, 128, 0, 0],
        [-5, 124, 10, -1],
        [-7, 110, 27, -2],
        [-7, 91, 48, -4],
        [-6, 70, 70, -6],
        [-4, 48, 92, -8],
        [-2, 27, 110, -7],
        [-1, 10, 124, -5],
    ],
    // alpha ~= -0.45
    [
        [0, 128, 0, 0],
        [-6, 124, 11, -1],
        [-8, 111, 28, -3],
        [-8, 92, 49, -5],
        [-7, 71, 71, -7],
        [-5, 49, 92, -8],
        [-3, 28, 111, -8],
        [-1, 11, 124, -6],
    ],
    // alpha ~= -0.50
    [
        [0, 128, 0, 0],
        [-6, 123, 12, -1],
        [-9, 111, 29, -3],
        [-9, 93, 50, -6],
        [-8, 72, 72, -8],
        [-6, 50, 93, -9],
        [-3, 29, 111, -9],
        [-1, 12, 123, -6],
    ],
    // alpha ~= -0.55
    [
        [0, 128, 0, 0],
        [-7, 124, 12, -1],
        [-10, 111, 30, -3],
        [-10, 93, 51, -6],
        [-9, 73, 73, -9],
        [-6, 51, 93, -10],
        [-3, 30, 111, -10],
        [-1, 12, 124, -7],
    ],
    // alpha ~= -0.60
    [
        [0, 128, 0, 0],
        [-7, 123, 13, -1],
        [-11, 112, 31, -4],
        [-11, 94, 52, -7],
        [-10, 74, 74, -10],
        [-7, 52, 94, -11],
        [-4, 31, 112, -11],
        [-1, 13, 123, -7],
    ],
    // alpha ~= -0.65
    [
        [0, 128, 0, 0],
        [-8, 124, 13, -1],
        [-12, 112, 32, -4],
        [-12, 94, 53, -7],
        [-10, 74, 74, -10],
        [-7, 53, 94, -12],
        [-4, 32, 112, -12],
        [-1, 13, 124, -8],
    ],
    // alpha ~= -0.70
    [
        [0, 128, 0, 0],
        [-9, 124, 14, -1],
        [-13, 112, 33, -4],
        [-13, 95, 54, -8],
        [-11, 75, 75, -11],
        [-8, 54, 95, -13],
        [-4, 33, 112, -13],
        [-1, 14, 124, -9],
    ],
    // alpha ~= -0.75
    [
        [0, 128, 0, 0],
        [-9, 123, 15, -1],
        [-14, 113, 34, -5],
        [-14, 95, 55, -8],
        [-12, 76, 76, -12],
        [-8, 55, 95, -14],
        [-5, 34, 112, -13],
        [-1, 15, 123, -9],
    ],
    // alpha ~= -0.80
    [
        [0, 128, 0, 0],
        [-10, 124, 15, -1],
        [-14, 113, 34, -5],
        [-15, 96, 56, -9],
        [-13, 77, 77, -13],
        [-9, 56, 96, -15],
        [-5, 34, 113, -14],
        [-1, 15, 124, -10],
    ],
    // alpha ~= -0.85
    [
        [0, 128, 0, 0],
        [-10, 123, 16, -1],
        [-15, 113, 35, -5],
        [-16, 98, 56, -10],
        [-14, 78, 78, -14],
        [-10, 56, 98, -16],
        [-5, 35, 113, -15],
        [-1, 16, 123, -10],
    ],
    // alpha ~= -0.90
    [
        [0, 128, 0, 0],
        [-11, 124, 17, -2],
        [-16, 113, 36, -5],
        [-17, 98, 57, -10],
        [-14, 78, 78, -14],
        [-10, 57, 98, -17],
        [-5, 36, 113, -16],
        [-2, 17, 124, -11],
    ],
    // alpha ~= -0.95
    [
        [0, 128, 0, 0],
        [-12, 125, 17, -2],
        [-17, 114, 37, -6],
        [-18, 99, 58, -11],
        [-15, 79, 79, -15],
        [-11, 58, 99, -18],
        [-6, 37, 114, -17],
        [-2, 17, 125, -12],
    ],
    // alpha ~= -1.00
    [
        [0, 128, 0, 0],
        [-12, 124, 18, -2],
        [-18, 114, 38, -6],
        [-19, 99, 59, -11],
        [-16, 80, 80, -16],
        [-11, 59, 99, -19],
        [-6, 38, 114, -18],
        [-2, 18, 124, -12],
    ],
    // VP6.1 bitstreams
    [
        [0, 128, 0, 0],
        [-4, 118, 16, -2],
        [-7, 106, 34, -5],
        [-8, 90, 53, -7],
        [-8, 72, 72, -8],
        [-7, 53, 90, -8],
        [-5, 34, 106, -7],
        [-2, 16, 118, -4],
    ],
];

/// Index of the VP6.1 coefficient set in [`BICUBIC_FILTER_SET`] (the
/// 17th entry, spec §11.4.2 "The 17th entry in the table is used for
/// VP6.1 bitstreams").
pub const BICUBIC_VP61_INDEX: usize = 16;

/// The shared filter rounding bias and descale shift (spec §11.4):
/// `OutputVal = (Σ taps + 64) >> 7`. `64 == 1 << (7 - 1)` is the
/// round-to-nearest bias for the 7-bit descale.
const FILTER_ROUND: i32 = 64;
const FILTER_SHIFT: u32 = 7;

/// One filtered point with the bilinear 2-tap kernel (spec §11.4.1).
///
/// `src` is the sample buffer; `pos` is the index of `SrcPtr[0]`;
/// `pixel_step` is the distance to the next sample (`1` for a horizontal
/// pass, the buffer stride for a vertical pass). `filter` is one
/// `{Filter[0], Filter[1]}` row from [`BILINEAR_LUMA_FILTERS`] or
/// [`BILINEAR_CHROMA_FILTERS`].
///
/// Per §11.4.1:
///
/// ```text
/// OutputVal = (SrcPtr[0] * Filter[0]) + (SrcPtr[PixelStep] * Filter[1]) + 64
/// OutputVal >>= 7
/// ```
///
/// Both bilinear taps are non-negative and sum to 128, so for inputs in
/// `0..=255` the result is already in `0..=255` and no clip is needed;
/// the result is returned as `i32` so this kernel can also feed the
/// vertical pass of a separable 2-D filter without re-clamping the
/// intermediate row.
#[inline]
pub fn bilinear_point(src: &[u8], pos: usize, pixel_step: usize, filter: [i32; 2]) -> i32 {
    let s0 = src[pos] as i32;
    let s1 = src[pos + pixel_step] as i32;
    ((s0 * filter[0]) + (s1 * filter[1]) + FILTER_ROUND) >> FILTER_SHIFT
}

/// One filtered point with the bicubic 4-tap kernel (spec §11.4.2).
///
/// `src` is the sample buffer; `pos` is the index of `SrcPtr[0]`;
/// `pixel_step` is the distance to the next sample (`1` horizontal, the
/// buffer stride vertical). `filter` is one `{Filter[0], Filter[1],
/// Filter[2], Filter[3]}` row from a [`BICUBIC_FILTER_SET`] phase.
///
/// The kernel reads four taps centred on `SrcPtr[0]`: `SrcPtr[-PixelStep]`,
/// `SrcPtr[0]`, `SrcPtr[PixelStep]`, `SrcPtr[2*PixelStep]`. `pos` must
/// therefore be at least `pixel_step` and have at least `2*pixel_step`
/// trailing room — the caller (motion compensation against the UMV-bordered
/// reconstruction buffer, §11.5) guarantees this.
///
/// Per §11.4.2:
///
/// ```text
/// OutputVal = (SrcPtr[-PixelStep] * Filter[0]) +
///             (SrcPtr[0]          * Filter[1]) +
///             (SrcPtr[PixelStep]  * Filter[2]) +
///             (SrcPtr[2*PixelStep]* Filter[3]) + 64
/// OutputVal >>= 7
/// // Clip the result to 0..=255
/// ```
///
/// Unlike the bilinear kernel the bicubic taps include negative values,
/// so the descaled result can fall outside `0..=255` (overshoot/ringing);
/// §11.4.2's pseudocode clips it inclusively, which this function does.
#[inline]
pub fn bicubic_point(src: &[u8], pos: usize, pixel_step: usize, filter: [i32; 4]) -> u8 {
    let m1 = src[pos - pixel_step] as i32;
    let s0 = src[pos] as i32;
    let p1 = src[pos + pixel_step] as i32;
    let p2 = src[pos + 2 * pixel_step] as i32;
    let out =
        ((m1 * filter[0]) + (s0 * filter[1]) + (p1 * filter[2]) + (p2 * filter[3]) + FILTER_ROUND)
            >> FILTER_SHIFT;
    out.clamp(0, 255) as u8
}

/// 16-point prediction-block variance metric (spec §11.4 `Var16Point`).
///
/// Used by the Advanced-Profile filter selector to decide between
/// bilinear and bicubic interpolation: the "measured variance of the
/// prediction block" §11.4 compares against `FilterVarThresh`. For
/// performance the spec does not consider all 64 samples; it samples 16
/// points — every other sample in every other row of the 8x8 whole-sample
/// aligned region — as:
///
/// ```text
/// for ( i=0; i<4; i+=2 ) {
///     XSum  += DiffPtr[0]; XXSum += DiffPtr[0]^2;
///     XSum  += DiffPtr[2]; XXSum += DiffPtr[2]^2;
///     XSum  += DiffPtr[4]; XXSum += DiffPtr[4]^2;
///     XSum  += DiffPtr[6]; XXSum += DiffPtr[6]^2;
///     DiffPtr += (Stride << 1);
/// }
/// return (( (XXSum<<4) - XSum*XSum )) >> 8
/// ```
///
/// `data` is the sample buffer, `pos` the index of the region's top-left
/// sample, `stride` the distance between consecutive rows. The loop's
/// `i < 4` with `i += 2` and a `Stride << 1` step visits rows 0 and 2;
/// the spec's "every other sample in every other row" wording and the
/// 16-point claim mean rows `{0, 2, 4, 6}` are intended — the literal
/// pseudocode bound `i < 4` only reaches rows `{0, 2}` (8 points), an
/// off-by-one against the prose's "16 sample points". This function
/// follows the prose: 4 rows × 4 columns = 16 points. See the
/// module-level note.
///
/// Returned as `i32`: the population-variance mismatch metric
/// `((Σx² << 4) − (Σx)²) >> 8`.
pub fn var_16_point(data: &[u8], pos: usize, stride: usize) -> i32 {
    let mut x_sum: i64 = 0;
    let mut xx_sum: i64 = 0;
    // Prose: 16 points = every other sample (cols 0,2,4,6) in every
    // other row (rows 0,2,4,6).
    for row in (0..8).step_by(2) {
        let base = pos + row * stride;
        for col in (0..8).step_by(2) {
            let v = data[base + col] as i64;
            x_sum += v;
            xx_sum += v * v;
        }
    }
    (((xx_sum << 4) - x_sum * x_sum) >> 8) as i32
}

/// §11.4 16-point variance, but with each sampled position **edge-clamped**
/// into `data`'s valid index range (the §11.5 out-of-range rule).
///
/// §11.4 computes the prediction-block variance over an 8×8 region whose
/// top-left corner is the *whole-sample-aligned* MV offset
/// (`WholeSampleAlignedX/Y = MvX/Y >> MvShift`). Because VP6 permits
/// unrestricted motion vectors (§11.5), that corner — and therefore the
/// sampled positions — can lie past the reference buffer's edge. §11.5
/// resolves every such out-of-range reference read by edge-replication:
///
/// > because the extension is built by edge replication, any read from a
/// > sample position … is equivalent to clamping the read position to the
/// > original image's valid range.
///
/// [`var_16_point`] assumes the caller pre-bounded `pos` so all 16 reads
/// land inside `data`; that holds for spec-conformant streams (MV
/// components capped at ±127 ¼-pel ⇒ within the 48-sample UMV border).
/// This variant additionally tolerates an MV that the encoder left
/// out-of-spec (the long-vector decode formula can yield magnitudes up to
/// 255 ¼-pel, i.e. ~64 whole pixels, beyond the 48-sample border): each
/// computed index is clamped to `0..data.len()`, replicating the §11.5
/// edge sample instead of indexing out of bounds.
///
/// For any `pos` whose full 8×8 sampling window already lies inside
/// `data`, every clamp is a no-op and the result is **bit-identical** to
/// [`var_16_point`]. `signed_pos` is the (possibly negative) top-left
/// index of the variance window relative to the buffer start, so an MV
/// pointing above/left of the bordered origin clamps to row/column 0.
pub fn var_16_point_clamped(data: &[u8], signed_pos: i64, stride: usize) -> i32 {
    debug_assert!(!data.is_empty(), "var_16_point_clamped: empty buffer");
    let last = data.len() as i64 - 1;
    let mut x_sum: i64 = 0;
    let mut xx_sum: i64 = 0;
    for row in (0..8i64).step_by(2) {
        let base = signed_pos + row * stride as i64;
        for col in (0..8i64).step_by(2) {
            let idx = (base + col).clamp(0, last) as usize;
            let v = data[idx] as i64;
            x_sum += v;
            xx_sum += v * v;
        }
    }
    (((xx_sum << 4) - x_sum * x_sum) >> 8) as i32
}

/// Apply the bilinear filter to an 8x8 prediction block (spec §11.4 +
/// §11.4.1), separably in x and/or y as the fractional phases require.
///
/// `src` is the (UMV-bordered) reference buffer; `src_pos` is the index of
/// the block's top-left whole-sample-aligned source; `src_stride` is the
/// reference buffer line length. `xf`/`yf` are the `{Filter[0],
/// Filter[1]}` taps for the x and y fractional phases respectively (use
/// phase-0 `[128, 0]`, the identity, when a direction is whole-pixel
/// aligned). The 64 interpolated samples are written to `dst` in raster
/// order.
///
/// When only x is fractional, this is a single horizontal pass; when only
/// y is fractional, a single vertical pass; when both are fractional, the
/// horizontal pass over a 8-wide × 9-tall intermediate region feeds the
/// vertical pass (§11.4: "an intermediate result ... applying the filter
/// in the x direction ... used as input to a second pass which filters in
/// the y direction"). The intermediate row count is `8 + 1` because the
/// 2-tap vertical pass reads one extra row below each output.
///
/// `src` must hold the full source extent: `src_pos + 8*src_stride + 8`
/// for the two-pass case. The caller (motion compensation) guarantees the
/// extent via the §11.5 UMV border.
pub fn bilinear_block(
    src: &[u8],
    src_pos: usize,
    src_stride: usize,
    xf: [i32; 2],
    yf: [i32; 2],
    dst: &mut [u8; 64],
) {
    let x_frac = xf[1] != 0;
    let y_frac = yf[1] != 0;

    if x_frac && y_frac {
        // Two-pass separable: horizontal first into a 8x9 scratch, then
        // vertical. The vertical 2-tap pass reads rows r and r+1, so the
        // scratch needs 9 rows for the 8 outputs.
        let mut tmp = [0i32; 8 * 9];
        for r in 0..9 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                tmp[r * 8 + c] = bilinear_point(src, row_base + c, 1, xf);
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                let s0 = tmp[r * 8 + c];
                let s1 = tmp[(r + 1) * 8 + c];
                let out = ((s0 * yf[0]) + (s1 * yf[1]) + FILTER_ROUND) >> FILTER_SHIFT;
                dst[r * 8 + c] = out as u8;
            }
        }
    } else if x_frac {
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = bilinear_point(src, row_base + c, 1, xf) as u8;
            }
        }
    } else if y_frac {
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = bilinear_point(src, row_base + c, src_stride, yf) as u8;
            }
        }
    } else {
        // Whole-pixel aligned in both directions: a straight copy.
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = src[row_base + c];
            }
        }
    }
}

/// Apply the bicubic filter to an 8x8 prediction block (spec §11.4 +
/// §11.4.2), separably in x and/or y.
///
/// Arguments mirror [`bilinear_block`] but with 4-tap `{Filter[0..=3]}`
/// kernels (use phase-0 `[0, 128, 0, 0]`, the identity, for a
/// whole-pixel-aligned direction). The bicubic kernel reads one sample
/// *before* and two *after* the centre, so:
///
/// * The horizontal pass reads columns `c-1 .. c+2` and so the source
///   region must start one column left of the block.
/// * The vertical pass reads rows `r-1 .. r+2`, so the two-pass scratch
///   spans rows `-1 .. 9` (11 rows) and the horizontal pass must produce
///   all of them.
///
/// `src_pos` indexes the block's top-left whole-sample-aligned source;
/// the function reads the §11.5 UMV-bordered neighbourhood around it. The
/// intermediate scratch is clipped to `0..=255` after the horizontal pass
/// (each bicubic point clips per §11.4.2) before feeding the vertical
/// pass, matching the spec's per-point clip on both passes.
pub fn bicubic_block(
    src: &[u8],
    src_pos: usize,
    src_stride: usize,
    xf: [i32; 4],
    yf: [i32; 4],
    dst: &mut [u8; 64],
) {
    let x_frac = !(xf[0] == 0 && xf[2] == 0 && xf[3] == 0);
    let y_frac = !(yf[0] == 0 && yf[2] == 0 && yf[3] == 0);

    if x_frac && y_frac {
        // Two-pass separable. The vertical 4-tap pass reads rows r-1..r+2;
        // for outputs r=0..7 that spans source rows -1..9 (11 rows). Store
        // the horizontal-pass output indexed by row+1 so index 0 is
        // source row -1.
        let mut tmp = [0u8; 8 * 11];
        for sr in 0..11 {
            // source row index is sr - 1 relative to the block top.
            let row_base = (src_pos + sr * src_stride) - src_stride;
            for c in 0..8 {
                tmp[sr * 8 + c] = bicubic_point(src, row_base + c, 1, xf);
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                // vertical taps centred on scratch row (r+1), reading
                // (r+1)-1 .. (r+1)+2 == r .. r+3.
                let m1 = tmp[r * 8 + c] as i32;
                let s0 = tmp[(r + 1) * 8 + c] as i32;
                let p1 = tmp[(r + 2) * 8 + c] as i32;
                let p2 = tmp[(r + 3) * 8 + c] as i32;
                let out =
                    ((m1 * yf[0]) + (s0 * yf[1]) + (p1 * yf[2]) + (p2 * yf[3]) + FILTER_ROUND)
                        >> FILTER_SHIFT;
                dst[r * 8 + c] = out.clamp(0, 255) as u8;
            }
        }
    } else if x_frac {
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = bicubic_point(src, row_base + c, 1, xf);
            }
        }
    } else if y_frac {
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = bicubic_point(src, row_base + c, src_stride, yf);
            }
        }
    } else {
        for r in 0..8 {
            let row_base = src_pos + r * src_stride;
            for c in 0..8 {
                dst[r * 8 + c] = src[row_base + c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every filter tap set sums to 128 (the §11.4 fixed-point scale).
    /// A row that doesn't sum to 128 would bias brightness; this guards
    /// the entire transcription against a single mistyped coefficient.
    #[test]
    fn all_filter_rows_sum_to_128() {
        for (i, row) in BILINEAR_LUMA_FILTERS.iter().enumerate() {
            assert_eq!(row.iter().sum::<i32>(), 128, "bilinear luma phase {i}");
        }
        for (i, row) in BILINEAR_CHROMA_FILTERS.iter().enumerate() {
            assert_eq!(row.iter().sum::<i32>(), 128, "bilinear chroma phase {i}");
        }
        for (a, alpha) in BICUBIC_FILTER_SET.iter().enumerate() {
            for (p, row) in alpha.iter().enumerate() {
                assert_eq!(row.iter().sum::<i32>(), 128, "bicubic alpha {a} phase {p}");
            }
        }
    }

    /// Phase 0 of every family is the full-sample-aligned identity.
    /// Bilinear `{128,0}`, bicubic `{0,128,0,0}`.
    #[test]
    fn phase_zero_is_identity() {
        assert_eq!(BILINEAR_LUMA_FILTERS[0], [128, 0]);
        assert_eq!(BILINEAR_CHROMA_FILTERS[0], [128, 0]);
        for alpha in BICUBIC_FILTER_SET.iter() {
            assert_eq!(alpha[0], [0, 128, 0, 0]);
        }
    }

    /// Table dimensions match the spec declarations:
    /// `BilinearLumaFilters[4][2]`, `BilinearChromaFilters[8][2]`,
    /// `BicubicFilterSet[17][8][4]`.
    #[test]
    fn table_dimensions_match_spec() {
        assert_eq!(BILINEAR_LUMA_FILTERS.len(), 4);
        assert_eq!(BILINEAR_CHROMA_FILTERS.len(), 8);
        assert_eq!(BICUBIC_FILTER_SET.len(), 17);
        for alpha in BICUBIC_FILTER_SET.iter() {
            assert_eq!(alpha.len(), 8);
        }
        assert_eq!(BICUBIC_VP61_INDEX, 16);
    }

    /// The bilinear 1/2-sample phase is the simple average with
    /// round-to-nearest: `(a*64 + b*64 + 64) >> 7 == (a + b + 1) >> 1`.
    #[test]
    fn bilinear_half_phase_is_rounded_average() {
        let src = [10u8, 21u8];
        // (10*64 + 21*64 + 64) >> 7 = (640 + 1344 + 64) >> 7 = 2048>>7 = 16
        let got = bilinear_point(&src, 0, 1, BILINEAR_LUMA_FILTERS[2]);
        assert_eq!(got, (10 + 21 + 1) >> 1);
        assert_eq!(got, 16);
    }

    /// The bilinear phase-0 identity returns SrcPtr[0] exactly:
    /// `(a*128 + b*0 + 64) >> 7 == a` for any `a` in `0..=255`.
    #[test]
    fn bilinear_identity_returns_src0() {
        for a in 0u8..=255 {
            let src = [a, 200];
            assert_eq!(bilinear_point(&src, 0, 1, [128, 0]), a as i32);
        }
    }

    /// A bilinear filter of a flat (constant) source returns the same
    /// constant: `(v*f0 + v*f1 + 64) >> 7 == (v*128 + 64) >> 7 == v`.
    #[test]
    fn bilinear_flat_source_is_unchanged() {
        for &phase in BILINEAR_LUMA_FILTERS.iter() {
            let src = [73u8; 2];
            assert_eq!(bilinear_point(&src, 0, 1, phase), 73);
        }
    }

    /// The bicubic phase-0 identity `{0,128,0,0}` returns SrcPtr[0].
    #[test]
    fn bicubic_identity_returns_src0() {
        // pos=1 so the -PixelStep tap at index 0 is valid.
        let src = [9u8, 130, 200, 50];
        assert_eq!(bicubic_point(&src, 1, 1, [0, 128, 0, 0]), 130);
    }

    /// A bicubic filter of a flat source returns the same constant (taps
    /// sum to 128): `(v*Σtaps + 64) >> 7 == (v*128 + 64) >> 7 == v`.
    #[test]
    fn bicubic_flat_source_is_unchanged() {
        for alpha in BICUBIC_FILTER_SET.iter() {
            for &phase in alpha.iter() {
                let src = [200u8; 4];
                assert_eq!(bicubic_point(&src, 1, 1, phase), 200);
            }
        }
    }

    /// The bicubic kernel clips overshoot to `0..=255` per §11.4.2.
    /// Construct a source whose ringing pushes the raw output above 255
    /// and below 0 and verify the inclusive clip.
    #[test]
    fn bicubic_clips_overshoot() {
        // Use the -1.00 alpha, 1/2 phase {-16,80,80,-16}: a sharp
        // edge maximises ringing.
        let phase = BICUBIC_FILTER_SET[15][4];
        // High output: centre taps see 255, side taps see 0.
        // (0*-16 + 255*80 + 255*80 + 0*-16 + 64) >> 7 = (40800+64)>>7 = 319 -> 255
        let hi = [0u8, 255, 255, 0];
        assert_eq!(bicubic_point(&hi, 1, 1, phase), 255);
        // Low output: side taps see 255, centre taps see 0.
        // (255*-16 + 0*80 + 0*80 + 255*-16 + 64) >> 7 = (-8160+64)>>7 = -64 -> 0
        let lo = [255u8, 0, 0, 255];
        assert_eq!(bicubic_point(&lo, 1, 1, phase), 0);
    }

    /// A whole-pixel-aligned block (both directions phase 0) is a straight
    /// copy of the source block.
    #[test]
    fn bilinear_block_whole_pixel_copies() {
        // 10x10 source so an 8x8 block plus the 2-tap trailing room fits.
        let stride = 10usize;
        let mut src = [0u8; 10 * 10];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let mut dst = [0u8; 64];
        bilinear_block(&src, stride + 1, stride, [128, 0], [128, 0], &mut dst);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 8 + c], src[(r + 1) * stride + (c + 1)]);
            }
        }
    }

    /// A bilinear block over a flat source stays flat at the same value,
    /// for every combination of x/y fractional phase (including the
    /// two-pass case). This exercises the separable scratch path.
    #[test]
    fn bilinear_block_flat_source_all_phases() {
        let stride = 16usize;
        let src = [99u8; 16 * 16];
        for &xf in BILINEAR_LUMA_FILTERS.iter() {
            for &yf in BILINEAR_LUMA_FILTERS.iter() {
                let mut dst = [0u8; 64];
                bilinear_block(&src, stride + 1, stride, xf, yf, &mut dst);
                assert_eq!(dst, [99u8; 64], "xf={xf:?} yf={yf:?}");
            }
        }
    }

    /// Horizontal-only bilinear block equals applying [`bilinear_point`]
    /// per sample with `pixel_step = 1`.
    #[test]
    fn bilinear_block_horizontal_matches_point() {
        let stride = 12usize;
        let mut src = [0u8; 12 * 12];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i * 7) % 251) as u8;
        }
        let xf = BILINEAR_LUMA_FILTERS[1]; // 1/4
        let mut dst = [0u8; 64];
        bilinear_block(&src, stride + 1, stride, xf, [128, 0], &mut dst);
        for r in 0..8 {
            for c in 0..8 {
                let pos = (r + 1) * stride + (c + 1);
                assert_eq!(dst[r * 8 + c], bilinear_point(&src, pos, 1, xf) as u8);
            }
        }
    }

    /// The bicubic block over a flat source stays flat, for every
    /// fractional-phase combination including the two-pass case. Exercises
    /// the 11-row scratch path and confirms the identity through the clip.
    #[test]
    fn bicubic_block_flat_source_all_phases() {
        let stride = 20usize;
        let src = [140u8; 20 * 20];
        let alpha = BICUBIC_FILTER_SET[8]; // arbitrary mid alpha
        for &xf in alpha.iter() {
            for &yf in alpha.iter() {
                let mut dst = [0u8; 64];
                // src_pos with room: one col/row before, two after the 8x8.
                bicubic_block(&src, 2 * stride + 2, stride, xf, yf, &mut dst);
                assert_eq!(dst, [140u8; 64], "xf={xf:?} yf={yf:?}");
            }
        }
    }

    /// Whole-pixel-aligned bicubic block (both phase 0) is a straight
    /// copy.
    #[test]
    fn bicubic_block_whole_pixel_copies() {
        let stride = 16usize;
        let mut src = [0u8; 16 * 16];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let mut dst = [0u8; 64];
        bicubic_block(
            &src,
            2 * stride + 2,
            stride,
            [0, 128, 0, 0],
            [0, 128, 0, 0],
            &mut dst,
        );
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 8 + c], src[(r + 2) * stride + (c + 2)]);
            }
        }
    }

    /// `var_16_point` of a flat block is zero: every sampled value is the
    /// same so `Σx² · 16 == (Σx)²`, giving `(16·16·v² − (16v)²) >> 8 = 0`.
    #[test]
    fn var_16_point_flat_is_zero() {
        let src = [55u8; 64];
        assert_eq!(var_16_point(&src, 0, 8), 0);
    }

    /// `var_16_point` of a two-level block is positive and matches the
    /// closed form. With half the 16 sampled points at `a` and half at
    /// `b`: `Σx = 8(a+b)`, `Σx² = 8(a²+b²)`, so
    /// `((Σx²<<4) − (Σx)²) >> 8 = (128(a²+b²) − 64(a+b)²) >> 8`.
    #[test]
    fn var_16_point_two_level_matches_closed_form() {
        // Build a block where sampled columns alternate a,b across the
        // four sampled (even) columns: cols 0,4 = a; cols 2,6 = b.
        let (a, b) = (40i64, 200i64);
        let mut src = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                src[r * 8 + c] = if (c / 2) % 2 == 0 { a as u8 } else { b as u8 };
            }
        }
        // 16 sampled points: 8 at a, 8 at b.
        let x_sum = 8 * (a + b);
        let xx_sum = 8 * (a * a + b * b);
        let expected = (((xx_sum << 4) - x_sum * x_sum) >> 8) as i32;
        assert_eq!(var_16_point(&src, 0, 8), expected);
        assert!(expected > 0);
    }

    /// `var_16_point` only reads even rows and even columns. Poisoning the
    /// odd positions with extreme values must not change the result.
    #[test]
    fn var_16_point_ignores_odd_positions() {
        let mut src = [100u8; 64];
        let baseline = var_16_point(&src, 0, 8);
        for r in 0..8 {
            for c in 0..8 {
                if r % 2 == 1 || c % 2 == 1 {
                    src[r * 8 + c] = 255;
                }
            }
        }
        assert_eq!(var_16_point(&src, 0, 8), baseline);
        assert_eq!(baseline, 0);
    }

    /// For a window fully inside the buffer, `var_16_point_clamped` is
    /// bit-identical to the unclamped `var_16_point` (every clamp a no-op).
    #[test]
    fn var_16_point_clamped_matches_unclamped_in_range() {
        let (a, b) = (40i64, 200i64);
        let mut src = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                src[r * 8 + c] = if (c / 2) % 2 == 0 { a as u8 } else { b as u8 };
            }
        }
        assert_eq!(var_16_point_clamped(&src, 0, 8), var_16_point(&src, 0, 8));
    }

    /// A negative `signed_pos` (window above/left of the buffer origin)
    /// clamps every sampled index to 0, so the §11.5 edge sample is
    /// replicated. With a flat buffer the variance is still 0.
    #[test]
    fn var_16_point_clamped_negative_pos_clamps_to_origin() {
        let src = [77u8; 64];
        // Window starting far before the origin: all reads clamp to idx 0.
        assert_eq!(var_16_point_clamped(&src, -1000, 8), 0);
    }

    /// A `signed_pos` whose window runs past the buffer end clamps the
    /// overrunning samples to the last index instead of panicking. The
    /// result equals the variance of the clamped sample set, which for a
    /// flat buffer is 0.
    #[test]
    fn var_16_point_clamped_past_end_clamps_to_last() {
        let src = [123u8; 64];
        let last = (src.len() - 1) as i64;
        // Start at the last valid index: every read clamps to it.
        assert_eq!(var_16_point_clamped(&src, last, 8), 0);
        // Start well past the end: same clamp behaviour, no panic.
        assert_eq!(var_16_point_clamped(&src, last + 500, 8), 0);
    }

    /// The clamped variance replicates the edge sample for an
    /// out-of-range window the same way a §11.5-bordered buffer would: a
    /// non-flat buffer read entirely off the right edge sees only the last
    /// column's value, giving zero variance.
    #[test]
    fn var_16_point_clamped_off_edge_replicates_edge_value() {
        // Stride-8 buffer whose last sample is 200, everything else 0.
        let mut src = [0u8; 64];
        src[63] = 200;
        // Start exactly at the last index: all 16 reads clamp to idx 63.
        assert_eq!(var_16_point_clamped(&src, 63, 8), 0);
    }
}

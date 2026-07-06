//! §13 Huffman-coded DCT coefficient decode + encode (spec §13.1,
//! §13.2.2, §13.3.2, §13.3.3.2, §13.4).
//!
//! When a frame's second data partition uses the Huffman coder
//! (`UseHuffman == 1`, §5/§6), the per-block DCT tokens are read as
//! whole raw bits (§7.2) instead of BoolCoder decisions. The trees are
//! not transmitted: the decoder derives them from the **same** §13
//! node-probability banks the arithmetic path uses, via the §13.1 /
//! §13.3.3.2 BoolCoder-tree → leaf-probability conversions, then the
//! §7.2.1 `VP6_CreateHuffmanTree` builder:
//!
//! * **DC** (§13.2.2) — one tree per Table 25 plane, from the *raw*
//!   `DcProbs[plane][11]` bank (note: the Huffman DC decode does **not**
//!   use the §13.2 Table 26 left/above context expansion; the tree is
//!   built from the un-contexted per-plane bank the Figure-5 updates
//!   mutate).
//! * **AC** (§13.3.2) — `AcHuffTree[plane][prec][band]` over the Table
//!   36 four-band split (coefficient 1 / 2–4 / 5–10 / 11–63), built from
//!   the first four Table 30 bands of `AcProbs[plane][prec][band][11]`.
//! * **Zero runs** (§13.3.3.2) — one tree per Table 37 ZRL band from
//!   the first eight `ZeroRunProbs[band]` nodes.
//!
//! ## Cross-block run state (§13.2.2 / §13.3.2 / §13.4)
//!
//! Unlike the arithmetic path, the Huffman coder amortises runs of
//! trivial blocks **across block boundaries within a plane**:
//!
//! * A `ZERO_TOKEN` in the DC position carries a §13.4 run of blocks in
//!   the same plane whose DC is also 0 (`CurrentDcRunLen`).
//! * A `DCT_EOB_TOKEN` at the first AC position carries a §13.4 run of
//!   blocks in the same plane whose ACs are all 0 (`CurrentAc1RunLen`,
//!   stored as "EOB Token Run − 1" per the §13.3.2 listing).
//!
//! Both counters live in [`HuffmanRunState`] and are indexed by the
//! Table 25/28 plane (Y = 0, U and V share index 1 — the state arrays
//! are `[2]` exactly like the probability banks, so a chroma run spans
//! U and V blocks in stream order).
//!
//! ## Disambiguated readings (spec-internal evidence only)
//!
//! Three spots in §13.3.3.2/§13.4/§13.2.2 are internally inconsistent
//! as printed; each is resolved here against the arithmetic coder's
//! value space (the Huffman trees are built from the *same* probability
//! banks, so their leaves must denote the same values):
//!
//! 1. **ZRL symbol → run length.** `ZRLBoolTreeToHuffProbs` maps
//!    `HuffProb[0..=7]` to the §13.3.3.1 tree's run-length leaves 1..=8
//!    and `HuffProb[8]` to the long-run escape (the products follow the
//!    arithmetic tree's branch structure exactly). The §13.3.3.2
//!    listing's `EncodedCoeffs += ZrlToken` (for `ZrlToken < 8`) would
//!    make symbol 0 advance zero positions — an infinite loop — so the
//!    operative reading is `run = symbol + 1` for symbols 0..=7.
//! 2. **Long ZRL escape.** The listing prints `8 + R(6)`, but run 8 is
//!    already leaf 7, and §13.3.3.1's escape is "run length minus nine
//!    … six bits" (9..=72). The operative escape is `9 + R(6)`.
//! 3. **DC zero-run store.** §13.2.2 prints `CurrentDcRunLen[Plane] =
//!    DC Run Length` while the same-shaped AC1 case stores "EOB Token
//!    Run − 1". The §13.4 run decoder's minimum value is 1, so a lone
//!    zero-DC block (no additional zero-DC blocks) is only encodable if
//!    the decoded value counts the run *inclusive* of the current block
//!    — the operative store is `DC Run Length − 1`, symmetric with the
//!    AC1 case.
//!
//! The in-tree encoder ([`encode_frame_blocks_huffman`]) mirrors each
//! reading bit-for-bit, so encode→decode round-trips are exact.
//!
//! ## Provenance
//!
//! Sourced exclusively from `docs/video/vp6/vp6_format.pdf` (§7.2,
//! §13.1–§13.4) and the staged errata. No external library code was
//! consulted.

use oxideav_core::bits::BitWriter;

use crate::block_decode::{AcProbBank, BlockCoeffs, ZeroRunProbBank, BLOCK_SIZE};
use crate::coeff_prob_update::CoeffProbBanks;
use crate::huffman::{codeword_for, create_huffman_tree, HuffNode};
use crate::raw_bits::RawBitReader;
use crate::token_encode::token_for_magnitude;
use crate::tokens::{AcPlane, AcPrecContext, DctToken, NUM_PLANES, NUM_TREE_NODES};
use crate::zrl::{ZrlBand, NUM_ZRL_BANDS, NUM_ZRL_HUFF_PROBS};
use crate::Error;

/// Number of Table 36 Huffman AC bands (coefficient 1 / 2–4 / 5–10 /
/// 11–63) — the §13.3.2 four-band split, versus the arithmetic coder's
/// six Table 30 bands.
pub const NUM_HUFF_AC_BANDS: usize = 4;

/// Table 36 `VP6_CoeffToHuffBand`: the Huffman AC band for a scan
/// position `1..=63`.
///
/// | band | positions |
/// |------|-----------|
/// | 0    | 1         |
/// | 1    | 2–4       |
/// | 2    | 5–10      |
/// | 3    | 11–63     |
///
/// Returns `None` for position 0 (the DC coefficient — §13.2.2 uses the
/// per-plane DC tree, not a band) and positions ≥ 64.
#[inline]
pub const fn huff_ac_band(coeff_index: usize) -> Option<usize> {
    match coeff_index {
        1 => Some(0),
        2..=4 => Some(1),
        5..=10 => Some(2),
        11..=63 => Some(3),
        _ => None,
    }
}

/// Clamp a §13.1 / §13.3.3.2 leaf probability into the §7 valid range
/// `1..=255`.
///
/// The conversion products (`(a * b) >> 8` chains) can reach 0 for
/// extreme node probabilities; §7.2.1 requires every leaf probability
/// in `1..=255` ("the value 0 is explicitly forbidden"). A zero-probability
/// leaf still needs a codeword (the token remains syntactically legal),
/// so it is clamped to the minimum weight and lands at the deepest
/// position its siblings allow.
#[inline]
fn clamp_prob(p: u8) -> u8 {
    if p == 0 {
        1
    } else {
        p
    }
}

/// One §7.2 Huffman tree plus its precomputed encode-side codewords.
#[derive(Debug, Clone)]
struct HuffTable {
    /// The §7.2.1 sort-list (root at `len - 1`).
    tree: Vec<HuffNode>,
    /// `codewords[symbol] = (pattern, len)`, MSB-first pattern in the
    /// low `len` bits.
    codewords: Vec<(u32, usize)>,
}

impl HuffTable {
    /// Build the tree for symbols `0..probs.len()` (clamping zero
    /// probabilities to 1) and precompute every codeword.
    fn build(probs: &[u8]) -> Self {
        let symbols: Vec<i32> = (0..probs.len() as i32).collect();
        let clamped: Vec<u8> = probs.iter().map(|&p| clamp_prob(p)).collect();
        let tree = create_huffman_tree(&symbols, &clamped)
            .expect("§13 Huffman alphabets always have ≥ 2 symbols with probs 1..=255");
        let codewords = symbols
            .iter()
            .map(|&s| codeword_for(&tree, s).expect("every symbol has a leaf"))
            .collect();
        Self { tree, codewords }
    }

    /// Decode one symbol from `r` (§7.2 traversal over raw bits).
    fn decode(&self, r: &mut RawBitReader<'_>) -> Result<usize, Error> {
        let sym = r
            .read_huffman_symbol(&self.tree)
            .map_err(|_| Error::Truncated)?;
        Ok(sym as usize)
    }

    /// Emit `symbol`'s codeword to `w`.
    fn encode(&self, w: &mut BitWriter, symbol: usize) {
        let (pattern, len) = self.codewords[symbol];
        w.write_u32(pattern, len as u32);
    }
}

/// The full per-frame §13 Huffman decoding surface: the DC, AC and
/// zero-run trees derived from the frame's post-update probability
/// banks (§13.2.2 `DcHuffTree[2]`, §13.3.2 `AcHuffTree[2][3][4]`,
/// §13.3.3.2 `ZeroHuffTree[2]`).
///
/// Build once per frame with [`HuffmanCoeffTables::from_banks`] (after
/// the §8 Figure-5 updates have mutated the banks); both
/// [`decode_block_coefficients_huffman`] and
/// [`encode_frame_blocks_huffman`] consume it.
#[derive(Debug, Clone)]
pub struct HuffmanCoeffTables {
    /// §13.2.2 per-plane DC trees.
    dc: [HuffTable; NUM_PLANES],
    /// §13.3.2 AC trees `[plane][prec][huff-band]`.
    ac: Vec<HuffTable>,
    /// §13.3.3.2 per-ZRL-band zero-run trees.
    zrl: [HuffTable; NUM_ZRL_BANDS],
}

impl HuffmanCoeffTables {
    /// Derive every §13 Huffman tree from the frame's raw probability
    /// banks.
    pub fn from_banks(banks: &CoeffProbBanks) -> Self {
        Self::from_parts(&banks.dc_probs, &banks.ac_probs, &banks.zrl_probs)
    }

    /// Derive the trees from the individual banks (the same data
    /// [`CoeffProbBanks`] carries, split out for callers holding the
    /// pieces).
    pub fn from_parts(
        dc_probs: &[[u8; NUM_TREE_NODES]; NUM_PLANES],
        ac_probs: &AcProbBank,
        zrl_probs: &ZeroRunProbBank,
    ) -> Self {
        // §13.2.2: DcHuffTree[plane] from the raw per-plane DC bank via
        // the DC variant of the §13.1 conversion (node-0 left branch
        // folded wholly into ZERO_TOKEN — §13.2.1: EOB is forbidden in
        // the DC position, so the DC BoolCoder tree skips the EOB/0
        // decision and the conversion credits `NodeProb[0]` to ZERO;
        // fixture-arbitrated, see
        // `dct_token_bool_tree_to_huff_probs_dc`).
        let dc = [
            HuffTable::build(&crate::tokens::dct_token_bool_tree_to_huff_probs_dc(
                &dc_probs[0],
            )),
            HuffTable::build(&crate::tokens::dct_token_bool_tree_to_huff_probs_dc(
                &dc_probs[1],
            )),
        ];

        // §13.3.2: AcHuffTree over [plane][prec][band 0..4] — "derived
        // from the probabilities in AcProbs[2][3][0-3][11], the first 4
        // bands".
        let mut ac = Vec::with_capacity(NUM_PLANES * 3 * NUM_HUFF_AC_BANDS);
        for plane_bank in ac_probs.iter() {
            for prec_bank in plane_bank.iter() {
                for band_nodes in prec_bank.iter().take(NUM_HUFF_AC_BANDS) {
                    ac.push(HuffTable::build(
                        &crate::tokens::dct_token_bool_tree_to_huff_probs(band_nodes),
                    ));
                }
            }
        }

        // §13.3.3.2: ZeroHuffTree[band] from the first eight ZRL nodes.
        let zrl = [0usize, 1].map(|b| {
            let mut nodes = [0u8; 8];
            nodes.copy_from_slice(&zrl_probs[b][..8]);
            HuffTable::build(&crate::zrl::zrl_bool_tree_to_huff_probs(&nodes))
        });

        Self { dc, ac, zrl }
    }

    #[inline]
    fn ac_table(&self, plane: AcPlane, prec: AcPrecContext, band: usize) -> &HuffTable {
        &self.ac[(plane.index() * 3 + prec.index()) * NUM_HUFF_AC_BANDS + band]
    }
}

/// The §13.2.2 / §13.3.2 cross-block run counters, per Table 25/28
/// plane (Y = 0; U and V share 1).
///
/// Reset to zero at the start of every frame's coefficient partition;
/// thread one instance through all block decodes (or encodes) of the
/// frame in stream order.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanRunState {
    /// `CurrentDcRunLen[plane]` — blocks remaining whose DC is 0
    /// without any bits in the stream.
    pub dc_run: [u32; NUM_PLANES],
    /// `CurrentAc1RunLen[plane]` — blocks remaining whose ACs are all
    /// 0 without any bits in the stream.
    pub ac1_run: [u32; NUM_PLANES],
}

impl HuffmanRunState {
    /// Fresh state (all runs exhausted) for the start of a frame.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Decode a §13.4 EOB/DC-zero block-run length from raw bits.
///
/// Implements the §13.4 listing exactly (the printed pseudo-code *is*
/// the Figure 17 fixed tree):
///
/// ```text
/// EOBRunCount = 1 + R(2)
/// if (EOBRunCount == 3)      EOBRunCount += R(2)
/// else if (EOBRunCount == 4) EOBRunCount = R(1) ? 11 + R(6) : 7 + R(2)
/// ```
///
/// Value space: `1..=74`. The count is *inclusive* of the block whose
/// token triggered it (see the module-level disambiguation note #3;
/// the §13.3.2 AC1 listing's "EOB Token Run − 1" store pins the same
/// convention).
pub fn decode_eob_or_dc0_run(r: &mut RawBitReader<'_>) -> Result<u32, Error> {
    let mut run = 1 + r.read(2).map_err(|_| Error::Truncated)?;
    if run == 3 {
        run += r.read(2).map_err(|_| Error::Truncated)?;
    } else if run == 4 {
        if r.read(1).map_err(|_| Error::Truncated)? != 0 {
            run = 11 + r.read(6).map_err(|_| Error::Truncated)?;
        } else {
            run = 7 + r.read(2).map_err(|_| Error::Truncated)?;
        }
    }
    Ok(run)
}

/// Largest §13.4 block-run value (`11 + 63`).
pub const MAX_EOB_DC0_RUN: u32 = 74;

/// Encode a §13.4 EOB/DC-zero block-run length — the bit-for-bit
/// inverse of [`decode_eob_or_dc0_run`]. `run` must be in `1..=74`.
pub fn encode_eob_or_dc0_run(w: &mut BitWriter, run: u32) {
    debug_assert!((1..=MAX_EOB_DC0_RUN).contains(&run));
    match run {
        1 | 2 => w.write_u32(run - 1, 2),
        3..=6 => {
            w.write_u32(2, 2);
            w.write_u32(run - 3, 2);
        }
        7..=10 => {
            w.write_u32(3, 2);
            w.write_u32(0, 1);
            w.write_u32(run - 7, 2);
        }
        _ => {
            w.write_u32(3, 2);
            w.write_u32(1, 1);
            w.write_u32(run - 11, 6);
        }
    }
}

/// Number of raw magnitude bits a Huffman-coded value token carries
/// (§13.2.2 / §13.3.2: `R(token − 4)` for the categories, `R(11)` for
/// category 6, none for `ONE..FOUR`).
#[inline]
fn huff_magnitude_bits(token: DctToken) -> u32 {
    match token {
        DctToken::Category1 => 1,
        DctToken::Category2 => 2,
        DctToken::Category3 => 3,
        DctToken::Category4 => 4,
        DctToken::Category5 => 5,
        DctToken::Category6 => 11,
        _ => 0,
    }
}

/// Read the raw extra bits of a Huffman value token (`ONE_TOKEN` ..
/// `DCT_VAL_CATEGORY6`): magnitude offset (MSB first) then sign, per
/// §13.2.2 / §13.3.2 ("In Huffman encodings these bits are just pumped
/// on to the bitstream", sign last).
fn read_huff_token_value(r: &mut RawBitReader<'_>, token: DctToken) -> Result<i32, Error> {
    let mut value = token.min_value() as i32;
    let mag_bits = huff_magnitude_bits(token);
    if mag_bits > 0 {
        value += r.read(mag_bits).map_err(|_| Error::Truncated)? as i32;
    }
    let sign = r.read(1).map_err(|_| Error::Truncated)?;
    Ok(if sign != 0 { -value } else { value })
}

/// Emit the raw extra bits of a Huffman value token — inverse of
/// [`read_huff_token_value`]. `coeff` must be non-zero and `token` must
/// be `token_for_magnitude(|coeff|)`.
fn write_huff_token_value(w: &mut BitWriter, token: DctToken, coeff: i32) {
    let magnitude = coeff.unsigned_abs();
    let mag_bits = huff_magnitude_bits(token);
    if mag_bits > 0 {
        w.write_u32(magnitude - token.min_value() as u32, mag_bits);
    }
    w.write_u32(u32::from(coeff < 0), 1);
}

/// Decode one 8×8 block of Huffman-coded DCT coefficients (§13.2.2 DC +
/// §13.3.2 AC + §13.3.3.2 zero runs + §13.4 block runs), in scan order.
///
/// `state` carries the cross-block `CurrentDcRunLen` /
/// `CurrentAc1RunLen` counters; thread one instance across the whole
/// frame's blocks in stream order. Returns the same [`BlockCoeffs`]
/// shape the arithmetic decoder produces (`coeffs[0]` is the DC
/// prediction *delta* exactly as tokenized; §14 reconstruction is the
/// caller's).
pub fn decode_block_coefficients_huffman(
    r: &mut RawBitReader<'_>,
    plane: AcPlane,
    tables: &HuffmanCoeffTables,
    state: &mut HuffmanRunState,
) -> Result<BlockCoeffs, Error> {
    let p = plane.index();
    let mut coeffs = [0i32; BLOCK_SIZE];

    // ---- §13.2.2 DC ----
    if state.dc_run[p] > 0 {
        state.dc_run[p] -= 1;
        // DC is 0, no bits read.
    } else {
        let token = DctToken::from_index(tables.dc[p].decode(r)?).ok_or(Error::Truncated)?;
        match token {
            DctToken::Zero => {
                // §13.4 DC zero run, inclusive of this block
                // (disambiguation note #3). NOTE: §13.2.2's prose
                // ("the number of additional blocks") and its listing
                // (`CurrentDcRunLen[Plane] = DC Run Length`, stored
                // unmodified) both read as run-EXclusive of this
                // block; the third-party fixture has not yet
                // arbitrated between the two (both parses desync
                // further downstream for an unrelated reason), so the
                // r384 inclusive reading — which the in-tree encoder
                // mirrors — is retained for now.
                let run = decode_eob_or_dc0_run(r)?;
                state.dc_run[p] = run - 1;
            }
            DctToken::EndOfBlock => {
                // Table 19: "EOB_TOKEN / DC: Not allowed!"
                return Err(Error::Truncated);
            }
            _ => {
                coeffs[0] = read_huff_token_value(r, token)?;
            }
        }
    }

    // ---- §13.3.2 AC ----
    let mut prec = AcPrecContext::seed_from_dc(coeffs[0]);
    let mut encoded: usize = 1;

    if state.ac1_run[p] > 0 {
        state.ac1_run[p] -= 1;
        // All ACs are 0, no bits read.
        return Ok(BlockCoeffs {
            coeffs,
            coeff_count: 1,
        });
    }

    while encoded < BLOCK_SIZE {
        let band = huff_ac_band(encoded).expect("positions 1..=63 map to a Table 36 band");
        let token = DctToken::from_index(tables.ac_table(plane, prec, band).decode(r)?)
            .ok_or(Error::Truncated)?;
        match token {
            DctToken::Zero => {
                // §13.3.3.2 zero run (inclusive of this position; see
                // disambiguation notes #1/#2).
                let zband = ZrlBand::for_coefficient_position(encoded)
                    .expect("positions 1..=63 map to a Table 37 band");
                let sym = tables.zrl[zband.index()].decode(r)?;
                let run = if sym < NUM_ZRL_HUFF_PROBS - 1 {
                    sym as u32 + 1
                } else {
                    9 + r.read(6).map_err(|_| Error::Truncated)?
                };
                prec = AcPrecContext::WasZero;
                encoded += run as usize;
            }
            DctToken::EndOfBlock => {
                if encoded == 1 {
                    // §13.4 AC1 EOB run: "CurrentAc1RunLen[Plane] =
                    // EOB Token Run - 1".
                    let run = decode_eob_or_dc0_run(r)?;
                    state.ac1_run[p] = run - 1;
                }
                break;
            }
            _ => {
                let value = read_huff_token_value(r, token)?;
                coeffs[encoded] = value;
                // §13.3.2: `Prec = (value > 1) ? 2 : 1` — the listing's
                // `value` is the unsigned magnitude accumulator.
                prec = if value.unsigned_abs() > 1 {
                    AcPrecContext::WasGreaterThanOne
                } else {
                    AcPrecContext::WasOne
                };
                encoded += 1;
            }
        }
    }

    Ok(BlockCoeffs {
        coeffs,
        coeff_count: encoded.min(BLOCK_SIZE),
    })
}

/// Encode a whole frame's blocks (in stream order, each tagged with its
/// Table 28 plane) as the §13 Huffman coefficient partition — the
/// bit-for-bit inverse of driving [`decode_block_coefficients_huffman`]
/// over the same sequence.
///
/// Each entry is `(plane, scan_coeffs)` where `scan_coeffs[0]` is the
/// §14 DC *delta* (the tokenized value) and `scan_coeffs[1..]` the AC
/// coefficients in the active scan order — exactly what the arithmetic
/// encoder hands `encode_block_coefficients`.
///
/// The frame-level view is what makes the §13.2.2 / §13.3.2 cross-block
/// runs encodable: a `ZERO_TOKEN` DC carries the §13.4 count of
/// *consecutive same-plane* blocks (this one included) whose DC delta
/// is 0, and an `EOB_TOKEN` at AC 1 the count of consecutive same-plane
/// all-AC-zero blocks; the emitter looks ahead in `blocks` to size each
/// run (capped at [`MAX_EOB_DC0_RUN`]) and then skips the covered
/// blocks' bits exactly as the decoder's [`HuffmanRunState`] does.
pub fn encode_frame_blocks_huffman(
    w: &mut BitWriter,
    tables: &HuffmanCoeffTables,
    blocks: &[(AcPlane, [i32; BLOCK_SIZE])],
) {
    let mut state = HuffmanRunState::new();

    for (i, &(plane, ref scan)) in blocks.iter().enumerate() {
        let p = plane.index();

        // ---- DC ----
        if state.dc_run[p] > 0 {
            state.dc_run[p] -= 1;
            // Invariant: the lookahead sized the run over zero-DC
            // blocks only.
            debug_assert_eq!(scan[0], 0);
        } else if scan[0] == 0 {
            // Size the inclusive same-plane zero-DC run by lookahead.
            let mut run = 1u32;
            for (q, s) in blocks.iter().skip(i + 1) {
                if q.index() != p {
                    continue;
                }
                if s[0] != 0 || run == MAX_EOB_DC0_RUN {
                    break;
                }
                run += 1;
            }
            tables.dc[p].encode(w, DctToken::Zero.index());
            encode_eob_or_dc0_run(w, run);
            state.dc_run[p] = run - 1;
        } else {
            let token = token_for_magnitude(scan[0].unsigned_abs() as u16);
            tables.dc[p].encode(w, token.index());
            write_huff_token_value(w, token, scan[0]);
        }

        // ---- AC ----
        if state.ac1_run[p] > 0 {
            state.ac1_run[p] -= 1;
            // Invariant: the lookahead sized the run over all-AC-zero
            // blocks only.
            debug_assert!(scan[1..].iter().all(|&c| c == 0));
            continue;
        }

        let mut prec = AcPrecContext::seed_from_dc(scan[0]);
        let all_ac_zero = scan[1..].iter().all(|&c| c == 0);
        if all_ac_zero {
            // EOB at AC 1 with a §13.4 inclusive same-plane run.
            let mut run = 1u32;
            for (q, s) in blocks.iter().skip(i + 1) {
                if q.index() != p {
                    continue;
                }
                if s[1..].iter().any(|&c| c != 0) || run == MAX_EOB_DC0_RUN {
                    break;
                }
                run += 1;
            }
            tables
                .ac_table(plane, prec, 0)
                .encode(w, DctToken::EndOfBlock.index());
            encode_eob_or_dc0_run(w, run);
            state.ac1_run[p] = run - 1;
            continue;
        }

        let last_nonzero = scan
            .iter()
            .rposition(|&c| c != 0)
            .expect("all_ac_zero is false");
        let mut encoded: usize = 1;
        while encoded < BLOCK_SIZE {
            let band = huff_ac_band(encoded).expect("positions 1..=63 map to a Table 36 band");
            if encoded > last_nonzero {
                // Trailing zeros: EOB (past AC 1 it carries no run).
                tables
                    .ac_table(plane, prec, band)
                    .encode(w, DctToken::EndOfBlock.index());
                break;
            }
            let coeff = scan[encoded];
            if coeff == 0 {
                // Inclusive zero run up to (not past) the last nonzero.
                let mut run = 1usize;
                while encoded + run <= last_nonzero && scan[encoded + run] == 0 {
                    run += 1;
                }
                tables
                    .ac_table(plane, prec, band)
                    .encode(w, DctToken::Zero.index());
                let zband = ZrlBand::for_coefficient_position(encoded)
                    .expect("positions 1..=63 map to a Table 37 band");
                if run < NUM_ZRL_HUFF_PROBS {
                    tables.zrl[zband.index()].encode(w, run - 1);
                } else {
                    tables.zrl[zband.index()].encode(w, NUM_ZRL_HUFF_PROBS - 1);
                    w.write_u32(run as u32 - 9, 6);
                }
                prec = AcPrecContext::WasZero;
                encoded += run;
            } else {
                let token = token_for_magnitude(coeff.unsigned_abs() as u16);
                tables.ac_table(plane, prec, band).encode(w, token.index());
                write_huff_token_value(w, token, coeff);
                prec = if coeff.unsigned_abs() > 1 {
                    AcPrecContext::WasGreaterThanOne
                } else {
                    AcPrecContext::WasOne
                };
                encoded += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_tables() -> HuffmanCoeffTables {
        HuffmanCoeffTables::from_banks(&CoeffProbBanks::keyframe())
    }

    // ---- §13.4 EOB/DC0 run round-trip ----

    /// Every §13.4 run value 1..=74 encodes and decodes back to itself.
    #[test]
    fn eob_dc0_run_round_trips_full_range() {
        for run in 1..=MAX_EOB_DC0_RUN {
            let mut w = BitWriter::new();
            encode_eob_or_dc0_run(&mut w, run);
            let bytes = w.finish();
            let mut r = RawBitReader::new(&bytes);
            assert_eq!(decode_eob_or_dc0_run(&mut r).unwrap(), run, "run {run}");
        }
    }

    /// The §13.4 decode consumes exactly the bits the encoder emitted:
    /// concatenated runs decode in sequence.
    #[test]
    fn eob_dc0_runs_concatenate() {
        let runs = [1u32, 4, 7, 10, 11, 74, 2, 3, 6];
        let mut w = BitWriter::new();
        for &run in &runs {
            encode_eob_or_dc0_run(&mut w, run);
        }
        let bytes = w.finish();
        let mut r = RawBitReader::new(&bytes);
        for &run in &runs {
            assert_eq!(decode_eob_or_dc0_run(&mut r).unwrap(), run);
        }
    }

    // ---- Table 36 band map ----

    #[test]
    fn huff_ac_band_matches_table_36() {
        assert_eq!(huff_ac_band(0), None);
        assert_eq!(huff_ac_band(1), Some(0));
        for i in 2..=4 {
            assert_eq!(huff_ac_band(i), Some(1), "pos {i}");
        }
        for i in 5..=10 {
            assert_eq!(huff_ac_band(i), Some(2), "pos {i}");
        }
        for i in 11..=63 {
            assert_eq!(huff_ac_band(i), Some(3), "pos {i}");
        }
        assert_eq!(huff_ac_band(64), None);
    }

    // ---- magnitude-bit counts (§13.2.2/§13.3.2 R(token-4) / R(11)) ----

    #[test]
    fn huffman_magnitude_bit_counts() {
        assert_eq!(huff_magnitude_bits(DctToken::One), 0);
        assert_eq!(huff_magnitude_bits(DctToken::Four), 0);
        assert_eq!(huff_magnitude_bits(DctToken::Category1), 1);
        assert_eq!(huff_magnitude_bits(DctToken::Category2), 2);
        assert_eq!(huff_magnitude_bits(DctToken::Category3), 3);
        assert_eq!(huff_magnitude_bits(DctToken::Category4), 4);
        assert_eq!(huff_magnitude_bits(DctToken::Category5), 5);
        assert_eq!(huff_magnitude_bits(DctToken::Category6), 11);
    }

    // ---- single-block round-trips ----

    fn round_trip(blocks: &[(AcPlane, [i32; BLOCK_SIZE])]) -> Vec<BlockCoeffs> {
        let tables = baseline_tables();
        let mut w = BitWriter::new();
        encode_frame_blocks_huffman(&mut w, &tables, blocks);
        let bytes = w.finish();
        let mut r = RawBitReader::new(&bytes);
        let mut state = HuffmanRunState::new();
        blocks
            .iter()
            .map(|&(plane, _)| {
                decode_block_coefficients_huffman(&mut r, plane, &tables, &mut state)
                    .expect("decode")
            })
            .collect()
    }

    fn assert_blocks_match(blocks: &[(AcPlane, [i32; BLOCK_SIZE])]) {
        let decoded = round_trip(blocks);
        for (i, ((_, expect), got)) in blocks.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(&got.coeffs, expect, "block {i}");
        }
    }

    /// An empty block (DC 0, all ACs 0) round-trips: ZERO_TOKEN DC with
    /// a run of 1, EOB at AC 1 with a run of 1.
    #[test]
    fn empty_block_round_trips() {
        assert_blocks_match(&[(AcPlane::Y, [0i32; BLOCK_SIZE])]);
    }

    /// A DC-only block round-trips across the full signed token range,
    /// including every category boundary.
    #[test]
    fn dc_only_blocks_round_trip() {
        for dc in [
            1i32, -1, 2, -2, 3, 4, 5, 6, -6, 7, 10, 11, 18, 19, 34, 35, 66, 67, -67, 100, 2114,
            -2114,
        ] {
            let mut scan = [0i32; BLOCK_SIZE];
            scan[0] = dc;
            assert_blocks_match(&[(AcPlane::Y, scan)]);
        }
    }

    /// Scattered AC values with interior zero runs round-trip, including
    /// values in every magnitude category and both ZRL bands.
    #[test]
    fn scattered_ac_block_round_trips() {
        let mut scan = [0i32; BLOCK_SIZE];
        scan[0] = -3;
        scan[1] = 1;
        scan[2] = -2114; // category 6
        scan[5] = 7; // after a short zero run
        scan[20] = -35; // after a long zero run crossing into ZRL band 1
        scan[63] = 1; // last scan position
        assert_blocks_match(&[(AcPlane::Y, scan)]);
    }

    /// A zero run longer than 8 uses the long escape (`9 + R(6)`).
    #[test]
    fn long_zero_run_uses_escape() {
        let mut scan = [0i32; BLOCK_SIZE];
        scan[0] = 1;
        scan[1] = 2;
        scan[40] = -4; // 38 consecutive zeros at positions 2..=39
        assert_blocks_match(&[(AcPlane::Y, scan)]);
    }

    /// A fully-populated block (no EOB, natural termination at 64)
    /// round-trips.
    #[test]
    fn full_block_round_trips() {
        let mut scan = [0i32; BLOCK_SIZE];
        for (i, c) in scan.iter_mut().enumerate() {
            *c = if i % 2 == 0 {
                1 + i as i32
            } else {
                -(i as i32) - 1
            };
        }
        assert_blocks_match(&[(AcPlane::Y, scan)]);
    }

    /// Zero DC with non-zero ACs: the DC ZERO_TOKEN run coexists with a
    /// real AC decode in the same block.
    #[test]
    fn zero_dc_with_ac_round_trips() {
        let mut scan = [0i32; BLOCK_SIZE];
        scan[1] = 5;
        scan[2] = -1;
        assert_blocks_match(&[(AcPlane::Y, scan)]);
    }

    // ---- cross-block run round-trips ----

    /// A run of same-plane zero-DC blocks is amortised: only the first
    /// block spends DC bits; the followers decode DC 0 from the run
    /// counter. Mixed AC content is preserved throughout.
    #[test]
    fn dc_zero_run_spans_blocks() {
        let mut b0 = [0i32; BLOCK_SIZE];
        b0[1] = 3;
        let mut b1 = [0i32; BLOCK_SIZE];
        b1[2] = -7;
        let mut b2 = [0i32; BLOCK_SIZE];
        b2[0] = 4; // DC run must stop before this block
        b2[1] = 1;
        assert_blocks_match(&[
            (AcPlane::Y, b0),
            (AcPlane::Y, b1),
            (AcPlane::Y, [0i32; BLOCK_SIZE]),
            (AcPlane::Y, b2),
        ]);
    }

    /// An AC1 EOB run spans consecutive all-AC-zero blocks of the same
    /// plane; a block with real ACs terminates it.
    #[test]
    fn ac1_eob_run_spans_blocks() {
        let mut dc_only = [0i32; BLOCK_SIZE];
        dc_only[0] = 9;
        let mut with_ac = [0i32; BLOCK_SIZE];
        with_ac[0] = -2;
        with_ac[3] = 6;
        assert_blocks_match(&[
            (AcPlane::Y, dc_only),
            (AcPlane::Y, [0i32; BLOCK_SIZE]),
            (AcPlane::Y, dc_only),
            (AcPlane::Y, with_ac),
        ]);
    }

    /// DC and AC1 runs are tracked per plane: interleaved Y and UV
    /// blocks maintain independent counters (U and V share the UV
    /// counter, exactly like the probability banks).
    #[test]
    fn runs_are_per_plane() {
        let mut y_busy = [0i32; BLOCK_SIZE];
        y_busy[0] = 2;
        y_busy[1] = -1;
        assert_blocks_match(&[
            (AcPlane::Y, [0i32; BLOCK_SIZE]),
            (AcPlane::UV, [0i32; BLOCK_SIZE]),
            (AcPlane::Y, [0i32; BLOCK_SIZE]),
            (AcPlane::UV, [0i32; BLOCK_SIZE]),
            (AcPlane::Y, y_busy),
            (AcPlane::UV, [0i32; BLOCK_SIZE]),
        ]);
    }

    /// A same-plane zero-DC run longer than the §13.4 maximum (74) is
    /// split into successive runs.
    #[test]
    fn dc_run_longer_than_74_splits() {
        let blocks: Vec<(AcPlane, [i32; BLOCK_SIZE])> =
            (0..80).map(|_| (AcPlane::Y, [0i32; BLOCK_SIZE])).collect();
        assert_blocks_match(&blocks);
    }

    /// The §13.3.2 Prec context threads across tokens: a mid-block
    /// sequence 1, 2, 0-run, 1 exercises WasOne → WasGreaterThanOne →
    /// WasZero transitions on the encode side matching the decode side
    /// (a mismatch would select a different tree and desynchronise the
    /// bitstream).
    #[test]
    fn prec_context_transitions_round_trip() {
        let mut scan = [0i32; BLOCK_SIZE];
        scan[0] = 1;
        scan[1] = 1;
        scan[2] = 2;
        scan[4] = 1; // zero at 3 → WasZero context for 4
        scan[5] = -66;
        scan[6] = 1;
        assert_blocks_match(&[(AcPlane::Y, scan)]);
    }

    /// Tables built from *updated* (non-baseline) banks still round-trip
    /// — the encoder and decoder derive identical trees from the same
    /// banks.
    #[test]
    fn updated_banks_round_trip() {
        let mut banks = CoeffProbBanks::keyframe();
        banks.dc_probs[0][0] = 200;
        banks.dc_probs[1][4] = 64;
        banks.ac_probs[0][0][0][0] = 100;
        banks.ac_probs[1][2][3][5] = 220;
        banks.zrl_probs[0][2] = 80;
        banks.zrl_probs[1][9] = 2;
        let tables = HuffmanCoeffTables::from_banks(&banks);

        let mut scan = [0i32; BLOCK_SIZE];
        scan[0] = -5;
        scan[1] = 1;
        scan[9] = 12;
        let blocks = [(AcPlane::Y, scan), (AcPlane::UV, scan)];

        let mut w = BitWriter::new();
        encode_frame_blocks_huffman(&mut w, &tables, &blocks);
        let bytes = w.finish();
        let mut r = RawBitReader::new(&bytes);
        let mut state = HuffmanRunState::new();
        for &(plane, ref expect) in &blocks {
            let got = decode_block_coefficients_huffman(&mut r, plane, &tables, &mut state)
                .expect("decode");
            assert_eq!(&got.coeffs, expect);
        }
    }

    /// Extreme node probabilities can drive a §13.1 leaf product to 0;
    /// the tree build clamps to 1 (§7 forbids 0) and decode still works.
    #[test]
    fn zero_leaf_probability_is_clamped() {
        let mut banks = CoeffProbBanks::keyframe();
        // Drive the DC plane-0 bank to extremes so several products
        // truncate to 0.
        banks.dc_probs[0] = [255, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let tables = HuffmanCoeffTables::from_banks(&banks);

        let mut scan = [0i32; BLOCK_SIZE];
        scan[0] = 2114; // deepest category token
        let blocks = [(AcPlane::Y, scan)];
        let mut w = BitWriter::new();
        encode_frame_blocks_huffman(&mut w, &tables, &blocks);
        let bytes = w.finish();
        let mut r = RawBitReader::new(&bytes);
        let mut state = HuffmanRunState::new();
        let got = decode_block_coefficients_huffman(&mut r, AcPlane::Y, &tables, &mut state)
            .expect("decode");
        assert_eq!(got.coeffs[0], 2114);
    }

    /// Truncated input surfaces `Error::Truncated`, not a panic.
    #[test]
    fn truncated_stream_errors() {
        let tables = baseline_tables();
        let mut state = HuffmanRunState::new();
        let mut r = RawBitReader::new(&[]);
        assert!(matches!(
            decode_block_coefficients_huffman(&mut r, AcPlane::Y, &tables, &mut state),
            Err(Error::Truncated)
        ));
    }

    /// `coeff_count` conventions match the arithmetic decoder: EOB exit
    /// reports the EOB scan position, a natural full block reports 64,
    /// an AC1-run block reports 1.
    #[test]
    fn coeff_count_conventions() {
        // DC-only → EOB at position 1 → count 1 (and its run makes the
        // *next* same-plane block a run block, also count 1).
        let mut dc_only = [0i32; BLOCK_SIZE];
        dc_only[0] = 3;
        let decoded = round_trip(&[(AcPlane::Y, dc_only), (AcPlane::Y, dc_only)]);
        assert_eq!(decoded[0].coeff_count, 1);
        assert_eq!(decoded[1].coeff_count, 1);

        // Last nonzero at 5 → EOB at 6.
        let mut short = [0i32; BLOCK_SIZE];
        short[0] = 1;
        short[5] = -2;
        let decoded = round_trip(&[(AcPlane::Y, short)]);
        assert_eq!(decoded[0].coeff_count, 6);

        // Full block → 64.
        let mut full = [0i32; BLOCK_SIZE];
        for c in full.iter_mut() {
            *c = 1;
        }
        let decoded = round_trip(&[(AcPlane::Y, full)]);
        assert_eq!(decoded[0].coeff_count, 64);
    }
}

//! Per-block coefficient source — the §6 partition/coder dispatch for
//! the §13 DCT-token decode.
//!
//! A VP6 frame's DCT tokens live in one of three places (§5/§6):
//!
//! * **Single stream** (`MultiStream == 0`): BoolCoder-coded in
//!   partition 1, interleaved with the mode/MV data.
//! * **MultiStream + BoolCoder** (`MultiStream == 1`, `UseHuffman ==
//!   0`): BoolCoder-coded in partition 2 (its own `VP6_StartDecode`
//!   at `Buff2Offset`).
//! * **MultiStream + Huffman** (`MultiStream == 1`, `UseHuffman == 1`):
//!   raw-bit Huffman-coded in partition 2 (§7.2 / §13.2.2 / §13.3.2).
//!
//! [`CoeffSource`] abstracts the three shapes behind one
//! [`CoeffSource::decode_block`] so the frame drivers
//! ([`crate::intra_frame`], [`crate::inter_frame`]) run the identical
//! §14/§15/§16/§17 reconstruction regardless of where the tokens come
//! from. The arithmetic arms thread the §13.2 Table 26 DC context into
//! [`crate::block_decode::decode_block_coefficients_ctx`]; the Huffman
//! arm ignores it (§13.2.2 builds its DC trees from the raw un-contexted
//! per-plane bank) and instead threads the §13.2.2/§13.3.2 cross-block
//! run state.
//!
//! Sourced from `docs/video/vp6/vp6_format.pdf` §5/§6/§13; no external
//! library code was consulted.

use crate::block_decode::{decode_block_coefficients_ctx, BlockCoeffs};
use crate::bool_coder::BoolCoder;
use crate::huff_coeff::{decode_block_coefficients_huffman, HuffmanCoeffTables, HuffmanRunState};
use crate::intra_frame::IntraProbs;
use crate::raw_bits::RawBitReader;
use crate::tokens::{AcPlane, DcContext};
use crate::Error;

/// Where (and with which coder) a frame's §13 DCT tokens are read from.
/// See the module docs for the three §5/§6 shapes.
pub enum CoeffSource<'a, 'b> {
    /// BoolCoder-coded tokens. For a single-stream frame this borrows
    /// the partition-1 coder (tokens interleave with mode/MV data); for
    /// a MultiStream BoolCoder frame it borrows a second coder the
    /// caller constructed over partition 2 at `Buff2Offset`.
    Bool(&'b mut BoolCoder<'a>),
    /// Raw-bit Huffman-coded tokens from partition 2 (`UseHuffman`).
    Huffman {
        /// The §3 raw-bit reader over partition 2.
        reader: RawBitReader<'a>,
        /// The per-frame §13.1/§13.3.3.2-derived trees (built from the
        /// post-Figure-5 banks).
        tables: &'b HuffmanCoeffTables,
        /// The §13.2.2/§13.3.2 cross-block run counters.
        state: HuffmanRunState,
    },
}

impl core::fmt::Debug for CoeffSource<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `RawBitReader` intentionally carries no `Debug` derive; report
        // just the active transport arm.
        match self {
            CoeffSource::Bool(_) => f.write_str("CoeffSource::Bool(..)"),
            CoeffSource::Huffman { .. } => f.write_str("CoeffSource::Huffman { .. }"),
        }
    }
}

impl<'a, 'b> CoeffSource<'a, 'b> {
    /// Construct the Huffman arm with fresh run state.
    pub fn huffman(partition2: &'a [u8], tables: &'b HuffmanCoeffTables) -> Self {
        CoeffSource::Huffman {
            reader: RawBitReader::new(partition2),
            tables,
            state: HuffmanRunState::new(),
        }
    }

    /// Decode one 8×8 block of scan-order coefficients.
    ///
    /// `plane` is the Table 25/28 plane; `dc_context` the §13.2 Table 26
    /// left/above context (used by the arithmetic arms only); `probs`
    /// the frame's post-update banks (arithmetic arms only — the Huffman
    /// arm's trees were derived from the same banks at construction).
    ///
    /// The returned [`BlockCoeffs`] carries the DC prediction *delta* at
    /// scan position 0 exactly as tokenized, whichever coder ran.
    pub fn decode_block(
        &mut self,
        plane: AcPlane,
        dc_context: DcContext,
        probs: &IntraProbs,
    ) -> Result<BlockCoeffs, Error> {
        match self {
            CoeffSource::Bool(bc) => decode_block_coefficients_ctx(
                bc,
                plane,
                &probs.dc_contexts,
                dc_context,
                &probs.ac_probs,
                &probs.zrl_probs,
            ),
            CoeffSource::Huffman {
                reader,
                tables,
                state,
            } => decode_block_coefficients_huffman(reader, plane, tables, state),
        }
    }
}

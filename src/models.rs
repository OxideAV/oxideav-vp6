//! VP6 probability models — the mutable state that
//! `vp6_default_models_init` / `vp6_parse_coeff_models` / friends
//! maintain across a stream.

use crate::range_coder::RangeCoder;
use crate::tables;

/// Mutable probability model. Matches the layout FFmpeg's
/// `VP56Model` struct carries per decoder instance. Fields keep the
/// same names the FFmpeg source uses so cross-referencing is easy.
#[derive(Clone, Debug)]
pub struct Vp6Model {
    pub coeff_reorder: [u8; 64],
    pub coeff_index_to_pos: [u8; 64],
    pub coeff_index_to_idct_selector: [u8; 64],
    pub vector_sig: [u8; 2],
    pub vector_dct: [u8; 2],
    pub vector_pdv: [[u8; 7]; 2],
    pub vector_fdv: [[u8; 8]; 2],
    pub coeff_dccv: [[u8; 11]; 2],
    pub coeff_ract: [[[[u8; 11]; 6]; 3]; 2],
    pub coeff_dcct: [[[u8; 5]; 36]; 2],
    pub coeff_runv: [[u8; 14]; 2],
    pub mb_type: [[[u8; 10]; 10]; 3],
    pub mb_types_stats: [[[u8; 2]; 10]; 3],
}

impl Default for Vp6Model {
    fn default() -> Self {
        Self {
            coeff_reorder: [0u8; 64],
            coeff_index_to_pos: [0u8; 64],
            coeff_index_to_idct_selector: [0u8; 64],
            vector_sig: [0u8; 2],
            vector_dct: [0u8; 2],
            vector_pdv: [[0u8; 7]; 2],
            vector_fdv: [[0u8; 8]; 2],
            coeff_dccv: [[0u8; 11]; 2],
            coeff_ract: [[[[0u8; 11]; 6]; 3]; 2],
            coeff_dcct: [[[0u8; 5]; 36]; 2],
            coeff_runv: [[0u8; 14]; 2],
            mb_type: [[[0u8; 10]; 10]; 3],
            mb_types_stats: [[[0u8; 2]; 10]; 3],
        }
    }
}

impl Vp6Model {
    /// Populate the model the way `vp6_default_models_init` does on
    /// every keyframe. `interlaced` selects the coefficient reorder
    /// table; `sub_version` is threaded through so the idct-selector
    /// adjustment at sub_version > 6 fires.
    pub fn reset_defaults(&mut self, interlaced: bool, sub_version: u8) {
        self.vector_dct = [0xA2, 0xA4];
        self.vector_sig = [0x80, 0x80];
        self.mb_types_stats = tables::DEF_MB_TYPES_STATS;
        self.vector_fdv = tables::VP6_DEF_FDV_MODEL;
        self.vector_pdv = tables::VP6_DEF_PDV_MODEL;
        self.coeff_runv = tables::VP6_DEF_RUNV_MODEL;
        self.coeff_reorder = if interlaced {
            tables::VP6_IL_COEFF_REORDER
        } else {
            tables::VP6_DEF_COEFF_REORDER
        };
        self.rebuild_coeff_tables(sub_version);
    }

    /// Recompute `coeff_index_to_pos` + `coeff_index_to_idct_selector`
    /// after a reorder update. Matches `vp6_coeff_order_table_init` in
    /// FFmpeg.
    pub fn rebuild_coeff_tables(&mut self, sub_version: u8) {
        self.coeff_index_to_pos[0] = 0;
        let mut idx = 1usize;
        for i in 0..16u8 {
            for pos in 1..64usize {
                if self.coeff_reorder[pos] == i {
                    self.coeff_index_to_pos[idx] = pos as u8;
                    idx += 1;
                }
            }
        }
        for idx in 0..64usize {
            let mut max = 0u8;
            for i in 0..=idx {
                let v = self.coeff_index_to_pos[i];
                if v > max {
                    max = v;
                }
            }
            if sub_version > 6 {
                max = max.saturating_add(1);
            }
            self.coeff_index_to_idct_selector[idx] = max;
        }
    }

    /// Recompute the MB-type binary tree probabilities from the raw
    /// `mb_types_stats`. Matches the second half of
    /// `vp56_parse_mb_type_models`.
    pub fn rebuild_mb_type_probs(&mut self) {
        for ctx in 0..3usize {
            let mut p = [0i32; 10];
            for t in 0..10 {
                p[t] = 100 * self.mb_types_stats[ctx][t][1] as i32;
            }

            for t in 0..10 {
                let mut p_local = p;
                self.mb_type[ctx][t][0] = (255
                    - (255 * self.mb_types_stats[ctx][t][0] as i32)
                        / (1 + self.mb_types_stats[ctx][t][0] as i32
                            + self.mb_types_stats[ctx][t][1] as i32))
                    .clamp(0, 255) as u8;

                p_local[t] = 0;

                let p02 = p_local[0] + p_local[2];
                let p34 = p_local[3] + p_local[4];
                let p0234 = p02 + p34;
                let p17 = p_local[1] + p_local[7];
                let p56 = p_local[5] + p_local[6];
                let p89 = p_local[8] + p_local[9];
                let p5689 = p56 + p89;
                let p156789 = p17 + p5689;

                self.mb_type[ctx][t][1] =
                    (1 + 255 * p0234 / (1 + p0234 + p156789)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][2] = (1 + 255 * p02 / (1 + p0234)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][3] = (1 + 255 * p17 / (1 + p156789)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][4] = (1 + 255 * p_local[0] / (1 + p02)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][5] = (1 + 255 * p_local[3] / (1 + p34)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][6] = (1 + 255 * p_local[1] / (1 + p17)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][7] = (1 + 255 * p56 / (1 + p5689)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][8] = (1 + 255 * p_local[5] / (1 + p56)).clamp(0, 255) as u8;
                self.mb_type[ctx][t][9] = (1 + 255 * p_local[8] / (1 + p89)).clamp(0, 255) as u8;
            }
        }
    }
}

/// Port of `vp56_parse_mb_type_models` — inter-frame only.
pub fn parse_mb_type_models(model: &mut Vp6Model, rac: &mut RangeCoder<'_>) {
    for ctx in 0..3usize {
        if rac.get_prob(174) != 0 {
            let idx = rac.get_bits(4) as usize;
            model.mb_types_stats[ctx] = tables::PRE_DEF_MB_TYPE_STATS[idx][ctx];
        }
        if rac.get_prob(254) != 0 {
            for t in 0..10usize {
                for i in 0..2usize {
                    if rac.get_prob(205) != 0 {
                        let sign = rac.get_bit();
                        let mut delta =
                            rac.get_tree(tables::PMBTM_TREE, &tables::MB_TYPE_MODEL_MODEL);
                        if delta == 0 {
                            delta = 4 * rac.get_bits(7) as i32;
                        }
                        let signed = if sign != 0 { -delta } else { delta };
                        // FFmpeg does a plain `+=` on u8 which wraps
                        // modulo 256 — match that exactly.
                        model.mb_types_stats[ctx][t][i] =
                            (model.mb_types_stats[ctx][t][i] as i32 + signed) as u8;
                    }
                }
            }
        }
    }
    model.rebuild_mb_type_probs();
}

/// Port of `vp6_parse_vector_models`.
pub fn parse_vector_models(model: &mut Vp6Model, rac: &mut RangeCoder<'_>) {
    for comp in 0..2usize {
        if rac.get_prob(tables::VP6_SIG_DCT_PCT[comp][0]) != 0 {
            model.vector_dct[comp] = rac.get_bits_nn() as u8;
        }
        if rac.get_prob(tables::VP6_SIG_DCT_PCT[comp][1]) != 0 {
            model.vector_sig[comp] = rac.get_bits_nn() as u8;
        }
    }
    for comp in 0..2usize {
        for node in 0..7usize {
            if rac.get_prob(tables::VP6_PDV_PCT[comp][node]) != 0 {
                model.vector_pdv[comp][node] = rac.get_bits_nn() as u8;
            }
        }
    }
    for comp in 0..2usize {
        for node in 0..8usize {
            if rac.get_prob(tables::VP6_FDV_PCT[comp][node]) != 0 {
                model.vector_fdv[comp][node] = rac.get_bits_nn() as u8;
            }
        }
    }
}

/// Port of `vp6_parse_coeff_models`.
///
/// On keyframes (when `key` is set) un-updated nodes copy through the
/// current `def_prob[node]`, matching FFmpeg's
/// `else if (s->frames[..].flags & AV_FRAME_FLAG_KEY)` branch.
pub fn parse_coeff_models(
    model: &mut Vp6Model,
    rac: &mut RangeCoder<'_>,
    sub_version: u8,
    key: bool,
) {
    let mut def_prob = [0x80u8; 11];
    for pt in 0..2usize {
        for node in 0..11usize {
            if rac.get_prob(tables::VP6_DCCV_PCT[pt][node]) != 0 {
                def_prob[node] = rac.get_bits_nn() as u8;
                model.coeff_dccv[pt][node] = def_prob[node];
            } else if key {
                model.coeff_dccv[pt][node] = def_prob[node];
            }
        }
    }

    if rac.get_bit() != 0 {
        for pos in 1..64usize {
            if rac.get_prob(tables::VP6_COEFF_REORDER_PCT[pos]) != 0 {
                model.coeff_reorder[pos] = rac.get_bits(4) as u8;
            }
        }
        model.rebuild_coeff_tables(sub_version);
    }

    for cg in 0..2usize {
        for node in 0..14usize {
            if rac.get_prob(tables::VP6_RUNV_PCT[cg][node]) != 0 {
                model.coeff_runv[cg][node] = rac.get_bits_nn() as u8;
            }
        }
    }

    for ct in 0..3usize {
        for pt in 0..2usize {
            for cg in 0..6usize {
                for node in 0..11usize {
                    if rac.get_prob(tables::VP6_RACT_PCT[ct][pt][cg][node]) != 0 {
                        def_prob[node] = rac.get_bits_nn() as u8;
                        model.coeff_ract[pt][ct][cg][node] = def_prob[node];
                    } else if key {
                        model.coeff_ract[pt][ct][cg][node] = def_prob[node];
                    }
                }
            }
        }
    }

    // coeff_dcct is a linear combination of coeff_dccv — rebuild here.
    for pt in 0..2usize {
        for ctx in 0..3usize {
            for node in 0..5usize {
                let v = ((model.coeff_dccv[pt][node] as i32 * tables::VP6_DCCV_LC[ctx][node][0]
                    + 128)
                    >> 8)
                    + tables::VP6_DCCV_LC[ctx][node][1];
                model.coeff_dcct[pt][ctx][node] = v.clamp(1, 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_init_matches_reference() {
        let mut model = Vp6Model::default();
        model.reset_defaults(false, 0);
        assert_eq!(model.vector_dct, [0xA2, 0xA4]);
        assert_eq!(model.vector_sig, [0x80, 0x80]);
        assert_eq!(model.vector_fdv, tables::VP6_DEF_FDV_MODEL);
    }

    #[test]
    fn coeff_tables_reset_consistent() {
        let mut model = Vp6Model::default();
        model.reset_defaults(false, 0);
        // coeff_index_to_pos[0] always 0
        assert_eq!(model.coeff_index_to_pos[0], 0);
        // No duplicate positions — 64 unique positions.
        let mut seen = [false; 64];
        for &p in &model.coeff_index_to_pos {
            assert!(!seen[p as usize], "duplicate pos {p}");
            seen[p as usize] = true;
        }
    }
}

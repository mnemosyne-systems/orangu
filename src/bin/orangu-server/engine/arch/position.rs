// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Position-dependent terms several architectures share: the YaRN rotary
//! parameters as a GGUF declares them, the logarithmic attention
//! temperature, and SSMax — a learned per-head query scale over the same
//! `log(pos + 1)` basis.
//!
//! Not an architecture. `engine::arch::mistral` (`mistral3`) and
//! `engine::arch::mistral_large4` (`mistral-large4`) both read their rotary
//! parameters through [`yarn_rope_params`]; the temperature and SSMax are
//! the two shapes of one formula, `log(floor(pos / floor) + 1)`, the first
//! as a fixed scale on Q past the trained context and the second as a
//! learned `s * log(pos + 1) + b` per head at every position.

use anyhow::{Context, Result};

use crate::engine::loader::LoadedModel;
use crate::engine::tensor::{RopeLayout, RopeParams};

/// `rope.scaling.original_context_length`, falling back to the trained
/// context — upstream's `n_ctx_orig_yarn`.
pub(crate) fn original_context_length(loaded: &LoadedModel) -> usize {
    loaded
        .metadata_u64("rope.scaling.original_context_length")
        .map(|v| v as usize)
        .unwrap_or(loaded.config.n_ctx_train)
}

/// ggml's YaRN magnitude: `0.1 * mscale * ln(scale) + 1`, or `1` when the
/// context is not stretched.
fn get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// The rotary parameters a file declares, including YaRN when it asks for
/// it, with the magnitude term `llama-context.cpp` hands the rope kernel.
///
/// YaRN is opt-in per file: upstream only sets a nonzero `ext_factor` when
/// `rope.scaling.type` is literally `"yarn"`, and with `ext_factor == 0`
/// both the ramp and the magnitude correction are skipped, leaving a plain
/// rope.
///
/// `attn_factor` follows upstream exactly. With
/// `rope.scaling.yarn_log_multiplier` set it is the ratio
/// `get_mscale(f, 1) / get_mscale(f, log_mul)` — `1.0` whenever the
/// multiplier is `1`, which is every `mistral3` and `mistral-large4` file;
/// without it, `get_mscale(f, 1)`. Either way it is then divided by the
/// `1 + 0.1 * ln(f)` the rope kernel multiplies back in, and finally scaled
/// by the file's own `rope.scaling.attn_factor`.
pub(crate) fn yarn_rope_params(
    loaded: &LoadedModel,
    n_ctx_orig: usize,
    layout: RopeLayout,
) -> RopeParams {
    let yarn = loaded.metadata_string("rope.scaling.type").as_deref() == Some("yarn");
    let scale_factor = loaded.metadata_f32("rope.scaling.factor").unwrap_or(0.0);
    let freq_scale = if scale_factor == 0.0 {
        1.0
    } else {
        1.0 / scale_factor
    };
    let mut attn_factor = 1.0f32;
    if yarn {
        let factor = 1.0 / freq_scale;
        let log_mul = loaded
            .metadata_f32("rope.scaling.yarn_log_multiplier")
            .unwrap_or(0.0);
        attn_factor = if log_mul != 0.0 {
            get_mscale(factor, 1.0) / get_mscale(factor, log_mul)
        } else {
            get_mscale(factor, 1.0)
        };
        attn_factor *= 1.0 / (1.0 + 0.1 * factor.ln());
    }
    attn_factor *= loaded
        .metadata_f32("rope.scaling.attn_factor")
        .unwrap_or(1.0);
    RopeParams {
        rope_dim: loaded.config.rope_dim,
        freq_base: loaded.config.rope_freq_base,
        freq_scale,
        ext_factor: if yarn { 1.0 } else { 0.0 },
        attn_factor,
        beta_fast: loaded
            .metadata_f32("rope.scaling.yarn_beta_fast")
            .unwrap_or(32.0),
        beta_slow: loaded
            .metadata_f32("rope.scaling.yarn_beta_slow")
            .unwrap_or(1.0),
        n_ctx_orig,
        layout,
    }
}

/// `log(floor(pos / floor_scale) + 1) * scale + 1` —
/// `llm_graph_input_attn_temp::set_input`'s logarithmic attention
/// temperature.
///
/// Exactly `1.0` for every position below `floor_scale`, because the floor
/// is then 0 and `log(1) == 0`: with `floor_scale = n_ctx_orig` (`mistral3`)
/// it only starts damping attention past the trained context.
pub(crate) fn log_position_temperature(pos: usize, floor_scale: usize, scale: f32) -> f32 {
    ((pos / floor_scale.max(1)) as f32 + 1.0).ln() * scale + 1.0
}

/// SSMax ("scalable softmax"): a learned per-head scale on the query,
/// `s[h] * log(pos + 1) + b[h]`, so that a head can sharpen its softmax as
/// the number of keys it spreads over grows.
///
/// Upstream builds the `log(pos + 1)` basis from the attention-temperature
/// input with `floor = 1`, `scale = 1` and subtracts the `+ 1` back off —
/// [`log_position_temperature`]`(pos, 1, 1.0) - 1`. Being one scalar per
/// head, it commutes with the rotation and with MLA's query absorption, so
/// it may be applied to the projected query at any point before the dot
/// product.
pub(crate) struct Ssmax {
    scale: Vec<f32>,
    bias: Vec<f32>,
}

impl Ssmax {
    /// Reads `<name>.weight` and `<name>.bias`, or `None` when the layer
    /// carries no `<name>.weight`. Both must be one value per head.
    pub(crate) fn load(loaded: &LoadedModel, name: &str, n_head: usize) -> Result<Option<Self>> {
        let weight = format!("{name}.weight");
        if !loaded.has_tensor(&weight) {
            return Ok(None);
        }
        let bias = format!("{name}.bias");
        let (scale, _) = loaded
            .tensor(&weight)
            .with_context(|| format!("loading {weight}"))?;
        let (bias_v, _) = loaded
            .tensor(&bias)
            .with_context(|| format!("loading {bias}"))?;
        anyhow::ensure!(
            scale.len() == n_head && bias_v.len() == n_head,
            "{name} has {}/{} values, not one per head ({n_head})",
            scale.len(),
            bias_v.len()
        );
        Ok(Some(Self {
            scale,
            bias: bias_v,
        }))
    }

    /// Head `h`'s factor at `pos`.
    pub(crate) fn factor(&self, h: usize, pos: usize) -> f32 {
        let basis = log_position_temperature(pos, 1, 1.0) - 1.0;
        self.scale[h] * basis + self.bias[h]
    }

    /// Scales each of the `n_head` heads of one token's `q`
    /// (`[n_head, head_dim]`) by its factor at `pos`.
    pub(crate) fn apply(&self, q: &mut [f32], head_dim: usize, pos: usize) {
        for (h, head) in q.chunks_exact_mut(head_dim).enumerate() {
            let f = self.factor(h, pos);
            for v in head {
                *v *= f;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn from_parts(scale: Vec<f32>, bias: Vec<f32>) -> Self {
        Self { scale, bias }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_temperature_is_identity_within_the_floor() {
        for pos in [0usize, 1, 100, 16383] {
            assert_eq!(log_position_temperature(pos, 16384, 0.1), 1.0, "pos {pos}");
        }
        assert!(log_position_temperature(16384, 16384, 0.1) > 1.0);
        assert!(
            log_position_temperature(65536, 16384, 0.1)
                > log_position_temperature(16384, 16384, 0.1)
        );
    }

    /// SSMax's basis is `log(pos + 1)`, so position 0 is the bias alone —
    /// a model whose first token is scaled by `s` instead of `b` has its
    /// basis off by one.
    #[test]
    fn ssmax_is_the_bias_at_position_zero_and_grows_with_log_position() {
        let ssmax = Ssmax::from_parts(vec![0.5, 2.0], vec![1.0, 0.25]);
        assert!((ssmax.factor(0, 0) - 1.0).abs() < 1e-6);
        assert!((ssmax.factor(1, 0) - 0.25).abs() < 1e-6);
        let e = std::f32::consts::E;
        // pos + 1 = e^2  ->  log = 2
        let pos = (e * e - 1.0).round() as usize;
        let expected = 0.5 * ((pos + 1) as f32).ln() + 1.0;
        assert!((ssmax.factor(0, pos) - expected).abs() < 1e-5);
    }

    #[test]
    fn ssmax_scales_each_head_by_its_own_factor() {
        let ssmax = Ssmax::from_parts(vec![0.0, 0.0], vec![2.0, 3.0]);
        let mut q = vec![1.0, 1.0, 1.0, 1.0];
        ssmax.apply(&mut q, 2, 7);
        assert_eq!(q, vec![2.0, 2.0, 3.0, 3.0]);
    }

    /// With `yarn_log_multiplier = 1` the two `get_mscale` calls cancel and
    /// only the pre-cancellation of the kernel's own magnitude remains.
    #[test]
    fn a_unit_log_multiplier_leaves_only_the_kernel_cancellation() {
        let f = 16.0f32;
        let ratio = get_mscale(f, 1.0) / get_mscale(f, 1.0);
        assert_eq!(ratio, 1.0);
        assert_eq!(get_mscale(1.0, 1.0), 1.0, "an unstretched context");
    }
}

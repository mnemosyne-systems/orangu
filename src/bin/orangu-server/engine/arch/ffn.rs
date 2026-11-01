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

//! The DeepSeek-style feed-forward layer: a dense SwiGLU block on the first
//! `leading_dense_block_count` layers, then routed SwiGLU experts plus one
//! always-on shared expert on every layer after.
//!
//! Not an architecture. `engine::arch::dsa` (`glm-dsa`, `mistral-large4`),
//! `engine::arch::glm5` and `engine::arch::bailingmoe` all load and run it
//! through [`LayerFfn`]; what differs between them is only the routing
//! rules ([`super::ExpertRouting`]) and the SwiGLU clamps ([`FfnLimits`]).

use anyhow::{Context, Result};

use super::{ExpertRouting, FfnScratch, SwigluLimit, SwigluMoe, SwigluSharedExpert};
use crate::engine::backend::Backend;
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, QuantMatrix};

/// Routed experts plus the always-on shared one.
pub(crate) struct SharedExpertMoe {
    gate_inp: QuantMatrix,
    /// `exp_probs_b.bias` — steers the expert *selection* only.
    exp_probs_b: Option<Vec<f32>>,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    gate_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
}

/// One layer's FFN: dense for the leading blocks, routed MoE for the rest.
pub(crate) enum LayerFfn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    /// Boxed: the MoE arm is several times the size of the dense one, and
    /// every layer carries one.
    Moe(Box<SharedExpertMoe>),
}

/// The SwiGLU clamps one layer applies — `swiglu_clamp_exp` /
/// `swiglu_clamp_shexp`, in whichever form the architecture takes them.
/// [`FfnLimits::NONE`] for a file that declares none.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfnLimits {
    /// The dense arm's limit. Only `glm5-next` clamps its dense blocks.
    pub dense: SwigluLimit,
    pub exp: SwigluLimit,
    pub shexp: SwigluLimit,
}

impl FfnLimits {
    pub(crate) const NONE: Self = Self {
        dense: SwigluLimit::None,
        exp: SwigluLimit::None,
        shexp: SwigluLimit::None,
    };
}

impl LayerFfn {
    /// Loads layer `layer`'s FFN: dense when `dense`, otherwise the routed
    /// plus shared-expert MoE. With `n_expert` the MoE's expert and router
    /// widths are checked against it — a router that scores a different
    /// number of experts than the tensors hold would cut the wrong rows.
    pub(crate) fn load(
        loaded: &LoadedModel,
        layer: usize,
        dense: bool,
        n_expert: Option<usize>,
    ) -> Result<Self> {
        let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };
        if dense {
            return Ok(Self::Dense {
                gate: get_matrix("ffn_gate.weight")?,
                up: get_matrix("ffn_up.weight")?,
                down: get_matrix("ffn_down.weight")?,
            });
        }
        let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .expert_matrix(&name)
                .with_context(|| format!("loading {name}"))
        };
        let moe = SharedExpertMoe {
            gate_inp: get_matrix("ffn_gate_inp.weight")?,
            exp_probs_b: loaded
                .tensor(&format!("blk.{layer}.exp_probs_b.bias"))
                .ok()
                .map(|(v, _)| v),
            gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
            up_exps: get_expert_matrix("ffn_up_exps.weight")?,
            down_exps: get_expert_matrix("ffn_down_exps.weight")?,
            gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
            up_shexp: get_matrix("ffn_up_shexp.weight")?,
            down_shexp: get_matrix("ffn_down_shexp.weight")?,
        };
        if let Some(n_expert) = n_expert {
            anyhow::ensure!(
                moe.gate_exps.n_expert == n_expert,
                "layer {layer} carries {} experts, not the declared expert_count ({n_expert})",
                moe.gate_exps.n_expert
            );
            anyhow::ensure!(
                moe.gate_inp.out_dim == n_expert,
                "layer {layer}'s router scores {} experts, not the declared expert_count \
                 ({n_expert})",
                moe.gate_inp.out_dim
            );
        }
        Ok(Self::Moe(Box::new(moe)))
    }

    pub(crate) fn is_dense(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }

    /// The layer's FFN over `normed` (`[n_tokens, n_embd]`) into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_into(
        &self,
        backend: &dyn Backend,
        routing: &ExpertRouting,
        out: &mut Vec<f32>,
        scratch: &mut FfnScratch,
        normed: &[f32],
        n_tokens: usize,
        n_embd: usize,
        limits: FfnLimits,
        il: usize,
    ) {
        match self {
            Self::Dense { gate, up, down } => super::swiglu_ffn_limited_into(
                backend,
                out,
                scratch,
                normed,
                n_tokens,
                gate,
                up,
                down,
                limits.dense,
            ),
            Self::Moe(moe) => {
                *out = super::swiglu_moe_ffn(
                    backend,
                    routing,
                    normed,
                    n_tokens,
                    n_embd,
                    &SwigluMoe {
                        gate_inp: &moe.gate_inp,
                        exp_probs_b: moe.exp_probs_b.as_deref(),
                        gate_exps: &moe.gate_exps,
                        up_exps: &moe.up_exps,
                        down_exps: &moe.down_exps,
                        shared: Some(SwigluSharedExpert {
                            gate: &moe.gate_shexp,
                            up: &moe.up_shexp,
                            down: &moe.down_shexp,
                        }),
                        clamp_exp: limits.exp,
                        clamp_shexp: limits.shexp,
                    },
                    il,
                )
            }
        }
    }
}

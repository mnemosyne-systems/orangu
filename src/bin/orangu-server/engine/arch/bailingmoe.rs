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

//! Ling 3.0 (`general.architecture = "bailingmoe3"`), e.g.
//! `bartowski/Ling-3.0-tiny-GGUF` and `bartowski/Ling-3.0-flash-GGUF`.
//!
//! Every part of this model is a part this engine already had, which is
//! why the module is short:
//!
//! * **The attention trunk** alternates three Kimi Delta Attention layers
//!   with one gated, absorbed multi-head latent attention layer — the pair
//!   in [`engine::arch::kda`](super::kda), shared with
//!   `engine::arch::kimi3`. Which layer is which is read from the file's
//!   per-layer `attention.head_count_kv` array, where `0` marks a
//!   recurrent (KDA) layer, exactly as upstream sets `is_recr`. Unlike
//!   `kimi-k3`'s, the latent layers here **do** rotate: NORM-paired RoPE
//!   over `rope.dimension_count` of each query head's tail and of the
//!   shared key half, as in `engine::arch::glm`.
//! * **The FFN** is a leading dense SwiGLU block (`leading_dense_block_
//!   count`), then routed + shared-expert SwiGLU MoE on every remaining
//!   layer — [`super::swiglu_moe_ffn`], shared with `engine::arch::glm`.
//! * **The routing** is DeepSeek-V3's: `sigmoid` probabilities, an
//!   `exp_probs_b` selection bias that steers the choice but never the
//!   weights, renormalization, a `expert_weights_scale`, and — new here,
//!   and the one piece of shared machinery this model needed built —
//!   group-limited selection ([`super::ExpertGroups`]), where the 128
//!   experts form 8 groups and only the best 4 groups may contribute.
//!
//! Transcribed from `src/models/bailingmoe3.cpp` in upstream `llama.cpp`
//! (read directly, not guessed), together with the grouped-selection arm
//! of `llm_graph_context::build_moe_ffn` and the delta-rule recurrence in
//! `src/models/delta-net-base.cpp` that the KDA half runs.
//!
//! Deliberately **not** implemented: the NextN/multi-token-prediction head
//! `Ling-3.0-flash` ships inside its `block_count`. It is a speculative
//! decoding accelerator, not part of the model's own output, and it is
//! trimmed here the same way every other architecture in this engine trims
//! one — the trunk stops before it and its tensors are never read.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::kda::{BAILINGMOE3_KDA_NAMES, KdaLayer, KdaShape, MlaLayer, MlaShape};
use super::{
    ExpertGating, ExpertGroups, ExpertRouting, ModelForward, SwigluMoe, SwigluSharedExpert,
};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor::{self, RopeLayout, RopeParams};

/// One MoE layer: routed experts plus the always-on shared one, both at
/// full width on the layer input.
struct Moe {
    gate_inp: QuantMatrix,
    exp_probs_b: Option<Vec<f32>>,
    gate_exps: ExpertQuantMatrix,
    up_exps: ExpertQuantMatrix,
    down_exps: ExpertQuantMatrix,
    gate_shexp: QuantMatrix,
    up_shexp: QuantMatrix,
    down_shexp: QuantMatrix,
    /// This layer's entries in `swiglu_clamp_exp` / `swiglu_clamp_shexp`,
    /// `0` when the file declares none.
    clamp_exp: f32,
    clamp_shexp: f32,
}

enum Ffn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    Moe(Box<Moe>),
}

enum Attn {
    Kda(Box<KdaLayer>),
    Mla(Box<MlaLayer>),
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn: Attn,
    ffn: Ffn,
}

pub struct BailingMoeModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    routing: ExpertRouting,
    kda: KdaShape,
    mla: MlaShape,
    kv_dims: Vec<usize>,
    recurrent_specs: Vec<RecurrentSpec>,
    layers: Vec<Layer>,
}

impl BailingMoeModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_head = loaded.config.n_head;
        // `block_count` counts the NextN/MTP block when a file ships one;
        // the trunk stops before it, and `config` keeps the file's own
        // count while `n_trunk_layer` reports what actually runs — the same
        // split `engine::arch::glm` and the Qwen 3.5 trunk use.
        let n_layer_nextn = loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize;
        anyhow::ensure!(
            n_layer_nextn < loaded.config.n_layer,
            "nextn_predict_layers ({n_layer_nextn}) must be fewer than block_count ({})",
            loaded.config.n_layer
        );
        let n_layer = loaded.config.n_layer - n_layer_nextn;

        let kda_head_dim = loaded
            .metadata_u64("kda.head_dim")
            .context("missing kda.head_dim")? as usize;
        let d_conv = loaded
            .metadata_u64("ssm.conv_kernel")
            .context("missing ssm.conv_kernel")? as usize;
        anyhow::ensure!(d_conv > 0, "ssm.conv_kernel must be at least 1");
        // Upstream asserts both of these rather than branching on them:
        // the released checkpoints are safe-gated, and the converter
        // refuses a model that is not.
        anyhow::ensure!(
            loaded.metadata_u64("kda.safe_gate").unwrap_or(1) != 0,
            "kda.safe_gate is false: this architecture's unbounded delta-net gate \
             has no released checkpoint to verify against"
        );
        let gate_lower_bound = loaded
            .metadata_f32("kda.gate_lower_bound")
            .context("missing kda.gate_lower_bound")?;
        anyhow::ensure!(
            gate_lower_bound < 0.0,
            "kda.gate_lower_bound must be negative (got {gate_lower_bound})"
        );
        let kda = KdaShape {
            n_head,
            head_dim: kda_head_dim,
            d_inner: kda_head_dim * n_head,
            d_conv,
            gate_lower_bound: Some(gate_lower_bound),
            eps: loaded.config.rms_eps,
        };

        let kv_lora_rank = loaded
            .metadata_u64("attention.kv_lora_rank")
            .context("missing attention.kv_lora_rank")? as usize;
        let head_k_mla = loaded
            .metadata_u64("attention.key_length_mla")
            .context("missing attention.key_length_mla")? as usize;
        let head_v_mla = loaded
            .metadata_u64("attention.value_length_mla")
            .context("missing attention.value_length_mla")? as usize;
        let rope_dim = loaded.config.rope_dim;
        anyhow::ensure!(
            head_k_mla > rope_dim,
            "attention.key_length_mla ({head_k_mla}) must exceed rope.dimension_count ({rope_dim})"
        );
        let mla = MlaShape {
            n_head,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            kv_row: kv_lora_rank + rope_dim,
            // NORM pairing, not NEOX: upstream's `llama_model_rope_type`
            // puts `bailingmoe3` in its `LLAMA_ROPE_TYPE_NORM` arm, next
            // to `glm-dsa`. The two conventions rotate different pairs of
            // numbers by the same angles, so the wrong one is silently
            // wrong on long prompts rather than an error.
            rope: Some(RopeParams {
                rope_dim,
                freq_base: loaded.config.rope_freq_base,
                layout: RopeLayout::Neox,
                ..RopeParams::default()
            }),
            rope_dim,
            // The *query head's* width, not the wider absorbed key's.
            kq_scale: 1.0 / (head_k_mla as f32).sqrt(),
            eps: loaded.config.rms_eps,
        };

        // `attention.head_count_kv` is a per-layer array here: 0 marks a
        // KDA (recurrent) layer, non-zero a latent-attention one.
        let head_count_kv = loaded
            .metadata_array_u64("attention.head_count_kv")
            .context("missing the per-layer attention.head_count_kv array")?;
        anyhow::ensure!(
            head_count_kv.len() >= n_layer,
            "attention.head_count_kv has {} entries, fewer than the {n_layer} trunk layers",
            head_count_kv.len()
        );

        let n_expert = loaded
            .metadata_u64("expert_count")
            .context("missing expert_count")? as usize;
        let routing = ExpertRouting {
            n_expert_used: loaded
                .metadata_u64("expert_used_count")
                .context("missing expert_used_count")? as usize,
            gating: ExpertGating::from_gguf(
                loaded
                    .metadata_u64("expert_gating_func")
                    .context("missing expert_gating_func")?,
            )?,
            weights_norm: loaded
                .metadata_u64("expert_weights_norm")
                .is_some_and(|v| v != 0),
            weights_scale: loaded.metadata_f32("expert_weights_scale").unwrap_or(1.0),
            groups: ExpertGroups::from_gguf(loaded, n_expert)?,
        };
        let n_layer_dense_lead = loaded
            .metadata_u64("leading_dense_block_count")
            .unwrap_or(0) as usize;
        let clamp_exp = loaded
            .metadata_array_f32("swiglu_clamp_exp")
            .unwrap_or_default();
        let clamp_shexp = loaded
            .metadata_array_f32("swiglu_clamp_shexp")
            .unwrap_or_default();

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // A file may tie the output projection to the embedding table
        // rather than carry its own, which upstream handles by pointing
        // both at `token_embd`.
        let output_weight = match loaded.matrix("output.weight") {
            Ok(matrix) => matrix,
            Err(_) => loaded
                .matrix("token_embd.weight")
                .context("loading output.weight, and token_embd.weight as its tied fallback")?,
        };

        let mut kv_dims = Vec::new();
        let mut recurrent_specs = Vec::new();
        let mut layers = Vec::with_capacity(n_layer);
        for (i, &kv_heads) in head_count_kv.iter().take(n_layer).enumerate() {
            let get = |suffix: &str| -> Result<Vec<f32>> {
                let name = format!("blk.{i}.{suffix}");
                Ok(loaded
                    .tensor(&name)
                    .with_context(|| format!("loading {name}"))?
                    .0)
            };
            let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
                let name = format!("blk.{i}.{suffix}");
                loaded
                    .expert_matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };

            let attn = if kv_heads == 0 {
                recurrent_specs.push(kda.recurrent_spec());
                Attn::Kda(Box::new(KdaLayer::load(
                    loaded,
                    i,
                    &kda,
                    &BAILINGMOE3_KDA_NAMES,
                    recurrent_specs.len() - 1,
                )?))
            } else {
                kv_dims.push(mla.kv_row);
                Attn::Mla(Box::new(MlaLayer::load(
                    loaded,
                    i,
                    &mla,
                    kv_dims.len() - 1,
                )?))
            };

            let ffn = if i < n_layer_dense_lead {
                Ffn::Dense {
                    gate: get_matrix("ffn_gate.weight")?,
                    up: get_matrix("ffn_up.weight")?,
                    down: get_matrix("ffn_down.weight")?,
                }
            } else {
                let moe = Moe {
                    gate_inp: get_matrix("ffn_gate_inp.weight")?,
                    exp_probs_b: get("exp_probs_b.bias").ok(),
                    gate_exps: get_expert_matrix("ffn_gate_exps.weight")?,
                    up_exps: get_expert_matrix("ffn_up_exps.weight")?,
                    down_exps: get_expert_matrix("ffn_down_exps.weight")?,
                    gate_shexp: get_matrix("ffn_gate_shexp.weight")?,
                    up_shexp: get_matrix("ffn_up_shexp.weight")?,
                    down_shexp: get_matrix("ffn_down_shexp.weight")?,
                    clamp_exp: clamp_exp.get(i).copied().unwrap_or(0.0),
                    clamp_shexp: clamp_shexp.get(i).copied().unwrap_or(0.0),
                };
                anyhow::ensure!(
                    moe.gate_exps.n_expert == n_expert,
                    "layer {i} carries {} experts, not the declared expert_count ({n_expert})",
                    moe.gate_exps.n_expert
                );
                anyhow::ensure!(
                    moe.gate_inp.out_dim == n_expert,
                    "layer {i}'s router scores {} experts, not the declared expert_count \
                     ({n_expert}) — grouped selection would then cut the wrong rows",
                    moe.gate_inp.out_dim
                );
                Ffn::Moe(Box::new(moe))
            };

            layers.push(Layer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                attn,
                ffn,
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            routing,
            kda,
            mla,
            kv_dims,
            recurrent_specs,
            layers,
        })
    }

    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        // Grown once and reused across layers rather than allocated per
        // norm: at prefill widths this is megabytes a layer. See
        // `tensor::rmsnorm_into`.
        let mut cur: Vec<f32> = Vec::new();
        // Projection outputs, on the same principle — see
        // `Backend::matmul_into`.
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut ffn_scratch = super::FfnScratch::default();
        // The attention half's projections, on the same principle.
        let mut attn_out: Vec<f32> = Vec::new();
        let mut kda_scratch = super::kda::KdaScratch::default();
        let mut mla_scratch = super::kda::MlaScratch::default();

        for layer in &self.layers {
            tensor::rmsnorm_into(
                &mut cur,
                &x,
                &layer.attn_norm,
                n_tokens,
                n_embd,
                self.config.rms_eps,
            );
            match &layer.attn {
                Attn::Kda(kda) => kda.forward_into(
                    self.backend.as_ref(),
                    &mut attn_out,
                    &mut kda_scratch,
                    &self.kda,
                    cache,
                    &cur,
                    n_tokens,
                ),
                Attn::Mla(mla) => mla.forward_into(
                    self.backend.as_ref(),
                    &mut attn_out,
                    &mut mla_scratch,
                    &self.mla,
                    cache,
                    &cur,
                    n_tokens,
                    start_pos,
                ),
            }
            tensor::add_inplace(&mut x, &attn_out);

            tensor::rmsnorm_into(
                &mut cur,
                &x,
                &layer.ffn_norm,
                n_tokens,
                n_embd,
                self.config.rms_eps,
            );
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => super::swiglu_ffn_into(
                    self.backend.as_ref(),
                    &mut ffn_out,
                    &mut ffn_scratch,
                    &cur,
                    n_tokens,
                    gate,
                    up,
                    down,
                ),
                Ffn::Moe(moe) => {
                    ffn_out = super::swiglu_moe_ffn(
                        self.backend.as_ref(),
                        &self.routing,
                        &cur,
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
                            clamp_exp: moe.clamp_exp,
                            clamp_shexp: moe.clamp_shexp,
                        },
                    )
                }
            }
            tensor::add_inplace(&mut x, &ffn_out);
        }

        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            n_tokens,
            n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }
}

impl ModelForward for BailingMoeModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn n_trunk_layer(&self) -> usize {
        self.layers.len()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_mixed(capacity, &self.kv_dims, &self.recurrent_specs)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        let last = &hidden[(tokens.len() - 1) * n_embd..];
        Ok(self.backend.matmul(last, 1, &self.output_weight))
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        self.run_layers(&mut cache, tokens, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ExpertGating, ExpertGroups, ExpertRouting};

    /// Group-limited selection must be able to *reject a better expert*
    /// because its group is weak — that is the whole point of it, and a
    /// router that quietly ignored the grouping would still pick sensible
    /// experts and still produce fluent text.
    ///
    /// Four groups of two, two groups used. Expert 1 has the second-highest
    /// score in the layer, but its group's two-member sum (`9.0 + 1.0`) is
    /// beaten by two others, so it must not be selected even though a plain
    /// top-k would take it.
    #[test]
    fn a_strong_expert_in_a_losing_group_is_not_selected() {
        let ungrouped = ExpertRouting {
            n_expert_used: 2,
            gating: ExpertGating::Sigmoid,
            weights_norm: false,
            weights_scale: 1.0,
            groups: None,
        };
        // Groups: [0,1] [2,3] [4,5] [6,7].
        let logits = vec![-9.0, 6.0, 5.0, 5.0, 4.5, 4.5, -9.0, -9.0];
        let (plain, _) = ungrouped.route(&logits, None, None);
        assert_eq!(plain, vec![1, 2], "a plain top-k takes expert 1");

        let grouped = ExpertRouting {
            groups: Some(ExpertGroups {
                count: 4,
                used: 2,
                size: 2,
            }),
            ..ungrouped
        };
        let (selected, _) = grouped.route(&logits, None, None);
        assert!(
            !selected.contains(&1),
            "expert 1's group scored lowest of the survivors, but it was selected: {selected:?}"
        );
        assert_eq!(
            selected,
            vec![2, 3],
            "the two members of the strongest group should win"
        );
    }

    /// The group score is the sum of a group's **two** best members, so a
    /// group with one outstanding expert and one poor one loses to a group
    /// of two good ones. Scoring by the best member alone — the obvious
    /// substitute — reverses this.
    #[test]
    fn a_group_is_scored_by_its_two_best_members_not_its_best() {
        let routing = ExpertRouting {
            n_expert_used: 1,
            gating: ExpertGating::Sigmoid,
            weights_norm: false,
            weights_scale: 1.0,
            groups: Some(ExpertGroups {
                count: 2,
                used: 1,
                size: 2,
            }),
        };
        // sigmoid is monotone, so group 0 sums to sigmoid(8)+sigmoid(-8)
        // ~= 1.0 and group 1 to sigmoid(2)+sigmoid(2) ~= 1.76.
        let logits = vec![8.0, -8.0, 2.0, 2.0];
        let (selected, _) = routing.route(&logits, None, None);
        assert_eq!(
            selected,
            vec![2],
            "the group of two good experts should beat the group with one great one"
        );
    }

    /// The selection bias reaches the *grouping* too — it is added before
    /// the groups are scored, which is what upstream's `selection_probs`
    /// carries into its per-group top-2.
    #[test]
    fn the_selection_bias_moves_which_group_survives() {
        let routing = ExpertRouting {
            n_expert_used: 1,
            gating: ExpertGating::Sigmoid,
            weights_norm: false,
            weights_scale: 1.0,
            groups: Some(ExpertGroups {
                count: 2,
                used: 1,
                size: 2,
            }),
        };
        let logits = vec![2.0, 2.0, 1.0, 1.0];
        let (unbiased, _) = routing.route(&logits, None, None);
        assert_eq!(unbiased, vec![0]);

        let (biased, weights) = routing.route(&logits, Some(&[0.0, 0.0, 5.0, 5.0]), None);
        assert_eq!(biased, vec![2], "the biased group should have survived");
        // ...and the weight is still the *unbiased* probability.
        assert!(
            (weights[0] - crate::engine::tensor::sigmoid(1.0)).abs() < 1e-6,
            "{weights:?}"
        );
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// The reference token ids and expected continuation for one prompt,
    /// taken from real `llama.cpp` (`llama-server` built from `master` at
    /// `9731ad3f2`, which is the first tree that can load this
    /// architecture at all) running `bartowski/Ling-3.0-tiny-GGUF:Q4_K_M`
    /// with `--device none`: `/tokenize` for the ids, then `/completion`
    /// with `n_probs` at `temperature 0` for the ranking.
    struct Reference {
        tokens: &'static [u32],
        /// The id real llama.cpp ranks first.
        top: u32,
        /// How far ahead of the runner-up it is, in nats. Both cases here
        /// are deliberately decisive: a near-tie would make this test fail
        /// on rounding rather than on a wrong graph.
        margin: f32,
    }

    /// "The quick brown fox jumps over the lazy" → " dog" (7339) at
    /// logprob -0.0608, ahead of " brown" (13187) at -4.0162.
    const SHORT: Reference = Reference {
        tokens: &[678, 3901, 13187, 46998, 40977, 997, 268, 27028],
        top: 7339,
        margin: 3.9,
    };

    /// An 89-token prompt that lists ten items and then replays nine of
    /// them, so the tenth is all but forced: " ju" (11959, the first piece
    /// of "jujube") at logprob -0.0001, ahead of " and" (301) at -10.4121.
    ///
    /// Length is the point. The short case exercises eight steps of the
    /// delta-net recurrence; this one exercises eighty-nine, across six
    /// latent-attention layers whose cache has to line up with them, and
    /// it can only be answered by reading something said sixty tokens
    /// earlier. A decay applied in the wrong place, a conv history off by
    /// one, or a key row read at the wrong offset all survive the short
    /// case and fail here.
    const LONG: Reference = Reference {
        tokens: &[
            56346, 268, 2538, 1746, 6604, 13, 18309, 810, 341, 22745, 13, 18309, 1307, 341, 45976,
            13, 18309, 2274, 341, 39030, 13, 18309, 3173, 341, 4548, 2072, 13, 18309, 4428, 341,
            20380, 14562, 13, 18309, 5039, 341, 8540, 13, 18309, 8897, 341, 42036, 13, 18309, 8793,
            341, 23534, 67, 443, 13, 18309, 13097, 341, 19340, 23425, 13, 18309, 6588, 341, 11959,
            73, 5278, 13, 4948, 41503, 25, 22745, 11, 45976, 11, 39030, 11, 4548, 2072, 11, 20380,
            14562, 11, 8540, 11, 42036, 11, 23534, 67, 443, 11, 19340, 23425, 11,
        ],
        top: 11959,
        margin: 10.0,
    };

    fn check(model: &BailingMoeModel, reference: &Reference) {
        let mut cache = model.new_kv_cache(reference.tokens.len() + 1);
        let logits = model
            .forward(&mut cache, reference.tokens, 0, 0)
            .expect("forward");
        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(
            ranked[0].0 as u32, reference.top,
            "top prediction was {} ({:.4}), not the reference {} — runner-up {} ({:.4})",
            ranked[0].0, ranked[0].1, reference.top, ranked[1].0, ranked[1].1
        );
        // The reference's margin is in log space and these are raw logits,
        // so this is not the same number — but a graph that merely
        // *happens* to rank the right token first would not keep this much
        // daylight behind it.
        assert!(
            ranked[0].1 - ranked[1].1 > reference.margin * 0.5,
            "the top token led by only {:.4} logits, far less than the reference's {:.1} nats — \
             the ranking is probably accidental",
            ranked[0].1 - ranked[1].1,
            reference.margin
        );
    }

    /// The whole graph at once — the delta-net recurrence, the rotated
    /// latent attention, the leading dense block, and group-limited expert
    /// selection, all at model-shaped dimensions, against real
    /// `llama.cpp`'s own ranking of the same token ids.
    ///
    /// Run with `ORANGU_TEST_BAILINGMOE_MODEL=/path/to/
    /// Ling-3.0-tiny-Q4_K_M.gguf cargo test --release --bin orangu-server
    /// bailingmoe::real_model_tests -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bailingmoe3_ranks_the_same_tokens_as_real_llama_cpp() {
        let path = std::env::var("ORANGU_TEST_BAILINGMOE_MODEL")
            .expect("set ORANGU_TEST_BAILINGMOE_MODEL to a Ling 3.0 GGUF");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "bailingmoe3");
        let model = BailingMoeModel::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");
        // The layer split is what the rest of the graph hangs off, and it
        // is read rather than assumed — a file whose array said something
        // else would exercise none of this.
        let kda = model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, Attn::Kda(_)))
            .count();
        assert_eq!(
            (kda, model.layers.len() - kda),
            (18, 6),
            "this file should be 18 recurrent layers to 6 latent-attention ones"
        );

        check(&model, &SHORT);
        check(&model, &LONG);
    }
}

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

//! GLM-5.3-Flash (`general.architecture = "glm5next"`), e.g.
//! `unsloth/GLM-5.3-Flash-GGUF`.
//!
//! Four things this engine already had, strung together in a way it had
//! not:
//!
//! * **The trunk alternates** three Kimi Delta Attention layers with one
//!   absorbed multi-head latent attention layer — the pair in
//!   [`engine::arch::kda`](super::kda), shared with `engine::arch::kimi3`
//!   and `engine::arch::bailingmoe`. Which layer is which is read from the
//!   per-layer `attention.head_count_kv` array, where `0` marks a recurrent
//!   one. Two small things are new and declared, not guessed: the KDA
//!   output gate is factored through a `kda.head_dim` bottleneck the way
//!   the decay gate already was ([`kda::GLM5NEXT_KDA_NAMES`]), and the
//!   latent attention rotates **nothing at all** — `rope.dimension_count`
//!   is `0`, so the query is entirely "nope" and the compressed key/value
//!   is the whole cache row.
//! * **There is no residual vector.** The state between sub-layers is
//!   `hyper_connection.count` parallel streams, mixed down to one vector on
//!   the way into each sub-layer and scattered back on the way out —
//!   [`engine::arch::hyper`](super::hyper), the same mHC formulation as
//!   `engine::arch::deepseek4`. The one difference is the end: DeepSeek-V4
//!   collapses the bundle with a mixer of its own, and this model takes the
//!   plain mean.
//! * **The latent layers attend sparsely**, choosing positions with a
//!   lightning indexer — [`engine::arch::indexer`](super::indexer), shared
//!   with `engine::arch::glm`. Here it is the *pooled* form: positions are
//!   grouped into fixed `attention.indexer.kpool` pools, one pooled key
//!   stands for each, and the cut is over whole pools with the query's own
//!   trailing partial pool always attended.
//! * **The FFN** is a leading dense SwiGLU block
//!   (`leading_dense_block_count`), then routed + shared-expert SwiGLU MoE
//!   on every remaining layer — [`super::swiglu_moe_ffn`]. Its
//!   `swiglu_clamp_exp`/`_shexp` limit is upstream's *pre-activation*
//!   clamp ([`super::SwigluLimit::PreActivation`]), the same branch
//!   DeepSeek-V4 takes, and it applies to the dense blocks too.
//!
//! Transcribed from the `glm5next` graph proposed upstream, cross-read
//! against the two independent implementations of it, which agree on every
//! formula used here.
//!
//! Deliberately **not** implemented: the NextN/multi-token-prediction block
//! this file ships inside its `block_count`, trimmed the way every other
//! architecture here trims one — the trunk stops before it and its tensors
//! are never read. Nor the vision tower shipped beside the model as a
//! separate file, which is out of scope for this engine as a whole.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::hyper::{HyperConnection, HyperScratch, HyperShape};
use super::indexer::{Indexer, IndexerShape};
use super::kda::{GLM5NEXT_KDA_NAMES, KdaLayer, KdaShape, MlaLayer, MlaShape};
use super::{
    ExpertGating, ExpertGroups, ExpertRouting, ModelForward, SwigluLimit, SwigluMoe,
    SwigluSharedExpert,
};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// One MoE layer: routed experts plus the always-on shared one.
struct Moe {
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

/// A layer's FFN: dense for the leading blocks, routed MoE for the rest.
enum Ffn {
    Dense {
        gate: QuantMatrix,
        up: QuantMatrix,
        down: QuantMatrix,
    },
    /// Boxed: a `Moe` is several times the size of the dense arm, and every
    /// layer carries one `Ffn`.
    Moe(Box<Moe>),
}

/// One latent-attention layer: the absorbed MLA block plus the lightning
/// indexer that decides what it may attend. The indexer is `Option` because
/// a file converted without indexer weights still describes a runnable
/// model — a dense one.
struct Sparse {
    mla: Box<MlaLayer>,
    indexer: Option<Indexer>,
}

enum Attn {
    Kda(Box<KdaLayer>),
    Mla(Box<Sparse>),
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: HyperConnection,
    hc_ffn: HyperConnection,
    attn: Attn,
    ffn: Ffn,
    /// This layer's `swiglu_clamp_exp` / `swiglu_clamp_shexp` entries.
    clamp_exp: SwigluLimit,
    clamp_shexp: SwigluLimit,
}

pub struct Glm5Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    hyper: HyperShape,
    kda: KdaShape,
    mla: MlaShape,
    indexer: IndexerShape,
    routing: ExpertRouting,
    kv_dims: Vec<(usize, usize)>,
    recurrent_specs: Vec<RecurrentSpec>,
    layers: Vec<Layer>,
}

/// The buffers every layer of a forward pass reuses, so the trunk allocates
/// once per pass rather than several times per layer.
#[derive(Default)]
struct Scratch {
    hc: HyperScratch,
    kda: super::kda::KdaScratch,
    mla: super::kda::MlaScratch,
    ffn: super::FfnScratch,
    attn_out: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl Glm5Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let n_head = loaded.config.n_head;
        let rope_dim = loaded.config.rope_dim;
        // The graph is position-free: no rotation anywhere, in either the
        // latent attention or the indexer. Position enters only through the
        // causal mask, the recurrent layers' order of arrival, and the
        // indexer's intra-pool bias. A file that declared rotary dimensions
        // would be a different model, and running it as this one would be
        // wrong rather than a load failure.
        anyhow::ensure!(
            rope_dim == 0,
            "glm5next is position-free: rope.dimension_count must be 0, not {rope_dim}"
        );

        // `block_count` counts the NextN/MTP block; the trunk stops before
        // it. Running one as a trunk layer would be silently wrong rather
        // than a load failure — its tensors are all present and correctly
        // shaped.
        let n_layer_nextn = loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize;
        anyhow::ensure!(
            n_layer_nextn < loaded.config.n_layer,
            "nextn_predict_layers ({n_layer_nextn}) must be fewer than block_count ({})",
            loaded.config.n_layer
        );
        let n_layer = loaded.config.n_layer - n_layer_nextn;

        let hyper = HyperShape::from_gguf(loaded)?;

        let kda_head_dim = loaded
            .metadata_u64("kda.head_dim")
            .context("missing kda.head_dim")? as usize;
        let d_conv = loaded
            .metadata_u64("ssm.conv_kernel")
            .context("missing ssm.conv_kernel")? as usize;
        anyhow::ensure!(d_conv > 0, "ssm.conv_kernel must be at least 1");
        let gate_lower_bound = loaded.metadata_f32("kda.gate_lower_bound").unwrap_or(-5.0);
        anyhow::ensure!(
            gate_lower_bound < 0.0,
            "kda.gate_lower_bound must be negative (got {gate_lower_bound}): the unbounded \
             softplus gate has no released checkpoint to verify against"
        );
        let kda = KdaShape {
            n_head,
            head_dim: kda_head_dim,
            d_inner: kda_head_dim * n_head,
            d_conv,
            gate_lower_bound: Some(gate_lower_bound),
            eps: loaded.config.rms_eps,
            // The reference hard-codes this one rather than reusing the
            // model's RMS epsilon, which is `1e-5` here.
            l2_eps: 1e-6,
        };

        let kv_lora_rank = loaded
            .metadata_u64("attention.kv_lora_rank")
            .context("missing attention.kv_lora_rank")? as usize;
        let head_k_mla = loaded
            .metadata_u64("attention.key_length_mla")
            .context("missing attention.key_length_mla (this build requires MLA)")?
            as usize;
        let head_v_mla = loaded
            .metadata_u64("attention.value_length_mla")
            .context("missing attention.value_length_mla (this build requires MLA)")?
            as usize;
        let mla = MlaShape {
            n_head,
            kv_lora_rank,
            head_k_mla,
            head_v_mla,
            // Nothing rotates, so a cache row is the compressed key/value
            // and nothing else.
            kv_row: kv_lora_rank,
            rope: None,
            rope_dim: 0,
            // The *query head's* width, not the wider compressed row's.
            kq_scale: 1.0 / (head_k_mla as f32).sqrt(),
            eps: loaded.config.rms_eps,
        };

        let indexer = IndexerShape::from_gguf(loaded, None)?;
        let kpool = loaded
            .metadata_u64("attention.indexer.kpool")
            .context("missing attention.indexer.kpool")? as usize;
        anyhow::ensure!(kpool > 0, "attention.indexer.kpool must be at least 1");
        anyhow::ensure!(
            indexer.top_k.is_multiple_of(kpool),
            "attention.indexer.top_k ({}) must be a multiple of attention.indexer.kpool \
             ({kpool}): the cut is spent in whole pools",
            indexer.top_k
        );

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
            .unwrap_or_else(|| clamp_exp.clone());

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        // A file may tie the output projection to the embedding table
        // rather than carry its own.
        let output_weight = match loaded.matrix("output.weight") {
            Ok(matrix) => matrix,
            Err(_) => tok_embeddings.clone(),
        };

        let mut kv_dims: Vec<(usize, usize)> = Vec::new();
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
                    &GLM5NEXT_KDA_NAMES,
                    recurrent_specs.len() - 1,
                )?))
            } else {
                kv_dims.push((mla.kv_row, 1));
                let mla_layer = MlaLayer::load(loaded, i, &mla, kv_dims.len() - 1)?;
                // Two slots per indexer: one row per token, carrying the
                // key and the pooling gate that cannot be recomputed from
                // it, and one row per completed pool.
                let layer_indexer = if loaded
                    .matrix(&format!("blk.{i}.indexer.attn_k.weight"))
                    .is_ok()
                {
                    kv_dims.push((indexer.head_size, 1));
                    let state_slot = kv_dims.len() - 1;
                    kv_dims.push((indexer.head_size, kpool));
                    let pool_slot = kv_dims.len() - 1;
                    Some(Indexer::load(
                        loaded,
                        i,
                        Some(kpool),
                        &indexer,
                        state_slot,
                        pool_slot,
                    )?)
                } else {
                    None
                };
                Attn::Mla(Box::new(Sparse {
                    mla: Box::new(mla_layer),
                    indexer: layer_indexer,
                }))
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
                };
                anyhow::ensure!(
                    moe.gate_exps.n_expert == n_expert,
                    "layer {i} carries {} experts, not the declared expert_count ({n_expert})",
                    moe.gate_exps.n_expert
                );
                anyhow::ensure!(
                    moe.gate_inp.out_dim == n_expert,
                    "layer {i}'s router scores {} experts, not the declared expert_count \
                     ({n_expert})",
                    moe.gate_inp.out_dim
                );
                Ffn::Moe(Box::new(moe))
            };

            layers.push(Layer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                hc_attn: HyperConnection::load(loaded, &format!("blk.{i}.hc_attn"))?,
                hc_ffn: HyperConnection::load(loaded, &format!("blk.{i}.hc_ffn"))?,
                attn,
                ffn,
                // The limit applies to the dense blocks as well as to the
                // experts, which is why it is a layer field and not a
                // property of the MoE arm.
                clamp_exp: SwigluLimit::pre_activation(clamp_exp.get(i).copied().unwrap_or(0.0)),
                clamp_shexp: SwigluLimit::pre_activation(
                    clamp_shexp.get(i).copied().unwrap_or(0.0),
                ),
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            hyper,
            kda,
            mla,
            indexer,
            routing,
            kv_dims,
            recurrent_specs,
            layers,
        })
    }

    /// Runs `tokens` through every trunk layer, collapses the stream bundle
    /// and returns each token's final normed hidden state (`[n_tokens,
    /// n_embd]`) — what the output projection consumes.
    fn run_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        let n_embd = self.config.n_embd;

        let mut embeddings = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(
                tok < self.config.n_vocab,
                "token id {tok} is out of vocab range"
            );
            embeddings[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }
        let mut x = self.hyper.seed(&embeddings, n_tokens);

        let mut scratch = Scratch::default();
        for layer in &self.layers {
            self.forward_layer(layer, cache, &mut x, n_tokens, start_pos, &mut scratch)?;
        }

        let mut out = Vec::new();
        self.hyper.mean_into(&mut out, &x, n_tokens);
        tensor::rmsnorm_inplace(
            &mut out,
            &self.output_norm,
            n_tokens,
            n_embd,
            self.config.rms_eps,
        );
        Ok(out)
    }

    fn forward_layer(
        &self,
        layer: &Layer,
        cache: &mut KvCache,
        x: &mut [f32],
        n_tokens: usize,
        start_pos: usize,
        scratch: &mut Scratch,
    ) -> Result<()> {
        let backend = self.backend.as_ref();
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let Scratch {
            hc:
                HyperScratch {
                    residual,
                    post,
                    comb,
                    flat,
                    mixes,
                    collapsed_attn,
                    collapsed_ffn,
                },
            kda: kda_scratch,
            mla: mla_scratch,
            ffn: ffn_scratch,
            attn_out,
            ffn_out,
        } = scratch;

        // A genuine copy, unlike the norm inputs: the out-mix writes `x`
        // while still reading the streams as they were before this
        // sub-layer. Only the *allocation* is hoisted here, not the copy.
        residual.clear();
        residual.extend_from_slice(x);
        self.hyper.collapse_into(
            backend,
            collapsed_attn,
            &layer.hc_attn,
            x,
            n_tokens,
            Some((post, comb)),
            flat,
            mixes,
        );
        let cur = collapsed_attn;
        tensor::rmsnorm_inplace(cur, &layer.attn_norm, n_tokens, n_embd, eps);
        match &layer.attn {
            Attn::Kda(kda) => kda.forward_into(
                backend,
                attn_out,
                kda_scratch,
                &self.kda,
                cache,
                cur,
                n_tokens,
            ),
            Attn::Mla(sparse) => {
                // The query first, because the indexer's own query is
                // projected off its LoRA intermediate — then the cut, then
                // the attention over what survived it.
                sparse
                    .mla
                    .project_query_into(backend, mla_scratch, &self.mla, cur, n_tokens);
                let selection = sparse.indexer.as_ref().map(|ix| {
                    let inputs = ix.inputs(
                        backend,
                        &self.indexer,
                        mla_scratch.qr(),
                        cur,
                        n_tokens,
                        start_pos,
                    );
                    (0..n_tokens)
                        .map(|t| ix.select(&self.indexer, cache, &inputs, t, start_pos + t))
                        .collect::<Vec<Vec<usize>>>()
                });
                sparse.mla.attend_into(
                    backend,
                    attn_out,
                    mla_scratch,
                    &self.mla,
                    cache,
                    cur,
                    n_tokens,
                    start_pos,
                    selection.as_deref(),
                );
            }
        }
        self.hyper
            .expand(x, attn_out, residual, post, comb, n_tokens);

        residual.clear();
        residual.extend_from_slice(x);
        self.hyper.collapse_into(
            backend,
            collapsed_ffn,
            &layer.hc_ffn,
            x,
            n_tokens,
            Some((post, comb)),
            flat,
            mixes,
        );
        let cur = collapsed_ffn;
        tensor::rmsnorm_inplace(cur, &layer.ffn_norm, n_tokens, n_embd, eps);
        match &layer.ffn {
            Ffn::Dense { gate, up, down } => super::swiglu_ffn_limited_into(
                backend,
                ffn_out,
                ffn_scratch,
                cur,
                n_tokens,
                gate,
                up,
                down,
                layer.clamp_exp,
            ),
            Ffn::Moe(moe) => {
                *ffn_out = super::swiglu_moe_ffn(
                    backend,
                    &self.routing,
                    cur,
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
                        clamp_exp: layer.clamp_exp,
                        clamp_shexp: layer.clamp_shexp,
                    },
                )
            }
        }
        self.hyper
            .expand(x, ffn_out, residual, post, comb, n_tokens);
        Ok(())
    }
}

impl ModelForward for Glm5Model {
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
        KvCache::new_mixed_strided(capacity, &self.kv_dims, &self.recurrent_specs)
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

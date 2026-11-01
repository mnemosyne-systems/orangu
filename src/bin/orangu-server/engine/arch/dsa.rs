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

//! The DeepSeek-sparse-attention (DSA) decoder: a plain pre-norm trunk —
//! attention, residual, FFN, residual — whose attention is absorbed
//! multi-head latent attention restricted to the positions a lightning
//! indexer picks, and whose FFN is a leading dense block then routed +
//! shared-expert MoE.
//!
//! Not tied to one architecture. `engine::arch::glm` (`glm-dsa`) and
//! `engine::arch::mistral_large4` (`mistral-large4`) each read their own
//! file conventions into a [`DsaModel`] and hand it the layers; everything
//! that runs is here, built from the pieces the other MLA architectures
//! share:
//!
//! * [`MlaLayer`](super::kda::MlaLayer) — the absorbed attention, with or
//!   without a query LoRA, with or without its own key/value projection.
//! * [`Indexer`](super::indexer::Indexer) — the per-token lightning
//!   indexer. A layer without one reuses the selection the last scoring
//!   layer made, carried down the stack per token.
//! * [`LayerFfn`](super::ffn::LayerFfn) — the dense-or-MoE FFN.
//!
//! A model may need **more than one attention shape**: `mistral-large4`
//! rotates three layers in four and leaves the fourth position-free, so its
//! rotary and NoPE layers differ in query split, cache-row width and
//! indexer rotation. Each layer names its shape by index into
//! [`DsaModel::shapes`].

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ffn::{FfnLimits, LayerFfn};
use super::indexer::{Indexer, IndexerShape};
use super::kda::{MlaLayer, MlaScratch, MlaShape};
use super::{ExpertRouting, ModelForward, Seqs};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// One attention shape: the latent attention's and its indexer's.
pub(crate) struct DsaShape {
    pub mla: MlaShape,
    pub indexer: IndexerShape,
}

/// How one layer is put together — what an architecture's loader decides
/// from its file, and [`DsaModel::load_layers`] then builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerPlan {
    /// Index into the model's [`DsaShape`]s.
    pub shape: usize,
    /// The layer whose compressed rows this one attends: itself for a layer
    /// that writes its own, an earlier one with the same shape otherwise.
    pub kv_source: usize,
    /// Whether this layer runs its own lightning indexer, or reuses the
    /// selection the last one that did made.
    pub scores: bool,
    /// A leading dense FFN block rather than MoE.
    pub dense: bool,
}

/// Checks a whole stack of [`LayerPlan`]s before any weight is read: the
/// first layer must score (every later one either scores or reuses), and a
/// layer may only read rows that an *earlier* layer of the *same* shape
/// writes itself.
pub(crate) fn validate_plans(plans: &[LayerPlan], n_shapes: usize) -> Result<()> {
    if let Some(first) = plans.first() {
        anyhow::ensure!(
            first.scores,
            "the first layer must run its own lightning indexer"
        );
    }
    for (il, plan) in plans.iter().enumerate() {
        anyhow::ensure!(
            plan.shape < n_shapes,
            "layer {il} names attention shape {} of {n_shapes}",
            plan.shape
        );
        let src = plan.kv_source;
        anyhow::ensure!(
            src <= il && plans[src].kv_source == src && plans[src].shape == plan.shape,
            "invalid KV source layer {src} for layer {il}"
        );
    }
    Ok(())
}

struct DsaLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    shape: usize,
    mla: MlaLayer,
    /// `Some` on a layer that scores positions itself.
    indexer: Option<Indexer>,
    ffn: LayerFfn,
}

/// The buffers every layer of a forward pass reuses.
#[derive(Default)]
struct LayerScratch {
    normed: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_out: Vec<f32>,
    ffn: super::FfnScratch,
    mla: MlaScratch,
    /// Per-token positions the current layer may attend, carried down to
    /// the layers that reuse it.
    selection: Vec<Vec<usize>>,
    /// Whether [`Self::selection`] is exactly the causal window for every
    /// token — then the dense attention path runs instead of a per-token
    /// gather, with the same answer.
    causal: bool,
}

pub struct DsaModel {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    shapes: Vec<DsaShape>,
    routing: ExpertRouting,
    /// `final_logit_softcapping` — `x -> cap * tanh(x / cap)` on the logits.
    softcap: Option<f32>,
    kv_dims: Vec<(usize, usize)>,
    layers: Vec<DsaLayer>,
}

/// The trunk's layer count: `block_count` less any multi-token-prediction
/// blocks the file ships inside it. Running one as a trunk layer would be
/// silently wrong rather than a load failure — its tensors are all present
/// and correctly shaped.
pub(crate) fn trunk_layer_count(loaded: &LoadedModel) -> Result<usize> {
    let n_layer_nextn = loaded.metadata_u64("nextn_predict_layers").unwrap_or(0) as usize;
    anyhow::ensure!(
        n_layer_nextn < loaded.config.n_layer,
        "nextn_predict_layers ({n_layer_nextn}) must be fewer than block_count ({})",
        loaded.config.n_layer
    );
    Ok(loaded.config.n_layer - n_layer_nextn)
}

/// `attention.indexer.types`, one flag per trunk layer: whether that layer
/// scores positions itself. A file without the key gets `default`.
pub(crate) fn indexer_types(
    loaded: &LoadedModel,
    n_layer: usize,
    default: impl Fn(usize) -> bool,
) -> Vec<bool> {
    let types = loaded.metadata_array_u64("attention.indexer.types");
    (0..n_layer)
        .map(|il| match &types {
            Some(t) => t.get(il).copied().unwrap_or(1) != 0,
            None => default(il),
        })
        .collect()
}

impl DsaModel {
    /// Builds the model around `plans`: the embeddings and the output head
    /// (tied to the embeddings when the file carries no `output.weight`),
    /// every layer's weights, and the cache layout.
    pub(crate) fn load(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        shapes: Vec<DsaShape>,
        plans: &[LayerPlan],
    ) -> Result<Self> {
        validate_plans(plans, shapes.len())?;
        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = if loaded.has_tensor("output.weight") {
            loaded
                .matrix("output.weight")
                .context("loading output.weight")?
        } else {
            tok_embeddings.clone()
        };
        let n_expert = loaded.metadata_u64("expert_count").map(|n| n as usize);

        let mut kv_dims: Vec<(usize, usize)> = Vec::new();
        // Each layer's own cache slot, so a reading layer can name its
        // source's.
        let mut own_slot = vec![usize::MAX; plans.len()];
        let mut layers = Vec::with_capacity(plans.len());
        for (i, plan) in plans.iter().enumerate() {
            let shape = &shapes[plan.shape];
            let get = |suffix: &str| -> Result<Vec<f32>> {
                let name = format!("blk.{i}.{suffix}");
                Ok(loaded
                    .tensor(&name)
                    .with_context(|| format!("loading {name}"))?
                    .0)
            };
            let mla = if plan.kv_source == i {
                kv_dims.push((shape.mla.kv_row, 1));
                own_slot[i] = kv_dims.len() - 1;
                MlaLayer::load(loaded, i, &shape.mla, own_slot[i])?
            } else {
                MlaLayer::load_reading(loaded, i, &shape.mla, own_slot[plan.kv_source])?
            };
            let indexer = if plan.scores {
                kv_dims.push((shape.indexer.head_size, 1));
                let slot = kv_dims.len() - 1;
                // Per token, not pooled: there is no pooled-key slot to
                // name, so the argument is never read.
                Some(Indexer::load(loaded, i, None, &shape.indexer, slot, slot)?)
            } else {
                None
            };
            layers.push(DsaLayer {
                attn_norm: get("attn_norm.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                shape: plan.shape,
                mla,
                indexer,
                ffn: LayerFfn::load(loaded, i, plan.dense, n_expert)?,
            });
        }

        Ok(Self {
            config: loaded.config.clone(),
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            shapes,
            routing: ExpertRouting::from_gguf(loaded)?,
            softcap: loaded
                .metadata_f32("final_logit_softcapping")
                .filter(|c| *c != 0.0),
            kv_dims,
            layers,
        })
    }

    /// Runs `tokens` through every trunk layer and returns each one's final
    /// normed hidden state (`[n_tokens, n_embd]`).
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

        let mut seqs = Seqs::One { cache, start_pos };
        let mut scratch = LayerScratch::default();
        for (il, layer) in self.layers.iter().enumerate() {
            self.forward_layer(il, layer, &mut seqs, &mut x, n_tokens, &mut scratch)?;
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

    fn forward_layer(
        &self,
        il: usize,
        layer: &DsaLayer,
        seqs: &mut Seqs<'_, '_>,
        x: &mut [f32],
        n_tokens: usize,
        scratch: &mut LayerScratch,
    ) -> Result<()> {
        let backend = self.backend.as_ref();
        let n_embd = self.config.n_embd;
        let eps = self.config.rms_eps;
        let shape = &self.shapes[layer.shape];
        let LayerScratch {
            normed,
            attn_out,
            ffn_out,
            ffn,
            mla,
            selection,
            causal,
        } = scratch;

        tensor::rmsnorm_into(normed, x, &layer.attn_norm, n_tokens, n_embd, eps);
        // The query first: a layer whose indexer shares the query's LoRA
        // intermediate needs it before deciding what to attend.
        layer
            .mla
            .project_query_into(backend, mla, &shape.mla, normed, n_tokens);
        if let Some(ix) = &layer.indexer {
            let inputs = ix.inputs(backend, &shape.indexer, mla.qr(), normed, n_tokens, &|t| {
                seqs.pos(t)
            });
            selection.clear();
            for t in 0..n_tokens {
                let pos = seqs.pos(t);
                selection.push(ix.select(&shape.indexer, seqs.cache(t), &inputs, t, pos));
            }
            *causal = selection
                .iter()
                .enumerate()
                .all(|(t, s)| s.len() == seqs.pos(t) + 1);
        }
        anyhow::ensure!(
            selection.len() == n_tokens,
            "layer {il} reuses a lightning-indexer selection, but no earlier layer produced one"
        );
        // A selection that is the whole causal window for every token is
        // the dense attention, and that path reads the prefix once instead
        // of gathering it per token.
        let chosen = (!*causal).then_some(selection.as_slice());
        layer.mla.attend_into(
            backend, attn_out, mla, &shape.mla, seqs, normed, n_tokens, chosen,
        );
        tensor::add_inplace(x, attn_out);

        tensor::rmsnorm_into(normed, x, &layer.ffn_norm, n_tokens, n_embd, eps);
        layer.ffn.forward_into(
            backend,
            &self.routing,
            ffn_out,
            ffn,
            normed,
            n_tokens,
            n_embd,
            FfnLimits::NONE,
            il,
        );
        tensor::add_inplace(x, ffn_out);
        Ok(())
    }

    /// One position's logits off its final hidden state.
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = self.backend.matmul(hidden, 1, &self.output_weight);
        if let Some(cap) = self.softcap {
            tensor::softcap_inplace(&mut logits, cap);
        }
        logits
    }
}

impl ModelForward for DsaModel {
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
        KvCache::new_with_strided_dims(capacity, &self.kv_dims)
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
        Ok(self.logits(&hidden[(tokens.len() - 1) * n_embd..]))
    }

    /// Every input position's next-token logits, for the speculative
    /// verify step — sound because every cache here is purely positional,
    /// so `KvCache::truncate` rolls a rejected draft's tail back exactly.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_all_logits called with no tokens"
        );
        let hidden = self.run_layers(cache, tokens, start_pos)?;
        let n_embd = self.config.n_embd;
        Ok(hidden
            .chunks_exact(n_embd)
            .map(|h| self.logits(h))
            .collect())
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
    use super::*;

    fn plan(shape: usize, kv_source: usize, scores: bool) -> LayerPlan {
        LayerPlan {
            shape,
            kv_source,
            scores,
            dense: false,
        }
    }

    #[test]
    fn a_stack_that_never_scored_is_refused() {
        let plans = [plan(0, 0, false), plan(0, 1, true)];
        assert!(validate_plans(&plans, 1).is_err());
    }

    /// A layer may read only rows an earlier layer of its own shape writes
    /// itself: a later layer's rows do not exist yet, a reading layer's
    /// rows are not its own, and a different shape's rows are a different
    /// width.
    #[test]
    fn a_kv_source_must_be_an_earlier_writer_of_the_same_shape() {
        let ok = [plan(0, 0, true), plan(1, 1, true), plan(1, 1, false)];
        assert!(validate_plans(&ok, 2).is_ok());

        let later = [plan(0, 1, true), plan(0, 1, true)];
        assert!(validate_plans(&later, 1).is_err(), "reads the future");

        let chained = [plan(0, 0, true), plan(0, 0, true), plan(0, 1, true)];
        assert!(validate_plans(&chained, 1).is_err(), "reads a reader");

        let mixed = [plan(0, 0, true), plan(1, 0, true)];
        assert!(validate_plans(&mixed, 2).is_err(), "reads another shape");
    }
}

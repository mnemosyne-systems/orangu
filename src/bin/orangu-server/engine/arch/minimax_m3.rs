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

//! MiniMax-M3 (`general.architecture = "minimax-m3"`), e.g.
//! `unsloth/MiniMax-M3.1-Flash-GGUF`, transcribed from upstream
//! `llama.cpp`'s `src/models/minimax-m3.cpp`.
//!
//! A pre-norm transformer — attention, residual, FFN, residual — whose two
//! halves are each something this engine already has, with one twist apiece:
//!
//! - **Attention** is plain GQA (64 query heads over 4 key/value heads of
//!   128) with a per-head QK RMSNorm and **partial** NEOX rotation: only the
//!   leading `rope.dimension_count` (64) of each head's 128 values rotate.
//!   The twist is **MiniMax Sparse Attention (MSA)** on every layer from
//!   `leading_dense_block_count` on: positions are cut into
//!   `attention.indexer.block_size` blocks, a small indexer (one 128-wide
//!   query head per key/value group, one shared 128-wide key per token,
//!   each RMS-normed and rotated like the attention's) scores every visible
//!   position, a block scores its best position, and each group attends
//!   only its `attention.indexer.top_k` best blocks — with the query's own
//!   block (`attention.indexer.local_blocks` of them, counting back) always
//!   among them. See [`select_blocks`].
//! - **The FFN** is the DeepSeek-V3 shape [`super::ffn::LayerFfn`] already
//!   builds — dense on the leading blocks, then sigmoid-routed experts with
//!   a selection-only bias, renormalized and scaled, beside one always-on
//!   shared expert. The twist is the activation: gpt-oss's `swiglu_oai`
//!   ([`super::SwigluLimit::OAI`]) on all three arms, dense included, and
//!   per-expert output scales (`ffn_*_exps.scale`), since the routed
//!   experts are NVFP4 stored without their tensor-wide scale.
//!
//! MSA reduces to ordinary causal attention whenever a query can see no
//! more than `top_k` blocks — every query in the first
//! `top_k * block_size` (2048) positions — and the dense path, with its GPU
//! kernel, serves those. Past that the sparse selection runs on the host.
//!
//! Like `muse-spark` and `glm`, the layer loop is host-orchestrated: every
//! matmul still dispatches to the `Backend`.
//!
//! Not carried over from upstream: the Hadamard rotation of the cached keys
//! and values (`self_k_rot`/`self_v_rot`), which exists for quantized caches
//! and is orthonormal on both sides of every product it touches; and the
//! vision tower, which ships beside the model as a separate `mmproj-*.gguf`.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use super::ffn::{FfnLimits, LayerFfn};
use super::{ExpertRouting, FfnScratch, ModelForward, Seqs};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::kv_cache::{KvCache, LayerCache};
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

/// MiniMax Sparse Attention's hyperparameters, the same for every sparse
/// layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MsaShape {
    /// `attention.indexer.block_size` — positions per block.
    pub block: usize,
    /// `attention.indexer.top_k` — blocks each key/value group attends.
    pub top_k: usize,
    /// `attention.indexer.local_blocks` — how many blocks, counting back
    /// from the query's own, are always attended.
    pub local: usize,
    /// `attention.indexer.key_length` — the indexer's query and key width.
    pub head_size: usize,
}

impl MsaShape {
    fn from_gguf(loaded: &LoadedModel) -> Result<Self> {
        let get = |key: &str| -> Result<usize> {
            Ok(loaded
                .metadata_u64(key)
                .with_context(|| format!("missing {key}"))? as usize)
        };
        let shape = Self {
            block: get("attention.indexer.block_size")?,
            top_k: get("attention.indexer.top_k")?,
            local: get("attention.indexer.local_blocks")?,
            head_size: get("attention.indexer.key_length")?,
        };
        anyhow::ensure!(
            shape.block > 0 && shape.top_k > 0,
            "attention.indexer.block_size and top_k must be nonzero"
        );
        Ok(shape)
    }

    /// Whether a query at `pos` sees few enough blocks that MSA attends all
    /// of them — ordinary causal attention.
    fn is_dense_at(&self, pos: usize) -> bool {
        pos / self.block < self.top_k
    }
}

/// The blocks one key/value group of a query at `pos` attends, ascending.
///
/// `scores[q]` is the indexer score of position `q` for `q <= pos`. A block
/// scores the maximum over its visible positions; the query's own block and
/// the `local - 1` before it are forced in (upstream adds `1e30` to their
/// scores before the top-k), and the rest of the `top_k` budget goes to the
/// best-scoring remaining blocks. A query that sees no more than `top_k`
/// blocks attends all of them.
pub(crate) fn select_blocks(scores: &[f32], pos: usize, shape: &MsaShape) -> Vec<usize> {
    debug_assert!(scores.len() > pos);
    let last = pos / shape.block;
    let n_visible = last + 1;
    if n_visible <= shape.top_k {
        return (0..n_visible).collect();
    }
    let forced_from = (last + 1).saturating_sub(shape.local.max(1).min(shape.top_k));
    let mut chosen: Vec<usize> = (forced_from..=last).collect();
    let block_scores: Vec<f32> = (0..forced_from)
        .map(|b| {
            let end = ((b + 1) * shape.block).min(pos + 1);
            scores[b * shape.block..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    let budget = shape.top_k - chosen.len();
    chosen.extend(super::top_k_indices(&block_scores, budget));
    chosen.sort_unstable();
    chosen
}

/// One sparse layer's indexer: its own query and key projections, each
/// RMS-normed per head and rotated like attention's.
struct MsaIndexer {
    /// `[n_embd, n_head_kv * head_size]` — one query head per key/value
    /// group.
    q_proj: QuantMatrix,
    /// `[n_embd, head_size]` — one key per token, shared by every group.
    k_proj: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// Where this layer's indexer keys live in `KvCache::layers`.
    cache_slot: usize,
}

struct MinimaxLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    /// `[head_dim]`, the same vector for every head, before rotation.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// `Some` on the sparse layers — every one from
    /// `leading_dense_block_count` on.
    indexer: Option<MsaIndexer>,
    ffn_norm: Vec<f32>,
    ffn: LayerFfn,
}

pub struct MinimaxM3Model {
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    tok_embeddings: QuantMatrix,
    output_norm: Vec<f32>,
    output_weight: QuantMatrix,
    layers: Vec<MinimaxLayer>,
    routing: ExpertRouting,
    msa: MsaShape,
    /// NEOX pairing over the leading `rope.dimension_count` of each head.
    rope: tensor::RopeParams,
    /// One `(kv_dim, 1)` per cache slot: the attention layers, then one
    /// indexer-key slot per sparse layer.
    kv_dims: Vec<usize>,
}

impl MinimaxM3Model {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = loaded.config.clone();
        let n_layer = config.n_layer;
        let n_dense = loaded
            .metadata_u64("leading_dense_block_count")
            .unwrap_or(0) as usize;
        let n_expert = loaded
            .metadata_u64("expert_count")
            .context("missing expert_count")? as usize;
        let routing = ExpertRouting::from_gguf(loaded)?;
        let msa = MsaShape::from_gguf(loaded)?;
        let n_idx_head = loaded
            .metadata_u64("attention.indexer.head_count")
            .context("missing attention.indexer.head_count")? as usize;
        anyhow::ensure!(
            n_idx_head == config.n_head_kv,
            "attention.indexer.head_count ({n_idx_head}) must equal attention.head_count_kv \
             ({}): MSA scores one indexer head per key/value group",
            config.n_head_kv
        );
        anyhow::ensure!(
            config.n_head.is_multiple_of(config.n_head_kv),
            "head_count ({}) is not a multiple of head_count_kv ({})",
            config.n_head,
            config.n_head_kv
        );
        anyhow::ensure!(
            msa.head_size >= config.rope_dim && config.head_dim >= config.rope_dim,
            "rope.dimension_count ({}) exceeds a head ({}) or the indexer width ({})",
            config.rope_dim,
            config.head_dim,
            msa.head_size
        );

        let tok_embeddings = loaded
            .matrix("token_embd.weight")
            .context("loading token_embd.weight")?;
        let (output_norm, _) = loaded
            .tensor("output_norm.weight")
            .context("loading output_norm.weight")?;
        let output_weight = loaded
            .matrix("output.weight")
            .context("loading output.weight")?;

        let kv_dim = config.n_head_kv * config.head_dim;
        let mut kv_dims = vec![kv_dim; n_layer];
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let get = |suffix: &str| -> Result<Vec<f32>> {
                let name = format!("blk.{il}.{suffix}");
                Ok(loaded
                    .tensor(&name)
                    .with_context(|| format!("loading {name}"))?
                    .0)
            };
            let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
                let name = format!("blk.{il}.{suffix}");
                loaded
                    .matrix(&name)
                    .with_context(|| format!("loading {name}"))
            };
            let dense = il < n_dense;
            let indexer = if dense {
                None
            } else {
                let cache_slot = kv_dims.len();
                kv_dims.push(msa.head_size);
                let indexer = MsaIndexer {
                    q_proj: get_matrix("indexer.q_proj.weight")?,
                    k_proj: get_matrix("indexer.k_proj.weight")?,
                    q_norm: get("indexer.q_norm.weight")?,
                    k_norm: get("indexer.k_norm.weight")?,
                    cache_slot,
                };
                anyhow::ensure!(
                    indexer.q_proj.out_dim == n_idx_head * msa.head_size
                        && indexer.k_proj.out_dim == msa.head_size,
                    "layer {il}'s indexer projects to {}/{}, not {n_idx_head} heads of {} \
                     and one key",
                    indexer.q_proj.out_dim,
                    indexer.k_proj.out_dim,
                    msa.head_size
                );
                Some(indexer)
            };
            layers.push(MinimaxLayer {
                attn_norm: get("attn_norm.weight")?,
                wq: get_matrix("attn_q.weight")?,
                wk: get_matrix("attn_k.weight")?,
                wv: get_matrix("attn_v.weight")?,
                wo: get_matrix("attn_output.weight")?,
                q_norm: get("attn_q_norm.weight")?,
                k_norm: get("attn_k_norm.weight")?,
                indexer,
                ffn_norm: get("ffn_norm.weight")?,
                ffn: LayerFfn::load(loaded, il, dense, Some(n_expert))?,
            });
        }

        let rope = tensor::RopeParams {
            rope_dim: config.rope_dim,
            freq_base: config.rope_freq_base,
            layout: tensor::RopeLayout::Neox,
            ..tensor::RopeParams::default()
        };
        Ok(Self {
            config,
            backend,
            tok_embeddings,
            output_norm,
            output_weight,
            layers,
            routing,
            msa,
            rope,
            kv_dims,
        })
    }

    /// Every layer, from the token embeddings to the last residual —
    /// `[n_tokens, n_embd]`. `seqs` says whose cache and position each token
    /// is: one sequence's run, or a batch's rows. The projections and the
    /// experts run over every token at once either way; each sequence's
    /// tokens rotate at their own positions and attend their own cache.
    fn run_layers(&self, seqs: &mut Seqs<'_, '_>, tokens: &[u32]) -> Result<Vec<f32>> {
        self.run_layers_observed(seqs, tokens, &mut |_, _| {})
    }

    /// [`Self::run_layers`], handing each layer's output residual to
    /// `observe` — for checking a layer at a time against a reference.
    fn run_layers_observed(
        &self,
        seqs: &mut Seqs<'_, '_>,
        tokens: &[u32],
        observe: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_tokens = tokens.len();
        let n_embd = cfg.n_embd;
        let eps = cfg.rms_eps;
        let _clock = super::hold_clock_for_step(self.backend.as_ref(), n_tokens);
        let _ops = super::OpSpan::open(self.backend.as_ref(), seqs.pos(0));

        let mut x = vec![0f32; n_tokens * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < cfg.n_vocab, "token id {tok} is out of vocab range");
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(&self.tok_embeddings.row(tok));
        }

        let runs = runs_of(seqs, n_tokens);
        let mut normed: Vec<f32> = Vec::new();
        let mut ffn_out: Vec<f32> = Vec::new();
        let mut scratch = FfnScratch::default();
        for (il, layer) in self.layers.iter().enumerate() {
            tensor::rmsnorm_into(&mut normed, &x, &layer.attn_norm, n_tokens, n_embd, eps);
            let attn = self.attention(layer, seqs, &runs, il, &normed, n_tokens);
            tensor::add_inplace(&mut x, &attn);

            tensor::rmsnorm_into(&mut normed, &x, &layer.ffn_norm, n_tokens, n_embd, eps);
            layer.ffn.forward_into(
                self.backend.as_ref(),
                &self.routing,
                &mut ffn_out,
                &mut scratch,
                &normed,
                n_tokens,
                n_embd,
                FfnLimits::OAI,
                il,
            );
            tensor::add_inplace(&mut x, &ffn_out);
            observe(il, &x);
        }
        Ok(x)
    }

    /// One layer's attention from its normed input to the output
    /// projection's result, appending each token's K/V (and, on a sparse
    /// layer, its indexer key) to its own sequence's cache on the way.
    /// `runs` is [`runs_of`] for `seqs`.
    fn attention(
        &self,
        layer: &MinimaxLayer,
        seqs: &mut Seqs<'_, '_>,
        runs: &[std::ops::Range<usize>],
        il: usize,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let (n_head, n_head_kv, head_dim) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
        let kv_dim = n_head_kv * head_dim;
        let q_dim = n_head * head_dim;
        let eps = cfg.rms_eps;
        let backend = self.backend.as_ref();

        let mut ops = vec![
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wq,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wk,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: &layer.wv,
            },
        ];
        if let Some(idx) = &layer.indexer {
            ops.push(MatmulOp {
                x: normed,
                n_tokens,
                w: &idx.q_proj,
            });
            ops.push(MatmulOp {
                x: normed,
                n_tokens,
                w: &idx.k_proj,
            });
        }
        let mut proj = backend.matmul_batch(&ops);
        let mut indexer_qk = layer.indexer.as_ref().map(|_| {
            let ik = proj.pop().unwrap();
            let iq = proj.pop().unwrap();
            (iq, ik)
        });
        let v = proj.pop().unwrap();
        let mut k = proj.pop().unwrap();
        let mut q = proj.pop().unwrap();

        tensor::rmsnorm_inplace(&mut q, &layer.q_norm, n_tokens * n_head, head_dim, eps);
        tensor::rmsnorm_inplace(&mut k, &layer.k_norm, n_tokens * n_head_kv, head_dim, eps);
        let idx_dim = self.msa.head_size;
        if let (Some(idx), Some((iq, ik))) = (&layer.indexer, indexer_qk.as_mut()) {
            tensor::rmsnorm_inplace(iq, &idx.q_norm, n_tokens * n_head_kv, idx_dim, eps);
            tensor::rmsnorm_inplace(ik, &idx.k_norm, n_tokens, idx_dim, eps);
        }

        // Rotate every token at its own position and append it to its own
        // sequence's cache, in order: a later token of the same run reads
        // the earlier ones.
        let zeros = vec![0f32; idx_dim];
        for t in 0..n_tokens {
            let pos = seqs.pos(t);
            tensor::rope_apply_params_inplace(
                &mut q[t * q_dim..(t + 1) * q_dim],
                n_head,
                head_dim,
                pos,
                None,
                &self.rope,
            );
            tensor::rope_apply_params_inplace(
                &mut k[t * kv_dim..(t + 1) * kv_dim],
                n_head_kv,
                head_dim,
                pos,
                None,
                &self.rope,
            );
            let cache = seqs.cache(t);
            cache.layers[il].push(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &v[t * kv_dim..(t + 1) * kv_dim],
            );
            if let (Some(idx), Some((iq, ik))) = (&layer.indexer, indexer_qk.as_mut()) {
                let iq_dim = n_head_kv * idx_dim;
                tensor::rope_apply_params_inplace(
                    &mut iq[t * iq_dim..(t + 1) * iq_dim],
                    n_head_kv,
                    idx_dim,
                    pos,
                    None,
                    &self.rope,
                );
                tensor::rope_apply_params_inplace(
                    &mut ik[t * idx_dim..(t + 1) * idx_dim],
                    1,
                    idx_dim,
                    pos,
                    None,
                    &self.rope,
                );
                // The value half of an indexer slot is never read.
                cache.layers[idx.cache_slot].push(&ik[t * idx_dim..(t + 1) * idx_dim], &zeros);
            }
        }

        // Each run attends its own cache, from its own first position.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0f32; n_tokens * q_dim];
        let mut run_out = Vec::new();
        for run in runs {
            let start_pos = seqs.pos(run.start);
            let n = run.len();
            let cache = seqs.cache(run.start);
            let q_run = &q[run.start * q_dim..run.end * q_dim];
            match (&layer.indexer, &indexer_qk) {
                // Sparse, and some query in this run sees more blocks than
                // the budget — select per query and group on the host.
                (Some(idx), Some((iq, _))) if !self.msa.is_dense_at(start_pos + n - 1) => {
                    let iq_dim = n_head_kv * idx_dim;
                    let (attn_layers, idx_layers) = cache.layers.split_at_mut(idx.cache_slot);
                    msa_attention(
                        &mut run_out,
                        q_run,
                        &iq[run.start * iq_dim..run.end * iq_dim],
                        &attn_layers[il],
                        &idx_layers[0],
                        &MsaParams {
                            shape: &self.msa,
                            n_head,
                            n_head_kv,
                            head_dim,
                            scale,
                            start_pos,
                            n_tokens: n,
                        },
                    );
                }
                // A dense layer, or a sparse one whose every query attends
                // all its blocks — ordinary causal attention.
                _ => {
                    crate::engine::attention::attention(
                        &mut run_out,
                        q_run,
                        &mut cache.layers[il],
                        &crate::engine::attention::Params {
                            backend,
                            device: layer.wo.device(),
                            n_head,
                            n_head_kv,
                            head_dim,
                            scale,
                            causal: true,
                            n_swa: 0,
                            start_pos,
                            n_tokens: n,
                        },
                        |t| (0, start_pos + t),
                    );
                }
            }
            attn_out[run.start * q_dim..run.end * q_dim].copy_from_slice(&run_out);
        }
        backend.matmul(&attn_out, n_tokens, &layer.wo)
    }

    /// The output norm and projection over `[n_tokens, n_embd]` hidden
    /// states.
    fn project(&self, hidden: &mut [f32], n_tokens: usize) -> Vec<f32> {
        tensor::rmsnorm_inplace(
            hidden,
            &self.output_norm,
            n_tokens,
            self.config.n_embd,
            self.config.rms_eps,
        );
        self.backend.matmul(hidden, n_tokens, &self.output_weight)
    }
}

/// The pass's tokens grouped by sequence: one range for a single run, one
/// per row for a decode batch, one per row's run for a verify batch. Every
/// range's tokens share a cache and sit at consecutive positions.
fn runs_of(seqs: &Seqs<'_, '_>, n_tokens: usize) -> Vec<std::ops::Range<usize>> {
    match seqs {
        Seqs::One { .. } => std::iter::once(0..n_tokens).collect(),
        Seqs::Rows(_) => (0..n_tokens).map(|t| t..t + 1).collect(),
        Seqs::Runs { owner, .. } => {
            let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
            for (t, &(row, _)) in owner.iter().enumerate() {
                match runs.last_mut() {
                    Some(run) if owner[run.start].0 == row => run.end = t + 1,
                    _ => runs.push(t..t + 1),
                }
            }
            runs
        }
    }
}

/// The shape [`msa_attention`] needs.
pub(crate) struct MsaParams<'a> {
    pub shape: &'a MsaShape,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub start_pos: usize,
    pub n_tokens: usize,
}

/// MSA over a batch of queries: for each query and key/value group, score
/// every visible position with the group's indexer head, pick the blocks
/// ([`select_blocks`]), and attend the group's query heads over just those
/// positions. `out` is resized to `[n_tokens, n_head * head_dim]`.
///
/// `q` is `[n_tokens, n_head * head_dim]`, `iq` is
/// `[n_tokens, n_head_kv * shape.head_size]`; `kv` and `index_keys` must
/// already hold every position up to the last query's.
pub(crate) fn msa_attention(
    out: &mut Vec<f32>,
    q: &[f32],
    iq: &[f32],
    kv: &LayerCache,
    index_keys: &LayerCache,
    p: &MsaParams<'_>,
) {
    let MsaParams {
        shape,
        n_head,
        n_head_kv,
        head_dim,
        scale,
        start_pos,
        n_tokens,
    } = *p;
    let group = n_head / n_head_kv;
    let idx_dim = shape.head_size;
    let width = n_head * head_dim;
    out.clear();
    out.resize(n_tokens * width, 0.0);
    out.par_chunks_mut(group * head_dim)
        .enumerate()
        .for_each(|(i, dst)| {
            let (t, g) = (i / n_head_kv, i % n_head_kv);
            let pos = start_pos + t;
            let iq_g = &iq[(t * n_head_kv + g) * idx_dim..(t * n_head_kv + g + 1) * idx_dim];
            let positions: Vec<usize> = if shape.is_dense_at(pos) {
                (0..=pos).collect()
            } else {
                let scores: Vec<f32> = (0..=pos)
                    .map(|j| tensor::dot(iq_g, index_keys.key_at(j, 0, idx_dim)))
                    .collect();
                select_blocks(&scores, pos, shape)
                    .into_iter()
                    .flat_map(|b| b * shape.block..((b + 1) * shape.block).min(pos + 1))
                    .collect()
            };
            let mut weights = vec![0f32; positions.len()];
            for h in 0..group {
                let head = g * group + h;
                let qh = &q[(t * n_head + head) * head_dim..(t * n_head + head + 1) * head_dim];
                for (w, &j) in weights.iter_mut().zip(&positions) {
                    *w = tensor::dot(qh, kv.key_at(j, g, head_dim)) * scale;
                }
                tensor::softmax_inplace(&mut weights);
                let o = &mut dst[h * head_dim..(h + 1) * head_dim];
                for (&w, &j) in weights.iter().zip(&positions) {
                    for (o, &v) in o.iter_mut().zip(kv.value_at(j, g, head_dim)) {
                        *o += w * v;
                    }
                }
            }
        });
}

impl ModelForward for MinimaxM3Model {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_with_dims(capacity, &self.kv_dims)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "forward called with no tokens");
        let n_embd = self.config.n_embd;
        let x = self.run_layers(&mut Seqs::One { cache, start_pos }, tokens)?;
        let mut last = x[(tokens.len() - 1) * n_embd..].to_vec();
        Ok(self.project(&mut last, 1))
    }

    /// Every input position's next-token logits, for the speculative
    /// verify step — sound because every cache slot here is positional, so
    /// `KvCache::truncate` rolls a rejected draft's tail back exactly.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let n_tokens = tokens.len();
        let n_vocab = self.config.n_vocab;
        let mut h = self.run_layers(&mut Seqs::One { cache, start_pos }, tokens)?;
        let flat = self.project(&mut h, n_tokens);
        anyhow::ensure!(
            flat.len() == n_tokens * n_vocab,
            "output projection produced {} logits, expected {}",
            flat.len(),
            n_tokens * n_vocab
        );
        Ok(flat.chunks(n_vocab).map(<[f32]>::to_vec).collect())
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "forward_hidden_states called with no tokens"
        );
        let mut cache = self.new_kv_cache(tokens.len());
        let mut x = self.run_layers(
            &mut Seqs::One {
                cache: &mut cache,
                start_pos: 0,
            },
            tokens,
        )?;
        tensor::rmsnorm_inplace(
            &mut x,
            &self.output_norm,
            tokens.len(),
            self.config.n_embd,
            self.config.rms_eps,
        );
        Ok(x)
    }

    /// Every layer runs on the host's turn of the step, so a batch is the
    /// same layer loop with each row's keys on its own cache.
    fn batches_decode(&self) -> bool {
        true
    }

    /// One decode step for several sequences: every projection and expert
    /// read once for all the rows, each row's attention on its own cache,
    /// and one vocabulary projection for every row.
    fn forward_decode_batch(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let n_rows = rows.len();
        if n_rows < 2 || tokens.len() != n_rows {
            return Ok(None);
        }
        let mut hidden = self.run_layers(&mut Seqs::Rows(rows), tokens)?;
        let flat = self.project(&mut hidden, n_rows);
        let n_vocab = flat.len() / n_rows;
        Ok(Some(
            flat.chunks_exact(n_vocab).map(<[f32]>::to_vec).collect(),
        ))
    }

    /// A verify is a run of tokens on the host's turn like any other row,
    /// so it takes its tokens into the batch.
    fn batches_verify(&self) -> bool {
        super::batch_verify()
    }

    /// One step for several sequences, each row bringing its own run of
    /// tokens from its own position — one for a decode step, a draft for a
    /// verify — with every token's logits back, as its own step or
    /// `forward_all_logits` gives them.
    fn forward_decode_batch_verify(
        &self,
        rows: &mut [super::DecodeRow<'_>],
        tokens: &[&[u32]],
        _samples: &[Option<super::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<Vec<super::ForwardOutcome>>>> {
        if !self.batches_verify()
            || tokens.len() != rows.len()
            || tokens.iter().any(|t| t.is_empty())
        {
            return Ok(None);
        }
        let widths: Vec<usize> = tokens.iter().map(|t| t.len()).collect();
        let flat: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        let n_tokens = flat.len();
        let mut hidden = self.run_layers(&mut Seqs::runs(rows, &widths), &flat)?;
        let logits = self.project(&mut hidden, n_tokens);
        let n_vocab = logits.len() / n_tokens;
        let mut rows_out = logits
            .chunks_exact(n_vocab)
            .map(|row| super::ForwardOutcome::Logits(row.to_vec()));
        Ok(Some(
            widths
                .iter()
                .map(|&w| rows_out.by_ref().take(w).collect())
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: MsaShape = MsaShape {
        block: 4,
        top_k: 3,
        local: 1,
        head_size: 2,
    };

    /// Up to `top_k` visible blocks, every one is attended — the reason the
    /// first `top_k * block` positions can take the dense path.
    #[test]
    fn a_query_with_few_blocks_attends_them_all() {
        let scores = vec![0.0; 12];
        assert_eq!(select_blocks(&scores, 0, &SHAPE), vec![0]);
        assert_eq!(select_blocks(&scores, 11, &SHAPE), vec![0, 1, 2]);
        assert!(SHAPE.is_dense_at(11));
        assert!(!SHAPE.is_dense_at(12));
    }

    /// The query's own block is always in, however badly it scores, and a
    /// block's score is its best *visible* position: the far-future scores
    /// at positions past `pos` must not count.
    #[test]
    fn the_own_block_is_forced_and_the_rest_go_by_best_position() {
        // Five blocks of four; the query sits at 17, in block 4.
        let mut scores = vec![0.0f32; 20];
        scores[1] = 5.0; // block 0
        scores[6] = 9.0; // block 1
        scores[10] = -1.0; // block 2 (its best)
        scores[13] = 7.0; // block 3
        scores[16] = -100.0; // block 4, the query's own
        scores[19] = 1000.0; // block 4, but past the query
        assert_eq!(select_blocks(&scores[..18], 17, &SHAPE), vec![1, 3, 4]);

        let two_local = MsaShape { local: 2, ..SHAPE };
        assert_eq!(select_blocks(&scores[..18], 17, &two_local), vec![1, 3, 4]);
        let mut low3 = scores.clone();
        low3[13] = -50.0;
        assert_eq!(select_blocks(&low3[..18], 17, &two_local), vec![1, 3, 4]);
        assert_eq!(select_blocks(&low3[..18], 17, &SHAPE), vec![0, 1, 4]);
    }

    fn cache_with(rows: &[(Vec<f32>, Vec<f32>)], dim: usize) -> LayerCache {
        let mut cache = KvCache::new(1, rows.len(), dim);
        for (k, v) in rows {
            cache.layers[0].push(k, v);
        }
        cache.layers.remove(0)
    }

    /// Below the block budget MSA *is* causal attention, so it must agree
    /// with the engine's own kernel to rounding.
    #[test]
    fn msa_matches_causal_attention_when_every_block_is_attended() {
        let (n_head, n_head_kv, head_dim, n) = (4, 2, 3, 10);
        let shape = MsaShape {
            block: 4,
            top_k: 8,
            local: 1,
            head_size: 2,
        };
        let val = |i: usize| ((i * 37 % 23) as f32 - 11.0) * 0.1;
        let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|p| {
                (
                    (0..n_head_kv * head_dim).map(|i| val(p * 7 + i)).collect(),
                    (0..n_head_kv * head_dim)
                        .map(|i| val(p * 11 + i + 3))
                        .collect(),
                )
            })
            .collect();
        let kv = cache_with(&rows, n_head_kv * head_dim);
        let idx = cache_with(
            &(0..n)
                .map(|p| (vec![val(p), val(p + 1)], vec![0.0; 2]))
                .collect::<Vec<_>>(),
            2,
        );
        let q: Vec<f32> = (0..n * n_head * head_dim).map(|i| val(i + 5)).collect();
        let iq: Vec<f32> = (0..n * n_head_kv * 2).map(|i| val(i + 9)).collect();
        let mut got = Vec::new();
        msa_attention(
            &mut got,
            &q,
            &iq,
            &kv,
            &idx,
            &MsaParams {
                shape: &shape,
                n_head,
                n_head_kv,
                head_dim,
                scale: 0.5,
                start_pos: 0,
                n_tokens: n,
            },
        );
        let mut want = vec![0f32; got.len()];
        crate::engine::attention::multi_head_attention(
            &mut want,
            &q,
            &kv,
            n_head,
            n_head / n_head_kv,
            head_dim,
            0.5,
            |t| (0, t),
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    /// Past the budget each group attends its own blocks: a group whose
    /// indexer prefers block 0 sees block 0's values and not block 1's, and
    /// the other group the reverse — so swapping which indexer head scores
    /// which group, or attending the union, gives a different answer.
    #[test]
    fn each_group_attends_the_blocks_its_own_indexer_head_chose() {
        let shape = MsaShape {
            block: 2,
            top_k: 2,
            local: 1,
            head_size: 1,
        };
        let (n_head, n_head_kv, head_dim) = (2, 2, 1);
        // Positions 0..5 in three blocks; the query is at 5 (block 2).
        // Keys all zero, so attention is a plain mean over what is chosen.
        let values = [10.0, 10.0, 20.0, 20.0, 1.0, 1.0];
        let rows: Vec<(Vec<f32>, Vec<f32>)> = values
            .iter()
            .map(|&v| (vec![0.0, 0.0], vec![v, -v]))
            .collect();
        let kv = cache_with(&rows, 2);
        // Indexer keys: +1 on block 0, -1 on block 1.
        let idx_keys = [1.0, 1.0, -1.0, -1.0, 0.0, 0.0];
        let idx = cache_with(
            &idx_keys
                .iter()
                .map(|&k| (vec![k], vec![0.0]))
                .collect::<Vec<_>>(),
            1,
        );
        // Group 0's indexer head is +1 (prefers block 0), group 1's -1.
        let iq = [1.0, -1.0];
        let q = [0.0, 0.0];
        let mut got = Vec::new();
        msa_attention(
            &mut got,
            &q,
            &iq,
            &kv,
            &idx,
            &MsaParams {
                shape: &shape,
                n_head,
                n_head_kv,
                head_dim,
                scale: 1.0,
                start_pos: 5,
                n_tokens: 1,
            },
        );
        // Group 0: blocks 0 and 2 -> mean(10, 10, 1, 1) = 5.5.
        // Group 1: blocks 1 and 2 -> mean(-20, -20, -1, -1) = -10.5.
        assert_eq!(got, vec![5.5, -10.5]);
    }
}

#[cfg(test)]
mod synthetic_tests {
    use super::*;
    use orangu::gguf::GgufValue;
    use orangu::gguf_write::TensorPlan;

    const VOCAB: u64 = 2304;
    const EMBD: u64 = 32;
    const HEADS: u64 = 4;
    const KV_HEADS: u64 = 2;
    const HEAD_DIM: u64 = 16;
    const IDX_DIM: u64 = 16;
    const EXPERTS: u64 = 4;
    const FF_EXP: u64 = 16;
    const FF: u64 = 24;
    const LAYERS: u64 = 3;
    /// A batch runs every projection and expert over all of its rows at
    /// once and a solo step over one: the sums run in a different order.
    const TOLERANCE: f64 = 1e-3;

    /// Deterministic weights in `[-scale, scale]`, different per tensor.
    fn weights(name: &str, n: usize, scale: f32) -> Vec<f32> {
        let mut h: u64 = name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        (0..n)
            .map(|_| {
                h ^= h << 13;
                h ^= h >> 7;
                h ^= h << 17;
                ((h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// A three-layer `minimax-m3` file — one dense-lead layer, two sparse
    /// ones whose 4-position blocks with a budget of 2 make MSA select from
    /// position 8 on, and experts with per-expert scales — small enough
    /// to build and run in a unit test.
    fn synthetic_loaded() -> LoadedModel {
        // One file per call: the tests run in parallel in one process.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "orangu-minimax-m3-synthetic-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_synthetic(&path);
        let loaded = LoadedModel::open(&path).expect("open the synthetic model");
        std::fs::remove_file(&path).ok();
        loaded
    }

    /// The synthetic file, with a byte-level vocabulary — id `b` is byte
    /// `b`, and no merges apply to a prompt of letters and spaces — so
    /// another engine can be handed the same file and the same text.
    fn write_synthetic(path: &std::path::Path) {
        let p = |s: &str| format!("minimax-m3.{s}");
        let bytes = crate::engine::tokenizer::byte_to_unicode_table();
        let tokens: Vec<GgufValue> = (0..VOCAB as usize)
            .map(|i| {
                GgufValue::String(match i {
                    0..256 => bytes[i].to_string(),
                    256 => "zz".to_string(),
                    _ => format!("<|filler_{i}|>"),
                })
            })
            .collect();
        let token_types: Vec<GgufValue> = (0..VOCAB as usize)
            .map(|i| GgufValue::I32(if i > 256 { 3 } else { 1 }))
            .collect();
        let metadata: Vec<(String, GgufValue)> = vec![
            (
                "tokenizer.ggml.model".into(),
                GgufValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                GgufValue::String("minimax-m2".into()),
            ),
            ("tokenizer.ggml.tokens".into(), GgufValue::Array(tokens)),
            (
                "tokenizer.ggml.token_type".into(),
                GgufValue::Array(token_types),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufValue::Array(vec![GgufValue::String("z z".into())]),
            ),
            ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(257)),
            (
                "tokenizer.ggml.add_bos_token".into(),
                GgufValue::Bool(false),
            ),
            (
                "general.architecture".into(),
                GgufValue::String("minimax-m3".into()),
            ),
            (p("vocab_size"), GgufValue::U32(VOCAB as u32)),
            (p("block_count"), GgufValue::U32(LAYERS as u32)),
            (p("context_length"), GgufValue::U32(256)),
            (p("embedding_length"), GgufValue::U32(EMBD as u32)),
            (p("feed_forward_length"), GgufValue::U32(FF as u32)),
            (p("attention.head_count"), GgufValue::U32(HEADS as u32)),
            (
                p("attention.head_count_kv"),
                GgufValue::U32(KV_HEADS as u32),
            ),
            (p("attention.key_length"), GgufValue::U32(HEAD_DIM as u32)),
            (p("attention.value_length"), GgufValue::U32(HEAD_DIM as u32)),
            (p("rope.dimension_count"), GgufValue::U32(8)),
            (p("rope.freq_base"), GgufValue::F32(10000.0)),
            (p("attention.layer_norm_rms_epsilon"), GgufValue::F32(1e-6)),
            (p("expert_count"), GgufValue::U32(EXPERTS as u32)),
            (p("expert_used_count"), GgufValue::U32(2)),
            (p("expert_gating_func"), GgufValue::U32(2)),
            (
                p("expert_feed_forward_length"),
                GgufValue::U32(FF_EXP as u32),
            ),
            (p("expert_shared_count"), GgufValue::U32(1)),
            (p("expert_weights_scale"), GgufValue::F32(2.0)),
            (p("expert_weights_norm"), GgufValue::Bool(true)),
            (p("leading_dense_block_count"), GgufValue::U32(1)),
            (
                p("attention.indexer.head_count"),
                GgufValue::U32(KV_HEADS as u32),
            ),
            (
                p("attention.indexer.key_length"),
                GgufValue::U32(IDX_DIM as u32),
            ),
            (p("attention.indexer.top_k"), GgufValue::U32(2)),
            (p("attention.indexer.block_size"), GgufValue::U32(4)),
            (p("attention.indexer.local_blocks"), GgufValue::U32(1)),
        ];

        let mut plans: Vec<(TensorPlan, f32)> = Vec::new();
        let mut add = |name: String, dims: &[u64], scale: f32| {
            plans.push((
                TensorPlan {
                    name,
                    dims: dims.to_vec(),
                    ggml_type: crate::engine::quant::GGML_TYPE_F32,
                },
                scale,
            ));
        };
        // A scale of 0 marks a norm weight, written as ones.
        add("token_embd.weight".into(), &[EMBD, VOCAB], 1.0);
        add("output_norm.weight".into(), &[EMBD], 0.0);
        add("output.weight".into(), &[EMBD, VOCAB], 0.3);
        for i in 0..LAYERS {
            let mut layer = |s: &str, dims: &[u64], scale: f32| {
                add(format!("blk.{i}.{s}"), dims, scale);
            };
            layer("attn_norm.weight", &[EMBD], 0.0);
            layer("attn_q.weight", &[EMBD, HEADS * HEAD_DIM], 0.3);
            layer("attn_k.weight", &[EMBD, KV_HEADS * HEAD_DIM], 0.3);
            layer("attn_v.weight", &[EMBD, KV_HEADS * HEAD_DIM], 0.3);
            layer("attn_output.weight", &[HEADS * HEAD_DIM, EMBD], 0.3);
            layer("attn_q_norm.weight", &[HEAD_DIM], 0.0);
            layer("attn_k_norm.weight", &[HEAD_DIM], 0.0);
            layer("indexer.q_proj.weight", &[EMBD, KV_HEADS * IDX_DIM], 0.5);
            layer("indexer.k_proj.weight", &[EMBD, IDX_DIM], 0.5);
            layer("indexer.q_norm.weight", &[IDX_DIM], 0.0);
            layer("indexer.k_norm.weight", &[IDX_DIM], 0.0);
            layer("ffn_norm.weight", &[EMBD], 0.0);
            if i == 0 {
                layer("ffn_gate.weight", &[EMBD, FF], 0.3);
                layer("ffn_up.weight", &[EMBD, FF], 0.3);
                layer("ffn_down.weight", &[FF, EMBD], 0.3);
            } else {
                layer("ffn_gate_inp.weight", &[EMBD, EXPERTS], 0.5);
                layer("exp_probs_b.bias", &[EXPERTS], 0.1);
                layer("ffn_gate_exps.weight", &[EMBD, FF_EXP, EXPERTS], 0.3);
                layer("ffn_up_exps.weight", &[EMBD, FF_EXP, EXPERTS], 0.3);
                layer("ffn_down_exps.weight", &[FF_EXP, EMBD, EXPERTS], 0.3);
                layer("ffn_gate_exps.scale", &[EXPERTS], 2.0);
                layer("ffn_up_exps.scale", &[EXPERTS], 2.0);
                layer("ffn_down_exps.scale", &[EXPERTS], 2.0);
                layer("ffn_gate_shexp.weight", &[EMBD, FF_EXP], 0.3);
                layer("ffn_up_shexp.weight", &[EMBD, FF_EXP], 0.3);
                layer("ffn_down_shexp.weight", &[FF_EXP, EMBD], 0.3);
            }
        }

        let scales: std::collections::HashMap<String, f32> =
            plans.iter().map(|(t, s)| (t.name.clone(), *s)).collect();
        let tensors: Vec<TensorPlan> = plans.into_iter().map(|(t, _)| t).collect();
        orangu::gguf_write::write(
            path,
            &metadata,
            &tensors,
            |t| {
                let scale = scales[&t.name];
                Ok(if scale == 0.0 {
                    vec![1.0; t.elements()]
                } else {
                    weights(&t.name, t.elements(), scale)
                })
            },
            &|_, _, _| {},
        )
        .expect("write the synthetic model");
    }

    fn synthetic_model() -> MinimaxM3Model {
        MinimaxM3Model::load_with_backend(
            &synthetic_loaded(),
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build the synthetic model")
    }

    /// Writes the synthetic file to `ORANGU_MM3_SYNTHETIC` and prints, for
    /// a 40-token prompt that MSA cuts from position 8 on, every layer's
    /// output and the logits in the layout a reference engine's tensor
    /// dump uses (first and last three values of the first and last three
    /// rows), so the two can be read side by side.
    #[test]
    #[ignore]
    fn dump_the_synthetic_model_for_a_reference() {
        let path = std::path::PathBuf::from(
            std::env::var("ORANGU_MM3_SYNTHETIC").expect("set ORANGU_MM3_SYNTHETIC"),
        );
        write_synthetic(&path);
        let loaded = LoadedModel::open(&path).expect("open");
        let gguf = orangu::gguf::GgufFile::open(&path).expect("open gguf");
        let tokenizer = crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("tokenizer");
        let model = MinimaxM3Model::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("model");
        let text = "the quick brown fox jumps over a lazy dog";
        let tokens = tokenizer.encode(text, tokenizer.wants_bos());
        println!("DUMP tokens {} {tokens:?}", tokens.len());
        let n = tokens.len();
        let w = EMBD as usize;
        let show = |name: &str, v: &[f32], width: usize| {
            println!("DUMP {name}");
            let rows = v.len() / width;
            for r in (0..rows).filter(|&r| r < 3 || r + 3 >= rows) {
                let row = &v[r * width..(r + 1) * width];
                println!("DUMP   row {r}: {:?} .. {:?}", &row[..3], &row[width - 3..]);
            }
        };
        let mut cache = model.new_kv_cache(n);
        let mut hidden = model
            .run_layers_observed(
                &mut Seqs::One {
                    cache: &mut cache,
                    start_pos: 0,
                },
                &tokens,
                &mut |il, x| show(&format!("l_out-{il}"), x, w),
            )
            .expect("forward");
        let last = (n - 1) * w;
        let logits = model.project(&mut hidden[last..], 1);
        show("result_output (last token)", &logits, VOCAB as usize);
    }

    #[test]
    fn the_synthetic_model_has_a_dense_lead_and_sparse_layers() {
        let model = synthetic_model();
        let sparse: Vec<bool> = model.layers.iter().map(|l| l.indexer.is_some()).collect();
        assert_eq!(sparse, [false, true, true]);
        assert!(!model.msa.is_dense_at(8), "the test prompts must reach MSA");
    }

    /// A batch row must rotate at its own position and attend — and select
    /// blocks from — its own cache: the prompts (40 and 23 tokens) are well
    /// past the point where MSA cuts.
    #[test]
    fn a_decode_batch_matches_each_rows_solo_step() {
        let model = synthetic_model();
        super::super::batch_checks::decode_batch_matches_each_rows_solo_step(&model, TOLERANCE);
    }

    /// The shared check compares each row's chosen token; this compares
    /// every logit, so a row rotated at its neighbour's position — a
    /// change too small to move a random model's argmax — still fails.
    #[test]
    fn a_decode_batch_gives_each_rows_own_logits() {
        let model = synthetic_model();
        let prompts: [Vec<u32>; 2] = [
            (0..40u32).map(|i| 100 + (i * 37) % 2000).collect(),
            (0..23u32).map(|i| 300 + (i * 53) % 1700).collect(),
        ];
        let next = [7u32, 11];
        let mut solo = Vec::new();
        let mut caches: Vec<KvCache> = Vec::new();
        for (prompt, &tok) in prompts.iter().zip(&next) {
            let mut cache = model.new_kv_cache(64);
            model.forward(&mut cache, prompt, 0, 0).unwrap();
            let mut own = model.new_kv_cache(64);
            model.forward(&mut own, prompt, 0, 0).unwrap();
            solo.push(model.forward(&mut own, &[tok], prompt.len(), 0).unwrap());
            caches.push(cache);
        }
        let (a, b) = caches.split_at_mut(1);
        let mut rows = [
            super::super::DecodeRow {
                cache: &mut a[0],
                pos: prompts[0].len(),
                slot: 0,
            },
            super::super::DecodeRow {
                cache: &mut b[0],
                pos: prompts[1].len(),
                slot: 1,
            },
        ];
        let batched = model
            .forward_decode_batch(&mut rows, &next)
            .unwrap()
            .expect("the model batches");
        for (r, (got, want)) in batched.iter().zip(&solo).enumerate() {
            let worst = got
                .iter()
                .zip(want)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-4,
                "row {r} differs from its solo step by {worst}"
            );
        }
    }

    #[test]
    fn a_batched_verify_matches_its_own_pass() {
        let model = synthetic_model();
        super::super::batch_checks::decode_batch_verifies_a_draft_as_its_own_pass_would(
            &model, TOLERANCE,
        );
    }

    /// A prompt split across two passes gives the same logits as one pass:
    /// the second pass's MSA selects over keys the first one cached.
    #[test]
    fn a_chunked_prefill_matches_a_single_pass() {
        let model = synthetic_model();
        let prompt: Vec<u32> = (0..31u32).map(|i| 100 + (i * 53) % 2000).collect();
        let mut whole = model.new_kv_cache(64);
        let a = model.forward(&mut whole, &prompt, 0, 0).unwrap();
        let mut split = model.new_kv_cache(64);
        model.forward(&mut split, &prompt[..17], 0, 0).unwrap();
        let b = model.forward(&mut split, &prompt[17..], 17, 0).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "chunked prefill differs by {worst}");
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// End-to-end against a real `unsloth/MiniMax-M3.1-Flash-GGUF` file, on
    /// the predicted token: a wrong-but-plausible forward pass survives a
    /// logit tolerance, not a factual one-word answer.
    ///
    /// Run with `ORANGU_TEST_MINIMAX_M3_MODEL=/path/to/MiniMax-M3.1-Flash-UD-Q4_K_XL-00001-of-00007.gguf
    /// cargo test --release --bin orangu-server minimax_m3::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn minimax_m3_answers_paris_to_capital_of_france() {
        let path = std::env::var("ORANGU_TEST_MINIMAX_M3_MODEL")
            .expect("set ORANGU_TEST_MINIMAX_M3_MODEL");
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let tokenizer =
            crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let loaded = LoadedModel::open(std::path::Path::new(&path)).expect("load model");
        assert_eq!(loaded.config.architecture, "minimax-m3");
        let model = MinimaxM3Model::load_with_backend(
            &loaded,
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .expect("build model");
        assert_eq!(
            model.layers.iter().filter(|l| l.indexer.is_some()).count(),
            57
        );

        // A chat turn, not a raw completion: this is an agent model, and
        // its raw continuation of "The capital of France is" is a newline
        // and a list item in the reference engine as well. The turn is the
        // template's own framing with reasoning switched off — the
        // `</mm:think>` prefix it writes for a turn without thinking.
        let prompt = "]~!b[]~b]system\nYou are a helpful assistant.[e~[\n\
                      ]~b]user\nWhat is the capital of France? Answer in one word.[e~[\n\
                      ]~b]ai\n</mm:think>";
        let tokens = tokenizer.encode(prompt, tokenizer.wants_bos());
        let mut cache = model.new_kv_cache(tokens.len() + 1);
        let logits = model.forward(&mut cache, &tokens, 0, 0).expect("forward");
        let (top_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(
            tokenizer
                .decode(&[top_id as u32])
                .trim_start()
                .starts_with("Paris"),
            "top prediction was {:?}",
            tokenizer.decode(&[top_id as u32])
        );
    }
}

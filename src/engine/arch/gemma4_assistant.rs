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

//! The **gemma-4 assistant** draft head (`general.architecture =
//! gemma4-assistant`, the `MTP/mtp-*.gguf` beside a `gemma4` model): a few
//! narrow decoder blocks that predict the token after the one the served
//! model is about to read, from that token and the served model's state at
//! the position before it.
//!
//! - **Input**: the served model's own token embedding of the token, scaled
//!   by `sqrt(n_embd)` of the served model, beside the served model's
//!   post-output-norm hidden state; one projection (`nextn.pre_projection`)
//!   takes the pair down to the head's width.
//! - **Attention with queries only.** A block has no key or value
//!   projection: it attends over the served model's cache — a
//!   sliding-window block over the layer the served model's last
//!   sliding-window layer reads, a full block over the one its last full
//!   layer reads — at the served model's positions. Nothing is written
//!   anywhere, so every draft step sits at the same position, the one the
//!   served model is about to fill, and sees only what it has committed.
//! - **Blocks**: QK-norm on the query alone, RoPE as the served model's own
//!   layer of the same kind takes it (`rope_freqs` on full blocks), an
//!   attention scale of 1, post-attention and post-FFN norms, a GELU-gated
//!   FFN and a per-block output scale — gemma-4's block without its keys.
//! - **Output**: the output norm, then logits through the head's own
//!   (narrow) token embedding, and the next step's state through
//!   `nextn.post_projection` back up to the served model's width.
//!
//! Runs on the host: its weights are a few tens of megabytes and the rows it
//! attends over are the ones the served model's host verify path keeps
//! there. A row the device wrote and the host never received ends the draft
//! instead (`MtpHead::forward_served` returns `None`).

use anyhow::{Context, Result};

use super::{MtpHead, MtpStep};
use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;

/// One of the served model's layers a block can read: where its keys and
/// values are cached and how they are shaped.
#[derive(Debug, Clone, Copy)]
pub struct ServedLayer {
    /// The cache layer holding its rows (a shared layer's donor).
    pub cache_layer: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
}

/// What the head takes from the model it drafts for.
pub struct Served {
    /// `token_embd`, read a row at a time for the head's input.
    pub tok_embeddings: QuantMatrix,
    pub n_embd: usize,
    /// The layer the last sliding-window layer reads, and the one the last
    /// full-attention layer reads.
    pub swa: ServedLayer,
    pub full: ServedLayer,
}

struct Block {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    q_norm: Vec<f32>,
    wo: QuantMatrix,
    post_attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    post_ffw_norm: Vec<f32>,
    output_scale: f32,
    is_swa: bool,
    head_dim: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    reads: ServedLayer,
}

pub struct Gemma4Assistant {
    n_embd: usize,
    n_head: usize,
    n_swa: usize,
    eps: f32,
    pre_projection: QuantMatrix,
    post_projection: QuantMatrix,
    output_norm: Vec<f32>,
    output: QuantMatrix,
    rope_freqs: Option<Vec<f32>>,
    blocks: Vec<Block>,
    served_embeddings: QuantMatrix,
    served_n_embd: usize,
}

impl Gemma4Assistant {
    pub fn load(loaded: &LoadedModel, served: Served) -> Result<Self> {
        let config = &loaded.config;
        anyhow::ensure!(
            config.architecture == "gemma4-assistant",
            "a {} file is not a gemma-4 assistant head",
            config.architecture
        );
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let width_out = loaded
            .metadata_u64("embedding_length_out")
            .context("the head names no embedding_length_out")? as usize;
        anyhow::ensure!(
            width_out == served.n_embd,
            "the head reads {width_out}-wide states and the served model's are {}",
            served.n_embd
        );
        let n_layer = loaded
            .metadata_u64("nextn_predict_layers")
            .unwrap_or(config.n_layer as u64) as usize;
        let pattern: Vec<bool> = loaded
            .metadata_array_u64("attention.sliding_window_pattern")
            .map(|a| a.iter().map(|&v| v != 0).collect())
            .context("the head names no attention.sliding_window_pattern")?;
        anyhow::ensure!(
            pattern.len() >= n_layer,
            "the sliding-window pattern names {} blocks of {n_layer}",
            pattern.len()
        );
        let n_swa = loaded
            .metadata_u64("attention.sliding_window")
            .context("the head names no attention.sliding_window")? as usize;
        let rope_dim_full = loaded
            .metadata_u64("rope.dimension_count")
            .unwrap_or(config.head_dim as u64) as usize;
        let rope_dim_swa = loaded
            .metadata_u64("rope.dimension_count_swa")
            .unwrap_or(rope_dim_full as u64) as usize;
        let base_full = loaded.metadata_f32("rope.freq_base").unwrap_or(1_000_000.0);
        let base_swa = loaded
            .metadata_f32("rope.freq_base_swa")
            .unwrap_or(10_000.0);
        let tensor = |name: &str| -> Result<Vec<f32>> {
            loaded
                .tensor(name)
                .map(|(v, _)| v)
                .with_context(|| format!("loading {name}"))
        };
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(name)
                .with_context(|| format!("loading {name}"))
        };
        let mut blocks = Vec::with_capacity(n_layer);
        for (il, &is_swa) in pattern.iter().enumerate().take(n_layer) {
            let wq = matrix(&format!("blk.{il}.attn_q.weight"))?;
            let head_dim = wq.out_dim / n_head;
            let reads = if is_swa { served.swa } else { served.full };
            anyhow::ensure!(
                reads.head_dim == head_dim,
                "block {il} has {head_dim}-wide heads and the served layer it reads has {}",
                reads.head_dim
            );
            anyhow::ensure!(
                n_head.is_multiple_of(reads.n_head_kv),
                "block {il}'s {n_head} query heads do not group over {} key heads",
                reads.n_head_kv
            );
            let output_scale = tensor(&format!("blk.{il}.layer_output_scale.weight"))?
                .first()
                .copied()
                .unwrap_or(1.0);
            blocks.push(Block {
                attn_norm: tensor(&format!("blk.{il}.attn_norm.weight"))?,
                wq,
                q_norm: tensor(&format!("blk.{il}.attn_q_norm.weight"))?,
                wo: matrix(&format!("blk.{il}.attn_output.weight"))?,
                post_attention_norm: tensor(&format!("blk.{il}.post_attention_norm.weight"))?,
                ffn_norm: tensor(&format!("blk.{il}.ffn_norm.weight"))?,
                ffn_gate: matrix(&format!("blk.{il}.ffn_gate.weight"))?,
                ffn_up: matrix(&format!("blk.{il}.ffn_up.weight"))?,
                ffn_down: matrix(&format!("blk.{il}.ffn_down.weight"))?,
                post_ffw_norm: tensor(&format!("blk.{il}.post_ffw_norm.weight"))?,
                output_scale,
                is_swa,
                head_dim,
                rope_dim: if is_swa { rope_dim_swa } else { rope_dim_full },
                rope_freq_base: if is_swa { base_swa } else { base_full },
                reads,
            });
        }
        let pre_projection = matrix("nextn.pre_projection.weight")?;
        anyhow::ensure!(
            pre_projection.in_dim == 2 * served.n_embd && pre_projection.out_dim == n_embd,
            "nextn.pre_projection is {}→{}, expected {}→{n_embd}",
            pre_projection.in_dim,
            pre_projection.out_dim,
            2 * served.n_embd
        );
        Ok(Self {
            n_embd,
            n_head,
            n_swa,
            eps: config.rms_eps,
            pre_projection,
            post_projection: matrix("nextn.post_projection.weight")?,
            output_norm: tensor("output_norm.weight")?,
            output: matrix("token_embd.weight")?,
            rope_freqs: loaded.tensor("rope_freqs.weight").ok().map(|(v, _)| v),
            blocks,
            served_n_embd: served.n_embd,
            served_embeddings: served.tok_embeddings,
        })
    }
}

impl MtpHead for Gemma4Assistant {
    fn state_width(&self) -> usize {
        self.served_n_embd
    }

    fn new_kv_cache(&self, _capacity: usize) -> KvCache {
        KvCache::new_with_dims(1, &[])
    }

    fn forward(
        &self,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _states: &[f32],
        _start_pos: usize,
        _first_pos: usize,
    ) -> Result<MtpStep> {
        anyhow::bail!("the gemma-4 assistant reads the served model's cache, not one of its own")
    }

    fn reads_served_cache(&self) -> bool {
        true
    }

    fn forward_served(
        &self,
        served: &KvCache,
        token: u32,
        state: &[f32],
        pos: usize,
    ) -> Result<Option<MtpStep>> {
        anyhow::ensure!(
            state.len() == self.served_n_embd,
            "a {}-wide state for a head that reads {}",
            state.len(),
            self.served_n_embd
        );
        if pos == 0 {
            return Ok(None);
        }
        // Every row below `pos`, on the host, for each layer a block reads.
        for block in &self.blocks {
            let layer = served
                .layers
                .get(block.reads.cache_layer)
                .context("the served cache has no layer the head reads")?;
            if layer.len < pos || layer.has_device_only_rows() {
                return Ok(None);
            }
        }
        let cpu = CpuBackend;
        let mut xh = self.served_embeddings.row(token as usize);
        let scale = (self.served_n_embd as f32).sqrt();
        xh.iter_mut().for_each(|v| *v *= scale);
        xh.extend_from_slice(state);
        let mut x = cpu.matmul(&xh, 1, &self.pre_projection);
        let n_embd = self.n_embd;
        let eps = self.eps;
        let mut attn = Vec::new();
        for block in &self.blocks {
            let mut normed = x.clone();
            tensor::rmsnorm_inplace(&mut normed, &block.attn_norm, 1, n_embd, eps);
            let mut q = cpu.matmul(&normed, 1, &block.wq);
            tensor::rmsnorm_inplace(&mut q, &block.q_norm, self.n_head, block.head_dim, eps);
            let freq_factors = (!block.is_swa)
                .then_some(self.rope_freqs.as_deref())
                .flatten();
            tensor::rope_apply_scaled_inplace(
                &mut q,
                self.n_head,
                block.head_dim,
                block.rope_dim,
                pos,
                block.rope_freq_base,
                freq_factors,
            );
            attn.clear();
            attn.resize(self.n_head * block.head_dim, 0.0);
            let first = if block.is_swa {
                pos.saturating_sub(self.n_swa - 1)
            } else {
                0
            };
            crate::engine::attention::multi_head_attention(
                &mut attn,
                &q,
                &served.layers[block.reads.cache_layer],
                self.n_head,
                self.n_head / block.reads.n_head_kv,
                block.head_dim,
                1.0,
                |_| (first, pos - 1),
            );
            let mut cur = cpu.matmul(&attn, 1, &block.wo);
            tensor::rmsnorm_inplace(&mut cur, &block.post_attention_norm, 1, n_embd, eps);
            cur.iter_mut().zip(&x).for_each(|(c, r)| *c += r);
            let attn_out = cur;
            let mut ffn_in = attn_out.clone();
            tensor::rmsnorm_inplace(&mut ffn_in, &block.ffn_norm, 1, n_embd, eps);
            let mut gate = cpu.matmul(&ffn_in, 1, &block.ffn_gate);
            let up = cpu.matmul(&ffn_in, 1, &block.ffn_up);
            tensor::gelu_inplace(&mut gate);
            gate.iter_mut().zip(&up).for_each(|(g, u)| *g *= u);
            let mut ffn = cpu.matmul(&gate, 1, &block.ffn_down);
            tensor::rmsnorm_inplace(&mut ffn, &block.post_ffw_norm, 1, n_embd, eps);
            x = ffn
                .iter()
                .zip(&attn_out)
                .map(|(f, a)| (f + a) * block.output_scale)
                .collect();
        }
        tensor::rmsnorm_inplace(&mut x, &self.output_norm, 1, n_embd, eps);
        let logits = cpu.matmul(&x, 1, &self.output);
        let state = cpu.matmul(&x, 1, &self.post_projection);
        Ok(Some(MtpStep { logits, state }))
    }
}

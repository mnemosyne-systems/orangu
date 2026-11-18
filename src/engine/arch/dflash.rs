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

//! DFlash / DSpark (`general.architecture = "dflash"`): a **block-diffusion
//! draft** that reads its target model's hidden states.
//!
//! These are not standalone models: a checkpoint carries no `token_embd`
//! and no `output` tensor. It drafts through the target's own embedding
//! table and LM head, from the residual stream the target carried *into*
//! the layers `dflash.target_layers` names.
//!
//! Two ways a file reaches the server:
//!
//! - **Named as the model.** Serving it means serving its target:
//!   `main::auto_pair_dflash_target` resolves the paired base model from the
//!   same Hugging Face repo and serves that. [`DFlashModel`] is the refusal
//!   for a file handed over with no target to pair it to.
//! - **Named as `draft_model`.** The file drafts for the served model as an
//!   [`MtpHead`] ([`DFlashHead`]), when the served architecture exports the
//!   layer inputs it reads (`ModelForward::load_mtp_head`).
//!
//! **How a draft is made.**
//!
//! - *Context.* For every position the target commits, the inputs of its
//!   tapped layers, concatenated (`[n_taps * n_embd]`), go through `fc` and
//!   the encoder norm; each block's key and value projections of that row
//!   (QK-normed key, NEOX rope at the position) become the block's cached
//!   row for the position. Nothing else of the context is kept.
//! - *Block.* `[anchor, mask, …, mask]` at the anchor's position onwards —
//!   the anchor is the token the target is about to read — embedded through
//!   the target's table and scaled by `dflash.embedding_scale`, through the
//!   blocks with **non-causal** attention over the cached context and the
//!   whole block, then the output norm and the target's LM head. The argmax
//!   at each mask is that position's draft, all of them from one pass.
//!
//! Blocks are pre-norm, a SiLU-gated FFN, with a sliding window on the
//! blocks `dflash.attention.sliding_window_pattern` marks. The DFlash2
//! convolution and selector and the DSpark Markov head are not run: a file
//! that carries them is refused.
//!
//! The blocks run on the host: a few hundred megabytes, read once per draft
//! for every row of the block together. The LM head runs where the served
//! model keeps it.

use anyhow::{Context, Result};

use super::{ModelForward, MtpHead, MtpStep};
use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use crate::engine::tensor;

pub struct DFlashModel {
    config: ModelConfig,
}

fn standalone_error(loaded: Option<&LoadedModel>) -> anyhow::Error {
    let targets = loaded
        .and_then(|l| l.metadata_array_u64("target_layers"))
        .map(|layers| {
            layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "unknown".to_string());
    anyhow::anyhow!(
        "this is a dflash draft sidecar, not a servable model: it carries no token embeddings \
         and no output projection, and reads its target model's hidden states (layers {targets}). \
         Serve the paired target model instead — selecting a draft from a Hugging Face cache \
         directory does that automatically"
    )
}

impl DFlashModel {
    pub fn load_with_backend(loaded: &LoadedModel) -> Result<Self> {
        Err(standalone_error(Some(loaded)))
    }
}

impl ModelForward for DFlashModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let _ = capacity;
        KvCache::new_with_dims(0, &[])
    }

    fn forward(
        &self,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        Err(standalone_error(None))
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        Err(standalone_error(None))
    }
}

/// What the head takes from the model it drafts for.
pub struct Target {
    /// `token_embd`, read a row at a time for the block's input.
    pub tok_embeddings: QuantMatrix,
    /// The LM head, on the backend that holds it: the widest matrix a
    /// draft reads, and the one the served model keeps on its device.
    pub output: QuantMatrix,
    pub backend: std::sync::Arc<dyn Backend>,
    pub n_embd: usize,
    pub n_layer: usize,
}

struct Block {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: QuantMatrix,
    ffn_up: QuantMatrix,
    ffn_down: QuantMatrix,
    is_swa: bool,
}

pub struct DFlashHead {
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    eps: f32,
    rope_freq_base: f32,
    n_swa: usize,
    embedding_scale: f32,
    block_size: usize,
    mask_token: u32,
    target_layers: Vec<usize>,
    fc: QuantMatrix,
    enc_norm: Vec<f32>,
    output_norm: Vec<f32>,
    blocks: Vec<Block>,
    target: Target,
}

impl DFlashHead {
    pub fn load(loaded: &LoadedModel, target: Target) -> Result<Self> {
        let config = &loaded.config;
        anyhow::ensure!(
            config.architecture == "dflash",
            "a {} file is not a dflash draft",
            config.architecture
        );
        for unsupported in [
            "markov_w1.weight",
            "selector_hidden.weight",
            "blk.0.attn_conv_proj.weight",
            "blk.0.hc_attn_fn.weight",
        ] {
            anyhow::ensure!(
                loaded.matrix(unsupported).is_err(),
                "this dflash draft carries {unsupported}, a head variant this engine does not run"
            );
        }
        let n_embd = config.n_embd;
        anyhow::ensure!(
            n_embd == target.n_embd,
            "the draft is {n_embd} wide and the served model {}",
            target.n_embd
        );
        let target_layers: Vec<usize> = loaded
            .metadata_array_u64("target_layers")
            .context("the draft names no target_layers")?
            .into_iter()
            .map(|l| l as usize)
            .collect();
        anyhow::ensure!(
            !target_layers.is_empty() && target_layers.iter().all(|&l| l < target.n_layer),
            "the draft reads layers {target_layers:?} of a {}-layer model",
            target.n_layer
        );
        let n_layer = config.n_layer;
        let pattern: Vec<bool> = loaded
            .metadata_array_u64("attention.sliding_window_pattern")
            .map(|a| a.iter().map(|&v| v != 0).collect())
            .unwrap_or_else(|| vec![false; n_layer]);
        anyhow::ensure!(
            pattern.len() >= n_layer,
            "the sliding-window pattern names {} blocks of {n_layer}",
            pattern.len()
        );
        let n_swa = loaded.metadata_u64("attention.sliding_window").unwrap_or(0) as usize;
        let mask_token = loaded
            .metadata
            .iter()
            .find(|(k, _)| *k == "tokenizer.ggml.mask_token_id")
            .and_then(|(_, v)| v.as_u64())
            .context("the draft names no tokenizer.ggml.mask_token_id")?
            as u32;
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
            blocks.push(Block {
                attn_norm: tensor(&format!("blk.{il}.attn_norm.weight"))?,
                wq: matrix(&format!("blk.{il}.attn_q.weight"))?,
                wk: matrix(&format!("blk.{il}.attn_k.weight"))?,
                wv: matrix(&format!("blk.{il}.attn_v.weight"))
                    .context("a dflash draft whose blocks share keys and values")?,
                wo: matrix(&format!("blk.{il}.attn_output.weight"))?,
                q_norm: tensor(&format!("blk.{il}.attn_q_norm.weight"))?,
                k_norm: tensor(&format!("blk.{il}.attn_k_norm.weight"))?,
                ffn_norm: tensor(&format!("blk.{il}.ffn_norm.weight"))?,
                ffn_gate: matrix(&format!("blk.{il}.ffn_gate.weight"))?,
                ffn_up: matrix(&format!("blk.{il}.ffn_up.weight"))?,
                ffn_down: matrix(&format!("blk.{il}.ffn_down.weight"))?,
                is_swa: is_swa && n_swa > 0,
            });
        }
        let fc = matrix("fc.weight")?;
        anyhow::ensure!(
            fc.in_dim == target_layers.len() * n_embd && fc.out_dim == n_embd,
            "fc is {}→{}, expected {}→{n_embd}",
            fc.in_dim,
            fc.out_dim,
            target_layers.len() * n_embd
        );
        let head_dim = config.head_dim;
        anyhow::ensure!(
            blocks[0].wq.out_dim == config.n_head * head_dim
                && blocks[0].wk.out_dim == config.n_head_kv * head_dim,
            "the draft's projections do not match its {} heads of {head_dim}",
            config.n_head
        );
        Ok(Self {
            n_embd,
            n_head: config.n_head,
            n_head_kv: config.n_head_kv,
            head_dim,
            eps: config.rms_eps,
            rope_freq_base: config.rope_freq_base,
            n_swa,
            embedding_scale: loaded
                .metadata_f32("embedding_scale")
                .unwrap_or((n_embd as f32).sqrt()),
            block_size: loaded.metadata_u64("block_size").unwrap_or(16) as usize,
            mask_token,
            target_layers,
            fc,
            enc_norm: tensor("enc.output_norm.weight")?,
            output_norm: tensor("output_norm.weight")?,
            blocks,
            target,
        })
    }

    /// The served model's layers whose inputs this head reads, in the order
    /// a state row concatenates them.
    pub fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// Per-head RMS norm, then NEOX rope at `positions[r]`, of `rows`
    /// `[positions.len(), n_heads * head_dim]`.
    fn norm_and_rope(&self, rows: &mut [f32], norm: &[f32], n_heads: usize, positions: &[usize]) {
        let width = n_heads * self.head_dim;
        tensor::rmsnorm_inplace(
            rows,
            norm,
            positions.len() * n_heads,
            self.head_dim,
            self.eps,
        );
        for (row, &pos) in rows.chunks_exact_mut(width).zip(positions) {
            tensor::rope_apply_scaled_inplace(
                row,
                n_heads,
                self.head_dim,
                self.head_dim,
                pos,
                self.rope_freq_base,
                None,
            );
        }
    }

    /// The window block query `t` (at `pos + t`) attends to: every cached
    /// context row from `first_real` and every row of the block, cut to the
    /// sliding window on a block that has one.
    fn window(
        &self,
        block: &Block,
        first_real: usize,
        pos: usize,
        t: usize,
        n_rows: usize,
    ) -> (usize, usize) {
        let last = pos + n_rows - 1;
        let first = if block.is_swa {
            (pos + t).saturating_sub(self.n_swa - 1).max(first_real)
        } else {
            first_real
        };
        (first, last)
    }
}

impl MtpHead for DFlashHead {
    fn state_width(&self) -> usize {
        self.target_layers.len() * self.target.n_embd
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        KvCache::new_with_dims(capacity, &vec![self.kv_dim(); self.blocks.len()])
    }

    /// Caches the context rows: `states[i]` is the served model's tapped
    /// layer inputs at `start_pos + i - 1`, the position before the token it
    /// is paired with, which is the row a draft from that token reads last.
    /// A zero state at position 0 stands for nothing and is skipped; rows
    /// below a reused prefix are held as placeholders, which no window
    /// reaches (`first_pos`).
    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        states: &[f32],
        start_pos: usize,
        _first_pos: usize,
    ) -> Result<MtpStep> {
        let width = self.state_width();
        anyhow::ensure!(
            states.len() == tokens.len() * width,
            "{} state values for {} tokens of {width}",
            states.len(),
            tokens.len()
        );
        let skip = usize::from(start_pos == 0);
        let rows = &states[skip * width..];
        let n_rows = rows.len() / width;
        if n_rows == 0 {
            return Ok(MtpStep {
                logits: Vec::new(),
                state: Vec::new(),
            });
        }
        let first = start_pos + skip - 1;
        let positions: Vec<usize> = (first..first + n_rows).collect();
        let cpu = CpuBackend;
        let mut fused = cpu.matmul(rows, n_rows, &self.fc);
        tensor::rmsnorm_inplace(&mut fused, &self.enc_norm, n_rows, self.n_embd, self.eps);
        let kv_dim = self.kv_dim();
        for (block, layer) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            let mut k = cpu.matmul(&fused, n_rows, &block.wk);
            self.norm_and_rope(&mut k, &block.k_norm, self.n_head_kv, &positions);
            let v = cpu.matmul(&fused, n_rows, &block.wv);
            layer.truncate(first);
            let zero = vec![0f32; kv_dim];
            while layer.len < first {
                layer.push(&zero, &zero);
            }
            for r in 0..n_rows {
                layer.push(
                    &k[r * kv_dim..(r + 1) * kv_dim],
                    &v[r * kv_dim..(r + 1) * kv_dim],
                );
            }
        }
        Ok(MtpStep {
            logits: Vec::new(),
            state: Vec::new(),
        })
    }

    fn block_size(&self) -> Option<usize> {
        Some(self.block_size)
    }

    fn draft_block(
        &self,
        cache: &mut KvCache,
        anchor: u32,
        pos: usize,
        n_draft: usize,
        first_pos: usize,
    ) -> Result<Vec<u32>> {
        let n_draft = n_draft.min(self.block_size.saturating_sub(1));
        // The row before the anchor is the one the block reads last; a
        // context that stops short of it has nothing current to draft from.
        if n_draft == 0 || pos == 0 || cache.layers.iter().any(|l| l.len < pos) {
            return Ok(Vec::new());
        }
        let first_real = first_pos.saturating_sub(1);
        let n_rows = n_draft + 1;
        let n_embd = self.n_embd;
        let positions: Vec<usize> = (pos..pos + n_rows).collect();
        let mut x = Vec::with_capacity(n_rows * n_embd);
        for t in 0..n_rows {
            let token = if t == 0 { anchor } else { self.mask_token };
            x.extend(
                self.target
                    .tok_embeddings
                    .row(token as usize)
                    .iter()
                    .map(|v| v * self.embedding_scale),
            );
        }
        let cpu = CpuBackend;
        let q_width = self.n_head * self.head_dim;
        let kv_dim = self.kv_dim();
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn = vec![0f32; n_rows * q_width];
        for (block, layer) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            let mut h = x.clone();
            tensor::rmsnorm_inplace(&mut h, &block.attn_norm, n_rows, n_embd, self.eps);
            let mut q = cpu.matmul(&h, n_rows, &block.wq);
            self.norm_and_rope(&mut q, &block.q_norm, self.n_head, &positions);
            let mut k = cpu.matmul(&h, n_rows, &block.wk);
            self.norm_and_rope(&mut k, &block.k_norm, self.n_head_kv, &positions);
            let v = cpu.matmul(&h, n_rows, &block.wv);
            // The block's own rows sit in the cache for the one attention
            // that reads them, and come off again: the next draft starts
            // from wherever the served model accepted to.
            layer.truncate(pos);
            for r in 0..n_rows {
                layer.push(
                    &k[r * kv_dim..(r + 1) * kv_dim],
                    &v[r * kv_dim..(r + 1) * kv_dim],
                );
            }
            attn.iter_mut().for_each(|a| *a = 0.0);
            crate::engine::attention::multi_head_attention(
                &mut attn,
                &q,
                layer,
                self.n_head,
                self.n_head / self.n_head_kv,
                self.head_dim,
                scale,
                |t| self.window(block, first_real, pos, t, n_rows),
            );
            layer.truncate(pos);
            let o = cpu.matmul(&attn, n_rows, &block.wo);
            x.iter_mut().zip(&o).for_each(|(r, a)| *r += a);
            let mut h = x.clone();
            tensor::rmsnorm_inplace(&mut h, &block.ffn_norm, n_rows, n_embd, self.eps);
            let mut gate = cpu.matmul(&h, n_rows, &block.ffn_gate);
            let up = cpu.matmul(&h, n_rows, &block.ffn_up);
            gate.iter_mut()
                .zip(&up)
                .for_each(|(g, u)| *g = tensor::silu(*g) * u);
            let down = cpu.matmul(&gate, n_rows, &block.ffn_down);
            x.iter_mut().zip(&down).for_each(|(r, d)| *r += d);
        }
        // The anchor's own row predicts nothing the target has not.
        let mut masks = x[n_embd..].to_vec();
        tensor::rmsnorm_inplace(&mut masks, &self.output_norm, n_draft, n_embd, self.eps);
        let logits = self
            .target
            .backend
            .matmul(&masks, n_draft, &self.target.output);
        let n_vocab = logits.len() / n_draft;
        Ok(logits
            .chunks_exact(n_vocab)
            .map(crate::engine::sampling::argmax)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::standalone_error;

    /// The message has to say what to do about it, not just that it
    /// failed: the only way a user reaches it is by naming a draft file
    /// directly, and the fix is to name the target.
    #[test]
    fn the_standalone_error_points_at_the_paired_target_model() {
        let message = standalone_error(None).to_string();
        assert!(message.contains("dflash draft sidecar"), "{message}");
        assert!(message.contains("paired target model"), "{message}");
    }
}

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

//! How a prompt becomes what LTX-2's transformer attends to.
//!
//! 1. **Gemma 4**, LTX's own 12B checkpoint, reads the prompt — `<bos>`
//!    and the prompt's tokens — and *every* layer's hidden state is kept:
//!    the embeddings, each decoder layer's output, and the last one after
//!    the final norm, `L + 1` of them (`output_hidden_states`).
//! 2. **The feature extractor** (`FeatureExtractorV2`) RMS-normalizes each
//!    token's state at each layer on its own, lays a token's layers out
//!    side by side (`[width · (L + 1)]`, the layer index fastest), and two
//!    linear heads — one for video, one for audio, both shipped in the
//!    encoder's file — project that to each stream's width, the input
//!    scaled by `sqrt(head width / width)` first.
//! 3. **The connectors** (`Embeddings1DConnector`, in the transformer's
//!    file) pad the prompt to 1024 tokens with *learnable registers* in
//!    place of the padding, run eight bidirectional transformer blocks over
//!    them with a 1-D rotary position, and RMS-normalize the result. All
//!    1024 rows are what the transformer cross-attends to.
//!
//! The reference tokenizes to 1024 positions left-padded and runs Gemma over
//! the padding masked out; the real tokens see only each other, causally,
//! and rotary attention depends on relative position alone, so running Gemma
//! over the real tokens by themselves gives the same states — and the
//! padding's own rows are discarded by the connector anyway.

use super::ops::{Attention, FeedForward, Linear, Rope, rms_norm_rows};
use crate::engine::arch::ModelForward;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::sync::Arc;

/// The prompt length the connectors run at (`TOKENIZER_MAX_LENGTH`).
pub const PROMPT_TOKENS: usize = 1024;

/// One of the two `Embeddings1DConnector`s.
pub struct Connector {
    blocks: Vec<(Attention, FeedForward)>,
    /// `[registers, dim]`, tiled over the padding.
    registers: Vec<f32>,
    dim: usize,
    heads: usize,
    max_pos: f32,
    theta: f64,
    eps: f32,
}

impl Connector {
    pub fn load(
        loaded: &LoadedModel,
        prefix: &str,
        heads: usize,
        max_pos: f32,
        theta: f64,
        eps: f32,
    ) -> Result<Self> {
        let mut blocks = Vec::new();
        while loaded.has_tensor(&format!(
            "{prefix}.transformer_1d_blocks.{}.attn1.to_q.weight",
            blocks.len()
        )) {
            let p = format!("{prefix}.transformer_1d_blocks.{}", blocks.len());
            blocks.push((
                Attention::load(loaded, &format!("{p}.attn1"), heads)?,
                FeedForward::load(loaded, &format!("{p}.ff"))?,
            ));
        }
        ensure!(!blocks.is_empty(), "ltxv: {prefix} has no blocks");
        let dim = blocks[0].0.inner();
        let (registers, _) = loaded.tensor(&format!("{prefix}.learnable_registers"))?;
        ensure!(
            !registers.is_empty() && registers.len().is_multiple_of(dim),
            "ltxv: {prefix}.learnable_registers is not rows of {dim}"
        );
        Ok(Self {
            blocks,
            registers,
            dim,
            heads,
            max_pos,
            theta,
            eps,
        })
    }

    /// `features` (`[n, dim]`, the prompt's real tokens) padded to
    /// [`PROMPT_TOKENS`] with the registers, through the blocks —
    /// `[PROMPT_TOKENS, dim]`.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        features: &[f32],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<f32>> {
        let (dim, total) = (self.dim, PROMPT_TOKENS);
        let n = features.len() / dim;
        ensure!(n <= total, "the prompt is longer than {total} tokens");
        let n_registers = self.registers.len() / dim;
        ensure!(
            total.is_multiple_of(n_registers),
            "ltxv: {n_registers} registers do not tile {total} positions"
        );
        let mut h = Vec::with_capacity(total * dim);
        h.extend_from_slice(features);
        for pos in n..total {
            let r = pos % n_registers;
            h.extend_from_slice(&self.registers[r * dim..(r + 1) * dim]);
        }
        let positions: Vec<f32> = (0..total).map(|p| p as f32).collect();
        let pe = Rope::new(&positions, 1, &[self.max_pos], dim, self.heads, self.theta);
        for (attn, ff) in &self.blocks {
            if super::cancelled(cancel) {
                anyhow::bail!("cancelled");
            }
            let normed = rms_norm_rows(&h, dim, self.eps);
            let y = attn.self_attend(backend, &normed, total, Some(&pe), self.eps);
            add(&mut h, &y);
            let normed = rms_norm_rows(&h, dim, self.eps);
            let y = ff.forward(backend, &normed, total);
            add(&mut h, &y);
        }
        Ok(rms_norm_rows(&h, dim, self.eps))
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    x.par_iter_mut()
        .zip(y.par_iter())
        .for_each(|(a, b)| *a += b);
}

/// Every hidden state of `tokens` through `model`: the embeddings, each
/// layer's output, the last one after the final norm — `[L + 1][n · width]`.
pub fn all_hidden_states(
    model: &dyn ModelForward,
    tokens: &[u32],
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<Vec<f32>>> {
    let n_layer = model.n_trunk_layer();
    let width = model.config().n_embd;
    let mut cache = model.new_kv_cache(tokens.len().max(1));
    let mut states = Vec::with_capacity(n_layer + 1);
    let mut h = model.embed(tokens)?;
    ensure!(
        h.len() == tokens.len() * width,
        "the text encoder's embeddings are not {width} wide"
    );
    states.push(h.clone());
    for layer in 0..n_layer {
        if super::cancelled(cancel) {
            anyhow::bail!("cancelled");
        }
        h = model.forward_layers(&mut cache, h, tokens, layer..layer + 1, 0)?;
        states.push(h.clone());
    }
    let last = states.last_mut().expect("at least the embeddings");
    model.final_norm(last, tokens.len())?;
    Ok(states)
}

/// The two projection heads of `FeatureExtractorV2`, from the encoder's
/// file.
pub struct FeatureHeads {
    video: Linear,
    audio: Linear,
    /// The encoder's width.
    width: usize,
    /// Hidden states per token: `L + 1`.
    layers: usize,
}

impl FeatureHeads {
    /// The heads under `text_embedding_projection.` (or the file's own
    /// prefix for them), checked against the encoder's shape.
    pub fn load(loaded: &LoadedModel, width: usize, layers: usize) -> Result<Self> {
        let name = |head: &str| -> Result<String> {
            loaded
                .tensor_directory()
                .into_iter()
                .map(|(name, _, _)| name)
                .find(|name| name.ends_with(&format!("{head}.weight")))
                .map(|name| name.trim_end_matches(".weight").to_string())
                .ok_or_else(|| anyhow::anyhow!("the text encoder has no {head} projection"))
        };
        let video = Linear::load(loaded, &name("video_aggregate_embed")?)?;
        let audio = Linear::load(loaded, &name("audio_aggregate_embed")?)?;
        ensure!(
            video.in_dim() == width * layers && audio.in_dim() == width * layers,
            "the text encoder's projections read {} values a token, and its {layers} hidden \
             states of {width} are {}",
            video.in_dim(),
            width * layers
        );
        Ok(Self {
            video,
            audio,
            width,
            layers,
        })
    }

    pub fn video_dim(&self) -> usize {
        self.video.out_dim()
    }

    pub fn audio_dim(&self) -> usize {
        self.audio.out_dim()
    }

    /// `(video, audio)` features of the prompt's `n` tokens from its hidden
    /// states (see [`all_hidden_states`]).
    pub fn features(
        &self,
        backend: &dyn Backend,
        states: &[Vec<f32>],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (width, layers) = (self.width, self.layers);
        ensure!(
            states.len() == layers && states.iter().all(|s| s.len() == n * width),
            "expected {layers} hidden states of {n} × {width}"
        );
        // Per token and layer: x / sqrt(mean(x²) + 1e-6), interleaved so
        // that column `d · layers + l` is layer `l`'s dimension `d`.
        let flat = width * layers;
        let mut normed = vec![0.0f32; n * flat];
        normed
            .par_chunks_mut(flat)
            .enumerate()
            .for_each(|(t, row)| {
                for (l, state) in states.iter().enumerate() {
                    let x = &state[t * width..(t + 1) * width];
                    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / width as f32;
                    let scale = 1.0 / (mean_sq + 1e-6).sqrt();
                    for (d, v) in x.iter().enumerate() {
                        row[d * layers + l] = v * scale;
                    }
                }
            });
        let scaled = |head: &Linear| -> Vec<f32> {
            let factor = (head.out_dim() as f32 / width as f32).sqrt();
            let input: Vec<f32> = normed.par_iter().map(|v| v * factor).collect();
            head.forward(backend, &input, n)
        };
        Ok((scaled(&self.video), scaled(&self.audio)))
    }
}

/// The prompt as the transformer reads it: `[PROMPT_TOKENS, dim]` per
/// stream.
pub struct Conditioning {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    /// The prompt's own tokens, `<bos>` included.
    pub tokens: usize,
}

/// The whole path from text to [`Conditioning`].
pub struct PromptEncoder {
    pub model: Arc<dyn ModelForward>,
    pub tokenizer: Arc<crate::engine::tokenizer::Tokenizer>,
    pub heads: FeatureHeads,
    pub video: Connector,
    pub audio: Connector,
}

impl PromptEncoder {
    /// `<bos>` and the prompt's tokens, at most [`PROMPT_TOKENS`].
    pub fn tokens(&self, prompt: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(prompt.trim(), true);
        tokens.truncate(PROMPT_TOKENS);
        tokens
    }

    pub fn encode(
        &self,
        backend: &dyn Backend,
        prompt: &str,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Conditioning> {
        let tokens = self.tokens(prompt);
        ensure!(!tokens.is_empty(), "the prompt encoded to no tokens");
        let states = all_hidden_states(&*self.model, &tokens, cancel)?;
        let (video, audio) = self.heads.features(backend, &states, tokens.len())?;
        drop(states);
        Ok(Conditioning {
            video: self.video.forward(backend, &video, cancel)?,
            audio: self.audio.forward(backend, &audio, cancel)?,
            tokens: tokens.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_tokens_layers_are_interleaved_with_the_layer_fastest() {
        // reshape(B, T, D * L) of a [B, T, D, L] stack: column d · L + l.
        let (d, l) = (3usize, 2usize);
        let col = |dd: usize, ll: usize| dd * l + ll;
        assert_eq!(col(0, 1), 1);
        assert_eq!(col(1, 0), 2);
        assert_eq!(col(2, 1), 5);
        assert_eq!(d * l, 6);
    }
}

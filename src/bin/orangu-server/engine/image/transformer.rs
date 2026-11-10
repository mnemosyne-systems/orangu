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

//! The `qwen_image` diffusion transformer — diffusers'
//! `QwenImageTransformer2DModel`, read from the GGUF the ComfyUI-GGUF
//! tooling writes (the `unsloth/Qwen-Image-*-GGUF` files).
//!
//! It is not a language model and shares nothing with `ModelForward`: there
//! are no tokens, no vocabulary and no KV cache. What it computes is one
//! **velocity** — for a latent image at noise level `sigma`, the direction
//! towards the clean image — conditioned on a prompt's hidden states from a
//! Qwen2.5-VL text encoder (`engine::image`'s pipeline runs that encoder
//! through the ordinary `arch::llama` path; this module only ever sees its
//! output).
//!
//! The graph is a *dual-stream* transformer (MMDiT): every block carries an
//! image stream and a text stream through their own projections and
//! feed-forwards, joined only in attention, where the two are concatenated
//! (text first) and attend to each other without a mask. Each stream is
//! modulated by the timestep — `x * (1 + scale) + shift` after a weightless
//! LayerNorm, and a `gate` on the residual — with six such vectors per
//! stream per block produced by one linear from the timestep embedding.
//! Position enters through a three-axis rotary embedding: image tokens
//! rotate by their (frame, row, column) on 8 + 28 + 28 complex pairs of the
//! 128-wide head, rows and columns *centred* on the image; text tokens sit
//! past the image on all three axes at once.
//!
//! Tensor names are diffusers' own (`transformer_blocks.N.attn.to_q`,
//! `img_mlp.net.0.proj`, ...): the GGUF converter keeps them, and the file
//! carries no hyperparameters at all — `general.architecture` and two
//! quantization keys are its whole metadata — so every dimension here is
//! read off a tensor's shape (see `engine::loader`'s `qwen_image` branch).
//!
//! Every projection goes through `Backend::matmul`, the same seam every
//! language model uses, so the transformer runs on whichever device the
//! server selected; the norms, modulation, rotation and attention are host
//! code over `f32` rows.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Dimensions read off the checkpoint's tensors.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Width of both streams (`3072`).
    pub dim: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    /// Width of the text encoder's hidden states (`3584` for Qwen2.5-VL-7B).
    pub txt_dim: usize,
    /// One image token's features: `16 latent channels * 2 * 2 patch`.
    pub in_channels: usize,
    /// Complex pairs of the head rotated by each of frame, row and column —
    /// diffusers' `axes_dims_rope = (16, 56, 56)`, halved.
    pub rope_axes: [usize; 3],
    pub eps: f32,
}

struct Linear {
    w: QuantMatrix,
    bias: Option<Vec<f32>>,
    /// The low-rank adapter riding on this linear, when one was loaded —
    /// see [`super::lora`].
    lora: Option<super::lora::LoraPair>,
}

struct Block {
    img_mod: Linear,
    txt_mod: Linear,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Linear,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm_added_q: Vec<f32>,
    norm_added_k: Vec<f32>,
    img_mlp_in: Linear,
    img_mlp_out: Linear,
    txt_mlp_in: Linear,
    txt_mlp_out: Linear,
}

pub struct QwenImageTransformer {
    backend: Arc<dyn Backend>,
    pub config: TransformerConfig,
    /// Where the passes since the last [`take_stages`](Self::take_stages)
    /// spent their time. A profile cannot say: the matmuls run on rayon's
    /// workers, whose stacks bottom out in the pool rather than in the
    /// layer that asked, so every linear of every block lands on one frame.
    /// This is the layer's own account, kept per operation class.
    stages: Mutex<Stages>,
    /// The timestep-dependent vectors of the last pass, keyed by its sigma:
    /// every block's `img_mod`/`txt_mod` output and the final `norm_out`
    /// modulation. They depend on nothing but the timestep, and under
    /// guidance the negative prompt's pass follows the positive one at the
    /// same sigma — so half the modulation linears (sixty `[3072 → 18432]`
    /// reads of one token each, weight-bound) were computing what the pass
    /// before had.
    modulation: Mutex<Option<ModulationCache>>,
    /// Nanoseconds the adapters took in the current pass — `linear` has no
    /// clock of its own, so it adds here and `forward` moves the total into
    /// [`Stages::lora`] (subtracting it from the class it was counted in).
    lora_nanos: std::sync::atomic::AtomicU64,
    img_in: Linear,
    txt_in: Linear,
    txt_norm: Vec<f32>,
    time_in: Linear,
    time_out: Linear,
    blocks: Vec<Block>,
    norm_out: Linear,
    proj_out: Linear,
}

/// A forward pass's time by operation class, summed over the blocks —
/// what `orangu-bench --image` prints under each row, and the only way to
/// tell the modulation linears (read for one token, sixty times a pass)
/// from the token-wide ones.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stages {
    /// `img_mod`/`txt_mod`: `[dim, 6 dim]` weights read for a single token,
    /// per block — pure weight traffic.
    pub modulation: Duration,
    /// The six attention input projections (`to_q/k/v`, `add_q/k/v_proj`).
    pub qkv: Duration,
    /// The head norms, RoPE, and the joint attention itself. When a pass
    /// overlaps its attention with the device, this is the attention's wall
    /// time, with the earlier chunks' output projection and MLP running on
    /// the device inside it.
    pub attention: Duration,
    /// The two attention output projections.
    pub out: Duration,
    /// The four feed-forward linears and their GELU. Under the overlap,
    /// the wait for the last chunk's output projection and MLP after the
    /// attention ends.
    pub mlp: Duration,
    /// Embeddings, layer norms, modulation arithmetic, gated residual adds,
    /// and the final projection — everything else.
    pub other: Duration,
    /// The low-rank adapter's two small products on every adapted linear
    /// (`super::lora`) — counted apart from the linear they ride on, and
    /// already excluded from the classes above.
    pub lora: Duration,
    /// Whole passes, so a caller can say "per pass".
    pub passes: u32,
    /// Under the overlap, outside [`Self::total`]: the device lane's time
    /// running chunks' tails, its time waiting for the next chunk's
    /// attention, and the share of that wait that is the first chunk's
    /// attention (which nothing can fill) — what says which side of the
    /// overlap a pass waits on.
    pub lane_busy: Duration,
    pub lane_idle: Duration,
    pub first_chunk: Duration,
    /// The lane's busy time by phase: the output projection's call, the
    /// feed-forward's call, and its own host work between them (the gated
    /// adds, the norm, the scale).
    pub lane_out: Duration,
    pub lane_mlp: Duration,
    /// The lane's time on the heads it runs on the card.
    pub lane_heads: Duration,
}

impl Stages {
    pub fn total(&self) -> Duration {
        self.modulation + self.qkv + self.attention + self.out + self.mlp + self.other + self.lora
    }

    pub(super) fn add(&mut self, other: &Stages) {
        self.modulation += other.modulation;
        self.qkv += other.qkv;
        self.attention += other.attention;
        self.out += other.out;
        self.mlp += other.mlp;
        self.other += other.other;
        self.lora += other.lora;
        self.passes += other.passes;
        self.lane_busy += other.lane_busy;
        self.lane_idle += other.lane_idle;
        self.first_chunk += other.first_chunk;
        self.lane_out += other.lane_out;
        self.lane_mlp += other.lane_mlp;
        self.lane_heads += other.lane_heads;
    }
}

/// Times the span from the last mark to now into one of a pass's stages.
pub(super) struct StageClock {
    pub(super) stages: Stages,
    last: Instant,
}

impl StageClock {
    pub(super) fn start() -> Self {
        Self {
            stages: Stages {
                passes: 1,
                ..Stages::default()
            },
            last: Instant::now(),
        }
    }

    pub(super) fn lap(&mut self, into: fn(&mut Stages) -> &mut Duration) {
        let now = Instant::now();
        *into(&mut self.stages) += now - self.last;
        self.last = now;
    }
}

/// See [`QwenImageTransformer::modulation`].
struct ModulationCache {
    sigma: f32,
    /// `[block][6 * dim]` — `img_mod`'s output per block.
    img: Vec<Vec<f32>>,
    /// `[block][6 * dim]` — `txt_mod`'s output per block.
    txt: Vec<Vec<f32>>,
    /// `norm_out.linear`'s output.
    out: Vec<f32>,
}

/// What one forward pass is given.
pub struct ForwardInput<'a> {
    /// `[n_img, in_channels]` packed latent patches, row-major over the
    /// `(rows, cols)` grid.
    pub img: &'a [f32],
    /// The token grid: `(rows, cols)` = `(H/16, W/16)`.
    pub grid: (usize, usize),
    /// `[n_txt, txt_dim]` text encoder hidden states.
    pub txt: &'a [f32],
    /// The noise level in `[0, 1]`.
    pub sigma: f32,
    /// Checked before every block: set, the pass stops with an error — a
    /// cancelled request, or the server shutting down, waits for one block
    /// (a couple of seconds at 1024 px) rather than for the whole pass.
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl QwenImageTransformer {
    /// Reads the checkpoint's dimensions from its tensors, for the loader.
    pub fn config_from(loaded: &LoadedModel) -> Result<TransformerConfig> {
        let (_, img_in) = loaded.tensor_dims("img_in.weight")?;
        let (_, txt_in) = loaded.tensor_dims("txt_in.weight")?;
        let (_, norm_q) = loaded.tensor_dims("transformer_blocks.0.attn.norm_q.weight")?;
        let in_channels = img_in[0] as usize;
        let dim = img_in[1] as usize;
        let txt_dim = txt_in[0] as usize;
        let head_dim = norm_q[0] as usize;
        ensure!(
            head_dim > 0 && dim.is_multiple_of(head_dim),
            "qwen_image: stream width {dim} is not a multiple of the head width {head_dim}"
        );
        let n_layer = (0..)
            .take_while(|i| loaded.has_tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")))
            .count();
        ensure!(n_layer > 0, "qwen_image: no transformer_blocks.N tensors");
        // diffusers' `axes_dims_rope = (16, 56, 56)` for a 128-wide head:
        // the frame axis takes 16 of the 128, rows and columns 56 each. The
        // checkpoint does not say so anywhere; the split is the model's.
        let frame = head_dim / 8;
        let spatial = (head_dim - frame) / 2;
        ensure!(
            frame + 2 * spatial == head_dim && frame.is_multiple_of(2) && spatial.is_multiple_of(2),
            "qwen_image: head width {head_dim} does not split into the rotary axes"
        );
        Ok(TransformerConfig {
            dim,
            n_head: dim / head_dim,
            head_dim,
            n_layer,
            txt_dim,
            in_channels,
            rope_axes: [frame / 2, spatial / 2, spatial / 2],
            eps: 1e-6,
        })
    }

    pub fn load(
        loaded: &LoadedModel,
        backend: Arc<dyn Backend>,
        mut lora: Option<super::lora::Lora>,
        merge_lora: bool,
        mut cache: Option<&mut super::lora::MergedCache>,
    ) -> Result<Self> {
        let config = Self::config_from(loaded)?;
        let mut linear = |name: &str| -> Result<Linear> {
            let w = loaded
                .matrix(&format!("{name}.weight"))
                .with_context(|| format!("qwen_image: loading {name}"))?;
            let bias_name = format!("{name}.bias");
            let bias = if loaded.has_tensor(&bias_name) {
                let (values, _) = loaded.tensor(&bias_name)?;
                ensure!(
                    values.len() == w.out_dim,
                    "qwen_image: {bias_name} has {} values for {} outputs",
                    values.len(),
                    w.out_dim
                );
                Some(values)
            } else {
                None
            };
            // The adapter is keyed by the linear's own name; a pair that
            // does not match the weight's shape is refused rather than
            // applied to the wrong thing.
            let (w, lora) = match lora.as_mut().and_then(|l| l.take(name)) {
                Some(pair) => {
                    ensure!(
                        pair.down.in_dim == w.in_dim && pair.up.out_dim == w.out_dim,
                        "qwen_image: the LoRA for {name} is {}→{}→{}, the linear {}→{}",
                        pair.down.in_dim,
                        pair.rank,
                        pair.up.out_dim,
                        w.in_dim,
                        w.out_dim
                    );
                    if merge_lora {
                        // The cache's copy when it has one, else merged now
                        // and recorded for the cache to write.
                        let cached = cache.as_deref().and_then(|c| c.get(name, &w));
                        let merged = match cached {
                            Some(merged) => merged,
                            None => {
                                let merged = pair.merge_into(&w, name)?;
                                if let Some(cache) = cache.as_deref_mut() {
                                    cache.put(name, &merged);
                                }
                                merged
                            }
                        };
                        (merged, None)
                    } else {
                        (w, Some(pair))
                    }
                }
                None => (w, None),
            };
            Ok(Linear { w, bias, lora })
        };
        let vector = |name: &str, len: usize| -> Result<Vec<f32>> {
            let (values, _) = loaded
                .tensor(name)
                .with_context(|| format!("qwen_image: loading {name}"))?;
            ensure!(
                values.len() == len,
                "qwen_image: {name} has {} values, expected {len}",
                values.len()
            );
            Ok(values)
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                img_mod: linear(&format!("{p}.img_mod.1"))?,
                txt_mod: linear(&format!("{p}.txt_mod.1"))?,
                to_q: linear(&format!("{p}.attn.to_q"))?,
                to_k: linear(&format!("{p}.attn.to_k"))?,
                to_v: linear(&format!("{p}.attn.to_v"))?,
                to_out: linear(&format!("{p}.attn.to_out.0"))?,
                add_q: linear(&format!("{p}.attn.add_q_proj"))?,
                add_k: linear(&format!("{p}.attn.add_k_proj"))?,
                add_v: linear(&format!("{p}.attn.add_v_proj"))?,
                to_add_out: linear(&format!("{p}.attn.to_add_out"))?,
                norm_q: vector(&format!("{p}.attn.norm_q.weight"), config.head_dim)?,
                norm_k: vector(&format!("{p}.attn.norm_k.weight"), config.head_dim)?,
                norm_added_q: vector(&format!("{p}.attn.norm_added_q.weight"), config.head_dim)?,
                norm_added_k: vector(&format!("{p}.attn.norm_added_k.weight"), config.head_dim)?,
                img_mlp_in: linear(&format!("{p}.img_mlp.net.0.proj"))?,
                img_mlp_out: linear(&format!("{p}.img_mlp.net.2"))?,
                txt_mlp_in: linear(&format!("{p}.txt_mlp.net.0.proj"))?,
                txt_mlp_out: linear(&format!("{p}.txt_mlp.net.2"))?,
            });
        }
        let img_in = linear("img_in")?;
        let txt_in = linear("txt_in")?;
        let time_in = linear("time_text_embed.timestep_embedder.linear_1")?;
        let time_out = linear("time_text_embed.timestep_embedder.linear_2")?;
        let norm_out = linear("norm_out.linear")?;
        let proj_out = linear("proj_out")?;
        if let Some(lora) = &lora
            && !lora.is_empty()
        {
            log::warn!(
                "orangu-server: [image] {} LoRA pair(s) name linears this transformer does not \
                 have and were not applied: {}",
                lora.len(),
                lora.remaining().join(", ")
            );
        }
        let model = Self {
            backend,
            img_in,
            txt_in,
            txt_norm: vector("txt_norm.weight", config.txt_dim)?,
            time_in,
            time_out,
            norm_out,
            proj_out,
            blocks,
            config,
            stages: Mutex::new(Stages::default()),
            modulation: Mutex::new(None),
            lora_nanos: std::sync::atomic::AtomicU64::new(0),
        };
        let c = &model.config;
        ensure!(
            model.time_in.w.in_dim == TIMESTEP_FREQUENCIES
                && model.time_out.w.out_dim == c.dim
                && model.norm_out.w.out_dim == 2 * c.dim
                && model.proj_out.w.out_dim == c.in_channels,
            "qwen_image: the embedding and output projections do not match the stream width"
        );
        for (i, block) in model.blocks.iter().enumerate() {
            ensure!(
                block.img_mod.w.out_dim == 6 * c.dim
                    && block.txt_mod.w.out_dim == 6 * c.dim
                    && block.to_q.w.out_dim == c.dim
                    && block.add_q.w.out_dim == c.dim
                    && block.img_mlp_in.w.in_dim == c.dim
                    && block.img_mlp_out.w.out_dim == c.dim,
                "qwen_image: transformer_blocks.{i} does not match the stream width {}",
                c.dim
            );
        }
        Ok(model)
    }

    /// The stage account of every pass since the last call, and a fresh
    /// start.
    pub fn take_stages(&self) -> Stages {
        let mut stages = self
            .stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::take(&mut *stages)
    }

    fn linear(&self, x: &[f32], n: usize, layer: &Linear) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * layer.w.in_dim);
        let mut out = self
            .backend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: n,
                w: &layer.w,
            }])
            .pop()
            .expect("one op in, one result out");
        if let Some(bias) = &layer.bias {
            tensor::add_bias_per_row(&mut out, bias, n);
        }
        if let Some(lora) = &layer.lora {
            let started = Instant::now();
            lora.apply(&*self.backend, x, n, &mut out);
            self.lora_nanos.fetch_add(
                started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        out
    }

    /// One velocity prediction: `[n_img, in_channels]`, the same packed
    /// layout the input latents came in.
    pub fn forward(&self, input: &ForwardInput<'_>) -> Result<Vec<f32>> {
        let c = &self.config;
        let (rows, cols) = input.grid;
        let n_img = rows * cols;
        ensure!(
            input.img.len() == n_img * c.in_channels,
            "qwen_image: {} latent values for a {rows}x{cols} grid of {}-wide tokens",
            input.img.len(),
            c.in_channels
        );
        ensure!(
            !input.txt.is_empty() && input.txt.len().is_multiple_of(c.txt_dim),
            "qwen_image: text hidden states are not rows of {}",
            c.txt_dim
        );
        let n_txt = input.txt.len() / c.txt_dim;
        let n = n_txt + n_img;
        let mut clock = StageClock::start();
        self.lora_nanos
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // Embeddings.
        let mut x = self.linear(input.img, n_img, &self.img_in);
        let mut txt_normed = Vec::new();
        tensor::rmsnorm_into(
            &mut txt_normed,
            input.txt,
            &self.txt_norm,
            n_txt,
            c.txt_dim,
            c.eps,
        );
        let mut t = self.linear(&txt_normed, n_txt, &self.txt_in);
        drop(txt_normed);

        // The timestep embedding, and its SiLU that every modulation reads.
        let temb = {
            let proj = timestep_embedding(input.sigma);
            let mut h = self.linear(&proj, 1, &self.time_in);
            silu_inplace(&mut h);
            self.linear(&h, 1, &self.time_out)
        };
        let mut temb_silu = temb;
        silu_inplace(&mut temb_silu);
        clock.lap(|s| &mut s.other);
        // The modulation vectors: reused from the previous pass when it was
        // at this sigma (guidance's second pass), computed otherwise.
        let modulation = {
            let cached = self
                .modulation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
                .filter(|m| m.sigma == input.sigma);
            match cached {
                Some(m) => m,
                None => ModulationCache {
                    sigma: input.sigma,
                    img: self
                        .blocks
                        .iter()
                        .map(|block| self.linear(&temb_silu, 1, &block.img_mod))
                        .collect(),
                    txt: self
                        .blocks
                        .iter()
                        .map(|block| self.linear(&temb_silu, 1, &block.txt_mod))
                        .collect(),
                    out: self.linear(&temb_silu, 1, &self.norm_out),
                },
            }
        };

        clock.lap(|s| &mut s.modulation);

        let rope = RopeTable::new(c, (rows, cols), n_txt);
        let scale = 1.0 / (c.head_dim as f32).sqrt();

        let mut normed = Vec::new();
        clock.lap(|s| &mut s.other);
        for (bi, block) in self.blocks.iter().enumerate() {
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            let img_mod = &modulation.img[bi];
            let txt_mod = &modulation.txt[bi];
            let [
                img_shift1,
                img_scale1,
                img_gate1,
                img_shift2,
                img_scale2,
                img_gate2,
            ] = split_modulation(img_mod, c.dim);
            let [
                txt_shift1,
                txt_scale1,
                txt_gate1,
                txt_shift2,
                txt_scale2,
                txt_gate2,
            ] = split_modulation(txt_mod, c.dim);

            // Attention, joint over [text | image].
            layer_norm_into(&mut normed, &x, c.dim, c.eps);
            modulate_inplace(&mut normed, img_shift1, img_scale1, c.dim);
            clock.lap(|s| &mut s.other);
            let mut q = self.linear(&normed, n_img, &block.to_q);
            let mut k = self.linear(&normed, n_img, &block.to_k);
            let v = self.linear(&normed, n_img, &block.to_v);
            clock.lap(|s| &mut s.qkv);
            layer_norm_into(&mut normed, &t, c.dim, c.eps);
            modulate_inplace(&mut normed, txt_shift1, txt_scale1, c.dim);
            clock.lap(|s| &mut s.other);
            let mut tq = self.linear(&normed, n_txt, &block.add_q);
            let mut tk = self.linear(&normed, n_txt, &block.add_k);
            let tv = self.linear(&normed, n_txt, &block.add_v);
            clock.lap(|s| &mut s.qkv);

            head_rms_norm(&mut q, &block.norm_q, c.head_dim, c.eps);
            head_rms_norm(&mut k, &block.norm_k, c.head_dim, c.eps);
            head_rms_norm(&mut tq, &block.norm_added_q, c.head_dim, c.eps);
            head_rms_norm(&mut tk, &block.norm_added_k, c.head_dim, c.eps);
            rope.apply(&mut q, c, &rope.img);
            rope.apply(&mut k, c, &rope.img);
            rope.apply(&mut tq, c, &rope.txt);
            rope.apply(&mut tk, c, &rope.txt);

            let joint_q = concat_rows(&tq, &q);
            let joint_k = concat_rows(&tk, &k);
            let joint_v = concat_rows(&tv, &v);
            drop((q, k, v, tq, tk, tv));
            let attn = step_attention(
                &joint_q, n, &joint_k, &joint_v, n, c.n_head, c.head_dim, scale, None,
            );
            drop((joint_q, joint_k, joint_v));
            clock.lap(|s| &mut s.attention);

            let txt_attn = self.linear(&attn[..n_txt * c.dim], n_txt, &block.to_add_out);
            let img_attn = self.linear(&attn[n_txt * c.dim..], n_img, &block.to_out);
            drop(attn);
            clock.lap(|s| &mut s.out);
            gated_add(&mut x, &img_attn, img_gate1, c.dim);
            gated_add(&mut t, &txt_attn, txt_gate1, c.dim);

            // Feed-forward, each stream on its own.
            layer_norm_into(&mut normed, &x, c.dim, c.eps);
            modulate_inplace(&mut normed, img_shift2, img_scale2, c.dim);
            clock.lap(|s| &mut s.other);
            let mut h = self.linear(&normed, n_img, &block.img_mlp_in);
            gelu_inplace(&mut h);
            let mlp = self.linear(&h, n_img, &block.img_mlp_out);
            drop(h);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut x, &mlp, img_gate2, c.dim);

            layer_norm_into(&mut normed, &t, c.dim, c.eps);
            modulate_inplace(&mut normed, txt_shift2, txt_scale2, c.dim);
            clock.lap(|s| &mut s.other);
            let mut h = self.linear(&normed, n_txt, &block.txt_mlp_in);
            gelu_inplace(&mut h);
            let mlp = self.linear(&h, n_txt, &block.txt_mlp_out);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut t, &mlp, txt_gate2, c.dim);
        }

        // `AdaLayerNormContinuous`: `scale, shift = chunk(2)` — the other
        // order from the blocks' `shift, scale, gate`.
        let (scale_v, shift_v) = modulation.out.split_at(c.dim);
        layer_norm_into(&mut normed, &x, c.dim, c.eps);
        modulate_inplace(&mut normed, shift_v, scale_v, c.dim);
        let out = self.linear(&normed, n_img, &self.proj_out);
        clock.lap(|s| &mut s.other);
        // The adapters' time was clocked inside the linears' classes; move
        // it to its own, taking it out of the largest — the MLP, which is
        // where most of the adapted linears are — as the nearest honest
        // split without a clock per call.
        let lora = Duration::from_nanos(self.lora_nanos.load(std::sync::atomic::Ordering::Relaxed));
        clock.stages.lora = lora;
        clock.stages.mlp = clock.stages.mlp.saturating_sub(lora);
        *self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(modulation);
        self.stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .add(&clock.stages);
        Ok(out)
    }
}

/// The 256-wide sinusoidal timestep projection: diffusers' `Timesteps(256,
/// flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)` applied to
/// `sigma` — `[cos(1000 σ ω_i) | sin(1000 σ ω_i)]` for `ω_i = 10000^(-i/128)`.
pub(super) const TIMESTEP_FREQUENCIES: usize = 256;

pub(super) fn timestep_embedding(sigma: f32) -> Vec<f32> {
    let half = TIMESTEP_FREQUENCIES / 2;
    let t = sigma as f64 * 1000.0;
    let mut out = vec![0.0f32; TIMESTEP_FREQUENCIES];
    for i in 0..half {
        let freq = (-(10000.0f64).ln() * i as f64 / half as f64).exp();
        let angle = t * freq;
        out[i] = angle.cos() as f32;
        out[half + i] = angle.sin() as f32;
    }
    out
}

/// `img_mod`/`txt_mod`'s `[6 * dim]` output as its six vectors: diffusers
/// chunks it in two, then each half into `shift, scale, gate`.
fn split_modulation(m: &[f32], dim: usize) -> [&[f32]; 6] {
    debug_assert_eq!(m.len(), 6 * dim);
    std::array::from_fn(|i| &m[i * dim..(i + 1) * dim])
}

/// Weightless LayerNorm over each `dim`-wide row of `src` into `dst`.
pub(super) fn layer_norm_into(dst: &mut Vec<f32>, src: &[f32], dim: usize, eps: f32) {
    dst.resize(src.len(), 0.0);
    dst.par_chunks_mut(dim)
        .zip(src.par_chunks(dim))
        .for_each(|(out, row)| {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (o, v) in out.iter_mut().zip(row) {
                *o = (v - mean) * inv;
            }
        });
}

/// `x * (1 + scale) + shift`, per row.
pub(super) fn modulate_inplace(x: &mut [f32], shift: &[f32], scale: &[f32], dim: usize) {
    x.par_chunks_mut(dim).for_each(|row| {
        for ((v, s), sh) in row.iter_mut().zip(scale).zip(shift) {
            *v = *v * (1.0 + s) + sh;
        }
    });
}

/// `x += gate * y`, per row.
pub(super) fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32], dim: usize) {
    x.par_chunks_mut(dim)
        .zip(y.par_chunks(dim))
        .for_each(|(row, add)| {
            for ((v, a), g) in row.iter_mut().zip(add).zip(gate) {
                *v += g * a;
            }
        });
}

/// RMSNorm over every `head_dim`-wide head of every row, one weight vector
/// shared by all heads — `attn.norm_q` and its siblings.
pub(super) fn head_rms_norm(x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    x.par_chunks_mut(head_dim).for_each(|head| {
        let mean_sq = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for (v, w) in head.iter_mut().zip(weight) {
            *v = *v * scale * w;
        }
    });
}

pub(super) fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = tensor::silu(*v);
    }
}

/// diffusers' `gelu-approximate` is the tanh form, which `tensor::gelu` is
/// — `tensor::gelu_inplace` is its vectorised, parallel form. Per element
/// through the scalar function this was 4% of a pass (`expm1f` and `tanhf`
/// under it), applied to `n × 12288` values sixty times.
fn gelu_inplace(x: &mut [f32]) {
    tensor::gelu_inplace(x);
}

pub(super) fn concat_rows(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// The rotary angles for one forward: `(cos, sin)` per complex pair, per
/// token, for the image tokens and for the text tokens.
struct RopeTable {
    /// `[n_img][pairs]` of `(cos, sin)`.
    img: Vec<(f32, f32)>,
    /// `[n_txt][pairs]`.
    txt: Vec<(f32, f32)>,
    pairs: usize,
}

impl RopeTable {
    fn new(c: &TransformerConfig, (rows, cols): (usize, usize), n_txt: usize) -> Self {
        let pairs = c.head_dim / 2;
        let [frame_pairs, row_pairs, col_pairs] = c.rope_axes;
        // `rope_params`: pair `j` of an axis of `d` pairs turns at
        // `theta^(-2j/(2d))` per unit of position.
        let freqs = |d: usize| -> Vec<f64> {
            (0..d)
                .map(|j| 1.0 / 10000f64.powf(2.0 * j as f64 / (2 * d) as f64))
                .collect()
        };
        let (f_frame, f_row, f_col) = (freqs(frame_pairs), freqs(row_pairs), freqs(col_pairs));
        let angle = |pos: f64, f: f64| {
            let a = pos * f;
            (a.cos() as f32, a.sin() as f32)
        };
        // `scale_rope`: rows and columns are centred — position `r` of
        // `rows` sits at `r - ceil(rows / 2)`, so the middle of the image is
        // near zero and the two halves rotate in opposite directions.
        let mut img = Vec::with_capacity(rows * cols * pairs);
        for r in 0..rows {
            let row_pos = r as f64 - (rows - rows / 2) as f64;
            for col in 0..cols {
                let col_pos = col as f64 - (cols - cols / 2) as f64;
                img.extend(f_frame.iter().map(|&f| angle(0.0, f)));
                img.extend(f_row.iter().map(|&f| angle(row_pos, f)));
                img.extend(f_col.iter().map(|&f| angle(col_pos, f)));
            }
        }
        // Text tokens start past the image's half-extent, at the same index
        // on all three axes.
        let max_vid_index = (rows / 2).max(cols / 2);
        let mut txt = Vec::with_capacity(n_txt * pairs);
        for i in 0..n_txt {
            let pos = (max_vid_index + i) as f64;
            txt.extend(f_frame.iter().map(|&f| angle(pos, f)));
            txt.extend(f_row.iter().map(|&f| angle(pos, f)));
            txt.extend(f_col.iter().map(|&f| angle(pos, f)));
        }
        Self { img, txt, pairs }
    }

    /// Rotates every head of every row of `x` (`[n, n_head * head_dim]`) by
    /// that row's angles, as a complex multiply on *adjacent* pairs —
    /// diffusers' `apply_rotary_emb_qwen(use_real=False)`.
    fn apply(&self, x: &mut [f32], c: &TransformerConfig, table: &[(f32, f32)]) {
        let dim = c.dim;
        let pairs = self.pairs;
        x.par_chunks_mut(dim)
            .zip(table.par_chunks(pairs))
            .for_each(|(row, angles)| {
                for head in row.chunks_mut(c.head_dim) {
                    for (pair, &(cos, sin)) in head.chunks_mut(2).zip(angles) {
                        let (a, b) = (pair[0], pair[1]);
                        pair[0] = a * cos - b * sin;
                        pair[1] = a * sin + b * cos;
                    }
                }
            });
    }
}

/// Full (unmasked, non-causal) multi-head attention over one sequence of
/// `n` rows: `[n, n_head * head_dim]` in, the same shape out.
///
/// Keys and values are regathered head-major first — keys as `[n,
/// head_dim]` per head, values **transposed** to `[head_dim, n]` — so each
/// head's products run over contiguous blocks. Parallel over `(head, block
/// of [`ATTN_QUERIES`] queries)`.
///
/// The block is the point. One query at a time (the first version, still
/// the fallback on other architectures), every query streamed its head's
/// whole key set and whole value set from cache or memory — at 1024 pixels
/// that is 2 MiB of keys and 2 MiB of values per query per head, 4,126
/// queries and 24 heads a pass, and attention was on its way to being most
/// of the step (13% of a pass at 256 pixels, 36% at 512). A block of
/// [`ATTN_QUERIES`] queries reads the keys once
/// ([`vecdot::gemm_f32_rows_many`], the key tile outermost) and the
/// transposed values once, each product a register-tiled GEMM rather than a
/// dot per row.
///
/// Same arithmetic as the direct form to `f32` rounding: the scores are
/// the same dot products in a different summation order, the softmax is
/// per query row exactly as before, and the weighted sum of values is the
/// same sum reordered.
pub(crate) fn joint_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    attention_blocked(q, n, k, v, n, n_head, head_dim, scale, None, false)
}

/// Multi-head attention of `n_q` query rows over `n_kv` key/value rows —
/// the same blocked kernel as [`joint_attention`], for Qwen-Image 2.1's
/// sequences, where the queries are a suffix of the keys: the picture's
/// tokens attend over the cached prompt prefix and themselves.
///
/// With `limits`, query `i` sees only the first `limits[i]` keys — the
/// prompt prefix's block-causal mask: a text token sees the keys up to its
/// own, a reference picture's token every key up to the end of its picture
/// (see [`causal_limits`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    limits: Option<&[usize]>,
) -> Vec<f32> {
    attention_blocked(q, n_q, k, v, n_kv, n_head, head_dim, scale, limits, false)
}

/// [`step_attention`] for the heads `heads` only — the other heads'
/// columns zero, for another device to fill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_attention_heads(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    heads: std::ops::Range<usize>,
) -> Vec<f32> {
    let int8 = int8_step_attention() && head_dim.is_multiple_of(8);
    attention_blocked_heads(
        q, n_q, k, v, n_kv, n_head, head_dim, scale, None, int8, heads,
    )
}

/// The non-causal attention of the heads `heads` on `vulkan`
/// (`VulkanBackend::gpu_attention_prefill`): their columns of the queries,
/// keys and values gathered, `[n_q][heads · head_dim]` back. The queries are
/// the last `n_q` of the `n_kv` keys. `cache` holds the gathered keys and
/// values (and their device mirror) from one call to the next and is
/// rebuilt when too short or of another width.
///
/// `ORANGU_IMAGE_HEADS_TRACE=1`: a line a call — the gather of the queries'
/// columns, the cache fill, the mirror's upload, the device's call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_head_attention(
    vulkan: &crate::engine::backend::VulkanBackend,
    cache: &mut Option<crate::engine::kv_cache::KvCache>,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    (n_q, n_kv): (usize, usize),
    (dim, head_dim): (usize, usize),
    heads: std::ops::Range<usize>,
    scale: f32,
) -> Vec<f32> {
    let (first, width) = (heads.start * head_dim, heads.len() * head_dim);
    let trace = crate::engine::env::flag_on("ORANGU_IMAGE_HEADS_TRACE").then(Instant::now);
    let mut q_sub = vec![0.0f32; n_q * width];
    q_sub
        .par_chunks_mut(width)
        .zip(q.par_chunks(dim))
        .with_min_len(64)
        .for_each(|(sub, row)| sub.copy_from_slice(&row[first..first + width]));
    let gathered = trace.map(|t| t.elapsed());
    let cache = match cache {
        Some(held) if held.layers[0].capacity() > n_kv && held.layers[0].kv_dim() == width => {
            held.layers[0].truncate(0);
            held
        }
        _ => cache.insert(crate::engine::kv_cache::KvCache::new(1, n_kv + 1, width)),
    };
    for (k, v) in keys.chunks(dim).zip(values.chunks(dim)) {
        cache.layers[0].push(&k[first..first + width], &v[first..first + width]);
    }
    let filled = trace.map(|t| t.elapsed());
    if trace.is_some() {
        vulkan.sync_kv_mirror(&mut cache.layers[0], heads.len());
    }
    let synced = trace.map(|t| t.elapsed());
    let out = vulkan.gpu_attention_prefill(
        &q_sub,
        &mut cache.layers[0],
        n_kv - n_q,
        n_q,
        heads.len(),
        heads.len(),
        head_dim,
        0,
        false,
        scale,
    );
    if let (Some(t), Some(g), Some(f), Some(y)) = (trace, gathered, filled, synced) {
        eprintln!(
            "orangu-server: [image] {} heads on the device: gather {:.1} ms, cache {:.1} ms, \
             kv upload {:.1} ms, device {:.1} ms",
            heads.len(),
            g.as_secs_f64() * 1e3,
            (f - g).as_secs_f64() * 1e3,
            (y - f).as_secs_f64() * 1e3,
            (t.elapsed() - y).as_secs_f64() * 1e3
        );
    }
    out
}

/// [`attention`] for a denoising step's picture tokens, where attention is
/// the part of a pass that grows with the square of the picture (39% of a
/// Qwen-Image 2.1 step at 1024 x 1024, against 14% at 512): the scores
/// `q · k` are computed in `int8` on the `smmla` kernel (`aarch64`) or
/// `vpmaddubsw` (`x86_64`) — several times the multiply-adds per
/// instruction of the `f32` path — while the softmax and
/// the value product stay `f32`. Each query and key row is quantized with
/// its own scale after the keys' per-channel mean over the sequence is
/// taken out (which moves every score of a query by the same amount, so the
/// softmax is unchanged, and leaves the rows centred for `int8`): the
/// SageAttention recipe. The heads are RMS-normed before this, so their
/// rows are well-conditioned.
///
/// The `f32` path runs instead without an `int8` score kernel
/// (`vecdot::have_i8_scores`), for a head width that is not a multiple of
/// 8, or with `ORANGU_IMAGE_ATTENTION=f32`, which is the switch for
/// comparing the two on one binary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_attention(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    limits: Option<&[usize]>,
) -> Vec<f32> {
    let int8 = int8_step_attention() && head_dim.is_multiple_of(8);
    attention_blocked(q, n_q, k, v, n_kv, n_head, head_dim, scale, limits, int8)
}

/// The host's exact `f32` step attention — no `int8` scores — for a
/// device probe to check itself against.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_attention_f32_for_probe(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    attention_blocked(q, n_q, k, v, n_kv, n_head, head_dim, scale, None, false)
}

/// Where attention's value product `P · V` runs, `ORANGU_IMAGE_PV` choosing for an A/B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueProduct {
    /// `bfmmla` on `bf16` probabilities and values, summed in `f32`
    /// (`vecdot::bf16_tiles`) — the default where the CPU has `bf16`.
    Bf16,
    /// `rten-gemm`'s `f32` kernel: its 4 × 16 packed tile does ~150 G
    /// MAC/s at the block shape (24 queries × 4,096 keys × 128) where
    /// `gemm_f32_rows` does ~100 (`ORANGU_IMAGE_PV=rten`).
    Rten,
    /// This crate's own `f32` tile (`ORANGU_IMAGE_PV=orangu`).
    Orangu,
}

pub(crate) fn value_product() -> ValueProduct {
    static CHOICE: std::sync::OnceLock<ValueProduct> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        let bf16 = crate::engine::vecdot::have_bf16mm();
        match crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_PV")
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Ok("orangu") => ValueProduct::Orangu,
            Ok("rten") => ValueProduct::Rten,
            _ if bf16 => ValueProduct::Bf16,
            _ => ValueProduct::Rten,
        }
    })
}

/// See [`step_attention`].
pub(crate) fn int8_step_attention() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::engine::vecdot::have_i8_scores()
            && !crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_ATTENTION")
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("f32"))
    })
}

/// The key limits of a plain causal mask over `n` positions: position `i`
/// sees keys `0..=i`.
pub(crate) fn causal_limits(n: usize) -> Vec<usize> {
    (1..=n).collect()
}

/// Queries per [`joint_attention_blocked`] task. Every block reads a head's
/// keys and values once, so more queries a block is less of that traffic;
/// fewer keep the block's score rows (`queries × n` floats) closer to the
/// core. On `aarch64` six tiles of four, the score rows in L2 through the
/// softmax and the value product (400 KiB at 1024 pixels); on `x86_64`,
/// whose score product is the `f32` tile against keys that do not fit L2,
/// twelve — the traffic is the larger cost there.
#[cfg(target_arch = "aarch64")]
const ATTN_QUERIES: usize = 24;
#[cfg(not(target_arch = "aarch64"))]
const ATTN_QUERIES: usize = 48;

/// The queries per [`attention_blocked`] task: [`ATTN_QUERIES`], or
/// `ORANGU_IMAGE_ATTN_QUERIES` (a multiple of four from 4 to 96) for an A/B
/// of the block size — more queries a block read the keys and values fewer
/// times, fewer keep its score rows in L2.
fn attn_queries() -> usize {
    static QUERIES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *QUERIES.get_or_init(|| {
        crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_ATTN_QUERIES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|q| (4..=96).contains(q) && q.is_multiple_of(4))
            .unwrap_or(ATTN_QUERIES)
    })
}

/// `scores[j] = ints[j] × sq × key_scales[j]`, returning the largest — the
/// `int8` scores to `f32` and the softmax's max in one pass.
fn convert_scores(scores: &mut [f32], ints: &[i32], key_scales: &[f32], sq: f32) -> f32 {
    debug_assert!(ints.len() == scores.len() && key_scales.len() == scores.len());
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // Safety: `AVX2` was detected; the lengths were checked above.
        return unsafe { convert_scores_avx2(scores, ints, key_scales, sq) };
    }
    let n = scores.len();
    #[cfg(not(target_arch = "aarch64"))]
    let i = 0;
    #[cfg(target_arch = "aarch64")]
    let mut i = 0;
    #[cfg(not(target_arch = "aarch64"))]
    let mut max = f32::NEG_INFINITY;
    #[cfg(target_arch = "aarch64")]
    // Safety: NEON is baseline on aarch64; every access is below `n`.
    let mut max = unsafe {
        use std::arch::aarch64::*;
        let q = vdupq_n_f32(sq);
        let mut m = [vdupq_n_f32(f32::NEG_INFINITY); 2];
        while i + 8 <= n {
            for (l, m) in m.iter_mut().enumerate() {
                let at = i + 4 * l;
                let v = vmulq_f32(
                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(ints.as_ptr().add(at))), q),
                    vld1q_f32(key_scales.as_ptr().add(at)),
                );
                vst1q_f32(scores.as_mut_ptr().add(at), v);
                *m = vmaxq_f32(*m, v);
            }
            i += 8;
        }
        vmaxvq_f32(vmaxq_f32(m[0], m[1]))
    };
    for j in i..n {
        let v = ints[j] as f32 * sq * key_scales[j];
        scores[j] = v;
        max = max.max(v);
    }
    max
}

/// [`convert_scores`] on `AVX2`: eight scores at a time, each the same
/// `ints[j] · sq · key_scales[j]` (in that order) as the scalar form, two
/// running maxima.
///
/// # Safety
/// `AVX2` must be available; the three slices the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_scores_avx2(
    scores: &mut [f32],
    ints: &[i32],
    key_scales: &[f32],
    sq: f32,
) -> f32 {
    use std::arch::x86_64::*;
    let n = scores.len();
    let mut i = 0;
    // Safety: every access is below `i + 16 <= n` in all three slices.
    let mut max = unsafe {
        let q = _mm256_set1_ps(sq);
        let mut m = [_mm256_set1_ps(f32::NEG_INFINITY); 2];
        while i + 16 <= n {
            for (l, m) in m.iter_mut().enumerate() {
                let at = i + 8 * l;
                let v = _mm256_mul_ps(
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(
                            ints.as_ptr().add(at) as *const __m256i
                        )),
                        q,
                    ),
                    _mm256_loadu_ps(key_scales.as_ptr().add(at)),
                );
                _mm256_storeu_ps(scores.as_mut_ptr().add(at), v);
                *m = _mm256_max_ps(*m, v);
            }
            i += 16;
        }
        let v = _mm256_max_ps(m[0], m[1]);
        let h = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        _mm_cvtss_f32(_mm_max_ss(h, _mm_shuffle_ps(h, h, 1)))
    };
    for j in i..n {
        let v = ints[j] as f32 * sq * key_scales[j];
        scores[j] = v;
        max = max.max(v);
    }
    max
}

/// Key pairs per chunk of a query block's `int8` scores
/// ([`int8_block_scores`]): 128 keys, 16 KiB of a 128-wide head — the chunk
/// and the block's queries stay in the first-level cache while every quad
/// of the block passes over it.
const INT8_KEY_PAIRS: usize = 64;

/// Where a query block sits: its first query `q0`, `nq` queries against
/// `n` keys, the softmax `scale`, and each query's key limit, if any.
struct BlockRows<'a> {
    q0: usize,
    nq: usize,
    n: usize,
    scale: f32,
    limits: Option<&'a [usize]>,
}

/// A query block's `int8` scores against its keys, converted to `f32` into
/// `scores` (`[nq][n]`, each row up to its limit) with each row's maximum
/// into `row_max` (which starts at `-∞`). Key chunks of `chunk_pairs` pairs
/// outermost and the block's quads inside: a head's keys are read once a
/// block rather than once a quad (a 1024² picture's are 512 KiB of `int8`,
/// an L2), and each chunk is converted while its integers are in cache.
/// The same scores, to the bit, at any chunk — a maximum does not depend on
/// the order it is taken in, and each score converts on its own.
/// `iscores` holds at least `4 · 2 · chunk_pairs` integers.
#[allow(clippy::too_many_arguments)]
fn int8_block_scores(
    qs: &crate::engine::vecdot::PairedI8,
    kq: &crate::engine::vecdot::PairedI8,
    rows: BlockRows<'_>,
    chunk_pairs: usize,
    iscores: &mut [i32],
    scores: &mut [f32],
    row_max: &mut [f32],
) {
    use crate::engine::vecdot::{F32_ROWS, i8_scores_4rows_in};
    let BlockRows {
        q0,
        nq,
        n,
        scale,
        limits,
    } = rows;
    let quads = nq.div_ceil(F32_ROWS);
    let k_pairs = n.div_ceil(2);
    let mut p = 0;
    while p < k_pairs {
        let pe = (p + chunk_pairs).min(k_pairs);
        let width = 2 * (pe - p);
        for quad in 0..quads {
            let (i0, rest) = iscores.split_at_mut(width);
            let (i1, rest) = rest.split_at_mut(width);
            let (i2, rest) = rest.split_at_mut(width);
            let i3 = &mut rest[..width];
            i8_scores_4rows_in(qs, 2 * quad, 2 * quad + 1, kq, p..pe, [i0, i1, i2, i3]);
            for r in 0..F32_ROWS {
                let row = quad * F32_ROWS + r;
                if row >= nq {
                    break;
                }
                let limit = limits.map_or(n, |l| l[q0 + row]);
                let (k0, k1) = (2 * p, (2 * pe).min(limit));
                if k0 >= k1 {
                    continue;
                }
                // The softmax's `scale` folded in here.
                let sq = qs.scales[row] * scale;
                let max = convert_scores(
                    &mut scores[row * n + k0..row * n + k1],
                    &iscores[r * width..r * width + (k1 - k0)],
                    &kq.scales[k0..k1],
                    sq,
                );
                row_max[row] = row_max[row].max(max);
            }
        }
        p = pe;
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_blocked(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    limits: Option<&[usize]>,
    int8: bool,
) -> Vec<f32> {
    attention_blocked_heads(
        q,
        n_q,
        k,
        v,
        n,
        n_head,
        head_dim,
        scale,
        limits,
        int8,
        0..n_head,
    )
}

/// [`attention_blocked`] for the heads `heads` only: the same rows for
/// those heads' columns of the `[n_q][n_head · head_dim]` output, the other
/// columns zero — what a step does when another device takes the rest of
/// the heads. Each head's arithmetic is the whole call's, to the bit.
#[allow(clippy::too_many_arguments)]
fn attention_blocked_heads(
    q: &[f32],
    n_q: usize,
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
    limits: Option<&[usize]>,
    int8: bool,
    heads: std::ops::Range<usize>,
) -> Vec<f32> {
    use crate::engine::vecdot::{F32_ROWS, PairedI8, gemm_f32_rows, gemm_f32_rows_many};
    let dim = n_head * head_dim;
    debug_assert_eq!(q.len(), n_q * dim);
    debug_assert_eq!(k.len(), n * dim);
    debug_assert!(n_q <= n);
    debug_assert!(limits.is_none_or(|l| l.len() == n_q && l.iter().all(|&m| m >= 1 && m <= n)));
    debug_assert_eq!(head_dim % F32_ROWS, 0);
    debug_assert!(heads.end <= n_head && !heads.is_empty());
    let nh = heads.len();
    // The range's keys head-major, `[nh][n][head_dim]`.
    let mut k_heads = vec![0.0f32; n * nh * head_dim];
    k_heads
        .par_chunks_mut(n * head_dim)
        .enumerate()
        .for_each(|(hl, block)| {
            let h = heads.start + hl;
            for (i, row) in block.chunks_mut(head_dim).enumerate() {
                row.copy_from_slice(&k[i * dim + h * head_dim..i * dim + (h + 1) * head_dim]);
            }
        });
    // The value product on `rten-gemm`'s `f32` kernel (see
    // [`rten_value_product`]): each head's values `[n, head_dim]`, read in
    // place through strides and packed once for every query block.
    let pv = match value_product() {
        // `bfmmla` takes the head in pairs of four: every head width here
        // is a multiple of eight, but a fallback costs nothing.
        ValueProduct::Bf16 if !head_dim.is_multiple_of(8) => ValueProduct::Rten,
        pv => pv,
    };
    let rten = pv == ValueProduct::Rten;
    let bf16 = pv == ValueProduct::Bf16;
    // For `bfmmla`: each head's values transposed (`[head_dim][n]`), in
    // `bf16`, packed once for every query block.
    let vt_bf16: Vec<crate::engine::vecdot::PackedBf16> = if bf16 {
        heads
            .clone()
            .into_par_iter()
            .map(|h| {
                crate::engine::vecdot::PackedBf16::pack_transposed(
                    head_dim,
                    n,
                    &v[h * head_dim..],
                    dim,
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let v_packed: Vec<rten_gemm::PackedBMatrix<f32>> = if rten {
        heads
            .clone()
            .into_par_iter()
            .map(|h| {
                let view = rten_tensor::NdTensorView::from_data_with_strides(
                    [n, head_dim],
                    &v[h * head_dim..],
                    [dim, 1],
                )
                .expect("a head's values fit the rows");
                rten_gemm::GemmExecutor::<f32>::new().prepack_b(view)
            })
            .collect()
    } else {
        Vec::new()
    };
    // Otherwise values head-major and transposed, `[n_head][head_dim][n]`,
    // so the value product is a GEMM over `n` with each output dimension a
    // row.
    let mut vt_heads = vec![0.0f32; if rten || bf16 { 0 } else { n * nh * head_dim }];
    vt_heads
        .par_chunks_mut(n * head_dim)
        .enumerate()
        .for_each(|(hl, block)| {
            let h = heads.start + hl;
            for (d, row) in block.chunks_mut(n).enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot = v[j * dim + h * head_dim + d];
                }
            }
        });

    // For the `int8` scores: every head's keys, less their mean over the
    // sequence, quantized per row and paired for `smmla`.
    let k_i8: Vec<PairedI8> = if int8 {
        (0..nh)
            .into_par_iter()
            .map(|h| {
                let keys = &k_heads[h * n * head_dim..(h + 1) * n * head_dim];
                let mut mean = vec![0.0f32; head_dim];
                for row in keys.chunks(head_dim) {
                    for (m, v) in mean.iter_mut().zip(row) {
                        *m += v;
                    }
                }
                for m in mean.iter_mut() {
                    *m /= n as f32;
                }
                let centred: Vec<f32> = keys
                    .chunks(head_dim)
                    .flat_map(|row| row.iter().zip(&mean).map(|(v, m)| v - m))
                    .collect();
                PairedI8::quantize(n, head_dim, |i| &centred[i * head_dim..(i + 1) * head_dim])
            })
            .collect()
    } else {
        Vec::new()
    };
    let zeros = vec![0.0f32; head_dim];

    let queries = attn_queries();
    let n_blocks = n_q.div_ceil(queries);
    let mut out = vec![0.0f32; n_q * dim];
    let sink = SharedOut(out.as_mut_ptr());
    let k_pairs = n.div_ceil(2);
    (0..nh * n_blocks).into_par_iter().for_each_init(
        || {
            (
                vec![0.0f32; queries * n],
                // Four output dimensions of a query block for this crate's
                // tile, a whole block's `[queries, head_dim]` for `rten`'s.
                vec![0.0f32; (F32_ROWS * queries).max(queries * head_dim)],
                vec![0i32; F32_ROWS * 2 * k_pairs],
                rten.then(rten_gemm::GemmExecutor::<f32>::new),
                // The block's probabilities in `bfmmla`'s layout, each row
                // written by the softmax that makes it.
                bf16.then(|| crate::engine::vecdot::PackedBf16::zeroed(queries, n)),
                // The block's queries times `scale`, for the `f32` scores.
                vec![0.0f32; queries * head_dim],
            )
        },
        move |(scores, tile, iscores, gemm, packed, q_scaled), task| {
            let sink = sink;
            let (hl, b) = (task / n_blocks, task % n_blocks);
            let h = heads.start + hl;
            let q0 = b * queries;
            let nq = queries.min(n_q - q0);
            let keys = &k_heads[hl * n * head_dim..(hl + 1) * n * head_dim];
            let values_t = if rten || bf16 {
                &[][..]
            } else {
                &vt_heads[hl * n * head_dim..(hl + 1) * n * head_dim]
            };

            // Each row's maximum, taken as the `int8` scores are converted.
            let mut row_max = vec![f32::NEG_INFINITY; nq];
            if int8 {
                // The block's queries for this head, padded to a whole
                // quad with zero rows whose scores are never read.
                let padded = nq.div_ceil(F32_ROWS) * F32_ROWS;
                let qs = PairedI8::quantize(padded, head_dim, |r| {
                    if r < nq {
                        let i = q0 + r;
                        &q[i * dim + h * head_dim..i * dim + (h + 1) * head_dim]
                    } else {
                        &zeros
                    }
                });
                int8_block_scores(
                    &qs,
                    &k_i8[hl],
                    BlockRows {
                        q0,
                        nq,
                        n,
                        scale,
                        limits,
                    },
                    INT8_KEY_PAIRS,
                    iscores,
                    scores,
                    &mut row_max,
                );
            }

            // Scores, the whole block against every key in one pass: the
            // tile's "rows" are queries, its "tokens" the keys, `in_dim` the
            // head. Key tile outermost, so a head's keys (larger than L2 at
            // 1024 pixels) are read once per block rather than once per four
            // queries. The block is padded to a whole quad with its last
            // query, whose extra score rows are never read. `scale` is
            // applied to the queries, `head_dim` multiplies a query, not to
            // the scores, `n` of them.
            if !int8 {
                for (r, dst) in q_scaled.chunks_mut(head_dim).take(nq).enumerate() {
                    let i = q0 + r;
                    let src = &q[i * dim + h * head_dim..i * dim + (h + 1) * head_dim];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = s * scale;
                    }
                }
                let padded = nq.div_ceil(F32_ROWS) * F32_ROWS;
                let rows: Vec<&[f32]> = (0..padded)
                    .map(|r| {
                        let r = r.min(nq - 1);
                        &q_scaled[r * head_dim..(r + 1) * head_dim]
                    })
                    .collect();
                let mut outs: Vec<&mut [f32]> = scores.chunks_mut(n).take(padded).collect();
                gemm_f32_rows_many(&rows, keys, head_dim, &mut outs);
            }
            // Softmax per query row, `scale` already in the scores: max,
            // then the exponentials and their sum in one pass
            // (`tensor::exp_shifted_sum`, a vector at a time — a third of
            // the blocked kernel's time was `expf`, one call per score).
            // Keys past a query's limit are masked: never read. The rows
            // stay unnormalised — the value product's output row is divided
            // by the sum instead, `head_dim` multiplies, not `n`.
            let mut inv = vec![0f32; nq];
            for (r, row) in scores[..nq * n].chunks_mut(n).enumerate() {
                let limit = limits.map_or(n, |l| l[q0 + r]);
                let live = &mut row[..limit];
                let max = if int8 {
                    row_max[r]
                } else {
                    tensor::max_f32(live)
                };
                match packed.as_mut() {
                    Some(packed) => {
                        inv[r] = 1.0 / packed.set_row_exp(r, live, max);
                    }
                    None => {
                        inv[r] = 1.0 / tensor::exp_shifted_sum(live, max);
                        row[limit..].fill(0.0);
                    }
                }
            }
            // The value product: rows are four output dimensions of the
            // transposed values, tokens the block's queries, `in_dim` the
            // keys. Written straight to each query's output row.
            let probs = &scores[..nq * n];
            if let Some(packed) = packed.as_ref() {
                let block = &mut tile[..packed.rows * head_dim];
                crate::engine::vecdot::bf16_tiles(packed, &vt_bf16[hl], block);
                for (t, row) in block.chunks_mut(head_dim).take(nq).enumerate() {
                    for v in row.iter_mut() {
                        *v *= inv[t];
                    }
                    // Safety: as below — this task's rectangle alone.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            row.as_ptr(),
                            sink.0.add((q0 + t) * dim + h * head_dim),
                            head_dim,
                        );
                    }
                }
                return;
            }
            if let Some(gemm) = gemm.as_ref() {
                let block = &mut tile[..nq * head_dim];
                gemm.gemm(
                    block,
                    rten_gemm::GemmInputA::Unpacked(rten_tensor::NdTensorView::from_data(
                        [nq, n],
                        probs,
                    )),
                    rten_gemm::GemmInputB::Packed(&v_packed[hl]),
                    rten_gemm::GemmOptions::default(),
                )
                .expect("the value product's shapes agree");
                for (t, row) in block.chunks_mut(head_dim).enumerate() {
                    for v in row.iter_mut() {
                        *v *= inv[t];
                    }
                    // Safety: as below — this task's rectangle alone.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            row.as_ptr(),
                            sink.0.add((q0 + t) * dim + h * head_dim),
                            head_dim,
                        );
                    }
                }
                return;
            }
            for d0 in (0..head_dim).step_by(F32_ROWS) {
                let vt = |r: usize| &values_t[(d0 + r) * n..(d0 + r + 1) * n];
                let (t0, rest) = tile.split_at_mut(nq);
                let (t1, rest) = rest.split_at_mut(nq);
                let (t2, rest) = rest.split_at_mut(nq);
                let (t3, _) = rest.split_at_mut(nq);
                gemm_f32_rows([vt(0), vt(1), vt(2), vt(3)], probs, n, [t0, t1, t2, t3]);
                // Safety: this task alone writes head `h`'s `head_dim` slots
                // of rows `q0..q0 + nq`; every other task's rectangle is
                // disjoint, and `out` is sized `n * dim` above.
                for t in 0..nq {
                    for r in 0..F32_ROWS {
                        unsafe {
                            *sink.0.add((q0 + t) * dim + h * head_dim + d0 + r) =
                                tile[r * nq + t] * inv[t];
                        }
                    }
                }
            }
        },
    );
    out
}

/// A raw output pointer the attention tasks share — see
/// [`joint_attention_blocked`]'s safety note.
#[derive(Clone, Copy)]
struct SharedOut(*mut f32);
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}

/// The direct form: one query row per task, a dot per key and an `axpy`
/// per value. Kept as the reference the blocked kernel is tested against.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn joint_attention_per_query(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let dim = n_head * head_dim;
    debug_assert_eq!(q.len(), n * dim);
    let gather = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; n * dim];
        out.par_chunks_mut(n * head_dim)
            .enumerate()
            .for_each(|(h, block)| {
                for (i, row) in block.chunks_mut(head_dim).enumerate() {
                    row.copy_from_slice(&src[i * dim + h * head_dim..i * dim + (h + 1) * head_dim]);
                }
            });
        out
    };
    let k_heads = gather(k);
    let v_heads = gather(v);
    let mut out = vec![0.0f32; n * dim];
    // Each task is one query row's every head, so the writes to `out` are
    // disjoint rows.
    out.par_chunks_mut(dim).enumerate().for_each_init(
        || vec![0.0f32; n],
        |scores, (i, out_row)| {
            for h in 0..n_head {
                let q_h = &q[i * dim + h * head_dim..i * dim + (h + 1) * head_dim];
                let keys = &k_heads[h * n * head_dim..(h + 1) * n * head_dim];
                let values = &v_heads[h * n * head_dim..(h + 1) * n * head_dim];
                tensor::dot_rows(q_h, keys, head_dim, scores);
                let mut max = f32::NEG_INFINITY;
                for s in scores.iter_mut() {
                    *s *= scale;
                    max = max.max(*s);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let o = &mut out_row[h * head_dim..(h + 1) * head_dim];
                o.fill(0.0);
                for (j, &p) in scores.iter().enumerate() {
                    tensor::axpy_inplace(o, &values[j * head_dim..(j + 1) * head_dim], p * inv);
                }
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the value product's arithmetic allows against an exact `f32`
    /// reference: `f32` rounding for the `f32` paths, `bf16`'s eight
    /// mantissa bits on the probabilities and the values for `bfmmla`
    /// (`ORANGU_IMAGE_PV` picks the path under test).
    fn pv_tolerance() -> f32 {
        match value_product() {
            ValueProduct::Bf16 => 3e-3,
            ValueProduct::Rten | ValueProduct::Orangu => 1e-5,
        }
    }

    #[test]
    fn the_timestep_embedding_is_cos_then_sin_of_the_scaled_step() {
        let e = timestep_embedding(0.5);
        assert_eq!(e.len(), 256);
        // Frequency 0 is 1: cos(500) and sin(500).
        assert!((e[0] - (500f64).cos() as f32).abs() < 1e-4);
        assert!((e[128] - (500f64).sin() as f32).abs() < 1e-4);
        // The last frequency is 10000^(-127/128).
        let f = (-(10000f64).ln() * 127.0 / 128.0).exp();
        assert!((e[127] - (500.0 * f).cos() as f32).abs() < 1e-5);
        assert!((e[255] - (500.0 * f).sin() as f32).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_rows_have_zero_mean_and_unit_variance() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut dst = Vec::new();
        layer_norm_into(&mut dst, &src, 4, 1e-6);
        for row in dst.chunks(4) {
            let mean = row.iter().sum::<f32>() / 4.0;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5);
            assert!((var - 1.0).abs() < 1e-3);
        }
        // The two rows differ only in scale, so their norms agree.
        assert!(
            dst[..4]
                .iter()
                .zip(&dst[4..])
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
    }

    #[test]
    fn modulation_splits_shift_scale_gate_twice() {
        let m: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let [s1, sc1, g1, s2, sc2, g2] = split_modulation(&m, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(sc1, &[2.0, 3.0]);
        assert_eq!(g1, &[4.0, 5.0]);
        assert_eq!(s2, &[6.0, 7.0]);
        assert_eq!(sc2, &[8.0, 9.0]);
        assert_eq!(g2, &[10.0, 11.0]);
        let mut x = vec![1.0, 1.0];
        modulate_inplace(&mut x, s1, sc1, 2);
        assert_eq!(x, vec![1.0 * 3.0 + 0.0, 1.0 * 4.0 + 1.0]);
    }

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            dim: 128,
            n_head: 1,
            head_dim: 128,
            n_layer: 0,
            txt_dim: 8,
            in_channels: 64,
            rope_axes: [8, 28, 28],
            eps: 1e-6,
        }
    }

    /// The image grid is centred and text starts past it, on the frame axis
    /// too — the exact `QwenEmbedRope(scale_rope=True)` layout.
    #[test]
    fn rope_positions_are_centred_for_the_image_and_offset_for_text() {
        let c = test_config();
        let table = RopeTable::new(&c, (4, 6), 3);
        assert_eq!(table.pairs, 64);
        // Image token (row 0, col 0): frame angle 0, row position -2, column
        // position -3, each at the axis' first frequency (1.0).
        let first = &table.img[..64];
        assert_eq!(first[0], (1.0, 0.0));
        assert!((first[8].0 - (-2f32).cos()).abs() < 1e-6);
        assert!((first[8].1 - (-2f32).sin()).abs() < 1e-6);
        assert!((first[36].0 - (-3f32).cos()).abs() < 1e-6);
        // Row 2 sits at position 0; column 3 at 0.
        let mid = &table.img[(2 * 6 + 3) * 64..(2 * 6 + 4) * 64];
        assert!(
            mid.iter()
                .all(|&(c, s)| (c - 1.0).abs() < 1e-6 && s.abs() < 1e-6)
        );
        // Text token 0 sits at max(4/2, 6/2) = 3 on every axis.
        let t0 = &table.txt[..64];
        assert!((t0[0].0 - 3f32.cos()).abs() < 1e-6);
        assert!((t0[8].0 - 3f32.cos()).abs() < 1e-6);
        assert!((t0[36].0 - 3f32.cos()).abs() < 1e-6);
        // Token 1 is one further along.
        assert!((table.txt[64].0 - 4f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_adjacent_pairs_as_complex_numbers() {
        let c = test_config();
        let table = RopeTable::new(&c, (1, 1), 1);
        // A single image token at (0,0) of a 1x1 grid: row position -1,
        // column position -1; frame 0.
        let mut x = vec![0.0f32; 128];
        x[0] = 1.0; // frame pair 0: unrotated
        x[16] = 1.0; // row pair 0 (dims 16,17): angle -1
        let mut y = x.clone();
        table.apply(&mut y, &c, &table.img);
        assert_eq!(&y[..2], &[1.0, 0.0]);
        assert!((y[16] - (-1f32).cos()).abs() < 1e-6);
        assert!((y[17] - (-1f32).sin()).abs() < 1e-6);
        // Rotation preserves the norm of every pair.
        x[16] = 0.6;
        x[17] = 0.8;
        table.apply(&mut x, &c, &table.img);
        assert!((x[16] * x[16] + x[17] * x[17] - 1.0).abs() < 1e-5);
    }

    /// The blocked kernel is the per-query form reordered: at a size with
    /// several query blocks, a short last block, and more than one head,
    /// every output agrees to `f32` rounding.
    #[test]
    fn blocked_attention_matches_the_per_query_form() {
        let n = 2 * ATTN_QUERIES + 7;
        let (n_head, head_dim) = (3, 8);
        let dim = n_head * head_dim;
        let q: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.25)
            .collect();
        let want = joint_attention_per_query(&q, &k, &v, n, n_head, head_dim, 0.35);
        let got = attention_blocked(&q, n, &k, &v, n, n_head, head_dim, 0.35, None, false);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() <= pv_tolerance() * e.abs().max(1.0),
                "at {i}: {g} vs {e}"
            );
        }
    }

    /// The blocked attention at a 1024² step's shape (4,096 tokens, 32
    /// heads of 128), `int8` scores and the default value product:
    ///
    /// ```text
    /// cargo test --profile release-with-debug --bin orangu-server \
    ///     attention_throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn attention_throughput() {
        let (n, n_head, head_dim) = (4096usize, 32usize, 128usize);
        let dim = n_head * head_dim;
        let data = |seed: usize| -> Vec<f32> {
            (0..n * dim)
                .map(|i| ((i * 2654435761 + seed) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        };
        let (q, k, v) = (data(1), data(2), data(3));
        let run = || attention_blocked(&q, n, &k, &v, n, n_head, head_dim, 0.088, None, true);
        let _ = run();
        let started = std::time::Instant::now();
        for _ in 0..3 {
            let _ = run();
        }
        let secs = started.elapsed().as_secs_f64() / 3.0;
        eprintln!(
            "attention {n} x {n} x {n_head} heads of {head_dim}: {:.1} ms (x 32 blocks = {:.1} s a step)",
            secs * 1e3,
            secs * 32.0
        );
    }

    /// A suffix of the queries over every key is the same as those rows of
    /// the square attention, and the causal form is the square attention
    /// with each row limited to the keys up to its own position.
    #[test]
    fn suffix_and_causal_attention_match_the_direct_forms() {
        let n = ATTN_QUERIES + 9;
        let (n_head, head_dim) = (2, 8);
        let dim = n_head * head_dim;
        let q: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.25)
            .collect();
        let full = joint_attention_per_query(&q, &k, &v, n, n_head, head_dim, 0.4);
        let n_q = 11;
        let suffix = attention(
            &q[(n - n_q) * dim..],
            n_q,
            &k,
            &v,
            n,
            n_head,
            head_dim,
            0.4,
            None,
        );
        for (g, e) in suffix.iter().zip(&full[(n - n_q) * dim..]) {
            assert!(
                (g - e).abs() <= pv_tolerance() * e.abs().max(1.0),
                "{g} vs {e}"
            );
        }
        let limits = causal_limits(n);
        let causal = attention(&q, n, &k, &v, n, n_head, head_dim, 0.4, Some(&limits));
        for i in [0, 1, ATTN_QUERIES, n - 1] {
            let m = i + 1;
            let want = joint_attention_per_query(
                &q[..m * dim],
                &k[..m * dim],
                &v[..m * dim],
                m,
                n_head,
                head_dim,
                0.4,
            );
            for (g, e) in causal[i * dim..(i + 1) * dim].iter().zip(&want[i * dim..]) {
                assert!(
                    (g - e).abs() <= pv_tolerance() * e.abs().max(1.0),
                    "row {i}: {g} vs {e}"
                );
            }
        }
    }

    /// The `int8` scores agree with the `f32` attention to within their
    /// quantization, on RMS-normed rows like the transformers' heads, and
    /// with a key offset shared by every key (which the mean-centring takes
    /// out exactly).
    #[test]
    fn int8_scores_track_the_f32_attention() {
        let n = ATTN_QUERIES * 2 + 5;
        let (n_head, head_dim) = (2, 16);
        let dim = n_head * head_dim;
        let unit = |i: usize, salt: usize| ((i * 31 + salt * 17) % 29) as f32 / 14.0 - 1.0;
        let q: Vec<f32> = (0..n * dim).map(|i| unit(i, 1)).collect();
        let k: Vec<f32> = (0..n * dim)
            .map(|i| unit(i, 2) + 3.0 * ((i % 7) == 0) as u8 as f32)
            .collect();
        let v: Vec<f32> = (0..n * dim).map(|i| unit(i, 3)).collect();
        let want = attention_blocked(&q, n, &k, &v, n, n_head, head_dim, 0.5, None, false);
        let got = attention_blocked(&q, n, &k, &v, n, n_head, head_dim, 0.5, None, true);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "worst {worst}");
        let limits = causal_limits(n);
        let want = attention_blocked(
            &q,
            n,
            &k,
            &v,
            n,
            n_head,
            head_dim,
            0.5,
            Some(&limits),
            false,
        );
        let got = attention_blocked(&q, n, &k, &v, n, n_head, head_dim, 0.5, Some(&limits), true);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "causal worst {worst}");
    }

    /// Attention over a range of the heads is the whole call's, to the bit,
    /// for those heads' columns, and zero elsewhere — so two ranges that
    /// cover the heads put together are the whole call. Both score paths,
    /// keys past the queries.
    #[test]
    fn attention_over_a_head_range_is_those_heads_of_the_whole() {
        let (n_q, n_kv, n_head, head_dim) = (ATTN_QUERIES + 5, ATTN_QUERIES + 16, 4usize, 16usize);
        let dim = n_head * head_dim;
        let unit = |i: usize, salt: usize| ((i * 31 + salt * 17) % 29) as f32 / 14.0 - 1.0;
        let q: Vec<f32> = (0..n_q * dim).map(|i| unit(i, 1)).collect();
        let k: Vec<f32> = (0..n_kv * dim).map(|i| unit(i, 2)).collect();
        let v: Vec<f32> = (0..n_kv * dim).map(|i| unit(i, 3)).collect();
        for int8 in [false, true] {
            let whole = attention_blocked(&q, n_q, &k, &v, n_kv, n_head, head_dim, 0.5, None, int8);
            let first = attention_blocked_heads(
                &q,
                n_q,
                &k,
                &v,
                n_kv,
                n_head,
                head_dim,
                0.5,
                None,
                int8,
                0..1,
            );
            let rest = attention_blocked_heads(
                &q,
                n_q,
                &k,
                &v,
                n_kv,
                n_head,
                head_dim,
                0.5,
                None,
                int8,
                1..n_head,
            );
            for (i, w) in whole.iter().enumerate() {
                let head = (i % dim) / head_dim;
                let (mine, other) = if head == 0 {
                    (&first, &rest)
                } else {
                    (&rest, &first)
                };
                assert_eq!(mine[i].to_bits(), w.to_bits(), "int8 {int8} at {i}");
                assert_eq!(
                    other[i], 0.0,
                    "int8 {int8} at {i}: a column outside the range"
                );
            }
        }
    }

    /// A query block's `int8` scores computed a key chunk at a time are the
    /// whole-row computation's, to the bit — scores and row maxima — at any
    /// chunk: one pair, three, more than the keys; an odd key count; a
    /// partial last quad; and causal limits that end inside a chunk.
    #[test]
    fn int8_scores_by_key_chunks_are_the_whole_rows() {
        use crate::engine::vecdot::PairedI8;
        let (nq, n, head_dim) = (7usize, 29usize, 16usize);
        let unit = |i: usize, salt: usize| ((i * 31 + salt * 17) % 29) as f32 / 14.0 - 1.0;
        let qv: Vec<f32> = (0..8 * head_dim).map(|i| unit(i, 1)).collect();
        let kv: Vec<f32> = (0..n * head_dim).map(|i| unit(i, 2)).collect();
        let qs = PairedI8::quantize(8, head_dim, |r| &qv[r * head_dim..(r + 1) * head_dim]);
        let kq = PairedI8::quantize(n, head_dim, |r| &kv[r * head_dim..(r + 1) * head_dim]);
        let limits: Vec<usize> = (0..nq).map(|r| 3 + r * 4).collect();
        for lim in [None, Some(&limits[..])] {
            let run = |chunk: usize| {
                let mut iscores = vec![0i32; 8 * chunk.max(n)];
                let mut scores = vec![0f32; nq * n];
                let mut row_max = vec![f32::NEG_INFINITY; nq];
                let rows = BlockRows {
                    q0: 0,
                    nq,
                    n,
                    scale: 0.25,
                    limits: lim,
                };
                int8_block_scores(
                    &qs,
                    &kq,
                    rows,
                    chunk,
                    &mut iscores,
                    &mut scores,
                    &mut row_max,
                );
                (scores, row_max)
            };
            let (whole, whole_max) = run(n.div_ceil(2));
            for chunk in [1usize, 3, 64] {
                let (got, got_max) = run(chunk);
                for r in 0..nq {
                    let limit = lim.map_or(n, |l| l[r]);
                    for j in 0..limit {
                        assert_eq!(
                            got[r * n + j].to_bits(),
                            whole[r * n + j].to_bits(),
                            "chunk {chunk} row {r} key {j}"
                        );
                    }
                    assert_eq!(
                        got_max[r].to_bits(),
                        whole_max[r].to_bits(),
                        "chunk {chunk} row {r}"
                    );
                }
            }
            assert!(whole_max.iter().all(|m| m.is_finite()));
        }
    }

    /// A step's attention cut into query chunks — what a step overlapping
    /// the device runs — is the whole picture's attention to the bit, on
    /// both score paths: a query's row depends on its own query and the
    /// keys, never on which queries share its block. Chunks that are no
    /// multiple of the query block, nor of four, and keys past the queries
    /// (the prompt's prefix).
    #[test]
    fn attention_by_query_chunks_is_the_whole_attention() {
        let n_q = ATTN_QUERIES * 3 + 7;
        let n_kv = n_q + 11;
        let (n_head, head_dim) = (2, 16);
        let dim = n_head * head_dim;
        let unit = |i: usize, salt: usize| ((i * 31 + salt * 17) % 29) as f32 / 14.0 - 1.0;
        let q: Vec<f32> = (0..n_q * dim).map(|i| unit(i, 1)).collect();
        let k: Vec<f32> = (0..n_kv * dim)
            .map(|i| unit(i, 2) + 3.0 * ((i % 7) == 0) as u8 as f32)
            .collect();
        let v: Vec<f32> = (0..n_kv * dim).map(|i| unit(i, 3)).collect();
        for int8 in [false, true] {
            let whole = attention_blocked(&q, n_q, &k, &v, n_kv, n_head, head_dim, 0.5, None, int8);
            for chunk in [37usize, 101] {
                let mut got = Vec::new();
                for qc in q.chunks(chunk * dim) {
                    got.extend(attention_blocked(
                        qc,
                        qc.len() / dim,
                        &k,
                        &v,
                        n_kv,
                        n_head,
                        head_dim,
                        0.5,
                        None,
                        int8,
                    ));
                }
                assert_eq!(got.len(), whole.len());
                let differ = got
                    .iter()
                    .zip(&whole)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert_eq!(differ, None, "int8 {int8} chunk {chunk}");
            }
        }
    }

    /// One query identical to one key attends almost entirely to it under a
    /// large scale, and a uniform key set averages the values.
    #[test]
    fn joint_attention_matches_a_direct_softmax() {
        let n = 3;
        let (n_head, head_dim) = (2, 4);
        let dim = n_head * head_dim;
        let q: Vec<f32> = (0..n * dim).map(|i| (i % 7) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..n * dim).map(|i| (i % 5) as f32 * 0.2).collect();
        let v: Vec<f32> = (0..n * dim).map(|i| (i % 3) as f32).collect();
        let out = joint_attention(&q, &k, &v, n, n_head, head_dim, 0.5);
        for i in 0..n {
            for h in 0..n_head {
                let qh = &q[i * dim + h * head_dim..i * dim + (h + 1) * head_dim];
                let scores: Vec<f32> = (0..n)
                    .map(|j| {
                        let kh = &k[j * dim + h * head_dim..j * dim + (h + 1) * head_dim];
                        qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * 0.5
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let expect: f32 = (0..n)
                        .map(|j| exps[j] / sum * v[j * dim + h * head_dim + d])
                        .sum();
                    let got = out[i * dim + h * head_dim + d];
                    assert!(
                        (got - expect).abs() < 1e-5,
                        "{i} {h} {d}: {got} vs {expect}"
                    );
                }
            }
        }
    }
}

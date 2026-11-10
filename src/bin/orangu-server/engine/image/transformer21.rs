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

//! The `qwen_image_2_1` diffusion transformer — diffusers'
//! `QwenImage21Transformer2DModel`, read from the GGUF the ComfyUI tooling
//! writes (`unsloth/Qwen-Image-2.1-GGUF`: no metadata at all, every tensor
//! under `model.diffusion_model.`, which the loader strips).
//!
//! Where Qwen-Image ([`super::transformer`]) is dual-stream, 2.1 is
//! **single-stream**: the prompt's tokens and the picture's are one
//! sequence — `[prompt | picture]` — through 32 identical blocks of
//! `LayerNorm · (1 + scale)`, attention, and a SwiGLU feed-forward, each
//! behind a `tanh`-gated residual. There is no shift and no per-block
//! modulation: one shared `modulation` linear turns the timestep embedding
//! into the four vectors (`scale, gate` for attention and for the
//! feed-forward) every block reads.
//!
//! Two properties of the model make the prompt a **prefix computed once per
//! picture** rather than once per step, which is diffusers' own KV cache:
//!
//! - *Block-causal attention.* The prompt is causal — a token sees only the
//!   ones before it — and the picture sees everything. Nothing the picture
//!   does reaches the prompt.
//! - *`causal_condition`.* The prompt's tokens are modulated from `t = 0`,
//!   not from the step's timestep, so they do not change between steps.
//!
//! So [`QwenImage21Transformer::prefill`] runs the prompt through every
//! block once and keeps each block's keys and values; a step
//! ([`QwenImage21Transformer::forward`]) then runs only the picture's
//! tokens, attending over `[cached prompt keys | own keys]`.
//!
//! Position is a three-axis rotary embedding like Qwen-Image's (8 + 28 + 28
//! complex pairs of the 128-wide head), laid out the other way round: the
//! prompt's token `i` sits at `i` on all three axes, and the picture at
//! frame `n_prompt` with rows and columns centred on zero.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::transformer::{
    StageClock, Stages, TIMESTEP_FREQUENCIES, TransformerConfig, device_head_attention,
    head_rms_norm, layer_norm_into, silu_inplace, step_attention, step_attention_heads,
    timestep_embedding,
};

/// The feed-forward's two input projections: ComfyUI's checkpoints fuse
/// them into one `gate_up` (gate first), diffusers' keep them apart.
enum MlpIn {
    /// `w` and its two halves as row views of the same bytes: the host runs
    /// `w` in one matmul, a device's fused feed-forward takes the halves.
    Fused {
        w: QuantMatrix,
        gate: QuantMatrix,
        up: QuantMatrix,
    },
    Split {
        gate: QuantMatrix,
        up: QuantMatrix,
    },
}

struct Block {
    to_q: QuantMatrix,
    to_k: QuantMatrix,
    to_v: QuantMatrix,
    to_out: QuantMatrix,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    mlp_in: MlpIn,
    mlp_out: QuantMatrix,
}

pub struct QwenImage21Transformer {
    backend: Arc<dyn Backend>,
    pub config: TransformerConfig,
    /// The feed-forward's inner width (`3 × dim`).
    mlp_dim: usize,
    /// See [`super::transformer::QwenImageTransformer`]'s field of the name.
    stages: Mutex<Stages>,
    /// The last pass's modulation, keyed by its sigma: under guidance the
    /// negative prompt's pass follows the positive one at the same sigma.
    modulation: Mutex<Option<Modulation>>,
    /// The card's heads' keys and values ([`Self::device_attention`]),
    /// kept from block to block and pass to pass so its device mirror is
    /// allocated once and only refilled — on a card at its driver budget,
    /// allocating it each pass made the driver move buffers mid-step —
    /// until [`Self::release_pass_scratch`] before the VAE.
    head_cache: Mutex<Option<crate::engine::kv_cache::KvCache>>,
    /// How many heads the card takes, tuned pass by pass ([`HeadTuner`]).
    head_tuner: Mutex<HeadTuner>,
    img_in: QuantMatrix,
    txt_norm: Vec<f32>,
    txt_in: QuantMatrix,
    txt_out: QuantMatrix,
    time_in: QuantMatrix,
    time_out: QuantMatrix,
    modulation_proj: QuantMatrix,
    norm_out: QuantMatrix,
    proj_out: QuantMatrix,
    blocks: Vec<Block>,
    /// The blocks' linears requantized to per-row `int8` for the 8 × 8
    /// `smmla` tile (`vecdot::RowI8`), keyed by
    /// the weight's bytes — see [`ImageWeights`].
    rowi8: Option<std::collections::HashMap<usize, crate::engine::vecdot::RowI8>>,
}

/// The timestep's vectors for one sigma: `[scale1 | gate1 | scale2 |
/// gate2]` every block reads, and the final norm's scale.
struct Modulation {
    sigma: f32,
    blocks: Vec<f32>,
    out: Vec<f32>,
}

/// A prompt, run through every block once: each block's rotated keys and
/// its values for the prompt's tokens, `[n_txt, dim]` each. What a step's
/// picture tokens attend to beside themselves.
pub struct Prefix {
    /// Tokens in the prefix — the prompt's, and a reference picture's.
    pub n_txt: usize,
    /// The rotary position the picture being drawn sits at: past the
    /// prompt's last token, where a reference picture counts as the larger
    /// of its sides rather than its token count.
    next_position: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// One run of the prompt prefix, in order.
pub enum Segment<'a> {
    /// Text-encoder hidden states, `[n, txt_dim]`: causal.
    Text(&'a [f32]),
    /// A reference picture's VAE latents, `[rows * cols, 64]` row-major —
    /// placed where the encoder read the picture, bidirectional within
    /// itself and causal towards everything else.
    Picture {
        latents: &'a [f32],
        rows: usize,
        cols: usize,
    },
}

/// What one step's pass is given.
pub struct ForwardInput<'a> {
    /// `[n_img, 64]` latent tokens, row-major over the `(rows, cols)` grid —
    /// one per latent pixel, unpatched.
    pub img: &'a [f32],
    pub grid: (usize, usize),
    pub prefix: &'a Prefix,
    /// The noise level in `[0, 1]`.
    pub sigma: f32,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl QwenImage21Transformer {
    /// Reads the checkpoint's dimensions from its tensors.
    pub fn config_from(loaded: &LoadedModel) -> Result<TransformerConfig> {
        let (_, img_in) = loaded.tensor_dims("img_in.weight")?;
        let (_, txt_in) = loaded.tensor_dims("txt_in.in_layer.weight")?;
        let (_, norm_q) = loaded.tensor_dims("transformer_blocks.0.attn.norm_q.weight")?;
        let in_channels = img_in[0] as usize;
        let dim = img_in[1] as usize;
        let txt_dim = txt_in[0] as usize;
        let head_dim = norm_q[0] as usize;
        ensure!(
            head_dim > 0 && dim.is_multiple_of(head_dim),
            "qwen_image_2_1: stream width {dim} is not a multiple of the head width {head_dim}"
        );
        let n_layer = (0..)
            .take_while(|i| loaded.has_tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")))
            .count();
        ensure!(
            n_layer > 0,
            "qwen_image_2_1: no transformer_blocks.N tensors"
        );
        // `axes_dims_rope = (16, 56, 56)`, as for Qwen-Image.
        let frame = head_dim / 8;
        let spatial = (head_dim - frame) / 2;
        ensure!(
            frame + 2 * spatial == head_dim && frame.is_multiple_of(2) && spatial.is_multiple_of(2),
            "qwen_image_2_1: head width {head_dim} does not split into the rotary axes"
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

    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = Self::config_from(loaded)?;
        let c = &config;
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(&format!("{name}.weight"))
                .with_context(|| format!("qwen_image_2_1: loading {name}"))
        };
        let vector = |name: &str, len: usize| -> Result<Vec<f32>> {
            let (values, _) = loaded
                .tensor(name)
                .with_context(|| format!("qwen_image_2_1: loading {name}"))?;
            ensure!(
                values.len() == len,
                "qwen_image_2_1: {name} has {} values, expected {len}",
                values.len()
            );
            Ok(values)
        };
        let mut blocks = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            let p = format!("transformer_blocks.{i}");
            let mlp_in = if loaded.has_tensor(&format!("{p}.img_mlp.gate_up.weight")) {
                let w = matrix(&format!("{p}.img_mlp.gate_up"))?;
                let m = w.out_dim / 2;
                MlpIn::Fused {
                    gate: w.rows(0, m),
                    up: w.rows(m, w.out_dim - m),
                    w,
                }
            } else {
                MlpIn::Split {
                    gate: matrix(&format!("{p}.img_mlp.gate_layer"))?,
                    up: matrix(&format!("{p}.img_mlp.proj"))?,
                }
            };
            blocks.push(Block {
                to_q: matrix(&format!("{p}.attn.to_q"))?,
                to_k: matrix(&format!("{p}.attn.to_k"))?,
                to_v: matrix(&format!("{p}.attn.to_v"))?,
                to_out: matrix(&format!("{p}.attn.to_out.0"))?,
                norm_q: vector(&format!("{p}.attn.norm_q.weight"), c.head_dim)?,
                norm_k: vector(&format!("{p}.attn.norm_k.weight"), c.head_dim)?,
                mlp_in,
                mlp_out: matrix(&format!("{p}.img_mlp.out"))?,
            });
        }
        let mlp_dim = blocks[0].mlp_out.in_dim;
        let copy_bytes: u64 = blocks
            .iter()
            .flat_map(block_weights)
            .map(|w| (w.in_dim * w.out_dim) as u64)
            .sum();
        // The per-row copy is the CPU's kernel: a transformer on a device
        // runs the device's own, and a copy there would only be 7 GB of RAM
        // its linears never read.
        let on_cpu = backend.is_cpu();
        if !on_cpu && configured_weights() == ImageWeights::Int8 {
            log::info!(
                "orangu-server: [image] image_weights = int8 applies to a transformer on the \
                 CPU; this one runs on the device, which keeps the file's weights"
            );
        }
        let rowi8 = (on_cpu && use_rowi8(copy_bytes)).then(|| {
            let started = std::time::Instant::now();
            let mut map = std::collections::HashMap::new();
            for block in &blocks {
                for w in block_weights(block) {
                    let rows =
                        crate::engine::vecdot::RowI8::quantize(w.out_dim, w.in_dim, |o| w.row(o));
                    map.insert(w.raw_bytes().as_ptr() as usize, rows);
                }
            }
            if !crate::engine::vecdot::have_rowi8_tile() {
                log::warn!(
                    "orangu-server: [image] image_weights = int8 on a core without an int8 \
                     tile (i8mm or AVX2) runs the scalar one, far slower than \
                     image_weights = file"
                );
            }
            let bytes: usize = map.values().map(|r| r.bytes()).sum();
            log::info!(
                "orangu-server: [image] the blocks' linears as per-row int8 ({:.1} GB, \
                 image_weights = file keeps the file's) in {:.0} s",
                bytes as f64 / 1e9,
                started.elapsed().as_secs_f64()
            );
            map
        });
        let model = Self {
            backend,
            mlp_dim,
            stages: Mutex::new(Stages::default()),
            modulation: Mutex::new(None),
            head_cache: Mutex::new(None),
            head_tuner: Mutex::new(HeadTuner::new()),
            img_in: matrix("img_in")?,
            txt_norm: vector("txt_in.text_norm.weight", c.txt_dim)?,
            txt_in: matrix("txt_in.in_layer")?,
            txt_out: matrix("txt_in.out_layer")?,
            time_in: matrix("time_text_embed.timestep_embedder.linear_1")?,
            time_out: matrix("time_text_embed.timestep_embedder.linear_2")?,
            modulation_proj: matrix("modulation.1")?,
            norm_out: matrix("norm_out.linear")?,
            proj_out: matrix("proj_out")?,
            blocks,
            rowi8,
            config,
        };
        let c = &model.config;
        ensure!(
            model.time_in.in_dim == TIMESTEP_FREQUENCIES
                && model.time_out.out_dim == c.dim
                && model.modulation_proj.out_dim == 4 * c.dim
                && model.norm_out.out_dim == c.dim
                && model.txt_out.out_dim == c.dim
                && model.proj_out.out_dim == c.in_channels,
            "qwen_image_2_1: the embedding and output projections do not match the stream width"
        );
        for (i, block) in model.blocks.iter().enumerate() {
            let mlp_ok = match &block.mlp_in {
                MlpIn::Fused { w, .. } => w.in_dim == c.dim && w.out_dim == 2 * model.mlp_dim,
                MlpIn::Split { gate, up } => {
                    gate.out_dim == model.mlp_dim && up.out_dim == model.mlp_dim
                }
            };
            ensure!(
                block.to_q.out_dim == c.dim
                    && block.to_out.out_dim == c.dim
                    && block.mlp_out.out_dim == c.dim
                    && mlp_ok,
                "qwen_image_2_1: transformer_blocks.{i} does not match the stream width {}",
                c.dim
            );
        }
        model.plan_streaming();
        Ok(model)
    }

    /// On a card smaller than the transformer: the leading blocks stay
    /// resident and the rest stream (`VulkanBackend::stream_weights`), each
    /// call group's weights crossing the bus once into a region of two
    /// blocks rather than living in host memory, where every tile of every
    /// kernel would read them across the bus. Planned against what the
    /// driver says the card has free now, less the region and
    /// [`STREAM_MARGIN_BYTES`] for the activations and the calls' own
    /// regions. Nothing on the host, with the per-row `int8` copy, when the
    /// blocks fit, or with `ORANGU_IMAGE_STREAM=0`.
    fn plan_streaming(&self) {
        if self.rowi8.is_some()
            || !crate::engine::env::flag_on_unless_disabled("ORANGU_IMAGE_STREAM")
        {
            return;
        }
        let Some(device) = self.blocks.first().map(|b| b.to_q.device()) else {
            return;
        };
        let Some(vulkan) = self.backend.as_wgpu_on(device) else {
            return;
        };
        let Some((budget, usage)) = vulkan.device_local_budget() else {
            return;
        };
        let bytes = |b: &Block| -> u64 {
            device_weights(b)
                .iter()
                .map(|w| w.raw_bytes().len() as u64)
                .sum()
        };
        let sizes: Vec<u64> = self.blocks.iter().map(bytes).collect();
        let region = 2 * sizes.iter().copied().max().unwrap_or(0);
        let room = budget
            .saturating_sub(usage)
            .saturating_sub(region + STREAM_MARGIN_BYTES);
        let mut placed = 0u64;
        let resident = sizes
            .iter()
            .take_while(|&&b| {
                placed += b;
                placed <= room
            })
            .count();
        if resident == self.blocks.len() {
            return;
        }
        let streamed: Vec<&QuantMatrix> = self.blocks[resident..]
            .iter()
            .flat_map(device_weights)
            .collect();
        vulkan.stream_weights(&streamed, region);
        log::info!(
            "orangu-server: [image] transformer blocks 0–{} resident on the card, {}–{} streamed \
             through a {} region ({} free)",
            resident.saturating_sub(1),
            resident,
            self.blocks.len() - 1,
            orangu::format::format_bytes(region),
            orangu::format::format_bytes(budget.saturating_sub(usage)),
        );
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

    fn linear(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * w.in_dim);
        if let Some(rows) = self
            .rowi8
            .as_ref()
            .and_then(|m| m.get(&(w.raw_bytes().as_ptr() as usize)))
        {
            return crate::engine::vecdot::matmul_rowi8(x, n, rows);
        }
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }

    /// A block's query, key and value projections of the same rows. On a
    /// device they are one call (`Backend::matmul_batch`), which stages the
    /// shared input once and runs the three in one sequence of submissions,
    /// one wait and one readback, rather than three of each.
    fn qkv(&self, x: &[f32], n: usize, block: &Block) -> [Vec<f32>; 3] {
        let weights = [&block.to_q, &block.to_k, &block.to_v];
        if self.rowi8.is_some() {
            return weights.map(|w| self.linear(x, n, w));
        }
        let ops = weights.map(|w| MatmulOp { x, n_tokens: n, w });
        let mut results = self.backend.matmul_batch(&ops).into_iter();
        let mut next = || results.next().expect("three ops in, three results out");
        [next(), next(), next()]
    }

    /// The timestep's modulation vectors, from the cache when the last pass
    /// was at this sigma.
    fn modulation_at(&self, sigma: f32) -> Modulation {
        let cached = self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .filter(|m| m.sigma == sigma);
        cached.unwrap_or_else(|| {
            // `silu(time_text_embed(t))`, which both the shared modulation
            // and the final norm read.
            let mut temb = self.linear(&timestep_embedding(sigma), 1, &self.time_in);
            silu_inplace(&mut temb);
            let mut temb = self.linear(&temb, 1, &self.time_out);
            silu_inplace(&mut temb);
            Modulation {
                sigma,
                blocks: self.linear(&temb, 1, &self.modulation_proj),
                out: self.linear(&temb, 1, &self.norm_out),
            }
        })
    }

    /// Runs the prompt's hidden states (`[n_txt, txt_dim]`, the text
    /// encoder's last layer before its final norm) through every block at
    /// `t = 0` under the causal mask, keeping each block's keys and values.
    pub fn prefill(
        &self,
        txt: &[f32],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Prefix> {
        self.prefill_segments(&[Segment::Text(txt)], cancel)
    }

    /// [`prefill`](Self::prefill) of a prompt interleaved with reference
    /// pictures — diffusers' joint sequence for editing: each picture's
    /// latent tokens stand where the text encoder read the picture, under
    /// the block-causal mask (`q >= k`, or the same picture).
    pub fn prefill_segments(
        &self,
        segments: &[Segment<'_>],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Prefix> {
        let c = &self.config;
        let mut clock = StageClock::start();
        // `txt_in` — a zero-centred RMSNorm (the checkpoint stores `scale -
        // 1`), then `Linear · GELU(tanh) · Linear` — for text, `img_in` for
        // a picture's latents.
        let weight: Vec<f32> = self.txt_norm.iter().map(|w| w + 1.0).collect();
        let mut normed = Vec::new();
        let mut x = Vec::new();
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut limits: Vec<usize> = Vec::new();
        let mut position = 0usize;
        for segment in segments {
            let start = limits.len();
            match *segment {
                Segment::Text(txt) => {
                    ensure!(
                        !txt.is_empty() && txt.len().is_multiple_of(c.txt_dim),
                        "qwen_image_2_1: text hidden states are not rows of {}",
                        c.txt_dim
                    );
                    let n = txt.len() / c.txt_dim;
                    tensor::rmsnorm_into(&mut normed, txt, &weight, n, c.txt_dim, c.eps);
                    let mut h = self.linear(&normed, n, &self.txt_in);
                    tensor::gelu_inplace(&mut h);
                    x.extend(self.linear(&h, n, &self.txt_out));
                    for i in 0..n {
                        positions.push([(position + i) as f64; 3]);
                        limits.push(start + i + 1);
                    }
                    position += n;
                }
                Segment::Picture {
                    latents,
                    rows,
                    cols,
                } => {
                    let n = rows * cols;
                    ensure!(
                        n > 0 && latents.len() == n * c.in_channels,
                        "qwen_image_2_1: a reference picture's latents are not {rows}x{cols} \
                         tokens of {}",
                        c.in_channels
                    );
                    x.extend(self.linear(latents, n, &self.img_in));
                    for r in 0..rows {
                        let row = r as f64 - (rows - rows / 2) as f64;
                        for q in 0..cols {
                            let col = q as f64 - (cols - cols / 2) as f64;
                            positions.push([position as f64, row, col]);
                        }
                    }
                    limits.extend(std::iter::repeat_n(start + n, n));
                    position += rows.max(cols);
                }
            }
        }
        let n = limits.len();
        ensure!(n > 0, "qwen_image_2_1: an empty prompt");
        let modulation = self.modulation_at(0.0);
        let [scale1, gate1, scale2, gate2] = split4(&modulation.blocks, c.dim);
        let (scale1, gate1, scale2, gate2) = (
            scale1.to_vec(),
            tanh_of(gate1),
            scale2.to_vec(),
            tanh_of(gate2),
        );
        *self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(modulation);
        let rope = rope_angles(c, &positions);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        clock.lap(|s| &mut s.other);

        let mut keys = Vec::with_capacity(c.n_layer);
        let mut values = Vec::with_capacity(c.n_layer);
        for (bi, block) in self.blocks.iter().enumerate() {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            layer_norm_into(&mut normed, &x, c.dim, c.eps);
            scale_inplace(&mut normed, &scale1, c.dim);
            clock.lap(|s| &mut s.other);
            let [mut q, mut k, v] = self.qkv(&normed, n, block);
            clock.lap(|s| &mut s.qkv);
            head_rms_norm(&mut q, &block.norm_q, c.head_dim, c.eps);
            head_rms_norm(&mut k, &block.norm_k, c.head_dim, c.eps);
            apply_rope(&mut q, c, &rope);
            apply_rope(&mut k, c, &rope);
            let last = bi + 1 == self.blocks.len();
            // The last block's prompt output feeds nothing: only its keys
            // and values are wanted.
            if !last {
                let attn =
                    step_attention(&q, n, &k, &v, n, c.n_head, c.head_dim, scale, Some(&limits));
                clock.lap(|s| &mut s.attention);
                let out = self.linear(&attn, n, &block.to_out);
                clock.lap(|s| &mut s.out);
                gated_add(&mut x, &out, &gate1, c.dim);
                layer_norm_into(&mut normed, &x, c.dim, c.eps);
                scale_inplace(&mut normed, &scale2, c.dim);
                clock.lap(|s| &mut s.other);
                let mlp = self.mlp(block, &normed, n);
                clock.lap(|s| &mut s.mlp);
                gated_add(&mut x, &mlp, &gate2, c.dim);
            }
            keys.push(k);
            values.push(v);
        }
        clock.lap(|s| &mut s.other);
        self.stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .add(&clock.stages);
        Ok(Prefix {
            n_txt: n,
            next_position: position,
            k: keys,
            v: values,
        })
    }

    /// Drops what the steps keep on the card between passes (the heads'
    /// cache and its mirror), for the VAE's decode, which follows on the
    /// same card.
    pub fn release_pass_scratch(&self) {
        *self
            .head_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// How many of a block's heads (the last ones) the card takes in an
    /// overlapped pass: none unless `ORANGU_IMAGE_DEVICE_HEADS` asks
    /// ([`fixed_device_heads`]: a count, or `auto` for [`HeadTuner`] to pick
    /// it from the passes before). None without a Vulkan card holding the
    /// block, and at most half of the block's heads.
    fn device_heads(&self, block: &Block) -> usize {
        if self.backend.as_wgpu_on(block.to_q.device()).is_none() {
            return 0;
        }
        let n = fixed_device_heads().unwrap_or_else(|| {
            self.head_tuner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .heads
        });
        n.min(self.config.n_head / 2)
    }

    /// The attention of the heads `heads` on the card holding `block`
    /// (`VulkanBackend::gpu_attention_prefill`, non-causal): their columns
    /// of the queries, keys and values gathered, `[n_q][heads · head_dim]`
    /// back.
    #[allow(clippy::too_many_arguments)]
    fn device_attention(
        &self,
        block: &Block,
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        n_q: usize,
        n_kv: usize,
        heads: std::ops::Range<usize>,
        scale: f32,
    ) -> Vec<f32> {
        let vulkan = self
            .backend
            .as_wgpu_on(block.to_q.device())
            .expect("device_heads saw the card");
        let mut held = self
            .head_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        device_head_attention(
            vulkan,
            &mut held,
            q,
            keys,
            values,
            (n_q, n_kv),
            (self.config.dim, self.config.head_dim),
            heads,
            scale,
        )
    }

    /// [`Self::mlp`] on the card that holds the block, as one submission
    /// (`VulkanBackend::fused_ffn_prefill`): the gate and up projections,
    /// the SwiGLU and the down projection with the `[n, mlp_dim]`
    /// intermediate never leaving the device. `None` — the host-orchestrated
    /// form — off a Vulkan device, with the per-row `int8` copy, or with
    /// `ORANGU_IMAGE_FUSED_FFN=0`.
    fn device_mlp(&self, block: &Block, x: &[f32], n: usize) -> Option<Vec<f32>> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON
            .get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_IMAGE_FUSED_FFN"));
        if !on || self.rowi8.is_some() {
            return None;
        }
        let vulkan = self.backend.as_wgpu_on(block.mlp_out.device())?;
        let (gate, up) = match &block.mlp_in {
            MlpIn::Fused { gate, up, .. } | MlpIn::Split { gate, up } => (gate, up),
        };
        vulkan.fused_ffn_prefill(
            x,
            n,
            gate,
            up,
            &block.mlp_out,
            crate::engine::backend::vulkan::FfnActivation::Swiglu,
            None,
        )
    }

    /// The query chunks, in tokens, when a step overlaps its attention with
    /// the device ([`Self::forward`]), or `None` to run each block's stages
    /// one after the other. Overlap needs the linears off the host — a
    /// device backend, no per-row `int8` copy — and pays from about a
    /// thousand tokens a chunk: by default four chunks, none under
    /// [`OVERLAP_MIN_CHUNK`] tokens, tapered — the first and the last an
    /// eighth of the picture each, the two between the rest — so the lane
    /// starts on a block sooner and the host waits less on its last tail.
    /// `ORANGU_IMAGE_OVERLAP` sets even chunks of that many tokens, `0`
    /// turns the overlap off; `ORANGU_IMAGE_OVERLAP_TAPER=0` keeps the
    /// default four chunks even.
    fn overlap_plan(&self, n_img: usize) -> Option<Vec<usize>> {
        if self.backend.is_cpu() || self.rowi8.is_some() {
            return None;
        }
        let configured = crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_OVERLAP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        let chunk = match configured {
            Some(0) => return None,
            Some(tokens) => tokens,
            None => n_img.div_ceil(OVERLAP_CHUNKS),
        };
        if chunk < OVERLAP_MIN_CHUNK || chunk >= n_img {
            return None;
        }
        let taper = configured.is_none()
            && crate::engine::env::flag_on_unless_disabled("ORANGU_IMAGE_OVERLAP_TAPER");
        Some(if taper {
            tapered_chunks(n_img)
        } else {
            even_chunks(n_img, chunk)
        })
    }

    /// One block's per-token tail over `x`'s rows, given their attention:
    /// `x += gate1 · to_out(attn)`, then `x += gate2 · mlp(norm(x) · scale2)`
    /// — what [`Self::forward`] runs on the whole picture, for a chunk.
    /// Returns the time in the output projection's and the feed-forward's
    /// calls.
    fn block_tail(
        &self,
        block: &Block,
        gates: &Gates<'_>,
        x: &mut [f32],
        attn: Vec<f32>,
    ) -> (Duration, Duration) {
        let c = &self.config;
        let n = x.len() / c.dim;
        let started = Instant::now();
        let out = self.linear(&attn, n, &block.to_out);
        let out_call = started.elapsed();
        self.backend.recycle(attn);
        gated_add(x, &out, gates.gate1, c.dim);
        self.backend.recycle(out);
        let mut normed = self.backend.take_scratch(x.len());
        layer_norm_into(&mut normed, x, c.dim, c.eps);
        scale_inplace(&mut normed, gates.scale2, c.dim);
        let started = Instant::now();
        let mlp = self.mlp(block, &normed, n);
        let mlp_call = started.elapsed();
        self.backend.recycle(normed);
        gated_add(x, &mlp, gates.gate2, c.dim);
        self.backend.recycle(mlp);
        (out_call, mlp_call)
    }

    /// `out(silu(gate) · up)`.
    fn mlp(&self, block: &Block, x: &[f32], n: usize) -> Vec<f32> {
        if let Some(out) = self.device_mlp(block, x, n) {
            return out;
        }
        let m = self.mlp_dim;
        let mut h = self.backend.take_scratch(n * m);
        h.resize(n * m, 0.0);
        match &block.mlp_in {
            MlpIn::Fused { w, .. } => {
                let gate_up = self.linear(x, n, w);
                h.par_chunks_mut(m)
                    .zip(gate_up.par_chunks(2 * m))
                    .for_each(|(out, row)| {
                        let (gate, up) = row.split_at(m);
                        for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
                            *o = tensor::silu(*g) * u;
                        }
                    });
                self.backend.recycle(gate_up);
            }
            MlpIn::Split { gate, up } => {
                let mut results = self.backend.matmul_batch(&[
                    MatmulOp {
                        x,
                        n_tokens: n,
                        w: gate,
                    },
                    MatmulOp {
                        x,
                        n_tokens: n,
                        w: up,
                    },
                ]);
                let up = results.pop().expect("two results");
                let gate = results.pop().expect("two results");
                h.par_chunks_mut(m)
                    .zip(gate.par_chunks(m).zip(up.par_chunks(m)))
                    .for_each(|(out, (g, u))| {
                        for ((o, g), u) in out.iter_mut().zip(g).zip(u) {
                            *o = tensor::silu(*g) * u;
                        }
                    });
                self.backend.recycle(gate);
                self.backend.recycle(up);
            }
        }
        let out = self.linear(&h, n, &block.mlp_out);
        self.backend.recycle(h);
        out
    }

    /// One velocity prediction for the picture's tokens: `[n_img, 64]`, the
    /// layout the latents came in.
    pub fn forward(&self, input: &ForwardInput<'_>) -> Result<Vec<f32>> {
        let c = &self.config;
        let (rows, cols) = input.grid;
        let n_img = rows * cols;
        ensure!(
            input.img.len() == n_img * c.in_channels,
            "qwen_image_2_1: {} latent values for a {rows}x{cols} grid of {}-wide tokens",
            input.img.len(),
            c.in_channels
        );
        let prefix = input.prefix;
        ensure!(
            prefix.k.len() == c.n_layer,
            "qwen_image_2_1: the prompt prefix was built for another model"
        );
        let n_txt = prefix.n_txt;
        let n_kv = n_txt + n_img;
        let mut clock = StageClock::start();

        let mut x = self.linear(input.img, n_img, &self.img_in);
        clock.lap(|s| &mut s.other);
        let modulation = self.modulation_at(input.sigma);
        clock.lap(|s| &mut s.modulation);
        let [scale1, gate1, scale2, gate2] = split4(&modulation.blocks, c.dim);
        let (gate1, gate2) = (tanh_of(gate1), tanh_of(gate2));

        // Rows and columns centred on zero, the frame just past the prompt.
        let mut positions = Vec::with_capacity(n_img);
        for r in 0..rows {
            let row = r as f64 - (rows - rows / 2) as f64;
            for q in 0..cols {
                let col = q as f64 - (cols - cols / 2) as f64;
                positions.push([prefix.next_position as f64, row, col]);
            }
        }
        let rope = rope_angles(c, &positions);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut normed = Vec::new();
        let mut keys = Vec::with_capacity(n_kv * c.dim);
        let mut values = Vec::with_capacity(n_kv * c.dim);
        let overlap = self.overlap_plan(n_img);
        clock.lap(|s| &mut s.other);
        for (bi, block) in self.blocks.iter().enumerate() {
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            layer_norm_into(&mut normed, &x, c.dim, c.eps);
            scale_inplace(&mut normed, scale1, c.dim);
            clock.lap(|s| &mut s.other);
            let [mut q, mut k, v] = self.qkv(&normed, n_img, block);
            clock.lap(|s| &mut s.qkv);
            head_rms_norm(&mut q, &block.norm_q, c.head_dim, c.eps);
            head_rms_norm(&mut k, &block.norm_k, c.head_dim, c.eps);
            apply_rope(&mut q, c, &rope);
            apply_rope(&mut k, c, &rope);
            keys.clear();
            keys.extend_from_slice(&prefix.k[bi]);
            keys.extend_from_slice(&k);
            values.clear();
            values.extend_from_slice(&prefix.v[bi]);
            values.extend_from_slice(&v);
            // Every wide buffer of the block goes back to the backend when
            // done with, for the next one to reuse (`Backend::recycle`).
            self.backend.recycle(k);
            self.backend.recycle(v);
            if let Some(plan) = &overlap {
                // The attention by query chunks on the host, and each
                // finished chunk's tail (`to_out`, residual, norm, MLP) on
                // the device lane meanwhile: a token's tail reads only its
                // own row, and a query's attention only its own query.
                //
                // The last `device_heads` heads' attention runs on the card,
                // first thing in the lane — which is otherwise idle until the
                // first chunk's attention is done — and their columns are
                // filled into each chunk before its tail.
                let gates = Gates {
                    gate1: &gate1,
                    scale2,
                    gate2: &gate2,
                };
                let device_heads = self.device_heads(block);
                let host_heads = 0..c.n_head - device_heads;
                std::thread::scope(|scope| {
                    let (send, receive) = std::sync::mpsc::channel::<Vec<f32>>();
                    let x = &mut x;
                    let (q, keys, values) = (&q, &keys, &values);
                    let lane = scope.spawn(move || {
                        lane_pool().install(move || {
                            let (mut busy, mut idle) = (Duration::ZERO, Duration::ZERO);
                            let (mut out, mut mlp) = (Duration::ZERO, Duration::ZERO);
                            let ran = Instant::now();
                            let device = (device_heads > 0).then(|| {
                                self.device_attention(
                                    block,
                                    q,
                                    keys,
                                    values,
                                    n_img,
                                    n_kv,
                                    c.n_head - device_heads..c.n_head,
                                    scale,
                                )
                            });
                            let heads = ran.elapsed();
                            busy += heads;
                            let width = device_heads * c.head_dim;
                            let column = (c.n_head - device_heads) * c.head_dim;
                            let mut t0 = 0;
                            let mut rest: &mut [f32] = x;
                            let mut chunks = plan.iter().map(move |&tokens| {
                                let (rows, tail) =
                                    std::mem::take(&mut rest).split_at_mut(tokens * c.dim);
                                rest = tail;
                                rows
                            });
                            loop {
                                let waited = Instant::now();
                                let Ok(attn) = receive.recv() else { break };
                                idle += waited.elapsed();
                                let Some(rows) = chunks.next() else { break };
                                let mut attn = attn;
                                let ran = Instant::now();
                                let n = rows.len() / c.dim;
                                if let Some(dev) = &device {
                                    for t in 0..n {
                                        attn[t * c.dim + column..t * c.dim + column + width]
                                            .copy_from_slice(
                                                &dev[(t0 + t) * width..(t0 + t + 1) * width],
                                            );
                                    }
                                }
                                t0 += n;
                                let (o, m) = self.block_tail(block, &gates, rows, attn);
                                busy += ran.elapsed();
                                out += o;
                                mlp += m;
                            }
                            (busy, idle, out, mlp, heads)
                        })
                    });
                    let attention_started = Instant::now();
                    let mut first = 0;
                    for (ci, &tokens) in plan.iter().enumerate() {
                        let qc = &q[first * c.dim..(first + tokens) * c.dim];
                        first += tokens;
                        let attn = step_attention_heads(
                            qc,
                            qc.len() / c.dim,
                            keys,
                            values,
                            n_kv,
                            c.n_head,
                            c.head_dim,
                            scale,
                            host_heads.clone(),
                        );
                        if ci == 0 {
                            clock.stages.first_chunk += attention_started.elapsed();
                        }
                        if send.send(attn).is_err() {
                            // The lane ended early: its panic is re-raised
                            // by the join below.
                            break;
                        }
                    }
                    drop(send);
                    clock.lap(|s| &mut s.attention);
                    match lane.join() {
                        Ok((busy, idle, out, mlp, heads)) => {
                            clock.stages.lane_heads += heads;
                            clock.stages.lane_busy += busy;
                            clock.stages.lane_idle += idle;
                            clock.stages.lane_out += out;
                            clock.stages.lane_mlp += mlp;
                        }
                        Err(panic) => std::panic::resume_unwind(panic),
                    }
                });
                self.backend.recycle(q);
                clock.lap(|s| &mut s.mlp);
                continue;
            }
            let attn = step_attention(
                &q, n_img, &keys, &values, n_kv, c.n_head, c.head_dim, scale, None,
            );
            self.backend.recycle(q);
            clock.lap(|s| &mut s.attention);
            let out = self.linear(&attn, n_img, &block.to_out);
            self.backend.recycle(attn);
            clock.lap(|s| &mut s.out);
            gated_add(&mut x, &out, &gate1, c.dim);
            self.backend.recycle(out);
            layer_norm_into(&mut normed, &x, c.dim, c.eps);
            scale_inplace(&mut normed, scale2, c.dim);
            clock.lap(|s| &mut s.other);
            let mlp = self.mlp(block, &normed, n_img);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut x, &mlp, &gate2, c.dim);
            self.backend.recycle(mlp);
        }

        // `AdaLayerNormContinuous`, scale only.
        layer_norm_into(&mut normed, &x, c.dim, c.eps);
        scale_inplace(&mut normed, &modulation.out, c.dim);
        let out = self.linear(&normed, n_img, &self.proj_out);
        clock.lap(|s| &mut s.other);
        if overlap.is_some()
            && fixed_device_heads().is_none()
            && self
                .blocks
                .first()
                .is_some_and(|b| self.backend.as_wgpu_on(b.to_q.device()).is_some())
        {
            let mut tuner = self
                .head_tuner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let heads = tuner.heads;
            let slack = tuner.observe(&clock.stages, c.n_head, c.n_head / 2);
            if crate::engine::env::flag_on("ORANGU_IMAGE_HEADS_TRACE") {
                eprintln!(
                    "orangu-server: [image] pass with {heads} heads on the device: window headroom \
                     {:+.2}s, next pass {}",
                    slack, tuner.heads
                );
            }
        }
        // `ORANGU_IMAGE_PASS_TRACE=1`: a line a pass, stamped with the wall
        // clock, with the lane's feed-forward and output projection — to
        // line a slow pass up against what the card was doing then.
        if overlap.is_some() && crate::engine::env::flag_on("ORANGU_IMAGE_PASS_TRACE") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let st = &clock.stages;
            eprintln!(
                "orangu-server: [image] pass ended at {:.1}: attention {:.2}s, feed-forward \
                 {:.2}s, output projection {:.2}s, wait on the last tails {:.2}s, q/k/v {:.2}s",
                now.as_secs_f64(),
                st.attention.as_secs_f64(),
                st.lane_mlp.as_secs_f64(),
                st.lane_out.as_secs_f64(),
                st.mlp.as_secs_f64(),
                st.qkv.as_secs_f64(),
            );
        }
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

/// The chunks a step's attention is cut into when it overlaps the device
/// ([`QwenImage21Transformer::overlap_plan`]).
const OVERLAP_CHUNKS: usize = 4;

/// `n` tokens in chunks of `chunk`, the last what is left.
fn even_chunks(n: usize, chunk: usize) -> Vec<usize> {
    (0..n)
        .step_by(chunk)
        .map(|start| chunk.min(n - start))
        .collect()
}

/// `n` tokens in four chunks: the first and the last an eighth each
/// (whole 64-token tiles), the two between halving the rest.
fn tapered_chunks(n: usize) -> Vec<usize> {
    let end = (n / 8 / 64 * 64).max(64);
    let middle = n - 2 * end;
    let half = middle / 2;
    vec![end, half, middle - half, end]
}

/// The narrowest chunk worth overlapping: below it the device's calls on
/// a chunk cost more than the overlap saves.
const OVERLAP_MIN_CHUNK: usize = 1024;

/// What a transformer leaves free on its card beyond its weights (and,
/// streamed, beyond the region): the activations, the calls' input and
/// output regions, their scratch.
pub const STREAM_MARGIN_BYTES: u64 = 512 << 20;

/// The tensors a block's device calls read: q, k, v, the output
/// projection, the feed-forward's gate and up (row views of a fused
/// tensor, as the device takes them) and down.
fn device_weights(block: &Block) -> Vec<&QuantMatrix> {
    let (gate, up) = match &block.mlp_in {
        MlpIn::Fused { gate, up, .. } | MlpIn::Split { gate, up } => (gate, up),
    };
    vec![
        &block.to_q,
        &block.to_k,
        &block.to_v,
        &block.to_out,
        gate,
        up,
        &block.mlp_out,
    ]
}

/// `ORANGU_IMAGE_DEVICE_HEADS`, read once: `None` for `auto` (the
/// [`HeadTuner`] picks the count), else the count, fixed — unset is `0`.
/// Off by default: with the card's heads on, the lane's feed-forward ran
/// slower and the step no faster and less steady, on the cards measured.
fn fixed_device_heads() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_DEVICE_HEADS") {
            Ok(v) if v.trim().eq_ignore_ascii_case("auto") => None,
            Ok(v) => Some(v.trim().parse().unwrap_or(0)),
            Err(_) => Some(0),
        }
    })
}

/// Picks how many heads the card takes: as many as it finishes while the
/// host computes the block's first query chunk — the lane's only idle time
/// in a block, before any tail reaches the card, so the heads do not run
/// beside the lane's feed-forward. Measured each pass: `a`, the first chunk's host time
/// per host head; `c`, the lane's time per card head (a first pass on the
/// card runs one head to measure it). The count is the largest `h` with
/// `h · c ≤ FILL · a · (n − h)`, kept from one picture to the next.
struct HeadTuner {
    heads: usize,
    /// The last measured lane time per card head, seconds over a pass.
    card: Option<f64>,
}

/// The share of the first chunk's window the card's heads may fill.
const FILL: f64 = 0.9;

impl HeadTuner {
    fn new() -> Self {
        Self {
            heads: 0,
            card: None,
        }
    }

    /// One pass's stages at the current count of a block's `n_head` heads;
    /// sets the count for the next pass, within `0..=most`. Returns the
    /// window's headroom, seconds over the pass: the first chunks' host
    /// time less the card's heads' lane time.
    fn observe(&mut self, pass: &Stages, n_head: usize, most: usize) -> f64 {
        let secs = |d: Duration| d.as_secs_f64();
        let window = secs(pass.first_chunk);
        let headroom = window - secs(pass.lane_heads);
        if self.heads > 0 {
            self.card = Some(secs(pass.lane_heads) / self.heads as f64);
        }
        let Some(c) = self.card else {
            self.heads = 1.min(most);
            return headroom;
        };
        let a = FILL * window / (n_head - self.heads).max(1) as f64;
        let fits = (a * n_head as f64 / (c + a).max(1e-9)).floor();
        self.heads = (fits.max(0.0) as usize).min(most);
        headroom
    }
}

/// A block's modulation for its per-token tail.
struct Gates<'a> {
    gate1: &'a [f32],
    scale2: &'a [f32],
    gate2: &'a [f32],
}

/// The threads the device lane's host work runs on (the residual adds, the
/// norm, the MLP's activation, the device calls' staging and copies) while
/// the attention holds the global pool: a quarter of it, at most four.
pub(super) fn lane_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads((rayon::current_num_threads() / 4).clamp(1, 4))
            .thread_name(|i| format!("image-lane-{i}"))
            .build()
            .expect("the image lane's thread pool")
    })
}

/// How the blocks' linears are held — `[orangu-server].image_weights`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageWeights {
    /// Per-row `int8` when the transformer runs on the CPU, the CPU has an
    /// `int8` tile (`vecdot::have_rowi8_tile`: `i8mm` or `AVX2`) and the
    /// machine has room for it: total memory at least three times the copy
    /// (21 GB for Qwen-Image 2.1's 7 GB).
    #[default]
    Auto,
    /// Per-row `int8` (`vecdot::RowI8`) whenever the transformer runs on the
    /// CPU — on a core with no `int8` tile, the scalar definition.
    Int8,
    /// The file's own weights on the K-quant kernel, and no copy.
    File,
}

impl ImageWeights {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "int8" => Some(Self::Int8),
            "file" => Some(Self::File),
            _ => None,
        }
    }
}

static WEIGHTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The configured [`ImageWeights`], set once by `main` before the pipeline
/// loads.
pub fn set_weights(choice: ImageWeights) {
    WEIGHTS.store(choice as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The configured [`ImageWeights`], `ORANGU_IMAGE_WEIGHTS` (`int8`/`file`)
/// over the configuration for an A/B.
fn configured_weights() -> ImageWeights {
    crate::engine::backend::tuning_profile::env_var("ORANGU_IMAGE_WEIGHTS")
        .ok()
        .and_then(|v| ImageWeights::parse(&v))
        .unwrap_or(match WEIGHTS.load(std::sync::atomic::Ordering::Relaxed) {
            1 => ImageWeights::Int8,
            2 => ImageWeights::File,
            _ => ImageWeights::Auto,
        })
}

/// Whether a transformer on the CPU holds its blocks' `copy_bytes` of
/// per-row `int8` ([`configured_weights`]; under `auto`, when the CPU has an
/// `int8` tile and the memory). The startup calibration asks too, so its
/// CPU half times the kernel a step runs.
pub fn use_rowi8(copy_bytes: u64) -> bool {
    match configured_weights() {
        ImageWeights::Int8 => true,
        ImageWeights::File => false,
        ImageWeights::Auto => {
            crate::engine::vecdot::have_rowi8_tile()
                && orangu::hardware::detect_cpu().total_memory_bytes >= 3 * copy_bytes
        }
    }
}

/// A block's token-wide linears — what [`ImageWeights`] requantizes.
fn block_weights(block: &Block) -> Vec<&QuantMatrix> {
    let mut weights = vec![&block.to_q, &block.to_k, &block.to_v, &block.to_out];
    weights.push(&block.mlp_out);
    match &block.mlp_in {
        MlpIn::Fused { w, .. } => weights.push(w),
        MlpIn::Split { gate, up } => weights.extend([gate, up]),
    }
    weights
}

/// The shared modulation's `[4 * dim]` as `scale1, gate1, scale2, gate2` —
/// diffusers chunks it in two (attention, feed-forward) and each half into
/// `scale, gate`.
fn split4(m: &[f32], dim: usize) -> [&[f32]; 4] {
    debug_assert_eq!(m.len(), 4 * dim);
    std::array::from_fn(|i| &m[i * dim..(i + 1) * dim])
}

fn tanh_of(gate: &[f32]) -> Vec<f32> {
    gate.iter().map(|g| g.tanh()).collect()
}

/// `x * (1 + scale)`, per row.
fn scale_inplace(x: &mut [f32], scale: &[f32], dim: usize) {
    x.par_chunks_mut(dim).for_each(|row| {
        for (v, s) in row.iter_mut().zip(scale) {
            *v *= 1.0 + s;
        }
    });
}

/// `x += gate * y`, per row, `gate` already through its `tanh`.
fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32], dim: usize) {
    x.par_chunks_mut(dim)
        .zip(y.par_chunks(dim))
        .for_each(|(row, add)| {
            for ((v, a), g) in row.iter_mut().zip(add).zip(gate) {
                *v += g * a;
            }
        });
}

/// `(cos, sin)` per complex pair of the head, per position: the frame axis
/// on the first `rope_axes[0]` pairs, rows and columns on the next two
/// runs — diffusers' `QwenImage21Rope`, whose pair `j` of an axis of `d`
/// pairs turns at `10000^(-j/d)` per unit of position.
fn rope_angles(c: &TransformerConfig, positions: &[[f64; 3]]) -> Vec<(f32, f32)> {
    let freqs: Vec<Vec<f64>> = c
        .rope_axes
        .iter()
        .map(|&d| {
            (0..d)
                .map(|j| 1.0 / 10000f64.powf(j as f64 / d as f64))
                .collect()
        })
        .collect();
    let mut table = Vec::with_capacity(positions.len() * c.head_dim / 2);
    for pos in positions {
        for (axis, f) in freqs.iter().enumerate() {
            table.extend(f.iter().map(|&f| {
                let a = pos[axis] * f;
                (a.cos() as f32, a.sin() as f32)
            }));
        }
    }
    table
}

/// Rotates every head of every row of `x` by that row's angles, as a
/// complex multiply on adjacent pairs.
fn apply_rope(x: &mut [f32], c: &TransformerConfig, table: &[(f32, f32)]) {
    let pairs = c.head_dim / 2;
    x.par_chunks_mut(c.dim)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TransformerConfig {
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

    /// The same frequencies as Qwen-Image's table: axis pair `j` of `d`
    /// turns at `10000^(-2j/(2d))`, and each axis starts its own run.
    #[test]
    fn rope_angles_split_the_head_across_three_axes() {
        let c = config();
        let t = rope_angles(&c, &[[3.0, -2.0, 5.0]]);
        assert_eq!(t.len(), 64);
        // First pair of each axis: frequency 1.
        assert!((t[0].0 - 3f32.cos()).abs() < 1e-6);
        assert!((t[8].0 - (-2f32).cos()).abs() < 1e-6);
        assert!((t[8].1 - (-2f32).sin()).abs() < 1e-6);
        assert!((t[36].0 - 5f32.cos()).abs() < 1e-6);
        // The frame axis' last pair: 10000^(-7/8).
        let f = 10000f64.powf(-7.0 / 8.0);
        assert!((t[7].1 - (3.0 * f).sin() as f32).abs() < 1e-6);
    }

    /// Every plan covers the picture's tokens exactly, in order; the taper
    /// puts an eighth at each end in whole 64-token tiles.
    #[test]
    fn overlap_plans_cover_the_picture() {
        assert_eq!(tapered_chunks(4096), vec![512, 1536, 1536, 512]);
        assert_eq!(even_chunks(4096, 1024), vec![1024; 4]);
        assert_eq!(even_chunks(4000, 1024), vec![1024, 1024, 1024, 928]);
        for n in [4096usize, 4000, 6144, 9216, 3999] {
            let t = tapered_chunks(n);
            assert_eq!(t.iter().sum::<usize>(), n, "{n}: {t:?}");
            assert_eq!(t[0], t[3]);
            assert!(t[0].is_multiple_of(64) && t[0] <= n / 8, "{n}: {t:?}");
            assert!(t[1] >= t[0] && t[2] >= t[0], "{n}: {t:?}");
        }
    }

    /// The tuner probes with one head, then fits the card's heads into the
    /// first chunk's window, and follows the card's measured rate.
    #[test]
    fn the_head_tuner_fills_the_first_chunk_window() {
        let ms = Duration::from_millis;
        let pass = |first: u64, on_card: u64| Stages {
            first_chunk: ms(first),
            lane_heads: ms(on_card),
            ..Stages::default()
        };
        let mut tuner = HeadTuner::new();
        tuner.observe(&pass(3_200, 0), 32, 16);
        assert_eq!(tuner.heads, 1, "a probe head first");
        // 31 host heads took 3.1 s of first chunk: 0.09 s a head with the
        // fill; the card took 0.2 s for its one. h ≤ 0.09·32/0.29 = 9.9.
        tuner.observe(&pass(3_100, 200), 32, 16);
        assert_eq!(tuner.heads, 9);
        // A slower card fits fewer.
        tuner.observe(&pass(2_300, 9 * 600), 32, 16);
        assert!(tuner.heads < 9, "{}", tuner.heads);
        // Bounded by `most`, and a card far slower than the window takes none
        // — and stays at none on its remembered rate, no probe again.
        tuner.heads = 4;
        tuner.observe(&pass(100_000, 4), 32, 16);
        assert_eq!(tuner.heads, 16);
        tuner.heads = 4;
        tuner.observe(&pass(1, 4 * 10_000), 32, 16);
        assert_eq!(tuner.heads, 0);
        tuner.observe(&pass(3_200, 0), 32, 16);
        assert_eq!(tuner.heads, 0);
    }

    #[test]
    fn modulation_is_scale_gate_twice() {
        let m: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let [s1, g1, s2, g2] = split4(&m, 2);
        assert_eq!(
            (s1, g1, s2, g2),
            (
                &[0.0, 1.0][..],
                &[2.0, 3.0][..],
                &[4.0, 5.0][..],
                &[6.0, 7.0][..]
            )
        );
        let mut x = vec![2.0, 2.0];
        scale_inplace(&mut x, s2, 2);
        assert_eq!(x, vec![10.0, 12.0]);
        let mut y = vec![1.0, 1.0];
        gated_add(&mut y, &[1.0, 2.0], &tanh_of(&[0.0, 100.0]), 2);
        assert_eq!(y, vec![1.0, 3.0]);
    }
}

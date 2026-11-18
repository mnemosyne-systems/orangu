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

//! The MiniMax-H3-Image diffusion transformer (`minimax_h3_image`), read
//! from the GGUFs of `unsloth/MiniMax-H3-Image-GGUF` (`general.architecture
//! = hyvid`, the name the MiniMax-H3 video files carry too; the picture
//! model is told apart by its tensors).
//!
//! It predicts one **velocity** for a latent picture at noise level
//! `sigma`, conditioned on a prompt's text-encoder states. Its graph:
//!
//! - **One stream throughout.** The prompt's tokens and the picture's are
//!   joined `[text | picture]` and every block runs them together: RMSNorm,
//!   modulation, one `qkv` projection to 56 heads of 128 (wider than the
//!   5376-wide stream), per-head RMS norms, rotary positions, full
//!   attention, a gated output projection; then the same around a SwiGLU
//!   feed-forward.
//! - **Modulation per modality.** Each block's `adaln_proj` writes three
//!   sets of `shift, scale, gate` pairs — picture, text and (unused here)
//!   audio — and a token is modulated by its own modality's set.
//! - **Curve-form time embedding.** The released checkpoint's time
//!   embedding is tabulated: `adaln_t_table` holds eight coordinates at
//!   each of 1025 points of `t = 1 - sigma`, linearly interpolated, and
//!   every modulation linear reads those eight values directly (no SiLU).
//! - **A text refiner.** The encoder's states go through a projection and
//!   two unmodulated blocks of the same shape before joining. That part
//!   depends on the prompt alone, so it runs once per prompt
//!   ([`H3Transformer::embed_text`]).
//! - **Three-axis rotary positions** over the first 96 of each head's 128
//!   values, rotate-half, sixteen frequencies an axis (`rope.inv_freq`): the
//!   `i`th text token at `(i, 0, 0)`, a picture token at `(n_text, y, x)`
//!   with `y` and `x` its row and column spread over the picture's aspect
//!   ratio on a 32-unit side.
//! - **Per-step output heads.** The model is distilled to six steps and its
//!   output layer holds six heads side by side: the first is the base and
//!   each later one an offset for its step of the release schedule
//!   ([`PDD_SIGMAS`]), weighted by how much of that step the current one
//!   covers ([`pdd_head_weights`]). The heads predict `x0 - noise`; the
//!   velocity is its negation.

use crate::engine::backend::{Backend, CpuBackend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use super::transformer::{
    StageClock, Stages, TransformerConfig, gated_add, head_rms_norm, modulate_inplace,
    step_attention,
};

/// Every RMSNorm's epsilon: the blocks', the per-head ones and the final.
const EPS: f32 = 1e-5;
/// The modalities each block's modulation is written for: picture, text,
/// audio.
const MODALITIES: usize = 3;
const PICTURE: usize = 0;
const TEXT: usize = 1;
/// Rotary axes: `(t, y, x)`.
const ROPE_AXES: usize = 3;
/// The side, in rotary units, a square picture's positions span.
const ROPE_SPAN: f32 = 32.0;

/// The release schedule the per-step heads were distilled on —
/// `pdd_config.json`'s `sigmas`.
pub const PDD_SIGMAS: [f32; 7] = [
    1.0,
    0.967_523_7,
    0.887_548_5,
    0.75,
    0.532_773_9,
    0.232_008_87,
    0.0,
];

/// A linear with an optional bias.
struct Linear {
    w: QuantMatrix,
    bias: Option<Vec<f32>>,
}

/// One block's weights. The text refiner's blocks have the same shape
/// without the modulation.
struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// `[q | k | v]`, each `n_head * head_dim` wide.
    qkv: QuantMatrix,
    out: QuantMatrix,
    /// `[gate | up]`, each `ffn_dim` wide.
    fc1: QuantMatrix,
    fc2: QuantMatrix,
    /// `[MODALITIES * 6 * dim]` from the curve coordinates; `None` in the
    /// text refiner.
    adaln: Option<Linear>,
}

/// The timestep-dependent vectors of one pass, keyed by its sigma.
struct Modulation {
    sigma: f32,
    /// Per block, `[MODALITIES * 6 * dim]`.
    blocks: Vec<Vec<f32>>,
    /// `[2 * dim]`: the output head's `shift, scale`.
    head: Vec<f32>,
}

pub struct H3Transformer {
    backend: Arc<dyn Backend>,
    /// Shared with the other picture transformers; `n_head * head_dim` is
    /// the attention width, wider than `dim`; `rope_axes` is unused.
    pub config: TransformerConfig,
    ffn_dim: usize,
    video_in: Linear,
    condition_proj: Linear,
    refiner: Vec<Block>,
    refiner_norm: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    final_adaln: Linear,
    /// `[in_channels * heads]` per token: the per-step heads side by side.
    video_out: Linear,
    heads: usize,
    /// `[grid, curve_dim]`, row-major.
    t_table: Vec<f32>,
    curve_dim: usize,
    /// Sixteen frequencies an axis.
    inv_freq: Vec<f32>,
    stages: Mutex<Stages>,
    modulation: Mutex<Option<Modulation>>,
}

/// What one forward pass is given.
pub struct ForwardInput<'a> {
    /// `[n_img, in_channels]` patchified latent tokens, row-major over the
    /// `(rows, cols)` grid.
    pub img: &'a [f32],
    pub grid: (usize, usize),
    /// `[n_txt, dim]`: the prompt through [`H3Transformer::embed_text`].
    pub txt: &'a [f32],
    pub sigma: f32,
    /// The level this step goes to, which picks the output heads.
    pub sigma_next: f32,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl H3Transformer {
    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(&format!("{name}.weight"))
                .with_context(|| format!("minimax_h3_image: loading {name}"))
        };
        let vector = |name: &str| -> Result<Vec<f32>> {
            Ok(loaded
                .tensor(name)
                .with_context(|| format!("minimax_h3_image: loading {name}"))?
                .0)
        };
        let linear = |name: &str| -> Result<Linear> {
            let bias_name = format!("{name}.bias");
            Ok(Linear {
                w: matrix(name)?,
                bias: if loaded.has_tensor(&bias_name) {
                    Some(vector(&bias_name)?)
                } else {
                    None
                },
            })
        };
        let block = |p: &str, modulated: bool| -> Result<Block> {
            Ok(Block {
                norm1: vector(&format!("{p}.norm1.weight"))?,
                norm2: vector(&format!("{p}.norm2.weight"))?,
                q_norm: vector(&format!("{p}.attn.q_norm.weight"))?,
                k_norm: vector(&format!("{p}.attn.k_norm.weight"))?,
                qkv: matrix(&format!("{p}.attn.qkv_proj"))?,
                out: matrix(&format!("{p}.attn.out_proj"))?,
                fc1: matrix(&format!("{p}.mlp.fc1"))?,
                fc2: matrix(&format!("{p}.mlp.fc2"))?,
                adaln: if modulated {
                    Some(linear(&format!("{p}.adaln_proj.linear"))?)
                } else {
                    None
                },
            })
        };
        let count = |prefix: &str| {
            (0..)
                .take_while(|i| loaded.has_tensor(&format!("{prefix}.{i}.norm1.weight")))
                .count()
        };
        let n_layer = count("blocks");
        let n_refiner = count("token_refiner.blocks");
        ensure!(n_layer > 0, "minimax_h3_image: no blocks.N tensors");
        ensure!(
            !loaded.has_tensor("audio_patch_proj.weight"),
            "minimax_h3_image: this is a MiniMax-H3 video-and-audio transformer, not the \
             picture model"
        );
        ensure!(
            loaded.has_tensor("adaln_t_table"),
            "minimax_h3_image: no adaln_t_table — only the curve-form (pruned) checkpoints \
             are read"
        );
        let video_in = linear("video_patch_proj")?;
        let condition_proj = linear("condition_proj")?;
        let (dim, in_channels, txt_dim) = (
            video_in.w.out_dim,
            video_in.w.in_dim,
            condition_proj.w.in_dim,
        );
        let head_dim = vector("blocks.0.attn.q_norm.weight")?.len();
        let blocks = (0..n_layer)
            .map(|i| block(&format!("blocks.{i}"), true))
            .collect::<Result<Vec<_>>>()?;
        let refiner = (0..n_refiner)
            .map(|i| block(&format!("token_refiner.blocks.{i}"), false))
            .collect::<Result<Vec<_>>>()?;
        let inner = blocks[0].qkv.out_dim / 3;
        ensure!(
            head_dim > 0 && inner.is_multiple_of(head_dim),
            "minimax_h3_image: attention width {inner} is not whole heads of {head_dim}"
        );
        let ffn_dim = blocks[0].fc2.in_dim;
        let (t_table, t_shape) = loaded.tensor("adaln_t_table")?;
        let curve_dim = t_shape.first().copied().unwrap_or(0) as usize;
        let inv_freq = vector("rope.inv_freq")?;
        let video_out = linear("final_layer.video_out")?;
        let heads = video_out.w.out_dim / in_channels.max(1);
        let model = Self {
            backend,
            config: TransformerConfig {
                dim,
                n_head: inner / head_dim,
                head_dim,
                n_layer,
                txt_dim,
                in_channels,
                rope_axes: [0; 3],
                eps: EPS,
            },
            ffn_dim,
            video_in,
            condition_proj,
            refiner,
            refiner_norm: vector("token_refiner.final_norm.weight")?,
            blocks,
            final_norm: vector("final_layer.norm.weight")?,
            final_adaln: linear("final_layer.adaln_proj.linear")?,
            video_out,
            heads,
            t_table,
            curve_dim,
            inv_freq,
            stages: Mutex::new(Stages::default()),
            modulation: Mutex::new(None),
        };
        model.check_shapes()?;
        Ok(model)
    }

    fn check_shapes(&self) -> Result<()> {
        let c = &self.config;
        let (d, f) = (c.dim, self.ffn_dim);
        let inner = c.n_head * c.head_dim;
        ensure!(
            self.curve_dim > 0
                && self.t_table.len().is_multiple_of(self.curve_dim)
                && self.t_table.len() / self.curve_dim >= 2,
            "minimax_h3_image: adaln_t_table is not a table of curve coordinates"
        );
        ensure!(
            self.inv_freq.len() * 2 * ROPE_AXES <= c.head_dim,
            "minimax_h3_image: {} rotary frequencies an axis do not fit a {}-wide head",
            self.inv_freq.len(),
            c.head_dim
        );
        ensure!(
            self.condition_proj.w.out_dim == d
                && self.refiner_norm.len() == d
                && self.final_norm.len() == d
                && self.final_adaln.w.in_dim == self.curve_dim
                && self.final_adaln.w.out_dim == 2 * d
                && self.video_out.w.in_dim == d
                && self.heads > 0
                && self.video_out.w.out_dim == self.heads * c.in_channels,
            "minimax_h3_image: the embedding, modulation and output projections do not match \
             the stream width {d}"
        );
        ensure!(
            self.heads == 1 || self.heads == PDD_SIGMAS.len() - 1,
            "minimax_h3_image: {} output heads, where the six-step model has {}",
            self.heads,
            PDD_SIGMAS.len() - 1
        );
        for (i, b) in self.blocks.iter().chain(&self.refiner).enumerate() {
            ensure!(
                b.qkv.in_dim == d
                    && b.qkv.out_dim == 3 * inner
                    && b.out.in_dim == inner
                    && b.out.out_dim == d
                    && b.fc1.in_dim == d
                    && b.fc1.out_dim == 2 * f
                    && b.fc2.out_dim == d
                    && b.norm1.len() == d
                    && b.q_norm.len() == c.head_dim
                    && b.adaln.as_ref().is_none_or(|a| {
                        a.w.in_dim == self.curve_dim && a.w.out_dim == MODALITIES * 6 * d
                    }),
                "minimax_h3_image: block {i} does not match the stream width {d}"
            );
        }
        Ok(())
    }

    /// The first block's feed-forward input projection — the widest linear
    /// of a step, which `calibrate` times.
    pub const CALIBRATION_TENSOR: &'static str = "blocks.0.mlp.fc1.weight";

    pub fn take_stages(&self) -> Stages {
        std::mem::take(
            &mut *self
                .stages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    fn matmul(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * w.in_dim);
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }

    fn linear(&self, x: &[f32], n: usize, l: &Linear) -> Vec<f32> {
        let mut out = self.matmul(x, n, &l.w);
        if let Some(bias) = &l.bias {
            tensor::add_bias_per_row(&mut out, bias, n);
        }
        out
    }

    /// A one-row linear on the CPU: the modulation projections are a few
    /// hundred thousand multiply-adds each, not worth a device's upload.
    fn host_linear(x: &[f32], l: &Linear) -> Vec<f32> {
        let mut out = CpuBackend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: 1,
                w: &l.w,
            }])
            .pop()
            .expect("one op in, one result out");
        if let Some(bias) = &l.bias {
            tensor::add_inplace(&mut out, bias);
        }
        out
    }

    /// The prompt as the blocks read it: the encoder's `[n, txt_dim]`
    /// states projected to the stream width and refined by the text blocks
    /// — `[n, dim]`. Independent of the picture and the noise level, so
    /// computed once per prompt.
    pub fn embed_text(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let c = &self.config;
        ensure!(
            !hidden.is_empty() && hidden.len().is_multiple_of(c.txt_dim),
            "minimax_h3_image: text states are not rows of {}",
            c.txt_dim
        );
        let n = hidden.len() / c.txt_dim;
        let mut x = self.linear(hidden, n, &self.condition_proj);
        let mut normed = Vec::new();
        for block in &self.refiner {
            tensor::rmsnorm_into(&mut normed, &x, &block.norm1, n, c.dim, EPS);
            let out = self.attention(block, &normed, n, None);
            tensor::add_inplace(&mut x, &out);
            tensor::rmsnorm_into(&mut normed, &x, &block.norm2, n, c.dim, EPS);
            let out = self.feed_forward(block, &normed, n);
            tensor::add_inplace(&mut x, &out);
        }
        tensor::rmsnorm_inplace(&mut x, &self.refiner_norm, n, c.dim, EPS);
        Ok(x)
    }

    /// Attention over `[n, dim]` normed rows, with the rotary table when
    /// there is one — the output projection's input, `[n, n_head *
    /// head_dim]`.
    fn attend(
        &self,
        block: &Block,
        normed: &[f32],
        n: usize,
        rope: Option<&[(f32, f32)]>,
    ) -> Vec<f32> {
        let c = &self.config;
        let inner = c.n_head * c.head_dim;
        let qkv = self.matmul(normed, n, &block.qkv);
        let (mut q, mut k, v) = split_thirds(&qkv, n, inner);
        drop(qkv);
        head_rms_norm(&mut q, &block.q_norm, c.head_dim, EPS);
        head_rms_norm(&mut k, &block.k_norm, c.head_dim, EPS);
        if let Some(rope) = rope {
            let pairs = self.inv_freq.len() * ROPE_AXES;
            apply_partial_rope(&mut q, inner, c.head_dim, pairs, rope);
            apply_partial_rope(&mut k, inner, c.head_dim, pairs, rope);
        }
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        step_attention(&q, n, &k, &v, n, c.n_head, c.head_dim, scale, None)
    }

    fn attention(
        &self,
        block: &Block,
        normed: &[f32],
        n: usize,
        rope: Option<&[(f32, f32)]>,
    ) -> Vec<f32> {
        let attn = self.attend(block, normed, n, rope);
        self.matmul(&attn, n, &block.out)
    }

    /// `fc2(silu(gate) * up)` over `[n, dim]` normed rows.
    fn feed_forward(&self, block: &Block, normed: &[f32], n: usize) -> Vec<f32> {
        let gate_up = self.matmul(normed, n, &block.fc1);
        let h = swiglu(&gate_up, self.ffn_dim);
        drop(gate_up);
        self.matmul(&h, n, &block.fc2)
    }

    /// The curve coordinates at `t = 1 - sigma`: the table's two nearest
    /// rows, linearly interpolated, the ends clamped.
    fn curve(&self, sigma: f32) -> Vec<f32> {
        let d = self.curve_dim;
        let grid = self.t_table.len() / d;
        let t = 1.0 - sigma.clamp(1e-6, 1.0);
        let position = t.clamp(0.0, 1.0) * (grid - 1) as f32;
        let i0 = (position.floor() as usize).min(grid - 2);
        let frac = position - i0 as f32;
        let (lo, hi) = (
            &self.t_table[i0 * d..(i0 + 1) * d],
            &self.t_table[(i0 + 1) * d..(i0 + 2) * d],
        );
        lo.iter().zip(hi).map(|(a, b)| a + (b - a) * frac).collect()
    }

    fn modulation(&self, sigma: f32) -> Modulation {
        let cached = self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .filter(|m| m.sigma == sigma);
        if let Some(m) = cached {
            return m;
        }
        let curve = self.curve(sigma);
        Modulation {
            sigma,
            blocks: self
                .blocks
                .par_iter()
                .map(|b| Self::host_linear(&curve, b.adaln.as_ref().expect("a modulated block")))
                .collect(),
            head: Self::host_linear(&curve, &self.final_adaln),
        }
    }

    /// One velocity prediction: `[n_img, in_channels]`, the layout the
    /// latent tokens came in.
    pub fn forward(&self, input: &ForwardInput<'_>) -> Result<Vec<f32>> {
        let c = &self.config;
        let d = c.dim;
        let (rows, cols) = input.grid;
        let n_img = rows * cols;
        ensure!(
            input.img.len() == n_img * c.in_channels,
            "minimax_h3_image: {} latent values for a {rows}x{cols} grid of {}-wide tokens",
            input.img.len(),
            c.in_channels
        );
        ensure!(
            !input.txt.is_empty() && input.txt.len().is_multiple_of(d),
            "minimax_h3_image: embedded text is not rows of {d}"
        );
        let n_txt = input.txt.len() / d;
        let n = n_txt + n_img;
        let mut clock = StageClock::start();

        // `[text | picture]`.
        let mut x = Vec::with_capacity(n * d);
        x.extend_from_slice(input.txt);
        x.extend_from_slice(&self.linear(input.img, n_img, &self.video_in));
        clock.lap(|s| &mut s.other);
        let modulation = self.modulation(input.sigma);
        clock.lap(|s| &mut s.modulation);
        let rope = rope_table(&self.inv_freq, n_txt, (rows, cols));
        let mut normed = Vec::new();
        let split = n_txt * d;

        for (block, m) in self.blocks.iter().zip(&modulation.blocks) {
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            let set =
                |modality: usize, k: usize| &m[(modality * 6 + k) * d..(modality * 6 + k + 1) * d];
            let modulate = |rows: &mut [f32], a: usize, b: usize| {
                let (txt, img) = rows.split_at_mut(split);
                modulate_inplace(txt, set(TEXT, a), set(TEXT, b), d);
                modulate_inplace(img, set(PICTURE, a), set(PICTURE, b), d);
            };
            let gated = |x: &mut [f32], y: &[f32], g: usize| {
                let (xt, xi) = x.split_at_mut(split);
                let (yt, yi) = y.split_at(split);
                gated_add(xt, yt, set(TEXT, g), d);
                gated_add(xi, yi, set(PICTURE, g), d);
            };
            tensor::rmsnorm_into(&mut normed, &x, &block.norm1, n, d, EPS);
            modulate(&mut normed, 0, 1);
            clock.lap(|s| &mut s.other);
            let attn = self.attend(block, &normed, n, Some(&rope));
            clock.lap(|s| &mut s.attention);
            let out = self.matmul(&attn, n, &block.out);
            drop(attn);
            clock.lap(|s| &mut s.out);
            gated(&mut x, &out, 2);

            tensor::rmsnorm_into(&mut normed, &x, &block.norm2, n, d, EPS);
            modulate(&mut normed, 3, 4);
            clock.lap(|s| &mut s.other);
            let out = self.feed_forward(block, &normed, n);
            clock.lap(|s| &mut s.mlp);
            gated(&mut x, &out, 5);
        }

        // The head over the picture's rows: `shift, scale = head`.
        let (shift, scale) = modulation.head.split_at(d);
        tensor::rmsnorm_into(&mut normed, &x[split..], &self.final_norm, n_img, d, EPS);
        modulate_inplace(&mut normed, shift, scale, d);
        let heads = self.linear(&normed, n_img, &self.video_out);
        let weights = if self.heads == 1 {
            vec![1.0]
        } else {
            pdd_head_weights(input.sigma, input.sigma_next).to_vec()
        };
        let ch = c.in_channels;
        let mut out = vec![0.0f32; n_img * ch];
        out.par_chunks_mut(ch)
            .zip(heads.par_chunks(ch * self.heads))
            .for_each(|(o, row)| {
                for (head, &w) in row.chunks(ch).zip(&weights) {
                    if w != 0.0 {
                        for (o, v) in o.iter_mut().zip(head) {
                            *o -= w * v;
                        }
                    }
                }
            });
        clock.lap(|s| &mut s.other);
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

/// The weight of each per-step output head for a step from `sigma` to
/// `sigma_next`: the base head always, and each release step the move
/// spans in proportion to its share of the move. On the release schedule
/// that is the base and the step's own head.
pub fn pdd_head_weights(sigma: f32, sigma_next: f32) -> [f32; 6] {
    let nearest = |value: f32| {
        let mut index = 0;
        for (i, s) in PDD_SIGMAS.iter().enumerate().skip(1) {
            if (s - value).abs() < (PDD_SIGMAS[index] - value).abs() {
                index = i;
            }
        }
        index
    };
    let start = nearest(sigma).min(PDD_SIGMAS.len() - 2);
    let stop = nearest(sigma_next).max(start + 1);
    let mut weights = [0.0f32; 6];
    weights[0] = 1.0;
    for (i, w) in weights.iter_mut().enumerate().take(stop).skip(start.max(1)) {
        *w = (PDD_SIGMAS[i] - PDD_SIGMAS[i + 1]) / (PDD_SIGMAS[start] - PDD_SIGMAS[stop]);
    }
    weights
}

/// `q`, `k` and `v` out of rows of `3 * width` values.
fn split_thirds(rows: &[f32], n: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(rows.len(), n * 3 * width);
    let mut q = vec![0.0f32; n * width];
    let mut k = vec![0.0f32; n * width];
    let mut v = vec![0.0f32; n * width];
    q.par_chunks_mut(width)
        .zip(k.par_chunks_mut(width))
        .zip(v.par_chunks_mut(width))
        .zip(rows.par_chunks(3 * width))
        .for_each(|(((q, k), v), row)| {
            q.copy_from_slice(&row[..width]);
            k.copy_from_slice(&row[width..2 * width]);
            v.copy_from_slice(&row[2 * width..]);
        });
    (q, k, v)
}

/// `silu(gate) * up` over rows of `[gate | up]`, each `hidden` wide.
pub(super) fn swiglu(rows: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows.len() / 2];
    out.par_chunks_mut(hidden)
        .zip(rows.par_chunks(2 * hidden))
        .for_each(|(o, row)| {
            let (gate, up) = row.split_at(hidden);
            for ((o, &g), &u) in o.iter_mut().zip(gate).zip(up) {
                *o = tensor::silu(g) * u;
            }
        });
    out
}

/// One axis of picture positions: `count = latent / 2` tokens spread over
/// `latent / sqrt(area)` of the [`ROPE_SPAN`], centred — the reference's
/// `spatial_axis`, in latent cells (two to a token).
fn spatial_axis(latent: usize, sqrt_area: f32) -> Vec<f32> {
    let count = latent / 2;
    let ratio = latent as f32 / sqrt_area;
    (0..count)
        .map(|i| (i as f32 * (ratio / count as f32) + (1.0 - ratio) * 0.5) * ROPE_SPAN)
        .collect()
}

/// `(cos, sin)` for every rotated pair of the head, for every row of the
/// joint sequence `[text | picture]`: pair `a * F + f` turns at
/// `position[a] · inv_freq[f]`.
fn rope_table(inv_freq: &[f32], n_txt: usize, (rows, cols): (usize, usize)) -> Vec<(f32, f32)> {
    let pairs = inv_freq.len() * ROPE_AXES;
    let mut table = Vec::with_capacity((n_txt + rows * cols) * pairs);
    let mut push = |pos: [f32; ROPE_AXES]| {
        for p in pos {
            for f in inv_freq {
                let a = p * f;
                table.push((a.cos(), a.sin()));
            }
        }
    };
    for i in 0..n_txt {
        push([i as f32, 0.0, 0.0]);
    }
    let sqrt_area = ((2 * rows * 2 * cols) as f32).sqrt();
    let ys = spatial_axis(2 * rows, sqrt_area);
    let xs = spatial_axis(2 * cols, sqrt_area);
    for &y in &ys {
        for &x in &xs {
            push([n_txt as f32, y, x]);
        }
    }
    table
}

/// Rotates the first `2 * pairs` values of every head of every row of `x`
/// (`[n, width]`) by that row's angles, rotate-half style within them: pair
/// `j` is `(x[j], x[j + pairs])`. The rest of the head is left as it is.
pub(super) fn apply_partial_rope(
    x: &mut [f32],
    width: usize,
    head_dim: usize,
    pairs: usize,
    table: &[(f32, f32)],
) {
    x.par_chunks_mut(width)
        .zip(table.par_chunks(pairs))
        .for_each(|(row, angles)| {
            for head in row.chunks_mut(head_dim) {
                let (lo, hi) = head[..2 * pairs].split_at_mut(pairs);
                for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi.iter_mut()).zip(angles) {
                    let (x0, x1) = (*a, *b);
                    *a = x0 * cos - x1 * sin;
                    *b = x0 * sin + x1 * cos;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On the release schedule a step takes the base head and its own.
    #[test]
    fn release_steps_take_the_base_and_their_own_head() {
        for step in 0..6 {
            let w = pdd_head_weights(PDD_SIGMAS[step], PDD_SIGMAS[step + 1]);
            let mut want = [0.0f32; 6];
            want[0] = 1.0;
            if step > 0 {
                want[step] = 1.0;
            }
            assert_eq!(w, want, "step {step}");
        }
        // Two release steps at once share their heads by their spans.
        let w = pdd_head_weights(PDD_SIGMAS[2], PDD_SIGMAS[4]);
        let span = PDD_SIGMAS[2] - PDD_SIGMAS[4];
        assert!((w[2] - (PDD_SIGMAS[2] - PDD_SIGMAS[3]) / span).abs() < 1e-6);
        assert!((w[3] - (PDD_SIGMAS[3] - PDD_SIGMAS[4]) / span).abs() < 1e-6);
        assert!((w[2] + w[3] - 1.0).abs() < 1e-6);
    }

    /// A square picture spans the whole side from zero; a wide one keeps
    /// its rows centred on a shorter span.
    #[test]
    fn picture_positions_follow_the_aspect_ratio() {
        let square = spatial_axis(64, 64.0);
        assert_eq!(square.len(), 32);
        assert_eq!(square[0], 0.0);
        assert_eq!(square[31], 31.0);
        // 64 latent rows by 256 columns: sqrt(area) = 128.
        let rows = spatial_axis(64, 128.0);
        assert!((rows[0] - 8.0).abs() < 1e-5);
        assert!((rows[1] - 8.5).abs() < 1e-5);
    }

    /// Text turns on the first axis only; a picture token sits at the text's
    /// length on the first axis and its place on the other two.
    #[test]
    fn rope_positions_index_the_text_and_place_the_picture() {
        let inv: Vec<f32> = (0..16).map(|f| 1.0 / (1.0 + f as f32)).collect();
        let table = rope_table(&inv, 3, (2, 2));
        let pairs = 48;
        assert_eq!(table.len(), (3 + 4) * pairs);
        let at = |row: usize, pair: usize| table[row * pairs + pair];
        // Text token 2: t = 2, y = x = 0.
        assert!((at(2, 1).1 - (2.0 * inv[1]).sin()).abs() < 1e-6);
        assert_eq!(at(2, 16 + 1), (1.0, 0.0));
        // Picture token (1, 0): t = 3, y = 16, x = 0 on a 4x4 latent.
        let p = 3 + 2;
        assert!((at(p, 0).1 - 3.0f32.sin()).abs() < 1e-6);
        assert!((at(p, 16).1 - 16.0f32.sin()).abs() < 1e-5);
        assert_eq!(at(p, 32), (1.0, 0.0));
    }

    /// Only the first `2 * pairs` values of a head turn.
    #[test]
    fn partial_rope_leaves_the_tail() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0, 5.0, 6.0];
        let a = 0.5f32;
        apply_partial_rope(&mut x, 6, 6, 2, &[(a.cos(), a.sin()), (a.cos(), a.sin())]);
        assert!((x[0] - a.cos()).abs() < 1e-6 && (x[2] - a.sin()).abs() < 1e-6);
        assert!((x[1] + a.sin()).abs() < 1e-6 && (x[3] - a.cos()).abs() < 1e-6);
        assert_eq!(&x[4..], &[5.0, 6.0]);
    }

    #[test]
    fn swiglu_gates_the_second_half() {
        let out = swiglu(&[1.0, -2.0, 3.0, 4.0], 2);
        assert!((out[0] - tensor::silu(1.0) * 3.0).abs() < 1e-6);
        assert!((out[1] - tensor::silu(-2.0) * 4.0).abs() < 1e-6);
    }
}

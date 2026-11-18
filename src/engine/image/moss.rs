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

//! The MOSS-Image diffusion transformer (`moss_image`) — OpenMOSS's
//! `MOSSImageTransformer`, read from the GGUF `unsloth/MOSS-Image-Base-GGUF`
//! ships in the reference's own tensor naming (`layers.N.proj.q`,
//! `text_embedder.blocks.N.qkv`).
//!
//! Like the other picture transformers it predicts one **velocity** for a
//! latent picture at noise level `sigma`, conditioned on a prompt's
//! text-encoder states. Its graph:
//!
//! - **One stream throughout.** The picture's tokens and the prompt's are
//!   joined `[image | text]` before the first block and every block runs
//!   them together: RMSNorm, modulation, `q`/`k`/`v` with per-head RMS norms
//!   and rotary positions, full attention, a gated output projection; then
//!   the same around the feed-forward.
//! - **One modulation for every block.** The timestep's `shift, scale,
//!   gate` pairs come from a single `shared_block_modulation` linear over
//!   the time embedding (no SiLU before it), read once a pass.
//! - **A text refiner.** The encoder's states go through a projection and
//!   two small blocks of their own (biased, unmodulated) before joining.
//!   That part depends on the prompt alone, so it runs once per prompt
//!   ([`MossTransformer::embed_text`]) rather than once per step.
//! - **SITU feed-forward**: `4 tanh(g / 4) σ(g) · 25 tanh(h / 25)` over the
//!   two halves of the input projection — a soft-capped gated SiLU.
//! - **Four-axis rotary positions**, rotate-half over the 128-wide head,
//!   sixteen frequencies an axis, `base = 10000`: a picture token at row
//!   `r`, column `c` sits at `(0, n_text, r - (rows - 1) / 2, c - (cols -
//!   1) / 2)` — centred, so fractional — the `i`th text token at
//!   `(0, i, 0, 0)`.
//!
//! Guidance is classifier-free, with the negative branch run with one
//! block skipped ([`ForwardInput::skip`]); the pipeline decides which.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use super::transformer::{
    StageClock, Stages, TIMESTEP_FREQUENCIES, TransformerConfig, gated_add, head_rms_norm,
    modulate_inplace, silu_inplace, step_attention, timestep_embedding,
};

/// The rotary base, `rope_base` in `transformer/config.json`.
const ROPE_BASE: f64 = 10000.0;
/// Rotary axes over the head: `(r, f, h, w)`.
const ROPE_AXES: usize = 4;
/// Every RMSNorm's epsilon.
const EPS: f32 = 1e-6;

/// A linear with an optional bias.
struct Linear {
    w: QuantMatrix,
    bias: Option<Vec<f32>>,
}

struct TextBlock {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: Linear,
    proj: Linear,
    ffn_gate: Linear,
    ffn_up: Linear,
    ffn_down: Linear,
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: QuantMatrix,
    k: QuantMatrix,
    v: QuantMatrix,
    o: QuantMatrix,
    /// `[hidden | gate]` in one projection.
    ffn_up_gate: QuantMatrix,
    ffn_down: QuantMatrix,
}

/// The timestep-dependent vectors of one pass, keyed by its sigma: under
/// guidance the negative prompt's pass follows the positive one at the
/// same sigma and reuses them.
struct Modulation {
    sigma: f32,
    /// `[6 * dim]`: attention `shift, scale, gate`, then the feed-forward's.
    block: Vec<f32>,
    /// `[2 * dim]`: the output head's `shift, scale`.
    head: Vec<f32>,
}

pub struct MossTransformer {
    backend: Arc<dyn Backend>,
    /// Shared with the other picture transformers; `txt_dim` is the text
    /// encoder's width, `rope_axes` is unused.
    pub config: TransformerConfig,
    /// The feed-forward's hidden width.
    ffn_dim: usize,
    video_embedder: Linear,
    text_proj: Linear,
    text_blocks: Vec<TextBlock>,
    time_in: Linear,
    time_out: Linear,
    block_modulation: Linear,
    head_modulation: Linear,
    final_norm: Vec<f32>,
    final_linear: QuantMatrix,
    layers: Vec<Block>,
    stages: Mutex<Stages>,
    modulation: Mutex<Option<Modulation>>,
}

/// What one forward pass is given.
pub struct ForwardInput<'a> {
    /// `[n_img, in_channels]` patchified latent tokens, row-major over the
    /// `(rows, cols)` grid.
    pub img: &'a [f32],
    pub grid: (usize, usize),
    /// `[n_txt, dim]`: the prompt through [`MossTransformer::embed_text`].
    pub txt: &'a [f32],
    pub sigma: f32,
    /// Blocks (zero-based) this pass leaves out — the negative branch's.
    pub skip: &'a [usize],
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl MossTransformer {
    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(&format!("{name}.weight"))
                .with_context(|| format!("moss_image: loading {name}"))
        };
        let vector = |name: &str| -> Result<Vec<f32>> {
            Ok(loaded
                .tensor(name)
                .with_context(|| format!("moss_image: loading {name}"))?
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
        let video_embedder = linear("video_embedder")?;
        let text_proj = linear("text_embedder.proj")?;
        let (dim, in_channels, txt_dim) = (
            video_embedder.w.out_dim,
            video_embedder.w.in_dim,
            text_proj.w.in_dim,
        );
        let head_dim = vector("layers.0.q_norm.weight")?.len();
        ensure!(
            head_dim > 0 && dim.is_multiple_of(head_dim) && head_dim.is_multiple_of(2 * ROPE_AXES),
            "moss_image: stream width {dim} and head width {head_dim} do not fit the rotary axes"
        );
        let count = |prefix: &str, leaf: &str| {
            (0..)
                .take_while(|i| loaded.has_tensor(&format!("{prefix}.{i}.{leaf}")))
                .count()
        };
        let n_layer = count("layers", "proj.q.weight");
        let n_text = count("text_embedder.blocks", "qkv.weight");
        ensure!(n_layer > 0, "moss_image: no layers.N tensors");
        let text_blocks = (0..n_text)
            .map(|i| {
                let p = format!("text_embedder.blocks.{i}");
                Ok(TextBlock {
                    norm1: vector(&format!("{p}.norm1.weight"))?,
                    norm2: vector(&format!("{p}.norm2.weight"))?,
                    q_norm: vector(&format!("{p}.q_norm.weight"))?,
                    k_norm: vector(&format!("{p}.k_norm.weight"))?,
                    qkv: linear(&format!("{p}.qkv"))?,
                    proj: linear(&format!("{p}.proj"))?,
                    ffn_gate: linear(&format!("{p}.ffn_gate"))?,
                    ffn_up: linear(&format!("{p}.ffn_up"))?,
                    ffn_down: linear(&format!("{p}.ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..n_layer)
            .map(|i| {
                let p = format!("layers.{i}");
                Ok(Block {
                    norm1: vector(&format!("{p}.norm1.weight"))?,
                    norm2: vector(&format!("{p}.norm2.weight"))?,
                    q_norm: vector(&format!("{p}.q_norm.weight"))?,
                    k_norm: vector(&format!("{p}.k_norm.weight"))?,
                    q: matrix(&format!("{p}.proj.q"))?,
                    k: matrix(&format!("{p}.proj.k"))?,
                    v: matrix(&format!("{p}.proj.v"))?,
                    o: matrix(&format!("{p}.proj.o"))?,
                    ffn_up_gate: matrix(&format!("{p}.proj.ffn_up_gate"))?,
                    ffn_down: matrix(&format!("{p}.proj.ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ffn_dim = layers[0].ffn_down.in_dim;
        let model = Self {
            backend,
            config: TransformerConfig {
                dim,
                n_head: dim / head_dim,
                head_dim,
                n_layer,
                txt_dim,
                in_channels,
                rope_axes: [0; 3],
                eps: EPS,
            },
            ffn_dim,
            video_embedder,
            text_proj,
            text_blocks,
            time_in: linear("time_embedding.0")?,
            time_out: linear("time_embedding.2")?,
            block_modulation: linear("shared_block_modulation")?,
            head_modulation: linear("video_head_modulation")?,
            final_norm: vector("final_norm_video.weight")?,
            final_linear: matrix("final_linear_video")?,
            layers,
            stages: Mutex::new(Stages::default()),
            modulation: Mutex::new(None),
        };
        model.check_shapes()?;
        Ok(model)
    }

    fn check_shapes(&self) -> Result<()> {
        let (d, f) = (self.config.dim, self.ffn_dim);
        ensure!(
            self.time_in.w.in_dim == TIMESTEP_FREQUENCIES
                && self.time_out.w.out_dim == d
                && self.block_modulation.w.out_dim == 6 * d
                && self.head_modulation.w.out_dim == 2 * d
                && self.text_proj.w.out_dim == d
                && self.final_norm.len() == d
                && self.final_linear.in_dim == d
                && self.final_linear.out_dim == self.config.in_channels,
            "moss_image: the embedding, modulation and output projections do not match the \
             stream width {d}"
        );
        for (i, b) in self.layers.iter().enumerate() {
            ensure!(
                [&b.q, &b.k, &b.v, &b.o]
                    .iter()
                    .all(|m| m.in_dim == d && m.out_dim == d)
                    && b.ffn_up_gate.out_dim == 2 * f
                    && b.ffn_down.out_dim == d
                    && b.norm1.len() == d
                    && b.q_norm.len() == self.config.head_dim,
                "moss_image: layers.{i} does not match the stream width {d}"
            );
        }
        for (i, b) in self.text_blocks.iter().enumerate() {
            ensure!(
                b.qkv.w.out_dim == 3 * d
                    && b.proj.w.out_dim == d
                    && b.ffn_gate.w.in_dim == d
                    && b.ffn_down.w.out_dim == d,
                "moss_image: text_embedder.blocks.{i} does not match the stream width {d}"
            );
        }
        Ok(())
    }

    /// The first block's feed-forward input projection — the widest linear
    /// of a step, which `calibrate` times.
    pub const CALIBRATION_TENSOR: &'static str = "layers.0.proj.ffn_up_gate.weight";

    pub fn take_stages(&self) -> Stages {
        std::mem::take(
            &mut *self
                .stages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// `x · Wᵀ` for each of `ws`, in one batch: they share the input, so a
    /// device reads it once.
    fn linears(&self, x: &[f32], n: usize, ws: &[&QuantMatrix]) -> Vec<Vec<f32>> {
        let ops: Vec<MatmulOp<'_>> = ws
            .iter()
            .map(|w| {
                debug_assert_eq!(x.len(), n * w.in_dim);
                MatmulOp { x, n_tokens: n, w }
            })
            .collect();
        self.backend.matmul_batch(&ops)
    }

    fn matmul(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        self.linears(x, n, &[w])
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

    /// The prompt as the blocks read it: the encoder's `[n, txt_dim]` final
    /// hidden states projected to the stream width and refined by the text
    /// blocks, with the text tokens' own rotary positions — `[n, dim]`.
    /// Independent of the picture and the noise level, so computed once per
    /// prompt.
    pub fn embed_text(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let c = &self.config;
        let d = c.dim;
        ensure!(
            !hidden.is_empty() && hidden.len().is_multiple_of(c.txt_dim),
            "moss_image: text states are not rows of {}",
            c.txt_dim
        );
        let n = hidden.len() / c.txt_dim;
        let mut x = self.linear(hidden, n, &self.text_proj);
        let rope = rope_table(c.head_dim, None, n);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut normed = Vec::new();
        for block in &self.text_blocks {
            tensor::rmsnorm_into(&mut normed, &x, &block.norm1, n, d, EPS);
            let qkv = self.linear(&normed, n, &block.qkv);
            let (mut q, mut k, v) = split_thirds(&qkv, n, d);
            drop(qkv);
            head_rms_norm(&mut q, &block.q_norm, c.head_dim, EPS);
            head_rms_norm(&mut k, &block.k_norm, c.head_dim, EPS);
            apply_rope(&mut q, d, c.head_dim, &rope);
            apply_rope(&mut k, d, c.head_dim, &rope);
            let attn = step_attention(&q, n, &k, &v, n, c.n_head, c.head_dim, scale, None);
            let out = self.linear(&attn, n, &block.proj);
            tensor::add_inplace(&mut x, &out);
            tensor::rmsnorm_into(&mut normed, &x, &block.norm2, n, d, EPS);
            let up = self.linear(&normed, n, &block.ffn_up);
            let gate = self.linear(&normed, n, &block.ffn_gate);
            let h = situ(&up, &gate);
            let out = self.linear(&h, n, &block.ffn_down);
            tensor::add_inplace(&mut x, &out);
        }
        Ok(x)
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
        let mut h = self.linear(&timestep_embedding(sigma), 1, &self.time_in);
        silu_inplace(&mut h);
        let time = self.linear(&h, 1, &self.time_out);
        Modulation {
            sigma,
            block: self.linear(&time, 1, &self.block_modulation),
            head: self.linear(&time, 1, &self.head_modulation),
        }
    }

    /// One velocity prediction: `[n_img, in_channels]`, the layout the
    /// latent tokens came in.
    pub fn forward(&self, input: &ForwardInput<'_>) -> Result<Vec<f32>> {
        let c = &self.config;
        let (d, f) = (c.dim, self.ffn_dim);
        let (rows, cols) = input.grid;
        let n_img = rows * cols;
        ensure!(
            input.img.len() == n_img * c.in_channels,
            "moss_image: {} latent values for a {rows}x{cols} grid of {}-wide tokens",
            input.img.len(),
            c.in_channels
        );
        ensure!(
            !input.txt.is_empty() && input.txt.len().is_multiple_of(d),
            "moss_image: embedded text is not rows of {d}"
        );
        let n_txt = input.txt.len() / d;
        let n = n_img + n_txt;
        let mut clock = StageClock::start();

        // `[image | text]`.
        let mut x = self.linear(input.img, n_img, &self.video_embedder);
        x.extend_from_slice(input.txt);
        clock.lap(|s| &mut s.other);
        let modulation = self.modulation(input.sigma);
        clock.lap(|s| &mut s.modulation);
        let m: Vec<&[f32]> = modulation.block.chunks(d).collect();
        let (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f) =
            (m[0], m[1], m[2], m[3], m[4], m[5]);

        let rope = rope_table(c.head_dim, Some((rows, cols)), n_txt);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut normed = Vec::new();
        clock.lap(|s| &mut s.other);

        for (index, block) in self.layers.iter().enumerate() {
            if input.skip.contains(&index) {
                continue;
            }
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            tensor::rmsnorm_into(&mut normed, &x, &block.norm1, n, d, EPS);
            modulate_inplace(&mut normed, shift_a, scale_a, d);
            clock.lap(|s| &mut s.other);
            let mut qkv = self.linears(&normed, n, &[&block.q, &block.k, &block.v]);
            clock.lap(|s| &mut s.qkv);
            let v = qkv.pop().expect("three projections");
            let mut k = qkv.pop().expect("three projections");
            let mut q = qkv.pop().expect("three projections");
            head_rms_norm(&mut q, &block.q_norm, c.head_dim, EPS);
            head_rms_norm(&mut k, &block.k_norm, c.head_dim, EPS);
            apply_rope(&mut q, d, c.head_dim, &rope);
            apply_rope(&mut k, d, c.head_dim, &rope);
            let attn = step_attention(&q, n, &k, &v, n, c.n_head, c.head_dim, scale, None);
            drop((q, k, v));
            clock.lap(|s| &mut s.attention);
            let out = self.matmul(&attn, n, &block.o);
            drop(attn);
            clock.lap(|s| &mut s.out);
            gated_add(&mut x, &out, gate_a, d);

            tensor::rmsnorm_into(&mut normed, &x, &block.norm2, n, d, EPS);
            modulate_inplace(&mut normed, shift_f, scale_f, d);
            clock.lap(|s| &mut s.other);
            let up_gate = self.matmul(&normed, n, &block.ffn_up_gate);
            let h = situ_fused(&up_gate, f);
            drop(up_gate);
            let out = self.matmul(&h, n, &block.ffn_down);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut x, &out, gate_f, d);
        }

        // The head over the picture's rows: `shift, scale = head`.
        let (shift, scale_v) = modulation.head.split_at(d);
        tensor::rmsnorm_into(
            &mut normed,
            &x[..n_img * d],
            &self.final_norm,
            n_img,
            d,
            EPS,
        );
        modulate_inplace(&mut normed, shift, scale_v, d);
        let out = self.matmul(&normed, n_img, &self.final_linear);
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

/// `q`, `k` and `v` out of rows of `3 * dim` values.
fn split_thirds(rows: &[f32], n: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(rows.len(), n * 3 * dim);
    let mut q = vec![0.0f32; n * dim];
    let mut k = vec![0.0f32; n * dim];
    let mut v = vec![0.0f32; n * dim];
    q.par_chunks_mut(dim)
        .zip(k.par_chunks_mut(dim))
        .zip(v.par_chunks_mut(dim))
        .zip(rows.par_chunks(3 * dim))
        .for_each(|(((q, k), v), row)| {
            q.copy_from_slice(&row[..dim]);
            k.copy_from_slice(&row[dim..2 * dim]);
            v.copy_from_slice(&row[2 * dim..]);
        });
    (q, k, v)
}

/// The SITU activation of one value pair: `4 tanh(g / 4) σ(g) · 25 tanh(h /
/// 25)`.
#[inline]
fn situ_one(hidden: f32, gate: f32) -> f32 {
    (4.0 * (gate / 4.0).tanh() * tensor::sigmoid(gate)) * (25.0 * (hidden / 25.0).tanh())
}

/// SITU over two separate projections of the same rows.
fn situ(hidden: &[f32], gate: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden.len()];
    out.par_iter_mut()
        .zip(hidden.par_iter().zip(gate.par_iter()))
        .for_each(|(o, (&h, &g))| *o = situ_one(h, g));
    out
}

/// SITU over rows of `2 * hidden`: `[hidden | gate]`.
fn situ_fused(rows: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows.len() / 2];
    out.par_chunks_mut(hidden)
        .zip(rows.par_chunks(2 * hidden))
        .for_each(|(o, row)| {
            let (h, g) = row.split_at(hidden);
            for ((o, &h), &g) in o.iter_mut().zip(h).zip(g) {
                *o = situ_one(h, g);
            }
        });
    out
}

/// `(cos, sin)` for every rotated pair of the head, for every row of the
/// joint sequence `[image | text]` (`grid` is `None` for the text alone):
/// pair `j = a * B + b` turns at `position[a] · base^(-b / B)`, `B =
/// head_dim / 8` frequencies an axis — the reference's `_RoPE`.
fn rope_table(head_dim: usize, grid: Option<(usize, usize)>, n_txt: usize) -> Vec<(f32, f32)> {
    let pairs = head_dim / 2;
    let bands = pairs / ROPE_AXES;
    let freqs: Vec<f64> = (0..bands)
        .map(|b| 1.0 / ROPE_BASE.powf(b as f64 / bands as f64))
        .collect();
    let (rows, cols) = grid.unwrap_or((0, 0));
    let mut table = Vec::with_capacity((rows * cols + n_txt) * pairs);
    let mut push = |pos: [f64; ROPE_AXES]| {
        for p in pos {
            for f in &freqs {
                // The phase is single precision in the reference.
                let a = (p as f32 * *f as f32) as f64;
                table.push((a.cos() as f32, a.sin() as f32));
            }
        }
    };
    let (r0, c0) = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
    for r in 0..rows {
        for col in 0..cols {
            push([0.0, n_txt as f64, r as f64 - r0, col as f64 - c0]);
        }
    }
    for i in 0..n_txt {
        push([0.0, i as f64, 0.0, 0.0]);
    }
    table
}

/// Rotates every head of every row of `x` (`[n, dim]`) by that row's
/// angles, rotate-half style: pair `j` is `(x[j], x[j + head_dim / 2])`.
fn apply_rope(x: &mut [f32], dim: usize, head_dim: usize, table: &[(f32, f32)]) {
    let pairs = head_dim / 2;
    x.par_chunks_mut(dim)
        .zip(table.par_chunks(pairs))
        .for_each(|(row, angles)| {
            for head in row.chunks_mut(head_dim) {
                let (lo, hi) = head.split_at_mut(pairs);
                for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi.iter_mut()).zip(angles) {
                    let (x0, x1) = (*a, *b);
                    *a = x0 * cos - x1 * sin;
                    *b = x1 * cos + x0 * sin;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture token is centred on the grid and placed after the text on
    /// the second axis; a text token turns only on the second axis, by its
    /// index; the first axis never turns.
    #[test]
    fn rope_positions_centre_the_picture_and_index_the_text() {
        let table = rope_table(128, Some((2, 3)), 2);
        let pairs = 64;
        assert_eq!(table.len(), (6 + 2) * pairs);
        let at = |row: usize, pair: usize| table[row * pairs + pair];
        let f = |b: usize| 1.0 / ROPE_BASE.powf(b as f64 / 16.0);
        // Picture token (row 1, column 2): (0, 2, 0.5, 1).
        let p = 3 + 2;
        assert_eq!(at(p, 0), (1.0, 0.0));
        assert!((at(p, 16 + 1).1 - (2.0 * f(1)).sin() as f32).abs() < 1e-6);
        assert!((at(p, 32 + 1).1 - (0.5 * f(1)).sin() as f32).abs() < 1e-6);
        assert!((at(p, 48 + 3).1 - (f(3)).sin() as f32).abs() < 1e-6);
        // Picture token (0, 0): (0, 2, -0.5, -1).
        assert!((at(0, 32).1 - (-0.5f64).sin() as f32).abs() < 1e-6);
        // Text token 1 after the six picture tokens: (0, 1, 0, 0).
        let t = 6 + 1;
        assert!((at(t, 16 + 2).1 - (f(2)).sin() as f32).abs() < 1e-6);
        assert_eq!(at(t, 32 + 2), (1.0, 0.0));
        assert_eq!(at(t, 48), (1.0, 0.0));
    }

    /// Rotate-half pairs element `j` with `j + head_dim / 2`.
    #[test]
    fn rope_rotates_the_two_halves() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
        let a = 0.5f32;
        apply_rope(&mut x, 4, 4, &[(a.cos(), a.sin()), (a.cos(), a.sin())]);
        // Pair 0 is (x0, x2) = (1, 0); pair 1 is (x1, x3) = (0, 1).
        assert!((x[0] - a.cos()).abs() < 1e-6 && (x[2] - a.sin()).abs() < 1e-6);
        assert!((x[1] + a.sin()).abs() < 1e-6 && (x[3] - a.cos()).abs() < 1e-6);
    }

    #[test]
    fn situ_matches_the_reference_formula() {
        let (h, g) = (30.0f32, -2.0f32);
        let want =
            (4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp()))) * (25.0 * (h / 25.0).tanh());
        assert!((situ_one(h, g) - want).abs() < 1e-5);
        // Fused rows are `[hidden | gate]`.
        let out = situ_fused(&[h, 1.0, g, 0.0], 2);
        assert!((out[0] - want).abs() < 1e-5);
        assert_eq!(out[1], 0.0);
        assert_eq!(situ(&[h], &[g]), vec![situ_one(h, g)]);
    }
}

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

//! The pieces LTX-2's transformer and its text connectors share: a linear
//! with an optional bias, the parameter-free norms, the gated multi-head
//! attention every block is built from, and LTX's *split* rotary
//! embedding.
//!
//! Activations are row-major `[tokens, width]` `f32` throughout, as in the
//! rest of the engine; the linears go through [`Backend::matmul_batch`].

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

/// `y = x · Wᵀ + b`.
pub struct Linear {
    pub w: QuantMatrix,
    pub b: Option<Vec<f32>>,
}

impl Linear {
    /// `{name}.weight` and, when the file has one, `{name}.bias`.
    pub fn load(loaded: &LoadedModel, name: &str) -> Result<Self> {
        let w = loaded
            .matrix(&format!("{name}.weight"))
            .with_context(|| format!("ltxv: loading {name}"))?;
        let bias = format!("{name}.bias");
        let b = if loaded.has_tensor(&bias) {
            let (values, _) = loaded.tensor(&bias)?;
            ensure!(
                values.len() == w.out_dim,
                "ltxv: {bias} has {} values for {} outputs",
                values.len(),
                w.out_dim
            );
            Some(values)
        } else {
            None
        };
        Ok(Self { w, b })
    }

    pub fn in_dim(&self) -> usize {
        self.w.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.w.out_dim
    }

    pub fn forward(&self, backend: &dyn Backend, x: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * self.w.in_dim);
        let mut y = backend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: n,
                w: &self.w,
            }])
            .pop()
            .expect("one op in, one result out");
        if let Some(b) = &self.b {
            add_row_bias(&mut y, b);
        }
        y
    }
}

/// Several linears of the same rows as one batch (the backend stages the
/// input once).
pub fn forward_many<const N: usize>(
    backend: &dyn Backend,
    x: &[f32],
    n: usize,
    layers: [&Linear; N],
) -> [Vec<f32>; N] {
    let ops = layers.map(|l| MatmulOp {
        x,
        n_tokens: n,
        w: &l.w,
    });
    let mut results = backend.matmul_batch(&ops).into_iter();
    let mut out: [Vec<f32>; N] = std::array::from_fn(|_| results.next().expect("one per op"));
    for (y, l) in out.iter_mut().zip(layers) {
        if let Some(b) = &l.b {
            add_row_bias(y, b);
        }
    }
    out
}

pub fn add_row_bias(y: &mut [f32], b: &[f32]) {
    y.par_chunks_mut(b.len()).for_each(|row| {
        for (v, b) in row.iter_mut().zip(b) {
            *v += b;
        }
    });
}

/// A 1-D tensor of exactly `len` values.
pub fn vector(loaded: &LoadedModel, name: &str, len: usize) -> Result<Vec<f32>> {
    let (values, _) = loaded
        .tensor(name)
        .with_context(|| format!("ltxv: loading {name}"))?;
    ensure!(
        values.len() == len,
        "ltxv: {name} has {} values, expected {len}",
        values.len()
    );
    Ok(values)
}

/// Each row of `x` divided by its root mean square — `torch`'s `rms_norm`
/// with no weight.
pub fn rms_norm_rows(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_mut(dim).for_each(|row| {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        row.iter_mut().for_each(|v| *v *= scale);
    });
    out
}

/// Each row normalized to zero mean and unit variance — `LayerNorm` with no
/// affine.
pub fn layer_norm_rows(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_mut(dim).for_each(|row| {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let scale = 1.0 / (var + eps).sqrt();
        row.iter_mut().for_each(|v| *v = (*v - mean) * scale);
    });
    out
}

/// `x * (1 + scale) + shift`, row by row, with `scale` and `shift` chosen
/// per row by `group[row]` out of `[groups, dim]` tables — every token of
/// a step shares one timestep, or one of a few (a conditioning frame's
/// tokens sit at zero).
pub fn modulate(x: &mut [f32], dim: usize, shift: &[f32], scale: &[f32], group: &[usize]) {
    x.par_chunks_mut(dim)
        .zip(group.par_iter())
        .for_each(|(row, &g)| {
            let (sh, sc) = (&shift[g * dim..][..dim], &scale[g * dim..][..dim]);
            for ((v, sh), sc) in row.iter_mut().zip(sh).zip(sc) {
                *v = *v * (1.0 + sc) + sh;
            }
        });
}

/// `x += y * gate`, the gate chosen per row as [`modulate`]'s tables are.
pub fn gated_add(x: &mut [f32], y: &[f32], dim: usize, gate: &[f32], group: &[usize]) {
    x.par_chunks_mut(dim)
        .zip(y.par_chunks(dim))
        .zip(group.par_iter())
        .for_each(|((row, y), &g)| {
            let gate = &gate[g * dim..][..dim];
            for ((v, y), g) in row.iter_mut().zip(y).zip(gate) {
                *v += y * g;
            }
        });
}

/// GELU with `tanh` — `torch.nn.functional.gelu(approximate="tanh")`.
pub fn gelu_tanh(x: &mut [f32]) {
    tensor::gelu_inplace(x);
}

pub fn silu(x: &mut [f32]) {
    x.par_iter_mut().for_each(|v| *v = tensor::silu(*v));
}

pub fn silu_one(v: f32) -> f32 {
    tensor::silu(v)
}

/// A rotary table: `cos` and `sin` per row, per head, per frequency —
/// `[n, heads, head_dim / 2]` — for LTX's *split* rotation, which turns the
/// first half of each head against the second (`apply_split_rotary_emb`).
pub struct Rope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub heads: usize,
    pub half: usize,
}

impl Rope {
    /// `ltx_core`'s `precompute_freqs_cis` for SPLIT rope: `positions` holds
    /// each row's coordinate on every axis (`[n][axes]`, already the
    /// midpoint of the patch's bounds where the model uses
    /// `use_middle_indices_grid`), `max_pos` their extent. The frequency
    /// grid is `theta ** linspace(0, 1, inner / (2 · axes)) · π/2`, computed
    /// in `f64` (`frequencies_precision = float64`) and kept in `f32`; each
    /// row's angles are `grid[f] · (2 · pos[a] / max_pos[a] − 1)`, laid out
    /// frequency-major and axis-minor, and padded at the front to
    /// `inner / 2` with angle zero — then split across the heads in order.
    pub fn new(
        positions: &[f32],
        axes: usize,
        max_pos: &[f32],
        inner: usize,
        heads: usize,
        theta: f64,
    ) -> Self {
        assert_eq!(max_pos.len(), axes);
        let n = positions.len() / axes;
        let nf = inner / (2 * axes);
        let grid: Vec<f32> = (0..nf)
            .map(|i| {
                let t = if nf > 1 {
                    i as f64 / (nf - 1) as f64
                } else {
                    0.0
                };
                (theta.powf(t) * std::f64::consts::PI / 2.0) as f32
            })
            .collect();
        let half_total = inner / 2;
        let pad = half_total - nf * axes;
        let half = half_total / heads;
        let mut cos = vec![1.0f32; n * half_total];
        let mut sin = vec![0.0f32; n * half_total];
        cos.par_chunks_mut(half_total)
            .zip(sin.par_chunks_mut(half_total))
            .enumerate()
            .for_each(|(t, (c, s))| {
                let pos = &positions[t * axes..][..axes];
                for (f, &g) in grid.iter().enumerate() {
                    for (a, (&p, &m)) in pos.iter().zip(max_pos).enumerate() {
                        let angle = g * (p / m * 2.0 - 1.0);
                        let at = pad + f * axes + a;
                        c[at] = angle.cos();
                        s[at] = angle.sin();
                    }
                }
            });
        Self {
            cos,
            sin,
            heads,
            half,
        }
    }

    /// Rotates every head of every row of `x` (`[n, heads · 2 · half]`).
    pub fn apply(&self, x: &mut [f32]) {
        let (heads, half) = (self.heads, self.half);
        let width = heads * 2 * half;
        x.par_chunks_mut(width)
            .zip(self.cos.par_chunks(heads * half))
            .zip(self.sin.par_chunks(heads * half))
            .for_each(|((row, cos), sin)| {
                for h in 0..heads {
                    let head = &mut row[h * 2 * half..][..2 * half];
                    let (c, s) = (&cos[h * half..][..half], &sin[h * half..][..half]);
                    let (a, b) = head.split_at_mut(half);
                    for i in 0..half {
                        let (x1, x2) = (a[i], b[i]);
                        a[i] = x1 * c[i] - x2 * s[i];
                        b[i] = x2 * c[i] + x1 * s[i];
                    }
                }
            });
    }
}

/// `ltx_core`'s `Attention`: query, key and value projections, RMS norms
/// over the whole query and key widths (with a weight), rotary positions,
/// scaled dot-product attention, an optional per-head gate
/// `2 · sigmoid(to_gate_logits(x))`, and the output projection.
pub struct Attention {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate: Option<Linear>,
    pub heads: usize,
    pub head_dim: usize,
}

/// Keys and values already projected, normed and rotated — what a
/// cross-attention whose context does not change can keep.
pub struct KeyValues {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n: usize,
}

impl Attention {
    pub fn load(loaded: &LoadedModel, name: &str, heads: usize) -> Result<Self> {
        let to_q = Linear::load(loaded, &format!("{name}.to_q"))?;
        let inner = to_q.out_dim();
        ensure!(
            inner.is_multiple_of(heads),
            "ltxv: {name} is {inner} wide, not a multiple of {heads} heads"
        );
        let gate_name = format!("{name}.to_gate_logits");
        let gate = if loaded.has_tensor(&format!("{gate_name}.weight")) {
            Some(Linear::load(loaded, &gate_name)?)
        } else {
            None
        };
        Ok(Self {
            to_k: Linear::load(loaded, &format!("{name}.to_k"))?,
            to_v: Linear::load(loaded, &format!("{name}.to_v"))?,
            to_out: Linear::load(loaded, &format!("{name}.to_out.0"))?,
            q_norm: vector(loaded, &format!("{name}.q_norm.weight"), inner)?,
            k_norm: vector(loaded, &format!("{name}.k_norm.weight"), inner)?,
            gate,
            heads,
            head_dim: inner / heads,
            to_q,
        })
    }

    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    /// The context's keys and values: `to_k` normed and rotated by `k_pe`,
    /// `to_v` as it is.
    pub fn key_values(
        &self,
        backend: &dyn Backend,
        ctx: &[f32],
        n_ctx: usize,
        k_pe: Option<&Rope>,
        eps: f32,
    ) -> KeyValues {
        let [mut k, v] = forward_many(backend, ctx, n_ctx, [&self.to_k, &self.to_v]);
        tensor::rmsnorm_inplace(&mut k, &self.k_norm, n_ctx, self.inner(), eps);
        if let Some(pe) = k_pe {
            pe.apply(&mut k);
        }
        KeyValues { k, v, n: n_ctx }
    }

    /// Attention of the `n` rows of `x` over `kv`, through the gate and
    /// `to_out`.
    pub fn attend(
        &self,
        backend: &dyn Backend,
        x: &[f32],
        n: usize,
        kv: &KeyValues,
        pe: Option<&Rope>,
        eps: f32,
    ) -> Vec<f32> {
        let mut q = self.to_q.forward(backend, x, n);
        tensor::rmsnorm_inplace(&mut q, &self.q_norm, n, self.inner(), eps);
        if let Some(pe) = pe {
            pe.apply(&mut q);
        }
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut out = crate::engine::image::transformer::step_attention(
            &q,
            n,
            &kv.k,
            &kv.v,
            kv.n,
            self.heads,
            self.head_dim,
            scale,
            None,
        );
        drop(q);
        if let Some(gate) = &self.gate {
            let logits = gate.forward(backend, x, n);
            let (heads, hd) = (self.heads, self.head_dim);
            out.par_chunks_mut(heads * hd)
                .zip(logits.par_chunks(heads))
                .for_each(|(row, logits)| {
                    for (head, &l) in row.chunks_mut(hd).zip(logits) {
                        let g = 2.0 * tensor::sigmoid(l);
                        head.iter_mut().for_each(|v| *v *= g);
                    }
                });
        }
        self.to_out.forward(backend, &out, n)
    }

    /// Self-attention of the rows of `x`.
    pub fn self_attend(
        &self,
        backend: &dyn Backend,
        x: &[f32],
        n: usize,
        pe: Option<&Rope>,
        eps: f32,
    ) -> Vec<f32> {
        let kv = self.key_values(backend, x, n, pe, eps);
        self.attend(backend, x, n, &kv, pe, eps)
    }
}

/// `ltx_core`'s `FeedForward`: `Linear(d, 4d)`, GELU (`tanh`), `Linear(4d, d)`.
pub struct FeedForward {
    pub up: Linear,
    pub down: Linear,
}

impl FeedForward {
    pub fn load(loaded: &LoadedModel, name: &str) -> Result<Self> {
        Ok(Self {
            up: Linear::load(loaded, &format!("{name}.net.0.proj"))?,
            down: Linear::load(loaded, &format!("{name}.net.2"))?,
        })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &[f32], n: usize) -> Vec<f32> {
        let mut h = self.up.forward(backend, x, n);
        gelu_tanh(&mut h);
        self.down.forward(backend, &h, n)
    }
}

/// The sinusoidal timestep embedding diffusers' `Timesteps(256,
/// flip_sin_to_cos=True, downscale_freq_shift=0)` makes: `[cos | sin]` of
/// `t · exp(−ln(10000) · i / 128)`.
pub fn timestep_embedding(t: f32) -> Vec<f32> {
    const HALF: usize = 128;
    let mut out = vec![0.0f32; 2 * HALF];
    for i in 0..HALF {
        let freq = (-(10000f64.ln()) * i as f64 / HALF as f64).exp();
        let arg = t as f64 * freq;
        out[i] = arg.cos() as f32;
        out[HALF + i] = arg.sin() as f32;
    }
    out
}

/// PixArt-α's `AdaLayerNormSingle`: the timestep's sinusoid through
/// `linear_1`, SiLU, `linear_2` — the *embedded timestep* — and that,
/// through SiLU and `linear`, the block modulation.
pub struct AdaLayerNorm {
    linear_1: Linear,
    linear_2: Linear,
    linear: Linear,
}

impl AdaLayerNorm {
    pub fn load(loaded: &LoadedModel, name: &str) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(loaded, &format!("{name}.emb.timestep_embedder.linear_1"))?,
            linear_2: Linear::load(loaded, &format!("{name}.emb.timestep_embedder.linear_2"))?,
            linear: Linear::load(loaded, &format!("{name}.linear"))?,
        })
    }

    /// `(modulation, embedded_timestep)` for each of `timesteps`, already
    /// scaled (σ · 1000) — `[len, coef · dim]` and `[len, dim]`.
    pub fn forward(&self, backend: &dyn Backend, timesteps: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = timesteps.len();
        let sinusoids: Vec<f32> = timesteps
            .iter()
            .flat_map(|&t| timestep_embedding(t))
            .collect();
        let mut h = self.linear_1.forward(backend, &sinusoids, n);
        silu(&mut h);
        let emb = self.linear_2.forward(backend, &h, n);
        let mut act = emb.clone();
        silu(&mut act);
        (self.linear.forward(backend, &act, n), emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_rotation_turns_the_first_half_against_the_second() {
        // One row, one axis at the far end of its range: angle = grid · 1.
        let pe = Rope::new(&[1.0], 1, &[1.0], 4, 1, 10000.0);
        assert_eq!(pe.half, 2);
        // nf = 4 / 2 = 2: grid = [π/2, 10000 · π/2]; no padding.
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        pe.apply(&mut x);
        // First pair: (x1, x2) = (1, 0) at π/2 → (cos, sin) = (0, 1).
        assert!((x[0] - 0.0).abs() < 1e-6 && (x[2] - 1.0).abs() < 1e-6);
        let a = (10000.0f64 * std::f64::consts::PI / 2.0) as f32;
        assert!((x[1] - (-a.sin())).abs() < 1e-4 && (x[3] - a.cos()).abs() < 1e-4);
    }

    #[test]
    fn rope_pads_unused_frequencies_at_the_front() {
        // Three axes over 8 wide: 8 / 6 = 1 frequency per axis, 3 of 4
        // slots used, the first padded.
        let pe = Rope::new(&[0.5, 0.5, 0.5], 3, &[1.0, 1.0, 1.0], 8, 2, 10000.0);
        assert_eq!(pe.cos[0], 1.0);
        assert_eq!(pe.sin[0], 0.0);
        // Position at the middle of its range: angle zero everywhere.
        assert!(pe.sin.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn timestep_embedding_is_cos_then_sin() {
        let e = timestep_embedding(0.0);
        assert!(e[..128].iter().all(|&v| v == 1.0));
        assert!(e[128..].iter().all(|&v| v == 0.0));
    }
}

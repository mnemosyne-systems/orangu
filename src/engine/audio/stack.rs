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

//! The decoder stack Qwen3-TTS's talker and its code predictor share: a
//! Qwen3 transformer — pre-norm RMSNorm, grouped-query attention with an
//! RMSNorm over each query and key head, NeoX rotary positions, a SwiGLU
//! feed-forward — run over input *embeddings* rather than token ids, with
//! its own key/value cache, ending in the final norm.
//!
//! The talker's rotary embedding is nominally multimodal (three position
//! streams, `mrope_section = [24, 20, 20]`, interleaved); speech carries
//! no picture, so the three streams always hold the same position and the
//! rotation is exactly the single-stream one.
//!
//! Activations are row-major `[rows, width]` `f32`; the linears go through
//! [`Backend::matmul_batch`].

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

/// One decoder layer's weights.
struct Layer {
    attn_norm: Vec<f32>,
    q: QuantMatrix,
    k: QuantMatrix,
    v: QuantMatrix,
    o: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
}

/// The shape of a stack, as the file's metadata gives it.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
}

/// A Qwen3 decoder over embeddings.
pub struct Stack {
    layers: Vec<Layer>,
    norm: Vec<f32>,
    /// The residual stream's width.
    pub dim: usize,
    shape: Shape,
    /// `theta^(-2i/head_dim)` for each rotated pair.
    inv_freq: Vec<f32>,
}

/// A sequence's keys and values, one `[positions, kv_heads · head_dim]`
/// buffer per layer.
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl Cache {
    /// Empties the cache, keeping its allocations.
    pub fn clear(&mut self) {
        self.k.iter_mut().for_each(Vec::clear);
        self.v.iter_mut().for_each(Vec::clear);
        self.len = 0;
    }
}

pub(super) fn vector(loaded: &LoadedModel, name: &str, len: usize) -> Result<Vec<f32>> {
    let (values, _) = loaded
        .tensor(name)
        .with_context(|| format!("qwen3tts: loading {name}"))?;
    ensure!(
        values.len() == len,
        "qwen3tts: {name} has {} values, expected {len}",
        values.len()
    );
    Ok(values)
}

pub(super) fn matrix(
    loaded: &LoadedModel,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<QuantMatrix> {
    let w = loaded
        .matrix(name)
        .with_context(|| format!("qwen3tts: loading {name}"))?;
    ensure!(
        w.in_dim == in_dim && w.out_dim == out_dim,
        "qwen3tts: {name} is {}x{}, expected {in_dim}x{out_dim}",
        w.in_dim,
        w.out_dim
    );
    Ok(w)
}

impl Stack {
    /// `{prefix}.blk.N.*` for every `N` the file has, and
    /// `{prefix}.output_norm.weight`.
    pub fn load(loaded: &LoadedModel, prefix: &str, dim: usize, shape: Shape) -> Result<Self> {
        let Shape {
            heads,
            kv_heads,
            head_dim,
            ..
        } = shape;
        ensure!(
            heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads) && head_dim % 2 == 0,
            "qwen3tts: {prefix} has {heads} query heads over {kv_heads} key/value heads of \
             {head_dim}"
        );
        let mut layers = Vec::new();
        while loaded.has_tensor(&format!("{prefix}.blk.{}.attn_q.weight", layers.len())) {
            let p = format!("{prefix}.blk.{}", layers.len());
            let ff = loaded
                .matrix(&format!("{p}.ffn_gate.weight"))
                .with_context(|| format!("qwen3tts: loading {p}.ffn_gate.weight"))?
                .out_dim;
            layers.push(Layer {
                attn_norm: vector(loaded, &format!("{p}.attn_norm.weight"), dim)?,
                q: matrix(loaded, &format!("{p}.attn_q.weight"), dim, heads * head_dim)?,
                k: matrix(
                    loaded,
                    &format!("{p}.attn_k.weight"),
                    dim,
                    kv_heads * head_dim,
                )?,
                v: matrix(
                    loaded,
                    &format!("{p}.attn_v.weight"),
                    dim,
                    kv_heads * head_dim,
                )?,
                o: matrix(
                    loaded,
                    &format!("{p}.attn_output.weight"),
                    heads * head_dim,
                    dim,
                )?,
                q_norm: vector(loaded, &format!("{p}.attn_q_norm.weight"), head_dim)?,
                k_norm: vector(loaded, &format!("{p}.attn_k_norm.weight"), head_dim)?,
                ffn_norm: vector(loaded, &format!("{p}.ffn_norm.weight"), dim)?,
                gate: matrix(loaded, &format!("{p}.ffn_gate.weight"), dim, ff)?,
                up: matrix(loaded, &format!("{p}.ffn_up.weight"), dim, ff)?,
                down: matrix(loaded, &format!("{p}.ffn_down.weight"), ff, dim)?,
            });
        }
        ensure!(!layers.is_empty(), "qwen3tts: {prefix} has no layers");
        let inv_freq = inv_freq(shape.theta, head_dim);
        Ok(Self {
            layers,
            norm: vector(loaded, &format!("{prefix}.output_norm.weight"), dim)?,
            dim,
            shape,
            inv_freq,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn new_cache(&self) -> Cache {
        Cache {
            k: vec![Vec::new(); self.layers.len()],
            v: vec![Vec::new(); self.layers.len()],
            len: 0,
        }
    }

    /// The `n` rows of `x` (`[n, dim]`) through every layer, appended to
    /// `cache` at the positions after what it holds, then the final norm.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        x: &[f32],
        n: usize,
        cache: &mut Cache,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * self.dim);
        let Shape {
            heads,
            kv_heads,
            head_dim: hd,
            eps,
            ..
        } = self.shape;
        let dim = self.dim;
        let start = cache.len;
        let mut h = x.to_vec();
        for (il, layer) in self.layers.iter().enumerate() {
            let mut xn = h.clone();
            tensor::rmsnorm_inplace(&mut xn, &layer.attn_norm, n, dim, eps);
            let mut qkv = backend
                .matmul_batch(&[
                    MatmulOp {
                        x: &xn,
                        n_tokens: n,
                        w: &layer.q,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: n,
                        w: &layer.k,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: n,
                        w: &layer.v,
                    },
                ])
                .into_iter();
            let (mut q, mut k, v) = (
                qkv.next().expect("three ops"),
                qkv.next().expect("three ops"),
                qkv.next().expect("three ops"),
            );
            tensor::rmsnorm_inplace(&mut q, &layer.q_norm, n * heads, hd, eps);
            tensor::rmsnorm_inplace(&mut k, &layer.k_norm, n * kv_heads, hd, eps);
            rope(&mut q, n, heads, hd, &self.inv_freq, start);
            rope(&mut k, n, kv_heads, hd, &self.inv_freq, start);
            cache.k[il].extend_from_slice(&k);
            cache.v[il].extend_from_slice(&v);
            let attn = causal_attention(
                &q,
                n,
                &cache.k[il],
                &cache.v[il],
                start,
                AttentionShape {
                    heads,
                    kv_heads,
                    head_dim: hd,
                    window: None,
                },
            );
            let o = backend
                .matmul_batch(&[MatmulOp {
                    x: &attn,
                    n_tokens: n,
                    w: &layer.o,
                }])
                .pop()
                .expect("one op");
            tensor::add_inplace(&mut h, &o);

            let mut xn = h.clone();
            tensor::rmsnorm_inplace(&mut xn, &layer.ffn_norm, n, dim, eps);
            let mut gu = backend
                .matmul_batch(&[
                    MatmulOp {
                        x: &xn,
                        n_tokens: n,
                        w: &layer.gate,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: n,
                        w: &layer.up,
                    },
                ])
                .into_iter();
            let (mut g, u) = (gu.next().expect("two ops"), gu.next().expect("two ops"));
            g.par_iter_mut()
                .zip(u.par_iter())
                .for_each(|(g, u)| *g = tensor::silu(*g) * u);
            let d = backend
                .matmul_batch(&[MatmulOp {
                    x: &g,
                    n_tokens: n,
                    w: &layer.down,
                }])
                .pop()
                .expect("one op");
            tensor::add_inplace(&mut h, &d);
        }
        cache.len += n;
        tensor::rmsnorm_inplace(&mut h, &self.norm, n, dim, eps);
        h
    }
}

/// `theta^(-2i/head_dim)` for each of a head's rotated pairs.
pub fn inv_freq(theta: f32, head_dim: usize) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| (theta as f64).powf(-(2.0 * i as f64) / head_dim as f64) as f32)
        .collect()
}

/// Rotates every head of the `n` rows of `x` (`[n, heads · head_dim]`), row
/// `t` at position `start + t`, NeoX style: the first half of a head turns
/// against the second.
pub fn rope(
    x: &mut [f32],
    n: usize,
    heads: usize,
    head_dim: usize,
    inv_freq: &[f32],
    start: usize,
) {
    let half = head_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    x.par_chunks_mut(heads * head_dim)
        .take(n)
        .enumerate()
        .for_each(|(t, row)| {
            let pos = (start + t) as f32;
            let (cos, sin): (Vec<f32>, Vec<f32>) = inv_freq
                .iter()
                .map(|f| {
                    let a = pos * f;
                    (a.cos(), a.sin())
                })
                .unzip();
            for head in row.chunks_mut(head_dim) {
                let (a, b) = head.split_at_mut(half);
                for i in 0..half {
                    let (x1, x2) = (a[i], b[i]);
                    a[i] = x1 * cos[i] - x2 * sin[i];
                    b[i] = x2 * cos[i] + x1 * sin[i];
                }
            }
        });
}

/// The geometry [`causal_attention`] runs at.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// How many keys back a query sees, itself included — `None` for all.
    pub window: Option<usize>,
}

/// Causal attention of `n` query rows, row `t` at position `start + t`,
/// over the keys and values at positions `0..start + n` (`k`, `v`:
/// `[positions, kv_heads · head_dim]`). Query head `h` reads key/value head
/// `h / (heads / kv_heads)`. Returns `[n, heads · head_dim]`.
pub fn causal_attention(
    q: &[f32],
    n: usize,
    k: &[f32],
    v: &[f32],
    start: usize,
    shape: AttentionShape,
) -> Vec<f32> {
    let AttentionShape {
        heads,
        kv_heads,
        head_dim: hd,
        window,
    } = shape;
    let group = heads / kv_heads;
    let kv_width = kv_heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; n * heads * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(index, o)| {
        let (t, h) = (index / heads, index % heads);
        let pos = start + t;
        let first = window.map_or(0, |w| (pos + 1).saturating_sub(w));
        let qh = &q[(t * heads + h) * hd..][..hd];
        let kvh = h / group;
        let mut scores: Vec<f32> = (first..=pos)
            .map(|j| tensor::dot(qh, &k[j * kv_width + kvh * hd..][..hd]) * scale)
            .collect();
        tensor::softmax_inplace(&mut scores);
        for (j, s) in (first..=pos).zip(&scores) {
            tensor::axpy_inplace(o, &v[j * kv_width + kvh * hd..][..hd], *s);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_turns_the_first_half_of_a_head_against_the_second() {
        // One head of 2: a single pair at frequency 1, position 1.
        let mut x = vec![1.0, 0.0];
        rope(&mut x, 1, 1, 2, &inv_freq(10000.0, 2), 1);
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[1] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn a_single_key_is_returned_as_its_value() {
        let shape = AttentionShape {
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            window: None,
        };
        let out = causal_attention(&[1.0, 0.0, 0.0, 1.0], 1, &[1.0, 1.0], &[3.0, 4.0], 0, shape);
        assert_eq!(out, vec![3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn the_window_hides_keys_older_than_it() {
        // Three positions, a window of one: each query sees only itself.
        let shape = AttentionShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 1,
            window: Some(1),
        };
        let out = causal_attention(
            &[1.0, 1.0, 1.0],
            3,
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            0,
            shape,
        );
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_causal_query_averages_the_keys_it_sees() {
        let shape = AttentionShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 1,
            window: None,
        };
        // Equal scores: the second query is the mean of the first two values.
        let out = causal_attention(&[0.0, 0.0], 2, &[0.0, 0.0], &[2.0, 4.0], 0, shape);
        assert_eq!(out, vec![2.0, 3.0]);
    }
}

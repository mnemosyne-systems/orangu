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

//! Qwen3-VL reading a picture — what Qwen-Image 2.1's editing conditions
//! on. Two halves:
//!
//! - [`VisionTower`]: the ViT in the `mmproj-*.gguf` llama.cpp writes beside
//!   a Qwen3-VL GGUF (`clip.projector_type = qwen3vl_merger`). 16-pixel
//!   patches (the two temporal kernels summed: a still picture is two
//!   identical frames), a learned 48x48 position table interpolated to the
//!   picture's grid, 2-D rotary attention, and a merger that folds each 2x2
//!   block of patches into one token of the language model's width — plus
//!   three **DeepStack** features taken after blocks 8, 16 and 24.
//! - [`TextModel`]: the language model with those tokens in it — a
//!   reference forward of the `qwen3vl` text model that differs from the
//!   served path (`arch::llama`) in exactly the two ways a picture needs:
//!   *interleaved M-RoPE* (a picture token's rotary position is
//!   `(t, row, column)`, spread over the head's frequencies `t h w t h w
//!   …`), and the DeepStack features added to the picture's rows after the
//!   first three layers. Text alone reduces to the served path, which a
//!   test holds it to.
//!
//! Both are diffusers' `Qwen3VLForConditionalGeneration` as
//! `QwenImage21Pipeline` calls it, cross-read against llama.cpp's
//! `tools/mtmd/models/qwen3vl.cpp` and `src/models/qwen3vl.cpp`.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::transformer::{causal_limits, step_attention};
use super::vae::{Feature, VaePrecision, conv_matrix, row_matrix};

/// Pixels per patch side.
const PATCH: usize = 16;
/// Patches per merged token side.
const MERGE: usize = 2;
/// Pixels a picture's side must be a multiple of.
pub const UNIT: usize = PATCH * MERGE;

struct VisionBlock {
    ln1: (Vec<f32>, Vec<f32>),
    qkv: QuantMatrix,
    qkv_b: Vec<f32>,
    out: QuantMatrix,
    out_b: Vec<f32>,
    ln2: (Vec<f32>, Vec<f32>),
    up: QuantMatrix,
    up_b: Vec<f32>,
    down: QuantMatrix,
    down_b: Vec<f32>,
}

/// `LayerNorm? · fc1 · GELU · fc2` over merged 2x2 blocks — the final
/// merger (norm before the merge, `post_ln`) and the DeepStack ones (norm
/// after it).
struct Merger {
    fc1: QuantMatrix,
    fc1_b: Vec<f32>,
    fc2: QuantMatrix,
    fc2_b: Vec<f32>,
}

struct Deepstack {
    after_block: usize,
    norm: (Vec<f32>, Vec<f32>),
    merger: Merger,
}

pub struct VisionTower {
    backend: Arc<dyn Backend>,
    width: usize,
    n_head: usize,
    eps: f32,
    /// `[768 → width]`, `(channel, ky, kx)` inputs: both temporal kernels.
    patch: QuantMatrix,
    patch_b: Vec<f32>,
    /// `[side * side][width]`.
    pos: Vec<f32>,
    side: usize,
    blocks: Vec<VisionBlock>,
    post_ln: (Vec<f32>, Vec<f32>),
    merger: Merger,
    deepstack: Vec<Deepstack>,
    /// The language model's width the merger writes.
    pub out_width: usize,
    /// The requantized matrices as per-row `int8` for the `smmla` tile, by
    /// the address of their `Q6_K` bytes — what `linear` runs on a CPU with
    /// `i8mm` ([`super::vae::row_matrix`]).
    rowi8: std::collections::HashMap<usize, crate::engine::vecdot::RowI8>,
}

/// A picture as the language model reads it.
pub struct VisionOutput {
    /// Merged tokens per side: `(rows, cols)` of 32-pixel cells.
    pub grid: (usize, usize),
    /// `[rows * cols, out_width]`, row-major.
    pub embeds: Vec<f32>,
    /// One `[rows * cols, out_width]` per DeepStack layer, in order.
    pub deepstack: Vec<Vec<f32>>,
}

impl VisionTower {
    pub fn load(path: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let loaded = LoadedModel::open_projector(path)
            .with_context(|| format!("opening the vision projector {}", path.display()))?;
        let meta_u64 = |key: &str| -> Option<u64> {
            loaded
                .metadata
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.as_u64())
        };
        let meta_str = |key: &str| -> Option<String> {
            loaded
                .metadata
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| match v {
                    orangu::gguf::GgufValue::String(s) => Some(s.clone()),
                    _ => None,
                })
        };
        ensure!(
            meta_str("clip.projector_type").as_deref() == Some("qwen3vl_merger"),
            "{} is not a Qwen3-VL vision projector (clip.projector_type = qwen3vl_merger)",
            path.display()
        );
        ensure!(
            meta_u64("clip.vision.patch_size") == Some(PATCH as u64),
            "the vision projector's patches are not {PATCH} pixels"
        );
        let width = meta_u64("clip.vision.embedding_length").context("embedding_length")? as usize;
        let n_head = meta_u64("clip.vision.attention.head_count").context("head_count")? as usize;
        let n_layer = meta_u64("clip.vision.block_count").context("block_count")? as usize;
        let eps = loaded
            .metadata
            .iter()
            .find(|(k, _)| k == "clip.vision.attention.layer_norm_epsilon")
            .and_then(|(_, v)| match v {
                orangu::gguf::GgufValue::F32(e) => Some(*e),
                _ => None,
            })
            .unwrap_or(1e-6);
        let vector = |name: &str| -> Result<Vec<f32>> { Ok(loaded.tensor(name)?.0) };
        let norm = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                vector(&format!("{name}.weight"))?,
                vector(&format!("{name}.bias"))?,
            ))
        };
        // The tower's weights are `BF16`, a float type the `int8` kernel
        // does not take: requantized at load to `Q6_K`, rows zero-padded to
        // its 256-wide blocks (`linear` pads the activations to match), as
        // the VAE's convolutions are. `ORANGU_IMAGE_VISION=f32` keeps them
        // as they are, for an A/B.
        let int8 = vision_int8();
        let rowi8 = std::sync::Mutex::new(std::collections::HashMap::new());
        let matrix = |name: &str| -> Result<QuantMatrix> {
            let w = loaded.matrix(&format!("{name}.weight"))?;
            if !int8 {
                return Ok(w);
            }
            let rows: Vec<f32> = (0..w.out_dim).flat_map(|o| w.row(o)).collect();
            let fast = row_matrix(&rows, w.in_dim, w.out_dim, VaePrecision::Int8);
            let q = conv_matrix(rows, w.in_dim, w.out_dim, VaePrecision::Int8);
            if let Some(fast) = fast {
                rowi8
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(q.raw_bytes().as_ptr() as usize, fast);
            }
            Ok(q)
        };
        let bias = |name: &str| vector(&format!("{name}.bias"));
        // `[16, 16, 3, width]` in ggml order: each output's inputs are
        // `(channel, ky, kx)`, row-major — the two frames' kernels summed.
        let (k0, _) = loaded.tensor("v.patch_embd.weight")?;
        let (k1, _) = loaded.tensor("v.patch_embd.weight.1")?;
        ensure!(
            k0.len() == width * 3 * PATCH * PATCH && k1.len() == k0.len(),
            "the patch embedding is not {width} x 3 x {PATCH} x {PATCH}"
        );
        let summed: Vec<f32> = k0.iter().zip(&k1).map(|(a, b)| a + b).collect();
        let patch = QuantMatrix::from_f32_rows(summed, 3 * PATCH * PATCH, width);
        let (pos, pos_dims) = loaded.tensor("v.position_embd.weight")?;
        let side = (pos_dims[1] as f64).sqrt() as usize;
        ensure!(
            side * side == pos_dims[1] as usize && pos_dims[0] as usize == width,
            "the position table is not a square of {width}-wide entries"
        );
        let merger = |p: &str| -> Result<Merger> {
            Ok(Merger {
                fc1: matrix(&format!("{p}fc1"))?,
                fc1_b: bias(&format!("{p}fc1"))?,
                fc2: matrix(&format!("{p}fc2"))?,
                fc2_b: bias(&format!("{p}fc2"))?,
            })
        };
        let mut blocks = Vec::with_capacity(n_layer);
        let mut deepstack = Vec::new();
        for i in 0..n_layer {
            let p = format!("v.blk.{i}");
            blocks.push(VisionBlock {
                ln1: norm(&format!("{p}.ln1"))?,
                qkv: matrix(&format!("{p}.attn_qkv"))?,
                qkv_b: bias(&format!("{p}.attn_qkv"))?,
                out: matrix(&format!("{p}.attn_out"))?,
                out_b: bias(&format!("{p}.attn_out"))?,
                ln2: norm(&format!("{p}.ln2"))?,
                up: matrix(&format!("{p}.ffn_up"))?,
                up_b: bias(&format!("{p}.ffn_up"))?,
                down: matrix(&format!("{p}.ffn_down"))?,
                down_b: bias(&format!("{p}.ffn_down"))?,
            });
            if loaded.has_tensor(&format!("v.deepstack.{i}.fc1.weight")) {
                deepstack.push(Deepstack {
                    after_block: i,
                    norm: norm(&format!("v.deepstack.{i}.norm"))?,
                    merger: merger(&format!("v.deepstack.{i}."))?,
                });
            }
        }
        let main = Merger {
            fc1: matrix("mm.0")?,
            fc1_b: bias("mm.0")?,
            fc2: matrix("mm.2")?,
            fc2_b: bias("mm.2")?,
        };
        let out_width = main.fc2.out_dim;
        ensure!(
            width.is_multiple_of(n_head) && (width / n_head).is_multiple_of(4),
            "the vision heads do not split {width}"
        );
        Ok(Self {
            backend,
            width,
            n_head,
            eps,
            patch,
            patch_b: vector("v.patch_embd.bias")?,
            pos,
            side,
            blocks,
            post_ln: norm("v.post_ln")?,
            merger: main,
            deepstack,
            out_width,
            rowi8: rowi8.into_inner().unwrap_or_else(|p| p.into_inner()),
        })
    }

    fn linear(&self, x: &[f32], n: usize, w: &QuantMatrix, b: &[f32]) -> Vec<f32> {
        // A requantized weight's rows are padded past the input's width;
        // the activations are padded with zeros to meet them.
        let width = x.len() / n;
        let padded;
        let x = if w.in_dim > width {
            let mut wide = vec![0f32; n * w.in_dim];
            wide.par_chunks_mut(w.in_dim)
                .zip(x.par_chunks(width))
                .for_each(|(dst, src)| dst[..width].copy_from_slice(src));
            padded = wide;
            &padded[..]
        } else {
            x
        };
        let mut y = match self.rowi8.get(&(w.raw_bytes().as_ptr() as usize)) {
            Some(rows) if self.backend.is_cpu() => crate::engine::vecdot::matmul_rowi8(x, n, rows),
            _ => self
                .backend
                .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
                .pop()
                .expect("one op in, one result out"),
        };
        tensor::add_bias_per_row(&mut y, b, n);
        y
    }

    /// Reads an RGB picture in `[-1, 1]` (the processor's `(v - 0.5) / 0.5`
    /// of `[0, 1]`) whose sides are multiples of [`UNIT`].
    pub fn encode(
        &self,
        rgb: &Feature,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<VisionOutput> {
        ensure!(rgb.channels == 3, "the vision tower reads RGB");
        ensure!(
            rgb.height.is_multiple_of(UNIT) && rgb.width.is_multiple_of(UNIT) && rgb.height > 0,
            "a picture for the vision tower is a multiple of {UNIT} pixels on each side"
        );
        let (rows, cols) = (rgb.height / PATCH, rgb.width / PATCH);
        let n = rows * cols;
        let width = self.width;
        // The patches in merge order: every 2x2 block's four together.
        let order: Vec<(usize, usize)> = (0..rows / MERGE)
            .flat_map(|y| {
                (0..cols / MERGE).flat_map(move |x| {
                    (0..MERGE).flat_map(move |dy| {
                        (0..MERGE).map(move |dx| (y * MERGE + dy, x * MERGE + dx))
                    })
                })
            })
            .collect();
        let cols_in = 3 * PATCH * PATCH;
        let mut patches = vec![0.0f32; n * cols_in];
        patches
            .par_chunks_mut(cols_in)
            .zip(&order)
            .for_each(|(out, &(r, c))| {
                for ch in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            let px = (r * PATCH + ky) * rgb.width + c * PATCH + kx;
                            out[(ch * PATCH + ky) * PATCH + kx] = rgb.data[px * 3 + ch];
                        }
                    }
                }
            });
        let mut x = self.linear(&patches, n, &self.patch, &self.patch_b);
        drop(patches);
        // The learned table, bilinear with aligned corners onto the grid.
        let axis = |len: usize| -> Vec<(usize, usize, f32)> {
            (0..len)
                .map(|i| {
                    let f = if len > 1 {
                        i as f64 * (self.side - 1) as f64 / (len - 1) as f64
                    } else {
                        0.0
                    };
                    let lo = f.floor() as usize;
                    let hi = (lo + 1).min(self.side - 1);
                    (lo, hi, (f - lo as f64) as f32)
                })
                .collect()
        };
        let (ry, cx) = (axis(rows), axis(cols));
        x.par_chunks_mut(width)
            .zip(&order)
            .for_each(|(row, &(r, c))| {
                let (r0, r1, wr) = ry[r];
                let (c0, c1, wc) = cx[c];
                let at = |a: usize, b: usize| &self.pos[(a * self.side + b) * width..][..width];
                let corners = [
                    (at(r0, c0), (1.0 - wr) * (1.0 - wc)),
                    (at(r0, c1), (1.0 - wr) * wc),
                    (at(r1, c0), wr * (1.0 - wc)),
                    (at(r1, c1), wr * wc),
                ];
                for (v, i) in row.iter_mut().zip(0..) {
                    *v += corners.iter().map(|(p, w)| p[i] * w).sum::<f32>();
                }
            });
        // 2-D rotary angles, half the head for rows and half for columns.
        let head_dim = width / self.n_head;
        let half = head_dim / 2;
        let quarter = half / 2;
        let freqs: Vec<f64> = (0..quarter)
            .map(|k| 10000f64.powf(-(k as f64) / quarter as f64))
            .collect();
        let angles: Vec<(f32, f32)> = order
            .iter()
            .flat_map(|&(r, c)| {
                let freqs = &freqs;
                (0..half).map(move |j| {
                    let a = if j < quarter {
                        r as f64 * freqs[j]
                    } else {
                        c as f64 * freqs[j - quarter]
                    };
                    (a.cos() as f32, a.sin() as f32)
                })
            })
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut deepstack = Vec::with_capacity(self.deepstack.len());
        let mut normed = Vec::new();
        for (bi, block) in self.blocks.iter().enumerate() {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            layer_norm_affine(&mut normed, &x, &block.ln1.0, &block.ln1.1, width, self.eps);
            let qkv = self.linear(&normed, n, &block.qkv, &block.qkv_b);
            let (mut q, mut k, mut v) = (
                Vec::with_capacity(n * width),
                Vec::with_capacity(n * width),
                Vec::with_capacity(n * width),
            );
            for row in qkv.chunks(3 * width) {
                q.extend_from_slice(&row[..width]);
                k.extend_from_slice(&row[width..2 * width]);
                v.extend_from_slice(&row[2 * width..]);
            }
            drop(qkv);
            rope_half(&mut q, width, head_dim, &angles);
            rope_half(&mut k, width, head_dim, &angles);
            let attn = step_attention(&q, n, &k, &v, n, self.n_head, head_dim, scale, None);
            drop((q, k, v));
            let out = self.linear(&attn, n, &block.out, &block.out_b);
            tensor::add_inplace(&mut x, &out);
            layer_norm_affine(&mut normed, &x, &block.ln2.0, &block.ln2.1, width, self.eps);
            let mut h = self.linear(&normed, n, &block.up, &block.up_b);
            tensor::gelu_inplace(&mut h);
            let down = self.linear(&h, n, &block.down, &block.down_b);
            tensor::add_inplace(&mut x, &down);
            if let Some(ds) = self.deepstack.iter().find(|d| d.after_block == bi) {
                // Merged first, then normed over the merged width.
                let merged_width = width * MERGE * MERGE;
                let mut m = Vec::new();
                layer_norm_affine(&mut m, &x, &ds.norm.0, &ds.norm.1, merged_width, self.eps);
                deepstack.push(self.merge(&ds.merger, &m, n / (MERGE * MERGE)));
            }
        }
        layer_norm_affine(
            &mut normed,
            &x,
            &self.post_ln.0,
            &self.post_ln.1,
            width,
            self.eps,
        );
        let embeds = self.merge(&self.merger, &normed, n / (MERGE * MERGE));
        Ok(VisionOutput {
            grid: (rows / MERGE, cols / MERGE),
            embeds,
            deepstack,
        })
    }

    /// `fc2(GELU(fc1(x)))` over `n` merged rows (the input's consecutive
    /// four patches are one row already).
    fn merge(&self, m: &Merger, x: &[f32], n: usize) -> Vec<f32> {
        let mut h = self.linear(x, n, &m.fc1, &m.fc1_b);
        gelu_erf_inplace(&mut h);
        self.linear(&h, n, &m.fc2, &m.fc2_b)
    }
}

/// LayerNorm with weight and bias over each `dim`-wide row.
fn layer_norm_affine(dst: &mut Vec<f32>, src: &[f32], w: &[f32], b: &[f32], dim: usize, eps: f32) {
    dst.resize(src.len(), 0.0);
    dst.par_chunks_mut(dim)
        .zip(src.par_chunks(dim))
        .for_each(|(out, row)| {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (((o, v), w), b) in out.iter_mut().zip(row).zip(w).zip(b) {
                *o = (v - mean) * inv * w + b;
            }
        });
}

/// Whether the vision tower's weights are requantized to `Q6_K` for the
/// `int8` kernel — the default; `ORANGU_IMAGE_VISION=f32` keeps them `BF16`
/// (`doc/PERF-IMAGE.md`, task 7).
fn vision_int8() -> bool {
    !std::env::var("ORANGU_IMAGE_VISION").is_ok_and(|v| v.trim().eq_ignore_ascii_case("f32"))
}

/// Rotates each head's `(j, j + head_dim/2)` pairs — the `rotate_half`
/// form — by that row's angles.
fn rope_half(x: &mut [f32], dim: usize, head_dim: usize, angles: &[(f32, f32)]) {
    let half = head_dim / 2;
    x.par_chunks_mut(dim)
        .zip(angles.par_chunks(half))
        .for_each(|(row, angles)| {
            for head in row.chunks_mut(head_dim) {
                let (lo, hi) = head.split_at_mut(half);
                for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi.iter_mut()).zip(angles) {
                    let (x0, x1) = (*a, *b);
                    *a = x0 * cos - x1 * sin;
                    *b = x0 * sin + x1 * cos;
                }
            }
        });
}

/// PyTorch's exact `nn.GELU()`, `x · Φ(x)`, with `erf` from Abramowitz &
/// Stegun 7.1.26 (absolute error under 1.5e-7).
fn gelu_erf_inplace(x: &mut [f32]) {
    fn erf(x: f64) -> f64 {
        let s = x.signum();
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        s * y
    }
    x.par_chunks_mut(4096).for_each(|chunk| {
        for v in chunk.iter_mut() {
            let z = *v as f64;
            *v = (0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2))) as f32;
        }
    });
}

struct TextLayer {
    attn_norm: Vec<f32>,
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    wo: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
}

/// The `qwen3vl` text model, for a prompt with pictures in it — see the
/// module doc.
pub struct TextModel {
    backend: Arc<dyn Backend>,
    embd: QuantMatrix,
    layers: Vec<TextLayer>,
    dim: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    eps: f32,
    theta: f64,
    /// Rotary frequencies per axis, `[t, h, w]` — `rope.dimension_sections`.
    sections: [usize; 3],
}

/// A picture in a prompt: its tokens start at `start`.
pub struct PromptPicture<'a> {
    pub start: usize,
    pub vision: &'a VisionOutput,
}

impl TextModel {
    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let c = &loaded.config;
        ensure!(
            c.architecture == "qwen3vl",
            "the picture-reading text model is a qwen3vl, not a {}",
            c.architecture
        );
        let sections = loaded
            .metadata
            .iter()
            .find(|(k, _)| k == "qwen3vl.rope.dimension_sections")
            .and_then(|(_, v)| match v {
                orangu::gguf::GgufValue::Array(items) => Some(
                    items
                        .iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .filter(|s| s.len() >= 3)
            .map(|s| [s[0], s[1], s[2]])
            .unwrap_or([24, 20, 20]);
        ensure!(
            sections.iter().sum::<usize>() * 2 == c.head_dim,
            "the rotary sections {sections:?} do not cover a {}-wide head",
            c.head_dim
        );
        let matrix = |name: &str| loaded.matrix(name).with_context(|| name.to_string());
        let vector = |name: &str| -> Result<Vec<f32>> { Ok(loaded.tensor(name)?.0) };
        let layers = (0..c.n_layer)
            .map(|i| {
                let p = format!("blk.{i}");
                Ok(TextLayer {
                    attn_norm: vector(&format!("{p}.attn_norm.weight"))?,
                    wq: matrix(&format!("{p}.attn_q.weight"))?,
                    wk: matrix(&format!("{p}.attn_k.weight"))?,
                    wv: matrix(&format!("{p}.attn_v.weight"))?,
                    wo: matrix(&format!("{p}.attn_output.weight"))?,
                    q_norm: vector(&format!("{p}.attn_q_norm.weight"))?,
                    k_norm: vector(&format!("{p}.attn_k_norm.weight"))?,
                    ffn_norm: vector(&format!("{p}.ffn_norm.weight"))?,
                    gate: matrix(&format!("{p}.ffn_gate.weight"))?,
                    up: matrix(&format!("{p}.ffn_up.weight"))?,
                    down: matrix(&format!("{p}.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backend,
            embd: matrix("token_embd.weight")?,
            layers,
            dim: c.n_embd,
            n_head: c.n_head,
            n_head_kv: c.n_head_kv,
            head_dim: c.head_dim,
            eps: c.rms_eps,
            theta: c.rope_freq_base as f64,
            sections,
        })
    }

    /// Every token's last-layer hidden state, **before** the final norm —
    /// `[tokens.len(), dim]` — with each picture's tokens (the
    /// `<|image_pad|>` run the prompt was expanded to) replaced by its
    /// vision embeddings.
    pub fn forward_pre_norm(
        &self,
        tokens: &[u32],
        pictures: &[PromptPicture<'_>],
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<f32>> {
        let (n, dim) = (tokens.len(), self.dim);
        ensure!(n > 0, "an empty prompt");
        let mut x = vec![0.0f32; n * dim];
        x.par_chunks_mut(dim)
            .zip(tokens.par_iter())
            .for_each(|(row, &t)| row.copy_from_slice(&self.embd.row(t as usize)));
        // Positions: text advances all three axes together; a picture sits
        // at one `t` with its rows and columns spread from there, and the
        // text after it resumes past its larger side.
        let mut positions = vec![[0usize; 3]; n];
        let mut picture_rows = vec![None; n];
        let mut next = 0usize;
        let mut i = 0usize;
        let mut sorted: Vec<&PromptPicture<'_>> = pictures.iter().collect();
        sorted.sort_by_key(|p| p.start);
        let mut pic = 0usize;
        while i < n {
            if let Some(p) = sorted.get(pic).filter(|p| p.start == i) {
                let (rows, cols) = p.vision.grid;
                let count = rows * cols;
                ensure!(
                    i + count <= n && p.vision.embeds.len() == count * dim,
                    "a picture's tokens do not fit the prompt"
                );
                for r in 0..rows {
                    for c in 0..cols {
                        let at = i + r * cols + c;
                        positions[at] = [next, next + r, next + c];
                        picture_rows[at] = Some((pic, r * cols + c));
                        x[at * dim..(at + 1) * dim]
                            .copy_from_slice(&p.vision.embeds[(r * cols + c) * dim..][..dim]);
                    }
                }
                next += rows.max(cols);
                i += count;
                pic += 1;
                continue;
            }
            positions[i] = [next; 3];
            next += 1;
            i += 1;
        }
        ensure!(pic == sorted.len(), "a picture starts outside the prompt");
        let angles = self.rope_angles(&positions);
        let (n_head, n_kv, hd) = (self.n_head, self.n_head_kv, self.head_dim);
        let limits = causal_limits(n);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut normed = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            tensor::rmsnorm_into(&mut normed, &x, &layer.attn_norm, n, dim, self.eps);
            let mut qkv = self.backend.matmul_batch(&[
                MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: &layer.wq,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: &layer.wk,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: &layer.wv,
                },
            ]);
            let v = qkv.pop().expect("three results");
            let mut k = qkv.pop().expect("three results");
            let mut q = qkv.pop().expect("three results");
            tensor::rmsnorm_inplace(&mut q, &layer.q_norm, n * n_head, hd, self.eps);
            tensor::rmsnorm_inplace(&mut k, &layer.k_norm, n * n_kv, hd, self.eps);
            rope_half(&mut q, n_head * hd, hd, &angles);
            rope_half(&mut k, n_kv * hd, hd, &angles);
            // Grouped-query attention: each key/value head serves
            // `n_head / n_kv` query heads.
            let group = n_head / n_kv;
            let expand = |src: &[f32]| -> Vec<f32> {
                let mut out = vec![0.0f32; n * n_head * hd];
                out.par_chunks_mut(n_head * hd)
                    .zip(src.par_chunks(n_kv * hd))
                    .for_each(|(dst, row)| {
                        for h in 0..n_head {
                            dst[h * hd..(h + 1) * hd]
                                .copy_from_slice(&row[(h / group) * hd..(h / group + 1) * hd]);
                        }
                    });
                out
            };
            let (k, v) = (expand(&k), expand(&v));
            let attn = step_attention(&q, n, &k, &v, n, n_head, hd, scale, Some(&limits));
            drop((q, k, v));
            let out = self.linear(&attn, n, &layer.wo);
            tensor::add_inplace(&mut x, &out);
            tensor::rmsnorm_into(&mut normed, &x, &layer.ffn_norm, n, dim, self.eps);
            let mut gu = self.backend.matmul_batch(&[
                MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: &layer.gate,
                },
                MatmulOp {
                    x: &normed,
                    n_tokens: n,
                    w: &layer.up,
                },
            ]);
            let up = gu.pop().expect("two results");
            let mut gate = gu.pop().expect("two results");
            gate.par_iter_mut()
                .zip(up.par_iter())
                .for_each(|(g, u)| *g = tensor::silu(*g) * u);
            let down = self.linear(&gate, n, &layer.down);
            tensor::add_inplace(&mut x, &down);
            // DeepStack: the vision tower's intermediate features, added to
            // the picture's rows after the first layers.
            for (at, row) in x.chunks_mut(dim).enumerate() {
                if let Some((pic, r)) = picture_rows[at]
                    && let Some(ds) = sorted[pic].vision.deepstack.get(li)
                {
                    for (v, d) in row.iter_mut().zip(&ds[r * dim..(r + 1) * dim]) {
                        *v += d;
                    }
                }
            }
        }
        Ok(x)
    }

    fn linear(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }

    /// Interleaved M-RoPE: frequency `j` of the half head turns with axis
    /// `h` when `j % 3 == 1`, `w` when `j % 3 == 2` (within `3 ×` their
    /// section), and `t` otherwise — Qwen3-VL's `apply_interleaved_mrope`.
    fn rope_angles(&self, positions: &[[usize; 3]]) -> Vec<(f32, f32)> {
        let half = self.head_dim / 2;
        let axis_of: Vec<usize> = (0..half)
            .map(|j| {
                if j % 3 == 1 && j < 3 * self.sections[1] {
                    1
                } else if j % 3 == 2 && j < 3 * self.sections[2] {
                    2
                } else {
                    0
                }
            })
            .collect();
        let freqs: Vec<f64> = (0..half)
            .map(|j| self.theta.powf(-2.0 * j as f64 / self.head_dim as f64))
            .collect();
        positions
            .iter()
            .flat_map(|pos| {
                let (axis_of, freqs) = (&axis_of, &freqs);
                (0..half).map(move |j| {
                    let a = pos[axis_of[j]] as f64 * freqs[j];
                    (a.cos() as f32, a.sin() as f32)
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On text alone the reference forward is the served one: the same
    /// hidden states as `arch::llama`'s `forward_hidden_states_pre_norm`.
    /// Needs a real Qwen3-VL GGUF: `ORANGU_TEST_QWEN3VL=/path/to.gguf cargo
    /// test … text_alone_matches -- --ignored`.
    #[test]
    #[ignore]
    fn text_alone_matches_the_served_forward() {
        use crate::engine::arch::ModelForward;
        let path = std::path::PathBuf::from(std::env::var("ORANGU_TEST_QWEN3VL").unwrap());
        let loaded = LoadedModel::open(&path).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let served =
            crate::engine::arch::llama::LlamaModel::load_with_backend(&loaded, backend.clone())
                .unwrap();
        let reference = TextModel::load(&loaded, backend).unwrap();
        // "A red apple on a wooden table, photograph".
        let tokens: Vec<u32> = vec![32, 2518, 23268, 389, 264, 22360, 1965, 11, 10300];
        let dim = reference.dim;
        let norm = loaded.tensor("output_norm.weight").unwrap().0;
        let served_out = served.forward_hidden_states(&tokens).unwrap();
        let mut mine = reference.forward_pre_norm(&tokens, &[], None).unwrap();
        tensor::rmsnorm_inplace(&mut mine, &norm, tokens.len(), dim, reference.eps);
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            dot / (a.iter().map(|v| v * v).sum::<f32>().sqrt()
                * b.iter().map(|v| v * v).sum::<f32>().sqrt())
        };
        // Per-token rows from `llama-embedding --pooling none
        // --embd-normalize -1`, when given.
        let llama: Option<Vec<Vec<f32>>> =
            std::env::var("ORANGU_TEST_QWEN3VL_LLAMA_CPP")
                .ok()
                .map(|p| {
                    std::fs::read_to_string(p)
                        .unwrap()
                        .lines()
                        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
                        .collect()
                });
        for t in 0..tokens.len() {
            let (m, s) = (
                &mine[t * dim..(t + 1) * dim],
                &served_out[t * dim..(t + 1) * dim],
            );
            match &llama {
                Some(l) => eprintln!(
                    "token {t}: mine~served {:.6}, mine~llama.cpp {:.6}, served~llama.cpp {:.6}",
                    cosine(m, s),
                    cosine(m, &l[t]),
                    cosine(s, &l[t])
                ),
                None => eprintln!("token {t}: mine~served {:.6}", cosine(m, s)),
            }
            if let Some(l) = &llama {
                assert!(cosine(m, &l[t]) > 0.999, "token {t} against llama.cpp");
            }
        }
    }

    /// The vision tower read end to end: Qwen3-VL describes a picture —
    /// greedy decoding over this module's hidden states, the whole prompt
    /// re-run per token (no cache; a few seconds a token). Set
    /// `ORANGU_TEST_QWEN3VL`, `ORANGU_TEST_QWEN3VL_MMPROJ` and
    /// `ORANGU_TEST_QWEN3VL_PICTURE` (a PNG or JPEG).
    #[test]
    #[ignore]
    fn the_model_describes_a_picture() {
        let path = std::path::PathBuf::from(std::env::var("ORANGU_TEST_QWEN3VL").unwrap());
        let mmproj = std::path::PathBuf::from(std::env::var("ORANGU_TEST_QWEN3VL_MMPROJ").unwrap());
        let picture = std::fs::read(std::env::var("ORANGU_TEST_QWEN3VL_PICTURE").unwrap()).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let loaded = LoadedModel::open(&path).unwrap();
        let text = TextModel::load(&loaded, backend.clone()).unwrap();
        let tower = VisionTower::load(&mmproj, backend.clone()).unwrap();
        let gguf = orangu::gguf::GgufFile::open(&path).unwrap();
        let tokenizer = crate::engine::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
        let rgb = super::super::codec::decode_to_feature(&picture, 256, 256).unwrap();
        let seen = tower.encode(&rgb, None).unwrap();
        let (rows, cols) = seen.grid;
        let pad = tokenizer.encode("<|image_pad|>", false)[0];
        let head = tokenizer.encode("<|im_start|>user\n<|vision_start|>", false);
        let tail = tokenizer.encode(
            "<|vision_end|>What is in this picture? Answer in one short sentence.<|im_end|>\n\
             <|im_start|>assistant\n",
            false,
        );
        let mut tokens = head.clone();
        tokens.extend(std::iter::repeat_n(pad, rows * cols));
        tokens.extend(tail);
        let norm = loaded.tensor("output_norm.weight").unwrap().0;
        let output = loaded.matrix("output.weight").unwrap();
        let mut answer = Vec::new();
        for _ in 0..16 {
            let hidden = text
                .forward_pre_norm(
                    &tokens,
                    &[PromptPicture {
                        start: head.len(),
                        vision: &seen,
                    }],
                    None,
                )
                .unwrap();
            let mut last = hidden[(tokens.len() - 1) * text.dim..].to_vec();
            tensor::rmsnorm_inplace(&mut last, &norm, 1, text.dim, text.eps);
            let logits = backend.matmul(&last, 1, &output);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            if next == 151645 {
                break;
            }
            tokens.push(next);
            answer.push(next);
        }
        let said = tokenizer.decode(&answer);
        eprintln!("the model says: {said}");
        let expect = std::env::var("ORANGU_TEST_QWEN3VL_EXPECT").unwrap_or_else(|_| "apple".into());
        assert!(said.to_lowercase().contains(&expect), "{said}");
    }

    #[test]
    fn exact_gelu_matches_known_values() {
        let mut x = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0];
        gelu_erf_inplace(&mut x);
        let want = [
            -0.004_049_7,
            -0.158_655_3,
            0.0,
            0.345_731_4,
            0.841_344_7,
            1.954_5,
        ];
        for (g, w) in x.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }

    #[test]
    fn rope_half_rotates_the_split_pairs() {
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
        let a = 0.3f32;
        rope_half(&mut x, 4, 4, &[(a.cos(), a.sin()), (1.0, 0.0)]);
        assert!((x[0] - a.cos()).abs() < 1e-6 && (x[2] - a.sin()).abs() < 1e-6);
        assert_eq!((x[1], x[3]), (0.0, 0.0));
    }

    /// With every axis at the same position, interleaved M-RoPE is plain
    /// RoPE — the reduction that makes text alone agree with the served
    /// path.
    #[test]
    fn equal_axes_reduce_interleaved_mrope_to_plain_rope() {
        let model = TextModel {
            backend: Arc::new(crate::engine::backend::CpuBackend),
            embd: QuantMatrix::from_f32_rows(vec![0.0; 4], 4, 1),
            layers: Vec::new(),
            dim: 128,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 128,
            eps: 1e-6,
            theta: 5e6,
            sections: [24, 20, 20],
        };
        let mixed = model.rope_angles(&[[7, 7, 7]]);
        for (j, &(c, s)) in mixed.iter().enumerate() {
            let a = 7.0 * 5e6f64.powf(-2.0 * j as f64 / 128.0);
            assert!((c - a.cos() as f32).abs() < 1e-6 && (s - a.sin() as f32).abs() < 1e-6);
        }
        // Frequency 1 follows the row, 2 the column, 0 and the tail `t`.
        let spread = model.rope_angles(&[[1, 2, 3]]);
        let f = |j: usize| 5e6f64.powf(-2.0 * j as f64 / 128.0);
        assert!((spread[1].1 - (2.0 * f(1)).sin() as f32).abs() < 1e-6);
        assert!((spread[2].1 - (3.0 * f(2)).sin() as f32).abs() < 1e-6);
        assert!((spread[0].1 - (1.0 * f(0)).sin() as f32).abs() < 1e-6);
        assert!((spread[61].1 - (1.0 * f(61)).sin() as f32).abs() < 1e-6);
    }
}

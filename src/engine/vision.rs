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

//! A chat model reading a picture. Three parts:
//!
//! - [`Encoder`]: gemma-4's vision tower, the `mmproj-*.gguf` beside the
//!   model (`clip.projector_type = gemma4v`). 16-pixel patches, learned
//!   per-axis position tables, 2-D rotary attention (NEOX pairing, the
//!   first half of each head by column and the second by row) over
//!   RMS-normed queries, keys and values, sandwich-normed blocks with a
//!   gated FFN, then a 3x3 average pool, a standardisation and the
//!   projection to the language model's width.
//! - [`fit_size`]: the picture's size on the way in — its aspect ratio
//!   kept, each side a multiple of 48 pixels, the pooled token count held
//!   inside the encoder's budget.
//! - [`ImageStore`]: where the encoded rows wait for the forward pass. A
//!   picture enters the prompt as a run of **virtual token ids** above the
//!   vocabulary, one per row, chosen from a hash of the picture; the model
//!   reads a virtual id's row from here in place of a vocabulary row. So
//!   the same picture in the next turn is the same ids, and the prompt
//!   cache reuses it like any other repeated text.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::image::transformer::attention;
use super::image::vae::Feature;

/// The projector type this module reads.
pub const PROJECTOR_TYPE: &str = "gemma4v";
/// Patches per pooled token side.
const POOL: usize = 3;
/// Rotary base of the tower's 2-D positions.
const ROPE_THETA: f64 = 100.0;
/// The fewest tokens a picture becomes.
pub const MIN_TOKENS: usize = 70;
/// The most tokens a picture becomes unless the server says otherwise.
pub const DEFAULT_MAX_TOKENS: usize = 280;
/// The most tokens a picture may become at all.
pub const MAX_TOKENS: usize = 1120;

/// A weight with the clamps gemma-4's `ClippableLinear` puts around it —
/// the input clamped before the product and the output after it, each
/// only where the file carries the bound.
struct Linear {
    w: QuantMatrix,
    clamp: Option<[f32; 4]>,
}

struct Block {
    ln1: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out: Linear,
    attn_post_norm: Vec<f32>,
    ln2: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
    ffn_post_norm: Vec<f32>,
}

pub struct Encoder {
    backend: Arc<dyn Backend>,
    width: usize,
    n_head: usize,
    patch_size: usize,
    eps: f32,
    /// `[3 * patch * patch → width]`, inputs `(channel, ky, kx)`.
    patch: QuantMatrix,
    /// `[positions][width]`, by column and by row.
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    positions: usize,
    blocks: Vec<Block>,
    std_bias: Option<Vec<f32>>,
    std_scale: Option<Vec<f32>>,
    proj: Linear,
    /// The language model's width the projection writes.
    pub out_width: usize,
    /// The most tokens a picture becomes.
    pub max_tokens: usize,
}

impl Encoder {
    /// The language-model width the vision tower at `path` writes, when it
    /// is one this module reads.
    pub fn projector_width(path: &Path) -> Option<usize> {
        let gguf = crate::gguf::GgufFile::open_summary(path).ok()?;
        let key = |name: &str| {
            gguf.metadata
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v)
        };
        matches!(
            key("clip.vision.projector_type"),
            Some(crate::gguf::GgufValue::String(s)) if s == PROJECTOR_TYPE
        )
        .then(|| key("clip.vision.projection_dim").and_then(|v| v.as_u64()))
        .flatten()
        .map(|d| d as usize)
    }

    pub fn load(path: &Path, backend: Arc<dyn Backend>, max_tokens: usize) -> Result<Self> {
        let loaded = LoadedModel::open_projector(path)
            .with_context(|| format!("opening the vision projector {}", path.display()))?;
        let meta = |key: &str| {
            loaded
                .metadata
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
        };
        let meta_u64 = |key: &str| meta(key).and_then(|v| v.as_u64());
        let projector = meta("clip.vision.projector_type").and_then(|v| match v {
            crate::gguf::GgufValue::String(s) => Some(s.clone()),
            _ => None,
        });
        ensure!(
            projector.as_deref() == Some(PROJECTOR_TYPE),
            "{} is not a gemma-4 vision projector (clip.vision.projector_type = {PROJECTOR_TYPE})",
            path.display()
        );
        let width = meta_u64("clip.vision.embedding_length").context("embedding_length")? as usize;
        let n_head = meta_u64("clip.vision.attention.head_count").context("head_count")? as usize;
        let n_layer = meta_u64("clip.vision.block_count").context("block_count")? as usize;
        let patch_size = meta_u64("clip.vision.patch_size").context("patch_size")? as usize;
        let eps = match meta("clip.vision.attention.layer_norm_epsilon") {
            Some(crate::gguf::GgufValue::F32(e)) => *e,
            _ => 1e-6,
        };
        ensure!(
            width.is_multiple_of(n_head) && (width / n_head).is_multiple_of(4),
            "the vision heads do not split {width}"
        );
        let vector = |name: &str| -> Result<Vec<f32>> { Ok(loaded.tensor(name)?.0) };
        let scalar = |name: &str| -> Option<f32> {
            loaded
                .has_tensor(name)
                .then(|| loaded.tensor(name).ok())
                .flatten()
                .and_then(|(v, _)| v.first().copied())
        };
        let linear = |name: &str| -> Result<Linear> {
            let w = loaded.matrix(&format!("{name}.weight"))?;
            let bounds = [
                scalar(&format!("{name}.input_min")),
                scalar(&format!("{name}.input_max")),
                scalar(&format!("{name}.output_min")),
                scalar(&format!("{name}.output_max")),
            ];
            let clamp = bounds.iter().any(Option::is_some).then(|| {
                [
                    bounds[0].unwrap_or(f32::NEG_INFINITY),
                    bounds[1].unwrap_or(f32::INFINITY),
                    bounds[2].unwrap_or(f32::NEG_INFINITY),
                    bounds[3].unwrap_or(f32::INFINITY),
                ]
            });
            Ok(Linear {
                w: requantize(&w),
                clamp,
            })
        };
        let (patch, dims) = loaded.tensor("v.patch_embd.weight")?;
        let cols = 3 * patch_size * patch_size;
        ensure!(
            patch.len() == width * cols && dims.first() == Some(&(patch_size as u64)),
            "the patch embedding is not {width} x 3 x {patch_size} x {patch_size}"
        );
        let patch = requantize(&QuantMatrix::from_f32_rows(patch, cols, width));
        let (pos, pos_dims) = loaded.tensor("v.position_embd.weight")?;
        ensure!(
            pos_dims.len() == 3 && pos_dims[0] as usize == width && pos_dims[2] == 2,
            "the position table is not two tables of {width}-wide entries"
        );
        let positions = pos_dims[1] as usize;
        let pos_y = pos[positions * width..].to_vec();
        let mut pos_x = pos;
        pos_x.truncate(positions * width);
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("v.blk.{i}");
            blocks.push(Block {
                ln1: vector(&format!("{p}.ln1.weight"))?,
                q: linear(&format!("{p}.attn_q"))?,
                k: linear(&format!("{p}.attn_k"))?,
                v: linear(&format!("{p}.attn_v"))?,
                q_norm: vector(&format!("{p}.attn_q_norm.weight"))?,
                k_norm: vector(&format!("{p}.attn_k_norm.weight"))?,
                out: linear(&format!("{p}.attn_out"))?,
                attn_post_norm: vector(&format!("{p}.attn_post_norm.weight"))?,
                ln2: vector(&format!("{p}.ln2.weight"))?,
                gate: linear(&format!("{p}.ffn_gate"))?,
                up: linear(&format!("{p}.ffn_up"))?,
                down: linear(&format!("{p}.ffn_down"))?,
                ffn_post_norm: vector(&format!("{p}.ffn_post_norm.weight"))?,
            });
        }
        let optional = |name: &str| -> Result<Option<Vec<f32>>> {
            loaded.has_tensor(name).then(|| vector(name)).transpose()
        };
        let proj = linear("mm.input_projection")?;
        let out_width = proj.w.out_dim;
        Ok(Self {
            backend,
            width,
            n_head,
            patch_size,
            eps,
            patch,
            pos_x,
            pos_y,
            positions,
            blocks,
            std_bias: optional("v.std_bias")?,
            std_scale: optional("v.std_scale")?,
            proj,
            out_width,
            max_tokens: max_tokens.clamp(MIN_TOKENS, MAX_TOKENS),
        })
    }

    /// Pixels a picture's side is a multiple of.
    pub fn unit(&self) -> usize {
        self.patch_size * POOL
    }

    /// Decodes `bytes` (PNG, JPEG, GIF, WebP or SVG) at the size
    /// [`fit_size`] gives it and encodes it: `[tokens, out_width]`.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let (w, h) = super::image::codec::dimensions(bytes)?;
        let (w, h) = fit_size(w, h, self.unit(), MIN_TOKENS, self.max_tokens);
        let rgb = super::image::codec::decode_to_feature(bytes, w, h)?;
        self.encode(&rgb)
    }

    /// Reads an RGB picture in `[-1, 1]` whose sides are multiples of
    /// [`Self::unit`]: one row of the language model's width per pooled
    /// 3x3 block of patches, row-major.
    pub fn encode(&self, rgb: &Feature) -> Result<Vec<f32>> {
        let unit = self.unit();
        ensure!(rgb.channels == 3, "the vision tower reads RGB");
        ensure!(
            rgb.height.is_multiple_of(unit) && rgb.width.is_multiple_of(unit) && rgb.height > 0,
            "a picture for the vision tower is a multiple of {unit} pixels on each side"
        );
        let p = self.patch_size;
        let (rows, cols) = (rgb.height / p, rgb.width / p);
        ensure!(
            rows <= self.positions && cols <= self.positions,
            "a picture of {rows} x {cols} patches is past the position table"
        );
        let n = rows * cols;
        let width = self.width;
        let cols_in = 3 * p * p;
        let mut patches = vec![0.0f32; n * cols_in];
        patches
            .par_chunks_mut(cols_in)
            .enumerate()
            .for_each(|(i, out)| {
                let (r, c) = (i / cols, i % cols);
                for ch in 0..3 {
                    for ky in 0..p {
                        for kx in 0..p {
                            let px = (r * p + ky) * rgb.width + c * p + kx;
                            out[(ch * p + ky) * p + kx] = rgb.data[px * 3 + ch];
                        }
                    }
                }
            });
        let mut x = self.matmul(&patches, n, &self.patch);
        drop(patches);
        x.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
            let (r, c) = (i / cols, i % cols);
            let px = &self.pos_x[c * width..][..width];
            let py = &self.pos_y[r * width..][..width];
            for ((v, a), b) in row.iter_mut().zip(px).zip(py) {
                *v += a + b;
            }
        });
        let head_dim = width / self.n_head;
        let angles = rotary_angles(rows, cols, head_dim);
        let mut normed = Vec::with_capacity(x.len());
        for block in &self.blocks {
            normed.clear();
            normed.extend_from_slice(&x);
            tensor::rmsnorm_inplace(&mut normed, &block.ln1, n, width, self.eps);
            let mut q = self.linear(&normed, n, &block.q);
            let mut k = self.linear(&normed, n, &block.k);
            let mut v = self.linear(&normed, n, &block.v);
            let heads = n * self.n_head;
            tensor::rmsnorm_inplace(&mut q, &block.q_norm, heads, head_dim, self.eps);
            tensor::rmsnorm_inplace(&mut k, &block.k_norm, heads, head_dim, self.eps);
            rms_unweighted(&mut v, head_dim, self.eps);
            rope_2d(&mut q, width, head_dim, &angles);
            rope_2d(&mut k, width, head_dim, &angles);
            let attn = attention(&q, n, &k, &v, n, self.n_head, head_dim, 1.0, None);
            drop((q, k, v));
            let mut out = self.linear(&attn, n, &block.out);
            tensor::rmsnorm_inplace(&mut out, &block.attn_post_norm, n, width, self.eps);
            tensor::add_inplace(&mut x, &out);
            normed.clear();
            normed.extend_from_slice(&x);
            tensor::rmsnorm_inplace(&mut normed, &block.ln2, n, width, self.eps);
            let mut gate = self.linear(&normed, n, &block.gate);
            let up = self.linear(&normed, n, &block.up);
            gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, u)| {
                *g = gelu_quick(*g) * u;
            });
            drop(up);
            let mut down = self.linear(&gate, n, &block.down);
            tensor::rmsnorm_inplace(&mut down, &block.ffn_post_norm, n, width, self.eps);
            tensor::add_inplace(&mut x, &down);
        }
        // The 3x3 average pool, scaled by the tower's width.
        let (out_rows, out_cols) = (rows / POOL, cols / POOL);
        let tokens = out_rows * out_cols;
        let scale = (width as f32).sqrt() / (POOL * POOL) as f32;
        let mut pooled = vec![0f32; tokens * width];
        pooled
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(t, dst)| {
                let (oy, ox) = (t / out_cols, t % out_cols);
                for dy in 0..POOL {
                    for dx in 0..POOL {
                        let src = ((oy * POOL + dy) * cols + ox * POOL + dx) * width;
                        for (d, s) in dst.iter_mut().zip(&x[src..src + width]) {
                            *d += s;
                        }
                    }
                }
                for d in dst.iter_mut() {
                    *d *= scale;
                }
            });
        drop(x);
        if let (Some(bias), Some(std)) = (&self.std_bias, &self.std_scale) {
            pooled.par_chunks_mut(width).for_each(|row| {
                for ((v, b), s) in row.iter_mut().zip(bias).zip(std) {
                    *v = (*v - b) * s;
                }
            });
        }
        rms_unweighted(&mut pooled, width, self.eps);
        Ok(self.linear(&pooled, tokens, &self.proj))
    }

    fn linear(&self, x: &[f32], n: usize, l: &Linear) -> Vec<f32> {
        match l.clamp {
            None => self.matmul(x, n, &l.w),
            Some([in_min, in_max, out_min, out_max]) => {
                let clamped: Vec<f32> = x.par_iter().map(|v| v.clamp(in_min, in_max)).collect();
                let mut y = self.matmul(&clamped, n, &l.w);
                y.par_iter_mut()
                    .for_each(|v| *v = v.clamp(out_min, out_max));
                y
            }
        }
    }

    fn matmul(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }
}

/// `w` as `F16`: the tower's `BF16` at the same precision, in the type
/// every backend's floating-point GEMM takes.
fn requantize(w: &QuantMatrix) -> QuantMatrix {
    let rows: Vec<f32> = (0..w.out_dim).flat_map(|o| w.row(o)).collect();
    let ggml_type = crate::engine::quant::GGML_TYPE_F16;
    let bytes = crate::quantize::encode(ggml_type, &rows, w.in_dim);
    QuantMatrix::from_encoded_rows(bytes, ggml_type, w.in_dim, w.out_dim, None)
}

/// The size a `width` x `height` picture is read at: its aspect ratio
/// kept, each side a multiple of `unit`, and `(width / unit) * (height /
/// unit)` — the tokens it becomes — between `min_tokens` and `max_tokens`.
pub fn fit_size(
    width: usize,
    height: usize,
    unit: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> (usize, usize) {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let u = unit as f64;
    let round = |v: f64| ((v / u).round() as usize * unit).max(unit);
    let floor = |v: f64| ((v / u).floor() as usize * unit).max(unit);
    let ceil = |v: f64| ((v / u).ceil() as usize * unit).max(unit);
    let (min_px, max_px) = (
        (min_tokens * unit * unit) as f64,
        (max_tokens * unit * unit) as f64,
    );
    let (rw, rh) = (round(w), round(h));
    if (rw * rh) as f64 > max_px {
        let beta = (w * h / max_px).sqrt();
        (floor(w / beta), floor(h / beta))
    } else if ((rw * rh) as f64) < min_px {
        let beta = (min_px / (w * h)).sqrt();
        (ceil(w * beta), ceil(h * beta))
    } else {
        (rw, rh)
    }
}

/// `(cos, sin)` per patch and rotated pair: the first quarter of a head's
/// pairs turn with the patch's column, the second with its row.
fn rotary_angles(rows: usize, cols: usize, head_dim: usize) -> Vec<(f32, f32)> {
    let half = head_dim / 2;
    let pairs = half / 2;
    let freqs: Vec<f64> = (0..pairs)
        .map(|j| ROPE_THETA.powf(-2.0 * j as f64 / half as f64))
        .collect();
    (0..rows * cols)
        .flat_map(|i| {
            let (r, c) = ((i / cols) as f64, (i % cols) as f64);
            let freqs = &freqs;
            (0..2 * pairs).map(move |j| {
                let a = if j < pairs {
                    c * freqs[j]
                } else {
                    r * freqs[j - pairs]
                };
                (a.cos() as f32, a.sin() as f32)
            })
        })
        .collect()
}

/// Rotates each head's two halves NEOX-style, `(j, j + quarter)` within
/// the half, by that patch's angles.
fn rope_2d(x: &mut [f32], dim: usize, head_dim: usize, angles: &[(f32, f32)]) {
    let half = head_dim / 2;
    let pairs = half / 2;
    x.par_chunks_mut(dim)
        .zip(angles.par_chunks(2 * pairs))
        .for_each(|(row, angles)| {
            for head in row.chunks_mut(head_dim) {
                for (part, angles) in head.chunks_mut(half).zip(angles.chunks(pairs)) {
                    let (lo, hi) = part.split_at_mut(pairs);
                    for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi.iter_mut()).zip(angles) {
                        let (x0, x1) = (*a, *b);
                        *a = x0 * cos - x1 * sin;
                        *b = x0 * sin + x1 * cos;
                    }
                }
            }
        });
}

/// RMSNorm without a weight over each `dim`-wide row.
fn rms_unweighted(x: &mut [f32], dim: usize, eps: f32) {
    x.par_chunks_mut(dim).for_each(|row| {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for v in row.iter_mut() {
            *v *= scale;
        }
    });
}

/// `x · σ(1.702 x)`.
fn gelu_quick(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// The first virtual id: past every vocabulary this server reads.
pub const FIRST_ID: u32 = 1 << 24;

/// Whether `id` names a picture's row rather than a vocabulary token.
pub fn is_virtual(id: u32) -> bool {
    id >= FIRST_ID
}

/// One picture's rows, under the virtual ids `base..base + rows`.
struct Entry {
    base: u32,
    rows: usize,
    data: Arc<[f32]>,
    used: u64,
}

#[derive(Default)]
struct StoreInner {
    by_key: HashMap<[u8; 32], Entry>,
    /// `base → key`, for finding the picture an id belongs to.
    by_base: BTreeMap<u32, [u8; 32]>,
    clock: u64,
    floats: usize,
}

/// The encoded pictures the prompts in flight refer to — see the module
/// doc. Bounded: past [`STORE_FLOATS`] the least recently used pictures
/// are dropped, and a prompt that still names one is encoded again.
#[derive(Default)]
pub struct ImageStore {
    inner: Mutex<StoreInner>,
}

/// Floats the store keeps, 512 MiB of them.
const STORE_FLOATS: usize = 128 << 20;

impl ImageStore {
    /// The virtual ids picture `key` already has, refreshed as used.
    pub fn ids(&self, key: &[u8; 32]) -> Option<std::ops::Range<u32>> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.clock += 1;
        let clock = inner.clock;
        let entry = inner.by_key.get_mut(key)?;
        entry.used = clock;
        Some(entry.base..entry.base + entry.rows as u32)
    }

    /// Stores picture `key`'s `rows` rows of `width` and gives them ids at
    /// or above `first_id`, placed by the key so that a picture keeps its
    /// ids for as long as the store keeps it.
    pub fn insert(
        &self,
        key: [u8; 32],
        data: Vec<f32>,
        width: usize,
        first_id: u32,
    ) -> std::ops::Range<u32> {
        let rows = data.len() / width;
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = inner.by_key.get(&key) {
            return entry.base..entry.base + entry.rows as u32;
        }
        let span = u32::MAX - first_id - rows as u32;
        let seed = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
        let mut base = first_id + seed % span;
        // Moved past whatever it overlaps, so no two pictures share an id.
        loop {
            let end = base + rows as u32;
            let clash = inner
                .by_base
                .range(..end)
                .next_back()
                .map(|(&b, k)| (b, inner.by_key[k].rows as u32))
                .filter(|&(b, r)| b + r > base);
            match clash {
                Some((b, r)) => {
                    base = b + r;
                    if base > u32::MAX - rows as u32 {
                        base = first_id;
                    }
                }
                None => break,
            }
        }
        inner.clock += 1;
        inner.floats += data.len();
        let used = inner.clock;
        inner.by_base.insert(base, key);
        inner.by_key.insert(
            key,
            Entry {
                base,
                rows,
                data: data.into(),
                used,
            },
        );
        while inner.floats > STORE_FLOATS && inner.by_key.len() > 1 {
            let oldest = inner
                .by_key
                .iter()
                .filter(|(k, _)| **k != key)
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| *k);
            let Some(oldest) = oldest else { break };
            if let Some(e) = inner.by_key.remove(&oldest) {
                inner.by_base.remove(&e.base);
                inner.floats -= e.data.len();
            }
        }
        base..base + rows as u32
    }

    /// The row virtual id `id` stands for, `width` wide.
    pub fn row(&self, id: u32) -> Option<(Arc<[f32]>, usize)> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let (&base, key) = inner.by_base.range(..=id).next_back()?;
        let entry = &inner.by_key[key];
        let index = (id - base) as usize;
        (index < entry.rows).then(|| (entry.data.clone(), index))
    }
}

/// The served model's way of reading pictures: the tower, and the tokens a
/// picture's rows are framed by in the prompt.
pub struct Vision {
    encoder: Encoder,
    /// Mixed into every picture's key: the tower and its budget, so a
    /// picture another tower read is never taken for this one's.
    salt: [u8; 32],
    begin: u32,
    end: u32,
    /// One picture through the tower at a time: it fills every core (or
    /// the device) on its own.
    busy: Mutex<()>,
}

impl Vision {
    /// The tower at `path`, for a model whose tokenizer is `tokenizer`.
    pub fn new(
        encoder: Encoder,
        path: &Path,
        tokenizer: &crate::engine::tokenizer::Tokenizer,
    ) -> Result<Self> {
        let single = |text: &str| -> Result<u32> {
            match tokenizer.encode(text, false).as_slice() {
                [id] => Ok(*id),
                _ => anyhow::bail!("the model's vocabulary has no {text} token"),
            }
        };
        use sha2::Digest;
        let mut salt = sha2::Sha256::new();
        salt.update(path.to_string_lossy().as_bytes());
        salt.update((encoder.max_tokens as u64).to_le_bytes());
        Ok(Self {
            begin: single("<|image>")?,
            end: single("<image|>")?,
            salt: salt.finalize().into(),
            encoder,
            busy: Mutex::new(()),
        })
    }

    /// The most tokens a picture becomes.
    pub fn max_tokens(&self) -> usize {
        self.encoder.max_tokens
    }

    /// The prompt tokens picture `bytes` becomes: its rows framed by the
    /// model's picture markers. Encoded once; a picture the [`store`]
    /// still holds is the same tokens again without the tower.
    pub fn tokens(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        use sha2::Digest;
        let mut key = sha2::Sha256::new();
        key.update(self.salt);
        key.update(bytes);
        let key: [u8; 32] = key.finalize().into();
        let ids = match store().ids(&key) {
            Some(ids) => ids,
            None => {
                let _busy = self.busy.lock().unwrap_or_else(|p| p.into_inner());
                match store().ids(&key) {
                    Some(ids) => ids,
                    None => {
                        let started = std::time::Instant::now();
                        let rows = self.encoder.encode_bytes(bytes)?;
                        log::info!(
                            "orangu-server: [vision] a picture read as {} tokens in {:.2?}",
                            rows.len() / self.encoder.out_width,
                            started.elapsed()
                        );
                        store().insert(key, rows, self.encoder.out_width, FIRST_ID)
                    }
                }
            }
        };
        let mut tokens = Vec::with_capacity(ids.len() + 2);
        tokens.push(self.begin);
        tokens.extend(ids);
        tokens.push(self.end);
        Ok(tokens)
    }
}

/// The process's one store: a model reads virtual ids from it, and every
/// request that carries a picture writes to it.
pub fn store() -> &'static ImageStore {
    static STORE: std::sync::OnceLock<ImageStore> = std::sync::OnceLock::new();
    STORE.get_or_init(ImageStore::default)
}

/// Writes token `id`'s embedding into `dst`: a vocabulary row through
/// `vocab`, a virtual id's row from the [`store`] as it is.
pub fn embed_row(
    id: u32,
    n_vocab: usize,
    dst: &mut [f32],
    vocab: impl FnOnce(&mut [f32]),
) -> Result<()> {
    if (id as usize) < n_vocab {
        vocab(dst);
        return Ok(());
    }
    let (data, index) = store()
        .row(id)
        .with_context(|| format!("token id {id} is out of vocab range"))?;
    let width = dst.len();
    let src = data
        .get(index * width..(index + 1) * width)
        .context("a picture's rows are not the model's width")?;
    dst.copy_from_slice(src);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_inside_the_budget_keeps_its_rounded_size() {
        // 16 x 12 cells of 48 pixels: 192 tokens.
        assert_eq!(fit_size(768, 576, 48, 70, 280), (768, 576));
        assert_eq!(fit_size(770, 580, 48, 70, 280), (768, 576));
    }

    #[test]
    fn a_large_picture_is_shrunk_to_the_budget() {
        let (w, h) = fit_size(4000, 3000, 48, 70, 280);
        assert!((w / 48) * (h / 48) <= 280, "{w} x {h}");
        assert!(w % 48 == 0 && h % 48 == 0);
        // The aspect ratio survives the shrink.
        assert!(((w as f64 / h as f64) - 4.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn a_small_picture_is_grown_to_the_minimum() {
        let (w, h) = fit_size(32, 32, 48, 70, 280);
        assert!((w / 48) * (h / 48) >= 70, "{w} x {h}");
    }

    #[test]
    fn a_requantized_weight_keeps_its_rows() {
        let rows: Vec<f32> = (0..3 * 40)
            .map(|i| ((i * 7919) % 201) as f32 / 50.0 - 2.0)
            .collect();
        let w = requantize(&QuantMatrix::from_f32_rows(rows.clone(), 40, 3));
        assert_eq!((w.in_dim, w.out_dim), (40, 3));
        for o in 0..3 {
            for (got, want) in w.row(o).iter().zip(&rows[o * 40..(o + 1) * 40]) {
                assert!((got - want).abs() < 2e-3, "row {o}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn rotary_angles_turn_columns_then_rows() {
        // Two patches side by side: only the column half differs.
        let a = rotary_angles(1, 2, 8);
        let (first, second) = a.split_at(4);
        assert_eq!(first, &[(1.0, 0.0); 4]);
        assert!((second[0].1 - 1f32.sin()).abs() < 1e-6);
        assert_eq!(&second[2..], &[(1.0, 0.0); 2]);
    }

    #[test]
    fn a_stored_picture_keeps_its_ids_and_gives_back_its_rows() {
        let store = ImageStore::default();
        let key = [7u8; 32];
        let ids = store.insert(key, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 1000);
        assert_eq!(ids.len(), 3);
        assert!(ids.start >= 1000);
        assert_eq!(store.ids(&key), Some(ids.clone()));
        assert_eq!(store.insert(key, vec![0.0; 6], 2, 1000), ids);
        let (data, index) = store.row(ids.start + 2).unwrap();
        assert_eq!(&data[index * 2..index * 2 + 2], &[5.0, 6.0]);
        assert!(store.row(ids.end).is_none_or(|(_, i)| i != 3));
    }

    /// The tower against a reference's own rows for one picture:
    /// `ORANGU_TEST_GEMMA4V` the projector, `_PICTURE` a picture whose
    /// sides are multiples of 48 (so neither side resizes it) and `_REF`
    /// the reference's dump (`[i32 tokens][i32 width][f32 rows]`).
    #[test]
    #[ignore]
    fn the_tower_matches_the_reference_rows() {
        let var = |k: &str| std::env::var(format!("ORANGU_TEST_GEMMA4V{k}")).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let encoder = Encoder::load(Path::new(&var("")), backend, MAX_TOKENS).unwrap();
        let picture = std::fs::read(var("_PICTURE")).unwrap();
        let reference = std::fs::read(var("_REF")).unwrap();
        let header = |i: usize| i32::from_le_bytes(reference[i * 4..i * 4 + 4].try_into().unwrap());
        let (tokens, width) = (header(0) as usize, header(1) as usize);
        let expected: Vec<f32> = reference[8..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let started = std::time::Instant::now();
        let got = encoder.encode_bytes(&picture).unwrap();
        eprintln!("encoded in {:.2?}", started.elapsed());
        assert_eq!(got.len(), tokens * width);
        let cosines: Vec<f64> = got
            .chunks(width)
            .zip(expected.chunks(width))
            .map(|(a, b)| {
                let dot: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (*x as f64) * (*y as f64))
                    .sum();
                let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb)
            })
            .collect();
        // A row of zeros has no cosine, and fails here rather than passing.
        let worst = cosines
            .iter()
            .copied()
            .fold(1.0f64, |w, c| if c < w || c.is_nan() { c } else { w });
        let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
        eprintln!("row cosine: worst {worst:.5}, mean {mean:.5}");
        let mut order: Vec<usize> = (0..cosines.len()).collect();
        order.sort_by(|&a, &b| cosines[a].total_cmp(&cosines[b]));
        for &i in order.iter().take(8) {
            eprintln!("  row {i}: {:.4}", cosines[i]);
        }
        assert!(
            mean > 0.999 && worst > 0.95,
            "row cosine: worst {worst}, mean {mean}"
        );
    }

    #[test]
    fn two_pictures_never_share_an_id() {
        let store = ImageStore::default();
        // Same placement seed, so the second lands on the first.
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        b[31] = 1;
        a[..4].copy_from_slice(&5u32.to_le_bytes());
        b[..4].copy_from_slice(&5u32.to_le_bytes());
        let ia = store.insert(a, vec![0.0; 10], 1, 100);
        let ib = store.insert(b, vec![1.0; 10], 1, 100);
        assert!(ia.end <= ib.start || ib.end <= ia.start, "{ia:?} {ib:?}");
        let (data, index) = store.row(ib.start).unwrap();
        assert_eq!(data[index], 1.0);
    }
}

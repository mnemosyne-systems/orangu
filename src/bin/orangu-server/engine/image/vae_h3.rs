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

//! MiniMax-H3-Image's VAE decoder (`MiniMaxH3ImageVAE`,
//! `vae/diffusion_pytorch_model.safetensors` of `MiniMaxAI/MiniMax-H3-Image`):
//! 24 latent channels at a sixteenth of the picture, decoded by a vision
//! transformer rather than convolutions.
//!
//! Each latent cell is one token: a 1x1 `post_quant_conv`, a linear to
//! 1024 wide, four learned register tokens and one zero token appended,
//! then 24 pre-norm blocks — RMSNorm, attention over 16 heads of 64 whose
//! queries and keys are RMS-normalised without a weight and turned by a
//! two-axis rotary embedding over their first 32 values, a SwiGLU
//! feed-forward, each branch scaled by a learned per-channel vector — and a
//! LayerNorm and a linear to the token's 16x16 RGB pixels, which are
//! ImageNet-normalised and clamped to `[0, 1]`.
//!
//! The picture is decoded in tiles of 16x16 cells (256 pixels) overlapping
//! by a quarter and blended with a smootherstep ramp, as the reference does
//! (`vae_decoder_tiling`): attention never spans more than a tile, and the
//! rotary positions are relative to it.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::safetensors::SafeTensors;
use super::transformer::step_attention;
use super::vae::{Feature, VaePrecision, conv_matrix};

/// Latent channels.
pub const Z_DIM: usize = 24;
/// Pixels a latent cell covers along each side.
pub const SPATIAL_COMPRESSION: usize = 16;
/// The latent statistics the transformer's latents are normalised by —
/// `vae/config.json`'s `latents_mean` and `latents_std`.
pub const LATENTS_MEAN: [f32; Z_DIM] = [
    0.858_090_34,
    -0.960_659_1,
    1.066_164,
    -0.509_032_5,
    -0.272_758_2,
    -1.367_541_4,
    -0.255_325_5,
    -0.269_075_54,
    -0.537_684_1,
    -0.046_409_73,
    0.665_737,
    0.196_901_27,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_341_99,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_44,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];
pub const LATENTS_STD: [f32; Z_DIM] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_8,
    0.799_680_95,
    0.449_889,
    0.719_74,
    0.693_629_3,
    2.961_095,
    2.769_42,
    3.049_618_5,
    2.108_805_4,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];
/// ImageNet's statistics, which the decoder's pixels are normalised by.
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];
const EPS: f32 = 1e-5;
const HEAD_DIM: usize = 64;
/// Learned register tokens; one zero token follows them.
const REGISTERS: usize = 4;
/// Rotary frequencies an axis, base 100, over `(row, column)`.
const ROPE_FREQS: usize = 8;
const ROPE_BASE: f32 = 100.0;
/// The tile side in latent cells, and the overlap the tiler aims for.
const TILE: usize = 16;
const TILE_OVERLAP: f32 = 0.25;

struct Linear {
    w: QuantMatrix,
    bias: Vec<f32>,
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    /// Per head `[q | k | v]`, each [`HEAD_DIM`] wide.
    qkv: Linear,
    out: Linear,
    /// `[gate | up]`.
    w1: Linear,
    w2: Linear,
    scale1: Vec<f32>,
    scale2: Vec<f32>,
}

pub struct H3ImageVae {
    backend: Arc<dyn Backend>,
    post_quant: Linear,
    embed: Linear,
    registers: Vec<f32>,
    blocks: Vec<Block>,
    norm_out: (Vec<f32>, Vec<f32>),
    proj_out: Linear,
    dim: usize,
}

impl H3ImageVae {
    pub fn load(path: &Path, backend: Arc<dyn Backend>, precision: VaePrecision) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        Self::from_safetensors(&file, backend, precision)
            .with_context(|| format!("reading the MiniMax-H3-Image VAE from {}", path.display()))
    }

    fn from_safetensors(
        file: &SafeTensors,
        backend: Arc<dyn Backend>,
        precision: VaePrecision,
    ) -> Result<Self> {
        let vector = |name: &str| -> Result<Vec<f32>> { Ok(file.tensor(name)?.0) };
        let linear = |name: &str| -> Result<Linear> {
            let (w, shape) = file.tensor(&format!("{name}.weight"))?;
            let (cout, cin) = match shape {
                [cout, cin] => (*cout, *cin),
                [cout, cin, 1, 1] => (*cout, *cin),
                other => bail!("{name}: unexpected weight shape {other:?}"),
            };
            // The int8 kernel reads rows in blocks of 256: a narrower input
            // (the 24 latent channels) stays exact.
            let w = if cin.is_multiple_of(256) {
                conv_matrix(w, cin, cout, precision)
            } else {
                QuantMatrix::from_f32_rows(w, cin, cout)
            };
            let bias = vector(&format!("{name}.bias"))?;
            ensure!(bias.len() == cout, "{name}: bias is not {cout} wide");
            Ok(Linear { w, bias })
        };
        let blocks = (0..)
            .map(|i| format!("decoder.transformer_blocks.{i}"))
            .take_while(|p| file.has(&format!("{p}.norm1.weight")))
            .map(|p| {
                Ok(Block {
                    norm1: vector(&format!("{p}.norm1.weight"))?,
                    norm2: vector(&format!("{p}.norm2.weight"))?,
                    qkv: linear(&format!("{p}.attn.to_qkv"))?,
                    out: linear(&format!("{p}.attn.to_out"))?,
                    w1: linear(&format!("{p}.ff.w1"))?,
                    w2: linear(&format!("{p}.ff.w2"))?,
                    scale1: vector(&format!("{p}.scale1"))?,
                    scale2: vector(&format!("{p}.scale2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!blocks.is_empty(), "no decoder.transformer_blocks");
        let embed = linear("decoder.x_embedder")?;
        let dim = embed.w.out_dim;
        let post_quant = linear("post_quant_conv")?;
        let proj_out = linear("decoder.proj_out")?;
        let registers = vector("decoder.register_tokens")?;
        ensure!(
            embed.w.in_dim == Z_DIM
                && post_quant.w.in_dim == Z_DIM
                && post_quant.w.out_dim == Z_DIM,
            "the VAE's latent width is not {Z_DIM}"
        );
        ensure!(
            dim.is_multiple_of(HEAD_DIM)
                && registers.len() == REGISTERS * dim
                && proj_out.w.out_dim == 3 * SPATIAL_COMPRESSION * SPATIAL_COMPRESSION,
            "the VAE decoder is not a {}x{} RGB vision transformer",
            SPATIAL_COMPRESSION,
            SPATIAL_COMPRESSION
        );
        for (i, b) in blocks.iter().enumerate() {
            ensure!(
                b.qkv.w.out_dim == 3 * dim
                    && b.out.w.out_dim == dim
                    && b.w1.w.out_dim == 2 * b.w2.w.in_dim
                    && b.w2.w.out_dim == dim
                    && b.scale1.len() == dim,
                "decoder.transformer_blocks.{i} does not match the width {dim}"
            );
        }
        Ok(Self {
            backend,
            post_quant,
            embed,
            registers,
            blocks,
            norm_out: (
                vector("decoder.norm_out.weight")?,
                vector("decoder.norm_out.bias")?,
            ),
            proj_out,
            dim,
        })
    }

    fn linear(&self, x: &[f32], n: usize, l: &Linear) -> Vec<f32> {
        let mut out = self
            .backend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: n,
                w: &l.w,
            }])
            .pop()
            .expect("one op in, one result out");
        tensor::add_bias_per_row(&mut out, &l.bias, n);
        out
    }

    /// Pixels from a latent: `[h/16 * w/16, 24]` in VAE space (already
    /// de-normalised) to `[h * w, 3]` RGB in `[-1, 1]`, tile by tile. Stops
    /// between tiles and layers once `cancel` is set.
    pub fn decode_unless(
        &self,
        latent: &Feature,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Feature> {
        ensure!(
            latent.channels == Z_DIM,
            "latent has {} channels",
            latent.channels
        );
        let (lh, lw) = (latent.height, latent.width);
        let s = SPATIAL_COMPRESSION;
        let (ph, pw) = (lh * s, lw * s);
        let mut out = vec![0.0f32; ph * pw * 3];
        let ys = tile_plan(lh, TILE, TILE_OVERLAP);
        let xs = tile_plan(lw, TILE, TILE_OVERLAP);
        let (th, tw) = (TILE.min(lh), TILE.min(lw));
        for y in &ys {
            for x in &xs {
                if super::cancelled(cancel) {
                    bail!("cancelled");
                }
                let mut tile = Vec::with_capacity(th * tw * Z_DIM);
                for r in 0..th {
                    let at = ((y.offset + r) * lw + x.offset) * Z_DIM;
                    tile.extend_from_slice(&latent.data[at..at + tw * Z_DIM]);
                }
                let pixels = self.decode_tile(&tile, th, tw, cancel)?;
                merge(&pixels, (th * s, tw * s), &mut out, pw, (y, x), s);
            }
        }
        for v in out.iter_mut() {
            *v = 2.0 * v.clamp(0.0, 1.0) - 1.0;
        }
        Ok(Feature::new(ph, pw, 3, out))
    }

    /// One tile of `[h * w, 24]` latent cells to `[16h * 16w, 3]` pixels in
    /// `[0, 1]`.
    fn decode_tile(
        &self,
        latent: &[f32],
        h: usize,
        w: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<f32>> {
        let dim = self.dim;
        let n = h * w;
        let z = self.linear(latent, n, &self.post_quant);
        let mut x = self.linear(&z, n, &self.embed);
        x.extend_from_slice(&self.registers);
        x.resize((n + REGISTERS + 1) * dim, 0.0);
        let m = n + REGISTERS + 1;
        let rope = rope_table(h, w, REGISTERS + 1);
        let n_head = dim / HEAD_DIM;
        let mut normed = Vec::new();
        for block in &self.blocks {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            tensor::rmsnorm_into(&mut normed, &x, &block.norm1, m, dim, EPS);
            let qkv = self.linear(&normed, m, &block.qkv);
            let (mut q, mut k, v) = split_heads(&qkv, m, n_head);
            drop(qkv);
            for t in [&mut q, &mut k] {
                t.par_chunks_mut(HEAD_DIM).for_each(|head| {
                    let mean_sq = head.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
                    let scale = 1.0 / (mean_sq + EPS).sqrt();
                    head.iter_mut().for_each(|v| *v *= scale);
                });
                super::minimax_h3::apply_partial_rope(t, dim, HEAD_DIM, 2 * ROPE_FREQS, &rope);
            }
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            let attn = step_attention(&q, m, &k, &v, m, n_head, HEAD_DIM, scale, None);
            drop((q, k, v));
            let out = self.linear(&attn, m, &block.out);
            scaled_add(&mut x, &out, &block.scale1, dim);
            tensor::rmsnorm_into(&mut normed, &x, &block.norm2, m, dim, EPS);
            let gate_up = self.linear(&normed, m, &block.w1);
            let hidden = super::minimax_h3::swiglu(&gate_up, block.w2.w.in_dim);
            drop(gate_up);
            let out = self.linear(&hidden, m, &block.w2);
            scaled_add(&mut x, &out, &block.scale2, dim);
        }
        x.truncate(n * dim);
        layer_norm_inplace(&mut x, &self.norm_out.0, &self.norm_out.1, dim);
        let patches = self.linear(&x, n, &self.proj_out);
        // A token's values are `[channel][dy][dx]`.
        let s = SPATIAL_COMPRESSION;
        let (ph, pw) = (h * s, w * s);
        let mut pixels = vec![0.0f32; ph * pw * 3];
        pixels
            .par_chunks_mut(pw * 3)
            .enumerate()
            .for_each(|(py, row)| {
                let (r, dy) = (py / s, py % s);
                for (px, rgb) in row.chunks_mut(3).enumerate() {
                    let (c, dx) = (px / s, px % s);
                    let token = &patches[(r * w + c) * 3 * s * s..];
                    for ch in 0..3 {
                        let v = token[ch * s * s + dy * s + dx];
                        rgb[ch] = (v * PIXEL_STD[ch] + PIXEL_MEAN[ch]).clamp(0.0, 1.0);
                    }
                }
            });
        Ok(pixels)
    }
}

/// `q`, `k` and `v` out of rows whose heads each hold `[q | k | v]`.
fn split_heads(rows: &[f32], n: usize, n_head: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let width = n_head * HEAD_DIM;
    debug_assert_eq!(rows.len(), n * 3 * width);
    let mut q = vec![0.0f32; n * width];
    let mut k = vec![0.0f32; n * width];
    let mut v = vec![0.0f32; n * width];
    q.par_chunks_mut(width)
        .zip(k.par_chunks_mut(width))
        .zip(v.par_chunks_mut(width))
        .zip(rows.par_chunks(3 * width))
        .for_each(|(((q, k), v), row)| {
            for (h, head) in row.chunks(3 * HEAD_DIM).enumerate() {
                let at = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                q[at.clone()].copy_from_slice(&head[..HEAD_DIM]);
                k[at.clone()].copy_from_slice(&head[HEAD_DIM..2 * HEAD_DIM]);
                v[at].copy_from_slice(&head[2 * HEAD_DIM..]);
            }
        });
    (q, k, v)
}

/// `x += scale * y`, per row.
fn scaled_add(x: &mut [f32], y: &[f32], scale: &[f32], dim: usize) {
    x.par_chunks_mut(dim)
        .zip(y.par_chunks(dim))
        .for_each(|(row, add)| {
            for ((v, a), s) in row.iter_mut().zip(add).zip(scale) {
                *v += s * a;
            }
        });
}

fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], dim: usize) {
    x.par_chunks_mut(dim).for_each(|row| {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = (*v - mean) * inv * w + b;
        }
    });
}

/// `(cos, sin)` for each of the 16 rotated pairs of every token of an `h x
/// w` tile, then `extra` tokens at the origin: pairs `0..8` turn with the
/// row, `8..16` with the column, each at `(2 (i + 0.5) / side - 1) · 2π`.
fn rope_table(h: usize, w: usize, extra: usize) -> Vec<(f32, f32)> {
    let freqs: Vec<f32> = (0..ROPE_FREQS)
        .map(|i| 1.0 / ROPE_BASE.powf(i as f32 / ROPE_FREQS as f32))
        .collect();
    let place = |i: usize, side: usize| {
        (2.0 * ((i as f32 + 0.5) / side as f32) - 1.0) * std::f32::consts::TAU
    };
    let mut table = Vec::with_capacity((h * w + extra) * 2 * ROPE_FREQS);
    for r in 0..h {
        for c in 0..w {
            for p in [place(r, h), place(c, w)] {
                for f in &freqs {
                    let a = p * f;
                    table.push((a.cos(), a.sin()));
                }
            }
        }
    }
    table.resize((h * w + extra) * 2 * ROPE_FREQS, (1.0, 0.0));
    table
}

/// One tile along an axis: where it starts and how much of it overlaps its
/// neighbours on either side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    offset: usize,
    before: usize,
    after: usize,
}

/// The tiles along an axis of `dimension` cells: evenly spaced, `tile`
/// wide, as close to `overlap` of a tile apart as whole intervals allow,
/// with no point under three tiles — the reference's tiling plan.
fn tile_plan(dimension: usize, tile: usize, overlap: f32) -> Vec<Span> {
    if tile >= dimension {
        return vec![Span {
            offset: 0,
            before: 0,
            after: 0,
        }];
    }
    let extent = dimension - tile;
    let target = extent as f32 / ((1.0 - overlap) * tile as f32);
    let min_intervals = 1 + (extent - 1) / (tile - 1);
    let max_intervals = (2 * extent / tile).max(1);
    let clamp = |v: f32| (v as usize).clamp(min_intervals, max_intervals.max(min_intervals));
    let (lower, upper) = (clamp(target.floor()), clamp(target.ceil()));
    let error = |count: usize| (1.0 / count as f32 - 1.0 / target).abs();
    let intervals = if error(upper) < error(lower) {
        upper
    } else {
        lower
    };
    let position = |i: usize| i * extent / intervals;
    (0..=intervals)
        .map(|i| {
            let offset = position(i);
            Span {
                offset,
                before: if i > 0 {
                    position(i - 1) + tile - offset
                } else {
                    0
                },
                after: if i < intervals {
                    offset + tile - position(i + 1)
                } else {
                    0
                },
            }
        })
        .collect()
}

/// Adds a decoded tile into the picture, its overlapping edges ramped by a
/// smootherstep so overlapping tiles sum to one.
fn merge(
    tile: &[f32],
    (th, tw): (usize, usize),
    out: &mut [f32],
    out_w: usize,
    (y, x): (&Span, &Span),
    scale: usize,
) {
    let smootherstep = |v: f32| v * v * v * (v * (6.0 * v - 15.0) + 10.0);
    let ramp = |i: usize, len: usize, before: usize, after: usize| {
        let mut f = 1.0f32;
        if i < before {
            f = i as f32 / before as f32;
        }
        if i >= len - after {
            f = (len - i) as f32 / after as f32;
        }
        smootherstep(f.clamp(0.0, 1.0))
    };
    let (top, bottom) = (y.before * scale, y.after * scale);
    let (left, right) = (x.before * scale, x.after * scale);
    let (oy, ox) = (y.offset * scale, x.offset * scale);
    for iy in 0..th {
        let wy = ramp(iy, th, top, bottom);
        let dst = &mut out[((oy + iy) * out_w + ox) * 3..((oy + iy) * out_w + ox + tw) * 3];
        let src = &tile[iy * tw * 3..(iy + 1) * tw * 3];
        for ix in 0..tw {
            let wgt = wy * ramp(ix, tw, left, right);
            for ch in 0..3 {
                dst[ix * 3 + ch] += wgt * src[ix * 3 + ch];
            }
        }
    }
}

/// The transformer's latents, `[cells, 24]` and normalised, into the VAE's
/// space in place.
pub fn denormalize(z: &mut [f32]) {
    z.par_chunks_mut(Z_DIM).for_each(|cell| {
        for ((v, m), s) in cell.iter_mut().zip(LATENTS_MEAN).zip(LATENTS_STD) {
            *v = *v * s + m;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's plan for a 64-cell side: five tiles 12 apart, each
    /// overlapping its neighbours by 4.
    #[test]
    fn a_64_cell_side_is_five_tiles() {
        let plan = tile_plan(64, 16, 0.25);
        let offsets: Vec<usize> = plan.iter().map(|s| s.offset).collect();
        assert_eq!(offsets, vec![0, 12, 24, 36, 48]);
        assert_eq!(plan[0].before, 0);
        assert_eq!(plan[0].after, 4);
        assert_eq!(plan[2].before, 4);
        assert_eq!(plan[4].after, 0);
        assert_eq!(tile_plan(16, 16, 0.25).len(), 1);
    }

    /// Overlapping ramps sum to one, so a constant picture stays constant.
    #[test]
    fn merged_tiles_keep_a_constant() {
        let lh = 40;
        let plan = tile_plan(lh, 16, 0.25);
        let s = 1;
        let mut out = vec![0.0f32; lh * lh * 3];
        let tile = vec![0.5f32; 16 * 16 * 3];
        for y in &plan {
            for x in &plan {
                merge(&tile, (16, 16), &mut out, lh, (y, x), s);
            }
        }
        for v in out {
            assert!((v - 0.5).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn heads_hold_q_k_v_in_turn() {
        // Two heads of 64: head h's q, k, v are 64 values each.
        let rows: Vec<f32> = (0..2 * 3 * HEAD_DIM).map(|i| i as f32).collect();
        let (q, k, v) = split_heads(&rows, 1, 2);
        assert_eq!(q[0], 0.0);
        assert_eq!(k[0], 64.0);
        assert_eq!(v[0], 128.0);
        assert_eq!(q[64], 192.0);
        assert_eq!(v[127], 383.0);
    }

    #[test]
    fn rope_table_places_the_tile_and_parks_the_registers() {
        let t = rope_table(2, 2, 5);
        assert_eq!(t.len(), (4 + 5) * 16);
        // Row 0 of 2: (2 * 0.25 - 1) * 2π = -π at frequency 1.
        assert!((t[0].0 - (-std::f32::consts::PI).cos()).abs() < 1e-6);
        assert_eq!(t[4 * 16], (1.0, 0.0));
    }
}

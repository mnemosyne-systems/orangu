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

//! The FLUX.2 VAE — diffusers' `AutoencoderKLFlux2`: the classic 2-D
//! `AutoencoderKL` (four stages of `128, 256, 512, 512` channels, two
//! residual blocks a stage going down and three coming up, one attention
//! block in the middle, GroupNorm of 32 groups throughout) with 32 latent
//! channels at an eighth of the picture, plus a running-statistics
//! `BatchNorm` over the *patchified* latent — 2x2 cells folded into 128
//! channels, the transformer's token — which the pipeline normalises by
//! ([`Flux2Vae::bn_mean`], [`Flux2Vae::bn_std`]).
//!
//! Read from a `safetensors` in either naming: diffusers' own
//! (`decoder.up_blocks.N.resnets.M`, the `vae/` folder of
//! `black-forest-labs/FLUX.2-klein-base-4B`) or the original one ComfyUI's
//! `flux2-vae.safetensors` keeps (`decoder.up.N.block.M`, its stages
//! numbered from the bottom). The convolutions, the channel-last feature
//! maps and the middle attention are [`super::vae`]'s.

use crate::engine::backend::Backend;
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::safetensors::SafeTensors;
use super::transformer::joint_attention;
use super::vae::{
    Conv, Feature, Padding, VaePrecision, load_conv, nearest_upsample, run_conv, silu_inplace,
};

/// Latent channels.
pub const Z_DIM: usize = 32;
/// Pixels per latent cell along each axis.
pub const SPATIAL_COMPRESSION: usize = 8;
/// Channels of a patchified latent cell: `Z_DIM` × the 2x2 patch.
pub const PATCHED: usize = Z_DIM * 4;
/// `batch_norm_eps` in `vae/config.json`.
const BATCH_NORM_EPS: f32 = 1e-4;
/// GroupNorm groups and epsilon.
const GROUPS: usize = 32;
const NORM_EPS: f32 = 1e-6;

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct ResBlock {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct AttnBlock {
    norm: Norm,
    q: Conv,
    k: Conv,
    v: Conv,
    out: Conv,
}

struct Stage {
    resnets: Vec<ResBlock>,
    /// The stride-2 convolution (encoder) or the nearest-2x upsample's
    /// convolution (decoder) after the residual blocks, if this stage has one.
    resample: Option<Conv>,
}

struct Tower {
    conv_in: Conv,
    stages: Vec<Stage>,
    mid: (ResBlock, AttnBlock, ResBlock),
    norm_out: Norm,
    conv_out: Conv,
}

pub struct Flux2Vae {
    backend: Arc<dyn Backend>,
    encoder: Tower,
    decoder: Tower,
    quant: Conv,
    post_quant: Conv,
    /// `bn.running_mean`, `[PATCHED]`.
    pub bn_mean: Vec<f32>,
    /// `sqrt(bn.running_var + eps)`, `[PATCHED]`.
    pub bn_std: Vec<f32>,
}

/// The original-naming stem of a diffusers-naming one, for a decoder of
/// `n_up` stages (the original numbers them from the bottom).
fn original_name(name: &str, n_up: usize) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let leaf = |rest: &[&str]| -> String {
        rest.iter()
            .map(|p| match *p {
                "conv_shortcut" => "nin_shortcut",
                "group_norm" => "norm",
                "to_q" => "q",
                "to_k" => "k",
                "to_v" => "v",
                other => other,
            })
            .collect::<Vec<_>>()
            .join(".")
    };
    let tower = parts[0];
    match parts.get(1).copied() {
        Some("conv_norm_out") => format!("{tower}.norm_out"),
        Some("down_blocks") | Some("up_blocks") if parts.len() >= 5 => {
            let up = parts[1] == "up_blocks";
            let i: usize = parts[2].parse().unwrap_or(0);
            let stage = if up { n_up.saturating_sub(1 + i) } else { i };
            let kind = if up { "up" } else { "down" };
            match parts[3] {
                "resnets" => format!(
                    "{tower}.{kind}.{stage}.block.{}.{}",
                    parts[4],
                    leaf(&parts[5..])
                ),
                // `downsamplers.0.conv` / `upsamplers.0.conv`.
                _ => format!(
                    "{tower}.{kind}.{stage}.{}.{}",
                    if up { "upsample" } else { "downsample" },
                    leaf(&parts[5..])
                ),
            }
        }
        Some("mid_block") if parts.len() >= 4 => match (parts[2], parts[3]) {
            ("resnets", "0") => format!("{tower}.mid.block_1.{}", leaf(&parts[4..])),
            ("resnets", _) => format!("{tower}.mid.block_2.{}", leaf(&parts[4..])),
            _ if parts.get(4) == Some(&"to_out") => format!("{tower}.mid.attn_1.proj_out"),
            _ => format!("{tower}.mid.attn_1.{}", leaf(&parts[4..])),
        },
        _ => name.to_string(),
    }
}

impl Flux2Vae {
    pub fn load(path: &Path, backend: Arc<dyn Backend>, precision: VaePrecision) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        Self::from_safetensors(&file, backend, precision)
            .with_context(|| format!("reading the FLUX.2 VAE from {}", path.display()))
    }

    fn from_safetensors(
        file: &SafeTensors,
        backend: Arc<dyn Backend>,
        precision: VaePrecision,
    ) -> Result<Self> {
        let diffusers = file.has("decoder.up_blocks.0.resnets.0.conv1.weight");
        // The original names its up stages from the bottom: count them.
        let n_up = if diffusers {
            (0..)
                .take_while(|i| file.has(&format!("decoder.up_blocks.{i}.resnets.0.conv1.weight")))
                .count()
        } else {
            (0..)
                .take_while(|i| file.has(&format!("decoder.up.{i}.block.0.conv1.weight")))
                .count()
        };
        let name = |stem: &str| -> String {
            if diffusers {
                stem.to_string()
            } else {
                original_name(stem, n_up)
            }
        };
        let has = |stem: &str| file.has(&format!("{}.weight", name(stem)));
        let conv = |stem: &str, padding: Padding| load_conv(file, &name(stem), padding, precision);
        let norm = |stem: &str| -> Result<Norm> {
            let n = name(stem);
            Ok(Norm {
                weight: file.tensor(&format!("{n}.weight"))?.0,
                bias: file.tensor(&format!("{n}.bias"))?.0,
            })
        };
        let res = |p: &str| -> Result<ResBlock> {
            Ok(ResBlock {
                norm1: norm(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"), Padding::Same)?,
                norm2: norm(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"), Padding::Same)?,
                shortcut: if has(&format!("{p}.conv_shortcut")) {
                    Some(conv(&format!("{p}.conv_shortcut"), Padding::Same)?)
                } else {
                    None
                },
            })
        };
        let tower = |t: &str| -> Result<Tower> {
            let encoder = t == "encoder";
            let (blocks, resample, padding) = if encoder {
                ("down_blocks", "downsamplers", Padding::HalveDownRight)
            } else {
                ("up_blocks", "upsamplers", Padding::Same)
            };
            let mut stages = Vec::new();
            for i in 0.. {
                let p = format!("{t}.{blocks}.{i}");
                if !has(&format!("{p}.resnets.0.conv1")) {
                    break;
                }
                let resnets = (0..)
                    .map(|j| format!("{p}.resnets.{j}"))
                    .take_while(|r| has(&format!("{r}.conv1")))
                    .map(|r| res(&r))
                    .collect::<Result<Vec<_>>>()?;
                let resample_stem = format!("{p}.{resample}.0.conv");
                let resample = if has(&resample_stem) {
                    Some(conv(&resample_stem, padding)?)
                } else {
                    None
                };
                stages.push(Stage { resnets, resample });
            }
            ensure!(!stages.is_empty(), "no {t}.{blocks}");
            let a = format!("{t}.mid_block.attentions.0");
            Ok(Tower {
                conv_in: conv(&format!("{t}.conv_in"), Padding::Same)?,
                stages,
                mid: (
                    res(&format!("{t}.mid_block.resnets.0"))?,
                    AttnBlock {
                        norm: norm(&format!("{a}.group_norm"))?,
                        q: conv(&format!("{a}.to_q"), Padding::Same)?,
                        k: conv(&format!("{a}.to_k"), Padding::Same)?,
                        v: conv(&format!("{a}.to_v"), Padding::Same)?,
                        out: conv(&format!("{a}.to_out.0"), Padding::Same)?,
                    },
                    res(&format!("{t}.mid_block.resnets.1"))?,
                ),
                norm_out: norm(&format!("{t}.conv_norm_out"))?,
                conv_out: conv(&format!("{t}.conv_out"), Padding::Same)?,
            })
        };
        let encoder = tower("encoder")?;
        let decoder = tower("decoder")?;
        let quant = conv("quant_conv", Padding::Same)?;
        let post_quant = conv("post_quant_conv", Padding::Same)?;
        let bn_mean = file.tensor("bn.running_mean")?.0;
        let bn_var = file.tensor("bn.running_var")?.0;
        ensure!(
            decoder.conv_in.cin == Z_DIM && post_quant.cin == Z_DIM && quant.cout == 2 * Z_DIM,
            "the VAE's latent width is not {Z_DIM}"
        );
        ensure!(
            encoder.conv_in.cin == 3 && decoder.conv_out.cout == 3,
            "the VAE does not read and write RGB"
        );
        ensure!(
            bn_mean.len() == PATCHED && bn_var.len() == PATCHED,
            "the VAE's latent statistics are not {PATCHED} wide"
        );
        for t in [&encoder, &decoder] {
            for c in t.stages.iter().flat_map(|s| s.resnets.iter()) {
                ensure!(
                    c.conv1.cin.is_multiple_of(GROUPS) && c.conv2.cout.is_multiple_of(GROUPS),
                    "a residual block's channels do not split into {GROUPS} groups"
                );
            }
        }
        let bn_std = bn_var.iter().map(|v| (v + BATCH_NORM_EPS).sqrt()).collect();
        Ok(Self {
            backend,
            encoder,
            decoder,
            quant,
            post_quant,
            bn_mean,
            bn_std,
        })
    }

    /// Pixels from a latent: `[h/8 * w/8, 32]` in VAE space (unpatchified,
    /// already de-normalised) to `[h * w, 3]` RGB in `[-1, 1]`. Stops
    /// between layers once `cancel` is set.
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
        let t = &self.decoder;
        let x = self.conv(&self.post_quant, latent);
        let mut x = self.conv(&t.conv_in, &x);
        x = self.middle(t, &x);
        for stage in &t.stages {
            for block in &stage.resnets {
                if super::cancelled(cancel) {
                    bail!("cancelled");
                }
                x = self.res_block(block, &x);
            }
            if let Some(conv) = &stage.resample {
                let up = nearest_upsample(&x);
                x = self.conv(conv, &up);
            }
        }
        let mut out = self.head(t, &x);
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(out)
    }

    /// A latent from pixels: `[h * w, 3]` RGB in `[-1, 1]` to the
    /// posterior's `(mean, logvar)`, each `[h/8 * w/8, 32]`.
    pub fn encode_unless(
        &self,
        rgb: &Feature,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Feature, Feature)> {
        ensure!(rgb.channels == 3, "picture has {} channels", rgb.channels);
        ensure!(
            rgb.height.is_multiple_of(SPATIAL_COMPRESSION)
                && rgb.width.is_multiple_of(SPATIAL_COMPRESSION),
            "a picture to encode must be a multiple of {SPATIAL_COMPRESSION} pixels on each side"
        );
        let t = &self.encoder;
        let mut x = self.conv(&t.conv_in, rgb);
        for stage in &t.stages {
            for block in &stage.resnets {
                if super::cancelled(cancel) {
                    bail!("cancelled");
                }
                x = self.res_block(block, &x);
            }
            if let Some(conv) = &stage.resample {
                x = self.conv(conv, &x);
            }
        }
        x = self.middle(t, &x);
        let moments = self.conv(&self.quant, &self.head(t, &x));
        let n = moments.height * moments.width;
        let mut mean = Vec::with_capacity(n * Z_DIM);
        let mut logvar = Vec::with_capacity(n * Z_DIM);
        for row in moments.data.chunks(2 * Z_DIM) {
            mean.extend_from_slice(&row[..Z_DIM]);
            logvar.extend_from_slice(&row[Z_DIM..]);
        }
        Ok((
            Feature::new(moments.height, moments.width, Z_DIM, mean),
            Feature::new(moments.height, moments.width, Z_DIM, logvar),
        ))
    }

    fn middle(&self, t: &Tower, x: &Feature) -> Feature {
        let x = self.res_block(&t.mid.0, x);
        let x = self.attn_block(&t.mid.1, &x);
        self.res_block(&t.mid.2, &x)
    }

    fn head(&self, t: &Tower, x: &Feature) -> Feature {
        let mut h = group_norm(x, &t.norm_out);
        silu_inplace(&mut h.data);
        self.conv(&t.conv_out, &h)
    }

    fn res_block(&self, block: &ResBlock, x: &Feature) -> Feature {
        let mut h = group_norm(x, &block.norm1);
        silu_inplace(&mut h.data);
        let h = self.conv(&block.conv1, &h);
        let mut h = group_norm(&h, &block.norm2);
        silu_inplace(&mut h.data);
        let mut h = self.conv(&block.conv2, &h);
        match &block.shortcut {
            Some(shortcut) => {
                let s = self.conv(shortcut, x);
                tensor::add_inplace(&mut h.data, &s.data);
            }
            None => tensor::add_inplace(&mut h.data, &x.data),
        }
        h
    }

    /// Single-head self-attention over every pixel of the latent-resolution
    /// map, with its residual.
    fn attn_block(&self, block: &AttnBlock, x: &Feature) -> Feature {
        let normed = group_norm(x, &block.norm);
        let (c, n) = (x.channels, x.height * x.width);
        let q = self.conv(&block.q, &normed);
        let k = self.conv(&block.k, &normed);
        let v = self.conv(&block.v, &normed);
        drop(normed);
        let attended = joint_attention(&q.data, &k.data, &v.data, n, 1, c, 1.0 / (c as f32).sqrt());
        let mut out = self.conv(&block.out, &Feature::new(x.height, x.width, c, attended));
        tensor::add_inplace(&mut out.data, &x.data);
        out
    }

    fn conv(&self, conv: &Conv, x: &Feature) -> Feature {
        run_conv(&*self.backend, conv, x)
    }
}

/// GroupNorm of [`GROUPS`] groups over a channel-last map: each group's
/// channels normalised over every pixel, then the per-channel affine.
fn group_norm(x: &Feature, norm: &Norm) -> Feature {
    let c = x.channels;
    let per = c / GROUPS;
    let count = (x.height * x.width * per) as f64;
    // Two passes, in `f64`: the mean, then the variance about it.
    let sums = |f: &(dyn Fn(usize, f32) -> f64 + Sync)| -> Vec<f64> {
        x.data
            .par_chunks(c * 64)
            .map(|chunk| {
                let mut acc = vec![0.0f64; GROUPS];
                for px in chunk.chunks(c) {
                    for (g, group) in px.chunks(per).enumerate() {
                        acc[g] += group.iter().map(|&v| f(g, v)).sum::<f64>();
                    }
                }
                acc
            })
            .reduce(
                || vec![0.0f64; GROUPS],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    };
    let mean: Vec<f64> = sums(&|_, v| v as f64).iter().map(|s| s / count).collect();
    let var: Vec<f64> = sums(&|g, v| {
        let d = v as f64 - mean[g];
        d * d
    })
    .iter()
    .map(|s| s / count)
    .collect();
    // Per channel: `y = x * a + b`.
    let (a, b): (Vec<f32>, Vec<f32>) = (0..c)
        .map(|ch| {
            let g = ch / per;
            let inv = 1.0 / (var[g] + NORM_EPS as f64).sqrt();
            let a = inv * norm.weight[ch] as f64;
            (a as f32, (norm.bias[ch] as f64 - mean[g] * a) as f32)
        })
        .unzip();
    let mut out = vec![0.0f32; x.data.len()];
    out.par_chunks_mut(c)
        .zip(x.data.par_chunks(c))
        .for_each(|(o, row)| {
            for (((o, v), a), b) in o.iter_mut().zip(row).zip(&a).zip(&b) {
                *o = v * a + b;
            }
        });
    Feature::new(x.height, x.width, c, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_names_number_the_decoder_stages_from_the_bottom() {
        let cases = [
            (
                "decoder.up_blocks.0.resnets.2.conv1",
                "decoder.up.3.block.2.conv1",
            ),
            (
                "decoder.up_blocks.3.resnets.0.conv_shortcut",
                "decoder.up.0.block.0.nin_shortcut",
            ),
            (
                "decoder.up_blocks.1.upsamplers.0.conv",
                "decoder.up.2.upsample.conv",
            ),
            (
                "encoder.down_blocks.1.resnets.1.norm2",
                "encoder.down.1.block.1.norm2",
            ),
            (
                "encoder.down_blocks.2.downsamplers.0.conv",
                "encoder.down.2.downsample.conv",
            ),
            (
                "encoder.mid_block.resnets.0.conv2",
                "encoder.mid.block_1.conv2",
            ),
            (
                "decoder.mid_block.resnets.1.norm1",
                "decoder.mid.block_2.norm1",
            ),
            (
                "decoder.mid_block.attentions.0.group_norm",
                "decoder.mid.attn_1.norm",
            ),
            (
                "decoder.mid_block.attentions.0.to_k",
                "decoder.mid.attn_1.k",
            ),
            (
                "decoder.mid_block.attentions.0.to_out.0",
                "decoder.mid.attn_1.proj_out",
            ),
            ("encoder.conv_norm_out", "encoder.norm_out"),
            ("encoder.conv_out", "encoder.conv_out"),
            ("post_quant_conv", "post_quant_conv"),
        ];
        for (diffusers, original) in cases {
            assert_eq!(original_name(diffusers, 4), original, "{diffusers}");
        }
    }

    #[test]
    fn group_norm_normalises_each_group_over_every_pixel() {
        // Two pixels, 64 channels: group 0 (channels 0, 1) holds 1, 3 and
        // 5, 7 — mean 4, variance 5.
        let mut data = vec![0.0f32; 2 * 64];
        data[0] = 1.0;
        data[1] = 3.0;
        data[64] = 5.0;
        data[65] = 7.0;
        let norm = Norm {
            weight: vec![2.0; 64],
            bias: vec![0.5; 64],
        };
        let out = group_norm(&Feature::new(1, 2, 64, data), &norm);
        let want = |v: f32| (v - 4.0) / (5.0f32 + NORM_EPS).sqrt() * 2.0 + 0.5;
        assert!((out.data[0] - want(1.0)).abs() < 1e-5);
        assert!((out.data[65] - want(7.0)).abs() < 1e-5);
        // An all-zero group stays at the bias.
        assert!((out.data[10] - 0.5).abs() < 1e-5);
    }
}

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

//! LTX-2's latent spatial upscaler (`LatentUpsampler`,
//! `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`): a latent at
//! half the clip's width and height in, one at the full size out — the
//! hinge of the distilled two-stage recipe, which draws the clip small in
//! eight steps and refines it large in three.
//!
//! It works on *un*-normalized latents (the video VAE's statistics taken
//! off first and put back after): a 3-D convolution to 512 channels, four
//! residual blocks of 3-D convolutions and group norms, a per-frame 2-D
//! convolution to four times the channels and a pixel shuffle that doubles
//! the rows and columns, four more residual blocks, and a convolution back
//! to 128 channels. Its 3-D convolutions are zero-padded in time and space.

use super::vae::{Conv3, SpatialPad, TimePad, Volume};
use crate::engine::backend::Backend;
use crate::engine::image::safetensors::SafeTensors;
use crate::engine::image::vae::VaePrecision;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// `GroupNorm(32, C)`, `eps = 1e-5`, with its affine.
struct GroupNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    groups: usize,
}

impl GroupNorm {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        Ok(Self {
            weight: file.tensor(&format!("{name}.weight"))?.0,
            bias: file.tensor(&format!("{name}.bias"))?.0,
            groups: 32,
        })
    }

    fn forward(&self, x: &mut Volume) {
        let c = x.channels;
        let per = c / self.groups;
        // Statistics per group over every pixel of every frame.
        let stats: Vec<(f64, f64)> = (0..self.groups)
            .into_par_iter()
            .map(|g| {
                let (mut sum, mut sq) = (0f64, 0f64);
                for p in x.data.chunks(c) {
                    for &v in &p[g * per..(g + 1) * per] {
                        sum += v as f64;
                        sq += (v as f64) * (v as f64);
                    }
                }
                let n = (x.data.len() / c * per) as f64;
                let mean = sum / n;
                (mean, 1.0 / ((sq / n - mean * mean).max(0.0) + 1e-5).sqrt())
            })
            .collect();
        x.data.par_chunks_mut(c).for_each(|p| {
            for (i, v) in p.iter_mut().enumerate() {
                let (mean, inv) = stats[i / per];
                *v = ((*v as f64 - mean) * inv) as f32 * self.weight[i] + self.bias[i];
            }
        });
    }
}

fn silu(x: &mut Volume) {
    super::ops::silu(&mut x.data);
}

struct ResBlock {
    conv1: Conv3,
    norm1: GroupNorm,
    conv2: Conv3,
    norm2: GroupNorm,
}

pub struct LatentUpscaler {
    backend: Arc<dyn Backend>,
    initial_conv: Conv3,
    initial_norm: GroupNorm,
    res_blocks: Vec<ResBlock>,
    upsample: Conv3,
    post_blocks: Vec<ResBlock>,
    final_conv: Conv3,
}

impl LatentUpscaler {
    pub fn load(path: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        if let Some(raw) = file.metadata("config") {
            let config: serde_json::Value =
                serde_json::from_str(raw).context("parsing the upscaler config")?;
            ensure!(
                !config["rational_resampler"].as_bool().unwrap_or(false)
                    && !config["temporal_upsample"].as_bool().unwrap_or(false)
                    && config["dims"].as_u64().unwrap_or(3) == 3
                    && config["spatial_scale"].as_f64().unwrap_or(2.0) == 2.0,
                "the latent upscaler is not the 3-D, two-times spatial one this engine runs"
            );
        }
        // Full-precision weights: the upscaler is small and its output is
        // what stage 2 starts from.
        let conv = |name: &str| Conv3::load(&file, name, VaePrecision::F32);
        let blocks = |prefix: &str| -> Result<Vec<ResBlock>> {
            let mut out = Vec::new();
            while file.has(&format!("{prefix}.{}.conv1.weight", out.len())) {
                let p = format!("{prefix}.{}", out.len());
                out.push(ResBlock {
                    conv1: conv(&format!("{p}.conv1"))?,
                    norm1: GroupNorm::load(&file, &format!("{p}.norm1"))?,
                    conv2: conv(&format!("{p}.conv2"))?,
                    norm2: GroupNorm::load(&file, &format!("{p}.norm2"))?,
                });
            }
            Ok(out)
        };
        let upsample = if file.has("upsampler.0.weight") {
            conv("upsampler.0")?
        } else {
            bail!("the latent upscaler has no upsampler.0 convolution")
        };
        Ok(Self {
            initial_conv: conv("initial_conv")?,
            initial_norm: GroupNorm::load(&file, "initial_norm")?,
            res_blocks: blocks("res_blocks")?,
            upsample,
            post_blocks: blocks("post_upsample_res_blocks")?,
            final_conv: conv("final_conv")?,
            backend,
        })
    }

    fn res_block(&self, block: &ResBlock, x: Volume) -> Volume {
        let b = &*self.backend;
        let mut h = block
            .conv1
            .forward(b, &x, TimePad::Zeros, SpatialPad::Zeros);
        block.norm1.forward(&mut h);
        silu(&mut h);
        let mut h = block
            .conv2
            .forward(b, &h, TimePad::Zeros, SpatialPad::Zeros);
        block.norm2.forward(&mut h);
        h.data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .for_each(|(h, x)| *h = super::ops::silu_one(*h + x));
        h
    }

    /// The latent at twice the width and height. `latent` and the result
    /// are normalized latents (`[f][h][w][128]`); `(mean, std)` are the
    /// video VAE's statistics.
    pub fn upscale(
        &self,
        latent: &[f32],
        (frames, height, width): (usize, usize, usize),
        (mean, std): (&[f32], &[f32]),
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<f32>> {
        let b = &*self.backend;
        let c = mean.len();
        ensure!(
            latent.len() == frames * height * width * c,
            "the latent is not {frames} × {height} × {width} × {c}"
        );
        let mut data = latent.to_vec();
        data.par_chunks_mut(c).for_each(|p| {
            for ((v, s), m) in p.iter_mut().zip(std).zip(mean) {
                *v = *v * s + m;
            }
        });
        let x = Volume::new(frames, height, width, c, data);
        let mut x = self
            .initial_conv
            .forward(b, &x, TimePad::Zeros, SpatialPad::Zeros);
        self.initial_norm.forward(&mut x);
        silu(&mut x);
        for block in &self.res_blocks {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = self.res_block(block, x);
        }
        let y = self
            .upsample
            .forward(b, &x, TimePad::Zeros, SpatialPad::Zeros);
        let mut x = pixel_shuffle2(&y);
        for block in &self.post_blocks {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = self.res_block(block, x);
        }
        let mut x = self
            .final_conv
            .forward(b, &x, TimePad::Zeros, SpatialPad::Zeros);
        x.data.par_chunks_mut(c).for_each(|p| {
            for ((v, s), m) in p.iter_mut().zip(std).zip(mean) {
                *v = (*v - m) / s;
            }
        });
        Ok(x.data)
    }
}

/// `(c p1 p2) h w -> c (h p1) (w p2)` on every frame.
fn pixel_shuffle2(x: &Volume) -> Volume {
    let c = x.channels / 4;
    let (h, w) = (x.height * 2, x.width * 2);
    let mut out = vec![0.0f32; x.frames * h * w * c];
    out.par_chunks_mut(h * w * c)
        .enumerate()
        .for_each(|(t, frame)| {
            for y in 0..h {
                for xx in 0..w {
                    let src = ((t * x.height + y / 2) * x.width + xx / 2) * x.channels;
                    let (p1, p2) = (y % 2, xx % 2);
                    for ci in 0..c {
                        frame[(y * w + xx) * c + ci] = x.data[src + (ci * 2 + p1) * 2 + p2];
                    }
                }
            }
        });
    Volume::new(x.frames, h, w, c, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pixel_shuffle_puts_the_row_offset_first() {
        let x = Volume::new(1, 1, 1, 4, vec![0.0, 1.0, 2.0, 3.0]);
        let y = pixel_shuffle2(&x);
        assert_eq!(y.data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!((y.height, y.width, y.channels), (2, 2, 1));
    }
}

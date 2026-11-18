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

//! LTX-2's convolutional video VAE decoder (`ConvVideoDecoder`, read from
//! `ltx-2.5-video-vae-conv-bf16.safetensors`): 128 latent channels at an
//! eighth of the frame rate and a thirty-second of the width and height,
//! back to RGB frames.
//!
//! The decoder's shape is the file's own — `config.vae.decoder_blocks` in
//! the header's metadata, read in reverse: residual stages
//! (`res_x`, with the decode timestep's scale and shift on every block),
//! depth-to-space upsamplers that double time, space or both
//! (`compress_*`, with a residual of the input rearranged the same way), a
//! final pixel norm, the timestep's last shift and scale, and a convolution
//! to 48 channels that unpatchifies into 4 × 4 pixels of RGB.
//!
//! Every convolution is 3 × 3 × 3, padded in time by repeating the first and
//! last frame (the decoder is non-causal) and in space by reflection. The
//! temporal upsamplers drop the first frame they make, which is how
//! `F'` latent frames become `8 (F' − 1) + 1` pixel frames.
//!
//! Activations are channel-last volumes (`[frames][height][width][channels]`)
//! so a convolution is a matmul over unrolled neighbourhoods, as in
//! `engine::image::vae`, and the weights are requantized to per-row `int8`
//! where the CPU has the tile for it.

use super::ops::{silu, timestep_embedding};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::image::safetensors::SafeTensors;
use crate::engine::image::vae::{VaePrecision, conv_matrix, row_matrix};
use crate::engine::loader::QuantMatrix;
use anyhow::{Context, Result, anyhow, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// A channel-last video tensor.
#[derive(Clone)]
pub struct Volume {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Volume {
    pub fn new(
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
        data: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(data.len(), frames * height * width * channels);
        Self {
            frames,
            height,
            width,
            channels,
            data,
        }
    }

    fn pixels(&self) -> usize {
        self.frames * self.height * self.width
    }
}

/// How a convolution pads its spatial edges.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpatialPad {
    Zeros,
    Reflect,
}

/// A 3-D convolution of kernel 1 or 3 as a matmul: rows `[cout][kt][ky][kx][cin]`.
pub struct Conv3 {
    w: QuantMatrix,
    rowi8: Option<crate::engine::vecdot::RowI8>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    /// The spatial kernel.
    kernel: usize,
    /// The temporal kernel: 1 for a per-frame 2-D convolution.
    kt: usize,
    /// The unpadded row length (`kt · kernel² · cin`); the matrix's may be
    /// padded past it with zeros.
    taps: usize,
}

/// How a convolution pads time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimePad {
    /// Both padding frames are copies of the first, in front (`causal`).
    Causal,
    /// One copy of the first frame in front, one of the last behind.
    Replicate,
    /// Zero frames either side (`nn.Conv3d(padding=1)`).
    Zeros,
}

impl Conv3 {
    /// A `[cout, cin, kt, k, k]` kernel, or a `[cout, cin, k, k]` one run
    /// on every frame alone.
    pub fn load(file: &SafeTensors, name: &str, precision: VaePrecision) -> Result<Self> {
        let (w, shape) = file
            .tensor(&format!("{name}.weight"))
            .with_context(|| format!("loading {name}"))?;
        let shape = shape.to_vec();
        let (cout, cin, kt, kernel) = match shape.as_slice() {
            [cout, cin, kt, kh, kw] if kh == kw => (*cout, *cin, *kt, *kh),
            [cout, cin, kh, kw] if kh == kw => (*cout, *cin, 1, *kh),
            other => bail!("{name}: unexpected kernel shape {other:?}"),
        };
        let bias = match file.tensor(&format!("{name}.bias")) {
            Ok((b, _)) => b,
            Err(_) => vec![0.0; cout],
        };
        ensure!(bias.len() == cout, "{name}: bias has {} values", bias.len());
        let k = kernel;
        let taps = kt * k * k * cin;
        let mut rows = Vec::with_capacity(cout * taps);
        for o in 0..cout {
            for t in 0..kt {
                for ky in 0..k {
                    for kx in 0..k {
                        for i in 0..cin {
                            rows.push(w[(((o * cin + i) * kt + t) * k + ky) * k + kx]);
                        }
                    }
                }
            }
        }
        Ok(Self {
            rowi8: row_matrix(&rows, taps, cout, precision),
            w: conv_matrix(rows, taps, cout, precision),
            bias,
            cin,
            cout,
            kernel,
            kt,
            taps,
        })
    }

    /// The convolution over `x`, time padded as `time` says and space as
    /// `pad` does.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        x: &Volume,
        time: TimePad,
        pad: SpatialPad,
    ) -> Volume {
        debug_assert_eq!(x.channels, self.cin);
        let (f, h, w) = (x.frames, x.height, x.width);
        let k = self.kernel as isize;
        let kt_n = self.kt as isize;
        let half = (k - 1) / 2;
        let half_t = (kt_n - 1) / 2;
        let cols = self.w.in_dim;
        let cin = self.cin;
        let mut out = vec![0.0f32; f * h * w * self.cout];
        // The frame an output frame's tap `kt` reads, `None` on a zero pad.
        let frame_of = |t: usize, kt: isize| -> Option<usize> {
            let src = match time {
                TimePad::Causal => t as isize + kt - (kt_n - 1),
                TimePad::Replicate | TimePad::Zeros => t as isize + kt - half_t,
            };
            if time == TimePad::Zeros && (src < 0 || src >= f as isize) {
                return None;
            }
            Some(src.clamp(0, f as isize - 1) as usize)
        };
        let coord = |c: isize, n: usize| -> Option<usize> {
            if c >= 0 && c < n as isize {
                return Some(c as usize);
            }
            match pad {
                SpatialPad::Zeros => None,
                SpatialPad::Reflect => {
                    let r = if c < 0 { -c } else { 2 * (n as isize - 1) - c };
                    Some(r.clamp(0, n as isize - 1) as usize)
                }
            }
        };
        // Bands of output rows sized to keep the unrolled table near 64 MiB.
        let band_rows = ((64usize << 20) / (cols * w * 4).max(1)).clamp(1, h);
        let mut patches = Vec::new();
        for t in 0..f {
            let mut y0 = 0;
            while y0 < h {
                let y1 = (y0 + band_rows).min(h);
                let n = (y1 - y0) * w;
                patches.clear();
                patches.resize(n * cols, 0.0);
                patches
                    .par_chunks_mut(w * cols)
                    .enumerate()
                    .for_each(|(r, row_patches)| {
                        let oy = (y0 + r) as isize;
                        for ox in 0..w {
                            let patch = &mut row_patches[ox * cols..(ox + 1) * cols];
                            let mut at = 0;
                            for kt in 0..kt_n {
                                let src_t = frame_of(t, kt);
                                for ky in 0..k {
                                    let iy = coord(oy + ky - half, h);
                                    for kx in 0..k {
                                        let ix = coord(ox as isize + kx - half, w);
                                        if let (Some(src_t), Some(iy), Some(ix)) = (src_t, iy, ix) {
                                            let src = ((src_t * h + iy) * w + ix) * cin;
                                            patch[at..at + cin]
                                                .copy_from_slice(&x.data[src..src + cin]);
                                        }
                                        at += cin;
                                    }
                                }
                            }
                            debug_assert_eq!(at, self.taps);
                        }
                    });
                let y = match &self.rowi8 {
                    Some(rows) => crate::engine::vecdot::matmul_rowi8(&patches, n, rows),
                    None => backend
                        .matmul_batch(&[MatmulOp {
                            x: &patches,
                            n_tokens: n,
                            w: &self.w,
                        }])
                        .pop()
                        .expect("one op in, one result out"),
                };
                let base = ((t * h + y0) * w) * self.cout;
                out[base..base + n * self.cout]
                    .par_chunks_mut(self.cout)
                    .zip(y.par_chunks(self.cout))
                    .for_each(|(o, y)| {
                        for ((o, y), b) in o.iter_mut().zip(y).zip(&self.bias) {
                            *o = y + b;
                        }
                    });
                y0 = y1;
            }
        }
        Volume::new(f, h, w, self.cout, out)
    }
}

/// `x / sqrt(mean_c(x²) + eps)` at every pixel.
pub fn pixel_norm(x: &mut Volume, eps: f32) {
    let c = x.channels;
    x.data.par_chunks_mut(c).for_each(|p| {
        let ms = p.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let s = 1.0 / (ms + eps).sqrt();
        p.iter_mut().for_each(|v| *v *= s);
    });
}

/// `x * (1 + scale) + shift` per channel.
fn scale_shift(x: &mut Volume, shift: &[f32], scale: &[f32]) {
    let c = x.channels;
    x.data.par_chunks_mut(c).for_each(|p| {
        for ((v, sh), sc) in p.iter_mut().zip(shift).zip(scale) {
            *v = *v * (1.0 + sc) + sh;
        }
    });
}

/// GroupNorm with one group (the residual's `norm3`): every value of the
/// volume normalized together, then a per-channel affine.
fn group_norm_1(x: &Volume, weight: &[f32], bias: &[f32], eps: f32) -> Volume {
    let n = x.data.len() as f64;
    let mean = x.data.par_iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .data
        .par_iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    let c = x.channels;
    let mut out = x.clone();
    out.data.par_chunks_mut(c).for_each(|p| {
        for ((v, w), b) in p.iter_mut().zip(weight).zip(bias) {
            *v = ((*v as f64 - mean) * inv) as f32 * w + b;
        }
    });
    out
}

/// A small dense layer kept in `f32` (the decoder's timestep embedders).
struct Dense {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
}

impl Dense {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        let (w, shape) = file.tensor(&format!("{name}.weight"))?;
        let shape = shape.to_vec();
        ensure!(shape.len() == 2, "{name}: not a matrix");
        let (b, _) = file.tensor(&format!("{name}.bias"))?;
        Ok(Self {
            w,
            b,
            cin: shape[1],
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.w
            .par_chunks(self.cin)
            .zip(self.b.par_iter())
            .map(|(row, b)| row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect()
    }
}

/// `PixArtAlphaCombinedTimestepSizeEmbeddings`: the sinusoid through
/// `linear_1`, SiLU, `linear_2`.
struct TimeEmbedder {
    linear_1: Dense,
    linear_2: Dense,
}

impl TimeEmbedder {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        Ok(Self {
            linear_1: Dense::load(file, &format!("{name}.timestep_embedder.linear_1"))?,
            linear_2: Dense::load(file, &format!("{name}.timestep_embedder.linear_2"))?,
        })
    }

    fn forward(&self, t: f32) -> Vec<f32> {
        let mut h = self.linear_1.forward(&timestep_embedding(t));
        silu(&mut h);
        self.linear_2.forward(&h)
    }
}

struct ResBlock {
    conv1: Conv3,
    conv2: Conv3,
    /// `scale_shift_table` (`[4, cin]`: shift1, scale1, shift2, scale2)
    /// when the block is timestep-conditioned.
    table: Option<Vec<f32>>,
    /// `per_channel_scale{1,2}` when the block injects noise.
    noise: Option<(Vec<f32>, Vec<f32>)>,
    /// `norm3` and `conv_shortcut` when the block changes width.
    shortcut: Option<(Vec<f32>, Vec<f32>, Conv3)>,
}

enum Stage {
    /// `res_x`: residual blocks behind one timestep embedder.
    Mid {
        time: Option<TimeEmbedder>,
        blocks: Vec<ResBlock>,
    },
    /// `res_x_y`: one block that narrows the width.
    Res(Box<ResBlock>),
    /// `compress_*`: depth-to-space by `stride`.
    Up {
        conv: Box<Conv3>,
        stride: [usize; 3],
        residual: bool,
        reduction: usize,
    },
}

/// The decoder's hyperparameters, from `config.vae` in the file's metadata.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub patch_size: usize,
    pub causal: bool,
    pub timestep_conditioning: bool,
    pub pad: SpatialPad,
    /// `[(name, params)]` as the file lists them (encoder order).
    pub blocks: Vec<(String, serde_json::Value)>,
    pub latent_channels: usize,
}

impl VaeConfig {
    fn from(file: &SafeTensors) -> Result<Self> {
        let raw = file
            .metadata("config")
            .ok_or_else(|| anyhow!("the video VAE has no `config` in its metadata"))?;
        let json: serde_json::Value =
            serde_json::from_str(raw).context("parsing the VAE config")?;
        let vae = json.get("vae").unwrap_or(&json);
        if let Some(class) = vae["_class_name"].as_str() {
            ensure!(
                !class.contains("Diffusion"),
                "the video VAE is LTX-2.5's diffusion decoder ({class}); serve the convolutional \
                 one (ltx-2.5-video-vae-conv-bf16.safetensors)"
            );
        }
        let norm = vae["norm_layer"].as_str().unwrap_or("pixel_norm");
        ensure!(
            norm == "pixel_norm",
            "the video VAE uses {norm}, not pixel_norm"
        );
        let blocks = vae["decoder_blocks"]
            .as_array()
            .ok_or_else(|| anyhow!("the VAE config lists no decoder_blocks"))?
            .iter()
            .map(|b| {
                let name = b[0].as_str().unwrap_or_default().to_string();
                let params = match &b[1] {
                    serde_json::Value::Number(n) => serde_json::json!({ "num_layers": n }),
                    other => other.clone(),
                };
                (name, params)
            })
            .collect();
        Ok(Self {
            patch_size: vae["patch_size"].as_u64().unwrap_or(4) as usize,
            causal: vae["causal_decoder"].as_bool().unwrap_or(false),
            timestep_conditioning: vae["timestep_conditioning"].as_bool().unwrap_or(true),
            pad: match vae["spatial_padding_mode"].as_str().unwrap_or("reflect") {
                "zeros" => SpatialPad::Zeros,
                _ => SpatialPad::Reflect,
            },
            blocks,
            latent_channels: vae["latent_channels"].as_u64().unwrap_or(128) as usize,
        })
    }
}

pub struct VideoDecoder {
    backend: Arc<dyn Backend>,
    pub config: VaeConfig,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv3,
    stages: Vec<Stage>,
    conv_out: Conv3,
    last_time: Option<(TimeEmbedder, Vec<f32>)>,
    /// `timestep_scale_multiplier`.
    timestep_scale: f32,
}

/// The decode timestep and the noise mixed into the latent first
/// (`decode_timestep`, `decode_noise_scale`).
const DECODE_TIMESTEP: f32 = 0.05;
const DECODE_NOISE: f32 = 0.025;

impl VideoDecoder {
    pub fn load(path: &Path, backend: Arc<dyn Backend>, precision: VaePrecision) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let config = VaeConfig::from(&file)?;
        let prefix = if file.has("vae.decoder.conv_in.conv.weight") {
            "vae.decoder."
        } else {
            "decoder."
        };
        let stats = ["per_channel_statistics.", "vae.per_channel_statistics."]
            .into_iter()
            .find(|p| file.has(&format!("{p}mean-of-means")))
            .ok_or_else(|| anyhow!("the video VAE has no latent statistics"))?;
        let (mean, _) = file.tensor(&format!("{stats}mean-of-means"))?;
        let (std, _) = file.tensor(&format!("{stats}std-of-means"))?;
        ensure!(
            mean.len() == config.latent_channels && std.len() == config.latent_channels,
            "the video VAE's statistics are not {} channels",
            config.latent_channels
        );
        let conv = |name: &str| Conv3::load(&file, &format!("{prefix}{name}"), precision);
        let tcond = config.timestep_conditioning;
        let res_block = |p: &str| -> Result<ResBlock> {
            let table = file
                .has(&format!("{prefix}{p}.scale_shift_table"))
                .then(|| {
                    file.tensor(&format!("{prefix}{p}.scale_shift_table"))
                        .map(|t| t.0)
                })
                .transpose()?;
            let noise = if file.has(&format!("{prefix}{p}.per_channel_scale1")) {
                Some((
                    file.tensor(&format!("{prefix}{p}.per_channel_scale1"))?.0,
                    file.tensor(&format!("{prefix}{p}.per_channel_scale2"))?.0,
                ))
            } else {
                None
            };
            let shortcut = if file.has(&format!("{prefix}{p}.conv_shortcut.weight")) {
                Some((
                    file.tensor(&format!("{prefix}{p}.norm3.weight"))?.0,
                    file.tensor(&format!("{prefix}{p}.norm3.bias"))?.0,
                    conv(&format!("{p}.conv_shortcut"))?,
                ))
            } else if file.has(&format!("{prefix}{p}.conv_shortcut.conv.weight")) {
                Some((
                    file.tensor(&format!("{prefix}{p}.norm3.weight"))?.0,
                    file.tensor(&format!("{prefix}{p}.norm3.bias"))?.0,
                    conv(&format!("{p}.conv_shortcut.conv"))?,
                ))
            } else {
                None
            };
            Ok(ResBlock {
                conv1: conv(&format!("{p}.conv1.conv"))?,
                conv2: conv(&format!("{p}.conv2.conv"))?,
                table,
                noise,
                shortcut,
            })
        };
        let mut stages = Vec::new();
        for (i, (name, params)) in config.blocks.iter().rev().enumerate() {
            let p = format!("up_blocks.{i}");
            let multiplier = params["multiplier"].as_u64();
            stages.push(match name.as_str() {
                "res_x" | "attn_res_x" => {
                    ensure!(
                        name == "res_x",
                        "the video VAE has an {name} stage, which this engine does not run"
                    );
                    let n = params["num_layers"].as_u64().unwrap_or(1) as usize;
                    let time = (tcond
                        && file.has(&format!(
                            "{prefix}{p}.time_embedder.timestep_embedder.linear_1.weight"
                        )))
                    .then(|| TimeEmbedder::load(&file, &format!("{prefix}{p}.time_embedder")))
                    .transpose()?;
                    Stage::Mid {
                        time,
                        blocks: (0..n)
                            .map(|j| res_block(&format!("{p}.res_blocks.{j}")))
                            .collect::<Result<_>>()?,
                    }
                }
                "res_x_y" => Stage::Res(Box::new(res_block(&p)?)),
                "compress_time" | "compress_space" | "compress_all" => Stage::Up {
                    conv: Box::new(conv(&format!("{p}.conv.conv"))?),
                    stride: match name.as_str() {
                        "compress_time" => [2, 1, 1],
                        "compress_space" => [1, 2, 2],
                        _ => [2, 2, 2],
                    },
                    residual: params["residual"].as_bool().unwrap_or(false),
                    reduction: multiplier.unwrap_or(1) as usize,
                },
                other => {
                    bail!("the video VAE has a '{other}' stage, which this engine does not run")
                }
            });
        }
        let last_time = (tcond
            && file.has(&format!(
                "{prefix}last_time_embedder.timestep_embedder.linear_1.weight"
            )))
        .then(|| -> Result<_> {
            Ok((
                TimeEmbedder::load(&file, &format!("{prefix}last_time_embedder"))?,
                file.tensor(&format!("{prefix}last_scale_shift_table"))?.0,
            ))
        })
        .transpose()?;
        let timestep_scale = file
            .tensor(&format!("{prefix}timestep_scale_multiplier"))
            .map(|t| t.0.first().copied().unwrap_or(1000.0))
            .unwrap_or(1000.0);
        Ok(Self {
            conv_in: conv("conv_in.conv")?,
            conv_out: conv("conv_out.conv")?,
            stages,
            last_time,
            timestep_scale,
            mean,
            std,
            config,
            backend,
        })
    }

    /// The latent space's per-channel `(mean, std)` — what the upscaler
    /// works outside of.
    pub fn latent_statistics(&self) -> (&[f32], &[f32]) {
        (&self.mean, &self.std)
    }

    fn time_pad(&self) -> TimePad {
        if self.config.causal {
            TimePad::Causal
        } else {
            TimePad::Replicate
        }
    }

    fn res_block(&self, block: &ResBlock, x: Volume, emb: Option<&[f32]>, rng: &mut Rng) -> Volume {
        let b = &*self.backend;
        let (time, pad) = (self.time_pad(), self.config.pad);
        let c = x.channels;
        let mut h = x.clone();
        pixel_norm(&mut h, 1e-8);
        let ada = match (&block.table, emb) {
            (Some(table), Some(emb)) => Some(
                table
                    .iter()
                    .zip(emb)
                    .map(|(t, e)| t + e)
                    .collect::<Vec<f32>>(),
            ),
            _ => None,
        };
        if let Some(a) = &ada {
            scale_shift(&mut h, &a[..c], &a[c..2 * c]);
        }
        silu(&mut h.data);
        let mut h = block.conv1.forward(b, &h, time, pad);
        if let Some((s1, _)) = &block.noise {
            add_spatial_noise(&mut h, s1, rng);
        }
        pixel_norm(&mut h, 1e-8);
        if let Some(a) = &ada {
            scale_shift(&mut h, &a[2 * c..3 * c], &a[3 * c..4 * c]);
        }
        silu(&mut h.data);
        let mut h = block.conv2.forward(b, &h, time, pad);
        if let Some((_, s2)) = &block.noise {
            add_spatial_noise(&mut h, s2, rng);
        }
        let skip = match &block.shortcut {
            Some((w, bias, conv)) => {
                let normed = group_norm_1(&x, w, bias, 1e-6);
                conv.forward(b, &normed, time, pad)
            }
            None => x,
        };
        h.data
            .par_iter_mut()
            .zip(skip.data.par_iter())
            .for_each(|(h, s)| *h += s);
        h
    }

    /// The decoded clip, `[frames][height][width][3]` in `[0, 1]`, from a
    /// latent laid out as the transformer's tokens: `[f][h][w][128]`.
    pub fn decode(
        &self,
        latent: &[f32],
        (frames, height, width): (usize, usize, usize),
        seed: u64,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Volume> {
        let b = &*self.backend;
        let ch = self.config.latent_channels;
        ensure!(
            latent.len() == frames * height * width * ch,
            "the latent is not {frames} × {height} × {width} × {ch}"
        );
        let mut rng = Rng::new(seed);
        let tcond = self.config.timestep_conditioning;
        let mut data = latent.to_vec();
        if tcond {
            data.iter_mut().for_each(|v| {
                *v = rng.normal() * DECODE_NOISE + (1.0 - DECODE_NOISE) * *v;
            });
        }
        data.par_chunks_mut(ch).for_each(|p| {
            for ((v, s), m) in p.iter_mut().zip(&self.std).zip(&self.mean) {
                *v = *v * s + m;
            }
        });
        let (time, pad) = (self.time_pad(), self.config.pad);
        let mut x =
            self.conv_in
                .forward(b, &Volume::new(frames, height, width, ch, data), time, pad);
        let t = DECODE_TIMESTEP * self.timestep_scale;
        for stage in &self.stages {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = match stage {
                Stage::Mid { time, blocks } => {
                    let emb = time.as_ref().map(|t_emb| t_emb.forward(t));
                    let mut x = x;
                    for block in blocks {
                        if super::cancelled(cancel) {
                            bail!("cancelled");
                        }
                        x = self.res_block(block, x, emb.as_deref(), &mut rng);
                    }
                    x
                }
                Stage::Res(block) => self.res_block(block, x, None, &mut rng),
                Stage::Up {
                    conv,
                    stride,
                    residual,
                    reduction,
                } => {
                    let skip = residual.then(|| {
                        let small = depth_to_space(&x, *stride);
                        let repeat = stride.iter().product::<usize>() / reduction;
                        tile_channels(&small, repeat)
                    });
                    let y = conv.forward(b, &x, time, pad);
                    let mut y = depth_to_space(&y, *stride);
                    if let Some(skip) = skip {
                        y.data
                            .par_iter_mut()
                            .zip(skip.data.par_iter())
                            .for_each(|(y, s)| *y += s);
                    }
                    y
                }
            };
        }
        pixel_norm(&mut x, 1e-8);
        if let Some((embedder, table)) = &self.last_time {
            let emb = embedder.forward(t);
            let c = x.channels;
            let ada: Vec<f32> = table.iter().zip(&emb).map(|(t, e)| t + e).collect();
            scale_shift(&mut x, &ada[..c], &ada[c..2 * c]);
        }
        silu(&mut x.data);
        let x = self.conv_out.forward(b, &x, time, pad);
        let p = self.config.patch_size;
        let out_c = x.channels / (p * p);
        ensure!(out_c == 3, "the video VAE writes {out_c} channels, not RGB");
        Ok(unpatchify(&x, p))
    }
}

/// `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, dropping the first
/// frame when time is doubled.
fn depth_to_space(x: &Volume, stride: [usize; 3]) -> Volume {
    let [st, sh, sw] = stride;
    let group = st * sh * sw;
    let c = x.channels / group;
    let (f, h, w) = (x.frames * st, x.height * sh, x.width * sw);
    let skip = usize::from(st == 2);
    let out_f = f - skip;
    let mut out = vec![0.0f32; out_f * h * w * c];
    out.par_chunks_mut(h * w * c)
        .enumerate()
        .for_each(|(of, frame)| {
            let t = of + skip;
            let (it, pt) = (t / st, t % st);
            for y in 0..h {
                let (iy, py) = (y / sh, y % sh);
                for xx in 0..w {
                    let (ix, px) = (xx / sw, xx % sw);
                    let src = ((it * x.height + iy) * x.width + ix) * x.channels;
                    let dst = (y * w + xx) * c;
                    for ci in 0..c {
                        let channel = ((ci * st + pt) * sh + py) * sw + px;
                        frame[dst + ci] = x.data[src + channel];
                    }
                }
            }
        });
    Volume::new(out_f, h, w, c, out)
}

/// `x.repeat(1, n, 1, 1, 1)`: the channels laid out `n` times over.
fn tile_channels(x: &Volume, n: usize) -> Volume {
    let c = x.channels;
    let mut out = vec![0.0f32; x.pixels() * c * n];
    out.par_chunks_mut(c * n)
        .zip(x.data.par_chunks(c))
        .for_each(|(o, p)| {
            for k in 0..n {
                o[k * c..(k + 1) * c].copy_from_slice(p);
            }
        });
    Volume::new(x.frames, x.height, x.width, c * n, out)
}

/// The decoder's last step: `b (c p r q) f h w -> b c f (h q) (w r)` with
/// `p = 1`, `q` the row and `r` the column within each `patch × patch` —
/// so the column is the *slower* index of a pixel's channels — mapped to
/// `[0, 1]` RGB.
fn unpatchify(x: &Volume, patch: usize) -> Volume {
    let c = x.channels / (patch * patch);
    let (h, w) = (x.height * patch, x.width * patch);
    let mut out = vec![0.0f32; x.frames * h * w * c];
    out.par_chunks_mut(h * w * c)
        .enumerate()
        .for_each(|(t, frame)| {
            for y in 0..h {
                let (iy, q) = (y / patch, y % patch);
                for xx in 0..w {
                    let (ix, r) = (xx / patch, xx % patch);
                    let src = ((t * x.height + iy) * x.width + ix) * x.channels;
                    for ci in 0..c {
                        let channel = (ci * patch + r) * patch + q;
                        let v = x.data[src + channel];
                        frame[(y * w + xx) * c + ci] = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                    }
                }
            }
        });
    Volume::new(x.frames, h, w, c, out)
}

/// `randn(H, W) · per_channel_scale[c]` added to every frame — the same
/// spatial noise for every frame and channel, scaled per channel.
fn add_spatial_noise(x: &mut Volume, scale: &[f32], rng: &mut Rng) {
    let (h, w, c) = (x.height, x.width, x.channels);
    let noise: Vec<f32> = (0..h * w).map(|_| rng.normal()).collect();
    x.data.par_chunks_mut(h * w * c).for_each(|frame| {
        for (p, n) in frame.chunks_mut(c).zip(&noise) {
            for (v, s) in p.iter_mut().zip(scale) {
                *v += n * s;
            }
        }
    });
}

/// A small seeded Gaussian source (SplitMix64 and Box–Muller).
pub struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((r * a.sin()) as f32);
        (r * a.cos()) as f32
    }

    pub fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_to_space_spreads_channels_over_time_and_space() {
        // One pixel of 8 channels, stride 2 everywhere: 2 frames (the first
        // dropped) of 2 x 2 pixels of 1 channel.
        let x = Volume::new(1, 1, 1, 8, (0..8).map(|v| v as f32).collect());
        let y = depth_to_space(&x, [2, 2, 2]);
        assert_eq!((y.frames, y.height, y.width, y.channels), (1, 2, 2, 1));
        // Frame index 1 → channels (1, py, px) = 4 + 2·py + px.
        assert_eq!(y.data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn unpatchify_puts_the_column_offset_on_the_slower_channel_index() {
        // patch 2, one RGB-less channel: c=1, channel = r·2 + q.
        let x = Volume::new(1, 1, 1, 4, vec![-1.0, -0.5, 0.0, 1.0]);
        let y = unpatchify(&x, 2);
        // (y=0,x=0): r=0,q=0 → ch0; (y=0,x=1): r=1,q=0 → ch2;
        // (y=1,x=0): r=0,q=1 → ch1; (y=1,x=1): r=1,q=1 → ch3.
        assert_eq!(y.data, vec![0.0, 0.5, 0.25, 1.0]);
    }

    #[test]
    fn tiling_repeats_the_whole_channel_block() {
        let x = Volume::new(1, 1, 1, 2, vec![1.0, 2.0]);
        assert_eq!(tile_channels(&x, 2).data, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn the_normal_source_is_roughly_standard() {
        let mut rng = Rng::new(7);
        let v = rng.fill(20000);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05);
    }
}

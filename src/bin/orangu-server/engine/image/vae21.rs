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

//! The Qwen-Image 2.1 VAE — diffusers' `AutoencoderKLQwenImage21`, read from
//! `qwen_image_2.1_vae_bf16.safetensors` in diffusers' own tensor naming
//! (`decoder.up_blocks.N.resnets.M.conv1`, `post_quant_conv`, ...).
//!
//! It is the Wan 2.2 autoencoder made for pictures: **RGBA** in and out
//! (four channels — the model draws transparency), 64 latent channels at a
//! sixteenth of the picture's side, and *residual* resampling stages — every
//! down- and up-sampling block adds a parameter-free shortcut of its input,
//! averaged down (`AvgDown3D`) or duplicated up (`DupUp3D`) across
//! channels. The convolutions are already 2-D in the file (the checkpoint
//! folded the single frame away), so only the shortcuts still carry the
//! video model's time axis, and on one frame that is a fixed rule each:
//!
//! - `AvgDown3D` with a temporal factor pads the lone frame with an **empty
//!   frame in front** before averaging, so half of every group it averages
//!   is zero.
//! - `DupUp3D` with a temporal factor makes two frames and diffusers keeps
//!   only the second (`first_chunk`), so each output channel reads the
//!   input channel of the *second* temporal slot.
//!
//! The `time_conv`s are skipped on the first frame by diffusers' own cache
//! bookkeeping and are never loaded. Everything else — the convolution as
//! a matmul over unrolled neighbourhoods, Wan's RMS norm, the middle
//! attention — is [`super::vae`]'s.

use crate::engine::backend::Backend;
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::safetensors::SafeTensors;
use super::transformer::joint_attention;
use super::vae::{
    Conv, Feature, Padding, VaePrecision, load_conv, nearest_upsample, rms_norm, run_conv,
    silu_inplace,
};

/// `vae/config.json`'s `latents_mean` and `latents_std`. (`-0.5236` is a
/// channel's measured mean, not an approximation of π/6.)
#[allow(clippy::approx_constant)]
pub const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
    2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
    3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
    3.7662, 3.6746, 3.4563, 3.8161,
];

/// Latent channels.
pub const Z_DIM: usize = 64;
/// Pixels per latent cell along each axis.
pub const SPATIAL_COMPRESSION: usize = 16;
/// Picture channels: red, green, blue and alpha.
pub const CHANNELS: usize = 4;

/// diffusers' `QwenImage21ResidualBlock`: `conv_shortcut(x) + conv2(silu(
/// norm2(conv1(silu(norm1(x))))))`.
struct ResBlock {
    norm1: Vec<f32>,
    conv1: Conv,
    norm2: Vec<f32>,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct AttnBlock {
    norm: Vec<f32>,
    to_qkv: Conv,
    proj: Conv,
}

/// The parameter-free shortcut of a resampling block, on one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shortcut {
    cin: usize,
    cout: usize,
    /// Whether the stage halves (encoder) or doubles (decoder) the picture
    /// (`factor_s = 2`). The encoder's last stage does neither and still
    /// adds its shortcut, then a plain channel average.
    spatial: bool,
    /// Whether the video model resamples time here too (`factor_t = 2`).
    temporal: bool,
}

impl Shortcut {
    fn factors(self) -> (usize, usize) {
        (
            if self.temporal { 2 } else { 1 },
            if self.spatial { 2 } else { 1 },
        )
    }
}

/// One of the encoder's `down_blocks` or the decoder's `up_blocks`.
struct Stage {
    resnets: Vec<ResBlock>,
    /// The stride-2 convolution (encoder) or the nearest-2x upsample and
    /// convolution (decoder), when the stage changes resolution.
    resample: Option<Conv>,
    /// Every encoder stage has one; a decoder stage only when it resamples.
    shortcut: Option<Shortcut>,
}

struct Middle {
    first: ResBlock,
    attn: AttnBlock,
    second: ResBlock,
}

struct Tower {
    conv_in: Conv,
    stages: Vec<Stage>,
    middle: Middle,
    norm_out: Vec<f32>,
    conv_out: Conv,
}

pub struct QwenImage21Vae {
    backend: Arc<dyn Backend>,
    encoder: Tower,
    decoder: Tower,
    /// `quant_conv`: the 128-channel `(mean | logvar)` mixing after the
    /// encoder.
    quant: Conv,
    /// `post_quant_conv`: the 64-channel mixing before the decoder.
    post_quant: Conv,
}

impl QwenImage21Vae {
    pub fn load(path: &Path, backend: Arc<dyn Backend>, precision: VaePrecision) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        Self::from_safetensors(&file, backend, precision)
            .with_context(|| format!("reading the Qwen-Image 2.1 VAE from {}", path.display()))
    }

    fn from_safetensors(
        file: &SafeTensors,
        backend: Arc<dyn Backend>,
        precision: VaePrecision,
    ) -> Result<Self> {
        let conv = |name: &str, padding: Padding| load_conv(file, name, padding, precision);
        let gamma =
            |name: &str| -> Result<Vec<f32>> { Ok(file.tensor(&format!("{name}.gamma"))?.0) };
        let res = |p: &str| -> Result<ResBlock> {
            Ok(ResBlock {
                norm1: gamma(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"), Padding::Same)?,
                norm2: gamma(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"), Padding::Same)?,
                shortcut: if file.has(&format!("{p}.conv_shortcut.weight")) {
                    Some(conv(&format!("{p}.conv_shortcut"), Padding::Same)?)
                } else {
                    None
                },
            })
        };
        let middle = |p: &str| -> Result<Middle> {
            Ok(Middle {
                first: res(&format!("{p}.mid_block.resnets.0"))?,
                attn: AttnBlock {
                    norm: gamma(&format!("{p}.mid_block.attentions.0.norm"))?,
                    to_qkv: conv(&format!("{p}.mid_block.attentions.0.to_qkv"), Padding::Same)?,
                    proj: conv(&format!("{p}.mid_block.attentions.0.proj"), Padding::Same)?,
                },
                second: res(&format!("{p}.mid_block.resnets.1"))?,
            })
        };
        let tower = |name: &str| -> Result<Tower> {
            let encoder = name == "encoder";
            let (blocks, resample, padding) = if encoder {
                ("down_blocks", "downsampler", Padding::HalveDownRight)
            } else {
                ("up_blocks", "upsampler", Padding::Same)
            };
            let mut stages = Vec::new();
            for i in 0.. {
                let p = format!("{name}.{blocks}.{i}");
                if !file.has(&format!("{p}.resnets.0.conv1.weight")) {
                    break;
                }
                let resnets = (0..)
                    .map(|j| format!("{p}.resnets.{j}"))
                    .take_while(|r| file.has(&format!("{r}.conv1.weight")))
                    .map(|r| res(&r))
                    .collect::<Result<Vec<_>>>()?;
                let resample_name = format!("{p}.{resample}.resample.1");
                let resample = if file.has(&format!("{resample_name}.weight")) {
                    Some(conv(&resample_name, padding)?)
                } else {
                    None
                };
                let shortcut = (encoder || resample.is_some()).then(|| Shortcut {
                    cin: resnets[0].conv1.cin,
                    cout: resnets[resnets.len() - 1].conv2.cout,
                    spatial: resample.is_some(),
                    // The stages that resample time carry the `time_conv`
                    // this engine never runs; its presence is the flag.
                    temporal: file.has(&format!(
                        "{p}.{}.time_conv.weight",
                        resample_name_stem(encoder)
                    )),
                });
                stages.push(Stage {
                    resnets,
                    resample,
                    shortcut,
                });
            }
            ensure!(!stages.is_empty(), "no {name}.{blocks}");
            Ok(Tower {
                conv_in: conv(&format!("{name}.conv_in"), Padding::Same)?,
                stages,
                middle: middle(name)?,
                norm_out: gamma(&format!("{name}.norm_out"))?,
                conv_out: conv(&format!("{name}.conv_out"), Padding::Same)?,
            })
        };
        let encoder = tower("encoder")?;
        let decoder = tower("decoder")?;
        let quant = conv("quant_conv", Padding::Same)?;
        let post_quant = conv("post_quant_conv", Padding::Same)?;
        ensure!(
            decoder.conv_in.cin == Z_DIM && post_quant.cin == Z_DIM && quant.cout == 2 * Z_DIM,
            "the VAE's latent width is not {Z_DIM}"
        );
        ensure!(
            encoder.conv_in.cin == CHANNELS && decoder.conv_out.cout == CHANNELS,
            "the VAE does not read and write RGBA"
        );
        for stage in encoder.stages.iter().chain(&decoder.stages) {
            if let Some(s) = stage.shortcut {
                let (ft, fs) = s.factors();
                let factor = ft * fs * fs;
                ensure!(
                    (s.cin * factor).is_multiple_of(s.cout)
                        && (s.cout * factor).is_multiple_of(s.cin),
                    "a resampling shortcut maps {} channels to {}",
                    s.cin,
                    s.cout
                );
            }
        }
        Ok(Self {
            backend,
            encoder,
            decoder,
            quant,
            post_quant,
        })
    }

    /// Pixels from a latent: `[h/16 * w/16, 64]` in VAE space (already
    /// de-normalised) to `[h * w, 4]` RGBA in `[-1, 1]`. Stops between
    /// layers once `cancel` is set (or the server is shutting down).
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
        let tower = &self.decoder;
        let x = self.conv(&self.post_quant, latent);
        let mut x = self.conv(&tower.conv_in, &x);
        x = self.middle(&tower.middle, &x);
        for stage in &tower.stages {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            let input = x;
            let mut h = self.res_block(&stage.resnets[0], &input);
            for block in &stage.resnets[1..] {
                h = self.res_block(block, &h);
            }
            if let Some(conv) = &stage.resample {
                h = self.conv(conv, &nearest_upsample(&h));
            }
            if let Some(shortcut) = stage.shortcut {
                tensor::add_inplace(&mut h.data, &dup_up(&input, shortcut).data);
            }
            x = h;
        }
        let mut out = self.head(tower, &x);
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(out)
    }

    /// A latent from pixels: `[h * w, 4]` RGBA in `[-1, 1]` to the
    /// posterior's `(mean, logvar)`, each `[h/16 * w/16, 64]`, in VAE space.
    pub fn encode_unless(
        &self,
        rgba: &Feature,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Feature, Feature)> {
        ensure!(
            rgba.channels == CHANNELS,
            "picture has {} channels",
            rgba.channels
        );
        ensure!(
            rgba.height.is_multiple_of(SPATIAL_COMPRESSION)
                && rgba.width.is_multiple_of(SPATIAL_COMPRESSION),
            "a picture to encode must be a multiple of {SPATIAL_COMPRESSION} pixels on each side"
        );
        let tower = &self.encoder;
        let mut x = self.conv(&tower.conv_in, rgba);
        for stage in &tower.stages {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            let input = x;
            let mut h = self.res_block(&stage.resnets[0], &input);
            for block in &stage.resnets[1..] {
                h = self.res_block(block, &h);
            }
            if let Some(conv) = &stage.resample {
                h = self.conv(conv, &h);
            }
            if let Some(shortcut) = stage.shortcut {
                tensor::add_inplace(&mut h.data, &avg_down(&input, shortcut).data);
            }
            x = h;
        }
        x = self.middle(&tower.middle, &x);
        let moments = self.head(tower, &x);
        let moments = self.conv(&self.quant, &moments);
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

    fn head(&self, tower: &Tower, x: &Feature) -> Feature {
        let mut h = rms_norm(x, &tower.norm_out);
        silu_inplace(&mut h.data);
        self.conv(&tower.conv_out, &h)
    }

    fn middle(&self, middle: &Middle, x: &Feature) -> Feature {
        let x = self.res_block(&middle.first, x);
        let x = self.attn_block(&middle.attn, &x);
        self.res_block(&middle.second, &x)
    }

    fn res_block(&self, block: &ResBlock, x: &Feature) -> Feature {
        let mut h = rms_norm(x, &block.norm1);
        silu_inplace(&mut h.data);
        let h = self.conv(&block.conv1, &h);
        let mut h = rms_norm(&h, &block.norm2);
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
    /// middle — the same block as [`super::vae`]'s.
    fn attn_block(&self, block: &AttnBlock, x: &Feature) -> Feature {
        let normed = rms_norm(x, &block.norm);
        let qkv = self.conv(&block.to_qkv, &normed);
        let c = x.channels;
        let n = x.height * x.width;
        let mut q = Vec::with_capacity(n * c);
        let mut k = Vec::with_capacity(n * c);
        let mut v = Vec::with_capacity(n * c);
        for row in qkv.data.chunks(3 * c) {
            q.extend_from_slice(&row[..c]);
            k.extend_from_slice(&row[c..2 * c]);
            v.extend_from_slice(&row[2 * c..]);
        }
        let attended = joint_attention(&q, &k, &v, n, 1, c, 1.0 / (c as f32).sqrt());
        let mut out = self.conv(&block.proj, &Feature::new(x.height, x.width, c, attended));
        tensor::add_inplace(&mut out.data, &x.data);
        out
    }

    fn conv(&self, conv: &Conv, x: &Feature) -> Feature {
        run_conv(&*self.backend, conv, x)
    }
}

/// The `time_conv`'s parent module in each tower — the flag
/// [`Shortcut::temporal`] is read from.
fn resample_name_stem(encoder: bool) -> &'static str {
    if encoder { "downsampler" } else { "upsampler" }
}

/// diffusers' `QwenImage21AvgDown3D` on one frame: fold each `fs x fs`
/// block (and, when `temporal`, a leading empty frame) into channels — flat
/// index `((c·ft + t)·fs + dy)·fs + dx` — and average consecutive groups of
/// that flat vector down to `cout` channels.
fn avg_down(x: &Feature, s: Shortcut) -> Feature {
    debug_assert_eq!(x.channels, s.cin);
    let (ft, fs) = s.factors();
    let factor = ft * fs * fs;
    let group = s.cin * factor / s.cout;
    let (h, w) = (x.height / fs, x.width / fs);
    let mut out = vec![0.0f32; h * w * s.cout];
    let inv = 1.0 / group as f32;
    out.par_chunks_mut(w * s.cout)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..w {
                let px = &mut row[ox * s.cout..(ox + 1) * s.cout];
                for (o, slot) in px.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for f in o * group..(o + 1) * group {
                        let dx = f % fs;
                        let dy = (f / fs) % fs;
                        let t = (f / (fs * fs)) % ft;
                        let c = f / factor;
                        // The padded frame is the first; only the last is
                        // the picture.
                        if t == ft - 1 {
                            let (iy, ix) = (fs * oy + dy, fs * ox + dx);
                            sum += x.data[(iy * x.width + ix) * s.cin + c];
                        }
                    }
                    *slot = sum * inv;
                }
            }
        });
    Feature::new(h, w, s.cout, out)
}

/// diffusers' `QwenImage21DupUp3D` on the first (and only) frame: double
/// the picture by repeating each input channel `cout · factor / cin` times
/// and unfolding the result into `(channel, t, dy, dx)`, keeping the last
/// temporal slot — so output channel `o` at offset `(dy, dx)` is input
/// channel `(o·factor + (ft-1)·4 + dy·2 + dx) / repeats`.
fn dup_up(x: &Feature, s: Shortcut) -> Feature {
    debug_assert_eq!(x.channels, s.cin);
    debug_assert!(s.spatial, "a decoder shortcut always doubles the picture");
    let ft = if s.temporal { 2 } else { 1 };
    let factor = ft * 4;
    let repeats = s.cout * factor / s.cin;
    let (h, w) = (x.height * 2, x.width * 2);
    let mut out = vec![0.0f32; h * w * s.cout];
    out.par_chunks_mut(w * s.cout)
        .enumerate()
        .for_each(|(y, row)| {
            let (iy, dy) = (y / 2, y % 2);
            for xo in 0..w {
                let (ix, dx) = (xo / 2, xo % 2);
                let src = &x.data[(iy * x.width + ix) * s.cin..(iy * x.width + ix + 1) * s.cin];
                let px = &mut row[xo * s.cout..(xo + 1) * s.cout];
                for (o, slot) in px.iter_mut().enumerate() {
                    *slot = src[(o * factor + (ft - 1) * 4 + dy * 2 + dx) / repeats];
                }
            }
        });
    Feature::new(h, w, s.cout, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A direct transcription of diffusers' `AvgDown3D.forward` for one
    /// frame (`B = T = 1`): pad the time axis in front, view, permute, view,
    /// mean — index arithmetic spelled the way torch does it, to hold the
    /// fused loop in [`avg_down`] to.
    fn avg_down_reference(x: &Feature, s: Shortcut) -> Feature {
        let (ft, fs) = s.factors();
        let (c_in, h, w) = (s.cin, x.height, x.width);
        // [C, T=ft, H, W] with the picture in the last frame.
        let at = |c: usize, t: usize, y: usize, xx: usize| -> f32 {
            if t == ft - 1 {
                x.data[(y * w + xx) * c_in + c]
            } else {
                0.0
            }
        };
        // view(C, 1, ft, H/fs, fs, W/fs, fs).permute(C, ft, fs, fs, 1, H/fs, W/fs)
        let (oh, ow) = (h / fs, w / fs);
        let mut flat = Vec::new(); // [C*ft*fs*fs][oh][ow]
        for c in 0..c_in {
            for t in 0..ft {
                for dy in 0..fs {
                    for dx in 0..fs {
                        let mut plane = vec![0.0f32; oh * ow];
                        for y in 0..oh {
                            for xx in 0..ow {
                                plane[y * ow + xx] = at(c, t, fs * y + dy, fs * xx + dx);
                            }
                        }
                        flat.push(plane);
                    }
                }
            }
        }
        let group = flat.len() / s.cout;
        let mut out = vec![0.0f32; oh * ow * s.cout];
        for o in 0..s.cout {
            for p in 0..oh * ow {
                let mean: f32 =
                    (0..group).map(|g| flat[o * group + g][p]).sum::<f32>() / group as f32;
                out[p * s.cout + o] = mean;
            }
        }
        Feature::new(oh, ow, s.cout, out)
    }

    /// Likewise for `DupUp3D.forward(first_chunk=True)`.
    fn dup_up_reference(x: &Feature, s: Shortcut) -> Feature {
        let ft = if s.temporal { 2 } else { 1 };
        let repeats = s.cout * ft * 4 / s.cin;
        let (h, w) = (x.height, x.width);
        // repeat_interleave(repeats, dim=1) → channel j reads j / repeats;
        // view(cout, ft, 2, 2, T, H, W).permute(cout, T, ft, H, 2, W, 2),
        // then keep t >= ft - 1.
        let mut out = vec![0.0f32; 4 * h * w * s.cout];
        for o in 0..s.cout {
            let t = ft - 1;
            for dy in 0..2 {
                for dx in 0..2 {
                    let j = ((o * ft + t) * 2 + dy) * 2 + dx;
                    let c = j / repeats;
                    for y in 0..h {
                        for xx in 0..w {
                            let (oy, ox) = (2 * y + dy, 2 * xx + dx);
                            out[(oy * 2 * w + ox) * s.cout + o] = x.data[(y * w + xx) * s.cin + c];
                        }
                    }
                }
            }
        }
        Feature::new(2 * h, 2 * w, s.cout, out)
    }

    fn feature(h: usize, w: usize, c: usize) -> Feature {
        Feature::new(
            h,
            w,
            c,
            (0..h * w * c)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.1)
                .collect(),
        )
    }

    /// Every shortcut shape the checkpoint has: the encoder's `96→96`
    /// (spatial only), `96→192`, `192→384`, `384→768` (temporal) and
    /// `768→768` (neither), and the decoder's `1152→1152`, `1152→576`
    /// (temporal) and `576→288` (spatial), scaled down by 48 so the test is
    /// instant.
    #[test]
    fn the_resampling_shortcuts_match_diffusers_index_arithmetic() {
        let down = [
            (2, 2, true, false),
            (2, 4, true, true),
            (4, 8, true, true),
            (16, 16, false, false),
        ];
        for (cin, cout, spatial, temporal) in down {
            let s = Shortcut {
                cin,
                cout,
                spatial,
                temporal,
            };
            let x = feature(4, 6, cin);
            assert_eq!(
                avg_down(&x, s).data,
                avg_down_reference(&x, s).data,
                "{s:?}"
            );
        }
        // The last encoder stage's average of one is the identity.
        let x = feature(4, 6, 16);
        let same = Shortcut {
            cin: 16,
            cout: 16,
            spatial: false,
            temporal: false,
        };
        assert_eq!(avg_down(&x, same).data, x.data);
        let up = [(24, 24, true), (24, 12, true), (12, 6, false)];
        for (cin, cout, temporal) in up {
            let s = Shortcut {
                cin,
                cout,
                spatial: true,
                temporal,
            };
            let x = feature(3, 5, cin);
            assert_eq!(dup_up(&x, s).data, dup_up_reference(&x, s).data, "{s:?}");
        }
    }

    /// With a temporal factor the empty frame in front is averaged in: of a
    /// constant picture's four output channels, the groups that fall on the
    /// padding are zero and the others the picture.
    #[test]
    fn a_temporal_average_counts_the_empty_frame() {
        let x = Feature::new(2, 2, 2, vec![1.0; 8]);
        let spatial = avg_down(
            &x,
            Shortcut {
                cin: 2,
                cout: 2,
                spatial: true,
                temporal: false,
            },
        );
        assert_eq!(spatial.data, vec![1.0, 1.0]);
        let temporal = avg_down(
            &x,
            Shortcut {
                cin: 2,
                cout: 4,
                spatial: true,
                temporal: true,
            },
        );
        assert_eq!(temporal.data, vec![0.0, 1.0, 0.0, 1.0]);
    }
}

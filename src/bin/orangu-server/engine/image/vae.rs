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

//! The Qwen-Image VAE — diffusers' `AutoencoderKLQwenImage`, which is the
//! Wan 2.1 video autoencoder: 16 latent channels at an eighth of the
//! picture's resolution, read from the `qwen_image_vae.safetensors` that
//! ComfyUI and stable-diffusion.cpp use, in the original Wan tensor
//! naming (`decoder.upsamples.N.residual.M`, `conv1`/`conv2` for the
//! quant convolutions).
//!
//! It is a *video* autoencoder, and that is the one thing worth knowing
//! before reading it: every convolution is a causal 3-D one, padded with two
//! empty frames in front, and the resampling stages carry a `time_conv`.
//! This engine only ever decodes (and encodes) a single picture — one
//! frame — and on one frame the causal padding makes every 3-D kernel act
//! through its **last** temporal slice alone, while the temporal
//! convolutions are skipped outright on the first frame (diffusers' own
//! `feat_cache` bookkeeping does exactly that). So what is built here is a
//! 2-D network: each `[out, in, 3, 3, 3]` kernel is read as its
//! `[.., .., 2, :, :]` slice, and the `time_conv` weights are never loaded.
//!
//! Activations are kept **channel-last** (`[height * width, channels]`),
//! which makes a 1x1 convolution a plain matmul, a 3x3 one a matmul over
//! unrolled 3x3 neighbourhoods, and the per-pixel RMS norm a row
//! operation. The matmuls go through `Backend::matmul` like everything else
//! in the engine; the unrolling runs in row bands so a 1024x1024 picture's
//! largest layer never needs its whole neighbourhood table at once.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::safetensors::SafeTensors;
use super::transformer::joint_attention;

/// `vae/config.json`'s `latents_mean` and `latents_std`: the per-channel
/// statistics the diffusion transformer's latent space is normalised by.
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

/// Latent channels.
pub const Z_DIM: usize = 16;
/// Pixels per latent cell along each axis.
pub const SPATIAL_COMPRESSION: usize = 8;

/// A channel-last feature map.
#[derive(Clone, Debug)]
pub struct Feature {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    /// `[height * width, channels]`.
    pub data: Vec<f32>,
}

impl Feature {
    pub fn new(height: usize, width: usize, channels: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), height * width * channels);
        Self {
            height,
            width,
            channels,
            data,
        }
    }
}

/// How a 3x3 convolution reads past its input's edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Padding {
    /// One zero on every side, stride one — the ordinary `padding=1` conv.
    Same,
    /// A zero on the right and bottom only, stride two — the encoder's
    /// `ZeroPad2d((0, 1, 0, 1))` followed by a stride-2 conv, which halves
    /// the picture.
    HalveDownRight,
}

pub(super) struct Conv {
    /// `[out, cols]`, each row laid out `(ky, kx, in)` so a tap's channel
    /// vector is one contiguous copy from a channel-last row; `cols` is
    /// `k * k * in` rounded up to the weight type's block (a multiple of
    /// 256 for the `int8` kernel), the padding zero on both sides.
    pub(super) w: QuantMatrix,
    /// The same rows as per-row `int8` for the 8 × 8 `smmla` tile
    /// ([`row_matrix`]), which a CPU with `i8mm` runs instead of `w`.
    pub(super) rowi8: Option<crate::engine::vecdot::RowI8>,
    pub(super) bias: Vec<f32>,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) kernel: usize,
    pub(super) padding: Padding,
}

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

// A few dozen layers per VAE: their size difference costs nothing.
#[allow(clippy::large_enum_variant)]
enum Layer {
    Res(ResBlock),
    Attn(AttnBlock),
    /// Nearest 2x upsample, then a 3x3 conv to half the channels.
    Upsample(Conv),
    /// The stride-2 conv that halves the picture.
    Downsample(Conv),
}

struct Tower {
    conv_in: Conv,
    layers: Vec<Layer>,
    head_norm: Vec<f32>,
    head_conv: Conv,
}

pub struct QwenImageVae {
    backend: Arc<dyn Backend>,
    encoder: Tower,
    decoder: Tower,
    /// `conv1`: the 32-channel `(mean | logvar)` mixing after the encoder.
    quant: Conv,
    /// `conv2`: the 16-channel mixing before the decoder.
    post_quant: Conv,
}

impl QwenImageVae {
    pub fn load(path: &Path, backend: Arc<dyn Backend>, precision: VaePrecision) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let vae = Self::from_safetensors(&file, backend, precision)
            .with_context(|| format!("reading the Qwen-Image VAE from {}", path.display()))?;
        Ok(vae)
    }

    fn from_safetensors(
        file: &SafeTensors,
        backend: Arc<dyn Backend>,
        precision: VaePrecision,
    ) -> Result<Self> {
        let conv = |name: &str, padding: Padding| load_conv(file, name, padding, precision);
        let gamma = |name: &str| -> Result<Vec<f32>> {
            let (g, _) = file.tensor(&format!("{name}.gamma"))?;
            Ok(g)
        };
        let res = |p: &str| -> Result<ResBlock> {
            Ok(ResBlock {
                norm1: gamma(&format!("{p}.residual.0"))?,
                conv1: conv(&format!("{p}.residual.2"), Padding::Same)?,
                norm2: gamma(&format!("{p}.residual.3"))?,
                conv2: conv(&format!("{p}.residual.6"), Padding::Same)?,
                shortcut: if file.has(&format!("{p}.shortcut.weight")) {
                    Some(conv(&format!("{p}.shortcut"), Padding::Same)?)
                } else {
                    None
                },
            })
        };
        let attn = |p: &str| -> Result<AttnBlock> {
            Ok(AttnBlock {
                norm: gamma(&format!("{p}.norm"))?,
                to_qkv: conv(&format!("{p}.to_qkv"), Padding::Same)?,
                proj: conv(&format!("{p}.proj"), Padding::Same)?,
            })
        };
        // The stack between `conv1` and `head`, in file order: residual
        // blocks, the one attention block in `middle`, and the resampling
        // stages, told apart by which tensors each index carries.
        let stack = |prefix: &str, resample: fn(Conv) -> Layer| -> Result<Vec<Layer>> {
            let mut layers = Vec::new();
            for i in 0.. {
                let p = format!("{prefix}.{i}");
                if file.has(&format!("{p}.residual.2.weight")) {
                    layers.push(Layer::Res(res(&p)?));
                } else if file.has(&format!("{p}.to_qkv.weight")) {
                    layers.push(Layer::Attn(attn(&p)?));
                } else if file.has(&format!("{p}.resample.1.weight")) {
                    let padding = if prefix.starts_with("encoder") {
                        Padding::HalveDownRight
                    } else {
                        Padding::Same
                    };
                    layers.push(resample(conv(&format!("{p}.resample.1"), padding)?));
                } else {
                    break;
                }
            }
            ensure!(!layers.is_empty(), "no layers under {prefix}");
            Ok(layers)
        };
        let tower = |name: &str, resample: fn(Conv) -> Layer| -> Result<Tower> {
            let mut layers = Vec::new();
            let stages = if name == "encoder" {
                "downsamples"
            } else {
                "upsamples"
            };
            // The encoder goes down first and through the middle after; the
            // decoder the other way round.
            if name == "encoder" {
                layers.extend(stack(&format!("{name}.{stages}"), resample)?);
                layers.extend(stack(&format!("{name}.middle"), resample)?);
            } else {
                layers.extend(stack(&format!("{name}.middle"), resample)?);
                layers.extend(stack(&format!("{name}.{stages}"), resample)?);
            }
            Ok(Tower {
                conv_in: conv(&format!("{name}.conv1"), Padding::Same)?,
                layers,
                head_norm: gamma(&format!("{name}.head.0"))?,
                head_conv: conv(&format!("{name}.head.2"), Padding::Same)?,
            })
        };
        let encoder = tower("encoder", Layer::Downsample)?;
        let decoder = tower("decoder", Layer::Upsample)?;
        let quant = conv("conv1", Padding::Same)?;
        let post_quant = conv("conv2", Padding::Same)?;
        ensure!(
            decoder.conv_in.cin == Z_DIM && post_quant.cin == Z_DIM && quant.cout == 2 * Z_DIM,
            "the VAE's latent width is not {Z_DIM}"
        );
        ensure!(
            encoder.conv_in.cin == 3 && decoder.head_conv.cout == 3,
            "the VAE does not read and write RGB"
        );
        Ok(Self {
            backend,
            encoder,
            decoder,
            quant,
            post_quant,
        })
    }

    /// Pixels from a latent: `[h/8 * w/8, 16]` in VAE space (already
    /// de-normalised) to `[h * w, 3]` RGB in `[-1, 1]`, stopping between
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
        let x = self.conv(&self.post_quant, latent);
        let mut out = self.run_tower(&self.decoder, x, cancel)?;
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(out)
    }

    /// A latent from pixels: `[h * w, 3]` RGB in `[-1, 1]` to the
    /// posterior's `(mean, logvar)`, each `[h/8 * w/8, 16]`, in VAE space.
    /// Stops between layers once `cancel` is set (or the server is shutting
    /// down).
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
        let moments = self.run_tower(&self.encoder, rgb.clone(), cancel)?;
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

    fn run_tower(
        &self,
        tower: &Tower,
        input: Feature,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Feature> {
        let mut x = self.conv(&tower.conv_in, &input);
        drop(input);
        for layer in &tower.layers {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = match layer {
                Layer::Res(block) => self.res_block(block, &x),
                Layer::Attn(block) => self.attn_block(block, &x),
                Layer::Upsample(conv) => {
                    let up = nearest_upsample(&x);
                    self.conv(conv, &up)
                }
                Layer::Downsample(conv) => self.conv(conv, &x),
            };
        }
        let mut h = rms_norm(&x, &tower.head_norm);
        silu_inplace(&mut h.data);
        Ok(self.conv(&tower.head_conv, &h))
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

    /// Single-head self-attention over every pixel, at the latent
    /// resolution only (`attn_scales` is empty, so `middle` is the one place
    /// it appears).
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

    /// One convolution, as a matmul over unrolled neighbourhoods.
    fn conv(&self, conv: &Conv, x: &Feature) -> Feature {
        run_conv(&*self.backend, conv, x)
    }
}

/// A convolution from `file`: `{name}.weight` and `{name}.bias`, a 2-D
/// kernel as it is or a causal 3-D one read through its last temporal
/// slice (see the module doc), laid out for [`run_conv`].
pub(super) fn load_conv(
    file: &SafeTensors,
    name: &str,
    padding: Padding,
    precision: VaePrecision,
) -> Result<Conv> {
    let (w, shape) = file.tensor(&format!("{name}.weight"))?;
    let (bias, _) = file.tensor(&format!("{name}.bias"))?;
    let (cout, cin, kernel, rows) = match shape {
        // A causal 3-D kernel: keep the last temporal slice only.
        [cout, cin, kt, kh, kw] => {
            ensure!(kh == kw, "{name}: kernel is {kh}x{kw}, not square");
            let per_slice = kh * kw;
            let mut rows = Vec::with_capacity(cout * per_slice * cin);
            for o in 0..*cout {
                for ky in 0..*kh {
                    for kx in 0..*kw {
                        for i in 0..*cin {
                            let idx = (((o * cin + i) * kt + (kt - 1)) * kh + ky) * kw + kx;
                            rows.push(w[idx]);
                        }
                    }
                }
            }
            (*cout, *cin, *kh, rows)
        }
        [cout, cin, kh, kw] => {
            ensure!(kh == kw, "{name}: kernel is {kh}x{kw}, not square");
            let mut rows = Vec::with_capacity(cout * kh * kw * cin);
            for o in 0..*cout {
                for ky in 0..*kh {
                    for kx in 0..*kw {
                        for i in 0..*cin {
                            rows.push(w[((o * cin + i) * kh + ky) * kw + kx]);
                        }
                    }
                }
            }
            (*cout, *cin, *kh, rows)
        }
        other => anyhow::bail!("{name}: unexpected kernel shape {other:?}"),
    };
    ensure!(
        bias.len() == cout,
        "{name}: bias has {} values for {cout} outputs",
        bias.len()
    );
    Ok(Conv {
        rowi8: row_matrix(&rows, kernel * kernel * cin, cout, precision),
        w: conv_matrix(rows, kernel * kernel * cin, cout, precision),
        bias,
        cin,
        cout,
        kernel,
        padding,
    })
}

/// One convolution, as a matmul over unrolled neighbourhoods, on `backend`.
pub(super) fn run_conv(backend: &dyn Backend, conv: &Conv, x: &Feature) -> Feature {
    debug_assert_eq!(x.channels, conv.cin);
    let (out_h, out_w) = match conv.padding {
        Padding::Same => (x.height, x.width),
        Padding::HalveDownRight => (x.height / 2, x.width / 2),
    };
    let cols = conv.w.in_dim;
    let mut out = vec![0.0f32; out_h * out_w * conv.cout];
    // Row bands, sized so the unrolled table stays around 64 MiB.
    let band_rows = ((64usize << 20) / (cols * out_w * 4).max(1)).clamp(1, out_h);
    // On the CPU with `int8` weights, the window of each pixel is
    // gathered straight into the quantizer's scratch row and the
    // `int8` kernel run on the result — the `f32` `im2col` band (9× the
    // feature map, 268 MiB at full resolution) never exists. Any other
    // backend, or `f32` weights, get the band through `matmul_batch`.
    let gathered = backend.is_cpu()
        && crate::engine::vecdot::have_i8mm()
        && (conv.rowi8.is_some() || crate::engine::vecdot::supports_k(conv.w.ggml_type(), cols));
    let mut patches = Vec::new();
    let mut y = Vec::new();
    let mut start = 0;
    while start < out_h {
        let end = (start + band_rows).min(out_h);
        let n = (end - start) * out_w;
        if gathered {
            let acts = crate::engine::vecdot::ActQ8Mm::quantize_with(cols, n, |t, row| {
                gather_window(x, conv, start + t / out_w, t % out_w, row);
            });
            if let Some(rows) = &conv.rowi8 {
                y = crate::engine::vecdot::matmul_rowi8_acts(&acts, n, rows);
            } else {
                crate::engine::backend::CpuBackend::matmul_k_mm_into(
                    &mut y,
                    &acts,
                    n,
                    conv.w.ggml_type(),
                    conv.w.raw_bytes(),
                    conv.w.row_bytes(),
                    cols,
                    conv.cout,
                );
            }
        } else {
            unroll(x, conv, start, end, out_w, &mut patches);
            y = backend
                .matmul_batch(&[MatmulOp {
                    x: &patches,
                    n_tokens: n,
                    w: &conv.w,
                }])
                .pop()
                .expect("one op in, one result out");
        }
        tensor::add_bias_per_row(&mut y, &conv.bias, n);
        out[start * out_w * conv.cout..end * out_w * conv.cout].copy_from_slice(&y);
        start = end;
    }
    Feature::new(out_h, out_w, conv.cout, out)
}

/// How the VAE's convolution weights are stored and multiplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VaePrecision {
    /// As the file has them, `f32`, through the tiled float matmul —
    /// exact.
    F32,
    /// Six-bit `Q6_K` weights against eight-bit activations, through the
    /// same `int8` kernel the transformer runs — about three times the
    /// rate on a core with `i8mm`. The rows are zero-padded to a multiple
    /// of 256 (a 3×3×96 tap row becomes 1024). The default: the picture
    /// it makes differs from the exact one by a level or two of a pixel.
    #[default]
    Int8,
}

impl VaePrecision {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "float" | "exact" => Some(Self::F32),
            "int8" | "q6_k" | "q6k" => Some(Self::Int8),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
        }
    }
}

/// `rows` (`[cout][cols]`, `f32`) as per-row `int8` for the 8 × 8 `smmla`
/// tile, padded as [`conv_matrix`] pads them — about four times the
/// `Q6_K` kernel's rate at the VAE's shapes (`doc/PERF-IMAGE.md`, task 12).
/// `None` under `F32`, on a CPU without `i8mm`, or with
/// `ORANGU_IMAGE_ROWI8_CONV=off` (an A/B against `Q6_K`).
pub(super) fn row_matrix(
    rows: &[f32],
    cols: usize,
    cout: usize,
    precision: VaePrecision,
) -> Option<crate::engine::vecdot::RowI8> {
    if precision != VaePrecision::Int8
        || !crate::engine::vecdot::have_i8mm()
        || std::env::var("ORANGU_IMAGE_ROWI8_CONV").is_ok_and(|v| v == "off")
    {
        return None;
    }
    let padded =
        cols.div_ceil(crate::engine::vecdot::SUPER_BLOCK) * crate::engine::vecdot::SUPER_BLOCK;
    Some(crate::engine::vecdot::RowI8::quantize(cout, padded, |o| {
        let mut row = vec![0f32; padded];
        row[..cols].copy_from_slice(&rows[o * cols..(o + 1) * cols]);
        row
    }))
}

/// The matrix a convolution runs as: `rows` is `[cout][cols]` in `f32`.
/// `Int8` pads each row to a multiple of 256 with zeros and encodes it as
/// `Q6_K` with the library's own GGUF encoder.
pub(super) fn conv_matrix(
    rows: Vec<f32>,
    cols: usize,
    cout: usize,
    precision: VaePrecision,
) -> QuantMatrix {
    match precision {
        VaePrecision::F32 => QuantMatrix::from_f32_rows(rows, cols, cout),
        VaePrecision::Int8 => {
            const BLOCK: usize = 256;
            let padded = cols.div_ceil(BLOCK) * BLOCK;
            let mut wide = vec![0.0f32; cout * padded];
            for (o, row) in rows.chunks_exact(cols).enumerate() {
                wide[o * padded..o * padded + cols].copy_from_slice(row);
            }
            let ggml_type = crate::engine::quant::GGML_TYPE_Q6_K;
            let bytes = orangu::quantize::encode(ggml_type, &wide, padded);
            QuantMatrix::from_encoded_rows(bytes, ggml_type, padded, cout, None)
        }
    }
}

/// One output pixel's `(ky, kx, in)` window into `row` — `unroll` for a
/// single pixel, taps off the edge and the padding past the taps zero.
fn gather_window(x: &Feature, conv: &Conv, oy: usize, ox: usize, row: &mut [f32]) {
    let k = conv.kernel;
    let cin = conv.cin;
    let (stride, offset) = match conv.padding {
        Padding::Same => (1isize, -((k as isize - 1) / 2)),
        Padding::HalveDownRight => (2isize, 0),
    };
    row.fill(0.0);
    for ky in 0..k {
        let iy = oy as isize * stride + offset + ky as isize;
        if iy < 0 || iy >= x.height as isize {
            continue;
        }
        for kx in 0..k {
            let ix = ox as isize * stride + offset + kx as isize;
            if ix < 0 || ix >= x.width as isize {
                continue;
            }
            let src = (iy as usize * x.width + ix as usize) * cin;
            let dst = (ky * k + kx) * cin;
            row[dst..dst + cin].copy_from_slice(&x.data[src..src + cin]);
        }
    }
}

/// Fills `patches` with the `(ky, kx, in)` neighbourhood of every output
/// pixel in rows `start..end`.
fn unroll(
    x: &Feature,
    conv: &Conv,
    start: usize,
    end: usize,
    out_w: usize,
    patches: &mut Vec<f32>,
) {
    let k = conv.kernel;
    let cin = conv.cin;
    // The matrix's row length, which may run past the taps: the padding
    // stays zero and multiplies zero weights.
    let cols = conv.w.in_dim;
    patches.clear();
    patches.resize((end - start) * out_w * cols, 0.0);
    let (stride, offset) = match conv.padding {
        Padding::Same => (1isize, -((k as isize - 1) / 2)),
        Padding::HalveDownRight => (2isize, 0),
    };
    patches
        .par_chunks_mut(out_w * cols)
        .enumerate()
        .for_each(|(band_row, row_patches)| {
            let oy = (start + band_row) as isize;
            for ox in 0..out_w {
                let patch = &mut row_patches[ox * cols..(ox + 1) * cols];
                for ky in 0..k {
                    let iy = oy * stride + offset + ky as isize;
                    if iy < 0 || iy >= x.height as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = ox as isize * stride + offset + kx as isize;
                        if ix < 0 || ix >= x.width as isize {
                            continue;
                        }
                        let src = (iy as usize * x.width + ix as usize) * cin;
                        let dst = (ky * k + kx) * cin;
                        patch[dst..dst + cin].copy_from_slice(&x.data[src..src + cin]);
                    }
                }
            }
        });
}

/// Wan's `RMS_norm`: each pixel's channel vector scaled to length
/// `sqrt(channels)`, times a per-channel `gamma`.
pub(super) fn rms_norm(x: &Feature, gamma: &[f32]) -> Feature {
    let c = x.channels;
    debug_assert_eq!(gamma.len(), c);
    let scale = (c as f32).sqrt();
    let mut out = vec![0.0f32; x.data.len()];
    out.par_chunks_mut(c)
        .zip(x.data.par_chunks(c))
        .for_each(|(o, row)| {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let s = scale / norm;
            for ((o, v), g) in o.iter_mut().zip(row).zip(gamma) {
                *o = v * s * g;
            }
        });
    Feature::new(x.height, x.width, c, out)
}

pub(super) fn silu_inplace(x: &mut [f32]) {
    x.par_chunks_mut(4096).for_each(|chunk| {
        for v in chunk.iter_mut() {
            *v = tensor::silu(*v);
        }
    });
}

/// `nearest-exact` at exactly 2x: every pixel becomes a 2x2 block.
pub(super) fn nearest_upsample(x: &Feature) -> Feature {
    let c = x.channels;
    let (h, w) = (x.height * 2, x.width * 2);
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        let src_row = &x.data[(y / 2) * x.width * c..(y / 2 + 1) * x.width * c];
        for ox in 0..w {
            row[ox * c..(ox + 1) * c].copy_from_slice(&src_row[(ox / 2) * c..(ox / 2 + 1) * c]);
        }
    });
    Feature::new(h, w, c, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;

    fn conv_with(rows: Vec<f32>, cin: usize, cout: usize, kernel: usize, padding: Padding) -> Conv {
        Conv {
            rowi8: None,
            w: QuantMatrix::from_f32_rows(rows, kernel * kernel * cin, cout),
            bias: vec![0.0; cout],
            cin,
            cout,
            kernel,
            padding,
        }
    }

    fn vae_shell() -> QwenImageVae {
        // Only `conv` is exercised; the towers are never run.
        let empty = |cin, cout| conv_with(vec![0.0; cin * cout], cin, cout, 1, Padding::Same);
        let tower = || Tower {
            conv_in: empty(1, 1),
            layers: Vec::new(),
            head_norm: vec![1.0],
            head_conv: empty(1, 1),
        };
        QwenImageVae {
            backend: Arc::new(CpuBackend),
            encoder: tower(),
            decoder: tower(),
            quant: empty(1, 1),
            post_quant: empty(1, 1),
        }
    }

    /// A 3x3 kernel that is `1` only at its centre tap is the identity, and
    /// one that is `1` only at the top-left tap shifts the picture down and
    /// right with zeros coming in — which pins both the tap order and the
    /// padding.
    #[test]
    fn a_same_convolution_reads_its_taps_in_ky_kx_order() {
        let vae = vae_shell();
        let x = Feature::new(2, 3, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut centre = vec![0.0; 9];
        centre[4] = 1.0;
        let y = vae.conv(&conv_with(centre, 1, 1, 3, Padding::Same), &x);
        assert_eq!(y.data, x.data);
        let mut top_left = vec![0.0; 9];
        top_left[0] = 1.0;
        let y = vae.conv(&conv_with(top_left, 1, 1, 3, Padding::Same), &x);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    /// A 1x1 convolution is a per-pixel matmul, with the bias added.
    #[test]
    fn a_pointwise_convolution_mixes_channels() {
        let vae = vae_shell();
        let x = Feature::new(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut conv = conv_with(vec![1.0, 1.0, 1.0, -1.0], 2, 2, 1, Padding::Same);
        conv.bias = vec![0.5, 0.0];
        let y = vae.conv(&conv, &x);
        assert_eq!(y.data, vec![3.5, -1.0, 7.5, -1.0]);
    }

    /// The encoder's halving conv pads one zero on the right and bottom and
    /// steps by two: output (0,0) reads input rows 0..3 and columns 0..3.
    #[test]
    fn the_halving_convolution_steps_by_two_from_the_top_left() {
        let vae = vae_shell();
        let x = Feature::new(4, 4, 1, (1..=16).map(|v| v as f32).collect());
        let mut top_left = vec![0.0; 9];
        top_left[0] = 1.0;
        let y = vae.conv(&conv_with(top_left, 1, 1, 3, Padding::HalveDownRight), &x);
        assert_eq!((y.height, y.width), (2, 2));
        assert_eq!(y.data, vec![1.0, 3.0, 9.0, 11.0]);
        // The bottom-right tap of the last output pixel is the padding.
        let mut bottom_right = vec![0.0; 9];
        bottom_right[8] = 1.0;
        let y = vae.conv(
            &conv_with(bottom_right, 1, 1, 3, Padding::HalveDownRight),
            &x,
        );
        assert_eq!(y.data, vec![11.0, 0.0, 0.0, 0.0]);
    }

    /// The `int8` convolution — `Q6_K` weights padded to 256, the window
    /// gathered straight into the quantizer — agrees with the exact `f32`
    /// one to the quantization's own error, on both padding modes and a
    /// channel count that is nothing like a multiple of anything.
    #[test]
    fn an_int8_convolution_matches_the_f32_one_within_quantization() {
        if !crate::engine::vecdot::have_i8mm() {
            eprintln!("no i8mm on this core; skipped");
            return;
        }
        let (cin, cout, k) = (5usize, 6usize, 3usize);
        let value = |i: usize| ((i * 7 % 13) as f32 - 6.0) * 0.1;
        let rows: Vec<f32> = (0..cout * k * k * cin).map(value).collect();
        let x = Feature::new(9, 7, cin, (0..9 * 7 * cin).map(|i| value(i + 3)).collect());
        let vae = vae_shell();
        for padding in [Padding::Same, Padding::HalveDownRight] {
            let exact = vae.conv(&conv_with(rows.clone(), cin, cout, k, padding), &x);
            // `Q6_K` on the K-quant kernel, and per-row `int8` on the
            // `smmla` tile.
            for rowi8 in [false, true] {
                let int8 = Conv {
                    rowi8: rowi8
                        .then(|| row_matrix(&rows, k * k * cin, cout, VaePrecision::Int8))
                        .flatten(),
                    w: conv_matrix(rows.clone(), k * k * cin, cout, VaePrecision::Int8),
                    bias: vec![0.0; cout],
                    cin,
                    cout,
                    kernel: k,
                    padding,
                };
                let fast = vae.conv(&int8, &x);
                assert_eq!(fast.data.len(), exact.data.len());
                let scale = exact.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                for (i, (f, e)) in fast.data.iter().zip(&exact.data).enumerate() {
                    assert!(
                        (f - e).abs() <= 0.02 * scale,
                        "{padding:?} rowi8 {rowi8} at {i}: {f} vs {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn nearest_upsample_doubles_every_pixel() {
        let x = Feature::new(1, 2, 1, vec![1.0, 2.0]);
        let y = nearest_upsample(&x);
        assert_eq!((y.height, y.width), (2, 4));
        assert_eq!(y.data, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn rms_norm_scales_each_pixel_to_root_channels() {
        let x = Feature::new(1, 1, 4, vec![3.0, 0.0, 4.0, 0.0]);
        let y = rms_norm(&x, &[1.0, 1.0, 1.0, 2.0]);
        // |x| = 5, scale sqrt(4)/5 = 0.4.
        assert_eq!(y.data, vec![1.2, 0.0, 1.6, 0.0]);
    }
}

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

//! LTX-2's sound: the audio VAE's decoder, which turns the transformer's
//! audio latent into a stereo mel spectrogram, and the vocoder that turns
//! the spectrogram into a waveform — both in
//! `ltx-2.5-audio-vae-bf16.safetensors`.
//!
//! **The decoder** is a 2-D convolutional network over (time, mel bin)
//! with convolutions causal in time: residual blocks with pixel norms,
//! an attention block in the middle, and nearest-neighbour upsamplers that
//! double both axes. Four latent frames make four spectrogram frames
//! (`4 A − 3`, the first latent frame covering one).
//!
//! **The vocoder** is BigVGAN v2: transposed convolutions that upsample the
//! spectrogram's frames to samples, each followed by anti-aliased
//! multi-periodicity residual blocks (SnakeBeta activations run at twice
//! the rate between a Kaiser-windowed sinc upsampler and downsampler).
//! From LTX-2.3 on it is followed by a **bandwidth extension**: the
//! waveform's own log-mel spectrogram (a causal STFT whose bases ship in
//! the file) through a second, smaller BigVGAN, added as a residual to the
//! waveform resampled to the output rate.
//!
//! The vocoder runs in `f32` throughout, as the reference does: in `bf16`
//! its hundred-odd chained convolutions lose most of the high end.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::image::safetensors::SafeTensors;
use crate::engine::image::vae::{VaePrecision, conv_matrix};
use crate::engine::loader::QuantMatrix;
use anyhow::{Context, Result, anyhow, bail, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Latent frames to spectrogram frames (`LATENT_DOWNSAMPLE_FACTOR`).
pub const LATENT_DOWNSAMPLE: usize = 4;

// ---------------------------------------------------------------------------
// The spectrogram decoder
// ---------------------------------------------------------------------------

/// A 2-D feature map `[rows = time][cols = mel][channels]`.
#[derive(Clone)]
struct Map {
    rows: usize,
    cols: usize,
    channels: usize,
    data: Vec<f32>,
}

/// A 2-D convolution, causal along rows (time): the kernel's `k − 1` rows
/// of padding all go above, its columns' padding is split — zeros either
/// way.
struct Conv2 {
    w: QuantMatrix,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
}

impl Conv2 {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        let (w, shape) = file
            .tensor(&format!("{name}.weight"))
            .with_context(|| format!("loading {name}"))?;
        let (cout, cin, k) = match shape {
            [cout, cin, kh, kw] if kh == kw => (*cout, *cin, *kh),
            other => bail!("{name}: unexpected kernel shape {other:?}"),
        };
        let (bias, _) = file.tensor(&format!("{name}.bias"))?;
        let mut rows = Vec::with_capacity(cout * k * k * cin);
        for o in 0..cout {
            for ky in 0..k {
                for kx in 0..k {
                    for i in 0..cin {
                        rows.push(w[((o * cin + i) * k + ky) * k + kx]);
                    }
                }
            }
        }
        Ok(Self {
            w: conv_matrix(rows, k * k * cin, cout, VaePrecision::F32),
            bias,
            cin,
            cout,
            k,
        })
    }

    fn forward(&self, backend: &dyn Backend, x: &Map) -> Map {
        debug_assert_eq!(x.channels, self.cin);
        let (h, w, k, cin) = (x.rows, x.cols, self.k as isize, self.cin);
        let cols = self.w.in_dim;
        let left = (k - 1) / 2;
        let n = h * w;
        let mut patches = vec![0.0f32; n * cols];
        patches
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(p, patch)| {
                let (oy, ox) = ((p / w) as isize, (p % w) as isize);
                for ky in 0..k {
                    let iy = oy + ky - (k - 1);
                    for kx in 0..k {
                        let ix = ox + kx - left;
                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            let src = (iy as usize * w + ix as usize) * cin;
                            let dst = ((ky * k + kx) as usize) * cin;
                            patch[dst..dst + cin].copy_from_slice(&x.data[src..src + cin]);
                        }
                    }
                }
            });
        let mut y = backend
            .matmul_batch(&[MatmulOp {
                x: &patches,
                n_tokens: n,
                w: &self.w,
            }])
            .pop()
            .expect("one op in, one result out");
        y.par_chunks_mut(self.cout).for_each(|row| {
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        });
        Map {
            rows: h,
            cols: w,
            channels: self.cout,
            data: y,
        }
    }
}

fn pixel_norm(x: &mut Map) {
    let c = x.channels;
    x.data.par_chunks_mut(c).for_each(|p| {
        let ms = p.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let s = 1.0 / (ms + 1e-6).sqrt();
        p.iter_mut().for_each(|v| *v *= s);
    });
}

fn silu(x: &mut Map) {
    super::ops::silu(&mut x.data);
}

struct ResBlock {
    conv1: Conv2,
    conv2: Conv2,
    shortcut: Option<Conv2>,
}

impl ResBlock {
    fn load(file: &SafeTensors, p: &str) -> Result<Self> {
        let shortcut = ["nin_shortcut", "conv_shortcut"]
            .into_iter()
            .find(|s| file.has(&format!("{p}.{s}.conv.weight")))
            .map(|s| Conv2::load(file, &format!("{p}.{s}.conv")))
            .transpose()?;
        Ok(Self {
            conv1: Conv2::load(file, &format!("{p}.conv1.conv"))?,
            conv2: Conv2::load(file, &format!("{p}.conv2.conv"))?,
            shortcut,
        })
    }

    fn forward(&self, b: &dyn Backend, x: Map) -> Map {
        let mut h = x.clone();
        pixel_norm(&mut h);
        silu(&mut h);
        let mut h = self.conv1.forward(b, &h);
        pixel_norm(&mut h);
        silu(&mut h);
        let mut h = self.conv2.forward(b, &h);
        let skip = match &self.shortcut {
            Some(conv) => conv.forward(b, &x),
            None => x,
        };
        h.data
            .par_iter_mut()
            .zip(skip.data.par_iter())
            .for_each(|(h, s)| *h += s);
        h
    }
}

/// The vanilla single-head attention block: pixel norm, 1 × 1 `q`, `k`,
/// `v`, softmax over every position, `proj_out`, residual.
struct AttnBlock {
    q: Conv2,
    k: Conv2,
    v: Conv2,
    proj: Conv2,
}

impl AttnBlock {
    fn load(file: &SafeTensors, p: &str) -> Result<Self> {
        let conv = |n: &str| -> Result<Conv2> {
            // Plain `Conv2d`s, not the causal wrapper.
            let name = if file.has(&format!("{p}.{n}.conv.weight")) {
                format!("{p}.{n}.conv")
            } else {
                format!("{p}.{n}")
            };
            Conv2::load(file, &name)
        };
        Ok(Self {
            q: conv("q")?,
            k: conv("k")?,
            v: conv("v")?,
            proj: conv("proj_out")?,
        })
    }

    fn forward(&self, b: &dyn Backend, x: Map) -> Map {
        let mut h = x.clone();
        pixel_norm(&mut h);
        let (q, k, v) = (
            self.q.forward(b, &h),
            self.k.forward(b, &h),
            self.v.forward(b, &h),
        );
        let n = x.rows * x.cols;
        let c = x.channels;
        let out = crate::engine::image::transformer::attention(
            &q.data,
            n,
            &k.data,
            &v.data,
            n,
            1,
            c,
            1.0 / (c as f32).sqrt(),
            None,
        );
        let mut y = self.proj.forward(
            b,
            &Map {
                data: out,
                ..x.clone()
            },
        );
        y.data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .for_each(|(y, x)| *y += x);
        y
    }
}

enum Layer {
    Res(ResBlock),
    Attn(AttnBlock),
    /// Nearest ×2 on both axes, a causal convolution, and the first row
    /// dropped.
    Up(Conv2),
}

pub struct AudioDecoder {
    backend: Arc<dyn Backend>,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv2,
    layers: Vec<Layer>,
    conv_out: Conv2,
    pub mel_bins: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
}

impl AudioDecoder {
    fn load(file: &SafeTensors, backend: Arc<dyn Backend>) -> Result<Self> {
        let p = "audio_vae.decoder";
        let config: serde_json::Value = file
            .metadata("config")
            .map(serde_json::from_str)
            .transpose()
            .context("parsing the audio VAE config")?
            .unwrap_or_default();
        let audio = &config["audio_vae"];
        let dd = &audio["model"]["params"]["ddconfig"];
        let causality = dd["causality_axis"].as_str().unwrap_or("height");
        ensure!(
            causality == "height",
            "the audio VAE is causal along {causality}, and this engine runs only height"
        );
        let norm = dd["norm_type"].as_str().unwrap_or("pixel");
        ensure!(
            norm == "pixel",
            "the audio VAE uses {norm} norms, not pixel"
        );
        let mel_bins = dd["mel_bins"]
            .as_u64()
            .or_else(|| audio["preprocessing"]["mel"]["n_mel_channels"].as_u64())
            .or_else(|| audio["variables"]["mel_bins"].as_u64())
            .unwrap_or(64) as usize;
        let n_levels = dd["ch_mult"].as_array().map(|a| a.len()).unwrap_or(3);
        let num_res = dd["num_res_blocks"].as_u64().unwrap_or(2) as usize;
        let (mean, _) = file.tensor("audio_vae.per_channel_statistics.mean-of-means")?;
        let (std, _) = file.tensor("audio_vae.per_channel_statistics.std-of-means")?;
        let conv = |name: &str| Conv2::load(file, &format!("{p}.{name}.conv"));
        let mut layers = vec![Layer::Res(ResBlock::load(
            file,
            &format!("{p}.mid.block_1"),
        )?)];
        if file.has(&format!("{p}.mid.attn_1.q.weight")) {
            layers.push(Layer::Attn(AttnBlock::load(
                file,
                &format!("{p}.mid.attn_1"),
            )?));
        }
        layers.push(Layer::Res(ResBlock::load(
            file,
            &format!("{p}.mid.block_2"),
        )?));
        for level in (0..n_levels).rev() {
            for j in 0..=num_res {
                layers.push(Layer::Res(ResBlock::load(
                    file,
                    &format!("{p}.up.{level}.block.{j}"),
                )?));
                if file.has(&format!("{p}.up.{level}.attn.{j}.q.weight")) {
                    layers.push(Layer::Attn(AttnBlock::load(
                        file,
                        &format!("{p}.up.{level}.attn.{j}"),
                    )?));
                }
            }
            if level != 0 {
                layers.push(Layer::Up(conv(&format!("up.{level}.upsample.conv"))?));
            }
        }
        let conv_in = conv("conv_in")?;
        let conv_out = conv("conv_out")?;
        let latent_channels = conv_in.cin;
        Ok(Self {
            out_channels: conv_out.cout,
            backend,
            mean,
            std,
            conv_in,
            layers,
            conv_out,
            mel_bins,
            latent_channels,
        })
    }

    /// The spectrogram of an audio latent given as the transformer's tokens
    /// (`[frames][channels · bins]`, channel-major): `[out_channels][time][mel_bins]`.
    fn decode(
        &self,
        tokens: &[f32],
        frames: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Vec<f32>, usize)> {
        let b = &*self.backend;
        let c = self.latent_channels;
        let width = self.mean.len();
        ensure!(
            tokens.len() == frames * width && width.is_multiple_of(c),
            "the audio latent is not {frames} tokens of {width}"
        );
        let bins = width / c;
        // Un-normalize per token position, then lay out as [t][f][c].
        let mut data = vec![0.0f32; frames * bins * c];
        for t in 0..frames {
            for ci in 0..c {
                for f in 0..bins {
                    let at = ci * bins + f;
                    let v = tokens[t * width + at] * self.std[at] + self.mean[at];
                    data[(t * bins + f) * c + ci] = v;
                }
            }
        }
        let mut x = self.conv_in.forward(
            b,
            &Map {
                rows: frames,
                cols: bins,
                channels: c,
                data,
            },
        );
        for layer in &self.layers {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = match layer {
                Layer::Res(block) => block.forward(b, x),
                Layer::Attn(block) => block.forward(b, x),
                Layer::Up(conv) => {
                    let up = nearest2(&x);
                    let mut y = conv.forward(b, &up);
                    // Drop the first row (time step).
                    let row = y.cols * y.channels;
                    y.data.drain(..row);
                    y.rows -= 1;
                    y
                }
            };
        }
        pixel_norm(&mut x);
        silu(&mut x);
        let x = self.conv_out.forward(b, &x);
        // Crop or zero-pad to the target shape.
        let target_t = (frames * LATENT_DOWNSAMPLE)
            .saturating_sub(LATENT_DOWNSAMPLE - 1)
            .max(1);
        let target_f = self.mel_bins;
        let oc = self.out_channels;
        let mut out = vec![0.0f32; oc * target_t * target_f];
        for ch in 0..oc {
            for t in 0..target_t.min(x.rows) {
                for f in 0..target_f.min(x.cols) {
                    out[(ch * target_t + t) * target_f + f] =
                        x.data[(t * x.cols + f) * x.channels + ch];
                }
            }
        }
        Ok((out, target_t))
    }
}

fn nearest2(x: &Map) -> Map {
    let (h, w, c) = (x.rows * 2, x.cols * 2, x.channels);
    let mut data = vec![0.0f32; h * w * c];
    data.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        for xx in 0..w {
            let src = ((y / 2) * x.cols + xx / 2) * c;
            row[xx * c..(xx + 1) * c].copy_from_slice(&x.data[src..src + c]);
        }
    });
    Map {
        rows: h,
        cols: w,
        channels: c,
        data,
    }
}

// ---------------------------------------------------------------------------
// The vocoder
// ---------------------------------------------------------------------------

/// A 1-D signal `[time][channels]`.
#[derive(Clone)]
struct Wave {
    len: usize,
    channels: usize,
    data: Vec<f32>,
}

/// A weight that may be stored weight-normed (`weight_g`, `weight_v`).
fn weight(file: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    if let Ok((w, shape)) = file.tensor(&format!("{name}.weight")) {
        return Ok((w, shape.to_vec()));
    }
    // `torch.nn.utils.weight_norm`'s names, or its parametrization's.
    let (g_name, v_name) = if file.has(&format!("{name}.weight_g")) {
        (format!("{name}.weight_g"), format!("{name}.weight_v"))
    } else {
        (
            format!("{name}.parametrizations.weight.original0"),
            format!("{name}.parametrizations.weight.original1"),
        )
    };
    let (g, _) = file.tensor(&g_name)?;
    let (mut v, shape) = file.tensor(&v_name)?;
    let shape = shape.to_vec();
    let per = v.len() / shape[0];
    for (o, row) in v.chunks_mut(per).enumerate() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        row.iter_mut().for_each(|x| *x *= g[o] / norm);
    }
    Ok((v, shape))
}

/// `Conv1d(cin, cout, k, dilation, padding)`.
struct Conv1 {
    w: QuantMatrix,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    dilation: usize,
    pad: usize,
}

impl Conv1 {
    fn load(file: &SafeTensors, name: &str, dilation: usize, pad: usize) -> Result<Self> {
        let (w, shape) = weight(file, name)?;
        let (cout, cin, k) = match shape.as_slice() {
            [cout, cin, k] => (*cout, *cin, *k),
            other => bail!("{name}: unexpected kernel shape {other:?}"),
        };
        let bias = file
            .tensor(&format!("{name}.bias"))
            .map(|t| t.0)
            .unwrap_or_else(|_| vec![0.0; cout]);
        let mut rows = Vec::with_capacity(cout * k * cin);
        for o in 0..cout {
            for kk in 0..k {
                for i in 0..cin {
                    rows.push(w[(o * cin + i) * k + kk]);
                }
            }
        }
        Ok(Self {
            w: conv_matrix(rows, k * cin, cout, VaePrecision::F32),
            bias,
            cin,
            cout,
            k,
            dilation,
            pad,
        })
    }

    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        debug_assert_eq!(x.channels, self.cin);
        let (k, d, cin) = (self.k, self.dilation, self.cin);
        let out_len = x.len + 2 * self.pad - d * (k - 1);
        let cols = self.w.in_dim;
        let mut out = Vec::with_capacity(out_len * self.cout);
        // Bands of 64 MiB of patches.
        let band = ((64usize << 20) / (cols * 4)).clamp(1, out_len.max(1));
        let mut start = 0;
        while start < out_len {
            let end = (start + band).min(out_len);
            let n = end - start;
            let mut patches = vec![0.0f32; n * cols];
            patches
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(i, patch)| {
                    let t = (start + i) as isize;
                    for kk in 0..k {
                        let src = t + (kk * d) as isize - self.pad as isize;
                        if src >= 0 && (src as usize) < x.len {
                            let s = src as usize * cin;
                            patch[kk * cin..(kk + 1) * cin].copy_from_slice(&x.data[s..s + cin]);
                        }
                    }
                });
            let mut y = b
                .matmul_batch(&[MatmulOp {
                    x: &patches,
                    n_tokens: n,
                    w: &self.w,
                }])
                .pop()
                .expect("one op in, one result out");
            y.par_chunks_mut(self.cout).for_each(|row| {
                for (v, bias) in row.iter_mut().zip(&self.bias) {
                    *v += bias;
                }
            });
            out.extend_from_slice(&y);
            start = end;
        }
        Wave {
            len: out_len,
            channels: self.cout,
            data: out,
        }
    }
}

/// `ConvTranspose1d(cin, cout, k, stride, padding = (k − stride) / 2)`:
/// every input step's `[k][cout]` contribution as one matmul, scattered.
struct ConvT1 {
    /// Rows `[k · cout][cin]`.
    w: QuantMatrix,
    bias: Vec<f32>,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl ConvT1 {
    fn load(file: &SafeTensors, name: &str, stride: usize) -> Result<Self> {
        let (w, shape) = weight(file, name)?;
        let (cin, cout, k) = match shape.as_slice() {
            [cin, cout, k] => (*cin, *cout, *k),
            other => bail!("{name}: unexpected kernel shape {other:?}"),
        };
        let (bias, _) = file.tensor(&format!("{name}.bias"))?;
        let mut rows = Vec::with_capacity(k * cout * cin);
        for kk in 0..k {
            for o in 0..cout {
                for i in 0..cin {
                    rows.push(w[(i * cout + o) * k + kk]);
                }
            }
        }
        Ok(Self {
            w: QuantMatrix::from_f32_rows(rows, cin, k * cout),
            bias,
            cout,
            k,
            stride,
            pad: (k - stride) / 2,
        })
    }

    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        let (k, s, cout) = (self.k, self.stride, self.cout);
        let full = (x.len - 1) * s + k;
        let out_len = full - 2 * self.pad;
        let z = b
            .matmul_batch(&[MatmulOp {
                x: &x.data,
                n_tokens: x.len,
                w: &self.w,
            }])
            .pop()
            .expect("one op in, one result out");
        // y[t] = Σ_{j, kk: j·s + kk − pad = t} z[j][kk].
        let mut out = vec![0.0f32; out_len * cout];
        out.par_chunks_mut(cout).enumerate().for_each(|(t, o)| {
            o.copy_from_slice(&self.bias);
            let tp = t + self.pad;
            // j·s + kk = tp, 0 ≤ kk < k.
            let j_max = (tp / s).min(x.len - 1);
            let j_min = (tp + 1).saturating_sub(k).div_ceil(s);
            for j in j_min..=j_max {
                let kk = tp - j * s;
                if kk < k {
                    let row = &z[(j * k + kk) * cout..][..cout];
                    for (v, r) in o.iter_mut().zip(row) {
                        *v += r;
                    }
                }
            }
        });
        Wave {
            len: out_len,
            channels: cout,
            data: out,
        }
    }
}

/// The modified Bessel function of the first kind, order zero.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..64 {
        term *= y / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// `kaiser_sinc_filter1d`: a Kaiser-windowed sinc low-pass of `k` taps.
fn kaiser_sinc(cutoff: f64, half_width: f64, k: usize) -> Vec<f32> {
    let even = k.is_multiple_of(2);
    let half = (k / 2) as f64;
    let delta_f = 4.0 * half_width;
    let amplitude = 2.285 * (half - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if amplitude > 50.0 {
        0.1102 * (amplitude - 8.7)
    } else if amplitude >= 21.0 {
        0.5842 * (amplitude - 21.0).powf(0.4) + 0.07886 * (amplitude - 21.0)
    } else {
        0.0
    };
    let window: Vec<f64> = (0..k)
        .map(|n| {
            let r = if k > 1 {
                2.0 * n as f64 / (k - 1) as f64 - 1.0
            } else {
                0.0
            };
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
        })
        .collect();
    let time: Vec<f64> = (0..k)
        .map(|i| {
            if even {
                i as f64 - half + 0.5
            } else {
                i as f64 - half
            }
        })
        .collect();
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        }
    };
    let mut f: Vec<f64> = window
        .iter()
        .zip(&time)
        .map(|(w, t)| 2.0 * cutoff * w * sinc(2.0 * cutoff * t))
        .collect();
    let sum: f64 = f.iter().sum();
    f.iter_mut().for_each(|v| *v /= sum);
    f.into_iter().map(|v| v as f32).collect()
}

/// `UpSample1d`: replicate-pad, a depthwise transposed convolution by
/// `ratio` with `filter`, times `ratio`, trimmed.
struct Upsample {
    filter: Vec<f32>,
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl Upsample {
    /// The Kaiser-windowed variant inside the activations.
    fn kaiser(ratio: usize, k: usize) -> Self {
        let pad = k / ratio - 1;
        Self {
            filter: kaiser_sinc(0.5 / ratio as f64, 0.6 / ratio as f64, k),
            ratio,
            pad,
            pad_left: pad * ratio + (k - ratio) / 2,
            pad_right: pad * ratio + (k - ratio).div_ceil(2),
        }
    }

    /// The Hann-windowed resampler of the bandwidth extension's skip path.
    fn hann(ratio: usize) -> Self {
        let rolloff = 0.99f64;
        let lowpass_width = 6.0f64;
        let width = (lowpass_width / rolloff).ceil() as usize;
        let k = 2 * width * ratio + 1;
        let filter = (0..k)
            .map(|i| {
                let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
                let tc = t.clamp(-lowpass_width, lowpass_width);
                let window = (tc * std::f64::consts::PI / lowpass_width / 2.0)
                    .cos()
                    .powi(2);
                let sinc = if t == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
                };
                (sinc * window * rolloff / ratio as f64) as f32
            })
            .collect();
        Self {
            filter,
            ratio,
            pad: width,
            pad_left: 2 * width * ratio,
            pad_right: k - ratio,
        }
    }

    fn forward(&self, x: &Wave) -> Wave {
        let (r, c, k) = (self.ratio, x.channels, self.filter.len());
        let padded = x.len + 2 * self.pad;
        let full = (padded - 1) * r + k;
        let out_len = full - self.pad_left - self.pad_right;
        let at = |j: usize| j.saturating_sub(self.pad).min(x.len - 1);
        let mut out = vec![0.0f32; out_len * c];
        out.par_chunks_mut(c).enumerate().for_each(|(i, o)| {
            let t = i + self.pad_left;
            // Σ over j with t − j·r ∈ [0, k).
            let j_max = (t / r).min(padded - 1);
            let j_min = (t + 1).saturating_sub(k).div_ceil(r);
            for j in j_min..=j_max {
                let tap = self.filter[t - j * r] * r as f32;
                let src = &x.data[at(j) * c..][..c];
                for (v, s) in o.iter_mut().zip(src) {
                    *v += tap * s;
                }
            }
        });
        Wave {
            len: out_len,
            channels: c,
            data: out,
        }
    }
}

/// `DownSample1d`: a Kaiser low-pass of `k` taps at stride `ratio`,
/// replicate-padded.
struct Downsample {
    filter: Vec<f32>,
    ratio: usize,
    pad_left: usize,
    pad_right: usize,
}

impl Downsample {
    fn new(ratio: usize, k: usize) -> Self {
        let even = k.is_multiple_of(2);
        Self {
            filter: kaiser_sinc(0.5 / ratio as f64, 0.6 / ratio as f64, k),
            ratio,
            pad_left: k / 2 - usize::from(even),
            pad_right: k / 2,
        }
    }

    fn forward(&self, x: &Wave) -> Wave {
        let (c, k) = (x.channels, self.filter.len());
        let padded = x.len + self.pad_left + self.pad_right;
        let out_len = (padded - k) / self.ratio + 1;
        let at = |j: usize| j.saturating_sub(self.pad_left).min(x.len - 1);
        let mut out = vec![0.0f32; out_len * c];
        out.par_chunks_mut(c).enumerate().for_each(|(i, o)| {
            for (kk, tap) in self.filter.iter().enumerate() {
                let src = &x.data[at(i * self.ratio + kk) * c..][..c];
                for (v, s) in o.iter_mut().zip(src) {
                    *v += tap * s;
                }
            }
        });
        Wave {
            len: out_len,
            channels: c,
            data: out,
        }
    }
}

/// SnakeBeta (or Snake, when there is no `beta`) between an upsampler and
/// a downsampler — BigVGAN's anti-aliased `Activation1d`.
struct Activation {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    up: Upsample,
    down: Downsample,
}

impl Activation {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        let (alpha, _) = file.tensor(&format!("{name}.act.alpha"))?;
        let beta = file
            .tensor(&format!("{name}.act.beta"))
            .map(|t| t.0)
            .unwrap_or_else(|_| alpha.clone());
        Ok(Self {
            alpha: alpha.iter().map(|a| a.exp()).collect(),
            beta: beta.iter().map(|b| b.exp()).collect(),
            up: Upsample::kaiser(2, 12),
            down: Downsample::new(2, 12),
        })
    }

    fn forward(&self, x: &Wave) -> Wave {
        let mut h = self.up.forward(x);
        h.data.par_chunks_mut(h.channels).for_each(|p| {
            for ((v, a), b) in p.iter_mut().zip(&self.alpha).zip(&self.beta) {
                let s = (*v * a).sin();
                *v += s * s / (b + 1e-9);
            }
        });
        self.down.forward(&h)
    }
}

struct AmpBlock {
    convs1: Vec<Conv1>,
    convs2: Vec<Conv1>,
    acts1: Vec<Activation>,
    acts2: Vec<Activation>,
}

impl AmpBlock {
    fn load(file: &SafeTensors, p: &str, kernel: usize, dilations: &[usize]) -> Result<Self> {
        let pad = |d: usize| (kernel * d - d) / 2;
        Ok(Self {
            convs1: dilations
                .iter()
                .enumerate()
                .map(|(i, &d)| Conv1::load(file, &format!("{p}.convs1.{i}"), d, pad(d)))
                .collect::<Result<_>>()?,
            convs2: (0..dilations.len())
                .map(|i| Conv1::load(file, &format!("{p}.convs2.{i}"), 1, pad(1)))
                .collect::<Result<_>>()?,
            acts1: (0..dilations.len())
                .map(|i| Activation::load(file, &format!("{p}.acts1.{i}")))
                .collect::<Result<_>>()?,
            acts2: (0..dilations.len())
                .map(|i| Activation::load(file, &format!("{p}.acts2.{i}")))
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, b: &dyn Backend, x: Wave) -> Wave {
        let mut x = x;
        for i in 0..self.convs1.len() {
            let h = self.acts1[i].forward(&x);
            let h = self.convs1[i].forward(b, &h);
            let h = self.acts2[i].forward(&h);
            let h = self.convs2[i].forward(b, &h);
            x.data
                .par_iter_mut()
                .zip(h.data.par_iter())
                .for_each(|(x, h)| *x += h);
        }
        x
    }
}

/// One BigVGAN generator (`Vocoder` with `resblock = "AMP1"`).
struct Generator {
    conv_pre: Conv1,
    ups: Vec<ConvT1>,
    /// `[upsample][kernel]`.
    blocks: Vec<Vec<AmpBlock>>,
    act_post: Activation,
    conv_post: Conv1,
    /// `use_tanh_at_final` — `tanh`, else a clamp — when the final
    /// activation applies at all.
    final_tanh: Option<bool>,
}

impl Generator {
    fn load(
        file: &SafeTensors,
        p: &str,
        cfg: &serde_json::Value,
        final_activation: bool,
    ) -> Result<Self> {
        let list = |key: &str, default: &[u64]| -> Vec<usize> {
            cfg[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_else(|| default.iter().map(|&v| v as usize).collect())
        };
        let resblock = cfg["resblock"].as_str().unwrap_or("1");
        ensure!(
            resblock == "AMP1",
            "the vocoder's residual blocks are '{resblock}', and this engine runs only AMP1"
        );
        let rates = list("upsample_rates", &[6, 5, 2, 2, 2]);
        let kernels = list("resblock_kernel_sizes", &[3, 7, 11]);
        let dilations: Vec<Vec<usize>> = cfg["resblock_dilation_sizes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|d| {
                        d.as_array()
                            .map(|d| {
                                d.iter()
                                    .filter_map(|v| v.as_u64())
                                    .map(|v| v as usize)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![vec![1, 3, 5]; kernels.len()]);
        ensure!(
            dilations.len() == kernels.len(),
            "the vocoder's kernel and dilation lists differ"
        );
        let ups = rates
            .iter()
            .enumerate()
            .map(|(i, &r)| ConvT1::load(file, &format!("{p}.ups.{i}"), r))
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = Vec::new();
        for i in 0..rates.len() {
            let mut level = Vec::new();
            for (j, (&k, d)) in kernels.iter().zip(&dilations).enumerate() {
                level.push(AmpBlock::load(
                    file,
                    &format!("{p}.resblocks.{}", i * kernels.len() + j),
                    k,
                    d,
                )?);
            }
            blocks.push(level);
        }
        Ok(Self {
            conv_pre: Conv1::load(file, &format!("{p}.conv_pre"), 1, 3)?,
            ups,
            blocks,
            act_post: Activation::load(file, &format!("{p}.act_post"))?,
            conv_post: Conv1::load(file, &format!("{p}.conv_post"), 1, 3)?,
            final_tanh: final_activation
                .then(|| cfg["use_tanh_at_final"].as_bool().unwrap_or(true)),
        })
    }

    /// `mel` is `[2 · bins][frames]` laid out `[frames][2 · bins]`
    /// (channel `s · bins + m`); the result is `[samples][2]`.
    fn forward(
        &self,
        b: &dyn Backend,
        mel: Wave,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Wave> {
        let mut x = self.conv_pre.forward(b, &mel);
        for (up, level) in self.ups.iter().zip(&self.blocks) {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            x = up.forward(b, &x);
            let outs: Vec<Wave> = level
                .iter()
                .map(|block| block.forward(b, x.clone()))
                .collect();
            let n = outs.len() as f32;
            let mut mean = outs[0].clone();
            for o in &outs[1..] {
                mean.data
                    .par_iter_mut()
                    .zip(o.data.par_iter())
                    .for_each(|(m, v)| *m += v);
            }
            mean.data.par_iter_mut().for_each(|v| *v /= n);
            x = mean;
        }
        let x = self.act_post.forward(&x);
        let mut x = self.conv_post.forward(b, &x);
        match self.final_tanh {
            Some(true) => x.data.par_iter_mut().for_each(|v| *v = v.tanh()),
            Some(false) => x.data.par_iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0)),
            None => {}
        }
        Ok(x)
    }
}

/// The bandwidth extension's causal log-mel spectrogram.
struct MelStft {
    /// `[2 · freqs][n_fft]`: real rows, then imaginary rows.
    basis: QuantMatrix,
    /// `[mels][freqs]`.
    mel_basis: Vec<f32>,
    freqs: usize,
    mels: usize,
    n_fft: usize,
    hop: usize,
    win: usize,
}

impl MelStft {
    fn load(file: &SafeTensors, p: &str, cfg: &serde_json::Value) -> Result<Self> {
        let (basis, shape) = file.tensor(&format!("{p}.stft_fn.forward_basis"))?;
        let (rows, n_fft) = match shape {
            [rows, 1, n_fft] => (*rows, *n_fft),
            other => bail!("{p}: unexpected STFT basis shape {other:?}"),
        };
        let (mel_basis, mshape) = file.tensor(&format!("{p}.mel_basis"))?;
        let mels = mshape[0];
        let freqs = rows / 2;
        ensure!(
            mshape[1] == freqs,
            "{p}: the mel basis does not match the STFT"
        );
        let hop = cfg["hop_length"].as_u64().unwrap_or(n_fft as u64 / 4) as usize;
        Ok(Self {
            basis: QuantMatrix::from_f32_rows(basis, n_fft, rows),
            mel_basis,
            freqs,
            mels,
            n_fft,
            hop,
            win: n_fft,
        })
    }

    /// One channel's log-mel frames, `[frames][mels]`.
    fn log_mel(&self, b: &dyn Backend, y: &[f32]) -> Vec<f32> {
        let left = self.win.saturating_sub(self.hop);
        let padded: Vec<f32> = std::iter::repeat_n(0.0, left)
            .chain(y.iter().copied())
            .collect();
        let frames = (padded.len() - self.n_fft) / self.hop + 1;
        let windows: Vec<f32> = (0..frames)
            .flat_map(|f| {
                padded[f * self.hop..f * self.hop + self.n_fft]
                    .iter()
                    .copied()
            })
            .collect();
        let spec = b
            .matmul_batch(&[MatmulOp {
                x: &windows,
                n_tokens: frames,
                w: &self.basis,
            }])
            .pop()
            .expect("one op in, one result out");
        let (nf, mels) = (self.freqs, self.mels);
        let mut out = vec![0.0f32; frames * mels];
        out.par_chunks_mut(mels).enumerate().for_each(|(f, row)| {
            let s = &spec[f * 2 * nf..(f + 1) * 2 * nf];
            let mag: Vec<f32> = (0..nf)
                .map(|i| (s[i] * s[i] + s[nf + i] * s[nf + i]).sqrt())
                .collect();
            for (m, v) in row.iter_mut().enumerate() {
                let e: f32 = self.mel_basis[m * nf..(m + 1) * nf]
                    .iter()
                    .zip(&mag)
                    .map(|(w, a)| w * a)
                    .sum();
                *v = e.max(1e-5).ln();
            }
        });
        out
    }
}

struct Bwe {
    generator: Generator,
    stft: MelStft,
    resampler: Upsample,
    input_rate: usize,
    output_rate: usize,
}

/// The audio VAE's decoder and the vocoder, loaded together from one file.
pub struct AudioPipeline {
    backend: Arc<dyn Backend>,
    decoder: AudioDecoder,
    vocoder: Generator,
    bwe: Option<Bwe>,
    pub sample_rate: usize,
}

/// A stereo waveform, interleaved `[samples][2]`.
pub struct Waveform {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: usize,
}

impl AudioPipeline {
    pub fn load(path: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let decoder = AudioDecoder::load(&file, backend.clone())?;
        let config: serde_json::Value = file
            .metadata("config")
            .map(serde_json::from_str)
            .transpose()
            .context("parsing the audio VAE config")?
            .unwrap_or_default();
        let voc = &config["vocoder"];
        let (vocoder, bwe, sample_rate) = if voc.get("bwe").is_some() {
            let prefix = if file.has("vocoder.vocoder.conv_pre.weight") {
                "vocoder.vocoder"
            } else {
                "vocoder"
            };
            let vocoder = Generator::load(&file, prefix, &voc["vocoder"], true)?;
            let bwe_cfg = &voc["bwe"];
            let input_rate = bwe_cfg["input_sampling_rate"]
                .as_u64()
                .ok_or_else(|| anyhow!("the bandwidth extension has no input rate"))?
                as usize;
            let output_rate = bwe_cfg["output_sampling_rate"]
                .as_u64()
                .ok_or_else(|| anyhow!("the bandwidth extension has no output rate"))?
                as usize;
            ensure!(
                output_rate.is_multiple_of(input_rate),
                "the bandwidth extension resamples {input_rate} Hz to {output_rate} Hz"
            );
            let bwe = Bwe {
                generator: Generator::load(&file, "vocoder.bwe_generator", bwe_cfg, false)?,
                stft: MelStft::load(&file, "vocoder.mel_stft", bwe_cfg)?,
                resampler: Upsample::hann(output_rate / input_rate),
                input_rate,
                output_rate,
            };
            (vocoder, Some(bwe), output_rate)
        } else {
            let rate = voc["output_sampling_rate"].as_u64().unwrap_or(24000) as usize;
            (Generator::load(&file, "vocoder", voc, true)?, None, rate)
        };
        Ok(Self {
            backend,
            decoder,
            vocoder,
            bwe,
            sample_rate,
        })
    }

    /// Mel bins × channels the latent tokens carry (the transformer's
    /// audio width).
    pub fn token_width(&self) -> usize {
        self.decoder.mean.len()
    }

    /// The waveform of an audio latent given as the transformer's tokens.
    pub fn decode(
        &self,
        tokens: &[f32],
        frames: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Waveform> {
        let (spec, t) = self.decoder.decode(tokens, frames, cancel)?;
        self.vocode(&spec, t, cancel)
    }

    /// The waveform of a log-mel spectrogram, `[channels][frames][mel_bins]`
    /// at 100 frames a second — the vocoder, and the bandwidth extension
    /// when there is one.
    pub fn vocode(
        &self,
        spec: &[f32],
        t: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Waveform> {
        let b = &*self.backend;
        let (oc, bins) = (self.decoder.out_channels, self.decoder.mel_bins);
        ensure!(
            spec.len() == oc * t * bins,
            "a spectrogram of {} values is not {oc} × {t} × {bins}",
            spec.len()
        );
        // [s][t][m] → [t][s · bins + m].
        let mut mel = vec![0.0f32; t * oc * bins];
        for s in 0..oc {
            for tt in 0..t {
                for m in 0..bins {
                    mel[tt * oc * bins + s * bins + m] = spec[(s * t + tt) * bins + m];
                }
            }
        }
        let mel = Wave {
            len: t,
            channels: oc * bins,
            data: mel,
        };
        let x = self.vocoder.forward(b, mel, cancel)?;
        let Some(bwe) = &self.bwe else {
            return Ok(Waveform {
                samples: x.data,
                channels: x.channels,
                sample_rate: self.sample_rate,
            });
        };
        let out_len = x.len * bwe.output_rate / bwe.input_rate;
        let hop = bwe.stft.hop;
        let mut x = x;
        let rem = x.len % hop;
        if rem != 0 {
            x.data
                .extend(std::iter::repeat_n(0.0, (hop - rem) * x.channels));
            x.len += hop - rem;
        }
        let ch = x.channels;
        // Each channel's log-mel, interleaved as the generator reads it.
        let per_channel: Vec<Vec<f32>> = (0..ch)
            .map(|c| {
                let y: Vec<f32> = x.data.iter().skip(c).step_by(ch).copied().collect();
                bwe.stft.log_mel(b, &y)
            })
            .collect();
        let mels = bwe.stft.mels;
        let frames = per_channel[0].len() / mels;
        let mut input = vec![0.0f32; frames * ch * mels];
        for (c, m) in per_channel.iter().enumerate() {
            for f in 0..frames {
                input[f * ch * mels + c * mels..][..mels]
                    .copy_from_slice(&m[f * mels..(f + 1) * mels]);
            }
        }
        let residual = bwe.generator.forward(
            b,
            Wave {
                len: frames,
                channels: ch * mels,
                data: input,
            },
            cancel,
        )?;
        let skip = bwe.resampler.forward(&x);
        ensure!(
            residual.len == skip.len && residual.channels == skip.channels,
            "the bandwidth extension's residual ({} samples) and skip ({}) differ",
            residual.len,
            skip.len
        );
        let n = out_len.min(skip.len) * ch;
        let samples = residual.data[..n]
            .iter()
            .zip(&skip.data[..n])
            .map(|(r, s)| (r + s).clamp(-1.0, 1.0))
            .collect();
        Ok(Waveform {
            samples,
            channels: ch,
            sample_rate: bwe.output_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kaiser_filter_is_normalized_and_symmetric() {
        let f = kaiser_sinc(0.25, 0.3, 12);
        let sum: f32 = f.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for i in 0..6 {
            assert!((f[i] - f[11 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn up_and_down_by_two_keep_the_length_and_a_constant() {
        let x = Wave {
            len: 50,
            channels: 1,
            data: vec![0.5; 50],
        };
        let up = Upsample::kaiser(2, 12).forward(&x);
        assert_eq!(up.len, 100);
        assert!(up.data.iter().all(|v| (v - 0.5).abs() < 1e-3));
        let down = Downsample::new(2, 12).forward(&up);
        assert_eq!(down.len, 50);
        assert!(down.data.iter().all(|v| (v - 0.5).abs() < 1e-3));
    }

    #[test]
    fn the_hann_resampler_multiplies_the_length() {
        let x = Wave {
            len: 40,
            channels: 2,
            data: vec![0.25; 80],
        };
        let y = Upsample::hann(3).forward(&x);
        assert_eq!(y.len, 120);
        assert!(y.data[20..200].iter().all(|v| (v - 0.25).abs() < 1e-2));
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-10);
    }
}

#[cfg(test)]
mod vocoder_probe {
    use super::*;

    /// librosa's Slaney mel scale: linear to 1 kHz, logarithmic above.
    fn hz_to_mel(f: f64) -> f64 {
        let (f_sp, min_log_hz, min_log_mel, logstep) =
            (200.0 / 3.0, 1000.0, 15.0, 6.4f64.ln() / 27.0);
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    }
    fn mel_to_hz(m: f64) -> f64 {
        let (f_sp, min_log_hz, min_log_mel, logstep) =
            (200.0 / 3.0, 1000.0, 15.0, 6.4f64.ln() / 27.0);
        if m >= min_log_mel {
            min_log_hz * ((m - min_log_mel) * logstep).exp()
        } else {
            m * f_sp
        }
    }

    /// `torchaudio.transforms.MelSpectrogram(16000, n_fft 1024, hop 160,
    /// 64 mels, f_max 8000, power 1, center, reflect, Slaney scale and
    /// norm)` then `log(clamp(·, 1e-5))`: `[frames][64]`.
    fn log_mel(y: &[f32]) -> Vec<f32> {
        let (sr, n_fft, hop, n_mels) = (16000.0f64, 1024usize, 160usize, 64usize);
        let n_freq = n_fft / 2 + 1;
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let hz: Vec<f64> = (0..n_mels + 2)
            .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let fft_hz: Vec<f64> = (0..n_freq).map(|k| k as f64 * sr / n_fft as f64).collect();
        let mut fb = vec![0.0f64; n_mels * n_freq];
        for m in 0..n_mels {
            let enorm = 2.0 / (hz[m + 2] - hz[m]);
            for (k, &f) in fft_hz.iter().enumerate() {
                let lower = (f - hz[m]) / (hz[m + 1] - hz[m]);
                let upper = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
                fb[m * n_freq + k] = lower.min(upper).max(0.0) * enorm;
            }
        }
        let pad = n_fft / 2;
        let n = y.len();
        let at = |i: isize| -> f64 {
            let j = if i < 0 {
                -i
            } else if i >= n as isize {
                2 * (n as isize - 1) - i
            } else {
                i
            };
            y[j as usize] as f64
        };
        let frames = 1 + n / hop;
        let window: Vec<f64> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .collect();
        (0..frames)
            .flat_map(|f| {
                let seg: Vec<f64> = (0..n_fft)
                    .map(|i| at((f * hop + i) as isize - pad as isize) * window[i])
                    .collect();
                let mag: Vec<f64> = (0..n_freq)
                    .map(|k| {
                        let (mut re, mut im) = (0.0, 0.0);
                        for (i, s) in seg.iter().enumerate() {
                            let a = -2.0 * std::f64::consts::PI * (k * i) as f64 / n_fft as f64;
                            re += s * a.cos();
                            im += s * a.sin();
                        }
                        (re * re + im * im).sqrt()
                    })
                    .collect();
                (0..n_mels)
                    .map(|m| {
                        let e: f64 = (0..n_freq).map(|k| fb[m * n_freq + k] * mag[k]).sum();
                        e.max(1e-5).ln() as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn write_wav(path: &str, samples: &[f32], channels: usize, rate: usize) {
        let n = samples.len() as u32;
        let mut wav = b"RIFF".to_vec();
        wav.extend((36 + n * 2).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend((channels as u16).to_le_bytes());
        wav.extend((rate as u32).to_le_bytes());
        wav.extend((rate as u32 * channels as u32 * 2).to_le_bytes());
        wav.extend((channels as u16 * 2).to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((n * 2).to_le_bytes());
        for s in samples {
            wav.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(path, wav).unwrap();
    }

    /// The power of `x` at `freq`.
    fn goertzel(x: &[f32], freq: f64, rate: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq / rate;
        let c = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in x {
            let s = v as f64 + c * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        s1 * s1 + s2 * s2 - c * s1 * s2
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64 - ma, *y as f64 - mb);
            c += x * y;
            va += x * x;
            vb += y * y;
        }
        c / (va * vb).sqrt()
    }

    /// With `ORANGU_LTX_AUDIO_VAE` naming the audio VAE: a harmonic tone's
    /// log-mel through the vocoder and the bandwidth extension, and the
    /// result's own log-mel (taken back to 16 kHz) against the input's.
    /// A vocoder that works reproduces the spectrogram it was given.
    #[test]
    #[ignore]
    fn the_vocoder_resynthesizes_a_known_spectrogram() {
        let Ok(path) = std::env::var("ORANGU_LTX_AUDIO_VAE") else {
            return;
        };
        let pipeline = AudioPipeline::load(
            Path::new(&path),
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .unwrap();
        let sr = 16000.0f64;
        let mut phase = 0.0f64;
        let tone: Vec<f32> = (0..32000)
            .map(|i| {
                let t = i as f64 / sr;
                let f0 = 220.0 * (1.0 + 0.03 * (2.0 * std::f64::consts::PI * 5.0 * t).sin());
                phase += 2.0 * std::f64::consts::PI * f0 / sr;
                let env = (t * 8.0).min(1.0) * ((2.0 - t) * 8.0).min(1.0);
                ((1..=10)
                    .map(|k| (k as f64 * phase).sin() * 0.3 / k as f64)
                    .sum::<f64>()
                    * env) as f32
            })
            .collect();
        let mel = log_mel(&tone);
        let frames = mel.len() / 64;
        // Both channels the same: [s][t][m].
        let spec: Vec<f32> = mel.iter().chain(mel.iter()).copied().collect();
        if let Ok(out) = std::env::var("ORANGU_VOCODER_MAIN_OUT") {
            // The 16 kHz generator alone, before the bandwidth extension.
            let mut input = vec![0.0f32; frames * 128];
            for f in 0..frames {
                input[f * 128..f * 128 + 64].copy_from_slice(&mel[f * 64..(f + 1) * 64]);
                input[f * 128 + 64..(f + 1) * 128].copy_from_slice(&mel[f * 64..(f + 1) * 64]);
            }
            let x = pipeline
                .vocoder
                .forward(
                    &*pipeline.backend,
                    Wave {
                        len: frames,
                        channels: 128,
                        data: input,
                    },
                    None,
                )
                .unwrap();
            let left: Vec<f32> = x.data.iter().step_by(x.channels).copied().collect();
            write_wav(&out, &left, 1, 16000);
        }
        let w = pipeline.vocode(&spec, frames, None).unwrap();
        eprintln!(
            "{} samples at {} Hz, {} channels",
            w.samples.len() / w.channels,
            w.sample_rate,
            w.channels
        );
        // Back to 16 kHz: the left channel, three-sample means.
        let ratio = w.sample_rate / 16000;
        let left: Vec<f32> = w.samples.iter().step_by(w.channels).copied().collect();
        let down: Vec<f32> = left
            .chunks(ratio)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect();
        let out_mel = log_mel(&down);
        let r = correlation(&mel, &out_mel);
        let rms = (left.iter().map(|v| v * v).sum::<f32>() / left.len() as f32).sqrt();
        eprintln!("log-mel correlation {r:.3}, output rms {rms:.4}");
        if let Ok(out) = std::env::var("ORANGU_VOCODER_OUT") {
            let mut wav = Vec::new();
            let n = w.samples.len() as u32;
            wav.extend_from_slice(b"RIFF");
            wav.extend((36 + n * 2).to_le_bytes());
            wav.extend(b"WAVEfmt ");
            wav.extend(16u32.to_le_bytes());
            wav.extend(1u16.to_le_bytes());
            wav.extend((w.channels as u16).to_le_bytes());
            wav.extend((w.sample_rate as u32).to_le_bytes());
            wav.extend((w.sample_rate as u32 * w.channels as u32 * 2).to_le_bytes());
            wav.extend((w.channels as u16 * 2).to_le_bytes());
            wav.extend(16u16.to_le_bytes());
            wav.extend(b"data");
            wav.extend((n * 2).to_le_bytes());
            for s in &w.samples {
                wav.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
            }
            std::fs::write(out, wav).unwrap();
        }
        assert!(
            r > 0.8,
            "the vocoder's output does not follow its spectrogram (r = {r:.3})"
        );
        // The log-mel bands are too wide to tell a tone from noise shaped
        // like it, so the pitch itself is checked: the fundamental's band
        // (212–228 Hz, the vibrato's spread) must stand well above the gap
        // below the second harmonic.
        let middle = &down[down.len() / 4..3 * down.len() / 4];
        let level = |lo: f64, hi: f64| -> f64 {
            let mut sum = 0.0;
            let mut f = lo;
            while f <= hi {
                sum += goertzel(middle, f, 16000.0);
                f += 1.0;
            }
            sum / (hi - lo + 1.0)
        };
        let (tone, gap) = (level(212.0, 228.0), level(300.0, 380.0));
        eprintln!("fundamental {tone:.3e}, between harmonics {gap:.3e}");
        assert!(
            tone > 20.0 * gap,
            "the vocoder's output has no 220 Hz pitch (tone {tone:.3e}, gap {gap:.3e})"
        );
    }
}

#[cfg(test)]
mod decoder_probe {
    use super::*;

    /// With `ORANGU_LTX_AUDIO_VAE` and `ORANGU_AUDIO_LATENT` (raw `f32`
    /// tokens, `[frames][128]`): the decoder's log-mel as a greyscale PGM
    /// (time across, low frequencies at the bottom) at `ORANGU_MEL_OUT`.
    #[test]
    #[ignore]
    fn decode_a_dumped_latent() {
        let (Ok(vae), Ok(latent)) = (
            std::env::var("ORANGU_LTX_AUDIO_VAE"),
            std::env::var("ORANGU_AUDIO_LATENT"),
        ) else {
            return;
        };
        let p = AudioPipeline::load(
            Path::new(&vae),
            Arc::new(crate::engine::backend::CpuBackend),
        )
        .unwrap();
        let bytes = std::fs::read(latent).unwrap();
        let tokens: Vec<f32> = bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let frames = tokens.len() / 128;
        let (spec, t) = p.decoder.decode(&tokens, frames, None).unwrap();
        let bins = p.decoder.mel_bins;
        let (lo, hi) = (-11.5f32, 3.0f32);
        let mut pgm = format!("P5 {t} {bins} 255\n").into_bytes();
        for m in (0..bins).rev() {
            for tt in 0..t {
                let v = spec[tt * bins + m];
                pgm.push((((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
        std::fs::write(
            std::env::var("ORANGU_MEL_OUT").unwrap_or_else(|_| "mel.pgm".into()),
            pgm,
        )
        .unwrap();
    }
}

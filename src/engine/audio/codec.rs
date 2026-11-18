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

//! The 12 Hz speech codec's decoder (`qwen3tts_tokenizer`): sixteen
//! codebook indices a frame back to 1920 samples of 24 kHz sound.
//!
//! 1. **Residual vector quantizer.** Codebook 0 (semantic) and codebooks
//!    1–15 (acoustic) are looked up — 256 wide each — and each group summed
//!    and projected to 512 by its own `1×1` convolution; the two are added.
//! 2. **`pre_conv`**, a causal convolution (`k = 3`) to 1024.
//! 3. **The pre-transformer**: projected to 512, eight Qwen-style layers
//!    (RMSNorm, sliding-window causal attention over the last 72 frames with
//!    NeoX rotary positions, SwiGLU, a learnt per-channel *layer scale* on
//!    each residual branch), the final norm, back to 1024.
//! 4. **Two ×2 upsamplers**: a transposed convolution (`k = s = 2`) and a
//!    ConvNeXt block (causal depthwise `k = 7`, LayerNorm, `4×` pointwise
//!    MLP with exact GELU, a per-channel gain, residual).
//! 5. **The vocoder**: a causal convolution to 1536, four blocks (SnakeBeta,
//!    a causal transposed convolution ×8, ×5, ×4, ×3 halving the channels,
//!    three residual units at dilations 1, 3, 9), SnakeBeta, a causal
//!    convolution to one channel, clamped to `[-1, 1]`.
//!
//! Every convolution is causal — padded on the left only — so a frame's
//! sound depends on that frame and the ones before it, which is what lets
//! a long clip be decoded in chunks with a little left context
//! ([`Codec::decode`]).
//!
//! Signals are `[time][channels]`, row-major; the convolutions are
//! `im2col` patches through [`Backend::matmul_batch`].

use super::stack::{self, AttentionShape};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

/// Frames a chunk decodes at once, and the frames before it decoded again
/// as its context — the reference's `chunked_decode` defaults.
const CHUNK_FRAMES: usize = 300;
const CONTEXT_FRAMES: usize = 25;

/// A signal: `len` steps of `channels` values each.
#[derive(Clone)]
struct Wave {
    len: usize,
    channels: usize,
    data: Vec<f32>,
}

fn tensor3(loaded: &LoadedModel, name: &str) -> Result<(Vec<f32>, [usize; 3])> {
    let (values, dims) = loaded
        .tensor(name)
        .with_context(|| format!("qwen3tts codec: loading {name}"))?;
    let dims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    match dims.as_slice() {
        [a, b, c] => Ok((values, [*a, *b, *c])),
        [a, b] => Ok((values, [*a, *b, 1])),
        other => bail!("qwen3tts codec: {name} has an unexpected shape {other:?}"),
    }
}

fn bias(loaded: &LoadedModel, name: &str, len: usize) -> Result<Vec<f32>> {
    stack::vector(loaded, name, len)
}

/// A causal `Conv1d(cin, cout, k, dilation)`, stride 1: the input padded by
/// `(k − 1) · dilation` on the left, as long out as in.
struct Conv {
    /// Rows `[cout][k · cin]`, the tap fastest-outer.
    w: QuantMatrix,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    dilation: usize,
}

impl Conv {
    /// `weight` in the file's `[k, cin, cout]` (PyTorch's `[cout][cin][k]`),
    /// `bias` optional.
    fn load(
        loaded: &LoadedModel,
        weight: &str,
        bias: Option<&str>,
        dilation: usize,
    ) -> Result<Self> {
        let (w, [k, cin, cout]) = tensor3(loaded, weight)?;
        let mut rows = Vec::with_capacity(cout * k * cin);
        for o in 0..cout {
            for kk in 0..k {
                for i in 0..cin {
                    rows.push(w[(o * cin + i) * k + kk]);
                }
            }
        }
        let bias = match bias {
            Some(name) => self::bias(loaded, name, cout)?,
            None => vec![0.0; cout],
        };
        Ok(Self {
            w: QuantMatrix::from_f32_rows(rows, k * cin, cout),
            bias,
            cin,
            cout,
            k,
            dilation,
        })
    }

    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        debug_assert_eq!(x.channels, self.cin);
        let (k, d, cin) = (self.k, self.dilation, self.cin);
        let pad = (k - 1) * d;
        let len = x.len;
        let cols = k * cin;
        let mut out = Vec::with_capacity(len * self.cout);
        // Bands of 64 MiB of patches.
        let band = ((64usize << 20) / (cols * 4)).clamp(1, len.max(1));
        let mut start = 0;
        while start < len {
            let end = (start + band).min(len);
            let n = end - start;
            let mut y = if k == 1 {
                b.matmul_batch(&[MatmulOp {
                    x: &x.data[start * cin..end * cin],
                    n_tokens: n,
                    w: &self.w,
                }])
            } else {
                let mut patches = vec![0.0f32; n * cols];
                patches
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(i, patch)| {
                        let t = start + i;
                        for kk in 0..k {
                            let shifted = t + kk * d;
                            if shifted >= pad {
                                let s = (shifted - pad) * cin;
                                patch[kk * cin..(kk + 1) * cin]
                                    .copy_from_slice(&x.data[s..s + cin]);
                            }
                        }
                    });
                b.matmul_batch(&[MatmulOp {
                    x: &patches,
                    n_tokens: n,
                    w: &self.w,
                }])
            }
            .pop()
            .expect("one op in, one result out");
            tensor::add_bias_per_row(&mut y, &self.bias, n);
            out.extend_from_slice(&y);
            start = end;
        }
        Wave {
            len,
            channels: self.cout,
            data: out,
        }
    }
}

/// A causal `ConvTranspose1d(cin, cout, k, stride)`: the full output with
/// its last `k − stride` steps cut, `len · stride` long.
struct ConvT {
    /// Rows `[k · cout][cin]`.
    w: QuantMatrix,
    bias: Vec<f32>,
    cout: usize,
    k: usize,
    stride: usize,
}

impl ConvT {
    /// `weight` in the file's `[k, cout, cin]` (PyTorch's `[cin][cout][k]`).
    fn load(loaded: &LoadedModel, weight: &str, bias_name: &str, stride: usize) -> Result<Self> {
        let (w, [k, cout, cin]) = tensor3(loaded, weight)?;
        ensure!(
            k >= stride,
            "qwen3tts codec: {weight} has a kernel of {k} for a stride of {stride}"
        );
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
            bias: bias(loaded, bias_name, cout)?,
            cout,
            k,
            stride,
        })
    }

    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        let (k, s, cout) = (self.k, self.stride, self.cout);
        let out_len = x.len * s;
        let z = b
            .matmul_batch(&[MatmulOp {
                x: &x.data,
                n_tokens: x.len,
                w: &self.w,
            }])
            .pop()
            .expect("one op in, one result out");
        // y[t] = bias + Σ_{j, kk: j·s + kk = t} z[j][kk].
        let mut out = vec![0.0f32; out_len * cout];
        out.par_chunks_mut(cout).enumerate().for_each(|(t, o)| {
            o.copy_from_slice(&self.bias);
            let j_max = (t / s).min(x.len - 1);
            let j_min = (t + 1).saturating_sub(k).div_ceil(s);
            for j in j_min..=j_max {
                let kk = t - j * s;
                if kk < k {
                    tensor::add_inplace(o, &z[(j * k + kk) * cout..][..cout]);
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

/// SnakeBeta: `x + sin²(x · e^α) / e^β`, per channel.
struct Snake {
    alpha: Vec<f32>,
    inv_beta: Vec<f32>,
}

impl Snake {
    fn load(loaded: &LoadedModel, alpha: &str, beta: &str, channels: usize) -> Result<Self> {
        let alpha = bias(loaded, alpha, channels)?;
        let beta = bias(loaded, beta, channels)?;
        Ok(Self {
            alpha: alpha.iter().map(|a| a.exp()).collect(),
            inv_beta: beta.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect(),
        })
    }

    fn apply(&self, x: &Wave) -> Wave {
        let mut out = x.clone();
        out.data.par_chunks_mut(x.channels).for_each(|row| {
            for ((v, a), ib) in row.iter_mut().zip(&self.alpha).zip(&self.inv_beta) {
                let s = (*v * a).sin();
                *v += s * s * ib;
            }
        });
        out
    }
}

/// SnakeBeta, a causal convolution (`k = 7`, dilated), SnakeBeta, a `1×1`
/// convolution, and the input added back.
struct ResidualUnit {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
}

impl ResidualUnit {
    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        let h = self.conv1.forward(b, &self.act1.apply(x));
        let mut h = self.conv2.forward(b, &self.act2.apply(&h));
        tensor::add_inplace(&mut h.data, &x.data);
        h
    }
}

struct DecoderBlock {
    snake: Snake,
    up: ConvT,
    units: Vec<ResidualUnit>,
}

/// ConvNeXt: causal depthwise `k = 7`, LayerNorm, `Linear(d, 4d)`, GELU,
/// `Linear(4d, d)`, a per-channel gain, residual.
struct ConvNext {
    dw: Vec<f32>,
    dw_bias: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    pw1: QuantMatrix,
    pw1_b: Vec<f32>,
    pw2: QuantMatrix,
    pw2_b: Vec<f32>,
    gamma: Vec<f32>,
    dim: usize,
    k: usize,
}

impl ConvNext {
    fn load(loaded: &LoadedModel, p: &str, dim: usize) -> Result<Self> {
        let (dw, [k, one, channels]) = tensor3(loaded, &format!("{p}.dw_w"))?;
        ensure!(
            one == 1 && channels == dim,
            "qwen3tts codec: {p}.dw_w is not depthwise over {dim} channels"
        );
        let pw1 = stack::matrix(loaded, &format!("{p}.pw1_w"), dim, 4 * dim)?;
        let pw2 = stack::matrix(loaded, &format!("{p}.pw2_w"), 4 * dim, dim)?;
        Ok(Self {
            dw,
            dw_bias: bias(loaded, &format!("{p}.dw_b"), dim)?,
            norm_w: bias(loaded, &format!("{p}.norm_w"), dim)?,
            norm_b: bias(loaded, &format!("{p}.norm_b"), dim)?,
            pw1_b: bias(loaded, &format!("{p}.pw1_b"), 4 * dim)?,
            pw2_b: bias(loaded, &format!("{p}.pw2_b"), dim)?,
            gamma: bias(loaded, &format!("{p}.gamma"), dim)?,
            pw1,
            pw2,
            dim,
            k,
        })
    }

    fn forward(&self, b: &dyn Backend, x: &Wave) -> Wave {
        let (dim, k, len) = (self.dim, self.k, x.len);
        let mut h = vec![0.0f32; len * dim];
        h.par_chunks_mut(dim).enumerate().for_each(|(t, row)| {
            row.copy_from_slice(&self.dw_bias);
            for kk in 0..k {
                let shifted = t + kk;
                if shifted < k - 1 {
                    continue;
                }
                let src = &x.data[(shifted - (k - 1)) * dim..][..dim];
                for (c, (v, s)) in row.iter_mut().zip(src).enumerate() {
                    *v += self.dw[c * k + kk] * s;
                }
            }
            // LayerNorm, eps 1e-6.
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let scale = 1.0 / (var + 1e-6).sqrt();
            for ((v, w), b) in row.iter_mut().zip(&self.norm_w).zip(&self.norm_b) {
                *v = (*v - mean) * scale * w + b;
            }
        });
        let mut u = b
            .matmul_batch(&[MatmulOp {
                x: &h,
                n_tokens: len,
                w: &self.pw1,
            }])
            .pop()
            .expect("one op");
        tensor::add_bias_per_row(&mut u, &self.pw1_b, len);
        crate::engine::image::qwen3vl::gelu_erf_inplace(&mut u);
        let mut y = b
            .matmul_batch(&[MatmulOp {
                x: &u,
                n_tokens: len,
                w: &self.pw2,
            }])
            .pop()
            .expect("one op");
        y.par_chunks_mut(dim)
            .zip(x.data.par_chunks(dim))
            .for_each(|(row, input)| {
                for (((v, b), g), i) in row.iter_mut().zip(&self.pw2_b).zip(&self.gamma).zip(input)
                {
                    *v = i + (*v + b) * g;
                }
            });
        Wave {
            len,
            channels: dim,
            data: y,
        }
    }
}

struct XfmrLayer {
    attn_norm: Vec<f32>,
    q: QuantMatrix,
    k: QuantMatrix,
    v: QuantMatrix,
    o: QuantMatrix,
    attn_scale: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
    ffn_scale: Vec<f32>,
}

/// The decoder's hyperparameters, from `qwen3tts_codec.dec.*`.
#[derive(Debug, Clone)]
pub struct CodecShape {
    pub n_quantizers: usize,
    pub codebook_size: usize,
    pub dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub theta: f32,
    pub eps: f32,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub sample_rate: u32,
}

/// The codec's decoder, loaded.
pub struct Codec {
    pub shape: CodecShape,
    /// `[codebook_size][256]` for codebook 0, then one per acoustic book.
    books: Vec<Vec<f32>>,
    book_dim: usize,
    first_out: QuantMatrix,
    rest_out: QuantMatrix,
    pre_conv: Conv,
    in_proj: QuantMatrix,
    in_proj_b: Vec<f32>,
    layers: Vec<XfmrLayer>,
    norm: Vec<f32>,
    out_proj: QuantMatrix,
    out_proj_b: Vec<f32>,
    inv_freq: Vec<f32>,
    up: Vec<(ConvT, ConvNext)>,
    in_conv: Conv,
    blocks: Vec<DecoderBlock>,
    out_snake: Snake,
    out_conv: Conv,
}

/// A `1×1` convolution with no bias, `[1, cin, cout]` in the file, as a
/// matrix.
fn pointwise(loaded: &LoadedModel, name: &str) -> Result<QuantMatrix> {
    let (w, [k, cin, cout]) = tensor3(loaded, name)?;
    ensure!(k == 1, "qwen3tts codec: {name} is not a 1x1 convolution");
    Ok(QuantMatrix::from_f32_rows(w, cin, cout))
}

impl Codec {
    pub fn load(loaded: &LoadedModel) -> Result<Self> {
        let key = |suffix: &str| format!("qwen3tts_codec.{suffix}");
        let u = |suffix: &str| -> Result<usize> {
            super::meta::u64(loaded, &key(suffix))
                .map(|v| v as usize)
                .with_context(|| format!("qwen3tts codec: missing {}", key(suffix)))
        };
        let f =
            |suffix: &str, default: f32| super::meta::f32(loaded, &key(suffix)).unwrap_or(default);
        let list = |suffix: &str| -> Result<Vec<usize>> {
            super::meta::u64_array(loaded, &key(suffix))
                .map(|v| v.into_iter().map(|x| x as usize).collect())
                .with_context(|| format!("qwen3tts codec: missing {}", key(suffix)))
        };
        let shape = CodecShape {
            n_quantizers: u("dec.n_quantizers")?,
            codebook_size: u("dec.codebook_size")?,
            dim: u("dec.d_model")?,
            latent_dim: u("dec.latent_dim")?,
            decoder_dim: u("dec.decoder_dim")?,
            heads: u("dec.n_heads")?,
            head_dim: u("dec.head_dim")?,
            window: u("dec.sliding_window")?,
            theta: f("dec.rope_theta", 10000.0),
            eps: f("dec.rms_norm_eps", 1e-5),
            upsample_rates: list("dec.upsample_rates")?,
            upsampling_ratios: list("dec.upsampling_ratios")?,
            sample_rate: super::meta::u64(loaded, &key("output_sample_rate")).unwrap_or(24000)
                as u32,
        };
        let CodecShape {
            n_quantizers,
            codebook_size,
            dim,
            latent_dim,
            decoder_dim,
            heads,
            head_dim,
            ..
        } = shape.clone();
        ensure!(
            n_quantizers >= 2,
            "qwen3tts codec: {n_quantizers} quantizers"
        );

        let mut books = Vec::with_capacity(n_quantizers);
        let (first, dims) = loaded
            .tensor("codec.dec.rvq_first.codebook")
            .context("qwen3tts codec: loading codec.dec.rvq_first.codebook")?;
        let book_dim = dims[0] as usize;
        ensure!(
            dims.len() == 2 && dims[1] as usize == codebook_size,
            "qwen3tts codec: codebook 0 is not {codebook_size} entries"
        );
        books.push(first);
        for q in 0..n_quantizers - 1 {
            let name = format!("codec.dec.rvq_rest.{q}.codebook");
            let book = stack::vector(loaded, &name, codebook_size * book_dim)?;
            books.push(book);
        }

        let mut layers = Vec::new();
        while loaded.has_tensor(&format!("codec.dec.xfmr.blk.{}.attn_q_w", layers.len())) {
            let p = format!("codec.dec.xfmr.blk.{}", layers.len());
            let inner = heads * head_dim;
            let ff = loaded
                .matrix(&format!("{p}.ffn_gate_w"))
                .with_context(|| format!("qwen3tts codec: loading {p}.ffn_gate_w"))?
                .out_dim;
            layers.push(XfmrLayer {
                attn_norm: bias(loaded, &format!("{p}.attn_norm_w"), dim)?,
                q: stack::matrix(loaded, &format!("{p}.attn_q_w"), dim, inner)?,
                k: stack::matrix(loaded, &format!("{p}.attn_k_w"), dim, inner)?,
                v: stack::matrix(loaded, &format!("{p}.attn_v_w"), dim, inner)?,
                o: stack::matrix(loaded, &format!("{p}.attn_o_w"), inner, dim)?,
                attn_scale: bias(loaded, &format!("{p}.attn_ls_w"), dim)?,
                ffn_norm: bias(loaded, &format!("{p}.ffn_norm_w"), dim)?,
                gate: stack::matrix(loaded, &format!("{p}.ffn_gate_w"), dim, ff)?,
                up: stack::matrix(loaded, &format!("{p}.ffn_up_w"), dim, ff)?,
                down: stack::matrix(loaded, &format!("{p}.ffn_down_w"), ff, dim)?,
                ffn_scale: bias(loaded, &format!("{p}.ffn_ls_w"), dim)?,
            });
        }
        ensure!(
            !layers.is_empty(),
            "qwen3tts codec: no pre-transformer layers"
        );

        let mut up = Vec::new();
        for (s, &ratio) in shape.upsampling_ratios.iter().enumerate() {
            let p = format!("codec.dec.up.{s}");
            up.push((
                ConvT::load(
                    loaded,
                    &format!("{p}.tconv_w"),
                    &format!("{p}.tconv_b"),
                    ratio,
                )?,
                ConvNext::load(loaded, &format!("{p}.cnx"), latent_dim)?,
            ));
        }

        let mut blocks = Vec::new();
        let mut channels = decoder_dim;
        for (bi, &rate) in shape.upsample_rates.iter().enumerate() {
            let p = format!("codec.dec.blk.{bi}");
            let out = channels / 2;
            let mut units = Vec::new();
            for (ri, dilation) in [1usize, 3, 9].into_iter().enumerate() {
                let r = format!("{p}.res.{ri}");
                units.push(ResidualUnit {
                    act1: Snake::load(loaded, &format!("{r}.act1_a"), &format!("{r}.act1_b"), out)?,
                    conv1: Conv::load(
                        loaded,
                        &format!("{r}.conv1_w"),
                        Some(&format!("{r}.conv1_b")),
                        dilation,
                    )?,
                    act2: Snake::load(loaded, &format!("{r}.act2_a"), &format!("{r}.act2_b"), out)?,
                    conv2: Conv::load(
                        loaded,
                        &format!("{r}.conv2_w"),
                        Some(&format!("{r}.conv2_b")),
                        1,
                    )?,
                });
            }
            blocks.push(DecoderBlock {
                snake: Snake::load(
                    loaded,
                    &format!("{p}.snake_a"),
                    &format!("{p}.snake_b"),
                    channels,
                )?,
                up: ConvT::load(
                    loaded,
                    &format!("{p}.tconv_w"),
                    &format!("{p}.tconv_b"),
                    rate,
                )?,
                units,
            });
            channels = out;
        }

        Ok(Self {
            books,
            book_dim,
            first_out: pointwise(loaded, "codec.dec.rvq_first.out_proj_w")?,
            rest_out: pointwise(loaded, "codec.dec.rvq_rest.out_proj_w")?,
            pre_conv: Conv::load(
                loaded,
                "codec.dec.pre_conv_w",
                Some("codec.dec.pre_conv_b"),
                1,
            )?,
            in_proj: stack::matrix(loaded, "codec.dec.xfmr.in_proj_w", latent_dim, dim)?,
            in_proj_b: bias(loaded, "codec.dec.xfmr.in_proj_b", dim)?,
            layers,
            norm: bias(loaded, "codec.dec.xfmr.norm_w", dim)?,
            out_proj: stack::matrix(loaded, "codec.dec.xfmr.out_proj_w", dim, latent_dim)?,
            out_proj_b: bias(loaded, "codec.dec.xfmr.out_proj_b", latent_dim)?,
            inv_freq: stack::inv_freq(shape.theta, head_dim),
            up,
            in_conv: Conv::load(
                loaded,
                "codec.dec.in_conv_w",
                Some("codec.dec.in_conv_b"),
                1,
            )?,
            blocks,
            out_snake: Snake::load(
                loaded,
                "codec.dec.out_snake_a",
                "codec.dec.out_snake_b",
                channels,
            )?,
            out_conv: Conv::load(
                loaded,
                "codec.dec.out_conv_w",
                Some("codec.dec.out_conv_b"),
                1,
            )?,
            shape,
        })
    }

    /// Samples of sound per frame of codes.
    pub fn samples_per_frame(&self) -> usize {
        self.shape
            .upsample_rates
            .iter()
            .chain(&self.shape.upsampling_ratios)
            .product()
    }

    /// The codec's frames per second.
    pub fn frame_rate(&self) -> f64 {
        self.shape.sample_rate as f64 / self.samples_per_frame() as f64
    }

    /// `frames` (each `n_quantizers` codes) to sound: chunks of
    /// [`CHUNK_FRAMES`], each decoded with up to [`CONTEXT_FRAMES`] frames
    /// before it whose sound is dropped. `progress` hears the frames done.
    pub fn decode(
        &self,
        backend: &dyn Backend,
        frames: &[Vec<u32>],
        progress: &mut dyn FnMut(usize),
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<f32>> {
        let per = self.samples_per_frame();
        let mut out = Vec::with_capacity(frames.len() * per);
        let mut start = 0;
        while start < frames.len() {
            if super::cancelled(cancel) {
                bail!("cancelled");
            }
            let end = (start + CHUNK_FRAMES).min(frames.len());
            let context = CONTEXT_FRAMES.min(start);
            let wav = self.decode_chunk(backend, &frames[start - context..end])?;
            out.extend_from_slice(&wav[context * per..]);
            progress(end);
            start = end;
        }
        Ok(out)
    }

    fn decode_chunk(&self, b: &dyn Backend, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let t = frames.len();
        let CodecShape {
            n_quantizers,
            codebook_size,
            dim,
            heads,
            head_dim,
            window,
            eps,
            ..
        } = self.shape;
        let bd = self.book_dim;
        // 1. The quantizer: codebook 0, and the sum of the rest, each
        // projected.
        let mut first = vec![0.0f32; t * bd];
        let mut rest = vec![0.0f32; t * bd];
        for (f, codes) in frames.iter().enumerate() {
            ensure!(
                codes.len() == n_quantizers,
                "a frame has {} codes, the codec reads {n_quantizers}",
                codes.len()
            );
            for (q, &code) in codes.iter().enumerate() {
                let code = (code as usize).min(codebook_size - 1);
                let entry = &self.books[q][code * bd..][..bd];
                let target = if q == 0 { &mut first } else { &mut rest };
                tensor::add_inplace(&mut target[f * bd..][..bd], entry);
            }
        }
        let mut proj = b
            .matmul_batch(&[
                MatmulOp {
                    x: &first,
                    n_tokens: t,
                    w: &self.first_out,
                },
                MatmulOp {
                    x: &rest,
                    n_tokens: t,
                    w: &self.rest_out,
                },
            ])
            .into_iter();
        let mut h = proj.next().expect("two ops");
        tensor::add_inplace(&mut h, &proj.next().expect("two ops"));
        let wave = Wave {
            len: t,
            channels: self.first_out.out_dim,
            data: h,
        };

        // 2. `pre_conv`.
        let wave = self.pre_conv.forward(b, &wave);

        // 3. The pre-transformer.
        let mut x = b
            .matmul_batch(&[MatmulOp {
                x: &wave.data,
                n_tokens: t,
                w: &self.in_proj,
            }])
            .pop()
            .expect("one op");
        tensor::add_bias_per_row(&mut x, &self.in_proj_b, t);
        let shape = AttentionShape {
            heads,
            kv_heads: heads,
            head_dim,
            window: Some(window),
        };
        for layer in &self.layers {
            let mut xn = x.clone();
            tensor::rmsnorm_inplace(&mut xn, &layer.attn_norm, t, dim, eps);
            let mut qkv = b
                .matmul_batch(&[
                    MatmulOp {
                        x: &xn,
                        n_tokens: t,
                        w: &layer.q,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: t,
                        w: &layer.k,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: t,
                        w: &layer.v,
                    },
                ])
                .into_iter();
            let (mut q, mut k, v) = (
                qkv.next().expect("three ops"),
                qkv.next().expect("three ops"),
                qkv.next().expect("three ops"),
            );
            stack::rope(&mut q, t, heads, head_dim, &self.inv_freq, 0);
            stack::rope(&mut k, t, heads, head_dim, &self.inv_freq, 0);
            let attn = stack::causal_attention(&q, t, &k, &v, 0, shape);
            let o = b
                .matmul_batch(&[MatmulOp {
                    x: &attn,
                    n_tokens: t,
                    w: &layer.o,
                }])
                .pop()
                .expect("one op");
            scaled_add(&mut x, &o, &layer.attn_scale);

            let mut xn = x.clone();
            tensor::rmsnorm_inplace(&mut xn, &layer.ffn_norm, t, dim, eps);
            let mut gu = b
                .matmul_batch(&[
                    MatmulOp {
                        x: &xn,
                        n_tokens: t,
                        w: &layer.gate,
                    },
                    MatmulOp {
                        x: &xn,
                        n_tokens: t,
                        w: &layer.up,
                    },
                ])
                .into_iter();
            let (mut g, u) = (gu.next().expect("two ops"), gu.next().expect("two ops"));
            g.par_iter_mut()
                .zip(u.par_iter())
                .for_each(|(g, u)| *g = tensor::silu(*g) * u);
            let d = b
                .matmul_batch(&[MatmulOp {
                    x: &g,
                    n_tokens: t,
                    w: &layer.down,
                }])
                .pop()
                .expect("one op");
            scaled_add(&mut x, &d, &layer.ffn_scale);
        }
        tensor::rmsnorm_inplace(&mut x, &self.norm, t, dim, eps);
        let mut y = b
            .matmul_batch(&[MatmulOp {
                x: &x,
                n_tokens: t,
                w: &self.out_proj,
            }])
            .pop()
            .expect("one op");
        tensor::add_bias_per_row(&mut y, &self.out_proj_b, t);
        let mut wave = Wave {
            len: t,
            channels: self.out_proj.out_dim,
            data: y,
        };

        // 4. The upsamplers.
        for (tconv, cnx) in &self.up {
            wave = tconv.forward(b, &wave);
            wave = cnx.forward(b, &wave);
        }

        // 5. The vocoder.
        wave = self.in_conv.forward(b, &wave);
        for block in &self.blocks {
            wave = block.up.forward(b, &block.snake.apply(&wave));
            for unit in &block.units {
                wave = unit.forward(b, &wave);
            }
        }
        let wave = self.out_conv.forward(b, &self.out_snake.apply(&wave));
        debug_assert_eq!(wave.channels, 1);
        Ok(wave.data.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect())
    }
}

/// `x += y · scale`, `scale` per channel.
fn scaled_add(x: &mut [f32], y: &[f32], scale: &[f32]) {
    x.par_chunks_mut(scale.len())
        .zip(y.par_chunks(scale.len()))
        .for_each(|(x, y)| {
            for ((x, y), s) in x.iter_mut().zip(y).zip(scale) {
                *x += y * s;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;

    fn wave(len: usize, channels: usize, data: Vec<f32>) -> Wave {
        Wave {
            len,
            channels,
            data,
        }
    }

    #[test]
    fn a_causal_convolution_sees_only_the_past() {
        // One channel, k = 2, weights [w0 = 1 (older), w1 = 10 (current)].
        let conv = Conv {
            w: QuantMatrix::from_f32_rows(vec![1.0, 10.0], 2, 1),
            bias: vec![0.0],
            cin: 1,
            cout: 1,
            k: 2,
            dilation: 1,
        };
        let y = conv.forward(&CpuBackend, &wave(3, 1, vec![1.0, 2.0, 3.0]));
        assert_eq!(y.data, vec![10.0, 21.0, 32.0]);
    }

    #[test]
    fn a_dilated_convolution_reaches_further_back() {
        let conv = Conv {
            w: QuantMatrix::from_f32_rows(vec![1.0, 0.0], 2, 1),
            bias: vec![0.0],
            cin: 1,
            cout: 1,
            k: 2,
            dilation: 2,
        };
        let y = conv.forward(&CpuBackend, &wave(4, 1, vec![1.0, 2.0, 3.0, 4.0]));
        // Output t reads input t − 2.
        assert_eq!(y.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn a_causal_transposed_convolution_is_stride_times_as_long() {
        // k = 4, s = 2, one channel, all taps 1: y[t] sums the inputs whose
        // span covers t; the last k − s steps of the full output are cut.
        let tconv = ConvT {
            w: QuantMatrix::from_f32_rows(vec![1.0; 4], 1, 4),
            bias: vec![0.0],
            cout: 1,
            k: 4,
            stride: 2,
        };
        let y = tconv.forward(&CpuBackend, &wave(2, 1, vec![1.0, 10.0]));
        assert_eq!(y.data, vec![1.0, 1.0, 11.0, 11.0]);
    }
}

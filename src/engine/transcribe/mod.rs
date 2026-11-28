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

//! Speech to text with Whisper (`whisper`), a proof of concept.
//!
//! Each thirty seconds of sound becomes a log-mel spectrogram ([`mel`]);
//! the encoder turns it into 1500 frames (two convolutions, the second
//! halving the time, then pre-norm transformer blocks over learned
//! positions); the decoder writes text greedily from
//! `<|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>`,
//! attending to its own tokens and to the encoder's frames. Longer sound is
//! cut into back-to-back thirty-second windows.

pub mod mel;
pub mod wav;

use crate::engine::audio::{meta, stack};
use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;
use crate::engine::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, ensure};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const EPS: f32 = 1e-5;

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Linear {
    w: QuantMatrix,
    b: Option<Vec<f32>>,
}

struct Attn {
    norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct Ffn {
    norm: Norm,
    up: Linear,
    down: Linear,
}

struct EncBlock {
    attn: Attn,
    ffn: Ffn,
}

struct DecBlock {
    attn: Attn,
    cross: Attn,
    ffn: Ffn,
}

struct Conv {
    /// `[out, in · 3]`.
    w: Vec<f32>,
    b: Vec<f32>,
    inputs: usize,
    outputs: usize,
}

struct Ids {
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
    /// `(code, id)` for every `<|xx|>` between `sot` and `<|translate|>`.
    languages: Vec<(String, u32)>,
}

pub struct Whisper {
    backend: Arc<dyn Backend>,
    tokenizer: Tokenizer,
    n_mels: usize,
    dim: usize,
    enc_heads: usize,
    dec_heads: usize,
    filters: Vec<f32>,
    conv1: Conv,
    conv2: Conv,
    enc_pos: Vec<f32>,
    enc: Vec<EncBlock>,
    enc_norm: Norm,
    token_embd: QuantMatrix,
    dec_pos: Vec<f32>,
    dec: Vec<DecBlock>,
    dec_norm: Norm,
    ids: Ids,
    text_ctx: usize,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
pub struct Timings {
    pub mel: Duration,
    pub encode: Duration,
    pub decode: Duration,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub language: String,
    pub seconds: f32,
    pub timings: Timings,
}

fn vector(loaded: &LoadedModel, name: &str, len: usize) -> Result<Vec<f32>> {
    let (v, _) = loaded
        .tensor(name)
        .with_context(|| format!("whisper: loading {name}"))?;
    ensure!(
        v.len() == len,
        "whisper: {name} has {} values, expected {len}",
        v.len()
    );
    Ok(v)
}

fn norm(loaded: &LoadedModel, name: &str, dim: usize) -> Result<Norm> {
    Ok(Norm {
        w: vector(loaded, &format!("{name}.weight"), dim)?,
        b: vector(loaded, &format!("{name}.bias"), dim)?,
    })
}

fn linear(loaded: &LoadedModel, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let w = loaded
        .matrix(&format!("{name}.weight"))
        .with_context(|| format!("whisper: loading {name}.weight"))?;
    ensure!(
        w.in_dim == in_dim && w.out_dim == out_dim,
        "whisper: {name}.weight is {}x{}, expected {in_dim}x{out_dim}",
        w.in_dim,
        w.out_dim
    );
    let bias = format!("{name}.bias");
    let b = if loaded.has_tensor(&bias) {
        Some(vector(loaded, &bias, out_dim)?)
    } else {
        None
    };
    Ok(Linear { w, b })
}

fn attn(loaded: &LoadedModel, p: &str, dim: usize) -> Result<Attn> {
    Ok(Attn {
        norm: norm(loaded, &format!("{p}_norm"), dim)?,
        q: linear(loaded, &format!("{p}_q"), dim, dim)?,
        k: linear(loaded, &format!("{p}_k"), dim, dim)?,
        v: linear(loaded, &format!("{p}_v"), dim, dim)?,
        o: linear(loaded, &format!("{p}_output"), dim, dim)?,
    })
}

fn ffn(loaded: &LoadedModel, p: &str, dim: usize) -> Result<Ffn> {
    let ff = loaded
        .matrix(&format!("{p}.ffn_up.weight"))
        .with_context(|| format!("whisper: loading {p}.ffn_up.weight"))?
        .out_dim;
    Ok(Ffn {
        norm: norm(loaded, &format!("{p}.ffn_norm"), dim)?,
        up: linear(loaded, &format!("{p}.ffn_up"), dim, ff)?,
        down: linear(loaded, &format!("{p}.ffn_down"), ff, dim)?,
    })
}

fn conv(loaded: &LoadedModel, name: &str, inputs: usize, outputs: usize) -> Result<Conv> {
    Ok(Conv {
        w: vector(loaded, &format!("{name}.weight"), outputs * inputs * 3)?,
        b: vector(loaded, &format!("{name}.bias"), outputs)?,
        inputs,
        outputs,
    })
}

/// Abramowitz and Stegun 7.1.26, within 1.5e-7.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu(x: &mut [f32]) {
    x.par_iter_mut()
        .for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2)));
}

fn layer_norm(x: &[f32], dim: usize, n: &Norm) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_mut(dim).for_each(|row| {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, w), b) in row.iter_mut().zip(&n.w).zip(&n.b) {
            *v = (*v - mean) * inv * w + b;
        }
    });
    out
}

/// Every query row against every key row, no mask. `[nq, heads · hd]`.
fn full_attention(q: &[f32], k: &[f32], v: &[f32], nk: usize, heads: usize, hd: usize) -> Vec<f32> {
    let width = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; q.len()];
    out.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
        let (t, h) = (i / heads, i % heads);
        let qh = &q[t * width + h * hd..][..hd];
        let mut s: Vec<f32> = (0..nk)
            .map(|j| tensor::dot(qh, &k[j * width + h * hd..][..hd]) * scale)
            .collect();
        tensor::softmax_inplace(&mut s);
        for (j, w) in s.iter().enumerate() {
            tensor::axpy_inplace(o, &v[j * width + h * hd..][..hd], *w);
        }
    });
    out
}

impl Whisper {
    pub fn load(path: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let loaded = LoadedModel::open_companion(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let gguf = crate::gguf::GgufFile::open(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let tokenizer = Tokenizer::from_gguf(&gguf).context("building the whisper tokenizer")?;
        drop(gguf);
        ensure!(
            meta::string(&loaded, "general.architecture").as_deref() == Some("whisper"),
            "{} is not a whisper model",
            path.display()
        );
        let need = |k: &str| -> Result<usize> {
            meta::u64(&loaded, &format!("whisper.{k}"))
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("whisper: missing whisper.{k}"))
        };
        let dim = need("embedding_length")?;
        let n_mels = need("audio.n_mels")?;
        let audio_ctx = need("encoder.context_length")?;
        let text_ctx = need("context_length")?;
        let mut enc = Vec::new();
        for i in 0..need("encoder.block_count")? {
            let p = format!("enc.blk.{i}");
            enc.push(EncBlock {
                attn: attn(&loaded, &format!("{p}.attn"), dim)?,
                ffn: ffn(&loaded, &p, dim)?,
            });
        }
        let mut dec = Vec::new();
        for i in 0..need("block_count")? {
            let p = format!("dec.blk.{i}");
            dec.push(DecBlock {
                attn: attn(&loaded, &format!("{p}.attn"), dim)?,
                cross: attn(&loaded, &format!("{p}.cross_attn"), dim)?,
                ffn: ffn(&loaded, &p, dim)?,
            });
        }
        let token_embd = loaded
            .matrix("dec.token_embd.weight")
            .context("whisper: loading dec.token_embd.weight")?;
        ensure!(
            token_embd.in_dim == dim,
            "whisper: the token embedding is not {dim} wide"
        );

        let find = |text: &str| -> Result<u32> {
            (0..tokenizer.vocab_size() as u32)
                .find(|&id| tokenizer.token_text(id) == Some(text))
                .ok_or_else(|| anyhow!("whisper: the vocabulary has no {text}"))
        };
        let sot = need("decoder_start_token_id")? as u32;
        let translate = find("<|translate|>")?;
        let languages = (sot + 1..translate)
            .filter_map(|id| {
                let t = tokenizer.token_text(id)?;
                let code = t.strip_prefix("<|")?.strip_suffix("|>")?;
                Some((code.to_string(), id))
            })
            .collect();
        let ids = Ids {
            sot,
            eot: need("eos_token_id")? as u32,
            transcribe: find("<|transcribe|>")?,
            no_timestamps: find("<|notimestamps|>")?,
            languages,
        };
        let list = |k: &str| {
            meta::u64_array(&loaded, &format!("whisper.{k}"))
                .unwrap_or_default()
                .into_iter()
                .map(|v| v as u32)
                .collect()
        };
        Ok(Self {
            n_mels,
            dim,
            enc_heads: need("encoder.attention.head_count")?,
            dec_heads: need("attention.head_count")?,
            filters: mel::filterbank(n_mels),
            conv1: conv(&loaded, "enc.conv1", n_mels, dim)?,
            conv2: conv(&loaded, "enc.conv2", dim, dim)?,
            enc_pos: vector(&loaded, "enc.pos_embd.weight", audio_ctx * dim)?,
            enc_norm: norm(&loaded, "enc.output_norm", dim)?,
            dec_pos: vector(&loaded, "dec.pos_embd.weight", text_ctx * dim)?,
            dec_norm: norm(&loaded, "dec.output_norm", dim)?,
            suppress: list("suppress_tokens"),
            begin_suppress: list("begin_suppress_tokens"),
            enc,
            dec,
            token_embd,
            ids,
            text_ctx,
            tokenizer,
            backend,
        })
    }

    pub fn languages(&self) -> impl Iterator<Item = &str> {
        self.ids.languages.iter().map(|(c, _)| c.as_str())
    }

    fn linear(&self, x: &[f32], n: usize, l: &Linear) -> Vec<f32> {
        let mut y = self
            .backend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: n,
                w: &l.w,
            }])
            .pop()
            .expect("one op");
        if let Some(b) = &l.b {
            y.par_chunks_mut(b.len())
                .for_each(|row| tensor::add_inplace(row, b));
        }
        y
    }

    /// A kernel-3, padding-1 convolution over time-major `x` (`[t, in]`),
    /// then GELU. Returns `[t / stride, out]`.
    fn conv(&self, x: &[f32], t: usize, c: &Conv, stride: usize) -> Vec<f32> {
        let t_out = t / stride;
        let row_len = c.inputs * 3;
        let mut y = vec![0.0f32; t_out * c.outputs];
        y.par_chunks_mut(c.outputs)
            .enumerate()
            .for_each(|(o, out)| {
                let mut col = vec![0.0f32; row_len];
                for k in 0..3 {
                    let s = (o * stride + k) as isize - 1;
                    if s < 0 || s as usize >= t {
                        continue;
                    }
                    let src = &x[s as usize * c.inputs..][..c.inputs];
                    for (i, v) in src.iter().enumerate() {
                        col[i * 3 + k] = *v;
                    }
                }
                tensor::dot_rows(&col, &c.w, row_len, out);
                tensor::add_inplace(out, &c.b);
            });
        gelu(&mut y);
        y
    }

    fn ffn(&self, h: &mut [f32], n: usize, f: &Ffn) {
        let x = layer_norm(h, self.dim, &f.norm);
        let mut u = self.linear(&x, n, &f.up);
        gelu(&mut u);
        let d = self.linear(&u, n, &f.down);
        tensor::add_inplace(h, &d);
    }

    /// `[frames, dim]` from a `[n_mels, FRAMES]` spectrogram.
    fn encode(&self, mel: &[f32]) -> Vec<f32> {
        let t = mel::FRAMES;
        let mut x = vec![0.0f32; t * self.n_mels];
        for m in 0..self.n_mels {
            for f in 0..t {
                x[f * self.n_mels + m] = mel[m * t + f];
            }
        }
        let x = self.conv(&x, t, &self.conv1, 1);
        let mut h = self.conv(&x, t, &self.conv2, 2);
        let n = t / 2;
        tensor::add_inplace(&mut h, &self.enc_pos[..n * self.dim]);
        let hd = self.dim / self.enc_heads;
        for b in &self.enc {
            let x = layer_norm(&h, self.dim, &b.attn.norm);
            let q = self.linear(&x, n, &b.attn.q);
            let k = self.linear(&x, n, &b.attn.k);
            let v = self.linear(&x, n, &b.attn.v);
            let a = full_attention(&q, &k, &v, n, self.enc_heads, hd);
            let o = self.linear(&a, n, &b.attn.o);
            tensor::add_inplace(&mut h, &o);
            self.ffn(&mut h, n, &b.ffn);
        }
        layer_norm(&h, self.dim, &self.enc_norm)
    }

    /// Runs `tokens` through the decoder after what `self_kv` holds and
    /// returns the logits of the last one.
    fn step(
        &self,
        tokens: &[u32],
        self_kv: &mut [(Vec<f32>, Vec<f32>)],
        cross: &[(Vec<f32>, Vec<f32>)],
        frames: usize,
        start: usize,
    ) -> Vec<f32> {
        let (d, n) = (self.dim, tokens.len());
        let hd = d / self.dec_heads;
        let mut h: Vec<f32> = tokens
            .iter()
            .enumerate()
            .flat_map(|(i, &id)| {
                let mut row = self.token_embd.row(id as usize);
                tensor::add_inplace(&mut row, &self.dec_pos[(start + i) * d..][..d]);
                row
            })
            .collect();
        let shape = stack::AttentionShape {
            heads: self.dec_heads,
            kv_heads: self.dec_heads,
            head_dim: hd,
            window: None,
        };
        for (l, b) in self.dec.iter().enumerate() {
            let x = layer_norm(&h, d, &b.attn.norm);
            let q = self.linear(&x, n, &b.attn.q);
            let k = self.linear(&x, n, &b.attn.k);
            let v = self.linear(&x, n, &b.attn.v);
            self_kv[l].0.extend_from_slice(&k);
            self_kv[l].1.extend_from_slice(&v);
            let a = stack::causal_attention(&q, n, &self_kv[l].0, &self_kv[l].1, start, shape);
            let o = self.linear(&a, n, &b.attn.o);
            tensor::add_inplace(&mut h, &o);

            let x = layer_norm(&h, d, &b.cross.norm);
            let q = self.linear(&x, n, &b.cross.q);
            let a = full_attention(&q, &cross[l].0, &cross[l].1, frames, self.dec_heads, hd);
            let o = self.linear(&a, n, &b.cross.o);
            tensor::add_inplace(&mut h, &o);

            self.ffn(&mut h, n, &b.ffn);
        }
        let last = layer_norm(&h[(n - 1) * d..], d, &self.dec_norm);
        self.backend
            .matmul_batch(&[MatmulOp {
                x: &last,
                n_tokens: 1,
                w: &self.token_embd,
            }])
            .pop()
            .expect("one op")
    }

    /// Thirty seconds or less: the text, and the language it was heard in.
    fn window(
        &self,
        audio: &[f32],
        language: Option<&str>,
        timings: &mut Timings,
    ) -> Result<(String, String)> {
        let started = Instant::now();
        let mel = mel::log_mel(audio, &self.filters, self.n_mels);
        timings.mel += started.elapsed();

        let started = Instant::now();
        let enc = self.encode(&mel);
        let frames = mel::FRAMES / 2;
        let cross: Vec<(Vec<f32>, Vec<f32>)> = self
            .dec
            .iter()
            .map(|b| {
                (
                    self.linear(&enc, frames, &b.cross.k),
                    self.linear(&enc, frames, &b.cross.v),
                )
            })
            .collect();
        timings.encode += started.elapsed();

        let started = Instant::now();
        let fresh = || vec![(Vec::new(), Vec::new()); self.dec.len()];
        let lang = match language {
            Some(code) => self
                .ids
                .languages
                .iter()
                .find(|(c, _)| c == code)
                .cloned()
                .ok_or_else(|| anyhow!("whisper does not know the language {code:?}"))?,
            None => {
                let logits = self.step(&[self.ids.sot], &mut fresh(), &cross, frames, 0);
                self.ids
                    .languages
                    .iter()
                    .max_by(|a, b| logits[a.1 as usize].total_cmp(&logits[b.1 as usize]))
                    .cloned()
                    .ok_or_else(|| anyhow!("whisper: the model has no languages"))?
            }
        };
        let prompt = [
            self.ids.sot,
            lang.1,
            self.ids.transcribe,
            self.ids.no_timestamps,
        ];
        let mut kv = fresh();
        let mut logits = self.step(&prompt, &mut kv, &cross, frames, 0);
        let mut pos = prompt.len();
        let mut out: Vec<u32> = Vec::new();
        let limit = self.text_ctx / 2;
        while out.len() < limit && pos < self.text_ctx {
            for &id in &self.suppress {
                if let Some(v) = logits.get_mut(id as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
            if out.is_empty() {
                for &id in &self.begin_suppress {
                    if let Some(v) = logits.get_mut(id as usize) {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            for v in logits.iter_mut().skip(self.ids.eot as usize + 1) {
                *v = f32::NEG_INFINITY;
            }
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.ids.eot);
            if next == self.ids.eot {
                break;
            }
            out.push(next);
            logits = self.step(&[next], &mut kv, &cross, frames, pos);
            pos += 1;
        }
        timings.decode += started.elapsed();
        timings.tokens += out.len();
        Ok((self.tokenizer.decode(&out).trim().to_string(), lang.0))
    }

    /// The text of `audio` (mono, 16 kHz), in `language` when given and
    /// in the language heard in the first thirty seconds otherwise.
    pub fn transcribe(&self, audio: &[f32], language: Option<&str>) -> Result<Transcript> {
        let mut timings = Timings::default();
        let mut parts = Vec::new();
        let mut heard: Option<String> = language.map(str::to_string);
        for chunk in audio.chunks(mel::CHUNK_SAMPLES) {
            let (text, lang) = self.window(chunk, heard.as_deref(), &mut timings)?;
            heard.get_or_insert(lang);
            if !text.is_empty() {
                parts.push(text);
            }
        }
        Ok(Transcript {
            text: parts.join(" "),
            language: heard.unwrap_or_default(),
            seconds: audio.len() as f32 / wav::SAMPLE_RATE as f32,
            timings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
        ] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x})");
        }
    }

    #[test]
    fn one_key_attention_returns_its_value() {
        let out = full_attention(&[1.0, 0.0], &[0.5, 0.5], &[3.0, 4.0], 1, 1, 2);
        assert_eq!(out, vec![3.0, 4.0]);
    }
}

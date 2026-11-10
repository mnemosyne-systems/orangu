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

//! The FLUX.2 [klein] diffusion transformer (`flux2_klein`) — diffusers'
//! `Flux2Transformer2DModel`, read from the GGUF the ComfyUI-GGUF tooling
//! writes (`unsloth/FLUX.2-klein-base-4B-GGUF`) in Black Forest Labs' own
//! tensor naming (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`).
//!
//! Like [`super::transformer`] it predicts one **velocity** for a latent
//! picture at noise level `sigma`, conditioned on a prompt's text-encoder
//! states; the graph differs in four ways:
//!
//! - Two kinds of block. The first `double_blocks` are dual-stream — image
//!   and text each through their own fused `qkv`, output projection and
//!   feed-forward, joined in attention (text first). The `single_blocks`
//!   after them run the joined sequence as one stream, with attention and
//!   the feed-forward side by side: one `linear1` makes `q | k | v | mlp`,
//!   one `linear2` reads `attention | activated mlp`.
//! - **Global modulation.** The timestep's `shift, scale, gate` vectors come
//!   from three linears shared by every block — one for the double blocks'
//!   image stream (two sets), one for their text stream, one for the single
//!   blocks — rather than one linear per block. A pass reads them once.
//! - A gated-SiLU feed-forward (`silu(a) * b` over the two halves of the
//!   first projection) and no biases anywhere.
//! - Four-axis rotary positions over the 128-wide head — `(t, row, column,
//!   index)`, sixteen adjacent pairs each, `theta = 2000`: a picture token
//!   sits at `(0, r, c, 0)`, the `i`th text token at `(0, 0, 0, i)`.
//!
//! The text states are three of the encoder's layers side by side (see
//! [`super::qwen3vl::TextModel::forward_layers_masked`]): `txt_in` reads
//! `3 × 2560` for the 4B's Qwen3-4B. The checkpoint has no hyperparameters
//! beyond `general.architecture`; every dimension is read off a tensor.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use super::transformer::{
    StageClock, Stages, TIMESTEP_FREQUENCIES, TransformerConfig, device_head_attention, gated_add,
    head_rms_norm, layer_norm_into, modulate_inplace, silu_inplace, step_attention,
    step_attention_heads, timestep_embedding,
};
use std::time::{Duration, Instant};

/// The rotary base, diffusers' `rope_theta`.
const ROPE_THETA: f64 = 2000.0;
/// Rotary axes over the head: `axes_dims_rope = (32, 32, 32, 32)`.
const ROPE_AXES: usize = 4;
/// LayerNorm and head RMS norm epsilon.
const EPS: f32 = 1e-6;

struct DoubleBlock {
    img_qkv: QuantMatrix,
    img_proj: QuantMatrix,
    img_q_norm: Vec<f32>,
    img_k_norm: Vec<f32>,
    img_mlp_in: QuantMatrix,
    img_mlp_out: QuantMatrix,
    txt_qkv: QuantMatrix,
    txt_proj: QuantMatrix,
    txt_q_norm: Vec<f32>,
    txt_k_norm: Vec<f32>,
    txt_mlp_in: QuantMatrix,
    txt_mlp_out: QuantMatrix,
}

struct SingleBlock {
    linear1: QuantMatrix,
    linear2: QuantMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

/// The timestep-dependent vectors of one pass, keyed by its sigma: under
/// guidance the negative prompt's pass follows the positive one at the
/// same sigma and reuses them.
struct Modulation {
    sigma: f32,
    /// `[6 * dim]`: the double blocks' image `shift, scale, gate` twice.
    img: Vec<f32>,
    /// `[6 * dim]`: the same for their text stream.
    txt: Vec<f32>,
    /// `[3 * dim]`: the single blocks' `shift, scale, gate`.
    single: Vec<f32>,
    /// `[2 * dim]`: the final layer's `shift, scale`.
    out: Vec<f32>,
}

pub struct Flux2Transformer {
    backend: Arc<dyn Backend>,
    /// Shared with the other picture transformers: `n_layer` counts both
    /// kinds of block, `rope_axes` is unused.
    pub config: TransformerConfig,
    /// The feed-forward's hidden width (`mlp_ratio × dim`).
    mlp_hidden: usize,
    img_in: QuantMatrix,
    txt_in: QuantMatrix,
    time_in: QuantMatrix,
    time_out: QuantMatrix,
    mod_img: QuantMatrix,
    mod_txt: QuantMatrix,
    mod_single: QuantMatrix,
    final_mod: QuantMatrix,
    final_linear: QuantMatrix,
    double: Vec<DoubleBlock>,
    single: Vec<SingleBlock>,
    stages: Mutex<Stages>,
    modulation: Mutex<Option<Modulation>>,
    /// How many heads (the last ones) each block's attention runs on the
    /// card, beside the host's.
    head_split: Mutex<HeadSplit>,
    /// The device heads' gathered keys and values, one a pass in flight.
    head_caches: Mutex<Vec<Option<crate::engine::kv_cache::KvCache>>>,
}

/// One pass's attention time by side: the host's heads and the card's,
/// which run at the same time.
#[derive(Default)]
struct HeadTimes {
    host: Duration,
    card: Duration,
}

/// What one forward pass is given.
pub struct ForwardInput<'a> {
    /// `[n_img, in_channels]` patchified latent tokens, row-major over the
    /// `(rows, cols)` grid.
    pub img: &'a [f32],
    pub grid: (usize, usize),
    /// `[n_txt, txt_dim]`: the encoder's stacked layers, padding included.
    pub txt: &'a [f32],
    pub sigma: f32,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl Flux2Transformer {
    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let matrix = |name: &str| -> Result<QuantMatrix> {
            loaded
                .matrix(&format!("{name}.weight"))
                .with_context(|| format!("flux2_klein: loading {name}"))
        };
        let vector = |name: &str| -> Result<Vec<f32>> {
            Ok(loaded
                .tensor(name)
                .with_context(|| format!("flux2_klein: loading {name}"))?
                .0)
        };
        ensure!(
            !loaded.has_tensor("guidance_in.in_layer.weight"),
            "flux2_klein: this checkpoint embeds a guidance scale (FLUX.2 [dev]), which this \
             pipeline does not serve"
        );
        let img_in = matrix("img_in")?;
        let txt_in = matrix("txt_in")?;
        let (dim, in_channels, txt_dim) = (img_in.out_dim, img_in.in_dim, txt_in.in_dim);
        let head_dim = vector("double_blocks.0.img_attn.norm.query_norm.scale")?.len();
        ensure!(
            head_dim > 0 && dim.is_multiple_of(head_dim) && head_dim.is_multiple_of(2 * ROPE_AXES),
            "flux2_klein: stream width {dim} and head width {head_dim} do not fit the rotary axes"
        );
        let count = |prefix: &str, leaf: &str| {
            (0..)
                .take_while(|i| loaded.has_tensor(&format!("{prefix}.{i}.{leaf}")))
                .count()
        };
        let n_double = count("double_blocks", "img_attn.qkv.weight");
        let n_single = count("single_blocks", "linear1.weight");
        ensure!(
            n_double > 0 && n_single > 0,
            "flux2_klein: no double_blocks.N / single_blocks.N tensors"
        );
        let double = (0..n_double)
            .map(|i| {
                let p = format!("double_blocks.{i}");
                Ok(DoubleBlock {
                    img_qkv: matrix(&format!("{p}.img_attn.qkv"))?,
                    img_proj: matrix(&format!("{p}.img_attn.proj"))?,
                    img_q_norm: vector(&format!("{p}.img_attn.norm.query_norm.scale"))?,
                    img_k_norm: vector(&format!("{p}.img_attn.norm.key_norm.scale"))?,
                    img_mlp_in: matrix(&format!("{p}.img_mlp.0"))?,
                    img_mlp_out: matrix(&format!("{p}.img_mlp.2"))?,
                    txt_qkv: matrix(&format!("{p}.txt_attn.qkv"))?,
                    txt_proj: matrix(&format!("{p}.txt_attn.proj"))?,
                    txt_q_norm: vector(&format!("{p}.txt_attn.norm.query_norm.scale"))?,
                    txt_k_norm: vector(&format!("{p}.txt_attn.norm.key_norm.scale"))?,
                    txt_mlp_in: matrix(&format!("{p}.txt_mlp.0"))?,
                    txt_mlp_out: matrix(&format!("{p}.txt_mlp.2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let single = (0..n_single)
            .map(|i| {
                let p = format!("single_blocks.{i}");
                Ok(SingleBlock {
                    linear1: matrix(&format!("{p}.linear1"))?,
                    linear2: matrix(&format!("{p}.linear2"))?,
                    q_norm: vector(&format!("{p}.norm.query_norm.scale"))?,
                    k_norm: vector(&format!("{p}.norm.key_norm.scale"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mlp_hidden = double[0].img_mlp_out.in_dim;
        let model = Self {
            backend,
            config: TransformerConfig {
                dim,
                n_head: dim / head_dim,
                head_dim,
                n_layer: n_double + n_single,
                txt_dim,
                in_channels,
                rope_axes: [0; 3],
                eps: EPS,
            },
            mlp_hidden,
            img_in,
            txt_in,
            time_in: matrix("time_in.in_layer")?,
            time_out: matrix("time_in.out_layer")?,
            mod_img: matrix("double_stream_modulation_img.lin")?,
            mod_txt: matrix("double_stream_modulation_txt.lin")?,
            mod_single: matrix("single_stream_modulation.lin")?,
            final_mod: matrix("final_layer.adaLN_modulation.1")?,
            final_linear: matrix("final_layer.linear")?,
            double,
            single,
            stages: Mutex::new(Stages::default()),
            modulation: Mutex::new(None),
            head_split: Mutex::new(HeadSplit { heads: None }),
            head_caches: Mutex::new(Vec::new()),
        };
        model.check_shapes()?;
        let device = model.double.first().map_or(0, |b| b.img_qkv.device());
        if let Some(vulkan) = model.backend.as_wgpu_on(device) {
            vulkan.hold_results_up_to(result_pool_bytes());
        }
        Ok(model)
    }

    fn check_shapes(&self) -> Result<()> {
        let (d, m) = (self.config.dim, self.mlp_hidden);
        ensure!(
            self.time_in.in_dim == TIMESTEP_FREQUENCIES
                && self.time_out.out_dim == d
                && self.mod_img.out_dim == 6 * d
                && self.mod_txt.out_dim == 6 * d
                && self.mod_single.out_dim == 3 * d
                && self.final_mod.out_dim == 2 * d
                && self.final_linear.out_dim == self.config.in_channels,
            "flux2_klein: the embedding, modulation and output projections do not match the \
             stream width {d}"
        );
        for (i, b) in self.double.iter().enumerate() {
            ensure!(
                b.img_qkv.out_dim == 3 * d
                    && b.txt_qkv.out_dim == 3 * d
                    && b.img_proj.out_dim == d
                    && b.img_mlp_in.out_dim == 2 * m
                    && b.txt_mlp_in.out_dim == 2 * m
                    && b.img_mlp_out.in_dim == m
                    && b.txt_mlp_out.in_dim == m,
                "flux2_klein: double_blocks.{i} does not match the stream width {d}"
            );
        }
        for (i, b) in self.single.iter().enumerate() {
            ensure!(
                b.linear1.out_dim == 3 * d + 2 * m && b.linear2.in_dim == d + m,
                "flux2_klein: single_blocks.{i} does not match the stream width {d}"
            );
        }
        Ok(())
    }

    /// The first double block's image feed-forward input projection — the
    /// widest linear of a step, which `calibrate` times.
    pub const CALIBRATION_TENSOR: &'static str = "double_blocks.0.img_mlp.0.weight";

    pub fn take_stages(&self) -> Stages {
        std::mem::take(
            &mut *self
                .stages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// Whether the linears run on a device — what lets a guided step's two
    /// passes overlap, one pass's host attention beside the other's device
    /// calls.
    pub fn on_device(&self) -> bool {
        !self.backend.is_cpu()
    }

    /// Drops the device heads' key and value copies and the card's
    /// attention buffers — before the VAE takes the card.
    pub fn release_pass_scratch(&self) {
        self.head_caches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let devices = self.double.iter().map(|b| b.img_qkv.device());
        let devices = devices.chain(self.single.iter().map(|b| b.linear1.device()));
        let mut seen = Vec::new();
        for device in devices {
            if !seen.contains(&device) {
                seen.push(device);
                if let Some(vulkan) = self.backend.as_wgpu_on(device) {
                    vulkan.release_prefill_attention_scratch();
                }
            }
        }
    }

    /// How many heads the card takes this pass: `ORANGU_FLUX2_DEVICE_HEADS`
    /// when a count, else [`HeadSplit`]'s.
    fn device_heads(&self) -> usize {
        let n_head = self.config.n_head;
        fixed_device_heads()
            .unwrap_or_else(|| {
                self.head_split
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .heads(n_head)
            })
            .min(n_head)
    }

    /// The joint attention of a block whose linears live where `w` does:
    /// the last `heads` heads on that device's card (when it is one) while
    /// the host computes the others, each side's time added to `times`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        w: &QuantMatrix,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n: usize,
        heads: usize,
        scale: f32,
        times: &mut HeadTimes,
    ) -> Vec<f32> {
        let c = &self.config;
        let vulkan = if heads > 0 {
            self.backend.as_wgpu_on(w.device())
        } else {
            None
        };
        let Some(vulkan) = vulkan else {
            return on_attention_pool(self.on_device(), || {
                step_attention(q, n, k, v, n, c.n_head, c.head_dim, scale, None)
            });
        };
        let split = c.n_head - heads;
        let mut cache = self
            .head_caches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
            .flatten();
        let held = &mut cache;
        let (mut out, card_out, host, card) = std::thread::scope(|scope| {
            // On the lane's own threads: the gather and the upload's
            // conversion queued on the global pool would wait behind the
            // host's heads.
            let lane = scope.spawn(move || {
                super::transformer21::lane_pool().install(|| {
                    let ran = Instant::now();
                    let out = device_head_attention(
                        vulkan,
                        held,
                        q,
                        k,
                        v,
                        (n, n),
                        (c.dim, c.head_dim),
                        split..c.n_head,
                        scale,
                    );
                    (out, ran.elapsed())
                })
            });
            let ran = Instant::now();
            let out = on_attention_pool(true, || {
                step_attention_heads(q, n, k, v, n, c.n_head, c.head_dim, scale, 0..split)
            });
            let host = ran.elapsed();
            let (card_out, card) = lane
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            (out, card_out, host, card)
        });
        self.head_caches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(cache);
        let (column, width) = (split * c.head_dim, heads * c.head_dim);
        out.par_chunks_mut(c.dim)
            .zip(card_out.par_chunks(width))
            .with_min_len(64)
            .for_each(|(row, dev)| row[column..column + width].copy_from_slice(dev));
        self.give(card_out);
        times.host += host;
        times.card += card;
        out
    }

    /// `len` zeros, in memory a finished buffer handed back — the wide
    /// buffers of a pass are hundreds of megabytes, and fresh ones are
    /// mappings the kernel faults in and zeroes page by page.
    fn take(&self, len: usize) -> Vec<f32> {
        let mut v = self.backend.take_scratch(len);
        v.resize(len, 0.0);
        v
    }

    /// Hands a finished wide buffer back for [`Self::take`] and the next
    /// device result (`Backend::recycle`).
    fn give(&self, v: Vec<f32>) {
        self.backend.recycle(v);
    }

    /// `[a | b]` as rows, in a recycled buffer.
    fn concat(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = self.backend.take_scratch(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        out
    }

    fn linear(&self, x: &[f32], n: usize, w: &QuantMatrix) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * w.in_dim);
        self.backend
            .matmul_batch(&[MatmulOp { x, n_tokens: n, w }])
            .pop()
            .expect("one op in, one result out")
    }

    fn modulation(&self, sigma: f32) -> Modulation {
        let cached = self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .filter(|m| m.sigma == sigma);
        if let Some(m) = cached {
            return m;
        }
        let mut h = self.linear(&timestep_embedding(sigma), 1, &self.time_in);
        silu_inplace(&mut h);
        let mut vec = self.linear(&h, 1, &self.time_out);
        // Every modulation reads `silu(vec)`.
        silu_inplace(&mut vec);
        Modulation {
            sigma,
            img: self.linear(&vec, 1, &self.mod_img),
            txt: self.linear(&vec, 1, &self.mod_txt),
            single: self.linear(&vec, 1, &self.mod_single),
            out: self.linear(&vec, 1, &self.final_mod),
        }
    }

    /// One velocity prediction: `[n_img, in_channels]`, the layout the
    /// latent tokens came in.
    pub fn forward(&self, input: &ForwardInput<'_>) -> Result<Vec<f32>> {
        let c = &self.config;
        let (d, m) = (c.dim, self.mlp_hidden);
        let (rows, cols) = input.grid;
        let n_img = rows * cols;
        ensure!(
            input.img.len() == n_img * c.in_channels,
            "flux2_klein: {} latent values for a {rows}x{cols} grid of {}-wide tokens",
            input.img.len(),
            c.in_channels
        );
        ensure!(
            !input.txt.is_empty() && input.txt.len().is_multiple_of(c.txt_dim),
            "flux2_klein: text states are not rows of {}",
            c.txt_dim
        );
        let n_txt = input.txt.len() / c.txt_dim;
        let n = n_txt + n_img;
        let mut clock = StageClock::start();

        let mut x = self.linear(input.img, n_img, &self.img_in);
        let mut t = self.linear(input.txt, n_txt, &self.txt_in);
        clock.lap(|s| &mut s.other);
        let modulation = self.modulation(input.sigma);
        clock.lap(|s| &mut s.modulation);
        let split = |v: &[f32], k: usize| -> Vec<Vec<f32>> {
            (0..k).map(|i| v[i * d..(i + 1) * d].to_vec()).collect()
        };
        let im = split(&modulation.img, 6);
        let tm = split(&modulation.txt, 6);
        let sm = split(&modulation.single, 3);

        let rope = rope_table(c.head_dim, (rows, cols), n_txt);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut normed = Vec::new();
        let heads = self.device_heads();
        let mut times = HeadTimes::default();
        clock.lap(|s| &mut s.other);

        for block in &self.double {
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            layer_norm_into(&mut normed, &x, d, EPS);
            modulate_inplace(&mut normed, &im[0], &im[1], d);
            clock.lap(|s| &mut s.other);
            let img_qkv = self.linear(&normed, n_img, &block.img_qkv);
            layer_norm_into(&mut normed, &t, d, EPS);
            modulate_inplace(&mut normed, &tm[0], &tm[1], d);
            clock.lap(|s| &mut s.other);
            let txt_qkv = self.linear(&normed, n_txt, &block.txt_qkv);
            clock.lap(|s| &mut s.qkv);
            let take = |len| self.take(len);
            let (mut iq, mut ik, iv) = split_qkv(&img_qkv, n_img, d, 3 * d, &take);
            let (mut tq, mut tk, tv) = split_qkv(&txt_qkv, n_txt, d, 3 * d, &take);
            self.give(img_qkv);
            self.give(txt_qkv);
            head_rms_norm(&mut iq, &block.img_q_norm, c.head_dim, EPS);
            head_rms_norm(&mut ik, &block.img_k_norm, c.head_dim, EPS);
            head_rms_norm(&mut tq, &block.txt_q_norm, c.head_dim, EPS);
            head_rms_norm(&mut tk, &block.txt_k_norm, c.head_dim, EPS);
            let mut q = self.concat(&tq, &iq);
            let mut k = self.concat(&tk, &ik);
            let v = self.concat(&tv, &iv);
            for buffer in [iq, ik, iv, tq, tk, tv] {
                self.give(buffer);
            }
            apply_rope(&mut q, d, c.head_dim, &rope);
            apply_rope(&mut k, d, c.head_dim, &rope);
            let attn = self.attention(&block.img_qkv, &q, &k, &v, n, heads, scale, &mut times);
            for buffer in [q, k, v] {
                self.give(buffer);
            }
            clock.lap(|s| &mut s.attention);
            let txt_attn = self.linear(&attn[..n_txt * d], n_txt, &block.txt_proj);
            let img_attn = self.linear(&attn[n_txt * d..], n_img, &block.img_proj);
            self.give(attn);
            clock.lap(|s| &mut s.out);
            gated_add(&mut x, &img_attn, &im[2], d);
            gated_add(&mut t, &txt_attn, &tm[2], d);
            self.give(img_attn);
            self.give(txt_attn);

            layer_norm_into(&mut normed, &x, d, EPS);
            modulate_inplace(&mut normed, &im[3], &im[4], d);
            clock.lap(|s| &mut s.other);
            let gate_up = self.linear(&normed, n_img, &block.img_mlp_in);
            let h = swiglu(&gate_up, m, &take);
            self.give(gate_up);
            let mlp = self.linear(&h, n_img, &block.img_mlp_out);
            self.give(h);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut x, &mlp, &im[5], d);
            self.give(mlp);
            layer_norm_into(&mut normed, &t, d, EPS);
            modulate_inplace(&mut normed, &tm[3], &tm[4], d);
            clock.lap(|s| &mut s.other);
            let gate_up = self.linear(&normed, n_txt, &block.txt_mlp_in);
            let h = swiglu(&gate_up, m, &take);
            self.give(gate_up);
            let mlp = self.linear(&h, n_txt, &block.txt_mlp_out);
            self.give(h);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut t, &mlp, &tm[5], d);
            self.give(mlp);
        }

        // One stream from here on: `[text | image]`.
        let joint = self.concat(&t, &x);
        self.give(t);
        self.give(x);
        let mut x = joint;
        let wide = 3 * d + 2 * m;
        for block in &self.single {
            if super::cancelled(input.cancel) {
                bail!("cancelled");
            }
            layer_norm_into(&mut normed, &x, d, EPS);
            modulate_inplace(&mut normed, &sm[0], &sm[1], d);
            clock.lap(|s| &mut s.other);
            let h = self.linear(&normed, n, &block.linear1);
            clock.lap(|s| &mut s.qkv);
            let (mut q, mut k, v) = split_qkv(&h, n, d, wide, &|len| self.take(len));
            head_rms_norm(&mut q, &block.q_norm, c.head_dim, EPS);
            head_rms_norm(&mut k, &block.k_norm, c.head_dim, EPS);
            apply_rope(&mut q, d, c.head_dim, &rope);
            apply_rope(&mut k, d, c.head_dim, &rope);
            let attn = self.attention(&block.linear1, &q, &k, &v, n, heads, scale, &mut times);
            for buffer in [q, k, v] {
                self.give(buffer);
            }
            clock.lap(|s| &mut s.attention);
            // `[attention | silu(a) * b]` for `linear2`.
            let mut joined = self.take(n * (d + m));
            joined
                .par_chunks_mut(d + m)
                .zip(attn.par_chunks(d))
                .zip(h.par_chunks(wide))
                .for_each(|((out, a), row)| {
                    out[..d].copy_from_slice(a);
                    let (gate, up) = row[3 * d..].split_at(m);
                    for ((o, g), u) in out[d..].iter_mut().zip(gate).zip(up) {
                        *o = crate::engine::tensor::silu(*g) * u;
                    }
                });
            self.give(attn);
            self.give(h);
            clock.lap(|s| &mut s.other);
            let out = self.linear(&joined, n, &block.linear2);
            self.give(joined);
            clock.lap(|s| &mut s.mlp);
            gated_add(&mut x, &out, &sm[2], d);
            self.give(out);
        }

        // `LastLayer`: `shift, scale = chunk(2)` over the image rows.
        let (shift, scale_v) = modulation.out.split_at(d);
        layer_norm_into(&mut normed, &x[n_txt * d..], d, EPS);
        modulate_inplace(&mut normed, shift, scale_v, d);
        let out = self.linear(&normed, n_img, &self.final_linear);
        self.give(x);
        clock.lap(|s| &mut s.other);
        *self
            .modulation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(modulation);
        if fixed_device_heads().is_none() && times.card > Duration::ZERO {
            let mut split = self
                .head_split
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            split.observe(heads, c.n_head, &times);
            if crate::engine::env::flag_on("ORANGU_IMAGE_HEADS_TRACE") {
                eprintln!(
                    "orangu-server: [image] pass with {heads} of {} heads on the device: host \
                     {:.2}s, card {:.2}s, next pass {}",
                    c.n_head,
                    times.host.as_secs_f64(),
                    times.card.as_secs_f64(),
                    split.heads(c.n_head)
                );
            }
        }
        self.stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .add(&clock.stages);
        Ok(out)
    }
}

/// `q`, `k` and `v` out of rows of `stride` values whose first `3 * dim`
/// are `q | k | v` — a fused `qkv` projection, or `linear1`'s leading part.
/// Runs `f` — a pass's host attention — on a pool of its own, as wide as
/// the global one, when the transformer is on a device: a guided step's
/// other pass meanwhile runs its norms, adds and device calls' staging on
/// the global pool, where they would otherwise queue behind this
/// attention's tasks; the threads of the two share the cores instead.
/// `ORANGU_FLUX2_ATTENTION_POOL=0` runs it on the caller's pool.
fn on_attention_pool<T: Send>(on_device: bool, f: impl FnOnce() -> T + Send) -> T {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    if !on_device
        || !*ON.get_or_init(|| {
            crate::engine::env::flag_on_unless_disabled("ORANGU_FLUX2_ATTENTION_POOL")
        })
    {
        return f();
    }
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .thread_name(|i| format!("image-attention-{i}"))
            .build()
            .expect("the image attention's thread pool")
    })
    .install(f)
}

/// What the device results' recycle pool may keep while this transformer
/// is served: a pass's wide buffers, two passes in flight — 4 GiB, or an
/// eighth of the machine's memory where that is less.
fn result_pool_bytes() -> usize {
    const MOST: usize = 4 << 30;
    #[cfg(unix)]
    {
        // Safety: `sysconf` reads a system constant and has no
        // preconditions.
        let (pages, page) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        if pages > 0 && page > 0 {
            return (pages as usize).saturating_mul(page as usize).min(8 * MOST) / 8;
        }
    }
    MOST / 4
}

/// `ORANGU_FLUX2_DEVICE_HEADS`, read once: a fixed count of heads for the
/// card (`0` keeps attention on the host), or `None` — unset or `auto` —
/// for [`HeadSplit`] to balance.
fn fixed_device_heads() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("ORANGU_FLUX2_DEVICE_HEADS") {
        Ok(v) if v.trim().eq_ignore_ascii_case("auto") => None,
        Ok(v) => Some(v.trim().parse().unwrap_or(0)),
        Err(_) => None,
    })
}

/// Splits a block's heads between the host and the card so both sides of
/// the attention end together: from a pass's time per host head `a` and
/// per card head `c`, the next pass gives the card `n · a / (a + c)` heads
/// — at least one, so the card's time stays measured, and at least one
/// left to the host. The first pass gives it a quarter.
struct HeadSplit {
    heads: Option<usize>,
}

impl HeadSplit {
    fn heads(&self, n_head: usize) -> usize {
        self.heads.unwrap_or(n_head / 4)
    }

    fn observe(&mut self, used: usize, n_head: usize, times: &HeadTimes) {
        if used == 0 || used >= n_head {
            return;
        }
        let a = times.host.as_secs_f64() / (n_head - used) as f64;
        let c = times.card.as_secs_f64() / used as f64;
        if a <= 0.0 || c <= 0.0 {
            return;
        }
        let target = n_head as f64 * a / (a + c);
        self.heads = Some((target.round() as usize).clamp(1, n_head.saturating_sub(1).max(1)));
    }
}

/// The first three `dim`-wide columns of rows `stride` wide, as three
/// `[n, dim]` buffers from `take` (`len` zeros).
fn split_qkv(
    rows: &[f32],
    n: usize,
    dim: usize,
    stride: usize,
    take: &dyn Fn(usize) -> Vec<f32>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(rows.len(), n * stride);
    let mut q = take(n * dim);
    let mut k = take(n * dim);
    let mut v = take(n * dim);
    q.par_chunks_mut(dim)
        .zip(k.par_chunks_mut(dim))
        .zip(v.par_chunks_mut(dim))
        .zip(rows.par_chunks(stride))
        .for_each(|(((q, k), v), row)| {
            q.copy_from_slice(&row[..dim]);
            k.copy_from_slice(&row[dim..2 * dim]);
            v.copy_from_slice(&row[2 * dim..3 * dim]);
        });
    (q, k, v)
}

/// The gated-SiLU activation over rows of `2 * hidden`: `silu(a) * b`
/// for the two halves, into a buffer from `take`.
fn swiglu(rows: &[f32], hidden: usize, take: &dyn Fn(usize) -> Vec<f32>) -> Vec<f32> {
    let mut out = take(rows.len() / 2);
    out.par_chunks_mut(hidden)
        .zip(rows.par_chunks(2 * hidden))
        .for_each(|(o, row)| {
            let (gate, up) = row.split_at(hidden);
            for ((o, g), u) in o.iter_mut().zip(gate).zip(up) {
                *o = crate::engine::tensor::silu(*g) * u;
            }
        });
    out
}

/// `(cos, sin)` for every pair of the head, for every row of the joint
/// sequence `[text | image]`: axis `a` takes pairs `a * P .. (a + 1) * P`
/// (`P = head_dim / 8`), pair `j` of an axis turning at `theta^(-2j / 2P)`
/// per unit of position — BFL's `EmbedND` over `rope(pos, 2P, theta)`.
fn rope_table(head_dim: usize, (rows, cols): (usize, usize), n_txt: usize) -> Vec<(f32, f32)> {
    let pairs = head_dim / 2;
    let per_axis = pairs / ROPE_AXES;
    let freqs: Vec<f64> = (0..per_axis)
        .map(|j| 1.0 / ROPE_THETA.powf(2.0 * j as f64 / (2 * per_axis) as f64))
        .collect();
    let mut table = Vec::with_capacity((n_txt + rows * cols) * pairs);
    let mut push = |pos: [usize; ROPE_AXES]| {
        for p in pos {
            for f in &freqs {
                let a = p as f64 * f;
                table.push((a.cos() as f32, a.sin() as f32));
            }
        }
    };
    for i in 0..n_txt {
        push([0, 0, 0, i]);
    }
    for r in 0..rows {
        for col in 0..cols {
            push([0, r, col, 0]);
        }
    }
    table
}

/// Rotates every head of every row of `x` (`[n, dim]`) by that row's
/// angles, as a complex multiply on adjacent pairs.
fn apply_rope(x: &mut [f32], dim: usize, head_dim: usize, table: &[(f32, f32)]) {
    let pairs = head_dim / 2;
    x.par_chunks_mut(dim)
        .zip(table.par_chunks(pairs))
        .for_each(|(row, angles)| {
            for head in row.chunks_mut(head_dim) {
                for (pair, &(cos, sin)) in head.chunks_mut(2).zip(angles) {
                    let (a, b) = (pair[0], pair[1]);
                    pair[0] = a * cos - b * sin;
                    pair[1] = a * sin + b * cos;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A text token turns only on the last axis, by its index; a picture
    /// token on the row and column axes; the first axis never turns.
    #[test]
    fn rope_positions_are_index_for_text_and_row_column_for_the_picture() {
        let table = rope_table(128, (2, 3), 2);
        let pairs = 64;
        assert_eq!(table.len(), (2 + 6) * pairs);
        let at = |row: usize, pair: usize| table[row * pairs + pair];
        let f = |j: usize| 1.0 / ROPE_THETA.powf(2.0 * j as f64 / 32.0);
        // Text token 1: axes 0..3 still, axis 3 at position 1.
        assert_eq!(at(1, 0), (1.0, 0.0));
        assert_eq!(at(1, 16), (1.0, 0.0));
        assert!((at(1, 48 + 2).1 - (f(2)).sin() as f32).abs() < 1e-6);
        // Picture token (row 1, column 2) is joint row 2 + 1 * 3 + 2.
        let p = 2 + 3 + 2;
        assert_eq!(at(p, 0), (1.0, 0.0));
        assert!((at(p, 16 + 1).1 - (f(1)).sin() as f32).abs() < 1e-6);
        assert!((at(p, 32 + 1).1 - (2.0 * f(1)).sin() as f32).abs() < 1e-6);
        assert_eq!(at(p, 48 + 5), (1.0, 0.0));
    }

    #[test]
    fn rope_rotates_adjacent_pairs() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
        let a = 0.5f32;
        apply_rope(&mut x, 4, 4, &[(a.cos(), a.sin()), (a.cos(), a.sin())]);
        assert!((x[0] - a.cos()).abs() < 1e-6 && (x[1] - a.sin()).abs() < 1e-6);
        assert!((x[2] + a.sin()).abs() < 1e-6 && (x[3] - a.cos()).abs() < 1e-6);
    }

    #[test]
    fn head_split_ends_both_sides_together() {
        let mut split = HeadSplit { heads: None };
        assert_eq!(split.heads(24), 6);
        // 18 host heads in 9 s (0.5 s each), 6 card heads in 1.5 s (0.25 s
        // each): the card takes two thirds.
        let times = HeadTimes {
            host: Duration::from_secs(9),
            card: Duration::from_millis(1500),
        };
        split.observe(6, 24, &times);
        assert_eq!(split.heads(24), 16);
        // A slow card still keeps one head.
        let slow = HeadTimes {
            host: Duration::from_millis(10),
            card: Duration::from_secs(100),
        };
        split.observe(16, 24, &slow);
        assert_eq!(split.heads(24), 1);
    }

    #[test]
    fn swiglu_gates_the_second_half_by_the_first() {
        let out = swiglu(&[0.0, 2.0, 5.0, 7.0], 2, &|len| vec![0.0; len]);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - crate::engine::tensor::silu(2.0) * 7.0).abs() < 1e-6);
    }

    #[test]
    fn qkv_rows_split_into_thirds_past_a_wider_stride() {
        let rows: Vec<f32> = (0..2 * 8).map(|v| v as f32).collect();
        let (q, k, v) = split_qkv(&rows, 2, 2, 8, &|len| vec![0.0; len]);
        assert_eq!(q, vec![0.0, 1.0, 8.0, 9.0]);
        assert_eq!(k, vec![2.0, 3.0, 10.0, 11.0]);
        assert_eq!(v, vec![4.0, 5.0, 12.0, 13.0]);
    }
}

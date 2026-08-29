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

//! The architecture, forward and backward.
//!
//! One dense decoder block, repeated: RMSNorm, grouped-query attention with
//! a per-head RMSNorm on Q and K before the rotation, RMSNorm, and a SwiGLU
//! feed-forward — the `qwen3` shape, which is the strongest thing the
//! inference side serves on its fully-supported dense path. Its two
//! departures from the plainer `llama` block both earn their place here:
//! the Q/K norms are what keep attention logits from drifting during a
//! from-scratch run, and grouped-query attention is what keeps the KV cache
//! affordable at the context lengths this tool declares.
//!
//! Everything is `f32` on the CPU. There is no autodiff framework
//! underneath — each operation's backward is written out beside its
//! forward, which is why `gradients_match_finite_differences` in this
//! module's tests is not a nicety: it is the only thing standing between a
//! subtly wrong derivative and a training run that quietly learns nothing.
//!
//! **Activations are recomputed, not stored.** The forward pass keeps only
//! each block's *input* — one `[tokens, hidden]` buffer per block — and the
//! backward pass re-runs a block's forward to get the intermediates it
//! needs. That trades one extra forward pass for an activation footprint
//! that does not grow with the feed-forward width, which is what makes a
//! long sequence fit at all.

use crate::aligned::Aligned;
use rayon::prelude::*;
use std::f32::consts::PI;
use wide::f32x8;

/// A named training size. The four the guide's staged plan uses, sharing
/// one vocabulary so a tokenizer trained once carries across all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub key: &'static str,
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Peak learning rate. Smaller models take — and need — a larger one.
    pub peak_lr_micro: u32,
}

impl Size {
    pub fn peak_lr(&self) -> f32 {
        self.peak_lr_micro as f32 * 1e-6
    }
}

/// Every one of these has a `hidden` and an `ffn` that 256 divides, and
/// that is a requirement rather than a preference: 256 is the K-quants'
/// super-block, and a row length it does not divide cannot be a K-quant at
/// all. The feed-forward *down* projection is the one at risk, because its
/// row length is `ffn` rather than `hidden` — at `ffn: 688` every block's
/// `ffn_down` fell back to `f16`, which is a bigger file and a different
/// mixture than the name on it promises.
///
/// `ffn` is otherwise the usual `8/3 * hidden` of a SwiGLU network, rounded
/// up to the next multiple of 256.
pub const SIZES: &[Size] = &[
    Size {
        key: "smoke",
        hidden: 256,
        ffn: 768,
        layers: 4,
        heads: 4,
        kv_heads: 4,
        head_dim: 64,
        peak_lr_micro: 1000,
    },
    Size {
        key: "0.5b",
        hidden: 1280,
        ffn: 3584,
        layers: 24,
        heads: 10,
        kv_heads: 5,
        head_dim: 128,
        peak_lr_micro: 400,
    },
    Size {
        key: "1b",
        hidden: 1536,
        ffn: 5632,
        layers: 28,
        heads: 12,
        kv_heads: 6,
        head_dim: 128,
        peak_lr_micro: 300,
    },
    Size {
        key: "2b",
        hidden: 2048,
        ffn: 8192,
        layers: 30,
        heads: 16,
        kv_heads: 8,
        head_dim: 128,
        peak_lr_micro: 250,
    },
];

pub fn size_named(key: &str) -> Option<&'static Size> {
    SIZES.iter().find(|s| s.key.eq_ignore_ascii_case(key))
}

/// Everything needed to build, train, and describe one model.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The context length the finished file declares.
    pub context: usize,
    pub rope_base: f32,
    pub eps: f32,
}

impl Config {
    pub fn from_size(size: &Size, vocab: usize, context: usize) -> Self {
        Config {
            vocab,
            hidden: size.hidden,
            ffn: size.ffn,
            layers: size.layers,
            heads: size.heads,
            kv_heads: size.kv_heads,
            head_dim: size.head_dim,
            context,
            rope_base: rope_base_for(context),
            eps: 1e-5,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// How many query heads share one key/value head.
    pub fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    pub fn parameters(&self) -> usize {
        Layout::new(self).total
    }
}

/// The rotation base for a declared context length.
///
/// The base is what sets the longest wavelength in the rotation, so it has
/// to grow with the context or the highest-index pairs stop separating
/// distant positions at all. `1e6` is the value the reference sizing uses
/// at 8192 tokens; beyond that it scales with the extension factor.
pub fn rope_base_for(context: usize) -> f32 {
    const REFERENCE_CONTEXT: usize = 8192;
    const REFERENCE_BASE: f32 = 1.0e6;
    if context <= REFERENCE_CONTEXT {
        REFERENCE_BASE
    } else {
        REFERENCE_BASE * (context as f32 / REFERENCE_CONTEXT as f32)
    }
}

/// One parameter tensor's place in the flat buffer, and its GGUF shape.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub offset: usize,
    /// GGUF dimension order: `ne0` (row length) first.
    pub dims: Vec<u64>,
}

impl TensorSpec {
    pub fn len(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Whether this is a matrix that a quantized file quantizes — the 1-D
    /// norms stay `f32` in every file format this writes.
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
}

/// Per-block parameter offsets, so the hot paths index rather than search.
#[derive(Debug, Clone, Copy)]
pub struct LayerOffsets {
    pub attn_norm: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    pub ffn_norm: usize,
    pub w_gate: usize,
    pub w_up: usize,
    pub w_down: usize,
}

/// Where every parameter lives in one flat `f32` buffer.
///
/// One buffer, not a tree of tensors: the optimizer, the gradient
/// accumulator, and the checkpoint are then all a single contiguous array
/// of the same length, and none of them needs to know the architecture.
#[derive(Debug, Clone)]
pub struct Layout {
    pub specs: Vec<TensorSpec>,
    pub tok_embd: usize,
    pub layers: Vec<LayerOffsets>,
    pub output_norm: usize,
    pub output: usize,
    pub total: usize,
}

impl Layout {
    pub fn new(cfg: &Config) -> Self {
        let mut specs = Vec::new();
        let mut at = 0usize;
        let push = |specs: &mut Vec<TensorSpec>, at: &mut usize, name: String, dims: Vec<u64>| {
            let offset = *at;
            let len: usize = dims.iter().map(|&d| d as usize).product();
            specs.push(TensorSpec { name, offset, dims });
            *at += len;
            offset
        };

        let (h, q, kv, f) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.ffn);
        let tok_embd = push(
            &mut specs,
            &mut at,
            "token_embd.weight".into(),
            vec![h as u64, cfg.vocab as u64],
        );

        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = |specs: &mut Vec<TensorSpec>, at: &mut usize, suffix: &str, dims: Vec<u64>| {
                let offset = *at;
                let len: usize = dims.iter().map(|&d| d as usize).product();
                specs.push(TensorSpec {
                    name: format!("blk.{l}.{suffix}"),
                    offset,
                    dims,
                });
                *at += len;
                offset
            };
            layers.push(LayerOffsets {
                attn_norm: p(&mut specs, &mut at, "attn_norm.weight", vec![h as u64]),
                wq: p(
                    &mut specs,
                    &mut at,
                    "attn_q.weight",
                    vec![h as u64, q as u64],
                ),
                wk: p(
                    &mut specs,
                    &mut at,
                    "attn_k.weight",
                    vec![h as u64, kv as u64],
                ),
                wv: p(
                    &mut specs,
                    &mut at,
                    "attn_v.weight",
                    vec![h as u64, kv as u64],
                ),
                wo: p(
                    &mut specs,
                    &mut at,
                    "attn_output.weight",
                    vec![q as u64, h as u64],
                ),
                q_norm: p(
                    &mut specs,
                    &mut at,
                    "attn_q_norm.weight",
                    vec![cfg.head_dim as u64],
                ),
                k_norm: p(
                    &mut specs,
                    &mut at,
                    "attn_k_norm.weight",
                    vec![cfg.head_dim as u64],
                ),
                ffn_norm: p(&mut specs, &mut at, "ffn_norm.weight", vec![h as u64]),
                w_gate: p(
                    &mut specs,
                    &mut at,
                    "ffn_gate.weight",
                    vec![h as u64, f as u64],
                ),
                w_up: p(
                    &mut specs,
                    &mut at,
                    "ffn_up.weight",
                    vec![h as u64, f as u64],
                ),
                w_down: p(
                    &mut specs,
                    &mut at,
                    "ffn_down.weight",
                    vec![f as u64, h as u64],
                ),
            });
        }

        let output_norm = push(
            &mut specs,
            &mut at,
            "output_norm.weight".into(),
            vec![h as u64],
        );
        let output = push(
            &mut specs,
            &mut at,
            "output.weight".into(),
            vec![h as u64, cfg.vocab as u64],
        );

        Layout {
            specs,
            tok_embd,
            layers,
            output_norm,
            output,
            total: at,
        }
    }
}

/// A model: its shape, its layout, and one flat parameter buffer.
pub struct Model {
    pub cfg: Config,
    pub layout: Layout,
    /// One flat buffer, aligned to a cache line so that every eight-float
    /// load the kernels make is vector-aligned — see [`crate::aligned`].
    pub params: Aligned,
}

impl Model {
    /// Random initial weights.
    ///
    /// Normal(0, 0.02) throughout, with the two projections that write into
    /// the residual stream (`attn_output`, `ffn_down`) scaled down by
    /// `1/sqrt(2 * layers)`. Without that scaling the residual variance
    /// grows with depth and the first few hundred steps are spent undoing
    /// it. Norm weights start at one, which is the identity.
    pub fn new(cfg: Config, seed: u64) -> Self {
        let layout = Layout::new(&cfg);
        let mut params = Aligned::zeros(layout.total);
        let mut rng = Rng::new(seed);
        let residual_scale = 1.0 / ((2 * cfg.layers) as f32).sqrt();

        for spec in &layout.specs {
            let slice = &mut params[spec.offset..spec.offset + spec.len()];
            if !spec.is_matrix() {
                slice.fill(1.0);
                continue;
            }
            let std = if spec.name.ends_with("attn_output.weight")
                || spec.name.ends_with("ffn_down.weight")
            {
                0.02 * residual_scale
            } else {
                0.02
            };
            for value in slice.iter_mut() {
                *value = rng.normal() * std;
            }
        }
        Model {
            cfg,
            layout,
            params,
        }
    }

    fn tensor(&self, offset: usize, len: usize) -> &[f32] {
        &self.params[offset..offset + len]
    }

    /// Runs the whole network over one sequence and returns the mean
    /// cross-entropy of predicting `targets`, adding this sequence's
    /// gradients into `grads` when one is supplied.
    ///
    /// `tokens` and `targets` are the same length; `targets[t]` is the token
    /// that follows `tokens[t]`.
    pub fn forward_backward(
        &self,
        tokens: &[u32],
        targets: &[u32],
        mut grads: Option<&mut [f32]>,
    ) -> f32 {
        let cfg = &self.cfg;
        let t = tokens.len();
        debug_assert_eq!(t, targets.len());
        let h = cfg.hidden;

        // The only activation kept per block: its input. Everything else is
        // recomputed on the way back down.
        let mut inputs: Vec<Aligned> = Vec::with_capacity(cfg.layers + 1);
        let mut x = Aligned::zeros(t * h);
        let embd = self.tensor(self.layout.tok_embd, cfg.vocab * h);
        for (i, &token) in tokens.iter().enumerate() {
            let row = token as usize * h;
            x[i * h..(i + 1) * h].copy_from_slice(&embd[row..row + h]);
        }

        for l in 0..cfg.layers {
            inputs.push(x.clone());
            let cache = self.layer_forward(l, &x, t, false);
            x = cache.out;
        }
        inputs.push(x.clone());

        // Final norm, then the vocabulary projection and the loss. Both are
        // done in row chunks: the full `[tokens, vocab]` logit matrix is the
        // largest single allocation in the whole pass, and nothing needs it
        // all at once.
        let final_norm = self.tensor(self.layout.output_norm, h);
        let (normed, final_rms) = rmsnorm_forward(&x, final_norm, t, h, cfg.eps);
        let head = self.tensor(self.layout.output, cfg.vocab * h);

        let mut loss = 0.0f64;
        let mut d_normed = Aligned::zeros(t * h);
        let inv = 1.0 / t as f32;
        // How many tokens' logits to hold at once.
        //
        // A fixed 32 was a memory bound wearing the wrong units: it caps
        // the buffer in *rows*, so a small vocabulary got a 512 KB buffer
        // and sixteen sequential passes over the largest matmul in the
        // network, each only four tiles wide — four of sixteen threads
        // busy. Budgeting in bytes instead gives the whole sequence in one
        // pass at this size, and still chunks a 32,768-token vocabulary at
        // a long sequence, which is what the bound was for.
        const LOGIT_BUDGET: usize = 32 << 20;
        let rows_per_pass = (LOGIT_BUDGET / (cfg.vocab * 4)).max(1).min(t);
        let mut logits = Aligned::zeros(rows_per_pass * cfg.vocab);

        for start in (0..t).step_by(rows_per_pass) {
            let rows = rows_per_pass.min(t - start);
            let logits = &mut logits[..rows * cfg.vocab];
            matmul(
                logits,
                &normed[start * h..(start + rows) * h],
                head,
                rows,
                h,
                cfg.vocab,
            );
            for r in 0..rows {
                let row = &mut logits[r * cfg.vocab..(r + 1) * cfg.vocab];
                let target = targets[start + r] as usize;
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                loss -= ((row[target] / sum).max(f32::MIN_POSITIVE) as f64).ln();
                if grads.is_some() {
                    let scale = inv / sum;
                    for value in row.iter_mut() {
                        *value *= scale;
                    }
                    row[target] -= inv;
                }
            }
            if let Some(g) = grads.as_deref_mut() {
                let dhead = grad_slice(g, self.layout.output, cfg.vocab * h);
                matmul_add_dw(
                    dhead,
                    logits,
                    &normed[start * h..(start + rows) * h],
                    rows,
                    h,
                    cfg.vocab,
                );
                matmul_add_dx(
                    &mut d_normed[start * h..(start + rows) * h],
                    logits,
                    head,
                    rows,
                    h,
                    cfg.vocab,
                );
            }
        }
        let loss = (loss / t as f64) as f32;

        let Some(grads) = grads else { return loss };

        // Backward: the final norm, then each block in reverse, then the
        // embedding table.
        let mut dx = Aligned::zeros(t * h);
        {
            let dnorm = grad_slice(grads, self.layout.output_norm, h);
            rmsnorm_backward(
                &mut dx,
                dnorm,
                &d_normed,
                &inputs[cfg.layers],
                final_norm,
                &final_rms,
                t,
                h,
            );
        }

        for l in (0..cfg.layers).rev() {
            dx = self.layer_backward(l, &inputs[l], &dx, t, grads);
        }

        let dembd = grad_slice(grads, self.layout.tok_embd, cfg.vocab * h);
        for (i, &token) in tokens.iter().enumerate() {
            let row = token as usize * h;
            for j in 0..h {
                dembd[row + j] += dx[i * h + j];
            }
        }
        loss
    }

    /// One block's forward. `keep` decides whether the intermediates are
    /// returned for a backward pass or dropped.
    fn layer_forward(&self, l: usize, x: &[f32], t: usize, keep: bool) -> LayerCache {
        let cfg = &self.cfg;
        let (h, qd, kvd, f) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.ffn);
        let o = self.layout.layers[l];

        let attn_w = self.tensor(o.attn_norm, h);
        let (normed, attn_rms) = rmsnorm_forward(x, attn_w, t, h, cfg.eps);

        let mut q = Aligned::zeros(t * qd);
        let mut k = Aligned::zeros(t * kvd);
        let mut v = Aligned::zeros(t * kvd);
        matmul_qkv(
            &mut q,
            &mut k,
            &mut v,
            &normed,
            self.tensor(o.wq, qd * h),
            self.tensor(o.wk, kvd * h),
            self.tensor(o.wv, kvd * h),
            t,
            h,
            qd,
            kvd,
        );

        let q_w = self.tensor(o.q_norm, cfg.head_dim);
        let k_w = self.tensor(o.k_norm, cfg.head_dim);
        let (q_normed, q_rms) = rmsnorm_forward(&q, q_w, t * cfg.heads, cfg.head_dim, cfg.eps);
        let (k_normed, k_rms) = rmsnorm_forward(&k, k_w, t * cfg.kv_heads, cfg.head_dim, cfg.eps);

        let mut q_rope = q_normed;
        let mut k_rope = k_normed;
        rope(&mut q_rope, cfg.heads, cfg.head_dim, cfg.rope_base, false);
        rope(
            &mut k_rope,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.rope_base,
            false,
        );

        let ctx = attention(&q_rope, &k_rope, &v, cfg, t);

        let mut mid = Aligned::zeros(t * h);
        matmul_residual(&mut mid, &ctx, self.tensor(o.wo, h * qd), x, t, qd, h);

        let ffn_w = self.tensor(o.ffn_norm, h);
        let (ffn_normed, ffn_rms) = rmsnorm_forward(&mid, ffn_w, t, h, cfg.eps);
        let mut gate = Aligned::zeros(t * f);
        let mut up = Aligned::zeros(t * f);
        let mut act = Aligned::zeros(t * f);
        matmul_swiglu(
            &mut gate,
            &mut up,
            &mut act,
            &ffn_normed,
            self.tensor(o.w_gate, f * h),
            self.tensor(o.w_up, f * h),
            t,
            h,
            f,
        );
        let mut out = Aligned::zeros(t * h);
        matmul_residual(&mut out, &act, self.tensor(o.w_down, h * f), &mid, t, f, h);

        if !keep {
            return LayerCache {
                out,
                ..LayerCache::default()
            };
        }
        LayerCache {
            normed,
            attn_rms,
            q_rope,
            k_rope,
            v,
            q_rms,
            k_rms,
            ctx,
            mid,
            ffn_normed,
            ffn_rms,
            gate,
            up,
            act,
            out,
        }
    }

    /// One block's backward. Returns the gradient with respect to the
    /// block's input and adds the parameter gradients into `grads`.
    fn layer_backward(
        &self,
        l: usize,
        x: &[f32],
        dout: &[f32],
        t: usize,
        grads: &mut [f32],
    ) -> Aligned {
        let cfg = &self.cfg;
        let (h, qd, kvd, f) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.ffn);
        let o = self.layout.layers[l];
        let c = self.layer_forward(l, x, t, true);

        // Feed-forward.
        let mut d_act = Aligned::zeros(t * f);
        {
            let dw_down = grad_slice(grads, o.w_down, h * f);
            matmul_add_dw(dw_down, dout, &c.act, t, f, h);
        }
        matmul_add_dx(&mut d_act, dout, self.tensor(o.w_down, h * f), t, f, h);

        let mut d_gate = Aligned::zeros(t * f);
        let mut d_up = Aligned::zeros(t * f);
        for i in 0..t * f {
            let s = silu(c.gate[i]);
            d_up[i] = d_act[i] * s;
            d_gate[i] = d_act[i] * c.up[i] * dsilu(c.gate[i]);
        }

        let mut d_ffn_normed = Aligned::zeros(t * h);
        {
            let dw_gate = grad_slice(grads, o.w_gate, f * h);
            matmul_add_dw(dw_gate, &d_gate, &c.ffn_normed, t, h, f);
        }
        matmul_add_dx(
            &mut d_ffn_normed,
            &d_gate,
            self.tensor(o.w_gate, f * h),
            t,
            h,
            f,
        );
        {
            let dw_up = grad_slice(grads, o.w_up, f * h);
            matmul_add_dw(dw_up, &d_up, &c.ffn_normed, t, h, f);
        }
        matmul_add_dx(
            &mut d_ffn_normed,
            &d_up,
            self.tensor(o.w_up, f * h),
            t,
            h,
            f,
        );

        // The residual around the feed-forward: `dout` flows straight
        // through to `mid` as well as through the block above.
        let mut d_mid = Aligned::from_slice(dout);
        {
            let dffn_norm = grad_slice(grads, o.ffn_norm, h);
            rmsnorm_backward(
                &mut d_mid,
                dffn_norm,
                &d_ffn_normed,
                &c.mid,
                self.tensor(o.ffn_norm, h),
                &c.ffn_rms,
                t,
                h,
            );
        }

        // Attention output projection.
        let mut d_ctx = Aligned::zeros(t * qd);
        {
            let dwo = grad_slice(grads, o.wo, h * qd);
            matmul_add_dw(dwo, &d_mid, &c.ctx, t, qd, h);
        }
        matmul_add_dx(&mut d_ctx, &d_mid, self.tensor(o.wo, h * qd), t, qd, h);

        let (mut d_q, mut d_k, d_v) =
            attention_backward(&c.q_rope, &c.k_rope, &c.v, &d_ctx, cfg, t);

        // Undo the rotation: it is orthogonal, so its backward is the same
        // rotation with the sine negated.
        rope(&mut d_q, cfg.heads, cfg.head_dim, cfg.rope_base, true);
        rope(&mut d_k, cfg.kv_heads, cfg.head_dim, cfg.rope_base, true);

        // The Q/K norms run per head, so their "rows" are heads, not tokens.
        let (q_pre, k_pre) = self.qk_pre_norm(l, t, &c);
        let mut d_q_pre = Aligned::zeros(t * qd);
        {
            let dq_norm = grad_slice(grads, o.q_norm, cfg.head_dim);
            rmsnorm_backward(
                &mut d_q_pre,
                dq_norm,
                &d_q,
                &q_pre,
                self.tensor(o.q_norm, cfg.head_dim),
                &c.q_rms,
                t * cfg.heads,
                cfg.head_dim,
            );
        }
        let mut d_k_pre = Aligned::zeros(t * kvd);
        {
            let dk_norm = grad_slice(grads, o.k_norm, cfg.head_dim);
            rmsnorm_backward(
                &mut d_k_pre,
                dk_norm,
                &d_k,
                &k_pre,
                self.tensor(o.k_norm, cfg.head_dim),
                &c.k_rms,
                t * cfg.kv_heads,
                cfg.head_dim,
            );
        }
        d_q = d_q_pre;
        d_k = d_k_pre;

        let mut d_normed = Aligned::zeros(t * h);
        {
            let dwq = grad_slice(grads, o.wq, qd * h);
            matmul_add_dw(dwq, &d_q, &c.normed, t, h, qd);
        }
        matmul_add_dx(&mut d_normed, &d_q, self.tensor(o.wq, qd * h), t, h, qd);
        {
            let dwk = grad_slice(grads, o.wk, kvd * h);
            matmul_add_dw(dwk, &d_k, &c.normed, t, h, kvd);
        }
        matmul_add_dx(&mut d_normed, &d_k, self.tensor(o.wk, kvd * h), t, h, kvd);
        {
            let dwv = grad_slice(grads, o.wv, kvd * h);
            matmul_add_dw(dwv, &d_v, &c.normed, t, h, kvd);
        }
        matmul_add_dx(&mut d_normed, &d_v, self.tensor(o.wv, kvd * h), t, h, kvd);

        // The residual around attention.
        let mut dx = d_mid;
        {
            let dattn_norm = grad_slice(grads, o.attn_norm, h);
            rmsnorm_backward(
                &mut dx,
                dattn_norm,
                &d_normed,
                x,
                self.tensor(o.attn_norm, h),
                &c.attn_rms,
                t,
                h,
            );
        }
        dx
    }

    /// Q and K as the projections produced them, before their per-head
    /// norms — the inputs those norms' backward needs. Recomputed rather
    /// than cached because they are two of the largest buffers in the block
    /// and are needed only here.
    fn qk_pre_norm(&self, l: usize, t: usize, c: &LayerCache) -> (Aligned, Aligned) {
        let cfg = &self.cfg;
        let o = self.layout.layers[l];
        let (h, qd, kvd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim());
        let mut q = Aligned::zeros(t * qd);
        let mut k = Aligned::zeros(t * kvd);
        matmul(&mut q, &c.normed, self.tensor(o.wq, qd * h), t, h, qd);
        matmul(&mut k, &c.normed, self.tensor(o.wk, kvd * h), t, h, kvd);
        (q, k)
    }
}

/// One block's intermediates, kept only while its backward runs.
#[derive(Default)]
struct LayerCache {
    normed: Aligned,
    attn_rms: Aligned,
    q_rope: Aligned,
    k_rope: Aligned,
    v: Aligned,
    q_rms: Aligned,
    k_rms: Aligned,
    ctx: Aligned,
    mid: Aligned,
    ffn_normed: Aligned,
    ffn_rms: Aligned,
    gate: Aligned,
    up: Aligned,
    act: Aligned,
    out: Aligned,
}

fn grad_slice(grads: &mut [f32], offset: usize, len: usize) -> &mut [f32] {
    &mut grads[offset..offset + len]
}

/// Rows handled together in one pass over the operand they share.
///
/// The three kernels below all had the same defect: the operand that does
/// not change across the inner loop was re-read from memory on every
/// iteration of the outer one. `matmul` read the whole weight matrix once
/// per token; `matmul_add_dw` read the whole activation matrix once per
/// output row. At `2b` shapes that is tens of gigabytes of traffic per
/// call against matrices of tens of megabytes, and no amount of arithmetic
/// throughput helps a kernel that is waiting on memory.
///
/// Handling eight rows at a time fixes it without changing a single inner
/// loop: the shared operand is read from memory once per *tile*, and the
/// other seven reads of it hit L1. Eight was chosen because a tile's
/// working set — eight rows of the widest activation this trains, plus the
/// row they are multiplied against — still fits in 32 KB of L1.
const TILE: usize = 8;

/// `y[t, n] = x[t, k] . w[n, k]` — the row-major weight layout a GGUF
/// matrix already has, so no transpose is ever materialized.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], t: usize, k: usize, n: usize) {
    debug_assert_eq!(y.len(), t * n);
    debug_assert_eq!(x.len(), t * k);
    debug_assert_eq!(w.len(), n * k);
    y.par_chunks_mut(n * TILE)
        .zip(x.par_chunks(k * TILE))
        .for_each(|(out_tile, in_tile)| tile(out_tile, in_tile, w, k, n));
}

/// `R` token rows against one weight row, two accumulators each.
///
/// [`dot`] issues two loads for every FMA — one vector of weights, one of
/// activations — and this core retires two loads and two FMAs a cycle, so
/// the loads are the ceiling and half the FMA capacity goes unused. Holding
/// the weight vector across four rows makes it ten loads for eight FMAs
/// instead of sixteen, and four rows at two accumulators each keeps eight
/// dependency chains in flight against an FMA latency of four or five
/// cycles while the whole working set still fits sixteen vector registers.
///
/// Four is measured, not assumed: eight rows at one accumulator each has a
/// better load ratio still and is *slower* on the wider shapes, where it
/// runs out of chains and registers at once.
#[inline]
fn dot_rows<const R: usize>(rows: [&[f32]; R], b: &[f32]) -> [f32; R] {
    let mut acc = [[f32x8::ZERO; 2]; R];
    let mut i = 0;
    while i + 2 * LANES <= b.len() {
        for j in 0..2 {
            let at = i + j * LANES;
            let weight = load8(&b[at..]);
            for (row, sums) in rows.iter().zip(acc.iter_mut()) {
                sums[j] = load8(&row[at..]).mul_add(weight, sums[j]);
            }
        }
        i += 2 * LANES;
    }
    let mut out = [0f32; R];
    for ((value, row), sums) in out.iter_mut().zip(rows.iter()).zip(acc.iter()) {
        let mut sum = horizontal(sums[0] + sums[1]);
        let mut at = i;
        while at < b.len() {
            sum += row[at] * b[at];
            at += 1;
        }
        *value = sum;
    }
    out
}

/// One tile of [`matmul`]: the whole weight matrix against `rows` tokens.
///
/// Split out so the fused kernels below can do several of these on one tile
/// without each needing a parallel region of its own.
#[inline]
fn tile(out_tile: &mut [f32], in_tile: &[f32], w: &[f32], k: usize, n: usize) {
    let rows = in_tile.len() / k;
    let row = |r: usize| &in_tile[r * k..(r + 1) * k];
    // Weight row outermost: each one is fetched once and then used by
    // every token in the tile while it is still in L1 — and, four at a
    // time, while it is still in a register.
    for (o, weight_row) in w.chunks_exact(k).enumerate() {
        let mut r = 0;
        while r + 4 <= rows {
            let group = [row(r), row(r + 1), row(r + 2), row(r + 3)];
            for (j, v) in dot_rows(group, weight_row).iter().enumerate() {
                out_tile[(r + j) * n + o] = *v;
            }
            r += 4;
        }
        // The tail takes the same shape rather than [`dot`], so every row
        // of every tensor is summed in one order.
        while r < rows {
            out_tile[r * n + o] = dot_rows([row(r)], weight_row)[0];
            r += 1;
        }
    }
}

/// The three attention projections, which share an input, in one parallel
/// region instead of three.
///
/// The arithmetic is unchanged and so is the result, bit for bit. What
/// changes is how many times a step asks the thread pool for work: a
/// forward pass over this network is a few dozen matmuls of a few hundred
/// microseconds each, and at that size the fork and join around one is not
/// far off the cost of doing it.
#[allow(clippy::too_many_arguments)]
fn matmul_qkv(
    q: &mut [f32],
    k_out: &mut [f32],
    v: &mut [f32],
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    t: usize,
    k: usize,
    qd: usize,
    kvd: usize,
) {
    debug_assert_eq!(q.len(), t * qd);
    debug_assert_eq!(k_out.len(), t * kvd);
    debug_assert_eq!(v.len(), t * kvd);
    q.par_chunks_mut(qd * TILE)
        .zip(k_out.par_chunks_mut(kvd * TILE))
        .zip(v.par_chunks_mut(kvd * TILE))
        .zip(x.par_chunks(k * TILE))
        .for_each(|(((q_tile, k_tile), v_tile), in_tile)| {
            tile(q_tile, in_tile, wq, k, qd);
            tile(k_tile, in_tile, wk, k, kvd);
            tile(v_tile, in_tile, wv, k, kvd);
        });
}

/// The gate and up projections and the activation between them, in one
/// parallel region: `act = silu(gate) * up`.
///
/// The activation is elementwise over a `[tokens, ffn]` buffer — the widest
/// one in the network — and it was a serial loop over all of it between two
/// parallel matmuls. Here each tile applies it to its own rows while they
/// are still in cache.
#[allow(clippy::too_many_arguments)]
fn matmul_swiglu(
    gate: &mut [f32],
    up: &mut [f32],
    act: &mut [f32],
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    t: usize,
    k: usize,
    f: usize,
) {
    debug_assert_eq!(gate.len(), t * f);
    gate.par_chunks_mut(f * TILE)
        .zip(up.par_chunks_mut(f * TILE))
        .zip(act.par_chunks_mut(f * TILE))
        .zip(x.par_chunks(k * TILE))
        .for_each(|(((gate_tile, up_tile), act_tile), in_tile)| {
            tile(gate_tile, in_tile, w_gate, k, f);
            tile(up_tile, in_tile, w_up, k, f);
            for i in 0..act_tile.len() {
                act_tile[i] = silu(gate_tile[i]) * up_tile[i];
            }
        });
}

/// `y[t, n] = x[t, k] . w[n, k] + residual[t, n]`.
///
/// The two residual adds in a block were serial loops over the whole
/// activation between two parallel regions. Folding each into the matmul
/// that produces its left-hand side costs nothing: the rows are in cache,
/// already in the right place, and already in a task.
#[allow(clippy::too_many_arguments)]
fn matmul_residual(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    residual: &[f32],
    t: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(y.len(), t * n);
    debug_assert_eq!(residual.len(), t * n);
    y.par_chunks_mut(n * TILE)
        .zip(x.par_chunks(k * TILE))
        .zip(residual.par_chunks(n * TILE))
        .for_each(|((out_tile, in_tile), residual_tile)| {
            tile(out_tile, in_tile, w, k, n);
            for (o, r) in out_tile.iter_mut().zip(residual_tile) {
                *o += *r;
            }
        });
}

/// `dx[t, k] += dy[t, n] . w[n, k]`.
fn matmul_add_dx(dx: &mut [f32], dy: &[f32], w: &[f32], t: usize, k: usize, n: usize) {
    debug_assert_eq!(dx.len(), t * k);
    dx.par_chunks_mut(k * TILE)
        .zip(dy.par_chunks(n * TILE))
        .for_each(|(dx_tile, dy_tile)| {
            let rows = dx_tile.len() / k;
            for (o, weight_row) in w.chunks_exact(k).enumerate() {
                let mut r = 0;
                while r + 4 <= rows {
                    let a = [
                        dy_tile[r * n + o],
                        dy_tile[(r + 1) * n + o],
                        dy_tile[(r + 2) * n + o],
                        dy_tile[(r + 3) * n + o],
                    ];
                    if a.iter().any(|g| *g != 0.0) {
                        axpy_rows4(four_rows(dx_tile, r, k), weight_row, a);
                    }
                    r += 4;
                }
                while r < rows {
                    let g = dy_tile[r * n + o];
                    if g != 0.0 {
                        axpy(&mut dx_tile[r * k..(r + 1) * k], weight_row, g);
                    }
                    r += 1;
                }
            }
        });
}

/// Four consecutive rows of a tile, as four disjoint mutable slices.
#[inline]
fn four_rows(tile: &mut [f32], first: usize, k: usize) -> [&mut [f32]; 4] {
    let rest = &mut tile[first * k..(first + 4) * k];
    let (a, rest) = rest.split_at_mut(k);
    let (b, rest) = rest.split_at_mut(k);
    let (c, d) = rest.split_at_mut(k);
    [a, b, c, d]
}

/// `dw[n, k] += dy[t, n]^T . x[t, k]`.
fn matmul_add_dw(dw: &mut [f32], dy: &[f32], x: &[f32], t: usize, k: usize, n: usize) {
    debug_assert_eq!(dw.len(), n * k);
    dw.par_chunks_mut(k * TILE)
        .enumerate()
        .for_each(|(tile, dw_tile)| {
            let rows = dw_tile.len() / k;
            let first = tile * TILE;
            // Token outermost, so a token's activations are read once for
            // the whole tile of output rows rather than once for each.
            for step in 0..t {
                let x_row = &x[step * k..(step + 1) * k];
                // Not blocked, unlike the two above: measured at 1.00-1.06x
                // on every shape this trains. The weight gradient reads a
                // *token's* activations against a tile of output rows, and
                // that row is already the thing staying in cache.
                for r in 0..rows {
                    let g = dy[step * n + first + r];
                    if g != 0.0 {
                        axpy(&mut dw_tile[r * k..(r + 1) * k], x_row, g);
                    }
                }
            }
        });
}

/// Loads eight consecutive floats. `copy_from_slice` of exactly eight
/// compiles to one unaligned vector load; the array is never materialized.
#[inline(always)]
fn load8(values: &[f32]) -> f32x8 {
    let mut lane = [0f32; LANES];
    lane.copy_from_slice(&values[..LANES]);
    f32x8::from(lane)
}

const LANES: usize = 8;
/// Elements the main loop takes per iteration: four vectors, so four
/// independent FMA chains are in flight at once. A fused multiply-add has
/// several cycles of latency and issues more than one per cycle, so a
/// single chain leaves most of the unit idle waiting on itself.
const BLOCK: usize = 4 * LANES;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Four vector accumulators, for the same reason the scalar version this
    // replaces had four scalar ones — except that a scalar version is all
    // the compiler is allowed to emit here. Reassociating a float sum
    // changes its result, so no autovectorizer may widen this reduction on
    // its own: the previous loop compiled to `vmulss`/`vaddss`, one element
    // at a time, on a target built for eight-wide AVX2 with FMA. Widening
    // it is a decision about numerics, which is the programmer's to make
    // and is made here.
    let mut acc = [f32x8::ZERO; 4];
    let mut i = 0;
    while i + BLOCK <= a.len() {
        for (k, sum) in acc.iter_mut().enumerate() {
            let at = i + k * LANES;
            *sum = load8(&a[at..]).mul_add(load8(&b[at..]), *sum);
        }
        i += BLOCK;
    }
    while i + LANES <= a.len() {
        acc[0] = load8(&a[i..]).mul_add(load8(&b[i..]), acc[0]);
        i += LANES;
    }

    let mut sum = horizontal(acc[0] + acc[1] + (acc[2] + acc[3]));
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// The eight lanes of an accumulator, added in a tree rather than a
/// chain — half the dependent additions, and no worse for accuracy.
#[inline(always)]
fn horizontal(v: f32x8) -> f32 {
    let lanes = v.to_array();
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

/// `sum(x * x)` — RMSNorm's first pass over a row.
///
/// A reduction, so the same rule that kept the compiler out of [`dot`]
/// applies: it may not widen this on its own, and left alone it emits one
/// `vmulss`/`vaddss` pair per element. Every norm in the network runs it
/// once per row, forward and again on the recompute.
#[inline]
fn sum_squares(x: &[f32]) -> f32 {
    let mut acc = [f32x8::ZERO; 4];
    let mut i = 0;
    while i + BLOCK <= x.len() {
        for (k, sum) in acc.iter_mut().enumerate() {
            let v = load8(&x[i + k * LANES..]);
            *sum = v.mul_add(v, *sum);
        }
        i += BLOCK;
    }
    while i + LANES <= x.len() {
        let v = load8(&x[i..]);
        acc[0] = v.mul_add(v, acc[0]);
        i += LANES;
    }
    let mut sum = horizontal(acc[0] + acc[1] + (acc[2] + acc[3]));
    while i < x.len() {
        sum += x[i] * x[i];
        i += 1;
    }
    sum
}

/// `sum(a * b * c)` — RMSNorm's backward needs this one term across the
/// whole row before it can write any of the row's gradients.
#[inline]
fn dot3(a: &[f32], b: &[f32], c: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let mut acc = [f32x8::ZERO; 2];
    let mut i = 0;
    while i + 2 * LANES <= a.len() {
        for (k, sum) in acc.iter_mut().enumerate() {
            let at = i + k * LANES;
            *sum = (load8(&a[at..]) * load8(&b[at..])).mul_add(load8(&c[at..]), *sum);
        }
        i += 2 * LANES;
    }
    let mut sum = horizontal(acc[0] + acc[1]);
    while i < a.len() {
        sum += a[i] * b[i] * c[i];
        i += 1;
    }
    sum
}

/// Four `y += a * x` against the same `x`, in one pass.
///
/// The mirror of [`dot_rows`] for the backward: `x` is loaded once and used
/// four times instead of once each, which is the same load ceiling in the
/// same place. Written as a plain loop on purpose — the autovectorizer
/// turns this shape into one broadcast load and four FMAs, and every
/// attempt to hand-write the stores has come out slower.
#[inline]
fn axpy_rows4(y: [&mut [f32]; 4], x: &[f32], a: [f32; 4]) {
    let [y0, y1, y2, y3] = y;
    debug_assert_eq!(y0.len(), x.len());
    for i in 0..x.len() {
        let v = x[i];
        y0[i] += a[0] * v;
        y1[i] += a[1] * v;
        y2[i] += a[2] * v;
        y3[i] += a[3] * v;
    }
}

#[inline]
fn axpy(y: &mut [f32], x: &[f32], a: f32) {
    debug_assert_eq!(y.len(), x.len());
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        *o += a * v;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn dsilu(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

/// RMSNorm over each `dim`-wide row, returning the normalized values and
/// each row's reciprocal RMS (which the backward needs and which is far
/// cheaper to keep than to recompute).
fn rmsnorm_forward(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> (Aligned, Aligned) {
    let mut out = Aligned::zeros(rows * dim);
    let mut scales = Aligned::zeros(rows);
    out.par_chunks_mut(dim)
        .zip(scales.par_iter_mut())
        .zip(x.par_chunks(dim))
        .for_each(|((out_row, scale), row)| {
            let mean = sum_squares(row) / dim as f32;
            let r = 1.0 / (mean + eps).sqrt();
            *scale = r;
            for ((o, &v), &weight) in out_row.iter_mut().zip(row.iter()).zip(w.iter()) {
                *o = v * r * weight;
            }
        });
    (out, scales)
}

/// Adds RMSNorm's input gradient into `dx` and its weight gradient into
/// `dw`.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_backward(
    dx: &mut [f32],
    dw: &mut [f32],
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    scales: &[f32],
    rows: usize,
    dim: usize,
) {
    debug_assert_eq!(scales.len(), rows);
    // The weight gradient sums over rows, so it is accumulated per group of
    // rows and folded in once rather than contended over.
    //
    // The groups are a *fixed* size rather than whatever rayon's fold
    // happened to leave in a leaf. A float sum is not associative, so a
    // grouping that depends on which thread stole what makes the result
    // depend on it too — and this gradient goes straight into the weights,
    // where the difference is amplified by every step after it. Fixed
    // groups, summed in index order, give the same answer at any thread
    // count and on any run.
    let partials: Vec<Aligned> = dx
        .par_chunks_mut(dim * NORM_GROUP)
        .zip(dy.par_chunks(dim * NORM_GROUP))
        .zip(x.par_chunks(dim * NORM_GROUP))
        .zip(scales.par_chunks(NORM_GROUP))
        .map(|(((dx_group, dy_group), x_group), scale_group)| {
            let mut acc = Aligned::zeros(dim);
            for (((dx_row, dy_row), x_row), &r) in dx_group
                .chunks_mut(dim)
                .zip(dy_group.chunks(dim))
                .zip(x_group.chunks(dim))
                .zip(scale_group.iter())
            {
                let inner = dot3(dy_row, w, x_row);
                let coeff = r * r * r * inner / dim as f32;
                for (i, ((dx_v, &g), &v)) in dx_row
                    .iter_mut()
                    .zip(dy_row.iter())
                    .zip(x_row.iter())
                    .enumerate()
                {
                    *dx_v += r * w[i] * g - coeff * v;
                    acc[i] += g * v * r;
                }
            }
            acc
        })
        .collect();
    for partial in &partials {
        for (o, v) in dw.iter_mut().zip(partial.iter()) {
            *o += v;
        }
    }
}

/// Rows per task in the norm backward. Fixed, so the order the weight
/// gradient is summed in does not depend on how many threads are running.
const NORM_GROUP: usize = 32;

/// Rotary position embedding, NeoX pairing (`i` against `i + dim/2`) —
/// which is the pairing the inference side applies to every architecture
/// but the original `llama` one.
///
/// `inverse` negates the sine, which transposes each 2x2 rotation and so
/// gives exactly the backward pass.
fn rope(x: &mut [f32], heads: usize, head_dim: usize, base: f32, inverse: bool) {
    let half = head_dim / 2;
    x.par_chunks_mut(heads * head_dim)
        .enumerate()
        .for_each(|(pos, row)| {
            for i in 0..half {
                let freq = base.powf(-2.0 * i as f32 / head_dim as f32);
                let (mut sin, cos) = (pos as f32 * freq).sin_cos();
                if inverse {
                    sin = -sin;
                }
                for head in row.chunks_exact_mut(head_dim) {
                    let a = head[i];
                    let b = head[i + half];
                    head[i] = a * cos - b * sin;
                    head[i + half] = a * sin + b * cos;
                }
            }
        });
}

/// Query positions handled by one attention task.
///
/// Attention parallelizes over heads *and* query positions, which is what
/// gives it more than four tasks on a four-head model. A query position
/// depends on every earlier key but on no other query, so the positions
/// split freely.
///
/// Causal masking makes the later chunks cost more than the earlier ones,
/// so there are deliberately more tasks than threads and the work-stealing
/// pool sorts the imbalance out.
const ATTN_CHUNK: usize = 64;

/// Causal grouped-query attention. Returns the per-token context vectors.
fn attention(q: &[f32], k: &[f32], v: &[f32], cfg: &Config, t: usize) -> Aligned {
    let (hd, heads, group) = (cfg.head_dim, cfg.heads, cfg.group());
    let kvd = cfg.kv_dim();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut ctx = Aligned::zeros(t * cfg.q_dim());

    let chunks = t.div_ceil(ATTN_CHUNK);
    let pieces: Vec<(usize, usize, Aligned)> = (0..heads * chunks)
        .into_par_iter()
        .map(|task| {
            let head = task / chunks;
            let chunk = task % chunks;
            let first = chunk * ATTN_CHUNK;
            let last = (first + ATTN_CHUNK).min(t);
            let kv = head / group;
            let mut out = Aligned::zeros((last - first) * hd);
            let mut scores = Aligned::zeros(t);
            for step in first..last {
                let q_row = &q[step * heads * hd + head * hd..step * heads * hd + (head + 1) * hd];
                let scores = &mut scores[..=step];
                for (s, score) in scores.iter_mut().enumerate() {
                    let k_row = &k[s * kvd + kv * hd..s * kvd + (kv + 1) * hd];
                    *score = dot(q_row, k_row) * scale;
                }
                softmax(scores);
                let at = (step - first) * hd;
                let out_row = &mut out[at..at + hd];
                for (s, &p) in scores.iter().enumerate() {
                    let v_row = &v[s * kvd + kv * hd..s * kvd + (kv + 1) * hd];
                    axpy(out_row, v_row, p);
                }
            }
            (head, first, out)
        })
        .collect();

    for (head, first, piece) in &pieces {
        for (row, values) in piece.chunks_exact(hd).enumerate() {
            let dst = (first + row) * heads * hd + head * hd;
            ctx[dst..dst + hd].copy_from_slice(values);
        }
    }
    ctx
}

/// The backward of [`attention`], recomputing the scores rather than
/// having stored a `[heads, tokens, tokens]` probability matrix — that
/// matrix is quadratic in the sequence length and is the single biggest
/// thing a training step could be made to hold.
fn attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_ctx: &[f32],
    cfg: &Config,
    t: usize,
) -> (Aligned, Aligned, Aligned) {
    let (hd, heads, group) = (cfg.head_dim, cfg.heads, cfg.group());
    let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
    let scale = 1.0 / (hd as f32).sqrt();

    let parts: Vec<(Aligned, Aligned, Aligned)> = (0..heads)
        .into_par_iter()
        .map(|head| {
            let kv = head / group;
            let mut dq = Aligned::zeros(t * hd);
            let mut dk = Aligned::zeros(t * hd);
            let mut dv = Aligned::zeros(t * hd);
            let mut probs = Aligned::zeros(t);
            let mut dscores = Aligned::zeros(t);

            for step in 0..t {
                let q_row = &q[step * qd + head * hd..step * qd + (head + 1) * hd];
                {
                    let scores = &mut probs[..=step];
                    for (s, score) in scores.iter_mut().enumerate() {
                        let k_row = &k[s * kvd + kv * hd..s * kvd + (kv + 1) * hd];
                        *score = dot(q_row, k_row) * scale;
                    }
                    softmax(scores);
                }
                let d_row = &d_ctx[step * qd + head * hd..step * qd + (head + 1) * hd];

                // The value gradient, and the raw probability gradient.
                let mut weighted = 0.0f32;
                for s in 0..=step {
                    let v_row = &v[s * kvd + kv * hd..s * kvd + (kv + 1) * hd];
                    let dp = dot(d_row, v_row);
                    dscores[s] = dp;
                    weighted += probs[s] * dp;
                    axpy(&mut dv[s * hd..(s + 1) * hd], d_row, probs[s]);
                }
                // Softmax backward, folded together with the score scaling.
                for s in 0..=step {
                    dscores[s] = probs[s] * (dscores[s] - weighted) * scale;
                }
                for s in 0..=step {
                    let g = dscores[s];
                    if g == 0.0 {
                        continue;
                    }
                    let k_row = &k[s * kvd + kv * hd..s * kvd + (kv + 1) * hd];
                    axpy(&mut dq[step * hd..(step + 1) * hd], k_row, g);
                    axpy(&mut dk[s * hd..(s + 1) * hd], q_row, g);
                }
            }
            (dq, dk, dv)
        })
        .collect();

    // Query heads own their own gradient column; every head in a group adds
    // into the one key/value head they share.
    let mut d_q = Aligned::zeros(t * qd);
    let mut d_k = Aligned::zeros(t * kvd);
    let mut d_v = Aligned::zeros(t * kvd);
    for (head, (dq, dk, dv)) in parts.iter().enumerate() {
        let kv = head / group;
        for step in 0..t {
            let q_at = step * qd + head * hd;
            d_q[q_at..q_at + hd].copy_from_slice(&dq[step * hd..(step + 1) * hd]);
            let kv_at = step * kvd + kv * hd;
            for j in 0..hd {
                d_k[kv_at + j] += dk[step * hd + j];
                d_v[kv_at + j] += dv[step * hd + j];
            }
        }
    }
    (d_q, d_k, d_v)
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inv = 1.0 / sum;
    for value in values.iter_mut() {
        *value *= inv;
    }
}

/// A small deterministic normal generator — the initializer needs
/// reproducibility across runs and machines, not cryptographic quality.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Box-Muller, one value per call.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {

    /// One weight row against four token rows at once.
    ///
    /// [`dot`] issues two loads for every FMA — the weight vector and the
    /// activation vector — and this chip can retire two loads and two FMAs
    /// a cycle, so the loads are the ceiling and half the FMA units are
    /// idle. Holding the weight vector across four rows makes it ten loads
    /// for eight FMAs instead of sixteen.
    ///
    /// Two accumulators per row, so eight independent chains are in flight
    /// against an FMA latency of four or five cycles, and the whole working
    /// set still fits the sixteen vector registers.
    #[inline]
    fn dot_rows4(rows: [&[f32]; 4], b: &[f32]) -> [f32; 4] {
        let mut acc = [[f32x8::ZERO; 2]; 4];
        let mut i = 0;
        while i + 2 * LANES <= b.len() {
            for j in 0..2 {
                let at = i + j * LANES;
                let weight = load8(&b[at..]);
                for (row, sums) in rows.iter().zip(acc.iter_mut()) {
                    sums[j] = load8(&row[at..]).mul_add(weight, sums[j]);
                }
            }
            i += 2 * LANES;
        }
        let mut out = [0f32; 4];
        for ((value, row), sums) in out.iter_mut().zip(rows.iter()).zip(acc.iter()) {
            let mut sum = horizontal(sums[0] + sums[1]);
            let mut at = i;
            while at < b.len() {
                sum += row[at] * b[at];
                at += 1;
            }
            *value = sum;
        }
        out
    }

    /// Eight rows against one weight vector, one accumulator each: the
    /// best load ratio available (nine loads per eight FMAs) at the cost of
    /// only eight dependency chains to hide the FMA latency with.
    #[inline]
    fn dot_rows8(rows: [&[f32]; 8], b: &[f32]) -> [f32; 8] {
        let mut acc = [f32x8::ZERO; 8];
        let mut i = 0;
        while i + LANES <= b.len() {
            let weight = load8(&b[i..]);
            for (r, row) in rows.iter().enumerate() {
                acc[r] = load8(&row[i..]).mul_add(weight, acc[r]);
            }
            i += LANES;
        }
        let mut out = [0f32; 8];
        for (r, row) in rows.iter().enumerate() {
            let mut sum = horizontal(acc[r]);
            let mut at = i;
            while at < b.len() {
                sum += row[at] * b[at];
                at += 1;
            }
            out[r] = sum;
        }
        out
    }

    fn tile_blocked8(out_tile: &mut [f32], in_tile: &[f32], w: &[f32], k: usize, n: usize) {
        let rows = in_tile.len() / k;
        let row = |r: usize| &in_tile[r * k..(r + 1) * k];
        for (o, weight_row) in w.chunks_exact(k).enumerate() {
            let mut r = 0;
            while r + 8 <= rows {
                let group = [
                    row(r),
                    row(r + 1),
                    row(r + 2),
                    row(r + 3),
                    row(r + 4),
                    row(r + 5),
                    row(r + 6),
                    row(r + 7),
                ];
                for (j, v) in dot_rows8(group, weight_row).iter().enumerate() {
                    out_tile[(r + j) * n + o] = *v;
                }
                r += 8;
            }
            while r < rows {
                out_tile[r * n + o] = dot(row(r), weight_row);
                r += 1;
            }
        }
    }

    fn matmul_blocked8(y: &mut [f32], x: &[f32], w: &[f32], k: usize, n: usize) {
        y.par_chunks_mut(n * TILE)
            .zip(x.par_chunks(k * TILE))
            .for_each(|(out_tile, in_tile)| tile_blocked8(out_tile, in_tile, w, k, n));
    }

    /// [`tile`] with the rows blocked four at a time.
    fn tile_blocked(out_tile: &mut [f32], in_tile: &[f32], w: &[f32], k: usize, n: usize) {
        let rows = in_tile.len() / k;
        for (o, weight_row) in w.chunks_exact(k).enumerate() {
            let mut r = 0;
            while r + 4 <= rows {
                let group = [
                    &in_tile[r * k..(r + 1) * k],
                    &in_tile[(r + 1) * k..(r + 2) * k],
                    &in_tile[(r + 2) * k..(r + 3) * k],
                    &in_tile[(r + 3) * k..(r + 4) * k],
                ];
                let values = dot_rows4(group, weight_row);
                for (j, v) in values.iter().enumerate() {
                    out_tile[(r + j) * n + o] = *v;
                }
                r += 4;
            }
            while r < rows {
                out_tile[r * n + o] = dot(&in_tile[r * k..(r + 1) * k], weight_row);
                r += 1;
            }
        }
    }

    fn matmul_blocked(y: &mut [f32], x: &[f32], w: &[f32], k: usize, n: usize) {
        y.par_chunks_mut(n * TILE)
            .zip(x.par_chunks(k * TILE))
            .for_each(|(out_tile, in_tile)| tile_blocked(out_tile, in_tile, w, k, n));
    }

    fn relative(reference: &[f32], got: &[f32]) -> f32 {
        let num: f32 = reference
            .iter()
            .zip(got)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f32 = reference.iter().map(|a| a * a).sum();
        (num / den).sqrt()
    }

    /// The backward kernels one row at a time, as they were before the
    /// blocking — kept here so the blocked ones can be measured against
    /// something rather than against a memory of a number.
    fn matmul_add_dx_unblocked(
        dx: &mut [f32],
        dy: &[f32],
        w: &[f32],
        t: usize,
        k: usize,
        n: usize,
    ) {
        debug_assert_eq!(dx.len(), t * k);
        dx.par_chunks_mut(k * TILE)
            .zip(dy.par_chunks(n * TILE))
            .for_each(|(dx_tile, dy_tile)| {
                let rows = dx_tile.len() / k;
                for (o, weight_row) in w.chunks_exact(k).enumerate() {
                    for r in 0..rows {
                        let g = dy_tile[r * n + o];
                        if g != 0.0 {
                            axpy(&mut dx_tile[r * k..(r + 1) * k], weight_row, g);
                        }
                    }
                }
            });
    }

    fn matmul_add_dw_unblocked(
        dw: &mut [f32],
        dy: &[f32],
        x: &[f32],
        t: usize,
        k: usize,
        n: usize,
    ) {
        dw.par_chunks_mut(k * TILE)
            .enumerate()
            .for_each(|(tile, dw_tile)| {
                let rows = dw_tile.len() / k;
                let first = tile * TILE;
                for step in 0..t {
                    let x_row = &x[step * k..(step + 1) * k];
                    for r in 0..rows {
                        let g = dy[step * n + first + r];
                        if g != 0.0 {
                            axpy(&mut dw_tile[r * k..(r + 1) * k], x_row, g);
                        }
                    }
                }
            });
    }

    /// Does the same trick pay on the backward, where every FMA also has a
    /// store?
    ///
    ///   cargo test --release --bin orangu-gguf backward_blocking -- --ignored --nocapture
    #[test]
    #[ignore]
    fn backward_blocking_versus_one_row_at_a_time() {
        let shapes: [(&str, usize, usize, usize); 4] = [
            ("attn q/k/v/o", 512, 256, 256),
            ("ffn gate/up ", 512, 256, 688),
            ("ffn down    ", 512, 688, 256),
            ("output head ", 512, 256, 8192),
        ];
        for (name, t, k, n) in shapes {
            let x: Vec<f32> = (0..t * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
                .collect();
            let dy: Vec<f32> = (0..t * n)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
                .collect();
            let flop = 2.0 * t as f64 * k as f64 * n as f64;
            let passes = 10;

            let mut dx = vec![0f32; t * k];
            let time = |f: &mut dyn FnMut()| {
                f();
                let start = std::time::Instant::now();
                for _ in 0..passes {
                    f();
                }
                flop * passes as f64 / start.elapsed().as_secs_f64() / 1e9
            };

            let plain_dx = time(&mut || matmul_add_dx_unblocked(&mut dx, &dy, &w, t, k, n));
            let reference = dx.clone();
            dx.fill(0.0);
            let blocked_dx = time(&mut || matmul_add_dx(&mut dx, &dy, &w, t, k, n));
            let dx_error = relative(&reference, &dx);

            let mut dw = vec![0f32; n * k];
            let plain_dw = time(&mut || matmul_add_dw_unblocked(&mut dw, &dy, &x, t, k, n));
            let reference = dw.clone();
            dw.fill(0.0);
            let blocked_dw = time(&mut || matmul_add_dw(&mut dw, &dy, &x, t, k, n));
            let dw_error = relative(&reference, &dw);

            println!(
                "{name}  dx {plain_dx:>6.1} -> {blocked_dx:>6.1} ({:>4.2}x, {dx_error:.0e})   dw {plain_dw:>6.1} -> {blocked_dw:>6.1} ({:>4.2}x, {dw_error:.0e})",
                blocked_dx / plain_dx,
                blocked_dw / plain_dw,
            );
        }
    }

    /// Does holding the weight vector across four rows pay?
    ///
    ///   cargo test --release --bin orangu-gguf row_blocking -- --ignored --nocapture
    #[test]
    #[ignore]
    fn row_blocking_versus_one_row_at_a_time() {
        let shapes: [(&str, usize, usize, usize); 5] = [
            ("attn q/k/v/o", 512, 256, 256),
            ("ffn gate/up ", 512, 256, 688),
            ("ffn down    ", 512, 688, 256),
            ("output head ", 512, 256, 8192),
            ("0.5b attn   ", 512, 1024, 1024),
        ];
        for (name, t, k, n) in shapes {
            let x: Vec<f32> = (0..t * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
                .collect();
            let mut y = vec![0f32; t * n];
            let flop = 2.0 * t as f64 * k as f64 * n as f64;
            let passes = 20;

            matmul(&mut y, &x, &w, t, k, n);
            let reference = y.clone();
            let start = std::time::Instant::now();
            for _ in 0..passes {
                matmul(&mut y, &x, &w, t, k, n);
            }
            let plain = flop * passes as f64 / start.elapsed().as_secs_f64() / 1e9;

            matmul_blocked(&mut y, &x, &w, k, n);
            let start = std::time::Instant::now();
            for _ in 0..passes {
                matmul_blocked(&mut y, &x, &w, k, n);
            }
            let four = flop * passes as f64 / start.elapsed().as_secs_f64() / 1e9;
            let error4 = relative(&reference, &y);

            matmul_blocked8(&mut y, &x, &w, k, n);
            let start = std::time::Instant::now();
            for _ in 0..passes {
                matmul_blocked8(&mut y, &x, &w, k, n);
            }
            let eight = flop * passes as f64 / start.elapsed().as_secs_f64() / 1e9;
            let error8 = relative(&reference, &y);

            println!(
                "{name} 1x{plain:>6.1}  4x{four:>6.1} ({:>4.2}x, {error4:.0e})  8x{eight:>6.1} ({:>4.2}x, {error8:.0e})",
                four / plain,
                eight / plain,
            );
        }
    }

    /// Eight `bf16` weights, widened into a vector of eight `f32`.
    ///
    /// `bf16` is the top half of an `f32`, so widening is a shift — no table,
    /// no rounding, no special cases. This chip has no `bf16` arithmetic, so
    /// the question the benchmark below asks is whether halving the bytes read
    /// pays for the shift that reading them costs.
    #[inline(always)]
    fn load8_bf16(raw: &[u16]) -> f32x8 {
        let mut lane = [0u32; LANES];
        for (out, &v) in lane.iter_mut().zip(raw[..LANES].iter()) {
            *out = (v as u32) << 16;
        }
        f32x8::from(bytemuck::cast::<[u32; LANES], [f32; LANES]>(lane))
    }

    /// [`dot`] with the second operand in `bf16`.
    #[inline]
    fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = [f32x8::ZERO; 4];
        let mut i = 0;
        while i + BLOCK <= a.len() {
            for (k, sum) in acc.iter_mut().enumerate() {
                let at = i + k * LANES;
                *sum = load8(&a[at..]).mul_add(load8_bf16(&b[at..]), *sum);
            }
            i += BLOCK;
        }
        while i + LANES <= a.len() {
            acc[0] = load8(&a[i..]).mul_add(load8_bf16(&b[i..]), acc[0]);
            i += LANES;
        }
        let mut sum = horizontal(acc[0] + acc[1] + (acc[2] + acc[3]));
        while i < a.len() {
            sum += a[i] * f32::from_bits((b[i] as u32) << 16);
            i += 1;
        }
        sum
    }

    /// [`matmul`] against a weight matrix already narrowed to `bf16`.
    fn matmul_bf16(y: &mut [f32], x: &[f32], w: &[u16], t: usize, k: usize, n: usize) {
        debug_assert_eq!(y.len(), t * n);
        debug_assert_eq!(w.len(), n * k);
        y.par_chunks_mut(n * TILE)
            .zip(x.par_chunks(k * TILE))
            .for_each(|(out_tile, in_tile)| {
                let rows = in_tile.len() / k;
                for (o, weight_row) in w.chunks_exact(k).enumerate() {
                    for r in 0..rows {
                        out_tile[r * n + o] = dot_bf16(&in_tile[r * k..(r + 1) * k], weight_row);
                    }
                }
            });
    }

    /// Is the matmul bound by the bytes it reads or the work it does?
    ///
    /// Every shape the smoke model trains, with the weight matrix in `f32`
    /// and again in `bf16`. `bf16` halves the bytes and costs a widening
    /// shift per eight of them, and this chip has no `bf16` arithmetic — so
    /// the answer is entirely empirical, and it decides whether narrowing
    /// the activations is worth attempting at all.
    ///
    ///   cargo test --release --bin orangu-gguf bf16_versus_f32 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bf16_versus_f32_matmul() {
        let shapes: [(&str, usize, usize, usize); 5] = [
            ("attn q/k/v/o", 512, 256, 256),
            ("ffn gate/up ", 512, 256, 688),
            ("ffn down    ", 512, 688, 256),
            ("output head ", 512, 256, 8192),
            ("0.5b attn   ", 512, 1024, 1024),
        ];
        println!(
            "{:<13} {:>10} {:>10} {:>8}",
            "shape", "f32", "bf16", "ratio"
        );
        for (name, t, k, n) in shapes {
            let x: Vec<f32> = (0..t * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
                .collect();
            let narrow: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let mut y = vec![0f32; t * n];
            let flop = 2.0 * t as f64 * k as f64 * n as f64;

            let rate = |elapsed: std::time::Duration, passes: u32| {
                flop * passes as f64 / elapsed.as_secs_f64() / 1e9
            };
            let passes = 20;

            matmul(&mut y, &x, &w, t, k, n);
            let start = std::time::Instant::now();
            for _ in 0..passes {
                matmul(&mut y, &x, &w, t, k, n);
            }
            let wide = rate(start.elapsed(), passes);
            let reference: Vec<f32> = y.clone();

            matmul_bf16(&mut y, &x, &narrow, t, k, n);
            let start = std::time::Instant::now();
            for _ in 0..passes {
                matmul_bf16(&mut y, &x, &narrow, t, k, n);
            }
            let short = rate(start.elapsed(), passes);

            // What the narrowing costs in accuracy, on the same inputs.
            let error = {
                let num: f32 = reference
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let den: f32 = reference.iter().map(|a| a * a).sum();
                (num / den).sqrt()
            };

            println!(
                "{name} {wide:>7.1} GF/s {short:>7.1} GF/s {:>7.2}x  rel err {error:.2e}",
                short / wide
            );
        }
    }

    use super::*;

    fn tiny() -> Config {
        Config {
            vocab: 11,
            hidden: 8,
            ffn: 16,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            context: 32,
            rope_base: 10000.0,
            eps: 1e-5,
        }
    }

    /// The written tensor set is a contract with the reader, so it is
    /// written out here rather than inferred from whatever the layout
    /// happens to contain.
    ///
    /// Two absences are as deliberate as the presences. There are **no
    /// query/key/value biases** — the previous generation of this
    /// architecture had them and this one does not, and a reader that finds
    /// them will apply them. There is **no `rope_freqs.weight`** — that
    /// tensor carries per-frequency scaling factors for stretching a
    /// context beyond what was trained, and this writer states a longer
    /// context by scaling the rotary base instead, which needs no tensor.
    #[test]
    fn every_tensor_the_reader_needs_is_written_and_nothing_else() {
        let cfg = tiny();
        let layout = Layout::new(&cfg);
        let names: Vec<&str> = layout.specs.iter().map(|s| s.name.as_str()).collect();

        // The three outside the blocks. `output.weight` is optional to a
        // reader — one that does not find it ties the output projection to
        // the embedding — and it is written, so nothing is tied.
        for name in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            assert!(names.contains(&name), "missing {name}");
        }

        // Nine a reader requires per block, and the two that make this
        // architecture itself rather than the one before it.
        let per_block = [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        for block in 0..cfg.layers {
            for suffix in per_block {
                let name = format!("blk.{block}.{suffix}");
                assert!(names.contains(&name.as_str()), "missing {name}");
            }
        }

        assert_eq!(
            names.len(),
            3 + cfg.layers * per_block.len(),
            "the file carries a tensor this test does not know about: {:?}",
            names
        );
        assert!(
            !names.iter().any(|n| n.ends_with(".bias")),
            "this architecture has no biases"
        );
    }

    #[test]
    fn the_layout_covers_every_parameter_exactly_once() {
        let cfg = tiny();
        let layout = Layout::new(&cfg);
        let mut covered = vec![false; layout.total];
        for spec in &layout.specs {
            for (i, seen) in covered[spec.offset..spec.offset + spec.len()]
                .iter_mut()
                .enumerate()
            {
                assert!(
                    !*seen,
                    "{} overlaps another tensor at {}",
                    spec.name,
                    spec.offset + i
                );
                *seen = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "a parameter is in no tensor");
    }

    /// The published sizes have to actually be the sizes they are named
    /// after, and every head width has to divide the hidden width.
    #[test]
    fn the_named_sizes_are_the_sizes_they_claim() {
        for size in SIZES {
            assert_eq!(
                size.hidden,
                size.heads * size.head_dim,
                "{}: hidden width is not heads x head width",
                size.key
            );
            assert_eq!(size.heads % size.kv_heads, 0, "{}", size.key);
            // 256 is the K-quants' super-block. A row length it does not
            // divide cannot be a K-quant at all, so a size that gets this
            // wrong is a size whose files quietly are not the format their
            // name promises. `ffn` is the one that bites: it is the row
            // length of the down projection.
            assert_eq!(
                size.ffn % 256,
                0,
                "{}: ffn {} is not a multiple of 256, so ffn_down cannot be a K-quant",
                size.key,
                size.ffn
            );
            assert_eq!(
                size.hidden % 256,
                0,
                "{}: hidden {} is not a multiple of 256",
                size.key,
                size.hidden
            );
        }
        let two_b = size_named("2b").unwrap();
        let cfg = Config::from_size(two_b, 32768, 262_144);
        let params = cfg.parameters();
        assert!(
            (1.9e9..2.2e9).contains(&(params as f64)),
            "2b is {params} parameters"
        );
    }

    #[test]
    fn the_rotation_base_grows_with_the_declared_context() {
        assert_eq!(rope_base_for(8192), 1.0e6);
        assert_eq!(rope_base_for(4096), 1.0e6);
        assert_eq!(rope_base_for(262_144), 3.2e7);
    }

    /// The one test that stands between a wrong derivative and a training
    /// run that silently learns nothing: for *every* parameter tensor, the
    /// analytic gradient at its largest-magnitude entry must match a
    /// central finite difference of the loss.
    #[test]
    fn gradients_match_finite_differences() {
        let cfg = tiny();
        let mut model = Model::new(cfg.clone(), 7);
        let tokens: Vec<u32> = vec![1, 5, 2, 9, 4, 4, 7, 3];
        let targets: Vec<u32> = vec![5, 2, 9, 4, 4, 7, 3, 1];

        let mut grads = Aligned::zeros(model.layout.total);
        model.forward_backward(&tokens, &targets, Some(&mut grads));

        let specs = model.layout.specs.clone();
        for spec in &specs {
            let range = spec.offset..spec.offset + spec.len();
            let (at, analytic) = range
                .clone()
                .map(|i| (i, grads[i]))
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap();
            assert!(
                analytic.abs() > 1e-6,
                "{} has no gradient anywhere — it is not connected to the loss",
                spec.name
            );

            // Small enough that the central difference's own truncation
            // error (which falls with the square of the step) is well under
            // the tolerance, large enough that f32 rounding of two nearly
            // equal losses is not.
            let eps = 1e-3f32;
            let original = model.params[at];
            model.params[at] = original + eps;
            let up = model.forward_backward(&tokens, &targets, None);
            model.params[at] = original - eps;
            let down = model.forward_backward(&tokens, &targets, None);
            model.params[at] = original;

            let numeric = (up - down) / (2.0 * eps);
            // Two tolerances, because a central difference has two error
            // terms. The relative one covers truncation; the absolute one
            // is the floor set by subtracting two nearly equal f32 losses
            // and dividing by the step — which is what dominates for a
            // tensor whose gradient is genuinely tiny, as the norms' are.
            let noise = f32::EPSILON * up.abs().max(1.0) / eps;
            let difference = (numeric - analytic).abs();
            assert!(
                difference <= noise + 1e-2 * analytic.abs(),
                "{}[{}]: analytic {analytic}, numeric {numeric} (difference {difference}, noise floor {noise})",
                spec.name,
                at - spec.offset
            );
        }
    }

    /// Attention must not see the future: changing a token can only change
    /// the predictions at or after its own position.
    #[test]
    fn attention_is_causal() {
        let cfg = tiny();
        let model = Model::new(cfg, 3);
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let mut b = a.clone();
        b[4] = 9;
        let targets: Vec<u32> = vec![2, 3, 4, 5, 6, 1];

        // Truncating both to the prefix before the edit must give the same
        // loss, which it can only do if position 4 never influenced 0..4.
        let loss_a = model.forward_backward(&a[..4], &targets[..4], None);
        let loss_b = model.forward_backward(&b[..4], &targets[..4], None);
        assert_eq!(loss_a, loss_b);
        assert_ne!(
            model.forward_backward(&a, &targets, None),
            model.forward_backward(&b, &targets, None)
        );
    }

    /// An untrained model's loss is the uniform-distribution loss, within
    /// the slack the random initialization leaves — a much tighter check
    /// than "it is finite", and it catches a scale error in the
    /// initializer.
    #[test]
    fn an_untrained_model_starts_near_uniform_loss() {
        let cfg = tiny();
        let model = Model::new(cfg.clone(), 11);
        let tokens: Vec<u32> = (0..16).map(|i| (i * 7 % 11) as u32).collect();
        let targets: Vec<u32> = (0..16).map(|i| ((i + 1) * 7 % 11) as u32).collect();
        let loss = model.forward_backward(&tokens, &targets, None);
        let uniform = (cfg.vocab as f32).ln();
        assert!(
            (loss - uniform).abs() < 0.5,
            "loss {loss} against a uniform {uniform}"
        );
    }

    #[test]
    fn the_rotation_is_its_own_inverse_when_reversed() {
        let mut x: Vec<f32> = (0..3 * 2 * 4).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let original = x.clone();
        rope(&mut x, 2, 4, 10000.0, false);
        assert_ne!(x, original);
        rope(&mut x, 2, 4, 10000.0, true);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

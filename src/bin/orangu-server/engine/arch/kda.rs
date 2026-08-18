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

//! The two half-layers a *delta-net / latent-attention hybrid* is built
//! from, shared by every architecture here that is one — not an
//! architecture itself, the way [`qwen_hybrid`](super::qwen_hybrid) is not.
//!
//! Both `engine::arch::kimi3` and `engine::arch::bailingmoe` alternate the
//! same pair, three of the first for every one of the second:
//!
//! * [`KdaLayer`] — Kimi Delta Attention: a short causal convolution over
//!   each of Q/K/V, then the delta rule with a **per-dimension** decay
//!   (a plain gated delta-net decays a whole head by one scalar), then a
//!   gated per-head RMSNorm. Recurrent, so it holds no key/value cache —
//!   only the conv history and one `[head_dim, head_dim]` accumulator per
//!   head.
//! * [`MlaLayer`] — absorbed multi-head latent attention, as in
//!   `engine::arch::glm`: one compressed vector per token stands in for
//!   both the key and the value, and each head's query is pushed through
//!   that head's key decompression up front so the cache never has to be
//!   expanded.
//!
//! What differs between the two architectures is small and entirely
//! *declared*, which is why one implementation serves both: whether the
//! decay gate is factored into two matrices or is one full-rank one
//! ([`KdaNames::f_b`]), which tensor holds the output gate
//! ([`KdaNames::g`]), whether `ssm_a` was folded to `-exp(A_log)` or left
//! at `+exp(A_log)` at conversion time ([`KdaNames::a_is_negated`]),
//! whether the latent attention rotates anything at all
//! ([`MlaShape::rope`]), and whether its output gate is one scalar per head
//! or one per value dimension (read from the tensor's own width). Every one
//! of those is a load-time decision; the arithmetic below is identical.
//!
//! Transcribed from upstream `llama.cpp`'s `src/models/bailingmoe3.cpp` and
//! `src/models/kimi-k3.cpp`, together with the delta rule both call — read
//! from ggml's `gated_delta_net` kernel itself (`ggml/src/ggml-cpu/
//! ops.cpp`) rather than from the graph that invokes it, because the
//! kernel is where the state's layout and the decayed axis are actually
//! decided. See [`delta_step`]. The `1/sqrt(head_dim)` query scale lives
//! there too, not in either architecture's own graph.

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::attend;
use crate::engine::backend::Backend;
use crate::engine::kv_cache::{KvCache, RecurrentSpec};
use crate::engine::loader::{ExpertQuantMatrix, LoadedModel, QuantMatrix};
use crate::engine::tensor::{self, RopeParams};

/// Which tensors a given file spells one KDA layer's gates with, and how it
/// signed `ssm_a`.
///
/// The names are not cosmetic. `ssm_f_a` alone is a *full-rank* decay
/// projection; `ssm_f_a` followed by `ssm_f_b` is the same map factored
/// through a rank-`head_dim` bottleneck. Reading one file's layout with the
/// other's expectation is a shape error at load, which is the good case —
/// the bad case is a file that happens to agree on widths.
pub(crate) struct KdaNames {
    /// The decay gate's first (or only) projection.
    pub f_a: &'static str,
    /// Its second, when the file factors the decay gate. `None` means
    /// `f_a` already projects to `d_inner`.
    pub f_b: Option<&'static str>,
    /// The full-rank output gate.
    pub g: &'static str,
    /// Whether `ssm_a` holds `-exp(A_log)` (folded at conversion time) or
    /// `+exp(A_log)` with the sign left to live in the gate's lower bound.
    /// Both conventions are in the wild and neither is discoverable from
    /// the tensor's shape — only from its values' sign, which is exactly
    /// the kind of check that passes on one file and lies on the next.
    pub a_is_negated: bool,
}

/// `kimi-k3`: a factored decay gate, a full-rank `ssm_g`, and `ssm_a`
/// already negated.
pub(crate) const KIMI3_KDA_NAMES: KdaNames = KdaNames {
    f_a: "ssm_f_a.weight",
    f_b: Some("ssm_f_b.weight"),
    g: "ssm_g.weight",
    a_is_negated: true,
};

/// `bailingmoe3`: a full-rank decay gate under the `_a` name anyway, the
/// output gate as `ssm_g_a`, and `ssm_a` left positive.
pub(crate) const BAILINGMOE3_KDA_NAMES: KdaNames = KdaNames {
    f_a: "ssm_f_a.weight",
    f_b: None,
    g: "ssm_g_a.weight",
    a_is_negated: false,
};

/// The KDA hyperparameters every layer of one model shares.
pub(crate) struct KdaShape {
    pub n_head: usize,
    /// `kda.head_dim`.
    pub head_dim: usize,
    /// `n_head * head_dim`.
    pub d_inner: usize,
    /// `ssm.conv_kernel`.
    pub d_conv: usize,
    /// `kda.gate_lower_bound`, when the file sets one — the "safe gate".
    pub gate_lower_bound: Option<f32>,
    pub eps: f32,
}

impl KdaShape {
    /// The cache shape one KDA layer asks for: three sets of conv channels
    /// (Q, K and V), and one square accumulator per head.
    pub(crate) fn recurrent_spec(&self) -> RecurrentSpec {
        RecurrentSpec::delta_net(3 * self.d_inner, self.d_conv, self.n_head, self.head_dim)
    }
}

/// One Kimi Delta Attention layer's weights.
pub(crate) struct KdaLayer {
    wq: QuantMatrix,
    wk: QuantMatrix,
    wv: QuantMatrix,
    /// The `ssm_conv1d_q`/`_k`/`_v` kernels concatenated into one
    /// `[3 * d_inner, d_conv]` buffer, channel-major.
    ///
    /// The convolution is depthwise — every channel is independent — so
    /// concatenating the three sets of channels and running one pass is
    /// exactly the three separate passes upstream runs over its one
    /// three-section conv state, and lets this share
    /// `RecurrentLayerState::conv_step` with `engine::arch::qwen35moe`
    /// unchanged.
    conv_kernel: Vec<f32>,
    /// The decay gate's projection: `[n_embd, d_inner]` on its own, or
    /// `[n_embd, head_dim]` followed by `f_b`'s `[head_dim, d_inner]`.
    f_a: QuantMatrix,
    f_b: Option<QuantMatrix>,
    dt_bias: Vec<f32>,
    /// `+exp(A_log)`, `[n_head]` — normalized at load whichever sign the
    /// file stored, so the arithmetic below has one form.
    a: Vec<f32>,
    /// `[n_embd, n_head]` — the per-head delta-rule write strength.
    beta: QuantMatrix,
    /// `[n_embd, d_inner]` — the full-rank output gate.
    g: QuantMatrix,
    /// RMSNorm weight applied per head to the scan output, `[head_dim]`.
    o_norm: Vec<f32>,
    wo: QuantMatrix,
    /// Index into `KvCache::recurrent`.
    cache_index: usize,
}

impl KdaLayer {
    pub(crate) fn load(
        loaded: &LoadedModel,
        layer: usize,
        shape: &KdaShape,
        names: &KdaNames,
        cache_index: usize,
    ) -> Result<Self> {
        let get = |suffix: &str| -> Result<Vec<f32>> {
            let name = format!("blk.{layer}.{suffix}");
            Ok(loaded
                .tensor(&name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };

        // One conv kernel per channel, Q's channels then K's then V's,
        // matching the concatenated projection fed to it.
        let mut conv_kernel = Vec::with_capacity(3 * shape.d_inner * shape.d_conv);
        for part in ["q", "k", "v"] {
            let kernel = get(&format!("ssm_conv1d_{part}.weight"))?;
            anyhow::ensure!(
                kernel.len() == shape.d_inner * shape.d_conv,
                "layer {layer}'s ssm_conv1d_{part} has {} values, not d_inner * conv_kernel ({})",
                kernel.len(),
                shape.d_inner * shape.d_conv
            );
            conv_kernel.extend_from_slice(&kernel);
        }

        let mut a = get("ssm_a")?;
        anyhow::ensure!(
            a.len() == shape.n_head,
            "layer {layer}'s ssm_a has {} values, not one per head ({})",
            a.len(),
            shape.n_head
        );
        if names.a_is_negated {
            for value in &mut a {
                *value = -*value;
            }
        }

        let f_a = get_matrix(names.f_a)?;
        let f_b = names.f_b.map(get_matrix).transpose()?;
        let decay_out = f_b.as_ref().unwrap_or(&f_a).out_dim;
        anyhow::ensure!(
            decay_out == shape.d_inner,
            "layer {layer}'s decay gate projects to {decay_out} outputs, not d_inner ({})",
            shape.d_inner
        );

        Ok(Self {
            wq: get_matrix("attn_q.weight")?,
            wk: get_matrix("attn_k.weight")?,
            wv: get_matrix("attn_v.weight")?,
            conv_kernel,
            f_a,
            f_b,
            dt_bias: get("ssm_dt.bias")?,
            a,
            beta: get_matrix("ssm_beta.weight")?,
            g: get_matrix(names.g)?,
            o_norm: get("ssm_norm.weight")?,
            wo: get_matrix("attn_output.weight")?,
            cache_index,
        })
    }

    /// The forward pass: convolve, decay, apply the delta rule, gate, norm.
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        shape: &KdaShape,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let n_head = shape.n_head;
        let head_dim = shape.head_dim;
        let d_inner = shape.d_inner;
        let eps = shape.eps;
        // The query scale upstream applies inside `build_delta_net`, not in
        // either architecture's own graph.
        let q_scale = 1.0 / (head_dim as f32).sqrt();

        // Token-independent projections, batched over the whole chunk. The
        // three conv inputs are concatenated to match `conv_kernel`.
        let q = backend.matmul(normed, n_tokens, &self.wq);
        let k = backend.matmul(normed, n_tokens, &self.wk);
        let v = backend.matmul(normed, n_tokens, &self.wv);
        let f = match &self.f_b {
            Some(f_b) => {
                let low = backend.matmul(normed, n_tokens, &self.f_a);
                backend.matmul(&low, n_tokens, f_b)
            }
            None => backend.matmul(normed, n_tokens, &self.f_a),
        };
        let beta = backend.matmul(normed, n_tokens, &self.beta);
        let gate = backend.matmul(normed, n_tokens, &self.g);

        let mut out = vec![0f32; n_tokens * d_inner];
        let state = &mut cache.recurrent[self.cache_index];
        for t in 0..n_tokens {
            let mut qkv = Vec::with_capacity(3 * d_inner);
            qkv.extend_from_slice(&q[t * d_inner..(t + 1) * d_inner]);
            qkv.extend_from_slice(&k[t * d_inner..(t + 1) * d_inner]);
            qkv.extend_from_slice(&v[t * d_inner..(t + 1) * d_inner]);
            let mut conv = state.conv_step(&qkv, &self.conv_kernel);
            for value in conv.iter_mut() {
                *value = tensor::silu(*value);
            }
            let (q_t, rest) = conv.split_at_mut(d_inner);
            let (k_t, v_t) = rest.split_at_mut(d_inner);

            // The decay: `lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`
            // when the file sets a lower bound (the "safe gate"), and
            // `-exp(A_log) * softplus(f + dt_bias)` when it does not. The
            // scan exponentiates this, making the per-dimension decay
            // `exp(lower_bound * sigmoid(..))`, which lives in
            // `(e^lower_bound, 1)` — the point of the bound being that it
            // cannot reach 0 and erase the state outright.
            let mut decay = vec![0f32; d_inner];
            for h in 0..n_head {
                for j in 0..head_dim {
                    let idx = h * head_dim + j;
                    let pre = f[t * d_inner + idx] + self.dt_bias[idx];
                    decay[idx] = match shape.gate_lower_bound {
                        Some(bound) => bound * tensor::sigmoid(pre * self.a[h]),
                        None => -self.a[h] * tensor::softplus(pre),
                    }
                    .exp();
                }
            }

            let dst = &mut out[t * d_inner..(t + 1) * d_inner];
            for h in 0..n_head {
                let q_h = &mut q_t[h * head_dim..(h + 1) * head_dim];
                tensor::l2_norm_inplace(q_h, eps);
                for value in q_h.iter_mut() {
                    *value *= q_scale;
                }
                let k_h = &mut k_t[h * head_dim..(h + 1) * head_dim];
                tensor::l2_norm_inplace(k_h, eps);
                let q_h = &q_t[h * head_dim..(h + 1) * head_dim];
                let k_h = &k_t[h * head_dim..(h + 1) * head_dim];
                let v_h = &v_t[h * head_dim..(h + 1) * head_dim];
                let beta_h = tensor::sigmoid(beta[t * n_head + h]);
                let decay_h = &decay[h * head_dim..(h + 1) * head_dim];

                let mut o = delta_step(state.delta_state_mut(h), decay_h, k_h, v_h, q_h, beta_h);

                // Gated RMSNorm: norm the scan output, then scale it by a
                // sigmoid gate read from the layer input.
                tensor::rmsnorm_inplace(&mut o, &self.o_norm, 1, head_dim, eps);
                let gate_h = &gate[t * d_inner + h * head_dim..t * d_inner + (h + 1) * head_dim];
                for (value, &g) in o.iter_mut().zip(gate_h.iter()) {
                    *value *= tensor::sigmoid(g);
                }
                dst[h * head_dim..(h + 1) * head_dim].copy_from_slice(&o);
            }
        }

        backend.matmul(&out, n_tokens, &self.wo)
    }
}

/// One head's delta-rule step, exactly ggml's `gated_delta_net`: the state
/// decays, absorbs this token's key and value, and is read out with the
/// query. `state` is `[head_dim, head_dim]` row-major — `state[i * head_dim
/// + j]` is upstream's `S[i][j]` — and is updated in place; the return is
/// that head's `head_dim`-wide output.
///
/// **`decay` indexes `i`, the axis `k` and `q` index — not `j`, the axis
/// `v` indexes.** Both are `head_dim` wide, so the wrong one is a perfectly
/// valid program, and on a sequence's first token the two agree exactly
/// (the state is still zero, so the decay multiplies nothing). The wrong
/// axis therefore yields a correct first token and fluent text after it,
/// from a model that quietly cannot recall anything said earlier. ggml's
/// kernel states the layout outright — "state is stored transposed:
/// `s_out[j*S_v + i] = S[i][j]`", then `S[i][:] *= exp(g[i])`.
///
/// A plain gated delta-net (`engine::arch::qwen_hybrid`) decays a whole
/// head by one scalar, where this decays each dimension separately; that
/// is the "delta" half of Kimi Delta Attention made per-dimension.
fn delta_step(
    state: &mut [f32],
    decay: &[f32],
    k: &[f32],
    v: &[f32],
    q: &[f32],
    beta: f32,
) -> Vec<f32> {
    let head_dim = decay.len();
    debug_assert_eq!(state.len(), head_dim * head_dim);

    for (i, &d) in decay.iter().enumerate() {
        for s in &mut state[i * head_dim..(i + 1) * head_dim] {
            *s *= d;
        }
    }
    let mut sk = vec![0f32; head_dim];
    for i in 0..head_dim {
        tensor::axpy_inplace(&mut sk, &state[i * head_dim..(i + 1) * head_dim], k[i]);
    }
    let delta: Vec<f32> = (0..head_dim).map(|j| beta * (v[j] - sk[j])).collect();
    for i in 0..head_dim {
        tensor::axpy_inplace(&mut state[i * head_dim..(i + 1) * head_dim], &delta, k[i]);
    }
    let mut out = vec![0f32; head_dim];
    for i in 0..head_dim {
        tensor::axpy_inplace(&mut out, &state[i * head_dim..(i + 1) * head_dim], q[i]);
    }
    out
}

/// The MLA hyperparameters every full-attention layer of one model shares.
pub(crate) struct MlaShape {
    pub n_head: usize,
    /// `attention.kv_lora_rank` — the width of the compressed key/value,
    /// which is also the value's width.
    pub kv_lora_rank: usize,
    /// `attention.key_length_mla` — one query head's width, rotary part
    /// included.
    pub head_k_mla: usize,
    /// `attention.value_length_mla` — one output head's width.
    pub head_v_mla: usize,
    /// Width of one cached row: `kv_lora_rank + rope dims`. The value is
    /// its leading `kv_lora_rank`.
    pub kv_row: usize,
    /// The rotary parameters, or `None` for a model that rotates nothing —
    /// in which case `rope.dimension_count` still names the width of the
    /// key's second half, it simply passes through unrotated.
    pub rope: Option<RopeParams>,
    /// `rope.dimension_count`, whether or not anything is rotated.
    pub rope_dim: usize,
    pub kq_scale: f32,
    pub eps: f32,
}

/// How a file spells the query projection: through a low-rank bottleneck
/// (`attn_q_a`/`attn_q_b`, `attention.q_lora_rank > 0`) or in one matrix.
enum Query {
    Lora {
        wq_a: QuantMatrix,
        q_a_norm: Vec<f32>,
        wq_b: QuantMatrix,
    },
    Plain(QuantMatrix),
}

/// One absorbed-MLA layer's weights.
pub(crate) struct MlaLayer {
    q: Query,
    wkv_a_mqa: QuantMatrix,
    kv_a_norm: Vec<f32>,
    wk_b: ExpertQuantMatrix,
    wv_b: ExpertQuantMatrix,
    /// `attn_gate` — sigmoid-gates the attention output before the output
    /// projection. Either `[n_embd, n_head]` (one scalar per head,
    /// broadcast over its value dimensions) or `[n_embd, n_head *
    /// value_length_mla]` (one per dimension); which one is read from the
    /// tensor's own width.
    gate: Option<QuantMatrix>,
    wo: QuantMatrix,
    /// Index into `KvCache::layers`.
    cache_index: usize,
}

impl MlaLayer {
    pub(crate) fn load(
        loaded: &LoadedModel,
        layer: usize,
        shape: &MlaShape,
        cache_index: usize,
    ) -> Result<Self> {
        let get = |suffix: &str| -> Result<Vec<f32>> {
            let name = format!("blk.{layer}.{suffix}");
            Ok(loaded
                .tensor(&name)
                .with_context(|| format!("loading {name}"))?
                .0)
        };
        let get_matrix = |suffix: &str| -> Result<QuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .matrix(&name)
                .with_context(|| format!("loading {name}"))
        };
        let get_expert_matrix = |suffix: &str| -> Result<ExpertQuantMatrix> {
            let name = format!("blk.{layer}.{suffix}");
            loaded
                .expert_matrix(&name)
                .with_context(|| format!("loading {name}"))
        };

        let q = match get_matrix("attn_q_a.weight") {
            Ok(wq_a) => Query::Lora {
                q_a_norm: get("attn_q_a_norm.weight")?,
                wq_b: get_matrix("attn_q_b.weight")?,
                wq_a,
            },
            Err(_) => Query::Plain(get_matrix("attn_q.weight")?),
        };

        let wkv_a_mqa = get_matrix("attn_kv_a_mqa.weight")?;
        anyhow::ensure!(
            wkv_a_mqa.out_dim == shape.kv_row,
            "layer {layer}'s attn_kv_a_mqa projects to {} outputs, not kv_lora_rank + rope dims ({})",
            wkv_a_mqa.out_dim,
            shape.kv_row
        );

        let gate = get_matrix("attn_gate.weight").ok();
        if let Some(gate) = &gate {
            let per_head = shape.n_head;
            let per_dim = shape.n_head * shape.head_v_mla;
            anyhow::ensure!(
                gate.out_dim == per_head || gate.out_dim == per_dim,
                "layer {layer}'s attn_gate projects to {} outputs, neither one per head \
                 ({per_head}) nor one per value dimension ({per_dim})",
                gate.out_dim
            );
        }

        Ok(Self {
            q,
            wkv_a_mqa,
            kv_a_norm: get("attn_kv_a_norm.weight")?,
            wk_b: get_expert_matrix("attn_k_b.weight")?,
            wv_b: get_expert_matrix("attn_v_b.weight")?,
            gate,
            wo: get_matrix("attn_output.weight")?,
            cache_index,
        })
    }

    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        shape: &MlaShape,
        cache: &mut KvCache,
        normed: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let n_head = shape.n_head;
        let eps = shape.eps;
        let nope = shape.head_k_mla - shape.rope_dim;
        let absorbed_dim = shape.kv_lora_rank + shape.rope_dim;

        let mut q = match &self.q {
            Query::Lora {
                wq_a,
                q_a_norm,
                wq_b,
            } => {
                let mut qr = backend.matmul(normed, n_tokens, wq_a);
                tensor::rmsnorm_inplace(&mut qr, q_a_norm, n_tokens, wq_a.out_dim, eps);
                backend.matmul(&qr, n_tokens, wq_b)
            }
            Query::Plain(wq) => backend.matmul(normed, n_tokens, wq),
        };
        let mut kv = backend.matmul(normed, n_tokens, &self.wkv_a_mqa);

        for t in 0..n_tokens {
            let pos = start_pos + t;
            // The compressed half is normed; the rotary half is roped (or,
            // for a model that rotates nothing, left alone). The two
            // together are one cache row.
            let row = &mut kv[t * shape.kv_row..(t + 1) * shape.kv_row];
            tensor::rmsnorm_inplace(
                &mut row[..shape.kv_lora_rank],
                &self.kv_a_norm,
                1,
                shape.kv_lora_rank,
                eps,
            );
            if let Some(rope) = &shape.rope {
                tensor::rope_apply_params_inplace(
                    &mut row[shape.kv_lora_rank..],
                    1,
                    shape.rope_dim,
                    pos,
                    None,
                    rope,
                );
                let q_t =
                    &mut q[t * n_head * shape.head_k_mla..(t + 1) * n_head * shape.head_k_mla];
                for h in 0..n_head {
                    tensor::rope_apply_params_inplace(
                        &mut q_t[h * shape.head_k_mla + nope..(h + 1) * shape.head_k_mla],
                        1,
                        shape.rope_dim,
                        pos,
                        None,
                        rope,
                    );
                }
            }
        }

        let mut attn_out = vec![0f32; n_tokens * n_head * shape.head_v_mla];
        for t in 0..n_tokens {
            let kv_t = &kv[t * shape.kv_row..(t + 1) * shape.kv_row];
            cache.layers[self.cache_index].push(kv_t, kv_t);

            let n_keys = start_pos + t + 1;
            let mut keys = vec![0f32; n_keys * shape.kv_row];
            {
                let slot = &cache.layers[self.cache_index];
                for p in 0..n_keys {
                    keys[p * shape.kv_row..(p + 1) * shape.kv_row].copy_from_slice(slot.key_at(
                        p,
                        0,
                        shape.kv_row,
                    ));
                }
            }

            let q_t = &q[t * n_head * shape.head_k_mla..(t + 1) * n_head * shape.head_k_mla];
            let heads: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    // Absorb the query through this head's key
                    // decompression, then carry its second half unchanged.
                    let q_nope = &q_t[h * shape.head_k_mla..h * shape.head_k_mla + nope];
                    let mut q_h = vec![0f32; absorbed_dim];
                    for (j, out) in q_h[..shape.kv_lora_rank].iter_mut().enumerate() {
                        *out = tensor::dot(q_nope, &self.wk_b.row(h, j));
                    }
                    q_h[shape.kv_lora_rank..].copy_from_slice(
                        &q_t[h * shape.head_k_mla + nope..(h + 1) * shape.head_k_mla],
                    );

                    let compressed = attend(
                        &q_h,
                        &keys,
                        shape.kv_row,
                        shape.kv_lora_rank,
                        shape.kq_scale,
                        None,
                    );
                    (0..shape.head_v_mla)
                        .map(|d| tensor::dot(&compressed, &self.wv_b.row(h, d)))
                        .collect()
                })
                .collect();
            for (h, head) in heads.iter().enumerate() {
                let at = (t * n_head + h) * shape.head_v_mla;
                attn_out[at..at + shape.head_v_mla].copy_from_slice(head);
            }
        }

        if let Some(gate) = &self.gate {
            let gate = backend.matmul(normed, n_tokens, gate);
            if gate.len() == attn_out.len() {
                for (o, &g) in attn_out.iter_mut().zip(gate.iter()) {
                    *o *= tensor::sigmoid(g);
                }
            } else {
                // One scalar per head, broadcast over that head's value
                // dimensions.
                for t in 0..n_tokens {
                    for h in 0..n_head {
                        let g = tensor::sigmoid(gate[t * n_head + h]);
                        let at = (t * n_head + h) * shape.head_v_mla;
                        for o in &mut attn_out[at..at + shape.head_v_mla] {
                            *o *= g;
                        }
                    }
                }
            }
        }
        backend.matmul(&attn_out, n_tokens, &self.wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-head gate and the per-dimension gate must be told apart by
    /// the tensor's width, not by the architecture — `bailingmoe3` ships
    /// the first and `kimi-k3` the second, and the two are the same code
    /// path.
    ///
    /// This pins the arithmetic of the broadcast half: every value
    /// dimension of head `h` is scaled by `sigmoid` of head `h`'s single
    /// logit, and heads do not bleed into each other.
    #[test]
    fn a_per_head_gate_scales_every_dimension_of_that_head() {
        let n_head = 3;
        let head_v = 4;
        let logits = [2.0f32, -2.0, 0.0];
        let mut out: Vec<f32> = (0..n_head * head_v).map(|i| (i + 1) as f32).collect();

        for h in 0..n_head {
            let g = tensor::sigmoid(logits[h]);
            for o in &mut out[h * head_v..(h + 1) * head_v] {
                *o *= g;
            }
        }

        for (h, &logit) in logits.iter().enumerate() {
            let g = tensor::sigmoid(logit);
            for d in 0..head_v {
                let i = h * head_v + d;
                assert!(
                    (out[i] - (i + 1) as f32 * g).abs() < 1e-6,
                    "head {h} dim {d} was not scaled by its own gate"
                );
            }
        }
    }

    /// Two steps of [`delta_step`], pinning **which axis the decay
    /// indexes** — against a hand-computed expectation, and against the
    /// other convention, which must not produce the same answer.
    ///
    /// The state after step one is `k1 ⊗ v1`, so it is row 0 alone. Step
    /// two decays it, then writes with `k2`, which selects row *1* — all
    /// zeros — so the write contributes nothing and the readout is the
    /// decayed row 0. Decaying by key scales that whole row by
    /// `decay[0] = 0.5`; decaying by value would scale its second element
    /// by `decay[1] = 0.25` instead. One number tells them apart.
    ///
    /// The first token cannot: with a zero state the decay multiplies
    /// nothing, which is exactly why a wrong axis survives a short prompt.
    #[test]
    fn the_decay_indexes_the_key_axis_not_the_value_axis() {
        let decay = [0.5f32, 0.25];
        let (k1, v1) = ([1.0f32, 0.0], [1.0f32, 1.0]);
        let (k2, v2, q2) = ([0.0f32, 1.0], [0.0f32, 0.0], [1.0f32, 1.0]);

        let mut state = [0.0f32; 4];
        // Step one, from a zero state: `state[i][j] = k1[i] * beta * v1[j]`,
        // which the decay cannot touch whichever axis it indexes.
        let first = delta_step(&mut state, &decay, &k1, &v1, &k1, 1.0);
        assert_eq!(state, [1.0, 1.0, 0.0, 0.0], "state should be k1 (x) v1");
        assert_eq!(first, vec![1.0, 1.0]);

        let out = delta_step(&mut state, &decay, &k2, &v2, &q2, 1.0);
        assert_eq!(
            out,
            vec![0.5, 0.5],
            "row 0 should have been scaled by decay[0] alone"
        );

        // The same two steps with the decay transposed onto the value
        // axis, to prove the assertion above can tell them apart at all.
        let mut mirrored = [0.0f32; 4];
        for (k, v) in [(k1, v1), (k2, v2)] {
            let head_dim = 2;
            for row in 0..head_dim {
                for (s, &d) in mirrored[row * head_dim..(row + 1) * head_dim]
                    .iter_mut()
                    .zip(decay.iter())
                {
                    *s *= d;
                }
            }
            let sk: Vec<f32> = (0..head_dim)
                .map(|j| {
                    (0..head_dim)
                        .map(|i| mirrored[i * head_dim + j] * k[i])
                        .sum()
                })
                .collect();
            for i in 0..head_dim {
                for j in 0..head_dim {
                    mirrored[i * head_dim + j] += k[i] * (v[j] - sk[j]);
                }
            }
        }
        let mirrored_out: Vec<f32> = (0..2)
            .map(|j| (0..2).map(|i| mirrored[i * 2 + j] * q2[i]).sum())
            .collect();
        assert_eq!(mirrored_out, vec![0.5, 0.25]);
        assert_ne!(
            out, mirrored_out,
            "the two conventions must be distinguishable, or this test proves nothing"
        );
    }

    /// The two `ssm_a` sign conventions must produce the *same* decay.
    ///
    /// A file storing `-exp(A_log)` and one storing `+exp(A_log)` describe
    /// the same model; the load-time negation is the only thing that makes
    /// them agree, and getting it backwards flips the gate's argument
    /// rather than failing — `sigmoid` is defined on both sides, so the
    /// model would run and drift.
    #[test]
    fn both_ssm_a_conventions_give_the_same_decay() {
        let bound = -5.0f32;
        let a_pos = 1.75f32;
        for pre in [-3.0f32, -0.25, 0.0, 0.5, 4.0] {
            let from_positive = bound * tensor::sigmoid(pre * a_pos);
            // What the negated convention stores, put back through the
            // same load-time flip.
            let stored = -a_pos;
            let from_negated = bound * tensor::sigmoid(pre * -stored);
            assert!((from_positive - from_negated).abs() < 1e-6, "pre {pre}");
        }
    }

    /// The safe gate's whole purpose: the per-dimension decay stays above
    /// `e^lower_bound`, so however long the sequence gets the recurrent
    /// state is never silently erased. (The upper end is the harmless
    /// direction — the decay saturates *at* 1, which is "remember
    /// everything".)
    #[test]
    fn the_safe_gate_bounds_the_decay_away_from_zero() {
        let bound = -5.0f32;
        let floor = bound.exp();
        assert!(floor > 0.0, "the floor is what makes the gate safe");
        for pre in [-50.0f32, -1.0, 0.0, 1.0, 50.0] {
            let decay = (bound * tensor::sigmoid(pre * 2.0)).exp();
            assert!(
                (floor..=1.0).contains(&decay),
                "decay {decay} for pre {pre} left [e^bound, 1]"
            );
        }
        // Without the bound the same input saturates the other way: an
        // unbounded `-exp(A_log) * softplus(..)` decays to zero, which is
        // the state being erased.
        let unbounded = (-2.0f32 * tensor::softplus(50.0)).exp();
        assert!(
            unbounded < floor / 1e30,
            "the unbounded form ({unbounded}) should collapse far below the safe floor \
             ({floor}) — that collapse is what the bound exists to avoid"
        );
    }
}

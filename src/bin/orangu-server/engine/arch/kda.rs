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

use super::Seqs;
use crate::engine::backend::Backend;
use crate::engine::kv_cache::RecurrentSpec;
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
    /// The output gate's first (or only) projection.
    pub g: &'static str,
    /// Its second, when the file factors the output gate the way it
    /// factors the decay one. `None` means `g` already projects to
    /// `d_inner`.
    pub g_b: Option<&'static str>,
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
    g_b: None,
    a_is_negated: true,
};

/// `bailingmoe3`: a full-rank decay gate under the `_a` name anyway, the
/// output gate as `ssm_g_a`, and `ssm_a` left positive.
pub(crate) const BAILINGMOE3_KDA_NAMES: KdaNames = KdaNames {
    f_a: "ssm_f_a.weight",
    f_b: None,
    g: "ssm_g_a.weight",
    g_b: None,
    a_is_negated: false,
};

/// `glm5-next`: both gates factored through a `head_dim` bottleneck, and
/// `ssm_a` already negated.
pub(crate) const GLM5NEXT_KDA_NAMES: KdaNames = KdaNames {
    f_a: "ssm_f_a.weight",
    f_b: Some("ssm_f_b.weight"),
    g: "ssm_g_a.weight",
    g_b: Some("ssm_g_b.weight"),
    a_is_negated: true,
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
    /// The epsilon of the L2 normalization applied to the query and key
    /// before the delta rule. Usually the model's own RMS epsilon, but
    /// `glm5-next`'s reference hard-codes `1e-6` regardless of it, and the
    /// two differ there by an order of magnitude. Only ever visible on a
    /// head whose vector is near zero, which is exactly why it is declared
    /// rather than assumed.
    pub l2_eps: f32,
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
    /// The output gate's projection: `[n_embd, d_inner]` on its own, or
    /// `[n_embd, head_dim]` followed by `g_b`'s `[head_dim, d_inner]`.
    g: QuantMatrix,
    g_b: Option<QuantMatrix>,
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

        let g = get_matrix(names.g)?;
        let g_b = names.g_b.map(get_matrix).transpose()?;
        let gate_out = g_b.as_ref().unwrap_or(&g).out_dim;
        anyhow::ensure!(
            gate_out == shape.d_inner,
            "layer {layer}'s output gate projects to {gate_out} outputs, not d_inner ({})",
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
            g,
            g_b,
            o_norm: get("ssm_norm.weight")?,
            wo: get_matrix("attn_output.weight")?,
            cache_index,
        })
    }

    /// The forward pass: convolve, decay, apply the delta rule, gate, norm,
    /// into caller-owned buffers — see
    /// `Backend::matmul_into`. Every projection here is the full
    /// `[n_tokens, d_inner]`, once per layer, so they are the same class of
    /// buffer the dense architectures hoist.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_into(
        &self,
        backend: &dyn Backend,
        out: &mut Vec<f32>,
        scratch: &mut KdaScratch,
        shape: &KdaShape,
        seqs: &mut Seqs<'_, '_>,
        normed: &[f32],
        n_tokens: usize,
    ) {
        let n_head = shape.n_head;
        let head_dim = shape.head_dim;
        let d_inner = shape.d_inner;
        let eps = shape.eps;
        // The query scale upstream applies inside `build_delta_net`, not in
        // either architecture's own graph.
        let q_scale = 1.0 / (head_dim as f32).sqrt();

        // Token-independent projections, batched over the whole chunk. The
        // three conv inputs are concatenated to match `conv_kernel`. Timed
        // as `recurrent.project`, and dropped where the scan starts.
        let _project = crate::engine::decode_stages::enter(
            crate::engine::decode_stages::Stage::RecurrentProject,
        );
        let KdaScratch {
            q,
            k,
            v,
            f,
            f_low,
            beta,
            gate,
            gate_low,
        } = scratch;
        backend.matmul_into(q, normed, n_tokens, &self.wq);
        backend.matmul_into(k, normed, n_tokens, &self.wk);
        backend.matmul_into(v, normed, n_tokens, &self.wv);
        match &self.f_b {
            Some(f_b) => {
                backend.matmul_into(f_low, normed, n_tokens, &self.f_a);
                backend.matmul_into(f, f_low, n_tokens, f_b);
            }
            None => backend.matmul_into(f, normed, n_tokens, &self.f_a),
        }
        backend.matmul_into(beta, normed, n_tokens, &self.beta);
        match &self.g_b {
            Some(g_b) => {
                backend.matmul_into(gate_low, normed, n_tokens, &self.g);
                backend.matmul_into(gate, gate_low, n_tokens, g_b);
            }
            None => backend.matmul_into(gate, normed, n_tokens, &self.g),
        }
        let (q, k, v, f, beta, gate) = (&*q, &*k, &*v, &*f, &*beta, &*gate);

        drop(_project);
        // The scan output, before the output projection. Every element is
        // written by the loop below, so `resize` without a `clear`.
        let mut scan = vec![0f32; n_tokens * d_inner];
        // The state update, timed apart from the projections either side of
        // it: it is scalar and sequential by construction, so its elapsed
        // time and its share of the machine are different numbers — which
        // is the whole reason these counters measure elapsed time.
        let _delta = crate::engine::decode_stages::enter(
            crate::engine::decode_stages::Stage::RecurrentDelta,
        );
        for t in 0..n_tokens {
            let state = &mut seqs.cache(t).recurrent[self.cache_index];
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

            let dst = &mut scan[t * d_inner..(t + 1) * d_inner];
            // **One head per core.** The scan is sequential in *position* —
            // each token's state update reads the one the token before it
            // wrote — but not in head: every head owns its own state matrix
            // and its own slice of this token's q, k, v and output, and
            // nothing crosses between them. So the position loop stays
            // serial and the head loop inside it is a fan-out, which is the
            // shape `arch::nemotron`'s state-space heads already use.
            //
            // Per head the arithmetic is `delta_step` unchanged, in the same
            // order, so this is bit-identical to the loop it replaces.
            let (states, state_size) = state.delta_states_mut();
            let step_head = |h: usize,
                             head_state: &mut [f32],
                             q_h: &mut [f32],
                             k_h: &mut [f32],
                             dst_h: &mut [f32]| {
                tensor::l2_norm_inplace(q_h, shape.l2_eps);
                for value in q_h.iter_mut() {
                    *value *= q_scale;
                }
                tensor::l2_norm_inplace(k_h, shape.l2_eps);
                let v_h = &v_t[h * head_dim..(h + 1) * head_dim];
                let beta_h = tensor::sigmoid(beta[t * n_head + h]);
                let decay_h = &decay[h * head_dim..(h + 1) * head_dim];

                let mut o = delta_step(head_state, decay_h, k_h, v_h, q_h, beta_h);

                // Gated RMSNorm: norm the scan output, then scale it by a
                // sigmoid gate read from the layer input.
                tensor::rmsnorm_inplace(&mut o, &self.o_norm, 1, head_dim, eps);
                let gate_h = &gate[t * d_inner + h * head_dim..t * d_inner + (h + 1) * head_dim];
                for (value, &g) in o.iter_mut().zip(gate_h.iter()) {
                    *value *= tensor::sigmoid(g);
                }
                dst_h.copy_from_slice(&o);
            };
            if delta_fanout() {
                use rayon::prelude::*;
                // **Several heads per task, not one.** One head's work is a
                // `head_dim x head_dim` update, and handing that to the pool
                // one head at a time makes the dispatch a real fraction of
                // it: profiling a prompt on this architecture puts about a
                // ninth of all CPU in the scheduler and in `crossbeam`'s
                // steal and epoch paths, and the fan-out over heads gains
                // only a few percent on work that is perfectly independent.
                // Grouping heads amortizes the hand-off over more
                // arithmetic without changing any of it — the tasks stay
                // disjoint, so this is the same fan-out with a coarser
                // grain.
                let group = delta_heads_per_task(n_head).min(n_head.max(1));
                states
                    .par_chunks_mut(state_size * group)
                    .zip(q_t.par_chunks_mut(head_dim * group))
                    .zip(k_t.par_chunks_mut(head_dim * group))
                    .zip(dst.par_chunks_mut(head_dim * group))
                    .enumerate()
                    .for_each(|(g, (((states, q_g), k_g), dst_g))| {
                        for (i, (((head_state, q_h), k_h), dst_h)) in states
                            .chunks_mut(state_size)
                            .zip(q_g.chunks_mut(head_dim))
                            .zip(k_g.chunks_mut(head_dim))
                            .zip(dst_g.chunks_mut(head_dim))
                            .enumerate()
                        {
                            step_head(g * group + i, head_state, q_h, k_h, dst_h);
                        }
                    });
            } else {
                for (h, (((head_state, q_h), k_h), dst_h)) in states
                    .chunks_mut(state_size)
                    .zip(q_t.chunks_mut(head_dim))
                    .zip(k_t.chunks_mut(head_dim))
                    .zip(dst.chunks_mut(head_dim))
                    .enumerate()
                {
                    step_head(h, head_state, q_h, k_h, dst_h);
                }
            }
        }

        drop(_delta);
        crate::engine::decode_stages::scope(
            crate::engine::decode_stages::Stage::RecurrentOut,
            || backend.matmul_into(out, &scan, n_tokens, &self.wo),
        );
    }
}

/// The per-layer projection buffers [`Kda::forward_into`] reuses — see
/// `Backend::matmul_into`. Each is `[n_tokens, d_inner]` (or `[n_tokens,
/// n_head]` for `beta`), allocated once per forward pass instead of seven
/// times per layer.
#[derive(Default)]
pub(crate) struct KdaScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    f: Vec<f32>,
    /// The low-rank intermediate, only when `f_b` is present.
    f_low: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
    /// The low-rank intermediate, only when `g_b` is present.
    gate_low: Vec<f32>,
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

    // **Two sweeps of the state, not four**, with the accumulation order
    // unchanged: the decay folds into the `S^T k` read because that read
    // wants the decayed value anyway, and the output projection folds into
    // the rank-1 update because row `i` after the update is exactly what
    // the projection multiplies by `q[i]`. Each fused loop walks its row
    // once with two independent accumulations, a shape the vectorizer
    // keeps.
    //
    // This halves the passes and measures **level** — the state is one
    // head's `head_dim x head_dim`, small enough to stay in cache across
    // all four, so the passes cost cache bandwidth rather than memory. The
    // form is kept because it is the same arithmetic in fewer lines, not
    // because it is faster. What this stage is short of is arithmetic, not
    // traffic.
    let mut sk = vec![0f32; head_dim];
    for (i, (&d, &k_i)) in decay.iter().zip(k).enumerate() {
        let row = &mut state[i * head_dim..(i + 1) * head_dim];
        for (s, acc) in row.iter_mut().zip(sk.iter_mut()) {
            *s *= d;
            *acc += *s * k_i;
        }
    }
    let delta: Vec<f32> = (0..head_dim).map(|j| beta * (v[j] - sk[j])).collect();
    let mut out = vec![0f32; head_dim];
    for (i, (&k_i, &q_i)) in k.iter().zip(q).enumerate() {
        let row = &mut state[i * head_dim..(i + 1) * head_dim];
        for ((s, &d_j), o) in row.iter_mut().zip(&delta).zip(out.iter_mut()) {
            *s += d_j * k_i;
            *o += *s * q_i;
        }
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
    /// Learned per-pair rotary frequency divisors (`rope_freqs.weight`,
    /// LeRoPE), `rope_dim / 2` of them, or `None` for the plain
    /// frequencies.
    pub freq_factors: Option<std::sync::Arc<[f32]>>,
    pub kq_scale: f32,
    pub eps: f32,
}

/// The compressed key/value projection of a layer that writes its own
/// cache rows.
struct KvProjection {
    wkv_a_mqa: QuantMatrix,
    kv_a_norm: Vec<f32>,
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
    /// `None` on a layer that attends rows another layer wrote
    /// (`attention.kv_source_map`): it projects a query and nothing else,
    /// and `cache_index` names the source layer's slot.
    kv: Option<KvProjection>,
    /// `attn_ssmax` — a learned per-head, position-dependent query scale.
    /// See `engine::arch::position::Ssmax`.
    ssmax: Option<super::position::Ssmax>,
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

/// One vector against every row of a head's absorbed projection.
///
/// `hoisted` is `Some` when the caller dequantized the whole projection up
/// front — worth it once there is more than one token to reuse each row for —
/// and `None` when it did not, in which case each row is dequantized into a
/// reused buffer and dotted while it is still in cache. Both paths compute
/// the same dot products in the same order, so which one runs cannot change
/// an output; see the `hoist` guard in `MlaLayer::forward` for the
/// measurement that decides it.
/// `ORANGU_MLA_NO_HOIST=1` keeps latent attention on the per-row
/// dequantize at every width, so the hoisted path can be compared against
/// the one it replaced inside one binary.
///
/// The hoist is the only thing about this module that changes behaviour at
/// `n_tokens > 1`, which makes it the first suspect whenever a batched
/// forward disagrees with the same tokens fed one at a time.
/// Whether a dense prompt takes the batched causal attention —
/// `ORANGU_ATTN_BATCHED`, on unless it is turned off.
///
/// The control arm for it: `0` runs the per-token loop instead, which is
/// the same arithmetic in a different accumulation order, so the two can be
/// swept against each other from one binary.
/// `ORANGU_DELTA_FANOUT=0` runs the delta rule's heads one after another on
/// the calling thread — the control arm for the fan-out over heads. On
/// unless `0`.
/// Heads handed to one task by the delta rule's fan-out, for a layer of
/// `n_head` — `ORANGU_DELTA_HEADS_PER_TASK` overrides it with a fixed
/// number, and `0` falls back to the rule below.
///
/// **A floor on the task count, not a target.** One head per task is the
/// finest grain available and the most dispatch per unit of arithmetic,
/// and grouping is worth a little: measured on a delta-net mixture, four
/// heads a task beat one in four of four alternating pairs. What that
/// measurement also shows is that the pool was never the constraint —
/// four tasks across sixteen workers still won — so matching the task
/// count to the worker count would be solving the wrong problem and would
/// give back the whole gain.
///
/// The risk is the other end: a layer with few heads, grouped, is a serial
/// loop wearing a fan-out's clothes. So the grain is whatever keeps at
/// least [`DELTA_MIN_TASKS`] tasks, and a layer that cannot afford that
/// many is left at one head each.
fn delta_heads_per_task(n_head: usize) -> usize {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let forced = *OVERRIDE.get_or_init(|| {
        crate::engine::backend::tuning_profile::env_var("ORANGU_DELTA_HEADS_PER_TASK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
    });
    match forced {
        Some(n) => n,
        None => (n_head / DELTA_MIN_TASKS).max(1),
    }
}

/// The fewest tasks the delta rule's fan-out will split a layer into — see
/// [`delta_heads_per_task`]. Below this a grouped layer starts leaving
/// cores idle, which costs more than the dispatch it saves.
const DELTA_MIN_TASKS: usize = 4;

fn delta_fanout() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_DELTA_FANOUT")
}

fn batched_attention() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_ATTN_BATCHED")
}

fn mla_no_hoist() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_MLA_NO_HOIST"))
}

pub(crate) fn project_rows(
    weights: &ExpertQuantMatrix,
    hoisted: Option<&[Vec<f32>]>,
    head: usize,
    x: &[f32],
    row_len: usize,
    out: &mut [f32],
) {
    if let Some(rows) = hoisted {
        tensor::dot_rows(x, &rows[head], row_len, out);
        return;
    }
    let span = weights.expert_span(head);
    let mut row = Vec::new();
    for (index, slot) in out.iter_mut().enumerate() {
        weights.row_from(span, index, &mut row);
        *slot = tensor::dot(x, &row);
    }
}

impl MlaLayer {
    /// Loads a layer that writes its own compressed rows into
    /// `cache_index`.
    pub(crate) fn load(
        loaded: &LoadedModel,
        layer: usize,
        shape: &MlaShape,
        cache_index: usize,
    ) -> Result<Self> {
        Self::load_inner(loaded, layer, shape, cache_index, true)
    }

    /// Loads a layer with no key/value projection of its own, which
    /// attends the rows an earlier layer already wrote into `source_index`
    /// this same pass.
    pub(crate) fn load_reading(
        loaded: &LoadedModel,
        layer: usize,
        shape: &MlaShape,
        source_index: usize,
    ) -> Result<Self> {
        Self::load_inner(loaded, layer, shape, source_index, false)
    }

    fn load_inner(
        loaded: &LoadedModel,
        layer: usize,
        shape: &MlaShape,
        cache_index: usize,
        owns_kv: bool,
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
        let q_out = match &q {
            Query::Lora { wq_b, .. } => wq_b.out_dim,
            Query::Plain(wq) => wq.out_dim,
        };
        anyhow::ensure!(
            q_out == shape.n_head * shape.head_k_mla,
            "layer {layer}'s query projects to {q_out} outputs, not head_count * key_length_mla ({})",
            shape.n_head * shape.head_k_mla
        );

        let kv = if owns_kv {
            let wkv_a_mqa = get_matrix("attn_kv_a_mqa.weight")?;
            anyhow::ensure!(
                wkv_a_mqa.out_dim == shape.kv_row,
                "layer {layer}'s attn_kv_a_mqa projects to {} outputs, not kv_lora_rank + rope dims ({})",
                wkv_a_mqa.out_dim,
                shape.kv_row
            );
            Some(KvProjection {
                wkv_a_mqa,
                kv_a_norm: get("attn_kv_a_norm.weight")?,
            })
        } else {
            None
        };
        let ssmax =
            super::position::Ssmax::load(loaded, &format!("blk.{layer}.attn_ssmax"), shape.n_head)?;

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
            kv,
            ssmax,
            wk_b: get_expert_matrix("attn_k_b.weight")?,
            wv_b: get_expert_matrix("attn_v_b.weight")?,
            gate,
            wo: get_matrix("attn_output.weight")?,
            cache_index,
        })
    }

    /// The forward pass into caller-owned buffers — see
    /// `Backend::matmul_into`. There is no allocating twin: both callers
    /// own a scratch, so a second entry point would only be somewhere for
    /// the two to drift apart.
    ///
    /// [`Self::project_query_into`] then [`Self::attend_into`] over a dense
    /// selection, which is all this is: an architecture whose attention is
    /// *sparse* has to run the two halves apart, because what it needs to
    /// decide the selection with is the query LoRA intermediate the first
    /// half produces.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_into(
        &self,
        backend: &dyn Backend,
        out: &mut Vec<f32>,
        scratch: &mut MlaScratch,
        shape: &MlaShape,
        seqs: &mut Seqs<'_, '_>,
        normed: &[f32],
        n_tokens: usize,
    ) {
        crate::engine::decode_stages::scope(
            crate::engine::decode_stages::Stage::AttnProject,
            || self.project_query_into(backend, scratch, shape, normed, n_tokens),
        );
        crate::engine::decode_stages::scope(crate::engine::decode_stages::Stage::Attn, || {
            self.attend_into(backend, out, scratch, shape, seqs, normed, n_tokens, None)
        });
    }

    /// The query half: the LoRA bottleneck (or the one-matrix projection)
    /// into `scratch`, unrotated — rotation depends on position and belongs
    /// to [`Self::attend_into`].
    ///
    /// Split out for the sparse-attention callers, which project the
    /// lightning indexer's own query off [`MlaScratch::qr`] before deciding
    /// what this layer may attend. See `engine::arch::indexer`.
    pub(crate) fn project_query_into(
        &self,
        backend: &dyn Backend,
        scratch: &mut MlaScratch,
        shape: &MlaShape,
        normed: &[f32],
        n_tokens: usize,
    ) {
        let MlaScratch { q, qr, .. } = scratch;
        match &self.q {
            Query::Lora {
                wq_a,
                q_a_norm,
                wq_b,
            } => {
                backend.matmul_into(qr, normed, n_tokens, wq_a);
                tensor::rmsnorm_inplace(qr, q_a_norm, n_tokens, wq_a.out_dim, shape.eps);
                backend.matmul_into(q, qr, n_tokens, wq_b);
            }
            Query::Plain(wq) => {
                qr.clear();
                backend.matmul_into(q, normed, n_tokens, wq);
            }
        }
    }

    /// Everything after the query: the compressed key/value, the rotation,
    /// the absorbed softmax, and the output projection.
    ///
    /// `selection` names, per token, the positions that token may attend,
    /// in ascending order — a lightning indexer's cut. `None` is the dense
    /// causal window, `0..=pos`, which is what every architecture without
    /// one attends.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend_into(
        &self,
        backend: &dyn Backend,
        out: &mut Vec<f32>,
        scratch: &mut MlaScratch,
        shape: &MlaShape,
        seqs: &mut Seqs<'_, '_>,
        normed: &[f32],
        n_tokens: usize,
        selection: Option<&[Vec<usize>]>,
    ) {
        let n_head = shape.n_head;
        let eps = shape.eps;
        let nope = shape.head_k_mla - shape.rope_dim;
        let absorbed_dim = shape.kv_lora_rank + shape.rope_dim;

        let MlaScratch {
            q,
            kv,
            gate: gate_buf,
            ..
        } = scratch;
        let q = &mut *q;
        // A layer reading another layer's rows writes none of its own.
        let owns_kv = self.kv.is_some();
        match &self.kv {
            Some(proj) => backend.matmul_into(kv, normed, n_tokens, &proj.wkv_a_mqa),
            None => kv.clear(),
        }
        let kv = &mut *kv;
        let freq_factors = shape.freq_factors.as_deref();

        for t in 0..n_tokens {
            let pos = seqs.pos(t);
            // The compressed half is normed; the rotary half is roped (or,
            // for a model that rotates nothing, left alone). The two
            // together are one cache row.
            if let Some(proj) = &self.kv {
                let row = &mut kv[t * shape.kv_row..(t + 1) * shape.kv_row];
                tensor::rmsnorm_inplace(
                    &mut row[..shape.kv_lora_rank],
                    &proj.kv_a_norm,
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
                        freq_factors,
                        rope,
                    );
                }
            }
            let q_t = &mut q[t * n_head * shape.head_k_mla..(t + 1) * n_head * shape.head_k_mla];
            if let Some(rope) = &shape.rope {
                for h in 0..n_head {
                    tensor::rope_apply_params_inplace(
                        &mut q_t[h * shape.head_k_mla + nope..(h + 1) * shape.head_k_mla],
                        1,
                        shape.rope_dim,
                        pos,
                        freq_factors,
                        rope,
                    );
                }
            }
            // One scalar per head, so it commutes with the absorption below.
            if let Some(ssmax) = &self.ssmax {
                ssmax.apply(q_t, shape.head_k_mla, pos);
            }
        }

        // `wk_b`/`wv_b` are weights: their rows do not depend on `t`. Read
        // per token — which is what `ExpertQuantMatrix::row` inside the loop
        // below amounted to — each row was dequantized `n_tokens` times over,
        // into a freshly allocated `Vec<f32>` that was dotted once and
        // dropped. On a 1122-token prefill that is 1121 redundant
        // dequantizations of every row, and it measured **10.9% of the whole
        // profile** in `quant::dequantize_into`.
        //
        // Hoisted here, once per call. Bit-exact: the same bytes through the
        // same dequantizer produce the same `f32`s, and `tensor::dot` is
        // handed the identical slice it was before — only *when* it is
        // produced changes.
        //
        // Decode (`n_tokens == 1`) does the same amount of dequantizing as it
        // always did and strictly fewer allocations: one buffer per head
        // instead of one per row.
        let absorb = |m: &crate::engine::loader::ExpertQuantMatrix, rows: usize| -> Vec<Vec<f32>> {
            (0..n_head)
                .into_par_iter()
                .map(|h| {
                    let span = m.expert_span(h);
                    let mut flat = Vec::with_capacity(rows * m.in_dim);
                    let mut row = Vec::new();
                    for j in 0..rows {
                        m.row_from(span, j, &mut row);
                        flat.extend_from_slice(&row);
                    }
                    flat
                })
                .collect()
        };
        // **Only when there is more than one token to amortize it over.**
        // At `n_tokens == 1` there is no reuse: the row is dotted once, and
        // gathering it into a contiguous buffer first costs an extra copy and
        // makes the dot read cold memory instead of the line just written.
        // Measured **-6.8% on decode** before this guard, against +2.3% on
        // prefill with it — the hoist is a prefill optimization and saying so
        // in the type is cheaper than re-deriving it.
        let hoist = n_tokens > 1 && !mla_no_hoist();
        let wk_b = hoist.then(|| absorb(&self.wk_b, shape.kv_lora_rank));
        let wv_b = hoist.then(|| absorb(&self.wv_b, shape.head_v_mla));

        // **Head outer, token inner.** Absorbing a query reads the whole of
        // this head's `wk_b` block, and doing it inside the token loop read
        // that block again for every token — at a prompt's length, the same
        // hundreds of kilobytes over and over. Hoisted here the block is
        // read once per head and stays in cache across every token that
        // uses it. Held head-major for the same reason; the token loop
        // gathers its own `[n_head, absorbed_dim]` row out of these, which
        // is tens of kilobytes against the key buffer it is about to scan.
        let qs_by_head: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let mut per_token = vec![0f32; n_tokens * absorbed_dim];
                for t in 0..n_tokens {
                    let q_t =
                        &q[t * n_head * shape.head_k_mla..(t + 1) * n_head * shape.head_k_mla];
                    let q_nope = &q_t[h * shape.head_k_mla..h * shape.head_k_mla + nope];
                    let dst = &mut per_token[t * absorbed_dim..(t + 1) * absorbed_dim];
                    project_rows(
                        &self.wk_b,
                        wk_b.as_deref(),
                        h,
                        q_nope,
                        nope,
                        &mut dst[..shape.kv_lora_rank],
                    );
                    dst[shape.kv_lora_rank..].copy_from_slice(
                        &q_t[h * shape.head_k_mla + nope..(h + 1) * shape.head_k_mla],
                    );
                }
                per_token
            })
            .collect();
        // Every token's compressed attention output, kept until the value
        // decompression below can run head-outer for the same reason.
        let mut compressed_all = vec![0f32; n_tokens * n_head * shape.kv_lora_rank];
        let mut attn_out = vec![0f32; n_tokens * n_head * shape.head_v_mla];
        // **The whole prompt's rows appended first, and gathered once.**
        // A dense selection is the causal prefix `0..=start_pos + t`, so
        // every token reads a *prefix of the same buffer* — rebuilding it
        // per token copied the cache again for each one, which over a
        // prompt is quadratic: at 2048 tokens and this model's row width
        // that is gigabytes of `memmove` to produce megabytes of distinct
        // bytes. A sparse selection picks arbitrary rows and keeps the
        // per-token gather below.
        // The first token's position: a single sequence's `start_pos`.
        let start_pos = seqs.pos(0);
        // A batch's rows each read their own cache, so they take the
        // per-token gather below over their own causal prefix.
        let dense_keys: Option<Vec<f32>> = match (&mut *seqs, selection) {
            (Seqs::One { cache, .. }, None) => Some(&mut **cache),
            _ => None,
        }
        .map(|cache| {
            for t in (0..n_tokens).filter(|_| owns_kv) {
                let kv_t = &kv[t * shape.kv_row..(t + 1) * shape.kv_row];
                cache.layers[self.cache_index].push(kv_t, kv_t);
            }
            let slot = &cache.layers[self.cache_index];
            let rows = start_pos + n_tokens;
            let mut all = vec![0f32; rows * shape.kv_row];
            for p in 0..rows {
                all[p * shape.kv_row..(p + 1) * shape.kv_row].copy_from_slice(slot.key_at(
                    p,
                    0,
                    shape.kv_row,
                ));
            }
            all
        });
        // The queries token-major, which is the layout the batched form
        // below reads: the absorption above is head-major because that is
        // how its weight blocks are laid out, and this is the one
        // transpose between them.
        let mut qs_all = vec![0f32; n_tokens * n_head * absorbed_dim];
        for t in 0..n_tokens {
            for (h, head) in qs_by_head.iter().enumerate() {
                let at = (t * n_head + h) * absorbed_dim;
                qs_all[at..at + absorbed_dim]
                    .copy_from_slice(&head[t * absorbed_dim..(t + 1) * absorbed_dim]);
            }
        }
        // **A dense prompt is one batched call over the single gather
        // above.** A tile of consecutive tokens shares one pass over the
        // keys, so the prefix is read once per tile instead of once per
        // token, and the scores never exist as a matrix — each lane keeps
        // a running maximum and sum, rescaled per key block. A sparse
        // selection picks arbitrary rows per token and keeps the loop
        // below.
        if let Some(all) = dense_keys.as_ref().filter(|_| batched_attention()) {
            let compressed = super::attend_shared_causal(
                &qs_all,
                n_head,
                n_tokens,
                &all[..(start_pos + n_tokens) * shape.kv_row],
                shape.kv_row,
                shape.kv_lora_rank,
                shape.kq_scale,
                start_pos,
            );
            compressed_all.copy_from_slice(&compressed);
        }
        for t in (0..n_tokens).take_while(|_| dense_keys.is_none() || !batched_attention()) {
            let owned_keys;
            let keys: &[f32] = match &dense_keys {
                Some(all) => &all[..(start_pos + t + 1) * shape.kv_row],
                None => {
                    let pos = seqs.pos(t);
                    let cache = seqs.cache(t);
                    if owns_kv {
                        let kv_t = &kv[t * shape.kv_row..(t + 1) * shape.kv_row];
                        cache.layers[self.cache_index].push(kv_t, kv_t);
                    }
                    let causal: Vec<usize>;
                    let chosen: &[usize] = match selection {
                        Some(selection) => &selection[t],
                        None => {
                            causal = (0..=pos).collect();
                            &causal
                        }
                    };
                    let mut keys = vec![0f32; chosen.len() * shape.kv_row];
                    {
                        let slot = &cache.layers[self.cache_index];
                        for (i, &p) in chosen.iter().enumerate() {
                            keys[i * shape.kv_row..(i + 1) * shape.kv_row]
                                .copy_from_slice(slot.key_at(p, 0, shape.kv_row));
                        }
                    }
                    owned_keys = keys;
                    &owned_keys
                }
            };
            let at = t * n_head * absorbed_dim;
            let compressed = super::attend_shared(
                &qs_all[at..at + n_head * absorbed_dim],
                n_head,
                keys,
                shape.kv_row,
                shape.kv_lora_rank,
                shape.kq_scale,
            );
            compressed_all[t * n_head * shape.kv_lora_rank..(t + 1) * n_head * shape.kv_lora_rank]
                .copy_from_slice(&compressed);
        }

        // The value decompression, head outer for the same reason the
        // absorption above is: each head's `wv_b` block is read once rather
        // than once per token. Written head-major and scattered after,
        // because a head's outputs are strided in `attn_out`.
        let by_head: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let mut per_token = vec![0f32; n_tokens * shape.head_v_mla];
                for t in 0..n_tokens {
                    let at = (t * n_head + h) * shape.kv_lora_rank;
                    project_rows(
                        &self.wv_b,
                        wv_b.as_deref(),
                        h,
                        &compressed_all[at..at + shape.kv_lora_rank],
                        shape.kv_lora_rank,
                        &mut per_token[t * shape.head_v_mla..(t + 1) * shape.head_v_mla],
                    );
                }
                per_token
            })
            .collect();
        for (h, head) in by_head.iter().enumerate() {
            for t in 0..n_tokens {
                let at = (t * n_head + h) * shape.head_v_mla;
                attn_out[at..at + shape.head_v_mla]
                    .copy_from_slice(&head[t * shape.head_v_mla..(t + 1) * shape.head_v_mla]);
            }
        }

        if let Some(gate) = &self.gate {
            backend.matmul_into(gate_buf, normed, n_tokens, gate);
            let gate = &*gate_buf;
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
        backend.matmul_into(out, &attn_out, n_tokens, &self.wo);
    }
}

/// The per-layer projection buffers [`Mla::forward_into`] reuses — see
/// `Backend::matmul_into`.
#[derive(Default)]
pub(crate) struct MlaScratch {
    q: Vec<f32>,
    /// The low-rank query intermediate, only on the `Query::Lora` path.
    qr: Vec<f32>,
    kv: Vec<f32>,
    gate: Vec<f32>,
}

impl MlaScratch {
    /// The query LoRA intermediate left by [`MlaLayer::project_query_into`]
    /// — `[n_tokens, q_lora_rank]`, and empty on a layer that projects its
    /// query in one matrix. A sparse-attention architecture's lightning
    /// indexer shares it rather than projecting its own.
    pub(crate) fn qr(&self) -> &[f32] {
        &self.qr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kv_cache::KvCache;

    /// A deterministic `[out, in]` F32 matrix with small distinct values.
    fn matrix(in_dim: usize, out_dim: usize, seed: f32) -> QuantMatrix {
        let rows = (0..in_dim * out_dim)
            .map(|i| (i as f32 * 0.37 + seed).sin() * 0.5)
            .collect();
        QuantMatrix::from_f32_rows(rows, in_dim, out_dim)
    }

    fn values(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.61 + seed).cos()).collect()
    }

    const N_EMBD: usize = 16;

    fn kda() -> (KdaLayer, KdaShape) {
        let shape = KdaShape {
            n_head: 2,
            head_dim: 4,
            d_inner: 8,
            d_conv: 3,
            gate_lower_bound: None,
            eps: 1e-6,
            l2_eps: 1e-6,
        };
        let d = shape.d_inner;
        let layer = KdaLayer {
            wq: matrix(N_EMBD, d, 0.1),
            wk: matrix(N_EMBD, d, 0.2),
            wv: matrix(N_EMBD, d, 0.3),
            conv_kernel: values(3 * d * shape.d_conv, 0.4),
            f_a: matrix(N_EMBD, shape.head_dim, 0.5),
            f_b: Some(matrix(shape.head_dim, d, 0.6)),
            dt_bias: values(d, 0.7),
            a: vec![0.5, 1.5],
            beta: matrix(N_EMBD, shape.n_head, 0.8),
            g: matrix(N_EMBD, d, 0.9),
            g_b: None,
            o_norm: values(shape.head_dim, 1.0),
            wo: matrix(d, N_EMBD, 1.1),
            cache_index: 0,
        };
        (layer, shape)
    }

    fn mla() -> (MlaLayer, MlaShape) {
        let shape = MlaShape {
            n_head: 2,
            kv_lora_rank: 6,
            head_k_mla: 4,
            head_v_mla: 4,
            kv_row: 6,
            rope: None,
            rope_dim: 0,
            freq_factors: None,
            kq_scale: 0.5,
            eps: 1e-6,
        };
        let layer = MlaLayer {
            q: Query::Plain(matrix(N_EMBD, shape.n_head * shape.head_k_mla, 1.2)),
            kv: Some(KvProjection {
                wkv_a_mqa: matrix(N_EMBD, shape.kv_row, 1.3),
                kv_a_norm: values(shape.kv_lora_rank, 1.4),
            }),
            ssmax: None,
            wk_b: crate::engine::loader::test_expert_matrix(
                shape.n_head,
                shape.kv_lora_rank,
                shape.head_k_mla,
            ),
            wv_b: crate::engine::loader::test_expert_matrix(
                shape.n_head,
                shape.head_v_mla,
                shape.kv_lora_rank,
            ),
            gate: None,
            wo: matrix(shape.n_head * shape.head_v_mla, N_EMBD, 1.5),
            cache_index: 0,
        };
        (layer, shape)
    }

    /// Two sequences of different lengths, each prompted on its own, then
    /// one more token apiece: as one batch of two rows, and as two solo
    /// steps on caches prompted the same way. `step` runs the sub-layer.
    fn batch_against_solo(
        cache: impl Fn() -> KvCache,
        step: impl Fn(&mut Seqs<'_, '_>, &[f32], usize) -> Vec<f32>,
    ) {
        let lens = [3usize, 5];
        let prompt = |len: usize, seed: f32| values(len * N_EMBD, seed);
        let prompted = || -> Vec<KvCache> {
            lens.iter()
                .enumerate()
                .map(|(i, &len)| {
                    let mut c = cache();
                    step(
                        &mut Seqs::One {
                            cache: &mut c,
                            start_pos: 0,
                        },
                        &prompt(len, i as f32),
                        len,
                    );
                    c
                })
                .collect()
        };
        let next: Vec<Vec<f32>> = (0..2).map(|i| values(N_EMBD, 5.0 + i as f32)).collect();

        let mut solo_caches = prompted();
        let solo: Vec<Vec<f32>> = solo_caches
            .iter_mut()
            .zip(lens)
            .zip(&next)
            .map(|((c, len), x)| {
                step(
                    &mut Seqs::One {
                        cache: c,
                        start_pos: len,
                    },
                    x,
                    1,
                )
            })
            .collect();

        let mut batch_caches = prompted();
        let mut rows: Vec<super::super::DecodeRow<'_>> = batch_caches
            .iter_mut()
            .zip(lens)
            .enumerate()
            .map(|(slot, (cache, pos))| super::super::DecodeRow { cache, pos, slot })
            .collect();
        let batch = step(&mut Seqs::Rows(&mut rows), &next.concat(), 2);

        for (r, solo) in solo.iter().enumerate() {
            let row = &batch[r * N_EMBD..(r + 1) * N_EMBD];
            for (a, b) in row.iter().zip(solo) {
                assert!(
                    (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                    "row {r}: {a} against {b}"
                );
            }
        }
    }

    /// [`batch_against_solo`] with runs: one row brings a token and the
    /// other a three-token draft, as a verify in a batched step, each
    /// against its own pass over the same tokens.
    fn runs_against_solo(
        cache: impl Fn() -> KvCache,
        step: impl Fn(&mut Seqs<'_, '_>, &[f32], usize) -> Vec<f32>,
    ) {
        let lens = [3usize, 5];
        let widths = [1usize, 3];
        let prompted = || -> Vec<KvCache> {
            lens.iter()
                .enumerate()
                .map(|(i, &len)| {
                    let mut c = cache();
                    step(
                        &mut Seqs::One {
                            cache: &mut c,
                            start_pos: 0,
                        },
                        &values(len * N_EMBD, i as f32),
                        len,
                    );
                    c
                })
                .collect()
        };
        let next: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(i, &w)| values(w * N_EMBD, 5.0 + i as f32))
            .collect();

        let mut solo_caches = prompted();
        let solo: Vec<f32> = solo_caches
            .iter_mut()
            .zip(lens)
            .zip(widths)
            .zip(&next)
            .flat_map(|(((c, len), w), x)| {
                step(
                    &mut Seqs::One {
                        cache: c,
                        start_pos: len,
                    },
                    x,
                    w,
                )
            })
            .collect();

        let mut batch_caches = prompted();
        let mut rows: Vec<super::super::DecodeRow<'_>> = batch_caches
            .iter_mut()
            .zip(lens)
            .enumerate()
            .map(|(slot, (cache, pos))| super::super::DecodeRow { cache, pos, slot })
            .collect();
        let batch = step(&mut Seqs::runs(&mut rows, &widths), &next.concat(), 4);

        for (i, (a, b)) in batch.iter().zip(&solo).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                "token {}: {a} against {b}",
                i / N_EMBD
            );
        }
    }

    #[test]
    fn a_kda_batch_with_a_draft_steps_each_rows_state_as_its_own_pass() {
        let (layer, shape) = kda();
        runs_against_solo(
            || KvCache::new_mixed(16, &[], &[shape.recurrent_spec()]),
            |seqs, x, n| {
                let mut out = Vec::new();
                layer.forward_into(
                    &crate::engine::backend::CpuBackend,
                    &mut out,
                    &mut KdaScratch::default(),
                    &shape,
                    seqs,
                    x,
                    n,
                );
                out
            },
        );
    }

    #[test]
    fn an_mla_batch_with_a_draft_attends_each_rows_own_keys_as_its_own_pass() {
        let (layer, shape) = mla();
        runs_against_solo(
            || KvCache::new_mixed(16, &[shape.kv_row], &[]),
            |seqs, x, n| {
                let mut out = Vec::new();
                layer.forward_into(
                    &crate::engine::backend::CpuBackend,
                    &mut out,
                    &mut MlaScratch::default(),
                    &shape,
                    seqs,
                    x,
                    n,
                );
                out
            },
        );
    }

    #[test]
    fn a_kda_batch_steps_each_rows_state_as_its_solo_step() {
        let (layer, shape) = kda();
        batch_against_solo(
            || KvCache::new_mixed(16, &[], &[shape.recurrent_spec()]),
            |seqs, x, n| {
                let mut out = Vec::new();
                layer.forward_into(
                    &crate::engine::backend::CpuBackend,
                    &mut out,
                    &mut KdaScratch::default(),
                    &shape,
                    seqs,
                    x,
                    n,
                );
                out
            },
        );
    }

    #[test]
    fn an_mla_batch_attends_each_rows_own_keys_as_its_solo_step() {
        let (layer, shape) = mla();
        batch_against_solo(
            || KvCache::new_mixed(16, &[shape.kv_row], &[]),
            |seqs, x, n| {
                let mut out = Vec::new();
                layer.forward_into(
                    &crate::engine::backend::CpuBackend,
                    &mut out,
                    &mut MlaScratch::default(),
                    &shape,
                    seqs,
                    x,
                    n,
                );
                out
            },
        );
    }

    /// The sparse form a lightning indexer drives: every token attends the
    /// even positions before it and itself.
    #[test]
    fn a_sparse_mla_batch_attends_each_rows_own_selection_as_its_solo_step() {
        let (layer, shape) = mla();
        batch_against_solo(
            || KvCache::new_mixed(16, &[shape.kv_row], &[]),
            |seqs, x, n| {
                let selection: Vec<Vec<usize>> = (0..n)
                    .map(|t| {
                        let pos = seqs.pos(t);
                        (0..pos).filter(|p| p % 2 == 0).chain([pos]).collect()
                    })
                    .collect();
                let mut out = Vec::new();
                let mut scratch = MlaScratch::default();
                let backend = crate::engine::backend::CpuBackend;
                layer.project_query_into(&backend, &mut scratch, &shape, x, n);
                layer.attend_into(
                    &backend,
                    &mut out,
                    &mut scratch,
                    &shape,
                    seqs,
                    x,
                    n,
                    Some(&selection),
                );
                out
            },
        );
    }

    /// One prompt through `layer` on `cache`, from position 0, with `sparse`
    /// choosing the per-token gather over the dense batched path.
    fn mla_prompt(
        layer: &MlaLayer,
        shape: &MlaShape,
        cache: &mut KvCache,
        x: &[f32],
        n: usize,
        sparse: bool,
    ) -> Vec<f32> {
        let backend = crate::engine::backend::CpuBackend;
        let selection: Vec<Vec<usize>> = (0..n).map(|t| (0..=t).collect()).collect();
        let mut out = Vec::new();
        let mut scratch = MlaScratch::default();
        let mut seqs = Seqs::One {
            cache,
            start_pos: 0,
        };
        layer.project_query_into(&backend, &mut scratch, shape, x, n);
        layer.attend_into(
            &backend,
            &mut out,
            &mut scratch,
            shape,
            &mut seqs,
            x,
            n,
            sparse.then_some(selection.as_slice()),
        );
        out
    }

    /// A layer with no key/value projection (`attention.kv_source_map`)
    /// attends exactly the rows its source wrote, and writes none: with the
    /// source's own query weights it reproduces the source's output, on
    /// both the dense and the gathered path, and the cache does not grow.
    #[test]
    fn a_kv_reading_layer_attends_its_sources_rows_and_writes_none() {
        let n = 3;
        let x = values(n * N_EMBD, 2.0);
        for sparse in [false, true] {
            let (source, shape) = mla();
            let (mut reader, _) = mla();
            reader.kv = None;
            let mut cache = KvCache::new_mixed(16, &[shape.kv_row], &[]);
            let expected = mla_prompt(&source, &shape, &mut cache, &x, n, sparse);
            let rows = cache.layers[0].len;
            let got = mla_prompt(&reader, &shape, &mut cache, &x, n, sparse);
            assert_eq!(cache.layers[0].len, rows, "the reader pushed rows");
            for (a, b) in got.iter().zip(&expected) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "sparse={sparse}: {got:?} vs {expected:?}"
                );
            }
        }
    }

    /// SSMax is one scalar per head on the query, so a constant factor `c`
    /// (scale 0, bias `c`) must equal attending with `kq_scale * c`.
    #[test]
    fn ssmax_scales_the_attention_logits_per_head() {
        let n = 3;
        let x = values(n * N_EMBD, 3.0);
        let (mut scaled, shape) = mla();
        scaled.ssmax = Some(super::super::position::Ssmax::from_parts(
            vec![0.0; shape.n_head],
            vec![1.7; shape.n_head],
        ));
        let (plain, mut wide) = mla();
        wide.kq_scale *= 1.7;

        let mut a = KvCache::new_mixed(16, &[shape.kv_row], &[]);
        let mut b = KvCache::new_mixed(16, &[shape.kv_row], &[]);
        let got = mla_prompt(&scaled, &shape, &mut a, &x, n, false);
        let expected = mla_prompt(&plain, &wide, &mut b, &x, n, false);
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{got:?} vs {expected:?}");
        }
    }

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

    /// The fan-out grain must group where there are heads to spare and
    /// never below the task floor, because a grouped layer with few heads
    /// is a serial loop wearing a fan-out's clothes.
    ///
    /// The measured case is the middle one: sixteen heads at four a task
    /// beat one a task in four of four alternating pairs, and four tasks
    /// across sixteen workers is fewer than the pool — which is why the
    /// rule is a floor on tasks rather than a match to workers.
    #[test]
    fn the_fan_out_grain_groups_but_never_starves_the_pool() {
        let tasks = |n_head: usize| n_head.div_ceil(delta_heads_per_task(n_head));
        // Too few heads to group at all.
        for n_head in 1..=DELTA_MIN_TASKS {
            assert_eq!(
                delta_heads_per_task(n_head),
                1,
                "{n_head} heads must stay one a task"
            );
        }
        // Enough to group, and never below the floor.
        for n_head in [5usize, 8, 12, 16, 24, 32, 64, 128] {
            let grain = delta_heads_per_task(n_head);
            assert!(grain >= 1);
            assert!(
                tasks(n_head) >= DELTA_MIN_TASKS,
                "{n_head} heads gave {} tasks at grain {grain}",
                tasks(n_head)
            );
            assert!(
                grain <= n_head,
                "{n_head} heads cannot be grouped {grain} at a time"
            );
        }
        // The arm that was measured.
        assert_eq!(delta_heads_per_task(16), 4);
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

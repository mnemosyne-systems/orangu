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

//! One implementor per architecture family - `llama` (GQA/RoPE/RMSNorm/
//! SwiGLU), `gemma` (soft-capping/sliding-window/GEGLU), `phi` (fused
//! QKV and gate/up, LongRoPE), `mistral` (YaRN, read head width,
//! attention temperature scaling), `qwen35` (hybrid full-attention/gated-
//! DeltaNet, dense FFN), `qwen35moe` (the same hybrid attention shape,
//! mixture-of-experts FFN), `qwen3next` (the Qwen3-Next hybrid
//! attention + MoE path — those three sharing one trunk in `qwen_hybrid`,
//! which is not itself an architecture), `deepseek4` (DeepSeek-V4:
//! hyper-connections,
//! shared-key attention over a sliding window plus compressed blocks, and
//! hash-routed experts), `glm` (GLM with DeepSeek sparse attention:
//! absorbed multi-head latent attention plus a lightning indexer), `kimi3`
//! (Kimi-K3: hybrid delta-net/latent attention with cross-layer residuals
//! and a latent MoE), `muse` (Muse-Glimmer: `llama`'s dense GQA block plus
//! gemma-style sandwich norms, a sigmoid output gate on attention, and an
//! alternating pattern of rotated sliding-window and unrotated
//! full-attention layers), `inkling` (Inkling: no rotation at all — a
//! learned relative-position bias plus causal short convolutions — over
//! sigmoid-routed experts that share their normalization with the shared
//! ones), `nemotron` (Nemotron-H: blocks that are a *single* sub-layer each
//! — a selective state-space mixer, an unrotated attention, or a squared-
//! ReLU mixture-of-experts FFN — rather than the usual attention-plus-FFN
//! pair), `bailingmoe` (Ling 3.0: the same three-in-four delta-net /
//! latent-attention trunk as `kimi3` — the two sharing one implementation
//! in `kda`, which is not itself an architecture — but with a *rotated*
//! latent attention, a per-head output gate, and group-limited routed
//! experts), and `dflash`
//! (DeepSeek draft sidecars, served through the target model they draft
//! for).

pub mod bailingmoe;
pub mod deepseek4;
pub mod dflash;
pub mod gemma;
pub mod glm;
pub mod inkling;
pub mod kda;
pub mod kimi3;
pub mod llama;
pub mod mistral;
pub mod muse;
pub mod nemotron;
pub mod phi;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3next;
pub mod qwen_hybrid;

use crate::engine::kv_cache::KvCache;
use crate::engine::loader::ModelConfig;
use crate::engine::tensor;
use crate::engine::vecdot;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;

/// The `k` highest-scoring indices, highest first. Ties keep their input
/// order.
///
/// ggml's `ggml_argsort_top_k`, which the sparse-attention architectures
/// (`deepseek4`, `glm`) use both to pick a lightning indexer's attended
/// positions and to pick a MoE layer's experts (as does every other MoE
/// architecture here, through [`ExpertRouting::route`]).
pub(crate) fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k.min(indexed.len()));
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Weightless RMSNorm over each `dim`-wide row of `x` — ggml's bare
/// `ggml_rms_norm`, with no learned weight, which several architectures
/// need alongside the usual weighted `tensor::rmsnorm_inplace`:
/// `deepseek4`'s hyper-connection projection input and per-head query norm,
/// and `kimi3`'s cross-layer residual scores.
pub(crate) fn rms_norm_rows(x: &mut [f32], dim: usize, eps: f32) {
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_mut(dim) {
        let scale = rms_norm_scale(row_mean_sq(row), eps);
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// [`rms_norm_rows`] from `src` into `dst` — the out-of-place form, for the
/// callers that need the input preserved and were copying it first.
///
/// Same reasoning as `tensor::rmsnorm_into`, and same contract: the values
/// are bit-identical to normalizing a copy, because the row kernel is the
/// same expression in the same order. `dst` is a caller-owned buffer reused
/// across layers.
pub(crate) fn rms_norm_rows_into(dst: &mut Vec<f32>, src: &[f32], dim: usize, eps: f32) {
    debug_assert_eq!(src.len() % dim, 0);
    dst.resize(src.len(), 0.0);
    for (out, row) in dst.chunks_mut(dim).zip(src.chunks(dim)) {
        let scale = rms_norm_scale(row_mean_sq(row), eps);
        for (o, v) in out.iter_mut().zip(row.iter()) {
            *o = *v * scale;
        }
    }
}

/// `1/sqrt(mean(x^2) + eps)` — the scale [`rms_norm_rows`] multiplies by,
/// exposed separately for callers that only need the *scaled dot product*
/// of a row with a weight vector and would otherwise normalize a copy of
/// the row just to throw it away (`kimi3`'s residual scores).
pub(crate) fn rms_norm_scale(mean_sq: f32, eps: f32) -> f32 {
    1.0 / (mean_sq + eps).sqrt()
}

pub(crate) fn row_mean_sq(row: &[f32]) -> f32 {
    row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32
}

/// One head's attention over an already-gathered, already-masked key set.
///
/// `keys` is `[n_keys, key_dim]` — the *selected* keys, contiguous, so the
/// mask is expressed by which rows the caller gathered rather than by
/// `-inf` entries. Each key's value is the leading `value_dim` of its own
/// row, which covers both callers: `deepseek4`, whose value *is* the whole
/// key, and `glm`, whose MLA key is `[compressed-KV | rotary]` and whose
/// value is the compressed-KV part alone.
///
/// `sink`, when given, is an extra logit that takes softmax mass without
/// contributing a value — ggml's `soft_max_ext` sink (`src2`), which folds
/// into the denominator only.
pub(crate) fn attend(
    q: &[f32],
    keys: &[f32],
    key_dim: usize,
    value_dim: usize,
    scale: f32,
    sink: Option<f32>,
) -> Vec<f32> {
    debug_assert!(value_dim <= key_dim);
    let n_keys = keys.len() / key_dim;
    let mut scores: Vec<f32> = (0..n_keys)
        .map(|i| tensor::dot(q, &keys[i * key_dim..(i + 1) * key_dim]) * scale)
        .collect();
    let mut max = sink.unwrap_or(f32::NEG_INFINITY);
    for &s in &scores {
        max = max.max(s);
    }
    let mut sum = sink.map_or(0.0, |s| (s - max).exp());
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let inv = 1.0 / sum;
    let mut out = vec![0.0; value_dim];
    for (i, &w) in scores.iter().enumerate() {
        tensor::axpy_inplace(
            &mut out,
            &keys[i * key_dim..i * key_dim + value_dim],
            w * inv,
        );
    }
    out
}

/// How a MoE layer turns its router's logits into probabilities —
/// `<arch>.expert_gating_func`, upstream's
/// `llama_expert_gating_func_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertGating {
    /// `2` — GLM's `sigmoid`.
    Sigmoid,
    /// `4` — DeepSeek-V4's `sqrt(softplus(x))`.
    SqrtSoftplus,
}

impl ExpertGating {
    pub(crate) fn from_gguf(value: u64) -> Result<Self> {
        match value {
            2 => Ok(Self::Sigmoid),
            4 => Ok(Self::SqrtSoftplus),
            other => anyhow::bail!(
                "expert_gating_func {other} is not implemented (2 = sigmoid, 4 = sqrt-softplus)"
            ),
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Sigmoid => tensor::sigmoid(x),
            Self::SqrtSoftplus => tensor::softplus(x).sqrt(),
        }
    }
}

/// Group-limited (node-limited) expert selection — DeepSeek-V3's, which
/// `bailingmoe3` inherits: the experts are cut into
/// `<arch>.expert_group_count` equal, *contiguous* groups, only the best
/// `<arch>.expert_group_used_count` of them may contribute, and the top-k
/// then runs over the survivors alone.
///
/// A group's score is the sum of its **two** highest member scores, not its
/// best or its mean — upstream's `build_moe_ffn` takes
/// `ggml_argsort_top_k(.., 2)` per group and sums. That detail decides
/// which groups survive, and every plausible substitute for it produces
/// fluent, subtly wrong output rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertGroups {
    /// How many groups the experts are cut into.
    pub count: usize,
    /// How many of them may contribute to one token.
    pub used: usize,
    /// `n_expert / count` — the size of each group.
    pub size: usize,
}

impl ExpertGroups {
    /// Reads `<arch>.expert_group_count` / `expert_group_used_count`,
    /// returning `None` for the ordinary ungrouped case (the key absent, or
    /// a single group, which is the same thing).
    pub(crate) fn from_gguf(
        loaded: &crate::engine::loader::LoadedModel,
        n_expert: usize,
    ) -> Result<Option<Self>> {
        let count = loaded.metadata_u64("expert_group_count").unwrap_or(1) as usize;
        if count <= 1 {
            return Ok(None);
        }
        let used = loaded.metadata_u64("expert_group_used_count").unwrap_or(1) as usize;
        anyhow::ensure!(
            n_expert > 0 && n_expert.is_multiple_of(count),
            "expert_group_count {count} does not divide expert_count {n_expert}"
        );
        anyhow::ensure!(
            used >= 1 && used <= count,
            "expert_group_used_count {used} is not between 1 and expert_group_count {count}"
        );
        Ok(Some(Self {
            count,
            used,
            size: n_expert / count,
        }))
    }

    /// The groups this token may draw from, best first — each group scored
    /// by the sum of its two highest selection scores.
    fn survivors(&self, selection: &[f32]) -> Vec<usize> {
        let scores: Vec<f32> = (0..self.count)
            .map(|g| {
                let members = &selection[g * self.size..(g + 1) * self.size];
                let mut best = f32::NEG_INFINITY;
                let mut second = f32::NEG_INFINITY;
                for &s in members {
                    if s > best {
                        second = best;
                        best = s;
                    } else if s > second {
                        second = s;
                    }
                }
                // A group of one has no second member to add; upstream's
                // top-2 over a one-element row repeats nothing either, it
                // simply has one row to sum.
                if second.is_finite() {
                    best + second
                } else {
                    best
                }
            })
            .collect();
        top_k_indices(&scores, self.used)
    }
}

/// A MoE layer's routing rules, all read from the file's metadata.
pub(crate) struct ExpertRouting {
    pub n_expert_used: usize,
    pub gating: ExpertGating,
    /// `<arch>.expert_weights_norm` — renormalize the selected experts'
    /// weights so they sum to one.
    pub weights_norm: bool,
    /// `<arch>.expert_weights_scale`, applied after any renormalization.
    pub weights_scale: f32,
    /// Group-limited selection, for the architectures that declare it.
    /// `None` is the ordinary "every expert is a candidate" case.
    pub groups: Option<ExpertGroups>,
}

/// The smallest denominator `build_moe_ffn` will divide expert weights by
/// (`ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)` — the smallest
/// normal `f16`).
const MIN_EXPERT_WEIGHT_SUM: f32 = 6.103_515_6e-5;

impl ExpertRouting {
    /// Which experts one token uses and with what weight, following
    /// `llm_graph_context::build_moe_ffn` exactly.
    ///
    /// The `bias` (`exp_probs_b`) steers the *selection* only — the weights
    /// come from the unbiased probabilities, which is the whole point of
    /// DeepSeek-V3's auxiliary-loss-free balancing and is easy to get wrong
    /// in a way that still produces fluent text. `forced` overrides the
    /// selection entirely, for an architecture whose experts are chosen by
    /// a lookup table rather than by score (`deepseek4`'s hash-routed
    /// layers).
    pub(crate) fn route(
        &self,
        logits: &[f32],
        bias: Option<&[f32]>,
        forced: Option<&[i32]>,
    ) -> (Vec<usize>, Vec<f32>) {
        let probs: Vec<f32> = logits.iter().map(|&l| self.gating.apply(l)).collect();
        let selected: Vec<usize> = match forced {
            Some(experts) => experts.iter().map(|&e| e as usize).collect(),
            None => {
                let mut selection = probs.clone();
                if let Some(bias) = bias {
                    tensor::add_inplace(&mut selection, bias);
                }
                // Group-limited selection masks the losing groups to
                // `-inf` *before* the top-k, so a strong expert in a weak
                // group cannot be picked. The mask is applied to the
                // biased scores, which is what upstream sorts on.
                if let Some(groups) = &self.groups {
                    let survivors = groups.survivors(&selection);
                    let mut masked = vec![f32::NEG_INFINITY; selection.len()];
                    for g in survivors {
                        let span = g * groups.size..(g + 1) * groups.size;
                        masked[span.clone()].copy_from_slice(&selection[span]);
                    }
                    selection = masked;
                }
                top_k_indices(&selection, self.n_expert_used)
            }
        };
        let mut weights: Vec<f32> = selected.iter().map(|&e| probs[e]).collect();
        if self.weights_norm {
            let denom = weights
                .iter()
                .copied()
                .sum::<f32>()
                .max(MIN_EXPERT_WEIGHT_SUM);
            for w in &mut weights {
                *w /= denom;
            }
        }
        if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
            for w in &mut weights {
                *w *= self.weights_scale;
            }
        }
        (selected, weights)
    }
}
/// Consecutive layers grouped by the device holding them, in layer order —
/// `[(0, 0..24)]` for an unsplit model, one entry per device for a split
/// one.
///
/// The one piece of the split decode path that is shared across
/// architectures, because it is the piece whose failure is silent: a
/// mis-grouped run records a layer's fused chain against the wrong card's
/// weights, which is wrong output rather than a crash.
///
/// `None` when any layer has no `wgpu` backend behind it. That is how a CPU
/// overflow tier declines the fused chain — it is a GPU chain, and a run of
/// layers on the host has to take the step-by-step route — and it is also
/// the answer on a machine with no GPU at all.
pub(crate) fn decode_device_runs(
    backend: &dyn crate::engine::backend::Backend,
    layer_devices: impl Iterator<Item = usize>,
) -> Option<Vec<(usize, std::ops::Range<usize>)>> {
    let mut runs: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    for (il, device) in layer_devices.enumerate() {
        backend.as_wgpu_on(device)?;
        match runs.last_mut() {
            Some((prev, range)) if *prev == device => range.end = il + 1,
            _ => runs.push((device, il..il + 1)),
        }
    }
    (!runs.is_empty()).then_some(runs)
}

/// [`matmul_batch_mixed`] when routed-expert weights are streamed: the ops
/// that *can* stream do, and the rest run on the CPU.
///
/// An op wider than the stripe threshold cannot stream, because striping
/// re-enters the dispatch once per token range and each entry rewinds the
/// region out from under the last range's weights. What it must **not** do
/// instead is fall through to the ordinary device path, whose arena never
/// evicts: one such expert per layer is enough to rebuild, permanently, the
/// residency streaming exists to avoid. Measured, on a 3.98 GiB card: 7.0
/// GiB of weights in the arena after a single 512-token prefill, and prefill
/// 3.0x slower than the host path — because *every* layer had one hot expert
/// over the threshold, which sent that layer's whole batch down the
/// non-streamed path.
///
/// The CPU is the right home for those few: it is where all of them run when
/// `gpu_experts` is off at all, and there are only ever a handful per layer
/// (an expert needs more than [`max_matmul_tokens_per_submission`] tokens
/// routed to it to qualify).
fn streamed_batch_mixed(
    vulkan: &crate::engine::backend::vulkan::VulkanBackend,
    ops: &[crate::engine::backend::MatmulOp<'_>],
    limit: usize,
) -> Vec<Vec<f32>> {
    use crate::engine::backend::{Backend as _, CpuBackend, MatmulOp};
    use rayon::prelude::*;

    let (narrow, wide): (Vec<usize>, Vec<usize>) =
        (0..ops.len()).partition(|&i| ops[i].n_tokens <= limit);
    let narrow_ops: Vec<MatmulOp<'_>> = narrow
        .iter()
        .map(|&i| MatmulOp {
            x: ops[i].x,
            n_tokens: ops[i].n_tokens,
            w: ops[i].w,
        })
        .collect();

    // The device batch blocks on its own readback, so the host ops fill that
    // wait instead of following it — the same overlap the routed and shared
    // branches already use.
    let (streamed, hosted) = rayon::join(
        || vulkan.matmul_batch_streamed(&narrow_ops),
        || {
            wide.par_iter()
                .map(|&i| CpuBackend.matmul(ops[i].x, ops[i].n_tokens, ops[i].w))
                .collect::<Vec<_>>()
        },
    );

    restore_order(ops.len(), vec![(narrow, streamed), (wide, hosted)])
}

/// Puts results computed in two separate groups back in the caller's order.
///
/// Separated out because this is the step that fails **silently**: every
/// expert still gets a plausible vector, just the wrong one, and the model
/// goes on producing fluent text from it. A panic or a wrong length would
/// have announced itself; a permutation does not.
fn restore_order(len: usize, groups: Vec<(Vec<usize>, Vec<Vec<f32>>)>) -> Vec<Vec<f32>> {
    let mut out: Vec<Option<Vec<f32>>> = (0..len).map(|_| None).collect();
    for (slots, results) in groups {
        assert_eq!(
            slots.len(),
            results.len(),
            "a group came back with a different number of results than it was given"
        );
        for (slot, result) in slots.into_iter().zip(results) {
            out[slot] = Some(result);
        }
    }
    out.into_iter()
        .map(|o| o.expect("every slot belongs to exactly one group"))
        .collect()
}

/// Issues `ops` as few `Backend::matmul_batch` calls as the trait allows —
/// one per distinct `n_tokens`, since a batch requires a uniform token
/// count — returning results in the caller's order.
///
/// At decode every routed expert has exactly one token, so this is one
/// call. At prefill the groups differ in size and it is one call per
/// distinct size, which is still far fewer than one per expert.
fn matmul_batch_mixed(
    backend: &dyn crate::engine::backend::Backend,
    ops: &[crate::engine::backend::MatmulOp<'_>],
) -> Vec<Vec<f32>> {
    if ops.is_empty() {
        return Vec::new();
    }
    // **One submission for the whole layer**, whatever mixture of token
    // counts its experts were routed.
    //
    // This used to bucket the ops by `n_tokens` and call `matmul_batch` once
    // per bucket, because that call demanded a single shared width. At
    // decode that is one bucket and costs nothing; at prefill every expert
    // has a different number of tokens routed to it, so a layer became
    // dozens of blocking submissions with a readback each — and the device
    // expert path lost to the host by 2x for that reason rather than any
    // arithmetic one. `Backend::matmul_batch` now takes a mixture directly
    // (every op was already resourced independently), so the bucketing is
    // gone and with it the per-bucket round trips.
    //
    // Above that threshold the old shape is still required: striping a wide
    // op into token ranges is a whole-batch decision and needs one width, so
    // a batch containing one is bucketed as before. That is not a rare edge
    // — a long enough prompt routes more than `max_matmul_tokens_per_
    // submission` tokens to a single expert — and without this it would be
    // an assertion failure rather than a slower path.
    let limit = crate::engine::backend::vulkan::max_matmul_tokens_per_submission();
    if expert_streaming()
        && let Some(vulkan) = backend.as_wgpu()
    {
        return streamed_batch_mixed(vulkan, ops, limit);
    }
    if ops.iter().all(|op| op.n_tokens <= limit) {
        return backend.matmul_batch(ops);
    }
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, op) in ops.iter().enumerate() {
        buckets.entry(op.n_tokens).or_default().push(index);
    }
    let mut widths: Vec<usize> = buckets.keys().copied().collect();
    widths.sort_unstable();
    let mut out: Vec<Option<Vec<f32>>> = (0..ops.len()).map(|_| None).collect();
    for width in widths {
        let indices = &buckets[&width];
        let group: Vec<crate::engine::backend::MatmulOp<'_>> = indices
            .iter()
            .map(|&i| crate::engine::backend::MatmulOp {
                x: ops[i].x,
                n_tokens: ops[i].n_tokens,
                w: ops[i].w,
            })
            .collect();
        for (slot, result) in indices.iter().zip(backend.matmul_batch(&group)) {
            out[*slot] = Some(result);
        }
    }
    out.into_iter()
        .map(|o| o.expect("every op is in exactly one bucket"))
        .collect()
}

/// One of the three matmuls a routed expert performs, named as a **row
/// range** of a stacked expert tensor plus an optional per-expert output
/// scalar.
///
/// The row range is what lets one tensor serve two projections. Several
/// architectures ship the gate and up weights fused into a single
/// `ffn_gate_up_exps` whose first half is the gate and second half the up;
/// naming them as two ranges of one tensor is the difference between the
/// batched path covering those models and not.
///
/// `scale` multiplies this projection's **output**, never its rows.
/// `(x · row) * s` and `x · (row * s)` are different `f32`s — the first
/// rounds one product, the second rounds every term of the accumulation —
/// and the first is what the reference computes, so it is what this has to
/// compute. Folding the scalar into the weights would be cheaper and wrong.
pub(crate) struct ExpertProjection<'a> {
    pub exps: &'a crate::engine::loader::ExpertQuantMatrix,
    /// First row of this projection within one expert's rows.
    pub first_row: usize,
    /// Row count — this projection's output width.
    pub n_rows: usize,
    /// Per-expert scalar applied to the output, indexed by expert.
    pub scale: Option<&'a [f32]>,
}

impl<'a> ExpertProjection<'a> {
    /// A projection that is a whole expert tensor, unscaled — the shape
    /// every architecture had before fused gate/up tensors needed naming.
    pub fn whole(exps: &'a crate::engine::loader::ExpertQuantMatrix) -> Self {
        Self {
            exps,
            first_row: 0,
            n_rows: exps.out_dim,
            scale: None,
        }
    }
}

/// [`evaluate_routed_experts`] with the three expert projections **batched
/// across experts** instead of issued one expert at a time.
///
/// Same contract as the per-expert form — contributions come back
/// `[token][selection rank]`, in the order the router picked them, so the
/// caller's summation is untouched — and the same grouping, so each expert's
/// weights are still read once for every token that selected it.
///
/// What changes is the dispatch: a decode step's routed experts become
/// three `matmul_batch` calls per layer rather than three per expert. That
/// is the ~8x reduction a GPU expert path needs to be worth measuring at
/// all; the blocking submit-and-readback per expert is what made the naive
/// dispatch lose to the host.
///
/// **Only for the GPU path.** It goes through `Backend::matmul`, which
/// reads the weights straight from the mapping — bypassing
/// `engine::expert_store`'s residency tier and its accounting. That is
/// correct for weights living in VRAM and wrong for the host path, which
/// keeps [`project_expert`]. It is also *sequential* over the groups it has
/// to fall back on, where [`evaluate_routed_experts`] runs them under a
/// `par_iter` — so a caller that takes this path with nothing resident does
/// strictly more work than one that never called it. Gate on
/// [`gpu_experts`], as every caller does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_routed_experts_batched(
    backend: &dyn crate::engine::backend::Backend,
    selection: &[Vec<(usize, f32)>],
    hidden: &[f32],
    n_embd: usize,
    gate_exps: &crate::engine::loader::ExpertQuantMatrix,
    up_exps: &crate::engine::loader::ExpertQuantMatrix,
    down_exps: &crate::engine::loader::ExpertQuantMatrix,
    activate: impl Fn(&[f32], &[f32]) -> Vec<f32> + Sync,
) -> Vec<Vec<Vec<f32>>> {
    evaluate_routed_experts_batched_views(
        backend,
        selection,
        hidden,
        n_embd,
        Some(&ExpertProjection::whole(gate_exps)),
        &ExpertProjection::whole(up_exps),
        &ExpertProjection::whole(down_exps),
        activate,
    )
}

/// [`evaluate_routed_experts_batched`] for a **gate-less** expert — one
/// projection into the hidden width, an activation on it alone, then the
/// down projection.
///
/// `nemotron`'s squared-ReLU experts are the only ones here with this shape,
/// and it was the last architecture on the unbatched per-expert path for no
/// better reason than that the batched helper had `gate` in its signature.
pub(crate) fn evaluate_routed_experts_batched_gateless(
    backend: &dyn crate::engine::backend::Backend,
    selection: &[Vec<(usize, f32)>],
    hidden: &[f32],
    n_embd: usize,
    up_exps: &crate::engine::loader::ExpertQuantMatrix,
    down_exps: &crate::engine::loader::ExpertQuantMatrix,
    activate: impl Fn(&[f32]) -> Vec<f32> + Sync,
) -> Vec<Vec<Vec<f32>>> {
    evaluate_routed_experts_batched_views(
        backend,
        selection,
        hidden,
        n_embd,
        None,
        &ExpertProjection::whole(up_exps),
        &ExpertProjection::whole(down_exps),
        |_, up| activate(up),
    )
}

/// One host-evaluated group's gate and up projections, tagged with the group
/// it belongs to — what the parallel fallback in
/// [`evaluate_routed_experts_batched_views`] hands back, since a `par_iter`
/// cannot write into the slot vector directly.
type HostProjections = (usize, (Vec<f32>, Vec<f32>));

/// [`evaluate_routed_experts_batched`] over row ranges rather than whole
/// tensors, so a fused gate/up tensor and per-expert output scalars are
/// expressible.
///
/// Everything the wrapper's documentation says applies here; this is the
/// implementation and that is the common case of it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_routed_experts_batched_views(
    backend: &dyn crate::engine::backend::Backend,
    selection: &[Vec<(usize, f32)>],
    hidden: &[f32],
    n_embd: usize,
    gate: Option<&ExpertProjection<'_>>,
    up: &ExpertProjection<'_>,
    down: &ExpertProjection<'_>,
    activate: impl Fn(&[f32], &[f32]) -> Vec<f32> + Sync,
) -> Vec<Vec<Vec<f32>>> {
    use crate::engine::backend::MatmulOp;

    // A gate-less expert — `nemotron`'s squared-ReLU FFN is `down(relu(up
    // (x))^2)`, with no second projection to multiply against — is the same
    // pipeline with one projection instead of two, so it is expressed as
    // `None` rather than as a second copy of this function. `activate` is
    // then called with an empty gate slice, and the up projection is what
    // sets the hidden width.
    let ops_per_group = if gate.is_some() { 2 } else { 1 };

    // Same first-seen grouping as `evaluate_routed_experts`, and for the
    // same reason: it keeps the evaluation order reproducible from the
    // routing alone.
    let mut experts: Vec<usize> = Vec::new();
    let mut members: Vec<Vec<(usize, f32)>> = Vec::new();
    let mut ranks: Vec<Vec<usize>> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for (token, picks) in selection.iter().enumerate() {
        for (rank, &(expert, weight)) in picks.iter().enumerate() {
            let group = *group_of.entry(expert).or_insert_with(|| {
                experts.push(expert);
                members.push(Vec::new());
                ranks.push(Vec::new());
                experts.len() - 1
            });
            members[group].push((token, weight));
            ranks[group].push(rank);
        }
    }
    if experts.is_empty() {
        return selection
            .iter()
            .map(|picks| vec![Vec::new(); picks.len()])
            .collect();
    }

    // Each group's inputs, contiguous: the tokens that routed to one expert
    // are scattered through `hidden`, and a `MatmulOp` wants one
    // `[n_tokens, n_embd]` run.
    // Only the experts a device actually holds take the batched GPU path;
    // the rest go through `project_expert`, which reads them through
    // `engine::expert_store`'s residency tier as it always has. Without
    // this split the batch would pull every routed expert into an arena
    // that never evicts — the tier would be unbounded, which is the one
    // thing it exists not to be.
    // `ORANGU_MOE_FORCE_HOST_GROUPS=1` sends every group down the host path
    // while keeping this batched helper's own structure — the control that
    // separates "the device is slow" from "getting here cost something".
    // Entering this function at all means gathering each group's activations
    // into a fresh contiguous buffer (`xs` below), which the per-expert path
    // never does: it hands `project_expert` slices of `hidden` directly. That
    // gather is charged to every group, not only the resident ones, so it is
    // not visible in any comparison that toggles residency.
    let force_host = crate::engine::env::flag_on("ORANGU_MOE_FORCE_HOST_GROUPS");
    // **With streaming on, residency stops deciding this.** The permanent
    // weight arena never evicts, so `is_device_resident` is a plan made
    // before the first token against total VRAM — on a card holding 4.00 GiB
    // against 12.0 GiB of experts that caps the device at ~12% of them
    // however fast it is. Streaming holds only *this call's* experts in a
    // bounded region that rewinds, so the question becomes whether the batch
    // fits rather than whether the model does, and every group can go.
    let streamed = expert_streaming();
    let on_device: Vec<bool> = experts
        .iter()
        .map(|&e| {
            !force_host
                && (streamed
                    || (gate.is_none_or(|g| g.exps.is_device_resident(e))
                        && up.exps.is_device_resident(e)
                        && down.exps.is_device_resident(e)))
        })
        .collect();

    // **Gathered only for the groups that are going to the device.** A
    // `MatmulOp` needs one contiguous `[n_tokens, n_embd]` run to upload, and
    // a group's tokens are scattered through `hidden`; the host path needs no
    // such thing, because `project_expert` takes `&[&[f32]]` and is perfectly
    // happy with slices of `hidden` itself — which is exactly what the
    // per-expert path hands it.
    //
    // Gathering for every group made entering this function cost **+253 ms**
    // of a 850 ms expert stage on a `pp` 512 prefill, charged whether or not
    // a single expert was device-resident. That overhead is what made the
    // device path look like a loss: with it removed from the comparison,
    // moving 11.8% of experts to the device *saves* 96 ms rather than costing
    // 126. Every earlier measurement of "the device expert path" on this
    // model was really measuring this gather.
    let xs: Vec<Vec<f32>> = members
        .iter()
        .zip(on_device.iter())
        .map(|(members, &device)| {
            if !device {
                return Vec::new();
            }
            let mut x = Vec::with_capacity(members.len() * n_embd);
            for &(token, _) in members {
                x.extend_from_slice(&hidden[token * n_embd..(token + 1) * n_embd]);
            }
            x
        })
        .collect();

    // Views first, then ops: a `MatmulOp` borrows its weight. `rows` is what
    // makes a fused gate/up tensor two projections rather than one.
    let gate_views: Vec<_> = gate
        .map(|gate| {
            experts
                .iter()
                .map(|&e| gate.exps.expert_matrix(e).rows(gate.first_row, gate.n_rows))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let up_views: Vec<_> = experts
        .iter()
        .map(|&e| up.exps.expert_matrix(e).rows(up.first_row, up.n_rows))
        .collect();
    let mut ops: Vec<MatmulOp<'_>> = Vec::new();
    let mut op_group: Vec<usize> = Vec::new();
    for group in 0..experts.len() {
        if !on_device[group] {
            continue;
        }
        if gate.is_some() {
            ops.push(MatmulOp {
                x: &xs[group],
                n_tokens: members[group].len(),
                w: &gate_views[group],
            });
        }
        ops.push(MatmulOp {
            x: &xs[group],
            n_tokens: members[group].len(),
            w: &up_views[group],
        });
        op_group.push(group);
    }
    // **The device batch and the host remainder run concurrently.** They ran
    // one after the other, which defeats the only reason this function
    // exists: it splits a layer's experts between two *different* processors,
    // and a split whose halves are serialized costs the sum of both instead
    // of the larger. That is not a small effect at a partial residency —
    // with 11.8% of experts on the device it made the whole layer 14%
    // *slower* than leaving every expert on the host, because the device's
    // share was added to the host's rather than hidden behind it.
    let (batched, host_gate_up) = rayon::join(
        || matmul_batch_mixed(backend, &ops),
        || -> Vec<HostProjections> {
            (0..experts.len())
                .filter(|&group| !on_device[group])
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|group| {
                    let inputs: Vec<&[f32]> = members[group]
                        .iter()
                        .map(|&(token, _)| &hidden[token * n_embd..(token + 1) * n_embd])
                        .collect();
                    let gate_out = gate
                        .map(|gate| {
                            project_expert(
                                backend,
                                gate.exps,
                                experts[group],
                                gate.first_row,
                                gate.n_rows,
                                &inputs,
                            )
                            .concat()
                        })
                        .unwrap_or_default();
                    let up_out = project_expert(
                        backend,
                        up.exps,
                        experts[group],
                        up.first_row,
                        up.n_rows,
                        &inputs,
                    );
                    (group, (gate_out, up_out.concat()))
                })
                .collect()
        },
    );
    // Back into per-group slots, so the activation loop reads the same way
    // whichever path produced a group's projections.
    let mut slots: Vec<Option<(Vec<f32>, Vec<f32>)>> = (0..experts.len()).map(|_| None).collect();
    for (slot, group) in op_group.iter().enumerate() {
        let base = slot * ops_per_group;
        let gate_out = if gate.is_some() {
            batched[base].clone()
        } else {
            Vec::new()
        };
        slots[*group] = Some((gate_out, batched[base + ops_per_group - 1].clone()));
    }
    for (group, projections) in host_gate_up {
        slots[group] = Some(projections);
    }
    let mut projected: Vec<(Vec<f32>, Vec<f32>)> = slots
        .into_iter()
        .map(|p| p.expect("every group took one path or the other"))
        .collect();

    // The per-expert output scalars, on the projections' outputs and before
    // the activation — see `ExpertProjection::scale` for why they are not
    // folded into the rows. Applied identically to both paths, so the device
    // and host halves of one layer stay comparable.
    projected
        .par_iter_mut()
        .zip(experts.par_iter())
        .for_each(|(projected, &expert)| {
            if let Some(scale) = gate.and_then(|g| g.scale) {
                let s = scale[expert];
                projected.0.iter_mut().for_each(|v| *v *= s);
            }
            if let Some(scale) = up.scale {
                let s = scale[expert];
                projected.1.iter_mut().for_each(|v| *v *= s);
            }
        });

    // Activation, per member, into one contiguous run per group — the down
    // projection's operand.
    let ffn_dim = gate.map_or(up.n_rows, |gate| gate.n_rows);
    // **Parallel across groups**, like every other stage here. These three
    // passes — the per-expert scales above, this activation, and the weighted
    // accumulation below — walk the same data volume the projections do, and
    // ran single-threaded. That is the whole reason this helper cost 280 ms
    // more per `pp` 512 prefill than the per-expert path it replaces, which
    // does its activation *inside* the per-expert parallel task and so never
    // had a serial pass at all. The activation is the expensive one: a GELU
    // and a multiply for every member of every group, ~3M elements per layer.
    let hs: Vec<Vec<f32>> = (0..experts.len())
        .into_par_iter()
        .map(|group| {
            let (gate_out, up_out) = &projected[group];
            let mut h = Vec::with_capacity(members[group].len() * ffn_dim);
            for m in 0..members[group].len() {
                let range = m * ffn_dim..(m + 1) * ffn_dim;
                let gate_slice = if gate.is_some() {
                    &gate_out[range.clone()]
                } else {
                    &[][..]
                };
                h.extend_from_slice(&activate(gate_slice, &up_out[range]));
            }
            h
        })
        .collect();

    let down_views: Vec<_> = experts
        .iter()
        .map(|&e| down.exps.expert_matrix(e).rows(down.first_row, down.n_rows))
        .collect();
    let mut down_ops: Vec<MatmulOp<'_>> = Vec::new();
    let mut down_group: Vec<usize> = Vec::new();
    for group in 0..experts.len() {
        if !on_device[group] {
            continue;
        }
        down_ops.push(MatmulOp {
            x: &hs[group],
            n_tokens: members[group].len(),
            w: &down_views[group],
        });
        down_group.push(group);
    }
    // Overlapped for the same reason as the gate/up half above.
    let ffn_dim_in = down.exps.in_dim;
    let (down_batched, host_down) = rayon::join(
        || matmul_batch_mixed(backend, &down_ops),
        || -> Vec<(usize, Vec<f32>)> {
            (0..experts.len())
                .filter(|&group| !on_device[group])
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|group| {
                    let inputs: Vec<&[f32]> = (0..members[group].len())
                        .map(|m| &hs[group][m * ffn_dim_in..(m + 1) * ffn_dim_in])
                        .collect();
                    (
                        group,
                        project_expert(
                            backend,
                            down.exps,
                            experts[group],
                            down.first_row,
                            down.n_rows,
                            &inputs,
                        )
                        .concat(),
                    )
                })
                .collect()
        },
    );
    let mut down_out: Vec<Option<Vec<f32>>> = (0..experts.len()).map(|_| None).collect();
    for (slot, group) in down_group.iter().enumerate() {
        down_out[*group] = Some(down_batched[slot].clone());
    }
    for (group, projection) in host_down {
        down_out[group] = Some(projection);
    }
    let down_out: Vec<Vec<f32>> = down_out
        .into_iter()
        .map(|d| d.expect("every group took one path or the other"))
        .collect();

    let out_dim = down.n_rows;
    let mut out: Vec<Vec<Vec<f32>>> = selection
        .iter()
        .map(|picks| vec![Vec::new(); picks.len()])
        .collect();
    // Parallel too, and this is the largest of the three passes by data
    // touched: one `n_embd`-wide copy-and-scale per *member*, where the other
    // two are `n_ff`-wide. Scaling happens in the parallel half and only the
    // placement is serial, because `out[token][rank]` is a scattered write
    // that rayon cannot be shown to be disjoint — it is (every `(token,
    // rank)` is written exactly once, by construction of `ranks`), but moving
    // an already-built `Vec` into place costs nothing next to building it.
    let scaled: Vec<Vec<(usize, usize, Vec<f32>)>> = (0..experts.len())
        .into_par_iter()
        .map(|group| {
            // The down projection's per-expert scalar and the routing weight
            // are both scalars on the same vector, so they multiply out once
            // rather than in two passes.
            let expert_scale = down.scale.map_or(1.0, |s| s[experts[group]]);
            members[group]
                .iter()
                .enumerate()
                .map(|(m, &(token, weight))| {
                    let mut contribution = down_out[group][m * out_dim..(m + 1) * out_dim].to_vec();
                    let scale = expert_scale * weight;
                    contribution.iter_mut().for_each(|v| *v *= scale);
                    (token, ranks[group][m], contribution)
                })
                .collect()
        })
        .collect();
    for group in scaled {
        for (token, rank, contribution) in group {
            out[token][rank] = contribution;
        }
    }
    out
}

/// The plain SwiGLU FFN — `LLM_FFN_SILU`/`LLM_FFN_PAR`: `down(silu(gate(x))
/// * up(x))`.
///
/// One computation, shared by every dense model in this engine that has it:
/// `llama` (which serves Qwen2 and Qwen3 as well as Llama and Mistral) and
/// the dense FFN of the Qwen 3.5 hybrid trunk (`engine::arch::qwen_hybrid::
/// DenseFfn`). `gate` and `up` are independent projections of the same input,
/// so they go out as one batched dispatch rather than two sequential
/// round-trips (see `Backend::matmul_batch`).
pub(crate) fn swiglu_ffn(
    backend: &dyn crate::engine::backend::Backend,
    normed: &[f32],
    n_tokens: usize,
    gate_w: &crate::engine::loader::QuantMatrix,
    up_w: &crate::engine::loader::QuantMatrix,
    down_w: &crate::engine::loader::QuantMatrix,
) -> Vec<f32> {
    let mut out = Vec::new();
    let mut scratch = FfnScratch::default();
    swiglu_ffn_into(
        backend,
        &mut out,
        &mut scratch,
        normed,
        n_tokens,
        gate_w,
        up_w,
        down_w,
    );
    out
}

/// The intermediates [`swiglu_ffn_into`] needs, owned by the caller so they
/// are allocated once per forward pass rather than once per layer.
///
/// These are the **largest** buffers in a transformer layer: the gate and up
/// projections are `n_tokens * n_ff`, where `n_ff` is several times `n_embd`.
/// See `Backend::matmul_into` for what reusing them does and does not buy —
/// allocator bookkeeping, not `mmap` traffic, at the default prefill-batch
/// and stripe caps.
#[derive(Default)]
pub(crate) struct FfnScratch {
    /// The gate and up projections, in that order — exactly the two ops
    /// `matmul_batch_into` is handed, and reused in place.
    gate_up: Vec<Vec<f32>>,
}

/// [`swiglu_ffn`] writing into caller-owned buffers.
///
/// Identical arithmetic — the same batched gate/up dispatch, the same
/// in-place SiLU, the same multiply, the same down projection — so it is
/// bit-identical by construction rather than by convention. What changes is
/// only where the three intermediates live.
#[allow(clippy::too_many_arguments)]
pub(crate) fn swiglu_ffn_into(
    backend: &dyn crate::engine::backend::Backend,
    out: &mut Vec<f32>,
    scratch: &mut FfnScratch,
    normed: &[f32],
    n_tokens: usize,
    gate_w: &crate::engine::loader::QuantMatrix,
    up_w: &crate::engine::loader::QuantMatrix,
    down_w: &crate::engine::loader::QuantMatrix,
) {
    use crate::engine::backend::MatmulOp;
    backend.matmul_batch_into(
        &mut scratch.gate_up,
        &[
            MatmulOp {
                x: normed,
                n_tokens,
                w: gate_w,
            },
            MatmulOp {
                x: normed,
                n_tokens,
                w: up_w,
            },
        ],
    );
    // `split_at_mut` rather than the old pair of `pop`s: the buffers stay in
    // `scratch` so the next layer inherits their capacity.
    let (gate, up) = scratch.gate_up.split_at_mut(1);
    let gate = &mut gate[0];
    for g in gate.iter_mut() {
        *g = tensor::silu(*g);
    }
    tensor::mul_inplace(gate, &up[0]);
    backend.matmul_into(out, gate, n_tokens, down_w);
}

/// One MoE layer's weights, for [`swiglu_moe_ffn`].
///
/// Borrowed rather than owned so each architecture keeps its own layer
/// struct — what they share is the computation, not the layout.
pub(crate) struct SwigluMoe<'a> {
    /// `[n_embd, n_expert]` — the router.
    pub gate_inp: &'a crate::engine::loader::QuantMatrix,
    /// `exp_probs_b`, DeepSeek-V3's selection bias, when the file has one.
    pub exp_probs_b: Option<&'a [f32]>,
    pub gate_exps: &'a crate::engine::loader::ExpertQuantMatrix,
    pub up_exps: &'a crate::engine::loader::ExpertQuantMatrix,
    pub down_exps: &'a crate::engine::loader::ExpertQuantMatrix,
    /// The always-on shared expert, which reads the same input as the
    /// router and adds its whole output in. `None` for a model with none.
    pub shared: Option<SwigluSharedExpert<'a>>,
    /// `<arch>.swiglu_clamp_exp` for this layer — the routed experts'
    /// SwiGLU limit. `0` (the usual case) leaves the activation alone.
    pub clamp_exp: f32,
    /// `<arch>.swiglu_clamp_shexp` for this layer — the same for the
    /// shared expert.
    pub clamp_shexp: f32,
}

/// SwiGLU with upstream's optional limit: the up branch is clamped to
/// `[-limit, limit]` and the *activated* gate branch from above only, so
/// the two halves are clamped differently and swapping them is silent. A
/// `limit` at or below `1e-6` — which is what a file that declares no
/// limit writes — means no clamp at all.
fn swiglu_limited(gate: &[f32], up: &[f32], limit: f32) -> Vec<f32> {
    let mut h: Vec<f32> = gate.iter().map(|&g| tensor::silu(g)).collect();
    if limit > 1e-6 {
        for (h, &u) in h.iter_mut().zip(up.iter()) {
            *h = h.min(limit) * u.clamp(-limit, limit);
        }
    } else {
        tensor::mul_inplace(&mut h, up);
    }
    h
}

/// Batch width at which a MoE layer's two FFN branches are evaluated
/// concurrently rather than one after the other, from
/// `ORANGU_MOE_OVERLAP_MIN_TOKENS`.
///
/// The branches use different processors — the shared MLP is device
/// matmuls, the routed branch is host `vecdot` — and running them in
/// sequence is why a MoE forward leaves both around 60% idle. Overlapping
/// them costs one worker blocked in `device.poll`, which is why it is a
/// width threshold and not a plain `true`.
///
/// **`1` — overlap at every width, including a single token.** An earlier
/// version defaulted to 24 on the reading that a decode step's narrow
/// fan-out could not spare a worker to a blocking `device.poll`. That
/// reading came from comparing two runs taken an hour apart, and it was
/// wrong: a *controlled* A/B through this variable, alternating the two
/// settings against one another and repeated, gives **6.19 against 6.72
/// tok/s — the overlap is +8.6% at decode** and reproduced to three
/// significant figures on the repeat. Prefill is +9.2% at 1024 tokens. The
/// two widths agree, so there is no crossover to place.
///
/// The threshold survives as the escape hatch and as the control for that
/// A/B: a large value restores the sequential form. Throughput on this
/// machine drifts several percent over a long session, which is what
/// produced the wrong reading — compare settings with `orangu-bench
/// --sweep`, never across sessions.
///
/// Lives here rather than in one architecture because the shape it exploits
/// — a device-side shared branch summed with a host-side routed branch — is
/// what *every* mixture-of-experts module in this directory builds.
pub(crate) fn moe_overlap_min_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ORANGU_MOE_OVERLAP_MIN_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1)
    })
}

/// Whether `ORANGU_MOE_ROUTER_PER_TOKEN=1` asked for the router to run one
/// matmul per token — the control arm for [`moe_router_logits`], and the
/// form every module here except `gemma` used to carry unconditionally.
fn router_per_token() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_MOE_ROUTER_PER_TOKEN"))
}

/// Every token's router scores for one MoE layer — `[n_tokens, n_expert]`,
/// row-major — from **one** matmul over the whole batch.
///
/// `x` is `[n_tokens, gate_inp.in_dim]`, whatever normalization the
/// architecture applies to the layer input having already been done.
///
/// The per-token form this replaces submitted a `[1, n_embd] × [n_embd,
/// n_expert]` GEMV per token per MoE layer: at `pp 1024` that is a thousand
/// device round-trips a layer, each on the shape a GPU is worst at, to
/// produce a `[n_expert]` row that decides only *which experts run*. One
/// GEMM produces the same rows.
///
/// Not bit-identical to the per-token form at `n_tokens > 1`, for the reason
/// [`crate::engine::backend::Backend::matmul_decode`]'s doc comment gives —
/// the multi-token kernels sum in a different order and quantize activations
/// at a coarser granularity. That is the same trade every other prefill
/// matmul in the engine already makes, and it is confined to prefill:
/// a decode step is `n_tokens == 1` and takes the identical single-row path.
pub(crate) fn moe_router_logits(
    backend: &dyn crate::engine::backend::Backend,
    x: &[f32],
    n_tokens: usize,
    gate_inp: &crate::engine::loader::QuantMatrix,
) -> Vec<f32> {
    if n_tokens > 1 && !router_per_token() {
        return backend.matmul(x, n_tokens, gate_inp);
    }
    let in_dim = gate_inp.in_dim;
    (0..n_tokens)
        .flat_map(|t| backend.matmul(&x[t * in_dim..(t + 1) * in_dim], 1, gate_inp))
        .collect()
}

/// Which experts each token selected, and with what weight, from the
/// `[n_tokens, n_expert]` block [`moe_router_logits`] returns.
///
/// Split from the matmul so the two can sit on opposite sides of a
/// `rayon::join`: the routing decides which experts run, so it is on the
/// routed branch's critical path either way, but the *matmul* in front of it
/// is a device submission that must not queue behind the shared branch's.
pub(crate) fn route_batch(
    routing: &ExpertRouting,
    logits: &[f32],
    n_tokens: usize,
    exp_probs_b: Option<&[f32]>,
) -> Vec<Vec<(usize, f32)>> {
    let n_expert = logits.len() / n_tokens.max(1);
    (0..n_tokens)
        .map(|t| {
            let row = &logits[t * n_expert..(t + 1) * n_expert];
            let (selected, weights) = routing.route(row, exp_probs_b, None);
            selected.into_iter().zip(weights).collect()
        })
        .collect()
}

/// The shared expert half of [`SwigluMoe`] — an ordinary SwiGLU FFN.
pub(crate) struct SwigluSharedExpert<'a> {
    pub gate: &'a crate::engine::loader::QuantMatrix,
    pub up: &'a crate::engine::loader::QuantMatrix,
    pub down: &'a crate::engine::loader::QuantMatrix,
}

/// The routed + shared-expert SwiGLU MoE FFN — upstream's `build_moe_ffn`
/// with `LLM_FFN_SILU`, plus the shared expert its callers add alongside.
///
/// One computation, shared by every architecture here whose experts run at
/// full width on the layer input: `engine::arch::glm` and
/// `engine::arch::bailingmoe`. (`kimi3`'s experts run in a *latent* space
/// and its activation is `situ`, not SwiGLU, so it keeps its own; the
/// Qwen 3.5 hybrid trunk's `MoeFfn` differs in its shared-expert gate.)
///
/// The routing decision — which experts, with what weight, under whatever
/// grouping the file declares — is entirely [`ExpertRouting::route`]'s, so
/// an architecture that only differs there differs nowhere here.
pub(crate) fn swiglu_moe_ffn(
    backend: &dyn crate::engine::backend::Backend,
    routing: &ExpertRouting,
    normed: &[f32],
    n_tokens: usize,
    n_embd: usize,
    moe: &SwigluMoe<'_>,
) -> Vec<f32> {
    let mut out = vec![0f32; n_tokens * n_embd];
    let mut experts = crate::engine::moe_stats::LayerRecorder::for_tensors(&[
        moe.gate_exps,
        moe.up_exps,
        moe.down_exps,
    ]);
    // Route the whole batch first, so the union can be taken before any
    // expert's weights are read — see `evaluate_routed_experts` — and so the
    // router's own matmul is out of the way before the two branches below
    // overlap: it is a device submission at the head of an otherwise
    // host-side branch, and leaving it inside the parallel region puts it in
    // a queue behind the shared MLP's submissions on the same device.
    let logits = moe_router_logits(backend, normed, n_tokens, moe.gate_inp);
    let mut selection = route_batch(routing, &logits, n_tokens, moe.exp_probs_b);
    // Trim to the expert budget *before* anything is recorded or read: the
    // counters should describe the work actually done, and a dropped
    // expert's weights must never be fetched.
    apply_expert_budget(&mut selection, moe.gate_exps);
    for picks in &selection {
        picks.iter().for_each(|&(e, _)| experts.select(e));
    }

    // The GPU expert path batches the three projections across experts —
    // see `evaluate_routed_experts_batched`.
    let routed_branch = || {
        if gpu_experts() && backend.as_wgpu().is_some() {
            evaluate_routed_experts_batched(
                backend,
                &selection,
                normed,
                n_embd,
                moe.gate_exps,
                moe.up_exps,
                moe.down_exps,
                |gate, up| swiglu_limited(gate, up, moe.clamp_exp),
            )
        } else {
            evaluate_routed_experts(&selection, |expert, members| {
                let inputs: Vec<&[f32]> = members
                    .iter()
                    .map(|&(t, _)| &normed[t * n_embd..(t + 1) * n_embd])
                    .collect();
                let gate = project_expert(
                    backend,
                    moe.gate_exps,
                    expert,
                    0,
                    moe.gate_exps.out_dim,
                    &inputs,
                );
                let up = project_expert(
                    backend,
                    moe.up_exps,
                    expert,
                    0,
                    moe.up_exps.out_dim,
                    &inputs,
                );
                let hidden: Vec<Vec<f32>> = gate
                    .into_iter()
                    .zip(up)
                    .map(|(gate, up)| swiglu_limited(&gate, &up, moe.clamp_exp))
                    .collect();
                let hidden_refs: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                project_expert(
                    backend,
                    moe.down_exps,
                    expert,
                    0,
                    moe.down_exps.out_dim,
                    &hidden_refs,
                )
                .into_iter()
                .zip(members)
                .map(|(mut contribution, &(_, weight))| {
                    contribution.iter_mut().for_each(|v| *v *= weight);
                    contribution
                })
                .collect()
            })
        }
    };

    // The shared expert runs for every token at once. Its matrices are
    // exempt from the backend's up-front type check (see
    // `matmul_host_fallback`), so they must not go straight to the device.
    let shared_branch = || {
        moe.shared.as_ref().map(|shared| {
            let gate = matmul_host_fallback(backend, normed, n_tokens, shared.gate);
            let up = matmul_host_fallback(backend, normed, n_tokens, shared.up);
            let h = swiglu_limited(&gate, &up, moe.clamp_shexp);
            matmul_host_fallback(backend, &h, n_tokens, shared.down)
        })
    };

    // Nothing in one branch depends on the other — they read the same
    // `normed` and are summed below — and they use *different processors*:
    // the shared MLP is three device matmuls, the routed branch is host
    // `vecdot` over a `par_iter`. Run in sequence each waits out the other,
    // which is why a MoE forward leaves the GPU engine and the CPU both
    // around 60% idle. See [`moe_overlap_min_tokens`] for the measurement
    // and for the control that restores the sequential form.
    //
    // `rayon::join` rather than a thread: the routed branch is itself a
    // `par_iter` over experts, so it has to run on the pool that owns those
    // workers, and the shared branch's blocking device polls then occupy one
    // worker while the rest keep evaluating experts. `join` returns a pair,
    // not a race, and each branch is internally deterministic, so the sum
    // below does not depend on which finishes first.
    let (shared_out, contribs) = if moe.shared.is_some() && n_tokens >= moe_overlap_min_tokens() {
        rayon::join(shared_branch, routed_branch)
    } else {
        (shared_branch(), routed_branch())
    };
    if let Some(shared_out) = shared_out {
        out = shared_out;
    }
    experts.loaded_once_per_distinct_expert();

    for t in 0..n_tokens {
        let dst = &mut out[t * n_embd..(t + 1) * n_embd];
        for contrib in &contribs[t] {
            tensor::add_inplace(dst, contrib);
        }
    }
    experts.commit(n_tokens);
    out
}

/// `Backend::matmul`, falling back to the host when the selected backend has
/// no kernel for this tensor's quantization.
///
/// Needed for exactly one class of tensor. `engine::backend::
/// unsupported_tensor_types` rejects a GPU backend outright if the model
/// carries a type it lacks — so every ordinary weight is guaranteed, by the
/// time a forward pass runs, to be one the device can take, and a plain
/// `backend.matmul` on one cannot meet a gap. The exception is the set
/// `engine::backend::is_cpu_only_tensor` names: the expert stacks and the
/// **shared-expert** matrices are exempt from that check on purpose, so that
/// a low-bit mixture-of-experts file still gets a GPU for the rest of the
/// model rather than being pushed onto the host wholesale.
///
/// That exemption is what makes this function necessary: a shared expert can
/// be a type the device has no shader for, and `VulkanBackend` *panics* on
/// that rather than returning zeros. So every shared-expert matmul has to go
/// through here. `qwen_hybrid::MoeFfn` — which is what `qwen35moe` and
/// `qwen3next` both run — carries its own copy of this check for the same
/// reason, spelled out inline because it also decides which backend the
/// `down` projection goes to.
pub(crate) fn matmul_host_fallback(
    backend: &dyn crate::engine::backend::Backend,
    x: &[f32],
    n_tokens: usize,
    w: &crate::engine::loader::QuantMatrix,
) -> Vec<f32> {
    let mut out = Vec::new();
    matmul_host_fallback_into(&mut out, backend, x, n_tokens, w);
    out
}

/// [`matmul_host_fallback`] into a caller-owned buffer — see
/// `Backend::matmul_into`. The allocating form is a wrapper over this, so
/// the routing decision lives in exactly one place.
pub(crate) fn matmul_host_fallback_into(
    out: &mut Vec<f32>,
    backend: &dyn crate::engine::backend::Backend,
    x: &[f32],
    n_tokens: usize,
    w: &crate::engine::loader::QuantMatrix,
) {
    use crate::engine::backend::Backend as _;
    if backend.supports_type(w.ggml_type()) {
        backend.matmul_into(out, x, n_tokens, w);
    } else {
        crate::engine::backend::CpuBackend.matmul_into(out, x, n_tokens, w);
    }
}

/// Whether `ORANGU_GPU_EXPERTS=1` asked for routed-expert matmuls to go to
/// the GPU instead of the host AVX2/rayon path.
///
/// **Off by default, and a measurement knob before it is a feature.** The
/// open question a device expert tier rests on is whether a GPU expert
/// matmul beats `engine::vecdot`'s tuned host path *at all* — colibri, the
/// only engine either reference tree has that ships such a tier, concludes
/// it "earns its VRAM only when the CPU is the weak link". This routes
/// every routed expert to the device so that question can be answered
/// before a residency policy, a heat profile and a batched dispatch are
/// built on top of the assumption that it can.
///
/// It deliberately does **no** residency management: every expert it
/// touches lands in the backend's weight arena, which never evicts. On a
/// model whose experts exceed VRAM that is driver paging, and the number it
/// produces is meaningless. Point it at a device that can hold them.
/// Whether `ORANGU_EXPERT_STREAM=1` asked for routed-expert weights to be
/// **streamed** into a bounded device region per call, instead of admitting a
/// fixed subset of them to the permanent arena up front.
///
/// Separate from [`gpu_experts`] on purpose: that knob decides whether expert
/// matmuls go to the device at all, this one decides how their weights get
/// there. Both are off by default, and this one implies nothing on its own —
/// it only widens what `gpu_experts` can reach, from the ~12% of a large
/// model's experts that fit a card to all of them.
pub(crate) fn expert_streaming() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag("ORANGU_EXPERT_STREAM"))
}

pub(crate) fn gpu_experts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag("ORANGU_GPU_EXPERTS"))
}

/// An on/off knob that reads `0` as **off** rather than as "set, therefore
/// on".
///
/// Presence alone is the usual convention here and is right for a knob one
/// exports by hand. It is wrong for one that gets **swept**: `--sweep
/// VAR=0,1` sets the variable at every point, so a presence test reports the
/// feature on for the control arm too and the A/B silently compares the
/// feature against itself — which is not visible in the result, only in the
/// two arms agreeing suspiciously well.
fn env_flag(name: &str) -> bool {
    flag_is_on(std::env::var(name).ok().as_deref())
}

/// [`env_flag`]'s decision, separated from the environment so it can be
/// tested without one test's `set_var` reaching another's thread.
fn flag_is_on(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "no" | "off" | "false"
        )
    })
}

/// Whether one routed expert may be dispatched to the device.
///
/// All four terms are necessary, and `resident` is the one that is easy to
/// leave out — it is also the only one that fails *silently and expensively*
/// when it is missing. `VulkanBackend::weight_buffer`'s arena never evicts,
/// so an expert that reaches the device stays there for the model's life;
/// `main::plan_expert_tier` therefore chooses a bounded resident set up
/// front, and this flag is that decision. Dispatching without consulting it
/// does not merely ignore the plan, it *inverts* it: every expert the
/// routing touches is admitted, the arena grows past the budget the tier was
/// given, and the driver starts paging device memory on a path whose whole
/// purpose was to avoid host traffic. `evaluate_routed_experts_batched`
/// consults it for the architectures that take the batched path; this is the
/// same check for the ones that do not.
///
/// Split out from [`gpu_project_expert`] so the policy can be pinned by a
/// test on a machine with no GPU, where the call site's other terms are
/// unreachable.
fn device_expert_admissible(enabled: bool, has_gpu: bool, kernel: bool, resident: bool) -> bool {
    enabled && has_gpu && kernel && resident
}

/// One expert's projection on the GPU, or `None` to take the host path.
///
/// The whole of the dispatch: an expert's rows are contiguous in the
/// stacked tensor, so `ExpertQuantMatrix::expert_matrix` views them as an
/// ordinary `QuantMatrix` and `Backend::matmul` already knows what to do
/// with one — no kernel is written for this path, it reuses the matmul
/// kernel the backend already compiled for that quantization.
///
/// Returns `None` — the host path — when the knob is off, the backend has
/// no GPU, the backend has no kernel for this quantization, or the expert
/// is not one the device tier holds. The residency term is what keeps the
/// tier a tier.
///
/// **The kernel term is not the reason the tier does not pay.** On Vulkan it
/// now excludes three quantizations, none of them ggml's own ids, and the
/// whole `IQ` range that large mixture-of-experts files are shipped in is
/// *not* among them — `Backend::supports_type`'s own doc names the excluded
/// set per backend and points at the test that pins it. So a routed
/// expert in an ordinary low-bit file reaches this path; what it then meets is
/// the dispatch shape `VulkanBackend::matmul_batch` describes, which is where
/// the measured loss recorded in `engine::expert_tier` comes from.
///
/// This paragraph used to say the opposite — that the `IQ*` types the Vulkan
/// backend lacks are exactly the ones large MoE models ship in — while the
/// paragraph above it simultaneously claimed every backend had a kernel for
/// every expert quantization. Both were wrong, they contradicted each other,
/// and the code between them went on compiling either way, because
/// `supports_type` is a lookup in a `const` array and no test compared that
/// array to anything. It stayed wrong long enough that an outside reader of
/// this source scoped a critical task around building kernels that already
/// existed. **A doc comment is part of the source, and this one was lying —
/// twice, in opposite directions.**
fn gpu_project_expert(
    backend: &dyn crate::engine::backend::Backend,
    weights: &crate::engine::loader::ExpertQuantMatrix,
    expert: usize,
    first_row: usize,
    n_rows: usize,
    inputs: &[&[f32]],
) -> Option<Vec<Vec<f32>>> {
    if !device_expert_admissible(
        gpu_experts(),
        backend.as_wgpu().is_some(),
        backend.supports_type(weights.ggml_type()),
        weights.is_device_resident(expert),
    ) {
        return None;
    }
    let in_dim = weights.in_dim;
    let mut x = Vec::with_capacity(inputs.len() * in_dim);
    for input in inputs {
        debug_assert_eq!(input.len(), in_dim);
        x.extend_from_slice(input);
    }
    let view = weights.expert_matrix(expert).rows(first_row, n_rows);
    // `matmul`, not `matmul_decode`: this is the same shape a prefill
    // matmul has — several independent input rows against one weight — and
    // the decode entry point exists for a *batch of sequences*, which these
    // rows are not.
    let y = backend.matmul(&x, inputs.len(), &view);
    Some(
        (0..inputs.len())
            .map(|t| y[t * n_rows..(t + 1) * n_rows].to_vec())
            .collect(),
    )
}

/// One expert matrix applied to every token routed to that expert,
/// dequantizing each row **exactly once** and never holding more than a few
/// rows at a time.
///
/// Returns `[input][n_rows]` — one output vector per input, in the order the
/// inputs were given.
///
/// The obvious way to reuse a row across several tokens is to dequantize the
/// whole expert first and then loop the tokens over it. That costs
/// `n_rows * in_dim` floats held live per expert — on GLM-5.2's dimensions,
/// **94 MB for one expert's three matrices**, times however many experts rayon
/// is evaluating at once. At decode, where an expert has exactly one token, all
/// of it is materialized to serve a single dot product per row. Inverting the
/// loops gets the same reuse for a row at a time: dequantize row `o`, dot it
/// against every input while it is still in cache, drop it.
///
/// Parallel over *rows*, which is where the width is — an expert has thousands
/// of rows and (at decode) one input. Each output element is an independent dot
/// product, so no accumulation crosses a task boundary and the result does not
/// depend on how rayon splits the range.
///
/// `first_row`/`n_rows` name a slice of the matrix's rows, for the
/// architectures whose gate and up projections share one fused tensor
/// (`gemma-4-26B-A4B`'s `ffn_gate_up_exps`, whose first half is the gate and
/// second half the up).
pub(crate) fn project_expert(
    backend: &dyn crate::engine::backend::Backend,
    weights: &crate::engine::loader::ExpertQuantMatrix,
    expert: usize,
    first_row: usize,
    n_rows: usize,
    inputs: &[&[f32]],
) -> Vec<Vec<f32>> {
    let n_inputs = inputs.len();
    if n_inputs == 0 {
        return Vec::new();
    }
    if let Some(out) = gpu_project_expert(backend, weights, expert, first_row, n_rows, inputs) {
        return out;
    }
    // Row-major in the *row* index: one contiguous run of `n_inputs` outputs
    // per row, so a task owns a disjoint slice and writes it without
    // synchronization. Transposed to per-input vectors on the way out, since
    // that is what every caller's activation math wants.
    // Claimed for the whole projection, not per row: the lease is what a
    // residency policy holds still while these bytes are read, and a policy
    // that could evict between row 3 and row 4 would be no policy at all.
    let lease = crate::engine::expert_store::global().acquire(weights, expert);
    // The tier's copy when it has one, the mapping otherwise. Identical bytes
    // either way — the tier holds a copy of exactly these — so which side
    // serves a row cannot change a single output value.
    let raw = lease.bytes().unwrap_or_else(|| weights.expert_span(expert));

    let ggml_type = weights.ggml_type();
    let in_dim = weights.in_dim;
    let row_bytes = weights.row_bytes();
    let row = |index: usize| {
        let offset = (first_row + index) * row_bytes;
        &raw[offset..offset + row_bytes]
    };

    let mut by_row = vec![0f32; n_rows * n_inputs];
    if vecdot::supports(ggml_type, in_dim) {
        // The integer-dot kernels `engine::backend::cpu` already uses for
        // every dense matmul, applied to the one matmul that was still
        // dequantizing to `f32` first.
        //
        // That mattered more here than anywhere else: a routed expert's rows
        // are read once and thrown away, so the `f32` row was pure overhead —
        // materialized, dotted once per token, discarded. On a
        // `nemotron_h_moe` decode profile `quant::dequantize_into` under this
        // function was **60% of all CPU time**, and it is the same shape for
        // every mixture-of-experts family here.
        //
        // The activation quantization this introduces is the same one the
        // dense path has always used, so an expert's arithmetic is now
        // consistent with the rest of the layer rather than more precise than
        // it.
        if n_inputs == 1 {
            // GEMV: the fused per-row kernels, which never spill an unpacked
            // row to memory. `engine::backend::cpu` documents why that beats
            // the unpack-once form when there is only one activation to
            // amortize it over.
            if vecdot::supports_k_row(ggml_type, in_dim) {
                let act = vecdot::quantize_act_k_row(inputs[0]);
                by_row
                    .par_chunks_mut(1)
                    .enumerate()
                    .for_each(|(index, out)| {
                        out[0] = vecdot::dot_k_row(ggml_type, row(index), &act);
                    });
            } else {
                let act = vecdot::quantize_act(inputs[0]);
                by_row
                    .par_chunks_mut(1)
                    .enumerate()
                    .for_each(|(index, out)| {
                        out[0] = vecdot::dot_row(ggml_type, row(index), &act);
                    });
            }
        } else {
            // GEMM: unpack each row once, then dot it against every token
            // routed to this expert — the same amortization the dequantizing
            // path had, minus the `f32` materialization.
            let acts: Vec<vecdot::ActQ8> = inputs.iter().map(|x| vecdot::quantize_act(x)).collect();
            by_row.par_chunks_mut(n_inputs).enumerate().for_each_init(
                vecdot::UnpackedRow::new,
                |unpacked, (index, out)| {
                    vecdot::unpack_row(ggml_type, row(index), in_dim, unpacked);
                    vecdot::dot_unpacked_multi(unpacked, &acts, out);
                },
            );
        }
    } else {
        // Types the integer-dot kernels have no unpacking for — the `IQ`
        // family, which is exactly what the largest mixture-of-experts models
        // ship in. Dequantize a row and dot it in `f32`, as this always did.
        by_row
            .par_chunks_mut(n_inputs)
            .enumerate()
            // One dequantization buffer per rayon job rather than per row: the
            // rows are all `in_dim` wide, so the buffer is filled and refilled
            // without ever reallocating. `for_each_init` runs the initializer
            // once per work split, not once per item, which is exactly the reuse
            // scope wanted here.
            .for_each_init(Vec::new, |weights_row, (index, out)| {
                weights.row_from(raw, first_row + index, weights_row);
                for (slot, input) in out.iter_mut().zip(inputs) {
                    *slot = tensor::dot(input, weights_row);
                }
            });
    }
    (0..n_inputs)
        .map(|i| (0..n_rows).map(|row| by_row[row * n_inputs + i]).collect())
        .collect()
}

/// The per-layer cap on distinct experts, from `ORANGU_EXPERT_BUDGET`. `0`
/// (the default) disables the whole mechanism.
///
/// **This is the only thing in this document that changes what the model
/// computes.** Everything else was held to bit-identity; this trades a bounded
/// quality cost for bytes not moved, and so is off unless asked for.
pub(crate) fn expert_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("ORANGU_EXPERT_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// The largest batch the budget may touch.
///
/// **Rule one, and it is not a tuning parameter.** During prefill the batch's
/// union is 30–100+ experts and a cap of 4–8 drops 80–90% of them — which does
/// not degrade the answer, it corrupts the hidden state, writes that into the
/// KV cache, and produces repetitive garbage for the rest of the generation
/// (colibri's #292). The budget is only ever safe token-by-token, once a
/// correct prefill cache already exists. `4` rather than `1` so a speculative
/// verify batch still qualifies.
const BUDGET_MAX_BATCH: usize = 4;

/// Trims each position's routing to fit [`expert_budget`] distinct experts
/// across the batch, dropping only what the store does not already hold and
/// keeping the highest aggregate gate weight.
///
/// Ported from `colibri.c`'s FASE B, including both of the rules its issue
/// tracker paid for — see [`BUDGET_MAX_BATCH`] for the first. The second is
/// below: **no position may be left with nothing routed.**
///
/// `weights_scale_preserved` renormalizes the survivors so dropping experts
/// does not silently shrink the routed branch's magnitude. One rule covers
/// both of colibri's two branches: scaling by `old_sum / new_sum` is invariant
/// to whatever scale the weights already carry, so it reproduces
/// "divide and re-apply `routed_scale`" for normalized routing and
/// "rescale by old/new" for unnormalized routing without having to know which
/// it is looking at.
pub(crate) fn apply_expert_budget(
    selection: &mut [Vec<(usize, f32)>],
    weights: &crate::engine::loader::ExpertQuantMatrix,
) {
    trim_to_expert_budget(selection, weights, expert_budget());
}

/// [`apply_expert_budget`] with the budget passed in rather than read from the
/// environment.
///
/// Split out because `expert_budget()` caches in a `OnceLock`: a test binary
/// can only ever observe one value, so tests written against the env-reading
/// form silently stop exercising anything the moment the cached budget is
/// wider than their fixture. Four of the five tests below did exactly that
/// before this split, passing while trimming nothing.
pub(crate) fn trim_to_expert_budget(
    selection: &mut [Vec<(usize, f32)>],
    weights: &crate::engine::loader::ExpertQuantMatrix,
    budget: usize,
) {
    if budget == 0 || selection.len() > BUDGET_MAX_BATCH {
        return;
    }
    let store = crate::engine::expert_store::global();

    // The batch's union, in first-seen order, with each expert's aggregate
    // gate weight across positions and whether it is already held.
    let mut union: Vec<usize> = Vec::new();
    let mut weight_of: HashMap<usize, f32> = HashMap::new();
    for picks in selection.iter() {
        for &(expert, weight) in picks {
            if !weight_of.contains_key(&expert) {
                union.push(expert);
            }
            *weight_of.entry(expert).or_insert(0.0) += weight;
        }
    }
    if union.len() <= budget {
        return;
    }

    // Hits are free — they cost no bytes — so they are never candidates for
    // dropping, and only the misses compete for what the budget has left.
    let mut keep: std::collections::HashSet<usize> = union
        .iter()
        .copied()
        .filter(|&e| store.is_resident(weights, e))
        .collect();
    let mut misses: Vec<usize> = union
        .iter()
        .copied()
        .filter(|e| !keep.contains(e))
        .collect();
    misses.sort_by(|a, b| {
        weight_of[b]
            .partial_cmp(&weight_of[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    for &expert in misses.iter().take(budget.saturating_sub(keep.len())) {
        keep.insert(expert);
    }

    // **Rule two.** With enough hits the miss budget reaches zero, and a
    // position whose entire top-k are misses is compacted to nothing. It would
    // then receive only the shared expert, and that wrong hidden state enters
    // the KV cache — the same failure as rule one, reached from the other
    // direction. Re-admit that position's best miss and count it.
    let mut rescued = 0u64;
    for picks in selection.iter() {
        if picks.is_empty() || picks.iter().any(|(e, _)| keep.contains(e)) {
            continue;
        }
        if let Some(&(best, _)) = picks
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            keep.insert(best);
            rescued += 1;
        }
    }

    let mut dropped = 0u64;
    for picks in selection.iter_mut() {
        let before: f32 = picks.iter().map(|(_, w)| w).sum();
        let removed = picks.len();
        picks.retain(|(expert, _)| keep.contains(expert));
        dropped += (removed - picks.len()) as u64;
        let after: f32 = picks.iter().map(|(_, w)| w).sum();
        // Guard the degenerate case: survivors whose weights sum to nothing
        // would scale to infinity.
        if picks.len() < removed && after > 1e-20 && before > after {
            let scale = before / after;
            for (_, weight) in picks.iter_mut() {
                *weight *= scale;
            }
        }
    }
    crate::engine::moe_stats::record_budget(dropped, rescued);
}

/// Evaluates a MoE layer's routed experts **once per distinct expert**
/// instead of once per `(token, expert)` pair.
///
/// Every architecture here used to walk positions and, for each, evaluate its
/// selected experts — so an expert that three of a batch's tokens routed to
/// had its weights read, dequantized and thrown away three times. The
/// redundancy is the batch's own: at top-8 over 128 experts a 512-token
/// prefill selects ~4000 times from a union that saturates at 128, and every
/// duplicate is a full re-read of that expert's gate, up and down matrices.
/// Grouping by expert reads each one once and applies it to every token that
/// selected it.
///
/// **The output is bit-identical, and the structure here is what makes it so.**
/// Floating-point addition does not commute, so the contributions must reach
/// each token's accumulator in the same order as before — the order the
/// *router* picked them, which is not the order the experts are now evaluated
/// in. Hence the return shape: `[token][selection rank]`, exactly the list
/// each architecture's existing summation loop already walks. The regrouping
/// is confined to how the vectors are *produced*; nothing about how they are
/// added changes, which is why the callers' accumulation code is untouched.
///
/// `eval(expert, members)` is given one expert and every `(token, weight)`
/// that selected it, and returns one contribution vector per member in the
/// same order. Architectures differ in the expert math itself — SwiGLU,
/// GEGLU, situ, clamped SwiGLU, fused or separate gate/up, per-expert QAT
/// scales — so that stays with each of them; only the grouping is shared.
///
/// Memory: the contributions are held until the caller sums them, which is
/// `n_tokens * n_expert_used * n_out` floats against the old code's one token's
/// worth. Bounded in practice by prefill already running in chunks
/// (`ORANGU_PREFILL_BATCH`), and the transient peak is freed as soon as the
/// caller's accumulation loop drops it.
pub(crate) fn evaluate_routed_experts<F>(
    selection: &[Vec<(usize, f32)>],
    eval: F,
) -> Vec<Vec<Vec<f32>>>
where
    F: Fn(usize, &[(usize, f32)]) -> Vec<Vec<f32>> + Sync,
{
    // Group by expert, remembering where each member came from. First-seen
    // order rather than sorted: it costs nothing, and it keeps the evaluation
    // order reproducible from the routing alone.
    let mut experts: Vec<usize> = Vec::new();
    let mut members: Vec<Vec<(usize, f32)>> = Vec::new();
    let mut ranks: Vec<Vec<usize>> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for (token, picks) in selection.iter().enumerate() {
        for (rank, &(expert, weight)) in picks.iter().enumerate() {
            let group = *group_of.entry(expert).or_insert_with(|| {
                experts.push(expert);
                members.push(Vec::new());
                ranks.push(Vec::new());
                experts.len() - 1
            });
            members[group].push((token, weight));
            ranks[group].push(rank);
        }
    }

    // One task per distinct expert. Where the old code's fan-out was over
    // `(token, expert)` pairs, this is over experts alone — narrower during
    // prefill, identical at decode (where every selection is already
    // distinct), and each task now carries every token that wants that
    // expert, so the work per task grew by exactly what the fan-out lost.
    let evaluated: Vec<Vec<Vec<f32>>> = experts
        .par_iter()
        .zip(members.par_iter())
        .map(|(&expert, members)| eval(expert, members))
        .collect();

    let mut out: Vec<Vec<Vec<f32>>> = selection
        .iter()
        .map(|picks| vec![Vec::new(); picks.len()])
        .collect();
    for (group, contributions) in evaluated.into_iter().enumerate() {
        debug_assert_eq!(
            contributions.len(),
            members[group].len(),
            "an expert returned a different number of contributions than it was given members"
        );
        for (index, contribution) in contributions.into_iter().enumerate() {
            let (token, _) = members[group][index];
            out[token][ranks[group][index]] = contribution;
        }
    }
    out
}

/// Reads one of the `testdata/` reference vectors the `#[ignore]`d
/// real-model embedding tests compare against, returning `None` (with a
/// note on stderr) when it isn't there.
///
/// Read at run time rather than through `include_str!` on purpose. These
/// fixtures are ground truth captured from real llama.cpp, so a checkout
/// may legitimately be missing one that nobody has generated yet — and a
/// compile-time include turns that into a build failure for the *whole*
/// test binary, including every test that never touches a fixture. A
/// missing fixture now skips its own test and nothing else, which is how
/// these tests already treat a missing `ORANGU_TEST_*_MODEL`.
#[cfg(test)]
pub fn read_reference_fixture(name: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/orangu-server/engine/arch/testdata")
        .join(name);
    match std::fs::read_to_string(&path) {
        Ok(contents) => Some(contents),
        Err(err) => {
            eprintln!("skipping: no reference fixture {} ({err})", path.display());
            None
        }
    }
}

/// Repeat-penalty state a caller passes to [`ModelForward::forward_maybe_
/// sampling`] when it wants greedy sampling done for it.
/// `recent_tokens` must already be trimmed to the sampler's own
/// `repeat_last_n` window (mirroring `engine::sampling`'s own
/// `apply_repeat_penalty`, which does the same trim before applying the
/// penalty) — the callee applies the penalty to exactly the slice it's
/// given, nothing more.
pub struct GreedySampleParams<'a> {
    pub recent_tokens: &'a [u32],
    pub repeat_penalty: f32,
}

/// [`ModelForward::forward_maybe_sampling`]'s result: either the callee
/// already picked the next token itself (`Token`, only possible when the
/// caller asked for greedy sampling *and* the backend has a GPU fast path
/// for it), or it didn't and the caller must run `engine::sampling::
/// Sampler::sample` over the returned logits itself, exactly as a plain
/// `forward` call would have required.
pub enum ForwardOutcome {
    Token(u32),
    Logits(Vec<f32>),
}

pub trait ModelForward: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// How many layers the forward pass actually runs.
    ///
    /// `config().n_layer` is the file's `block_count`, and on a file carrying
    /// a trailing multi-token-prediction block that counts one this engine
    /// never executes. The architectures able to load such a file stop before
    /// it and override this; for everything else the two are the same number.
    ///
    /// Reported rather than derived at the call site because the startup
    /// banner said `65 layers` for a `Qwen3.8-27B` that runs 64, and a layer
    /// count nobody can act on is worse than none.
    fn n_trunk_layer(&self) -> usize {
        self.config().n_layer
    }

    /// This model's Vulkan backend, when it has one.
    ///
    /// Exists for cross-architecture *instrumentation* rather than for work:
    /// `engine::generate` counts GPU submissions per decode step, and that
    /// number is only interesting compared **between** architectures — which
    /// nothing inside a single arch module can do. Defaults to `None` so a
    /// CPU-only or not-yet-converted arch needs no change and simply reports
    /// nothing.
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        None
    }

    /// A fresh KV cache sized for `capacity` positions, for a new sequence.
    fn new_kv_cache(&self, capacity: usize) -> KvCache;

    /// Runs `tokens` (a contiguous chunk of one sequence, starting at
    /// absolute position `start_pos`) through the model, appending their
    /// key/value vectors to `cache` as it goes, and returns the next-token
    /// logits (`[n_vocab]`) for the *last* token in `tokens` only — the one
    /// prediction a caller doing either prefill (find where generation
    /// starts) or decode (one token at a time) actually needs.
    ///
    /// `slot_id` is the caller's own `engine::scheduler::SlotGuard::id()`,
    /// and a real per-request id rather than a shared constant is
    /// load-bearing: `GemmaModel`'s Vulkan decode path threads it into the
    /// per-sequence GPU resource cache, so two
    /// `slots > 1` requests decoding concurrently don't collide on the same
    /// cached buffers. Architectures/backends with no such per-caller cache
    /// (every non-Vulkan-decode path) simply ignore it.
    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>>;

    /// Like `forward`, but lets the implementor sample the next token
    /// itself when `greedy_sample` is `Some` — skipping the full
    /// `[n_vocab]` logits readback entirely when it can. The default
    /// implementation always falls back to `forward` plus
    /// `ForwardOutcome::Logits`, so every architecture and backend
    /// combination keeps working correctly with no override needed; only
    /// `GemmaModel`'s Vulkan decode path currently overrides this to fuse
    /// the argmax into the same GPU submission.
    fn forward_maybe_sampling(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        greedy_sample: Option<GreedySampleParams<'_>>,
        slot_id: usize,
    ) -> Result<ForwardOutcome> {
        let _ = greedy_sample;
        self.forward(cache, tokens, start_pos, slot_id)
            .map(ForwardOutcome::Logits)
    }

    /// Like [`Self::forward`], but returns next-token logits for *every* input
    /// position (`Vec` of `n_tokens` `[n_vocab]` rows), not just the last —
    /// what speculative decoding's verify step needs to check each drafted
    /// token against the model's own prediction at that position. Appends all
    /// `tokens` to `cache` (which a later [`KvCache::truncate`] can roll back
    /// to the accepted prefix), and runs the multi-token / CPU-orchestrated
    /// path so one weight stream covers the whole draft at once.
    ///
    /// The default errors: an architecture opts in only by overriding this. A
    /// caller must not reach it for a model whose `forward` uses a KV path this
    /// can't stay consistent with (e.g. one that leaves keys/values GPU-only) —
    /// speculative decoding is gated to the plain CPU-KV path for that reason.
    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let _ = (cache, tokens, start_pos, slot_id);
        anyhow::bail!("this architecture does not support multi-position (speculative) decoding")
    }

    /// Every token's final hidden state (`[n_tokens, n_embd]`, before the
    /// output projection to vocab logits) — what an embeddings request
    /// pools over. A one-shot call: no KV cache reuse across calls.
    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Applied to the pooled embedding vector (`[n_embd]`, after mean/CLS/
    /// last-token pooling) before L2 normalization. The default is the
    /// identity — most architectures have nothing here — but a model
    /// converted with extra sentence-transformers "Dense" adapter layers
    /// (e.g. `gemma-embedding`'s `dense_2`/`dense_3`, confirmed against
    /// upstream `llama.cpp`'s `llm_graph_context::build_dense_out`: applied
    /// *after* pooling, not before) overrides this to run them. May change
    /// the vector's length (`gemma-embedding`'s `dense_2` widens
    /// `n_embd -> 4*n_embd` before `dense_3` narrows it back).
    fn post_pool_projection(&self, pooled: Vec<f32>) -> Result<Vec<f32>> {
        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertGating, ExpertProjection, ExpertRouting, attend, device_expert_admissible,
        evaluate_routed_experts, evaluate_routed_experts_batched_views, flag_is_on,
        matmul_host_fallback, project_expert, restore_order, top_k_indices,
    };
    use crate::engine::backend::Backend;
    use crate::engine::loader::{QuantMatrix, test_quant_matrix};
    use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_Q8_0};
    use crate::engine::tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Streaming splits a layer's experts into a device group and a host
    /// group by width, and the two come back in their own orders. Every
    /// expert's result then has to land on the expert it belongs to.
    ///
    /// Getting this wrong is invisible from the outside: each expert still
    /// receives a well-formed vector of the right length, just another
    /// expert's, and generation continues fluently from it. There is no
    /// panic, no shape error, and nothing in a throughput number that would
    /// hint at it.
    #[test]
    fn results_from_the_device_and_host_groups_land_on_their_own_ops() {
        // Ops 0..5 with 1, 3 on the host and 0, 2, 4 on the device — the
        // interleaving is the point; contiguous groups would pass under a
        // naive concatenation too.
        let device = (vec![0, 2, 4], vec![vec![0.0], vec![2.0], vec![4.0]]);
        let host = (vec![1, 3], vec![vec![1.0], vec![3.0]]);
        let out = restore_order(5, vec![device, host]);
        assert_eq!(
            out,
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]
        );
    }

    #[test]
    #[should_panic(expected = "different number of results")]
    fn a_group_that_returns_the_wrong_number_of_results_is_not_silently_shifted() {
        restore_order(3, vec![(vec![0, 1, 2], vec![vec![0.0], vec![1.0]])]);
    }

    /// `VAR=0` has to mean **off**, or a sweep of `VAR=0,1` runs the feature
    /// on both arms and reports the difference between two identical
    /// configurations as the patch's effect. That is not a hypothetical: it
    /// is how the first streaming A/B was set up, and the only outward sign
    /// was two arms agreeing closely.
    #[test]
    fn a_knob_set_to_zero_is_off_and_an_unset_one_stays_off() {
        for off in ["0", "", "  ", "no", "off", "false", "FALSE", " Off "] {
            assert!(!flag_is_on(Some(off)), "{off:?} should read as off");
        }
        for on in ["1", "yes", "on", "true", "2"] {
            assert!(flag_is_on(Some(on)), "{on:?} should read as on");
        }
        assert!(!flag_is_on(None));
    }

    /// A backend that panics on any type it does not list, the way
    /// `engine::backend::vulkan` panics when asked for a shader it has no
    /// pipeline for. The panic is the point: a fallback that silently sent
    /// the call through anyway would be indistinguishable from a working one
    /// until a real low-bit file hit it.
    struct Picky(&'static [u32]);

    impl Backend for Picky {
        fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
            assert!(
                self.0.contains(&w.ggml_type()),
                "backend has no kernel for ggml_type {}",
                w.ggml_type()
            );
            let _ = (x, n_tokens);
            vec![-1.0; n_tokens * w.out_dim]
        }
        fn supports_type(&self, ggml_type: u32) -> bool {
            self.0.contains(&ggml_type)
        }
    }

    /// A shared-expert matrix is exempt from the startup device-capability
    /// check (`engine::backend::is_cpu_only_tensor`), so it is the one weight
    /// that can reach a forward pass in a type the device cannot run. The
    /// fallback has to notice and route it to the host instead of handing the
    /// backend a call it will panic on.
    /// One `project_expert` call over a row range must equal two calls over
    /// its halves, concatenated.
    ///
    /// This is what lets a fused `ffn_gate_up_exps` be projected in one call
    /// instead of two, which is not a refactor but the point: each call is
    /// its own `rayon` region with its own join, its own `expert_store`
    /// lease, and — because both halves take the *same* activation — its own
    /// redundant `quantize_act` of a byte-identical input.
    ///
    /// It could fail for a real reason rather than a typo. `project_expert`
    /// parallelizes over rows and its kernel selection reads `in_dim`, not
    /// `n_rows`, so widening the range must not change how any single row is
    /// computed. If a future kernel ever picked a different accumulation for
    /// a wider range, the two forms would drift and this is what would say so.
    #[test]
    fn one_projection_over_both_halves_equals_two_over_each() {
        use crate::engine::backend::CpuBackend;
        use crate::engine::loader::test_expert_matrix;

        const N_EXPERT: usize = 3;
        const N_FF: usize = 5;
        const N_EMBD: usize = 4;

        let gate_up = test_expert_matrix(N_EXPERT, 2 * N_FF, N_EMBD);
        let a: Vec<f32> = (0..N_EMBD).map(|i| (i as f32 + 1.0) * 0.03).collect();
        let b: Vec<f32> = (0..N_EMBD).map(|i| 0.4 - i as f32 * 0.07).collect();
        let inputs: Vec<&[f32]> = vec![&a, &b];

        for expert in 0..N_EXPERT {
            let gate = project_expert(&CpuBackend, &gate_up, expert, 0, N_FF, &inputs);
            let up = project_expert(&CpuBackend, &gate_up, expert, N_FF, N_FF, &inputs);
            let both = project_expert(&CpuBackend, &gate_up, expert, 0, 2 * N_FF, &inputs);

            for input in 0..inputs.len() {
                assert_eq!(both[input].len(), 2 * N_FF);
                assert_eq!(
                    &both[input][..N_FF],
                    gate[input].as_slice(),
                    "expert {expert} input {input}: first half is not the gate"
                );
                assert_eq!(
                    &both[input][N_FF..],
                    up[input].as_slice(),
                    "expert {expert} input {input}: second half is not the up"
                );
            }
            // The halves must differ, or the equality above holds for a
            // projection that read the same rows twice.
            assert_ne!(gate[0], up[0], "expert {expert}: halves are identical");
        }
    }

    /// A fused gate/up tensor, addressed as two row ranges, must compute
    /// exactly what two calls against the two halves compute.
    ///
    /// This is the whole of what the row-range form adds, and every way of
    /// getting it wrong is silent: an off-by-`n_ff` `first_row` reads the up
    /// weights as the gate, a `n_rows` taken from the tensor rather than the
    /// half reads both halves as one projection, and a per-expert scalar
    /// indexed by *group* rather than by *expert* is correct exactly when the
    /// routing happens to select experts in ascending order — which it does
    /// in most small tests. The reference here is the per-expert path this
    /// replaces, run on the same weights, so a disagreement is this code's.
    ///
    /// No expert is device-resident (`test_expert_matrix` builds none), so
    /// both sides run the same host kernel and the comparison is exact
    /// rather than approximate. What the device does with these ops is the
    /// batched dispatch six other architectures already share.
    #[test]
    fn a_fused_gate_up_tensor_as_two_row_ranges_matches_the_per_expert_path() {
        use crate::engine::backend::CpuBackend;
        use crate::engine::loader::{test_expert_matrix, test_expert_matrix_resident};

        const N_EXPERT: usize = 4;
        const N_FF: usize = 3;
        const N_EMBD: usize = 5;

        // Gate and up fused: rows `0..N_FF` are the gate, `N_FF..2*N_FF` the up.
        let gate_up = test_expert_matrix(N_EXPERT, 2 * N_FF, N_EMBD);
        let down_exps = test_expert_matrix(N_EXPERT, N_EMBD, N_FF);
        // Per-expert scalars, distinct and not in expert order, so indexing
        // by group instead of by expert produces different numbers.
        let gate_up_scale: Vec<f32> = vec![1.75, 0.5, 2.25, 0.125];
        let down_scale: Vec<f32> = vec![0.375, 3.0, 0.75, 1.5];

        // Small and **positive**: `test_expert_matrix`'s weights run to ~50
        // for the higher expert indices, so an activation that sums negative
        // drives every gate output far enough negative that GELU returns
        // zero, the products are zero, and the two paths agree on a vector
        // of zeros. That is a test that passes while proving nothing, and
        // the assertion at the end of this one exists because it happened.
        let hidden: Vec<f32> = (0..2 * N_EMBD).map(|i| (i as f32 + 1.0) * 0.004).collect();
        // Descending experts in the first token's picks: an implementation
        // that indexes a scalar by group order disagrees here and not on a
        // sorted selection.
        let selection: Vec<Vec<(usize, f32)>> =
            vec![vec![(3, 0.6), (1, 0.4)], vec![(0, 0.7), (3, 0.3)]];

        let activate = |gate: &[f32], up: &[f32]| {
            let mut h = gate.to_vec();
            tensor::gelu_inplace(&mut h);
            tensor::mul_inplace(&mut h, up);
            h
        };

        // The reference: exactly the shape the per-expert path had.
        let expected = evaluate_routed_experts(&selection, |expert, members| {
            let inputs: Vec<&[f32]> = members
                .iter()
                .map(|&(t, _)| &hidden[t * N_EMBD..(t + 1) * N_EMBD])
                .collect();
            let mut gate = project_expert(&CpuBackend, &gate_up, expert, 0, N_FF, &inputs);
            let mut up = project_expert(&CpuBackend, &gate_up, expert, N_FF, N_FF, &inputs);
            let s = gate_up_scale[expert];
            for row in gate.iter_mut().chain(up.iter_mut()) {
                row.iter_mut().for_each(|v| *v *= s);
            }
            let hs: Vec<Vec<f32>> = gate.iter().zip(&up).map(|(g, u)| activate(g, u)).collect();
            let refs: Vec<&[f32]> = hs.iter().map(Vec::as_slice).collect();
            project_expert(&CpuBackend, &down_exps, expert, 0, N_EMBD, &refs)
                .into_iter()
                .zip(members)
                .map(|(mut contribution, &(_, weight))| {
                    let scale = down_scale[expert] * weight;
                    contribution.iter_mut().for_each(|v| *v *= scale);
                    contribution
                })
                .collect()
        });

        let actual = evaluate_routed_experts_batched_views(
            &CpuBackend,
            &selection,
            &hidden,
            N_EMBD,
            Some(&ExpertProjection {
                exps: &gate_up,
                first_row: 0,
                n_rows: N_FF,
                scale: Some(&gate_up_scale),
            }),
            &ExpertProjection {
                exps: &gate_up,
                first_row: N_FF,
                n_rows: N_FF,
                scale: Some(&gate_up_scale),
            },
            &ExpertProjection {
                scale: Some(&down_scale),
                ..ExpertProjection::whole(&down_exps)
            },
            activate,
        );

        assert_eq!(actual, expected, "row-range form disagreed with per-expert");
        // And the comparison is not vacuous: a contribution is a real vector.
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].len(), 2);
        assert_eq!(actual[0][0].len(), N_EMBD);
        for (token, picks) in actual.iter().enumerate() {
            for (rank, contribution) in picks.iter().enumerate() {
                assert!(
                    contribution.iter().any(|v| *v != 0.0),
                    "token {token} rank {rank} contributed only zeros, so the \
                     comparison proved nothing about it"
                );
            }
        }
        // The two tokens must not agree either, or a path that ignored its
        // input entirely would still match.
        assert_ne!(actual[0][0], actual[1][0]);

        // The same call again with every expert marked resident, which is
        // the *other* branch: batched `matmul_batch` over row-range views
        // instead of per-group `project_expert`. Those views are the only
        // place `first_row` reaches the dispatch path, and with no residency
        // no test reaches them at all. `CpuBackend` runs the batch, so this
        // compares bookkeeping rather than kernels — but a wrong row range
        // is a wrong *weight*, which no tolerance hides.
        let gate_up_res = test_expert_matrix_resident(N_EXPERT, 2 * N_FF, N_EMBD);
        let down_res = test_expert_matrix_resident(N_EXPERT, N_EMBD, N_FF);
        let dispatched = evaluate_routed_experts_batched_views(
            &CpuBackend,
            &selection,
            &hidden,
            N_EMBD,
            Some(&ExpertProjection {
                exps: &gate_up_res,
                first_row: 0,
                n_rows: N_FF,
                scale: Some(&gate_up_scale),
            }),
            &ExpertProjection {
                exps: &gate_up_res,
                first_row: N_FF,
                n_rows: N_FF,
                scale: Some(&gate_up_scale),
            },
            &ExpertProjection {
                scale: Some(&down_scale),
                ..ExpertProjection::whole(&down_res)
            },
            activate,
        );
        for (token, picks) in dispatched.iter().enumerate() {
            for (rank, contribution) in picks.iter().enumerate() {
                let reference = &expected[token][rank];
                assert_eq!(contribution.len(), reference.len());
                for (i, (got, want)) in contribution.iter().zip(reference).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "batched dispatch disagreed at token {token} rank {rank} \
                         element {i}: {got} against {want}"
                    );
                }
            }
        }
    }

    /// A routed expert may only be dispatched to the device when the tier
    /// actually holds it.
    ///
    /// The three other terms are properties of the build and the file and
    /// are stable for a whole run; `resident` is per expert, and it is the
    /// only thing bounding an arena that never evicts. A dispatch policy
    /// that admits a non-resident expert grows the arena past the budget
    /// `main::plan_expert_tier` set, so the tier stops being one — and the
    /// symptom is not a wrong answer, it is device-memory paging, which
    /// reads as "the GPU expert path is slow" rather than as a defect. This
    /// pins the term so it cannot be dropped quietly.
    #[test]
    fn a_routed_expert_reaches_the_device_only_when_the_tier_holds_it() {
        // Everything else admissible, residency the only variable.
        assert!(device_expert_admissible(true, true, true, true));
        assert!(
            !device_expert_admissible(true, true, true, false),
            "a non-resident expert must take the host path even with the knob on, \
             a GPU present and a kernel for its type"
        );
        // And the default a model without a tier presents: no expert is
        // resident, so no expert is dispatched, whatever the knob says.
        for &enabled in &[true, false] {
            assert!(!device_expert_admissible(enabled, true, true, false));
        }
        // The other three terms each still veto on their own.
        assert!(!device_expert_admissible(false, true, true, true));
        assert!(!device_expert_admissible(true, false, true, true));
        assert!(!device_expert_admissible(true, true, false, true));
    }

    #[test]
    fn matmul_host_fallback_routes_a_type_the_backend_lacks_to_the_host() {
        // One row of eight `f32` weights, all 1.0, as `F32` — so the host
        // result is just the sum of the input.
        let weights: Vec<u8> = (0..8).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let w = test_quant_matrix(&weights, GGML_TYPE_F32, 8, 1);
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // A backend that has this type runs it, and we see the backend's own
        // sentinel rather than the host's answer.
        assert_eq!(
            matmul_host_fallback(&Picky(&[GGML_TYPE_F32]), &x, 1, &w),
            vec![-1.0]
        );
        // A backend that lacks it must not be called at all — the host
        // computes the real dot instead of the sentinel.
        assert_eq!(
            matmul_host_fallback(&Picky(&[GGML_TYPE_Q8_0]), &x, 1, &w),
            vec![36.0]
        );
    }

    /// A stand-in for one expert's arithmetic, deliberately order-sensitive
    /// in floating point.
    ///
    /// The magnitudes span `f32`'s precision on purpose: two experts near
    /// `1e8` that nearly cancel, and two near `1`. Summed in selection order
    /// the large pair cancels first and the small pair survives; summed the
    /// other way round the small pair is absorbed into the large one and
    /// vanishes. That is the whole hazard the batch union has to avoid, so
    /// the fixture has to contain it — a well-conditioned one would let a
    /// reordering implementation pass. `the_reference_contributions_are_
    /// order_sensitive_in_f32` is what holds this claim to account.
    fn contribution(expert: usize, token: usize, weight: f32, n_out: usize) -> Vec<f32> {
        let magnitude = match expert {
            1 => 1.0e8,
            9 => -1.0e8,
            _ => 1.0,
        };
        (0..n_out)
            .map(|o| magnitude * (1.0 + weight * 1e-3 + token as f32 * 1e-4 + o as f32 * 1e-5))
            .collect()
    }

    /// What every architecture did before the batch union: walk positions,
    /// and for each evaluate its selected experts in selection order.
    fn per_token_reference(selection: &[Vec<(usize, f32)>], n_out: usize) -> Vec<Vec<Vec<f32>>> {
        selection
            .iter()
            .enumerate()
            .map(|(token, picks)| {
                picks
                    .iter()
                    .map(|&(expert, weight)| contribution(expert, token, weight, n_out))
                    .collect()
            })
            .collect()
    }

    /// A batch whose tokens deliberately overlap: expert 3 is selected by
    /// three different tokens at three different ranks, which is the case the
    /// union exists for and the case a naive scatter would misplace.
    fn overlapping_selection() -> Vec<Vec<(usize, f32)>> {
        vec![
            vec![(3, 0.5), (7, 0.25), (1, 0.25)],
            vec![(7, 0.6), (3, 0.4)],
            vec![(1, 0.1), (9, 0.2), (3, 0.3), (7, 0.4)],
            vec![(9, 1.0)],
        ]
    }

    /// The kill criterion for the whole batch-union change: regrouping the
    /// evaluation must not move a single bit of the result. Compared with
    /// `==` on `f32`, not an epsilon — an epsilon would pass for a change
    /// that quietly reordered the arithmetic, which is exactly what this is
    /// built to rule out.
    #[test]
    fn grouping_by_expert_reproduces_the_per_token_result_bit_for_bit() {
        let selection = overlapping_selection();
        let n_out = 64;
        let expected = per_token_reference(&selection, n_out);
        let actual = evaluate_routed_experts(&selection, |expert, members| {
            members
                .iter()
                .map(|&(token, weight)| contribution(expert, token, weight, n_out))
                .collect()
        });
        assert_eq!(actual, expected);
    }

    /// The bit-for-bit test above is only meaningful if the values it
    /// compares would actually differ under a reordering. Prove they do,
    /// rather than trusting that they do.
    #[test]
    fn the_reference_contributions_are_order_sensitive_in_f32() {
        let picks = &overlapping_selection()[2];
        let n_out = 64;
        let sum = |order: &mut dyn Iterator<Item = &(usize, f32)>| -> f32 {
            let mut acc = vec![0f32; n_out];
            for &(expert, weight) in order {
                for (a, c) in acc
                    .iter_mut()
                    .zip(contribution(expert, 2, weight, n_out).iter())
                {
                    *a += c;
                }
            }
            acc.iter().sum()
        };
        let forward = sum(&mut picks.iter());
        let reversed = sum(&mut picks.iter().rev());
        assert_ne!(
            forward, reversed,
            "these values sum identically in any order, so the bit-identity test proves nothing"
        );
    }

    /// The saving itself: an expert three tokens selected is evaluated once,
    /// not three times. Counted rather than inferred from a rate.
    #[test]
    fn each_distinct_expert_is_evaluated_exactly_once() {
        let selection = overlapping_selection();
        let calls = AtomicUsize::new(0);
        let members_seen = AtomicUsize::new(0);
        evaluate_routed_experts(&selection, |expert, members| {
            calls.fetch_add(1, Ordering::Relaxed);
            members_seen.fetch_add(members.len(), Ordering::Relaxed);
            members
                .iter()
                .map(|&(token, weight)| contribution(expert, token, weight, 8))
                .collect()
        });
        // Ten (token, expert) selections over four distinct experts.
        assert_eq!(members_seen.load(Ordering::Relaxed), 10);
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }

    /// Every contribution must land at its own token *and its own rank* —
    /// the rank is what the caller's summation order depends on, and an
    /// expert selected at rank 0 by one token and rank 2 by another is where
    /// a scatter keyed only on the token would silently go wrong.
    #[test]
    fn contributions_land_at_the_rank_their_router_chose() {
        let selection = overlapping_selection();
        let out = evaluate_routed_experts(&selection, |expert, members| {
            members
                .iter()
                .map(|&(_, weight)| vec![expert as f32, weight])
                .collect()
        });
        assert_eq!(out[0][0], vec![3.0, 0.5]);
        assert_eq!(out[0][2], vec![1.0, 0.25]);
        assert_eq!(out[1][1], vec![3.0, 0.4]);
        assert_eq!(out[2][2], vec![3.0, 0.3]);
        assert_eq!(out[3][0], vec![9.0, 1.0]);
        assert!(
            out.iter().flatten().all(|c| !c.is_empty()),
            "a slot was left unfilled"
        );
    }

    /// `project_expert` inverts the loops — rows outside, inputs inside —
    /// so its results must still be exactly what dotting each input against
    /// each row separately gives. Element-wise `==`, since each output is a
    /// single independent dot and nothing here is allowed to change it.
    #[test]
    fn projecting_several_inputs_at_once_matches_projecting_them_one_by_one() {
        use crate::engine::loader::test_expert_matrix;
        let n_expert = 3;
        let out_dim = 5;
        let in_dim = 8;
        let weights = test_expert_matrix(n_expert, out_dim, in_dim);
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                (0..in_dim)
                    .map(|d| (i * 13 + d * 7) as f32 * 0.031)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();

        for expert in 0..n_expert {
            let together = project_expert(
                &crate::engine::backend::CpuBackend,
                &weights,
                expert,
                0,
                out_dim,
                &refs,
            );
            for (i, input) in refs.iter().enumerate() {
                let alone: Vec<f32> = (0..out_dim)
                    .map(|o| tensor::dot(input, &weights.row(expert, o)))
                    .collect();
                assert_eq!(together[i], alone, "expert {expert}, input {i}");
            }
        }
    }

    /// A row range picks out a slice of the matrix — what a fused gate/up
    /// tensor needs, where the same matrix is two projections back to back.
    /// The offset must reach the right rows, not row 0 onwards.
    #[test]
    fn a_row_range_projects_that_range_and_no_other() {
        use crate::engine::loader::test_expert_matrix;
        let weights = test_expert_matrix(2, 6, 4);
        let input: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let refs: [&[f32]; 1] = [&input];

        let second_half = project_expert(
            &crate::engine::backend::CpuBackend,
            &weights,
            1,
            3,
            3,
            &refs,
        );
        let expected: Vec<f32> = (3..6)
            .map(|o| tensor::dot(&input, &weights.row(1, o)))
            .collect();
        assert_eq!(second_half[0], expected);
    }

    /// An expert nothing routed to must not be read at all — the guard that
    /// keeps a stray empty group from dequantizing a whole matrix for nobody.
    #[test]
    fn projecting_no_inputs_reads_nothing() {
        use crate::engine::loader::test_expert_matrix;
        let weights = test_expert_matrix(2, 4, 4);
        assert!(
            project_expert(&crate::engine::backend::CpuBackend, &weights, 0, 0, 4, &[]).is_empty()
        );
    }

    mod budget {
        use super::super::{BUDGET_MAX_BATCH, trim_to_expert_budget};
        use crate::engine::loader::test_expert_matrix;

        fn picks(experts: &[(usize, f32)]) -> Vec<(usize, f32)> {
            experts.to_vec()
        }

        /// **Rule one.** A prefill batch's union is far larger than any
        /// budget, and trimming it corrupts the hidden state that goes into
        /// the KV cache. The guard is a batch-size check, and it must reject
        /// anything wider than a speculative verify.
        #[test]
        fn the_budget_never_touches_a_prefill_sized_batch() {
            let weights = test_expert_matrix(64, 4, 8);
            let mut wide: Vec<Vec<(usize, f32)>> = (0..BUDGET_MAX_BATCH + 1)
                .map(|t| picks(&[(t, 0.5), (t + 20, 0.5)]))
                .collect();
            let before = wide.clone();
            trim_to_expert_budget(&mut wide, &weights, 1);
            assert_eq!(wide, before, "a batch wider than the guard was trimmed");
        }

        /// **Rule two.** A position whose whole selection is dropped receives
        /// only the shared expert, and that wrong state enters the KV cache.
        /// Its best expert has to come back.
        #[test]
        fn no_position_is_left_with_nothing_routed() {
            let weights = test_expert_matrix(64, 4, 8);
            // Two positions with entirely disjoint selections and a budget of
            // one: whichever expert wins, the other position is emptied.
            let mut selection = vec![picks(&[(1, 0.9)]), picks(&[(2, 0.1)])];
            trim_to_expert_budget(&mut selection, &weights, 1);
            assert!(
                selection.iter().all(|p| !p.is_empty()),
                "a position was left with nothing routed: {selection:?}"
            );
        }

        /// Dropping experts must not silently shrink the routed branch: the
        /// survivors carry the weight the dropped ones were carrying.
        #[test]
        fn the_surviving_weights_preserve_the_original_magnitude() {
            let weights = test_expert_matrix(64, 4, 8);
            let mut selection = vec![picks(&[(1, 0.6), (2, 0.3), (3, 0.1)])];
            let before: f32 = selection[0].iter().map(|(_, w)| w).sum();
            trim_to_expert_budget(&mut selection, &weights, 2);
            let after: f32 = selection[0].iter().map(|(_, w)| w).sum();
            assert!(selection[0].len() < 3, "nothing was dropped");
            assert!(
                (before - after).abs() < 1e-5,
                "magnitude moved: {before} -> {after}"
            );
        }

        /// What survives is chosen by aggregate gate weight across the batch,
        /// not by which position happened to be looked at first.
        #[test]
        fn the_heaviest_experts_across_the_batch_are_the_ones_kept() {
            let weights = test_expert_matrix(64, 4, 8);
            // Expert 7 is small in both positions but adds up to more than
            // expert 5, which is large in only one.
            let mut selection = vec![
                picks(&[(5, 0.50), (7, 0.30)]),
                picks(&[(9, 0.05), (7, 0.30)]),
            ];
            trim_to_expert_budget(&mut selection, &weights, 1);
            assert!(
                selection.iter().all(|p| p.iter().any(|(e, _)| *e == 7)),
                "the batch-heaviest expert was dropped: {selection:?}"
            );
        }

        /// A union already inside the budget is left exactly alone — no
        /// reordering, no reweighting.
        #[test]
        fn a_selection_within_budget_is_untouched() {
            let weights = test_expert_matrix(64, 4, 8);
            let mut selection = vec![picks(&[(3, 0.7), (4, 0.3)])];
            let before = selection.clone();
            trim_to_expert_budget(&mut selection, &weights, 8);
            assert_eq!(selection, before);
        }
    }

    /// Decode is one token: nothing to group, and the result must still be
    /// exactly the per-token one.
    #[test]
    fn a_single_token_batch_is_unchanged() {
        let selection = vec![vec![(5, 0.7), (2, 0.3)]];
        let expected = per_token_reference(&selection, 16);
        let actual = evaluate_routed_experts(&selection, |expert, members| {
            members
                .iter()
                .map(|&(token, weight)| contribution(expert, token, weight, 16))
                .collect()
        });
        assert_eq!(actual, expected);
    }

    /// A dense layer, or a batch nobody routed: no experts, no panic, and one
    /// empty slot list per token.
    #[test]
    fn an_empty_selection_evaluates_nothing() {
        let selection = vec![Vec::new(), Vec::new()];
        let calls = AtomicUsize::new(0);
        let out = evaluate_routed_experts(&selection, |_, _: &[(usize, f32)]| {
            calls.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        });
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(Vec::is_empty));
    }

    #[test]
    fn top_k_indices_keeps_only_the_largest_entries_highest_first() {
        let scores = vec![1.0, 5.0, 3.0, 4.0];
        assert_eq!(top_k_indices(&scores, 2), vec![1, 3]);
        // Asking for more than there are is not an error.
        assert_eq!(top_k_indices(&scores, 9).len(), 4);
    }

    #[test]
    fn attention_without_a_sink_is_a_plain_softmax_average() {
        let q = vec![1.0, 0.0];
        let keys = vec![
            1.0, 0.0, //
            1.0, 0.0, //
        ];
        // Two identical keys: the average is the key.
        let out = attend(&q, &keys, 2, 2, 1.0, None);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn an_attention_sink_takes_softmax_mass_without_contributing_a_value() {
        let q = vec![1.0, 0.0];
        let keys = vec![1.0, 0.0];
        // One key scoring 0 (after scale) against a sink of 0: the sink
        // halves the weight the single value gets.
        let out = attend(&q, &keys, 2, 2, 0.0, Some(0.0));
        assert!((out[0] - 0.5).abs() < 1e-6, "{out:?}");
    }

    /// An MLA key is `[compressed-KV | rotary]` and only the first part is
    /// the value, so the scores must see the whole row while the weighted
    /// sum sees only its head.
    #[test]
    fn a_narrower_value_reads_the_leading_part_of_each_key_row() {
        let q = vec![0.0, 0.0, 1.0];
        let keys = vec![
            7.0, 8.0, 0.0, // scores 0
            1.0, 2.0, 100.0, // scores 100 — wins the softmax outright
        ];
        let out = attend(&q, &keys, 3, 2, 1.0, None);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-4, "{out:?}");
        assert!((out[1] - 2.0).abs() < 1e-4, "{out:?}");
    }

    /// The selection bias steers which experts are picked but must not
    /// reach the weights — DeepSeek-V3's auxiliary-loss-free balancing.
    #[test]
    fn the_selection_bias_changes_the_choice_but_never_the_weights() {
        let routing = ExpertRouting {
            n_expert_used: 1,
            gating: ExpertGating::Sigmoid,
            weights_norm: false,
            weights_scale: 1.0,
            groups: None,
        };
        let logits = vec![1.0, 0.0];
        let (unbiased, _) = routing.route(&logits, None, None);
        assert_eq!(unbiased, vec![0]);

        let (biased, weights) = routing.route(&logits, Some(&[0.0, 10.0]), None);
        assert_eq!(biased, vec![1]);
        // The weight is expert 1's own unbiased probability, not the
        // biased score that selected it.
        assert!(
            (weights[0] - crate::engine::tensor::sigmoid(0.0)).abs() < 1e-6,
            "{weights:?}"
        );
    }

    #[test]
    fn renormalized_weights_sum_to_the_configured_scale() {
        let routing = ExpertRouting {
            n_expert_used: 2,
            gating: ExpertGating::Sigmoid,
            weights_norm: true,
            weights_scale: 2.5,
            groups: None,
        };
        let (_, weights) = routing.route(&[1.0, 0.5, -3.0], None, None);
        assert!(
            (weights.iter().sum::<f32>() - 2.5).abs() < 1e-5,
            "{weights:?}"
        );
    }

    #[test]
    fn only_the_implemented_gating_functions_load() {
        assert_eq!(ExpertGating::from_gguf(2).unwrap(), ExpertGating::Sigmoid);
        assert_eq!(
            ExpertGating::from_gguf(4).unwrap(),
            ExpertGating::SqrtSoftplus
        );
        assert!(ExpertGating::from_gguf(1).is_err());
    }
}

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
//! attention + MoE path), `deepseek4` (DeepSeek-V4: hyper-connections,
//! shared-key attention over a sliding window plus compressed blocks, and
//! hash-routed experts), `glm` (GLM with DeepSeek sparse attention:
//! absorbed multi-head latent attention plus a lightning indexer), `kimi3`
//! (Kimi-K3: hybrid delta-net/latent attention with cross-layer residuals
//! and a latent MoE), and `dflash` (DeepSeek draft sidecars, served through
//! the target model they draft for).

pub mod deepseek4;
pub mod dflash;
pub mod gemma;
pub mod glm;
pub mod kimi3;
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3next;

use crate::engine::kv_cache::KvCache;
use crate::engine::loader::ModelConfig;
use crate::engine::tensor;
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

/// A MoE layer's routing rules, all read from the file's metadata.
pub(crate) struct ExpertRouting {
    pub n_expert_used: usize,
    pub gating: ExpertGating,
    /// `<arch>.expert_weights_norm` — renormalize the selected experts'
    /// weights so they sum to one.
    pub weights_norm: bool,
    /// `<arch>.expert_weights_scale`, applied after any renormalization.
    pub weights_scale: f32,
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
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, op) in ops.iter().enumerate() {
        buckets.entry(op.n_tokens).or_default().push(index);
    }
    let mut out: Vec<Option<Vec<f32>>> = (0..ops.len()).map(|_| None).collect();
    for indices in buckets.into_values() {
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
/// keeps [`project_expert`].
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
    use crate::engine::backend::MatmulOp;

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
    let on_device: Vec<bool> = experts
        .iter()
        .map(|&e| {
            gate_exps.is_device_resident(e)
                && up_exps.is_device_resident(e)
                && down_exps.is_device_resident(e)
        })
        .collect();

    let xs: Vec<Vec<f32>> = members
        .iter()
        .map(|members| {
            let mut x = Vec::with_capacity(members.len() * n_embd);
            for &(token, _) in members {
                x.extend_from_slice(&hidden[token * n_embd..(token + 1) * n_embd]);
            }
            x
        })
        .collect();

    // Views first, then ops: a `MatmulOp` borrows its weight.
    let gate_views: Vec<_> = experts
        .iter()
        .map(|&e| gate_exps.expert_matrix(e))
        .collect();
    let up_views: Vec<_> = experts.iter().map(|&e| up_exps.expert_matrix(e)).collect();
    let mut ops: Vec<MatmulOp<'_>> = Vec::new();
    let mut op_group: Vec<usize> = Vec::new();
    for group in 0..experts.len() {
        if !on_device[group] {
            continue;
        }
        ops.push(MatmulOp {
            x: &xs[group],
            n_tokens: members[group].len(),
            w: &gate_views[group],
        });
        ops.push(MatmulOp {
            x: &xs[group],
            n_tokens: members[group].len(),
            w: &up_views[group],
        });
        op_group.push(group);
    }
    let batched = matmul_batch_mixed(backend, &ops);
    // Back into per-group slots, so the activation loop reads the same way
    // whichever path produced a group's projections.
    let mut slots: Vec<Option<(Vec<f32>, Vec<f32>)>> = (0..experts.len()).map(|_| None).collect();
    for (slot, group) in op_group.iter().enumerate() {
        slots[*group] = Some((batched[slot * 2].clone(), batched[slot * 2 + 1].clone()));
    }
    for group in 0..experts.len() {
        if on_device[group] {
            continue;
        }
        let inputs: Vec<&[f32]> = (0..members[group].len())
            .map(|m| &xs[group][m * n_embd..(m + 1) * n_embd])
            .collect();
        let gate = project_expert(
            backend,
            gate_exps,
            experts[group],
            0,
            gate_exps.out_dim,
            &inputs,
        );
        let up = project_expert(
            backend,
            up_exps,
            experts[group],
            0,
            up_exps.out_dim,
            &inputs,
        );
        slots[group] = Some((gate.concat(), up.concat()));
    }
    let projected: Vec<(Vec<f32>, Vec<f32>)> = slots
        .into_iter()
        .map(|p| p.expect("every group took one path or the other"))
        .collect();

    // Activation, per member, into one contiguous run per group — the down
    // projection's operand.
    let ffn_dim = gate_exps.out_dim;
    let hs: Vec<Vec<f32>> = (0..experts.len())
        .map(|group| {
            let (gate, up) = &projected[group];
            let mut h = Vec::with_capacity(members[group].len() * ffn_dim);
            for m in 0..members[group].len() {
                let range = m * ffn_dim..(m + 1) * ffn_dim;
                h.extend_from_slice(&activate(&gate[range.clone()], &up[range]));
            }
            h
        })
        .collect();

    let down_views: Vec<_> = experts
        .iter()
        .map(|&e| down_exps.expert_matrix(e))
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
    let down_batched = matmul_batch_mixed(backend, &down_ops);
    let mut down: Vec<Option<Vec<f32>>> = (0..experts.len()).map(|_| None).collect();
    for (slot, group) in down_group.iter().enumerate() {
        down[*group] = Some(down_batched[slot].clone());
    }
    let ffn_dim_in = down_exps.in_dim;
    for group in 0..experts.len() {
        if on_device[group] {
            continue;
        }
        let inputs: Vec<&[f32]> = (0..members[group].len())
            .map(|m| &hs[group][m * ffn_dim_in..(m + 1) * ffn_dim_in])
            .collect();
        down[group] = Some(
            project_expert(
                backend,
                down_exps,
                experts[group],
                0,
                down_exps.out_dim,
                &inputs,
            )
            .concat(),
        );
    }
    let down: Vec<Vec<f32>> = down
        .into_iter()
        .map(|d| d.expect("every group took one path or the other"))
        .collect();

    let out_dim = down_exps.out_dim;
    let mut out: Vec<Vec<Vec<f32>>> = selection
        .iter()
        .map(|picks| vec![Vec::new(); picks.len()])
        .collect();
    for group in 0..experts.len() {
        for (m, &(token, weight)) in members[group].iter().enumerate() {
            let mut contribution = down[group][m * out_dim..(m + 1) * out_dim].to_vec();
            contribution.iter_mut().for_each(|v| *v *= weight);
            out[token][ranks[group][m]] = contribution;
        }
    }
    out
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
pub(crate) fn gpu_experts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ORANGU_GPU_EXPERTS").is_some())
}

/// One expert's projection on the GPU, or `None` to take the host path.
///
/// The whole of the dispatch: an expert's rows are contiguous in the
/// stacked tensor, so `ExpertQuantMatrix::expert_matrix` views them as an
/// ordinary `QuantMatrix` and `Backend::matmul` already knows what to do
/// with one. No new kernel — every GPU backend already has one for every
/// quantization an expert is stored in, which is what
/// `engine::expert_tier` is written against.
///
/// Returns `None` — the host path — when the knob is off, the backend has
/// no GPU, or the backend has no kernel for this quantization. That last
/// one matters: the `IQ*` types the Vulkan backend lacks are exactly the
/// ones large MoE models are shipped in.
fn gpu_project_expert(
    backend: &dyn crate::engine::backend::Backend,
    weights: &crate::engine::loader::ExpertQuantMatrix,
    expert: usize,
    first_row: usize,
    n_rows: usize,
    inputs: &[&[f32]],
) -> Option<Vec<Vec<f32>>> {
    if !gpu_experts() || backend.as_wgpu().is_none() || !backend.supports_type(weights.ggml_type())
    {
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
    let span = lease.bytes();

    let mut by_row = vec![0f32; n_rows * n_inputs];
    by_row
        .par_chunks_mut(n_inputs)
        .enumerate()
        // One dequantization buffer per rayon job rather than per row: the
        // rows are all `in_dim` wide, so the buffer is filled and refilled
        // without ever reallocating. `for_each_init` runs the initializer
        // once per work split, not once per item, which is exactly the reuse
        // scope wanted here.
        .for_each_init(Vec::new, |weights_row, (row, out)| {
            match span {
                Some(span) => weights.row_from(span, first_row + row, weights_row),
                None => weights.row_into(expert, first_row + row, weights_row),
            }
            for (slot, input) in out.iter_mut().zip(inputs) {
                *slot = tensor::dot(input, weights_row);
            }
        });
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

/// One sequence's pending single-token decode step, as an element of a
/// cross-sequence batch (see `engine::batch::BatchCoordinator`).
/// Each sequence keeps its own `cache`/`start_pos`/`greedy_sample`
/// (attention, RoPE, and the KV-cache write all stay per-sequence even
/// when the *matmul* steps in between are fused across every item in the
/// batch — see [`ModelForward::forward_batch_decode`]'s own doc comment).
pub struct BatchDecodeItem<'a> {
    pub cache: &'a mut KvCache,
    pub token: u32,
    pub start_pos: usize,
    pub greedy_sample: Option<GreedySampleParams<'a>>,
    /// The submitting request's own `engine::scheduler::SlotGuard::id()`
    /// — **not** this item's position within `items`. `BatchCoordinator`
    /// can run two *different* batches' own `forward_batch_decode` calls
    /// genuinely concurrently (its own doc comment: the coordinator's
    /// lock is released before processing a batch, specifically so a new
    /// window can start collecting while an older one is still being
    /// processed) — a GPU-resident implementation that caches per-
    /// sequence resources by *array index within one call*
    /// (`GemmaModel::record_batched_decode_forward`'s first, broken
    /// attempt at this: `1..=items.len()`) would let two unrelated
    /// sequences from two concurrently-running batches collide on the
    /// exact same cached buffers — since the array index resets to `0`
    /// every call, two overlapping batches' own "index 0" always
    /// coincide, corrupting whichever one loses the race, silently. A
    /// `SlotGuard`'s own `id()` is unique across every *concurrently
    /// held* slot for that slot's entire lifetime (`SlotPool::acquire`'s
    /// own guarantee — capped by the semaphore, one live guard per index
    /// at a time), which is exactly the uniqueness overlapping batches
    /// need and a per-call array index doesn't provide.
    pub slot_id: usize,
}

pub trait ModelForward: Send + Sync {
    fn config(&self) -> &ModelConfig;

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
    /// `slot_id` is the caller's own `engine::scheduler::SlotGuard::id()` —
    /// see [`BatchDecodeItem::slot_id`]'s doc comment for why a real,
    /// per-request slot id (not a shared constant) is load-bearing here:
    /// `GemmaModel`'s Vulkan decode path threads it into the same
    /// per-sequence GPU resource cache the batched path uses, so two
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

    /// Runs a *cross-sequence batch* of independent single-token decode
    /// steps, driven by `engine::batch::BatchCoordinator`.
    /// Each `items[i]` is one sequence's own pending decode step (its own
    /// KV cache, position, token); the *matmul* steps every layer needs
    /// (QKV, `wo`, FFN, PLE, `lm_head` — the weight-bandwidth-heavy ones,
    /// the actual thing "GEMM batching" amortizes) are fused into one call
    /// across every item in the batch instead of one call per sequence,
    /// while attention, RoPE, and the KV-cache write stay per-sequence
    /// (each sequence has its own cache and its own position — there is
    /// no shared state to fuse there, unlike the matmuls).
    ///
    /// The default implementation just loops over `items` calling
    /// `forward_maybe_sampling` on each independently (correct, no fusion,
    /// no batching win) — so `llama`/`qwen35moe` need no override to keep
    /// working; only `GemmaModel` currently overrides this with the real
    /// batched implementation. That override is correctness-verified but
    /// **measured slower** under real concurrent load than not batching at
    /// all — `engine::generate::Engine::batch_coordinator`'s own doc
    /// comment has the numbers and the likely cause — so it's only ever
    /// reached when a caller opts in (`ORANGU_BATCH_DECODE=1`); this
    /// default implementation is what every other configuration actually
    /// runs, including a `slots > 1` deployment that hasn't opted in.
    fn forward_batch_decode(
        &self,
        items: &mut [BatchDecodeItem<'_>],
    ) -> Result<Vec<ForwardOutcome>> {
        items
            .iter_mut()
            .map(|item| {
                self.forward_maybe_sampling(
                    item.cache,
                    &[item.token],
                    item.start_pos,
                    item.greedy_sample.take(),
                    item.slot_id,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertGating, ExpertRouting, attend, evaluate_routed_experts, project_expert, top_k_indices,
    };
    use crate::engine::tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

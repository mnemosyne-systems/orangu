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

//! Hyper-connections (mHC) — the residual *stream bundle* that replaces the
//! single residual vector in `engine::arch::deepseek4` and
//! `engine::arch::glm5`. Not an architecture itself, the way
//! [`kda`](super::kda) and [`qwen_hybrid`](super::qwen_hybrid) are not.
//!
//! The state carried between sub-layers is `hyper_connection.count`
//! parallel `n_embd`-wide streams rather than one vector. Each sub-layer
//! (attention, then FFN) is wrapped by a pair:
//!
//! * [`HyperShape::collapse_into`] — one weightless RMSNorm over all the
//!   streams flattened, one low-rank projection off that, and the result
//!   split into three sets of weights: an in-mix (`pre`) that folds the
//!   streams down to the single vector the sub-layer reads, an out-mix
//!   (`post`) that scales the sub-layer's output on the way back, and a
//!   `[count, count]` stream-combination matrix (`comb`) that is
//!   Sinkhorn-normalized into a doubly stochastic one.
//! * [`HyperShape::expand`] — writes the sub-layer's output back into every
//!   stream: `post[dst]` times the output, plus the `comb`-weighted mix of
//!   the streams as they were *before* the sub-layer ran.
//!
//! The two architectures differ only in how the bundle is finally reduced
//! to the one vector the output projection reads: `deepseek4` collapses it
//! with a mixer of its own (`output_hc_*`, an in-mix and nothing else),
//! `glm5` takes an unweighted [`HyperShape::mean_into`]. Everything above
//! that line is the same arithmetic on both, which is why it lives here.
//!
//! Transcribed from upstream `llama.cpp`'s `build_hc_pre` / `build_hc_post`
//! / `build_hc_sinkhorn`.

use anyhow::{Context, Result};

use super::rms_norm_rows_into;
use crate::engine::backend::Backend;
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::tensor;

/// One sub-layer's hyper-connection mixer: the projection that predicts the
/// in-mix, out-mix, and stream-combination weights from the streams, plus
/// that projection's affine post-scaling.
pub(crate) struct HyperConnection {
    /// `[count * n_embd, (2 + count) * count]`.
    pub weights: QuantMatrix,
    /// `[(2 + count) * count]`.
    pub base: Vec<f32>,
    /// `[3]` — one scale each for the in-mix, out-mix, and combination
    /// parts of the projection's output.
    pub scale: Vec<f32>,
}

impl HyperConnection {
    /// Loads the `<prefix>_fn` / `<prefix>_base` / `<prefix>_scale` triple,
    /// e.g. `blk.7.hc_attn` or `output_hc`.
    pub(crate) fn load(loaded: &LoadedModel, prefix: &str) -> Result<Self> {
        Ok(Self {
            weights: loaded
                .matrix(&format!("{prefix}_fn.weight"))
                .with_context(|| format!("loading {prefix}_fn.weight"))?,
            base: loaded
                .tensor(&format!("{prefix}_base.weight"))
                .with_context(|| format!("loading {prefix}_base.weight"))?
                .0,
            scale: loaded
                .tensor(&format!("{prefix}_scale.weight"))
                .with_context(|| format!("loading {prefix}_scale.weight"))?
                .0,
        })
    }
}

/// The hyper-connection hyperparameters every layer of one model shares.
pub(crate) struct HyperShape {
    /// `hyper_connection.count` — how many parallel residual streams.
    pub count: usize,
    /// `hyper_connection.sinkhorn_iterations`.
    pub sinkhorn_iters: usize,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
    pub n_embd: usize,
    /// The model's RMSNorm epsilon, for the weightless norm the mixer
    /// projects from.
    pub rms_eps: f32,
}

impl HyperShape {
    /// Reads the three `hyper_connection.*` keys, all of which are
    /// required: a file that declares hyper-connected layers and omits the
    /// stream count has no residual shape at all.
    pub(crate) fn from_gguf(loaded: &LoadedModel) -> Result<Self> {
        let count = loaded
            .metadata_u64("hyper_connection.count")
            .context("missing hyper_connection.count")? as usize;
        anyhow::ensure!(count > 0, "hyper_connection.count must be at least 1");
        let sinkhorn_iters = loaded
            .metadata_u64("hyper_connection.sinkhorn_iterations")
            .context("missing hyper_connection.sinkhorn_iterations")?
            as usize;
        anyhow::ensure!(
            sinkhorn_iters > 0,
            "hyper_connection.sinkhorn_iterations must be at least 1"
        );
        Ok(Self {
            count,
            sinkhorn_iters,
            // Absent in a file that leaves it at the reference default,
            // which is what every released checkpoint of either
            // architecture happens to write out anyway.
            eps: loaded
                .metadata_f32("hyper_connection.epsilon")
                .unwrap_or(1e-6),
            n_embd: loaded.config.n_embd,
            rms_eps: loaded.config.rms_eps,
        })
    }

    /// Seeds the stream bundle for `tokens`: every stream starts as a copy
    /// of the same token embedding (upstream's `ggml_repeat_4d` over the
    /// hyper-connection axis).
    pub(crate) fn seed(&self, embeddings: &[f32], n_tokens: usize) -> Vec<f32> {
        debug_assert_eq!(embeddings.len(), n_tokens * self.n_embd);
        let mut x = vec![0f32; n_tokens * self.count * self.n_embd];
        for t in 0..n_tokens {
            let embd = &embeddings[t * self.n_embd..(t + 1) * self.n_embd];
            for s in 0..self.count {
                let at = (t * self.count + s) * self.n_embd;
                x[at..at + self.n_embd].copy_from_slice(embd);
            }
        }
        x
    }

    /// The in-mix: predicts this sub-layer's mixing weights from the
    /// streams and collapses them to one vector per token. When `out` is
    /// given it also yields the out-mix (`post`) and stream-combination
    /// (`comb`) weights [`Self::expand`] needs; a final collapse passes
    /// `None`, which is upstream's separate `build_hc_head` (one scale, no
    /// `post`/`comb` at all).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collapse_into(
        &self,
        backend: &dyn Backend,
        cur: &mut Vec<f32>,
        mixer: &HyperConnection,
        x: &[f32],
        n_tokens: usize,
        out: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
        flat: &mut Vec<f32>,
        mixes_buf: &mut Vec<f32>,
    ) {
        let n_embd = self.n_embd;
        let hc = self.count;
        let flat_dim = hc * n_embd;

        // `count` times the hidden state — the largest per-layer copy in
        // these architectures, and it was being made only to be overwritten
        // by the norm that follows.
        rms_norm_rows_into(flat, x, flat_dim, self.rms_eps);
        backend.matmul_into(mixes_buf, flat, n_tokens, &mixer.weights);
        let mixes = &*mixes_buf;
        let mix_dim = mixer.weights.out_dim;

        // Accumulated into, not overwritten — see `tensor::zeroed_to`.
        let cur = tensor::zeroed_to(cur, n_tokens * n_embd);
        let (post_out, comb_out) = match out {
            Some((post, comb)) => {
                post.clear();
                post.resize(n_tokens * hc, 0.0);
                comb.clear();
                comb.resize(n_tokens * hc * hc, 0.0);
                (Some(post), Some(comb))
            }
            None => (None, None),
        };
        let mut post_out = post_out;
        let mut comb_out = comb_out;

        for t in 0..n_tokens {
            let mix = &mixes[t * mix_dim..(t + 1) * mix_dim];
            let pre = match (&mut post_out, &mut comb_out) {
                (Some(post), Some(comb)) => {
                    let (pre, p, c) = hc_pre_parts(
                        mix,
                        hc,
                        self.eps,
                        self.sinkhorn_iters,
                        &mixer.scale,
                        &mixer.base,
                    );
                    post[t * hc..(t + 1) * hc].copy_from_slice(&p);
                    comb[t * hc * hc..(t + 1) * hc * hc].copy_from_slice(&c);
                    pre
                }
                _ => {
                    let mut pre = mix.to_vec();
                    hc_affine(&mut pre, mixer.scale[0], &mixer.base);
                    hc_sigmoid_eps_inplace(&mut pre, self.eps);
                    pre
                }
            };
            let dst = &mut cur[t * n_embd..(t + 1) * n_embd];
            for (s, &w) in pre.iter().enumerate() {
                let src = &x[(t * hc + s) * n_embd..(t * hc + s + 1) * n_embd];
                tensor::axpy_inplace(dst, src, w);
            }
        }
    }

    /// The out-mix: writes this sub-layer's output back into every stream,
    /// each a `post`-weighted copy of the output plus a `comb`-weighted mix
    /// of the streams it came from.
    pub(crate) fn expand(
        &self,
        x: &mut [f32],
        sub_out: &[f32],
        residual: &[f32],
        post: &[f32],
        comb: &[f32],
        n_tokens: usize,
    ) {
        let n_embd = self.n_embd;
        let hc = self.count;
        for t in 0..n_tokens {
            let out_t = &sub_out[t * n_embd..(t + 1) * n_embd];
            for dst in 0..hc {
                let target = &mut x[(t * hc + dst) * n_embd..(t * hc + dst + 1) * n_embd];
                let scale = post[t * hc + dst];
                for (o, &v) in target.iter_mut().zip(out_t.iter()) {
                    *o = v * scale;
                }
                for src in 0..hc {
                    let w = comb[t * hc * hc + src * hc + dst];
                    let from = &residual[(t * hc + src) * n_embd..(t * hc + src + 1) * n_embd];
                    tensor::axpy_inplace(target, from, w);
                }
            }
        }
    }

    /// The final collapse when a file carries no head mixer for it: the
    /// unweighted mean of the streams (upstream's `Glm5NextTextHyperHead`).
    pub(crate) fn mean_into(&self, out: &mut Vec<f32>, x: &[f32], n_tokens: usize) {
        let n_embd = self.n_embd;
        let hc = self.count;
        let out = tensor::zeroed_to(out, n_tokens * n_embd);
        let scale = 1.0 / hc as f32;
        for t in 0..n_tokens {
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            for s in 0..hc {
                let src = &x[(t * hc + s) * n_embd..(t * hc + s + 1) * n_embd];
                tensor::axpy_inplace(dst, src, scale);
            }
        }
    }
}

/// The buffers a hyper-connected forward pass reuses, so the machinery
/// allocates once per pass rather than twice per layer.
///
/// `residual` and `flat` are both the full stream bundle
/// (`n_tokens * count * n_embd`), which is what makes them worth hoisting:
/// on a wide prefill each is several times the hidden state itself.
#[derive(Default)]
pub(crate) struct HyperScratch {
    /// The streams as they were before this sub-layer, which
    /// [`HyperShape::expand`] reads while writing `x`. A real copy — only
    /// its allocation is saved.
    pub residual: Vec<f32>,
    /// Per-token out-mix weights (`n_tokens * count`).
    pub post: Vec<f32>,
    /// Per-token stream-combination weights (`n_tokens * count * count`).
    pub comb: Vec<f32>,
    /// The row-normalized streams the mixer projects from.
    pub flat: Vec<f32>,
    /// The mixer's raw output.
    pub mixes: Vec<f32>,
    /// The collapsed sub-layer input, one per sub-layer so the attention
    /// half's value is not overwritten while still in use.
    pub collapsed_attn: Vec<f32>,
    pub collapsed_ffn: Vec<f32>,
}

pub(crate) fn hc_affine(x: &mut [f32], scale: f32, base: &[f32]) {
    debug_assert_eq!(x.len(), base.len());
    for (v, &b) in x.iter_mut().zip(base.iter()) {
        *v = *v * scale + b;
    }
}

pub(crate) fn hc_sigmoid_eps_inplace(x: &mut [f32], eps: f32) {
    for v in x.iter_mut() {
        *v = tensor::sigmoid(*v) + eps;
    }
}

pub(crate) fn hc_sigmoid_times_two_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = tensor::sigmoid(*v) * 2.0;
    }
}

/// Sinkhorn-normalizes each token's `[count, count]` stream-combination
/// matrix (destination index fastest) into a doubly stochastic one: a
/// softmax over destinations, then alternating column/row normalizations.
/// Upstream's `build_hc_sinkhorn`.
pub(crate) fn hc_sinkhorn_inplace(
    comb: &mut [f32],
    hc: usize,
    n_tokens: usize,
    eps: f32,
    iters: usize,
) {
    debug_assert_eq!(comb.len(), hc * hc * n_tokens);
    for t in 0..n_tokens {
        let block = &mut comb[t * hc * hc..(t + 1) * hc * hc];
        for src in 0..hc {
            let mut max = f32::NEG_INFINITY;
            for dst in 0..hc {
                max = max.max(block[dst + src * hc]);
            }
            let mut sum = 0.0;
            for dst in 0..hc {
                let e = (block[dst + src * hc] - max).exp();
                block[dst + src * hc] = e;
                sum += e;
            }
            for dst in 0..hc {
                block[dst + src * hc] = block[dst + src * hc] / sum.max(f32::MIN_POSITIVE) + eps;
            }
        }

        let normalize_cols = |block: &mut [f32]| {
            for dst in 0..hc {
                let mut sum = eps;
                for src in 0..hc {
                    sum += block[dst + src * hc];
                }
                for src in 0..hc {
                    block[dst + src * hc] /= sum;
                }
            }
        };
        let normalize_rows = |block: &mut [f32]| {
            for src in 0..hc {
                let mut sum = eps;
                for dst in 0..hc {
                    sum += block[dst + src * hc];
                }
                for dst in 0..hc {
                    block[dst + src * hc] /= sum;
                }
            }
        };

        normalize_cols(block);
        for _ in 1..iters {
            normalize_rows(block);
            normalize_cols(block);
        }
    }
}

/// Splits one token's hyper-connection projection into its three parts: the
/// in-mix (`pre`), the out-mix (`post`), and the Sinkhorn-normalized
/// stream-combination matrix (`comb`).
pub(crate) fn hc_pre_parts(
    mixes: &[f32],
    hc: usize,
    eps: f32,
    sinkhorn_iters: usize,
    scale: &[f32],
    base: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mixes.len(), hc * (2 + hc));
    let mut pre = mixes[..hc].to_vec();
    hc_affine(&mut pre, scale[0], &base[..hc]);
    hc_sigmoid_eps_inplace(&mut pre, eps);

    let mut post = mixes[hc..2 * hc].to_vec();
    hc_affine(&mut post, scale[1], &base[hc..2 * hc]);
    hc_sigmoid_times_two_inplace(&mut post);

    let mut comb = mixes[2 * hc..].to_vec();
    hc_affine(&mut comb, scale[2], &base[2 * hc..2 * hc + hc * hc]);
    hc_sinkhorn_inplace(&mut comb, hc, 1, eps, sinkhorn_iters);
    (pre, post, comb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(count: usize, n_embd: usize) -> HyperShape {
        HyperShape {
            count,
            sinkhorn_iters: 3,
            eps: 1e-6,
            n_embd,
            rms_eps: 1e-5,
        }
    }

    #[test]
    fn hc_affine_scales_then_biases() {
        let mut x = vec![1.0, 2.0, 3.0];
        hc_affine(&mut x, 0.5, &[10.0, 20.0, 30.0]);
        assert_eq!(x, vec![10.5, 21.0, 31.5]);
    }

    #[test]
    fn hc_sigmoid_variants_match_reference_formulas() {
        let mut pre = vec![0.0, 1.0];
        hc_sigmoid_eps_inplace(&mut pre, 1e-6);
        assert!((pre[0] - 0.500001).abs() < 1e-6);
        assert!((pre[1] - (tensor::sigmoid(1.0) + 1e-6)).abs() < 1e-6);

        let mut post = vec![0.0, 1.0];
        hc_sigmoid_times_two_inplace(&mut post);
        assert!((post[0] - 1.0).abs() < 1e-6);
        assert!((post[1] - tensor::sigmoid(1.0) * 2.0).abs() < 1e-6);
    }

    #[test]
    fn hc_sinkhorn_produces_positive_column_normalized_weights() {
        let mut comb = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ];
        hc_sinkhorn_inplace(&mut comb, 2, 2, 1e-6, 3);
        for t in 0..2 {
            let block = &comb[t * 4..(t + 1) * 4];
            for dst in 0..2 {
                let col = block[dst] + block[dst + 2];
                assert!((col - 1.0).abs() < 1e-4, "{col}");
            }
            assert!(block.iter().all(|v| *v > 0.0));
        }
    }

    #[test]
    fn hc_pre_parts_split_and_normalize_the_mix_vector() {
        let mixes = vec![
            0.0, 1.0, //
            2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
        ];
        let scale = vec![1.0, 1.0, 1.0];
        let base = vec![0.0; 2 + 2 + 4];
        let (pre, post, comb) = hc_pre_parts(&mixes, 2, 1e-6, 3, &scale, &base);
        assert_eq!(pre.len(), 2);
        assert_eq!(post.len(), 2);
        assert_eq!(comb.len(), 4);
        assert!(pre.iter().all(|v| *v > 0.0));
        assert!(post.iter().all(|v| *v > 0.0));
        let col0 = comb[0] + comb[2];
        let col1 = comb[1] + comb[3];
        assert!((col0 - 1.0).abs() < 1e-4, "{col0}");
        assert!((col1 - 1.0).abs() < 1e-4, "{col1}");
    }

    /// Every stream starts life as the same embedding, which is what makes
    /// the first layer's in-mix a plain (weighted) copy of it rather than a
    /// mixture of four different things.
    #[test]
    fn seed_repeats_the_embedding_into_every_stream() {
        let s = shape(3, 2);
        let x = s.seed(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(
            x,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    /// The mean head is the whole of `glm5`'s final collapse: getting it
    /// wrong by a factor of `count` would rescale every logit.
    #[test]
    fn mean_into_averages_the_streams() {
        let s = shape(2, 2);
        let mut out = Vec::new();
        s.mean_into(&mut out, &[1.0, 3.0, 3.0, 5.0], 1);
        assert_eq!(out, vec![2.0, 4.0]);
    }

    /// With `post = 1` and `comb` the identity, the out-mix degenerates to
    /// exactly the plain residual add it generalizes — one stream at a
    /// time, output plus what was there before.
    #[test]
    fn expand_with_identity_weights_is_a_plain_residual_add() {
        let s = shape(2, 2);
        let residual = vec![1.0, 2.0, 10.0, 20.0];
        let mut x = residual.clone();
        let post = vec![1.0, 1.0];
        // `comb[dst + src * hc]`: the identity in that layout.
        let comb = vec![1.0, 0.0, 0.0, 1.0];
        s.expand(&mut x, &[0.5, 0.5], &residual, &post, &comb, 1);
        assert_eq!(x, vec![1.5, 2.5, 10.5, 20.5]);
    }
}

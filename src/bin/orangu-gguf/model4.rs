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

//! The Qwen 4 (`qwen4exp`) architecture, forward and backward.
//!
//! The residual is `hc` streams wide. Every sub-layer is wrapped by a
//! hyper-connection: an in-mix that reads the streams and hands the
//! sub-layer one vector, and an out-mix that adds the sub-layer's output
//! back into every stream with a weight of its own. A last in-mix, with no
//! out-mix, collapses the streams for the output projection; there is no
//! output norm.
//!
//! The in-mix, per token, over `x` = the `hc` streams of `hidden` values:
//!
//! ```text
//! n     = rmsnorm of each stream, times a gamma spanning all of them
//! low   = silu((n · down) / hc)                  [hc·hidden → rank]
//! gate  = sigmoid(low · up)                      [rank → hc·hidden]
//! out   = mean over streams of (gate ⊙ n)
//! inj   = n · inject                             [hc·hidden → hc]
//! ```
//!
//! and the out-mix adds `2·sigmoid(inj[c] / hc)` times the sub-layer's
//! output to stream `c`: an all-zero `inject` is the plain residual add.
//!
//! Every `interval`-th block mixes tokens with gated full attention: each
//! head's query has its output gate beside it, Q and K are normed per head
//! and rotated over the first `rope_dim` of the head, and the attention
//! output is multiplied by `sigmoid(gate)`. The others use the gated delta
//! rule (see [`Model::delta_forward`]).
//!
//! This is the arithmetic the engine runs (`engine::arch::qwen4exp`,
//! `qwen_hybrid`), so a file trained here is served as it was trained.
//! Everything is `f32` on the CPU, and each operation's backward is written
//! beside its forward and held to finite differences, as the dense block's
//! is ([`crate::model`]). A block's intermediates are kept for its backward
//! while the process's activation budget has room, and recomputed from the
//! block's input otherwise.

use crate::accel;
use crate::aligned::Aligned;
use crate::gemm::View;
use crate::model::{
    self, Rng, attention, attention_backward, grad_slice, matmul, matmul_add_dw, matmul_add_dx,
    silu,
};
use crate::stages::{Stage, time};
use crate::vector::{self, V8};
use rayon::prelude::*;

/// Everything needed to build and train one model.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    /// The routed experts, how many each token uses, and their width.
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ffn: usize,
    /// The shared expert's width.
    pub shared_ffn: usize,
    /// The load-balancing term's weight (the checkpoint's
    /// `router_aux_loss_coef`). A training knob, not part of the model.
    pub aux_coef: f32,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `hyper_connection.count`: the residual's streams.
    pub hc: usize,
    /// `hyper_connection.low_rank`: the in-mix's inner width.
    pub hc_rank: usize,
    pub context: usize,
    pub rope_base: f32,
    /// How many of each head's dimensions are rotated, from its start.
    pub rope_dim: usize,
    pub eps: f32,
    /// `full_attention_interval`: every this-many-th block attends, the
    /// rest run the delta rule.
    pub interval: usize,
    /// The delta rule's key heads (`ssm.group_count`).
    pub ssm_k_heads: usize,
    /// Its value heads (`ssm.time_step_rank`), a multiple of the key heads.
    pub ssm_v_heads: usize,
    /// One key or value head's width (`ssm.state_size`).
    pub ssm_dim: usize,
    /// The short convolution's kernel (`ssm.conv_kernel`).
    pub conv: usize,
    /// Tokens between the delta rule's state checkpoints in the backward.
    /// A training knob, not part of the model.
    pub chunk: usize,
    /// The sparse-attention indexer: its heads, their width, how many
    /// positions a query attends past the incomplete block, and how many
    /// positions one scored block covers (`attention.compress_ratios`).
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    pub ratio: usize,
    /// The indexer's loss weight. A training knob, not part of the model.
    pub idx_coef: f32,
    /// The n-gram hash embedding, on one block, or none.
    pub ple: Option<PleConfig>,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The width of the residual: every stream side by side.
    pub fn wide(&self) -> usize {
        self.hc * self.hidden
    }

    /// The delta rule's query and key width, and its value width.
    pub fn key_dim(&self) -> usize {
        self.ssm_k_heads * self.ssm_dim
    }

    pub fn value_dim(&self) -> usize {
        self.ssm_v_heads * self.ssm_dim
    }

    /// The channels the short convolution runs over: Q, K and V.
    pub fn channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Whether block `l` attends (rather than running the delta rule).
    pub fn is_full(&self, l: usize) -> bool {
        (l + 1).is_multiple_of(self.interval)
    }

    /// The attention kernels' view of the shape.
    fn attention(&self) -> model::Config {
        model::Config {
            vocab: self.vocab,
            hidden: self.hidden,
            ffn: self.expert_ffn,
            layers: self.layers,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            context: self.context,
            rope_base: self.rope_base,
            eps: self.eps,
        }
    }

    pub fn parameters(&self) -> usize {
        Layout::new(self).total
    }
}

/// A named training size: the shape of the network, without the vocabulary
/// and context a run supplies.
#[derive(Debug, Clone, Copy)]
pub struct Size {
    pub key: &'static str,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub hc: usize,
    pub hc_rank: usize,
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ffn: usize,
    pub shared_ffn: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_dim: usize,
    pub idx_top_k: usize,
    /// Peak learning rate, in millionths.
    pub peak_lr_micro: u32,
}

impl Size {
    pub fn peak_lr(&self) -> f32 {
        self.peak_lr_micro as f32 * 1e-6
    }
}

/// The sizes a manifest names. Every `hidden`, expert width and shared
/// width is a multiple of 256, the K-quants' super-block. Each has three
/// delta-rule blocks to one attention block, an indexer on every attention
/// block, and the n-gram hash embedding on block 1.
///
/// `tiny` is the shape a first run on a machine wants. `smoke` is the one a
/// change is measured on: grouped-query attention, value heads that share
/// key heads, more experts than a token takes, and an indexer that selects
/// at the sequence lengths it trains at.
pub const SIZES: &[Size] = &[
    Size {
        key: "tiny",
        hidden: 256,
        layers: 4,
        heads: 4,
        kv_heads: 2,
        head_dim: 64,
        hc: 2,
        hc_rank: 32,
        experts: 8,
        experts_used: 2,
        expert_ffn: 256,
        shared_ffn: 256,
        ssm_k_heads: 2,
        ssm_v_heads: 4,
        ssm_dim: 64,
        idx_top_k: 128,
        peak_lr_micro: 1000,
    },
    Size {
        key: "smoke",
        hidden: 768,
        layers: 8,
        heads: 6,
        kv_heads: 2,
        head_dim: 128,
        hc: 2,
        hc_rank: 64,
        experts: 16,
        experts_used: 4,
        expert_ffn: 512,
        shared_ffn: 512,
        ssm_k_heads: 4,
        ssm_v_heads: 8,
        ssm_dim: 128,
        idx_top_k: 256,
        peak_lr_micro: 700,
    },
    Size {
        key: "1b",
        hidden: 1024,
        layers: 16,
        heads: 8,
        kv_heads: 2,
        head_dim: 128,
        hc: 4,
        hc_rank: 64,
        experts: 32,
        experts_used: 4,
        expert_ffn: 512,
        shared_ffn: 1024,
        ssm_k_heads: 8,
        ssm_v_heads: 16,
        ssm_dim: 128,
        idx_top_k: 2048,
        peak_lr_micro: 300,
    },
];

pub fn size_named(key: &str) -> Option<&'static Size> {
    SIZES.iter().find(|s| s.key.eq_ignore_ascii_case(key))
}

/// The n-gram hash's per-position multipliers: large odd constants, so the
/// fold of one position cannot cancel another's low bits.
const PLE_MULTIPLIERS: [u64; 3] = [
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
];

/// Each hash head's rows: distinct primes, so two heads of one n-gram width
/// collide on different pairs of n-grams.
const PLE_ROWS: [u64; 4] = [8191, 8209, 8219, 8221];

impl Config {
    /// The model of `size` over a vocabulary of `vocab` tokens, `eos` the
    /// one that cuts the n-grams, attending `context` positions.
    pub fn from_size(size: &Size, vocab: usize, eos: u32, context: usize) -> Self {
        Config {
            vocab,
            hidden: size.hidden,
            experts: size.experts,
            experts_used: size.experts_used,
            expert_ffn: size.expert_ffn,
            shared_ffn: size.shared_ffn,
            aux_coef: 0.001,
            layers: size.layers,
            heads: size.heads,
            kv_heads: size.kv_heads,
            head_dim: size.head_dim,
            hc: size.hc,
            hc_rank: size.hc_rank,
            context,
            rope_base: model::rope_base_for(context),
            rope_dim: size.head_dim / 4,
            eps: 1e-6,
            interval: 4,
            ssm_k_heads: size.ssm_k_heads,
            ssm_v_heads: size.ssm_v_heads,
            ssm_dim: size.ssm_dim,
            conv: 4,
            chunk: 64,
            idx_heads: 4,
            idx_dim: 64,
            idx_top_k: size.idx_top_k,
            ratio: 4,
            idx_coef: 0.01,
            ple: Some(PleConfig {
                layer: 1,
                ngram: 3,
                heads_per_ngram: 2,
                kernel: 4,
                multipliers: PLE_MULTIPLIERS.to_vec(),
                sizes: PLE_ROWS.to_vec(),
                eos,
            }),
        }
    }
}

fn put(out: &mut impl std::io::Write, v: u64) -> anyhow::Result<()> {
    out.write_all(&v.to_le_bytes())?;
    Ok(())
}

impl crate::train::Network for Model {
    type Config = Config;
    const KIND: u32 = 2;

    fn config(&self) -> &Config {
        &self.cfg
    }
    fn specs(&self) -> &[TensorSpec] {
        &self.layout.specs
    }
    fn params(&self) -> &[f32] {
        &self.params
    }
    fn parts(&mut self) -> (&mut [f32], &[TensorSpec]) {
        (&mut self.params, &self.layout.specs)
    }
    fn pass(&self, inputs: &[u32], targets: &[u32], grads: Option<&mut [f32]>) -> f32 {
        self.forward_backward(inputs, targets, grads).ce
    }
    fn parameters(cfg: &Config) -> usize {
        cfg.parameters()
    }
    fn assemble(cfg: Config, params: Aligned) -> Self {
        Model {
            layout: Layout::new(&cfg),
            cfg,
            params,
        }
    }
    fn write_config(out: &mut impl std::io::Write, cfg: &Config) -> anyhow::Result<()> {
        for v in [
            cfg.vocab,
            cfg.hidden,
            cfg.experts,
            cfg.experts_used,
            cfg.expert_ffn,
            cfg.shared_ffn,
            cfg.layers,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.hc,
            cfg.hc_rank,
            cfg.context,
            cfg.rope_dim,
            cfg.interval,
            cfg.ssm_k_heads,
            cfg.ssm_v_heads,
            cfg.ssm_dim,
            cfg.conv,
            cfg.chunk,
            cfg.idx_heads,
            cfg.idx_dim,
            cfg.idx_top_k,
            cfg.ratio,
        ] {
            put(out, v as u64)?;
        }
        for v in [cfg.aux_coef, cfg.rope_base, cfg.eps, cfg.idx_coef] {
            out.write_all(&v.to_le_bytes())?;
        }
        match &cfg.ple {
            None => put(out, 0)?,
            Some(p) => {
                put(out, 1)?;
                for v in [
                    p.layer,
                    p.ngram,
                    p.heads_per_ngram,
                    p.kernel,
                    p.eos as usize,
                ] {
                    put(out, v as u64)?;
                }
                for list in [&p.multipliers, &p.sizes] {
                    put(out, list.len() as u64)?;
                    for &v in list.iter() {
                        put(out, v)?;
                    }
                }
            }
        }
        Ok(())
    }
    fn read_config(input: &mut impl std::io::Read) -> anyhow::Result<Config> {
        use crate::train::{read_bytes, read_u64};
        let mut u = [0usize; 24];
        for v in u.iter_mut() {
            *v = read_u64(input)? as usize;
        }
        let mut f = [0f32; 4];
        for v in f.iter_mut() {
            *v = f32::from_le_bytes(read_bytes(input)?);
        }
        let ple = match read_u64(input)? {
            0 => None,
            _ => {
                let mut p = [0usize; 5];
                for v in p.iter_mut() {
                    *v = read_u64(input)? as usize;
                }
                let mut lists = [Vec::new(), Vec::new()];
                for list in lists.iter_mut() {
                    let n = read_u64(input)?;
                    anyhow::ensure!(n <= 1 << 16, "a checkpoint's n-gram hash is corrupt");
                    for _ in 0..n {
                        list.push(read_u64(input)?);
                    }
                }
                let [multipliers, sizes] = lists;
                Some(PleConfig {
                    layer: p[0],
                    ngram: p[1],
                    heads_per_ngram: p[2],
                    kernel: p[3],
                    eos: p[4] as u32,
                    multipliers,
                    sizes,
                })
            }
        };
        Ok(Config {
            vocab: u[0],
            hidden: u[1],
            experts: u[2],
            experts_used: u[3],
            expert_ffn: u[4],
            shared_ffn: u[5],
            layers: u[6],
            heads: u[7],
            kv_heads: u[8],
            head_dim: u[9],
            hc: u[10],
            hc_rank: u[11],
            context: u[12],
            rope_dim: u[13],
            interval: u[14],
            ssm_k_heads: u[15],
            ssm_v_heads: u[16],
            ssm_dim: u[17],
            conv: u[18],
            chunk: u[19],
            idx_heads: u[20],
            idx_dim: u[21],
            idx_top_k: u[22],
            ratio: u[23],
            aux_coef: f[0],
            rope_base: f[1],
            eps: f[2],
            idx_coef: f[3],
            ple,
        })
    }
}

/// One parameter tensor's place in the flat buffer, and its GGUF shape.
pub use model::TensorSpec;

/// A hyper-connection mixer's parameter offsets.
#[derive(Debug, Clone, Copy)]
pub struct MixerOffsets {
    pub norm: usize,
    pub down: usize,
    pub up: usize,
    /// `None` on the final collapse, which writes nothing back.
    pub inject: Option<usize>,
}

/// A full-attention block's token mixer.
#[derive(Debug, Clone, Copy)]
pub struct FullOffsets {
    /// `[Q(head_dim), gate(head_dim)]` per head.
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    /// The indexer's projections, `[idx_heads · idx_dim, hidden]` and
    /// `[idx_dim, hidden]`, and their norms.
    pub idx_q: usize,
    pub idx_k: usize,
    pub idx_q_norm: usize,
    pub idx_k_norm: usize,
}

/// A delta-rule block's token mixer.
#[derive(Debug, Clone, Copy)]
pub struct LinearOffsets {
    /// `[q | k | v]`, `channels` rows.
    pub qkv: usize,
    /// The output gate `z`, `value_dim` rows.
    pub gate: usize,
    pub beta: usize,
    pub alpha: usize,
    /// `[channels, conv]`.
    pub conv: usize,
    /// The decay rate's logarithm per value head; the file carries
    /// `ssm_a = -exp(this)`.
    pub a_log: usize,
    pub dt: usize,
    pub norm: usize,
    pub out: usize,
}

/// How a block mixes tokens.
#[derive(Debug, Clone, Copy)]
pub enum TokenMixer {
    Full(FullOffsets),
    Linear(LinearOffsets),
}

/// One block's parameter offsets.
#[derive(Debug, Clone, Copy)]
pub struct LayerOffsets {
    pub hc_attn: MixerOffsets,
    pub hc_ffn: MixerOffsets,
    pub mixer: TokenMixer,
    pub moe: MoeOffsets,
    /// The n-gram hash embedding's own tensors, on its block.
    pub ple: Option<PleOffsets>,
}

/// The n-gram hash embedding's settings: the block it feeds, the hash, and
/// its convolution. The hash's constants are the model's own, written into
/// the file for the engine to hash with.
#[derive(Debug, Clone, PartialEq)]
pub struct PleConfig {
    pub layer: usize,
    /// The longest n-gram; every width from 2 up to it has its heads.
    pub ngram: usize,
    pub heads_per_ngram: usize,
    /// The convolution's taps, `ngram` positions apart.
    pub kernel: usize,
    /// Per n-gram position, the multiplier it is hashed with.
    pub multipliers: Vec<u64>,
    /// Per head, its rows of the table.
    pub sizes: Vec<u64>,
    /// The token that cuts the n-grams: a predecessor at or before it
    /// reads as this token.
    pub eos: u32,
}

impl PleConfig {
    /// How many table rows each token gathers.
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    /// Where each head's rows start in the one table.
    pub fn offsets(&self) -> Vec<u64> {
        self.sizes
            .iter()
            .scan(0u64, |at, &s| {
                let start = *at;
                *at += s;
                Some(start)
            })
            .collect()
    }

    /// The table's rows.
    pub fn rows(&self) -> usize {
        self.sizes.iter().sum::<u64>() as usize
    }

    /// The table rows each token gathers, `[t, heads]`: per n-gram width
    /// `n`, the token and its `n - 1` predecessors folded into one value
    /// `(t₀·m₀) ^ (t₁·m₁) ^ …`, taken modulo each of that width's heads'
    /// sizes and offset into the head's rows. A predecessor before the
    /// sequence, or at or before an EOS, reads as EOS. The engine's
    /// `NgramHash::rows`.
    pub fn hash(&self, tokens: &[u32]) -> Vec<usize> {
        let (n_gram, eos, heads) = (self.ngram, self.eos, self.heads());
        let offsets = self.offsets();
        let mut rows = vec![0usize; tokens.len() * heads];
        for (i, &token) in tokens.iter().enumerate() {
            let mut ctx = vec![token as u64; n_gram];
            let mut cut = false;
            for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
                let tok = if cut {
                    eos
                } else {
                    i.checked_sub(s).map_or(eos, |j| tokens[j])
                };
                *slot = tok as u64;
                cut |= tok == eos;
            }
            for n in 2..=n_gram {
                let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
                for (c, m) in ctx[1..n].iter().zip(&self.multipliers[1..n]) {
                    mixed ^= c.wrapping_mul(*m);
                }
                for g in 0..self.heads_per_ngram {
                    let head = (n - 2) * self.heads_per_ngram + g;
                    rows[i * heads + head] = (mixed % self.sizes[head] + offsets[head]) as usize;
                }
            }
        }
        rows
    }
}

/// The n-gram hash embedding's tensors on its block.
#[derive(Debug, Clone, Copy)]
pub struct PleOffsets {
    /// `[hc · hidden, hidden]` and `[hidden, hidden]`.
    pub key: usize,
    pub value: usize,
    /// `[hc · hidden]` each.
    pub norm_key: usize,
    pub norm_query: usize,
    pub norm_conv: usize,
    /// `[hc · hidden, kernel]`.
    pub conv: usize,
}

/// A block's routed feed-forward.
#[derive(Debug, Clone, Copy)]
pub struct MoeOffsets {
    /// `[experts, hidden]`.
    pub router: usize,
    /// `[experts, expert_ffn, hidden]` each, `down` `[experts, hidden, expert_ffn]`.
    pub gate_exps: usize,
    pub up_exps: usize,
    pub down_exps: usize,
    /// `[hidden]`: the shared expert's gate.
    pub shared_gate: usize,
    pub gate_shexp: usize,
    pub up_shexp: usize,
    pub down_shexp: usize,
}

/// Where every parameter lives in one flat `f32` buffer.
#[derive(Debug, Clone)]
pub struct Layout {
    pub specs: Vec<TensorSpec>,
    pub tok_embd: usize,
    pub layers: Vec<LayerOffsets>,
    pub head: MixerOffsets,
    /// The n-gram table, `[rows, hidden / heads]`, when there is one.
    pub ple_table: Option<usize>,
    pub output: usize,
    pub total: usize,
}

/// Places tensors one after another in the flat buffer.
struct Builder {
    specs: Vec<TensorSpec>,
    at: usize,
}

impl Builder {
    fn push(&mut self, name: String, dims: Vec<usize>) -> usize {
        let offset = self.at;
        self.at += dims.iter().product::<usize>();
        self.specs.push(TensorSpec {
            name,
            offset,
            dims: dims.into_iter().map(|d| d as u64).collect(),
        });
        offset
    }

    /// A mixer's tensors: `{prefix}_norm`, `_down`, `_up` and, unless it is
    /// the final collapse, `_inject`.
    fn mixer(&mut self, cfg: &Config, prefix: &str, inject: bool) -> MixerOffsets {
        let (w, r) = (cfg.wide(), cfg.hc_rank);
        MixerOffsets {
            norm: self.push(format!("{prefix}_norm.weight"), vec![w]),
            down: self.push(format!("{prefix}_down.weight"), vec![w, r]),
            up: self.push(format!("{prefix}_up.weight"), vec![r, w]),
            inject: inject.then(|| self.push(format!("{prefix}_inject.weight"), vec![w, cfg.hc])),
        }
    }
}

impl Layout {
    pub fn new(cfg: &Config) -> Self {
        let mut b = Builder {
            specs: Vec::new(),
            at: 0,
        };
        let (h, q, kv) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim());
        let (ne, fe, fs) = (cfg.experts, cfg.expert_ffn, cfg.shared_ffn);
        let (ch, vd, nv) = (cfg.channels(), cfg.value_dim(), cfg.ssm_v_heads);
        let tok_embd = b.push("token_embd.weight".into(), vec![h, cfg.vocab]);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let hc_attn = b.mixer(cfg, &format!("blk.{l}.hc_attn"), true);
            let hc_ffn = b.mixer(cfg, &format!("blk.{l}.hc_ffn"), true);
            let mut p = |suffix: &str, dims: Vec<usize>| b.push(format!("blk.{l}.{suffix}"), dims);
            let mixer = if cfg.is_full(l) {
                TokenMixer::Full(FullOffsets {
                    wq: p("attn_q.weight", vec![h, 2 * q]),
                    wk: p("attn_k.weight", vec![h, kv]),
                    wv: p("attn_v.weight", vec![h, kv]),
                    wo: p("attn_output.weight", vec![q, h]),
                    q_norm: p("attn_q_norm.weight", vec![cfg.head_dim]),
                    k_norm: p("attn_k_norm.weight", vec![cfg.head_dim]),
                    idx_q: p(
                        "indexer.q_proj.weight",
                        vec![h, cfg.idx_heads * cfg.idx_dim],
                    ),
                    idx_k: p("indexer.k_proj.weight", vec![h, cfg.idx_dim]),
                    idx_q_norm: p("indexer.q_norm.weight", vec![cfg.idx_dim]),
                    idx_k_norm: p("indexer.k_norm.weight", vec![cfg.idx_dim]),
                })
            } else {
                TokenMixer::Linear(LinearOffsets {
                    qkv: p("attn_qkv.weight", vec![h, ch]),
                    gate: p("attn_gate.weight", vec![h, vd]),
                    beta: p("ssm_beta.weight", vec![h, nv]),
                    alpha: p("ssm_alpha.weight", vec![h, nv]),
                    conv: p("ssm_conv1d.weight", vec![cfg.conv, ch]),
                    a_log: p("ssm_a_log", vec![nv]),
                    dt: p("ssm_dt.bias", vec![nv]),
                    norm: p("ssm_norm.weight", vec![cfg.ssm_dim]),
                    out: p("ssm_out.weight", vec![vd, h]),
                })
            };
            layers.push(LayerOffsets {
                hc_attn,
                hc_ffn,
                mixer,
                moe: MoeOffsets {
                    router: p("ffn_gate_inp.weight", vec![h, ne]),
                    gate_exps: p("ffn_gate_exps.weight", vec![h, fe, ne]),
                    up_exps: p("ffn_up_exps.weight", vec![h, fe, ne]),
                    down_exps: p("ffn_down_exps.weight", vec![fe, h, ne]),
                    shared_gate: p("ffn_gate_inp_shexp.weight", vec![h]),
                    gate_shexp: p("ffn_gate_shexp.weight", vec![h, fs]),
                    up_shexp: p("ffn_up_shexp.weight", vec![h, fs]),
                    down_shexp: p("ffn_down_shexp.weight", vec![fs, h]),
                },
                ple: cfg.ple.as_ref().filter(|p| p.layer == l).map(|p| {
                    let w = cfg.wide();
                    PleOffsets {
                        key: b.push(format!("blk.{l}.ple_key.weight"), vec![h, w]),
                        value: b.push(format!("blk.{l}.ple_value.weight"), vec![h, h]),
                        norm_key: b.push(format!("blk.{l}.ple_norm_key.weight"), vec![w]),
                        norm_query: b.push(format!("blk.{l}.ple_norm_query.weight"), vec![w]),
                        norm_conv: b.push(format!("blk.{l}.ple_norm_conv.weight"), vec![w]),
                        conv: b.push(format!("blk.{l}.ple_conv1d.weight"), vec![p.kernel, w]),
                    }
                }),
            });
        }
        let head = b.mixer(cfg, "output_hc", false);
        let output = b.push("output.weight".into(), vec![h, cfg.vocab]);
        let ple_table = cfg.ple.as_ref().map(|p| {
            b.push(
                "per_layer_token_embd.weight".into(),
                vec![h / p.heads(), p.rows()],
            )
        });
        Layout {
            specs: b.specs,
            tok_embd,
            layers,
            head,
            ple_table,
            output,
            total: b.at,
        }
    }
}

/// Rotary position embedding over the first `rope_dim` of each head,
/// NEOX pairing (`i` against `i + rope_dim/2`), the rest left as they are:
/// the engine's `rope_apply_inplace`. Position is the row. `inverse`
/// negates the sine, which is the rotation's transpose and so its backward.
fn rope_partial(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    base: f32,
    inverse: bool,
) {
    let half = rope_dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| base.powf(-2.0 * i as f32 / rope_dim as f32))
        .collect();
    x.par_chunks_mut(heads * head_dim)
        .enumerate()
        .for_each(|(pos, row)| {
            for (i, &freq) in freqs.iter().enumerate() {
                let (mut sin, cos) = (pos as f32 * freq).sin_cos();
                if inverse {
                    sin = -sin;
                }
                for head in row.chunks_exact_mut(head_dim) {
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cos - b * sin;
                    head[i + half] = a * sin + b * cos;
                }
            }
        });
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Columns of a norm's gain gradient summed by one task.
const GAMMA_BLOCK: usize = 64;

/// An in-mix's intermediates, for its backward.
#[derive(Default)]
struct MixCache {
    /// `[t, wide]` — each stream normed, times the gamma.
    normed: Aligned,
    /// `[t * hc]` — each stream's `1/rms`.
    scales: Aligned,
    /// `[t, rank]` — `(normed · down) / hc`, before the `silu`.
    pre: Aligned,
    /// `[t, rank]`.
    low: Aligned,
    /// `[t, wide]` — `sigmoid(low · up)`.
    gate: Aligned,
    /// `[t, hc]` — the out-mix's raw weights; empty without an inject.
    inject: Aligned,
}

/// Full attention's intermediates.
#[derive(Default)]
struct AttnCache {
    q_pre: Aligned,
    k_pre: Aligned,
    q_rms: Aligned,
    k_rms: Aligned,
    q_rope: Aligned,
    k_rope: Aligned,
    v: Aligned,
    ctx: Aligned,
    /// `[t, q]` — each head's output gate, before its sigmoid.
    q_gate: Aligned,
    /// `[t, q]` — the attention output, gated.
    gated: Aligned,
    /// Per query, the positions it attends, when the indexer chose them;
    /// `None` when every visible position is attended.
    selection: Option<Vec<Vec<usize>>>,
    /// The indexer's loss for this block.
    idx_loss: f32,
    /// Per query and head, its probabilities over its selection.
    sel_probs: Vec<Vec<f32>>,
    indexer: IndexerCache,
}

/// The delta rule's intermediates. The recurrent state itself is kept only
/// every `chunk` tokens; the backward rebuilds it in between.
#[derive(Default)]
struct DeltaCache {
    /// `[t, channels]` — the projection the convolution reads.
    mixed: Aligned,
    /// `[t, channels]` — the convolution's output, before its `silu`.
    conv: Aligned,
    /// `[t, k_heads]` each — the query and key norms, before the floor.
    q_len: Aligned,
    k_len: Aligned,
    /// `[t, key_dim]` — the queries (normalized and scaled) and keys
    /// (normalized) the recurrence reads; `[t, value_dim]` the values.
    q: Aligned,
    k: Aligned,
    v: Aligned,
    /// `[t, v_heads]`.
    beta: Aligned,
    /// `[t, v_heads]` — `alpha + dt` and the decay it gives.
    g: Aligned,
    decay: Aligned,
    /// `[t, value_dim]` — the recurrence's output, its norm's scales, the
    /// normed output, and the output gate before its sigmoid.
    o: Aligned,
    o_rms: Aligned,
    o_normed: Aligned,
    z: Aligned,
    /// `[t, value_dim]` — `o_normed ⊙ sigmoid(z)`, what `ssm_out` reads.
    y: Aligned,
    /// The state before every `chunk`-th token, `[v_heads, dim, dim]` each.
    checkpoints: Vec<Aligned>,
}

/// A block's token mixer's intermediates.
enum MixerCache {
    Full(AttnCache),
    Linear(DeltaCache),
}

/// One block's intermediates.
struct LayerCache {
    /// The n-gram embedding, on its block.
    ple: Option<PleCache>,
    attn_mix: MixCache,
    /// `[t, hidden]` — what the token mixer reads, and what it gives back.
    attn_in: Aligned,
    attn_out: Aligned,
    mixer: MixerCache,
    /// `[t, wide]` — the streams after the token mixer's out-mix.
    mid: Aligned,
    ffn_mix: MixCache,
    ffn_in: Aligned,
    moe: MoeCache,
    ffn_out: Aligned,
    out: Aligned,
}

/// Bytes a cache holds on the heap.
trait Held {
    fn held(&self) -> usize;
}

impl Held for Aligned {
    fn held(&self) -> usize {
        self.len() * 4
    }
}

impl Held for Vec<f32> {
    fn held(&self) -> usize {
        self.capacity() * 4
    }
}

impl Held for Vec<usize> {
    fn held(&self) -> usize {
        self.capacity() * std::mem::size_of::<usize>()
    }
}

impl<T: Copy> Held for Vec<Vec<T>> {
    fn held(&self) -> usize {
        self.capacity() * std::mem::size_of::<Vec<T>>()
            + self
                .iter()
                .map(|v| v.capacity() * std::mem::size_of::<T>())
                .sum::<usize>()
    }
}

impl<T: Held> Held for Option<T> {
    fn held(&self) -> usize {
        self.as_ref().map_or(0, Held::held)
    }
}

impl Held for MixCache {
    fn held(&self) -> usize {
        [
            &self.normed,
            &self.scales,
            &self.pre,
            &self.low,
            &self.gate,
            &self.inject,
        ]
        .iter()
        .map(|a| a.held())
        .sum()
    }
}

impl Held for AttnCache {
    fn held(&self) -> usize {
        [
            &self.q_pre,
            &self.k_pre,
            &self.q_rms,
            &self.k_rms,
            &self.q_rope,
            &self.k_rope,
            &self.v,
            &self.ctx,
            &self.q_gate,
            &self.gated,
        ]
        .iter()
        .map(|a| a.held())
        .sum::<usize>()
            + self.selection.held()
            + self.sel_probs.held()
            + self.indexer.held()
    }
}

impl Held for DeltaCache {
    fn held(&self) -> usize {
        [
            &self.mixed,
            &self.conv,
            &self.q_len,
            &self.k_len,
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            &self.decay,
            &self.o,
            &self.o_rms,
            &self.o_normed,
            &self.z,
            &self.y,
        ]
        .iter()
        .map(|a| a.held())
        .sum::<usize>()
            + self.checkpoints.iter().map(Held::held).sum::<usize>()
    }
}

impl Held for IndexerCache {
    fn held(&self) -> usize {
        [
            &self.pooled,
            &self.pooled_rms,
            &self.blocks,
            &self.q_raw,
            &self.q_rms,
            &self.queries,
        ]
        .iter()
        .map(|a| a.held())
        .sum::<usize>()
            + self.pred.held()
            + self.target.held()
    }
}

impl Held for PleCache {
    fn held(&self) -> usize {
        [
            &self.embd,
            &self.key,
            &self.value,
            &self.key_n,
            &self.key_s,
            &self.query,
            &self.query_s,
            &self.gated,
            &self.normed,
            &self.normed_s,
            &self.conv,
            &self.out,
        ]
        .iter()
        .map(|a| a.held())
        .sum::<usize>()
            + self.rows.held()
            + self.score.held()
    }
}

impl Held for MoeCache {
    fn held(&self) -> usize {
        [
            &self.probs,
            &self.s_gate,
            &self.s_up,
            &self.s_act,
            &self.s_out,
        ]
        .iter()
        .map(|a| a.held())
        .sum::<usize>()
            + self.chosen.held()
            + self.sums.held()
            + self.members.held()
            + self
                .experts
                .iter()
                .map(|e| e.xin.held() + e.gate.held() + e.up.held() + e.act.held() + e.out.held())
                .sum::<usize>()
            + self.s_score.held()
            + self.share.held()
    }
}

impl Held for LayerCache {
    fn held(&self) -> usize {
        self.ple.held()
            + self.attn_mix.held()
            + self.attn_in.held()
            + self.attn_out.held()
            + match &self.mixer {
                MixerCache::Full(c) => c.held(),
                MixerCache::Linear(c) => c.held(),
            }
            + self.mid.held()
            + self.ffn_mix.held()
            + self.ffn_in.held()
            + self.moe.held()
            + self.ffn_out.held()
            + self.out.held()
    }
}

/// Bytes of block intermediates every pass in the process holds for its
/// backward, against [`model::activation_budget`].
static KEPT_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A block's intermediates, counted against the budget until dropped.
struct Kept {
    cache: LayerCache,
    bytes: usize,
}

impl Kept {
    /// `cache` kept when the budget has room for it.
    fn new(cache: LayerCache) -> Option<Kept> {
        use std::sync::atomic::Ordering::Relaxed;
        let bytes = cache.held();
        let limit = model::activation_budget();
        KEPT_BYTES
            .try_update(Relaxed, Relaxed, |k| {
                (k + bytes <= limit).then_some(k + bytes)
            })
            .ok()?;
        Some(Kept { cache, bytes })
    }
}

impl Drop for Kept {
    fn drop(&mut self) {
        KEPT_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Weights one generator draws at initialization.
const INIT_CHUNK: usize = 1 << 16;

/// A model: its shape, its layout, and one flat parameter buffer.
pub struct Model {
    pub cfg: Config,
    pub layout: Layout,
    pub params: Aligned,
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// `softplus`'s derivative, with the same cut-over.
fn softplus_grad(x: f32) -> f32 {
    if x > 20.0 { 1.0 } else { sigmoid(x) }
}

/// `x / max(|x|, eps)` and its length, in place.
fn l2_normalize(x: &mut [f32], eps: f32) -> f32 {
    let len = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let n = len.max(eps);
    for v in x.iter_mut() {
        *v /= n;
    }
    len
}

/// [`l2_normalize`]'s backward: the gradient of `x` from that of its
/// normalized `y` and `x`'s length.
fn l2_normalize_backward(dy: &[f32], y: &[f32], len: f32, eps: f32, dx: &mut [f32]) {
    if len > eps {
        let dot: f32 = dy.iter().zip(y).map(|(a, b)| a * b).sum();
        for ((d, &g), &v) in dx.iter_mut().zip(dy).zip(y) {
            *d += (g - v * dot) / len;
        }
    } else {
        for (d, &g) in dx.iter_mut().zip(dy) {
            *d += g / eps;
        }
    }
}

impl Model {
    /// Random initial weights: Normal(0, 0.02) for the projections, the
    /// two that write into the streams scaled by `1/sqrt(2 * layers)`; the
    /// norms at one; the out-mix weights at zero, so every block starts as
    /// a plain residual add into every stream. The delta rule's decay rates
    /// start between 1 and 16 (`A_log = ln A`), its time steps between
    /// 0.001 and 0.1 (through the inverse of `softplus`), and its
    /// convolution at the scale of its kernel.
    pub fn new(cfg: Config, seed: u64) -> Self {
        let layout = Layout::new(&cfg);
        let mut params = Aligned::zeros(layout.total);
        let mut rng = Rng::new(seed);
        let residual = 1.0 / ((2 * cfg.layers) as f32).sqrt();
        for spec in &layout.specs {
            let slice = &mut params[spec.offset..spec.offset + spec.len()];
            let name = spec.name.as_str();
            if name.ends_with("ssm_a_log") {
                for v in slice.iter_mut() {
                    *v = (1.0 + 15.0 * rng.next_f32()).ln();
                }
                continue;
            }
            if name.ends_with("ssm_dt.bias") {
                for v in slice.iter_mut() {
                    let dt = (0.001f32.ln() + (0.1f32.ln() - 0.001f32.ln()) * rng.next_f32()).exp();
                    // softplus⁻¹(dt).
                    *v = dt + (-(-dt).exp_m1()).ln();
                }
                continue;
            }
            if name.ends_with("ffn_gate_inp_shexp.weight") {
                for v in slice.iter_mut() {
                    *v = rng.normal() * 0.02;
                }
                continue;
            }
            if spec.dims.len() == 1 {
                slice.fill(1.0);
                continue;
            }
            if name.ends_with("_inject.weight") {
                continue;
            }
            let std = if name.ends_with("ssm_conv1d.weight") {
                1.0 / (cfg.conv as f32).sqrt()
            } else if name.ends_with("attn_output.weight")
                || name.ends_with("ssm_out.weight")
                || name.ends_with("ffn_down_exps.weight")
                || name.ends_with("ffn_down_shexp.weight")
            {
                0.02 * residual
            } else {
                0.02
            };
            // In chunks side by side, each drawn from its own generator
            // seeded from the run's seed, the tensor and the chunk: the same
            // weights however many threads draw them.
            let tensor = rng.next_u64();
            slice
                .par_chunks_mut(INIT_CHUNK)
                .enumerate()
                .for_each(|(chunk, values)| {
                    let mut rng =
                        Rng::new(tensor ^ (chunk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    for v in values {
                        *v = rng.normal() * std;
                    }
                });
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

    /// The in-mix of `m` over `x` (`[t, wide]`): the one vector the
    /// sub-layer reads, and the intermediates.
    fn mix(&self, m: MixerOffsets, x: &[f32], t: usize) -> (Aligned, MixCache) {
        let cfg = &self.cfg;
        let (h, hc, w, r) = (cfg.hidden, cfg.hc, cfg.wide(), cfg.hc_rank);
        let gamma = self.tensor(m.norm, w);
        let mut normed = Aligned::zeros(t * w);
        let mut scales = Aligned::zeros(t * hc);
        normed
            .par_chunks_mut(h)
            .zip(scales.par_iter_mut())
            .zip(x.par_chunks(h))
            .enumerate()
            .for_each(|(row, ((out, scale), xs))| {
                let c = row % hc;
                let mean = vector::sum([xs], |[v]| v * v) / h as f32;
                let s = 1.0 / (mean + cfg.eps).sqrt();
                *scale = s;
                let (g, s) = (&gamma[c * h..(c + 1) * h], V8::splat(s));
                vector::map(out, [xs, g], |[v, g]| v * s * g);
            });
        let mut pre = Aligned::zeros(t * r);
        matmul(&mut pre, &normed, self.tensor(m.down, r * w), t, w, r);
        let inv = 1.0 / hc as f32;
        let iv = V8::splat(inv);
        let mut low = Aligned::zeros(t * r);
        pre.par_chunks_mut(r)
            .zip(low.par_chunks_mut(r))
            .for_each(|(pre, low)| {
                vector::update(pre, [], |p, []| p * iv);
                vector::map(low, [pre], |[p]| p * p.sigmoid());
            });
        let mut gate = Aligned::zeros(t * w);
        matmul(&mut gate, &low, self.tensor(m.up, w * r), t, r, w);
        gate.par_chunks_mut(w)
            .for_each(|row| vector::update(row, [], |v, []| v.sigmoid()));
        let mut out = Aligned::zeros(t * h);
        let (gates, norms): (&[f32], &[f32]) = (&gate, &normed);
        out.par_chunks_mut(h).enumerate().for_each(|(i, dst)| {
            for c in 0..hc {
                let at = i * w + c * h;
                vector::update(
                    dst,
                    [&gates[at..at + h], &norms[at..at + h]],
                    |d, [g, n]| (g * n).mul_add(iv, d),
                );
            }
        });
        let mut inject = Aligned::zeros(0);
        if let Some(inj) = m.inject {
            inject = Aligned::zeros(t * hc);
            matmul(&mut inject, &normed, self.tensor(inj, hc * w), t, w, hc);
        }
        (
            out,
            MixCache {
                normed,
                scales,
                pre,
                low,
                gate,
                inject,
            },
        )
    }

    /// The in-mix's backward: adds into `dx` (`[t, wide]`) and the mixer's
    /// parameter gradients, from the gradient of its output and (with an
    /// inject) of its out-mix weights.
    #[allow(clippy::too_many_arguments)]
    fn mix_backward(
        &self,
        m: MixerOffsets,
        x: &[f32],
        c: &MixCache,
        d_out: &[f32],
        d_inject: Option<&[f32]>,
        dx: &mut [f32],
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, hc, w, r) = (cfg.hidden, cfg.hc, cfg.wide(), cfg.hc_rank);
        let iv = V8::splat(1.0 / hc as f32);
        let one = V8::splat(1.0);
        // out = mean over streams of gate ⊙ normed.
        let mut d_normed = Aligned::zeros(t * w);
        let mut d_gate_pre = Aligned::zeros(t * w);
        let (gates, norms): (&[f32], &[f32]) = (&c.gate, &c.normed);
        d_normed
            .par_chunks_mut(w)
            .zip(d_gate_pre.par_chunks_mut(w))
            .enumerate()
            .for_each(|(i, (dn, dg))| {
                let dy = &d_out[i * h..(i + 1) * h];
                for s in 0..hc {
                    let row = i * w + s * h..i * w + (s + 1) * h;
                    let (g, n) = (&gates[row.clone()], &norms[row]);
                    let span = s * h..(s + 1) * h;
                    vector::map(&mut dn[span.clone()], [dy, g], |[y, g]| y * iv * g);
                    // Through the sigmoid.
                    vector::map(&mut dg[span], [dy, g, n], |[y, g, n]| {
                        y * iv * n * g * (one - g)
                    });
                }
            });
        // gate = sigmoid(low · up).
        matmul_add_dw(grad_slice(grads, m.up, w * r), &d_gate_pre, &c.low, t, r, w);
        let mut d_low = Aligned::zeros(t * r);
        matmul_add_dx(&mut d_low, &d_gate_pre, self.tensor(m.up, w * r), t, r, w);
        // low = silu(pre), pre = (normed · down) / hc.
        let mut d_pre = Aligned::zeros(t * r);
        d_pre
            .par_chunks_mut(r)
            .zip(d_low.par_chunks(r).zip(c.pre.par_chunks(r)))
            .for_each(|(d_pre, (d_low, pre))| {
                vector::map(d_pre, [d_low, pre], |[dl, p]| {
                    let s = p.sigmoid();
                    dl * s * (one + p * (one - s)) * iv
                });
            });
        matmul_add_dw(grad_slice(grads, m.down, r * w), &d_pre, &c.normed, t, w, r);
        matmul_add_dx(&mut d_normed, &d_pre, self.tensor(m.down, r * w), t, w, r);
        if let (Some(inj), Some(d_inj)) = (m.inject, d_inject) {
            matmul_add_dw(grad_slice(grads, inj, hc * w), d_inj, &c.normed, t, w, hc);
            matmul_add_dx(&mut d_normed, d_inj, self.tensor(inj, hc * w), t, w, hc);
        }
        // The grouped norm: each stream normed alone, one gamma across all.
        let gamma = self.tensor(m.norm, w);
        dx.par_chunks_mut(w)
            .zip(d_normed.par_chunks(w))
            .zip(x.par_chunks(w))
            .zip(c.scales.par_chunks(hc))
            .for_each(|(((dx_row, dn_row), x_row), scales)| {
                for (s, &sc) in scales.iter().enumerate() {
                    let range = s * h..(s + 1) * h;
                    let (xs, dn, g) = (
                        &x_row[range.clone()],
                        &dn_row[range.clone()],
                        &gamma[range.clone()],
                    );
                    let inner = vector::sum([g, dn, xs], |[g, d, x]| g * d * x);
                    let coeff = V8::splat(sc * sc * sc * inner / h as f32);
                    let sc = V8::splat(sc);
                    vector::update(&mut dx_row[range], [g, dn, xs], |o, [g, d, x]| {
                        o + (sc * g * d - coeff * x)
                    });
                }
            });
        // Each column's sum over the tokens in order, blocks of columns
        // side by side, each block split where the streams meet.
        let dgamma = grad_slice(grads, m.norm, w);
        let (dn_all, scales) = (&d_normed[..], &c.scales[..]);
        dgamma
            .par_chunks_mut(GAMMA_BLOCK)
            .enumerate()
            .for_each(|(block, out)| {
                let col = block * GAMMA_BLOCK;
                let mut a = 0;
                while a < out.len() {
                    let s = (col + a) / h;
                    let b = ((s + 1) * h - col).min(out.len());
                    for i in 0..t {
                        let sc = V8::splat(scales[i * hc + s]);
                        let at = i * w + col;
                        vector::update(
                            &mut out[a..b],
                            [&dn_all[at + a..at + b], &x[at + a..at + b]],
                            |o, [d, x]| o + d * x * sc,
                        );
                    }
                    a = b;
                }
            });
    }

    /// The out-mix: `sub` (`[t, hidden]`) added into every stream of `x`
    /// with weight `2·sigmoid(inject / hc)`.
    fn combine(&self, x: &mut [f32], sub: &[f32], inject: &[f32], t: usize) {
        let (h, hc) = (self.cfg.hidden, self.cfg.hc);
        x.par_chunks_mut(self.cfg.wide())
            .enumerate()
            .for_each(|(i, row)| {
                let sub = &sub[i * h..(i + 1) * h];
                for s in 0..hc {
                    let wgt = V8::splat(2.0 * sigmoid(inject[i * hc + s] / hc as f32));
                    vector::update(&mut row[s * h..(s + 1) * h], [sub], |o, [v]| {
                        wgt.mul_add(v, o)
                    });
                }
            });
        let _ = t;
    }

    /// The out-mix's backward: the gradients of the sub-layer's output and
    /// of the raw out-mix weights. The streams' own gradient passes through
    /// unchanged.
    fn combine_backward(
        &self,
        dx: &[f32],
        sub: &[f32],
        inject: &[f32],
        t: usize,
    ) -> (Aligned, Aligned) {
        let (h, hc, w) = (self.cfg.hidden, self.cfg.hc, self.cfg.wide());
        let mut d_sub = Aligned::zeros(t * h);
        let mut d_inject = Aligned::zeros(t * hc);
        d_sub
            .par_chunks_mut(h)
            .zip(d_inject.par_chunks_mut(hc))
            .enumerate()
            .for_each(|(i, (ds, di))| {
                let sub = &sub[i * h..(i + 1) * h];
                for s in 0..hc {
                    let sg = sigmoid(inject[i * hc + s] / hc as f32);
                    let wgt = V8::splat(2.0 * sg);
                    let row = &dx[i * w + s * h..i * w + (s + 1) * h];
                    vector::update(ds, [row], |o, [v]| wgt.mul_add(v, o));
                    let dot = vector::sum([row, sub], |[v, u]| v * u);
                    di[s] = dot * 2.0 * sg * (1.0 - sg) / hc as f32;
                }
            });
        (d_sub, d_inject)
    }

    /// Gated full attention over `x` (`[t, hidden]`).
    fn attn_forward(&self, o: FullOffsets, x: &[f32], t: usize) -> (Aligned, AttnCache) {
        let cfg = &self.cfg;
        let (h, qd, kvd, hd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);
        let shape = cfg.attention();
        // `attn_q` gives each head its query and, beside it, its output
        // gate: `[Q(head_dim), gate(head_dim)]` per head.
        let mut joint = Aligned::zeros(t * 2 * qd);
        let mut k = Aligned::zeros(t * kvd);
        let mut v = Aligned::zeros(t * kvd);
        matmul(&mut joint, x, self.tensor(o.wq, 2 * qd * h), t, h, 2 * qd);
        matmul(&mut k, x, self.tensor(o.wk, kvd * h), t, h, kvd);
        matmul(&mut v, x, self.tensor(o.wv, kvd * h), t, h, kvd);
        let mut q = Aligned::zeros(t * qd);
        let mut q_gate = Aligned::zeros(t * qd);
        for (pair, (qh, gh)) in joint
            .chunks(2 * hd)
            .zip(q.chunks_mut(hd).zip(q_gate.chunks_mut(hd)))
        {
            qh.copy_from_slice(&pair[..hd]);
            gh.copy_from_slice(&pair[hd..]);
        }
        let (mut q_rope, q_rms) =
            model::rmsnorm_forward(&q, self.tensor(o.q_norm, hd), t * cfg.heads, hd, cfg.eps);
        let (mut k_rope, k_rms) =
            model::rmsnorm_forward(&k, self.tensor(o.k_norm, hd), t * cfg.kv_heads, hd, cfg.eps);
        rope_partial(
            &mut q_rope,
            cfg.heads,
            hd,
            cfg.rope_dim,
            cfg.rope_base,
            false,
        );
        rope_partial(
            &mut k_rope,
            cfg.kv_heads,
            hd,
            cfg.rope_dim,
            cfg.rope_base,
            false,
        );
        let (selection, idx_loss, indexer) = time(Stage::Indexer, || {
            self.indexer_forward(o, x, &q_rope, &k_rope, t)
        });
        let (ctx, sel_probs) = match &selection {
            None => (attention(&q_rope, &k_rope, &v, &shape, t), Vec::new()),
            Some(sel) => {
                attention_selected(&q_rope, &k_rope, &v, sel, (cfg.heads, cfg.kv_heads, hd))
            }
        };
        let mut gated = Aligned::zeros(t * qd);
        for ((g, &c), &z) in gated.iter_mut().zip(ctx.iter()).zip(q_gate.iter()) {
            *g = c * sigmoid(z);
        }
        let mut out = Aligned::zeros(t * h);
        matmul(&mut out, &gated, self.tensor(o.wo, h * qd), t, qd, h);
        (
            out,
            AttnCache {
                q_pre: q,
                k_pre: k,
                q_rms,
                k_rms,
                q_rope,
                k_rope,
                v,
                ctx,
                q_gate,
                gated,
                selection,
                idx_loss,
                sel_probs,
                indexer,
            },
        )
    }

    /// [`Self::attn_forward`]'s backward: adds the gradient of its input
    /// into `dx`.
    #[allow(clippy::too_many_arguments)]
    fn attn_backward(
        &self,
        o: FullOffsets,
        x: &[f32],
        c: &AttnCache,
        d_out: &[f32],
        dx: &mut [f32],
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, qd, kvd, hd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);
        let shape = cfg.attention();
        matmul_add_dw(grad_slice(grads, o.wo, h * qd), d_out, &c.gated, t, qd, h);
        let mut d_gated = Aligned::zeros(t * qd);
        matmul_add_dx(&mut d_gated, d_out, self.tensor(o.wo, h * qd), t, qd, h);
        let mut d_ctx = Aligned::zeros(t * qd);
        let mut d_q_gate = Aligned::zeros(t * qd);
        for i in 0..t * qd {
            let s = sigmoid(c.q_gate[i]);
            d_ctx[i] = d_gated[i] * s;
            d_q_gate[i] = d_gated[i] * c.ctx[i] * s * (1.0 - s);
        }
        let (mut d_q, mut d_k, d_v) = match &c.selection {
            None => attention_backward(&c.q_rope, &c.k_rope, &c.v, &d_ctx, &shape, t),
            Some(sel) => attention_selected_backward(
                &c.q_rope,
                &c.k_rope,
                &c.v,
                sel,
                &c.sel_probs,
                &d_ctx,
                (cfg.heads, cfg.kv_heads, hd),
            ),
        };
        time(Stage::IndexerBackward, || {
            self.indexer_backward(o, x, &c.indexer, grads, t)
        });
        rope_partial(&mut d_q, cfg.heads, hd, cfg.rope_dim, cfg.rope_base, true);
        rope_partial(
            &mut d_k,
            cfg.kv_heads,
            hd,
            cfg.rope_dim,
            cfg.rope_base,
            true,
        );
        let mut d_q_pre = Aligned::zeros(t * qd);
        model::rmsnorm_backward(
            &mut d_q_pre,
            grad_slice(grads, o.q_norm, hd),
            &d_q,
            &c.q_pre,
            self.tensor(o.q_norm, hd),
            &c.q_rms,
            t * cfg.heads,
            hd,
        );
        let mut d_k_pre = Aligned::zeros(t * kvd);
        model::rmsnorm_backward(
            &mut d_k_pre,
            grad_slice(grads, o.k_norm, hd),
            &d_k,
            &c.k_pre,
            self.tensor(o.k_norm, hd),
            &c.k_rms,
            t * cfg.kv_heads,
            hd,
        );
        let mut d_joint = Aligned::zeros(t * 2 * qd);
        for (pair, (dq, dg)) in d_joint
            .chunks_mut(2 * hd)
            .zip(d_q_pre.chunks(hd).zip(d_q_gate.chunks(hd)))
        {
            pair[..hd].copy_from_slice(dq);
            pair[hd..].copy_from_slice(dg);
        }
        for (w_at, dy, n) in [
            (o.wq, &d_joint, 2 * qd),
            (o.wk, &d_k_pre, kvd),
            (o.wv, &d_v, kvd),
        ] {
            matmul_add_dw(grad_slice(grads, w_at, n * h), dy, x, t, h, n);
            matmul_add_dx(dx, dy, self.tensor(w_at, n * h), t, h, n);
        }
    }

    /// The sparse-attention indexer over `x` (`[t, hidden]`): which
    /// positions each query attends (`None` while every visible position
    /// fits the budget), and its loss.
    ///
    /// A block's key is the mean of its `ratio` positions' raw keys, normed
    /// and rotated at the block's first position; a query is normed per
    /// head and rotated at its own. A block's score is the sum over heads of
    /// `relu(query · key)`. The loss is the KL divergence from the full
    /// attention's own mass on each complete block (`q_rope`, `k_rope`:
    /// every head's probabilities, averaged over heads, summed over the
    /// block, renormalized) to the softmax of the scores, averaged over the
    /// queries that see a complete block. Its gradient reaches the
    /// indexer's own weights only: the attention is its target, not its
    /// student.
    fn indexer_forward(
        &self,
        o: FullOffsets,
        x: &[f32],
        q_rope: &[f32],
        k_rope: &[f32],
        t: usize,
    ) -> (Option<Vec<Vec<usize>>>, f32, IndexerCache) {
        let cfg = &self.cfg;
        let (h, d, nh, r) = (cfg.hidden, cfg.idx_dim, cfg.idx_heads, cfg.ratio);
        let mut keys = Aligned::zeros(t * d);
        matmul(&mut keys, x, self.tensor(o.idx_k, d * h), t, h, d);
        let n_blocks = t / r;
        let mut pooled = Aligned::zeros(n_blocks * d);
        for b in 0..n_blocks {
            for p in b * r..(b + 1) * r {
                for j in 0..d {
                    pooled[b * d + j] += keys[p * d + j] / r as f32;
                }
            }
        }
        let (mut blocks, pooled_rms) =
            model::rmsnorm_forward(&pooled, self.tensor(o.idx_k_norm, d), n_blocks, d, cfg.eps);
        for b in 0..n_blocks {
            rope_at(
                &mut blocks[b * d..(b + 1) * d],
                d,
                cfg.rope_dim,
                cfg.rope_base,
                b * r,
                false,
            );
        }
        let mut q_raw = Aligned::zeros(t * nh * d);
        matmul(
            &mut q_raw,
            x,
            self.tensor(o.idx_q, nh * d * h),
            t,
            h,
            nh * d,
        );
        let (mut queries, q_rms) =
            model::rmsnorm_forward(&q_raw, self.tensor(o.idx_q_norm, d), t * nh, d, cfg.eps);
        for i in 0..t {
            rope_at(
                &mut queries[i * nh * d..(i + 1) * nh * d],
                d,
                cfg.rope_dim,
                cfg.rope_base,
                i,
                false,
            );
        }
        let score = |i: usize, b: usize| -> f32 {
            (0..nh)
                .map(|hh| {
                    let qv = &queries[i * nh * d + hh * d..i * nh * d + (hh + 1) * d];
                    let kv = &blocks[b * d..(b + 1) * d];
                    qv.iter().zip(kv).map(|(a, c)| a * c).sum::<f32>().max(0.0)
                })
                .sum()
        };

        // The selection, once a query sees more than the budget.
        let width = cfg.idx_top_k + r - 1;
        let selection = (t > width).then(|| {
            (0..t)
                .into_par_iter()
                .map(|i| {
                    let visible = i + 1;
                    let scores: Vec<f32> = (0..visible / r).map(|b| score(i, b)).collect();
                    select_positions(&scores, visible, r, width)
                })
                .collect::<Vec<_>>()
        });

        // The loss, against the full attention's mass per block. Every query
        // on its own; the loss's terms are then added in query order.
        let (heads, kv_heads, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let (qd, kvd, group) = (heads * hd, kv_heads * hd, heads / kv_heads);
        let scale = 1.0 / (hd as f32).sqrt();
        let per_query: Vec<Option<QueryLoss>> = (0..t)
            .into_par_iter()
            .map(|i| {
                let scored = (i + 1) / r;
                if scored == 0 {
                    return None;
                }
                let mut mass = vec![0.0f32; scored];
                for hh in 0..heads {
                    let kh = hh / group;
                    let qv = &q_rope[i * qd + hh * hd..i * qd + (hh + 1) * hd];
                    let logits: Vec<f32> = (0..=i)
                        .map(|j| {
                            qv.iter()
                                .zip(&k_rope[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    for (b, m) in mass.iter_mut().enumerate() {
                        for e in &exps[b * r..(b + 1) * r] {
                            *m += e / sum / heads as f32;
                        }
                    }
                }
                let mass_sum: f32 = mass.iter().sum();
                let tgt: Vec<f32> = mass
                    .iter()
                    .map(|m| m / mass_sum.max(f32::MIN_POSITIVE))
                    .collect();
                let s: Vec<f32> = (0..scored).map(|b| score(i, b)).collect();
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = s.iter().map(|v| (v - max).exp()).sum();
                let p: Vec<f32> = s.iter().map(|v| (v - max).exp() / sum).collect();
                let terms: Vec<f64> = tgt
                    .iter()
                    .zip(&p)
                    .filter(|&(&a, _)| a > 0.0)
                    .map(|(&a, &b)| (a * (a.ln() - b.max(f32::MIN_POSITIVE).ln())) as f64)
                    .collect();
                Some((p, tgt, terms))
            })
            .collect();
        let mut pred = vec![Vec::new(); t];
        let mut target = vec![Vec::new(); t];
        let mut total = 0.0f64;
        let mut counted = 0usize;
        for (i, query) in per_query.into_iter().enumerate() {
            let Some((p, tgt, terms)) = query else {
                continue;
            };
            for term in terms {
                total += term;
            }
            pred[i] = p;
            target[i] = tgt;
            counted += 1;
        }
        let loss = if counted == 0 {
            0.0
        } else {
            cfg.idx_coef * (total / counted as f64) as f32
        };
        (
            selection,
            loss,
            IndexerCache {
                pooled,
                pooled_rms,
                blocks,
                q_raw,
                q_rms,
                queries,
                pred,
                target,
                counted,
            },
        )
    }

    /// The indexer loss's gradient, into the indexer's own weights.
    fn indexer_backward(
        &self,
        o: FullOffsets,
        x: &[f32],
        c: &IndexerCache,
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, d, nh, r) = (cfg.hidden, cfg.idx_dim, cfg.idx_heads, cfg.ratio);
        if c.counted == 0 {
            return;
        }
        let n_blocks = t / r;
        let unit = cfg.idx_coef / c.counted as f32;
        let mut d_queries = Aligned::zeros(t * nh * d);
        let mut d_blocks = Aligned::zeros(n_blocks * d);
        for i in 0..t {
            for (b, (&p, &tg)) in c.pred[i].iter().zip(&c.target[i]).enumerate() {
                let ds = unit * (p - tg);
                for hh in 0..nh {
                    let at = i * nh * d + hh * d;
                    let qv = &c.queries[at..at + d];
                    let kv = &c.blocks[b * d..(b + 1) * d];
                    if qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() > 0.0 {
                        for j in 0..d {
                            d_queries[at + j] += ds * kv[j];
                            d_blocks[b * d + j] += ds * qv[j];
                        }
                    }
                }
            }
        }
        for i in 0..t {
            rope_at(
                &mut d_queries[i * nh * d..(i + 1) * nh * d],
                d,
                cfg.rope_dim,
                cfg.rope_base,
                i,
                true,
            );
        }
        let mut d_q_raw = Aligned::zeros(t * nh * d);
        model::rmsnorm_backward(
            &mut d_q_raw,
            grad_slice(grads, o.idx_q_norm, d),
            &d_queries,
            &c.q_raw,
            self.tensor(o.idx_q_norm, d),
            &c.q_rms,
            t * nh,
            d,
        );
        matmul_add_dw(
            grad_slice(grads, o.idx_q, nh * d * h),
            &d_q_raw,
            x,
            t,
            h,
            nh * d,
        );
        for b in 0..n_blocks {
            rope_at(
                &mut d_blocks[b * d..(b + 1) * d],
                d,
                cfg.rope_dim,
                cfg.rope_base,
                b * r,
                true,
            );
        }
        let mut d_pooled = Aligned::zeros(n_blocks * d);
        model::rmsnorm_backward(
            &mut d_pooled,
            grad_slice(grads, o.idx_k_norm, d),
            &d_blocks,
            &c.pooled,
            self.tensor(o.idx_k_norm, d),
            &c.pooled_rms,
            n_blocks,
            d,
        );
        let mut d_keys = Aligned::zeros(t * d);
        for b in 0..n_blocks {
            for p in b * r..(b + 1) * r {
                for j in 0..d {
                    d_keys[p * d + j] = d_pooled[b * d + j] / r as f32;
                }
            }
        }
        matmul_add_dw(grad_slice(grads, o.idx_k, d * h), &d_keys, x, t, h, d);
    }

    /// The n-gram hash embedding added into the streams `x` (`[t, wide]`):
    ///
    /// ```text
    /// e      = the hashed rows of every head, side by side      [hidden]
    /// key    = gnorm(e · key);  query = gnorm(x)                 [wide]
    /// gate_c = ple_gate(key_c · query_c / sqrt(hidden))          per stream
    /// g_c    = (e · value) · gate_c
    /// x_c   += g_c + silu(dilated causal conv(gnorm(g))_c)
    /// ```
    ///
    /// `gnorm` normalizes each stream alone with a gamma across all of
    /// them; the convolution's taps are `ngram` positions apart, the last
    /// on the current token. The engine's `forward_ple`.
    fn ple_forward(&self, p: &PleConfig, o: PleOffsets, x: &[f32], tokens: &[u32]) -> PleCache {
        let cfg = &self.cfg;
        let (t, h, hc, w) = (tokens.len(), cfg.hidden, cfg.hc, cfg.wide());
        let hd = h / p.heads();
        let table = self.tensor(
            self.layout.ple_table.expect("a PLE block has a table"),
            p.rows() * hd,
        );
        let rows = p.hash(tokens);
        let mut embd = Aligned::zeros(t * h);
        for (i, token_rows) in rows.chunks(p.heads()).enumerate() {
            for (head, &r) in token_rows.iter().enumerate() {
                embd[i * h + head * hd..i * h + (head + 1) * hd]
                    .copy_from_slice(&table[r * hd..(r + 1) * hd]);
            }
        }
        let mut key = Aligned::zeros(t * w);
        let mut value = Aligned::zeros(t * h);
        matmul(&mut key, &embd, self.tensor(o.key, w * h), t, h, w);
        matmul(&mut value, &embd, self.tensor(o.value, h * h), t, h, h);
        let (key_n, key_s) = grouped_norm(&key, self.tensor(o.norm_key, w), hc, h, cfg.eps);
        let (query, query_s) = grouped_norm(x, self.tensor(o.norm_query, w), hc, h, cfg.eps);
        let scale = 1.0 / (h as f32).sqrt();
        let mut score = vec![0.0f32; t * hc];
        let mut gated = Aligned::zeros(t * w);
        for i in 0..t {
            for c in 0..hc {
                let at = i * w + c * h;
                let s: f32 = key_n[at..at + h]
                    .iter()
                    .zip(&query[at..at + h])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                score[i * hc + c] = s;
                let gate = ple_gate(s);
                for j in 0..h {
                    gated[at + j] = value[i * h + j] * gate;
                }
            }
        }
        let (normed, normed_s) = grouped_norm(&gated, self.tensor(o.norm_conv, w), hc, h, cfg.eps);
        let kernel = self.tensor(o.conv, w * p.kernel);
        let (kw, dil) = (p.kernel, p.ngram);
        let mut conv = Aligned::zeros(t * w);
        for i in 0..t {
            for ch in 0..w {
                let mut sum = 0.0f32;
                for k in 0..kw {
                    if let Some(src) = i.checked_sub((kw - 1 - k) * dil) {
                        sum += kernel[ch * kw + k] * normed[src * w + ch];
                    }
                }
                conv[i * w + ch] = sum;
            }
        }
        let mut out = Aligned::from_slice(x);
        for i in 0..t * w {
            out[i] += gated[i] + silu(conv[i]);
        }
        PleCache {
            rows,
            embd,
            key,
            value,
            key_n,
            key_s,
            query,
            query_s,
            score,
            gated,
            normed,
            normed_s,
            conv,
            out,
        }
    }

    /// [`Self::ple_forward`]'s backward: `d` is the gradient of its output
    /// streams on the way in, and of its input streams on the way out.
    #[allow(clippy::too_many_arguments)]
    fn ple_backward(
        &self,
        p: &PleConfig,
        o: PleOffsets,
        x: &[f32],
        c: &PleCache,
        d: &mut [f32],
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, hc, w) = (cfg.hidden, cfg.hc, cfg.wide());
        let hd = h / p.heads();
        let (kw, dil) = (p.kernel, p.ngram);
        // x_out = x + gated + silu(conv(normed)): `d` passes straight through.
        let mut d_gated = Aligned::from_slice(d);
        let mut d_conv = Aligned::zeros(t * w);
        for i in 0..t * w {
            let v = c.conv[i];
            let s = sigmoid(v);
            d_conv[i] = d[i] * s * (1.0 + v * (1.0 - s));
        }
        let kernel = self.tensor(o.conv, w * kw).to_vec();
        let mut d_normed = Aligned::zeros(t * w);
        {
            let d_kernel = grad_slice(grads, o.conv, w * kw);
            for i in 0..t {
                for ch in 0..w {
                    let g = d_conv[i * w + ch];
                    for k in 0..kw {
                        if let Some(src) = i.checked_sub((kw - 1 - k) * dil) {
                            d_normed[src * w + ch] += kernel[ch * kw + k] * g;
                            d_kernel[ch * kw + k] += c.normed[src * w + ch] * g;
                        }
                    }
                }
            }
        }
        let gamma = self.tensor(o.norm_conv, w).to_vec();
        grouped_norm_backward(
            &mut d_gated,
            grad_slice(grads, o.norm_conv, w),
            &d_normed,
            &c.gated,
            &gamma,
            &c.normed_s,
            (hc, h),
        );
        // gated_c = value · gate(score_c).
        let scale = 1.0 / (h as f32).sqrt();
        let mut d_value = Aligned::zeros(t * h);
        let mut d_key_n = Aligned::zeros(t * w);
        let mut d_query = Aligned::zeros(t * w);
        for i in 0..t {
            for cc in 0..hc {
                let at = i * w + cc * h;
                let s = c.score[i * hc + cc];
                let gate = ple_gate(s);
                let mut dot = 0.0f32;
                for j in 0..h {
                    d_value[i * h + j] += d_gated[at + j] * gate;
                    dot += d_gated[at + j] * c.value[i * h + j];
                }
                let ds = dot * ple_gate_grad(s) * scale;
                for j in 0..h {
                    d_key_n[at + j] = ds * c.query[at + j];
                    d_query[at + j] = ds * c.key_n[at + j];
                }
            }
        }
        let gamma_q = self.tensor(o.norm_query, w).to_vec();
        grouped_norm_backward(
            d,
            grad_slice(grads, o.norm_query, w),
            &d_query,
            x,
            &gamma_q,
            &c.query_s,
            (hc, h),
        );
        let gamma_k = self.tensor(o.norm_key, w).to_vec();
        let mut d_key = Aligned::zeros(t * w);
        grouped_norm_backward(
            &mut d_key,
            grad_slice(grads, o.norm_key, w),
            &d_key_n,
            &c.key,
            &gamma_k,
            &c.key_s,
            (hc, h),
        );
        let mut d_embd = Aligned::zeros(t * h);
        matmul_add_dw(grad_slice(grads, o.key, w * h), &d_key, &c.embd, t, h, w);
        matmul_add_dx(&mut d_embd, &d_key, self.tensor(o.key, w * h), t, h, w);
        matmul_add_dw(
            grad_slice(grads, o.value, h * h),
            &d_value,
            &c.embd,
            t,
            h,
            h,
        );
        matmul_add_dx(&mut d_embd, &d_value, self.tensor(o.value, h * h), t, h, h);
        let table = grad_slice(
            grads,
            self.layout.ple_table.expect("a PLE block has a table"),
            p.rows() * hd,
        );
        for (i, token_rows) in c.rows.chunks(p.heads()).enumerate() {
            for (head, &r) in token_rows.iter().enumerate() {
                for j in 0..hd {
                    table[r * hd + j] += d_embd[i * h + head * hd + j];
                }
            }
        }
    }

    /// The gated delta rule over `x` (`[t, hidden]`):
    ///
    /// ```text
    /// [q|k|v] = silu(causal depthwise conv(x · qkv))
    /// q, k    = per key head: x / max(|x|, eps); q also × 1/sqrt(dim)
    /// beta    = sigmoid(x · beta);  decay = exp(softplus(x · alpha + dt) · A)
    /// per value head, its state S [dim, dim] from zero:
    ///     S ← decay·S;  d ← beta·(v − kᵀS);  S ← S + k dᵀ;  o ← qᵀS
    /// out     = (rmsnorm(o) per head · norm) ⊙ sigmoid(x · gate) · out
    /// ```
    ///
    /// with `A = −exp(A_log)`, and value head `h` reading key head
    /// `h mod k_heads` (the heads tiled). The engine's `qwen_hybrid`
    /// recurrence, token for token.
    fn delta_forward(&self, o: LinearOffsets, x: &[f32], t: usize) -> (Aligned, DeltaCache) {
        let cfg = &self.cfg;
        let (h, ch, kd, vd) = (cfg.hidden, cfg.channels(), cfg.key_dim(), cfg.value_dim());
        let (nk, nv, dim, kw) = (cfg.ssm_k_heads, cfg.ssm_v_heads, cfg.ssm_dim, cfg.conv);
        let mut mixed = Aligned::zeros(t * ch);
        let mut z = Aligned::zeros(t * vd);
        let mut braw = Aligned::zeros(t * nv);
        let mut alpha = Aligned::zeros(t * nv);
        matmul(&mut mixed, x, self.tensor(o.qkv, ch * h), t, h, ch);
        matmul(&mut z, x, self.tensor(o.gate, vd * h), t, h, vd);
        matmul(&mut braw, x, self.tensor(o.beta, nv * h), t, h, nv);
        matmul(&mut alpha, x, self.tensor(o.alpha, nv * h), t, h, nv);

        // The causal convolution: the kernel's last tap weighs the current
        // token, and positions before the sequence read as zero.
        let kernel = self.tensor(o.conv, ch * kw);
        let mut conv = Aligned::zeros(t * ch);
        conv.par_chunks_mut(ch).enumerate().for_each(|(i, row)| {
            for (c, out) in row.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for j in 0..kw {
                    if let Some(src) = (i + j + 1).checked_sub(kw) {
                        sum += kernel[c * kw + j] * mixed[src * ch + c];
                    }
                }
                *out = sum;
            }
        });
        let scale = 1.0 / (dim as f32).sqrt();
        let mut q = Aligned::zeros(t * kd);
        let mut k = Aligned::zeros(t * kd);
        let mut v = Aligned::zeros(t * vd);
        let mut q_len = Aligned::zeros(t * nk);
        let mut k_len = Aligned::zeros(t * nk);
        q.par_chunks_mut(kd)
            .zip(k.par_chunks_mut(kd))
            .zip(v.par_chunks_mut(vd))
            .zip(q_len.par_chunks_mut(nk).zip(k_len.par_chunks_mut(nk)))
            .zip(conv.par_chunks(ch))
            .for_each(|((((q, k), v), (q_len, k_len)), row)| {
                for (dst, &c) in q.iter_mut().zip(&row[..kd]) {
                    *dst = silu(c);
                }
                for (dst, &c) in k.iter_mut().zip(&row[kd..2 * kd]) {
                    *dst = silu(c);
                }
                for (dst, &c) in v.iter_mut().zip(&row[2 * kd..]) {
                    *dst = silu(c);
                }
                for head in 0..nk {
                    let r = head * dim..(head + 1) * dim;
                    q_len[head] = l2_normalize(&mut q[r.clone()], cfg.eps);
                    k_len[head] = l2_normalize(&mut k[r.clone()], cfg.eps);
                    for v in &mut q[r] {
                        *v *= scale;
                    }
                }
            });
        let a_log = self.tensor(o.a_log, nv);
        let dt = self.tensor(o.dt, nv);
        let mut beta = Aligned::zeros(t * nv);
        let mut g = Aligned::zeros(t * nv);
        let mut decay = Aligned::zeros(t * nv);
        beta.par_chunks_mut(nv)
            .zip(g.par_chunks_mut(nv).zip(decay.par_chunks_mut(nv)))
            .zip(braw.par_chunks(nv).zip(alpha.par_chunks(nv)))
            .for_each(|((beta, (g, decay)), (braw, alpha))| {
                for hv in 0..nv {
                    beta[hv] = sigmoid(braw[hv]);
                    g[hv] = alpha[hv] + dt[hv];
                    decay[hv] = (softplus(g[hv]) * -a_log[hv].exp()).exp();
                }
            });

        // The recurrence. A value column of a head's state never reads
        // another, so each head runs as blocks of `cols` columns side by
        // side; the state is kept at every chunk boundary for the backward.
        let chunk = cfg.chunk.max(1);
        let cols = column_block(dim);
        let blocks = dim / cols;
        let mut o_all = Aligned::zeros(t * vd);
        let mut checkpoints: Vec<Aligned> = (0..t.div_ceil(chunk))
            .map(|_| Aligned::zeros(nv * dim * dim))
            .collect();
        let block_outputs: Vec<(Vec<f32>, Vec<Vec<f32>>)> = (0..nv * blocks)
            .into_par_iter()
            .map(|task| {
                let (hv, a0) = (task / blocks, task % blocks * cols);
                let hk = hv % nk;
                let mut state = vec![0.0f32; dim * cols];
                let mut out = vec![0.0f32; t * cols];
                let mut kept = Vec::new();
                for i in 0..t {
                    if i % chunk == 0 {
                        kept.push(state.clone());
                    }
                    let at_v = i * vd + hv * dim + a0;
                    delta_step(
                        &mut state,
                        &q[i * kd + hk * dim..i * kd + (hk + 1) * dim],
                        &k[i * kd + hk * dim..i * kd + (hk + 1) * dim],
                        &v[at_v..at_v + cols],
                        beta[i * nv + hv],
                        decay[i * nv + hv],
                        &mut out[i * cols..(i + 1) * cols],
                    );
                }
                (out, kept)
            })
            .collect();
        o_all.par_chunks_mut(vd).enumerate().for_each(|(i, row)| {
            for (task, (out, _)) in block_outputs.iter().enumerate() {
                let at = task / blocks * dim + task % blocks * cols;
                row[at..at + cols].copy_from_slice(&out[i * cols..(i + 1) * cols]);
            }
        });
        checkpoints
            .par_iter_mut()
            .enumerate()
            .for_each(|(c, checkpoint)| {
                for (task, (_, kept)) in block_outputs.iter().enumerate() {
                    let (hv, a0) = (task / blocks, task % blocks * cols);
                    let head = &mut checkpoint[hv * dim * dim..(hv + 1) * dim * dim];
                    for (row, src) in head.chunks_exact_mut(dim).zip(kept[c].chunks_exact(cols)) {
                        row[a0..a0 + cols].copy_from_slice(src);
                    }
                }
            });

        // The gated norm, per head, and the output projection.
        let (o_normed, o_rms) =
            model::rmsnorm_forward(&o_all, self.tensor(o.norm, dim), t * nv, dim, cfg.eps);
        let mut y = Aligned::zeros(t * vd);
        y.par_chunks_mut(vd)
            .zip(o_normed.par_chunks(vd).zip(z.par_chunks(vd)))
            .for_each(|(y, (o_normed, z))| {
                for ((yv, &n), &zv) in y.iter_mut().zip(o_normed).zip(z) {
                    *yv = n * sigmoid(zv);
                }
            });
        let mut out = Aligned::zeros(t * h);
        matmul(&mut out, &y, self.tensor(o.out, h * vd), t, vd, h);
        (
            out,
            DeltaCache {
                mixed,
                conv,
                q_len,
                k_len,
                q,
                k,
                v,
                beta,
                g,
                decay,
                o: o_all,
                o_rms,
                o_normed,
                z,
                y,
                checkpoints,
            },
        )
    }

    /// [`Self::delta_forward`]'s backward: adds the gradient of its input
    /// into `dx`. The recurrence is run back a chunk at a time, its states
    /// rebuilt forward from the chunk's checkpoint first.
    #[allow(clippy::too_many_arguments)]
    fn delta_backward(
        &self,
        o: LinearOffsets,
        x: &[f32],
        c: &DeltaCache,
        d_out: &[f32],
        dx: &mut [f32],
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, ch, kd, vd) = (cfg.hidden, cfg.channels(), cfg.key_dim(), cfg.value_dim());
        let (nk, nv, dim, kw) = (cfg.ssm_k_heads, cfg.ssm_v_heads, cfg.ssm_dim, cfg.conv);

        // The output projection, the gate and the per-head norm.
        matmul_add_dw(grad_slice(grads, o.out, h * vd), d_out, &c.y, t, vd, h);
        let mut d_y = Aligned::zeros(t * vd);
        matmul_add_dx(&mut d_y, d_out, self.tensor(o.out, h * vd), t, vd, h);
        let mut d_normed = Aligned::zeros(t * vd);
        let mut d_z = Aligned::zeros(t * vd);
        d_normed
            .par_chunks_mut(vd)
            .zip(d_z.par_chunks_mut(vd))
            .enumerate()
            .for_each(|(i, (d_normed, d_z))| {
                for e in 0..vd {
                    let at = i * vd + e;
                    let s = sigmoid(c.z[at]);
                    d_normed[e] = d_y[at] * s;
                    d_z[e] = d_y[at] * c.o_normed[at] * s * (1.0 - s);
                }
            });
        let mut d_o = Aligned::zeros(t * vd);
        model::rmsnorm_backward(
            &mut d_o,
            grad_slice(grads, o.norm, dim),
            &d_normed,
            &c.o,
            self.tensor(o.norm, dim),
            &c.o_rms,
            t * nv,
            dim,
        );

        // Back through time, a value head at a time, its column blocks side
        // by side a chunk at a time; a block's share of each token's query,
        // key, beta and decay gradient is added in block order.
        let chunk = cfg.chunk.max(1);
        let cols = column_block(dim);
        let blocks = dim / cols;
        let per_head: Vec<HeadGrads> = (0..nv)
            .into_par_iter()
            .map(|hv| {
                let hk = hv % nk;
                let at_q = |i: usize| i * kd + hk * dim..i * kd + (hk + 1) * dim;
                let mut gr = HeadGrads {
                    q: vec![0.0; t * dim],
                    k: vec![0.0; t * dim],
                    v: vec![0.0; t * dim],
                    beta: vec![0.0; t],
                    decay: vec![0.0; t],
                };
                // Each block's gradient of its state after the chunk, its
                // states before each token of the chunk, and its share.
                let size = dim * cols;
                let mut works: Vec<BlockWork> = (0..blocks)
                    .map(|_| BlockWork {
                        g_state: vec![0.0; size],
                        before: vec![0.0; chunk.min(t) * size],
                        scratch: vec![0.0; STEP_SCRATCH * dim],
                        share: HeadGrads {
                            q: vec![0.0; chunk * dim],
                            k: vec![0.0; chunk * dim],
                            v: vec![0.0; chunk * cols],
                            beta: vec![0.0; chunk],
                            decay: vec![0.0; chunk],
                        },
                    })
                    .collect();
                for start in (0..t).step_by(chunk).rev() {
                    let end = (start + chunk).min(t);
                    let n = end - start;
                    let checkpoint =
                        &c.checkpoints[start / chunk][hv * dim * dim..(hv + 1) * dim * dim];
                    works.par_iter_mut().enumerate().for_each(|(block, work)| {
                        let a0 = block * cols;
                        let at_v = |i: usize| i * vd + hv * dim + a0..i * vd + hv * dim + a0 + cols;
                        let before = &mut work.before;
                        for (dst, row) in before[..size]
                            .chunks_exact_mut(cols)
                            .zip(checkpoint.chunks_exact(dim))
                        {
                            dst.copy_from_slice(&row[a0..a0 + cols]);
                        }
                        let mut out = vec![0.0f32; cols];
                        for r in 1..n {
                            let i = start + r - 1;
                            let (done, next) = before.split_at_mut(r * size);
                            let next = &mut next[..size];
                            next.copy_from_slice(&done[(r - 1) * size..]);
                            out.fill(0.0);
                            delta_step(
                                next,
                                &c.q[at_q(i)],
                                &c.k[at_q(i)],
                                &c.v[at_v(i)],
                                c.beta[i * nv + hv],
                                c.decay[i * nv + hv],
                                &mut out,
                            );
                        }
                        let share = &mut work.share;
                        for r in (0..n).rev() {
                            let i = start + r;
                            let (db, dd) = delta_step_backward(
                                &before[r * size..(r + 1) * size],
                                &c.q[at_q(i)],
                                &c.k[at_q(i)],
                                &c.v[at_v(i)],
                                c.beta[i * nv + hv],
                                c.decay[i * nv + hv],
                                &d_o[at_v(i)],
                                &mut work.g_state,
                                (
                                    &mut share.q[r * dim..(r + 1) * dim],
                                    &mut share.k[r * dim..(r + 1) * dim],
                                    &mut share.v[r * cols..(r + 1) * cols],
                                ),
                                &mut work.scratch,
                            );
                            share.beta[r] = db;
                            share.decay[r] = dd;
                        }
                    });
                    for (block, work) in works.iter().enumerate() {
                        let share = &work.share;
                        for r in 0..n {
                            let i = start + r;
                            let span = i * dim..(i + 1) * dim;
                            let own = r * dim..(r + 1) * dim;
                            for (d, s) in gr.q[span.clone()].iter_mut().zip(&share.q[own.clone()]) {
                                *d += s;
                            }
                            for (d, s) in gr.k[span].iter_mut().zip(&share.k[own]) {
                                *d += s;
                            }
                            gr.v[i * dim + block * cols..i * dim + (block + 1) * cols]
                                .copy_from_slice(&share.v[r * cols..(r + 1) * cols]);
                            gr.beta[i] += share.beta[r];
                            gr.decay[i] += share.decay[r];
                        }
                    }
                }
                gr
            })
            .collect();

        // Each key head's gradient is the sum over the value heads reading
        // it, taken in head order.
        let mut d_q = Aligned::zeros(t * kd);
        let mut d_k = Aligned::zeros(t * kd);
        let mut d_v = Aligned::zeros(t * vd);
        let mut d_braw = Aligned::zeros(t * nv);
        let mut d_g = Aligned::zeros(t * nv);
        let a_log = self.tensor(o.a_log, nv);
        // A token at a time; each key head's sum over its value heads in
        // head order.
        d_q.par_chunks_mut(kd)
            .zip(d_k.par_chunks_mut(kd))
            .zip(d_v.par_chunks_mut(vd))
            .zip(d_braw.par_chunks_mut(nv).zip(d_g.par_chunks_mut(nv)))
            .enumerate()
            .for_each(|(i, (((d_q, d_k), d_v), (d_braw, d_g)))| {
                for (hv, gr) in per_head.iter().enumerate() {
                    let hk = hv % nk;
                    let a = -a_log[hv].exp();
                    for j in 0..dim {
                        d_q[hk * dim + j] += gr.q[i * dim + j];
                        d_k[hk * dim + j] += gr.k[i * dim + j];
                        d_v[hv * dim + j] = gr.v[i * dim + j];
                    }
                    let at = i * nv + hv;
                    let b = c.beta[at];
                    d_braw[hv] = gr.beta[i] * b * (1.0 - b);
                    // decay = exp(softplus(g) · a).
                    let dl = gr.decay[i] * c.decay[at];
                    d_g[hv] = dl * a * softplus_grad(c.g[at]);
                }
            });
        // A head at a time, over the tokens in order: a = −exp(a_log), so
        // da/da_log = a.
        let d_a_log: Vec<f32> = per_head
            .par_iter()
            .enumerate()
            .map(|(hv, gr)| {
                let a = -a_log[hv].exp();
                let mut sum = 0.0f32;
                for (i, &decay_i) in gr.decay.iter().enumerate().take(t) {
                    let at = i * nv + hv;
                    let dl = decay_i * c.decay[at];
                    sum += dl * softplus(c.g[at]) * a;
                }
                sum
            })
            .collect();
        for (dst, src) in grad_slice(grads, o.a_log, nv).iter_mut().zip(&d_a_log) {
            *dst += src;
        }
        {
            let d_dt = grad_slice(grads, o.dt, nv);
            for i in 0..t {
                for hv in 0..nv {
                    d_dt[hv] += d_g[i * nv + hv];
                }
            }
        }

        // The normalizations, the silu and the convolution.
        let scale = 1.0 / (dim as f32).sqrt();
        let mut d_act = Aligned::zeros(t * ch);
        d_act.par_chunks_mut(ch).enumerate().for_each(|(i, d_act)| {
            for head in 0..nk {
                let r = i * kd + head * dim..i * kd + (head + 1) * dim;
                // q carries the scale after its normalization.
                let qn: Vec<f32> = c.q[r.clone()].iter().map(|v| v / scale).collect();
                let dqn: Vec<f32> = d_q[r.clone()].iter().map(|v| v * scale).collect();
                l2_normalize_backward(
                    &dqn,
                    &qn,
                    c.q_len[i * nk + head],
                    cfg.eps,
                    &mut d_act[head * dim..(head + 1) * dim],
                );
                l2_normalize_backward(
                    &d_k[r.clone()],
                    &c.k[r],
                    c.k_len[i * nk + head],
                    cfg.eps,
                    &mut d_act[kd + head * dim..kd + (head + 1) * dim],
                );
            }
            d_act[2 * kd..].copy_from_slice(&d_v[i * vd..(i + 1) * vd]);
        });
        let mut d_conv = Aligned::zeros(t * ch);
        d_conv
            .par_chunks_mut(ch)
            .zip(d_act.par_chunks(ch).zip(c.conv.par_chunks(ch)))
            .for_each(|(d_conv, (d_act, conv))| {
                for ((dc, &da), &cv) in d_conv.iter_mut().zip(d_act).zip(conv) {
                    let s = sigmoid(cv);
                    *dc = da * s * (1.0 + cv * (1.0 - s));
                }
            });
        // Token `i` reads rows `i + 1 − kw ..= i` through taps `0..kw`. The
        // input's gradient gathered a row at a time, from the later tokens
        // in order; the kernel's a channel at a time, over the tokens in
        // order.
        let kernel = self.tensor(o.conv, ch * kw);
        let mut d_mixed = Aligned::zeros(t * ch);
        d_mixed
            .par_chunks_mut(ch)
            .enumerate()
            .for_each(|(src, d_mixed)| {
                for i in src..(src + kw).min(t) {
                    let j = src + kw - 1 - i;
                    for (cc, d) in d_mixed.iter_mut().enumerate() {
                        *d += kernel[cc * kw + j] * d_conv[i * ch + cc];
                    }
                }
            });
        grad_slice(grads, o.conv, ch * kw)
            .par_chunks_mut(kw)
            .enumerate()
            .for_each(|(cc, d_kernel)| {
                for i in 0..t {
                    let g = d_conv[i * ch + cc];
                    for (j, d) in d_kernel.iter_mut().enumerate() {
                        if let Some(src) = (i + j + 1).checked_sub(kw) {
                            *d += c.mixed[src * ch + cc] * g;
                        }
                    }
                }
            });

        // The four projections of the input.
        for (w_at, dy, n) in [
            (o.qkv, &d_mixed, ch),
            (o.gate, &d_z, vd),
            (o.beta, &d_braw, nv),
            (o.alpha, &d_g, nv),
        ] {
            matmul_add_dw(grad_slice(grads, w_at, n * h), dy, x, t, h, n);
            matmul_add_dx(dx, dy, self.tensor(w_at, n * h), t, h, n);
        }
    }

    /// The routed feed-forward over `x` (`[t, hidden]`): a softmax router,
    /// the top `experts_used` experts by probability (ties to the lower
    /// index) weighted by their probabilities renormalized over the chosen
    /// ones, each a SwiGLU; plus a shared SwiGLU expert gated per token by
    /// `sigmoid(x · shared_gate)`. The engine's `MoeFfn`.
    fn moe_forward(&self, m: MoeOffsets, x: &[f32], t: usize) -> (Aligned, MoeCache) {
        let cfg = &self.cfg;
        let (h, ne, k, fe, fs) = (
            cfg.hidden,
            cfg.experts,
            cfg.experts_used,
            cfg.expert_ffn,
            cfg.shared_ffn,
        );
        let mut probs = Aligned::zeros(t * ne);
        matmul(&mut probs, x, self.tensor(m.router, ne * h), t, h, ne);
        let (chosen, sums): (Vec<Vec<(usize, f32)>>, Vec<f32>) = probs
            .par_chunks_mut(ne)
            .map(|row| {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                order.truncate(k);
                let sum: f32 = order.iter().map(|&e| row[e]).sum();
                let s = sum.max(WEIGHT_FLOOR);
                (order.into_iter().map(|e| (e, row[e] / s)).collect(), sum)
            })
            .unzip();

        // Each expert's tokens together, in expert order.
        let mut members: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
        for (i, picks) in chosen.iter().enumerate() {
            for (slot, &(e, _)) in picks.iter().enumerate() {
                members[e].push((i, slot));
            }
        }
        let experts: Vec<ExpertCache> = match accel::for_split(t * k, fe, h) {
            // Divided between the card and the processor, side by side, in
            // proportion to what each has been measured to manage.
            Some(card) => {
                let on_card =
                    split_experts(&members, accel::card_share(accel::Split::ExpertsForward));
                let mine: Vec<Vec<(usize, usize)>> = (0..ne)
                    .map(|e| {
                        if on_card[e] {
                            members[e].clone()
                        } else {
                            Vec::new()
                        }
                    })
                    .collect();
                let ((from_card, card_secs), (from_cpu, cpu_secs)) = rayon::join(
                    || timed(|| self.experts_on_card(card, m, x, &mine)),
                    || {
                        timed(|| {
                            (0..ne)
                                .into_par_iter()
                                .map(|e| {
                                    (!on_card[e]).then(|| self.expert_forward(m, x, &members, e))
                                })
                                .collect::<Vec<_>>()
                        })
                    },
                );
                let rows = |card: bool| {
                    (0..ne)
                        .filter(|&e| on_card[e] == card)
                        .map(|e| members[e].len())
                        .sum()
                };
                accel::measured(
                    accel::Split::ExpertsForward,
                    (rows(true), card_secs),
                    (rows(false), cpu_secs),
                );
                from_card
                    .into_iter()
                    .zip(from_cpu)
                    .map(|(card, cpu)| cpu.unwrap_or(card))
                    .collect()
            }
            None => (0..ne)
                .into_par_iter()
                .map(|e| self.expert_forward(m, x, &members, e))
                .collect(),
        };

        // The shared expert and its gate.
        let mut s_gate = Aligned::zeros(t * fs);
        let mut s_up = Aligned::zeros(t * fs);
        matmul(&mut s_gate, x, self.tensor(m.gate_shexp, fs * h), t, h, fs);
        matmul(&mut s_up, x, self.tensor(m.up_shexp, fs * h), t, h, fs);
        let mut s_act = Aligned::zeros(t * fs);
        s_act
            .par_chunks_mut(fs)
            .zip(s_gate.par_chunks(fs).zip(s_up.par_chunks(fs)))
            .for_each(|(act, (gate, up))| {
                for ((a, &g), &u) in act.iter_mut().zip(gate).zip(up) {
                    *a = silu(g) * u;
                }
            });
        let mut s_out = Aligned::zeros(t * h);
        matmul(
            &mut s_out,
            &s_act,
            self.tensor(m.down_shexp, h * fs),
            t,
            fs,
            h,
        );
        let shared_w = self.tensor(m.shared_gate, h);
        let s_score: Vec<f32> = (0..t)
            .into_par_iter()
            .map(|i| {
                x[i * h..(i + 1) * h]
                    .iter()
                    .zip(shared_w)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        // Shared expert first, then the routed ones in the order chosen.
        let mut out = Aligned::zeros(t * h);
        let mut position = vec![vec![0usize; k]; t];
        for rows in &members {
            for (r, &(i, slot)) in rows.iter().enumerate() {
                position[i][slot] = r;
            }
        }
        out.par_chunks_mut(h).enumerate().for_each(|(i, dst)| {
            let g = sigmoid(s_score[i]);
            for (d, &s) in dst.iter_mut().zip(&s_out[i * h..(i + 1) * h]) {
                *d = g * s;
            }
            for (slot, &(e, wgt)) in chosen[i].iter().enumerate() {
                let r = position[i][slot];
                for (d, &v) in dst.iter_mut().zip(&experts[e].out[r * h..(r + 1) * h]) {
                    *d += wgt * v;
                }
            }
        });

        // The load-balancing term: experts_count · Σ share · mean probability,
        // the share counted over every routing slot.
        let mut share = vec![0.0f32; ne];
        for (e, rows) in members.iter().enumerate() {
            share[e] = rows.len() as f32 / (t * k) as f32;
        }
        let mut mean = vec![0.0f32; ne];
        for row in probs.chunks(ne) {
            for (m_e, &p) in mean.iter_mut().zip(row) {
                *m_e += p / t as f32;
            }
        }
        let aux =
            cfg.aux_coef * ne as f32 * share.iter().zip(&mean).map(|(a, b)| a * b).sum::<f32>();
        (
            out,
            MoeCache {
                probs,
                chosen,
                sums,
                members,
                experts,
                s_gate,
                s_up,
                s_act,
                s_out,
                s_score,
                share,
                aux,
            },
        )
    }

    /// [`Self::moe_forward`]'s backward, the load-balancing term's included:
    /// adds the gradient of its input into `dx`. The choice of experts is
    /// held fixed; the gradient reaches the router through the chosen
    /// experts' weights and through the balancing term.
    #[allow(clippy::too_many_arguments)]
    fn moe_backward(
        &self,
        m: MoeOffsets,
        x: &[f32],
        c: &MoeCache,
        d_out: &[f32],
        dx: &mut [f32],
        grads: &mut [f32],
        t: usize,
    ) {
        let cfg = &self.cfg;
        let (h, ne, k, fe, fs) = (
            cfg.hidden,
            cfg.experts,
            cfg.experts_used,
            cfg.expert_ffn,
            cfg.shared_ffn,
        );

        // The shared expert and its gate.
        let mut d_s_out = Aligned::zeros(t * h);
        let mut d_score = vec![0.0f32; t];
        d_s_out
            .par_chunks_mut(h)
            .zip(d_score.par_iter_mut())
            .enumerate()
            .for_each(|(i, (d_s_out, d_score))| {
                let g = sigmoid(c.s_score[i]);
                let dy = &d_out[i * h..(i + 1) * h];
                let so = &c.s_out[i * h..(i + 1) * h];
                let gv = V8::splat(g);
                vector::map(d_s_out, [dy], |[y]| y * gv);
                let dot = vector::sum([dy, so], |[y, o]| y * o);
                *d_score = dot * g * (1.0 - g);
            });
        {
            let shared_w = self.tensor(m.shared_gate, h);
            // Each column's gradient summed over the tokens in order, and
            // each token's input gradient on its own row.
            grad_slice(grads, m.shared_gate, h)
                .par_chunks_mut(GAMMA_BLOCK)
                .enumerate()
                .for_each(|(block, d_w)| {
                    let col = block * GAMMA_BLOCK;
                    for (i, &score) in d_score.iter().enumerate() {
                        let (score, at) = (V8::splat(score), i * h + col);
                        vector::update(d_w, [&x[at..at + d_w.len()]], |o, [v]| o + score * v);
                    }
                });
            dx.par_chunks_mut(h)
                .zip(d_score.par_iter())
                .for_each(|(dx, &score)| {
                    for (d, &w) in dx.iter_mut().zip(shared_w) {
                        *d += score * w;
                    }
                });
        }
        {
            let [g_gate, g_up, g_down] = regions(
                grads,
                [
                    (m.gate_shexp, fs * h),
                    (m.up_shexp, fs * h),
                    (m.down_shexp, h * fs),
                ],
            );
            swiglu_backward(
                (
                    self.tensor(m.gate_shexp, fs * h),
                    self.tensor(m.up_shexp, fs * h),
                    self.tensor(m.down_shexp, h * fs),
                ),
                (g_gate, g_up, g_down),
                (h, fs),
                x,
                (&c.s_gate, &c.s_up, &c.s_act),
                &d_s_out,
                dx,
                t,
            );
        }

        // The routed experts, all at once: each writes only its own
        // matrices' gradients. What they give back for their tokens — the
        // input gradient and each chosen weight's gradient — is added in
        // expert order afterwards, the order one expert at a time adds it.
        let [g_gate, g_up, g_down] = regions(
            grads,
            [
                (m.gate_exps, ne * fe * h),
                (m.up_exps, ne * fe * h),
                (m.down_exps, ne * h * fe),
            ],
        );
        let chunks: Vec<(usize, ExpertGrads<'_>)> = g_gate
            .chunks_mut(fe * h)
            .zip(g_up.chunks_mut(fe * h))
            .zip(g_down.chunks_mut(h * fe))
            .map(|((a, b), d)| (a, b, d))
            .enumerate()
            .collect();
        let given: Vec<Option<(Aligned, Vec<f32>)>> = match accel::for_split(t * k, fe, h) {
            // Divided as the forward is, by its own measured rates.
            Some(card) => {
                let on_card =
                    split_experts(&c.members, accel::card_share(accel::Split::ExpertsBackward));
                let (card_chunks, cpu_chunks): (Vec<_>, Vec<_>) =
                    chunks.into_iter().partition(|(e, _)| on_card[*e]);
                let ((from_card, card_secs), (from_cpu, cpu_secs)) = rayon::join(
                    || timed(|| self.experts_backward_on_card(card, m, c, d_out, card_chunks)),
                    || {
                        timed(|| {
                            cpu_chunks
                                .into_par_iter()
                                .map(|(e, grads)| (e, self.expert_backward(m, c, d_out, e, grads)))
                                .collect::<Vec<_>>()
                        })
                    },
                );
                let rows = |card: bool| {
                    (0..ne)
                        .filter(|&e| on_card[e] == card)
                        .map(|e| c.members[e].len())
                        .sum()
                };
                accel::measured(
                    accel::Split::ExpertsBackward,
                    (rows(true), card_secs),
                    (rows(false), cpu_secs),
                );
                let mut given: Vec<Option<(Aligned, Vec<f32>)>> = (0..ne).map(|_| None).collect();
                for (e, g) in from_card.into_iter().chain(from_cpu) {
                    given[e] = g;
                }
                given
            }
            None => chunks
                .into_par_iter()
                .map(|(e, grads)| self.expert_backward(m, c, d_out, e, grads))
                .collect(),
        };
        // Each token's row in each of its experts' lists, then each token's
        // share added in expert order, the tokens side by side.
        let mut row_in = vec![vec![0usize; k]; t];
        for rows in &c.members {
            for (r, &(i, slot)) in rows.iter().enumerate() {
                row_in[i][slot] = r;
            }
        }
        let mut d_weight = vec![vec![0.0f32; k]; t];
        dx.par_chunks_mut(h)
            .zip(d_weight.par_iter_mut())
            .enumerate()
            .for_each(|(i, (dx, d_weight))| {
                let mut picks: Vec<(usize, usize)> = c.chosen[i]
                    .iter()
                    .enumerate()
                    .map(|(slot, &(e, _))| (e, slot))
                    .collect();
                picks.sort_unstable();
                for (e, slot) in picks {
                    let Some((d_xin, dots)) = &given[e] else {
                        continue;
                    };
                    let r = row_in[i][slot];
                    d_weight[slot] = dots[r];
                    vector::update(dx, [&d_xin[r * h..(r + 1) * h]], |o, [v]| o + v);
                }
            });

        // The router: through the renormalized weights, the balancing term,
        // and the softmax.
        let mut d_logits = Aligned::zeros(t * ne);
        let balance = cfg.aux_coef * ne as f32 / t as f32;
        for i in 0..t {
            let p = &c.probs[i * ne..(i + 1) * ne];
            let mut dp: Vec<f32> = c.share.iter().map(|s| balance * s).collect();
            let s = c.sums[i];
            if s > WEIGHT_FLOOR {
                let along: f32 = c.chosen[i]
                    .iter()
                    .zip(&d_weight[i])
                    .map(|(&(_, w), &dw)| w * dw)
                    .sum();
                for (&(e, _), &dw) in c.chosen[i].iter().zip(&d_weight[i]) {
                    dp[e] += (dw - along) / s;
                }
            } else {
                for (&(e, _), &dw) in c.chosen[i].iter().zip(&d_weight[i]) {
                    dp[e] += dw / WEIGHT_FLOOR;
                }
            }
            let inner: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
            for e in 0..ne {
                d_logits[i * ne + e] = p[e] * (dp[e] - inner);
            }
        }
        matmul_add_dw(grad_slice(grads, m.router, ne * h), &d_logits, x, t, h, ne);
        matmul_add_dx(dx, &d_logits, self.tensor(m.router, ne * h), t, h, ne);
    }

    /// One routed expert's forward on the processor, over its tokens.
    fn expert_forward(
        &self,
        m: MoeOffsets,
        x: &[f32],
        members: &[Vec<(usize, usize)>],
        e: usize,
    ) -> ExpertCache {
        let cfg = &self.cfg;
        let (h, fe) = (cfg.hidden, cfg.expert_ffn);
        let rows = &members[e];
        let m_n = rows.len();
        let mut xin = Aligned::zeros(m_n * h);
        for (r, &(i, _)) in rows.iter().enumerate() {
            xin[r * h..(r + 1) * h].copy_from_slice(&x[i * h..(i + 1) * h]);
        }
        let mut gate = Aligned::zeros(m_n * fe);
        let mut up = Aligned::zeros(m_n * fe);
        matmul(
            &mut gate,
            &xin,
            self.tensor(m.gate_exps + e * fe * h, fe * h),
            m_n,
            h,
            fe,
        );
        matmul(
            &mut up,
            &xin,
            self.tensor(m.up_exps + e * fe * h, fe * h),
            m_n,
            h,
            fe,
        );
        let mut act = Aligned::zeros(m_n * fe);
        for ((a, &g), &u) in act.iter_mut().zip(gate.iter()).zip(up.iter()) {
            *a = silu(g) * u;
        }
        let mut out = Aligned::zeros(m_n * h);
        matmul(
            &mut out,
            &act,
            self.tensor(m.down_exps + e * h * fe, h * fe),
            m_n,
            fe,
            h,
        );
        ExpertCache {
            xin,
            gate,
            up,
            act,
            out,
        }
    }

    /// One routed expert's backward on the processor: adds into its own
    /// matrices' gradients, and gives back its tokens' input gradient and
    /// the gradients of their weights.
    fn expert_backward(
        &self,
        m: MoeOffsets,
        c: &MoeCache,
        d_out: &[f32],
        e: usize,
        (g_gate, g_up, g_down): ExpertGrads<'_>,
    ) -> Option<(Aligned, Vec<f32>)> {
        let cfg = &self.cfg;
        let (h, fe) = (cfg.hidden, cfg.expert_ffn);
        let rows = &c.members[e];
        if rows.is_empty() {
            return None;
        }
        let ec = &c.experts[e];
        let m_n = rows.len();
        let mut d_eout = Aligned::zeros(m_n * h);
        let mut dots = vec![0.0f32; m_n];
        for (r, &(i, slot)) in rows.iter().enumerate() {
            let wgt = c.chosen[i][slot].1;
            let dy = &d_out[i * h..(i + 1) * h];
            let eo = &ec.out[r * h..(r + 1) * h];
            let mut dot = 0.0f32;
            for j in 0..h {
                d_eout[r * h + j] = wgt * dy[j];
                dot += dy[j] * eo[j];
            }
            dots[r] = dot;
        }
        let mut d_xin = Aligned::zeros(m_n * h);
        swiglu_backward(
            (
                self.tensor(m.gate_exps + e * fe * h, fe * h),
                self.tensor(m.up_exps + e * fe * h, fe * h),
                self.tensor(m.down_exps + e * h * fe, h * fe),
            ),
            (g_gate, g_up, g_down),
            (h, fe),
            &ec.xin,
            (&ec.gate, &ec.up, &ec.act),
            &d_eout,
            &mut d_xin,
            m_n,
        );
        Some((d_xin, dots))
    }

    /// The routed experts' forward on a card: every expert's tokens stacked
    /// in expert order, and each of the three products one dispatch over
    /// all the experts.
    fn experts_on_card(
        &self,
        card: &accel::Accel,
        m: MoeOffsets,
        x: &[f32],
        members: &[Vec<(usize, usize)>],
    ) -> Vec<ExpertCache> {
        let cfg = &self.cfg;
        let (h, ne, fe) = (cfg.hidden, cfg.experts, cfg.expert_ffn);
        let (starts, total) = stack_starts(members);
        let order: Vec<usize> = members.iter().flatten().map(|&(i, _)| i).collect();
        let mut xin = Aligned::zeros(total * h);
        xin.par_chunks_mut(h)
            .zip(order.par_iter())
            .for_each(|(row, &i)| row.copy_from_slice(&x[i * h..(i + 1) * h]));
        let parts = |a_width: usize, b_size: usize, c_width: usize, depth: usize| {
            members
                .iter()
                .enumerate()
                .filter(|(_, rows)| !rows.is_empty())
                .map(|(e, rows)| accel::Part {
                    a: starts[e] * a_width,
                    rows: rows.len(),
                    b: e * b_size,
                    c: starts[e] * c_width,
                    depth,
                })
                .collect::<Vec<_>>()
        };
        let mut gate = Aligned::zeros(total * fe);
        let mut up = Aligned::zeros(total * fe);
        let into = parts(h, fe * h, fe, h);
        for (out, w) in [(&mut gate, m.gate_exps), (&mut up, m.up_exps)] {
            card.grouped(
                out,
                View {
                    data: &xin,
                    row: h,
                    col: 1,
                },
                View {
                    data: self.tensor(w, ne * fe * h),
                    row: 1,
                    col: h,
                },
                fe,
                &into,
                false,
                true,
            );
        }
        let mut act = Aligned::zeros(total * fe);
        act.par_chunks_mut(fe)
            .zip(gate.par_chunks(fe).zip(up.par_chunks(fe)))
            .for_each(|(act, (gate, up))| {
                for ((a, &g), &u) in act.iter_mut().zip(gate).zip(up) {
                    *a = silu(g) * u;
                }
            });
        let mut out = Aligned::zeros(total * h);
        card.grouped(
            &mut out,
            View {
                data: &act,
                row: fe,
                col: 1,
            },
            View {
                data: self.tensor(m.down_exps, ne * h * fe),
                row: 1,
                col: fe,
            },
            h,
            &parts(fe, h * fe, h, fe),
            false,
            true,
        );
        let take = |all: &Aligned, width: usize, e: usize| {
            Aligned::from_slice(&all[starts[e] * width..(starts[e] + members[e].len()) * width])
        };
        (0..ne)
            .into_par_iter()
            .map(|e| ExpertCache {
                xin: take(&xin, h, e),
                gate: take(&gate, fe, e),
                up: take(&up, fe, e),
                act: take(&act, fe, e),
                out: take(&out, h, e),
            })
            .collect()
    }

    /// The routed experts' backward on a card, for the experts `grads`
    /// names, stacked as [`Self::experts_on_card`] stacks them: adds each
    /// one's matrices' gradients into the slices given for it, and gives
    /// back each one's input gradient and the gradients of its tokens'
    /// weights.
    fn experts_backward_on_card(
        &self,
        card: &accel::Accel,
        m: MoeOffsets,
        c: &MoeCache,
        d_out: &[f32],
        mut grads: Vec<(usize, ExpertGrads<'_>)>,
    ) -> Vec<(usize, Option<ExpertGiven>)> {
        let cfg = &self.cfg;
        let (h, ne, fe) = (cfg.hidden, cfg.experts, cfg.expert_ffn);
        grads.retain(|(e, _)| !c.members[*e].is_empty());
        let used: Vec<usize> = grads.iter().map(|(e, _)| *e).collect();
        let mut mine: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
        for &e in &used {
            mine[e] = c.members[e].clone();
        }
        let (starts, total) = stack_starts(&mine);
        let stack = |width: usize, of: &dyn Fn(&ExpertCache) -> &Aligned| {
            let mut all = Aligned::zeros(total * width);
            for &e in &used {
                let at = starts[e] * width;
                all[at..at + mine[e].len() * width].copy_from_slice(of(&c.experts[e]));
            }
            all
        };
        let xin = stack(h, &|ec| &ec.xin);
        let gate = stack(fe, &|ec| &ec.gate);
        let up = stack(fe, &|ec| &ec.up);
        let act = stack(fe, &|ec| &ec.act);
        let out = stack(h, &|ec| &ec.out);
        // Each routed row's output gradient, and the dot its weight's
        // gradient is.
        let rows: Vec<(usize, usize)> = mine.iter().flatten().copied().collect();
        let mut d_eout = Aligned::zeros(total * h);
        let mut dots = vec![0.0f32; total];
        d_eout
            .par_chunks_mut(h)
            .zip(dots.par_iter_mut())
            .enumerate()
            .for_each(|(r, (d, dot))| {
                let (i, slot) = rows[r];
                let wgt = c.chosen[i][slot].1;
                let dy = &d_out[i * h..(i + 1) * h];
                let eo = &out[r * h..(r + 1) * h];
                let mut sum = 0.0f32;
                for j in 0..h {
                    d[j] = wgt * dy[j];
                    sum += dy[j] * eo[j];
                }
                *dot = sum;
            });
        // Rows of a stack against an expert's matrix.
        let by_rows = |a_width: usize, b_size: usize, c_width: usize, depth: usize| {
            used.iter()
                .map(|&e| accel::Part {
                    a: starts[e] * a_width,
                    rows: mine[e].len(),
                    b: e * b_size,
                    c: starts[e] * c_width,
                    depth,
                })
                .collect::<Vec<_>>()
        };
        // An expert's matrix gradient, summed over its rows; the `i`th
        // product's result is the `i`th used expert's.
        let by_matrix = |a_width: usize, rows: usize, b_width: usize, c_size: usize| {
            used.iter()
                .enumerate()
                .map(|(i, &e)| accel::Part {
                    a: starts[e] * a_width,
                    rows,
                    b: starts[e] * b_width,
                    c: i * c_size,
                    depth: mine[e].len(),
                })
                .collect::<Vec<_>>()
        };
        let add_into = |dst: &mut [f32], got: &[f32]| {
            for (o, v) in dst.iter_mut().zip(got) {
                *o += v;
            }
        };
        // out = act · Dᵀ: into D and into act.
        card.grouped_with(
            used.len() * h * fe,
            View {
                data: &d_eout,
                row: 1,
                col: h,
            },
            View {
                data: &act,
                row: fe,
                col: 1,
            },
            fe,
            &by_matrix(h, h, fe, h * fe),
            false,
            &mut |i, got| add_into(grads[i].1.2, got),
        );
        let mut d_act = Aligned::zeros(total * fe);
        card.grouped(
            &mut d_act,
            View {
                data: &d_eout,
                row: h,
                col: 1,
            },
            View {
                data: self.tensor(m.down_exps, ne * h * fe),
                row: fe,
                col: 1,
            },
            fe,
            &by_rows(h, h * fe, fe, h),
            false,
            true,
        );
        let mut d_gate = Aligned::zeros(total * fe);
        let mut d_up = Aligned::zeros(total * fe);
        d_gate
            .par_chunks_mut(fe)
            .zip(d_up.par_chunks_mut(fe))
            .zip(d_act.par_chunks(fe))
            .zip(gate.par_chunks(fe).zip(up.par_chunks(fe)))
            .for_each(|(((d_gate, d_up), d_act), (gate, up))| {
                for i in 0..d_act.len() {
                    let g = gate[i];
                    let s = sigmoid(g);
                    d_up[i] = d_act[i] * silu(g);
                    d_gate[i] = d_act[i] * up[i] * s * (1.0 + g * (1.0 - s));
                }
            });
        // gate = x · Gᵀ, up = x · Uᵀ: into G, U and x.
        let mut d_xin = Aligned::zeros(total * h);
        for (which, w_at, dy) in [(0, m.gate_exps, &d_gate), (1, m.up_exps, &d_up)] {
            card.grouped_with(
                used.len() * fe * h,
                View {
                    data: dy,
                    row: 1,
                    col: fe,
                },
                View {
                    data: &xin,
                    row: h,
                    col: 1,
                },
                h,
                &by_matrix(fe, fe, h, fe * h),
                false,
                &mut |i, got| {
                    let dst: &mut [f32] = if which == 0 {
                        grads[i].1.0
                    } else {
                        grads[i].1.1
                    };
                    add_into(dst, got)
                },
            );
            card.grouped(
                &mut d_xin,
                View {
                    data: dy,
                    row: fe,
                    col: 1,
                },
                View {
                    data: self.tensor(w_at, ne * fe * h),
                    row: h,
                    col: 1,
                },
                h,
                &by_rows(fe, fe * h, h, fe),
                which == 1,
                true,
            );
        }
        used.iter()
            .map(|&e| {
                let (from, len) = (starts[e], mine[e].len());
                (
                    e,
                    Some((
                        Aligned::from_slice(&d_xin[from * h..(from + len) * h]),
                        dots[from..from + len].to_vec(),
                    )),
                )
            })
            .collect()
    }

    /// One block's forward, over the streams `x`.
    fn layer_forward(&self, l: usize, x: &[f32], tokens: &[u32], t: usize) -> LayerCache {
        let o = self.layout.layers[l];
        // The n-gram embedding first, on its block.
        let ple = match (self.cfg.ple.as_ref(), o.ple) {
            (Some(p), Some(po)) => Some(time(Stage::Ple, || self.ple_forward(p, po, x, tokens))),
            _ => None,
        };
        let x: &[f32] = ple.as_ref().map_or(x, |c| &c.out);
        let (attn_in, attn_mix) = time(Stage::HcMix, || self.mix(o.hc_attn, x, t));
        let (attn_out, mixer) = match o.mixer {
            TokenMixer::Full(f) => {
                let (out, c) = time(Stage::FullAttention, || self.attn_forward(f, &attn_in, t));
                (out, MixerCache::Full(c))
            }
            TokenMixer::Linear(lo) => {
                let (out, c) = time(Stage::DeltaRule, || self.delta_forward(lo, &attn_in, t));
                (out, MixerCache::Linear(c))
            }
        };
        let mut mid = Aligned::from_slice(x);
        time(Stage::HcCombine, || {
            self.combine(&mut mid, &attn_out, &attn_mix.inject, t)
        });
        let (ffn_in, ffn_mix) = time(Stage::HcMix, || self.mix(o.hc_ffn, &mid, t));
        let (ffn_out, moe) = time(Stage::Moe, || self.moe_forward(o.moe, &ffn_in, t));
        let out = time(Stage::HcCombine, || {
            let mut out = mid.clone();
            self.combine(&mut out, &ffn_out, &ffn_mix.inject, t);
            out
        });
        LayerCache {
            ple,
            attn_mix,
            attn_in,
            attn_out,
            mixer,
            mid,
            ffn_mix,
            ffn_in,
            moe,
            ffn_out,
            out,
        }
    }

    /// One block's backward: the gradient of its input streams, and its
    /// parameter gradients added into `grads`.
    #[allow(clippy::too_many_arguments)]
    fn layer_backward(
        &self,
        l: usize,
        x: &[f32],
        tokens: &[u32],
        dout: &[f32],
        t: usize,
        grads: &mut [f32],
        kept: Option<&LayerCache>,
    ) -> Aligned {
        let h = self.cfg.hidden;
        let o = self.layout.layers[l];
        // The forward's own intermediates when they were kept, else the
        // same arithmetic again.
        let rebuilt;
        let c = match kept {
            Some(c) => c,
            None => {
                rebuilt = self.layer_forward(l, x, tokens, t);
                &rebuilt
            }
        };
        let x_in: &[f32] = c.ple.as_ref().map_or(x, |p| &p.out);

        // The feed-forward's out-mix and the feed-forward.
        let (d_ffn_out, d_ffn_inject) = time(Stage::HcCombineBackward, || {
            self.combine_backward(dout, &c.ffn_out, &c.ffn_mix.inject, t)
        });
        let mut d_mid = Aligned::from_slice(dout);
        let mut d_ffn_in = Aligned::zeros(t * h);
        time(Stage::MoeBackward, || {
            self.moe_backward(
                o.moe,
                &c.ffn_in,
                &c.moe,
                &d_ffn_out,
                &mut d_ffn_in,
                grads,
                t,
            )
        });
        time(Stage::HcMixBackward, || {
            self.mix_backward(
                o.hc_ffn,
                &c.mid,
                &c.ffn_mix,
                &d_ffn_in,
                Some(&d_ffn_inject),
                &mut d_mid,
                grads,
                t,
            )
        });

        // The token mixer's out-mix and the token mixer.
        let (d_attn_out, d_attn_inject) = time(Stage::HcCombineBackward, || {
            self.combine_backward(&d_mid, &c.attn_out, &c.attn_mix.inject, t)
        });
        let mut dx = d_mid;
        let mut d_attn_in = Aligned::zeros(t * h);
        match (o.mixer, &c.mixer) {
            (TokenMixer::Full(f), MixerCache::Full(ac)) => {
                time(Stage::FullAttentionBackward, || {
                    self.attn_backward(f, &c.attn_in, ac, &d_attn_out, &mut d_attn_in, grads, t)
                })
            }
            (TokenMixer::Linear(lo), MixerCache::Linear(dc)) => {
                time(Stage::DeltaRuleBackward, || {
                    self.delta_backward(lo, &c.attn_in, dc, &d_attn_out, &mut d_attn_in, grads, t)
                })
            }
            _ => unreachable!("a block's cache is of its own kind"),
        }
        time(Stage::HcMixBackward, || {
            self.mix_backward(
                o.hc_attn,
                x_in,
                &c.attn_mix,
                &d_attn_in,
                Some(&d_attn_inject),
                &mut dx,
                grads,
                t,
            )
        });
        if let (Some(p), Some(po), Some(pc)) = (self.cfg.ple.as_ref(), o.ple, c.ple.as_ref()) {
            time(Stage::PleBackward, || {
                self.ple_backward(p, po, x, pc, &mut dx, grads, t)
            });
        }
        dx
    }

    #[cfg(test)]
    /// The next-token logits at every position, `[t, vocab]`.
    pub fn logits(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (t, h, hc, w) = (tokens.len(), cfg.hidden, cfg.hc, cfg.wide());
        let embd = self.tensor(self.layout.tok_embd, cfg.vocab * h);
        let mut x = time(Stage::Embed, || {
            let mut x = Aligned::zeros(t * w);
            for (i, &tok) in tokens.iter().enumerate() {
                let row = &embd[tok as usize * h..(tok as usize + 1) * h];
                for s in 0..hc {
                    x[i * w + s * h..i * w + (s + 1) * h].copy_from_slice(row);
                }
            }
            x
        });
        for l in 0..cfg.layers {
            x = self.layer_forward(l, &x, tokens, t).out;
        }
        let (collapsed, _) = self.mix(self.layout.head, &x, t);
        let mut logits = Aligned::zeros(t * cfg.vocab);
        matmul(
            &mut logits,
            &collapsed,
            self.tensor(self.layout.output, cfg.vocab * h),
            t,
            h,
            cfg.vocab,
        );
        logits.to_vec()
    }

    /// The architecture's metadata, as the engine reads it (`qwen4exp.*`).
    pub fn file_metadata(&self) -> Vec<(String, orangu::gguf::GgufValue)> {
        use orangu::gguf::GgufValue as V;
        let cfg = &self.cfg;
        let key = |k: &str| format!("qwen4exp.{k}");
        let u = |v: usize| V::U32(v as u32);
        let mut m = vec![
            (
                "general.architecture".to_string(),
                V::String("qwen4exp".into()),
            ),
            (key("block_count"), u(cfg.layers)),
            (key("context_length"), u(cfg.context)),
            (key("embedding_length"), u(cfg.hidden)),
            (key("vocab_size"), u(cfg.vocab)),
            (key("attention.head_count"), u(cfg.heads)),
            (key("attention.head_count_kv"), u(cfg.kv_heads)),
            (
                key("rope.dimension_sections"),
                V::Array(vec![
                    V::I32((cfg.rope_dim / 2) as i32),
                    V::I32(0),
                    V::I32(0),
                    V::I32(0),
                ]),
            ),
            (key("rope.freq_base"), V::F32(cfg.rope_base)),
            (key("attention.layer_norm_rms_epsilon"), V::F32(cfg.eps)),
            (key("expert_count"), u(cfg.experts)),
            (key("expert_used_count"), u(cfg.experts_used)),
            (key("attention.key_length"), u(cfg.head_dim)),
            (key("attention.value_length"), u(cfg.head_dim)),
            (key("expert_feed_forward_length"), u(cfg.expert_ffn)),
            (key("expert_shared_feed_forward_length"), u(cfg.shared_ffn)),
            (key("ssm.conv_kernel"), u(cfg.conv)),
            (key("ssm.state_size"), u(cfg.ssm_dim)),
            (key("ssm.group_count"), u(cfg.ssm_k_heads)),
            (key("ssm.time_step_rank"), u(cfg.ssm_v_heads)),
            (key("ssm.inner_size"), u(cfg.value_dim())),
            (key("full_attention_interval"), u(cfg.interval)),
            (key("rope.dimension_count"), u(cfg.rope_dim)),
            (key("hyper_connection.count"), u(cfg.hc)),
            (key("hyper_connection.low_rank"), u(cfg.hc_rank)),
            (key("attention.indexer.head_count"), u(cfg.idx_heads)),
            (key("attention.indexer.key_length"), u(cfg.idx_dim)),
            (key("attention.indexer.top_k"), u(cfg.idx_top_k)),
            (
                key("attention.compress_ratios"),
                V::Array(
                    (0..cfg.layers)
                        .map(|l| V::I32(if cfg.is_full(l) { cfg.ratio as i32 } else { 0 }))
                        .collect(),
                ),
            ),
        ];
        if let Some(p) = &cfg.ple {
            let u64s = |v: &[u64]| V::Array(v.iter().map(|&x| V::U64(x)).collect());
            m.extend([
                (key("ple.layers"), V::Array(vec![V::I32(p.layer as i32)])),
                (key("ple.ngram_size"), u(p.ngram)),
                (key("ple.heads_per_ngram"), u(p.heads_per_ngram)),
                (key("ple.conv_kernel"), u(p.kernel)),
                (key("ple.eos_token_id"), u(p.eos as usize)),
                (
                    key("embedding_length_per_layer_input"),
                    u(cfg.hidden / p.heads()),
                ),
                (key("ple.layer_multipliers"), u64s(&p.multipliers)),
                (key("ple.head_offsets"), u64s(&p.offsets())),
                (key("ple.head_vocab_sizes"), u64s(&p.sizes)),
            ]);
        }
        m
    }
    /// Every tensor as the file names and shapes it. The delta rule's decay
    /// rate is trained as its logarithm `ssm_a_log` and written as `ssm_a`.
    pub fn file_specs(&self) -> Vec<(String, Vec<u64>)> {
        self.layout
            .specs
            .iter()
            .map(|s| match s.name.strip_suffix("ssm_a_log") {
                Some(prefix) => (format!("{prefix}ssm_a"), s.dims.clone()),
                None => (s.name.clone(), s.dims.clone()),
            })
            .collect()
    }

    /// The values of the file's tensor `name`: `ssm_a = −exp(A_log)`, every
    /// other tensor as trained.
    pub fn file_values(&self, name: &str) -> Option<Vec<f32>> {
        let (trained, decay) = match name.strip_suffix("ssm_a") {
            Some(prefix) => (format!("{prefix}ssm_a_log"), true),
            None => (name.to_string(), false),
        };
        let s = self.layout.specs.iter().find(|s| s.name == trained)?;
        let values = &self.params[s.offset..s.offset + s.len()];
        Some(if decay {
            values.iter().map(|v| -v.exp()).collect()
        } else {
            values.to_vec()
        })
    }

    /// Runs the whole network over one sequence: the mean cross-entropy of
    /// predicting `targets` and the experts' load-balancing term, with the
    /// gradient of their sum added into `grads` when one is supplied.
    pub fn forward_backward(
        &self,
        tokens: &[u32],
        targets: &[u32],
        grads: Option<&mut [f32]>,
    ) -> Loss {
        let cfg = &self.cfg;
        let (t, h, hc, w) = (tokens.len(), cfg.hidden, cfg.hc, cfg.wide());
        let embd = self.tensor(self.layout.tok_embd, cfg.vocab * h);
        // The streams start as `hc` copies of the token's embedding.
        let mut x = Aligned::zeros(t * w);
        for (i, &tok) in tokens.iter().enumerate() {
            let row = &embd[tok as usize * h..(tok as usize + 1) * h];
            for s in 0..hc {
                x[i * w + s * h..i * w + (s + 1) * h].copy_from_slice(row);
            }
        }
        let mut inputs: Vec<Aligned> = Vec::with_capacity(cfg.layers + 1);
        let mut aux = 0.0f32;
        let mut idx = 0.0f32;
        // A block's intermediates are kept for its backward while the
        // process's budget for them has room; the rest are recomputed.
        let mut kept: Vec<Option<Kept>> = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            inputs.push(x.clone());
            let mut c = self.layer_forward(l, &x, tokens, t);
            aux += c.moe.aux;
            if let MixerCache::Full(ac) = &c.mixer {
                idx += ac.idx_loss;
            }
            x = std::mem::take(&mut c.out);
            kept.push(if grads.is_some() { Kept::new(c) } else { None });
        }
        let (collapsed, head_mix) = time(Stage::HcMix, || self.mix(self.layout.head, &x, t));
        let out_w = self.tensor(self.layout.output, cfg.vocab * h);
        let want = grads.is_some();
        let (logits, loss) = time(Stage::OutputHead, || {
            let mut logits = Aligned::zeros(t * cfg.vocab);
            matmul(&mut logits, &collapsed, out_w, t, h, cfg.vocab);
            let inv = 1.0 / t as f32;
            let mut loss = 0.0f64;
            for (row, &target) in logits.chunks_mut(cfg.vocab).zip(targets) {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                loss -= ((row[target as usize] / sum).max(f32::MIN_POSITIVE) as f64).ln();
                if want {
                    for v in row.iter_mut() {
                        *v *= inv / sum;
                    }
                    row[target as usize] -= inv;
                }
            }
            (logits, loss)
        });
        let loss = Loss {
            ce: (loss / t as f64) as f32,
            aux,
            indexer: idx,
        };
        let Some(grads) = grads else { return loss };

        let d_collapsed = time(Stage::OutputHeadBackward, || {
            matmul_add_dw(
                grad_slice(grads, self.layout.output, cfg.vocab * h),
                &logits,
                &collapsed,
                t,
                h,
                cfg.vocab,
            );
            let mut d_collapsed = Aligned::zeros(t * h);
            matmul_add_dx(&mut d_collapsed, &logits, out_w, t, h, cfg.vocab);
            d_collapsed
        });
        let mut dx = Aligned::zeros(t * w);
        time(Stage::HcMixBackward, || {
            self.mix_backward(
                self.layout.head,
                &x,
                &head_mix,
                &d_collapsed,
                None,
                &mut dx,
                grads,
                t,
            )
        });
        for l in (0..cfg.layers).rev() {
            let cache = kept.pop().flatten();
            dx = self.layer_backward(
                l,
                &inputs[l],
                tokens,
                &dx,
                t,
                grads,
                cache.as_ref().map(|k| &k.cache),
            );
        }
        time(Stage::EmbedBackward, || {
            let d_embd = grad_slice(grads, self.layout.tok_embd, cfg.vocab * h);
            for (i, &tok) in tokens.iter().enumerate() {
                let row = &mut d_embd[tok as usize * h..(tok as usize + 1) * h];
                for s in 0..hc {
                    for j in 0..h {
                        row[j] += dx[i * w + s * h + j];
                    }
                }
            }
        });
        loss
    }
}

/// One query's share of the indexer loss: its predicted and target
/// distributions over the blocks, and the loss's terms in block order.
type QueryLoss = (Vec<f32>, Vec<f32>, Vec<f64>);

/// The smallest sum the chosen experts' probabilities are divided by.
const WEIGHT_FLOOR: f32 = 6.103_515_6e-5;

/// What a pass costs: the cross-entropy of the next token, and the experts'
/// load-balancing term. Training follows the gradient of their sum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loss {
    pub ce: f32,
    pub aux: f32,
    /// The sparse-attention indexer's loss, which trains the indexer alone.
    pub indexer: f32,
}

impl Loss {
    /// What the gradient is of.
    #[cfg(test)]
    pub fn objective(&self) -> f32 {
        self.ce + self.aux + self.indexer
    }
}

/// One expert's intermediates over the tokens routed to it.
struct ExpertCache {
    xin: Aligned,
    gate: Aligned,
    up: Aligned,
    act: Aligned,
    out: Aligned,
}

/// The routed feed-forward's intermediates.
struct MoeCache {
    /// `[t, experts]` — the router's probabilities.
    probs: Aligned,
    /// Per token, the chosen experts and their renormalized weights.
    chosen: Vec<Vec<(usize, f32)>>,
    /// Per token, the chosen probabilities' sum before the floor.
    sums: Vec<f32>,
    /// Per expert, its tokens: `(token, slot in that token's choice)`.
    members: Vec<Vec<(usize, usize)>>,
    experts: Vec<ExpertCache>,
    s_gate: Aligned,
    s_up: Aligned,
    s_act: Aligned,
    s_out: Aligned,
    /// Per token, `x · shared_gate`, before the sigmoid.
    s_score: Vec<f32>,
    /// Per expert, its share of the routing slots.
    share: Vec<f32>,
    aux: f32,
}

/// A SwiGLU's backward: `out = (silu(x·G) ⊙ x·U)·D` with the matrices
/// `(G, U, D)` and their gradients `grads`, widths `(hidden, inner)`. Adds
/// the gradient of `x` into `dx` and of the three matrices into `grads`.
#[allow(clippy::too_many_arguments)]
fn swiglu_backward(
    (w_gate, w_up, w_down): (&[f32], &[f32], &[f32]),
    (g_gate, g_up, g_down): (&mut [f32], &mut [f32], &mut [f32]),
    (h, f): (usize, usize),
    x: &[f32],
    (gate, up, act): (&[f32], &[f32], &[f32]),
    d_out: &[f32],
    dx: &mut [f32],
    t: usize,
) {
    matmul_add_dw(g_down, d_out, act, t, f, h);
    let mut d_act = Aligned::zeros(t * f);
    matmul_add_dx(&mut d_act, d_out, w_down, t, f, h);
    let mut d_gate = Aligned::zeros(t * f);
    let mut d_up = Aligned::zeros(t * f);
    d_gate
        .par_chunks_mut(f)
        .zip(d_up.par_chunks_mut(f))
        .zip(d_act.par_chunks(f))
        .zip(gate.par_chunks(f).zip(up.par_chunks(f)))
        .for_each(|(((d_gate, d_up), d_act), (gate, up))| {
            for i in 0..d_act.len() {
                let g = gate[i];
                let s = sigmoid(g);
                d_up[i] = d_act[i] * silu(g);
                d_gate[i] = d_act[i] * up[i] * s * (1.0 + g * (1.0 - s));
            }
        });
    for (grad, w, dy) in [(g_gate, w_gate, &d_gate), (g_up, w_up, &d_up)] {
        matmul_add_dw(grad, dy, x, t, h, f);
        matmul_add_dx(dx, dy, w, t, h, f);
    }
}

/// One routed expert's gate, up and down gradients.
type ExpertGrads<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [f32]);

/// What a routed expert's backward gives back: its tokens' input gradient
/// and the gradients of their weights.
type ExpertGiven = (Aligned, Vec<f32>);

/// `body`'s result and the seconds it took.
fn timed<T>(body: impl FnOnce() -> T) -> (T, f64) {
    let started = std::time::Instant::now();
    let out = body();
    (out, started.elapsed().as_secs_f64())
}

/// Which experts go to the card: from the first, until the card holds
/// `share` of the routed rows.
fn split_experts(members: &[Vec<(usize, usize)>], share: f64) -> Vec<bool> {
    let total: usize = members.iter().map(Vec::len).sum();
    let goal = (total as f64 * share).round() as usize;
    let mut taken = 0;
    members
        .iter()
        .map(|rows| {
            let take = taken < goal;
            if take {
                taken += rows.len();
            }
            take
        })
        .collect()
}

/// Where each expert's rows start when every expert's tokens are stacked
/// in expert order, and how many rows there are in all.
fn stack_starts(members: &[Vec<(usize, usize)>]) -> (Vec<usize>, usize) {
    let mut starts = Vec::with_capacity(members.len());
    let mut total = 0;
    for rows in members {
        starts.push(total);
        total += rows.len();
    }
    (starts, total)
}

/// Disjoint mutable views of `grads` at `(offset, len)` each, in the order
/// asked for. Ranges that overlap are a layout error.
fn regions<const N: usize>(grads: &mut [f32], at: [(usize, usize); N]) -> [&mut [f32]; N] {
    let mut order: [usize; N] = std::array::from_fn(|i| i);
    order.sort_by_key(|&i| at[i].0);
    let mut out: [Option<&mut [f32]>; N] = std::array::from_fn(|_| None);
    let mut rest = grads;
    let mut taken = 0;
    for &i in &order {
        let (offset, len) = at[i];
        assert!(offset >= taken, "gradient regions overlap");
        let (_, tail) = rest.split_at_mut(offset - taken);
        let (region, tail) = tail.split_at_mut(len);
        out[i] = Some(region);
        rest = tail;
        taken = offset + len;
    }
    out.map(|r| r.expect("every region is taken"))
}

/// [`rope_partial`] on one row at position `pos`.
fn rope_at(
    row: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    base: f32,
    pos: usize,
    inverse: bool,
) {
    let half = rope_dim / 2;
    for i in 0..half {
        let freq = base.powf(-2.0 * i as f32 / rope_dim as f32);
        let (mut sin, cos) = (pos as f32 * freq).sin_cos();
        if inverse {
            sin = -sin;
        }
        for head in row.chunks_exact_mut(head_dim) {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// Which of a query's `visible` positions it attends, ascending: all of
/// them up to `width`; past that, the incomplete tail block and the
/// `width` best positions by their block's score (`block_scores[b]` for
/// positions `b·ratio..(b+1)·ratio`), ties to the earlier position. The
/// engine's `select_positions`.
fn select_positions(
    block_scores: &[f32],
    visible: usize,
    ratio: usize,
    width: usize,
) -> Vec<usize> {
    if visible <= width {
        return (0..visible).collect();
    }
    let tail = visible / ratio * ratio;
    let score = |j: usize| {
        if j >= tail {
            f32::INFINITY
        } else {
            block_scores[j / ratio]
        }
    };
    let mut order: Vec<usize> = (0..visible).collect();
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
    order.truncate(width);
    order.sort_unstable();
    order
}

/// Attention over each query's own `selection` of positions (GQA, scale
/// `1/sqrt(head_dim)`): `[t, heads·head_dim]` out, and the probabilities.
fn attention_selected(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    selection: &[Vec<usize>],
    (heads, kv_heads, hd): (usize, usize, usize),
) -> (Aligned, Vec<Vec<f32>>) {
    let t = selection.len();
    let (qd, kvd, group) = (heads * hd, kv_heads * hd, heads / kv_heads);
    let scale = 1.0 / (hd as f32).sqrt();
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..t * heads)
        .into_par_iter()
        .map(|th| {
            let (i, h) = (th / heads, th % heads);
            let kh = h / group;
            let qv = &q[i * qd + h * hd..i * qd + (h + 1) * hd];
            let sel = &selection[i];
            let mut p: Vec<f32> = sel
                .iter()
                .map(|&j| {
                    qv.iter()
                        .zip(&k[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in p.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in p.iter_mut() {
                *x /= sum;
            }
            let mut out = vec![0.0f32; hd];
            for (&j, &pj) in sel.iter().zip(&p) {
                for (o, &vv) in out
                    .iter_mut()
                    .zip(&v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
                {
                    *o += pj * vv;
                }
            }
            (out, p)
        })
        .collect();
    let mut ctx = Aligned::zeros(t * qd);
    let mut probs = Vec::with_capacity(t * heads);
    for (th, (out, p)) in rows.into_iter().enumerate() {
        let (i, h) = (th / heads, th % heads);
        ctx[i * qd + h * hd..i * qd + (h + 1) * hd].copy_from_slice(&out);
        probs.push(p);
    }
    (ctx, probs)
}

/// [`attention_selected`]'s backward: the gradients of `q`, `k` and `v`.
#[allow(clippy::type_complexity)]
fn attention_selected_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    selection: &[Vec<usize>],
    probs: &[Vec<f32>],
    d_ctx: &[f32],
    (heads, kv_heads, hd): (usize, usize, usize),
) -> (Aligned, Aligned, Aligned) {
    let t = selection.len();
    let (qd, kvd, group) = (heads * hd, kv_heads * hd, heads / kv_heads);
    let scale = 1.0 / (hd as f32).sqrt();
    // By query: its own gradient, and each score's gradient `ds`, per head.
    let mut dq = Aligned::zeros(t * qd);
    let ds: Vec<Vec<Vec<f32>>> = dq
        .par_chunks_mut(qd)
        .enumerate()
        .map(|(i, dq)| {
            let sel = &selection[i];
            (0..heads)
                .map(|h| {
                    let kh = h / group;
                    let p = &probs[i * heads + h];
                    let dc = &d_ctx[i * qd + h * hd..i * qd + (h + 1) * hd];
                    let dp: Vec<f32> = sel
                        .iter()
                        .map(|&j| {
                            dc.iter()
                                .zip(&v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
                                .map(|(a, b)| a * b)
                                .sum()
                        })
                        .collect();
                    let inner: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                    let ds: Vec<f32> = p
                        .iter()
                        .zip(&dp)
                        .map(|(&p, &dp)| p * (dp - inner) * scale)
                        .collect();
                    let row = &mut dq[h * hd..(h + 1) * hd];
                    for (&j, &ds) in sel.iter().zip(&ds) {
                        for d in 0..hd {
                            row[d] += ds * k[j * kvd + kh * hd + d];
                        }
                    }
                    ds
                })
                .collect()
        })
        .collect();

    // By key: every query that selected it, in query order, and within a
    // query its heads in order — each element takes its terms in the order
    // a single pass over the queries gives them.
    let mut readers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); t];
    for (i, sel) in selection.iter().enumerate() {
        for (n, &j) in sel.iter().enumerate() {
            readers[j].push((i, n));
        }
    }
    let mut dk = Aligned::zeros(t * kvd);
    let mut dv = Aligned::zeros(t * kvd);
    dk.par_chunks_mut(kvd)
        .zip(dv.par_chunks_mut(kvd))
        .zip(readers.par_iter())
        .for_each(|((dk, dv), readers)| {
            for &(i, n) in readers {
                for h in 0..heads {
                    let kh = h / group;
                    let p = probs[i * heads + h][n];
                    let ds = ds[i][h][n];
                    let dc = &d_ctx[i * qd + h * hd..i * qd + (h + 1) * hd];
                    let qv = &q[i * qd + h * hd..i * qd + (h + 1) * hd];
                    for d in 0..hd {
                        dv[kh * hd + d] += p * dc[d];
                        dk[kh * hd + d] += ds * qv[d];
                    }
                }
            }
        });
    (dq, dk, dv)
}

/// The indexer's intermediates for one full-attention block.
#[derive(Default)]
struct IndexerCache {
    /// `[blocks, idx_dim]` — each complete block's mean key, its norm's
    /// scale, and the normed, rotated block key.
    pooled: Aligned,
    pooled_rms: Aligned,
    blocks: Aligned,
    /// `[t, idx_heads · idx_dim]` — the raw queries, their norms' scales,
    /// and the normed, rotated queries.
    q_raw: Aligned,
    q_rms: Aligned,
    queries: Aligned,
    /// Per query, the softmax of its block scores and the attention's own
    /// mass per block — the indexer's prediction and target.
    pred: Vec<Vec<f32>>,
    target: Vec<Vec<f32>>,
    /// Queries the loss is averaged over.
    counted: usize,
}

/// Each `h`-wide stream of `x` (`[rows, hc·h]`) RMS-normed alone, times a
/// gamma spanning every stream: the output and each stream's `1/rms`.
fn grouped_norm(x: &[f32], gamma: &[f32], hc: usize, h: usize, eps: f32) -> (Aligned, Aligned) {
    let mut out = Aligned::zeros(x.len());
    let mut scales = Aligned::zeros(x.len() / h);
    out.par_chunks_mut(h)
        .zip(scales.par_iter_mut())
        .zip(x.par_chunks(h))
        .enumerate()
        .for_each(|(row, ((o, s), xs))| {
            let c = row % hc;
            let mean = xs.iter().map(|v| v * v).sum::<f32>() / h as f32;
            *s = 1.0 / (mean + eps).sqrt();
            for ((ov, &v), &g) in o.iter_mut().zip(xs).zip(&gamma[c * h..(c + 1) * h]) {
                *ov = v * *s * g;
            }
        });
    (out, scales)
}

/// [`grouped_norm`]'s backward: adds into `dx` and `d_gamma`.
fn grouped_norm_backward(
    dx: &mut [f32],
    d_gamma: &mut [f32],
    dy: &[f32],
    x: &[f32],
    gamma: &[f32],
    scales: &[f32],
    (hc, h): (usize, usize),
) {
    for (row, ((dxr, dyr), xr)) in dx
        .chunks_mut(h)
        .zip(dy.chunks(h))
        .zip(x.chunks(h))
        .enumerate()
    {
        let c = row % hc;
        let g = &gamma[c * h..(c + 1) * h];
        let s = scales[row];
        let inner: f32 = (0..h).map(|j| g[j] * dyr[j] * xr[j]).sum();
        let coeff = s * s * s * inner / h as f32;
        for j in 0..h {
            dxr[j] += s * g[j] * dyr[j] - coeff * xr[j];
            d_gamma[c * h + j] += dyr[j] * xr[j] * s;
        }
    }
}

/// The PLE gate: `sigmoid(sign(s) · sqrt(max(|s|, 1e-6)))`.
fn ple_gate(s: f32) -> f32 {
    sigmoid(s.signum() * s.abs().max(1e-6).sqrt())
}

/// [`ple_gate`]'s derivative.
fn ple_gate_grad(s: f32) -> f32 {
    let g = ple_gate(s);
    if s.abs() <= 1e-6 {
        0.0
    } else {
        g * (1.0 - g) * 0.5 / s.abs().sqrt()
    }
}

/// The n-gram hash embedding's intermediates.
struct PleCache {
    rows: Vec<usize>,
    /// `[t, hidden]` — the gathered rows, side by side.
    embd: Aligned,
    /// `[t, wide]` the raw key, `[t, hidden]` the value.
    key: Aligned,
    value: Aligned,
    key_n: Aligned,
    key_s: Aligned,
    query: Aligned,
    query_s: Aligned,
    /// `[t, hc]` — each stream's score.
    score: Vec<f32>,
    /// `[t, wide]` — the gated value in every stream, and its norm.
    gated: Aligned,
    normed: Aligned,
    normed_s: Aligned,
    /// `[t, wide]` — the convolution, before its `silu`.
    conv: Aligned,
    /// `[t, wide]` — the streams after the embedding is added.
    out: Aligned,
}

/// One value head's gradients from the recurrence, a token at a time, or a
/// column block's share of them over a chunk.
struct HeadGrads {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
}

/// The columns of a head's state one block holds: two eight-lane vectors, so a
/// block's running sums stay in registers and its state in the first-level
/// cache, and a head splits into several blocks. Any width runs, through the
/// portable kernels, when the head is not a multiple of it.
const BLOCK: usize = 16;

fn column_block(dim: usize) -> usize {
    if dim.is_multiple_of(BLOCK) {
        BLOCK
    } else {
        dim
    }
}

/// One block's working set in the delta rule's backward, kept across its
/// chunks.
struct BlockWork {
    g_state: Vec<f32>,
    before: Vec<f32>,
    scratch: Vec<f32>,
    share: HeadGrads,
}

/// `delta_step_backward`'s scratch, in floats per row of the state.
const STEP_SCRATCH: usize = 24;

/// Whether the vector kernels take a block `cols` wide over `rows` rows.
#[inline]
fn vector_block(rows: usize, cols: usize) -> bool {
    cols == BLOCK && rows.is_multiple_of(8)
}

/// One token of one head's gated delta rule, on a `[k.len(), out.len()]`
/// block of its state `S` (row `b` the key index, column `a` the value
/// index): `S ← decay·S`, `d ← beta·(v − kᵀS)`, `S ← S + k dᵀ`,
/// `out ← qᵀS`. `out` arrives zeroed.
fn delta_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    out: &mut [f32],
) {
    if vector_block(k.len(), out.len()) {
        block::step(state, q, k, v, beta, decay, out);
        return;
    }
    delta_step_portable(state, q, k, v, beta, decay, out);
}

/// [`delta_step`] for any block, summed in the engine's `delta_head_step`
/// order.
fn delta_step_portable(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    out: &mut [f32],
) {
    let dim = out.len();
    let mut sk = vec![0.0f32; dim];
    for (row, &kb) in state.chunks_exact_mut(dim).zip(k) {
        for (s, ska) in row.iter_mut().zip(sk.iter_mut()) {
            *s *= decay;
            *ska += kb * *s;
        }
    }
    let d: Vec<f32> = v.iter().zip(&sk).map(|(&va, &s)| beta * (va - s)).collect();
    for ((row, &ki), &qi) in state.chunks_exact_mut(dim).zip(k).zip(q) {
        for ((s, &dj), o) in row.iter_mut().zip(&d).zip(out.iter_mut()) {
            *s += ki * dj;
            *o += qi * *s;
        }
    }
}

/// [`delta_step`]'s backward on a block of columns, from the block's state
/// before the step, the gradient of its output, and `g_state` — the
/// gradient of the block after the step, which becomes that of the block
/// before it. Writes the block's share of the `q` and `k` gradients and its
/// columns of `v`'s; returns its share of the `beta` and `decay` ones.
/// `scratch` holds [`STEP_SCRATCH`] floats per row.
#[allow(clippy::too_many_arguments)]
fn delta_step_backward(
    before: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    d_out: &[f32],
    g_state: &mut [f32],
    grads: (&mut [f32], &mut [f32], &mut [f32]),
    scratch: &mut [f32],
) -> (f32, f32) {
    if vector_block(k.len(), v.len()) {
        return block::step_backward(before, q, k, v, beta, decay, d_out, g_state, grads, scratch);
    }
    delta_step_backward_portable(before, q, k, v, beta, decay, d_out, g_state, grads)
}

#[allow(clippy::too_many_arguments)]
fn delta_step_backward_portable(
    before: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    d_out: &[f32],
    g_state: &mut [f32],
    (dq, dk, dv): (&mut [f32], &mut [f32], &mut [f32]),
) -> (f32, f32) {
    let cols = v.len();
    // The step's intermediates again: S1 = decay·S0, kᵀS1, d.
    let mut sk = vec![0.0f32; cols];
    for (row, &kb) in before.chunks_exact(cols).zip(k) {
        for (&s0, ska) in row.iter().zip(sk.iter_mut()) {
            *ska += kb * (s0 * decay);
        }
    }
    let d: Vec<f32> = v.iter().zip(&sk).map(|(&va, &s)| beta * (va - s)).collect();
    // out = qᵀS2, S2 = S1 + k dᵀ: into q, the state, and k and d.
    let mut dd = vec![0.0f32; cols];
    for (i, (row, g_row)) in before
        .chunks_exact(cols)
        .zip(g_state.chunks_exact_mut(cols))
        .enumerate()
    {
        let mut acc = 0.0f32;
        for ((&s0, &dj), &da) in row.iter().zip(&d).zip(d_out) {
            acc += da * (s0 * decay + k[i] * dj);
        }
        dq[i] = acc;
        for (g, &da) in g_row.iter_mut().zip(d_out) {
            *g += q[i] * da;
        }
        let mut acc = 0.0f32;
        for ((&g, &dj), ddj) in g_row.iter().zip(&d).zip(dd.iter_mut()) {
            acc += g * dj;
            *ddj += k[i] * g;
        }
        dk[i] = acc;
    }
    // d = beta·(v − kᵀS1).
    let mut d_beta = 0.0f32;
    let mut dsk = vec![0.0f32; cols];
    for a in 0..cols {
        d_beta += dd[a] * (v[a] - sk[a]);
        dv[a] = beta * dd[a];
        dsk[a] = -beta * dd[a];
    }
    // kᵀS1 into S1's gradient and k's, then S1 = decay·S0.
    let mut d_decay = 0.0f32;
    for (b, (row, g_row)) in before
        .chunks_exact(cols)
        .zip(g_state.chunks_exact_mut(cols))
        .enumerate()
    {
        let mut acc = 0.0f32;
        for ((&s0, g), &dska) in row.iter().zip(g_row.iter_mut()).zip(&dsk) {
            acc += dska * (s0 * decay);
            *g += k[b] * dska;
        }
        dk[b] += acc;
        for (g, &s0) in g_row.iter_mut().zip(row) {
            d_decay += *g * s0;
            *g *= decay;
        }
    }
    (d_beta, d_decay)
}

/// The delta rule's step and its backward on a block of [`BLOCK`] columns,
/// two eight-lane vectors a row, with every running sum in a register.
mod block {
    use crate::vector::V8;

    /// Running sums kept four rows apart, so consecutive rows' multiply-adds
    /// do not wait on each other; folded `(0 + 1) + (2 + 3)`.
    #[derive(Clone, Copy)]
    struct Four([V8; 4]);

    impl Four {
        #[inline(always)]
        fn new() -> Four {
            Four([V8::splat(0.0); 4])
        }
        #[inline(always)]
        fn total(self) -> V8 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }

    /// Writes the lane sums of `rows` vectors, stored eight floats each in
    /// `parts`, into `out`; adds them when `add`.
    fn reduce_rows(parts: &[f32], out: &mut [f32], add: bool) {
        for (group, dst) in parts
            .as_chunks::<64>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<8>().0)
        {
            let mut s = V8::sums8(std::array::from_fn(|i| V8::load(&group[i * 8..])));
            if add {
                s = s + V8::load(dst);
            }
            s.store(dst);
        }
    }

    pub(super) fn step(
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: f32,
        decay: f32,
        out: &mut [f32],
    ) {
        let rows = k.len();
        assert!(state.len() == rows * 16 && q.len() == rows && v.len() == 16 && out.len() == 16);
        let dv = V8::splat(decay);
        let (mut sk0, mut sk1) = (Four::new(), Four::new());
        for b4 in (0..rows).step_by(4) {
            for j in 0..4 {
                let b = b4 + j;
                // SAFETY: row `b` of `rows` rows of 16, `rows` a multiple
                // of eight.
                unsafe {
                    let s0 = V8::at(state, b * 16) * dv;
                    let s1 = V8::at(state, b * 16 + 8) * dv;
                    s0.put(state, b * 16);
                    s1.put(state, b * 16 + 8);
                    let kv = V8::splat(*k.get_unchecked(b));
                    sk0.0[j] = kv.mul_add(s0, sk0.0[j]);
                    sk1.0[j] = kv.mul_add(s1, sk1.0[j]);
                }
            }
        }
        let bv = V8::splat(beta);
        let d0 = bv * (V8::load(v) - sk0.total());
        let d1 = bv * (V8::load(&v[8..]) - sk1.total());
        let (mut o0, mut o1) = (Four::new(), Four::new());
        for b4 in (0..rows).step_by(4) {
            for j in 0..4 {
                let b = b4 + j;
                // SAFETY: as above.
                unsafe {
                    let kv = V8::splat(*k.get_unchecked(b));
                    let qv = V8::splat(*q.get_unchecked(b));
                    let s0 = kv.mul_add(d0, V8::at(state, b * 16));
                    let s1 = kv.mul_add(d1, V8::at(state, b * 16 + 8));
                    s0.put(state, b * 16);
                    s1.put(state, b * 16 + 8);
                    o0.0[j] = qv.mul_add(s0, o0.0[j]);
                    o1.0[j] = qv.mul_add(s1, o1.0[j]);
                }
            }
        }
        (V8::load(out) + o0.total()).store(out);
        (V8::load(&out[8..]) + o1.total()).store(&mut out[8..]);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn step_backward(
        before: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: f32,
        decay: f32,
        d_out: &[f32],
        g_state: &mut [f32],
        (dq, dk, dv): (&mut [f32], &mut [f32], &mut [f32]),
        scratch: &mut [f32],
    ) -> (f32, f32) {
        let rows = k.len();
        assert!(before.len() == rows * 16 && g_state.len() == rows * 16 && q.len() == rows);
        assert!(scratch.len() >= rows * super::STEP_SCRATCH && dq.len() >= rows);
        assert!(dk.len() >= rows && dv.len() >= 16 && v.len() >= 16 && d_out.len() >= 16);
        let (q_parts, rest) = scratch.split_at_mut(rows * 8);
        let (k_parts, rest) = rest.split_at_mut(rows * 8);
        let k_more = &mut rest[..rows * 8];
        let dc = V8::splat(decay);

        // S1 = decay·S0, kᵀS1.
        let (mut sk0, mut sk1) = (Four::new(), Four::new());
        for b4 in (0..rows).step_by(4) {
            for j in 0..4 {
                let b = b4 + j;
                // SAFETY: row `b` of `rows` rows of 16, `rows` a multiple
                // of eight.
                unsafe {
                    let kv = V8::splat(*k.get_unchecked(b));
                    sk0.0[j] = kv.mul_add(V8::at(before, b * 16) * dc, sk0.0[j]);
                    sk1.0[j] = kv.mul_add(V8::at(before, b * 16 + 8) * dc, sk1.0[j]);
                }
            }
        }
        let (sk0, sk1) = (sk0.total(), sk1.total());
        let bv = V8::splat(beta);
        let (v0, v1) = (V8::load(v), V8::load(&v[8..]));
        let d0 = bv * (v0 - sk0);
        let d1 = bv * (v1 - sk1);
        let (o0, o1) = (V8::load(d_out), V8::load(&d_out[8..]));

        // out = qᵀS2 with S2 = S1 + k dᵀ: into q, the state, and k and d.
        let (mut dd0, mut dd1) = (Four::new(), Four::new());
        for b4 in (0..rows).step_by(4) {
            for j in 0..4 {
                let b = b4 + j;
                // SAFETY: row `b` of the state and of the eight-a-row parts.
                unsafe {
                    let kv = V8::splat(*k.get_unchecked(b));
                    let qv = V8::splat(*q.get_unchecked(b));
                    let s20 = kv.mul_add(d0, V8::at(before, b * 16) * dc);
                    let s21 = kv.mul_add(d1, V8::at(before, b * 16 + 8) * dc);
                    o1.mul_add(s21, o0 * s20).put(q_parts, b * 8);
                    let g0 = qv.mul_add(o0, V8::at(g_state, b * 16));
                    let g1 = qv.mul_add(o1, V8::at(g_state, b * 16 + 8));
                    g0.put(g_state, b * 16);
                    g1.put(g_state, b * 16 + 8);
                    g1.mul_add(d1, g0 * d0).put(k_parts, b * 8);
                    dd0.0[j] = kv.mul_add(g0, dd0.0[j]);
                    dd1.0[j] = kv.mul_add(g1, dd1.0[j]);
                }
            }
        }
        let (dd0, dd1) = (dd0.total(), dd1.total());
        reduce_rows(q_parts, dq, false);

        // d = beta·(v − kᵀS1).
        let d_beta = dd1.mul_add(v1 - sk1, dd0 * (v0 - sk0)).sum();
        (bv * dd0).store(dv);
        (bv * dd1).store(&mut dv[8..]);
        let nb = V8::splat(-beta);
        let (dsk0, dsk1) = (nb * dd0, nb * dd1);

        // kᵀS1 into S1's gradient and k's, then S1 = decay·S0.
        let (mut dec0, mut dec1) = (Four::new(), Four::new());
        for b4 in (0..rows).step_by(4) {
            for j in 0..4 {
                let b = b4 + j;
                // SAFETY: as above.
                unsafe {
                    let kv = V8::splat(*k.get_unchecked(b));
                    let (a0, a1) = (V8::at(before, b * 16), V8::at(before, b * 16 + 8));
                    (dc * dsk1.mul_add(a1, dsk0 * a0)).put(k_more, b * 8);
                    let g0 = kv.mul_add(dsk0, V8::at(g_state, b * 16));
                    let g1 = kv.mul_add(dsk1, V8::at(g_state, b * 16 + 8));
                    dec0.0[j] = g0.mul_add(a0, dec0.0[j]);
                    dec1.0[j] = g1.mul_add(a1, dec1.0[j]);
                    (g0 * dc).put(g_state, b * 16);
                    (g1 * dc).put(g_state, b * 16 + 8);
                }
            }
        }
        reduce_rows(k_parts, dk, false);
        reduce_rows(k_more, dk, true);
        (d_beta, (dec0.total() + dec1.total()).sum())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// What a tensor's gradient is of: the indexer's loss for the
    /// indexer's own weights, the rest of the objective for every other.
    fn along(name: &str, loss: Loss) -> f32 {
        if name.contains("indexer.") {
            loss.indexer
        } else {
            loss.ce + loss.aux
        }
    }

    pub(crate) fn tiny() -> Config {
        Config {
            vocab: 11,
            hidden: 8,
            experts: 4,
            experts_used: 2,
            expert_ffn: 6,
            shared_ffn: 5,
            aux_coef: 0.01,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            hc: 3,
            hc_rank: 5,
            context: 64,
            rope_base: 10000.0,
            rope_dim: 2,
            eps: 1e-5,
            interval: 2,
            ssm_k_heads: 1,
            ssm_v_heads: 2,
            ssm_dim: 4,
            conv: 3,
            chunk: 3,
            idx_heads: 2,
            idx_dim: 4,
            idx_top_k: 8,
            ratio: 2,
            idx_coef: 0.3,
            ple: Some(PleConfig {
                layer: 0,
                ngram: 3,
                heads_per_ngram: 2,
                kernel: 2,
                multipliers: vec![1_000_003, 2_000_029, 3_000_017],
                sizes: vec![5, 7, 11, 13],
                eos: 10,
            }),
        }
    }

    /// Every tensor's largest gradient against a central difference.
    #[test]
    fn gradients_match_finite_differences() {
        let mut model = Model::new(tiny(), 7);
        // Every matrix at one moderate scale, the out-mix weights off zero:
        // at initialization some paths are products of several small
        // projections, too faint for a finite difference to see.
        let mut rng = Rng::new(3);
        for spec in model.layout.specs.clone() {
            if spec.dims.len() == 2 {
                for v in &mut model.params[spec.offset..spec.offset + spec.len()] {
                    *v = rng.normal() * 0.5;
                }
            }
        }
        let tokens: Vec<u32> = vec![1, 5, 2, 9, 4, 4, 7, 3];
        let targets: Vec<u32> = vec![5, 2, 9, 4, 4, 7, 3, 1];
        let mut grads = Aligned::zeros(model.layout.total);
        model.forward_backward(&tokens, &targets, Some(&mut grads));
        for spec in model.layout.specs.clone() {
            let (at, analytic) = (spec.offset..spec.offset + spec.len())
                .map(|i| (i, grads[i]))
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap();
            assert!(
                analytic.abs() > 1e-6,
                "{} is not connected to the loss",
                spec.name
            );
            let eps = 1e-3f32;
            let original = model.params[at];
            model.params[at] = original + eps;
            let up = along(&spec.name, model.forward_backward(&tokens, &targets, None));
            model.params[at] = original - eps;
            let down = along(&spec.name, model.forward_backward(&tokens, &targets, None));
            model.params[at] = original;
            let numeric = (up - down) / (2.0 * eps);
            let noise = f32::EPSILON * up.abs().max(1.0) / eps;
            let difference = (numeric - analytic).abs();
            assert!(
                difference <= noise + 1e-2 * analytic.abs(),
                "{}[{}]: analytic {analytic}, numeric {numeric}",
                spec.name,
                at - spec.offset
            );
        }
    }

    /// The partial rotation is the engine's, position by position, and its
    /// inverse undoes it.
    #[test]
    fn the_partial_rotation_is_the_engines() {
        let (heads, hd, rd, base, rows) = (2, 8, 4, 10000.0f32, 5);
        let original: Vec<f32> = (0..rows * heads * hd)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let mut ours = original.clone();
        rope_partial(&mut ours, heads, hd, rd, base, false);
        for pos in 0..rows {
            let mut theirs = original[pos * heads * hd..(pos + 1) * heads * hd].to_vec();
            orangu::engine::tensor::rope_apply_inplace(&mut theirs, heads, hd, rd, pos, base);
            for (a, b) in ours[pos * heads * hd..(pos + 1) * heads * hd]
                .iter()
                .zip(&theirs)
            {
                assert!((a - b).abs() < 1e-6, "position {pos}: {a} vs {b}");
            }
        }
        rope_partial(&mut ours, heads, hd, rd, base, true);
        for (a, b) in ours.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// One step of the recurrence is the engine's, value for value.
    #[test]
    fn the_delta_step_is_the_engines() {
        let dim = 4;
        let wave =
            |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * k).sin()).collect() };
        let (q, k, v) = (wave(dim, 0.7), wave(dim, 1.3), wave(dim, 0.4));
        let mut ours = wave(dim * dim, 0.21);
        let mut theirs = ours.clone();
        let mut out_ours = vec![0.0; dim];
        let mut out_theirs = vec![0.0; dim];
        let (mut sk, mut d) = (vec![0.0; dim], vec![0.0; dim]);
        delta_step(&mut ours, &q, &k, &v, 0.6, 0.9, &mut out_ours);
        orangu::engine::arch::qwen_hybrid::delta_head_step(
            &mut theirs,
            &q,
            &k,
            &v,
            0.6,
            0.9,
            &mut out_theirs,
            &mut sk,
            &mut d,
        );
        assert_eq!(ours, theirs);
        assert_eq!(out_ours, out_theirs);
    }

    /// The vector kernels give the portable ones' step and backward, to
    /// rounding, on a block of [`BLOCK`] columns.
    #[test]
    fn the_vector_step_is_the_portable_one() {
        let rows = 32;
        let wave =
            |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * k).sin()).collect() };
        let (q, k, v, d_out) = (
            wave(rows, 0.7),
            wave(rows, 1.3),
            wave(BLOCK, 0.4),
            wave(BLOCK, 0.9),
        );
        let close = |a: &[f32], b: &[f32]| {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= 1e-5 * x.abs().max(1.0), "{x} vs {y}");
            }
        };
        let start = wave(rows * BLOCK, 0.21);
        let (mut fast, mut slow) = (start.clone(), start.clone());
        let (mut out_fast, mut out_slow) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        delta_step(&mut fast, &q, &k, &v, 0.6, 0.9, &mut out_fast);
        delta_step_portable(&mut slow, &q, &k, &v, 0.6, 0.9, &mut out_slow);
        close(&fast, &slow);
        close(&out_fast, &out_slow);

        let mut g_fast = wave(rows * BLOCK, 0.33);
        let mut g_slow = g_fast.clone();
        let mut fast = (vec![0.0; rows], vec![0.0; rows], vec![0.0; BLOCK]);
        let mut slow = fast.clone();
        let mut scratch = vec![0.0; STEP_SCRATCH * rows];
        let a = delta_step_backward(
            &start,
            &q,
            &k,
            &v,
            0.6,
            0.9,
            &d_out,
            &mut g_fast,
            (&mut fast.0, &mut fast.1, &mut fast.2),
            &mut scratch,
        );
        let b = delta_step_backward_portable(
            &start,
            &q,
            &k,
            &v,
            0.6,
            0.9,
            &d_out,
            &mut g_slow,
            (&mut slow.0, &mut slow.1, &mut slow.2),
        );
        close(&[a.0, a.1], &[b.0, b.1]);
        close(&g_fast, &g_slow);
        close(&fast.0, &slow.0);
        close(&fast.1, &slow.1);
        close(&fast.2, &slow.2);
    }

    /// The whole model on the card — every product, routed experts
    /// included — gives the processor's loss and gradients, to rounding.
    #[test]
    fn the_card_trains_what_the_processor_trains() {
        if accel::card().is_none() {
            println!("no card; skipping");
            return;
        }
        let model = Model::new(
            Config {
                experts_used: 4,
                ..tiny()
            },
            17,
        );
        let vocab = model.cfg.vocab as u32;
        let tokens: Vec<u32> = (0..40).map(|i| (i * 7) % vocab).collect();
        let targets: Vec<u32> = tokens.iter().map(|t| (t + 1) % vocab).collect();
        let on = |name: &'static str| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .thread_name(move |i| format!("{name}-{i}"))
                .build()
                .unwrap();
            pool.install(|| {
                let mut g = Aligned::zeros(model.layout.total);
                let loss = model.forward_backward(&tokens, &targets, Some(&mut g));
                (loss.objective(), g.to_vec())
            })
        };
        let (cpu_loss, cpu) = on(accel::FORCE_CPU);
        let (card_loss, card) = on(accel::FORCE_CARD);
        assert!(
            (cpu_loss - card_loss).abs() <= 1e-4 * cpu_loss.abs(),
            "{cpu_loss} vs {card_loss}"
        );
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for spec in &model.layout.specs {
            for at in spec.offset..spec.offset + spec.len() {
                assert!(
                    (cpu[at] - card[at]).abs() <= 1e-3 * scale,
                    "{}[{}]: processor {}, card {}",
                    spec.name,
                    at - spec.offset,
                    cpu[at],
                    card[at]
                );
            }
        }
    }

    /// Where the delta rule's backward keeps its state changes nothing: a
    /// checkpoint every token, every third and none at all give the same
    /// gradients, bit for bit.
    #[test]
    fn checkpoints_change_nothing() {
        let tokens: Vec<u32> = vec![1, 5, 2, 9, 4, 4, 7, 3, 2, 8];
        let targets: Vec<u32> = vec![5, 2, 9, 4, 4, 7, 3, 2, 8, 1];
        let grads = |chunk: usize| {
            let model = Model::new(Config { chunk, ..tiny() }, 11);
            let mut g = Aligned::zeros(model.layout.total);
            let loss = model
                .forward_backward(&tokens, &targets, Some(&mut g))
                .objective();
            (loss, g.to_vec())
        };
        let reference = grads(100);
        assert_eq!(grads(1), reference);
        assert_eq!(grads(3), reference);
    }

    /// Kept intermediates and recomputed ones give the same gradients, bit
    /// for bit.
    #[test]
    fn kept_intermediates_change_nothing() {
        use std::sync::atomic::Ordering::Relaxed;
        let tokens: Vec<u32> = vec![1, 5, 2, 9, 4, 4, 7, 3, 2, 8];
        let targets: Vec<u32> = vec![5, 2, 9, 4, 4, 7, 3, 2, 8, 1];
        let model = Model::new(tiny(), 11);
        let grads = || {
            let mut g = Aligned::zeros(model.layout.total);
            let loss = model
                .forward_backward(&tokens, &targets, Some(&mut g))
                .objective();
            (loss, g.to_vec())
        };
        let kept = grads();
        // A budget already spent keeps nothing.
        KEPT_BYTES.fetch_add(usize::MAX / 2, Relaxed);
        let recomputed = grads();
        KEPT_BYTES.fetch_sub(usize::MAX / 2, Relaxed);
        assert_eq!(kept, recomputed);
    }

    /// Every element of the delta rule's own parameters against a central
    /// difference, not just each tensor's largest.
    #[test]
    fn every_delta_rule_parameter_has_its_gradient() {
        // Every token uses every expert: a finite difference must not move
        // a token from one expert to another, which is a jump, not a slope.
        let cfg = Config {
            experts_used: 4,
            ..tiny()
        };
        let mut model = Model::new(cfg, 9);
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let targets: Vec<u32> = vec![1, 4, 1, 5, 9, 2, 6, 5];
        let mut grads = Aligned::zeros(model.layout.total);
        model.forward_backward(&tokens, &targets, Some(&mut grads));
        for spec in model.layout.specs.clone() {
            let n = spec.name.as_str();
            if !(n.ends_with("ssm_a_log")
                || n.ends_with("ssm_dt.bias")
                || n.ends_with("ssm_conv1d.weight")
                || n.ends_with("ssm_norm.weight"))
            {
                continue;
            }
            for at in spec.offset..spec.offset + spec.len() {
                let eps = 1e-3f32;
                let original = model.params[at];
                model.params[at] = original + eps;
                let up = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original - eps;
                let down = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original;
                let numeric = (up - down) / (2.0 * eps);
                let noise = f32::EPSILON * up.abs().max(1.0) / eps;
                assert!(
                    (numeric - grads[at]).abs() <= noise + 1e-2 * grads[at].abs(),
                    "{n}[{}]: analytic {}, numeric {numeric}",
                    at - spec.offset,
                    grads[at]
                );
            }
        }
    }

    /// Every element of the router, the shared gate and the experts against
    /// a central difference, the balancing term included. Every token uses
    /// every expert, so a difference cannot move a token between experts.
    #[test]
    fn every_routing_parameter_has_its_gradient() {
        let cfg = Config {
            experts_used: 4,
            aux_coef: 0.5,
            ..tiny()
        };
        let mut model = Model::new(cfg, 13);
        let tokens: Vec<u32> = vec![2, 7, 1, 8, 2, 8, 1, 8];
        let targets: Vec<u32> = vec![7, 1, 8, 2, 8, 1, 8, 2];
        let mut grads = Aligned::zeros(model.layout.total);
        model.forward_backward(&tokens, &targets, Some(&mut grads));
        for spec in model.layout.specs.clone() {
            let n = spec.name.as_str();
            if !(n.contains("ffn_gate_inp") || n.contains("_exps") || n.contains("_shexp")) {
                continue;
            }
            for at in (spec.offset..spec.offset + spec.len()).step_by(7) {
                let eps = 1e-3f32;
                let original = model.params[at];
                model.params[at] = original + eps;
                let up = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original - eps;
                let down = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original;
                let numeric = (up - down) / (2.0 * eps);
                let noise = f32::EPSILON * up.abs().max(1.0) / eps;
                assert!(
                    (numeric - grads[at]).abs() <= noise + 1e-2 * grads[at].abs(),
                    "{n}[{}]: analytic {}, numeric {numeric}",
                    at - spec.offset,
                    grads[at]
                );
            }
        }
    }

    /// The routing rule: the top `experts_used` by probability, ties to the
    /// lower index, weighted by their probabilities over the chosen ones.
    #[test]
    fn a_token_takes_its_most_probable_experts() {
        let model = Model::new(tiny(), 17);
        let (h, ne) = (model.cfg.hidden, model.cfg.experts);
        let x: Vec<f32> = (0..3 * h).map(|i| (i as f32 * 0.31).cos()).collect();
        let m = model.layout.layers[0].moe;
        let (_, c) = model.moe_forward(m, &x, 3);
        for i in 0..3 {
            let p = &c.probs[i * ne..(i + 1) * ne];
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
            let want: Vec<usize> = order[..model.cfg.experts_used].to_vec();
            let got: Vec<usize> = c.chosen[i].iter().map(|&(e, _)| e).collect();
            assert_eq!(got, want);
            let total: f32 = c.chosen[i].iter().map(|&(_, w)| w).sum();
            assert!((total - 1.0).abs() < 1e-5);
        }
    }

    /// The selection rule, on the engine's own cases: a short context sees
    /// everything, the tail is always attended and the budget buys whole
    /// blocks, and equal scores break towards the earlier position.
    #[test]
    fn the_selection_is_the_engines() {
        assert_eq!(select_positions(&[], 5, 4, 7), vec![0, 1, 2, 3, 4]);
        // Ten visible, blocks of 4: positions 8 and 9 are the tail; a
        // budget of 6 takes it and the better block.
        assert_eq!(
            select_positions(&[1.0, 3.0], 10, 4, 6),
            vec![4, 5, 6, 7, 8, 9]
        );
        assert_eq!(
            select_positions(&[2.0, 2.0], 10, 4, 6),
            vec![0, 1, 2, 3, 8, 9]
        );
    }

    /// Attention over a selection of every visible position is the dense
    /// causal attention, forward and back; and its gradient matches a
    /// central difference on a real selection.
    #[test]
    fn attention_over_a_selection() {
        let cfg = tiny();
        let (heads, kvh, hd, t) = (cfg.heads, cfg.kv_heads, cfg.head_dim, 6);
        let wave =
            |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * k).sin()).collect() };
        let (q, k, v) = (
            wave(t * heads * hd, 0.37),
            wave(t * kvh * hd, 0.53),
            wave(t * kvh * hd, 0.29),
        );
        let all: Vec<Vec<usize>> = (0..t).map(|i| (0..=i).collect()).collect();
        let (ctx, probs) = attention_selected(&q, &k, &v, &all, (heads, kvh, hd));
        let dense = attention(&q, &k, &v, &cfg.attention(), t);
        for (a, b) in ctx.iter().zip(dense.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        let d_ctx = wave(t * heads * hd, 0.71);
        let (dq, dk, dv) =
            attention_selected_backward(&q, &k, &v, &all, &probs, &d_ctx, (heads, kvh, hd));
        let (eq, ek, ev) = attention_backward(&q, &k, &v, &d_ctx, &cfg.attention(), t);
        for (a, b) in dq
            .iter()
            .chain(dk.iter())
            .chain(dv.iter())
            .zip(eq.iter().chain(ek.iter()).chain(ev.iter()))
        {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // A real selection: the loss `sum(d_ctx ⊙ ctx)` against a central
        // difference in every input.
        let sel: Vec<Vec<usize>> = (0..t)
            .map(|i| (0..=i).filter(|j| j % 2 == 0 || *j == i).collect())
            .collect();
        let (_, probs) = attention_selected(&q, &k, &v, &sel, (heads, kvh, hd));
        let (dq, dk, dv) =
            attention_selected_backward(&q, &k, &v, &sel, &probs, &d_ctx, (heads, kvh, hd));
        let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            let (c, _) = attention_selected(q, k, v, &sel, (heads, kvh, hd));
            c.iter().zip(&d_ctx).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3f32;
        for (which, grad) in [(0, &dq), (1, &dk), (2, &dv)] {
            for at in 0..grad.len() {
                let mut ins = [q.clone(), k.clone(), v.clone()];
                ins[which][at] += eps;
                let up = loss(&ins[0], &ins[1], &ins[2]);
                ins[which][at] -= 2.0 * eps;
                let down = loss(&ins[0], &ins[1], &ins[2]);
                let numeric = (up - down) / (2.0 * eps);
                assert!(
                    (numeric - grad[at]).abs() < 2e-3 + 1e-2 * grad[at].abs(),
                    "input {which}[{at}]: {} vs {numeric}",
                    grad[at]
                );
            }
        }
    }

    /// The indexer learns where the attention looks: training lowers its
    /// KL divergence from the attention's own mass per block.
    #[test]
    fn the_indexer_learns_where_attention_looks() {
        let mut model = Model::new(
            Config {
                idx_coef: 1.0,
                ..tiny()
            },
            21,
        );
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let targets: Vec<u32> = tokens[1..].iter().copied().chain([1]).collect();
        let first = model.forward_backward(&tokens, &targets, None).indexer;
        let idx: Vec<std::ops::Range<usize>> = model
            .layout
            .specs
            .iter()
            .filter(|s| s.name.contains("indexer."))
            .map(|s| s.offset..s.offset + s.len())
            .collect();
        for _ in 0..300 {
            let mut grads = Aligned::zeros(model.layout.total);
            model.forward_backward(&tokens, &targets, Some(&mut grads));
            for r in &idx {
                for i in r.clone() {
                    model.params[i] -= 2.0 * grads[i];
                }
            }
        }
        let last = model.forward_backward(&tokens, &targets, None).indexer;
        assert!(last < first * 0.7, "indexer loss {first} -> {last}");
    }

    /// With the budget below the sequence, attention reads only what the
    /// indexer selects, and the model still learns.
    #[test]
    fn a_sparse_model_learns_too() {
        let mut model = Model::new(
            Config {
                idx_top_k: 2,
                ..tiny()
            },
            23,
        );
        let tokens: Vec<u32> = (0..16).map(|i| (i % 4) as u32 + 1).collect();
        let targets: Vec<u32> = tokens[1..].iter().copied().chain([1]).collect();
        let first = model.forward_backward(&tokens, &targets, None).ce;
        for _ in 0..200 {
            let mut grads = Aligned::zeros(model.layout.total);
            model.forward_backward(&tokens, &targets, Some(&mut grads));
            for (p, g) in model.params.iter_mut().zip(grads.iter()) {
                *p -= 0.5 * g;
            }
        }
        let last = model.forward_backward(&tokens, &targets, None).ce;
        assert!(last < first * 0.5, "loss {first} -> {last}");
    }

    /// The hash puts every head in its own slice of the table, and an EOS
    /// cuts the context before it but not the token itself.
    #[test]
    fn the_ngram_hash_keeps_heads_apart_and_stops_at_eos() {
        let p = tiny().ple.unwrap();
        let offsets = p.offsets();
        let rows = p.hash(&[1, 2, 3, 10, 4, 5]);
        for token in rows.chunks(p.heads()) {
            for (h, &r) in token.iter().enumerate() {
                assert!(r as u64 >= offsets[h] && (r as u64) < offsets[h] + p.sizes[h]);
            }
        }
        // After the EOS, 4 sees no predecessors: it hashes as the first
        // token of a sequence would.
        let fresh = p.hash(&[4]);
        assert_eq!(&rows[4 * p.heads()..5 * p.heads()], &fresh[..]);
        // The EOS itself still sees the token before it.
        let eos_alone = p.hash(&[10]);
        assert_ne!(&rows[3 * p.heads()..4 * p.heads()], &eos_alone[..]);
    }

    /// Every few elements of the n-gram embedding's tensors, the table
    /// included, against a central difference. Routing held fixed.
    #[test]
    fn every_ple_parameter_has_its_gradient() {
        let mut model = Model::new(
            Config {
                experts_used: 4,
                ..tiny()
            },
            31,
        );
        let tokens: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 10, 4];
        let targets: Vec<u32> = vec![2, 3, 1, 2, 3, 10, 4, 5];
        let mut grads = Aligned::zeros(model.layout.total);
        model.forward_backward(&tokens, &targets, Some(&mut grads));
        let mut checked = 0;
        for spec in model.layout.specs.clone() {
            if !(spec.name.contains("ple_") || spec.name.contains("per_layer_token_embd")) {
                continue;
            }
            for at in (spec.offset..spec.offset + spec.len()).step_by(3) {
                let eps = 3e-4f32;
                let original = model.params[at];
                model.params[at] = original + eps;
                let up = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original - eps;
                let down = along(&spec.name, model.forward_backward(&tokens, &targets, None));
                model.params[at] = original;
                let numeric = (up - down) / (2.0 * eps);
                let noise = f32::EPSILON * up.abs().max(1.0) / eps;
                assert!(
                    (numeric - grads[at]).abs() <= noise + 1e-2 * grads[at].abs(),
                    "{}[{}]: analytic {}, numeric {numeric}",
                    spec.name,
                    at - spec.offset,
                    grads[at]
                );
                checked += 1;
            }
        }
        assert!(checked > 100);
    }

    /// Writes `model` as an F32 `qwen4exp` file the engine can load.
    pub(crate) fn write_f32(model: &Model, path: &std::path::Path) {
        let plans: Vec<crate::write::TensorPlan> = model
            .file_specs()
            .into_iter()
            .map(|(name, dims)| crate::write::TensorPlan {
                name,
                dims,
                ggml_type: orangu::quantize::GGML_TYPE_F32,
            })
            .collect();
        crate::write::write(
            path,
            &model.file_metadata(),
            &plans,
            |p| Ok(model.file_values(&p.name).unwrap()),
            &|_, _, _| {},
        )
        .unwrap();
    }

    /// A trained model, written out and served by the engine, gives the
    /// trainer's own logits at every position: densely, and with the
    /// indexer choosing what attention reads.
    #[test]
    fn the_engine_serves_what_was_trained() {
        for top_k in [8, 2] {
            let mut model = Model::new(
                Config {
                    idx_top_k: top_k,
                    ..tiny()
                },
                41,
            );
            let tokens: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 10, 4, 5, 6];
            let targets: Vec<u32> = tokens[1..].iter().copied().chain([1]).collect();
            for _ in 0..30 {
                let mut grads = Aligned::zeros(model.layout.total);
                model.forward_backward(&tokens, &targets, Some(&mut grads));
                for (p, g) in model.params.iter_mut().zip(grads.iter()) {
                    *p -= 0.3 * g;
                }
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("q4.gguf");
            write_f32(&model, &path);
            let loaded = orangu::engine::loader::LoadedModel::open(&path).unwrap();
            let backend: std::sync::Arc<dyn orangu::engine::backend::Backend> =
                std::sync::Arc::new(orangu::engine::backend::CpuBackend);
            let served = orangu::engine::build::model(&loaded, &backend).unwrap();
            let mut cache = served.new_kv_cache(64);
            let theirs = served
                .forward_all_logits(&mut cache, &tokens, 0, 0)
                .unwrap();
            let ours = model.logits(&tokens);
            let v = model.cfg.vocab;
            assert_eq!(theirs.len(), tokens.len());
            assert!(theirs.iter().all(|r| r.len() == v));
            // Not a model that answers the same whatever it was trained to.
            let fresh = Model::new(
                Config {
                    idx_top_k: top_k,
                    ..tiny()
                },
                41,
            )
            .logits(&tokens);
            assert!(fresh.iter().zip(&ours).any(|(a, b)| (a - b).abs() > 1e-2));
            for (i, row) in theirs.iter().enumerate() {
                for (j, (&a, &b)) in row.iter().zip(&ours[i * v..(i + 1) * v]).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-3 * (1.0 + b.abs()),
                        "top_k {top_k}, position {i}, logit {j}: engine {a}, trainer {b}"
                    );
                }
            }
        }
    }

    /// A checkpoint gives back the model it was written from, the n-gram
    /// hash included, and a dense model's loader refuses it by name.
    #[test]
    fn a_checkpoint_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");
        let cfg = tiny();
        let model = Model::new(cfg.clone(), 9);
        let optimizer = crate::train::Optimizer::new(model.layout.total);
        crate::train::save(&path, &model, &optimizer, 3).unwrap();
        assert_eq!(crate::train::peek(&path).unwrap(), 3);
        let (back, _, step) = crate::train::load::<Model>(&path, &cfg).unwrap();
        assert_eq!(step, 3);
        assert_eq!(back.cfg, cfg);
        assert_eq!(&back.params[..], &model.params[..]);
        let dense = crate::model::Config {
            vocab: 1,
            hidden: 1,
            ffn: 1,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 1,
            context: 1,
            rope_base: 1.0,
            eps: 1.0,
        };
        let err = crate::train::load::<crate::model::Model>(&path, &dense)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("qwen4exp"), "{err}");
    }

    /// Every size builds, and is the size its name says.
    #[test]
    fn the_sizes_are_what_they_say() {
        for size in SIZES {
            let cfg = Config::from_size(size, 32768, 0, 8192);
            let p = cfg.parameters();
            println!("{}: {p}", size.key);
            if size.key == "1b" {
                assert!((0.7e9..1.5e9).contains(&(p as f64)), "1b is {p}");
            }
        }
    }

    /// A zero out-mix is the plain residual add, per stream.
    #[test]
    fn a_zero_injection_adds_the_output_to_every_stream() {
        let model = Model::new(tiny(), 1);
        let (h, hc) = (model.cfg.hidden, model.cfg.hc);
        let mut x = Aligned::zeros(hc * h);
        let sub: Vec<f32> = (0..h).map(|j| j as f32).collect();
        model.combine(&mut x, &sub, &vec![0.0; hc], 1);
        for s in 0..hc {
            assert_eq!(&x[s * h..(s + 1) * h], &sub[..]);
        }
    }

    /// Training lowers the loss on a repeating sequence.
    #[test]
    fn a_tiny_model_learns_a_repeating_sequence() {
        let mut model = Model::new(tiny(), 5);
        let tokens: Vec<u32> = (0..16).map(|i| (i % 4) as u32 + 1).collect();
        let targets: Vec<u32> = tokens[1..].iter().copied().chain([1]).collect();
        let first = model.forward_backward(&tokens, &targets, None).ce;
        for _ in 0..200 {
            let mut grads = Aligned::zeros(model.layout.total);
            model.forward_backward(&tokens, &targets, Some(&mut grads));
            for (p, g) in model.params.iter_mut().zip(grads.iter()) {
                *p -= 0.5 * g;
            }
        }
        let last = model.forward_backward(&tokens, &targets, None).ce;
        assert!(last < first * 0.5, "loss {first} -> {last}");
    }
}

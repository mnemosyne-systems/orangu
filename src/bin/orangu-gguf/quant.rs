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

//! Weights to bytes: the float encodings, the block quantizations, and the
//! per-tensor type rules that turn a file type like `Q4_K_M` into an actual
//! type for each tensor.
//!
//! A `Q4_K_M` file is not a file of `Q4_K` tensors. The M is a *mixture*:
//! the tensors that a quantization error hurts most — the vocabulary
//! projection, the value projections and the down projections in the outer
//! blocks — are carried at higher precision, and the rest at four bits.
//! The rules here are the established ones, spelled out so a file this tool
//! writes is the same mixture a reader expects when it sees that name.
//!
//! Two properties matter more than the compression ratio:
//!
//! - **Norms and any other 1-D tensor stay `f32`.** They are a rounding
//!   error's worth of file size and the thing every activation is divided
//!   by.
//! - **A block quantization needs its row length to divide the block.**
//!   A row that does not divide 256 cannot be a K-quant at all, so it falls
//!   back — to a 32-wide block of about the same precision, and to `f16`
//!   when even that does not fit. Falling back silently would be the wrong
//!   thing to do, so [`Plan`] records every fallback for the caller to
//!   report.

use anyhow::{Result, bail};
use half::f16;

// The file types this tool writes are the K-quants and the 4-bit codebook
// types. The round-number quantizations — `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0` —
// are not offered as file types: a K-quant beats them at every size. They
// are written only as the fallback for a row a K-quant's block does not
// divide, which is where every file of that type carries them.
pub use orangu::quantize::*;

/// A whole-file type: what `--quantization` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ftype {
    F32,
    F16,
    Bf16,
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    IQ4NL,
    IQ4XS,
    Q2KS,
    IQ3M,
    IQ3S,
    IQ3XS,
    IQ3XXS,
    IQ2M,
    IQ2S,
    IQ2XS,
    IQ2XXS,
    IQ1M,
    IQ1S,
}

impl Ftype {
    /// Parses the spelling used on the command line.
    pub fn parse(value: &str) -> Result<Self> {
        let spelling = value.trim().to_ascii_lowercase();
        Ok(match spelling.as_str() {
            "f32" | "fp32" => Ftype::F32,
            "f16" | "fp16" => Ftype::F16,
            "bf16" | "bfloat16" => Ftype::Bf16,
            "q2_k" | "q2k" => Ftype::Q2K,
            "q3_k_s" | "q3ks" => Ftype::Q3KS,
            "q3_k_m" | "q3km" | "q3_k" => Ftype::Q3KM,
            "q3_k_l" | "q3kl" => Ftype::Q3KL,
            "q4_k_s" | "q4ks" => Ftype::Q4KS,
            "q4_k_m" | "q4km" | "q4_k" => Ftype::Q4KM,
            "q5_k_s" | "q5ks" => Ftype::Q5KS,
            "q5_k_m" | "q5km" | "q5_k" => Ftype::Q5KM,
            "q6_k" | "q6k" => Ftype::Q6K,
            "iq4_nl" | "iq4nl" => Ftype::IQ4NL,
            "iq4_xs" | "iq4xs" => Ftype::IQ4XS,
            "q2_k_s" | "q2ks" => Ftype::Q2KS,
            "iq3_m" | "iq3m" => Ftype::IQ3M,
            "iq3_s" | "iq3s" => Ftype::IQ3S,
            "iq3_xs" | "iq3xs" => Ftype::IQ3XS,
            "iq3_xxs" | "iq3xxs" => Ftype::IQ3XXS,
            "iq2_m" | "iq2m" => Ftype::IQ2M,
            "iq2_s" | "iq2s" => Ftype::IQ2S,
            "iq2_xs" | "iq2xs" => Ftype::IQ2XS,
            "iq2_xxs" | "iq2xxs" => Ftype::IQ2XXS,
            "iq1_m" | "iq1m" => Ftype::IQ1M,
            "iq1_s" | "iq1s" => Ftype::IQ1S,
            other
                if other.starts_with("q4_0")
                    || other.starts_with("q4_1")
                    || other.starts_with("q5_0")
                    || other.starts_with("q5_1")
                    || other.starts_with("q8_0")
                    || other.starts_with("q8_1") =>
            {
                let instead = if other.starts_with("q8") {
                    "Q6_K is a quarter smaller and nearly as accurate"
                } else if other.starts_with("q5") {
                    "Q5_K_M is about the same size and more accurate"
                } else {
                    "Q4_K_M is about the same size and more accurate"
                };
                bail!(
                    "{} is a pre-K-quant format and is not written — {instead}",
                    other.to_ascii_uppercase()
                )
            }
            other => bail!(
                "unknown quantization {other:?} — one of: {}",
                Ftype::ALL
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
    }

    /// Largest first, so a list of them reads as a size ladder.
    pub const ALL: [Ftype; 25] = [
        Ftype::Bf16,
        Ftype::F16,
        Ftype::F32,
        Ftype::Q6K,
        Ftype::Q5KM,
        Ftype::Q5KS,
        Ftype::Q4KM,
        Ftype::Q4KS,
        Ftype::IQ4NL,
        Ftype::IQ4XS,
        Ftype::Q3KL,
        Ftype::Q3KM,
        Ftype::Q3KS,
        Ftype::Q2K,
        Ftype::IQ3M,
        Ftype::IQ3S,
        Ftype::IQ3XS,
        Ftype::IQ3XXS,
        Ftype::Q2KS,
        Ftype::IQ2M,
        Ftype::IQ2S,
        Ftype::IQ2XS,
        Ftype::IQ2XXS,
        Ftype::IQ1M,
        Ftype::IQ1S,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Ftype::F32 => "F32",
            Ftype::F16 => "F16",
            Ftype::Bf16 => "BF16",
            Ftype::Q2K => "Q2_K",
            Ftype::Q3KS => "Q3_K_S",
            Ftype::Q3KM => "Q3_K_M",
            Ftype::Q3KL => "Q3_K_L",
            Ftype::Q4KS => "Q4_K_S",
            Ftype::Q4KM => "Q4_K_M",
            Ftype::Q5KS => "Q5_K_S",
            Ftype::Q5KM => "Q5_K_M",
            Ftype::Q6K => "Q6_K",
            Ftype::IQ4NL => "IQ4_NL",
            Ftype::IQ4XS => "IQ4_XS",
            Ftype::Q2KS => "Q2_K_S",
            Ftype::IQ3M => "IQ3_M",
            Ftype::IQ3S => "IQ3_S",
            Ftype::IQ3XS => "IQ3_XS",
            Ftype::IQ3XXS => "IQ3_XXS",
            Ftype::IQ2M => "IQ2_M",
            Ftype::IQ2S => "IQ2_S",
            Ftype::IQ2XS => "IQ2_XS",
            Ftype::IQ2XXS => "IQ2_XXS",
            Ftype::IQ1M => "IQ1_M",
            Ftype::IQ1S => "IQ1_S",
        }
    }

    /// One line on what this file type is for.
    pub fn description(&self) -> &'static str {
        match self {
            Ftype::F32 => {
                "Every weight in full precision. Twice the size of BF16, no more accurate in practice."
            }
            Ftype::F16 => {
                "Half precision. Same size as BF16, less range — BF16 is the better default for trained weights."
            }
            Ftype::Bf16 => {
                "Half precision with the exponent range of F32. The reference file: quantize from this, not to it."
            }
            Ftype::Q2K => {
                "Two bits. The smallest file worth writing, and the first one where the model is visibly worse."
            }
            Ftype::Q3KS => "Three bits throughout. Smaller than Q3_K_M and measurably weaker.",
            Ftype::Q3KM => {
                "Three bits, with four and five where they pay: the value and down projections."
            }
            Ftype::Q3KL => "Three bits, with five on the tensors Q3_K_M gives four.",
            Ftype::Q4KS => "Four bits, with five on the outermost down projections only.",
            Ftype::Q4KM => {
                "Four bits, with six on the outer value and down projections. The size/quality sweet spot."
            }
            Ftype::Q5KS => "Five bits throughout. Close to Q6_K for noticeably less size.",
            Ftype::Q5KM => "Five bits, with six on the outer value and down projections.",
            Ftype::Q6K => {
                "Six bits. Indistinguishable from BF16 on every benchmark that matters, at 40% of the size."
            }
            Ftype::IQ4NL => {
                "Four non-linear bits in 32-wide blocks. Fits rows a K-quant cannot, at Q4_K_S's size."
            }
            Ftype::IQ4XS => {
                "Four non-linear bits in 256-wide blocks. The smallest four-bit file, just under Q4_K_S."
            }
            Ftype::Q2KS => {
                "Two bits with fewer reinforced tensors than Q2_K. Needs an importance matrix."
            }
            Ftype::IQ3M => {
                "Three-bit lattice, with four bits on the value, output and first down projections."
            }
            Ftype::IQ3S => {
                "Three-bit lattice throughout, value projections at four bits on grouped-query models."
            }
            Ftype::IQ3XS => {
                "Three-bit lattice, the middle gate and up projections at the smaller three-bit lattice."
            }
            Ftype::IQ3XXS => "The smaller three-bit lattice. Near Q3_K_S quality in less space.",
            Ftype::IQ2M => {
                "Two-and-a-half-bit lattice. The smallest file most models still answer well from."
            }
            Ftype::IQ2S => {
                "Two-bit lattice with eight sign bits a group. Needs an importance matrix."
            }
            Ftype::IQ2XS => "Two-bit lattice, 512 points. Needs an importance matrix.",
            Ftype::IQ2XXS => "Two-bit lattice, 256 points. Needs an importance matrix.",
            Ftype::IQ1M => "One-and-three-quarter bits. For models too large to fit any other way.",
            Ftype::IQ1S => "One-and-a-half bits. The smallest file; expect a much weaker model.",
        }
    }

    /// The file type a `general.file_type` names, if this tool writes it.
    #[cfg(test)]
    pub fn from_file_type(value: u32) -> Option<Self> {
        Ftype::ALL.into_iter().find(|f| f.file_type() == value)
    }

    /// The `general.file_type` this writes.
    pub fn file_type(&self) -> u32 {
        match self {
            Ftype::F32 => 0,
            Ftype::F16 => 1,
            Ftype::Q2K => 10,
            Ftype::Q3KS => 11,
            Ftype::Q3KM => 12,
            Ftype::Q3KL => 13,
            Ftype::Q4KS => 14,
            Ftype::Q4KM => 15,
            Ftype::Q5KS => 16,
            Ftype::Q5KM => 17,
            Ftype::Q6K => 18,
            Ftype::IQ4NL => 25,
            Ftype::IQ4XS => 30,
            Ftype::Q2KS => 21,
            Ftype::IQ3M => 27,
            Ftype::IQ3S => 26,
            Ftype::IQ3XS => 22,
            Ftype::IQ3XXS => 23,
            Ftype::IQ2M => 29,
            Ftype::IQ2S => 28,
            Ftype::IQ2XS => 20,
            Ftype::IQ2XXS => 19,
            Ftype::IQ1M => 31,
            Ftype::IQ1S => 24,
            Ftype::Bf16 => 32,
        }
    }

    /// The type most of the file is written as, before the mixture rules
    /// move individual tensors.
    /// Whether a file of this type is refused without an importance matrix:
    /// its blocks' weights are written in a format that needs one.
    pub fn needs_importance(&self) -> bool {
        *self == Ftype::Q2KS
            || matches!(
                self.base_type(),
                GGML_TYPE_IQ3_XXS
                    | GGML_TYPE_IQ2_XXS
                    | GGML_TYPE_IQ2_XS
                    | GGML_TYPE_IQ2_S
                    | GGML_TYPE_IQ1_M
                    | GGML_TYPE_IQ1_S
            )
    }

    fn base_type(&self) -> u32 {
        match self {
            Ftype::F32 => GGML_TYPE_F32,
            Ftype::F16 => GGML_TYPE_F16,
            Ftype::Bf16 => GGML_TYPE_BF16,
            Ftype::Q2K => GGML_TYPE_Q2_K,
            Ftype::Q3KS | Ftype::Q3KM | Ftype::Q3KL => GGML_TYPE_Q3_K,
            Ftype::Q4KS | Ftype::Q4KM => GGML_TYPE_Q4_K,
            Ftype::Q5KS | Ftype::Q5KM => GGML_TYPE_Q5_K,
            Ftype::Q6K => GGML_TYPE_Q6_K,
            Ftype::IQ4NL => GGML_TYPE_IQ4_NL,
            Ftype::IQ4XS => GGML_TYPE_IQ4_XS,
            Ftype::Q2KS => GGML_TYPE_Q2_K,
            Ftype::IQ3M | Ftype::IQ3S | Ftype::IQ3XS => GGML_TYPE_IQ3_S,
            Ftype::IQ3XXS => GGML_TYPE_IQ3_XXS,
            Ftype::IQ2M => GGML_TYPE_IQ2_S,
            Ftype::IQ2S | Ftype::IQ2XS => GGML_TYPE_IQ2_XS,
            Ftype::IQ2XXS => GGML_TYPE_IQ2_XXS,
            Ftype::IQ1M => GGML_TYPE_IQ1_M,
            Ftype::IQ1S => GGML_TYPE_IQ1_S,
        }
    }

    /// Whether the file is quantized at all, which is what decides if the
    /// mixture rules run.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Ftype::F32 | Ftype::F16 | Ftype::Bf16)
    }
}

/// What a tensor is to the mixture rules, read from its name so it works
/// for a file this tool writes and for one it only reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    TokenEmbedding,
    Output,
    AttentionQkv,
    AttentionKvB,
    AttentionValue,
    AttentionKey,
    AttentionQuery,
    AttentionOutput,
    AttentionGate,
    FeedForwardUp,
    FeedForwardGate,
    FeedForwardLatent,
    FeedForwardDown,
    Other,
}

impl Category {
    fn of(name: &str) -> Self {
        let has = |s: &str| name.contains(s);
        match name {
            "output.weight" => Category::Output,
            "token_embd.weight" | "per_layer_token_embd.weight" => Category::TokenEmbedding,
            _ if has("attn_qkv.weight") || has("attn_qkv_gate.weight") => Category::AttentionQkv,
            _ if has("attn_kv_b.weight") => Category::AttentionKvB,
            _ if has("attn_v.weight") => Category::AttentionValue,
            _ if has("attn_k.weight") => Category::AttentionKey,
            _ if has("attn_q.weight") => Category::AttentionQuery,
            _ if has("attn_output.weight") => Category::AttentionOutput,
            _ if has("ffn_up") => Category::FeedForwardUp,
            _ if has("ffn_gate") => Category::FeedForwardGate,
            _ if has("ffn_latent") => Category::FeedForwardLatent,
            _ if has("attn_gate.weight") => Category::AttentionGate,
            _ if has("ffn_down") => Category::FeedForwardDown,
            _ => Category::Other,
        }
    }

    /// The value-like projections: the tensors an error costs most in.
    fn is_value(self) -> bool {
        matches!(
            self,
            Category::AttentionValue | Category::AttentionQkv | Category::AttentionKvB
        )
    }
}

/// The block index in a `blk.N.` tensor name.
pub fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Puts tensors in the order a quantized file lists them: the tensors
/// outside any block first, then block by block, by name within each. The
/// mixture's counters run in this order, so a checkpoint whose names sort
/// `blk.10` before `blk.2` still gets its extra bits in the right blocks.
pub fn sort_for_quantizing<T>(items: &mut [T], name: impl Fn(&T) -> &str) {
    items.sort_by(|a, b| {
        let (a, b) = (name(a), name(b));
        let block = |n: &str| block_index(n).map_or(-1, |i| i as i64);
        block(a).cmp(&block(b)).then_with(|| a.cmp(b))
    });
}

/// The rule that spends the extra bits on the outer blocks and on every
/// third block in between, rather than spreading them evenly.
fn use_more_bits(layer: usize, layers: usize) -> bool {
    let (layer, layers) = (layer as i64, layers as i64);
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}

/// Whether a tensor is kept at `f32` in every file type: anything that is
/// not a matrix, every norm, and the small tensors that steer the rest — a
/// mixture's router, position and token-type tables, short convolutions.
pub fn requires_f32(name: &str, dims: &[u64]) -> bool {
    const KEPT: [&str; 17] = [
        "ffn_gate_inp",
        "position_embd",
        "token_types",
        "ssm_conv1d",
        "shortconv.conv",
        "time_mix_first",
        "time_mix_w1",
        "time_mix_w2",
        "time_mix_decay_w1",
        "time_mix_decay_w2",
        "time_mix_lerp_fused",
        "posnet",
        "v.position_embd",
        "a.position_embd",
        "altup_correct_coef",
        "altup_predict_coef",
        "indexer.proj",
    ];
    dims.len() < 2
        || name.ends_with("_norm.weight")
        || !(name.ends_with(".weight") || name.ends_with(".lora_a") || name.ends_with(".lora_b"))
        || KEPT.iter().any(|k| name.contains(k))
}

/// Whether a quantized file quantizes this tensor at all. One it does not
/// keeps the type it had in the source.
fn allows_quantization(name: &str, dims: &[u64], architecture: &str) -> bool {
    let has = |s: &str| name.contains(s);
    if dims.len() < 2 || !name.ends_with("weight") || has("_norm.weight") {
        return false;
    }
    const NEVER: [&str; 36] = [
        "ffn_gate_inp.weight",
        "ffn_gate_tid2eid.weight",
        "engram_gate.weight",
        "altup",
        "laurel",
        "per_layer_model_proj",
        "ssm_conv1d",
        "shortconv.conv.weight",
        "conv2d_spatial.weight",
        "conv_dw",
        "indexer.k_proj.weight",
        "indexer.q_proj.weight",
        "time_mix_first.weight",
        "time_mix_w0.weight",
        "time_mix_w1.weight",
        "time_mix_w2.weight",
        "time_mix_v0.weight",
        "time_mix_v1.weight",
        "time_mix_v2.weight",
        "time_mix_a0.weight",
        "time_mix_a1.weight",
        "time_mix_a2.weight",
        "time_mix_g1.weight",
        "time_mix_g2.weight",
        "time_mix_decay_w1.weight",
        "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight",
        "attn_rel_b.weight",
        ".position_embd",
        "sam.pos_embd",
        "sam.neck.",
        "sam.net_",
        ".rel_pos",
        ".patch_embd",
        ".patch_merger",
        "a.rvq.codebook",
    ];
    if NEVER.iter().any(|s| has(s))
        || has("mm.a.code_embd")
        || name.starts_with("a.conv1d.")
        || name.starts_with("a.gen.wav.quant.")
        || name == "position_embd.weight"
        || name == "token_types.weight"
    {
        return false;
    }
    let kept: &[&str] = match architecture {
        "limite" => &["attn_gate.weight", "attn_ve_gate.weight", "mudd_"],
        "inkling2" => &[
            "_conv.weight",
            "_clip.weight",
            "_res_query.weight",
            "attn_rel_bias.weight",
        ],
        "zaya1" => &["ffn_router_", "cca_conv_grp"],
        "metis16" => &["stream_embd", "attn_pass_lora", "ffn_exp_embd", "hc_"],
        "kimi-k31" => &["indexer.attn_q_b", "indexer.attn_k", "indexer.proj"],
        "glm5-next" => &[
            "hc_",
            "indexer.attn_q_b",
            "indexer.attn_k",
            "indexer.proj",
            "indexer_compressor_gate",
            "indexer_compressor_ape",
            "ssm_f_a.weight",
            "ssm_f_b.weight",
            "ssm_g_a.weight",
            "ssm_g_b.weight",
            "ssm_beta.weight",
            "attn_kv_a_mqa.weight",
            "attn_k_b.weight",
            "attn_v_b.weight",
        ],
        "xing4_0" => &["hc_"],
        _ => &[],
    };
    if kept.iter().any(|s| has(s)) {
        return false;
    }
    if architecture == "metis16" && (name.starts_with("depth_mem_") || name.starts_with("cont_")) {
        return false;
    }
    !(architecture == "vllm-decision2" && name.starts_with("decision."))
}

/// What the mixture rules need to know about the model beyond one tensor's
/// own name and shape.
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// `general.architecture`.
    pub architecture: String,
    pub layers: usize,
    /// Query heads and key/value heads, of the first layer.
    pub heads: usize,
    pub kv_heads: usize,
    /// Routed experts; 0 for a dense model.
    pub experts: usize,
    /// Whether the weights are fitted to an importance matrix, which
    /// changes a few choices.
    pub imatrix: bool,
    /// One type for every tensor that may be quantized: the format's base
    /// type, none of its mixture.
    pub pure: bool,
}

impl Model {
    /// Query heads per key/value head. Grouped-query attention makes the
    /// value projection small enough that carrying it at a higher precision
    /// costs almost nothing.
    fn gqa(&self) -> usize {
        self.heads.checked_div(self.kv_heads).unwrap_or(0)
    }

    /// The 70-billion-parameter shape the value projections are carried
    /// higher in: eight query heads to each key/value head.
    fn is_70b(&self) -> bool {
        match self.architecture.as_str() {
            "llama" => self.layers == 80 && self.heads != self.kv_heads,
            "qwen2" | "deci" | "olmo" => self.layers == 80,
            "jais2" => self.layers == 68,
            _ => false,
        }
    }

    /// Reads the fields from a file's metadata.
    pub fn from_metadata(metadata: &[(String, orangu::gguf::GgufValue)]) -> Self {
        use orangu::gguf::GgufValue;
        let get = |key: &str| metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let architecture = match get("general.architecture") {
            Some(GgufValue::String(s)) => s.clone(),
            _ => String::new(),
        };
        // A per-layer array reads as its first layer.
        let number = |key: &str| {
            let value = get(&format!("{architecture}.{key}"));
            match value {
                Some(GgufValue::Array(items)) => items.first().and_then(GgufValue::as_u64),
                Some(v) => v.as_u64(),
                None => None,
            }
            .unwrap_or(0) as usize
        };
        let heads = number("attention.head_count");
        let kv_heads = match get(&format!("{architecture}.attention.head_count_kv")) {
            Some(_) => number("attention.head_count_kv"),
            None => heads,
        };
        Model {
            layers: number("block_count"),
            heads,
            kv_heads,
            experts: number("expert_count"),
            imatrix: false,
            pure: false,
            architecture,
        }
    }
}

/// One tensor as the planner sees it.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    pub name: &'a str,
    /// GGUF order: `dims[0]` is the row length.
    pub dims: &'a [u64],
    /// The type it has in the source, which a tensor the file does not
    /// quantize keeps.
    pub source: u32,
}

/// What one tensor is written as, and why it might not be what was asked
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub ggml_type: u32,
    /// The type the mixture rules chose, when a shape forced something
    /// else. `None` when nothing was overridden.
    pub fallback_from: Option<u32>,
}

/// Chooses every tensor's type, in file order.
///
/// The rules are the standard ones behind the `_S`/`_M`/`_L` names, so a
/// file of a given type here carries the same tensors at the same types as
/// one of that type made anywhere else. Some depend on the whole file: the
/// value projections are counted so the outer ones can be told apart, and
/// a model with no separate output projection reads its token embedding as
/// the output.
pub fn plan_tensors(ftype: Ftype, tensors: &[Tensor<'_>], model: &Model) -> Vec<Plan> {
    let mut planner = Planner {
        ftype,
        model,
        values: tensors
            .iter()
            .filter(|t| Category::of(t.name).is_value())
            .count(),
        tied: !tensors.iter().any(|t| t.name == "output.weight"),
        value_at: 0,
        down_at: 0,
        gate_at: 0,
        up_at: 0,
    };
    tensors.iter().map(|t| planner.plan(t)).collect()
}

/// The planner's running state over one file.
struct Planner<'a> {
    ftype: Ftype,
    model: &'a Model,
    /// How many value projections the file has.
    values: usize,
    /// Whether the token embedding doubles as the output projection.
    tied: bool,
    /// Value projections and down projections planned so far.
    value_at: usize,
    down_at: usize,
    /// Gate and up projections planned so far.
    gate_at: usize,
    up_at: usize,
}

impl Planner<'_> {
    fn plan(&mut self, tensor: &Tensor<'_>) -> Plan {
        let keep = |ggml_type| Plan {
            ggml_type,
            fallback_from: None,
        };
        if !self.ftype.is_quantized() {
            return keep(if requires_f32(tensor.name, tensor.dims) {
                GGML_TYPE_F32
            } else {
                self.ftype.base_type()
            });
        }
        if !allows_quantization(tensor.name, tensor.dims, &self.model.architecture) {
            return keep(if requires_f32(tensor.name, tensor.dims) {
                GGML_TYPE_F32
            } else {
                tensor.source
            });
        }
        let wanted = if self.model.pure {
            self.ftype.base_type()
        } else {
            self.mixture(tensor)
        };
        let ggml_type = fallback(wanted, tensor.dims[0] as usize);
        Plan {
            ggml_type,
            fallback_from: (ggml_type != wanted).then_some(wanted),
        }
    }

    /// The per-tensor rules behind the `_S`/`_M`/`_L` in a file type's
    /// name: which tensors are carried above the base type, and by how
    /// much. The vocabulary projection, because it is the last thing
    /// before the softmax; the value and down projections, because an
    /// error there lands on every token that attends through them.
    fn mixture(&mut self, tensor: &Tensor<'_>) -> u32 {
        use Ftype::*;
        let (ftype, model) = (self.ftype, self.model);
        let name = tensor.name;
        let category = Category::of(name);
        let arch = model.architecture.as_str();
        let falcon = arch == "falcon";
        let mut t = ftype.base_type();
        let low_lattice = matches!(ftype, IQ2XXS | IQ2XS | IQ1S | IQ2S | IQ2M | IQ1M);
        let three_bit_lattice = matches!(ftype, IQ2S | IQ2M);

        if arch == "glm5-next"
            && ["attn_q_a", "attn_q_b", "nextn.eh_proj"]
                .iter()
                .any(|s| name.contains(s))
        {
            return GGML_TYPE_Q8_0;
        }
        if arch == "muse-spark"
            && matches!(
                category,
                Category::FeedForwardLatent | Category::AttentionGate
            )
        {
            return GGML_TYPE_Q8_0;
        }

        if category == Category::Output || (self.tied && category == Category::TokenEmbedding) {
            if falcon || !(tensor.dims[0] as usize).is_multiple_of(block_size(t)) {
                t = GGML_TYPE_Q8_0;
            } else if low_lattice || ftype == IQ3XXS {
                t = GGML_TYPE_Q5_K;
            } else if t != GGML_TYPE_Q8_0 {
                t = GGML_TYPE_Q6_K;
            }
            return t;
        }
        let gqa = model.gqa();
        // A mixture's experts are not stored in layer order, so its layer
        // comes from the name; a dense model's from the running count.
        let layer_of = |count: usize| {
            if model.experts > 1 {
                block_index(name).unwrap_or(0)
            } else {
                count
            }
        };
        let layers = model.layers;
        if category == Category::TokenEmbedding {
            match ftype {
                IQ2XXS | IQ2XS | IQ1S | IQ1M => t = GGML_TYPE_Q2_K,
                IQ2S | IQ2M | IQ3XXS => t = GGML_TYPE_IQ3_S,
                _ => {}
            }
            return t;
        }
        if low_lattice {
            // Below two and a half bits every category has its own rule,
            // and nothing else applies.
            let small = if three_bit_lattice {
                GGML_TYPE_IQ3_S
            } else {
                GGML_TYPE_Q2_K
            };
            if category.is_value() {
                t = if gqa >= 4 || model.experts >= 4 {
                    GGML_TYPE_Q4_K
                } else {
                    small
                };
                self.value_at += 1;
            } else if model.experts == 8 && category == Category::AttentionKey {
                t = GGML_TYPE_Q4_K;
            } else if category == Category::FeedForwardDown {
                if self.down_at < layers / 8 {
                    t = small;
                }
                self.down_at += 1;
            } else if category == Category::AttentionOutput {
                if model.experts == 8 {
                    t = GGML_TYPE_Q5_K;
                } else if matches!(ftype, IQ1S | IQ1M) {
                    t = GGML_TYPE_IQ2_XXS;
                } else if three_bit_lattice {
                    t = GGML_TYPE_IQ3_S;
                }
            }
            return t;
        }
        match category {
            c if c.is_value() => {
                let (at, of) = (self.value_at, self.values);
                match ftype {
                    Q2K => {
                        t = if gqa >= 4 {
                            GGML_TYPE_Q4_K
                        } else {
                            GGML_TYPE_Q3_K
                        }
                    }
                    Q2KS if gqa >= 4 => t = GGML_TYPE_Q4_K,
                    IQ3XXS => {
                        t = if gqa >= 4 {
                            GGML_TYPE_Q4_K
                        } else if !model.imatrix {
                            GGML_TYPE_IQ3_S
                        } else {
                            GGML_TYPE_IQ3_XXS
                        }
                    }
                    IQ3XS | IQ3S if gqa >= 4 => t = GGML_TYPE_Q4_K,
                    IQ3M => t = GGML_TYPE_Q4_K,
                    Q3KM => {
                        t = if at < 2 {
                            GGML_TYPE_Q5_K
                        } else {
                            GGML_TYPE_Q4_K
                        }
                    }
                    Q3KL => t = GGML_TYPE_Q5_K,
                    IQ4NL | IQ4XS if gqa >= 4 => t = GGML_TYPE_Q5_K,
                    Q4KM | Q5KM if use_more_bits(at, of) => t = GGML_TYPE_Q6_K,
                    Q4KS if at < 4 => t = GGML_TYPE_Q5_K,
                    _ => {}
                }
                if model.is_70b() && (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K) {
                    t = GGML_TYPE_Q5_K;
                }
                if model.experts == 8 {
                    t = GGML_TYPE_Q8_0;
                }
                self.value_at += 1;
            }
            Category::AttentionKey => {
                if model.experts == 8 {
                    t = GGML_TYPE_Q8_0;
                } else if ftype == IQ3XS {
                    t = GGML_TYPE_IQ3_XXS;
                } else if ftype == IQ3XXS {
                    t = GGML_TYPE_IQ2_S;
                }
            }
            Category::AttentionQuery => {
                if ftype == IQ3XS {
                    t = GGML_TYPE_IQ3_XXS;
                } else if ftype == IQ3XXS {
                    t = GGML_TYPE_IQ2_S;
                }
            }
            Category::FeedForwardDown => {
                let layer = layer_of(self.down_at);
                let more = use_more_bits(layer, layers);
                match ftype {
                    Q2K => t = GGML_TYPE_Q3_K,
                    Q2KS => {
                        if layer < layers / 8 {
                            t = GGML_TYPE_Q4_K;
                        }
                    }
                    IQ3XXS if !model.imatrix => {
                        t = if layer < layers / 8 {
                            GGML_TYPE_Q4_K
                        } else {
                            GGML_TYPE_Q3_K
                        }
                    }
                    Q3KM => {
                        t = if layer < layers / 16 {
                            GGML_TYPE_Q5_K
                        } else if !falcon || more {
                            GGML_TYPE_Q4_K
                        } else {
                            GGML_TYPE_Q3_K
                        }
                    }
                    IQ3M if layer < layers / 8 || (model.experts == 8 && more) => {
                        t = GGML_TYPE_Q4_K
                    }
                    Q3KL => {
                        t = if falcon {
                            GGML_TYPE_Q4_K
                        } else {
                            GGML_TYPE_Q5_K
                        }
                    }
                    Q4KM if falcon => {
                        t = if layer < layers / 16 {
                            GGML_TYPE_Q6_K
                        } else if more {
                            GGML_TYPE_Q5_K
                        } else {
                            GGML_TYPE_Q4_K
                        }
                    }
                    Q4KM if more => t = GGML_TYPE_Q6_K,
                    IQ4NL | IQ4XS if layer < layers / 8 && !model.imatrix => t = GGML_TYPE_Q5_K,
                    Q5KM if more => t = GGML_TYPE_Q6_K,
                    Q4KS if !falcon && layer < layers / 8 => t = GGML_TYPE_Q5_K,
                    _ => {}
                }
                self.down_at += 1;
            }
            Category::AttentionOutput => {
                if falcon {
                    if ftype == Q3KL {
                        t = GGML_TYPE_Q4_K;
                    }
                } else if model.experts == 8 {
                    if matches!(
                        ftype,
                        Q2K | IQ3XS
                            | IQ3XXS
                            | Q3KS
                            | Q3KM
                            | IQ4NL
                            | Q4KS
                            | Q4KM
                            | IQ3S
                            | IQ3M
                            | IQ4XS
                    ) {
                        t = GGML_TYPE_Q5_K;
                    }
                } else {
                    match ftype {
                        Q2K => t = GGML_TYPE_Q3_K,
                        IQ3XXS => t = GGML_TYPE_IQ3_S,
                        Q3KM | IQ3M => t = GGML_TYPE_Q4_K,
                        Q3KL => t = GGML_TYPE_Q5_K,
                        _ => {}
                    }
                }
            }
            Category::AttentionQkv => match ftype {
                Q3KM | Q3KL | IQ3M => t = GGML_TYPE_Q4_K,
                Q4KM => t = GGML_TYPE_Q5_K,
                Q5KM => t = GGML_TYPE_Q6_K,
                _ => {}
            },
            Category::FeedForwardGate | Category::FeedForwardUp => {
                let count = if category == Category::FeedForwardGate {
                    &mut self.gate_at
                } else {
                    &mut self.up_at
                };
                let layer = if model.experts > 1 {
                    block_index(name).unwrap_or(0)
                } else {
                    *count
                };
                if ftype == IQ3XS && layer >= layers / 8 && layer < 7 * layers / 8 {
                    t = GGML_TYPE_IQ3_XXS;
                }
                *count += 1;
            }
            _ => {}
        }
        t
    }
}

/// The type a tensor falls back to when the chosen type's block does not
/// divide its rows: the 32-wide type of about the same precision, and
/// `f16` when not even that divides.
fn fallback(wanted: u32, ncols: usize) -> u32 {
    if ncols.is_multiple_of(block_size(wanted)) {
        return wanted;
    }
    let narrower = match wanted {
        GGML_TYPE_IQ1_S | GGML_TYPE_IQ1_M | GGML_TYPE_IQ2_XXS | GGML_TYPE_IQ2_XS
        | GGML_TYPE_IQ2_S | GGML_TYPE_IQ3_XXS | GGML_TYPE_IQ3_S | GGML_TYPE_IQ4_XS => {
            GGML_TYPE_IQ4_NL
        }
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K => GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K => GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_K => GGML_TYPE_Q5_1,
        GGML_TYPE_Q6_K => GGML_TYPE_Q8_0,
        other => other,
    };
    if ncols.is_multiple_of(block_size(narrower)) {
        narrower
    } else {
        GGML_TYPE_F16
    }
}

/// Reads a tensor's stored bytes back as `f32`.
///
/// Only the float encodings are handled, and that is deliberate: this tool
/// quantizes *from* a full-precision file. Quantizing a file that is
/// already quantized would compound two roundings and quietly produce
/// something worse than quantizing the original once, so it is refused with
/// a message that says so.
pub fn decode(ggml_type: u32, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(elements)
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        GGML_TYPE_F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(elements)
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect()),
        GGML_TYPE_BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(elements)
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect()),
        other => bail!(
            "this file's tensors are already {} — quantize from the F32, F16 or BF16 file instead",
            type_name(other)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoding is the reader's half of the contract, written out here so
    /// the round-trip tests measure the real error rather than this file's
    /// own idea of it.
    fn dequantize(ggml_type: u32, bytes: &[u8], elements: usize) -> Vec<f32> {
        match ggml_type {
            GGML_TYPE_Q2_K => bytes
                .as_chunks::<{ QK_K / 16 + QK_K / 4 + 4 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let scales = &block[..QK_K / 16];
                    let quants = &block[QK_K / 16..QK_K / 16 + QK_K / 4];
                    let at = QK_K / 16 + QK_K / 4;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[at + 2], block[at + 3]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let q = (quants[chunk * 32 + within % 32] >> (2 * (within / 32))) & 3;
                        let scale = scales[i / 16];
                        d * (scale & 0xF) as f32 * q as f32 - dmin * (scale >> 4) as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q3_K => bytes
                .as_chunks::<{ QK_K / 8 + QK_K / 4 + 12 + 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let mask = &block[..QK_K / 8];
                    let quants = &block[QK_K / 8..QK_K / 8 + QK_K / 4];
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12]);
                    let at = QK_K / 8 + QK_K / 4 + 12;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let low = (quants[chunk * 32 + within % 32] >> (2 * (within / 32))) & 3;
                        let high = (mask[i % (QK_K / 8)] >> (i / (QK_K / 8))) & 1;
                        let q = low as i32 - if high == 0 { 4 } else { 0 };
                        d * unpack_signed_scale(i / 16, &packed) as f32 * q as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q5_K => bytes
                .as_chunks::<{ 2 + 2 + 12 + QK_K / 8 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[4..16]);
                    let high = &block[16..16 + QK_K / 8];
                    let low = &block[16 + QK_K / 8..];
                    (0..QK_K).map(move |i| {
                        let j = i / 32;
                        let (sc, m) = unpack_scale_min(j, &packed);
                        let chunk = i / 64;
                        let within = i % 64;
                        let byte = low[chunk * 32 + within % 32];
                        let nibble = if within < 32 { byte & 0xF } else { byte >> 4 };
                        let bit = (high[i % 32] >> (2 * chunk + within / 32)) & 1;
                        let q = nibble | (bit << 4);
                        d * sc as f32 * q as f32 - dmin * m as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_IQ4_NL => bytes
                .as_chunks::<{ 2 + QK / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let quants = &block[2..];
                    (0..QK).map(move |i| {
                        let byte = quants[i % (QK / 2)];
                        let level = if i < QK / 2 { byte & 0xF } else { byte >> 4 };
                        d * IQ4_VALUES[level as usize] as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_IQ4_XS => bytes
                .as_chunks::<{ 2 + 2 + QK_K / 64 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let scales_h = u16::from_le_bytes([block[2], block[3]]);
                    let scales_l = &block[4..4 + QK_K / 64];
                    let quants = &block[4 + QK_K / 64..];
                    (0..QK_K).map(move |i| {
                        let ib = i / 32;
                        let low = if ib % 2 == 0 {
                            scales_l[ib / 2] & 0xF
                        } else {
                            scales_l[ib / 2] >> 4
                        };
                        let high = ((scales_h >> (2 * ib)) & 3) as u8;
                        let scale = (low | (high << 4)) as i32 - 32;
                        let within = i % 32;
                        let byte = quants[16 * ib + within % 16];
                        let level = if within < 16 { byte & 0xF } else { byte >> 4 };
                        d * scale as f32 * IQ4_VALUES[level as usize] as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q4_K => bytes
                .as_chunks::<{ 2 + 2 + 12 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[4..16]);
                    let quants = &block[16..];
                    (0..QK_K).map(move |i| {
                        let j = i / 32;
                        let (sc, m) = unpack_scale_min(j, &packed);
                        let chunk = i / 64;
                        let within = i % 64;
                        let byte = quants[chunk * 32 + within % 32];
                        let q = if within < 32 { byte & 0xF } else { byte >> 4 };
                        d * sc as f32 * q as f32 - dmin * m as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q6_K => bytes
                .as_chunks::<{ QK_K / 2 + QK_K / 4 + QK_K / 16 + 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let low = &block[..QK_K / 2];
                    let high = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
                    let scales = &block[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
                    let at = QK_K / 2 + QK_K / 4 + QK_K / 16;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let quarter = within / 32;
                        let lane = within % 32;
                        let low_byte =
                            low[chunk * 64 + lane + if quarter % 2 == 1 { 32 } else { 0 }];
                        let nibble = if quarter < 2 {
                            low_byte & 0xF
                        } else {
                            low_byte >> 4
                        };
                        let bits = (high[chunk * 32 + lane] >> (2 * quarter)) & 3;
                        let q = (nibble | (bits << 4)) as i32 - 32;
                        let scale = scales[i / 16] as i8;
                        d * scale as f32 * q as f32
                    })
                })
                .take(elements)
                .collect(),
            other => orangu::dequantize::dequantize(other, bytes, elements).unwrap(),
        }
    }

    /// Weight-shaped values: a deterministic normal sample with the
    /// occasional large outlier, which is what a trained matrix actually
    /// looks like and what the block quantizations are tuned for. A uniform
    /// sample would flatter them; a sawtooth would libel them.
    fn sample(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u32 << 24) as f32
        };
        (0..n)
            .map(|i| {
                let u1 = next().max(f32::MIN_POSITIVE);
                let u2 = next();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                normal * 0.02 * if i % 401 == 0 { 5.0 } else { 1.0 }
            })
            .collect()
    }

    fn relative_error(original: &[f32], restored: &[f32]) -> f32 {
        let num: f32 = original
            .iter()
            .zip(restored)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f32 = original.iter().map(|a| a * a).sum();
        (num / den).sqrt()
    }

    /// The types refused without an importance matrix: every two-bit and
    /// one-bit lattice file, IQ3_XXS and Q2_K_S — and nothing that merely
    /// improves with one.
    #[test]
    fn the_formats_below_three_bits_need_a_matrix() {
        let needing: Vec<&str> = Ftype::ALL
            .iter()
            .filter(|f| f.needs_importance())
            .map(|f| f.name())
            .collect();
        assert_eq!(
            needing,
            [
                "IQ3_XXS", "Q2_K_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"
            ]
        );
    }

    #[test]
    fn bf16_keeps_the_exponent_and_rounds_the_mantissa() {
        assert_eq!(bf16_bytes(1.0), [0x80, 0x3f]);
        assert_eq!(bf16_bytes(-2.0), [0x00, 0xc0]);
        // A value f16 cannot hold at all still round-trips in bf16.
        let big = 1.0e30f32;
        let restored = decode(GGML_TYPE_BF16, &bf16_bytes(big), 1).unwrap()[0];
        assert!((restored / big - 1.0).abs() < 0.01, "{restored}");
    }

    #[test]
    fn every_encoder_round_trips_within_its_precision() {
        let values = sample(2048);
        // Each bound is just above the error the encoder actually
        // achieves on this sample, and each is within a few percent of the
        // theoretical one for that many levels over a Gaussian block — so a
        // regression that costs real precision fails here rather than
        // passing under a bound set loosely enough to hide it.
        for (ggml_type, tolerance) in [
            (GGML_TYPE_F32, 0.0),
            (GGML_TYPE_F16, 3e-4),
            (GGML_TYPE_BF16, 2e-3),
            (GGML_TYPE_Q6_K, 2e-2),
            (GGML_TYPE_Q5_K, 3.8e-2),
            (GGML_TYPE_Q4_K, 7.5e-2),
            (GGML_TYPE_IQ4_XS, 8.2e-2),
            (GGML_TYPE_IQ4_NL, 8.1e-2),
            (GGML_TYPE_Q5_1, 4.0e-2),
            (GGML_TYPE_Q5_0, 4.7e-2),
            (GGML_TYPE_Q4_0, 9.4e-2),
            (GGML_TYPE_Q3_K, 1.6e-1),
            (GGML_TYPE_Q2_K, 3.1e-1),
        ] {
            let bytes = encode(ggml_type, &values, values.len());
            assert_eq!(
                bytes.len(),
                row_bytes(ggml_type, values.len()),
                "{} size",
                type_name(ggml_type)
            );
            let restored = dequantize(ggml_type, &bytes, values.len());
            let error = relative_error(&values, &restored);
            assert!(
                error <= tolerance,
                "{}: relative error {error} over {tolerance}",
                type_name(ggml_type)
            );
        }
    }

    /// Tensors outside a block first, then blocks by number, not by text.
    #[test]
    fn quantized_files_list_tensors_block_by_block() {
        let mut names = vec![
            "blk.10.attn_v.weight",
            "output.weight",
            "blk.2.ffn_down.weight",
            "blk.2.attn_v.weight",
            "token_embd.weight",
            "blk.0.attn_v.weight",
        ];
        sort_for_quantizing(&mut names, |n| n);
        assert_eq!(
            names,
            [
                "output.weight",
                "token_embd.weight",
                "blk.0.attn_v.weight",
                "blk.2.attn_v.weight",
                "blk.2.ffn_down.weight",
                "blk.10.attn_v.weight",
            ]
        );
    }

    /// More bits must mean less error — a mixture built on types that do
    /// not order this way would be spending size for nothing.
    #[test]
    fn the_types_order_by_accuracy() {
        let values = sample(1024);
        let error = |t: u32| {
            relative_error(
                &values,
                &dequantize(t, &encode(t, &values, values.len()), values.len()),
            )
        };
        assert!(error(GGML_TYPE_Q6_K) < error(GGML_TYPE_Q5_K));
        assert!(error(GGML_TYPE_Q5_K) < error(GGML_TYPE_Q4_K));
        assert!(error(GGML_TYPE_Q4_K) < error(GGML_TYPE_Q3_K));
        assert!(error(GGML_TYPE_Q3_K) < error(GGML_TYPE_Q2_K));
        // The two non-linear four-bit types sit beside the linear one
        // rather than above or below it — that is the whole claim of a
        // codebook, and it is worth pinning.
        assert!(error(GGML_TYPE_IQ4_XS) < error(GGML_TYPE_Q3_K));
        assert!(error(GGML_TYPE_IQ4_NL) < error(GGML_TYPE_Q3_K));
    }

    #[test]
    fn an_all_zero_block_encodes_and_reads_back_as_zero() {
        let values = vec![0.0f32; 512];
        for ggml_type in [
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ4_XS,
        ] {
            let bytes = encode(ggml_type, &values, values.len());
            let restored = dequantize(ggml_type, &bytes, values.len());
            assert!(
                restored.iter().all(|v| *v == 0.0),
                "{} produced non-zero values from zeros",
                type_name(ggml_type)
            );
        }
    }

    /// A dense model with grouped-query attention: 24 blocks, four query
    /// heads per key/value head.
    fn model() -> Model {
        Model {
            architecture: "llama".into(),
            layers: 24,
            heads: 16,
            kv_heads: 4,
            ..Default::default()
        }
    }

    /// The tensors of a dense model with `model.layers` blocks, in file
    /// order, with a separate output projection when `output` is set.
    fn dense(model: &Model, output: bool) -> Vec<(String, Vec<u64>)> {
        let mut out = vec![("token_embd.weight".to_string(), vec![2048, 32768])];
        for l in 0..model.layers {
            for (name, dims) in [
                ("attn_norm.weight", vec![2048]),
                ("attn_q.weight", vec![2048, 2048]),
                ("attn_k.weight", vec![2048, 512]),
                ("attn_v.weight", vec![2048, 512]),
                ("attn_output.weight", vec![2048, 2048]),
                ("ffn_gate.weight", vec![2048, 8192]),
                ("ffn_up.weight", vec![2048, 8192]),
                ("ffn_down.weight", vec![8192, 2048]),
            ] {
                out.push((format!("blk.{l}.{name}"), dims));
            }
        }
        out.push(("output_norm.weight".into(), vec![2048]));
        if output {
            out.push(("output.weight".into(), vec![2048, 32768]));
        }
        out
    }

    /// Plans `tensors` and returns each one's type by name.
    fn plan_all(
        ftype: Ftype,
        tensors: &[(String, Vec<u64>)],
        model: &Model,
    ) -> std::collections::HashMap<String, Plan> {
        let inputs: Vec<Tensor<'_>> = tensors
            .iter()
            .map(|(name, dims)| Tensor {
                name,
                dims,
                source: GGML_TYPE_BF16,
            })
            .collect();
        tensors
            .iter()
            .map(|(n, _)| n.clone())
            .zip(plan_tensors(ftype, &inputs, model))
            .collect()
    }

    /// The mixture is the whole point of the M in Q4_K_M.
    #[test]
    fn q4_k_m_spends_its_extra_bits_where_the_rules_say() {
        let model = model();
        let plans = plan_all(Ftype::Q4KM, &dense(&model, true), &model);
        let at = |name: &str| plans[name].ggml_type;
        assert_eq!(at("output.weight"), GGML_TYPE_Q6_K);
        assert_eq!(at("token_embd.weight"), GGML_TYPE_Q4_K);
        // The outer blocks get six bits, the middle ones four.
        assert_eq!(at("blk.0.attn_v.weight"), GGML_TYPE_Q6_K);
        assert_eq!(at("blk.23.ffn_down.weight"), GGML_TYPE_Q6_K);
        assert_eq!(at("blk.4.attn_v.weight"), GGML_TYPE_Q4_K);
        assert_eq!(at("blk.4.attn_q.weight"), GGML_TYPE_Q4_K);
        assert_eq!(at("blk.4.attn_norm.weight"), GGML_TYPE_F32);
        // Norms are never quantized.
        for ftype in Ftype::ALL {
            let plans = plan_all(ftype, &dense(&model, true), &model);
            assert_eq!(
                plans["blk.4.attn_norm.weight"].ggml_type,
                GGML_TYPE_F32,
                "{} quantized a norm",
                ftype.name()
            );
        }
    }

    /// With no separate output projection the token embedding is the
    /// output, and is carried as one.
    #[test]
    fn a_tied_embedding_is_carried_as_the_output() {
        let model = model();
        let plans = plan_all(Ftype::Q4KM, &dense(&model, false), &model);
        assert_eq!(plans["token_embd.weight"].ggml_type, GGML_TYPE_Q6_K);
        // A row no K-quant block divides takes eight bits instead.
        let tensors = vec![("token_embd.weight".to_string(), vec![960, 49152])];
        let plans = plan_all(Ftype::Q4KM, &tensors, &model);
        assert_eq!(plans["token_embd.weight"].ggml_type, GGML_TYPE_Q8_0);
    }

    /// A row that no block divides falls back to the 32-wide type of about
    /// the same precision, and says that it did.
    #[test]
    fn an_awkward_row_length_falls_back_and_is_recorded() {
        let model = model();
        let tensors = vec![
            ("blk.1.attn_q.weight".to_string(), vec![896, 896]),
            ("blk.1.ffn_up.weight".to_string(), vec![960, 2560]),
            ("blk.1.attn_k.weight".to_string(), vec![2048, 2048]),
            ("output.weight".to_string(), vec![100, 32]),
        ];
        let plans = plan_all(Ftype::Q4KM, &tensors, &model);
        let q = plans["blk.1.attn_q.weight"];
        assert_eq!(q.ggml_type, GGML_TYPE_Q5_0);
        assert_eq!(q.fallback_from, Some(GGML_TYPE_Q4_K));
        let plans3 = plan_all(Ftype::Q3KS, &tensors, &model);
        assert_eq!(plans3["blk.1.ffn_up.weight"].ggml_type, GGML_TYPE_Q4_0);
        let plans6 = plan_all(Ftype::Q6K, &tensors, &model);
        assert_eq!(plans6["blk.1.ffn_up.weight"].ggml_type, GGML_TYPE_Q8_0);
        // Not even 32 divides 100: f16, which needs no block.
        let out = plans["output.weight"];
        assert_eq!(out.ggml_type, GGML_TYPE_F16);
        assert_eq!(out.fallback_from, Some(GGML_TYPE_Q8_0));
        assert_eq!(plans["blk.1.attn_k.weight"].fallback_from, None);
    }

    /// The suffix is a promise about which tensors are carried above the
    /// base type. Each of these is the difference between two file types
    /// that would otherwise be the same file.
    #[test]
    fn the_suffixes_differ_where_they_claim_to() {
        let model = model();
        let tensors = dense(&model, true);
        let plan = |ftype, name: &str| plan_all(ftype, &tensors, &model)[name].ggml_type;

        // S carries nothing extra in the middle of the stack; M and L do.
        assert_eq!(plan(Ftype::Q3KS, "blk.12.ffn_down.weight"), GGML_TYPE_Q3_K);
        assert_eq!(plan(Ftype::Q3KM, "blk.12.ffn_down.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q3KL, "blk.12.ffn_down.weight"), GGML_TYPE_Q5_K);

        assert_eq!(plan(Ftype::Q4KS, "blk.12.ffn_down.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q4KM, "blk.0.ffn_down.weight"), GGML_TYPE_Q6_K);
        assert_eq!(plan(Ftype::Q5KM, "blk.0.attn_v.weight"), GGML_TYPE_Q6_K);
        assert_eq!(plan(Ftype::Q5KS, "blk.0.attn_v.weight"), GGML_TYPE_Q5_K);

        // Two bits never lands on the tensors that cannot take it.
        assert_eq!(plan(Ftype::Q2K, "blk.12.ffn_down.weight"), GGML_TYPE_Q3_K);
        assert_eq!(plan(Ftype::Q2K, "blk.12.attn_v.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q2K, "blk.12.attn_q.weight"), GGML_TYPE_Q2_K);

        // The vocabulary projection is six bits in every quantized file —
        // five below three bits, where six would be a fifth of the file —
        // and the plain body type everywhere it is not named.
        for ftype in Ftype::ALL {
            let output = plan(ftype, "output.weight");
            let body = plan(ftype, "blk.12.attn_q.weight");
            use Ftype::*;
            let lowest = matches!(ftype, IQ1S | IQ1M | IQ2XXS | IQ2XS | IQ2S | IQ2M | IQ3XXS);
            if !ftype.is_quantized() {
                assert_eq!(output, ftype.base_type(), "{}", ftype.name());
            } else if lowest {
                assert_eq!(output, GGML_TYPE_Q5_K, "{}", ftype.name());
            } else {
                assert_eq!(output, GGML_TYPE_Q6_K, "{}", ftype.name());
            }
            // The two 3-bit lattice files that spend less on the query
            // projection.
            let expected = match ftype {
                IQ3XS => GGML_TYPE_IQ3_XXS,
                IQ3XXS => GGML_TYPE_IQ2_S,
                _ => ftype.base_type(),
            };
            if ftype.is_quantized() {
                assert_eq!(body, expected, "{}", ftype.name());
            }
        }
    }

    /// Grouped-query attention makes the value projection cheap to carry
    /// high, and the rules only spend there when the ratio earns it.
    #[test]
    fn grouped_query_attention_changes_where_the_bits_go() {
        let grouped = model();
        let plain = Model {
            kv_heads: 16,
            ..model()
        };
        let at = |model: &Model| {
            plan_all(Ftype::Q2K, &dense(model, true), model)["blk.12.attn_v.weight"].ggml_type
        };
        assert_eq!(at(&grouped), GGML_TYPE_Q4_K);
        assert_eq!(at(&plain), GGML_TYPE_Q3_K);
    }

    /// The router and the other small steering tensors are never
    /// quantized, and keep the type the source had.
    #[test]
    fn a_router_keeps_its_source_type() {
        let model = Model {
            experts: 64,
            ..model()
        };
        let tensors = vec![
            ("blk.0.ffn_gate_inp.weight".to_string(), vec![2048, 64]),
            (
                "blk.0.ffn_down_exps.weight".to_string(),
                vec![768, 2048, 64],
            ),
        ];
        let plans = plan_all(Ftype::Q4KM, &tensors, &model);
        assert_eq!(plans["blk.0.ffn_gate_inp.weight"].ggml_type, GGML_TYPE_F32);
        // A mixture's down projections are placed by the layer in their
        // name: block 0 is an outer block.
        assert_eq!(
            plans["blk.0.ffn_down_exps.weight"].ggml_type,
            GGML_TYPE_Q6_K
        );
    }

    /// Every tensor of every published K-quant in the models directory,
    /// planned again from its 16-bit reference: the same type each time.
    /// Needs bartowski's SmolLM2-360M-Instruct (F16 and its quants) and
    /// Llama-3.2-1B-Instruct. Mixture-of-experts files are left out: their
    /// publishers pick many tensors' types by hand.
    /// `cargo test -- --ignored published_mixtures`.
    #[test]
    #[ignore]
    fn published_mixtures_are_reproduced() {
        use orangu::gguf::GgufFile;
        let models = crate::hf::models_dir().expect("a models directory");
        let mut checked = 0;
        let mut wrong = Vec::new();
        for (repo, reference) in [
            ("bartowski/SmolLM2-360M-Instruct-GGUF", "F16"),
            ("bartowski/Llama-3.2-1B-Instruct-GGUF", "F16"),
        ] {
            let folder = models.join(orangu::model_download::repo_folder_name(repo));
            let commit = std::fs::read_to_string(folder.join("refs/main")).unwrap();
            let dir = folder.join("snapshots").join(commit.trim());
            let find = |tag: &str| {
                orangu::model_spec::resolve_show_target(&models, &format!("{repo}:{tag}")).unwrap()
            };
            let source = GgufFile::open(&find(reference)).unwrap();
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .collect();
            files.sort();
            for path in files {
                let file = GgufFile::open(&path).unwrap();
                let file_type = file
                    .metadata
                    .iter()
                    .find(|(k, _)| k == "general.file_type")
                    .and_then(|(_, v)| v.as_u64())
                    .unwrap_or(0) as u32;
                // `_L` and `_XL` files carry hand-picked embedding and output
                // types on top of a mixture; only the standard ones are
                // compared.
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                let Some(ftype) = Ftype::from_file_type(file_type) else {
                    continue;
                };
                if !ftype.is_quantized() || stem.ends_with("_L") || stem.ends_with("_XL") {
                    continue;
                }
                let model = Model {
                    imatrix: file
                        .metadata
                        .iter()
                        .any(|(k, _)| k.starts_with("quantize.imatrix.")),
                    ..Model::from_metadata(&file.metadata)
                };
                let inputs: Vec<Tensor<'_>> = source
                    .tensors
                    .iter()
                    .map(|t| Tensor {
                        name: &t.name,
                        dims: &t.dims,
                        source: t.ggml_type,
                    })
                    .collect();
                let plans = plan_tensors(ftype, &inputs, &model);
                eprintln!("{stem}: {} tensors as {}", plans.len(), ftype.name());
                for (tensor, plan) in source.tensors.iter().zip(plans) {
                    let published = file
                        .tensors
                        .iter()
                        .find(|t| t.name == tensor.name)
                        .unwrap()
                        .ggml_type;
                    checked += 1;
                    // Files made before the fallback for two and three bits
                    // moved from IQ4_NL to Q4_0 carry the older choice.
                    let older = plan.fallback_from.is_some()
                        && plan.ggml_type == GGML_TYPE_Q4_0
                        && published == GGML_TYPE_IQ4_NL;
                    if published != plan.ggml_type && !older {
                        wrong.push(format!(
                            "{stem}: {} is {} here, {} published",
                            tensor.name,
                            type_name(plan.ggml_type),
                            type_name(published)
                        ));
                    }
                }
            }
        }
        for line in wrong.iter().take(40) {
            eprintln!("{line}");
        }
        assert!(checked > 0, "no published file was found");
        assert!(
            wrong.is_empty(),
            "{} of {checked} tensors differ",
            wrong.len()
        );
    }

    #[test]
    fn a_quantized_source_is_refused_by_name() {
        let err = decode(GGML_TYPE_Q4_K, &[0; 144], 256)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Q4_K"), "{err}");
        assert!(err.contains("BF16"), "{err}");
    }

    #[test]
    fn file_types_parse_by_every_spelling_they_are_offered_under() {
        assert_eq!(Ftype::parse("bf16").unwrap(), Ftype::Bf16);
        assert_eq!(Ftype::parse("q4_k_m").unwrap(), Ftype::Q4KM);
        assert_eq!(Ftype::parse("Q3_K_L").unwrap(), Ftype::Q3KL);
        assert_eq!(Ftype::parse("iq4_xs").unwrap(), Ftype::IQ4XS);
        // Every name this tool prints has to parse back.
        for ftype in Ftype::ALL {
            assert_eq!(Ftype::parse(ftype.name()).unwrap(), ftype);
        }
    }

    /// A name the format defines but this tool will not write should say
    /// why, not "unknown".
    #[test]
    fn the_types_that_are_not_written_say_why() {
        assert_eq!(Ftype::parse("iq2_xs").unwrap(), Ftype::IQ2XS);
        let refused = |name: &str| Ftype::parse(name).unwrap_err().to_string();

        let q8 = refused("q8_0");
        assert!(q8.contains("Q6_K"), "{q8}");

        let nonsense = refused("q9_z");
        assert!(nonsense.contains("unknown"), "{nonsense}");
    }
}

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
//!   back — to a 32-wide block where one exists, and to `f16` when even
//!   that does not fit. Falling back silently would be the wrong thing to
//!   do, so [`Plan`] records every fallback for the caller to report.

use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

// The types this tool writes, and the ones it only has to name. The
// round-number quantizations — `Q4_0`, `Q5_0`, `Q8_0` and the rest — are
// deliberately absent: they are the pre-K-quant generation, a K-quant beats
// them at every size, and a writer that offers both mostly offers a way to
// pick the worse one.
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
pub const GGML_TYPE_IQ2_XS: u32 = 17;
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
pub const GGML_TYPE_IQ1_S: u32 = 19;
pub const GGML_TYPE_IQ4_NL: u32 = 20;
pub const GGML_TYPE_IQ3_S: u32 = 21;
pub const GGML_TYPE_IQ2_S: u32 = 22;
pub const GGML_TYPE_IQ4_XS: u32 = 23;
pub const GGML_TYPE_I8: u32 = 24;
pub const GGML_TYPE_I16: u32 = 25;
pub const GGML_TYPE_I32: u32 = 26;
pub const GGML_TYPE_I64: u32 = 27;
pub const GGML_TYPE_F64: u32 = 28;
pub const GGML_TYPE_IQ1_M: u32 = 29;
pub const GGML_TYPE_BF16: u32 = 30;

/// The super-block width every K-quant works in.
const QK_K: usize = 256;
/// The block width `IQ4_NL` works in — the only 32-wide type left, and so
/// the only place a row that does not divide 256 can still be four bits.
const QK: usize = 32;

/// `IQ4_NL`'s sixteen levels. They are not evenly spaced: a trained weight
/// is roughly Gaussian, so levels crowd near zero where the values are and
/// spread out into the tail where they are not. That is the whole trick,
/// and it is why four non-linear bits beat four linear ones.
const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
/// Below this, a block is all zeros and its scale is zero.
const MAX_EPS: f32 = 1e-15;

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
}

/// The file types that exist in the format but that this tool will not
/// write, and the reason, so `--quantization iq2_xs` says something more
/// useful than "unknown".
///
/// Every one of them is a search against a fixed codebook of lattice
/// points, and below about three bits that search only lands anywhere
/// useful when it is told which weights matter — an importance matrix,
/// measured by running the model over calibration text. Without one the
/// reference implementation refuses outright rather than write a file that
/// looks fine and answers badly. This tool has no importance matrix pass
/// yet, so it refuses for the same reason.
const NEEDS_IMPORTANCE: [&str; 10] = [
    "q2_k_s", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_s", "iq3_m",
];

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
            other if NEEDS_IMPORTANCE.contains(&other) => bail!(
                "{} needs an importance matrix, which this tool does not measure yet — \
                 at that many bits a file written without one is worse than the next size up",
                other.to_ascii_uppercase()
            ),
            other
                if other.starts_with("q4_0")
                    || other.starts_with("q4_1")
                    || other.starts_with("q5_0")
                    || other.starts_with("q5_1")
                    || other.starts_with("q8_0")
                    || other.starts_with("q8_1") =>
            {
                bail!(
                    "{} is a pre-K-quant format and is not written — Q6_K is the same size and more accurate",
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
    pub const ALL: [Ftype; 14] = [
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
        }
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
            Ftype::Bf16 => 32,
        }
    }

    /// The type most of the file is written as, before the mixture rules
    /// move individual tensors.
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
        }
    }

    /// Whether the file is quantized at all, which is what decides if the
    /// mixture rules run.
    fn is_quantized(&self) -> bool {
        !matches!(self, Ftype::F32 | Ftype::F16 | Ftype::Bf16)
    }
}

/// Where a tensor sits in the network, which is what the mixture rules key
/// off. Derived from the tensor's name so it works for a file this tool
/// wrote and for one it is only reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    TokenEmbedding,
    Output,
    AttentionValue,
    AttentionOutput,
    FeedForwardDown,
    Other,
}

pub fn role_of(name: &str) -> Role {
    if name == "token_embd.weight" {
        Role::TokenEmbedding
    } else if name == "output.weight" {
        Role::Output
    } else if name.ends_with("attn_v.weight") {
        Role::AttentionValue
    } else if name.ends_with("attn_output.weight") {
        Role::AttentionOutput
    } else if name.ends_with("ffn_down.weight") {
        Role::FeedForwardDown
    } else {
        Role::Other
    }
}

/// The block index in a `blk.N.` tensor name.
pub fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// The rule that spends the extra bits on the outer blocks and on every
/// third block in between, rather than spreading them evenly.
fn use_more_bits(layer: usize, layers: usize) -> bool {
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}

/// What the mixture rules need to know about the model beyond one tensor's
/// own name and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub layers: usize,
    /// Query heads per key/value head. Grouped-query attention makes the
    /// value projection small enough that carrying it at a higher precision
    /// costs almost nothing, so the rules spend bits there when it is 4 or
    /// more.
    pub gqa: usize,
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

/// Chooses the type for one tensor.
///
/// `dims` is in GGUF order, so `dims[0]` is the row length — the dimension
/// a block has to divide.
pub fn plan_tensor(ftype: Ftype, name: &str, dims: &[u64], model: Model) -> Plan {
    // Anything that is not a matrix — every norm — stays f32, in every
    // file type. They are the divisor of every activation and a rounding
    // error's worth of file size, and a reader expects to find them exact.
    if dims.len() < 2 {
        return Plan {
            ggml_type: GGML_TYPE_F32,
            fallback_from: None,
        };
    }

    let wanted = if ftype.is_quantized() {
        mixture(ftype, name, model)
    } else {
        ftype.base_type()
    };

    let ncols = dims[0] as usize;
    let ggml_type = fit(wanted, ncols);
    Plan {
        ggml_type,
        fallback_from: (ggml_type != wanted).then_some(wanted),
    }
}

/// The per-tensor rules behind the `_S`/`_M`/`_L` in a file type's name.
///
/// A `Q4_K_M` file is not a file of `Q4_K` tensors, and the difference
/// between the three suffixes is entirely here: which tensors are carried
/// above the file's base type, and by how much. The tensors that earn it
/// are always the same ones — the vocabulary projection, because it is the
/// last thing before the softmax; the value and down projections, because
/// an error there lands on every token that attends through them.
fn mixture(ftype: Ftype, name: &str, model: Model) -> u32 {
    let base = ftype.base_type();
    let layers = model.layers.max(1);
    let layer = block_index(name).unwrap_or(0);

    match role_of(name) {
        // The vocabulary projection: six bits in every quantized file.
        Role::Output => GGML_TYPE_Q6_K,
        Role::TokenEmbedding => base,
        Role::AttentionValue => match ftype {
            Ftype::Q2K if model.gqa >= 4 => GGML_TYPE_Q4_K,
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM if layer < 2 => GGML_TYPE_Q5_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            Ftype::IQ4NL | Ftype::IQ4XS if model.gqa >= 4 => GGML_TYPE_Q5_K,
            Ftype::Q4KM | Ftype::Q5KM if use_more_bits(layer, layers) => GGML_TYPE_Q6_K,
            Ftype::Q4KS if layer < 4 => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::FeedForwardDown => match ftype {
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM if layer < layers / 16 => GGML_TYPE_Q5_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            Ftype::Q4KM | Ftype::Q5KM if use_more_bits(layer, layers) => GGML_TYPE_Q6_K,
            Ftype::Q4KS | Ftype::IQ4NL | Ftype::IQ4XS if layer < layers / 8 => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::AttentionOutput => match ftype {
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::Other => base,
    }
}

/// Demotes a type whose block does not divide the row length.
///
/// A 256-wide K-quant that does not fit has one place left to go: `IQ4_NL`
/// is the only 32-wide type here, so anything at four bits or below lands
/// there. Five and six bits do not — dropping them to four to save a fifth
/// of a tensor is the wrong trade on the two tensors that are carried high
/// precisely because they matter — so those fall to `f16`, which needs no
/// block at all.
fn fit(wanted: u32, ncols: usize) -> u32 {
    if ncols.is_multiple_of(block_size(wanted)) {
        return wanted;
    }
    let demoted = match wanted {
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_IQ4_XS => GGML_TYPE_IQ4_NL,
        other => other,
    };
    if ncols.is_multiple_of(block_size(demoted)) {
        demoted
    } else {
        GGML_TYPE_F16
    }
}

pub fn block_size(ggml_type: u32) -> usize {
    match ggml_type {
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
        | GGML_TYPE_Q8_K | GGML_TYPE_IQ2_XXS | GGML_TYPE_IQ2_XS | GGML_TYPE_IQ2_S
        | GGML_TYPE_IQ3_XXS | GGML_TYPE_IQ3_S | GGML_TYPE_IQ1_S | GGML_TYPE_IQ1_M
        | GGML_TYPE_IQ4_XS => QK_K,
        GGML_TYPE_IQ4_NL => QK,
        _ => 1,
    }
}

/// Every type the format defines that this tool has any business naming —
/// what it writes, and what it may find in a file it is asked to convert.
pub fn type_name(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_F64 => "F64",
        GGML_TYPE_I8 => "I8",
        GGML_TYPE_I16 => "I16",
        GGML_TYPE_I32 => "I32",
        GGML_TYPE_I64 => "I64",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_IQ1_S => "IQ1_S",
        GGML_TYPE_IQ1_M => "IQ1_M",
        GGML_TYPE_IQ2_XXS => "IQ2_XXS",
        GGML_TYPE_IQ2_XS => "IQ2_XS",
        GGML_TYPE_IQ2_S => "IQ2_S",
        GGML_TYPE_IQ3_XXS => "IQ3_XXS",
        GGML_TYPE_IQ3_S => "IQ3_S",
        GGML_TYPE_IQ4_NL => "IQ4_NL",
        GGML_TYPE_IQ4_XS => "IQ4_XS",
        _ => "?",
    }
}

/// Bytes one row of `ncols` elements takes.
pub fn row_bytes(ggml_type: u32, ncols: usize) -> usize {
    match ggml_type {
        GGML_TYPE_F64 | GGML_TYPE_I64 => ncols * 8,
        GGML_TYPE_F32 | GGML_TYPE_I32 => ncols * 4,
        GGML_TYPE_F16 | GGML_TYPE_BF16 | GGML_TYPE_I16 => ncols * 2,
        GGML_TYPE_I8 => ncols,
        GGML_TYPE_Q2_K => ncols / QK_K * (QK_K / 16 + QK_K / 4 + 2 + 2),
        GGML_TYPE_Q3_K => ncols / QK_K * (QK_K / 8 + QK_K / 4 + 12 + 2),
        GGML_TYPE_Q4_K => ncols / QK_K * (2 + 2 + 12 + QK_K / 2),
        GGML_TYPE_Q5_K => ncols / QK_K * (2 + 2 + 12 + QK_K / 8 + QK_K / 2),
        GGML_TYPE_Q6_K => ncols / QK_K * (QK_K / 2 + QK_K / 4 + QK_K / 16 + 2),
        GGML_TYPE_IQ4_NL => ncols / QK * (2 + QK / 2),
        GGML_TYPE_IQ4_XS => ncols / QK_K * (2 + 2 + QK_K / 64 + QK_K / 2),
        _ => 0,
    }
}

/// Encodes a whole tensor. `values` is row-major with `ncols` per row.
///
/// Rows are encoded in parallel groups. Every block-quantized type this
/// writes has a block that divides the row length (that is what
/// [`plan_tensor`] guarantees), so a group boundary is always a block
/// boundary and the result is byte-identical to encoding the tensor in one
/// pass.
pub fn encode(ggml_type: u32, values: &[f32], ncols: usize) -> Vec<u8> {
    debug_assert!(ncols > 0 && values.len().is_multiple_of(ncols));
    debug_assert_eq!(ncols % block_size(ggml_type), 0);
    let rows = values.len() / ncols;
    // Big enough that a group is real work, small enough that a tall thin
    // tensor still spreads over the pool.
    let group = (rows / (rayon::current_num_threads() * 4).max(1)).max(1);
    values
        .par_chunks(group * ncols)
        .map(|chunk| encode_chunk(ggml_type, chunk))
        .collect::<Vec<_>>()
        .concat()
        .tap_len(ggml_type, values.len(), ncols)
}

fn encode_chunk(ggml_type: u32, values: &[f32]) -> Vec<u8> {
    match ggml_type {
        GGML_TYPE_F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GGML_TYPE_F16 => values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        GGML_TYPE_BF16 => values.iter().flat_map(|&v| bf16_bytes(v)).collect(),
        GGML_TYPE_Q2_K => blocks(values, QK_K, quantize_q2_k),
        GGML_TYPE_Q3_K => blocks(values, QK_K, quantize_q3_k),
        GGML_TYPE_Q4_K => blocks(values, QK_K, quantize_q4_k),
        GGML_TYPE_Q5_K => blocks(values, QK_K, quantize_q5_k),
        GGML_TYPE_Q6_K => blocks(values, QK_K, quantize_q6_k),
        GGML_TYPE_IQ4_NL => blocks(values, QK, quantize_iq4_nl),
        GGML_TYPE_IQ4_XS => blocks(values, QK_K, quantize_iq4_xs),
        _ => {
            debug_assert!(false, "no encoder for ggml type {ggml_type}");
            Vec::new()
        }
    }
}

/// A trailing size check on every encode: a wrong block layout usually
/// shows up first as a tensor whose bytes do not add up, and finding that
/// here beats finding it in a reader.
trait TapLen {
    fn tap_len(self, ggml_type: u32, elements: usize, ncols: usize) -> Vec<u8>;
}

impl TapLen for Vec<u8> {
    fn tap_len(self, ggml_type: u32, elements: usize, ncols: usize) -> Vec<u8> {
        debug_assert_eq!(
            self.len(),
            row_bytes(ggml_type, ncols) * (elements / ncols.max(1)),
            "{} encoded to the wrong size",
            type_name(ggml_type)
        );
        self
    }
}

fn blocks(values: &[f32], width: usize, encode: fn(&[f32], &mut Vec<u8>)) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len());
    for block in values.chunks(width) {
        encode(block, &mut out);
    }
    out
}

/// `bf16` is `f32` with the low 16 bits of the mantissa dropped, rounded to
/// nearest even — which is why it is the natural way to halve a trained
/// weight: the exponent, and so the dynamic range, is untouched.
fn bf16_bytes(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16 | 0x0040).to_le_bytes();
    }
    let rounded = bits + 0x7fff + ((bits >> 16) & 1);
    ((rounded >> 16) as u16).to_le_bytes()
}

#[inline]
fn nearest(value: f32) -> i32 {
    // Round half away from zero, matching the reference encoders.
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        -((-value + 0.5) as i32)
    }
}

/// Copies one block into a fixed 256-wide buffer, zero-filling a short
/// tail. Every K-quant starts here, and none of them may read past what it
/// was given.
fn super_block(block: &[f32]) -> [f32; QK_K] {
    let mut values = [0.0f32; QK_K];
    let n = block.len().min(QK_K);
    values[..n].copy_from_slice(&block[..n]);
    values
}

/// 256 values as sixteen 16-wide sub-blocks, each with a 4-bit scale and a
/// 4-bit minimum against a shared `f16` pair, and 2-bit quants.
///
/// Two bits is four levels for sixteen values, which is why this one fits
/// per sub-block rather than per block, and why it weights by magnitude and
/// scores by absolute error: at this width the fit is decided by the values
/// it gets worst.
fn quantize_q2_k(block: &[f32], out: &mut Vec<u8>) {
    let values = super_block(block);
    let mut scales = [0.0f32; QK_K / 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut quants = [0u8; QK_K];
    let mut weights = [0.0f32; 16];

    for j in 0..QK_K / 16 {
        let sub = &values[16 * j..16 * (j + 1)];
        for (w, &v) in weights.iter_mut().zip(sub.iter()) {
            *w = v.abs();
        }
        let (scale, min) =
            fit_scale_and_min(sub, &weights, 3, NARROW, &mut quants[16 * j..16 * (j + 1)]);
        scales[j] = scale;
        mins[j] = min;
    }

    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);

    // Four bits for each of the sixteen scales and sixteen minimums, both
    // against their own `f16`.
    let mut packed = [0u8; QK_K / 16];
    let d = if max_scale > 0.0 {
        let inv = 15.0 / max_scale;
        for (byte, &scale) in packed.iter_mut().zip(scales.iter()) {
            *byte = nearest(inv * scale).clamp(0, 15) as u8;
        }
        f16::from_f32(max_scale / 15.0)
    } else {
        f16::from_f32(0.0)
    };
    let dmin = if max_min > 0.0 {
        let inv = 15.0 / max_min;
        for (byte, &min) in packed.iter_mut().zip(mins.iter()) {
            *byte |= (nearest(inv * min).clamp(0, 15) as u8) << 4;
        }
        f16::from_f32(max_min / 15.0)
    } else {
        f16::from_f32(0.0)
    };

    // Requantize against the scales as a reader reconstructs them.
    for j in 0..QK_K / 16 {
        let scale = d.to_f32() * (packed[j] & 0xF) as f32;
        if scale == 0.0 {
            continue;
        }
        let offset = dmin.to_f32() * (packed[j] >> 4) as f32;
        for i in 0..16 {
            quants[16 * j + i] = nearest((values[16 * j + i] + offset) / scale).clamp(0, 3) as u8;
        }
    }

    out.extend_from_slice(&packed);
    for chunk in 0..QK_K / 128 {
        for i in 0..32 {
            let at = chunk * 128 + i;
            out.push(
                quants[at]
                    | (quants[at + 32] << 2)
                    | (quants[at + 64] << 4)
                    | (quants[at + 96] << 6),
            );
        }
    }
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
}

/// 256 values as sixteen 16-wide sub-blocks with signed 6-bit scales
/// against one `f16`, and 3-bit quants: two bits in a packed byte and the
/// third in a separate mask.
///
/// There is no minimum here — the quants are signed — so this one fits a
/// symmetric scale per sub-block, refined by the coordinate sweep that
/// three bits repays and eight would not.
fn quantize_q3_k(block: &[f32], out: &mut Vec<u8>) {
    let values = super_block(block);
    let mut quants = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for j in 0..QK_K / 16 {
        let scale = fit_signed_scale(
            &values[16 * j..16 * (j + 1)],
            4,
            &mut quants[16 * j..16 * (j + 1)],
        );
        scales[j] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }

    // Twelve bytes: four bits per scale for sixteen scales, with the top
    // two bits of each folded into the last four bytes.
    let mut packed = [0u8; 12];
    let d = if max_abs_scale > 0.0 {
        let inv = -32.0 / max_scale;
        for j in 0..QK_K / 16 {
            let mut l = (nearest(inv * scales[j]).clamp(-32, 31) + 32) as u8;
            if j < 8 {
                packed[j] = l & 0xF;
            } else {
                packed[j - 8] |= (l & 0xF) << 4;
            }
            l >>= 4;
            packed[j % 4 + 8] |= l << (2 * (j / 4));
        }
        f16::from_f32(1.0 / inv)
    } else {
        f16::from_f32(0.0)
    };

    let mut levels = [0u8; QK_K];
    for j in 0..QK_K / 16 {
        let scale = d.to_f32() * unpack_signed_scale(j, &packed) as f32;
        if scale == 0.0 {
            continue;
        }
        for i in 0..16 {
            let l = nearest(values[16 * j + i] / scale).clamp(-4, 3);
            levels[16 * j + i] = (l + 4) as u8;
        }
    }

    // The high bit of all 256 quants, eight values to a byte position: the
    // first 32 in bit 0, the next 32 in bit 1, and so on.
    let mut mask = [0u8; QK_K / 8];
    let mut at = 0usize;
    let mut bit = 1u8;
    for level in levels.iter_mut() {
        if *level > 3 {
            mask[at] |= bit;
            *level -= 4;
        }
        at += 1;
        if at == QK_K / 8 {
            at = 0;
            bit <<= 1;
        }
    }

    out.extend_from_slice(&mask);
    for chunk in 0..QK_K / 128 {
        for i in 0..32 {
            let at = chunk * 128 + i;
            out.push(
                levels[at]
                    | (levels[at + 32] << 2)
                    | (levels[at + 64] << 4)
                    | (levels[at + 96] << 6),
            );
        }
    }
    out.extend_from_slice(&packed);
    out.extend_from_slice(&d.to_le_bytes());
}

/// The signed 6-bit scale for sub-block `j`, read back out of the 12 packed
/// bytes exactly as a reader does.
fn unpack_signed_scale(j: usize, packed: &[u8; 12]) -> i32 {
    let low = if j < 8 {
        packed[j] & 0xF
    } else {
        packed[j - 8] >> 4
    };
    let high = (packed[8 + j % 4] >> (2 * (j / 4))) & 3;
    (low | (high << 4)) as i32 - 32
}

/// 256 values as eight 32-wide sub-blocks, each with its own 6-bit scale
/// and 6-bit minimum against a shared `f16` pair, and 4-bit quants.
fn quantize_q4_k(block: &[f32], out: &mut Vec<u8>) {
    let mut values = [0.0f32; QK_K];
    values[..block.len().min(QK_K)].copy_from_slice(&block[..block.len().min(QK_K)]);

    let mut scales = [0.0f32; QK_K / 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut quants = [0u8; QK_K];
    let mut weights = [0.0f32; 32];

    for j in 0..QK_K / 32 {
        let sub = &values[32 * j..32 * (j + 1)];
        let sum_sq: f32 = sub.iter().map(|v| v * v).sum();
        let average = (sum_sq / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(sub.iter()) {
            *w = average + v.abs();
        }
        let (scale, min) =
            fit_scale_and_min(sub, &weights, 15, WIDE, &mut quants[32 * j..32 * (j + 1)]);
        scales[j] = scale;
        mins[j] = min;
    }

    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    // The 12 scale bytes: six bits each for eight scales and eight
    // minimums, with the top two bits of the last four of each pair folded
    // into the high bits of the first four.
    let mut packed = [0u8; 12];
    for j in 0..QK_K / 32 {
        let ls = (nearest(inv_scale * scales[j]).clamp(0, 63)) as u8;
        let lm = (nearest(inv_min * mins[j]).clamp(0, 63)) as u8;
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }

    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    // Requantize against the scales as they will be *read back*, not as
    // they were fitted: the 6-bit packing and the f16 scale both round, and
    // ignoring that costs more accuracy than the packing itself.
    for j in 0..QK_K / 32 {
        let (sc, m) = unpack_scale_min(j, &packed);
        let scale = d.to_f32() * sc as f32;
        if scale == 0.0 {
            continue;
        }
        let offset = dmin.to_f32() * m as f32;
        for i in 0..32 {
            let l = nearest((values[32 * j + i] + offset) / scale).clamp(0, 15);
            quants[32 * j + i] = l as u8;
        }
    }

    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&packed);
    for chunk in 0..QK_K / 64 {
        for i in 0..32 {
            let low = quants[chunk * 64 + i];
            let high = quants[chunk * 64 + i + 32];
            out.push(low | (high << 4));
        }
    }
}

/// The 6-bit scale and minimum for sub-block `j`, read back out of the 12
/// packed bytes exactly as a reader does.
fn unpack_scale_min(j: usize, packed: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        (
            (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4),
            (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4),
        )
    }
}

/// How wide a net [`fit_scale_and_min`] casts, and what it calls an error.
///
/// The two-bit search differs from the four- and five-bit one in both, and
/// not by accident: with four levels the fit is dominated by whichever
/// value lands worst, so absolute error finds a better compromise than
/// squared error, which would chase the outlier and give up the rest.
#[derive(Clone, Copy)]
struct Search {
    rmin: f32,
    rdelta: f32,
    steps: i32,
    absolute_error: bool,
}

/// The four- and five-bit search: 21 candidate scales, least squares.
const WIDE: Search = Search {
    rmin: -1.0,
    rdelta: 0.1,
    steps: 20,
    absolute_error: false,
};
/// The two-bit search: 16 candidates around a tighter start, absolute
/// error.
const NARROW: Search = Search {
    rmin: -0.5,
    rdelta: 0.1,
    steps: 15,
    absolute_error: true,
};
/// Five bits: the narrow sweep, but least squares like the wider types.
const NARROW_SQUARED: Search = Search {
    absolute_error: false,
    ..NARROW
};

/// Fits `x ~ scale * q + min` over one sub-block with `q` in `0..=nmax`,
/// weighting each value by `weights`.
///
/// The search is the established one: start from the range, then try a
/// sweep of slightly different scales and keep the best. It matters most on
/// the blocks that hold one large outlier, where fitting the range exactly
/// wastes every level on a value nothing else is near.
fn fit_scale_and_min(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    search: Search,
    quants: &mut [u8],
) -> (f32, f32) {
    let Search {
        rmin,
        rdelta,
        steps,
        absolute_error,
    } = search;
    let residual = |diff: f32| {
        if absolute_error {
            diff.abs()
        } else {
            diff * diff
        }
    };

    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for i in 1..x.len() {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        quants.fill(0);
        return (0.0, -min);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..x.len() {
        let l = nearest(iscale * (x[i] - min)).clamp(0, nmax);
        quants[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        best_error += weights[i] * residual(diff);
    }

    let mut candidate = vec![0u8; x.len()];
    for step in 0..=steps {
        iscale = (rmin + rdelta * step as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..x.len() {
            let l = nearest(iscale * (x[i] - min)).clamp(0, nmax);
            candidate[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l * l) as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant <= 0.0 {
            continue;
        }
        let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
        let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
        if this_min > 0.0 {
            this_min = 0.0;
            this_scale = sum_xl / sum_l2;
        }
        let mut error = 0.0f32;
        for i in 0..x.len() {
            let diff = this_scale * candidate[i] as f32 + this_min - x[i];
            error += weights[i] * residual(diff);
        }
        if error < best_error {
            quants.copy_from_slice(&candidate);
            best_error = error;
            scale = this_scale;
            min = this_min;
        }
    }
    (scale, -min)
}

/// 256 values as sixteen 16-wide sub-blocks with signed 8-bit scales
/// against one `f16`, and 6-bit quants split across a low nibble and a
/// high pair of bits.
fn quantize_q6_k(block: &[f32], out: &mut Vec<u8>) {
    let mut values = [0.0f32; QK_K];
    values[..block.len().min(QK_K)].copy_from_slice(&block[..block.len().min(QK_K)]);

    let mut quants = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for ib in 0..QK_K / 16 {
        let scale = fit_symmetric_scale(
            &values[16 * ib..16 * (ib + 1)],
            32,
            &mut quants[16 * ib..16 * (ib + 1)],
        );
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }

    if max_abs_scale < MAX_EPS {
        out.extend(std::iter::repeat_n(0u8, row_bytes(GGML_TYPE_Q6_K, QK_K)));
        return;
    }

    let iscale = -128.0 / max_scale;
    let d = f16::from_f32(1.0 / iscale);
    let mut packed_scales = [0i8; QK_K / 16];
    for ib in 0..QK_K / 16 {
        packed_scales[ib] = nearest(iscale * scales[ib]).min(127) as i8;
    }
    // Same reasoning as Q4_K: requantize against the scale a reader will
    // reconstruct, not the one that was fitted.
    for ib in 0..QK_K / 16 {
        let scale = d.to_f32() * packed_scales[ib] as f32;
        if scale == 0.0 {
            continue;
        }
        for i in 0..16 {
            quants[16 * ib + i] = nearest(values[16 * ib + i] / scale).clamp(-32, 31) as i8;
        }
    }

    let mut low = Vec::with_capacity(QK_K / 2);
    let mut high = Vec::with_capacity(QK_K / 4);
    for chunk in 0..QK_K / 128 {
        let base = chunk * 128;
        let mut low_part = [0u8; 64];
        let mut high_part = [0u8; 32];
        for i in 0..32 {
            let q = |at: usize| (quants[base + at] + 32) as u8;
            let (q1, q2, q3, q4) = (q(i), q(i + 32), q(i + 64), q(i + 96));
            low_part[i] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            low_part[i + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            high_part[i] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
        low.extend_from_slice(&low_part);
        high.extend_from_slice(&high_part);
    }

    out.extend_from_slice(&low);
    out.extend_from_slice(&high);
    for &s in &packed_scales {
        out.push(s as u8);
    }
    out.extend_from_slice(&d.to_le_bytes());
}

/// Fits `x ~ scale * q` with `q` in `-nmax..nmax-1` and no offset, trying
/// 19 scales around the one the range implies and keeping the one with the
/// best magnitude-weighted fit.
fn fit_symmetric_scale(x: &[f32], nmax: i32, quants: &mut [i8]) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < MAX_EPS {
        quants.fill(0);
        return 0.0;
    }

    let mut iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..x.len() {
        let l = nearest(iscale * x[i]).clamp(-nmax, nmax - 1);
        quants[i] = l as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l * l) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    for step in -9..=9 {
        if step == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * step as f32) / max;
        let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
        for &value in x {
            let l = nearest(iscale * value).clamp(-nmax, nmax - 1);
            let w = value * value;
            sumlx += w * value * l as f32;
            suml2 += w * (l * l) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (q, &value) in quants.iter_mut().zip(x.iter()) {
                *q = nearest(iscale * value).clamp(-nmax, nmax - 1) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// 256 values as eight 32-wide sub-blocks, each with its own 6-bit scale
/// and 6-bit minimum against a shared `f16` pair, and 5-bit quants: four
/// bits packed in a nibble and the fifth in a separate bit mask.
///
/// The same shape as `Q4_K` with one more bit per weight, which buys about
/// half the error for a fifth more size.
fn quantize_q5_k(block: &[f32], out: &mut Vec<u8>) {
    let values = super_block(block);
    let mut scales = [0.0f32; QK_K / 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut quants = [0u8; QK_K];
    let mut weights = [0.0f32; 32];

    for j in 0..QK_K / 32 {
        let sub = &values[32 * j..32 * (j + 1)];
        let sum_sq: f32 = sub.iter().map(|v| v * v).sum();
        let average = (sum_sq / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(sub.iter()) {
            *w = average + v.abs();
        }
        let (scale, min) = fit_scale_and_min(
            sub,
            &weights,
            31,
            NARROW_SQUARED,
            &mut quants[32 * j..32 * (j + 1)],
        );
        scales[j] = scale;
        mins[j] = min;
    }

    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    let mut packed = [0u8; 12];
    for j in 0..QK_K / 32 {
        let ls = (nearest(inv_scale * scales[j]).clamp(0, 63)) as u8;
        let lm = (nearest(inv_min * mins[j]).clamp(0, 63)) as u8;
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }

    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    for j in 0..QK_K / 32 {
        let (sc, m) = unpack_scale_min(j, &packed);
        let scale = d.to_f32() * sc as f32;
        if scale == 0.0 {
            continue;
        }
        let offset = dmin.to_f32() * m as f32;
        for i in 0..32 {
            quants[32 * j + i] = nearest((values[32 * j + i] + offset) / scale).clamp(0, 31) as u8;
        }
    }

    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&packed);

    let mut high = [0u8; QK_K / 8];
    let mut low = Vec::with_capacity(QK_K / 2);
    let (mut bit1, mut bit2) = (1u8, 2u8);
    for chunk in 0..QK_K / 64 {
        for i in 0..32 {
            let mut l1 = quants[chunk * 64 + i];
            let mut l2 = quants[chunk * 64 + i + 32];
            if l1 > 15 {
                l1 -= 16;
                high[i] |= bit1;
            }
            if l2 > 15 {
                l2 -= 16;
                high[i] |= bit2;
            }
            low.push(l1 | (l2 << 4));
        }
        bit1 <<= 2;
        bit2 <<= 2;
    }
    out.extend_from_slice(&high);
    out.extend_from_slice(&low);
}

/// The index of the codebook level nearest `value`.
///
/// Sixteen entries, ascending, so this is a short linear scan — and a scan
/// is the right shape here: it runs inside the innermost quantization loop
/// where a branchy binary search costs more than the seven extra compares.
#[inline]
fn nearest_level(value: f32) -> usize {
    let mut best = 0;
    let mut best_distance = f32::INFINITY;
    for (index, &level) in IQ4_VALUES.iter().enumerate() {
        let distance = (value - level as f32).abs();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

/// Fits one 32-wide sub-block against the non-linear codebook, returning
/// the scale that minimizes the magnitude-weighted error.
///
/// Unlike a linear quantization there is no closed form for the scale: the
/// levels are not evenly spaced, so which level each value lands on changes
/// as the scale moves. The fix is the established one — try a handful of
/// scales around the range, and for each solve the least-squares scale
/// given the levels it produced.
fn fit_codebook_scale(x: &[f32], levels: &mut [u8], tries: i32) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < MAX_EPS {
        levels.fill(0);
        return 0.0;
    }

    let first = IQ4_VALUES[0] as f32;
    let mut d = if tries > 0 { -max / first } else { max / first };
    let mut id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
    for (i, &value) in x.iter().enumerate() {
        let l = nearest_level(id * value);
        levels[i] = l as u8;
        let q = IQ4_VALUES[l] as f32;
        let w = value * value;
        sumqx += w * q * value;
        sumq2 += w * q * q;
    }
    d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
    let mut best = d * sumqx;

    for step in -tries..=tries {
        id = (step as f32 + first) / max;
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for &value in x {
            let q = IQ4_VALUES[nearest_level(id * value)] as f32;
            let w = value * value;
            sumqx += w * q * value;
            sumq2 += w * q * q;
        }
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }

    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (level, &value) in levels.iter_mut().zip(x.iter()) {
        *level = nearest_level(id * value) as u8;
    }
    d
}

/// 32 values, one `f16` scale, and 4-bit indices into the non-linear
/// codebook. The only 32-wide type here, and so the one a row that does not
/// divide 256 falls back to.
fn quantize_iq4_nl(block: &[f32], out: &mut Vec<u8>) {
    let mut values = [0.0f32; QK];
    let n = block.len().min(QK);
    values[..n].copy_from_slice(&block[..n]);

    let mut levels = [0u8; QK];
    let d = fit_codebook_scale(&values, &mut levels, 7);
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for i in 0..QK / 2 {
        out.push(levels[i] | (levels[i + QK / 2] << 4));
    }
}

/// 256 values as eight 32-wide sub-blocks against the same codebook, with a
/// signed 6-bit scale per sub-block against one `f16` — the K-quants' way
/// of spending scale bits, applied to the non-linear levels.
fn quantize_iq4_xs(block: &[f32], out: &mut Vec<u8>) {
    let values = super_block(block);
    let mut levels = [0u8; QK_K];
    let mut scales = [0.0f32; QK_K / 32];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for ib in 0..QK_K / 32 {
        let scale = fit_codebook_scale(
            &values[32 * ib..32 * (ib + 1)],
            &mut levels[32 * ib..32 * (ib + 1)],
            7,
        );
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }

    let d = -max_scale / 32.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut scales_high: u16 = 0;
    let mut scales_low = [0u8; QK_K / 64];
    for ib in 0..QK_K / 32 {
        let l = nearest(id * scales[ib]).clamp(-32, 31);
        // Requantize against the scale as it will be read back, exactly as
        // the K-quants do.
        let dl = d * l as f32;
        if dl != 0.0 {
            let inverse = 1.0 / dl;
            for i in 0..32 {
                levels[32 * ib + i] = nearest_level(inverse * values[32 * ib + i]) as u8;
            }
        }
        let stored = (l + 32) as u8;
        if ib % 2 == 0 {
            scales_low[ib / 2] = stored & 0xF;
        } else {
            scales_low[ib / 2] |= (stored & 0xF) << 4;
        }
        scales_high |= ((stored >> 4) as u16) << (2 * ib);
    }

    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    out.extend_from_slice(&scales_high.to_le_bytes());
    out.extend_from_slice(&scales_low);
    for ib in 0..QK_K / 32 {
        for i in 0..16 {
            out.push(levels[32 * ib + i] | (levels[32 * ib + 16 + i] << 4));
        }
    }
}

/// Fits `x ~ scale * q` with `q` in `-nmax..nmax-1`, by coordinate descent:
/// start from the range, then repeatedly move whichever single value most
/// improves the magnitude-weighted fit.
///
/// The wider types get a scale sweep instead ([`fit_symmetric_scale`]).
/// Three bits is where the sweep stops paying and the descent starts: eight
/// levels are few enough that moving one value between them changes the
/// whole block's best scale.
fn fit_signed_scale(x: &[f32], nmax: i32, quants: &mut [i8]) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < MAX_EPS {
        quants.fill(0);
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for (i, &value) in x.iter().enumerate() {
        let l = nearest(iscale * value).clamp(-nmax, nmax - 1);
        quants[i] = l as i8;
        let w = value * value;
        sumlx += w * value * l as f32;
        suml2 += w * (l * l) as f32;
    }

    for _ in 0..5 {
        let mut changed = 0;
        for (i, &value) in x.iter().enumerate() {
            let w = value * value;
            let without_lx = sumlx - w * value * quants[i] as f32;
            if without_lx <= 0.0 {
                continue;
            }
            let without_l2 = suml2 - w * (quants[i] as f32) * (quants[i] as f32);
            let proposed = nearest(value * without_l2 / without_lx).clamp(-nmax, nmax - 1);
            if proposed == quants[i] as i32 {
                continue;
            }
            let with_lx = without_lx + w * value * proposed as f32;
            let with_l2 = without_l2 + w * (proposed * proposed) as f32;
            // Keep the move only if it improves the fit as a whole, which
            // is what `sumlx^2 / suml2` measures.
            if with_l2 > 0.0 && with_lx * with_lx * suml2 > sumlx * sumlx * with_l2 {
                quants[i] = proposed as i8;
                sumlx = with_lx;
                suml2 = with_l2;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// Reads a tensor's stored bytes back as `f32`.
///
/// Only the float encodings are handled, and that is deliberate: this tool
/// quantizes *from* a full-precision file. Quantizing a file that is
/// already quantized would compound two roundings and quietly produce
/// something worse than quantizing the original once, so it is refused with
/// a message that says what to do instead.
pub fn decode(ggml_type: u32, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(bytes
            .chunks_exact(4)
            .take(elements)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        GGML_TYPE_F16 => Ok(bytes
            .chunks_exact(2)
            .take(elements)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        GGML_TYPE_BF16 => Ok(bytes
            .chunks_exact(2)
            .take(elements)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
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
                .chunks_exact(QK_K / 16 + QK_K / 4 + 4)
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
                .chunks_exact(QK_K / 8 + QK_K / 4 + 12 + 2)
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
                .chunks_exact(2 + 2 + 12 + QK_K / 8 + QK_K / 2)
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
                .chunks_exact(2 + QK / 2)
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
                .chunks_exact(2 + 2 + QK_K / 64 + QK_K / 2)
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
                .chunks_exact(2 + 2 + 12 + QK_K / 2)
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
                .chunks_exact(QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)
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
            other => decode(other, bytes, elements).unwrap(),
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
    const MODEL: Model = Model { layers: 24, gqa: 4 };

    /// The mixture is the whole point of the M in Q4_K_M.
    #[test]
    fn q4_k_m_spends_its_extra_bits_where_the_rules_say() {
        let plan = |name: &str, dims: &[u64]| plan_tensor(Ftype::Q4KM, name, dims, MODEL);

        assert_eq!(
            plan("output.weight", &[2048, 32768]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("token_embd.weight", &[2048, 32768]).ggml_type,
            GGML_TYPE_Q4_K
        );
        // The outer blocks get six bits, the middle ones four.
        assert_eq!(
            plan("blk.0.attn_v.weight", &[2048, 1024]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("blk.23.ffn_down.weight", &[8192, 2048]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("blk.4.attn_v.weight", &[2048, 1024]).ggml_type,
            GGML_TYPE_Q4_K
        );
        assert_eq!(
            plan("blk.4.attn_q.weight", &[2048, 2048]).ggml_type,
            GGML_TYPE_Q4_K
        );
        // Norms are never quantized.
        assert_eq!(
            plan("blk.4.attn_norm.weight", &[2048]).ggml_type,
            GGML_TYPE_F32
        );
        for ftype in Ftype::ALL {
            assert_eq!(
                plan_tensor(ftype, "blk.4.attn_norm.weight", &[2048], MODEL).ggml_type,
                GGML_TYPE_F32,
                "{} quantized a norm",
                ftype.name()
            );
        }
    }

    /// A row that no block divides has to demote, and say that it did.
    #[test]
    fn an_awkward_row_length_falls_back_and_is_recorded() {
        // 896 is not a multiple of 256 but is one of 32, so four bits
        // survive as the non-linear type.
        let plan = plan_tensor(Ftype::Q4KM, "blk.1.attn_q.weight", &[896, 896], MODEL);
        assert_eq!(plan.ggml_type, GGML_TYPE_IQ4_NL);
        assert_eq!(plan.fallback_from, Some(GGML_TYPE_Q4_K));

        // Six bits has nowhere narrower to go that is still six bits, so
        // it goes to f16 rather than halving the precision of the tensor
        // it was chosen for.
        let plan = plan_tensor(Ftype::Q4KM, "output.weight", &[100, 32], MODEL);
        assert_eq!(plan.ggml_type, GGML_TYPE_F16);
        assert_eq!(plan.fallback_from, Some(GGML_TYPE_Q6_K));

        let plan = plan_tensor(Ftype::Q4KM, "blk.1.attn_q.weight", &[2048, 2048], MODEL);
        assert_eq!(plan.fallback_from, None);
    }

    /// The suffix is a promise about which tensors are carried above the
    /// base type. Each of these is the difference between two file types
    /// that would otherwise be the same file.
    #[test]
    fn the_suffixes_differ_where_they_claim_to() {
        let plan = |ftype, name: &str| plan_tensor(ftype, name, &[2048, 2048], MODEL).ggml_type;

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

        // The vocabulary projection is six bits in every quantized file,
        // and the plain body type everywhere it is not named.
        for ftype in Ftype::ALL {
            let output = plan(ftype, "output.weight");
            let body = plan(ftype, "blk.12.attn_q.weight");
            if ftype.is_quantized() {
                assert_eq!(output, GGML_TYPE_Q6_K, "{}", ftype.name());
                assert_eq!(body, ftype.base_type(), "{}", ftype.name());
            } else {
                assert_eq!(output, ftype.base_type(), "{}", ftype.name());
            }
        }
    }

    /// Grouped-query attention makes the value projection cheap to carry
    /// high, and the rules only spend there when the ratio earns it.
    #[test]
    fn grouped_query_attention_changes_where_the_bits_go() {
        let grouped = Model { layers: 24, gqa: 4 };
        let plain = Model { layers: 24, gqa: 1 };
        let at =
            |model| plan_tensor(Ftype::Q2K, "blk.12.attn_v.weight", &[2048, 512], model).ggml_type;
        assert_eq!(at(grouped), GGML_TYPE_Q4_K);
        assert_eq!(at(plain), GGML_TYPE_Q3_K);
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
        let refused = |name: &str| Ftype::parse(name).unwrap_err().to_string();

        let iq2 = refused("iq2_xs");
        assert!(iq2.contains("importance matrix"), "{iq2}");

        let q8 = refused("q8_0");
        assert!(q8.contains("Q6_K"), "{q8}");

        let nonsense = refused("q9_z");
        assert!(nonsense.contains("unknown"), "{nonsense}");
    }
}

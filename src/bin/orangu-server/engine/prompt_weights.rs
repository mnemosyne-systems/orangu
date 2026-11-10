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

//! The weights a prompt multiplies on the CPU, as copies laid out for the
//! cores' matrix instructions when this machine is faster with them —
//! `[orangu-server].prompt_weights`: per-row `int8` for the K-quant
//! projections, `bf16` for the weights the file stores unquantized.
//!
//! A K-quant row carries a scale per 32 (or 16) elements, and the `smmla`
//! GEMM (`vecdot::gemm_k_rows_mmla`) has to fold every one of them: per 32
//! elements a multiply-accumulate and a re-zero per accumulator beside four
//! `smmla`, and `Q4_K`'s min correction on top. On the CIX P1's A720s that
//! holds the kernel to ~0.9 `smmla` a cycle where the core issues two. A
//! copy with one scale per *row* (`vecdot::RowI8`, built for the picture
//! transformers) accumulates a whole 256-element super-block in `i32` on an
//! 8 × 8 tile and folds once.
//!
//! Weights the file stores as `F32`/`F16`/`BF16` (`gemma-4-E2B`'s per-layer
//! embedding matrices) run on `CpuBackend`'s `f32` tile, a `BF16` one
//! widened to `f32` on every call; a `bf16` copy (`vecdot::Bf16Weights`)
//! puts them on the 8 × 8 `bfmmla` tile at 2.4–3.7× — `int8` would be
//! faster still but would quantize weights the file keeps exact, where
//! `bf16` is within 0.15% of the `f32` product.
//!
//! Whether either is worth its memory is the machine's question, so it is
//! measured, not assumed, for each kind on its own. A kind's copies are
//! held only when every one of these holds:
//!
//! - the CPU has the kind's instruction (`i8mm` for `smmla`, `bf16` for
//!   `bfmmla`);
//! - prompts run on the CPU (`prefill_backend::prompts_on_cpu`) — decode
//!   keeps the file's weights, which are fewer bytes to stream a token;
//! - the copies fit: all kinds together at most half of the memory
//!   available once the model is loaded;
//! - the kind's widest matrix (an FFN projection, for the K-quants) at a
//!   prompt chunk's width is faster through the copy than through the
//!   file's weights by at least [`MIN_GAIN`], timed here, on this machine.
//!
//! `prompt_weights = copy` skips the last two checks, `file` never builds a
//! copy; `ORANGU_PROMPT_WEIGHTS` overrides either for an A/B on one binary.
//!
//! The copies are keyed by the file's own tensor bytes (their address in
//! the mapping), so no architecture's forward pass knows about them:
//! `CpuBackend::matmul_into` looks a weight up for any multi-token call.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::loader::{LoadedModel, QuantMatrix};
use crate::engine::vecdot::{Bf16Weights, RowI8};

/// How a prompt's weights are held — `[orangu-server].prompt_weights`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptWeights {
    /// Copies where this machine measures faster with them and has the
    /// memory (see the module).
    #[default]
    Auto,
    /// Copies wherever the CPU has the instruction, unmeasured.
    Copy,
    /// The file's weights only.
    File,
}

impl PromptWeights {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "copy" => Some(Self::Copy),
            "file" => Some(Self::File),
            _ => None,
        }
    }
}

/// A weight's copy for the prompt path.
pub enum PromptCopy {
    /// Per-row `int8`, for a K-quant projection (`smmla`).
    Int8(RowI8),
    /// `bf16`, for a weight the file stores unquantized (`bfmmla`).
    Bf16(Bf16Weights),
}

impl PromptCopy {
    /// `x · wᵀ` through the copy, `[n_tokens][out_dim]`.
    pub fn matmul(&self, x: &[f32], n_tokens: usize) -> Vec<f32> {
        match self {
            Self::Int8(w) => crate::engine::vecdot::matmul_rowi8(x, n_tokens, w),
            Self::Bf16(w) => crate::engine::vecdot::matmul_bf16(x, n_tokens, w),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::Int8(w) => w.bytes(),
            Self::Bf16(w) => w.bytes(),
        }
    }
}

/// The two kinds, and what each is decided and reported as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Int8,
    Bf16,
}

impl Kind {
    /// The `[adapt]` layer the kind's decision is reported under.
    fn layer(self) -> &'static str {
        match self {
            Self::Int8 => "prompt weights (K-quant)",
            Self::Bf16 => "prompt weights (float)",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Bf16 => "bf16",
        }
    }

    /// "an int8 copy", "a bf16 copy".
    fn a_copy(self) -> &'static str {
        match self {
            Self::Int8 => "an int8 copy",
            Self::Bf16 => "a bf16 copy",
        }
    }

    /// Whether the CPU has the kind's instruction, and which one it lacks.
    fn available(self) -> Result<(), &'static str> {
        match self {
            Self::Int8 if !crate::engine::vecdot::have_i8mm() => Err("i8mm"),
            Self::Bf16 if !crate::engine::vecdot::have_bf16mm() => Err("bf16"),
            _ => Ok(()),
        }
    }

    fn copy_bytes(self, w: &QuantMatrix) -> u64 {
        let n = (w.in_dim * w.out_dim) as u64;
        match self {
            Self::Int8 => n,
            Self::Bf16 => 2 * n,
        }
    }

    fn build(self, w: &QuantMatrix) -> PromptCopy {
        match self {
            Self::Int8 => PromptCopy::Int8(RowI8::quantize(w.out_dim, w.in_dim, |o| w.row(o))),
            Self::Bf16 => {
                PromptCopy::Bf16(Bf16Weights::from_rows(w.out_dim, w.in_dim, |o| w.row(o)))
            }
        }
    }
}

/// The fewest tokens a call must have to take a copy: narrower passes are
/// not what the copy is for, and a one-token decode reads fewer bytes from
/// the file's weights.
pub const MIN_TOKENS: usize = 16;

/// How much faster a copy must measure to be worth its memory.
const MIN_GAIN: f64 = 1.10;

/// The probe's width: a prompt chunk's (the CPU path takes ~350–400 tokens
/// a chunk at 2 k, and the kernels' rates are level from ~256 up).
const PROBE_TOKENS: usize = 256;

static COPIES: OnceLock<HashMap<usize, PromptCopy>> = OnceLock::new();

/// Tensor name to its key in [`COPIES`], for a caller holding its own
/// mapping of the file (the NPU's timing), whose addresses differ.
static NAMES: OnceLock<HashMap<String, usize>> = OnceLock::new();

/// The copy of `w`, when one was built.
pub fn copy_of(w: &QuantMatrix) -> Option<&'static PromptCopy> {
    COPIES.get()?.get(&(w.raw_bytes().as_ptr() as usize))
}

/// The copy of the tensor named `name`, when one was built.
pub fn copy_named(name: &str) -> Option<&'static PromptCopy> {
    COPIES.get()?.get(NAMES.get()?.get(name)?)
}

/// The weights a prompt multiplies, with the kind of copy each would get:
/// every two-dimensional `blk.*` tensor, and gemma's `per_layer_model_proj`
/// (projected once per chunk). K-quant rows of whole super-blocks take
/// `int8`, unquantized ones `bf16`; anything else keeps the file's.
/// Routed-expert stacks (three-dimensional) and the token tables are not
/// matmuls of a prompt's tokens and are left out.
fn prompt_matrices(loaded: &LoadedModel) -> Vec<(String, QuantMatrix, Kind)> {
    let names: Vec<String> = loaded
        .tensor_types()
        .filter(|(name, _)| {
            (name.starts_with("blk.") && name.ends_with(".weight") && !name.contains("_exps"))
                || *name == "per_layer_model_proj.weight"
        })
        .map(|(name, _)| name.to_string())
        .collect();
    names
        .into_iter()
        .filter_map(|name| loaded.matrix(&name).ok().map(|w| (name, w)))
        .filter(|(_, w)| w.out_dim >= 8 && w.in_dim >= 8)
        .filter_map(|(name, w)| {
            let ty = w.ggml_type();
            let kind = if w.in_dim.is_multiple_of(crate::engine::vecdot::SUPER_BLOCK)
                && crate::engine::vecdot::supports_k(ty, w.in_dim)
            {
                Kind::Int8
            } else if crate::engine::vecdot::supports_float(ty) {
                Kind::Bf16
            } else {
                return None;
            };
            Some((name, w, kind))
        })
        .collect()
}

/// The model's widest FFN projection a prompt multiplies (the widest
/// projection, if it names none) — what a prompt-shaped probe times.
pub fn probe_matrix(loaded: &LoadedModel) -> Option<QuantMatrix> {
    let all = prompt_matrices(loaded);
    let size = |m: &&(String, QuantMatrix, Kind)| m.1.in_dim * m.1.out_dim;
    all.iter()
        .filter(|(name, _, k)| *k == Kind::Int8 && name.contains(".ffn_"))
        .max_by_key(size)
        .or_else(|| all.iter().max_by_key(size))
        .map(|(_, w, _)| w.clone())
}

/// Seconds for `f`: best of three after a warm-up.
fn best_of(f: &mut dyn FnMut()) -> f64 {
    f();
    (0..3)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Decides, and builds the copies when the answer is yes. Called by `main`
/// once the model is built and where its prompts run is settled.
pub fn prepare(loaded: &LoadedModel, choice: PromptWeights) {
    let choice = crate::engine::backend::tuning_profile::env_var("ORANGU_PROMPT_WEIGHTS")
        .ok()
        .and_then(|v| PromptWeights::parse(&v))
        .unwrap_or(choice);
    let note_all = |because: &str| {
        for kind in [Kind::Int8, Kind::Bf16] {
            crate::engine::adapt::note(kind.layer(), "the file's", because);
        }
    };
    if choice == PromptWeights::File {
        note_all("prompt_weights = file");
        return;
    }
    if crate::engine::prefill_backend::prompts_on_cpu() != Some(true) {
        note_all("prompts run on the device, not the cores");
        return;
    }
    let all = prompt_matrices(loaded);
    let gb = |b: u64| b as f64 / 1e9;
    let available = orangu::hardware::detect_cpu().available_memory_bytes;
    // Half of what is free after load, shared by both kinds.
    let mut budget = available / 2;
    let mut chosen: Vec<(String, QuantMatrix, Kind, String)> = Vec::new();
    for kind in [Kind::Int8, Kind::Bf16] {
        let note =
            |chose: &str, because: String| crate::engine::adapt::note(kind.layer(), chose, because);
        let mats: Vec<&(String, QuantMatrix, Kind)> =
            all.iter().filter(|(_, _, k)| *k == kind).collect();
        if mats.is_empty() {
            continue;
        }
        if let Err(missing) = kind.available() {
            note(
                "the file's",
                format!(
                    "the CPU has no {missing}, which the {} copy's kernel needs",
                    kind.name()
                ),
            );
            continue;
        }
        let bytes: u64 = mats.iter().map(|(_, w, _)| kind.copy_bytes(w)).sum();
        let because = if choice == PromptWeights::Copy {
            "prompt_weights = copy".to_string()
        } else {
            if bytes > budget {
                note(
                    "the file's",
                    format!(
                        "the {} copies would be {:.2} GB, past half of the {:.1} GB available",
                        kind.name(),
                        gb(bytes),
                        gb(available)
                    ),
                );
                continue;
            }
            // The widest matrix — for the K-quants, the widest FFN
            // projection: a prompt's cost is its FFN.
            let size = |m: &&&(String, QuantMatrix, Kind)| m.1.in_dim * m.1.out_dim;
            let Some((_, probe, _)) = mats
                .iter()
                .filter(|(name, _, _)| kind == Kind::Bf16 || name.contains(".ffn_"))
                .max_by_key(size)
                .or_else(|| mats.iter().max_by_key(size))
            else {
                continue;
            };
            let x: Vec<f32> = (0..PROBE_TOKENS * probe.in_dim)
                .map(|i| ((i * 7919) % 257) as f32 / 257.0 - 0.5)
                .collect();
            let copy = kind.build(probe);
            let file = best_of(&mut || {
                let _ = CpuBackend.matmul(&x, PROBE_TOKENS, probe);
            });
            let fast = best_of(&mut || {
                let _ = copy.matmul(&x, PROBE_TOKENS);
            });
            let measured = format!(
                "a {PROBE_TOKENS}-token {}x{} GEMM takes {:.1} ms through {}, {:.1} ms on the file's weights",
                probe.in_dim,
                probe.out_dim,
                fast * 1e3,
                kind.a_copy(),
                file * 1e3
            );
            if file < fast * MIN_GAIN {
                note("the file's", measured);
                continue;
            }
            format!(
                "{measured}; {:.2} GB of {:.1} available",
                gb(bytes),
                gb(available)
            )
        };
        budget = budget.saturating_sub(bytes);
        chosen.extend(
            mats.into_iter()
                .map(|(name, w, k)| (name.clone(), w.clone(), *k, because.clone())),
        );
    }
    if !chosen.is_empty() {
        build(chosen);
    }
}

/// Which kinds of copy a tree's prompts go through — decided once by its
/// top-level node ([`decide`]) and handed to every worker with its layers,
/// so every node of a tree rounds as the top does: a tree whose nodes
/// decided apart answered otherwise than one server. A server alone keeps
/// [`prepare`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    pub int8: bool,
    pub bf16: bool,
}

impl Decision {
    /// One byte on the wire: bit 0 `int8`, bit 1 `bf16`.
    pub fn bits(self) -> u8 {
        u8::from(self.int8) | (u8::from(self.bf16) << 1)
    }

    pub fn from_bits(bits: u8) -> Self {
        Self {
            int8: bits & 1 != 0,
            bf16: bits & 2 != 0,
        }
    }

    fn takes(self, kind: Kind) -> bool {
        match kind {
            Kind::Int8 => self.int8,
            Kind::Bf16 => self.bf16,
        }
    }
}

/// [`prepare`]'s decision for a tree, measured the same way — the CPU's
/// instructions, `prompt_weights`, the widest matrix timed through a copy
/// against the file's weights — but building nothing: a node builds the
/// copies of its own layers once a plan gives it some
/// ([`build_for_layers`]). Whether the copies fit is asked there, of the
/// layers the node runs.
pub fn decide(loaded: &LoadedModel, choice: PromptWeights) -> Decision {
    let choice = crate::engine::backend::tuning_profile::env_var("ORANGU_PROMPT_WEIGHTS")
        .ok()
        .and_then(|v| PromptWeights::parse(&v))
        .unwrap_or(choice);
    let mut decision = Decision::default();
    if choice == PromptWeights::File
        || crate::engine::prefill_backend::prompts_on_cpu() != Some(true)
    {
        return decision;
    }
    let all = prompt_matrices(loaded);
    for kind in [Kind::Int8, Kind::Bf16] {
        let mats: Vec<&(String, QuantMatrix, Kind)> =
            all.iter().filter(|(_, _, k)| *k == kind).collect();
        if mats.is_empty() || kind.available().is_err() {
            continue;
        }
        let take = choice == PromptWeights::Copy || {
            let size = |m: &&&(String, QuantMatrix, Kind)| m.1.in_dim * m.1.out_dim;
            let probe = mats
                .iter()
                .filter(|(name, _, _)| kind == Kind::Bf16 || name.contains(".ffn_"))
                .max_by_key(size)
                .or_else(|| mats.iter().max_by_key(size));
            probe.is_some_and(|(_, probe, _)| {
                let x: Vec<f32> = (0..PROBE_TOKENS * probe.in_dim)
                    .map(|i| ((i * 7919) % 257) as f32 / 257.0 - 0.5)
                    .collect();
                let copy = kind.build(probe);
                let file = best_of(&mut || {
                    let _ = CpuBackend.matmul(&x, PROBE_TOKENS, probe);
                });
                let fast = best_of(&mut || {
                    let _ = copy.matmul(&x, PROBE_TOKENS);
                });
                file >= fast * MIN_GAIN
            })
        };
        match kind {
            Kind::Int8 => decision.int8 = take,
            Kind::Bf16 => decision.bf16 = take,
        }
    }
    decision
}

/// A tree node's copies: of its own layers, by one decision. Replaced
/// whenever a plan moves the node's layers.
static RANGED: std::sync::RwLock<Option<std::sync::Arc<HashMap<usize, PromptCopy>>>> =
    std::sync::RwLock::new(None);

/// Whether this process keeps [`RANGED`] — a tree node. A server alone
/// never looks there.
static RANGED_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Builds the copies `decision` names for the matrices of `layers` (and the
/// per-layer embedding projection a Gemma 4 worker computes its inputs
/// with), replacing any a previous plan built. A kind whose copies would
/// take more than half the memory available is left on the file's weights.
/// Returns how many matrices were copied.
pub fn build_for_layers(
    loaded: &LoadedModel,
    decision: Decision,
    layers: std::ops::Range<usize>,
    n_layer: usize,
    decided_by: &str,
) -> usize {
    use rayon::prelude::*;
    RANGED_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let started = Instant::now();
    let available = orangu::hardware::detect_cpu().available_memory_bytes;
    let mut chosen: Vec<(QuantMatrix, Kind)> = Vec::new();
    for kind in [Kind::Int8, Kind::Bf16] {
        let mats: Vec<QuantMatrix> = prompt_matrices(loaded)
            .into_iter()
            .filter(|(name, _, k)| {
                *k == kind
                    && match crate::engine::loader::block_index(name).filter(|il| *il < n_layer) {
                        Some(il) => layers.contains(&il),
                        None => true,
                    }
            })
            .map(|(_, w, _)| w)
            .collect();
        if !decision.takes(kind) || mats.is_empty() {
            crate::engine::adapt::note(
                kind.layer(),
                "the file's",
                format!("{decided_by}, for layers {}..{}", layers.start, layers.end),
            );
            continue;
        }
        // Decided on the top-level node's cores, which may have what these
        // lack: this node then rounds as the file's weights do.
        if let Err(missing) = kind.available() {
            crate::engine::adapt::note(
                kind.layer(),
                "the file's",
                format!("{decided_by}, but no {missing} on these cores"),
            );
            continue;
        }
        let bytes: u64 = mats.iter().map(|w| kind.copy_bytes(w)).sum();
        if bytes > available / 2 {
            crate::engine::adapt::note(
                kind.layer(),
                "the file's",
                format!(
                    "the {} copies of layers {}..{} would be {:.2} GB, past half of the {:.1} GB \
                     available",
                    kind.name(),
                    layers.start,
                    layers.end,
                    bytes as f64 / 1e9,
                    available as f64 / 1e9
                ),
            );
            continue;
        }
        chosen.extend(mats.into_iter().map(|w| (w, kind)));
    }
    let copies: Vec<(usize, Kind, PromptCopy)> = chosen
        .par_iter()
        .map(|(w, kind)| (w.raw_bytes().as_ptr() as usize, *kind, kind.build(w)))
        .collect();
    for kind in [Kind::Int8, Kind::Bf16] {
        let mine: Vec<&PromptCopy> = copies
            .iter()
            .filter(|(_, k, _)| *k == kind)
            .map(|(_, _, c)| c)
            .collect();
        if !mine.is_empty() {
            crate::engine::adapt::note(
                kind.layer(),
                format!(
                    "{} copies of {} matrices of layers {}..{} ({:.2} GB; built in {:.1} s)",
                    kind.name(),
                    mine.len(),
                    layers.start,
                    layers.end,
                    mine.iter().map(|c| c.bytes()).sum::<usize>() as f64 / 1e9,
                    started.elapsed().as_secs_f64()
                ),
                decided_by.to_string(),
            );
        }
    }
    let n = copies.len();
    let map: HashMap<usize, PromptCopy> = copies.into_iter().map(|(at, _, c)| (at, c)).collect();
    *RANGED.write().unwrap_or_else(|p| p.into_inner()) = Some(std::sync::Arc::new(map));
    n
}

/// A multi-token product through a tree node's copy of `w`, when it has
/// one ([`build_for_layers`]). `None` at once in a server alone.
pub fn ranged_matmul(w: &QuantMatrix, x: &[f32], n_tokens: usize) -> Option<Vec<f32>> {
    if !RANGED_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let copies = RANGED.read().unwrap_or_else(|p| p.into_inner()).clone()?;
    copies
        .get(&(w.raw_bytes().as_ptr() as usize))
        .map(|copy| copy.matmul(x, n_tokens))
}

fn build(matrices: Vec<(String, QuantMatrix, Kind, String)>) {
    use rayon::prelude::*;
    let started = Instant::now();
    let names: HashMap<String, usize> = matrices
        .iter()
        .map(|(name, w, _, _)| (name.clone(), w.raw_bytes().as_ptr() as usize))
        .collect();
    let copies: Vec<(usize, Kind, PromptCopy)> = matrices
        .par_iter()
        .map(|(_, w, kind, _)| (w.raw_bytes().as_ptr() as usize, *kind, kind.build(w)))
        .collect();
    let seconds = started.elapsed().as_secs_f64();
    let mut report = Vec::new();
    for kind in [Kind::Int8, Kind::Bf16] {
        let mine: Vec<&(usize, Kind, PromptCopy)> =
            copies.iter().filter(|(_, k, _)| *k == kind).collect();
        let Some(because) = matrices
            .iter()
            .find(|(_, _, k, _)| *k == kind)
            .map(|(_, _, _, b)| b.clone())
        else {
            continue;
        };
        let bytes: usize = mine.iter().map(|(_, _, c)| c.bytes()).sum();
        report.push((kind, mine.len(), bytes, because));
    }
    let map: HashMap<usize, PromptCopy> = copies.into_iter().map(|(at, _, c)| (at, c)).collect();
    if COPIES.set(map).is_err() || NAMES.set(names).is_err() {
        // A second model in the process (a draft model): the first one's
        // copies stay, and this one runs on its file's weights.
        return;
    }
    for (kind, n, bytes, because) in report {
        crate::engine::adapt::note(
            kind.layer(),
            format!(
                "{} copies of {n} matrices ({:.2} GB; all copies built in {:.1} s)",
                kind.name(),
                bytes as f64 / 1e9,
                seconds
            ),
            because,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decision_fits_in_one_byte() {
        for bits in 0..4 {
            assert_eq!(Decision::from_bits(bits).bits(), bits);
        }
        assert_eq!(
            Decision::from_bits(2),
            Decision {
                int8: false,
                bf16: true
            }
        );
    }
}

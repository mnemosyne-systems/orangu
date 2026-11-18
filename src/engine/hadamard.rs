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

//! Hadamard-folded weights — the `prism.hadamard.*` metadata Prism ML's
//! `Ternary-Bonsai-2` releases carry, and the activation-side transform
//! that makes their weights mean what they say.
//!
//! A ternary weight matrix quantizes far better if its inputs are first
//! *rotated* so that no single input channel dominates: the release folds a
//! blockwise Walsh–Hadamard rotation (block 1024, normalized, with a fixed
//! ±1 sign per input channel) **into the stored weights**, so that the
//! file's `W'` is `W · Hᵀ` for some orthogonal `H`. `W'·(H·x)` is then
//! `W·x`, and a runtime that multiplies by `W'` as if it were `W` computes
//! `W·Hᵀ·x` instead — every projection of every layer wrong, on every
//! token, with no error anywhere. That is what a stock `llama.cpp` does
//! with these files, and what this engine did before this module existed:
//! the `F16` release loaded as an ordinary `qwen35` and produced nonsense.
//!
//! The file declares the fold rather than hiding it, precisely so a runtime
//! can refuse what it does not implement. Everything here is read from
//! `PrismML-Eng/llama.cpp`'s `llama-model.cpp` (the validation) and
//! `llama-graph.cpp` (where the transform is applied), not from the model
//! card:
//!
//! - **Forward** (`prism.hadamard.weight_names`, 401 tensors on the 27B):
//!   immediately before the matmul, the activation is multiplied
//!   elementwise by the sign vector for its width, then each 1024-block is
//!   multiplied by the normalized Sylvester–Hadamard matrix
//!   (`H[r][c] = (-1)^popcount(r & c) / sqrt(1024)`). `build_lora_mm`
//!   memoizes the transformed activation per input, so three projections
//!   of one normed vector share one rotation — [`Rotation::apply`] is
//!   written to be called once per input for the same reason.
//! - **Inverse** (`prism.hadamard.inverse_weight_names`, the embedding
//!   table): a looked-up row is in the rotated basis and comes back to the
//!   model's through `s · (H · z)` — the same two steps in the other order,
//!   which is the inverse because `H` is symmetric and self-inverse and
//!   `s² = 1`.
//! - **`gdn_v_grouped`**: the gated-DeltaNet output feeding `ssm_out`
//!   arrives with its value heads in *tiled* order (`v = k + n_k · r`, the
//!   `ggml_repeat` layout `engine::arch::qwen_hybrid` also uses) but the
//!   fold was computed in *grouped* order (`v = r + rep · k`), so that one
//!   input is permuted between the two before signs and rotation.
//!
//! The transform costs `n log n` adds per input vector — under 3% of a
//! decode step on the 27B, against matmuls that read gigabytes — so it runs
//! on the host in `f32` for every backend, in the architecture module, in
//! front of the matmul whose weight is folded. `engine::arch::qwen_hybrid`
//! is the one trunk that applies it today; [`HadamardFold::verify_covered`]
//! is how it proves at load time that every folded weight in the file went
//! through a site that transforms its input, so an architecture (or a
//! tensor kind) this module has not been wired into refuses the file
//! instead of running it wrong.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use rayon::prelude::*;

use crate::gguf::GgufValue;

const KEY_VERSION: &str = "prism.hadamard.version";
const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
const KEY_TRANSFORM: &str = "prism.hadamard.transform";
const KEY_AXIS: &str = "prism.hadamard.axis";
const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
const KEY_INVERSE_NAMES: &str = "prism.hadamard.inverse_weight_names";
const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";

/// The one transform and axis the format defines today; anything else is
/// a newer file than this build.
const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const AXIS: &str = "input-last-dimension";

/// Which weights of a file are Hadamard-folded, and with what — read once
/// at load from the file's `prism.hadamard.*` keys.
pub struct HadamardFold {
    block_size: usize,
    /// One ±1 vector per input width the file folds at, keyed by width.
    /// Empty in `identity` sign mode.
    signs: HashMap<usize, Arc<[f32]>>,
    /// Weights whose *input* is rotated before the matmul.
    weights: HashSet<String>,
    /// Row-lookup tables whose *output* is un-rotated after the lookup.
    inverses: HashSet<String>,
    gdn_v_grouped: bool,
}

/// The transform one folded weight's activation goes through, ready to
/// apply — cheap to clone (the sign vector is shared).
#[derive(Clone, Debug)]
pub struct Rotation {
    block_size: usize,
    signs: Option<Arc<[f32]>>,
    /// The tiled→grouped head permutation for a `gdn_v_grouped` `ssm_out`
    /// input: `(head_dim, n_k, rep)`.
    perm: Option<(usize, usize, usize)>,
}

impl HadamardFold {
    /// Reads the fold a file declares, or `None` for a file that declares
    /// none. An error is a fold this build cannot run — a version, transform
    /// or sign mode it does not know, or a declaration that contradicts
    /// itself — and a load must stop on it, because the alternative is
    /// running every matmul in the wrong basis.
    pub fn from_metadata(metadata: &[(String, GgufValue)]) -> Result<Option<Self>> {
        let Some(spec) = Spec::parse(metadata)? else {
            return Ok(None);
        };
        let mut signs = HashMap::new();
        if spec.explicit_signs {
            let widths = spec.sign_widths;
            let values: Vec<i64> = array_i64(metadata, KEY_SIGN_VALUES)
                .ok_or_else(|| anyhow!("{KEY_SIGN_VALUES} is missing"))?;
            let mut off = 0usize;
            for width in widths {
                ensure!(
                    off + width <= values.len(),
                    "{KEY_SIGN_VALUES} is shorter than {KEY_SIGN_WIDTHS} adds up to"
                );
                let vec: Vec<f32> = values[off..off + width]
                    .iter()
                    .map(|&v| match v {
                        1 => Ok(1.0f32),
                        -1 => Ok(-1.0f32),
                        other => Err(anyhow!("{KEY_SIGN_VALUES} holds {other}; must be +1/-1")),
                    })
                    .collect::<Result<_>>()?;
                ensure!(
                    signs.insert(width, Arc::from(vec)).is_none(),
                    "{KEY_SIGN_WIDTHS} lists width {width} twice"
                );
                off += width;
            }
            ensure!(
                off == values.len(),
                "{KEY_SIGN_VALUES} length {} does not match {KEY_SIGN_WIDTHS} ({off})",
                values.len()
            );
        }
        Ok(Some(Self {
            block_size: spec.block_size,
            signs,
            weights: spec.weight_names.into_iter().collect(),
            inverses: spec.inverse_names.into_iter().collect(),
            gdn_v_grouped: spec.gdn_v_grouped,
        }))
    }

    /// Why `list` should say `No` for a file — the reason string for its
    /// `SUPPORTED` cell — or `None` when the file declares no fold or one
    /// this build runs.
    ///
    /// Reads only what a summary header keeps: the scalar keys, the weight
    /// names and the sign widths. The sign *values* (28,672 on the 27B) are
    /// past `open_summary`'s array limit and are not needed to answer the
    /// question; [`Self::from_metadata`] reads them at load.
    pub fn header_support(metadata: &[(String, GgufValue)]) -> Option<String> {
        match Spec::parse(metadata) {
            Ok(_) => None,
            Err(err) => Some(format!("prism.hadamard: {err}")),
        }
    }

    #[cfg(test)]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Whether `name`'s input is rotated before the matmul.
    #[cfg(test)]
    pub fn is_folded(&self, name: &str) -> bool {
        self.weights.contains(name)
    }

    /// The transform for the activation feeding `name` (a matrix whose
    /// input width is `in_dim`), or `None` if `name` is not folded.
    ///
    /// `gdn_heads` is `Some((n_v, n_k))` for a gated-DeltaNet output
    /// projection — the value-head and key-head counts that define the
    /// tiled→grouped permutation `gdn_v_grouped` asks for. It is ignored for
    /// every other weight, and for `ssm_out` when the file does not set
    /// `gdn_v_grouped`.
    pub fn input_rotation(
        &self,
        name: &str,
        in_dim: usize,
        gdn_heads: Option<(usize, usize)>,
    ) -> Result<Option<Rotation>> {
        if !self.weights.contains(name) {
            return Ok(None);
        }
        let mut rotation = self.rotation_for(name, in_dim)?;
        if self.gdn_v_grouped && name.contains(".ssm_out.") {
            let (n_v, n_k) = gdn_heads.ok_or_else(|| {
                anyhow!("{name}: gdn_v_grouped fold on a weight that is not a delta-net output")
            })?;
            ensure!(
                n_k > 0 && n_v > 0 && n_v.is_multiple_of(n_k) && in_dim.is_multiple_of(n_v),
                "{name}: bad delta-net head geometry for a gdn_v_grouped fold \
                 (n_v {n_v}, n_k {n_k}, width {in_dim})"
            );
            rotation.perm = Some((in_dim / n_v, n_k, n_v / n_k));
        }
        Ok(Some(rotation))
    }

    /// The transform for a row looked up from `name` (a table whose rows
    /// are `dim` wide), or `None` if `name` is not an inverse table.
    pub fn inverse_rotation(&self, name: &str, dim: usize) -> Result<Option<Rotation>> {
        if !self.inverses.contains(name) {
            return Ok(None);
        }
        self.rotation_for(name, dim).map(Some)
    }

    fn rotation_for(&self, name: &str, width: usize) -> Result<Rotation> {
        ensure!(
            width > 0 && width.is_multiple_of(self.block_size),
            "{name}: fold block size {} does not divide input width {width}",
            self.block_size
        );
        let signs = if self.signs.is_empty() {
            None
        } else {
            Some(
                self.signs
                    .get(&width)
                    .cloned()
                    .ok_or_else(|| anyhow!("{name}: no fold sign vector for width {width}"))?,
            )
        };
        Ok(Rotation {
            block_size: self.block_size,
            signs,
            perm: None,
        })
    }

    /// Every folded name (forward and inverse) — what a loader must have
    /// claimed by the time it is done.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights
            .iter()
            .chain(self.inverses.iter())
            .map(String::as_str)
    }

    /// Fails if the file folds a weight that `claimed` — the names an
    /// architecture's loader routed through [`Self::input_rotation`] or
    /// [`Self::inverse_rotation`] — does not include. That weight would be
    /// multiplied in the wrong basis; refusing here is what turns a silent
    /// wrong answer into an error naming the tensor.
    pub fn verify_covered(&self, claimed: &HashSet<String>) -> Result<()> {
        let mut missing: Vec<&str> = self
            .names()
            .filter(|name| !claimed.contains(*name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        let shown: Vec<&str> = missing.iter().copied().take(4).collect();
        bail!(
            "{} Hadamard-folded weight(s) are not on a matmul path this build transforms \
             (first: {}); this architecture cannot serve this file",
            missing.len(),
            shown.join(", ")
        )
    }
}

impl Rotation {
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The ±1 vector, `None` in identity sign mode. The `Arc` is shared by
    /// every rotation of the same width, so its address identifies it —
    /// what a device-side cache keys on.
    pub fn signs(&self) -> Option<&Arc<[f32]>> {
        self.signs.as_ref()
    }

    /// Whether this rotation permutes its input first (a `gdn_v_grouped`
    /// `ssm_out`), which the device kernel does not do.
    pub fn has_permutation(&self) -> bool {
        self.perm.is_some()
    }

    /// This rotation's signs and blocks with the head permutation dropped —
    /// for a caller that has already stored its input in grouped order (the
    /// device delta-net kernel) and wants only the rest.
    pub fn without_permutation(&self) -> Rotation {
        Rotation {
            block_size: self.block_size,
            signs: self.signs.clone(),
            perm: None,
        }
    }

    /// Rotates `x` — `n_tokens` rows of `width` — in place into the basis
    /// the folded weight was stored in: head permutation (if any), signs,
    /// then the blockwise normalized Hadamard. Rows are independent, so a
    /// multi-token prefill fans out across the pool.
    pub fn apply(&self, x: &mut [f32], width: usize) {
        debug_assert_eq!(x.len() % width, 0);
        let block = self.block_size;
        let each = |row: &mut [f32], scratch: &mut Vec<f32>| {
            if let Some((hd, nk, rep)) = self.perm {
                permute_tiled_to_grouped(row, hd, nk, rep, scratch);
            }
            if let Some(signs) = &self.signs {
                for (v, s) in row.iter_mut().zip(signs.iter()) {
                    *v *= s;
                }
            }
            for chunk in row.chunks_exact_mut(block) {
                fwht_normalized(chunk);
            }
        };
        if x.len() == width {
            // One row — a decode step — stays on the calling thread. Fanning
            // its 5–17 blocks across the pool was measured (27B `PTQ1_0`, 8
            // threads) and lost: the fork-join per layer cost more than the
            // ~300 µs of butterflies it spread.
            each(x, &mut Vec::new());
        } else {
            x.par_chunks_mut(width)
                .for_each_init(Vec::new, |scratch, row| each(row, scratch));
        }
    }

    /// The inverse, for a row read out of a folded lookup table: blockwise
    /// Hadamard, then signs. No permutation — a table has none.
    pub fn apply_inverse(&self, x: &mut [f32], width: usize) {
        debug_assert_eq!(x.len() % width, 0);
        debug_assert!(self.perm.is_none());
        let block = self.block_size;
        let each = |row: &mut [f32]| {
            for chunk in row.chunks_exact_mut(block) {
                fwht_normalized(chunk);
            }
            if let Some(signs) = &self.signs {
                for (v, s) in row.iter_mut().zip(signs.iter()) {
                    *v *= s;
                }
            }
        };
        if x.len() == width {
            each(x);
        } else {
            x.par_chunks_mut(width).for_each(each);
        }
    }

    /// Whether two sites can share one transformed copy of the same input —
    /// what `qwen_hybrid` checks before feeding three projections from one
    /// rotation.
    pub fn same_as(&self, other: &Rotation) -> bool {
        self.block_size == other.block_size
            && self.perm == other.perm
            && match (&self.signs, &other.signs) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

/// `x ← H·x / sqrt(n)` in place, for `H` the Sylvester (natural-order)
/// Hadamard matrix of `x.len()`, a power of two. The iterative butterfly:
/// `log₂ n` passes of `(a, b) ← (a + b, a − b)` at doubling strides is
/// exactly the matrix product, and the fork's own `H` — `(-1)^popcount(r
/// & c)` — is the natural ordering this produces.
pub fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for chunk in x.chunks_exact_mut(2 * h) {
            let (lo, hi) = chunk.split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (s, d) = (*a + *b, *a - *b);
                *a = s;
                *b = d;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Reorders a delta-net output row from tiled head order — element
/// `h + hd·(k + nk·r)` — to grouped order, `h + hd·(r + rep·k)`. Mirrors
/// the fork's `ggml_reshape_4d(x, hd, nk, rep, …)` + `ggml_permute(0, 2,
/// 1, 3)`, which is a plain index remap once written out.
fn permute_tiled_to_grouped(
    row: &mut [f32],
    hd: usize,
    nk: usize,
    rep: usize,
    scratch: &mut Vec<f32>,
) {
    debug_assert_eq!(row.len(), hd * nk * rep);
    scratch.clear();
    scratch.extend_from_slice(row);
    for k in 0..nk {
        for r in 0..rep {
            let src = hd * (k + nk * r);
            let dst = hd * (r + rep * k);
            row[dst..dst + hd].copy_from_slice(&scratch[src..src + hd]);
        }
    }
}

/// The declaration, validated the way the fork validates it — everything
/// but the sign values, which a summary header does not carry.
struct Spec {
    block_size: usize,
    explicit_signs: bool,
    sign_widths: Vec<usize>,
    weight_names: Vec<String>,
    inverse_names: Vec<String>,
    gdn_v_grouped: bool,
}

impl Spec {
    fn parse(metadata: &[(String, GgufValue)]) -> Result<Option<Self>> {
        let Some(version) = scalar_u64(metadata, KEY_VERSION) else {
            return Ok(None);
        };
        ensure!(version == 1, "unsupported version {version}");
        let block_size = scalar_u64(metadata, KEY_BLOCK_SIZE)
            .ok_or_else(|| anyhow!("{KEY_BLOCK_SIZE} is missing"))?
            as usize;
        ensure!(
            block_size > 0 && block_size.is_power_of_two(),
            "block size {block_size} is not a power of two"
        );
        let transform =
            string(metadata, KEY_TRANSFORM).ok_or_else(|| anyhow!("{KEY_TRANSFORM} is missing"))?;
        ensure!(
            transform == TRANSFORM,
            "unsupported transform '{transform}'"
        );
        let axis = string(metadata, KEY_AXIS).ok_or_else(|| anyhow!("{KEY_AXIS} is missing"))?;
        ensure!(axis == AXIS, "unsupported axis '{axis}'");
        let sign_mode =
            string(metadata, KEY_SIGN_MODE).ok_or_else(|| anyhow!("{KEY_SIGN_MODE} is missing"))?;
        let explicit_signs = match sign_mode.as_str() {
            "identity" => false,
            "explicit" => true,
            other => bail!("unsupported sign mode '{other}'"),
        };
        let weight_names = array_strings(metadata, KEY_WEIGHT_NAMES)
            .ok_or_else(|| anyhow!("{KEY_WEIGHT_NAMES} is missing"))?;
        ensure!(!weight_names.is_empty(), "{KEY_WEIGHT_NAMES} is empty");
        let mut seen = HashSet::new();
        for name in &weight_names {
            ensure!(
                seen.insert(name.as_str()),
                "{KEY_WEIGHT_NAMES} lists {name} twice"
            );
        }
        let inverse_names = array_strings(metadata, KEY_INVERSE_NAMES).unwrap_or_default();
        for name in &inverse_names {
            ensure!(
                seen.insert(name.as_str()),
                "{name} is listed as both a folded weight and an inverse table"
            );
        }
        let mut sign_widths = Vec::new();
        if explicit_signs {
            let widths = array_i64(metadata, KEY_SIGN_WIDTHS)
                .ok_or_else(|| anyhow!("{KEY_SIGN_WIDTHS} is missing"))?;
            ensure!(
                !widths.is_empty(),
                "sign mode is explicit but {KEY_SIGN_WIDTHS} is empty"
            );
            for width in widths {
                ensure!(
                    width > 0 && (width as usize).is_multiple_of(block_size),
                    "sign width {width} is not a positive multiple of the block size"
                );
                sign_widths.push(width as usize);
            }
        }
        let gdn_v_grouped = scalar_u64(metadata, KEY_GDN_V_GROUPED).unwrap_or(0) != 0;
        Ok(Some(Self {
            block_size,
            explicit_signs,
            sign_widths,
            weight_names,
            inverse_names,
            gdn_v_grouped,
        }))
    }
}

fn lookup<'a>(metadata: &'a [(String, GgufValue)], key: &str) -> Option<&'a GgufValue> {
    metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn scalar_u64(metadata: &[(String, GgufValue)], key: &str) -> Option<u64> {
    lookup(metadata, key).and_then(GgufValue::as_u64)
}

fn string(metadata: &[(String, GgufValue)], key: &str) -> Option<String> {
    match lookup(metadata, key) {
        Some(GgufValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn array_strings(metadata: &[(String, GgufValue)], key: &str) -> Option<Vec<String>> {
    match lookup(metadata, key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn array_i64(metadata: &[(String, GgufValue)], key: &str) -> Option<Vec<i64>> {
    match lookup(metadata, key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GgufValue::I8(v) => Some(*v as i64),
                GgufValue::I16(v) => Some(*v as i64),
                GgufValue::I32(v) => Some(*v as i64),
                GgufValue::I64(v) => Some(*v),
                other => other.as_u64().map(|v| v as i64),
            })
            .collect(),
        _ => None,
    }
}

/// Reads a fold from `loaded`'s metadata with the file named in the error.
pub fn from_loaded(loaded: &crate::engine::loader::LoadedModel) -> Result<Option<HadamardFold>> {
    HadamardFold::from_metadata(&loaded.metadata).context("reading prism.hadamard metadata")
}

/// What an architecture's loader carries while it reads its weights: the
/// fold the file declares (if any) and the ledger of every folded name it
/// has routed through a transform so far. [`FoldLedger::finish`] is the
/// proof that the two agree.
///
/// A file without a fold hands out `None` from every lookup and passes
/// `finish` — a loader written against this costs an unfolded model
/// nothing.
pub struct FoldLedger<'a> {
    fold: Option<&'a HadamardFold>,
    claimed: HashSet<String>,
}

impl<'a> FoldLedger<'a> {
    pub fn new(fold: Option<&'a HadamardFold>) -> Self {
        Self {
            fold,
            claimed: HashSet::new(),
        }
    }

    /// The transform for the input of weight `name` — see
    /// [`HadamardFold::input_rotation`] — recorded as covered.
    pub fn input(
        &mut self,
        name: &str,
        in_dim: usize,
        gdn_heads: Option<(usize, usize)>,
    ) -> Result<Option<Rotation>> {
        let Some(fold) = self.fold else {
            return Ok(None);
        };
        let rotation = fold.input_rotation(name, in_dim, gdn_heads)?;
        if rotation.is_some() {
            self.claimed.insert(name.to_string());
        }
        Ok(rotation)
    }

    /// One transform for several weights that read the *same* activation —
    /// a joint Q/K/V, or gate and up. Either every name is folded the same
    /// way, or none is; a file that folded only some of them would need the
    /// one input in two bases at once, and is refused rather than guessed.
    pub fn shared_input(&mut self, names: &[&str], in_dim: usize) -> Result<Option<Rotation>> {
        let mut shared: Option<Rotation> = None;
        for (i, name) in names.iter().enumerate() {
            let rotation = self.input(name, in_dim, None)?;
            if i == 0 {
                shared = rotation;
                continue;
            }
            let agree = match (&shared, &rotation) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same_as(b),
                _ => false,
            };
            ensure!(
                agree,
                "{} and {name} read the same activation but are not folded alike",
                names[0]
            );
        }
        Ok(shared)
    }

    /// The transform for rows looked up from `name` — see
    /// [`HadamardFold::inverse_rotation`] — recorded as covered.
    pub fn inverse(&mut self, name: &str, dim: usize) -> Result<Option<Rotation>> {
        let Some(fold) = self.fold else {
            return Ok(None);
        };
        let rotation = fold.inverse_rotation(name, dim)?;
        if rotation.is_some() {
            self.claimed.insert(name.to_string());
        }
        Ok(rotation)
    }

    /// [`HadamardFold::verify_covered`] over everything claimed so far.
    pub fn finish(&self) -> Result<()> {
        match self.fold {
            Some(fold) => fold.verify_covered(&self.claimed),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(key: &str, value: GgufValue) -> (String, GgufValue) {
        (key.to_string(), value)
    }

    fn strings(items: &[&str]) -> GgufValue {
        GgufValue::Array(
            items
                .iter()
                .map(|s| GgufValue::String(s.to_string()))
                .collect(),
        )
    }

    fn ints(items: &[i64]) -> GgufValue {
        GgufValue::Array(items.iter().map(|&v| GgufValue::I32(v as i32)).collect())
    }

    /// A fold at block 4 over widths 8 and 12, so a whole matrix fits in a
    /// test — the 27B's is the same shape at block 1024.
    fn metadata(sign_mode: &str) -> Vec<(String, GgufValue)> {
        let mut m = vec![
            kv(KEY_VERSION, GgufValue::U32(1)),
            kv(KEY_BLOCK_SIZE, GgufValue::U32(4)),
            kv(KEY_TRANSFORM, GgufValue::String(TRANSFORM.into())),
            kv(KEY_AXIS, GgufValue::String(AXIS.into())),
            kv(KEY_SIGN_MODE, GgufValue::String(sign_mode.into())),
            kv(
                KEY_WEIGHT_NAMES,
                strings(&[
                    "output.weight",
                    "blk.0.ffn_up.weight",
                    "blk.0.ssm_out.weight",
                ]),
            ),
            kv(KEY_INVERSE_NAMES, strings(&["token_embd.weight"])),
            kv(KEY_GDN_V_GROUPED, GgufValue::Bool(true)),
        ];
        if sign_mode == "explicit" {
            m.push(kv(KEY_SIGN_WIDTHS, ints(&[8, 12])));
            m.push(kv(
                KEY_SIGN_VALUES,
                ints(&[
                    1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
                ]),
            ));
        }
        m
    }

    /// The fork's rotation matrix, built the way it builds it (a parity
    /// loop), applied as a dense product — the independent reference for
    /// the butterfly.
    fn dense_hadamard(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let scale = 1.0 / (n as f32).sqrt();
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| {
                        let sign = if (r & c).count_ones() % 2 == 1 {
                            -scale
                        } else {
                            scale
                        };
                        sign * x[c]
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn the_butterfly_is_the_forks_matrix() {
        for n in [1usize, 2, 4, 64, 1024] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 7919) % 13) as f32 - 6.0).collect();
            let want = dense_hadamard(&x);
            let mut got = x.clone();
            fwht_normalized(&mut got);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "n {n}: {g} vs {w}"
                );
            }
            // Self-inverse: applying it twice is the identity.
            fwht_normalized(&mut got);
            for (g, w) in got.iter().zip(&x) {
                assert!((g - w).abs() <= 1e-4, "n {n}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn a_file_without_the_keys_declares_no_fold() {
        assert!(HadamardFold::from_metadata(&[]).unwrap().is_none());
        assert!(HadamardFold::header_support(&[]).is_none());
    }

    #[test]
    fn the_declaration_is_read_and_validated() {
        let fold = HadamardFold::from_metadata(&metadata("explicit"))
            .unwrap()
            .unwrap();
        assert_eq!(fold.block_size(), 4);
        assert!(fold.is_folded("output.weight"));
        assert!(!fold.is_folded("token_embd.weight"));
        assert!(!fold.is_folded("blk.0.ssm_beta.weight"));
        assert!(
            fold.inverse_rotation("token_embd.weight", 8)
                .unwrap()
                .is_some()
        );
        assert!(fold.inverse_rotation("output.weight", 8).unwrap().is_none());
        assert!(
            fold.input_rotation("blk.0.ffn_gate.weight", 8, None)
                .unwrap()
                .is_none()
        );
        assert!(HadamardFold::header_support(&metadata("explicit")).is_none());

        // Identity sign mode needs no widths or values.
        let fold = HadamardFold::from_metadata(&metadata("identity"))
            .unwrap()
            .unwrap();
        let rot = fold
            .input_rotation("output.weight", 8, None)
            .unwrap()
            .unwrap();
        assert!(rot.signs.is_none());
    }

    #[test]
    fn a_fold_this_build_cannot_run_is_refused_by_both_readers() {
        let cases: Vec<(&str, Vec<(String, GgufValue)>)> = vec![
            ("version", {
                let mut m = metadata("explicit");
                m[0] = kv(KEY_VERSION, GgufValue::U32(2));
                m
            }),
            ("transform", {
                let mut m = metadata("explicit");
                m[2] = kv(KEY_TRANSFORM, GgufValue::String("dct".into()));
                m
            }),
            ("axis", {
                let mut m = metadata("explicit");
                m[3] = kv(KEY_AXIS, GgufValue::String("output".into()));
                m
            }),
            ("sign mode", {
                let mut m = metadata("explicit");
                m[4] = kv(KEY_SIGN_MODE, GgufValue::String("hashed".into()));
                m
            }),
            ("block size", {
                let mut m = metadata("explicit");
                m[1] = kv(KEY_BLOCK_SIZE, GgufValue::U32(6));
                m
            }),
            ("sign width", {
                let mut m = metadata("explicit");
                m[8] = kv(KEY_SIGN_WIDTHS, ints(&[6, 14]));
                m
            }),
        ];
        for (what, m) in cases {
            assert!(HadamardFold::from_metadata(&m).is_err(), "{what} at load");
            assert!(HadamardFold::header_support(&m).is_some(), "{what} in list");
        }
        // A bad sign *value* is only visible with the values present —
        // which the summary header does not carry, so only the load sees it.
        let mut m = metadata("explicit");
        m[9] = kv(KEY_SIGN_VALUES, ints(&[1; 20]).clone());
        assert!(HadamardFold::from_metadata(&m).is_ok());
        let mut bad = vec![1i64; 20];
        bad[3] = 2;
        m[9] = kv(KEY_SIGN_VALUES, ints(&bad));
        assert!(HadamardFold::from_metadata(&m).is_err());
        assert!(HadamardFold::header_support(&m).is_none());
    }

    /// The forward transform is `H · (s ⊙ x)` per block — checked against
    /// the dense matrix — and the inverse undoes it exactly, which is what
    /// makes a looked-up embedding row come back in the model's basis.
    #[test]
    fn forward_is_signs_then_hadamard_and_inverse_undoes_it() {
        let fold = HadamardFold::from_metadata(&metadata("explicit"))
            .unwrap()
            .unwrap();
        let rot = fold
            .input_rotation("output.weight", 8, None)
            .unwrap()
            .unwrap();
        let x: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5, 2.0];
        let signs = [1.0f32, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut want = Vec::new();
        for b in 0..2 {
            let signed: Vec<f32> = (0..4).map(|i| x[b * 4 + i] * signs[b * 4 + i]).collect();
            want.extend(dense_hadamard(&signed));
        }
        let mut got = x.clone();
        rot.apply(&mut got, 8);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-5, "{got:?} vs {want:?}");
        }
        let inv = fold
            .inverse_rotation("token_embd.weight", 8)
            .unwrap()
            .unwrap();
        inv.apply_inverse(&mut got, 8);
        for (g, w) in got.iter().zip(&x) {
            assert!((g - w).abs() <= 1e-5, "{got:?} vs {x:?}");
        }

        // Several rows at once go through the parallel arm and must match
        // the one-row arm row for row.
        let rows = 5;
        let mut many: Vec<f32> = (0..rows).flat_map(|_| x.iter().copied()).collect();
        rot.apply(&mut many, 8);
        for row in many.as_chunks::<8>().0 {
            for (g, w) in row.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn a_width_with_no_sign_vector_or_off_the_block_grid_is_refused() {
        let fold = HadamardFold::from_metadata(&metadata("explicit"))
            .unwrap()
            .unwrap();
        assert!(
            fold.input_rotation("output.weight", 16, None).is_err(),
            "no signs at 16"
        );
        assert!(
            fold.input_rotation("output.weight", 6, None).is_err(),
            "6 is not 4-aligned"
        );
        assert!(
            fold.input_rotation("output.weight", 12, None)
                .unwrap()
                .is_some()
        );
    }

    /// The `ssm_out` input arrives tiled (`v = k + n_k·r`) and the fold was
    /// taken grouped (`v = r + rep·k`); with `hd = 2, n_k = 2, rep = 3` the
    /// twelve elements move as the index formula says, before any signs.
    #[test]
    fn a_gdn_v_grouped_ssm_out_input_is_regrouped_before_the_rotation() {
        let fold = HadamardFold::from_metadata(&metadata("identity"))
            .unwrap()
            .unwrap();
        let rot = fold
            .input_rotation("blk.0.ssm_out.weight", 12, Some((6, 2)))
            .unwrap()
            .unwrap();
        assert_eq!(rot.perm, Some((2, 2, 3)));
        // Tiled: head v = k + 2r holds values 10v, 10v+1.
        let x: Vec<f32> = (0..6)
            .flat_map(|v| [10.0 * v as f32, 10.0 * v as f32 + 1.0])
            .collect();
        let mut scratch = Vec::new();
        let mut row = x.clone();
        permute_tiled_to_grouped(&mut row, 2, 2, 3, &mut scratch);
        // Grouped: position r + 3k holds tiled head k + 2r.
        let mut want = vec![0f32; 12];
        for k in 0..2 {
            for r in 0..3 {
                let v = k + 2 * r;
                want[2 * (r + 3 * k)] = 10.0 * v as f32;
                want[2 * (r + 3 * k) + 1] = 10.0 * v as f32 + 1.0;
            }
        }
        assert_eq!(row, want);
        // And the whole transform is that permutation followed by the
        // rotation.
        let mut got = x.clone();
        rot.apply(&mut got, 12);
        let mut expect = want.clone();
        for chunk in expect.as_chunks_mut::<4>().0 {
            fwht_normalized(chunk);
        }
        assert_eq!(got, expect);

        // Without the head geometry the fold cannot be applied, and a
        // non-`ssm_out` weight ignores it.
        assert!(
            fold.input_rotation("blk.0.ssm_out.weight", 12, None)
                .is_err()
        );
        let plain = fold
            .input_rotation("output.weight", 8, Some((6, 2)))
            .unwrap()
            .unwrap();
        assert!(plain.perm.is_none());
    }

    #[test]
    fn coverage_names_the_weight_that_was_never_transformed() {
        let fold = HadamardFold::from_metadata(&metadata("identity"))
            .unwrap()
            .unwrap();
        let mut claimed: HashSet<String> =
            ["output.weight", "blk.0.ffn_up.weight", "token_embd.weight"]
                .into_iter()
                .map(String::from)
                .collect();
        let err = fold.verify_covered(&claimed).unwrap_err().to_string();
        assert!(err.contains("blk.0.ssm_out.weight"), "{err}");
        claimed.insert("blk.0.ssm_out.weight".into());
        fold.verify_covered(&claimed).unwrap();
    }

    /// The ledger is how a loader proves coverage: it records what it
    /// routed, refuses a shared input whose readers disagree, and reports
    /// what the file folds that nothing claimed.
    #[test]
    fn the_ledger_records_claims_and_refuses_a_split_shared_input() {
        let fold = HadamardFold::from_metadata(&metadata("explicit"))
            .unwrap()
            .unwrap();
        let mut ledger = FoldLedger::new(Some(&fold));
        assert!(ledger.inverse("token_embd.weight", 8).unwrap().is_some());
        assert!(ledger.input("output.weight", 8, None).unwrap().is_some());
        // `ffn_up` is folded, `ffn_gate` is not: one input, two bases.
        let err = ledger
            .shared_input(&["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"], 8)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not folded alike"), "{err}");
        // Not everything is claimed yet.
        let err = ledger.finish().unwrap_err().to_string();
        assert!(err.contains("blk.0.ssm_out.weight"), "{err}");
        ledger
            .input("blk.0.ssm_out.weight", 12, Some((6, 2)))
            .unwrap();
        ledger.finish().unwrap();

        // No fold: every lookup is `None` and `finish` passes.
        let mut none = FoldLedger::new(None);
        assert!(none.input("output.weight", 8, None).unwrap().is_none());
        assert!(none.inverse("token_embd.weight", 8).unwrap().is_none());
        none.finish().unwrap();
    }

    #[test]
    fn two_sites_share_a_rotation_only_when_it_is_the_same_one() {
        let fold = HadamardFold::from_metadata(&metadata("explicit"))
            .unwrap()
            .unwrap();
        let a = fold
            .input_rotation("output.weight", 8, None)
            .unwrap()
            .unwrap();
        let b = fold
            .input_rotation("blk.0.ffn_up.weight", 8, None)
            .unwrap()
            .unwrap();
        let c = fold
            .input_rotation("blk.0.ffn_up.weight", 12, None)
            .unwrap()
            .unwrap();
        assert!(a.same_as(&b));
        assert!(!a.same_as(&c));
    }
}

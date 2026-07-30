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

//! Loads a GGUF file: memory-maps it, reads the `<arch>.*` hyperparameters
//! llama.cpp itself reads (key names confirmed directly against
//! `llama.cpp/src/llama-arch.cpp`'s `LLM_KV_*` table, not guessed), and
//! resolves each tensor's byte range for on-demand dequantization.

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use orangu::gguf::{GgufFile, GgufValue};
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

use super::quant;

/// Hyperparameters for a Llama-style (GQA + RoPE + RMSNorm + SwiGLU)
/// architecture — the family covering Llama/Llama3/Qwen2/Qwen3/Mistral-
/// shaped GGUFs. Gemma's own soft-capping/sliding-window variant reuses
/// this same struct and layers its further hyperparameters on top (see
/// `engine::arch::gemma`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Per-head width — `<arch>.attention.key_length` when the file sets it,
    /// otherwise the usual `n_embd / n_head`.
    ///
    /// These are **not** always the same number, and assuming they are is a
    /// silent shape error rather than a load failure. `Ministral-3-3B`
    /// (`mistral3`) has `n_embd = 3072` and `n_head = 32`, which would imply
    /// 96, but declares `key_length = 128` — and its `attn_q.weight` really
    /// is `[3072, 4096]` (`n_head * 128`), not `[3072, 3072]`. Upstream reads
    /// this into `hparams.n_embd_head_k` with the same fallback.
    pub head_dim: usize,
    pub n_ctx_train: usize,
    /// RoPE rotary dimension — defaults to `n_embd / n_head` when the file
    /// doesn't set `<arch>.rope.dimension_count`.
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub pooling_type: PoolingType,
}

/// `<arch>.pooling_type` — how `http::openai::pooled_embedding` reduces a
/// model's per-token hidden states to one embedding vector. Only `Mean`
/// (`gemma-embedding`, llama.cpp's own `LLAMA_POOLING_TYPE_MEAN = 1`) and
/// `Last` (`qwen3vl`-embedding models, `LLAMA_POOLING_TYPE_LAST = 3`) are
/// implemented; every other value (`NONE = 0`, `CLS = 2`, `RANK = 4`, or
/// the key being absent) falls back to `Mean` — the same unconditional
/// behavior this engine used before `<arch>.pooling_type` was read at all,
/// so this is additive, not a behavior change for any model already in
/// use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    Mean,
    Last,
}

/// Architecture families this engine's forward pass can run. Anything else
/// is rejected at load time with a clear error, rather than silently
/// running the wrong math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    /// Llama, Llama3, Qwen2, Qwen3, Mistral — the plain GQA+RoPE+RMSNorm+
    /// SwiGLU transformer, no soft-capping or sliding-window attention.
    LlamaStyle,
    /// Gemma/Gemma2/Gemma3/Gemma4 — QK-norm, per-layer-varying SWA/full
    /// attention, cross-layer KV sharing, per-layer embeddings, GEGLU FFN,
    /// final logit softcapping, and (`gemma-4-26B-A4B`) optional per-layer
    /// routed-expert MoE alongside a dense shared MLP. See
    /// `engine::arch::gemma`.
    Gemma,
    /// Qwen3.5/3.6-MoE — layers alternate between full attention (joint
    /// query+gate projection, partial rotary) and gated-DeltaNet linear
    /// attention (a chunked/recurrent SSM, here always run recurrently —
    /// see `engine::arch::qwen35moe`), each with a routed+shared-expert MoE
    /// FFN.
    Qwen35Moe,
    /// Qwen3.5 dense — the same hybrid full-attention/gated-DeltaNet layer
    /// shape as [`ArchFamily::Qwen35Moe`], but a plain dense SwiGLU FFN
    /// instead of MoE routing. See `engine::arch::qwen35`.
    Qwen35,
    /// Phi-3 / Phi-4-mini — GQA + RoPE + RMSNorm + SwiGLU like
    /// [`ArchFamily::LlamaStyle`], but with a fused QKV projection, a fused
    /// gate/up FFN projection, partial NEOX RoPE carrying LongRoPE
    /// frequency factors, and a RoPE magnitude factor. See
    /// `engine::arch::phi`.
    Phi3,
    /// Mistral 3 / Ministral-3 — `arch::llama`'s block shape plus YaRN RoPE
    /// scaling, NORM rope pairing, a head width read from
    /// `attention.key_length` rather than derived, and an attention
    /// temperature scale. See `engine::arch::mistral`.
    Mistral3,
}

/// GGUF `general.architecture` values that map to [`ArchFamily::LlamaStyle`]
/// — this engine treats them identically, since they share one forward
/// pass shape (only hyperparameters differ, all read from the file itself).
/// `qwen3vl` (e.g. `mradermacher/Qwen3-VL-Embedding-8B-GGUF`) is Qwen3-VL's
/// text backbone — same causal, GQA+RoPE+RMSNorm+SwiGLU shape as plain
/// `qwen3`, plus the per-head Q/K-RMSNorm `engine::arch::llama::LlamaLayer`
/// now loads generically for both. For *text-only* input specifically
/// (no image/video tokens), its M-RoPE position encoding is provably
/// identical to plain single-position RoPE: confirmed directly against
/// upstream `llama.cpp`'s `llm_graph_input_pos::set_input` (`src/llama-
/// graph.cpp`) — "in case we're using M-RoPE with text tokens, convert
/// the 1D positions to 4D: the 3 first dims are the same, and 4th dim is
/// all 0" — so every rotated dimension pair ends up using the exact same
/// position value regardless of which M-RoPE "section" it nominally
/// belongs to. Its DeepStack visual-feature injection (`n_deepstack_
/// layers`) is *also* a no-op for text-only input by the same reasoning:
/// `llm_graph_context::build_inp_embd` zero-pads a token (not raw
/// embedding) input up to the DeepStack-widened width, so the "inject
/// this layer's DeepStack slice" add is adding zero. Multimodal (image/
/// video) input itself is out of scope, per this project's existing
/// deferred-multimodal decision.
const LLAMA_STYLE_ARCHITECTURES: &[&str] = &["llama", "qwen2", "qwen3", "mistral", "qwen3vl"];
/// `mistral3` (e.g. `unsloth/Ministral-3-3B-Instruct-2512-GGUF`) — see
/// [`ArchFamily::Mistral3`] and `engine::arch::mistral`.
const MISTRAL_ARCHITECTURES: &[&str] = &["mistral3"];
/// `gemma-embedding` (e.g. `ggml-org/embeddinggemma-300M-GGUF`) is the
/// bidirectional-attention, embeddings-only sibling of the causal
/// gemma3/gemma4 decoders — same per-layer block shape (QK-norm, sandwich
/// norms, GEGLU FFN), read by the same `engine::arch::gemma` module, which
/// switches attention masking, the attention scale, and whether `forward`
/// (generation) is even allowed based on `general.architecture` itself
/// (confirmed directly against upstream `llama.cpp`'s `src/models/
/// gemma-embedding.cpp`: `hparams.causal_attn = false` is hardcoded per-arch
/// there, not read from GGUF metadata or a runtime flag).
const GEMMA_ARCHITECTURES: &[&str] = &["gemma", "gemma2", "gemma3", "gemma4", "gemma-embedding"];
const QWEN35MOE_ARCHITECTURES: &[&str] = &["qwen35moe"];
/// `qwen35` (e.g. `unsloth/Ornith-1.0-9B-GGUF`) — the dense sibling of
/// `qwen35moe`; see [`ArchFamily::Qwen35`].
const QWEN35_ARCHITECTURES: &[&str] = &["qwen35"];
/// `phi3` covers both Phi-3 and Phi-4-mini (e.g. `unsloth/Phi-4-mini-
/// instruct-GGUF`) — upstream converts both under the one
/// `general.architecture` string, and `llama_model_phi3` serves both from
/// one graph. `phi2` is *not* here: it's a different shape entirely
/// (LayerNorm rather than RMSNorm, GELU rather than SwiGLU, parallel
/// attention/FFN branches, biases throughout). `phimoe` (Phi-3.5-MoE) isn't
/// either — same attention block, but routed experts this module has no
/// path for.
const PHI3_ARCHITECTURES: &[&str] = &["phi3"];

pub fn resolve_arch_family(architecture: &str) -> Result<ArchFamily> {
    if LLAMA_STYLE_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::LlamaStyle);
    }
    if GEMMA_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Gemma);
    }
    if QWEN35MOE_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Qwen35Moe);
    }
    if QWEN35_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Qwen35);
    }
    if PHI3_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Phi3);
    }
    if MISTRAL_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Mistral3);
    }
    bail!(
        "architecture '{architecture}' is not yet supported by orangu-server \
         (supported: {})",
        LLAMA_STYLE_ARCHITECTURES
            .iter()
            .chain(GEMMA_ARCHITECTURES)
            .chain(QWEN35MOE_ARCHITECTURES)
            .chain(QWEN35_ARCHITECTURES)
            .chain(PHI3_ARCHITECTURES)
            .chain(MISTRAL_ARCHITECTURES)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// The architecture label and whether this build can actually load a model,
/// judged from its GGUF header alone (metadata + tensor directory — no
/// tensor data), for `list`'s `SUPPORTED` column and the interactive
/// pickers. This is stricter than [`resolve_arch_family`], which only knows
/// the architecture *string*: a model whose architecture is recognised can
/// still carry tensors this build rejects when it goes to build the model,
/// so a bare `resolve_arch_family` "yes" would promise a load that then
/// fails.
///
/// Both halves are checked. Beyond the architecture string, every tensor's
/// `ggml_type` must be one [`quant::dequantize`] can actually read: a
/// recognised architecture quantized to a type this build lacks (an
/// `unsloth` `IQ`-mix is the usual way to meet one) loads far enough to
/// report `Yes` from its header and then fails on the first tensor. Only
/// the tensor *directory* is consulted, never the data, so this stays as
/// cheap as reading the header.
///
/// Returns `(architecture, unsupported_quant)`. `architecture` is `None`
/// only when the file has no `general.architecture` at all;
/// `unsupported_quant` names the first tensor type this build can't decode,
/// and is `None` when every type is readable.
pub fn model_load_support(gguf: &GgufFile) -> (Option<String>, Option<String>) {
    let architecture = metadata_string(gguf, "general.architecture");
    let unsupported = gguf
        .tensors
        .iter()
        .find(|t| !quant::supports_type(t.ggml_type))
        .map(|t| orangu::gguf::ggml_type_name(t.ggml_type));
    (architecture, unsupported)
}

/// A tensor's bytes: normally the shard `Mmap` they live in, but an owned
/// buffer for a tensor the loader had to rewrite before anything could read
/// it (see [`LoadedModel::open`]'s repacked-`Q4_0` conversion).
///
/// `Mmap` and `Vec<u8>` both deref to `[u8]`, so one trait object serves
/// both and every reader stays identical. Behind an `Arc` because a tensor
/// view (`QuantMatrix`) is cloned freely and must keep its bytes alive; the
/// address is also what `QuantMatrix::cache_key` hands the GPU backends as
/// a stable identity, which holds for either variant as long as the model
/// does.
pub(crate) type TensorBytes = Arc<dyn std::ops::Deref<Target = [u8]> + Send + Sync>;

/// A tensor's resolved location and shape, ready for [`quant::dequantize`].
#[derive(Clone)]
struct TensorLocation {
    ggml_type: u32,
    dims: Vec<u64>,
    /// Absolute byte offset into [`TensorLocation::bytes`].
    start: usize,
    len: usize,
    /// Where this tensor's bytes live. A single-file model has one mmap
    /// shared by every tensor; a split one (`model-00001-of-00003.gguf` …)
    /// has a different mapping per shard, and each tensor's `start` is
    /// relative to its own. A rewritten tensor owns its buffer outright and
    /// starts at 0.
    bytes: TensorBytes,
}

pub struct LoadedModel {
    pub config: ModelConfig,
    /// The GGUF file's raw metadata key/value pairs — beyond the common
    /// subset [`ModelConfig`] captures, an architecture module (e.g.
    /// `engine::arch::gemma`) reads its own further hyperparameters
    /// (per-layer arrays, architecture-specific keys) directly from this.
    pub metadata: Vec<(String, GgufValue)>,
    tensors: HashMap<String, TensorLocation>,
}

/// A lazy view onto a 2D GGUF tensor (an `[in_dim, out_dim]` matmul weight,
/// or an embedding table read by row) — `mmap`-backed, dequantizing one row
/// at a time on demand rather than materializing the whole matrix as `f32`
/// at load time. A `Q4_K`-quantized model's resident footprint under the
/// old eager-dequant-everything approach was roughly 4x its file size (fine
/// for the small models this build originally targeted, but a hard blocker
/// for anything in the tens-of-billions-of-parameters range on ordinary
/// hardware); this cuts it to roughly 1x (the mmap itself, lazily paged in)
/// plus whatever rows are transiently live during a single matmul call.
#[derive(Clone)]
pub struct QuantMatrix {
    bytes: TensorBytes,
    ggml_type: u32,
    start: usize,
    row_bytes: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl QuantMatrix {
    /// Dequantizes row `index` (one output unit's `in_dim` input weights,
    /// or one embedding table entry) to `f32`. `index` must be `< out_dim`.
    pub fn row(&self, index: usize) -> Vec<f32> {
        let offset = self.start + index * self.row_bytes;
        let bytes = &self.bytes[offset..offset + self.row_bytes];
        quant::dequantize(self.ggml_type, bytes, self.in_dim)
            .expect("row byte range was validated when this QuantMatrix was constructed")
    }

    /// The `ggml_type` this matrix's rows are still quantized as — a GPU
    /// backend (`engine::backend::vulkan`) dispatches to a type-specific
    /// dequantizing shader rather than dequantizing on the CPU via `row`.
    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }

    /// Bytes per row (before dequantizing).
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The whole matrix's raw, still-quantized bytes (`row_bytes * out_dim`
    /// long, one row after another) — for a GPU backend that uploads them
    /// as-is and dequantizes on the shader, rather than row-by-row on the
    /// CPU like [`QuantMatrix::row`].
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.row_bytes * self.out_dim]
    }

    /// A row-range **view** — the same bytes, no copy, describing `count`
    /// output units starting at row `first`.
    ///
    /// This is what lets an architecture whose checkpoint concatenates several
    /// projections into one tensor (`phi3`'s `attn_qkv.weight`, and its
    /// `ffn_up.weight` holding gate and up together) hand the rest of the
    /// engine the separate matrices it expects. Rows are fixed-size and
    /// self-contained — `row_bytes` covers a whole number of quantization
    /// blocks, since blocks run along `in_dim` — so a row boundary is always a
    /// valid place to cut.
    ///
    /// Panics rather than truncates on an out-of-range range: every caller
    /// derives it from a validated tensor shape, so a bad range is a loader
    /// bug, and silently returning fewer rows would surface as wrong output
    /// rather than as an error.
    pub fn rows(&self, first: usize, count: usize) -> QuantMatrix {
        assert!(
            first + count <= self.out_dim,
            "row range {first}..{} exceeds out_dim {}",
            first + count,
            self.out_dim,
        );
        QuantMatrix {
            bytes: self.bytes.clone(),
            ggml_type: self.ggml_type,
            start: self.start + first * self.row_bytes,
            row_bytes: self.row_bytes,
            in_dim: self.in_dim,
            out_dim: count,
        }
    }

    /// A stable identity for this tensor's byte range, valid for as long as
    /// the underlying `mmap` is kept alive (the model's whole lifetime) —
    /// lets a GPU backend cache an uploaded copy of this matrix keyed by
    /// identity, so a weight already on the GPU isn't re-uploaded on every
    /// `matmul` call (every decode step reuses the same weight tensors).
    ///
    /// `(absolute start address, byte length)`, **not** `(mmap base, start)`.
    /// The two are equivalent for whole tensors and differ for the row-range
    /// views [`QuantMatrix::rows`] returns: a view of the leading rows shares
    /// its parent's `start`, so a key that omitted the length would hand the
    /// view every cache entry belonging to the whole tensor — the same buffer,
    /// the same bind groups, the wrong output width. Including the length makes
    /// the key a complete description of the range it names.
    pub fn cache_key(&self) -> (usize, usize) {
        (
            self.bytes.as_ptr() as usize + self.start,
            self.row_bytes * self.out_dim,
        )
    }
}

/// Builds a `QuantMatrix` directly over `row_bytes * out_dim` raw bytes,
/// bypassing `LoadedModel`/a real GGUF file entirely — for tests (e.g.
/// `engine::backend::vulkan`'s CPU/GPU cross-check) that need a matrix with
/// known, hand-built quantized content rather than one read from a
/// downloaded model. `bytes` is written to a temp file and `mmap`ped, since
/// `QuantMatrix` always holds an `Arc<Mmap>` — the file itself can be
/// (and, once mapped, safely is) dropped immediately after, per the usual
/// POSIX "unlinking doesn't invalidate an existing mapping" guarantee.
///
/// The `Arc<Mmap>` is *also* pushed into a process-lifetime registry below
/// rather than left to drop with its `QuantMatrix` at the end of a test —
/// see that registry's own doc comment for why: every `VulkanBackend`
/// cache keyed off `QuantMatrix::cache_key()` (a raw `(mmap.as_ptr(),
/// start)` pair) assumes that address is a stable identity, which silently
/// stops being true the moment an address gets freed and reused.
#[cfg(test)]
pub(crate) fn test_quant_matrix(
    bytes: &[u8],
    ggml_type: u32,
    in_dim: usize,
    out_dim: usize,
) -> QuantMatrix {
    use std::io::Write;

    /// Every `Arc<Mmap>` any test-built `QuantMatrix` has ever used,
    /// deliberately never cleared. `engine::backend::vulkan::tests` shares
    /// *one* `VulkanBackend` (and hence one set of `QuantMatrix::
    /// cache_key()`-addressed caches: `op_cache`, `weight_cache`,
    /// `fused_cache`, `fused_attn_layer_cache`, `fused_layer_cache`)
    /// across every test in the binary — a real production `LoadedModel`'s
    /// mmap lives for the whole server process, so those caches were never
    /// designed to detect an address becoming invalid and getting reused
    /// for something else entirely. Without this registry, a test's
    /// `QuantMatrix` (and this function's temp-file `Mmap`) drops at scope
    /// end, the OS is free to hand that exact virtual address to a *later*
    /// test's `Mmap::map` call (routine for same-sized mappings on Linux),
    /// and that later test would silently inherit an *unrelated* earlier
    /// test's stale cached GPU buffers instead of missing the cache and
    /// rebuilding correctly-sized, correctly-valued ones. Caught by, not
    /// just anticipated for, exactly that scenario: `cargo test --
    /// --test-threads=1` reliably collided two `fused_layer` tests that
    /// happen to share `n_embd`/`eps` before this fix existed, at fixed
    /// values shape-validated cache keys alone couldn't have ruled out
    /// (test shapes routinely repeat small round numbers like `n_embd =
    /// 24`) — keeping every mmap's address permanently allocated for the
    /// test binary's whole lifetime closes this at the actual root cause
    /// (address reuse) instead of chasing it key by key.
    static LEAKED_TEST_MMAPS: std::sync::Mutex<Vec<Arc<Mmap>>> = std::sync::Mutex::new(Vec::new());

    let mut file = tempfile::NamedTempFile::new().expect("failed to create temp file");
    file.write_all(bytes).expect("failed to write temp file");
    file.flush().expect("failed to flush temp file");
    let mmap = Arc::new(unsafe { Mmap::map(file.as_file()) }.expect("failed to mmap temp file"));
    LEAKED_TEST_MMAPS
        .lock()
        .expect("leaked test mmap registry poisoned")
        .push(mmap.clone());
    QuantMatrix {
        bytes: mmap,
        ggml_type,
        start: 0,
        row_bytes: bytes.len() / out_dim,
        in_dim,
        out_dim,
    }
}

/// Like [`QuantMatrix`], but for a 3D "stacked per-expert" GGUF tensor
/// (`engine::arch::qwen35moe`'s `ffn_*_exps.weight`) — `n_expert` separate
/// `[in_dim, out_dim]` matrices concatenated along a third dimension. A MoE
/// layer only ever evaluates a handful of experts per token (8 out of 256,
/// for the model this was verified against), so — even more than
/// [`QuantMatrix`] — materializing every expert's weights would be almost
/// entirely wasted work, not just wasted memory.
#[derive(Clone)]
pub struct ExpertQuantMatrix {
    bytes: TensorBytes,
    ggml_type: u32,
    start: usize,
    row_bytes: usize,
    expert_stride: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_expert: usize,
}

impl ExpertQuantMatrix {
    /// Dequantizes row `index` of expert `expert` (`in_dim` values).
    pub fn row(&self, expert: usize, index: usize) -> Vec<f32> {
        debug_assert!(
            expert < self.n_expert,
            "expert {expert} >= {}",
            self.n_expert
        );
        let offset = self.start + expert * self.expert_stride + index * self.row_bytes;
        let bytes = &self.bytes[offset..offset + self.row_bytes];
        quant::dequantize(self.ggml_type, bytes, self.in_dim)
            .expect("row byte range was validated when this ExpertQuantMatrix was constructed")
    }
}

/// Every file making up this model, in shard order — just `[path]` for an
/// ordinary single-file GGUF.
///
/// A model too large for one file is written as `<prefix>-00001-of-000NN.
/// gguf` … `<prefix>-000NN-of-000NN.gguf`, with the hyperparameters and most
/// tensors in shard 1 and the remaining tensors spread across the rest —
/// `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` keeps `output_norm.weight`
/// and the last 59 tensors in shard 2. Mapping only the named file therefore
/// doesn't fail cleanly; it loads a model that is *missing tensors*, and the
/// first architecture module to ask for one reports it as absent.
///
/// The shard count comes from `split.count` (upstream's own `LLM_KV_SPLIT_
/// COUNT`) rather than from globbing the directory, so a stray file that
/// merely looks like a shard can't join the set. The filename pattern is
/// upstream's `llama_split_path` format, `%s-%05d-of-%05d.gguf`.
fn shard_paths(path: &Path, gguf: &GgufFile) -> Result<Vec<std::path::PathBuf>> {
    let count = metadata_u64(gguf, "split.count").unwrap_or(0);
    if count <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("{} has no usable file name", path.display()))?;
    // `<prefix>-00001-of-000NN.gguf` — split off the last two fields.
    let stem = name
        .strip_suffix(".gguf")
        .ok_or_else(|| anyhow!("split model shard {name} is not a .gguf file"))?;
    let (prefix, rest) = stem
        .rsplit_once("-of-")
        .and_then(|(head, tail)| head.rsplit_once('-').map(|(p, no)| (p, (no, tail))))
        .ok_or_else(|| {
            anyhow!(
                "{name} declares split.count = {count} but isn't named \
                 <prefix>-00001-of-{count:05}.gguf"
            )
        })?;
    anyhow::ensure!(
        rest.0.len() == 5 && rest.1.len() == 5,
        "{name} declares split.count = {count} but its shard numbering isn't 5 digits"
    );
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let paths: Vec<std::path::PathBuf> = (1..=count)
        .map(|no| dir.join(format!("{prefix}-{no:05}-of-{count:05}.gguf")))
        .collect();
    for shard in &paths {
        anyhow::ensure!(
            shard.exists(),
            "split model is missing shard {} of {count} ({})",
            shard
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default(),
            shard.display()
        );
    }
    Ok(paths)
}

impl LoadedModel {
    /// Every tensor's `(name, ggml_type)`, across **every** shard — unlike
    /// walking a single [`GgufFile`]'s directory, which for a split model
    /// only sees shard 1. Used by `engine::backend::unsupported_tensor_types`
    /// to decide whether the selected backend can run this model at all.
    pub fn tensor_types(&self) -> impl Iterator<Item = (&str, u32)> {
        self.tensors
            .iter()
            .map(|(name, loc)| (name.as_str(), loc.ggml_type))
    }

    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let architecture = metadata_string(&gguf, "general.architecture")
            .ok_or_else(|| anyhow!("GGUF file is missing general.architecture"))?;
        resolve_arch_family(&architecture)?;

        let config = read_model_config(&gguf, &architecture)?;

        // Every shard's tensor directory, merged. A single-file model is
        // just the one-shard case of this.
        let mut tensors = HashMap::with_capacity(gguf.tensors.len());
        let shards = shard_paths(path, &gguf)?;
        let mut total_tensors = 0usize;
        for (index, shard_path) in shards.iter().enumerate() {
            // Shard 1's header is already parsed; the rest still need
            // reading. Only shard 1 carries the model's hyperparameters —
            // the others hold `split.*` and their own tensor directory.
            let shard_gguf = if index == 0 {
                None
            } else {
                Some(GgufFile::open(shard_path)?)
            };
            let shard_gguf = shard_gguf.as_ref().unwrap_or(&gguf);

            let file = File::open(shard_path)
                .with_context(|| format!("failed to open {}", shard_path.display()))?;
            // Safety: the file is opened read-only and not mutated by anything
            // else for the lifetime of this mapping — the standard caveat of
            // `Mmap::map` (another process truncating the file underneath us
            // would be undefined behavior, same risk llama.cpp itself accepts
            // when it mmaps a GGUF file).
            let mmap = Arc::new(
                unsafe { Mmap::map(&file) }
                    .with_context(|| format!("failed to mmap {}", shard_path.display()))?,
            );

            for tensor in &shard_gguf.tensors {
                let element_count: u64 = tensor.dims.iter().product();
                let len = quant::tensor_byte_size(tensor.ggml_type, element_count)
                    .with_context(|| format!("tensor '{}'", tensor.name))?
                    as usize;
                let start = shard_gguf.data_offset as usize + tensor.offset as usize;
                if start + len > mmap.len() {
                    bail!(
                        "tensor '{}' extends past the end of {}",
                        tensor.name,
                        shard_path.display()
                    );
                }
                total_tensors += 1;
                tensors.insert(
                    tensor.name.clone(),
                    TensorLocation {
                        ggml_type: tensor.ggml_type,
                        dims: tensor.dims.clone(),
                        start,
                        len,
                        bytes: mmap.clone(),
                    },
                );
            }
        }

        // Rewrite every row-interleaved repack (`Q4_0_4_4`/`_4_8`/`_8_8`
        // and the `IQ4_NL` trio) into the plain `Q4_0`/`IQ4_NL` it was built
        // from, before anything can read one. This is the only point where
        // those six `ggml_type`s exist at all: past here the model is
        // indistinguishable from one quantized to the base type directly,
        // so the CPU fused kernel (`engine::vecdot`) and every GPU backend's
        // existing shader serve it unchanged, with no new kernel anywhere.
        //
        // Done eagerly rather than per row because the packing interleaves
        // *rows* — a row's blocks are strided across its 4- or 8-row group,
        // so there is no row-shaped slice to be lazy about. The rewritten
        // tensor is exactly as large as the mapped bytes it replaces, and
        // only these files pay for it.
        for (name, loc) in tensors.iter_mut() {
            if !quant::is_repacked(loc.ggml_type) {
                continue;
            }
            anyhow::ensure!(
                loc.dims.len() == 2,
                "tensor '{name}' is {} but not a 2D matrix (dims: {:?}); the repacked layouts \
                 only ever applied to 2D weights",
                orangu::gguf::ggml_type_name(loc.ggml_type),
                loc.dims
            );
            let (base, plain) = quant::deinterleave_repack(
                loc.ggml_type,
                &loc.bytes[loc.start..loc.start + loc.len],
                loc.dims[0] as usize,
                loc.dims[1] as usize,
            )
            .with_context(|| format!("tensor '{name}'"))?;
            loc.ggml_type = base;
            loc.start = 0;
            loc.len = plain.len();
            loc.bytes = Arc::new(plain);
        }

        // `split.tensors.count` is the whole model's tensor count, so this
        // catches a shard that is present but truncated, or two shards that
        // collide on a tensor name — either of which would otherwise surface
        // much later as a confusing "model is missing tensor X".
        if let Some(expected) = metadata_u64(&gguf, "split.tensors.count")
            && total_tensors != expected as usize
        {
            bail!(
                "split model has {total_tensors} tensors across {} shard(s), but its header \
                 declares {expected}",
                shards.len()
            );
        }

        Ok(Self {
            config,
            metadata: gguf.metadata,
            tensors,
        })
    }

    /// A `<arch>.<suffix>` metadata value, widened to `u64` — for scalar
    /// hyperparameters. See [`LoadedModel::metadata_array_u64`] for arrays
    /// (e.g. Gemma's per-layer `feed_forward_length`).
    pub fn metadata_u64(&self, suffix: &str) -> Option<u64> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata
            .iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, v)| v.as_u64())
    }

    /// A `<arch>.<suffix>` string metadata value — e.g.
    /// `rope.scaling.type`, which selects between an unscaled rope and
    /// YaRN.
    pub fn metadata_string(&self, suffix: &str) -> Option<String> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata.iter().find_map(|(k, v)| {
            (*k == key).then_some(v).and_then(|v| match v {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
        })
    }

    /// A `<arch>.<suffix>` numeric metadata value as `f32`.
    ///
    /// Integer-typed values are accepted too, not just `F32`/`F64`: a
    /// conversion script is free to write a whole-numbered hyperparameter
    /// (`mistral3.rope.scaling.yarn_beta_fast = 32`) as an integer, and
    /// silently returning `None` for it would fall back to a default that
    /// happens to be plausible — the worst kind of wrong.
    pub fn metadata_f32(&self, suffix: &str) -> Option<f32> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata.iter().find_map(|(k, v)| {
            (*k == key).then_some(v).and_then(|v| match v {
                GgufValue::F32(f) => Some(*f),
                GgufValue::F64(f) => Some(*f as f32),
                other => other.as_u64().map(|i| i as f32),
            })
        })
    }

    /// A `<arch>.<suffix>` array metadata value, each element widened to
    /// `u64` — e.g. Gemma's per-layer `feed_forward_length` or the boolean
    /// `attention.sliding_window_pattern`.
    pub fn metadata_array_u64(&self, suffix: &str) -> Option<Vec<u64>> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata.iter().find_map(|(k, v)| {
            (*k == key).then_some(v).and_then(|v| match v {
                GgufValue::Array(items) => Some(items.iter().filter_map(|i| i.as_u64()).collect()),
                _ => None,
            })
        })
    }

    /// Dequantizes tensor `name` to `f32`, in GGUF's own (reversed-from-
    /// row-major) dimension order — callers index it the same way ggml
    /// tensor shapes are documented (`dims[0]` is the fastest-varying).
    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, &[u64])> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("model is missing tensor '{name}'"))?;
        let bytes = &loc.bytes[loc.start..loc.start + loc.len];
        let element_count: u64 = loc.dims.iter().product();
        let values = quant::dequantize(loc.ggml_type, bytes, element_count as usize)
            .with_context(|| format!("tensor '{name}'"))?;
        Ok((values, &loc.dims))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// A lazy, `mmap`-backed view of tensor `name`, for weight matrices and
    /// embedding tables (see [`QuantMatrix`]) — anything large enough that
    /// eagerly dequantizing the whole thing at load time would matter. The
    /// tensor must be 2D; `dims[0]` (ggml's fastest-varying dimension) is
    /// each row's length, `dims[1]` the row count — the same shape
    /// [`LoadedModel::tensor`] already returns, just read lazily per row.
    pub fn matrix(&self, name: &str) -> Result<QuantMatrix> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("model is missing tensor '{name}'"))?;
        anyhow::ensure!(
            loc.dims.len() == 2,
            "tensor '{name}' is not a 2D matrix (dims: {:?})",
            loc.dims
        );
        let in_dim = loc.dims[0] as usize;
        let out_dim = loc.dims[1] as usize;
        let row_bytes = quant::tensor_byte_size(loc.ggml_type, in_dim as u64)
            .with_context(|| format!("tensor '{name}'"))? as usize;
        anyhow::ensure!(
            row_bytes * out_dim == loc.len,
            "tensor '{name}': row size {row_bytes} x {out_dim} rows doesn't match the tensor's {} total bytes",
            loc.len
        );
        Ok(QuantMatrix {
            bytes: loc.bytes.clone(),
            ggml_type: loc.ggml_type,
            start: loc.start,
            row_bytes,
            in_dim,
            out_dim,
        })
    }

    /// Like [`LoadedModel::matrix`], for a 3D "stacked per-expert" tensor
    /// (see [`ExpertQuantMatrix`]). `dims[0]` is each row's length,
    /// `dims[1]` the row count per expert, `dims[2]` the expert count.
    pub fn expert_matrix(&self, name: &str) -> Result<ExpertQuantMatrix> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("model is missing tensor '{name}'"))?;
        anyhow::ensure!(
            loc.dims.len() == 3,
            "tensor '{name}' is not a 3D stacked-expert tensor (dims: {:?})",
            loc.dims
        );
        let in_dim = loc.dims[0] as usize;
        let out_dim = loc.dims[1] as usize;
        let n_expert = loc.dims[2] as usize;
        let row_bytes = quant::tensor_byte_size(loc.ggml_type, in_dim as u64)
            .with_context(|| format!("tensor '{name}'"))? as usize;
        let expert_stride = row_bytes * out_dim;
        anyhow::ensure!(
            expert_stride * n_expert == loc.len,
            "tensor '{name}': row size {row_bytes} x {out_dim} rows x {n_expert} experts doesn't match the tensor's {} total bytes",
            loc.len
        );
        Ok(ExpertQuantMatrix {
            bytes: loc.bytes.clone(),
            ggml_type: loc.ggml_type,
            start: loc.start,
            row_bytes,
            expert_stride,
            in_dim,
            out_dim,
            n_expert,
        })
    }
}

fn metadata_string(gguf: &GgufFile, key: &str) -> Option<String> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
    })
}

fn metadata_u64(gguf: &GgufFile, key: &str) -> Option<u64> {
    gguf.metadata
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.as_u64())
}

fn metadata_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::F32(f) => Some(*f),
            GgufValue::F64(f) => Some(*f as f32),
            _ => None,
        })
    })
}

fn required_u64(gguf: &GgufFile, architecture: &str, suffix: &str) -> Result<u64> {
    let key = format!("{architecture}.{suffix}");
    metadata_u64(gguf, &key).ok_or_else(|| anyhow!("GGUF file is missing {key}"))
}

fn read_model_config(gguf: &GgufFile, architecture: &str) -> Result<ModelConfig> {
    let n_embd = required_u64(gguf, architecture, "embedding_length")? as usize;
    let n_layer = required_u64(gguf, architecture, "block_count")? as usize;
    let n_head = required_u64(gguf, architecture, "attention.head_count")? as usize;
    let n_head_kv = metadata_u64(gguf, &format!("{architecture}.attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(n_head);
    let n_ctx_train = required_u64(gguf, architecture, "context_length")? as usize;
    let n_vocab = metadata_u64(gguf, &format!("{architecture}.vocab_size"))
        .map(|v| v as usize)
        .or_else(|| {
            gguf.metadata
                .iter()
                .find(|(k, _)| k == "tokenizer.ggml.tokens")
                .and_then(|(_, v)| match v {
                    GgufValue::Array(items) => Some(items.len()),
                    _ => None,
                })
        })
        .ok_or_else(|| anyhow!("GGUF file has no vocab_size and no tokenizer.ggml.tokens"))?;

    if n_head == 0 || n_head_kv == 0 {
        bail!("{architecture}.attention.head_count(_kv) must be nonzero");
    }
    let head_dim = metadata_u64(gguf, &format!("{architecture}.attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or(n_embd / n_head);
    if head_dim == 0 {
        bail!("{architecture}.attention.key_length must be nonzero");
    }
    let rope_dim = metadata_u64(gguf, &format!("{architecture}.rope.dimension_count"))
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    let rope_freq_base =
        metadata_f32(gguf, &format!("{architecture}.rope.freq_base")).unwrap_or(10000.0);
    let rms_eps = metadata_f32(
        gguf,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
    )
    .unwrap_or(1e-5);
    // llama.cpp's `enum llama_pooling_type`: NONE=0, MEAN=1, CLS=2, LAST=3,
    // RANK=4 — only LAST is distinguished here, everything else (including
    // absent) falls back to MEAN; see `PoolingType`'s own doc comment.
    let pooling_type = match metadata_u64(gguf, &format!("{architecture}.pooling_type")) {
        Some(3) => PoolingType::Last,
        _ => PoolingType::Mean,
    };

    Ok(ModelConfig {
        architecture: architecture.to_string(),
        n_vocab,
        n_embd,
        n_layer,
        n_head,
        n_head_kv,
        head_dim,
        n_ctx_train,
        rope_dim,
        rope_freq_base,
        rms_eps,
        pooling_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `[in_dim, out_dim]` Q8_0 matrix whose row `r` is filled with the
    /// byte `r + 1`, so a view can be checked to start where it claims.
    fn patterned_matrix(in_dim: usize, out_dim: usize) -> QuantMatrix {
        // Q8_0: 32 weights per block, 2 scale bytes + 32 value bytes.
        let row_bytes = in_dim / 32 * 34;
        let mut bytes = vec![0u8; row_bytes * out_dim];
        for r in 0..out_dim {
            for b in &mut bytes[r * row_bytes..(r + 1) * row_bytes] {
                *b = (r + 1) as u8;
            }
        }
        test_quant_matrix(&bytes, 8, in_dim, out_dim)
    }

    #[test]
    fn rows_view_starts_where_it_says_and_keeps_the_row_width() {
        let m = patterned_matrix(64, 8);
        let tail = m.rows(3, 5);
        assert_eq!(tail.out_dim, 5);
        assert_eq!(tail.in_dim, m.in_dim);
        assert_eq!(tail.row_bytes(), m.row_bytes());
        // The view's row 0 must be the parent's row 3, byte for byte.
        assert_eq!(
            tail.raw_bytes()[..tail.row_bytes()],
            m.raw_bytes()[3 * m.row_bytes()..4 * m.row_bytes()],
        );
        assert_eq!(tail.raw_bytes().len(), 5 * m.row_bytes());
    }

    /// The regression test for the reason [`QuantMatrix::cache_key`] carries a
    /// length. A view of the *leading* rows shares its parent's start address,
    /// so a key of `(mmap base, start)` — what this returned before row views
    /// existed — cannot tell the two apart, and every backend cache addressed
    /// by it would hand the view the whole tensor's buffers and bind groups.
    #[test]
    fn cache_key_distinguishes_a_leading_row_view_from_the_whole_tensor() {
        let m = patterned_matrix(64, 8);
        let head = m.rows(0, 4);
        assert_eq!(
            head.cache_key().0,
            m.cache_key().0,
            "a leading view does start at the same address — which is the trap",
        );
        assert_ne!(
            head.cache_key(),
            m.cache_key(),
            "leading view and whole tensor must not share a cache key",
        );
        // And the two halves of a split must differ from each other.
        assert_ne!(m.rows(0, 4).cache_key(), m.rows(4, 4).cache_key());
        // A view spanning everything *is* the tensor, and may share its key.
        assert_eq!(m.rows(0, 8).cache_key(), m.cache_key());
    }

    #[test]
    #[should_panic(expected = "exceeds out_dim")]
    fn rows_view_rejects_a_range_past_the_end() {
        patterned_matrix(64, 8).rows(6, 4);
    }

    #[test]
    fn resolve_arch_family_accepts_llama_style_architectures() {
        for arch in LLAMA_STYLE_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::LlamaStyle);
        }
    }

    #[test]
    fn resolve_arch_family_accepts_gemma_architectures() {
        for arch in GEMMA_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Gemma);
        }
    }

    #[test]
    fn resolve_arch_family_accepts_qwen35moe() {
        for arch in QWEN35MOE_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Qwen35Moe);
        }
    }

    #[test]
    fn resolve_arch_family_accepts_qwen35() {
        for arch in QWEN35_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Qwen35);
        }
    }

    #[test]
    fn resolve_arch_family_accepts_phi3() {
        for arch in PHI3_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Phi3);
        }
    }

    /// `phi2` and `phimoe` share a name prefix with `phi3` but neither
    /// shares its forward pass (`phi2`: LayerNorm + GELU + parallel
    /// attention/FFN; `phimoe`: routed experts). A prefix-match rather than
    /// an exact-match here would load either one through `arch::phi` and
    /// produce garbage instead of an "unsupported" error.
    #[test]
    fn resolve_arch_family_rejects_other_phi_architectures() {
        for arch in ["phi2", "phimoe"] {
            let err = resolve_arch_family(arch).unwrap_err();
            assert!(err.to_string().contains("not yet supported"), "{err}");
        }
    }

    #[test]
    fn resolve_arch_family_rejects_unknown_architectures() {
        let err = resolve_arch_family("bert").unwrap_err();
        assert!(err.to_string().contains("not yet supported"), "{err}");
    }

    /// A header-only `GgufFile` (no tensor data) carrying one architecture
    /// key and the given tensor names — enough to exercise `model_load_support`.
    fn header_only(architecture: &str, tensor_names: &[&str]) -> GgufFile {
        GgufFile {
            version: 3,
            metadata: vec![(
                "general.architecture".to_string(),
                GgufValue::String(architecture.to_string()),
            )],
            tensors: tensor_names
                .iter()
                .map(|name| orangu::gguf::TensorInfo {
                    name: name.to_string(),
                    dims: vec![1],
                    ggml_type: 0,
                    offset: 0,
                })
                .collect(),
            alignment: 32,
            data_offset: 0,
        }
    }

    #[test]
    fn model_load_support_accepts_a_dense_gemma_model() {
        let (arch, bad_quant) =
            model_load_support(&header_only("gemma4", &["blk.0.ffn_gate.weight"]));
        assert_eq!(arch.as_deref(), Some("gemma4"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_a_moe_gemma_model() {
        // A gemma checkpoint with per-layer MoE expert tensors
        // (`gemma-4-26B-A4B`) — `arch::gemma` now loads the routed-expert
        // path, so it's reported supported under the plain architecture.
        let (arch, bad_quant) =
            model_load_support(&header_only("gemma4", &["blk.0.ffn_gate_inp.weight"]));
        assert_eq!(arch.as_deref(), Some("gemma4"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_a_moe_qwen_model() {
        let (arch, bad_quant) =
            model_load_support(&header_only("qwen35moe", &["blk.0.ffn_gate_inp.weight"]));
        assert_eq!(arch.as_deref(), Some("qwen35moe"));
        assert_eq!(bad_quant, None);
    }

    /// A recognised architecture whose tensors are quantized to a type this
    /// build has no dequantizer for. The header alone says `llama`, so
    /// checking only the architecture would advertise a load that dies on
    /// the first tensor — `TQ1_0` (ggml type 34) is such a type today.
    #[test]
    fn model_load_support_rejects_an_unreadable_tensor_type() {
        let mut gguf = header_only("llama", &["blk.0.attn_q.weight"]);
        gguf.tensors[0].ggml_type = 34;
        let (arch, bad_quant) = model_load_support(&gguf);
        assert_eq!(arch.as_deref(), Some("llama"));
        assert_eq!(bad_quant.as_deref(), Some("TQ1_0"));
    }

    #[test]
    fn model_load_support_reports_an_unknown_arch_unsupported() {
        let (arch, bad_quant) = model_load_support(&header_only("glm-dsa", &[]));
        assert_eq!(arch.as_deref(), Some("glm-dsa"));
        // Unreadable because of its *architecture*, not its tensor types.
        assert_eq!(bad_quant, None);
        assert!(resolve_arch_family("glm-dsa").is_err());
    }
}

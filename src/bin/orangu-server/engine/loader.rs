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
    /// final logit softcapping. See `engine::arch::gemma`.
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
    bail!(
        "architecture '{architecture}' is not yet supported by orangu-server \
         (supported: {})",
        LLAMA_STYLE_ARCHITECTURES
            .iter()
            .chain(GEMMA_ARCHITECTURES)
            .chain(QWEN35MOE_ARCHITECTURES)
            .chain(QWEN35_ARCHITECTURES)
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
/// fails. The one such case today is a gemma checkpoint carrying per-layer
/// MoE expert tensors (`ffn_gate_inp`) — [`super::arch::gemma`] is
/// dense-only and bails on it at build time — which is reported unsupported
/// under a refined `<arch>-moe` label (e.g. `gemma4` → `gemma4-moe`) so the
/// column names the actual blocker rather than the bare architecture.
///
/// Returns `(architecture, supported)`: `architecture` is `None` only when
/// the file has no `general.architecture` at all.
pub fn model_load_support(gguf: &GgufFile) -> (Option<String>, bool) {
    let architecture = metadata_string(gguf, "general.architecture");
    let Some(arch) = architecture.as_deref() else {
        return (None, false);
    };
    let Ok(family) = resolve_arch_family(arch) else {
        return (architecture, false);
    };
    if matches!(family, ArchFamily::Gemma) && has_moe_expert_tensors(gguf) {
        return (Some(format!("{arch}-moe")), false);
    }
    (architecture, true)
}

/// Whether any layer carries a routed-MoE gate (`blk.{i}.ffn_gate_inp.weight`)
/// — the same tensor `arch::gemma`'s per-layer build guard checks for.
fn has_moe_expert_tensors(gguf: &GgufFile) -> bool {
    gguf.tensors
        .iter()
        .any(|t| t.name.starts_with("blk.") && t.name.ends_with(".ffn_gate_inp.weight"))
}

/// A tensor's resolved location and shape, ready for [`quant::dequantize`].
#[derive(Debug, Clone)]
struct TensorLocation {
    ggml_type: u32,
    dims: Vec<u64>,
    /// Absolute byte offset into the mmap'd file.
    start: usize,
    len: usize,
}

pub struct LoadedModel {
    pub config: ModelConfig,
    /// The GGUF file's raw metadata key/value pairs — beyond the common
    /// subset [`ModelConfig`] captures, an architecture module (e.g.
    /// `engine::arch::gemma`) reads its own further hyperparameters
    /// (per-layer arrays, architecture-specific keys) directly from this.
    pub metadata: Vec<(String, GgufValue)>,
    mmap: Arc<Mmap>,
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
    mmap: Arc<Mmap>,
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
        let bytes = &self.mmap[offset..offset + self.row_bytes];
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
        &self.mmap[self.start..self.start + self.row_bytes * self.out_dim]
    }

    /// A stable identity for this tensor's byte range, valid for as long as
    /// the underlying `mmap` is kept alive (the model's whole lifetime) —
    /// lets a GPU backend cache an uploaded copy of this matrix keyed by
    /// identity, so a weight already on the GPU isn't re-uploaded on every
    /// `matmul` call (every decode step reuses the same weight tensors).
    pub fn cache_key(&self) -> (usize, usize) {
        (self.mmap.as_ptr() as usize, self.start)
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
        mmap,
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
    mmap: Arc<Mmap>,
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
        let bytes = &self.mmap[offset..offset + self.row_bytes];
        quant::dequantize(self.ggml_type, bytes, self.in_dim)
            .expect("row byte range was validated when this ExpertQuantMatrix was constructed")
    }
}

impl LoadedModel {
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let architecture = metadata_string(&gguf, "general.architecture")
            .ok_or_else(|| anyhow!("GGUF file is missing general.architecture"))?;
        resolve_arch_family(&architecture)?;

        let config = read_model_config(&gguf, &architecture)?;

        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // Safety: the file is opened read-only and not mutated by anything
        // else for the lifetime of this mapping — the standard caveat of
        // `Mmap::map` (another process truncating the file underneath us
        // would be undefined behavior, same risk llama.cpp itself accepts
        // when it mmaps a GGUF file).
        let mmap = Arc::new(
            unsafe { Mmap::map(&file) }
                .with_context(|| format!("failed to mmap {}", path.display()))?,
        );

        let mut tensors = HashMap::with_capacity(gguf.tensors.len());
        for tensor in &gguf.tensors {
            let element_count: u64 = tensor.dims.iter().product();
            let len = quant::tensor_byte_size(tensor.ggml_type, element_count)
                .with_context(|| format!("tensor '{}'", tensor.name))?
                as usize;
            let start = gguf.data_offset as usize + tensor.offset as usize;
            if start + len > mmap.len() {
                bail!("tensor '{}' extends past the end of the file", tensor.name);
            }
            tensors.insert(
                tensor.name.clone(),
                TensorLocation {
                    ggml_type: tensor.ggml_type,
                    dims: tensor.dims.clone(),
                    start,
                    len,
                },
            );
        }

        Ok(Self {
            config,
            metadata: gguf.metadata,
            mmap,
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

    pub fn metadata_f32(&self, suffix: &str) -> Option<f32> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata.iter().find_map(|(k, v)| {
            (*k == key).then_some(v).and_then(|v| match v {
                GgufValue::F32(f) => Some(*f),
                GgufValue::F64(f) => Some(*f as f32),
                _ => None,
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
        let bytes = &self.mmap[loc.start..loc.start + loc.len];
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
            mmap: self.mmap.clone(),
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
            mmap: self.mmap.clone(),
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
    let rope_dim = metadata_u64(gguf, &format!("{architecture}.rope.dimension_count"))
        .map(|v| v as usize)
        .unwrap_or(n_embd / n_head);
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
        let (arch, supported) =
            model_load_support(&header_only("gemma4", &["blk.0.ffn_gate.weight"]));
        assert_eq!(arch.as_deref(), Some("gemma4"));
        assert!(supported);
    }

    #[test]
    fn model_load_support_rejects_a_moe_gemma_model_as_gemma4_moe() {
        // A gemma checkpoint with per-layer MoE expert tensors — recognised
        // architecture, but `arch::gemma` bails on `ffn_gate_inp` at build
        // time, so it must be reported unsupported up front, under `-moe`.
        let (arch, supported) =
            model_load_support(&header_only("gemma4", &["blk.0.ffn_gate_inp.weight"]));
        assert_eq!(arch.as_deref(), Some("gemma4-moe"));
        assert!(!supported);
    }

    #[test]
    fn model_load_support_does_not_moe_flag_a_non_gemma_arch() {
        // `qwen35moe` genuinely supports `ffn_gate_inp`, so the MoE guard
        // must not fire for it — only gemma is dense-only here.
        let (arch, supported) =
            model_load_support(&header_only("qwen35moe", &["blk.0.ffn_gate_inp.weight"]));
        assert_eq!(arch.as_deref(), Some("qwen35moe"));
        assert!(supported);
    }

    #[test]
    fn model_load_support_reports_an_unknown_arch_unsupported() {
        let (arch, supported) = model_load_support(&header_only("glm-dsa", &[]));
        assert_eq!(arch.as_deref(), Some("glm-dsa"));
        assert!(!supported);
    }
}

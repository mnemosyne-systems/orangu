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
    /// Qwen3-Next - the same hybrid full-attention/gated-DeltaNet block
    /// family as Qwen3.5, but with the Qwen3-Next MoE tensor layout. See
    /// `engine::arch::qwen3next`.
    Qwen3Next,
    /// GLM with DeepSeek sparse attention (`glm-dsa`, e.g. GLM-5.2) —
    /// absorbed multi-head latent attention over a compressed key/value
    /// cache, a lightning indexer that picks which positions each layer
    /// attends, leading dense layers, and sigmoid-routed experts. See
    /// `engine::arch::glm`.
    GlmDsa,
    /// Kimi-K3 (`kimi-k3`) — three-in-four Kimi Delta Attention layers
    /// (a gated delta-net with a per-dimension decay) alternating with
    /// absorbed multi-head latent attention, cross-layer residual
    /// attention, routed experts running in a latent space, and the situ
    /// activation in place of SwiGLU. See `engine::arch::kimi3`.
    KimiK3,
    /// DeepSeek draft sidecars (`dflash`/DSpark) — recognized by the
    /// inventory, and served by resolving the paired target model they
    /// draft for (`main::auto_pair_dflash_target`). They carry no token
    /// embeddings and no output projection, so there is no standalone model
    /// to run; see `engine::arch::dflash`.
    DFlash,
    /// DeepSeek-V4 (`deepseek4`) — four-stream hyper-connections, one
    /// shared key/value vector per token across all query heads, per-layer
    /// compressed attention (128-token blocks, or indexer-selected 4-token
    /// blocks) on top of a sliding window, and hash-routed experts on the
    /// first layers. See `engine::arch::deepseek4`.
    Deepseek4,
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
    /// Muse-Glimmer (`muse-glimmer`) — a dense GQA decoder with gemma-style
    /// sandwich norms around both sub-layers, per-head QK-norm, a sigmoid
    /// output gate on attention (`attn_gate`), an alternating pattern of
    /// sliding-window and full-attention layers where **only the
    /// sliding-window ones rotate**, and both a logit scale and final logit
    /// softcapping on the tail. See `engine::arch::muse`.
    Muse,
    /// Inkling (`inkling`) — a rotation-free MoE decoder: position enters
    /// through a learned per-head relative-position bias and a causal
    /// depthwise short convolution on the key/value projections and on
    /// each sub-layer's output, layers alternate sliding-window and
    /// full-attention (the latter with a context-length attention
    /// temperature), and sigmoid-routed experts share their normalization
    /// with the always-on shared experts. See `engine::arch::inkling`.
    Inkling,
    /// Ling 3.0 (`bailingmoe3`) — three-in-four Kimi Delta Attention
    /// layers alternating with gated, **rotated** absorbed multi-head
    /// latent attention (the pair `engine::arch::kda` shares with
    /// [`ArchFamily::KimiK3`]), a leading dense block, and sigmoid-routed
    /// experts under DeepSeek-V3 group-limited selection. See
    /// `engine::arch::bailingmoe`.
    BailingMoe3,
    /// Nemotron-H (`nemotron_h_moe`) — a hybrid whose blocks are a *single*
    /// sub-layer each rather than the usual attention-plus-FFN pair: a
    /// selective state-space mixer, an unrotated (position-free) attention,
    /// or a squared-ReLU mixture-of-experts FFN, chosen per block by the
    /// file's own per-layer `feed_forward_length` and
    /// `attention.head_count_kv`. See `engine::arch::nemotron`.
    NemotronHMoe,
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
/// `qwen3next` (e.g. `unsloth/Qwen3-Coder-Next-GGUF`) - Qwen3-Next's
/// hybrid attention and MoE architecture.
const QWEN3NEXT_ARCHITECTURES: &[&str] = &["qwen3next"];
const DFLASH_ARCHITECTURES: &[&str] = &["dflash"];
const DEEPSEEK4_ARCHITECTURES: &[&str] = &["deepseek4"];
/// `glm-dsa` (e.g. `unsloth/GLM-5.2-GGUF`) — GLM's transformer block with
/// DeepSeek's sparse (indexer-selected) latent attention. Plain `glm4`/
/// `glm4moe` are *not* here: they are ordinary GQA models with none of
/// this module's MLA or indexer machinery.
const GLM_DSA_ARCHITECTURES: &[&str] = &["glm-dsa"];
/// `kimi-k3` (e.g. `unsloth/Kimi-K3-GGUF`). `kimi-linear` is *not* here:
/// it shares the delta-net attention but none of K3's cross-layer
/// residuals, latent MoE, or situ activation.
const KIMI_K3_ARCHITECTURES: &[&str] = &["kimi-k3"];
/// `phi3` covers both Phi-3 and Phi-4-mini (e.g. `unsloth/Phi-4-mini-
/// instruct-GGUF`) — upstream converts both under the one
/// `general.architecture` string, and `llama_model_phi3` serves both from
/// one graph. `phi2` is *not* here: it's a different shape entirely
/// (LayerNorm rather than RMSNorm, GELU rather than SwiGLU, parallel
/// attention/FFN branches, biases throughout). `phimoe` (Phi-3.5-MoE) isn't
/// either — same attention block, but routed experts this module has no
/// path for.
const PHI3_ARCHITECTURES: &[&str] = &["phi3"];
/// `muse-glimmer` (e.g. `unsloth/Muse-Glimmer-30B-GGUF`) — see
/// [`ArchFamily::Muse`] and `engine::arch::muse`.
const MUSE_ARCHITECTURES: &[&str] = &["muse-glimmer"];
/// `inkling` (e.g. `unsloth/Inkling-Small-GGUF`) — see
/// [`ArchFamily::Inkling`] and `engine::arch::inkling`. The `mmproj-*.gguf`
/// shipped alongside is a separate `clip`-architecture model and is not
/// this.
const INKLING_ARCHITECTURES: &[&str] = &["inkling"];
/// `nemotron_h_moe` (e.g. `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-
/// GGUF`) — see [`ArchFamily::NemotronHMoe`] and `engine::arch::nemotron`.
/// `nemotron_h` is the dense sibling — the same one-sub-layer-per-block
/// trunk with an ordinary two-matrix FFN where the MoE variant routes
/// experts (`engine::arch::nemotron`'s `Layer::Ffn`), confirmed against
/// `bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF`. The unrelated,
/// ordinary-transformer `nemotron` is *not* here.
const NEMOTRON_ARCHITECTURES: &[&str] = &["nemotron_h_moe", "nemotron_h"];
/// `bailingmoe3` (e.g. `bartowski/Ling-3.0-tiny-GGUF`,
/// `bartowski/Ling-3.0-flash-GGUF`) — see [`ArchFamily::BailingMoe3`] and
/// `engine::arch::bailingmoe`. Ling 1.0's `bailingmoe` and Ling 2.0's
/// `bailingmoe2` are *not* here: despite the name they are ordinary
/// GQA+RoPE transformers with routed experts, sharing none of this
/// module's delta-net or latent-attention machinery.
const BAILINGMOE3_ARCHITECTURES: &[&str] = &["bailingmoe3"];

/// Architectures this engine recognises by name and deliberately does **not**
/// serve, with the reason.
///
/// Every entry is a *near miss*: a name one character or one word away from a
/// supported family, whose graph differs in a way that would produce
/// plausible, wrong output rather than an error. Those are exactly the files
/// someone reaches for a workaround over — "it says `glm4`, and `glm-dsa` is
/// supported, so surely…" — and exactly the ones where a workaround is worst.
///
/// The reasons are not new analysis: they were already written down beside the
/// lists above, where only someone reading the source could find them. Saying
/// them in the error is the whole point, because the person who needs them is
/// holding a file that will not load and has no reason to open `loader.rs`.
const KNOWN_UNSUPPORTED: &[(&str, &str)] = &[
    (
        "glm4",
        "an ordinary GQA model; the supported 'glm-dsa' is a different graph \
         (latent attention with a sparse indexer), not a newer version of this one",
    ),
    (
        "glm4moe",
        "an ordinary GQA model with routed experts; the supported 'glm-dsa' shares its \
         name and not its graph",
    ),
    (
        "kimi-linear",
        "shares delta-net attention with the supported 'kimi-k3', but none of its \
         cross-layer residuals, latent MoE or situ activation",
    ),
    (
        "phi2",
        "a different shape from the supported 'phi3': LayerNorm rather than RMSNorm, \
         GELU rather than SwiGLU, parallel attention and FFN branches, biases throughout",
    ),
    (
        "phimoe",
        "the supported 'phi3' attention block with routed experts, which that module \
         has no path for",
    ),
    (
        "nemotron",
        "an ordinary transformer, unrelated to the supported 'nemotron_h_moe' despite the \
         shared name",
    ),
    (
        "bailingmoe",
        "Ling 1.0: an ordinary GQA transformer with routed experts, sharing none of the \
         supported 'bailingmoe3' delta-net trunk or latent attention",
    ),
    (
        "bailingmoe2",
        "Ling 2.0: an ordinary GQA transformer with routed experts; 'bailingmoe3' is a \
         different graph, not a newer version of this one",
    ),
];

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
    if QWEN3NEXT_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Qwen3Next);
    }
    if DFLASH_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::DFlash);
    }
    if DEEPSEEK4_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Deepseek4);
    }
    if GLM_DSA_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::GlmDsa);
    }
    if KIMI_K3_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::KimiK3);
    }
    if PHI3_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Phi3);
    }
    if MISTRAL_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Mistral3);
    }
    if MUSE_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Muse);
    }
    if INKLING_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::Inkling);
    }
    if NEMOTRON_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::NemotronHMoe);
    }
    if BAILINGMOE3_ARCHITECTURES.contains(&architecture) {
        return Ok(ArchFamily::BailingMoe3);
    }
    // A recognised near miss gets its own reason. The alternative — dropping
    // the reader into a list of fourteen names that looks like it *should*
    // contain theirs — is what makes an out-of-tree alias mechanism sound
    // attractive, and an alias is the one answer that produces wrong output
    // instead of no output.
    if let Some((_, why)) = KNOWN_UNSUPPORTED
        .iter()
        .find(|(name, _)| *name == architecture)
    {
        bail!(
            "architecture '{architecture}' is recognised but not supported: {why}. \
             Serving it needs its own forward pass, verified against a real checkpoint — \
             see doc/ENGINE.md. Pointing this file at a similar architecture would load \
             and generate confident nonsense."
        );
    }
    bail!(
        "architecture '{architecture}' is not yet supported by orangu-server \
         (supported: {})",
        LLAMA_STYLE_ARCHITECTURES
            .iter()
            .chain(GEMMA_ARCHITECTURES)
            .chain(QWEN35MOE_ARCHITECTURES)
            .chain(QWEN35_ARCHITECTURES)
            .chain(QWEN3NEXT_ARCHITECTURES)
            .chain(DFLASH_ARCHITECTURES)
            .chain(DEEPSEEK4_ARCHITECTURES)
            .chain(GLM_DSA_ARCHITECTURES)
            .chain(KIMI_K3_ARCHITECTURES)
            .chain(PHI3_ARCHITECTURES)
            .chain(MISTRAL_ARCHITECTURES)
            .chain(MUSE_ARCHITECTURES)
            .chain(INKLING_ARCHITECTURES)
            .chain(NEMOTRON_ARCHITECTURES)
            .chain(BAILINGMOE3_ARCHITECTURES)
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
        .find(|t| {
            !quant::supports_type(t.ggml_type)
                && !matches!(
                    (t.ggml_type, t.name.as_str()),
                    (26, name) if name.ends_with(".ffn_gate_tid2eid.weight")
                )
        })
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

/// Drops the resident pages behind a mapped byte range that has just been
/// copied into owned storage, so the data is not resident twice for the rest
/// of the model's life.
///
/// Only the repacked layouts reach this (see [`LoadedModel::open`]), and only
/// they had the problem: `SmolLM2-360M-Instruct-Q4_0_4_4` held **426 MiB
/// resident for a 217 MiB file** — 217 mapped and touched by the un-repack,
/// plus ~210 rewritten — where every other model in `doc/perf/models.tsv`
/// sits at file size plus 60–110 MiB.
///
/// `madvise` rather than dropping the mapping, because the mapping cannot be
/// dropped: a repacked file may store some tensors un-repacked (that model's
/// `token_embd` is `Q8_0`) and those still read through it. `MADV_DONTNEED` on
/// a read-only `MAP_PRIVATE` file mapping frees the page-cache references
/// without invalidating any address — a later read of the same range would
/// simply fault it back from the file, unchanged. Nothing does read it again;
/// the range's only reader was the un-repack that just finished.
///
/// Rounded *inward* to whole pages, since a page is the unit `madvise` works
/// in and a tensor boundary need not be page-aligned. Advising a partial page
/// would reach into a neighbouring tensor's bytes — harmless in itself (it
/// would fault back) but not this function's business, and at these sizes the
/// two skipped pages are noise.
///
/// Best-effort by construction: failure means pages stay resident, which is
/// exactly the status quo this improves on, so there is nothing to report and
/// nothing a caller could do about it.
fn release_mapped_range(range: &[u8]) {
    #[cfg(target_os = "linux")]
    {
        // Safety: `sysconf` is a pure query with no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return;
        }
        let page = page as usize;
        let start = range.as_ptr() as usize;
        let end = start + range.len();
        let aligned_start = start.next_multiple_of(page);
        let aligned_end = end - (end % page);
        if aligned_end <= aligned_start {
            return;
        }
        // Safety: the range is inside a live mapping this crate created
        // read-only (the caller holds its `Arc` across the call), the pointer
        // and length are page-aligned as `madvise` requires, and
        // `MADV_DONTNEED` on a read-only file mapping only discards clean
        // pages — it cannot lose data or invalidate the address.
        unsafe {
            libc::madvise(
                aligned_start as *mut libc::c_void,
                aligned_end - aligned_start,
                libc::MADV_DONTNEED,
            );
        }
    }
    // Every other target keeps both copies resident, as this did everywhere
    // before. `MADV_DONTNEED` is POSIX but its effect on a private file
    // mapping is not: on some BSDs it is advisory to the point of being a
    // no-op, and nothing here is measured on one.
    #[cfg(not(target_os = "linux"))]
    let _ = range;
}

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
    /// Which device holds each transformer layer, when the model is spread
    /// across more than one (`engine::placement`). Empty for the ordinary
    /// single-device case, which is also what every tensor outside a
    /// numbered `blk.<n>.` block gets.
    ///
    /// Set once, by `main` between loading the weights and building the
    /// model, because [`Self::matrix`] is what stamps it onto each tensor
    /// and every architecture calls that during construction.
    layer_device: Vec<usize>,
    /// Which experts of each `*_exps.weight` tensor a device tier holds —
    /// see `ExpertQuantMatrix::residency`. Empty when no tier is active,
    /// and set at the same moment (and for the same reason) as
    /// `layer_device`.
    expert_residency: HashMap<String, Arc<[bool]>>,
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
    /// Which device in the selected set holds this tensor — a position in
    /// `engine::placement`'s plan, `0` for every tensor of an unsplit
    /// model.
    ///
    /// Stamped by [`LoadedModel::matrix`] from the layer number in the
    /// tensor's name, and read by `engine::backend::multi::
    /// MultiDeviceBackend` to route the matmul. It rides on the matrix
    /// rather than being looked up per call because the matrix is the only
    /// thing `Backend::matmul` is given that can identify the layer at all
    /// — which is what lets a split model need no change to any
    /// architecture's forward pass.
    device: usize,
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
            // A slice of a tensor is on the same device as the tensor. Not
            // inheriting this would send half a layer's rows to device 0
            // and produce a result that is wrong only on a split model.
            device: self.device,
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
    /// Which device in the selected set holds this tensor — see the field's
    /// own doc. `0` unless a split plan said otherwise.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Overrides the device tag, for tests that need a matrix on a
    /// particular device without a `LoadedModel` behind it.
    #[cfg(test)]
    pub fn set_device(&mut self, device: usize) {
        self.device = device;
    }

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
        device: 0,
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
    /// The tensor's GGUF name (`blk.7.ffn_gate_exps.weight`).
    ///
    /// Carried because the runtime identity [`Self::tensor_id`] returns is an
    /// *address*, which is meaningless once the process exits. Anything that
    /// remembers something about an expert across restarts — routing history,
    /// in `engine::expert_store` — has to key on a name instead.
    name: Arc<str>,
    bytes: TensorBytes,
    ggml_type: u32,
    start: usize,
    row_bytes: usize,
    expert_stride: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_expert: usize,
    /// Which of this tensor's experts a device holds, when a device expert
    /// tier is active — one flag per expert, `None` when there is no tier.
    ///
    /// Per expert rather than per tensor because that is the unit a tier
    /// works in: `engine::expert_tier` places whole experts, and one
    /// tensor's experts routinely straddle the budget. Shared behind an
    /// `Arc` because every `ExpertQuantMatrix` for a tensor is a fresh
    /// value built by `LoadedModel::expert_matrix`, and the flags are the
    /// same for all of them.
    residency: Option<Arc<[bool]>>,
}

/// An `F32` [`ExpertQuantMatrix`] over deterministic generated values, for
/// tests that need a real per-expert tensor rather than a real model.
///
/// `F32` on purpose: a test of *how rows are read* must not also depend on a
/// quantization kernel, or a failure in either one looks like a failure in the
/// other. Values are distinct across `(expert, row, column)` so a projection
/// that reads the wrong expert, the wrong row or the wrong offset produces a
/// wrong number rather than a coincidentally right one.
#[cfg(test)]
pub(crate) fn test_expert_matrix(
    n_expert: usize,
    out_dim: usize,
    in_dim: usize,
) -> ExpertQuantMatrix {
    test_expert_matrix_named("test.exps.weight", n_expert, out_dim, in_dim)
}

/// [`test_expert_matrix`] with every expert marked device-resident, so a
/// test can reach the batched-dispatch branch that a residency check
/// otherwise gates off.
///
/// The point is not to pretend there is a GPU: paired with `CpuBackend` the
/// "device" branch runs `Backend::matmul_batch` on the host, which is what
/// makes the branch's *bookkeeping* — the row-range views it builds, the
/// order it puts results back in — testable on a machine with no GPU at all.
/// Without this the branch is dead code in every test, and the row ranges a
/// fused gate/up tensor depends on are only exercised on the fallback side.
#[cfg(test)]
pub(crate) fn test_expert_matrix_resident(
    n_expert: usize,
    out_dim: usize,
    in_dim: usize,
) -> ExpertQuantMatrix {
    let mut matrix = test_expert_matrix(n_expert, out_dim, in_dim);
    matrix.residency = Some(std::iter::repeat_n(true, n_expert).collect());
    matrix
}

/// [`test_expert_matrix`] with a chosen tensor name, for tests about anything
/// that keys on the name rather than on the runtime address.
#[cfg(test)]
pub(crate) fn test_expert_matrix_named(
    name: &str,
    n_expert: usize,
    out_dim: usize,
    in_dim: usize,
) -> ExpertQuantMatrix {
    let mut bytes = Vec::with_capacity(n_expert * out_dim * in_dim * 4);
    for expert in 0..n_expert {
        for row in 0..out_dim {
            for column in 0..in_dim {
                let value = (expert * 1000 + row * 31 + column) as f32 * 0.017 - 1.0;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    let matrix = test_quant_matrix(
        &bytes,
        crate::engine::quant::GGML_TYPE_F32,
        in_dim,
        n_expert * out_dim,
    );
    ExpertQuantMatrix {
        name: Arc::from(name),
        residency: None,
        bytes: matrix.bytes,
        ggml_type: matrix.ggml_type,
        start: 0,
        row_bytes: in_dim * 4,
        expert_stride: out_dim * in_dim * 4,
        in_dim,
        out_dim,
        n_expert,
    }
}

impl ExpertQuantMatrix {
    /// One expert's own weight bytes in this tensor, still quantized —
    /// `out_dim` rows of `row_bytes`. What reading that expert costs, which
    /// `engine::moe_stats` multiplies out per routed selection.
    ///
    /// Deliberately the rows' own extent rather than `expert_stride`: the
    /// stride is where the *next* expert starts and would fold any padding
    /// between them into a figure meant to be comparable against bytes a
    /// batch-union implementation would move.
    pub fn expert_bytes(&self) -> u64 {
        (self.row_bytes * self.out_dim) as u64
    }

    /// The quantization these experts are stored in — what a GPU backend
    /// needs a kernel for before it can be handed one.
    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }

    /// Bytes per row, still quantized — how [`Self::expert_span`] (or a
    /// residency tier's copy of it) is sliced by a kernel that reads the
    /// quantized bytes directly instead of dequantizing a row first. Same
    /// stride [`Self::row_from`] uses.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Whether a device expert tier holds `expert`.
    ///
    /// `false` when there is no tier at all, which is the default and which
    /// keeps every expert on the host path. A tier that answered `true`
    /// everywhere would be the unbounded arena this exists to bound.
    pub fn is_device_resident(&self, expert: usize) -> bool {
        self.residency
            .as_ref()
            .and_then(|flags| flags.get(expert))
            .copied()
            .unwrap_or(false)
    }

    /// The tensor's GGUF name — stable across runs, unlike
    /// [`Self::tensor_id`].
    pub fn name(&self) -> &Arc<str> {
        &self.name
    }

    /// This tensor's stable process-lifetime identity — the address its
    /// mapped bytes begin at, which is what the GPU backends already key a
    /// weight on. Two views of one tensor agree; two different tensors do
    /// not collide, since the mappings are held for the model's whole life.
    pub fn tensor_id(&self) -> usize {
        self.bytes.as_ptr() as usize + self.start
    }

    /// One expert as an ordinary [`QuantMatrix`] — the same bytes, viewed as
    /// the `[in_dim, out_dim]` matrix they already are.
    ///
    /// Zero-copy: an expert's rows are contiguous in the stacked tensor, so
    /// this is the same `start`/`row_bytes`/`in_dim`/`out_dim` arithmetic
    /// [`Self::row`] does, hoisted out of the per-row loop.
    ///
    /// # Why this exists
    ///
    /// It is the seam a device-resident expert tier needs, and the reason
    /// such a tier needs no new kernels. `Backend::matmul` takes a
    /// `QuantMatrix`; every GPU backend already has a kernel for every quant
    /// type an expert is stored in; and `engine::backend::multi` already
    /// routes a `QuantMatrix` by the device stamped on it. So an expert that
    /// a placement policy put on a device is a `matmul` call, not a porting
    /// exercise.
    ///
    /// Nothing calls this on a hot path yet — see `engine::expert_tier` for
    /// what would have to be true first — but it is what makes the rest of
    /// that work a dispatch question rather than a kernel one.
    ///
    pub fn expert_matrix(&self, expert: usize) -> QuantMatrix {
        assert!(
            expert < self.n_expert,
            "expert {expert} >= {}",
            self.n_expert
        );
        QuantMatrix {
            bytes: self.bytes.clone(),
            ggml_type: self.ggml_type,
            start: self.start + expert * self.expert_stride,
            row_bytes: self.row_bytes,
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            // Expert tensors are host-resident today (`engine::backend::
            // is_cpu_only_tensor`), so a view of one is device 0's until a
            // tier says otherwise. A tier would stamp this from its own
            // placement, exactly as `LoadedModel::matrix` does from the
            // layer plan.
            device: 0,
        }
    }

    /// One expert's still-quantized bytes, as they lie in the mapping.
    ///
    /// The unit a residency policy works in: `engine::expert_store` asks the
    /// kernel whether *this* range is in RAM, which is a question about a
    /// slice of a tensor rather than about the whole shard.
    pub fn expert_span(&self, expert: usize) -> &[u8] {
        let offset = self.start + expert * self.expert_stride;
        &self.bytes[offset..offset + self.row_bytes * self.out_dim]
    }

    /// [`Self::row_into`], but reading from `span` — one expert's bytes as
    /// [`Self::expert_span`] lays them out — rather than from the mapping.
    ///
    /// What lets a residency tier serve a row from its own copy: the bytes
    /// are the same bytes, so the values are the same values, and nothing
    /// downstream can tell which side of the tier they came from.
    pub fn row_from(&self, span: &[u8], index: usize, out: &mut Vec<f32>) {
        let offset = index * self.row_bytes;
        let bytes = &span[offset..offset + self.row_bytes];
        quant::dequantize_into(self.ggml_type, bytes, self.in_dim, out)
            .expect("row byte range was validated when this ExpertQuantMatrix was constructed");
    }

    /// Dequantizes row `index` of expert `expert` (`in_dim` values).
    ///
    /// **Test-only.** Production dequantizes through [`Self::row_from`] into
    /// a caller-owned buffer instead: this signature allocates a fresh
    /// `Vec<f32>` per row, and its only two callers — the MLA absorb paths in
    /// `engine::arch::kda` and `engine::arch::glm` — were calling it from
    /// inside a per-token loop, re-dequantizing every weight row once per
    /// token. That measured 10.9% of a prefill profile. Both now hoist the
    /// dequantization out of the loop, which left this with no production
    /// caller at all; it survives as the reference the stride cross-check
    /// below compares `expert_matrix(e).row(r)` against.
    #[cfg(test)]
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

/// Where one shard's bytes live: the file to map, and how far into it the
/// shard's GGUF structure starts.
///
/// `offset` is `0` for every ordinary model — a `.gguf` file is a shard that
/// starts where its file does. It is non-zero only for a bundled
/// `orangu-server` (`crate::bundle`), where the shards are appended to the
/// executable one after another and each therefore begins somewhere in the
/// middle of the file being mapped.
struct ShardSource {
    path: std::path::PathBuf,
    offset: u64,
}

/// Every shard of the model at `path`, in shard order, as plain files —
/// [`shard_paths`] with the zero offsets an on-disk model always has.
fn shard_sources(path: &Path) -> Result<Vec<ShardSource>> {
    let gguf = GgufFile::open(path)?;
    Ok(shard_paths(path, &gguf)?
        .into_iter()
        .map(|path| ShardSource { path, offset: 0 })
        .collect())
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
pub(crate) fn shard_paths(path: &Path, gguf: &GgufFile) -> Result<Vec<std::path::PathBuf>> {
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

/// The block index in a `blk.<i>.<...>` tensor name, or `None` for a tensor
/// that belongs to no block (`token_embd`, `output_norm`, `output`).
///
/// Lives here rather than in `engine::plan` because both need it and this is
/// the module the other one already depends on: a plan reads it off a GGUF
/// tensor table, `resident_tensor_sizes` reads it off a loaded model, and the
/// two must agree about which block a tensor belongs to or their byte totals
/// will not.
pub(crate) fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
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

    /// Records which device each transformer layer will run on, so that
    /// every `QuantMatrix` built afterwards carries its device with it.
    ///
    /// Must be called *before* the model is constructed — an architecture's
    /// `load_with_backend` is where `matrix` is called, and a tensor
    /// fetched before this is stamped for device 0. `main` does it
    /// immediately after `select_backend`, which is the only place both the
    /// plan and the loaded weights exist at once.
    pub fn set_layer_devices(&mut self, layer_device: Vec<usize>) {
        self.layer_device = layer_device;
    }

    /// Records which experts a device tier holds, per `*_exps.weight`
    /// tensor. Must be called before the model is built, for the same
    /// reason [`Self::set_layer_devices`] must.
    pub fn set_expert_residency(&mut self, residency: HashMap<String, Arc<[bool]>>) {
        self.expert_residency = residency;
    }

    /// Every stacked per-expert tensor's name and expert count — what a
    /// residency plan has to be built over — **in name order**.
    ///
    /// The sort is load-bearing twice over, and `self.tensors` is a
    /// `HashMap`, so without it this returns hash order.
    ///
    /// *Reproducibility.* `engine::expert_tier::plan` breaks equal heat by
    /// index, and its own test pins that so "the same profile always
    /// produces the same tier". Indexes into a hash-ordered list defeat
    /// that: with no routing profile every expert has heat zero, the
    /// tie-break becomes the entire policy, and the tier changes shape from
    /// run to run. It changes *size* too, because a model's expert tensors
    /// are not all the same size — a fused `ffn_gate_up_exps` is twice a
    /// `ffn_down_exps` — so an order that happens to front-load the smaller
    /// one fits more experts into the same byte budget. Two identical starts
    /// reported 972 and 1100 of 7680 before this.
    ///
    /// *Granularity.* Name order groups one layer's expert tensors together,
    /// so a budget that holds a fraction of the model holds whole layers'
    /// experts rather than a scatter across every layer. That is the
    /// difference between some layers running their expert branch entirely
    /// on the device and every layer running most of it on the host — and
    /// with it, whether a device expert path is ever measured in the
    /// configuration that could win.
    pub fn expert_tensors(&self) -> Vec<(String, usize, u64)> {
        let mut tensors: Vec<(String, usize, u64)> = self
            .tensors
            .iter()
            .filter(|(name, _)| name.ends_with("_exps.weight"))
            .filter_map(|(name, loc)| {
                // `[in_dim, out_dim, n_expert]`; anything else is not a
                // stacked expert tensor whatever it is called.
                let n_expert = *loc.dims.get(2)? as usize;
                (n_expert > 0).then(|| (name.clone(), n_expert, (loc.len / n_expert) as u64))
            })
            .collect();
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        tensors
    }

    /// The device holding `name`'s layer.
    ///
    /// Everything outside a numbered `blk.<n>.` block — token embeddings,
    /// the output norm, `lm_head` — goes to device 0. Those are touched
    /// once per token at the very start and the very end of a forward pass,
    /// so placing them anywhere else would add two bus crossings per token
    /// to save nothing: they are a small fraction of a model's bytes, and
    /// the first device is the largest by construction (the set is ranked).
    pub fn device_for_tensor(&self, name: &str) -> usize {
        if self.layer_device.is_empty() {
            return 0;
        }
        let Some(layer) = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|digits| digits.parse::<usize>().ok())
        else {
            return 0;
        };
        // A layer number past the plan is a model whose `n_layer` and
        // tensor names disagree — not something to guess at, and device 0
        // is the answer that behaves exactly like an unsplit model.
        self.layer_device.get(layer).copied().unwrap_or(0)
    }

    /// Every tensor's name and stored byte length.
    ///
    /// The bytes as the file holds them, which is also the bytes a GPU
    /// backend uploads: nothing is dequantized on the way to the device
    /// (`VulkanBackend::weight_buffer` writes `QuantMatrix::raw_bytes`
    /// straight into its arena), so this is the model's device footprint
    /// and not an approximation of it.
    ///
    /// Reads the tensor *directory* only — the mapping is already open, so
    /// this touches no weight bytes and pages nothing in.
    pub fn tensor_sizes(&self) -> impl Iterator<Item = (&str, u64)> {
        self.tensors
            .iter()
            .map(|(name, loc)| (name.as_str(), loc.len as u64))
    }

    /// The first `blk.<n>` index belonging to a trailing multi-token-
    /// prediction block, or `None` when the file declares none — which is
    /// most files.
    ///
    /// `block_count` counts the MTP block, and the architectures that can
    /// load such a file (`glm`, `deepseek4`, `nemotron`, and the shared
    /// `qwen_hybrid` trunk) all stop before it, so its tensors are mapped and
    /// never read. See `engine::plan`'s `trunk_block_count`, which asks the
    /// same question of a GGUF nobody has opened.
    pub fn draft_block_start(&self) -> Option<usize> {
        let n_draft = self
            .metadata_u64("nextn_predict_layers")
            .filter(|&n| n > 0)?;
        self.metadata_u64("block_count")?
            .checked_sub(n_draft)
            .map(|n| n as usize)
    }

    /// [`tensor_sizes`](Self::tensor_sizes) without the tensors nothing ever
    /// loads — the trailing draft block.
    ///
    /// The right input for every memory question, and the reason it is a
    /// separate accessor rather than a change to `tensor_sizes`: those bytes
    /// really are in the file, so a caller asking what the *file* holds
    /// should still see them. A caller asking what a *run* holds should not.
    /// Counting them made `orangu-server` report `weights 21.30 GiB on
    /// device` for a model whose plan said 21.0 GiB, the 0.3 GiB difference
    /// being a block the forward pass never reaches.
    pub fn resident_tensor_sizes(&self) -> impl Iterator<Item = (&str, u64)> {
        let draft_start = self.draft_block_start();
        self.tensor_sizes().filter(move |(name, _)| {
            let Some(start) = draft_start else {
                return true;
            };
            block_index(name).is_none_or(|block| block < start)
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::open_shards(&shard_sources(path)?)
    }

    /// Loads a model whose shards are byte ranges inside one file that is
    /// not itself a `.gguf` — a bundled `orangu-server`, where the model was
    /// appended to the executable (see `crate::bundle`). Every shard names
    /// the same `path` and differs only in where it starts.
    ///
    /// Nothing downstream can tell the difference: the mapping is of the
    /// whole carrying file either way, and a tensor's `start` was always a
    /// byte offset into its shard's mapping rather than a file position of
    /// its own.
    pub fn open_bundled(path: &Path, offsets: &[u64]) -> Result<Self> {
        let sources: Vec<ShardSource> = offsets
            .iter()
            .map(|&offset| ShardSource {
                path: path.to_path_buf(),
                offset,
            })
            .collect();
        Self::open_shards(&sources)
    }

    fn open_shards(shards: &[ShardSource]) -> Result<Self> {
        let first = shards
            .first()
            .ok_or_else(|| anyhow!("a model needs at least one shard"))?;
        let gguf = GgufFile::open_at(&first.path, first.offset)?;
        let architecture = metadata_string(&gguf, "general.architecture")
            .ok_or_else(|| anyhow!("GGUF file is missing general.architecture"))?;
        resolve_arch_family(&architecture)?;

        let config = read_model_config(&gguf, &architecture)?;

        // Every shard's tensor directory, merged. A single-file model is
        // just the one-shard case of this.
        let mut tensors = HashMap::with_capacity(gguf.tensors.len());
        let mut total_tensors = 0usize;
        for (index, shard) in shards.iter().enumerate() {
            let shard_path = shard.path.as_path();
            // Shard 1's header is already parsed; the rest still need
            // reading. Only shard 1 carries the model's hyperparameters —
            // the others hold `split.*` and their own tensor directory.
            let shard_gguf = if index == 0 {
                None
            } else {
                Some(GgufFile::open_at(shard_path, shard.offset)?)
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
            // So `engine::page_cache` can measure how much of this shard is in
            // RAM, and evict it — the two halves of telling a cold run from a
            // warm one on a model too large to hold.
            super::page_cache::register_shard(shard_path, &mmap);

            for tensor in &shard_gguf.tensors {
                let element_count: u64 = tensor.dims.iter().product();
                let len = quant::tensor_byte_size(tensor.ggml_type, element_count)
                    .with_context(|| format!("tensor '{}'", tensor.name))?
                    as usize;
                // `data_offset` is relative to where this shard's GGUF
                // structure begins, which for a bundled model is not where
                // the mapped file begins — hence the segment's own offset on
                // top. Zero for a plain `.gguf`, where the two coincide.
                let start = shard.offset as usize
                    + shard_gguf.data_offset as usize
                    + tensor.offset as usize;
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
        //
        // Which is why each rewritten range is handed to
        // [`release_mapped_range`] afterwards: the owned copy and the mapped
        // original are otherwise both resident for the model's whole life, and
        // the mapping cannot simply be dropped because any tensor the file
        // stores *un*-repacked (`SmolLM2-Q4_0_4_4`'s `token_embd`, `Q8_0`)
        // still reads through it.
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
            // Held past the assignment below on purpose: the range being
            // released is an address in *this* mapping, so it has to outlive
            // the `loc.bytes` that pointed at it.
            let mapped = std::mem::replace(&mut loc.bytes, Arc::new(plain));
            release_mapped_range(&mapped[loc.start..loc.start + loc.len]);
            loc.ggml_type = base;
            loc.start = 0;
            loc.len = loc.bytes.len();
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
            // A model is single-device until `main` says otherwise; see
            // `Self::set_layer_devices`.
            layer_device: Vec::new(),
            expert_residency: HashMap::new(),
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
                // Signed integers are matched before `as_u64`, which
                // refuses negative values — and a negative whole number is
                // exactly what some of these keys hold
                // (`kimi-k3.kda.gate_lower_bound = -5`). Falling through to
                // `None` there would silently substitute a default.
                GgufValue::I8(i) => Some(*i as f32),
                GgufValue::I16(i) => Some(*i as f32),
                GgufValue::I32(i) => Some(*i as f32),
                GgufValue::I64(i) => Some(*i as f32),
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

    /// A `<arch>.<suffix>` array metadata value, each element widened to
    /// `f32`.
    pub fn metadata_array_f32(&self, suffix: &str) -> Option<Vec<f32>> {
        let key = format!("{}.{suffix}", self.config.architecture);
        self.metadata.iter().find_map(|(k, v)| {
            (*k == key).then_some(v).and_then(|v| match v {
                GgufValue::Array(items) => Some(
                    items
                        .iter()
                        .filter_map(|i| match i {
                            GgufValue::F32(f) => Some(*f),
                            GgufValue::F64(f) => Some(*f as f32),
                            other => other.as_u64().map(|u| u as f32),
                        })
                        .collect(),
                ),
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

    pub fn tensor_i32(&self, name: &str) -> Result<(Vec<i32>, &[u64])> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("model is missing tensor '{name}'"))?;
        anyhow::ensure!(
            loc.ggml_type == crate::engine::quant::GGML_TYPE_I32,
            "tensor '{name}' is not an I32 tensor"
        );
        let bytes = &loc.bytes[loc.start..loc.start + loc.len];
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
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
            device: self.device_for_tensor(name),
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
            residency: self.expert_residency.get(name).cloned(),
            name: Arc::from(name),
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

    /// The device seam: an expert viewed as a `QuantMatrix` must dequantize
    /// to exactly what reading it row by row gives, for *every* expert.
    ///
    /// The failure this guards is the one that would matter: an
    /// `expert_stride` mistake reads a neighbouring expert's weights, which
    /// is not a crash and not obviously wrong output — it is a model that
    /// quietly answers with the wrong expert. Checking every expert rather
    /// than one is what catches an off-by-one stride, which agrees on
    /// expert 0 by construction.
    #[test]
    fn an_expert_view_reads_the_same_weights_as_the_row_reader() {
        let stacked = test_expert_matrix(4, 6, 32);
        for expert in 0..stacked.n_expert {
            let view = stacked.expert_matrix(expert);
            assert_eq!(view.in_dim, stacked.in_dim);
            assert_eq!(view.out_dim, stacked.out_dim);
            for row in 0..stacked.out_dim {
                assert_eq!(
                    view.row(row),
                    stacked.row(expert, row),
                    "expert {expert} row {row}"
                );
            }
        }
    }

    /// Two experts' views must not share a cache key, or a GPU weight arena
    /// would upload one of them and serve it for both.
    #[test]
    fn expert_views_have_distinct_cache_keys() {
        let stacked = test_expert_matrix(4, 6, 32);
        let keys: std::collections::HashSet<_> = (0..stacked.n_expert)
            .map(|expert| stacked.expert_matrix(expert).cache_key())
            .collect();
        assert_eq!(keys.len(), stacked.n_expert);
    }

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

    /// The claim [`release_mapped_range`] rests on: `MADV_DONTNEED` over a
    /// read-only file mapping discards clean pages, so a range that *is* read
    /// again reads back exactly what it held. Nothing in the loader re-reads a
    /// released range — that is checked in `doc/perf/` by watching the
    /// mapping's `smaps` residency stay flat across a benchmark — but if that
    /// ever stopped being true, the failure has to be a slow re-fault and not
    /// wrong weights.
    ///
    /// Deliberately releases the *middle* of the mapping and asserts the
    /// untouched ends too, since the real caller releases one tensor's range
    /// out of many in the same file.
    #[test]
    fn releasing_a_mapped_range_does_not_change_what_it_reads_back() {
        use std::io::Write;

        // Several pages, so there is a whole page strictly inside the released
        // range after the inward rounding.
        let len = 64 * 1024;
        let want: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&want).expect("write");
        file.flush().expect("flush");

        let mapped = unsafe { Mmap::map(file.as_file()) }.expect("mmap");
        let mid = len / 4..len * 3 / 4;
        assert_eq!(&mapped[..], &want[..], "mapping differs before release");
        release_mapped_range(&mapped[mid.clone()]);
        assert_eq!(&mapped[mid.clone()], &want[mid], "released range changed");
        assert_eq!(&mapped[..], &want[..], "mapping changed outside the range");
    }

    /// A range shorter than a page, or one whose interior rounds away, must be
    /// a no-op rather than an `madvise` on a rounded-outward — and therefore
    /// not-ours — address. Reached in practice by a small `F32` tensor sharing
    /// a page with its neighbours.
    #[test]
    fn releasing_a_sub_page_range_is_a_harmless_no_op() {
        use std::io::Write;

        let want: Vec<u8> = (0..8192u32).map(|i| (i % 97) as u8).collect();
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&want).expect("write");
        file.flush().expect("flush");

        let mapped = unsafe { Mmap::map(file.as_file()) }.expect("mmap");
        // 64 bytes starting one byte into the first page: no whole page inside.
        release_mapped_range(&mapped[1..65]);
        assert_eq!(&mapped[..], &want[..]);
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
    fn resolve_arch_family_accepts_nemotron() {
        for arch in NEMOTRON_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::NemotronHMoe);
        }
    }

    /// The dense sibling now loads through the same module — it is the same
    /// trunk with a two-matrix FFN — but the unrelated older `nemotron` still
    /// must not, sharing a name prefix and none of the block structure.
    #[test]
    fn the_dense_sibling_loads_and_the_unrelated_nemotron_does_not() {
        assert_eq!(
            resolve_arch_family("nemotron_h").unwrap(),
            ArchFamily::NemotronHMoe
        );
        assert!(resolve_arch_family("nemotron").is_err());
    }

    #[test]
    fn resolve_arch_family_accepts_qwen3next() {
        for arch in QWEN3NEXT_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Qwen3Next);
        }
    }

    /// Every near miss refuses with **its own** reason, naming the family it
    /// is not.
    ///
    /// A bare "not supported" is what makes an out-of-tree alias mechanism
    /// look attractive, and an alias is the one answer here that produces
    /// wrong output rather than no output: these graphs differ in ways no
    /// structural check catches, because they have the tensors the similar
    /// architecture expects.
    #[test]
    fn a_recognised_near_miss_refuses_with_the_reason_rather_than_a_list() {
        for (name, _) in KNOWN_UNSUPPORTED {
            let err = resolve_arch_family(name).unwrap_err().to_string();
            assert!(err.contains(name), "{name}: {err}");
            assert!(
                err.contains("recognised but not supported"),
                "{name} fell through to the generic list: {err}"
            );
            // The message has to say what to do, not only what went wrong.
            assert!(err.contains("doc/ENGINE.md"), "{name}: {err}");
            assert!(
                err.contains("real checkpoint"),
                "{name} must say why it cannot simply be aliased: {err}"
            );
        }
    }

    /// **The test that keeps the table from going stale.** Adding support for
    /// one of these means deleting its entry; leaving both would make
    /// `resolve_arch_family` answer "supported" while the error text still
    /// explains why it is not.
    #[test]
    fn nothing_is_both_supported_and_listed_as_a_near_miss() {
        for (name, _) in KNOWN_UNSUPPORTED {
            assert!(
                resolve_arch_family(name).is_err(),
                "'{name}' is supported now — remove it from KNOWN_UNSUPPORTED"
            );
        }
    }

    /// A name nobody has heard of still gets the list. The near-miss table is
    /// an addition to that path, not a replacement for it.
    #[test]
    fn an_entirely_unknown_architecture_still_gets_the_supported_list() {
        let err = resolve_arch_family("not-a-real-architecture")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not yet supported"), "{err}");
        assert!(err.contains("llama"), "the list is still there: {err}");
    }

    #[test]
    fn resolve_arch_family_accepts_dflash() {
        for arch in DFLASH_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::DFlash);
        }
    }

    #[test]
    fn resolve_arch_family_accepts_deepseek4() {
        for arch in DEEPSEEK4_ARCHITECTURES {
            assert_eq!(resolve_arch_family(arch).unwrap(), ArchFamily::Deepseek4);
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
            let err = resolve_arch_family(arch).unwrap_err().to_string();
            // Rejected is the property; the wording is whichever refusal
            // applies. Both are recognised near misses, so both name their
            // own reason rather than falling through to the generic list —
            // see `KNOWN_UNSUPPORTED`.
            assert!(err.contains(arch), "{err}");
            assert!(err.contains("not supported"), "{err}");
            assert!(err.contains("phi3"), "names the family it is not: {err}");
        }
    }

    #[test]
    fn resolve_arch_family_rejects_unknown_architectures() {
        let err = resolve_arch_family("bert").unwrap_err();
        assert!(err.to_string().contains("not yet supported"), "{err}");
    }

    /// A complete, loadable single-tensor `llama` GGUF: the five
    /// hyperparameters `read_model_config` requires, and a `token_embd.
    /// weight` whose 32 `F32` values are `0.0, 1.0, 2.0, …` so a byte range
    /// that is off by anything at all is visible rather than merely wrong.
    fn minimal_llama_gguf() -> Vec<u8> {
        fn string(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        fn u32_kv(buf: &mut Vec<u8>, key: &str, value: u32) {
            string(buf, key);
            buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
            buf.extend_from_slice(&value.to_le_bytes());
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&6u64.to_le_bytes()); // metadata_kv_count

        string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        string(&mut buf, "llama");
        u32_kv(&mut buf, "llama.embedding_length", 8);
        u32_kv(&mut buf, "llama.block_count", 1);
        u32_kv(&mut buf, "llama.attention.head_count", 2);
        u32_kv(&mut buf, "llama.context_length", 16);
        u32_kv(&mut buf, "llama.vocab_size", 4);

        string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset within the data

        // Pad to the 32-byte alignment the reader computes `data_offset` at,
        // then the tensor data itself.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for value in 0..32u32 {
            buf.extend_from_slice(&(value as f32).to_le_bytes());
        }
        buf
    }

    /// The load path a bundled `orangu-server` takes has to produce exactly
    /// what the ordinary one does — same hyperparameters, same tensor, same
    /// bytes — from the same model sitting at a non-zero offset inside a
    /// larger file. This is the whole correctness claim of `crate::bundle`,
    /// and the one thing a wrong offset would silently turn into plausible
    /// nonsense at generation time rather than an error at load time.
    #[test]
    fn a_model_embedded_in_a_larger_file_loads_identically_to_one_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf = minimal_llama_gguf();

        let plain = dir.path().join("model.gguf");
        std::fs::write(&plain, &gguf).expect("write model");

        // What `bundle` writes: a program image, padding to a page boundary,
        // then the model — and trailing bytes after it, which nothing that
        // reads the model may notice.
        const OFFSET: u64 = 4096;
        let mut carrier = vec![0xAAu8; OFFSET as usize];
        carrier.extend_from_slice(&gguf);
        carrier.extend_from_slice(b"manifest and footer go here");
        let embedded = dir.path().join("orangu-server-bundle");
        std::fs::write(&embedded, &carrier).expect("write bundle");

        let from_disk = LoadedModel::open(&plain).expect("load from disk");
        let from_bundle =
            LoadedModel::open_bundled(&embedded, &[OFFSET]).expect("load from bundle");

        assert_eq!(from_disk.config.n_embd, from_bundle.config.n_embd);
        assert_eq!(from_disk.config.n_layer, from_bundle.config.n_layer);
        assert_eq!(from_disk.config.n_vocab, from_bundle.config.n_vocab);
        assert_eq!(from_disk.config.n_ctx_train, from_bundle.config.n_ctx_train);
        assert_eq!(from_disk.metadata.len(), from_bundle.metadata.len());

        let disk_tensor = &from_disk.tensors["token_embd.weight"];
        let bundle_tensor = &from_bundle.tensors["token_embd.weight"];
        assert_eq!(disk_tensor.len, bundle_tensor.len);
        assert_eq!(disk_tensor.dims, bundle_tensor.dims);
        // The offsets are deliberately *not* equal — the point is that the
        // bytes they land on are.
        assert_eq!(bundle_tensor.start, disk_tensor.start + OFFSET as usize);
        assert_eq!(
            &disk_tensor.bytes[disk_tensor.start..disk_tensor.start + disk_tensor.len],
            &bundle_tensor.bytes[bundle_tensor.start..bundle_tensor.start + bundle_tensor.len],
        );
        // And that they are the values that were written, not merely two
        // copies of the same wrong range.
        let first = f32::from_le_bytes(
            bundle_tensor.bytes[bundle_tensor.start..bundle_tensor.start + 4]
                .try_into()
                .expect("4 bytes"),
        );
        assert_eq!(first, 0.0);
        let last_start = bundle_tensor.start + bundle_tensor.len - 4;
        let last = f32::from_le_bytes(
            bundle_tensor.bytes[last_start..last_start + 4]
                .try_into()
                .expect("4 bytes"),
        );
        assert_eq!(last, 31.0);
    }

    /// A negative whole-numbered hyperparameter written as an integer
    /// must read back as itself, not as the caller's default:
    /// `as_u64` refuses negatives, so this is the one arm of
    /// `metadata_f32`'s integer fallback that a plain `.as_u64()` misses.
    #[test]
    fn metadata_f32_reads_negative_integers() {
        let model = LoadedModel {
            config: ModelConfig {
                architecture: "kimi-k3".to_string(),
                n_vocab: 0,
                n_embd: 0,
                n_layer: 0,
                n_head: 1,
                n_head_kv: 1,
                head_dim: 1,
                n_ctx_train: 0,
                rope_dim: 0,
                rope_freq_base: 0.0,
                rms_eps: 0.0,
                pooling_type: PoolingType::Mean,
            },
            metadata: vec![
                (
                    "kimi-k3.kda.gate_lower_bound".to_string(),
                    GgufValue::I32(-5),
                ),
                ("kimi-k3.situ_beta".to_string(), GgufValue::F32(4.0)),
                ("kimi-k3.positive".to_string(), GgufValue::U32(7)),
            ],
            tensors: HashMap::new(),
            layer_device: Vec::new(),
            expert_residency: HashMap::new(),
        };
        assert_eq!(model.metadata_f32("kda.gate_lower_bound"), Some(-5.0));
        assert_eq!(model.metadata_f32("situ_beta"), Some(4.0));
        assert_eq!(model.metadata_f32("positive"), Some(7.0));
        assert_eq!(model.metadata_f32("absent"), None);
    }

    /// A residency plan has to be reproducible and layer-granular, and
    /// `self.tensors` is a `HashMap`, so `expert_tensors` has to impose an
    /// order rather than inherit one.
    ///
    /// Both properties are checked because they fail differently and only
    /// one of them is loud. Non-reproducibility shows up as a tier that
    /// changes *size* between identical starts — `expert_tier::plan` fills a
    /// byte budget, and the two expert tensors here differ 2:1 per expert
    /// exactly as a fused `ffn_gate_up_exps` differs from a `ffn_down_exps`,
    /// so which one comes first changes how many experts fit. Scattering
    /// shows up as nothing at all: the tier is the right size and simply
    /// never gives any layer its whole expert set.
    #[test]
    fn expert_tensors_come_back_layer_grouped_and_in_a_stable_order() {
        let mut tensors = HashMap::new();
        // Inserted in an order that is neither sorted nor layer-grouped, and
        // interleaved with tensors that must not be selected at all.
        for name in [
            "blk.2.ffn_gate_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "output_norm.weight",
            "blk.1.ffn_gate_up_exps.weight",
            "blk.2.ffn_down_exps.weight",
            "blk.0.ffn_gate_up_exps.weight",
            "blk.1.attn_q.weight",
            "blk.1.ffn_down_exps.weight",
        ] {
            // `ffn_down_exps` half the per-expert bytes of `ffn_gate_up_exps`,
            // the shape that made hash order change the tier's size.
            let out_dim = if name.contains("gate_up") { 8 } else { 4 };
            let n_expert = if name.ends_with("_exps.weight") { 4 } else { 0 };
            let dims = if n_expert > 0 {
                vec![2, out_dim, n_expert as u64]
            } else {
                vec![2, out_dim]
            };
            tensors.insert(
                name.to_string(),
                TensorLocation {
                    ggml_type: crate::engine::quant::GGML_TYPE_F32,
                    dims,
                    start: 0,
                    len: 2 * out_dim as usize * n_expert.max(1) * 4,
                    bytes: Arc::new(Vec::<u8>::new()) as TensorBytes,
                },
            );
        }
        let model = LoadedModel {
            config: ModelConfig {
                architecture: "gemma4".to_string(),
                n_vocab: 0,
                n_embd: 0,
                n_layer: 3,
                n_head: 1,
                n_head_kv: 1,
                head_dim: 1,
                n_ctx_train: 0,
                rope_dim: 0,
                rope_freq_base: 0.0,
                rms_eps: 0.0,
                pooling_type: PoolingType::Mean,
            },
            metadata: Vec::new(),
            tensors,
            layer_device: Vec::new(),
            expert_residency: HashMap::new(),
        };

        let names: Vec<String> = model.expert_tensors().into_iter().map(|t| t.0).collect();
        assert_eq!(
            names,
            vec![
                "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.1.ffn_down_exps.weight",
                "blk.1.ffn_gate_up_exps.weight",
                "blk.2.ffn_down_exps.weight",
                "blk.2.ffn_gate_up_exps.weight",
            ],
            "expert tensors must come back sorted, so a tier is reproducible \
             and one layer's tensors are adjacent"
        );
        // Layer-grouped is the property a budget consumes, so state it
        // directly rather than leaving it implicit in the list above: a
        // prefix of any length splits at most one layer.
        for prefix in 0..=names.len() {
            let straddling = (0..3)
                .filter(|layer| {
                    let tag = format!("blk.{layer}.");
                    let held = names[..prefix]
                        .iter()
                        .filter(|n| n.starts_with(&tag))
                        .count();
                    held == 1
                })
                .count();
            assert!(
                straddling <= 1,
                "a budget holding {prefix} tensors split {straddling} layers"
            );
        }
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

    #[test]
    fn model_load_support_accepts_a_nemotron_model() {
        let (arch, bad_quant) =
            model_load_support(&header_only("nemotron_h_moe", &["blk.0.ssm_in.weight"]));
        assert_eq!(arch.as_deref(), Some("nemotron_h_moe"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_a_qwen3next_model() {
        let (arch, bad_quant) =
            model_load_support(&header_only("qwen3next", &["blk.0.ssm_beta_alpha.weight"]));
        assert_eq!(arch.as_deref(), Some("qwen3next"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_qwen3next_mxfp4_tensors() {
        let mut gguf = header_only("qwen3next", &["blk.0.ssm_beta_alpha.weight"]);
        gguf.tensors[0].ggml_type = crate::engine::quant::GGML_TYPE_MXFP4;
        let (arch, bad_quant) = model_load_support(&gguf);
        assert_eq!(arch.as_deref(), Some("qwen3next"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_dflash() {
        let (arch, bad_quant) = model_load_support(&header_only("dflash", &["fc.weight"]));
        assert_eq!(arch.as_deref(), Some("dflash"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_deepseek4_hash_routing_tables() {
        let mut gguf = header_only("deepseek4", &["blk.0.ffn_gate_tid2eid.weight"]);
        gguf.tensors[0].ggml_type = 26;
        let (arch, bad_quant) = model_load_support(&gguf);
        assert_eq!(arch.as_deref(), Some("deepseek4"));
        assert_eq!(bad_quant, None);
    }

    #[test]
    fn model_load_support_accepts_deepseek4_hash_routing_tables_without_arch_metadata() {
        let mut gguf = header_only("llama", &["blk.0.ffn_gate_tid2eid.weight"]);
        gguf.metadata.clear();
        gguf.tensors[0].ggml_type = 26;
        let (arch, bad_quant) = model_load_support(&gguf);
        assert_eq!(arch, None);
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
        let (arch, bad_quant) = model_load_support(&header_only("kimi-linear", &[]));
        assert_eq!(arch.as_deref(), Some("kimi-linear"));
        // Unreadable because of its *architecture*, not its tensor types.
        assert_eq!(bad_quant, None);
        assert!(resolve_arch_family("kimi-linear").is_err());
    }

    #[test]
    fn resolve_arch_family_accepts_kimi_k3() {
        assert_eq!(resolve_arch_family("kimi-k3").unwrap(), ArchFamily::KimiK3);
        // Kimi-Linear shares K3's delta-net attention but none of its
        // cross-layer residuals, latent MoE or situ activation, so it is
        // deliberately not routed here.
        assert!(resolve_arch_family("kimi-linear").is_err());
    }

    /// `inkling` is the text decoder. The multimodal projector shipped in
    /// the same repository is a `clip` model, and routing it here would
    /// promise a load that fails on the first missing tensor.
    #[test]
    fn resolve_arch_family_accepts_inkling_but_not_its_projector() {
        assert_eq!(resolve_arch_family("inkling").unwrap(), ArchFamily::Inkling);
        assert!(resolve_arch_family("clip").is_err());
    }

    #[test]
    fn resolve_arch_family_accepts_glm_dsa() {
        assert_eq!(resolve_arch_family("glm-dsa").unwrap(), ArchFamily::GlmDsa);
        // Plain GLM is a different architecture entirely, and not one this
        // build reads — `glm.rs` is the sparse-attention variant only.
        assert!(resolve_arch_family("glm4").is_err());
    }
}

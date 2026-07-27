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

//! The seam a GPU backend plugs into without touching the model code in
//! `engine::arch`: everything the forward pass needs from "the thing that
//! actually multiplies matrices" is this one trait.
//!
//! Implementors: `CpuBackend` (scalar with runtime AVX2 dispatch, always
//! available); `VulkanBackend` (compute shaders via `wgpu`, the most
//! mature GPU backend — real fused attention/RoPE/layer submissions,
//! verified against real AMD hardware, see its own module doc); `cuda`'s
//! `CudaBackend` and `opencl`'s `OpenClBackend` (both dlopen their vendor
//! library at runtime, same as `wgpu`, so both are always compiled in);
//! `rocm`'s `RocmBackend` (behind the `rocm` Cargo feature, off by
//! default — the one exception, since `cubecl-hip-sys` hard-links a vendor
//! library at *build* time, see that module's own doc comment for why).
//! `CudaBackend`/`OpenClBackend`/`RocmBackend` are each a real but
//! smaller-scoped `matmul`-only implementation — see their module docs for
//! exactly what's ported and what isn't.
//!
//! Earlier revisions of this file claimed AMD GPUs are reached only through
//! `VulkanBackend` (Mesa/RADV implements Vulkan on AMD hardware directly)
//! and that there was no separate ROCm/HIP backend. That's still true for
//! Vulkan/RADV as a *path* to AMD hardware — it's real, verified, and the
//! default `auto` backend selection still prefers it — but `rocm::
//! RocmBackend` now also exists as a genuine, separate HIP-based backend
//! for when it's specifically asked for (`backend = rocm`).

pub mod cpu;
pub mod cuda;
pub mod opencl;
#[cfg(feature = "rocm")]
pub mod rocm;
pub mod vulkan;
pub mod vulkan_replay;
pub mod vulkan_shaders;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use opencl::OpenClBackend;
#[cfg(feature = "rocm")]
pub use rocm::RocmBackend;
pub use vulkan::VulkanBackend;

use super::loader::QuantMatrix;

/// One `matmul` call's operands, for [`Backend::matmul_batch`] — a slice of
/// these describes several matmuls that don't depend on each other's
/// results (e.g. a transformer layer's Q/K/V projections, all reading the
/// same normed input) and so can be issued together.
pub struct MatmulOp<'a> {
    pub x: &'a [f32],
    pub n_tokens: usize,
    pub w: &'a QuantMatrix,
}

pub trait Backend: Send + Sync {
    /// `y[t, o] = sum_i x[t, i] * w.row(o)[i]` — `x` is `[n_tokens,
    /// w.in_dim]`, `y` is `[n_tokens, w.out_dim]`. `w`'s rows are
    /// dequantized on demand, not pre-materialized.
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32>;

    /// Runs several *independent* matmuls (no result of one feeds another
    /// — see [`MatmulOp`]) as a batch, returning results in the same
    /// order. The default implementation just calls `matmul` once per op;
    /// only a backend that actually benefits from batching (a GPU backend,
    /// which can submit one command buffer and block on it once instead of
    /// once per op) needs to override it. `CpuBackend` doesn't: its
    /// `matmul` is already parallelized internally and has no per-call
    /// dispatch/sync overhead to amortize.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        ops.iter()
            .map(|op| self.matmul(op.x, op.n_tokens, op.w))
            .collect()
    }

    /// The same product as [`Backend::matmul`], for a **decode** batch:
    /// `x`'s rows are one per *sequence* rather than one per position in a
    /// prompt.
    ///
    /// It exists because `n_tokens > 1` is not, on its own, a safe signal
    /// to switch kernels. `CpuBackend` picks its kernel from `n_tokens`,
    /// and the multi-token (GEMM) kernels are *not* bit-identical to the
    /// single-token (GEMV) one — they sum in a different order, and the
    /// K-quant GEMM quantizes activations one scale per 256 elements
    /// against the GEMV's one per 32. Both stay inside `engine::vecdot`'s
    /// error budget, but routing a decode batch through them would make a
    /// sequence's logits depend on how many *other* sequences happened to
    /// be decoding alongside it in that window — the same token, decoded
    /// alone or in a busy batch, would come out differently. Prefill has
    /// no such twin to agree with and keeps the faster GEMM.
    ///
    /// The default is `matmul` itself, which is right for every backend
    /// whose kernels don't vary with `n_tokens`.
    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.matmul(x, n_tokens, w)
    }

    /// [`Backend::matmul_batch`]'s decode counterpart — see
    /// [`Backend::matmul_decode`] for why decode needs its own entry point.
    /// Defaults to `matmul_batch` so a backend that overrode *that* for
    /// submission batching keeps it here too.
    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.matmul_batch(ops)
    }

    /// Downcast hook for the one GPU-specific fast path that doesn't fit
    /// this trait's backend-agnostic shape: `VulkanBackend::
    /// fused_post_attention` chains a whole gemma4 sub-layer's matmuls and
    /// elementwise/norm ops into a single GPU submission, which needs
    /// `VulkanBackend`'s own buffer-cache internals, not just `matmul`/
    /// `matmul_batch`. `CpuBackend` has no round-trip cost to amortize
    /// there, so it keeps the default `None` and callers fall back to the
    /// ordinary step-by-step path.
    fn as_vulkan(&self) -> Option<&VulkanBackend> {
        None
    }

    /// Whether this backend has a kernel for `ggml_type`.
    ///
    /// `CpuBackend` goes straight through `quant::dequantize`, so it covers
    /// everything `quant::supports_type` does and keeps this default. Every
    /// GPU backend has to compile a kernel per type and covers less —
    /// `VulkanBackend` lacks the three lowest-bit `IQ*` types,
    /// `CudaBackend`/`RocmBackend`/`OpenClBackend` lack all the
    /// codebook-indexed ones (see their `SUPPORTED_TYPES`).
    ///
    /// This exists so that gap is reported by
    /// [`unsupported_tensor_types`] as a startup error naming the type and
    /// the backend, instead of surfacing as a panic from inside `matmul`
    /// partway through the first request — which is what it did while every
    /// GPU backend was assumed to be at parity with the CPU path.
    fn supports_type(&self, _ggml_type: u32) -> bool {
        true
    }
}

/// Every distinct tensor type in `tensors` that `backend` has no kernel
/// for, as type names sorted for a stable message — empty when the backend
/// can run every tensor.
///
/// Takes `(name, ggml_type)` pairs rather than a `GgufFile` so the caller
/// can pass `LoadedModel::tensor_types`, which spans every shard; a split
/// model's `GgufFile` is only shard 1, and a type that appears solely in a
/// later shard would slip through. Reports the whole set rather than the
/// first hit because the pairs arrive in hash order, and because a mixed
/// low-bit file usually carries more than one type a backend lacks —
/// naming all of them means one restart, not one per type.
///
/// Reads the tensor *directory* only, never the data, so this is as cheap
/// as reading the header. Runs after `quant::supports_type`'s own check
/// (`loader::model_load_support`), so a type reported here is one this
/// build *can* read on the CPU but not on the selected device.
pub fn unsupported_tensor_types<'a>(
    tensors: impl Iterator<Item = (&'a str, u32)>,
    backend: &dyn Backend,
) -> Vec<String> {
    let mut names: Vec<String> = tensors
        .filter(|(_, ggml_type)| !backend.supports_type(*ggml_type))
        .map(|(_, ggml_type)| orangu::gguf::ggml_type_name(ggml_type))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL};

    /// A backend that accepts everything except the listed types, standing
    /// in for a real GPU backend's `SUPPORTED_TYPES` gap without needing a
    /// device.
    struct Picky<'a>(&'a [u32]);

    impl Backend for Picky<'_> {
        fn matmul(&self, _x: &[f32], _n_tokens: usize, _w: &QuantMatrix) -> Vec<f32> {
            unreachable!("supports_type is the only thing under test here")
        }
        fn supports_type(&self, ggml_type: u32) -> bool {
            !self.0.contains(&ggml_type)
        }
    }

    /// A backend missing two types must name both, deduped and sorted, and
    /// must not name the ones it does support. Sorted so the message doesn't
    /// change between runs — `LoadedModel::tensor_types` iterates a
    /// `HashMap`.
    #[test]
    fn unsupported_tensor_types_reports_every_missing_type_once() {
        let tensors = [
            ("output_norm.weight", GGML_TYPE_F32),
            ("blk.0.attn_k.weight", GGML_TYPE_IQ4_NL),
            ("blk.0.attn_q.weight", GGML_TYPE_IQ1_S),
            ("blk.1.attn_k.weight", GGML_TYPE_IQ4_NL),
        ];
        let found = unsupported_tensor_types(
            tensors.iter().copied(),
            &Picky(&[GGML_TYPE_IQ4_NL, GGML_TYPE_IQ1_S]),
        );
        assert_eq!(found, vec!["IQ1_S".to_string(), "IQ4_NL".to_string()]);
    }

    /// `CpuBackend` keeps the permissive default, so a file it can decode
    /// never trips this check — which is what keeps the startup gate from
    /// rejecting models that previously ran.
    #[test]
    fn unsupported_tensor_types_passes_a_backend_that_covers_everything() {
        let tensors = [
            ("blk.0.attn_k.weight", GGML_TYPE_IQ4_NL),
            ("blk.0.attn_q.weight", GGML_TYPE_IQ1_S),
        ];
        assert!(unsupported_tensor_types(tensors.iter().copied(), &CpuBackend).is_empty());
    }
}

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
}

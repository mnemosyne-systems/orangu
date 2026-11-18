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

//! A GPU `Backend` for Apple hardware, via `wgpu`'s Metal backend.
//!
//! # This is the Vulkan backend's kernels, on Metal
//!
//! Unlike `cuda`/`opencl`/`rocm` — each a separate, smaller-scoped
//! `matmul`-only implementation in its own kernel language — this backend
//! is not a reimplementation of anything. `engine::backend::vulkan` is
//! written entirely against portable `wgpu` and WGSL: every compute
//! pipeline, every kernel in `vulkan_shaders`, the weight/op/uniform
//! arenas, the fused decode and prefill submissions, split-k attention,
//! GPU sampling. None of that is Vulkan-specific — it only ever *ran* on
//! Vulkan because `VulkanBackend::try_init` asked `wgpu` for a Vulkan
//! adapter and nothing else.
//!
//! So `MetalBackend` asks for a Metal adapter instead
//! ([`VulkanBackend::try_init_backends`] with `wgpu::Backends::METAL`) and
//! wraps the result. `naga` translates the same WGSL to MSL rather than
//! SPIR-V, and Apple's Metal compiler takes it from there. What this buys
//! over the previous behaviour on macOS — which was `CpuBackend`, because
//! Apple ships no Vulkan driver and `try_init` therefore always returned
//! `None` — is the whole GPU path, not a matmul kernel: [`Self::as_wgpu`]
//! returns the inner backend, so every fused submission `engine::arch`
//! reaches for through that hook is live here too.
//!
//! # What Metal does and does not give the shared engine
//!
//! Bring-up is entirely feature-negotiated in `try_init_backends`, so this
//! module has no adapter logic of its own. For the record, on Apple
//! silicon:
//!
//! - `SHADER_F16` is present, so the `f16` KV cache (llama.cpp's own
//!   default) is on, same as on a Vulkan adapter that supports it.
//! - `SUBGROUP` is present (Metal SIMD-scoped reductions), so the
//!   cooperative-reduction attention kernel is on by default.
//! - `TIMESTAMP_QUERY` is present, so `ORANGU_GPU_TIMESTAMPS` works.
//! - `PIPELINE_CACHE` is not — `wgpu::util::pipeline_cache_key` returns
//!   `None` for anything but Vulkan, so no on-disk pipeline cache is built
//!   and cold start recompiles the kernels. Metal's own shader cache
//!   absorbs most of that cost.
//! - `PASSTHROUGH_SHADERS` *is* advertised, but on Metal it means MSL
//!   passthrough, not SPIR-V. `try_init_backends` gates the one
//!   passthrough kernel (`ORANGU_Q4K_GLSL`, a glslc-compiled Q4_K GEMV
//!   handed to RADV) on the API and not just the feature bit, so that path
//!   stays off here rather than feeding a Metal driver a SPIR-V blob.
//!
//! One kernel also needs a different *shape* here, for correctness rather
//! than speed: the tiled prefill GEMM holds its shared tiles as scalars
//! instead of `vec4`, because a component-wise store into a shared `vec4`
//! is not a 4-byte write on this API and four threads filling one vector
//! clobber each other. `vulkan_shaders::coop_vec4_tiles` decides that and
//! carries the measurement; `vulkan::tests::
//! coop_tiles_are_vec4_only_where_component_stores_work` holds it to it.
//!
//! The raw-Vulkan replay path (`ORANGU_REPLAY`, `vulkan_replay`) is Vulkan
//! by construction — it reaches through `as_hal` for an `ash::Device` —
//! and is already compiled out on Apple targets, reporting "no Vulkan
//! here" through the `None` it already returns for a non-Vulkan adapter.
//! Setting the variable on macOS is a no-op, not a failure.
//!
//! # Verification
//!
//! The per-`ggml_type` cross-checks against `CpuBackend` in `vulkan`'s own
//! test module are ordinary tests, not `#[ignore]`d ones, and they run
//! against whichever GPU `vulkan::shared_test_backend` finds — which is a
//! Metal device on Apple. They are the same tests that gate the Vulkan
//! path, so CI's macOS runner exercises this backend on every push rather
//! than skipping for want of a Vulkan adapter.

use super::vulkan::VulkanBackend;
use super::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;

/// The shared `wgpu` compute engine, brought up on Metal.
///
/// A newtype rather than a `VulkanBackend` handed straight to
/// `select_backend` so the two are distinct at the type level (a
/// `MetalBackend` in a backtrace or a `Debug` print is not a Vulkan one)
/// and so this module owns the Metal-specific bring-up and documentation.
/// It holds no state of its own: every method below forwards.
pub struct MetalBackend {
    inner: VulkanBackend,
}

impl MetalBackend {
    /// Looks for a usable Metal adapter and builds every quant type's
    /// compute pipeline up front. Returns `None` (never panics) if this
    /// platform has no Metal device or device/pipeline creation fails —
    /// callers fall back the same way they do for Vulkan.
    ///
    /// Non-Apple platforms have no Metal driver, so this is `None` there
    /// rather than being compiled out: `select_backend`'s error for an
    /// explicit `backend = metal` should say "no usable Metal device was
    /// found", which is true and actionable on every platform, rather than
    /// having `metal` be a config value that parses on Linux and then
    /// names a variant that does not exist.
    ///
    /// **Tests only** — see [`VulkanBackend::try_init`]. `select_backend`
    /// goes through [`Self::devices`] and [`Self::try_init_selected`] so the
    /// operator's `[orangu-server].device` is honoured and the device list
    /// is reported.
    #[cfg(test)]
    pub fn try_init() -> Option<Self> {
        let inner = VulkanBackend::try_init_backends(wgpu::Backends::METAL)?;
        // `wgpu` only ever hands back an adapter from the requested set, so
        // this cannot fire — but the whole reason the SPIR-V passthrough
        // kernel is now gated on the API rather than the feature bit is
        // that "which API is this really" stopped being obvious once one
        // type served two of them. Assert the invariant the rest of this
        // module's doc comment is written against.
        debug_assert_eq!(inner.wgpu_backend(), wgpu::Backend::Metal);
        Some(Self { inner })
    }

    /// Brings the engine up on one specific Metal device — see
    /// [`VulkanBackend::try_init_selected`], which this forwards to
    /// unchanged.
    ///
    /// Apple has shipped Macs with more than one GPU (the discrete/
    /// integrated pairs of the Intel era, and external GPUs over
    /// Thunderbolt), so `Backends::METAL` is not automatically a
    /// single-device list and the same selection policy applies here.
    pub fn try_init_selected(selected: &[usize]) -> Option<Self> {
        let inner = VulkanBackend::try_init_selected(wgpu::Backends::METAL, selected)?;
        debug_assert_eq!(inner.wgpu_backend(), wgpu::Backend::Metal);
        Some(Self { inner })
    }

    /// Every Metal device `wgpu` reports, in enumeration order.
    pub fn devices() -> Vec<super::device::DeviceCandidate> {
        VulkanBackend::devices(wgpu::Backends::METAL)
    }

    /// The adapter's own description (the Metal device name) — for the
    /// startup banner, e.g. `Metal/Apple M1 Pro (Metal)`.
    pub fn device_name(&self) -> &str {
        &self.inner.adapter_name
    }
}

impl Backend for MetalBackend {
    fn supports_type(&self, ggml_type: u32) -> bool {
        self.inner.supports_type(ggml_type)
    }

    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.inner.matmul(x, n_tokens, w)
    }

    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.inner.matmul_batch(ops)
    }

    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.inner.matmul_decode(x, n_tokens, w)
    }

    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.inner.matmul_batch_decode(ops)
    }

    /// The point of the newtype: hand callers the inner engine so every
    /// fused GPU path in `engine::arch` — which reaches for it through
    /// this hook — runs on Metal exactly as it does on Vulkan, instead of
    /// falling back to the step-by-step path a `None` would force.
    fn as_wgpu(&self) -> Option<&VulkanBackend> {
        Some(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_init` must answer, not panic or hang, on a machine with no
    /// Metal device — that is every CI runner except the macOS one, and it
    /// is the contract `select_backend`'s `auto` path depends on.
    ///
    /// On a machine that *does* have one, this additionally pins the two
    /// facts the rest of the module is written against: the adapter really
    /// is Metal, and [`MetalBackend::as_wgpu`] hands back the engine (so
    /// the fused paths are reached rather than skipped). The per-quant-type
    /// numerical cross-checks against `CpuBackend` live in `vulkan`'s test
    /// module and run against this same device — see this module's doc.
    #[test]
    fn try_init_answers_on_every_platform() {
        match MetalBackend::try_init() {
            Some(backend) => {
                assert!(!backend.device_name().is_empty());
                let inner = backend
                    .as_wgpu()
                    .expect("MetalBackend must expose the wgpu engine for the fused paths");
                assert_eq!(inner.wgpu_backend(), wgpu::Backend::Metal);
            }
            // Two `#[cfg]`'d arms rather than one `assert!(!cfg!(...))`:
            // the latter is a constant condition, which clippy rejects.
            #[cfg(target_vendor = "apple")]
            None => panic!(
                "an Apple target reported no Metal adapter — if this is real hardware \
                 rather than a build without a GPU, the Metal backend is broken"
            ),
            #[cfg(not(target_vendor = "apple"))]
            None => {}
        }
    }
}

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

//! CUDA backend, via `cudarc`'s driver-API + NVRTC bindings (dlopens
//! `libcuda.so`/`libnvrtc.so` at runtime — no CUDA toolkit needed to
//! *build* orangu-server, only to *run* it with GPU acceleration on a
//! machine with an NVIDIA driver installed). Structurally mirrors
//! `engine::backend::vulkan`: one dequantizing matmul kernel per
//! `ggml_type`, compiled once at [`CudaBackend::try_init`] time, weight
//! uploads cached by [`QuantMatrix::cache_key`] so a layer's weights are
//! uploaded to device memory once, not on every decode step.
//!
//! Scope: only [`Backend::matmul`] is implemented — the trait's actual
//! required surface, correct for every `n_tokens` (the kernel below is a
//! direct CUDA-C port of `vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` dispatch
//! strategy, not the cooperative/tiled variants `VulkanBackend` also has).
//! `VulkanBackend`'s much larger surface — GPU-resident attention, RoPE,
//! per-head RMSNorm, fused whole-layer submissions, GPU-side argmax
//! sampling, a disk pipeline cache — took real iteration against actual
//! AMD hardware to get right (long prompts, for example, were found to
//! reliably hang the GPU driver on real hardware — a bug only real
//! hardware testing surfaced); none of that exists here, and none of it can be
//! verified on a machine with no NVIDIA GPU. `CudaBackend::as_wgpu`
//! correctly returns `None` (the trait's default), so callers fall back to
//! the ordinary step-by-step path exactly like `CpuBackend` already does.
//! Not verified on real NVIDIA hardware — no such hardware was available
//! when this was built (confirmed via `nvidia-smi` on the dev machine);
//! correctness instead rests on the kernel math being a direct,
//! side-by-side port of `engine::quant::dequantize_*` (already verified
//! against real llama.cpp output) and the same CPU cross-check test
//! pattern `vulkan.rs` uses, which — like those tests — gracefully skips
//! here rather than failing when [`CudaBackend::try_init`] returns `None`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::engine::loader::QuantMatrix;

use super::device::{DeviceCandidate, DeviceClass};
use super::{Backend, MatmulOp};

// The kernels, the type list and the entry-point name are all
// `super::vendor_shaders`' - see that module for why they are shared
// rather than written once per backend.
use super::vendor_shaders::{KERNEL_NAME, SUPPORTED_TYPES};

/// The complete, compile-ready CUDA-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    super::vendor_shaders::kernel_source(super::vendor_shaders::Dialect::Cuda, ggml_type)
}

/// `QuantMatrix::cache_key()`'s return type — named, like `vulkan.rs`'s own
/// `WeightCacheKey`, so `weight_cache`'s type doesn't trip clippy's
/// `type_complexity` lint.
type WeightCacheKey = (usize, usize);

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    functions: HashMap<u32, CudaFunction>,
    weight_cache: Mutex<HashMap<WeightCacheKey, Arc<CudaSlice<u8>>>>,
    /// The `IQ*` codebooks, uploaded once at init — see
    /// `engine::iq_grids::packed`.
    iq_grids: CudaSlice<u32>,
    /// The device's own name (e.g. `"NVIDIA GeForce RTX 4090"`) — for the
    /// startup banner.
    pub device_name: String,
}

impl CudaBackend {
    /// Looks for a usable CUDA device (ordinal 0 — [`Self::try_init_index`]
    /// names another) and compiles every
    /// supported quant type's kernel via NVRTC up front. Returns `None`
    /// (never panics) if no CUDA driver is present, or compilation
    /// otherwise fails — callers fall back to `CpuBackend` in that case,
    /// the same contract `VulkanBackend::try_init` has.
    ///
    /// Unlike every other fallible step here, `cudarc` doesn't surface "no
    /// `libcuda.so`/`libnvrtc.so` found" as a `Result::Err` — it `panic!`s,
    /// from inside a lazy static its FFI wrappers all share, the first time
    /// *any* driver or NVRTC call is made (confirmed directly: this
    /// backend's own tests hit it on this project's dev machine, which has
    /// no NVIDIA driver installed). Since `cudarc` is an always-on default
    /// dependency (unlike `opencl3`/`cubecl-hip-sys`, which are opt-in
    /// features precisely because they can't degrade gracefully at *build*
    /// time), that panic would otherwise crash the whole server on startup
    /// for every non-NVIDIA machine using the default `auto` backend — the
    /// common case. `Self::try_init_inner` runs under `catch_unwind` with
    /// the panic hook silenced for the duration (so a normal "no CUDA GPU
    /// here" outcome doesn't also print a scary backtrace), turning that
    /// panic into the same graceful `None` every other missing-backend path
    /// already returns.
    ///
    /// **Tests only** — see `VulkanBackend::try_init`. `select_backend`
    /// goes through [`Self::devices`] and [`Self::try_init_index`] so the
    /// operator's `[orangu-server].device` is honoured and the device list
    /// is reported.
    #[cfg(test)]
    pub fn try_init() -> Option<Self> {
        Self::try_init_index(0)
    }

    /// [`Self::try_init`] against a specific CUDA ordinal, so a machine
    /// with several NVIDIA cards can be told which one to use rather than
    /// always getting ordinal 0.
    pub fn try_init_index(index: usize) -> Option<Self> {
        Self::guarded(|| Self::try_init_inner(index))
    }

    /// Every CUDA device this driver reports.
    ///
    /// Enumeration only — [`CudaContext::new`] is not called, so a machine
    /// with no NVIDIA driver answers with an empty list instead of paying
    /// context creation to find that out.
    ///
    /// Every device is [`DeviceClass::Discrete`]: CUDA does report an
    /// `INTEGRATED` attribute (Jetson, Grace-Blackwell), but reading it
    /// needs a context per device, which is the expensive thing this
    /// deliberately avoids. The distinction costs nothing here — an
    /// integrated NVIDIA part is never sharing the machine with a discrete
    /// one, so the class can't change the ranking between two devices this
    /// function returns. Size still can, and is read.
    pub fn devices() -> Vec<DeviceCandidate> {
        Self::guarded(|| {
            let count = CudaContext::device_count().ok()?.max(0) as usize;
            Some(
                (0..count)
                    .map(|index| DeviceCandidate {
                        index,
                        name: Self::device_name_at(index)
                            .unwrap_or_else(|| format!("CUDA device {index}")),
                        class: DeviceClass::Discrete,
                        vram_total_bytes: Self::device_memory_at(index),
                        id: None,
                        driver: None,
                    })
                    .collect(),
            )
        })
        .unwrap_or_default()
    }

    /// The device name at `index`, without holding on to the context.
    fn device_name_at(index: usize) -> Option<String> {
        CudaContext::new(index).ok()?.name().ok()
    }

    /// Total device memory at `index`, for the selection policy's size
    /// tie-break. `None` when the driver declines to say.
    fn device_memory_at(index: usize) -> Option<u64> {
        let device = cudarc::driver::result::device::get(index as i32).ok()?;
        // SAFETY: `device` is a valid `CUdevice` from the call above, and
        // `total_mem` only reads a property of it.
        let bytes = unsafe { cudarc::driver::result::device::total_mem(device) }.ok()?;
        (bytes > 0).then_some(bytes as u64)
    }

    /// Runs `f` with the panic hook silenced and unwinding caught — see
    /// [`Self::try_init`] for why every entry point into `cudarc` needs
    /// this and not just the one that builds a backend.
    fn guarded<T>(f: impl FnOnce() -> Option<T> + std::panic::UnwindSafe) -> Option<T> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(f);
        std::panic::set_hook(previous_hook);
        result.ok().flatten()
    }

    fn try_init_inner(index: usize) -> Option<Self> {
        let ctx = CudaContext::new(index).ok()?;
        let stream = ctx.default_stream();
        let device_name = ctx.name().unwrap_or_else(|_| "CUDA".to_string());

        let mut functions = HashMap::new();
        for &ggml_type in SUPPORTED_TYPES {
            let source = kernel_source(ggml_type)?;
            let ptx = cudarc::nvrtc::compile_ptx(&source).ok()?;
            let module = ctx.load_module(ptx).ok()?;
            let function = module.load_function(KERNEL_NAME).ok()?;
            functions.insert(ggml_type, function);
        }

        // The `IQ*` lattice codebooks, uploaded once and bound to every
        // launch. ~33 KiB, read-only, shared by every kernel — see
        // `vendor_shaders::iq_grid_prelude` for why every kernel takes the
        // pointer whether or not it reads one.
        let iq_grids = stream
            .clone_htod(&crate::engine::iq_grids::packed::words())
            .ok()?;

        Some(Self {
            stream,
            functions,
            weight_cache: Mutex::new(HashMap::new()),
            iq_grids,
            device_name,
        })
    }

    fn weight_buffer(&self, w: &QuantMatrix) -> Arc<CudaSlice<u8>> {
        let key = w.cache_key();
        if let Some(existing) = self
            .weight_cache
            .lock()
            .expect("cuda weight cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let uploaded = Arc::new(
            self.stream
                .clone_htod(w.raw_bytes())
                .expect("cuda weight upload failed"),
        );
        self.weight_cache
            .lock()
            .expect("cuda weight cache poisoned")
            .insert(key, uploaded.clone());
        uploaded
    }
}

impl Backend for CudaBackend {
    /// See [`Backend::reduced_surface`] — this backend implements
    /// [`Backend::matmul`] and nothing else.
    fn reduced_surface(&self) -> Option<&'static str> {
        Some(super::MATMUL_ONLY_SURFACE)
    }

    fn supports_type(&self, ggml_type: u32) -> bool {
        SUPPORTED_TYPES.contains(&ggml_type)
    }

    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let launched = self.launch(x, n_tokens, w);
        self.stream
            .clone_dtoh(&launched.y)
            .expect("cuda y readback failed")
    }

    /// Every op launched before any is read back, so the device runs one
    /// kernel while the host is still enqueuing the next.
    ///
    /// The default implementation is one `matmul` per op, and `matmul` ends
    /// in `clone_dtoh` — a device-to-*pageable*-host copy, which the CUDA
    /// driver completes before returning. A batch of `n` ops therefore cost
    /// `n` full round trips, with the device idle across each one, for a set
    /// of ops [`MatmulOp`] already guarantees are independent.
    ///
    /// Here the readbacks all happen after the launches, so only the *first*
    /// one waits on a kernel that is not already running. One synchronization
    /// point per chunk instead of one per op — see
    /// [`crate::engine::backend::plan_batch`] for what a chunk is and why
    /// there is more than one.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        use crate::engine::backend::{BATCH_DEVICE_BUDGET_BYTES, plan_batch};

        let (stripes, chunks) = plan_batch(ops, BATCH_DEVICE_BUDGET_BYTES);
        let mut out: Vec<Vec<f32>> = ops
            .iter()
            .map(|op| Vec::with_capacity(op.n_tokens * op.w.out_dim))
            .collect();
        for chunk in chunks {
            // Held as a whole until the chunk is drained. `x` is *not* the
            // kernel's output but it is its input, and on a context without
            // async allocation `CudaSlice::drop` synchronizes the stream
            // before freeing — dropping each upload right after its launch
            // would put back exactly the per-op wait this function exists to
            // remove, and on a context *with* async allocation it would free
            // in stream order and be merely wasteful. Keeping both buffers
            // alive to the end of the chunk is the one shape that is right
            // under either.
            let launched: Vec<(usize, Launched)> = stripes[chunk]
                .iter()
                .map(|stripe| {
                    let op = &ops[stripe.op];
                    let in_dim = op.w.in_dim;
                    let rows = stripe.start * in_dim..(stripe.start + stripe.n_tokens) * in_dim;
                    (stripe.op, self.launch(&op.x[rows], stripe.n_tokens, op.w))
                })
                .collect();
            for (op, launched) in &launched {
                out[*op].extend(
                    self.stream
                        .clone_dtoh(&launched.y)
                        .expect("cuda y readback failed"),
                );
            }
        }
        out
    }
}

/// One in-flight `matmul`: the uploaded activations and the output buffer the
/// kernel writes. Both are returned because both must outlive the launch —
/// see [`CudaBackend::matmul_batch`].
struct Launched {
    #[allow(
        dead_code,
        reason = "kept alive until the kernel that reads it has run"
    )]
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
}

impl CudaBackend {
    /// Uploads `x`, allocates the output and launches the kernel — every step
    /// of a `matmul` except the readback, which is what a batch defers.
    fn launch(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Launched {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let row_bytes = w.row_bytes() as u32;
        let weights = self.weight_buffer(w);
        let x_dev = self.stream.clone_htod(x).expect("cuda x upload failed");
        let mut y_dev = self
            .stream
            .alloc_zeros::<f32>(n_tokens * out_dim)
            .expect("cuda y alloc failed");

        let function = self.functions.get(&w.ggml_type()).unwrap_or_else(|| {
            panic!(
                "ggml_type {} reached CudaBackend::matmul without a compiled kernel \
                 (QuantMatrix construction should have rejected it earlier)",
                w.ggml_type()
            )
        });

        let n_row_groups = out_dim.div_ceil(4);
        let num_blocks = (n_row_groups * n_tokens).max(1) as u32;
        let in_dim_u32 = in_dim as u32;
        let out_dim_u32 = out_dim as u32;
        let n_tokens_u32 = n_tokens as u32;

        let mut builder = self.stream.launch_builder(function);
        builder.arg(&*weights);
        builder.arg(&x_dev);
        builder.arg(&mut y_dev);
        builder.arg(&in_dim_u32);
        builder.arg(&out_dim_u32);
        builder.arg(&n_tokens_u32);
        builder.arg(&row_bytes);
        builder.arg(&self.iq_grids);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.expect("cuda kernel launch failed");

        Launched { x: x_dev, y: y_dev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{
        GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    };

    /// One `CudaBackend`, lazily built and shared across every test in this
    /// module — same rationale as `vulkan::tests::shared_vulkan`: creating
    /// a CUDA context per test would be wasteful even where one exists, and
    /// on every machine this project was developed/tested on (confirmed via
    /// `nvidia-smi`: no NVIDIA GPU present), `try_init()` returns `None` and
    /// every test below skips via `let Some(cuda) = shared_cuda() else {
    /// return; }` — the same graceful-skip convention `vulkan.rs`'s own
    /// tests use, not a failure.
    fn shared_cuda() -> Option<&'static CudaBackend> {
        static CUDA: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        CUDA.get_or_init(CudaBackend::try_init).as_ref()
    }

    fn next_byte(seed: &mut u64) -> u8 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed & 0xFF) as u8
    }

    fn next_bytes(seed: &mut u64, n: usize) -> Vec<u8> {
        (0..n).map(|_| next_byte(seed)).collect()
    }

    /// Straight from `engine::quant`, which is where a block layout is
    /// defined. This module used to keep its own copy of that table; so did
    /// the other two vendor backends, and a type added to `quant` and to
    /// `vendor_shaders` but not to all three copies would fail here as
    /// `unreachable!()` rather than as a missing kernel.
    fn block_bytes_for(ggml_type: u32) -> usize {
        crate::engine::quant::block_layout(ggml_type)
            .expect("a type under cross-check has a block layout")
            .0
    }

    fn block_elems_for(ggml_type: u32) -> usize {
        crate::engine::quant::block_layout(ggml_type)
            .expect("a type under cross-check has a block layout")
            .1
    }

    /// Cross-checks `CudaBackend::matmul` against `CpuBackend::matmul` for
    /// every supported `ggml_type`, on randomized (but reproducible — fixed
    /// seed) quantized weight bytes — the exact same methodology `vulkan
    /// .rs`'s `cross_check`/`cross_check_n_tokens` use, so both backends'
    /// kernels are held to the same bar. Skips (doesn't fail) when no CUDA
    /// device is available, per `shared_cuda`'s doc comment.
    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(cuda) = shared_cuda() else {
            return;
        };
        let block_bytes = block_bytes_for(ggml_type);
        let block_elems = block_elems_for(ggml_type);
        assert!(in_dim.is_multiple_of(block_elems));
        let row_bytes = (in_dim / block_elems) * block_bytes;
        let mut seed = 0x1234_5678_9abc_def0u64
            ^ (ggml_type as u64) << 32
            ^ (in_dim as u64) << 16
            ^ out_dim as u64;
        let bytes = next_bytes(&mut seed, row_bytes * out_dim);
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        let expected = CpuBackend.matmul_dequant(&x, n_tokens, &w);
        let actual = cuda.matmul(&x, n_tokens, &w);
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-2 * e.abs().max(1.0),
                "index {i}: expected {e}, got {a} (ggml_type {ggml_type}, n_tokens {n_tokens})"
            );
        }
    }

    #[test]
    fn matmul_matches_cpu_backend_for_f32() {
        cross_check(GGML_TYPE_F32, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_f16() {
        cross_check(GGML_TYPE_F16, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_bf16() {
        cross_check(GGML_TYPE_BF16, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q4_0() {
        cross_check(GGML_TYPE_Q4_0, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q5_0() {
        cross_check(GGML_TYPE_Q5_0, 64, 6, 1);
    }
    /// The two affine block quants this engine could always read on the CPU
    /// and could not run on this device — no codebook, no extra binding,
    /// simply never written. See `PARITY.md` C3.
    #[test]
    fn matmul_matches_cpu_backend_for_q4_1() {
        cross_check(GGML_TYPE_Q4_1, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q5_1() {
        cross_check(GGML_TYPE_Q5_1, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q8_0() {
        cross_check(GGML_TYPE_Q8_0, 64, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q4_k() {
        cross_check(GGML_TYPE_Q4_K, 256, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q5_k() {
        cross_check(GGML_TYPE_Q5_K, 256, 6, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_q6_k() {
        cross_check(GGML_TYPE_Q6_K, 256, 6, 1);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q2_k() {
        cross_check(GGML_TYPE_Q2_K, 256, 6, 1);
    }

    #[test]
    fn matmul_matches_cpu_backend_for_q3_k() {
        cross_check(GGML_TYPE_Q3_K, 256, 6, 1);
    }

    /// `in_dim = 896` because that is the width that makes `IQ4_NL` appear
    /// in the first place: a K-quant needs 256 | `in_dim`, so upstream
    /// substitutes `IQ4_NL` on rows like this one. A 256-wide check would
    /// not distinguish a correct block stride from one that assumed `QK_K`.
    /// The eight `IQ*` types this device could not take at all until the
    /// lattice codebooks were uploaded — a capability gap, not a speed one:
    /// `engine::backend::unsupported_tensor_types` refused the whole GPU for
    /// a file carrying any of them. See `PARITY.md` C3.
    #[test]
    fn matmul_matches_cpu_backend_for_iq1_s() {
        cross_check(GGML_TYPE_IQ1_S, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq1_m() {
        cross_check(GGML_TYPE_IQ1_M, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq2_xxs() {
        cross_check(GGML_TYPE_IQ2_XXS, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq2_xs() {
        cross_check(GGML_TYPE_IQ2_XS, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq2_s() {
        cross_check(GGML_TYPE_IQ2_S, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq3_xxs() {
        cross_check(GGML_TYPE_IQ3_XXS, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq3_s() {
        cross_check(GGML_TYPE_IQ3_S, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq4_xs() {
        cross_check(GGML_TYPE_IQ4_XS, 256, 4, 1);
    }
    #[test]
    fn matmul_matches_cpu_backend_for_iq4_nl() {
        cross_check(GGML_TYPE_IQ4_NL, 896, 6, 1);
    }

    #[test]
    fn matmul_handles_multiple_tokens() {
        cross_check(GGML_TYPE_Q4_K, 256, 9, 5);
    }

    #[test]
    fn matmul_batch_matches_sequential_cpu_matmuls() {
        let Some(cuda) = shared_cuda() else {
            return;
        };
        let mut seed = 42u64;
        let bytes_a = next_bytes(&mut seed, 144 * 8);
        let wa = test_quant_matrix(&bytes_a, GGML_TYPE_Q4_K, 256, 8);
        let bytes_b = next_bytes(&mut seed, 4 * 5);
        let wb = test_quant_matrix(&bytes_b, GGML_TYPE_F32, 5, 1);
        let xa: Vec<f32> = (0..256).map(|i| (i % 7) as f32 * 0.05).collect();
        let xb: Vec<f32> = (0..5).map(|i| (i % 3) as f32 * 0.2).collect();

        let ops = [
            MatmulOp {
                x: &xa,
                n_tokens: 1,
                w: &wa,
            },
            MatmulOp {
                x: &xb,
                n_tokens: 1,
                w: &wb,
            },
        ];
        let batched = cuda.matmul_batch(&ops);
        let expected_a = cuda.matmul(&xa, 1, &wa);
        let expected_b = cuda.matmul(&xb, 1, &wb);
        assert_eq!(batched[0], expected_a);
        assert_eq!(batched[1], expected_b);
    }
}

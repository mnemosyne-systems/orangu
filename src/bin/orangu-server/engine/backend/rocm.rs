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

//! AMD ROCm/HIP backend, via `cubecl-hip-sys` (raw bindgen'd HIP + HIPRTC
//! bindings from the Burn/CubeCL project — chosen over several much
//! lower-traffic "rocm"-named crates found on crates.io that looked, on
//! inspection, like they may have been created to be picked up by exactly
//! this kind of dependency search: all published within the last few
//! months, single-digit-to-low-hundreds downloads, no real history —
//! `cubecl-hip-sys` has ~900K downloads, a real GitHub org, and active
//! maintenance). Structurally the same scope as `engine::backend::cuda`'s
//! `CudaBackend`/`engine::backend::opencl`'s `OpenClBackend` — see the
//! `cuda` module doc for what's implemented ([`Backend::matmul`] only) and
//! what isn't (`VulkanBackend`'s much larger fused/GPU-resident surface).
//! Not verified on real ROCm hardware — no AMD GPU with ROCm installed was
//! available when this was built.
//!
//! `cubecl-hip-sys` has no safe wrapper (unlike `cudarc`/`opencl3`, this is
//! the raw FFI layer directly), so every call here is genuinely `unsafe`
//! and there's no crate-provided guidance on thread safety. Rather than
//! guess at HIP's actual concurrency guarantees with no hardware to verify
//! against, [`RocmBackend`] takes the simplest safe-by-construction
//! approach: one [`std::sync::Mutex`] around the *entire* per-call device
//! interaction (allocate, upload, launch, read back), so at most one
//! thread ever touches the HIP runtime at a time. This gives up the
//! `engine::scheduler`'s cross-slot GPU concurrency `VulkanBackend`/
//! `CudaBackend`/`OpenClBackend` allow (their weight/op caches are each
//! individually lockable), in exchange for not shipping unverified
//! assumptions about an FFI surface this project has no way to test.
//!
//! **Behind the `rocm` Cargo feature (off by default)**, for the same
//! reason `engine::backend::opencl` is: `cubecl-hip-sys`'s build script
//! links `-lamdhip64 -lhiprtc` at build time whenever it finds a ROCm
//! install via `hipconfig` — fine on a machine with ROCm, but breaks a
//! plain `cargo build` everywhere else. See that module's doc comment for
//! the fuller explanation (this backend has the identical constraint).

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::ptr;
use std::sync::Mutex;

use cubecl_hip_sys::{
    HIP_SUCCESS, hipDeviceGetName, hipDeviceSynchronize, hipDeviceTotalMem, hipFree, hipFunction_t,
    hipGetDeviceCount, hipInit, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipSetDevice, hipStream_t, hipStreamCreate, hiprtcCompileProgram,
    hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode, hiprtcGetCodeSize,
    hiprtcGetProgramLog, hiprtcGetProgramLogSize, hiprtcProgram, hiprtcResult_HIPRTC_SUCCESS,
};

use crate::engine::loader::QuantMatrix;

use super::device::{DeviceCandidate, DeviceClass};
use super::{Backend, MatmulOp};

// The kernels, the type list and the entry-point name are all
// `super::vendor_shaders`' - see that module for why they are shared
// rather than written once per backend.
use super::vendor_shaders::{KERNEL_NAME, SUPPORTED_TYPES};

/// The complete, compile-ready HIP-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    super::vendor_shaders::kernel_source(super::vendor_shaders::Dialect::Hip, ggml_type)
}

unsafe fn compile_and_load(source: &str) -> Option<(hipModule_t, hipFunction_t)> {
    unsafe {
        let c_source = CString::new(source).ok()?;
        let mut program: hiprtcProgram = ptr::null_mut();
        if hiprtcCreateProgram(
            &mut program,
            c_source.as_ptr(),
            ptr::null(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        ) != hiprtcResult_HIPRTC_SUCCESS
        {
            return None;
        }

        let compile_status = hiprtcCompileProgram(program, 0, ptr::null_mut());
        if compile_status != hiprtcResult_HIPRTC_SUCCESS {
            let mut log_size: usize = 0;
            if hiprtcGetProgramLogSize(program, &mut log_size) == hiprtcResult_HIPRTC_SUCCESS
                && log_size > 0
            {
                let mut log_buffer = vec![0i8; log_size];
                if hiprtcGetProgramLog(program, log_buffer.as_mut_ptr())
                    == hiprtcResult_HIPRTC_SUCCESS
                {
                    let log = CStr::from_ptr(log_buffer.as_ptr());
                    eprintln!(
                        "orangu-server: HIPRTC compile error: {}",
                        log.to_string_lossy()
                    );
                }
            }
            hiprtcDestroyProgram(&mut program);
            return None;
        }

        let mut code_size: usize = 0;
        if hiprtcGetCodeSize(program, &mut code_size) != hiprtcResult_HIPRTC_SUCCESS {
            hiprtcDestroyProgram(&mut program);
            return None;
        }
        let mut code: Vec<u8> = vec![0; code_size];
        if hiprtcGetCode(program, code.as_mut_ptr() as *mut _) != hiprtcResult_HIPRTC_SUCCESS {
            hiprtcDestroyProgram(&mut program);
            return None;
        }
        hiprtcDestroyProgram(&mut program);

        let mut module: hipModule_t = ptr::null_mut();
        if hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) != HIP_SUCCESS {
            return None;
        }
        let func_name = CString::new(KERNEL_NAME).ok()?;
        let mut function: hipFunction_t = ptr::null_mut();
        if hipModuleGetFunction(&mut function, module, func_name.as_ptr()) != HIP_SUCCESS {
            return None;
        }
        Some((module, function))
    }
}

/// Everything the HIP runtime needs, behind one lock — see the module doc
/// comment for why this backend serializes *all* device interaction rather
/// than caching per-op resources the way `VulkanBackend`/`CudaBackend`/
/// `OpenClBackend` do.
struct RocmState {
    stream: hipStream_t,
    functions: HashMap<u32, (hipModule_t, hipFunction_t)>,
    weight_buffers: HashMap<(usize, usize), (*mut c_void, usize)>,
    /// The `IQ*` codebooks, uploaded once at init — see
    /// `engine::iq_grids::packed`.
    iq_grids: *mut c_void,
}

// Raw HIP handles (`*mut c_void`-shaped opaque pointers) carry no thread
// affinity of their own in the HIP API — only concurrent *use* needs
// synchronizing, which `RocmBackend::state`'s `Mutex` already provides.
unsafe impl Send for RocmState {}

pub struct RocmBackend {
    state: Mutex<RocmState>,
    /// The device's own name (e.g. `"AMD Radeon RX 7900 XTX"`) — for the
    /// startup banner.
    pub device_name: String,
}

impl RocmBackend {
    /// Looks for a usable HIP device (ordinal 0 — [`Self::try_init_index`]
    /// names another) and compiles every
    /// supported quant type's kernel via HIPRTC up front. Returns `None`
    /// (never panics) if no HIP runtime/device is present, or compilation
    /// otherwise fails — callers fall back to `CpuBackend`, the same
    /// contract every other backend's `try_init` has.
    ///
    /// **Tests only** — see `VulkanBackend::try_init`. `select_backend`
    /// goes through [`Self::devices`] and [`Self::try_init_index`] so the
    /// operator's `[orangu-server].device` is honoured and the device list
    /// is reported.
    #[cfg(test)]
    pub fn try_init() -> Option<Self> {
        Self::try_init_index(0)
    }

    /// Every HIP device this runtime reports.
    ///
    /// Every device is [`DeviceClass::Discrete`]. HIP does expose an
    /// `integrated` flag, but only through `hipGetDevicePropertiesR0600`,
    /// whose struct layout differs across the `cubecl-hip-sys` binding sets
    /// this can be built against — an FFI shape this project has no ROCm
    /// machine to verify. Since a discrete-vs-integrated distinction can
    /// only re-rank devices *within* this list, and ROCm on an APU beside a
    /// discrete Radeon is not a configuration anyone has reported, the flag
    /// is not read rather than read unverifiably. Size is, and is what
    /// actually orders a multi-card box.
    ///
    /// The name query needs no context, so an empty list here really does
    /// mean "no HIP device", not "not asked yet".
    pub fn devices() -> Vec<DeviceCandidate> {
        unsafe {
            if hipInit(0) != HIP_SUCCESS {
                return Vec::new();
            }
            let mut count: std::os::raw::c_int = 0;
            if hipGetDeviceCount(&mut count) != HIP_SUCCESS || count <= 0 {
                return Vec::new();
            }
            (0..count)
                .map(|ordinal| DeviceCandidate {
                    index: ordinal as usize,
                    name: Self::device_name_at(ordinal)
                        .unwrap_or_else(|| format!("HIP device {ordinal}")),
                    class: DeviceClass::Discrete,
                    vram_total_bytes: Self::device_memory_at(ordinal),
                    id: None,
                    driver: None,
                })
                .collect()
        }
    }

    unsafe fn device_name_at(ordinal: std::os::raw::c_int) -> Option<String> {
        let mut name_buf = [0i8; 256];
        unsafe {
            (hipDeviceGetName(name_buf.as_mut_ptr(), 256, ordinal) == HIP_SUCCESS).then(|| {
                CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            })
        }
    }

    unsafe fn device_memory_at(ordinal: std::os::raw::c_int) -> Option<u64> {
        let mut bytes: usize = 0;
        unsafe {
            (hipDeviceTotalMem(&mut bytes, ordinal) == HIP_SUCCESS && bytes > 0)
                .then_some(bytes as u64)
        }
    }

    /// [`Self::try_init`] against a specific HIP ordinal.
    pub fn try_init_index(index: usize) -> Option<Self> {
        unsafe {
            if hipInit(0) != HIP_SUCCESS {
                return None;
            }
            let mut count: std::os::raw::c_int = 0;
            if hipGetDeviceCount(&mut count) != HIP_SUCCESS || count == 0 {
                return None;
            }
            let ordinal = std::os::raw::c_int::try_from(index).ok()?;
            if ordinal >= count {
                return None;
            }
            // Every HIP call below is thread-local to the selected device,
            // which is what makes an ordinal other than 0 reachable at all:
            // the kernels, the stream and the weight buffers all land on
            // whichever device this call named.
            if hipSetDevice(ordinal) != HIP_SUCCESS {
                return None;
            }

            let device_name = Self::device_name_at(ordinal).unwrap_or_else(|| "ROCm".to_string());

            let mut stream: hipStream_t = ptr::null_mut();
            if hipStreamCreate(&mut stream) != HIP_SUCCESS {
                return None;
            }

            let mut functions = HashMap::new();
            for &ggml_type in SUPPORTED_TYPES {
                let source = kernel_source(ggml_type)?;
                let (module, function) = compile_and_load(&source)?;
                functions.insert(ggml_type, (module, function));
            }

            // The `IQ*` lattice codebooks, uploaded once and bound to every
            // launch — see `vendor_shaders::iq_grid_prelude` for why every
            // kernel takes the pointer whether or not it reads one.
            let grid_words = crate::engine::iq_grids::packed::words();
            let grid_bytes = std::mem::size_of_val(grid_words.as_slice());
            let mut iq_grids: *mut c_void = ptr::null_mut();
            if hipMalloc(&mut iq_grids, grid_bytes) != HIP_SUCCESS {
                return None;
            }
            if hipMemcpy(
                iq_grids,
                grid_words.as_ptr() as *const c_void,
                grid_bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
            ) != HIP_SUCCESS
            {
                return None;
            }

            Some(Self {
                state: Mutex::new(RocmState {
                    stream,
                    functions,
                    weight_buffers: HashMap::new(),
                    iq_grids,
                }),
                device_name,
            })
        }
    }
}

impl Backend for RocmBackend {
    /// See [`Backend::reduced_surface`] — this backend implements
    /// [`Backend::matmul`] and nothing else.
    fn reduced_surface(&self) -> Option<&'static str> {
        Some(super::MATMUL_ONLY_SURFACE)
    }

    fn supports_type(&self, ggml_type: u32) -> bool {
        SUPPORTED_TYPES.contains(&ggml_type)
    }

    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        let row_bytes = w.row_bytes();
        let bytes = w.raw_bytes();
        let key = w.cache_key();
        let y_len = n_tokens * out_dim;

        let mut state = self.state.lock().expect("rocm state poisoned");
        unsafe {
            let weights_ptr = if let Some(&(ptr, _len)) = state.weight_buffers.get(&key) {
                ptr
            } else {
                let mut device_ptr: *mut c_void = ptr::null_mut();
                assert_eq!(
                    hipMalloc(&mut device_ptr, bytes.len().max(1)),
                    HIP_SUCCESS,
                    "rocm weight buffer allocation failed"
                );
                assert_eq!(
                    hipMemcpy(
                        device_ptr,
                        bytes.as_ptr() as *const c_void,
                        bytes.len(),
                        hipMemcpyKind_hipMemcpyHostToDevice,
                    ),
                    HIP_SUCCESS,
                    "rocm weight upload failed"
                );
                state.weight_buffers.insert(key, (device_ptr, bytes.len()));
                device_ptr
            };

            let x_bytes = std::mem::size_of_val(x);
            let mut x_ptr: *mut c_void = ptr::null_mut();
            assert_eq!(
                hipMalloc(&mut x_ptr, x_bytes.max(1)),
                HIP_SUCCESS,
                "rocm x allocation failed"
            );
            assert_eq!(
                hipMemcpy(
                    x_ptr,
                    x.as_ptr() as *const c_void,
                    x_bytes,
                    hipMemcpyKind_hipMemcpyHostToDevice,
                ),
                HIP_SUCCESS,
                "rocm x upload failed"
            );

            let y_bytes = y_len * std::mem::size_of::<f32>();
            let mut y_ptr: *mut c_void = ptr::null_mut();
            assert_eq!(
                hipMalloc(&mut y_ptr, y_bytes.max(1)),
                HIP_SUCCESS,
                "rocm y allocation failed"
            );

            let &(_module, function) = state.functions.get(&w.ggml_type()).unwrap_or_else(|| {
                panic!(
                    "ggml_type {} reached RocmBackend::matmul without a compiled kernel \
                     (QuantMatrix construction should have rejected it earlier)",
                    w.ggml_type()
                )
            });

            let in_dim_u32 = in_dim as u32;
            let out_dim_u32 = out_dim as u32;
            let n_tokens_u32 = n_tokens as u32;
            let row_bytes_u32 = row_bytes as u32;
            let grids_ptr = state.iq_grids;
            let n_row_groups = out_dim.div_ceil(4);
            let num_blocks = (n_row_groups * n_tokens).max(1) as u32;

            let mut args: [*mut c_void; 8] = [
                &weights_ptr as *const _ as *mut c_void,
                &x_ptr as *const _ as *mut c_void,
                &y_ptr as *const _ as *mut c_void,
                &in_dim_u32 as *const _ as *mut c_void,
                &out_dim_u32 as *const _ as *mut c_void,
                &n_tokens_u32 as *const _ as *mut c_void,
                &row_bytes_u32 as *const _ as *mut c_void,
                &grids_ptr as *const _ as *mut c_void,
            ];
            assert_eq!(
                hipModuleLaunchKernel(
                    function,
                    num_blocks,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    state.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                HIP_SUCCESS,
                "rocm kernel launch failed"
            );
            assert_eq!(
                hipDeviceSynchronize(),
                HIP_SUCCESS,
                "rocm device sync failed"
            );

            let mut y = vec![0f32; y_len];
            assert_eq!(
                hipMemcpy(
                    y.as_mut_ptr() as *mut c_void,
                    y_ptr,
                    y_bytes,
                    hipMemcpyKind_hipMemcpyDeviceToHost,
                ),
                HIP_SUCCESS,
                "rocm y readback failed"
            );
            hipFree(x_ptr);
            hipFree(y_ptr);
            y
        }
    }

    /// Every op launched before any is read back — see
    /// [`crate::engine::backend::CudaBackend::matmul_batch`] for the shape.
    ///
    /// The saving here is the largest of the three vendor backends, because
    /// `matmul` ends in `hipDeviceSynchronize` — a *whole device* barrier,
    /// not a stream or a copy — and the default one-`matmul`-per-op
    /// implementation therefore drains the device once per op. This issues a
    /// chunk's launches back to back and drains **once**.
    ///
    /// It also takes this backend's one global `Mutex` once per chunk rather
    /// than once per op. That lock is the module's deliberate
    /// safe-by-construction answer to having no thread-safety guidance for
    /// the raw HIP FFI (see the module doc), so holding it across a chunk
    /// keeps exactly the property it was taken for: at most one thread ever
    /// inside the HIP runtime.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        use crate::engine::backend::{BATCH_DEVICE_BUDGET_BYTES, plan_batch};

        let (stripes, chunks) = plan_batch(ops, BATCH_DEVICE_BUDGET_BYTES);
        let mut out: Vec<Vec<f32>> = ops
            .iter()
            .map(|op| Vec::with_capacity(op.n_tokens * op.w.out_dim))
            .collect();
        let mut state = self.state.lock().expect("rocm state poisoned");
        for chunk in chunks {
            // Per stripe, kept until after the single sync below: the device
            // pointers the kernels read and write, and the length to copy
            // back. Freeing any of them before the sync would free memory a
            // queued kernel is about to touch.
            let mut pending: Vec<(usize, *mut c_void, *mut c_void, usize)> =
                Vec::with_capacity(chunk.len());
            unsafe {
                for stripe in &stripes[chunk] {
                    let op = &ops[stripe.op];
                    let in_dim = op.w.in_dim;
                    let out_dim = op.w.out_dim;
                    let rows = stripe.start * in_dim..(stripe.start + stripe.n_tokens) * in_dim;
                    let x = &op.x[rows];
                    let y_len = stripe.n_tokens * out_dim;
                    let key = op.w.cache_key();

                    let weights_ptr = if let Some(&(ptr, _len)) = state.weight_buffers.get(&key) {
                        ptr
                    } else {
                        let bytes = op.w.raw_bytes();
                        let mut device_ptr: *mut c_void = ptr::null_mut();
                        assert_eq!(
                            hipMalloc(&mut device_ptr, bytes.len().max(1)),
                            HIP_SUCCESS,
                            "rocm weight buffer allocation failed"
                        );
                        assert_eq!(
                            hipMemcpy(
                                device_ptr,
                                bytes.as_ptr() as *const c_void,
                                bytes.len(),
                                hipMemcpyKind_hipMemcpyHostToDevice,
                            ),
                            HIP_SUCCESS,
                            "rocm weight upload failed"
                        );
                        state.weight_buffers.insert(key, (device_ptr, bytes.len()));
                        device_ptr
                    };

                    let x_bytes = std::mem::size_of_val(x);
                    let mut x_ptr: *mut c_void = ptr::null_mut();
                    assert_eq!(
                        hipMalloc(&mut x_ptr, x_bytes.max(1)),
                        HIP_SUCCESS,
                        "rocm x allocation failed"
                    );
                    assert_eq!(
                        hipMemcpy(
                            x_ptr,
                            x.as_ptr() as *const c_void,
                            x_bytes,
                            hipMemcpyKind_hipMemcpyHostToDevice,
                        ),
                        HIP_SUCCESS,
                        "rocm x upload failed"
                    );

                    let y_bytes = y_len * std::mem::size_of::<f32>();
                    let mut y_ptr: *mut c_void = ptr::null_mut();
                    assert_eq!(
                        hipMalloc(&mut y_ptr, y_bytes.max(1)),
                        HIP_SUCCESS,
                        "rocm y allocation failed"
                    );

                    let &(_module, function) =
                        state.functions.get(&op.w.ggml_type()).unwrap_or_else(|| {
                            panic!(
                                "ggml_type {} reached RocmBackend::matmul_batch without a \
                                 compiled kernel (QuantMatrix construction should have \
                                 rejected it earlier)",
                                op.w.ggml_type()
                            )
                        });

                    let in_dim_u32 = in_dim as u32;
                    let out_dim_u32 = out_dim as u32;
                    let n_tokens_u32 = stripe.n_tokens as u32;
                    let row_bytes_u32 = op.w.row_bytes() as u32;
                    let grids_ptr = state.iq_grids;
                    let n_row_groups = out_dim.div_ceil(4);
                    let num_blocks = (n_row_groups * stripe.n_tokens).max(1) as u32;

                    // The driver reads this array during the launch call, so
                    // these locals only have to outlive the call itself.
                    let mut args: [*mut c_void; 8] = [
                        &weights_ptr as *const _ as *mut c_void,
                        &x_ptr as *const _ as *mut c_void,
                        &y_ptr as *const _ as *mut c_void,
                        &in_dim_u32 as *const _ as *mut c_void,
                        &out_dim_u32 as *const _ as *mut c_void,
                        &n_tokens_u32 as *const _ as *mut c_void,
                        &row_bytes_u32 as *const _ as *mut c_void,
                        &grids_ptr as *const _ as *mut c_void,
                    ];
                    assert_eq!(
                        hipModuleLaunchKernel(
                            function,
                            num_blocks,
                            1,
                            1,
                            64,
                            1,
                            1,
                            0,
                            state.stream,
                            args.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        HIP_SUCCESS,
                        "rocm kernel launch failed"
                    );
                    pending.push((stripe.op, x_ptr, y_ptr, y_len));
                }

                // Once for the chunk, where the per-op path drained the
                // device once per op.
                assert_eq!(
                    hipDeviceSynchronize(),
                    HIP_SUCCESS,
                    "rocm device sync failed"
                );

                for (op, x_ptr, y_ptr, y_len) in pending {
                    let mut y = vec![0f32; y_len];
                    assert_eq!(
                        hipMemcpy(
                            y.as_mut_ptr() as *mut c_void,
                            y_ptr,
                            y_len * std::mem::size_of::<f32>(),
                            hipMemcpyKind_hipMemcpyDeviceToHost,
                        ),
                        HIP_SUCCESS,
                        "rocm y readback failed"
                    );
                    hipFree(x_ptr);
                    hipFree(y_ptr);
                    out[op].extend(y);
                }
            }
        }
        out
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

    /// One `RocmBackend`, lazily built and shared across every test in this
    /// module — see `cuda::tests::shared_cuda`'s doc comment for the
    /// identical rationale. No ROCm runtime is installed on this project's
    /// dev machine, so `try_init()` returns `None` and every test below
    /// skips, per the same convention every other backend's tests use.
    fn shared_rocm() -> Option<&'static RocmBackend> {
        static ROCM: std::sync::OnceLock<Option<RocmBackend>> = std::sync::OnceLock::new();
        ROCM.get_or_init(RocmBackend::try_init).as_ref()
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

    /// Cross-checks `RocmBackend::matmul` against `CpuBackend::matmul` for
    /// every supported `ggml_type` — the same methodology `vulkan.rs`/
    /// `cuda.rs`/`opencl.rs` use. Skips (doesn't fail) when no HIP device is
    /// available, per `shared_rocm`'s doc comment.
    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(rocm) = shared_rocm() else {
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

        let expected = CpuBackend.matmul(&x, n_tokens, &w);
        let actual = rocm.matmul(&x, n_tokens, &w);
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
        let Some(rocm) = shared_rocm() else {
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
        let batched = rocm.matmul_batch(&ops);
        let expected_a = rocm.matmul(&xa, 1, &wa);
        let expected_b = rocm.matmul(&xb, 1, &wb);
        assert_eq!(batched[0], expected_a);
        assert_eq!(batched[1], expected_b);
    }
}

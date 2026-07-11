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
    HIP_SUCCESS, hipDeviceGetName, hipDeviceSynchronize, hipFree, hipFunction_t, hipGetDeviceCount,
    hipInit, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipModule_t, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoadData, hipSetDevice, hipStream_t, hipStreamCreate, hiprtcCompileProgram,
    hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode, hiprtcGetCodeSize,
    hiprtcGetProgramLog, hiprtcGetProgramLogSize, hiprtcProgram, hiprtcResult_HIPRTC_SUCCESS,
};

use crate::engine::loader::QuantMatrix;
use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

use super::{Backend, MatmulOp};

/// The `ggml_type`s a kernel exists for — the same set `engine::quant`
/// supports on the CPU path.
const SUPPORTED_TYPES: &[u32] = &[
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
];

const KERNEL_NAME: &str = "matmul_reduce";

/// Deliberately near-identical to `engine::backend::cuda`'s `PRELUDE` —
/// HIP-C is designed to be source-compatible with CUDA-C for exactly this
/// kind of kernel (no vendor-specific intrinsics beyond `__int_as_float`,
/// which HIP-clang also provides for CUDA-porting compatibility). Kept as
/// its own copy per this backend's module rather than shared across files,
/// matching `cuda.rs`/`opencl.rs`'s own precedent.
const PRELUDE: &str = r#"
extern "C" __device__ float orangu_half_to_float(unsigned short h) {
    unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = -1;
            do {
                mant <<= 1;
                e++;
            } while ((mant & 0x400u) == 0u);
            mant &= 0x3FFu;
            bits = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return __int_as_float((int)bits);
}

// bfloat16 -> f32: the top 16 bits of an f32, left-shifted into place —
// mirrors `quant::dequantize`'s `GGML_TYPE_BF16` arm exactly.
extern "C" __device__ float orangu_bf16_to_float(unsigned short h) {
    unsigned int bits = ((unsigned int)h) << 16;
    return __int_as_float((int)bits);
}

// ggml's `get_scale_min_k4`: unpacks the 6-bit scale and 6-bit min for
// sub-block `j` (0..8) of a Q4_K/Q5_K super-block's 12-byte `scales` region
// starting at byte `base`. Mirrors `quant::get_scale_min_k4` exactly.
extern "C" __device__ void orangu_get_scale_min_k4(
    const unsigned char *w, unsigned int base, unsigned int j,
    unsigned int *sc, unsigned int *m) {
    if (j < 4u) {
        *sc = w[base + j] & 63u;
        *m = w[base + j + 4u] & 63u;
    } else {
        *sc = (w[base + j + 4u] & 0xFu) | ((w[base + j - 4u] >> 6) << 4);
        *m = (w[base + j + 4u] >> 4) | ((w[base + j] >> 6) << 4);
    }
}
"#;

/// Deliberately near-identical to `engine::backend::cuda`'s `MAIN` — see
/// that constant's doc comment; this is the same `MAIN_REDUCE_SUFFIX` port.
const MAIN: &str = r#"
extern "C" __global__ void matmul_reduce(
    const unsigned char *weights,
    const float *x,
    float *y,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int n_tokens,
    unsigned int row_bytes) {
    __shared__ float partial_sums[256];

    unsigned int n_row_groups = (out_dim + 3u) / 4u;
    unsigned int flat = blockIdx.x;
    if (flat >= n_row_groups * n_tokens) {
        return;
    }
    unsigned int rg = flat / n_tokens;
    unsigned int t = flat % n_tokens;
    unsigned int o0 = rg * 4u;
    unsigned int o1 = o0 + 1u;
    unsigned int o2 = o0 + 2u;
    unsigned int o3 = o0 + 3u;
    unsigned int local = threadIdx.x;
    unsigned int x_base = t * in_dim;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    for (unsigned int k = local; k < in_dim; k += 64u) {
        unsigned int block_idx = k / BLOCK_ELEMS;
        unsigned int local_k = k % BLOCK_ELEMS;
        unsigned int block_off = block_idx * BLOCK_BYTES;
        float xv = x[x_base + k];
        partial0 += dequant_element(weights, o0 * row_bytes + block_off, local_k) * xv;
        if (o1 < out_dim) {
            partial1 += dequant_element(weights, o1 * row_bytes + block_off, local_k) * xv;
        }
        if (o2 < out_dim) {
            partial2 += dequant_element(weights, o2 * row_bytes + block_off, local_k) * xv;
        }
        if (o3 < out_dim) {
            partial3 += dequant_element(weights, o3 * row_bytes + block_off, local_k) * xv;
        }
    }

    partial_sums[local] = partial0;
    partial_sums[64u + local] = partial1;
    partial_sums[128u + local] = partial2;
    partial_sums[192u + local] = partial3;
    __syncthreads();
    for (unsigned int stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            partial_sums[local] += partial_sums[local + stride];
            partial_sums[64u + local] += partial_sums[64u + local + stride];
            partial_sums[128u + local] += partial_sums[128u + local + stride];
            partial_sums[192u + local] += partial_sums[192u + local + stride];
        }
        __syncthreads();
    }
    if (local == 0u) {
        y[t * out_dim + o0] = partial_sums[0];
        if (o1 < out_dim) {
            y[t * out_dim + o1] = partial_sums[64u];
        }
        if (o2 < out_dim) {
            y[t * out_dim + o2] = partial_sums[128u];
        }
        if (o3 < out_dim) {
            y[t * out_dim + o3] = partial_sums[192u];
        }
    }
}
"#;

/// Deliberately near-identical to `engine::backend::cuda`'s
/// `dequant_element_source` — see that function's doc comment.
fn dequant_element_source(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        t if t == GGML_TYPE_F32 => {
            r#"
const unsigned int BLOCK_BYTES = 4u;
const unsigned int BLOCK_ELEMS = 1u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    unsigned int bits = (unsigned int)w[byte_offset] | ((unsigned int)w[byte_offset + 1] << 8)
        | ((unsigned int)w[byte_offset + 2] << 16) | ((unsigned int)w[byte_offset + 3] << 24);
    return __int_as_float((int)bits);
}
"#
        }
        t if t == GGML_TYPE_F16 => {
            r#"
const unsigned int BLOCK_BYTES = 2u;
const unsigned int BLOCK_ELEMS = 1u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    unsigned short bits = (unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8);
    return orangu_half_to_float(bits);
}
"#
        }
        t if t == GGML_TYPE_BF16 => {
            r#"
const unsigned int BLOCK_BYTES = 2u;
const unsigned int BLOCK_ELEMS = 1u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    unsigned short bits = (unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8);
    return orangu_bf16_to_float(bits);
}
"#
        }
        t if t == GGML_TYPE_Q4_0 => {
            r#"
const unsigned int BLOCK_BYTES = 18u;
const unsigned int BLOCK_ELEMS = 32u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    float d = orangu_half_to_float((unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8));
    if (k < 16u) {
        unsigned char byte = w[byte_offset + 2u + k];
        return ((float)((int)(byte & 0xFu) - 8)) * d;
    }
    unsigned char byte = w[byte_offset + 2u + (k - 16u)];
    return ((float)((int)(byte >> 4) - 8)) * d;
}
"#
        }
        t if t == GGML_TYPE_Q5_0 => {
            r#"
const unsigned int BLOCK_BYTES = 22u;
const unsigned int BLOCK_ELEMS = 32u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    float d = orangu_half_to_float((unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8));
    unsigned int qh = (unsigned int)w[byte_offset + 2] | ((unsigned int)w[byte_offset + 3] << 8)
        | ((unsigned int)w[byte_offset + 4] << 16) | ((unsigned int)w[byte_offset + 5] << 24);
    if (k < 16u) {
        unsigned char byte = w[byte_offset + 6u + k];
        unsigned int xh0 = ((qh >> k) << 4) & 0x10u;
        return ((float)((int)((byte & 0xFu) | xh0) - 16)) * d;
    }
    unsigned int j = k - 16u;
    unsigned char byte = w[byte_offset + 6u + j];
    unsigned int xh1 = (qh >> (j + 12u)) & 0x10u;
    return ((float)((int)((byte >> 4) | xh1) - 16)) * d;
}
"#
        }
        t if t == GGML_TYPE_Q8_0 => {
            r#"
const unsigned int BLOCK_BYTES = 34u;
const unsigned int BLOCK_ELEMS = 32u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    float d = orangu_half_to_float((unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8));
    signed char q = (signed char)w[byte_offset + 2u + k];
    return ((float)q) * d;
}
"#
        }
        t if t == GGML_TYPE_Q4_K => {
            r#"
const unsigned int BLOCK_BYTES = 144u;
const unsigned int BLOCK_ELEMS = 256u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    float d = orangu_half_to_float((unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((unsigned short)w[byte_offset + 2] | ((unsigned short)w[byte_offset + 3] << 8));
    unsigned int scales_off = byte_offset + 4u;
    unsigned int qs_off = byte_offset + 16u;
    unsigned int q_offset = (k / 64u) * 64u;
    unsigned int local_in_group = k % 64u;
    unsigned int is_base = (q_offset / 64u) * 2u;
    unsigned int q_base = qs_off + q_offset / 2u;
    unsigned int sc, m;
    if (local_in_group < 32u) {
        unsigned char byte = w[q_base + local_in_group];
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)(byte & 0xFu) - m1;
    }
    unsigned int l = local_in_group - 32u;
    unsigned char byte = w[q_base + l];
    orangu_get_scale_min_k4(w, scales_off, is_base + 1u, &sc, &m);
    float d2 = d * (float)sc;
    float m2 = dmin * (float)m;
    return d2 * (float)(byte >> 4) - m2;
}
"#
        }
        t if t == GGML_TYPE_Q5_K => {
            r#"
const unsigned int BLOCK_BYTES = 176u;
const unsigned int BLOCK_ELEMS = 256u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    float d = orangu_half_to_float((unsigned short)w[byte_offset] | ((unsigned short)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((unsigned short)w[byte_offset + 2] | ((unsigned short)w[byte_offset + 3] << 8));
    unsigned int scales_off = byte_offset + 4u;
    unsigned int qh_off = byte_offset + 16u;
    unsigned int qs_off = byte_offset + 48u;
    unsigned int q_offset = (k / 64u) * 64u;
    unsigned int idx = q_offset / 64u;
    unsigned int local_in_group = k % 64u;
    unsigned int is_base = idx * 2u;
    unsigned int ql_offset = idx * 32u;
    unsigned int u1 = 1u << (2u * idx);
    unsigned int u2 = 2u << (2u * idx);
    unsigned int sc, m;
    if (local_in_group < 32u) {
        unsigned int l = local_in_group;
        unsigned char byte = w[qs_off + ql_offset + l];
        unsigned char qhbyte = w[qh_off + l];
        int hi_bit = (qhbyte & u1) != 0u ? 16 : 0;
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)((int)(byte & 0xFu) + hi_bit) - m1;
    }
    unsigned int l = local_in_group - 32u;
    unsigned char byte = w[qs_off + ql_offset + l];
    unsigned char qhbyte = w[qh_off + l];
    int hi_bit = (qhbyte & u2) != 0u ? 16 : 0;
    orangu_get_scale_min_k4(w, scales_off, is_base + 1u, &sc, &m);
    float d2 = d * (float)sc;
    float m2 = dmin * (float)m;
    return d2 * (float)((int)(byte >> 4) + hi_bit) - m2;
}
"#
        }
        t if t == GGML_TYPE_Q6_K => {
            r#"
const unsigned int BLOCK_BYTES = 210u;
const unsigned int BLOCK_ELEMS = 256u;
extern "C" __device__ float dequant_element(const unsigned char *w, unsigned int byte_offset, unsigned int k) {
    unsigned int ql_off = byte_offset;
    unsigned int qh_off = byte_offset + 128u;
    unsigned int sc_off = byte_offset + 192u;
    float d = orangu_half_to_float((unsigned short)w[byte_offset + 208] | ((unsigned short)w[byte_offset + 209] << 8));
    unsigned int y_off = (k / 128u) * 128u;
    unsigned int idx = y_off / 128u;
    unsigned int local_in_group = k % 128u;
    unsigned int which_q = local_in_group / 32u;
    unsigned int l = local_in_group % 32u;
    unsigned int ql_o = idx * 64u;
    unsigned int qh_o = idx * 32u;
    unsigned int sc_o = idx * 8u;
    unsigned int is = l / 16u;
    unsigned char ql_l = w[ql_off + ql_o + l];
    unsigned char ql_l32 = w[ql_off + ql_o + l + 32u];
    unsigned char qh_l = w[qh_off + qh_o + l];
    int q;
    unsigned int sc_idx;
    if (which_q == 0u) {
        q = (int)((ql_l & 0xFu) | ((qh_l & 3u) << 4)) - 32;
        sc_idx = is;
    } else if (which_q == 1u) {
        q = (int)((ql_l32 & 0xFu) | (((qh_l >> 2) & 3u) << 4)) - 32;
        sc_idx = is + 2u;
    } else if (which_q == 2u) {
        q = (int)((ql_l >> 4) | (((qh_l >> 4) & 3u) << 4)) - 32;
        sc_idx = is + 4u;
    } else {
        q = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3u) << 4)) - 32;
        sc_idx = is + 6u;
    }
    signed char sc = (signed char)w[sc_off + sc_o + sc_idx];
    return d * (float)sc * (float)q;
}
"#
        }
        _ => return None,
    })
}

/// The complete, compile-ready HIP-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    let middle = dequant_element_source(ggml_type)?;
    Some(format!("{PRELUDE}\n{middle}\n{MAIN}"))
}

/// Compiles `source` via HIPRTC and loads it as a module, returning the
/// module and the `KERNEL_NAME` function within it — the HIP-C mirror of
/// `cudarc::nvrtc::compile_ptx` + `CudaContext::load_module`/
/// `load_function`, following the exact same steps `cubecl-hip-sys`'s own
/// `test_launch_kernel_end_to_end` test demonstrates (create program,
/// compile, pull out the compiled code, destroy the program, load the
/// code as a module, resolve the function by name).
///
/// # Safety
/// Requires a HIP device already selected on this thread (`hipSetDevice`).
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
    /// `(device pointer, byte length)`, keyed by `QuantMatrix::cache_key`
    /// — same reuse discipline as the other backends' weight caches, just
    /// folded into the one state struct this backend locks as a whole.
    weight_buffers: HashMap<(usize, usize), (*mut c_void, usize)>,
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
    /// Looks for a usable HIP device (ordinal 0) and compiles every
    /// supported quant type's kernel via HIPRTC up front. Returns `None`
    /// (never panics) if no HIP runtime/device is present, or compilation
    /// otherwise fails — callers fall back to `CpuBackend`, the same
    /// contract every other backend's `try_init` has.
    pub fn try_init() -> Option<Self> {
        unsafe {
            if hipInit(0) != HIP_SUCCESS {
                return None;
            }
            let mut count: std::os::raw::c_int = 0;
            if hipGetDeviceCount(&mut count) != HIP_SUCCESS || count == 0 {
                return None;
            }
            if hipSetDevice(0) != HIP_SUCCESS {
                return None;
            }

            let mut name_buf = [0i8; 256];
            let device_name = if hipDeviceGetName(name_buf.as_mut_ptr(), 256, 0) == HIP_SUCCESS {
                CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                "ROCm".to_string()
            };

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

            Some(Self {
                state: Mutex::new(RocmState {
                    stream,
                    functions,
                    weight_buffers: HashMap::new(),
                }),
                device_name,
            })
        }
    }
}

impl Backend for RocmBackend {
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
            let n_row_groups = out_dim.div_ceil(4);
            let num_blocks = (n_row_groups * n_tokens).max(1) as u32;

            let mut args: [*mut c_void; 7] = [
                &weights_ptr as *const _ as *mut c_void,
                &x_ptr as *const _ as *mut c_void,
                &y_ptr as *const _ as *mut c_void,
                &in_dim_u32 as *const _ as *mut c_void,
                &out_dim_u32 as *const _ as *mut c_void,
                &n_tokens_u32 as *const _ as *mut c_void,
                &row_bytes_u32 as *const _ as *mut c_void,
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

    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        ops.iter()
            .map(|op| self.matmul(op.x, op.n_tokens, op.w))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{
        GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
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

    fn block_bytes_for(ggml_type: u32) -> usize {
        match ggml_type {
            t if t == GGML_TYPE_F32 => 4,
            t if t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 => 2,
            t if t == GGML_TYPE_Q4_0 => 18,
            t if t == GGML_TYPE_Q5_0 => 22,
            t if t == GGML_TYPE_Q8_0 => 34,
            t if t == GGML_TYPE_Q4_K => 144,
            t if t == GGML_TYPE_Q5_K => 176,
            t if t == GGML_TYPE_Q6_K => 210,
            _ => unreachable!(),
        }
    }

    fn block_elems_for(ggml_type: u32) -> usize {
        match ggml_type {
            t if t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 => 1,
            t if t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q8_0 => 32,
            _ => 256,
        }
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

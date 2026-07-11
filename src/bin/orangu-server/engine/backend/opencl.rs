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

//! OpenCL backend, via `opencl3`. Structurally and functionally identical
//! in scope to `engine::backend::cuda`'s `CudaBackend` — see that module's
//! doc comment for what's implemented ([`Backend::matmul`] only, a direct
//! port of `vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` reduction kernel) and
//! what isn't (`VulkanBackend`'s much larger fused/GPU-resident surface).
//! Not verified on real OpenCL-capable hardware — this project's dev
//! machine has the ICD loader installed (`ocl-icd`) but no vendor ICD
//! registered, so [`OpenClBackend::try_init`] finds zero platforms here and
//! gracefully returns `None`, the same as every other machine with no
//! OpenCL device — this module's cross-check tests skip in exactly that
//! case, per the same convention `vulkan.rs`/`cuda.rs` use.
//!
//! **Always compiled in**, like `cudarc`/`wgpu` — no Cargo feature needed.
//! The `opencl3` version resolved here defaults to its `dynamic` feature
//! (`cl3`'s own dlopen-based loader), which dlopens the ICD loader
//! (`libOpenCL.so`) at *runtime* and returns a real `Result` if that fails,
//! rather than requiring anything at build time or panicking — checked
//! directly against the actual resolved dependency version rather than
//! assumed (an older generation of `opencl-sys` did hard-link `-lOpenCL`
//! at build time; that's no longer what gets pulled in here).

use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

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

/// The reduction workgroup size — must match `LOCAL_WORK_SIZE` used at
/// dispatch time and the hardcoded `64`/`partial_sums[256]` layout below,
/// the same relationship `vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` has to
/// `@workgroup_size(64)`.
const LOCAL_WORK_SIZE: usize = 64;

/// Shared by every type's kernel: a manual IEEE-754 binary16 -> float
/// decoder (OpenCL C's `half` type/`vload_half` requires the `cl_khr_fp16`
/// extension, not guaranteed present, so this avoids depending on it) —
/// see `engine::backend::cuda`'s near-identical sibling copy of this
/// prelude for why the same logic is duplicated per backend rather than
/// shared across files.
const PRELUDE: &str = r#"
inline float orangu_half_to_float(ushort h) {
    uint sign = ((uint)(h & 0x8000u)) << 16;
    uint exp = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    uint bits;
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
            bits = sign | ((uint)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return as_float(bits);
}

// bfloat16 -> f32: the top 16 bits of an f32, left-shifted into place —
// mirrors `quant::dequantize`'s `GGML_TYPE_BF16` arm exactly.
inline float orangu_bf16_to_float(ushort h) {
    uint bits = ((uint)h) << 16;
    return as_float(bits);
}

// ggml's `get_scale_min_k4`: unpacks the 6-bit scale and 6-bit min for
// sub-block `j` (0..8) of a Q4_K/Q5_K super-block's 12-byte `scales` region
// starting at byte `base`. Mirrors `quant::get_scale_min_k4` exactly.
inline void orangu_get_scale_min_k4(
    __global const uchar *w, uint base, uint j, uint *sc, uint *m) {
    if (j < 4u) {
        *sc = w[base + j] & 63u;
        *m = w[base + j + 4u] & 63u;
    } else {
        *sc = (w[base + j + 4u] & 0xFu) | ((w[base + j - 4u] >> 6) << 4);
        *m = (w[base + j + 4u] >> 4) | ((w[base + j] >> 6) << 4);
    }
}
"#;

/// The compute entry point — an OpenCL-C port of `vulkan_shaders`'s
/// `MAIN_REDUCE_SUFFIX` (see `engine::backend::cuda`'s `MAIN` for the
/// CUDA-C sibling this was translated alongside): one workgroup per
/// (output-row group of 4, token) pair, all `LOCAL_WORK_SIZE` work-items
/// splitting `in_dim` elements and reducing their partial dot products in
/// local memory.
const MAIN: &str = r#"
__kernel void matmul_reduce(
    __global const uchar *weights,
    __global const float *x,
    __global float *y,
    uint in_dim,
    uint out_dim,
    uint n_tokens,
    uint row_bytes) {
    __local float partial_sums[256];

    uint n_row_groups = (out_dim + 3u) / 4u;
    uint flat = get_group_id(0);
    if (flat >= n_row_groups * n_tokens) {
        return;
    }
    uint rg = flat / n_tokens;
    uint t = flat % n_tokens;
    uint o0 = rg * 4u;
    uint o1 = o0 + 1u;
    uint o2 = o0 + 2u;
    uint o3 = o0 + 3u;
    uint local_id = get_local_id(0);
    uint x_base = t * in_dim;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    for (uint k = local_id; k < in_dim; k += 64u) {
        uint block_idx = k / BLOCK_ELEMS;
        uint local_k = k % BLOCK_ELEMS;
        uint block_off = block_idx * BLOCK_BYTES;
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

    partial_sums[local_id] = partial0;
    partial_sums[64u + local_id] = partial1;
    partial_sums[128u + local_id] = partial2;
    partial_sums[192u + local_id] = partial3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 32u; stride > 0u; stride /= 2u) {
        if (local_id < stride) {
            partial_sums[local_id] += partial_sums[local_id + stride];
            partial_sums[64u + local_id] += partial_sums[64u + local_id + stride];
            partial_sums[128u + local_id] += partial_sums[128u + local_id + stride];
            partial_sums[192u + local_id] += partial_sums[192u + local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0u) {
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

/// Each type's `dequant_element(weights, byte_offset, k)` — see
/// `engine::backend::cuda::dequant_element_source`'s doc comment; this is
/// the same line-for-line port of `engine::quant::dequantize_*`, restated
/// in OpenCL C.
fn dequant_element_source(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        t if t == GGML_TYPE_F32 => {
            r#"
constant uint BLOCK_BYTES = 4u;
constant uint BLOCK_ELEMS = 1u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    uint bits = (uint)w[byte_offset] | ((uint)w[byte_offset + 1] << 8)
        | ((uint)w[byte_offset + 2] << 16) | ((uint)w[byte_offset + 3] << 24);
    return as_float(bits);
}
"#
        }
        t if t == GGML_TYPE_F16 => {
            r#"
constant uint BLOCK_BYTES = 2u;
constant uint BLOCK_ELEMS = 1u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    ushort bits = (ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8);
    return orangu_half_to_float(bits);
}
"#
        }
        t if t == GGML_TYPE_BF16 => {
            r#"
constant uint BLOCK_BYTES = 2u;
constant uint BLOCK_ELEMS = 1u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    ushort bits = (ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8);
    return orangu_bf16_to_float(bits);
}
"#
        }
        t if t == GGML_TYPE_Q4_0 => {
            r#"
constant uint BLOCK_BYTES = 18u;
constant uint BLOCK_ELEMS = 32u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    float d = orangu_half_to_float((ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8));
    if (k < 16u) {
        uchar byte = w[byte_offset + 2u + k];
        return ((float)((int)(byte & 0xFu) - 8)) * d;
    }
    uchar byte = w[byte_offset + 2u + (k - 16u)];
    return ((float)((int)(byte >> 4) - 8)) * d;
}
"#
        }
        t if t == GGML_TYPE_Q5_0 => {
            r#"
constant uint BLOCK_BYTES = 22u;
constant uint BLOCK_ELEMS = 32u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    float d = orangu_half_to_float((ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8));
    uint qh = (uint)w[byte_offset + 2] | ((uint)w[byte_offset + 3] << 8)
        | ((uint)w[byte_offset + 4] << 16) | ((uint)w[byte_offset + 5] << 24);
    if (k < 16u) {
        uchar byte = w[byte_offset + 6u + k];
        uint xh0 = ((qh >> k) << 4) & 0x10u;
        return ((float)((int)((byte & 0xFu) | xh0) - 16)) * d;
    }
    uint j = k - 16u;
    uchar byte = w[byte_offset + 6u + j];
    uint xh1 = (qh >> (j + 12u)) & 0x10u;
    return ((float)((int)((byte >> 4) | xh1) - 16)) * d;
}
"#
        }
        t if t == GGML_TYPE_Q8_0 => {
            r#"
constant uint BLOCK_BYTES = 34u;
constant uint BLOCK_ELEMS = 32u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    float d = orangu_half_to_float((ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8));
    char q = (char)w[byte_offset + 2u + k];
    return ((float)q) * d;
}
"#
        }
        t if t == GGML_TYPE_Q4_K => {
            r#"
constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_ELEMS = 256u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    float d = orangu_half_to_float((ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((ushort)w[byte_offset + 2] | ((ushort)w[byte_offset + 3] << 8));
    uint scales_off = byte_offset + 4u;
    uint qs_off = byte_offset + 16u;
    uint q_offset = (k / 64u) * 64u;
    uint local_in_group = k % 64u;
    uint is_base = (q_offset / 64u) * 2u;
    uint q_base = qs_off + q_offset / 2u;
    uint sc, m;
    if (local_in_group < 32u) {
        uchar byte = w[q_base + local_in_group];
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)(byte & 0xFu) - m1;
    }
    uint l = local_in_group - 32u;
    uchar byte = w[q_base + l];
    orangu_get_scale_min_k4(w, scales_off, is_base + 1u, &sc, &m);
    float d2 = d * (float)sc;
    float m2 = dmin * (float)m;
    return d2 * (float)(byte >> 4) - m2;
}
"#
        }
        t if t == GGML_TYPE_Q5_K => {
            r#"
constant uint BLOCK_BYTES = 176u;
constant uint BLOCK_ELEMS = 256u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    float d = orangu_half_to_float((ushort)w[byte_offset] | ((ushort)w[byte_offset + 1] << 8));
    float dmin = orangu_half_to_float((ushort)w[byte_offset + 2] | ((ushort)w[byte_offset + 3] << 8));
    uint scales_off = byte_offset + 4u;
    uint qh_off = byte_offset + 16u;
    uint qs_off = byte_offset + 48u;
    uint q_offset = (k / 64u) * 64u;
    uint idx = q_offset / 64u;
    uint local_in_group = k % 64u;
    uint is_base = idx * 2u;
    uint ql_offset = idx * 32u;
    uint u1 = 1u << (2u * idx);
    uint u2 = 2u << (2u * idx);
    uint sc, m;
    if (local_in_group < 32u) {
        uint l = local_in_group;
        uchar byte = w[qs_off + ql_offset + l];
        uchar qhbyte = w[qh_off + l];
        int hi_bit = (qhbyte & u1) != 0u ? 16 : 0;
        orangu_get_scale_min_k4(w, scales_off, is_base, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        return d1 * (float)((int)(byte & 0xFu) + hi_bit) - m1;
    }
    uint l = local_in_group - 32u;
    uchar byte = w[qs_off + ql_offset + l];
    uchar qhbyte = w[qh_off + l];
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
constant uint BLOCK_BYTES = 210u;
constant uint BLOCK_ELEMS = 256u;
inline float dequant_element(__global const uchar *w, uint byte_offset, uint k) {
    uint ql_off = byte_offset;
    uint qh_off = byte_offset + 128u;
    uint sc_off = byte_offset + 192u;
    float d = orangu_half_to_float((ushort)w[byte_offset + 208] | ((ushort)w[byte_offset + 209] << 8));
    uint y_off = (k / 128u) * 128u;
    uint idx = y_off / 128u;
    uint local_in_group = k % 128u;
    uint which_q = local_in_group / 32u;
    uint l = local_in_group % 32u;
    uint ql_o = idx * 64u;
    uint qh_o = idx * 32u;
    uint sc_o = idx * 8u;
    uint is = l / 16u;
    uchar ql_l = w[ql_off + ql_o + l];
    uchar ql_l32 = w[ql_off + ql_o + l + 32u];
    uchar qh_l = w[qh_off + qh_o + l];
    int q;
    uint sc_idx;
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
    char sc = (char)w[sc_off + sc_o + sc_idx];
    return d * (float)sc * (float)q;
}
"#
        }
        _ => return None,
    })
}

/// The complete, compile-ready OpenCL-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    let middle = dequant_element_source(ggml_type)?;
    Some(format!("{PRELUDE}\n{middle}\n{MAIN}"))
}

/// `QuantMatrix::cache_key()`'s return type — named, like `vulkan.rs`'s own
/// `WeightCacheKey`, so `weight_cache`'s type doesn't trip clippy's
/// `type_complexity` lint.
type WeightCacheKey = (usize, usize);

pub struct OpenClBackend {
    context: Context,
    /// Guards both the command queue and (implicitly, since a launch is
    /// entirely done while holding this) each kernel object's argument
    /// state — `clSetKernelArg` mutates the `cl_kernel` object in place, so
    /// two threads racing to set arguments on the *same* `Kernel` before
    /// either enqueues would silently launch with a mixed-up argument set.
    /// `engine::scheduler::SlotPool` runs multiple request slots' forward
    /// passes concurrently, each potentially calling into the same
    /// `OpenClBackend`, so this genuinely needs guarding — unlike
    /// `VulkanBackend`, which instead gives each `(weight, n_tokens)` op
    /// its own cached, independently-lockable resources.
    queue: Mutex<CommandQueue>,
    /// One kernel per `ggml_type`, plus the `Program` each was built from
    /// (kept alive — a `Kernel` internally holds a `cl_program` reference,
    /// but keeping the owning `Program` here too avoids relying on that).
    /// `Kernel` is `Send` but not `Sync` (its raw `cl_kernel` handle can't
    /// safely be touched by two threads at once — the same
    /// `clSetKernelArg` hazard `Self::queue`'s doc comment explains), so
    /// the whole map lives behind one more `Mutex` rather than each
    /// `Kernel` individually — simpler, and this backend never needs two
    /// *different* types' kernels running concurrently anyway.
    kernels: Mutex<HashMap<u32, (Program, Kernel)>>,
    /// Same identity-keyed reuse discipline as `VulkanBackend::weight_
    /// buffer`/`CudaBackend::weight_buffer`.
    weight_cache: Mutex<HashMap<WeightCacheKey, Arc<Buffer<u8>>>>,
    /// The device's own name — for the startup banner.
    pub device_name: String,
}

impl OpenClBackend {
    /// Looks for the first GPU-type OpenCL device and builds every
    /// supported quant type's kernel up front. Returns `None` (never
    /// panics) if no OpenCL platform/device is found, or compilation
    /// otherwise fails — callers fall back to `CpuBackend`, the same
    /// contract `VulkanBackend`/`CudaBackend::try_init` have.
    pub fn try_init() -> Option<Self> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU).ok()?.first()?;
        let device = Device::new(device_id);
        let device_name = device.name().unwrap_or_else(|_| "OpenCL".to_string());
        let context = Context::from_device(&device).ok()?;
        let queue = CommandQueue::create_default(&context, 0).ok()?;

        let mut kernels = HashMap::new();
        for &ggml_type in SUPPORTED_TYPES {
            let source = kernel_source(ggml_type)?;
            let program = Program::create_and_build_from_source(&context, &source, "").ok()?;
            let kernel = Kernel::create(&program, KERNEL_NAME).ok()?;
            kernels.insert(ggml_type, (program, kernel));
        }

        Some(Self {
            context,
            queue: Mutex::new(queue),
            kernels: Mutex::new(kernels),
            weight_cache: Mutex::new(HashMap::new()),
            device_name,
        })
    }

    fn weight_buffer(&self, w: &QuantMatrix) -> Arc<Buffer<u8>> {
        let key = w.cache_key();
        if let Some(existing) = self
            .weight_cache
            .lock()
            .expect("opencl weight cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let bytes = w.raw_bytes();
        let mut buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                bytes.len().max(1),
                ptr::null_mut(),
            )
        }
        .expect("opencl weight buffer allocation failed");
        let queue = self.queue.lock().expect("opencl queue poisoned");
        unsafe { queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, bytes, &[]) }
            .expect("opencl weight upload failed");
        drop(queue);
        let uploaded = Arc::new(buf);
        self.weight_cache
            .lock()
            .expect("opencl weight cache poisoned")
            .insert(key, uploaded.clone());
        uploaded
    }
}

impl Backend for OpenClBackend {
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let in_dim = w.in_dim as u32;
        let out_dim = w.out_dim as u32;
        let row_bytes = w.row_bytes() as u32;
        let weights = self.weight_buffer(w);
        let y_len = n_tokens * w.out_dim;
        let n_tokens_u32 = n_tokens as u32;

        let n_row_groups = (out_dim as usize).div_ceil(4);
        let num_groups = (n_row_groups * n_tokens).max(1);
        let global_work_size = num_groups * LOCAL_WORK_SIZE;

        let queue = self.queue.lock().expect("opencl queue poisoned");
        let mut x_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                x.len().max(1),
                ptr::null_mut(),
            )
        }
        .expect("opencl x buffer allocation failed");
        unsafe { queue.enqueue_write_buffer(&mut x_buf, CL_BLOCKING, 0, x, &[]) }
            .expect("opencl x upload failed");
        let y_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                y_len.max(1),
                ptr::null_mut(),
            )
        }
        .expect("opencl y buffer allocation failed");

        let kernels = self.kernels.lock().expect("opencl kernels poisoned");
        let (_program, kernel) = kernels.get(&w.ggml_type()).unwrap_or_else(|| {
            panic!(
                "ggml_type {} reached OpenClBackend::matmul without a compiled kernel \
                 (QuantMatrix construction should have rejected it earlier)",
                w.ggml_type()
            )
        });
        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&weights)
                .set_arg(&x_buf)
                .set_arg(&y_buf)
                .set_arg(&in_dim)
                .set_arg(&out_dim)
                .set_arg(&n_tokens_u32)
                .set_arg(&row_bytes)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .set_global_work_size(global_work_size)
                .enqueue_nd_range(&queue)
        }
        .expect("opencl kernel launch failed");
        drop(kernels);

        let mut y = vec![0f32; y_len];
        let read_event = unsafe {
            queue.enqueue_read_buffer(&y_buf, CL_NON_BLOCKING, 0, &mut y, &[kernel_event.get()])
        }
        .expect("opencl y readback enqueue failed");
        read_event.wait().expect("opencl y readback failed");
        y
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

    /// One `OpenClBackend`, lazily built and shared across every test in
    /// this module — see `cuda::tests::shared_cuda`'s doc comment for the
    /// identical rationale. On this project's dev machine (ICD loader
    /// present, no vendor ICD registered — see this module's own doc
    /// comment) `try_init()` returns `None` and every test below skips.
    fn shared_opencl() -> Option<&'static OpenClBackend> {
        static OPENCL: std::sync::OnceLock<Option<OpenClBackend>> = std::sync::OnceLock::new();
        OPENCL.get_or_init(OpenClBackend::try_init).as_ref()
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

    /// Cross-checks `OpenClBackend::matmul` against `CpuBackend::matmul`
    /// for every supported `ggml_type` — the same methodology `vulkan.rs`/
    /// `cuda.rs` use. Skips (doesn't fail) when no OpenCL device is
    /// available, per `shared_opencl`'s doc comment.
    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(opencl) = shared_opencl() else {
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
        let actual = opencl.matmul(&x, n_tokens, &w);
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
        let Some(opencl) = shared_opencl() else {
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
        let batched = opencl.matmul_batch(&ops);
        let expected_a = opencl.matmul(&xa, 1, &wa);
        let expected_b = opencl.matmul(&xb, 1, &wb);
        assert_eq!(batched[0], expected_a);
        assert_eq!(batched[1], expected_b);
    }
}

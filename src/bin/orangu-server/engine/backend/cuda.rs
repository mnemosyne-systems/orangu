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
//! AMD hardware to get right (see `doc/SERVER_ROADMAP.md`'s "Long prompts
//! can hang the GPU" entry for one example of a bug only real hardware
//! testing surfaced); none of that exists here, and none of it can be
//! verified on a machine with no NVIDIA GPU. `CudaBackend::as_vulkan`
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
use crate::engine::quant::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

use super::{Backend, MatmulOp};

/// The `ggml_type`s a kernel exists for — the same set `engine::quant`
/// supports on the CPU path (see its module doc for what's missing).
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

/// Shared by every type's kernel: a manual (no vendor-specific `__half`
/// intrinsic) IEEE-754 binary16 -> float decoder, so the same source works
/// unmodified across CUDA/HIP/OpenCL — see `engine::backend::opencl`/
/// `engine::backend::rocm` for the (intentionally near-identical) sibling
/// copies of this prelude in each backend's own kernel language.
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

/// The compute entry point — a direct CUDA-C port of `vulkan_shaders`'s
/// `MAIN_REDUCE_SUFFIX`: one block per (output-row group of 4, token) pair,
/// all 64 threads splitting `in_dim` elements grid-stride style and
/// reducing their partial dot products in shared memory. Correct for any
/// `n_tokens` (unlike `VulkanBackend`'s separate cooperative/tiled paths,
/// this is the only dispatch strategy this backend has — see the module
/// doc comment for why those aren't ported here).
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

/// Each type's `dequant_element(weights, byte_offset, k)` — a line-for-line
/// restatement of its `engine::quant::dequantize_*` Rust counterpart (read
/// them side by side when changing either), just addressed per output
/// element `k` instead of sequentially over a whole block, the same
/// transformation `vulkan_shaders`'s `*_COOP_MIDDLE` constants apply. Byte
/// reads are direct (`weights[byte_offset]`) rather than WGSL's `read_u8`
/// word-unpacking trick — CUDA/HIP/OpenCL C don't share WGSL's storage
/// buffer 4-byte-alignment restriction.
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

/// The complete, compile-ready CUDA-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    let middle = dequant_element_source(ggml_type)?;
    Some(format!("{PRELUDE}\n{middle}\n{MAIN}"))
}

/// `QuantMatrix::cache_key()`'s return type — named, like `vulkan.rs`'s own
/// `WeightCacheKey`, so `weight_cache`'s type doesn't trip clippy's
/// `type_complexity` lint.
type WeightCacheKey = (usize, usize);

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    functions: HashMap<u32, CudaFunction>,
    /// Same identity-keyed reuse discipline as `VulkanBackend::weight_
    /// buffer`: a layer's weight tensor is uploaded to device memory once,
    /// not re-uploaded on every decode step.
    weight_cache: Mutex<HashMap<WeightCacheKey, Arc<CudaSlice<u8>>>>,
    /// The device's own name (e.g. `"NVIDIA GeForce RTX 4090"`) — for the
    /// startup banner.
    pub device_name: String,
}

impl CudaBackend {
    /// Looks for a usable CUDA device (ordinal 0) and compiles every
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
    pub fn try_init() -> Option<Self> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(Self::try_init_inner);
        std::panic::set_hook(previous_hook);
        result.ok().flatten()
    }

    fn try_init_inner() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
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

        Some(Self {
            stream,
            functions,
            weight_cache: Mutex::new(HashMap::new()),
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
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
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
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.expect("cuda kernel launch failed");

        self.stream
            .clone_dtoh(&y_dev)
            .expect("cuda y readback failed")
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

        let expected = CpuBackend.matmul(&x, n_tokens, &w);
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

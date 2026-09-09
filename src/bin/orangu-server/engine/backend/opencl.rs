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
//! Verified against real OpenCL hardware: an `ARM Platform` /
//! `Mali-G720-Immortalis` device reporting OpenCL 3.0, bound through
//! `/opt/cixgpu-pro/lib/aarch64-linux-gnu/libOpenCL.so` — the vendor
//! library itself, which is why there is no `/etc/OpenCL/vendors` ICD
//! registration to find. On a machine with no OpenCL device
//! [`OpenClBackend::try_init`] finds zero platforms and gracefully returns
//! `None`, and this module's cross-check tests skip in exactly that case,
//! per the same convention `vulkan.rs`/`cuda.rs` use.
//!
//! That distinction is not bookkeeping. This module was written and
//! reviewed while its cross-checks were skipping, and the first run on a
//! real device failed all 24 of them, on three defects that no amount of
//! reading had caught. `matmul` bound its weight argument as
//! `&Arc<Buffer<u8>>` rather than `&*weights`, and since
//! `ExecuteKernel::set_arg<T>` copies `size_of::<T>()` bytes straight from
//! the reference it handed the driver the `Arc`'s heap address in place of
//! the `cl_mem` handle — a `CL_INVALID_MEM_OBJECT` at launch, on every
//! single-op call, in production and not only under test. `matmul_batch`
//! had always derefed correctly, which is why the two paths disagreed.
//! And once launches succeeded, the cross-check below turned out not to be
//! checking much: it referenced `CpuBackend::matmul`, whose `int8`
//! activation rounding these kernels do not perform (see
//! `CpuBackend::matmul_dequant`, which says so, and which `vulkan.rs` and
//! `cuda.rs` already used), and it built weights from uniformly random
//! bytes rather than valid blocks (see `engine::backend::probe_blocks`).
//! All three are fixed; the 26 tests here now pass on this device.
//!
//! Apple targets are the one exception: their OpenCL ICD does report a
//! device, but segfaults on the first real matmul call, so `try_init`
//! refuses to initialize there at all rather than crash.
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

use super::device::{DeviceCandidate, DeviceClass};
use super::{Backend, MatmulOp};

// The kernels, the type list and the entry-point name are all
// `super::vendor_shaders`' - see that module for why they are shared
// rather than written once per backend.
use super::vendor_shaders::{KERNEL_NAME, SUPPORTED_TYPES};

/// The reduction workgroup size — must match `LOCAL_WORK_SIZE` used at
/// dispatch time and the hardcoded `64`/`partial_sums[256]` layout below,
/// the same relationship `vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` has to
/// `@workgroup_size(64)`.
const LOCAL_WORK_SIZE: usize = 64;

/// The complete, compile-ready OpenCL-C source for `ggml_type`'s matmul
/// kernel, or `None` if this backend has no kernel for it.
fn kernel_source(ggml_type: u32) -> Option<String> {
    super::vendor_shaders::kernel_source(super::vendor_shaders::Dialect::OpenCl, ggml_type)
}

/// `QuantMatrix::cache_key()`'s return type — named, like `vulkan.rs`'s own
/// `WeightCacheKey`, so `weight_cache`'s type doesn't trip clippy's
/// `type_complexity` lint.
type WeightCacheKey = (usize, usize);

/// Arm's kbase driver node. See [`OpenClBackend::vendor_device_is_openable`]
/// for why this module knows about it at all.
#[cfg(target_os = "linux")]
const MALI_DEVICE: &str = "/dev/mali0";

pub struct OpenClBackend {
    context: Context,
    queue: Mutex<CommandQueue>,
    kernels: Mutex<HashMap<u32, (Program, Kernel)>>,
    weight_cache: Mutex<HashMap<WeightCacheKey, Arc<Buffer<u8>>>>,
    /// The `IQ*` codebooks, uploaded once at init — see
    /// `engine::iq_grids::packed`.
    iq_grids: Buffer<u32>,
    /// The device's own name — for the startup banner.
    pub device_name: String,
}

impl OpenClBackend {
    /// Looks for the first GPU-type OpenCL device ([`Self::try_init_index`]
    /// names another) and builds every
    /// supported quant type's kernel up front. Returns `None` (never
    /// panics) if no OpenCL platform/device is found, or compilation
    /// otherwise fails — callers fall back to `CpuBackend`, the same
    /// contract `VulkanBackend`/`CudaBackend::try_init` have.
    ///
    /// **Tests only** — see `VulkanBackend::try_init`. `select_backend`
    /// goes through [`Self::devices`] and [`Self::try_init_index`] so the
    /// operator's `[orangu-server].device` is honoured and the device list
    /// is reported.
    #[cfg(test)]
    pub fn try_init() -> Option<Self> {
        Self::try_init_index(0)
    }

    /// Every GPU-type OpenCL device across every installed platform, in
    /// `clGetDeviceIDs` order — the order an index names.
    ///
    /// Only `CL_DEVICE_TYPE_GPU`, matching what [`Self::try_init_index`]
    /// will bind: a CPU-type OpenCL device is a software path this engine
    /// has a faster answer for, and an accelerator is not something these
    /// kernels are written against.
    ///
    /// Class comes from `CL_DEVICE_HOST_UNIFIED_MEMORY`, which is what
    /// OpenCL offers in place of a device-type distinction — a device
    /// sharing the host's memory is an iGPU/APU by any other name.
    pub fn devices() -> Vec<DeviceCandidate> {
        if cfg!(target_vendor = "apple") {
            return Vec::new();
        }
        if !Self::vendor_device_is_openable() {
            return Vec::new();
        }
        let Ok(ids) = get_all_devices(CL_DEVICE_TYPE_GPU) else {
            return Vec::new();
        };
        ids.into_iter()
            .enumerate()
            .map(|(index, id)| {
                let device = Device::new(id);
                DeviceCandidate {
                    index,
                    name: device
                        .name()
                        .unwrap_or_else(|_| format!("OpenCL device {index}")),
                    class: match device.host_unified_memory() {
                        Ok(false) => DeviceClass::Discrete,
                        Ok(true) => DeviceClass::Integrated,
                        // Not "assume discrete": an ICD that won't answer
                        // this has told us nothing, and `Other` is the class
                        // that says so.
                        Err(_) => DeviceClass::Other,
                    },
                    vram_total_bytes: device.global_mem_size().ok().filter(|size| *size > 0),
                    id: None,
                    driver: device.version().ok(),
                }
            })
            .collect()
    }

    /// Whether the vendor driver's own device node can be opened — checked
    /// **before** any OpenCL call, because on one real driver the answer to
    /// "no" is a segmentation fault rather than an error.
    ///
    /// Arm's Mali userspace blob (`libmali.so.1`, symlinked as
    /// `libOpenCL.so.1` on an RK3588) prints `failed to open device file
    /// /dev/mali0 with errno 13 (Permission denied)` and then crashes inside
    /// `get_all_devices` — taking `orangu-server` down during backend
    /// selection, before it has served anything. A server run by a user who
    /// is not in the `video` group is exactly that case, and it is the
    /// ordinary case on a fresh install: the blob is system-wide, the group
    /// membership is per user.
    ///
    /// Deliberately narrow. It guards only the node whose absence-of-a-check
    /// was observed to crash, and only when that node **exists** — on any
    /// machine without `/dev/mali0` this returns `true` and nothing about the
    /// existing AMD/Intel/NVIDIA paths changes. That is the same shape as this
    /// module's other carve-out, the Apple ICD that reports a device and then
    /// segfaults on first use: a known-crashing driver is refused by name
    /// rather than discovered the hard way.
    ///
    /// Opened read-write and immediately dropped, because read-only succeeds
    /// on a node the driver still cannot use.
    #[cfg(target_os = "linux")]
    fn vendor_device_is_openable() -> bool {
        let mali = std::path::Path::new(MALI_DEVICE);
        !mali.exists()
            || std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(mali)
                .is_ok()
    }

    /// No kbase driver anywhere but Linux, so nothing to guard.
    #[cfg(not(target_os = "linux"))]
    fn vendor_device_is_openable() -> bool {
        true
    }

    /// [`Self::try_init`] against a specific device in [`Self::devices`]'s
    /// order.
    pub fn try_init_index(index: usize) -> Option<Self> {
        // Both entry points, not just `devices`: `select_backend` calls them
        // separately, and a crash in either is the same crash.
        if !Self::vendor_device_is_openable() {
            return None;
        }
        // Apple's own OpenCL ICD reports a device but segfaults inside
        // clSetKernelArg on the first real matmul call (confirmed on Apple
        // Silicon). Refuse to initialize there rather than crash.
        if cfg!(target_vendor = "apple") {
            return None;
        }
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU).ok()?.get(index)?;
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

        // The `IQ*` lattice codebooks, uploaded once and bound to every
        // launch — see `vendor_shaders::iq_grid_prelude` for why every
        // kernel takes the pointer whether or not it reads one.
        let grid_words = crate::engine::iq_grids::packed::words();
        let mut iq_grids = unsafe {
            Buffer::<u32>::create(
                &context,
                CL_MEM_READ_ONLY,
                grid_words.len(),
                ptr::null_mut(),
            )
        }
        .ok()?;
        unsafe { queue.enqueue_write_buffer(&mut iq_grids, CL_BLOCKING, 0, &grid_words, &[]) }
            .ok()?;

        Some(Self {
            context,
            queue: Mutex::new(queue),
            kernels: Mutex::new(kernels),
            weight_cache: Mutex::new(HashMap::new()),
            iq_grids,
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
    /// See [`Backend::reduced_surface`] — this backend implements
    /// [`Backend::matmul`] and nothing else.
    fn reduced_surface(&self) -> Option<&'static str> {
        Some(super::MATMUL_ONLY_SURFACE)
    }

    fn supports_type(&self, ggml_type: u32) -> bool {
        SUPPORTED_TYPES.contains(&ggml_type)
    }

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
                .set_arg(&*weights)
                .set_arg(&x_buf)
                .set_arg(&y_buf)
                .set_arg(&in_dim)
                .set_arg(&out_dim)
                .set_arg(&n_tokens_u32)
                .set_arg(&row_bytes)
                .set_arg(&self.iq_grids)
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

    /// Every op enqueued before any is read back — see
    /// [`crate::engine::backend::CudaBackend::matmul_batch`] for the shape and
    /// why it is the same one here. `matmul` ends in `read_event.wait()`, so
    /// the default one-`matmul`-per-op implementation blocks the host once
    /// per op with the device idle across each wait.
    ///
    /// Two OpenCL-specific notes. The command queue lock is taken **once per
    /// chunk** rather than once per op, which is itself a serialization this
    /// used to pay on every call. And every weight buffer is resolved
    /// *before* that lock is taken: `weight_buffer` takes the same lock to
    /// upload a weight it has not seen, and `Mutex` here is not reentrant, so
    /// resolving one inside the chunk would deadlock on the first cache miss.
    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        use crate::engine::backend::{BATCH_DEVICE_BUDGET_BYTES, plan_batch};

        let (stripes, chunks) = plan_batch(ops, BATCH_DEVICE_BUDGET_BYTES);
        let mut out: Vec<Vec<f32>> = ops
            .iter()
            .map(|op| Vec::with_capacity(op.n_tokens * op.w.out_dim))
            .collect();
        for chunk in chunks {
            let chunk = &stripes[chunk];
            // Before the queue lock. See this function's doc comment.
            let weights: Vec<Arc<Buffer<u8>>> = chunk
                .iter()
                .map(|stripe| self.weight_buffer(ops[stripe.op].w))
                .collect();

            let queue = self.queue.lock().expect("opencl queue poisoned");
            let kernels = self.kernels.lock().expect("opencl kernels poisoned");
            // `x` and `y` both stay alive until every read below has
            // completed: the kernel reads one and writes the other, and
            // neither has run yet at the point the enqueue call returns.
            let mut buffers: Vec<(Buffer<f32>, Buffer<f32>, opencl3::event::Event)> =
                Vec::with_capacity(chunk.len());
            for (stripe, weights) in chunk.iter().zip(&weights) {
                let op = &ops[stripe.op];
                let in_dim = op.w.in_dim as u32;
                let out_dim = op.w.out_dim as u32;
                let row_bytes = op.w.row_bytes() as u32;
                let n_tokens_u32 = stripe.n_tokens as u32;
                let rows =
                    stripe.start * op.w.in_dim..(stripe.start + stripe.n_tokens) * op.w.in_dim;
                let x = &op.x[rows];
                let y_len = stripe.n_tokens * op.w.out_dim;

                let n_row_groups = (out_dim as usize).div_ceil(4);
                let num_groups = (n_row_groups * stripe.n_tokens).max(1);
                let global_work_size = num_groups * LOCAL_WORK_SIZE;

                let mut x_buf = unsafe {
                    Buffer::<f32>::create(
                        &self.context,
                        CL_MEM_READ_ONLY,
                        x.len().max(1),
                        ptr::null_mut(),
                    )
                }
                .expect("opencl x buffer allocation failed");
                // Blocking, as in `matmul`: this waits for a host-to-device
                // copy, not for a kernel, so it does not serialize the batch
                // against the device.
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

                let (_program, kernel) = kernels.get(&op.w.ggml_type()).unwrap_or_else(|| {
                    panic!(
                        "ggml_type {} reached OpenClBackend::matmul_batch without a \
                         compiled kernel (QuantMatrix construction should have rejected \
                         it earlier)",
                        op.w.ggml_type()
                    )
                });
                let kernel_event = unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(&**weights)
                        .set_arg(&x_buf)
                        .set_arg(&y_buf)
                        .set_arg(&in_dim)
                        .set_arg(&out_dim)
                        .set_arg(&n_tokens_u32)
                        .set_arg(&row_bytes)
                        .set_arg(&self.iq_grids)
                        .set_local_work_size(LOCAL_WORK_SIZE)
                        .set_global_work_size(global_work_size)
                        .enqueue_nd_range(&queue)
                }
                .expect("opencl kernel launch failed");
                buffers.push((x_buf, y_buf, kernel_event));
            }
            drop(kernels);

            let mut hosts: Vec<Vec<f32>> = chunk
                .iter()
                .map(|stripe| vec![0f32; stripe.n_tokens * ops[stripe.op].w.out_dim])
                .collect();
            let mut reads = Vec::with_capacity(chunk.len());
            for (host, (_x, y_buf, kernel_event)) in hosts.iter_mut().zip(&buffers) {
                reads.push(
                    unsafe {
                        queue.enqueue_read_buffer(
                            y_buf,
                            CL_NON_BLOCKING,
                            0,
                            host,
                            &[kernel_event.get()],
                        )
                    }
                    .expect("opencl y readback enqueue failed"),
                );
            }
            for read in &reads {
                read.wait().expect("opencl y readback failed");
            }
            drop(queue);

            for (stripe, host) in chunk.iter().zip(hosts) {
                out[stripe.op].extend(host);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::CpuBackend;
    use crate::engine::backend::probe_blocks::build_block;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{
        GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    };

    /// One `OpenClBackend`, lazily built and shared across every test in
    /// this module — see `cuda::tests::shared_cuda`'s doc comment for the
    /// identical rationale. Where no OpenCL device is present `try_init()`
    /// returns `None` and every test below skips; on this project's dev
    /// machine it binds the Mali device named in this module's own doc
    /// comment and they all run.
    fn shared_opencl() -> Option<&'static OpenClBackend> {
        static OPENCL: std::sync::OnceLock<Option<OpenClBackend>> = std::sync::OnceLock::new();
        OPENCL.get_or_init(OpenClBackend::try_init).as_ref()
    }

    /// Straight from `engine::quant`, which is where a block layout is
    /// defined. This module used to keep its own copy of that table; so did
    /// the other two vendor backends, and a type added to `quant` and to
    /// `vendor_shaders` but not to all three copies would fail here as
    /// `unreachable!()` rather than as a missing kernel.
    fn block_elems_for(ggml_type: u32) -> usize {
        crate::engine::quant::block_layout(ggml_type)
            .expect("a type under cross-check has a block layout")
            .1
    }

    /// Cross-checks `OpenClBackend::matmul` against `CpuBackend::matmul`
    /// for every supported `ggml_type` — the same methodology `vulkan.rs`/
    /// `cuda.rs` use. Skips (doesn't fail) when no OpenCL device is
    /// available, per `shared_opencl`'s doc comment.
    fn cross_check(ggml_type: u32, in_dim: usize, out_dim: usize, n_tokens: usize) {
        let Some(opencl) = shared_opencl() else {
            return;
        };
        let block_elems = block_elems_for(ggml_type);
        assert!(in_dim.is_multiple_of(block_elems));
        let mut seed = 0x1234_5678_9abc_def0u64
            ^ (ggml_type as u64) << 32
            ^ (in_dim as u64) << 16
            ^ out_dim as u64;
        let bytes: Vec<u8> = (0..out_dim * (in_dim / block_elems))
            .flat_map(|_| build_block(ggml_type, &mut seed))
            .collect();
        let w = test_quant_matrix(&bytes, ggml_type, in_dim, out_dim);
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        let expected = CpuBackend.matmul_dequant(&x, n_tokens, &w);
        let actual = opencl.matmul(&x, n_tokens, &w);
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-2 * e.abs().max(1.0),
                "index {i}: expected {e}, got {a} (ggml_type {ggml_type}, n_tokens {n_tokens})"
            );
        }
    }

    /// **Enumerating devices must not take the process down.** This is a
    /// regression test for a real crash, not a smoke test: with Arm's Mali
    /// blob installed as the system `libOpenCL.so`, a process that cannot open
    /// `/dev/mali0` — any user outside the `video` group — segfaulted inside
    /// `get_all_devices` during backend selection, before the server had
    /// answered anything. A segmentation fault fails this test by killing the
    /// runner, which is the only way a crash can be asserted against.
    ///
    /// It runs on every machine, including ones with no OpenCL at all, since
    /// "returns an empty list" is the correct answer there and the crash being
    /// guarded is in the enumeration itself.
    #[test]
    fn enumerating_devices_never_crashes_however_the_driver_feels() {
        let _ = OpenClBackend::devices();
        let _ = OpenClBackend::devices();
    }

    /// The device list and bring-up must agree, because `select_backend`
    /// treats a listed device that then fails to initialize as a hard error
    /// rather than a reason to try the next backend. The guard that skips an
    /// unopenable driver has to be on both or it creates exactly that
    /// mismatch.
    #[test]
    fn the_device_list_and_bring_up_agree_about_this_machine() {
        assert_eq!(
            OpenClBackend::devices().is_empty(),
            shared_opencl().is_none(),
            "devices() and try_init() must not disagree"
        );
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
        let Some(opencl) = shared_opencl() else {
            return;
        };
        let mut seed = 42u64;
        // Valid blocks, not random bytes — see `probe_blocks`. This test
        // compares the device against *itself*, so a `NaN` weight would
        // fail it just as surely: `assert_eq!` on two `NaN`s is false.
        let bytes_a: Vec<u8> = (0..8)
            .flat_map(|_| build_block(GGML_TYPE_Q4_K, &mut seed))
            .collect();
        let wa = test_quant_matrix(&bytes_a, GGML_TYPE_Q4_K, 256, 8);
        let bytes_b: Vec<u8> = (0..5)
            .flat_map(|_| build_block(GGML_TYPE_F32, &mut seed))
            .collect();
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

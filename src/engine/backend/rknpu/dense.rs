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

//! Dense `C[m, n] = A[m, k] · B[k, n]` products on the RKNPU with both
//! operands supplied per call, for work whose operands all change between
//! calls (training).
//!
//! `float16 × float16 → float32`, normal layouts. A context is made per shape
//! and per NPU core, each pinned to its core when the runtime can pin, so
//! products of one shape issued from several threads run on separate cores.
//! `A` and `C` stay bound to a context for its life; `B` is written and bound
//! again on every call, since binding is where the runtime converts it into
//! the device's own layout.
//!
//! `K` must be a multiple of [`Dense::K_ALIGN`] and `N` of [`Dense::N_ALIGN`];
//! a caller pads both with zeros. Every context, and the device as a whole at
//! [`Dense::open`], is checked against a host product before it is used.

use std::collections::HashMap;
use std::ffi::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use half::f16;
use libloading::Library;

use super::{
    Api, MatmulInfo, MatmulIoAttr, MatmulShape, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, RKNN_SUCC,
    RKNN_TENSOR_FLOAT16, RKNN_TENSOR_FLOAT32, Residency, RknnContext, TensorMem, budget_bytes,
    device_node_is_openable, open_runtime, rknpu_info, zeroed_attr,
};

/// `rknn_matmul_set_core_mask`, which older runtimes lack.
type CoreMask = unsafe extern "C" fn(RknnContext, c_int) -> c_int;

/// One context of one shape, and the device memory bound to it.
struct Context {
    api: Api,
    ctx: RknnContext,
    attr: MatmulIoAttr,
    a: *mut TensorMem,
    b: *mut TensorMem,
    c: *mut TensorMem,
    bytes: u64,
    residency: Arc<Residency>,
}

// SAFETY: the pointers are device memory this context owns, used only while
// its `Mutex` is held.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: each pointer came from `create_mem` on this context, which
        // is destroyed last.
        unsafe {
            for mem in [self.a, self.b, self.c] {
                if !mem.is_null() {
                    (self.api.destroy_mem)(self.ctx, mem);
                }
            }
            (self.api.destroy)(self.ctx);
        }
        self.residency.release(self.bytes);
    }
}

/// The contexts of one shape, one per core at most.
#[derive(Default)]
struct Shape {
    contexts: Vec<Arc<Mutex<Context>>>,
    next: AtomicUsize,
}

/// `(m, k, n)`.
type ShapeKey = (usize, usize, usize);

/// The RKNPU as a device for dense fp16 products.
pub struct Dense {
    _library: Arc<Library>,
    api: Api,
    core_mask: Option<CoreMask>,
    cores: usize,
    residency: Arc<Residency>,
    shapes: Mutex<HashMap<ShapeKey, Arc<Mutex<Shape>>>>,
}

// SAFETY: `Api` and `CoreMask` are function pointers into a library this
// value keeps loaded; every context is behind its own `Mutex`.
unsafe impl Send for Dense {}
unsafe impl Sync for Dense {}

impl Dense {
    /// `K`'s alignment, in elements.
    pub const K_ALIGN: usize = super::K_ALIGN_ELEMS;
    /// `N`'s alignment in fp16, in elements.
    pub const N_ALIGN: usize = 16;

    /// The machine's RKNPU, or `None` when there is none, its node cannot be
    /// opened, the runtime does not load, or a first product disagrees with
    /// the host.
    pub fn open() -> Option<Self> {
        let info = rknpu_info()?;
        if !device_node_is_openable(&info) {
            return None;
        }
        let (library, api, _) = open_runtime()?;
        // SAFETY: the signature matches `rknn_matmul_api.h`.
        let core_mask = unsafe {
            library
                .get::<CoreMask>(b"rknn_matmul_set_core_mask\0")
                .ok()
                .map(|f| *f)
        };
        let dense = Self {
            _library: library,
            api,
            core_mask,
            cores: if core_mask.is_some() {
                (info.cores as usize).clamp(1, 3)
            } else {
                1
            },
            residency: Arc::new(Residency::new(budget_bytes())),
            shapes: Mutex::new(HashMap::new()),
        };
        let (m, k, n) = (5, 2 * Self::K_ALIGN, 2 * Self::N_ALIGN);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.125)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        dense.agrees(m, k, n, &a, &b).then_some(dense)
    }

    /// NPU cores products are spread over.
    pub fn cores(&self) -> usize {
        self.cores
    }

    /// Device memory held, in bytes.
    pub fn resident_bytes(&self) -> u64 {
        *self
            .residency
            .used
            .lock()
            .expect("rknpu residency poisoned")
    }

    /// `C = A · B`: `fill` writes `A` (`m × k`, row-major) and `B`
    /// (`k × n`, row-major) in fp16, then `read` receives `C` (`m × n`,
    /// row-major, fp32). `false`, with `read` not called, when the device
    /// declines the shape, has no memory left for it, or fails the call; the
    /// caller then computes the product itself.
    pub fn product(
        &self,
        m: usize,
        k: usize,
        n: usize,
        fill: impl FnOnce(&mut [f16], &mut [f16]),
        read: impl FnOnce(&[f32]),
    ) -> bool {
        if m == 0
            || k == 0
            || n == 0
            || !k.is_multiple_of(Self::K_ALIGN)
            || !n.is_multiple_of(Self::N_ALIGN)
        {
            return false;
        }
        let Some(context) = self.context(m, k, n) else {
            return false;
        };
        let mut context = context.lock().expect("rknpu context poisoned");
        context.run(m, k, n, fill, read)
    }

    /// A context of this shape: a free one, a new one while there are cores
    /// without one, or the next in turn.
    fn context(&self, m: usize, k: usize, n: usize) -> Option<Arc<Mutex<Context>>> {
        let shape = self
            .shapes
            .lock()
            .expect("rknpu shapes poisoned")
            .entry((m, k, n))
            .or_default()
            .clone();
        let mut shape = shape.lock().expect("rknpu shape poisoned");
        if let Some(free) = shape.contexts.iter().find(|c| c.try_lock().is_ok()) {
            return Some(free.clone());
        }
        if shape.contexts.len() < self.cores {
            let core = shape.contexts.len();
            if let Some(made) = self.make(m, k, n, core) {
                let made = Arc::new(Mutex::new(made));
                shape.contexts.push(made.clone());
                return Some(made);
            }
        }
        if shape.contexts.is_empty() {
            return None;
        }
        let turn = shape.next.fetch_add(1, Ordering::Relaxed) % shape.contexts.len();
        Some(shape.contexts[turn].clone())
    }

    /// A context for one shape on core `core`, checked against the host.
    fn make(&self, m: usize, k: usize, n: usize, core: usize) -> Option<Context> {
        let mut info = MatmulInfo {
            m: m as i32,
            k: k as i32,
            n: n as i32,
            kind: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            b_layout: 0,
            b_quant_type: 0,
            ac_layout: 0,
            ac_quant_type: 0,
            iommu_domain_id: 0,
            group_size: 0,
            reserved: [0; 34],
        };
        let mut shape = MatmulShape {
            m: m as i32,
            k: k as i32,
            n: n as i32,
        };
        let mut attr = MatmulIoAttr {
            a: zeroed_attr(),
            b: zeroed_attr(),
            c: zeroed_attr(),
        };
        let mut ctx: RknnContext = 0;
        // SAFETY: one shape and one attribute set, both live across the call.
        let status = unsafe {
            (self.api.create_dynamic_shape)(&mut ctx, &mut info, 1, &mut shape, &mut attr)
        };
        if status != RKNN_SUCC || ctx == 0 {
            return None;
        }
        let bytes = u64::from(attr.a.size) + u64::from(attr.b.size) + u64::from(attr.c.size);
        let mut context = Context {
            api: self.api,
            ctx,
            attr,
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            bytes: 0,
            residency: self.residency.clone(),
        };
        if attr.a.kind != RKNN_TENSOR_FLOAT16
            || attr.b.kind != RKNN_TENSOR_FLOAT16
            || attr.c.kind != RKNN_TENSOR_FLOAT32
            || (attr.a.size as usize) < m * k * 2
            || (attr.b.size as usize) < k * n * 2
            || (attr.c.size as usize) < m * n * 4
        {
            return None;
        }
        if !self.residency.try_reserve(bytes) {
            return None;
        }
        context.bytes = bytes;
        // SAFETY: the context is live; each size is the runtime's own.
        unsafe {
            context.a = (self.api.create_mem)(ctx, attr.a.size);
            context.b = (self.api.create_mem)(ctx, attr.b.size);
            context.c = (self.api.create_mem)(ctx, attr.c.size);
        }
        if [context.a, context.b, context.c]
            .iter()
            .any(|mem| mem.is_null() || unsafe { (**mem).virt_addr.is_null() })
        {
            self.residency
                .note_device_refusal(std::io::Error::last_os_error());
            return None;
        }
        let mut a_attr = attr.a;
        let mut c_attr = attr.c;
        // SAFETY: every pointer belongs to this context.
        let bound = unsafe {
            (self.api.set_dynamic_shape)(ctx, &mut shape) == RKNN_SUCC
                && (self.api.set_io_mem)(ctx, context.a, &mut a_attr) == RKNN_SUCC
                && (self.api.set_io_mem)(ctx, context.c, &mut c_attr) == RKNN_SUCC
        };
        if !bound {
            return None;
        }
        if let Some(set_core) = self.core_mask {
            // SAFETY: the context is live. Cores are bits 0, 1 and 2.
            unsafe { set_core(ctx, 1 << core) };
        }
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + core) % 19) as f32 * 0.125 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 3) % 11) as f32 * 0.25 - 1.25)
            .collect();
        check(&mut context, m, k, n, &a, &b).then_some(context)
    }

    /// Whether one product of this shape matches the host.
    fn agrees(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> bool {
        let Some(context) = self.context(m, k, n) else {
            return false;
        };
        let mut context = context.lock().expect("rknpu context poisoned");
        check(&mut context, m, k, n, a, b)
    }
}

/// Runs `a · b` on `context` and compares a sample of the outputs with the
/// host's product of the same fp16 operands.
fn check(context: &mut Context, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> bool {
    let mut got = Vec::new();
    let ran = context.run(
        m,
        k,
        n,
        |a16, b16| {
            for (d, s) in a16.iter_mut().zip(a) {
                *d = f16::from_f32(*s);
            }
            for (d, s) in b16.iter_mut().zip(b) {
                *d = f16::from_f32(*s);
            }
        },
        |c| got.extend_from_slice(c),
    );
    if !ran {
        return false;
    }
    let rounded = |v: f32| f64::from(f32::from(f16::from_f32(v)));
    let rows = m.min(8);
    let columns = (n / 64).max(1);
    (0..rows).all(|i| {
        (0..n).step_by(columns).all(|j| {
            let want: f64 = (0..k)
                .map(|q| rounded(a[i * k + q]) * rounded(b[q * n + j]))
                .sum();
            (f64::from(got[i * n + j]) - want).abs() <= 1e-3 * want.abs().max(1.0)
        })
    })
}

impl Context {
    fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        fill: impl FnOnce(&mut [f16], &mut [f16]),
        read: impl FnOnce(&[f32]),
    ) -> bool {
        // SAFETY: `a` and `b` are this context's mappings, at least `m * k`
        // and `k * n` halves long (checked when made), and nothing else
        // touches them while `self` is borrowed mutably.
        let (a, b) = unsafe {
            (
                std::slice::from_raw_parts_mut((*self.a).virt_addr.cast::<f16>(), m * k),
                std::slice::from_raw_parts_mut((*self.b).virt_addr.cast::<f16>(), k * n),
            )
        };
        fill(a, b);
        let mut b_attr = self.attr.b;
        // SAFETY: `b` belongs to this context; binding it converts what was
        // just written into the device's layout.
        if unsafe { (self.api.set_io_mem)(self.ctx, self.b, &mut b_attr) } != RKNN_SUCC {
            return false;
        }
        // SAFETY: the context is live and every operand is bound.
        if unsafe { (self.api.run)(self.ctx) } != RKNN_SUCC {
            return false;
        }
        // SAFETY: `c` is this context's mapping, at least `m * n` floats,
        // written in full by the run. It is never written by the host: a host
        // write leaves cache lines that hide what the device puts in memory.
        let c = unsafe { std::slice::from_raw_parts((*self.c).virt_addr.cast::<f32>(), m * n) };
        read(c);
        true
    }
}

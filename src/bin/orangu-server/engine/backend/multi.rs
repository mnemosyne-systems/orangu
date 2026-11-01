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

//! One model across several devices, by routing each weight to the device
//! its layer was placed on.
//!
//! # How this avoids touching the model code
//!
//! Every architecture in `engine::arch` reaches its backend through
//! [`Backend::matmul`], whose operand is a `QuantMatrix` — and a
//! `QuantMatrix` knows which device it belongs to, because
//! `LoadedModel::matrix` stamped it from `engine::placement`'s plan when
//! the model was built. So the routing is a lookup, and not one line of any
//! forward pass changes.
//!
//! The cross-device transfer falls out of the same shape. `Backend::matmul`
//! takes host `&[f32]` and returns host `Vec<f32>`; a layer on device 0
//! therefore *already* ends with its output in host memory, and the next
//! layer's first matmul uploads from there to device 1. There is no
//! peer-to-peer path to write and no residual to shuttle by hand: the
//! boundary crossing is the trait's existing contract, paid once per
//! matmul rather than once per boundary.
//!
//! # What a spread model keeps, and what it gives up
//!
//! The distinction is between work scoped to *one layer* and work that
//! spans layers, and it is drawn by two hooks rather than one.
//!
//! **Kept** — everything per-layer, through [`Self::as_wgpu_on`]: fused
//! attention, the fused post-attention/FFN chain, the device-side KV
//! mirror. None of it needs cross-layer state; each takes host input,
//! returns host output, and touches only that layer's weights and that
//! layer's cache. It runs on the card the layer's weights are on.
//!
//! **Given up** — everything that spans layers, because [`Self::as_wgpu`]
//! answers `None`: the whole-step decode submission (which records every
//! layer into one command buffer and drops a decode step from ~37 GPU
//! submissions to 1), GPU sampling, the logits readback. Those assume one
//! `wgpu::Device` holds the whole chain, and handing one a model split
//! across two devices would be silently wrong.
//!
//! Measured on the dev machine, release build, a 0.5B model split 3:1 over
//! two GPUs: 11.9 tok/s with the per-layer paths off, 14.9 with them on,
//! against 27.8 unsplit. So per-layer fusion is worth about a quarter, and
//! the whole-step submission is most of what remains. What a split buys is
//! still **capacity** — a model larger than any one card runs at all,
//! instead of the driver paging VRAM on every token.
//!
//! The KV mirror is safe by construction rather than by review: a layer's
//! device never changes, `sync_gpu` is only ever called from inside a
//! `VulkanBackend` (so the mirror lands on that backend's own device), and
//! `LayerCache::copy_prefix_from` drops the mirror, so no buffer can
//! survive into a cache being reused elsewhere.
//!
//! # A host layer's projections, at prefill width
//!
//! A layer the plan left on the host runs its projections on the CPU — at
//! one token that is right, the weights are already in RAM and the card
//! would spend longer fetching them than multiplying. At prefill width it
//! is not: the same weights multiply every token of the batch, so sending
//! them across the bus once per call and running the GEMM on the card
//! costs the upload plus a fraction of the host GEMM. Measured on a 48-layer
//! model split 18:30 over a card and the host, a 257-token chunk spent
//! 540 ms per host layer in three projections against 200 ms per device
//! layer for the whole of attention and the FFN.
//!
//! So [`Self::matmul`] and [`Self::matmul_batch`] send a host layer's op to
//! the first card's **streaming region** ([`VulkanBackend::matmul_batch_streamed`])
//! once the batch is [`host_stream_min_tokens`] wide. The region is bounded
//! and recycled, so the card's residency plan is untouched; the result
//! comes back to the host exactly as the CPU's would, so nothing about the
//! layer's attention or cache changes. Decode never takes it:
//! [`Self::matmul_decode`] and [`Self::matmul_batch_decode`] route to the
//! weight's own device regardless of width, for the reason
//! [`Backend::matmul_decode`] exists at all.
//!
//! # Same backend, always
//!
//! Every device here comes from one API's own enumeration, so a set can
//! never mix a Vulkan device with a CUDA one. That is worth stating rather
//! than merely being true: two vendors' kernels sum in different orders,
//! and a model split across them would produce output that depends on
//! which layers landed where — reproducible only against itself.

use std::sync::Arc;

use super::vulkan::VulkanBackend;
use super::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;

/// Whether `ORANGU_NO_SPLIT_FUSION=1` asked for a split model's per-layer
/// work to stay off the GPU — the behaviour a split had before
/// [`Backend::as_wgpu_on`] existed, when every fused path was off and only
/// matmuls reached a device.
///
/// Two jobs, and the first is why it exists. **It makes the change
/// measurable from one binary**: a split run with and without it is the
/// only honest A/B for "did per-layer fusion help", and measuring it by
/// building a second binary is how a stale copy ends up being the thing
/// timed. Second, it is the escape hatch if a driver turns out to dislike
/// two devices recording fused chains in one process — the same
/// opt-out-of-a-default shape as `ORANGU_NO_KV_F16` and
/// `ORANGU_NO_TILED_PREFILL`.
fn no_split_fusion() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_SPLIT_FUSION"))
}

/// The narrowest batch at which a host layer's projection is streamed to a
/// card rather than run on the CPU — `ORANGU_HOST_STREAM_TOKENS`, default
/// [`HOST_STREAM_TOKENS_DEFAULT`]; `0` keeps every host layer on the host.
///
/// A width rather than a switch, because the crossover is one: below it
/// the upload outweighs the GEMM and above it the GEMM outweighs the
/// upload, and a sweep of the value is how the default was set.
pub(crate) fn host_stream_min_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        crate::engine::backend::tuning_profile::env_var("ORANGU_HOST_STREAM_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(HOST_STREAM_TOKENS_DEFAULT)
    })
}

/// Where the streamed projection overtakes the host GEMM.
const HOST_STREAM_TOKENS_DEFAULT: usize = 32;

/// A `Backend` that forwards each operation to the device holding its
/// weights.
pub struct MultiDeviceBackend {
    /// One backend per device in the selected set, in the same order
    /// `engine::placement`'s plan indexes them.
    devices: Vec<Arc<dyn Backend>>,
}

impl MultiDeviceBackend {
    /// Panics on fewer than two devices: a one-device "split" is a plan
    /// that `engine::placement` refuses to produce, and building this
    /// wrapper around a single backend would add a layer of indirection to
    /// every matmul while giving up `as_wgpu` — the fused paths — for
    /// nothing.
    pub fn new(devices: Vec<Arc<dyn Backend>>) -> Self {
        assert!(
            devices.len() > 1,
            "a split model needs more than one device (got {})",
            devices.len()
        );
        Self { devices }
    }

    /// The backend for `w`, clamped to the set.
    ///
    /// A weight tagged with a device that is not in this set means the
    /// placement plan and the backend set disagree, which is a bug — but
    /// one whose right response at run time is to compute the right answer
    /// on the wrong device rather than to panic mid-generation. The clamp
    /// is the safety net; `select_backend` builds both from the same plan
    /// so it should never fire.
    fn backend_for(&self, w: &QuantMatrix) -> &dyn Backend {
        let index = w.device().min(self.devices.len() - 1);
        self.devices[index].as_ref()
    }

    /// The card a host layer's wide projection streams to: the first
    /// device in the set with a streaming region, which is the card the
    /// plan filled first and so the one whose layers neighbour the host's.
    ///
    /// Not gated on [`no_split_fusion`]: that switch takes the *fused*
    /// per-layer chains off a split model, and a streamed projection is a
    /// plain matmul — the one kind of work that switch leaves on a device.
    fn streamer(&self) -> Option<&VulkanBackend> {
        self.devices.iter().find_map(|device| device.as_wgpu())
    }

    /// Whether `op` is a host layer's projection wide enough to stream —
    /// see the module doc. Never true for a weight that already lives on a
    /// card, and never for a type that card has no kernel for.
    fn streams(&self, op: &MatmulOp<'_>) -> bool {
        let min = host_stream_min_tokens();
        min > 0
            && op.n_tokens >= min
            && self.backend_for(op.w).as_wgpu().is_none()
            && self
                .streamer()
                .is_some_and(|card| card.supports_type(op.w.ggml_type()))
    }

    /// Runs `ops` grouped by device, so each device still sees its share as
    /// one batch.
    ///
    /// The grouping is what keeps a GPU backend's own submission batching
    /// alive across the split: `VulkanBackend::matmul_batch` submits one
    /// command buffer and blocks once for a whole group, and routing each
    /// op individually would turn a layer's Q/K/V projections back into
    /// three round trips.
    ///
    /// With `stream`, a host layer's ops wide enough to stream
    /// ([`Self::streams`]) form one more group, sent to the card's
    /// streaming region as one batch — a layer's Q/K/V upload once and
    /// submit once, the same as they would on their own device.
    fn batch_by_device(
        &self,
        ops: &[MatmulOp<'_>],
        stream: bool,
        run: impl Fn(&dyn Backend, &[MatmulOp<'_>]) -> Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let mut results: Vec<Option<Vec<f32>>> = (0..ops.len()).map(|_| None).collect();
        let streamed: Vec<usize> = if stream {
            (0..ops.len()).filter(|&i| self.streams(&ops[i])).collect()
        } else {
            Vec::new()
        };
        for device in 0..self.devices.len() {
            let mine: Vec<usize> = ops
                .iter()
                .enumerate()
                .filter(|(i, op)| {
                    op.w.device().min(self.devices.len() - 1) == device && !streamed.contains(i)
                })
                .map(|(i, _)| i)
                .collect();
            if mine.is_empty() {
                continue;
            }
            let group: Vec<MatmulOp<'_>> = mine
                .iter()
                .map(|&i| MatmulOp {
                    x: ops[i].x,
                    n_tokens: ops[i].n_tokens,
                    w: ops[i].w,
                })
                .collect();
            for (slot, out) in mine.iter().zip(run(self.devices[device].as_ref(), &group)) {
                results[*slot] = Some(out);
            }
        }
        if !streamed.is_empty() {
            let card = self
                .streamer()
                .expect("an op streams only when the set has a card to stream to");
            let group: Vec<MatmulOp<'_>> = streamed
                .iter()
                .map(|&i| MatmulOp {
                    x: ops[i].x,
                    n_tokens: ops[i].n_tokens,
                    w: ops[i].w,
                })
                .collect();
            for (slot, out) in streamed.iter().zip(card.matmul_batch_streamed(&group)) {
                results[*slot] = Some(out);
            }
        }
        results
            .into_iter()
            .map(|out| out.expect("every op belongs to exactly one device"))
            .collect()
    }
}

impl Backend for MultiDeviceBackend {
    fn matmul(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        let op = MatmulOp { x, n_tokens, w };
        if self.streams(&op) {
            return self
                .streamer()
                .expect("an op streams only when the set has a card to stream to")
                .matmul_batch_streamed(std::slice::from_ref(&op))
                .pop()
                .expect("matmul_batch_streamed returns exactly one result per input op");
        }
        self.backend_for(w).matmul(x, n_tokens, w)
    }

    fn matmul_batch(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.batch_by_device(ops, true, |backend, group| backend.matmul_batch(group))
    }

    /// Routed to each device's *decode* entry point, not its `matmul` —
    /// the distinction exists so a sequence's logits don't depend on how
    /// many other sequences were decoding beside it (see
    /// [`Backend::matmul_decode`]), and it would be lost by falling through
    /// to the default.
    fn matmul_decode(&self, x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
        self.backend_for(w).matmul_decode(x, n_tokens, w)
    }

    fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
        self.batch_by_device(ops, false, |backend, group| {
            backend.matmul_batch_decode(group)
        })
    }

    /// Any device takes it: the pool a result is recycled into is the
    /// process's, not a device's.
    fn recycle(&self, buffer: Vec<f32>) {
        self.devices[0].recycle(buffer);
    }

    fn take_scratch(&self, len: usize) -> Vec<f32> {
        self.devices[0].take_scratch(len)
    }

    fn split_device(&self, index: usize) -> Option<&dyn Backend> {
        self.devices.get(index).map(|device| device.as_ref())
    }

    /// Always `None` — see the module doc. This is the single line that
    /// keeps every *cross-layer* fused path off a split model: the
    /// whole-step decode submission, GPU sampling, the logits readback.
    /// Per-layer work goes through [`Self::as_wgpu_on`] instead and keeps
    /// its fused kernels.
    fn as_wgpu(&self) -> Option<&VulkanBackend> {
        None
    }

    /// The device holding this layer, so a per-layer fused chain runs on
    /// the same card its weights are on.
    ///
    /// Clamped for the same reason [`Self::backend_for`] is: a device tag
    /// outside the set means the plan and the backend set disagree, and the
    /// right run-time answer is the right result on the wrong device rather
    /// than a panic mid-generation.
    fn as_wgpu_on(&self, device: usize) -> Option<&VulkanBackend> {
        if no_split_fusion() {
            return None;
        }
        self.devices[device.min(self.devices.len() - 1)].as_wgpu()
    }

    /// A type is supported only if *every* device has a kernel for it. The
    /// plan can put any layer on any device, so a gap on one of them is a
    /// gap for the model.
    fn supports_type(&self, ggml_type: u32) -> bool {
        self.devices
            .iter()
            .all(|device| device.supports_type(ggml_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::loader::test_quant_matrix;
    use crate::engine::quant::{GGML_TYPE_F32, GGML_TYPE_Q4_K};
    use std::sync::Mutex;

    /// A backend that records what it was asked to do and answers with its
    /// own id, so a test can tell which device ran an op.
    struct Recording {
        id: f32,
        calls: Mutex<Vec<usize>>,
        decode_calls: Mutex<Vec<usize>>,
        missing_type: Option<u32>,
    }

    impl Recording {
        fn new(id: f32) -> Arc<Self> {
            Arc::new(Self {
                id,
                calls: Mutex::new(Vec::new()),
                decode_calls: Mutex::new(Vec::new()),
                missing_type: None,
            })
        }

        fn without(id: f32, ggml_type: u32) -> Arc<Self> {
            Arc::new(Self {
                id,
                calls: Mutex::new(Vec::new()),
                decode_calls: Mutex::new(Vec::new()),
                missing_type: Some(ggml_type),
            })
        }
    }

    impl Backend for Recording {
        fn matmul(&self, _x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
            self.calls.lock().unwrap().push(n_tokens);
            vec![self.id; n_tokens * w.out_dim]
        }

        fn matmul_decode(&self, _x: &[f32], n_tokens: usize, w: &QuantMatrix) -> Vec<f32> {
            self.decode_calls.lock().unwrap().push(n_tokens);
            vec![self.id + 100.0; n_tokens * w.out_dim]
        }

        /// Overridden so the stub actually *has* a distinct batch-decode
        /// path. Without this the trait's own default would route batch
        /// decode down to `matmul`, and a test asserting the wrapper picked
        /// the decode entry point would pass whether it did or not.
        fn matmul_batch_decode(&self, ops: &[MatmulOp<'_>]) -> Vec<Vec<f32>> {
            ops.iter()
                .map(|op| self.matmul_decode(op.x, op.n_tokens, op.w))
                .collect()
        }

        fn supports_type(&self, ggml_type: u32) -> bool {
            self.missing_type != Some(ggml_type)
        }
    }

    fn matrix(device: usize) -> QuantMatrix {
        let mut m = test_quant_matrix(&[0; 4 * 3 * 2], GGML_TYPE_F32, 3, 2);
        m.set_device(device);
        m
    }

    #[test]
    fn a_matmul_runs_on_the_device_its_weight_was_placed_on() {
        let zero = Recording::new(0.0);
        let one = Recording::new(1.0);
        let multi = MultiDeviceBackend::new(vec![zero.clone(), one.clone()]);

        assert_eq!(multi.matmul(&[0.0; 3], 1, &matrix(0))[0], 0.0);
        assert_eq!(multi.matmul(&[0.0; 3], 1, &matrix(1))[0], 1.0);
        assert_eq!(zero.calls.lock().unwrap().len(), 1);
        assert_eq!(one.calls.lock().unwrap().len(), 1);
    }

    /// The decode entry point must not collapse into `matmul` on the way
    /// through: the two use different kernels, and mixing them would make a
    /// sequence's output depend on its batch.
    #[test]
    fn decode_routes_to_the_devices_decode_path() {
        let zero = Recording::new(0.0);
        let one = Recording::new(1.0);
        let multi = MultiDeviceBackend::new(vec![zero.clone(), one.clone()]);

        assert_eq!(multi.matmul_decode(&[0.0; 3], 1, &matrix(1))[0], 101.0);
        assert!(one.calls.lock().unwrap().is_empty());
        assert_eq!(one.decode_calls.lock().unwrap().len(), 1);
        assert!(zero.decode_calls.lock().unwrap().is_empty());
    }

    /// A batch spanning both devices comes back **in the caller's order**,
    /// not in device order — the results are positional and a reordering
    /// would silently swap a layer's Q and V projections.
    #[test]
    fn a_split_batch_returns_results_in_the_order_they_were_given() {
        let zero = Recording::new(0.0);
        let one = Recording::new(1.0);
        let multi = MultiDeviceBackend::new(vec![zero.clone(), one.clone()]);

        let x = [0.0f32; 3];
        let (a, b, c) = (matrix(1), matrix(0), matrix(1));
        let out = multi.matmul_batch(&[
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &a,
            },
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &b,
            },
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &c,
            },
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0][0], 1.0);
        assert_eq!(out[1][0], 0.0);
        assert_eq!(out[2][0], 1.0);
        // And each device saw its own ops as one batch, not one at a time.
        assert_eq!(one.calls.lock().unwrap().len(), 2);
        assert_eq!(zero.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_batch_decode_also_keeps_order_and_uses_the_decode_path() {
        let zero = Recording::new(0.0);
        let one = Recording::new(1.0);
        let multi = MultiDeviceBackend::new(vec![zero.clone(), one.clone()]);

        let x = [0.0f32; 3];
        let (a, b) = (matrix(1), matrix(0));
        let out = multi.matmul_batch_decode(&[
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &a,
            },
            MatmulOp {
                x: &x,
                n_tokens: 1,
                w: &b,
            },
        ]);
        assert_eq!(out[0][0], 101.0);
        assert_eq!(out[1][0], 100.0);
    }

    /// A weight tagged for a device that isn't in the set must still be
    /// computed, on the last device, rather than panicking mid-generation.
    #[test]
    fn an_out_of_range_device_tag_is_clamped_rather_than_fatal() {
        let zero = Recording::new(0.0);
        let one = Recording::new(1.0);
        let multi = MultiDeviceBackend::new(vec![zero, one.clone()]);
        assert_eq!(multi.matmul(&[0.0; 3], 1, &matrix(7))[0], 1.0);
        assert_eq!(one.calls.lock().unwrap().len(), 1);
    }

    /// A type only one device can run is not a type the model can run: the
    /// plan may put any layer anywhere.
    #[test]
    fn a_type_missing_on_one_device_is_unsupported_for_the_set() {
        let multi = MultiDeviceBackend::new(vec![
            Recording::new(0.0),
            Recording::without(1.0, GGML_TYPE_Q4_K),
        ]);
        assert!(multi.supports_type(GGML_TYPE_F32));
        assert!(!multi.supports_type(GGML_TYPE_Q4_K));
    }

    /// No *cross-layer* fused path may ever see a split model: the
    /// whole-step decode submission, GPU sampling and the logits readback
    /// all span layers, and so would span devices.
    #[test]
    fn a_split_model_never_exposes_a_wgpu_backend() {
        let multi = MultiDeviceBackend::new(vec![Recording::new(0.0), Recording::new(1.0)]);
        assert!(multi.as_wgpu().is_none());
    }

    /// Per-layer work does keep its fused kernels, on the layer's own
    /// device — and `as_wgpu_on` must ask *that* device, not the head.
    ///
    /// `Recording` is not a `wgpu` backend, so both answers here are
    /// `None`; what this pins is which backend was consulted, which is the
    /// part that would send a layer's fused chain to the wrong card.
    #[test]
    fn per_layer_work_asks_the_layer_s_own_device() {
        struct Probe(Mutex<Vec<usize>>, usize);
        impl Backend for Probe {
            fn matmul(&self, _x: &[f32], _n: usize, _w: &QuantMatrix) -> Vec<f32> {
                unreachable!("this probe only answers `as_wgpu`")
            }
            fn as_wgpu(&self) -> Option<&VulkanBackend> {
                self.0.lock().unwrap().push(self.1);
                None
            }
        }
        let zero = Arc::new(Probe(Mutex::new(Vec::new()), 0));
        let one = Arc::new(Probe(Mutex::new(Vec::new()), 1));
        let multi = MultiDeviceBackend::new(vec![zero.clone(), one.clone()]);

        assert!(multi.as_wgpu_on(1).is_none());
        assert!(zero.0.lock().unwrap().is_empty(), "device 0 was not asked");
        assert_eq!(*one.0.lock().unwrap(), vec![1]);

        assert!(multi.as_wgpu_on(0).is_none());
        assert_eq!(*zero.0.lock().unwrap(), vec![0]);

        // Out of range clamps rather than panicking mid-generation, same as
        // the matmul route.
        assert!(multi.as_wgpu_on(9).is_none());
        assert_eq!(*one.0.lock().unwrap(), vec![1, 1]);
    }

    /// A single-device backend answers the same thing whatever layer asks —
    /// which is what keeps the unsplit path byte-for-byte what it was.
    #[test]
    fn a_single_device_backend_ignores_the_layer_s_device() {
        let solo = Recording::new(0.0);
        assert!(solo.as_wgpu_on(0).is_none());
        assert!(solo.as_wgpu_on(7).is_none());
    }

    #[test]
    #[should_panic(expected = "more than one device")]
    fn one_device_is_not_a_split() {
        MultiDeviceBackend::new(vec![Recording::new(0.0)]);
    }
}

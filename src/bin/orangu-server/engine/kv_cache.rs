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

//! Per-sequence KV cache: one `[capacity, n_head_kv * head_dim]` buffer per
//! layer for keys and for values, appended to one token at a time as a
//! sequence is prefilled/decoded. Each request/slot owns one `KvCache` —
//! there is no cross-sequence sharing (no prompt-prefix reuse) in this
//! build.

/// Converts a slice of `f32` KV values into little-endian `f16` bytes, for
/// `LayerCache::sync_gpu`'s `f16` KV-mirror upload path. A plain
/// per-element loop, not `bytemuck::cast_slice` — unlike the `f32` path,
/// this genuinely *converts* values, not just reinterprets bytes.
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Converts a slice of `f32` KV values into the
/// [`crate::engine::backend::vulkan_shaders::KvStorage::Q8_0`] byte
/// layout, for `LayerCache::sync_gpu`'s CPU-side upload path — the
/// standalone (non-fused) `gpu_attention`/test entry points; the fused
/// decode hot path quantizes on the GPU instead
/// (`KV_QUANTIZE_Q8_0_SHADER`). `data.len()` must be a multiple of 32
/// (`KvStorage::Q8_0`'s own doc comment covers why this is always true in
/// practice for real GQA-shaped models). 36 bytes per 32-element block — a
/// plain little-endian `f32` scale followed by 32 signed-byte quants —
/// deliberately produces the *exact* same bytes the GPU quantize shader
/// does (both compute `amax`, `d = amax / 127`, `round(v / d)` identically,
/// and GPU storage buffers are little-endian on every platform this
/// backend targets), so a cross-check test can compare either path's
/// output directly.
fn f32_to_q8_0_bytes(data: &[f32]) -> Vec<u8> {
    debug_assert_eq!(
        data.len() % 32,
        0,
        "q8_0 KV storage requires kv_dim to be a multiple of 32"
    );
    let mut out = Vec::with_capacity(data.len() / 32 * 36);
    for block in data.chunks_exact(32) {
        let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let d = amax / 127.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&d.to_le_bytes());
        for &v in block {
            let q = (v * inv_d).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

pub struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    kv_dim: usize,
    capacity: usize,
    pub len: usize,
    /// How many token positions one stored row stands for. `1` for an
    /// ordinary per-token key/value slot — every layer of every
    /// architecture except the block-compressed ones.
    ///
    /// `engine::arch::deepseek4` keeps its compressed attention blocks in
    /// slots of this same type, and a block covers `ratio` (4 or 128)
    /// consecutive tokens rather than one. The rows are still positional
    /// (row `b` is the block over tokens `[b*stride, b*stride + stride)`),
    /// so rollback and prefix reuse stay exact — they just have to convert
    /// a *token* count to a row count, which is what this field is for.
    /// Nothing else changes: [`Self::push`] appends one row per completed
    /// block exactly as a per-token slot appends one row per token.
    stride: usize,
    /// GPU-resident mirror of `k`/`v`, built lazily on the first call that
    /// needs it (a Vulkan-backed decode step) — `None` for every other
    /// backend/request. See [`Self::sync_gpu`].
    gpu: Option<GpuLayerCache>,
}

/// One layer's GPU-resident KV cache mirror, plus the softmax scratch
/// buffer `VulkanBackend::gpu_attention` needs (sized `[n_head, capacity]`
/// once, up front, reused every call — allocating it fresh per decode step
/// would mean 35 multi-megabyte allocations per generated token). Lives
/// here (not in `engine::backend::vulkan`) because it's owned by this
/// per-request `LayerCache`, not by the shared `VulkanBackend` singleton —
/// a KV cache is per-session state, unlike a model's weights.
struct GpuLayerCache {
    /// One backing buffer holding this layer's key and value regions as
    /// aligned sub-ranges — a single BO instead of two. On a per-token decode
    /// submission the kernel re-validates and VM-maps every referenced BO
    /// (a significant share of per-submit decode CPU), so merging k+v → 1
    /// BO/layer shrinks that per-submit BO list by one entry per layer. Bind
    /// groups bind the *sub-ranges* (`k`/`v`), which
    /// makes the attention shader's position index relative to each region's
    /// start — so only explicit copy/`write_buffer` destinations add the
    /// region base offset, never the shader. `probs_scratch` stays a separate
    /// buffer: attention *writes* it while *reading* k/v in the same dispatch,
    /// and `wgpu` forbids one buffer being both read-only and read-write
    /// within a single dispatch's usage scope.
    kv_buffer: wgpu::Buffer,
    k_off: u64,
    k_size: u64,
    v_off: u64,
    v_size: u64,
    probs_scratch: wgpu::Buffer,
    /// How many of `LayerCache::len`'s positions have already been
    /// uploaded — lets a multi-token prefill's worth of pushes get synced
    /// in one bulk upload on the first decode step that needs the GPU
    /// mirror, rather than uploading position-by-position as prefill runs
    /// (prefill never touches this mirror at all today; only decode's
    /// fused GPU attention path does).
    synced_len: usize,
    /// Which of `f32`/`f16`/`q8_0` `k_buf`/`v_buf` above are stored as —
    /// fixed for this mirror's whole lifetime once [`Self::new`] decides
    /// it, so [`LayerCache::sync_gpu`]'s CPU→GPU upload path can check it
    /// without needing its own copy.
    kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    /// Cached attention-dispatch resources, keyed by the *calling layer's*
    /// `wq` tensor identity (`QuantMatrix::cache_key()`) — see
    /// [`GpuAttnDispatch`]'s doc comment for why one `LayerCache` can need
    /// more than one entry here.
    #[allow(dead_code)]
    attn_dispatch: std::collections::HashMap<(usize, usize), GpuAttnDispatch>,
}

/// Sub-range handles into a [`GpuLayerCache::buffer`] returned by
/// [`LayerCache::sync_gpu`] — the shared backing buffer plus each region's
/// `(offset, size)`, so a caller binds `k`/`v`/`probs` as sub-ranges of the
/// one BO. `buffer` is an `Arc`-backed clone, so holding it releases the
/// `&mut LayerCache` borrow `sync_gpu` needed.
pub struct GpuKvRefs {
    /// The shared key/value buffer; `k`/`v` are sub-ranges of it.
    pub buffer: wgpu::Buffer,
    pub k_off: u64,
    pub k_size: u64,
    pub v_off: u64,
    pub v_size: u64,
    /// Softmax scratch — a separate buffer (read-write, so it can't share
    /// `buffer` with the read-only k/v in one dispatch), bound whole.
    pub probs: wgpu::Buffer,
}

/// `VulkanBackend::fused_attention`'s own bind group and small buffers,
/// built once per (layer, `LayerCache`) pair and reused every later
/// decode step for that pair. Lives here (opaque `wgpu` types only, no
/// dependency on `engine::backend::vulkan`'s `AttnMeta`/bind-group-layout
/// specifics) because the bind group references *this* `LayerCache`'s own
/// `k_buf`/`v_buf`/`probs_scratch` — request-scoped state a
/// `VulkanBackend`-level cache (keyed only by weight-tensor identity, as
/// `fused_post_attention`'s `FusedResources` is) can't safely reuse
/// across two different requests' KV caches. Being a field on the
/// request-owned `LayerCache` instead sidesteps that cross-request risk
/// entirely.
///
/// **Keyed per calling layer, not just per `LayerCache`**, because
/// Gemma4's cross-layer KV-donor layers share *one* `LayerCache` (the
/// owning layer's) across several layers, each with its own distinct
/// `wq` — the bind group's `q` binding points at a *specific* layer's Q
/// output buffer (`VulkanBackend::op_cache`, keyed by that layer's own
/// `wq`), so reusing the owning layer's cached dispatch for a donor
/// layer's call would silently bind the *wrong* layer's Q data. A single
/// `Option<GpuAttnDispatch>` here missed exactly that the first time this
/// was built — caught by a real end-to-end request against the actual
/// `E2B` model (incoherent output), not by any synthetic unit test, since
/// every synthetic test used only one `(LayerCache, wq)` pair. Only
/// `meta_buf`'s *contents* (this call's `pos`/`n_pos`/`window_start`)
/// change call to call within one entry — the bind group and every
/// buffer identity stay fixed once built.
#[allow(dead_code)]
pub struct GpuAttnDispatch {
    pub bind_group: wgpu::BindGroup,
    pub out_buf: wgpu::Buffer,
    pub meta_buf: wgpu::Buffer,
    pub readback_buf: wgpu::Buffer,
    /// This layer's K-cast/quantize dispatch (its `f32` K-projection
    /// output → this `LayerCache`'s `f16`- or `q8_0`-stored `k_buf`) —
    /// `Some` only when [`GpuLayerCache::kv_storage`] isn't `F32`; `None`
    /// (and the plain `copy_buffer_to_buffer` path used instead)
    /// otherwise. Same
    /// per-calling-layer keying rationale as this struct's own doc
    /// comment: `k_buf` is per-`LayerCache`, but the cast's *source*
    /// (this layer's own K-projection output buffer) is per-layer, so
    /// this can't be cached anywhere but here either.
    pub k_cast: Option<KvCastDispatch>,
    /// Same as `k_cast`, for V.
    pub v_cast: Option<KvCastDispatch>,
    /// Split-k attention — `None` unless `VulkanBackend::attn_split` is
    /// set. See [`AttnSplitDispatch`]'s own doc comment.
    pub split: Option<AttnSplitDispatch>,
}

/// Split-k attention's own per-(calling layer, `LayerCache`) resources —
/// same per-calling-layer keying rationale as [`GpuAttnDispatch`] itself
/// (the `split_bind_group`'s `aq` binding points at a specific layer's Q
/// output buffer, the same cross-layer-donor hazard that struct's own doc
/// comment describes). `reduce_bind_group` writes into the *same*
/// [`GpuAttnDispatch::out_buf`] the un-split path would have written
/// directly, so downstream readers of `out_buf` (the readback that turns
/// it into `attn_out`) don't need to know or care which path actually
/// filled it.
#[allow(dead_code)]
pub struct AttnSplitDispatch {
    pub split_bind_group: wgpu::BindGroup,
    pub split_meta_buf: wgpu::Buffer,
    pub reduce_bind_group: wgpu::BindGroup,
    pub reduce_meta_buf: wgpu::Buffer,
    /// The phase-1 softmax partials (`partial_ml`, `partial_acc`) — bound by
    /// both bind groups; exposed as named fields so the raw-Vulkan replay
    /// capture can enumerate the exact buffers the attention dispatches bind.
    /// Only read by the replay path.
    pub partial_ml: wgpu::Buffer,
    pub partial_acc: wgpu::Buffer,
}

/// One cached `f32 -> f16` cast dispatch (`VulkanBackend::kv_cast_pipeline`)
/// — a bind group over a fixed `(source, destination)` buffer pair, plus
/// the small meta buffer whose *contents* (the destination offset, this
/// call's `write_pos * kv_dim`) change every call. See [`GpuAttnDispatch::
/// k_cast`]/`v_cast`.
#[allow(dead_code)]
pub struct KvCastDispatch {
    pub bind_group: wgpu::BindGroup,
    pub meta_buf: wgpu::Buffer,
}

impl GpuLayerCache {
    fn new(
        device: &wgpu::Device,
        capacity: usize,
        kv_dim: usize,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> Self {
        // `Q8_0`'s 9-word (36-byte), 32-element blocks aren't expressible
        // as a fixed per-element byte count the way `f32`/`f16` are — size
        // by block count directly instead.
        let kv_bytes: u64 = match kv_storage {
            crate::engine::backend::vulkan_shaders::KvStorage::F32 => {
                (capacity * kv_dim * 4) as u64
            }
            crate::engine::backend::vulkan_shaders::KvStorage::F16 => {
                (capacity * kv_dim * 2) as u64
            }
            crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => {
                debug_assert_eq!(
                    (capacity * kv_dim) % 32,
                    0,
                    "q8_0 KV storage requires capacity * kv_dim to be a multiple of 32"
                );
                (capacity * kv_dim / 32 * 36) as u64
            }
        }
        .max(1);

        // Pack k | v into one buffer, each region starting on a storage-
        // binding-aligned offset so a sub-range binding is valid.
        let align = (device.limits().min_storage_buffer_offset_alignment as u64).max(4);
        let k_off = 0u64;
        let k_size = kv_bytes;
        let v_off = (k_off + k_size).next_multiple_of(align);
        let v_size = kv_bytes;
        let kv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server kv cache (k|v)"),
            size: v_off + v_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probs_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server kv cache attention scratch"),
            size: ((capacity * n_head).max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            kv_buffer,
            k_off,
            k_size,
            v_off,
            v_size,
            probs_scratch,
            synced_len: 0,
            kv_storage,
            attn_dispatch: std::collections::HashMap::new(),
        }
    }
}

impl LayerCache {
    fn new(capacity: usize, kv_dim: usize) -> Self {
        Self::new_strided(capacity, kv_dim, 1)
    }

    /// A slot whose rows each stand for `stride` token positions — see
    /// [`Self::stride`]. `capacity` is still given in *tokens*, so a caller
    /// sizes every slot of a sequence from the one context budget.
    fn new_strided(capacity: usize, kv_dim: usize, stride: usize) -> Self {
        assert!(stride > 0, "a KV slot's stride must be at least one token");
        let rows = capacity.div_ceil(stride);
        Self {
            k: vec![0.0; rows * kv_dim],
            v: vec![0.0; rows * kv_dim],
            kv_dim,
            capacity: rows,
            len: 0,
            gpu: None,
            stride,
        }
    }

    /// Rebuilds a layer from a slot-persistence snapshot: `k`/`v` hold
    /// exactly `len * kv_dim` committed floats each, so `capacity` is set to
    /// `len` — the minimum that keeps this a valid [`Self::copy_prefix_from`]
    /// *source* (only `len`, `kv_dim`, and the `[0, len)` floats are ever
    /// read from a source; its `capacity` is never consulted). A restored
    /// cache is only ever used as a reuse source, never pushed to directly.
    fn from_parts(kv_dim: usize, len: usize, k: Vec<f32>, v: Vec<f32>) -> Self {
        debug_assert_eq!(k.len(), len * kv_dim);
        debug_assert_eq!(v.len(), len * kv_dim);
        Self {
            k,
            v,
            kv_dim,
            capacity: len,
            len,
            gpu: None,
            // A restored layer is only ever a `copy_prefix_from` *source*,
            // and only the destination's stride governs how many rows a
            // token count means, so this never needs the original's.
            stride: 1,
        }
    }

    /// A CPU-only deep copy (no GPU mirror). Used by slot persistence to
    /// snapshot a completed cache for a slot that is also being deposited
    /// into the [`crate::engine::prefix_cache`] pool — the one case where a
    /// single completed cache is needed in two places at once.
    fn duplicate(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            kv_dim: self.kv_dim,
            capacity: self.capacity,
            len: self.len,
            gpu: None,
            stride: self.stride,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drops every position from `new_len` onward, rolling this layer back to
    /// exactly `new_len` cached keys/values (a no-op if it already holds
    /// `new_len` or fewer). The stored `k`/`v` beyond `new_len` are left as-is
    /// — only [`Self::len`] moves, so the next [`Self::push`] overwrites them
    /// in place. If a GPU mirror exists, its synced watermark is pulled back to
    /// at most `new_len` too, so the next [`Self::sync_gpu`] re-uploads any
    /// positions that get written over the rolled-back range. Used to discard a
    /// speculative draft's rejected tail after verification keeps only its
    /// accepted prefix.
    ///
    /// `new_len` is a **token** count, converted to a row count through
    /// [`Self::stride`]: a block-compressed slot keeps only the blocks that
    /// are wholly inside the retained tokens.
    pub fn truncate(&mut self, new_len: usize) {
        let new_len = new_len / self.stride;
        if new_len >= self.len {
            return;
        }
        self.len = new_len;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.synced_len = gpu.synced_len.min(new_len);
        }
    }

    /// A CPU-only snapshot (no GPU mirror) for building an independent
    /// reference in cross-check tests — `engine::backend::vulkan::tests`
    /// needs a plain CPU copy of a cache-in-progress to compute the
    /// expected result against, without disturbing the real `LayerCache`
    /// (and its GPU mirror) the test also feeds to `fused_attention`.
    #[cfg(test)]
    pub fn clone_for_test(&self) -> Self {
        self.duplicate()
    }

    /// Lazily builds this layer's GPU-resident mirror (sized once, for the
    /// cache's whole lifetime — `n_head` is a fixed model property, always
    /// the same across every call for a given layer) and uploads any
    /// positions [`Self::push`]ed since the last sync. The first call
    /// after a multi-token prefill uploads that whole range in one bulk
    /// `write_buffer`; every call after that uploads at most the one new
    /// position a decode step just pushed. Returns the mirror's key/value/
    /// softmax-scratch buffers for `VulkanBackend::gpu_attention` to bind.
    pub fn sync_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> GpuKvRefs {
        let capacity = self.capacity;
        let kv_dim = self.kv_dim;
        let gpu = self.gpu.get_or_insert_with(|| {
            GpuLayerCache::new(device, capacity, kv_dim, n_head, kv_storage)
        });
        if gpu.synced_len < self.len {
            let start = gpu.synced_len * kv_dim;
            let end = self.len * kv_dim;
            // Local byte offset of `start` within each region, by storage
            // format; `k_off`/`v_off` shift it to the region's base in the
            // shared buffer.
            match gpu.kv_storage {
                crate::engine::backend::vulkan_shaders::KvStorage::F16 => {
                    let local = (start * 2) as u64;
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.k_off + local,
                        &f32_to_f16_bytes(&self.k[start..end]),
                    );
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.v_off + local,
                        &f32_to_f16_bytes(&self.v[start..end]),
                    );
                }
                crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => {
                    let local = (start / 32 * 36) as u64;
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.k_off + local,
                        &f32_to_q8_0_bytes(&self.k[start..end]),
                    );
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.v_off + local,
                        &f32_to_q8_0_bytes(&self.v[start..end]),
                    );
                }
                crate::engine::backend::vulkan_shaders::KvStorage::F32 => {
                    let local = (start * 4) as u64;
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.k_off + local,
                        bytemuck::cast_slice(&self.k[start..end]),
                    );
                    queue.write_buffer(
                        &gpu.kv_buffer,
                        gpu.v_off + local,
                        bytemuck::cast_slice(&self.v[start..end]),
                    );
                }
            }
            gpu.synced_len = self.len;
        }
        GpuKvRefs {
            buffer: gpu.kv_buffer.clone(),
            k_off: gpu.k_off,
            k_size: gpu.k_size,
            v_off: gpu.v_off,
            v_size: gpu.v_size,
            probs: gpu.probs_scratch.clone(),
        }
    }

    /// This `(calling layer, LayerCache)` pair's cached attention-dispatch
    /// resources, if [`Self::set_attn_dispatch`] has already built them
    /// for this `wq_key` (a calling layer's `QuantMatrix::cache_key()`) —
    /// `None` on the first call for this key. See [`GpuAttnDispatch`]'s
    /// doc comment for why the key is the *calling layer's* `wq`, not
    /// just this `LayerCache`'s own identity (cross-layer KV donors share
    /// one `LayerCache` across several distinct `wq`s). Only valid to
    /// call after [`Self::sync_gpu`] (the GPU mirror, hence `self.gpu`,
    /// must already exist).
    #[allow(dead_code)]
    pub fn attn_dispatch(&self, wq_key: (usize, usize)) -> Option<&GpuAttnDispatch> {
        self.gpu.as_ref().and_then(|g| g.attn_dispatch.get(&wq_key))
    }

    /// Stores this `(calling layer, LayerCache)` pair's attention-dispatch
    /// resources, built by the caller (`VulkanBackend::fused_attention`)
    /// on a [`Self::attn_dispatch`] cache miss. Panics if
    /// [`Self::sync_gpu`] hasn't run yet — the same precondition
    /// `attn_dispatch` has.
    #[allow(dead_code)]
    pub fn set_attn_dispatch(&mut self, wq_key: (usize, usize), dispatch: GpuAttnDispatch) {
        self.gpu
            .as_mut()
            .expect("set_attn_dispatch called before sync_gpu built the GPU mirror")
            .attn_dispatch
            .insert(wq_key, dispatch);
    }

    /// Like [`Self::push`], but for a key/value the caller has already
    /// written *directly* into the GPU mirror (a `copy_buffer_to_buffer`
    /// inside the same encoder that computed them, at byte offset
    /// `self.len * kv_dim * 4` — see `VulkanBackend::fused_attention`)
    /// instead of going through `push` + `sync_gpu`'s CPU round trip.
    /// Just advances the position counters; the CPU-side `k`/`v` vecs at
    /// this position are **not** populated (left at their zeroed
    /// default).
    ///
    /// That's safe *today* only because nothing ever reads them back:
    /// this module's own doc comment already establishes "no
    /// cross-sequence sharing (no prompt-prefix reuse)," so a cache's
    /// lifetime is strictly one prefill (CPU-computed, uses `push`) then
    /// decode-only pushes — never prefill again after decode has started
    /// (confirmed against `engine::generate::run`, which creates a fresh
    /// `KvCache` per request and never reuses one across requests) — and
    /// `sync_gpu`'s `gpu.synced_len < self.len` check means it will never
    /// try to re-upload (and so never exposes the zeroed gap) once this
    /// advances `synced_len` to match. **If prompt-prefix reuse (slot
    /// save/restore) is ever built, this becomes unsafe** — a resumed
    /// cache could need this position's real data for a later multi-token
    /// prefill's CPU attention path, which
    /// would silently read zeros instead. Whoever builds that should
    /// either make this always mirror to CPU too, or make prompt-prefix
    /// continuation itself GPU-resident.
    #[allow(dead_code)]
    pub fn advance_gpu_only(&mut self) {
        assert!(
            self.len < self.capacity,
            "KV cache is full ({} positions)",
            self.capacity
        );
        self.len += 1;
        if let Some(gpu) = &mut self.gpu {
            gpu.synced_len = self.len;
        }
    }

    /// Appends one token's key/value vectors (`[kv_dim]` each). Panics if
    /// the cache is already at `capacity` — the scheduler is responsible
    /// for never handing a sequence more tokens than its context window
    /// allows.
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        assert!(
            self.len < self.capacity,
            "KV cache is full ({} positions)",
            self.capacity
        );
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);
        let start = self.len * self.kv_dim;
        self.k[start..start + self.kv_dim].copy_from_slice(k);
        self.v[start..start + self.kv_dim].copy_from_slice(v);
        self.len += 1;
    }

    /// Commits `n = k_rows.len() / kv_dim` positions whose K/V a GPU-resident
    /// prefill has **already written straight into the mirror**, mirroring the
    /// same values into the host `k`/`v` so the two stay consistent.
    ///
    /// This is the batched counterpart of [`Self::advance_gpu_only`], and it
    /// deliberately does *not* share that method's shortcut. `advance_gpu_only`
    /// leaves the host copy zeroed, which is only safe under the condition its
    /// own doc comment states — a cache is CPU-prefilled and then decode-only.
    /// A GPU-resident *prefill* removes that condition, and the host copy is
    /// what [`KvCache::to_bytes`] serializes for slot save and what the CPU
    /// attention path reads, so those positions have to be real.
    ///
    /// `k_rows`/`v_rows` must be the **f32 values fed to the mirror write**
    /// (K after its norm and RoPE, V after its weightless norm), not a readback
    /// of the mirror itself: the mirror may be `f16` or block-quantized, and
    /// the host side should hold what the CPU path would have held. The mirror
    /// is then marked current so [`Self::sync_gpu`] does not re-upload rows the
    /// GPU just wrote.
    // Not wired into a forward pass yet — the batched prefill recorder that
    // will call it is still being built; its own tests cover it meanwhile.
    #[allow(dead_code)]
    pub fn commit_gpu_written(&mut self, k_rows: &[f32], v_rows: &[f32]) {
        assert_eq!(
            k_rows.len(),
            v_rows.len(),
            "K and V must commit the same number of positions"
        );
        assert_eq!(
            k_rows.len() % self.kv_dim,
            0,
            "committed rows ({}) are not a whole number of kv_dim ({}) positions",
            k_rows.len(),
            self.kv_dim
        );
        for (k, v) in k_rows
            .chunks_exact(self.kv_dim)
            .zip(v_rows.chunks_exact(self.kv_dim))
        {
            self.push(k, v);
        }
        // The GPU wrote these into the mirror itself, so the incremental upload
        // in `sync_gpu` has nothing left to do for them.
        if let Some(gpu) = &mut self.gpu {
            gpu.synced_len = self.len;
        }
    }

    /// The key vector at cached position `pos` for KV head `kv_head`
    /// (`[head_dim]`).
    pub fn key_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let row = &self.k[pos * self.kv_dim..(pos + 1) * self.kv_dim];
        &row[kv_head * head_dim..(kv_head + 1) * head_dim]
    }

    pub fn value_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let row = &self.v[pos * self.kv_dim..(pos + 1) * self.kv_dim];
        &row[kv_head * head_dim..(kv_head + 1) * head_dim]
    }

    /// Overwrites this (freshly allocated, empty) layer's first `len`
    /// cached positions with `src`'s own already-computed ones — the raw
    /// float copy [`KvCache::copy_prefix_from`] needs. `src`'s positions
    /// `[0, len)` were computed from the exact same token ids this layer
    /// is about to be asked to continue from, so there's nothing to
    /// recompute for them. Drops any GPU mirror `self` already had so one
    /// gets rebuilt lazily, sized to `self`'s own capacity, the next time
    /// [`Self::sync_gpu`] runs — `src` and `self` can have different
    /// capacities (two different requests' own prompt-plus-max-tokens
    /// budgets), so this never tries to reuse `src`'s GPU buffers
    /// directly.
    ///
    /// A no-op when `src.len == 0` — a cross-layer KV-donor layer's own
    /// array slot (`engine::arch::gemma`'s `kv_donor`) never gets pushed
    /// to directly (its writes/reads always redirect to the donor
    /// target's own slot instead), so it stays at `len == 0` for its
    /// whole lifetime no matter how far the model has actually
    /// progressed — nothing downstream ever reads such a slot's own
    /// `len`/`k`/`v`, so leaving `self`'s corresponding slot at its own
    /// freshly allocated (all-zero) state is exactly correct, not a
    /// partial or best-effort copy. [`KvCache::copy_prefix_from`]'s
    /// caller (`engine::prefix_cache::PrefixCache::take_best_match`)
    /// already bounds `len` by the *maximum* `len` across every layer
    /// precisely so a real owning layer's `src.len` is never smaller than
    /// `len` — only a permanently-dead donor slot can still be `0` here.
    ///
    /// `len` is a **token** count, converted to a row count through
    /// [`Self::stride`] exactly as [`Self::truncate`] does, so a
    /// block-compressed slot carries over only the blocks wholly inside the
    /// reused prefix.
    fn copy_prefix_from(&mut self, src: &LayerCache, len: usize) {
        debug_assert_eq!(self.kv_dim, src.kv_dim);
        if src.len == 0 {
            return;
        }
        let len = len / self.stride;
        assert!(
            len <= self.capacity,
            "reused prefix ({len}) exceeds this request's own KV capacity ({})",
            self.capacity
        );
        assert!(
            len <= src.len,
            "reused prefix ({len}) exceeds the source cache's own committed length ({})",
            src.len
        );
        let n = len * self.kv_dim;
        self.k[..n].copy_from_slice(&src.k[..n]);
        self.v[..n].copy_from_slice(&src.v[..n]);
        self.len = len;
        self.gpu = None;
    }
}

pub struct KvCache {
    pub layers: Vec<LayerCache>,
    /// Recurrent (SSM / gated-delta-net) layer state, for architectures
    /// that mix attention and linear-attention layers (`engine::arch::
    /// qwen35moe`) — densely packed in that architecture's own recurrent-
    /// layer order, entirely separate from `layers` above (a positional
    /// KV cache and a recurrent state have nothing in common). Empty for
    /// every other architecture.
    pub recurrent: Vec<RecurrentLayerState>,
}

impl KvCache {
    pub fn new(n_layer: usize, capacity: usize, kv_dim: usize) -> Self {
        Self::new_with_dims(capacity, &vec![kv_dim; n_layer])
    }

    /// Like [`KvCache::new`], but each layer gets its own `kv_dim` — for
    /// architectures where key/value head size varies by layer (e.g.
    /// Gemma's SWA vs. full-attention layers using different head dims).
    pub fn new_with_dims(capacity: usize, kv_dims: &[usize]) -> Self {
        Self {
            layers: kv_dims
                .iter()
                .map(|&dim| LayerCache::new(capacity, dim))
                .collect(),
            recurrent: Vec::new(),
        }
    }

    /// Like [`KvCache::new_with_dims`], but each slot also carries a
    /// *stride*: how many token positions one of its rows stands for (see
    /// [`LayerCache::stride`]). `capacity` stays a token count for every
    /// slot. `engine::arch::deepseek4` uses this to keep its
    /// block-compressed attention state (one row per 4- or 128-token block)
    /// in the same positional cache as its per-token keys, so rollback,
    /// prefix reuse, and slot persistence apply to all of it unchanged.
    pub fn new_with_strided_dims(capacity: usize, kv_dims: &[(usize, usize)]) -> Self {
        Self {
            layers: kv_dims
                .iter()
                .map(|&(dim, stride)| LayerCache::new_strided(capacity, dim, stride))
                .collect(),
            recurrent: Vec::new(),
        }
    }

    /// Like [`KvCache::new_with_dims`], plus a recurrent state per entry in
    /// `recurrent_specs` (`(conv_channels, d_conv, num_heads, head_dim)`),
    /// for a mixed attention/linear-attention architecture.
    pub fn new_mixed(
        capacity: usize,
        kv_dims: &[usize],
        recurrent_specs: &[(usize, usize, usize, usize)],
    ) -> Self {
        let mut cache = Self::new_with_dims(capacity, kv_dims);
        cache.recurrent = recurrent_specs
            .iter()
            .map(|&(conv_channels, d_conv, num_heads, head_dim)| {
                RecurrentLayerState::new(conv_channels, d_conv, num_heads, head_dim)
            })
            .collect();
        cache
    }

    /// Reuses `src`'s already-computed positions `[0, len)` instead of
    /// recomputing them — the mechanism `crate::engine::prefix_cache`
    /// needs to skip re-prefilling a prompt prefix a previous request
    /// already processed (e.g. the same conversation's prior turns, or a
    /// system prompt shared with an earlier, unrelated request). `self`
    /// must already be a freshly allocated cache (this request's own
    /// `capacity`, `len == 0` on every layer) — see [`LayerCache::
    /// copy_prefix_from`] for why this always copies into a fresh cache
    /// rather than adopting `src`'s buffers directly.
    ///
    /// Recurrent (SSM / gated-delta-net) layer state has no per-position
    /// history to truncate — a caller may only pass `len == src`'s own
    /// full committed length when `src.recurrent` is non-empty (i.e. the
    /// new request's prompt is `src`'s own tokens plus a strict suffix,
    /// never a shorter, older prefix of them); `crate::engine::
    /// prefix_cache::PrefixCache::take_best_match` enforces exactly that
    /// restriction before this is ever called on a mixed-architecture
    /// cache.
    pub fn copy_prefix_from(&mut self, src: &KvCache, len: usize) {
        for (dst, src_layer) in self.layers.iter_mut().zip(src.layers.iter()) {
            dst.copy_prefix_from(src_layer, len);
        }
        for (dst, src_r) in self.recurrent.iter_mut().zip(src.recurrent.iter()) {
            dst.copy_from(src_r);
        }
    }

    /// Rolls every attention layer back to `new_len` positions (see
    /// [`LayerCache::truncate`]). Only valid for a cache with no recurrent
    /// (SSM / gated-delta-net) layers: those carry a single evolving state with
    /// no per-position history to roll back, so a partial rollback can't be
    /// expressed — the caller (speculative decoding) is gated to architectures
    /// without them, and this asserts that precondition.
    pub fn truncate(&mut self, new_len: usize) {
        debug_assert!(
            self.recurrent.is_empty(),
            "KvCache::truncate is not valid for architectures with recurrent layers"
        );
        for layer in &mut self.layers {
            layer.truncate(new_len);
        }
    }

    /// How many token positions are actually committed to this cache — the
    /// maximum `len` across every attention layer, so a permanently-empty
    /// cross-layer KV-donor slot (`engine::arch::gemma`'s `kv_donor`) never
    /// drags the count to zero. `0` for a freshly allocated, never-pushed
    /// cache. This is what a saved slot reports as its reusable token count.
    pub fn committed_len(&self) -> usize {
        self.layers.iter().map(|l| l.len).max().unwrap_or(0)
    }

    /// A CPU-only deep copy (no GPU mirror) of the whole cache — the
    /// [`crate::engine::slot_store`] uses it to snapshot a slot's completed
    /// cache when that same cache is also being handed to the
    /// [`crate::engine::prefix_cache`] pool.
    pub fn duplicate(&self) -> Self {
        Self {
            layers: self.layers.iter().map(LayerCache::duplicate).collect(),
            recurrent: self
                .recurrent
                .iter()
                .map(RecurrentLayerState::duplicate)
                .collect(),
        }
    }

    /// A structural signature — layer count and each layer's `kv_dim`, plus
    /// every recurrent layer's `(conv_channels, d_conv, num_heads, head_dim)`
    /// — with no per-position data or capacity in it. Two caches from the
    /// same model architecture always agree here; two from different models
    /// (or different KV shapes) never do. Feeds the on-disk slot fingerprint
    /// so a snapshot can only ever be restored into a structurally identical
    /// model. Deterministic across runs (no hashing here — the caller hashes
    /// it together with the model label).
    pub fn structure_tag(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, self.layers.len() as u32);
        for l in &self.layers {
            push_u32(&mut out, l.kv_dim as u32);
        }
        push_u32(&mut out, self.recurrent.len() as u32);
        for r in &self.recurrent {
            push_u32(&mut out, r.conv_channels as u32);
            push_u32(&mut out, r.d_conv as u32);
            push_u32(&mut out, r.num_heads() as u32);
            push_u32(&mut out, r.head_dim as u32);
        }
        out
    }

    /// Serializes every committed KV position (and all recurrent state) to a
    /// self-describing little-endian byte blob — the payload
    /// [`crate::engine::slot_store`] writes under
    /// `~/.orangu/server/<fp>/slots/`. Only the `[0, len)` floats of each
    /// layer are written (never the unused tail of a larger `capacity`), so
    /// a saved file is sized to the conversation, not the context window.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(KV_CACHE_MAGIC);
        push_u32(&mut out, self.layers.len() as u32);
        for l in &self.layers {
            push_u32(&mut out, l.kv_dim as u32);
            push_u32(&mut out, l.len as u32);
            let n = l.len * l.kv_dim;
            push_f32s(&mut out, &l.k[..n]);
            push_f32s(&mut out, &l.v[..n]);
        }
        push_u32(&mut out, self.recurrent.len() as u32);
        for r in &self.recurrent {
            push_u32(&mut out, r.conv_channels as u32);
            push_u32(&mut out, r.d_conv as u32);
            push_u32(&mut out, r.num_heads() as u32);
            push_u32(&mut out, r.head_dim as u32);
            push_f32s(&mut out, &r.conv_history);
            push_f32s(&mut out, &r.delta_state);
        }
        out
    }

    /// Inverse of [`Self::to_bytes`]. Every length is validated against the
    /// remaining input before it is read, so a truncated or corrupt file
    /// yields an `Err` rather than a panic or an out-of-bounds read — the
    /// caller ([`crate::engine::slot_store`]) treats any `Err` as "nothing to
    /// restore" and falls back to a normal prefill.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let mut cur = ByteReader::new(bytes);
        if cur.take(KV_CACHE_MAGIC.len())? != KV_CACHE_MAGIC {
            anyhow::bail!("not an orangu KV-cache blob (bad magic)");
        }
        let n_layer = cur.u32()? as usize;
        let mut layers = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            let kv_dim = cur.u32()? as usize;
            let len = cur.u32()? as usize;
            let n = len
                .checked_mul(kv_dim)
                .ok_or_else(|| anyhow::anyhow!("KV layer dimensions overflow"))?;
            let k = cur.f32s(n)?;
            let v = cur.f32s(n)?;
            layers.push(LayerCache::from_parts(kv_dim, len, k, v));
        }
        let n_rec = cur.u32()? as usize;
        let mut recurrent = Vec::with_capacity(n_rec);
        for _ in 0..n_rec {
            let conv_channels = cur.u32()? as usize;
            let d_conv = cur.u32()? as usize;
            let num_heads = cur.u32()? as usize;
            let head_dim = cur.u32()? as usize;
            let conv_history = cur.f32s(conv_channels * d_conv.saturating_sub(1))?;
            let delta_state = cur.f32s(num_heads * head_dim * head_dim)?;
            recurrent.push(RecurrentLayerState::from_parts(
                conv_channels,
                d_conv,
                head_dim,
                conv_history,
                delta_state,
            ));
        }
        if !cur.is_empty() {
            anyhow::bail!("trailing bytes after KV-cache blob");
        }
        Ok(Self { layers, recurrent })
    }
}

const KV_CACHE_MAGIC: &[u8] = b"ORGUKVC1";

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_f32s(out: &mut Vec<u8>, data: &[f32]) {
    out.reserve(data.len() * 4);
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// A tiny bounds-checked forward cursor over the slot-persistence byte
/// format — every read validates it stays within the buffer, so a
/// malformed file can never panic or read out of bounds.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| anyhow::anyhow!("unexpected end of KV-cache blob"))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s(&mut self, count: usize) -> anyhow::Result<Vec<f32>> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("KV-cache float count overflows"))?;
        let b = self.take(bytes)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// One recurrent (SSM / gated-delta-net) layer's persistent state: a
/// causal-conv1d rolling history and a per-head delta-net state matrix.
/// Unlike [`LayerCache`], there's no per-position history to index —
/// linear attention/SSM layers carry a single evolving state forward.
pub struct RecurrentLayerState {
    /// `[conv_channels, d_conv - 1]`, channel-major, oldest-first per
    /// channel — the causal conv1d's rolling window of prior inputs.
    conv_history: Vec<f32>,
    conv_channels: usize,
    d_conv: usize,
    /// Per-head delta-net state matrices, flattened
    /// `[num_heads, head_dim, head_dim]` (`state[head][i][j]`).
    delta_state: Vec<f32>,
    head_dim: usize,
}

impl RecurrentLayerState {
    fn new(conv_channels: usize, d_conv: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            conv_history: vec![0.0; conv_channels * d_conv.saturating_sub(1)],
            conv_channels,
            d_conv,
            delta_state: vec![0.0; num_heads * head_dim * head_dim],
            head_dim,
        }
    }

    /// Rebuilds a recurrent state from a slot-persistence snapshot. `num_heads`
    /// is recovered from `delta_state`'s length rather than stored separately.
    fn from_parts(
        conv_channels: usize,
        d_conv: usize,
        head_dim: usize,
        conv_history: Vec<f32>,
        delta_state: Vec<f32>,
    ) -> Self {
        Self {
            conv_history,
            conv_channels,
            d_conv,
            delta_state,
            head_dim,
        }
    }

    /// How many delta-net heads this state carries — `delta_state` is a dense
    /// `[num_heads, head_dim, head_dim]`, so the head count is implied by its
    /// length. `0` for the degenerate `head_dim == 0` case (never real).
    fn num_heads(&self) -> usize {
        self.delta_state
            .len()
            .checked_div(self.head_dim * self.head_dim)
            .unwrap_or(0)
    }

    /// A deep copy — every field is owned data (`Vec<f32>` plus dimensions),
    /// so this is a plain clone, used by [`KvCache::duplicate`].
    fn duplicate(&self) -> Self {
        Self {
            conv_history: self.conv_history.clone(),
            conv_channels: self.conv_channels,
            d_conv: self.d_conv,
            delta_state: self.delta_state.clone(),
            head_dim: self.head_dim,
        }
    }

    /// Overwrites this state with `src`'s own — the whole-state carryover
    /// [`KvCache::copy_prefix_from`] uses for the recurrent-layer case
    /// (never a partial/truncated copy; see that method's own doc comment
    /// for why only a full carryover is ever valid here).
    fn copy_from(&mut self, src: &RecurrentLayerState) {
        debug_assert_eq!(self.conv_channels, src.conv_channels);
        debug_assert_eq!(self.d_conv, src.d_conv);
        debug_assert_eq!(self.head_dim, src.head_dim);
        self.conv_history.copy_from_slice(&src.conv_history);
        self.delta_state.copy_from_slice(&src.delta_state);
    }

    /// One timestep of causal depthwise conv1d: convolves `input`
    /// (`[conv_channels]`) against `kernel` (`[conv_channels, d_conv]`,
    /// channel-major — ggml's own `ssm_conv1d.weight` element order, `{
    /// d_conv, conv_channels }` with `d_conv` fastest-varying), using this
    /// layer's rolling history for the taps that reach before the current
    /// token, then slides the window forward. Returns the convolved output
    /// (`[conv_channels]`); the caller applies SiLU itself.
    pub fn conv_step(&mut self, input: &[f32], kernel: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.conv_channels);
        debug_assert_eq!(kernel.len(), self.conv_channels * self.d_conv);
        let hist_w = self.d_conv - 1;
        let mut out = vec![0f32; self.conv_channels];
        for c in 0..self.conv_channels {
            let hist = &self.conv_history[c * hist_w..(c + 1) * hist_w];
            let ker = &kernel[c * self.d_conv..(c + 1) * self.d_conv];
            let mut sum = 0f32;
            for (tap, &h) in hist.iter().enumerate() {
                sum += h * ker[tap];
            }
            // The last tap always weights the current (newest) token.
            sum += input[c] * ker[hist_w];
            out[c] = sum;
        }
        if hist_w > 0 {
            for (c, &v) in input.iter().enumerate() {
                let base = c * hist_w;
                self.conv_history.copy_within(base + 1..base + hist_w, base);
                self.conv_history[base + hist_w - 1] = v;
            }
        }
        out
    }

    /// The delta-net state matrix for `head` (`[head_dim, head_dim]`,
    /// mutable — the recurrence updates it in place every token).
    pub fn delta_state_mut(&mut self, head: usize) -> &mut [f32] {
        let start = head * self.head_dim * self.head_dim;
        &mut self.delta_state[start..start + self.head_dim * self.head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_then_read_back_key_and_value() {
        let mut cache = LayerCache::new(4, 6);
        cache.push(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        );
        assert_eq!(cache.len, 1);
        // head_dim=3, kv_head=1 -> elements [3..6).
        assert_eq!(cache.key_at(0, 1, 3), &[4.0, 5.0, 6.0]);
        assert_eq!(cache.value_at(0, 0, 3), &[6.0, 5.0, 4.0]);
    }

    /// A strided slot (`engine::arch::deepseek4`'s compressed blocks) is
    /// sized, rolled back, and prefix-copied in *token* units like every
    /// other slot, while storing one row per `stride` tokens.
    #[test]
    fn a_strided_slot_converts_token_counts_to_row_counts() {
        // 10 tokens at stride 4 is room for 3 blocks (the last is partial).
        let mut cache = LayerCache::new_strided(10, 2, 4);
        assert_eq!(cache.capacity(), 3);
        for b in 0..3 {
            cache.push(&[b as f32, 0.0], &[0.0, b as f32]);
        }
        assert_eq!(cache.len, 3);

        // Rolling back to 9 tokens keeps the two blocks wholly inside them.
        cache.truncate(9);
        assert_eq!(cache.len, 2);
        // ...and to 4 tokens, only the first.
        cache.truncate(4);
        assert_eq!(cache.len, 1);
        assert_eq!(cache.key_at(0, 0, 2), &[0.0, 0.0]);
        cache.truncate(3);
        assert_eq!(cache.len, 0);
    }

    #[test]
    fn a_strided_slot_reuses_only_the_blocks_inside_the_reused_prefix() {
        let mut src = LayerCache::new_strided(12, 2, 4);
        for b in 0..3 {
            src.push(&[10.0 + b as f32, 0.0], &[0.0, 0.0]);
        }
        let mut dst = LayerCache::new_strided(12, 2, 4);
        // 11 tokens of prefix: blocks 0 and 1 are inside it, block 2 is not.
        dst.copy_prefix_from(&src, 11);
        assert_eq!(dst.len, 2);
        assert_eq!(dst.key_at(0, 0, 2), &[10.0, 0.0]);
        assert_eq!(dst.key_at(1, 0, 2), &[11.0, 0.0]);
    }

    /// Mixing strides in one cache is the point: a deepseek4 sequence keeps
    /// per-token keys and per-block compressed keys side by side, and one
    /// token count has to mean the right thing for both.
    #[test]
    fn a_mixed_stride_cache_rolls_every_slot_back_to_the_same_token_count() {
        let mut cache = KvCache::new_with_strided_dims(8, &[(2, 1), (2, 4)]);
        for t in 0..8 {
            cache.layers[0].push(&[t as f32, 0.0], &[0.0, 0.0]);
            if (t + 1) % 4 == 0 {
                cache.layers[1].push(&[t as f32, 0.0], &[0.0, 0.0]);
            }
        }
        assert_eq!(cache.committed_len(), 8);
        assert_eq!(cache.layers[1].len, 2);
        cache.truncate(5);
        assert_eq!(cache.layers[0].len, 5);
        assert_eq!(cache.layers[1].len, 1);
    }

    /// A GPU-resident prefill commits a whole batch at once, and the host copy
    /// it leaves behind must be indistinguishable from the one the CPU path's
    /// per-token `push` loop would have left — that copy is what slot save
    /// serializes and what the CPU attention path reads.
    #[test]
    fn commit_gpu_written_leaves_the_same_host_state_as_pushing_each_row() {
        let kv_dim = 6;
        let rows: Vec<f32> = (0..3 * kv_dim).map(|i| i as f32).collect();
        let vrows: Vec<f32> = (0..3 * kv_dim).map(|i| -(i as f32)).collect();

        let mut pushed = LayerCache::new(8, kv_dim);
        for t in 0..3 {
            pushed.push(
                &rows[t * kv_dim..(t + 1) * kv_dim],
                &vrows[t * kv_dim..(t + 1) * kv_dim],
            );
        }

        let mut committed = LayerCache::new(8, kv_dim);
        committed.commit_gpu_written(&rows, &vrows);

        assert_eq!(committed.len, pushed.len);
        assert_eq!(committed.k, pushed.k);
        assert_eq!(committed.v, pushed.v);
        // head_dim=3, kv_head=1 on the last committed position.
        assert_eq!(committed.key_at(2, 1, 3), pushed.key_at(2, 1, 3));
    }

    /// Committing rows the GPU already wrote must not leave `sync_gpu` wanting
    /// to upload them again — the whole point of the batched write is that the
    /// mirror is already current.
    #[test]
    fn commit_gpu_written_marks_the_mirror_current() {
        let mut cache = LayerCache::new(8, 4);
        // No GPU mirror allocated yet: the bookkeeping must still be sound, and
        // a later `sync_gpu` builds one and uploads from the host copy.
        cache.commit_gpu_written(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len, 1);
        assert!(cache.gpu.is_none());
    }

    #[test]
    #[should_panic(expected = "not a whole number of kv_dim")]
    fn commit_gpu_written_rejects_a_partial_row() {
        let mut cache = LayerCache::new(4, 4);
        cache.commit_gpu_written(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "KV cache is full")]
    fn push_past_capacity_panics() {
        let mut cache = LayerCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[1.0, 2.0]);
        cache.push(&[1.0, 2.0], &[1.0, 2.0]);
    }

    #[test]
    fn conv_step_uses_zeroed_history_for_the_first_tokens() {
        // 1 channel, d_conv=3 (2 taps of history + the current token).
        // kernel = [tap0, tap1, tap2] for this channel.
        let mut state = RecurrentLayerState::new(1, 3, 1, 1);
        let kernel = [1.0, 10.0, 100.0];
        // History starts at [0, 0]; first token contributes only via the
        // last tap: 0*1 + 0*10 + 5*100 = 500.
        let out = state.conv_step(&[5.0], &kernel);
        assert_eq!(out, vec![500.0]);
    }

    #[test]
    fn conv_step_slides_the_window_across_tokens() {
        let mut state = RecurrentLayerState::new(1, 3, 1, 1);
        let kernel = [1.0, 10.0, 100.0];
        let _ = state.conv_step(&[5.0], &kernel); // history becomes [0, 5]
        // Second token=7: taps see history [0, 5] then current 7:
        // 0*1 + 5*10 + 7*100 = 750.
        let out = state.conv_step(&[7.0], &kernel);
        assert_eq!(out, vec![750.0]);
        // Third token=9: history is now [5, 7]:
        // 5*1 + 7*10 + 9*100 = 975.
        let out = state.conv_step(&[9.0], &kernel);
        assert_eq!(out, vec![975.0]);
    }

    #[test]
    fn delta_state_mut_is_independent_per_head() {
        let mut state = RecurrentLayerState::new(1, 2, 2, 2);
        state
            .delta_state_mut(0)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        state
            .delta_state_mut(1)
            .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(state.delta_state_mut(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(state.delta_state_mut(1), &[5.0, 6.0, 7.0, 8.0]);
    }
}

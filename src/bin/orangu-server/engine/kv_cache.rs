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

use std::sync::Mutex;

/// Converts a slice of `f32` KV values into little-endian `f16` bytes, for
/// `LayerCache::sync_gpu`'s `f16` KV-mirror upload path. A plain
/// per-element loop, not `bytemuck::cast_slice` — unlike the `f32` path,
/// this genuinely *converts* values, not just reinterprets bytes.
pub(crate) fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
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
pub(crate) fn f32_to_q8_0_bytes(data: &[f32]) -> Vec<u8> {
    debug_assert_eq!(
        data.len() % 32,
        0,
        "q8_0 KV storage requires kv_dim to be a multiple of 32"
    );
    let mut out = Vec::with_capacity(data.len() / 32 * 36);
    for block in data.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let d = amax / 127.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&d.to_le_bytes());
        // Built into a fixed array and appended once, rather than a `push`
        // per byte. The `push` is a capacity check and a length update per
        // element, which is what kept LLVM from widening this loop past
        // 128 bits — the block is 32 quantized values whether it is written
        // one at a time or in one go. Bit-identical: same expression, same
        // order, same bytes.
        let mut q = [0u8; 32];
        for (slot, &v) in q.iter_mut().zip(block) {
            *slot = ((v * inv_d).round().clamp(-127.0, 127.0) as i8) as u8;
        }
        out.extend_from_slice(&q);
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
    /// Sealed pages in a shared [`crate::engine::kv_pool::KvPool`], when this
    /// layer is paged — `None` for the ordinary contiguous layout.
    ///
    /// When it is `Some`, `k`/`v` above stop being the whole layer and become
    /// only its **tail**: the rows of the page currently being written. Every
    /// row before that lives in the pool. The split is what lets a page be
    /// immutable the moment it is shareable — a sequence writes into its own
    /// buffer and hands the page over only when it is full, so the pool never
    /// holds anything that is still changing.
    paged: Option<PagedRows>,
    /// GPU-resident mirror of `k`/`v`, built lazily on the first call that
    /// needs it (a Vulkan-backed decode step) — `None` for every other
    /// backend/request. See [`Self::sync_gpu`].
    gpu: Option<GpuLayerCache>,
    /// Whether this layer's device-side keys and values are the pool's.
    ///
    /// Once true, `gpu` is a one-row stub that holds only softmax scratch and
    /// the cached dispatch — it has no rows and will never be uploaded to. So
    /// a reader that silently fell back to it would read zeros and produce a
    /// plausible answer, which is the failure this flag exists to make loud.
    pool_backed: bool,
}

/// The pages of **one sequence**, shared by every layer of it.
///
/// A page index addresses the same token positions in every layer — that is
/// what makes a block table per sequence rather than per layer — so a page has
/// to be taken from the pool exactly once, no matter how many layers then write
/// their own region of it. Getting this wrong is not subtle in its
/// consequences but is entirely invisible with one layer: the second layer to
/// seal finds the first layer's page already published, and either shares a
/// page it is about to overwrite or is refused by `fill`.
///
/// So acquisition, fill-counting and release live here, once, and the layers
/// coordinate through it.
struct SequencePages {
    pool: std::sync::Arc<crate::engine::kv_pool::KvPool>,
    /// Where this sequence's block table lives in the pool's shared table
    /// buffer, and how many entries were reserved — `None` when the pool has no
    /// device, or when the table buffer was full, in which case this sequence
    /// uses the per-request mirror and is correct but not shared on the device.
    table: Option<(usize, usize)>,
    inner: Mutex<SeqPages>,
}

#[derive(Default)]
struct SeqPages {
    /// Physical page per logical page, in order — the block table.
    pages: Vec<u32>,
    /// How many layers have written their region of each page. A page is
    /// sealed, and so becomes shareable, only when every layer has.
    filled: Vec<usize>,
    /// Content identity per logical page, from `engine::prefix_index`.
    tags: Vec<u64>,
    /// How many entries of this sequence's block table are on the device.
    table_synced: usize,
    /// Layers that have ever written a row.
    ///
    /// **Not every layer of the model writes.** A cross-layer KV donor has its
    /// own slot in the cache that is never pushed to — its writes redirect to
    /// the layer it donates from — so a page that waited for *all* layers
    /// would wait for a write that never arrives. It would never seal, never
    /// publish, and never be shareable, and nothing would report a fault: the
    /// index would go on advertising prefixes, every adoption would fail, and
    /// the feature would look switched off.
    ///
    /// Discovered rather than declared, because which layers write is a
    /// property of the architecture and not of the cache's geometry. Every
    /// token touches every participating layer, so the set is complete well
    /// before the first page fills.
    participants: std::collections::BTreeSet<usize>,
}

impl SequencePages {
    fn new(pool: std::sync::Arc<crate::engine::kv_pool::KvPool>, max_pages: usize) -> Self {
        // Reserved up front, for the whole sequence's possible length: growing
        // it later would move the region, and the base is baked into every
        // dispatch's meta uniform.
        let table = pool
            .device_pages()
            .and_then(|_| pool.alloc_table(max_pages))
            .map(|base| (base, max_pages));
        Self {
            pool,
            table,
            inner: Mutex::new(SeqPages::default()),
        }
    }

    /// Pushes the block table to the device if it has changed since the last
    /// upload, and reports where the kernel should read it.
    ///
    /// The table is per *sequence*, not per layer — a page index means the same
    /// token positions in every layer — so this uploads once however many
    /// layers ask.
    fn sync_table(&self, queue: &wgpu::Queue) -> Option<(usize, usize)> {
        let (base, cap) = self.table?;
        let mut inner = self.inner.lock().expect("sequence pages poisoned");
        if inner.table_synced != inner.pages.len() {
            if inner.pages.len() > cap {
                // More pages than were reserved: this sequence outgrew its
                // table. Correct to decline — the caller falls back to the
                // mirror — and silent growth would overwrite a neighbour's.
                return None;
            }
            self.pool.write_table(queue, base, &inner.pages);
            inner.table_synced = inner.pages.len();
        }
        Some((base, inner.pages.len()))
    }

    /// Notes that `layer` writes rows, the first time it does.
    fn joins(&self, layer: usize) {
        self.inner
            .lock()
            .expect("sequence pages poisoned")
            .participants
            .insert(layer);
    }

    /// The page for logical index `i`, and whether it already holds this
    /// content.
    ///
    /// A hit means some other sequence has already built and published exactly
    /// these positions — same tag, and the index confirmed the token run behind
    /// it, so the keys and values are the ones this sequence just computed.
    /// It takes that page and throws its own copy away.
    ///
    /// This is not an edge case to be avoided. It is the ordinary outcome
    /// whenever a request rebuilds a page it was deliberately not given: the
    /// prefix match leaves the last matched page unshared so the forward pass
    /// has something to produce logits from, and that page is still resident.
    /// It also covers two requests racing to build the same page, where
    /// whichever seals second finds the first one's work waiting.
    ///
    /// A hit is always a *sealed* page — the tag lookup contains nothing else —
    /// so it is complete and immutable, and adopting it cannot pick up
    /// half-written rows.
    fn page_for(&self, i: usize) -> (u32, bool) {
        let mut inner = self.inner.lock().expect("sequence pages poisoned");
        if let Some(&page) = inner.pages.get(i) {
            // Another layer of this sequence got here first.
            return (page, self.pool.is_sealed(page));
        }
        debug_assert_eq!(i, inner.pages.len(), "pages are sealed in order");
        let tag = inner.tags.get(i).copied().unwrap_or(0);
        let got = self
            .pool
            .acquire(&[tag])
            .expect("the scheduler admits a request only against pool room")[0];
        inner.pages.push(got.page);
        // A page adopted whole is already complete; counting it as filled by
        // every participant keeps `seal_complete` from trying to publish it a
        // second time.
        let complete = inner.participants.len();
        inner.filled.push(if got.hit { complete } else { 0 });
        (got.page, got.hit)
    }

    /// Records that one more layer has written `i`, sealing it once all have.
    fn layer_filled(&self, i: usize) {
        let mut inner = self.inner.lock().expect("sequence pages poisoned");
        inner.filled[i] += 1;
        // **Deliberately no seal here.** Sealing publishes a page for sharing,
        // and a page is only publishable once every layer that will write it
        // has. This tried to detect that by counting fills against the set of
        // layers that write — and that assumes the layers reach a page
        // boundary together, which they do not.
        //
        // `arch::gemma` is the counter-example: its cross-layer KV donors send
        // several model layers' writes into one `LayerCache`, so that cache
        // advances through pages at a different rate from its neighbours. The
        // count reached the participant total while other layers were still
        // several tokens behind, the page was published, and the next layer to
        // write it hit `fill`'s "already sealed" assertion — which is the
        // assertion doing its job, catching a page about to be handed out
        // half-written rather than letting it be shared.
        //
        // The fix is not a cleverer count. Whether token positions are
        // complete is a fact the *forward pass* knows and the cache cannot
        // infer, so it has to be told — a commit point per prefill chunk and
        // per decode step. Until that exists, pages stay unsealed: paging
        // works and is private, `KvPool::holds` reports nothing, and the index
        // advertises nothing it cannot deliver.
        let _ = i;
    }

    /// Drops every page past `keep`, once — idempotent, because every layer
    /// rolls back to the same token count and each of them asks.
    fn truncate_to(&self, keep: usize) {
        let mut inner = self.inner.lock().expect("sequence pages poisoned");
        if inner.pages.len() <= keep {
            return;
        }
        let released: Vec<u32> = inner.pages.drain(keep..).collect();
        inner.filled.truncate(keep);
        // The device copy of the table now names pages this sequence has given
        // back. Marking it stale is not an optimisation: leaving it would have
        // the kernel read whoever takes those pages next.
        inner.table_synced = inner.table_synced.min(inner.pages.len());
        self.pool.release(&released);
    }

    /// Seals every page every participating layer has finished writing.
    ///
    /// The commit point. Whether a page is complete is a fact about the forward
    /// pass — it knows when it has run every layer over a span of positions —
    /// and the cache cannot infer it, which is what the removed auto-seal was
    /// wrongly trying to do. So the caller says when, and this seals whatever
    /// has genuinely been finished by then.
    ///
    /// Idempotent: a page already sealed is skipped, so calling this after
    /// every chunk costs a scan of the fill counts and nothing else.
    fn seal_complete(&self) {
        let inner = self.inner.lock().expect("sequence pages poisoned");
        let participants = inner.participants.len();
        if participants == 0 {
            return;
        }
        for (i, &page) in inner.pages.iter().enumerate() {
            if inner.filled[i] == participants && !self.pool.is_sealed(page) {
                self.pool.seal(page);
            }
        }
    }

    fn adopt(&self, pages: &[u32]) {
        let mut inner = self.inner.lock().expect("sequence pages poisoned");
        inner.pages = pages.to_vec();
        inner.table_synced = 0;
        // Adopted pages are already sealed by whoever built them; counting them
        // as fully filled keeps `layer_filled` from sealing them a second time.
        let participants = inner.participants.len().max(1);
        inner.filled = vec![participants; pages.len()];
    }

    fn set_tags(&self, tags: &[u64]) {
        self.inner.lock().expect("sequence pages poisoned").tags = tags.to_vec();
    }

    fn pages(&self) -> Vec<u32> {
        self.inner
            .lock()
            .expect("sequence pages poisoned")
            .pages
            .clone()
    }
}

impl Drop for SequencePages {
    /// A sequence's pages go back when the sequence does. Without this a
    /// finished request's pages stay held for the life of the process, and the
    /// pool runs out while every page in it is reclaimable.
    fn drop(&mut self) {
        let inner = self.inner.lock().expect("sequence pages poisoned");
        self.pool.release(&inner.pages);
        if let Some((base, entries)) = self.table {
            self.pool.free_table(base, entries);
        }
    }
}

/// One layer's view of its sequence's pages.
struct PagedRows {
    seq: std::sync::Arc<SequencePages>,
    /// Which layer of the pool's geometry this is — pages are shared across
    /// layers, so a page index means "these token positions", and the layer
    /// selects which of its regions to read.
    layer: usize,
    /// Rows one page holds for *this* layer — `page_tokens / stride`, rounded
    /// up, so a block-compressed layer stores fewer rows per page than a
    /// per-token one.
    rows_per_page: usize,
    /// A local copy of the block table, so a read never takes the sequence
    /// lock. `key_at` is in the CPU attention inner loop; the shared list is
    /// consulted only when a page is sealed or released.
    pages: Vec<u32>,
    /// How many of `pages` are *complete* — every position they cover has been
    /// written.
    ///
    /// Not the same as `pages.len()`, and the difference is the tail. A page is
    /// allocated as soon as the sequence starts writing into it, because the
    /// attention kernels address positions through the block table and a
    /// position with no entry there reads whatever the table happens to hold.
    /// But it is not *complete* until its last row is written, and host reads
    /// must keep taking those rows from the local tail buffer, which is the
    /// only place they are authoritative.
    full_pages: usize,
    /// Rows of the *tail* page already on the device.
    ///
    /// The tail grows a row per decode step, and without this every step
    /// re-sent the whole page — `page_tokens` times the traffic the step
    /// actually produced, per layer.
    tail_uploaded: usize,
    /// How many of `pages` have been written to the pool's **device** pages.
    ///
    /// The device copy lags the host one: a page is filled on the host as its
    /// tail completes, and pushed to the device when something asks for the
    /// mirror. Tracked per layer because each layer writes its own region of a
    /// page and they do not finish together.
    device_synced: usize,
}

impl PagedRows {
    /// The row range of `pool_page` inside the pool's per-layer buffer.
    fn row(&self, pool_page: u32, row_in_page: usize) -> std::ops::Range<usize> {
        let start = self.seq.pool.row_offset(self.layer, pool_page, row_in_page);
        start..start + self.seq.pool.layers()[self.layer].kv_dim
    }
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
    /// Rows this mirror was allocated for — **not** the layer's `capacity`.
    ///
    /// The mirror used to be sized to the whole capacity on first use, which
    /// meant a request's `max_tokens` was reserved in VRAM whether or not it
    /// was ever generated. Measured on a 3.98 GiB card with a two-token
    /// prompt: 2191 MiB at `max_tokens = 64` against **3727 MiB** at
    /// `max_tokens = 32768`, for the same one-word answer. Host buffers do
    /// not have this problem — a large zeroed `Vec` is `mmap`ed and the
    /// kernel commits pages only as they are written — but device memory is
    /// not overcommitted, so on the GPU the reservation is real.
    rows: usize,
    /// Cached attention-dispatch resources, keyed by the *calling layer's*
    /// `wq` tensor identity (`QuantMatrix::cache_key()`) — see
    /// [`GpuAttnDispatch`]'s doc comment for why one `LayerCache` can need
    /// more than one entry here.
    #[allow(dead_code)]
    attn_dispatch: std::collections::HashMap<(usize, usize), GpuAttnDispatch>,
}

/// Everything the fused decode dispatch needs to serve one position out of
/// the pool's device pages instead of out of a per-request mirror.
///
/// The write and the read have to move together. A decode step writes this
/// token's key and value on the device and then, in the same submission,
/// reads the whole attention window back — so the destination of the write and
/// the addressing the kernel reads through must name the same storage. That is
/// why this carries both the region layout (`layer_buffer`, `half`) and the
/// one physical row this step writes (`write_slot`), rather than the caller
/// deriving either.
pub struct PagedFusedRefs {
    /// The pool's buffer for this layer: keys from zero, values from `half`.
    pub layer_buffer: wgpu::Buffer,
    /// The block table, read by the kernel to turn a position into a row.
    pub table: wgpu::Buffer,
    /// Byte offset of the value region, and equally the size of each region.
    pub half: u64,
    /// This sequence's first entry in the shared table.
    pub table_base: u32,
    /// Positions per page — the divisor in the kernel's address computation.
    pub page_tokens: u32,
    /// The physical row `write_pos` maps to, in rows from each region's base.
    pub write_slot: u32,
}

/// One contiguous stretch of a write that stays inside a single page.
///
/// A range of positions is contiguous on the host and contiguous inside each
/// page, but not across a page boundary — so a write of `n` positions becomes
/// one of these per page it touches, and each is a straight copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRun {
    /// Rows into the source, from the start of the range being written.
    pub src_row: u32,
    /// Rows into each of the layer buffer's regions.
    pub dst_row: u32,
    /// How many rows this run covers.
    pub rows: u32,
}

/// Splits `[write_pos, write_pos + n_tokens)` into runs that each stay inside
/// one page.
///
/// `pages` is the block table — logical page index to physical page — and must
/// already cover the last position in the range.
fn page_runs(
    pages: &[u32],
    write_pos: usize,
    n_tokens: usize,
    rows_per_page: usize,
) -> Vec<PageRun> {
    let mut runs = Vec::new();
    let mut done = 0usize;
    while done < n_tokens {
        let p = write_pos + done;
        let in_page = p % rows_per_page;
        let rows = (rows_per_page - in_page).min(n_tokens - done);
        runs.push(PageRun {
            src_row: done as u32,
            dst_row: pages[p / rows_per_page] * rows_per_page as u32 + in_page as u32,
            rows: rows as u32,
        });
        done += rows;
    }
    runs
}

/// [`PagedFusedRefs`] for a range of positions rather than one.
pub struct PagedRangeRefs {
    pub layer_buffer: wgpu::Buffer,
    pub table: wgpu::Buffer,
    pub half: u64,
    pub table_base: u32,
    pub page_tokens: u32,
    /// The range split at page boundaries, in order.
    pub runs: Vec<PageRun>,
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

/// The storage-buffer offset alignment [`gpu_layer_bytes`] assumes when no
/// device has been created yet.
///
/// 256 bytes is what every desktop driver this runs on reports, and it is
/// `wgpu`'s own portable default. Using it without a device can only
/// *overstate* the padding between the `k` and `v` regions (by at most 252
/// bytes per layer), which is the safe direction for a budget.
const ASSUMED_STORAGE_ALIGN: u64 = 256;

/// The device bytes one layer's GPU mirror occupies at `rows` stored rows.
///
/// Deliberately adjacent to [`GpuLayerCache::new`], which is the allocation
/// this has to agree with: two `kv_bytes` regions packed into one buffer at
/// an aligned offset, plus the `[rows * n_head]` f32 attention scratch. A
/// budget computed from a stale copy of that layout is worse than no budget
/// at all, so `kv_mirror_bytes_agree_with_the_allocation` holds the two
/// together.
fn gpu_layer_bytes(
    rows: usize,
    kv_dim: usize,
    n_head: usize,
    kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    align: u64,
) -> u64 {
    let kv_bytes: u64 = match kv_storage {
        crate::engine::backend::vulkan_shaders::KvStorage::F32 => (rows * kv_dim * 4) as u64,
        crate::engine::backend::vulkan_shaders::KvStorage::F16 => (rows * kv_dim * 2) as u64,
        crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => (rows * kv_dim / 32 * 36) as u64,
    }
    .max(1);
    let v_off = kv_bytes.next_multiple_of(align.max(4));
    v_off + kv_bytes + ((rows * n_head).max(1) * 4) as u64
}

impl GpuLayerCache {
    /// Allocates a mirror for exactly `rows` positions.
    ///
    /// `rows` is what [`LayerCache::sync_gpu`] decided to grow to, never the
    /// layer's capacity — see [`Self::rows`].
    fn new(
        device: &wgpu::Device,
        rows: usize,
        kv_dim: usize,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> Self {
        Self::new_sized(device, rows, rows, kv_dim, n_head, kv_storage)
    }

    /// A mirror whose key/value region and whose softmax scratch are sized
    /// independently.
    ///
    /// The paged decode path needs the second without the first. Its keys and
    /// values live in the pool's pages, so the per-request mirror holds
    /// nothing — but the attention dispatch it caches, and the scratch that
    /// dispatch writes its per-position softmax terms into, are still
    /// per-request. Sizing the mirror to one row rather than to the context
    /// window is what stops a paged sequence from paying for a copy of a cache
    /// it reads through the block table instead.
    fn new_sized(
        device: &wgpu::Device,
        rows: usize,
        probs_rows: usize,
        kv_dim: usize,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> Self {
        let capacity = rows;
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
            // `COPY_SRC` as well as `COPY_DST`: growing the mirror copies the
            // rows already on the device straight across rather than
            // re-uploading them from the host, which would put the whole
            // cache back over the bus every time it doubled.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probs_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orangu-server kv cache attention scratch"),
            size: ((probs_rows * n_head).max(1) * 4) as u64,
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
            rows,
            attn_dispatch: std::collections::HashMap::new(),
        }
    }
}

/// How many rows a mirror holding `len` positions is allocated for.
///
/// Doubling from a small floor, capped at the layer's own capacity. Doubling
/// rather than a fixed block because the cost of being wrong is a buffer
/// reallocation plus a device-side copy: a fixed block size would pay that
/// every `BLOCK` tokens for the whole generation, where doubling pays it
/// `log2` times in total and then never again.
///
/// The floor keeps a short answer — the overwhelmingly common case — to one
/// allocation, and the cap means a request that really does run to its full
/// budget ends up with exactly what it would have had before, having paid a
/// handful of copies to get there.
/// Bytes `rows` stored positions occupy in one k or v region.
///
/// The same arithmetic [`GpuLayerCache::new`] sizes a region with, named once
/// so the grow-copy cannot disagree with the allocation it is copying between.
fn row_bytes(
    rows: usize,
    kv_dim: usize,
    kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
) -> u64 {
    match kv_storage {
        crate::engine::backend::vulkan_shaders::KvStorage::F32 => (rows * kv_dim * 4) as u64,
        crate::engine::backend::vulkan_shaders::KvStorage::F16 => (rows * kv_dim * 2) as u64,
        crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => (rows * kv_dim / 32 * 36) as u64,
    }
}

fn mirror_rows_for(rows_needed: usize, capacity: usize) -> usize {
    let len = rows_needed;
    const FLOOR: usize = 256;
    let mut rows = FLOOR;
    while rows < len {
        rows = rows.saturating_mul(2);
    }
    rows.min(capacity).max(len).max(1)
}

impl LayerCache {
    /// An ordinary per-token slot — `new_strided` at stride 1.
    ///
    /// Test-only since every constructor was funnelled through
    /// [`KvCache::build`]: production has one shape of call and this spelling
    /// only survives because a test that says `new(4, 6)` reads better than
    /// one that says `new_strided(4, 6, 1)` and leaves the reader wondering
    /// what the 1 was about.
    #[cfg(test)]
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
            // Reserved, not filled. `k.len()` tracks the committed rows from
            // here on, which is what lets a reused prefix hand its buffers
            // over wholesale instead of being copied into a pre-sized one —
            // see [`Self::adopt`]. Reserving the whole context up front still
            // costs nothing until it is written: a large allocation is
            // `mmap`ed and the kernel commits pages lazily, measured at 0.0
            // MiB of RSS for two gigabytes.
            k: Vec::with_capacity(rows * kv_dim),
            v: Vec::with_capacity(rows * kv_dim),
            kv_dim,
            capacity: rows,
            len: 0,
            gpu: None,
            stride,
            paged: None,
            pool_backed: false,
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
            paged: None,
            // A restored layer is only ever a `copy_prefix_from` *source*,
            // and only the destination's stride governs how many rows a
            // token count means, so this never needs the original's.
            stride: 1,
            pool_backed: false,
        }
    }

    /// A CPU-only deep copy (no GPU mirror). Used by slot persistence to
    /// snapshot a completed cache for a slot that is also being deposited
    /// into the [`crate::engine::prefix_cache`] pool — the one case where a
    /// single completed cache is needed in two places at once.
    fn duplicate(&self) -> Self {
        Self {
            // Flattened, not cloned: under paging `k`/`v` hold only the tail,
            // and a snapshot of the tail is not a snapshot of the layer.
            k: self.flatten(self.host_len()).0,
            v: self.flatten(self.host_len()).1,
            kv_dim: self.kv_dim,
            capacity: self.capacity,
            len: self.len,
            gpu: None,
            stride: self.stride,
            // A duplicate is a plain host snapshot: it is taken so a completed
            // cache can be in two places at once, and sharing pages between
            // those two would defeat the point of taking it.
            paged: None,
            pool_backed: false,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Floats in one stored row — this layer's `n_head_kv * head_dim`.
    ///
    /// Exposed so a pool can be sized from a model's own probe cache
    /// (`ModelForward::new_kv_cache(1)`) rather than from a second description
    /// of the same geometry. A second description is a second thing to keep in
    /// step, and this one varies along a model's depth in ways
    /// (`head_count_kv` per layer, a block-compressed layer, a layer with no
    /// positional state at all) that a summary would flatten.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Token positions one stored row stands for — see [`Self::stride`].
    pub fn row_stride(&self) -> usize {
        self.stride
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
        if self.paged.is_some() {
            self.truncate_paged(new_len);
        } else {
            // The buffers carry the committed rows and nothing else now, so
            // rolling back the length has to roll them back too.
            // `Vec::truncate` keeps the allocation, which is right: the rows
            // are about to be written again by whatever replaces them.
            self.k.truncate(new_len * self.kv_dim);
            self.v.truncate(new_len * self.kv_dim);
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.synced_len = gpu.synced_len.min(new_len);
        }
    }

    /// Takes `src`'s committed rows as this layer's own, without copying them.
    ///
    /// The buffers move; `self` keeps its own `capacity`, `kv_dim` and
    /// `stride`, because those describe *this* request and the rows are just
    /// bytes. The GPU mirror is dropped for the same reason
    /// [`Self::copy_prefix_from`] drops it — it belonged to the old cache's
    /// buffers and has to be rebuilt against these.
    ///
    /// Only valid where the caller owns `src` outright.
    /// `engine::prefix_cache::PrefixCache::take_best_match` *removes* the
    /// entry it returns, so the reuse path does; `engine::slot_store` retains
    /// its snapshot for the slot's next request and therefore cannot, and
    /// still copies.
    fn adopt(&mut self, src: &mut LayerCache) {
        debug_assert_eq!(self.kv_dim, src.kv_dim);
        assert!(
            self.paged.is_none() && src.paged.is_none(),
            "adopting moves whole buffers between two layers, which only \
             describes the contiguous layout; a paged layer hands over page \
             indices instead, and mixing the two would leave one of them \
             reading the other's tail as if it were a whole layer"
        );
        self.k = std::mem::take(&mut src.k);
        self.v = std::mem::take(&mut src.v);
        self.len = src.len;
        // One reallocation now, at a known point, rather than one during
        // whichever decode step first runs past the old request's ceiling.
        let want = self.capacity * self.kv_dim;
        self.k.reserve(want.saturating_sub(self.k.len()));
        self.v.reserve(want.saturating_sub(self.v.len()));
        self.gpu = None;
        // The stub goes with the mirror it stood in for.
        self.pool_backed = false;
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

    /// Lazily builds this layer's GPU-resident mirror — **grown to the rows
    /// actually in use, not to the layer's capacity** (`n_head` is a fixed
    /// model property, always the same across every call for a given layer)
    /// — and uploads any
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
        assert!(
            !self.pool_backed,
            "sync_gpu on a layer served from the pool: its mirror holds no rows"
        );
        let capacity = self.capacity;
        let kv_dim = self.kv_dim;
        // `len + 1`, not `len`: the fused decode path binds these buffers and
        // then writes the *current* token's key and value at row `len`, before
        // the host-side `push` that will make that row committed. Sizing to
        // `len` leaves that write one row past the end of the k region — which
        // lands inside the shared k|v buffer rather than outside it, so the
        // driver never objects and the damage is silent: it lands on row 0 of
        // v. Sizing to `capacity` used to hide this, because capacity always
        // exceeds `len`.
        // A paged layer's rows are spread across pages, so the incremental
        // upload below cannot slice `k` directly and materializes the range
        // instead (see `rows_between`). That path is now only reached by a
        // sequence that could get no pages at all — every dispatch that can
        // read through the block table does, and the assertion above is what
        // holds the two apart.
        let want = mirror_rows_for(self.len + 1, capacity);
        // Grow before syncing, never shrink. A mirror that is already big
        // enough is left exactly as it is, so the steady state — every decode
        // step after the first — does no work here at all.
        match &self.gpu {
            None => {
                self.gpu = Some(GpuLayerCache::new(device, want, kv_dim, n_head, kv_storage));
            }
            Some(gpu) if gpu.rows < self.len + 1 => {
                let old = self.gpu.take().expect("checked present");
                let mut grown = GpuLayerCache::new(device, want, kv_dim, n_head, kv_storage);
                // Carry the rows already on the device across on the device.
                // They are identical bytes in an identical layout — only the
                // region length changed — so this is two straight copies, and
                // it keeps `synced_len` meaningful instead of forcing the
                // whole cache back over the bus.
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("orangu-server kv mirror grow"),
                });
                // Clamped to what the source actually holds. `synced_len`
                // should never exceed it, and if a future path makes it
                // possible again this copies a short prefix instead of reading
                // off the end of a buffer.
                let carried = old.synced_len.min(old.rows);
                let synced_bytes = row_bytes(carried, kv_dim, kv_storage);
                if synced_bytes > 0 {
                    encoder.copy_buffer_to_buffer(
                        &old.kv_buffer,
                        old.k_off,
                        &grown.kv_buffer,
                        grown.k_off,
                        synced_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        &old.kv_buffer,
                        old.v_off,
                        &grown.kv_buffer,
                        grown.v_off,
                        synced_bytes,
                    );
                }
                queue.submit(std::iter::once(encoder.finish()));
                grown.synced_len = carried;
                // The cached bind groups name the *old* buffer, so they are
                // stale the moment it is replaced. Dropped rather than
                // rebuilt: the caller rebuilds on a miss, and rebuilding here
                // would need the resources only it has.
                self.gpu = Some(grown);
            }
            Some(_) => {}
        }
        // The bytes to upload are built **before** the mirror is borrowed
        // mutably, because on the paged path assembling them reads back
        // through `self`. Owned either way, which the two quantizing arms
        // already were; only the `f32` arm previously handed the queue a
        // borrow of `k`, and that is one `memcpy` of the range being uploaded
        // — a row per decode step.
        let storage = self.gpu.as_ref().map(|g| g.kv_storage);
        let (row_from, row_to) = (self.gpu.as_ref().map_or(0, |g| g.synced_len), self.len);
        let payload = (row_from < row_to).then(|| {
            let k_rows = self.rows_between(row_from, row_to);
            let v_rows = self.values_between(row_from, row_to);
            match storage.expect("mirror present when rows are pending") {
                crate::engine::backend::vulkan_shaders::KvStorage::F16 => {
                    (f32_to_f16_bytes(&k_rows), f32_to_f16_bytes(&v_rows))
                }
                crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => {
                    (f32_to_q8_0_bytes(&k_rows), f32_to_q8_0_bytes(&v_rows))
                }
                crate::engine::backend::vulkan_shaders::KvStorage::F32 => (
                    bytemuck::cast_slice(&k_rows).to_vec(),
                    bytemuck::cast_slice(&v_rows).to_vec(),
                ),
            }
        });
        let gpu = self.gpu.as_mut().expect("mirror present after growth");
        if let Some((k_bytes, v_bytes)) = payload {
            let start = row_from * kv_dim;
            // Local byte offset of `start` within each region, by storage
            // format; `k_off`/`v_off` shift it to the region's base in the
            // shared buffer.
            let local = match gpu.kv_storage {
                crate::engine::backend::vulkan_shaders::KvStorage::F16 => (start * 2) as u64,
                crate::engine::backend::vulkan_shaders::KvStorage::Q8_0 => (start / 32 * 36) as u64,
                crate::engine::backend::vulkan_shaders::KvStorage::F32 => (start * 4) as u64,
            };
            queue.write_buffer(&gpu.kv_buffer, gpu.k_off + local, &k_bytes);
            queue.write_buffer(&gpu.kv_buffer, gpu.v_off + local, &v_bytes);
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
        debug_assert_eq!(self.k.len(), (self.len - self.sealed_rows()) * self.kv_dim);
        if let Some(paged) = self.paged.as_ref()
            && self.k.is_empty()
            && paged.pages.is_empty()
        {
            // First write from this layer: it is a participant, and a page is
            // sealed once every participant has written it.
            paged.seq.joins(paged.layer);
        }
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.len += 1;
        // A full tail becomes a page. Done here rather than lazily on the next
        // read so that a page is sealed at the moment it stops changing, which
        // is the property everything else about the pool is built on.
        if let Some(paged) = self.paged.as_ref()
            && self.k.len() / self.kv_dim == paged.rows_per_page
        {
            self.seal_tail();
        }
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

    /// This layer's committed length **in tokens**.
    ///
    /// [`len`](Self::len) counts *rows*, and a block-compressed row stands for
    /// [`stride`](Self::stride) tokens, so the two are the same number only on
    /// an ordinary per-token slot. Every other length-converting operation
    /// here — [`truncate`](Self::truncate), [`copy_prefix_from`] — already
    /// goes through `stride`; this is the read-side counterpart, so a caller
    /// asking "how many tokens does this hold" cannot get rows back.
    fn committed_tokens(&self) -> usize {
        self.len * self.stride
    }

    /// A layer backed by pages from `pool`, with its tail held locally.
    ///
    /// `capacity` is still a token count, and still this sequence's own — the
    /// pool bounds how much can be resident across every sequence at once, and
    /// this bounds how far *this* one may run. Two different limits; a pool
    /// with room does not entitle one request to the whole context.
    #[allow(dead_code)]
    fn new_paged(
        capacity: usize,
        kv_dim: usize,
        stride: usize,
        seq: std::sync::Arc<SequencePages>,
        layer: usize,
    ) -> Self {
        let mut me = Self::new_strided(capacity, kv_dim, stride);
        let rows_per_page = seq.pool.page_tokens().div_ceil(stride);
        me.paged = Some(PagedRows {
            seq,
            layer,
            rows_per_page,
            pages: Vec::new(),
            full_pages: 0,
            tail_uploaded: 0,
            device_synced: 0,
        });
        me
    }

    /// Pushes any pages this layer has sealed but not yet mirrored into the
    /// pool's device pages.
    ///
    /// Kept apart from [`sync_gpu`](Self::sync_gpu) rather than folded into it,
    /// deliberately. That one maintains the **per-request** mirror the
    /// contiguous kernels read; this one maintains the **shared** pages the
    /// paged kernels read. Both are correct, they are read by different
    /// pipelines, and a single method that decided between them would make the
    /// choice of pipeline and the choice of storage two things that could
    /// disagree — with the failure being a kernel reading the wrong buffer and
    /// answering from another sequence's tokens.
    ///
    /// Only whole sealed pages are pushed. The tail is still being written and
    /// is not part of any block table yet.
    pub fn sync_pool_device(&mut self, queue: &wgpu::Queue) {
        let Some(paged) = self.paged.as_mut() else {
            return;
        };
        if paged.seq.pool.device_pages().is_none() {
            return;
        }
        let rows = paged.rows_per_page;
        let kv_dim = self.kv_dim;
        // The tail needs a page too. The kernels resolve *every* position they
        // read through the block table, including the ones in a page that is
        // not finished, so a tail with no entry there sends them to whatever
        // the table happens to hold. Allocated here rather than at seal time,
        // and left unsealed — unsealed is what keeps it private, so a partial
        // page can be on the device without being shareable.
        let tail_rows = self.k.len() / kv_dim.max(1);
        if tail_rows > 0 && paged.pages.len() == paged.full_pages {
            let (page, _) = paged.seq.page_for(paged.full_pages);
            paged.pages.push(page);
        }
        // Only the rows added since the last call. The tail grows one row per
        // decode step; sending the whole page each time was the same bytes as
        // sixteen steps' worth, per layer, per token.
        if tail_rows > paged.tail_uploaded {
            let page = paged.pages[paged.full_pages];
            let from = paged.tail_uploaded * kv_dim;
            let to = tail_rows * kv_dim;
            paged.seq.pool.fill_device_rows(
                queue,
                paged.layer,
                page,
                paged.tail_uploaded,
                &self.k[from..to],
                &self.v[from..to],
            );
            paged.tail_uploaded = tail_rows;
        }
        // Sealed pages, uploaded in **runs of consecutive physical pages**.
        //
        // Two things make the naive form expensive, and a prefill pays both
        // once per page per layer. A page's rows are already contiguous in the
        // pool's host buffer, so copying them row by row into a temporary was
        // building a copy of something that could be sliced; and pages come out
        // of the free list in ascending order most of the time, so a whole
        // prefill is usually one run rather than sixty-four separate transfers.
        //
        // Measured on a model whose layers are all full-attention: time to
        // first token at depth 1024 was three times the contiguous path's, and
        // did not move when the page size changed — which is what said the cost
        // was per *transfer* rather than per lookup.
        let mut i = paged.device_synced;
        while i < paged.full_pages {
            let first = paged.pages[i];
            let mut n = 1;
            while i + n < paged.full_pages && paged.pages[i + n] == first + n as u32 {
                n += 1;
            }
            let per = rows * kv_dim;
            let base = first as usize * per;
            let span = n * per;
            paged.seq.pool.fill_device_rows(
                queue,
                paged.layer,
                first,
                0,
                &paged.seq.pool.page_k_all(paged.layer)[base..base + span],
                &paged.seq.pool.page_v_all(paged.layer)[base..base + span],
            );
            i += n;
        }
        paged.device_synced = paged.full_pages;
    }

    /// Everything a paged attention dispatch needs, or `None` when this layer
    /// is not served that way.
    ///
    /// `None` covers three separate cases, and all three are correct rather
    /// than degraded: the layer is contiguous, the pool has no device, or the
    /// sequence could not reserve a block table. In each the caller uses the
    /// per-request mirror, which is the path that has always run.
    ///
    /// Pushes any newly sealed pages and the table to the device as a side
    /// effect, because a dispatch is exactly the moment they have to be there
    /// and nothing else knows when that is.
    pub fn paged_device_refs(
        &mut self,
        queue: &wgpu::Queue,
    ) -> Option<(wgpu::Buffer, wgpu::Buffer, u64, u32, u32)> {
        self.sync_pool_device(queue);
        let paged = self.paged.as_ref()?;
        let pool = &paged.seq.pool;
        let pages = pool.device_pages()?;
        let (base, len) = paged.seq.sync_table(queue)?;
        // The table must already cover every page this layer will read.
        if len < paged.pages.len() {
            return None;
        }
        let (k_off, v_off, _) = pool.device_page_offsets(paged.layer, 0)?;
        Some((
            pages.layers[paged.layer].clone(),
            pages.table.clone(),
            v_off - k_off,
            base as u32,
            pool.page_tokens() as u32,
        ))
    }

    /// Reserves this position's row in the pool's device pages and hands back
    /// everything a fused decode step needs to write it and read it.
    ///
    /// `None` where [`Self::paged_device_refs`] is `None`, and for one case of
    /// its own: a layer whose rows and the pool's positions are not one to one
    /// (a block-compressed slot) addresses pages differently from what the
    /// kernel's `position / page_tokens` computes, so it keeps the mirror.
    ///
    /// The reservation is the part that has no counterpart on the host path.
    /// Every other writer pushes a row to the host first and the page is
    /// allocated because there is a tail to upload; this one writes on the
    /// device and there may be no tail at all — at a page boundary the
    /// previous page has just been sealed and the next does not exist yet. So
    /// the page is taken here.
    ///
    /// Those pages are never sealed, and that is deliberate rather than a gap.
    /// A sealed page is offered to every other sequence, and these rows exist
    /// only on the device — the host copy the pool would publish is still
    /// zeroed. Leaving them unsealed keeps them private to this sequence,
    /// which is the same bound `PrefixCache` already applies when it reuses
    /// only up to `host_committed_len`.
    pub fn paged_fused_refs(
        &mut self,
        queue: &wgpu::Queue,
        write_pos: usize,
    ) -> Option<PagedFusedRefs> {
        let refs = self.paged_fused_refs_inner(queue, write_pos);
        assert!(
            refs.is_some() || !self.pool_backed,
            "a layer already served from the pool cannot fall back to its mirror: \
             the mirror was deliberately left at one row and never uploaded to, \
             so reading it would silently answer from zeros"
        );
        refs
    }

    fn paged_fused_refs_inner(
        &mut self,
        queue: &wgpu::Queue,
        write_pos: usize,
    ) -> Option<PagedFusedRefs> {
        let r = self.paged_range_refs(queue, write_pos, 1)?;
        let run = *r.runs.first()?;
        Some(PagedFusedRefs {
            layer_buffer: r.layer_buffer,
            table: r.table,
            half: r.half,
            table_base: r.table_base,
            page_tokens: r.page_tokens,
            write_slot: run.dst_row,
        })
    }

    /// [`Self::paged_fused_refs`] for a whole range of positions.
    ///
    /// The prefill path writes `n_tokens` positions at once. On the host and
    /// inside any one page those rows are contiguous, but a range that crosses
    /// a page boundary is not contiguous in the pool — so it comes back split
    /// into runs, each of which the caller can write as a straight copy.
    ///
    /// Same reservation and same never-sealed rule as the one-position form:
    /// the pages are taken here because the device write precedes any host
    /// row, and they stay private until a host-side writer fills and seals
    /// them.
    pub fn paged_range_refs(
        &mut self,
        queue: &wgpu::Queue,
        write_pos: usize,
        n_tokens: usize,
    ) -> Option<PagedRangeRefs> {
        self.sync_pool_device(queue);
        let rows_per_page = self.paged.as_ref()?.rows_per_page;
        let page_tokens = self.paged.as_ref()?.seq.pool.page_tokens();
        if rows_per_page == 0 || rows_per_page != page_tokens || n_tokens == 0 {
            return None;
        }
        let last = (write_pos + n_tokens - 1) / rows_per_page;
        {
            let paged = self.paged.as_mut()?;
            while paged.pages.len() <= last {
                let (page, _) = paged.seq.page_for(paged.pages.len());
                paged.pages.push(page);
            }
        }
        let paged = self.paged.as_ref()?;
        let pool = &paged.seq.pool;
        let pages = pool.device_pages()?;
        let (base, len) = paged.seq.sync_table(queue)?;
        if len < paged.pages.len() {
            return None;
        }
        let (k_off, v_off, _) = pool.device_page_offsets(paged.layer, 0)?;

        let runs = page_runs(&paged.pages, write_pos, n_tokens, rows_per_page);
        Some(PagedRangeRefs {
            layer_buffer: pages.layers[paged.layer].clone(),
            table: pages.table.clone(),
            half: v_off - k_off,
            table_base: base as u32,
            page_tokens: page_tokens as u32,
            runs,
        })
    }

    /// The per-request scratch a paged decode step still needs: the softmax
    /// partials buffer, and the home for its cached attention dispatch.
    ///
    /// This is [`Self::sync_gpu`] with the mirror taken out. Nothing uploads,
    /// and the key/value region is one row rather than the context window,
    /// because on this path the keys and values are the pool's.
    pub fn pool_scratch(
        &mut self,
        device: &wgpu::Device,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) {
        let probs_rows = self.capacity;
        let kv_dim = self.kv_dim;
        self.pool_backed = true;
        if self.gpu.is_none() {
            self.gpu = Some(GpuLayerCache::new_sized(
                device, 1, probs_rows, kv_dim, n_head, kv_storage,
            ));
        }
        // Nothing here will ever upload, so the watermark is simply the truth:
        // there are no host rows this mirror is behind on.
        if let Some(gpu) = &mut self.gpu {
            gpu.synced_len = self.len;
        }
    }

    /// Whether this layer's device keys and values are the pool's.
    ///
    /// A white-box assertion, for the same reason `device_synced_pages` is one:
    /// a test that only compares outputs cannot tell the paged path from the
    /// mirrored fallback, because the fallback is materialized from the very
    /// pages the paged path reads and is therefore correct either way.
    #[cfg(test)]
    pub fn is_pool_backed(&self) -> bool {
        self.pool_backed
    }

    /// The mirror's softmax scratch, once one exists.
    pub fn probs_scratch(&self) -> Option<wgpu::Buffer> {
        self.gpu.as_ref().map(|g| g.probs_scratch.clone())
    }

    /// This layer's block table — the physical page per logical page.
    #[allow(dead_code)]
    pub fn block_table(&self) -> &[u32] {
        self.paged.as_ref().map_or(&[], |p| &p.pages)
    }

    /// How many pages have been mirrored to the device.
    ///
    /// Exposed for a test that would otherwise be unable to see the thing it
    /// claims to check: the watermark's only observable effect is which pages
    /// a later `sync_pool_device` uploads, and reading the device back to
    /// discover that needs a `f16` readback path this module does not have. A
    /// white-box assertion that names the invariant beats an end-to-end one
    /// that cannot fail.
    #[cfg(test)]
    pub fn device_synced_pages(&self) -> usize {
        self.paged.as_ref().map_or(0, |p| p.device_synced)
    }

    /// Rows `[from, to)` as a contiguous run, borrowed when they already are.
    ///
    /// The upload path wants one slice per range; a paged layer has one slice
    /// per page. Rather than teach every caller about pages, this materializes
    /// the range — which is a copy the contiguous path does not pay, bounded by
    /// the range being uploaded (one row per decode step, one prefill's worth
    /// on the first sync).
    fn rows_between(&self, from: usize, to: usize) -> std::borrow::Cow<'_, [f32]> {
        self.rows_between_side(from, to, true)
    }

    fn values_between(&self, from: usize, to: usize) -> std::borrow::Cow<'_, [f32]> {
        self.rows_between_side(from, to, false)
    }

    fn rows_between_side(&self, from: usize, to: usize, keys: bool) -> std::borrow::Cow<'_, [f32]> {
        if self.paged.is_none() {
            let buf = if keys { &self.k } else { &self.v };
            return std::borrow::Cow::Borrowed(&buf[from * self.kv_dim..to * self.kv_dim]);
        }
        let mut out = Vec::with_capacity((to - from) * self.kv_dim);
        for r in from..to {
            out.extend_from_slice(if keys { self.row_k(r) } else { self.row_v(r) });
        }
        std::borrow::Cow::Owned(out)
    }

    /// The first `rows` rows as one contiguous buffer, whichever layout they
    /// are in — for the paths that genuinely need a flat copy (a snapshot, a
    /// serialization, a prefix hand-over) rather than a row at a time.
    ///
    /// Under the contiguous layout this is the buffer itself and the clone is
    /// the same one those paths already made. Under paging it walks rows, which
    /// costs a copy those paths were already paying.
    fn flatten(&self, rows: usize) -> (Vec<f32>, Vec<f32>) {
        if self.paged.is_none() {
            let n = rows * self.kv_dim;
            return (self.k[..n].to_vec(), self.v[..n].to_vec());
        }
        let mut k = Vec::with_capacity(rows * self.kv_dim);
        let mut v = Vec::with_capacity(rows * self.kv_dim);
        for r in 0..rows {
            k.extend_from_slice(self.row_k(r));
            v.extend_from_slice(self.row_v(r));
        }
        (k, v)
    }

    /// Rows held in sealed pages — everything before the tail.
    fn sealed_rows(&self) -> usize {
        self.paged
            .as_ref()
            .map_or(0, |p| p.full_pages * p.rows_per_page)
    }

    /// The tail buffer's row count, which for a contiguous layer is the whole
    /// layer.
    fn tail_rows(&self) -> usize {
        if self.kv_dim == 0 {
            return 0;
        }
        self.k.len() / self.kv_dim
    }

    /// Seals the tail into a pool page once it is full, and starts a new one.
    ///
    /// Only ever called with a *full* tail. A partial tail is never handed to
    /// the pool: a page becomes shareable when it is sealed, and a page that is
    /// still being appended to is precisely what must not be shared.
    fn seal_tail(&mut self) {
        let Some(paged) = self.paged.as_mut() else {
            return;
        };
        debug_assert_eq!(
            self.k.len() / self.kv_dim,
            paged.rows_per_page,
            "only a full tail is sealed"
        );
        // The page for this logical index, taken from the pool by whichever
        // layer reaches it first. Every layer writes its own region of the same
        // page, and the last one to do so seals it.
        let index = paged.full_pages;
        let (page, already) = paged.seq.page_for(index);
        if paged.pages.len() == index {
            paged.pages.push(page);
        }
        if !already {
            paged.seq.pool.fill(paged.layer, page, &self.k, &self.v);
            paged.seq.layer_filled(index);
        }
        // Complete now: the rows it covers are all written, so host reads move
        // to the page and the tail starts empty again.
        paged.full_pages += 1;
        paged.tail_uploaded = 0;
        self.k.clear();
        self.v.clear();
    }

    /// Takes `tags`' pages as this layer's leading pages, without computing
    /// them.
    ///
    /// This is the whole point of the pool. The pages already hold the keys and
    /// values for these token positions — computed by whichever request got
    /// here first — so this request takes a reference to them and starts its
    /// forward pass after them. Nothing is copied: the two sequences read the
    /// same bytes.
    ///
    /// Returns how many rows were adopted, or `None` if any tag is not resident
    /// — all or nothing, because page `i` is meaningless without `0..i`.
    fn adopt_pages(&mut self, tags: &[u64]) -> Option<usize> {
        let paged = self.paged.as_mut()?;
        debug_assert!(
            paged.pages.is_empty() && self.k.is_empty(),
            "pages are adopted into a fresh layer, before anything is pushed"
        );
        // Taken once for the sequence, by layer 0; later layers read the list
        // it recorded. Acquiring per layer would take one reference per layer
        // for a page that is one page.
        if paged.layer == 0 {
            let got = paged.seq.pool.acquire(tags).ok()?;
            if !got.iter().all(|a| a.hit) {
                // Something was reclaimed between the index promising it and
                // this call. Give back what was taken rather than filling the
                // gaps: a half-adopted prefix is not a prefix.
                let pages: Vec<u32> = got.iter().map(|a| a.page).collect();
                paged.seq.pool.release(&pages);
                return None;
            }
            paged
                .seq
                .adopt(&got.iter().map(|a| a.page).collect::<Vec<_>>());
        }
        paged.pages = paged.seq.pages();
        if paged.pages.len() != tags.len() {
            return None;
        }
        // Adopted pages are finished by construction — they were sealed by
        // whoever built them — so host reads must take their rows from the
        // pool, not from an empty tail.
        paged.full_pages = paged.pages.len();
        let rows = paged.pages.len() * paged.rows_per_page;
        self.len = rows;
        Some(rows)
    }

    /// Rolls a paged layer back to `new_len` rows.
    ///
    /// Whole pages past the cut go back to the pool. The interesting case is a
    /// cut that lands *inside* a sealed page: that page cannot simply be kept
    /// and appended to, because sealing is what made it shareable and another
    /// sequence may be reading it right now. So its retained rows are copied
    /// into the tail buffer and the page is released — after which writing
    /// continues into the tail exactly as it would have.
    ///
    /// That copy is bounded by one page and happens once per rollback, against
    /// a rollback that discards at least one token. Speculative decoding is
    /// what does this, and it rolls back a draft's rejected tail, not a
    /// conversation.
    fn truncate_paged(&mut self, new_len: usize) {
        let kv_dim = self.kv_dim;
        let Some(paged) = self.paged.as_mut() else {
            return;
        };
        let rows_per_page = paged.rows_per_page;
        let whole = new_len / rows_per_page;
        let partial = new_len % rows_per_page;

        if whole >= paged.full_pages {
            // The cut lands inside the page currently being written, whose rows
            // live in the tail buffer and not in the pool. Nothing to release
            // and nothing to carry back — drop the tail's later rows and stop.
            self.k.truncate(partial * kv_dim);
            self.v.truncate(partial * kv_dim);
            // Rows past the cut are gone; whatever the device still holds for
            // them will be overwritten before it is read again, but the
            // watermark must not claim they are current.
            paged.tail_uploaded = paged.tail_uploaded.min(partial);
            return;
        }

        // The cut lands in a completed page. Everything from there on leaves
        // this sequence — including any page allocated for the tail, which is
        // why this drains to `whole` and not to `full_pages`.
        let released: Vec<u32> = paged.pages.drain(whole..).collect();
        if partial > 0 {
            // A completed page cannot be appended to: sealing is what made it
            // shareable and another sequence may be reading it. So its retained
            // rows are copied into the tail and the page is let go.
            let straddling = released[0];
            let mut k = Vec::with_capacity(partial * kv_dim);
            let mut v = Vec::with_capacity(partial * kv_dim);
            for row in 0..partial {
                let range = paged.row(straddling, row);
                k.extend_from_slice(&paged.seq.pool.page_k_all(paged.layer)[range.clone()]);
                v.extend_from_slice(&paged.seq.pool.page_v_all(paged.layer)[range]);
            }
            self.k = k;
            self.v = v;
        } else {
            self.k.clear();
            self.v.clear();
        }
        paged.full_pages = whole;
        paged.tail_uploaded = 0;
        // The device copy cannot be ahead of the pages that still exist.
        paged.device_synced = paged.device_synced.min(whole);
        // The release is the sequence's, not this layer's: every layer rolls
        // back to the same token count and would otherwise release the same
        // pages once each. `truncate_to` is idempotent for exactly that reason.
        paged.seq.truncate_to(whole);
    }

    /// How many rows are actually **in the host buffers**.
    ///
    /// Not the same as [`len`](Self::len), and the gap is the whole reason
    /// this exists. The fused GPU decode path writes a token's key and value
    /// straight into the device mirror and calls
    /// [`advance_gpu_only`](Self::advance_gpu_only), which moves `len` without
    /// pushing anything here — so after N decode steps `len` is N rows ahead
    /// of `k`/`v`. Anything that reads the host side (prefix reuse, slot save,
    /// CPU attention) has to bound itself by *this*, not by `len`.
    ///
    /// Derived from the buffer rather than tracked in a field on purpose: a
    /// second counter is a second thing to keep in step, and this one cannot
    /// drift from what it describes.
    fn host_len(&self) -> usize {
        if self.kv_dim == 0 {
            return 0;
        }
        // Sealed pages are as host-resident as the tail is; the split between
        // them is where the bytes live, not whether they exist.
        self.sealed_rows() + self.tail_rows()
    }

    /// [`host_len`](Self::host_len) in tokens rather than rows.
    fn host_tokens(&self) -> usize {
        self.host_len() * self.stride
    }

    /// The key vector at cached position `pos` for KV head `kv_head`
    /// (`[head_dim]`).
    pub fn key_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let row = self.row_k(pos);
        &row[kv_head * head_dim..(kv_head + 1) * head_dim]
    }

    pub fn value_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let row = self.row_v(pos);
        &row[kv_head * head_dim..(kv_head + 1) * head_dim]
    }

    /// Row `pos`'s keys, from whichever side of the seal it is on.
    ///
    /// The one place the paged indirection lives on the read path. A page's
    /// rows are contiguous, so a row is still a plain slice either way and the
    /// attention loops above are unchanged — this is a lookup and an offset,
    /// not a gather.
    fn row_k(&self, pos: usize) -> &[f32] {
        match self.paged.as_ref() {
            None => &self.k[pos * self.kv_dim..(pos + 1) * self.kv_dim],
            Some(paged) => {
                let complete = paged.full_pages * paged.rows_per_page;
                if pos < complete {
                    let (page, row) = (pos / paged.rows_per_page, pos % paged.rows_per_page);
                    let range = paged.row(paged.pages[page], row);
                    &paged.seq.pool.page_k_all(paged.layer)[range]
                } else {
                    let local = pos - complete;
                    &self.k[local * self.kv_dim..(local + 1) * self.kv_dim]
                }
            }
        }
    }

    fn row_v(&self, pos: usize) -> &[f32] {
        match self.paged.as_ref() {
            None => &self.v[pos * self.kv_dim..(pos + 1) * self.kv_dim],
            Some(paged) => {
                let complete = paged.full_pages * paged.rows_per_page;
                if pos < complete {
                    let (page, row) = (pos / paged.rows_per_page, pos % paged.rows_per_page);
                    let range = paged.row(paged.pages[page], row);
                    &paged.seq.pool.page_v_all(paged.layer)[range]
                } else {
                    let local = pos - complete;
                    &self.v[local * self.kv_dim..(local + 1) * self.kv_dim]
                }
            }
        }
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
        // Against the *host* rows, not `src.len`: a cache whose generated
        // tail was written by the fused GPU path has a `len` that runs past
        // its host buffers, and copying against `len` reads off the end of
        // them. Callers clamp before getting here
        // (`KvCache::host_committed_len`); this is the backstop that names
        // the real bound if one ever does not.
        assert!(
            len <= src.host_len(),
            "reused prefix ({len}) exceeds the source cache's host-resident length ({}); \
             its `len` is {} because the fused decode path wrote those rows to the device only",
            src.host_len(),
            src.len
        );
        assert!(
            self.paged.is_none(),
            "a paged layer is filled by pushing rows into it, not by copying a \
             flat prefix over its buffers; the reuse path for paged caches is \
             sharing the source's pages, not duplicating its bytes"
        );
        self.k.clear();
        self.v.clear();
        let (k, v) = src.flatten(len);
        self.k = k;
        self.v = v;
        self.len = len;
        self.gpu = None;
        self.pool_backed = false;
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
    /// The most recent token *ids* committed to this cache, oldest first —
    /// the short lookback an architecture needs when part of its forward
    /// pass is a function of the token ids themselves rather than of any
    /// hidden state.
    ///
    /// Only `engine::arch::qwen4exp` writes or reads it: its per-layer
    /// embedding hashes the current token together with its
    /// `ple.ngram_size` minus one predecessors, and during decode — or at a
    /// chunked prefill's seam — those predecessors are not in the batch.
    /// Nothing else in a `KvCache` records them: an attention layer holds
    /// projected keys, and a recurrent layer holds one evolving state.
    /// Empty for every other architecture, and bounded to the few ids the
    /// hash actually reaches back for, so this is not a second copy of the
    /// conversation.
    ///
    /// It travels with the cache through every path a recurrent
    /// architecture can take — carryover, adoption, duplication, and
    /// slot persistence — so a restored or reused cache hashes the same
    /// n-grams a freshly prefilled one would.
    pub recent_tokens: Vec<u32>,
}

impl KvCache {
    /// The one constructor. Every other is a thin wrapper that fills in the
    /// parts its callers do not vary.
    ///
    /// Written this way because it was not: there were four independent
    /// constructors, and between them they could express per-layer dims,
    /// per-layer strides, and recurrent state — but *not* strides and
    /// recurrent state together, because the two grew on separate branches
    /// and nothing joined them. No architecture needs that combination today.
    /// The next one to need it should find a constructor rather than a fifth
    /// entry point, and every shape should already flow through the same code
    /// so a change to how rows are allocated lands in one place.
    fn build(capacity: usize, kv_dims: &[(usize, usize)], recurrent: &[RecurrentSpec]) -> Self {
        Self {
            layers: kv_dims
                .iter()
                .map(|&(dim, stride)| LayerCache::new_strided(capacity, dim, stride))
                .collect(),
            recurrent: recurrent
                .iter()
                .copied()
                .map(RecurrentLayerState::new)
                .collect(),
            recent_tokens: Vec::new(),
        }
    }

    /// A uniform cache: `n_layer` per-token slots of the same width.
    pub fn new(n_layer: usize, capacity: usize, kv_dim: usize) -> Self {
        Self::new_with_dims(capacity, &vec![kv_dim; n_layer])
    }

    /// Like [`KvCache::new`], but each layer gets its own `kv_dim` — for
    /// architectures where key/value head size varies by layer (e.g.
    /// Gemma's SWA vs. full-attention layers using different head dims).
    pub fn new_with_dims(capacity: usize, kv_dims: &[usize]) -> Self {
        let strided: Vec<(usize, usize)> = kv_dims.iter().map(|&dim| (dim, 1)).collect();
        Self::build(capacity, &strided, &[])
    }

    /// Like [`KvCache::new_with_dims`], but each slot also carries a
    /// *stride*: how many token positions one of its rows stands for (see
    /// [`LayerCache::stride`]). `capacity` stays a token count for every
    /// slot. `engine::arch::deepseek4` uses this to keep its
    /// block-compressed attention state (one row per 4- or 128-token block)
    /// in the same positional cache as its per-token keys, so rollback,
    /// prefix reuse, and slot persistence apply to all of it unchanged.
    pub fn new_with_strided_dims(capacity: usize, kv_dims: &[(usize, usize)]) -> Self {
        Self::build(capacity, kv_dims, &[])
    }

    /// A cache whose layers draw their rows from `pool`.
    ///
    /// The pool's geometry must be this model's: it is built from a probe cache
    /// of the same architecture (`kv_pool::LayerGeometry::of`), so a mismatch
    /// means a pool built for one model is being handed to another — the same
    /// class of error the slot fingerprint refuses, and it is checked here for
    /// the same reason. A cache that silently ran on another model's geometry
    /// would not crash; it would answer from the wrong rows.
    /// Converts this cache's positional layers to draw their rows from `pool`,
    /// keeping everything else about it.
    ///
    /// **Built from the architecture's own cache, not from the pool's
    /// geometry.** A `KvCache` is not only its attention layers: the mixed
    /// attention/linear-attention architectures — `qwen35moe`, `qwen3next`,
    /// `nemotron_h_moe`, `kda`, `inkling` and the rest — carry per-layer
    /// recurrent state alongside, and index it directly. A paged cache
    /// assembled from the pool's layer list alone would have none of it, and
    /// the first recurrent layer of such a model would index an empty vector.
    ///
    /// Recurrent state is not paged and could not be: it is one evolving value
    /// per layer with no per-position history, which is why
    /// `prefix_cache::CachedPrefill::reusable_prefix_len` already forces
    /// all-or-nothing reuse on it. So it is carried across untouched, and only
    /// the positional layers change where their rows live.
    pub fn into_paged(mut self, pool: std::sync::Arc<crate::engine::kv_pool::KvPool>) -> Self {
        assert_eq!(
            self.layers.len(),
            pool.layers().len(),
            "the pool was built for a different model's layer count"
        );
        let max_pages = pool.pages_for(self.layers.first().map_or(0, |l| l.capacity * l.stride));
        let seq = std::sync::Arc::new(SequencePages::new(pool.clone(), max_pages));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let geom = pool.layers()[i];
            assert_eq!(
                (layer.kv_dim, layer.stride),
                (geom.kv_dim, geom.stride),
                "layer {i}'s geometry does not match the pool's"
            );
            layer.k.clear();
            layer.v.clear();
            layer.len = 0;
            layer.gpu = None;
            layer.paged = Some(PagedRows {
                seq: seq.clone(),
                layer: i,
                rows_per_page: pool.page_tokens().div_ceil(geom.stride.max(1)),
                pages: Vec::new(),
                full_pages: 0,
                tail_uploaded: 0,
                device_synced: 0,
            });
        }
        self
    }

    /// Publishes every page whose positions are complete, making them
    /// shareable.
    ///
    /// Called by the forward pass at a point where it knows a span of
    /// positions has been through every layer — after a prefill chunk, after a
    /// decode step. Nothing else can know that: a page is written layer by
    /// layer, and a cache watching its own fill counts cannot distinguish "all
    /// layers are done" from "the layers that happen to have arrived so far".
    pub fn commit_pages(&self) {
        if let Some(paged) = self.layers.first().and_then(|l| l.paged.as_ref()) {
            paged.seq.seal_complete();
        }
    }

    /// Gives every layer the page identities this sequence's tokens imply, so
    /// the pages it seals become findable by the next request that shares them.
    ///
    /// Set before the forward pass, from `engine::prefix_index::page_tags` over
    /// the prompt. A cache without them still pages — it just keeps everything
    /// to itself.
    pub fn set_page_tags(&mut self, tags: &[u64]) {
        if let Some(paged) = self.layers.first().and_then(|l| l.paged.as_ref()) {
            paged.seq.set_tags(tags);
        }
    }

    /// Adopts the pages behind `tags` into every layer, sharing them rather
    /// than recomputing them.
    ///
    /// Returns the number of **token** positions adopted, or `None` if the
    /// pages were not all resident — in which case nothing is taken and the
    /// caller prefills as it would have.
    ///
    /// All layers or none. A cache whose layer 3 adopted a prefix and whose
    /// layer 4 did not is not in a state any forward pass can describe, so the
    /// failure is handled here rather than left to be discovered mid-pass.
    pub fn adopt_shared_pages(&mut self, tags: &[u64], page_tokens: usize) -> Option<usize> {
        if tags.is_empty() {
            return Some(0);
        }
        let mut adopted = None;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer.adopt_pages(tags) {
                Some(rows) => {
                    debug_assert!(
                        adopted.is_none_or(|r| r == rows) || layer.stride != 1,
                        "layers disagreed on how many rows one prefix is"
                    );
                    adopted = Some(rows);
                }
                None if i == 0 => return None,
                None => unreachable!(
                    "layer {i} refused a prefix layer 0 accepted; every layer \
                     draws on the same pool and the same page list"
                ),
            }
        }
        Some(tags.len() * page_tokens)
    }

    /// Device bytes this cache's *shape* would need at `token_capacity`
    /// tokens — the GPU mirror only, not the host buffers.
    ///
    /// Takes the capacity as an argument rather than reading `self`'s so a
    /// caller can size a context it has not allocated. That is the whole
    /// point: `ModelForward::new_kv_cache(1)` costs nothing and answers the
    /// shape question (how many layers, each layer's `kv_dim` and stride,
    /// which is all fixed model geometry), and this then scales it to the
    /// context an operator is actually asking about — which may be tens of
    /// gigabytes and must not be allocated to be counted.
    ///
    /// Recurrent state is excluded: it has no GPU mirror (`sync_gpu` is a
    /// method on the positional layers alone), so counting it here would
    /// charge a linear-attention model for device memory it never takes.
    pub fn gpu_mirror_bytes(
        &self,
        token_capacity: usize,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
    ) -> u64 {
        self.gpu_mirror_bytes_where(token_capacity, n_head, kv_storage, |_| true)
    }

    /// The same, over the layers `keep` accepts by index — what one device of
    /// a split model holds.
    ///
    /// A predicate over layer indices rather than a count, because a device's
    /// share of the KV cache is **not** its share of the layers: `kv_dim` and
    /// `stride` vary along a model's own depth (a per-layer `head_count_kv`,
    /// a block-compressed layer, a linear-attention layer with no positional
    /// state at all), so scaling the total by `layers_on_device / n_layer`
    /// would be wrong by whatever that variation is — and wrong in the
    /// direction of the architectures where the answer matters most. The
    /// indices line up with the placement plan's `layer_device`: both are
    /// indexed by transformer block.
    pub fn gpu_mirror_bytes_where(
        &self,
        token_capacity: usize,
        n_head: usize,
        kv_storage: crate::engine::backend::vulkan_shaders::KvStorage,
        keep: impl Fn(usize) -> bool,
    ) -> u64 {
        self.layers
            .iter()
            .enumerate()
            .filter(|(index, layer)| layer.kv_dim > 0 && keep(*index))
            .map(|(_, layer)| {
                gpu_layer_bytes(
                    token_capacity.div_ceil(layer.stride),
                    layer.kv_dim,
                    n_head,
                    kv_storage,
                    ASSUMED_STORAGE_ALIGN,
                )
            })
            .sum()
    }

    /// Like [`KvCache::new_with_dims`], plus a recurrent state per entry in
    /// `recurrent_specs`, for a mixed attention/linear-attention
    /// architecture.
    pub fn new_mixed(
        capacity: usize,
        kv_dims: &[usize],
        recurrent_specs: &[RecurrentSpec],
    ) -> Self {
        let strided: Vec<(usize, usize)> = kv_dims.iter().map(|&dim| (dim, 1)).collect();
        Self::build(capacity, &strided, recurrent_specs)
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
        self.recent_tokens.clone_from(&src.recent_tokens);
    }

    /// Takes `src`'s first `len` token positions as this cache's own, moving
    /// the buffers instead of copying them.
    ///
    /// The same result as [`Self::copy_prefix_from`] and the same
    /// preconditions — `self` freshly allocated, `len` already bounded by
    /// `CachedPrefill::reusable_prefix_len` — but it consumes `src`, which is
    /// what makes the move sound. Only a caller that owns the source outright
    /// may use it: `engine::prefix_cache::PrefixCache::take_best_match`
    /// *removes* the entry it hands back, so the cross-request pool qualifies,
    /// while `engine::slot_store` keeps its snapshot for the same slot's next
    /// request and must go on copying.
    ///
    /// Worth the second method because the copy is not small. Measured on a
    /// 2001-token conversational prefix (16 layers, `kv_dim` 512): **67.7 ms**
    /// the first time and **18.1 ms** warm, against a total reuse-path prefill
    /// of 167 ms — and the copy transiently doubles the prefix's resident
    /// footprint, on the machine least able to spare it, at the exact moment
    /// the source is about to be dropped.
    pub fn adopt_prefix(&mut self, mut src: KvCache, len: usize) {
        // Trim the source to what is actually being reused before taking its
        // buffers — the caller's `len` can be shorter than what the source
        // holds, because the reuse path always leaves at least one prompt
        // token for the forward pass to have something to do.
        //
        // Skipped when there is nothing to trim, which is also the only case a
        // recurrent cache can be in: its state has no per-position history to
        // roll back, so `reusable_prefix_len` forces all-or-nothing there and
        // `truncate` refuses recurrent caches outright.
        if src.committed_len() > len {
            src.truncate(len);
        }
        for (dst, s) in self.layers.iter_mut().zip(src.layers.iter_mut()) {
            dst.adopt(s);
        }
        for (dst, s) in self.recurrent.iter_mut().zip(src.recurrent.iter()) {
            dst.copy_from(s);
        }
        self.recent_tokens = std::mem::take(&mut src.recent_tokens);
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
    /// How much of this cache another request may actually reuse.
    ///
    /// [`committed_len`](Self::committed_len) counts every position the cache
    /// *logically* holds, including rows the fused GPU decode path wrote only
    /// to the device mirror. Those rows are real for continuing this request
    /// and unreadable to anyone else, so reuse is bounded by the host side —
    /// and by the **shortest** layer, not the longest, because a prefix is
    /// only reusable to the depth every layer can supply it.
    pub fn host_committed_len(&self) -> usize {
        self.layers
            .iter()
            // Layers that hold nothing are skipped rather than counted as
            // zero: `engine::arch::gemma`'s cross-layer KV donor is
            // permanently empty by design, and a plain minimum would read it
            // as "no prefix is reusable" and switch reuse off for the whole
            // architecture. `LayerCache::copy_prefix_from` returns early on an
            // empty source, so a donor is safe to leave out of the bound.
            .filter(|layer| layer.len > 0)
            .map(LayerCache::host_tokens)
            .min()
            .unwrap_or(0)
    }

    pub fn committed_len(&self) -> usize {
        // `committed_tokens`, not `len`: a block-compressed slot's `len` is a
        // *row* count. Taking the raw maximum happens to give the right answer
        // on every architecture built so far, because each of them also has
        // ordinary per-token slots and those are always the longest — but that
        // is a coincidence of the current models rather than a property
        // anything guarantees, and it is the kind of coincidence a change to
        // how the cache is allocated would quietly break.
        self.layers
            .iter()
            .map(LayerCache::committed_tokens)
            .max()
            .unwrap_or(0)
    }

    /// Records `token` as the newest entry of [`Self::recent_tokens`],
    /// keeping at most `keep` of them.
    ///
    /// `keep` is the caller's own lookback (`ple.ngram_size - 1` for
    /// `engine::arch::qwen4exp`), not a cache-wide constant: the cache has
    /// no opinion on how far back a model reaches, only on carrying what it
    /// is told to.
    pub fn push_recent_token(&mut self, token: u32, keep: usize) {
        if keep == 0 {
            return;
        }
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > keep {
            let drop = self.recent_tokens.len() - keep;
            self.recent_tokens.drain(..drop);
        }
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
            recent_tokens: self.recent_tokens.clone(),
        }
    }

    /// A structural signature — layer count and each layer's `kv_dim`, plus
    /// every recurrent layer's [`RecurrentSpec`] — with no per-position data
    /// or capacity in it. Two caches from the
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
            // Part of the shape, and it was missing: two caches differing only
            // in stride are laid out differently and mean different things by
            // a row, so a signature that omitted it did not describe the
            // structure it claims to. Masked until now by the model label the
            // caller hashes alongside this, which is not the same as correct.
            push_u32(&mut out, l.stride as u32);
        }
        push_u32(&mut out, self.recurrent.len() as u32);
        for r in &self.recurrent {
            push_u32(&mut out, r.conv_channels as u32);
            push_u32(&mut out, r.d_conv as u32);
            push_u32(&mut out, r.num_heads() as u32);
            push_u32(&mut out, r.head_dim as u32);
            push_u32(&mut out, r.state_dim as u32);
        }
        out
    }

    /// Serializes every **host-resident** KV position (and all recurrent
    /// state) to a self-describing little-endian byte blob — the payload
    /// [`crate::engine::slot_store`] writes under
    /// `~/.orangu/server/<fp>/slots/`. Only the committed floats of each
    /// layer are written (never the unused tail of a larger `capacity`), so
    /// a saved file is sized to the conversation, not the context window.
    ///
    /// Host-resident, not `len`: the fused decode path commits a generated
    /// token's key and value to the device mirror alone
    /// ([`LayerCache::advance_gpu_only`]), which leaves `len` counting rows
    /// that `k`/`v` do not hold. Writing `len` rows sliced off the end of
    /// them and took the whole request down with
    /// `range end index … out of range for slice of length …` — the same
    /// crash [`Self::host_committed_len`] exists to prevent on the reuse
    /// path, reached here through `slot_store::save` instead.
    ///
    /// What a save therefore leaves behind is the *generated tail*, which the
    /// next turn re-prefills; the prompt prefix — the large part, and the
    /// part worth persisting — is host-resident and is written in full. That
    /// is the same trade [`crate::engine::prefix_cache::CachedPrefill::
    /// reusable_prefix_len`] already makes for in-memory reuse, so a restored
    /// slot and a pooled one now agree on what a cache is worth.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(KV_CACHE_MAGIC);
        push_u32(&mut out, self.layers.len() as u32);
        for l in &self.layers {
            let rows = l.len.min(l.host_len());
            push_u32(&mut out, l.kv_dim as u32);
            push_u32(&mut out, rows as u32);
            // Through `flatten`, because a paged layer's `k` is only its tail
            // and a snapshot written from that would restore a fraction of the
            // conversation while claiming the whole of it.
            let (k, v) = l.flatten(rows);
            push_f32s(&mut out, &k);
            push_f32s(&mut out, &v);
        }
        push_u32(&mut out, self.recurrent.len() as u32);
        for r in &self.recurrent {
            push_u32(&mut out, r.conv_channels as u32);
            push_u32(&mut out, r.d_conv as u32);
            push_u32(&mut out, r.num_heads() as u32);
            push_u32(&mut out, r.head_dim as u32);
            push_u32(&mut out, r.state_dim as u32);
            push_f32s(&mut out, &r.conv_history);
            push_f32s(&mut out, &r.delta_state);
        }
        // A trailing section rather than a new format version, so a blob
        // written before this field existed still restores: see
        // [`Self::from_bytes`] for why an absent section is never the wrong
        // answer.
        if !self.recent_tokens.is_empty() {
            push_u32(&mut out, self.recent_tokens.len() as u32);
            for &tok in &self.recent_tokens {
                push_u32(&mut out, tok);
            }
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
            let state_dim = cur.u32()? as usize;
            let conv_history = cur.f32s(conv_channels * d_conv.saturating_sub(1))?;
            let delta_state = cur.f32s(num_heads * head_dim * state_dim)?;
            recurrent.push(RecurrentLayerState::from_parts(
                conv_channels,
                d_conv,
                head_dim,
                state_dim,
                conv_history,
                delta_state,
            ));
        }
        // Optional, and absent from every blob written before
        // [`KvCache::recent_tokens`] existed. Reading nothing there is safe
        // rather than merely tolerable: the only architecture that reads the
        // field is `engine::arch::qwen4exp`, which no earlier build could
        // serve, so no old slot can be one whose lookback this would be
        // silently dropping. The strict "nothing left over" check below is
        // kept, so a corrupt blob is still rejected.
        let recent_tokens = if cur.is_empty() {
            Vec::new()
        } else {
            let n = cur.u32()? as usize;
            let mut toks = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                toks.push(cur.u32()?);
            }
            toks
        };
        if !cur.is_empty() {
            anyhow::bail!("trailing bytes after KV-cache blob");
        }
        Ok(Self {
            layers,
            recurrent,
            recent_tokens,
        })
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
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// The shape of one recurrent layer's state — what [`KvCache::new_mixed`]
/// needs to allocate it.
///
/// The two constructors name the two shapes that occur, which differ only in
/// whether each head's state matrix is square. Nothing else about the two
/// paths differs here: both carry a causal-conv1d history alongside it, and
/// both evolve a single state forward rather than keeping per-position
/// history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentSpec {
    pub conv_channels: usize,
    pub d_conv: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub state_dim: usize,
}

impl RecurrentSpec {
    /// A gated-delta-net layer: keys and values are the same width, so each
    /// head's state is the square `[head_dim, head_dim]` outer-product
    /// accumulator (`engine::arch::qwen35moe`, `qwen35`, `qwen3next`,
    /// `kimi3`). Also the shape a conv-only layer asks for with
    /// `num_heads == 0` (`engine::arch::inkling`'s short convolutions),
    /// which allocates no state matrix at all.
    pub fn delta_net(
        conv_channels: usize,
        d_conv: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            conv_channels,
            d_conv,
            num_heads,
            head_dim,
            state_dim: head_dim,
        }
    }

    /// A selective state-space layer: the state axis is the model's own
    /// `ssm.state_size`, which is independent of the head width, so each
    /// head's state is the rectangular `[head_dim, state_dim]`
    /// (`engine::arch::nemotron`).
    pub fn ssm(
        conv_channels: usize,
        d_conv: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> Self {
        Self {
            conv_channels,
            d_conv,
            num_heads,
            head_dim,
            state_dim,
        }
    }
}

/// One recurrent (SSM / gated-delta-net) layer's persistent state: a
/// causal-conv1d rolling history and a per-head state matrix. Unlike
/// [`LayerCache`], there's no per-position history to index — linear
/// attention/SSM layers carry a single evolving state forward.
pub struct RecurrentLayerState {
    /// `[conv_channels, d_conv - 1]`, channel-major, oldest-first per
    /// channel — the causal conv1d's rolling window of prior inputs.
    conv_history: Vec<f32>,
    conv_channels: usize,
    d_conv: usize,
    /// Per-head state matrices, flattened `[num_heads, head_dim, state_dim]`
    /// (`state[head][i][j]`), with `state_dim` fastest-varying.
    delta_state: Vec<f32>,
    head_dim: usize,
    /// The second axis of each head's state matrix — equal to `head_dim` for
    /// a delta-net layer, `ssm.state_size` for a selective-SSM one. See
    /// [`RecurrentSpec`].
    state_dim: usize,
}

impl RecurrentLayerState {
    fn new(spec: RecurrentSpec) -> Self {
        Self {
            conv_history: vec![0.0; spec.conv_channels * spec.d_conv.saturating_sub(1)],
            conv_channels: spec.conv_channels,
            d_conv: spec.d_conv,
            delta_state: vec![0.0; spec.num_heads * spec.head_dim * spec.state_dim],
            head_dim: spec.head_dim,
            state_dim: spec.state_dim,
        }
    }

    /// Rebuilds a recurrent state from a slot-persistence snapshot. `num_heads`
    /// is recovered from `delta_state`'s length rather than stored separately.
    fn from_parts(
        conv_channels: usize,
        d_conv: usize,
        head_dim: usize,
        state_dim: usize,
        conv_history: Vec<f32>,
        delta_state: Vec<f32>,
    ) -> Self {
        Self {
            conv_history,
            conv_channels,
            d_conv,
            delta_state,
            head_dim,
            state_dim,
        }
    }

    /// How many heads this state carries — `delta_state` is a dense
    /// `[num_heads, head_dim, state_dim]`, so the head count is implied by
    /// its length. `0` for the degenerate `head_dim == 0` case (a conv-only
    /// layer, which carries no state matrix).
    fn num_heads(&self) -> usize {
        self.delta_state
            .len()
            .checked_div(self.head_dim * self.state_dim)
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
            state_dim: self.state_dim,
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
        debug_assert_eq!(self.state_dim, src.state_dim);
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

    /// The state matrix for `head` (`[head_dim, state_dim]`, `state_dim`
    /// fastest-varying, mutable — the recurrence updates it in place every
    /// token).
    pub fn delta_state_mut(&mut self, head: usize) -> &mut [f32] {
        let size = self.head_dim * self.state_dim;
        let start = head * size;
        &mut self.delta_state[start..start + size]
    }
}

/// The `(kv_dim, stride)` list a pool describes — for tests that build a cache
/// matching a pool without a model to ask.
#[cfg(test)]
pub(crate) fn strided_dims(
    pool: &std::sync::Arc<crate::engine::kv_pool::KvPool>,
) -> Vec<(usize, usize)> {
    pool.layers().iter().map(|g| (g.kv_dim, g.stride)).collect()
}

#[cfg(test)]
mod tests {
    /// The run split is the whole of the paged prefill write: a range that
    /// crosses a page boundary is contiguous on the host and in each page, but
    /// not across them, so getting the boundary wrong writes a chunk of a
    /// prefill into the middle of somebody else's page.
    ///
    /// The table here is deliberately *not* ascending, because an ascending one
    /// makes `dst_row` land where a plain `write_pos + i` would and the test
    /// stops being able to fail.
    #[test]
    fn page_runs_split_at_page_boundaries() {
        let pages = [7u32, 3, 9, 1];
        // Wholly inside one page.
        assert_eq!(
            super::page_runs(&pages, 1, 2, 4),
            vec![super::PageRun {
                src_row: 0,
                dst_row: 7 * 4 + 1,
                rows: 2
            }]
        );
        // Exactly one whole page.
        assert_eq!(
            super::page_runs(&pages, 4, 4, 4),
            vec![super::PageRun {
                src_row: 0,
                dst_row: 3 * 4,
                rows: 4
            }]
        );
        // Straddling a boundary: the tail of page 0, then the head of page 1.
        assert_eq!(
            super::page_runs(&pages, 2, 4, 4),
            vec![
                super::PageRun {
                    src_row: 0,
                    dst_row: 7 * 4 + 2,
                    rows: 2
                },
                super::PageRun {
                    src_row: 2,
                    dst_row: 3 * 4,
                    rows: 2
                },
            ]
        );
        // Spanning three pages, starting and ending part way in.
        assert_eq!(
            super::page_runs(&pages, 3, 8, 4),
            vec![
                super::PageRun {
                    src_row: 0,
                    dst_row: 7 * 4 + 3,
                    rows: 1
                },
                super::PageRun {
                    src_row: 1,
                    dst_row: 3 * 4,
                    rows: 4
                },
                super::PageRun {
                    src_row: 5,
                    dst_row: 9 * 4,
                    rows: 3
                },
            ]
        );
        // Every run together covers the range exactly once, in order.
        let runs = super::page_runs(&pages, 3, 8, 4);
        let mut next = 0;
        for r in &runs {
            assert_eq!(r.src_row, next, "runs must tile the source without gaps");
            next += r.rows;
        }
        assert_eq!(next, 8);
    }

    /// The block-at-a-time rewrite of [`f32_to_q8_0_bytes`] must produce the
    /// same bytes as the `push`-per-element form it replaced — this is a KV
    /// cache the GPU then reads, so a one-byte difference is a silently
    /// wrong attention output rather than a crash.
    ///
    /// The reference is written out here rather than kept in the module,
    /// because the whole point of the change was to delete the `push` loop.
    #[test]
    fn q8_0_kv_bytes_match_the_push_per_element_form() {
        fn reference(data: &[f32]) -> Vec<u8> {
            let mut out = Vec::new();
            for block in data.as_chunks::<32>().0 {
                let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
                let d = amax / 127.0;
                let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&d.to_le_bytes());
                for &v in block {
                    out.push(((v * inv_d).round().clamp(-127.0, 127.0) as i8) as u8);
                }
            }
            out
        }

        let mut data: Vec<f32> = Vec::new();
        // An all-zero block, where `d` is zero and `inv_d` must not be a
        // division by it.
        data.extend(std::iter::repeat_n(0.0f32, 32));
        // A block whose peak magnitude is negative and unique, so dropping
        // the `abs` from the max fold would change `d`.
        let mut neg = [0.25f32; 32];
        neg[11] = -90.0;
        data.extend_from_slice(&neg);
        // Ties: `amax` 254 makes `inv_d` exactly 0.5, so 1.0 and 3.0 land on
        // 0.5 and 1.5 — the two values that separate ties-away-from-zero
        // from ties-to-even.
        let mut ties = [254.0f32; 32];
        ties[1] = 1.0;
        ties[2] = -1.0;
        ties[3] = 3.0;
        ties[4] = -3.0;
        data.extend_from_slice(&ties);
        // A plain spread.
        let mut lcg = 0x2545_F491u32;
        data.extend((0..64).map(|_| {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (lcg >> 8) as f32 / 8_388_608.0 - 0.5
        }));

        assert_eq!(super::f32_to_q8_0_bytes(&data), reference(&data));
    }
    use super::*;
    use crate::engine::backend::vulkan_shaders::KvStorage;

    /// A cache in the state the fused GPU decode path leaves it: a prefill
    /// pushed to the host, then decode steps committed to the device mirror
    /// only.
    fn cache_with_gpu_only_tail(prefilled: usize, decoded: usize) -> KvCache {
        let mut cache = KvCache::new(1, 512, 4);
        for i in 0..prefilled {
            cache.layers[0].push(&[i as f32; 4], &[i as f32; 4]);
        }
        for _ in 0..decoded {
            cache.layers[0].advance_gpu_only();
        }
        cache
    }

    /// **The crash this pair of methods exists to prevent.**
    ///
    /// `len` counts every position the cache logically holds, including the
    /// ones the fused decode path wrote straight to the device. The host
    /// buffers hold only the prefilled part, so anything that reuses this
    /// cache has to bound itself by `host_committed_len` — bounding by
    /// `committed_len` reads off the end of `k`/`v`.
    ///
    /// Found in the wild, not in review: a second conversational turn against
    /// a slot's retained cache panicked with `range end index 31488 out of
    /// range for slice of length 30592` — 246 rows claimed against 239 held,
    /// the difference being exactly the tokens the first turn generated.
    #[test]
    fn a_gpu_written_tail_counts_toward_len_but_not_toward_what_can_be_reused() {
        let cache = cache_with_gpu_only_tail(239, 7);
        assert_eq!(cache.committed_len(), 246, "the cache holds 246 positions");
        assert_eq!(
            cache.host_committed_len(),
            239,
            "but only 239 of them can be read back on the host"
        );
    }

    /// **The crash again, one path over.** Saving a slot serialized `len`
    /// rows out of buffers that hold fewer, so a request that had generated
    /// anything on the fused GPU path took the server down on
    /// `POST /slots/{id}?action=save`: `range end index 1653504 out of range
    /// for slice of length 1038848`.
    #[test]
    fn saving_a_cache_with_a_gpu_written_tail_writes_only_the_host_rows() {
        let cache = cache_with_gpu_only_tail(239, 7);
        let bytes = cache.to_bytes();
        let restored = KvCache::from_bytes(&bytes).expect("round trip");
        assert_eq!(
            restored.committed_len(),
            239,
            "only the host-resident rows are persisted"
        );
        assert_eq!(restored.layers[0].k.len(), 239 * 4);
        assert_eq!(restored.layers[0].v.len(), 239 * 4);
        for i in 0..239 {
            assert_eq!(restored.layers[0].key_at(i, 0, 4), [i as f32; 4]);
        }
    }

    /// The n-gram lookback keeps only the last `keep` ids, and keeps them
    /// oldest-first — the order `arch::qwen4exp`'s hash indexes them in.
    /// Reversing it, or keeping the *first* `keep`, hashes real predecessors
    /// in the wrong slots: a plausible embedding for an n-gram that never
    /// occurred, which nothing downstream can see.
    #[test]
    fn the_recent_token_lookback_keeps_the_newest_ids_oldest_first() {
        let mut cache = KvCache::new(1, 8, 4);
        for tok in [10u32, 11, 12, 13] {
            cache.push_recent_token(tok, 2);
        }
        assert_eq!(cache.recent_tokens, vec![12, 13]);
    }

    /// A model with no lookback must not accumulate one — every other
    /// architecture passes `keep == 0` by never calling this at all, but the
    /// guard is what makes an unbounded `Vec` impossible rather than merely
    /// unused.
    #[test]
    fn a_zero_lookback_records_nothing() {
        let mut cache = KvCache::new(1, 8, 4);
        cache.push_recent_token(7, 0);
        assert!(cache.recent_tokens.is_empty());
    }

    /// The lookback has to survive a slot save/restore, or a conversation
    /// resumed from disk would hash its first token against EOS padding
    /// where a live one hashes it against its real predecessors — the same
    /// prompt, two different answers, depending only on whether the slot
    /// happened to be persisted.
    #[test]
    fn the_recent_token_lookback_round_trips_through_a_slot_blob() {
        let mut cache = KvCache::new(1, 8, 4);
        cache.layers[0].push(&[1.0; 4], &[2.0; 4]);
        cache.push_recent_token(4242, 2);
        cache.push_recent_token(99, 2);
        let restored = KvCache::from_bytes(&cache.to_bytes()).expect("round trip");
        assert_eq!(restored.recent_tokens, vec![4242, 99]);
    }

    /// A blob written before the lookback existed carries no trailing
    /// section, and must still restore rather than being rejected as
    /// corrupt. Only `qwen4exp` reads the field, and no build that could
    /// write such a blob could serve it, so an empty lookback here is the
    /// truth rather than a loss.
    #[test]
    fn a_blob_without_a_lookback_section_still_restores() {
        let mut cache = KvCache::new(1, 8, 4);
        cache.layers[0].push(&[1.0; 4], &[2.0; 4]);
        let bytes = cache.to_bytes();
        let restored = KvCache::from_bytes(&bytes).expect("round trip");
        assert!(restored.recent_tokens.is_empty());
        assert_eq!(restored.committed_len(), 1);
    }

    /// The reuse bound must be per-*layer*, and taken from the shortest
    /// layer that holds anything.
    #[test]
    fn the_reuse_bound_follows_the_shortest_layer_that_holds_anything() {
        let mut cache = KvCache::new(3, 512, 4);
        for i in 0..100 {
            cache.layers[0].push(&[i as f32; 4], &[i as f32; 4]);
        }
        for i in 0..40 {
            cache.layers[1].push(&[i as f32; 4], &[i as f32; 4]);
        }
        // Layer 2 stays empty, standing in for gemma's cross-layer KV donor.
        assert_eq!(cache.host_committed_len(), 40);
    }

    /// An empty donor layer must not switch reuse off for the whole cache —
    /// a plain minimum over every layer would read it as "nothing reusable".
    #[test]
    fn a_permanently_empty_donor_layer_does_not_veto_reuse() {
        let mut cache = KvCache::new(2, 512, 4);
        for i in 0..64 {
            cache.layers[0].push(&[i as f32; 4], &[i as f32; 4]);
        }
        assert_eq!(cache.host_committed_len(), 64);
    }

    /// Copying exactly the host-resident length is fine; the row past it is
    /// what used to panic on a slice index instead of on the bound.
    #[test]
    fn copying_up_to_the_host_length_succeeds_and_past_it_is_refused() {
        let src = cache_with_gpu_only_tail(239, 7);
        let mut dst = KvCache::new(1, 512, 4);
        dst.copy_prefix_from(&src, src.host_committed_len());
        assert_eq!(dst.layers[0].len, 239);

        let mut dst = KvCache::new(1, 512, 4);
        let over = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dst.copy_prefix_from(&src, src.committed_len())
        }));
        assert!(
            over.is_err(),
            "copying past the host rows must not be allowed"
        );
    }

    /// [`gpu_layer_bytes`] has to agree with what [`GpuLayerCache::new`]
    /// actually allocates, and nothing but a reading of both enforces it —
    /// the allocation needs a `wgpu::Device`, so this restates its
    /// arithmetic rather than calling it. Restated as *literals*, not as a
    /// second expression of the same formula: a copy of the formula would
    /// follow a mistake in the original into the test.
    #[test]
    fn kv_mirror_bytes_agree_with_the_allocation() {
        // 8 rows x 64 kv_dim in f16 = 1024 bytes per region. `v` starts at
        // the next 256-byte boundary, which 1024 already is, so the buffer
        // is 2048; scratch is 8 rows x 4 heads x 4 bytes = 128.
        assert_eq!(
            gpu_layer_bytes(8, 64, 4, KvStorage::F16, 256),
            2048 + 128,
            "f16"
        );
        // f32 doubles both regions: 2048 each, `v` at 2048, buffer 4096.
        assert_eq!(
            gpu_layer_bytes(8, 64, 4, KvStorage::F32, 256),
            4096 + 128,
            "f32"
        );
        // q8_0 stores 32 elements in 36 bytes: 8*64/32 = 16 blocks = 576
        // bytes, `v` padded up to 768, so 768 + 576 = 1344.
        assert_eq!(
            gpu_layer_bytes(8, 64, 4, KvStorage::Q8_0, 256),
            1344 + 128,
            "q8_0"
        );
    }

    /// The sizing scales a probe cache built at capacity 1 — which is how
    /// `engine::footprint` asks about a context far too large to allocate.
    #[test]
    fn gpu_mirror_bytes_scale_a_capacity_one_probe_to_a_real_context() {
        let probe = KvCache::new(4, 1, 64);
        let real = KvCache::new(4, 1024, 64);
        assert_eq!(
            probe.gpu_mirror_bytes(1024, 8, KvStorage::F16),
            real.gpu_mirror_bytes(1024, 8, KvStorage::F16)
        );
        // And it is four layers' worth, not one.
        assert_eq!(
            probe.gpu_mirror_bytes(1024, 8, KvStorage::F16),
            4 * gpu_layer_bytes(1024, 64, 8, KvStorage::F16, ASSUMED_STORAGE_ALIGN)
        );
    }

    /// One device of a split holds the KV of the layers it holds — and its
    /// share is not a fraction of the total unless every layer is the same
    /// size, which is exactly what a per-layer `head_count_kv` or a
    /// block-compressed layer breaks.
    ///
    /// Two 64-wide layers and two 256-wide ones: the device holding the two
    /// narrow ones holds a fifth of the cache, not half of it. A footprint
    /// that scaled by layer count would report four times too much headroom
    /// used on one card and far too little on the other.
    #[test]
    fn a_devices_share_of_the_cache_is_its_layers_not_its_layer_count() {
        let mixed = KvCache::new_with_dims(1, &[64, 64, 256, 256]);
        let bytes =
            |keep: fn(usize) -> bool| mixed.gpu_mirror_bytes_where(1024, 8, KvStorage::F16, keep);
        let narrow = bytes(|layer| layer < 2);
        let wide = bytes(|layer| layer >= 2);
        assert_eq!(
            narrow + wide,
            mixed.gpu_mirror_bytes(1024, 8, KvStorage::F16)
        );
        // Not half and half: the wide pair is four times the narrow pair, so
        // splitting two layers each puts 20% on one card and 80% on the other.
        assert!(
            wide > narrow * 3,
            "expected the wide layers to dominate: {narrow} vs {wide}"
        );
        // And an empty selection is zero rather than the whole thing.
        assert_eq!(bytes(|_| false), 0);
    }

    /// A strided layer stores one row per `stride` tokens, so its mirror is
    /// that much smaller — the same rule `LayerCache::new_strided` applies
    /// to the host buffers.
    #[test]
    fn gpu_mirror_bytes_follow_a_layers_stride() {
        let dense = KvCache::new_with_strided_dims(1, &[(64, 1)]);
        let blocked = KvCache::new_with_strided_dims(1, &[(64, 4)]);
        assert_eq!(
            blocked.gpu_mirror_bytes(1024, 8, KvStorage::F16),
            gpu_layer_bytes(256, 64, 8, KvStorage::F16, ASSUMED_STORAGE_ALIGN)
        );
        assert!(
            blocked.gpu_mirror_bytes(1024, 8, KvStorage::F16)
                < dense.gpu_mirror_bytes(1024, 8, KvStorage::F16)
        );
    }

    /// A recurrent layer has no GPU mirror, so it must not be charged for
    /// one — and a mixed model's attention layers must still be counted.
    #[test]
    fn gpu_mirror_bytes_ignore_recurrent_state() {
        let mixed = KvCache::new_mixed(1, &[64, 64], &[RecurrentSpec::delta_net(128, 4, 8, 16)]);
        let attention_only = KvCache::new(2, 1, 64);
        assert_eq!(
            mixed.gpu_mirror_bytes(512, 8, KvStorage::F16),
            attention_only.gpu_mirror_bytes(512, 8, KvStorage::F16)
        );
    }

    /// **The differential test the paged backing has to pass.** Same pushes
    /// into both layouts, then every row compared — paging changes where a row
    /// lives and must not change what it is.
    ///
    /// Compared for **equality**, not a tolerance: nothing here is arithmetic.
    /// A row is copied into a page and read back out, so anything but the same
    /// bits is a addressing bug, and a tolerance would hide exactly the
    /// off-by-one-page error this is written to catch.
    fn paged_and_contiguous_agree(kv_dim: usize, stride: usize, page_tokens: usize, rows: usize) {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;

        let geom = vec![LayerGeometry { kv_dim, stride }];
        // Room for far more pages than the sequence needs, so this measures
        // addressing rather than reclaim.
        let pool = Arc::new(KvPool::with_policy(64, page_tokens, geom, Policy::Lru));
        let capacity = rows * stride + stride;
        let mut paged =
            KvCache::new_with_strided_dims(capacity, &strided_dims(&pool)).into_paged(pool);
        let mut plain = KvCache::new_with_strided_dims(capacity, &[(kv_dim, stride)]);

        for r in 0..rows {
            let k: Vec<f32> = (0..kv_dim).map(|d| (r * kv_dim + d) as f32).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            paged.layers[0].push(&k, &v);
            plain.layers[0].push(&k, &v);
        }

        assert_eq!(paged.layers[0].len, plain.layers[0].len, "row count");
        assert_eq!(
            paged.host_committed_len(),
            plain.host_committed_len(),
            "host-resident token count"
        );
        for r in 0..rows {
            assert_eq!(
                paged.layers[0].key_at(r, 0, kv_dim),
                plain.layers[0].key_at(r, 0, kv_dim),
                "key row {r} (kv_dim {kv_dim}, stride {stride}, page {page_tokens})"
            );
            assert_eq!(
                paged.layers[0].value_at(r, 0, kv_dim),
                plain.layers[0].value_at(r, 0, kv_dim),
                "value row {r}"
            );
        }
    }

    #[test]
    fn paged_rows_read_back_exactly_as_contiguous_ones() {
        // Sequence lengths either side of a page boundary, and a page size that
        // does not divide them, so a partial tail is always in play.
        for rows in [1usize, 3, 4, 5, 8, 9, 17, 33] {
            paged_and_contiguous_agree(8, 1, 4, rows);
        }
    }

    /// The same across page sizes, including one page per token — the
    /// degenerate setting that turns the block table into an identity map and
    /// would hide a page-arithmetic error.
    #[test]
    fn paged_rows_agree_at_every_page_size() {
        for page_tokens in [1usize, 2, 4, 8, 16] {
            paged_and_contiguous_agree(8, 1, page_tokens, 20);
        }
    }

    /// And for a block-compressed layer, whose rows stand for several tokens —
    /// the case where rows-per-page is not page-tokens.
    #[test]
    fn paged_rows_agree_for_a_strided_layer() {
        for stride in [2usize, 4, 128] {
            paged_and_contiguous_agree(4, stride, 8, 10);
        }
    }

    /// Rollback must land in the same place under both layouts, including the
    /// hard case: a cut *inside* a sealed page, which cannot be rewritten and
    /// so has to be carried back into the tail.
    #[test]
    fn paged_truncate_matches_contiguous_including_mid_page_cuts() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        const KV: usize = 8;
        const PAGE: usize = 4;

        for cut in [0usize, 1, 3, 4, 5, 7, 8, 11] {
            let geom = vec![LayerGeometry {
                kv_dim: KV,
                stride: 1,
            }];
            let pool = Arc::new(KvPool::with_policy(64, PAGE, geom, Policy::Lru));
            let mut paged =
                KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool.clone());
            let mut plain = KvCache::new_with_strided_dims(64, &[(KV, 1)]);
            for r in 0..12usize {
                let k: Vec<f32> = (0..KV).map(|d| (r * KV + d) as f32).collect();
                let v: Vec<f32> = k.iter().map(|x| -x).collect();
                paged.layers[0].push(&k, &v);
                plain.layers[0].push(&k, &v);
            }
            paged.truncate(cut);
            plain.truncate(cut);
            assert_eq!(paged.layers[0].len, plain.layers[0].len, "cut {cut}");

            // And writing continues correctly afterwards, which is what a
            // rejected speculative draft does next.
            for r in 0..5usize {
                let k: Vec<f32> = (0..KV).map(|d| (1000 + r * KV + d) as f32).collect();
                let v: Vec<f32> = k.iter().map(|x| -x).collect();
                paged.layers[0].push(&k, &v);
                plain.layers[0].push(&k, &v);
            }
            for r in 0..(cut + 5) {
                assert_eq!(
                    paged.layers[0].key_at(r, 0, KV),
                    plain.layers[0].key_at(r, 0, KV),
                    "after cut {cut}, row {r}"
                );
            }
        }
    }

    /// A finished sequence's pages go back, or the pool leaks until it is full
    /// of content nobody holds and nobody can reclaim.
    #[test]
    fn dropping_a_paged_cache_returns_its_pages() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        let geom = vec![LayerGeometry {
            kv_dim: 4,
            stride: 1,
        }];
        let pool = Arc::new(KvPool::with_policy(16, 4, geom, Policy::Lru));
        {
            let mut cache =
                KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool.clone());
            for r in 0..9usize {
                cache.layers[0].push(&[r as f32; 4], &[r as f32; 4]);
            }
            assert_eq!(pool.stats().live_pages, 2, "two sealed pages while live");
        }
        assert_eq!(
            pool.stats().live_pages,
            0,
            "a dropped sequence left its pages held"
        );
    }

    /// A paged cache must serialize the whole conversation, not the tail page
    /// it happens to be holding — a snapshot that restored a fraction while
    /// claiming the whole is the quiet failure this guards.
    #[test]
    fn a_paged_cache_round_trips_through_bytes() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        const KV: usize = 8;
        let geom = vec![LayerGeometry {
            kv_dim: KV,
            stride: 1,
        }];
        let pool = Arc::new(KvPool::with_policy(32, 4, geom, Policy::Lru));
        let mut paged = KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool);
        let mut plain = KvCache::new_with_strided_dims(64, &[(KV, 1)]);
        for r in 0..11usize {
            let k: Vec<f32> = (0..KV).map(|d| (r * KV + d) as f32).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            paged.layers[0].push(&k, &v);
            plain.layers[0].push(&k, &v);
        }
        assert_eq!(
            paged.to_bytes(),
            plain.to_bytes(),
            "a paged cache serialized to different bytes than the same rows \
             stored contiguously"
        );
        let restored = KvCache::from_bytes(&paged.to_bytes()).expect("round trip");
        for r in 0..11usize {
            assert_eq!(
                restored.layers[0].key_at(r, 0, KV),
                plain.layers[0].key_at(r, 0, KV),
                "row {r}"
            );
        }
    }

    /// The same for a duplicate, which slot persistence takes.
    #[test]
    fn duplicating_a_paged_cache_snapshots_every_row() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        const KV: usize = 4;
        let geom = vec![LayerGeometry {
            kv_dim: KV,
            stride: 1,
        }];
        let pool = Arc::new(KvPool::with_policy(32, 4, geom, Policy::Lru));
        let mut paged = KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool);
        for r in 0..9usize {
            paged.layers[0].push(&[r as f32; KV], &[-(r as f32); KV]);
        }
        let copy = paged.duplicate();
        assert_eq!(copy.layers[0].len, 9);
        for r in 0..9usize {
            assert_eq!(
                copy.layers[0].key_at(r, 0, KV),
                &vec![r as f32; KV][..],
                "row {r}"
            );
        }
    }

    /// **End to end: two sequences share a prefix, and the sharer reads exactly
    /// what the builder computed.**
    ///
    /// This is the thing the whole pool exists for, and the thing the engine
    /// as it stands cannot do — measured on the running server, a second
    /// concurrent request on a shared prefix re-prefills the whole of it.
    ///
    /// The check that matters is not that sharing *happened* (page counts would
    /// show that) but that the shared rows are bit-identical to what a private
    /// recompute produces. A pool that shares the wrong page is worse than one
    /// that shares nothing.
    #[test]
    fn a_second_sequence_shares_a_prefix_instead_of_rebuilding_it() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use crate::engine::prefix_index::{PrefixIndex, page_tags};
        use std::sync::Arc;
        const KV: usize = 8;
        const PAGE: usize = 4;

        let geom = vec![
            LayerGeometry {
                kv_dim: KV,
                stride: 1,
            },
            LayerGeometry {
                kv_dim: KV,
                stride: 1,
            },
        ];
        let pool = Arc::new(KvPool::with_policy(32, PAGE, geom, Policy::Lru));
        let index = PrefixIndex::new(PAGE);

        // A shared 12-token prompt, then two different continuations.
        let shared: Vec<u32> = (0..12).collect();
        let row = |r: usize, d: usize| (r * KV + d) as f32;

        // First sequence: builds the shared prefix itself.
        let tags = page_tags(&shared, PAGE);
        let mut first =
            KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool.clone());
        first.set_page_tags(&tags);
        for r in 0..shared.len() {
            let k: Vec<f32> = (0..KV).map(|d| row(r, d)).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            for layer in &mut first.layers {
                layer.push(&k, &v);
            }
        }
        // The commit point the forward pass owns: these positions have been
        // through every layer, so the pages covering them may be published.
        first.commit_pages();
        for (i, &t) in tags.iter().enumerate() {
            index.remember(t, &shared[i * PAGE..(i + 1) * PAGE]);
        }
        let pages_after_first = pool.stats().live_pages;
        assert_eq!(pages_after_first, 3, "12 tokens at 4 per page");

        // Second sequence: resolves the same prompt and adopts it.
        let resolved = index.resolve(&shared, false);
        assert_eq!(resolved.shared.len(), 3, "the whole prompt is known");
        let mut second =
            KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool.clone());
        let adopted = second
            .adopt_shared_pages(&resolved.shared, PAGE)
            .expect("the pages are resident");
        assert_eq!(adopted, 12);

        // **No new pages.** The prefix is one copy with two holders, which is
        // the difference between sharing and the pool merely holding two
        // copies of the same thing.
        assert_eq!(
            pool.stats().live_pages,
            pages_after_first,
            "adopting a prefix allocated fresh pages instead of sharing"
        );
        for &t in &resolved.shared {
            // Both sequences hold every shared page.
            let page = pool.acquire(&[t]).expect("resident");
            assert_eq!(pool.refs(page[0].page), 3, "two holders plus this probe");
            pool.release(&[page[0].page]);
        }

        // And it reads what the first sequence computed, in every layer.
        for layer in 0..2 {
            for r in 0..12 {
                let expect: Vec<f32> = (0..KV).map(|d| row(r, d)).collect();
                assert_eq!(
                    second.layers[layer].key_at(r, 0, KV),
                    &expect[..],
                    "layer {layer} row {r} keys"
                );
                let expect_v: Vec<f32> = expect.iter().map(|x| -x).collect();
                assert_eq!(
                    second.layers[layer].value_at(r, 0, KV),
                    &expect_v[..],
                    "layer {layer} row {r} values"
                );
            }
        }

        // The second sequence then continues past the shared part, privately.
        for r in 12..16usize {
            let k: Vec<f32> = (0..KV).map(|d| row(r, d)).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            for layer in &mut second.layers {
                layer.push(&k, &v);
            }
        }
        assert_eq!(second.layers[0].len, 16);
        for r in 12..16 {
            let expect: Vec<f32> = (0..KV).map(|d| row(r, d)).collect();
            assert_eq!(second.layers[0].key_at(r, 0, KV), &expect[..], "row {r}");
        }
        // The first sequence is untouched by any of it.
        for r in 0..12 {
            let expect: Vec<f32> = (0..KV).map(|d| row(r, d)).collect();
            assert_eq!(first.layers[0].key_at(r, 0, KV), &expect[..], "row {r}");
        }
    }

    /// A prefix the pool has since reclaimed must be declined, not
    /// half-adopted — the caller prefills instead, which is correct and slow
    /// rather than fast and wrong.
    #[test]
    fn adopting_a_reclaimed_prefix_takes_nothing() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        let geom = vec![LayerGeometry {
            kv_dim: 4,
            stride: 1,
        }];
        let pool = Arc::new(KvPool::with_policy(4, 4, geom, Policy::Lru));
        let mut cache =
            KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool.clone());
        // Tags nothing ever sealed.
        assert!(cache.adopt_shared_pages(&[11, 22], 4).is_none());
        assert_eq!(
            pool.stats().live_pages,
            0,
            "a refused adoption left pages held"
        );
    }

    /// The block table is what the paged kernel reads to turn a position into
    /// a row, so it has to name this sequence's pages in logical order.
    #[test]
    fn the_block_table_names_the_pages_in_order() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        const KV: usize = 4;
        let pool = Arc::new(KvPool::with_policy(
            8,
            4,
            vec![LayerGeometry {
                kv_dim: KV,
                stride: 1,
            }],
            Policy::Lru,
        ));
        let mut cache = KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool);
        assert!(cache.layers[0].block_table().is_empty());
        for r in 0..9usize {
            cache.layers[0].push(&[r as f32; KV], &[r as f32; KV]);
        }
        // Two whole pages sealed; the ninth row is a tail with no page yet.
        assert_eq!(cache.layers[0].block_table().len(), 2);
        let table: Vec<u32> = cache.layers[0].block_table().to_vec();
        assert_ne!(
            table[0], table[1],
            "two logical pages share one physical page"
        );
    }

    /// Rolling back must not leave the device believing it has mirrored pages
    /// the sequence no longer holds — the next pages to take those indices
    /// would then never be uploaded, and the kernel would read whatever the
    /// previous occupant left.
    #[test]
    fn a_rollback_pulls_the_device_watermark_back() {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        let Some(vulkan) = crate::engine::backend::vulkan::shared_test_backend() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        const KV: usize = 32;
        let mut pool = KvPool::with_policy(
            16,
            4,
            vec![LayerGeometry {
                kv_dim: KV,
                stride: 1,
            }],
            Policy::Lru,
        );
        let (device, queue) = vulkan.device_and_queue();
        pool.attach_device(device, KvStorage::F16, 64);
        let pool = Arc::new(pool);

        let mut cache = KvCache::new_with_strided_dims(64, &strided_dims(&pool)).into_paged(pool);
        for r in 0..12usize {
            cache.layers[0].push(&[r as f32; KV], &[r as f32; KV]);
        }
        cache.layers[0].sync_pool_device(queue);
        assert_eq!(cache.layers[0].block_table().len(), 3);

        assert_eq!(cache.layers[0].device_synced_pages(), 3);

        // Roll back into the first page and write again.
        cache.truncate(2);
        assert!(cache.layers[0].block_table().is_empty());
        assert_eq!(
            cache.layers[0].device_synced_pages(),
            0,
            "the device watermark outlived the pages it counted; the next pages \
             to take those indices would never be uploaded"
        );
        for r in 100..108usize {
            cache.layers[0].push(&[r as f32; KV], &[r as f32; KV]);
        }
        // Two fresh pages, and the sync must upload both rather than believing
        // three were already done.
        assert_eq!(cache.layers[0].block_table().len(), 2);
        cache.layers[0].sync_pool_device(queue);
        assert_eq!(cache.layers[0].device_synced_pages(), 2);
        for r in 0..2usize {
            assert_eq!(cache.layers[0].key_at(r, 0, KV), &[r as f32; KV][..]);
        }
        for r in 2..10usize {
            assert_eq!(
                cache.layers[0].key_at(r, 0, KV),
                &[(98 + r) as f32; KV][..],
                "row {r}"
            );
        }
    }

    /// **A mixed architecture keeps its recurrent state when it goes paged.**
    ///
    /// This is the bug that shipped for one commit and that nothing caught: the
    /// paged cache used to be built from the pool's layer list, which knows only
    /// about positional layers, so `recurrent` came out empty. Eight
    /// architectures index `cache.recurrent[..]` directly — `qwen35moe`,
    /// `qwen3next`, `nemotron_h_moe`, `kda`, `inkling` among them — and every
    /// one of them would have panicked on its first linear-attention layer the
    /// moment paging became the default.
    ///
    /// It survived because every paged test until now used a purely positional
    /// cache, which cannot express the difference.
    #[test]
    fn going_paged_keeps_a_mixed_architecture_s_recurrent_state() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        const KV: usize = 8;
        let pool = Arc::new(KvPool::with_policy(
            16,
            4,
            vec![LayerGeometry {
                kv_dim: KV,
                stride: 1,
            }],
            Policy::Lru,
        ));
        // What a mixed architecture hands back: one attention layer and one
        // recurrent state beside it.
        let built = KvCache::new_mixed(64, &[KV], &[RecurrentSpec::delta_net(2, 3, 1, 2)]);
        assert_eq!(built.recurrent.len(), 1);

        let paged = built.into_paged(pool);
        assert_eq!(
            paged.recurrent.len(),
            1,
            "the recurrent state was dropped on the way into the pool; every \
             linear-attention layer of this model would index an empty vector"
        );
        assert!(
            paged.layers[0].paged.is_some(),
            "the attention layer is paged"
        );
    }

    /// And the geometry has to match, or the pool belongs to another model —
    /// the same class of error the slot fingerprint refuses.
    #[test]
    #[should_panic(expected = "does not match the pool")]
    fn going_paged_refuses_a_pool_built_for_a_different_shape() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        use std::sync::Arc;
        let pool = Arc::new(KvPool::with_policy(
            8,
            4,
            vec![LayerGeometry {
                kv_dim: 8,
                stride: 1,
            }],
            Policy::Lru,
        ));
        let _ = KvCache::new_with_strided_dims(64, &[(16, 1)]).into_paged(pool);
    }

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

    /// `committed_len` must answer in **tokens** even when every slot is
    /// block-compressed.
    ///
    /// This is the shape no architecture builds today: `deepseek4` is the only
    /// one that strides at all, and it always has ordinary per-token slots
    /// alongside, which are the longest — so the old `max(len)` returned the
    /// token count by coincidence rather than by construction. A change to how
    /// the cache is allocated is exactly what would remove that coincidence,
    /// and it would show up as a prefix cache quietly reusing a fraction of
    /// what it could, which nothing would fail on.
    #[test]
    fn committed_len_is_tokens_even_when_every_slot_is_strided() {
        let mut cache = KvCache::new_with_strided_dims(64, &[(4, 4), (4, 4)]);
        for _ in 0..5 {
            for layer in &mut cache.layers {
                layer.push(&[1.0; 4], &[2.0; 4]);
            }
        }
        // Five rows of stride 4 stand for twenty token positions.
        assert_eq!(cache.layers[0].len, 5, "rows");
        assert_eq!(cache.committed_len(), 20, "tokens");
    }

    /// A mixed cache compares its slots in one unit.
    #[test]
    fn committed_len_compares_slots_in_the_same_unit() {
        // One per-token slot at 8 tokens, one stride-4 slot at 3 rows — which
        // is 12 tokens, and therefore the longer of the two. Reading rows as
        // tokens would pick the 8.
        let mut cache = KvCache::new_with_strided_dims(64, &[(4, 1), (4, 4)]);
        for _ in 0..8 {
            cache.layers[0].push(&[1.0; 4], &[2.0; 4]);
        }
        for _ in 0..3 {
            cache.layers[1].push(&[1.0; 4], &[2.0; 4]);
        }
        assert_eq!(cache.committed_len(), 12);
    }

    /// The structural signature has to distinguish two caches that differ
    /// only in stride — they lay their rows out differently and mean
    /// different things by one.
    ///
    /// Reachable only through the slot fingerprint, where the model label is
    /// hashed alongside and masks it. That makes this a latent gap rather than
    /// a live bug, and a signature that is right only because something else
    /// is also checked is not one worth relying on.
    #[test]
    fn the_structure_tag_distinguishes_stride() {
        let plain = KvCache::new_with_strided_dims(64, &[(4, 1)]);
        let strided = KvCache::new_with_strided_dims(64, &[(4, 4)]);
        assert_ne!(
            plain.structure_tag(),
            strided.structure_tag(),
            "two different KV layouts share a structural signature"
        );
        // Same shape, different capacity, is still the same structure —
        // capacity is a per-request size, not a property of the model.
        assert_eq!(
            KvCache::new_with_strided_dims(64, &[(4, 4)]).structure_tag(),
            KvCache::new_with_strided_dims(128, &[(4, 4)]).structure_tag(),
        );
    }

    /// Adoption and copying must leave the destination in exactly the same
    /// state — same committed length, same bytes, same readback.
    ///
    /// This is the whole correctness claim of moving the buffers instead of
    /// copying them, and it is checked against the copy rather than against a
    /// hand-written expectation, so the two paths cannot drift.
    #[test]
    fn adopting_a_prefix_leaves_the_same_state_as_copying_it() {
        let source = || {
            let mut c = KvCache::new(2, 16, 4);
            for i in 0..10u32 {
                for layer in &mut c.layers {
                    let f = i as f32;
                    layer.push(&[f, f + 1.0, f + 2.0, f + 3.0], &[-f, -f, -f, -f]);
                }
            }
            c
        };
        for len in [1, 5, 9, 10] {
            let mut copied = KvCache::new(2, 16, 4);
            copied.copy_prefix_from(&source(), len);
            let mut adopted = KvCache::new(2, 16, 4);
            adopted.adopt_prefix(source(), len);

            assert_eq!(adopted.committed_len(), copied.committed_len(), "len {len}");
            for (a, c) in adopted.layers.iter().zip(copied.layers.iter()) {
                assert_eq!(a.len, c.len, "len {len}");
                for pos in 0..c.len {
                    assert_eq!(
                        a.key_at(pos, 0, 4),
                        c.key_at(pos, 0, 4),
                        "len {len} pos {pos}"
                    );
                    assert_eq!(
                        a.value_at(pos, 0, 4),
                        c.value_at(pos, 0, 4),
                        "len {len} pos {pos}"
                    );
                }
            }
        }
    }

    /// An adopted cache is still this request's cache: it keeps the capacity
    /// it was built with, not the source's, and can be pushed to right up to
    /// that ceiling.
    ///
    /// The failure this guards is a conversation that reuses a short earlier
    /// turn and then panics part-way through generating a long answer,
    /// because it inherited the earlier turn's smaller ceiling.
    #[test]
    fn an_adopted_cache_keeps_its_own_capacity() {
        let mut short = KvCache::new(1, 4, 4);
        for i in 0..4u32 {
            short.layers[0].push(&[i as f32; 4], &[i as f32; 4]);
        }
        let mut long = KvCache::new(1, 64, 4);
        long.adopt_prefix(short, 4);
        assert_eq!(
            long.layers[0].capacity(),
            64,
            "capacity came from the source"
        );
        // Everything the new request is still entitled to generate.
        for i in 4..64u32 {
            long.layers[0].push(&[i as f32; 4], &[i as f32; 4]);
        }
        assert_eq!(long.committed_len(), 64);
        assert_eq!(long.layers[0].key_at(63, 0, 4), &[63.0; 4]);
    }

    /// Recurrent state is carried across, and it is copied rather than moved —
    /// it is a fixed-size evolving state, not a per-position history, so there
    /// is nothing to take.
    #[test]
    fn adopting_carries_recurrent_state() {
        let spec = RecurrentSpec::delta_net(4, 2, 2, 2);
        let mut src = KvCache::new_mixed(8, &[4], &[spec]);
        src.layers[0].push(&[1.0; 4], &[2.0; 4]);
        src.recurrent[0].delta_state_mut(0)[0] = 42.0;

        let mut dst = KvCache::new_mixed(8, &[4], &[spec]);
        dst.adopt_prefix(src, 1);
        assert_eq!(dst.committed_len(), 1);
        assert_eq!(dst.recurrent[0].delta_state_mut(0)[0], 42.0);
    }

    /// The mirror is sized to what has been generated, not to what was asked
    /// for — which is the whole change.
    ///
    /// Measured before it, on a 3.98 GiB card with a two-token prompt and the
    /// same one-word answer: 2191 MiB of VRAM at `max_tokens = 64` against
    /// 3727 MiB at `max_tokens = 32768`. Host buffers never had this problem,
    /// because a large zeroed `Vec` is `mmap`ed and the kernel commits pages
    /// only as they are written; device memory is not overcommitted, so there
    /// the reservation was real.
    #[test]
    fn the_mirror_is_sized_to_what_was_generated_not_what_was_asked_for() {
        let huge = 32768;
        // A short answer in a request that asked for a lot: one floor-sized
        // allocation, not the budget.
        assert_eq!(mirror_rows_for(1, huge), 256);
        assert_eq!(mirror_rows_for(200, huge), 256);
        // Then doubling, so a long generation pays a handful of copies rather
        // than one per block for its whole length.
        assert_eq!(mirror_rows_for(257, huge), 512);
        assert_eq!(mirror_rows_for(1000, huge), 1024);
        // And a request that really does run to its budget ends up with
        // exactly what it would have been given up front.
        assert_eq!(mirror_rows_for(huge, huge), huge);
    }

    /// The mirror must always have room for **one row past the committed
    /// length**, and this is the test that says why.
    ///
    /// The fused decode path binds the mirror and then writes the current
    /// token's key and value at row `len`, *before* the host-side `push` that
    /// makes that row committed. A mirror sized to exactly `len` puts that
    /// write one row past the end of the k region — and because k and v are
    /// two sub-ranges of one buffer, it lands on row 0 of v rather than
    /// outside the allocation, so the driver raises nothing and the corruption
    /// is silent.
    ///
    /// Sizing to `capacity` hid this for as long as the mirror was allocated
    /// that way, because capacity always exceeds `len`. It surfaced within an
    /// hour of sizing to demand — as a buffer-overrun validation error one
    /// growth step later, when the *copy* tried to carry more rows than the
    /// source held.
    #[test]
    fn the_mirror_always_has_room_for_the_row_decode_is_about_to_write() {
        for capacity in [1, 2, 255, 256, 257, 4096] {
            for len in 0..capacity {
                // The call site asks for `len + 1`; this is that contract.
                let rows = mirror_rows_for(len + 1, capacity);
                assert!(
                    rows > len,
                    "a layer at {len} of {capacity} got {rows} rows — no room for the \
                     in-flight row, which would land on row 0 of the value region"
                );
                assert!(rows <= capacity, "{rows} rows over capacity {capacity}");
            }
        }
    }

    /// Growth never exceeds the capacity the layer was built for, and never
    /// returns fewer rows than are already committed — the two ways a sizing
    /// rule can be wrong are over-allocating the thing this exists to stop
    /// and under-allocating into an out-of-bounds write.
    #[test]
    fn mirror_growth_stays_between_the_committed_length_and_the_capacity() {
        for capacity in [1, 7, 256, 300, 4096, 100_000] {
            for len in [0, 1, 2, 255, 256, 257, 1023, 4096, 99_999] {
                if len > capacity {
                    continue;
                }
                let rows = mirror_rows_for(len, capacity);
                assert!(rows >= len, "{rows} rows for {len} committed");
                assert!(rows <= capacity, "{rows} rows over capacity {capacity}");
                assert!(rows >= 1, "a mirror always has at least one row");
            }
        }
    }

    /// The grow-copy and the allocation have to agree about how many bytes a
    /// row takes, or growth copies the wrong range — silently, since both
    /// sides would still be in bounds.
    #[test]
    fn row_bytes_agrees_with_the_allocation_for_every_storage() {
        use crate::engine::backend::vulkan_shaders::KvStorage;
        let (rows, kv_dim, n_head) = (64, 128, 4);
        for storage in [KvStorage::F32, KvStorage::F16, KvStorage::Q8_0] {
            let region = row_bytes(rows, kv_dim, storage);
            // `gpu_layer_bytes` is the independently-written sizing the
            // footprint report uses: two regions plus the scratch.
            let whole = gpu_layer_bytes(rows, kv_dim, n_head, storage, 4);
            assert_eq!(
                whole,
                region.max(1).next_multiple_of(4) + region.max(1) + (rows * n_head * 4) as u64,
                "{storage:?}"
            );
        }
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
        let mut state = RecurrentLayerState::new(RecurrentSpec::delta_net(1, 3, 1, 1));
        let kernel = [1.0, 10.0, 100.0];
        // History starts at [0, 0]; first token contributes only via the
        // last tap: 0*1 + 0*10 + 5*100 = 500.
        let out = state.conv_step(&[5.0], &kernel);
        assert_eq!(out, vec![500.0]);
    }

    #[test]
    fn conv_step_slides_the_window_across_tokens() {
        let mut state = RecurrentLayerState::new(RecurrentSpec::delta_net(1, 3, 1, 1));
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
        let mut state = RecurrentLayerState::new(RecurrentSpec::delta_net(1, 2, 2, 2));
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

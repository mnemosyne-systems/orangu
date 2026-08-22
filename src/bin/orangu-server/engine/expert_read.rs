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

//! Where an expert's bytes are read *from*, when the tier decides to keep a
//! copy of them.
//!
//! `engine::expert_store`'s tier owns copies of hot experts. Until now those
//! copies came from the `mmap` — a `memcpy` out of the page cache. That is the
//! right thing when the page cache holds the model. It is the wrong thing when
//! it cannot: the read still goes through the cache, so every expert fetched
//! evicts something else, and the bytes are copied twice (disk → page cache →
//! tier) for data that will never be read through the mapping again.
//!
//! This module supplies the alternative colibri uses: read the expert straight
//! off the file with `pread`, optionally with `O_DIRECT` so the page cache is
//! bypassed entirely.
//!
//! # Choose it on evidence, not on principle
//!
//! **On a model that fits in RAM, `O_DIRECT` is dramatically slower and that
//! is not a bug.** The `mmap` path is a `memcpy` from memory; `O_DIRECT` is a
//! real disk read — measured on this box's storage at ~20 ms per 4 MiB against
//! a `memcpy`'s microseconds. It earns its place only where the page cache
//! *cannot* hold the model, which is the regime it exists for and the regime
//! this machine's storage cannot currently demonstrate (`BIG.md` M11: the
//! model drive is an NVMe behind a USB bridge, ~225 MB/s, where queue depth
//! buys 12.8%).
//!
//! Default is `mmap`. `ORANGU_EXPERT_READ=pread|direct` selects the others.
//!
//! # `O_DIRECT`'s alignment rules, and why they are not optional
//!
//! A tensor's byte range is aligned to nothing — GGUF places tensors at
//! whatever offset the previous one ended. `O_DIRECT` requires the file
//! offset, the length and the *buffer address* all aligned to the device's
//! logical block size. So a read is widened outward to a 4096-byte boundary
//! (a page, which is a multiple of every real device's block size), landed in
//! an aligned buffer, and the wanted bytes copied out of the middle. Getting
//! any of the three wrong returns `EINVAL`, not wrong data — which is at least
//! a loud failure, but the fallback below turns it into a slow one instead.

use std::sync::OnceLock;

/// Where the tier's copies come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// `memcpy` from the mapping — through the page cache.
    #[default]
    Mmap,
    /// `pread` from the shard file, still through the page cache.
    Pread,
    /// `pread` with `O_DIRECT`: the page cache is bypassed in both
    /// directions, so the read neither reads from it nor pollutes it.
    Direct,
}

impl Source {
    /// Whether reads through this route go through the page cache.
    ///
    /// Decides whether a `MADV_WILLNEED` prefetch hint is worth anything: a
    /// hint warms the page cache, and under `O_DIRECT` nothing reads from it.
    pub fn uses_page_cache(self) -> bool {
        matches!(self, Source::Mmap | Source::Pread)
    }
}

pub fn source() -> Source {
    static SOURCE: OnceLock<Source> = OnceLock::new();
    *SOURCE.get_or_init(|| match std::env::var("ORANGU_EXPERT_READ").as_deref() {
        Ok("pread") => Source::Pread,
        Ok("direct") => Source::Direct,
        _ => Source::Mmap,
    })
}

/// Alignment for `O_DIRECT`. A page is a multiple of every logical block size
/// a real device reports, so it satisfies the requirement without having to
/// interrogate the device for its own.
const DIRECT_ALIGN: u64 = 4096;

/// The read granule, in bytes — `[orangu-server].read_size`.
///
/// Set once at startup by [`set_read_size`]; [`crate::config::
/// DEFAULT_READ_SIZE`] until then, so a test or a tool that never
/// configures anything still reads at the documented default rather than at
/// zero.
static READ_SIZE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new((crate::config::DEFAULT_READ_SIZE as u64) * 1024);

/// Records the configured read granule. Called once, from `main`, before any
/// expert is read.
pub fn set_read_size(kib: usize) {
    READ_SIZE.store((kib as u64) * 1024, std::sync::atomic::Ordering::Relaxed);
}

/// The read granule in bytes, never below one page.
fn read_size() -> u64 {
    READ_SIZE
        .load(std::sync::atomic::Ordering::Relaxed)
        .max(DIRECT_ALIGN)
}

/// The file range to read for a span at `offset` of `len` bytes, as
/// `(start, total)` — both multiples of [`DIRECT_ALIGN`], with
/// `start <= offset` and `start + total >= offset + len`.
///
/// Widening to a granule rather than to a page is the whole point of the
/// knob: a device that pays one round trip per request delivers a small read
/// and a large one in nearly the same time, so a span well under the granule
/// costs nothing extra to fetch with its neighbours — and on a stacked
/// per-expert tensor the neighbours are the adjacent experts, which a batch
/// is disproportionately likely to want too.
///
/// The widening is **centred on the span, then clamped to the file's start**,
/// rather than extending only forwards: an expert near the end of a tensor
/// would otherwise widen into the next tensor and never into its own
/// neighbours.
///
/// Separated from the I/O so the arithmetic can be tested without a file.
fn read_window(offset: u64, len: usize, granule: u64) -> (u64, usize) {
    let want = (offset + len as u64).next_multiple_of(DIRECT_ALIGN)
        - (offset / DIRECT_ALIGN * DIRECT_ALIGN);
    if want >= granule {
        // Already at or over the granule: read exactly what is needed,
        // page-aligned. Splitting it further would only add round trips.
        let start = offset / DIRECT_ALIGN * DIRECT_ALIGN;
        return (start, want as usize);
    }
    let slack = granule - want;
    let back = (slack / 2 / DIRECT_ALIGN) * DIRECT_ALIGN;
    let start = (offset / DIRECT_ALIGN * DIRECT_ALIGN).saturating_sub(back);
    (start, granule as usize)
}

/// Reads `span`'s bytes by the configured route.
///
/// `span` is the expert's slice *of the mapping*, which is both the fallback
/// source and the way its file location is found. Any failure — the address
/// belongs to no registered shard, the file will not open, the read is short —
/// falls back to the mapping, because this is a performance route and must
/// never be the reason an expert cannot be read.
pub fn read_expert(span: &[u8]) -> Vec<u8> {
    match source() {
        Source::Mmap => span.to_vec(),
        Source::Pread => read_via_file(span, false).unwrap_or_else(|| span.to_vec()),
        Source::Direct => read_via_file(span, true).unwrap_or_else(|| span.to_vec()),
    }
}

#[cfg(target_os = "linux")]
fn read_via_file(span: &[u8], direct: bool) -> Option<Vec<u8>> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    let (path, offset) = crate::engine::page_cache::locate(span.as_ptr() as usize, span.len())?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    if direct {
        options.custom_flags(libc::O_DIRECT);
    }
    let file = options.open(&path).ok()?;

    // Widen outward to the configured granule. `O_DIRECT` needs page
    // alignment at minimum; the buffered path does not, but sharing one code
    // path keeps the two comparable — an A/B where the arms read different
    // byte ranges would be measuring the widening.
    let (start, total) = read_window(offset, span.len(), read_size());
    let lead = (offset - start) as usize;

    let mut buffer = AlignedBuffer::new(total)?;
    let mut filled = 0usize;
    while filled < total {
        // Safety: `buffer` owns `total` bytes from its base, `filled` is
        // within it, and the fd is open for the duration of the call.
        let got = unsafe {
            libc::pread(
                file.as_raw_fd(),
                buffer.as_mut_ptr().add(filled).cast(),
                total - filled,
                (start + filled as u64) as libc::off_t,
            )
        };
        match got {
            // A short read at the file's end is expected: the widened range
            // can run past it. Anything else short means the read failed.
            0 => break,
            n if n > 0 => filled += n as usize,
            _ => return None,
        }
    }
    // A widened window routinely runs past the end of the file, so a short
    // read is only a failure if it stopped before the bytes actually wanted.
    if filled < lead + span.len() {
        return None;
    }
    Some(buffer.as_slice()[lead..lead + span.len()].to_vec())
}

#[cfg(not(target_os = "linux"))]
fn read_via_file(_span: &[u8], _direct: bool) -> Option<Vec<u8>> {
    None
}

/// A heap buffer whose base address is `DIRECT_ALIGN`-aligned.
///
/// `Vec<u8>` gives no alignment guarantee beyond one byte, and `O_DIRECT`
/// rejects an unaligned destination with `EINVAL`. Allocated and freed through
/// `std::alloc` with an explicit `Layout` rather than by over-allocating a
/// `Vec` and offsetting into it, so the alignment is a property of the
/// allocation rather than of arithmetic done at every use.
#[cfg(target_os = "linux")]
struct AlignedBuffer {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

#[cfg(target_os = "linux")]
impl AlignedBuffer {
    fn new(size: usize) -> Option<Self> {
        let layout = std::alloc::Layout::from_size_align(size, DIRECT_ALIGN as usize).ok()?;
        // Safety: `size` is non-zero for every caller (a widened expert span),
        // and the layout was validated above.
        let ptr = unsafe { std::alloc::alloc(layout) };
        (!ptr.is_null()).then_some(Self { ptr, layout })
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn as_slice(&self) -> &[u8] {
        // Safety: `ptr` owns `layout.size()` initialized-or-not bytes; every
        // read here follows a `pread` that filled at least the range used.
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // Safety: `ptr` came from `alloc` with exactly this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::{DIRECT_ALIGN, read_window};

    /// A span far smaller than the granule is widened *to* the granule, still
    /// covers the span, and stays page-aligned at both ends.
    #[test]
    fn a_small_span_is_widened_to_the_whole_granule() {
        let granule = 8 * 1024 * 1024;
        let offset = 40 * 1024 * 1024 + 1234;
        let len = 274 * 1024;
        let (start, total) = read_window(offset, len, granule);
        assert_eq!(total, granule as usize, "should read a whole granule");
        assert_eq!(start % DIRECT_ALIGN, 0, "start must be page-aligned");
        assert_eq!(
            total as u64 % DIRECT_ALIGN,
            0,
            "length must be page-aligned"
        );
        assert!(start <= offset, "window must start at or before the span");
        assert!(
            start + total as u64 >= offset + len as u64,
            "window must cover the whole span"
        );
    }

    /// A span already larger than the granule is read as itself, page-aligned
    /// — never split into granule-sized pieces, which would only add round
    /// trips to a request the device already handles well.
    #[test]
    fn a_span_over_the_granule_is_not_split() {
        let granule = 1024 * 1024;
        let len = 5 * 1024 * 1024;
        let (start, total) = read_window(4096 * 7, len, granule);
        assert!(total as u64 >= len as u64);
        assert!(total as u64 - len as u64 <= DIRECT_ALIGN);
        assert_eq!(start, 4096 * 7);
    }

    /// The widening is clamped at the file's start rather than going
    /// negative — an expert in the first granule of a shard.
    #[test]
    fn widening_near_the_file_start_is_clamped_to_zero() {
        let granule = 8 * 1024 * 1024;
        let (start, total) = read_window(8192, 4096, granule);
        assert_eq!(start, 0, "cannot read before the start of the file");
        assert_eq!(total, granule as usize);
    }

    /// Every granule the config accepts produces a window that is aligned and
    /// covering, at offsets that are deliberately awkward — GGUF places a
    /// tensor at whatever offset the previous one ended.
    #[test]
    fn every_accepted_granule_covers_and_aligns_at_awkward_offsets() {
        for kib in [4usize, 64, 512, 8192, 65536] {
            let granule = (kib * 1024) as u64;
            for offset in [0u64, 1, 4095, 4096, 4097, 1_234_567, 987_654_321] {
                for len in [1usize, 4095, 4096, 300 * 1024, 9 * 1024 * 1024] {
                    let (start, total) = read_window(offset, len, granule);
                    assert_eq!(start % DIRECT_ALIGN, 0, "kib={kib} off={offset} len={len}");
                    assert_eq!(
                        total as u64 % DIRECT_ALIGN,
                        0,
                        "kib={kib} off={offset} len={len}"
                    );
                    assert!(start <= offset, "kib={kib} off={offset} len={len}");
                    assert!(
                        start + total as u64 >= offset + len as u64,
                        "kib={kib} off={offset} len={len}"
                    );
                }
            }
        }
    }

    use super::*;

    /// A hint warms the page cache; `O_DIRECT` reads bypass it. Hinting for a
    /// route that will not read from it spends bandwidth for nothing —
    /// measured at 4.15 GB per short request before this was checked.
    #[test]
    fn only_the_cached_routes_are_worth_hinting() {
        assert!(Source::Mmap.uses_page_cache());
        assert!(Source::Pread.uses_page_cache());
        assert!(!Source::Direct.uses_page_cache());
    }

    #[test]
    fn the_default_route_is_the_mapping() {
        // The env var is read once per process, so this asserts the mapping
        // is what an unconfigured build gets rather than re-reading it.
        assert_eq!(Source::default(), Source::Mmap);
    }

    /// An address in no registered shard has no file to read from, and must
    /// come back through the mapping rather than failing.
    #[test]
    fn an_unregistered_address_falls_back_to_the_mapping() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(9000).collect();
        assert_eq!(read_expert(&bytes), bytes);
        #[cfg(target_os = "linux")]
        {
            assert!(read_via_file(&bytes, false).is_none());
            assert!(read_via_file(&bytes, true).is_none());
        }
    }

    /// `O_DIRECT` rejects an unaligned destination outright, so the buffer's
    /// alignment is a correctness property rather than a tuning detail.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_read_buffer_is_page_aligned() {
        for size in [4096usize, 8192, 4096 * 7] {
            let buffer = AlignedBuffer::new(size).expect("allocation");
            assert_eq!(
                buffer.ptr as usize % DIRECT_ALIGN as usize,
                0,
                "size {size} came back unaligned"
            );
            assert_eq!(buffer.as_slice().len(), size);
        }
    }

    /// A tensor starts at an arbitrary offset, so the widened range has to
    /// cover it on both sides and the wanted bytes come out of the middle.
    #[test]
    fn the_widened_range_covers_an_unaligned_span() {
        for (offset, len) in [(0u64, 4096usize), (1, 10), (4095, 2), (5000, 9000)] {
            let start = offset / DIRECT_ALIGN * DIRECT_ALIGN;
            let lead = (offset - start) as usize;
            let total = (lead + len).next_multiple_of(DIRECT_ALIGN as usize);
            assert_eq!(start % DIRECT_ALIGN, 0, "start unaligned");
            assert_eq!(total % DIRECT_ALIGN as usize, 0, "length unaligned");
            assert!(lead + len <= total, "the wanted bytes fall outside");
        }
    }
}

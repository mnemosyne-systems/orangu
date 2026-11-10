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

//! How much of the model is in RAM, and how to put it back on the disk.
//!
//! A model whose weights do not fit in memory is measured twice — warm, with
//! whatever the page cache happens to be holding, and cold, with nothing — and
//! the two are different experiments, not a good and a bad run of the same
//! one. A benchmark that cannot say which it took cannot be compared against
//! another, so this module supplies both halves: the residency figure that
//! *describes* a run's starting state, and the reset that *chooses* it.
//!
//! **Dropping needs two calls, and only one of them is the obvious one.**
//! `posix_fadvise(POSIX_FADV_DONTNEED)` on the file is the usual answer, and
//! on its own it does nearly nothing here: the kernel's
//! `invalidate_mapping_pages` skips any page currently mapped into a process's
//! page tables, and every resident page of a loaded model is mapped into this
//! one. So the mapping is dropped first (`madvise(MADV_DONTNEED)`, which on a
//! read-only `MAP_PRIVATE` file mapping frees the page-table references
//! without invalidating an address — the same call and the same reasoning as
//! `loader::release_mapped_range`), and only then is the file advised. In that
//! order the pages are unmapped and clean, which is exactly the state
//! `POSIX_FADV_DONTNEED` will evict.
//!
//! Reversing the two, or doing either alone, produces a *partial* drop that
//! still reads as a cold run — the failure this module's before/after
//! residency figures exist to make visible rather than plausible.
//!
//! Linux only. Everywhere else residency is unknown and dropping is refused,
//! reported as such rather than as a successful no-op: a benchmark told "cold"
//! by a platform that cannot make it cold is worse off than one told nothing.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use memmap2::Mmap;

/// One mapped model shard, kept alive for the process lifetime.
///
/// Holding the `Arc` rather than a bare address is what makes the address
/// safe to use later: a registered mapping can never be unmapped underneath
/// an `madvise`. That pins the mapping, which is already how a loaded model
/// behaves — `QuantMatrix` clones keep their shard's `Arc<Mmap>` alive for as
/// long as the model exists, and this server serves one model.
struct Shard {
    path: PathBuf,
    mmap: Arc<Mmap>,
}

static SHARDS: Mutex<Vec<Shard>> = Mutex::new(Vec::new());

/// Records a shard mapping, so it can be measured and dropped later. Called
/// once per shard by `loader::LoadedModel::open_shards`.
pub fn register_shard(path: &Path, mmap: &Arc<Mmap>) {
    if let Ok(mut shards) = SHARDS.lock() {
        shards.push(Shard {
            path: path.to_path_buf(),
            mmap: Arc::clone(mmap),
        });
    }
}

/// Which shard file a mapped address belongs to, and its byte offset within
/// that file.
///
/// The bridge from "a slice of the mapping" back to "a place on disk", which
/// is what any explicit-read path needs: `mmap` hides the file, and `pread`
/// (with or without `O_DIRECT`) needs it back. A mapping covers its whole
/// shard from byte zero, so the offset is just the distance from the base.
///
/// `None` for an address in no registered shard — a test-built matrix, or a
/// bundled model whose segments were registered differently. The caller then
/// falls back to reading through the mapping, which always works.
pub fn locate(ptr: usize, len: usize) -> Option<(PathBuf, u64)> {
    let shards = SHARDS.lock().ok()?;
    shards.iter().find_map(|shard| {
        let base = shard.mmap.as_ptr() as usize;
        let end = base + shard.mmap.len();
        (ptr >= base && ptr + len <= end).then(|| (shard.path.clone(), (ptr - base) as u64))
    })
}

/// One shard's page-cache state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardResidency {
    pub path: PathBuf,
    pub bytes: u64,
    /// Bytes of this shard currently in RAM, or `None` where the platform
    /// cannot say. Never silently zero — "not resident" and "not knowable"
    /// are different answers, and only one of them means a run was cold.
    pub resident_bytes: Option<u64>,
}

impl ShardResidency {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "path": self.path.display().to_string(),
            "bytes": self.bytes,
            "resident_bytes": self.resident_bytes,
        })
    }
}

/// Every registered shard's size and how much of it is in RAM.
pub fn residency() -> Vec<ShardResidency> {
    let Ok(shards) = SHARDS.lock() else {
        return Vec::new();
    };
    shards
        .iter()
        .map(|shard| ShardResidency {
            path: shard.path.clone(),
            bytes: shard.mmap.len() as u64,
            resident_bytes: resident_bytes(&shard.mmap),
        })
        .collect()
}

/// Total model bytes and total resident bytes across every shard.
/// `resident` is `None` unless *every* shard could be measured — a partial
/// total would understate residency and read as a colder run than happened.
pub fn residency_totals(shards: &[ShardResidency]) -> (u64, Option<u64>) {
    let bytes = shards.iter().map(|s| s.bytes).sum();
    let resident = shards.iter().map(|s| s.resident_bytes).sum::<Option<u64>>();
    (bytes, resident)
}

/// What a drop attempt did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropReport {
    pub supported: bool,
    pub before: Vec<ShardResidency>,
    pub after: Vec<ShardResidency>,
}

impl DropReport {
    pub fn to_json(&self) -> serde_json::Value {
        let (bytes, before) = residency_totals(&self.before);
        let (_, after) = residency_totals(&self.after);
        serde_json::json!({
            "supported": self.supported,
            "model_bytes": bytes,
            "resident_before": before,
            "resident_after": after,
            "shards": self.after.iter().map(ShardResidency::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Evicts every registered shard from the page cache, reporting residency on
/// both sides of the attempt so a caller can see how much actually went.
///
/// Best-effort against the rest of the system: another process reading the
/// same file, or the kernel refusing an unmap, leaves pages behind. That is
/// why the report carries `resident_after` rather than a success flag — the
/// number is the result, and a caller wanting a guarantee must check it.
pub fn drop_model_page_cache() -> DropReport {
    let before = residency();
    #[cfg(target_os = "linux")]
    {
        if let Ok(shards) = SHARDS.lock() {
            for shard in shards.iter() {
                unmap_pages(&shard.mmap);
                advise_file_dontneed(&shard.path);
            }
        }
        let after = residency();
        DropReport {
            supported: true,
            before,
            after,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let after = before.clone();
        DropReport {
            supported: false,
            before,
            after,
        }
    }
}

/// Bytes of `mapping` currently in RAM, via `mincore`.
#[cfg(target_os = "linux")]
fn resident_bytes(mapping: &Mmap) -> Option<u64> {
    resident_bytes_of(mapping)
}

#[cfg(not(target_os = "linux"))]
fn resident_bytes(_mapping: &Mmap) -> Option<u64> {
    None
}

/// Bytes of an arbitrary mapped byte range currently in RAM.
///
/// Takes a slice rather than a whole mapping so a caller can ask about *part*
/// of one — `engine::expert_store` asks per expert, which is how the page
/// cache's own hit rate becomes measurable at the granularity a residency
/// policy would work at.
///
/// Walked in bounded chunks because the result vector is one byte per page:
/// asking about a 434 GB model in one call would allocate ~106 MB just to
/// count, which is a strange amount of memory to spend measuring memory.
///
/// `mincore` requires a page-aligned start, and a tensor's byte range is not
/// aligned to anything. The range is rounded **outward** to whole pages and
/// the result clamped back to the range's own length: a partial first or last
/// page is genuinely resident-or-not as a unit, and the alternative — rounding
/// inward — would report a range smaller than one page as never resident.
#[cfg(target_os = "linux")]
pub fn resident_bytes_of(range: &[u8]) -> Option<u64> {
    let page = page_size()?;
    if range.is_empty() {
        return Some(0);
    }
    let start = range.as_ptr() as usize;
    let aligned_start = start - start % page;
    let aligned_len = (start - aligned_start) + range.len();
    // 1 GiB per call — 256 KiB of result vector at a 4 KiB page.
    let chunk_pages = (1 << 30) / page;
    let mut vec = vec![0u8; chunk_pages.min(aligned_len.div_ceil(page))];
    let mut resident_pages = 0u64;
    let mut offset = 0usize;
    while offset < aligned_len {
        let this = (aligned_len - offset).min(chunk_pages * page);
        let pages = this.div_ceil(page);
        // Safety: `aligned_start + offset` is page-aligned, the range lies
        // inside a live mapping the caller holds across the call, and `vec`
        // has room for one byte per page of it as `mincore` requires.
        let rc = unsafe {
            libc::mincore(
                (aligned_start + offset) as *mut libc::c_void,
                this,
                vec.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return None;
        }
        resident_pages += vec[..pages].iter().filter(|b| *b & 1 == 1).count() as u64;
        offset += this;
    }
    Some((resident_pages * page as u64).min(range.len() as u64))
}

#[cfg(not(target_os = "linux"))]
pub fn resident_bytes_of(_range: &[u8]) -> Option<u64> {
    None
}

/// `MADV_DONTNEED` over a whole mapping: drops this process's page-table
/// references so the pages become evictable. On a read-only `MAP_PRIVATE`
/// file mapping this cannot lose data — a later read faults the same bytes
/// back from the file.
#[cfg(target_os = "linux")]
fn unmap_pages(mapping: &Mmap) {
    let Some(page) = page_size() else {
        return;
    };
    let base = mapping.as_ptr() as usize;
    // Rounded inward, as `madvise` works in whole pages and a mapping's tail
    // page is shared with nothing this could reach.
    let end = (base + mapping.len()) / page * page;
    if end <= base {
        return;
    }
    // Safety: the range is inside a live mapping this crate created read-only
    // and holds an `Arc` to across the call; base is page-aligned and the
    // length is a whole number of pages.
    unsafe {
        libc::madvise(base as *mut libc::c_void, end - base, libc::MADV_DONTNEED);
    }
}

/// Starts reading every registered shard into the page cache in the
/// background, when they all fit in half the memory the kernel says is
/// available — so weights a server reads through their mappings on its
/// first requests (a picture model's encoder and transformer, uploaded
/// lazily; a model on the CPU, or a split's host layers) come from RAM
/// rather than from a slow disk if any idle time passes first. Returns the bytes it
/// will read, or `None` when they do not fit (reading ahead a model larger
/// than RAM only evicts what is already there).
///
/// A thread reads each file through, sequentially, rather than asking the
/// kernel (`MADV_WILLNEED`): on a compressed btrfs volume the advice
/// returned at once having read 4 MiB of a 3.9 GiB file. The thread runs at
/// the idle I/O priority, so a request's own page faults go first.
pub fn prefetch_registered() -> Option<u64> {
    // `ORANGU_PREFETCH=0` turns it off (an A/B, a fast disk, a machine whose
    // memory is better left to something else).
    if crate::engine::backend::tuning_profile::env_var("ORANGU_PREFETCH")
        .is_ok_and(|v| v.trim() == "0")
    {
        return None;
    }
    // Once a process: a second call would only start a second reader.
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    // One read a file: a picture server maps its transformer and its text
    // encoder more than once (the calibration's open, the vision path's).
    let mut seen = std::collections::HashSet::new();
    let paths: Vec<(PathBuf, u64)> = SHARDS
        .lock()
        .ok()?
        .iter()
        .filter(|shard| seen.insert(shard.path.clone()))
        .map(|shard| (shard.path.clone(), shard.mmap.len() as u64))
        .collect();
    let total: u64 = paths.iter().map(|(_, len)| len).sum();
    let available = available_memory_bytes()?;
    if total == 0 || total > available / 2 {
        return None;
    }
    if STARTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return None;
    }
    std::thread::Builder::new()
        .name("orangu-prefetch".to_string())
        .spawn(move || {
            set_idle_io_priority();
            let mut buffer = vec![0u8; 8 << 20];
            for (path, _) in &paths {
                let Ok(mut file) = std::fs::File::open(path) else {
                    continue;
                };
                while matches!(std::io::Read::read(&mut file, &mut buffer), Ok(n) if n > 0) {}
            }
        })
        .ok()?;
    Some(total)
}

/// Puts the calling thread in the idle I/O scheduling class: its reads are
/// served only when nothing else wants the disk.
#[cfg(target_os = "linux")]
fn set_idle_io_priority() {
    const IOPRIO_WHO_PROCESS: libc::c_long = 1;
    const IOPRIO_CLASS_IDLE: libc::c_long = 3;
    const IOPRIO_CLASS_SHIFT: libc::c_long = 13;
    // Safety: `ioprio_set` on the calling thread (`who` 0) with a valid class
    // only changes this thread's I/O scheduling.
    unsafe {
        libc::syscall(
            libc::SYS_ioprio_set,
            IOPRIO_WHO_PROCESS,
            0 as libc::c_long,
            IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn set_idle_io_priority() {}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
fn available_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = meminfo
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib * 1024)
}

/// `MADV_WILLNEED` over a mapped byte range: asks the kernel to start reading
/// those pages in, and returns without waiting.
///
/// The asynchronous half of `MADV_DONTNEED`, and the only prefetch available
/// without an I/O thread of our own. Advisory in both directions — the kernel
/// may ignore it, and a range already resident costs nothing.
#[cfg(target_os = "linux")]
pub fn advise_willneed(range: &[u8]) {
    let Some(page) = page_size() else {
        return;
    };
    if range.is_empty() {
        return;
    }
    let start = range.as_ptr() as usize;
    let aligned_start = start - start % page;
    let len = (start - aligned_start) + range.len();
    // Safety: the range lies inside a live mapping the caller holds across
    // the call; `aligned_start` is page-aligned as `madvise` requires, and
    // `MADV_WILLNEED` neither writes nor invalidates anything.
    unsafe {
        libc::madvise(aligned_start as *mut libc::c_void, len, libc::MADV_WILLNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_willneed(_range: &[u8]) {}

/// `POSIX_FADV_DONTNEED` over a whole file — the second half of the drop,
/// which only bites once [`unmap_pages`] has made the pages unmapped.
#[cfg(target_os = "linux")]
fn advise_file_dontneed(path: &Path) {
    use std::os::fd::AsRawFd;
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    // Safety: `file` is open for the duration of the call; a length of 0 means
    // "to the end of the file" for `posix_fadvise`.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(target_os = "linux")]
fn page_size() -> Option<usize> {
    // Safety: `sysconf` is a pure query with no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (page > 0).then_some(page as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temporary file of `bytes`, mapped — for the tests of the Linux
    /// paths, which are the only ones that read a mapping.
    #[cfg(target_os = "linux")]
    fn mapped(bytes: &[u8]) -> (tempfile::NamedTempFile, Arc<Mmap>) {
        use std::io::Write as _;
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(bytes).expect("write");
        file.flush().expect("flush");
        // Safety: the file is this test's own and nothing truncates it.
        let mmap = Arc::new(unsafe { Mmap::map(file.as_file()) }.expect("mmap"));
        (file, mmap)
    }

    /// The figure a benchmark reads to decide whether its run was cold. A
    /// file just written and mapped is in the page cache, so this must report
    /// close to all of it — a measurement that reported zero here would call
    /// every warm run cold.
    /// The prefetch asks for nothing when nothing is registered, and reads
    /// the kernel's available memory where it can.
    #[test]
    fn the_prefetch_needs_shards_and_reads_available_memory() {
        if SHARDS.lock().map(|s| s.is_empty()).unwrap_or(false) {
            assert_eq!(prefetch_registered(), None);
        }
        #[cfg(target_os = "linux")]
        assert!(available_memory_bytes().is_some_and(|b| b > 0));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_freshly_written_mapping_reads_as_resident() {
        let (_file, mmap) = mapped(&vec![7u8; 512 * 1024]);
        // Touch it, so residency is not merely likely.
        assert_eq!(
            mmap.iter().map(|b| u64::from(*b)).sum::<u64>(),
            7 * 512 * 1024
        );
        let resident = resident_bytes(&mmap).expect("mincore works on Linux");
        assert!(
            resident >= 256 * 1024,
            "expected most of 512 KiB resident, got {resident}"
        );
    }

    /// Residency can never exceed the shard, however the final partial page
    /// is counted — a total above 100% would make any "how cold was it"
    /// ratio nonsense.
    #[test]
    #[cfg(target_os = "linux")]
    fn residency_never_exceeds_the_mapping() {
        // Deliberately not a page multiple: the tail page is partial.
        let len = 4096 * 3 + 17;
        let (_file, mmap) = mapped(&vec![1u8; len]);
        let resident = resident_bytes(&mmap).expect("mincore works on Linux");
        assert!(resident <= len as u64, "{resident} > {len}");
    }

    /// A partial total would understate residency and make a warm run look
    /// cold, so an unmeasurable shard must poison the total rather than
    /// contribute zero.
    #[test]
    fn one_unmeasurable_shard_makes_the_total_unknown() {
        let shards = vec![
            ShardResidency {
                path: "a".into(),
                bytes: 100,
                resident_bytes: Some(60),
            },
            ShardResidency {
                path: "b".into(),
                bytes: 100,
                resident_bytes: None,
            },
        ];
        assert_eq!(residency_totals(&shards), (200, None));
    }

    #[test]
    fn totals_sum_when_every_shard_is_measurable() {
        let shards = vec![
            ShardResidency {
                path: "a".into(),
                bytes: 100,
                resident_bytes: Some(60),
            },
            ShardResidency {
                path: "b".into(),
                bytes: 50,
                resident_bytes: Some(50),
            },
        ];
        assert_eq!(residency_totals(&shards), (150, Some(110)));
    }

    /// Off Linux the report must say it did nothing, not report a successful
    /// drop that never happened.
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn dropping_is_refused_rather_than_faked() {
        assert!(!drop_model_page_cache().supported);
    }

    /// The registry is what makes a shard reachable at all; a model that
    /// registered nothing must report nothing rather than an empty success.
    #[test]
    fn an_unregistered_model_has_no_residency_to_report() {
        // Deliberately reads the real registry: in a test binary that loads
        // no model it is empty, which is the case being asserted.
        let (bytes, resident) = residency_totals(&residency());
        if bytes == 0 {
            assert_eq!(
                resident,
                Some(0),
                "an empty registry totals to zero, not None"
            );
        }
    }
}

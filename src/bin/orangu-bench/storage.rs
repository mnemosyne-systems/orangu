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

//! `--storage-probe`: how fast this machine's storage reads, against request
//! size.
//!
//! # Why the engine needs this and a stopwatch will not do
//!
//! `[orangu-server].read_size` exists because throughput on real hardware is
//! a **step function of request size**, not a constant. On the drive this was
//! developed against, 512 KiB reads sustain 27.9 MB/s and 1 MiB reads sustain
//! 205.8 — an eight-fold cliff at the controller's `max_sectors_kb`. A second
//! drive in the same machine has no cliff at all and is still improving at
//! 4 MiB. Neither shape is derivable from `sysfs`: `max_sectors_kb` predicts
//! the first and is wrong about the second.
//!
//! So the granule has to be measured per machine, and until now that meant an
//! `fio` command line and a hand-built history file. This is that measurement
//! as a bench mode, so the curve is one command and lands in `--history`
//! beside everything else.
//!
//! # `O_DIRECT`, because the page cache is the thing being avoided
//!
//! A buffered read of a file the kernel already has cached measures memcpy,
//! not storage — and on the second repetition of anything, that is exactly
//! what it would measure. `O_DIRECT` bypasses the cache in both directions:
//! the read does not consult it and does not populate it, so the probe
//! neither lies nor evicts the model a subsequent benchmark is about to want.
//!
//! It comes with alignment rules that are easy to get wrong and fail at
//! runtime rather than compile time: the buffer address, the file offset and
//! the length must all be multiples of the device's logical block size.
//! [`AlignedBuffer`] owns that discipline in one place.
//!
//! # Both directions, because this drive degrades under load
//!
//! **The single most expensive mistake available here is sweeping sizes in
//! one order.** This machine's drive runs a plain sequential read at
//! 207 MB/s for twenty-two seconds and 30.7 MB/s from forty-four seconds on —
//! so in an ascending sweep every large request is measured on a drive the
//! small ones already tired out, and the cliff appears to be smaller than it
//! is. The same hazard in the throughput sweeps cost this project four
//! measurements and reversed the sign of one: a fixed-order A/B put a +0.11%
//! change at ±10%, and a real +3.9% one at −3.1%.
//!
//! [`sweep`] therefore measures every size **twice, once ascending and once
//! descending**, and reports the mean of the pair. Each size then holds one
//! early slot and one late one, so time-under-load lands on every point
//! equally instead of on the tail. The two halves are reported separately as
//! well: when they disagree, the drive changed state mid-probe and the mean
//! is the least interesting number on the page.

use std::io;
use std::path::Path;
use std::time::Instant;

/// One request size's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Request size in KiB — the x-axis, and the `n` of the history record.
    pub kib: u32,
    /// Ascending-pass rate, MB/s (10^6 bytes, the unit drive datasheets and
    /// `fio` both use — not MiB/s, which would understate every figure here
    /// by 4.9% against the numbers they are compared with).
    pub up: f64,
    /// Descending-pass rate, MB/s.
    pub down: f64,
}

impl Point {
    /// The figure to report: the mean of the two passes.
    pub fn mean(self) -> f64 {
        (self.up + self.down) / 2.0
    }

    /// How far apart the two passes landed, as a fraction of the mean.
    ///
    /// This is the honesty check. A drive in a steady state gives the same
    /// rate in both passes and this is near zero; one that degrades under
    /// load gives a large spread, and then the mean describes a machine
    /// state rather than a request size.
    pub fn spread(self) -> f64 {
        let mean = self.mean();
        if mean <= 0.0 {
            return 0.0;
        }
        (self.up - self.down).abs() / mean
    }
}

/// A page-aligned heap buffer, which `O_DIRECT` requires and `Vec<u8>` does
/// not guarantee.
///
/// Allocated through [`std::alloc`] rather than by over-allocating a `Vec`
/// and slicing to an aligned offset: the latter works and is what most
/// examples do, but it silently gives back a buffer whose *length* is no
/// longer a multiple of the block size, which is the other half of the same
/// rule and fails as `EINVAL` at read time.
struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    /// `len` is rounded up to a whole number of `align`-sized blocks.
    fn new(len: usize, align: usize) -> io::Result<Self> {
        let len = len.div_ceil(align) * align;
        let layout = std::alloc::Layout::from_size_align(len, align)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        // Safety: `layout` has a non-zero size — `len` is rounded up to at
        // least one block — and the pointer is checked for null below.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(io::Error::other(
                "could not allocate an aligned read buffer",
            ));
        }
        Ok(Self { ptr, len, layout })
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // Safety: `ptr` came from `alloc` with exactly this layout and is
        // freed once, here.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// The alignment `O_DIRECT` needs. 4096 covers every logical block size in
/// practice (512 and 4096 are the only ones that exist), and over-aligning is
/// always safe where under-aligning is `EINVAL`.
const DIRECT_ALIGN: usize = 4096;

/// Read `span` bytes of `path` sequentially in `request`-sized reads with
/// `O_DIRECT`, returning MB/s.
///
/// `skip` bytes are read and discarded first — the ramp. Without it the first
/// request of each point pays the drive's wake-up and the smallest sizes,
/// which do the most requests, are charged for it most.
#[cfg(target_os = "linux")]
fn read_at_size(path: &Path, request: usize, span: u64, skip: u64) -> io::Result<f64> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?;
    let fd = file.as_raw_fd();
    let buf = AlignedBuffer::new(request, DIRECT_ALIGN)?;

    let mut offset: i64 = 0;
    let read_one = |offset: &mut i64| -> io::Result<usize> {
        // Safety: `buf.ptr` is `buf.len` bytes of live, `DIRECT_ALIGN`-aligned
        // memory this call owns exclusively, and `fd` is open for reading.
        let got = unsafe { libc::pread(fd, buf.ptr.cast(), buf.len, *offset) };
        if got < 0 {
            return Err(io::Error::last_os_error());
        }
        *offset += got as i64;
        Ok(got as usize)
    };

    let mut ramped = 0u64;
    while ramped < skip {
        match read_one(&mut offset)? {
            0 => break,
            n => ramped += n as u64,
        }
    }

    let started = Instant::now();
    let mut read = 0u64;
    while read < span {
        match read_one(&mut offset)? {
            // The file ran out. Wrap rather than stop short, so every request
            // size is timed over the same number of bytes and the rates are
            // comparable — which is the entire point of the sweep.
            0 => offset = 0,
            n => read += n as u64,
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    if seconds <= 0.0 {
        return Err(io::Error::other(
            "the probe completed in no measurable time",
        ));
    }
    Ok(read as f64 / seconds / 1e6)
}

#[cfg(not(target_os = "linux"))]
fn read_at_size(_path: &Path, _request: usize, _span: u64, _skip: u64) -> io::Result<f64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "the storage probe needs O_DIRECT, which is a Linux interface",
    ))
}

/// Sweep `sizes` (KiB) against `path`, ascending then descending.
///
/// The descending pass is not a repetition for its own sake — see the module
/// docs. It is what stops time-under-load from being read as a property of
/// the largest request size.
pub fn sweep(path: &Path, sizes: &[u32], span: u64, skip: u64) -> io::Result<Vec<Point>> {
    let mut ascending = Vec::with_capacity(sizes.len());
    for &kib in sizes {
        ascending.push(read_at_size(path, kib as usize * 1024, span, skip)?);
    }
    let mut descending = vec![0.0; sizes.len()];
    for (i, &kib) in sizes.iter().enumerate().rev() {
        descending[i] = read_at_size(path, kib as usize * 1024, span, skip)?;
    }
    Ok(sizes
        .iter()
        .zip(ascending)
        .zip(descending)
        .map(|((&kib, up), down)| Point { kib, up, down })
        .collect())
}

/// The probe's table, as the report prints it.
pub fn table(points: &[Point]) -> String {
    let mut out = String::from(
        "  request |     MB/s |       up |     down |  spread\n\
         ----------------------------------------------------\n",
    );
    for p in points {
        out.push_str(&format!(
            "  {:>5} K | {:8.1} | {:8.1} | {:8.1} | {:6.1}%\n",
            p.kib,
            p.mean(),
            p.up,
            p.down,
            100.0 * p.spread(),
        ));
    }
    // Only worth saying when it happened, and worth saying loudly then: a
    // wide spread means the two passes disagree, so the curve describes the
    // drive's state over time rather than its response to request size.
    if let Some(worst) = points
        .iter()
        .max_by(|a, b| a.spread().total_cmp(&b.spread()))
        && worst.spread() > 0.25
    {
        out.push_str(&format!(
            "  the {} K point differs by {:.0}% between the ascending and descending passes —\n  \
             this drive changed state during the probe, so read the two columns, not the mean\n",
            worst.kib,
            100.0 * worst.spread(),
        ));
    }
    out
}

/// The probe's history rows: one per request size, `n` in KiB.
///
/// `mode` is its own panel rather than sharing one with a tok/s series: this
/// is MB/s against request size, which is a different measurement in
/// different units on a different x-axis, and overlaying it on a throughput
/// chart would put two unrelated curves in one frame.
pub fn records(points: &[Point], label: &str) -> Vec<super::history::Record> {
    let date = super::history::today();
    points
        .iter()
        .map(|p| super::history::Record {
            date: date.clone(),
            label: label.to_string(),
            mode: "storage_mb_s".to_string(),
            n: p.kib,
            // `best` is the faster of the two passes and `mean` their mean, so
            // the pair carries the disagreement rather than hiding it — the
            // same shape the throughput modes use.
            best: p.up.max(p.down),
            mean: p.mean(),
            sd: (p.up - p.down).abs() / 2.0,
            sd_sample: None,
            device: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(kib: u32, up: f64, down: f64) -> Point {
        Point { kib, up, down }
    }

    #[test]
    fn the_reported_rate_is_the_mean_of_both_passes() {
        let p = point(1024, 200.0, 100.0);
        assert!((p.mean() - 150.0).abs() < 1e-9);
    }

    /// The number that says whether the mean means anything.
    #[test]
    fn spread_is_zero_on_a_steady_drive_and_large_on_one_that_degrades() {
        assert!(point(1024, 200.0, 200.0).spread() < 1e-9);
        let degraded = point(1024, 200.0, 100.0);
        assert!(
            (degraded.spread() - 2.0 / 3.0).abs() < 1e-9,
            "{}",
            degraded.spread()
        );
    }

    /// A drive that changed state mid-probe must say so in the table rather
    /// than present a mean that describes neither pass. This is the failure
    /// the two-pass design exists to expose, so it has to be visible.
    #[test]
    fn a_drive_that_changed_state_is_called_out_in_the_table() {
        let steady = table(&[point(512, 27.9, 28.1), point(1024, 205.0, 206.0)]);
        assert!(
            !steady.contains("changed state"),
            "a steady drive must not be warned about:\n{steady}"
        );

        let degraded = table(&[point(512, 27.9, 28.1), point(1024, 205.0, 31.0)]);
        assert!(
            degraded.contains("changed state"),
            "the warning is the whole point of two passes:\n{degraded}"
        );
        assert!(degraded.contains("1024 K"), "{degraded}");
    }

    /// The history rows carry the disagreement too, so a chart drawn from the
    /// file can show it without the table beside it.
    #[test]
    fn the_records_keep_both_passes_reachable() {
        let recs = records(&[point(1024, 200.0, 100.0)], "sdf1");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].mode, "storage_mb_s");
        assert_eq!(recs[0].n, 1024);
        assert!(
            (recs[0].best - 200.0).abs() < 1e-9,
            "best is the faster pass"
        );
        assert!((recs[0].mean - 150.0).abs() < 1e-9);
        assert!(
            (recs[0].sd - 50.0).abs() < 1e-9,
            "half the gap between passes"
        );
    }

    /// Every size must appear once, and the descending pass must line up with
    /// the size it measured — an off-by-one in the reversed loop would pair
    /// each size with a neighbour's rate and produce a plausible, wrong curve.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_descending_pass_is_matched_to_the_right_size() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&vec![0u8; 8 * 1024 * 1024]).expect("write");
        file.flush().expect("flush");

        let sizes = [4u32, 64, 1024];
        // O_DIRECT is refused on some filesystems (tmpfs among them), and a
        // test that silently passes because the probe never ran would be
        // worse than no test. Skip explicitly on that error alone.
        match sweep(file.path(), &sizes, 1 << 20, 0) {
            Ok(points) => {
                assert_eq!(points.len(), sizes.len());
                for (p, &kib) in points.iter().zip(sizes.iter()) {
                    assert_eq!(p.kib, kib);
                    assert!(p.up > 0.0 && p.down > 0.0, "{p:?}");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {}
            Err(e) => panic!("the probe failed for a reason other than O_DIRECT support: {e}"),
        }
    }
}

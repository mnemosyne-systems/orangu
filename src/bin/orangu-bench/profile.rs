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

//! CPU profile capture for the server under measurement.
//!
//! A throughput number says *how* slow something is; it never says *where* the
//! time goes. The two questions are always asked together during performance
//! work, and until now they were answered by two different procedures — this
//! tool for the rate, a hand-assembled `perf record` pipeline for the profile —
//! which meant the profile was routinely taken over a *different* workload than
//! the number it was supposed to explain: a different prompt length, a warmup
//! run included, or a window that started before the server was busy.
//!
//! [`Recorder`] closes that gap. It brackets exactly the measured window — it
//! starts after warmup and stops when the last repetition finishes — so the
//! flamegraph and the tok/s on the line above it describe the same seconds of
//! the same process. The same applies to `llama-server`: it is profiled through
//! this same path, over the same prompt, at the same clock, so the two profiles
//! are comparable as well as the two rates.
//!
//! `perf` is the only external program on this path. It has to be: it reads
//! the kernel's perf events, and nothing in userspace can stand in for it.
//! Everything downstream — collapsing `perf script` output into folded stacks,
//! folding recursion, and rendering the SVG — is [`super::flamegraph`], in
//! process:
//!
//! ```text
//! perf record -F <freq> -g --call-graph <mode> -p <pid>
//! perf script | flamegraph::collapse > FILE.folded
//! flamegraph::render                 > FILE.svg
//! ```
//!
//! **Frame pointers.** `--call-graph fp` needs them. A stock release build of
//! `orangu-server` drops them and the call chain is lost for most samples in
//! the hot leaf, which renders as a flamegraph of a process doing nothing:
//! build the profiled server with
//! `RUSTFLAGS="-C force-frame-pointers=yes"`. A binary you do not control —
//! `llama-server` from a distribution package — needs `--call-graph dwarf`
//! instead, which is why the mode is an option rather than a constant.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::flamegraph;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

/// Everything a capture needs. Built by the caller from the command line.
pub struct Options {
    /// Where the rendered flamegraph goes. `.folded` and `.png` siblings are
    /// derived from it.
    pub svg: PathBuf,
    /// Process to sample.
    pub pid: u32,
    /// `perf record -F`.
    pub freq: u32,
    /// `perf record --call-graph`: `fp` or `dwarf`.
    pub call_graph: String,
    /// Also render a PNG beside the SVG.
    pub png: bool,
    /// Title drawn on the flamegraph.
    pub title: String,
}

/// A running `perf record`. Dropping one without calling [`Recorder::finish`]
/// leaves the child running, so the caller must always finish it — including on
/// the error path, which is why `finish` takes `self` and reports rather than
/// panics.
pub struct Recorder {
    perf: Child,
    data: PathBuf,
    stderr_log: PathBuf,
    opts: Options,
    started: Instant,
    /// The target's own CPU time when sampling began, from
    /// `/proc/<pid>/stat`. The check it enables is in [`Recorder::finish`].
    cpu_at_start: Option<f64>,
}

/// What the collapsed stacks say, once. Printed under the measurement it
/// belongs to.
pub struct Summary {
    pub svg: PathBuf,
    pub folded: PathBuf,
    pub png: Option<PathBuf>,
    pub samples: u64,
    pub seconds: f64,
    /// Mean number of the server's threads that were **on a CPU** during the
    /// window: `samples / (freq × seconds)`.
    ///
    /// This is the number that makes two engines' profiles comparable at all. A
    /// flamegraph is normalised to its own total, so it can only ever say how an
    /// engine divided the CPU time it used — never how much it used. And a
    /// thread blocked in the kernel produces no samples while a thread spinning
    /// on `_mm_pause` produces them at full rate, so "waiting" is invisible in
    /// one engine and dominant in the other purely as an artifact of *how* each
    /// waits. Cores-busy is immune to that: it counts occupancy, and it is what
    /// turns a share back into work.
    pub cores_busy: f64,
    /// Share of samples on a thread waiting for the GPU.
    pub gpu_wait: f64,
    /// Share of samples on a parked or work-stealing thread.
    pub pool_idle: f64,
    /// Cores the target actually used over the window, from `/proc`, or `None`
    /// where that could not be read. Compared against `cores_busy` to catch a
    /// profile that missed most of the process — see [`Recorder::finish`].
    pub cores_from_proc: Option<f64>,
    /// Self-time share per bucket, largest first.
    pub buckets: Vec<(&'static str, f64)>,
    /// The heaviest individual leaf frames, largest first: `(frame, share)`.
    pub leaves: Vec<(String, f64)>,
}

impl Recorder {
    /// Start sampling `opts.pid`. Fails *before* any measurement runs if the
    /// tooling is missing or `perf` cannot attach — a benchmark that silently
    /// produced no profile after a twenty-minute sweep would be worse than one
    /// that refused to start.
    pub fn start(opts: Options) -> anyhow::Result<Recorder> {
        if let Some(parent) = opts.svg.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let data = sibling(&opts.svg, "perf.data");
        let stderr_log = sibling(&opts.svg, "perf.log");
        let log = std::fs::File::create(&stderr_log)?;

        let perf = Command::new("perf")
            .args(["record", "-F"])
            .arg(opts.freq.to_string())
            .arg("-g")
            .arg("--call-graph")
            .arg(&opts.call_graph)
            .arg("-p")
            .arg(opts.pid.to_string())
            .arg("-o")
            .arg(&data)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|e| anyhow::anyhow!("could not run `perf record`: {e}"))?;

        let cpu_at_start = proc_cpu_seconds(opts.pid);
        let mut rec = Recorder {
            perf,
            data,
            stderr_log,
            opts,
            started: Instant::now(),
            cpu_at_start,
        };

        // `perf` fails asynchronously — a bad pid or a paranoid setting is
        // reported on stderr after the spawn has already succeeded. Give it a
        // moment and check it is still alive, so the failure surfaces here
        // rather than as an empty profile later.
        std::thread::sleep(std::time::Duration::from_millis(400));
        if let Ok(Some(status)) = rec.perf.try_wait() {
            let why = std::fs::read_to_string(&rec.stderr_log).unwrap_or_default();
            anyhow::bail!(
                "`perf record` exited immediately ({status}): {}",
                why.trim().replace('\n', " ")
            );
        }
        Ok(rec)
    }

    /// Stop sampling and render. Returns what the profile says.
    pub fn finish(mut self) -> anyhow::Result<Summary> {
        let seconds = self.started.elapsed().as_secs_f64();
        // Read before `perf` is asked to stop, so the window matches.
        let cores_from_proc = self
            .cpu_at_start
            .zip(proc_cpu_seconds(self.opts.pid))
            .filter(|_| seconds > 0.0)
            .map(|(before, after)| (after - before).max(0.0) / seconds);

        // SIGINT, not SIGKILL: `perf record` writes its data file while
        // shutting down, and a killed one leaves an unreadable stub.
        let pid = self.perf.id().to_string();
        let _ = Command::new("kill").args(["-INT", &pid]).status();
        let status = self.perf.wait()?;
        if !status.success() && !self.data.exists() {
            let why = std::fs::read_to_string(&self.stderr_log).unwrap_or_default();
            anyhow::bail!("`perf record` failed: {}", why.trim().replace('\n', " "));
        }

        let folded_path = sibling(&self.opts.svg, "folded");
        let folded = collapse(&self.data)?;
        if folded.trim().is_empty() {
            anyhow::bail!(
                "no samples collapsed — the profiled process may have been idle, \
                 or `--call-graph {}` produced no usable stacks",
                self.opts.call_graph
            );
        }
        std::fs::write(&folded_path, &folded)?;

        let svg = render(&folded, &self.opts, seconds, attribution_samples(&folded))?;
        let png = if self.opts.png {
            render_png(&svg)?
        } else {
            None
        };

        // The raw `perf.data` is the largest artifact by an order of magnitude
        // and nothing downstream reads it; the collapsed file is the one worth
        // keeping (it diffs, and it re-renders without a re-run).
        let _ = std::fs::remove_file(&self.data);
        let _ = std::fs::remove_file(&self.stderr_log);

        let attribution = summarize(&folded);
        let cores_busy = attribution.samples as f64 / (f64::from(self.opts.freq) * seconds);

        // The one check that catches a profile which missed most of the
        // process. `perf record -p` attaches to the threads that exist *at
        // that moment* and does not pick up ones created later — so profiling
        // a server that builds its compute threads lazily, on its first
        // request, samples almost nothing while still producing a perfectly
        // well-formed flamegraph and a confident `cores_busy`. Measured here:
        // 0.02 cores reported against 0.44 actually used, a 20x understatement
        // that looked like "decode is not CPU-bound".
        //
        // `/proc` cannot miss a thread, so disagreeing with it by this much
        // means the samples are not describing the process. Warn rather than
        // fail: a profile can legitimately undercount a little (perf's own
        // startup, threads that exit early), and the numbers are still printed
        // so the reader can judge.
        if let Some(actual) = cores_from_proc
            && actual > 0.05
            && cores_busy < actual * 0.5
        {
            eprintln!(
                "  WARNING: the profile accounts for {cores_busy:.2} cores but /proc says the \n\
                 \x20          process used {actual:.2}. `perf record -p` does not follow threads \n\
                 \x20          created after it attached — run a warmup before profiling so the \n\
                 \x20          workload's threads already exist. This profile is not trustworthy."
            );
        }

        // A sidecar rather than a header line inside the collapsed file:
        // `flamegraph.pl` parses every line of its input as `stack count`, and
        // a file that renders is worth more than one that carries its own
        // metadata.
        let meta = sibling(&self.opts.svg, "meta.json");
        let _ = std::fs::write(
            &meta,
            serde_json::to_vec_pretty(&serde_json::json!({
                "title": self.opts.title,
                "pid": self.opts.pid,
                "freq_hz": self.opts.freq,
                "call_graph": self.opts.call_graph,
                "seconds": seconds,
                "samples": attribution.samples,
                "cores_busy": cores_busy,
                "gpu_wait_pct": attribution.gpu_wait,
                "pool_idle_pct": attribution.pool_idle,
                "cores_working": cores_busy
                    * (1.0 - (attribution.gpu_wait + attribution.pool_idle) / 100.0),
                // The kernel's own accounting for the same window, so a stored
                // profile carries the evidence for whether to trust itself.
                "cores_from_proc": cores_from_proc,
            }))
            .unwrap_or_default(),
        );

        Ok(Summary {
            svg,
            folded: folded_path,
            png,
            samples: attribution.samples,
            seconds,
            cores_busy,
            cores_from_proc,
            gpu_wait: attribution.gpu_wait,
            pool_idle: attribution.pool_idle,
            buckets: attribution.buckets,
            leaves: attribution.leaves,
        })
    }
}

/// One already-collapsed profile, read back off disk.
pub struct Profile {
    pub name: String,
    pub samples: u64,
    /// From the `.meta.json` written beside the collapsed file, when it is
    /// still there. A `.folded` carried off the machine on its own keeps every
    /// share it ever had; only the occupancy needs the sidecar.
    pub cores_busy: Option<f64>,
    pub gpu_wait: f64,
    pub pool_idle: f64,
    pub buckets: Vec<(&'static str, f64)>,
    pub leaves: Vec<(String, f64)>,
}

/// Re-read collapsed profiles and report what they say — the `--chart-only`
/// of profiling: no server, no measurement, no re-run.
///
/// Two profiles of the same workload on two engines is the shape this exists
/// for. A flamegraph answers "where does *this* engine spend its time"; putting
/// two side by side answers the question a comparison is actually run to
/// settle, which is where the *difference* is. Reading it off two SVGs by eye
/// does not work — they are normalised to their own totals, so the wider
/// plateau belongs to whichever engine had fewer samples.
pub fn read_profiles(paths: &[PathBuf]) -> anyhow::Result<Vec<Profile>> {
    let mut out = Vec::new();
    for path in paths {
        let text = read_maybe_gzipped(path)?;
        let a = summarize(&text);
        if a.samples == 0 {
            anyhow::bail!("{} has no collapsed samples", path.display());
        }
        // `a.folded.gz` has stem `a.folded`, so strip the extension twice to
        // land on the same `a.meta.json` the uncompressed file resolves to.
        let stem_base = if path.extension().is_some_and(|e| e == "gz") {
            path.with_extension("")
        } else {
            path.to_path_buf()
        };
        let cores_busy = std::fs::read_to_string(sibling(&stem_base, "meta.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("cores_busy").and_then(serde_json::Value::as_f64));
        out.push(Profile {
            name: stem_base
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            samples: a.samples,
            cores_busy,
            gpu_wait: a.gpu_wait,
            pool_idle: a.pool_idle,
            buckets: a.buckets,
            leaves: a.leaves,
        });
    }
    Ok(out)
}

/// A collapsed profile, gzipped or not.
///
/// Collapsed stacks compress by roughly 40× — they are long, highly repetitive
/// Rust and C++ symbol names — so a set worth keeping alongside a document is
/// worth keeping compressed. Decompression is delegated to `gzip` rather than
/// linked in: this is one call in a developer tool, on a platform that has the
/// binary, and it is not worth a dependency in the served product's manifest.
fn read_maybe_gzipped(path: &Path) -> anyhow::Result<String> {
    if path.extension().is_some_and(|e| e == "gz") {
        let out = Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .output()
            .map_err(|e| anyhow::anyhow!("could not run gzip for {}: {e}", path.display()))?;
        if !out.status.success() {
            anyhow::bail!("gzip could not read {}", path.display());
        }
        return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
    }
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.display()))
}

/// Every bucket named by any of `profiles`, ordered by their combined share, so
/// a table built from it has one row per bucket and no gaps.
pub fn bucket_union(profiles: &[Profile]) -> Vec<&'static str> {
    let mut totals: HashMap<&'static str, f64> = HashMap::new();
    for p in profiles {
        for (bucket, pct) in &p.buckets {
            *totals.entry(bucket).or_default() += pct;
        }
    }
    let mut names: Vec<&'static str> = totals.keys().copied().collect();
    names.sort_by(|a, b| totals[b].total_cmp(&totals[a]).then(a.cmp(b)));
    names
}

/// `dir/stem.ext`, where `stem` is the SVG's file stem — so one `--flamegraph
/// out.svg` names `out.folded`, `out.png` and the transient `out.perf.data` as
/// one set.
/// CPU seconds (user + system, **all threads**) the process has consumed, from
/// `/proc/<pid>/stat`.
///
/// The kernel's own accounting, which no sampling artefact can distort — which
/// is exactly why it is worth reading twice and comparing against what the
/// samples imply.
///
/// The `comm` field is an arbitrary string in parentheses and may itself
/// contain spaces or `)`, so the fields are counted from after the **last**
/// `)`, never by splitting the whole line.
pub fn proc_cpu_seconds(pid: u32) -> Option<f64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After `comm`, field 0 is `state`; `utime`/`stime` are stat fields 14/15,
    // which land at indices 11 and 12 here.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    let hz = clock_ticks_per_second();
    Some((utime + stime) as f64 / hz)
}

/// `sysconf(_SC_CLK_TCK)`, the unit `/proc/<pid>/stat` reports CPU time in.
/// Effectively always 100 on Linux; read rather than assumed, and falling back
/// to 100 if the call is unavailable.
fn clock_ticks_per_second() -> f64 {
    // SAFETY: `sysconf` takes an int and returns a long; no pointers involved.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if hz > 0 { hz as f64 } else { 100.0 }
}

fn sibling(svg: &Path, ext: &str) -> PathBuf {
    let stem = svg
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "flamegraph".to_string());
    svg.with_file_name(format!("{stem}.{ext}"))
}

/// `perf script | flamegraph::collapse`, streamed rather than buffered: a
/// decode profile is a few hundred megabytes of `perf script` text before it
/// collapses, and only the collapsed form is ever needed.
fn collapse(data: &Path) -> anyhow::Result<String> {
    let script = Command::new("perf")
        .args(["script", "--no-inline", "-i"])
        .arg(data)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| anyhow::anyhow!("could not run `perf script`: {e}"))?;
    if !script.status.success() {
        anyhow::bail!("`perf script` failed on {}", data.display());
    }
    let totals = flamegraph::collapse(&String::from_utf8_lossy(&script.stdout));
    Ok(flamegraph::to_folded(&totals))
}

/// Total samples in a collapsed profile — the subtitle's denominator.
fn attribution_samples(folded: &str) -> u64 {
    folded
        .lines()
        .filter_map(|l| l.rsplit_once(' '))
        .filter_map(|(_, c)| c.trim().parse::<u64>().ok())
        .sum()
}

/// Write the SVG.
fn render(folded: &str, opts: &Options, seconds: f64, samples: u64) -> anyhow::Result<PathBuf> {
    let subtitle = format!(
        "{samples} samples over {seconds:.0}s at {} Hz, pid {} (--call-graph {})",
        opts.freq, opts.pid, opts.call_graph
    );
    std::fs::write(
        &opts.svg,
        flamegraph::render(folded, &opts.title, &subtitle),
    )?;
    Ok(opts.svg.clone())
}

/// A raster copy beside the SVG, for documents and diffs that cannot embed one.
/// Absence of a converter is reported, not fatal — the SVG is already written
/// and is the better artifact of the two.
///
/// Shared with the throughput chart (`--chart-png`): both artifacts want the
/// same "SVG is canonical, PNG is for embedding" treatment, and both are
/// useless if a missing rasterizer aborts the run that produced them.
pub fn render_png(svg: &Path) -> anyhow::Result<Option<PathBuf>> {
    let png = svg.with_extension("png");
    let status = Command::new("rsvg-convert")
        // Wide enough that frame labels survive rasterization; a flamegraph
        // rendered narrow is a coloured bar chart with no text.
        .args(["--width", "1800", "--keep-aspect-ratio", "-o"])
        .arg(&png)
        .arg(svg)
        .stderr(Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => Ok(Some(png)),
        _ => {
            eprintln!("orangu-bench: rsvg-convert unavailable — PNG not written");
            Ok(None)
        }
    }
}

/// Buckets a leaf frame is attributed to, tried in order. The first entry
/// whose pattern appears in the frame name wins.
///
/// Kernel-mode frames are separated first, on the `_[k]` suffix
/// `stackcollapse-perf.pl --kernel` adds, because that is the one part of the
/// attribution the profiler *knows* rather than guesses. The split that matters
/// for this project's comparisons then falls out cleanly: `radv_*` is the
/// userspace Vulkan driver and costs the calling thread directly, while
/// `amdgpu_*` / `ttm_*` are the kernel-mode driver reached through an ioctl and
/// are as much a measure of submission frequency as of work done.
///
/// Everything below the kernel line is a **heuristic over symbol names**, not a
/// module attribution: `perf` is not asked which object a frame came from, so a
/// name matching nothing here falls into `app/other`. That residual is the point
/// of the leaf table printed beside it — for `orangu-server` it is
/// overwhelmingly its own Rust code, but "overwhelmingly" is a claim the named
/// leaves let a reader check rather than take on trust.
const BUCKETS: &[(&str, &[&str])] = &[
    (
        "kernel:gpu",
        &["amdgpu", "ttm_", "drm_", "dma_fence", "dma_resv"],
    ),
    ("kernel", &[KERNEL_MARK]),
    (
        "radv/vulkan",
        &[
            "radv_",
            "vk_common_",
            "ac_nir",
            "aco_",
            "nir_",
            "libvulkan",
            "vulkan_radeon",
            "winsys",
        ],
    ),
    (
        "wgpu",
        &["wgpu", "naga", "ash::", "gpu_alloc", "gpu_descriptor"],
    ),
    (
        "ggml",
        &["ggml", "llama_", "llm_", "quantize_row", "dequantize_"],
    ),
    (
        "libc/alloc",
        &[
            "__memcpy",
            "__memset",
            "_int_malloc",
            "_int_free",
            "malloc",
            "free",
            "cfree",
            "operator new",
            "__libc_",
            "tcache",
        ],
    ),
];

/// The suffix `stackcollapse-perf.pl --kernel` puts on kernel-mode frames.
const KERNEL_MARK: &str = "_[k]";

/// Frames that mean a thread is waiting for the **GPU**, whether it does so by
/// spinning or by blocking.
///
/// This is what makes two engines' CPU costs comparable, and getting it wrong
/// inverts the answer. `llama-server` waits by spinning on `_mm_pause`, which
/// samples at full rate; `orangu-server` waits by blocking, which samples not at
/// all. Read naively, that makes the engine which wastes a whole core look busy
/// with useful work and the one which yields it look idle. Naming the wait
/// explicitly is the only way the two numbers mean the same thing.
const GPU_WAIT: &[&str] = &[
    "_mm_pause",
    "ggml_vk_wait_for_fence",
    "WaitForFences",
    "wait_for_fence",
    "vk_sync_wait",
    "SyncobjTimelineWait",
    "syncobj_wait",
    "wait_mapped",
    "WaitSemaphores",
    "device_poll",
];

/// Frames that mean a thread is parked, spinning for work, or otherwise not on
/// the model at all — a thread pool's own overhead. Distinct from [`GPU_WAIT`]
/// because the two have different remedies: one is answered by giving the GPU
/// less to wait for, the other by not waking as many threads.
const POOL_IDLE: &[&str] = &[
    "wait_until_out_of_work",
    "no_work_found",
    "rayon_core::latch",
    "rayon_core::sleep",
    "park",
    "futex_wait",
    "epoll_wait",
    "sched_yield",
    "__lll_lock_wait",
    "cond_wait",
    "nanosleep",
];

/// What [`summarize`] reads out of the collapsed stacks.
struct Attribution {
    samples: u64,
    buckets: Vec<(&'static str, f64)>,
    leaves: Vec<(String, f64)>,
    /// Share of samples on a thread waiting for the GPU ([`GPU_WAIT`]).
    gpu_wait: f64,
    /// Share of samples on a parked or work-stealing thread ([`POOL_IDLE`]).
    pool_idle: f64,
}

/// Total samples, self-time share per bucket, and the heaviest leaf frames.
///
/// Attribution is by **leaf frame**: a collapsed line's whole count is charged
/// to the function that was actually executing. That is what a flamegraph's
/// plateau widths show and what "53% of samples were in the driver" has always
/// meant in this project's notes; charging to any ancestor instead would count
/// the same sample once per level.
fn summarize(folded: &str) -> Attribution {
    let mut total = 0u64;
    let mut by_bucket: HashMap<&'static str, u64> = HashMap::new();
    let mut by_leaf: HashMap<&str, u64> = HashMap::new();
    let mut gpu_wait = 0u64;
    let mut pool_idle = 0u64;

    for line in folded.lines() {
        let Some((stack, count)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(count) = count.trim().parse::<u64>() else {
            continue;
        };
        let leaf = stack.rsplit(';').next().unwrap_or(stack);
        total += count;
        *by_leaf.entry(leaf).or_default() += count;
        *by_bucket.entry(classify(leaf)).or_default() += count;

        match wait_kind(stack) {
            Some(Waiting::Gpu) => gpu_wait += count,
            Some(Waiting::Pool) => pool_idle += count,
            None => {}
        }
    }
    if total == 0 {
        return Attribution {
            samples: 0,
            buckets: Vec::new(),
            leaves: Vec::new(),
            gpu_wait: 0.0,
            pool_idle: 0.0,
        };
    }

    let pct = |n: u64| n as f64 * 100.0 / total as f64;
    let mut buckets: Vec<(&'static str, f64)> =
        by_bucket.into_iter().map(|(k, v)| (k, pct(v))).collect();
    buckets.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut leaves: Vec<(String, f64)> = by_leaf
        .into_iter()
        .map(|(k, v)| (k.to_string(), pct(v)))
        .collect();
    leaves.sort_by(|a, b| b.1.total_cmp(&a.1));
    leaves.truncate(20);

    Attribution {
        samples: total,
        buckets,
        leaves,
        gpu_wait: pct(gpu_wait),
        pool_idle: pct(pool_idle),
    }
}

/// What a thread was doing when it was sampled, when it was not working.
enum Waiting {
    Gpu,
    Pool,
}

/// Frames that may legitimately appear *below* a wait marker without the thread
/// having gone back to work: the syscall and ioctl shims a blocking wait
/// descends through on its way into the kernel or the driver.
const WAIT_DESCENT: &[&str] = &[
    "syscall", "ioctl", "drm", "radv_", "vk_", "amdgpu", "entry_", "__x64", "wait", "sched",
];

/// Whether a stack is a thread waiting, and for what.
///
/// The obvious rule — "the stack mentions a wait primitive" — is wrong, and
/// wrong in the direction that flatters nobody. Rayon reaches its *working*
/// closures through `wait_until<SpinLatch>`, so every parallel attention thread
/// in an `orangu-server` prefill has a latch frame in its ancestry. Testing for
/// containment filed 87% of a profile that was measurably doing SIMD arithmetic
/// as "pool idle", and the CPU-per-token figures derived from it were low by
/// roughly the same factor.
///
/// So the marker has to be the *last* thing on the stack that means anything:
/// find the deepest wait marker and require everything below it to be
/// kernel-mode or one of the syscall/ioctl shims a wait descends through. A
/// frame like `dot_avx2` under a latch is work, and says so.
fn wait_kind(stack: &str) -> Option<Waiting> {
    let frames: Vec<&str> = stack.split(';').collect();
    let deepest = frames.iter().rposition(|f| {
        GPU_WAIT.iter().any(|m| f.contains(m)) || POOL_IDLE.iter().any(|m| f.contains(m))
    })?;
    let descended = frames[deepest + 1..]
        .iter()
        .all(|f| f.contains(KERNEL_MARK) || WAIT_DESCENT.iter().any(|m| f.contains(m)));
    if !descended {
        return None;
    }
    // The *deepest* marker decides that this is a wait; the *kind* is decided
    // by the whole stack, because a device wait blocks on the same futex a
    // parked worker does. `wait_mapped` above `futex_wait_[k]` is the GPU
    // owing an answer, not a thread pool with nothing to do.
    if frames
        .iter()
        .any(|f| GPU_WAIT.iter().any(|m| f.contains(m)))
    {
        Some(Waiting::Gpu)
    } else {
        Some(Waiting::Pool)
    }
}

/// Which bucket a leaf frame belongs to.
fn classify(frame: &str) -> &'static str {
    for (name, patterns) in BUCKETS {
        if patterns.iter().any(|p| frame.contains(p)) {
            return name;
        }
    }
    "app/other"
}

/// The pid listening on `port`, found by matching the listening socket's inode
/// against every process's open descriptors.
///
/// `llama-server` reports no pid over HTTP, and profiling needs one. Asking the
/// operating system which process owns the port under test is both accurate and
/// self-checking: it names the process that answered the benchmark's requests,
/// not a process that merely has a matching name — which is the exact
/// confusion that has produced measurements of the wrong binary here before.
pub fn pid_listening_on(port: u16) -> Option<u32> {
    let mut inodes = Vec::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(text) = std::fs::read_to_string(table) else {
            continue;
        };
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 10 {
                continue;
            }
            // `local_address` is `HEX_ADDR:HEX_PORT`; state `0A` is LISTEN.
            let Some((_, hex_port)) = f[1].rsplit_once(':') else {
                continue;
            };
            if f[3] != "0A" || u16::from_str_radix(hex_port, 16).ok() != Some(port) {
                continue;
            }
            if let Ok(inode) = f[9].parse::<u64>() {
                inodes.push(inode);
            }
        }
    }
    if inodes.is_empty() {
        return None;
    }

    let entries = std::fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else {
            continue; // another user's process, or one that exited mid-walk
        };
        for fd in fds.flatten() {
            let Ok(target) = std::fs::read_link(fd.path()) else {
                continue;
            };
            let Some(inode) = target
                .to_str()
                .and_then(|t| t.strip_prefix("socket:["))
                .and_then(|t| t.strip_suffix(']'))
                .and_then(|t| t.parse::<u64>().ok())
            else {
                continue;
            };
            if inodes.contains(&inode) {
                return Some(pid);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {

    /// `/proc/<pid>/stat`'s `comm` field is an arbitrary string in parens and
    /// may contain spaces *and* `)`. Counting fields from the start of the line
    /// then puts `utime`/`stime` at the wrong offsets and silently reports
    /// nonsense CPU time — which, since this is the check that validates the
    /// profiler, would break the thing meant to catch breakage.
    #[test]
    fn proc_stat_fields_are_counted_from_after_the_last_paren() {
        // A `comm` with a space and a close-paren in it, which is legal.
        let line = "1234 (my )evil proc) S 1 1234 1234 0 -1 4194304 100 0 0 0                     250 750 0 0 20 0 24 0 999 0 0";
        let rest = &line[line.rfind(')').unwrap() + 1..];
        let f: Vec<&str> = rest.split_whitespace().collect();
        assert_eq!(f[11], "250", "utime");
        assert_eq!(f[12], "750", "stime");
    }

    /// The clock unit must be read, not assumed — the CPU-time comparison is
    /// only meaningful in the same units as the wall clock.
    #[test]
    fn clock_ticks_are_positive() {
        assert!(clock_ticks_per_second() > 0.0);
    }

    /// This process is running, so its own CPU time must be readable and
    /// non-negative; a bogus pid must give `None` rather than a wrong number.
    #[test]
    fn proc_cpu_seconds_reads_this_process_and_rejects_a_bogus_pid() {
        let me = std::process::id();
        let t = proc_cpu_seconds(me).expect("own /proc/<pid>/stat should be readable");
        assert!(t >= 0.0 && t.is_finite(), "got {t}");
        assert!(proc_cpu_seconds(u32::MAX).is_none());
    }
    use super::*;

    #[test]
    fn leaf_attribution_charges_the_whole_stack_to_the_executing_frame() {
        // Two stacks sharing an ancestor: the ancestor must not be counted.
        // Charging to any ancestor would report 100% driver here, which is the
        // failure mode this attribution exists to avoid.
        let folded = "main;submit;radv_queue_submit 70\nmain;submit;my_own_code 30\n";
        let a = summarize(folded);
        assert_eq!(a.samples, 100);
        assert_eq!(a.buckets[0], ("radv/vulkan", 70.0));
        assert_eq!(a.buckets[1], ("app/other", 30.0));
        assert_eq!(a.leaves[0], ("radv_queue_submit".to_string(), 70.0));
    }

    #[test]
    fn a_kernel_frame_is_recognised_by_its_mark_not_by_its_name() {
        // The two frames below were both filed under "app" by a name-pattern
        // classifier — they are ordinary-looking symbols that happen to live in
        // the kernel. The `_[k]` mark is what makes them unmistakable, and a
        // regression here would quietly inflate whatever "app/other" is used to
        // argue.
        let folded = "a;read_hpet_[k] 30\nb;perf_event_update_userpage_[k] 20\nc;my_own_code 50\n";
        let a = summarize(folded);
        assert!(a.buckets.contains(&("kernel", 50.0)));
        assert!(a.buckets.contains(&("app/other", 50.0)));
    }

    #[test]
    fn the_kernel_mode_gpu_driver_is_separated_from_the_userspace_one() {
        // `radv_*` runs on the calling thread; `amdgpu_*` is reached through an
        // ioctl. Collapsing them into one "gpu driver" number would hide which
        // of the two a change moved.
        let folded = "a;radv_CmdDispatch 40\nb;amdgpu_vm_bo_update_[k] 60\n";
        let a = summarize(folded);
        assert!(a.buckets.contains(&("radv/vulkan", 40.0)));
        assert!(a.buckets.contains(&("kernel:gpu", 60.0)));
    }

    #[test]
    fn a_frame_matching_nothing_is_reported_as_unclassified_not_dropped() {
        // The residual has to stay visible: a bucket table that silently drops
        // what it cannot name would read as "everything is accounted for".
        let folded = "a;rmsnorm_inplace 40\nb;ggml_compute_forward 60\n";
        let a = summarize(folded);
        assert_eq!(a.samples, 100);
        assert_eq!(a.buckets.iter().map(|(_, p)| p).sum::<f64>(), 100.0);
        assert!(a.buckets.contains(&("app/other", 40.0)));
        assert!(a.buckets.contains(&("ggml", 60.0)));
    }

    #[test]
    fn a_spinning_wait_and_a_blocking_wait_are_both_counted_as_waiting() {
        // The whole point: these two stacks are the *same* state — a thread
        // with nothing to do until the GPU finishes — expressed by two engines
        // that wait differently. Counting only one of them is what makes the
        // engine that burns a core look like the busier worker.
        let folded = concat!(
            "llama;ggml_vk_wait_for_fence;_mm_pause 50\n",
            "orangu;wgpu;wait_mapped;futex_wait_[k];schedule_[k] 30\n",
            "orangu;compute;my_own_code 20\n",
        );
        let a = summarize(folded);
        assert_eq!(a.gpu_wait, 80.0);
        assert_eq!(a.pool_idle, 0.0);
    }

    #[test]
    fn work_reached_through_a_latch_is_work_not_idle() {
        // The regression this rule exists for. Rayon runs its *working* closures
        // under `wait_until<SpinLatch>`; a containment test filed 87% of a
        // profile doing measurable SIMD arithmetic as idle, and every
        // CPU-per-token figure derived from it was wrong by that factor.
        let folded = concat!(
            "orangu;rayon;wait_until<rayon_core::latch::SpinLatch>;bridge;dot_avx2 60\n",
            "orangu;rayon;wait_until_out_of_work;futex_wait_[k];schedule_[k];sched_in_[k] 40\n",
        );
        let a = summarize(folded);
        assert_eq!(a.pool_idle, 40.0);
        assert_eq!(a.gpu_wait, 0.0);
    }

    #[test]
    fn a_device_wait_that_descends_through_an_ioctl_still_counts_as_waiting() {
        // wgpu's blocking map goes user → libdrm → ioctl → kernel. None of the
        // frames below the marker is work, and a rule that only allowed kernel
        // frames would miss the two userspace shims in between.
        let folded = "orangu;wgpu;wait_mapped;drmSyncobjTimelineWait;drmIoctl;__ioctl;entry_SYSCALL_64_[k] 10\n";
        let a = summarize(folded);
        assert_eq!(a.gpu_wait, 100.0);
    }

    #[test]
    fn a_parked_worker_is_pool_idle_not_gpu_wait() {
        // Rayon workers parking between layers are overhead of the engine's own
        // threading, not time the GPU owed anyone. Merging the two would credit
        // a thread-pool problem to the device.
        let folded = concat!(
            "orangu;rayon;wait_until_out_of_work;futex_wait_[k] 40\n",
            "orangu;compute;my_own_code 60\n",
        );
        let a = summarize(folded);
        assert_eq!(a.gpu_wait, 0.0);
        assert_eq!(a.pool_idle, 40.0);
    }

    #[test]
    fn sibling_paths_share_the_svgs_stem() {
        let svg = PathBuf::from("/tmp/decode-gemma.svg");
        assert_eq!(
            sibling(&svg, "folded"),
            PathBuf::from("/tmp/decode-gemma.folded")
        );
        assert_eq!(
            sibling(&svg, "perf.data"),
            PathBuf::from("/tmp/decode-gemma.perf.data")
        );
    }

    #[test]
    fn a_folded_line_with_no_count_is_skipped_rather_than_counted_as_zero() {
        assert_eq!(summarize("garbage-with-no-count\na;b 5\n").samples, 5);
    }
}

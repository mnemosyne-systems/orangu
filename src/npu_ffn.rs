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

//! Running a model's feed-forward blocks on the NPU, from the engine.
//!
//! [`NpuFfn::open`] loads whatever the cache holds for a model and hands
//! back something the forward pass can ask: *do you have layer `n` at this
//! token count, and if so, here is `x`*. Everything else in this module
//! exists to make that question answerable from a thread that is not the
//! one holding the device.
//!
//! **Why a thread.** [`crate::npu::NpuRuntime`] and its graphs are built on
//! [`std::rc::Rc`] — the vendor runtime is not thread-safe and the types say
//! so by not being [`Send`]. The engine's forward pass is not single
//! threaded and is not going to become so for this. So one thread owns the
//! device and every graph on it, and callers reach it through a channel.
//! The caller blocks until the answer comes back, which costs nothing it
//! would not have spent computing the same block itself.
//!
//! **One width, and the prefill chunker is told what it is.** A graph is
//! compiled for one token count, so a chunk of any other size is not
//! offered the device at all — and the chunk sizer, which prices
//! submissions against a time budget, has no reason to pick that number.
//! Left alone it issued `[16, 16, 25, 16, 25, 19, 16, 24, 17, ...]` on a
//! 980-token prompt and more than half the prompt missed. [`prefill_width`]
//! is how it now knows, and it is worth 1.87x on top of everything below:
//! that prompt went 5.52 -> 10.30 tok/s when every chunk started landing.
//!
//! [`prefill_width`]: NpuFfn::prefill_width
//!
//! A decode step can also be served, through the seam in
//! `LlamaModel::record_decode_run` that stops a fused layer at `ffn_norm`
//! and finishes it on the host. It is correct and slightly slower than
//! leaving decode on the GPU, so it is off unless asked for — see
//! `npu_tool::decode_enabled`, which has both the measurement and the
//! reason.
//!
//! **What it is worth.** A 720-token prompt through the HTTP API, device
//! off against device on, both runs reaching the measured prompt with the
//! same history:
//!
//! | model | off | on | |
//! |---|---|---|---|
//! | gemma 4 E2B `Q8_0` | 4.44 tok/s | **12.30** | 2.8x |
//! | gemma 4 E2B `Q4_K_M` | 9.57 | **16.96** | 1.8x |
//! | gemma 4 E4B `Q8_0` | 2.43 | **7.00** | 2.9x |
//! | qwen3 4B `Q8_0` | 2.72 | **7.15** | 2.6x |
//! | llama 3.2 3B `Q8_0` | 3.65 | **10.30** | 2.8x |
//!
//! **A prompt has to be long enough to reach the chunker.** One forward
//! pass over a prompt that fits a single batch is more efficient than the
//! same prompt in device-width pieces, so `prefill_in_chunks` keeps its
//! single-pass path and the device only sees prompts past it. Across
//! lengths, llama 3.2 3B `Q8_0`:
//!
//! | prompt | off | on |
//! |---|---|---|
//! | 75 tokens | 9.38 tok/s | 9.47 |
//! | 193 tokens | 19.61 | 19.90 |
//! | 631 tokens | 3.71 | **10.56** |
//! | 2080 tokens | 3.53 | **10.09** |
//!
//! Short prompts are left exactly as fast as they were, which is the
//! result of measuring the alternative: forcing them through the device
//! took 193 tokens from 19.81 tok/s to 10.80.
//!
//! The `Q4_K_M` row is the fastest configuration measured on this machine
//! and the smallest multiple, which is the same fact twice: its GPU
//! baseline is already 2.2x the `Q8_0` one, so there is less left for the
//! device to take. What the NPU is worth depends on what it is taking work
//! away from.
//!
//! All four answer the question correctly with the device on. Taking a
//! block here also means declining the GPU's fused post-attention path for
//! that layer — that fusion swallows the FFN, so there is no way to have
//! both — and the numbers above are *after* paying that.
//!
//! **It changes the output.** Requantizing a matrix changes which token
//! wins a close decision, so at temperature 0 the same prompt answered with
//! and without the device produces two different, equally good answers.
//! Nothing here is bit-reproducible against the CPU path, and a caller that
//! needs identical output across backends should set `npu_precompile = off`.
//!
//! Three things had to be right before any of that held. Two are measured
//! in `npu_tool::captured_calibration` and `npu_ort::smoothing_scales`: the
//! blocks are calibrated on activations captured from the model's own first
//! prompt, not on synthetic noise, and the per-channel outliers in those
//! activations are folded into the weights rather than left to set the
//! activation scale for every channel. Without the first, blocks return
//! output with no signal in it; without the second, the model answers a
//! different question than the one asked. The third is
//! `npu_tool::CALIBRATION_ROWS`: how many rows that calibration is
//! estimated from is not the width of the graph being built, and conflating
//! them made every decode block a per-channel scale fitted to a sample of
//! one.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::npu_cache::{BlockKey, NpuCache, fingerprint};

/// A bound artifact, ready to serve.
///
/// The feed-forward variant is much the larger, and both are boxed rather
/// than padding every attention entry out to its size — there is one of
/// these per layer per width and the map holds all of them.
enum Bound {
    Ffn(Box<crate::npu_ort::NpuGatedFfn>),
    Attn(Vec<crate::npu_ort::NpuLinear>),
}

/// A compiled artifact on its way to the device thread.
///
/// Two kinds travel the same path because they share one device, one
/// thread and one capacity limit — see [`NpuFfn::open`]'s note on what the
/// device will hold.
enum Artifact {
    Ffn(Box<crate::npu_ort::CompiledGatedFfn>),
    Attn(crate::npu_ort::CompiledProjections),
}

/// One block's worth of work, and where to send the answer.
/// Which of a layer's two offloadable pieces a job is for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Kind {
    /// The gated feed-forward network.
    Ffn,
    /// The attention `Q`/`K`/`V` projections, whose three outputs come back
    /// concatenated in that order — one buffer, so the job plumbing and its
    /// pool are the same for both kinds.
    Attn,
}

struct Job {
    kind: Kind,
    layer: usize,
    tokens: usize,
    /// The activations, and the buffer the answer is written into. Both are
    /// carried *to* the device thread and handed back in the [`Reply`], so a
    /// steady-state forward pass allocates nothing: the pair makes a round
    /// trip and returns to [`NpuFfn::spare`] for the next block to use.
    x: Vec<f32>,
    out: Vec<f32>,
    reply: Sender<Reply>,
}

/// A finished job, with both of its buffers returned for reuse.
struct Reply {
    /// `Ok` leaves the answer in `out`; `Err` describes what went wrong and
    /// leaves `out` in whatever state the failed call left it.
    result: Result<(), String>,
    x: Vec<f32>,
    out: Vec<f32>,
}

/// The feed-forward blocks this machine can run on its NPU for one model.
pub struct NpuFfn {
    /// `(layer, tokens)` pairs that are compiled and loaded. Consulted on
    /// every layer of every forward pass, so it is a plain set lookup and
    /// never touches the device thread.
    available: std::collections::HashSet<(Kind, usize, usize)>,
    /// `None` once [`shutdown`] has taken it. Dropping the sender is what
    /// ends the device thread's `recv` loop, so it is the handle on the
    /// thread's lifetime as much as it is the way to reach it.
    jobs: Mutex<Option<Sender<Job>>>,
    /// The device thread, kept so it can be **joined** at exit.
    ///
    /// Discarding this is what made `orangu-server` abort on Ctrl+C with
    /// `pure virtual method called` / `terminate called without an active
    /// exception`. The thread owns `NpuRuntime` and every graph on it, all
    /// vendor C++ objects; the service that holds its sender lives in a
    /// `OnceLock` static, and Rust does not drop statics. So the sender
    /// outlived `main`, the loop never ended, and the vendor library's own
    /// static destructors ran while that thread was still inside its
    /// objects. Joining first destroys them on the thread that made them,
    /// in the order the library expects.
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Buffer pairs not currently in flight.
    ///
    /// A forward pass takes a pair, fills it, and puts it back when the
    /// device answers. The pool grows to the number of blocks running
    /// concurrently — one, for a single request — and then never allocates
    /// again.
    spare: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Whether any block has actually run here yet. See [`Self::forward`].
    used: std::sync::atomic::AtomicBool,
    /// Blocks the device thread has taken off the device. See [`Self::has`].
    withdrawn: Mutex<std::collections::HashSet<(Kind, usize, usize)>>,
    /// Whether [`Self::withdrawn`] is non-empty, so the common case never
    /// takes that lock.
    withdrew: std::sync::atomic::AtomicBool,
}

impl NpuFfn {
    /// Loads every cached block for `model` onto the device.
    ///
    /// `None` — never an error — when there is no NPU, no cache directory,
    /// or nothing cached for this model. A machine without the hardware pays
    /// one `detect_npu` call for this.
    pub fn open(
        model: &Path,
        blocks: &[(usize, String)],
        widths: &[usize],
        calibration: u64,
    ) -> Option<Self> {
        // The cheapest check first: an ordinary machine has no NPU and
        // should pay nothing beyond this for the feature.
        crate::npu::detect_npu()?;
        let gguf = crate::gguf::GgufFile::open(model).ok()?;
        let model_fingerprint = fingerprint(&gguf);
        let cache = NpuCache::open(NpuCache::default_dir()).ok()?;

        // Read the artifacts here, on the caller's thread: they are plain
        // bytes, and a failed read should be an early `None` rather than a
        // thread that starts and then has nothing to do.
        let mut artifacts = Vec::new();
        // **Every feed-forward block first, then the attention ones.**
        //
        // The device refuses graphs past a capacity, and whatever is offered
        // last is what gets dropped. Interleaving the two kinds by layer —
        // which this did — spends that capacity in layer order regardless of
        // what the blocks are worth, and on this board it silently bought
        // the wrong thing: 52 slots went to 16 feed-forward and 16 attention
        // blocks at the prefill width, then layers 0..9 at the decode width,
        // leaving **10 of 16** decode feed-forward blocks. The seam needs
        // 80% of the layers (`DECODE_SEAM_COVERAGE_PERCENT`), so it declined
        // and decode never reached the device at all — while the startup
        // line still said `widths [128, 1]`, 52 blocks in use.
        //
        // The two are not worth the same. A feed-forward block is the whole
        // of a layer's arithmetic and measured 1.8-2.9x on prefill and 1.42x
        // on decode at two slots; the attention projections measured 2.1%.
        // So the cheap thing must not be able to crowd out the valuable one,
        // and ordering is all it takes.
        for &tokens in widths {
            for (layer, prefix) in blocks {
                let key = BlockKey {
                    model: &model_fingerprint,
                    block: prefix,
                    tokens,
                    calibration,
                };
                match cache.load(key) {
                    Ok(Some(c)) => artifacts.push((Artifact::Ffn(Box::new(c)), *layer, tokens)),
                    // A corrupt artifact is worth saying out loud; a missing
                    // one is just a block that stays on the CPU or GPU.
                    Err(e) => eprintln!("orangu-server: [npu] {prefix}: {e}"),
                    Ok(None) => {}
                }
            }
        }
        for &tokens in widths {
            for (layer, prefix) in blocks {
                // The same layer's attention projections, when they were
                // compiled. Loaded here so both kinds go to the device
                // thread together and share its capacity bookkeeping.
                let attn = BlockKey {
                    model: &model_fingerprint,
                    block: &format!("attn.{prefix}"),
                    tokens,
                    calibration,
                };
                match cache.load_projections(attn) {
                    Ok(Some(compiled)) => {
                        artifacts.push((Artifact::Attn(compiled), *layer, tokens))
                    }
                    Err(e) => eprintln!("orangu-server: [npu] attn.{prefix}: {e}"),
                    Ok(None) => {}
                }
            }
        }
        if artifacts.is_empty() {
            return None;
        }
        // **What this device was able to hold last time.** Binding a graph
        // allocates its job's working buffers on the device, and the device
        // is smaller than the host cache that decides how many artifacts
        // exist. Past its limit `noe_create_job` answers
        // `NOE_STATUS_ERROR_BUF_ALLOC_FAIL` (22) — and leaves buffers it
        // then cannot free, which surfaces at exit as
        // `[UMD ERR] ukmemory.cpp: free buffer ... [fail]`. Measured on a
        // Zhouyi X2 with gemma 4 E2B `Q4_K_M`: 35 artifacts offered, the
        // 35th refused, four unfreeable buffers at shutdown; the same model
        // held to 26 artifacts bound them all and exited silently.
        //
        // The limit is not knowable in advance — it depends on the device,
        // the block shape and the width — so it is learned by hitting it
        // once and then remembered, exactly as the prefill width is.
        //
        // It is deliberately never raised again. The capacity belongs to the
        // device, the block shape and the width, none of which change while
        // the hint is valid; compiling *more* blocks later does not make the
        // device bigger, so re-offering the full set would only refuse again
        // and leak another four buffers. Growing out of a limit is the
        // operator's move — delete the file named below — which is why the
        // limit says where it lives.
        let capacity = capacity_hint_path(&model_fingerprint, widths);
        if let Some(limit) = read_capacity_hint(&capacity)
            && limit < artifacts.len()
        {
            eprintln!(
                "orangu-server: [npu] holding to {limit} block-width(s) of {}, the most this \
                 device accepted before — delete {} to measure it again",
                artifacts.len(),
                capacity.display()
            );
            artifacts.truncate(limit);
        }
        let offered = artifacts.len();

        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Vec<(Kind, usize, usize)>>();
        let thread = std::thread::Builder::new()
            .name("orangu-npu".into())
            .spawn(move || device_thread(artifacts, rx, ready_tx))
            .ok()?;
        // What actually bound, not what was asked for. A graph can fail to
        // load — this device refuses to create a job past a certain number
        // of resident graphs — and a layer that is offered but not there
        // would be asked once per forward pass, fail, and be reported every
        // time. So the availability set is built from the thread's answer.
        let bound: Vec<(Kind, usize, usize)> = ready_rx.recv().ok()?;
        if bound.is_empty() {
            return None;
        }
        // Remembered only when the device actually refused, so a machine
        // that fits everything never writes a limit and never has one to
        // grow out of.
        //
        // **And only when the device was ours alone.** What the refusal
        // measures is how much room was left, which is the device's capacity
        // only if nothing else was on it. Persisting a number learned beside
        // another process would cap this model for every later run — the
        // hint only ever truncates and is never re-measured — so a contended
        // startup uses its smaller set for this run and writes nothing. The
        // next uncontended start then learns the real limit.
        //
        // This is not hypothetical on this board: an orphaned `npu-compile`
        // was measured holding the device eleven minutes after its parent
        // died, which is exactly the window a server would start in.
        if bound.len() < offered {
            let sharing = crate::npu::other_npu_users();
            if sharing.is_empty() {
                write_capacity_hint(&capacity, bound.len());
            } else {
                eprintln!(
                    "orangu-server: [npu] the device refused past {} block-width(s), but \
                     pid(s) {sharing:?} were using it too — not remembering a limit measured \
                     against someone else's memory",
                    bound.len()
                );
            }
        }
        Some(Self {
            available: bound.into_iter().collect(),
            jobs: Mutex::new(Some(tx)),
            thread: Mutex::new(Some(thread)),
            spare: Mutex::new(Vec::new()),
            used: std::sync::atomic::AtomicBool::new(false),
            withdrawn: Mutex::new(std::collections::HashSet::new()),
            withdrew: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Whether layer `layer` at `tokens` tokens can run on the device.
    ///
    /// **Withdrawn blocks are not available.** The device thread takes a
    /// block off the device when it answers a real input with nothing, and
    /// a caller that kept being told the block was there would keep asking:
    /// the decode seam decides once per step whether to break the fused GPU
    /// chain, and it made that decision on a set that no longer matched
    /// what the device would serve — so every step paid a submit-and-read
    /// per layer, failed on the withdrawn one, computed it unfused anyway,
    /// and printed an error. Measured on llama 3.2 3B `Q8_0`, where layer 1
    /// is withdrawn, that is the worst of both paths.
    ///
    /// The common case pays one relaxed atomic load: nothing is ever
    /// withdrawn on a healthy model, and the lock is only taken once
    /// something has been.
    pub fn has(&self, layer: usize, tokens: usize) -> bool {
        self.holds(Kind::Ffn, layer, tokens)
    }

    /// Whether this layer's attention projections can run on the device.
    pub fn has_attn(&self, layer: usize, tokens: usize) -> bool {
        self.holds(Kind::Attn, layer, tokens)
    }

    fn holds(&self, kind: Kind, layer: usize, tokens: usize) -> bool {
        if !self.available.contains(&(kind, layer, tokens)) {
            return false;
        }
        if !self.withdrew.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        self.withdrawn
            .lock()
            .is_ok_and(|set| !set.contains(&(kind, layer, tokens)))
    }

    /// Takes a block off the device for the life of the process.
    ///
    /// Called from the device thread, which has already stopped serving it;
    /// this is how the rest of the engine finds out.
    fn withdraw(&self, kind: Kind, layer: usize, tokens: usize) {
        if let Ok(mut set) = self.withdrawn.lock() {
            set.insert((kind, layer, tokens));
            self.withdrew
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The width this model's prefill blocks were compiled for, if any —
    /// the widest bound width above one.
    ///
    /// A graph has one static shape, so a chunk of any other size is not
    /// offered the device at all. The prefill chunker consults this and
    /// issues exactly this width wherever its own budget allows, which is
    /// the difference between the device seeing a prompt and not: left to
    /// itself the sizer issued `[16, 16, 25, 16, 25, 19, 16, 24, 17, ...]`
    /// on a 980-token prompt — 52 submissions of which about half missed —
    /// and a prompt shorter than one batch was one pass of 980 tokens that
    /// missed entirely.
    pub fn prefill_width(&self) -> Option<usize> {
        self.available
            .iter()
            .map(|(_, _, tokens)| *tokens)
            .filter(|tokens| *tokens > 1)
            .max()
    }

    /// Every width this model has graphs for, ascending.
    pub fn widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self
            .available
            .iter()
            .map(|(_, _, tokens)| *tokens)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        widths.sort_unstable();
        widths
    }

    /// How many blocks are loaded.
    pub fn len(&self) -> usize {
        self.available.len()
    }

    /// Loaded feed-forward blocks at `tokens`, which is the number the
    /// decode seam's coverage test is actually about.
    ///
    /// [`Self::widths`] and [`Self::len`] do not distinguish a feed-forward
    /// block from an attention one, so a server reporting `widths [1, 128]`
    /// may hold no width-1 *feed-forward* block at all — and the seam, which
    /// needs most of the layers, then declines while every other indicator
    /// says the device is busy at width 1. That is exactly how a decode path
    /// that never ran came to look like one that did.
    pub fn ffn_blocks_at(&self, tokens: usize) -> usize {
        self.available
            .iter()
            .filter(|(kind, _, width)| *kind == Kind::Ffn && *width == tokens)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Runs one block into `out`, blocking until the device answers.
    ///
    /// Returns `false` if this block is not on the device or the device
    /// failed, leaving `out` untouched in the first case, in which case the
    /// caller computes the block the way it otherwise would. A failure is
    /// reported once — a device that has started returning errors will not
    /// stop because it was asked more times.
    ///
    /// Allocates nothing in steady state: the buffers it hands the device
    /// thread come from [`Self::spare`] and go back there.
    pub fn forward_into(&self, layer: usize, tokens: usize, x: &[f32], out: &mut Vec<f32>) -> bool {
        self.run(Kind::Ffn, layer, tokens, x, out)
    }

    /// Runs this layer's `Q`/`K`/`V` into `out`, concatenated in that order.
    ///
    /// The caller knows each projection's width and slices the result; see
    /// [`Kind::Attn`] for why one buffer rather than three.
    pub fn forward_attn_into(
        &self,
        layer: usize,
        tokens: usize,
        x: &[f32],
        out: &mut Vec<f32>,
    ) -> bool {
        self.run(Kind::Attn, layer, tokens, x, out)
    }

    fn run(&self, kind: Kind, layer: usize, tokens: usize, x: &[f32], out: &mut Vec<f32>) -> bool {
        if !self.holds(kind, layer, tokens) {
            return false;
        }
        let (mut job_x, job_out) = self
            .spare
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_default();
        job_x.clear();
        job_x.extend_from_slice(x);

        let (reply, answer) = channel();
        let job = Job {
            kind,
            layer,
            tokens,
            x: job_x,
            out: job_out,
            reply,
        };
        // Scoped so the lock is released before the wait below: the device
        // serializes the work anyway, but holding this across a blocking
        // `recv` would stop any other thread even queueing behind it.
        let queued = {
            let Ok(sender) = self.jobs.lock() else {
                return false;
            };
            // `None` after `shutdown`: the device is gone, and the caller
            // computes the block the way it would have without one.
            match sender.as_ref() {
                Some(sender) => sender.send(job).is_ok(),
                None => false,
            }
        };
        if !queued {
            return false;
        }
        let Ok(answer) = answer.recv() else {
            return false;
        };

        let ok = match &answer.result {
            Ok(()) => {
                out.clear();
                out.extend_from_slice(&answer.out);
                // Once, on the first block that actually runs. Whether the
                // device is *being used* is not the same question as whether
                // it loaded, and the difference is easy to miss: the blocks
                // were first compiled for a width the prefill chunker never
                // issues, so everything loaded, reported success, and was
                // never asked for anything.
                if !self.used.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "orangu-server: [npu] first feed-forward block ran on the NPU \
                         (layer {layer}, {tokens} tokens)"
                    );
                }
                true
            }
            Err(e) => {
                eprintln!("orangu-server: [npu] layer {layer}: {e}");
                false
            }
        };
        if let Ok(mut pool) = self.spare.lock() {
            pool.push((answer.x, answer.out));
        }
        ok
    }
}

/// Where this model's learned device capacity is remembered.
///
/// Keyed by the model *and* the widths, because both decide how large a
/// graph's job buffers are and therefore how many of them fit.
fn capacity_hint_path(fingerprint: &str, widths: &[usize]) -> std::path::PathBuf {
    let widths: Vec<String> = widths.iter().map(usize::to_string).collect();
    NpuCache::default_dir().join(format!("graphs-{fingerprint}-{}", widths.join("x")))
}

/// The most artifacts this device accepted for `fingerprint` at `widths`, if
/// that has been measured.
///
/// Exposed so the *compile* pass can consult it. Binding is where the limit is
/// discovered, but compiling is where the cost is paid: a block the device
/// cannot hold takes the same seven seconds and the same disk as one it can,
/// and then never runs. Measured on gemma 4 E2B `Q4_K_M`: 35 feed-forward
/// blocks compiled, 34 accepted, so one block's compile was spent on nothing.
///
/// `None` when the device has never refused — which is the ordinary case, and
/// means "no reason to compile less".
pub fn capacity_hint(fingerprint: &str, widths: &[usize]) -> Option<usize> {
    read_capacity_hint(&capacity_hint_path(fingerprint, widths))
}

fn read_capacity_hint(path: &Path) -> Option<usize> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|limit| *limit > 0)
}

fn write_capacity_hint(path: &Path, bound: usize) {
    let _ = std::fs::write(path, bound.to_string());
}

/// The one thread that touches the device.
///
/// Binds every artifact, reports which ones bound, then serves jobs until the
/// channel closes — which happens when the last [`NpuFfn`] is dropped, so
/// the graphs are released with the model rather than held for the life of
/// the process.
fn device_thread(
    artifacts: Vec<(Artifact, usize, usize)>,
    jobs: Receiver<Job>,
    ready: Sender<Vec<(Kind, usize, usize)>>,
) {
    let Some(runtime) = crate::npu::NpuRuntime::open() else {
        let _ = ready.send(Vec::new());
        return;
    };
    // Keyed by `(layer, tokens)`: a block is compiled once per width, and
    // decode's width-1 graph and prefill's width-16 graph are different
    // executables for the same layer.
    let mut bound: HashMap<(Kind, usize, usize), Bound> = HashMap::new();
    for (compiled, layer, tokens) in artifacts {
        let kind = match &compiled {
            Artifact::Ffn(_) => Kind::Ffn,
            Artifact::Attn(_) => Kind::Attn,
        };
        let outcome = match compiled {
            Artifact::Ffn(c) => c.bind(&runtime).map(|f| Bound::Ffn(Box::new(f))),
            Artifact::Attn(c) => c.bind(&runtime).map(Bound::Attn),
        };
        match outcome {
            Ok(ffn) => {
                bound.insert((kind, layer, tokens), ffn);
            }
            Err(e) => {
                // **Stop, rather than try the rest.** This device refuses a
                // graph when it has no room for the job's buffers, so every
                // remaining artifact would fail the same way — and each
                // failed attempt leaves the driver holding buffers it
                // cannot free at exit. `NpuFfn::open` records how many did
                // bind so the next run does not walk into it again.
                eprintln!(
                    "orangu-server: [npu] layer {layer} at {tokens} tokens did not load, so \
                     the device is full at {} block-width(s): {e}",
                    bound.len()
                );
                break;
            }
        }
    }
    let mut loaded: Vec<(Kind, usize, usize)> = bound.keys().copied().collect();
    loaded.sort_unstable();
    if ready.send(loaded).is_err() || bound.is_empty() {
        return;
    }

    // Blocks that have returned something once. A block only has to prove
    // itself once, so the check below costs one pass over one output per
    // block for the life of the process.
    let mut proven: std::collections::HashSet<(Kind, usize, usize)> =
        std::collections::HashSet::new();
    // How long the device sits with nothing to do, under `ORANGU_NPU_TIME`.
    //
    // This is the question behind "overlap the fence": a job's own
    // dispatches are already overlapped (gate with up, and `Q`/`K`/`V` all
    // three before any collect), so the only time left to win back is the
    // gap between one job finishing and the next arriving. When a caller has
    // no other work — one slot, and every layer waiting on the one before —
    // that gap is the caller thinking, and no amount of asynchrony on this
    // side removes it. When several slots are decoding, the queue should
    // already be full and the gap near zero. Measured rather than argued.
    /// The longest gap between two jobs still counted as the device
    /// waiting on the pipeline rather than on a user. A decode layer's
    /// host-side work is tens of microseconds and a whole step is a few
    /// milliseconds, so 50 ms is far above any real inter-job wait and far
    /// below the pauses between requests.
    const IDLE_GAP_CEILING_NS: u64 = 50_000_000;
    let mut idle_ns = 0u64;
    let mut busy_ns = 0u64;
    let mut jobs_done = 0u64;
    let mut since = std::time::Instant::now();
    while let Ok(mut job) = jobs.recv() {
        if crate::npu::stage_timing() {
            // Only gaps short enough to be *this* pipeline waiting. A longer
            // one means no request was in flight at all, and counting the
            // server's quiet minutes as device idleness reports "100%
            // waiting for work" on a box that simply had nothing to do —
            // which is what the first version of this said.
            let gap = since.elapsed().as_nanos() as u64;
            if gap < IDLE_GAP_CEILING_NS {
                idle_ns += gap;
            }
        }
        let started = std::time::Instant::now();
        // Straight into the caller's buffer — nothing is cloned on the way
        // back, and `NpuGatedFfn` keeps its own scratch, so a served job
        // allocates only the first time each buffer grows.
        let key = (job.kind, job.layer, job.tokens);
        let mut result = match bound.get(&key) {
            Some(Bound::Ffn(ffn)) => ffn
                .forward_into(&job.x, job.tokens, &mut job.out)
                .map_err(|e| e.to_string()),
            // `Q`, `K` and `V` back to back in one buffer, in that order.
            // Three outputs would need three buffers through the pool for
            // no gain: the caller knows each width and slices.
            Some(Bound::Attn(parts)) => {
                // **All three dispatched before any is waited for.** They
                // read the same activations and none reads another, so the
                // only thing ordering them would be the driver's queue —
                // the same argument as `gate` and `up` in a gated block,
                // where overlapping took a width-1 block from 3.27 ms to
                // 2.65. Running them one at a time here would pay three
                // full round trips for three small matrices.
                job.out.clear();
                let dispatched = parts.iter().try_fold((), |(), part| {
                    part.dispatch(&job.x, job.tokens).map_err(|e| e.to_string())
                });
                let mut piece = Vec::new();
                dispatched.and_then(|()| {
                    parts.iter().try_fold((), |(), part| {
                        part.collect(job.tokens, &mut piece)
                            .map_err(|e| e.to_string())?;
                        job.out.extend_from_slice(&piece);
                        Ok(())
                    })
                })
            }
            None => Err(format!(
                "layer {} at {} tokens is not on the device",
                job.layer, job.tokens
            )),
        };
        // **A block that answers a real input with nothing is not computing
        // this network**, and it is the one failure the per-model error gate
        // cannot catch: that gate samples five blocks and measures them on
        // the rows they were calibrated from, so a block that collapses only
        // on the inputs it actually serves passes it. Measured on llama 3.2
        // 3B `Q8_0`, layer 1 comes back exactly zero at both widths while
        // every projection inside it uses 40-90 of its 256 output levels —
        // so this is not a starved quantization, and dropping the layer to
        // the GPU is the honest response to a result with no signal in it.
        if crate::npu::stage_timing() {
            busy_ns += started.elapsed().as_nanos() as u64;
            jobs_done += 1;
            if jobs_done.is_multiple_of(256) {
                let us = |v: u64| v as f64 / (jobs_done as f64 * 1000.0);
                eprintln!(
                    "orangu-server: [npu] {jobs_done} jobs: {:.0} us busy, {:.0} us idle \
                     between them ({:.0}% of the device's time is waiting for work)",
                    us(busy_ns),
                    us(idle_ns),
                    100.0 * idle_ns as f64 / (idle_ns + busy_ns).max(1) as f64
                );
            }
        }
        if result.is_ok() && !proven.contains(&key) {
            if job.out.iter().any(|v| *v != 0.0) {
                proven.insert(key);
            } else if job.x.iter().any(|v| *v != 0.0) {
                bound.remove(&key);
                // And tell the rest of the engine, so the decode seam stops
                // being offered a layer this will now decline — see
                // [`NpuFfn::has`]. The service is installed by the time any
                // job reaches here, because jobs only come from it.
                if let Some(service) = service() {
                    service.withdraw(job.kind, job.layer, job.tokens);
                }
                eprintln!(
                    "orangu-server: [npu] layer {} at {} tokens returned all zeros for a \
                     non-zero input and has been taken off the device",
                    job.layer, job.tokens
                );
                result = Err("the block returned no signal".into());
            }
        }
        // A caller that gave up waiting is not an error worth reporting.
        // The idle clock starts the moment this job is answered, so what it
        // measures is the device waiting for the *next* one.
        since = std::time::Instant::now();
        let _ = job.reply.send(Reply {
            result,
            x: job.x,
            out: job.out,
        });
    }
}

/// The service the forward pass consults, installed once at model load.
///
/// A process-global, matching how `engine::backend::vulkan`'s own
/// preferences reach the same place: the alternative is threading an
/// `Option<&NpuFfn>` through every architecture's forward signature for a
/// feature most machines do not have.
static SERVICE: std::sync::OnceLock<Option<NpuFfn>> = std::sync::OnceLock::new();

/// Installs the service. Later calls are ignored — the first model loaded
/// owns the device for the life of the process.
pub fn install(ffn: Option<NpuFfn>) {
    let _ = SERVICE.set(ffn);
}

/// Releases the device and **joins** the thread that owns it.
///
/// Call once, on the way out of `main`. Without it `orangu-server` aborted
/// on Ctrl+C — `pure virtual method called`, then `terminate called without
/// an active exception` — because the vendor runtime's static destructors
/// ran while the device thread was still alive inside its objects. See
/// [`NpuFfn::thread`] for why the thread could never end on its own.
///
/// Verified both ways on a Mali-G720 + Zhouyi X2 box, with 35 blocks bound
/// and a request served so the device thread was live: without this call
/// the process exits 134 and prints
///
/// ```text
/// shutting down
/// pure virtual method called
/// terminate called without an active exception
/// ```
///
/// and with it, 0 and nothing.
///
/// Safe to call when there is no NPU, when nothing was installed, and more
/// than once: each step is a `take`.
pub fn shutdown() {
    let Some(Some(ffn)) = SERVICE.get() else {
        return;
    };
    ffn.stop();
}

impl NpuFfn {
    /// Ends the device thread and **waits for it**.
    ///
    /// Idempotent: each step is a `take`, so calling it twice — or dropping a
    /// service [`shutdown`] has already stopped — does nothing the second
    /// time.
    fn stop(&self) {
        // Dropping the sender is what ends the loop; taking it first means a
        // request arriving during shutdown is declined rather than queued
        // onto a thread that is going away.
        if let Ok(mut sender) = self.jobs.lock() {
            sender.take();
        }
        if let Ok(mut thread) = self.thread.lock()
            && let Some(handle) = thread.take()
        {
            // A join that fails is a thread that already panicked, which it
            // has reported for itself; there is nothing left to wait for
            // either way.
            let _ = handle.join();
        }
    }
}

/// **Every service joins its thread, not just the installed one.**
///
/// [`shutdown`] can only reach the one in `SERVICE`, and `NpuFfn` is also
/// built outside that static: the prefill-width probe opens one to time a
/// single block at each candidate width and drops it once it has an answer,
/// and `install` drops the service it is handed if one is already set.
///
/// Dropping ended the loop and then **detached** the thread, because dropping
/// a `JoinHandle` does not wait. That leaves a thread unwinding `NpuRuntime`
/// and its graphs — vendor C++ objects — with nothing keeping the process
/// alive until it finishes, which is the hazard the [`NpuFfn::thread`] field
/// documents.
///
/// # What this does not fix
///
/// It does **not** fix the `pure virtual method called` abort seen on Ctrl+C
/// during a first run. That was measured against this change and is unmoved
/// by it: the cause is the detached `orangu-npu-prepare` thread, whose
/// `JoinHandle` is discarded at its spawn, still inside vendor code when
/// `main` returns. This closes a real and separate hole — a detached
/// device thread — and the two should not be confused.
impl Drop for NpuFfn {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The installed service, or `None` on a machine that has no NPU, no cached
/// blocks for this model, or where nothing would load.
pub fn service() -> Option<&'static NpuFfn> {
    if serving_disabled() {
        return None;
    }
    SERVICE.get()?.as_ref()
}

/// `ORANGU_NPU_FFN=0` keeps every compiled block off the device.
///
/// The blocks are still compiled and cached — only the serving stops — so
/// the two answers differ in where the feed-forward runs and in nothing
/// else. That is the point: comparing two *builds* means comparing things
/// that differ by more than the question, which on this board has produced
/// more wrong answers than right ones.
fn serving_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ORANGU_NPU_FFN").is_ok_and(|v| v == "0"))
}

#[cfg(test)]
mod tests {
    /// **The compile pass reads this, so the key has to match the one the
    /// binding pass writes.** They are in different crates — the limit is
    /// learned in `NpuFfn::open` here and consulted by `precompile` in the
    /// server binary — and a key that disagreed would not fail: it would
    /// silently answer "no limit known" and go on compiling blocks the device
    /// has already refused, which is the waste this exists to stop.
    #[test]
    fn a_written_capacity_limit_is_read_back_for_the_same_model_and_widths() {
        let dir = std::env::temp_dir().join(format!("orangu-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fingerprint = "deadbeefcafe0001";
        let path = dir.join(format!("graphs-{fingerprint}-128"));
        super::write_capacity_hint(&path, 34);

        assert_eq!(
            super::read_capacity_hint(&path),
            Some(34),
            "the limit must survive the round trip"
        );
        // The path the public accessor derives must be the one just written,
        // modulo the cache directory it is rooted in.
        assert_eq!(
            super::capacity_hint_path(fingerprint, &[128])
                .file_name()
                .and_then(|n| n.to_str()),
            path.file_name().and_then(|n| n.to_str()),
            "the compile side and the binding side must agree on the file name"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A limit of zero is not a limit, it is a file that says the device holds
    /// nothing — which would stop every block from being compiled. It must
    /// read as absent.
    #[test]
    fn a_zero_capacity_limit_is_ignored() {
        let dir = std::env::temp_dir().join(format!("orangu-cap0-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graphs-x-128");
        super::write_capacity_hint(&path, 0);
        assert_eq!(super::read_capacity_hint(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Different widths are different limits: what the device holds at one
    /// width says nothing about another, so the names must not collide.
    #[test]
    fn different_width_sets_get_different_files() {
        let one = super::capacity_hint_path("abc", &[128]);
        let two = super::capacity_hint_path("abc", &[128, 1]);
        assert_ne!(one, two);
        assert!(two.to_string_lossy().contains("128x1"));
    }

    /// `shutdown` is reachable on a machine with no NPU and no service
    /// installed, and survives being called twice — `main` calls it on
    /// every exit path, including the ones that never touched the device.
    #[test]
    fn shutdown_without_a_device_is_a_no_op() {
        super::shutdown();
        super::shutdown();
    }
}

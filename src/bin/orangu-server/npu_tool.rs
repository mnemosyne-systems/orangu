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

//! Preparing a model's feed-forward blocks for the NPU.
//!
//! [`prepare_in_background`] is the entry point that matters: the server
//! calls it and carries straight on serving. On its own thread it notices
//! what is not compiled yet, compiles it, and binds the result. The operator
//! never asks for any of this — there is nothing to remember and no command
//! to run first, and nothing waits on it: until the thread finishes the
//! model simply has no NPU and every layer takes the path it would have
//! taken anyway.
//!
//! The work happens in a **child process**, and that is not an
//! implementation detail. Compiling loads the vendor's ONNX Runtime provider
//! and executing loads its NOE runtime; both carry their own copy of the
//! same user-mode driver, and whichever is loaded first captures the other's
//! symbol bindings. In one process that produces wrong answers rather than
//! errors — see [`orangu::npu_ort`]. So the server re-runs its own
//! executable to compile, and only ever *reads* the cache itself.
//!
//! **Sharded models see only one shard.** Everything here reads a single
//! GGUF, and a multi-part model's tensor table only covers the tensors
//! stored in that part. Given part 1 of 3 of a 28-layer model,
//! [`discover_blocks`] finds the 13 complete blocks that live there and the
//! rest are invisible. That is partial coverage rather than a wrong answer
//! — the budget path already treats a model it cannot fully cover as a
//! partial speedup — but it is not a decision anyone made, and a sharded
//! model quietly gets less of the device than it could.
//!
//! [`compile`] and [`run`] are the two halves, reachable as hidden
//! subcommands. They are hidden because they are how this module talks to
//! itself across that process boundary, not something anyone should have to
//! invoke.

use anyhow::{Context, Result, bail};
use std::path::Path;

use orangu::gguf::GgufFile;
use orangu::npu_cache::{BlockKey, NpuCache, fingerprint};

/// The prefill width language-model blocks are compiled for.
///
/// A compiled graph has one static shape, so the device can only serve a
/// prefill of exactly this many tokens; anything else takes the CPU or GPU
/// path unchanged.
///
/// The token widths worth compiling **for this model**.
///
/// Not a constant, because the answer is a property of the checkpoint rather
/// than of the machine or of this project. Two widths matter, and whether a
/// model can use the first of them is read from its tensors:
///
/// - `1` is a decode step. It needs the layer to *stop* at `ffn_norm`, run
///   the network elsewhere and finish on the host — see
///   `FusedPostAttentionInput::stop_at_ffn_norm` and
///   `LlamaModel::record_decode_run`. That only works when nothing else is
///   recorded after the network in the same fused block, which is every
///   llama-style model and no gemma one.
/// - `16` is what this engine's prefill chunker issues, and the width the
///   gemma prefill hook asks for.
///
/// **Prefill always, decode when [`decode_enabled`] asks.** Prefill comes
/// first in the list because `precompile` walks it in order against a cache
/// budget, and the two widths are not worth the same per byte: prefill is
/// worth 2.8x and decode is worth slightly less than nothing.
///
/// This spent a long time returning the prefill width and only that, under
/// a long argument that decode was both slower and less accurate. The
/// accuracy half of that argument was measuring a compile bug — a width-1
/// graph was calibrated on a one-row sample, see [`CALIBRATION_ROWS`] — and
/// the speed half was measured against a decode step whose feed-forward
/// share the GPU has since taken most of. The numbers that replaced it are
/// in [`decode_enabled`].
///
/// This used to return nothing at all for llama-style models, because a
/// llama-style prefill never consulted the device and width 16 would have
/// been gigabytes of blocks nobody asked for. `LlamaModel::run_layers` now
/// asks, exactly as gemma's prefill does — it declines the fused
/// post-attention chain when the device holds the layer, which is the only
/// thing that was keeping the whole family off it. It earns its keep:
/// qwen3 4B at `Q8_0` prefills a 720-token prompt at 7.15 tok/s against
/// 2.72 with the device off, and answers correctly.
pub fn widths_for(gguf: &GgufFile) -> Vec<usize> {
    // **Prefill first**, because `precompile` walks this list in order and a
    // budget that runs out should run out on the width worth less. Prefill
    // is worth 2.8x on this machine and decode is a small loss, so a byte of
    // cache spent on a prefill block is the only one that pays. This used to
    // be the other way around, from when decode was the experiment and
    // prefill the thing that already worked.
    let mut widths = Vec::with_capacity(2);
    widths.push(prefill_width());
    if decode_enabled() && decode_seam_usable(gguf) {
        widths.push(DECODE_WIDTH);
    }
    widths
}

/// Whether to compile the decode width: **on when this server runs more
/// than one slot**, off at one, and `ORANGU_NPU_DECODE` overrides either
/// way.
///
/// **The slot count is the whole answer, and it took three wrong verdicts
/// to find that.** The first said decode was inaccurate; it was measuring
/// the synthetic calibration that had every block returning nothing. The
/// second said the same after that was fixed and blamed the KV cache; it
/// was measuring `CALIBRATION_ROWS`, a width-1 graph built from a one-row
/// sample. The third said decode was simply slower, and it was right about
/// what it measured — one request at a time.
///
/// A single decode stream is strictly serial: attention on the GPU, then
/// the network on the device, then the next layer's attention. One of the
/// two is always idle, and the seam's cost is paid with nothing to hide it
/// behind. `LlamaModel::record_decode_run` prices that cost under
/// `ORANGU_NPU_TIME` — 112.6 ms a step reading back and 110.3 ms on the
/// device, against the 186.7 ms of feed-forward it takes off the GPU — and
/// `VulkanBackend::record_readback_split` shows 3374 us of the 3.76 ms
/// readback is the fence, not the copy. There is no tuning left in that.
///
/// Give it a second generation and the idle stops. Llama 3.2 1B `Q8_0`,
/// aggregate tokens per second over concurrent requests, two runs:
///
/// | slots | decode on the GPU | decode on the NPU |
/// |---|---|---|
/// | 1 | 7.30, 7.25 | 6.93, 6.71 |
/// | 2 | 7.95, 7.94 | **11.24, 11.26** |
///
/// The GPU column is why `Role::default_slots` returns 1: a second slot
/// buys 9% because both slots stream the same weights through the same
/// device. The NPU column is a different machine — 1.62x from the second
/// slot and **1.42x over the GPU** at two — because the two slots are no
/// longer queueing for one device. That is the whole point of having an
/// NPU, and it is invisible to any measurement that sends one request at a
/// time.
///
/// **That advantage is gone, and this project removed it.** The table above
/// was measured when this card's `Q8_0` matvec ran at 9.5 GB/s; rewriting
/// its block-hoisted kernel to read a word at a time took it to 24.3 and
/// llama-3.2-1B `Q8_0` decode from 8.45 to 17.41 tok/s. Re-measured after
/// that, on the same model at two slots, with `ffn_blocks` confirming all
/// sixteen decode blocks bound each time:
///
/// | | aggregate tok/s |
/// | --- | --: |
/// | device | 15.77, 15.59 |
/// | GPU | **21.94, 18.06** |
///
/// So the device is now the slower of the two for decode, by 15-28%, and
/// the default is off. `ORANGU_NPU_DECODE=1` restores it for a machine
/// whose GPU is weaker than this one's — which is the case the original
/// table describes, and it has not stopped being true there.
///
/// The compile budget follows the same answer: with decode off the width is
/// not compiled at all and [`widths_for`] returns prefill alone.
fn decode_enabled() -> bool {
    match std::env::var("ORANGU_NPU_DECODE") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        ),
        // Unset: off. It was `slots > 1` — on exactly when there is a
        // second generation to keep the GPU busy while the device works —
        // and that was right until the GPU stopped being slow.
        Err(_) => false,
    }
}

/// How many generations this server runs at once, from `[orangu-server]
/// slots`. One until `set_slots` says otherwise, which is what a process
/// that never configured it should assume.
static SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Records the configured slot count for [`decode_enabled`].
///
/// Called once from `main` before preparation starts. A static rather than
/// a parameter because `widths_for` is reached from the compile child, the
/// serving process and the tool, and only one of those has a `Config` in
/// its hand.
pub fn set_slots(slots: usize) {
    SLOTS.store(slots.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// One token: a decode step, served through the seam that stops a fused
/// layer at `ffn_norm` and finishes it on the host.
///
/// Only for an architecture that has nothing recorded after the network in
/// the same fused block — see [`decode_seam_usable`], which reads that from
/// the file rather than from a list of names.
pub const DECODE_WIDTH: usize = 1;

/// Whether a layer of this model can stop at `ffn_norm` and be finished off
/// the device.
///
/// Read from the tensors rather than matched on an architecture name: what
/// matters is whether anything is recorded *after* the feed-forward network
/// in the same fused block, and the tensors say so directly. A per-layer
/// embedding (`per_layer_*`) is projected after it; a post-FFN or
/// post-attention norm sits around it. Gemma has all three, llama has none,
/// and a checkpoint this has never seen is judged on the same evidence
/// rather than on whether someone remembered to add it to a list.
fn decode_seam_usable(gguf: &GgufFile) -> bool {
    !gguf.tensors.iter().any(|t| {
        t.name.contains("per_layer")
            || t.name.ends_with("post_ffw_norm.weight")
            || t.name.ends_with("post_attention_norm.weight")
    })
}

/// The prefill chunk this engine issues, and so the only width a block is
/// ever compiled for.
///
/// A graph on this device has one static shape, so this is part of a
/// block's cache identity: a request at any other width finds nothing and
/// goes to the CPU or GPU as before.
pub const PREFILL_WIDTH: usize = 16;

/// The prefill width to compile for, `ORANGU_NPU_PREFILL_WIDTH` overriding
/// [`PREFILL_WIDTH`].
///
/// A measurement knob first: the device is not equally efficient at every
/// width, and 16 is what this engine's chunk sizer happens to issue rather
/// than what the hardware likes. One block of llama 3.2 3B `Q8_0`, per
/// token on the device:
///
/// | width | per token | rate |
/// |---|---|---|
/// | 1 | 3.27 ms | 46 GF/s |
/// | 16 | 0.342 ms | 441 GF/s |
/// | 32 | 0.415 ms | 364 GF/s |
/// | 64 | **0.256 ms** | **590 GF/s** |
/// | 128 | 0.281 ms | 537 GF/s |
///
/// Whether that 1.34x survives end to end is a different question, because
/// a wider graph only gets used if `prefill_in_chunks` issues chunks that
/// wide, and its sizer is bounded by `ORANGU_PREFILL_CHUNK_MS`. The two
/// have to move together, which is what this knob is for.
pub fn prefill_width() -> usize {
    if let Some(forced) = forced_prefill_width() {
        return forced;
    }
    match MEASURED_WIDTH.load(std::sync::atomic::Ordering::Relaxed) {
        0 => PREFILL_WIDTH,
        measured => measured,
    }
}

/// `ORANGU_NPU_PREFILL_WIDTH`, when an operator has pinned the width.
fn forced_prefill_width() -> Option<usize> {
    static FORCED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("ORANGU_NPU_PREFILL_WIDTH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|w| *w > 1)
    })
}

/// What [`probe_prefill_width`] measured on this device, or `0` before it
/// has run.
static MEASURED_WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Widths [`probe_prefill_width`] times.
///
/// Powers of two from the old fixed value up. Below 16 the device is
/// latency-bound and never wins; above 128 a chunk starts to be a large GPU
/// submission for everything the device is *not* doing, and the measured
/// curve has turned by then.
///
/// 128 was briefly removed from this list because it crashed a model, and
/// the crash was worth having: `attention::gpu_split_decode` handed a
/// pool-backed layer to a split-attention kernel that binds the
/// per-request mirror and has no block table, so `KvLayer::sync_gpu`
/// refused with `its mirror holds no rows`. Only a wide chunk reached it —
/// a narrow prefill never took the paged range path, so the layer was never
/// marked — which is why granite 3.1 2B `Q8_0` had passed at 16 for as long
/// as 16 was the only width. Fixed where the choice is made rather than
/// bounded here.
const WIDTH_CANDIDATES: [usize; 4] = [16, 32, 64, 128];

/// Where a measured width is remembered, keyed by the model.
///
/// **Per model, not per machine**, which is the correction to how this
/// first shipped. The best width is a property of the device *and* of the
/// block shape it is running, so caching one answer machine-wide measured
/// the first model to be loaded and then applied its answer to every other
/// one. A 33-model sweep did exactly that — granite 3.1 2B chose 128 and
/// the other 32 inherited it — and while 27 models got faster by up to
/// 5.1x, three large `Q8_0` checkpoints went *backwards*: Qwen2.5-Coder-7B
/// 2.87 -> 1.64 tok/s, Meta-Llama-3.1-8B 2.54 -> 1.50, granite-3.1-8B
/// 1.95 -> 1.26. Measuring once and generalising is the same mistake as a
/// hardcoded width with an extra step in front of it.
fn probe_cache_path(fingerprint: &str) -> std::path::PathBuf {
    NpuCache::default_dir().join(format!("prefill-width-{fingerprint}"))
}

/// Measures which prefill width this device runs fastest **per token**, and
/// remembers it.
///
/// **This is the number that was most wrong, and it was wrong because it
/// was a constant.** 16 was never a property of the hardware: it is what
/// this engine's chunk sizer happened to issue, so the graphs were built
/// for the chunker instead of the chunker being told what the device wants.
/// One block of llama 3.2 3B `Q8_0`, per token on the device:
///
/// | width | per token | rate |
/// |---|---|---|
/// | 16 | 0.342 ms | 441 GF/s |
/// | 32 | 0.415 ms | 364 GF/s |
/// | 64 | **0.256 ms** | **590 GF/s** |
/// | 128 | 0.281 ms | 537 GF/s |
///
/// End to end that is worth more than the device-side ratio suggests,
/// because a wider graph also means fewer chunks and so fewer submissions:
/// a 980-token prompt went from 62 submissions to 17 and from 10.35 tok/s
/// to **23.46**, which is 6.4x the 3.65 tok/s the same prompt gets with the
/// device off.
///
/// The curve is not monotone — 32 is worse than 16 here — so it has to be
/// measured rather than reasoned about, and it is a property of the device
/// rather than of the model, which is why the answer is cached beside the
/// blocks and probed once.
fn probe_prefill_width(model: &Path, executable: &Path) -> Option<usize> {
    let gguf = open_model(model)?;
    let fingerprint = fingerprint(&gguf);
    if let Ok(text) = std::fs::read_to_string(probe_cache_path(&fingerprint))
        && let Ok(width) = text.trim().parse::<usize>()
        && WIDTH_CANDIDATES.contains(&width)
    {
        return Some(width);
    }
    let blocks = discover_blocks(&gguf, model, Some("blk."));
    let block = blocks.first()?;
    let index: usize = block.prefix.strip_prefix("blk.")?.parse().ok()?;
    let activation = host_activation(&gguf);

    for width in WIDTH_CANDIDATES {
        if compile_child(executable, model, width, 1).is_err() {
            return None;
        }
    }
    if stopping() {
        return None;
    }
    let names = vec![(index, block.prefix.clone())];
    let service =
        orangu::npu_ffn::NpuFfn::open(model, &names, &WIDTH_CANDIDATES, calibration_digest())?;

    let mut best: Option<(usize, f64)> = None;
    for width in WIDTH_CANDIDATES {
        if !service.has(index, width) {
            continue;
        }
        let x = calibration_for(model, block, width);
        let mut out = Vec::new();
        // One run to warm the graph, then a few to time.
        if !service.forward_into(index, width, &x, &mut out) {
            continue;
        }
        let started = std::time::Instant::now();
        const RUNS: usize = 4;
        for _ in 0..RUNS {
            service.forward_into(index, width, &x, &mut out);
        }
        let per_token = started.elapsed().as_secs_f64() / (RUNS as f64) / (width as f64) * 1000.0;
        eprintln!("orangu-server: [npu] width {width}: {per_token:.3} ms/token");
        if best.is_none_or(|(_, seen)| per_token < seen) {
            best = Some((width, per_token));
        }
    }
    let _ = activation;
    let (width, _) = best?;
    let _ = std::fs::write(probe_cache_path(&fingerprint), width.to_string());
    Some(width)
}

/// A feed-forward block found in a GGUF.
#[derive(Debug)]
struct BlockSpec {
    /// Tensor-name prefix, e.g. `v.blk.0`.
    prefix: String,
    /// Which triple of tensors under `prefix` this block is.
    names: FfnNames,
    d_model: usize,
    d_ff: usize,
    /// The activation range the model ships, where it ships one.
    range: Option<(f32, f32)>,
}

impl BlockSpec {
    /// A tensor of this block, by role.
    fn tensor(&self, role: FfnRole) -> String {
        format!("{}.{}.weight", self.prefix, self.names.suffix(role))
    }

    /// This block's weights for `role`, in `f32`.
    ///
    /// The one place that knows a fused block's gate and up are the two
    /// halves of one tensor, so nothing downstream — the compiler, the
    /// reference, the probe — has to.
    fn weights(&self, path: &Path, role: FfnRole) -> Result<Vec<f32>> {
        let raw = read_dequantized(path, &self.tensor(role))?;
        if !self.names.fused_gate_up || role == FfnRole::Down {
            return Ok(raw);
        }
        let half = self.d_ff * self.d_model;
        if raw.len() < half * 2 {
            bail!(
                "{} is {} values, too few for two {}x{} halves",
                self.tensor(role),
                raw.len(),
                self.d_ff,
                self.d_model
            );
        }
        // Gate first — see `FfnNames::fused_gate_up`.
        Ok(match role {
            FfnRole::Gate => raw[..half].to_vec(),
            _ => raw[half..half * 2].to_vec(),
        })
    }

    /// The name this block is cached under. The prefix alone is not enough:
    /// a model may hold a dense feed-forward *and* a mixture-of-experts
    /// shared expert at the same layer, and they are different programs.
    fn cache_name(&self) -> String {
        format!("{}{}", self.prefix, self.names.tag)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FfnRole {
    Gate,
    Up,
    Down,
}

/// The three tensors a gated feed-forward block is made of, named.
///
/// Two shapes qualify. A dense block is `ffn_gate`/`ffn_up`/`ffn_down`. A
/// mixture-of-experts layer's **shared expert** is the same arithmetic under
/// `ffn_*_shexp`: unlike the routed experts beside it, it runs for every
/// token, which is what makes it compilable at all — a graph has one static
/// shape and no way to select an expert per row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FfnNames {
    gate: &'static str,
    up: &'static str,
    down: &'static str,
    /// Appended to the prefix in the cache key, so the two never collide.
    tag: &'static str,
    /// Whether `gate` and `up` are two halves of one tensor rather than two
    /// tensors. `phi3` ships `ffn_up.weight` as `[n_embd, 2 * n_ff]` with no
    /// `ffn_gate.weight` at all — **gate first**, which is
    /// `ggml_vec_swiglu_f32`'s `swapped = 0` and the order
    /// `arch::phi::PhiLayer::ffn_gate_up` splits it in. Getting that
    /// backwards produces fluent, wrong text rather than an error.
    fused_gate_up: bool,
}

impl FfnNames {
    fn suffix(&self, role: FfnRole) -> &'static str {
        match role {
            FfnRole::Gate => self.gate,
            FfnRole::Up => self.up,
            FfnRole::Down => self.down,
        }
    }
}

/// A dense gated feed-forward block.
const FFN_DENSE: FfnNames = FfnNames {
    gate: "ffn_gate",
    up: "ffn_up",
    down: "ffn_down",
    tag: "",
    fused_gate_up: false,
};

/// A mixture-of-experts layer's always-on shared expert.
const FFN_SHARED_EXPERT: FfnNames = FfnNames {
    gate: "ffn_gate_shexp",
    up: "ffn_up_shexp",
    down: "ffn_down_shexp",
    tag: "#shexp",
    fused_gate_up: false,
};

/// `phi3`, whose gate and up are one tensor. Named last in [`FFN_SHAPES`]
/// and guarded on the *absence* of a separate `ffn_gate`, so a model that
/// has both is read as dense.
const FFN_FUSED_GATE_UP: FfnNames = FfnNames {
    gate: "ffn_up",
    up: "ffn_up",
    down: "ffn_down",
    tag: "",
    fused_gate_up: true,
};

/// Every feed-forward shape this compiler knows how to take.
const FFN_SHAPES: &[FfnNames] = &[FFN_DENSE, FFN_SHARED_EXPERT, FFN_FUSED_GATE_UP];

/// This machine's NPU, probed once.
///
/// Cached because more than one caller wants it — the startup inventory and
/// `/props` — and each probe `dlopen`s the vendor driver, which is not a
/// thing to do concurrently with the thread that is using it.
pub fn npu_info() -> Option<&'static orangu::npu::NpuInfo> {
    static INFO: std::sync::OnceLock<Option<orangu::npu::NpuInfo>> = std::sync::OnceLock::new();
    INFO.get_or_init(orangu::npu::detect_npu).as_ref()
}

/// What `/props` says about the NPU, or `null` on a machine without one.
///
/// **So that a benchmark result can say which processors produced it.**
/// `orangu-bench` archives `/props` with every run, and until this existed a
/// stored result could not be told apart from one measured with the device
/// switched off — which, given the device is worth about a factor of two on
/// prefill, is the single most important thing about the number.
///
/// Reported live rather than stamped at startup: the service installs
/// asynchronously, so a run that begins before the blocks are compiled and
/// one that begins after are genuinely different measurements.
pub fn npu_props() -> serde_json::Value {
    let Some(info) = npu_info() else {
        return serde_json::Value::Null;
    };
    let service = orangu::npu_ffn::service();
    serde_json::json!({
        "vendor": info.vendor,
        "target": info.target,
        "cores": info.cores,
        "in_use": service.is_some(),
        "blocks": service.map_or(0, orangu::npu_ffn::NpuFfn::len),
        "widths": service.map(orangu::npu_ffn::NpuFfn::widths),
        // Per width, and feed-forward only: `blocks` and `widths` count
        // attention blocks too, so they can say "width 1" about a server
        // whose decode seam has nothing to run.
        "ffn_blocks": service.map(|npu| {
            let mut per_width = serde_json::Map::new();
            for w in npu.widths() {
                per_width.insert(w.to_string(), npu.ffn_blocks_at(w).into());
            }
            per_width
        }),
    })
}

/// Every shard of `path`, in order, each with the file it came from.
///
/// A single-file model is one entry. A split one is all of them, which is
/// the point: `GgufFile::open` reads the tensor table of the file it was
/// given, so a split model previously showed this module only the layers
/// that happened to live in the first shard. Qwen2.5-Coder 7B `Q8_0`
/// compiled **28** feed-forward blocks as a single file and **12** as
/// `00001-of-00003` — the same weights, two thirds of the layers left on
/// the GPU, and nothing said so.
fn model_shards(path: &Path) -> Vec<(GgufFile, std::path::PathBuf)> {
    let Ok(first) = GgufFile::open(path) else {
        return Vec::new();
    };
    let Ok(paths) = crate::engine::loader::shard_paths(path, &first) else {
        return vec![(first, path.to_path_buf())];
    };
    if paths.len() <= 1 {
        return vec![(first, path.to_path_buf())];
    }
    let mut shards = Vec::with_capacity(paths.len());
    for shard in paths {
        match GgufFile::open(&shard) {
            Ok(gguf) => shards.push((gguf, shard)),
            // A shard that will not open is one the engine would refuse to
            // run anyway; reporting it here would duplicate that.
            Err(_) => return vec![(first, path.to_path_buf())],
        }
    }
    shards
}

/// A view of every shard's tensors under one [`GgufFile`].
///
/// **Names and shapes only — never read tensor data through this.** Each
/// shard has its own `data_offset` and `alignment`, and this carries the
/// first shard's, so an offset computed here is wrong for every tensor that
/// lives elsewhere. [`read_dequantized`] resolves a name to the shard that
/// owns it and reads through *that* one.
fn merged_tensor_view(shards: Vec<(GgufFile, std::path::PathBuf)>) -> Option<GgufFile> {
    let mut shards = shards.into_iter();
    let (mut merged, _) = shards.next()?;
    for (gguf, _) in shards {
        merged.tensors.extend(gguf.tensors);
    }
    Some(merged)
}

/// The whole model's tensor table, however many files it is split across.
fn open_model(path: &Path) -> Option<GgufFile> {
    merged_tensor_view(model_shards(path))
}

/// Every `<prefix>.ffn_{gate,up,down}.weight` triple in the file.
///
/// Only complete triples: a block missing one of the three is not a gated
/// feed-forward network this can compile, and skipping it quietly is better
/// than guessing at what it is.
fn discover_blocks(gguf: &GgufFile, path: &Path, filter: Option<&str>) -> Vec<BlockSpec> {
    let mut blocks = Vec::new();
    for tensor in &gguf.tensors {
        for &names in FFN_SHAPES {
            let Some(prefix) = tensor.name.strip_suffix(&format!(".{}.weight", names.gate)) else {
                continue;
            };
            if filter.is_some_and(|f| !prefix.contains(f)) {
                continue;
            }
            let has = |suffix: &str| {
                gguf.tensors
                    .iter()
                    .any(|t| t.name == format!("{prefix}.{suffix}.weight"))
            };
            if !has(names.up) || !has(names.down) || tensor.dims.len() < 2 {
                continue;
            }
            // A fused block is one whose gate is not a tensor of its own.
            // Without this a `phi3` file would match twice — once as fused,
            // once as whatever the dense arm made of the same `ffn_up` — and
            // a model with a real `ffn_gate` would be read as fused.
            if names.fused_gate_up && has("ffn_gate") {
                continue;
            }
            // Half the output width is the gate's, half the up's; an odd
            // width is not this shape at all.
            let d_ff = if names.fused_gate_up {
                if !(tensor.dims[1] as usize).is_multiple_of(2) {
                    continue;
                }
                tensor.dims[1] as usize / 2
            } else {
                tensor.dims[1] as usize
            };
            let range = match (
                gguf.read_scalar(path, &format!("{prefix}.{}.input_min", names.gate)),
                gguf.read_scalar(path, &format!("{prefix}.{}.input_max", names.gate)),
            ) {
                (Some(lo), Some(hi)) if hi > lo => Some((lo, hi)),
                _ => None,
            };
            blocks.push(BlockSpec {
                prefix: prefix.to_string(),
                names,
                d_model: tensor.dims[0] as usize,
                d_ff,
                range,
            });
        }
    }
    blocks
}

/// How far a block's device output is from the same block computed in
/// `f32`, as a percentage of the reference's own magnitude.
///
/// `None` when the weights cannot be read. The reference is the *whole*
/// block — both projections, the activation, and `down` — so this is the
/// error a caller actually inherits, not a per-stage figure.
fn reference_error(
    path: &Path,
    block: &BlockSpec,
    tokens: usize,
    x: &[f32],
    got: &[f32],
    activation: orangu::npu_ort::HostActivation,
) -> Option<Deviation> {
    let gate = block.weights(path, FfnRole::Gate).ok()?;
    let up = block.weights(path, FfnRole::Up).ok()?;
    let down = block.weights(path, FfnRole::Down).ok()?;

    let project = |input: &[f32], w: &[f32], in_dim: usize, out_dim: usize| {
        let mut out = vec![0.0f32; tokens * out_dim];
        for t in 0..tokens {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += input[t * in_dim + i] * w[o * in_dim + i];
                }
                out[t * out_dim + o] = acc;
            }
        }
        out
    };
    let g = project(x, &gate, block.d_model, block.d_ff);
    let u = project(x, &up, block.d_model, block.d_ff);
    let hidden: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| activation.apply_host(*g) * u)
        .collect();
    let want = project(&hidden, &down, block.d_ff, block.d_model);

    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (w, g)| m.max((w - g).abs()));
    let sq = |acc: f64, v: f32| acc + (v as f64) * (v as f64);
    let signal = want.iter().fold(0.0f64, |a, v| sq(a, *v));
    let noise = want.iter().zip(got).fold(0.0f64, |a, (w, g)| sq(a, w - g));
    // What is left after the best single rescaling of the device's answer.
    // `alpha` minimizes `||alpha * got - want||`, so this separates an error
    // in the *magnitude* of the block's output — which one `uint8` scale per
    // matrix is expected to produce, and which one number per block could
    // undo — from error in its shape, which nothing cheap can.
    let dot = want
        .iter()
        .zip(got)
        .fold(0.0f64, |a, (w, g)| a + (*w as f64) * (*g as f64));
    let energy = got.iter().fold(0.0f64, |a, g| sq(a, *g));
    let scaled = (energy > 0.0).then(|| {
        let alpha = dot / energy;
        let residual = want.iter().zip(got).fold(0.0f64, |a, (w, g)| {
            let d = *w as f64 - alpha * (*g as f64);
            a + d * d
        });
        (100.0 * (residual / signal).sqrt() as f32, alpha as f32)
    });

    (peak > 0.0 && signal > 0.0).then(|| Deviation {
        rms: 100.0 * (noise / signal).sqrt() as f32,
        peak: 100.0 * worst / peak,
        rescaled: scaled.map(|(rms, _)| rms).unwrap_or(f32::NAN),
        alpha: scaled.map(|(_, a)| a).unwrap_or(f32::NAN),
    })
}

/// How far a block's output on the device is from the same block in `f32`,
/// by two measures that disagree by a factor of three.
///
/// `rms` is the error energy over the signal energy, the ordinary relative
/// error, and the one that decides. `peak` is the single worst element over
/// the reference's largest — a maximum over forty thousand numbers, one
/// outlier deep.
///
/// **They disagree, and `peak` is the one that misled.** gemma 4 E2B at
/// `Q8_0` reproduces the reference token stream exactly on a short prompt,
/// and its blocks measure like this on `peak`:
///
/// | block | 0 | 4 | 13 | 16 | 32 |
/// |---|---|---|---|---|---|
/// | peak | 3.6% | 1.7% | 12.0% | 16.1% | 12.5% |
///
/// A model three times over a 5% bar in four of its blocks, reproducing the
/// reference stream character for character. `peak` is a tail statistic: it
/// grows with the size of the tensor it is taken over, so a wider model
/// looks worse for having more elements rather than worse arithmetic, and a
/// single unlucky element condemns a block whose other 40,959 are fine.
///
/// The block's output is then *added to a residual stream* that is larger
/// than it, which dilutes error by a further factor nothing here measures.
/// Between the two, `peak` at 16% is not a claim that anything is 16%
/// wrong.
#[derive(Clone, Copy, Debug)]
struct Deviation {
    rms: f32,
    peak: f32,
    /// `rms` after the best single rescaling of the device's output, and the
    /// scale that achieved it. If these are far apart, most of what the
    /// device costs is a magnitude the host could correct with one multiply
    /// per block.
    rescaled: f32,
    alpha: f32,
}

/// One tensor as `f32`, through the engine's own dequantizer.
///
/// Not `GgufFile::read_tensor`, which handles `F32`, `F16` and `Q8_0` and
/// nothing else — and `Q4_K_M` is the quantization most models ship in, so
/// that reader covers the case that matters least. `engine::quant` already
/// knows every type this project reads, and it lives in this binary, so the
/// fix is to call it rather than teach the library reader K-quants a second
/// time.
fn read_dequantized(path: &Path, name: &str) -> Result<Vec<f32>> {
    // **Resolved to the shard that owns it.** Taking a `&GgufFile` from the
    // caller invited reading through `merged_tensor_view`, whose offsets are
    // the first shard's and wrong for everything else. Finding the file here
    // means a caller cannot get that wrong.
    for (gguf, shard) in model_shards(path) {
        if gguf.tensors.iter().any(|t| t.name == name) {
            return read_dequantized_from(&gguf, &shard, name);
        }
    }
    anyhow::bail!("{name} is not in {}", path.display())
}

fn read_dequantized_from(gguf: &GgufFile, path: &Path, name: &str) -> Result<Vec<f32>> {
    use std::io::{Read, Seek, SeekFrom};
    let info = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow::anyhow!("{name} is not in {}", path.display()))?;
    let (block_bytes, block_elems) = crate::engine::quant::block_layout(info.ggml_type)
        .ok_or_else(|| {
            anyhow::anyhow!("{name} is {} , which has no block layout", info.ggml_type)
        })?;
    let elements = info.element_count() as usize;
    let len = elements / block_elems * block_bytes;
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(gguf.data_offset + info.offset))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    crate::engine::quant::dequantize(info.ggml_type, &bytes, elements)
}

/// The gate activation this architecture uses.
///
/// Gemma's gated blocks are GELU; llama, mistral, qwen and phi are SiLU,
/// which is also the sane default for an architecture not named here — every
/// SwiGLU descendant of llama is SiLU, and a block compiled with the wrong
/// one is wrong rather than slow, so this is worth reading from the file
/// rather than assuming one.
fn host_activation(gguf: &GgufFile) -> orangu::npu_ort::HostActivation {
    use orangu::npu_ort::HostActivation;
    let arch = gguf
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .map(|(_, v)| v.display(1))
        .unwrap_or_default();
    if arch.contains("gemma") {
        HostActivation::Gelu
    } else {
        HostActivation::Silu
    }
}

/// Calibration input spread across `range`.
///
/// **The fallback, for a model whose `ffn_norm` cannot be read.** Static
/// quantization wants activations from real inputs; this synthesizes them
/// from the range alone, uniformly, which is the worst case for
/// quantization error rather than a typical one. Prefer
/// [`realistic_calibration`], which is shaped like what a block is really
/// fed.
fn synthetic_calibration(d_model: usize, rows: usize, range: (f32, f32)) -> Vec<f32> {
    let mut state = 0x51ed_1234u64;
    (0..rows * d_model)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = (state >> 33) as f32 / (1u64 << 31) as f32;
            range.0 + unit * (range.1 - range.0)
        })
        .collect()
}

/// Calibration shaped like what the block is actually fed.
///
/// Every gated block in every architecture here reads the output of an
/// RMSNorm, and that output is not free-form: RMSNorm divides each token row
/// by its own RMS and then multiplies by a learned per-channel weight. So
/// two things are known about the real input without running the model at
/// all — each row has unit RMS, and the *shape* across channels is the
/// block's own `ffn_norm.weight`, which is sitting in the file.
///
/// That matters because the device quantizes activations to `uint8` at one
/// scale, set from the calibration. Calibrating on uniform noise in
/// `[-1, 1]` — which is what every model got, since no GGUF carries the
/// `input_min`/`input_max` metadata [`discover_blocks`] looks for — sets
/// that scale from a distribution real activations do not have: flat where
/// they are peaked, and identical for every model where the per-channel
/// weight makes them very different. A model with a few large `ffn_norm`
/// channels has activations reaching far past 1 in those channels and
/// hugging zero elsewhere, and a scale calibrated on flat noise spends its
/// 256 levels in the wrong place.
///
/// Gaussian rather than uniform for the same reason: it is what a
/// normalized hidden state looks like.
///
/// **And its tail is nothing like one, which is the part that mattered.**
/// Drawing every channel from the same Gaussian gives a largest value about
/// 4.2 times the row RMS. A real gemma 4 E2B hidden state, captured from a
/// 720-token prefill by `engine::dump_ffn_input`, reaches **38.9** times —
/// a handful of channels sitting an order of magnitude outside anything
/// this generates. The activation quantizer's range is set from the
/// calibration's own extremes (`npu_ort::observed_quant`, plus a 1.25
/// margin), so every one of those channels *saturates* on every token that
/// carries it.
///
/// What that costs is not a few percent. Measured on real input, against an
/// `f32` reference, `Q8_0`:
///
/// | block | calibrated on this | calibrated on the real thing |
/// |---|---|---|
/// | `blk.0` | 100.0% rms | 20.0% |
/// | `blk.5` | 100.0% rms | 22.8% |
/// | `blk.16` | 112.3% rms | 18.3% |
///
/// 100% rms is not a degraded answer, it is no answer: the error equals the
/// signal, and for two of those blocks the device returned an output with
/// no energy in it at all. The 3.6% this same block used to report was
/// measured on the calibration data it was built from — a loop that
/// confirms itself. See [`captured_calibration`] for the way out of it.
fn realistic_calibration(path: &Path, block: &BlockSpec, tokens: usize) -> Option<Vec<f32>> {
    let norm = read_dequantized(path, &format!("{}.ffn_norm.weight", block.prefix)).ok()?;
    if norm.len() != block.d_model {
        return None;
    }
    let d_model = block.d_model;
    let mut state = 0x51ed_1234u64;
    let mut unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as f32 / (1u64 << 31) as f32).clamp(f32::MIN_POSITIVE, 1.0)
    };
    let mut x = vec![0.0f32; tokens * d_model];
    for row in x.chunks_mut(d_model) {
        // Box-Muller, two at a time.
        for pair in row.chunks_mut(2) {
            let (u1, u2) = (unit(), unit());
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            pair[0] = r * theta.cos();
            if let Some(second) = pair.get_mut(1) {
                *second = r * theta.sin();
            }
        }
        // What RMSNorm guarantees, then what it applies.
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
        let scale = 1.0 / mean_sq.max(f32::MIN_POSITIVE).sqrt();
        for (v, w) in row.iter_mut().zip(&norm) {
            *v = *v * scale * w;
        }
    }
    Some(x)
}

/// The real thing if it was captured, [`realistic_calibration`] otherwise,
/// falling back to uniform noise when the block has no readable `ffn_norm`
/// — a projector tower, say, that normalizes somewhere this cannot see.
fn calibration_for(path: &Path, block: &BlockSpec, rows: usize) -> Vec<f32> {
    captured_calibration(block, rows)
        .or_else(|| realistic_calibration(path, block, rows))
        .unwrap_or_else(|| {
            synthetic_calibration(block.d_model, rows, block.range.unwrap_or((-1.0, 1.0)))
        })
}

/// How many rows of activations a *compile* is calibrated on, whatever
/// width the graph itself is.
///
/// **These are two different numbers and conflating them broke every
/// width-1 block.** A graph's width is how many tokens one inference takes;
/// a calibration is the sample every static range and every smoothing scale
/// is estimated from. Passing the width as the row count meant a decode
/// graph was compiled against exactly **one** row, and
/// `NpuOrt::smoothing_scales` then read that row's own values as its
/// per-channel maxima — a statistic with a sample size of one. Folded into
/// the weights, it is a per-channel distortion that cancels exactly on that
/// row and on nothing else.
///
/// It measured as a decode problem for a long time because decode is the
/// only path that asks for width 1. It is not: the same row, at the same
/// instant, through the same layer's width-16 graph, on llama 3.2 3B
/// `Q8_0`:
///
/// | layer | width 1 | width 16 |
/// |---|---|---|
/// | 2 | 71.1% rms, cos 0.711 | **16.6% rms, cos 0.989** |
/// | 5 | 71.7%, cos 0.697 | **11.6%, cos 0.994** |
/// | 8 | 69.3%, cos 0.732 | **13.3%, cos 0.992** |
///
/// Nothing about the input distinguishes those two columns. The width-1
/// graph was simply built from a one-row sample.
///
/// The floor is 32 because that is where the per-channel maxima stop moving
/// on this project's captures, and because the host reference the compile
/// runs over these rows is `O(rows)` — every row past the point the
/// statistic settles is compile time spent on nothing.
const CALIBRATION_ROWS: usize = 32;

/// Removes [`compile_scratch_dir`] on the way out, however that happens.
///
/// A guard rather than a call at the end of `compile`, because `compile`
/// has several early returns and each one would otherwise leave the
/// directory behind — which is the failure this whole arrangement exists to
/// stop.
struct ScratchDir(Option<std::path::PathBuf>);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Some(dir) = &self.0 {
            // Out of it before removing it: a process whose working
            // directory has been unlinked is a confusing thing to leave
            // behind, and `/` is somewhere that always exists.
            let _ = std::env::set_current_dir("/");
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Runs `npu-compile` as a child for one width.
///
/// A child process, because this one will go on to open the NPU runtime and
/// the two vendor stacks cannot share an address space. Shared by the
/// budgeted compile pass and by [`probe_prefill_width`], which needs the
/// same thing for one block at a time.
///
/// # It has to die with its parent
///
/// The parent here is a *server*, and servers get killed: by a benchmark
/// sweep moving to its next point, by a Ctrl-C, by an operator restarting
/// one. The parent blocks in `status()` while this runs, so a signal that
/// takes the parent out mid-compile leaves this child reparented and still
/// running — and it is not an idle orphan. It holds `/dev/aipu` and every
/// NPU core for the rest of its compile, it goes on writing into
/// `~/.orangu/npu`, and the scratch cleanup below never runs because the
/// parent that owned it is gone. Measured here before this was fixed: one
/// such orphan was still compiling eleven minutes after its server died,
/// and had to be killed by hand.
///
/// That orphan is invisible to everything that looks for a leftover server:
/// it has no port, it answers no `/health`, and it is not named
/// `orangu-server` in the way a process scan would expect. What it does is
/// make the *next* run share the NPU with a compiler, which shows up as a
/// slower number and no error at all — see `orangu-bench`'s sweep module,
/// which now refuses to start a point while an accelerator is still held by
/// something it did not start. The stale `/tmp/orangu-npu-compile-*`
/// directories this used to strew around were the visible half of it.
fn compile_child(
    executable: &Path,
    model: &Path,
    tokens: usize,
    limit: usize,
) -> std::io::Result<std::process::ExitStatus> {
    let mut command = std::process::Command::new(executable);
    command
        .arg("npu-compile")
        .arg(model)
        .arg("--tokens")
        .arg(tokens.to_string())
        .arg("--limit")
        .arg(limit.to_string())
        .stdout(std::process::Stdio::null())
        // **Captured, not inherited.** The vendor graph compiler writes to
        // stderr — a `Total errors: 0, warnings: 0` per block, among other
        // things — and this child runs *in the background of a serving
        // server*. Inherited, that lands in the middle of whatever the
        // server is printing, and what it lands on is the generation
        // progress line, which is redrawn in place with `\r` and so assumes
        // it owns the line. The result on a real console:
        //
        // ```text
        // [slot 0] prompt 16 tokens in 9.24s (1.73 tok/s), generated 207 tokens in 21.58s (9.59 tok/s)Total errors: 0,  warnings: 0
        // ```
        //
        // Two unrelated processes writing one line. Capturing it also turns
        // a *failed* compile from worse to better: the arm below could only
        // say "precompile exited with <status>", because the reason had
        // already scrolled past interleaved with something else. Now the
        // reason travels with the failure.
        //
        // Nothing is lost on success. An operator who wants the compiler's
        // full chatter runs the same thing in the foreground — `npu-compile`
        // is a subcommand of this binary, which is exactly what this spawns.
        .stderr(std::process::Stdio::piped());
    // `PR_SET_PDEATHSIG` is the only thing that closes the window described
    // above: it asks the kernel to signal this child when its parent dies,
    // which covers `SIGKILL` on the parent — the one case no amount of
    // parent-side cleanup can handle, because a `SIGKILL`ed parent runs no
    // code. `SIGKILL` rather than something gentler because a compile whose
    // requester is gone has nothing worth finishing. See
    // `orangu::child::die_with_parent` for the `exec` and reparenting
    // subtleties.
    #[cfg(target_os = "linux")]
    orangu::child::die_with_parent(&mut command, libc::SIGKILL);
    // **The calibration goes with it.** Compiling without this is what
    // built every block against a synthetic Gaussian whose tail is a ninth
    // of a real hidden state's — see `captured_calibration`.
    if let Some(dir) = calibration_dir() {
        command.env(CALIB_DIR_ENV, dir);
    }
    let scratch = compile_scratch_dir();
    if let Some(dir) = &scratch {
        command.current_dir(dir);
    }
    // Nothing to start if the server is already going away.
    if stopping() {
        return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
    }
    // Spawned rather than run in one call, so the pid can be recorded before
    // the wait: this child is the reason a shutdown would otherwise block
    // for minutes, and `stop_preparation` needs something to aim at.
    let child = command.spawn();
    // The scratch directory is cleaned on every path below, including this
    // one, so the early return does not leak it.
    let mut child = match child {
        Ok(child) => child,
        Err(e) => {
            if let Some(dir) = &scratch {
                let _ = std::fs::remove_dir_all(dir);
            }
            return Err(e);
        }
    };
    if let Ok(mut slot) = COMPILE_CHILD.lock() {
        *slot = Some(child.id());
    }
    // The flag may have been set between the check above and the pid being
    // recorded, in which case `stop_preparation` looked at an empty slot and
    // this child would outlive the shutdown. Re-check now that it is
    // findable, and end it here if so.
    if stopping() {
        let _ = child.kill();
    }
    // `wait_with_output()` rather than `wait()`: it drains the pipe while the
    // child runs. Waiting without draining would leave a compiler chatty
    // enough to fill the pipe buffer blocked forever writing into it — a
    // hang, in the middle of a compile that otherwise works.
    let finished = child.wait_with_output();
    if let Ok(mut slot) = COMPILE_CHILD.lock() {
        *slot = None;
    }
    // Whatever it left behind, with the directory it was left in. Best
    // effort: a compiler that wrote something unexpected is not a reason to
    // fail a compile that otherwise worked.
    if let Some(dir) = &scratch {
        let _ = std::fs::remove_dir_all(dir);
    }
    let finished = finished?;
    if !finished.status.success() {
        report_compiler_stderr(&finished.stderr);
    }
    Ok(finished.status)
}

/// Prints what the graph compiler said, for a compile that failed.
///
/// The **tail**, and a bounded amount of it. The vendor compiler narrates its
/// way through every block, so the beginning of that stream is the part least
/// likely to say why the end went wrong, and reprinting all of it would push
/// the server's own explanation off the screen — the thing the reader
/// actually needs. Blank lines go too: this output is mostly whitespace.
fn report_compiler_stderr(stderr: &[u8]) {
    for line in compiler_report_lines(stderr) {
        eprintln!("orangu-server: [npu] compiler: {line}");
    }
}

/// Enough for a stack of vendor diagnostics, few enough to read.
const COMPILER_REPORT_LINES: usize = 20;

/// The lines [`report_compiler_stderr`] would print, without printing them.
fn compiler_report_lines(stderr: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let skipped = lines.len().saturating_sub(COMPILER_REPORT_LINES);
    let mut out = Vec::new();
    if skipped > 0 {
        out.push(format!("... {skipped} earlier line(s) omitted"));
    }
    out.extend(lines[skipped..].iter().map(|l| (*l).to_string()));
    out
}

/// A temporary directory for the graph compiler to litter in.
///
/// **In `/tmp`, not the block cache.** `~/.orangu/npu` holds the compiled
/// blocks — the things worth keeping, that a later run loads — and the
/// vendor compiler's assembler output is neither. It is scratch, so it goes
/// where scratch goes, next to the context files `npu_ort::ContextFile`
/// already writes there for the same reason.
///
/// Named by process id and a counter so two compiles cannot collide, and
/// removed by the caller when the child exits. `None` when it cannot be
/// created, in which case the child keeps the parent's working directory
/// and behaves exactly as it did before this existed.
fn compile_scratch_dir() -> Option<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "orangu-npu-compile-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    // Sweep up what earlier runs could not. The cleanup in `compile_child`
    // belongs to the parent, so a parent that was killed mid-compile leaves
    // its child's directory behind for good; `PR_SET_PDEATHSIG` stops the
    // orphaned *compiler*, but nothing is left alive to remove the
    // directory. They are small and they accumulate for as long as the
    // machine stays up, so each new compile clears the dead ones.
    reap_stale_scratch_dirs();
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Removes `orangu-npu-compile-*` directories whose owning process is gone.
///
/// Keyed on the pid in the name, and only ever deleting when that pid is
/// **not** running: a recycled pid that happens to be alive keeps its
/// directory, which errs towards leaving litter rather than towards deleting
/// a compile in progress. Best effort throughout — this is tidying, and a
/// failure to tidy is not a reason to fail a compile.
fn reap_stale_scratch_dirs() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix("orangu-npu-compile-") else {
            continue;
        };
        // `<pid>-<counter>`; anything else is not ours to remove.
        let Some((pid, _)) = rest.split_once('-') else {
            continue;
        };
        let Ok(pid) = pid.parse::<u32>() else {
            continue;
        };
        if !pid_is_running(pid) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Whether `pid` names a live process.
///
/// `/proc` rather than `kill(pid, 0)` so that a process belonging to another
/// user still reads as alive — `kill` would report `EPERM` there, which is
/// indistinguishable from success only if you squint, and reading it as
/// "dead" would delete a running compile's scratch directory out from under
/// it.
fn pid_is_running(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    } else {
        // Without `/proc` there is no cheap way to ask, and guessing wrong
        // in the deleting direction is the expensive mistake. Assume alive.
        true
    }
}

/// This model's capture directory, beside the compiled blocks it feeds.
///
/// Keyed by the same fingerprint the block cache uses, so two models never
/// read each other's activations — which would be worse than synthesizing
/// them, since the numbers would look plausible and be from another network.
fn capture_dir(model: &Path) -> Option<std::path::PathBuf> {
    let gguf = open_model(model)?;
    Some(
        NpuCache::default_dir()
            .join("calibration")
            .join(fingerprint(&gguf)),
    )
}

/// Whether `dir` holds a capture for every layer of this model.
///
/// By count rather than by content: a partial capture is one a run was
/// interrupted in the middle of, and compiling half a model against real
/// activations and half against nothing is not a state worth having.
fn capture_complete(dir: &Path, n_layer: usize) -> bool {
    captured_layers(dir, "blk.") >= n_layer
}

/// How many files in `dir` a capture has written under `prefix`.
///
/// **Counted by prefix, not by directory entry.** The directory holds one
/// file per layer per capture, and there is more than one capture: the
/// feed-forward's input under `blk.` and the attention projections' under
/// `attn.`. Counting entries meant that adding the second one would have
/// declared the first complete at half the layers, and compiled every block
/// against a capture that was still being written.
fn captured_layers(dir: &Path, prefix: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
        .count()
}

/// Waits for the serving path to capture every layer of a **whole prompt**,
/// or gives up.
///
/// Two conditions, and the second one is the one that was learned the hard
/// way. Every layer having a file means the first chunk has been through —
/// 16 tokens of a 720-token prompt — and calibrating on that measured 38.1%
/// against 22.3% for the whole prompt, because the range a static quantizer
/// needs is the range over everything it will see, not over its first
/// sixteen tokens. So this then waits for the capture to stop *growing*: a
/// prefill appends a chunk at a time, and when the bytes hold still the
/// prompt is done.
///
/// The wait is unbounded in prompts and bounded in time: a server nobody
/// sends anything to has nothing to calibrate on, and holding a thread
/// forever for that is worse than saying so and leaving the device alone.
/// Ten minutes is long enough for a person to type one message.
fn await_capture(dir: &Path, n_layer: usize) -> Capture {
    const BEAT: std::time::Duration = std::time::Duration::from_millis(250);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    while std::time::Instant::now() < deadline && !capture_complete(dir, n_layer) {
        std::thread::sleep(BEAT);
        // **A model that has run and captured nothing never will.** Only
        // `LlamaModel::run_layers` and gemma's prefill offer their
        // feed-forward input to `engine::dump_ffn_input`; every other
        // architecture forms it somewhere this cannot see. Waiting the full
        // ten minutes for one of those and then blaming a prompt that did
        // arrive is what this used to do — Ling 3.0 (`bailingmoe3`) served a
        // 21-token prompt in 2.28 s and was told no prompt arrived.
        if crate::engine::forward_passes() > 0 && capture_bytes(dir) == 0 {
            return Capture::Unsupported;
        }
    }
    if !capture_complete(dir, n_layer) {
        return Capture::Incomplete;
    }
    // Settled rather than merely present. Four quiet beats, because a chunk
    // of a large model takes longer than one — and the cost of stopping too
    // early is a calibration that misses the widest values in the prompt,
    // which is exactly the failure this whole path exists to avoid.
    let mut quiet = 0;
    let mut last = capture_bytes(dir);
    while std::time::Instant::now() < deadline && quiet < 4 {
        std::thread::sleep(BEAT);
        let now = capture_bytes(dir);
        quiet = if now == last { quiet + 1 } else { 0 };
        last = now;
    }
    Capture::Complete
}

/// What [`await_capture`] found.
enum Capture {
    /// Every layer's activations are on disk and have stopped growing.
    Complete,
    /// The model ran and offered nothing, so this architecture has no hook
    /// into the capture and never will.
    Unsupported,
    /// Some layers arrived and the rest did not inside the deadline —
    /// an idle server, or a prompt that stopped early.
    Incomplete,
}

/// How much a capture holds, for telling "still being written" from "done".
fn capture_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// The capture file for one layer: this width if it was captured, else any
/// other width of the same layer.
///
/// **A capture is rows of activations, not a width.** The file name records
/// the width the forward pass happened to be running at, but what it holds
/// is `d_model`-wide rows, and a graph compiled for any width wants some
/// number of those. A decode block needs one row and the serving path only
/// ever captures at the prefill width, so keying strictly on the name left
/// every width-1 block calibrated on synthetic noise — the failure this
/// whole path exists to end.
///
/// The exact width still wins when it is there, so nothing changes for the
/// prefill blocks that have their own file.
fn captured_file(dir: &Path, layer: &str, tokens: usize) -> Option<std::path::PathBuf> {
    let exact = dir.join(format!("blk.{layer}.{tokens}.f32"));
    if exact.is_file() {
        return Some(exact);
    }
    let prefix = format!("blk.{layer}.");
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".f32"))
        })
}

/// A digest of the capture the blocks are compiled against, or `0` when
/// there is none.
///
/// From file names and sizes rather than contents: the capture is hundreds
/// of megabytes and both processes have to agree on this cheaply, and what
/// it needs to detect is "a different capture", which a re-run of a
/// different prompt changes in size on essentially every layer. It is a
/// cache key, not a checksum — being wrong costs a recompile, not a wrong
/// answer, because the compile reads the calibration itself.
fn calibration_digest() -> u64 {
    let Some(dir) = calibration_dir() else {
        return 0;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return 0;
    };
    // FNV-1a over each entry's name and length, order-independent so two
    // readings of one directory agree whatever order the filesystem hands
    // them back in.
    let mut digest = 0u64;
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let mut one = 0xcbf2_9ce4_8422_2325u64;
        for byte in entry.file_name().to_string_lossy().bytes() {
            one = (one ^ byte as u64).wrapping_mul(0x1000_0000_01b3);
        }
        for byte in meta.len().to_le_bytes() {
            one = (one ^ byte as u64).wrapping_mul(0x1000_0000_01b3);
        }
        digest ^= one;
    }
    digest
}

/// Where the captured activations for the model being served live.
///
/// Set once by [`set_calibration_dir`] in the process that serves, and
/// passed to the process that compiles through `ORANGU_NPU_CALIB_DIR` —
/// they are different processes on purpose (the vendor runtime and the
/// serving runtime cannot share one), so the answer has to survive a
/// `Command`. The environment also lets a measurement point either process
/// at a capture by hand, which is how this path was developed.
fn calibration_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = CALIBRATION_DIR.get().cloned().flatten() {
        return Some(dir);
    }
    std::env::var_os(CALIB_DIR_ENV).map(std::path::PathBuf::from)
}

/// Records where this process's captured activations are. Later calls are
/// ignored, matching `npu_ffn::install`: one model per process.
pub fn set_calibration_dir(dir: Option<std::path::PathBuf>) {
    let _ = CALIBRATION_DIR.set(dir);
}

static CALIBRATION_DIR: std::sync::OnceLock<Option<std::path::PathBuf>> =
    std::sync::OnceLock::new();

/// The name the serving process hands the compiling one.
const CALIB_DIR_ENV: &str = "ORANGU_NPU_CALIB_DIR";

/// A layer's real feed-forward input, captured from a forward pass by
/// `engine::dump_ffn_input` and named by `ORANGU_NPU_CALIB_DIR`.
///
/// **The point of the whole detour.** Static quantization fixes the
/// activation range when the block is compiled, from whatever calibration
/// it was given, and a value outside that range saturates rather than
/// rounds — `npu_ort::observed_quant` measures saturation as the dominant
/// cost by a wide margin. [`realistic_calibration`] gets the *shape* of a
/// normalized hidden state right and its tail wrong, because it draws every
/// channel from the same Gaussian, and a transformer's hidden state does
/// not: a handful of channels sit persistently far outside the bulk. This
/// reads the actual values instead, so the range is the range.
///
/// **What it is worth**, on gemma 4 E2B at `Q8_0`, a 720-token prompt, all
/// 35 blocks on the device, against 4.56 tok/s of prefill with none:
///
/// | calibration | worst block | prefill | answer |
/// |---|---|---|---|
/// | synthetic Gaussian | 17.8% rms* | 13.59 tok/s | `**\n    \n    \n ...` |
/// | first chunk of the prompt | 30.5% | 13.58 | coherent, on topic |
/// | widest rows of the whole prompt | 22.3% | 13.41 | coherent, on topic |
///
/// \* measured against the same synthetic input it was built from, which is
/// the loop this replaced — on real input those blocks measure 100% rms and
/// two of the three sampled returned an output with no energy in it at all.
///
/// **Measured properly** — device off and device on, same model, same two
/// prompts, the second one the comparison so both runs reach it with the
/// same history — gemma 4 E2B at `Q8_0` asked to explain a B-tree:
///
/// | | prefill | answer |
/// |---|---|---|
/// | off | 4.44 tok/s | `A B-tree is a self-balancing tree data structure that maintains its keys in sorted order.` |
/// | on, calibrated | 12.48 tok/s | `The provided text is a highly fragmented and nonsensical collection of phrases, likely res` |
/// | on, calibrated and smoothed | **12.30 tok/s** | `A B-tree is a self-balancing tree data structure designed to efficiently store and retriev` |
///
/// Three states of the same model on the same prompt: garbage before the
/// calibration was real, a coherent answer to the wrong question once it
/// was, and a correct answer once `npu_ort::smoothing_scales` moved the
/// activation outliers into the weights. 2.8x prefill throughout — the
/// accuracy work costs nothing in speed.
///
/// The output still diverges token for token from the reference, as any
/// requantization does. What changed is that it is now an equally good
/// answer rather than a worse one.
///
/// Silent when the directory is unset, missing the file, or holding one of
/// the wrong length — a capture from a different model or width is not
/// calibration for this one, and using it would be worse than the synthetic
/// input it replaced.
fn captured_calibration(block: &BlockSpec, rows_wanted: usize) -> Option<Vec<f32>> {
    let dir = calibration_dir()?;
    let layer = block.prefix.strip_prefix("blk.")?;
    let path = captured_file(&dir, layer, rows_wanted)?;
    let bytes = std::fs::read(&path).ok()?;
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let d_model = block.d_model;
    if values.is_empty() || !values.len().is_multiple_of(d_model) {
        eprintln!(
            "orangu-server: [npu] {} holds {} floats, not a whole number of {d_model}-wide rows \
             — ignored",
            path.display(),
            values.len()
        );
        return None;
    }

    // **The widest rows, not the first ones.** The capture holds the whole
    // prompt and only `rows_wanted` of it is kept, so which rows are kept
    // decides the range the activation quantizer is built around. Keeping
    // the widest means a value seen anywhere in the prompt is inside the
    // calibrated range rather than saturating against it — and saturation is
    // the failure this whole path exists to avoid.
    let mut rows: Vec<&[f32]> = values.chunks_exact(d_model).collect();
    rows.sort_unstable_by(|a, b| {
        let width = |r: &&[f32]| r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        width(b).total_cmp(&width(a))
    });
    // **Widest for the range, then a spread for everything else.** Keeping
    // only the widest rows covers the range, and that is all it covers: the
    // quantization step, the smoothing scales and — since
    // `NpuOrt::choose_smoothing_alpha` — the migration strength are all
    // estimated from whatever is kept, and estimating them from the tail of
    // the distribution describes a distribution the model does not have.
    //
    // It shows up as a block that is fine on the rows it was built from and
    // ruined on the rows it serves. Llama 3.2 3B `Q8_0` layer 1 measured
    // 3.6% `rms` on its own calibration row while returning *exactly zero*
    // for a real decode row, with its `down` input using 9 of 256 levels.
    // A quarter widest and the rest spread evenly keeps the range and
    // describes the bulk.
    let widest = (rows_wanted / 4).max(1).min(rows.len());
    let mut kept: Vec<&[f32]> = rows[..widest].to_vec();
    let rest = &rows[widest..];
    let wanted = rows_wanted - widest;
    for i in 0..wanted {
        // Cycled when the capture is shorter than the caller asked for, so
        // the result is always exactly `rows_wanted` rows: a compile wants
        // statistics and a measurement wants a tensor of a fixed shape, and
        // neither should have to ask how long the prompt happened to be.
        let pool = if rest.is_empty() { &rows[..] } else { rest };
        kept.push(pool[i * pool.len() / wanted.max(1) % pool.len()]);
    }
    let mut out = Vec::with_capacity(rows_wanted * d_model);
    for row in kept {
        out.extend_from_slice(row);
    }
    Some(out)
}

/// Compiles one layer's `Q`/`K`/`V` into the cache, if its input was
/// captured.
///
/// `Ok(None)` when the artifact is already there or this model has no
/// capture of `attn_norm(x)` to calibrate against — neither is a failure,
/// and a model whose architecture never offers that input simply keeps its
/// attention on the GPU.
///
/// **Measured before it was built.** `npu-attn` put these projections at
/// 1.9-4.2% `rms` against `f32` across layers 0, 1, 8 and 15 of llama 3.2
/// 1B `Q8_0` — three to ten times better than the feed-forward blocks the
/// device already runs, and notably fine at layer 1, whose feed-forward is
/// the one this family cannot represent at all.
fn compile_attention(
    compiler: &orangu::npu_ort::NpuOrt,
    cache: &NpuCache,
    path: &Path,
    model: &str,
    layer: usize,
    tokens: usize,
    force: bool,
) -> Result<Option<u64>> {
    let block = attention_block_name(layer);
    let key = BlockKey {
        model,
        block: &block,
        tokens,
        calibration: calibration_digest(),
    };
    if !force && cache.contains_projections(key) {
        return Ok(None);
    }
    let name = |part: &str| format!("blk.{layer}.attn_{part}.weight");
    let parts = ["q", "k", "v"];
    let mut weights = Vec::with_capacity(parts.len());
    for part in parts {
        match read_dequantized(path, &name(part)) {
            Ok(w) => weights.push(w),
            // A family that fuses or omits one of the three is not one this
            // compiles; the feed-forward blocks are unaffected.
            Err(_) => return Ok(None),
        }
    }
    let Some(k) = attention_input_width(path, layer) else {
        return Ok(None);
    };
    let Some(calibration) = attn_calibration(layer, tokens.max(CALIBRATION_ROWS), k) else {
        return Ok(None);
    };
    let shapes: Vec<usize> = weights.iter().map(|w| w.len() / k).collect();
    let refs: Vec<&[f32]> = weights.iter().map(Vec::as_slice).collect();
    let compiled = orangu::npu_ort::CompiledProjections {
        parts: compiler
            .compile_shared_input(&refs, &shapes, k, tokens, &calibration)
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    };
    let size = compiled.binary_len() as u64;
    cache.store_projections(key, &compiled)?;
    Ok(Some(size))
}

/// How this layer's attention projections are named in the cache.
fn attention_block_name(layer: usize) -> String {
    format!("attn.blk.{layer}")
}

/// The input width of a layer's attention projections.
fn attention_input_width(path: &Path, layer: usize) -> Option<usize> {
    let gguf = open_model(path)?;
    gguf.tensors
        .iter()
        .find(|t| t.name == format!("blk.{layer}.attn_q.weight"))
        .and_then(|t| t.dims.first().copied())
        .map(|d| d as usize)
}

/// Whether the attention projections may run on the device, on unless
/// `ORANGU_NPU_ATTN=0`.
///
/// A serving-time switch rather than a compile-time one, so one cache
/// serves both arms of a measurement and the two differ in nothing else.
pub fn attention_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_NPU_ATTN"))
}

/// Measures this model's attention projections on the device.
///
/// Two invocations, because compiling needs ONNX Runtime and executing
/// needs NOE and the two cannot share a process. The first compiles
/// `Q`/`K`/`V` for one layer against the captured `attn_norm(x)` and writes
/// the artifact; the second binds it and reports how far each projection
/// lands from the `f32` answer on the same rows.
///
/// A measurement, not a feature: it exists to decide whether moving the
/// attention projections to the device is worth building at all, before the
/// cache format, the accuracy gate and the forward hook are written.
pub fn attn_probe(
    path: &Path,
    tokens: usize,
    layer: usize,
    run: bool,
    artifact: &Path,
) -> Result<()> {
    let gguf = open_model(path).with_context(|| format!("reading {}", path.display()))?;
    let name = |part: &str| format!("blk.{layer}.attn_{part}.weight");
    let k = gguf
        .tensors
        .iter()
        .find(|t| t.name == name("q"))
        .and_then(|t| t.dims.first().copied())
        .ok_or_else(|| anyhow::anyhow!("{} has no {}", path.display(), name("q")))?
        as usize;

    let calibration = attn_calibration(layer, tokens, k).ok_or_else(|| {
        anyhow::anyhow!(
            "no captured attn.{layer}.* to calibrate on — run a prompt with ORANGU_NPU_DUMP_ACT set"
        )
    })?;
    let rows = calibration.len() / k;

    let parts = ["q", "k", "v"];
    let weights: Vec<Vec<f32>> = parts
        .iter()
        .map(|part| read_dequantized(path, &name(part)))
        .collect::<Result<_>>()?;
    let shapes: Vec<usize> = weights.iter().map(|w| w.len() / k).collect();

    if !run {
        let compiler = orangu::npu_ort::NpuOrt::open()
            .ok_or_else(|| anyhow::anyhow!("the NPU compiler is not available here"))?;
        let refs: Vec<&[f32]> = weights.iter().map(Vec::as_slice).collect();
        let parts = compiler
            .compile_shared_input(&refs, &shapes, k, tokens, &calibration)
            .map_err(|e| anyhow::anyhow!("compiling: {e}"))?;
        let compiled = orangu::npu_ort::CompiledProjections { parts };
        std::fs::write(artifact, compiled.to_bytes())?;
        println!(
            "compiled layer {layer} q/k/v at {tokens} tokens ({k} -> {shapes:?}) into {}",
            artifact.display()
        );
        return Ok(());
    }

    let bytes = std::fs::read(artifact)
        .with_context(|| format!("reading {} — compile it first", artifact.display()))?;
    let compiled = orangu::npu_ort::CompiledProjections::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("reading the artifact: {e}"))?;
    let runtime = orangu::npu::NpuRuntime::open().ok_or_else(|| anyhow::anyhow!("no NPU here"))?;
    let bound = compiled
        .bind(&runtime)
        .map_err(|e| anyhow::anyhow!("binding: {e}"))?;

    println!("  {:<6} {:>10} {:>10}", "part", "rms", "peak");
    for ((part, w), graph) in parts.iter().zip(&weights).zip(&bound) {
        let n = w.len() / k;
        let mut got = Vec::new();
        graph
            .forward_into(&calibration, tokens, &mut got)
            .map_err(|e| anyhow::anyhow!("running {part}: {e}"))?;
        // The same rows through the same matrix in `f32`.
        let mut want = vec![0.0f32; rows * n];
        for r in 0..rows {
            for o in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += calibration[r * k + i] * w[o * k + i];
                }
                want[r * n + o] = acc;
            }
        }
        let (mut num, mut den, mut worst, mut peak) = (0.0f64, 0.0f64, 0.0f32, 0.0f32);
        for (g, t) in got.iter().zip(&want) {
            num += f64::from(g - t).powi(2);
            den += f64::from(*t).powi(2);
            worst = worst.max((g - t).abs());
            peak = peak.max(t.abs());
        }
        let rms = if den > 0.0 {
            (num / den).sqrt() * 100.0
        } else {
            0.0
        };
        println!(
            "  {part:<6} {:>9.1}% {:>9.1}%",
            rms,
            if peak > 0.0 {
                100.0 * worst / peak
            } else {
                0.0
            }
        );
    }
    Ok(())
}

/// The captured `attn_norm(x)` for one layer, widest rows first.
fn attn_calibration(layer: usize, tokens: usize, k: usize) -> Option<Vec<f32>> {
    attn_calibration_in(&calibration_dir()?, layer, tokens, k)
}

/// [`attn_calibration`] against a named directory, for a probe that knows
/// which model it is measuring and so need not be told where its capture is.
fn attn_calibration_in(
    dir: &std::path::Path,
    layer: usize,
    tokens: usize,
    k: usize,
) -> Option<Vec<f32>> {
    let dir = dir.to_path_buf();
    let mut best: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(&dir).ok()?.filter_map(|e| e.ok()) {
        let file = entry.file_name();
        let file = file.to_string_lossy();
        let Some(rest) = file.strip_prefix(&format!("attn.{layer}.")) else {
            continue;
        };
        // Prefer the widest capture: more rows is a better sample, and the
        // reader keeps the widest of them anyway.
        if rest.ends_with(".f32") {
            let wider = best.as_ref().is_none_or(|b| {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
                    > std::fs::metadata(b).map(|m| m.len()).unwrap_or(0)
            });
            if wider {
                best = Some(entry.path());
            }
        }
    }
    let values: Vec<f32> = std::fs::read(best?)
        .ok()?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if values.is_empty() || !values.len().is_multiple_of(k) {
        return None;
    }
    let mut rows: Vec<&[f32]> = values.chunks_exact(k).collect();
    rows.sort_unstable_by(|a, b| {
        let width = |r: &&[f32]| r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        width(b).total_cmp(&width(a))
    });
    // Cycled when the capture is shorter than asked for, exactly as
    // `captured_calibration` does. The prompt that triggers a capture is
    // often a few tokens long — the server sends one to wake the model —
    // while a graph is compiled for 128, and refusing on that basis meant
    // the attention projections were silently never compiled at all.
    Some(
        rows.iter()
            .cycle()
            .take(tokens)
            .flat_map(|r| r.iter().copied())
            .collect(),
    )
}

/// **Is the vocabulary projection worth moving to the device?** Compiles
/// `output.weight` alone and reports what it costs to run and how far it
/// lands from `f32`.
///
/// Hidden, and a measurement rather than a feature, for the same reason
/// [`attn_probe`] is: compiling and executing cannot share a process, so it
/// runs twice — once to compile, once with `--run`.
///
/// The projection is the largest single matrix in the file (`n_vocab x
/// n_embd`, five times an entire feed-forward block on a small model) and
/// unlike a block it is bandwidth and nothing else: one row of activations
/// against every weight, once per token. So the question the timing answers
/// is narrow — does this device read those bytes faster than the GPU does —
/// and the accuracy column answers whether logits survive `uint8` at all,
/// which matters more here than anywhere else in the model because logits
/// are compared against each other rather than summed into a residual.
pub fn head_probe(path: &Path, tokens: usize, run: bool, artifact: &Path) -> Result<()> {
    let gguf = open_model(path).with_context(|| format!("reading {}", path.display()))?;
    let name = head_tensor_name(&gguf)
        .ok_or_else(|| anyhow::anyhow!("{} has no vocabulary projection", path.display()))?;
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow::anyhow!("{} has no {name}", path.display()))?;
    let k = *tensor
        .dims
        .first()
        .ok_or_else(|| anyhow::anyhow!("{name} has no shape"))? as usize;

    let dir = capture_dir(path)
        .or_else(calibration_dir)
        .ok_or_else(|| anyhow::anyhow!("no capture directory for {}", path.display()))?;
    let calibration = head_calibration(&dir, tokens, k).ok_or_else(|| {
        anyhow::anyhow!(
            "no captured activations to calibrate on — run a prompt with ORANGU_NPU_DUMP_ACT set"
        )
    })?;
    let rows = calibration.len() / k;

    if !run {
        let started = std::time::Instant::now();
        let weight = read_dequantized(path, &name)?;
        let n = weight.len() / k;
        let compiler = orangu::npu_ort::NpuOrt::open()
            .ok_or_else(|| anyhow::anyhow!("the NPU compiler is not available here"))?;
        let parts = compiler
            .compile_shared_input(&[weight.as_slice()], &[n], k, tokens, &calibration)
            .map_err(|e| anyhow::anyhow!("compiling: {e}"))?;
        let compiled = orangu::npu_ort::CompiledProjections { parts };
        let bytes = compiled.to_bytes();
        let len = bytes.len();
        std::fs::write(artifact, bytes)?;
        println!(
            "compiled {name} at {tokens} tokens ({k} -> {n}) in {:.1}s into {} ({:.0} MiB)",
            started.elapsed().as_secs_f64(),
            artifact.display(),
            len as f64 / (1024.0 * 1024.0)
        );
        return Ok(());
    }

    let bytes = std::fs::read(artifact)
        .with_context(|| format!("reading {} — compile it first", artifact.display()))?;
    let compiled = orangu::npu_ort::CompiledProjections::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("reading the artifact: {e}"))?;
    let runtime = orangu::npu::NpuRuntime::open().ok_or_else(|| anyhow::anyhow!("no NPU here"))?;
    let bound = compiled
        .bind(&runtime)
        .map_err(|e| anyhow::anyhow!("binding: {e}"))?;
    let graph = bound
        .first()
        .ok_or_else(|| anyhow::anyhow!("the artifact holds no graph"))?;

    let mut got = Vec::new();
    graph
        .forward_into(&calibration, tokens, &mut got)
        .map_err(|e| anyhow::anyhow!("running: {e}"))?;
    let n = got.len() / rows;

    // Median of a handful, for the same reason `decode_matvec_probe` takes
    // one: a scheduler hiccup moves a mean by more than the difference this
    // exists to see.
    let mut samples: Vec<f64> = (0..HEAD_PROBE_REPS)
        .map(|_| {
            let at = std::time::Instant::now();
            let _ = graph.forward_into(&calibration, tokens, &mut got);
            at.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let us = samples[samples.len() / 2];
    println!(
        "  {name}  {k} -> {n} at {tokens} tokens: {us:.0} us  ({:.1} GB/s of weights)",
        (k * n) as f64 / (us * 1e3)
    );

    // `f32` through the same matrix, for as many rows as are affordable:
    // this is `rows * n * k` multiply-accumulates in scalar host code, and
    // at a 128k vocabulary even one row is a quarter of a billion of them.
    let weight = read_dequantized(path, &name)?;
    let checked = rows.min(HEAD_PROBE_ROWS);
    let (mut num, mut den, mut agree) = (0.0f64, 0.0f64, 0usize);
    for r in 0..checked {
        let mut want = vec![0.0f32; n];
        for (o, want) in want.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += calibration[r * k + i] * weight[o * k + i];
            }
            *want = acc;
        }
        for (g, t) in got[r * n..(r + 1) * n].iter().zip(&want) {
            num += f64::from(g - t).powi(2);
            den += f64::from(*t).powi(2);
        }
        // The only thing a logit vector is ever asked: which token wins.
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 { (i, x) } else { best }
                })
                .0
        };
        if top(&got[r * n..(r + 1) * n]) == top(&want) {
            agree += 1;
        }
    }
    println!(
        "  rms {:.1}%   top-1 agrees on {agree}/{checked} rows",
        if den > 0.0 {
            (num / den).sqrt() * 100.0
        } else {
            0.0
        }
    );

    // The same matrix where it runs today. Without this the device number
    // means nothing: this projection is bandwidth and nothing else, both
    // engines read the same DRAM, and the only question is which reads it
    // faster — in the file's own quantization, which is part of the answer.
    let loaded = crate::engine::loader::LoadedModel::open(path)?;
    let matrix = loaded.matrix(&name)?;
    let row = &calibration[..k];
    let gpu =
        crate::engine::backend::VulkanBackend::try_init_selected(wgpu::Backends::VULKAN, &[0])
            .map(|b| std::sync::Arc::new(b) as std::sync::Arc<dyn crate::engine::backend::Backend>);
    let cpu = std::sync::Arc::new(crate::engine::backend::CpuBackend)
        as std::sync::Arc<dyn crate::engine::backend::Backend>;
    for (label, backend) in [("gpu", gpu), ("cpu", Some(cpu))] {
        let Some(backend) = backend else {
            continue;
        };
        let _ = backend.matmul(row, 1, &matrix);
        let mut samples: Vec<f64> = (0..HEAD_PROBE_REPS)
            .map(|_| {
                let at = std::time::Instant::now();
                let _ = backend.matmul(row, 1, &matrix);
                at.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        samples.sort_by(f64::total_cmp);
        let us = samples[samples.len() / 2];
        println!(
            "  {label:<3} {us:>8.0} us  ({:.1} GB/s of weights)",
            (k * n) as f64 / (us * 1e3)
        );
    }
    Ok(())
}

/// How many timed runs [`head_probe`] takes the median of.
const HEAD_PROBE_REPS: usize = 9;

/// How many rows [`head_probe`] checks against `f32`. Each one is `n_vocab
/// x n_embd` multiply-accumulates in scalar host code — a quarter of a
/// billion on a small model — so this is the number of seconds it is worth
/// spending, not a sample size chosen for its own sake.
const HEAD_PROBE_ROWS: usize = 4;

/// The vocabulary projection's tensor. Models that tie the projection to the
/// embedding table — most small ones — omit `output.weight` and read
/// `token_embd.weight` back, exactly as `LlamaModel::load` does.
fn head_tensor_name(gguf: &GgufFile) -> Option<String> {
    for name in ["output.weight", "token_embd.weight"] {
        if gguf.tensors.iter().any(|t| t.name == name) {
            return Some(name.to_string());
        }
    }
    None
}

/// Captured activations to calibrate the projection on, from the deepest
/// layer captured.
///
/// What the projection actually sees is `output_norm(x)` after the last
/// layer, which nothing captures yet. The deepest layer's `attn_norm(x)` is
/// the closest thing on hand and the same width; it is a stand-in for a
/// speed measurement, where the calibration only has to be the right shape,
/// and it is why the accuracy column here is a floor rather than a verdict.
fn head_calibration(dir: &std::path::Path, tokens: usize, k: usize) -> Option<Vec<f32>> {
    let mut deepest = None;
    for entry in std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()) {
        let file = entry.file_name();
        let file = file.to_string_lossy();
        let Some(rest) = file.strip_prefix("attn.") else {
            continue;
        };
        let Some(layer) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        deepest = Some(deepest.map_or(layer, |d: usize| d.max(layer)));
    }
    attn_calibration_in(dir, deepest?, tokens, k)
}

/// Compiles a model's feed-forward blocks into the cache.
pub fn compile(
    path: &Path,
    tokens: usize,
    filter: Option<&str>,
    force: bool,
    limit: Option<usize>,
) -> Result<()> {
    if tokens == 0 {
        bail!("--tokens must be at least 1");
    }
    // **Work somewhere disposable.** The vendor graph compiler assembles
    // each block through files in a hash-named directory beside the
    // process's working directory, and removes it only on a clean exit — so
    // an interrupted compile leaves something like
    // `1c864f03e6bb4681999d273682a22cc5727c3d507b21549165a9b0bfdbb5e/ld/main.o`
    // wherever the process was started, which for this project's own
    // developers is the middle of a checkout. `precompile` already gives
    // the child it spawns a directory of its own; this covers the same
    // process run by hand, so the behaviour does not depend on who started
    // it.
    //
    // Every path this then uses is made absolute first: the block cache and
    // the model are, after the calls below, and a relative
    // `ORANGU_NPU_CALIB_DIR` would otherwise silently stop resolving.
    let path = &std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Some(dir) = calibration_dir()
        && let Ok(absolute) = std::fs::canonicalize(&dir)
    {
        set_calibration_dir(Some(absolute));
    }
    let scratch = compile_scratch_dir();
    if let Some(dir) = &scratch {
        // Failing to move is not a reason to refuse to compile; it only
        // means the litter lands where it used to.
        let _ = std::env::set_current_dir(dir);
    }
    let _scratch = ScratchDir(scratch);
    let gguf = open_model(path).with_context(|| format!("reading {}", path.display()))?;
    let model = fingerprint(&gguf);
    let blocks = discover_blocks(&gguf, path, filter);
    if blocks.is_empty() {
        bail!(
            "no gated feed-forward blocks in {} — expected `<block>.ffn_gate/ffn_up/ffn_down.weight` triples",
            path.display()
        );
    }

    // `--limit` takes the first `n` blocks in file order. The caller that
    // uses it is `precompile_at_startup`, which has a cache budget to keep
    // and wants "the first n that fit" rather than a name filter — `blk.1`
    // as a substring also selects `blk.10` through `blk.19`, which is not a
    // thing anyone means.
    // Whether the caller's budget covered every block. When it did not,
    // the attention projections are left off entirely — see the note where
    // they are compiled.
    let all_blocks_fit = limit.is_none_or(|n| n >= blocks.len());
    let blocks = match limit {
        Some(n) => blocks.into_iter().take(n).collect::<Vec<_>>(),
        None => blocks,
    };
    if blocks.is_empty() {
        return Ok(());
    }

    let cache = NpuCache::open(NpuCache::default_dir())?;
    println!(
        "Compiling {} block(s) of {model} at {tokens} tokens into {}",
        blocks.len(),
        cache.dir().display()
    );

    let Some(compiler) = orangu::npu_ort::NpuOrt::open() else {
        bail!("no NPU compiler on this machine (the vendor ONNX Runtime is not installed)");
    };

    // Which activation sits between the projections, from the file itself.
    // The device never computes it — the host does — so this only has to
    // name the right function, not find ops the provider gets right.
    let activation = host_activation(&gguf);

    let (mut compiled, mut skipped, mut failed, mut bytes) = (0usize, 0usize, 0usize, 0u64);
    let started = std::time::Instant::now();
    for block in &blocks {
        // The attention projections of the same layer, when this model
        // captured their input too. Compiled beside the feed-forward rather
        // than in a pass of their own: they share the layer, the width and
        // the calibration capture, and a second pass would re-read the same
        // weights from the same file to learn the same things.
        // **Only when every feed-forward block already fits.** These are
        // worth about 2% of prefill; a feed-forward block is worth two to
        // six times the whole run. Letting the attention projections take
        // budget a block would have used trades the large win for the small
        // one, and the sweep showed the 7-8B models spending their entire
        // budget on blocks alone.
        if all_blocks_fit
            && let Some(index) = block
                .prefix
                .strip_prefix("blk.")
                .and_then(|n| n.parse().ok())
        {
            match compile_attention(&compiler, &cache, path, &model, index, tokens, force) {
                Ok(Some(size)) => {
                    compiled += 1;
                    bytes += size;
                }
                Ok(None) => skipped += 1,
                Err(e) => {
                    println!("  blk.{index} attention skipped: {e}");
                    failed += 1;
                }
            }
        }
        let key = BlockKey {
            model: &model,
            block: &block.cache_name(),
            tokens,
            calibration: calibration_digest(),
        };
        if !force && cache.contains(key) {
            skipped += 1;
            continue;
        }

        let read = |role: FfnRole| block.weights(path, role);
        let (gate, up, down) = match (read(FfnRole::Gate), read(FfnRole::Up), read(FfnRole::Down)) {
            (Ok(g), Ok(u), Ok(d)) => (g, u, d),
            _ => {
                println!(
                    "  {:<16} skipped: weights are not a type this reads",
                    block.prefix
                );
                failed += 1;
                continue;
            }
        };

        // Statistics, not a tensor: at least `CALIBRATION_ROWS` rows
        // whatever this graph's width is.
        let calibration = calibration_for(path, block, tokens.max(CALIBRATION_ROWS));
        let started_block = std::time::Instant::now();
        // The three-projection form, with `gelu(gate) * up` on the host.
        // The one-graph form is faster and wrong on this provider — see
        // `orangu::npu_ort::NpuOrt::compile_gated_ffn` for the bisection.
        let result = compiler.compile_gated_ffn_host(
            &orangu::npu_ort::GatedFfn {
                d_model: block.d_model,
                d_ff: block.d_ff,
                gate: &gate,
                up: &up,
                down: &down,
            },
            tokens,
            &calibration,
            activation,
        );
        match result {
            Ok(artifact) => {
                cache.store(key, &artifact)?;
                bytes += artifact.binary_len() as u64;
                compiled += 1;
                println!(
                    "  {:<16} {} -> {} in {:.1} s, {}",
                    block.prefix,
                    block.d_model,
                    block.d_ff,
                    started_block.elapsed().as_secs_f64(),
                    orangu::format::format_bytes(artifact.binary_len() as u64),
                );
            }
            Err(e) => {
                println!("  {:<16} FAILED: {e}", block.prefix);
                failed += 1;
            }
        }
    }

    println!(
        "\n{compiled} compiled ({}), {skipped} already cached, {failed} failed, in {:.1} s",
        orangu::format::format_bytes(bytes),
        started.elapsed().as_secs_f64()
    );
    if failed > 0 {
        bail!("{failed} block(s) did not compile");
    }
    Ok(())
}

/// Runs a model's cached blocks on the NPU and reports what they achieve.
pub fn run(
    path: &Path,
    tokens: usize,
    filter: Option<&str>,
    repeats: usize,
    limit: Option<usize>,
) -> Result<()> {
    if tokens == 0 || repeats == 0 {
        bail!("--tokens and --repeats must be at least 1");
    }
    let gguf = open_model(path).with_context(|| format!("reading {}", path.display()))?;
    let model = fingerprint(&gguf);
    let mut blocks = discover_blocks(&gguf, path, filter);
    // **How many blocks are resident, as a variable.** This device accepts
    // more graphs than it can hold and returns wrong answers rather than
    // refusing past some point — 48 blocks of gemma 3 12B, 5.1 GiB of
    // compiled graph, produced five blocks between 100% and 600% `rms` on
    // one run and none on the next. `Config::npu_cache_gb` is the bound
    // that keeps a served model under it, and this is how the number
    // underneath that bound gets measured on a machine nobody has seen.
    if let Some(n) = limit {
        blocks.truncate(n);
    }
    let cache = NpuCache::open(NpuCache::default_dir())?;

    let Some(runtime) = orangu::npu::NpuRuntime::open() else {
        bail!("no NPU on this machine");
    };
    let activation = host_activation(&gguf);

    println!("Running {model} at {tokens} tokens, {repeats} repeats per block\n");
    println!(
        "  {:<16} {:>10} {:>12} {:>14}",
        "BLOCK", "LOAD", "PER RUN", "RATE"
    );

    let (mut ran, mut missing) = (0usize, 0usize);
    let (mut total_ms, mut total_flops) = (0.0f64, 0.0f64);
    for block in &blocks {
        let key = BlockKey {
            model: &model,
            block: &block.cache_name(),
            tokens,
            calibration: calibration_digest(),
        };
        let Some(artifact) = cache.load(key)? else {
            missing += 1;
            continue;
        };

        let loading = std::time::Instant::now();
        let bound = match artifact.bind(&runtime) {
            Ok(b) => b,
            Err(e) => {
                println!("  {:<16} FAILED to load: {e}", block.prefix);
                continue;
            }
        };
        let load = loading.elapsed().as_secs_f64();

        let x = calibration_for(path, block, tokens);
        let mut out = Vec::new();
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = std::time::Instant::now();
            bound
                .forward_into(&x, tokens, &mut out)
                .with_context(|| format!("running {}", block.prefix))?;
            samples.push(started.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let per = samples[samples.len() / 2];

        // The same block in `f32`, on the same input. Speed without this is
        // half the answer: the device's `uint8` quantization has a real cost
        // and it varies with the model — a `Q4_K` checkpoint is dequantized
        // to `f32` and requantized to a byte on the way in, which a `Q8_0`
        // one is not.
        let error = reference_error(path, block, tokens, &x, &out, activation);

        // Three projections, so three times one projection's arithmetic.
        let flops = 3.0 * 2.0 * tokens as f64 * block.d_model as f64 * block.d_ff as f64;
        total_ms += per * 1000.0;
        total_flops += flops;
        ran += 1;
        println!(
            "  {:<16} {:>8.1} ms {:>10.3} ms {:>11.1} GF/s{}",
            block.prefix,
            load * 1000.0,
            per * 1000.0,
            flops / per / 1e9,
            match error {
                // Both, because this is the diagnostic tool: `rms` is what
                // the gate reads and `peak` is what a single bad element
                // does to it.
                Some(d) => format!(
                    "  {:>5.1}% rms {:>6.1}% peak {:>6.1}% rescaled (x{:.3})",
                    d.rms, d.peak, d.rescaled, d.alpha
                ),
                None => "                —".to_string(),
            }
        );
    }

    if ran == 0 {
        bail!(
            "nothing to run: {missing} block(s) are not in the cache — compile them first with `npu-compile`"
        );
    }
    println!(
        "\n{ran} block(s): {total_ms:.2} ms total, {:.1} GFLOP/s aggregate",
        total_flops / (total_ms / 1000.0) / 1e9
    );
    if missing > 0 {
        println!("{missing} block(s) not cached — run `npu-compile` to add them");
    }
    Ok(())
}
/// Gets the NPU ready for `model` **without holding up the server**.
///
/// Compiling a block takes about seven seconds and a model has dozens, so
/// the first start for a given model used to spend minutes before it would
/// answer anything — with a message explaining itself, which does not make
/// a server that is not yet serving any more useful.
///
/// So the whole of it — compile, then bind — runs on its own thread and the
/// caller returns immediately. Until that thread finishes, the model simply
/// has no NPU: `npu_ffn::service()` answers `None` and every layer takes the
/// path it would have taken anyway. Nothing waits, and nothing is wrong in
/// the meantime; the device joins in when it is ready.
///
/// A warm cache — every start after the first for this model — reaches the
/// bind in a second or two, so the window is only long when there is
/// genuinely minutes of work to do.
/// Set once the server is on its way out, so the preparation thread stops at
/// its next safe point instead of being abandoned mid-vendor-call.
static STOPPING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The `npu-compile` child running right now, if any.
///
/// Recorded so [`stop_preparation`] can end it. Without this the wait below
/// is unbounded in practice: a compile pass is one child that runs for as
/// long as the model has blocks — measured at 447 s for a 23-block model —
/// and `Ctrl+C` cannot be allowed to take that long.
static COMPILE_CHILD: std::sync::Mutex<Option<u32>> = std::sync::Mutex::new(None);

/// The background preparation thread, kept so it can be **joined** at exit.
static PREPARE: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> = std::sync::Mutex::new(None);

/// Whether the server is shutting down and preparation should stop.
fn stopping() -> bool {
    STOPPING.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stops NPU preparation and **waits for it to leave the vendor library**.
///
/// # Why a join, and not just a flag
///
/// `prepare_in_background` used to discard its `JoinHandle`, which detached
/// the thread. Nothing then kept the process alive until that thread was
/// finished, and `orangu::npu_ffn::shutdown` could not help: it reaches the
/// service in `SERVICE`, and during a first run there is no service yet —
/// the thread is still probing widths, compiling, or binding graphs. `main`
/// returned, the vendor library's static destructors ran underneath a thread
/// still inside its objects, and the process died with
///
/// ```text
/// shutting down
/// pure virtual method called
/// terminate called without an active exception
/// ```
///
/// Measured on a Zhouyi X2: every first run of a model interrupted during
/// preparation aborted this way (3 of 3), while runs whose blocks were
/// already cached — nothing in flight to abandon — exited cleanly (6 of 6).
///
/// # Why it is bounded
///
/// The flag alone would not be enough, because the thread spends most of its
/// time blocked in one `npu-compile` child. So the child is killed as well,
/// with `SIGKILL`: a compile whose requester is shutting down has nothing
/// worth finishing, which is the same reasoning `child::die_with_parent`
/// applies to the orphan case. What remains after that is a few checks and
/// at most one graph-binding pass, which is seconds.
///
/// Safe to call when there is no NPU, when nothing was started, and twice.
pub fn stop_preparation() {
    STOPPING.store(true, std::sync::atomic::Ordering::Relaxed);
    // The child first: the thread is almost certainly blocked waiting for
    // it, and it will not look at the flag until it comes back.
    #[cfg(unix)]
    if let Ok(child) = COMPILE_CHILD.lock()
        && let Some(pid) = *child
    {
        // SAFETY: a plain `kill(2)`. The pid is one this process spawned and
        // has not yet reaped, so it cannot have been recycled onto another
        // process.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }
    let handle = PREPARE.lock().ok().and_then(|mut h| h.take());
    if let Some(handle) = handle {
        // A join that fails is a thread that already panicked and said so;
        // either way there is nothing left to wait for.
        let _ = handle.join();
    }
}

pub fn prepare_in_background(model: &Path, enabled: bool, budget_bytes: u64) {
    // Nothing to schedule on a machine with no NPU, and this is the cheap
    // check, so make it before spending a thread on the question.
    if orangu::npu::detect_npu().is_none() {
        return;
    }

    // **Where this model's captured activations live, before anything is
    // compiled against them.** A block's activation quantizer takes its
    // range from calibration data, and synthesizing that data gets the
    // range badly wrong — see [`captured_calibration`]. So the first thing
    // this path needs is a real forward pass, and the cheapest real forward
    // pass is the one the server is about to do anyway for its first
    // request.
    // The blocks that will be compiled, which is also what has to be
    // captured — counted here rather than passed in, because this runs
    // before the weights are mapped and the caller has only the file.
    let n_layer = open_model(model)
        .ok_or(())
        .map(|gguf| discover_blocks(&gguf, model, Some("blk.")).len())
        .unwrap_or(0);
    if n_layer == 0 {
        return;
    }
    let capture = enabled
        .then(|| capture_dir(model))
        .flatten()
        .filter(|dir| !capture_complete(dir, n_layer));
    set_calibration_dir(capture_dir(model));
    if let Some(dir) = &capture {
        crate::engine::capture_ffn_inputs_into(dir.clone());
        eprintln!(
            "orangu-server: [npu] capturing this model's own activations from the first \
             prompt — the device is calibrated on them, and until it is it stays out of the \
             way"
        );
    }

    let model = model.to_path_buf();
    let spawned = std::thread::Builder::new()
        .name("orangu-npu-prepare".into())
        .spawn(move || {
            if let Some(dir) = capture {
                // Nothing to compile against yet. The wait is the point:
                // blocks built before this arrives are built against a
                // distribution the model does not have.
                let verdict = await_capture(&dir, n_layer);
                // **Before any of the arms below return.** Capturing costs
                // the model its fused GPU chains — they never form the
                // vector being captured — so leaving it on after giving up
                // charges every later prompt for a calibration that is not
                // going to happen. It was left on: both arms below returned
                // early and the stop was after the `match`, so an
                // architecture the device could not calibrate ran the rest
                // of the process unfused — for every model the device had
                // no hooks for, which until the mixture-of-experts families
                // were wired was every architecture but llama and gemma.
                crate::engine::stop_capturing_ffn_inputs();
                match verdict {
                    Capture::Complete => {}
                    Capture::Unsupported => {
                        eprintln!(
                            "orangu-server: [npu] this model's architecture never hands its \
                             feed-forward input to the device, so it cannot be calibrated and \
                             the device is not used. Only the llama and gemma families offer \
                             it today."
                        );
                        return;
                    }
                    Capture::Incomplete => {
                        eprintln!(
                            "orangu-server: [npu] no prompt arrived to calibrate on, so the \
                             device is not used this run. It will be ready for the first \
                             prompt of the next one."
                        );
                        return;
                    }
                }
                eprintln!(
                    "orangu-server: [npu] calibrated on the first prompt; compiling in the \
                     background"
                );
            }
            prepare(&model, enabled, budget_bytes)
        });
    // **Kept, not discarded.** The handle is what makes the thread joinable
    // at exit; dropping it detaches the thread and is exactly how the abort
    // in [`stop_preparation`] came about.
    match spawned {
        Ok(handle) => {
            if let Ok(mut slot) = PREPARE.lock() {
                *slot = Some(handle);
            }
        }
        Err(e) => {
            eprintln!("orangu-server: [npu] could not start the preparation thread: {e}");
        }
    }
}

/// Compile one block, measure it, and only then commit to the rest.
///
/// **The order is the point.** Compiling every block of a 42-layer model is
/// minutes and gigabytes, and most models do not survive the accuracy check
/// at the end of it — so the check comes first, on a single real block.
/// Between two and eight seconds buys the actual answer for this
/// checkpoint, on this device, and a model that fails costs that and
/// nothing more.
///
/// This replaced a cheap proxy that read `blk.0`'s gate weights and refused
/// anything whose `uint8` step was coarser than 0.105 sigma. The proxy was
/// derived from five models measured against *uniform* calibration input,
/// and it did order them. It does not order them against the calibration
/// this now compiles with (see [`realistic_calibration`]): qwen2.5 0.5B at
/// 0.155 sigma comes back 4.9% off and passes, while qwen2.5 1.5B at 0.143
/// — a finer step — comes back 8.9% and fails. A proxy that inverts on the
/// two models either side of the line is not a screen, and no threshold
/// placed on it would be honest. Measuring one block costs seconds and
/// cannot be wrong about the thing it measures.
///
/// The trial service is opened and dropped before the full one is opened,
/// so at most one bound set of blocks is resident at a time.
fn prepare(model: &Path, enabled: bool, budget_bytes: u64) {
    // The operator's off switch, and it has to be checked *here* rather
    // than only in `precompile`. Skipping the compile alone left a machine
    // with a warm cache binding and serving from the device after it had
    // been turned off — which also made every A/B measurement of "with the
    // NPU" against "without" a comparison of the device against itself.
    if !enabled {
        return;
    }
    // Between every stage below. Each one is minutes of work that ends in
    // the vendor library, and the point of stopping is to not be inside it
    // when `main` returns — see `stop_preparation`.
    if stopping() {
        return;
    }
    // One block, compiled if it is not cached already. If the model is
    // going to be refused, this is the whole cost of finding out.
    precompile(model, enabled, budget_bytes, Some(1));
    if stopping() {
        return;
    }
    let bar = max_block_error();
    match trial_block_error(model) {
        Some(error) if error > bar => {
            eprintln!(
                "orangu-server: [npu] not used for this model: one block comes back {:.1}% \
                 off f32, past the {:.0}% this accepts. The device holds each weight matrix \
                 at one `uint8` scale, so a matrix with a wide tail leaves its bulk too few \
                 levels — a property of this model's weights, not of the file's \
                 quantization. A higher-precision build of the *same* model will not help.",
                error * 100.0,
                bar * 100.0
            );
            return;
        }
        // Nothing compiled, nothing to bind, or no readable reference. Not
        // a refusal to announce — `precompile` and `install_ffn_service`
        // have already said whatever there was to say — and the full path
        // below will reach the same conclusion and report it properly.
        None => {}
        Some(_) => {}
    }
    // **Ask the device how wide it wants to be, before compiling the set.**
    // One block at each candidate, once per machine — see
    // `probe_prefill_width`.
    if stopping() {
        return;
    }
    if forced_prefill_width().is_none()
        && let Ok(executable) = std::env::current_exe()
        && let Some(width) = probe_prefill_width(model, &executable)
    {
        MEASURED_WIDTH.store(width, std::sync::atomic::Ordering::Relaxed);
        eprintln!("orangu-server: [npu] compiling for {width}-token chunks on this device");
    }
    if stopping() {
        return;
    }
    precompile(model, enabled, budget_bytes, None);
    // The last one, and the one that matters most: installing binds every
    // graph on the device. Starting that while the process is on its way out
    // is the worst of both — a long vendor call nobody will use.
    if stopping() {
        return;
    }
    install_ffn_service(model, budget_bytes);
}

/// [`block_error`] on `blk.0` alone, with the service opened and dropped
/// around it.
///
/// `None` when there is nothing to measure — no gated blocks, nothing
/// compiled for a width this model can use, or weights this cannot read as
/// `f32`. A caller must not read that as a pass or a failure.
fn trial_block_error(model: &Path) -> Option<f32> {
    let gguf = open_model(model)?;
    let first = discover_blocks(&gguf, model, Some("blk."))
        .into_iter()
        .next()?;
    let index = first.prefix.strip_prefix("blk.")?.parse().ok()?;
    let widths = widths_for(&gguf);
    let service = orangu::npu_ffn::NpuFfn::open(
        model,
        &[(index, first.prefix)],
        &widths,
        calibration_digest(),
    )?;
    block_error(&gguf, model, &service, &widths)
}

/// Loads this model's cached language-model blocks onto the NPU, so the
/// prefill path can use them.
///
/// Language-model blocks only — `blk.N`, not `v.blk.N`. A vision tower has
/// nothing to consume it (this engine has no vision encoder), and the
/// language model's prefill does.
///
/// Silent and cheap when there is nothing to do, which is every machine
/// without an NPU and every model that has not been compiled for this
/// width. `tokens` must be the prefill width the blocks were compiled at; a
/// request of any other width goes to the CPU or GPU as before.
pub fn install_ffn_service(model: &Path, budget_bytes: u64) {
    let Some(gguf) = open_model(model) else {
        return;
    };
    // **Bounded here too, not only in `precompile`.** The budget is a bound
    // on resident memory, and a block is resident once it *binds* — so a
    // warm cache bound everything it had regardless of what the operator
    // allowed, and `npu_cache_gb` silently stopped meaning anything after
    // the first run. Same estimate as `precompile` uses, so the two agree
    // on what fits.
    let mut spent = 0u64;
    let blocks: Vec<(usize, String)> = discover_blocks(&gguf, model, Some("blk."))
        .into_iter()
        .take_while(|block| {
            spent += (3 * block.d_model * block.d_ff) as u64;
            spent <= budget_bytes
        })
        .filter_map(|block| {
            let index = block.prefix.strip_prefix("blk.")?.parse().ok()?;
            Some((index, block.prefix))
        })
        .collect();
    if blocks.is_empty() {
        return;
    }
    let widths = widths_for(&gguf);
    let Some(service) =
        orangu::npu_ffn::NpuFfn::open(model, &blocks, &widths, calibration_digest())
    else {
        return;
    };

    // **Whether this model should use the device is this model's question**,
    // and it is answered by measurement rather than by any property of the
    // file. `prepare` has already asked it of one block before any of these
    // were compiled; this asks again of the set that actually bound, which
    // is the one that will serve. Cheap either way: one inference and one
    // host-side reference.
    match block_error(&gguf, model, &service, &widths) {
        Some(error) if error <= max_block_error() => {
            // `in use` rather than a bare count, to pair with the startup
            // inventory's `— preparing feed-forward blocks` and with the
            // refusal below, which says `not used for this model`. A reader
            // scanning for which processors are working should find the
            // same words on each of them.
            eprintln!(
                "orangu-server: [npu] in use — {} feed-forward block-width(s) \
                 (widths {widths:?}, {:.1}% off f32)",
                service.len(),
                error * 100.0
            );
            orangu::npu_ffn::install(Some(service));
        }
        Some(error) => eprintln!(
            "orangu-server: [npu] not used for this model: its blocks come back {:.1}% off \
             f32, past the {:.0}% this accepts. The device holds each weight matrix at one \
             `uint8` scale, so a matrix with a wide tail leaves its bulk too few levels — \
             a property of this model's weights, not of the file's quantization. A \
             higher-precision build of the *same* model will not help.",
            error * 100.0,
            max_block_error() * 100.0
        ),
        // Weights this cannot read, so nothing to compare against. Refusing
        // is the safe direction: an unmeasured path is not a qualified one.
        None => eprintln!(
            "orangu-server: [npu] not used for this model: its blocks could not be checked \
             against an `f32` reference"
        ),
    }
}

/// The quantization step one `uint8` scale gives a weight tensor, in
/// standard deviations of its own values.
///
/// How coarsely the device can represent the bulk of a matrix: it holds
/// each at a single scale, so the 256 levels are spread over whatever the
/// widest value is and a fat tail starves the middle. This was once a
/// *screen* — see [`prepare`] for why it is not one any more, and why the
/// one-block trial replaced it. It stays because it is the one number that
/// says something about a checkpoint without touching the device, which is
/// what the measurement test reports.
#[cfg(test)]
fn conditioning(path: &Path, name: &str) -> Option<f32> {
    let w = read_dequantized(path, name).ok()?;
    let (lo, hi) = w
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let mean = w.iter().sum::<f32>() / w.len() as f32;
    let variance = w.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w.len() as f32;
    let sigma = variance.sqrt();
    (sigma > 0.0).then(|| (hi - lo) / 255.0 / sigma)
}

/// How far off `f32` a block may come back and still be worth running on
/// the device.
///
/// **One bar, and a loose one, because that is what the measurements
/// support.** Every model put through the whole path — captured
/// calibration, per-channel smoothing, a 720-token prompt, device off
/// against device on — answers correctly, at block errors spread over an
/// order of magnitude:
///
/// | model | worst sampled block | prefill | answer |
/// |---|---|---|---|
/// | gemma 4 E2B `Q8_0` | ~11% rms | 4.44 -> 12.30 tok/s | correct |
/// | gemma 4 E4B `Q8_0` | under 75% | 2.43 -> 7.00 | correct |
/// | qwen3 4B `Q8_0` | **53.6%** | 2.72 -> 7.15 | correct |
/// | llama 3.2 3B `Q8_0` | under 75% | 3.65 -> 10.21 | correct |
///
/// A block 53.6% off `f32` in every one of 36 layers still produced a
/// sound explanation of a B-tree. The residual stream dilutes what the
/// feed-forward network contributes, and the model absorbs far more than a
/// per-block figure suggests.
///
/// So this is a screen against a *broken* block rather than a quality
/// judgment: 100% rms means the error equals the signal — which is what a
/// block compiled against the wrong activation distribution measures, and
/// those return output with no energy in it at all. 75% sits between the
/// worst thing that works and the best thing that does not.
///
/// **There were two bars here and the split did not survive.** The tighter
/// one applied to models that add their feed-forward output to the residual
/// stream raw (llama, qwen, granite) against those that normalize it first
/// (gemma), on the evidence that qwen3 stopped generating where gemma did
/// not. Both of those measurements predated the calibration fix, when
/// blocks on both sides were returning nothing; qwen3 4B is the model in
/// the table above.
///
/// **The table below is a different statistic** — `Deviation::peak`, on
/// `blk.0`, at the width each model served when it was measured, against
/// calibration shaped like a real hidden state ([`realistic_calibration`]).
/// The gate reads `Deviation::rms` now, which runs two to three times
/// higher, and every model is compiled at one width. It is kept because
/// what it shows does not depend on either choice. All `Q8_0` unless noted:
///
/// | model | w | peak error | |
/// |---|---|---|---|
/// | qwen3 4B | 1 | 3.2% | used |
/// | gemma 4 E2B | 16 | 3.6% | used |
/// | gemma 4 E4B | 16 | 3.7% | used |
/// | qwen2.5 coder 0.5B | 1 | 4.0% | used |
/// | gemma 3 4B | 16 | 4.3% | used |
/// | gemma 4 E2B `Q4_K_M` | 16 | 4.7% | used |
/// | gemma 3 4B `Q4_K_M` | 16 | 4.8% | used |
/// | mistral 7B v0.3 | 1 | 5.6% | refused |
/// | gemma 2 2B | 16 | 8.5% | refused |
/// | gemma 3 1B | 16 | 10.2% | refused |
/// | qwen2.5 coder 1.5B | 1 | 10.2% | refused |
/// | llama 3.2 3B | 1 | 10.3% | refused |
/// | granite 3.1 2B | 1 | 12.5% | refused |
/// | SmolLM2 1.7B | 1 | 13.9% | refused |
/// | llama 3.2 1B | 1 | 15.1% | refused |
/// | qwen2.5 1.5B | 1 | 15.8% | refused |
/// | granite 3.1 8B | 1 | 23.4% - 31.3% | refused |
///
/// **Width belongs in the table.** An earlier version measured every model
/// at 16 and read across families as though that were comparable. It is
/// not: llama-3.2-1B is 7.7% at width 16 and 15.1% at width 1, which is the
/// width it serves. Reading a decode-seam model at a prefill width flatters
/// it by half, and a table of such numbers looks like an effect when
/// nothing changed but the column.
///
/// A bar of 5% on *that* statistic sat in a real gap: seven measurements
/// between 3.2% and 4.8%, then nothing until 5.6%. The line matters because
/// this compounds across every layer — at around 12% the model answered a
/// question about hash tables with hash browns.
///
/// What the gap did not survive is measuring more than one block. `blk.0`
/// turns out to be a lottery ticket: gemma 4 E2B, which is *token-identical*
/// with the device on, ranges from 6.7% to 23.8% `rms` across its 35
/// blocks, and its `blk.0` is one of its best. Reading a whole model's
/// suitability off its first block is how a 5% bar and a 23.8% model
/// coexisted for as long as they did. See [`block_error`], which now
/// samples over depth.
///
/// Nothing about a model predicts where it lands. Not size within a family
/// (qwen2.5 coder: 4.0% at 0.5B, 10.2% at 1.5B, 6.4% at 7B). Not the family
/// (gemma 3 is 4.3% at 4B and 10.2% at 1B). Not the fine-tune, in the
/// direction anyone would guess — qwen2.5 1.5B *base* is 15.8% against its
/// coder tune's 10.2%, so the tune improved conditioning. Not the lineage:
/// SmolLM2 and granite are separate ancestries and both land badly. And not
/// even the name — two uploads of Qwen2.5-Coder-7B-Instruct at `Q8_0` have
/// identical tensor tables (same names, shapes, offsets), different weight
/// bytes, and measure 6.4% and 13.0%. There is no shortcut around measuring
/// the file in hand, which is what [`prepare`] does.
///
/// The file's own quantization moves it, in whichever direction the model's
/// tail needs: `Q4_K_M` beats `Q8_0` on a badly-conditioned checkpoint
/// (gemma 3 1B: 7.6% against 10.2%) and loses on a well-conditioned one
/// (gemma 3 4B: 4.8% against 4.3%). See `orangu::npu_ort`'s
/// `weight_grid_clipping_does_not_help` for what that hint was worth when
/// followed up: nothing, twice.
///
/// **What this threshold does not cover.** It is a check on the device, and
/// a block that passes says nothing about the rest of the engine. Three
/// models measured here generated degenerate text on this machine *with the
/// NPU off*, and two of the three turned out to be bugs elsewhere in this
/// engine rather than facts about the checkpoints:
///
/// - gemma 4 E2B at `Q4_K_M`, blamed on the Vulkan backend reporting its
///   own `Q4_K` matmul "3.3% adrift" at startup. That figure was the
///   probe's own `int8` activation rounding, and the real fault was five
///   kernels built against `supports_subgroup` — a capability — where every
///   other kernel reads the `ORANGU_SUBGROUP` policy. It serves correctly
///   now, at 13.34 tok/s of prefill against 2.11. Two other checkpoints
///   written off for the same reason — qwen3 1.7B and llama 3.2 3B, both
///   `Q4_K_M` — also answer correctly on that backend now, so the whole
///   "this device gets `Q4_K` wrong" line of reasoning was mistaken and
///   anything resting on it is void.
/// - gemma 3 4B, which is wrong on the Vulkan *and* the CPU backend and so
///   is broken in its architecture rather than in any backend. Still open.
/// - qwen3 4B, a fused decode chain passing `None` for per-head Q/K norms
///   that this family does have. It answers correctly now.
///
/// A device refused for a bug somewhere else is the failure mode this
/// paragraph exists for, and it has now caught two.
///
/// Which is why every claim here is checked against the reference path
/// rather than against this bar alone — and why the check has to be made at
/// the length a prompt actually is. gemma 4 E2B at `Q8_0` and gemma 3 4B
/// both generate *token-identical* output at temperature 0 with the device
/// on and off, on a sixteen-token prompt, and that is the whole of what
/// that measurement showed. See [`error_accumulates_across_chunks`].
const MAX_BLOCK_ERROR: f32 = 0.75;

/// The bar in force for this model, and the override that lets a refused
/// one through for measurement.
///
/// **A measurement aid, not a setting.** It is not in the config file and
/// is not documented for users, because the honest answer to "my model was
/// refused" is a different checkpoint rather than a wider bar. It exists
/// because the bar can only be calibrated by watching what a model *does*
/// when it is let through, and that experiment needs a way to let a refused
/// model through:
///
/// ```text
/// ORANGU_NPU_MAX_BLOCK_ERROR=1.0 orangu-server <model>
/// ```
///
/// Read fresh each time rather than cached in a `OnceLock`: the two callers
/// are a background thread at load, so this runs a handful of times per
/// process and being able to see the value change between runs in a test is
/// worth more than the lookup.
fn max_block_error() -> f32 {
    std::env::var("ORANGU_NPU_MAX_BLOCK_ERROR")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(MAX_BLOCK_ERROR)
}

/// The worst relative error the device introduces, across a sample of this
/// model's blocks, against the same blocks in `f32`.
///
/// **The worst over depth, not layer 0's.** This measured layer 0 alone
/// until gemma 3 12B showed what that misses. One checkpoint, one width,
/// four blocks:
///
/// | block | 0 | 10 | 20 | 40 |
/// |---|---|---|---|---|
/// | error | 6.5% | 5.1% | 7.2% | 16.3% |
///
/// Conditioning is not a property a checkpoint has once. It drifts with
/// depth, and it does not drift monotonically either — layer 10 is better
/// than layer 0 in that model and layer 40 is three times the bar. A model
/// whose first block passed and whose last was 16% off would have bound
/// every block and served the bad ones in silence, which is the one failure
/// this gate exists to prevent.
///
/// Layer 0 stays a fair proxy where it was checked against the whole model
/// — gemma 3 4B measures 4.3% at `blk.0` and 4.3% installed, gemma 4 E2B
/// 3.6% and 3.6% — so this is not a correction of those numbers. It is a
/// guard against the checkpoint where it would not have been.
///
/// **What sampling cannot promise.** A block that measures well at load can
/// still come back wrong later. Binding all 48 blocks of gemma 3 12B at
/// once — 5.1 GiB of compiled graph, well past the 3 GiB
/// `Config::npu_cache_gb` allows a served model — produced five blocks
/// between 100% and 600% `rms` on one run and none at all on the next, with
/// the same cache and the same inputs. Nothing was reported by the runtime
/// either time. So this gate is a screen at load rather than a guarantee
/// about every later inference.
///
/// **It is not simply a function of how much is resident.** Binding
/// granite 3.1 8B a block at a time and measuring every bound block, worst
/// `rms`:
///
/// | resident | 1.2 GiB | 2.3 | 3.5 | 4.7 | 5.9 |
/// |---|---|---|---|---|---|
/// | worst | 29.2% | 29.2% | 36.4% | 36.4% | 39.5% |
///
/// Nothing over 90% at any size, past the 5.1 GiB where the corruption
/// appeared. The creep from 29% to 39% is more blocks meaning more chances
/// at the tail, not a cliff. So "too many bytes bound" is ruled out as the
/// whole story, and what is left is an intermittent fault that has now
/// failed to reproduce deliberately on a second model — which is a reason
/// to keep `Config::npu_cache_gb` where it is rather than a reason to
/// trust any particular value of it. `npu-run --limit N` is how this was
/// measured and how to measure it again.
///
/// [`SAMPLED_BLOCKS`] blocks spread over the model at the narrowest
/// compiled width, worst one decides. Blocks the service does not hold are
/// skipped rather than failing the model: a budget-truncated cache binds
/// only its first N, and the ones it never bound are not the ones it will
/// serve.
fn block_error(
    gguf: &GgufFile,
    path: &Path,
    service: &orangu::npu_ffn::NpuFfn,
    widths: &[usize],
) -> Option<f32> {
    let blocks = discover_blocks(gguf, path, Some("blk."));
    let activation = host_activation(gguf);

    let mut worst: Option<f32> = None;
    for block in sample_over_depth(&blocks, SAMPLED_BLOCKS) {
        // The layer index the service knows this block by, from its own
        // name. Position in `blocks` is not it: a file whose blocks are not
        // a complete run from zero would be measured against the wrong
        // weights, which is a quieter kind of wrong than not measuring.
        let Some(index) = block
            .prefix
            .strip_prefix("blk.")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(&tokens) = widths.iter().find(|w| service.has(index, **w)) else {
            continue;
        };
        let x = calibration_for(path, block, tokens);
        let mut got = Vec::new();
        if !service.forward_into(index, tokens, &x, &mut got) {
            continue;
        }
        if let Some(d) = reference_error(path, block, tokens, &x, &got, activation) {
            worst = Some(worst.map_or(d.rms, |seen: f32| seen.max(d.rms)));
        }
    }
    // Still `None` when nothing could be measured, which `install_ffn_service`
    // reads as "refuse" — an unmeasured path is not a qualified one.
    worst.map(|pct| pct / 100.0)
}

/// How many blocks [`block_error`] measures.
///
/// Five is the point where the cost stops mattering and the coverage stops
/// improving much: five inferences and five host-side references are under
/// a second against a compile that is seconds *per block*, and a model with
/// a bad patch narrow enough to hide between five evenly spread samples is
/// not one this could catch by sampling at all.
const SAMPLED_BLOCKS: usize = 5;

/// Up to `n` of these blocks, evenly spread, first and last always among
/// them.
///
/// The ends matter most — degradation showed up at the *end* of gemma 3 12B
/// — so the sample is anchored there rather than being taken from the front
/// like a `--limit`.
fn sample_over_depth(blocks: &[BlockSpec], n: usize) -> Vec<&BlockSpec> {
    if blocks.len() <= n || n < 2 {
        return blocks.iter().collect();
    }
    (0..n)
        .map(|i| &blocks[i * (blocks.len() - 1) / (n - 1)])
        .collect()
}

/// Compiles this model's feed-forward blocks for the NPU, before it is
/// served.
///
/// **Language-model blocks**, not the vision tower's. This used to compile
/// `v.blk.*` from the companion projector, which was work with no consumer:
/// this engine has no vision encoder, so nothing ever ran them. `blk.*` is
/// what `install_ffn_service` binds and what the prefill path asks for.
///
/// Quiet and cheap in every case but one. Nothing without an NPU, nothing
/// when the cache is already warm — which is every start after the first —
/// and nothing at all if `npu_precompile` is off. The one case that costs
/// anything is a model this machine has not compiled, and that is announced
/// rather than left to look like a hang: it is about seven seconds and 55
/// MiB per block, so a 42-block model is three and a half minutes and 2.4
/// GiB.
///
/// **Bounded by `budget_bytes`**, because that cache is not free: every
/// block compiled is its weights again in `uint8` beside the model's own,
/// and all of it is resident once bound. Blocks are compiled in order until
/// the budget is reached, and what is already cached counts against it. A
/// model too large for the budget gets its first N layers on the device and
/// the rest on the CPU or GPU, which is a partial speedup rather than an
/// error — 12 of 42 blocks measured 18.5 tok/s of prefill against 15.0 with
/// none.
///
/// Failures here are reported and then ignored. Nothing about serving a
/// model depends on this having worked.
///
/// Called from [`prepare_in_background`], never on the startup path: this
/// takes about seven seconds a block, and a server that waits for it is a
/// server that takes minutes to answer its first request.
fn precompile(model: &Path, enabled: bool, budget_bytes: u64, cap: Option<usize>) {
    if !enabled || budget_bytes == 0 {
        return;
    }
    // Cheapest check first: an ordinary machine has no NPU and should pay
    // nothing at all for this.
    if orangu::npu::detect_npu().is_none() {
        return;
    }
    let Some(gguf) = open_model(model) else {
        return;
    };
    let fingerprint = fingerprint(&gguf);
    let mut blocks = discover_blocks(&gguf, model, Some("blk."));
    // `cap` is how the trial in `prepare` asks for one block and no more.
    if let Some(cap) = cap {
        blocks.truncate(cap);
    }
    // Only the widths this model can actually use — see `widths_for`.
    let widths = widths_for(&gguf);
    // **Do not compile blocks the device has already refused to hold.**
    //
    // The limit is discovered when binding, but it is paid for here: a block
    // past it costs the same seven seconds and the same disk as one that
    // fits, and then never runs. Measured on gemma 4 E2B `Q4_K_M` — 35
    // feed-forward blocks compiled, 34 accepted — the last block's compile
    // bought nothing at all.
    //
    // Only for a single width, and that restriction is the point rather than
    // caution. `NpuFfn::open` offers artifacts width-major — every layer at
    // the first width, then every layer at the second — and truncates at the
    // device's limit, so with two widths a limit of 46 means all of the
    // prefill width and half of the decode width. Dividing the limit by the
    // number of widths would cap blocks at 23 and throw away prefill
    // coverage that binds perfectly well today, to protect decode coverage
    // that the seam's 80% rule may decline anyway. With one width, artifacts
    // and blocks are the same thing and the arithmetic is exact.
    if widths.len() == 1
        && let Some(limit) = orangu::npu_ffn::capacity_hint(&fingerprint, &widths)
        && limit < blocks.len()
    {
        eprintln!(
            "orangu-server: [npu] compiling {limit} of {} feed-forward block(s): this device \
             would not hold more at {} tokens last time",
            blocks.len(),
            widths[0]
        );
        blocks.truncate(limit);
    }
    let Ok(cache) = NpuCache::open(NpuCache::default_dir()) else {
        return;
    };

    // What a block costs at one width, from its own weights rather than a
    // guess: three matrices quantized to a byte each, which is what the
    // compiled graphs carry. Measured against the real artifacts this runs
    // about 20% high, so the budget is respected with room rather than
    // overshot.
    let cost = |block: &BlockSpec| (3 * block.d_model * block.d_ff) as u64;

    let Ok(executable) = std::env::current_exe() else {
        return;
    };
    // Nothing to compile is not a budget problem, and saying so as one sent
    // me looking for a broken cache accountant when the real answer was that
    // phi3 fuses `ffn_gate` and `ffn_up` into a single tensor, so
    // `discover_blocks` correctly finds no gated blocks at all.
    if blocks.is_empty() {
        return;
    }

    let name = model.file_name().unwrap_or_default().to_string_lossy();
    // **What this model actually needs**, which is the number the budget
    // should be compared against: every block, at every width being
    // compiled. A ceiling derived from the machine says what *can* be
    // spent; this says what is worth spending, and reserving more than a
    // model can use helps nothing.
    let needed: u64 = blocks.iter().map(cost).sum::<u64>() * widths.len() as u64;
    let mut budget = budget_bytes.min(needed);
    if needed > budget_bytes {
        eprintln!(
            "orangu-server: [npu] {name} wants {} of compiled blocks and the budget is {} — \
             raise `npu_cache_gb` to put all of it on the device. Measured on Meta-Llama 3.1 \
             8B `Q8_0`: 18 of 32 blocks prefills at 2.52 tok/s, all 32 at 5.36.",
            orangu::format::format_bytes(needed),
            orangu::format::format_bytes(budget_bytes)
        );
    }

    // Widths in order, and the order is the priority: `widths_for` puts
    // prefill first because that is where the device beats what is deployed
    // by the largest margin. A budget that runs out mid-way leaves the later
    // widths — and the later layers of the width it ran out on — where they
    // were, which is a partial speedup rather than a failure.
    for &tokens in &widths {
        let mut affordable = blocks
            .iter()
            .scan(0u64, |spent, block| {
                *spent += cost(block);
                Some(*spent)
            })
            .take_while(|spent| *spent <= budget)
            .count();
        // **Decode is all or nothing.** Prefill asks the device one layer at
        // a time and keeps whatever it is given, so half a model's blocks is
        // half a speedup. The decode seam is not like that: taking it costs
        // a submit-and-read *per layer* for the whole step, and a layer the
        // device does not hold pays that round trip and then computes the
        // network on the backend anyway. Measured on llama 3.2 3B `Q8_0`
        // with 14 of 28 decode blocks affordable, decode ran at 3.01 tok/s
        // against 3.13 with the width switched off entirely — so a partial
        // decode set is worse than none, and compiling it would also spend
        // half a gigabyte of cache on blocks `record_decode_run` declines to
        // use. See the layer sweep in `LlamaModel::record_decode_run`.
        if tokens == DECODE_WIDTH && affordable < blocks.len() {
            eprintln!(
                "orangu-server: [npu] the {} cache budget does not cover every layer at \
                 {tokens} token, and a partial decode set is slower than none — skipped. \
                 Raise `npu_cache_gb` to use the device on decode too.",
                orangu::format::format_bytes(budget_bytes)
            );
            affordable = 0;
        }
        if affordable == 0 {
            if tokens != DECODE_WIDTH {
                eprintln!(
                    "orangu-server: [npu] no room left for {tokens}-token blocks within the {} \
                     cache budget. Raise `npu_cache_gb` to use more of the device.",
                    orangu::format::format_bytes(budget_bytes)
                );
            }
            break;
        }
        budget -= blocks[..affordable].iter().map(cost).sum::<u64>();

        let missing = blocks[..affordable]
            .iter()
            .filter(|block| {
                !cache.contains(BlockKey {
                    model: &fingerprint,
                    block: &block.prefix,
                    tokens,
                    calibration: calibration_digest(),
                })
            })
            .count();
        if affordable < blocks.len() {
            eprintln!(
                "orangu-server: [npu] {} block(s) at {tokens} tokens left off: they would \
                 exceed the {} cache budget",
                blocks.len() - affordable,
                orangu::format::format_bytes(budget_bytes)
            );
        }
        if missing == 0 {
            continue;
        }
        if cap.is_none() {
            eprintln!(
                "orangu-server: [npu] compiling {missing} feed-forward block(s) of {name} at \
                 {tokens} tokens — one time, about seven seconds each"
            );
        }

        // A child process, because this one will go on to open the NPU
        // runtime and the two stacks cannot share an address space. One
        // invocation per width, not per block: the compiler loads the
        // vendor's ONNX Runtime on the way in, and paying that 42 times
        // would be minutes of nothing.
        let started = std::time::Instant::now();
        let status = compile_child(&executable, model, tokens, affordable);
        match status {
            Ok(status) if status.success() => {
                if cap.is_none() {
                    eprintln!(
                        "orangu-server: [npu] {missing} block(s) at {tokens} tokens compiled \
                         in {:.0}s, cached in {}",
                        started.elapsed().as_secs_f64(),
                        cache.dir().display()
                    );
                }
            }
            Ok(status) => {
                eprintln!(
                    "orangu-server: [npu] precompile exited with {status}; continuing without it"
                );
                break;
            }
            Err(e) => {
                eprintln!("orangu-server: [npu] could not run the compiler: {e}");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// **`main` calls this on every exit path**, including the ones that
    /// never touched the device — no NPU, no model, a config error before
    /// anything started. It has to be a no-op there rather than a panic or a
    /// wait, and it has to survive being called twice.
    ///
    /// The flag is restored afterwards because the whole suite shares one
    /// process: leaving it set would tell any later `compile_child` that the
    /// server is shutting down.
    #[test]
    fn stopping_preparation_that_never_started_is_a_no_op() {
        use std::sync::atomic::Ordering;
        let was = super::STOPPING.load(Ordering::Relaxed);
        super::stop_preparation();
        super::stop_preparation();
        assert!(
            super::STOPPING.load(Ordering::Relaxed),
            "the flag must be set even when there was no thread to stop, or a \
             preparation starting a moment later would not see it"
        );
        assert!(
            super::COMPILE_CHILD
                .lock()
                .is_ok_and(|child| child.is_none()),
            "there was no compile child to record"
        );
        super::STOPPING.store(was, Ordering::Relaxed);
    }

    /// **A failed compile has to carry its reason.** Before this, the only
    /// thing said about one was `precompile exited with <status>` — the
    /// compiler's own explanation had gone to the terminal, interleaved with
    /// whatever the server was printing at the time, and was effectively
    /// unrecoverable. The tail is what matters: the vendor compiler narrates
    /// every block, so the start of its output is the part least likely to
    /// say why the end failed.
    #[test]
    fn a_failed_compile_reports_the_tail_of_what_the_compiler_said() {
        let mut chatter = String::new();
        for i in 0..50 {
            chatter.push_str(&format!("compiling block {i}\n"));
        }
        chatter.push_str("error: out of device memory\n");

        let lines = super::compiler_report_lines(chatter.as_bytes());
        assert_eq!(
            lines.last().map(String::as_str),
            Some("error: out of device memory"),
            "the reason the compile failed must survive: {lines:?}"
        );
        // One elision notice plus the cap, and the reader is told what was
        // dropped rather than silently given a slice.
        assert_eq!(lines.len(), super::COMPILER_REPORT_LINES + 1);
        assert!(
            lines[0].contains("31 earlier line(s) omitted"),
            "the reader must be told how much was cut, got {:?}",
            lines[0]
        );
    }

    /// Short output is passed through whole — no elision notice for output
    /// that was never elided.
    #[test]
    fn a_short_compiler_failure_is_reported_in_full() {
        let lines = super::compiler_report_lines(b"error: bad graph\naborting\n");
        assert_eq!(lines, ["error: bad graph", "aborting"]);
    }

    /// **Silence stays silent.** A compiler that failed without saying
    /// anything must not produce a bare `compiler:` prefix with nothing after
    /// it, and the blank lines this output is mostly made of must not each
    /// become a line of their own.
    #[test]
    fn compiler_output_that_is_only_whitespace_reports_nothing() {
        assert!(super::compiler_report_lines(b"").is_empty());
        assert!(super::compiler_report_lines(b"\n\n   \n\t\n").is_empty());
    }

    use super::*;
    use orangu::gguf::TensorInfo;

    fn tensor(name: &str, dims: Vec<u64>) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims,
            ggml_type: 8,
            offset: 0,
        }
    }

    fn model(tensors: Vec<TensorInfo>) -> GgufFile {
        GgufFile {
            version: 3,
            metadata: Vec::new(),
            tensors,
            alignment: 32,
            data_offset: 64,
        }
    }

    fn specs(n: usize) -> Vec<BlockSpec> {
        (0..n)
            .map(|i| BlockSpec {
                prefix: format!("blk.{i}"),
                names: FFN_DENSE,
                d_model: 8,
                d_ff: 16,
                range: None,
            })
            .collect()
    }

    /// The sample the quality gate measures spans the whole model, ends
    /// included. gemma 3 12B degrades towards its last block — a sample
    /// that stopped early would call it a 6.5% model when its `blk.40` is
    /// 16.3%.
    #[test]
    fn the_sampled_blocks_span_the_model_and_include_both_ends() {
        let blocks = specs(48);
        let picked: Vec<&str> = sample_over_depth(&blocks, SAMPLED_BLOCKS)
            .iter()
            .map(|b| b.prefix.as_str())
            .collect();
        assert_eq!(picked, ["blk.0", "blk.11", "blk.23", "blk.35", "blk.47"]);
    }

    /// A model with fewer blocks than the sample size is measured whole,
    /// without repeating any block or indexing past the end.
    #[test]
    fn a_short_model_is_sampled_entirely() {
        for n in 0..=SAMPLED_BLOCKS {
            let blocks = specs(n);
            assert_eq!(sample_over_depth(&blocks, SAMPLED_BLOCKS).len(), n);
        }
    }

    /// How coarsely one `uint8` scale represents each of a model's three
    /// projections — [`conditioning`] for `ffn_gate`, `ffn_up` and
    /// `ffn_down` of `blk.0`.
    ///
    /// Reported, not acted on. This was a screen once and is not one now:
    /// it does not order the models by the error they actually come back
    /// with — see [`prepare`]. It is still the cheapest thing that says
    /// anything about a checkpoint without a device, so it is worth being
    /// able to look at when a new model behaves unexpectedly.
    ///
    /// Ignored, because it needs real checkpoints and those are not in the
    /// tree. Point `ORANGU_MODELS` at a directory of `.gguf` files and run
    ///
    /// ```text
    /// ORANGU_MODELS=... cargo test --bin orangu-server -- --ignored --nocapture
    /// ```
    ///
    /// to place a new model against the ones already measured, without
    /// compiling anything or needing a device. It reads one tensor per file.
    #[test]
    #[ignore = "needs real checkpoints; set ORANGU_MODELS"]
    fn gate_conditioning_of_the_models_on_this_machine() {
        let Ok(dir) = std::env::var("ORANGU_MODELS") else {
            eprintln!("set ORANGU_MODELS to a directory of .gguf files");
            return;
        };
        let mut files = Vec::new();
        collect_gguf(Path::new(&dir), &mut files);
        files.sort();
        for file in files {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            let Ok(gguf) = GgufFile::open(&file) else {
                println!("{name:<44} unreadable");
                continue;
            };
            let blocks = discover_blocks(&gguf, &file, Some("blk."));
            let Some(first) = blocks.first() else {
                // Phi-3.5 lands here: it fuses gate and up into one tensor,
                // so there is no gated block to compile or to measure.
                println!("{name:<44} no gated blocks");
                continue;
            };
            // All three, because the gate alone was what the old screen
            // read and the gate alone is what made it wrong.
            let steps: Vec<String> = ["ffn_gate", "ffn_up", "ffn_down"]
                .iter()
                .map(|projection| {
                    match conditioning(&file, &format!("{}.{projection}.weight", first.prefix)) {
                        Some(step) => format!("{step:>6.3}"),
                        None => "     ?".to_string(),
                    }
                })
                .collect();
            println!(
                "{name:<44} {} sigma  ({} blocks)",
                steps.join(" "),
                blocks.len()
            );
        }
    }

    /// What splitting a projection into row groups would buy.
    ///
    /// The device holds one `uint8` scale per weight matrix, and that single
    /// scale is the whole reason most models are refused: a fat tail spreads
    /// the 256 levels over a range the bulk never reaches. Per-channel
    /// scales are not available from this provider — asking for them
    /// produced 79% and 104% error — but a *group* of output channels
    /// compiled as its own operator is still one scale per matrix, which is
    /// the supported mode.
    ///
    /// So this asks the only question worth asking before building any of
    /// it: on the real weights, how far does the error fall as the rows are
    /// split into 2, 4, 8, 16 groups? If it barely moves, the tail is spread
    /// across every channel and splitting is wasted work. Ignored, and run
    /// the same way as [`gate_conditioning_of_the_models_on_this_machine`].
    #[test]
    #[ignore = "needs real checkpoints; set ORANGU_MODELS"]
    fn row_groups_against_one_scale_per_matrix() {
        let Ok(dir) = std::env::var("ORANGU_MODELS") else {
            eprintln!("set ORANGU_MODELS to a directory of .gguf files");
            return;
        };
        let mut files = Vec::new();
        collect_gguf(Path::new(&dir), &mut files);
        files.sort();
        println!(
            "{:<44} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "model", "1", "2", "4", "8", "16"
        );
        for file in files {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            let Ok(gguf) = GgufFile::open(&file) else {
                continue;
            };
            let blocks = discover_blocks(&gguf, &file, Some("blk."));
            let Some(first) = blocks.first() else {
                continue;
            };
            let Ok(w) = read_dequantized(&file, &format!("{}.ffn_gate.weight", first.prefix))
            else {
                continue;
            };
            let rows = first.d_ff;
            let cols = first.d_model;
            if w.len() != rows * cols {
                continue;
            }
            let errors: Vec<String> = [1usize, 2, 4, 8, 16]
                .iter()
                .map(|g| format!("{:>6.2}%", row_group_error(&w, rows, cols, *g) * 100.0))
                .collect();
            println!("{name:<44} {}", errors.join(" "));
        }
    }

    /// Relative reconstruction error of `w` when its `rows` are split into
    /// `groups` contiguous groups, each quantized to `uint8` at its own
    /// scale — exactly what one operator per group would hold.
    fn row_group_error(w: &[f32], rows: usize, cols: usize, groups: usize) -> f32 {
        let per_group = rows.div_ceil(groups);
        let mut squared_error = 0.0f64;
        let mut squared = 0.0f64;
        for group in 0..groups {
            let lo_row = group * per_group;
            let hi_row = ((group + 1) * per_group).min(rows);
            if lo_row >= hi_row {
                break;
            }
            let slice = &w[lo_row * cols..hi_row * cols];
            let (lo, hi) = slice
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            let scale = (hi - lo) / 255.0;
            for v in slice {
                let reconstructed = if scale > 0.0 {
                    lo + ((v - lo) / scale).round().clamp(0.0, 255.0) * scale
                } else {
                    *v
                };
                let d = (v - reconstructed) as f64;
                squared_error += d * d;
                squared += (*v as f64) * (*v as f64);
            }
        }
        (squared_error / squared.max(f64::MIN_POSITIVE)).sqrt() as f32
    }

    fn collect_gguf(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_gguf(&path, out);
            } else if path.extension().is_some_and(|e| e == "gguf") {
                out.push(path);
            }
        }
    }

    /// A block is only a block when all three projections are present.
    #[test]
    fn only_complete_gated_triples_are_blocks() {
        let gguf = model(vec![
            tensor("v.blk.0.ffn_gate.weight", vec![768, 3072]),
            tensor("v.blk.0.ffn_up.weight", vec![768, 3072]),
            tensor("v.blk.0.ffn_down.weight", vec![3072, 768]),
            // Missing `ffn_down`, so not a gated FFN this can compile.
            tensor("v.blk.1.ffn_gate.weight", vec![768, 3072]),
            tensor("v.blk.1.ffn_up.weight", vec![768, 3072]),
            // Not a feed-forward tensor at all.
            tensor("v.blk.0.attn_q.weight", vec![768, 768]),
        ]);

        let blocks = discover_blocks(&gguf, Path::new("/nonexistent"), None);
        assert_eq!(blocks.len(), 1, "only v.blk.0 is complete");
        assert_eq!(blocks[0].prefix, "v.blk.0");
        assert_eq!(blocks[0].d_model, 768);
        assert_eq!(blocks[0].d_ff, 3072);
        // No file to read a range out of, so none is claimed.
        assert_eq!(blocks[0].range, None);
    }

    /// A mixture-of-experts layer's shared expert is a gated feed-forward
    /// block like any other, and is found as one.
    ///
    /// The model shape here is `bailingmoe3`'s and not invented: a leading
    /// dense block, then layers whose feed-forward is routed experts plus
    /// one always-on shared expert. Only the shared expert is compilable —
    /// the routed ones are selected per token, and a graph has one static
    /// shape — so the routed tensors must be passed over rather than
    /// mistaken for a block.
    #[test]
    fn a_shared_expert_is_discovered_and_the_routed_experts_are_not() {
        let mut tensors = vec![
            // The leading dense layer.
            tensor("blk.0.ffn_gate.weight", vec![512, 2048]),
            tensor("blk.0.ffn_up.weight", vec![512, 2048]),
            tensor("blk.0.ffn_down.weight", vec![2048, 512]),
        ];
        for layer in 1..3 {
            let p = format!("blk.{layer}");
            // Routed: one tensor per role holding every expert.
            tensors.push(tensor(&format!("{p}.ffn_gate_exps.weight"), vec![512, 256]));
            tensors.push(tensor(&format!("{p}.ffn_up_exps.weight"), vec![512, 256]));
            tensors.push(tensor(&format!("{p}.ffn_down_exps.weight"), vec![256, 512]));
            tensors.push(tensor(&format!("{p}.ffn_gate_inp.weight"), vec![512, 128]));
            // Shared: one expert, every token.
            tensors.push(tensor(
                &format!("{p}.ffn_gate_shexp.weight"),
                vec![512, 512],
            ));
            tensors.push(tensor(&format!("{p}.ffn_up_shexp.weight"), vec![512, 512]));
            tensors.push(tensor(
                &format!("{p}.ffn_down_shexp.weight"),
                vec![512, 512],
            ));
        }
        let blocks = discover_blocks(&model(tensors), Path::new("/nonexistent"), None);

        let names: Vec<String> = blocks.iter().map(|b| b.cache_name()).collect();
        assert_eq!(
            names,
            vec!["blk.0", "blk.1#shexp", "blk.2#shexp"],
            "the dense block and each shared expert, and nothing routed"
        );
        // The shared expert's own shape, not the routed experts' beside it.
        let shared = blocks.iter().find(|b| b.prefix == "blk.1").unwrap();
        assert_eq!((shared.d_model, shared.d_ff), (512, 512));
        assert_eq!(shared.tensor(FfnRole::Down), "blk.1.ffn_down_shexp.weight");
        // The tag is what keeps a dense block and a shared expert at the
        // same layer from sharing one cached artifact.
        assert_ne!(blocks[0].cache_name(), "blk.0#shexp");
    }

    /// `phi3`'s fused gate/up is one block of half the tensor's width.
    #[test]
    fn a_fused_gate_up_block_is_discovered_at_half_the_width() {
        let blocks = discover_blocks(
            &model(vec![
                // `[n_embd, 2 * n_ff]`, and no `ffn_gate` anywhere.
                tensor("blk.0.ffn_up.weight", vec![512, 2048]),
                tensor("blk.0.ffn_down.weight", vec![1024, 512]),
            ]),
            Path::new("/nonexistent"),
            None,
        );
        assert_eq!(blocks.len(), 1, "one block, not one per matching suffix");
        assert_eq!(blocks[0].d_model, 512);
        assert_eq!(blocks[0].d_ff, 1024, "half of the fused width");
        assert!(blocks[0].names.fused_gate_up);
    }

    /// A model with a real `ffn_gate` is dense, even though its `ffn_up`
    /// matches the fused shape's suffix too.
    #[test]
    fn a_dense_block_is_not_also_read_as_fused() {
        let blocks = discover_blocks(
            &model(vec![
                tensor("blk.0.ffn_gate.weight", vec![512, 1024]),
                tensor("blk.0.ffn_up.weight", vec![512, 1024]),
                tensor("blk.0.ffn_down.weight", vec![1024, 512]),
            ]),
            Path::new("/nonexistent"),
            None,
        );
        assert_eq!(blocks.len(), 1);
        assert!(!blocks[0].names.fused_gate_up);
        assert_eq!(blocks[0].d_ff, 1024, "not halved");
    }

    /// An odd fused width cannot be two equal halves, so it is not this
    /// shape and is passed over rather than truncated.
    #[test]
    fn an_odd_fused_width_is_not_a_block() {
        let blocks = discover_blocks(
            &model(vec![
                tensor("blk.0.ffn_up.weight", vec![512, 1025]),
                tensor("blk.0.ffn_down.weight", vec![512, 512]),
            ]),
            Path::new("/nonexistent"),
            None,
        );
        assert!(blocks.is_empty());
    }

    /// The filter selects by substring, which is how a caller asks for one
    /// tower out of a projector that has two.
    #[test]
    fn the_filter_selects_a_subset_of_blocks() {
        let mut tensors = Vec::new();
        for prefix in ["v.blk.0", "v.blk.1", "a.blk.0"] {
            tensors.push(tensor(&format!("{prefix}.ffn_gate.weight"), vec![64, 128]));
            tensors.push(tensor(&format!("{prefix}.ffn_up.weight"), vec![64, 128]));
            tensors.push(tensor(&format!("{prefix}.ffn_down.weight"), vec![128, 64]));
        }
        let gguf = model(tensors);
        let path = Path::new("/nonexistent");

        assert_eq!(discover_blocks(&gguf, path, None).len(), 3);
        assert_eq!(discover_blocks(&gguf, path, Some("v.blk")).len(), 2);
        assert_eq!(discover_blocks(&gguf, path, Some("a.blk")).len(), 1);
        assert_eq!(discover_blocks(&gguf, path, Some("nothing")).len(), 0);
    }

    /// Calibration covers the range it is given and nothing outside it.
    #[test]
    fn synthetic_calibration_stays_inside_its_range() {
        let values = synthetic_calibration(8, 4, (-2.5, 1.5));
        assert_eq!(values.len(), 32);
        assert!(values.iter().all(|v| (-2.5..=1.5).contains(v)));
        // And it varies — a constant would calibrate every range to a point.
        let first = values[0];
        assert!(values.iter().any(|v| *v != first));
    }
}

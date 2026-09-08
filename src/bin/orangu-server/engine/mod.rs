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

pub mod arch;
pub mod attention;
pub mod backend;
pub mod chat_template;
pub mod constraint;
pub mod decode_stages;
pub mod dense_residency;
pub mod env;
pub mod expert_read;
pub mod expert_store;
pub mod expert_tier;
pub mod footprint;
pub mod generate;
pub mod iq_grids;
pub mod kv_cache;
pub mod kv_pool;
pub mod loader;
pub mod metrics;
pub mod moe_stats;
pub mod page_cache;
pub mod placement;
pub mod plan;
pub mod prefix_cache;
pub mod prefix_index;
pub mod quant;
pub mod route_ahead;
pub mod sampling;
pub mod scheduler;
pub mod slot_store;
pub mod tensor;
pub mod tokenizer;
pub mod tool_calls;
pub mod vecdot;

/// The NPU feed-forward service, if this machine has one holding blocks for
/// the model being served.
///
/// A thin re-export so an architecture's forward pass reaches it the way it
/// reaches everything else in `engine`, rather than naming the crate root
/// mid-loop.
pub fn npu_ffn_service() -> Option<&'static orangu::npu_ffn::NpuFfn> {
    orangu::npu_ffn::service()
}

/// Where [`dump_ffn_input`] writes, or `None` when nothing asked it to.
///
/// A `RwLock` rather than a `OnceLock` because capturing *stops*: it is on
/// for as long as it takes one prompt to reach every layer, and every
/// forward pass after that should take the fused paths again. Read on the
/// FFN path of every layer of every chunk, so the uncontended read matters
/// and the writes are two per process.
pub fn dump_ffn_dir() -> Option<std::path::PathBuf> {
    static ENV: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let from_env = ENV.get_or_init(|| {
        let dir = std::env::var_os("ORANGU_NPU_DUMP_ACT").map(std::path::PathBuf::from)?;
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    });
    if from_env.is_some() {
        return from_env.clone();
    }
    capture_slot().read().ok().and_then(|dir| dir.clone())
}

/// Turns capturing on, writing into `dir`.
///
/// Called by `npu_tool::prepare_in_background` when the device has no
/// calibration for this model yet. The cost is that the fused GPU chains
/// are declined while it is on, because they never form the vector being
/// captured — so the prompt that pays for the calibration is slower than
/// the ones after it.
pub fn capture_ffn_inputs_into(dir: std::path::PathBuf) {
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(mut slot) = capture_slot().write() {
        *slot = Some(dir);
    }
}

/// Turns capturing off again, once a whole prompt has been through.
pub fn stop_capturing_ffn_inputs() {
    if let Ok(mut slot) = capture_slot().write() {
        *slot = None;
    }
}

/// The one slot both of the above and [`dump_ffn_dir`] share.
fn capture_slot() -> &'static std::sync::RwLock<Option<std::path::PathBuf>> {
    static DIR: std::sync::RwLock<Option<std::path::PathBuf>> = std::sync::RwLock::new(None);
    &DIR
}

/// Writes one layer's feed-forward input to `ORANGU_NPU_DUMP_ACT`, once per
/// layer, when that variable names a directory.
///
/// **What the NPU's calibration is missing.** The device quantizes
/// activations to `uint8` at a scale fixed when the block is compiled, from
/// calibration data — and `npu_tool::realistic_calibration` synthesizes that
/// data from a Gaussian shaped by `ffn_norm.weight`, because nothing else
/// was available without running the model. A real normalized hidden state
/// is not Gaussian across channels: transformers carry a few persistent
/// channels far outside the bulk, and a value past the calibrated range does
/// not degrade, it *saturates*. This is how the real thing gets captured so
/// the two can be compared, and so a block can be calibrated on it.
///
/// The fused GPU chains compute the network without ever forming this
/// vector on the host, so a caller that is dumping declines them — see
/// `LlamaModel::run_layers`. That makes the dumping run slower than the run
/// it is characterizing, which does not matter: the values are the same
/// ones either path would feed the network.
///
/// Off unless asked for, one file per layer (`blk.<layer>.<tokens>.f32`,
/// raw little-endian `f32`, `tokens * d_model` of them), and it writes the
/// first batch it sees for a layer and no more — a prefill of 45 chunks
/// would otherwise write 45 times per layer for no extra information.
/// How many forward passes this process has run, across every
/// architecture.
///
/// Always on, unlike `decode_stages`' own counter, because it answers a
/// question the NPU's calibration has to ask before it can wait sensibly:
/// *has this model run at all yet?* An empty capture means one of two
/// completely different things — an idle server that has not been asked
/// anything, or an architecture whose forward pass never offers its
/// feed-forward input to [`dump_ffn_input`] — and waiting ten minutes to
/// tell an operator the wrong one of those is what it used to do.
static FORWARD_PASSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Counts one forward pass. A relaxed increment on a path that is already
/// about to run a whole model.
pub(crate) fn note_forward_pass() {
    FORWARD_PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// See [`FORWARD_PASSES`].
pub fn forward_passes() -> u64 {
    FORWARD_PASSES.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn dump_ffn_input(layer: usize, tokens: usize, x: &[f32]) {
    dump_input("blk", layer, tokens, x);
}

/// The layer index for a feed-forward that is **not one of the model's
/// layers** — the multi-token-prediction draft head's.
///
/// `il` exists so a feed-forward can find its compiled NPU block and file its
/// activations for calibration. The draft head has neither: it is a separate
/// head trained to guess the next token, not layer *n* of the network, so it
/// has no compiled block and its activations are not any layer's.
///
/// Passing a real layer's index would be worse than useless — the head's
/// activations would be filed as that layer's and the compiler would
/// calibrate a block on numbers the layer never sees, which is exactly the
/// mistake `captured_calibration` exists to prevent. Passing an out-of-range
/// index without saying so would be worse still: [`dump_input`] names files
/// `blk.<layer>.…` and `capture_complete` counts them against the model's
/// layer count with `>=`, so one stray file would report a partial capture as
/// finished.
///
/// So it is named, and [`dump_input`] declines it.
pub const NOT_A_MODEL_LAYER: usize = usize::MAX;

/// The same, for the input the attention projections read — `attn_norm(x)`
/// rather than `ffn_norm(x)`.
///
/// A separate file per layer, under its own prefix, because it is a
/// different distribution: `Q`/`K`/`V` see the residual stream before
/// attention has touched it, and calibrating them on the feed-forward's
/// input would repeat the mistake `captured_calibration` was written to
/// avoid — building a quantizer around numbers the layer does not see.
pub fn dump_attn_input(layer: usize, tokens: usize, x: &[f32]) {
    dump_input("attn", layer, tokens, x);
}

fn dump_input(prefix: &str, layer: usize, tokens: usize, x: &[f32]) {
    use std::io::Write;
    // Not a layer, so not a layer's calibration — see [`NOT_A_MODEL_LAYER`].
    if layer == NOT_A_MODEL_LAYER {
        return;
    }
    let Some(dir) = dump_ffn_dir() else {
        return;
    };
    let path = dir.join(format!("{prefix}.{layer}.{tokens}.f32"));
    // **Every chunk, not the first.** A static quantizer's range comes from
    // the extremes of its calibration data, so one chunk of one prompt sets
    // the range from whatever that chunk happened to contain and every
    // wider value later in the prefill saturates against it. Appending lets
    // the range be taken over the whole prompt — `npu_tool`'s reader keeps
    // the widest rows and throws the rest away.
    if std::fs::metadata(&path).is_ok_and(|m| m.len() >= CAPTURE_LIMIT) {
        return;
    }
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let mut out = std::io::BufWriter::new(file);
    for v in x {
        if out.write_all(&v.to_le_bytes()).is_err() {
            return;
        }
    }
}

/// How much of one layer's input [`dump_ffn_input`] keeps, in bytes.
///
/// The extra rows exist so the widest ones can be *chosen* — a graph is
/// compiled for sixteen, and which sixteen decides the range the activation
/// quantizer is built around. Coverage is worth real disk here: calibrating
/// gemma 4 E2B on the first chunk of a prompt measured 38.1% against 22.3%
/// for the whole of it.
///
/// 8 MiB is about 780 rows of a 2560-wide model, or 550 of a 3840-wide one
/// — a 720-token prompt end to end either way, and a bound on a prompt of
/// any length. Per layer, so a 35-layer model's capture is around 280 MiB,
/// beside block artifacts that are already several gigabytes. It is kept
/// rather than deleted after compiling because the quality gate re-measures
/// against it on every later start, and a gate with no calibration measures
/// the synthetic distribution and refuses the model.
const CAPTURE_LIMIT: u64 = 8 << 20;

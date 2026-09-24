//! Where a multi-token pass runs when decode is on a device —
//! `[orangu-server].prefill_backend`: `auto` (the default: whichever does a
//! prompt-shaped GEMM faster, measured once at load, so each machine
//! decides for itself), `device` (with decode), or `cpu` (the CPU
//! backend, while single-token decode stays on the device).
//! `ORANGU_HYBRID_PREFILL_CPU=1` is `cpu` from the environment.
//! Set by `main` before the model is built; read by the architectures
//! that can split a pass that way when they load
//! (`arch::gemma::GemmaModel`, `arch::qwen_hybrid::Trunk`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use super::backend::{Backend, CpuBackend};
use super::loader::QuantMatrix;

/// The configured choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefillBackend {
    /// Prompts run where decode runs.
    Device,
    /// Prompts run on the CPU backend.
    Cpu,
    /// Measured at load: one prompt-shaped GEMM on each backend, the
    /// faster one takes the prompts. The default.
    #[default]
    Auto,
}

static CHOICE: AtomicU8 = AtomicU8::new(PrefillBackend::Auto as u8);

pub fn set(choice: PrefillBackend) {
    CHOICE.store(choice as u8, Ordering::Relaxed);
}

pub fn choice() -> PrefillBackend {
    if super::env::flag_on("ORANGU_HYBRID_PREFILL_CPU") {
        return PrefillBackend::Cpu;
    }
    match CHOICE.load(Ordering::Relaxed) {
        1 => PrefillBackend::Cpu,
        2 => PrefillBackend::Auto,
        _ => PrefillBackend::Device,
    }
}

/// Where prompts ended up once the model was built: `0` not yet known,
/// `1` on the CPU backend, `2` on a device. Read by the NPU's install
/// (`npu_tool::install_ffn_service`), which decides by the same kind of
/// measurement whether its feed-forward blocks beat what prompts would
/// otherwise run on.
static PROMPTS: AtomicU8 = AtomicU8::new(0);

/// Records where prompts run. The first answer stands: the architecture
/// that measured decides, `main`'s [`settle`] only fills in for one that
/// did not.
fn record(on_cpu: bool) {
    let _ = PROMPTS.compare_exchange(
        0,
        if on_cpu { 1 } else { 2 },
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
}

/// Called by `main` once the model is built: prompts run on `backend`
/// unless the architecture already said otherwise. Also the signal that
/// the CPU worker pool is configured, so a measurement on the CPU backend
/// no longer races its `build_global`.
pub fn settle(backend: &dyn Backend) {
    record(backend.is_cpu());
}

/// Set by `main` once the CPU prompt path is final — the model built, where
/// prompts run settled, and `engine::prompt_weights` prepared. What the
/// NPU's install waits for before it times the CPU against the NPU, so it
/// measures the CPU a prompt will actually run on.
static CPU_PATH_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// See [`CPU_PATH_READY`].
pub fn mark_cpu_path_ready() {
    CPU_PATH_READY.store(true, Ordering::Release);
}

/// See [`CPU_PATH_READY`].
pub fn cpu_path_ready() -> bool {
    CPU_PATH_READY.load(Ordering::Acquire)
}

/// Decode moved to the CPU backend after the model's prompts were placed
/// on the device (`engine::decode_backend`): prompts go with it, whatever
/// the device model recorded first.
pub fn force_prompts_on_cpu() {
    PROMPTS.store(1, Ordering::Relaxed);
}

/// Whether prompts run on the CPU backend — `None` until the model is built.
pub fn prompts_on_cpu() -> Option<bool> {
    match PROMPTS.load(Ordering::Relaxed) {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }
}

/// The tokens the probe's GEMM carries: a short prompt, past the point
/// where a device switches to its batched kernels.
const PROBE_TOKENS: usize = 64;

/// The CPU backend when prompts should run there rather than on
/// `device`, by the configured choice — for `auto`, by timing `w` (a
/// projection of the model, the FFN gate's shape is the right one) at
/// [`PROBE_TOKENS`] tokens on each. `None` when `device` *is* the CPU
/// backend, or when prompts stay with it.
pub fn cpu_for_prompts(device: &Arc<dyn Backend>, w: &QuantMatrix) -> Option<Arc<dyn Backend>> {
    let chosen = cpu_for_prompts_unrecorded(device, w);
    record(chosen.is_some() || device.is_cpu());
    chosen
}

/// [`cpu_for_prompts`] deciding `auto` by the model's own work rather than
/// one GEMM: `time_pass(prompts)` runs a prompt-shaped pass of the model
/// with its prompts on `prompts` (`None`: the device) and returns its
/// seconds; `pass` says what it runs, for the log. One GEMM put the cores
/// 1.5–3× ahead on the CIX P1 where a whole prompt is 4.5×, and read the
/// device anywhere from 4.7 to 39.8 ms as its clock ramped — right by a
/// wide margin there, not a measurement to trust where the two are close
/// (`doc/PERF-ALL.md`, task 8). The GEMM stays as the fallback for a pass
/// that cannot be timed.
pub fn cpu_for_prompts_by(
    device: &Arc<dyn Backend>,
    w: &QuantMatrix,
    pass: &str,
    mut time_pass: impl FnMut(Option<Arc<dyn Backend>>) -> Option<f64>,
) -> Option<Arc<dyn Backend>> {
    if device.as_wgpu().is_none() || choice() != PrefillBackend::Auto {
        return cpu_for_prompts(device, w);
    }
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend);
    let (Some(on_device), Some(on_cpu)) = (time_pass(None), time_pass(Some(cpu.clone()))) else {
        return cpu_for_prompts(device, w);
    };
    let winner = if on_cpu < on_device {
        "the cpu"
    } else {
        "the device"
    };
    crate::engine::adapt::note(
        "prompts",
        format!("on {winner}"),
        format!(
            "{pass} takes {:.1} ms on the device, {:.1} ms on the cpu",
            on_device * 1e3,
            on_cpu * 1e3,
        ),
    );
    let chosen = (on_cpu < on_device).then_some(cpu);
    record(chosen.is_some());
    chosen
}

fn cpu_for_prompts_unrecorded(
    device: &Arc<dyn Backend>,
    w: &QuantMatrix,
) -> Option<Arc<dyn Backend>> {
    device.as_wgpu()?;
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend);
    match choice() {
        PrefillBackend::Device => None,
        PrefillBackend::Cpu => Some(cpu),
        PrefillBackend::Auto => {
            let on_device = time_gemm(device.as_ref(), w);
            let on_cpu = time_gemm(cpu.as_ref(), w);
            let winner = if on_cpu < on_device {
                "the cpu"
            } else {
                "the device"
            };
            crate::engine::adapt::note(
                "prompts",
                format!("on {winner}"),
                format!(
                    "a {PROBE_TOKENS}-token {}x{} GEMM takes {:.1} ms on the device, {:.1} ms on the cpu",
                    w.in_dim,
                    w.out_dim,
                    on_device * 1e3,
                    on_cpu * 1e3,
                ),
            );
            (on_cpu < on_device).then_some(cpu)
        }
    }
}

/// Warm-up calls before the timed ones: a device's first call builds its
/// pipelines, and an integrated GPU's governor needs a few calls to lift
/// its clock — on the CIX P1's Mali one warm-up measured the same GEMM at
/// 6.5 ms in one start and 39.8 ms in the next.
const PROBE_WARMUPS: usize = 3;

/// Timed calls, of which the best is kept.
const PROBE_RUNS: usize = 5;

/// Seconds for one `matmul` of `w` at [`PROBE_TOKENS`] tokens on
/// `backend`: the best of [`PROBE_RUNS`] after [`PROBE_WARMUPS`].
fn time_gemm(backend: &dyn Backend, w: &QuantMatrix) -> f64 {
    let x: Vec<f32> = (0..w.in_dim * PROBE_TOKENS)
        .map(|i| ((i * 7919) % 257) as f32 / 257.0 - 0.5)
        .collect();
    for _ in 0..PROBE_WARMUPS {
        let _ = backend.matmul(&x, PROBE_TOKENS, w);
    }
    (0..PROBE_RUNS)
        .map(|_| {
            let t = Instant::now();
            let _ = backend.matmul(&x, PROBE_TOKENS, w);
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

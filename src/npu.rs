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

//! The machine's NPU, if it has one: what is present ([`detect_npu`], in
//! the same spirit as [`crate::hardware`]'s CPU and GPU probes) and how to
//! run a compiled graph on it ([`NpuRuntime`]).
//!
//! Today that means one family, reached one way: an Arm China Zhouyi AIPU
//! through CIX's NOE user-mode driver (`libnoe`). That is the stack shipped
//! on CIX P1/CD8180-class boards, where the kernel side is `aipu.ko` behind
//! `/dev/aipu`. Other vendors' NPUs (Rockchip's RKNN, Intel's, Qualcomm's
//! Hexagon) have entirely separate userspace stacks and are simply not
//! detected — they would each need their own probe here, and a machine
//! carrying one reports no NPU rather than a wrong one.
//!
//! # Why `dlopen` rather than linking
//!
//! `libnoe` ships with a vendor BSP, lives outside the default loader path
//! (`/usr/share/cix/lib`), and exists on approximately no other machine. A
//! build that linked it would not run anywhere else, and orangu is built
//! once and run on whatever hardware is in front of it. So the library is
//! opened at *runtime* and its absence is an ordinary "no NPU" answer, not
//! an error — the same contract `engine::backend`'s CUDA and OpenCL
//! backends have with their own vendor libraries, and for the same reason.
//!
//! # Two halves: inventory, and execution
//!
//! [`detect_npu`] answers "what is present". [`NpuRuntime`] runs work on it:
//! `noe_load_graph` → `noe_create_job` → `noe_load_tensor` →
//! `noe_job_infer_sync` → `noe_get_tensor`. Both are verified against real
//! hardware — a 968 MiB Stable Diffusion UNet executes through this module
//! at ~2.2 s/inference, an int8 face-embedding model at ~6.5 ms.
//!
//! # What this is *not* yet
//!
//! Worth stating plainly, because a working NPU runtime invites the
//! assumption that orangu can serve a GGUF model on it. It cannot, for two
//! reasons that are about the engine rather than this module.
//!
//! First, NOE executes graphs **compiled ahead of time** into `.cix` by the
//! Zhouyi toolchain. It has no per-operation entry point, so
//! `engine::backend::Backend`'s `matmul` — which takes one `QuantMatrix`
//! per call — has nothing to bind to. Feeding a transformer to this needs a
//! graph-level seam in the engine first.
//!
//! Second, the hardware is an integer engine. Every tensor a graph declares
//! carries a single per-tensor `scale`/`zero_point` (see [`NpuTensor`]),
//! where GGUF's K-quants carry per-block scales over 32-256 elements, and
//! the Zhouyi compiler refuses a graph whose tensors have no calibrated
//! quantization at all. NPU-resident weights would have to be requantized.
//!
//! So this module is the runtime half, complete and usable on its own for
//! any already-compiled graph. To *build* one, see [`crate::npu_ort`],
//! which compiles a linear layer through ONNX Runtime's Zhouyi provider —
//! but note the two cannot be used in the same process: NOE and the
//! provider's stack both drive the device and conflict once either has a
//! context open.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::PathBuf;

use libloading::Library;

/// `NOE_STATUS_SUCCESS` from `cix_noe_standard_api.h`. Every entry point
/// here returns this enum, and every non-zero value is a failure we treat
/// the same way: report no NPU.
const NOE_STATUS_SUCCESS: c_int = 0;

/// `job_status_t` from `cix_noe_standard_api.h`: dispatched but not
/// finished, and finished successfully. The other two values it defines —
/// exception and coredump — are failures, and [`NpuGraph::finish`] reports
/// anything that is not `DONE` the same way.
const JOB_STATUS_NO_STATUS: c_int = 0;
const JOB_STATUS_DONE: c_int = 1;

/// How much room `noe_get_target` is given to write the architecture string.
///
/// The API takes a bare `char*` and documents no maximum, which is a buffer
/// the caller has to size by judgement. Real hardware answers `X2_1204MP3`
/// — ten bytes and a NUL — so 256 is two orders of magnitude of headroom
/// while staying a stack allocation. The result is still parsed defensively
/// below: a reply with no NUL inside the buffer is discarded rather than
/// read past.
const TARGET_BUF_LEN: usize = 256;

/// Where to look for the NOE user-mode driver, in order.
///
/// The bare sonames come first so a machine that has installed the BSP
/// properly (an `ld.so.conf.d` entry, or `LD_LIBRARY_PATH`) is found
/// through the normal loader search. The absolute paths are the fallback
/// for the stock CIX layout, which installs under `/usr/share/cix/lib` and
/// does *not* add it to the loader path — so the soname alone finds nothing
/// on an otherwise perfectly working board.
const LIBRARY_CANDIDATES: &[&str] = &[
    "libnoe.so.0",
    "libnoe.so",
    "/usr/share/cix/lib/libnoe.so.0",
    "/usr/share/cix/lib/libnoe.so",
];

/// Overrides [`LIBRARY_CANDIDATES`] with a single explicit path, for a BSP
/// installed somewhere this doesn't guess.
const LIBRARY_ENV: &str = "ORANGU_NPU_LIB";

/// One NPU, as the vendor runtime describes it.
///
/// The counts are a hierarchy, not three independent numbers: a device has
/// partitions, each partition has clusters, each cluster has cores.
/// [`Self::clusters`] and [`Self::cores`] are the totals across that whole
/// tree, since a machine with one partition — every one seen so far — makes
/// the distinction invisible anyway, and the total is the figure a reader
/// is actually after.
pub struct NpuInfo {
    /// Who makes the accelerator, as a display string. Fixed per probe
    /// rather than queried: NOE drives Zhouyi parts and nothing else, so
    /// reaching this struct through `libnoe` *is* the vendor answer.
    pub vendor: String,
    /// The hardware architecture string straight from `noe_get_target`,
    /// e.g. `X2_1204MP3` — Zhouyi X2, configuration 1204, three cores.
    pub target: String,
    pub partitions: u32,
    pub clusters: u32,
    pub cores: u32,
    /// Which library file answered, so a report names the stack it actually
    /// talked to rather than leaving the reader to guess between a BSP
    /// install and one on the loader path.
    pub runtime: PathBuf,
}

/// The NOE entry points this probe needs, resolved out of an open library.
///
/// `noe_status_t` is a C enum, so it comes back as [`c_int`]; the context is
/// an opaque handle the API only ever passes back to itself, so it stays a
/// raw pointer rather than being given a made-up Rust shape.
///
/// Deliberately *not* the whole API. Binding only what is called keeps the
/// unsafe surface to the five functions that are actually exercised, and
/// every one of these was checked against the exported symbols of a real
/// `libnoe.so.0.6.0` — the header also declares entry points (such as
/// `noe_get_device_info`) that the shipped library does not export, and
/// resolving one of those would fail the whole probe for no reason.
struct Noe {
    init_context: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    deinit_context: unsafe extern "C" fn(*const c_void) -> c_int,
    get_target: unsafe extern "C" fn(*const c_void, *mut c_char) -> c_int,
    get_partition_count: unsafe extern "C" fn(*const c_void, *mut c_uint) -> c_int,
    get_cluster_count: unsafe extern "C" fn(*const c_void, c_uint, *mut c_uint) -> c_int,
    get_core_count: unsafe extern "C" fn(*const c_void, c_uint, c_uint, *mut c_uint) -> c_int,
    load_graph: unsafe extern "C" fn(*const c_void, *const c_char, *mut u64, *mut c_void) -> c_int,
    load_graph_helper: unsafe extern "C" fn(
        *const c_void,
        *const c_char,
        c_uint,
        *mut u64,
        *mut GraphConfig,
    ) -> c_int,
    unload_graph: unsafe extern "C" fn(*const c_void, u64) -> c_int,
    get_tensor_count: unsafe extern "C" fn(*const c_void, u64, c_int, *mut c_uint) -> c_int,
    get_tensor_descriptor:
        unsafe extern "C" fn(*const c_void, u64, c_int, c_uint, *mut NpuTensor) -> c_int,
    create_job: unsafe extern "C" fn(*const c_void, u64, *mut u64, *mut JobConfig) -> c_int,
    load_tensor: unsafe extern "C" fn(*const c_void, u64, c_uint, *const c_void) -> c_int,
    infer_sync: unsafe extern "C" fn(*const c_void, u64, i32) -> c_int,
    /// Dispatches a job and returns immediately. The third argument is an
    /// optional completion callback, which this never uses — the caller
    /// waits with [`Noe::get_job_status`] instead, because it has other
    /// work to start first and a callback would only move the wait.
    ///
    /// **Optional**, unlike everything else here: this pair is what lets two
    /// independent graphs overlap, and a driver too old to export it should
    /// still run graphs one at a time rather than report no NPU at all.
    /// [`NpuGraph::start`] falls back to `infer_sync` when it is missing.
    infer_async: Option<unsafe extern "C" fn(*const c_void, u64, *const c_void) -> c_int>,
    /// Waits up to `timeout` milliseconds for a dispatched job, writing one
    /// of the `JOB_STATUS_*` values. Optional with [`Noe::infer_async`], and
    /// only ever used together with it.
    get_job_status: Option<unsafe extern "C" fn(*const c_void, u64, *mut c_int, i32) -> c_int>,
    get_tensor: unsafe extern "C" fn(*const c_void, u64, c_int, c_uint, *mut c_void) -> c_int,
    clean_job: unsafe extern "C" fn(*const c_void, u64) -> c_int,
}

impl Noe {
    /// Resolves every entry point, or `None` if any one is missing — which
    /// is the right answer for a file that happens to be named `libnoe` but
    /// is not this API.
    fn load(library: &Library) -> Option<Self> {
        // SAFETY: each symbol's type here matches its declaration in
        // `cix_noe_standard_api.h`, and the header declares them `extern
        // "C"` (the shipped library exports them unmangled, which is what
        // makes them resolvable by these plain names at all).
        unsafe {
            Some(Self {
                init_context: *library.get(b"noe_init_context\0").ok()?,
                deinit_context: *library.get(b"noe_deinit_context\0").ok()?,
                get_target: *library.get(b"noe_get_target\0").ok()?,
                get_partition_count: *library.get(b"noe_get_partition_count\0").ok()?,
                get_cluster_count: *library.get(b"noe_get_cluster_count\0").ok()?,
                get_core_count: *library.get(b"noe_get_core_count\0").ok()?,
                load_graph: *library.get(b"noe_load_graph\0").ok()?,
                load_graph_helper: *library.get(b"noe_load_graph_helper\0").ok()?,
                unload_graph: *library.get(b"noe_unload_graph\0").ok()?,
                get_tensor_count: *library.get(b"noe_get_tensor_count\0").ok()?,
                get_tensor_descriptor: *library.get(b"noe_get_tensor_descriptor\0").ok()?,
                create_job: *library.get(b"noe_create_job\0").ok()?,
                load_tensor: *library.get(b"noe_load_tensor\0").ok()?,
                infer_sync: *library.get(b"noe_job_infer_sync\0").ok()?,
                // `.ok().map(|f| *f)`, not `.ok()?`: a driver without these
                // still drives the device, one job at a time.
                infer_async: library.get(b"noe_job_infer_async\0").ok().map(|f| *f),
                get_job_status: library.get(b"noe_get_job_status\0").ok().map(|f| *f),
                get_tensor: *library.get(b"noe_get_tensor\0").ok()?,
                clean_job: *library.get(b"noe_clean_job\0").ok()?,
            })
        }
    }
}

/// The machine's NPU, or `None` when it has none this knows how to see.
///
/// Never an error and never a panic: a missing library, a library that
/// isn't NOE, a driver that refuses to initialize, and a board with no NPU
/// at all are one answer here, because they are one answer to the question
/// the caller asked. That matches [`crate::hardware::detect_gpus`], where a
/// card no source recognizes simply doesn't appear.
///
/// The context is opened and closed within this call. Holding one would
/// keep a claim on the device for the lifetime of a process that, today,
/// has nothing to run on it.
pub fn detect_npu() -> Option<NpuInfo> {
    for candidate in candidates() {
        // SAFETY: `dlopen` runs the library's initializers, which is
        // unavoidable for any runtime-loaded driver and is exactly what the
        // vendor stack expects. The path is either the operator's own
        // (`ORANGU_NPU_LIB`) or one of the fixed sonames above.
        let Ok(library) = (unsafe { Library::new(&candidate) }) else {
            continue;
        };
        let Some(noe) = Noe::load(&library) else {
            continue;
        };
        if let Some(info) = probe(&noe, candidate) {
            return Some(info);
        }
    }
    None
}

/// [`LIBRARY_ENV`] if it is set to something non-empty, otherwise the
/// built-in list. An override replaces the candidates rather than being
/// prepended to them, so pointing at a specific build and having the probe
/// silently succeed against a *different* one already on the path isn't
/// possible.
fn candidates() -> Vec<PathBuf> {
    match std::env::var_os(LIBRARY_ENV) {
        Some(path) if !path.is_empty() => vec![PathBuf::from(path)],
        _ => LIBRARY_CANDIDATES.iter().map(PathBuf::from).collect(),
    }
}

/// Opens a context, asks it what it is, and closes it again.
///
/// Returns `None` the moment anything answers with a failure status: a
/// half-populated inventory entry would be worse than no entry, since the
/// whole point of the report is to be trusted about what is present.
fn probe(noe: &Noe, runtime: PathBuf) -> Option<NpuInfo> {
    let mut ctx: *mut c_void = std::ptr::null_mut();
    // SAFETY: `init_context` writes a single out-pointer, which is what
    // `&mut ctx` provides.
    if unsafe { (noe.init_context)(&mut ctx) } != NOE_STATUS_SUCCESS || ctx.is_null() {
        return None;
    }

    let info = read_info(noe, ctx, runtime);

    // Closed on both paths, including the early returns inside `read_info`
    // — the device is a shared, exclusive-ish resource and leaking a
    // context out of a *reporting* call would be a poor trade for slightly
    // simpler control flow.
    //
    // SAFETY: `ctx` came from a successful `init_context` above and has not
    // been closed yet.
    unsafe { (noe.deinit_context)(ctx) };

    info
}

/// The queries themselves, split out so [`probe`] can close the context on
/// every path with one call rather than one per failure.
fn read_info(noe: &Noe, ctx: *mut c_void, runtime: PathBuf) -> Option<NpuInfo> {
    let mut buf = [0u8; TARGET_BUF_LEN];
    // SAFETY: the API writes a NUL-terminated string into caller memory;
    // `buf` is `TARGET_BUF_LEN` bytes of it. See that constant for why the
    // size is a judgement call and how the result is bounded below.
    if unsafe { (noe.get_target)(ctx, buf.as_mut_ptr().cast::<c_char>()) } != NOE_STATUS_SUCCESS {
        return None;
    }
    // Bounded by the buffer, not by the driver: `from_bytes_until_nul`
    // fails rather than reads on a reply that never terminated.
    let target = CStr::from_bytes_until_nul(&buf).ok()?.to_str().ok()?.trim();
    if target.is_empty() {
        return None;
    }

    // These counts drive the loops below, so a driver answering with
    // nonsense would *spin* this probe rather than fail it — and it runs on
    // every `orangu-server system`. Real parts report one partition of one
    // cluster, so the cap is generous enough that no plausible device is
    // truncated by it and small enough that a wild answer costs a bounded
    // handful of cheap calls instead of a hung report.
    const MAX_UNITS: u32 = 64;

    let partitions = count(|out| {
        // SAFETY: single out-pointer, as above.
        unsafe { (noe.get_partition_count)(ctx, out) }
    })?
    .min(MAX_UNITS);

    // Walk the tree rather than assuming one partition of one cluster:
    // the API models several of each, and summing what it reports costs
    // two cheap calls per partition.
    let mut clusters = 0u32;
    let mut cores = 0u32;
    for partition in 0..partitions {
        let in_partition = count(|out| {
            // SAFETY: single out-pointer, `partition` is in the range the
            // count above reported.
            unsafe { (noe.get_cluster_count)(ctx, partition, out) }
        })?
        .min(MAX_UNITS);
        clusters = clusters.saturating_add(in_partition);
        for cluster in 0..in_partition {
            let in_cluster = count(|out| {
                // SAFETY: single out-pointer; both indices are in the
                // ranges their own counts reported.
                unsafe { (noe.get_core_count)(ctx, partition, cluster, out) }
            })?;
            cores = cores.saturating_add(in_cluster);
        }
    }

    Some(NpuInfo {
        vendor: "Arm China".to_string(),
        target: target.to_string(),
        partitions,
        clusters,
        cores,
        runtime,
    })
}

/// Runs one of NOE's `*_count` queries, turning its status-plus-out-pointer
/// convention into an `Option`.
fn count(query: impl FnOnce(*mut c_uint) -> c_int) -> Option<u32> {
    let mut value: c_uint = 0;
    (query(&mut value) == NOE_STATUS_SUCCESS).then_some(value)
}

/// One tensor a graph declares, exactly `tensor_desc_t` from
/// `cix_noe_standard_api.h`.
///
/// `scale` and `zero_point` are the whole quantization story for a tensor —
/// **one pair for the entire tensor**, not per block. That is the field that
/// decides whether a GGUF model could ever live here: `Q4_K` carries a scale
/// per 32-256 element block, and there is nowhere in this struct to put
/// them. See the module doc.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NpuTensor {
    pub id: u32,
    /// Size **in bytes**, not elements — what [`NpuGraph::run`] expects and
    /// returns for this slot.
    pub size: u32,
    pub scale: f32,
    pub zero_point: i32,
    /// `noe_data_type_t`: 3 = S8, 4 = U16, 5 = S16, 7 = S32, 0xa = F16.
    /// Left as the raw value rather than an enum — this module does not
    /// interpret tensor contents, and inventing names for codes the vendor
    /// may extend would age badly.
    pub data_type: c_int,
}

/// `graph_config_npu_t`, the NPU half of a graph-loading configuration.
///
/// Every field is left at its default. The struct exists because it must be
/// *present*, not because this module configures anything through it.
#[repr(C)]
struct GraphConfigNpu {
    misc: c_uint,
    wt_idxes: *mut i32,
    wt_idxes_cnt: i32,
    extra_weight_path: *const c_char,
}

/// `graph_config_t`, a wrapper around a pointer to the above.
///
/// Always supplied, never null — the same header-versus-implementation
/// divergence [`JobConfig`] documents. `noe_load_graph_helper` gives this
/// parameter a `= nullptr` default and then dereferences it: passing null
/// **segfaults**, verified on real hardware. (`noe_load_graph`, confusingly,
/// tolerates null for the same parameter.)
#[repr(C)]
struct GraphConfig {
    conf_g_npu: *mut GraphConfigNpu,
}

/// `job_config_npu_t`: a bitfield union that is zero by default, then
/// feature-map region hints and an optional dynamic-shape parameter.
///
/// `repr(C)` reproduces the C padding exactly — `misc` at 0, `fm_idxes` at
/// 8, `fm_idxes_cnt` at 16, `dynshape` at 24.
#[repr(C)]
struct JobConfigNpu {
    misc: c_uint,
    fm_idxes: *mut i32,
    fm_idxes_cnt: i32,
    dynshape: *mut c_void,
}

/// `job_config_t`, which is only a wrapper around a pointer to the above.
///
/// Always supplied, never null. The header gives `noe_create_job`'s config
/// parameter a `= nullptr` default, but the shipped library dereferences it
/// regardless: passing null **segfaults**, verified on real hardware. This
/// is exactly the kind of divergence between header and implementation that
/// makes a defaulted argument worth pinning down rather than trusting.
#[repr(C)]
struct JobConfig {
    conf_j_npu: *mut JobConfigNpu,
}

/// An open NOE context: the library, and the device handle it returned.
///
/// Kept behind an [`std::rc::Rc`] by [`NpuGraph`] so a graph can outlive the
/// binding that produced it without the context being torn down under it —
/// `Drop` order between the two would otherwise be the caller's problem.
///
/// `Rc` and not `Arc`, deliberately. This holds raw device handles from a
/// vendor library that documents no thread-safety guarantee, so a runtime
/// and its graphs stay on the thread that created them. `Arc` would compile
/// and would be a lie: it advertises a sharing property nothing here
/// establishes.
struct RuntimeInner {
    /// Held only to keep the library mapped: dropping it would unload the
    /// code every function pointer in `noe` points at.
    _library: Library,
    noe: Noe,
    ctx: *mut c_void,
}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from a successful `init_context` and nothing
        // derived from it survives this point — `NpuGraph` holds an `Arc` to
        // this struct, so no graph or job can still be alive here.
        unsafe { (self.noe.deinit_context)(self.ctx) };
    }
}

/// A usable connection to the NPU, for running compiled graphs.
///
/// Separate from [`detect_npu`] because they answer different questions and
/// cost different amounts: detection opens a context and closes it again,
/// while this one holds the device for as long as the caller needs it.
pub struct NpuRuntime {
    inner: std::rc::Rc<RuntimeInner>,
}

impl NpuRuntime {
    /// Opens the NPU, or `None` when there isn't one — the same "no NPU is
    /// an answer, not an error" contract [`detect_npu`] has.
    pub fn open() -> Option<Self> {
        for candidate in candidates() {
            // SAFETY: see `detect_npu` — `dlopen` of a fixed soname or an
            // operator-supplied path.
            let Ok(library) = (unsafe { Library::new(&candidate) }) else {
                continue;
            };
            let Some(noe) = Noe::load(&library) else {
                continue;
            };
            let mut ctx: *mut c_void = std::ptr::null_mut();
            // SAFETY: single out-pointer.
            if unsafe { (noe.init_context)(&mut ctx) } != NOE_STATUS_SUCCESS || ctx.is_null() {
                continue;
            }
            return Some(Self {
                inner: std::rc::Rc::new(RuntimeInner {
                    _library: library,
                    noe,
                    ctx,
                }),
            });
        }
        None
    }

    /// Loads a `.cix` graph produced by the Zhouyi toolchain.
    ///
    /// The path is passed to the vendor loader as a C string, so a path
    /// containing an interior NUL is rejected here rather than silently
    /// truncated.
    pub fn load_graph(&self, path: &std::path::Path) -> Result<NpuGraph, NpuError> {
        // Checked here rather than left to the vendor loader, which
        // **segfaults** on a path that does not exist instead of returning
        // a status — reproduced on real hardware, and the reason this guard
        // is a correctness requirement and not a courtesy. A caller passing
        // a bad path must get an error, not lose the process.
        if !path.is_file() {
            return Err(NpuError::new(
                &format!("no such graph file: {}", path.display()),
                0,
            ));
        }
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| NpuError::new("graph path contains an interior NUL", 0))?;
        let noe = &self.inner.noe;
        let ctx = self.inner.ctx;

        let mut graph: u64 = 0;
        // SAFETY: `c_path` outlives the call; `graph` is a single out-param.
        let st =
            unsafe { (noe.load_graph)(ctx, c_path.as_ptr(), &mut graph, std::ptr::null_mut()) };
        if st != NOE_STATUS_SUCCESS {
            return Err(NpuError::new("noe_load_graph failed", st));
        }
        self.bind(graph, None)
    }

    /// Loads a graph the caller already holds in memory.
    ///
    /// The bytes are an AIPU executable — the `.text`/`.rodata` image the
    /// Zhouyi compiler emits, which is what [`crate::npu_ort`] produces and
    /// what a `.cix` file wraps. Note the asymmetry: this entry point wants
    /// the *bare* executable and rejects a `.cix` container with
    /// "graph version unsupported", while [`Self::load_graph`] wants the
    /// container and would reject the executable. They are not two spellings
    /// of one loader.
    ///
    /// This is the half of the NPU story that makes compilation worth doing.
    /// A graph compiled once can be loaded here in well under a millisecond
    /// and then run repeatedly against a reused job, which is the only
    /// arrangement on this device that amortizes a compile.
    pub fn load_graph_bytes(&self, binary: &[u8]) -> Result<NpuGraph, NpuError> {
        if binary.is_empty() {
            return Err(NpuError::new("graph binary is empty", 0));
        }
        let noe = &self.inner.noe;
        let ctx = self.inner.ctx;

        let mut inner_cfg = Box::new(GraphConfigNpu {
            misc: 0,
            wt_idxes: std::ptr::null_mut(),
            wt_idxes_cnt: 0,
            extra_weight_path: std::ptr::null(),
        });
        let mut cfg = Box::new(GraphConfig {
            conf_g_npu: &mut *inner_cfg,
        });
        let mut graph: u64 = 0;
        // SAFETY: `binary` outlives the call, its length is passed exactly,
        // and `cfg` is non-null as the loader requires — see `GraphConfig`.
        let st = unsafe {
            (noe.load_graph_helper)(
                ctx,
                binary.as_ptr().cast(),
                binary.len() as c_uint,
                &mut graph,
                &mut *cfg,
            )
        };
        if st != NOE_STATUS_SUCCESS {
            return Err(NpuError::new("noe_load_graph_helper failed", st));
        }
        self.bind(
            graph,
            Some(HeldConfig {
                _cfg: cfg,
                _inner_cfg: inner_cfg,
            }),
        )
    }

    /// Reads a loaded graph's shape and binds a job to it.
    ///
    /// Shared by both loaders: everything after "the driver has a graph id"
    /// is identical no matter where the graph came from.
    fn bind(&self, graph: u64, held: Option<HeldConfig>) -> Result<NpuGraph, NpuError> {
        let noe = &self.inner.noe;
        let ctx = self.inner.ctx;

        // Descriptors first: a caller needs the byte sizes before it can
        // supply inputs, and a graph whose shape can't be read is not one
        // worth handing back.
        let inputs = describe(noe, ctx, graph, TENSOR_TYPE_INPUT);
        let outputs = describe(noe, ctx, graph, TENSOR_TYPE_OUTPUT);

        let mut inner_cfg = Box::new(JobConfigNpu {
            misc: 0,
            fm_idxes: std::ptr::null_mut(),
            fm_idxes_cnt: 0,
            dynshape: std::ptr::null_mut(),
        });
        let mut cfg = Box::new(JobConfig {
            conf_j_npu: &mut *inner_cfg,
        });
        let mut job: u64 = 0;
        // SAFETY: `cfg` and the `JobConfigNpu` it points at both outlive
        // this call — and the job — and must be non-null; see `JobConfig`.
        let st = unsafe { (noe.create_job)(ctx, graph, &mut job, &mut *cfg) };
        if st != NOE_STATUS_SUCCESS {
            // SAFETY: `graph` loaded successfully before this was called.
            unsafe { (noe.unload_graph)(ctx, graph) };
            return Err(NpuError::new("noe_create_job failed", st));
        }

        Ok(NpuGraph {
            inner: std::rc::Rc::clone(&self.inner),
            graph,
            job,
            inputs,
            outputs,
            _job_config: HeldJobConfig {
                _cfg: cfg,
                _inner_cfg: inner_cfg,
            },
            _graph_config: held,
        })
    }
}

/// The graph-loading configuration, kept alive for as long as the graph is.
///
/// The vendor library takes these by pointer and is not documented to copy
/// them, so holding them is the defensible reading of an undocumented ABI —
/// a stack temporary would be a bet that it does copy. Boxed so the
/// addresses stay put as the owner moves.
struct HeldConfig {
    _cfg: Box<GraphConfig>,
    _inner_cfg: Box<GraphConfigNpu>,
}

/// The job configuration, kept alive for the same reason.
struct HeldJobConfig {
    _cfg: Box<JobConfig>,
    _inner_cfg: Box<JobConfigNpu>,
}

/// `NOE_TENSOR_TYPE_INPUT` / `_OUTPUT`.
const TENSOR_TYPE_INPUT: c_int = 0;
const TENSOR_TYPE_OUTPUT: c_int = 1;

/// Reads every tensor descriptor of one type, stopping at the first the
/// driver declines to describe — a partial list would silently mis-size a
/// caller's buffers.
fn describe(noe: &Noe, ctx: *mut c_void, graph: u64, kind: c_int) -> Vec<NpuTensor> {
    let mut n: c_uint = 0;
    // SAFETY: single out-pointer.
    if unsafe { (noe.get_tensor_count)(ctx, graph, kind, &mut n) } != NOE_STATUS_SUCCESS {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut desc = NpuTensor::default();
        // SAFETY: `i` is within the count just reported; `desc` is a single
        // out-param of exactly the layout `tensor_desc_t` declares.
        if unsafe { (noe.get_tensor_descriptor)(ctx, graph, kind, i, &mut desc) }
            != NOE_STATUS_SUCCESS
        {
            break;
        }
        out.push(desc);
    }
    out
}

/// A loaded graph with a job bound to it, ready to run.
///
/// The job is created once and reused across [`Self::run`] calls rather than
/// per inference: job setup is the expensive half, and reuse is what turns a
/// measured 6.5 ms first inference into 6.5 ms *steady state* instead of
/// paying setup every time.
pub struct NpuGraph {
    inner: std::rc::Rc<RuntimeInner>,
    graph: u64,
    job: u64,
    inputs: Vec<NpuTensor>,
    outputs: Vec<NpuTensor>,
    /// Held, not used: see [`HeldJobConfig`].
    _job_config: HeldJobConfig,
    /// Held for the same reason, when the graph came from memory.
    _graph_config: Option<HeldConfig>,
}

impl NpuGraph {
    /// What this graph expects, in order — each entry's `size` is the exact
    /// byte count [`Self::run`] requires for that slot.
    pub fn inputs(&self) -> &[NpuTensor] {
        &self.inputs
    }

    /// What this graph produces, in order.
    pub fn outputs(&self) -> &[NpuTensor] {
        &self.outputs
    }

    /// Runs one inference: loads every input, executes, and returns one
    /// buffer per output.
    ///
    /// Input sizes are checked against the descriptors *before* anything is
    /// handed to the driver. `noe_load_tensor` takes a bare pointer with no
    /// length, so a short slice would have the device read past the end of
    /// caller memory — a wrong size has to be an error here, not a segfault
    /// inside the vendor library.
    ///
    /// `timeout_ms` of zero or less means wait indefinitely, which is the
    /// vendor's own convention for this argument.
    pub fn run(&self, inputs: &[&[u8]], timeout_ms: i32) -> Result<Vec<Vec<u8>>, NpuError> {
        let mut outputs = Vec::new();
        self.run_into(inputs, &mut outputs, timeout_ms)?;
        Ok(outputs)
    }

    /// [`Self::run`], reusing the caller's buffers instead of allocating.
    ///
    /// `outputs` is resized to one buffer per output tensor, each exactly
    /// the descriptor's byte size; whatever capacity it already holds is
    /// kept. For a caller running the same graph in a loop — the only way
    /// this device earns its compile — that turns two allocations per
    /// inference into none.
    pub fn run_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
        timeout_ms: i32,
    ) -> Result<(), NpuError> {
        if inputs.len() != self.inputs.len() {
            return Err(NpuError::new(
                &format!(
                    "graph takes {} input(s), {} supplied",
                    self.inputs.len(),
                    inputs.len()
                ),
                0,
            ));
        }
        let timing = stage_timing();
        let started = timing.then(std::time::Instant::now);
        let _ = (timing, started);
        self.start(inputs, timeout_ms)?;
        // `finish` does the accounting for both callers of the pair.
        self.finish(outputs, timeout_ms)
    }

    /// Loads `inputs` and **dispatches** the job without waiting for it.
    ///
    /// Paired with [`Self::finish`], which is where the waiting happens. The
    /// pair exists so that two graphs with no dependency between them can be
    /// in flight at once: a gated feed-forward block's `gate` and `up`
    /// projections both read the same activations and neither reads the
    /// other, and the driver log shows them dispatched strictly one after
    /// the next — `dispatch job: 0x100000001` / `job ... status is DONE` /
    /// `dispatch job: 0x200000001` — each about a millisecond, while the
    /// device moves roughly 24 GB/s against a memory system that can do
    /// more. `UMD_LOG_LEVEL=3` prints that timeline.
    ///
    /// A job dispatched and never finished holds the device until it is
    /// cleaned up, so every `start` needs its `finish`.
    pub fn start(&self, inputs: &[&[u8]], timeout_ms: i32) -> Result<(), NpuError> {
        if inputs.len() != self.inputs.len() {
            return Err(NpuError::new(
                &format!(
                    "graph takes {} input(s), {} supplied",
                    self.inputs.len(),
                    inputs.len()
                ),
                0,
            ));
        }
        let noe = &self.inner.noe;
        let ctx = self.inner.ctx;

        for (i, (data, desc)) in inputs.iter().zip(&self.inputs).enumerate() {
            if data.len() != desc.size as usize {
                return Err(NpuError::new(
                    &format!(
                        "input {i} is {} bytes, graph expects {}",
                        data.len(),
                        desc.size
                    ),
                    0,
                ));
            }
            // SAFETY: length checked against the descriptor immediately
            // above, which is the only bound the driver has.
            let st = unsafe { (noe.load_tensor)(ctx, self.job, i as c_uint, data.as_ptr().cast()) };
            if st != NOE_STATUS_SUCCESS {
                return Err(NpuError::new(&format!("noe_load_tensor({i}) failed"), st));
            }
        }

        let Some(infer_async) = noe.infer_async else {
            // No async on this driver: run it here and let `finish` just
            // read the outputs. Correct, and simply never overlapped.
            //
            // SAFETY: `job` was created against `ctx` and is still alive.
            let st = unsafe { (noe.infer_sync)(ctx, self.job, timeout_ms) };
            return if st == NOE_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(NpuError::new("noe_job_infer_sync failed", st))
            };
        };
        // SAFETY: `job` was created against `ctx` and is still alive. The
        // null third argument is the optional completion callback.
        let st = unsafe { infer_async(ctx, self.job, std::ptr::null()) };
        if st != NOE_STATUS_SUCCESS {
            return Err(NpuError::new("noe_job_infer_async failed", st));
        }
        Ok(())
    }

    /// Waits for a job dispatched by [`Self::start`] and reads its outputs.
    pub fn finish(&self, outputs: &mut Vec<Vec<u8>>, timeout_ms: i32) -> Result<(), NpuError> {
        // Timed here rather than only in `run_into`, because the gated
        // feed-forward block dispatches and collects directly and would
        // otherwise report nothing at all under `ORANGU_NPU_TIME`. What this
        // measures is the wait plus the read back, which is where a graph
        // run's time actually is: 988 us of a 998 us run at width 1.
        let started = stage_timing().then(std::time::Instant::now);
        let noe = &self.inner.noe;
        let ctx = self.inner.ctx;
        // Nothing to wait for when `start` ran the job synchronously.
        if let Some(get_job_status) = noe.get_job_status {
            let mut status: c_int = JOB_STATUS_NO_STATUS;
            // SAFETY: `job` was created against `ctx`, is still alive and
            // has been dispatched; `status` is a live `c_int`.
            let st = unsafe { get_job_status(ctx, self.job, &mut status, timeout_ms) };
            if st != NOE_STATUS_SUCCESS {
                return Err(NpuError::new("noe_get_job_status failed", st));
            }
            if status != JOB_STATUS_DONE {
                return Err(NpuError::new(
                    &format!("job did not finish: status {status}"),
                    0,
                ));
            }
        }

        outputs.resize_with(self.outputs.len(), Vec::new);
        for (i, (desc, buf)) in self.outputs.iter().zip(outputs.iter_mut()).enumerate() {
            // `resize` rather than a fresh allocation: an existing buffer of
            // the right length is left exactly as it is, so a loop reuses
            // one buffer for good.
            buf.resize(desc.size as usize, 0);
            // SAFETY: `buf` is exactly the byte size the descriptor reports,
            // which is what the driver writes.
            let st = unsafe {
                (noe.get_tensor)(
                    ctx,
                    self.job,
                    TENSOR_TYPE_OUTPUT,
                    i as c_uint,
                    buf.as_mut_ptr().cast(),
                )
            };
            if st != NOE_STATUS_SUCCESS {
                return Err(NpuError::new(&format!("noe_get_tensor({i}) failed"), st));
            }
        }
        if let Some(started) = started {
            let waited = started.elapsed();
            record_stage_timing(waited, std::time::Duration::ZERO);
        }
        Ok(())
    }
}

/// Whether `ORANGU_NPU_TIME=1` asked for a breakdown of where a graph run
/// goes: copying the input in, the driver's synchronous inference, and
/// copying the output back.
///
/// The question it exists to answer is why a one-token block reaches 62.9
/// GFLOP/s when a sixteen-token block of the same weights reaches 508 —
/// which is the same fact as *almost all of a narrow run is fixed cost*,
/// and this says which fixed cost.
pub(crate) fn stage_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ORANGU_NPU_TIME").is_ok_and(|v| !matches!(v.as_str(), "" | "0"))
    })
}

/// Accumulates [`stage_timing`] and reports every 256 runs, so a serving
/// loop produces a handful of lines rather than one per projection.
fn record_stage_timing(total: std::time::Duration, dispatch: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static RUNS: AtomicU64 = AtomicU64::new(0);
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static LOAD: AtomicU64 = AtomicU64::new(0);
    TOTAL.fetch_add(total.as_nanos() as u64, Relaxed);
    LOAD.fetch_add(dispatch.as_nanos() as u64, Relaxed);
    let n = RUNS.fetch_add(1, Relaxed) + 1;
    if !n.is_multiple_of(256) {
        return;
    }
    let us = |v: u64| v as f64 / (n as f64 * 1000.0);
    let (total, dispatch) = (us(TOTAL.load(Relaxed)), us(LOAD.load(Relaxed)));
    eprintln!(
        "orangu-server: [npu] {n} graph runs: {total:.0} us each — {dispatch:.0} to dispatch, \
         {:.0} waiting and reading back",
        total - dispatch
    );
}

impl Drop for NpuGraph {
    fn drop(&mut self) {
        // Job before graph: the job holds buffers allocated against the
        // graph, and the vendor runtime does not unwind that order itself.
        //
        // SAFETY: both handles came from successful calls in `load_graph`
        // and the context outlives this via the `Arc`.
        unsafe {
            (self.inner.noe.clean_job)(self.inner.ctx, self.job);
            (self.inner.noe.unload_graph)(self.inner.ctx, self.graph);
        }
    }
}

/// The device node every process using this NPU holds open.
///
/// Named here rather than discovered, because the question this answers —
/// "is anyone else on the device right now?" — has to be cheap enough to ask
/// during startup and must not itself open the device.
#[cfg(target_os = "linux")]
const DEVICE_NODE: &str = "/dev/aipu";

/// The pids, other than this process, currently holding the NPU open.
///
/// # What this is for
///
/// The device's graph capacity is learned by hitting it: bind artifacts until
/// `noe_create_job` refuses, then remember the count. That number is only the
/// *device's* capacity if the device was otherwise idle when it was measured.
/// Measured while something else held NPU memory, it is a measurement of the
/// leftovers — and because the limit is then persisted and only ever
/// truncates, a single contended startup would quietly cap the model for
/// every run afterwards, with no error and no way back short of deleting the
/// file.
///
/// This board makes that easy to hit: `npu-compile` runs as a separate
/// process and, before [`crate::child::die_with_parent`], an orphaned one was
/// measured still holding `/dev/aipu` **eleven minutes** after its parent was
/// killed. A server started in that window would learn a limit set by a
/// compiler nobody was waiting for.
///
/// # What it can and cannot see
///
/// It walks `/proc/<pid>/fd`, so it sees only processes this user owns.
/// Another user's job on the device is invisible here and would still
/// contaminate a measurement — which is why the caller *declines to persist*
/// on contention rather than claiming the device is clean when this is empty.
/// Best effort, and deliberately so: a `/proc` that cannot be read reports no
/// contention and leaves behaviour exactly as it was before this existed.
#[cfg(target_os = "linux")]
pub fn other_npu_users() -> Vec<u32> {
    let me = std::process::id();
    let mut users = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return users;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me {
            continue;
        }
        let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else {
            // Exited between the listing and the read, or another user's.
            continue;
        };
        let holds = fds
            .flatten()
            .any(|fd| std::fs::read_link(fd.path()).is_ok_and(|t| t.to_str() == Some(DEVICE_NODE)));
        if holds {
            users.push(pid);
        }
    }
    users
}

/// Without `/proc` there is no cheap way to ask, so nothing is reported and
/// the caller behaves as it did before this existed.
#[cfg(not(target_os = "linux"))]
pub fn other_npu_users() -> Vec<u32> {
    Vec::new()
}

/// What went wrong, with the vendor status code kept alongside the message.
///
/// The code is retained rather than folded into the text because NOE's
/// status values are the only thing that distinguishes, say, a timeout from
/// a device exception, and a caller deciding whether to retry needs that.
#[derive(Debug, Clone)]
pub struct NpuError {
    pub message: String,
    /// The raw `noe_status_t`, or 0 when the error was raised here rather
    /// than by the driver.
    pub status: c_int,
}

impl NpuError {
    fn new(message: &str, status: c_int) -> Self {
        Self {
            message: message.to_string(),
            status,
        }
    }
}

impl std::fmt::Display for NpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.status == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{} (NOE status {})", self.message, self.status)
        }
    }
}

impl std::error::Error for NpuError {}

#[cfg(test)]
mod tests {
    /// **Nothing may be reported when nothing holds the device.** A false
    /// positive here is not harmless: the caller reads contention as "do not
    /// remember this limit", so a scan that always found someone would stop
    /// the capacity hint from ever being written and put every start back to
    /// hitting the device's refusal — four leaked buffers each time.
    ///
    /// This machine's own NPU may legitimately be in use by another process,
    /// so the assertion is the one that always holds: whatever comes back,
    /// this process is never in it, and every pid is real.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_npu_user_scan_never_reports_this_process() {
        let users = super::other_npu_users();
        let me = std::process::id();
        assert!(
            !users.contains(&me),
            "the scan must exclude the caller — a server that saw itself \
             would never persist a capacity limit, got {users:?}"
        );
        for pid in &users {
            assert!(
                std::path::Path::new(&format!("/proc/{pid}")).exists(),
                "pid {pid} was reported as holding the NPU but does not exist"
            );
        }
    }

    /// The scan has to survive a `/proc` full of processes it cannot read —
    /// other users' entries, and entries that exit underneath it. It runs at
    /// startup on a busy machine, and a panic there would take the server
    /// down rather than cost a capacity hint.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_npu_user_scan_tolerates_unreadable_processes() {
        // Two passes over a live /proc: between them, processes come and go,
        // which is precisely the condition that must not panic.
        let first = super::other_npu_users();
        let second = super::other_npu_users();
        // No assertion on equality — the machine is allowed to change. The
        // test is that both completed.
        let _ = (first, second);
    }

    use super::*;

    /// The probe's whole contract on a machine with no NPU — which is every
    /// CI runner and nearly every developer machine — is that it answers
    /// rather than failing. There is no NPU to assert against here, so what
    /// is under test is that asking is safe: no panic, no error, no hang.
    #[test]
    fn detecting_an_npu_answers_rather_than_failing_when_there_is_none() {
        // Both outcomes are correct; which one occurs depends on the
        // hardware this happens to run on.
        let _ = detect_npu();
    }

    /// An explicit override replaces the built-in search rather than adding
    /// to it, so a probe pointed at one library can't quietly succeed
    /// against another that happens to be installed.
    #[test]
    fn an_explicit_library_path_replaces_the_built_in_candidates() {
        // SAFETY: single-threaded test process; the variable is read only
        // by `candidates` and restored before returning.
        unsafe { std::env::set_var(LIBRARY_ENV, "/nonexistent/libnoe.so") };
        let paths = candidates();
        unsafe { std::env::remove_var(LIBRARY_ENV) };

        assert_eq!(paths, vec![PathBuf::from("/nonexistent/libnoe.so")]);
    }

    /// An unset or empty override falls back to the built-in list, rather
    /// than to an empty search that would report "no NPU" on a board that
    /// has one.
    #[test]
    fn an_unset_library_path_leaves_the_built_in_candidates_in_place() {
        // SAFETY: as above.
        unsafe { std::env::remove_var(LIBRARY_ENV) };
        assert_eq!(candidates().len(), LIBRARY_CANDIDATES.len());
    }

    /// A library that opens but isn't NOE resolves none of the entry points
    /// and must be rejected, not half-adopted. `libc` is the one shared
    /// object guaranteed to be loadable in this process on the platforms
    /// this probe targets.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_without_the_noe_entry_points_is_rejected() {
        // SAFETY: opening the already-resident C library adds a reference
        // to something this process is linked against regardless.
        let Ok(library) = (unsafe { Library::new("libc.so.6") }) else {
            return; // Not a glibc system; nothing to assert against.
        };
        assert!(Noe::load(&library).is_none());
    }

    /// A graph the vendor BSP ships, used when it is present. Not a
    /// dependency: every assertion below is skipped on a machine without
    /// both an NPU and this file, which is every CI runner.
    const DEMO_GRAPH: &str = "/opt/face-recognition-gradio/arcface.cix";

    fn demo_graph() -> Option<(NpuRuntime, NpuGraph)> {
        let runtime = NpuRuntime::open()?;
        let path = std::path::Path::new(DEMO_GRAPH);
        if !path.is_file() {
            return None;
        }
        let graph = runtime.load_graph(path).ok()?;
        Some((runtime, graph))
    }

    /// Opening the runtime has the same contract as detecting the device:
    /// it answers rather than failing, on any machine.
    #[test]
    fn opening_the_runtime_answers_rather_than_failing() {
        let _ = NpuRuntime::open();
    }

    /// A graph that isn't there is an error, not a panic — and not a
    /// success that would hand back an unusable handle.
    #[test]
    fn loading_a_missing_graph_is_an_error() {
        let Some(runtime) = NpuRuntime::open() else {
            return; // No NPU on this machine.
        };
        let err = runtime.load_graph(std::path::Path::new("/nonexistent/graph.cix"));
        assert!(err.is_err(), "a missing graph must not load");
    }

    /// The end-to-end path against real hardware: load, describe, run, read
    /// back. Asserts the output is *populated*, not merely returned — a
    /// zero-filled buffer is what a silently-failed inference looks like.
    #[test]
    fn a_compiled_graph_runs_on_the_npu_and_produces_output() {
        let Some((_runtime, graph)) = demo_graph() else {
            return; // No NPU, or the vendor demo graph isn't installed.
        };

        assert!(!graph.inputs().is_empty(), "graph declares no inputs");
        assert!(!graph.outputs().is_empty(), "graph declares no outputs");

        let buffers: Vec<Vec<u8>> = graph
            .inputs()
            .iter()
            .map(|t| (0..t.size).map(|i| (i % 251) as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();

        let outputs = graph.run(&refs, 10_000).expect("inference failed");
        assert_eq!(outputs.len(), graph.outputs().len());
        for (out, desc) in outputs.iter().zip(graph.outputs()) {
            assert_eq!(out.len(), desc.size as usize, "output size mismatch");
        }
        assert!(
            outputs.iter().any(|o| o.iter().any(|b| *b != 0)),
            "every output was zero, which is what a failed inference looks like"
        );
    }

    /// The job is reused across runs, so running twice must work and must
    /// not accumulate state — the second inference over identical input has
    /// to match the first.
    #[test]
    fn running_the_same_graph_twice_is_stable() {
        let Some((_runtime, graph)) = demo_graph() else {
            return;
        };
        let buffers: Vec<Vec<u8>> = graph
            .inputs()
            .iter()
            .map(|t| (0..t.size).map(|i| (i % 251) as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();

        let first = graph.run(&refs, 10_000).expect("first inference");
        let second = graph.run(&refs, 10_000).expect("second inference");
        assert_eq!(first, second, "identical input produced different output");
    }

    /// `noe_load_tensor` takes a pointer with no length, so a short input
    /// would have the device read past the end of caller memory. The size
    /// check has to reject it here rather than let the driver run.
    #[test]
    fn an_input_of_the_wrong_size_is_rejected_before_reaching_the_driver() {
        let Some((_runtime, graph)) = demo_graph() else {
            return;
        };
        let short = vec![0u8; 1];
        let refs: Vec<&[u8]> = graph.inputs().iter().map(|_| short.as_slice()).collect();
        let err = graph
            .run(&refs, 10_000)
            .expect_err("short input must be rejected");
        assert!(err.message.contains("bytes"), "unhelpful error: {err}");
    }

    /// Supplying the wrong number of inputs is caught for the same reason,
    /// before any of them is handed over.
    #[test]
    fn the_wrong_number_of_inputs_is_rejected() {
        let Some((_runtime, graph)) = demo_graph() else {
            return;
        };
        let err = graph
            .run(&[], 10_000)
            .expect_err("empty input must be rejected");
        assert!(err.message.contains("input"), "unhelpful error: {err}");
    }
}

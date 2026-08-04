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

//! `--web`: a console for defining benchmark runs and reading their results,
//! built the same way as `orangu-server`'s own web UI — vanilla HTML/CSS/JS
//! embedded into the binary, no build step, no network dependency — so the
//! two look and behave like one product.
//!
//! A run defined in the browser is executed by **re-running this same
//! executable** with the flags the form describes, rather than by calling the
//! measurement functions in-process. That is deliberate:
//!
//! - The number the console reports is then produced by exactly the code path
//!   the command line produces it with. A console that measured through its
//!   own copy of the harness could disagree with `orangu-bench` on the same
//!   machine, and a benchmark tool that reports two different answers for one
//!   workload is worse than no tool.
//! - Every artifact the CLI writes — bundle, chart, flamegraph, the collapsed
//!   profile — lands on disk under the run's own directory with no extra code,
//!   which is what the console then serves.
//! - A run that hangs or has to be abandoned is one process to kill, and
//!   killing it cannot take the console down with it.
//!
//! The command line is built from a typed [`RunSpec`] by [`build_argv`], never
//! from a string the browser sends: the console offers no way to run an
//! arbitrary command. `--sweep`/`--sweep-cmd` are the one part of the tool
//! deliberately left out of the UI for that reason — a sweep's whole input is
//! a shell command to start a server with.
//!
//! **Access is the same posture as `orangu-server`'s console**: no
//! authentication, on the assumption of a trusted network. It differs in its
//! *default* — loopback, where that console defaults to every interface —
//! because this one starts processes rather than answering requests. `--host`
//! opens it up for the real case that wants it (the machine under test is not
//! the machine you are sitting at), and binding off loopback prints a line
//! saying what that means.

use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path as FsPath, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::bundle;
use crate::points;

const INDEX_HTML: &str = include_str!("assets/index.html");
const APP_CSS: &str = include_str!("assets/app.css");
const APP_JS: &str = include_str!("assets/app.js");

/// How many log lines one run keeps in memory. A bench run prints a header, a
/// row per point and a handful of artifact lines — three orders of magnitude
/// below this — so the cap only ever catches a server that has started
/// spewing, which is exactly when an unbounded buffer would be the problem.
const MAX_LOG_LINES: usize = 20_000;

/// The files a finished run may serve, by name. A whitelist rather than a
/// path check: every one of these is written by this tool under the run's own
/// directory, so the set is known, and nothing outside it can be requested at
/// all.
const ARTIFACTS: &[&str] = &[
    "bundle.json",
    "chart.svg",
    "chart.png",
    "flamegraph.svg",
    "flamegraph.png",
    "flamegraph.folded",
    "flamegraph.meta.json",
    "log.txt",
    // The run as one document — provenance, measurements, and both pictures
    // folded in. Written by every run, like the bundle: it is the artifact
    // that leaves this machine.
    "report.pdf",
    // Written by a comparison against an earlier run — see `compare_runs`.
    "compare.txt",
    "compare.svg",
    "compare.png",
    "compare.pdf",
];

const TERMINAL_TITLE: &str = "orangu-bench";

/// Sets the terminal window/tab title via the standard OSC 0 escape sequence
/// (supported by essentially every modern terminal emulator), and restores it
/// (clears it back) on drop. Mirrors `orangu`'s, `orangu-coordinator`'s and
/// `orangu-server`'s own `TerminalTitleGuard`.
///
/// Only the console does this, and only because it is the one mode of this
/// binary that *stays running*: a measurement prints its table and exits, so
/// there is no tab left over to name. A console left open in a background
/// terminal for an afternoon is exactly the window whose title has to say
/// what it is.
struct TerminalTitleGuard;

impl TerminalTitleGuard {
    /// `None` — nothing printed, nothing to restore — when stdout isn't a
    /// terminal, so `orangu-bench --web > console.log` records the startup
    /// banner and not the raw escape bytes.
    fn new(title: &str) -> Option<Self> {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        print!("\x1b]0;{title}\x07");
        // The sequence ends in BEL, not a newline, so a line-buffered stdout
        // would otherwise hold the title back until the first line printed
        // after it.
        let _ = std::io::stdout().flush();
        Some(Self)
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        print!("\x1b]0;\x07");
        let _ = std::io::stdout().flush();
    }
}

/// One run as the browser defines it. Every field is validated into flags by
/// [`build_argv`]; nothing here reaches a shell.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RunSpec {
    /// Base URL of the server under test — *not* this console's own port.
    pub url: String,
    /// `--model`, when the server serves more than one.
    pub model: String,
    /// `--label`: the series name in the bundle and the chart.
    pub label: String,
    /// Which measurement to take: `tg`, `pp`, `pg`, `pp-continue`, `curve`,
    /// `streams`, `embed` or `decode-cpu`. Named for the `mode` each writes
    /// into its records, so a result and the form that produced it agree.
    pub mode: String,
    /// The mode's sweep points, comma-separated: context depths for `tg`,
    /// prompt lengths for `pp`/`pg`/`embed`, added tokens for `pp-continue`,
    /// stream counts for `streams`. Ranges are accepted exactly as the command
    /// line accepts them (`128-2048*2`) — the same parser, so the console
    /// cannot take a sweep a run would refuse. Unused by `curve`, one pass.
    pub points: String,
    /// `--gen`, or the length of the single generation in `curve` mode.
    pub n_gen: u32,
    pub reps: u32,
    /// `--bucket`, `curve` mode only.
    pub bucket: u32,
    /// `--pp-continue-base`, `pp-continue` mode only.
    pub pp_continue_base: u32,
    pub timeout: u64,
    /// `--delay`: seconds between measured points, for a card that heats up
    /// through a sweep.
    pub delay: u64,
    /// Warmup on. Off is `--no-warmup`, which the CLI refuses to combine with
    /// a flamegraph — the console keeps that refusal rather than hiding it.
    pub warmup: bool,
    pub flamegraph: bool,
    pub flamegraph_freq: u32,
    /// `fp` or `dwarf`.
    pub flamegraph_call_graph: String,
    /// `--flamegraph-pid`, when the process to profile is not the one that
    /// owns the URL's port.
    pub flamegraph_pid: Option<u32>,
    /// Render the throughput chart for this run's own points.
    pub chart: bool,
}

impl RunSpec {
    /// What a fresh form starts from. The CLI's own defaults, so a run
    /// launched from the console with nothing touched measures what
    /// `orangu-bench` with nothing passed measures.
    fn defaults() -> RunSpec {
        RunSpec {
            url: "http://127.0.0.1:8100".to_string(),
            model: String::new(),
            label: String::new(),
            mode: "tg".to_string(),
            points: "0".to_string(),
            n_gen: 128,
            reps: 3,
            bucket: 256,
            pp_continue_base: 512,
            timeout: 600,
            delay: 0,
            warmup: true,
            flamegraph: false,
            flamegraph_freq: 999,
            flamegraph_call_graph: "fp".to_string(),
            flamegraph_pid: None,
            chart: true,
        }
    }
}

/// A ready-made scaling sweep for one measurement: the points that answer
/// "how does this scale?" for that axis, rather than the single point the
/// bare defaults measure.
///
/// Every measurement in this tool has an axis it is *about* — decode has
/// context depth, prefill has prompt length, concurrency has stream count —
/// and picking useful points along it is the part that takes knowing the
/// tool. A preset is that knowledge, written down once. Choosing one fills
/// the fields it owns and locks them, so what is about to run is on screen
/// rather than implied; **None** is the default and leaves every field to the
/// user, which is what a one-off measurement and every A/B against a
/// hand-picked depth needs.
///
/// Kept here rather than in `app.js` because a preset is only worth offering
/// if it produces a runnable command line, and that is a claim only this side
/// can test — see `every_measurement_has_a_scaling_preset_that_runs`.
#[derive(Debug, Clone, Serialize)]
struct Preset {
    /// The measurement this belongs to — one preset per mode, matched by the
    /// UI against the selected measurement.
    mode: &'static str,
    /// What the drop-down says: the **range swept**, e.g. `0 to 4096`. A
    /// preset's name is the one thing about it that has to be legible at a
    /// glance from a closed menu, and the range is that thing — "Prefill"
    /// plus "0 to 4096" says the whole sweep in five words.
    range: &'static str,
    /// What the sweep shows, in the console's own words. Shown under the
    /// drop-down, where there is room for the part the range cannot say.
    about: &'static str,
    points: &'static str,
    n_gen: u32,
    reps: u32,
    bucket: u32,
    pp_continue_base: u32,
}

/// One scaling test per measurement — **the sweep this project actually ran**
/// to find what it found, not a set of round numbers.
///
/// Every one of these is lifted from the repository's own performance record,
/// and each is named for the problem it was the instrument of. A preset that
/// swept somewhere else would produce a curve nobody here has a baseline for,
/// which is the opposite of what a preset is for:
///
/// - **decode `0 to 2048`** and **prefill `128 to 3072`** are the two series
///   `perf-history.tsv` has tracked from the beginning — the same points
///   `PERF-GAP.md`'s standard harness runs (`--depths 0,512,1024 --gen 128
///   --reps 3`, `--pp 128,512,1024 --reps 3`, best of 3), extended to the
///   longest depth already in the file. Run them and the numbers land beside
///   every historical row rather than beside nothing.
/// - **continuation prefill `10 to 130 added`** is `PERF-GAP.md` increment
///   7's table exactly. That sweep found a **2× cliff between 50 and 66
///   tokens** — `COOP_MIN_N_TOKENS`, the width at which matmul switches to
///   the tiled cooperative GEMM — and the widths on either side of it are
///   what make the cliff visible. `--pp` cannot reach this regime at all: a
///   whole prompt carries a chat template, so every `--pp` row is a wide
///   batch.
/// - **concurrency `1 to 8 streams`** is the sweep behind "gemma pins the
///   engine at 99% with two streams where the generic path never passes 66%
///   with eight" (`PERF-GAP.md` item 7, re-measured through this tool), and
///   the one `RESEARCH.md` names for re-testing `forward_batch_decode`.
/// - **decode CPU `0 to 1024`** is the depth set that separated a claimed
///   +58% growth in CPU per token from the real +8.8% — the 58% was the
///   prefill's, charged to the generated tokens.
/// - **curve `3072, 256-token buckets`** is the invocation the manual
///   documents for decode-vs-context without a deep-context prefill.
///   `reps` is 1: the mode makes exactly one pass whatever it is set to.
/// - **embeddings `64 to 256`, 15 reps** is `embeddinggemma-300M`'s own sweep
///   from the manual — short prompts and many reps, because a 300M forward
///   pass is fast enough that the run-to-run spread is the measurement.
const PRESETS: &[Preset] = &[
    Preset {
        mode: "tg",
        range: "0 to 2048",
        about: "the tracked decode series: 0, 512, 1024, 2048 tokens of context, best of 3 \
                — the same points perf-history.tsv holds for every build measured so far",
        points: "0,512,1024,2048",
        n_gen: 128,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "pp",
        range: "128 to 3072",
        about: "the tracked prefill series: 128, 512, 1024, 2048, 3072 tokens, best of 3 \
                — the sweep every recorded pp row in perf-history.tsv came from",
        points: "128,512,1024,2048,3072",
        n_gen: 1,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "pg",
        range: "128 to 3072",
        about: "the tracked prefill lengths with the tracked decode length: prompt 128 to 3072 \
                with 128 generated, timed as one turn — the two halves this tool measures \
                separately, measured together",
        points: "128,512,1024,2048,3072",
        n_gen: 128,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "pp-continue",
        range: "10 to 130 added",
        about: "the narrow-batch regime, on a 512-token cached base: the widths that showed \
                the 2× cooperative-GEMM cliff between 50 and 66 tokens (PERF-GAP increment 7)",
        points: "10,18,26,34,50,66,98,130",
        n_gen: 1,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "curve",
        range: "0 to 3072 in one pass",
        about: "one 3072-token generation bucketed every 256 tokens — decode-vs-context \
                scaling without the slow, VRAM-heavy deep-context prefill the depth sweep needs",
        points: "",
        n_gen: 3072,
        reps: 1,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "streams",
        range: "1 to 8 streams",
        about: "the concurrency sweep behind \"99% engine occupancy at two streams, against a \
                generic path that never passes 66% at eight\" (PERF-GAP item 7)",
        points: "1,2,4,8",
        n_gen: 128,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "embed",
        range: "64 to 256",
        about: "embeddinggemma-300M's own sweep: 64, 128, 256 tokens over 15 reps, because a \
                300M forward pass is fast enough that the spread is the measurement",
        points: "64,128,256",
        n_gen: 1,
        reps: 15,
        bucket: 256,
        pp_continue_base: 512,
    },
    Preset {
        mode: "decode-cpu",
        range: "0 to 1024",
        about: "the depths that separated a claimed +58% growth in CPU per token from the real \
                +8.8% — the 58% was the prefill's, charged to the generated tokens",
        points: "0,512,1024",
        n_gen: 192,
        reps: 3,
        bucket: 256,
        pp_continue_base: 512,
    },
];

/// A run's outcome, and the reason a browser can tell "still going" from
/// "finished badly" without reading the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Running,
    Ok,
    Failed,
    Cancelled,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Ok => "ok",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }

    fn from_label(s: &str) -> Status {
        match s {
            "ok" => Status::Ok,
            "cancelled" => Status::Cancelled,
            "running" => Status::Running,
            _ => Status::Failed,
        }
    }
}

/// One line the child wrote, and which stream it came from — `orangu-bench`
/// puts its measurements on stdout and its warnings (a retried send, a
/// flamegraph that could not be written, a profile that undercounted) on
/// stderr, and the console draws that difference rather than flattening it.
#[derive(Debug, Clone, Serialize)]
struct LogLine {
    err: bool,
    text: String,
}

/// A run, live. Everything mutable behind its own lock: the log is appended by
/// two reader threads, the status by the waiter, and read by every poll.
struct Run {
    id: String,
    dir: PathBuf,
    spec: RunSpec,
    started: u64,
    log: Mutex<Vec<LogLine>>,
    status: Mutex<Status>,
    seconds: Mutex<f64>,
    child: Mutex<Option<Child>>,
    /// Set before the kill, so the waiter reports "cancelled" rather than the
    /// "failed" a signal-terminated exit status would otherwise look like.
    cancelled: AtomicBool,
}

pub struct BenchWeb {
    /// `~/.orangu/bench/runs`, where every run gets a directory.
    root: PathBuf,
    /// This executable, re-run once per benchmark.
    exe: PathBuf,
    version: &'static str,
    /// The one run that may be in flight. Benchmarking is exclusive by nature
    /// — two runs sharing a server measure each other's interference — so the
    /// console refuses a second rather than quietly producing two bad numbers.
    current: Mutex<Option<Arc<Run>>>,
}

/// Serve the console on `port`, until Ctrl-C.
pub fn serve(host: &str, port: u16) -> anyhow::Result<()> {
    // Held for as long as the console serves; dropping it on the way out
    // (including the Ctrl-C path below, which returns rather than exits) puts
    // the terminal's own title back.
    let _title = TerminalTitleGuard::new(TERMINAL_TITLE);
    let root = runs_root()?;
    std::fs::create_dir_all(&root)
        .map_err(|e| anyhow::anyhow!("could not create {}: {e}", root.display()))?;
    let state = Arc::new(BenchWeb {
        root,
        exe: std::env::current_exe()
            .map_err(|e| anyhow::anyhow!("could not find this executable: {e}"))?,
        version: env!("CARGO_PKG_VERSION"),
        current: Mutex::new(None),
    });

    let addr = format!("{}:{port}", resolve_bind_host(host));
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| anyhow::anyhow!("could not bind {addr}: {e}"))?;
        println!("orangu-bench {}", state.version);
        let mut rows = vec![
            // The bound address, not the requested host: `all` says nothing
            // about where to point a browser, `0.0.0.0:8300` does — the same
            // reason `orangu-server` prints its listener rather than its
            // config.
            ("Console", format!("http://{}", listener.local_addr()?)),
            ("Runs", state.root.display().to_string()),
        ];
        // Off loopback this console is reachable by anyone who can route to
        // the machine, it has no authentication, and it starts processes.
        // Said once, at the moment the choice takes effect.
        // `is_loopback` is false for the wildcard `0.0.0.0`/`::` as well as
        // for a real interface address, which is exactly the set that needs
        // the warning.
        if !listener.local_addr()?.ip().is_loopback() {
            rows.push((
                "Note",
                "this console has no authentication and starts processes on this machine \
                 — bound off loopback, anyone who can reach it can run benchmarks here"
                    .to_string(),
            ));
        }
        for line in banner_lines(&rows, terminal_width()) {
            println!("{line}");
        }
        let app = build_router(state.clone());
        tokio::select! {
            result = axum::serve(listener, app) => result?,
            _ = tokio::signal::ctrl_c() => println!("shutting down"),
        }
        // A benchmark left running after its console is gone would keep a
        // server busy with nobody watching, and its artifacts would never be
        // finalized. Nothing else can stop it — this process is the only thing
        // holding the handle.
        if let Some(run) = state.current.lock().unwrap().as_ref()
            && run.cancel()
        {
            println!("cancelled the run in flight ({})", run.id);
        }
        Ok::<(), anyhow::Error>(())
    })
}

fn build_router(state: Arc<BenchWeb>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/app.css", get(app_css))
        .route("/static/app.js", get(app_js))
        .route("/api/defaults", get(defaults))
        // `delete` on both, matching `orangu-server`'s console: one row's
        // cross, and the history panel's **Clear all**. These are this
        // console's own scratch results, and being unable to tidy up your own
        // measurements is not a posture anyone asked for.
        .route(
            "/api/runs",
            get(list_runs).post(start_run).delete(clear_runs),
        )
        .route("/api/runs/{id}", get(get_run).delete(delete_run))
        .route("/api/runs/{id}/cancel", post(cancel_run))
        .route("/api/runs/{id}/compare", post(compare_runs))
        .route("/api/runs/{id}/report", post(build_report))
        .route("/api/runs/{id}/artifacts/{name}", get(artifact))
        .with_state(state)
}

/// The startup banner as one aligned table: labels padded to a common width,
/// and a value too long for the terminal continued **under its own column**
/// rather than back at the left margin.
///
/// That last part is the whole reason this exists. The rows were already
/// padded, but the off-loopback note is a sentence, and a sentence printed
/// straight wraps at column 0 — so the table held for two rows and fell apart
/// on the third, which is the one row a reader most needs to take in.
///
/// A value is only ever broken at a space, so a URL or a path stays in one
/// piece and overflows instead: an address split across two lines cannot be
/// copied, and being able to copy it is what the line is for.
fn banner_lines(rows: &[(&str, String)], width: usize) -> Vec<String> {
    let label_width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    // Two spaces between the columns, like `orangu-server`'s own banner.
    let indent = label_width + 2;
    // Never negative, and never so narrow that wrapping makes things worse:
    // below this a value simply overflows, which at least stays readable.
    let room = width.saturating_sub(indent).max(24);
    let mut lines = Vec::new();
    for (label, value) in rows {
        let mut current = format!("{label:<label_width$}  ");
        let mut empty = true;
        for word in value.split_whitespace() {
            if !empty && current.len() - indent + 1 + word.len() > room {
                lines.push(current);
                current = " ".repeat(indent);
                empty = true;
            }
            if !empty {
                current.push(' ');
            }
            current.push_str(word);
            empty = false;
        }
        lines.push(current);
    }
    lines
}

/// The terminal's width, for [`banner_lines`]. Mirrors `orangu`'s own
/// `current_terminal_width` — which lives inside that binary, not the library,
/// so it cannot be called from here — including its `COLUMNS` fallback for a
/// pipe that still wants a sensible width.
fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(width), _)| usize::from(width))
        .filter(|width| *width > 0)
        .or_else(|| {
            std::env::var("COLUMNS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|width| *width > 0)
        })
        .unwrap_or(80)
}

/// Turn a `--host` value into something `bind` understands, spelled the same
/// way `orangu-server`'s `host` is: `all` (and its `*` alias) mean every
/// interface and become the IPv4 wildcard; anything else is a literal address
/// passed through untouched, for `bind` itself to reject if it is not one of
/// this machine's.
///
/// Deliberately duplicated rather than shared: it lives in `orangu-server`'s
/// own `config` module, which is inside that *binary*, not the library. Four
/// lines is a smaller price than moving a server configuration concept into
/// the library so a developer tool can borrow it — but the spelling has to
/// match, because it is the same word in the same product.
fn resolve_bind_host(host: &str) -> &str {
    let host = host.trim();
    if host.eq_ignore_ascii_case("all") || host == "*" {
        "0.0.0.0"
    } else {
        host
    }
}

/// `~/.orangu/orangu-bench/runs`, or `./.orangu/orangu-bench/runs` where there
/// is no home directory to put it in.
///
/// Named for the binary, beside `~/.orangu/server/`'s own sessions: a
/// directory under someone's home should say which tool put it there.
fn runs_root() -> anyhow::Result<PathBuf> {
    let base = match home::home_dir() {
        Some(home) => home.join(".orangu"),
        None => PathBuf::from(".orangu"),
    };
    Ok(base.join("orangu-bench").join("runs"))
}

async fn index(State(state): State<Arc<BenchWeb>>) -> impl IntoResponse {
    Html(
        INDEX_HTML
            .replace("{{VERSION}}", state.version)
            .replace("{{YEAR}}", &current_year().to_string()),
    )
}

async fn app_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

async fn app_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        APP_JS,
    )
}

/// What the form starts from, plus the two facts that decide whether a run
/// *can* produce what the form is about to promise: `perf` for the profile
/// itself, `rsvg-convert` for its PNG. Both are checked here rather than left
/// to fail mid-run, so the UI can say so next to the checkbox instead of the
/// user discovering it twenty minutes into a sweep.
async fn defaults(State(state): State<Arc<BenchWeb>>) -> impl IntoResponse {
    Json(json!({
        "version": state.version,
        "spec": RunSpec::defaults(),
        // The scaling sweeps the measurement drop-down offers, one per
        // measurement. The UI shows the ones whose `mode` matches.
        "presets": PRESETS,
        "have_perf": have_program("perf"),
        "have_rsvg": have_program("rsvg-convert"),
    }))
}

#[derive(Deserialize)]
struct LogQuery {
    /// Index of the first log line the client does not have yet, so a poll
    /// carries the new lines and not the whole log again.
    #[serde(default)]
    from: usize,
}

async fn start_run(
    State(state): State<Arc<BenchWeb>>,
    Json(spec): Json<RunSpec>,
) -> Result<impl IntoResponse, ApiError> {
    let run = state.start(spec)?;
    Ok((StatusCode::CREATED, Json(view(&state, &run.id, 0))))
}

async fn get_run(
    State(state): State<Arc<BenchWeb>>,
    Path(id): Path<String>,
    Query(q): Query<LogQuery>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    let v = view(&state, &id, q.from);
    if v.is_null() {
        return Err(ApiError::not_found(format!("no run {id}")));
    }
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(v)))
}

async fn list_runs(State(state): State<Arc<BenchWeb>>) -> impl IntoResponse {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&state.root) {
        for entry in entries.flatten() {
            let path = entry.path().join("run.json");
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(mut doc) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            // A run recorded as running whose process is gone (the console was
            // killed, the machine rebooted) would otherwise sit in the list
            // claiming to still be measuring, forever. Only the live handle can
            // say it is still going.
            if doc.get("status").and_then(|s| s.as_str()) == Some("running")
                && !state.is_running(doc.get("id").and_then(|s| s.as_str()).unwrap_or(""))
            {
                doc["status"] = json!("failed");
            }
            // Only a run that archived a bundle can be compared against, so
            // the console's comparison menu can offer exactly those.
            doc["has_bundle"] = json!(entry.path().join("bundle.json").is_file());
            rows.push(doc);
        }
    }
    rows.sort_by(|a, b| {
        b.get("started")
            .and_then(serde_json::Value::as_u64)
            .cmp(&a.get("started").and_then(serde_json::Value::as_u64))
    });
    Json(json!({ "runs": rows }))
}

async fn cancel_run(
    State(state): State<Arc<BenchWeb>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    match state.live(&id) {
        // `cancel` reports whether it had anything to stop, so a run that
        // finished between the browser drawing the button and the click
        // reaching here is answered honestly instead of being logged as
        // cancelled after the fact.
        Some(run) if run.cancel() => Ok(Json(json!({ "cancelled": true }))),
        Some(_) => Err(ApiError::conflict(format!("run {id} has already finished"))),
        None => Err(ApiError::not_found(format!("run {id} is not running"))),
    }
}

/// **Report**: build this run's PDF, now.
///
/// Not written by the run itself. A report is a document to send someone, and
/// most runs are not sent anywhere — writing one every time would leave a
/// megabyte of PDFs nobody opened in the runs directory. Everything it needs
/// is already archived (`bundle.json` and the pictures beside it), so it is
/// rebuilt from those on the one click that wants it.
///
/// It is `orangu-bench --read-bundle bundle.json --report report.pdf`, the
/// same path the command line offers for the same job.
async fn build_report(
    State(state): State<Arc<BenchWeb>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    let dir = state.root.join(&id);
    let bundle = dir.join("bundle.json");
    if !bundle.is_file() {
        return Err(ApiError::bad_request(
            "this run has no bundle.json — it did not finish".to_string(),
        ));
    }
    let report = dir.join("report.pdf");
    let output = Command::new(&state.exe)
        .arg("--read-bundle")
        .arg(&bundle)
        .arg("--report")
        .arg(&report)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| ApiError::internal(format!("could not run {}: {e}", state.exe.display())))?;
    if !report.is_file() {
        let why = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::internal(format!(
            "the report was not written{}",
            if why.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", why.trim())
            }
        )));
    }
    Ok(Json(json!({ "artifact": "report.pdf" })))
}

#[derive(Deserialize)]
struct CompareRequest {
    /// The earlier run to compare against.
    with: String,
}

/// Put this run beside an earlier one: what differed in the configuration,
/// what each measured, the ratio between them, and one chart holding both.
///
/// Not reimplemented here either — it is `orangu-bench --read-bundle A,B`,
/// the same comparison the command line does, run against the two runs'
/// archived bundles. That is the whole reason every run writes one.
///
/// Allowed while a benchmark is running, unlike starting a second run: this
/// reads two files that were written when their runs ended and talks to no
/// server, so it cannot disturb a measurement in flight.
async fn compare_runs(
    State(state): State<Arc<BenchWeb>>,
    Path(id): Path<String>,
    Json(req): Json<CompareRequest>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    check_id(&req.with)?;
    if id == req.with {
        return Err(ApiError::bad_request(
            "a run compared against itself says nothing".to_string(),
        ));
    }
    let dir = state.root.join(&id);
    let older = state.root.join(&req.with);
    for (which, path) in [("this run", &dir), ("the earlier run", &older)] {
        if !path.join("bundle.json").is_file() {
            return Err(ApiError::bad_request(format!(
                "{which} has no bundle.json — it did not finish, or it was measured before \
                 bundles were kept"
            )));
        }
    }

    // A working directory rather than two paths straight into the runs: the
    // comparison table names its columns after the bundle *file names*, and
    // every run's bundle is called `bundle.json`, so both columns would read
    // "bundle". Copies named for their role are what make the table legible.
    let work = dir.join("compare");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)
        .map_err(|e| ApiError::internal(format!("could not create {}: {e}", work.display())))?;
    let a = copy_bundle_tagged(&older.join("bundle.json"), &work, "old", &req.with)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let b = copy_bundle_tagged(&dir.join("bundle.json"), &work, "new", &id)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let chart = dir.join("compare.svg");
    let output = Command::new(&state.exe)
        .arg("--read-bundle")
        .arg(format!("{},{}", a.display(), b.display()))
        .arg("--chart")
        .arg(&chart)
        .arg("--chart-png")
        .arg("--chart-x-label")
        .arg("context / prompt length (tokens)")
        // The comparison as a document too — this is the artifact most likely
        // to be attached to a pull request.
        .arg("--report")
        .arg(dir.join("compare.pdf"))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| ApiError::internal(format!("could not run {}: {e}", state.exe.display())))?;

    // The copies exist only to give the comparison two distinguishable
    // names; the tables and the chart are made, so they are dead weight — two
    // whole bundles per comparison.
    let _ = std::fs::remove_dir_all(&work);

    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        text.push('\n');
        text.push_str(&stderr);
    }
    if !output.status.success() && text.trim().is_empty() {
        text = format!("comparison failed ({})", output.status);
    }
    // Kept on disk, both so it can be saved like every other result and so a
    // reloaded page shows the comparison that was already made rather than an
    // empty panel — see `view`.
    let _ = std::fs::write(dir.join("compare.txt"), &text);
    let _ = std::fs::write(
        dir.join("compare.json"),
        serde_json::to_string_pretty(&json!({ "with": req.with })).unwrap_or_default(),
    );

    let artifacts: Vec<&str> = ["compare.svg", "compare.png", "compare.txt", "compare.pdf"]
        .into_iter()
        .filter(|name| dir.join(name).is_file())
        .collect();
    Ok(Json(json!({
        "with": req.with,
        "text": text,
        "artifacts": artifacts,
    })))
}

/// Copy a run's bundle into `work` under a name that says which side of the
/// comparison it is, tagging its series labels to match.
///
/// Both halves are needed for the same reason: two runs of one server carry
/// the *same* label (the model's name) and the same file name, so without
/// this the table's two columns would both read "bundle" and the chart would
/// draw both runs as one line. Only the copy is touched — the run's own
/// bundle keeps the label it was measured under.
fn copy_bundle_tagged(src: &FsPath, work: &FsPath, tag: &str, id: &str) -> anyhow::Result<PathBuf> {
    let text = std::fs::read_to_string(src)
        .map_err(|e| anyhow::anyhow!("could not read {}: {e}", src.display()))?;
    let mut doc: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("{} is not a bundle: {e}", src.display()))?;
    if let Some(records) = doc.get_mut("records").and_then(|r| r.as_array_mut()) {
        for record in records {
            let label = record
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("unknown")
                .to_string();
            record["label"] = json!(format!("{tag} · {label}"));
        }
    }
    let dest = work.join(format!("{tag}-{id}.json"));
    std::fs::write(&dest, serde_json::to_string_pretty(&doc)? + "\n")?;
    Ok(dest)
}

async fn delete_run(
    State(state): State<Arc<BenchWeb>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    if state.is_running(&id) {
        return Err(ApiError::conflict(
            "that run is still going — cancel it first".to_string(),
        ));
    }
    let dir = state.root.join(&id);
    if !dir.is_dir() {
        return Err(ApiError::not_found(format!("no run {id}")));
    }
    std::fs::remove_dir_all(&dir)
        .map_err(|e| ApiError::internal(format!("could not delete {}: {e}", dir.display())))?;
    // The last finished run stays in `current` so a reloaded page can attach
    // to it. Deleting it has to let go of that handle too, or the run keeps
    // answering out of memory — table, log and all — after its directory and
    // every artifact in it are gone.
    let mut current = state.current.lock().unwrap();
    if matches!(current.as_ref(), Some(r) if r.id == id) {
        *current = None;
    }
    Ok(Json(json!({ "deleted": id })))
}

/// **Clear all**: delete every run this console has kept.
///
/// A run still measuring is kept and named in the reply rather than killed —
/// tidying up a list of finished results should never be the thing that ends
/// a twenty-minute sweep. Cancel is one button away and says what it does.
async fn clear_runs(State(state): State<Arc<BenchWeb>>) -> impl IntoResponse {
    let (deleted, kept) = state.clear();
    Json(json!({ "deleted": deleted, "kept": kept }))
}

async fn artifact(
    State(state): State<Arc<BenchWeb>>,
    Path((id, name)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    check_id(&id)?;
    if !ARTIFACTS.contains(&name.as_str()) {
        return Err(ApiError::not_found(format!("no artifact {name}")));
    }
    let path = state.root.join(&id).join(&name);
    let body = std::fs::read(&path).map_err(|_| ApiError::not_found(format!("no {name} here")))?;
    Ok((
        [
            (header::CONTENT_TYPE, content_type(&name)),
            // The flamegraph SVG is interactive (click to zoom, Ctrl-F to
            // search) and the console shows it in a frame, so it must not be
            // cached across a re-run that wrote a new one under the same name.
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    ))
}

fn content_type(name: &str) -> &'static str {
    match FsPath::new(name).extension().and_then(|e| e.to_str()) {
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        Some("pdf") => "application/pdf",
        _ => "text/plain; charset=utf-8",
    }
}

impl BenchWeb {
    /// The live handle for `id`, if this console is the one running it.
    fn live(&self, id: &str) -> Option<Arc<Run>> {
        self.current.lock().unwrap().clone().filter(|r| r.id == id)
    }

    /// Is this run *still measuring*? The finished one stays in `current`
    /// (it is what a fresh page attaches to), so "we know about it" and "it is
    /// going" are different questions — and only the second one may block a
    /// delete.
    fn is_running(&self, id: &str) -> bool {
        matches!(self.live(id), Some(r) if r.status() == Status::Running)
    }

    /// Delete every run directory, and say how many went and which one
    /// stayed. Returns `(deleted, kept)` — `kept` is the run that is still
    /// measuring, when there is one.
    ///
    /// Errors on individual directories are swallowed on purpose: "Clear all"
    /// that removes eleven of twelve and then fails has still done what it
    /// was for, and the count says what happened.
    fn clear(&self) -> (usize, Option<String>) {
        let mut current = self.current.lock().unwrap();
        let running = current
            .as_ref()
            .filter(|r| r.status() == Status::Running)
            .map(|r| r.id.clone());
        let mut deleted = 0;
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if Some(&name) == running.as_ref() {
                    continue;
                }
                if entry.path().is_dir() && std::fs::remove_dir_all(entry.path()).is_ok() {
                    deleted += 1;
                }
            }
        }
        // The finished run this console was still showing has just been
        // deleted with the rest; holding on to its handle would keep it
        // answering out of memory — the same reason `delete_run` lets go.
        if running.is_none() {
            *current = None;
        }
        (deleted, running)
    }

    /// Validate the spec, lay out the run's directory, and start the child.
    ///
    /// The busy check and the store of the new run happen under one lock, so
    /// two browsers pressing Run at the same instant cannot both get past it.
    fn start(&self, spec: RunSpec) -> Result<Arc<Run>, ApiError> {
        let mut current = self.current.lock().unwrap();
        if let Some(run) = current.as_ref()
            && *run.status.lock().unwrap() == Status::Running
        {
            return Err(ApiError::conflict(format!(
                "a run is already going ({}) — benchmarks are measured one at a time",
                run.id
            )));
        }

        let id = new_id();
        let dir = self.root.join(&id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| ApiError::internal(format!("could not create {}: {e}", dir.display())))?;
        let argv = build_argv(&spec, &dir).map_err(|e| ApiError::bad_request(e.to_string()))?;

        let mut child = Command::new(&self.exe)
            .args(&argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ApiError::internal(format!("could not run {}: {e}", self.exe.display()))
            })?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let run = Arc::new(Run {
            id: id.clone(),
            dir: dir.clone(),
            spec,
            started: now_secs(),
            // The command line, first line of the log: a result read months
            // later says exactly what produced it, without the reader having
            // to reconstruct it from the form.
            log: Mutex::new(vec![LogLine {
                err: false,
                text: format!(
                    "$ {} {}",
                    self.exe.display(),
                    argv.iter()
                        .map(|a| shell_quote(a))
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            }]),
            status: Mutex::new(Status::Running),
            seconds: Mutex::new(0.0),
            child: Mutex::new(Some(child)),
            cancelled: AtomicBool::new(false),
        });
        run.persist();

        // Two readers rather than one merged pipe: `orangu-bench` reports its
        // measurements on stdout and its warnings on stderr, and the console
        // colours them differently. Their relative order can interleave — an
        // acceptable price for keeping the distinction at all.
        if let Some(stdout) = stdout {
            spawn_reader(run.clone(), stdout, false);
        }
        if let Some(stderr) = stderr {
            spawn_reader(run.clone(), stderr, true);
        }

        watch(run.clone());

        *current = Some(run.clone());
        Ok(run)
    }
}

/// Follow the child to its end, and own the transition out of `Running` — the
/// one place a run's final state is written.
///
/// Polled with `try_wait` rather than parked in `wait`, because this and
/// [`Run::cancel`] need the same `Child` handle and therefore the same lock. A
/// blocking `wait` holds that lock for the whole run, so a cancel does not
/// fail — it *waits for the run it was cancelling to finish* and then reports
/// success. Measured exactly that way before this was a loop: the Cancel
/// button appeared to work, the run took its full time, and the result came
/// back "ok".
fn watch(run: Arc<Run>) {
    let started = Instant::now();
    std::thread::spawn(move || {
        let status = loop {
            let polled = {
                let mut slot = run.child.lock().unwrap();
                match slot.as_mut() {
                    Some(child) => child.try_wait(),
                    None => break None,
                }
            };
            match polled {
                Ok(Some(status)) => break Some(Ok(status)),
                Err(e) => break Some(Err(e)),
                // Lock released between polls: this is the window a cancel
                // needs, and the only cost is up to this long before a
                // finished run is reported as finished.
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
            }
        };
        let outcome = match status {
            Some(Ok(s)) if s.success() => Status::Ok,
            _ if run.cancelled.load(Ordering::SeqCst) => Status::Cancelled,
            _ => Status::Failed,
        };
        *run.seconds.lock().unwrap() = started.elapsed().as_secs_f64();
        *run.status.lock().unwrap() = outcome;
        *run.child.lock().unwrap() = None;
        run.write_log_file();
        run.persist();
    });
}

/// Drain one of the child's pipes into the run's log, a line at a time, so the
/// console shows a long sweep's rows as they are measured rather than all at
/// once when it finishes.
fn spawn_reader<R: std::io::Read + Send + 'static>(run: Arc<Run>, stream: R, err: bool) {
    std::thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            run.push_log(err, line);
        }
    });
}

impl Run {
    fn push_log(&self, err: bool, text: String) {
        let mut log = self.log.lock().unwrap();
        if log.len() >= MAX_LOG_LINES {
            return;
        }
        log.push(LogLine { err, text });
    }

    /// Kill the child, and say whether there was one to kill. `cancelled` is
    /// set first so the waiter can tell this apart from a run that failed on
    /// its own.
    ///
    /// `false` for a run that has already finished: it is not an error worth
    /// failing a shutdown over, but it must not append "cancelled" to a log
    /// whose run completed normally.
    fn cancel(&self) -> bool {
        if self.status() != Status::Running {
            return false;
        }
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some(child) = self.child.lock().unwrap().as_mut() {
            let _ = child.kill();
        }
        self.push_log(true, "cancelled".to_string());
        true
    }

    fn status(&self) -> Status {
        *self.status.lock().unwrap()
    }

    /// The run's own record on disk, rewritten at start and at finish. It is
    /// what the history list is built from, so a run outlives the console
    /// process that started it.
    fn persist(&self) {
        let doc = json!({
            "id": self.id,
            "spec": self.spec,
            "started": self.started,
            "status": self.status().as_str(),
            "seconds": *self.seconds.lock().unwrap(),
        });
        let _ = std::fs::write(
            self.dir.join("run.json"),
            serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n",
        );
    }

    /// The console's log pane, as a file, so a finished run reads back the
    /// same on a console restart as it did live.
    fn write_log_file(&self) {
        let log = self.log.lock().unwrap();
        let mut text = String::new();
        for line in log.iter() {
            text.push_str(&line.text);
            text.push('\n');
        }
        let _ = std::fs::write(self.dir.join("log.txt"), text);
    }
}

/// Everything the browser draws for one run: its definition, its state, the
/// log from `from` on, and — once it has finished — the measurements and the
/// artifacts, read back off disk rather than kept in memory.
///
/// `Null` when there is no such run, which the handler turns into a 404.
fn view(state: &BenchWeb, id: &str, from: usize) -> serde_json::Value {
    let live = state.live(id);
    let dir = state.root.join(id);

    let (spec, started, status, seconds, lines, total) = match &live {
        Some(run) => {
            let log = run.log.lock().unwrap();
            let total = log.len();
            let lines: Vec<LogLine> = log.iter().skip(from).cloned().collect();
            drop(log);
            (
                serde_json::to_value(&run.spec).unwrap_or(serde_json::Value::Null),
                run.started,
                run.status(),
                *run.seconds.lock().unwrap(),
                lines,
                total,
            )
        }
        None => {
            let Ok(text) = std::fs::read_to_string(dir.join("run.json")) else {
                return serde_json::Value::Null;
            };
            let doc: serde_json::Value = serde_json::from_str(&text).unwrap_or(json!({}));
            // A run.json still saying "running" with nothing live behind it is
            // a console that was killed mid-run; see `list_runs`.
            let status = match doc.get("status").and_then(|s| s.as_str()) {
                Some("running") | None => Status::Failed,
                Some(s) => Status::from_label(s),
            };
            let text = std::fs::read_to_string(dir.join("log.txt")).unwrap_or_default();
            let all: Vec<LogLine> = text
                .lines()
                .map(|l| LogLine {
                    err: false,
                    text: l.to_string(),
                })
                .collect();
            let total = all.len();
            (
                doc.get("spec").cloned().unwrap_or(serde_json::Value::Null),
                doc.get("started")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                status,
                doc.get("seconds")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
                all.into_iter().skip(from).collect(),
                total,
            )
        }
    };

    let bundle = bundle::read(&dir.join("bundle.json").display().to_string()).ok();
    let records = bundle.as_ref().map_or_else(Vec::new, |b| {
        b.records
            .iter()
            .map(|r| {
                json!({
                    "date": r.date, "label": r.label, "mode": r.mode,
                    "n": r.n, "best": r.best, "mean": r.mean,
                    // Both estimators, named apart: `sd` is the population
                    // one the history file has always carried, `sd_sample`
                    // (÷ n-1) is the standard one the table shows — `null`
                    // for a single repetition and for a bundle written
                    // before the field existed.
                    "sd": r.sd, "sd_sample": r.sd_sample,
                })
            })
            .collect()
    });

    json!({
        "id": id,
        "spec": spec,
        "started": started,
        "status": status.as_str(),
        "seconds": seconds,
        "log": lines,
        "log_total": total,
        "records": records,
        "props": bundle.as_ref().map(|b| b.props.clone()),
        "host": bundle.as_ref().map(|b| b.host.clone()),
        "run": bundle.as_ref().map(|b| b.run.clone()),
        "gpu_timings": bundle.as_ref().map(|b| b.gpu_timings.clone()),
        // The profile's own sidecar (`profile::Recorder::finish`): samples,
        // cores busy, how much of the window was spent waiting for the GPU.
        // A flamegraph shows how time was divided; only these say how much
        // there was to divide.
        "profile": std::fs::read_to_string(dir.join("flamegraph.meta.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()),
        "artifacts": ARTIFACTS
            .iter()
            .filter(|name| dir.join(name).is_file())
            .collect::<Vec<_>>(),
        // A comparison already made against an earlier run, so reopening this
        // run shows it rather than an empty panel — see `compare_runs`.
        "compare": std::fs::read_to_string(dir.join("compare.txt"))
            .ok()
            .map(|text| json!({
                "text": text,
                "with": std::fs::read_to_string(dir.join("compare.json"))
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                    .and_then(|v| v.get("with").and_then(|w| w.as_str()).map(str::to_string)),
            })),
    })
}

/// Turn a run definition into the command line that measures it.
///
/// The whole surface between the browser and this machine. Every value is
/// typed or parsed here — a mode that is not one of the seven, a sweep point
/// that is not a number, a URL that is not HTTP — and the result is an argv
/// passed to `Command::args`, never a shell string.
fn build_argv(spec: &RunSpec, dir: &FsPath) -> anyhow::Result<Vec<String>> {
    if !(spec.url.starts_with("http://") || spec.url.starts_with("https://")) {
        anyhow::bail!(
            "the target URL must start with http:// or https://, got {:?}",
            spec.url
        );
    }
    let mut argv: Vec<String> = vec!["--url".into(), spec.url.clone()];

    // Parsed before anything is added, so a typo in the points list is one
    // clean message rather than a partly-built command line.
    // Validated with the command line's own parser — ranges included — so the
    // console refuses exactly what a run would refuse, and says the same thing
    // about it. What gets passed on is the *spec* as typed, not the expansion:
    // `--pp 128-2048*2` in the echoed command line is what the user wrote, and
    // pasting that line back into a terminal does the same thing.
    let points = || -> anyhow::Result<String> {
        let items: Vec<&str> = spec
            .points
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if points::expand_list(&items)?.is_empty() {
            anyhow::bail!(
                "this mode needs at least one point — a comma-separated list like 0,512,1024, \
                 or a range like 128-2048*2"
            );
        }
        Ok(items.join(","))
    };

    match spec.mode.as_str() {
        "tg" => {
            argv.push("--depths".into());
            argv.push(points()?);
            argv.push("--gen".into());
            argv.push(spec.n_gen.to_string());
        }
        "pp" => {
            argv.push("--pp".into());
            argv.push(points()?);
        }
        // The combined test: prompt lengths swept, generation length fixed by
        // `--gen`, reported as one rate over the whole turn.
        "pg" => {
            argv.push("--pg".into());
            argv.push(points()?);
            argv.push("--gen".into());
            argv.push(spec.n_gen.to_string());
        }
        "pp-continue" => {
            argv.push("--pp-continue".into());
            argv.push(points()?);
            argv.push("--pp-continue-base".into());
            argv.push(spec.pp_continue_base.to_string());
        }
        "embed" => {
            argv.push("--embed".into());
            argv.push(points()?);
        }
        "streams" => {
            argv.push("--streams".into());
            argv.push(points()?);
            argv.push("--gen".into());
            argv.push(spec.n_gen.to_string());
        }
        "decode-cpu" => {
            argv.push("--decode-cpu".into());
            argv.push("--depths".into());
            argv.push(points()?);
            argv.push("--gen".into());
            argv.push(spec.n_gen.to_string());
        }
        // One pass, bucketed by context — so the form's "tokens to generate"
        // is the length of that pass, and the points list has no meaning here.
        "curve" => {
            if spec.n_gen == 0 {
                anyhow::bail!("curve mode needs a token count to generate");
            }
            argv.push("--curve".into());
            argv.push(spec.n_gen.to_string());
            argv.push("--bucket".into());
            argv.push(spec.bucket.max(1).to_string());
        }
        other => anyhow::bail!("unknown mode {other:?}"),
    }

    argv.push("--reps".into());
    argv.push(spec.reps.max(1).to_string());
    argv.push("--timeout".into());
    argv.push(spec.timeout.max(1).to_string());
    if spec.delay > 0 {
        argv.push("--delay".into());
        argv.push(spec.delay.to_string());
    }
    if !spec.model.trim().is_empty() {
        argv.push("--model".into());
        argv.push(spec.model.trim().to_string());
    }
    if !spec.label.trim().is_empty() {
        argv.push("--label".into());
        argv.push(spec.label.trim().to_string());
    }
    if !spec.warmup {
        argv.push("--no-warmup".into());
    }

    // Always written: the bundle *is* the result the console reads back —
    // records, the server's configuration while measuring, and the host — so
    // a run always leaves the file its summary table is built from.
    argv.push("--bundle".into());
    argv.push(dir.join("bundle.json").display().to_string());

    // No `--report` here. The PDF is built on demand (see `build_report`),
    // from the bundle and the pictures this run leaves behind — so a run that
    // nobody asks a report of does not write one, and the runs directory keeps
    // only what cannot be regenerated from it.

    if spec.chart {
        argv.push("--chart".into());
        argv.push(dir.join("chart.svg").display().to_string());
        argv.push("--chart-png".into());
    }

    if spec.flamegraph {
        argv.push("--flamegraph".into());
        argv.push(dir.join("flamegraph.svg").display().to_string());
        // Both artifacts, always: the console offers the SVG to read and the
        // PNG to paste, and a profile is expensive enough to record that
        // asking for the other format later means running it again.
        argv.push("--flamegraph-png".into());
        argv.push("--flamegraph-freq".into());
        argv.push(spec.flamegraph_freq.max(1).to_string());
        argv.push("--flamegraph-call-graph".into());
        argv.push(match spec.flamegraph_call_graph.as_str() {
            "dwarf" => "dwarf".into(),
            _ => "fp".to_string(),
        });
        if let Some(pid) = spec.flamegraph_pid {
            argv.push("--flamegraph-pid".into());
            argv.push(pid.to_string());
        }
    }

    Ok(argv)
}

/// A run id is minted here and is always digits — so this is not a sanitizer
/// covering for a loose format, it is the check that a path segment arriving
/// from a browser is one of ours before it is joined to the runs directory.
fn check_id(id: &str) -> Result<(), ApiError> {
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ApiError::bad_request(format!("not a run id: {id:?}")));
    }
    Ok(())
}

/// Milliseconds since the epoch: unique (runs are serialized), sortable as a
/// string, and digits only — see [`check_id`].
fn new_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{millis:013}")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The year, for the footer. Days since the epoch through the civil-date
/// arithmetic `history` already carries, rather than a date crate.
fn current_year() -> i64 {
    crate::history::today()
        .split('-')
        .next()
        .and_then(|y| y.parse().ok())
        .unwrap_or(2026)
}

/// Is `name` runnable from `PATH`? Used only to tell the browser what this
/// machine can produce before a run promises it.
fn have_program(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(name).is_file())
}

/// Quote an argument for the echoed command line in the log. Display only —
/// nothing built here is ever executed — but it has to be *correct* display,
/// because the line's whole purpose is to be copied into a terminal and re-run.
fn shell_quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_=/.:,+@".contains(&b))
    {
        return arg.to_string();
    }
    format!("'{}'", arg.replace('\'', r"'\''"))
}

/// An error with the status the browser should see. `anyhow` everywhere else
/// in this tool; here the code matters — a busy console (409) and a malformed
/// spec (400) are different things to the UI.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: String) -> ApiError {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            message,
        }
    }
    fn not_found(message: String) -> ApiError {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message,
        }
    }
    fn conflict(message: String) -> ApiError {
        ApiError {
            status: StatusCode::CONFLICT,
            message,
        }
    }
    fn internal(message: String) -> ApiError {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> RunSpec {
        RunSpec::defaults()
    }

    fn argv(spec: &RunSpec) -> Vec<String> {
        build_argv(spec, RUN_DIR.as_ref()).expect("builds")
    }

    const RUN_DIR: &str = "/runs/1";

    /// `--bundle <run dir>/bundle.json`, spelled with **this platform's**
    /// separator.
    ///
    /// The paths are built by `Path::join`, which writes `\` on Windows —
    /// so an assertion containing a hard-coded `/` passes everywhere the
    /// developer looked and fails in CI on one platform, describing a defect
    /// that is not there. Building the expectation the same way the code does
    /// keeps the test about *what goes where* rather than about separators.
    fn flag(name: &str, file: &str) -> String {
        format!("{name} {}", FsPath::new(RUN_DIR).join(file).display())
    }

    #[test]
    fn a_decode_run_becomes_the_command_line_that_measures_it() {
        let mut s = spec();
        s.points = "0,512,1024".to_string();
        let argv = argv(&s);
        let joined = argv.join(" ");
        assert!(joined.contains("--depths 0,512,1024"), "{joined}");
        assert!(joined.contains("--gen 128"), "{joined}");
        assert!(joined.contains("--reps 3"), "{joined}");
        // Always, because the summary table is read back out of it.
        assert!(
            joined.contains(&flag("--bundle", "bundle.json")),
            "{joined}"
        );
    }

    #[test]
    fn each_mode_reaches_for_its_own_flag() {
        for (mode, flag) in [
            ("pp", "--pp"),
            ("pp-continue", "--pp-continue"),
            ("embed", "--embed"),
            ("streams", "--streams"),
            ("decode-cpu", "--decode-cpu"),
        ] {
            let mut s = spec();
            s.mode = mode.to_string();
            s.points = "128,512".to_string();
            let joined = argv(&s).join(" ");
            assert!(joined.contains(flag), "{mode}: {joined}");
        }
    }

    #[test]
    fn curve_mode_generates_once_and_ignores_the_points() {
        let mut s = spec();
        s.mode = "curve".to_string();
        s.n_gen = 512;
        s.bucket = 128;
        // Deliberately unparseable: curve takes no sweep points, so a stale
        // value left in the form must not fail the run.
        s.points = "not a list".to_string();
        let joined = argv(&s).join(" ");
        assert!(joined.contains("--curve 512"), "{joined}");
        assert!(joined.contains("--bucket 128"), "{joined}");
    }

    #[test]
    fn both_flamegraph_formats_are_asked_for_together() {
        let mut s = spec();
        s.flamegraph = true;
        let joined = argv(&s).join(" ");
        assert!(
            joined.contains(&flag("--flamegraph", "flamegraph.svg")),
            "{joined}"
        );
        assert!(joined.contains("--flamegraph-png"), "{joined}");
        assert!(joined.contains("--flamegraph-call-graph fp"), "{joined}");
        // The chart is on by default and gets the same treatment.
        assert!(joined.contains(&flag("--chart", "chart.svg")), "{joined}");
        assert!(joined.contains("--chart-png"), "{joined}");
    }

    /// A range typed into the console has to reach the run as the range —
    /// not as its expansion, and not rejected. The console and the command
    /// line share one parser precisely so a sweep cannot be accepted by one
    /// and refused by the other.
    #[test]
    fn a_range_survives_the_console_and_stays_a_range() {
        let mut s = spec();
        s.mode = "pp".to_string();
        s.points = "128-2048*2, 3072".to_string();
        let joined = argv(&s).join(" ");
        assert!(joined.contains("--pp 128-2048*2,3072"), "{joined}");

        // And what the CLI would refuse, the console refuses here — before a
        // run is started, with the parser's own reason.
        let mut s = spec();
        s.points = "128-4096".to_string();
        let err = build_argv(&s, RUN_DIR.as_ref()).expect_err("capped");
        assert!(err.to_string().contains("more than"), "{err}");
    }

    #[test]
    fn the_delay_is_passed_only_when_it_is_asked_for() {
        let mut s = spec();
        assert!(!argv(&s).join(" ").contains("--delay"), "0 means no flag");
        s.delay = 30;
        assert!(argv(&s).join(" ").contains("--delay 30"));
    }

    #[test]
    fn a_bad_spec_is_refused_before_anything_runs() {
        let mut s = spec();
        s.url = "file:///etc/passwd".to_string();
        assert!(build_argv(&s, RUN_DIR.as_ref()).is_err());

        let mut s = spec();
        s.mode = "rm -rf".to_string();
        assert!(build_argv(&s, RUN_DIR.as_ref()).is_err());

        let mut s = spec();
        s.points = "512; rm -rf /".to_string();
        assert!(build_argv(&s, RUN_DIR.as_ref()).is_err());

        let mut s = spec();
        s.points = " ".to_string();
        assert!(build_argv(&s, RUN_DIR.as_ref()).is_err());
    }

    #[test]
    fn free_text_fields_travel_as_one_argument_each() {
        // `--label`/`--model` are the only fields a user types freely. They
        // reach `Command::args`, so a value with a space or a quote in it is
        // one argument and not a shell fragment — this test is what keeps
        // that true if the construction is ever rewritten.
        let mut s = spec();
        s.label = "orangu #7 \"tuned\"".to_string();
        s.model = "a b".to_string();
        let argv = argv(&s);
        assert!(
            argv.contains(&"orangu #7 \"tuned\"".to_string()),
            "{argv:?}"
        );
        assert!(argv.contains(&"a b".to_string()), "{argv:?}");
    }

    #[test]
    fn only_a_minted_id_reaches_the_runs_directory() {
        assert!(check_id(&new_id()).is_ok());
        for bad in ["", "..", "../../etc", "1234abc", "12/34", "."] {
            assert!(check_id(bad).is_err(), "accepted {bad:?}");
        }
    }

    /// Every measurement the form offers has a scaling sweep behind it, and
    /// every one of those sweeps builds a command line this tool would run.
    ///
    /// The second half is the point. A preset is a promise made in a menu —
    /// pick this and get a scaling curve — and a set of points that failed
    /// validation would break that promise at the moment the user pressed
    /// Run, which is the worst place to find out. Building the argv here is
    /// the same call the console makes.
    #[test]
    fn every_measurement_has_a_scaling_preset_that_runs() {
        // The modes the drop-down offers, i.e. everything `build_argv`
        // accepts. A mode added without a preset fails here.
        let modes = [
            "tg",
            "pp",
            "pg",
            "pp-continue",
            "curve",
            "streams",
            "embed",
            "decode-cpu",
        ];
        for mode in modes {
            let found: Vec<&Preset> = PRESETS.iter().filter(|p| p.mode == mode).collect();
            assert_eq!(found.len(), 1, "{mode} should have exactly one preset");
            let preset = found[0];
            assert!(!preset.range.is_empty(), "{mode}: the menu needs a label");

            let mut s = spec();
            s.mode = mode.to_string();
            s.points = preset.points.to_string();
            s.n_gen = preset.n_gen;
            s.reps = preset.reps;
            s.bucket = preset.bucket;
            s.pp_continue_base = preset.pp_continue_base;
            let argv = build_argv(&s, RUN_DIR.as_ref())
                .unwrap_or_else(|e| panic!("{mode} preset does not run: {e}"));
            // A scaling test is a sweep, so it has to leave more than the one
            // point the bare defaults measure.
            let swept = if mode == "curve" {
                // Curve's points come out of one pass, bucketed — its sweep is
                // in `--bucket`, not in a list.
                argv.contains(&"--bucket".to_string())
            } else {
                preset.points.split(',').count() > 2
            };
            assert!(swept, "{mode} preset does not sweep anything: {argv:?}");
        }
        assert_eq!(PRESETS.len(), modes.len(), "a preset names an unknown mode");
    }

    /// Build a console over a fresh temporary runs directory, holding `run`
    /// as its current one.
    ///
    /// `name` is the caller's, and it has to be distinct per test: run ids are
    /// milliseconds, tests run concurrently, and two of these built in the
    /// same millisecond shared one directory — each deleting the other's runs
    /// mid-assertion. It failed only when the whole suite ran, which is the
    /// worst way to find out.
    fn console(name: &str, run: Option<Arc<Run>>) -> (Arc<BenchWeb>, PathBuf) {
        let root = std::env::temp_dir().join(format!("orangu-bench-{name}-{}", new_id()));
        std::fs::create_dir_all(&root).expect("temp root");
        let web = Arc::new(BenchWeb {
            root: root.clone(),
            exe: PathBuf::from("/nonexistent"),
            version: "test",
            current: Mutex::new(run),
        });
        (web, root)
    }

    fn fake_run(root: &FsPath, id: &str) -> PathBuf {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).expect("run dir");
        std::fs::write(dir.join("run.json"), "{}").expect("run.json");
        dir
    }

    #[test]
    fn clear_all_removes_every_finished_run() {
        let (web, root) = console("clear-finished", None);
        for id in ["1", "2", "3"] {
            fake_run(&root, id);
        }
        let (deleted, kept) = web.clear();
        assert_eq!(deleted, 3);
        assert_eq!(kept, None);
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        // And it lets go of the run it was showing, which no longer exists.
        assert!(web.current.lock().unwrap().is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn clear_all_keeps_the_run_that_is_still_measuring() {
        // Tidying a list of finished results must never be the thing that
        // ends a twenty-minute sweep — so the live run survives, and the
        // caller is told which one it was rather than left to infer it.
        let (web, root) = console("clear-live", None);
        for id in ["1", "2"] {
            fake_run(&root, id);
        }
        let live = Arc::new(Run {
            id: "2".to_string(),
            dir: root.join("2"),
            spec: spec(),
            started: now_secs(),
            log: Mutex::new(Vec::new()),
            status: Mutex::new(Status::Running),
            seconds: Mutex::new(0.0),
            child: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        });
        *web.current.lock().unwrap() = Some(live);

        let (deleted, kept) = web.clear();
        assert_eq!(deleted, 1);
        assert_eq!(kept.as_deref(), Some("2"));
        assert!(root.join("2").is_dir(), "the live run's directory survives");
        assert!(!root.join("1").is_dir());
        // Still attached to it: it is still measuring.
        assert!(web.current.lock().unwrap().is_some());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_banner_is_one_table_however_long_a_row_gets() {
        let rows = [
            ("Console", "http://127.0.0.1:8300".to_string()),
            ("Runs", "/home/you/.orangu/bench/runs".to_string()),
            (
                "Note",
                "this console has no authentication and starts processes on this machine"
                    .to_string(),
            ),
        ];
        let lines = banner_lines(&rows, 60);
        // Labels padded to the longest, values all starting in one column.
        let value_at = lines[0].find("http").expect("the value is on the line");
        assert_eq!(&lines[0][..value_at], "Console  ");
        assert_eq!(lines[1].find('/'), Some(value_at));
        // The long row wrapped — and its continuation lines start under the
        // value column, not at the left margin. That is the whole point: a
        // sentence printed straight breaks the table at exactly the row a
        // reader most needs to take in.
        let wrapped: Vec<&String> = lines[2..].iter().collect();
        assert!(wrapped.len() > 1, "the note should wrap at 60 columns");
        for line in &wrapped[1..] {
            assert_eq!(&line[..value_at], " ".repeat(value_at));
            assert!(!line[value_at..].starts_with(' '), "{line:?}");
        }
        for line in &lines {
            assert!(line.len() <= 60, "{} columns: {line:?}", line.len());
        }
    }

    #[test]
    fn a_long_address_is_never_broken_in_half() {
        // A URL or a path is one word, and half a URL cannot be copied — so
        // an over-long value overflows rather than wrapping.
        let long = format!("http://{}:8300", "a".repeat(80));
        let lines = banner_lines(&[("Console", long.clone())], 40);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].ends_with(&long), "{:?}", lines[0]);
    }

    #[test]
    fn the_host_is_spelled_the_way_the_rest_of_the_product_spells_it() {
        for value in ["all", "ALL", " All ", "*", " * "] {
            assert_eq!(resolve_bind_host(value), "0.0.0.0", "--host {value}");
        }
        // Everything else is a literal address, passed through for `bind` to
        // accept or refuse — this must not become a guess at what was meant.
        for value in ["127.0.0.1", "0.0.0.0", "192.168.1.10", "::1"] {
            assert_eq!(resolve_bind_host(value), value);
        }
        assert_eq!(resolve_bind_host("  127.0.0.1 "), "127.0.0.1");
    }

    #[test]
    fn the_artifact_list_is_a_whitelist() {
        assert!(ARTIFACTS.contains(&"flamegraph.svg"));
        assert!(ARTIFACTS.contains(&"flamegraph.png"));
        // run.json is the console's own bookkeeping, not a result.
        assert!(!ARTIFACTS.contains(&"run.json"));
    }

    /// Cancel has to stop a run that is *still going*, which is the only state
    /// it is ever pressed in — so the child outlives the test by two orders of
    /// magnitude, and what is asserted is how fast it dies.
    ///
    /// Two details are what make this bite rather than pass by luck. The
    /// **pause before cancelling** puts the waiter where a real cancel finds
    /// it — already following the child — instead of racing the thread spawn;
    /// without it the test passed against the very bug it exists for. And
    /// cancel is **timed**, because the bug's symptom was not an error, it was
    /// a cancel that blocked until the run it was cancelling had finished on
    /// its own and then reported success.
    #[cfg(unix)]
    #[test]
    fn cancelling_a_run_stops_it_now_and_says_so() {
        let dir = std::env::temp_dir().join(format!("orangu-bench-test-{}", new_id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawns");
        let run = Arc::new(Run {
            id: "1".to_string(),
            dir: dir.clone(),
            spec: spec(),
            started: now_secs(),
            log: Mutex::new(Vec::new()),
            status: Mutex::new(Status::Running),
            seconds: Mutex::new(0.0),
            child: Mutex::new(Some(child)),
            cancelled: AtomicBool::new(false),
        });
        watch(run.clone());
        // Long enough for the waiter to be following the child rather than
        // still starting up — see this test's own doc comment.
        std::thread::sleep(std::time::Duration::from_millis(250));

        let pressed = Instant::now();
        assert!(run.cancel(), "a running run reports that it was cancelled");
        assert!(
            pressed.elapsed() < std::time::Duration::from_secs(5),
            "cancel took {:?} — it waited for the run instead of stopping it",
            pressed.elapsed()
        );
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while run.status() == Status::Running && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(run.status(), Status::Cancelled);
        // Not "failed": a killed child and a benchmark that fell over exit the
        // same way, and only this flag tells them apart.
        assert!(run.cancelled.load(Ordering::SeqCst));
        // A second press has nothing to stop, and must not say it did.
        assert!(!run.cancel());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_echoed_command_line_can_be_pasted_back_into_a_shell() {
        assert_eq!(shell_quote("--depths"), "--depths");
        assert_eq!(
            shell_quote("http://127.0.0.1:8100"),
            "http://127.0.0.1:8100"
        );
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }
}

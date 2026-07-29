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

//! `orangu-bench` — a small developer tool that measures the throughput of a
//! running OpenAI-compatible server over HTTP. Point it at any server that
//! speaks `POST /v1/completions` with SSE streaming — **both `orangu-server`
//! and `llama-server`** do — and it reports either:
//!
//! - **decode** (the default, and `--curve`): steady-state token-generation
//!   tok/s at one or more context depths, timed from the first streamed token
//!   to the last so prefill and TTFT are excluded — `llama-bench`'s `tg`.
//! - **prefill** (`--pp`): prompt-processing tok/s, taken from the server's
//!   own `timings` so the token count is exact and a prefix-cache hit is
//!   visible rather than disguised as a fast run — `llama-bench`'s `pp`.
//!
//! It exists because "how fast is decode, and how does it scale with context?"
//! needs the *same* measurement applied to both engines through the *same*
//! path — not `llama-bench` (in-process) compared against an ad-hoc HTTP curl
//! of orangu. This tool is that apples-to-apples harness.
//!
//! This is a **developer tool**, not part of the served product; it is
//! documented only in `doc/manual/en/79-bench.md`.
//!
//! Example:
//! ```text
//! # orangu-server on :8100, sweep decode rate across context depths
//! orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024,2048,3072 --gen 128
//! # llama-server on :8300, same harness (uses the OpenAI-compat endpoint)
//! orangu-bench --url http://127.0.0.1:8300 --depths 0,512,1024,2048,3072 --gen 128
//! # prefill throughput at a few prompt lengths
//! orangu-bench --url http://127.0.0.1:8100 --pp 128,512,1024,2048
//! ```

use std::io::{BufRead, BufReader};
use std::time::Instant;

use clap::Parser;

mod chart;
mod flamegraph;
mod history;
mod profile;

/// Measure decode (token-generation) throughput of an OpenAI-compatible
/// server over HTTP, at one or more context depths.
#[derive(Parser, Debug)]
#[command(name = "orangu-bench", version, about)]
struct Args {
    /// Base URL of the server.
    #[arg(long, default_value = "http://127.0.0.1:8100")]
    url: String,

    /// Comma-separated context depths to sweep.
    #[arg(long, default_value = "0", value_delimiter = ',')]
    depths: Vec<u32>,

    /// Prefill mode: comma-separated prompt lengths (in tokens) to sweep,
    /// reporting **prompt-processing** throughput instead of the decode sweep
    /// — `llama-bench`'s `pp` to the default `tg`. Rates come from the
    /// server's own `timings`, so the token count is exact rather than
    /// inferred from the requested length, and the reported cache hit makes a
    /// prompt that was never actually processed impossible to mistake for a
    /// fast one.
    #[arg(long, value_delimiter = ',')]
    pp: Vec<u32>,

    /// Number of tokens to generate per timed run.
    #[arg(long = "gen", default_value_t = 128)]
    n_gen: u32,

    /// Curve mode: instead of the depth sweep, do ONE generation of this many
    /// tokens and report the instantaneous decode rate bucketed by context
    /// position. Measures decode-vs-context scaling without the slow, VRAM-heavy
    /// deep-context prefill the depth sweep needs. `0` disables it.
    #[arg(long, default_value_t = 0)]
    curve: u32,

    /// Bucket width (in context tokens) for `--curve`.
    #[arg(long, default_value_t = 256)]
    bucket: u32,

    /// Repetitions per depth; the reported rate is the best run with mean±sd.
    #[arg(long, default_value_t = 3)]
    reps: u32,

    /// Skip the initial warmup run.
    #[arg(long, default_value_t = false)]
    no_warmup: bool,

    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    timeout: u64,

    /// Model id to request.
    #[arg(long)]
    model: Option<String>,

    /// Emit machine-readable JSON.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Append each measured point to this tab-separated history file.
    #[arg(long)]
    history: Option<String>,

    /// Series name recorded in the history file (defaults to the server's model).
    #[arg(long)]
    label: Option<String>,

    /// Render the history file to this SVG after measuring.
    #[arg(long)]
    chart: Option<String>,

    /// Only render the chart from an existing history file; measure nothing.
    #[arg(long, default_value_t = false)]
    chart_only: bool,

    /// Record a CPU flamegraph of the server over the measured window.
    #[arg(long)]
    flamegraph: Option<String>,

    /// Process to profile (default: the server's own, else the URL port's owner).
    #[arg(long)]
    flamegraph_pid: Option<u32>,

    /// Sampling frequency in Hz for `--flamegraph`.
    #[arg(long, default_value_t = 999)]
    flamegraph_freq: u32,

    /// Call-graph mode for `--flamegraph`: `fp` or `dwarf`.
    #[arg(long, default_value = "fp")]
    flamegraph_call_graph: String,

    /// Also render a PNG beside the flamegraph SVG.
    #[arg(long, default_value_t = false)]
    flamegraph_png: bool,

    /// Compare already-collapsed `.folded` profiles side by side; measure nothing.
    #[arg(long, value_delimiter = ',')]
    compare_profiles: Vec<String>,
}

/// POST `body` to `endpoint`, retrying once on a connection-level failure.
///
/// Not defensive programming — a specific, reproduced failure. `llama-server`
/// closes an idle keep-alive connection on its own schedule, and a client that
/// reuses it at the wrong moment gets a reset. In a long sweep that surfaces as
/// one lost measurement out of hundreds, aborting the run: four of eight models'
/// prefill profiles were lost to it, and every one of them succeeded on the
/// first retry against the *same still-running server*.
///
/// Exactly one retry, and only for `send` failing — a refused connection retried
/// forever would turn "the server is not up" into a hang, and an HTTP error
/// status is a real answer that must not be papered over. The retry is announced
/// so a run that needed one is never mistaken for a clean one.
fn post_with_one_retry(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    body: &serde_json::Value,
) -> anyhow::Result<reqwest::blocking::Response> {
    match client.post(endpoint).json(body).send() {
        Ok(resp) => Ok(resp),
        Err(first) => {
            eprintln!("orangu-bench: retrying after a failed send ({first})");
            // Long enough for a server that is closing connections in a batch
            // to finish, short enough not to distort a timed run that recovers.
            std::thread::sleep(std::time::Duration::from_millis(500));
            client
                .post(endpoint)
                .json(body)
                .send()
                // Carry reqwest's own reason. "Error sending request" on its own
                // cannot distinguish a server that was never up from one that
                // dropped a connection mid-sweep, and those need different
                // responses — the difference cost an afternoon once.
                .map_err(|e| anyhow::anyhow!("Error sending request to url ({endpoint}): {e}"))
        }
    }
}

/// One decode measurement: how many tokens streamed, time-to-first-token,
/// and the pure decode window (first→last token).
struct Sample {
    gen_tokens: u32,
    ttft_ms: f64,
    decode_s: f64,
}

impl Sample {
    fn tok_per_s(&self) -> f64 {
        if self.decode_s > 0.0 && self.gen_tokens > 1 {
            (self.gen_tokens - 1) as f64 / self.decode_s
        } else {
            0.0
        }
    }
}

/// Build a prompt whose token count is approximately `depth`. Content is
/// irrelevant to decode speed — only the resulting KV length matters — but it
/// must be *coherent* text ending on an open-ended instruction, or a greedy
/// model given a degenerate repeated-token prompt just emits end-of-sequence
/// immediately and generates nothing to time. So we pad with a repeated
/// natural-language paragraph (~1 token/word) and close with a forceful
/// "continue, do not stop" instruction. `depth == 0` returns just the
/// instruction.
fn build_prompt(depth: u32) -> String {
    // A strong open-ended tail keeps a temperature-0 model generating rather
    // than stopping. Kept explicit ("do not stop") on purpose.
    let tail = "\n\nContinue this narrative in vivid detail for many paragraphs, \
                and do not stop or conclude:";
    if depth == 0 {
        return format!(
            "Tell a long, continuous, detailed story about a journey across a continent.{tail}"
        );
    }
    // One coherent ~18-word sentence, repeated to fill ~depth tokens. Real
    // words (≈ one BPE token each) keep the model in "continue prose" mode
    // instead of the immediate-EOS a degenerate repeat provokes.
    let sentence = "The travelers pressed on through the valley as the pale morning light \
                    spread over the hills and the road wound slowly toward the distant sea. ";
    let words_per = 24u32; // approximate token count of `sentence`
    let repeats = (depth / words_per).max(1);
    let mut s = String::with_capacity(repeats as usize * sentence.len() + tail.len() + 64);
    s.push_str("Here is the story so far:\n\n");
    for _ in 0..repeats {
        s.push_str(sentence);
    }
    s.push_str(tail);
    s
}

/// One prefill measurement, as the server reported it.
struct PrefillSample {
    prompt_tokens: u32,
    cached_tokens: u32,
    prompt_ms: f64,
    /// `None` when the server reported no `timings` — the rate then falls back
    /// to wall-clock time-to-first-token over the *requested* length, which is
    /// approximate, and the caller marks the row.
    server_reported: bool,
}

impl PrefillSample {
    fn tok_per_s(&self) -> f64 {
        if self.prompt_ms > 0.0 {
            self.prompt_tokens as f64 / (self.prompt_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Send one prompt, generate a single token, and report what prefill cost.
///
/// `cache_prompt: false` is what keeps this honest: without it the second and
/// later reps would find their prompt already in the server's prefix cache and
/// report the speed of a cache lookup. Both `orangu-server` and `llama-server`
/// honour it; the `cached` column the caller prints is the check that whatever
/// server answered actually did.
fn run_prefill_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    model: &Option<String>,
) -> anyhow::Result<PrefillSample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        // One token: enough to force the whole prompt through prefill and to
        // get a final chunk, with as little decode as possible in the way.
        "max_tokens": 1,
        "n_predict": 1,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
        // llama-server only attaches `timings` to its OpenAI-compatible
        // responses when asked; orangu-server always sends them.
        "timings_per_token": true,
    });
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    let endpoint = format!("{url}/v1/completions");
    let t0 = Instant::now();
    let resp = post_with_one_retry(client, &endpoint, &body)?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned HTTP {}", resp.status());
    }

    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut first: Option<Instant> = None;
    let mut timings: Option<(u32, f64)> = None;
    let mut cached = 0u32;

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let payload = match line.trim_start().strip_prefix("data:") {
            Some(p) => p.trim(),
            None => continue,
        };
        if payload == "[DONE]" {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| v.get("content").and_then(|t| t.as_str()))
            .unwrap_or("");
        if !text.is_empty() && first.is_none() {
            first = Some(Instant::now());
        }
        if let Some(t) = v.get("timings") {
            let n = t.get("prompt_n").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            let ms = t.get("prompt_ms").and_then(|m| m.as_f64()).unwrap_or(0.0);
            if n > 0 && ms > 0.0 {
                timings = Some((n, ms));
            }
        }
        if let Some(p) = v.get("prompt_progress") {
            cached = p.get("cache").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
        }
    }

    match timings {
        Some((prompt_tokens, prompt_ms)) => Ok(PrefillSample {
            prompt_tokens,
            cached_tokens: cached,
            prompt_ms,
            server_reported: true,
        }),
        // No `timings` from this server: fall back to time-to-first-token,
        // which includes queueing and the first decode step. The caller marks
        // these rows so the two are never silently compared.
        None => Ok(PrefillSample {
            prompt_tokens: 0,
            cached_tokens: cached,
            prompt_ms: (first.unwrap_or_else(Instant::now) - t0).as_secs_f64() * 1000.0,
            server_reported: false,
        }),
    }
}

/// Send one streaming completion and time the decode window.
fn run_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    n_gen: u32,
    model: &Option<String>,
) -> anyhow::Result<Sample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": n_gen,
        // llama.cpp's native field name, harmless to OpenAI servers that
        // ignore it — sending both maximizes cross-server compatibility.
        "n_predict": n_gen,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
        // Generate exactly `n_gen` tokens regardless of content — without this a
        // greedy model handed a depth-padded (repetitive) prompt emits EOS on
        // the first token, so the non-zero-depth rows timed **0** tokens. This
        // is the same "measure decode, not content" contract `llama-bench -d`
        // uses.
        "ignore_eos": true,
    });
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    let endpoint = format!("{url}/v1/completions");
    let t0 = Instant::now();
    let resp = post_with_one_retry(client, &endpoint, &body)?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned HTTP {}", resp.status());
    }

    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut first: Option<Instant> = None;
    let mut last = t0;
    let mut n: u32 = 0;

    loop {
        line.clear();
        // A mid-stream read error (server dropped the connection, timeout) ends
        // the stream rather than crashing — we time whatever tokens did arrive.
        let read = match reader.read_line(&mut line) {
            Ok(read) => read,
            Err(_) => break,
        };
        if read == 0 {
            break;
        }
        let trimmed = line.trim_start();
        let payload = match trimmed.strip_prefix("data:") {
            Some(p) => p.trim(),
            None => continue,
        };
        if payload == "[DONE]" {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // OpenAI `choices[0].text`, or llama.cpp native `content`.
        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| v.get("content").and_then(|t| t.as_str()))
            .unwrap_or("");
        if !text.is_empty() {
            let now = Instant::now();
            if first.is_none() {
                first = Some(now);
            }
            last = now;
            n += 1;
        }
    }

    let first = first.unwrap_or(last);
    Ok(Sample {
        gen_tokens: n,
        ttft_ms: (first - t0).as_secs_f64() * 1000.0,
        decode_s: (last - first).as_secs_f64(),
    })
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        // A single clean line (e.g. a refused connection), not anyhow's
        // multi-line "Error: … Caused by: …" chain.
        eprintln!("orangu-bench: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> anyhow::Result<()> {
    // Chart-only never touches the network, so it works on a history file
    // carried off the machine that produced it — and, more usefully, after a
    // hand-edit of that file, without needing a server up to redraw.
    if args.chart_only {
        return write_chart(args, &[]);
    }

    // Same shape as `--chart-only`: reads artifacts, touches no server.
    if !args.compare_profiles.is_empty() {
        return compare_profiles(args);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    // Warmup first (it also validates the connection), so a failure here prints
    // just the clean error above rather than a header followed by an error.
    if !args.no_warmup {
        let p = build_prompt(0);
        run_once(&client, &args.url, &p, 8, &args.model)?;
    }

    let label = args
        .label
        .clone()
        .unwrap_or_else(|| server_label(&client, args));

    report_environment(&client, args);

    // Started here — after warmup, after the environment probe — so the profile
    // covers the timed workload and nothing else. Anything before this point is
    // load, allocation and HTTP that the reported rate already excludes, and a
    // flamegraph that included it would attribute time the number does not.
    let recorder = match &args.flamegraph {
        Some(path) => Some(start_profile(&client, args, path, &label)?),
        None => None,
    };

    let clocks = ClockWatch::start();
    let measured = measure(&client, args, &label);
    report_clocks(&clocks.stop(), args);

    if let Some(recorder) = recorder {
        match recorder.finish() {
            Ok(summary) => report_profile(&summary, args),
            // The rate is the deliverable; a profile that failed to render is
            // reported and does not discard the measurement that just ran.
            Err(e) => eprintln!("orangu-bench: flamegraph not written: {e}"),
        }
    }

    record_and_chart(args, &measured?)
}

/// The timed part of a run, split out so [`run`] can bracket exactly this with
/// the profiler.
fn measure(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    if !args.pp.is_empty() {
        return run_pp(client, args, label);
    }

    if args.curve > 0 {
        // Curve mode reports one generation bucketed by position, not a rate at
        // a named workload; there is no series for it to extend, so it is not
        // recorded. A chart requested alongside it still redraws the file.
        run_curve(client, args)?;
        return Ok(Vec::new());
    }

    run_tg(client, args, label)
}

/// The decode sweep: one row per requested context depth.
fn run_tg(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();

    if !args.json {
        println!(
            "{:>8} | {:>5} | {:>7} | {:>8} | {:>8} | {:>16}",
            "depth", "gen", "ttft_ms", "n_tok", "best", "mean ± sd"
        );
        println!("{}", "-".repeat(67));
    }

    for &depth in &args.depths {
        let prompt = build_prompt(depth);
        let mut rates = Vec::new();
        let mut last_sample: Option<Sample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_once(client, &args.url, &prompt, args.n_gen, &args.model)?;
            rates.push(s.tok_per_s());
            last_sample = Some(s);
        }
        let best = rates.iter().cloned().fold(0.0_f64, f64::max);
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
        let sd = var.sqrt();
        let s = last_sample.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "depth": depth,
                    "n_gen": args.n_gen,
                    "ttft_ms": s.ttft_ms,
                    "tok_per_s_best": best,
                    "tok_per_s_mean": mean,
                    "tok_per_s_sd": sd,
                    "gen_tokens": s.gen_tokens,
                })
            );
        } else {
            println!(
                "{:>8} | {:>5} | {:>7.0} | {:>8} | {:>8.2} | {:>8.2} ± {:>5.2}",
                depth, args.n_gen, s.ttft_ms, s.gen_tokens, best, mean, sd
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "tg".to_string(),
            n: depth,
            best,
            mean,
            sd,
        });
    }

    Ok(records)
}

/// Append this run's points to `--history` (when given) and redraw `--chart`
/// (when given) from the file *including* them.
///
/// Drawing from the file rather than from `records` is what makes the chart a
/// history rather than a snapshot: a run that measured only orangu still
/// redraws llama.cpp's line beside it.
fn record_and_chart(args: &Args, records: &[history::Record]) -> anyhow::Result<()> {
    if let Some(path) = &args.history
        && !records.is_empty()
    {
        history::append(path, records)?;
        if !args.json {
            println!("  history  {} rows appended to {path}", records.len());
        }
    }
    write_chart(args, records)
}

/// Render `--chart` from `--history`. `extra` is folded in so a chart requested
/// without a history file still shows the run that just happened.
fn write_chart(args: &Args, extra: &[history::Record]) -> anyhow::Result<()> {
    let Some(chart_path) = &args.chart else {
        if args.chart_only {
            anyhow::bail!("--chart-only needs --chart <FILE.svg>");
        }
        return Ok(());
    };
    let mut records = match &args.history {
        Some(h) => history::read(h)?,
        None => Vec::new(),
    };
    // Only when there is no history file: otherwise these rows are already in
    // it, and folding them in again would double every point of this run.
    if args.history.is_none() {
        records.extend_from_slice(extra);
    }
    if records.is_empty() {
        anyhow::bail!("nothing to chart — pass --history <FILE> with recorded rows");
    }
    let subtitle = format!(
        "{} · rendered {}",
        args.history.as_deref().unwrap_or("this run"),
        history::today()
    );
    std::fs::write(chart_path, chart::render(&records, &subtitle))?;
    if !args.json {
        println!("  chart    {chart_path}");
    }
    Ok(())
}

/// The series name for a server that was not given one: its model id, which is
/// the field that actually distinguishes two rows in the history file.
fn server_label(client: &reqwest::blocking::Client, args: &Args) -> String {
    client
        .get(format!("{}/props", args.url))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|p| {
            p.get("model")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Print `--compare-profiles` as one table: buckets down the side, one column
/// per collapsed profile, then each profile's heaviest leaves.
///
/// The comparison is in *shares*, not sample counts, and the counts are printed
/// so nobody reads a share as an absolute. Two engines on the same workload
/// produce different sample totals — the faster one finishes sooner and is
/// sampled for less wall-clock — so "36% here against 12% there" is a statement
/// about how each engine divides its own CPU time, and the rate printed by the
/// run that produced each file is what converts it back to seconds.
fn compare_profiles(args: &Args) -> anyhow::Result<()> {
    let paths: Vec<std::path::PathBuf> = args
        .compare_profiles
        .iter()
        .map(std::path::PathBuf::from)
        .collect();
    let profiles = profile::read_profiles(&paths)?;

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "profile_comparison",
                "profiles": profiles.iter().map(|p| serde_json::json!({
                    "name": p.name,
                    "samples": p.samples,
                    "cores_busy": p.cores_busy,
                    "gpu_wait_pct": p.gpu_wait,
                    "pool_idle_pct": p.pool_idle,
                    "buckets": p.buckets.iter().map(|(k, v)| serde_json::json!({"bucket": k, "pct": v})).collect::<Vec<_>>(),
                    "leaves": p.leaves.iter().map(|(k, v)| serde_json::json!({"frame": k, "pct": v})).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            })
        );
        return Ok(());
    }

    let width = profiles
        .iter()
        .map(|p| p.name.chars().count().max(9))
        .collect::<Vec<_>>();
    print!("{:<14}", "bucket");
    for (p, w) in profiles.iter().zip(&width) {
        print!(" | {:>w$}", p.name, w = w);
    }
    println!();
    println!(
        "{}",
        "-".repeat(14 + width.iter().map(|w| w + 3).sum::<usize>())
    );
    for bucket in profile::bucket_union(&profiles) {
        print!("{bucket:<14}");
        for (p, w) in profiles.iter().zip(&width) {
            let pct = p
                .buckets
                .iter()
                .find(|(b, _)| *b == bucket)
                .map_or(0.0, |(_, v)| *v);
            print!(" | {:>w$}", format!("{pct:.1}%"), w = w);
        }
        println!();
    }
    print!("{:<14}", "samples");
    for (p, w) in profiles.iter().zip(&width) {
        print!(" | {:>w$}", p.samples, w = w);
    }
    println!();
    // The row that stops the shares above being read as work: an engine that
    // blocks in the kernel is sampled less than one that spins, whatever each
    // is doing with the time.
    // Shares first: they come from the collapsed file itself, so they survive
    // a `.folded` carried off the machine on its own. Occupancy needs the
    // sidecar and is printed as `?` without it rather than guessed at.
    type ShareRow = (&'static str, fn(&profile::Profile) -> f64);
    let share: [ShareRow; 3] = [
        ("gpu-wait", |p| p.gpu_wait),
        ("pool-idle", |p| p.pool_idle),
        ("working", |p| 100.0 - p.gpu_wait - p.pool_idle),
    ];
    for (row, share) in share {
        print!("{row:<14}");
        for (p, w) in profiles.iter().zip(&width) {
            print!(" | {:>w$}", format!("{:.1}%", share(p)), w = w);
        }
        println!();
    }
    for (row, share) in [("cores busy", None), ("— working", Some(()))] {
        print!("{row:<14}");
        for (p, w) in profiles.iter().zip(&width) {
            let cell = p.cores_busy.map_or_else(
                || "?".to_string(),
                |c| match share {
                    None => format!("{c:.2}"),
                    Some(()) => format!("{:.2}", c * (100.0 - p.gpu_wait - p.pool_idle) / 100.0),
                },
            );
            print!(" | {:>w$}", cell, w = w);
        }
        println!();
    }
    println!();

    for p in &profiles {
        println!("{} — heaviest self-time frames:", p.name);
        for (frame, pct) in p.leaves.iter().take(12) {
            let short: String = frame.chars().take(72).collect();
            println!("  {pct:>5.1}%  {short}");
        }
        println!();
    }
    Ok(())
}

/// Begin a flamegraph capture of whichever process is answering `--url`.
fn start_profile(
    client: &reqwest::blocking::Client,
    args: &Args,
    svg: &str,
    label: &str,
) -> anyhow::Result<profile::Recorder> {
    let pid = match args.flamegraph_pid {
        Some(pid) => pid,
        None => resolve_server_pid(client, args)?,
    };
    if !args.json {
        println!("  profiling pid {pid} at {} Hz", args.flamegraph_freq);
    }
    profile::Recorder::start(profile::Options {
        svg: std::path::PathBuf::from(svg),
        pid,
        freq: args.flamegraph_freq,
        call_graph: args.flamegraph_call_graph.clone(),
        png: args.flamegraph_png,
        title: format!("{label} · {}", workload_name(args)),
    })
}

/// Which process to sample: the one the server names, else the one the
/// operating system says owns the port under test.
///
/// The second route is not a fallback for tidiness — `llama-server` reports no
/// pid at all, and it is half of every comparison this tool exists to make.
/// Both routes identify the process that *answered these requests*, which is
/// the only definition that cannot profile the wrong binary.
fn resolve_server_pid(client: &reqwest::blocking::Client, args: &Args) -> anyhow::Result<u32> {
    let reported = client
        .get(format!("{}/props", args.url))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|p| p.get("pid").and_then(serde_json::Value::as_u64))
        .map(|p| p as u32);
    if let Some(pid) = reported {
        return Ok(pid);
    }

    let port = url_port(&args.url)
        .ok_or_else(|| anyhow::anyhow!("could not read a port out of --url {}", args.url))?;
    profile::pid_listening_on(port).ok_or_else(|| {
        anyhow::anyhow!(
            "the server did not report a pid and nothing owns port {port} that this user \
             can see — pass --flamegraph-pid <PID>"
        )
    })
}

/// The port in a `scheme://host:port/…` URL.
fn url_port(url: &str) -> Option<u16> {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme.split(['/', '?']).next()?;
    authority.rsplit_once(':')?.1.parse().ok()
}

/// A short name for what was measured, drawn on the flamegraph so the SVG
/// carries its own workload rather than depending on its filename.
fn workload_name(args: &Args) -> String {
    let list = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    if !args.pp.is_empty() {
        format!("prefill pp {}", list(&args.pp))
    } else if args.curve > 0 {
        format!("decode curve {}", args.curve)
    } else {
        format!("decode gen {} at depth {}", args.n_gen, list(&args.depths))
    }
}

/// What the profile says, printed under the rate it explains.
fn report_profile(s: &profile::Summary, args: &Args) {
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "profile",
                "svg": s.svg,
                "folded": s.folded,
                "png": s.png,
                "samples": s.samples,
                "seconds": s.seconds,
                "cores_busy": s.cores_busy,
                "gpu_wait_pct": s.gpu_wait,
                "pool_idle_pct": s.pool_idle,
                "buckets": s.buckets.iter().map(|(k, v)| serde_json::json!({"bucket": k, "pct": v})).collect::<Vec<_>>(),
                "leaves": s.leaves.iter().map(|(k, v)| serde_json::json!({"frame": k, "pct": v})).collect::<Vec<_>>(),
            })
        );
        return;
    }
    println!(
        "  profile  {} ({} samples over {:.0}s — {:.2} cores busy)",
        s.svg.display(),
        s.samples,
        s.seconds,
        s.cores_busy
    );
    println!(
        "           {:.2} gpu-wait  {:.2} pool-idle  {:.2} working (cores)",
        s.cores_busy * s.gpu_wait / 100.0,
        s.cores_busy * s.pool_idle / 100.0,
        s.cores_busy * (100.0 - s.gpu_wait - s.pool_idle) / 100.0,
    );
    println!("           {}", s.folded.display());
    if let Some(png) = &s.png {
        println!("           {}", png.display());
    }
    for (bucket, pct) in &s.buckets {
        println!("           {bucket:<12} {pct:>5.1}%");
    }
    println!("           top self-time frames:");
    for (frame, pct) in s.leaves.iter().take(10) {
        // Long Rust and C++ symbols would wrap and destroy the column; the SVG
        // and the collapsed file carry the untruncated name.
        let short: String = frame.chars().take(64).collect();
        println!("           {pct:>5.1}%  {short}");
    }
}

/// Prefill mode: for each requested prompt length, time prompt processing at
/// that length. The rate is `prompt_n / prompt_ms` straight from the server, so
/// it excludes decode entirely — the `pp` counterpart to the decode sweep.
fn run_pp(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>7} | {:>9} | {:>8} | {:>16}",
            "pp", "n_tok", "cached", "prompt_ms", "best", "mean ± sd"
        );
        println!("{}", "-".repeat(70));
    }

    for &len in &args.pp {
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<PrefillSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_prefill_once(client, &args.url, &prompt, &args.model)?;
            rates.push(s.tok_per_s());
            last = Some(s);
        }
        let best = rates.iter().cloned().fold(0.0_f64, f64::max);
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
        let sd = var.sqrt();
        let s = last.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "pp": len,
                    "prompt_tokens": s.prompt_tokens,
                    "cached_tokens": s.cached_tokens,
                    "prompt_ms": s.prompt_ms,
                    "tok_per_s_best": best,
                    "tok_per_s_mean": mean,
                    "tok_per_s_sd": sd,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9.1} | {:>8.2} | {:>8.2} ± {:>5.2}",
                len, s.prompt_tokens, s.cached_tokens, s.prompt_ms, best, mean, sd
            );
        } else {
            // No server timings: the only honest thing to print is the
            // wall-clock TTFT, and that it is not the same measurement.
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9.1} | {:>8} | no server timings (ttft only)",
                len, "?", s.cached_tokens, s.prompt_ms, "-"
            );
        }

        // A row without server timings is a time-to-first-token, not a prefill
        // rate. Recording it would put a different measurement on the same line
        // as the real ones, so it is printed and dropped.
        if s.server_reported {
            records.push(history::Record {
                date: history::today(),
                label: label.to_string(),
                mode: "pp".to_string(),
                n: s.prompt_tokens,
                best,
                mean,
                sd,
            });
        }
    }
    Ok(records)
}

/// What was measured, printed before the numbers: which server process, the
/// model, the backend, and the GPU's clock state.
///
/// The GPU clock matters because an AMD card left at
/// `power_dpm_force_performance_level = auto` can idle its core clock down
/// between requests, which moves throughput enough to swamp the difference a
/// benchmark is usually run to detect — a rate recorded without it is not
/// comparable against a later one.
///
/// `pid`/`up` matter for the same reason, and catch a sharper failure: an A/B
/// that never actually swapped binaries. A launcher that stops the old server
/// by process name misses a build copied under a different filename, the new
/// server then fails to bind the port and exits, and the benchmark happily
/// measures the *old* one — reporting the two builds as identical, which reads
/// as a credible "no change" result rather than as the broken measurement it
/// is. A pid that does not change between runs, or an uptime far longer than
/// this run, says so immediately.
fn report_environment(client: &reqwest::blocking::Client, args: &Args) {
    let props: Option<serde_json::Value> = client
        .get(format!("{}/props", args.url))
        .send()
        .ok()
        .and_then(|r| r.json().ok());
    let field = |key: &str| -> String {
        props
            .as_ref()
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    };
    let num = |key: &str| -> Option<u64> {
        props
            .as_ref()
            .and_then(|p| p.get(key))
            .and_then(serde_json::Value::as_u64)
    };
    let model = field("model");
    let backend = field("backend");
    let pid = num("pid");
    let uptime = num("uptime_seconds");
    let gpus = gpu_clock_states();

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "env",
                "url": args.url,
                "model": model,
                "backend": backend,
                "pid": pid,
                "uptime_seconds": uptime,
                "gpus": gpus,
            })
        );
    } else {
        println!("orangu-bench → {}", args.url);
        println!("  model    {model}");
        println!("  backend  {backend}");
        // Only for a server that reports them; llama-server does not, and a
        // missing field is not worth a line of output.
        if pid.is_some() || uptime.is_some() {
            let show = |v: Option<u64>| v.map_or_else(|| "?".to_string(), |n| n.to_string());
            println!("  server   pid {} up {}s", show(pid), show(uptime));
        }
        for gpu in &gpus {
            println!(
                "  gpu      {} sclk {} ({})",
                gpu.card, gpu.sclk, gpu.power_level
            );
        }
    }
}

/// One GPU's current core clock and power-management mode, read from sysfs.
#[derive(serde::Serialize)]
struct GpuClock {
    card: String,
    sclk: String,
    power_level: String,
}

/// Current core clock and DPM mode of every AMD card exposing them under
/// `/sys/class/drm`. Empty on any platform or driver that does not (nothing to
/// report is not an error — the rest of the run is unaffected).
fn gpu_clock_states() -> Vec<GpuClock> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut cards: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card") && !n.contains('-'))
        })
        .collect();
    cards.sort();
    for card in cards {
        let device = card.join("device");
        let Ok(sclk_raw) = std::fs::read_to_string(device.join("pp_dpm_sclk")) else {
            continue;
        };
        // The active level is the one sysfs marks with a trailing `*`; its
        // line reads `<level>: <freq> *`, and only the frequency is wanted.
        let sclk = sclk_raw
            .lines()
            .find(|l| l.trim_end().ends_with('*'))
            .and_then(|l| l.trim_end().trim_end_matches('*').trim().split_once(": "))
            .map(|(_, freq)| freq.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let power_level = std::fs::read_to_string(device.join("power_dpm_force_performance_level"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        out.push(GpuClock {
            card: card
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("card")
                .to_string(),
            sclk,
            power_level,
        });
    }
    out
}

/// The core clock each GPU actually reached while the workload ran.
///
/// The header line reports the clock at the moment the run *starts*, which on an
/// idle laptop dGPU is its sleep state: `card1 sclk 0Mhz (high)` is what a
/// correctly pinned card looks like between requests, and it reads exactly like
/// a misconfigured one. Worse, it is evidence about a second when nothing was
/// being measured. Sampling through the run answers the question the header line
/// was added to answer — did the card reach its top level *for this
/// measurement* — and it is the check that catches a card that quietly stayed
/// parked while a whole sweep was recorded against it.
struct ClockWatch {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    sampler: Option<std::thread::JoinHandle<Vec<(String, u32)>>>,
}

impl ClockWatch {
    fn start() -> ClockWatch {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let sampler = std::thread::spawn(move || {
            let mut peak: std::collections::BTreeMap<String, u32> =
                std::collections::BTreeMap::new();
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                for gpu in gpu_clock_states() {
                    let mhz = parse_mhz(&gpu.sclk).unwrap_or(0);
                    let slot = peak.entry(gpu.card).or_default();
                    *slot = (*slot).max(mhz);
                }
                // Fast enough to catch the ramp on a short run, slow enough
                // that reading sysfs is not itself part of the measurement.
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
            peak.into_iter().collect()
        });
        ClockWatch {
            stop,
            sampler: Some(sampler),
        }
    }

    fn stop(mut self) -> Vec<(String, u32)> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.sampler
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default()
    }
}

/// `1700Mhz` → `1700`. Anything else — including the `0Mhz` sleep level, which
/// parses fine — is left to the caller to interpret.
fn parse_mhz(sclk: &str) -> Option<u32> {
    let digits: String = sclk.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Peak clocks, printed under the numbers they qualify.
fn report_clocks(peaks: &[(String, u32)], args: &Args) {
    if peaks.is_empty() {
        return;
    }
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "clock_peaks",
                "peaks": peaks.iter().map(|(c, m)| serde_json::json!({"card": c, "peak_mhz": m}))
                    .collect::<Vec<_>>(),
            })
        );
        return;
    }
    let line = peaks
        .iter()
        .map(|(card, mhz)| format!("{card} {mhz}Mhz"))
        .collect::<Vec<_>>()
        .join("  ");
    println!("  gpu peak {line} (while measuring)");
}

/// Curve mode: one generation of `args.curve` tokens, timestamping each streamed
/// token, then reporting the instantaneous decode rate per `args.bucket`-token
/// context window. Measures decode-vs-context scaling directly — no prompt
/// padding, so no slow/VRAM-heavy deep-context prefill. Context position is
/// approximated by the generated-token index (the prompt is short).
fn run_curve(client: &reqwest::blocking::Client, args: &Args) -> anyhow::Result<()> {
    let prompt = build_prompt(0);
    let endpoint = format!("{}/v1/completions", args.url);
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": args.curve,
        "n_predict": args.curve,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
    });
    if let Some(m) = &args.model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    let resp = post_with_one_retry(client, &endpoint, &body)?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned HTTP {}", resp.status());
    }

    // Arrival time of each generated token.
    let mut stamps: Vec<Instant> = Vec::with_capacity(args.curve as usize);
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    loop {
        line.clear();
        // Tolerate a mid-stream read error: end the curve with whatever tokens
        // arrived rather than crashing on a dropped connection.
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let payload = match line.trim_start().strip_prefix("data:") {
            Some(p) => p.trim(),
            None => continue,
        };
        if payload == "[DONE]" {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| v.get("content").and_then(|t| t.as_str()))
            .unwrap_or("");
        if !text.is_empty() {
            stamps.push(Instant::now());
        }
    }

    let n = stamps.len();
    if n < 2 {
        anyhow::bail!("generation produced {n} tokens — need at least 2 for a curve");
    }

    if !args.json {
        println!("curve: {} tokens, bucket {}", n, args.bucket);
        println!("{:>8} | {:>8}", "ctx", "tok/s");
        println!("{}", "-".repeat(19));
    }
    let bucket = args.bucket.max(1) as usize;
    let mut lo = 0usize;
    while lo < n {
        let hi = (lo + bucket).min(n);
        // Rate over the window: tokens produced from the arrival of the token
        // just before `lo` to the arrival of token `hi-1`.
        let (count, dt) = if lo == 0 {
            (hi - 1, (stamps[hi - 1] - stamps[0]).as_secs_f64())
        } else {
            (hi - lo, (stamps[hi - 1] - stamps[lo - 1]).as_secs_f64())
        };
        let rate = if dt > 0.0 { count as f64 / dt } else { 0.0 };
        if args.json {
            println!(
                "{}",
                serde_json::json!({ "ctx": lo, "tok_per_s": rate, "tokens": count })
            );
        } else {
            println!("{:>8} | {:>8.2}", lo, rate);
        }
        lo = hi;
    }
    Ok(())
}

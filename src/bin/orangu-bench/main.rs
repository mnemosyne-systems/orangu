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
//! and every other conformant server** do — and it reports either:
//!
//! - **decode** (the default, and `--curve`): steady-state token-generation
//!   tok/s at one or more context depths, timed from the first streamed token
//!   to the last so prefill and TTFT are excluded — the standard `tg` test.
//! - **prefill** (`--pp`): prompt-processing tok/s, taken from the server's
//!   own `timings` so the token count is exact and a prefix-cache hit is
//!   visible rather than disguised as a fast run — the standard `pp` test.
//!
//! It exists because "how fast is decode, and how does it scale with context?"
//! needs the *same* measurement applied to both engines through the *same*
//! path — not an in-process benchmark compared against an ad-hoc HTTP curl
//! of orangu. This tool is that apples-to-apples harness.
//!
//! This is a **developer tool**, not part of the served product; it is
//! documented only in `doc/manual/en/79-bench.md`.
//!
//! Example:
//! ```text
//! # orangu-server on :8100, sweep decode rate across context depths
//! orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024,2048,3072 --gen 128
//! # another OpenAI-compatible server on :8300, same harness
//! orangu-bench --url http://127.0.0.1:8300 --depths 0,512,1024,2048,3072 --gen 128
//! # prefill throughput at a few prompt lengths
//! orangu-bench --url http://127.0.0.1:8100 --pp 128,512,1024,2048
//! ```

use std::io::{BufRead, BufReader};
use std::path::{Path as FsPath, PathBuf};
use std::time::Instant;

use clap::Parser;

mod bundle;
mod chart;
mod flamegraph;
mod history;
mod moe;
mod points;
mod profile;
mod report;
mod storage;
mod sweep;
mod web;

/// Measure decode (token-generation) throughput of an OpenAI-compatible
/// server over HTTP, at one or more context depths.
#[derive(Parser, Debug)]
#[command(
    name = "orangu-bench",
    version = orangu::build_info::VERSION,
    about = "Measure throughput of an OpenAI-compatible server"
)]
struct Args {
    /// Base URL of the server.
    #[arg(long, default_value = "http://127.0.0.1:8100", value_name = "URL")]
    url: String,

    /// Comma-separated context depths to sweep. Ranges too: `0-2048+512`.
    #[arg(
        long = "depths",
        default_value = "0",
        value_delimiter = ',',
        value_name = "LIST"
    )]
    depths_spec: Vec<String>,

    /// [`Args::depths_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    depths: Vec<u32>,

    /// Prefill mode: prompt lengths to sweep, reporting prompt-processing rate.
    #[arg(long = "pp", value_delimiter = ',', value_name = "LIST")]
    pp_spec: Vec<String>,

    /// [`Args::pp_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    pp: Vec<u32>,

    /// Continuation-prefill mode: comma-separated *added* token counts to sweep.
    #[arg(long = "pp-continue", value_delimiter = ',', value_name = "LIST")]
    pp_continue_spec: Vec<String>,

    /// [`Args::pp_continue_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    pp_continue: Vec<u32>,

    /// Combined mode: prompt lengths prefilled and generated in one request.
    #[arg(long = "pg", value_delimiter = ',', value_name = "LIST")]
    pg_spec: Vec<String>,

    /// [`Args::pg_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    pg: Vec<u32>,

    /// Report the server's CPU time per generated token, with prefill excluded.
    #[arg(long, default_value_t = false)]
    decode_cpu: bool,

    /// Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s.
    #[arg(long = "streams", value_delimiter = ',', value_name = "LIST")]
    streams_spec: Vec<String>,

    /// [`Args::streams_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    streams: Vec<u32>,

    /// Shared-prefix mode: stream counts that all send the SAME long prefix.
    #[arg(long = "shared-prefix", value_delimiter = ',', value_name = "LIST")]
    shared_prefix_spec: Vec<String>,

    /// [`Args::shared_prefix_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    shared_prefix: Vec<u32>,

    /// Length in tokens of the prefix `--shared-prefix` streams have in common.
    #[arg(long, default_value_t = 2048, value_name = "N")]
    shared_prefix_tokens: u32,

    /// Scan-resistance mode: unique prompts to push through between two uses
    /// of one hot prefix.
    #[arg(long = "prefix-scan", value_delimiter = ',', value_name = "LIST")]
    prefix_scan_spec: Vec<String>,

    /// [`Args::prefix_scan_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    prefix_scan: Vec<u32>,

    /// Prompt length (tokens) to prime the prefix cache with for `--pp-continue`.
    #[arg(long, default_value_t = 512, value_name = "N")]
    pp_continue_base: u32,

    /// Embedding mode: prompt lengths to sweep against /v1/embeddings.
    #[arg(long = "embed", value_delimiter = ',', value_name = "LIST")]
    embed_spec: Vec<String>,

    /// [`Args::embed_spec`] expanded — see [`Args::expand_lists`].
    #[arg(skip)]
    embed: Vec<u32>,

    /// Number of tokens to generate per timed run.
    #[arg(long = "gen", default_value_t = 128, value_name = "N")]
    n_gen: u32,

    /// Curve mode: one generation of this many tokens, bucketed by context; 0 disables.
    #[arg(long, default_value_t = 0, value_name = "N")]
    curve: u32,

    /// Bucket width (in context tokens) for `--curve`.
    #[arg(long, default_value_t = 256, value_name = "N")]
    bucket: u32,

    /// Repetitions per depth; the reported rate is the best run with mean±sd.
    #[arg(long, default_value_t = 3, value_name = "N")]
    reps: u32,

    /// Evict the model from the page cache before every repetition.
    #[arg(long, default_value_t = false)]
    drop_model_cache: bool,

    /// Skip the initial warmup run.
    #[arg(long, default_value_t = false)]
    no_warmup: bool,

    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600, value_name = "SECONDS")]
    timeout: u64,

    /// Model id to request.
    #[arg(long, value_name = "ID")]
    model: Option<String>,

    /// Emit machine-readable JSON.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Append each measured point to this tab-separated history file.
    #[arg(long, value_name = "PATH")]
    history: Option<String>,

    /// Series name recorded in the history file (defaults to the server's model); prefixes each `--sweep` point.
    #[arg(long, value_name = "NAME")]
    label: Option<String>,

    /// Render the history file to this SVG after measuring.
    #[arg(long, value_name = "PATH")]
    chart: Option<String>,

    /// Only render the chart from an existing history file; measure nothing.
    #[arg(long, default_value_t = false)]
    chart_only: bool,

    /// Storage mode: comma-separated read request sizes in KiB to sweep.
    #[arg(long, value_name = "LIST")]
    storage_probe: Option<String>,

    /// File the storage probe reads. Defaults to the server's largest shard.
    #[arg(long, value_name = "PATH")]
    storage_file: Option<String>,

    /// MiB to read at each request size, per pass.
    #[arg(long, default_value_t = 256, value_name = "MIB")]
    storage_span: u64,

    /// MiB read and discarded before timing starts, at each size.
    #[arg(long, default_value_t = 32, value_name = "MIB")]
    storage_ramp: u64,

    /// Run each `--sweep` server under this memory cap (e.g. `4G`).
    #[arg(long, value_name = "SIZE")]
    cap: Option<String>,

    /// Also render a PNG beside the chart SVG.
    #[arg(long, default_value_t = false)]
    chart_png: bool,

    /// Pin the chart's tok/s axis to `MIN:MAX` so a pair of charts compare.
    #[arg(long, value_name = "MIN:MAX")]
    chart_scale: Option<String>,

    /// Label for the chart's y-axis.
    #[arg(long, default_value = "tok/s (log)", value_name = "TEXT")]
    chart_y_label: String,

    /// Label for the chart's x-axis.
    #[arg(long, value_name = "TEXT")]
    chart_x_label: Option<String>,

    /// Record a CPU flamegraph of the server over the measured window.
    #[arg(long, value_name = "PATH")]
    flamegraph: Option<String>,

    /// Process to profile (default: the server's own, else the URL port's owner).
    #[arg(long, value_name = "PID")]
    flamegraph_pid: Option<u32>,

    /// Sampling frequency in Hz for `--flamegraph`.
    #[arg(long, default_value_t = 999, value_name = "HZ")]
    flamegraph_freq: u32,

    /// Call-graph mode for `--flamegraph`: `fp` or `dwarf`.
    #[arg(long, default_value = "fp", value_name = "MODE")]
    flamegraph_call_graph: String,

    /// Also render a PNG beside the flamegraph SVG.
    #[arg(long, default_value_t = false)]
    flamegraph_png: bool,

    /// Compare already-collapsed `.folded` profiles side by side; measure nothing.
    #[arg(long, value_delimiter = ',', value_name = "LIST")]
    compare_profiles: Vec<String>,

    /// Write the whole run — measurements, configuration, host — to one JSON file.
    #[arg(long, value_name = "PATH")]
    bundle: Option<String>,

    /// Read bundles and report them side by side; measure nothing.
    #[arg(long, value_delimiter = ',', value_name = "LIST")]
    read_bundle: Vec<String>,

    /// Sweep one tuning variable: `VAR=v1,v2,...` (`;` if a value has a comma); needs `--sweep-cmd`.
    #[arg(long, value_name = "SPEC")]
    sweep: Option<String>,

    /// Shell command that starts the server, run once per `--sweep` value.
    #[arg(long, value_name = "CMD")]
    sweep_cmd: Option<String>,

    /// Environment held constant across every `--sweep` point (repeatable).
    #[arg(long = "sweep-env", value_name = "K=V")]
    sweep_env: Vec<String>,

    /// Seconds to wait for a swept server to come up.
    #[arg(long, default_value_t = 300, value_name = "SECONDS")]
    sweep_start_timeout: u64,

    /// Re-render an already-collapsed `.folded` profile to SVG; measure nothing.
    #[arg(long, value_name = "PATH")]
    render_profile: Option<String>,

    /// Write the run — provenance, measurements, chart, flamegraph — to one PDF.
    #[arg(long, value_name = "PATH")]
    report: Option<String>,

    /// Serve the web console instead of measuring.
    #[arg(long, default_value_t = false)]
    web: bool,

    /// Address the web console binds: "all" (or "*") for every interface.
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    host: String,

    /// Port the web console listens on.
    #[arg(long, default_value_t = 8300, value_name = "PORT")]
    port: u16,

    /// Seconds to wait between measured points, for a card that heats up.
    #[arg(long, default_value_t = 0, value_name = "SECONDS")]
    delay: u64,
}

impl Args {
    /// Turn every list flag's text into the numbers it names, once, before
    /// anything is measured.
    ///
    /// Done here rather than in a `value_parser` because one item can expand
    /// to many — `128-2048*2` is five points from one argument — which clap's
    /// one-value-per-argument parsing cannot express. Failing here also means
    /// a mistyped range costs nothing: the error arrives before the first
    /// request, not twenty minutes into a sweep.
    fn expand_lists(&mut self) -> anyhow::Result<()> {
        for (what, spec, out) in [
            ("--depths", &self.depths_spec, &mut self.depths),
            ("--pp", &self.pp_spec, &mut self.pp),
            ("--pg", &self.pg_spec, &mut self.pg),
            (
                "--pp-continue",
                &self.pp_continue_spec,
                &mut self.pp_continue,
            ),
            ("--embed", &self.embed_spec, &mut self.embed),
            ("--streams", &self.streams_spec, &mut self.streams),
            (
                "--shared-prefix",
                &self.shared_prefix_spec,
                &mut self.shared_prefix,
            ),
            (
                "--prefix-scan",
                &self.prefix_scan_spec,
                &mut self.prefix_scan,
            ),
        ] {
            *out = points::expand_list(spec).map_err(|e| anyhow::anyhow!("{what}: {e}"))?;
        }
        Ok(())
    }

    /// Wait out `--delay` before the next measured point.
    ///
    /// Between points, never before the first or after the last: the delay is
    /// there to let a card cool between measurements, and padding the ends
    /// only makes the run longer. A laptop GPU that heats through a sweep
    /// reports a falling curve that looks exactly like the thing these sweeps
    /// are run to find, which is why this exists at all.
    fn settle(&self) {
        if self.delay > 0 {
            std::thread::sleep(std::time::Duration::from_secs(self.delay));
        }
    }

    /// Evict the model from the server's page cache, if `--drop-model-cache`
    /// asked for it.
    ///
    /// Before *every repetition*, not once per point: the first read of a
    /// weight warms it for every later one, so a point whose cache was
    /// dropped once measures one cold repetition and two warm ones, and
    /// reports the best of them — which is the warm number, under a cold
    /// heading.
    ///
    /// Silent when the server has no such endpoint. That is a real risk worth
    /// naming: `--drop-model-cache` against an engine that ignores it
    /// produces warm numbers labelled cold. The residency figure recorded
    /// alongside the run (see [`moe::residency_line`]) is what catches it,
    /// which is why it is printed and archived rather than merely available.
    fn drop_page_cache(&self, client: &reqwest::blocking::Client) {
        if self.drop_model_cache {
            let _ = moe::drop_cache(client, &self.url);
        }
    }
}

/// POST `body` to `endpoint`, retrying once on a connection-level failure.
///
/// Not defensive programming — a specific, reproduced failure. A server
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

/// What one measured point reports across its repetitions.
///
/// Two standard deviations, on purpose. `sd` divides by `n` (the *population*
/// estimator) and is what `perf-history.tsv`'s `sd` column has held since the
/// file was created — a column in an append-only record means one thing
/// forever, so it keeps that meaning. `sd_sample` divides by `n - 1`, which is
/// the standard estimator and the one every other benchmark reports; without
/// it a `±` figure from this tool could not be put beside a `±` figure from
/// anywhere else. At the default three repetitions the two differ by
/// `sqrt(3/2)` — 22% — which is larger than most of the differences this tool
/// is run to detect.
#[derive(Clone, Copy, Debug)]
struct Stats {
    /// The best of the repetitions: the largest rate, or the smallest where
    /// lower is better (`--decode-cpu` reports milliseconds).
    best: f64,
    mean: f64,
    /// Population standard deviation (÷ n).
    sd: f64,
    /// Sample standard deviation (÷ n-1). `None` for a single repetition,
    /// where it is undefined — reported as `—` rather than as `0.00`, which
    /// would claim a spread was measured and found to be zero.
    sd_sample: Option<f64>,
}

impl Stats {
    fn of(values: &[f64], lower_is_better: bool) -> Stats {
        let n = values.len();
        if n == 0 {
            return Stats {
                best: 0.0,
                mean: 0.0,
                sd: 0.0,
                sd_sample: None,
            };
        }
        let best = if lower_is_better {
            values.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            values.iter().copied().fold(0.0_f64, f64::max)
        };
        let mean = values.iter().sum::<f64>() / n as f64;
        let sum_sq: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
        Stats {
            best,
            mean,
            sd: (sum_sq / n as f64).sqrt(),
            sd_sample: (n > 1).then(|| (sum_sq / (n - 1) as f64).sqrt()),
        }
    }

    /// The `±` figure as printed: the sample standard deviation, or `—` when
    /// one repetition makes it undefined.
    fn plus_minus(&self, width: usize, precision: usize) -> String {
        match self.sd_sample {
            Some(sd) => format!("{sd:>width$.precision$}"),
            None => format!("{:>width$}", "—"),
        }
    }
}

/// One decode measurement: how many tokens streamed, time-to-first-token,
/// and the pure decode window (first→last token).
struct Sample {
    gen_tokens: u32,
    ttft_ms: f64,
    decode_s: f64,
    /// `predicted_n` and `predicted_ms` from the server's own `timings`, when
    /// it reported them. Preferred over the streamed count — see
    /// [`Sample::tok_per_s`].
    reported: Option<(u32, f64)>,
}

impl Sample {
    /// Decode rate, from the server's own accounting when it offered one.
    ///
    /// Counting streamed chunks under-reports, and the error is not small. A
    /// generated token whose text is empty — a special token the server
    /// filters out of the stream, a partial UTF-8 or BPE continuation that
    /// decodes to nothing on its own — costs a full forward pass and arrives
    /// as no visible text, so the chunk loop cannot see it. The elapsed window
    /// still spans it, because the clock keeps running. So the denominator
    /// counts the token and the numerator does not, and the rate comes out low
    /// by whatever share of the generation was invisible.
    ///
    /// Measured on `gemma-4-E2B-it:Q4_K_M` at depth 1024: the server generated
    /// 128 tokens in 2.88 s (44.5 tok/s) and only 79 of them carried text, so
    /// this reported **27.6 tok/s** — a 38% under-read that looked exactly
    /// like a decode cliff between depth 768 and 1024, and was not one.
    ///
    /// `predicted_n` / `predicted_ms` are what the server actually did, and
    /// they already exclude prefill. The streamed count remains the fallback
    /// for a server that reports no timings at all.
    fn tok_per_s(&self) -> f64 {
        if let Some((n, ms)) = self.reported.filter(|&(n, ms)| ms > 0.0 && n > 0) {
            return f64::from(n) / (ms / 1000.0);
        }
        if self.decode_s > 0.0 && self.gen_tokens > 1 {
            (self.gen_tokens - 1) as f64 / self.decode_s
        } else {
            0.0
        }
    }

    /// Tokens generated: the server's count when it gave one.
    fn generated(&self) -> u32 {
        self.reported.map_or(self.gen_tokens, |(n, _)| n)
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
    // Below one whole sentence the repeat-and-pad construction cannot get
    // near the requested length: its preamble and tail alone are ~28 tokens,
    // so `--pp 8`, `--pp 16` and `--pp 32` all came out as the same ~52-token
    // prompt. That is not a rounding error, it is three requested lengths
    // measuring one prefill — and it is why `ORANGU_COOP_MIN_TOKENS`, whose
    // whole regime is forwards of 2..24 positions, read as having no effect
    // anywhere below its default.
    //
    // So short prompts get their own construction. Every length the old one
    // could actually produce (`depth >= words_per`, i.e. one sentence or
    // more) still takes the identical path below and builds a byte-identical
    // prompt, so no previously-recorded `pp` point moves.
    if depth < words_per {
        return short_prompt(sentence, depth);
    }
    let repeats = (depth / words_per).max(1);
    let mut s = String::with_capacity(repeats as usize * sentence.len() + tail.len() + 64);
    s.push_str("Here is the story so far:\n\n");
    for _ in 0..repeats {
        s.push_str(sentence);
    }
    s.push_str(tail);
    s
}

/// A prompt of roughly `depth` tokens, for `depth` below one sentence.
///
/// Just `depth` words of the same sentence, cycled — no preamble and **no
/// "continue, do not stop" tail**. The tail is ~20 tokens on its own, which is
/// most of the budget at these lengths, and it earns its place only in the
/// modes that then *generate*: it stops a greedy model emitting end-of-sequence
/// immediately. A prompt this short is only useful for `--pp`, which measures
/// the prefill and generates a single token, so nothing here depends on what
/// the model would have said next.
///
/// Words rather than a truncated sentence so the text stays ordinary prose —
/// a degenerate repeat is the one thing that provokes the immediate-EOS this
/// whole builder is shaped around, and it would be just as wrong here.
fn short_prompt(sentence: &str, depth: u32) -> String {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut s = String::with_capacity(depth as usize * 8);
    for i in 0..depth.max(1) as usize {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(words[i % words.len()]);
    }
    s
}

/// Build the *added* half of a continuation prompt: text that extends a cached
/// base by roughly `added` tokens and that differs from every other rep's
/// extension in its **first** token.
///
/// Both properties are load-bearing. A prefix cache matches on the longest
/// common token prefix, so if two reps extended the base with the same words,
/// the second rep would find its whole prompt cached and prefill nothing —
/// the measurement would silently become a cache lookup. Varying the opening
/// word per rep keeps the shared prefix at exactly the base.
fn build_continuation(added: u32, rep: usize) -> String {
    // Distinct openers, one per rep, so rep N's extension diverges from rep
    // N-1's immediately rather than sharing a prefix with it.
    const OPENERS: [&str; 8] = [
        "Meanwhile",
        "Afterwards",
        "Elsewhere",
        "Nevertheless",
        "Consequently",
        "Regardless",
        "Furthermore",
        "Eventually",
    ];
    // Word-granular, not sentence-granular: this sweep's whole subject is the
    // 1..64-token range, where a 22-token sentence is the entire axis. Common
    // words are ~1 BPE token each, so word count tracks token count closely
    // enough for a threshold sweep — and `processed` reports the truth anyway.
    const WORDS: [&str; 16] = [
        "the", "river", "narrowed", "between", "cliffs", "and", "crew", "counted", "their",
        "stores", "before", "long", "crossing", "began", "in", "earnest",
    ];
    let mut s = String::with_capacity(added as usize * 8 + 32);
    s.push_str(OPENERS[rep % OPENERS.len()]);
    for i in 0..added.max(1) as usize {
        s.push(' ');
        s.push_str(WORDS[i % WORDS.len()]);
    }
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

    /// Tokens that actually went through a forward pass — the prompt minus
    /// whatever the prefix cache supplied.
    fn processed_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }

    /// Prefill rate for a *continuation*: only the uncached tokens were
    /// forwarded, so they are the only ones `prompt_ms` paid for. Dividing by
    /// the full prompt length instead (what [`tok_per_s`](Self::tok_per_s)
    /// does) would credit the cached prefix with work nobody did.
    fn continuation_tok_per_s(&self) -> f64 {
        if self.prompt_ms > 0.0 {
            self.processed_tokens() as f64 / (self.prompt_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Send one prompt, generate a single token, and report what prefill cost.
///
/// `cache_prompt: false` is what keeps the plain sweep honest: without it the
/// second and later reps would find their prompt already in the server's prefix
/// cache and report the speed of a cache lookup. Both `orangu-server` and
/// other engines honour it; the `cached` column the caller prints is the check
/// that whatever server answered actually did.
///
/// `--pp-continue` passes `true` instead, because a cache hit is precisely what
/// it is trying to produce — there the `cached` column is read as a
/// *requirement* rather than as a warning, and the rate is computed from the
/// uncached remainder.
fn run_prefill_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    model: &Option<String>,
    cache_prompt: bool,
) -> anyhow::Result<PrefillSample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        // One token: enough to force the whole prompt through prefill and to
        // get a final chunk, with as little decode as possible in the way.
        "max_tokens": 1,
        "n_predict": 1,
        "temperature": 0,
        "stream": true,
        // `false` for the plain sweep, where a cache hit would be a lie; `true`
        // only for `--pp-continue`, whose whole subject is the prefill that
        // happens *after* a hit.
        "cache_prompt": cache_prompt,
        // Some servers only attach `timings` to their OpenAI-compatible
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

/// One embedding measurement: how many tokens the server says it embedded,
/// and how long the whole request took.
struct EmbedSample {
    prompt_tokens: u32,
    wall_ms: f64,
    /// `false` when the server sent no `usage.prompt_tokens`. There is then no
    /// token count to divide by — unlike [`PrefillSample`], which can fall
    /// back to a requested length, an embedding response carries no other
    /// clue — so the caller prints the latency and records nothing.
    server_reported: bool,
}

impl EmbedSample {
    fn tok_per_s(&self) -> f64 {
        if self.wall_ms > 0.0 && self.prompt_tokens > 0 {
            self.prompt_tokens as f64 / (self.wall_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Embed one prompt and time the round trip.
///
/// Wall-clock, not a server-reported figure: `/v1/embeddings` has no
/// `timings` object on either server, so unlike [`run_prefill_once`] there is
/// nothing to prefer over the clock. That makes this measurement *inclusive*
/// of HTTP and JSON encoding of an `n_embd`-long float array, which is real
/// but small beside a forward pass — and identical on both engines, since the
/// same client sends both. It is the same trade `doc/perf/embed_bench.sh`
/// made with `curl`; the difference here is that the token count comes from
/// the server rather than from a separate tokenizer run.
///
/// No `stream`: neither server streams embeddings, and there is no first
/// token to time to.
fn run_embed_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    model: &Option<String>,
) -> anyhow::Result<EmbedSample> {
    let mut body = serde_json::json!({"input": prompt});
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    let endpoint = format!("{url}/v1/embeddings");
    let t0 = Instant::now();
    let resp = client
        .post(&endpoint)
        .json(&body)
        .send()
        .map_err(|_| anyhow::anyhow!("Error sending request to url ({endpoint})"))?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned HTTP {}", resp.status());
    }
    let v: serde_json::Value = resp.json()?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // A 200 with no embedding in it is a different failure from a non-200 and
    // would otherwise be reported as a very fast run.
    anyhow::ensure!(
        v.get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("embedding"))
            .and_then(|e| e.as_array())
            .is_some_and(|e| !e.is_empty()),
        "response carried no embedding"
    );

    let prompt_tokens = v
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;

    Ok(EmbedSample {
        prompt_tokens,
        wall_ms,
        server_reported: prompt_tokens > 0,
    })
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
        // The native (non-OpenAI) field name, harmless to servers that
        // ignore it — sending both maximizes cross-server compatibility.
        "n_predict": n_gen,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
        // Generate exactly `n_gen` tokens regardless of content — without this a
        // greedy model handed a depth-padded (repetitive) prompt emits EOS on
        // the first token, so the non-zero-depth rows timed **0** tokens. This
        // is the same "measure decode, not content" contract a depth sweep
        // uses.
        "ignore_eos": true,
    });
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    stream_and_time(client, url, &body)
}

/// Send one streaming completion and time the decode window: from the first
/// streamed token to the last, so prefill and time-to-first-token are excluded.
///
/// Split out of [`run_once`] so `--decode-cpu`'s cache-enabled variant times the
/// window exactly the same way rather than keeping a second copy of the loop.
fn stream_and_time(
    client: &reqwest::blocking::Client,
    url: &str,
    body: &serde_json::Value,
) -> anyhow::Result<Sample> {
    let endpoint = format!("{url}/v1/completions");
    let t0 = Instant::now();
    let resp = post_with_one_retry(client, &endpoint, body)?;
    if !resp.status().is_success() {
        anyhow::bail!("server returned HTTP {}", resp.status());
    }

    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut first: Option<Instant> = None;
    let mut last = t0;
    let mut n: u32 = 0;
    let mut reported: Option<(u32, f64)> = None;

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
        // OpenAI `choices[0].text`, or the native `content`.
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
        // The server's own accounting, carried on the final chunk. Preferred
        // over the count above, which cannot see a generated token that
        // streamed no text — see `Sample::tok_per_s`.
        if let Some(t) = v.get("timings") {
            let n = t.get("predicted_n").and_then(serde_json::Value::as_u64);
            let ms = t.get("predicted_ms").and_then(serde_json::Value::as_f64);
            if let (Some(n), Some(ms)) = (n, ms) {
                reported = Some((n as u32, ms));
            }
        }
    }

    let first = first.unwrap_or(last);
    Ok(Sample {
        gen_tokens: n,
        ttft_ms: (first - t0).as_secs_f64() * 1000.0,
        decode_s: (last - first).as_secs_f64(),
        reported,
    })
}

fn main() {
    let mut args = Args::parse();
    if let Err(e) = args.expand_lists() {
        eprintln!("orangu-bench: {e}");
        std::process::exit(1);
    }
    let args = args;
    if let Err(e) = run(&args) {
        // A single clean line (e.g. a refused connection), not anyhow's
        // multi-line "Error: … Caused by: …" chain.
        eprintln!("orangu-bench: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> anyhow::Result<()> {
    // First of all the modes: the console starts nothing of its own and
    // measures nothing itself — it runs *this* binary, once per benchmark the
    // browser defines, so every other branch below is reachable through it.
    if args.web {
        return web::serve(&args.host, args.port);
    }

    // Chart-only never touches the network, so it works on a history file
    // carried off the machine that produced it — and, more usefully, after a
    // hand-edit of that file, without needing a server up to redraw.
    if args.chart_only {
        return write_chart(args, &[]);
    }

    // Before every server-touching mode: the probe measures the *device*, not
    // the engine, so it neither needs a server nor should be charged for one
    // being up. It does need a real file on the filesystem under test —
    // `--storage-file`, or the largest shard the server reports.
    if let Some(sizes) = &args.storage_probe {
        return storage_probe(args, sizes);
    }

    // Same shape as `--chart-only`: reads artifacts, touches no server.
    if !args.compare_profiles.is_empty() {
        return compare_profiles(args);
    }

    // The other half of `--bundle`: profile there, analyze here.
    if !args.read_bundle.is_empty() {
        return compare_bundles(args);
    }

    // Likewise — and the reason the `.folded` is the artifact worth keeping.
    // Grouped with the other read-only modes above rather than with `--sweep`
    // below, because it touches no server and starts nothing.
    if let Some(folded) = &args.render_profile {
        let folded = std::path::Path::new(folded);
        // Default the output beside the input, so the common case ("give me
        // the PNG for this profile") needs one flag rather than two.
        let svg = match &args.flamegraph {
            Some(s) => std::path::PathBuf::from(s),
            None => folded.with_extension("svg"),
        };
        let (svg, png) = profile::rerender(folded, &svg, args.flamegraph_png, None)?;
        println!("  profile  {}", svg.display());
        if let Some(p) = png {
            println!("           {}", p.display());
        }
        return Ok(());
    }

    // Starts its own servers, so it comes before the client below is pointed
    // at one that is not up yet.
    if args.sweep.is_some() {
        return run_sweep(args);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    // `perf record -p` attaches to the threads that exist at that instant and
    // never picks up ones created later. A server builds its compute threads
    // lazily, on its first request — so profiling with no warmup samples almost
    // nothing while still producing a well-formed flamegraph and a confident
    // `cores_busy`. Measured: 0.02 cores reported against 0.44 actually used.
    //
    // The warmup is what makes those threads exist first, so the two flags are
    // refused together rather than silently producing a bad profile. (The
    // profiler also cross-checks itself against `/proc` and warns, which covers
    // the causes this cannot see.)
    if args.no_warmup && args.flamegraph.is_some() {
        anyhow::bail!(
            "--flamegraph cannot be combined with --no-warmup: `perf record -p` does not \
             follow threads created after it attaches, and a server creates its compute \
             threads on the first request, so the profile would miss nearly all of them"
        );
    }

    // Warmup first (it also validates the connection), so a failure here prints
    // just the clean error above rather than a header followed by an error.
    if !args.no_warmup {
        let p = build_prompt(0);
        // Warm up through the endpoint the run will actually use. An
        // embedding-only server answers `/v1/completions` with HTTP 501, so
        // warming up with a completion would fail the run before it started —
        // which is exactly what kept `--embed`'s models unmeasurable.
        if args.embed.is_empty() {
            run_once(&client, &args.url, &p, 8, &args.model)?;
        } else {
            run_embed_once(&client, &args.url, &p, &args.model)?;
        }
    }

    let label = args
        .label
        .clone()
        .unwrap_or_else(|| server_label(&client, args));

    let env = report_environment(&client, args);

    // Started here — after warmup, after the environment probe — so the profile
    // covers the timed workload and nothing else. Anything before this point is
    // load, allocation and HTTP that the reported rate already excludes, and a
    // flamegraph that included it would attribute time the number does not.
    let recorder = match &args.flamegraph {
        Some(path) => Some(start_profile(&client, args, path, &label)?),
        None => None,
    };

    // Discarded: drains whatever the warmup accumulated, so what the second
    // read returns is the measured window and nothing else.
    let _ = take_gpu_timings(&client, &args.url);
    let clocks = ClockWatch::start();
    let measured = measure(&client, args, &label);
    report_clocks(&clocks.stop(), args);
    let mut env = env;
    env.gpu_timings = take_gpu_timings(&client, &args.url);
    report_gpu_timings(&env.gpu_timings, args);

    // Kept, not just printed: `--report` folds the profile's own numbers
    // (samples, window, cores busy, what it was waiting on) in beside the
    // picture, and they exist nowhere else once this scope ends.
    let mut profile = None;
    if let Some(recorder) = recorder {
        match recorder.finish() {
            Ok(summary) => {
                report_profile(&summary, args);
                profile = Some(summary);
            }
            // The rate is the deliverable; a profile that failed to render is
            // reported and does not discard the measurement that just ran.
            Err(e) => eprintln!("orangu-bench: flamegraph not written: {e}"),
        }
    }

    let mut measured = measured?;
    stamp_device(&mut measured, device_tag(&env.props).as_deref());
    write_bundle(args, args.bundle.as_deref(), &env, &measured)?;
    record_and_chart(args, &measured)?;
    write_report(args, &env, &measured, profile.as_ref())
}

/// Record which device produced these measurements.
///
/// Stamped here, once, rather than threaded through every `run_*` path: the
/// device is a property of the server, and a server cannot change device
/// inside one invocation. A sweep restarts one per point and stamps each
/// point's own — which is the case that needs it, since `ORANGU_DEVICE=0,1`
/// is one invocation measuring two cards.
fn stamp_device(records: &mut [history::Record], device: Option<&str>) {
    for r in records {
        r.device = device.map(str::to_string);
    }
}

/// The short device identity recorded beside every measurement — e.g.
/// `Vulkan/Some Card`, `Vulkan/Some Card (split 2)`, `CPU/AVX2`.
///
/// Short because it has to fit a chart legend and a table column; `props` in
/// the bundle keeps the full name, the driver string, and the whole placement
/// plan. This is an identity, not provenance: the question it answers is "are
/// these two rows the same device or different ones", and the answer has to be
/// readable at a glance beside a number.
///
/// **A split is tagged as one.** `props.backend` on a split model reads
/// `"… + 1 more (2 devices, split)"`, and shortening that as ordinary text
/// would trim it back to the head device's name — which would tag a split run
/// identically to a single-card run on its head card, the exact confusion the
/// column exists to prevent. So the split flag is read from `props.gpu` and
/// spelled out.
///
/// The tag names the *devices*, not the placement: `--device-split 3,1` and
/// `--device-split all` over the same two cards produce the same tag. A sweep
/// over ratios is told apart by its own `VAR=value` label, and the plan itself
/// is in the bundle.
fn device_tag(props: &serde_json::Value) -> Option<String> {
    let backend = props
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|b| !b.is_empty())?;
    let gpu = props.get("gpu").filter(|g| !g.is_null());
    let split = gpu
        .and_then(|g| g.get("split"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if split {
        // Everything before the `" + N more"` the server appends is the head
        // device's own label, which is what the shortener understands.
        let head = short_device(backend.split(" + ").next().unwrap_or(backend));
        let devices = gpu
            .and_then(|g| g.get("devices"))
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        return Some(format!("{head} (split {devices})"));
    }
    Some(short_device(backend))
}

/// `Vulkan/AMD Some Card (DRIVER TAG)` → `Vulkan/Some Card`.
///
/// Two things come off: the driver parenthetical, and a leading vendor word.
/// Neither distinguishes two cards in one machine — which is what this string
/// is for — and both cost the legend width that does. The API prefix stays:
/// the same card under Vulkan and under DX12 is a real A/B, and it is the one
/// pair this would otherwise collapse.
fn short_device(label: &str) -> String {
    let (api, name) = match label.split_once('/') {
        Some((api, name)) => (Some(api), name),
        None => (None, label),
    };
    // Only when something is left: `"(DRIVER TAG)"` as an entire name is
    // unhelpful, but it is better than an empty tag.
    let name = match name.split_once(" (") {
        Some((before, _)) if !before.trim().is_empty() => before.trim(),
        _ => name.trim(),
    };
    let name = ["AMD", "NVIDIA", "Intel(R)", "Intel", "Apple"]
        .iter()
        .find_map(|vendor| name.strip_prefix(vendor).map(str::trim_start))
        .filter(|rest| !rest.is_empty())
        .unwrap_or(name);
    match api {
        Some(api) => format!("{api}/{name}"),
        None => name.to_string(),
    }
}

/// Write `--bundle`, if asked for.
///
/// After the measurement rather than before, because the bundle's whole point
/// is to be the run *and* what produced it in one file — a configuration
/// archived next to no numbers is not evidence of anything.
/// `path` rather than `args.bundle` because a sweep writes one bundle per
/// point, each under its own name — see [`bundle_point_path`].
fn write_bundle(
    args: &Args,
    path: Option<&str>,
    env: &Environment,
    records: &[history::Record],
) -> anyhow::Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    // How the tool was invoked, so a bundle answers "what workload was this"
    // without the reader having to infer it from the record shapes.
    let run = serde_json::json!({
        "url": args.url,
        "workload": workload_name(args),
        "depths": args.depths,
        "pp": args.pp,
        "pg": args.pg,
        "pp_continue": args.pp_continue,
        "embed": args.embed,
        "streams": args.streams,
        "n_gen": args.n_gen,
        "reps": args.reps,
        "curve": args.curve,
    });
    // The host facts the server does not report. `os`/`arch` are what tell a
    // later reader that two bundles came off different machines at all —
    // `props.backend` names the *device*, not the platform, and on a `wgpu`
    // engine those are separate questions.
    let host = serde_json::json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "clocks": env.gpus.iter().map(|g| serde_json::json!({
            "card": g.card, "sclk": g.sclk, "power_level": g.power_level,
        })).collect::<Vec<_>>(),
    });
    bundle::write(
        path,
        &env.props,
        host,
        run,
        &env.gpu_timings,
        &env.model_cache,
        records,
    )?;
    if !args.json {
        println!("  bundle   {path} ({} rows)", records.len());
    }
    Ok(())
}

/// `--report`: the whole run as one PDF — what produced it, what it measured,
/// and the pictures.
///
/// A PDF rather than a markdown table because **the pictures are the point**:
/// a rate quoted without the chart it sits on, or without the profile that
/// explains it, is a number a reader has to take on trust. The chart and the
/// flamegraph are already written as SVG and PNG; this is what puts them in
/// one file with the numbers and the provenance.
///
/// The PNG is what gets embedded (a PDF cannot carry an SVG), so a run that
/// asked for a report renders the raster twin even when `--chart-png` /
/// `--flamegraph-png` were not passed. A machine with no rasterizer says so
/// once and the report is written without that image rather than not at all.
fn write_report(
    args: &Args,
    env: &Environment,
    records: &[history::Record],
    profile: Option<&profile::Summary>,
) -> anyhow::Result<()> {
    let Some(path) = &args.report else {
        return Ok(());
    };
    let blocks = run_report_blocks(&ReportSource {
        provenance: provenance_fields(args, env),
        records,
        reps: args.reps.max(1),
        drifting: env
            .gpus
            .iter()
            .filter(|gpu| gpu.power_level.eq_ignore_ascii_case("auto"))
            .map(|gpu| gpu.card.clone())
            .collect(),
        chart_png: args.chart.as_deref().and_then(raster_twin),
        chart_caption: chart_caption(args),
        profile: profile.map(ProfileFacts::from_summary),
        flamegraph_png: profile.and_then(|s| {
            s.png
                .clone()
                .or_else(|| raster_twin(&s.svg.display().to_string()))
        }),
    });

    let path = std::path::Path::new(path);
    report::write(
        path,
        "orangu-bench report",
        &format!("{} · {}", workload_name(args), history::today()),
        &format!("orangu-bench {}", orangu::build_info::id()),
        &blocks,
    )?;
    if !args.json {
        println!("  report   {}", path.display());
    }
    Ok(())
}

/// Everything a run report is made of, whichever side it came from — a run
/// that just finished, or a bundle read back off disk months later.
///
/// One layout, two sources. A second copy of the page would drift, and the
/// two would disagree about the same run.
struct ReportSource<'a> {
    provenance: Vec<(String, String)>,
    records: &'a [history::Record],
    reps: u32,
    /// Cards measured at `auto`, which invalidates a comparison rather than
    /// qualifying it.
    drifting: Vec<String>,
    chart_png: Option<PathBuf>,
    chart_caption: String,
    profile: Option<ProfileFacts>,
    flamegraph_png: Option<PathBuf>,
}

/// The profile's own numbers, from either a live capture or the `meta.json`
/// written beside a flamegraph.
struct ProfileFacts {
    samples: u64,
    seconds: f64,
    cores_busy: f64,
    gpu_wait: f64,
    pool_idle: f64,
    buckets: Vec<(String, f64)>,
}

impl ProfileFacts {
    fn from_summary(s: &profile::Summary) -> ProfileFacts {
        ProfileFacts {
            samples: s.samples,
            seconds: s.seconds,
            cores_busy: s.cores_busy,
            gpu_wait: s.gpu_wait,
            pool_idle: s.pool_idle,
            buckets: s
                .buckets
                .iter()
                .map(|(name, share)| ((*name).to_string(), *share))
                .collect(),
        }
    }

    /// Read back from `<flamegraph>.meta.json` — the sidecar
    /// `profile::Recorder::finish` leaves precisely so these numbers outlive
    /// the process that measured them. Bucket shares are not in it, so a
    /// reconstructed report carries the totals and the picture without the
    /// self-time table.
    fn from_meta(path: &FsPath) -> Option<ProfileFacts> {
        let text = std::fs::read_to_string(path).ok()?;
        let meta: serde_json::Value = serde_json::from_str(&text).ok()?;
        let num = |key: &str| meta.get(key).and_then(serde_json::Value::as_f64);
        Some(ProfileFacts {
            samples: meta.get("samples").and_then(serde_json::Value::as_u64)?,
            seconds: num("seconds")?,
            cores_busy: num("cores_busy").unwrap_or(0.0),
            gpu_wait: num("gpu_wait_pct").unwrap_or(0.0),
            pool_idle: num("pool_idle_pct").unwrap_or(0.0),
            buckets: Vec::new(),
        })
    }
}

/// The run report's blocks, in the order they are laid out.
fn run_report_blocks(source: &ReportSource) -> Vec<report::Block> {
    let mut blocks = vec![
        report::Block::Heading("What produced it".to_string()),
        report::Block::Fields(source.provenance.clone()),
        report::Block::Heading("What it measured".to_string()),
        report::Block::Table {
            columns: vec![
                report::Column::left("measurement"),
                report::Column::right("n"),
                report::Column::right("best"),
                report::Column::right("mean"),
                report::Column::right("± sd (n-1)"),
                report::Column::left("unit"),
            ],
            rows: source.records.iter().map(record_row).collect(),
        },
        // The headline is the *best* run, not the mean, and a reader comparing
        // this against a number produced elsewhere has to know that: a best is
        // always the flattering one. Said on the page, since the page is what
        // travels.
        report::Block::Note(format!(
            "best of {} repetitions, with the mean and the sample standard deviation \
             (divided by n-1, the standard estimator) beside it. The n column is what the \
             server reported processing, not what was requested.",
            source.reps
        )),
    ];

    // A card left on `auto` idles its clock down between requests, which moves
    // throughput by more than most of the differences this tool is run to
    // detect — and it is invisible in the rate itself.
    if !source.drifting.is_empty() {
        blocks.push(report::Block::Note(format!(
            "Caution: {} was measured at power_dpm_force_performance_level = auto, so its clock \
             was free to idle down between requests. These rates are not comparable with rates \
             taken on a pinned clock.",
            source.drifting.join(", ")
        )));
    }

    if let Some(png) = &source.chart_png {
        blocks.push(report::Block::Heading("Throughput".to_string()));
        blocks.push(report::Block::Image {
            caption: format!("{} — every recorded point", source.chart_caption),
            path: png.clone(),
        });
    }

    if let Some(profile) = &source.profile {
        blocks.push(report::Block::Heading("Where the time went".to_string()));
        blocks.push(report::Block::Fields(vec![
            ("samples".to_string(), profile.samples.to_string()),
            ("window".to_string(), format!("{:.1} s", profile.seconds)),
            (
                "cores busy".to_string(),
                format!("{:.2}", profile.cores_busy),
            ),
            (
                "of which waiting".to_string(),
                format!(
                    "{:.1}% GPU, {:.1}% pool-idle",
                    profile.gpu_wait, profile.pool_idle
                ),
            ),
        ]));
        // A flamegraph is normalised to its own total, so the buckets are the
        // only part of it that survives being quoted on its own.
        if !profile.buckets.is_empty() {
            blocks.push(report::Block::Table {
                columns: vec![
                    report::Column::left("bucket"),
                    report::Column::right("self"),
                ],
                rows: profile
                    .buckets
                    .iter()
                    .map(|(name, share)| vec![name.clone(), format!("{share:.1}%")])
                    .collect(),
            });
        }
    }
    if let Some(png) = &source.flamegraph_png {
        blocks.push(report::Block::Image {
            caption: "CPU profile over the measured window".to_string(),
            path: png.clone(),
        });
    }
    blocks
}

/// The provenance block: everything a reader needs to know whether two
/// reports are comparable, in the order they would ask.
fn provenance_fields(args: &Args, env: &Environment) -> Vec<(String, String)> {
    let text = |key: &str| -> Option<String> {
        env.props
            .get(key)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let mut fields = vec![("url".to_string(), args.url.clone())];
    for (label, key) in [("model", "model"), ("backend", "backend")] {
        if let Some(value) = text(key) {
            fields.push((label.to_string(), value));
        }
    }
    if let Some(build) = server_build(Some(&env.props)) {
        fields.push(("server build".to_string(), build));
    }
    if let Some(pid) = env.props.get("pid").and_then(serde_json::Value::as_u64) {
        let uptime = env
            .props
            .get("uptime_seconds")
            .and_then(serde_json::Value::as_u64);
        fields.push((
            "server process".to_string(),
            match uptime {
                Some(up) => format!("pid {pid}, up {up}s"),
                None => format!("pid {pid}"),
            },
        ));
    }
    for line in format_gpu_tuning(env.props.get("gpu")) {
        // `format_gpu_tuning` returns "label  value" pairs already padded for
        // a terminal; the report's own columns replace that padding.
        if let Some((label, value)) = line.split_once("  ") {
            fields.push((label.trim().to_string(), value.trim().to_string()));
        }
    }
    for gpu in &env.gpus {
        fields.push((
            gpu.card.clone(),
            format!("sclk {} ({})", gpu.sclk, gpu.power_level),
        ));
    }
    fields.push((
        "host".to_string(),
        format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
    ));
    fields.push(("workload".to_string(), workload_detail(args)));
    // To the second, in UTC. Two runs of an A/B are usually the same
    // afternoon, and a date cannot tell them apart.
    fields.push(("measured".to_string(), history::now_utc()));
    fields.push((
        "harness".to_string(),
        format!("orangu-bench {}", orangu::build_info::id()),
    ));
    fields
}

/// One row of the summary table. `cpu` is milliseconds per token and every
/// other mode is tokens per second — the unit column is what stops the two
/// being read as one.
fn record_row(record: &history::Record) -> Vec<String> {
    let unit = if record.mode == "cpu" {
        "ms/token"
    } else {
        "tok/s"
    };
    vec![
        measurement_name(&record.mode).to_string(),
        record.n.to_string(),
        format!("{:.2}", record.best),
        format!("{:.2}", record.mean),
        // The sample estimator, so this figure can be read against one from
        // another benchmark; `—` where a single repetition leaves it
        // undefined. The population figure stays in the bundle and the
        // history file, where the column has always meant that.
        record
            .sd_sample
            .map_or_else(|| "—".to_string(), |sd| format!("{sd:.2}")),
        unit.to_string(),
    ]
}

/// A record's `mode` in words. The bundle and the history file keep the short
/// form (they are data); a report has room to say it.
fn measurement_name(mode: &str) -> &str {
    match mode {
        "tg" => "decode",
        "pp" => "prefill",
        "pg" => "prefill + decode",
        "curve" => "decode @ context",
        "cpu" => "decode CPU",
        "embed" => "embedding",
        other => other,
    }
}

/// The sweep behind the run, spelled out for the provenance block.
fn workload_detail(args: &Args) -> String {
    let list = |values: &[u32]| {
        values
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let points = if !args.pg.is_empty() {
        format!("prompt lengths {}", list(&args.pg))
    } else if !args.pp.is_empty() {
        format!("prompt lengths {}", list(&args.pp))
    } else if !args.pp_continue.is_empty() {
        format!(
            "added tokens {} on a {}-token base",
            list(&args.pp_continue),
            args.pp_continue_base
        )
    } else if !args.embed.is_empty() {
        format!("prompt lengths {}", list(&args.embed))
    } else if !args.streams.is_empty() {
        format!("streams {}", list(&args.streams))
    } else if !args.shared_prefix.is_empty() {
        format!(
            "streams {} sharing a {}-token prefix",
            list(&args.shared_prefix),
            args.shared_prefix_tokens
        )
    } else if !args.prefix_scan.is_empty() {
        format!(
            "{} unique prompts between two uses of a {}-token prefix",
            list(&args.prefix_scan),
            args.shared_prefix_tokens
        )
    } else if args.curve > 0 {
        format!("{} tokens in {}-token buckets", args.curve, args.bucket)
    } else {
        format!("context depths {}", list(&args.depths))
    };
    format!(
        "{} · {points} · {} tokens, best of {}{}",
        workload_name(args),
        args.n_gen,
        args.reps.max(1),
        if args.no_warmup { ", no warmup" } else { "" }
    )
}

fn chart_caption(args: &Args) -> String {
    match &args.history {
        Some(path) => path.to_string(),
        None => "this run".to_string(),
    }
}

/// The `.png` beside an `.svg`, rendering it if it is not there yet.
///
/// A report needs the raster: PDF has no way to carry an SVG. Rendering it
/// here rather than requiring `--chart-png`/`--flamegraph-png` is what makes
/// `--report` a complete instruction on its own.
fn raster_twin(svg: &str) -> Option<std::path::PathBuf> {
    let svg = std::path::Path::new(svg);
    let png = svg.with_extension("png");
    if png.is_file() {
        return Some(png);
    }
    if !svg.is_file() {
        return None;
    }
    profile::render_png(svg).ok().flatten()
}

/// `--sweep`: one server per value of one tuning variable, measured in turn.
///
/// Every point is a full restart, because nearly every knob worth sweeping is
/// read once at device init and most are then baked into generated WGSL —
/// there is no runtime setter that could exist for a workgroup width. See the
/// `sweep` module for why the launching and stopping is more careful than it
/// looks like it needs to be.
///
/// Each point's records go into `--history` under the label `VAR=value` and,
/// with `--bundle`, into its own `<stem>-<value>.json` so the sweep leaves the
/// same artifacts a hand-run A/B would. The comparison table at the end is
/// against the **first** value, so listing the current default first makes the
/// percentages read as "what changing it would buy".
/// `--cap`: run the swept server inside a memory-capped cgroup.
///
/// **The regime this document spends most of its length in is not reachable
/// without one.** A model smaller than RAM is served from the page cache and
/// measures the CPU; the interesting behaviour — streaming, `dense_pass_ratio`,
/// the whole D1 family — only appears once the working set exceeds what the
/// kernel will hold, and on a 62 GiB host that means either a 60 GiB model or
/// a cap. The cap is repeatable and takes seconds.
///
/// `systemd-run --user --scope` rather than a `cgroup` written by hand: it
/// needs no root, cleans the transient scope up when the process exits, and —
/// the part that matters to [`sweep::start`] — **`exec`s into the command**,
/// so the supervised pid is still the server's and not a wrapper's. That is
/// load-bearing: the sweep identifies its server by pid and would otherwise
/// kill a wrapper and leave the server running into the next point.
///
/// `MemorySwapMax=0` is not optional. Without it the kernel satisfies the cap
/// by swapping instead of by evicting page cache, which measures the swap
/// device rather than the model's read path and looks like a much slower disk.
fn capped(cmd: &str, cap: Option<&str>) -> String {
    match cap {
        None => cmd.to_string(),
        Some(cap) => {
            format!("systemd-run --user --scope -q -p MemoryMax={cap} -p MemorySwapMax=0 -- {cmd}")
        }
    }
}

fn run_sweep(args: &Args) -> anyhow::Result<()> {
    let spec = sweep::Spec::parse(args.sweep.as_deref().expect("checked by the caller"))?;
    let cmd = args.sweep_cmd.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--sweep needs --sweep-cmd \"<command that starts orangu-server>\": the values \
             being swept are read once at device init, so each one needs its own server"
        )
    })?;
    let port = url_port(&args.url)
        .ok_or_else(|| anyhow::anyhow!("could not read a port out of --url {}", args.url))?;
    let fixed: Vec<(String, String)> = args
        .sweep_env
        .iter()
        .map(|kv| {
            kv.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .ok_or_else(|| anyhow::anyhow!("--sweep-env wants K=V, got {kv:?}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let timeout = std::time::Duration::from_secs(args.sweep_start_timeout);
    let cmd = capped(cmd, args.cap.as_deref());
    let cmd = cmd.as_str();

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    let mut all = Vec::new();
    let mut bundles = Vec::new();
    for value in &spec.values {
        let label = sweep_point_label(args, &spec, value);
        println!("\n=== {label}");
        // Named after the point, not the run: a sweep that fails at value 3
        // of 5 must leave the log that explains *that* start, not the last
        // one's.
        let log = std::path::PathBuf::from(format!("orangu-sweep-{}.log", slug(&label)));
        let mut env = fixed.clone();
        env.push((spec.var.clone(), value.clone()));
        let server = sweep::start(cmd, &env, port, &log, timeout)?;
        sweep::wait_until_serving(&client, &args.url, &server, timeout)?;

        if !args.no_warmup {
            warm_up_for_sweep(&client, args)?;
        }
        let mut env_report = report_environment(&client, args);
        // Per point, into a path named after the point. A sweep is the one
        // shape where two profiles are *meant* to be compared — that is what
        // a sweep is — and until this existed `--sweep` accepted
        // `--flamegraph` and silently profiled nothing, so the question a
        // sweep raises ("the rate moved; what moved with it?") had no answer
        // from this tool at all. Sharing one path between points would be
        // worse than none: every point would overwrite the last and the file
        // would carry the final configuration under the run's name.
        let recorder = match &args.flamegraph {
            Some(path) => Some(start_profile(
                &client,
                args,
                &profile_path_for_point(path, &label),
                &label,
            )?),
            None => None,
        };
        let _ = take_gpu_timings(&client, &args.url);
        let clocks = ClockWatch::start();
        let measured = measure(&client, args, &label);
        report_clocks(&clocks.stop(), args);
        if let Some(recorder) = recorder {
            match recorder.finish() {
                Ok(summary) => report_profile(&summary, args),
                Err(e) => eprintln!("  profile: {e}"),
            }
        }
        env_report.gpu_timings = take_gpu_timings(&client, &args.url);
        report_gpu_timings(&env_report.gpu_timings, args);
        let mut measured = measured?;
        // Per point, from that point's own server: a sweep of `ORANGU_DEVICE`
        // is one invocation whose points ran on different cards, which is the
        // case the column exists for.
        stamp_device(&mut measured, device_tag(&env_report.props).as_deref());

        if let Some(stem) = &args.bundle {
            let path = bundle_point_path(stem, &label);
            write_bundle(args, Some(&path), &env_report, &measured)?;
            bundles.push(path);
        }
        if let Some(path) = &args.history {
            history::append(path, &measured)?;
        }
        all.extend(measured);
        // Explicit, so the next point's pre-flight port check is not racing
        // this one's teardown.
        drop(server);
    }

    println!(
        "\nswept {} — mean tok/s, against {}",
        spec.var,
        sweep_point_label(args, &spec, &spec.values[0])
    );
    print_sweep_table(args, &spec, &all);
    print_sweep_devices(args, &spec, &all);
    if !bundles.is_empty() {
        println!("\nbundles: {}", bundles.join(" "));
    }
    write_chart(args, &all)
}

/// Warm a freshly-started server on **the workload it is about to be measured
/// on**, discarding the result.
///
/// The single-server path warms up with an eight-token generation, which is
/// enough to make the compute threads exist (which is what `--flamegraph`
/// needs) and *not* enough to warm the kernels a prefill sweep then measures.
/// Measured: a sweep warmed that way put one configuration's first prefill
/// point 16% below the same configuration's later points, with its own sd an
/// order of magnitude above theirs. In a comparison table that is
/// indistinguishable from the swept variable being catastrophic at that value
/// — and because every point of a sweep starts from a cold process, the
/// contamination lands on a different configuration each time, which is worse
/// than a bias that at least cancels.
///
/// Two passes at the *longest* requested prompt: the first is the cold one,
/// the second confirms the path is warm before anything is recorded. Prints
/// nothing — a discarded pass that printed a table would be mistaken for the
/// result.
fn warm_up_for_sweep(client: &reqwest::blocking::Client, args: &Args) -> anyhow::Result<()> {
    let p = build_prompt(0);
    // Always: this is what creates the compute threads, and it is the only
    // warmup an embedding-only server's endpoints allow.
    if args.embed.is_empty() {
        run_once(client, &args.url, &p, 8, &args.model)?;
    } else {
        run_embed_once(client, &args.url, &p, &args.model)?;
    }
    // Then the workload's own path, where there is one to warm.
    let longest = args.pp.iter().chain(args.embed.iter()).copied().max();
    if let Some(depth) = longest {
        let prompt = build_prompt(depth);
        for _ in 0..2 {
            if args.embed.is_empty() {
                // `cache_prompt: false` — a warmup that seeded the prefix
                // cache would make the first measured rep a cache hit and
                // report a lookup as a prefill.
                run_prefill_once(client, &args.url, &prompt, &args.model, false)?;
            } else {
                run_embed_once(client, &args.url, &prompt, &args.model)?;
            }
        }
    }
    Ok(())
}

/// The series name for one swept point: `VAR=value`, or `<label> · VAR=value`
/// when the run carries a `--label`.
///
/// A sweep names its own points, which is right when the sweep *is* the
/// experiment — but it made `--label` do nothing at all, and that turned the
/// one shape a matrix needs into rows that collide. Two models swept over
/// `ORANGU_DEVICE=0,1` wrote four series names between eight measurements: the
/// device column told the cards apart and **nothing told the models apart**,
/// because the label — the only field that could have — was being discarded.
/// The same discard also aimed both models' `--bundle` points at one filename.
///
/// Prefix rather than replace: inside one sweep the `VAR=value` half is still
/// what distinguishes a point, and a `--label` that overwrote it would trade
/// one collision for another.
fn sweep_point_label(args: &Args, spec: &sweep::Spec, value: &str) -> String {
    match &args.label {
        Some(prefix) => format!("{prefix} · {}", spec.label(value)),
        None => spec.label(value),
    }
}

/// `ORANGU_COOP_MIN_TOKENS=16` → `orangu_coop_min_tokens-16`, for a filename.
/// Where one sweep point's flamegraph goes: the `--flamegraph` path with the
/// point's own label folded into the file stem.
///
/// `perf/pp.svg` at point `ORANGU_NO_CHUNK_COST_FIT=0` becomes
/// `perf/pp-orangu-no-chunk-cost-fit-0.svg`, and the `.folded`/`.png`/
/// `.meta.json` siblings follow it, because [`profile::Recorder`] derives them
/// from the stem it is given.
fn profile_path_for_point(svg: &str, label: &str) -> String {
    let path = std::path::Path::new(svg);
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned());
    let Some(stem) = stem else {
        return svg.to_string();
    };
    let named = match ext {
        Some(ext) => format!("{stem}-{}.{ext}", slug(label)),
        None => format!("{stem}-{}", slug(label)),
    };
    path.parent()
        .map(|dir| dir.join(&named).to_string_lossy().into_owned())
        .unwrap_or(named)
}

fn slug(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// `perf/coop.json` + `VAR=16` → `perf/coop-var-16.json`, so a sweep's points
/// sit next to each other and sort in value order rather than overwriting one
/// file per point.
fn bundle_point_path(stem: &str, label: &str) -> String {
    let p = std::path::Path::new(stem);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("json");
    let base = p.with_extension("");
    format!("{}-{}.{ext}", base.display(), slug(label))
}

/// The sweep's answer, as one row per measured point and one column per value.
fn print_sweep_table(args: &Args, spec: &sweep::Spec, records: &[history::Record]) {
    let mut points: Vec<(String, u32)> = records.iter().map(|r| (r.mode.clone(), r.n)).collect();
    points.sort_unstable();
    points.dedup();
    let at = |value: &str, mode: &str, n: u32| -> Option<f64> {
        let label = sweep_point_label(args, spec, value);
        records
            .iter()
            .find(|r| r.label == label && r.mode == mode && r.n == n)
            .map(|r| r.mean)
    };
    print!("  {:<12}", "point");
    for v in &spec.values {
        print!("  {:>18}", if v.is_empty() { "<unset>" } else { v });
    }
    println!();
    for (mode, n) in &points {
        print!("  {:<12}", format!("{mode} {n}"));
        let base = at(&spec.values[0], mode, *n);
        for v in &spec.values {
            match (at(v, mode, *n), base) {
                (Some(got), Some(b)) if b > 0.0 => {
                    print!("  {got:>11.2} {:>+5.1}%", (got / b - 1.0) * 100.0);
                }
                (Some(got), _) => print!("  {got:>18.2}"),
                (None, _) => print!("  {:>18}", "—"),
            }
        }
        println!();
    }
}

/// Which device each swept point actually ran on — printed only when they
/// differ.
///
/// `--sweep ORANGU_DEVICE=0,1` heads its columns `0` and `1`, which are the
/// indices asked for and not the cards that answered: whether index 1 is the
/// discrete card or the integrated one is the whole content of the comparison,
/// and it is knowable only from the server that came up. Silent when every
/// point ran on the same device, which is every sweep of an ordinary tuning
/// knob — a column earns its width only when it varies, which is also why an
/// existing sweep's output is unchanged.
fn print_sweep_devices(args: &Args, spec: &sweep::Spec, records: &[history::Record]) {
    if !history::devices_differ(records) {
        return;
    }
    for value in &spec.values {
        let label = sweep_point_label(args, spec, value);
        let device = records
            .iter()
            .find(|r| r.label == label)
            .and_then(|r| r.device.as_deref())
            .unwrap_or("?");
        println!("  {label:<24} {device}");
    }
}

/// `--read-bundle`: put runs side by side and say what differed between them.
///
/// Two tables, in this order on purpose. **What differed** comes first, because
/// a throughput comparison is only readable once you know whether the two runs
/// were the same experiment — and across two machines the honest answer is
/// usually "no, and here is the list". **What it measured** comes second, with
/// the ratio against the first bundle, which is the number the run was for.
///
/// Renders `--chart`/`--chart-png` from the bundles' own records when asked, so
/// the same command produces the picture that goes in the document.
fn compare_bundles(args: &Args) -> anyhow::Result<()> {
    let bundles: Vec<bundle::Bundle> = args
        .read_bundle
        .iter()
        .map(|p| bundle::read(p))
        .collect::<anyhow::Result<_>>()?;
    let (first, rest) = bundles
        .split_first()
        .expect("caller checked --read-bundle is non-empty");

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "bundles",
                "bundles": bundles.iter().map(|b| serde_json::json!({
                    "name": b.name,
                    "date": b.date,
                    "label": b.label(),
                    "props": b.props,
                    "host": b.host,
                    "run": b.run,
                })).collect::<Vec<_>>(),
                "diffs": rest.iter().map(|b| serde_json::json!({
                    "against": first.name,
                    "name": b.name,
                    "fields": bundle::diff(first, b).into_iter()
                        .map(|(k, l, r)| serde_json::json!([k, l, r]))
                        .collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        for b in &bundles {
            println!("{}  {}  {}", b.name, b.date, b.label());
        }
        for b in rest {
            let fields = bundle::diff(first, b);
            println!("\nwhat differed: {} → {}", first.name, b.name);
            if fields.is_empty() {
                // Worth saying out loud: "same configuration" is a finding,
                // and an empty table would read as a broken diff.
                println!("  (nothing — same configuration)");
            }
            let width = fields.iter().map(|(k, _, _)| k.len()).max().unwrap_or(0);
            for (k, l, r) in &fields {
                println!("  {k:<width$}  {l}  →  {r}");
            }
        }
        // Before the throughput table, because it is the half that answers
        // *why* — and on a Mac it is the only such answer available, since
        // `--flamegraph` needs `perf`.
        let with_timings: Vec<&bundle::Bundle> = bundles
            .iter()
            .filter(|b| !b.gpu_timings.is_null())
            .collect();
        if !with_timings.is_empty() {
            println!("\nwhere the GPU time went (ms per decode step)");
            let stages = ["total", "qkv", "attn", "ffn", "ple", "tail"];
            print!("  {:<8}", "stage");
            for b in &with_timings {
                print!("  {:>18}", b.name);
            }
            println!();
            let at = |b: &bundle::Bundle, stage: &str| {
                b.gpu_timings
                    .get("per_step_ms")
                    .and_then(|m| m.get(stage))
                    .and_then(serde_json::Value::as_f64)
            };
            for stage in stages {
                print!("  {stage:<8}");
                let base = at(with_timings[0], stage);
                for b in &with_timings {
                    match (at(b, stage), base) {
                        (Some(v), Some(b0)) if b0 > 0.0 => {
                            print!("  {v:>11.3} {:>+5.1}%", (v / b0 - 1.0) * 100.0);
                        }
                        (Some(v), _) => print!("  {v:>18.3}"),
                        (None, _) => print!("  {:>18}", "—"),
                    }
                }
                println!();
            }
        }

        println!("\nwhat it measured (mean tok/s)");
        let mut points: Vec<(String, u32)> = bundles
            .iter()
            .flat_map(|b| b.by_point().into_keys())
            .collect();
        points.sort_unstable();
        points.dedup();
        let per_bundle: Vec<_> = bundles.iter().map(bundle::Bundle::by_point).collect();
        print!("  {:<10}", "point");
        for b in &bundles {
            print!("  {:>18}", b.name);
        }
        println!();
        for (mode, n) in &points {
            print!("  {:<10}", format!("{mode} {n}"));
            let base = per_bundle[0].get(&(mode.clone(), *n)).copied();
            for got in &per_bundle {
                match got.get(&(mode.clone(), *n)) {
                    // The ratio, not just the rate: the comparison is the
                    // reason two bundles are open at once.
                    Some(v) => match base {
                        Some(b0) if b0 > 0.0 => {
                            print!("  {v:>11.2} {:>+5.1}%", (v / b0 - 1.0) * 100.0)
                        }
                        _ => print!("  {v:>18.2}"),
                    },
                    None => print!("  {:>18}", "—"),
                }
            }
            println!();
        }
    }

    // The records of every bundle, so one chart holds every machine.
    let records: Vec<history::Record> = bundles.iter().flat_map(|b| b.records.clone()).collect();
    write_chart(args, &records)?;
    write_comparison_report(args, &bundles)
}

/// `--read-bundle one.json --report out.pdf`: a run's report, rebuilt from
/// what the run archived.
///
/// Everything the live path knows is in the bundle — `props`, `host`, the
/// records, the workload — except the pictures, which are files. Those are
/// looked for **beside the bundle**, because that is how a run's directory is
/// laid out: `bundle.json`, `chart.svg`/`.png`, `flamegraph.svg`/`.png` and
/// the profile's `meta.json` in one place. A missing picture leaves a line in
/// the document rather than failing it.
fn write_report_from_bundle(
    args: &Args,
    bundle: &bundle::Bundle,
    path: &FsPath,
) -> anyhow::Result<()> {
    let dir = args
        .read_bundle
        .first()
        .map(|p| {
            FsPath::new(p)
                .parent()
                .unwrap_or(FsPath::new("."))
                .to_path_buf()
        })
        .unwrap_or_else(|| FsPath::new(".").to_path_buf());
    let beside = |name: &str| -> Option<PathBuf> {
        let png = dir.join(name).with_extension("png");
        if png.is_file() {
            return Some(png);
        }
        // Not rendered yet: the SVG is the canonical artifact, so make its
        // raster twin now rather than leaving a hole in the document.
        raster_twin(&dir.join(name).with_extension("svg").display().to_string())
    };

    let text = |key: &str| -> Option<String> {
        bundle
            .props
            .get(key)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let mut provenance = Vec::new();
    if let Some(url) = bundle.run.get("url").and_then(|v| v.as_str()) {
        provenance.push(("url".to_string(), url.to_string()));
    }
    for (label, key) in [("model", "model"), ("backend", "backend")] {
        if let Some(value) = text(key) {
            provenance.push((label.to_string(), value));
        }
    }
    if let Some(build) = server_build(Some(&bundle.props)) {
        provenance.push(("server build".to_string(), build));
    }
    for line in format_gpu_tuning(bundle.props.get("gpu")) {
        if let Some((label, value)) = line.split_once("  ") {
            provenance.push((label.trim().to_string(), value.trim().to_string()));
        }
    }
    // The clocks the *run* saw, from its own host block — not this machine's,
    // which is very likely a different one.
    let clocks = bundle.host.get("clocks").and_then(|c| c.as_array());
    for gpu in clocks.into_iter().flatten() {
        let field = |key: &str| gpu.get(key).and_then(|v| v.as_str()).unwrap_or("?");
        provenance.push((
            field("card").to_string(),
            format!("sclk {} ({})", field("sclk"), field("power_level")),
        ));
    }
    if let (Some(os), Some(arch)) = (
        bundle.host.get("os").and_then(|v| v.as_str()),
        bundle.host.get("arch").and_then(|v| v.as_str()),
    ) {
        provenance.push(("host".to_string(), format!("{os}/{arch}")));
    }
    if let Some(workload) = bundle.run.get("workload").and_then(|v| v.as_str()) {
        provenance.push(("workload".to_string(), workload.to_string()));
    }
    provenance.push((
        "measured".to_string(),
        // The instant the bundle recorded; its date only for one written
        // before the field existed.
        bundle
            .measured_at
            .clone()
            .unwrap_or_else(|| bundle.date.clone()),
    ));
    provenance.push((
        "harness".to_string(),
        // The build that *measured* it, not the one drawing this page.
        bundle
            .tool
            .clone()
            .unwrap_or_else(|| format!("orangu-bench {}", orangu::build_info::id())),
    ));

    let drifting: Vec<String> = clocks
        .into_iter()
        .flatten()
        .filter(|gpu| {
            gpu.get("power_level")
                .and_then(|v| v.as_str())
                .is_some_and(|level| level.eq_ignore_ascii_case("auto"))
        })
        .filter_map(|gpu| gpu.get("card").and_then(|v| v.as_str()).map(str::to_string))
        .collect();

    let blocks = run_report_blocks(&ReportSource {
        provenance,
        records: &bundle.records,
        reps: bundle
            .run
            .get("reps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as u32,
        drifting,
        chart_png: beside("chart"),
        chart_caption: "this run".to_string(),
        profile: ProfileFacts::from_meta(&dir.join("flamegraph.meta.json")),
        flamegraph_png: beside("flamegraph"),
    });

    report::write(
        path,
        "orangu-bench report",
        &format!(
            "{} · {}",
            bundle
                .run
                .get("workload")
                .and_then(|v| v.as_str())
                .unwrap_or(bundle.label()),
            bundle.date
        ),
        &format!("orangu-bench {}", orangu::build_info::id()),
        &blocks,
    )?;
    if !args.json {
        println!("  report   {}", path.display());
    }
    Ok(())
}

/// `--read-bundle A,B --report out.pdf`: the comparison as a document.
///
/// The same two tables the terminal prints — what differed between the
/// configurations, then what each measured with the ratio — plus the chart
/// holding both runs. Written here rather than left to the measuring path
/// because a comparison is the thing most likely to be *sent to someone*, and
/// a `--report` that was silently ignored on this route would be worse than
/// one that did not exist.
fn write_comparison_report(args: &Args, bundles: &[bundle::Bundle]) -> anyhow::Result<()> {
    let Some(path) = &args.report else {
        return Ok(());
    };
    let Some((first, rest)) = bundles.split_first() else {
        return Ok(());
    };
    // One bundle is not a comparison — it is a run, read back. Reporting it as
    // "what was compared: one thing" would be a document about nothing, so it
    // takes the run-report path instead. This is what makes a report producible
    // *after* the fact, from what a run archived, rather than only at the
    // moment it finished.
    if rest.is_empty() {
        return write_report_from_bundle(args, first, std::path::Path::new(path));
    }

    let mut blocks = vec![report::Block::Heading("What was compared".to_string())];
    blocks.push(report::Block::Fields(
        bundles
            .iter()
            .map(|b| (b.name.clone(), format!("{} · {}", b.label(), b.date)))
            .collect(),
    ));

    for b in rest {
        let fields = bundle::diff(first, b);
        blocks.push(report::Block::Heading(format!(
            "What differed: {} → {}",
            first.name, b.name
        )));
        if fields.is_empty() {
            // Worth saying out loud, exactly as the terminal does: "same
            // configuration" is a finding, and an empty space is not.
            blocks.push(report::Block::Note(
                "Nothing — same configuration.".to_string(),
            ));
        } else {
            blocks.push(report::Block::Table {
                columns: vec![
                    report::Column::left("field"),
                    report::Column::left(&first.name),
                    report::Column::left(&b.name),
                ],
                rows: fields
                    .into_iter()
                    .map(|(key, left, right)| vec![key, left, right])
                    .collect(),
            });
        }
    }

    // One row per measured point, one column per bundle, and the ratio against
    // the first — the number the comparison was run for.
    let per_bundle: Vec<_> = bundles.iter().map(bundle::Bundle::by_point).collect();
    let mut points: Vec<(String, u32)> = per_bundle
        .iter()
        .flat_map(|b| b.keys().cloned())
        .collect::<Vec<_>>();
    points.sort_unstable();
    points.dedup();
    let mut columns = vec![
        report::Column::left("measurement"),
        report::Column::right("n"),
    ];
    for b in bundles {
        columns.push(report::Column::right(&b.name));
    }
    let rows = points
        .iter()
        .map(|(mode, n)| {
            let mut row = vec![measurement_name(mode).to_string(), n.to_string()];
            let base = per_bundle[0].get(&(mode.clone(), *n)).copied();
            for got in &per_bundle {
                row.push(match (got.get(&(mode.clone(), *n)), base) {
                    (Some(v), Some(b0)) if b0 > 0.0 => {
                        format!("{v:.2}  {:+.1}%", (v / b0 - 1.0) * 100.0)
                    }
                    (Some(v), _) => format!("{v:.2}"),
                    (None, _) => "—".to_string(),
                });
            }
            row
        })
        .collect();
    blocks.push(report::Block::Heading("What it measured".to_string()));
    blocks.push(report::Block::Table { columns, rows });
    blocks.push(report::Block::Note(format!(
        "Mean of each run's repetitions, as a percentage against {}.",
        first.name
    )));

    if let Some(chart) = &args.chart
        && let Some(png) = raster_twin(chart)
    {
        blocks.push(report::Block::Heading("Throughput".to_string()));
        blocks.push(report::Block::Image {
            caption: "Both runs on one chart".to_string(),
            path: png,
        });
    }

    let path = std::path::Path::new(path);
    report::write(
        path,
        "orangu-bench comparison",
        &format!(
            "{} against {} · {}",
            rest.iter()
                .map(|b| b.name.clone())
                .collect::<Vec<_>>()
                .join(", "),
            first.name,
            history::today()
        ),
        &format!("orangu-bench {}", orangu::build_info::id()),
        &blocks,
    )?;
    if !args.json {
        println!("  report   {}", path.display());
    }
    Ok(())
}

/// The timed part of a run, split out so [`run`] can bracket exactly this with
/// the profiler.
fn measure(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    // Before `--pp`, and exclusive of it: an embedding-only server can serve
    // nothing else, and on a generative one the two measure different models'
    // worth of work, so combining them in a single sweep would produce two
    // unrelated tables under one header.
    if !args.embed.is_empty() {
        return run_embed(client, args, label);
    }

    if !args.pg.is_empty() {
        return run_pg(client, args, label);
    }

    if !args.pp.is_empty() {
        return run_pp(client, args, label);
    }

    if !args.pp_continue.is_empty() {
        return run_pp_continue(client, args, label);
    }

    if args.decode_cpu {
        return run_decode_cpu(client, args, label);
    }

    if !args.streams.is_empty() {
        return run_streams(client, args, label);
    }

    if !args.shared_prefix.is_empty() {
        return run_shared_prefix(client, args, label);
    }

    if !args.prefix_scan.is_empty() {
        return run_prefix_scan(client, args, label);
    }

    if args.curve > 0 {
        return run_curve(client, args, label);
    }

    run_tg(client, args, label)
}

/// One combined measurement: the whole turn, prompt through last token.
struct PgSample {
    /// What the server said it processed, or the requested length when it
    /// reported no `timings`.
    prompt_tokens: u32,
    /// Counted off the stream, not requested — a server that stopped early
    /// must not be credited with tokens it never sent.
    gen_tokens: u32,
    /// Prefill wall time, from the server's own `timings`, for the split
    /// column. `None` when it reports none.
    prompt_ms: Option<f64>,
    /// Send to last token: prefill, queueing and generation together.
    total_s: f64,
}

impl PgSample {
    /// `(prompt + generated) / total` — the whole turn over the whole time,
    /// which is the quantity a combined test exists to report.
    fn tok_per_s(&self) -> f64 {
        if self.total_s > 0.0 {
            f64::from(self.prompt_tokens + self.gen_tokens) / self.total_s
        } else {
            0.0
        }
    }
}

/// `--pg`: prefill *and* generate in one request, timed as one thing.
///
/// Every other mode here splits the turn deliberately — `--pp` times prefill
/// with a single token generated, the decode sweep times generation with
/// prefill excluded — because that is what a diagnosis needs. This mode is the
/// opposite and answers the other question: what a user actually waits for. It
/// is also the figure most third-party comparisons quote, and it could not be
/// reconstructed from the two halves, since neither carries the queueing and
/// hand-off between them.
///
/// The generated length is `--gen`, so the swept axis is the prompt — matching
/// `--pp`, and keeping one meaning for `n` per mode in the history file.
fn run_pg(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>7} | {:>9} | {:>9} | {:>8} | {:>16}",
            "pp", "n_tok", "gen", "prompt_ms", "total_ms", "best", "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(81));
    }

    for (point, &len) in args.pg.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<PgSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_pg_once(client, &args.url, &prompt, len, args.n_gen, &args.model)?;
            rates.push(s.tok_per_s());
            last = Some(s);
        }
        let stats = Stats::of(&rates, false);
        let s = last.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "pg": len,
                    "prompt_tokens": s.prompt_tokens,
                    "gen_tokens": s.gen_tokens,
                    "prompt_ms": s.prompt_ms,
                    "total_ms": s.total_s * 1000.0,
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    "tok_per_s_sd": stats.sd,
                    "tok_per_s_sd_sample": stats.sd_sample,
                })
            );
        } else {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9} | {:>9.1} | {:>8.2} | {:>8.2} ± {}",
                len,
                s.prompt_tokens,
                s.gen_tokens,
                s.prompt_ms
                    .map_or_else(|| "—".to_string(), |ms| format!("{ms:.1}")),
                s.total_s * 1000.0,
                stats.best,
                stats.mean,
                stats.plus_minus(5, 2)
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "pg".to_string(),
            // The prompt the server says it processed, as `--pp` records —
            // the requested length is a target, the processed count is what
            // the rate was computed from.
            n: s.prompt_tokens,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }

    Ok(records)
}

/// One combined request: prompt of roughly `len` tokens, `n_gen` generated,
/// timed from before the send to the last streamed token.
///
/// `cache_prompt: false` for the same reason the plain prefill sweep sets it:
/// a cached prompt would make the prefill half of this number vanish, and the
/// combined figure would quietly become a decode figure.
fn run_pg_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    requested: u32,
    n_gen: u32,
    model: &Option<String>,
) -> anyhow::Result<PgSample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": n_gen,
        "n_predict": n_gen,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
        "ignore_eos": true,
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
    let mut last = t0;
    let mut gen_tokens = 0u32;
    let mut prompt_tokens = None;
    let mut prompt_ms = None;

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
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| v.get("content").and_then(|t| t.as_str()))
            .unwrap_or("");
        if !text.is_empty() {
            last = Instant::now();
            gen_tokens += 1;
        }
        if let Some(t) = v.get("timings") {
            if let Some(n) = t.get("prompt_n").and_then(serde_json::Value::as_u64) {
                prompt_tokens = Some(n as u32);
            }
            if let Some(ms) = t.get("prompt_ms").and_then(serde_json::Value::as_f64) {
                prompt_ms = Some(ms);
            }
        }
    }

    Ok(PgSample {
        // The requested length only when the server reported nothing — an
        // approximation, and the row's `n_tok` column shows which it is.
        prompt_tokens: prompt_tokens.unwrap_or(requested),
        gen_tokens,
        prompt_ms,
        total_s: (last - t0).as_secs_f64(),
    })
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
            "depth", "gen", "ttft_ms", "n_tok", "best", "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(67));
    }

    for (point, &depth) in args.depths.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let prompt = build_prompt(depth);
        let mut rates = Vec::new();
        let mut last_sample: Option<Sample> = None;
        // Discarded, for the reason `run_pp` documents.
        let _ = moe::take_stats(client, &args.url);
        for _ in 0..args.reps.max(1) {
            args.drop_page_cache(client);
            let s = run_once(client, &args.url, &prompt, args.n_gen, &args.model)?;
            rates.push(s.tok_per_s());
            last_sample = Some(s);
        }
        let moe_stats = moe::take_stats(client, &args.url);
        let stats = Stats::of(&rates, false);
        let s = last_sample.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "depth": depth,
                    "n_gen": args.n_gen,
                    "ttft_ms": s.ttft_ms,
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    // Unchanged meaning: the population estimator, as every
                    // consumer of this stream has always read it.
                    "tok_per_s_sd": stats.sd,
                    // The standard estimator, for putting this number beside
                    // one from another benchmark.
                    "tok_per_s_sd_sample": stats.sd_sample,
                    "gen_tokens": s.generated(),
                })
            );
        } else {
            println!(
                "{:>8} | {:>5} | {:>7.0} | {:>8} | {:>8.2} | {:>8.2} ± {}",
                depth,
                args.n_gen,
                s.ttft_ms,
                s.generated(),
                stats.best,
                stats.mean,
                stats.plus_minus(5, 2)
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "tg".to_string(),
            n: depth,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
        records.extend(moe::records(&moe_stats, label, depth, args.reps.max(1)));
        // Every repetition in the window generated `--gen` tokens, and the
        // window is bracketed by the two `take_stats` calls above, so this is
        // what the disk was asked for per token of output.
        let generated = f64::from(args.n_gen) * f64::from(args.reps.max(1));
        records.extend(moe::io_records(
            &moe_stats,
            label,
            depth,
            generated,
            args.reps.max(1),
        ));
        if !args.json {
            if let Some(line) = moe::summary_line(&moe_stats) {
                println!("{line}");
            }
            if let Some(line) = moe::io_line(&moe_stats, generated) {
                println!("{line}");
            }
        }
    }

    Ok(records)
}

/// Append this run's points to `--history` (when given) and redraw `--chart`
/// (when given) from the file *including* them.
///
/// Drawing from the file rather than from `records` is what makes the chart a
/// history rather than a snapshot: a run that measured only orangu still
/// redraws the reference engine's line beside it.
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

/// `--storage-probe`: the block-size curve `[orangu-server].read_size` is set
/// from, as one command.
///
/// Touches no server for the measurement itself — the subject is the device.
/// It will *ask* a server which file to read, because the interesting file is
/// the one the engine actually streams and nobody wants to type a shard path;
/// `--storage-file` skips that and works with nothing running.
fn storage_probe(args: &Args, sizes: &str) -> anyhow::Result<()> {
    let sizes: Vec<u32> = sizes
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|e| anyhow::anyhow!("--storage-probe wants KiB sizes, got {s:?}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;
    if sizes.is_empty() {
        anyhow::bail!("--storage-probe needs at least one request size in KiB");
    }

    let path = match &args.storage_file {
        Some(p) => std::path::PathBuf::from(p),
        None => largest_shard(args)?,
    };
    // Named before the probe runs, not after: the sweep takes minutes and the
    // one thing that invalidates all of it is reading the wrong device.
    if !args.json {
        println!("  probe    {}", path.display());
        println!(
            "  sweep    {} sizes x 2 passes x {} MiB, {} MiB ramp",
            sizes.len(),
            args.storage_span,
            args.storage_ramp
        );
    }

    let points = storage::sweep(
        &path,
        &sizes,
        args.storage_span * 1024 * 1024,
        args.storage_ramp * 1024 * 1024,
    )
    .map_err(|e| anyhow::anyhow!("storage probe on {}: {e}", path.display()))?;

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "file": path.display().to_string(),
                "points": points.iter().map(|p| serde_json::json!({
                    "kib": p.kib, "mb_s": p.mean(), "up": p.up, "down": p.down,
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        print!("{}", storage::table(&points));
    }

    let label = args
        .label
        .clone()
        .unwrap_or_else(|| device_label(&path).unwrap_or_else(|| "storage".to_string()));
    record_and_chart(args, &storage::records(&points, &label))
}

/// The biggest file backing the running model, from `/model-cache`.
///
/// The largest shard rather than the first: a probe wants a file long enough
/// to read hundreds of MiB out of without wrapping, and on a sharded model
/// the small shards are not that.
fn largest_shard(args: &Args) -> anyhow::Result<std::path::PathBuf> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let residency = moe::take_residency(&client, &args.url);
    let best = residency
        .get("shards")
        .and_then(|s| s.as_array())
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some((
                s.get("bytes")?.as_u64()?,
                s.get("path")?.as_str()?.to_string(),
            ))
        })
        .max_by_key(|(bytes, _)| *bytes);
    match best {
        Some((_, path)) => Ok(std::path::PathBuf::from(path)),
        // The probe is useful with no server at all, so this is a nudge to the
        // flag that makes it so rather than a hard failure about the server.
        None => anyhow::bail!(
            "no server at {} to name a model file — pass --storage-file <PATH> to probe a \
             file directly",
            args.url
        ),
    }
}

/// The block device a path lives on, for the history `label` — so two drives
/// in one machine draw two lines instead of overwriting each other.
///
/// `None` when it cannot be worked out, which is not worth failing over: the
/// caller falls back to a generic label and the curve is still recorded.
fn device_label(path: &std::path::Path) -> Option<String> {
    let mount = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let canonical = path.canonicalize().ok()?;
    // The longest mount point that is a prefix of the file's path is the
    // filesystem it is on — `/mnt/ai` beats `/` for a file under `/mnt/ai`.
    mount
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(' ');
            let point = fields.nth(4)?;
            let source = line.rsplit(' ').nth(1)?;
            canonical
                .starts_with(point)
                .then(|| (point.len(), source.to_string()))
        })
        .max_by_key(|(len, _)| *len)
        .map(|(_, source)| source.rsplit('/').next().unwrap_or(&source).to_string())
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
    // A pinned axis is what makes two separately-rendered charts comparable;
    // see `chart::render_with_scale`. Parsed here rather than in the renderer so
    // a typo is a clean error before anything is drawn.
    let scale = match &args.chart_scale {
        Some(spec) => {
            let (lo, hi) = spec
                .split_once(':')
                .ok_or_else(|| anyhow::anyhow!("--chart-scale wants MIN:MAX, got {spec:?}"))?;
            Some((
                lo.trim()
                    .parse::<f64>()
                    .map_err(|e| anyhow::anyhow!("--chart-scale MIN: {e}"))?,
                hi.trim()
                    .parse::<f64>()
                    .map_err(|e| anyhow::anyhow!("--chart-scale MAX: {e}"))?,
            ))
        }
        None => None,
    };
    std::fs::write(
        chart_path,
        chart::render_labelled(
            &records,
            &subtitle,
            scale,
            chart::Labels {
                y: args.chart_y_label.clone(),
                x: args.chart_x_label.clone(),
            },
        ),
    )?;
    if !args.json {
        println!("  chart    {chart_path}");
    }
    // A raster twin for documents that cannot embed an SVG. The SVG stays the
    // canonical artifact; this is derived from it, so the two cannot disagree.
    if args.chart_png {
        let png = profile::render_png(std::path::Path::new(chart_path))?;
        if let (Some(png), false) = (png, args.json) {
            println!("  chart    {}", png.display());
        }
    }
    Ok(())
}

/// The build the server under test is running, as `1.2.0 (52c0443ab)`, from
/// its own `/props`.
///
/// `None` when the server reports neither field — an older `orangu-server`, or
/// another engine entirely. The header then omits the line rather than
/// printing `unknown`, the same rule `pid`/`uptime` already follow: a report
/// should say what it knows and be silent about the rest, because a line that
/// reads "unknown" gets skimmed as a failure.
///
/// Version alone is accepted (a release build that could not resolve a commit
/// still knows its version); a commit alone is not, since a bare hash with no
/// version is less legible than the pid already printed beside it.
fn server_build(props: Option<&serde_json::Value>) -> Option<String> {
    let text = |key: &str| -> Option<String> {
        props
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty() && *v != "unknown")
            .map(str::to_string)
    };
    let version = text("version")?;
    match text("commit") {
        Some(commit) => Some(format!("{version} ({commit})")),
        None => Some(version),
    }
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
/// The second route is not a fallback for tidiness — a third-party server may
/// report no pid at all, and it is half of every comparison this tool exists to make.
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
    if !args.pg.is_empty() {
        format!("prefill+decode pg {} gen {}", list(&args.pg), args.n_gen)
    } else if !args.pp.is_empty() {
        format!("prefill pp {}", list(&args.pp))
    } else if !args.pp_continue.is_empty() {
        format!(
            "prefill +{} on {} cached",
            list(&args.pp_continue),
            args.pp_continue_base
        )
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
        "  profile  {} ({} samples over {:.0}s — {:.2} cores busy{})",
        s.svg.display(),
        s.samples,
        s.seconds,
        s.cores_busy,
        // The independent check, printed beside the number it validates rather
        // than only in the sidecar: sampling can miss threads, `/proc` cannot,
        // and a reader comparing the two sees at a glance whether this profile
        // describes the process or a corner of it.
        match s.cores_from_proc {
            Some(actual) => format!(", {actual:.2} per /proc"),
            None => String::new(),
        }
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
            "pp", "n_tok", "cached", "prompt_ms", "best", "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(70));
    }

    for (point, &len) in args.pp.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<PrefillSample> = None;
        // Discarded: drains the previous point's (or the warmup's) counters,
        // so what the read after the loop returns is this point alone.
        let _ = moe::take_stats(client, &args.url);
        for _ in 0..args.reps.max(1) {
            args.drop_page_cache(client);
            let s = run_prefill_once(client, &args.url, &prompt, &args.model, false)?;
            rates.push(s.tok_per_s());
            last = Some(s);
        }
        let moe_stats = moe::take_stats(client, &args.url);
        let stats = Stats::of(&rates, false);
        let s = last.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "pp": len,
                    "prompt_tokens": s.prompt_tokens,
                    "cached_tokens": s.cached_tokens,
                    "prompt_ms": s.prompt_ms,
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    // Unchanged meaning: the population estimator, as every
                    // consumer of this stream has always read it.
                    "tok_per_s_sd": stats.sd,
                    // The standard estimator, for putting this number beside
                    // one from another benchmark.
                    "tok_per_s_sd_sample": stats.sd_sample,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9.1} | {:>8.2} | {:>8.2} ± {}",
                len,
                s.prompt_tokens,
                s.cached_tokens,
                s.prompt_ms,
                stats.best,
                stats.mean,
                stats.plus_minus(5, 2)
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
                best: stats.best,
                mean: stats.mean,
                sd: stats.sd,
                sd_sample: stats.sd_sample,
                device: None,
            });
            // Keyed on the token count the server reported, exactly as the
            // rate row is, so a mechanism figure and the rate it explains sit
            // at the same `n` on two panels of the same chart.
            records.extend(moe::records(
                &moe_stats,
                label,
                s.prompt_tokens,
                args.reps.max(1),
            ));
            // Prefill's per-token I/O is the figure that separates a model
            // that streams from one that fits: at one pass it is a fraction
            // of the model, and at one pass per chunk it is the whole model
            // over and over.
            records.extend(moe::io_records(
                &moe_stats,
                label,
                s.prompt_tokens,
                f64::from(s.prompt_tokens) * f64::from(args.reps.max(1)),
                args.reps.max(1),
            ));
        }
        if !args.json {
            if let Some(line) = moe::summary_line(&moe_stats) {
                println!("{line}");
            }
            if let Some(line) = moe::io_line(
                &moe_stats,
                f64::from(s.prompt_tokens) * f64::from(args.reps.max(1)),
            ) {
                println!("{line}");
            }
        }
    }
    Ok(records)
}

/// Concurrency mode: aggregate decode throughput against the number of
/// concurrent streams.
///
/// The question this answers is whether the engine can *fill* the device, which
/// is not visible from any single-stream number. A decode step is a chain of
/// dependent dispatches, each too small to occupy the GPU on its own; whether
/// independent requests can interleave into the gaps depends on whether the
/// engine blocks the host between those dispatches.
///
/// Measured on this rig: an architecture whose decode is one GPU submission per
/// token goes 27.5 → 47.9 aggregate tok/s from one stream to two and pins the
/// engine at 99%, while one that round-trips per dispatch goes 21.3 → 24.5 and
/// never passes 66% however many streams are offered. Same device, same driver.
///
/// Reports the aggregate — the sum across streams, which is what a server's
/// capacity actually is — and the per-stream share beside it, since a rate that
/// is flat in aggregate and falling as 1/n means the streams are taking turns.
/// Pair it with the `gpu busy` line: aggregate that stops rising while the
/// engine is *not* at 100% is the interesting case.
fn run_streams(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>12} | {:>11} | {:>8}",
            "streams", "aggregate", "per-stream", "tokens"
        );
        println!("{}", "-".repeat(48));
    }

    for (point, &n) in args.streams.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let n = n.max(1);
        let mut aggregate = Vec::new();
        let mut total_tokens = 0u32;
        for _ in 0..args.reps.max(1) {
            let start = Instant::now();
            // One thread per stream, each its own request. Scoped threads so no
            // clones of the client are needed and no task can outlive the run.
            let results: Vec<Sample> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n)
                    .map(|i| {
                        scope.spawn(move || {
                            // Distinct prompts, so no two streams share a prefix
                            // cache entry and get a free ride.
                            let prompt =
                                format!("Stream {i}: tell a long, continuous story, do not stop:");
                            run_once(client, &args.url, &prompt, args.n_gen, &args.model)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().ok().and_then(Result::ok))
                    .collect()
            });
            let wall = start.elapsed().as_secs_f64();
            let tokens: u32 = results.iter().map(Sample::generated).sum();
            if tokens == 0 || wall <= 0.0 {
                anyhow::bail!("no tokens generated across {n} streams");
            }
            aggregate.push(f64::from(tokens) / wall);
            total_tokens = tokens;
        }
        let stats = Stats::of(&aggregate, false);

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "streams": n,
                    "aggregate_tok_per_s_best": stats.best,
                    "aggregate_tok_per_s_mean": stats.mean,
                    "aggregate_tok_per_s_sd": stats.sd,
                    "aggregate_tok_per_s_sd_sample": stats.sd_sample,
                    "per_stream_tok_per_s": stats.mean / f64::from(n),
                    "tokens": total_tokens,
                })
            );
        } else {
            println!(
                "{:>8} | {:>6.2} ± {} | {:>11.2} | {:>8}",
                n,
                stats.mean,
                stats.plus_minus(4, 2),
                stats.mean / f64::from(n),
                total_tokens
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "tg".to_string(),
            n,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }
    Ok(records)
}

/// Shared-prefix mode: `n` concurrent requests that all carry the **same**
/// long prefix, reporting the spread in time to first token.
///
/// [`run_streams`] deliberately gives every stream a distinct prompt, so that
/// none of them gets "a free ride" off the prefix cache. This mode is the
/// opposite experiment, and it measures something that number cannot show: what
/// a *shared* prefix costs when more than one request wants it at once.
///
/// The prefix cache pools finished requests' caches and hands the best match to
/// a new one — but the entry is **removed** when it is taken, so exactly one
/// in-flight request can hold it. The others find an empty pool and prefill the
/// shared prefix again from position zero. That is invisible to aggregate
/// throughput (the tokens still get generated) and invisible to a single-stream
/// TTFT (there is nobody to contend with). It shows up here, as a spread:
/// the stream that got the entry answers immediately, and every other stream
/// pays the whole prefill.
///
/// So the number to read is `slowest / fastest`. At 1 it is shared; well above
/// 1 the prefix is being recomputed per concurrent peer, and the absolute
/// `slowest` is roughly what a cold prefill of `--shared-prefix-tokens` costs.
///
/// One priming request runs first and is not timed — without it the first
/// measured round has nothing in the pool and every stream is cold, which
/// measures cold prefill rather than contention for a warm entry.
///
/// Sends `cache_prompt: true`, unlike every other mode here. That is the whole
/// experiment: with the cache off, all streams prefill and the mode measures
/// nothing but contention for the device.
fn run_shared_prefix(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    // The shared part. Built by the same padding helper the depth sweep uses,
    // so its token count is the one `--shared-prefix-tokens` asked for rather
    // than a character count that happens to tokenize near it.
    let prefix = build_prompt(args.shared_prefix_tokens);

    if !args.json {
        println!(
            "{:>8} | {:>11} | {:>11} | {:>7} | {:>8}",
            "streams", "ttft_fast", "ttft_slow", "ratio", "tokens"
        );
        println!("{}", "-".repeat(56));
    }

    // Prime the pool: one request establishes the prefix so the measured
    // rounds are contending for a warm entry rather than all arriving cold.
    let _ = run_once_cached(client, &args.url, &prefix, 1, &args.model);

    for (point, &n) in args.shared_prefix.iter().enumerate() {
        if point > 0 {
            args.settle();
        }
        let n = n.max(1);
        let mut fastest = f64::MAX;
        let mut slowest: f64 = 0.0;
        let mut total_tokens = 0u32;
        for _ in 0..args.reps.max(1) {
            let results: Vec<Sample> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n)
                    .map(|i| {
                        let prefix = prefix.as_str();
                        scope.spawn(move || {
                            // Identical prefix, distinct suffix: every stream
                            // can reuse everything but its own last few tokens,
                            // which is the shape a shared system prompt has.
                            let prompt = format!("{prefix}\nRequest {i}, continue:");
                            run_once_cached(client, &args.url, &prompt, args.n_gen, &args.model)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|h| h.join().ok().and_then(Result::ok))
                    .collect()
            });
            if results.is_empty() {
                anyhow::bail!("no streams completed at {n} sharing a prefix");
            }
            for s in &results {
                fastest = fastest.min(s.ttft_ms);
                slowest = slowest.max(s.ttft_ms);
                total_tokens += s.generated();
            }
        }
        let ratio = if fastest > 0.0 {
            slowest / fastest
        } else {
            0.0
        };

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "streams": n,
                    "shared_prefix_tokens": args.shared_prefix_tokens,
                    "ttft_ms_fastest": fastest,
                    "ttft_ms_slowest": slowest,
                    "ttft_ratio": ratio,
                    "tokens": total_tokens,
                })
            );
        } else {
            println!(
                "{:>8} | {:>9.0}ms | {:>9.0}ms | {:>6.1}x | {:>8}",
                n, fastest, slowest, ratio, total_tokens
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            // The recorded value is the *slowest* stream's TTFT: it is the one
            // a user actually waits for, and the one sharing would remove.
            mode: "ttft".to_string(),
            n,
            best: slowest,
            mean: slowest,
            sd: 0.0,
            sd_sample: None,
            device: None,
        });
    }
    Ok(records)
}

/// Scan-resistance mode: does a hot prefix survive a burst of unrelated
/// traffic?
///
/// A cache's replacement policy is invisible until something competes for it.
/// [`run_shared_prefix`] measures contention between requests that all want the
/// *same* prefix; this measures the opposite pressure — one prefix worth
/// keeping, and a stream of one-shot prompts that will never be asked for
/// again. That is what a served deployment actually looks like: a system prompt
/// reused by everyone, and everybody's own conversation behind it.
///
/// Under a purely recency-ordered policy the one-shot prompts are the most
/// recently used thing in the cache, so they evict the prefix that is about to
/// be needed again — the classic scan. A policy that also weighs how *often*
/// something is used should keep it.
///
/// Each point: use the hot prefix and time it, push `n` unique prompts through,
/// use the hot prefix again and time that. The number to read is the ratio. At
/// 1 the prefix survived; at the cost of a full cold prefill it was evicted,
/// and `n` is the depth at which that happens.
///
/// The unique prompts differ in their **first** tokens, not their last, so they
/// share no prefix with the hot one or with each other — otherwise they would
/// be partial hits and the pressure would be softer than intended.
fn run_prefix_scan(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    let hot = format!("HOT PREFIX. {}", build_prompt(args.shared_prefix_tokens));

    if !args.json {
        println!(
            "{:>8} | {:>11} | {:>11} | {:>8}",
            "scan", "ttft_warm", "ttft_after", "ratio"
        );
        println!("{}", "-".repeat(46));
    }

    for (point, &n) in args.prefix_scan.iter().enumerate() {
        if point > 0 {
            args.settle();
        }
        // Establish the hot prefix, then time it warm. Two calls, because the
        // first one is what puts it in the cache and the second is the reading.
        let _ = run_once_cached(client, &args.url, &hot, 1, &args.model);
        let warm = run_once_cached(client, &args.url, &hot, args.n_gen, &args.model)?;

        for i in 0..n {
            // Distinct leading text: a unique prompt must not be a partial hit
            // on the hot one, or it refreshes what it is supposed to displace.
            let junk = format!(
                "UNIQUE {point}-{i}. {}",
                build_prompt(args.shared_prefix_tokens)
            );
            let _ = run_once_cached(client, &args.url, &junk, 1, &args.model);
        }

        let after = run_once_cached(client, &args.url, &hot, args.n_gen, &args.model)?;
        let ratio = if warm.ttft_ms > 0.0 {
            after.ttft_ms / warm.ttft_ms
        } else {
            0.0
        };

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "scan": n,
                    "prefix_tokens": args.shared_prefix_tokens,
                    "ttft_ms_warm": warm.ttft_ms,
                    "ttft_ms_after_scan": after.ttft_ms,
                    "ttft_ratio": ratio,
                })
            );
        } else {
            println!(
                "{:>8} | {:>9.0}ms | {:>9.0}ms | {:>7.1}x",
                n, warm.ttft_ms, after.ttft_ms, ratio
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "ttft".to_string(),
            n,
            best: after.ttft_ms,
            mean: after.ttft_ms,
            sd: 0.0,
            sd_sample: None,
            device: None,
        });
    }
    Ok(records)
}

/// Decode-CPU mode: the server's own CPU time per **generated** token, with
/// prefill excluded.
///
/// Exists because the obvious way to get this number is wrong. Reading the
/// server's CPU over a whole `--depths N` run and dividing by the tokens
/// generated attributes the cost of prefilling N tokens to decode — and prefill
/// at depth 1024 is several CPU-seconds. Done that way, decode CPU per token
/// appears to grow 58% from depth 0 to 1024 on Llama-3.2-3B. Measured properly
/// it grows 17% over a 94x context increase: most of the "growth" was the
/// prefill, and it scaled with depth because the prefill did.
///
/// So each depth is measured twice. The first request pays the prefill and puts
/// it in the prefix cache; the CPU counter is read *after* that, and the second
/// request's prefill is a cache hit. What lands between the two readings is
/// decode and nothing else — which the printed `prefilled` column proves, by
/// reading 1 (the cache deliberately leaves one token to re-process).
///
/// Throughput is unaffected by the confound and is reported by the ordinary
/// depth sweep: that timing already starts at the first streamed token. This is
/// specifically for the CPU number, where nothing was excluding prefill.
fn run_decode_cpu(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    // Same resolution the profiler uses, and for the same reason: this reads
    // `/proc/<pid>/stat`, so the server has to be a local process.
    let pid = match args.flamegraph_pid {
        Some(pid) => pid,
        None => resolve_server_pid(client, args)?,
    };

    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>9} | {:>13} | {:>8}",
            "depth", "n_tok", "prefilled", "cpu_ms/token", "tok/s"
        );
        println!("{}", "-".repeat(58));
    }

    for (point, &depth) in args.depths.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let prompt = build_prompt(depth);
        let mut per_token = Vec::new();
        let mut rate = 0.0;
        let mut reported = (0u32, 0u32);
        for _ in 0..args.reps.max(1) {
            // Pay the prefill and leave it in the prefix cache.
            let primed = run_prefill_once(client, &args.url, &prompt, &args.model, true)?;
            let before = profile::proc_cpu_seconds(pid)
                .ok_or_else(|| anyhow::anyhow!("could not read /proc/{pid}/stat"))?;
            let s = run_once_cached(client, &args.url, &prompt, args.n_gen, &args.model)?;
            let after = profile::proc_cpu_seconds(pid)
                .ok_or_else(|| anyhow::anyhow!("could not read /proc/{pid}/stat"))?;
            if s.generated() == 0 {
                anyhow::bail!("no tokens generated at depth {depth}; cannot divide by zero");
            }
            per_token.push((after - before).max(0.0) / f64::from(s.generated()) * 1000.0);
            rate = s.tok_per_s();
            reported = (primed.prompt_tokens, primed.processed_tokens());
        }
        let stats = Stats::of(&per_token, true);

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "depth": depth,
                    "prompt_tokens": reported.0,
                    "prefilled_tokens": reported.1,
                    "cpu_ms_per_token_best": stats.best,
                    "cpu_ms_per_token_mean": stats.mean,
                    "cpu_ms_per_token_sd": stats.sd,
                    "cpu_ms_per_token_sd_sample": stats.sd_sample,
                    "tok_per_s": rate,
                })
            );
        } else {
            println!(
                "{:>8} | {:>7} | {:>9} | {:>6.3} ± {} | {:>8.2}",
                depth,
                reported.0,
                reported.1,
                stats.mean,
                stats.plus_minus(4, 3),
                rate
            );
        }

        // Recorded as its own mode: this is milliseconds of CPU, not tok/s, and
        // charting it on the same axis as a throughput series would be a
        // category error.
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "cpu".to_string(),
            n: reported.0,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }
    Ok(records)
}

/// [`run_once`] with the prefix cache **enabled**, so an already-primed prompt
/// costs no prefill. Only `--decode-cpu` wants this; every other mode sends
/// `cache_prompt: false` on purpose.
fn run_once_cached(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    n_gen: u32,
    model: &Option<String>,
) -> anyhow::Result<Sample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": n_gen,
        "n_predict": n_gen,
        "temperature": 0,
        "stream": true,
        "cache_prompt": true,
        "ignore_eos": true,
    });
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }
    stream_and_time(client, url, &body)
}

/// Continuation-prefill mode: time the prefill of a *small addition* to an
/// already-cached prompt.
///
/// `--pp` cannot reach this regime. It sends whole prompts with
/// `cache_prompt: false`, and a whole prompt carrying a chat template is
/// hundreds of tokens before any user text — so every `--pp` row exercises a
/// wide batch. Real multi-turn chat does the opposite: the prefix cache
/// supplies everything but the newest message, and the server prefills a
/// handful of tokens. That is a different point on the batch-width curve, and
/// it is the one a batch-width threshold actually governs.
///
/// Each rep primes the base prompt, then sends base + extension and reads what
/// the server says it processed. The extension's first token differs per rep so
/// the cache can only ever match the base.
fn run_pp_continue(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>7} | {:>9} | {:>9} | {:>8} | {:>16}",
            "added", "n_tok", "cached", "processed", "prompt_ms", "best", "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(82));
    }

    let base = build_prompt(args.pp_continue_base);
    for (point, &added) in args.pp_continue.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let mut rates = Vec::new();
        let mut last: Option<PrefillSample> = None;
        for rep in 0..args.reps.max(1) as usize {
            // Prime: put the base in the prefix cache. Its own cost is not
            // timed — only the extension that follows it is.
            run_prefill_once(client, &args.url, &base, &args.model, true)?;
            let prompt = format!("{base}{}", build_continuation(added, rep));
            let s = run_prefill_once(client, &args.url, &prompt, &args.model, true)?;
            rates.push(s.continuation_tok_per_s());
            last = Some(s);
        }
        let stats = Stats::of(&rates, false);
        let s = last.expect("at least one rep ran");
        let processed = s.processed_tokens();

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "added": added,
                    "prompt_tokens": s.prompt_tokens,
                    "cached_tokens": s.cached_tokens,
                    "processed_tokens": processed,
                    "prompt_ms": s.prompt_ms,
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    // Unchanged meaning: the population estimator, as every
                    // consumer of this stream has always read it.
                    "tok_per_s_sd": stats.sd,
                    // The standard estimator, for putting this number beside
                    // one from another benchmark.
                    "tok_per_s_sd_sample": stats.sd_sample,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9} | {:>9.1} | {:>8.2} | {:>8.2} ± {}",
                added,
                s.prompt_tokens,
                s.cached_tokens,
                processed,
                s.prompt_ms,
                stats.best,
                stats.mean,
                stats.plus_minus(5, 2)
            );
        } else {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9} | {:>9.1} | {:>8} | no server timings",
                added, "?", s.cached_tokens, "?", s.prompt_ms, "-"
            );
        }

        // Two ways this row can be meaningless, and both are silent unless
        // checked: no server timings at all, or a cache that supplied nothing
        // (so `processed` is the whole prompt and this is an ordinary wide
        // prefill wearing a continuation's label).
        if !s.server_reported {
            continue;
        }
        if s.cached_tokens == 0 {
            eprintln!(
                "  note: added={added} processed the whole prompt — no cache hit, \
                 so this is not a continuation; row dropped"
            );
            continue;
        }
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "pp".to_string(),
            n: processed,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }
    Ok(records)
}

/// The embedding sweep: one row per requested prompt length.
///
/// Reported as tok/s so the number lines up with the `pp` column of a
/// generative model — an embedding forward pass is prompt processing without
/// the decode that follows it, and that is the comparison worth making
/// (`embeddinggemma-300M` against the reference engine on the same file).
fn run_embed(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>9} | {:>8} | {:>16}",
            "embed", "n_tok", "wall_ms", "best", "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(60));
    }

    for (point, &len) in args.embed.iter().enumerate() {
        // Between points, so a card that heats up through a sweep is
        // given time to come back down — see `Args::settle`.
        if point > 0 {
            args.settle();
        }
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<EmbedSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_embed_once(client, &args.url, &prompt, &args.model)?;
            rates.push(s.tok_per_s());
            last = Some(s);
        }
        let stats = Stats::of(&rates, false);
        let s = last.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "embed": len,
                    "prompt_tokens": s.prompt_tokens,
                    "wall_ms": s.wall_ms,
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    // Unchanged meaning: the population estimator, as every
                    // consumer of this stream has always read it.
                    "tok_per_s_sd": stats.sd,
                    // The standard estimator, for putting this number beside
                    // one from another benchmark.
                    "tok_per_s_sd_sample": stats.sd_sample,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>9.1} | {:>8.2} | {:>8.2} ± {}",
                len,
                s.prompt_tokens,
                s.wall_ms,
                stats.best,
                stats.mean,
                stats.plus_minus(5, 2)
            );
        } else {
            // Same rule as `run_pp`: without the token count there is no rate,
            // and printing the latency alone is the only honest option.
            println!(
                "{:>8} | {:>7} | {:>9.1} | {:>8} | no usage.prompt_tokens (latency only)",
                len, "?", s.wall_ms, "-"
            );
        }

        if s.server_reported {
            records.push(history::Record {
                date: history::today(),
                label: label.to_string(),
                mode: "embed".to_string(),
                // The server's count, not the requested length — `build_prompt`
                // only approximates a token target, and two servers tokenizing
                // the same text can disagree.
                n: s.prompt_tokens,
                best: stats.best,
                mean: stats.mean,
                sd: stats.sd,
                sd_sample: stats.sd_sample,
                device: None,
            });
        }
    }
    Ok(records)
}

/// What the run was taken *on*, captured while it was taken — see
/// [`report_environment`].
struct Environment {
    /// The server's `/props`, `Null` when it did not answer.
    props: serde_json::Value,
    /// GPU clock state where the platform exposes it — empty elsewhere.
    gpus: Vec<GpuClock>,
    /// The GPU timestamp breakdown for the measured window — see
    /// [`take_gpu_timings`]. `Null` unless the server reports one.
    gpu_timings: serde_json::Value,
    /// How much of the model was in the page cache when the run started —
    /// `GET /model-cache`. On a model larger than memory this is not context,
    /// it is half the measurement: the same build measured cold and warm
    /// produces two rates that must not be compared, and nothing else in a
    /// bundle records which one was taken.
    model_cache: serde_json::Value,
}

/// Drain the server's accumulated GPU timestamp breakdown.
///
/// `GET /gpu-timings` is read-and-reset, so this is called **twice**: once
/// before the workload, whose result is thrown away, and once after, whose
/// result is exactly that window. Without the discard the reported breakdown
/// would include the warmup — which for a sweep is a whole extra pass of the
/// same workload, i.e. roughly double, and wrong in a way that looks
/// plausible.
///
/// Silent when the server has no such endpoint: this is the only profiling
/// instrument that works on a platform without `perf`, but it is still
/// optional, and another engine may not have it at all.
fn take_gpu_timings(client: &reqwest::blocking::Client, url: &str) -> serde_json::Value {
    client
        .get(format!("{url}/gpu-timings"))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| v.get("timings").cloned())
        .unwrap_or(serde_json::Value::Null)
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
///
/// Returns what it read as well as printing it, so [`write_bundle`] archives
/// the configuration that was live *during* the measurement. Re-fetching it
/// afterwards would usually agree and would occasionally, silently, not.
fn report_environment(client: &reqwest::blocking::Client, args: &Args) -> Environment {
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
    // Which build answered. Absent from a server too old to report it, and
    // from any other engine — hence `Option`, and hence the line below is
    // omitted rather than printed as "unknown".
    let build = server_build(props.as_ref());
    let gpus = gpu_clock_states();
    // `null` from a server without one, and from orangu-server on a non-`wgpu`
    // backend; a full `VulkanBackend::tuning_report` otherwise.
    let gpu_tuning = props.as_ref().and_then(|p| p.get("gpu")).cloned();
    // Probed here, before the workload: this is the state the run *started*
    // from. Read after it, a streaming model would report itself warm no
    // matter how cold it began.
    let model_cache = moe::take_residency(client, &args.url);

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "env",
                "url": args.url,
                "model": model,
                "backend": backend,
                // The one field that tells two builds of one version apart.
                "build": build,
                "pid": pid,
                "uptime_seconds": uptime,
                "gpus": gpus,
                // Verbatim, not summarized: the JSON stream is what gets
                // archived next to a throughput number, and a run measured
                // six months ago is only re-interpretable if the whole
                // configuration travelled with it.
                "gpu_tuning": gpu_tuning,
                "model_cache": model_cache,
            })
        );
    } else {
        println!("orangu-bench → {}", args.url);
        println!("  model    {model}");
        println!("  backend  {backend}");
        // Above the pid line, because it answers the same question one level
        // up: `pid`/`uptime` prove *which process* answered, this proves
        // *which build* it is running.
        if let Some(build) = &build {
            println!("  build    {build}");
        }
        for line in format_gpu_tuning(gpu_tuning.as_ref()) {
            println!("  {line}");
        }
        // Only for a server that reports them; not every server does, and a
        // missing field is not worth a line of output.
        if pid.is_some() || uptime.is_some() {
            let show = |v: Option<u64>| v.map_or_else(|| "?".to_string(), |n| n.to_string());
            println!("  server   pid {} up {}s", show(pid), show(uptime));
        }
        // Beside the GPU clock, because it is the same class of evidence and
        // the more consequential of the two for anything CPU-bound. A MoE
        // decode measured under `schedutil` came out **2.6x** slower than the
        // same build under `performance` on this project's own hardware,
        // where a GPU-heavy dense model lost 11%. The server's own banner
        // reports it, but a bench run is what a perf number is copied out of,
        // and a header that shows the card's clock while staying silent about
        // the CPU's invites the reading that only the card was checked.
        if let Some(governor) = orangu::hardware::cpu_governor() {
            println!("  cpu      governor {governor}");
        }
        for gpu in &gpus {
            println!(
                "  gpu      {} sclk {} ({})",
                gpu.card, gpu.sclk, gpu.power_level
            );
        }
        // Say that the clock is unknown, rather than printing nothing and
        // letting the absence read as "clocks were fine". `gpu_clock_states`
        // reads amdgpu's sysfs, which exists on no other platform — on macOS
        // in particular there is no unprivileged equivalent, so a Metal run's
        // numbers carry no evidence either way about whether the GPU stayed
        // at its boost clock while they were taken. That is a real limit on
        // what a Mac A/B can conclude and it belongs in the header, not in
        // the reader's memory. (Same reasoning as `ORANGU_GPU_TIMESTAMPS`'s
        // own "this adapter can't" warning: a diagnostic that silently does
        // nothing invites the conclusion that what it measures costs zero.)
        if gpus.is_empty() && !backend.starts_with("CPU") {
            println!("  gpu      clocks unreadable on this platform — no drift evidence");
        }
        // Beside the clock line, and for the same reason: a number taken with
        // the model on the disk and one taken with it in RAM are as
        // incomparable as two taken at different core clocks, and neither
        // difference is visible in the rate itself.
        if let Some(line) = moe::residency_line(&model_cache) {
            println!("{line}");
        }
    }
    Environment {
        props: props.unwrap_or(serde_json::Value::Null),
        gpus,
        // Filled in by the caller after the workload — this function runs
        // before it. Reported here as `Null` rather than as an absent field so
        // a bundle's shape does not depend on whether timestamps were on.
        gpu_timings: serde_json::Value::Null,
        model_cache,
    }
}

/// The `gpu` block of `/props` (`VulkanBackend::tuning_report`) as header
/// lines, or nothing at all when the server did not report one — most engines
/// never do, and neither does orangu-server on a CPU/CUDA/OpenCL/ROCm
/// backend.
///
/// Every A/B in this project's history has at some point been confounded by a
/// server that was not running the kernels the experimenter believed it was —
/// a stale binary, an `ORANGU_*` var left set in a different shell, an
/// adapter that quietly declined a feature. The `--json` stream carries the
/// whole report for the archive; these lines are the subset worth reading
/// *before* trusting the run: which quantized matmul kernel is live, which
/// attention path, and the geometry constants that were swept on one AMD card
/// and inherited everywhere else.
fn format_gpu_tuning(gpu: Option<&serde_json::Value>) -> Vec<String> {
    let Some(gpu) = gpu.filter(|g| !g.is_null()) else {
        return Vec::new();
    };
    // A split model reports its placement plan here instead of a tuning
    // report — a different document under the same key, and one this used to
    // read as a tuning report with every field missing.
    if gpu
        .get("split")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return format_device_split(gpu);
    }
    // Likewise a backend with no kernel *selection* to report, which says
    // what it runs instead — the CUDA/ROCm/OpenCL backends, which implement
    // `matmul` and nothing else (`Backend::reduced_surface`). A third
    // document under the same key, and reading it as a tuning report would
    // print the same wall of `?` the split used to: "this server declined to
    // say" where the truth is "there was nothing to choose".
    if let Some(surface) = gpu.get("surface").and_then(serde_json::Value::as_str) {
        return vec![format!("surface  {surface}")];
    }
    let get = |path: [&str; 2]| -> Option<&serde_json::Value> {
        gpu.get(path[0]).and_then(|v| v.get(path[1]))
    };
    let show = |v: Option<&serde_json::Value>| match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => "?".to_string(),
    };
    let mut out = Vec::new();
    out.push(format!(
        "api      {} · {}",
        show(gpu.get("api")),
        show(gpu.get("adapter"))
    ));
    // Q4_K and Q6_K specifically: between them they are `ffn_down`/`attn_v`
    // and everything else on the models this is benchmarked against, and they
    // are the two types with more than one kernel competing for them.
    let decode = gpu.get("kernels").and_then(|k| k.get("decode"));
    let prefill = gpu.get("kernels").and_then(|k| k.get("prefill"));
    let kernel = |m: Option<&serde_json::Value>, ty: &str| {
        show(m.and_then(|m: &serde_json::Value| m.get(ty)))
    };
    out.push(format!(
        "kernels  decode q4_k {} q6_k {} · prefill q4_k {}",
        kernel(decode, "Q4_K"),
        kernel(decode, "Q6_K"),
        kernel(prefill, "Q4_K"),
    ));
    // `coop_vec4_tiles` was one flag before the two tiles were answered
    // separately; a bundle or a server from before that split still reports
    // it, and reading a run against an older build is the ordinary case
    // during a bisect. Fall back rather than printing `?/?`.
    let tiles = match (
        get(["flags", "coop_vec4_tile_w"]),
        get(["flags", "coop_vec4_tile_x"]),
    ) {
        (Some(w), Some(x)) => format!("w {} x {}", show(Some(w)), show(Some(x))),
        _ => show(get(["flags", "coop_vec4_tiles"])),
    };
    out.push(format!(
        "gpu      kv {} · coop-tiles {tiles} · attn-coop {} · flash {} · f16 {} · subgroup {}",
        show(get(["flags", "kv_storage"])),
        show(get(["flags", "attn_coop"])),
        show(get(["flags", "flash_attn"])),
        show(get(["features", "shader_f16"])),
        show(get(["features", "subgroup"])),
    ));
    out.push(format!(
        "tuning   coop≥{} tok · n_rows {} · {} · split_k {} · geom {} · lds {} B",
        show(get(["tuning", "coop_min_n_tokens"])),
        show(get(["tuning", "reduce_n_rows"])),
        // `norm_wg` became a rule over row width rather than one constant,
        // so report what it answers at a representative width — and say when
        // `ORANGU_NORM_WG` has pinned it, which is what a sweep needs to see.
        {
            let at_3072 = show(
                gpu.get("tuning")
                    .and_then(|t| t.get("norm_wg_by_n_embd"))
                    .and_then(|m| m.get("3072")),
            );
            match show(get(["tuning", "norm_wg_pinned"])).as_str() {
                "true" => format!("norm_wg {at_3072} (pinned)"),
                // A server from before the rule existed still reports the flat
                // key; fall back to it rather than printing `?`.
                "?" => format!("norm_wg {}", show(get(["tuning", "norm_wg"]))),
                _ => format!("norm_wg {at_3072}@3072"),
            }
        },
        show(get(["tuning", "attn_split_k"])),
        show(get(["tuning", "coop_geom"])),
        show(get(["limits", "max_compute_workgroup_storage_size"])),
    ));
    out
}

/// The placement plan of a split model, as header lines: which devices, how
/// many layers each, how much of each card that filled, and how many times a
/// token crosses the bus.
///
/// This stands where the tuning report stands on a single device, because on a
/// split there is no single device to report one for (`as_wgpu` is `None` on
/// the multi-device backend, so the server never builds one). What it replaces
/// is worse than nothing: the single-device formatter reading this document
/// printed four lines of `?` — an adapter, kernels, flags and geometry all
/// "unknown" — while the plan sat unread in the same object. A reader could
/// not tell that from a server whose feature negotiation had gone wrong, and
/// nothing on screen said the model had been split at all.
///
/// The last line says the kernel selection is *unreported* rather than leaving
/// it out. An absent flag list reads as a default flag list, and this tool's
/// standing rule is that "not measured" must never be printable as a value —
/// the same reason `report_gpu_timings` prints its step count.
fn format_device_split(gpu: &serde_json::Value) -> Vec<String> {
    let devices = gpu
        .get("devices")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let api = gpu
        .get("api")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let boundaries = gpu
        .get("boundaries_per_token")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let gib = |v: Option<&serde_json::Value>| {
        v.and_then(serde_json::Value::as_u64)
            .map(|b| format!("{:.2} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0)))
    };
    let mut out = vec![format!(
        "api      {api} · split across {} device{} · {boundaries} hand-off{}/token",
        devices.len(),
        if devices.len() == 1 { "" } else { "s" },
        if boundaries == 1 { "" } else { "s" },
    )];
    for (i, device) in devices.iter().enumerate() {
        let name = device
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let layers = device
            .get("layers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        // "of ?" rather than a bare weight figure when the device did not
        // report a capacity: how full a card is is the number that says
        // whether a split is about to page, and a weight with nothing to
        // compare it against cannot answer that.
        let weights = gib(device.get("weights_bytes")).unwrap_or_else(|| "?".to_string());
        let total = gib(device.get("total_bytes")).unwrap_or_else(|| "?".to_string());
        let mut line = format!(
            "device {i}  {} · {layers} layer{} · {weights} of {total}",
            short_device(name),
            if layers == 1 { "" } else { "s" },
        );
        // The footprint, when the server reports one. Weights against capacity
        // says how full a card is; the headroom says whether what is left can
        // hold any context — which is the question a split is chosen to
        // answer, and the one a layer count cannot reach, since two devices
        // holding the same number of layers can hold very different shares of
        // the KV cache.
        let footprint = device.get("footprint").filter(|f| !f.is_null());
        if let Some(footprint) = footprint {
            match gib(footprint.get("headroom_bytes")) {
                Some(free) => line.push_str(&format!(" · {free} free")),
                // A device that declined to report a size. Said rather than
                // left blank: no headroom figure and a headroom of zero are
                // opposite answers.
                None => line.push_str(" · size unreported, so no headroom figure"),
            }
            if let Some(tokens) = footprint
                .get("kv_tokens_in_headroom")
                .and_then(serde_json::Value::as_u64)
            {
                // "room for", not "usable context": this is what the *card*
                // has space for across its own layers, which a model's trained
                // context may well be smaller than.
                line.push_str(&format!(" · KV room ~{}k tok", tokens / 1000));
            }
        }
        out.push(line);
        // The one thing knowable in advance, and the reason the footprint is
        // worth carrying at all: past this point the driver pages weights on
        // every token, which reads as "orangu is slow on this card" rather
        // than as a placement that needs changing.
        if let Some(short) = footprint.and_then(|f| gib(f.get("shortfall_bytes"))) {
            out.push(format!(
                "device {i}  OVER by {short} — the driver will page this device's weights on \
                 every token; give it a smaller share of the split"
            ));
        }
    }
    out.push(
        "kernels  not reported on a split — the tuning report describes one device, and this \
         run has none"
            .to_string(),
    );
    // Said here, unconditionally, rather than left to the absence of a timings
    // line after the table. `/gpu-timings` resolves a timestamp query set that
    // belongs to one device, so a split resolves none and reports nothing —
    // and a diagnostic that silently does nothing invites the reading that
    // what it measures cost nothing. `--flamegraph` still works, so the line
    // says what to reach for instead.
    out.push(
        "timings  no GPU stage breakdown on a split — query sets belong to one device; \
         --flamegraph still profiles the CPU side"
            .to_string(),
    );
    out
}

/// The GPU timestamp breakdown, as one line under the numbers it explains.
///
/// Per-step means rather than window totals: a window's totals depend on how
/// many tokens were generated in it, so two configurations measured with
/// different `--gen` would look different for a reason that has nothing to do
/// with either. The mean is the comparable figure.
///
/// Nothing at all when the server reports no timings — most engines have no
/// such endpoint, and orangu-server has none unless `ORANGU_GPU_TIMESTAMPS=1`
/// is set and the adapter has the query. That absence is why the "steps" count
/// is printed: zero steps says "not measured", which is a different statement
/// from zero milliseconds.
fn report_gpu_timings(timings: &serde_json::Value, args: &Args) {
    if args.json || timings.is_null() {
        return;
    }
    let per = |k: &str| {
        timings
            .get("per_step_ms")
            .and_then(|m| m.get(k))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
    };
    let steps = timings
        .get("steps")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if steps == 0 {
        return;
    }
    println!(
        "  gpu ms/step  total {:.3}  qkv {:.3}  attn {:.3}  ffn {:.3}  ple {:.3}  tail {:.3}  ({steps} steps)",
        per("total"),
        per("qkv"),
        per("attn"),
        per("ffn"),
        per("ple"),
        per("tail"),
    );
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
/// Peak core clock and mean engine/memory occupancy per card, over the measured
/// window.
pub struct GpuActivity {
    pub card: String,
    pub peak_mhz: u32,
    /// Mean `gpu_busy_percent` — the graphics engine.
    pub gpu_busy: Option<f64>,
    /// Mean `mem_busy_percent` — the **memory controller**.
    ///
    /// The one number that separates "this kernel is bandwidth-bound" from
    /// "this kernel is stalled". A workload at the card's streaming ceiling
    /// drives this to ~85%; orangu's decode sits at ~20% while delivering
    /// 48 GB/s, which says the memory system has four to five times the
    /// headroom the kernel is asking for. Getting this number was the blocking
    /// item on G3, and it had been assumed to require `RadeonGPUProfiler` and a
    /// display; `amdgpu` publishes it in sysfs.
    pub mem_busy: Option<f64>,
}

struct ClockWatch {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    sampler: Option<std::thread::JoinHandle<Vec<GpuActivity>>>,
}

impl ClockWatch {
    fn start() -> ClockWatch {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let sampler = std::thread::spawn(move || {
            // `(peak mhz, gpu_busy sum, mem_busy sum, samples)` per card.
            let mut acc: std::collections::BTreeMap<String, (u32, u64, u64, u64)> =
                std::collections::BTreeMap::new();
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                for gpu in gpu_clock_states() {
                    let mhz = parse_mhz(&gpu.sclk).unwrap_or(0);
                    let g = read_busy_percent(&gpu.card, "gpu_busy_percent");
                    let m = read_busy_percent(&gpu.card, "mem_busy_percent");
                    let slot = acc.entry(gpu.card).or_default();
                    slot.0 = slot.0.max(mhz);
                    // Only count a sample when both counters read, so the mean
                    // has one denominator rather than two.
                    if let (Some(g), Some(m)) = (g, m) {
                        slot.1 += u64::from(g);
                        slot.2 += u64::from(m);
                        slot.3 += 1;
                    }
                }
                // Fast enough to catch the ramp on a short run, slow enough
                // that reading sysfs is not itself part of the measurement.
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
            acc.into_iter()
                .map(|(card, (peak_mhz, gsum, msum, n))| GpuActivity {
                    card,
                    peak_mhz,
                    gpu_busy: (n > 0).then(|| gsum as f64 / n as f64),
                    mem_busy: (n > 0).then(|| msum as f64 / n as f64),
                })
                .collect()
        });
        ClockWatch {
            stop,
            sampler: Some(sampler),
        }
    }

    fn stop(mut self) -> Vec<GpuActivity> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.sampler
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default()
    }
}

/// One of `amdgpu`'s occupancy counters for `card`, as a percentage.
///
/// `None` on a non-AMD card, an older kernel, or anything that does not publish
/// the file — the caller prints what it has rather than claiming a zero.
fn read_busy_percent(card: &str, file: &str) -> Option<u32> {
    std::fs::read_to_string(format!("/sys/class/drm/{card}/device/{file}"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// `1700Mhz` → `1700`. Anything else — including the `0Mhz` sleep level, which
/// parses fine — is left to the caller to interpret.
fn parse_mhz(sclk: &str) -> Option<u32> {
    let digits: String = sclk.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Peak clocks, printed under the numbers they qualify.
fn report_clocks(activity: &[GpuActivity], args: &Args) {
    if activity.is_empty() {
        return;
    }
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "type": "gpu_activity",
                "cards": activity.iter().map(|a| serde_json::json!({
                    "card": a.card,
                    "peak_mhz": a.peak_mhz,
                    "gpu_busy_pct": a.gpu_busy,
                    "mem_busy_pct": a.mem_busy,
                })).collect::<Vec<_>>(),
            })
        );
        return;
    }
    let line = activity
        .iter()
        .map(|a| format!("{} {}Mhz", a.card, a.peak_mhz))
        .collect::<Vec<_>>()
        .join("  ");
    println!("  gpu peak {line} (while measuring)");

    // Printed next to the rate it explains. A rate that looks low is a very
    // different problem depending on whether the memory controller was at 20%
    // or 90% while producing it, and until this line existed the answer took a
    // separate experiment every time.
    let busy = activity
        .iter()
        .filter_map(|a| {
            let (g, m) = (a.gpu_busy?, a.mem_busy?);
            Some(format!("{} engine {g:.0}%  memory {m:.0}%", a.card))
        })
        .collect::<Vec<_>>();
    if !busy.is_empty() {
        println!("  gpu busy {} (mean while measuring)", busy.join("  "));
    }
}

/// Curve mode: one generation of `args.curve` tokens, timestamping each streamed
/// token, then reporting the instantaneous decode rate per `args.bucket`-token
/// context window. Measures decode-vs-context scaling directly — no prompt
/// padding, so no slow/VRAM-heavy deep-context prefill. Context position is
/// approximated by the generated-token index (the prompt is short).
///
/// Recorded to `--history` under its own `curve` mode. The depth sweep reaches
/// a context by *padding the prompt*, so every row costs a full prefill at that
/// depth per repetition — which on a model whose prefill is itself the thing
/// being investigated is both slow and a measurement of the wrong phase. One
/// generation gives the whole curve for one prefill, which is what makes a
/// decode-vs-context before/after affordable on a slow build.
fn run_curve(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    let prompt = build_prompt(0);
    let endpoint = format!("{}/v1/completions", args.url);
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": args.curve,
        "n_predict": args.curve,
        "temperature": 0,
        "stream": true,
        "cache_prompt": false,
        // Same contract as the depth sweep: measure decode, not content. A
        // greedy model will emit EOS well before a long curve is done — asking
        // for 192 tokens returned **45** on TinyLlama `Q8_0`, which is one
        // bucket, and a decode-vs-context curve of a single point is not a
        // curve. The whole mode exists to reach depth cheaply.
        "ignore_eos": true,
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
        // One bucket is one measurement of decode at a context length, which is
        // the same quantity `run_tg` records — but arrived at by a single pass
        // rather than best-of-N, so `best == mean` and `sd` is 0 by
        // construction. That is exactly why these rows get their own mode
        // rather than joining `tg`: merging a single-sample rate into a
        // best-of-N series would silently make the curve look like the same
        // statistic, and the noisier one would win the `best` reduction the
        // chart applies per (label, n).
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "curve".to_string(),
            n: lo as u32,
            best: rate,
            mean: rate,
            sd: 0.0,
            // One sample: a sample standard deviation is undefined, and the
            // sentence above is the reason this mode is kept apart from `tg`.
            sd_sample: None,
            device: None,
        });
        lo = hi;
    }
    Ok(records)
}

#[cfg(test)]
mod tests {

    /// `--cap` has to keep the server the *last* thing on the command line,
    /// after `--`, or `systemd-run` swallows the server's own flags as its
    /// own. It also has to pin swap off: a cap the kernel can satisfy by
    /// swapping measures the swap device, not the model's read path.
    /// A generated token that streams no text still cost a forward pass, and
    /// the decode window still spans it. Counting visible chunks therefore
    /// divides a full-length elapsed time by a short count and under-reports
    /// the rate — which is not a rounding error: measured on
    /// `gemma-4-E2B-it:Q4_K_M` at depth 1024, 128 generated tokens streamed as
    /// 79 visible ones and the rate read 27.6 instead of 44.5 tok/s. It looked
    /// exactly like a decode cliff, and there was no cliff.
    #[test]
    fn the_servers_own_token_count_beats_counting_visible_chunks() {
        // 128 tokens in 2.88 s is 44.4 tok/s, whatever streamed.
        let measured = Sample {
            gen_tokens: 79,
            ttft_ms: 7109.0,
            decode_s: 2.88,
            reported: Some((128, 2880.0)),
        };
        assert!(
            (measured.tok_per_s() - 44.44).abs() < 0.05,
            "reported {}",
            measured.tok_per_s()
        );
        assert_eq!(measured.generated(), 128);

        // Counting chunks on the same stream is the under-read this avoids.
        let visible_only = Sample {
            reported: None,
            ..measured
        };
        assert!(
            visible_only.tok_per_s() < 28.0,
            "the fallback should be the low read: {}",
            visible_only.tok_per_s()
        );

        // A server that reports no timings still gets a number.
        assert_eq!(visible_only.generated(), 79);
        // And a nonsense timings block falls back rather than dividing by zero.
        let zeroed = Sample {
            reported: Some((0, 0.0)),
            ..measured
        };
        assert!(zeroed.tok_per_s() > 27.0);
    }

    /// Each sweep point's profile must land in its own file. Sharing one path
    /// would leave the last point's profile under the run's name, which reads
    /// as a profile of the whole sweep and is a profile of one arm of it.
    #[test]
    fn each_sweep_point_gets_its_own_profile_path() {
        let a = profile_path_for_point("perf/pp.svg", "ORANGU_NO_CHUNK_COST_FIT=1");
        let b = profile_path_for_point("perf/pp.svg", "ORANGU_NO_CHUNK_COST_FIT=0");
        assert_ne!(a, b, "two points must not share a path");
        // The directory is rejoined with the platform's separator.
        let in_perf = |name: &str| {
            std::path::Path::new("perf")
                .join(name)
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(a, in_perf("pp-orangu-no-chunk-cost-fit-1.svg"));
        assert_eq!(b, in_perf("pp-orangu-no-chunk-cost-fit-0.svg"));
        // A bare filename keeps working, and so does one with no extension.
        assert_eq!(profile_path_for_point("pp.svg", "x=1"), "pp-x-1.svg");
        assert_eq!(profile_path_for_point("pp", "x=1"), "pp-x-1");
    }

    #[test]
    fn the_cap_wraps_the_command_without_eating_its_arguments() {
        let wrapped = capped("orangu-server -c cfg.conf model:Q4_K_M", Some("4G"));
        assert!(
            wrapped.ends_with("-- orangu-server -c cfg.conf model:Q4_K_M"),
            "{wrapped}"
        );
        assert!(wrapped.contains("MemoryMax=4G"), "{wrapped}");
        assert!(wrapped.contains("MemorySwapMax=0"), "{wrapped}");
    }

    /// No cap must leave the command byte-identical: `--sweep` without
    /// `--cap` is the control arm for every capped run, and a wrapper that
    /// is "almost nothing" is still a difference between the arms.
    #[test]
    fn no_cap_leaves_the_command_untouched() {
        let plain = "orangu-server -c cfg.conf model";
        assert_eq!(capped(plain, None), plain);
    }
    use super::*;

    /// The placement plan a split server reports under `props.gpu`, as
    /// `apply_device_split`'s `to_json` writes it.
    fn split_props() -> serde_json::Value {
        serde_json::json!({
            "model": "gemma-4-E2B",
            "backend": "Vulkan/AMD Some Card (DRIVER TAG) + 1 more (2 devices, split)",
            "gpu": {
                "api": "vulkan",
                "split": true,
                "boundaries_per_token": 1,
                "devices": [
                    {
                        "name": "AMD Some Card (DRIVER TAG)",
                        "total_bytes": 4_294_967_296_u64,
                        "weights_bytes": 3_221_225_472_u64,
                        "layers": 18,
                        "footprint": {
                            "weights_device_bytes": 3_221_225_472_u64,
                            "device_total_bytes": 4_294_967_296_u64,
                            "headroom_bytes": 1_073_741_824_u64,
                            "shortfall_bytes": serde_json::Value::Null,
                            "kv_tokens_in_headroom": 24_000,
                        },
                    },
                    {
                        "name": "AMD Other Card (DRIVER TAG)",
                        "total_bytes": 12_884_901_888_u64,
                        "weights_bytes": 1_073_741_824_u64,
                        "layers": 6,
                        "footprint": {
                            "weights_device_bytes": 1_073_741_824_u64,
                            "device_total_bytes": 12_884_901_888_u64,
                            "headroom_bytes": 11_811_160_064_u64,
                            "shortfall_bytes": serde_json::Value::Null,
                            "kv_tokens_in_headroom": 512_000,
                        },
                    },
                ],
            },
        })
    }

    /// The same failure as the split one below, for the third document that
    /// can arrive under `gpu`. A backend that implements `matmul` and nothing
    /// else has no kernel table, no flags and no geometry, so every field a
    /// tuning report asks for is absent — and a header full of `?` reads as a
    /// device that lost its kernels rather than one that never had a choice
    /// to report. The point of the row is that it is *never* silent on a GPU:
    /// a run on one of these backends must not be mistaken for a full-path
    /// run by anything reading the header.
    #[test]
    fn a_matmul_only_backend_reports_its_surface_instead_of_a_wall_of_question_marks() {
        let gpu = serde_json::json!({
            "surface": "matmul only - no fused layer chain, GPU attention or GPU sampling",
            "kernels": serde_json::Value::Null,
        });
        let text = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(!text.contains('?'), "{text}");
        assert!(text.contains("matmul only"), "{text}");
        assert!(text.contains("no fused layer chain"), "{text}");
        // One line, not a tuning report with one field filled in.
        assert_eq!(text.lines().count(), 1, "{text}");
    }

    /// The header used to read a split's placement plan as a tuning report
    /// with every field missing, and print four lines of `?`. A `?` in that
    /// block means "this server declined to say", which is how a broken
    /// feature negotiation looks — so a split was indistinguishable from a
    /// card that had quietly lost half its kernels, and nothing said the model
    /// had been split at all.
    #[test]
    fn a_split_reports_its_plan_instead_of_a_wall_of_question_marks() {
        let props = split_props();
        let lines = format_gpu_tuning(props.get("gpu"));
        let text = lines.join("\n");
        assert!(!text.contains('?'), "{text}");
        assert!(text.contains("split across 2 devices"), "{text}");
        assert!(text.contains("1 hand-off/token"), "{text}");
        // Each device, with what it is holding and what it had to hold it in.
        assert!(text.contains("Some Card"), "{text}");
        assert!(text.contains("18 layers"), "{text}");
        assert!(text.contains("3.00 GiB of 4.00 GiB"), "{text}");
        assert!(text.contains("6 layers"), "{text}");
        // And the absence of a kernel report is stated, not left to be read
        // as a default kernel report.
        assert!(text.contains("not reported on a split"), "{text}");
        // Same for the GPU stage breakdown: a split resolves no timestamp
        // query set, and a run that printed nothing about it would look like a
        // run whose GPU stages cost nothing.
        assert!(text.contains("no GPU stage breakdown on a split"), "{text}");
        assert!(text.contains("--flamegraph"), "{text}");
        // The footprint: what is left on each card, and what it buys. A layer
        // count cannot answer this — these two devices hold 18 and 6 layers
        // and have 1.00 and 11.00 GiB free.
        assert!(text.contains("1.00 GiB free"), "{text}");
        assert!(text.contains("11.00 GiB free"), "{text}");
        assert!(text.contains("KV room ~24k tok"), "{text}");
        // `provenance_fields` splits these on the first double space, so every
        // line has to carry one — otherwise the PDF gets a row with no label.
        for line in &lines {
            assert!(line.split_once("  ").is_some(), "{line:?}");
        }
    }

    /// The single-device path is the one every existing run takes, and this
    /// change must not touch it.
    #[test]
    fn a_single_device_still_gets_the_tuning_report() {
        let gpu = serde_json::json!({
            "api": "vulkan",
            "adapter": "AMD Some Card (DRIVER TAG)",
            "kernels": {"decode": {"Q4_K": "coop", "Q6_K": "scalar"}, "prefill": {"Q4_K": "tile"}},
            "flags": {"kv_storage": "F16", "attn_coop": true, "flash_attn": false},
            "features": {"shader_f16": true, "subgroup": true},
            "tuning": {"coop_min_n_tokens": 8, "reduce_n_rows": 4},
        });
        let text = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(text.contains("api      vulkan · AMD Some Card"), "{text}");
        assert!(text.contains("decode q4_k coop"), "{text}");
        // `split_k` is a tuning constant on this path; the placement plan's
        // wording is what must not appear.
        assert!(!text.contains("split across"), "{text}");
    }

    /// A matrix is a loop over models wrapping a sweep over devices, into one
    /// history file. That only works if the two axes land in *different*
    /// fields: the sweep names its points, the device column fills itself in,
    /// and the model has nothing left but `--label` — which a sweep used to
    /// throw away, so two models wrote four series names between eight
    /// measurements and the file could not say which model a row came from.
    #[test]
    fn a_labelled_sweep_keeps_the_label_so_two_models_do_not_collide() {
        let spec = sweep::Spec::parse("ORANGU_DEVICE=0,1").expect("a valid spec");
        let mut args = Args::parse_from(["orangu-bench"]);
        // Unlabelled: the sweep names its own points, exactly as before.
        assert_eq!(sweep_point_label(&args, &spec, "0"), "ORANGU_DEVICE=0");

        // Labelled: the label scopes them, and two models' points differ.
        args.label = Some("llama-1b".to_string());
        let llama = sweep_point_label(&args, &spec, "0");
        args.label = Some("smollm-360m".to_string());
        let smol = sweep_point_label(&args, &spec, "0");
        assert_ne!(llama, smol, "two models must not share a series name");
        assert!(llama.contains("llama-1b") && llama.contains("ORANGU_DEVICE=0"));
        // And the point is still distinguished within one model's sweep —
        // a label that replaced the point name would trade one collision for
        // another.
        args.label = Some("llama-1b".to_string());
        assert_ne!(
            sweep_point_label(&args, &spec, "0"),
            sweep_point_label(&args, &spec, "1")
        );
    }

    /// The case the footprint exists for: a device the plan overfilled. The
    /// rate collapses because the driver pages weights on every token, and
    /// nothing else in the header would say so — a layer count and a weight
    /// figure both look ordinary.
    #[test]
    fn an_over_subscribed_device_of_a_split_says_so() {
        let gpu = serde_json::json!({
            "api": "vulkan",
            "split": true,
            "boundaries_per_token": 1,
            "devices": [{
                "name": "AMD Some Card (DRIVER TAG)",
                "total_bytes": 4_294_967_296_u64,
                "weights_bytes": 6_442_450_944_u64,
                "layers": 40,
                "footprint": {
                    "headroom_bytes": 0,
                    "shortfall_bytes": 2_147_483_648_u64,
                    "kv_tokens_in_headroom": 0,
                },
            }],
        });
        let text = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(text.contains("OVER by 2.00 GiB"), "{text}");
        assert!(text.contains("page this device's weights"), "{text}");
    }

    /// A server from before the footprint existed still reports the plan, and
    /// reading a run against an older build is the ordinary case during a
    /// bisect. The device lines have to survive it without inventing a
    /// headroom figure.
    #[test]
    fn a_split_without_a_footprint_still_reports_its_plan() {
        let gpu = serde_json::json!({
            "api": "vulkan",
            "split": true,
            "boundaries_per_token": 1,
            "devices": [{
                "name": "AMD Some Card (DRIVER TAG)",
                "total_bytes": 4_294_967_296_u64,
                "weights_bytes": 3_221_225_472_u64,
                "layers": 18,
            }],
        });
        let text = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(text.contains("18 layers · 3.00 GiB of 4.00 GiB"), "{text}");
        assert!(!text.contains("free"), "{text}");
        assert!(!text.contains("OVER"), "{text}");
    }

    /// A split must not be tagged as the card at the head of it. The server's
    /// backend label starts with that card's name, so any shortener that
    /// treated it as ordinary text would file a two-card run under the same
    /// device as a one-card run — silently making the two comparable in the
    /// history file.
    #[test]
    fn a_split_is_not_tagged_as_its_head_device() {
        let split = device_tag(&split_props()).expect("a backend was reported");
        let single = device_tag(&serde_json::json!({
            "backend": "Vulkan/AMD Some Card (DRIVER TAG)",
            "gpu": {"api": "vulkan", "adapter": "AMD Some Card (DRIVER TAG)"},
        }))
        .expect("a backend was reported");
        assert_eq!(single, "Vulkan/Some Card");
        assert_eq!(split, "Vulkan/Some Card (split 2)");
        assert_ne!(split, single);
    }

    /// The tag has to survive the servers that report no `gpu` block at all —
    /// the CPU backend, and every other engine this harness is pointed at.
    #[test]
    fn a_device_tag_without_a_gpu_block_is_the_backend_itself() {
        let tag = |backend: &str| device_tag(&serde_json::json!({"backend": backend}));
        assert_eq!(tag("CPU/AVX2").as_deref(), Some("CPU/AVX2"));
        assert_eq!(
            tag("Metal/Apple M1 Pro (Metal)").as_deref(),
            Some("Metal/M1 Pro")
        );
        // Nothing to record is recorded as nothing, not as an empty device.
        assert_eq!(tag(""), None);
        assert_eq!(device_tag(&serde_json::json!({})), None);
        // A name that is *only* a parenthetical keeps it rather than becoming
        // an empty tag.
        assert_eq!(tag("(DRIVER TAG)").as_deref(), Some("(DRIVER TAG)"));
    }

    /// Two standard deviations, and the difference between them is not
    /// cosmetic: at three repetitions the sample estimator is 22% larger than
    /// the population one, which is bigger than most of the differences this
    /// tool is run to detect. A `±` quoted against another benchmark's `±`
    /// has to be the same estimator, and everyone else reports `n-1`.
    #[test]
    fn both_estimators_are_reported_and_they_are_not_the_same_number() {
        let stats = Stats::of(&[10.0, 12.0, 14.0], false);
        assert_eq!(stats.best, 14.0);
        assert!((stats.mean - 12.0).abs() < 1e-9);
        // population: sqrt(8/3) = 1.632…, sample: sqrt(8/2) = 2.0
        assert!(
            (stats.sd - (8.0_f64 / 3.0).sqrt()).abs() < 1e-9,
            "{stats:?}"
        );
        assert!(
            (stats.sd_sample.expect("defined") - 2.0).abs() < 1e-9,
            "{stats:?}"
        );
        assert!(stats.sd_sample.expect("defined") > stats.sd);
    }

    /// One repetition has no spread to report. `0.00` would say one was
    /// measured and found to be zero, which is a different — and false —
    /// claim; the row says `—` instead.
    #[test]
    fn a_single_repetition_has_no_sample_deviation() {
        let stats = Stats::of(&[41.0], false);
        assert_eq!(stats.best, 41.0);
        assert_eq!(stats.mean, 41.0);
        assert_eq!(stats.sd, 0.0);
        assert_eq!(stats.sd_sample, None);
        assert_eq!(stats.plus_minus(5, 2).trim(), "—");
    }

    /// `--decode-cpu` reports milliseconds per token, where the best run is
    /// the *smallest*. A shared statistic that always maximised would quietly
    /// report the worst row of that table as its headline.
    #[test]
    fn the_best_of_a_lower_is_better_measurement_is_the_smallest() {
        assert_eq!(Stats::of(&[15.3, 14.4, 15.7], true).best, 14.4);
        assert_eq!(Stats::of(&[15.3, 14.4, 15.7], false).best, 15.7);
    }

    /// The build line is the answer to "which build produced this number", so
    /// what it does when it *cannot* answer matters as much as what it prints
    /// when it can: an older `orangu-server` and every other engine report
    /// neither field, and inventing "unknown
    /// (unknown)" for them would put a
    /// failure-shaped line in every such report.
    #[test]
    fn the_build_line_says_what_it_knows_and_nothing_else() {
        let props = |json| serde_json::from_str::<serde_json::Value>(json).expect("json");

        assert_eq!(
            server_build(Some(&props(r#"{"version":"1.2.0","commit":"52c0443ab"}"#))).as_deref(),
            Some("1.2.0 (52c0443ab)")
        );
        // A release built from a tarball knows its version and nothing more —
        // that is the whole truth available, so it is what gets printed.
        assert_eq!(
            server_build(Some(&props(r#"{"version":"1.2.0","commit":"unknown"}"#))).as_deref(),
            Some("1.2.0")
        );
        assert_eq!(
            server_build(Some(&props(r#"{"version":"1.2.0"}"#))).as_deref(),
            Some("1.2.0")
        );
        // A dirty build must say so — the commit alone would be a lie about
        // what was measured.
        assert_eq!(
            server_build(Some(&props(
                r#"{"version":"1.2.0","commit":"52c0443ab-dirty"}"#
            )))
            .as_deref(),
            Some("1.2.0 (52c0443ab-dirty)")
        );
        // No line at all: an older server, another engine, or no `/props`.
        assert_eq!(server_build(Some(&props(r#"{"model":"x"}"#))), None);
        assert_eq!(server_build(Some(&props(r#"{"version":""}"#))), None);
        assert_eq!(
            server_build(Some(&props(r#"{"commit":"52c0443ab"}"#))),
            None
        );
        assert_eq!(server_build(None), None);
    }

    /// A server that reports no `gpu` block gets no header lines — not a row
    /// of `?`s. A server without a pid is the case that matters: this tool is pointed at
    /// it for every cross-implementation comparison, and four lines of unknowns
    /// under its name would read as "orangu-server's kernels, unreadable"
    /// rather than "a different server, which has none of these".
    #[test]
    fn no_gpu_block_prints_no_gpu_lines() {
        assert!(format_gpu_tuning(None).is_empty());
        assert!(format_gpu_tuning(Some(&serde_json::Value::Null)).is_empty());
    }

    /// A report that *is* present must survive missing sub-keys — an older
    /// server on the far end of the socket is the ordinary case during a
    /// bisect, and the header is a diagnostic, not something worth aborting a
    /// benchmark over. Every value it does carry must still appear.
    #[test]
    fn gpu_lines_report_what_is_present_and_tolerate_what_is_not() {
        let gpu = serde_json::json!({
            "api": "Metal",
            "adapter": "Apple M3 Max (Metal)",
            "flags": { "kv_storage": "F16", "coop_vec4_tile_w": false, "coop_vec4_tile_x": true },
            "kernels": { "decode": { "Q4_K": "q4_k-light" } },
        });
        let lines = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(lines.contains("Metal"), "{lines}");
        assert!(lines.contains("Apple M3 Max"), "{lines}");
        assert!(lines.contains("q4_k-light"), "{lines}");
        assert!(lines.contains("F16"), "{lines}");
        assert!(lines.contains("w false x true"), "{lines}");
        // The absent ones degrade to `?` rather than panicking or vanishing:
        // a blank where a kernel name belongs is indistinguishable from a
        // kernel actually named nothing.
        assert!(lines.contains('?'), "{lines}");
    }

    /// Short prompts must actually be short, and must differ from each other
    /// by roughly the amount asked for.
    ///
    /// The bug this pins: the repeat-and-pad builder's preamble and tail are
    /// ~28 tokens before a single word of content, so every requested length
    /// below one sentence produced the *same* ~52-token prompt. Three
    /// `--pp` values measuring one prefill is not a small inaccuracy — it made
    /// a whole knob (`ORANGU_COOP_MIN_TOKENS`, whose regime is forwards of
    /// 2..24 positions) read as having no effect below its default, because
    /// every value below it was the same configuration by construction.
    #[test]
    fn short_prompts_are_short_and_distinct() {
        let words = |s: &str| s.split_whitespace().count();
        // Monotone, and near the requested count rather than 40 tokens above.
        for depth in [1u32, 4, 8, 16, 23] {
            let got = words(&build_prompt(depth));
            assert_eq!(got, depth as usize, "--pp {depth} should be ~{depth} words");
        }
        assert_ne!(build_prompt(8), build_prompt(16));
        // No "continue, do not stop" tail down here: it is ~20 tokens, it only
        // matters to modes that generate, and at these lengths it *is* the
        // prompt.
        assert!(!build_prompt(8).contains("do not stop"));
    }

    /// ...and every length the old builder could really produce is untouched,
    /// so no `pp` point already in a history file moves under it.
    #[test]
    fn prompts_of_a_sentence_or_more_keep_the_original_construction() {
        for depth in [24u32, 48, 256, 1024, 2048] {
            let p = build_prompt(depth);
            assert!(
                p.starts_with("Here is the story so far:"),
                "--pp {depth} must keep the padded construction"
            );
            assert!(p.contains("do not stop"), "--pp {depth} must keep the tail");
        }
    }

    /// A bundle or server from before the tile flag was split in two still
    /// reports the single `coop_vec4_tiles`, and reading an old run against a
    /// new build is the ordinary case during a bisect. The header must show
    /// what it says rather than two question marks.
    #[test]
    fn the_pre_split_tile_flag_is_still_read() {
        let gpu = serde_json::json!({
            "api": "Vulkan",
            "flags": { "coop_vec4_tiles": true },
        });
        let lines = format_gpu_tuning(Some(&gpu)).join("\n");
        assert!(lines.contains("coop-tiles true"), "{lines}");
    }

    /// The one thing `--pp-continue` must get right. A prefix cache matches on
    /// the longest common token prefix, so two reps whose extensions began with
    /// the same word would share more than the base — and in the limit the
    /// second rep would find everything cached and time a lookup instead of a
    /// prefill. Diverging at the first character is what bounds the match to
    /// the base.
    #[test]
    fn continuations_for_different_reps_diverge_at_the_first_word() {
        let a = build_continuation(64, 0);
        let b = build_continuation(64, 1);
        assert_ne!(a, b);
        let first = |s: &str| s.split_whitespace().next().unwrap().to_string();
        assert_ne!(first(&a), first(&b));
        // ...and the shared prefix is empty, not merely short.
        assert_ne!(a.as_bytes()[0], b.as_bytes()[0]);
    }

    /// Reps beyond the opener list wrap around, which is fine — but adjacent
    /// reps must never collide, since a cache holds the immediately preceding
    /// prompt.
    #[test]
    fn adjacent_reps_never_share_an_opener() {
        for rep in 0..20 {
            assert_ne!(
                build_continuation(32, rep),
                build_continuation(32, rep + 1),
                "reps {rep} and {} collided",
                rep + 1
            );
        }
    }

    /// A longer request must actually produce more text; otherwise every row of
    /// the sweep would measure the same batch width under different labels.
    #[test]
    fn a_larger_addition_produces_a_longer_continuation() {
        let small = build_continuation(32, 0);
        let large = build_continuation(256, 0);
        assert!(
            large.len() > small.len() * 2,
            "32 -> {} bytes, 256 -> {} bytes",
            small.len(),
            large.len()
        );
    }

    /// The rate must come from the tokens that were forwarded, not from the
    /// whole prompt. Getting this wrong inflates a continuation's rate by the
    /// cache ratio — here 8x — which would look like a spectacular result.
    #[test]
    fn a_continuation_rate_counts_only_the_uncached_tokens() {
        let s = PrefillSample {
            prompt_tokens: 512,
            cached_tokens: 448,
            prompt_ms: 1000.0,
            server_reported: true,
        };
        assert_eq!(s.processed_tokens(), 64);
        assert!((s.continuation_tok_per_s() - 64.0).abs() < 1e-9);
        // What the plain sweep would have reported for the same response.
        assert!((s.tok_per_s() - 512.0).abs() < 1e-9);
    }
}

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

mod bundle;
mod chart;
mod flamegraph;
mod history;
mod profile;
mod sweep;

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

    /// Continuation-prefill mode: comma-separated *added* token counts to sweep.
    #[arg(long, value_delimiter = ',')]
    pp_continue: Vec<u32>,

    /// Report the server's CPU time per generated token, with prefill excluded.
    #[arg(long, default_value_t = false)]
    decode_cpu: bool,

    /// Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s.
    #[arg(long, value_delimiter = ',')]
    streams: Vec<u32>,

    /// Prompt length (tokens) to prime the prefix cache with for `--pp-continue`.
    #[arg(long, default_value_t = 512)]
    pp_continue_base: u32,

    /// Embedding mode: comma-separated prompt lengths (in tokens) to sweep
    /// against `POST /v1/embeddings`, reporting **forward-pass** throughput —
    /// the embedding-model equivalent of `--pp`, and the only mode that works
    /// on an embedding-only server at all.
    ///
    /// Such a server answers `/v1/completions` with HTTP 501, so both the
    /// default decode sweep and `--pp` fail outright on it: `embeddinggemma-
    /// 300M` is a supported, working model that this tool simply could not
    /// measure. Rates come from the response's `usage.prompt_tokens` — which
    /// both `orangu-server` and `llama-server` report — so the token count is
    /// the one the forward pass actually ran, not an estimate from the prompt
    /// text.
    #[arg(long, value_delimiter = ',')]
    embed: Vec<u32>,

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

    /// Also render a PNG beside the chart SVG.
    #[arg(long, default_value_t = false)]
    chart_png: bool,

    /// Pin the chart's tok/s axis to `MIN:MAX` so a pair of charts compare.
    #[arg(long)]
    chart_scale: Option<String>,

    /// Label for the chart's y-axis.
    #[arg(long, default_value = "tok/s (log)")]
    chart_y_label: String,

    /// Label for the chart's x-axis.
    #[arg(long)]
    chart_x_label: Option<String>,

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

    /// Write the whole run — measurements, server configuration, host — to one
    /// JSON file to carry off this machine.
    #[arg(long)]
    bundle: Option<String>,

    /// Read bundles and report them side by side; measure nothing.
    #[arg(long, value_delimiter = ',')]
    read_bundle: Vec<String>,

    /// Sweep one tuning variable: `VAR=v1,v2,...`, restarting the server per
    /// value. Needs `--sweep-cmd`.
    #[arg(long)]
    sweep: Option<String>,

    /// Shell command that starts the server, run once per `--sweep` value.
    #[arg(long)]
    sweep_cmd: Option<String>,

    /// Environment held constant across every `--sweep` point (repeatable).
    #[arg(long = "sweep-env")]
    sweep_env: Vec<String>,

    /// Seconds to wait for a swept server to come up.
    #[arg(long, default_value_t = 300)]
    sweep_start_timeout: u64,
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
/// `llama-server` honour it; the `cached` column the caller prints is the check
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
/// the server rather than from a separate `llama-tokenize` run.
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

    // The other half of `--bundle`: profile there, analyze here.
    if !args.read_bundle.is_empty() {
        return compare_bundles(args);
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

    if let Some(recorder) = recorder {
        match recorder.finish() {
            Ok(summary) => report_profile(&summary, args),
            // The rate is the deliverable; a profile that failed to render is
            // reported and does not discard the measurement that just ran.
            Err(e) => eprintln!("orangu-bench: flamegraph not written: {e}"),
        }
    }

    let measured = measured?;
    write_bundle(args, args.bundle.as_deref(), &env, &measured)?;
    record_and_chart(args, &measured)
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
    bundle::write(path, &env.props, host, run, &env.gpu_timings, records)?;
    if !args.json {
        println!("  bundle   {path} ({} rows)", records.len());
    }
    Ok(())
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

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    let mut all = Vec::new();
    let mut bundles = Vec::new();
    for value in &spec.values {
        let label = spec.label(value);
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
        let _ = take_gpu_timings(&client, &args.url);
        let clocks = ClockWatch::start();
        let measured = measure(&client, args, &label);
        report_clocks(&clocks.stop(), args);
        env_report.gpu_timings = take_gpu_timings(&client, &args.url);
        report_gpu_timings(&env_report.gpu_timings, args);
        let measured = measured?;

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
        spec.label(&spec.values[0])
    );
    print_sweep_table(&spec, &all);
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

/// `ORANGU_COOP_MIN_TOKENS=16` → `orangu_coop_min_tokens-16`, for a filename.
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
fn print_sweep_table(spec: &sweep::Spec, records: &[history::Record]) {
    let mut points: Vec<(String, u32)> = records.iter().map(|r| (r.mode.clone(), r.n)).collect();
    points.sort_unstable();
    points.dedup();
    let at = |value: &str, mode: &str, n: u32| -> Option<f64> {
        let label = spec.label(value);
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
    write_chart(args, &records)
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
            "pp", "n_tok", "cached", "prompt_ms", "best", "mean ± sd"
        );
        println!("{}", "-".repeat(70));
    }

    for &len in &args.pp {
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<PrefillSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_prefill_once(client, &args.url, &prompt, &args.model, false)?;
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

    for &n in &args.streams {
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
            let tokens: u32 = results.iter().map(|s| s.gen_tokens).sum();
            if tokens == 0 || wall <= 0.0 {
                anyhow::bail!("no tokens generated across {n} streams");
            }
            aggregate.push(f64::from(tokens) / wall);
            total_tokens = tokens;
        }
        let best = aggregate.iter().cloned().fold(0.0_f64, f64::max);
        let mean = aggregate.iter().sum::<f64>() / aggregate.len() as f64;
        let var =
            aggregate.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / aggregate.len() as f64;
        let sd = var.sqrt();

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "streams": n,
                    "aggregate_tok_per_s_best": best,
                    "aggregate_tok_per_s_mean": mean,
                    "aggregate_tok_per_s_sd": sd,
                    "per_stream_tok_per_s": mean / f64::from(n),
                    "tokens": total_tokens,
                })
            );
        } else {
            println!(
                "{:>8} | {:>6.2} ± {:>4.2} | {:>11.2} | {:>8}",
                n,
                mean,
                sd,
                mean / f64::from(n),
                total_tokens
            );
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "tg".to_string(),
            n,
            best,
            mean,
            sd,
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

    for &depth in &args.depths {
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
            if s.gen_tokens == 0 {
                anyhow::bail!("no tokens generated at depth {depth}; cannot divide by zero");
            }
            per_token.push((after - before).max(0.0) / f64::from(s.gen_tokens) * 1000.0);
            rate = s.tok_per_s();
            reported = (primed.prompt_tokens, primed.processed_tokens());
        }
        let best = per_token.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = per_token.iter().sum::<f64>() / per_token.len() as f64;
        let var =
            per_token.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / per_token.len() as f64;
        let sd = var.sqrt();

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "depth": depth,
                    "prompt_tokens": reported.0,
                    "prefilled_tokens": reported.1,
                    "cpu_ms_per_token_best": best,
                    "cpu_ms_per_token_mean": mean,
                    "cpu_ms_per_token_sd": sd,
                    "tok_per_s": rate,
                })
            );
        } else {
            println!(
                "{:>8} | {:>7} | {:>9} | {:>6.3} ± {:>4.3} | {:>8.2}",
                depth, reported.0, reported.1, mean, sd, rate
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
            best,
            mean,
            sd,
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
            "added", "n_tok", "cached", "processed", "prompt_ms", "best", "mean ± sd"
        );
        println!("{}", "-".repeat(82));
    }

    let base = build_prompt(args.pp_continue_base);
    for &added in &args.pp_continue {
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
        let best = rates.iter().cloned().fold(0.0_f64, f64::max);
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
        let sd = var.sqrt();
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
                    "tok_per_s_best": best,
                    "tok_per_s_mean": mean,
                    "tok_per_s_sd": sd,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>7} | {:>9} | {:>9.1} | {:>8.2} | {:>8.2} ± {:>5.2}",
                added, s.prompt_tokens, s.cached_tokens, processed, s.prompt_ms, best, mean, sd
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
            best,
            mean,
            sd,
        });
    }
    Ok(records)
}

/// The embedding sweep: one row per requested prompt length.
///
/// Reported as tok/s so the number lines up with the `pp` column of a
/// generative model — an embedding forward pass is prompt processing without
/// the decode that follows it, and that is the comparison worth making
/// (`embeddinggemma-300M` against `llama-server` on the same file).
fn run_embed(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>8} | {:>7} | {:>9} | {:>8} | {:>16}",
            "embed", "n_tok", "wall_ms", "best", "mean ± sd"
        );
        println!("{}", "-".repeat(60));
    }

    for &len in &args.embed {
        let prompt = build_prompt(len);
        let mut rates = Vec::new();
        let mut last: Option<EmbedSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_embed_once(client, &args.url, &prompt, &args.model)?;
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
                    "embed": len,
                    "prompt_tokens": s.prompt_tokens,
                    "wall_ms": s.wall_ms,
                    "tok_per_s_best": best,
                    "tok_per_s_mean": mean,
                    "tok_per_s_sd": sd,
                    "server_reported": s.server_reported,
                })
            );
        } else if s.server_reported {
            println!(
                "{:>8} | {:>7} | {:>9.1} | {:>8.2} | {:>8.2} ± {:>5.2}",
                len, s.prompt_tokens, s.wall_ms, best, mean, sd
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
                best,
                mean,
                sd,
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
/// optional, and llama-server does not have it at all.
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
    let gpus = gpu_clock_states();
    // `null` from llama-server and from orangu-server on a non-`wgpu`
    // backend; a full `VulkanBackend::tuning_report` otherwise.
    let gpu_tuning = props.as_ref().and_then(|p| p.get("gpu")).cloned();

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
                // Verbatim, not summarized: the JSON stream is what gets
                // archived next to a throughput number, and a run measured
                // six months ago is only re-interpretable if the whole
                // configuration travelled with it.
                "gpu_tuning": gpu_tuning,
            })
        );
    } else {
        println!("orangu-bench → {}", args.url);
        println!("  model    {model}");
        println!("  backend  {backend}");
        for line in format_gpu_tuning(gpu_tuning.as_ref()) {
            println!("  {line}");
        }
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
    }
    Environment {
        props: props.unwrap_or(serde_json::Value::Null),
        gpus,
        // Filled in by the caller after the workload — this function runs
        // before it. Reported here as `Null` rather than as an absent field so
        // a bundle's shape does not depend on whether timestamps were on.
        gpu_timings: serde_json::Value::Null,
    }
}

/// The `gpu` block of `/props` (`VulkanBackend::tuning_report`) as header
/// lines, or nothing at all when the server did not report one — llama-server
/// never does, and neither does orangu-server on a CPU/CUDA/OpenCL/ROCm
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

/// The GPU timestamp breakdown, as one line under the numbers it explains.
///
/// Per-step means rather than window totals: a window's totals depend on how
/// many tokens were generated in it, so two configurations measured with
/// different `--gen` would look different for a reason that has nothing to do
/// with either. The mean is the comparable figure.
///
/// Nothing at all when the server reports no timings — llama-server has no
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A server that reports no `gpu` block gets no header lines — not a row
    /// of `?`s. llama-server is the case that matters: this tool is pointed at
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

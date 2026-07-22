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

//! `orangu-bench` — a small developer tool that measures **decode
//! throughput** (token-generation tok/s) of a running OpenAI-compatible
//! server over HTTP, at one or more context depths. It is the HTTP-client
//! analogue of `llama-bench`'s `tg` test: point it at any server that speaks
//! `POST /v1/completions` with SSE streaming — **both `orangu-server` and
//! `llama-server`** do — and it reports steady-state decode tok/s, isolating
//! generation from prompt processing (it times from the first streamed token
//! to the last, so prefill/TTFT is excluded from the rate).
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
//! ```

use std::io::{BufRead, BufReader};
use std::time::Instant;

use clap::Parser;

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
    let resp = client
        .post(&endpoint)
        .json(&body)
        .send()
        .map_err(|_| anyhow::anyhow!("Error sending request to url ({endpoint})"))?;
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
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    // Warmup first (it also validates the connection), so a failure here prints
    // just the clean error above rather than a header followed by an error.
    if !args.no_warmup {
        let p = build_prompt(0);
        run_once(&client, &args.url, &p, 8, &args.model)?;
    }

    if args.curve > 0 {
        return run_curve(&client, args);
    }

    if !args.json {
        println!("orangu-bench → {}", args.url);
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
            let s = run_once(&client, &args.url, &prompt, args.n_gen, &args.model)?;
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
    }

    Ok(())
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

    let resp = client
        .post(&endpoint)
        .json(&body)
        .send()
        .map_err(|_| anyhow::anyhow!("Error sending request to url ({endpoint})"))?;
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
        println!(
            "orangu-bench → {} (curve: {} tokens, bucket {})",
            args.url, n, args.bucket
        );
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

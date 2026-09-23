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

//! `--image`: how long a picture takes, and where in the three models the
//! time goes.
//!
//! A picture is three models in a row — the prompt through the text
//! encoder, the latent through the diffusion transformer once per step (twice
//! under guidance), and the latent through the VAE — and a user who typed
//! "Create an image of a cat" into the console and waited sees one number:
//! the wait. This mode takes the same request through the same endpoint
//! (`POST /v1/images/generations`, streamed) and splits the wait three ways
//! from the server's own phase timings, so a slow picture is a slow *phase*,
//! and — with `--flamegraph` — a slow phase is a hot function.
//!
//! The rate recorded beside the split is the transformer's, because that is
//! where a picture's time goes at any size worth generating:
//! **image-token passes per second** — the number of latent tokens (one per
//! 16×16 pixel patch) times the number of transformer passes (steps, doubled
//! by guidance), over the denoising window. It is to a picture what
//! prompt-processing tok/s is to a chat turn: how fast the model chews
//! through positions, independent of how many the request asked for — so a
//! 256-pixel run at two steps and a 1024-pixel run at fifty can be put on the
//! same chart.
//!
//! Steps default to two rather than the server's fifty, and sizes are what
//! `--image` lists: every step is the same work, so two of them say what
//! fifty would at a fortieth of the wait — and profiling the fiftieth step
//! finds nothing the second did not.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use crate::{Args, Stats, history};

/// The default prompt: the one the question was asked about.
pub const DEFAULT_PROMPT: &str = "Create an image of a cat";

/// One picture, timed. Wall times from this side; the phase split from the
/// server's `timings`, which is the only place the phases are visible.
pub struct ImageSample {
    pub width: u32,
    pub height: u32,
    /// Denoising steps the server ran.
    pub steps: u32,
    /// Whether the server ran two transformer passes per step (guidance).
    pub guided: bool,
    /// The prompts through the text encoder, seconds.
    pub encode_s: f64,
    /// The denoising loop, seconds.
    pub denoise_s: f64,
    /// The VAE and the file format, seconds.
    pub decode_s: f64,
    /// Send to `[DONE]`, seconds — what the user waits.
    pub total_s: f64,
    /// Time from the request to the first progress event: the encode phase
    /// plus the first step, as seen from here. A cross-check on the server's
    /// own `encode_s` that needs no server cooperation at all.
    pub first_progress_s: f64,
    /// The transformer passes by operation class, seconds — the server's
    /// `timings.stages`, in the order it is printed. Empty on a server that
    /// reports none.
    pub stages: Vec<(&'static str, f64)>,
    /// What the server said the denoising loop would take before it started
    /// — the step-0 progress event's `seconds_per_step × steps` — or `None`
    /// from a server that does not announce one. Printed beside `denoise_s`
    /// so the estimate a console shows can be held to the measurement.
    pub announced_s: Option<f64>,
}

/// The stage keys the server reports, and the word each prints under.
const STAGE_KEYS: [(&str, &str); 7] = [
    ("modulation_ms", "modulation"),
    ("qkv_ms", "qkv"),
    ("attention_ms", "attention"),
    ("out_ms", "out"),
    ("mlp_ms", "mlp"),
    ("other_ms", "other"),
    ("lora_ms", "lora"),
];

impl ImageSample {
    /// Latent tokens: one per 16×16 pixel patch (the VAE's 8× compression
    /// times the transformer's 2×2 patch).
    pub fn tokens(&self) -> u32 {
        (self.width / 16) * (self.height / 16)
    }

    /// Transformer passes: one per step, two under guidance.
    pub fn passes(&self) -> u32 {
        self.steps * if self.guided { 2 } else { 1 }
    }

    /// Image-token passes per second over the denoising window — see the
    /// module doc for why this is the recorded rate.
    pub fn token_passes_per_s(&self) -> f64 {
        if self.denoise_s > 0.0 {
            f64::from(self.tokens() * self.passes()) / self.denoise_s
        } else {
            0.0
        }
    }

    /// Seconds per step: the number the console's own progress bar shows,
    /// for reading one against the other.
    pub fn seconds_per_step(&self) -> f64 {
        if self.steps > 0 {
            self.denoise_s / f64::from(self.steps)
        } else {
            0.0
        }
    }

    /// The stage split as one line: each stage's share of the transformer
    /// time, largest first — `mlp 58%  qkv 21%  …` — or nothing to print.
    pub fn stage_line(&self) -> Option<String> {
        let total: f64 = self.stages.iter().map(|(_, s)| s).sum();
        if total <= 0.0 {
            return None;
        }
        let mut stages = self.stages.clone();
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        Some(
            stages
                .iter()
                .map(|(name, s)| format!("{name} {:.0}%", 100.0 * s / total))
                .collect::<Vec<_>>()
                .join("  "),
        )
    }
}

/// One `POST /v1/images/generations`, streamed, timed.
///
/// `cfg_scale: None` leaves guidance to the server's default, which is how
/// the console's own request arrives — the bench reproduces the wait it was
/// asked about unless told to change it. `seed` is fixed so every repetition
/// draws the same noise: the picture is not the point, and a varying one
/// would only add a question.
// Eight parameters: the request's own fields, each from a different flag;
// a struct would only move the same names one line down.
#[allow(clippy::too_many_arguments)]
pub fn run_image_once(
    client: &reqwest::blocking::Client,
    url: &str,
    prompt: &str,
    side: u32,
    steps: u32,
    cfg_scale: Option<f64>,
    init: Option<&str>,
    model: &Option<String>,
) -> anyhow::Result<ImageSample> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "size": format!("{side}x{side}"),
        "steps": steps,
        "seed": 1,
        "stream": true,
        "response_format": "b64_json",
    });
    if let Some(cfg) = cfg_scale {
        body["cfg_scale"] = serde_json::json!(cfg);
    }
    if let Some(init) = init {
        body["image"] = serde_json::Value::String(init.to_string());
    }
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m.clone());
    }

    let endpoint = format!("{url}/v1/images/generations");
    let t0 = Instant::now();
    let resp = client
        .post(&endpoint)
        .json(&body)
        .send()
        .map_err(|_| anyhow::anyhow!("Error sending request to url ({endpoint})"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("server returned HTTP {status}: {}", text.trim());
    }

    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut first_progress: Option<Instant> = None;
    let mut announced_s: Option<f64> = None;
    let mut completed: Option<serde_json::Value> = None;
    loop {
        line.clear();
        let read = match reader.read_line(&mut line) {
            Ok(read) => read,
            Err(_) => break,
        };
        if read == 0 {
            break;
        }
        let Some(payload) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(err) = v.get("error") {
            anyhow::bail!("server reported: {err}");
        }
        match v.get("type").and_then(serde_json::Value::as_str) {
            Some("image_generation.progress") => {
                let step = v
                    .get("step")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1);
                if step == 0 {
                    // The announcement, not a step: the estimate of the
                    // whole loop, kept apart from the first real progress.
                    announced_s = v
                        .get("seconds_per_step")
                        .and_then(serde_json::Value::as_f64)
                        .zip(v.get("steps").and_then(serde_json::Value::as_f64))
                        .map(|(per_step, steps)| per_step * steps);
                } else {
                    first_progress.get_or_insert_with(Instant::now);
                }
            }
            Some("image_generation.completed") => completed = Some(v),
            _ => {}
        }
    }
    let total_s = t0.elapsed().as_secs_f64();
    let completed = completed.ok_or_else(|| {
        anyhow::anyhow!("the stream ended without an image_generation.completed event")
    })?;

    let timings = completed.get("timings").ok_or_else(|| {
        anyhow::anyhow!(
            "the completed event carries no `timings`: the server is older than this mode, \
             and cannot say where the time went"
        )
    })?;
    let ms = |key: &str| -> f64 {
        timings
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            / 1000.0
    };
    let stages = timings
        .get("stages")
        .map(|st| {
            STAGE_KEYS
                .iter()
                .map(|(key, name)| {
                    (
                        *name,
                        st.get(key)
                            .and_then(serde_json::Value::as_f64)
                            .unwrap_or(0.0)
                            / 1000.0,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let (width, height) = completed
        .get("size")
        .and_then(serde_json::Value::as_str)
        .and_then(|s| s.split_once('x'))
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .unwrap_or((side, side));
    let steps = completed
        .get("steps")
        .and_then(serde_json::Value::as_u64)
        .map_or(steps, |s| s as u32);
    Ok(ImageSample {
        width,
        height,
        steps,
        // What the server did rather than what was asked: with no
        // `cfg_scale` in the request the server's default decides, and the
        // reply does not say — so it is read off the request, or else the
        // server's default from `/props`.
        guided: cfg_scale.unwrap_or_else(|| image_props(client, url).1) > 1.0,
        encode_s: ms("encode_ms"),
        denoise_s: ms("steps_ms"),
        decode_s: ms("decode_ms"),
        total_s,
        first_progress_s: first_progress
            .map(|t| t.duration_since(t0).as_secs_f64())
            .unwrap_or(total_s),
        stages,
        announced_s,
    })
}

/// The smallest picture there is, once: a warmup through the endpoint the
/// run will use, so the pipeline's threads exist before `perf` attaches and
/// the first measured picture pays no first-use cost. One side of the
/// server's `size_unit` (16 pixels, 32 for Qwen-Image 2.1) is the fewest
/// latent tokens; one step, unguided, is one transformer pass over them.
pub fn warmup(
    client: &reqwest::blocking::Client,
    url: &str,
    model: &Option<String>,
) -> anyhow::Result<()> {
    let side = image_props(client, url).0;
    run_image_once(client, url, DEFAULT_PROMPT, side, 1, Some(1.0), None, model).map(|_| ())
}

/// The server's picture `size_unit` and default guidance, from `/props`
/// (16 and guided when it does not say — a server from before either).
pub fn image_props(client: &reqwest::blocking::Client, url: &str) -> (u32, f64) {
    let image = client
        .get(format!("{}/props", url.trim_end_matches('/')))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| v.get("image").cloned())
        .unwrap_or_default();
    let unit = image
        .get("size_unit")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(16) as u32;
    let cfg = image
        .pointer("/defaults/cfg_scale")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(4.0);
    (unit, cfg)
}

/// `--image`: one row per size, each the best of `--reps` pictures.
pub fn run_image(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let mut records = Vec::new();
    if !args.json {
        println!(
            "{:>9} | {:>6} | {:>5} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>16}",
            "image",
            "tokens",
            "steps",
            "encode_s",
            "step_s",
            "decode_s",
            "total_s",
            "best",
            "mean ± sd(n-1)"
        );
        println!("{}", "-".repeat(104));
    }

    // The attachment as a data URL, read once.
    let init = match &args.image_init {
        Some(path) => {
            let bytes = std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("reading --image-init {}: {e}", path.display()))?;
            let mime = match path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("jpg" | "jpeg") => "image/jpeg",
                Some("gif") => "image/gif",
                Some("webp") => "image/webp",
                Some("svg") => "image/svg+xml",
                _ => "image/png",
            };
            use base64::Engine as _;
            Some(format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            ))
        }
        None => None,
    };
    for (point, &side) in args.image.iter().enumerate() {
        if point > 0 {
            args.settle();
        }
        let mut rates = Vec::new();
        let mut last: Option<ImageSample> = None;
        for _ in 0..args.reps.max(1) {
            let s = run_image_once(
                client,
                &args.url,
                &args.image_prompt,
                side,
                args.image_steps,
                args.image_cfg,
                init.as_deref(),
                &args.model,
            )?;
            rates.push(s.token_passes_per_s());
            last = Some(s);
        }
        let stats = Stats::of(&rates, false);
        let s = last.expect("at least one rep ran");

        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "image": side,
                    "width": s.width,
                    "height": s.height,
                    "tokens": s.tokens(),
                    "steps": s.steps,
                    "guided": s.guided,
                    "passes": s.passes(),
                    "encode_s": s.encode_s,
                    "denoise_s": s.denoise_s,
                    "seconds_per_step": s.seconds_per_step(),
                    "decode_s": s.decode_s,
                    "total_s": s.total_s,
                    "first_progress_s": s.first_progress_s,
                    "announced_s": s.announced_s,
                    "stages_s": s.stages.iter().map(|(name, secs)| (name.to_string(), serde_json::json!(secs))).collect::<serde_json::Map<_, _>>(),
                    "token_passes_per_s_best": stats.best,
                    "token_passes_per_s_mean": stats.mean,
                    "token_passes_per_s_sd": stats.sd,
                    "token_passes_per_s_sd_sample": stats.sd_sample,
                })
            );
        } else {
            println!(
                "{:>4}x{:<4} | {:>6} | {:>3}{:>2} | {:>8.1} | {:>8.1} | {:>8.1} | {:>8.1} | {:>8.1} | {:>8.1} ± {}",
                s.width,
                s.height,
                s.tokens(),
                s.steps,
                if s.guided { "x2" } else { "" },
                s.encode_s,
                s.seconds_per_step(),
                s.decode_s,
                s.total_s,
                stats.best,
                stats.mean,
                stats.plus_minus(5, 1)
            );
            if let Some(line) = s.stage_line() {
                println!("{:>9} | transformer: {line}", "");
            }
            if let Some(announced) = s.announced_s {
                println!(
                    "{:>9} | announced {announced:.1} s for the steps before starting, took {:.1} s",
                    "", s.denoise_s
                );
            }
        }

        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "image".to_string(),
            n: side,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }

    if !args.json {
        println!(
            "\n  rate: image-token passes per second over the denoising loop \
             (latent tokens x transformer passes / step_s x steps);\n  \
             encode_s is the prompts through the text encoder, decode_s the VAE and the file"
        );
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(side: u32, steps: u32, guided: bool, denoise_s: f64) -> ImageSample {
        ImageSample {
            width: side,
            height: side,
            steps,
            guided,
            encode_s: 1.0,
            denoise_s,
            decode_s: 1.0,
            total_s: denoise_s + 2.0,
            first_progress_s: 1.5,
            stages: vec![("mlp", 6.0), ("qkv", 3.0), ("attention", 1.0)],
            announced_s: None,
        }
    }

    /// The stage line is shares of the transformer's own total, largest
    /// first, and nothing at all from a server that reports no stages.
    #[test]
    fn the_stage_line_orders_shares_largest_first() {
        let s = sample(256, 2, false, 40.0);
        assert_eq!(
            s.stage_line().as_deref(),
            Some("mlp 60%  qkv 30%  attention 10%")
        );
        let mut bare = sample(256, 2, false, 40.0);
        bare.stages.clear();
        assert_eq!(bare.stage_line(), None);
    }

    /// The rate is per token *pass*, so a guided run at the same speed per
    /// pass scores the same as an unguided one — twice the passes over twice
    /// the time — and a bigger picture with the same seconds per token
    /// scores the same as a small one.
    #[test]
    fn the_rate_counts_transformer_passes_over_latent_tokens() {
        let unguided = sample(256, 2, false, 40.0);
        assert_eq!(unguided.tokens(), 256);
        assert_eq!(unguided.passes(), 2);
        assert!((unguided.token_passes_per_s() - 12.8).abs() < 1e-9);
        assert!((unguided.seconds_per_step() - 20.0).abs() < 1e-9);

        let guided = sample(256, 2, true, 80.0);
        assert_eq!(guided.passes(), 4);
        assert!((guided.token_passes_per_s() - 12.8).abs() < 1e-9);

        let large = sample(512, 2, false, 160.0);
        assert_eq!(large.tokens(), 1024);
        assert!((large.token_passes_per_s() - 12.8).abs() < 1e-9);

        assert_eq!(sample(256, 0, false, 0.0).token_passes_per_s(), 0.0);
        assert_eq!(sample(256, 0, false, 0.0).seconds_per_step(), 0.0);
    }
}

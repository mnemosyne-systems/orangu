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

//! `--kl`: how far a server's next-token distributions sit from a
//! reference's, teacher-forced over the same tokens.
//!
//! The first run against a path that does not exist yet saves the server's
//! distributions there as the reference. Every later run sends the
//! reference's exact tokens to `/logits` and compares position by position:
//! the KL divergence from the reference (mean, 99th percentile, worst) and
//! how often the most likely token is the same. A precision change — a KV
//! storage format, a kernel's reduction order, a weight layout — is judged
//! by these, not by whether a few generated replies look alright.
//!
//! Each side keeps only its top candidates, so the divergence is taken over
//! the reference's candidates plus one bucket for everything else. A
//! candidate the other side did not keep is given that side's smallest kept
//! probability, an upper bound on what it held.

use std::path::Path;

use crate::{Args, history};

/// A saved reference, or one server's answer.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Distributions {
    pub tokens: Vec<u32>,
    pub rows: usize,
    pub positions: Vec<Vec<(u32, f32)>>,
    pub seconds: f64,
}

/// One position compared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub kl: f64,
    pub same_top: bool,
}

/// KL(reference ‖ other) over the reference's candidates and the rest.
pub fn compare(reference: &[(u32, f32)], other: &[(u32, f32)]) -> Position {
    let floor = other
        .iter()
        .map(|&(_, lp)| f64::from(lp).exp())
        .fold(f64::INFINITY, f64::min);
    let floor = if floor.is_finite() { floor } else { 1e-12 };
    let mut kl = 0.0;
    let (mut p_kept, mut q_kept) = (0.0, 0.0);
    for &(id, lp) in reference {
        let p = f64::from(lp).exp();
        let q = other
            .iter()
            .find(|c| c.0 == id)
            .map_or(floor, |c| f64::from(c.1).exp());
        p_kept += p;
        q_kept += q;
        if p > 0.0 {
            kl += p * (p / q.max(1e-12)).ln();
        }
    }
    let p_rest = (1.0 - p_kept).max(0.0);
    let q_rest = (1.0 - q_kept).max(1e-12);
    if p_rest > 1e-9 {
        kl += p_rest * (p_rest / q_rest).ln();
    }
    Position {
        kl: kl.max(0.0),
        same_top: reference.first().map(|c| c.0) == other.first().map(|c| c.0),
    }
}

/// The text a reference is taken over: the chat fixture's system prompt
/// and turns, which are mostly code and tool definitions.
/// Set when a run's measurement crossed its threshold — read after the run
/// is recorded, so `--fail-on-regression` exits `2` with the rows on file.
static OVER_THRESHOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Marks this run as over its threshold.
pub fn note_over_threshold() {
    OVER_THRESHOLD.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a measurement of this run crossed its threshold.
pub fn over_threshold() -> bool {
    OVER_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
}

/// The weight type a model label names: the tag after `:`, or the last
/// `-`-separated part of a file name that reads as one (`Q4_K_M`,
/// `UD-Q6_K_XL`, `Q8_0`, `IQ3_XXS`, `BF16`).
pub fn quant_of(label: &str) -> Option<String> {
    let name = label.rsplit('/').next().unwrap_or(label);
    let name = name.strip_suffix(".gguf").unwrap_or(name);
    let tag = match name.rsplit_once(':') {
        Some((_, tag)) => tag.to_string(),
        None => {
            let parts: Vec<&str> = name.split('-').collect();
            let at = parts.iter().rposition(|p| looks_like_quant(p))?;
            if at > 0 && parts[at - 1].eq_ignore_ascii_case("UD") {
                format!("UD-{}", parts[at])
            } else {
                parts[at].to_string()
            }
        }
    };
    looks_like_quant(tag.trim_start_matches("UD-")).then_some(tag.to_ascii_uppercase())
}

fn looks_like_quant(part: &str) -> bool {
    let p = part.to_ascii_uppercase();
    matches!(p.as_str(), "F32" | "F16" | "BF16")
        || ((p.starts_with('Q') || p.starts_with("IQ"))
            && p.chars()
                .nth(if p.starts_with("IQ") { 2 } else { 1 })
                .is_some_and(|c| c.is_ascii_digit()))
}

/// The drift a weight type is allowed against its reference: the most mean
/// KL, and the fewest positions whose top token may agree, in percent.
///
/// Set from the baseline (PERF-BASELINE.md §1): every architecture and type
/// measured drifted under 0.007 nats with at least 98.4% agreement against
/// one-row decode, so 4-bit and wider types are held to 0.01 and 98%, and the
/// 3- and 2-bit types, whose rounding is coarser, to twice the KL. Floats
/// leave only the order of summation, and are held tighter.
pub fn drift_threshold(quant: Option<&str>) -> (f64, f64) {
    let q = quant.unwrap_or("").trim_start_matches("UD-");
    match q {
        "F32" | "F16" | "BF16" => (0.002, 99.0),
        _ if q.starts_with("Q2")
            || q.starts_with("Q3")
            || q.starts_with("IQ1")
            || q.starts_with("IQ2")
            || q.starts_with("IQ3") =>
        {
            (0.02, 97.0)
        }
        _ => (0.01, 98.0),
    }
}

fn text(args: &Args) -> anyhow::Result<String> {
    if let Some(path) = &args.kl_text {
        return std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()));
    }
    let fixture = crate::chat::Fixture::load(args.chat_fixture.as_deref())?;
    let mut text = fixture.system;
    for turn in fixture.turns {
        text.push_str("\n\n");
        text.push_str(&turn);
    }
    Ok(text)
}

fn fetch(
    client: &reqwest::blocking::Client,
    args: &Args,
    tokens: Vec<u32>,
) -> anyhow::Result<Distributions> {
    let body = serde_json::json!({
        "tokens": tokens,
        "rows": args.kl_rows,
        "top": args.kl_top,
    });
    let resp = crate::post_with_one_retry(client, &format!("{}/logits", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    Ok(resp.json()?)
}

fn tokens_of(
    client: &reqwest::blocking::Client,
    args: &Args,
    text: &str,
) -> anyhow::Result<Vec<u32>> {
    let body = serde_json::json!({ "content": text, "add_special": true });
    let resp = crate::post_with_one_retry(client, &format!("{}/tokenize", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let v: serde_json::Value = resp.json()?;
    let mut tokens: Vec<u32> = serde_json::from_value(v["tokens"].clone())?;
    tokens.truncate(args.kl_tokens as usize);
    Ok(tokens)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

pub fn run_kl(
    client: &reqwest::blocking::Client,
    args: &Args,
    path: &Path,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    if !path.exists() {
        let tokens = tokens_of(client, args, &text(args)?)?;
        let reference = fetch(client, args, tokens)?;
        std::fs::write(path, serde_json::to_string(&reference)?)?;
        println!(
            "reference: {} positions, {} rows a pass, {:.2} ms a position — saved to {}",
            reference.positions.len(),
            reference.rows,
            reference.seconds * 1000.0 / reference.positions.len().max(1) as f64,
            path.display()
        );
        return Ok(Vec::new());
    }
    let reference: Distributions = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let other = fetch(client, args, reference.tokens.clone())?;
    let compared: Vec<Position> = reference
        .positions
        .iter()
        .zip(&other.positions)
        .map(|(r, o)| compare(r, o))
        .collect();
    let mut kls: Vec<f64> = compared.iter().map(|p| p.kl).collect();
    kls.sort_by(f64::total_cmp);
    let mean = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
    let agree = 100.0 * compared.iter().filter(|p| p.same_top).count() as f64
        / compared.len().max(1) as f64;
    let ms_ref = reference.seconds * 1000.0 / reference.positions.len().max(1) as f64;
    let ms = other.seconds * 1000.0 / other.positions.len().max(1) as f64;
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "positions": compared.len(),
                "rows": other.rows,
                "reference_rows": reference.rows,
                "kl_mean": mean,
                "kl_p99": percentile(&kls, 0.99),
                "kl_max": kls.last().copied().unwrap_or(0.0),
                "top1_agree_pct": agree,
                "ms_per_position": ms,
                "reference_ms_per_position": ms_ref,
            })
        );
    } else {
        println!(
            "positions | rows (ref) |   KL mean |    KL p99 |    KL max | top-1 same | ms/pos (ref)"
        );
        println!(
            "{:>9} | {:>4} ({:>3}) | {:>9.6} | {:>9.6} | {:>9.6} | {:>9.2}% | {:>6.2} ({:.2})",
            compared.len(),
            other.rows,
            reference.rows,
            mean,
            percentile(&kls, 0.99),
            kls.last().copied().unwrap_or(0.0),
            agree,
            ms,
            ms_ref
        );
    }
    // Judged against `--kl-max` / `--kl-min-top1` where given, else against
    // the served weight type's own threshold.
    let quant = crate::moe::get_json(client, &format!("{}/props", args.url))
        .get("model")
        .and_then(|m| m.as_str())
        .and_then(quant_of);
    let (max_kl, min_top1) = drift_threshold(quant.as_deref());
    let (max_kl, min_top1) = (
        args.kl_max.unwrap_or(max_kl),
        args.kl_min_top1.unwrap_or(min_top1),
    );
    let over = mean > max_kl || agree < min_top1;
    if over {
        note_over_threshold();
    }
    if !args.json {
        println!(
            "threshold  KL mean ≤ {max_kl} and top-1 ≥ {min_top1}% ({}): {}",
            quant.as_deref().unwrap_or("type unknown"),
            if over { "OVER" } else { "within" }
        );
    }
    let record = |mode: &str, value: f64| history::Record {
        date: history::today(),
        label: label.to_string(),
        mode: mode.to_string(),
        n: other.rows as u32,
        best: value,
        mean: value,
        sd: 0.0,
        sd_sample: None,
        device: None,
    };
    Ok(vec![record("kl", mean), record("top1", agree)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(p: &[(u32, f64)]) -> Vec<(u32, f32)> {
        p.iter().map(|&(id, p)| (id, p.ln() as f32)).collect()
    }

    #[test]
    fn identical_distributions_are_zero_apart() {
        let r = dist(&[(3, 0.6), (1, 0.3), (2, 0.05)]);
        let c = compare(&r, &r);
        assert!(c.kl < 1e-7, "{}", c.kl);
        assert!(c.same_top);
    }

    #[test]
    fn a_moved_winner_is_counted_and_costs_divergence() {
        let r = dist(&[(3, 0.6), (1, 0.3), (2, 0.05)]);
        let o = dist(&[(1, 0.6), (3, 0.3), (2, 0.05)]);
        let c = compare(&r, &o);
        assert!(!c.same_top);
        // 0.6 ln 2 + 0.3 ln 0.5
        assert!((c.kl - 0.3 * 2f64.ln()).abs() < 1e-6, "{}", c.kl);
    }

    #[test]
    fn a_candidate_the_other_side_dropped_takes_its_smallest_kept_probability() {
        let r = dist(&[(3, 0.5), (9, 0.4)]);
        let o = dist(&[(3, 0.5), (1, 0.1)]);
        let c = compare(&r, &o);
        let rest = 0.1f64;
        let q_rest = 1.0 - 0.5 - 0.1;
        let want = 0.4 * (0.4f64 / 0.1).ln() + rest * (rest / q_rest).ln();
        assert!((c.kl - want).abs() < 1e-6, "{} vs {want}", c.kl);
    }

    /// A label names its weight type in either spelling a server reports.
    #[test]
    fn the_weight_type_is_read_from_the_label() {
        assert_eq!(
            quant_of("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_of("/m/gemma-4-E2B-it-UD-Q6_K_XL.gguf").as_deref(),
            Some("UD-Q6_K_XL")
        );
        assert_eq!(quant_of("/m/model-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(
            quant_of("/m/model-IQ3_XXS.gguf").as_deref(),
            Some("IQ3_XXS")
        );
        assert_eq!(quant_of("/m/model-bf16.gguf").as_deref(), Some("BF16"));
        assert_eq!(quant_of("smoke"), None);
        assert_eq!(drift_threshold(Some("Q4_K_M")), (0.01, 98.0));
        assert_eq!(drift_threshold(Some("UD-Q2_K_XL")), (0.02, 97.0));
        assert_eq!(drift_threshold(Some("BF16")), (0.002, 99.0));
        assert_eq!(drift_threshold(None), (0.01, 98.0));
    }

    /// The estimator keeps the top `k` of each distribution and one bucket
    /// for the rest, so it never reads above the exact divergence: what the
    /// tail adds is merged away. Over a 32k vocabulary with a long tail it
    /// reads 8–42% low at 32 candidates and 5–26% low at 256, worst at the
    /// smallest divergences — the range drift is measured in — which is why
    /// `--kl-top` keeps 256.
    #[test]
    fn the_truncated_estimator_is_close_to_the_exact_divergence_at_256() {
        let n = 32_000usize;
        let zipf = |shift: f64, wobble: f64| -> Vec<f64> {
            let w: Vec<f64> = (0..n)
                .map(|i| {
                    let r = (i as f64 + 1.0 + shift).powf(-1.1);
                    r * (1.0 + wobble * ((i * 2_654_435_761) % 1000) as f64 / 1000.0)
                })
                .collect();
            let total: f64 = w.iter().sum();
            w.iter().map(|v| v / total).collect()
        };
        let top = |p: &[f64], k: usize| -> Vec<(u32, f32)> {
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            idx.truncate(k);
            idx.iter().map(|&i| (i as u32, p[i].ln() as f32)).collect()
        };
        let p = zipf(0.0, 0.0);
        let mut worst = [0.0f64; 2];
        for (case, wobble) in [0.05, 0.2, 0.5].into_iter().enumerate() {
            let q = zipf(0.3 * case as f64, wobble);
            let exact: f64 = p.iter().zip(&q).map(|(a, b)| a * (a / b).ln()).sum();
            for (slot, k) in [32usize, 256].into_iter().enumerate() {
                let estimate = compare(&top(&p, k), &top(&q, k)).kl;
                let bias = (estimate - exact).abs() / exact;
                assert!(
                    estimate <= exact * (1.0 + 1e-9),
                    "a lower bound: {estimate} > {exact}"
                );
                worst[slot] = worst[slot].max(bias);
            }
        }
        assert!(worst[1] < 0.3, "bias at 256 candidates: {:.3}", worst[1]);
        assert!(
            worst[0] > worst[1],
            "fewer candidates read further low: {worst:?}"
        );
        assert!(
            worst[1] <= worst[0] + 1e-9,
            "256 is no worse than 32: {worst:?}"
        );
    }
}

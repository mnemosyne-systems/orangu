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

//! The mixture-of-experts and page-cache side of a measurement: what the
//! server's `/moe-stats`, `/model-cache` and `/model-cache/drop` endpoints
//! report, and how it reaches the table, the history file and the bundle.
//!
//! Throughput alone cannot score work on a model too large to hold. A change
//! that moves fewer expert bytes and one that moves the same bytes faster
//! produce the same tok/s, and on a model streaming from disk neither is
//! visible over the I/O at all. So each measured point carries three
//! mechanism figures beside its rate:
//!
//! - **`moe_union`** — how many times the average expert's weights were
//!   *read* within one layer call, against reading each distinct expert once.
//!   `1.0` is the floor. Not the routing's own redundancy, which no
//!   implementation changes and which the server reports separately as
//!   `selection_ratio`: this is the implementation's, and it is the number a
//!   batch-union change moves.
//! - **`moe_mb`** — expert weight megabytes dequantized per token per MoE
//!   layer. Per *layer* rather than per token, so two models with different
//!   depths are comparable.
//! - **`moe_majflt`** — major faults per repetition: the only honest signal
//!   that weights came off the disk rather than out of the page cache.
//!
//! These are counts, not repeated measurements. Their `best` and `mean` are
//! the same number and their spread columns are empty — the counters are
//! exact, and the rate columns' "best of N repetitions" reading does not
//! apply to them. Each gets its own chart panel for that reason, as `cpu`
//! already does for having a different unit and direction.
//!
//! No hit-rate figure is reported: nothing measures one yet. It arrives with
//! the expert store, and reporting a zero in the meantime would make "not
//! implemented" indistinguishable from "never hit".

use crate::history;

/// Drain the server's accumulated MoE counters.
///
/// `GET /moe-stats` is read-and-reset, so a caller measuring a window calls
/// this **twice**: once before, whose result is discarded, and once after.
/// `Null` when the server has no such endpoint — another engine will not, and
/// that must degrade to "no mechanism figures" rather than to an error.
pub fn take_stats(client: &reqwest::blocking::Client, url: &str) -> serde_json::Value {
    get_json(client, &format!("{url}/moe-stats"))
}

/// How much of the model is in the page cache right now — the state a run
/// started from, which on a model larger than memory decides what the run
/// even measured.
pub fn take_residency(client: &reqwest::blocking::Client, url: &str) -> serde_json::Value {
    get_json(client, &format!("{url}/model-cache"))
}

/// Evict the model from the page cache, so the next repetition reads it from
/// the disk. Returns the server's before/after report, or `Null` if it has no
/// such endpoint.
pub fn drop_cache(client: &reqwest::blocking::Client, url: &str) -> serde_json::Value {
    client
        .post(format!("{url}/model-cache/drop"))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .unwrap_or(serde_json::Value::Null)
}

fn get_json(client: &reqwest::blocking::Client, url: &str) -> serde_json::Value {
    client
        .get(url)
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .unwrap_or(serde_json::Value::Null)
}

fn number(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut at = value;
    for key in path {
        at = at.get(key)?;
    }
    at.as_f64()
}

/// The `moe_*` rows for one measured point.
///
/// Empty when the server reported no MoE work in the window — a dense model,
/// or an engine without the endpoint. Recording zeros there would put "this
/// model has no experts" and "these experts moved nothing" on the same line.
///
/// `reps` divides the fault count only. The ratios and the per-token byte
/// figure are already rep-independent (each repetition routes the same prompt
/// the same way), whereas faults accumulate across repetitions and are only
/// meaningful per run.
pub fn records(stats: &serde_json::Value, label: &str, n: u32, reps: u32) -> Vec<history::Record> {
    let mut out = Vec::new();
    if number(stats, &["stats", "layer_calls"]).unwrap_or(0.0) <= 0.0 {
        return out;
    }
    let date = history::today();
    let mut push = |mode: &str, value: f64| {
        out.push(history::Record {
            date: date.clone(),
            label: label.to_string(),
            mode: mode.to_string(),
            n,
            best: value,
            mean: value,
            // A count is one exact observation, not a sample of repetitions.
            // `sd` has no way to say "undefined" so it says zero; `sd_sample`
            // does, and uses it — the same reading a single-repetition run
            // already writes into that column.
            sd: 0.0,
            sd_sample: None,
            device: None,
        });
    };

    if let Some(ratio) = number(stats, &["stats", "union_ratio"]) {
        push("moe_union", ratio);
    }
    if let (Some(bytes), Some(slots)) = (
        number(stats, &["stats", "bytes_dequantized"]),
        number(stats, &["stats", "token_slots"]),
    ) && slots > 0.0
    {
        push("moe_mb", bytes / slots / (1024.0 * 1024.0));
    }
    if let Some(faults) = number(stats, &["process", "major_faults_window"]) {
        push("moe_majflt", faults / f64::from(reps.max(1)));
    }
    // Absent unless the server actually asked the kernel where the weights
    // were (`ORANGU_EXPERT_RESIDENCY=1`). A row of zeros would say "never
    // hit" where the truth is "never looked".
    if let Some(rate) = number(stats, &["store", "hit_rate"]) {
        push("moe_hit", 100.0 * rate);
    }
    out
}

/// The `moe` line under a table, or `None` when there is nothing to say.
///
/// Deliberately reports `bytes_dequantized` against `bytes_unique` in the
/// same line: the first is what the engine read, the second what it would
/// have read having deduplicated the batch's experts, and the gap between
/// them is the size of the prize rather than a ratio to be taken on trust.
pub fn summary_line(stats: &serde_json::Value) -> Option<String> {
    if number(stats, &["stats", "layer_calls"]).unwrap_or(0.0) <= 0.0 {
        return None;
    }
    let gib = |path: &[&str]| number(stats, path).unwrap_or(0.0) / (1024.0 * 1024.0 * 1024.0);
    let mut line = format!(
        "  moe      union {:.2}x  ·  dequantized {:.1} GiB vs {:.1} GiB unique  ·  {:.1} experts/layer-call",
        number(stats, &["stats", "union_ratio"]).unwrap_or(0.0),
        gib(&["stats", "bytes_dequantized"]),
        gib(&["stats", "bytes_unique"]),
        number(stats, &["stats", "distinct_per_layer_call"]).unwrap_or(0.0),
    );
    if let Some(faults) = number(stats, &["process", "major_faults_window"]) {
        line.push_str(&format!("  ·  {faults:.0} major faults"));
    }
    Some(line)
}

/// The `cache` line: how much of the model was in RAM, and whether that is
/// even knowable here. `None` when the server does not report it.
pub fn residency_line(residency: &serde_json::Value) -> Option<String> {
    let bytes = number(residency, &["model_bytes"])?;
    if bytes <= 0.0 {
        return None;
    }
    let gib = bytes / (1024.0 * 1024.0 * 1024.0);
    Some(match number(residency, &["resident_bytes"]) {
        Some(resident) => format!(
            "  cache    model {gib:.1} GiB  ·  {:.1} GiB resident ({:.0}%)",
            resident / (1024.0 * 1024.0 * 1024.0),
            100.0 * resident / bytes,
        ),
        // Not "0% resident": the platform cannot measure it, and a run
        // reported as cold when nobody knows is worse than one reported as
        // unknown.
        None => format!("  cache    model {gib:.1} GiB  ·  residency not measurable here"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(
        layer_calls: u64,
        union_ratio: f64,
        bytes: u64,
        slots: u64,
        faults: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "stats": {
                "layer_calls": layer_calls,
                "token_slots": slots,
                "union_ratio": union_ratio,
                "bytes_dequantized": bytes,
                "bytes_unique": bytes / 2,
                "distinct_per_layer_call": 12.5,
            },
            "process": {"major_faults_window": faults},
        })
    }

    /// A dense model has no experts. Emitting zero-valued rows for it would
    /// draw a flat line on the union panel that reads as "no redundancy" —
    /// the exact conclusion a batch-union change is trying to earn.
    #[test]
    fn a_model_with_no_moe_layers_records_nothing() {
        let dense = stats(0, 0.0, 0, 0, 0);
        assert!(records(&dense, "x", 512, 3).is_empty());
        assert_eq!(summary_line(&dense), None);
    }

    #[test]
    fn a_missing_endpoint_records_nothing() {
        assert!(records(&serde_json::Value::Null, "x", 512, 3).is_empty());
        assert_eq!(summary_line(&serde_json::Value::Null), None);
        assert_eq!(residency_line(&serde_json::Value::Null), None);
    }

    #[test]
    fn each_metric_becomes_one_row_at_the_point_s_own_n() {
        let rows = records(&stats(10, 3.25, 4 * 1024 * 1024, 2, 900), "build-a", 512, 3);
        let modes: Vec<&str> = rows.iter().map(|r| r.mode.as_str()).collect();
        assert_eq!(modes, ["moe_union", "moe_mb", "moe_majflt"]);
        assert!(rows.iter().all(|r| r.n == 512 && r.label == "build-a"));
        assert_eq!(rows[0].best, 3.25);
        // 4 MiB over 2 token-slots.
        assert_eq!(rows[1].best, 2.0);
    }

    /// Faults accumulate over repetitions; the ratios do not. Dividing the
    /// wrong one would make a three-rep run look three times as redundant as
    /// a one-rep run of the same workload.
    #[test]
    fn only_the_fault_count_is_divided_by_the_repetition_count() {
        let one = records(&stats(10, 3.25, 4 * 1024 * 1024, 2, 900), "x", 512, 1);
        let three = records(&stats(10, 3.25, 4 * 1024 * 1024, 2, 900), "x", 512, 3);
        assert_eq!(one[0].best, three[0].best, "union ratio is rep-independent");
        assert_eq!(one[1].best, three[1].best, "MB per token-layer too");
        assert_eq!(one[2].best, 900.0);
        assert_eq!(three[2].best, 300.0);
    }

    /// These are exact counts, so the columns that mean "spread over
    /// repetitions" must not carry a number nobody measured.
    #[test]
    fn counts_carry_no_spread() {
        let rows = records(&stats(10, 3.25, 1024, 2, 0), "x", 128, 3);
        assert!(
            rows.iter()
                .all(|r| r.best == r.mean && r.sd == 0.0 && r.sd_sample.is_none())
        );
    }

    /// "Nothing is cached" and "this machine cannot tell you" must not print
    /// the same way — only one of them means the run was cold.
    #[test]
    fn unmeasurable_residency_says_so_instead_of_reporting_zero() {
        let unknown = serde_json::json!({"model_bytes": 1u64 << 30, "resident_bytes": null});
        let line = residency_line(&unknown).expect("a model size is enough to report");
        assert!(line.contains("not measurable"), "{line}");
        assert!(!line.contains("0%"), "{line}");

        let cold = serde_json::json!({"model_bytes": 1u64 << 30, "resident_bytes": 0u64});
        let line = residency_line(&cold).expect("zero is a measurement");
        assert!(line.contains("0.0 GiB resident (0%)"), "{line}");
    }

    #[test]
    fn the_summary_reports_both_byte_figures_not_only_their_ratio() {
        let line = summary_line(&stats(10, 3.25, 4 * (1 << 30), 2, 12)).expect("moe work happened");
        assert!(line.contains("union 3.25x"), "{line}");
        assert!(line.contains("4.0 GiB vs 2.0 GiB unique"), "{line}");
        assert!(line.contains("12 major faults"), "{line}");
    }
}

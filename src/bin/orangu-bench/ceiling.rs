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

//! The `ceiling` line under a decode point: the weight bytes the host reads
//! for one token, times the measured rate, against what this machine's
//! memory can be read at.
//!
//! A decode step on host-resident weights reads each of them once, so
//! `bytes × tok/s` is the read rate the step achieved. Near the measured
//! ceiling, a faster kernel cannot help and only fewer bytes or more tokens
//! per read can; far below it, the kernels or the turn around them are
//! where the time goes. Above it, more than one token came out of one read
//! of the weights — speculation.
//!
//! The ceiling is measured here, by reading a buffer much larger than any
//! cache from every core, so it is only meaningful when the server runs on
//! this machine: [`is_local`] decides that from the URL.

use std::sync::OnceLock;

/// The buffer the ceiling is read from: far past any last-level cache.
const PROBE_BYTES: usize = 512 << 20;
/// Passes over the buffer; the best one is the ceiling.
const PROBE_PASSES: usize = 4;

/// Whether `url` names a server on this machine.
pub fn is_local(url: &str) -> bool {
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("");
    host.starts_with("127.")
        || host.starts_with("localhost")
        || host.starts_with("[::1]")
        || host.starts_with("0.0.0.0")
}

/// This machine's memory read rate in GiB/s, measured once per run: every
/// core summing its own share of a [`PROBE_BYTES`] buffer, the best of
/// [`PROBE_PASSES`] passes.
pub fn host_read_gib_s() -> Option<f64> {
    static RATE: OnceLock<Option<f64>> = OnceLock::new();
    *RATE.get_or_init(|| {
        let words = PROBE_BYTES / std::mem::size_of::<u64>();
        let buffer: Vec<u64> = (0..words as u64).collect();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let share = words.div_ceil(threads);
        let mut best = f64::INFINITY;
        for _ in 0..PROBE_PASSES {
            let start = std::time::Instant::now();
            let sum: u64 = std::thread::scope(|scope| {
                let handles: Vec<_> = buffer
                    .chunks(share)
                    .map(|part| {
                        scope.spawn(move || part.iter().fold(0u64, |acc, &w| acc.wrapping_add(w)))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or(0))
                    .fold(0u64, u64::wrapping_add)
            });
            std::hint::black_box(sum);
            best = best.min(start.elapsed().as_secs_f64());
        }
        (best > 0.0 && best.is_finite()).then(|| PROBE_BYTES as f64 / best / (1u64 << 30) as f64)
    })
}

/// The weight bytes one token reads from host memory, from `/props`'s
/// `gpu` object: every device that reports no capacity of its own (the
/// host's share of a split) plus the routed experts kept in system RAM.
/// The second flag is whether any of it is routed experts, of which a
/// token reads only its selection — the figure is then an upper bound.
pub fn host_weight_bytes(gpu: Option<&serde_json::Value>) -> Option<(u64, bool)> {
    let gpu = gpu.filter(|g| !g.is_null())?;
    let devices = gpu.get("devices")?.as_array()?;
    let on_host: u64 = devices
        .iter()
        .filter(|d| d.get("total_bytes").is_none_or(serde_json::Value::is_null))
        .filter_map(|d| d.get("weights_bytes").and_then(serde_json::Value::as_u64))
        .sum();
    let experts = gpu
        .get("weights_host_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = on_host + experts;
    (total > 0).then_some((total, experts > 0))
}

/// The `ceiling` line for a decode point at `tok_s`, or `None` when no
/// weights are read from host memory or the rate is unknown.
pub fn line(
    bytes: u64,
    upper_bound: bool,
    tok_s: f64,
    ceiling_gib_s: Option<f64>,
) -> Option<String> {
    if bytes == 0 || tok_s.is_nan() || tok_s <= 0.0 {
        return None;
    }
    let gib = bytes as f64 / (1u64 << 30) as f64;
    let achieved = gib * tok_s;
    let mut line = format!(
        "  ceiling  host weights {gib:.2} GiB/token × {tok_s:.2} tok/s = {achieved:.1} GiB/s"
    );
    if let Some(ceiling) = ceiling_gib_s.filter(|c| *c > 0.0) {
        let share = 100.0 * achieved / ceiling;
        line.push_str(&format!(" of {ceiling:.1} GiB/s measured ({share:.0}%)"));
        // Past the ceiling only on bytes every token reads: a figure that
        // counts every routed expert overstates them, not the tokens.
        if share > 100.0 && !upper_bound {
            line.push_str(" — more than one token per read of the weights");
        }
    }
    if upper_bound {
        line.push_str(" — upper bound: a token reads only its routed experts");
    }
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_capacity_less_devices_and_routed_experts_count_as_host() {
        let gpu = serde_json::json!({
            "weights_host_bytes": 0,
            "devices": [
                {"name": "card", "total_bytes": 4u64 << 30, "weights_bytes": 3u64 << 30},
                {"name": "cpu", "total_bytes": null, "weights_bytes": 12u64 << 30},
            ],
        });
        assert_eq!(host_weight_bytes(Some(&gpu)), Some((12u64 << 30, false)));
        let moe = serde_json::json!({
            "weights_host_bytes": 10u64 << 30,
            "devices": [{"name": "card", "total_bytes": 4u64 << 30, "weights_bytes": 1u64 << 30}],
        });
        assert_eq!(host_weight_bytes(Some(&moe)), Some((10u64 << 30, true)));
        assert_eq!(host_weight_bytes(None), None);
    }

    #[test]
    fn the_line_names_the_share_and_a_rate_past_the_ceiling() {
        let l = line(10u64 << 30, false, 2.0, Some(40.0)).expect("a line");
        assert!(
            l.contains("10.00 GiB/token × 2.00 tok/s = 20.0 GiB/s of 40.0 GiB/s measured (50%)"),
            "{l}"
        );
        let over = line(10u64 << 30, false, 5.0, Some(40.0)).expect("a line");
        assert!(over.contains("more than one token per read"), "{over}");
        let moe = line(98u64 << 30, true, 0.6, Some(34.0)).expect("a line");
        assert!(!moe.contains("more than one token per read"), "{moe}");
        assert!(moe.contains("upper bound"), "{moe}");
        assert!(line(0, false, 2.0, Some(40.0)).is_none());
    }

    #[test]
    fn local_urls() {
        assert!(is_local("http://127.0.0.1:8100"));
        assert!(is_local("http://localhost:8100/v1"));
        assert!(!is_local("http://gpu-box:8100"));
    }
}

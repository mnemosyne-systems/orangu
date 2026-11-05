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

//! What a node's `/metrics` says about its workers: per worker, how long a
//! forward takes there (network and its whole subtree), how many bytes go
//! each way, how often it failed, and whether it is up; per node, how often
//! it planned and how often a sequence was rebuilt after a lost worker.
//!
//! Labelled by the worker's address as `[workers].workers` lists it — a
//! handful of values per node, so the cardinality stays what the config
//! says.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Forward round-trip buckets, in seconds: a decode step on a LAN is
/// milliseconds, a long prefill chunk through a CPU subtree is seconds.
const BOUNDS: [f64; 12] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0,
];

#[derive(Default)]
pub struct LinkMetrics {
    buckets: [AtomicU64; BOUNDS.len() + 1],
    sum_micros: AtomicU64,
    count: AtomicU64,
    sent: AtomicU64,
    received: AtomicU64,
    failures: AtomicU64,
}

impl LinkMetrics {
    pub fn observe_forward(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let index = BOUNDS
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(BOUNDS.len());
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn sent(&self, bytes: usize) {
        self.sent.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub fn received(&self, bytes: usize) {
        self.received.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub fn failed(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub struct NodeMetrics {
    pub plans: AtomicU64,
    pub recoveries: AtomicU64,
    pub takeovers: AtomicU64,
}

/// One worker's figures, for [`render`].
pub struct LinkSnapshot<'a> {
    pub addr: &'a str,
    pub up: bool,
    pub metrics: &'a LinkMetrics,
}

fn escape(label: &str) -> String {
    label.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The `orangu_server_worker*` families, in Prometheus text format.
pub fn render(node: &NodeMetrics, assigned: bool, links: &[LinkSnapshot<'_>]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# HELP orangu_server_workers_assigned Whether this node is working for a parent.\n\
         # TYPE orangu_server_workers_assigned gauge\n\
         orangu_server_workers_assigned {}\n\
         # HELP orangu_server_workers_plans_total Times this node planned its layers over its \
         workers.\n\
         # TYPE orangu_server_workers_plans_total counter\n\
         orangu_server_workers_plans_total {}\n\
         # HELP orangu_server_workers_recoveries_total Times a lost worker made this node plan \
         again mid-request.\n\
         # TYPE orangu_server_workers_recoveries_total counter\n\
         orangu_server_workers_recoveries_total {}\n\
         # HELP orangu_server_workers_takeovers_total Times a standby took over a lost \
         worker's layers.\n\
         # TYPE orangu_server_workers_takeovers_total counter\n\
         orangu_server_workers_takeovers_total {}\n\
         # HELP orangu_server_workers_kv_refusals_total Prompts refused because a node of the \
         tree has no memory for their KV cache.\n\
         # TYPE orangu_server_workers_kv_refusals_total counter\n\
         orangu_server_workers_kv_refusals_total {}\n",
        u8::from(assigned),
        node.plans.load(Ordering::Relaxed),
        node.recoveries.load(Ordering::Relaxed),
        node.takeovers.load(Ordering::Relaxed),
        super::room::refused(),
    ));
    if links.is_empty() {
        return out;
    }
    let family = |out: &mut String, name: &str, kind: &str, help: &str| {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
    };
    family(
        &mut out,
        "orangu_server_worker_up",
        "gauge",
        "Whether the connection to this worker is up.",
    );
    for link in links {
        out.push_str(&format!(
            "orangu_server_worker_up{{worker=\"{}\"}} {}\n",
            escape(link.addr),
            u8::from(link.up)
        ));
    }
    let name = "orangu_server_worker_forward_seconds";
    family(
        &mut out,
        name,
        "histogram",
        "Round trip of one forward through this worker's subtree, network included.",
    );
    for link in links {
        let label = escape(link.addr);
        let m = link.metrics;
        let mut cumulative = 0;
        for (i, bound) in BOUNDS.iter().enumerate() {
            cumulative += m.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{worker=\"{label}\",le=\"{bound}\"}} {cumulative}\n"
            ));
        }
        cumulative += m.buckets[BOUNDS.len()].load(Ordering::Relaxed);
        out.push_str(&format!(
            "{name}_bucket{{worker=\"{label}\",le=\"+Inf\"}} {cumulative}\n\
             {name}_sum{{worker=\"{label}\"}} {:.6}\n\
             {name}_count{{worker=\"{label}\"}} {cumulative}\n",
            m.sum_micros.load(Ordering::Relaxed) as f64 / 1e6
        ));
    }
    for (name, help, read) in [
        (
            "orangu_server_worker_sent_bytes_total",
            "Bytes sent to this worker.",
            (|m: &LinkMetrics| m.sent.load(Ordering::Relaxed)) as fn(&LinkMetrics) -> u64,
        ),
        (
            "orangu_server_worker_received_bytes_total",
            "Bytes received from this worker.",
            |m: &LinkMetrics| m.received.load(Ordering::Relaxed),
        ),
        (
            "orangu_server_worker_failures_total",
            "Requests to this worker that failed: unreachable, closed, or timed out.",
            |m: &LinkMetrics| m.failures.load(Ordering::Relaxed),
        ),
    ] {
        family(&mut out, name, "counter", help);
        for link in links {
            out.push_str(&format!(
                "{name}{{worker=\"{}\"}} {}\n",
                escape(link.addr),
                read(link.metrics)
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_worker_s_figures_render_with_its_label() {
        let link = LinkMetrics::default();
        link.observe_forward(Duration::from_millis(3));
        link.observe_forward(Duration::from_secs(20));
        link.sent(100);
        link.received(40);
        link.failed();
        let node = NodeMetrics::default();
        node.plans.fetch_add(2, Ordering::Relaxed);
        let text = render(
            &node,
            false,
            &[LinkSnapshot {
                addr: "b:8400",
                up: true,
                metrics: &link,
            }],
        );
        for line in [
            "orangu_server_workers_plans_total 2",
            "orangu_server_worker_up{worker=\"b:8400\"} 1",
            "orangu_server_worker_forward_seconds_bucket{worker=\"b:8400\",le=\"0.005\"} 1",
            "orangu_server_worker_forward_seconds_bucket{worker=\"b:8400\",le=\"+Inf\"} 2",
            "orangu_server_worker_forward_seconds_count{worker=\"b:8400\"} 2",
            "orangu_server_worker_sent_bytes_total{worker=\"b:8400\"} 100",
            "orangu_server_worker_received_bytes_total{worker=\"b:8400\"} 40",
            "orangu_server_worker_failures_total{worker=\"b:8400\"} 1",
        ] {
            assert!(text.contains(line), "missing {line} in\n{text}");
        }
        // Every family is declared once.
        assert_eq!(text.matches("# TYPE orangu_server_worker_up ").count(), 1);
    }
}

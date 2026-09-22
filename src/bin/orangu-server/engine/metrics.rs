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

//! What this server is doing, in a form an operator can graph and alert on.
//!
//! # Why histograms rather than more gauges
//!
//! `/metrics` used to be gauges: slots, busy slots, queue depth. Gauges answer
//! "what is happening right now", which is the wrong question for latency —
//! a mean would be dominated by whichever requests happened to be long, and a
//! last-value gauge reports one arbitrary request. What an operator actually
//! needs is a *distribution*, because the useful questions are all about the
//! tail: how slow is the slowest one in twenty, and did that change.
//!
//! Prometheus histograms answer that from cumulative bucket counts, so a
//! scrape carries the whole shape at a fixed cost and the quantile is computed
//! at query time over any window.
//!
//! # What is measured, and why these four
//!
//! - **queue wait** — arrival to holding a slot. The one number that separates
//!   "this server is overloaded" from "this request was expensive", which
//!   otherwise look identical from the outside and have opposite fixes.
//! - **time to first token** — arrival to the first token *produced*. What an
//!   interactive user waits through, and what capacity planning is about.
//! - **inter-token** — the gap between consecutive tokens, observed per token
//!   rather than once per request. A per-request mean would hide exactly the
//!   thing this is worth watching for: a stall in the middle of an otherwise
//!   fast answer.
//! - **request duration** — end to end, so the whole is visible next to its
//!   parts.
//!
//! Counters carry the totals a rate is taken from: requests by how they
//! finished, and tokens in and out.
//!
//! [`ProcessMetrics`], further down, covers the server process itself: build
//! and host identity, uptime, CPU and memory, and HTTP requests in flight.
//! Where Prometheus has a conventional name (`process_*`) it is used, so stock
//! dashboards work unmodified.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Latency bucket bounds, in seconds.
///
/// Spans milliseconds to a minute because this server's own range does: a
/// cached prefix answers in milliseconds, and a long prompt against a model
/// that overflows the device takes tens of seconds. Roughly 2–3x apart, which
/// is the spacing that keeps a quantile estimate honest without paying for
/// buckets nothing lands in.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Inter-token bucket bounds, in seconds.
///
/// A decode step is milliseconds to a second, so the latency bounds above
/// would put almost every observation in one bucket and report nothing. The
/// top bound is deliberately far above any healthy step: what this metric is
/// watched for is the step that *isn't* healthy.
const TOKEN_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];

/// A Prometheus histogram over fixed bounds.
///
/// Counts are per-bucket and made cumulative only when rendered, because a
/// cumulative *write* would mean touching every bucket at or above the
/// observed value on every observation — on the inter-token histogram that is
/// once per generated token.
///
/// Sums are microseconds in a `u64` rather than a float: `AtomicU64` is
/// lock-free everywhere this runs, and a float sum accumulated by
/// compare-and-swap would be both slower and non-deterministic in its last
/// digits across runs, which makes two measurements of the same workload
/// disagree for no reason.
pub struct Histogram {
    bounds: &'static [f64],
    /// One per bound, plus one for everything above the last.
    counts: Vec<AtomicU64>,
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: (0..bounds.len() + 1).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, value: Duration) {
        let seconds = value.as_secs_f64();
        // Linear scan: thirteen comparisons on a path that just finished a
        // forward pass. A binary search would be fewer instructions and
        // harder to read, for a saving that does not exist at this scale.
        let index = self
            .bounds
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(self.bounds.len());
        self.counts[index].fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add(value.as_micros() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// This histogram in Prometheus text format, under `name`.
    ///
    /// Buckets are cumulative, as the format requires: `le="0.1"` counts every
    /// observation at or below 0.1, not those between the previous bound and
    /// this one. Getting that wrong produces a histogram that renders and
    /// scrapes cleanly and gives nonsense quantiles.
    pub fn render(&self, name: &str, help: &str) -> String {
        let mut out = format!("# HELP {name} {help}\n# TYPE {name} histogram\n");
        let mut cumulative = 0u64;
        for (i, bound) in self.bounds.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.counts[self.bounds.len()].load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        // `_count` and the `+Inf` bucket must agree; they are two reads of the
        // same thing and a scraper checks it.
        out.push_str(&format!(
            "{name}_sum {:.6}\n{name}_count {}\n",
            self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.count.load(Ordering::Relaxed),
        ));
        out
    }

    #[cfg(test)]
    fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

/// How a request ended, as a `/metrics` label.
///
/// Kept apart from `FinishReason` because they answer different questions:
/// that one is what a client is told, this one is what an operator counts.
/// A refusal has no finish reason at all and is exactly the outcome worth
/// alerting on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The model stopped on its own — a complete answer.
    Stop,
    /// `max_tokens` or the context ran out. A truncated answer, which is not
    /// an error and is still worth seeing the rate of.
    Length,
    /// The client hung up before the answer finished. Not a failure — the
    /// server did exactly what it was asked and then was told to stop — but
    /// worth its own count: a rate of these that climbs is a client-side
    /// timeout set below what this server can deliver.
    Cancelled,
    /// The request never got a slot: the admission queue was full.
    Overloaded,
    /// Generation failed.
    Error,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Stop => "stop",
            Outcome::Length => "length",
            Outcome::Cancelled => "cancelled",
            Outcome::Overloaded => "overloaded",
            Outcome::Error => "error",
        }
    }

    /// Every outcome, so the exporter emits a zero for the ones that have not
    /// happened yet. A counter that appears only once it is non-zero makes an
    /// alert on `rate(...) > 0` fire on the *first* occurrence and then look
    /// like it was always there.
    pub const ALL: [Outcome; 5] = [
        Outcome::Stop,
        Outcome::Length,
        Outcome::Cancelled,
        Outcome::Overloaded,
        Outcome::Error,
    ];

    fn index(self) -> usize {
        match self {
            Outcome::Stop => 0,
            Outcome::Length => 1,
            Outcome::Cancelled => 2,
            Outcome::Overloaded => 3,
            Outcome::Error => 4,
        }
    }
}

/// Everything `/metrics` reports that is not read live off the slot pool.
pub struct ServerMetrics {
    queue_wait: Histogram,
    time_to_first_token: Histogram,
    inter_token: Histogram,
    request: Histogram,
    outcomes: [AtomicU64; 5],
    prompt_tokens: AtomicU64,
    cached_prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self {
            queue_wait: Histogram::new(LATENCY_BUCKETS),
            time_to_first_token: Histogram::new(LATENCY_BUCKETS),
            inter_token: Histogram::new(TOKEN_BUCKETS),
            request: Histogram::new(LATENCY_BUCKETS),
            outcomes: [const { AtomicU64::new(0) }; 5],
            prompt_tokens: AtomicU64::new(0),
            cached_prompt_tokens: AtomicU64::new(0),
            generated_tokens: AtomicU64::new(0),
        }
    }

    pub fn observe_queue_wait(&self, waited: Duration) {
        self.queue_wait.observe(waited);
    }

    pub fn observe_first_token(&self, elapsed: Duration) {
        self.time_to_first_token.observe(elapsed);
    }

    pub fn observe_inter_token(&self, gap: Duration) {
        self.inter_token.observe(gap);
    }

    /// A finished request: how long it took, how it ended, and what it cost.
    ///
    /// `cached` is the part of the prompt served from a reused KV prefix and
    /// never forwarded. Reported separately rather than subtracted, because
    /// the two together are the only way to read a cache-hit rate off
    /// `/metrics` — and a prompt-token counter that silently excluded cached
    /// tokens would make a well-cached server look idle.
    pub fn observe_request(
        &self,
        duration: Duration,
        outcome: Outcome,
        prompt: usize,
        cached: usize,
        generated: usize,
    ) {
        self.request.observe(duration);
        self.outcomes[outcome.index()].fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(prompt as u64, Ordering::Relaxed);
        self.cached_prompt_tokens
            .fetch_add(cached as u64, Ordering::Relaxed);
        self.generated_tokens
            .fetch_add(generated as u64, Ordering::Relaxed);
    }

    /// A request refused before it ever reached a slot.
    ///
    /// Its own entry point rather than an `observe_request` with zeros: a
    /// refusal has no duration worth putting in the latency histogram, and
    /// folding a handful of microseconds into it would drag every quantile
    /// down exactly when the server is under the load those quantiles are
    /// being watched for.
    pub fn observe_refusal(&self) {
        self.outcomes[Outcome::Overloaded.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The whole set, in Prometheus text format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.queue_wait.render(
            "orangu_server_queue_wait_seconds",
            "Time from a request arriving to it holding a slot.",
        ));
        out.push_str(&self.time_to_first_token.render(
            "orangu_server_time_to_first_token_seconds",
            "Time from a request arriving to its first generated token.",
        ));
        out.push_str(&self.inter_token.render(
            "orangu_server_inter_token_seconds",
            "Gap between consecutive generated tokens.",
        ));
        out.push_str(&self.request.render(
            "orangu_server_request_seconds",
            "Time from a request arriving to its last token.",
        ));
        out.push_str(
            "# HELP orangu_server_requests_total Requests by how they finished.\n\
             # TYPE orangu_server_requests_total counter\n",
        );
        for outcome in Outcome::ALL {
            out.push_str(&format!(
                "orangu_server_requests_total{{outcome=\"{}\"}} {}\n",
                outcome.label(),
                self.outcomes[outcome.index()].load(Ordering::Relaxed)
            ));
        }
        for (name, help, value) in [
            (
                "orangu_server_prompt_tokens_total",
                "Prompt tokens accepted, cached or not.",
                &self.prompt_tokens,
            ),
            (
                "orangu_server_cached_prompt_tokens_total",
                "Prompt tokens served from a reused KV prefix.",
                &self.cached_prompt_tokens,
            ),
            (
                "orangu_server_generated_tokens_total",
                "Tokens generated.",
                &self.generated_tokens,
            ),
        ] {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
                value.load(Ordering::Relaxed)
            ));
        }
        out
    }
}

// ------------------------------------------------------------- process ---

pub struct ProcessMetrics {
    http_in_flight: Arc<AtomicU64>,
    http_total: AtomicU64,
    http_client_errors: AtomicU64,
    http_server_errors: AtomicU64,
    system: Mutex<System>,
    pid: Pid,
    /// Unix seconds; the OS's own answer, falling back to when this was built.
    start_time_seconds: u64,
    logical_cores: usize,
    host_info: String,
}

/// Counts one in-flight HTTP request for as long as it is alive. Owned, so it
/// can ride along with a response body until the last byte has been sent.
pub struct InFlightGuard(Arc<AtomicU64>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessMetrics {
    pub fn new() -> Self {
        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::everything(),
        );
        let start_time_seconds = system
            .process(pid)
            .map(|process| process.start_time())
            .filter(|&start| start != 0)
            .unwrap_or_else(now_unix_seconds);

        let host_info = info_line(
            "orangu_server_host_info",
            "Build of this server and the host it runs on; the value is always 1.",
            &[
                ("name", env!("CARGO_BIN_NAME")),
                ("version", orangu::build_info::VERSION),
                ("commit", orangu::build_info::COMMIT),
                ("rustc", orangu::build_info::RUSTC),
                ("profile", orangu::build_info::PROFILE),
                ("target", orangu::build_info::TARGET),
                ("os", &System::name().unwrap_or_default()),
                ("os_version", &System::os_version().unwrap_or_default()),
                ("kernel", &System::kernel_version().unwrap_or_default()),
                ("arch", std::env::consts::ARCH),
            ],
        );

        Self {
            http_in_flight: Arc::new(AtomicU64::new(0)),
            http_total: AtomicU64::new(0),
            http_client_errors: AtomicU64::new(0),
            http_server_errors: AtomicU64::new(0),
            system: Mutex::new(system),
            pid,
            start_time_seconds,
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            host_info,
        }
    }

    /// Marks a request as in flight until the returned guard is dropped.
    pub fn track_request(&self) -> InFlightGuard {
        self.http_in_flight.fetch_add(1, Ordering::Relaxed);
        InFlightGuard(self.http_in_flight.clone())
    }

    /// Records the status a finished request was answered with.
    pub fn record_response(&self, status: u16) {
        self.http_total.fetch_add(1, Ordering::Relaxed);
        match status {
            400..=499 => self.http_client_errors.fetch_add(1, Ordering::Relaxed),
            500..=599 => self.http_server_errors.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }

    pub fn in_flight(&self) -> u64 {
        self.http_in_flight.load(Ordering::Relaxed)
    }

    pub fn render(&self, uptime: Duration) -> String {
        let (cpu_seconds, resident, virtual_bytes, mem_total, mem_available) = {
            let mut system = self.system.lock().unwrap_or_else(|e| e.into_inner());
            system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[self.pid]),
                true,
                ProcessRefreshKind::everything(),
            );
            system.refresh_memory();
            let process = system.process(self.pid);
            (
                process.map_or(0.0, |p| p.accumulated_cpu_time() as f64 / 1000.0),
                process.map_or(0, |p| p.memory()),
                process.map_or(0, |p| p.virtual_memory()),
                system.total_memory(),
                system.available_memory(),
            )
        };

        let mut out = String::new();
        out.push_str(&self.host_info);
        labelled_gauge(
            &mut out,
            "orangu_server_uptime",
            "Seconds since this server started; start_time is when, as a Unix timestamp.",
            "start_time",
            &[(
                self.start_time_seconds.to_string(),
                uptime.as_secs_f64().to_string(),
            )],
        );

        let mut cpu = vec![
            ("process_seconds".to_string(), cpu_seconds.to_string()),
            ("logical_cores".to_string(), self.logical_cores.to_string()),
        ];
        cpu.extend(load_average_stats());
        labelled_gauge(
            &mut out,
            "orangu_server_cpu",
            "CPU: process_seconds is CPU time this process has used, load1/load5/load15 are the host's load averages, logical_cores what the process may use.",
            "stat",
            &cpu,
        );

        gauge(
            &mut out,
            "process_resident_memory_bytes",
            "Resident memory size in bytes.",
            resident,
        );
        gauge(
            &mut out,
            "process_virtual_memory_bytes",
            "Virtual memory size in bytes.",
            virtual_bytes,
        );
        #[cfg(target_os = "linux")]
        if let Ok(fds) = std::fs::read_dir("/proc/self/fd") {
            gauge(
                &mut out,
                "process_open_fds",
                "Number of open file descriptors.",
                fds.count(),
            );
        }
        gauge(
            &mut out,
            "orangu_server_host_memory_total_bytes",
            "Total physical memory of the host, in bytes.",
            mem_total,
        );
        gauge(
            &mut out,
            "orangu_server_host_memory_available_bytes",
            "Physical memory of the host available to start new work, in bytes.",
            mem_available,
        );
        labelled_gauge(
            &mut out,
            "orangu_server_http_requests",
            "HTTP requests on the API and web console listeners (not the dedicated metrics port): in_flight now, then total, client_errors (4xx) and server_errors (5xx) answered since start.",
            "stat",
            &[
                ("in_flight".to_string(), self.in_flight().to_string()),
                (
                    "total".to_string(),
                    self.http_total.load(Ordering::Relaxed).to_string(),
                ),
                (
                    "client_errors".to_string(),
                    self.http_client_errors.load(Ordering::Relaxed).to_string(),
                ),
                (
                    "server_errors".to_string(),
                    self.http_server_errors.load(Ordering::Relaxed).to_string(),
                ),
            ],
        );
        out
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn gauge(out: &mut String, name: &str, help: &str, value: impl std::fmt::Display) {
    let _ = write!(
        out,
        "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
    );
}

/// A gauge family with one series per row, each labelled `label_key="<row.0>"`
/// and valued `row.1`.
fn labelled_gauge(
    out: &mut String,
    name: &str,
    help: &str,
    label_key: &str,
    rows: &[(String, String)],
) {
    let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n");
    for (label, value) in rows {
        let _ = writeln!(
            out,
            "{name}{{{label_key}=\"{}\"}} {value}",
            escape_label(label)
        );
    }
}

/// One `_info`-style gauge: constant `1`, the facts carried as labels.
fn info_line(name: &str, help: &str, labels: &[(&str, &str)]) -> String {
    let labels = labels
        .iter()
        .map(|(key, value)| format!("{key}=\"{}\"", escape_label(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("# HELP {name} {help}\n# TYPE {name} gauge\n{name}{{{labels}}} 1\n")
}

/// Label values escape `\`, `"` and newline, and nothing else.
fn escape_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// The host's one-, five- and fifteen-minute load averages, or nothing where
/// the platform has none.
///
/// `sysinfo` answers zeros on Windows rather than failing, so reporting them
/// would put three convincing-looking numbers on a scrape that mean nothing.
/// A platform difference in one place, and a value the caller always has to
/// append, so the `cpu` list is built the same way everywhere — a `cfg` block
/// that pushed into it left the binding unused on Windows, which is a lint
/// error there and invisible on any other machine.
fn load_average_stats() -> Vec<(String, String)> {
    #[cfg(target_os = "windows")]
    {
        Vec::new()
    }
    #[cfg(not(target_os = "windows"))]
    {
        let load = System::load_average();
        vec![
            ("load1".to_string(), load.one.to_string()),
            ("load5".to_string(), load.five.to_string()),
            ("load15".to_string(), load.fifteen.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn secs(v: f64) -> Duration {
        Duration::from_secs_f64(v)
    }

    /// Buckets are cumulative when rendered. Getting this wrong produces a
    /// histogram that scrapes cleanly and gives nonsense quantiles, which is
    /// the worst way for a metric to be broken.
    #[test]
    fn buckets_are_cumulative_and_agree_with_the_count() {
        let h = Histogram::new(LATENCY_BUCKETS);
        for v in [0.001, 0.03, 0.03, 0.4, 90.0] {
            h.observe(secs(v));
        }
        let text = h.render("test", "help");
        let bucket = |le: &str| -> u64 {
            text.lines()
                .find_map(|l| l.strip_prefix(&format!("test_bucket{{le=\"{le}\"}} ")))
                .unwrap_or_else(|| panic!("no bucket {le} in\n{text}"))
                .parse()
                .unwrap()
        };
        assert_eq!(bucket("0.005"), 1, "0.001");
        assert_eq!(bucket("0.01"), 1);
        assert_eq!(bucket("0.05"), 3, "0.001 plus both 0.03s");
        assert_eq!(bucket("0.5"), 4);
        assert_eq!(bucket("60"), 4, "90s is above every bound");
        assert_eq!(bucket("+Inf"), 5);

        // `_count` and the `+Inf` bucket are two reads of the same thing, and
        // a scraper checks that they agree.
        assert!(text.contains("test_count 5"), "{text}");
        // 0.001 + 0.03 + 0.03 + 0.4 + 90 = 90.461
        assert!(text.contains("test_sum 90.461000"), "{text}");
    }

    /// A bound is inclusive: `le="0.05"` counts an observation of exactly
    /// 0.05. Off by one here shifts every quantile by a bucket.
    #[test]
    fn an_observation_lands_in_the_bucket_it_is_equal_to() {
        let h = Histogram::new(LATENCY_BUCKETS);
        h.observe(secs(0.05));
        let text = h.render("t", "h");
        assert!(text.contains("t_bucket{le=\"0.05\"} 1"), "{text}");
        assert!(text.contains("t_bucket{le=\"0.025\"} 0"), "{text}");
    }

    #[test]
    fn an_empty_histogram_still_renders_every_bucket() {
        let text = Histogram::new(TOKEN_BUCKETS).render("t", "h");
        assert_eq!(text.matches("t_bucket{").count(), TOKEN_BUCKETS.len() + 1);
        assert!(text.contains("t_count 0"), "{text}");
        assert!(text.contains("t_sum 0.000000"), "{text}");
    }

    /// Every outcome appears even at zero. A counter that materialises only
    /// once it is non-zero makes `rate(...) > 0` fire on the first occurrence
    /// and then look as though it had always been there.
    #[test]
    fn every_outcome_is_exported_from_the_start() {
        let m = ServerMetrics::new();
        let text = m.render();
        for outcome in Outcome::ALL {
            assert!(
                text.contains(&format!(
                    "orangu_server_requests_total{{outcome=\"{}\"}} 0",
                    outcome.label()
                )),
                "{} missing from\n{text}",
                outcome.label()
            );
        }
        let mut labels: Vec<&str> = Outcome::ALL.iter().map(|o| o.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), Outcome::ALL.len(), "duplicate outcome label");
    }

    /// A refusal is counted but must not enter the latency histograms: it has
    /// no duration worth reporting, and folding microseconds in would drag
    /// every quantile down exactly when the server is under the load those
    /// quantiles are being watched for.
    #[test]
    fn a_refusal_counts_without_touching_the_latency_distributions() {
        let m = ServerMetrics::new();
        m.observe_refusal();
        assert_eq!(m.request.count(), 0);
        assert_eq!(m.queue_wait.count(), 0);
        assert_eq!(m.time_to_first_token.count(), 0);
        let text = m.render();
        assert!(
            text.contains("orangu_server_requests_total{outcome=\"overloaded\"} 1"),
            "{text}"
        );
    }

    #[test]
    fn a_finished_request_records_its_tokens_and_its_outcome() {
        let m = ServerMetrics::new();
        m.observe_request(secs(1.5), Outcome::Length, 100, 40, 7);
        let text = m.render();
        assert!(
            text.contains("orangu_server_prompt_tokens_total 100"),
            "{text}"
        );
        assert!(
            text.contains("orangu_server_cached_prompt_tokens_total 40"),
            "{text}"
        );
        assert!(
            text.contains("orangu_server_generated_tokens_total 7"),
            "{text}"
        );
        assert!(
            text.contains("orangu_server_requests_total{outcome=\"length\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("orangu_server_request_seconds_count 1"),
            "{text}"
        );
    }

    /// Inter-token gaps need their own bounds: on the latency set almost
    /// every decode step lands in one bucket and the histogram reports
    /// nothing at all.
    #[test]
    fn inter_token_bounds_resolve_a_realistic_decode_step() {
        let h = Histogram::new(TOKEN_BUCKETS);
        // 30 tok/s and 3 tok/s — both ordinary for this engine, and they must
        // not share a bucket.
        h.observe(secs(1.0 / 30.0));
        h.observe(secs(1.0 / 3.0));
        let text = h.render("t", "h");
        assert!(text.contains("t_bucket{le=\"0.05\"} 1"), "{text}");
        assert!(text.contains("t_bucket{le=\"0.5\"} 2"), "{text}");
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\\b\"c\nd"), "a\\\\b\\\"c\\nd");
    }

    #[test]
    fn in_flight_returns_to_zero_when_guards_drop() {
        let metrics = ProcessMetrics::new();
        assert_eq!(metrics.in_flight(), 0);
        let first = metrics.track_request();
        let second = metrics.track_request();
        assert_eq!(metrics.in_flight(), 2);
        drop(first);
        assert_eq!(metrics.in_flight(), 1);
        drop(second);
        assert_eq!(metrics.in_flight(), 0);
    }

    /// Every sample sits under a `# HELP`/`# TYPE` pair for its own name, no
    /// name is declared twice, and every value parses as a number.
    #[test]
    fn output_is_well_formed_exposition_text() {
        let body = ProcessMetrics::new().render(Duration::from_secs(5));
        let mut typed = HashSet::new();
        let mut helped = HashSet::new();
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let name = rest.split_whitespace().next().unwrap();
                assert!(helped.insert(name.to_string()), "duplicate HELP {name}");
            } else if let Some(rest) = line.strip_prefix("# TYPE ") {
                let name = rest.split_whitespace().next().unwrap();
                assert!(helped.contains(name), "TYPE before HELP for {name}");
                assert!(typed.insert(name.to_string()), "duplicate TYPE {name}");
            } else {
                let (series, value) = line.rsplit_once(' ').expect(line);
                let name = series.split('{').next().unwrap();
                assert!(typed.contains(name), "sample without TYPE: {line}");
                value.parse::<f64>().unwrap_or_else(|_| panic!("{line}"));
            }
        }
    }

    #[test]
    fn responses_are_counted_by_class() {
        let metrics = ProcessMetrics::new();
        for status in [200, 204, 404, 401, 503] {
            metrics.record_response(status);
        }
        let body = metrics.render(Duration::ZERO);
        for expected in [
            "orangu_server_http_requests{stat=\"total\"} 5\n",
            "orangu_server_http_requests{stat=\"client_errors\"} 2\n",
            "orangu_server_http_requests{stat=\"server_errors\"} 1\n",
        ] {
            assert!(body.contains(expected), "{expected}\n{body}");
        }
    }

    #[test]
    fn reports_identity_time_and_memory() {
        let body = ProcessMetrics::new().render(Duration::from_secs(42));
        assert!(
            body.contains("\"} 42\n") && body.contains("orangu_server_uptime{start_time=\""),
            "{body}"
        );
        assert!(
            body.contains(&format!(
                "orangu_server_host_info{{name=\"orangu-server\",version=\"{}\"",
                orangu::build_info::VERSION
            )),
            "{body}"
        );
        assert!(
            body.contains(&format!("arch=\"{}\"}} 1\n", std::env::consts::ARCH)),
            "{body}"
        );
        assert!(
            body.contains("orangu_server_http_requests{stat=\"in_flight\"} 0\n"),
            "{body}"
        );
        assert!(
            body.contains("orangu_server_cpu{stat=\"process_seconds\"} "),
            "{body}"
        );
        assert!(!body.contains("orangu_server_build_info"), "{body}");
        let start: u64 = body
            .lines()
            .find_map(|l| l.strip_prefix("orangu_server_uptime{start_time=\""))
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            start > 1_600_000_000 && start <= now_unix_seconds(),
            "{start}"
        );
        let rss: u64 = body
            .lines()
            .find_map(|l| l.strip_prefix("process_resident_memory_bytes "))
            .unwrap()
            .parse()
            .unwrap();
        assert!(rss > 0);
    }
}

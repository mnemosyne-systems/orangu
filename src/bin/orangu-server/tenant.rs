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

//! Who is asking, and how much of the machine they may have.
//!
//! # What this protects, and what it does not
//!
//! `[orangu-server].queue_limit` already stops the server from collapsing
//! under load: past the bound a request is refused rather than queued. That is
//! a limit on the *server*. It says nothing about how the capacity is divided,
//! so one client opening thirty streams fills the queue and every other client
//! sees `503` — the server is healthy and useless.
//!
//! What is here is the other half: a limit *per caller*, so one caller's
//! excess is refused before it becomes everyone's outage. It needs a notion of
//! who a caller is, which is why it could not exist before the server checked
//! a bearer token at all.
//!
//! # Three limits, because they fail differently
//!
//! - `max_concurrent` — requests in flight at once. The direct bound on the
//!   scarce thing: a generation occupies a slot from admission until its last
//!   token, and slots are what everyone is queueing for.
//! - `requests_per_minute` — arrival rate. Concurrency alone does not stop a
//!   loop firing thousands of one-token requests; each is brief, so none of
//!   them is ever concurrent with the next.
//! - `tokens_per_minute` — work done. The only one denominated in what a
//!   request actually costs. Two requests are not two units of anything: one
//!   may be forty tokens and the next four thousand, and a limit that cannot
//!   tell them apart is not a limit on the machine's time.
//!
//! Every limit defaults to `0`, meaning unlimited, so a tenant declared
//! without one is authenticated and unmetered.
//!
//! # Refusal, not queueing
//!
//! Over a limit the answer is `429` immediately. Making the request wait
//! instead would put it in the admission queue — the shared resource the limit
//! exists to keep it out of — so a tenant over its bound would still be
//! occupying capacity, just silently. A refusal with `Retry-After` is
//! something a client can act on; a wait is something it can only endure.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// How long a rate window looks back. One minute, matching the units the
/// config keys are named in — a caller reading `requests_per_minute = 60`
/// should not have to learn that it is enforced over some other interval.
const WINDOW_SECONDS: u64 = 60;

/// Number of buckets in a window: one per second of it.
const BUCKETS: usize = WINDOW_SECONDS as usize;

/// What a tenant may consume. `0` is unlimited, everywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TenantLimits {
    pub max_concurrent: usize,
    pub requests_per_minute: u64,
    pub tokens_per_minute: u64,
}

impl TenantLimits {
    /// Whether this tenant is metered at all — a tenant declared with a key
    /// and no limits is authentication only, and saying so lets the caller
    /// skip the bookkeeping entirely.
    pub fn any(&self) -> bool {
        self.max_concurrent > 0 || self.requests_per_minute > 0 || self.tokens_per_minute > 0
    }
}

/// A tenant exactly as the configuration file declares it.
#[derive(Clone, Debug)]
pub struct TenantConfig {
    /// The part after `tenant:` in the section header. Appears in `/metrics`
    /// and in refusal messages, so it is a name a human chose.
    pub name: String,
    /// The bearer token that identifies this tenant. Resolved at startup —
    /// from the section or from the environment variable it names — so a
    /// missing secret is a startup failure rather than a tenant nobody can
    /// authenticate as.
    pub api_key: String,
    pub limits: TenantLimits,
}

/// Which limit refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    Concurrency,
    Requests,
    Tokens,
}

impl Limit {
    /// The `limit=` label on `/metrics`, and the word in the refusal message.
    pub fn label(self) -> &'static str {
        match self {
            Limit::Concurrency => "concurrency",
            Limit::Requests => "requests",
            Limit::Tokens => "tokens",
        }
    }

    /// Every limit, so a metrics exporter and a test can both enumerate them
    /// rather than repeat the list.
    pub const ALL: [Limit; 3] = [Limit::Concurrency, Limit::Requests, Limit::Tokens];

    fn index(self) -> usize {
        match self {
            Limit::Concurrency => 0,
            Limit::Requests => 1,
            Limit::Tokens => 2,
        }
    }
}

/// A refusal, with everything the HTTP layer needs to explain it.
#[derive(Clone, Debug)]
pub struct Denied {
    pub limit: Limit,
    /// Seconds until the request could succeed — a real number, not a
    /// constant: for a rate window it is when the oldest counted second rolls
    /// off, which is the earliest moment there is room.
    pub retry_after: u64,
    pub message: String,
}

/// A rolling count over the last [`WINDOW_SECONDS`] seconds.
///
/// One bucket per second, indexed by `second % 60` and stamped with the second
/// it stands for, so a bucket last written more than a minute ago reads as
/// empty without anything having to sweep it. Fixed size (under a kilobyte per
/// window), no allocation, and no background task.
///
/// A single fixed window would have been less code and is the usual shortcut,
/// but it lets a caller spend the whole minute's budget in the last second of
/// one window and the whole of the next in the first second of the following —
/// twice the configured rate, arriving as a burst, which is the exact shape
/// the limit exists to prevent.
#[derive(Debug)]
struct Window {
    counts: [u64; BUCKETS],
    /// The second each bucket's count belongs to.
    stamps: [u64; BUCKETS],
    /// Whether a bucket has ever been written. Without it, second `0` is
    /// indistinguishable from "never used" and the first minute of uptime
    /// counts nothing.
    used: [bool; BUCKETS],
}

impl Default for Window {
    fn default() -> Self {
        Self {
            counts: [0; BUCKETS],
            stamps: [0; BUCKETS],
            used: [false; BUCKETS],
        }
    }
}

impl Window {
    /// Live buckets: written at some point, and within the last minute.
    fn live(&self, now: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..BUCKETS).filter_map(move |i| {
            (self.used[i] && now.saturating_sub(self.stamps[i]) < WINDOW_SECONDS)
                .then_some((self.stamps[i], self.counts[i]))
        })
    }

    fn total(&self, now: u64) -> u64 {
        self.live(now).map(|(_, count)| count).sum()
    }

    fn add(&mut self, now: u64, amount: u64) {
        let i = (now % WINDOW_SECONDS) as usize;
        if !self.used[i] || self.stamps[i] != now {
            self.stamps[i] = now;
            self.counts[i] = 0;
            self.used[i] = true;
        }
        self.counts[i] = self.counts[i].saturating_add(amount);
    }

    /// When the window next has room: the oldest counted second plus a minute.
    ///
    /// At least one, because `Retry-After: 0` invites a client to retry in a
    /// tight loop, which is what it was just refused for doing.
    fn retry_after(&self, now: u64) -> u64 {
        self.live(now)
            .filter(|(_, count)| *count > 0)
            .map(|(second, _)| (second + WINDOW_SECONDS).saturating_sub(now))
            .min()
            .unwrap_or(1)
            .max(1)
    }
}

/// One tenant's live accounting.
///
/// Shared by `Arc` between the middleware that admits a request and the
/// generation that finishes it, because the two happen on different tasks and
/// minutes apart.
pub struct TenantMeter {
    name: String,
    limits: TenantLimits,
    /// Monotonic origin for the rate windows. `Instant`, not the wall clock: a
    /// clock stepped backwards by NTP would make a window's arithmetic
    /// nonsense, and a limit that unblocks or hangs when the machine
    /// synchronises its time is worse than no limit.
    started: Instant,
    in_flight: AtomicUsize,
    requests: Mutex<Window>,
    tokens: Mutex<Window>,
    total_requests: AtomicU64,
    total_tokens: AtomicU64,
    denied: [AtomicU64; 3],
}

impl TenantMeter {
    pub fn new(config: &TenantConfig) -> Self {
        Self {
            name: config.name.clone(),
            limits: config.limits,
            started: Instant::now(),
            in_flight: AtomicUsize::new(0),
            requests: Mutex::new(Window::default()),
            tokens: Mutex::new(Window::default()),
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            denied: [const { AtomicU64::new(0) }; 3],
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn limits(&self) -> TenantLimits {
        self.limits
    }

    fn now(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    /// Takes a place for one request, or says which limit refused it.
    ///
    /// Ordered cheapest first — concurrency is one atomic, the windows take a
    /// lock — and, more importantly, ordered so that the answer names the
    /// limit a caller can most directly act on.
    pub fn admit(self: &Arc<Self>) -> Result<InFlight, Denied> {
        self.admit_at(self.now())
    }

    /// [`admit`](Self::admit) against a caller-supplied second, so the rate
    /// windows can be tested without sleeping through them.
    fn admit_at(self: &Arc<Self>, now: u64) -> Result<InFlight, Denied> {
        // Claimed before the rate checks, so that two requests arriving
        // together cannot both read "one below the limit" and both proceed.
        // The guard is built immediately, so a refusal further down releases
        // the claim by dropping it rather than by remembering to.
        if self.limits.max_concurrent > 0 {
            let claimed = self
                .in_flight
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                    (n < self.limits.max_concurrent).then_some(n + 1)
                })
                .is_ok();
            if !claimed {
                return Err(self.deny(
                    Limit::Concurrency,
                    1,
                    format!(
                        "{} requests already in flight for tenant '{}'",
                        self.limits.max_concurrent, self.name
                    ),
                ));
            }
        } else {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
        }
        let guard = InFlight {
            meter: self.clone(),
        };

        if self.limits.requests_per_minute > 0 {
            let window = self.requests.lock().unwrap();
            if window.total(now) >= self.limits.requests_per_minute {
                let retry_after = window.retry_after(now);
                drop(window);
                return Err(self.deny(
                    Limit::Requests,
                    retry_after,
                    format!(
                        "{} requests per minute for tenant '{}'",
                        self.limits.requests_per_minute, self.name
                    ),
                ));
            }
        }

        // Checked *before* the request runs against tokens *already* spent, so
        // the request in flight can overshoot the budget by its own length.
        // Unavoidable rather than sloppy: what a generation will cost is not
        // known until it stops, and refusing on a guess would refuse requests
        // that fit. The overshoot is bounded by `max_tokens` and is repaid out
        // of the next minute.
        if self.limits.tokens_per_minute > 0 {
            let window = self.tokens.lock().unwrap();
            if window.total(now) >= self.limits.tokens_per_minute {
                let retry_after = window.retry_after(now);
                drop(window);
                return Err(self.deny(
                    Limit::Tokens,
                    retry_after,
                    format!(
                        "{} tokens per minute for tenant '{}'",
                        self.limits.tokens_per_minute, self.name
                    ),
                ));
            }
        }

        self.requests.lock().unwrap().add(now, 1);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        Ok(guard)
    }

    fn deny(&self, limit: Limit, retry_after: u64, detail: String) -> Denied {
        self.denied[limit.index()].fetch_add(1, Ordering::Relaxed);
        Denied {
            limit,
            retry_after,
            message: format!(
                "rate limit exceeded ({}): {detail}. Retry in {retry_after}s.",
                limit.label()
            ),
        }
    }

    /// Records the tokens a finished request cost.
    ///
    /// Prompt tokens count as well as generated ones: prefill is a forward
    /// pass over every one of them, and on a long prompt it is the larger half
    /// of the work. A budget that counted only the answer would let a caller
    /// spend the machine entirely on prompts it never reads a reply to.
    pub fn charge(&self, tokens: u64) {
        self.charge_at(self.now(), tokens);
    }

    /// [`charge`](Self::charge) against a caller-supplied second, so the token
    /// window can be tested without sleeping through it.
    fn charge_at(&self, now: u64, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
        // Recorded whether or not a token limit is set, because `/metrics`
        // reports the window for every tenant and a figure that appeared only
        // for limited tenants would read as "this one used nothing".
        self.tokens.lock().unwrap().add(now, tokens);
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    pub fn denied(&self, limit: Limit) -> u64 {
        self.denied[limit.index()].load(Ordering::Relaxed)
    }

    /// Tokens counted in the current window, for `/metrics`.
    pub fn tokens_in_window(&self) -> u64 {
        let now = self.now();
        self.tokens.lock().unwrap().total(now)
    }

    /// Requests counted in the current window, for `/metrics`.
    pub fn requests_in_window(&self) -> u64 {
        let now = self.now();
        self.requests.lock().unwrap().total(now)
    }
}

/// A request's place against its tenant's concurrency limit.
///
/// Its `Drop` is the whole mechanism, and where it is dropped is the whole
/// question — see `http::hold_until_finished`. A streamed generation is still
/// running long after the handler that started it has returned, so a guard
/// released when the handler returns would count the one shape of request the
/// limit exists for as instantaneous.
pub struct InFlight {
    meter: Arc<TenantMeter>,
}

/// Named rather than derived: the meter behind it holds locks and atomics
/// whose values would be a lie by the time anyone read them, and the only
/// thing worth printing is whose place this is.
impl std::fmt::Debug for InFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InFlight({})", self.meter.name)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        // Saturating, not wrapping: an underflow here would read as a tenant
        // with four billion requests in flight and lock them out permanently.
        let _ = self
            .meter
            .in_flight
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(1))
            });
    }
}

/// Every declared tenant, indexed by the key that identifies it.
pub struct TenantRegistry {
    meters: Vec<(String, Arc<TenantMeter>)>,
}

impl TenantRegistry {
    pub fn new(tenants: &[TenantConfig]) -> Self {
        Self {
            meters: tenants
                .iter()
                .map(|t| (t.api_key.clone(), Arc::new(TenantMeter::new(t))))
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.meters.is_empty()
    }

    /// The tenant a presented bearer token belongs to.
    ///
    /// Scans every entry with no early exit, and compares each in constant
    /// time. Returning on the first match would make the answer arrive sooner
    /// for a key declared early in the file, which leaks nothing about the
    /// secret itself but does leak which tenant was matched — and stopping
    /// early on a *partial* match is the thing that leaks the key one byte at
    /// a time, which is why the comparison is not `==`.
    pub fn resolve(&self, presented: &str) -> Option<Arc<TenantMeter>> {
        let mut found: Option<&Arc<TenantMeter>> = None;
        for (key, meter) in &self.meters {
            if constant_time_eq(presented.as_bytes(), key.as_bytes()) {
                found = Some(meter);
            }
        }
        found.cloned()
    }

    /// Every meter, in declaration order, for `/metrics`.
    pub fn meters(&self) -> impl Iterator<Item = &Arc<TenantMeter>> {
        self.meters.iter().map(|(_, meter)| meter)
    }
}

/// Compares two byte strings without leaking where they first differ.
///
/// A `==` on secrets returns as soon as it finds a mismatch, so the time it
/// takes reports how many leading bytes were right — enough, over many
/// attempts, to recover a key one byte at a time.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenant(name: &str, limits: TenantLimits) -> TenantConfig {
        TenantConfig {
            name: name.to_string(),
            api_key: format!("{name}-key"),
            limits,
        }
    }

    fn meter(limits: TenantLimits) -> Arc<TenantMeter> {
        Arc::new(TenantMeter::new(&tenant("t", limits)))
    }

    #[test]
    fn an_unlimited_tenant_is_never_refused() {
        let meter = meter(TenantLimits::default());
        assert!(!meter.limits().any());
        let mut held = Vec::new();
        for _ in 0..1000 {
            held.push(meter.admit_at(0).expect("no limit can refuse"));
        }
        assert_eq!(meter.in_flight(), 1000);
        drop(held);
        assert_eq!(meter.in_flight(), 0);
    }

    #[test]
    fn concurrency_refuses_the_request_past_the_limit_and_admits_it_again_after_a_release() {
        let meter = meter(TenantLimits {
            max_concurrent: 2,
            ..Default::default()
        });
        let a = meter.admit_at(0).unwrap();
        let b = meter.admit_at(0).unwrap();
        let denied = meter.admit_at(0).expect_err("a third must be refused");
        assert_eq!(denied.limit, Limit::Concurrency);
        assert_eq!(meter.denied(Limit::Concurrency), 1);

        drop(a);
        let c = meter.admit_at(0).expect("a place was released");
        drop((b, c));
        assert_eq!(meter.in_flight(), 0);
    }

    /// A refusal by a *later* check must not leave the concurrency claim
    /// behind. Taking the claim first is what makes the check atomic, so
    /// releasing it on every path out is not optional — and a leak here is
    /// invisible until the tenant is permanently locked out.
    #[test]
    fn a_rate_refusal_releases_the_concurrency_claim_it_took() {
        let meter = meter(TenantLimits {
            max_concurrent: 4,
            requests_per_minute: 1,
            ..Default::default()
        });
        let first = meter.admit_at(0).unwrap();
        drop(first);
        assert_eq!(meter.in_flight(), 0);

        for _ in 0..10 {
            let denied = meter.admit_at(0).expect_err("over the request rate");
            assert_eq!(denied.limit, Limit::Requests);
        }
        assert_eq!(
            meter.in_flight(),
            0,
            "a refused request left its concurrency claim behind"
        );
    }

    #[test]
    fn the_request_window_rolls_forward_a_second_at_a_time() {
        let meter = meter(TenantLimits {
            requests_per_minute: 2,
            ..Default::default()
        });
        drop(meter.admit_at(0).unwrap());
        drop(meter.admit_at(30).unwrap());
        let denied = meter.admit_at(31).expect_err("two already in the window");
        assert_eq!(denied.limit, Limit::Requests);
        // The second at t=0 rolls off at t=60, and that is what is reported.
        assert_eq!(denied.retry_after, 29);

        // At t=60 the first has left the window, so there is room for exactly
        // one more — not two, which is what a fixed window would allow.
        drop(meter.admit_at(60).expect("the oldest second rolled off"));
        assert!(meter.admit_at(60).is_err(), "the t=30 request still counts");
        drop(meter.admit_at(91).expect("both have now rolled off"));
    }

    /// A whole window with no traffic must leave nothing behind. The bucket
    /// array is indexed by `second % 60`, so a stale bucket that failed to
    /// expire would be counted again exactly one minute later.
    #[test]
    fn a_window_older_than_a_minute_counts_for_nothing() {
        let meter = meter(TenantLimits {
            requests_per_minute: 1,
            ..Default::default()
        });
        drop(meter.admit_at(5).unwrap());
        assert!(meter.admit_at(6).is_err());
        drop(meter.admit_at(65).expect("a full minute later"));
        drop(meter.admit_at(3600).expect("an hour later"));
    }

    #[test]
    fn tokens_are_charged_and_bound_the_next_request() {
        let meter = meter(TenantLimits {
            tokens_per_minute: 100,
            ..Default::default()
        });
        drop(meter.admit_at(0).expect("nothing spent yet"));
        // Prompt and generated tokens both count.
        meter.charge_at(0, 40);
        meter.charge_at(0, 70);
        assert_eq!(meter.total_tokens(), 110);
        let denied = meter.admit_at(0).expect_err("over budget");
        assert_eq!(denied.limit, Limit::Tokens);
        assert_eq!(meter.denied(Limit::Tokens), 1);
    }

    /// The in-flight request may overshoot its budget, and the next one is
    /// refused for it. Stated as a test because it is the behaviour a reader
    /// would otherwise call a bug.
    #[test]
    fn a_request_may_overshoot_the_token_budget_and_the_next_one_pays() {
        let meter = meter(TenantLimits {
            tokens_per_minute: 10,
            ..Default::default()
        });
        let held = meter.admit_at(0).expect("nothing spent yet");
        meter.charge_at(0, 4000);
        drop(held);
        assert!(meter.admit_at(0).is_err());
        // ...and the budget is clear again a minute later, not sooner.
        assert!(meter.admit_at(59).is_err());
        drop(meter.admit_at(60).expect("the window rolled"));
    }

    /// A tenant with no token limit is still *accounted* for, because
    /// `/metrics` reports usage for every tenant — attribution is the half of
    /// this feature that works without anyone setting a limit at all.
    #[test]
    fn usage_is_recorded_even_when_no_token_limit_is_set() {
        let meter = meter(TenantLimits::default());
        meter.charge_at(0, 1234);
        assert_eq!(meter.total_tokens(), 1234);
        assert_eq!(meter.tokens_in_window(), 1234);
        drop(meter.admit_at(0).unwrap());
        assert_eq!(meter.total_requests(), 1);
        assert_eq!(meter.requests_in_window(), 1);
    }

    #[test]
    fn a_key_resolves_to_its_own_tenant_and_nothing_else_resolves_at_all() {
        let registry = TenantRegistry::new(&[
            tenant("alice", TenantLimits::default()),
            tenant("bob", TenantLimits::default()),
        ]);
        assert_eq!(registry.resolve("alice-key").unwrap().name(), "alice");
        assert_eq!(registry.resolve("bob-key").unwrap().name(), "bob");
        assert!(registry.resolve("").is_none());
        assert!(registry.resolve("alice-ke").is_none());
        assert!(registry.resolve("alice-keys").is_none());
        assert!(registry.resolve("ALICE-KEY").is_none());
    }

    #[test]
    fn an_empty_registry_resolves_nothing() {
        let registry = TenantRegistry::new(&[]);
        assert!(registry.is_empty());
        assert!(registry.resolve("").is_none());
        assert!(registry.resolve("anything").is_none());
    }

    #[test]
    fn constant_time_eq_matches_only_on_the_whole_string() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"sec"));
        assert!(!constant_time_eq(b"sec", b"secret"));
        assert!(!constant_time_eq(b"", b"secret"));
    }

    /// Every limit has a distinct label and counter slot — a collision would
    /// silently merge two `/metrics` series.
    #[test]
    fn every_limit_has_its_own_label_and_counter() {
        let meter = meter(TenantLimits::default());
        let mut labels: Vec<&str> = Limit::ALL.iter().map(|l| l.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), Limit::ALL.len());

        for limit in Limit::ALL {
            meter.deny(limit, 1, String::new());
        }
        for limit in Limit::ALL {
            assert_eq!(meter.denied(limit), 1, "{}", limit.label());
        }
    }

    /// `Retry-After: 0` tells a client to retry immediately, which is what it
    /// was just refused for.
    #[test]
    fn a_refusal_never_asks_for_an_immediate_retry() {
        let meter = meter(TenantLimits {
            requests_per_minute: 1,
            ..Default::default()
        });
        drop(meter.admit_at(0).unwrap());
        for second in 0..60 {
            let denied = meter.admit_at(second).expect_err("over the rate");
            assert!(denied.retry_after >= 1, "at second {second}");
        }
    }
}

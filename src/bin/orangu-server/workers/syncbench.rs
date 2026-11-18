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

//! What dividing every layer between two nodes would cost in
//! synchronisation: `POST /v1/workers/sync-bench`.
//!
//! Splitting each layer's matrices between two nodes halves the weights each
//! reads a token, but every layer then sums its two halves twice — after
//! attention and after the feed-forward — so each node sends the other a
//! hidden-size vector `2 × n_layer` times a token and waits for the other's.
//! [`exchange`] times exactly that over the worker transport: a node echoes
//! a hidden-size payload off a peer of its channel ([`Message::Echo`]),
//! `2 × layers` times a simulated token, and [`report`] sets the time it took
//! beside this node's predicted decode alone.

use super::link::ChildLink;
use super::protocol::{Message, WorkerError};
use std::time::{Duration, Instant};

/// The most simulated tokens one benchmark runs.
pub const MAX_TOKENS: usize = 1000;
/// The most layers a simulated token has.
pub const MAX_LAYERS: usize = 512;
/// The widest hidden state, in values.
pub const MAX_HIDDEN: usize = 65536;
/// The most round trips one payload format may take: a benchmark is held
/// to a few minutes on a slow link.
pub const MAX_ROUND_TRIPS: usize = 200_000;

/// What a benchmark was asked, with the model's own sizes where it did not
/// say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ask {
    /// The peer: its `[workers]` name, its address, or its node id.
    pub peer: String,
    pub tokens: usize,
    pub layers: usize,
    pub hidden: usize,
}

impl Ask {
    /// Checks the sizes against the limits.
    pub fn check(&self) -> Result<(), String> {
        if self.peer.trim().is_empty() {
            return Err("name a peer: its [workers] name or address".to_string());
        }
        if !(1..=MAX_TOKENS).contains(&self.tokens) {
            return Err(format!("tokens must be 1 to {MAX_TOKENS}"));
        }
        if !(1..=MAX_LAYERS).contains(&self.layers) {
            return Err(format!("layers must be 1 to {MAX_LAYERS}"));
        }
        if !(1..=MAX_HIDDEN).contains(&self.hidden) {
            return Err(format!("hidden must be 1 to {MAX_HIDDEN}"));
        }
        if self.tokens * self.round_trips() > MAX_ROUND_TRIPS {
            return Err(format!(
                "tokens × 2 × layers must be at most {MAX_ROUND_TRIPS}"
            ));
        }
        Ok(())
    }

    /// Round trips a simulated token: two sums a layer.
    pub fn round_trips(&self) -> usize {
        2 * self.layers
    }
}

/// Echoes `bytes` bytes off the worker behind `link`, `round_trips` times a
/// token for `tokens` tokens, after one token untimed: each round trip's
/// time, by token.
pub fn exchange(
    link: &ChildLink,
    tokens: usize,
    round_trips: usize,
    bytes: usize,
) -> Result<Vec<Vec<Duration>>, WorkerError> {
    // Something like activations, not a run of one byte.
    let data: Vec<u8> = (0..bytes).map(|i| (i * 131 + 7) as u8).collect();
    let message = Message::Echo { data };
    let once = || -> Result<Duration, WorkerError> {
        let at = Instant::now();
        match link.request(&message)? {
            Message::Echo { data } if data.len() == bytes => Ok(at.elapsed()),
            Message::Error(error) => Err(error),
            other => Err(WorkerError::new(
                super::protocol::ErrorCode::Internal,
                format!(
                    "expected {bytes} bytes back, got {}",
                    match other {
                        Message::Echo { data } => format!("{} bytes", data.len()),
                        other => format!("{other:?}"),
                    }
                ),
            )),
        }
    };
    for _ in 0..round_trips {
        once()?;
    }
    let mut out = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        let mut token = Vec::with_capacity(round_trips);
        for _ in 0..round_trips {
            token.push(once()?);
        }
        out.push(token);
    }
    Ok(out)
}

/// The `q` quantile of `values` (sorted in place), by nearest rank; `0`
/// for none.
pub fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let rank = (q * values.len() as f64).ceil() as usize;
    values[rank.clamp(1, values.len()) - 1]
}

/// One payload format's numbers: a token's sync time (every round trip of
/// it added up) and a round trip's, median and 90th percentile, in
/// milliseconds, and the payload's throughput — its bytes there and back
/// over the time they took.
#[derive(Clone, Debug, PartialEq)]
pub struct Timing {
    pub payload_bytes: usize,
    pub token_median_ms: f64,
    pub token_p90_ms: f64,
    pub round_trip_median_ms: f64,
    pub round_trip_p90_ms: f64,
    pub mb_per_s: f64,
}

impl Timing {
    pub fn of(times: &[Vec<Duration>], payload_bytes: usize) -> Self {
        let ms = |d: &Duration| d.as_secs_f64() * 1e3;
        let mut tokens: Vec<f64> = times.iter().map(|t| t.iter().map(ms).sum()).collect();
        let mut trips: Vec<f64> = times.iter().flatten().map(ms).collect();
        let total_s: f64 = trips.iter().sum::<f64>() / 1e3;
        let moved = 2.0 * payload_bytes as f64 * trips.len() as f64;
        Self {
            payload_bytes,
            token_median_ms: quantile(&mut tokens, 0.5),
            token_p90_ms: quantile(&mut tokens, 0.9),
            round_trip_median_ms: quantile(&mut trips, 0.5),
            round_trip_p90_ms: quantile(&mut trips, 0.9),
            mb_per_s: if total_s > 0.0 {
                moved / total_s / 1e6
            } else {
                0.0
            },
        }
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "payload_bytes": self.payload_bytes,
            "per_token_ms": { "median": self.token_median_ms, "p90": self.token_p90_ms },
            "round_trip_ms": { "median": self.round_trip_median_ms, "p90": self.round_trip_p90_ms },
            "mb_per_s": self.mb_per_s,
        })
    }
}

/// The go/no-go arithmetic for two nodes splitting every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// This node's decode step alone: every layer and the head at its
    /// measured rate, and its fixed cost.
    pub alone_ms: f64,
    /// Half of every layer and of the head on each node, at the slower of
    /// the two rates, and the fixed cost once: the two halves run at once.
    pub tp2_compute_ms: f64,
    /// A token's sync time, median, in the format hidden states travel in.
    pub sync_ms: f64,
    pub tp2_total_ms: f64,
}

impl Prediction {
    /// `bytes` of weights a token (layers and head), this node's decode
    /// `rate` (GB a second) and `fixed_ms`, the peer's rate (`None` when it
    /// did not measure: taken as this node's), and the sync time.
    pub fn new(
        bytes: u64,
        rate: f64,
        fixed_ms: f64,
        peer_rate: Option<f64>,
        sync_ms: f64,
    ) -> Option<Self> {
        let alone_ms = super::divide::step_ms(bytes, rate, fixed_ms)?;
        let slower = peer_rate.filter(|r| *r > 0.0).map_or(rate, |r| r.min(rate));
        let tp2_compute_ms = super::divide::step_ms(bytes / 2, slower, fixed_ms)?;
        Some(Self {
            alone_ms,
            tp2_compute_ms,
            sync_ms,
            tp2_total_ms: tp2_compute_ms + sync_ms,
        })
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "alone_ms": self.alone_ms,
            "tp2_compute_ms": self.tp2_compute_ms,
            "sync_ms": self.sync_ms,
            "tp2_total_ms": self.tp2_total_ms,
            "pays": self.tp2_total_ms < self.alone_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_go_by_nearest_rank() {
        let mut values = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(&mut values, 0.5), 3.0);
        assert_eq!(quantile(&mut values, 0.9), 5.0);
        assert_eq!(quantile(&mut values, 0.0), 1.0);
        let mut ten: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(quantile(&mut ten, 0.9), 9.0);
        assert_eq!(quantile(&mut [], 0.5), 0.0);
    }

    /// Three tokens of two round trips: each token's sum, each trip, and
    /// the bytes there and back over the time.
    #[test]
    fn a_timing_adds_a_token_up_from_its_round_trips() {
        let ms = Duration::from_millis;
        let times = vec![vec![ms(1), ms(1)], vec![ms(1), ms(3)], vec![ms(2), ms(2)]];
        let timing = Timing::of(&times, 1000);
        assert_eq!(timing.token_median_ms, 4.0);
        assert_eq!(timing.token_p90_ms, 4.0);
        assert_eq!(timing.round_trip_median_ms, 1.0);
        assert_eq!(timing.round_trip_p90_ms, 3.0);
        // 6 trips × 2 kB in 10 ms.
        assert!((timing.mb_per_s - 1.2).abs() < 1e-9, "{}", timing.mb_per_s);
        let json = timing.json();
        assert_eq!(json["per_token_ms"]["median"], 4.0);
        assert_eq!(json["payload_bytes"], 1000);
    }

    /// A node that reads 2 GB a token at 20 GB/s with 2 ms fixed: 102 ms
    /// alone; split in two, 52 ms of compute, plus the sync.
    #[test]
    fn two_halves_run_at_once_and_pay_the_sync() {
        let p = Prediction::new(2_000_000_000, 20.0, 2.0, None, 30.0).unwrap();
        assert_eq!(p.alone_ms, 102.0);
        assert_eq!(p.tp2_compute_ms, 52.0);
        assert_eq!(p.tp2_total_ms, 82.0);
        assert_eq!(p.json()["pays"], true);
        // A peer half as fast: its half takes as long as the whole here.
        let p = Prediction::new(2_000_000_000, 20.0, 2.0, Some(10.0), 30.0).unwrap();
        assert_eq!(p.tp2_compute_ms, 102.0);
        assert_eq!(p.json()["pays"], false);
        // A faster peer waits for this node's half.
        let p = Prediction::new(2_000_000_000, 20.0, 2.0, Some(80.0), 0.0).unwrap();
        assert_eq!(p.tp2_compute_ms, 52.0);
        assert_eq!(Prediction::new(1, 0.0, 0.0, None, 1.0), None);
    }

    #[test]
    fn an_ask_is_held_to_its_limits() {
        let ask = Ask {
            peer: "beta".to_string(),
            tokens: 50,
            layers: 35,
            hidden: 1536,
        };
        assert_eq!(ask.check(), Ok(()));
        assert_eq!(ask.round_trips(), 70);
        for bad in [
            Ask {
                peer: " ".to_string(),
                ..ask.clone()
            },
            Ask {
                tokens: 0,
                ..ask.clone()
            },
            Ask {
                layers: MAX_LAYERS + 1,
                ..ask.clone()
            },
            Ask {
                hidden: MAX_HIDDEN + 1,
                ..ask.clone()
            },
            Ask {
                tokens: MAX_TOKENS,
                layers: MAX_LAYERS,
                ..ask.clone()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
    }
}

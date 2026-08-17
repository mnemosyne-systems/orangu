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

//! How well the *next* MoE layer's routing can be guessed before its turn.
//!
//! Streaming an expert off a disk costs milliseconds; deciding which expert
//! costs microseconds. The whole value of a prefetch is that the decision can
//! be made early enough to hide the cost — which is only true if the early
//! decision is usually the same as the real one.
//!
//! colibri reports routing as **71.6% predictable one layer ahead** and builds
//! its `PILOT` prefetcher on that. That is a number about *its* model, and
//! porting a lever without first sizing it on the target is a mistake this
//! project has a lesson written about. So this module measures the same thing
//! on orangu's own routing, and nothing prefetches anything until it has.
//!
//! # What is being predicted
//!
//! Layer `L+1`'s router reads the residual stream after layer `L`'s FFN and
//! after `L+1`'s own attention. The prediction runs that same router earlier —
//! on the stream as it stands partway through layer `L` — so it is wrong by
//! exactly the two sub-layers it skipped. Whether that matters is an empirical
//! question about how much those sub-layers move the router's argmax, which is
//! what the counters here answer.
//!
//! # How accuracy is counted
//!
//! Per position, per layer: how many of the experts the router *actually*
//! chose were in the predicted set. Summed over positions and divided by the
//! total selections, matching colibri's own `la_hit[]`/`la_tot[]` accounting
//! so the two numbers mean the same thing and can be put side by side.
//!
//! Set overlap rather than exact-sequence equality on purpose: a prefetcher
//! does not care in which order the experts were ranked, only whether the
//! bytes it fetched are the bytes that got used.
//!
//! **Off by default** (`ORANGU_ROUTE_AHEAD=1`). The prediction is a real
//! router matmul per layer, and paying for it on every request to collect a
//! statistic nothing consumes yet would be exactly the unmeasured cost this
//! engine keeps refusing to add.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static PREDICTED_SELECTIONS: AtomicU64 = AtomicU64::new(0);
static MATCHED_SELECTIONS: AtomicU64 = AtomicU64::new(0);
static SCORED_LAYERS: AtomicU64 = AtomicU64::new(0);

/// How many prediction ranks are tracked separately. A prefetcher does not
/// have to fetch every expert the lookahead names — colibri caps it at
/// `PILOT_K` — and the *precision of the first k* is what decides how much of
/// a narrower prefetch would be wasted. Aggregate accuracy cannot answer that:
/// it averages a confident first guess together with a speculative eighth.
const RANKS: usize = 16;

/// Predictions issued at each rank, and how many of them were used.
static BY_RANK_ISSUED: [AtomicU64; RANKS] = [const { AtomicU64::new(0) }; RANKS];
static BY_RANK_USED: [AtomicU64; RANKS] = [const { AtomicU64::new(0) }; RANKS];

thread_local! {
    /// The most recent prediction for each layer, waiting for that layer to
    /// route for real.
    ///
    /// Thread-local because one forward pass walks its layers on one thread —
    /// the parallelism inside a layer never crosses layers — so a prediction
    /// made at layer `L` and scored at layer `L+1` is always the same
    /// thread's. A shared map would need a lock on the hot path to hold data
    /// that is never shared.
    static PENDING: RefCell<Vec<Option<Vec<Vec<usize>>>>> = const { RefCell::new(Vec::new()) };
}

/// How many of the lookahead's predictions per position to actually prefetch,
/// from `ORANGU_PREFETCH_K`. `0` (the default) predicts but fetches nothing.
///
/// **Narrow on purpose.** Measured on `gemma-4-26B-A4B`, the precision of the
/// lookahead's top `k` falls steeply with `k`: 87.0% at 1, 83.3% at 2, 78.9%
/// at 3, 60.8% across the full top-8. Since every named expert costs its bytes
/// whether or not the router agrees later, `1 - precision` *is* the share of a
/// prefetch that is wasted — so a 20% waste budget is met only at `k <= 2`,
/// and fetching the whole predicted set would throw away 39% of what it moved.
/// colibri caps the same thing with `PILOT_K` for the same reason.
pub fn prefetch_width() -> usize {
    static WIDTH: OnceLock<usize> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        std::env::var("ORANGU_PREFETCH_K")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Whether to run the lookahead router at all.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| crate::engine::env::flag_on("ORANGU_ROUTE_AHEAD") || prefetch_width() > 0)
}

/// Records what layer `layer`'s router is expected to pick, one predicted
/// expert set per position.
pub fn predict(layer: usize, predicted: Vec<Vec<usize>>) {
    PENDING.with_borrow_mut(|pending| {
        if pending.len() <= layer {
            pending.resize_with(layer + 1, || None);
        }
        pending[layer] = Some(predicted);
    });
}

/// Scores layer `layer`'s real routing against whatever was predicted for it,
/// then forgets the prediction — a stale one must never be scored twice, or a
/// layer that was predicted once and routed twice would count as two
/// successes.
pub fn score(layer: usize, actual: &[Vec<(usize, f32)>]) {
    let Some(predicted) = PENDING
        .with_borrow_mut(|pending| pending.get_mut(layer).and_then(std::option::Option::take))
    else {
        return;
    };
    let mut total = 0u64;
    let mut matched = 0u64;
    for (chosen, guess) in actual.iter().zip(&predicted) {
        for (expert, _) in chosen {
            total += 1;
            if guess.contains(expert) {
                matched += 1;
            }
        }
        // The other direction: of the experts the lookahead *named*, at each
        // rank, how many were actually wanted. This is the precision a
        // prefetcher pays for — every named expert it fetches costs bytes
        // whether or not the router agrees later.
        for (rank, expert) in guess.iter().take(RANKS).enumerate() {
            BY_RANK_ISSUED[rank].fetch_add(1, Ordering::Relaxed);
            if chosen.iter().any(|(e, _)| e == expert) {
                BY_RANK_USED[rank].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    PREDICTED_SELECTIONS.fetch_add(total, Ordering::Relaxed);
    MATCHED_SELECTIONS.fetch_add(matched, Ordering::Relaxed);
    SCORED_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// Forgets every outstanding prediction. Called when a forward pass starts, so
/// a prediction left behind by an abandoned pass cannot be scored against an
/// unrelated one.
pub fn reset() {
    PENDING.with_borrow_mut(std::vec::Vec::clear);
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RouteAheadStats {
    /// Expert selections that were compared against a prediction.
    pub selections: u64,
    /// Of those, how many the prediction had already named.
    pub matched: u64,
    pub scored_layers: u64,
    /// Per prediction rank: how many were issued, and how many were used.
    pub issued_by_rank: [u64; RANKS],
    pub used_by_rank: [u64; RANKS],
}

impl RouteAheadStats {
    /// `None` when nothing was predicted — never `0.0`, which would read as
    /// "predicted everything wrong" rather than "did not predict".
    pub fn accuracy(&self) -> Option<f64> {
        (self.selections > 0).then(|| self.matched as f64 / self.selections as f64)
    }

    /// Of the first `k` predictions per position, the share that were
    /// actually routed to — one minus the share of a `k`-wide prefetch's
    /// bytes that would have been wasted.
    pub fn precision_at(&self, k: usize) -> Option<f64> {
        let k = k.min(RANKS);
        let issued: u64 = self.issued_by_rank[..k].iter().sum();
        let used: u64 = self.used_by_rank[..k].iter().sum();
        (issued > 0).then(|| used as f64 / issued as f64)
    }

    pub fn to_json(self) -> serde_json::Value {
        let precision: Vec<serde_json::Value> = (1..=RANKS)
            .map_while(|k| self.precision_at(k).map(|p| serde_json::json!(p)))
            .collect();
        serde_json::json!({
            "selections": self.selections,
            "matched": self.matched,
            "scored_layers": self.scored_layers,
            "accuracy": self.accuracy(),
            // `precision[k-1]` is the precision of a prefetch that takes the
            // lookahead's top `k`.
            "precision_at_k": precision,
        })
    }
}

pub fn take() -> RouteAheadStats {
    RouteAheadStats {
        selections: PREDICTED_SELECTIONS.swap(0, Ordering::Relaxed),
        matched: MATCHED_SELECTIONS.swap(0, Ordering::Relaxed),
        scored_layers: SCORED_LAYERS.swap(0, Ordering::Relaxed),
        issued_by_rank: std::array::from_fn(|r| BY_RANK_ISSUED[r].swap(0, Ordering::Relaxed)),
        used_by_rank: std::array::from_fn(|r| BY_RANK_USED[r].swap(0, Ordering::Relaxed)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-wide; two draining tests interleaving would
    /// each see the other's numbers.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn chosen(experts: &[usize]) -> Vec<(usize, f32)> {
        experts.iter().map(|&e| (e, 1.0)).collect()
    }

    #[test]
    fn a_perfect_prediction_scores_every_selection() {
        let _guard = serial();
        let _ = take();
        reset();

        predict(3, vec![vec![1, 2, 3], vec![4, 5, 6]]);
        score(3, &[chosen(&[3, 1, 2]), chosen(&[6, 5, 4])]);

        let stats = take();
        assert_eq!(stats.selections, 6);
        assert_eq!(stats.matched, 6);
        assert_eq!(stats.accuracy(), Some(1.0));
    }

    /// Overlap, not order: a prefetcher cares which bytes it fetched, not how
    /// the router ranked them.
    #[test]
    fn ranking_differences_do_not_count_as_misses() {
        let _guard = serial();
        let _ = take();
        reset();

        predict(0, vec![vec![9, 4, 7]]);
        score(0, &[chosen(&[7, 9, 4])]);
        assert_eq!(take().accuracy(), Some(1.0));
    }

    #[test]
    fn a_partial_prediction_scores_the_part_it_got() {
        let _guard = serial();
        let _ = take();
        reset();

        predict(1, vec![vec![1, 2, 3, 4]]);
        score(1, &[chosen(&[1, 2, 30, 40])]);

        let stats = take();
        assert_eq!(stats.selections, 4);
        assert_eq!(stats.matched, 2);
        assert_eq!(stats.accuracy(), Some(0.5));
    }

    /// A layer that routes without a prediction outstanding must not be
    /// counted at all — folding it in as a miss would understate accuracy by
    /// however many layers are simply not predicted.
    #[test]
    fn an_unpredicted_layer_is_not_scored() {
        let _guard = serial();
        let _ = take();
        reset();

        score(7, &[chosen(&[1, 2])]);
        assert_eq!(take(), RouteAheadStats::default());
    }

    /// A prediction is consumed when scored. Scoring twice against one
    /// prediction would credit a layer that was only guessed once.
    #[test]
    fn a_prediction_is_scored_at_most_once() {
        let _guard = serial();
        let _ = take();
        reset();

        predict(2, vec![vec![1]]);
        score(2, &[chosen(&[1])]);
        score(2, &[chosen(&[1])]);

        let stats = take();
        assert_eq!(stats.selections, 1, "the second scoring found nothing");
        assert_eq!(stats.scored_layers, 1);
    }

    /// An abandoned pass must not leave a prediction that a later, unrelated
    /// pass gets credit for.
    #[test]
    fn reset_drops_outstanding_predictions() {
        let _guard = serial();
        let _ = take();
        reset();

        predict(4, vec![vec![1, 2]]);
        reset();
        score(4, &[chosen(&[1, 2])]);
        assert_eq!(take(), RouteAheadStats::default());
    }

    #[test]
    fn an_empty_window_reports_no_accuracy() {
        assert_eq!(RouteAheadStats::default().accuracy(), None);
    }
}

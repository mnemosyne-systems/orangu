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

//! Unserved demand: requests a node got for a model no free node of its
//! channel served, counted per model over the last [`WINDOW`]. A node shares
//! its counts in its probe answers ([`super::protocol::Message::ProbeAck`]),
//! so every node of the channel sees what is wanted where.

use super::protocol::{MAX_DEMAND, Wanted};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How far back a request counts.
pub const WINDOW: Duration = Duration::from_secs(300);

/// Requests within one bucket of this length share a count: what bounds a
/// model's history to `WINDOW / BUCKET` entries however many requests come.
const BUCKET: Duration = Duration::from_secs(5);

/// The most models a node keeps counts for; a new one past it replaces the
/// one asked for least recently.
const MAX_MODELS: usize = 64;

/// The longest model name counted, in bytes: a longer `model` field is no
/// model's name.
pub const MAX_NAME_BYTES: usize = 256;

/// One model's requests: `(bucket start, requests)`, oldest first.
#[derive(Debug)]
struct Model {
    name: String,
    buckets: VecDeque<(Instant, u32)>,
}

impl Model {
    fn drop_expired(&mut self, now: Instant) {
        while self
            .buckets
            .front()
            .is_some_and(|(start, _)| now.saturating_duration_since(*start) >= WINDOW)
        {
            self.buckets.pop_front();
        }
    }

    fn total(&self) -> u32 {
        self.buckets
            .iter()
            .fold(0u32, |sum, (_, n)| sum.saturating_add(*n))
    }

    fn last(&self) -> Option<Instant> {
        self.buckets.back().map(|(start, _)| *start)
    }
}

/// A node's own unserved requests, per model, over the last [`WINDOW`].
#[derive(Debug, Default)]
pub struct Window {
    models: Vec<Model>,
}

impl Window {
    /// Counts one request for `name` at `now`. Names compare ignoring case;
    /// an empty or overlong one is not counted.
    pub fn note(&mut self, name: &str, now: Instant) {
        let name = name.trim();
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return;
        }
        let index = match self
            .models
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(name))
        {
            Some(index) => index,
            None => {
                self.prune(now);
                if self.models.len() >= MAX_MODELS
                    && let Some(oldest) = self
                        .models
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, m)| m.last())
                        .map(|(i, _)| i)
                {
                    self.models.swap_remove(oldest);
                }
                self.models.push(Model {
                    name: name.to_string(),
                    buckets: VecDeque::new(),
                });
                self.models.len() - 1
            }
        };
        let model = &mut self.models[index];
        model.drop_expired(now);
        match model.buckets.back_mut() {
            Some((start, n)) if now.saturating_duration_since(*start) < BUCKET => {
                *n = n.saturating_add(1);
            }
            _ => model.buckets.push_back((now, 1)),
        }
    }

    /// Drops what has left the window, and the models left with nothing.
    fn prune(&mut self, now: Instant) {
        for model in &mut self.models {
            model.drop_expired(now);
        }
        self.models.retain(|m| !m.buckets.is_empty());
    }

    /// Each model's requests within the window at `now`, most first, then
    /// by name.
    pub fn counts(&mut self, now: Instant) -> Vec<Wanted> {
        self.prune(now);
        let mut out: Vec<Wanted> = self
            .models
            .iter()
            .map(|m| Wanted {
                model: m.name.clone(),
                count: m.total(),
            })
            .collect();
        out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.model.cmp(&b.model)));
        out
    }

    /// What a probe answer carries: the [`MAX_DEMAND`] most wanted.
    pub fn shared(&mut self, now: Instant) -> Vec<Wanted> {
        let mut out = self.counts(now);
        out.truncate(MAX_DEMAND);
        out
    }
}

/// Adds `more` to `into`, one entry per name ignoring case.
pub fn merge(into: &mut Vec<Wanted>, more: &[Wanted]) {
    for wanted in more {
        match into
            .iter_mut()
            .find(|w| w.model.eq_ignore_ascii_case(&wanted.model))
        {
            Some(w) => w.count = w.count.saturating_add(wanted.count),
            None => into.push(wanted.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_of(window: &mut Window, name: &str, now: Instant) -> u32 {
        window
            .counts(now)
            .iter()
            .find(|w| w.model == name)
            .map_or(0, |w| w.count)
    }

    #[test]
    fn requests_are_counted_per_model_ignoring_case() {
        let mut window = Window::default();
        let t0 = Instant::now();
        window.note("org/a:Q4_K_M", t0);
        window.note("ORG/A:q4_k_m", t0 + Duration::from_secs(1));
        window.note("org/a:Q4_K_M", t0 + Duration::from_secs(30));
        window.note("gpt-4", t0 + Duration::from_secs(31));
        let counts = window.counts(t0 + Duration::from_secs(32));
        assert_eq!(
            counts,
            vec![
                Wanted {
                    model: "org/a:Q4_K_M".to_string(),
                    count: 3
                },
                Wanted {
                    model: "gpt-4".to_string(),
                    count: 1
                },
            ]
        );
    }

    #[test]
    fn requests_leave_the_window_after_five_minutes() {
        let mut window = Window::default();
        let t0 = Instant::now();
        window.note("m", t0);
        window.note("m", t0 + Duration::from_secs(120));
        assert_eq!(count_of(&mut window, "m", t0 + Duration::from_secs(299)), 2);
        assert_eq!(count_of(&mut window, "m", t0 + Duration::from_secs(300)), 1);
        assert_eq!(count_of(&mut window, "m", t0 + Duration::from_secs(420)), 0);
        // A model left with nothing is gone from the list.
        assert!(window.counts(t0 + Duration::from_secs(420)).is_empty());
    }

    #[test]
    fn empty_and_overlong_names_are_not_counted() {
        let mut window = Window::default();
        let t0 = Instant::now();
        window.note("  ", t0);
        window.note(&"x".repeat(MAX_NAME_BYTES + 1), t0);
        assert!(window.counts(t0).is_empty());
    }

    #[test]
    fn a_probe_shares_the_most_wanted_only() {
        let mut window = Window::default();
        let t0 = Instant::now();
        for i in 0..MAX_DEMAND + 4 {
            for _ in 0..=i {
                window.note(&format!("m{i}"), t0);
            }
        }
        let shared = window.shared(t0);
        assert_eq!(shared.len(), MAX_DEMAND);
        assert_eq!(shared[0].model, format!("m{}", MAX_DEMAND + 3));
        assert!(shared.windows(2).all(|w| w[0].count >= w[1].count));
    }

    #[test]
    fn past_the_model_cap_the_least_recent_gives_way() {
        let mut window = Window::default();
        let t0 = Instant::now();
        for i in 0..MAX_MODELS {
            window.note(&format!("m{i}"), t0 + Duration::from_secs(i as u64));
        }
        window.note("new", t0 + Duration::from_secs(MAX_MODELS as u64));
        let names: Vec<String> = window
            .counts(t0 + Duration::from_secs(MAX_MODELS as u64))
            .into_iter()
            .map(|w| w.model)
            .collect();
        assert_eq!(names.len(), MAX_MODELS);
        assert!(names.contains(&"new".to_string()));
        assert!(!names.contains(&"m0".to_string()));
    }

    #[test]
    fn merging_adds_counts_by_name() {
        let mut into = vec![Wanted {
            model: "A".to_string(),
            count: 2,
        }];
        merge(
            &mut into,
            &[
                Wanted {
                    model: "a".to_string(),
                    count: 3,
                },
                Wanted {
                    model: "b".to_string(),
                    count: 1,
                },
            ],
        );
        assert_eq!(
            into,
            vec![
                Wanted {
                    model: "A".to_string(),
                    count: 5
                },
                Wanted {
                    model: "b".to_string(),
                    count: 1
                },
            ]
        );
    }
}

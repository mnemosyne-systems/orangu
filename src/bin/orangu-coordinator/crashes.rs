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

//! A profile whose `orangu-server` keeps crashing is paused: after
//! [`CRASHES_TO_PAUSE`] unexplained crashes within [`CRASH_WINDOW`], it is not
//! started again for [`PAUSE_FOR`], and its requests are answered `503` with
//! a `Retry-After`. A server that exits over a lost GPU device, or one this
//! coordinator stopped, has not crashed and is not counted.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Crashes within [`CRASH_WINDOW`] that pause a profile.
pub const CRASHES_TO_PAUSE: usize = 3;

/// How far back a crash counts toward a pause.
pub const CRASH_WINDOW: Duration = Duration::from_secs(600);

/// How long a paused profile is not started.
pub const PAUSE_FOR: Duration = Duration::from_secs(600);

/// One profile's recent crashes, and the pause they led to.
#[derive(Debug, Default)]
struct Record {
    crashes: VecDeque<Instant>,
    paused_until: Option<Instant>,
}

/// Every profile's crashes and pauses.
#[derive(Debug, Default)]
pub struct Crashes {
    profiles: HashMap<String, Record>,
}

impl Crashes {
    /// Counts a crash of `profile`'s server at `now`. Returns when the pause
    /// ends when this crash starts one; counting then starts over.
    pub fn note(&mut self, profile: &str, now: Instant) -> Option<Instant> {
        let record = self.profiles.entry(profile.to_string()).or_default();
        while record
            .crashes
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= CRASH_WINDOW)
        {
            record.crashes.pop_front();
        }
        record.crashes.push_back(now);
        if record.crashes.len() < CRASHES_TO_PAUSE {
            return None;
        }
        record.crashes.clear();
        let until = now + PAUSE_FOR;
        record.paused_until = Some(until);
        Some(until)
    }

    /// When `profile`'s pause ends, while it is paused at `now`. A pause that
    /// has run out is forgotten: the profile is tried again.
    pub fn paused_until(&mut self, profile: &str, now: Instant) -> Option<Instant> {
        let record = self.profiles.get_mut(profile)?;
        match record.paused_until {
            Some(until) if now < until => Some(until),
            Some(_) => {
                record.paused_until = None;
                None
            }
            None => None,
        }
    }

    /// Every profile paused at `now`, with when its pause ends, by name.
    pub fn paused(&mut self, now: Instant) -> Vec<(String, Instant)> {
        let names: Vec<String> = self.profiles.keys().cloned().collect();
        let mut out: Vec<(String, Instant)> = names
            .into_iter()
            .filter_map(|name| {
                let until = self.paused_until(&name, now)?;
                Some((name, until))
            })
            .collect();
        out.sort();
        out
    }

    /// Forgets every crash and pause: what a reloaded configuration starts
    /// from.
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
}

/// The error an activation of a paused profile ends with: what the proxy
/// answers `503` with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paused {
    pub profile: String,
    /// How long until the pause ends.
    pub remaining: Duration,
}

impl Paused {
    /// The whole seconds a client should wait, at least one.
    pub fn retry_after_seconds(&self) -> u64 {
        self.remaining.as_secs_f64().ceil().max(1.0) as u64
    }
}

impl std::fmt::Display for Paused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "profile '{}' is paused after {CRASHES_TO_PAUSE} crashes within {} minutes; it is \
             started again in {}s",
            self.profile,
            CRASH_WINDOW.as_secs() / 60,
            self.retry_after_seconds()
        )
    }
}

impl std::error::Error for Paused {}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(t0: Instant, seconds: u64) -> Instant {
        t0 + Duration::from_secs(seconds)
    }

    #[test]
    fn three_crashes_within_ten_minutes_pause_the_profile() {
        let mut crashes = Crashes::default();
        let t0 = Instant::now();
        assert_eq!(crashes.note("main", t0), None);
        assert_eq!(crashes.note("main", at(t0, 60)), None);
        assert_eq!(crashes.paused_until("main", at(t0, 61)), None);
        let until = crashes.note("main", at(t0, 120)).expect("the third pauses");
        assert_eq!(until, at(t0, 120 + PAUSE_FOR.as_secs()));
        assert_eq!(crashes.paused_until("main", at(t0, 121)), Some(until));
        // Another profile is not touched.
        assert_eq!(crashes.paused_until("other", at(t0, 121)), None);
        assert_eq!(
            crashes.paused(at(t0, 121)),
            vec![("main".to_string(), until)]
        );
    }

    #[test]
    fn crashes_further_apart_than_the_window_do_not_pause() {
        let mut crashes = Crashes::default();
        let t0 = Instant::now();
        crashes.note("main", t0);
        crashes.note("main", at(t0, 300));
        // The first has left the window by the third.
        assert_eq!(crashes.note("main", at(t0, 600)), None);
        assert_eq!(crashes.paused_until("main", at(t0, 601)), None);
        // The fourth makes three within ten minutes.
        assert!(crashes.note("main", at(t0, 610)).is_some());
    }

    #[test]
    fn a_pause_runs_out_and_counting_starts_over() {
        let mut crashes = Crashes::default();
        let t0 = Instant::now();
        for s in [0, 1, 2] {
            crashes.note("main", at(t0, s));
        }
        let end = 2 + PAUSE_FOR.as_secs();
        assert!(crashes.paused_until("main", at(t0, end - 1)).is_some());
        assert_eq!(crashes.paused_until("main", at(t0, end)), None);
        assert!(crashes.paused(at(t0, end)).is_empty());
        // One more try: a single crash after the pause does not pause again.
        assert_eq!(crashes.note("main", at(t0, end + 1)), None);
        assert_eq!(crashes.paused_until("main", at(t0, end + 2)), None);
    }

    #[test]
    fn clearing_forgets_pauses_and_crashes() {
        let mut crashes = Crashes::default();
        let t0 = Instant::now();
        for s in [0, 1, 2] {
            crashes.note("main", at(t0, s));
        }
        crashes.note("other", at(t0, 3));
        crashes.note("other", at(t0, 4));
        crashes.clear();
        assert_eq!(crashes.paused_until("main", at(t0, 5)), None);
        // `other`'s two crashes are gone too: one more does not pause it.
        assert_eq!(crashes.note("other", at(t0, 6)), None);
    }

    #[test]
    fn a_pause_says_when_it_ends() {
        let paused = Paused {
            profile: "main".to_string(),
            remaining: Duration::from_millis(90_500),
        };
        assert_eq!(paused.retry_after_seconds(), 91);
        let text = paused.to_string();
        assert!(text.contains("'main'"), "{text}");
        assert!(text.contains("91s"), "{text}");
        let none_left = Paused {
            remaining: Duration::ZERO,
            ..paused
        };
        assert_eq!(none_left.retry_after_seconds(), 1);
    }
}

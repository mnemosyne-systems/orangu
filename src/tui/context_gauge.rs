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

//! The status line's context gauge: how many tokens the active conversation
//! fills of the model's context, for example `ctx 18k/32k`.

use std::sync::Mutex;

/// From this share of the context the gauge takes the theme's warning colour.
pub const WARN_PERCENT: u64 = 80;

/// How full one conversation's context is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextGauge {
    /// Tokens the conversation holds: the last response's prompt and answer.
    pub used: u64,
    /// The context size the server reports for one request, when it does.
    pub limit: Option<u64>,
    /// `used` is scaled from the last count after the conversation shrank,
    /// not counted by the server.
    pub estimated: bool,
}

impl ContextGauge {
    /// `used` as a whole percentage of `limit`.
    pub fn percent(&self) -> Option<u64> {
        self.limit
            .filter(|&limit| limit > 0)
            .map(|limit| self.used.saturating_mul(100) / limit)
    }

    /// Whether the conversation fills at least `percent` of the context.
    pub fn at_least(&self, percent: u64) -> bool {
        self.percent().is_some_and(|used| used >= percent)
    }

    /// Whether the gauge shows in the warning colour.
    pub fn is_high(&self) -> bool {
        self.at_least(WARN_PERCENT)
    }

    /// The status-line text: `ctx 18k/32k`, `ctx ~9k/32k` after a
    /// compaction, or `ctx 18k` when the server reports no context size.
    pub fn label(&self) -> String {
        let used = format!(
            "{}{}",
            if self.estimated { "~" } else { "" },
            short_count(self.used)
        );
        match self.limit {
            Some(limit) => format!("ctx {used}/{}", short_count(limit)),
            None => format!("ctx {used}"),
        }
    }
}

/// A token count in a few characters: `950`, `1.2k`, `18k`, `1.1M`.
pub fn short_count(count: u64) -> String {
    match count {
        0..1_000 => count.to_string(),
        1_000..10_000 => format!("{:.1}k", count as f64 / 1_000.0),
        10_000..1_000_000 => format!("{}k", count / 1_000),
        _ => format!("{:.1}M", count as f64 / 1_000_000.0),
    }
}

/// The gauge of the active tab's conversation; `None` hides it.
static SHOWN: Mutex<Option<ContextGauge>> = Mutex::new(None);

/// Makes `gauge` the one the status line shows.
pub fn show(gauge: Option<ContextGauge>) {
    if let Ok(mut shown) = SHOWN.lock() {
        *shown = gauge;
    }
}

/// The gauge the status line shows.
pub fn shown() -> Option<ContextGauge> {
    SHOWN.lock().ok().and_then(|shown| *shown)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_counts_stay_short() {
        assert_eq!(short_count(950), "950");
        assert_eq!(short_count(1_234), "1.2k");
        assert_eq!(short_count(18_400), "18k");
        assert_eq!(short_count(32_768), "32k");
        assert_eq!(short_count(1_100_000), "1.1M");
    }

    #[test]
    fn the_label_names_what_is_known() {
        let gauge = ContextGauge {
            used: 18_000,
            limit: Some(32_768),
            estimated: false,
        };
        assert_eq!(gauge.label(), "ctx 18k/32k");
        assert_eq!(gauge.percent(), Some(54));
        assert!(!gauge.is_high());
        let estimated = ContextGauge {
            used: 9_000,
            estimated: true,
            ..gauge
        };
        assert_eq!(estimated.label(), "ctx ~9.0k/32k");
        let unbounded = ContextGauge {
            limit: None,
            ..gauge
        };
        assert_eq!(unbounded.label(), "ctx 18k");
        assert_eq!(unbounded.percent(), None);
        assert!(!unbounded.is_high());
    }

    #[test]
    fn the_warning_starts_at_eighty_percent() {
        let gauge = |used| ContextGauge {
            used,
            limit: Some(1_000),
            estimated: false,
        };
        assert!(!gauge(799).is_high());
        assert!(gauge(800).is_high());
        assert!(gauge(1_200).is_high());
    }
}

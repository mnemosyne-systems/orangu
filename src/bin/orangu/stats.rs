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

use crate::*;

pub(crate) struct UsageStats {
    pub(crate) app_start: std::time::Instant,
    pub(crate) total_llm_duration: std::time::Duration,
    pub(crate) total_tool_duration: std::time::Duration,
    pub(crate) total_tokens: usize,
    pub(crate) session_id: String,
}

impl UsageStats {
    pub(crate) fn new() -> Self {
        Self {
            app_start: std::time::Instant::now(),
            total_llm_duration: std::time::Duration::ZERO,
            total_tool_duration: std::time::Duration::ZERO,
            total_tokens: 0,
            session_id: String::new(),
        }
    }

    pub(crate) fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = session_id.to_string();
        self
    }

    /// Record the time spent on a turn, splitting it into tool time and LLM
    /// time. Called for every outcome — success, cancellation, and failure —
    /// so the LLM time before a failure or cancellation is still counted.
    pub(crate) fn record_elapsed(
        &mut self,
        total_duration: std::time::Duration,
        tool_duration: std::time::Duration,
    ) {
        self.total_tool_duration += tool_duration;
        self.total_llm_duration += total_duration.saturating_sub(tool_duration);
    }

    pub(crate) fn record_response(
        &mut self,
        total_duration: std::time::Duration,
        response: &str,
        tool_duration: std::time::Duration,
    ) {
        self.record_elapsed(total_duration, tool_duration);
        if let Ok(tokenizer) = cl100k_base() {
            self.total_tokens += tokenizer.encode_with_special_tokens(response).len();
        }
    }

    pub(crate) fn format(&self, tools: &orangu::tools::ToolExecutor) -> String {
        let app_elapsed = self.app_start.elapsed();
        let avg_tps = if self.total_llm_duration.as_secs_f64() > 0.0 {
            self.total_tokens as f64 / self.total_llm_duration.as_secs_f64()
        } else {
            0.0
        };

        let mut out = format!(
            "Application time : {}\nLLM time         : {}\nTool time        : {}\nTotal tokens     : {}\nAvg tokens/sec   : {:.1}\nSession          : {}\nPID              : {}",
            format_duration(app_elapsed),
            format_duration(self.total_llm_duration),
            format_duration(self.total_tool_duration),
            self.total_tokens,
            avg_tps,
            self.session_id,
            std::process::id(),
        );

        if let Ok(cache) = tools.context_cache().lock() {
            let s = cache.stats();
            let saved_kb = s.bytes_saved as f64 / 1024.0;
            let hit_rate = if s.total_reads > 0 {
                (s.cache_hits as f64 / s.total_reads as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "\n\nContext Cache:\nTotal reads      : {}\nCache hits       : {}\nCache misses     : {}\nCache rate       : {:.1}%\nBytes saved      : {:.1} KB",
                s.total_reads,
                s.cache_hits,
                s.cache_misses,
                hit_rate,
                saved_kb
            ));
        }

        if let Ok(metrics) = tools.compression_metrics.lock() {
            let saved = metrics
                .total_original_lines
                .saturating_sub(metrics.total_compressed_lines);
            let saved_pct = if metrics.total_original_lines > 0 {
                (saved as f64 / metrics.total_original_lines as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "\n\nContext Compression:\nOriginal lines   : {}\nCompressed lines : {}\nLines saved      : {} ({:.1}%)",
                metrics.total_original_lines,
                metrics.total_compressed_lines,
                saved,
                saved_pct
            ));

            if !metrics.pattern_hits.is_empty() {
                out.push_str("\nPatterns applied :");
                let mut hits: Vec<_> = metrics.pattern_hits.iter().collect();
                hits.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
                for (pattern, count) in hits {
                    out.push_str(&format!("\n  - {} ({}x)", pattern, count));
                }
            }
        }

        out
    }
}

pub(crate) fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {}m {}s", h, m, s)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn record_elapsed_counts_llm_time_without_tokens() {
        use std::time::Duration;

        let mut stats = super::UsageStats::new();
        // A failed or cancelled turn: time is spent but no response is recorded.
        stats.record_elapsed(Duration::from_secs(5), Duration::from_secs(2));

        assert_eq!(stats.total_llm_duration, Duration::from_secs(3));
        assert_eq!(stats.total_tool_duration, Duration::from_secs(2));
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn record_response_counts_llm_time_and_tokens() {
        use std::time::Duration;

        let mut stats = super::UsageStats::new();
        stats.record_response(
            Duration::from_secs(4),
            "hello world",
            Duration::from_secs(1),
        );

        assert_eq!(stats.total_llm_duration, Duration::from_secs(3));
        assert_eq!(stats.total_tool_duration, Duration::from_secs(1));
        assert!(stats.total_tokens > 0);
    }
}

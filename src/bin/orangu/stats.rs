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
    pub(crate) workspace: std::path::PathBuf,
    pub(crate) skill_counts: std::collections::HashMap<String, usize>,
    /// The prompt-cache totals ([`orangu::llm::prompt_cache_totals`]) when
    /// the last turn ended, and that turn's own share of them.
    cache_at_turn_end: (u64, u64),
    last_turn_cache: Option<(u64, u64)>,
}

impl UsageStats {
    pub(crate) fn new() -> Self {
        Self {
            app_start: std::time::Instant::now(),
            total_llm_duration: std::time::Duration::ZERO,
            total_tool_duration: std::time::Duration::ZERO,
            total_tokens: 0,
            session_id: String::new(),
            workspace: std::path::PathBuf::new(),
            skill_counts: std::collections::HashMap::new(),
            cache_at_turn_end: orangu::llm::prompt_cache_totals(),
            last_turn_cache: None,
        }
    }

    /// Takes the prompt-cache totals at the end of a turn: what they grew by
    /// since the last turn ended is this turn's — every request it made, the
    /// tool rounds included. A turn that sent no prompt leaves the previous
    /// turn's figure.
    fn note_turn_cache(&mut self, totals: (u64, u64)) {
        let (prompt, cached) = (
            totals.0.saturating_sub(self.cache_at_turn_end.0),
            totals.1.saturating_sub(self.cache_at_turn_end.1),
        );
        self.cache_at_turn_end = totals;
        if prompt > 0 {
            self.last_turn_cache = Some((prompt, cached.min(prompt)));
        }
    }

    pub(crate) fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = session_id.to_string();
        self
    }

    pub(crate) fn record_skill(&mut self, name: &str) {
        *self.skill_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    pub(crate) fn with_workspace(mut self, workspace: &std::path::Path) -> Self {
        self.workspace = workspace.to_path_buf();
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
        self.record_turn(total_duration, tool_duration, 0);
    }

    pub(crate) fn record_response(
        &mut self,
        total_duration: std::time::Duration,
        response: &str,
        tool_duration: std::time::Duration,
    ) {
        let tokens = cl100k_base()
            .map(|tokenizer| tokenizer.encode_with_special_tokens(response).len())
            .unwrap_or(0);
        self.record_turn(total_duration, tool_duration, tokens);
    }

    /// The single choke point for every turn outcome: updates the in-memory
    /// session totals `/usage` reports, and appends one record to the
    /// persistent, per-workspace activity log `/statistics` reads back
    /// (`~/.orangu/workspace/<hash>/stats/activity.json`). Skips the disk
    /// write under test so `cargo test` never touches the real log.
    fn record_turn(
        &mut self,
        total_duration: std::time::Duration,
        tool_duration: std::time::Duration,
        tokens: usize,
    ) {
        self.total_tool_duration += tool_duration;
        let llm_duration = total_duration.saturating_sub(tool_duration);
        self.total_llm_duration += llm_duration;
        self.total_tokens += tokens;
        self.note_turn_cache(orangu::llm::prompt_cache_totals());

        if !cfg!(test) {
            crate::activity_log::append_activity(
                &self.workspace,
                &crate::activity_log::ActivityRecord {
                    day: crate::activity_log::today(),
                    session_id: self.session_id.clone(),
                    tokens,
                    llm_ms: llm_duration.as_millis() as u64,
                    tool_ms: tool_duration.as_millis() as u64,
                },
            );
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

        let (prompt, cached) = orangu::llm::prompt_cache_totals();
        if prompt > 0 {
            out.push_str(&format!(
                "\n\nPrompt cache:\nPrompt tokens    : {prompt}\nServed from cache: {cached}\nCache rate       : {:.1}%",
                cached as f64 * 100.0 / prompt as f64
            ));
            if let Some((prompt, cached)) = self.last_turn_cache {
                out.push_str(&format!(
                    "\nLast turn        : {cached} of {prompt} ({:.1}%)",
                    cached as f64 * 100.0 / prompt as f64
                ));
            }
        }

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

        if let Ok(counts) = tools.tool_counts.lock() {
            let total: usize = counts.values().sum();
            if total > 0 {
                out.push_str("\n\nTool Invocations:");
                let mut sorted: Vec<_> = counts.iter().collect();
                sorted.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
                for (name, count) in sorted {
                    let pct = (*count as f64 / total as f64) * 100.0;
                    out.push_str(&format!("\n  - {:<18.18} ({}x, {:.1}%)", name, count, pct));
                }
            }
        }

        let total_skills: usize = self.skill_counts.values().sum();
        if total_skills > 0 {
            out.push_str("\n\nSkill Invocations:");
            let mut sorted: Vec<_> = self.skill_counts.iter().collect();
            sorted.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
            for (name, count) in sorted {
                let pct = (*count as f64 / total_skills as f64) * 100.0;
                out.push_str(&format!("\n  - {:<18.18} ({}x, {:.1}%)", name, count, pct));
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
    fn the_last_turn_is_what_the_totals_grew_by() {
        let mut stats = super::UsageStats::new();
        stats.cache_at_turn_end = (1000, 900);
        stats.note_turn_cache((1500, 1300));
        assert_eq!(stats.last_turn_cache, Some((500, 400)));
        // A turn that sent nothing keeps the previous turn's figure.
        stats.note_turn_cache((1500, 1300));
        assert_eq!(stats.last_turn_cache, Some((500, 400)));
        // A turn that missed the cache shows as one, whatever the total says.
        stats.note_turn_cache((2500, 1300));
        assert_eq!(stats.last_turn_cache, Some((1000, 0)));
    }

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

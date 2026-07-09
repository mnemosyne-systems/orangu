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

mod auto_review;
mod header;
mod helper;
mod review;
mod screen;
mod text;

pub use auto_review::*;
pub use header::*;
pub use helper::*;
pub use review::*;
pub use screen::*;
pub use text::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptLine {
    Plain(String),
    UserInput(String),
    Wide(String),
}

impl TranscriptLine {
    pub fn as_str(&self) -> &str {
        match self {
            TranscriptLine::Plain(s) | TranscriptLine::UserInput(s) | TranscriptLine::Wide(s) => s,
        }
    }
}

pub(crate) const STATUS_GREEN: &str = "\x1b[38;2;80;200;120m";
pub(crate) const STATUS_RED: &str = "\x1b[38;2;220;80;80m";
pub(crate) const STATUS_WHITE: &str = "\x1b[38;2;230;230;230m";
/// The Ignore dot in `/auto_review`: a file marked Ignore (Alt+m) is skipped
/// from the run and shown with this blue dot.
pub(crate) const STATUS_BLUE: &str = "\x1b[38;2;100;160;230m";
/// The Deep dot in `/auto_review`: a file marked Deep (Alt+m) is reviewed with
/// extra passes (no diff compression, cross-file graph context, and a verify
/// pass on rejected findings) and shown with this purple dot before the run
/// starts.
pub(crate) const STATUS_PURPLE: &str = "\x1b[38;2;170;120;220m";
pub(crate) const ANSI_RESET: &str = "\x1b[0m";
pub const FEEDBACK_OK: &str = "\x1b[38;2;80;200;120m●\x1b[0m";
pub const FEEDBACK_ERR: &str = "\x1b[38;2;220;80;80m●\x1b[0m";

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    pub(crate) fn review_entry(
        path: &str,
        status: ReviewStatus,
        diff_lines: &[&str],
    ) -> ReviewEntry {
        ReviewEntry {
            path: path.to_string(),
            status,
            diff_lines: diff_lines.iter().map(|line| line.to_string()).collect(),
            patch: String::new(),
        }
    }

    pub(crate) fn review_args<'a>(
        files: &'a [ReviewEntry],
        selected: usize,
        scroll: usize,
        actual_width: usize,
        actual_height: usize,
    ) -> ReviewScreenArgs<'a> {
        ReviewScreenArgs {
            files,
            selected,
            line: 0,
            scroll,
            x_offset: 0,
            feedback: None,
            comment_editor: None,
            commented_lines: &[],
            current_model: "model",
            prompt_branch: None,
            input: "",
            cursor: 0,
            ghost: "",
            left_status: None,
            pending_count: 0,
            actual_width,
            actual_height,
        }
    }

    pub(crate) fn auto_review_args<'a>(
        files: &'a [ReviewEntry],
        report_lines: &'a [String],
        actual_width: usize,
        actual_height: usize,
    ) -> AutoReviewScreenArgs<'a> {
        AutoReviewScreenArgs {
            files,
            selected: None,
            reviewing: None,
            browsing: false,
            prestart: false,
            modes: &[],
            reject: None,
            diff: None,
            report_lines,
            selected_lines: None,
            scroll: 0,
            x_offset: 0,
            status: "File: a.rs (1/1)  Category: Code  Progress: 0/7 (0%)  Time: 5s",
            input: "",
            cursor: 0,
            ghost: "",
            current_model: "model",
            prompt_branch: Some("feature/x"),
            left_status: None,
            pending_count: 0,
            graph_status: None,
            actual_width,
            actual_height,
        }
    }
}

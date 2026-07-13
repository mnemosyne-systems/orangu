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

use super::*;

/// The Alt+r reject window of `/auto_review`, drawn over the panes: a category
/// selector and a multi-line Markdown comment editor; Tab moves the focus
/// between them.
pub struct AutoReviewRejectView<'a> {
    /// The file being rejected, shown in the title bar.
    pub path: &'a str,
    /// The report categories offered by the selector, in display order.
    pub categories: &'a [&'a str],
    /// Index of the chosen category.
    pub category: usize,
    /// `true` while the focus is on the category selector; `false` while it is
    /// on the comment editor (where the caret is then drawn).
    pub selector_focused: bool,
    /// The comment text, with embedded newlines.
    pub text: &'a str,
    /// Byte cursor within `text`.
    pub cursor: usize,
}

/// The Enter diff popup of `/auto_review`, drawn over the panes: a title bar
/// and the colorized diff of the changes under review (the `/diff` view),
/// scrolled with the Up/Down keys.
pub struct AutoReviewDiffView<'a> {
    /// The title shown in the bar (e.g. `Diff`).
    pub title: &'a str,
    /// The colorized diff lines (ANSI), drawn in the popup body.
    pub lines: &'a [String],
    pub scroll: usize,
    pub x_offset: usize,
}

/// Alt+m in the `/auto_review` pre-start phase cycles a file through these
/// three modes, in order: `Normal` is the default review; `Deep` still
/// reviews the file but with extra passes (no diff compression, cross-file
/// graph context, and a verify pass on rejected findings); `Ignore` skips the
/// file from the run entirely. Only `Ignore` changes what runs — `Deep` opts
/// into more, not less.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoReviewFileMode {
    #[default]
    Normal,
    Deep,
    Ignore,
}

impl AutoReviewFileMode {
    /// Alt+m: advance to the next mode in the Normal → Deep → Ignore → Normal
    /// cycle.
    pub fn next(self) -> Self {
        match self {
            Self::Normal => Self::Deep,
            Self::Deep => Self::Ignore,
            Self::Ignore => Self::Normal,
        }
    }
}

/// Inputs for the `/auto_review` screen: the categorized report in the left
/// pane — topped by the status area — and the file checklist (with auto-set
/// status dots) in the right pane.
pub struct AutoReviewScreenArgs<'a> {
    pub files: &'a [ReviewEntry],
    /// Index of the file highlighted in the right pane: the one being reviewed
    /// while the run is in progress, or the one picked with Alt+j/Alt+k while
    /// browsing afterwards. `None` shows no highlight (the run has ended and
    /// nothing has been picked).
    pub selected: Option<usize>,
    /// Scroll offset of the file checklist in the right pane.
    pub list_offset: usize,
    /// The rendered report lines shown in the left pane.
    pub report_lines: &'a [String],
    /// The line range (start inclusive, end exclusive, into `report_lines`) of
    /// the report item highlighted with the Up/Down item cursor while browsing.
    /// Those lines are drawn with the line-cursor background. `None` while the
    /// run is in progress or when no item is highlighted.
    pub selected_lines: Option<(usize, usize)>,
    pub scroll: usize,
    pub x_offset: usize,
    /// The status area's text: the file and category being worked on, e.g.
    /// `File: src/main.rs (2/5)  Category: Security`.
    pub status: &'a str,
    /// Index of the file whose status box shows the white "being reviewed"
    /// dot. The caller pulses this between `Some` and `None` on its render
    /// tick, which makes the dot blink.
    pub reviewing: Option<usize>,
    /// The run has ended and the report is being browsed: the header shows the
    /// browse keys (Alt+j/k, Alt+a, Alt+r, Alt+e) instead of the run keys.
    pub browsing: bool,
    /// The run has not started yet (pre-start phase): the header offers Alt+s
    /// Start, Alt+j/k Switch file, and Alt+m Mode, and Ignore/Deep files show
    /// their blue/purple dot. Cleared once the run begins.
    pub prestart: bool,
    /// Per-file mode (parallel to `files`), cycled with Alt+m. Read with
    /// `.get().copied().unwrap_or_default()` so a shorter (or empty) slice is
    /// treated as "every file Normal".
    pub modes: &'a [AutoReviewFileMode],
    /// When set, the Alt+r reject window is drawn over the panes.
    pub reject: Option<AutoReviewRejectView<'a>>,
    /// When set, the Enter diff popup is drawn over the panes.
    pub diff: Option<AutoReviewDiffView<'a>>,
    /// The input window contents. Empty while the run is in progress; once the
    /// run is done the browse loop fills it in so `/open_file <path>` and
    /// `open <path>` can open any project file in `$EDITOR`.
    pub input: &'a str,
    pub cursor: usize,
    /// The grey inline completion ghost drawn after the input cursor (empty for
    /// none), previewing the file path or command Tab would fill in.
    pub ghost: &'a str,
    pub current_model: &'a str,
    pub prompt_branch: Option<&'a str>,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    /// The knowledge graph's build status, shown as a `Graph: ●` indicator in
    /// the status bar — the one screen this is wired up for, since Deep
    /// mode's cross-file context depends on the graph having finished
    /// building. `None` while the caller hasn't resolved a status yet.
    pub graph_status: Option<ConnStatus>,
    pub actual_width: usize,
    pub actual_height: usize,
}

/// Number of scrollable body rows in the auto review report pane: one less
/// than the `/review` panes, since the status area takes the left pane's first
/// body row. `input` is empty while the run is in progress; once the run is
/// done the browse loop's `/open_file` input window can grow the prompt frame,
/// shrinking the report by the same rows the renderer reserves.
pub fn auto_review_pane_body_height(
    actual_height: usize,
    input: &str,
    prompt_branch: Option<&str>,
    actual_width: usize,
) -> usize {
    review_pane_body_height(actual_height, input, prompt_branch, actual_width)
        .saturating_sub(1)
        .max(1)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_review_pane_body_height_reserves_the_status_row() {
        // One row less than the `/review` panes (the status area takes it),
        // never less than one.
        let review = review_pane_body_height(24, "", Some("main"), 80);
        assert_eq!(
            auto_review_pane_body_height(24, "", Some("main"), 80),
            review - 1
        );
        assert_eq!(auto_review_pane_body_height(1, "", Some("main"), 80), 1);
    }
}

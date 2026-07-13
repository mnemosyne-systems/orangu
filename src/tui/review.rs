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

/// Review status for a single changed file in `/review` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewStatus {
    Unreviewed,
    Approved,
    Rejected,
}

/// One changed file in the review checklist, carrying its own diff lines so the
/// left pane can show just the selected file's diff.
#[derive(Clone, Debug)]
pub struct ReviewEntry {
    pub path: String,
    pub status: ReviewStatus,
    /// Colorized diff lines shown in the left pane.
    pub diff_lines: Vec<String>,
    /// Plain unified diff sent to the LLM when reviewing this file.
    pub patch: String,
}

/// The Alt+o feedback popup: the LLM's review of a file, shown over the panes.
pub struct ReviewFeedbackView<'a> {
    pub title: &'a str,
    /// The request that was asked, echoed below the title like a submitted
    /// prompt. `None` for a plain "review this file" request.
    pub question: Option<&'a str>,
    pub lines: &'a [String],
    pub scroll: usize,
    pub x_offset: usize,
}

/// The inline comment editor shown below the highlighted line: a single
/// category-selector row above the multi-line comment text.
pub struct ReviewCommentEditor<'a> {
    /// The chosen category name (e.g. `Overall`), shown on the selector row.
    pub category: &'a str,
    /// `true` while the focus is on the category selector (Up/Down move it);
    /// `false` while it is on the comment text (where the caret is drawn).
    pub selector_focused: bool,
    pub text: &'a str,
    pub cursor: usize,
}

pub struct ReviewScreenArgs<'a> {
    pub files: &'a [ReviewEntry],
    pub selected: usize,
    pub list_offset: usize,
    /// Highlighted line within the selected file's diff.
    pub line: usize,
    pub scroll: usize,
    pub x_offset: usize,
    /// When set, the feedback popup is drawn over the panes.
    pub feedback: Option<ReviewFeedbackView<'a>>,
    /// When set, the inline comment editor is drawn below the highlighted line.
    pub comment_editor: Option<ReviewCommentEditor<'a>>,
    /// Diff-line indices in the selected file that carry a comment.
    pub commented_lines: &'a [usize],
    pub current_model: &'a str,
    pub prompt_branch: Option<&'a str>,
    pub input: &'a str,
    pub cursor: usize,
    /// The grey inline completion ghost drawn after the input cursor (empty for
    /// none), previewing the file path or command Tab would fill in.
    pub ghost: &'a str,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub actual_width: usize,
    pub actual_height: usize,
}

/// The vertical pane separator used by `/review` and the manual viewer.
pub const REVIEW_SEPARATOR: &str = "\x1b[38;2;88;88;88m│\x1b[0m";
const REVIEW_LINE_CURSOR_BG: &str = "\x1b[48;2;60;60;90m";

/// Height of the inline comment editor window, in rows.
pub const REVIEW_COMMENT_BOX_HEIGHT: usize = 5;

/// Number of scrollable body rows in a review pane (or feedback popup): the
/// space above the prompt frame, minus the single header row.
pub fn review_pane_body_height(
    actual_height: usize,
    input: &str,
    prompt_branch: Option<&str>,
    actual_width: usize,
) -> usize {
    let prefix = prompt_prefix(prompt_branch);
    let input_lines = wrapped_input_lines(input, actual_width.max(1), &prefix);
    let prompt_frame_height = input_lines.len() + 3;
    actual_height
        .max(1)
        .saturating_sub(prompt_frame_height + 1)
        .max(1)
}

/// Width of the right (file list) pane: as small as possible while still
/// fitting the longest full file path, capped so the left pane stays usable.
pub fn review_right_width(files: &[ReviewEntry], actual_width: usize) -> usize {
    let actual_width = actual_width.max(1);
    let longest = files
        .iter()
        // "[x] " prefix is 4 visible columns, then the full path.
        .map(|file| 4 + file.path.chars().count())
        .max()
        .unwrap_or(0);
    let desired = longest.max("Files".chars().count());
    // Give the code (left pane) priority by capping the file list (right pane)
    // at 25% of the screen (or 25 columns on very small terminals).
    let cap = (actual_width / 4)
        .max(25)
        .min(actual_width.saturating_sub(2).max(1));
    desired.clamp(1, cap)
}

pub(crate) fn review_status_box(status: ReviewStatus) -> String {
    match status {
        ReviewStatus::Unreviewed => "[ ]".to_string(),
        ReviewStatus::Approved => format!("[{STATUS_GREEN}●{ANSI_RESET}]"),
        ReviewStatus::Rejected => format!("[{STATUS_RED}●{ANSI_RESET}]"),
    }
}

/// Clip `content` to `width` visible columns (honoring a horizontal pan) and
/// pad it with spaces so the cell occupies exactly `width` columns. This keeps
/// the vertical separator aligned in a single straight column on every row.
pub fn review_pane_cell(content: &str, x_offset: usize, width: usize) -> String {
    let mut cell = clip_line(content, x_offset, width);
    cell.push_str(ANSI_RESET);
    let visible = visible_line_width(&cell);
    if visible < width {
        cell.push_str(&" ".repeat(width - visible));
    }
    cell
}

/// Re-apply reverse video after every reset so a highlighted row stays
/// inverted across the embedded color codes (e.g. the status dot).
pub fn review_highlight(cell: &str) -> String {
    let reactivated = cell.replace(ANSI_RESET, &format!("{ANSI_RESET}\x1b[7m"));
    format!("\x1b[7m{reactivated}{ANSI_RESET}")
}

/// Apply a background to the whole cell — the highlighted line under the
/// Up/Down cursor — re-applying it after every reset so it spans the line's
/// own color codes and the trailing padding.
pub fn review_line_highlight(cell: &str) -> String {
    let reapplied = cell.replace(ANSI_RESET, &format!("{ANSI_RESET}{REVIEW_LINE_CURSOR_BG}"));
    format!("{REVIEW_LINE_CURSOR_BG}{reapplied}{ANSI_RESET}")
}

pub(crate) fn review_wrapped_lines(logical: &str, width: usize) -> Vec<String> {
    let mut lines = crate::tui::screen::wrapped_input_lines(logical, width, "");
    if !logical.is_empty() && logical.chars().count().is_multiple_of(width) {
        lines.push(String::new());
    }
    lines
}

/// Wrap multi-line text to `width` visible columns: each logical line (split
/// on `\n`) wraps independently, an empty logical line keeping its own row.
pub(crate) fn wrapped_multiline_lines(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    text.split('\n')
        .flat_map(|logical| review_wrapped_lines(logical, width))
        .collect()
}

/// The (row, column) of a byte cursor within multi-line text wrapped to
/// `width` columns — the multi-line counterpart of `cursor_position`.
pub(crate) fn multiline_cursor_position(text: &str, cursor: usize, width: usize) -> (usize, usize) {
    let width = width.max(1);
    let cursor = cursor.min(text.len());
    let line_start = text[..cursor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let mut row = 0usize;
    if line_start > 0 {
        for logical in text[..line_start - 1].split('\n') {
            row += review_wrapped_lines(logical, width).len();
        }
    }
    let prefix_chars = text[line_start..cursor].chars().count();
    (row + prefix_chars / width, prefix_chars % width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::test_fixtures::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn render_to_buffer(args: ReviewScreenArgs<'_>) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(
            args.actual_width as u16,
            args.actual_height as u16,
        ))
        .unwrap();
        terminal
            .draw(|f| crate::tui::review_native::draw_review_screen(f, args))
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn buffer_to_string(buffer: &ratatui::buffer::Buffer) -> String {
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            out.push_str("\r\n");
        }
        out
    }

    #[test]
    fn multiline_cursor_position_counts_logical_lines_and_wraps() {
        use super::{multiline_cursor_position, wrapped_multiline_lines};

        assert_eq!(multiline_cursor_position("ab\ncd", 4, 10), (1, 1));
        assert_eq!(wrapped_multiline_lines("abc\ncd", 2).len(), 4);
        assert_eq!(multiline_cursor_position("abc\ncd", 5, 2), (2, 1));
        assert_eq!(wrapped_multiline_lines("a\n\nb", 10).len(), 3);
        assert_eq!(multiline_cursor_position("a\n\nb", 4, 10), (2, 1));
    }

    #[test]
    fn review_right_width_fits_longest_full_path() {
        let files = vec![
            review_entry("README.md", ReviewStatus::Unreviewed, &[]),
            review_entry("src/bin/orangu/main.rs", ReviewStatus::Approved, &[]),
        ];
        assert_eq!(
            review_right_width(&files, 200),
            4 + "src/bin/orangu/main.rs".len()
        );
    }

    #[test]
    fn review_right_width_is_capped_on_narrow_terminals() {
        let files = vec![review_entry(
            "a/very/long/path/that/exceeds/the/terminal.rs",
            ReviewStatus::Unreviewed,
            &[],
        )];
        assert_eq!(review_right_width(&files, 20), 18);
    }
}

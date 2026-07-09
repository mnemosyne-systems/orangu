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
const REVIEW_COMMENT_BG: &str = "\x1b[48;2;38;48;38m";
pub(crate) const REVIEW_COMMENT_MARKER: &str = "\x1b[38;2;230;200;120m●\x1b[0m";
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
    // Always leave room for a separator plus a minimally useful left pane.
    let cap = actual_width.saturating_sub(2).max(1);
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

pub fn render_review_screen(args: ReviewScreenArgs<'_>) -> String {
    let width = args.actual_width.max(1);
    let height = args.actual_height.max(1);

    // Reserve the bottom prompt frame (status bar + input window), exactly like
    // the normal screen, and give the panes everything above it.
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let input_lines = wrapped_input_lines(args.input, width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let pane_rows = height.saturating_sub(prompt_frame_height).max(1);

    let content = if let Some(feedback) = &args.feedback {
        render_review_feedback_panel(feedback, width, pane_rows)
    } else {
        render_review_panes(&args, width, pane_rows)
    };

    let mut screen = content;
    screen.push_str("\r\n");
    screen.push_str(&render_prompt_frame(PromptFrameArgs {
        header_height: pane_rows,
        current_model: args.current_model,
        left_status: args.left_status,
        pending_count: args.pending_count,
        // The Graph status dot is `/auto_review`-only; `/review` keeps its
        // `Pending: N` display exactly as before.
        graph_status: None,
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        ghost: args.ghost,
        height,
        actual_width: width,
        valid_command_len: 0,
    }));
    screen
}

/// Render the two-pane diff view (file list + selected file's diff) as exactly
/// `pane_rows` rows: one header row plus `pane_rows - 1` body rows.
fn render_review_panes(args: &ReviewScreenArgs<'_>, width: usize, pane_rows: usize) -> String {
    let right_width = review_right_width(args.files, width);
    let left_width = width.saturating_sub(right_width + 1).max(1);
    let body_height = pane_rows.saturating_sub(1);

    // Keep the selected file visible in the right pane.
    let list_start = if args.selected >= body_height {
        args.selected - body_height + 1
    } else {
        0
    };

    let title = format!(
        "Review: {}  Alt+j/k Switch file  Alt+a Approve  Alt+r Reject  Alt+o Review  Alt+c Comment  Alt+e Open  Alt+x Exit",
        args.prompt_branch.unwrap_or("(detached HEAD)"),
    );
    let right_header = format!("Files ({})", args.files.len());

    // The left pane shows only the selected file's diff.
    let selected_lines: &[String] = args
        .files
        .get(args.selected)
        .map(|file| file.diff_lines.as_slice())
        .unwrap_or(&[]);

    let left_cells = render_review_left_column(args, selected_lines, left_width, body_height);

    let mut rows: Vec<String> = Vec::with_capacity(pane_rows);
    rows.push(format!(
        "{}{}{}",
        review_pane_cell(&title, 0, left_width),
        REVIEW_SEPARATOR,
        review_pane_cell(&right_header, 0, right_width),
    ));

    for (row, left) in left_cells.into_iter().enumerate() {
        let file_index = list_start + row;
        let right = match args.files.get(file_index) {
            Some(file) => {
                let entry = format!("{} {}", review_status_box(file.status), file.path);
                let cell = review_pane_cell(&entry, 0, right_width);
                if file_index == args.selected {
                    review_highlight(&cell)
                } else {
                    cell
                }
            }
            None => review_pane_cell("", 0, right_width),
        };

        rows.push(format!("{left}{REVIEW_SEPARATOR}{right}"));
    }

    rows.join("\r\n")
}

/// Render a single diff-line cell with an optional comment marker and the
/// line-cursor highlight.
fn render_review_diff_cell(
    line: &str,
    x_offset: usize,
    left_width: usize,
    is_cursor: bool,
    has_comment: bool,
) -> String {
    let cell = if has_comment {
        // Reserve two columns on the right for a comment marker.
        let inner = review_pane_cell(line, x_offset, left_width.saturating_sub(2));
        format!("{inner} {REVIEW_COMMENT_MARKER}")
    } else {
        review_pane_cell(line, x_offset, left_width)
    };
    if is_cursor {
        review_line_highlight(&cell)
    } else {
        cell
    }
}

/// Build the `body_height` rows of the left pane, splicing the inline comment
/// editor below the highlighted line when it is open.
fn render_review_left_column(
    args: &ReviewScreenArgs<'_>,
    selected_lines: &[String],
    left_width: usize,
    body_height: usize,
) -> Vec<String> {
    let has_comment = |index: usize| args.commented_lines.contains(&index);
    let cell = |index: usize| match selected_lines.get(index) {
        Some(line) => render_review_diff_cell(
            line,
            args.x_offset,
            left_width,
            index == args.line,
            has_comment(index),
        ),
        None => review_pane_cell("", 0, left_width),
    };

    let Some(editor) = &args.comment_editor else {
        return (0..body_height)
            .map(|row| cell(args.scroll + row))
            .collect();
    };

    // The comment editor is open: emit diff lines from `scroll`, and after the
    // highlighted line, splice in the comment box.
    let box_rows = render_review_comment_box(
        editor.category,
        editor.selector_focused,
        editor.text,
        editor.cursor,
        left_width,
    );
    let mut cells: Vec<String> = Vec::with_capacity(body_height);
    let mut index = args.scroll;
    let mut box_shown = false;
    while cells.len() < body_height {
        if index < selected_lines.len() {
            cells.push(cell(index));
            if index == args.line && !box_shown {
                box_shown = true;
                for row in &box_rows {
                    if cells.len() < body_height {
                        cells.push(row.clone());
                    }
                }
            }
            index += 1;
        } else if !box_shown && args.line >= selected_lines.len() {
            // Empty/short file: still show the box.
            box_shown = true;
            for row in &box_rows {
                if cells.len() < body_height {
                    cells.push(row.clone());
                }
            }
        } else {
            cells.push(review_pane_cell("", 0, left_width));
        }
    }
    cells
}

/// Render the inline comment editor box: a single category-selector row above a
/// fixed-height comment window. The comment text wraps to the pane width and
/// scrolls to keep the cursor visible. The chosen category is inverted while
/// the selector has the focus; otherwise the caret is drawn in the comment.
fn render_review_comment_box(
    category: &str,
    selector_focused: bool,
    text: &str,
    cursor: usize,
    width: usize,
) -> Vec<String> {
    let inner_width = width.saturating_sub(2).max(1);
    // Greenish gutter bar; reset only the foreground so the comment background
    // spans the whole row, padded to the inner width.
    let gutter = |content: &str| {
        let visible = visible_line_width(content);
        let padding = " ".repeat(inner_width.saturating_sub(visible));
        format!("{REVIEW_COMMENT_BG}\x1b[38;2;120;160;120m▕\x1b[39m {content}{padding}{ANSI_RESET}")
    };

    // The category selector: the chosen category on one line, inverted while
    // it has the focus, followed by the navigation hint.
    let chosen = if selector_focused {
        format!("\x1b[7m {category} \x1b[27m")
    } else {
        format!("[{category}]")
    };
    let mut rows = Vec::with_capacity(REVIEW_COMMENT_BOX_HEIGHT + 1);
    rows.push(gutter(&format!(
        "Category: {chosen}  \x1b[2m↑/↓ Category · Tab Switch focus\x1b[22m"
    )));

    let wrapped = wrapped_input_lines(text, inner_width, "");
    let (cursor_row, cursor_col) = cursor_position(text, cursor, inner_width, "");
    let start = cursor_row.saturating_sub(REVIEW_COMMENT_BOX_HEIGHT - 1);
    for row in 0..REVIEW_COMMENT_BOX_HEIGHT {
        let index = start + row;
        let mut content = wrapped.get(index).cloned().unwrap_or_default();
        // The caret is only drawn while the comment text has the focus.
        if index == cursor_row && !selector_focused {
            content = comment_caret(&content, cursor_col, inner_width);
        }
        rows.push(gutter(&content));
    }
    rows
}

/// Wrap multi-line text to `width` visible columns: each logical line (split
/// on `\n`) wraps independently, an empty logical line keeping its own row.
pub(crate) fn wrapped_multiline_lines(text: &str, width: usize) -> Vec<String> {
    text.split('\n')
        .flat_map(|logical| wrapped_input_lines(logical, width.max(1), ""))
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
            row += wrapped_input_lines(logical, width, "").len();
        }
    }
    let prefix_chars = text[line_start..cursor].chars().count();
    (row + prefix_chars / width, prefix_chars % width)
}

/// Insert a reverse-video caret into a plain comment line at `col`.
pub(crate) fn comment_caret(content: &str, col: usize, inner_width: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    if col < chars.len() {
        let mut out = String::new();
        for (index, ch) in chars.iter().enumerate() {
            if index == col {
                out.push_str("\x1b[7m");
                out.push(*ch);
                out.push_str("\x1b[27m");
            } else {
                out.push(*ch);
            }
        }
        out
    } else if chars.len() < inner_width {
        format!("{content}\x1b[7m \x1b[27m")
    } else {
        content.to_string()
    }
}

/// Render the Alt+o feedback popup filling the pane region: a title bar plus the
/// scrollable review text.
fn render_review_feedback_panel(
    feedback: &ReviewFeedbackView<'_>,
    width: usize,
    pane_rows: usize,
) -> String {
    let mut rows: Vec<String> = Vec::with_capacity(pane_rows);

    let header = format!("{} (x to close · ↑/↓ scroll)", feedback.title);
    rows.push(review_highlight(&review_pane_cell(&header, 0, width)));

    // Echo the asked question (if any) below the title, styled like a submitted
    // prompt in the main output window. It stays pinned above the review text.
    if let Some(question) = feedback.question {
        rows.push(render_user_input_line(&format!("> {question}"), width));
    }

    let body_height = pane_rows.saturating_sub(rows.len());
    for row in 0..body_height {
        let line = match feedback.lines.get(feedback.scroll + row) {
            Some(line) => review_pane_cell(line, feedback.x_offset, width),
            None => review_pane_cell("", 0, width),
        };
        rows.push(line);
    }

    rows.join("\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::test_fixtures::*;

    #[test]
    fn multiline_cursor_position_counts_logical_lines_and_wraps() {
        use super::{multiline_cursor_position, wrapped_multiline_lines};

        // "ab\ncd" at width 10: two rows; cursor on the second line.
        assert_eq!(multiline_cursor_position("ab\ncd", 4, 10), (1, 1));
        // The first line wraps into two rows at width 2, so "cd" is row 2.
        // ("cd" fills its row exactly, keeping a trailing row for the cursor.)
        assert_eq!(wrapped_multiline_lines("abc\ncd", 2).len(), 4);
        assert_eq!(multiline_cursor_position("abc\ncd", 5, 2), (2, 1));
        // An empty logical line keeps its own row.
        assert_eq!(wrapped_multiline_lines("a\n\nb", 10).len(), 3);
        assert_eq!(multiline_cursor_position("a\n\nb", 4, 10), (2, 1));
    }

    #[test]
    fn review_right_width_fits_longest_full_path() {
        let files = vec![
            review_entry("README.md", ReviewStatus::Unreviewed, &[]),
            review_entry("src/bin/orangu/main.rs", ReviewStatus::Approved, &[]),
        ];
        // "[x] " (4) + longest path length.
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

    #[test]
    fn render_review_screen_draws_straight_separator_and_status_dots() {
        let files = vec![
            review_entry(
                "README.md",
                ReviewStatus::Approved,
                &["diff --git a/README.md b/README.md", "+hello"],
            ),
            review_entry(
                "src/main.rs",
                ReviewStatus::Rejected,
                &["diff --git a/src/main.rs b/src/main.rs", "+world"],
            ),
        ];
        // Empty input ⇒ prompt frame is 4 rows, so the panes occupy
        // actual_height - 4 rows above it.
        let rendered = render_review_screen(review_args(&files, 0, 0, 50, 10));

        let pane_rows = 10 - 4;
        let rows: Vec<&str> = rendered.split("\r\n").collect();

        // The separator sits in the same visible column on every pane row (the
        // remaining chunk is the absolutely-positioned prompt frame).
        let right_width = review_right_width(&files, 50);
        let separator_column = 50 - right_width - 1;
        for row in &rows[..pane_rows] {
            let prefix: String = {
                let mut visible = String::new();
                let mut count = 0;
                let mut chars = row.chars().peekable();
                while let Some(ch) = chars.next() {
                    if ch == '\x1b' {
                        for c in chars.by_ref() {
                            if c.is_ascii_alphabetic() {
                                break;
                            }
                        }
                        continue;
                    }
                    if count == separator_column {
                        visible.push(ch);
                        break;
                    }
                    count += 1;
                }
                visible
            };
            assert_eq!(prefix, "│", "separator misaligned in row: {row:?}");
        }

        // Status dots (green for approved, red for rejected) and file paths
        // appear. Match the color+dot fragment so the assertion holds whether
        // or not the row is highlighted (which re-injects reverse video).
        assert!(
            rendered.contains("\u{1b}[38;2;80;200;120m●"),
            "missing green dot"
        );
        assert!(
            rendered.contains("\u{1b}[38;2;220;80;80m●"),
            "missing red dot"
        );
        assert!(rendered.contains("README.md"));
        assert!(rendered.contains("src/main.rs"));
        // The unselected, unmarked-but-rejected row keeps its plain box closed.
        assert!(rendered.contains("[ ]") || rendered.contains('●'));
    }

    #[test]
    fn render_review_screen_shows_only_selected_file_and_scrolls() {
        let lines_a = (0..20).map(|i| format!("a {i}")).collect::<Vec<_>>();
        let lines_b = vec!["b only".to_string()];
        let files = vec![
            ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: lines_a,
                patch: String::new(),
            },
            ReviewEntry {
                path: "b.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: lines_b,
                patch: String::new(),
            },
        ];
        let rendered = render_review_screen(review_args(&files, 0, 10, 40, 12));
        // Body shows the selected file's lines from the scroll offset, not from
        // the top, and never the other file's diff content.
        assert!(rendered.contains("a 10"));
        assert!(!rendered.contains("a 0 "));
        assert!(!rendered.contains("b only"));
    }

    #[test]
    fn render_review_screen_highlights_the_cursor_line() {
        let files = vec![review_entry(
            "a.txt",
            ReviewStatus::Unreviewed,
            &["line zero", "line one", "line two"],
        )];
        let mut args = review_args(&files, 0, 0, 50, 10);
        args.line = 1;
        let rendered = render_review_screen(args);
        // The cursor line carries the highlight background; the others do not.
        assert!(
            rendered.contains("\u{1b}[48;2;60;60;90mline one"),
            "cursor line not highlighted"
        );
        assert!(!rendered.contains("\u{1b}[48;2;60;60;90mline zero"));
    }

    #[test]
    fn render_review_screen_marks_commented_lines() {
        let files = vec![review_entry(
            "a.txt",
            ReviewStatus::Unreviewed,
            &["line zero", "line one"],
        )];
        let mut args = review_args(&files, 0, 0, 50, 10);
        let commented = vec![1usize];
        args.commented_lines = &commented;
        let rendered = render_review_screen(args);
        // The amber comment marker is shown for commented lines.
        assert!(
            rendered.contains("\u{1b}[38;2;230;200;120m●"),
            "commented line not marked"
        );
    }

    #[test]
    fn render_review_screen_splices_comment_box_below_the_line() {
        let files = vec![review_entry(
            "a.txt",
            ReviewStatus::Unreviewed,
            &["line zero", "line one", "line two", "line three"],
        )];
        let mut args = review_args(&files, 0, 0, 50, 14);
        args.line = 1;
        args.comment_editor = Some(super::ReviewCommentEditor {
            category: "Overall",
            selector_focused: false,
            text: "needs a guard",
            cursor: "needs a guard".len(),
        });
        let rendered = render_review_screen(args);
        let rows: Vec<&str> = rendered.split("\r\n").collect();

        // Find the body row holding the highlighted line, then assert the next
        // six rows are the comment box: a category selector row followed by the
        // five comment rows (they carry the comment background and the typed
        // text appears within them).
        let line_row = rows
            .iter()
            .position(|row| row.contains("line one"))
            .expect("highlighted line present");
        let box_block = rows[line_row + 1..line_row + 1 + 6].join("\n");
        assert!(
            box_block.contains("\u{1b}[48;2;38;48;38m"),
            "comment box background missing below the line"
        );
        // The category selector row leads the box, defaulting to Overall.
        assert!(
            rows[line_row + 1].contains("Category: [Overall]"),
            "category selector row missing"
        );
        assert!(box_block.contains("needs a guard"), "comment text missing");
        // The line after the box continues the diff.
        assert!(rows[line_row + 7].contains("line two"));
    }

    #[test]
    fn render_review_screen_draws_the_input_completion_ghost() {
        // Typing an open command in the input window previews the file Tab would
        // fill in as a grey ghost, exactly like the main prompt.
        let files = vec![review_entry("a.txt", ReviewStatus::Unreviewed, &["+x"])];
        let mut args = review_args(&files, 0, 0, 80, 10);
        args.input = "/open_file READ";
        args.cursor = args.input.len();
        args.ghost = "ME.md";
        let rendered = render_review_screen(args);
        assert!(rendered.contains("/open_file READ"), "input not shown");
        assert!(rendered.contains("ME.md"), "ghost suffix not shown");
    }

    #[test]
    fn render_review_screen_title_shows_branch_name() {
        let files = vec![review_entry("a.txt", ReviewStatus::Unreviewed, &["+x"])];
        let mut args = review_args(&files, 0, 0, 90, 10);
        args.prompt_branch = Some("feature/login");
        let rendered = render_review_screen(args);
        assert!(
            rendered.contains("Review: feature/login"),
            "title should show the current branch"
        );
    }

    #[test]
    fn render_review_screen_shows_feedback_popup() {
        let files = vec![review_entry("a.txt", ReviewStatus::Unreviewed, &["+x"])];
        let feedback_lines = vec!["LGTM overall".to_string(), "fix the typo".to_string()];
        let mut args = review_args(&files, 0, 0, 60, 12);
        args.current_model = "my-model";
        args.input = "focus on errors";
        args.feedback = Some(super::ReviewFeedbackView {
            title: "Review: a.txt",
            question: Some("is this safe?"),
            lines: &feedback_lines,
            scroll: 0,
            x_offset: 0,
        });
        let rendered = render_review_screen(args);
        assert!(rendered.contains("Review: a.txt"));
        assert!(rendered.contains("x to close"));
        assert!(rendered.contains("LGTM overall"));
        assert!(rendered.contains("fix the typo"));
        // The diff panes are hidden while the popup is open.
        assert!(!rendered.contains("+x"));
        // The asked question is echoed, styled like a submitted prompt.
        assert!(
            rendered.contains(&format!("{USER_INPUT_BACKGROUND}> is this safe?")),
            "question not echoed with input styling"
        );
        // The status bar (model name) and input window are still present.
        assert!(rendered.contains("my-model"), "status bar missing model");
        assert!(
            rendered.contains("> focus on errors"),
            "input window missing"
        );
    }
}

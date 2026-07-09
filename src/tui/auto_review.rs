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

pub fn render_auto_review_screen(args: AutoReviewScreenArgs<'_>) -> String {
    let width = args.actual_width.max(1);
    let height = args.actual_height.max(1);

    // Reserve the bottom prompt frame exactly like `/review`. The input window
    // stays empty during the run and carries the browse-phase `/open_file`
    // line afterwards.
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let input_lines = wrapped_input_lines(args.input, width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let pane_rows = height.saturating_sub(prompt_frame_height).max(2);

    let content = if let Some(diff) = &args.diff {
        render_auto_review_diff_panel(diff, width, pane_rows)
    } else if let Some(reject) = &args.reject {
        render_auto_review_reject_panel(reject, width, pane_rows)
    } else {
        render_auto_review_panes(&args, width, pane_rows)
    };
    let mut screen = content.join("\r\n");
    screen.push_str("\r\n");
    screen.push_str(&render_prompt_frame(PromptFrameArgs {
        header_height: pane_rows,
        current_model: args.current_model,
        left_status: args.left_status,
        pending_count: args.pending_count,
        graph_status: args.graph_status,
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

/// Render the two auto review panes (report + file checklist) as exactly
/// `pane_rows` rows: one header row plus `pane_rows - 1` body rows. The first
/// body row of the left pane is the status area — it stays inside the left
/// pane, so the file checklist keeps every body row on the right.
fn render_auto_review_panes(
    args: &AutoReviewScreenArgs<'_>,
    width: usize,
    pane_rows: usize,
) -> Vec<String> {
    let right_width = review_right_width(args.files, width);
    let left_width = width.saturating_sub(right_width + 1).max(1);
    let body_height = pane_rows.saturating_sub(1);

    // Keep the highlighted file (if any) visible in the right pane.
    let anchor = args.selected.unwrap_or(0);
    let list_start = if anchor >= body_height {
        anchor - body_height + 1
    } else {
        0
    };

    // The header keys depend on the phase: before the run starts it offers the
    // pre-start keys (start, file switching, and Ignore mode); while the run is
    // in progress only the run keys; once the report is browsed the per-file
    // browse keys.
    let keys = if args.prestart {
        "Alt+s Start  Alt+j/k Switch file  Alt+m Mode  Alt+e Diff  Esc Esc Cancel  Alt+x Exit"
    } else if args.browsing {
        "Alt+j/k Switch file  Alt+a Approve  Alt+r Reject  Alt+e Open  ↑/↓ Item  Enter Diff  PgUp/PgDn Category  - Remove  Alt+x Exit"
    } else {
        "Esc Esc Cancel  Alt+x Exit"
    };
    let title = format!(
        "Auto review: {}  {keys}",
        args.prompt_branch.unwrap_or("(detached HEAD)"),
    );
    let right_header = format!("Files ({})", args.files.len());

    let mut rows: Vec<String> = Vec::with_capacity(pane_rows);
    rows.push(format!(
        "{}{}{}",
        review_pane_cell(&title, 0, left_width),
        REVIEW_SEPARATOR,
        review_pane_cell(&right_header, 0, right_width),
    ));

    for row in 0..body_height {
        let left = if row == 0 {
            // The status area: a highlighted bar across the left pane showing
            // which file and category is being worked on.
            review_highlight(&review_pane_cell(args.status, 0, left_width))
        } else {
            let line_index = args.scroll + row - 1;
            match args.report_lines.get(line_index) {
                Some(line) => {
                    let cell = review_pane_cell(line, args.x_offset, left_width);
                    // Lines of the item under the Up/Down cursor get the
                    // line-cursor background, like the `/review` line cursor.
                    match args.selected_lines {
                        Some((start, end)) if line_index >= start && line_index < end => {
                            review_line_highlight(&cell)
                        }
                        _ => cell,
                    }
                }
                None => review_pane_cell("", 0, left_width),
            }
        };
        let file_index = list_start + row;
        let right = match args.files.get(file_index) {
            Some(file) => {
                // Before the run starts every file shows its mode as a dot —
                // white for Normal, blue for Ignore (skipped), purple for Deep
                // (reviewed with extra passes) — so the three modes read the
                // same way at a glance. Once the run starts the dot follows
                // the file's real status instead (Ignore is approved up
                // front, Deep and Normal are reviewed alike). The file under
                // review blinks a white dot meanwhile.
                let mode = args.modes.get(file_index).copied().unwrap_or_default();
                let status_box = if args.prestart && mode == AutoReviewFileMode::Ignore {
                    format!("[{STATUS_BLUE}●{ANSI_RESET}]")
                } else if args.prestart && mode == AutoReviewFileMode::Deep {
                    format!("[{STATUS_PURPLE}●{ANSI_RESET}]")
                } else if (args.prestart && mode == AutoReviewFileMode::Normal)
                    || args.reviewing == Some(file_index)
                {
                    format!("[{STATUS_WHITE}●{ANSI_RESET}]")
                } else {
                    review_status_box(file.status)
                };
                let entry = format!("{status_box} {}", file.path);
                let cell = review_pane_cell(&entry, 0, right_width);
                if args.selected == Some(file_index) {
                    review_highlight(&cell)
                } else {
                    cell
                }
            }
            None => review_pane_cell("", 0, right_width),
        };
        rows.push(format!("{left}{REVIEW_SEPARATOR}{right}"));
    }

    rows
}

/// Push a reject-window section label — inverted while its section has the
/// focus — and the grey underline fitted to the label's width.
fn push_reject_section_label(rows: &mut Vec<String>, label: &str, focused: bool, width: usize) {
    let text = if focused {
        review_highlight(label)
    } else {
        label.to_string()
    };
    rows.push(review_pane_cell(&text, 0, width));
    let underline = format!(
        "\x1b[38;2;88;88;88m{}{ANSI_RESET}",
        "─".repeat(label.chars().count())
    );
    rows.push(review_pane_cell(&underline, 0, width));
}

/// Render the `/auto_review` reject window filling the pane region: a title
/// bar, the `Category:` selector, and the `Comment:` editor (multi-line
/// Markdown) taking the remaining rows. Each section label sits over a grey
/// underline of its own width and is inverted while its section has the
/// focus; the chosen category carries a dot in its box (and the selection
/// highlight while the selector has the focus), and the editor draws its
/// caret only while it has the focus.
fn render_auto_review_reject_panel(
    reject: &AutoReviewRejectView<'_>,
    width: usize,
    pane_rows: usize,
) -> Vec<String> {
    let mut rows: Vec<String> = Vec::with_capacity(pane_rows);

    let header = format!(
        "Reject: {}  (Tab Switch focus · ↑/↓ Category · Alt+Enter Save · Esc Cancel)",
        reject.path
    );
    rows.push(review_highlight(&review_pane_cell(&header, 0, width)));
    rows.push(review_pane_cell("", 0, width));

    push_reject_section_label(&mut rows, "Category:", reject.selector_focused, width);
    for (index, name) in reject.categories.iter().enumerate() {
        let chosen = index == reject.category;
        let marker = if chosen {
            format!("[{REVIEW_COMMENT_MARKER}]")
        } else {
            "[ ]".to_string()
        };
        let cell = review_pane_cell(&format!("{marker} {name}"), 0, width);
        if chosen && reject.selector_focused {
            rows.push(review_highlight(&cell));
        } else {
            rows.push(cell);
        }
    }

    rows.push(review_pane_cell("", 0, width));
    push_reject_section_label(&mut rows, "Comment:", !reject.selector_focused, width);

    // The editor takes every remaining row, scrolled to keep the caret
    // visible.
    let editor_rows = pane_rows.saturating_sub(rows.len()).max(1);
    let inner_width = width.max(1);
    let wrapped = wrapped_multiline_lines(reject.text, inner_width);
    let (cursor_row, cursor_col) =
        multiline_cursor_position(reject.text, reject.cursor, inner_width);
    let start = cursor_row.saturating_sub(editor_rows - 1);
    for row in 0..editor_rows {
        let index = start + row;
        let mut content = wrapped.get(index).cloned().unwrap_or_default();
        if index == cursor_row && !reject.selector_focused {
            content = comment_caret(&content, cursor_col, inner_width);
        }
        rows.push(review_pane_cell(&content, 0, width));
    }

    rows.truncate(pane_rows);
    rows
}

/// Render the `/auto_review` diff popup filling the pane region: a title bar,
/// then the colorized diff scrolled to `scroll` and panned to `x_offset`. The
/// panes are hidden while it is open.
fn render_auto_review_diff_panel(
    diff: &AutoReviewDiffView<'_>,
    width: usize,
    pane_rows: usize,
) -> Vec<String> {
    let mut rows: Vec<String> = Vec::with_capacity(pane_rows);
    let header = format!("{}  (↑/↓ Scroll · Esc Close)", diff.title);
    rows.push(review_highlight(&review_pane_cell(&header, 0, width)));

    let body_height = pane_rows.saturating_sub(1);
    for row in 0..body_height {
        let cell = match diff.lines.get(diff.scroll + row) {
            Some(line) => review_pane_cell(line, diff.x_offset, width),
            None => review_pane_cell("", 0, width),
        };
        rows.push(cell);
    }
    rows.truncate(pane_rows);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::test_fixtures::*;

    #[test]
    fn auto_review_screen_places_the_status_area_below_the_header() {
        // The path is longer than the `Files (1)` header, so the right pane is
        // wide enough to show the header unclipped.
        let files = vec![review_entry("src/main.rs", ReviewStatus::Approved, &[])];
        let report: Vec<String> = vec!["Overall".to_string(), "  - ready".to_string()];
        let screen = render_auto_review_screen(auto_review_args(&files, &report, 80, 12));
        let rows: Vec<&str> = screen.split("\r\n").collect();

        // The header row comes first (tool title + file-list header); the
        // status area sits below it, inside the left pane, with the file
        // checklist continuing alongside on the right.
        assert!(rows[0].contains("Auto review: feature/x"), "{:?}", rows[0]);
        assert!(rows[0].contains("Files (1)"), "{:?}", rows[0]);
        assert!(rows[1].contains("Category: Code"), "{:?}", rows[1]);
        assert!(rows[1].contains("Time: 5s"), "{:?}", rows[1]);
        assert!(rows[1].contains("src/main.rs"), "{:?}", rows[1]);

        // The report fills the left pane below the status area.
        assert!(rows[2].contains("Overall"), "{:?}", rows[2]);
        assert!(!rows[2].contains("src/main.rs"), "{:?}", rows[2]);
        assert!(rows[3].contains("  - ready"), "{:?}", rows[3]);
    }

    #[test]
    fn auto_review_reviewing_file_shows_a_white_dot() {
        // Match the box opening and the colored dot; the trailing reset is
        // rewritten when the row carries the selection highlight.
        let white_dot = format!("[{}●", super::STATUS_WHITE);
        let files = vec![review_entry("src/main.rs", ReviewStatus::Unreviewed, &[])];
        let report: Vec<String> = Vec::new();

        // The blink-off phase (or no file under review) keeps the empty box.
        let screen = render_auto_review_screen(auto_review_args(&files, &report, 80, 12));
        assert!(screen.contains("[ ] src/main.rs"), "{screen:?}");
        assert!(!screen.contains(&white_dot), "{screen:?}");

        // The blink-on phase paints the white "being reviewed" dot.
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.reviewing = Some(0);
        let screen = render_auto_review_screen(args);
        assert!(screen.contains(&white_dot), "{screen:?}");
    }

    #[test]
    fn auto_review_header_offers_browse_keys_once_the_run_ends() {
        let files = vec![review_entry("src/main.rs", ReviewStatus::Approved, &[])];
        let report: Vec<String> = Vec::new();

        // During the run only the run keys are offered.
        let screen = render_auto_review_screen(auto_review_args(&files, &report, 120, 12));
        assert!(screen.contains("Esc Esc Cancel"), "{screen:?}");
        assert!(!screen.contains("Alt+a Approve"), "{screen:?}");

        // Browsing swaps in the per-file keys.
        let mut args = auto_review_args(&files, &report, 120, 12);
        args.browsing = true;
        let screen = render_auto_review_screen(args);
        assert!(screen.contains("Alt+j/k Switch file"), "{screen:?}");
        assert!(screen.contains("Alt+a Approve"), "{screen:?}");
        assert!(screen.contains("Alt+r Reject"), "{screen:?}");
        assert!(screen.contains("Alt+e Open"), "{screen:?}");
    }

    #[test]
    fn auto_review_prestart_header_offers_start_and_mode_keys() {
        let files = vec![review_entry("src/main.rs", ReviewStatus::Unreviewed, &[])];
        let report: Vec<String> = Vec::new();

        // The pre-start header offers Alt+s Start, file switching, and Alt+m
        // Mode, and none of the run/browse-only keys.
        let mut args = auto_review_args(&files, &report, 120, 12);
        args.prestart = true;
        let screen = render_auto_review_screen(args);
        assert!(screen.contains("Alt+s Start"), "{screen:?}");
        assert!(screen.contains("Alt+j/k Switch file"), "{screen:?}");
        assert!(screen.contains("Alt+m Mode"), "{screen:?}");
        assert!(screen.contains("Alt+e Diff"), "{screen:?}");
        assert!(!screen.contains("Alt+a Approve"), "{screen:?}");

        // Once the run starts those keys are gone, leaving only the run keys.
        let screen = render_auto_review_screen(auto_review_args(&files, &report, 120, 12));
        assert!(!screen.contains("Alt+s Start"), "{screen:?}");
        assert!(!screen.contains("Alt+m Mode"), "{screen:?}");
        assert!(screen.contains("Esc Esc Cancel"), "{screen:?}");
    }

    #[test]
    fn auto_review_ignored_file_shows_a_blue_dot_until_the_run_starts() {
        let blue_dot = format!("[{}●", super::STATUS_BLUE);
        let white_dot = format!("[{}●", super::STATUS_WHITE);
        let green_dot = format!("[{}●", super::STATUS_GREEN);
        let modes = [AutoReviewFileMode::Normal, AutoReviewFileMode::Ignore];
        let report: Vec<String> = Vec::new();

        // Pre-start: the ignored file (b.rs) carries the blue dot; the normal
        // one (a.rs) carries the white dot.
        let files = vec![
            review_entry("a.rs", ReviewStatus::Unreviewed, &[]),
            review_entry("b.rs", ReviewStatus::Unreviewed, &[]),
        ];
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.prestart = true;
        args.modes = &modes;
        let screen = render_auto_review_screen(args);
        assert!(screen.contains(&blue_dot), "{screen:?}");
        assert!(screen.contains(&white_dot), "{screen:?}");
        assert!(screen.contains("a.rs") && screen.contains("b.rs"), "{screen:?}");

        // Once the run starts the ignored file is approved: no blue dot any
        // more, just the green status dot.
        let files = vec![
            review_entry("a.rs", ReviewStatus::Approved, &[]),
            review_entry("b.rs", ReviewStatus::Approved, &[]),
        ];
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.prestart = false;
        args.modes = &modes;
        let screen = render_auto_review_screen(args);
        assert!(!screen.contains(&blue_dot), "{screen:?}");
        assert!(screen.contains(&green_dot), "{screen:?}");
    }

    #[test]
    fn auto_review_deep_file_shows_a_purple_dot_only_before_the_run_starts() {
        let purple_dot = format!("[{}●", super::STATUS_PURPLE);
        let modes = [AutoReviewFileMode::Deep];
        let report: Vec<String> = Vec::new();

        // Pre-start: the Deep file carries the purple dot.
        let files = vec![review_entry("a.rs", ReviewStatus::Unreviewed, &[])];
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.prestart = true;
        args.modes = &modes;
        let screen = render_auto_review_screen(args);
        assert!(screen.contains(&purple_dot), "{screen:?}");

        // Once reviewed, Deep is reviewed like Normal: its real status dot
        // shows, not purple.
        let files = vec![review_entry("a.rs", ReviewStatus::Approved, &[])];
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.prestart = false;
        args.modes = &modes;
        let screen = render_auto_review_screen(args);
        assert!(!screen.contains(&purple_dot), "{screen:?}");
    }

    #[test]
    fn auto_review_browse_shows_the_open_input_and_ghost() {
        // After the run, the input window accepts `/open_file <path>` to open any
        // project file, with the same grey completion ghost as the main prompt.
        let files = vec![review_entry("src/main.rs", ReviewStatus::Approved, &[])];
        let report: Vec<String> = Vec::new();
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.browsing = true;
        args.input = "/open_file READ";
        args.cursor = args.input.len();
        args.ghost = "ME.md";
        let screen = render_auto_review_screen(args);
        assert!(screen.contains("/open_file READ"), "{screen:?}");
        assert!(screen.contains("ME.md"), "{screen:?}");
    }

    #[test]
    fn auto_review_reject_window_covers_the_panes() {
        let files = vec![review_entry("src/main.rs", ReviewStatus::Rejected, &[])];
        let report: Vec<String> = vec!["Overall".to_string()];
        let categories = ["Overall", "Code", "Security"];
        let mut args = auto_review_args(&files, &report, 80, 16);
        args.browsing = true;
        args.reject = Some(super::AutoReviewRejectView {
            path: "src/main.rs",
            categories: &categories,
            category: 1,
            selector_focused: true,
            text: "first line\nsecond line",
            cursor: 0,
        });
        let screen = render_auto_review_screen(args);

        // Title bar, the section labels, the category selector with the
        // chosen category marked, and the editor showing both logical lines.
        assert!(screen.contains("Reject: src/main.rs"), "{screen:?}");
        assert!(screen.contains("Category:"), "{screen:?}");
        assert!(screen.contains("[ ] Overall"), "{screen:?}");
        assert!(
            screen.contains("\u{1b}[38;2;230;200;120m●"),
            "chosen category not marked: {screen:?}"
        );
        assert!(screen.contains("Comment:"), "{screen:?}");
        assert!(screen.contains("first line"), "{screen:?}");
        assert!(screen.contains("second line"), "{screen:?}");
        // Each label sits over a grey underline of exactly its own width.
        let underline = |label: &str| {
            format!(
                "\u{1b}[38;2;88;88;88m{}{ANSI_RESET}",
                "─".repeat(label.chars().count())
            )
        };
        assert!(screen.contains(&underline("Category:")), "{screen:?}");
        assert!(screen.contains(&underline("Comment:")), "{screen:?}");
        // The editor keeps the window's default background — no comment-box
        // green and no gutter bar in front of the comment part.
        assert!(!screen.contains("\u{1b}[48;2;38;48;38m"), "{screen:?}");
        assert!(!screen.contains('▕'), "{screen:?}");
        // The panes are hidden while the window is open.
        assert!(!screen.contains("Files (1)"), "{screen:?}");
    }

    #[test]
    fn auto_review_diff_view_covers_the_panes() {
        let files = vec![review_entry("src/main.rs", ReviewStatus::Rejected, &[])];
        let report: Vec<String> = vec!["Overall".to_string()];
        let lines = vec!["@@ -1,2 +1,3 @@".to_string(), "+let x = 1;".to_string()];
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.browsing = true;
        args.diff = Some(super::AutoReviewDiffView {
            title: "Diff",
            lines: &lines,
            scroll: 0,
            x_offset: 0,
        });
        let screen = render_auto_review_screen(args);

        // The title bar and the diff body show, with the close hint.
        assert!(
            screen.contains("Diff  (↑/↓ Scroll · Esc Close)"),
            "{screen:?}"
        );
        assert!(screen.contains("let x = 1;"), "{screen:?}");
        // The panes are hidden while the popup is open.
        assert!(!screen.contains("Files (1)"), "{screen:?}");
    }

    #[test]
    fn auto_review_browse_header_documents_item_keys() {
        // The Up/Down item cursor, the PageUp/PageDown category jump, and `-`
        // removal are documented in the browse key help, before Alt+x.
        let files = vec![review_entry("src/main.rs", ReviewStatus::Rejected, &[])];
        let report: Vec<String> = Vec::new();
        let mut args = auto_review_args(&files, &report, 200, 12);
        args.browsing = true;
        let screen = render_auto_review_screen(args);
        let header = screen.split("\r\n").next().unwrap_or_default();
        let item = header.find("↑/↓ Item").expect("item key documented");
        let category = header
            .find("PgUp/PgDn Category")
            .expect("category key documented");
        let remove = header.find("- Remove").expect("remove key documented");
        let exit = header.find("Alt+x").expect("exit key documented");
        assert!(
            item < category && category < exit && remove < exit,
            "{header:?}"
        );
    }

    #[test]
    fn auto_review_selected_item_lines_get_the_cursor_background() {
        // The line-cursor background marks the highlighted item's lines and
        // nothing else.
        let files = vec![review_entry("src/main.rs", ReviewStatus::Rejected, &[])];
        let report: Vec<String> = vec![
            "Code".to_string(),
            String::new(),
            "- finding one".to_string(),
            "- finding two".to_string(),
        ];
        let cursor_bg = "\u{1b}[48;2;60;60;90m";

        // No selection: no line carries the cursor background.
        let screen = render_auto_review_screen(auto_review_args(&files, &report, 80, 12));
        assert!(!screen.contains(cursor_bg), "{screen:?}");

        // Selecting the second finding's line highlights only it.
        let mut args = auto_review_args(&files, &report, 80, 12);
        args.browsing = true;
        args.selected_lines = Some((3, 4));
        let screen = render_auto_review_screen(args);
        let highlighted: Vec<&str> = screen
            .split("\r\n")
            .filter(|row| row.contains(cursor_bg))
            .collect();
        assert_eq!(highlighted.len(), 1, "{screen:?}");
        assert!(highlighted[0].contains("finding two"), "{highlighted:?}");
    }

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

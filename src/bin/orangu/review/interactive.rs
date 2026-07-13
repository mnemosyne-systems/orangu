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

/// The Alt+o feedback popup contents.
pub(crate) struct FeedbackWindow {
    pub(crate) title: String,
    /// The asked request, echoed below the title; `None` for a plain review.
    pub(crate) question: Option<String>,
    pub(crate) lines: Vec<String>,
    pub(crate) scroll: usize,
    pub(crate) x_offset: usize,
}

/// A review comment kept against a specific diff line of a file.
#[derive(Clone)]
pub(crate) struct ReviewComment {
    pub(crate) file: String,
    /// Diff-line index within the file (0-based).
    pub(crate) line: usize,
    /// Index of the comment's category, into `AUTO_REVIEW_CATEGORIES`.
    pub(crate) category: usize,
    pub(crate) text: String,
}

/// The inline Alt+c comment editor: the chosen category, which part has the
/// focus, and the comment text. Tab switches the focus between the single-line
/// category selector and the comment text; Up/Down move the category while the
/// selector has the focus.
pub(crate) struct CommentEditor {
    /// Index of the chosen category, into `AUTO_REVIEW_CATEGORIES`.
    pub(crate) category: usize,
    /// `true` while the focus is on the category selector.
    pub(crate) selector_focused: bool,
    pub(crate) input: InputState,
}

/// The category name for a comment's category index, falling back to the first
/// category (`Overall`) when the index is out of range.
pub(crate) fn review_category_name(category: usize) -> &'static str {
    crate::review::AUTO_REVIEW_CATEGORIES
        .get(category)
        .copied()
        .unwrap_or(crate::review::AUTO_REVIEW_CATEGORIES[0])
}

/// Interactive state for `/review` mode.
pub(crate) struct ReviewState {
    pub(crate) files: Vec<ReviewEntry>,
    pub(crate) selected: usize,
    pub(crate) list_offset: usize,
    /// Index of the highlighted line within the selected file's diff (moved
    /// with Up/Down).
    pub(crate) line: usize,
    /// Index of the first line shown in the left pane, within the selected
    /// file's diff.
    pub(crate) scroll: usize,
    /// Horizontal pan offset for the left pane.
    pub(crate) x_offset: usize,
    /// When set, the LLM feedback popup is open over the panes.
    pub(crate) feedback: Option<FeedbackWindow>,
    /// Comments recorded against diff lines, keyed by (file, line).
    pub(crate) comments: Vec<ReviewComment>,
    /// General notes entered in the input window as `# <note>`.
    pub(crate) general_notes: Vec<String>,
    /// When set, the inline comment editor is open for the highlighted line.
    pub(crate) comment_editor: Option<CommentEditor>,
}

/// Why `run_review_mode` returned control to the caller.
pub(crate) enum ReviewSignal {
    /// Leave review mode.
    Exit,
    /// Run an LLM review of the selected file using the typed request.
    RequestReview {
        path: String,
        patch: String,
        request: String,
    },
    /// Open the selected file in the configured editor.
    OpenFile { path: String },
}

/// Static rendering pieces for the review prompt frame.
#[derive(Clone, Copy)]
pub(crate) struct ReviewChrome<'a> {
    pub(crate) current_model: &'a str,
    pub(crate) prompt_branch: Option<&'a str>,
    pub(crate) pending_count: usize,
    pub(crate) skills: &'a orangu::skills::SkillRegistry,
}

impl ReviewState {
    pub(crate) fn update_list_offset(&mut self, right_body_height: usize) {
        if self.selected < self.list_offset {
            self.list_offset = self.selected;
        } else if self.selected >= self.list_offset + right_body_height {
            self.list_offset = self
                .selected
                .saturating_sub(right_body_height)
                .saturating_add(1);
        }
    }

    pub(crate) fn new(launch: ReviewLaunch) -> Self {
        Self {
            files: launch.files,
            selected: 0,
            list_offset: 0,
            line: 0,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        }
    }

    /// Record a `# <note>` typed in the input window as a general note.
    pub(crate) fn add_general_note(&mut self, text: &str) {
        let body = general_comment_body(text);
        if !body.is_empty() {
            self.general_notes.push(body);
        }
    }

    pub(crate) fn selected_lines(&self) -> &[String] {
        self.files
            .get(self.selected)
            .map(|file| file.diff_lines.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn selected_path(&self) -> Option<&str> {
        self.files.get(self.selected).map(|file| file.path.as_str())
    }

    fn max_scroll_for_lines(line_count: usize, visible_rows: usize) -> usize {
        line_count.saturating_sub(visible_rows.max(1))
    }

    fn apply_scroll_offset(offset: &mut usize, delta: isize, max_scroll: usize) {
        *offset = offset.saturating_add_signed(delta).min(max_scroll);
    }

    pub(crate) fn scroll_diff(&mut self, delta: isize, body_height: usize) {
        let max_scroll = Self::max_scroll_for_lines(self.selected_lines().len(), body_height);
        Self::apply_scroll_offset(&mut self.scroll, delta, max_scroll);
        self.line = self.line.min(self.selected_lines().len().saturating_sub(1));
    }

    pub(crate) fn scroll_feedback(&mut self, delta: isize, body_height: usize) {
        let Some(feedback) = &mut self.feedback else {
            return;
        };
        let review_rows = body_height
            .saturating_sub(usize::from(feedback.question.is_some()))
            .max(1);
        let max_scroll = Self::max_scroll_for_lines(feedback.lines.len(), review_rows);
        Self::apply_scroll_offset(&mut feedback.scroll, delta, max_scroll);
    }

    /// The existing comment recorded against the highlighted line, if any.
    pub(crate) fn comment_for_selected_line(&self) -> Option<&ReviewComment> {
        let path = self.selected_path()?;
        self.comments
            .iter()
            .find(|comment| comment.file == path && comment.line == self.line)
    }

    /// Diff-line indices of the selected file that carry a comment.
    pub(crate) fn commented_lines(&self) -> Vec<usize> {
        let Some(path) = self.selected_path() else {
            return Vec::new();
        };
        self.comments
            .iter()
            .filter(|comment| comment.file == path)
            .map(|comment| comment.line)
            .collect()
    }

    /// Open the inline comment editor for the highlighted line, pre-filled with
    /// any existing comment (its category and text), and scroll so the editor
    /// box fits below the line. A fresh comment defaults to the first category,
    /// `Overall`, with the focus on the comment text.
    pub(crate) fn open_comment_editor(&mut self, body_height: usize) {
        let existing = self.comment_for_selected_line();
        let category = existing.map(|comment| comment.category).unwrap_or(0);
        let text = existing
            .map(|comment| comment.text.clone())
            .unwrap_or_default();
        let mut input = InputState::default();
        input.set_buffer(text);
        self.comment_editor = Some(CommentEditor {
            category,
            selector_focused: false,
            input,
        });

        // Keep the highlighted line high enough that the box (the category row
        // plus the comment window) fits beneath it.
        let room = body_height.saturating_sub(orangu::tui::REVIEW_COMMENT_BOX_HEIGHT + 2);
        if self.line.saturating_sub(self.scroll) > room {
            self.scroll = self.line.saturating_sub(room);
        }
        if self.scroll > self.line {
            self.scroll = self.line;
        }
    }

    /// Save the editor's text as the comment for the highlighted line (an empty
    /// comment removes any existing one) and close the editor.
    pub(crate) fn commit_comment(&mut self) {
        let Some(editor) = self.comment_editor.take() else {
            return;
        };
        let Some(path) = self.selected_path().map(str::to_string) else {
            return;
        };
        let line = self.line;
        let text = editor.input.as_str().trim().to_string();
        self.comments
            .retain(|comment| !(comment.file == path && comment.line == line));
        if !text.is_empty() {
            self.comments.push(ReviewComment {
                file: path,
                line,
                category: editor.category,
                text,
            });
        }
    }

    /// Clamp scroll/pan offsets for whichever view is active.
    pub(crate) fn clamp(&mut self, body_height: usize, left_width: usize, full_width: usize) {
        if let Some(feedback) = &mut self.feedback {
            // A pinned question line costs one row of review text.
            let review_rows = body_height.saturating_sub(usize::from(feedback.question.is_some()));
            let max_scroll = feedback.lines.len().saturating_sub(review_rows);
            feedback.scroll = feedback.scroll.min(max_scroll);
            let content_width = feedback
                .lines
                .iter()
                .map(|line| orangu::tui::visible_line_width(line))
                .max()
                .unwrap_or(0);
            feedback.x_offset = feedback
                .x_offset
                .min(content_width.saturating_sub(full_width));
        } else {
            self.line = self.line.min(self.selected_lines().len().saturating_sub(1));
            let max_scroll = self.selected_lines().len().saturating_sub(body_height);
            self.scroll = self.scroll.min(max_scroll);
            let content_width = self
                .selected_lines()
                .iter()
                .map(|line| orangu::tui::visible_line_width(line))
                .max()
                .unwrap_or(0);
            self.x_offset = self.x_offset.min(content_width.saturating_sub(left_width));
        }
    }

    /// Move the highlighted line up, scrolling the pane to keep it visible.
    pub(crate) fn cursor_up(&mut self) {
        self.line = self.line.saturating_sub(1);
        if self.line < self.scroll {
            self.scroll = self.line;
        }
    }

    /// Move the highlighted line down, scrolling the pane to keep it visible.
    pub(crate) fn cursor_down(&mut self, body_height: usize) {
        let last = self.selected_lines().len().saturating_sub(1);
        self.line = (self.line + 1).min(last);
        if body_height > 0 && self.line >= self.scroll + body_height {
            self.scroll = self.line + 1 - body_height;
        }
    }

    pub(crate) fn select_next(&mut self) {
        if self.selected + 1 < self.files.len() {
            self.selected += 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    pub(crate) fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    pub(crate) fn set_status(&mut self, status: ReviewStatus) {
        if let Some(file) = self.files.get_mut(self.selected) {
            file.status = status;
        }
    }
}

pub(crate) fn build_review_prompt_with_stats(
    path: &str,
    request: &str,
    patch: &str,
    compression_enabled: bool,
    diff_file_cap: usize,
    store: Option<&orangu::compression_cache::CompressionStore>,
) -> (String, orangu::compression::CompressionStats) {
    let request = request.trim();
    let instruction = if request.is_empty() {
        format!(
            "Please review the following changes to `{path}` and give concise, actionable feedback."
        )
    } else {
        format!("Please review the following changes to `{path}`. {request}")
    };
    let (context, stats) = orangu::compression::prepare_llm_diff_context_with_stats(
        patch,
        compression_enabled,
        diff_file_cap,
        store,
    );
    let note = context
        .note
        .map(|note| format!("\n\n{note}"))
        .unwrap_or_default();
    (
        format!("{instruction}{note}\n\n```diff\n{}\n```", context.content),
        stats,
    )
}

/// The recorded review comments bucketed by category, in
/// `AUTO_REVIEW_CATEGORIES` order. Within each bucket the comments are ordered
/// by file then line and formatted like an auto review finding —
/// `**file:line**: text`, so the location renders bold — with the general
/// `# <note>` notes folded into the first bucket (`Overall`) as whole-patch
/// commentary. The line shown is the comment's **source-file** line, mapped
/// from its diff position through the file's patch (so the report and the export
/// appendix agree); it falls back to the diff position when the source line
/// cannot be determined.
fn review_findings_by_category(
    files: &[ReviewEntry],
    comments: &[ReviewComment],
    general_notes: &[String],
) -> Vec<Vec<String>> {
    let count = crate::review::AUTO_REVIEW_CATEGORIES.len();
    let mut buckets: Vec<Vec<String>> = vec![Vec::new(); count];

    // General notes are about the whole patch, so they lead the `Overall`
    // category.
    for note in general_notes {
        buckets[0].push(note.clone());
    }

    let mut ordered: Vec<&ReviewComment> = comments.iter().collect();
    ordered.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
    for comment in ordered {
        let index = comment.category.min(count - 1);
        buckets[index].push(format!(
            "**{}:{}**: {}",
            comment.file,
            review_comment_display_line(files, comment),
            comment.text
        ));
    }
    buckets
}

/// The 1-based line a comment is shown at: its **source-file** line, mapped from
/// the diff position it was recorded against through the file's patch, or the
/// diff position itself (`line + 1`) when the source line cannot be determined
/// (e.g. no matching file, or a paged diff the position cannot be traced in).
fn review_comment_display_line(files: &[ReviewEntry], comment: &ReviewComment) -> usize {
    files
        .iter()
        .find(|file| file.path == comment.file)
        .and_then(|file| review_comment_source_line(&file.patch, comment.line))
        .unwrap_or(comment.line + 1)
}

/// The new-file source line for diff-line index `index` into a file's unified
/// `patch` (the right side of the diff). New-file lines are counted across `@@`
/// hunk headers, context, and `+` lines (a `-` line removes content and so
/// occupies no new-file line). `None` when the index is before the first hunk or
/// past the end of the patch.
pub(crate) fn review_comment_source_line(patch: &str, index: usize) -> Option<usize> {
    let mut new_line = 0usize;
    let mut in_hunk = false;
    for (position, line) in patch.lines().enumerate() {
        if let Some(start) = review_hunk_new_start(line) {
            new_line = start;
            in_hunk = true;
            if position == index {
                return Some(new_line);
            }
            continue;
        }
        if !in_hunk {
            if position == index {
                return None;
            }
            continue;
        }
        match line.as_bytes().first() {
            // A removed line has no new-file line; anchor to the next one.
            Some(b'-') if position == index => return Some(new_line.max(1)),
            Some(b'-') => {}
            // Context, added, or any other body line occupies a new-file line.
            _ => {
                if position == index {
                    return Some(new_line.max(1));
                }
                new_line += 1;
            }
        }
    }
    None
}

/// The new-file start line of a unified-diff hunk header (`@@ -a,b +c,d @@`),
/// read from the `+c` field, or `None` when `line` is not a hunk header.
fn review_hunk_new_start(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("@@")?;
    let plus = rest.find('+')?;
    rest[plus + 1..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// The `/review` source appendix for the PDF export: the recorded comments
/// (and general notes), bucketed by category like the report, each paired with
/// the source code around its line — the same appendix the `/auto_review` export
/// produces. Empty when there are no comments or notes.
pub(crate) fn review_export_appendix(
    files: &[ReviewEntry],
    comments: &[ReviewComment],
    general_notes: &[String],
    workspace: &Path,
) -> Vec<crate::export::AutoReviewAppendixEntry> {
    let buckets = review_findings_by_category(files, comments, general_notes);
    crate::review::build_appendix_entries(&buckets, workspace)
}

/// The body of a `# <note>` general comment, with the leading `#` removed.
pub(crate) fn general_comment_body(text: &str) -> String {
    let trimmed = text.trim_start();
    trimmed
        .strip_prefix('#')
        .unwrap_or(trimmed)
        .trim_start()
        .to_string()
}

/// Build the `/review` exit report, grouped by category like `/auto_review`:
/// the lines rendered for the output window, and the raw Markdown kept for
/// `/comment <n> with review` and `/export review` (and copied to the clipboard
/// on exit). Each category — `Overall` through `Documentation` — is a heading
/// followed by its line comments as a bullet list (`No issues found` when
/// empty); the general `# <note>` notes lead the `Overall` category. The closing
/// `Conclusion` carries the bold verdict — `Patch approved` when every file is
/// approved, otherwise `Patch rejected` — followed by any rejected or
/// not-reviewed files, then the same closing `Generated by: **orangu
/// <version>**` attribution `/auto_review` closes with. The rendered lines
/// display the same report with the Markdown syntax consumed (bold headings,
/// `**file**` resolved to bold).
pub(crate) fn review_exit_output(
    files: &[ReviewEntry],
    comments: &[ReviewComment],
    general_notes: &[String],
    model: &str,
) -> (Vec<String>, String) {
    let buckets = review_findings_by_category(files, comments, general_notes);

    // The two variants stay in lockstep: `lines` goes to the output window,
    // `markdown` is kept for the report (and copied to the clipboard).
    let mut lines = Vec::new();
    let mut markdown = Vec::new();
    for (index, name) in crate::review::AUTO_REVIEW_CATEGORIES.iter().enumerate() {
        lines.push(format!("\x1b[1m{name}\x1b[0m"));
        markdown.push(format!("## {name}"));
        lines.push(String::new());
        markdown.push(String::new());
        let bucket = &buckets[index];
        if bucket.is_empty() {
            lines.push("No issues found".to_string());
            markdown.push("No issues found".to_string());
        } else {
            for finding in bucket {
                let bullet = format!("- {finding}");
                lines.extend(
                    render_markdown_for_console(&bullet)
                        .lines()
                        .map(str::to_string),
                );
                markdown.push(bullet);
            }
        }
        lines.push(String::new());
        markdown.push(String::new());
    }

    // The synthesized `Conclusion`: the verdict, then any rejected or
    // not-reviewed files (approved files are implied by the verdict).
    let conclusion = crate::review::AUTO_REVIEW_CONCLUSION;
    lines.push(format!("\x1b[1m{conclusion}\x1b[0m"));
    markdown.push(format!("## {conclusion}"));
    lines.push(String::new());
    markdown.push(String::new());

    let all_approved = !files.is_empty()
        && files
            .iter()
            .all(|file| file.status == ReviewStatus::Approved);
    let verdict = if all_approved {
        "Patch approved"
    } else {
        "Patch rejected"
    };
    lines.push(format!("\x1b[1m{verdict}\x1b[0m"));
    markdown.push(format!("**{verdict}**"));

    let mut findings = Vec::new();
    for file in files {
        if file.status == ReviewStatus::Rejected {
            findings.push(format!("Rejected: **{}**", file.path));
        }
    }
    for file in files {
        if file.status == ReviewStatus::Unreviewed {
            findings.push(format!("Not reviewed: **{}**", file.path));
        }
    }
    if !findings.is_empty() {
        lines.push(String::new());
        markdown.push(String::new());
        for finding in findings {
            lines.push(render_markdown_for_console(&format!("- {finding}")));
            markdown.push(format!("- {finding}"));
        }
    }

    // The report closes with the orangu version and reviewing model, same as
    // `/auto_review`.
    let generated_by = generated_by_markdown(model);
    lines.push(String::new());
    lines.push(render_markdown_for_console(&generated_by));
    markdown.push(String::new());
    markdown.push(generated_by);

    (lines, markdown.join("\n"))
}

pub(crate) fn print_review_screen(
    state: &ReviewState,
    input_state: &InputState,
    viewport: &ViewportState,
    chrome: ReviewChrome<'_>,
    left_status: Option<StatusFragment>,
    ghost: &str,
    print_screen_fn: &mut impl FnMut(ReviewScreenArgs<'_>),
) {
    let feedback = state.feedback.as_ref().map(|feedback| ReviewFeedbackView {
        title: &feedback.title,
        question: feedback.question.as_deref(),
        lines: &feedback.lines,
        scroll: feedback.scroll,
        x_offset: feedback.x_offset,
    });
    let comment_editor = state
        .comment_editor
        .as_ref()
        .map(|editor| ReviewCommentEditor {
            category: review_category_name(editor.category),
            selector_focused: editor.selector_focused,
            text: editor.input.as_str(),
            cursor: editor.input.cursor(),
        });
    let commented_lines = state.commented_lines();
    print_screen_fn(ReviewScreenArgs {
        files: &state.files,
        selected: state.selected,
        list_offset: state.list_offset,
        line: state.line,
        scroll: state.scroll,
        x_offset: state.x_offset,
        feedback,
        comment_editor,
        commented_lines: &commented_lines,
        current_model: chrome.current_model,
        prompt_branch: chrome.prompt_branch,
        input: input_state.as_str(),
        cursor: input_state.cursor(),
        ghost,
        left_status,
        pending_count: chrome.pending_count,
        actual_width: viewport.actual_width,
        actual_height: viewport.actual_height,
    });
}

/// Run the review event loop until the user exits or asks for an LLM review.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_review_mode(
    state: &mut ReviewState,
    viewport: &mut ViewportState,
    input_state: &mut InputState,
    chrome: ReviewChrome<'_>,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
    print_screen_fn: &mut impl FnMut(ReviewScreenArgs<'_>),
) -> Result<ReviewSignal> {
    let mut escape_cancel = EscapeCancelState::default();
    loop {
        let body_height = review_pane_body_height(
            viewport.actual_height,
            input_state.as_str(),
            chrome.prompt_branch,
            viewport.actual_width,
        );
        let right_width = orangu::tui::review_right_width(&state.files, viewport.actual_width);
        let left_width = viewport.actual_width.saturating_sub(right_width).max(1);
        state.clamp(body_height, left_width, viewport.actual_width);
        // Preview the file/command Tab would fill in, the same way the main
        // prompt does, so `/open_file ` and `open ` complete project files.
        let ghost = crate::completion::input_ghost_suffix(
            input_state.as_str(),
            input_state.cursor(),
            input_state.ghost_index,
            workspace,
            server_names,
            available_models,
            chrome.skills,
        )
        .unwrap_or_default();

        let prefix = orangu::tui::screen::prompt_prefix(chrome.prompt_branch);
        let input_lines_count = orangu::tui::screen::wrapped_input_lines(
            input_state.as_str(),
            viewport.actual_width,
            &prefix,
        )
        .len();
        let prompt_frame_height = input_lines_count + 3;
        // The native layout uses two rows for the mode header, plus the
        // right pane's one-row top padding and `Files` heading.
        let right_body_height = viewport
            .actual_height
            .saturating_sub(prompt_frame_height)
            .saturating_sub(4)
            .max(1);
        state.update_list_offset(right_body_height);

        print_review_screen(
            state,
            input_state,
            viewport,
            chrome,
            None,
            &ghost,
            print_screen_fn,
        );
        std::io::stdout().flush()?;

        let (code, modifiers) = match event::read()? {
            Event::Resize(width, height) => {
                viewport.on_resize(usize::from(width), usize::from(height));
                continue;
            }
            Event::Mouse(crossterm::event::MouseEvent {
                kind, column, row, ..
            }) => {
                match kind {
                    crossterm::event::MouseEventKind::ScrollUp => {
                        let scroll_rows = crate::input::wheel_scroll_lines(body_height);
                        if state.feedback.is_some() {
                            state.scroll_feedback(-(scroll_rows as isize), body_height);
                        } else {
                            state.scroll_diff(-(scroll_rows as isize), body_height);
                        }
                    }
                    crossterm::event::MouseEventKind::ScrollDown => {
                        let scroll_rows = crate::input::wheel_scroll_lines(body_height);
                        if state.feedback.is_some() {
                            state.scroll_feedback(scroll_rows as isize, body_height);
                        } else {
                            state.scroll_diff(scroll_rows as isize, body_height);
                        }
                    }
                    crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if column as usize >= left_width =>
                    {
                        let list_start = state.list_offset;
                        // Rows 0-1 are the mode header, row 2 is the
                        // right pane's top padding, and row 3 is its
                        // heading; file entries start at row 4.
                        if row >= 4 {
                            let click_y = (row - 4) as usize;
                            let file_index = list_start + click_y;
                            if file_index < state.files.len() {
                                state.selected = file_index;
                                state.line = 0;
                                state.scroll = 0;
                            }
                        }
                    }
                    _ => {}
                }
                continue;
            }
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind,
                ..
            }) if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat => (code, modifiers),
            _ => continue,
        };

        let alt =
            modifiers.contains(KeyModifiers::ALT) && !modifiers.contains(KeyModifiers::CONTROL);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);

        // While the feedback popup is open it is modal: scroll it, or close it.
        if state.feedback.is_some() {
            escape_cancel.reset();
            match code {
                KeyCode::Char('x') | KeyCode::Esc => state.feedback = None,
                KeyCode::Up => state.scroll_feedback(-1, body_height),
                KeyCode::Down => state.scroll_feedback(1, body_height),
                KeyCode::Left => {
                    let feedback = state.feedback.as_mut().unwrap();
                    feedback.x_offset = feedback.x_offset.saturating_sub(1);
                }
                KeyCode::Right => {
                    let feedback = state.feedback.as_mut().unwrap();
                    feedback.x_offset = feedback.x_offset.saturating_add(1);
                }
                KeyCode::PageUp => state.scroll_feedback(
                    -(crate::input::page_scroll_lines(body_height) as isize),
                    body_height,
                ),
                KeyCode::PageDown => state.scroll_feedback(
                    crate::input::page_scroll_lines(body_height) as isize,
                    body_height,
                ),
                _ => {}
            }
            continue;
        }

        // While the inline comment editor is open it is modal: Tab switches the
        // focus between the category selector and the comment text, Up/Down move
        // the category while the selector has the focus, Enter saves it, and Esc
        // discards it.
        if let Some(editor) = state.comment_editor.as_ref() {
            escape_cancel.reset();
            let selector_focused = editor.selector_focused;
            match (code, alt, ctrl) {
                (KeyCode::Enter, _, _) => state.commit_comment(),
                (KeyCode::Esc, _, _) => state.comment_editor = None,
                (KeyCode::Tab, _, _) | (KeyCode::BackTab, _, _) => {
                    let editor = state.comment_editor.as_mut().unwrap();
                    editor.selector_focused = !editor.selector_focused;
                }
                // While the selector has the focus, Up/Down pick the category
                // and every other key is ignored (no typing into the comment).
                (KeyCode::Up, _, _) if selector_focused => {
                    let editor = state.comment_editor.as_mut().unwrap();
                    editor.category = editor.category.saturating_sub(1);
                }
                (KeyCode::Down, _, _) if selector_focused => {
                    let editor = state.comment_editor.as_mut().unwrap();
                    editor.category =
                        (editor.category + 1).min(crate::review::AUTO_REVIEW_CATEGORIES.len() - 1);
                }
                _ if selector_focused => {}
                (KeyCode::Backspace, true, _) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .input
                        .delete_backward_readline_word();
                }
                (KeyCode::Backspace, _, _) => {
                    state.comment_editor.as_mut().unwrap().input.backspace();
                }
                (KeyCode::Delete, _, _) => state.comment_editor.as_mut().unwrap().input.delete(),
                (KeyCode::Left, _, true) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .input
                        .move_backward_readline_word();
                }
                (KeyCode::Right, _, true) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .input
                        .move_forward_readline_word();
                }
                (KeyCode::Left, _, _) => state.comment_editor.as_mut().unwrap().input.move_left(),
                (KeyCode::Right, _, _) => state.comment_editor.as_mut().unwrap().input.move_right(),
                (KeyCode::Home, _, _) => state.comment_editor.as_mut().unwrap().input.move_home(),
                (KeyCode::End, _, _) => state.comment_editor.as_mut().unwrap().input.move_end(),
                (KeyCode::Char(ch), false, false) => {
                    state.comment_editor.as_mut().unwrap().input.insert_char(ch);
                }
                _ => {}
            }
            continue;
        }

        // A second Esc within the timeout leaves review mode; the first arms it.
        if code == KeyCode::Esc {
            if escape_cancel.handle_escape(std::time::Instant::now()) {
                return Ok(ReviewSignal::Exit);
            }
            continue;
        }
        escape_cancel.reset();

        match (code, alt, ctrl) {
            (KeyCode::Char('x'), true, _) => return Ok(ReviewSignal::Exit),
            (KeyCode::Char('j'), true, _) => state.select_next(),
            (KeyCode::Char('k'), true, _) => state.select_prev(),
            (KeyCode::Char('a'), true, _) => state.set_status(ReviewStatus::Approved),
            (KeyCode::Char('r'), true, _) => state.set_status(ReviewStatus::Rejected),
            (KeyCode::Char('c'), true, _) => state.open_comment_editor(body_height),
            (KeyCode::Char('e'), true, _) => {
                if let Some(file) = state.files.get(state.selected) {
                    return Ok(ReviewSignal::OpenFile {
                        path: file.path.clone(),
                    });
                }
            }
            (KeyCode::Char('o'), true, _) | (KeyCode::Enter, _, _) => {
                if input_state.as_str().trim_start().starts_with('#') {
                    // A `# <note>` in the input window is a general note, not an
                    // LLM request.
                    state.add_general_note(input_state.as_str());
                    input_state.clear();
                } else if let Some(path) =
                    crate::commands::parse_open_command_target(input_state.as_str())
                {
                    // `/open_file <path>` or `open <path>` opens any project
                    // file in the editor — not only the changed files. Available
                    // the whole time in `/review`.
                    let path = path.to_string();
                    input_state.clear();
                    return Ok(ReviewSignal::OpenFile { path });
                } else if let Some(file) = state.files.get(state.selected) {
                    return Ok(ReviewSignal::RequestReview {
                        path: file.path.clone(),
                        patch: file.patch.clone(),
                        request: input_state.as_str().to_string(),
                    });
                }
            }
            // Left-pane scrolling (Alt+arrows / PageUp/Down), mirroring the
            // main output window.
            (KeyCode::Up, true, _) => state.scroll_diff(-1, body_height),
            (KeyCode::Down, true, _) => state.scroll_diff(1, body_height),
            (KeyCode::Left, true, _) => state.x_offset = state.x_offset.saturating_sub(1),
            (KeyCode::Right, true, _) => state.x_offset = state.x_offset.saturating_add(1),
            (KeyCode::PageUp, _, _) => state.scroll_diff(
                -(crate::input::page_scroll_lines(body_height) as isize),
                body_height,
            ),
            (KeyCode::PageDown, _, _) => state.scroll_diff(
                crate::input::page_scroll_lines(body_height) as isize,
                body_height,
            ),
            // Move the highlighted line through the diff, view following.
            (KeyCode::Up, false, _) => state.cursor_up(),
            (KeyCode::Down, false, _) => state.cursor_down(body_height),
            // Tab completes the input window — `/open_file ` and `open ` over
            // every project file, like the main prompt — and Shift+Tab cycles
            // the ghost preview.
            (KeyCode::Tab, _, _) => {
                crate::input::apply_completion(input_state, workspace, &[], &[], chrome.skills);
            }
            (KeyCode::BackTab, _, _) => crate::input::cycle_ghost_suggestion(input_state),
            // Input window editing.
            (KeyCode::Backspace, true, _) => input_state.delete_backward_readline_word(),
            (KeyCode::Backspace, _, _) => input_state.backspace(),
            (KeyCode::Delete, _, _) => input_state.delete(),
            (KeyCode::Left, _, true) => input_state.move_backward_readline_word(),
            (KeyCode::Right, _, true) => input_state.move_forward_readline_word(),
            (KeyCode::Left, _, _) => input_state.move_left(),
            (KeyCode::Right, _, _) => input_state.move_right(),
            (KeyCode::Home, _, _) | (KeyCode::Char('a'), _, true) => input_state.move_home(),
            (KeyCode::End, _, _) | (KeyCode::Char('e'), _, true) => input_state.move_end(),
            (KeyCode::Char('k'), _, true) => input_state.kill_to_end(),
            (KeyCode::Char('u'), _, true) => input_state.kill_to_start(),
            (KeyCode::Char('w'), _, true) => input_state.delete_prev_word(),
            (KeyCode::Char(ch), false, false) => input_state.insert_char(ch),
            _ => {}
        }
    }
}

/// Result of an Alt+o review request.
pub(crate) enum ReviewRequestOutcome {
    /// The model responded (`Ok`) or the request errored (`Err`); either way the
    /// outcome is shown in the feedback popup.
    Completed(Result<String>),
    /// The user pressed Esc twice — abort and return to the panes.
    Cancelled,
    /// The user pressed Alt+x — leave review mode entirely.
    Exit,
}

/// Ask the LLM to review the selected file, rendering the review screen with a
/// thinking indicator until the response arrives. The exchange is recorded in
/// the session so it can be followed up after leaving review mode. While the
/// model works, `Esc` `Esc` cancels the request and `Alt+x` exits review mode;
/// either way the pending exchange is rolled back out of the session.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_review_request(
    session: &mut ChatSession,
    prompt: &str,
    profile: &LlmConfiguration,
    tools: &ToolExecutor,
    state: &ReviewState,
    input_state: &InputState,
    viewport: &mut ViewportState,
    chrome: ReviewChrome<'_>,
    print_screen_fn: &mut impl FnMut(ReviewScreenArgs<'_>),
) -> Result<ReviewRequestOutcome> {
    let checkpoint = session.checkpoint();
    let mut future =
        Box::pin(session.prompt(prompt, profile, tools, |_| {}, |_| {}, |_| {}, |_| {}));
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let started = std::time::Instant::now();
    let mut escape_cancel = EscapeCancelState::default();

    loop {
        tokio::select! {
            result = &mut future => return Ok(ReviewRequestOutcome::Completed(result)),
            _ = interval.tick() => {
                while event::poll(std::time::Duration::ZERO)? {
                    let (code, modifiers) = match event::read()? {
                        Event::Resize(width, height) => {
                            viewport.on_resize(usize::from(width), usize::from(height));
                            continue;
                        }
                        Event::Key(KeyEvent { code, modifiers, kind, .. })
                            if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat =>
                        {
                            (code, modifiers)
                        }
                        _ => continue,
                    };
                    let alt = modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL);
                    if code == KeyCode::Char('x') && alt {
                        drop(future);
                        session.rollback(checkpoint);
                        return Ok(ReviewRequestOutcome::Exit);
                    }
                    if code == KeyCode::Esc {
                        if escape_cancel.handle_escape(std::time::Instant::now()) {
                            drop(future);
                            session.rollback(checkpoint);
                            return Ok(ReviewRequestOutcome::Cancelled);
                        }
                    } else {
                        escape_cancel.reset();
                    }
                }
                let frame = (started.elapsed().as_millis()
                    / THINKING_FRAME_INTERVAL.as_millis().max(1)) as usize;
                let status = render_thinking_status(frame, started.elapsed());
                print_review_screen(state, input_state, viewport, chrome, Some(status), "", print_screen_fn);
                std::io::stdout().flush()?;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn review_state_navigation_shows_only_selected_file_diff() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![
                ReviewEntry {
                    path: "a.txt".to_string(),
                    status: ReviewStatus::Unreviewed,
                    diff_lines: (0..30).map(|i| format!("a {i}")).collect(),
                    patch: String::new(),
                },
                ReviewEntry {
                    path: "b.txt".to_string(),
                    status: ReviewStatus::Unreviewed,
                    diff_lines: (0..8).map(|i| format!("b {i}")).collect(),
                    patch: String::new(),
                },
            ],
            selected: 0,
            list_offset: 0,
            line: 0,
            scroll: 7,
            x_offset: 5,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // The left pane reflects the selected file's own diff.
        assert_eq!(state.selected_lines().len(), 30);

        // Moving to the next file shows it from the top.
        state.select_next();
        assert_eq!(state.selected, 1);
        assert_eq!(state.scroll, 0, "scroll resets on file change");
        assert_eq!(state.x_offset, 0, "horizontal pan resets on file change");
        assert_eq!(state.selected_lines().len(), 8);

        // Cannot move past the last file.
        state.select_next();
        assert_eq!(state.selected, 1);

        // Scroll is clamped to the selected file's diff length minus the body.
        state.scroll = 999;
        state.clamp(5, 20, 40);
        assert_eq!(state.scroll, 8 - 5);

        // Marking sets status on the selected file only.
        state.set_status(ReviewStatus::Approved);
        assert_eq!(state.files[1].status, ReviewStatus::Approved);
        assert_eq!(state.files[0].status, ReviewStatus::Unreviewed);

        state.select_prev();
        assert_eq!(state.selected, 0);
        assert_eq!(state.scroll, 0);
        assert_eq!(state.line, 0, "line cursor resets on file change");
    }

    #[test]
    fn review_cursor_moves_and_scrolls_to_follow() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: (0..20).map(|i| format!("a {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            list_offset: 0,
            line: 0,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Down past the visible body height scrolls the pane to follow.
        let body = 5;
        for _ in 0..6 {
            state.cursor_down(body);
        }
        assert_eq!(state.line, 6);
        assert_eq!(state.scroll, 6 + 1 - body, "view follows the cursor down");

        // Back up above the top scrolls the pane back up.
        for _ in 0..5 {
            state.cursor_up();
        }
        assert_eq!(state.line, 1);
        assert_eq!(state.scroll, 1, "view follows the cursor up");

        // The cursor cannot move past the last line.
        for _ in 0..100 {
            state.cursor_down(body);
        }
        assert_eq!(state.line, 19);
    }

    #[test]
    fn review_comments_are_recorded_per_file_and_line() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Unreviewed,
            diff_lines: (0..10).map(|i| format!("x {i}")).collect(),
            patch: String::new(),
        };
        let mut state = ReviewState {
            files: vec![entry("a.txt"), entry("b.txt")],
            selected: 0,
            list_offset: 0,
            line: 3,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Open the editor (pre-filled empty), type, and commit. A fresh comment
        // defaults to the first category, Overall, with the focus on the text.
        state.open_comment_editor(20);
        assert!(state.comment_editor.is_some());
        assert_eq!(state.comment_editor.as_ref().unwrap().category, 0);
        assert!(!state.comment_editor.as_ref().unwrap().selector_focused);
        for ch in "looks off".chars() {
            state.comment_editor.as_mut().unwrap().input.insert_char(ch);
        }
        state.commit_comment();
        assert!(state.comment_editor.is_none());
        assert_eq!(state.comments.len(), 1);
        assert_eq!(state.comments[0].file, "a.txt");
        assert_eq!(state.comments[0].line, 3);
        assert_eq!(state.comments[0].category, 0);
        assert_eq!(state.comments[0].text, "looks off");
        assert_eq!(state.commented_lines(), vec![3]);

        // Re-opening pre-fills the existing comment; editing replaces it.
        state.open_comment_editor(20);
        assert_eq!(
            state.comment_editor.as_ref().unwrap().input.as_str(),
            "looks off"
        );
        state.commit_comment();
        assert_eq!(state.comments.len(), 1, "no duplicate for the same line");

        // An empty comment removes it.
        state.open_comment_editor(20);
        state.comment_editor.as_mut().unwrap().input.kill_to_start();
        state.commit_comment();
        assert!(state.comments.is_empty());
        assert!(state.commented_lines().is_empty());

        // Comments are scoped to the selected file.
        state.open_comment_editor(20);
        for ch in "note".chars() {
            state.comment_editor.as_mut().unwrap().input.insert_char(ch);
        }
        state.commit_comment();
        state.select_next();
        assert_eq!(state.selected, 1);
        assert!(
            state.commented_lines().is_empty(),
            "b.txt has no comments yet"
        );
    }

    #[test]
    fn alt_c_on_commented_line_opens_editor_prefilled_in_the_box() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewCommentEditor, ReviewEntry, ReviewScreenArgs, ReviewStatus};
        use ratatui::{Terminal, backend::TestBackend};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: (0..10).map(|i| format!("x {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            list_offset: 0,
            line: 2,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Record a comment on the highlighted line, then re-open with Alt+c.
        state.open_comment_editor(12);
        for ch in "needs a guard".chars() {
            state.comment_editor.as_mut().unwrap().input.insert_char(ch);
        }
        state.commit_comment();
        state.open_comment_editor(12);

        // The editor holds the existing comment.
        assert_eq!(
            state.comment_editor.as_ref().unwrap().input.as_str(),
            "needs a guard"
        );

        let editor = state
            .comment_editor
            .as_ref()
            .map(|editor| ReviewCommentEditor {
                category: crate::review::review_category_name(editor.category),
                selector_focused: editor.selector_focused,
                text: editor.input.as_str(),
                cursor: editor.input.cursor(),
            });
        let commented = state.commented_lines();
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
        terminal
            .draw(|frame| {
                orangu::tui::review_native::draw_review_screen(
                    frame,
                    ReviewScreenArgs {
                        files: &state.files,
                        selected: state.selected,
                        list_offset: state.list_offset,
                        line: state.line,
                        scroll: state.scroll,
                        x_offset: state.x_offset,
                        feedback: None,
                        comment_editor: editor,
                        commented_lines: &commented,
                        current_model: "model",
                        prompt_branch: Some("main"),
                        input: "",
                        cursor: 0,
                        ghost: "",
                        left_status: None,
                        pending_count: 0,
                        actual_width: 80,
                        actual_height: 20,
                    },
                );
            })
            .unwrap();

        let mut rendered = String::new();
        for y in 0..20 {
            for x in 0..80 {
                rendered.push_str(terminal.backend().buffer().cell((x, y)).unwrap().symbol());
            }
            rendered.push('\n');
        }

        assert!(
            rendered.contains("needs a guard"),
            "existing comment not rendered in native review screen"
        );
    }

    #[test]
    fn comment_editor_keeps_and_reloads_the_chosen_category() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: (0..10).map(|i| format!("x {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            list_offset: 0,
            line: 1,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Open the editor, pick a non-default category (as Up/Down would), type
        // the comment, and commit it.
        state.open_comment_editor(20);
        {
            let editor = state.comment_editor.as_mut().unwrap();
            editor.category = 2; // Security
            for ch in "unsafe input".chars() {
                editor.input.insert_char(ch);
            }
        }
        state.commit_comment();
        assert_eq!(state.comments[0].category, 2);

        // Re-opening the line pre-fills both the category and the text.
        state.open_comment_editor(20);
        let editor = state.comment_editor.as_ref().unwrap();
        assert_eq!(editor.category, 2);
        assert_eq!(editor.input.as_str(), "unsafe input");
    }

    #[test]
    fn review_exit_output_groups_comments_by_category() {
        use crate::review::{ReviewComment, review_exit_output};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str, status| ReviewEntry {
            path: path.to_string(),
            status,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };

        let files = vec![
            entry("a.txt", ReviewStatus::Approved),
            entry("b.txt", ReviewStatus::Rejected),
            entry("c.txt", ReviewStatus::Unreviewed),
        ];
        let comments = vec![
            ReviewComment {
                file: "b.txt".to_string(),
                line: 2,
                category: 2, // Security
                text: "fix this".to_string(),
            },
            ReviewComment {
                file: "a.txt".to_string(),
                line: 0,
                category: 0, // Overall
                text: "nit".to_string(),
            },
        ];
        // General notes lead the Overall category.
        let notes = vec!["ship after nits".to_string()];

        let (_lines, markdown) = review_exit_output(&files, &comments, &notes, "gemma");

        // The Markdown is category-grouped like /auto_review: each category is a
        // `## heading`, the general note leads Overall, each line comment sits
        // under its category, empty categories read "No issues found", and the
        // Conclusion carries the verdict and the rejected/not-reviewed files.
        assert_eq!(
            markdown,
            format!(
                "## Overall\n\
             \n\
             - ship after nits\n\
             - **a.txt:1**: nit\n\
             \n\
             ## Code\n\
             \n\
             No issues found\n\
             \n\
             ## Security\n\
             \n\
             - **b.txt:3**: fix this\n\
             \n\
             ## Memory\n\
             \n\
             No issues found\n\
             \n\
             ## Performance\n\
             \n\
             No issues found\n\
             \n\
             ## Test Suite\n\
             \n\
             No issues found\n\
             \n\
             ## Documentation\n\
             \n\
             No issues found\n\
             \n\
             ## Conclusion\n\
             \n\
             **Patch rejected**\n\
             \n\
             - Rejected: **b.txt**\n\
             - Not reviewed: **c.txt**\n\
             \n\
             Generated by: **[orangu](https://mnemosyne-systems.github.io/orangu/) {}** (gemma)",
                crate::VERSION,
            )
        );
    }

    #[test]
    fn review_comment_source_line_maps_through_the_patch() {
        use crate::review::review_comment_source_line;

        let patch = "diff --git a/a.rs b/a.rs\n\
                     index 0000000..1111111 100644\n\
                     --- a/a.rs\n\
                     +++ b/a.rs\n\
                     @@ -1,3 +5,4 @@\n\
                      ctx1\n\
                      ctx2\n\
                     +added\n\
                      ctx3";
        // The hunk starts at new-file line 5; the `+added` line sits at index 7.
        assert_eq!(review_comment_source_line(patch, 5), Some(5)); // ctx1
        assert_eq!(review_comment_source_line(patch, 6), Some(6)); // ctx2
        assert_eq!(review_comment_source_line(patch, 7), Some(7)); // +added
        assert_eq!(review_comment_source_line(patch, 8), Some(8)); // ctx3
        // A position before the first hunk has no source line; an empty patch too.
        assert_eq!(review_comment_source_line(patch, 0), None);
        assert_eq!(review_comment_source_line("", 0), None);
    }

    #[test]
    fn review_export_appendix_windows_source_around_the_mapped_line() {
        use crate::review::{ReviewComment, review_export_appendix};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        // A workspace with the reviewed file on disk; the appendix reads it.
        let workspace = tempfile::tempdir().expect("workspace");
        let body: String = (1..=20).map(|n| format!("line {n}\n")).collect();
        std::fs::write(workspace.path().join("a.rs"), body).expect("write file");

        // A patch whose `+` line at diff index 7 is new-file line 7.
        let patch = "@@ -1,3 +5,4 @@\n line 5\n line 6\n+line 7\n line 8";
        let files = vec![ReviewEntry {
            path: "a.rs".to_string(),
            status: ReviewStatus::Rejected,
            diff_lines: vec!["@@".to_string()],
            patch: patch.to_string(),
        }];
        // The comment is recorded against diff index 3 (the `+line 7` line is the
        // 4th line of this patch, index 3).
        let comments = vec![ReviewComment {
            file: "a.rs".to_string(),
            line: 3,
            category: 1, // Code
            text: "boom".to_string(),
        }];

        let appendix = review_export_appendix(&files, &comments, &[], workspace.path());
        assert_eq!(appendix.len(), 1);
        assert_eq!(appendix[0].category, "Code");
        // The finding and window use the mapped source line (7), not the diff
        // index (4): ±3 lines around line 7 → lines 4..=10, starting at 4.
        assert_eq!(appendix[0].finding, "**a.rs:7**: boom");
        assert_eq!(appendix[0].highlight, Some((7, 7)));
        assert_eq!(appendix[0].start_line, 4);
        assert_eq!(appendix[0].code.first().map(String::as_str), Some("line 4"));
        assert_eq!(appendix[0].code.last().map(String::as_str), Some("line 10"));
    }

    #[test]
    fn review_exit_output_approves_when_every_file_is_approved() {
        use crate::review::review_exit_output;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let files = vec![ReviewEntry {
            path: "a.txt".to_string(),
            status: ReviewStatus::Approved,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        }];

        let (lines, markdown) = review_exit_output(&files, &[], &[], "");
        // Every category reports "No issues found" and the Conclusion approves
        // the patch with no rejected/not-reviewed files to list, followed by the
        // closing attribution (no model suffix since none was given).
        assert!(markdown.contains("## Conclusion\n\n**Patch approved**"));
        assert!(markdown.ends_with(&format!(
            "Generated by: **[orangu](https://mnemosyne-systems.github.io/orangu/) {}**",
            crate::VERSION
        )));
        assert!(markdown.contains("## Overall\n\nNo issues found"));
        // The rendered output window shows the bold verdict.
        assert!(
            lines.contains(&"\x1b[1mPatch approved\x1b[0m".to_string()),
            "{lines:?}"
        );
    }

    #[test]
    fn add_general_note_strips_hash_and_keeps_line_comments_separate() {
        use crate::review::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: (0..5).map(|i| format!("x {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            list_offset: 0,
            line: 2,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Input-window "# ..." is stored as a general note with the '#' removed.
        state.add_general_note("# please add a test");
        state.add_general_note("#no space");
        // Whitespace-only / bare '#' notes are ignored.
        state.add_general_note("#   ");

        assert_eq!(
            state.general_notes,
            vec!["please add a test".to_string(), "no space".to_string()]
        );
        // General notes do not become line comments.
        assert!(state.comments.is_empty());
    }
}

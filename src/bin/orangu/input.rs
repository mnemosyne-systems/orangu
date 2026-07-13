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

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use orangu::{
    llm::{ChatMessage, StreamMetrics},
    session::ChatSession,
    tui::{
        Banner, HeaderStatus, StatusFragment, TabStatus, TranscriptLine, WorkspaceTabsView,
        visible_line_width,
    },
};
use std::{
    collections::VecDeque,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::VERSION;

use super::completion::{
    completion_candidates, first_ghost_word, natural_language_ghost_candidates,
    natural_language_ghost_suffix_at,
};

pub const ESC_CANCEL_TIMEOUT: Duration = Duration::from_secs(2);
pub const CTRL_C_EXIT_TIMEOUT: Duration = Duration::from_secs(2);
pub const CTRL_C_EXIT_MESSAGE: &str = "Press Ctrl+c again to quit";
pub const IDLE_STATUS_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
pub const TRANSCRIPT_MAX_LINES: usize = 10_000;

pub(crate) fn wheel_scroll_lines(visible_rows: usize) -> usize {
    match visible_rows {
        0..=10 => 1,
        11..=24 => 2,
        _ => 3,
    }
}

pub(crate) fn page_scroll_lines(visible_rows: usize) -> usize {
    visible_rows.saturating_sub(2).max(1)
}

pub struct CompletionState {
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub candidates: Vec<String>,
    pub index: usize,
}

pub enum InputResult {
    Submitted(String),
    Refresh,
    Quit,
    /// Move focus to the previous workspace tab (`Alt+,`).
    WorkspacePrevious,
    /// Move focus to the next workspace tab (`Alt+.`).
    WorkspaceNext,
    /// Start a new workspace tab (`Alt+Insert`).
    WorkspaceNew,
    /// Close the active workspace tab (`Alt+Delete`).
    WorkspaceClose,
}

/// An LLM response that is streaming in a background tokio task while the user
/// has switched to another workspace tab. The handle owns the `ChatSession`
/// and returns it together with the result when it finishes.
pub struct PendingResponse {
    pub(crate) stream_state: Arc<Mutex<StreamRenderState>>,
    pub(crate) handle: tokio::task::JoinHandle<(ChatSession, anyhow::Result<String>)>,
    pub(crate) llm_start: std::time::Instant,
    pub(crate) tool_time_before: std::time::Duration,
    /// Snapshot of session messages taken before the spawn so we can restore a
    /// clean session if the user later presses Escape to cancel the response.
    pub(crate) saved_messages: Vec<ChatMessage>,
}

pub enum WaitResult {
    Response(String),
    Cancelled(String),
    Failed {
        partial: String,
        error: anyhow::Error,
    },
    Quit,
    /// The prompt future is running in a background task; the caller should
    /// store the [`PendingResponse`] on the current tab and switch to another.
    BackgroundStreaming(PendingResponse),
}

pub struct InputEventResult {
    pub redraw: bool,
    pub outcome: Option<InputResult>,
}

#[derive(Clone, Copy)]
pub struct RenderContext<'a> {
    pub current_model: &'a str,
    pub endpoint: &'a str,
    pub workspace: &'a Path,
    pub prompt_branch: Option<&'a str>,
    pub header_status: HeaderStatus,
    pub virtual_width: usize,
    pub actual_width: usize,
    pub actual_height: usize,
    pub x_offset: usize,
    pub banner: Banner,
    pub drop_down: bool,
    /// Currently unread: carried from `config.feedback` for future use by the
    /// screen renderer.
    #[allow(dead_code)]
    pub feedback: bool,
    /// Configured server names, used to hint `/server` argument completions in
    /// the inline ghost.
    pub server_names: &'a [String],
    /// Available model ids, used to hint `/model` argument completions in the
    /// inline ghost.
    pub available_models: &'a [String],
    pub skills: &'a orangu::skills::SkillRegistry,
    /// The workspace tab bar to draw, or `None` when a single workspace is
    /// open. Placed per the `workspaces` configuration.
    pub tab_bar: Option<WorkspaceTabsView>,
    /// Per-tab status dots for the tab bar, in left-to-right order. Empty when
    /// `feedback` is off or only one tab is open.
    pub tab_statuses: &'a [TabStatus],
}

#[derive(Clone, Debug)]
pub struct ViewportState {
    pub virtual_width: usize,
    pub actual_width: usize,
    pub actual_height: usize,
    pub x_offset: usize,
}

impl ViewportState {
    pub fn new(virtual_width: usize, actual_width: usize, actual_height: usize) -> Self {
        Self {
            virtual_width: virtual_width.max(actual_width),
            actual_width,
            actual_height,
            x_offset: 0,
        }
    }

    pub fn on_resize(&mut self, new_width: usize, new_height: usize) {
        self.actual_width = new_width;
        self.actual_height = new_height;
        if new_width > self.virtual_width {
            self.virtual_width = new_width;
        }
        self.clamp_offset();
    }

    pub fn pan_left(&mut self) {
        self.x_offset = self.x_offset.saturating_sub(1);
    }

    pub fn pan_right(&mut self, max_content_width: usize, visible_width: usize) {
        self.x_offset = self.x_offset.saturating_add(1);
        self.clamp_to_content(max_content_width, visible_width);
    }

    fn clamp_offset(&mut self) {
        let max_offset = self.virtual_width.saturating_sub(self.actual_width);
        self.x_offset = self.x_offset.min(max_offset);
    }

    pub fn clamp_to_content(&mut self, max_content_width: usize, visible_width: usize) {
        let effective_right = max_content_width.min(self.virtual_width);
        let max_offset = effective_right.saturating_sub(visible_width.max(1));
        self.x_offset = self.x_offset.min(max_offset);
    }
}

#[derive(Clone)]
pub struct ScreenState<'a> {
    pub transcript: &'a [TranscriptLine],
    pub transcript_epoch: usize,
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_lines: &'a [TranscriptLine],
    pub input: &'a str,
    pub cursor: usize,
    pub ghost_index: usize,
    pub dropdown: Option<&'a DropdownState>,
}

#[derive(Clone, Default)]
pub struct StreamRenderState {
    pub output: String,
    pub metrics: StreamMetrics,
    pub tool_running_since: Option<std::time::Instant>,
    pub native_tool_calls: Vec<orangu::llm::ToolCall>,
}

#[derive(Debug, Default)]
pub struct EscapeCancelState {
    pub last_escape: Option<Instant>,
}

impl EscapeCancelState {
    pub fn reset(&mut self) {
        self.last_escape = None;
    }

    pub fn handle_escape(&mut self, now: Instant) -> bool {
        if let Some(last_escape) = self.last_escape
            && now.duration_since(last_escape) <= ESC_CANCEL_TIMEOUT
        {
            self.last_escape = None;
            return true;
        }

        self.last_escape = Some(now);
        false
    }
}

pub enum InterruptAction {
    Continue,
    Exit,
}

#[derive(Debug, Default)]
pub struct InterruptState {
    pub last_interrupt: Option<Instant>,
}

impl InterruptState {
    pub fn reset(&mut self) {
        self.last_interrupt = None;
    }

    pub fn handle_interrupt(&mut self, now: Instant) -> InterruptAction {
        if let Some(last_interrupt) = self.last_interrupt
            && now.duration_since(last_interrupt) <= CTRL_C_EXIT_TIMEOUT
        {
            self.last_interrupt = None;
            return InterruptAction::Exit;
        }

        self.last_interrupt = Some(now);
        InterruptAction::Continue
    }
}

#[derive(Clone, Copy)]
pub struct InputContext<'a> {
    pub history: &'a [String],
    pub workspace: &'a Path,
    pub server_names: &'a [String],
    pub available_models: &'a [String],
    pub render: RenderContext<'a>,
    pub skills: &'a orangu::skills::SkillRegistry,
}

pub struct WaitContext<'a> {
    pub render: RenderContext<'a>,
    pub history: &'a mut Vec<String>,
    pub history_path: &'a Path,
    pub server_names: &'a [String],
    pub available_models: &'a [String],
    pub interrupt_state: &'a mut InterruptState,
    pub output_state: &'a mut OutputState,
    pub input_state: &'a mut InputState,
    pub pending_commands: &'a mut VecDeque<crate::wait::PendingCommand>,
    pub thinking_quote: Option<&'static str>,
    pub viewport: &'a mut ViewportState,
    pub skills: &'a orangu::skills::SkillRegistry,
    /// When a workspace-switch key (Alt+,/./Insert/Delete) is pressed during a
    /// wait (streaming, blocking command), the action is stored here instead of
    /// dropped, so `main` can apply it as soon as the wait returns.
    pub deferred_tab: &'a mut Option<crate::workspace_tab::TabAction>,
    pub parked_tabs: &'a [crate::workspace_tab::WorkspaceTab],
}

#[derive(Default)]
pub struct OutputState {
    pub transcript: Vec<TranscriptLine>,
    pub render_epoch: usize,
    pub scroll_offset: usize,
}

impl OutputState {
    pub fn lines(&self) -> &[TranscriptLine] {
        &self.transcript
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    pub fn render_epoch(&self) -> usize {
        self.render_epoch
    }

    pub fn clear(&mut self) {
        self.transcript.clear();
        self.render_epoch = self.render_epoch.wrapping_add(1);
        self.scroll_offset = 0;
    }

    pub fn toggle_collapse(&mut self, id: usize) -> bool {
        for line in &mut self.transcript {
            if let TranscriptLine::Collapsible {
                id: line_id,
                expanded,
                ..
            } = line
                && *line_id == id
            {
                *expanded = !*expanded;
                self.render_epoch = self.render_epoch.wrapping_add(1);
                return true;
            }
        }
        false
    }

    pub fn collapsible_id_at(
        &self,
        click_y: usize,
        available_output_rows: usize,
        inner_width: usize,
        has_status_line: bool,
    ) -> Option<usize> {
        let mut visual_lines = Vec::new();
        let mut user_inputs = Vec::new();
        let theme = orangu::tui::theme::Theme::current();
        for line in &self.transcript {
            let start = visual_lines.len();
            let rendered = orangu::tui::render_transcript_line_multi(line, inner_width, &theme);
            let id = match line {
                TranscriptLine::Collapsible { id, .. } => Some(*id),
                _ => None,
            };
            visual_lines.extend(std::iter::repeat_n(id, rendered.len()));
            if matches!(line, TranscriptLine::UserInput(_)) {
                user_inputs.push((start, visual_lines[start..].to_vec()));
            }
        }

        // Keep this mapping identical to `draw_screen`: while at the bottom,
        // a status row consumes one transcript row and the latest prior user
        // prompt is pinned at the top of the visible view.
        let max_scroll_offset = if has_status_line {
            visual_lines
                .len()
                .saturating_sub(available_output_rows.saturating_sub(1))
        } else {
            visual_lines.len().saturating_sub(available_output_rows)
        };
        let scroll_offset = self.scroll_offset.min(max_scroll_offset);
        let text_rows = if has_status_line && scroll_offset == 0 {
            available_output_rows.saturating_sub(1)
        } else {
            available_output_rows
        };
        let visible_end = visual_lines.len().saturating_sub(scroll_offset);
        let visible_start = visible_end.saturating_sub(text_rows);
        let mut visible_lines = visual_lines[visible_start..visible_end].to_vec();

        if let Some(sticky_idx) = user_inputs
            .iter()
            .rposition(|(start, _)| *start < visible_start)
        {
            let (_, sticky_lines) = &user_inputs[sticky_idx];
            let next_start = user_inputs
                .get(sticky_idx + 1)
                .map(|(start, _)| *start)
                .unwrap_or(usize::MAX);
            let draw_count = sticky_lines
                .len()
                .min(next_start.saturating_sub(visible_start))
                .min(visible_lines.len());
            visible_lines[..draw_count]
                .clone_from_slice(&sticky_lines[sticky_lines.len() - draw_count..]);
        }

        visible_lines.get(click_y).copied().flatten()
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn push_text(&mut self, text: &str) {
        self.push_lines(
            text.lines()
                .map(|line| TranscriptLine::Plain(line.to_owned())),
        );
    }

    pub fn push_wide(&mut self, text: &str) {
        self.push_lines(
            text.lines()
                .map(|line| TranscriptLine::Wide(line.to_owned())),
        );
    }

    pub fn push_input(&mut self, text: &str) {
        self.push_lines(std::iter::once(TranscriptLine::UserInput(text.to_owned())));
    }

    pub fn push_lines<I>(&mut self, lines: I)
    where
        I: Iterator<Item = TranscriptLine>,
    {
        let collected = lines.collect::<Vec<_>>();
        let added_lines = collected.len();
        self.transcript.extend(collected);
        if added_lines > 0 {
            self.render_epoch = self.render_epoch.wrapping_add(1);
        }
        if self.scroll_offset > 0 {
            self.scroll_offset = self.scroll_offset.saturating_add(added_lines);
        }

        let excess = self.transcript.len().saturating_sub(TRANSCRIPT_MAX_LINES);
        if excess > 0 {
            self.transcript.drain(0..excess);
            self.scroll_offset = self.scroll_offset.saturating_sub(excess);
        }
    }

    pub fn push_markdown(&mut self, text: &str) {
        self.push_parsed(text, false);
    }

    pub fn push_parsed(&mut self, text: &str, is_streaming: bool) {
        let mut remaining = text;
        let mut next_id = self.transcript.len() + 1000;

        while !remaining.is_empty() {
            let think_pos = remaining.find("<think>");
            let tool_pos = remaining.find("<tool_call");

            let (tag_type, start_idx) = match (think_pos, tool_pos) {
                (Some(t1), Some(t2)) if t1 < t2 => ("think", t1),
                (Some(_t1), Some(t2)) => ("tool_call", t2),
                (Some(t1), None) => ("think", t1),
                (None, Some(t2)) => ("tool_call", t2),
                (None, None) => {
                    if !remaining.trim().is_empty() {
                        for chunk in crate::render::parse_markdown_chunks(remaining) {
                            match chunk {
                                crate::render::MarkdownChunk::Text(t) => self.push_text(&t),
                                crate::render::MarkdownChunk::Code { language, content } => {
                                    let ansi_lines = crate::render::render_syntax_highlighted_code(
                                        language.as_deref(),
                                        &content,
                                    );
                                    self.push_lines(std::iter::once(
                                        orangu::tui::TranscriptLine::CodeBlock {
                                            language,
                                            ansi_lines,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    break;
                }
            };

            let before = &remaining[..start_idx];
            if !before.trim().is_empty() {
                for chunk in crate::render::parse_markdown_chunks(before) {
                    match chunk {
                        crate::render::MarkdownChunk::Text(t) => self.push_text(&t),
                        crate::render::MarkdownChunk::Code { language, content } => {
                            let ansi_lines = crate::render::render_syntax_highlighted_code(
                                language.as_deref(),
                                &content,
                            );
                            self.push_lines(std::iter::once(
                                orangu::tui::TranscriptLine::CodeBlock {
                                    language,
                                    ansi_lines,
                                },
                            ));
                        }
                    }
                }
            }

            let title: String;
            let tag_end: usize;
            let end_tag: &str;

            if tag_type == "think" {
                title = "Thought".to_string();
                tag_end = start_idx + "<think>".len();
                end_tag = "</think>";
            } else {
                let tag_str = &remaining[start_idx..];
                if let Some(close_bracket) = tag_str.find('>') {
                    let full_open_tag = &tag_str[..=close_bracket];
                    if let Some(name_start) = full_open_tag.find("name=\"") {
                        let name_part = &full_open_tag[name_start + 6..];
                        if let Some(name_end) = name_part.find('"') {
                            title = format!("Tool Call: {}", &name_part[..name_end]);
                        } else {
                            title = "Tool Call".to_string();
                        }
                    } else {
                        title = "Tool Call".to_string();
                    }
                    tag_end = start_idx + close_bracket + 1;
                } else {
                    tag_end = start_idx + "<tool_call".len();
                    title = "Tool Call".to_string();
                }
                end_tag = "</tool_call>";
            }

            remaining = &remaining[tag_end..];

            let (content, after) = if let Some(end_idx) = remaining.find(end_tag) {
                (&remaining[..end_idx], &remaining[end_idx + end_tag.len()..])
            } else {
                (remaining, "")
            };

            // During streaming, keep only the newest thinking/tool block open.
            // Earlier blocks are complete context and should fold away when the
            // model starts the next block, matching the transcript behavior of
            // the finalized response.
            if is_streaming {
                for line in &mut self.transcript {
                    if let TranscriptLine::Collapsible { expanded, .. } = line {
                        *expanded = false;
                    }
                }
            }
            self.push_lines(std::iter::once(TranscriptLine::Collapsible {
                id: next_id,
                title,
                content: content.trim().to_string(),
                expanded: is_streaming,
            }));
            next_id += 1;
            remaining = after;
        }
    }

    pub fn max_content_width(&self) -> usize {
        self.transcript
            .iter()
            .filter(|line| !matches!(line, TranscriptLine::UserInput(_)))
            .map(|line| visible_line_width(line.as_str()))
            .max()
            .unwrap_or(0)
    }

    pub fn page_up(&mut self, rows: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(page_scroll_lines(rows));
    }

    pub fn page_down(&mut self, rows: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(page_scroll_lines(rows));
    }

    pub fn scroll_up(&mut self, rows: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(rows.max(1));
    }

    pub fn scroll_down(&mut self, rows: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(rows.max(1));
    }

    pub fn line_up(&mut self) {
        self.scroll_up(1);
    }

    pub fn line_down(&mut self) {
        self.scroll_down(1);
    }
}

#[derive(Clone, Default, Debug)]
pub struct DropdownState {
    pub candidates: Vec<(String, String)>,
    pub selected_index: usize,
    pub replacement_start: usize,
}

impl DropdownState {
    /// Move selection up, wrapping from first item to last.
    pub fn select_previous(&mut self) {
        let len = self.candidates.len();
        if len == 0 {
            return;
        }
        self.selected_index = if self.selected_index == 0 {
            len - 1
        } else {
            self.selected_index - 1
        };
    }

    /// Move selection down, wrapping from last item to first.
    pub fn select_next(&mut self) {
        let len = self.candidates.len();
        if len == 0 {
            return;
        }
        self.selected_index = (self.selected_index + 1) % len;
    }
}

#[derive(Default)]
pub struct InputState {
    pub buffer: String,
    pub cursor: usize,
    pub completion: Option<CompletionState>,
    pub ghost_index: usize,
    pub history_index: Option<usize>,
    pub history_draft: String,
    pub dropdown: Option<DropdownState>,
    pub last_mouse_click: Option<(std::time::Instant, u16, u16)>,
}

impl InputState {
    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.completion = None;
        self.ghost_index = 0;
        self.history_index = None;
        self.history_draft.clear();
        self.dropdown = None;
    }

    pub fn set_buffer(&mut self, buffer: String) {
        self.buffer = buffer;
        self.cursor = self.buffer.len();
        self.completion = None;
        self.ghost_index = 0;
        self.dropdown = None;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn insert_str(&mut self, text: &str) {
        self.buffer.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn backspace(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(previous..self.cursor);
            self.cursor = previous;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(self.cursor..next);
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn move_left(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.cursor = previous;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.cursor = next;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.buffer.len();
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn kill_to_end(&mut self) {
        self.buffer.truncate(self.cursor);
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn kill_to_start(&mut self) {
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn delete_prev_word(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let mut start = self.cursor;
        while let Some(previous) = previous_boundary(&self.buffer, start) {
            if !self.buffer[previous..start]
                .chars()
                .all(char::is_whitespace)
            {
                start = previous;
                break;
            }
            start = previous;
            if start == 0 {
                break;
            }
        }

        while let Some(previous) = previous_boundary(&self.buffer, start) {
            if self.buffer[previous..start]
                .chars()
                .all(char::is_whitespace)
            {
                break;
            }
            start = previous;
            if start == 0 {
                break;
            }
        }

        self.buffer.drain(start..self.cursor);
        self.cursor = start;
        self.completion = None;
        self.ghost_index = 0;
    }

    pub fn delete_backward_readline_word(&mut self) {
        let start = readline_word_start(&self.buffer, self.cursor);
        if start < self.cursor {
            self.buffer.drain(start..self.cursor);
            self.cursor = start;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn delete_forward_readline_word(&mut self) {
        let end = readline_word_end(&self.buffer, self.cursor);
        if end > self.cursor {
            self.buffer.drain(self.cursor..end);
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn move_backward_readline_word(&mut self) {
        let start = readline_word_start(&self.buffer, self.cursor);
        if start != self.cursor {
            self.cursor = start;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn move_forward_readline_word(&mut self) {
        let end = readline_word_end(&self.buffer, self.cursor);
        if end != self.cursor {
            self.cursor = end;
            self.completion = None;
            self.ghost_index = 0;
        }
    }

    pub fn update_dropdown(&mut self, skills: &orangu::skills::SkillRegistry, enabled: bool) {
        if !enabled {
            self.dropdown = None;
            return;
        }

        let is_theme_arg = self.buffer.starts_with("/theme ");
        let no_space = !self.buffer.contains(' ');

        if self.buffer.starts_with('/') && (no_space || is_theme_arg) {
            let candidates = if is_theme_arg {
                let prefix = &self.buffer["/theme ".len()..];
                orangu::tui::Theme::available_session_theme_names()
                    .into_iter()
                    .filter(|t| t.starts_with(prefix))
                    .map(|t| (t.to_string(), "".to_string()))
                    .collect()
            } else {
                super::completion::slash_command_dropdown_candidates(&self.buffer, skills)
            };

            if !candidates.is_empty() {
                let replacement_start = if is_theme_arg { "/theme ".len() } else { 0 };
                let mut new_dropdown = DropdownState {
                    candidates,
                    selected_index: 0,
                    replacement_start,
                };
                if let Some(old) = &self.dropdown {
                    // Try to preserve selection if the currently selected command is still in the list
                    if old.selected_index < old.candidates.len() {
                        let selected_cmd = &old.candidates[old.selected_index].0;
                        if let Some(pos) = new_dropdown
                            .candidates
                            .iter()
                            .position(|(c, _)| c == selected_cmd)
                        {
                            new_dropdown.selected_index = pos;
                        }
                    }
                }
                self.dropdown = Some(new_dropdown);
            } else {
                self.dropdown = None;
            }
        } else {
            self.dropdown = None;
        }
    }
}

pub fn idle_status_refresh_timeout(refresh_deadline: Instant, now: Instant) -> Duration {
    refresh_deadline
        .checked_duration_since(now)
        .unwrap_or(Duration::ZERO)
}

#[allow(clippy::too_many_arguments)]
pub fn read_input(
    input_state: &mut InputState,
    interrupt_state: &mut InterruptState,
    output_state: &mut OutputState,
    pending_count: usize,
    viewport: &mut ViewportState,
    input_context: InputContext<'_>,
    mut print_screen_fn: impl FnMut(RenderContext<'_>, ScreenState<'_>),
    max_idle: Duration,
) -> anyhow::Result<InputResult> {
    use crossterm::event;

    let refresh_deadline = Instant::now() + max_idle;

    loop {
        let timeout = idle_status_refresh_timeout(refresh_deadline, Instant::now());
        if !event::poll(timeout)? {
            return Ok(InputResult::Refresh);
        }

        let result = handle_input_event_with_status(
            event::read()?,
            input_state,
            interrupt_state,
            output_state,
            viewport,
            input_context,
            pending_count,
        );

        if let Some(outcome) = result.outcome {
            return Ok(outcome);
        }

        if Instant::now() >= refresh_deadline {
            return Ok(InputResult::Refresh);
        }

        if result.redraw {
            let render = RenderContext {
                virtual_width: viewport.virtual_width,
                actual_width: viewport.actual_width,
                actual_height: viewport.actual_height,
                x_offset: viewport.x_offset,
                ..input_context.render
            };
            print_screen_fn(
                render,
                ScreenState {
                    transcript: output_state.lines(),
                    transcript_epoch: output_state.render_epoch(),
                    scroll_offset: output_state.scroll_offset(),
                    left_status: None,
                    pending_count: 0,
                    pending_lines: &[],
                    input: input_state.as_str(),
                    cursor: input_state.cursor(),
                    ghost_index: input_state.ghost_index,
                    dropdown: input_state.dropdown.as_ref(),
                },
            );
            std::io::stdout().flush()?;
        }
    }
}

#[cfg(test)]
pub fn handle_input_event(
    event: Event,
    input_state: &mut InputState,
    interrupt_state: &mut InterruptState,
    output_state: &mut OutputState,
    viewport: &mut ViewportState,
    input_context: InputContext<'_>,
) -> InputEventResult {
    handle_input_event_with_status(
        event,
        input_state,
        interrupt_state,
        output_state,
        viewport,
        input_context,
        0,
    )
}

pub fn handle_input_event_with_status(
    event: Event,
    input_state: &mut InputState,
    interrupt_state: &mut InterruptState,
    output_state: &mut OutputState,
    viewport: &mut ViewportState,
    input_context: InputContext<'_>,
    pending_count: usize,
) -> InputEventResult {
    let mut redraw = false;

    match event {
        Event::Resize(w, h) => {
            viewport.on_resize(usize::from(w), usize::from(h));
            viewport.clamp_to_content(
                output_state.max_content_width(),
                current_output_width(input_context.render, input_state.as_str(), output_state),
            );
            redraw = true;
        }
        Event::Paste(text) => {
            interrupt_state.reset();
            output_state.reset_scroll();
            input_state.insert_str(&text);
            redraw = true;
        }
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind,
            ..
        }) if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat => {
            match (code, modifiers) {
                (KeyCode::Left, modifiers)
                    if modifiers.contains(KeyModifiers::CONTROL)
                        && !modifiers.contains(KeyModifiers::ALT) =>
                {
                    interrupt_state.reset();
                    input_state.move_backward_readline_word();
                    redraw = true;
                }
                (KeyCode::Right, modifiers)
                    if modifiers.contains(KeyModifiers::CONTROL)
                        && !modifiers.contains(KeyModifiers::ALT) =>
                {
                    interrupt_state.reset();
                    input_state.move_forward_readline_word();
                    redraw = true;
                }
                (KeyCode::Backspace, modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    interrupt_state.reset();
                    input_state.delete_backward_readline_word();
                    redraw = true;
                }
                (KeyCode::Char(ch), modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL)
                        && ch.eq_ignore_ascii_case(&'d') =>
                {
                    interrupt_state.reset();
                    input_state.delete_forward_readline_word();
                    redraw = true;
                }
                (KeyCode::Char(','), modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::WorkspacePrevious),
                    };
                }
                (KeyCode::Char('.'), modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::WorkspaceNext),
                    };
                }
                (KeyCode::Insert, modifiers) if modifiers.contains(KeyModifiers::ALT) => {
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::WorkspaceNew),
                    };
                }
                (KeyCode::Delete, modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::WorkspaceClose),
                    };
                }
                (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                    match interrupt_state.handle_interrupt(Instant::now()) {
                        InterruptAction::Continue => {
                            output_state.push_text(CTRL_C_EXIT_MESSAGE);
                            output_state.reset_scroll();
                            input_state.clear();
                            return InputEventResult {
                                redraw: true,
                                outcome: Some(InputResult::Submitted(String::new())),
                            };
                        }
                        InterruptAction::Exit => {
                            return InputEventResult {
                                redraw: false,
                                outcome: Some(InputResult::Quit),
                            };
                        }
                    }
                }
                (KeyCode::Char('d'), KeyModifiers::CONTROL) if input_state.as_str().is_empty() => {
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::Quit),
                    };
                }
                (KeyCode::Enter, KeyModifiers::NONE) => {
                    interrupt_state.reset();
                    if let Some(dropdown) = &input_state.dropdown
                        && dropdown.selected_index < dropdown.candidates.len()
                    {
                        let cmd = &dropdown.candidates[dropdown.selected_index].0;
                        input_state.buffer.truncate(dropdown.replacement_start);
                        input_state.buffer.push_str(cmd);
                        input_state.buffer.push(' ');
                        input_state.cursor = input_state.buffer.len();
                        input_state.dropdown = None;
                        redraw = true;
                    } else {
                        let input = input_state.buffer.clone();
                        input_state.clear();
                        return InputEventResult {
                            redraw: false,
                            outcome: Some(InputResult::Submitted(input)),
                        };
                    }
                }
                (KeyCode::Backspace, _) => {
                    interrupt_state.reset();
                    input_state.backspace();
                    redraw = true;
                }
                (KeyCode::Delete, _) | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.delete();
                    redraw = true;
                }
                (KeyCode::Left, modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    viewport.pan_left();
                    redraw = true;
                }
                (KeyCode::Right, modifiers)
                    if modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    viewport.pan_right(
                        output_state.max_content_width(),
                        current_output_width(
                            input_context.render,
                            input_state.as_str(),
                            output_state,
                        ),
                    );
                    redraw = true;
                }
                (KeyCode::Left, _) => {
                    interrupt_state.reset();
                    input_state.move_left();
                    redraw = true;
                }
                (KeyCode::Right, _) => {
                    interrupt_state.reset();
                    input_state.move_right();
                    redraw = true;
                }
                (KeyCode::Home, _) | (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.move_home();
                    redraw = true;
                }
                (KeyCode::End, _) | (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.move_end();
                    redraw = true;
                }
                (KeyCode::Char('k'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.kill_to_end();
                    redraw = true;
                }
                (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.kill_to_start();
                    redraw = true;
                }
                (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
                    interrupt_state.reset();
                    input_state.delete_prev_word();
                    redraw = true;
                }
                (KeyCode::Up, modifiers) if modifiers.contains(KeyModifiers::ALT) => {
                    interrupt_state.reset();
                    output_state.line_up();
                    redraw = true;
                }
                (KeyCode::Down, modifiers) if modifiers.contains(KeyModifiers::ALT) => {
                    interrupt_state.reset();
                    output_state.line_down();
                    redraw = true;
                }
                (KeyCode::Up, _) => {
                    interrupt_state.reset();
                    if let Some(dropdown) = &mut input_state.dropdown {
                        dropdown.select_previous();
                        redraw = true;
                    } else {
                        history_previous(input_state, input_context.history);
                        redraw = true;
                    }
                }
                (KeyCode::Down, _) => {
                    interrupt_state.reset();
                    if let Some(dropdown) = &mut input_state.dropdown {
                        dropdown.select_next();
                        redraw = true;
                    } else {
                        history_next(input_state, input_context.history);
                        redraw = true;
                    }
                }
                (KeyCode::Tab, _) => {
                    interrupt_state.reset();
                    if let Some(dropdown) = &input_state.dropdown {
                        if dropdown.selected_index < dropdown.candidates.len() {
                            let cmd = &dropdown.candidates[dropdown.selected_index].0;
                            input_state.buffer.truncate(dropdown.replacement_start);
                            input_state.buffer.push_str(cmd);
                            input_state.buffer.push(' ');
                            input_state.cursor = input_state.buffer.len();
                            input_state.dropdown = None;
                            redraw = true;
                        }
                    } else {
                        apply_completion(
                            input_state,
                            input_context.workspace,
                            input_context.server_names,
                            input_context.available_models,
                            input_context.skills,
                        );
                        redraw = true;
                    }
                }
                (KeyCode::BackTab, _) => {
                    interrupt_state.reset();
                    cycle_ghost_suggestion(input_state);
                    redraw = true;
                }
                (KeyCode::PageUp, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                    interrupt_state.reset();
                    let layout = orangu::tui::main_screen_layout(
                        input_context.render.actual_width,
                        input_context.render.actual_height,
                        input_context.render.prompt_branch,
                        input_state.as_str(),
                        input_context.render.tab_bar,
                        input_context.render.tab_statuses,
                        true,
                    );
                    output_state.page_up(layout.output_area.height as usize);
                    redraw = true;
                }
                (KeyCode::PageDown, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                    interrupt_state.reset();
                    let layout = orangu::tui::main_screen_layout(
                        input_context.render.actual_width,
                        input_context.render.actual_height,
                        input_context.render.prompt_branch,
                        input_state.as_str(),
                        input_context.render.tab_bar,
                        input_context.render.tab_statuses,
                        true,
                    );
                    output_state.page_down(layout.output_area.height as usize);
                    redraw = true;
                }
                (KeyCode::Char(ch), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                    interrupt_state.reset();
                    output_state.reset_scroll();
                    input_state.insert_char(ch);
                    redraw = true;
                }
                (KeyCode::Esc, _) => {
                    output_state.reset_scroll();
                    redraw = true;
                }
                _ => {}
            }
        }
        Event::Mouse(crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
            row,
            ..
        }) => {
            let now = std::time::Instant::now();
            let mut is_double_click = false;

            if let Some((last_time, last_row, _last_col)) = input_state.last_mouse_click {
                if last_row == row
                    && now.duration_since(last_time) < std::time::Duration::from_millis(500)
                {
                    is_double_click = true;
                    input_state.last_mouse_click = None;
                } else {
                    input_state.last_mouse_click = Some((now, row, 0));
                }
            } else {
                input_state.last_mouse_click = Some((now, row, 0));
            }

            if is_double_click {
                let layout = orangu::tui::main_screen_layout(
                    input_context.render.actual_width,
                    input_context.render.actual_height,
                    input_context.render.prompt_branch,
                    input_state.as_str(),
                    input_context.render.tab_bar,
                    input_context.render.tab_statuses,
                    !output_state.lines().is_empty(),
                );

                let output_y = usize::from(layout.output_area.y);
                let output_height = usize::from(layout.output_area.height);
                if (row as usize) >= output_y && (row as usize) < output_y + output_height {
                    let click_y = (row as usize) - output_y;

                    if let Some(id) = output_state.collapsible_id_at(
                        click_y,
                        output_height,
                        usize::from(layout.output_area.width),
                        pending_count > 0,
                    ) && output_state.toggle_collapse(id)
                    {
                        redraw = true;
                    }
                }
            }
        }
        Event::Mouse(crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::ScrollUp,
            ..
        }) => {
            let layout = orangu::tui::main_screen_layout(
                input_context.render.actual_width,
                input_context.render.actual_height,
                input_context.render.prompt_branch,
                input_state.as_str(),
                input_context.render.tab_bar,
                input_context.render.tab_statuses,
                !output_state.lines().is_empty(),
            );
            output_state.scroll_up(wheel_scroll_lines(layout.output_area.height as usize));
            redraw = true;
        }
        Event::Mouse(crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::ScrollDown,
            ..
        }) => {
            let layout = orangu::tui::main_screen_layout(
                input_context.render.actual_width,
                input_context.render.actual_height,
                input_context.render.prompt_branch,
                input_state.as_str(),
                input_context.render.tab_bar,
                input_context.render.tab_statuses,
                !output_state.lines().is_empty(),
            );
            output_state.scroll_down(wheel_scroll_lines(layout.output_area.height as usize));
            redraw = true;
        }
        _ => {}
    }

    if redraw {
        input_state.update_dropdown(input_context.skills, input_context.render.drop_down);
        sync_theme_preview(input_state);
    }

    InputEventResult {
        redraw,
        outcome: None,
    }
}

/// Live-preview the theme under the `/theme` dropdown (or filled argument).
/// Preview is temporary until `/theme <name>` is submitted; leaving the command
/// restores the committed theme.
fn sync_theme_preview(input_state: &InputState) {
    if let Some(dropdown) = &input_state.dropdown
        && input_state.buffer.starts_with("/theme ")
        && let Some((name, _)) = dropdown.candidates.get(dropdown.selected_index)
    {
        let _ = orangu::tui::Theme::preview_named(name);
        return;
    }

    if let Some(rest) = input_state.buffer.strip_prefix("/theme ") {
        let name = rest.trim();
        if !name.is_empty() {
            let _ = orangu::tui::Theme::preview_named(name);
            return;
        }
    }

    orangu::tui::Theme::clear_preview();
}

fn current_output_width(
    render: RenderContext<'_>,
    input: &str,
    output_state: &OutputState,
) -> usize {
    orangu::tui::output_view_width(
        VERSION,
        render.current_model,
        render.endpoint,
        render.workspace,
        render.prompt_branch,
        render.header_status,
        input,
        render.actual_width,
        render.actual_height,
        render.tab_bar,
        render.tab_statuses,
        !output_state.lines().is_empty(),
    )
}

pub fn history_previous(input_state: &mut InputState, history: &[String]) {
    if history.is_empty() {
        return;
    }

    let new_index = match input_state.history_index {
        Some(0) => 0,
        Some(index) => index.saturating_sub(1),
        None => {
            input_state.history_draft = input_state.buffer.clone();
            history.len() - 1
        }
    };

    input_state.history_index = Some(new_index);
    input_state.set_buffer(history[new_index].clone());
}

pub fn history_next(input_state: &mut InputState, history: &[String]) {
    let Some(index) = input_state.history_index else {
        return;
    };

    if index + 1 >= history.len() {
        input_state.history_index = None;
        let draft = std::mem::take(&mut input_state.history_draft);
        input_state.set_buffer(draft);
        return;
    }

    let new_index = index + 1;
    input_state.history_index = Some(new_index);
    input_state.set_buffer(history[new_index].clone());
}

/// Advance the inline natural-language ghost preview to the next candidate
/// (Shift+Tab). For input like `c`, this cycles `current model` -> `code review` ->
/// `checkout ` -> ... -> back to `current model`. Tab then accepts whatever is shown.
/// No-op when the cursor is not at the end of the line, or when there is nothing
/// (or only one thing) to cycle through.
pub fn cycle_ghost_suggestion(input_state: &mut InputState) {
    if input_state.cursor() != input_state.buffer.len() {
        return;
    }
    let count = natural_language_ghost_candidates(input_state.as_str()).len();
    if count > 1 {
        input_state.ghost_index = (input_state.ghost_index + 1) % count;
    }
}

pub fn apply_completion(
    input_state: &mut InputState,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
    skills: &orangu::skills::SkillRegistry,
) {
    if let Some(state) = input_state.completion.as_mut()
        && !state.candidates.is_empty()
    {
        state.index = (state.index + 1) % state.candidates.len();
        let start = state.start;
        let end = state.end;
        let original = state.original.clone();
        let candidate = state.candidates[state.index].clone();
        apply_completion_candidate(input_state, start, end, &original, &candidate);
        return;
    }

    // Accept the inline natural-language ghost the user is seeing (e.g. "c" ->
    // "current model") before falling back to generic file completion, so Tab fills in
    // the hint that is actually rendered rather than a same-prefixed filename.
    // The cycle position chosen with Shift+Tab decides which candidate is taken.
    // Only when the cursor is at the end of the line, matching where the ghost
    // is drawn. Slash commands return `None` here and use the cycling path below.
    //
    // Tab accepts one word at a time so a multi-word binding fills in
    // progressively: `pus` -> `push ` (with `force` then previewed), then
    // `push ` -> `push force`. The remaining words stay rendered as the ghost.
    if input_state.cursor() == input_state.buffer.len()
        && let Some(suffix) =
            natural_language_ghost_suffix_at(input_state.as_str(), input_state.ghost_index)
    {
        let word = first_ghost_word(suffix);
        let end = input_state.buffer.len();
        let original = input_state.buffer.clone();
        apply_completion_candidate(input_state, end, end, &original, word);
        return;
    }

    if let Some((start, end, candidates)) = completion_candidates(
        input_state.as_str(),
        input_state.cursor(),
        workspace,
        server_names,
        available_models,
        skills,
    ) && !candidates.is_empty()
    {
        let original = input_state.buffer.clone();
        let candidate = candidates[0].clone();
        apply_completion_candidate(input_state, start, end, &original, &candidate);
        input_state.completion = Some(CompletionState {
            start,
            end,
            original,
            candidates,
            index: 0,
        });
    }
}

pub fn apply_completion_candidate(
    input_state: &mut InputState,
    start: usize,
    end: usize,
    original: &str,
    candidate: &str,
) {
    let mut buffer = String::new();
    buffer.push_str(&original[..start]);
    buffer.push_str(candidate);
    buffer.push_str(&original[end..]);
    input_state.buffer = buffer;
    input_state.cursor = start + candidate.len();
}

fn previous_boundary(input: &str, cursor: usize) -> Option<usize> {
    input[..cursor.min(input.len())]
        .char_indices()
        .last()
        .map(|(index, _)| index)
}

fn next_boundary(input: &str, cursor: usize) -> Option<usize> {
    let cursor = cursor.min(input.len());
    input[cursor..]
        .char_indices()
        .nth(1)
        .map(|(index, _)| cursor + index)
        .or_else(|| (cursor < input.len()).then_some(input.len()))
}

// Underscore is not a word char: Alt+Backspace/Alt+D stop at `_` boundaries in identifiers and paths.
fn is_readline_word_char(ch: char) -> bool {
    ch.is_alphanumeric()
}

fn readline_word_start(buffer: &str, cursor: usize) -> usize {
    let mut pos = cursor;
    while let Some(prev) = previous_boundary(buffer, pos) {
        if buffer[prev..pos]
            .chars()
            .all(|ch| !is_readline_word_char(ch))
        {
            pos = prev;
        } else {
            break;
        }
    }
    while let Some(prev) = previous_boundary(buffer, pos) {
        if buffer[prev..pos].chars().all(is_readline_word_char) {
            pos = prev;
        } else {
            break;
        }
    }
    pos
}

fn readline_word_end(buffer: &str, cursor: usize) -> usize {
    let mut pos = cursor;
    while let Some(next) = next_boundary(buffer, pos) {
        if buffer[pos..next]
            .chars()
            .all(|ch| !is_readline_word_char(ch))
        {
            pos = next;
        } else {
            break;
        }
    }
    while let Some(next) = next_boundary(buffer, pos) {
        if buffer[pos..next].chars().all(is_readline_word_char) {
            pos = next;
        } else {
            break;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use orangu::tui::{TabStatus, WorkspaceTabsView};
    use orangu::workspaces::WorkspacePlacement;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    #[test]
    fn output_state_keeps_last_ten_thousand_lines() {
        let mut output_state = OutputState::default();
        for index in 0..10_005 {
            output_state.push_text(&format!("line {index}"));
        }

        assert_eq!(output_state.lines().len(), 10_000);
        assert_eq!(
            output_state.lines().first().map(TranscriptLine::as_str),
            Some("line 5")
        );
        assert_eq!(
            output_state.lines().last().map(TranscriptLine::as_str),
            Some("line 10004")
        );
    }

    #[test]
    fn output_state_styles_echoed_user_input() {
        let mut output_state = OutputState::default();

        output_state.push_input("> show README.md");
        output_state.push_text("plain output");

        assert!(
            matches!(output_state.lines().first(), Some(TranscriptLine::UserInput(s)) if s == "> show README.md")
        );
        assert!(
            matches!(output_state.lines().get(1), Some(TranscriptLine::Plain(s)) if s == "plain output")
        );
    }

    #[test]
    fn streaming_collapses_previous_thinking_or_tool_blocks() {
        let mut output_state = OutputState::default();

        output_state.push_parsed(
            "<think>first thought</think>between<tool_call name=\"read\">file</tool_call>",
            true,
        );

        let collapsibles = output_state
            .lines()
            .iter()
            .filter_map(|line| match line {
                TranscriptLine::Collapsible {
                    title, expanded, ..
                } => Some((title.as_str(), *expanded)),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            collapsibles,
            vec![("Thought", false), ("Tool Call: read", true)]
        );
    }

    #[test]
    fn finalized_output_starts_collapsed() {
        let mut output_state = OutputState::default();
        output_state.push_markdown("<think>done</think><tool_call name=\"read\">file</tool_call>");

        assert!(output_state.lines().iter().all(|line| {
            matches!(
                line,
                TranscriptLine::Collapsible {
                    expanded: false,
                    ..
                }
            )
        }));
    }

    #[test]
    fn alt_backspace_deletes_previous_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Backspace,
                KeyModifiers::ALT,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.as_str(), "src/tui.");
        assert_eq!(input_state.cursor(), "src/tui.".len());
    }

    #[test]
    fn alt_d_deletes_next_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        input_state.move_home();
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('d'),
                KeyModifiers::ALT,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.as_str(), "/tui.rs");
        assert_eq!(input_state.cursor(), 0);
    }

    #[test]
    fn ctrl_left_moves_to_previous_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Left,
                KeyModifiers::CONTROL,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.cursor(), "src/tui.".len());
    }

    #[test]
    fn ctrl_right_moves_to_next_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        input_state.move_home();
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Right,
                KeyModifiers::CONTROL,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.cursor(), 3);
    }

    #[test]
    fn ctrl_w_keeps_whitespace_based_word_deletion() {
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());

        input_state.delete_prev_word();

        assert_eq!(input_state.as_str(), "");
        assert_eq!(input_state.cursor(), 0);
    }

    #[test]
    fn transcript_pan_reaches_right_edge_of_narrowed_output_pane() {
        let visible_width = orangu::tui::output_view_width(
            VERSION,
            "default",
            "http://localhost:11434/v1",
            std::path::Path::new("/tmp"),
            None,
            HeaderStatus {
                workspace_ok: true,
                server_ok: orangu::tui::ConnStatus::Ok,
                model_ok: orangu::tui::ConnStatus::Ok,
                is_coordinator: false,
            },
            "",
            80,
            24,
            Some(WorkspaceTabsView {
                count: 12,
                active: 0,
                placement: WorkspacePlacement::Left,
            }),
            &[TabStatus::Valid; 12],
            true,
        );
        let mut viewport = ViewportState::new(160, 80, 24);

        for _ in 0..200 {
            viewport.pan_right(120, visible_width);
        }

        assert_eq!(viewport.x_offset, 120usize.saturating_sub(visible_width));
    }

    #[test]
    fn resize_clamps_horizontal_pan_to_current_output_width() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        output_state.push_wide(&"x".repeat(110));
        let mut viewport = ViewportState::new(160, 80, 24);
        viewport.x_offset = 120;

        let mut context = test_input_context(workspace.path());
        context.render.actual_width = 50;
        context.render.actual_height = 24;

        let result = handle_input_event(
            Event::Resize(50, 24),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            context,
        );

        assert!(result.redraw);
        let visible_width = orangu::tui::output_view_width(
            VERSION,
            context.render.current_model,
            context.render.endpoint,
            context.render.workspace,
            context.render.prompt_branch,
            context.render.header_status,
            input_state.as_str(),
            50,
            24,
            context.render.tab_bar,
            context.render.tab_statuses,
            true,
        );
        assert_eq!(viewport.x_offset, 110usize.saturating_sub(visible_width));
    }

    #[test]
    fn idle_refresh_timeout_hits_zero_at_deadline() {
        let start = Instant::now();

        assert_eq!(
            idle_status_refresh_timeout(start + Duration::from_secs(60), start),
            Duration::from_secs(60)
        );
        assert_eq!(
            idle_status_refresh_timeout(
                start + Duration::from_secs(60),
                start + Duration::from_secs(61)
            ),
            Duration::ZERO
        );
    }

    #[test]
    fn dropdown_selection_wraps_around() {
        let mut dropdown = DropdownState {
            candidates: vec![
                ("a".into(), "".into()),
                ("b".into(), "".into()),
                ("c".into(), "".into()),
            ],
            selected_index: 0,
            replacement_start: 0,
        };

        dropdown.select_previous();
        assert_eq!(dropdown.selected_index, 2);
        dropdown.select_next();
        assert_eq!(dropdown.selected_index, 0);
        dropdown.select_next();
        assert_eq!(dropdown.selected_index, 1);
        dropdown.selected_index = 2;
        dropdown.select_next();
        assert_eq!(dropdown.selected_index, 0);
    }

    #[test]
    fn theme_dropdown_up_down_cycles_and_previews() {
        let workspace = tempdir().expect("workspace");
        orangu::tui::Theme::apply_named("classic").expect("commit classic");

        let mut input_state = InputState::default();
        input_state.set_buffer("/theme ".to_string());
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);
        let context = test_input_context(workspace.path());

        // Seed dropdown (set_buffer clears it).
        input_state.update_dropdown(context.skills, true);
        sync_theme_preview(&input_state);
        let dropdown = input_state.dropdown.as_ref().expect("theme dropdown");
        let count = dropdown.candidates.len();
        assert!(count >= 2);
        assert_eq!(dropdown.selected_index, 0);

        // Down from first advances; last + Down wraps to first.
        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Down,
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            context,
        );
        assert!(result.redraw);
        assert_eq!(input_state.dropdown.as_ref().unwrap().selected_index, 1);

        input_state.dropdown.as_mut().unwrap().selected_index = count - 1;
        let context = test_input_context(workspace.path());
        handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Down,
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            context,
        );
        assert_eq!(input_state.dropdown.as_ref().unwrap().selected_index, 0);

        // Up from first wraps to last.
        let context = test_input_context(workspace.path());
        handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Up,
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            context,
        );
        assert_eq!(
            input_state.dropdown.as_ref().unwrap().selected_index,
            count - 1
        );

        // Selecting a real theme (not "default") should live-preview without committing.
        let selected = input_state.dropdown.as_ref().unwrap().candidates
            [input_state.dropdown.as_ref().unwrap().selected_index]
            .0
            .clone();
        if selected != "default" {
            assert!(orangu::tui::Theme::is_previewing());
            assert_eq!(orangu::tui::Theme::current_theme_name(), "classic");
        }

        // Leaving `/theme` restores the committed theme.
        let context = test_input_context(workspace.path());
        input_state.set_buffer("/help".to_string());
        handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char(' '),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            context,
        );
        assert!(!orangu::tui::Theme::is_previewing());
        assert_eq!(orangu::tui::Theme::current_theme_name(), "classic");
    }

    #[test]
    fn escape_cancel_requires_two_presses_within_timeout() {
        let mut cancel_state = EscapeCancelState::default();
        let start = Instant::now();

        assert!(!cancel_state.handle_escape(start));
        assert!(cancel_state.handle_escape(start + Duration::from_millis(500)));

        assert!(!cancel_state.handle_escape(start + Duration::from_secs(5)));
        assert!(!cancel_state.handle_escape(start + Duration::from_secs(8)));
    }
}

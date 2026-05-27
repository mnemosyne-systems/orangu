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
    llm::StreamMetrics,
    tui::{Banner, HeaderStatus, StatusFragment, TranscriptLine, visible_line_width},
};
use std::{
    collections::VecDeque,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use super::completion::completion_candidates;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub const ESC_CANCEL_TIMEOUT: Duration = Duration::from_secs(2);
pub const CTRL_C_EXIT_TIMEOUT: Duration = Duration::from_secs(2);
pub const CTRL_C_EXIT_MESSAGE: &str = "Press Ctrl+c again to quit";
pub const IDLE_STATUS_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
pub const TRANSCRIPT_MAX_LINES: usize = 10_000;

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
}

pub enum WaitResult {
    Response(String),
    Cancelled(String),
    Quit,
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
    pub feedback: bool,
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

    pub fn pan_right(&mut self, max_content_width: usize) {
        self.x_offset = self.x_offset.saturating_add(1);
        let effective_right = max_content_width.min(self.virtual_width);
        let max_offset = effective_right.saturating_sub(self.actual_width);
        self.x_offset = self.x_offset.min(max_offset);
    }

    fn clamp_offset(&mut self) {
        let max_offset = self.virtual_width.saturating_sub(self.actual_width);
        self.x_offset = self.x_offset.min(max_offset);
    }
}

#[derive(Clone)]
pub struct ScreenState<'a> {
    pub transcript: &'a [TranscriptLine],
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_line: Option<&'a str>,
    pub input: &'a str,
    pub cursor: usize,
}

#[derive(Clone, Default)]
pub struct StreamRenderState {
    pub output: String,
    pub metrics: StreamMetrics,
    pub tool_running_since: Option<std::time::Instant>,
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
    pub model_names: &'a [String],
    pub render: RenderContext<'a>,
}

pub struct WaitContext<'a> {
    pub render: RenderContext<'a>,
    pub history: &'a mut Vec<String>,
    pub history_path: &'a Path,
    pub model_names: &'a [String],
    pub interrupt_state: &'a mut InterruptState,
    pub output_state: &'a mut OutputState,
    pub input_state: &'a mut InputState,
    pub pending_commands: &'a mut VecDeque<String>,
    pub thinking_quote: Option<&'static str>,
    pub viewport: &'a mut ViewportState,
}

#[derive(Default)]
pub struct OutputState {
    pub transcript: Vec<TranscriptLine>,
    pub scroll_offset: usize,
}

impl OutputState {
    pub fn lines(&self) -> &[TranscriptLine] {
        &self.transcript
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    pub fn clear(&mut self) {
        self.transcript.clear();
        self.scroll_offset = 0;
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
        self.push_lines(
            text.lines()
                .map(|line| TranscriptLine::UserInput(line.to_owned())),
        );
    }

    pub fn push_lines<I>(&mut self, lines: I)
    where
        I: Iterator<Item = TranscriptLine>,
    {
        let collected = lines.collect::<Vec<_>>();
        let added_lines = collected.len();
        self.transcript.extend(collected);
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
        self.push_text(&super::render::render_markdown_for_console(text));
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
        self.scroll_offset = self.scroll_offset.saturating_add(rows.max(1));
    }

    pub fn page_down(&mut self, rows: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(rows.max(1));
    }

    pub fn line_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    pub fn line_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }
}

#[derive(Default)]
pub struct InputState {
    pub buffer: String,
    pub cursor: usize,
    pub completion: Option<CompletionState>,
    pub history_index: Option<usize>,
    pub history_draft: String,
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
        self.history_index = None;
        self.history_draft.clear();
    }

    pub fn set_buffer(&mut self, buffer: String) {
        self.buffer = buffer;
        self.cursor = self.buffer.len();
        self.completion = None;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.completion = None;
    }

    pub fn insert_str(&mut self, text: &str) {
        self.buffer.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.completion = None;
    }

    pub fn backspace(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(previous..self.cursor);
            self.cursor = previous;
            self.completion = None;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(self.cursor..next);
            self.completion = None;
        }
    }

    pub fn move_left(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.cursor = previous;
            self.completion = None;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.cursor = next;
            self.completion = None;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
        self.completion = None;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.buffer.len();
        self.completion = None;
    }

    pub fn kill_to_end(&mut self) {
        self.buffer.truncate(self.cursor);
        self.completion = None;
    }

    pub fn kill_to_start(&mut self) {
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
        self.completion = None;
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
    }

    pub fn delete_backward_readline_word(&mut self) {
        let start = readline_word_start(&self.buffer, self.cursor);
        if start < self.cursor {
            self.buffer.drain(start..self.cursor);
            self.cursor = start;
            self.completion = None;
        }
    }

    pub fn delete_forward_readline_word(&mut self) {
        let end = readline_word_end(&self.buffer, self.cursor);
        if end > self.cursor {
            self.buffer.drain(self.cursor..end);
            self.completion = None;
        }
    }

    pub fn move_backward_readline_word(&mut self) {
        let start = readline_word_start(&self.buffer, self.cursor);
        if start != self.cursor {
            self.cursor = start;
            self.completion = None;
        }
    }

    pub fn move_forward_readline_word(&mut self) {
        let end = readline_word_end(&self.buffer, self.cursor);
        if end != self.cursor {
            self.cursor = end;
            self.completion = None;
        }
    }
}

pub fn idle_status_refresh_timeout(refresh_deadline: Instant, now: Instant) -> Duration {
    refresh_deadline
        .checked_duration_since(now)
        .unwrap_or(Duration::ZERO)
}

pub fn read_input(
    input_state: &mut InputState,
    interrupt_state: &mut InterruptState,
    output_state: &mut OutputState,
    pending_count: usize,
    viewport: &mut ViewportState,
    input_context: InputContext<'_>,
    print_screen_fn: impl Fn(RenderContext<'_>, ScreenState<'_>),
) -> anyhow::Result<InputResult> {
    use crossterm::event;

    let refresh_deadline = Instant::now() + IDLE_STATUS_REFRESH_INTERVAL;

    loop {
        let timeout = idle_status_refresh_timeout(refresh_deadline, Instant::now());
        if !event::poll(timeout)? {
            return Ok(InputResult::Refresh);
        }

        let result = handle_input_event(
            event::read()?,
            input_state,
            interrupt_state,
            output_state,
            viewport,
            input_context,
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
                    scroll_offset: output_state.scroll_offset(),
                    left_status: None,
                    pending_count,
                    pending_line: None,
                    input: input_state.as_str(),
                    cursor: input_state.cursor(),
                },
            );
            std::io::stdout().flush()?;
        }
    }
}

pub fn handle_input_event(
    event: Event,
    input_state: &mut InputState,
    interrupt_state: &mut InterruptState,
    output_state: &mut OutputState,
    viewport: &mut ViewportState,
    input_context: InputContext<'_>,
) -> InputEventResult {
    let mut redraw = false;

    match event {
        Event::Resize(w, h) => {
            viewport.on_resize(usize::from(w), usize::from(h));
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
                    let input = input_state.buffer.clone();
                    input_state.clear();
                    return InputEventResult {
                        redraw: false,
                        outcome: Some(InputResult::Submitted(input)),
                    };
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
                    viewport.pan_right(output_state.max_content_width());
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
                    history_previous(input_state, input_context.history);
                    redraw = true;
                }
                (KeyCode::Down, _) => {
                    interrupt_state.reset();
                    history_next(input_state, input_context.history);
                    redraw = true;
                }
                (KeyCode::Tab, _) => {
                    interrupt_state.reset();
                    apply_completion(
                        input_state,
                        input_context.workspace,
                        input_context.model_names,
                    );
                    redraw = true;
                }
                (KeyCode::PageUp, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                    interrupt_state.reset();
                    output_state.page_up(orangu::tui::output_view_rows(
                        VERSION,
                        input_context.render.current_model,
                        input_context.render.endpoint,
                        input_context.render.workspace,
                        input_context.render.prompt_branch,
                        input_context.render.header_status,
                        input_state.as_str(),
                        input_context.render.actual_width,
                        input_context.render.actual_height,
                    ));
                    redraw = true;
                }
                (KeyCode::PageDown, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                    interrupt_state.reset();
                    output_state.page_down(orangu::tui::output_view_rows(
                        VERSION,
                        input_context.render.current_model,
                        input_context.render.endpoint,
                        input_context.render.workspace,
                        input_context.render.prompt_branch,
                        input_context.render.header_status,
                        input_state.as_str(),
                        input_context.render.actual_width,
                        input_context.render.actual_height,
                    ));
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
        _ => {}
    }

    InputEventResult {
        redraw,
        outcome: None,
    }
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

pub fn apply_completion(input_state: &mut InputState, workspace: &Path, model_names: &[String]) {
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

    let Some((start, end, candidates)) = completion_candidates(
        input_state.as_str(),
        input_state.cursor(),
        workspace,
        model_names,
    ) else {
        return;
    };
    if candidates.is_empty() {
        return;
    }

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

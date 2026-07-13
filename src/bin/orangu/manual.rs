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

//! The built-in user manual (`/manual`).
//!
//! The manual chapters under `doc/manual/en` are embedded into the binary at
//! compile time, so the viewer never reads external files at run time. The
//! full-screen viewer mirrors the `/review` layout: the section text in the
//! left pane, the table of contents in the right pane, and the status bar and
//! input window kept at the bottom. Links are shown as their underlined
//! labels only, and fenced code blocks are shown syntax-highlighted without
//! the ``` fence lines. Alt+s opens a search window over the whole manual:
//! Enter jumps to the next match, Esc closes it.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use orangu::tui::{
    StatusFragment, Theme, prompt_prefix, review_pane_body_height,
    text::{clip_line, clip_ratatui_line},
    visible_line_width,
    widgets::PromptFrameWidget,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use super::input::{EscapeCancelState, InputState, ViewportState};
use super::render::{ANSI_FG_CODE, render_markdown_for_console};

/// Raw manual chapters, embedded at compile time in reading order. A chapter
/// added under `doc/manual/en` must also be added here. The pandoc front page
/// (`00-frontpage.md`) carries no readable text and is left out.
const MANUAL_SOURCES: &[&str] = &[
    include_str!("../../../doc/manual/en/01-introduction.md"),
    include_str!("../../../doc/manual/en/03-quickstart.md"),
    include_str!("../../../doc/manual/en/20-configuration.md"),
    include_str!("../../../doc/manual/en/30-tools.md"),
    include_str!("../../../doc/manual/en/31-workspaces.md"),
    include_str!("../../../doc/manual/en/32-skills.md"),
    include_str!("../../../doc/manual/en/40-terminal.md"),
    include_str!("../../../doc/manual/en/41-core_tools.md"),
    include_str!("../../../doc/manual/en/42-git_tools.md"),
    include_str!("../../../doc/manual/en/43-usage_tools.md"),
    include_str!("../../../doc/manual/en/44-coordinator.md"),
    include_str!("../../../doc/manual/en/46-server.md"),
    include_str!("../../../doc/manual/en/70-dev.md"),
    include_str!("../../../doc/manual/en/71-git.md"),
    include_str!("../../../doc/manual/en/72-extra.md"),
    include_str!("../../../doc/manual/en/73-openai.md"),
    include_str!("../../../doc/manual/en/74-completions.md"),
    include_str!("../../../doc/manual/en/75-compression.md"),
    include_str!("../../../doc/manual/en/76-coordinator.md"),
    include_str!("../../../doc/manual/en/78-server.md"),
    include_str!("../../../doc/manual/en/95-sponsors.md"),
    include_str!("../../../doc/manual/en/97-acknowledgement.md"),
    include_str!("../../../doc/manual/en/98-licenses.md"),
];

/// The shared `[id]: url` link definitions, appended to every page before
/// rendering so reference-style links (`[label][id]`) resolve and pick up the
/// link styling.
const MANUAL_LINK_DEFINITIONS: &str = include_str!("../../../doc/manual/en/99-references.md");

/// One `\newpage`-delimited page of the manual: a table-of-contents entry plus
/// the text shown in the left pane.
struct ManualSection {
    /// The page's first heading, without the `#` markers.
    title: String,
    /// Heading depth (1 = chapter, 2 = section, …), used to indent the TOC.
    level: usize,
    /// Pre-rendered text lines shown in the left pane.
    lines: Vec<String>,
}

impl ManualSection {
    /// The table-of-contents label: the title indented two spaces per heading
    /// level below the chapter.
    fn toc_label(&self) -> String {
        format!(
            "{}{}",
            "  ".repeat(self.level.saturating_sub(1)),
            self.title
        )
    }
}

/// Split a chapter into its pages: a new page starts at every line holding
/// only `\newpage` (the page-break marker the printed manual uses).
fn split_pages(source: &str) -> Vec<String> {
    let mut pages = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in source.lines() {
        if line.trim() == "\\newpage" {
            pages.push(current.join("\n"));
            current = Vec::new();
        } else {
            current.push(line);
        }
    }
    pages.push(current.join("\n"));
    pages
}

/// The page's first heading as `(title, level)`. A page without a heading
/// falls back to its first non-empty line as a level-1 entry.
fn page_heading(page: &str) -> (String, usize) {
    for line in page.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&ch| ch == '#').count();
            let title = trimmed.trim_start_matches('#').trim().to_string();
            if !title.is_empty() {
                return (title, level);
            }
        }
    }
    let first = page
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim()
        .to_string();
    (first, 1)
}

/// True for the rendered ``` fence lines around a code block; the manual
/// drops them and keeps only the syntax-highlighted code itself.
fn is_code_fence_line(line: &str) -> bool {
    line.strip_prefix(ANSI_FG_CODE)
        .is_some_and(|rest| rest.starts_with("```"))
}

/// `line` with its ANSI escape sequences removed.
fn strip_ansi(line: &str) -> String {
    let mut plain = String::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    for c in chars.by_ref() {
                        if c.is_ascii_alphabetic() || c == '~' || c == '@' {
                            break;
                        }
                    }
                }
                Some(&'O') => {
                    chars.next();
                    chars.next();
                }
                // An OSC sequence (e.g. an OSC 8 hyperlink) draws nothing, so
                // skip it: `ESC ] ... ST`, terminated by BEL or `ESC \`.
                Some(&']') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some('\x07') => break,
                            Some('\x1b') => {
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                            Some(_) => {}
                            None => break,
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        plain.push(ch);
    }
    plain
}

/// Render a page's markdown for the left pane: resolve reference links against
/// the shared definitions and drop the ``` fence lines. Links render as OSC 8
/// hyperlinks (just their labels, made clickable), so no URL suffix is shown.
fn page_lines(page: &str) -> Vec<String> {
    let source = format!("{page}\n\n{MANUAL_LINK_DEFINITIONS}");
    render_markdown_for_console(&source)
        .lines()
        .filter(|line| !is_code_fence_line(line))
        .map(str::to_string)
        .collect()
}

/// Parse the embedded chapters into sections, rendering each page's markdown
/// for the console once up front.
fn manual_sections() -> Vec<ManualSection> {
    let mut sections = Vec::new();
    for source in MANUAL_SOURCES {
        for page in split_pages(source) {
            if page.trim().is_empty() {
                continue;
            }
            let (title, level) = page_heading(&page);
            sections.push(ManualSection {
                title,
                level,
                lines: page_lines(&page),
            });
        }
    }
    sections
}

/// Static rendering pieces for the manual screen's prompt frame.
#[derive(Clone, Copy)]
pub struct ManualChrome<'a> {
    pub current_model: &'a str,
    pub prompt_branch: Option<&'a str>,
    pub pending_count: usize,
}

/// Interactive state for the manual viewer.
struct ManualState {
    sections: Vec<ManualSection>,
    selected: usize,
    /// Index of the highlighted line within the selected section's text
    /// (moved with Up/Down).
    line: usize,
    /// Index of the first line shown in the left pane.
    scroll: usize,
    /// Horizontal pan offset for the left pane.
    x_offset: usize,
    /// When set, the Alt+s search window is open at the top of the left pane.
    search: Option<InputState>,
    /// A transient status-bar message (failed search), cleared on the next
    /// key press.
    notice: Option<String>,
}

impl ManualState {
    fn new(sections: Vec<ManualSection>) -> Self {
        Self {
            sections,
            selected: 0,
            line: 0,
            scroll: 0,
            x_offset: 0,
            search: None,
            notice: None,
        }
    }

    fn selected_lines(&self) -> &[String] {
        self.sections
            .get(self.selected)
            .map(|section| section.lines.as_slice())
            .unwrap_or(&[])
    }

    /// Clamp the scroll/pan offsets to the selected section's text.
    fn clamp(&mut self, text_height: usize, left_width: usize) {
        self.line = self.line.min(self.selected_lines().len().saturating_sub(1));
        let max_scroll = self.selected_lines().len().saturating_sub(text_height);
        self.scroll = self.scroll.min(max_scroll);
        let content_width = self
            .selected_lines()
            .iter()
            .map(|line| visible_line_width(line))
            .max()
            .unwrap_or(0);
        self.x_offset = self.x_offset.min(content_width.saturating_sub(left_width));
    }

    /// Move the highlighted line up, scrolling the pane to keep it visible.
    fn cursor_up(&mut self) {
        self.line = self.line.saturating_sub(1);
        if self.line < self.scroll {
            self.scroll = self.line;
        }
    }

    /// Move the highlighted line down, scrolling the pane to keep it visible.
    fn cursor_down(&mut self, text_height: usize) {
        let last = self.selected_lines().len().saturating_sub(1);
        self.line = (self.line + 1).min(last);
        if text_height > 0 && self.line >= self.scroll + text_height {
            self.scroll = self.line + 1 - text_height;
        }
    }

    fn select_next(&mut self) {
        if self.selected + 1 < self.sections.len() {
            self.selected += 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    /// Jump to the next line containing the search text, case-insensitively,
    /// scanning forward from the highlighted line across section boundaries
    /// and wrapping around the whole manual. A miss is reported on the status
    /// bar and leaves the position unchanged.
    fn search_next(&mut self, text_height: usize) {
        let Some(query) = self
            .search
            .as_ref()
            .map(|editor| editor.as_str().trim().to_string())
            .filter(|query| !query.is_empty())
        else {
            return;
        };
        let needle = query.to_lowercase();
        let total: usize = self.sections.iter().map(|s| s.lines.len()).sum();
        let mut section = self.selected;
        let mut line = self.line;
        for _ in 0..total {
            line += 1;
            while line >= self.sections[section].lines.len() {
                section = (section + 1) % self.sections.len();
                line = 0;
                if !self.sections[section].lines.is_empty() {
                    break;
                }
            }
            if !strip_ansi(&self.sections[section].lines[line])
                .to_lowercase()
                .contains(&needle)
            {
                continue;
            }
            // Found: highlight the line and bring it into view (clamp trims
            // any overshoot).
            if section != self.selected {
                self.selected = section;
                self.x_offset = 0;
                self.scroll = line.saturating_sub(text_height / 2);
            } else if line < self.scroll || line >= self.scroll + text_height.max(1) {
                self.scroll = line.saturating_sub(text_height / 2);
            }
            self.line = line;
            return;
        }
        self.notice = Some(format!("No match for '{query}'"));
    }
}

/// Width of the right (table of contents) pane: as small as possible while
/// still fitting the longest entry, capped so the text pane stays usable.
fn manual_right_width(sections: &[ManualSection], actual_width: usize) -> usize {
    let actual_width = actual_width.max(1);
    let longest = sections
        .iter()
        .map(|section| section.toc_label().chars().count())
        .max()
        .unwrap_or(0);
    let header = format!("Contents ({})", sections.len());
    let desired = longest.max(header.chars().count());
    // Give the content (left pane) priority by capping the TOC (right pane)
    // at 25% of the screen (or 25 columns on very small terminals).
    let cap = (actual_width / 4)
        .max(25)
        .min(actual_width.saturating_sub(2).max(1));
    desired.clamp(1, cap)
}

/// Render the Alt+s search window: the query on the user-input background
/// with a reverse-video caret, padded to `width`.
fn render_search_bar<'a>(
    query: &'a str,
    cursor: usize,
    width: usize,
    theme: &'a Theme,
) -> Line<'a> {
    let mut spans = vec![Span::raw("Search: ")];
    let cursor_chars = query[..cursor.min(query.len())].chars().count();
    let chars: Vec<char> = query.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if index == cursor_chars {
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().add_modifier(Modifier::REVERSED),
            ));
        } else {
            spans.push(Span::raw(ch.to_string()));
        }
    }
    if cursor_chars >= chars.len() {
        spans.push(Span::styled(
            " ",
            Style::default().add_modifier(Modifier::REVERSED),
        ));
    }

    let line = Line::from(spans);
    let clipped = clip_ratatui_line(&line, 0, width);

    let w = clipped.width();
    let pad = " ".repeat(width.saturating_sub(w));

    let mut padded_spans = clipped.spans;
    padded_spans.push(Span::raw(pad));

    Line::from(padded_spans).style(theme.user_input.bg(Color::Rgb(40, 40, 40)))
}

/// Rendering inputs for the manual screen.
fn draw_manual_screen(
    frame: &mut ratatui::Frame,
    state: &ManualState,
    viewport: &ViewportState,
    chrome: ManualChrome<'_>,
    theme: &Theme,
) {
    let width = viewport.actual_width.max(1) as u16;
    let height = viewport.actual_height.max(1) as u16;

    let prompt_prefix = prompt_prefix(chrome.prompt_branch);
    let pane_rows = height.saturating_sub(4).max(1);

    let right_width = manual_right_width(&state.sections, width as usize) as u16;
    let left_width = width.saturating_sub(right_width + 1).max(1);
    let body_height = pane_rows.saturating_sub(1) as usize;

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(pane_rows), Constraint::Length(4)])
        .split(Rect::new(0, 0, width, height));

    let panes_area = layout[0];
    let prompt_area = layout[1];

    let panes_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(left_width),
            Constraint::Length(1),
            Constraint::Length(right_width),
        ])
        .split(panes_area);

    let left_area = panes_layout[0];
    let sep_area = panes_layout[1];
    let right_area = panes_layout[2];

    let list_start = if state.selected >= body_height {
        state.selected - body_height + 1
    } else {
        0
    };

    let title = "Manual  Alt+j/k Switch section  Alt+s Search  Alt+x Exit";
    let right_header = format!("Contents ({})", state.sections.len());

    let mut left_lines = vec![Line::from(clip_line(title, 0, left_area.width as usize))];
    let mut right_lines = vec![Line::from(clip_line(
        &right_header,
        0,
        right_area.width as usize,
    ))];
    let mut sep_lines = vec![Line::from(Span::styled("│", theme.muted))];

    let selected_lines: &[String] = state
        .sections
        .get(state.selected)
        .map(|section| section.lines.as_slice())
        .unwrap_or(&[]);

    let search_bar = state.search.as_ref().map(|editor| {
        render_search_bar(
            editor.as_str(),
            editor.cursor(),
            left_area.width as usize,
            theme,
        )
    });
    let text_row_offset = usize::from(search_bar.is_some());

    for row in 0..body_height {
        sep_lines.push(Line::from(Span::styled("│", theme.muted)));

        if row == 0 && search_bar.is_some() {
            left_lines.push(search_bar.clone().unwrap());
        } else {
            let line_index = state.scroll + row - text_row_offset;
            match selected_lines.get(line_index) {
                Some(line) => {
                    let clipped = clip_line(line, state.x_offset, left_area.width as usize);
                    let mut spans = Vec::new();
                    if let Ok(text) = ansi_to_tui::IntoText::into_text(&clipped) {
                        if let Some(parsed_line) = text.lines.into_iter().next() {
                            spans.extend(parsed_line.spans);
                        }
                    } else {
                        spans.push(Span::raw(clipped));
                    }

                    let mut line_rat = Line::from(spans);
                    if line_index == state.line {
                        line_rat = line_rat.style(theme.cursor_line_bg);
                    }
                    left_lines.push(line_rat);
                }
                None => left_lines.push(Line::raw("")),
            }
        };

        let section_index = list_start + row;
        match state.sections.get(section_index) {
            Some(section) => {
                let clipped = clip_line(&section.toc_label(), 0, right_area.width as usize);
                let mut line = Line::from(clipped);
                if section_index == state.selected {
                    line = line.style(theme.cursor_line_bg);
                }
                right_lines.push(line);
            }
            None => right_lines.push(Line::raw("")),
        }
    }

    frame.render_widget(Paragraph::new(left_lines), left_area);
    frame.render_widget(Paragraph::new(sep_lines), sep_area);
    frame.render_widget(Paragraph::new(right_lines), right_area);

    let current_model = orangu::tui::header::display_model_name(false, chrome.current_model);
    let notice = state
        .notice
        .as_ref()
        .map(|notice| StatusFragment::plain(notice.clone()));
    let prompt_widget = PromptFrameWidget {
        current_model,
        prompt_prefix: &prompt_prefix,
        input: "",
        cursor: 0,
        ghost: "",
        valid_command_len: 0,
        left_status: notice.as_ref(),
        pending_count: chrome.pending_count,
        graph_status: None,
        prompt_branch: None,
    };
    frame.render_widget(prompt_widget, prompt_area);
}

/// Run the manual viewer event loop until the user exits (Alt+x or Esc Esc).
pub fn run_manual_mode(
    viewport: &mut ViewportState,
    chrome: ManualChrome<'_>,
    terminal_guard: &mut super::terminal::TerminalUiGuard,
) -> Result<()> {
    let theme = Theme::current();
    let mut state = ManualState::new(manual_sections());
    let mut escape_cancel = EscapeCancelState::default();
    loop {
        // The input window is always empty here, so the body height matches a
        // review pane with an empty input. The open search window costs one
        // text row.
        let body_height = review_pane_body_height(
            viewport.actual_height,
            "",
            chrome.prompt_branch,
            viewport.actual_width,
        );
        let text_height = body_height.saturating_sub(usize::from(state.search.is_some()));
        let right_width = manual_right_width(&state.sections, viewport.actual_width);
        let left_width = viewport.actual_width.saturating_sub(right_width + 1).max(1);
        state.clamp(text_height, left_width);

        terminal_guard.terminal.draw(|f| {
            draw_manual_screen(f, &state, viewport, chrome, &theme);
        })?;

        let (code, modifiers) = match event::read()? {
            Event::Resize(width, height) => {
                viewport.on_resize(usize::from(width), usize::from(height));
                continue;
            }
            Event::Mouse(crossterm::event::MouseEvent {
                kind: crossterm::event::MouseEventKind::ScrollUp,
                ..
            }) => {
                state.scroll = state.scroll.saturating_sub(1);
                continue;
            }
            Event::Mouse(crossterm::event::MouseEvent {
                kind: crossterm::event::MouseEventKind::ScrollDown,
                ..
            }) => {
                state.scroll = state.scroll.saturating_add(1);
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

        // Any key press clears the search notice from the status bar.
        state.notice = None;

        let alt =
            modifiers.contains(KeyModifiers::ALT) && !modifiers.contains(KeyModifiers::CONTROL);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);

        // While the search window is open it is modal: type the text, Enter
        // jumps to its next occurrence, Esc closes the window.
        if state.search.is_some() {
            escape_cancel.reset();
            match (code, alt, ctrl) {
                (KeyCode::Esc, _, _) => state.search = None,
                (KeyCode::Enter, _, _) => state.search_next(text_height),
                (KeyCode::Backspace, true, _) => {
                    state
                        .search
                        .as_mut()
                        .unwrap()
                        .delete_backward_readline_word();
                }
                (KeyCode::Backspace, _, _) => state.search.as_mut().unwrap().backspace(),
                (KeyCode::Delete, _, _) => state.search.as_mut().unwrap().delete(),
                (KeyCode::Left, _, true) => {
                    state.search.as_mut().unwrap().move_backward_readline_word();
                }
                (KeyCode::Right, _, true) => {
                    state.search.as_mut().unwrap().move_forward_readline_word();
                }
                (KeyCode::Left, _, _) => state.search.as_mut().unwrap().move_left(),
                (KeyCode::Right, _, _) => state.search.as_mut().unwrap().move_right(),
                (KeyCode::Home, _, _) => state.search.as_mut().unwrap().move_home(),
                (KeyCode::End, _, _) => state.search.as_mut().unwrap().move_end(),
                (KeyCode::Char(ch), false, false) => {
                    state.search.as_mut().unwrap().insert_char(ch);
                }
                _ => {}
            }
            continue;
        }

        // A second Esc within the timeout leaves the manual; the first arms it.
        if code == KeyCode::Esc {
            if escape_cancel.handle_escape(std::time::Instant::now()) {
                return Ok(());
            }
            continue;
        }
        escape_cancel.reset();

        match (code, alt) {
            (KeyCode::Char('x'), true) => return Ok(()),
            (KeyCode::Char('j'), true) => state.select_next(),
            (KeyCode::Char('k'), true) => state.select_prev(),
            (KeyCode::Char('s'), true) => state.search = Some(InputState::default()),
            // Left-pane scrolling (Alt+arrows / PageUp/Down), mirroring
            // review mode.
            (KeyCode::Up, true) => state.scroll = state.scroll.saturating_sub(1),
            (KeyCode::Down, true) => state.scroll = state.scroll.saturating_add(1),
            (KeyCode::PageUp, _) => state.scroll = state.scroll.saturating_sub(text_height),
            (KeyCode::PageDown, _) => state.scroll = state.scroll.saturating_add(text_height),
            // Move the highlighted line through the text, view following.
            (KeyCode::Up, false) => state.cursor_up(),
            (KeyCode::Down, false) => state.cursor_down(text_height),
            // There is no input window to edit, so plain and Alt arrows both
            // pan the text horizontally.
            (KeyCode::Left, _) => state.x_offset = state.x_offset.saturating_sub(1),
            (KeyCode::Right, _) => state.x_offset = state.x_offset.saturating_add(1),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ManualSection, ManualState, is_code_fence_line, manual_sections, page_heading};
    use crate::input::InputState;
    use crate::render::ANSI_FG_LINK;

    fn section(title: &str, level: usize, lines: &[&str]) -> ManualSection {
        ManualSection {
            title: title.to_string(),
            level,
            lines: lines.iter().map(|line| line.to_string()).collect(),
        }
    }

    fn searching_state(sections: Vec<ManualSection>, query: &str) -> ManualState {
        let mut state = ManualState::new(sections);
        let mut editor = InputState::default();
        editor.set_buffer(query.to_string());
        state.search = Some(editor);
        state
    }

    #[test]
    fn manual_sections_split_on_newpage_with_titles() {
        let sections = manual_sections();
        assert!(sections.len() > 20, "expected one section per manual page");
        assert_eq!(sections[0].title, "Introduction");
        assert_eq!(sections[0].level, 1);
        assert!(sections.iter().all(|section| !section.title.is_empty()));
        assert!(sections.iter().all(|section| !section.lines.is_empty()));
    }

    #[test]
    fn toc_labels_indent_sections_below_chapters() {
        let sections = manual_sections();
        let help = sections
            .iter()
            .find(|section| section.title == "/help")
            .expect("/help section present");
        assert_eq!(help.level, 2);
        assert_eq!(help.toc_label(), "  /help");
    }

    #[test]
    fn manual_sections_show_links_as_styled_labels() {
        let sections = manual_sections();
        let intro = &sections[0];
        // Reference links resolve through the shared definitions and keep the
        // link styling, while the dimmed ` (url)` suffix is stripped.
        assert!(
            intro.lines.iter().any(|line| line.contains(ANSI_FG_LINK)),
            "introduction should carry link-styled labels"
        );
        assert!(
            intro
                .lines
                .iter()
                .all(|line| !line.contains("(https://github.com/mnemosyne-systems/orangu)")),
            "the URL suffix should not be shown in the text"
        );
    }

    #[test]
    fn manual_sections_drop_code_fence_lines() {
        let sections = manual_sections();
        let help = sections
            .iter()
            .find(|section| section.title == "/help")
            .expect("/help section present");
        assert!(
            help.lines.iter().all(|line| !is_code_fence_line(line)),
            "fence lines should be removed"
        );
        // The example code itself is kept.
        assert!(
            help.lines
                .iter()
                .any(|line| line.contains("show available commands")),
            "code-block content should remain"
        );
    }

    #[test]
    fn search_next_wraps_across_sections_case_insensitively() {
        let sections = vec![
            section("One", 1, &["alpha", "the Needle here"]),
            section("Two", 1, &["other text", "needle again"]),
        ];
        let mut state = searching_state(sections, "needle");
        state.search_next(10);
        assert_eq!((state.selected, state.line), (0, 1));
        state.search_next(10);
        assert_eq!((state.selected, state.line), (1, 1));
        // Wraps around the whole manual back to the first match.
        state.search_next(10);
        assert_eq!((state.selected, state.line), (0, 1));
        assert!(state.notice.is_none());
    }

    #[test]
    fn search_next_matches_visible_text_behind_ansi_codes() {
        let sections = vec![section(
            "One",
            1,
            &["plain", "\u{1b}[1mbold needle\u{1b}[22m"],
        )];
        let mut state = searching_state(sections, "bold needle");
        state.search_next(10);
        assert_eq!((state.selected, state.line), (0, 1));
    }

    #[test]
    fn search_next_reports_a_miss_on_the_status_bar() {
        let sections = vec![section("One", 1, &["alpha", "beta"])];
        let mut state = searching_state(sections, "missing");
        state.search_next(10);
        assert_eq!((state.selected, state.line), (0, 0), "position unchanged");
        assert_eq!(state.notice.as_deref(), Some("No match for 'missing'"));
    }

    #[test]
    fn search_next_scrolls_the_match_into_view() {
        let lines: Vec<String> = (0..50).map(|index| format!("line {index}")).collect();
        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let sections = vec![section("One", 1, &line_refs)];
        let mut state = searching_state(sections, "line 40");
        state.search_next(10);
        assert_eq!(state.line, 40);
        assert!(
            state.scroll <= 40 && 40 < state.scroll + 10,
            "match should be scrolled into view (scroll {})",
            state.scroll
        );
    }

    #[test]
    fn split_pages_breaks_on_newpage_lines_only() {
        let pages = super::split_pages("\\newpage\n# One\ntext\n\\newpage\n# Two");
        let non_empty: Vec<&String> = pages
            .iter()
            .filter(|page| !page.trim().is_empty())
            .collect();
        assert_eq!(
            non_empty,
            [&"# One\ntext".to_string(), &"# Two".to_string()]
        );
    }

    #[test]
    fn page_heading_parses_title_and_level() {
        assert_eq!(
            page_heading("# Introduction\ntext"),
            ("Introduction".to_string(), 1)
        );
        assert_eq!(
            page_heading("intro\n## /help\nbody"),
            ("/help".to_string(), 2)
        );
        assert_eq!(
            page_heading("plain text only"),
            ("plain text only".to_string(), 1)
        );
    }

    #[test]
    fn code_fence_lines_are_detected() {
        assert!(is_code_fence_line(&format!(
            "{}```text",
            crate::render::ANSI_FG_CODE
        )));
        assert!(is_code_fence_line(&format!(
            "{}```",
            crate::render::ANSI_FG_CODE
        )));
        assert!(!is_code_fence_line(&format!(
            "{}`inline`",
            crate::render::ANSI_FG_CODE
        )));
        assert!(!is_code_fence_line("plain text"));
    }
}

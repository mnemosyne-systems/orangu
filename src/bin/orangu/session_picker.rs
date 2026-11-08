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

//! The session picker behind `orangu -r` without a UUID.
//!
//! A full-screen list of every stored session — the `-l` columns (SESSION,
//! WORKSPACE, BRANCH, DATE), most recently updated first — with one row
//! highlighted, and a final `New` row that starts a fresh session instead.
//! Up/Down move the highlight, Enter takes the row and Esc leaves without
//! one. It runs before the interface exists, on its own terminal guard, so
//! the pick can decide which session (and which workspace) the first tab
//! opens on.

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use orangu::tui::{Theme, text::clip_line};
use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Paragraph},
};
use std::path::PathBuf;

use super::session_store::{SessionMetadata, format_unix_timestamp_human, stored_sessions};
use super::terminal::{TerminalUiGuard, mouse_selection};

/// One row of the list, already rendered to the strings the table shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PickerEntry {
    pub(crate) uuid: String,
    /// The workspace the session was started in, or `-` when unknown.
    pub(crate) workspace: String,
    pub(crate) branch: String,
    /// The last-updated timestamp as `YYYY-MM-DD HH:MM`, or `-`.
    pub(crate) date: String,
}

impl PickerEntry {
    fn from_stored(uuid: String, meta: Option<&SessionMetadata>) -> Self {
        let dash = || "-".to_string();
        Self {
            uuid,
            workspace: meta
                .filter(|m| !m.workspace.is_empty())
                .map(|m| m.workspace.clone())
                .unwrap_or_else(dash),
            branch: meta
                .filter(|m| !m.branch.is_empty())
                .map(|m| m.branch.clone())
                .unwrap_or_else(dash),
            date: meta
                .map(|m| format_unix_timestamp_human(m.last_updated_at))
                .unwrap_or_else(dash),
        }
    }

    /// The workspace to open the session in, when the metadata recorded one.
    pub(crate) fn workspace_path(&self) -> Option<PathBuf> {
        (self.workspace != "-").then(|| PathBuf::from(&self.workspace))
    }
}

/// What the user took from the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PickerChoice {
    /// A stored session: its UUID and, when its metadata recorded one, the
    /// workspace it was started in.
    Resume {
        uuid: String,
        workspace: Option<PathBuf>,
    },
    /// The `New` row: no resume at all, not even the automatic one for the
    /// workspace and branch — a fresh session.
    New,
}

/// Every stored session as a picker row, most recently updated first. A
/// session without metadata has no timestamp to order by and sorts last.
pub(crate) fn picker_entries() -> Result<Vec<PickerEntry>> {
    let mut sessions = stored_sessions()?;
    sessions.sort_by_key(|(_, meta)| {
        std::cmp::Reverse(meta.as_ref().map(|m| m.last_updated_at).unwrap_or(0))
    });
    Ok(sessions
        .into_iter()
        .map(|(uuid, meta)| PickerEntry::from_stored(uuid, meta.as_ref()))
        .collect())
}

/// The list and the row highlighted in it. The rows are the entries followed
/// by one `New` row, so `selected == entries.len()` is that row.
pub(crate) struct PickerState {
    pub(crate) entries: Vec<PickerEntry>,
    pub(crate) selected: usize,
    /// Index of the first row on screen; kept so the highlight stays visible.
    pub(crate) scroll: usize,
}

impl PickerState {
    pub(crate) fn new(entries: Vec<PickerEntry>) -> Self {
        Self {
            entries,
            selected: 0,
            scroll: 0,
        }
    }

    /// Every row, the trailing `New` one included.
    pub(crate) fn row_count(&self) -> usize {
        self.entries.len() + 1
    }

    pub(crate) fn select_next(&mut self) {
        if self.selected + 1 < self.row_count() {
            self.selected += 1;
        }
    }

    pub(crate) fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub(crate) fn page_down(&mut self, rows: usize) {
        let last = self.row_count() - 1;
        self.selected = (self.selected + rows.max(1)).min(last);
    }

    pub(crate) fn page_up(&mut self, rows: usize) {
        self.selected = self.selected.saturating_sub(rows.max(1));
    }

    pub(crate) fn select_first(&mut self) {
        self.selected = 0;
    }

    pub(crate) fn select_last(&mut self) {
        self.selected = self.row_count() - 1;
    }

    /// Slide the window so the highlighted row is inside `rows` visible rows.
    pub(crate) fn clamp(&mut self, rows: usize) {
        let rows = rows.max(1);
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + rows {
            self.scroll = self.selected + 1 - rows;
        }
    }

    pub(crate) fn picked(&self) -> PickerChoice {
        match self.entries.get(self.selected) {
            Some(entry) => PickerChoice::Resume {
                uuid: entry.uuid.clone(),
                workspace: entry.workspace_path(),
            },
            None => PickerChoice::New,
        }
    }
}

/// The label of the `New` row, in the SESSION column.
pub(crate) const NEW_ROW_LABEL: &str = "New";
const NEW_ROW_TEXT: &str = "Start a new session";

/// The header and rows of the table, each padded to the widest value of its
/// column — the same layout `-l` prints, so the two read alike — followed
/// by the `New` row.
pub(crate) fn table_lines(entries: &[PickerEntry]) -> (String, Vec<String>) {
    let col_width = |header: &str, value: &dyn Fn(&PickerEntry) -> &str| {
        entries
            .iter()
            .map(|e| value(e).chars().count())
            .chain(std::iter::once(header.chars().count()))
            .max()
            .unwrap_or(0)
    };
    let w_session = col_width("SESSION", &|e| &e.uuid);
    let w_workspace = col_width("WORKSPACE", &|e| &e.workspace);
    let w_branch = col_width("BRANCH", &|e| &e.branch);
    let header = format!(
        "{:<w_session$}  {:<w_workspace$}  {:<w_branch$}  {}",
        "SESSION", "WORKSPACE", "BRANCH", "DATE"
    );
    let mut rows: Vec<String> = entries
        .iter()
        .map(|e| {
            format!(
                "{:<w_session$}  {:<w_workspace$}  {:<w_branch$}  {}",
                e.uuid, e.workspace, e.branch, e.date
            )
        })
        .collect();
    rows.push(format!("{NEW_ROW_LABEL:<w_session$}  {NEW_ROW_TEXT}"));
    (header, rows)
}

const TITLE: &str = "Resume a session  ↑/↓ Move  Enter Select  Esc Quit";

/// Rows of the list body: the screen minus the title, the column header and
/// the status line.
fn body_rows(height: u16) -> usize {
    usize::from(height.saturating_sub(3).max(1))
}

fn draw_picker(frame: &mut ratatui::Frame, state: &PickerState, theme: &Theme) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(theme.bg_base).fg(theme.text_primary)),
        area,
    );
    let width = usize::from(area.width.max(1));
    let rows = body_rows(area.height);
    let (header, table) = table_lines(&state.entries);

    let mut lines = Vec::with_capacity(rows + 3);
    lines.push(Line::from(clip_line(TITLE, 0, width)));
    lines.push(Line::from(Span::styled(
        clip_line(&header, 0, width),
        theme.muted,
    )));
    for row in 0..rows {
        let index = state.scroll + row;
        match table.get(index) {
            Some(text) => {
                // Pad the row to the full width so the highlight spans it.
                let mut clipped = clip_line(text, 0, width);
                let shown = clipped.chars().count();
                clipped.extend(std::iter::repeat_n(' ', width.saturating_sub(shown)));
                let mut line = Line::from(clipped);
                if index == state.selected {
                    line = line.style(theme.cursor_line_bg);
                }
                lines.push(line);
            }
            None => lines.push(Line::raw("")),
        }
    }
    let status = if state.selected < state.entries.len() {
        format!("Session {} of {}", state.selected + 1, state.entries.len())
    } else {
        NEW_ROW_TEXT.to_string()
    };
    lines.push(Line::from(Span::styled(
        clip_line(&status, 0, width),
        theme.muted,
    )));
    frame.render_widget(
        Paragraph::new(lines),
        Rect::new(0, 0, area.width, area.height),
    );
}

/// Show the picker and wait for a choice. `Ok(None)` is a cancel (Esc, `q`
/// or Ctrl+C). With no stored session the list is the `New` row alone.
pub(crate) fn run_session_picker(mouse_capture: bool) -> Result<Option<PickerChoice>> {
    let entries = picker_entries()?;
    let mut guard = TerminalUiGuard::new(mouse_capture)?;
    let picked = pick_from(&mut guard, PickerState::new(entries));
    // Leaving the alternate screen reset the terminal's colours; the interface
    // that opens next expects the theme's.
    drop(guard);
    Theme::reapply_terminal_palette();
    picked
}

fn pick_from(guard: &mut TerminalUiGuard, mut state: PickerState) -> Result<Option<PickerChoice>> {
    let theme = Theme::current();
    loop {
        let rows = body_rows(guard.terminal.size()?.height);
        state.clamp(rows);
        guard.draw(|frame| draw_picker(frame, &state, &theme))?;

        let event = crate::terminal::read_typed_event()?;
        if mouse_selection(&event).consumed() {
            continue;
        }
        let (code, modifiers) = match event {
            Event::Mouse(crossterm::event::MouseEvent {
                kind: crossterm::event::MouseEventKind::ScrollUp,
                ..
            }) => {
                state.select_prev();
                continue;
            }
            Event::Mouse(crossterm::event::MouseEvent {
                kind: crossterm::event::MouseEventKind::ScrollDown,
                ..
            }) => {
                state.select_next();
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
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        match (code, ctrl) {
            (KeyCode::Enter, _) => return Ok(Some(state.picked())),
            (KeyCode::Esc, _) | (KeyCode::Char('q'), false) | (KeyCode::Char('c'), true) => {
                return Ok(None);
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), false) => state.select_prev(),
            (KeyCode::Down, _) | (KeyCode::Char('j'), false) => state.select_next(),
            (KeyCode::PageUp, _) => state.page_up(rows),
            (KeyCode::PageDown, _) => state.page_down(rows),
            (KeyCode::Home, _) => state.select_first(),
            (KeyCode::End, _) => state.select_last(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PickerChoice, PickerEntry, PickerState, table_lines};
    use std::path::PathBuf;

    fn entry(uuid: &str, workspace: &str, branch: &str, date: &str) -> PickerEntry {
        PickerEntry {
            uuid: uuid.to_string(),
            workspace: workspace.to_string(),
            branch: branch.to_string(),
            date: date.to_string(),
        }
    }

    fn entries(count: usize) -> Vec<PickerEntry> {
        (0..count)
            .map(|i| entry(&format!("uuid-{i}"), "/ws", "main", "2026-01-01 00:00"))
            .collect()
    }

    #[test]
    fn selection_stays_inside_the_list_new_row_included() {
        // Three sessions and the `New` row: four rows, 0..=3.
        let mut state = PickerState::new(entries(3));
        state.select_prev();
        assert_eq!(state.selected, 0);
        for _ in 0..5 {
            state.select_next();
        }
        assert_eq!(state.selected, 3);
        assert_eq!(state.picked(), PickerChoice::New);
        state.page_up(10);
        assert_eq!(state.selected, 0);
        state.page_down(10);
        assert_eq!(state.selected, 3);
        state.select_first();
        assert_eq!(state.selected, 0);
        state.select_last();
        assert_eq!(state.selected, 3);
    }

    #[test]
    fn scroll_follows_the_highlight_both_ways() {
        let mut state = PickerState::new(entries(10));
        state.clamp(4);
        assert_eq!(state.scroll, 0);
        state.page_down(6);
        state.clamp(4);
        // Row 6 is the last of the four shown: rows 3..=6.
        assert_eq!((state.selected, state.scroll), (6, 3));
        state.select_first();
        state.clamp(4);
        assert_eq!(state.scroll, 0);
    }

    #[test]
    fn picked_carries_the_workspace_only_when_known() {
        let mut state = PickerState::new(vec![
            entry("a", "/home/u/p", "main", "2026-01-01 00:00"),
            entry("b", "-", "-", "-"),
        ]);
        assert_eq!(
            state.picked(),
            PickerChoice::Resume {
                uuid: "a".to_string(),
                workspace: Some(PathBuf::from("/home/u/p")),
            }
        );
        state.select_next();
        assert_eq!(
            state.picked(),
            PickerChoice::Resume {
                uuid: "b".to_string(),
                workspace: None,
            }
        );
        state.select_next();
        assert_eq!(state.picked(), PickerChoice::New);
    }

    #[test]
    fn an_empty_store_offers_only_the_new_row() {
        let state = PickerState::new(Vec::new());
        assert_eq!(state.row_count(), 1);
        assert_eq!(state.picked(), PickerChoice::New);
        let (_, rows) = table_lines(&[]);
        assert_eq!(rows, vec!["New      Start a new session".to_string()]);
    }

    #[test]
    fn table_columns_are_sized_to_the_widest_value() {
        let (header, rows) = table_lines(&[
            entry(
                "550e8400-e29b-41d4-a716-446655440000",
                "/home/user/project",
                "main",
                "2026-06-26 11:04",
            ),
            entry(
                "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                "/home/user/other",
                "feature/login",
                "2026-06-26 03:27",
            ),
        ]);
        assert_eq!(
            header,
            "SESSION                               WORKSPACE           BRANCH         DATE"
        );
        assert_eq!(
            rows[0],
            "550e8400-e29b-41d4-a716-446655440000  /home/user/project  main           2026-06-26 11:04"
        );
        assert_eq!(
            rows[1],
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8  /home/user/other    feature/login  2026-06-26 03:27"
        );
        // The `New` row closes the list, its label in the SESSION column.
        assert_eq!(
            rows[2],
            "New                                   Start a new session"
        );
        assert_eq!(rows.len(), 3);
    }
}

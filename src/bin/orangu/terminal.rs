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
use strum::IntoEnumIterator;

pub(crate) const CLEAR_TERMINAL_SEQUENCE: &str = "\x1b[2J\x1b[H";
pub(crate) const TERMINAL_TITLE: &str = "orangu";

pub(crate) struct TerminalTitleGuard;

impl TerminalTitleGuard {
    pub(crate) fn new(title: &str) -> Self {
        set_terminal_title(Some(title));
        Self
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        set_terminal_title(None);
    }
}

pub(crate) fn set_terminal_title(title: Option<&str>) {
    match title {
        Some(title) => print!("\x1b]0;{title}\x07"),
        None => print!("\x1b]0;\x07"),
    }
}

/// Ring the terminal bell (ASCII `BEL`), which terminals surface as the
/// standard notification sound (or a visual flash when configured). Used to
/// announce that a long-running `/auto_review` has finished.
pub(crate) fn ring_terminal_bell() {
    print!("\x07");
}
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::Stdout;

pub(crate) struct TerminalUiGuard {
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
    mouse_capture: bool,
}

impl TerminalUiGuard {
    pub(crate) fn new(mouse_capture: bool) -> Result<Self> {
        enable_raw_mode()?;
        if mouse_capture {
            execute!(
                std::io::stdout(),
                crossterm::event::EnableMouseCapture,
                EnterAlternateScreen
            )?;
        } else {
            execute!(std::io::stdout(), EnterAlternateScreen)?;
        }
        let _ = execute!(
            std::io::stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
        let backend = CrosstermBackend::new(std::io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            mouse_capture,
        })
    }
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        use std::io::Write;
        print!("\x1b]111\x07\x1b]110\x07\x1b]112\x07");
        let _ = std::io::stdout().flush();

        if self.mouse_capture {
            let _ = execute!(
                std::io::stdout(),
                crossterm::event::DisableMouseCapture,
                PopKeyboardEnhancementFlags,
                LeaveAlternateScreen
            );
        } else {
            let _ = execute!(
                std::io::stdout(),
                PopKeyboardEnhancementFlags,
                LeaveAlternateScreen
            );
        }
        let _ = disable_raw_mode();
    }
}

// Currently unused; kept for callers that need to drop out of raw mode
// temporarily (e.g. handing the terminal to a child process).
#[allow(dead_code)]
pub struct RawModePauseGuard;

impl RawModePauseGuard {
    #[allow(dead_code)]
    pub fn new() -> Result<Self> {
        disable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModePauseGuard {
    fn drop(&mut self) {
        let _ = enable_raw_mode();
    }
}

impl TerminalUiGuard {
    pub fn print_screen(&mut self, render: RenderContext<'_>, screen: ScreenState<'_>) {
        // Only hint a completion while the cursor sits at the end of what was typed.
        // Slash commands take priority over natural-language bindings; for the latter,
        // `ghost_index` selects which candidate to preview (cycled with Shift+Tab).
        // Structured argument completions (branches, tags, files, models, servers)
        // fall last, previewing the first candidate Tab would fill in.
        let structured_ghost = completion::input_ghost_suffix(
            screen.input,
            screen.cursor,
            screen.ghost_index,
            render.workspace,
            render.server_names,
            render.available_models,
            render.skills,
        );
        let ghost = structured_ghost.as_deref().unwrap_or("");
        let mut valid_command_len = 0;
        if screen.input.starts_with('/') {
            let first_word = screen.input.split_whitespace().next().unwrap_or("");
            if crate::slash_command::SlashCommand::iter().any(|c| c.command() == first_word)
                || render
                    .skills
                    .find(first_word.trim_start_matches('/'))
                    .is_some()
            {
                valid_command_len = first_word.chars().count();
            }
        }

        let args = ScreenRenderArgs {
            version: VERSION,
            current_model: render.current_model,
            endpoint: render.endpoint,
            workspace: render.workspace,
            prompt_branch: render.prompt_branch,
            status: render.header_status,
            banner: render.banner,
            tab_bar: render.tab_bar,
            tab_statuses: render.tab_statuses,
            transcript: screen.transcript,
            transcript_epoch: screen.transcript_epoch,
            scroll_offset: screen.scroll_offset,
            left_status: screen.left_status,
            pending_count: screen.pending_count,
            pending_lines: screen.pending_lines,
            input: screen.input,
            cursor: screen.cursor,
            ghost,
            virtual_width: render.virtual_width,
            actual_width: render.actual_width,
            actual_height: render.actual_height,
            x_offset: render.x_offset,
            dropdown_candidates: if render.drop_down {
                screen.dropdown.map(|d| d.candidates.as_slice())
            } else {
                None
            },
            dropdown_selected: screen.dropdown.map_or(0, |d| d.selected_index),
            valid_command_len,
        };

        if let Err(err) = self.terminal.draw(|f| {
            orangu::tui::renderer::render(f, &args);
        }) {
            eprintln!("failed to draw terminal screen: {err}");
        }
    }
}

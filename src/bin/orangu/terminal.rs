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
/// The terminal title whenever orangu is idle: `orangu ★`. The star is the
/// resting state — a run in progress replaces it with a blinking dot (see
/// `auto_review_working_terminal_title`), so a glance at a backgrounded window
/// tells "waiting for you" apart from "still working".
pub(crate) const TERMINAL_TITLE_IDLE: &str = "orangu ★";

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

/// The dot the terminal title blinks with while orangu works, per whether
/// the work is Deep (only `/auto_review` distinguishes this) and which half
/// of the blink this is. A title is set via a plain OSC escape — its text
/// reaches the OS's native window-title widget, not the terminal's own text
/// renderer, so it can't carry ANSI color the way the tab bar's dots do. `◆`
/// vs `●` distinguishes Deep by shape instead, so both render at the same
/// size, and the blink alternates the solid glyph with its hollow twin
/// (`○`/`◇`) rather than dropping the dot altogether, so the title keeps a
/// fixed width and the shape stays readable in either phase.
pub(crate) fn working_title_dot(deep: bool, solid: bool) -> &'static str {
    match (deep, solid) {
        (true, true) => "◆",
        (true, false) => "◇",
        (false, true) => "●",
        (false, false) => "○",
    }
}

/// Whether the working dot is in its solid phase at `elapsed` into the work:
/// one second solid, one second hollow.
pub(crate) fn working_title_solid(elapsed: std::time::Duration) -> bool {
    elapsed.as_secs().is_multiple_of(2)
}

/// Blinks the terminal title between `orangu ●` and `orangu ○` for as long as
/// it lives, so a backgrounded window shows orangu is working (thinking,
/// streaming, or running a command). Created at the start of a wait loop and
/// ticked with the loop's elapsed time; the title is only re-sent when the dot
/// actually changes, and dropping the guard restores the idle `orangu ★`.
pub(crate) struct WorkingTitle {
    dot: Option<&'static str>,
}

impl WorkingTitle {
    /// Start blinking: the title shows the solid dot at once.
    pub(crate) fn new() -> Self {
        let mut title = Self { dot: None };
        title.set(working_title_dot(false, true));
        title
    }

    /// Advance the blink to `elapsed` into the work.
    pub(crate) fn tick(&mut self, elapsed: std::time::Duration) {
        self.set(working_title_dot(false, working_title_solid(elapsed)));
    }

    /// Show `dot` in the title, re-sending the OSC escape only on a change.
    pub(crate) fn set(&mut self, dot: &'static str) {
        if self.dot == Some(dot) {
            return;
        }
        self.dot = Some(dot);
        set_terminal_title(Some(&format!("{TERMINAL_TITLE} {dot}")));
        let _ = std::io::stdout().flush();
    }
}

impl Drop for WorkingTitle {
    fn drop(&mut self) {
        set_terminal_title(Some(TERMINAL_TITLE_IDLE));
        let _ = std::io::stdout().flush();
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
        MOUSE_CAPTURE.store(mouse_capture, Ordering::Relaxed);
        enter_ui_modes()?;
        let backend = CrosstermBackend::new(std::io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            mouse_capture,
        })
    }

    /// Draw one frame. Every screen goes through here so the mouse selection
    /// is painted over whatever was drawn and its cells are kept for the copy
    /// on release.
    pub(crate) fn draw(
        &mut self,
        render: impl FnOnce(&mut ratatui::Frame),
    ) -> std::io::Result<ratatui::CompletedFrame<'_>> {
        // Something else had the screen (the external editor): a blank frame
        // first, so the real one paints every cell rather than the difference
        // from the frame before the editor.
        if REPAINT.swap(false, Ordering::Relaxed) {
            execute!(
                std::io::stdout(),
                crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
            )?;
            self.terminal.draw(|_| {})?;
        }
        self.terminal.draw(|frame| {
            render(frame);
            orangu::tui::selection::finish_frame(frame.buffer_mut());
        })
    }
}

/// What the mouse selection made of an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectionOutcome {
    /// Not the selection's business.
    Untouched,
    /// The selection changed under an event the caller still handles (a press
    /// that dropped the old highlight, a key or scroll that cleared it).
    Changed,
    /// A drag or release: the event was the selection's alone.
    Consumed,
}

impl SelectionOutcome {
    pub(crate) fn consumed(self) -> bool {
        self == Self::Consumed
    }

    pub(crate) fn redraw(self) -> bool {
        self != Self::Untouched
    }
}

/// Feed an event to the mouse selection before the screen's own handling: a
/// left press anchors a new selection, a drag extends it, and the release
/// copies its text to the clipboard. Anything that moves what is under the
/// highlight — a key, the wheel, a resize — drops it. Every event loop calls
/// this first, so selecting works the same on every screen.
pub(crate) fn mouse_selection(event: &Event) -> SelectionOutcome {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    use orangu::tui::selection;

    match event {
        Event::Mouse(MouseEvent {
            kind, column, row, ..
        }) => match kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if selection::press((*column, *row)) {
                    SelectionOutcome::Changed
                } else {
                    SelectionOutcome::Untouched
                }
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                selection::drag((*column, *row));
                SelectionOutcome::Consumed
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some(text) = selection::release() {
                    copy_selection(&text);
                }
                SelectionOutcome::Consumed
            }
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => cleared(),
            _ => SelectionOutcome::Untouched,
        },
        Event::Key(_) | Event::Resize(..) => cleared(),
        _ => SelectionOutcome::Untouched,
    }
}

fn cleared() -> SelectionOutcome {
    if orangu::tui::selection::clear() {
        SelectionOutcome::Changed
    } else {
        SelectionOutcome::Untouched
    }
}

/// Copy selected text to the clipboard and say so on the status line.
pub(crate) fn copy_selection(text: &str) {
    use orangu::tui::selection;

    let notice = match crate::review::copy_to_clipboard(text) {
        Ok(()) => match text.lines().count() {
            1 => "Copied 1 line to the clipboard".to_string(),
            lines => format!("Copied {lines} lines to the clipboard"),
        },
        Err(err) => format!("Could not copy to the clipboard: {err:#}"),
    };
    selection::set_notice(notice);
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        use std::io::Write;
        print!("\x1b]111\x07\x1b]110\x07\x1b]112\x07");
        let _ = std::io::stdout().flush();
        debug_assert_eq!(self.mouse_capture, MOUSE_CAPTURE.load(Ordering::Relaxed));
        leave_ui_modes();
    }
}

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the full-screen UI captures the mouse, for leaving and re-entering
/// its terminal modes around the external editor.
static MOUSE_CAPTURE: AtomicBool = AtomicBool::new(false);
/// The next frame repaints the whole screen.
static REPAINT: AtomicBool = AtomicBool::new(false);

/// Raw mode, the alternate screen, bracketed paste, disambiguated keys and,
/// when configured, mouse capture.
fn enter_ui_modes() -> Result<()> {
    enable_raw_mode()?;
    if MOUSE_CAPTURE.load(Ordering::Relaxed) {
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
        crossterm::event::EnableBracketedPaste,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    Ok(())
}

fn leave_ui_modes() {
    if MOUSE_CAPTURE.load(Ordering::Relaxed) {
        let _ = execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    }
    let _ = execute!(
        std::io::stdout(),
        crossterm::event::DisableBracketedPaste,
        PopKeyboardEnhancementFlags,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
}

/// Open `text` in `$VISUAL`, `$EDITOR` or `vi` on this terminal, as bash's
/// Ctrl+X Ctrl+E does, and return what was saved without its trailing line
/// breaks. `None` when the editor fails or the saved text is blank.
pub(crate) fn edit_text_externally(text: &str) -> Result<Option<String>> {
    let editor = ["VISUAL", "EDITOR"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "vi".to_string());
    let words = crate::commands::shell_words(&editor)?;
    let (program, args) = words
        .split_first()
        .ok_or_else(|| anyhow!("the editor command is empty"))?;

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let path =
        std::env::temp_dir().join(format!("orangu-prompt-{}-{stamp}.md", std::process::id()));
    std::fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))?;

    leave_ui_modes();
    let status = std::process::Command::new(program)
        .args(args)
        .arg(&path)
        .status();
    let restored = enter_ui_modes();
    REPAINT.store(true, Ordering::Relaxed);

    let saved = std::fs::read_to_string(&path);
    let _ = std::fs::remove_file(&path);
    restored?;
    let status = status.with_context(|| format!("failed to run the editor '{program}'"))?;
    if !status.success() {
        return Ok(None);
    }
    let saved = saved.with_context(|| format!("failed to read {}", path.display()))?;
    let saved = saved.trim_end_matches(['\n', '\r']);
    Ok((!saved.trim().is_empty()).then(|| saved.to_string()))
}

/// Events typed out from a bracketed paste, waiting to be read.
static TYPED_PASTE: std::sync::Mutex<std::collections::VecDeque<Event>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

/// `crossterm::event::poll` that also sees a paste still being typed out.
pub(crate) fn poll_typed_event(timeout: std::time::Duration) -> std::io::Result<bool> {
    let queued = TYPED_PASTE.lock().is_ok_and(|queue| !queue.is_empty());
    if queued {
        return Ok(true);
    }
    crossterm::event::poll(timeout)
}

/// Read the next event for a screen that takes keys only: a bracketed paste is
/// typed out one key at a time, each line break as Enter.
pub(crate) fn read_typed_event() -> std::io::Result<Event> {
    if let Some(event) = TYPED_PASTE
        .lock()
        .ok()
        .and_then(|mut queue| queue.pop_front())
    {
        return Ok(event);
    }
    match crossterm::event::read()? {
        Event::Paste(text) => {
            let mut keys = paste_as_keys(&text);
            let first = keys.pop_front();
            if let Ok(mut queue) = TYPED_PASTE.lock() {
                queue.extend(keys);
            }
            match first {
                Some(event) => Ok(event),
                None => read_typed_event(),
            }
        }
        event => Ok(event),
    }
}

fn paste_as_keys(text: &str) -> std::collections::VecDeque<Event> {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    crate::input::pasted_text(text)
        .chars()
        .map(|ch| {
            let code = match ch {
                '\n' => KeyCode::Enter,
                '\t' => KeyCode::Tab,
                ch => KeyCode::Char(ch),
            };
            Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
        })
        .collect()
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
        self.draw_screen(render, screen, true);
    }

    /// Draw the screen for a prompt that is not a command line, such as a
    /// credential prompt: no completion hint and no command highlighting.
    pub fn print_prompt_screen(&mut self, render: RenderContext<'_>, screen: ScreenState<'_>) {
        self.draw_screen(render, screen, false);
    }

    fn draw_screen(&mut self, render: RenderContext<'_>, screen: ScreenState<'_>, hints: bool) {
        // Only hint a completion while the cursor sits at the end of what was typed.
        // Slash commands take priority over natural-language bindings; for the latter,
        // `ghost_index` selects which candidate to preview (cycled with Shift+Tab).
        // Structured argument completions (branches, tags, files, models, servers)
        // fall last, previewing the first candidate Tab would fill in.
        // A running Ctrl+R search owns the hint: the grey text after the line is
        // the history command it found, filled in with Tab, and the status line
        // carries the bash-style search prompt.
        // With nothing typed and nothing to complete, the developer prompt draws
        // its greeting instead. That one is a label rather than a suggestion —
        // it is not in the Tab candidates, so it can never be filled in — which
        // is why it is added here and not in `input_ghost_suffix`.
        let structured_ghost = (hints && screen.reverse_search.is_none())
            .then(|| {
                completion::input_ghost_suffix(
                    screen.input,
                    screen.cursor,
                    screen.ghost_index,
                    render.workspace,
                    render.server_names,
                    render.available_models,
                    render.skills,
                )
                .or_else(|| crate::mode::opening_ghost(screen.input).map(str::to_string))
            })
            .flatten();
        let ghost = match screen.reverse_search {
            Some(search) => search.ghost(screen.input),
            None => structured_ghost.as_deref().unwrap_or(""),
        };
        let mut valid_command_len = 0;
        if hints && screen.input.starts_with('/') {
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
            left_status: screen
                .reverse_search
                .map(|search| StatusFragment::plain(search.prompt(screen.input)))
                .or(screen.left_status),
            pending_count: screen.pending_count,
            pending_lines: screen.pending_lines,
            input: screen.input,
            cursor: screen.cursor,
            ghost,
            virtual_width: render.virtual_width,
            actual_width: render.actual_width,
            actual_height: render.actual_height,
            x_offset: render.x_offset,
            word_wrap: render.word_wrap,
            dropdown_candidates: if render.drop_down {
                screen.dropdown.map(|d| d.candidates.as_slice())
            } else {
                None
            },
            dropdown_selected: screen.dropdown.map_or(0, |d| d.selected_index),
            valid_command_len,
        };

        if let Err(err) = self.draw(|f| {
            orangu::tui::renderer::render(f, &args);
        }) {
            eprintln!("failed to draw terminal screen: {err}");
        }
    }
}

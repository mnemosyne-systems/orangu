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

use rustyline::{
    CompletionType, Config, Context, EditMode, Helper,
    completion::{Completer, FilenameCompleter, Pair},
    highlight::Highlighter,
    hint::Hinter,
    validate::{ValidationContext, ValidationResult, Validator},
};
use std::time::Duration;
use terminal_size::{Height, Width, terminal_size};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptLine {
    Plain(String),
    UserInput(String),
}

impl TranscriptLine {
    pub fn as_str(&self) -> &str {
        match self {
            TranscriptLine::Plain(s) | TranscriptLine::UserInput(s) => s,
        }
    }
}

pub fn editor_config() -> Config {
    Config::builder()
        .history_ignore_space(true)
        .completion_type(CompletionType::Circular)
        .edit_mode(EditMode::Emacs)
        .build()
}

const CLIENT_LOGO_ART: &[&str] = &[
    " ██████  ██████   █████  ███    ██  ██████  ██    ██ ",
    "██    ██ ██   ██ ██   ██ ████   ██ ██       ██    ██ ",
    "██    ██ ██████  ███████ ██ ██  ██ ██   ███ ██    ██ ",
    "██    ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██    ██ ",
    " ██████  ██   ██ ██   ██ ██   ████  ██████   ██████  ",
];
const ORANGU_BROWN: &str = "\x1b[38;2;139;90;43m";
const STATUS_GREEN: &str = "\x1b[38;2;80;200;120m";
const STATUS_RED: &str = "\x1b[38;2;220;80;80m";
const ANSI_RESET: &str = "\x1b[0m";
const USER_INPUT_BACKGROUND: &str = "\x1b[48;2;44;44;44m";
const THINKING_TEXT: &str = "Thinking";
const WORKING_TEXT: &str = "Working";
const THINKING_SHADE_LEVELS: &[u8] = &[230, 210, 190, 170, 150, 130, 110, 90];

#[derive(Debug, Clone, Copy)]
pub struct HeaderStatus {
    pub workspace_ok: bool,
    pub server_ok: bool,
    pub model_ok: bool,
}

pub fn render_header(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    status: HeaderStatus,
) -> String {
    let status_lines = [
        status_text_line(&format!("Version: {version}")),
        status_text_line(""),
        status_indicator_line(
            &format!("Workspace: {}", workspace.display()),
            status.workspace_ok,
        ),
        status_indicator_line(&format!("Server: {endpoint}"), status.server_ok),
        status_indicator_line(&format!("Model: {current_model}"), status.model_ok),
        status_text_line(""),
        status_text_line("Help: /help"),
    ];
    let logo_width = CLIENT_LOGO_ART
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0);
    let status_width = status_lines
        .iter()
        .map(|line| line.visible_width)
        .max()
        .unwrap_or(0);
    let gap_width = 2;
    let width = logo_width + gap_width + status_width;
    let top_border = format!("┏{}┓", "━".repeat(width + 2));
    let bottom_border = format!("┗{}┛", "━".repeat(width + 2));

    let line_count = CLIENT_LOGO_ART.len().max(status_lines.len());
    let mut lines = Vec::with_capacity(line_count + 2);
    lines.push(top_border);

    for index in 0..line_count {
        let logo_line = CLIENT_LOGO_ART.get(index).copied().unwrap_or_default();
        let colored_logo_line = format!("{ORANGU_BROWN}{logo_line}{ANSI_RESET}");
        let status_line = status_lines.get(index).cloned().unwrap_or_default();
        let visible_content_width = logo_line.chars().count()
            + logo_width.saturating_sub(logo_line.chars().count())
            + gap_width
            + status_line.visible_width;
        let content = format!(
            "{}{}{}",
            colored_logo_line,
            " ".repeat(logo_width.saturating_sub(logo_line.chars().count()) + gap_width),
            status_line.rendered
        );
        let padding = width.saturating_sub(visible_content_width);
        lines.push(format!("┃ {content}{} ┃", " ".repeat(padding)));
    }

    lines.push(bottom_border);
    lines.join("\r\n")
}

pub fn help_text() -> &'static str {
    r#"/help                                         Show available commands
/connect [url]                                Connect to the configured server, or a specific server
/disconnect                                   Disconnect from the current server
/reload                                       Restore the configured model and server
/models                                       List models
/list_files                                   List workspace files as a tree
/show_file [--hash] [--author] <path> [<ref>] Show a file; optional ref uses git show
/tools                                        List tools
/model [name]                                 Switch to the configured model, or a specific model
/diff                                         Show a color unified diff against the current branch
/status                                       Show working tree status with color highlighting
/log                                          Show commit log (uses git lg alias if configured)
/pull <number>                                Check out a GitHub pull request on a dedicated branch
/rebase                                       Rebase the current branch against master/main
/merge <branch>                               Merge a branch into the current branch
/checkout <branch|file>                       Switch branch or restore a file
/add_file <path>                              Stage a file or directory with git add
/remove_file <path>                           Remove a file or directory from Git tracking
/move_file <source> <destination>             Rename or move a tracked file with git mv
/cherry_pick <commit>                         Cherry-pick a commit onto the current branch
/commit <message>                             Commit all tracked changes with git commit -a -m
/amend <message>                              Rewrite the last commit message with git commit --amend
/push [--force]                               Push the current branch to origin
/init_repo                                    Initialize a Git repository in the workspace
/squash                                       Squash all branch commits into one
/delete <branch>                              Delete a local branch with git branch -D
/open_file <path>                             Open a workspace file in $EDITOR
/session [uuid]                               List sessions or print resume command for a UUID (Tab cycles UUIDs)
/sessions [workspace]                         List all sessions, optionally filtered by workspace path
/usage                                        Show usage statistics for this session
/clear                                        Clear the current conversation
/quit                                         Exit the client

Natural-language forms such as `open README.md`, `list models`, `list files`, `pull 58`, `log`, `status`, `rebase`, `squash`, `merge feature/foo`, `checkout main`, `add README.md`, `remove README.md`, `move old.rs new.rs`, `cherry pick abc1234`, `commit "[#42] My feature"`, `amend "[#42] My feature"`, `push`, `force push`, `init repo`, `delete feature/foo`, and `show help` are also handled locally.

The prompt uses standard Unix shell keys, including Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+E, Ctrl+K, Ctrl+U, Ctrl+W, Alt+Backspace, Alt+D, and Tab completion."#
}

pub struct ScreenRenderArgs<'a> {
    pub version: &'a str,
    pub current_model: &'a str,
    pub endpoint: &'a str,
    pub workspace: &'a std::path::Path,
    pub prompt_branch: Option<&'a str>,
    pub status: HeaderStatus,
    pub transcript: &'a [TranscriptLine],
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_line: Option<&'a str>,
    pub input: &'a str,
    pub cursor: usize,
}

struct PromptFrameArgs<'a> {
    header_height: usize,
    current_model: &'a str,
    left_status: Option<StatusFragment>,
    pending_count: usize,
    prompt_prefix: &'a str,
    input: &'a str,
    cursor: usize,
    width: usize,
    height: usize,
}

pub fn render_screen(args: ScreenRenderArgs<'_>) -> String {
    let header = render_header(
        args.version,
        args.current_model,
        args.endpoint,
        args.workspace,
        args.status,
    );
    let header_line_count = header.lines().count();
    let width = terminal_width().max(1);
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let input_lines = wrapped_input_lines(args.input, width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let height = terminal_height().max(header_line_count + prompt_frame_height + 1);
    let available_output_rows =
        available_output_rows(header_line_count, prompt_frame_height, height);
    let mut output_lines = args
        .transcript
        .iter()
        .map(|line| render_transcript_line(line, width))
        .collect::<Vec<_>>();
    if let Some(pending_line) = args.pending_line {
        if pending_line.is_empty() {
            output_lines.push(String::new());
        } else {
            output_lines.extend(pending_line.lines().map(ToOwned::to_owned));
        }
    }
    let max_scroll_offset = output_lines.len().saturating_sub(available_output_rows);
    let scroll_offset = args.scroll_offset.min(max_scroll_offset);
    let visible_end = output_lines.len().saturating_sub(scroll_offset);
    let visible_start = visible_end.saturating_sub(available_output_rows);
    let visible_lines = &output_lines[visible_start..visible_end];

    let mut screen = String::new();
    screen.push_str(&header);
    screen.push_str("\r\n\r\n");
    if !visible_lines.is_empty() {
        screen.push_str(&visible_lines.join("\r\n"));
        screen.push_str("\r\n");
    }
    screen.push_str(&render_prompt_frame(PromptFrameArgs {
        header_height: header_line_count,
        current_model: args.current_model,
        left_status: args.left_status,
        pending_count: args.pending_count,
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        width,
        height,
    }));
    screen
}

pub fn output_view_rows(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    prompt_branch: Option<&str>,
    status: HeaderStatus,
    input: &str,
) -> usize {
    let header = render_header(version, current_model, endpoint, workspace, status);
    let header_line_count = header.lines().count();
    let width = terminal_width().max(1);
    let prompt_prefix = prompt_prefix(prompt_branch);
    let input_lines = wrapped_input_lines(input, width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let height = terminal_height().max(header_line_count + prompt_frame_height + 1);

    available_output_rows(header_line_count, prompt_frame_height, height)
}

pub fn render_thinking_status(frame: usize, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(THINKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: THINKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

pub fn render_tool_running_status(frame: usize, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(WORKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: WORKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusFragment {
    pub rendered: String,
    pub visible_width: usize,
}

impl StatusFragment {
    pub fn plain(rendered: String) -> Self {
        let visible_width = rendered.chars().count();
        Self {
            rendered,
            visible_width,
        }
    }
}

pub fn render_working_status(frame: usize, rate: f64, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" @ {rate:.1} t/s {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(WORKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: WORKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

fn format_elapsed_timer(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("({hours}h{minutes}m{seconds}s)")
    } else if minutes > 0 {
        format!("({minutes}m{seconds}s)")
    } else {
        format!("({seconds}s)")
    }
}

fn available_output_rows(
    header_line_count: usize,
    prompt_frame_height: usize,
    height: usize,
) -> usize {
    let output_start_row = header_line_count + 2;
    height
        .saturating_sub(prompt_frame_height)
        .saturating_sub(output_start_row)
        .saturating_add(1)
}

fn indicator(ok: bool) -> String {
    if ok {
        format!("{STATUS_GREEN}●{ANSI_RESET}")
    } else {
        format!("{STATUS_RED}●{ANSI_RESET}")
    }
}

#[derive(Clone, Default)]
struct HeaderLine {
    rendered: String,
    visible_width: usize,
}

fn status_text_line(text: &str) -> HeaderLine {
    HeaderLine {
        rendered: text.to_string(),
        visible_width: text.chars().count(),
    }
}

fn status_indicator_line(text: &str, ok: bool) -> HeaderLine {
    HeaderLine {
        rendered: format!("{text} {}", indicator(ok)),
        visible_width: text.chars().count() + 2,
    }
}

fn render_prompt_frame(args: PromptFrameArgs<'_>) -> String {
    let input_lines = wrapped_input_lines(args.input, args.width, args.prompt_prefix);
    let input_height = input_lines.len();
    let height = args.height.max(args.header_height + input_height + 3);
    let top_row = (height.saturating_sub(input_height + 2)).max(args.header_height + 1);
    let input_start_row = top_row + 1;
    let bottom_row = input_start_row + input_height;
    let model_row = bottom_row + 1;
    let line = "━".repeat(args.width);
    let prompt_width = args.prompt_prefix.chars().count();
    let mut frame = format!("\x1b[{top_row};1H{line}");

    for (index, input_line) in input_lines.iter().enumerate() {
        let row = input_start_row + index;
        let content = truncate_to_width(input_line, args.width.saturating_sub(prompt_width));
        let content_width = content.chars().count();
        frame.push_str(&format!("\x1b[{row};1H{}{}", args.prompt_prefix, content));
        if args.width > content_width + prompt_width {
            frame.push_str(&" ".repeat(args.width - content_width - prompt_width));
        }
    }

    let (cursor_row_offset, cursor_col_offset) =
        cursor_position(args.input, args.cursor, args.width, args.prompt_prefix);
    let cursor_row = input_start_row + cursor_row_offset;
    let cursor_col = 1 + prompt_width + cursor_col_offset;

    let status_line = render_status_line(
        args.width,
        args.left_status.as_ref(),
        args.current_model,
        args.pending_count,
    );
    frame.push_str(&format!(
        "\x1b[{bottom_row};1H{line}\x1b[{model_row};1H{status_line}\x1b[{cursor_row};{cursor_col}H"
    ));
    frame
}

fn render_transcript_line(line: &TranscriptLine, width: usize) -> String {
    match line {
        TranscriptLine::Plain(content) => content.clone(),
        TranscriptLine::UserInput(content) => {
            let padding = width.saturating_sub(content.chars().count());
            format!(
                "{USER_INPUT_BACKGROUND}{content}{}{ANSI_RESET}",
                " ".repeat(padding)
            )
        }
    }
}

fn wrapped_input_lines(input: &str, width: usize, prompt_prefix: &str) -> Vec<String> {
    let input_width = width.saturating_sub(prompt_prefix.chars().count()).max(1);
    if input.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for ch in input.chars() {
        current.push(ch);
        if current.chars().count() == input_width {
            lines.push(current);
            current = String::new();
        }
    }

    if current.is_empty() {
        lines.push(String::new());
    } else {
        lines.push(current);
    }

    lines
}

fn cursor_position(
    input: &str,
    cursor: usize,
    width: usize,
    prompt_prefix: &str,
) -> (usize, usize) {
    let input_width = width.saturating_sub(prompt_prefix.chars().count()).max(1);
    let prefix_chars = input[..cursor.min(input.len())].chars().count();
    (prefix_chars / input_width, prefix_chars % input_width)
}

fn prompt_prefix(branch_name: Option<&str>) -> String {
    match branch_name {
        Some(branch_name) if !branch_name.trim().is_empty() => format!("{branch_name}> "),
        _ => "> ".to_string(),
    }
}

fn render_status_line(
    width: usize,
    left_status: Option<&StatusFragment>,
    current_model: &str,
    pending_count: usize,
) -> String {
    let mut cells = vec![' '; width];
    if pending_count > 0 {
        let pending = format!("Pending: {pending_count}");
        let pending_width = pending.chars().count();
        let pending_start = width.saturating_sub(pending_width) / 2;
        for (index, ch) in pending.chars().enumerate() {
            if pending_start + index < width {
                cells[pending_start + index] = ch;
            }
        }
    }

    let model_width = current_model.chars().count();
    let model_start = width.saturating_sub(model_width);
    for (index, ch) in current_model.chars().enumerate() {
        if model_start + index < width {
            cells[model_start + index] = ch;
        }
    }

    if let Some(left_status) = left_status.filter(|status| status.visible_width > 0) {
        let left_width = left_status.visible_width.min(width);
        let suffix: String = cells.into_iter().skip(left_width).collect();
        format!("{}{}", left_status.rendered, suffix)
    } else {
        cells.into_iter().collect()
    }
}

fn render_rolling_text(text: &str, frame: usize) -> String {
    let mut rendered = String::new();
    let offset = frame % THINKING_SHADE_LEVELS.len();

    for (index, ch) in text.chars().enumerate() {
        let shade_index =
            (index + THINKING_SHADE_LEVELS.len() - offset) % THINKING_SHADE_LEVELS.len();
        let shade = THINKING_SHADE_LEVELS[shade_index];
        rendered.push_str(&format!("\x1b[38;2;{shade};{shade};{shade}m{ch}"));
    }

    rendered
}

fn truncate_to_width(input: &str, width: usize) -> String {
    input.chars().take(width).collect()
}

fn terminal_width() -> usize {
    terminal_size()
        .map(|(Width(width), _)| usize::from(width))
        .filter(|width| *width > 0)
        .or_else(|| {
            std::env::var("COLUMNS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|width| *width > 0)
        })
        .unwrap_or(80)
}

fn terminal_height() -> usize {
    terminal_size()
        .map(|(_, Height(height))| usize::from(height))
        .filter(|height| *height > 0)
        .or_else(|| {
            std::env::var("LINES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|height| *height > 0)
        })
        .unwrap_or(24)
}

pub struct OranguHelper {
    file_completer: FilenameCompleter,
    commands: Vec<String>,
    models: Vec<String>,
}

impl OranguHelper {
    pub fn new(models: Vec<String>) -> Self {
        Self {
            file_completer: FilenameCompleter::new(),
            commands: vec![
                "/help".to_string(),
                "/connect".to_string(),
                "/disconnect".to_string(),
                "/reload".to_string(),
                "/models".to_string(),
                "/list_files".to_string(),
                "/show_file".to_string(),
                "/tools".to_string(),
                "/model".to_string(),
                "/diff".to_string(),
                "/status".to_string(),
                "/log".to_string(),
                "/pull".to_string(),
                "/rebase".to_string(),
                "/merge".to_string(),
                "/checkout".to_string(),
                "/add_file".to_string(),
                "/remove_file".to_string(),
                "/move_file".to_string(),
                "/cherry_pick".to_string(),
                "/commit".to_string(),
                "/amend".to_string(),
                "/push".to_string(),
                "/init_repo".to_string(),
                "/squash".to_string(),
                "/delete".to_string(),
                "/clear".to_string(),
                "/quit".to_string(),
            ],
            models,
        }
    }
}

impl Helper for OranguHelper {}

impl Validator for OranguHelper {
    fn validate(&self, _: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}

impl Highlighter for OranguHelper {}

impl Hinter for OranguHelper {
    type Hint = String;
}

impl Completer for OranguHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if let Some(remainder) = line.strip_prefix("/model ") {
            let prefix = &remainder[..pos.saturating_sub(7)];
            let matches = self
                .models
                .iter()
                .filter(|model| model.starts_with(prefix))
                .map(|model| Pair {
                    display: model.clone(),
                    replacement: model.clone(),
                })
                .collect();
            return Ok((7, matches));
        }

        if line.starts_with('/') {
            let prefix = &line[..pos];
            let matches = self
                .commands
                .iter()
                .filter(|command| command.starts_with(prefix))
                .map(|command| Pair {
                    display: command.clone(),
                    replacement: command.clone(),
                })
                .collect();
            return Ok((0, matches));
        }

        self.file_completer.complete(line, pos, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANSI_RESET, StatusFragment, THINKING_TEXT, TranscriptLine, USER_INPUT_BACKGROUND,
        WORKING_TEXT, available_output_rows, prompt_prefix, render_status_line,
        render_thinking_status, render_transcript_line, render_working_status, wrapped_input_lines,
    };
    use std::time::Duration;

    #[test]
    fn thinking_status_rolls_and_formats_elapsed() {
        let frame_zero = render_thinking_status(0, Duration::from_secs(61));
        let frame_one = render_thinking_status(1, Duration::from_secs(61));

        assert!(frame_zero.rendered.contains('T'));
        assert!(frame_zero.rendered.contains('g'));
        assert_ne!(frame_zero.rendered, frame_one.rendered);
        assert!(frame_zero.rendered.contains("(1m1s)"));
        assert!(frame_one.rendered.contains("(1m1s)"));
        assert_eq!(
            frame_zero.visible_width,
            THINKING_TEXT.chars().count() + " (1m1s)".chars().count()
        );
        for ch in THINKING_TEXT.chars() {
            assert!(frame_zero.rendered.contains(ch));
        }
    }

    #[test]
    fn working_status_rolls_and_formats_rate() {
        let frame_zero = render_working_status(0, 42.5, Duration::from_secs(65));
        let frame_one = render_working_status(1, 42.5, Duration::from_secs(65));

        assert!(frame_zero.rendered.contains("42.5 t/s"));
        assert!(frame_zero.rendered.contains("(1m5s)"));
        assert_eq!(
            frame_zero.visible_width,
            WORKING_TEXT.chars().count() + " @ 42.5 t/s (1m5s)".chars().count()
        );
        assert_ne!(frame_zero.rendered, frame_one.rendered);
    }

    #[test]
    fn prompt_prefix_uses_branch_name() {
        assert_eq!(prompt_prefix(Some("main")), "main> ");
        assert_eq!(prompt_prefix(None), "> ");
    }

    #[test]
    fn wrapped_input_lines_respect_prompt_width() {
        assert_eq!(
            wrapped_input_lines("abc", 8, "main> "),
            vec!["ab".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn status_line_centers_pending_count() {
        let line = render_status_line(
            30,
            Some(&StatusFragment::plain("2.5t/s".to_string())),
            "gpt-4.1",
            3,
        );
        assert!(line.starts_with("2.5t/s"));
        assert!(line.contains("Pending: 3"));
        assert!(line.ends_with("gpt-4.1"));
    }

    #[test]
    fn available_output_rows_matches_current_layout_math() {
        assert_eq!(available_output_rows(8, 4, 24), 11);
    }

    #[test]
    fn transcript_input_highlight_fills_the_row() {
        let rendered =
            render_transcript_line(&TranscriptLine::UserInput("> Hello World!".to_string()), 20);

        assert_eq!(
            rendered,
            format!("{USER_INPUT_BACKGROUND}> Hello World!      {ANSI_RESET}")
        );
    }
}

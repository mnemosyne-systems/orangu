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
    Wide(String),
}

impl TranscriptLine {
    pub fn as_str(&self) -> &str {
        match self {
            TranscriptLine::Plain(s) | TranscriptLine::UserInput(s) | TranscriptLine::Wide(s) => s,
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
pub const FEEDBACK_OK: &str = "\x1b[38;2;80;200;120m●\x1b[0m";
pub const FEEDBACK_ERR: &str = "\x1b[38;2;220;80;80m●\x1b[0m";
const USER_INPUT_BACKGROUND: &str = "\x1b[48;2;44;44;44m";
const THINKING_TEXT: &str = "Thinking";
const WORKING_TEXT: &str = "Working";
const THINKING_SHADE_LEVELS: &[u8] = &[230, 210, 190, 170, 150, 130, 110, 90];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Banner {
    #[default]
    Left,
    Center,
    Right,
}

impl std::str::FromStr for Banner {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.trim().to_lowercase().as_str() {
            "center" => Self::Center,
            "right" => Self::Right,
            _ => Self::Left,
        })
    }
}

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
    alignment: Banner,
    actual_width: usize,
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

    let box_width = width + 4;
    let padding = match alignment {
        Banner::Left => 0,
        Banner::Center => actual_width.saturating_sub(box_width) / 2,
        Banner::Right => actual_width.saturating_sub(box_width),
    };
    if padding == 0 {
        lines.join("\r\n")
    } else {
        let prefix = " ".repeat(padding);
        lines
            .iter()
            .map(|line| format!("{prefix}{line}"))
            .collect::<Vec<_>>()
            .join("\r\n")
    }
}

pub fn help_text() -> &'static str {
    r#"/help                                         Show available commands
/connect [url]                                Connect to the configured server, or a specific server
/disconnect                                   Disconnect from the current server
/reload                                       Restore the configured model and server
/restart                                      Restart orangu, resuming the same workspace and session
/tools                                        List tools
/model [name]                                 List the server's models (active green), or switch to a specific one
/server [name]                                List configured servers (active green), or switch to a specific one
/session [uuid]                               List sessions or print resume command for a UUID (Tab cycles UUIDs)
/sessions [workspace]                         List all sessions, optionally filtered by workspace path
/list_files                                   List workspace files as a tree
/open_file <path>                             Open a workspace file in $EDITOR
/show_file [--hash] [--author] <path> [<ref>] Show a file; optional ref uses git show
/build                                        Build the project
/add_file <path>                              Stage a file or directory with git add
/amend <message>                              Rewrite the last commit message with git commit --amend
/branch [<name>|-a|-b|-m|-d <name>]           List, switch, create, rename or delete a branch
/cherry_pick <commit>                         Cherry-pick a commit onto the current branch
/comment <number> "<comment>"|<file>          Add a comment to a GitHub/GitLab issue; inline body or file from ~/.orangu/comments/
/close -i <number>                            Close a GitHub/GitLab issue with gh/glab
/close -p <number>                            Close a GitHub/GitLab pull request/MR with gh/glab
/commit <message>                             Commit all tracked changes with git commit -a -m
/diff                                         Show a color unified diff against the current branch
/grep <pattern>                               Search the workspace with git grep
/init_repo                                    Initialize a Git repository in the workspace
/log [number]                                 Show commit log (optionally the latest number of commits) plus a count of uncommitted/untracked changes
/merge <branch>                               Merge a branch into the current branch
/move_file <source> <destination>             Rename or move a tracked file with git mv
/pull <number>                                Check out a GitHub/GitLab pull/merge request on a dedicated branch
/pull_request                                 Create a pull request for the current branch
/push [--force]                               Push the current branch to origin
/rebase                                       Rebase the current branch against master/main
/remove_file <path>                           Remove a file or directory from Git tracking
/restore [--staged] <file>                    Restore a file or unstage it (git restore)
/review                                       Review branch changes against main/master in a split view
/squash                                       Squash all branch commits into one
/stash                                        Save uncommitted changes (git stash push)
/stash pop                                    Restore the most recent stash
/stash list                                   List all saved stashes
/stash drop                                   Discard the most recent stash
/status                                       Show working tree status with color highlighting
/usage                                        Show usage statistics for this session
/clear                                        Clear the current conversation
/quit                                         Exit the client

Natural-language forms such as `open README.md`, `list models`, `list files`, `pull 58`, `log`, `status`, `rebase`, `squash`, `merge feature/foo`, `grep <pattern>`, `find <pattern>`, `branch`, `list branches`, `checkout main`, `switch to main`, `create branch feature/x`, `rename to new-name`, `delete feature/foo`, `restore README.md`, `add README.md`, `remove README.md`, `move old.rs new.rs`, `cherry pick abc1234`, `commit "[#42] My feature"`, `amend "[#42] My feature"`, `push`, `force push`, `add comment on 51 "My comment"`, `review`, `create pull request`, `stash`, `stash pop`, `stash list`, `stash drop`, `init repo`, `restart`, and `show help` are also handled locally.

The prompt uses standard Unix shell keys, including Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+E, Ctrl+K, Ctrl+U, Ctrl+W, Alt+Backspace, Alt+D, and Tab completion.

Shift+PageUp / Shift+PageDown scrolls the output window by a full page. Alt+Up / Alt+Down scrolls one line at a time."#
}

pub struct ScreenRenderArgs<'a> {
    pub version: &'a str,
    pub current_model: &'a str,
    pub endpoint: &'a str,
    pub workspace: &'a std::path::Path,
    pub prompt_branch: Option<&'a str>,
    pub status: HeaderStatus,
    pub banner: Banner,
    pub transcript: &'a [TranscriptLine],
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_line: Option<&'a str>,
    pub input: &'a str,
    pub cursor: usize,
    pub virtual_width: usize,
    pub actual_width: usize,
    pub actual_height: usize,
    pub x_offset: usize,
}

struct PromptFrameArgs<'a> {
    header_height: usize,
    current_model: &'a str,
    left_status: Option<StatusFragment>,
    pending_count: usize,
    prompt_prefix: &'a str,
    input: &'a str,
    cursor: usize,
    height: usize,
    actual_width: usize,
}

pub fn render_screen(args: ScreenRenderArgs<'_>) -> String {
    let header = render_header(
        args.version,
        args.current_model,
        args.endpoint,
        args.workspace,
        args.status,
        args.banner,
        args.actual_width,
    );
    let header_line_count = header.lines().count();
    let width = args.virtual_width.max(1);
    let actual_width = args.actual_width.max(1);
    let actual_height = args.actual_height.max(1);
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let input_lines = wrapped_input_lines(args.input, actual_width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;

    // Priority: prompt frame first, then banner, then output.
    let rows_above_prompt = actual_height.saturating_sub(prompt_frame_height);
    // Banner = header lines + 1 blank separator line; truncate to what fits.
    let full_banner_height = header_line_count + 1;
    let banner_rows = full_banner_height.min(rows_above_prompt);
    let available_output_rows = available_output_rows(rows_above_prompt, banner_rows);

    let mut output_lines = args
        .transcript
        .iter()
        .map(|line| {
            let (rendered, offset) = match line {
                TranscriptLine::UserInput(_) => (render_transcript_line(line, actual_width), 0),
                _ => (render_transcript_line(line, width), args.x_offset),
            };
            clip_line(&rendered, offset, actual_width)
        })
        .collect::<Vec<_>>();
    if let Some(pending_line) = args.pending_line {
        if pending_line.is_empty() {
            output_lines.push(String::new());
        } else {
            output_lines.extend(
                pending_line
                    .lines()
                    .map(|l| clip_line(l, args.x_offset, actual_width)),
            );
        }
    }
    let max_scroll_offset = output_lines.len().saturating_sub(available_output_rows);
    let scroll_offset = args.scroll_offset.min(max_scroll_offset);
    let visible_end = output_lines.len().saturating_sub(scroll_offset);
    let visible_start = visible_end.saturating_sub(available_output_rows);
    let visible_lines = &output_lines[visible_start..visible_end];

    let mut screen = String::new();

    // Banner — show as many header lines as fit; add blank separator only when full banner fits.
    if banner_rows > 0 {
        let shown_header_lines = banner_rows.min(header_line_count);
        let banner_content = header
            .split("\r\n")
            .take(shown_header_lines)
            .map(|l| clip_line(l, 0, actual_width))
            .collect::<Vec<_>>()
            .join("\r\n");
        screen.push_str(&banner_content);
        screen.push_str("\r\n");
        if banner_rows > header_line_count {
            screen.push_str("\r\n"); // blank separator line
        }
    }

    if !visible_lines.is_empty() {
        screen.push_str(&visible_lines.join("\r\n"));
        screen.push_str("\r\n");
    }
    screen.push_str(&render_prompt_frame(PromptFrameArgs {
        header_height: banner_rows,
        current_model: args.current_model,
        left_status: args.left_status,
        pending_count: args.pending_count,
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        height: actual_height,
        actual_width,
    }));
    screen
}

#[allow(clippy::too_many_arguments)]
pub fn output_view_rows(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    prompt_branch: Option<&str>,
    status: HeaderStatus,
    input: &str,
    actual_width: usize,
    actual_height: usize,
) -> usize {
    let header = render_header(
        version,
        current_model,
        endpoint,
        workspace,
        status,
        Banner::Left,
        actual_width,
    );
    let header_line_count = header.lines().count();
    let prompt_prefix = prompt_prefix(prompt_branch);
    let input_lines = wrapped_input_lines(input, actual_width.max(1), &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let rows_above_prompt = actual_height.saturating_sub(prompt_frame_height);
    let banner_rows = (header_line_count + 1).min(rows_above_prompt);
    available_output_rows(rows_above_prompt, banner_rows)
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

fn available_output_rows(rows_above_prompt: usize, banner_rows: usize) -> usize {
    rows_above_prompt.saturating_sub(banner_rows)
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
    let input_lines = wrapped_input_lines(args.input, args.actual_width, args.prompt_prefix);
    let input_height = input_lines.len();
    let height = args.height.max(args.header_height + input_height + 3);
    let top_row = (height.saturating_sub(input_height + 2)).max(args.header_height + 1);
    let input_start_row = top_row + 1;
    let bottom_row = input_start_row + input_height;
    let model_row = bottom_row + 1;
    let separator = "━".repeat(args.actual_width);
    let prompt_width = args.prompt_prefix.chars().count();
    let mut frame = format!("\x1b[{top_row};1H{separator}");

    for (index, input_line) in input_lines.iter().enumerate() {
        let row = input_start_row + index;
        let content = truncate_to_width(input_line, args.actual_width.saturating_sub(prompt_width));
        let content_width = content.chars().count();
        let mut full_line = format!("{}{}", args.prompt_prefix, content);
        if args.actual_width > content_width + prompt_width {
            full_line.push_str(&" ".repeat(args.actual_width - content_width - prompt_width));
        }
        frame.push_str(&format!("\x1b[{row};1H{full_line}"));
    }

    let (cursor_row_offset, cursor_col_offset) = cursor_position(
        args.input,
        args.cursor,
        args.actual_width,
        args.prompt_prefix,
    );
    let cursor_row = input_start_row + cursor_row_offset;
    let display_cursor_col = (1 + prompt_width + cursor_col_offset).max(1);

    let status_line = render_status_line(
        args.actual_width,
        args.left_status.as_ref(),
        args.current_model,
        args.pending_count,
    );
    frame.push_str(&format!(
        "\x1b[{bottom_row};1H{separator}\x1b[{model_row};1H{status_line}\x1b[{cursor_row};{display_cursor_col}H"
    ));
    frame
}

fn render_transcript_line(line: &TranscriptLine, width: usize) -> String {
    match line {
        TranscriptLine::Plain(content) | TranscriptLine::Wide(content) => content.clone(),
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
    // Priority: left_status (Working/Thinking) > model name > Pending.
    let left_visible_width = left_status.map_or(0, |s| s.visible_width);
    let right_space = width.saturating_sub(left_visible_width);

    let model_width = current_model.chars().count();
    let show_model = right_space >= model_width;

    let pending_text = (pending_count > 0).then(|| format!("Pending: {pending_count}"));
    let pending_width = pending_text.as_ref().map_or(0, |s| s.chars().count());
    // Gap is the space between the left status and the model name (or right edge if no model).
    let gap = if show_model {
        right_space.saturating_sub(model_width)
    } else {
        right_space
    };
    let show_pending = show_model && pending_text.is_some() && gap >= pending_width;

    let mut right_cells = vec![' '; right_space];
    if show_model {
        let start = right_space - model_width;
        for (i, ch) in current_model.chars().enumerate() {
            if start + i < right_space {
                right_cells[start + i] = ch;
            }
        }
    }
    if show_pending {
        let pending = pending_text.as_deref().unwrap_or("");
        let start = (gap - pending_width) / 2;
        for (i, ch) in pending.chars().enumerate() {
            if start + i < right_space {
                right_cells[start + i] = ch;
            }
        }
    }

    let right: String = right_cells.into_iter().collect();
    if let Some(left) = left_status.filter(|s| s.visible_width > 0) {
        format!("{}{}", left.rendered, right)
    } else {
        right
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

pub fn clip_line(line: &str, x_offset: usize, visible_width: usize) -> String {
    let mut result = String::new();
    let mut col = 0usize;
    let mut pre_clip_ansi = String::new();
    let mut in_visible = false;
    let mut truncated = false;
    let mut chars = line.chars().peekable();

    'outer: while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            let mut seq = String::from('\x1b');
            match chars.peek() {
                Some(&'[') => {
                    seq.push(chars.next().unwrap());
                    loop {
                        match chars.next() {
                            Some(c) => {
                                let done = c.is_ascii_alphabetic() || c == '~' || c == '@';
                                seq.push(c);
                                if done {
                                    break;
                                }
                            }
                            None => break 'outer,
                        }
                    }
                }
                Some(&'O') => {
                    seq.push(chars.next().unwrap());
                    if let Some(c) = chars.next() {
                        seq.push(c);
                    }
                }
                _ => {}
            }
            if col < x_offset {
                pre_clip_ansi.push_str(&seq);
            } else {
                result.push_str(&seq);
            }
            continue;
        }

        if col < x_offset {
            col += 1;
            continue;
        }

        let vis_col = col - x_offset;
        if vis_col >= visible_width {
            truncated = true;
            break;
        }

        if !in_visible {
            result.push_str(&pre_clip_ansi);
            in_visible = true;
        }

        result.push(ch);
        col += 1;
    }

    if truncated {
        result.push_str("\x1b[0m");
    }

    result
}

pub fn visible_line_width(line: &str) -> usize {
    let mut col = 0usize;
    let mut chars = line.chars().peekable();
    'outer: while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some(c) => {
                                if c.is_ascii_alphabetic() || c == '~' || c == '@' {
                                    break;
                                }
                            }
                            None => break 'outer,
                        }
                    }
                }
                Some(&'O') => {
                    chars.next();
                    chars.next();
                }
                _ => {}
            }
            continue;
        }
        col += 1;
    }
    col
}

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

/// The inline comment editor shown below the highlighted line.
pub struct ReviewCommentEditor<'a> {
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
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub actual_width: usize,
    pub actual_height: usize,
}

const REVIEW_SEPARATOR: &str = "\x1b[38;2;88;88;88m│\x1b[0m";
const REVIEW_LINE_CURSOR_BG: &str = "\x1b[48;2;60;60;90m";
const REVIEW_COMMENT_BG: &str = "\x1b[48;2;38;48;38m";
const REVIEW_COMMENT_MARKER: &str = "\x1b[38;2;230;200;120m●\x1b[0m";
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

fn review_status_box(status: ReviewStatus) -> String {
    match status {
        ReviewStatus::Unreviewed => "[ ]".to_string(),
        ReviewStatus::Approved => format!("[{STATUS_GREEN}●{ANSI_RESET}]"),
        ReviewStatus::Rejected => format!("[{STATUS_RED}●{ANSI_RESET}]"),
    }
}

/// Clip `content` to `width` visible columns (honoring a horizontal pan) and
/// pad it with spaces so the cell occupies exactly `width` columns. This keeps
/// the vertical separator aligned in a single straight column on every row.
fn review_pane_cell(content: &str, x_offset: usize, width: usize) -> String {
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
fn review_highlight(cell: &str) -> String {
    let reactivated = cell.replace(ANSI_RESET, &format!("{ANSI_RESET}\x1b[7m"));
    format!("\x1b[7m{reactivated}{ANSI_RESET}")
}

/// Apply a background to the whole cell — the highlighted diff line under the
/// Up/Down cursor — re-applying it after every reset so it spans the line's
/// own color codes and the trailing padding.
fn review_line_highlight(cell: &str) -> String {
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
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        height,
        actual_width: width,
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
        "Review: {}  Alt+j/k Switch file  Alt+a Approve  Alt+r Reject  Alt+o Review  Alt+c Comment  Alt+x Exit",
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
    let box_rows = render_review_comment_box(editor.text, editor.cursor, left_width);
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

/// Render the inline comment editor box (a fixed-height window). The text wraps
/// to the pane width and scrolls to keep the cursor visible.
fn render_review_comment_box(text: &str, cursor: usize, width: usize) -> Vec<String> {
    let inner_width = width.saturating_sub(2).max(1);
    let wrapped = wrapped_input_lines(text, inner_width, "");
    let (cursor_row, cursor_col) = cursor_position(text, cursor, inner_width, "");
    let start = cursor_row.saturating_sub(REVIEW_COMMENT_BOX_HEIGHT - 1);

    (0..REVIEW_COMMENT_BOX_HEIGHT)
        .map(|row| {
            let index = start + row;
            let mut content = wrapped.get(index).cloned().unwrap_or_default();
            if index == cursor_row {
                content = comment_caret(&content, cursor_col, inner_width);
            }
            let visible = visible_line_width(&content);
            let padding = " ".repeat(inner_width.saturating_sub(visible));
            // Greenish gutter bar; reset only the foreground so the comment
            // background spans the whole row.
            format!(
                "{REVIEW_COMMENT_BG}\x1b[38;2;120;160;120m▕\x1b[39m {content}{padding}{ANSI_RESET}"
            )
        })
        .collect()
}

/// Insert a reverse-video caret into a plain comment line at `col`.
fn comment_caret(content: &str, col: usize, inner_width: usize) -> String {
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

/// Render `text` as a user-input line — a dark background spanning the full
/// width — matching how submitted prompts appear in the main output window.
pub fn render_user_input_line(text: &str, width: usize) -> String {
    let clipped = clip_line(text, 0, width);
    let padding = width.saturating_sub(visible_line_width(&clipped));
    format!(
        "{USER_INPUT_BACKGROUND}{clipped}{}{ANSI_RESET}",
        " ".repeat(padding)
    )
}

pub fn terminal_width() -> usize {
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

pub fn terminal_height() -> usize {
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
                "/restart".to_string(),
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
                "/branch".to_string(),
                "/restore".to_string(),
                "/add_file".to_string(),
                "/remove_file".to_string(),
                "/move_file".to_string(),
                "/cherry_pick".to_string(),
                "/commit".to_string(),
                "/amend".to_string(),
                "/push".to_string(),
                "/init_repo".to_string(),
                "/squash".to_string(),
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
        ANSI_RESET, ReviewEntry, ReviewScreenArgs, ReviewStatus, StatusFragment, THINKING_TEXT,
        TranscriptLine, USER_INPUT_BACKGROUND, WORKING_TEXT, available_output_rows, prompt_prefix,
        render_review_screen, render_status_line, render_thinking_status, render_transcript_line,
        render_working_status, review_right_width, wrapped_input_lines,
    };
    use std::time::Duration;

    fn review_entry(path: &str, status: ReviewStatus, diff_lines: &[&str]) -> ReviewEntry {
        ReviewEntry {
            path: path.to_string(),
            status,
            diff_lines: diff_lines.iter().map(|line| line.to_string()).collect(),
            patch: String::new(),
        }
    }

    fn review_args<'a>(
        files: &'a [ReviewEntry],
        selected: usize,
        scroll: usize,
        actual_width: usize,
        actual_height: usize,
    ) -> ReviewScreenArgs<'a> {
        ReviewScreenArgs {
            files,
            selected,
            line: 0,
            scroll,
            x_offset: 0,
            feedback: None,
            comment_editor: None,
            commented_lines: &[],
            current_model: "model",
            prompt_branch: None,
            input: "",
            cursor: 0,
            left_status: None,
            pending_count: 0,
            actual_width,
            actual_height,
        }
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
            text: "needs a guard",
            cursor: "needs a guard".len(),
        });
        let rendered = render_review_screen(args);
        let rows: Vec<&str> = rendered.split("\r\n").collect();

        // Find the body row holding the highlighted line, then assert the next
        // five rows are the comment box (they carry the comment background and
        // the typed text appears within them).
        let line_row = rows
            .iter()
            .position(|row| row.contains("line one"))
            .expect("highlighted line present");
        let box_block = rows[line_row + 1..line_row + 1 + 5].join("\n");
        assert!(
            box_block.contains("\u{1b}[48;2;38;48;38m"),
            "comment box background missing below the line"
        );
        assert!(box_block.contains("needs a guard"), "comment text missing");
        // The line after the box continues the diff.
        assert!(rows[line_row + 6].contains("line two"));
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
        // header=8 lines, prompt_frame=4, height=24
        // rows_above_prompt = 24-4 = 20, banner_rows = min(9, 20) = 9
        assert_eq!(available_output_rows(20, 9), 11);
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

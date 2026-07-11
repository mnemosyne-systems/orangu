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

const CLIENT_LOGO_ART: &[&str] = &[
    " ██████  ██████   █████  ███    ██  ██████  ██    ██ ",
    "██    ██ ██   ██ ██   ██ ████   ██ ██       ██    ██ ",
    "██    ██ ██████  ███████ ██ ██  ██ ██   ███ ██    ██ ",
    "██    ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██    ██ ",
    " ██████  ██   ██ ██   ██ ██   ████  ██████   ██████  ",
];
const ORANGU_BROWN: &str = "\x1b[38;2;139;90;43m";

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

/// Tri-state connectivity indicator for the header's `Server`/`Model` rows:
/// `Pending` (white dot) while a check is still in flight — most notably at
/// startup, so the UI can render instantly instead of blocking on the first
/// probe — `Ok` (green) once confirmed good, `Failed` (red) once confirmed
/// bad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnStatus {
    Pending,
    Ok,
    Failed,
}

impl ConnStatus {
    pub fn from_bool(ok: bool) -> Self {
        if ok { Self::Ok } else { Self::Failed }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HeaderStatus {
    pub workspace_ok: bool,
    pub server_ok: ConnStatus,
    pub model_ok: ConnStatus,
    /// Whether the active server identified itself as an orangu-coordinator
    /// proxy (`GET /v1/coordinator`) rather than llama.cpp or a generic
    /// OpenAI-compatible server. When set, the header shows "Automatic" for
    /// the model instead of the configured wire model id, since the
    /// coordinator — not the user — decides which model is actually loaded.
    pub is_coordinator: bool,
}

/// The model name to display anywhere in the UI: under a confirmed
/// coordinator, the wire model id pinned here isn't a meaningful "what's
/// loaded" answer (the coordinator — not the user — decides which model is
/// actually loaded), so "Automatic" is shown instead; otherwise `model_id`
/// itself.
pub fn display_model_name(is_coordinator: bool, model_id: &str) -> &str {
    if is_coordinator {
        "Automatic"
    } else {
        model_id
    }
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
    let current_model = display_model_name(status.is_coordinator, current_model);
    let status_lines = [
        status_text_line(&format!("Version: {version}")),
        status_text_line(""),
        bool_status_indicator_line(
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
    r#"/help                                           Show available commands
/server [name]                                  List configured servers (active green), or switch to a specific one
/disconnect                                     Disconnect from the current server
/reload                                         Restore the configured model and server
/restart                                        Restart orangu, resuming the same workspace and session
/tools                                          List tools
/model [name]                                   List the server's models (active green), or switch to a specific one
/information                                    Report which OpenAI/llama.cpp server capabilities are available and enabled
/prune [<uuid>|-w <path>|-o <days>|all]         Remove sessions
/session [uuid|workspace]                       List/switch sessions, or open a workspace directory (Tab completes UUIDs, workspaces, then filesystem paths)
/workspace [number|path]                        Show the active workspace, switch to a tab by number, or open a directory (Tab completes workspaces, then filesystem paths)
/create_workspace <dir>                         Open a new workspace tab on an existing directory (like Alt+Insert + /workspace)
/delete_workspace                               Close the active workspace tab (like Alt+Delete)
/list_files                                     List workspace files as a tree
/open_file <path>                               Open a workspace file in $EDITOR
/show_file [--hash] [--author] <path> [<ref>]   Show a file; optional ref uses git show
/build [debug|release] [<target>]               Build the project (default: release), optionally one target (a Makefile rule, cargo binary, Maven goal, ...; Tab completes discovered targets)
/shell <command>                                Run a shell command in the workspace, streaming its output (Tab completes workspace files one path segment at a time)
/export [console|review|auto review|duplicates|pr|statistics] Export the output window (console), the last review report (review), the last auto-review report (auto review), a fresh duplicate-code report (duplicates), open pull requests (pr), or persistent activity history (statistics) to a PDF in the workspace root
/duplicates [<threshold>]                       Scan source functions (Rust, C/C++, Go, Python, JS/TS, and ~20 more languages) for duplicates; on a non-default branch only the functions it adds/changes are compared against the project; optional threshold is a percentage (default 80%)
/add_file <path>                                Stage a file or directory with git add
/auto_review [<file>|all] [immediate] [deep]    LLM auto review in a split view: the whole branch, one Tab-completed file (the full file on main/master, its changes on a branch), or every Git-tracked project file with all; add immediate to start the run at once; add deep to start every file in Deep mode (no diff truncation, cross-file graph context, a verify pass on rejects)
/amend <message>                                Rewrite the last commit message with git commit --amend
/bisect [start|good|bad|skip|reset|log]         Binary-search history for the commit that introduced a bug (git bisect); bare /bisect shows the session status
/branch [<name>|-a|-b|-m|-d <name>]             List, switch, create, rename or delete a branch
/cherry_pick <commit>                           Cherry-pick a commit onto the current branch
/comment <number> "<comment>"|<file>            Add a comment to a GitHub/GitLab issue; inline body, file from ~/.orangu/comments/, or `with [auto] review` to post the last /review or /auto_review report
/close -i <number>|-p <number>                  Close a GitHub/GitLab issue or pull request with gh/glab
/commit <message>                               Commit all tracked changes with git commit -a -m
/diff                                           Show a color unified diff against the current branch
/fetch [remote]                                 Fetch from a remote with git fetch (Tab completes git remotes; defaults to the first remote)
/get_comments -i <number>|-p <number>           List comments on a GitHub/GitLab issue or pull request with gh/glab
/grep <pattern>                                 Search the workspace with git grep
/init_repo                                      Initialize a Git repository in the workspace
/log [number]                                   Show commit log (optionally the latest number of commits) plus a count of uncommitted/untracked changes
/merge <branch>                                 Merge a branch into the current branch
/move_file <source> <destination>               Rename or move a tracked file with git mv
/pending [delete <n>]                           List queued commands, or delete one by number
/pull <number>                                  Check out a GitHub/GitLab pull/merge request on a dedicated branch
/pull_request                                   Create a pull request for the current branch
/push [--force]                                 Push the current branch to origin
/rebase [target]                                Rebase the current branch onto master/main, or onto a given target (Tab completes local branches, then remotes, then remote branches)
/remove_file <path>                             Remove a file or directory from Git tracking
/restore [--staged] <file>                      Restore a file or unstage it (git restore)
/review                                         Review branch changes against main/master in a split view
/show [<commit>]                                Show a single commit (its header and diff) with git show; defaults to HEAD (Tab completes the latest 25 commits)
/squash                                         Squash all branch commits into one
/stash [pop|list|drop]                          Save uncommitted changes (git stash push), restore, list or discard
/status                                         Show working tree status with color highlighting
/manual                                         Open the built-in manual in a full-screen viewer
/usage                                          Show usage statistics for this session
/statistics [total]                             Show persistent, cross-session activity history and a daily heatmap for this workspace, or every workspace (total)
/schedule                                       List the cron-style scheduled jobs from ~/.orangu/schedule with their next run times (UTC)
/skills                                         List discovered Agent Skills; invoke one with /skill-name
/clear                                          Clear the current conversation
/quit                                           Exit the client

Natural-language forms such as `open README.md`, `list models`, `show information`, `list files`, `pull 58`, `log`, `git show abc1234`, `fetch`, `fetch upstream`, `status`, `rebase`, `rebase origin/main`, `squash`, `merge feature/foo`, `grep <pattern>`, `find <pattern>`, `branch`, `list branches`, `checkout main`, `switch to main`, `create branch feature/x`, `rename to new-name`, `delete feature/foo`, `restore README.md`, `add README.md`, `remove README.md`, `move old.rs new.rs`, `cherry pick abc1234`, `commit "[#42] My feature"`, `amend "[#42] My feature"`, `push`, `force push`, `add comment on 51 "My comment"`, `comment on 48 with review`, `comment on 48 with auto review`, `get comments for issue 51`, `get comments for pull request 58`, `review`, `auto review`, `find duplicates`, `export console`, `export review`, `export auto review`, `export duplicates`, `export statistics`, `show statistics`, `show schedule`, `create pull request`, `stash`, `stash pop`, `stash list`, `stash drop`, `bisect start`, `mark good`, `mark bad`, `bisect reset`, `init repo`, `prune session <uuid>`, `prune all`, `prune sessions older than <days>`, `prune sessions in <path>`, `restart`, `pending`, `workspace`, `workspace 1`, `switch workspace ~/project`, `create workspace ~/project`, `delete workspace`, `show manual`, and `show help` are also handled locally.

The prompt uses standard Unix shell keys, including Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+E, Ctrl+K, Ctrl+U, Ctrl+W, Alt+Backspace, Alt+D, and Tab completion.

As you type, a grey inline hint previews the matching command (e.g. `q` suggests `quit`). Press Tab to accept it. When several commands match, Shift+Tab cycles the hint through them; Tab then accepts the one shown.

Shift+PageUp / Shift+PageDown scrolls the output window by a full page. Alt+Up / Alt+Down scrolls one line at a time.

/manual opens the built-in manual in a full-screen viewer: the text on the left, the table of contents on the right. Alt+J/Alt+K switch sections, Up/Down move the highlighted line, Alt+S opens a search window over the entire manual (Enter jumps to the next match, Esc closes it), Alt+Up/Alt+Down scroll, PageUp/PageDown page, Left/Right pan, and Alt+X (or Esc Esc) exits."#
}

pub(crate) fn indicator(status: ConnStatus) -> String {
    match status {
        ConnStatus::Pending => format!("{STATUS_WHITE}●{ANSI_RESET}"),
        ConnStatus::Ok => format!("{STATUS_GREEN}●{ANSI_RESET}"),
        ConnStatus::Failed => format!("{STATUS_RED}●{ANSI_RESET}"),
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

fn status_indicator_line(text: &str, status: ConnStatus) -> HeaderLine {
    HeaderLine {
        rendered: format!("{text} {}", indicator(status)),
        visible_width: text.chars().count() + 2,
    }
}

fn bool_status_indicator_line(text: &str, ok: bool) -> HeaderLine {
    status_indicator_line(text, ConnStatus::from_bool(ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(is_coordinator: bool) -> HeaderStatus {
        HeaderStatus {
            workspace_ok: true,
            server_ok: ConnStatus::Ok,
            model_ok: ConnStatus::Ok,
            is_coordinator,
        }
    }

    #[test]
    fn display_model_name_shows_automatic_only_when_coordinator() {
        assert_eq!(
            display_model_name(true, "ggml-org/gemma-4-E4B-it-GGUF"),
            "Automatic"
        );
        assert_eq!(
            display_model_name(false, "ggml-org/gemma-4-E4B-it-GGUF"),
            "ggml-org/gemma-4-E4B-it-GGUF"
        );
    }

    #[test]
    fn shows_automatic_when_connected_to_a_coordinator() {
        let header = render_header(
            "1.0.0",
            "ggml-org/gemma-4-E4B-it-GGUF",
            "http://localhost:9000/v1",
            std::path::Path::new("/tmp"),
            status(true),
            Banner::Left,
            120,
        );
        assert!(header.contains("Model: Automatic"));
        assert!(!header.contains("gemma"));
    }

    #[test]
    fn shows_the_real_model_id_otherwise() {
        let header = render_header(
            "1.0.0",
            "ggml-org/gemma-4-E4B-it-GGUF",
            "http://localhost:8100/v1",
            std::path::Path::new("/tmp"),
            status(false),
            Banner::Left,
            120,
        );
        assert!(header.contains("Model: ggml-org/gemma-4-E4B-it-GGUF"));
        assert!(!header.contains("Automatic"));
    }

    #[test]
    fn indicator_shows_a_distinct_color_for_each_conn_status() {
        // Regression test: Pending (white, still resolving — used while
        // startup connectivity checks are still running in the background)
        // must render differently from both Ok (green) and Failed (red).
        let pending = indicator(ConnStatus::Pending);
        let ok = indicator(ConnStatus::Ok);
        let failed = indicator(ConnStatus::Failed);
        assert!(pending.contains(STATUS_WHITE));
        assert!(ok.contains(STATUS_GREEN));
        assert!(failed.contains(STATUS_RED));
        assert_ne!(pending, ok);
        assert_ne!(pending, failed);
        assert_ne!(ok, failed);
    }

    #[test]
    fn conn_status_from_bool_maps_true_to_ok_and_false_to_failed() {
        assert_eq!(ConnStatus::from_bool(true), ConnStatus::Ok);
        assert_eq!(ConnStatus::from_bool(false), ConnStatus::Failed);
    }

    #[test]
    fn render_header_shows_pending_as_a_white_dot() {
        let header = render_header(
            "1.0.0",
            "ggml-org/gemma-4-E4B-it-GGUF",
            "http://localhost:8100/v1",
            std::path::Path::new("/tmp"),
            HeaderStatus {
                workspace_ok: true,
                server_ok: ConnStatus::Pending,
                model_ok: ConnStatus::Pending,
                is_coordinator: false,
            },
            Banner::Left,
            120,
        );
        assert!(header.contains(&format!("Server: http://localhost:8100/v1 {STATUS_WHITE}")));
        assert!(header.contains(&format!(
            "Model: ggml-org/gemma-4-E4B-it-GGUF {STATUS_WHITE}"
        )));
    }
}

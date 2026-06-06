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

use anyhow::{Result, anyhow};
use orangu::{
    config::LlmConfiguration, session::ChatSession, tools::ToolExecutor, tui::ReviewEntry,
};
use std::{borrow::Cow, collections::HashMap, path::Path};
use terminal_size::{Width, terminal_size};

#[derive(Clone, Copy, Default)]
pub struct ShowFileOptions {
    pub show_hash: bool,
    pub show_author: bool,
}

#[derive(Debug)]
pub enum LocalError {
    Usage(String),
}

impl std::fmt::Display for LocalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalError::Usage(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for LocalError {}

pub fn current_terminal_width() -> usize {
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

pub fn shell_words(input: &str) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut quote = None;

    while let Some(ch) = chars.next() {
        match quote {
            Some(active_quote) => {
                if ch == active_quote {
                    quote = None;
                } else if ch == '\\' && active_quote == '"' {
                    if let Some(escaped) = chars.next() {
                        current.push(escaped);
                    }
                } else {
                    current.push(ch);
                }
            }
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
            }
            None if ch == '\\' => {
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
            }
            None => current.push(ch),
        }
    }

    if quote.is_some() {
        return Err(anyhow!("EDITOR contains unterminated quotes"));
    }
    if !current.is_empty() {
        words.push(current);
    }
    if words.is_empty() {
        return Err(anyhow!("EDITOR is empty"));
    }

    Ok(words)
}

pub enum CommandOutcome {
    Unhandled,
    Quiet,
    /// Command ran and produced informational output (success).
    Output(String),
    WideOutput(String),
    /// Command failed — invalid usage, unknown command, or other error.
    OutputError(String),
    Cleared,
    Quit,
    Restart,
    /// Re-exec into a different existing session, resuming the given UUID.
    SwitchSession(String),
    Blocking(Box<dyn FnOnce() -> anyhow::Result<String> + Send + 'static>),
    /// A long-running command that streams its output line by line through the
    /// sink as it is produced, rather than returning it all at once.
    Streaming(
        Box<
            dyn FnOnce(tokio::sync::mpsc::UnboundedSender<String>) -> anyhow::Result<()>
                + Send
                + 'static,
        >,
    ),
    /// Enter the interactive `/review` mode with a collected branch diff.
    Review(ReviewLaunch),
}

/// Data handed to the interactive review mode: the changed files, each with its
/// own rendered diff lines.
pub struct ReviewLaunch {
    pub files: Vec<ReviewEntry>,
}

pub enum CommentBody<'a> {
    /// An inline comment body supplied directly in the command (`"..."` or bare text).
    Inline(Cow<'a, str>),
    /// A filename under `~/.orangu/comments/` whose content is the comment body.
    File(Cow<'a, str>),
}

pub enum CloseTarget {
    Issue(u64),
    PullRequest(u64),
}

pub enum StashSubcommand {
    Push,
    Pop,
    List,
    Drop,
}

pub enum BranchSubcommand<'a> {
    List,
    ListAll,
    Switch(Cow<'a, str>),
    Create(Cow<'a, str>),
    Rename(Cow<'a, str>),
    Delete(Cow<'a, str>),
}

pub enum LocalCommand<'a> {
    Help,
    ConnectDefault,
    ConnectTo(&'a str),
    Disconnect,
    Reload,
    Restart,
    ListFiles,
    ShowFile(Cow<'a, str>),
    Tools,
    ModelInfo,
    SetModelId(&'a str),
    ServerInfo,
    SetServer(&'a str),
    Diff(Option<Cow<'a, str>>),
    Grep(Option<Cow<'a, str>>),
    Review,
    Status,
    Log(Option<u64>),
    Pull(Option<u64>),
    Comment(Option<(u64, CommentBody<'a>)>),
    Close(Option<CloseTarget>),
    CreatePullRequest,
    Rebase,
    Merge(Option<Cow<'a, str>>),
    Branch(BranchSubcommand<'a>),
    Restore(Option<Cow<'a, str>>),
    AddFile(Option<Cow<'a, str>>),
    RemoveFile(Option<Cow<'a, str>>),
    MoveFile(Option<(Cow<'a, str>, Cow<'a, str>)>),
    CherryPick(Option<Cow<'a, str>>),
    Commit(Option<Cow<'a, str>>),
    Amend(Option<Cow<'a, str>>),
    Push(bool),
    InitRepo,
    Squash,
    Stash(StashSubcommand),
    OpenFile(&'a str),
    Session(Option<Cow<'a, str>>),
    Sessions(Option<Cow<'a, str>>),
    Usage,
    Build,
    Clear,
    Quit,
}

pub struct CommandContext<'a> {
    pub startup_model: &'a str,
    pub startup_endpoint: &'a str,
    pub llms: &'a HashMap<String, LlmConfiguration>,
    pub tools: &'a ToolExecutor,
    pub workspace: &'a Path,
    pub usage_stats: &'a crate::UsageStats,
    pub available_models: &'a [String],
    pub virtual_width: usize,
    pub auto_rebase: bool,
    pub auto_squash: bool,
    pub terminal: &'a str,
    pub forge: crate::git::Forge,
}

pub struct CommandState<'a> {
    pub active_model: &'a mut String,
    pub active_model_id: &'a mut String,
    pub current_endpoint: &'a mut Option<String>,
    pub session: &'a mut ChatSession,
    pub detect_model: &'a mut bool,
}

pub fn parse_local_command(input: &str) -> Option<LocalCommand<'_>> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    parse_slash_command(input).or_else(|| parse_natural_language_command(input))
}

pub fn parse_slash_command(input: &str) -> Option<LocalCommand<'_>> {
    match input {
        "/help" => Some(LocalCommand::Help),
        "/connect" => Some(LocalCommand::ConnectDefault),
        "/disconnect" => Some(LocalCommand::Disconnect),
        "/reload" => Some(LocalCommand::Reload),
        "/restart" => Some(LocalCommand::Restart),
        "/tools" => Some(LocalCommand::Tools),
        "/model" => Some(LocalCommand::ModelInfo),
        "/server" => Some(LocalCommand::ServerInfo),
        "/session" => Some(LocalCommand::Session(None)),
        "/sessions" => Some(LocalCommand::Sessions(None)),
        "/list_files" => Some(LocalCommand::ListFiles),
        "/open_file" => Some(LocalCommand::OpenFile("")),
        "/show_file" => Some(LocalCommand::ShowFile(Cow::Borrowed(""))),
        "/build" => Some(LocalCommand::Build),
        "/add_file" => Some(LocalCommand::AddFile(None)),
        "/amend" => Some(LocalCommand::Amend(None)),
        "/branch" => Some(LocalCommand::Branch(BranchSubcommand::List)),
        "/cherry_pick" => Some(LocalCommand::CherryPick(None)),
        "/commit" => Some(LocalCommand::Commit(None)),
        "/restore" => Some(LocalCommand::Restore(None)),
        "/diff" => Some(LocalCommand::Diff(None)),
        "/grep" => Some(LocalCommand::Grep(None)),
        "/init_repo" => Some(LocalCommand::InitRepo),
        "/log" => Some(LocalCommand::Log(None)),
        "/merge" => Some(LocalCommand::Merge(None)),
        "/move_file" => Some(LocalCommand::MoveFile(None)),
        "/pull" => Some(LocalCommand::Pull(None)),
        "/comment" => Some(LocalCommand::Comment(None)),
        "/close" => Some(LocalCommand::Close(None)),
        "/pull_request" => Some(LocalCommand::CreatePullRequest),
        "/review" => Some(LocalCommand::Review),
        "/push" => Some(LocalCommand::Push(false)),
        "/rebase" => Some(LocalCommand::Rebase),
        "/remove_file" => Some(LocalCommand::RemoveFile(None)),
        "/squash" => Some(LocalCommand::Squash),
        "/stash" => Some(LocalCommand::Stash(StashSubcommand::Push)),
        "/status" => Some(LocalCommand::Status),
        "/usage" => Some(LocalCommand::Usage),
        "/clear" => Some(LocalCommand::Clear),
        "/quit" => Some(LocalCommand::Quit),
        _ => {
            if let Some(args) = input.strip_prefix("/session ") {
                let uuid = args.trim();
                return Some(LocalCommand::Session(if uuid.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(uuid))
                }));
            }
            if let Some(args) = input.strip_prefix("/sessions ") {
                let workspace = args.trim();
                return Some(LocalCommand::Sessions(if workspace.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(workspace))
                }));
            }
            if let Some(endpoint) = input.strip_prefix("/connect ") {
                return Some(LocalCommand::ConnectTo(endpoint.trim()));
            }
            if let Some(args) = input.strip_prefix("/diff ") {
                let branch = args.trim();
                return Some(LocalCommand::Diff(if branch.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(branch))
                }));
            }
            if let Some(args) = input.strip_prefix("/grep ") {
                let pattern = args.trim();
                return Some(LocalCommand::Grep(if pattern.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(pattern))
                }));
            }
            if let Some(name) = input.strip_prefix("/model ") {
                return Some(LocalCommand::SetModelId(name.trim()));
            }
            if let Some(name) = input.strip_prefix("/server ") {
                return Some(LocalCommand::SetServer(name.trim()));
            }
            if let Some(args) = input.strip_prefix("/show_file ") {
                return Some(LocalCommand::ShowFile(Cow::Borrowed(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/log ") {
                return Some(LocalCommand::Log(args.trim().parse::<u64>().ok()));
            }
            if let Some(args) = input.strip_prefix("/pull ") {
                return Some(LocalCommand::Pull(args.trim().parse::<u64>().ok()));
            }
            if let Some(args) = input.strip_prefix("/comment ") {
                return Some(LocalCommand::Comment(parse_comment_args(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/close ") {
                return Some(LocalCommand::Close(parse_close_args(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/merge ") {
                let branch = args.trim();
                return Some(LocalCommand::Merge(if branch.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(branch))
                }));
            }
            if let Some(args) = input.strip_prefix("/checkout ") {
                let target = args.trim();
                if !target.is_empty() {
                    return Some(LocalCommand::Branch(BranchSubcommand::Switch(
                        Cow::Borrowed(target),
                    )));
                }
            }
            if let Some(args) = input.strip_prefix("/add_file ") {
                let path = args.trim();
                return Some(LocalCommand::AddFile(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
            }
            if let Some(args) = input.strip_prefix("/remove_file ") {
                let path = args.trim();
                return Some(LocalCommand::RemoveFile(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
            }
            if let Some(args) = input.strip_prefix("/move_file ") {
                let args = args.trim();
                return Some(match shell_words(args) {
                    Ok(words) if words.len() >= 2 => LocalCommand::MoveFile(Some((
                        Cow::Owned(words[0].clone()),
                        Cow::Owned(words[1].clone()),
                    ))),
                    _ => LocalCommand::MoveFile(None),
                });
            }
            if let Some(args) = input.strip_prefix("/cherry_pick ") {
                let commit = args.trim();
                return Some(LocalCommand::CherryPick(if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                }));
            }
            if let Some(args) = input.strip_prefix("/commit ") {
                let message = strip_matching_quotes(args.trim());
                return Some(LocalCommand::Commit(if message.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(message))
                }));
            }
            if let Some(args) = input.strip_prefix("/amend ") {
                let message = strip_matching_quotes(args.trim());
                return Some(LocalCommand::Amend(if message.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(message))
                }));
            }
            if let Some(flag) = input.strip_prefix("/push ") {
                let flag = flag.trim();
                if flag == "--force" || flag == "-f" || flag.eq_ignore_ascii_case("force") {
                    return Some(LocalCommand::Push(true));
                }
            }
            if let Some(sub) = input.strip_prefix("/branch ") {
                let sub = sub.trim();
                return Some(LocalCommand::Branch(match sub {
                    "-a" | "--all" => BranchSubcommand::ListAll,
                    _ if sub.starts_with("-b ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Create(Cow::Borrowed(name))
                        }
                    }
                    _ if sub.starts_with("-m ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Rename(Cow::Borrowed(name))
                        }
                    }
                    _ if sub.starts_with("-d ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Delete(Cow::Borrowed(name))
                        }
                    }
                    _ if !sub.is_empty() => BranchSubcommand::Switch(Cow::Borrowed(sub)),
                    _ => BranchSubcommand::List,
                }));
            }
            if let Some(path) = input.strip_prefix("/restore ") {
                let path = path.trim();
                let staged = path.starts_with("--staged ") || path.starts_with("-S ");
                let file = if staged {
                    path.split_once(' ').map(|x| x.1).unwrap_or("").trim()
                } else {
                    path
                };
                return Some(LocalCommand::Restore(if file.is_empty() {
                    None
                } else {
                    Some(Cow::Owned(format!(
                        "{}{}",
                        if staged { "--staged " } else { "" },
                        file
                    )))
                }));
            }
            if let Some(sub) = input.strip_prefix("/stash ") {
                return Some(LocalCommand::Stash(match sub.trim() {
                    "pop" => StashSubcommand::Pop,
                    "list" => StashSubcommand::List,
                    "drop" => StashSubcommand::Drop,
                    _ => StashSubcommand::Push,
                }));
            }
            if let Some(args) = input.strip_prefix("/delete ") {
                let branch = args.trim();
                if !branch.is_empty() {
                    return Some(LocalCommand::Branch(BranchSubcommand::Delete(
                        Cow::Borrowed(branch),
                    )));
                }
            }
            if let Some(args) = input.strip_prefix("/open_file ")
                && args.trim().is_empty()
            {
                return Some(LocalCommand::OpenFile(""));
            }
            parse_open_file_target(input, "/open_file ").map(LocalCommand::OpenFile)
        }
    }
}

pub fn parse_natural_language_command(input: &str) -> Option<LocalCommand<'_>> {
    if matches_ci(
        input,
        &[
            "help",
            "show help",
            "show commands",
            "show available commands",
        ],
    ) {
        return Some(LocalCommand::Help);
    }
    if matches_ci(input, &["connect", "reconnect"]) {
        return Some(LocalCommand::ConnectDefault);
    }
    if let Some(endpoint) = strip_ascii_prefix(input, "connect to ") {
        return Some(LocalCommand::ConnectTo(endpoint.trim()));
    }
    if matches_ci(input, &["disconnect"]) {
        return Some(LocalCommand::Disconnect);
    }
    if matches_ci(input, &["reload", "reload configuration", "reset session"]) {
        return Some(LocalCommand::Reload);
    }
    if matches_ci(input, &["restart", "restart orangu"]) {
        return Some(LocalCommand::Restart);
    }
    if matches_ci(
        input,
        &[
            "list files",
            "show files",
            "show workspace files",
            "list workspace files",
        ],
    ) {
        return Some(LocalCommand::ListFiles);
    }
    if matches_ci(
        input,
        &["show tools", "list tools", "show local tools", "tools"],
    ) {
        return Some(LocalCommand::Tools);
    }
    if matches_ci(
        input,
        &[
            "show model",
            "current model",
            "what model am i using",
            "model",
            "list models",
            "show models",
            "show available models",
            "models",
        ],
    ) {
        return Some(LocalCommand::ModelInfo);
    }
    if matches_ci(input, &["build", "build project", "run build"]) {
        return Some(LocalCommand::Build);
    }
    if matches_ci(input, &["diff", "show diff", "git diff"]) {
        return Some(LocalCommand::Diff(None));
    }
    for prefix in ["diff against ", "show diff against ", "git diff "] {
        if let Some(branch) = strip_ascii_prefix(input, prefix) {
            let branch = branch.trim();
            return Some(LocalCommand::Diff(if branch.is_empty() {
                None
            } else {
                Some(Cow::Borrowed(branch))
            }));
        }
    }
    if matches_ci(
        input,
        &["review", "review changes", "code review", "review branch"],
    ) {
        return Some(LocalCommand::Review);
    }
    if matches_ci(input, &["status", "show status", "git status"]) {
        return Some(LocalCommand::Status);
    }
    for prefix in ["grep ", "find ", "git grep "] {
        if let Some(pattern) = strip_ascii_prefix(input, prefix) {
            let pattern = pattern.trim();
            if !pattern.is_empty() {
                return Some(LocalCommand::Grep(Some(Cow::Borrowed(pattern))));
            }
        }
    }
    for prefix in ["log ", "show log ", "git log ", "git lg "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix)
            && let Ok(count) = rest.trim().parse::<u64>()
        {
            return Some(LocalCommand::Log(Some(count)));
        }
    }
    if matches_ci(input, &["log", "show log", "git log", "git lg"]) {
        return Some(LocalCommand::Log(None));
    }
    for prefix in [
        "use server ",
        "switch server to ",
        "set server to ",
        "select server ",
    ] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::SetServer(name.trim()));
        }
    }
    for prefix in [
        "use model ",
        "switch model to ",
        "set model to ",
        "select model ",
    ] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::SetModelId(name.trim()));
        }
    }
    if let Some(path) = parse_open_file_target(input, "/open_file ") {
        return Some(LocalCommand::OpenFile(path));
    }
    for prefix in ["open file ", "open ", "edit file ", "edit "] {
        if let Some(path) = parse_open_file_target(input, prefix) {
            return Some(LocalCommand::OpenFile(path));
        }
    }
    if let Some(args) = parse_show_file_natural_language_args(input) {
        return Some(LocalCommand::ShowFile(args));
    }
    if let Some(pr_number) = parse_pull_pr_number(input) {
        return Some(LocalCommand::Pull(Some(pr_number)));
    }
    for prefix in ["add comment on ", "add comment to ", "comment on "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::Comment(parse_comment_args(rest.trim())));
        }
    }
    if matches_ci(
        input,
        &[
            "pull request",
            "create pull request",
            "open pull request",
            "new pull request",
            "create pr",
            "open pr",
            "new pr",
        ],
    ) {
        return Some(LocalCommand::CreatePullRequest);
    }
    for prefix in ["close issue ", "close -i "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            let n = rest.trim().trim_start_matches('#').parse::<u64>().ok();
            return Some(LocalCommand::Close(n.map(CloseTarget::Issue)));
        }
    }
    for prefix in ["close pr ", "close pull request ", "close -p "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            let n = rest.trim().trim_start_matches('#').parse::<u64>().ok();
            return Some(LocalCommand::Close(n.map(CloseTarget::PullRequest)));
        }
    }
    if matches_ci(input, &["stash", "git stash", "git stash push"]) {
        return Some(LocalCommand::Stash(StashSubcommand::Push));
    }
    if matches_ci(input, &["stash pop", "pop stash", "git stash pop"]) {
        return Some(LocalCommand::Stash(StashSubcommand::Pop));
    }
    if matches_ci(input, &["stash list", "list stashes", "git stash list"]) {
        return Some(LocalCommand::Stash(StashSubcommand::List));
    }
    if matches_ci(input, &["stash drop", "drop stash", "git stash drop"]) {
        return Some(LocalCommand::Stash(StashSubcommand::Drop));
    }
    if matches_ci(input, &["rebase", "git rebase"]) {
        return Some(LocalCommand::Rebase);
    }
    for prefix in ["git merge ", "merge "] {
        if let Some(branch) = strip_ascii_prefix(input, prefix) {
            let branch = branch.trim();
            if !branch.is_empty() {
                return Some(LocalCommand::Merge(Some(Cow::Borrowed(branch))));
            }
        }
    }
    if matches_ci(input, &["merge"]) {
        return Some(LocalCommand::Merge(None));
    }
    for prefix in [
        "git checkout ",
        "checkout ",
        "switch to branch ",
        "switch to ",
    ] {
        if let Some(target) = strip_ascii_prefix(input, prefix) {
            let target = strip_ascii_suffix(target.trim(), " branch")
                .map(str::trim)
                .unwrap_or(target.trim());
            if !target.is_empty() {
                return Some(LocalCommand::Branch(BranchSubcommand::Switch(
                    Cow::Borrowed(target),
                )));
            }
        }
    }
    for prefix in ["create branch ", "new branch ", "branch -b "] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            let name = name.trim();
            if !name.is_empty() {
                return Some(LocalCommand::Branch(BranchSubcommand::Create(
                    Cow::Borrowed(name),
                )));
            }
        }
    }
    for prefix in ["rename branch to ", "rename to ", "branch -m "] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            let name = name.trim();
            if !name.is_empty() {
                return Some(LocalCommand::Branch(BranchSubcommand::Rename(
                    Cow::Borrowed(name),
                )));
            }
        }
    }
    if matches_ci(
        input,
        &["branch", "list branches", "git branch", "checkout"],
    ) {
        return Some(LocalCommand::Branch(BranchSubcommand::List));
    }
    if matches_ci(input, &["list all branches", "branch -a", "branch --all"]) {
        return Some(LocalCommand::Branch(BranchSubcommand::ListAll));
    }
    for prefix in ["restore ", "git restore "] {
        if let Some(path) = strip_ascii_prefix(input, prefix) {
            let path = path.trim();
            if !path.is_empty() {
                return Some(LocalCommand::Restore(Some(Cow::Borrowed(path))));
            }
        }
    }
    for prefix in ["git add ", "add file ", "add "] {
        if let Some(path) = strip_ascii_prefix(input, prefix) {
            let path = path.trim();
            if !path.is_empty() {
                return Some(LocalCommand::AddFile(Some(Cow::Borrowed(path))));
            }
        }
    }
    if matches_ci(input, &["add"]) {
        return Some(LocalCommand::AddFile(None));
    }
    for prefix in ["git rm ", "remove file ", "remove "] {
        if let Some(path) = strip_ascii_prefix(input, prefix) {
            let path = path.trim();
            if !path.is_empty() {
                return Some(LocalCommand::RemoveFile(Some(Cow::Borrowed(path))));
            }
        }
    }
    if matches_ci(input, &["remove"]) {
        return Some(LocalCommand::RemoveFile(None));
    }
    for prefix in ["git mv ", "move file ", "move "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            let rest = rest.trim();
            if let Ok(words) = shell_words(rest)
                && words.len() >= 2
            {
                return Some(LocalCommand::MoveFile(Some((
                    Cow::Owned(words[0].clone()),
                    Cow::Owned(words[1].clone()),
                ))));
            }
        }
    }
    if matches_ci(input, &["move"]) {
        return Some(LocalCommand::MoveFile(None));
    }
    for prefix in ["git cherry-pick ", "cherry-pick ", "cherry pick "] {
        if let Some(commit) = strip_ascii_prefix(input, prefix) {
            let commit = commit.trim();
            if !commit.is_empty() {
                return Some(LocalCommand::CherryPick(Some(Cow::Borrowed(commit))));
            }
        }
    }
    if matches_ci(input, &["cherry pick", "cherry-pick"]) {
        return Some(LocalCommand::CherryPick(None));
    }
    for prefix in ["git commit -a -m ", "git commit -m ", "commit "] {
        if let Some(msg) = strip_ascii_prefix(input, prefix) {
            let msg = strip_matching_quotes(msg.trim());
            if !msg.is_empty() {
                return Some(LocalCommand::Commit(Some(Cow::Borrowed(msg))));
            }
        }
    }
    if matches_ci(input, &["commit"]) {
        return Some(LocalCommand::Commit(None));
    }
    for prefix in [
        "git commit --amend -m ",
        "git amend -m ",
        "git amend ",
        "amend message ",
        "amend ",
    ] {
        if let Some(msg) = strip_ascii_prefix(input, prefix) {
            let msg = strip_matching_quotes(msg.trim());
            if !msg.is_empty() {
                return Some(LocalCommand::Amend(Some(Cow::Borrowed(msg))));
            }
        }
    }
    if matches_ci(input, &["amend", "git amend", "git commit --amend"]) {
        return Some(LocalCommand::Amend(None));
    }
    if matches_ci(
        input,
        &[
            "force push",
            "push force",
            "push --force",
            "push -f",
            "git push --force",
            "git push -f",
            "git push origin --force",
            "git push origin -f",
        ],
    ) {
        return Some(LocalCommand::Push(true));
    }
    if matches_ci(input, &["push", "git push", "git push origin"]) {
        return Some(LocalCommand::Push(false));
    }
    if matches_ci(input, &["init", "init repo", "git init"]) {
        return Some(LocalCommand::InitRepo);
    }
    if matches_ci(
        input,
        &["squash", "squash branch", "squash commits", "git squash"],
    ) {
        return Some(LocalCommand::Squash);
    }
    if matches_ci(input, &["delete", "delete branch"]) {
        return Some(LocalCommand::Branch(BranchSubcommand::List));
    }
    for prefix in ["git branch -D ", "delete branch ", "delete "] {
        if let Some(branch) = strip_ascii_prefix(input, prefix) {
            let branch = branch.trim();
            if !branch.is_empty() {
                return Some(LocalCommand::Branch(BranchSubcommand::Delete(
                    Cow::Borrowed(branch),
                )));
            }
        }
    }
    if matches_ci(input, &["session", "switch session"]) {
        return Some(LocalCommand::Session(None));
    }
    if matches_ci(input, &["sessions", "list sessions", "show sessions"]) {
        return Some(LocalCommand::Sessions(None));
    }
    if matches_ci(input, &["usage", "show usage"]) {
        return Some(LocalCommand::Usage);
    }
    if matches_ci(
        input,
        &["clear", "clear conversation", "reset conversation"],
    ) {
        return Some(LocalCommand::Clear);
    }
    if matches_ci(input, &["quit", "exit"]) {
        return Some(LocalCommand::Quit);
    }

    None
}

pub fn parse_open_file_target<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let path = strip_ascii_prefix(input, prefix)?.trim();
    if path.is_empty() {
        return None;
    }
    Some(strip_matching_quotes(path))
}

pub fn parse_show_file_natural_language_args(input: &str) -> Option<Cow<'_, str>> {
    parse_show_file_natural_language_args_with_prefix(input, "show file ", false)
        .or_else(|| parse_show_file_natural_language_args_with_prefix(input, "show ", true))
}

pub fn parse_show_file_natural_language_args_with_prefix<'a>(
    input: &'a str,
    prefix: &str,
    single_token_only: bool,
) -> Option<Cow<'a, str>> {
    let raw = strip_ascii_prefix(input, prefix)?.trim();
    let (path, options) = parse_show_file_natural_language_target(raw, single_token_only)?;
    if !options.show_hash && !options.show_author {
        return Some(Cow::Borrowed(path));
    }

    let mut args = String::new();
    if options.show_hash {
        args.push_str("--hash ");
    }
    if options.show_author {
        args.push_str("--author ");
    }
    args.push_str(&quote_shell_argument(path));
    Some(Cow::Owned(args))
}

pub fn parse_show_file_natural_language_target(
    raw: &str,
    single_token_only: bool,
) -> Option<(&str, ShowFileOptions)> {
    for (suffix, options) in [
        (
            " with hash and author",
            ShowFileOptions {
                show_hash: true,
                show_author: true,
            },
        ),
        (
            " with author and hash",
            ShowFileOptions {
                show_hash: true,
                show_author: true,
            },
        ),
        (
            " with hash",
            ShowFileOptions {
                show_hash: true,
                show_author: false,
            },
        ),
        (
            " with author",
            ShowFileOptions {
                show_hash: false,
                show_author: true,
            },
        ),
    ] {
        if let Some(path) = strip_ascii_suffix(raw, suffix) {
            let path = parse_show_file_target(path.trim(), single_token_only)?;
            return Some((path, options));
        }
    }

    parse_show_file_target(raw, single_token_only).map(|path| (path, ShowFileOptions::default()))
}

pub fn parse_show_file_target(path: &str, single_token_only: bool) -> Option<&str> {
    if path.is_empty() {
        return None;
    }
    let quoted = matches!(path.chars().next(), Some('"') | Some('\''));
    if single_token_only && !quoted && path.chars().any(char::is_whitespace) {
        return None;
    }
    Some(strip_matching_quotes(path))
}

pub fn parse_pull_pr_number(input: &str) -> Option<u64> {
    for prefix in [
        "pull pull request ",
        "pull request ",
        "pull pr ",
        "pull #",
        "pull ",
    ] {
        if let Some(rest) = strip_ascii_prefix(input, prefix)
            && let Ok(num) = rest.trim().parse::<u64>()
        {
            return Some(num);
        }
    }
    None
}

pub fn parse_comment_args(input: &str) -> Option<(u64, CommentBody<'_>)> {
    let input = input.trim();
    let (number, rest) = input.split_once(char::is_whitespace)?;
    let number = number.trim_start_matches('#').parse::<u64>().ok()?;
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    if rest.starts_with('"') || rest.starts_with('\'') {
        let body = strip_matching_quotes(rest);
        if body.is_empty() {
            return None;
        }
        Some((number, CommentBody::Inline(Cow::Borrowed(body))))
    } else {
        Some((number, CommentBody::File(Cow::Borrowed(rest))))
    }
}

pub fn strip_ascii_suffix<'a>(input: &'a str, suffix: &str) -> Option<&'a str> {
    if input.len() >= suffix.len()
        && input[input.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
    {
        Some(&input[..input.len() - suffix.len()])
    } else {
        None
    }
}

pub fn strip_ascii_prefix<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    if input.len() >= prefix.len() && input[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&input[prefix.len()..])
    } else {
        None
    }
}

pub fn quote_shell_argument(argument: &str) -> String {
    if !argument.is_empty()
        && !argument
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '\'' | '\\' | '$' | '`'))
    {
        return argument.to_string();
    }

    let mut quoted = String::from("\"");
    for ch in argument.chars() {
        match ch {
            '"' | '\\' | '$' | '`' => {
                quoted.push('\\');
                quoted.push(ch);
            }
            _ => quoted.push(ch),
        }
    }
    quoted.push('"');
    quoted
}

pub fn strip_matching_quotes(input: &str) -> &str {
    if input.len() >= 2 {
        let bytes = input.as_bytes();
        let first = bytes[0];
        let last = bytes[input.len() - 1];
        if (first == b'"' || first == b'\'') && first == last {
            return &input[1..input.len() - 1];
        }
    }
    input
}

pub fn matches_ci(input: &str, options: &[&str]) -> bool {
    options
        .iter()
        .any(|option| input.eq_ignore_ascii_case(option))
}

pub fn system_prompt(profile: &LlmConfiguration) -> &str {
    if profile.system_prompt.is_empty() {
        "You are Orangu, a coding environment assistant connected to a local workspace. Use the available local tools to inspect files, edit files on disk, fetch external URLs for knowledge, and run shell commands when needed. Be precise, explain what you changed, and surface tool failures explicitly."
    } else {
        &profile.system_prompt
    }
}

pub fn sorted_model_names(llms: &HashMap<String, LlmConfiguration>) -> Vec<String> {
    let mut names = llms.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

pub fn open_file_usage_message() -> &'static str {
    "Usage: /open_file <path>. Use /help to see available commands."
}

pub fn model_usage_message() -> &'static str {
    "Usage: /model <name>. Use /help to see available commands."
}

pub fn server_usage_message() -> &'static str {
    "Usage: /server <name>. Use /help to see available commands."
}

pub fn connect_usage_message() -> &'static str {
    "Usage: /connect <endpoint>. Use /help to see available commands."
}

pub fn pull_usage_message() -> &'static str {
    "Usage: /pull <number>. Use /help to see available commands."
}

pub fn parse_close_args(input: &str) -> Option<CloseTarget> {
    let input = input.trim();
    if let Some(rest) = input.strip_prefix("-i ") {
        return rest
            .trim()
            .trim_start_matches('#')
            .parse::<u64>()
            .ok()
            .map(CloseTarget::Issue);
    }
    if let Some(rest) = input.strip_prefix("-p ") {
        return rest
            .trim()
            .trim_start_matches('#')
            .parse::<u64>()
            .ok()
            .map(CloseTarget::PullRequest);
    }
    None
}

pub fn close_usage_message() -> &'static str {
    "Usage: /close -i <number> or /close -p <number>. Use /help to see available commands."
}

pub fn comment_usage_message() -> &'static str {
    "Usage: /comment <number> \"<comment>\" or /comment <number> <file>. Use /help to see available commands."
}

pub fn merge_usage_message() -> &'static str {
    "Usage: /merge <branch>. Use /help to see available commands."
}

pub fn restore_usage_message() -> &'static str {
    "Usage: /restore [--staged] <file>. Use /help to see available commands."
}

pub fn grep_usage_message() -> &'static str {
    "Usage: /grep <pattern>. Use /help to see available commands."
}

pub fn add_file_usage_message() -> &'static str {
    "Usage: /add_file <path>. Use /help to see available commands."
}

pub fn remove_file_usage_message() -> &'static str {
    "Usage: /remove_file <path>. Use /help to see available commands."
}

pub fn move_file_usage_message() -> &'static str {
    "Usage: /move_file <source> <destination>. Use /help to see available commands."
}

pub fn cherry_pick_usage_message() -> &'static str {
    "Usage: /cherry_pick <commit>. Use /help to see available commands."
}

pub fn commit_usage_message() -> &'static str {
    "Usage: /commit <message>. Use /help to see available commands."
}

pub fn amend_usage_message() -> &'static str {
    "Usage: /amend <message>. Use /help to see available commands."
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leaves_regular_prompts_unhandled() {
        assert!(parse_local_command("help me understand this code").is_none());
        assert!(parse_local_command("show me the files in the workspace").is_none());
    }

    #[test]
    fn parses_open_file_commands() {
        match parse_local_command("/open_file README.md") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
            _ => panic!("expected open file slash command"),
        }
        match parse_local_command("Open README.md") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
            _ => panic!("expected open file natural language command"),
        }
        match parse_local_command("open \"docs/user guide.md\"") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "docs/user guide.md"),
            _ => panic!("expected quoted natural language open file command"),
        }
    }

    #[test]
    fn parses_show_file_natural_language_commands() {
        match parse_local_command("show README.md") {
            Some(LocalCommand::ShowFile(path)) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected natural language show file command"),
        }
        match parse_local_command("show file \"docs/user guide.md\"") {
            Some(LocalCommand::ShowFile(path)) => assert_eq!(path.as_ref(), "docs/user guide.md"),
            _ => panic!("expected quoted natural language show file command"),
        }
        match parse_local_command("show src/tui.rs with hash") {
            Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "--hash src/tui.rs"),
            _ => panic!("expected natural language show file hash command"),
        }
        match parse_local_command("show src/tui.rs with author") {
            Some(LocalCommand::ShowFile(args)) => {
                assert_eq!(args.as_ref(), "--author src/tui.rs")
            }
            _ => panic!("expected natural language show file author command"),
        }
        match parse_local_command("show file \"docs/user guide.md\" with hash and author") {
            Some(LocalCommand::ShowFile(args)) => {
                assert_eq!(args.as_ref(), "--hash --author \"docs/user guide.md\"")
            }
            _ => panic!("expected natural language show file metadata command"),
        }
    }

    #[test]
    fn parses_show_file_commands() {
        match parse_local_command("/show_file README.md") {
            Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "README.md"),
            _ => panic!("expected show file slash command"),
        }

        let (path, options, rev) = super::super::render::parse_show_file_arguments(
            "--hash --author \"docs/user guide.md\"",
        )
        .expect("show file args");
        assert_eq!(path, "docs/user guide.md");
        assert!(options.show_hash);
        assert!(options.show_author);
        assert!(rev.is_none());
    }

    #[test]
    fn parses_list_files_commands() {
        assert!(matches!(
            parse_local_command("/list_files"),
            Some(LocalCommand::ListFiles)
        ));
        assert!(matches!(
            parse_local_command("list files"),
            Some(LocalCommand::ListFiles)
        ));
        assert!(matches!(
            parse_local_command("show workspace files"),
            Some(LocalCommand::ListFiles)
        ));
    }

    #[test]
    fn parses_natural_language_command_aliases() {
        assert!(matches!(
            parse_local_command("show commands"),
            Some(LocalCommand::Help)
        ));
        assert!(matches!(
            parse_local_command("diff"),
            Some(LocalCommand::Diff(None))
        ));
        assert!(matches!(
            parse_local_command("list models"),
            Some(LocalCommand::ModelInfo)
        ));
        assert!(matches!(
            parse_local_command("show tools"),
            Some(LocalCommand::Tools)
        ));
        assert!(matches!(
            parse_local_command("disconnect"),
            Some(LocalCommand::Disconnect)
        ));
        assert!(matches!(
            parse_local_command("reset conversation"),
            Some(LocalCommand::Clear)
        ));
        assert!(matches!(
            parse_local_command("exit"),
            Some(LocalCommand::Quit)
        ));
    }

    #[test]
    fn parses_natural_language_commands_with_arguments() {
        match parse_local_command("connect to http://localhost:8080/v1") {
            Some(LocalCommand::ConnectTo(endpoint)) => {
                assert_eq!(endpoint, "http://localhost:8080/v1")
            }
            _ => panic!("expected connect command"),
        }
        match parse_local_command("switch model to local") {
            Some(LocalCommand::SetModelId(name)) => assert_eq!(name, "local"),
            _ => panic!("expected set model command"),
        }
        match parse_local_command("switch server to main") {
            Some(LocalCommand::SetServer(name)) => assert_eq!(name, "main"),
            _ => panic!("expected set server command"),
        }
        match parse_local_command("/server main") {
            Some(LocalCommand::SetServer(name)) => assert_eq!(name, "main"),
            _ => panic!("expected set server command"),
        }
    }

    #[test]
    fn parses_pull_request_commands() {
        assert!(matches!(
            parse_local_command("/pull 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("/pull"),
            Some(LocalCommand::Pull(None))
        ));
        assert!(matches!(
            parse_local_command("/pull notanumber"),
            Some(LocalCommand::Pull(None))
        ));
        assert!(matches!(
            parse_local_command("pull 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("Pull 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("pull pr 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("pull request 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("pull pull request 58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
        assert!(matches!(
            parse_local_command("pull #58"),
            Some(LocalCommand::Pull(Some(58)))
        ));
    }

    #[test]
    fn parses_comment_commands() {
        assert!(matches!(
            parse_local_command("/comment 51 \"My comment\""),
            Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
        ));
        assert!(matches!(
            parse_local_command("/comment 51 My comment"),
            Some(LocalCommand::Comment(Some((51, CommentBody::File(ref name))))) if name == "My comment"
        ));
        assert!(matches!(
            parse_local_command("/comment #51 \"My comment\""),
            Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
        ));
        assert!(matches!(
            parse_local_command("Add comment on 51 \"My comment\""),
            Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
        ));
        assert!(matches!(
            parse_local_command("comment on 51 \"My comment\""),
            Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
        ));
        assert!(matches!(
            parse_local_command("/comment 51 merged.md"),
            Some(LocalCommand::Comment(Some((51, CommentBody::File(ref name))))) if name == "merged.md"
        ));
        assert!(matches!(
            parse_local_command("/comment"),
            Some(LocalCommand::Comment(None))
        ));
        assert!(matches!(
            parse_local_command("/comment 51"),
            Some(LocalCommand::Comment(None))
        ));
        assert!(matches!(
            parse_local_command("/comment 51 \"\""),
            Some(LocalCommand::Comment(None))
        ));
        assert!(matches!(
            parse_local_command("/comment notanumber \"My comment\""),
            Some(LocalCommand::Comment(None))
        ));
    }

    #[test]
    fn parses_close_commands() {
        assert!(matches!(
            parse_local_command("/close -i 69"),
            Some(LocalCommand::Close(Some(CloseTarget::Issue(69))))
        ));
        assert!(matches!(
            parse_local_command("/close -p 42"),
            Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
        ));
        assert!(matches!(
            parse_local_command("close issue 69"),
            Some(LocalCommand::Close(Some(CloseTarget::Issue(69))))
        ));
        assert!(matches!(
            parse_local_command("close pr 42"),
            Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
        ));
        assert!(matches!(
            parse_local_command("close pull request 42"),
            Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
        ));
        assert!(matches!(
            parse_local_command("/close"),
            Some(LocalCommand::Close(None))
        ));
        assert!(matches!(
            parse_local_command("/close -i"),
            Some(LocalCommand::Close(None))
        ));
        assert!(matches!(
            parse_local_command("/close -p notanumber"),
            Some(LocalCommand::Close(None))
        ));
    }

    #[test]
    fn parses_review_commands() {
        for input in [
            "/review",
            "review",
            "Review",
            "review changes",
            "code review",
        ] {
            assert!(
                matches!(parse_local_command(input), Some(LocalCommand::Review)),
                "expected {input:?} to parse as Review"
            );
        }
    }

    #[test]
    fn parses_status_commands() {
        assert!(matches!(
            parse_local_command("/status"),
            Some(LocalCommand::Status)
        ));
        assert!(matches!(
            parse_local_command("status"),
            Some(LocalCommand::Status)
        ));
        assert!(matches!(
            parse_local_command("Status"),
            Some(LocalCommand::Status)
        ));
        assert!(matches!(
            parse_local_command("show status"),
            Some(LocalCommand::Status)
        ));
        assert!(matches!(
            parse_local_command("git status"),
            Some(LocalCommand::Status)
        ));
    }

    #[test]
    fn parses_log_commands() {
        assert!(matches!(
            parse_local_command("/log"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("log"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("Log"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("show log"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("git log"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("git lg"),
            Some(LocalCommand::Log(None))
        ));
        assert!(matches!(
            parse_local_command("/log 5"),
            Some(LocalCommand::Log(Some(5)))
        ));
        assert!(matches!(
            parse_local_command("log 10"),
            Some(LocalCommand::Log(Some(10)))
        ));
        assert!(matches!(
            parse_local_command("show log 3"),
            Some(LocalCommand::Log(Some(3)))
        ));
        assert!(matches!(
            parse_local_command("git lg 7"),
            Some(LocalCommand::Log(Some(7)))
        ));
    }

    #[test]
    fn parses_rebase_commands() {
        assert!(matches!(
            parse_local_command("/rebase"),
            Some(LocalCommand::Rebase)
        ));
        assert!(matches!(
            parse_local_command("rebase"),
            Some(LocalCommand::Rebase)
        ));
        assert!(matches!(
            parse_local_command("Rebase"),
            Some(LocalCommand::Rebase)
        ));
        assert!(matches!(
            parse_local_command("git rebase"),
            Some(LocalCommand::Rebase)
        ));
    }

    #[test]
    fn parses_merge_commands() {
        assert!(matches!(
            parse_local_command("/merge"),
            Some(LocalCommand::Merge(None))
        ));
        assert!(matches!(
            parse_local_command("/merge "),
            Some(LocalCommand::Merge(None))
        ));
        assert!(matches!(
            parse_local_command("merge"),
            Some(LocalCommand::Merge(None))
        ));
        assert!(matches!(
            parse_local_command("Merge"),
            Some(LocalCommand::Merge(None))
        ));
        match parse_local_command("/merge feature/foo") {
            Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
            _ => panic!("expected merge with branch"),
        }
        match parse_local_command("merge feature/foo") {
            Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
            _ => panic!("expected natural merge with branch"),
        }
        match parse_local_command("Merge feature/foo") {
            Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
            _ => panic!("expected case-insensitive merge with branch"),
        }
        match parse_local_command("git merge feature/foo") {
            Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
            _ => panic!("expected git merge natural language with branch"),
        }
    }

    #[test]
    fn parses_branch_commands() {
        assert!(matches!(
            parse_local_command("/branch"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
        assert!(matches!(
            parse_local_command("branch"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
        assert!(matches!(
            parse_local_command("list branches"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
        assert!(matches!(
            parse_local_command("checkout"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
        assert!(matches!(
            parse_local_command("/branch -a"),
            Some(LocalCommand::Branch(BranchSubcommand::ListAll))
        ));
        assert!(matches!(
            parse_local_command("list all branches"),
            Some(LocalCommand::Branch(BranchSubcommand::ListAll))
        ));
        match parse_local_command("/branch feature/foo") {
            Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected branch switch"),
        }
        match parse_local_command("/checkout feature/foo") {
            Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected checkout alias switch"),
        }
        match parse_local_command("checkout feature/foo") {
            Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected natural checkout switch"),
        }
        match parse_local_command("switch to main") {
            Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
                assert_eq!(target.as_ref(), "main")
            }
            _ => panic!("expected switch to main"),
        }
        match parse_local_command("switch to main branch") {
            Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
                assert_eq!(target.as_ref(), "main")
            }
            _ => panic!("expected switch to main branch -> main"),
        }
        match parse_local_command("/branch -b feature/new") {
            Some(LocalCommand::Branch(BranchSubcommand::Create(name))) => {
                assert_eq!(name.as_ref(), "feature/new")
            }
            _ => panic!("expected branch create"),
        }
        match parse_local_command("create branch feature/new") {
            Some(LocalCommand::Branch(BranchSubcommand::Create(name))) => {
                assert_eq!(name.as_ref(), "feature/new")
            }
            _ => panic!("expected NL branch create"),
        }
        match parse_local_command("/branch -m new-name") {
            Some(LocalCommand::Branch(BranchSubcommand::Rename(name))) => {
                assert_eq!(name.as_ref(), "new-name")
            }
            _ => panic!("expected branch rename"),
        }
        match parse_local_command("/branch -d feature/old") {
            Some(LocalCommand::Branch(BranchSubcommand::Delete(name))) => {
                assert_eq!(name.as_ref(), "feature/old")
            }
            _ => panic!("expected branch delete"),
        }
    }

    #[test]
    fn parses_add_file_commands() {
        assert!(matches!(
            parse_local_command("/add_file"),
            Some(LocalCommand::AddFile(None))
        ));
        assert!(matches!(
            parse_local_command("/add_file "),
            Some(LocalCommand::AddFile(None))
        ));
        assert!(matches!(
            parse_local_command("add"),
            Some(LocalCommand::AddFile(None))
        ));
        assert!(matches!(
            parse_local_command("Add"),
            Some(LocalCommand::AddFile(None))
        ));
        match parse_local_command("/add_file README.md") {
            Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected add_file with path"),
        }
        match parse_local_command("add README.md") {
            Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected natural add with path"),
        }
        match parse_local_command("Add src/") {
            Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "src/"),
            _ => panic!("expected case-insensitive add with directory"),
        }
        match parse_local_command("add file README.md") {
            Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected add file prefix"),
        }
        match parse_local_command("git add README.md") {
            Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected git add natural language"),
        }
    }

    #[test]
    fn parses_remove_file_commands() {
        assert!(matches!(
            parse_local_command("/remove_file"),
            Some(LocalCommand::RemoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("/remove_file "),
            Some(LocalCommand::RemoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("remove"),
            Some(LocalCommand::RemoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("Remove"),
            Some(LocalCommand::RemoveFile(None))
        ));
        match parse_local_command("/remove_file README.md") {
            Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected remove_file with path"),
        }
        match parse_local_command("remove README.md") {
            Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected natural remove with path"),
        }
        match parse_local_command("Remove src/") {
            Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "src/"),
            _ => panic!("expected case-insensitive remove with directory"),
        }
        match parse_local_command("remove file README.md") {
            Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected remove file prefix"),
        }
        match parse_local_command("git rm README.md") {
            Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
            _ => panic!("expected git rm natural language"),
        }
    }

    #[test]
    fn parses_move_file_commands() {
        assert!(matches!(
            parse_local_command("/move_file"),
            Some(LocalCommand::MoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("/move_file "),
            Some(LocalCommand::MoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("/move_file onlyone"),
            Some(LocalCommand::MoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("move"),
            Some(LocalCommand::MoveFile(None))
        ));
        assert!(matches!(
            parse_local_command("Move"),
            Some(LocalCommand::MoveFile(None))
        ));
        match parse_local_command("/move_file old.rs new.rs") {
            Some(LocalCommand::MoveFile(Some((src, dst)))) => {
                assert_eq!(src.as_ref(), "old.rs");
                assert_eq!(dst.as_ref(), "new.rs");
            }
            _ => panic!("expected move_file with source and destination"),
        }
        match parse_local_command("move old.rs new.rs") {
            Some(LocalCommand::MoveFile(Some((src, dst)))) => {
                assert_eq!(src.as_ref(), "old.rs");
                assert_eq!(dst.as_ref(), "new.rs");
            }
            _ => panic!("expected natural move with source and destination"),
        }
        match parse_local_command("move file old.rs new.rs") {
            Some(LocalCommand::MoveFile(Some((src, dst)))) => {
                assert_eq!(src.as_ref(), "old.rs");
                assert_eq!(dst.as_ref(), "new.rs");
            }
            _ => panic!("expected move file prefix"),
        }
        match parse_local_command("git mv old.rs new.rs") {
            Some(LocalCommand::MoveFile(Some((src, dst)))) => {
                assert_eq!(src.as_ref(), "old.rs");
                assert_eq!(dst.as_ref(), "new.rs");
            }
            _ => panic!("expected git mv natural language"),
        }
    }

    #[test]
    fn parses_cherry_pick_commands() {
        assert!(matches!(
            parse_local_command("/cherry_pick"),
            Some(LocalCommand::CherryPick(None))
        ));
        match parse_local_command("/cherry_pick abc1234") {
            Some(LocalCommand::CherryPick(Some(commit))) => {
                assert_eq!(commit.as_ref(), "abc1234");
            }
            _ => panic!("expected cherry_pick with commit"),
        }
        match parse_local_command("cherry pick abc1234") {
            Some(LocalCommand::CherryPick(Some(commit))) => {
                assert_eq!(commit.as_ref(), "abc1234");
            }
            _ => panic!("expected natural cherry pick with commit"),
        }
        match parse_local_command("cherry-pick abc1234") {
            Some(LocalCommand::CherryPick(Some(commit))) => {
                assert_eq!(commit.as_ref(), "abc1234");
            }
            _ => panic!("expected cherry-pick with commit"),
        }
        match parse_local_command("git cherry-pick abc1234") {
            Some(LocalCommand::CherryPick(Some(commit))) => {
                assert_eq!(commit.as_ref(), "abc1234");
            }
            _ => panic!("expected git cherry-pick with commit"),
        }
        assert!(matches!(
            parse_local_command("cherry pick"),
            Some(LocalCommand::CherryPick(None))
        ));
        assert!(matches!(
            parse_local_command("cherry-pick"),
            Some(LocalCommand::CherryPick(None))
        ));
    }

    #[test]
    fn parses_commit_commands() {
        assert!(matches!(
            parse_local_command("/commit"),
            Some(LocalCommand::Commit(None))
        ));
        assert!(matches!(
            parse_local_command("commit"),
            Some(LocalCommand::Commit(None))
        ));
        match parse_local_command("/commit [#42] My feature") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected commit with plain message"),
        }
        match parse_local_command("/commit \"[#42] My feature\"") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected commit with double-quoted message"),
        }
        match parse_local_command("Commit \"[#42] My feature\"") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected natural commit with quoted message"),
        }
        match parse_local_command("commit [#42] My feature") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected natural commit without quotes"),
        }
        match parse_local_command("git commit -a -m \"[#42] My feature\"") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected git commit -a -m with quoted message"),
        }
        match parse_local_command("git commit -m fixed") {
            Some(LocalCommand::Commit(Some(msg))) => {
                assert_eq!(msg.as_ref(), "fixed");
            }
            _ => panic!("expected git commit -m form"),
        }
    }

    #[test]
    fn parses_amend_commands() {
        assert!(matches!(
            parse_local_command("/amend"),
            Some(LocalCommand::Amend(None))
        ));
        assert!(matches!(
            parse_local_command("amend"),
            Some(LocalCommand::Amend(None))
        ));
        assert!(matches!(
            parse_local_command("git amend"),
            Some(LocalCommand::Amend(None))
        ));
        assert!(matches!(
            parse_local_command("git commit --amend"),
            Some(LocalCommand::Amend(None))
        ));
        match parse_local_command("/amend [#42] My feature") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected amend with plain message"),
        }
        match parse_local_command("/amend \"[#42] My feature\"") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected amend with double-quoted message"),
        }
        match parse_local_command("amend \"[#42] My feature\"") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected natural amend with quoted message"),
        }
        match parse_local_command("amend message \"[#42] My feature\"") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected amend message form"),
        }
        match parse_local_command("git commit --amend -m \"[#42] My feature\"") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected git commit --amend -m form"),
        }
        match parse_local_command("git amend \"[#42] My feature\"") {
            Some(LocalCommand::Amend(Some(msg))) => {
                assert_eq!(msg.as_ref(), "[#42] My feature");
            }
            _ => panic!("expected git amend form"),
        }
    }

    #[test]
    fn parses_push_commands() {
        assert!(matches!(
            parse_local_command("/push"),
            Some(LocalCommand::Push(false))
        ));
        assert!(matches!(
            parse_local_command("/push --force"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("/push -f"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("/push force"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("push"),
            Some(LocalCommand::Push(false))
        ));
        assert!(matches!(
            parse_local_command("Push"),
            Some(LocalCommand::Push(false))
        ));
        assert!(matches!(
            parse_local_command("git push"),
            Some(LocalCommand::Push(false))
        ));
        assert!(matches!(
            parse_local_command("force push"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("push force"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("push --force"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("git push --force"),
            Some(LocalCommand::Push(true))
        ));
        assert!(matches!(
            parse_local_command("git push origin --force"),
            Some(LocalCommand::Push(true))
        ));
    }

    #[test]
    fn parses_init_repo_commands() {
        assert!(matches!(
            parse_local_command("/init_repo"),
            Some(LocalCommand::InitRepo)
        ));
        assert!(matches!(
            parse_local_command("init"),
            Some(LocalCommand::InitRepo)
        ));
        assert!(matches!(
            parse_local_command("Init"),
            Some(LocalCommand::InitRepo)
        ));
        assert!(matches!(
            parse_local_command("init repo"),
            Some(LocalCommand::InitRepo)
        ));
        assert!(matches!(
            parse_local_command("Init Repo"),
            Some(LocalCommand::InitRepo)
        ));
        assert!(matches!(
            parse_local_command("git init"),
            Some(LocalCommand::InitRepo)
        ));
    }

    #[test]
    fn parses_delete_branch_commands() {
        assert!(matches!(
            parse_local_command("/branch -d feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("delete feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("Delete feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("delete branch feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("Delete Branch feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("git branch -D feature/foo"),
            Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
        ));
        assert!(matches!(
            parse_local_command("delete branch"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
        assert!(matches!(
            parse_local_command("delete"),
            Some(LocalCommand::Branch(BranchSubcommand::List))
        ));
    }

    #[test]
    fn parses_squash_commands() {
        assert!(matches!(
            parse_local_command("/squash"),
            Some(LocalCommand::Squash)
        ));
        assert!(matches!(
            parse_local_command("squash"),
            Some(LocalCommand::Squash)
        ));
        assert!(matches!(
            parse_local_command("Squash"),
            Some(LocalCommand::Squash)
        ));
        assert!(matches!(
            parse_local_command("squash branch"),
            Some(LocalCommand::Squash)
        ));
        assert!(matches!(
            parse_local_command("squash commits"),
            Some(LocalCommand::Squash)
        ));
        assert!(matches!(
            parse_local_command("git squash"),
            Some(LocalCommand::Squash)
        ));
    }

    #[test]
    fn splits_editor_command_and_flags() {
        assert_eq!(
            shell_words("code --wait").expect("editor command"),
            vec!["code".to_string(), "--wait".to_string()]
        );
        assert_eq!(
            shell_words("\"/tmp/my editor\" --flag").expect("quoted editor command"),
            vec!["/tmp/my editor".to_string(), "--flag".to_string()]
        );
    }
}

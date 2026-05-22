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
use orangu::{config::LlmConfiguration, session::ChatSession, tools::ToolExecutor};
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
    Output(String),
    Cleared,
    Quit,
}

pub enum LocalCommand<'a> {
    Help,
    ConnectDefault,
    ConnectTo(&'a str),
    Disconnect,
    Reload,
    ListModels,
    ListFiles,
    ShowFile(Cow<'a, str>),
    Tools,
    ModelInfo,
    SetModel(&'a str),
    Diff(Option<Cow<'a, str>>),
    Status,
    Log,
    Pull(Option<u64>),
    Rebase,
    Merge(Option<Cow<'a, str>>),
    Checkout(Option<Cow<'a, str>>),
    AddFile(Option<Cow<'a, str>>),
    RemoveFile(Option<Cow<'a, str>>),
    MoveFile(Option<(Cow<'a, str>, Cow<'a, str>)>),
    CherryPick(Option<Cow<'a, str>>),
    Commit(Option<Cow<'a, str>>),
    Amend(Option<Cow<'a, str>>),
    Push(bool),
    InitRepo,
    Squash,
    DeleteBranch(Option<Cow<'a, str>>),
    OpenFile(&'a str),
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
}

pub struct CommandState<'a> {
    pub active_model: &'a mut String,
    pub current_endpoint: &'a mut Option<String>,
    pub session: &'a mut ChatSession,
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
        "/list_models" => Some(LocalCommand::ListModels),
        "/list_files" => Some(LocalCommand::ListFiles),
        "/show_file" => Some(LocalCommand::ShowFile(Cow::Borrowed(""))),
        "/open_file" => Some(LocalCommand::OpenFile("")),
        "/tools" => Some(LocalCommand::Tools),
        "/model" => Some(LocalCommand::ModelInfo),
        "/diff" => Some(LocalCommand::Diff(None)),
        "/status" => Some(LocalCommand::Status),
        "/log" => Some(LocalCommand::Log),
        "/pull" => Some(LocalCommand::Pull(None)),
        "/rebase" => Some(LocalCommand::Rebase),
        "/merge" => Some(LocalCommand::Merge(None)),
        "/checkout" => Some(LocalCommand::Checkout(None)),
        "/add_file" => Some(LocalCommand::AddFile(None)),
        "/remove_file" => Some(LocalCommand::RemoveFile(None)),
        "/move_file" => Some(LocalCommand::MoveFile(None)),
        "/cherry_pick" => Some(LocalCommand::CherryPick(None)),
        "/commit" => Some(LocalCommand::Commit(None)),
        "/amend" => Some(LocalCommand::Amend(None)),
        "/push" => Some(LocalCommand::Push(false)),
        "/init_repo" => Some(LocalCommand::InitRepo),
        "/squash" => Some(LocalCommand::Squash),
        "/delete" => Some(LocalCommand::DeleteBranch(None)),
        "/sessions" => Some(LocalCommand::Sessions(None)),
        "/usage" => Some(LocalCommand::Usage),
        "/build" => Some(LocalCommand::Build),
        "/clear" => Some(LocalCommand::Clear),
        "/quit" => Some(LocalCommand::Quit),
        _ => {
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
            if let Some(name) = input.strip_prefix("/model ") {
                return Some(LocalCommand::SetModel(name.trim()));
            }
            if let Some(args) = input.strip_prefix("/show_file ") {
                return Some(LocalCommand::ShowFile(Cow::Borrowed(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/pull ") {
                return Some(LocalCommand::Pull(args.trim().parse::<u64>().ok()));
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
                return Some(LocalCommand::Checkout(if target.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(target))
                }));
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
            if let Some(args) = input.strip_prefix("/delete ") {
                let branch = args.trim();
                return Some(LocalCommand::DeleteBranch(if branch.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(branch))
                }));
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
    if matches_ci(
        input,
        &[
            "list models",
            "show models",
            "show available models",
            "models",
        ],
    ) {
        return Some(LocalCommand::ListModels);
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
    if matches_ci(input, &["status", "show status", "git status"]) {
        return Some(LocalCommand::Status);
    }
    if matches_ci(input, &["log", "show log", "git log", "git lg"]) {
        return Some(LocalCommand::Log);
    }
    for prefix in [
        "use model ",
        "switch model to ",
        "set model to ",
        "select model ",
    ] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::SetModel(name.trim()));
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
    for prefix in ["git checkout ", "checkout "] {
        if let Some(target) = strip_ascii_prefix(input, prefix) {
            let target = target.trim();
            if !target.is_empty() {
                return Some(LocalCommand::Checkout(Some(Cow::Borrowed(target))));
            }
        }
    }
    if let Some(target) = strip_ascii_prefix(input, "switch to ") {
        let target = strip_ascii_suffix(target.trim(), " branch")
            .map(str::trim)
            .unwrap_or(target.trim());
        if !target.is_empty() {
            return Some(LocalCommand::Checkout(Some(Cow::Borrowed(target))));
        }
    }
    if matches_ci(input, &["checkout", "switch to"]) {
        return Some(LocalCommand::Checkout(None));
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
        return Some(LocalCommand::DeleteBranch(None));
    }
    for prefix in ["git branch -D ", "delete branch ", "delete "] {
        if let Some(branch) = strip_ascii_prefix(input, prefix) {
            let branch = branch.trim();
            if !branch.is_empty() {
                return Some(LocalCommand::DeleteBranch(Some(Cow::Borrowed(branch))));
            }
        }
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

pub fn format_models(llms: &HashMap<String, LlmConfiguration>) -> String {
    let mut names = sorted_model_names(llms);
    let mut lines = Vec::with_capacity(names.len());
    for name in names.drain(..) {
        if let Some(llm) = llms.get(&name) {
            lines.push(format!("- {}: {} ({})", name, llm.model, llm.provider));
        }
    }
    lines.join("\n")
}

pub fn open_file_usage_message() -> &'static str {
    "Usage: /open_file <path>. Use /help to see available commands."
}

pub fn model_usage_message() -> &'static str {
    "Usage: /model <name>. Use /help to see available commands."
}

pub fn connect_usage_message() -> &'static str {
    "Usage: /connect <endpoint>. Use /help to see available commands."
}

pub fn pull_usage_message() -> &'static str {
    "Usage: /pull <number>. Use /help to see available commands."
}

pub fn merge_usage_message() -> &'static str {
    "Usage: /merge <branch>. Use /help to see available commands."
}

pub fn checkout_usage_message() -> &'static str {
    "Usage: /checkout <branch|file>. Use /help to see available commands."
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

pub fn delete_branch_usage_message() -> &'static str {
    "Usage: /delete <branch>"
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
            Some(LocalCommand::ListModels)
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
            Some(LocalCommand::SetModel(name)) => assert_eq!(name, "local"),
            _ => panic!("expected set model command"),
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
            Some(LocalCommand::Log)
        ));
        assert!(matches!(
            parse_local_command("log"),
            Some(LocalCommand::Log)
        ));
        assert!(matches!(
            parse_local_command("Log"),
            Some(LocalCommand::Log)
        ));
        assert!(matches!(
            parse_local_command("show log"),
            Some(LocalCommand::Log)
        ));
        assert!(matches!(
            parse_local_command("git log"),
            Some(LocalCommand::Log)
        ));
        assert!(matches!(
            parse_local_command("git lg"),
            Some(LocalCommand::Log)
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
    fn parses_checkout_commands() {
        assert!(matches!(
            parse_local_command("/checkout"),
            Some(LocalCommand::Checkout(None))
        ));
        assert!(matches!(
            parse_local_command("/checkout "),
            Some(LocalCommand::Checkout(None))
        ));
        assert!(matches!(
            parse_local_command("checkout"),
            Some(LocalCommand::Checkout(None))
        ));
        assert!(matches!(
            parse_local_command("Checkout"),
            Some(LocalCommand::Checkout(None))
        ));
        match parse_local_command("/checkout feature/foo") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected checkout with branch"),
        }
        match parse_local_command("checkout feature/foo") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected natural checkout with branch"),
        }
        match parse_local_command("Checkout README.md") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "README.md")
            }
            _ => panic!("expected case-insensitive checkout with file"),
        }
        match parse_local_command("git checkout feature/foo") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected git checkout natural language"),
        }
        match parse_local_command("switch to main") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "main")
            }
            _ => panic!("expected switch to main"),
        }
        match parse_local_command("Switch to main") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "main")
            }
            _ => panic!("expected case-insensitive switch to main"),
        }
        match parse_local_command("switch to feature/foo") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "feature/foo")
            }
            _ => panic!("expected switch to feature/foo"),
        }
        match parse_local_command("switch to main branch") {
            Some(LocalCommand::Checkout(Some(target))) => {
                assert_eq!(target.as_ref(), "main")
            }
            _ => panic!("expected switch to main branch -> main"),
        }
        assert!(matches!(
            parse_local_command("switch to"),
            Some(LocalCommand::Checkout(None))
        ));
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
            parse_local_command("/delete feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("/delete"),
            Some(LocalCommand::DeleteBranch(None))
        ));
        assert!(matches!(
            parse_local_command("delete feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("Delete feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("delete branch feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("Delete Branch feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("git branch -D feature/foo"),
            Some(LocalCommand::DeleteBranch(Some(_)))
        ));
        assert!(matches!(
            parse_local_command("delete branch"),
            Some(LocalCommand::DeleteBranch(None))
        ));
        assert!(matches!(
            parse_local_command("delete"),
            Some(LocalCommand::DeleteBranch(None))
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

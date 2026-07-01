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
use std::{
    borrow::Cow,
    collections::HashMap,
    path::{Path, PathBuf},
};
use terminal_size::{Width, terminal_size};

mod args;
mod natural;
mod slash;

pub use args::*;
pub use natural::*;
pub use slash::*;

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
    /// The input was intercepted locally and should be replaced with the
    /// provided prompt text before sending anything to the model.
    OverridePrompt(String),
    Quiet,
    /// Command ran and produced informational output (success).
    Output(String),
    /// Output rendered as Markdown.
    MarkdownOutput(String),
    OutputWithLlmContext {
        display: String,
        llm_context: String,
    },
    WideOutputWithLlmContext {
        display: String,
        llm_context: String,
    },
    /// Command failed — invalid usage, unknown command, or other error.
    OutputError(String),
    Cleared,
    Quit,
    Restart,
    /// Re-exec into a different existing session, resuming the given UUID.
    SwitchSession(String),
    /// Re-exec into a different workspace directory, starting (or auto-resuming)
    /// a session there.
    SwitchWorkspace(PathBuf),
    /// Switch to the open workspace tab at the given full-order index
    /// (`/workspace <number>`).
    SwitchWorkspaceTab(usize),
    /// Switch the active tab's workspace to the given path in-place, or switch
    /// to the existing tab that already has it open (`/workspace <path>`).
    ChangeWorkspace(PathBuf),
    /// Open the given directory in a new workspace tab (`/create_workspace <dir>`).
    OpenWorkspaceTab(PathBuf),
    /// Close the current workspace tab (`/delete_workspace`).
    CloseWorkspaceTab,
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
    /// Enter the LLM-driven `/auto_review` mode with a collected branch diff.
    AutoReview(ReviewLaunch),
    /// Scan the workspace for duplicated functions at the given similarity
    /// threshold (`0.0`–`1.0`), then show the report and cache it so a later
    /// `/export duplicates` can reuse it without re-scanning. Run off the UI
    /// thread (like [`CommandOutcome::Blocking`]) since the scan can be slow.
    Duplicates(f64),
    /// Write the console output window or the last review report to a PDF in
    /// the workspace root.
    Export(ExportTarget),
    /// Enter the built-in manual viewer.
    Manual,
    /// List the commands queued while a request is in flight (`/pending`).
    PendingList,
    /// Remove the queued command at the given 1-based index
    /// (`/pending delete <n>`).
    PendingDelete(usize),
}

/// Data handed to the interactive review mode: the changed files, each with its
/// own rendered diff lines.
pub struct ReviewLaunch {
    pub files: Vec<ReviewEntry>,
    /// `/auto_review` only: start the run at once, skipping the pre-start phase
    /// (the `immediate` argument). Always `false` for `/review`.
    pub immediate: bool,
}

/// The `/auto_review` keyword argument that starts the run at once, skipping
/// the pre-start phase. Offered by Tab completion and the inline ghost.
pub const AUTO_REVIEW_IMMEDIATE: &str = "immediate";

/// The `/comment` keyword that submits the last `/review` summary as the
/// comment body. Matched case-insensitively against the whole argument, so a
/// `~/.orangu/comments/` template whose name merely starts with `w` is still
/// a filename.
pub const COMMENT_REVIEW_KEYWORD: &str = "with review";

/// The `/comment` keyword that submits the last `/auto_review` report as the
/// comment body.
pub const COMMENT_AUTO_REVIEW_KEYWORD: &str = "with auto review";

pub enum CommentBody<'a> {
    /// An inline comment body supplied directly in the command (`"..."` or bare text).
    Inline(Cow<'a, str>),
    /// A filename under `~/.orangu/comments/` whose content is the comment body.
    File(Cow<'a, str>),
    /// The last `/review` summary (`with review`).
    Review,
    /// The last `/auto_review` report (`with auto review`).
    AutoReview,
}

pub enum CloseTarget {
    Issue(u64),
    PullRequest(u64),
}

/// A field of an issue or pull/merge request that `/issue` adds a value to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueField {
    /// A review request (pull/merge requests only).
    Reviewer,
    /// An assigned user (issues and pull/merge requests).
    Assignee,
    /// A label (issues and pull/merge requests).
    Label,
}

/// The `/issue` subcommand names, in offer order, used for parsing and Tab
/// completion. Kept in step with [`IssueField`].
pub const ISSUE_FIELDS: [&str; 3] = ["reviewer", "assignee", "label"];

impl IssueField {
    /// Parse a subcommand word (case-insensitive, exact) into a field.
    pub fn parse(word: &str) -> Option<IssueField> {
        match word.to_ascii_lowercase().as_str() {
            "reviewer" => Some(IssueField::Reviewer),
            "assignee" => Some(IssueField::Assignee),
            "label" => Some(IssueField::Label),
            _ => None,
        }
    }

    /// The subcommand word for this field.
    pub fn as_str(self) -> &'static str {
        match self {
            IssueField::Reviewer => "reviewer",
            IssueField::Assignee => "assignee",
            IssueField::Label => "label",
        }
    }
}

/// A parsed `/issue <field> <number> <value>` command: add `value` to `field`
/// of issue/PR `number`. The number may be an issue or a pull/merge request —
/// which it is is detected against the forge when the command runs.
pub struct IssueAction<'a> {
    pub field: IssueField,
    pub number: u64,
    pub value: Cow<'a, str>,
}

pub enum GetCommentsTarget {
    Issue(u64),
    PullRequest(u64),
}

pub enum PruneTarget {
    Uuid(String),
    Workspace(String),
    OlderThan(u64),
    All,
}

pub enum StashSubcommand {
    Push,
    Pop,
    List,
    Drop,
}

/// What the `/export` tool should write to PDF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportTarget {
    /// The console output window (`export console`, the default).
    Console,
    /// The last `/review` report — or, if none, the last `/auto_review` report
    /// (`export review`).
    Review,
    /// The last `/auto_review` report specifically (`export auto review`).
    AutoReview,
    /// A fresh duplicate-code report for the workspace (`export duplicates`),
    /// computed at export time.
    Duplicates,
}

/// The `/export` target words, in offer order, used for Tab completion and the
/// inline ghost. Kept in step with [`parse_export_target`].
pub const EXPORT_TARGETS: [&str; 4] = ["console", "review", "auto review", "duplicates"];

/// A `/bisect` subcommand, wrapping `git bisect`. The `Start`, `Good`, `Bad`,
/// and `Skip` variants carry an optional commit/rev argument; when it is `None`
/// the corresponding `git bisect` command runs against the current `HEAD`.
pub enum BisectSubcommand<'a> {
    /// `start [<bad> [<good>...]]` — begin a session, optionally seeding the
    /// known-bad and known-good revisions.
    Start(Option<Cow<'a, str>>),
    /// `good [<commit>]` — mark a commit (default `HEAD`) as good.
    Good(Option<Cow<'a, str>>),
    /// `bad [<commit>]` — mark a commit (default `HEAD`) as bad.
    Bad(Option<Cow<'a, str>>),
    /// `skip [<commit>]` — skip a commit (default `HEAD`) that cannot be tested.
    Skip(Option<Cow<'a, str>>),
    /// `reset` — end the session and return to the original branch.
    Reset,
    /// `log` — print the commits marked so far.
    Log,
    /// `status` — report whether a session is in progress (the default for a
    /// bare `/bisect`).
    Status,
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

    SetVerbosity(&'a str),
    Diff(Option<Cow<'a, str>>),
    Grep(Option<Cow<'a, str>>),
    Review,
    /// `/auto_review [<file>] [immediate]`: an optional single-file target and
    /// whether to start the run at once (the `immediate` keyword).
    AutoReview(Option<Cow<'a, str>>, bool),
    /// `/duplicates [<threshold>]`: scan the workspace for duplicated functions
    /// (across every language in [`orangu::duplicates::LANGUAGES`]) and print a
    /// similarity report. The optional argument overrides
    /// the default similarity threshold (a percentage `1`–`100`, or a fraction
    /// `0.0`–`1.0`); `None` uses [`orangu::duplicates::DEFAULT_THRESHOLD`].
    Duplicates(Option<f64>),
    Status,
    Log(Option<u64>),
    /// `/show [<commit>]`: show a single commit (header plus diff). `None`
    /// defaults to `HEAD`.
    Show(Option<Cow<'a, str>>),
    Fetch(Option<Cow<'a, str>>),
    Pull(Option<u64>),
    Comment(Option<(u64, CommentBody<'a>)>),
    Close(Option<CloseTarget>),
    /// `/issue <reviewer|assignee|label> <number> <value>`: add a reviewer,
    /// assignee, or label to an issue or pull/merge request. `None` is a usage
    /// error (missing or malformed arguments).
    Issue(Option<IssueAction<'a>>),
    GetComments(Option<GetCommentsTarget>),
    Prune(Option<PruneTarget>),
    CreatePullRequest,
    Rebase(Option<Cow<'a, str>>),
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
    Bisect(BisectSubcommand<'a>),
    OpenFile(&'a str),
    Session(Option<Cow<'a, str>>),
    /// `/workspace [<number>|<path>]`: with no argument, report the active
    /// workspace; with a number, switch to that tab; with anything else, open
    /// (or switch to) that directory. The number-vs-path split happens in
    /// dispatch, mirroring how `Session` disambiguates its argument.
    Workspace(Option<Cow<'a, str>>),
    /// `/create_workspace <dir>`: open a new tab for the given directory.
    CreateWorkspace(Cow<'a, str>),
    /// `/delete_workspace`: close the current workspace tab.
    DeleteWorkspace,
    Export(ExportTarget),
    Manual,
    Usage,
    /// `/build [debug|release]`: build the workspace project. Defaults to
    /// `release`.
    Build(crate::build::BuildProfile),
    Clear,
    Quit,
    PendingList,
    PendingDelete(Option<usize>),
    Skills,
    Graph,
}

pub struct CommandContext<'a> {
    pub startup_model: &'a str,
    pub startup_endpoint: &'a str,
    pub llms: &'a HashMap<String, LlmConfiguration>,
    pub tools: &'a ToolExecutor,
    pub workspace: &'a Path,
    pub session_dir: &'a std::path::Path,
    pub usage_stats: &'a crate::UsageStats,
    pub available_models: &'a [String],
    pub virtual_width: usize,
    pub auto_rebase: bool,
    pub auto_squash: bool,
    pub compression: bool,
    pub terminal: &'a str,
    pub forge: crate::git::Forge,
    pub review_reports: crate::git::ReviewReports<'a>,
    pub skills: &'a orangu::skills::SkillRegistry,
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

pub fn system_prompt<'a>(
    profile: &'a LlmConfiguration,
    verbosity_override: Option<&str>,
) -> Cow<'a, str> {
    let mut base = if profile.system_prompt.is_empty() {
        Cow::Borrowed(
            "You are Orangu, a coding environment assistant connected to a local workspace. Use the available local tools to inspect files, edit files on disk, fetch external URLs for knowledge, and run shell commands when needed. Be precise, explain what you changed, and surface tool failures explicitly.",
        )
    } else {
        Cow::Borrowed(profile.system_prompt.as_str())
    };

    let verbosity = verbosity_override.or(profile.model_verbosity.as_deref());
    if let Some(v) = verbosity {
        if v == "terse" {
            let mut s = base.into_owned();
            s.push_str("\n\nBe extremely terse and concise. Do not restate context. Provide only the requested answer or code.");
            base = Cow::Owned(s);
        } else if v == "verbose" {
            let mut s = base.into_owned();
            s.push_str(
                "\n\nBe very detailed and explain your thought process and any decisions made.",
            );
            base = Cow::Owned(s);
        }
    }

    base
}

pub fn build_workspace_system_prompt(
    profile: &LlmConfiguration,
    skills: &orangu::skills::SkillRegistry,
    workspace: &Path,
    verbosity_override: Option<&str>,
) -> String {
    let base = system_prompt(profile, verbosity_override);
    let index = skills.system_prompt_index();
    let mut ep = if index.is_empty() {
        base.into_owned()
    } else {
        format!("{base}\n\n{index}")
    };
    ep.push_str(&orangu::config::load_agents_instructions(workspace));
    ep
}

pub fn sorted_model_names(llms: &HashMap<String, LlmConfiguration>) -> Vec<String> {
    let mut names = llms.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

#[cfg(test)]
mod tests;

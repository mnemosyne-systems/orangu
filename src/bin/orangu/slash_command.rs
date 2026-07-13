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

use strum_macros::{AsRefStr, EnumIter, EnumString, IntoStaticStr};

/// Commands that can be invoked by starting a message with a leading slash.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString, EnumIter, AsRefStr, IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum SlashCommand {
    Help,
    Disconnect,
    Reload,
    Restart,
    ListFiles,
    ShowFile,
    Tools,
    Model,
    Server,
    Information,
    Diff,
    Grep,
    Search,
    Review,
    Status,
    Log,
    Show,
    Fetch,
    Pull,
    Comment,
    Close,
    Issue,
    GetComments,
    Prune,
    Rebase,
    Merge,
    Branch,
    Restore,
    AutoReview,
    Duplicates,
    Export,
    CreateFile,
    DeleteFile,
    MoveFile,
    CreateDirectory,
    MoveDirectory,
    DeleteDirectory,
    CherryPick,
    Commit,
    Amend,
    PullRequest,
    Push,
    InitRepo,
    Squash,
    Stash,
    Bisect,
    OpenFile,
    Pending,
    Session,
    Workspace,
    CreateWorkspace,
    DeleteWorkspace,
    Manual,
    Usage,
    Statistics,
    Schedule,
    Build,
    Shell,
    Skills,
    Graph,
    Clear,
    Theme,
    Quit,
}

impl SlashCommand {
    /// User-visible description shown in the popup.
    pub fn description(self) -> &'static str {
        match self {
            SlashCommand::Help => "Shows the list of available commands",
            SlashCommand::Disconnect => "Disconnects from the current server",
            SlashCommand::Reload => "Restores configured model and server",
            SlashCommand::Restart => "Restarts orangu in place",
            SlashCommand::ListFiles => "Lists the workspace files as a tree",
            SlashCommand::ShowFile => "Shows the contents of a workspace file",
            SlashCommand::Tools => "Lists the model-facing workspace tools",
            SlashCommand::Model => "Selects the model used for requests",
            SlashCommand::Server => "Selects the server orangu talks to",
            SlashCommand::Information => {
                "Reports which OpenAI/orangu-server capabilities are available"
            }
            SlashCommand::Diff => "Shows git diff",
            SlashCommand::Grep => "Searches using git grep",
            SlashCommand::Search => "Semantic code search over the embedding index",
            SlashCommand::Review => "Opens a split view for code review",
            SlashCommand::Status => "Shows git status",
            SlashCommand::Log => "Shows git log",
            SlashCommand::Show => "Shows a commit (git show)",
            SlashCommand::Fetch => "Fetches from remote",
            SlashCommand::Pull => "Pulls from remote",
            SlashCommand::Comment => "Creates a comment",
            SlashCommand::Close => "Closes an issue or PR",
            SlashCommand::Issue => "Manages issues",
            SlashCommand::GetComments => "Fetches comments",
            SlashCommand::Prune => "Deletes older session directories",
            SlashCommand::Rebase => "Rebases the current branch",
            SlashCommand::Merge => "Merges a branch",
            SlashCommand::Branch => "Manages git branches",
            SlashCommand::Restore => "Restores working tree files",
            SlashCommand::AutoReview => "Runs an LLM-driven branch review",
            SlashCommand::Duplicates => "Finds duplicated Rust/C functions",
            SlashCommand::Export => "Writes session or review to a PDF",
            SlashCommand::CreateFile => {
                "Creates a file, with optional content and permissions (also stages an existing one)"
            }
            SlashCommand::DeleteFile => "Deletes a file",
            SlashCommand::MoveFile => "Moves a file",
            SlashCommand::CreateDirectory => "Creates a directory, with optional permissions",
            SlashCommand::MoveDirectory => "Moves a directory and everything under it",
            SlashCommand::DeleteDirectory => "Deletes an empty directory",
            SlashCommand::CherryPick => "Cherry-picks a commit",
            SlashCommand::Commit => "Commits changes",
            SlashCommand::Amend => "Amends the last commit",
            SlashCommand::PullRequest => "Creates a pull request",
            SlashCommand::Push => "Pushes to remote",
            SlashCommand::InitRepo => "Initializes a git repository",
            SlashCommand::Squash => "Squashes commits",
            SlashCommand::Stash => "Manages git stash",
            SlashCommand::Bisect => "Manages git bisect",
            SlashCommand::OpenFile => "Opens a workspace file in your editor",
            SlashCommand::Pending => "Manages pending commands",
            SlashCommand::Session => "Lists, switches, and opens sessions",
            SlashCommand::Workspace => "Manages workspaces",
            SlashCommand::CreateWorkspace => "Creates a workspace",
            SlashCommand::DeleteWorkspace => "Closes the active workspace tab",
            SlashCommand::Manual => "Opens the built-in manual",
            SlashCommand::Usage => "Shows token usage statistics",
            SlashCommand::Statistics => {
                "Shows persistent, cross-session activity history and a daily heatmap"
            }
            SlashCommand::Schedule => "Lists the cron-style scheduled jobs from ~/.orangu/schedule",
            SlashCommand::Build => "Builds the workspace project",
            SlashCommand::Shell => "Executes a shell command",
            SlashCommand::Skills => "Lists the discovered Agent Skills",
            SlashCommand::Graph => {
                "Generates an HTML visualization of the codebase Knowledge Graph"
            }
            SlashCommand::Clear => "Clears the terminal screen",
            SlashCommand::Theme => "Switches the active TUI theme",
            SlashCommand::Quit => "Exits the application",
        }
    }

    /// Command string with the leading '/'.
    pub fn command(self) -> String {
        let name: &'static str = self.into();
        format!("/{}", name)
    }
}

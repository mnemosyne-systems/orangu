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
use std::borrow::Cow;

/// Every natural-language trigger phrase recognised by
/// [`parse_natural_language_command`], in the same priority order the parser
/// tries them. Phrases that take an argument keep their trailing space (e.g.
/// `"diff against "`); bare aliases do not (e.g. `"diff"`).
///
/// Entries are split into `// ---`-commented groups, one per command the
/// phrases trigger, to make it obvious where a new alias belongs. To add a
/// phrase, drop it into the matching group; order still matters, both across
/// groups and within one, because it sets the parser's match priority and the
/// order the grey ghost hint cycles through. Add a new group only when you add
/// a new command, keeping it in the same priority position the parser uses.
///
/// This drives the grey inline "ghost" completion (see
/// `completion::natural_language_ghost_candidates`), so it must stay in sync
/// with the parser below. The `binding_phrases_all_parse` test guards against
/// drift.
pub const NATURAL_LANGUAGE_BINDINGS: &[&str] = &[
    // --- help ---
    "help",
    "show help",
    "show commands",
    "show available commands",
    // --- manual ---
    "manual",
    "show manual",
    "open manual",
    // --- disconnect ---
    "disconnect",
    // --- reload configuration ---
    "reload",
    "reload configuration",
    // --- reset session / restart ---
    "reset session",
    "restart",
    "restart orangu",
    // --- list files ---
    "list files",
    "show files",
    "show workspace files",
    "list workspace files",
    // --- tools ---
    "show tools",
    "list tools",
    "show local tools",
    "tools",
    // --- model (current) ---
    "show model",
    "current model",
    "what model am i using",
    "model",
    // --- models (list) ---
    "list models",
    "show models",
    "show available models",
    "models",
    // --- build ---
    "build",
    "build project",
    "run build",
    // --- diff ---
    "diff",
    "show diff",
    "git diff",
    "diff against ",
    "show diff against ",
    "git diff ",
    // --- review ---
    "review",
    "review changes",
    "code review",
    "review branch",
    // --- auto review ---
    "auto review",
    // --- export ---
    "export",
    "export console",
    "export review",
    // --- status ---
    "status",
    "show status",
    "git status",
    // --- grep ---
    "grep ",
    "find ",
    "git grep ",
    // --- log ---
    "log ",
    "show log ",
    "git log ",
    "git lg ",
    "log",
    "show log",
    "git log",
    "git lg",
    // --- fetch ---
    "fetch",
    "git fetch",
    "fetch ",
    "git fetch ",
    // --- server (select) ---
    "use server ",
    "switch server to ",
    "set server to ",
    "select server ",
    // --- model (select) ---
    "use model ",
    "switch model to ",
    "set model to ",
    "select model ",
    // --- open / edit / show file ---
    "open file ",
    "open ",
    "edit file ",
    "edit ",
    "show file ",
    "show ",
    // --- pull (fetch pull request) ---
    "pull request ",
    "pull pr ",
    "pull #",
    "pull ",
    // --- comment on pull request / issue ---
    "add comment on ",
    "add comment to ",
    "comment on ",
    // --- create pull request ---
    "pull request",
    "create pull request",
    "open pull request",
    "new pull request",
    "create pr",
    "open pr",
    "new pr",
    // --- close issue / pull request ---
    "close issue ",
    "close -i ",
    "close pr ",
    "close pull request ",
    "close -p ",
    // --- get comments for issue / pull request ---
    "get comments for issue ",
    "get comments for pull request ",
    // --- stash ---
    "stash",
    "git stash",
    "git stash push",
    "stash pop",
    "pop stash",
    "git stash pop",
    "stash list",
    "list stashes",
    "git stash list",
    "stash drop",
    "drop stash",
    "git stash drop",
    // --- bisect ---
    "bisect start",
    "start bisect",
    "git bisect start",
    "bisect good",
    "mark good",
    "git bisect good",
    "bisect bad",
    "mark bad",
    "git bisect bad",
    "bisect skip",
    "skip commit",
    "git bisect skip",
    "bisect reset",
    "reset bisect",
    "git bisect reset",
    "bisect log",
    "git bisect log",
    "bisect status",
    "bisect",
    "git bisect",
    // --- rebase ---
    "rebase",
    "git rebase",
    "rebase ",
    "git rebase ",
    // --- merge ---
    "git merge ",
    "merge ",
    "merge",
    // --- checkout ---
    "git checkout ",
    "checkout ",
    "switch to branch ",
    "switch to ",
    // --- create branch ---
    "create branch ",
    "new branch ",
    "branch -b ",
    // --- rename branch ---
    "rename branch to ",
    "rename to ",
    "branch -m ",
    // --- list branches / checkout ---
    "branch",
    "list branches",
    "git branch",
    "checkout",
    "list all branches",
    "branch -a",
    "branch --all",
    // --- restore ---
    "restore ",
    "git restore ",
    // --- add ---
    "git add ",
    "add file ",
    "add ",
    "add",
    // --- remove ---
    "git rm ",
    "remove file ",
    "remove ",
    "remove",
    // --- move ---
    "git mv ",
    "move file ",
    "move ",
    "move",
    // --- cherry-pick ---
    "git cherry-pick ",
    "cherry-pick ",
    "cherry pick ",
    "cherry pick",
    "cherry-pick",
    // --- commit ---
    "git commit -a -m ",
    "git commit -m ",
    "commit ",
    "commit",
    // --- amend ---
    "git commit --amend -m ",
    "git amend -m ",
    "git amend ",
    "amend message ",
    "amend ",
    "amend",
    "git amend",
    "git commit --amend",
    // --- push ---
    "force push",
    "push force",
    "push --force",
    "push -f",
    "git push --force",
    "git push -f",
    "git push origin --force",
    "git push origin -f",
    "push",
    "git push",
    "git push origin",
    // --- init ---
    "init",
    "init repo",
    "git init",
    // --- squash ---
    "squash",
    "squash branch",
    "squash commits",
    "git squash",
    // --- delete branch ---
    "delete",
    "delete branch",
    "git branch -D ",
    "delete branch ",
    "delete ",
    // --- session ---
    "session",
    "switch session",
    "sessions",
    "list sessions",
    "show sessions",
    // --- prune ---
    "prune session ",
    "prune sessions older than ",
    "prune sessions in ",
    "prune all",
    "prune",
    // --- pending ---
    "pending",
    "list pending",
    "pending list",
    "show pending",
    // --- usage ---
    "usage",
    "show usage",
    // --- clear conversation ---
    "clear",
    "clear conversation",
    "reset conversation",
    // --- quit ---
    "quit",
    "exit",
];

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
    // Checked before the `open `/`show ` file prefixes below, so `open manual`
    // and `show manual` open the manual rather than a file of that name.
    if matches_ci(input, &["manual", "show manual", "open manual"]) {
        return Some(LocalCommand::Manual);
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
    if let Some(file) = strip_ascii_prefix(input, "auto review ") {
        let file = file.trim();
        if !file.is_empty() {
            return Some(LocalCommand::AutoReview(Some(Cow::Borrowed(file))));
        }
    }
    if matches_ci(input, &["auto review"]) {
        return Some(LocalCommand::AutoReview(None));
    }
    // Checked before the more specific buffers: "export console" / "export
    // review" select a buffer, while a bare "export" defaults to the console.
    if matches_ci(input, &["export review"]) {
        return Some(LocalCommand::Export(ExportTarget::Review));
    }
    if matches_ci(input, &["export", "export console"]) {
        return Some(LocalCommand::Export(ExportTarget::Console));
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
    for prefix in ["fetch ", "git fetch "] {
        if let Some(remote) = strip_ascii_prefix(input, prefix) {
            let remote = remote.trim();
            if !remote.is_empty() {
                return Some(LocalCommand::Fetch(Some(Cow::Borrowed(remote))));
            }
        }
    }
    if matches_ci(input, &["fetch", "git fetch"]) {
        return Some(LocalCommand::Fetch(None));
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
    // `show pending` is the pending-queue command, not a request to show a file
    // named "pending"; resolve it before the `show <file>` natural form claims
    // it. (The other pending phrasings do not collide and are handled below.)
    if matches_ci(input, &["show pending"]) {
        return Some(LocalCommand::PendingList);
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
    if let Some(rest) = strip_ascii_prefix(input, "get comments for issue ") {
        let n = rest.trim().trim_start_matches('#').parse::<u64>().ok();
        return Some(LocalCommand::GetComments(n.map(GetCommentsTarget::Issue)));
    }
    if let Some(rest) = strip_ascii_prefix(input, "get comments for pull request ") {
        let n = rest.trim().trim_start_matches('#').parse::<u64>().ok();
        return Some(LocalCommand::GetComments(
            n.map(GetCommentsTarget::PullRequest),
        ));
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
    // bisect: each commit-taking subcommand tries its `<verb> ` prefixes first
    // to capture an optional commit/rev argument, then falls back to the
    // argument-less aliases (`mark good`, `mark bad`, `skip commit`,
    // `reset bisect`, …). A bare `bisect`/`git bisect` reports status, the same
    // as `/bisect`.
    for prefix in ["bisect start ", "start bisect ", "git bisect start "] {
        if let Some(args) = strip_ascii_prefix(input, prefix) {
            let args = args.trim();
            return Some(LocalCommand::Bisect(BisectSubcommand::Start(
                if args.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(args))
                },
            )));
        }
    }
    if matches_ci(input, &["bisect start", "start bisect", "git bisect start"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Start(None)));
    }
    for prefix in ["bisect good ", "git bisect good "] {
        if let Some(commit) = strip_ascii_prefix(input, prefix) {
            let commit = commit.trim();
            return Some(LocalCommand::Bisect(BisectSubcommand::Good(
                if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                },
            )));
        }
    }
    if matches_ci(input, &["bisect good", "mark good", "git bisect good"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Good(None)));
    }
    for prefix in ["bisect bad ", "git bisect bad "] {
        if let Some(commit) = strip_ascii_prefix(input, prefix) {
            let commit = commit.trim();
            return Some(LocalCommand::Bisect(BisectSubcommand::Bad(
                if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                },
            )));
        }
    }
    if matches_ci(input, &["bisect bad", "mark bad", "git bisect bad"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Bad(None)));
    }
    for prefix in ["bisect skip ", "git bisect skip "] {
        if let Some(commit) = strip_ascii_prefix(input, prefix) {
            let commit = commit.trim();
            return Some(LocalCommand::Bisect(BisectSubcommand::Skip(
                if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                },
            )));
        }
    }
    if matches_ci(input, &["bisect skip", "skip commit", "git bisect skip"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Skip(None)));
    }
    if matches_ci(input, &["bisect reset", "reset bisect", "git bisect reset"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Reset));
    }
    if matches_ci(input, &["bisect log", "git bisect log"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Log));
    }
    if matches_ci(input, &["bisect status", "bisect", "git bisect"]) {
        return Some(LocalCommand::Bisect(BisectSubcommand::Status));
    }
    for prefix in ["rebase ", "git rebase "] {
        if let Some(target) = strip_ascii_prefix(input, prefix) {
            let target = target.trim();
            if !target.is_empty() {
                return Some(LocalCommand::Rebase(Some(Cow::Borrowed(target))));
            }
        }
    }
    if matches_ci(input, &["rebase", "git rebase"]) {
        return Some(LocalCommand::Rebase(None));
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
    if matches_ci(
        input,
        &[
            "session",
            "switch session",
            "sessions",
            "list sessions",
            "show sessions",
        ],
    ) {
        return Some(LocalCommand::Session(None));
    }
    // `workspace <number>` / `switch workspace <path>` — the argument forms are
    // checked before the bare-word forms so `workspace 1` is a switch, not a
    // no-op listing. `switch workspace` does not collide with the `switch to`
    // branch-checkout prefix.
    for prefix in ["switch workspace ", "workspace "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            let arg = rest.trim();
            if !arg.is_empty() {
                return Some(LocalCommand::Workspace(Some(Cow::Borrowed(arg))));
            }
        }
    }
    if matches_ci(input, &["workspace", "workspaces", "switch workspace"]) {
        return Some(LocalCommand::Workspace(None));
    }
    for prefix in ["prune session ", "prune sessions older than "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::Prune(parse_prune_args(rest.trim())));
        }
    }
    if let Some(rest) = strip_ascii_prefix(input, "prune sessions in ") {
        let path = rest.trim();
        if !path.is_empty() {
            return Some(LocalCommand::Prune(Some(PruneTarget::Workspace(
                path.to_string(),
            ))));
        }
    }
    if matches_ci(input, &["prune all"]) {
        return Some(LocalCommand::Prune(Some(PruneTarget::All)));
    }
    if matches_ci(input, &["prune"]) {
        return Some(LocalCommand::Prune(None));
    }
    if matches_ci(
        input,
        &["pending", "list pending", "pending list", "show pending"],
    ) {
        return Some(LocalCommand::PendingList);
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

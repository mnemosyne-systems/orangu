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

use std::path::Path;

use crate::git::{
    discover_git_root, git_branch_names, git_local_branch_names, is_protected_branch,
};

mod bisect;
mod files;
mod ghost;
mod git_refs;
mod pull;
mod session;

pub use bisect::*;
pub use files::*;
pub use ghost::*;
pub use git_refs::*;
pub use pull::*;
pub(crate) use session::*;

pub const COMMANDS: &[&str] = &[
    "/help",
    "/disconnect",
    "/reload",
    "/restart",
    "/list_files",
    "/show_file",
    "/tools",
    "/model",
    "/server",
    "/diff",
    "/grep",
    "/review",
    "/status",
    "/log",
    "/fetch",
    "/pull",
    "/comment",
    "/close",
    "/get_comments",
    "/prune",
    "/rebase",
    "/merge",
    "/branch",
    "/restore",
    "/add_file",
    "/auto_review",
    "/export",
    "/remove_file",
    "/move_file",
    "/cherry_pick",
    "/commit",
    "/amend",
    "/pull_request",
    "/push",
    "/init_repo",
    "/squash",
    "/stash",
    "/bisect",
    "/open_file",
    "/pending",
    "/session",
    "/workspace",
    "/manual",
    "/usage",
    "/build",
    "/clear",
    "/quit",
];

pub fn completion_candidates(
    input: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
) -> Option<(usize, usize, Vec<String>)> {
    let cursor = cursor.min(input.len());
    let prefix = &input[..cursor];

    if let Some(result) =
        structured_completion_candidates(prefix, cursor, workspace, server_names, available_models)
    {
        return Some(result);
    }

    if prefix.starts_with('/') {
        return Some((
            0,
            cursor,
            COMMANDS
                .iter()
                .filter(|command| command.starts_with(prefix))
                .map(|command| (*command).to_string())
                .collect(),
        ));
    }

    let start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    let token = &prefix[start..];
    Some((start, cursor, file_completion_candidates(token, workspace)))
}

/// The grey inline "ghost" suffix previewing the first structured completion
/// candidate at the cursor, e.g. `switch to branch m` -> `ain` (completing the
/// `main` branch) or `/server lo` -> `cal`. This complements the command and
/// natural-language ghosts so branch, tag, file, model, and server arguments get
/// the same inline hint that Tab fills in.
///
/// Limited to the structured completions (those tied to a recognised command);
/// the generic last-word file completion that also fires for ordinary prose is
/// deliberately excluded so plain prompts are not littered with hints. Returns
/// `None` unless the cursor sits at the end of the input and the first candidate
/// extends the typed token rather than rewriting it.
pub fn completion_ghost_suffix(
    input: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
) -> Option<String> {
    if cursor != input.len() {
        return None;
    }
    let (start, end, candidates) =
        structured_completion_candidates(input, cursor, workspace, server_names, available_models)?;
    let candidate = candidates.into_iter().next()?;
    let typed = input.get(start..end)?;
    candidate
        .strip_prefix(typed)
        .filter(|rest| !rest.is_empty())
        .map(str::to_string)
}

/// The completion candidates tied to a recognised command (branch, tag, file,
/// model, server, session, ... arguments). Returns `None` when the input is not
/// one of those forms, leaving [`completion_candidates`] to fall back to the
/// slash-command list or generic file completion. Split out so the inline ghost
/// hint can reuse exactly these structured candidates without the prose-noisy
/// fallback.
fn structured_completion_candidates(
    prefix: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
) -> Option<(usize, usize, Vec<String>)> {
    if let Some((start, candidates)) = show_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = auto_review_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, path_prefix)) = open_file_completion_prefix(prefix) {
        return Some((
            start,
            cursor,
            open_file_completion_candidates(path_prefix, workspace),
        ));
    }

    if let Some((start, path_prefix)) = natural_show_file_completion_prefix(prefix) {
        return Some((
            start,
            cursor,
            open_file_completion_candidates(path_prefix, workspace),
        ));
    }

    if let Some(model_prefix) = prefix.strip_prefix("/model ") {
        return Some((
            7,
            cursor,
            available_models
                .iter()
                .filter(|model| model.starts_with(model_prefix))
                .cloned()
                .collect(),
        ));
    }

    if let Some(server_prefix) = prefix.strip_prefix("/server ") {
        return Some((
            8,
            cursor,
            server_names
                .iter()
                .filter(|server| server.starts_with(server_prefix))
                .cloned()
                .collect(),
        ));
    }

    if let Some((start, token)) = pull_completion_prefix(prefix) {
        return Some((start, cursor, pull_number_candidates(token)));
    }

    if let Some((start, candidates)) = comment_file_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = checkout_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = add_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = remove_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = move_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = cherry_pick_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = fetch_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = rebase_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = bisect_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, branch_prefix)) = diff_completion_prefix(prefix) {
        let branches = discover_git_root(workspace)
            .map(|root| {
                let local = git_local_branch_names(&root);
                let all = git_branch_names(&root);
                let local_set: std::collections::HashSet<&str> =
                    local.iter().map(String::as_str).collect();
                let remote_only: Vec<String> = all
                    .into_iter()
                    .filter(|b| !local_set.contains(b.as_str()))
                    .collect();
                local.into_iter().chain(remote_only).collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|b| b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some((start, branch_prefix)) = merge_completion_prefix(prefix) {
        let branches = discover_git_root(workspace)
            .map(|root| {
                let local = git_local_branch_names(&root);
                let all = git_branch_names(&root);
                let local_set: std::collections::HashSet<&str> =
                    local.iter().map(String::as_str).collect();
                let remote_only: Vec<String> = all
                    .into_iter()
                    .filter(|b| !local_set.contains(b.as_str()))
                    .collect();
                local.into_iter().chain(remote_only).collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|b| b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some((start, branch_prefix)) = delete_branch_completion_prefix(prefix) {
        let branches = discover_git_root(workspace)
            .map(|root| git_local_branch_names(&root))
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !is_protected_branch(b) && b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some(arg_prefix) = prefix.strip_prefix("/session ") {
        // `/session <arg>` takes either a session UUID (to switch) or a workspace
        // path (to filter the listing), so offer both: UUIDs first, newest-first,
        // then the unique workspace paths.
        let mut candidates: Vec<String> = session_uuids_newest_first()
            .into_iter()
            .filter(|u| u.starts_with(arg_prefix))
            .collect();
        candidates.extend(
            session_workspaces_newest_first()
                .into_iter()
                .filter(|w| w.starts_with(arg_prefix)),
        );
        // When the argument matches no session UUID or known workspace, fall back
        // to filesystem directory completion so a brand-new workspace can be
        // navigated to (e.g. `/session ~/Pr<TAB>/po<TAB>`).
        if candidates.is_empty() {
            candidates = session_path_completion_candidates(arg_prefix);
        }
        return Some(("/session ".len(), cursor, candidates));
    }

    if let Some(arg_prefix) = prefix.strip_prefix("/workspace ") {
        // `/workspace <arg>` takes a tab number or a directory. A bare number is
        // typed directly, so completion only helps with the directory form:
        // offer the workspaces seen in past sessions first, then fall back to
        // filesystem directory completion so a brand-new directory can be
        // navigated to (e.g. `/workspace ~/Pr<TAB>/po<TAB>`).
        let mut candidates: Vec<String> = session_workspaces_newest_first()
            .into_iter()
            .filter(|w| w.starts_with(arg_prefix))
            .collect();
        if candidates.is_empty() {
            candidates = session_path_completion_candidates(arg_prefix);
        }
        return Some(("/workspace ".len(), cursor, candidates));
    }

    if let Some(uuid_prefix) = prefix.strip_prefix("/prune ")
        && !uuid_prefix.starts_with('-')
        && !uuid_prefix.eq_ignore_ascii_case("all")
    {
        let candidates = session_uuids_newest_first()
            .into_iter()
            .filter(|u| u.starts_with(uuid_prefix))
            .collect();
        return Some(("/prune ".len(), cursor, candidates));
    }

    None
}

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_review_command_completes_from_the_command_list() {
        // `/auto_review` is registered in the completion list...
        assert!(COMMANDS.contains(&"/auto_review"));
        // ...the inline ghost hint completes it once the prefix is unambiguous...
        assert_eq!(command_ghost_suffix("/auto"), Some("_review"));
        // ...and the natural-language ghost knows the alias too.
        assert!(natural_language_ghost_candidates("auto re").contains(&"view"));
    }

    #[test]
    fn every_completion_command_parses_as_a_slash_command() {
        // The completion list is maintained by hand; every entry must stay a
        // real command, so a typo or a renamed command fails here instead of
        // silently completing to something the parser rejects.
        for command in COMMANDS {
            assert!(
                crate::commands::parse_slash_command(command).is_some(),
                "completion entry {command:?} does not parse as a slash command"
            );
        }
    }
}

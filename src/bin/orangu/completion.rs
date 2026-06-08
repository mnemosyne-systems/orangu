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

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::RwLock,
};
use walkdir::WalkDir;

use super::commands::{NATURAL_LANGUAGE_BINDINGS, shell_words, strip_ascii_prefix};
use super::git::PullRequest;
use super::git::{
    discover_git_root, git_branch_names, git_file_commit_hashes, git_local_branch_names,
    git_tag_names, is_protected_branch,
};

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
    "/pull",
    "/comment",
    "/close",
    "/rebase",
    "/merge",
    "/branch",
    "/restore",
    "/add_file",
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
    "/open_file",
    "/session",
    "/usage",
    "/build",
    "/clear",
    "/quit",
];

/// Open pull/merge requests fetched once at startup (see
/// `crate::git::fetch_active_pull_requests`) and cached here in memory, so `/pull`
/// completion can offer numbers without shelling out to `gh`/`glab` on every
/// keystroke. Holds the request number paired with its title; only the number is
/// ever inserted into the prompt, the title is kept for future menu display.
static ACTIVE_PULL_REQUESTS: RwLock<Vec<(u64, String)>> = RwLock::new(Vec::new());

/// Replace the cached open pull/merge requests. Called once when the startup
/// fetch finishes; a poisoned lock is recovered rather than panicking, since a
/// stale cache is harmless.
pub fn set_active_pull_requests(requests: &[PullRequest]) {
    let mut guard = ACTIVE_PULL_REQUESTS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = requests
        .iter()
        .map(|request| (request.number, request.title.clone()))
        .collect();
}

/// The cached open pull/merge request numbers whose decimal spelling starts with
/// `token`, as the strings `/pull` completion inserts. Numeric order, so the
/// lowest matching number is offered first.
fn pull_number_candidates(token: &str) -> Vec<String> {
    let guard = ACTIVE_PULL_REQUESTS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut numbers: Vec<u64> = guard
        .iter()
        .map(|(number, _)| *number)
        .filter(|number| number.to_string().starts_with(token))
        .collect();
    numbers.sort_unstable();
    numbers.iter().map(u64::to_string).collect()
}

/// For a `/pull <number>` argument (or its natural-language aliases `pull `,
/// `pull pr `, `pull request `, `pull #`), the offset where the number token
/// starts and the partial number typed so far. Mirrors the prefixes
/// `commands::parse_pull_pr_number` accepts, longest first so `pull request 5`
/// keeps `5` as the token rather than treating `request 5` as the number.
/// Returns `None` for anything else, including the bare `/pull` slash command
/// still being typed (no argument yet).
pub fn pull_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(number) = prefix.strip_prefix("/pull ") {
        return Some(("/pull ".len(), number));
    }
    for command_prefix in ["pull request ", "pull pr ", "pull #", "pull "] {
        if let Some(number) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - number.len(), number));
        }
    }
    None
}

/// Returns the trailing characters needed to finish the slash command the user
/// is part-way through typing, e.g. `/q` -> `uit` (completing `/quit`). This is
/// the grey "ghost" hint shown inline after the cursor; pressing Tab fills it in.
///
/// Returns `None` unless `input` is a lone slash-command prefix still being
/// typed (no whitespace yet) that matches a known command. The first matching
/// command in [`COMMANDS`] wins, so the suggestion narrows as more letters are
/// typed. An already-complete command yields `None`.
pub fn command_ghost_suffix(input: &str) -> Option<&'static str> {
    if !input.starts_with('/') || input.chars().any(char::is_whitespace) {
        return None;
    }
    let candidate = COMMANDS.iter().find(|command| command.starts_with(input))?;
    candidate
        .strip_prefix(input)
        .filter(|rest| !rest.is_empty())
}

/// Every natural-language binding the user's part-typed input could still grow
/// into, as the trailing characters needed to complete each one. For input `c`
/// this yields `urrent model`, `ode review`, `heckout `, ... (completing
/// `current model`, `code review`, `checkout `, ...). The list drives the grey "ghost" hint and
/// its Shift+Tab cycling; index 0 is what `natural_language_ghost_suffix`
/// returns.
///
/// Matching is ASCII case-insensitive, mirroring the parser, and candidates keep
/// [`NATURAL_LANGUAGE_BINDINGS`] (parser priority) order. Bindings that differ
/// only by trailing whitespace (e.g. `checkout ` vs `checkout`) render
/// identically, so only the first is kept. Empty input, slash input, and input
/// that already spells a complete binding (e.g. `status`, `diff`) yield an
/// empty list — there is nothing left to hint.
pub fn natural_language_ghost_candidates(input: &str) -> Vec<&'static str> {
    if input.is_empty() || input.starts_with('/') {
        return Vec::new();
    }
    if NATURAL_LANGUAGE_BINDINGS
        .iter()
        .any(|binding| binding.eq_ignore_ascii_case(input))
    {
        return Vec::new();
    }
    let mut seen: Vec<&str> = Vec::new();
    let mut candidates: Vec<&'static str> = Vec::new();
    for binding in NATURAL_LANGUAGE_BINDINGS {
        if binding.len() <= input.len()
            || !binding.as_bytes()[..input.len()].eq_ignore_ascii_case(input.as_bytes())
        {
            continue;
        }
        let suffix = &binding[input.len()..];
        if suffix.trim().is_empty() {
            continue;
        }
        let key = binding.trim_end();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        candidates.push(suffix);
    }
    candidates
}

/// The natural-language ghost suffix to preview at cycle position `index`,
/// wrapping around the candidate list (see [`natural_language_ghost_candidates`]).
/// `index` 0 is the first match (e.g. `c` -> `urrent model`, completing `current model`);
/// Shift+Tab advances it. This is the grey hint shown inline after the cursor.
pub fn natural_language_ghost_suffix_at(input: &str, index: usize) -> Option<&'static str> {
    let candidates = natural_language_ghost_candidates(input);
    if candidates.is_empty() {
        return None;
    }
    Some(candidates[index % candidates.len()])
}

/// The leading single word of a natural-language ghost `suffix`, including the
/// whitespace that trails it, so Tab accepts a multi-word binding one word at a
/// time. For `"h force"` (completing `push` then `force`) this is `"h "`; for a
/// suffix with no internal whitespace such as `"onnect"` it is the whole suffix.
/// Keeping the trailing space matters: accepting `pus` -> `push ` leaves the
/// ghost alive so the next word (`force`) can be previewed and accepted in turn.
pub fn first_ghost_word(suffix: &str) -> &str {
    let Some(word_start) = suffix.find(|ch: char| !ch.is_whitespace()) else {
        return suffix;
    };
    let Some(rel_end) = suffix[word_start..].find(char::is_whitespace) else {
        return suffix;
    };
    let word_end = word_start + rel_end;
    let next = suffix[word_end..]
        .find(|ch: char| !ch.is_whitespace())
        .map(|index| word_end + index)
        .unwrap_or(suffix.len());
    &suffix[..next]
}

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

    None
}

pub fn file_completion_candidates(token: &str, workspace: &Path) -> Vec<String> {
    let (directory, prefix) = match token.rsplit_once('/') {
        Some((directory, prefix)) => (directory, prefix),
        None => ("", token),
    };
    let gitignore = workspace_gitignore(workspace);
    let search_dir = if directory.is_empty() {
        workspace.to_path_buf()
    } else {
        workspace.join(directory)
    };

    let Ok(entries) = fs::read_dir(search_dir) else {
        return Vec::new();
    };

    let mut matches = entries
        .flatten()
        .filter_map(|entry| {
            let entry_type = entry.file_type().ok()?;
            if !should_include_completion_path(
                workspace,
                &entry.path(),
                entry_type.is_dir(),
                gitignore.as_ref(),
            ) {
                return None;
            }

            let file_name = entry.file_name().to_string_lossy().to_string();
            if !file_name.starts_with(prefix) {
                return None;
            }

            let suffix = if entry_type.is_dir() { "/" } else { "" };
            Some(if directory.is_empty() {
                format!("{file_name}{suffix}")
            } else {
                format!("{directory}/{file_name}{suffix}")
            })
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
}

pub fn show_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let remainder = prefix.strip_prefix("/show_file ")?;
    let (token_start, token) = last_shell_token(remainder);
    let previous = remainder[..token_start].trim_end();
    let previous_tokens = if previous.is_empty() {
        Vec::new()
    } else {
        shell_words(previous).unwrap_or_default()
    };
    let has_path = previous_tokens.iter().any(|value| !value.starts_with('-'));

    let mut candidates = if token.starts_with('-') {
        show_file_flag_candidates(token)
    } else if has_path {
        let path_str = previous_tokens
            .iter()
            .find(|t| !t.starts_with('-'))
            .map(String::as_str)
            .unwrap_or("");
        discover_git_root(workspace)
            .map(|root| {
                let resolved = if std::path::Path::new(path_str).is_absolute() {
                    std::path::PathBuf::from(path_str)
                } else {
                    workspace.join(path_str)
                };
                let relative = resolved
                    .strip_prefix(&root)
                    .unwrap_or(resolved.as_path())
                    .to_path_buf();
                git_file_commit_hashes(&root, &relative)
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|h| h.starts_with(token))
            .collect()
    } else {
        open_file_completion_candidates(token, workspace)
    };
    candidates.sort();
    candidates.dedup();
    Some(("/show_file ".len() + token_start, candidates))
}

pub fn open_file_completion_candidates(token: &str, workspace: &Path) -> Vec<String> {
    let (quoted, token) = match token.chars().next() {
        Some(quote @ '"') | Some(quote @ '\'') => (Some(quote), &token[quote.len_utf8()..]),
        _ => (None, token),
    };
    let gitignore = workspace_gitignore(workspace);

    let mut matches = WalkDir::new(workspace)
        .into_iter()
        .filter_entry(|entry| {
            should_include_completion_path(
                workspace,
                entry.path(),
                entry.file_type().is_dir(),
                gitignore.as_ref(),
            )
        })
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(workspace).ok()?;
            let relative = relative.to_string_lossy().replace('\\', "/");
            let file_name = entry.file_name().to_string_lossy();
            if !open_file_completion_matches(&relative, &file_name, token) {
                return None;
            }

            Some(match quoted {
                Some(quote) => format!("{quote}{relative}"),
                None => relative,
            })
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
}

pub fn open_file_completion_matches(relative: &str, file_name: &str, token: &str) -> bool {
    token.is_empty()
        || relative.starts_with(token)
        || (!token.contains('/') && file_name.starts_with(token))
}

pub fn checkout_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token, switch_form) = if let Some(rest) = prefix.strip_prefix("/branch ") {
        ("/branch ".len(), rest, false)
    } else if let Some(rest) = prefix.strip_prefix("/checkout ") {
        ("/checkout ".len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git checkout ") {
        (prefix.len() - rest.len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "checkout ") {
        (prefix.len() - rest.len(), rest, false)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "switch to branch ") {
        // Checked before the shorter `switch to ` so `switch to branch m` keeps
        // `m` as the token rather than treating `branch m` as the branch prefix.
        (prefix.len() - rest.len(), rest, true)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "switch to ") {
        (prefix.len() - rest.len(), rest, true)
    } else {
        return None;
    };

    let mut candidates: Vec<String> = discover_git_root(workspace)
        .map(|root| {
            let mut refs = git_branch_names(&root);
            if switch_form {
                refs.extend(git_tag_names(&root));
                refs.sort();
                refs.dedup();
            }
            refs
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.starts_with(token))
        .collect();

    if !switch_form {
        for file in file_completion_candidates(token, workspace) {
            if !candidates.contains(&file) {
                candidates.push(file);
            }
        }
    }

    Some((start, candidates))
}

pub fn add_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/add_file ") {
        ("/add_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git add ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "add file ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "add ") {
        (prefix.len() - rest.len(), rest)
    } else {
        return None;
    };

    let candidates = discover_git_root(workspace)
        .map(|root| git_untracked_candidates(&root, token))
        .unwrap_or_default();

    Some((start, candidates))
}

pub fn remove_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = if let Some(rest) = prefix.strip_prefix("/remove_file ") {
        ("/remove_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git rm ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "remove file ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "remove ") {
        (prefix.len() - rest.len(), rest)
    } else {
        return None;
    };

    let candidates = discover_git_root(workspace)
        .map(|root| git_tracked_candidates(&root, token))
        .unwrap_or_default();

    Some((start, candidates))
}

pub fn move_file_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (cmd_len, args) = if let Some(rest) = prefix.strip_prefix("/move_file ") {
        ("/move_file ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git mv ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "move file ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "move ") {
        (prefix.len() - rest.len(), rest)
    } else {
        return None;
    };

    let (token_start, token) = last_shell_token(args);
    let previous = args[..token_start].trim_end();
    let previous_count = if previous.is_empty() {
        0
    } else {
        shell_words(previous).unwrap_or_default().len()
    };

    let absolute_start = cmd_len + token_start;
    let candidates = if previous_count == 0 {
        discover_git_root(workspace)
            .map(|root| git_tracked_candidates(&root, token))
            .unwrap_or_default()
    } else {
        file_completion_candidates(token, workspace)
    };

    Some((absolute_start, candidates))
}

pub fn cherry_pick_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (cmd_len, token) = if let Some(rest) = prefix.strip_prefix("/cherry_pick ") {
        ("/cherry_pick ".len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "git cherry-pick ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "cherry-pick ") {
        (prefix.len() - rest.len(), rest)
    } else if let Some(rest) = strip_ascii_prefix(prefix, "cherry pick ") {
        (prefix.len() - rest.len(), rest)
    } else {
        return None;
    };
    let token = token.trim_start();
    let candidates = discover_git_root(workspace)
        .map(|root| git_commit_hashes(&root, token))
        .unwrap_or_default();
    Some((cmd_len, candidates))
}

pub fn diff_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/diff ") {
        return Some(("/diff ".len(), branch));
    }
    for command_prefix in ["diff against ", "show diff against ", "git diff "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn merge_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/merge ") {
        return Some(("/merge ".len(), branch));
    }
    for command_prefix in ["git merge ", "merge "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn delete_branch_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(branch) = prefix.strip_prefix("/delete ") {
        return Some(("/delete ".len(), branch));
    }
    for command_prefix in ["git branch -D ", "delete branch ", "delete "] {
        if let Some(branch) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - branch.len(), branch));
        }
    }
    None
}

pub fn git_untracked_candidates(repo_root: &Path, token: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["ls-files", "--others", "--exclude-standard", "--directory"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.is_empty() || !line.starts_with(token) {
            continue;
        }
        if line.ends_with('/') {
            dirs.push(line.to_string());
        } else {
            files.push(line.to_string());
        }
    }
    dirs.sort();
    files.sort();
    dirs.extend(files);
    dirs
}

pub fn git_tracked_candidates(repo_root: &Path, token: &str) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["ls-files"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut dirs = std::collections::BTreeSet::new();
    let mut files = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.is_empty() || !line.starts_with(token) {
            continue;
        }
        let rest = &line[token.len()..];
        if let Some(slash) = rest.find('/') {
            dirs.insert(format!("{}{}/", token, &rest[..slash]));
        } else {
            files.push(line.to_string());
        }
    }
    let mut result: Vec<String> = dirs.into_iter().collect();
    files.sort();
    result.extend(files);
    result
}

pub fn git_commit_hashes(repo_root: &Path, token: &str) -> Vec<String> {
    for branch in ["origin/main", "origin/master", "main", "master"] {
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", "--verify", branch])
            .output();
        if !matches!(check, Ok(ref o) if o.status.success()) {
            continue;
        }
        let Ok(output) = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["log", "--abbrev-commit", "--format=%h", branch])
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let hashes: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|h| !h.is_empty() && h.starts_with(token))
            .take(50)
            .map(str::to_string)
            .collect();
        if !hashes.is_empty() || token.is_empty() {
            return hashes;
        }
    }
    Vec::new()
}

pub fn workspace_gitignore(workspace: &Path) -> Option<Gitignore> {
    let ignore_root = discover_git_root(workspace).unwrap_or_else(|| workspace.to_path_buf());
    let mut builder = GitignoreBuilder::new(&ignore_root);
    let root_gitignore_path = ignore_root.join(".gitignore");
    if root_gitignore_path.is_file() {
        builder.add(root_gitignore_path);
    }
    let workspace_gitignore_path = workspace.join(".gitignore");
    if workspace != ignore_root && workspace_gitignore_path.is_file() {
        builder.add(workspace_gitignore_path);
    }
    builder.build().ok()
}

pub fn should_include_completion_path(
    workspace: &Path,
    path: &Path,
    is_dir: bool,
    gitignore: Option<&Gitignore>,
) -> bool {
    let Ok(relative) = path.strip_prefix(workspace) else {
        return false;
    };

    if gitignore.is_some_and(|matcher| {
        matcher
            .matched_path_or_any_parents(path, is_dir)
            .is_ignore()
    }) {
        return false;
    }

    if relative.as_os_str().is_empty() {
        return true;
    }

    let relative = relative.to_string_lossy().replace('\\', "/");
    !(relative == ".git"
        || relative.starts_with(".git/")
        || relative == "build"
        || relative.starts_with("build/")
        || relative == "target"
        || relative.starts_with("target/"))
}

pub fn last_shell_token(input: &str) -> (usize, &str) {
    let mut quote = None;
    let mut escaped = false;
    let mut token_start = 0;
    let mut in_token = false;

    for (index, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }

        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            } else if active_quote == '"' && ch == '\\' {
                escaped = true;
            }
            continue;
        }

        if ch.is_whitespace() {
            in_token = false;
            token_start = index + ch.len_utf8();
            continue;
        }

        if !in_token {
            token_start = index;
            in_token = true;
        }

        if ch == '"' || ch == '\'' {
            quote = Some(ch);
        } else if ch == '\\' {
            escaped = true;
        }
    }

    (token_start, &input[token_start..])
}

pub fn show_file_flag_candidates(token: &str) -> Vec<String> {
    ["--hash", "--author"]
        .into_iter()
        .filter(|flag| flag.starts_with(token))
        .map(str::to_string)
        .collect()
}

pub fn open_file_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(path_prefix) = prefix.strip_prefix("/open_file ") {
        return Some(("/open_file ".len(), path_prefix));
    }

    for command_prefix in ["open file ", "open ", "edit file ", "edit "] {
        if let Some(path_prefix) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - path_prefix.len(), path_prefix));
        }
    }

    None
}

fn session_uuids_newest_first() -> Vec<String> {
    let Some(home) = home::home_dir() else {
        return Vec::new();
    };
    let sessions_dir = home.join(".orangu/sessions");
    let Ok(entries) = std::fs::read_dir(&sessions_dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<(String, u64)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let mtime = e
                .metadata()
                .ok()?
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Some((name, mtime))
        })
        .collect();
    dirs.sort_by_key(|e| std::cmp::Reverse(e.1));
    dirs.into_iter().map(|(name, _)| name).collect()
}

/// The distinct workspace paths recorded across all sessions, ordered by the
/// most recently updated session that used each one, newest first. Drives the
/// workspace half of `/session <arg>` completion; empty workspaces (sessions
/// started outside a Git repository) are skipped.
fn session_workspaces_newest_first() -> Vec<String> {
    let Some(home) = home::home_dir() else {
        return Vec::new();
    };
    let sessions_dir = home.join(".orangu/sessions");
    let Ok(entries) = std::fs::read_dir(&sessions_dir) else {
        return Vec::new();
    };
    let mut rows: Vec<(String, u64)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let meta = crate::load_session_metadata(&e.path().join("metadata"))
                .ok()
                .flatten()?;
            if meta.workspace.is_empty() {
                return None;
            }
            Some((meta.workspace, meta.last_updated_at))
        })
        .collect();
    rows.sort_by_key(|(_, updated)| std::cmp::Reverse(*updated));
    let mut workspaces: Vec<String> = Vec::new();
    for (workspace, _) in rows {
        if !workspaces.contains(&workspace) {
            workspaces.push(workspace);
        }
    }
    workspaces
}

/// Filesystem directory completion for a `/session <path>` argument, used as a
/// fallback when the typed text matches no session UUID or known workspace, so a
/// new workspace can be navigated to. Only fires for path-like input (starting
/// with `~`, `/`, or `.`, or containing a `/`). A leading `~`/`~/` is expanded to
/// the home directory for the lookup but kept verbatim in the returned
/// candidates. Only directories are offered, since a workspace is always a
/// directory; candidates carry no trailing slash, matching how the user types
/// the next `/` segment themselves.
fn session_path_completion_candidates(arg: &str) -> Vec<String> {
    let looks_like_path =
        arg.starts_with('~') || arg.starts_with('/') || arg.starts_with('.') || arg.contains('/');
    if !looks_like_path {
        return Vec::new();
    }
    // Split into the directory portion already typed (kept verbatim in each
    // candidate) and the partial final segment being completed.
    let split = arg.rfind('/').map(|i| i + 1).unwrap_or(0);
    let (typed_dir, partial) = arg.split_at(split);
    let Ok(entries) = fs::read_dir(expand_tilde_dir(typed_dir)) else {
        return Vec::new();
    };
    let mut candidates: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            name.starts_with(partial)
                .then(|| format!("{typed_dir}{name}"))
        })
        .collect();
    candidates.sort();
    candidates
}

/// The real filesystem directory to scan for the already-typed directory portion
/// of a `/session` path argument, expanding a leading `~`/`~/` to the home
/// directory. An empty portion (the argument has no `/` yet) scans the current
/// directory.
fn expand_tilde_dir(typed_dir: &str) -> PathBuf {
    if typed_dir == "~" || typed_dir == "~/" {
        return home::home_dir().unwrap_or_else(|| PathBuf::from(typed_dir));
    }
    if let Some(rest) = typed_dir.strip_prefix("~/")
        && let Some(home) = home::home_dir()
    {
        return home.join(rest);
    }
    if typed_dir.is_empty() {
        return PathBuf::from(".");
    }
    PathBuf::from(typed_dir)
}

pub fn natural_show_file_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(path_prefix) = strip_ascii_prefix(prefix, "show file ") {
        return Some((prefix.len() - path_prefix.len(), path_prefix));
    }

    let path_prefix = strip_ascii_prefix(prefix, "show ")?;
    let (token_start, _) = last_shell_token(path_prefix);
    if token_start != 0 {
        return None;
    }

    Some((prefix.len() - path_prefix.len(), path_prefix))
}

/// Returns `(start, candidates)` for a comment command's `<number> <file-prefix>` argument
/// where the file argument is a bare word (no leading quote), completing against
/// `~/.orangu/comments/`. Handles both `/comment` and the natural-language forms
/// (`add comment on`, `add comment to`, `comment on`).
pub fn comment_file_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    let rest = if let Some(rest) = prefix.strip_prefix("/comment ") {
        rest
    } else {
        let mut found = None;
        for command_prefix in ["add comment on ", "add comment to ", "comment on "] {
            if let Some(rest) = strip_ascii_prefix(prefix, command_prefix) {
                found = Some(rest);
                break;
            }
        }
        found?
    };
    let rest = rest.trim_start();
    // skip the issue number token
    let (_, after_number) = rest.split_once(char::is_whitespace)?;
    let file_prefix = after_number.trim_start();
    // quoted argument = inline comment body, not a file
    if file_prefix.starts_with('"') || file_prefix.starts_with('\'') {
        return None;
    }
    let comments_dir = home::home_dir()?.join(".orangu/comments");
    let entries = fs::read_dir(comments_dir).ok()?;
    let mut candidates: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            if !entry.file_type().ok()?.is_file() {
                return None;
            }
            let name = entry.file_name().to_str()?.to_string();
            if name.starts_with(file_prefix) {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    candidates.sort();
    let start = prefix.len() - file_prefix.len();
    Some((start, candidates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn session_path_completion_lists_matching_subdirectories() {
        let root = tempdir().expect("tempdir");
        let base = root.path();
        std::fs::create_dir(base.join("PostgreSQL")).expect("dir");
        std::fs::create_dir(base.join("Postfix")).expect("dir");
        std::fs::create_dir(base.join("Redis")).expect("dir");
        std::fs::write(base.join("Postscript.txt"), b"x").expect("file");

        // The typed directory portion is kept verbatim; only directories whose
        // name extends the partial segment are offered, sorted, with no trailing
        // slash. The plain file is skipped.
        let prefix = format!("{}/Post", base.display());
        let candidates = session_path_completion_candidates(&prefix);
        assert_eq!(
            candidates,
            vec![
                format!("{}/Postfix", base.display()),
                format!("{}/PostgreSQL", base.display()),
            ]
        );
    }

    #[test]
    fn pull_completion_prefix_keeps_number_token_offset() {
        // The token offset must point at the number, not mid-command, so the
        // accepted candidate replaces just the `9` (e.g. `pull 9` -> `pull 90`)
        // rather than splicing the number into the middle of the command word.
        assert_eq!(pull_completion_prefix("/pull 9"), Some((6, "9")));
        assert_eq!(pull_completion_prefix("pull 9"), Some((5, "9")));
        assert_eq!(pull_completion_prefix("pull #9"), Some((6, "9")));
        assert_eq!(pull_completion_prefix("pull pr 9"), Some((8, "9")));
        assert_eq!(pull_completion_prefix("pull request 9"), Some((13, "9")));
        // Empty argument is offered (all numbers); bare slash command is not.
        assert_eq!(pull_completion_prefix("/pull "), Some((6, "")));
        assert_eq!(pull_completion_prefix("/pull"), None);
        assert_eq!(pull_completion_prefix("/pull_request"), None);
    }

    #[test]
    fn pull_number_candidates_filter_and_sort() {
        set_active_pull_requests(&[
            PullRequest {
                number: 90,
                title: "Add pull completion".to_string(),
            },
            PullRequest {
                number: 9,
                title: "Older".to_string(),
            },
            PullRequest {
                number: 58,
                title: "Fix rebase".to_string(),
            },
        ]);
        // `9` matches 9 and 90, numeric order; the candidate is the bare number.
        assert_eq!(pull_number_candidates("9"), vec!["9", "90"]);
        assert_eq!(pull_number_candidates("5"), vec!["58"]);
        assert!(pull_number_candidates("7").is_empty());
        // Empty token offers every cached number.
        assert_eq!(pull_number_candidates(""), vec!["9", "58", "90"]);
        set_active_pull_requests(&[]);
    }

    #[test]
    fn session_path_completion_ignores_non_path_arguments() {
        // A bare token (no separators, not `~`/`/`/`.`) is a UUID/workspace
        // prefix, not a path, so filesystem completion stays out of the way.
        assert!(session_path_completion_candidates("Postgre").is_empty());
    }
}

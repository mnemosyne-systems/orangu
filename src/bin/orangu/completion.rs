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
use std::{fs, path::Path};
use walkdir::WalkDir;

use super::commands::{shell_words, strip_ascii_prefix};
use super::git::{
    discover_git_root, git_branch_names, git_file_commit_hashes, git_local_branch_names,
    git_tag_names, is_protected_branch,
};

pub const COMMANDS: &[&str] = &[
    "/help",
    "/connect",
    "/disconnect",
    "/reload",
    "/restart",
    "/models",
    "/list_files",
    "/show_file",
    "/tools",
    "/model",
    "/diff",
    "/grep",
    "/review",
    "/status",
    "/log",
    "/pull",
    "/comment",
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
    "/sessions",
    "/usage",
    "/build",
    "/clear",
    "/quit",
];

pub fn completion_candidates(
    input: &str,
    cursor: usize,
    workspace: &Path,
    model_names: &[String],
) -> Option<(usize, usize, Vec<String>)> {
    let cursor = cursor.min(input.len());
    let prefix = &input[..cursor];

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
            model_names
                .iter()
                .filter(|model| model.starts_with(model_prefix))
                .cloned()
                .collect(),
        ));
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

    if let Some(uuid_prefix) = prefix.strip_prefix("/session ") {
        let candidates = session_uuids_newest_first()
            .into_iter()
            .filter(|u| u.starts_with(uuid_prefix))
            .collect();
        return Some(("/session ".len(), cursor, candidates));
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

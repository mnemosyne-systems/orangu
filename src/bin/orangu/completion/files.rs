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

use std::{fs, path::Path};
use walkdir::WalkDir;

use super::*;
use crate::commands::{
    AUTO_REVIEW_ALL, AUTO_REVIEW_DEEP, AUTO_REVIEW_IMMEDIATE, shell_words, strip_ascii_prefix,
};
use crate::git::{
    discover_git_root, git_current_branch, git_file_commit_hashes, is_protected_branch,
    review_changed_paths,
};

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

/// Tab completion for `/shell <command>`: the token currently being typed
/// completes against files in the workspace, one path segment at a time —
/// `/shell ./te` offers `./test/`, and once inside that directory `/shell
/// ./test/c` offers `./test/check.sh` — exactly like a real shell's filename
/// completion. Earlier words in the command line (the program name and any
/// prior arguments) are left alone; only the last (possibly quoted) token is
/// completed.
pub fn shell_completion_candidates(prefix: &str, workspace: &Path) -> Option<(usize, Vec<String>)> {
    let args = prefix.strip_prefix("/shell ")?;
    let (token_start, token) = last_shell_token(args);
    let absolute_start = "/shell ".len() + token_start;
    Some((absolute_start, file_completion_candidates(token, workspace)))
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

/// The auto-review file argument inside `prefix`, as `(token_start, token)`:
/// the slash command `/auto_review <file>` or the natural-language `auto review
/// <file>` (case-insensitive). `None` when `prefix` is neither.
fn auto_review_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(token) = prefix.strip_prefix("/auto_review ") {
        return Some(("/auto_review ".len(), token));
    }
    let token = strip_ascii_prefix(prefix, "auto review ")?;
    Some((prefix.len() - token.len(), token))
}

/// Tab completion for `/auto_review <file>` and its natural-language form
/// `auto review <file>`: the file argument completes by name, not by location —
/// typing `t` offers `src/tui.rs`. On main/master every tracked file is a
/// candidate (gitignored files excluded); on any other branch only the files
/// that differ from the default branch are, matching what a single-file
/// `/auto_review` will actually review there. Also offers the `immediate`,
/// `all`, and `deep` keywords (see `AUTO_REVIEW_IMMEDIATE`, `AUTO_REVIEW_ALL`,
/// and `AUTO_REVIEW_DEEP`) once their prefix is typed. Returns the token
/// start and the candidate relative paths, or `None` when `prefix` is not an
/// auto-review argument.
pub fn auto_review_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let (start, token) = auto_review_completion_prefix(prefix)?;

    // Outside a repository, or when the branch cannot be determined, fall back
    // to offering every file (as on main/master).
    let on_protected = discover_git_root(workspace)
        .and_then(|root| git_current_branch(&root).ok())
        .map(|branch| is_protected_branch(&branch))
        .unwrap_or(true);

    let mut candidates = if on_protected {
        open_file_completion_candidates(token, workspace)
    } else {
        review_changed_paths(workspace)
            .into_iter()
            .filter(|path| {
                let file_name = Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(path.as_str());
                open_file_completion_matches(path, file_name, token)
            })
            .collect()
    };
    // Offer the `immediate`, `all`, and `deep` keywords (start the run at
    // once, review every project file, start every file in Deep mode) when
    // the typed token is a prefix of one, so `/auto_review imm`, `/auto_review
    // al`, and `/auto_review de` complete — and ghost — to them.
    if !token.is_empty() && AUTO_REVIEW_IMMEDIATE.starts_with(&token.to_ascii_lowercase()) {
        candidates.push(AUTO_REVIEW_IMMEDIATE.to_string());
    }
    if !token.is_empty() && AUTO_REVIEW_ALL.starts_with(&token.to_ascii_lowercase()) {
        candidates.push(AUTO_REVIEW_ALL.to_string());
    }
    if !token.is_empty() && AUTO_REVIEW_DEEP.starts_with(&token.to_ascii_lowercase()) {
        candidates.push(AUTO_REVIEW_DEEP.to_string());
    }
    Some((start, candidates))
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

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

use std::sync::RwLock;

use crate::commands::{
    ISSUE_CREATE_COMMAND, ISSUE_CREATE_FLAGS, ISSUE_FIELDS, IssueField, strip_ascii_prefix,
};
use crate::git::IssueMetadata;

/// The repository's candidate reviewers, assignees, and labels, fetched once at
/// startup (see `crate::git::fetch_issue_metadata`) and cached here so `/issue`
/// value completion needs no network call on every keystroke.
static ISSUE_METADATA: RwLock<IssueMetadata> = RwLock::new(IssueMetadata {
    reviewers: Vec::new(),
    assignees: Vec::new(),
    labels: Vec::new(),
});

/// Replace the cached `/issue` metadata. Called once the startup fetch finishes;
/// a poisoned lock is recovered rather than panicking, since a stale cache only
/// affects completion hints.
pub fn set_issue_metadata(metadata: IssueMetadata) {
    let mut guard = ISSUE_METADATA
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = metadata;
}

/// The cached completion values for a `/issue` field whose decimal/text spelling
/// starts with `token`: reviewers and assignees are logins, labels are names.
fn issue_value_candidates(field: IssueField, token: &str) -> Vec<String> {
    let guard = ISSUE_METADATA
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let list = match field {
        IssueField::Reviewer => &guard.reviewers,
        IssueField::Assignee => &guard.assignees,
        IssueField::Label => &guard.labels,
    };
    list.iter()
        .filter(|value| value.starts_with(token))
        .cloned()
        .collect()
}

/// Tab/ghost completion for `/issue <field> <number> <value>` and
/// `/issue create <title> [--body <text>] [--label <label>] [--assignee
/// <user>]` (plus the natural-language `create issue ` / `new issue ` forms),
/// as `(token_start, candidates)`:
///
/// - typing the **field** offers `reviewer`, `assignee`, `label`, `create`;
/// - typing the **number** offers nothing (it is typed directly);
/// - typing the **value** offers the cached reviewers / assignees / labels for
///   the chosen field;
/// - after `create`, a `-`-prefixed token offers the `--body`/`--description`/
///   `--label`/`--assignee` flags, and the token after `--label`/`--assignee`
///   (or the value of a `--label=`/`--assignee=` token) offers the cached
///   labels/assignees. The title and body are typed directly.
///
/// So `/issue re` → `reviewer`, and `/issue reviewer 114 je` → the matching
/// logins. Returns `None` when `prefix` is not an `/issue` argument.
pub fn issue_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    // The natural-language create forms complete the same flags/values as the
    // slash form once the phrase itself is typed.
    for form in ["create issue ", "new issue "] {
        if let Some(rest) = strip_ascii_prefix(prefix, form) {
            let base = prefix.len() - rest.len();
            return Some(issue_create_completion(base, rest));
        }
    }
    let base = "/issue ".len();
    let after = prefix.strip_prefix("/issue ")?;

    // First token: the field. While it carries no trailing space it is still
    // being typed, so offer the field names (and `create`) that extend it.
    let Some(field_ws) = after.find(char::is_whitespace) else {
        let candidates = ISSUE_FIELDS
            .iter()
            .copied()
            .chain(std::iter::once(ISSUE_CREATE_COMMAND))
            .filter(|name| name.starts_with(after))
            .map(str::to_string)
            .collect();
        return Some((base, candidates));
    };
    let field = &after[..field_ws];

    // `/issue create ...`: flag and label/assignee-value completion. The match
    // is case-insensitive, mirroring the parser; any whitespace (space or tab)
    // may separate `create` from the title.
    if field.eq_ignore_ascii_case(ISSUE_CREATE_COMMAND) {
        let rest_with_ws = &after[field_ws..];
        let lead = rest_with_ws.len() - rest_with_ws.trim_start_matches([' ', '\t']).len();
        let rest = &rest_with_ws[lead..];
        return Some(issue_create_completion(base + field_ws + lead, rest));
    }

    // Second token: the number. No completion — but it must be complete (have a
    // trailing space) before the value can be.
    let rest = &after[field_ws..];
    let number_lead = rest.len() - rest.trim_start().len();
    let after_field = rest.trim_start();
    if after_field.is_empty() {
        return None;
    }
    let number_ws = after_field.find(char::is_whitespace)?;

    // Third token onward: the value (the rest of the line, so multi-word labels
    // match as one). Offer the cached values for the chosen field.
    let field = IssueField::parse(field)?;
    let value_rel = &after_field[number_ws..];
    let value_lead = value_rel.len() - value_rel.trim_start().len();
    let value = value_rel.trim_start();
    let value_start = base + field_ws + number_lead + number_ws + value_lead;
    Some((value_start, issue_value_candidates(field, value)))
}

/// Completion inside `/issue create ...` (or its natural-language forms), as
/// `(token_start, candidates)` with `base` the offset `rest` starts at: a
/// `-`-prefixed token offers the option flags; the token after `--label` /
/// `--assignee` (or the value of a `--label=` / `--assignee=` token) offers
/// the cached labels / assignees; the title and body are typed directly and
/// offer nothing.
fn issue_create_completion(base: usize, rest: &str) -> (usize, Vec<String>) {
    let token_start_in_rest = rest
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    let token = &rest[token_start_in_rest..];
    // `--flag=value`: complete the value part only.
    if let Some((flag, value)) = token.split_once('=') {
        let start = base + token_start_in_rest + flag.len() + 1;
        return (
            start,
            match flag {
                "--label" | "-l" => cached_labels(value),
                "--assignee" | "-a" => cached_assignees(value),
                _ => Vec::new(),
            },
        );
    }
    if token.starts_with('-') {
        let candidates = ISSUE_CREATE_FLAGS
            .iter()
            .filter(|flag| flag.starts_with(token))
            .map(|flag| (*flag).to_string())
            .collect();
        return (base + token_start_in_rest, candidates);
    }
    // The previous whitespace-delimited token decides value completion.
    let prev = rest[..token_start_in_rest]
        .split_whitespace()
        .next_back()
        .unwrap_or("");
    let (flag, _) = match prev.split_once('=') {
        Some((flag, _)) if flag.starts_with('-') => (flag, true),
        _ => (prev, false),
    };
    let start = base + token_start_in_rest;
    match flag {
        "--label" | "-l" => (start, cached_labels(token)),
        "--assignee" | "-a" => (start, cached_assignees(token)),
        _ => (start, Vec::new()),
    }
}

/// The cached label names starting with `token`.
fn cached_labels(token: &str) -> Vec<String> {
    let guard = ISSUE_METADATA
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .labels
        .iter()
        .filter(|label| label.starts_with(token))
        .cloned()
        .collect()
}

/// The cached assignee logins starting with `token`.
fn cached_assignees(token: &str) -> Vec<String> {
    let guard = ISSUE_METADATA
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .assignees
        .iter()
        .filter(|user| user.starts_with(token))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() {
        set_issue_metadata(IssueMetadata {
            reviewers: vec!["jesperpedersen".to_string(), "alice".to_string()],
            assignees: vec!["bob".to_string(), "jesperpedersen".to_string()],
            labels: vec!["bug".to_string(), "needs triage".to_string()],
        });
    }

    #[test]
    fn completes_the_field_subcommand() {
        let (start, candidates) = issue_completion_candidates("/issue re").expect("candidates");
        assert_eq!(start, "/issue ".len());
        assert_eq!(candidates, vec!["reviewer".to_string()]);

        // No prefix offers all three fields plus `create`.
        let (_, all) = issue_completion_candidates("/issue ").expect("candidates");
        assert_eq!(all, vec!["reviewer", "assignee", "label", "create"]);
    }

    #[test]
    fn offers_nothing_while_typing_the_number() {
        // Mid-number: still typing the second token, so no candidates.
        assert!(issue_completion_candidates("/issue reviewer 11").is_none());
        // Field token complete but the number not yet started.
        assert!(issue_completion_candidates("/issue reviewer ").is_none());
    }

    #[test]
    fn completes_the_value_per_field() {
        sample_metadata();

        // Reviewers come from the reviewer list; the token start points at the
        // value so the accepted candidate replaces just `je`.
        let (start, candidates) =
            issue_completion_candidates("/issue reviewer 114 je").expect("candidates");
        assert_eq!(start, "/issue reviewer 114 ".len());
        assert_eq!(candidates, vec!["jesperpedersen".to_string()]);

        // Assignees come from the assignee list.
        let (_, assignees) =
            issue_completion_candidates("/issue assignee 114 ").expect("candidates");
        assert_eq!(assignees, vec!["bob", "jesperpedersen"]);

        // Labels can carry spaces and still match as one value.
        let (start, labels) =
            issue_completion_candidates("/issue label 5 needs").expect("candidates");
        assert_eq!(start, "/issue label 5 ".len());
        assert_eq!(labels, vec!["needs triage".to_string()]);
    }

    #[test]
    fn ignores_non_issue_input() {
        assert!(issue_completion_candidates("/issuelike").is_none());
        assert!(issue_completion_candidates("/close 5").is_none());
    }

    #[test]
    fn create_offers_flags_and_cached_values() {
        sample_metadata();

        // A `-`-prefixed token offers the option flags.
        let (start, flags) =
            issue_completion_candidates("/issue create Crash --l").expect("candidates");
        assert_eq!(start, "/issue create Crash ".len());
        assert_eq!(flags, vec!["--label".to_string()]);

        // The token after `--label` offers the cached labels...
        let (start, labels) =
            issue_completion_candidates("/issue create Crash --label ").expect("candidates");
        assert_eq!(start, "/issue create Crash --label ".len());
        assert_eq!(labels, vec!["bug", "needs triage"]);

        // ...narrowed by what is typed, and likewise for assignees.
        let (_, narrowed) =
            issue_completion_candidates("/issue create Crash --label needs").expect("candidates");
        assert_eq!(narrowed, vec!["needs triage".to_string()]);
        let (_, users) =
            issue_completion_candidates("create issue Crash --assignee je").expect("candidates");
        assert_eq!(users, vec!["jesperpedersen".to_string()]);

        // The `--flag=value` form completes the value part only.
        let (start, valued) =
            issue_completion_candidates("/issue create Crash --label=bu").expect("candidates");
        assert_eq!(start, "/issue create Crash --label=".len());
        assert_eq!(valued, vec!["bug".to_string()]);

        // The title itself offers nothing to replace it with.
        let (_, title) = issue_completion_candidates("/issue create Crash").expect("candidates");
        assert!(title.is_empty());
    }
}

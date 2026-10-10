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

/// Parse the optional `/export` argument. An empty argument defaults to the
/// console; `console` and `review` select their buffers; anything else is
/// rejected (returns `None`).
pub fn parse_export_target(arg: &str) -> Option<ExportTarget> {
    match arg.trim().to_ascii_lowercase().as_str() {
        "" | "console" => Some(ExportTarget::Console),
        "review" => Some(ExportTarget::Review),
        "auto review" | "auto_review" | "auto-review" => Some(ExportTarget::AutoReview),
        "duplicates" => Some(ExportTarget::Duplicates),
        "pr" | "pull requests" | "pull_requests" | "pull-requests" => Some(ExportTarget::Pr),
        "issue" | "issues" => Some(ExportTarget::Issue),
        "statistics" => Some(ExportTarget::Statistics(false)),
        "statistics total" => Some(ExportTarget::Statistics(true)),
        _ => None,
    }
}

/// Parse the optional `/duplicates` threshold argument into a `0.0`–`1.0`
/// fraction. A bare number is read as a percentage when greater than `1`
/// (`80` → `0.80`) and as a fraction otherwise (`0.8` → `0.80`); a trailing `%`
/// is allowed (`80%`). The result is clamped to `0.0`–`1.0`. An empty or
/// unparseable argument yields `None`, leaving the default threshold in place.
pub fn parse_similarity_threshold(arg: &str) -> Option<f64> {
    let trimmed = arg.trim().trim_end_matches('%').trim();
    if trimmed.is_empty() {
        return None;
    }
    let value = trimmed.parse::<f64>().ok()?;
    let fraction = if value > 1.0 { value / 100.0 } else { value };
    Some(fraction.clamp(0.0, 1.0))
}

/// Whether `argument` is the one token a *bare* verb prefix is allowed to take.
///
/// The natural-language bindings include bare English verbs — `create `,
/// `open `, `delete `, `merge ` — that are also how a person opens a sentence
/// meant for the model. `create pacman.rs` names a file; "Create a Pacman like
/// game" is a request, and reading its remainder as a filename both creates
/// nonsense on disk and stops the prompt ever reaching the model. One
/// whitespace-free word is the shape of a path, a branch, or a remote; more
/// than one is prose. Quoting overrides the rule, so a path that genuinely
/// contains a space is still reachable as `open "docs/user guide.md"`.
///
/// The explicit forms (`create file `, `delete branch `, …) name their object
/// and are never held to this — there the user has already said what they mean.
pub fn is_single_argument(argument: &str) -> bool {
    let argument = argument.trim();
    !argument.is_empty()
        && (matches!(argument.chars().next(), Some('"' | '\''))
            || !argument.chars().any(char::is_whitespace))
}

pub fn parse_open_file_target<'a>(
    input: &'a str,
    prefix: &str,
    single_token_only: bool,
) -> Option<&'a str> {
    let path = strip_ascii_prefix(input, prefix)?.trim();
    if path.is_empty() || (single_token_only && !is_single_argument(path)) {
        return None;
    }
    Some(strip_matching_quotes(path))
}

/// The file path of an open command submitted in a review input window —
/// `/open_file <file>`, `open file <file>`, `open <file>`, `edit file <file>`,
/// or `edit <file>` (case-insensitive), with any wrapping quotes removed —
/// matching the open/edit forms `parse_natural_language_command` accepts in the
/// main prompt. `None` when the input is not one of those forms. This is what
/// lets `/review` (always) and `/auto_review` (once the run is done) open any
/// project file in `$EDITOR`, not just the changed files. `open file ` is tried
/// before the bare `open `, so `open file x` yields `x` rather than `file x`.
pub fn parse_open_command_target(input: &str) -> Option<&str> {
    for (prefix, bare) in [
        ("/open_file ", false),
        ("open file ", false),
        ("open ", true),
        ("edit file ", false),
        ("edit ", true),
    ] {
        if let Some(path) = parse_open_file_target(input, prefix, bare) {
            return Some(path);
        }
    }
    None
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
    if path.is_empty() || (single_token_only && !is_single_argument(path)) {
        return None;
    }
    Some(strip_matching_quotes(path))
}

pub fn parse_pull_pr_number(input: &str) -> Option<u64> {
    for prefix in ["pull request ", "pull pr ", "pull #", "pull "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix)
            && let Ok(num) = rest.trim().parse::<u64>()
        {
            return Some(num);
        }
    }
    None
}

/// Parses the arguments of a comment command into either a single numbered
/// target ([`LocalCommand::Comment`]) or, when the first word is `all`
/// (optionally after `on`, as in `/comment on all <body>`), every open
/// pull/merge request ([`LocalCommand::CommentAll`]).
pub fn parse_comment_command(input: &str) -> LocalCommand<'_> {
    let input = input.trim();
    let input = strip_ascii_prefix(input, "on ").map_or(input, str::trim_start);
    let (first, rest) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
    if first.eq_ignore_ascii_case(COMMENT_ALL_KEYWORD) {
        return LocalCommand::CommentAll(parse_comment_body(rest));
    }
    LocalCommand::Comment(parse_comment_args(input))
}

pub fn parse_comment_args(input: &str) -> Option<(u64, CommentBody<'_>)> {
    let input = input.trim();
    let (number, rest) = input.split_once(char::is_whitespace)?;
    let number = number.trim_start_matches('#').parse::<u64>().ok()?;
    parse_comment_body(rest).map(|body| (number, body))
}

/// Parses a comment body: a report keyword, a quoted inline body, or a
/// `~/.orangu/comments/` template filename. `None` when it is empty.
fn parse_comment_body(rest: &str) -> Option<CommentBody<'_>> {
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    // The report keywords match the whole argument only; anything else stays
    // an inline body or a `~/.orangu/comments/` template filename.
    if rest.eq_ignore_ascii_case(COMMENT_AUTO_REVIEW_KEYWORD) {
        return Some(CommentBody::AutoReview);
    }
    if rest.eq_ignore_ascii_case(COMMENT_REVIEW_KEYWORD) {
        return Some(CommentBody::Review);
    }
    if rest.starts_with('"') || rest.starts_with('\'') {
        let body = strip_matching_quotes(rest);
        if body.is_empty() {
            return None;
        }
        Some(CommentBody::Inline(Cow::Borrowed(body)))
    } else {
        Some(CommentBody::File(Cow::Borrowed(rest)))
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

pub fn open_file_usage_message() -> &'static str {
    "Usage: /open_file <path>. Use /help to see available commands."
}

pub fn model_usage_message() -> &'static str {
    "Usage: /model <name>. Use /help to see available commands."
}

pub fn server_usage_message() -> &'static str {
    "Usage: /server <name>. Use /help to see available commands."
}

pub fn mcp_usage_message() -> &'static str {
    "Usage: /mcp [refresh|add <name> <endpoint>|modify <name> <endpoint>]"
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

/// Parse `/issue` arguments — `<field> <number> <value>` — into an
/// [`IssueAction`]. The field is one of `reviewer`, `assignee`, `label`; the
/// number is an issue or pull/merge-request number (a leading `#` is allowed);
/// and the value is the rest of the line (so a multi-word label like
/// `needs triage` is kept whole). Returns `None` on any malformed input.
pub fn parse_issue_args(input: &str) -> Option<IssueAction<'_>> {
    let input = input.trim();
    let (field_word, rest) = input.split_once(char::is_whitespace)?;
    let field = IssueField::parse(field_word)?;
    let rest = rest.trim_start();
    let (number_word, value) = rest.split_once(char::is_whitespace)?;
    let number = number_word.trim_start_matches('#').parse::<u64>().ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    Some(IssueAction {
        field,
        number,
        value: Cow::Borrowed(value),
    })
}

pub fn issue_usage_message() -> &'static str {
    "Usage: /issue <reviewer|assignee|label> <number> <value> or /issue create <title> [--body <text>] [--label <label>] [--assignee <user>]. Use /help to see available commands."
}

/// Parse `/issue create` arguments — `<title> [--body <text>] [--label
/// <label>] [--assignee <user>]` — into an [`IssueCreateArgs`]. The title is
/// every word up to the first `--flag` (quote it when it carries spaces, as in
/// `/issue create "Crash on startup" --label bug`); `--description` is an
/// alias of `--body`, and `-b`/`-d` (body), `-l` (label), `-a` (assignee) are
/// the short forms. `--label`/`--assignee` repeat and accept comma-separated
/// lists. Returns `None` when the title is missing, a flag has no value, or an
/// unknown `--flag` is given.
pub fn parse_issue_create_args(input: &str) -> Option<IssueCreateArgs<'_>> {
    let words = shell_words(input.trim()).ok()?;
    if words.is_empty() {
        return None;
    }
    let mut title_parts: Vec<String> = Vec::new();
    let mut body: Option<String> = None;
    let mut labels: Vec<Cow<'_, str>> = Vec::new();
    let mut assignees: Vec<Cow<'_, str>> = Vec::new();
    let mut index = 0;
    // The title is everything up to the first flag.
    while index < words.len() && !is_issue_create_flag(&words[index]) {
        title_parts.push(words[index].clone());
        index += 1;
    }
    // Split a `--flag=value` token into `(flag, Some(value))`, a bare `--flag`
    // into `(flag, None)`, and anything else into `(token, None)` with no flag.
    fn split_flag_value(token: &str) -> (&str, Option<&str>) {
        match token.split_once('=') {
            Some((flag, value)) if flag.starts_with('-') => (flag, Some(value)),
            _ => (token, None),
        }
    }
    while index < words.len() {
        let (flag, inline) = split_flag_value(words[index].as_str());
        let kind = issue_create_flag_kind(flag)?;
        let value: &str = match inline {
            Some(value) => value,
            None => {
                index += 1;
                if index >= words.len() {
                    return None;
                }
                // A flag where the value should be means the value is missing.
                if is_issue_create_flag(words[index].as_str()) {
                    return None;
                }
                words[index].as_str()
            }
        };
        let value = value.trim();
        if value.is_empty() {
            return None;
        }
        match kind {
            IssueCreateFlag::Body => body = Some(value.to_string()),
            IssueCreateFlag::Label => {
                for label in value.split(',') {
                    let label = label.trim();
                    if label.is_empty() {
                        return None;
                    }
                    labels.push(Cow::Owned(label.to_string()));
                }
            }
            IssueCreateFlag::Assignee => {
                for user in value.split(',') {
                    let user = user.trim();
                    if user.is_empty() {
                        return None;
                    }
                    assignees.push(Cow::Owned(user.to_string()));
                }
            }
        }
        index += 1;
    }
    let title = title_parts.join(" ").trim().to_string();
    if title.is_empty() {
        return None;
    }
    Some(IssueCreateArgs {
        title: Cow::Owned(title),
        body: Cow::Owned(body.unwrap_or_default()),
        labels,
        assignees,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IssueCreateFlag {
    Body,
    Label,
    Assignee,
}

/// Whether `token` starts an `/issue create` flag (long, short, or
/// `--flag=value` form). Unknown but flag-shaped tokens (`--bogus`, `-z`)
/// count too, so a typo surfaces as a usage error instead of silently becoming
/// title text; a bare `-` and negative numbers (`-5`) stay ordinary title
/// words.
fn is_issue_create_flag(token: &str) -> bool {
    let flag = match token.split_once('=') {
        Some((flag, _)) if flag.starts_with('-') => flag,
        _ => token,
    };
    if issue_create_flag_kind(flag).is_some() {
        return true;
    }
    looks_like_flag(flag)
}

/// Whether `word` is shaped like a CLI flag: `-` followed by a non-digit.
fn looks_like_flag(word: &str) -> bool {
    let mut chars = word.chars();
    match (chars.next(), chars.next()) {
        (Some('-'), Some(second)) => !second.is_ascii_digit(),
        _ => false,
    }
}

/// Map an `/issue create` flag word to what it sets. `--description`/`-d` are
/// the GitLab-spelled alias of `--body`/`-b`; anything else is `None`.
fn issue_create_flag_kind(flag: &str) -> Option<IssueCreateFlag> {
    match flag {
        "--body" | "--description" | "-b" | "-d" => Some(IssueCreateFlag::Body),
        "--label" | "-l" => Some(IssueCreateFlag::Label),
        "--assignee" | "-a" => Some(IssueCreateFlag::Assignee),
        _ => None,
    }
}

pub fn issue_create_usage_message() -> &'static str {
    "Usage: /issue create <title> [--body <text>] [--label <label>] [--assignee <user>]. Use /help to see available commands."
}

pub fn parse_get_comments_args(input: &str) -> Option<GetCommentsTarget> {
    let input = input.trim();
    if let Some(rest) = input.strip_prefix("-i ") {
        return rest
            .trim()
            .trim_start_matches('#')
            .parse::<u64>()
            .ok()
            .map(GetCommentsTarget::Issue);
    }
    if let Some(rest) = input.strip_prefix("-p ") {
        return rest
            .trim()
            .trim_start_matches('#')
            .parse::<u64>()
            .ok()
            .map(GetCommentsTarget::PullRequest);
    }
    None
}

pub fn get_comments_usage_message() -> &'static str {
    "Usage: /get_comments -i <number> or /get_comments -p <number>. Use /help to see available commands."
}

pub fn comment_usage_message() -> &'static str {
    "Usage: /comment <number> \"<comment>\", /comment <number> <file>, /comment <number> with [auto] review, or /comment all <file> for every open pull request. Use /help to see available commands."
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

pub fn parse_prune_args(input: &str) -> Option<PruneTarget> {
    if input.eq_ignore_ascii_case("all") {
        return Some(PruneTarget::All);
    }
    if let Some(rest) = input
        .strip_prefix("--workspace ")
        .or_else(|| input.strip_prefix("-w "))
    {
        let path = rest.trim();
        if !path.is_empty() {
            return Some(PruneTarget::Workspace(path.to_string()));
        }
        return None;
    }
    if let Some(rest) = input
        .strip_prefix("--older-than ")
        .or_else(|| input.strip_prefix("-o "))
    {
        return rest.trim().parse::<u64>().ok().map(PruneTarget::OlderThan);
    }
    if !input.is_empty() {
        return Some(PruneTarget::Uuid(input.to_string()));
    }
    None
}

pub fn prune_usage_message() -> &'static str {
    "Usage: /prune <uuid> | /prune --workspace <path> | /prune --older-than <days>. Use /help to see available commands."
}

pub fn create_file_usage_message() -> &'static str {
    "Usage: /create_file <path> [with <mode>] [containing <text>]. Use /help to see available commands."
}

pub fn delete_file_usage_message() -> &'static str {
    "Usage: /remove_file <path>. Use /help to see available commands."
}

pub fn move_file_usage_message() -> &'static str {
    "Usage: /move_file <source> <destination>. Use /help to see available commands."
}

pub fn add_repository_usage_message() -> &'static str {
    "Usage: /add_repository <user> [<branch>]. Use /help to see available commands."
}

pub fn cherry_pick_usage_message() -> &'static str {
    "Usage: /cherry_pick <commit>. Use /help to see available commands."
}

pub fn revert_usage_message() -> &'static str {
    "Usage: /revert <commit>|abort. Use /help to see available commands."
}

pub fn commit_usage_message() -> &'static str {
    "Usage: /commit <message>. Use /help to see available commands."
}

pub fn amend_usage_message() -> &'static str {
    "Usage: /amend <message>. Use /help to see available commands."
}

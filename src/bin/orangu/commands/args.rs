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
        _ => None,
    }
}

pub fn parse_open_file_target<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let path = strip_ascii_prefix(input, prefix)?.trim();
    if path.is_empty() {
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
    for prefix in ["/open_file ", "open file ", "open ", "edit file ", "edit "] {
        if let Some(path) = parse_open_file_target(input, prefix) {
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
    for prefix in ["pull request ", "pull pr ", "pull #", "pull "] {
        if let Some(rest) = strip_ascii_prefix(input, prefix)
            && let Ok(num) = rest.trim().parse::<u64>()
        {
            return Some(num);
        }
    }
    None
}

pub fn parse_comment_args(input: &str) -> Option<(u64, CommentBody<'_>)> {
    let input = input.trim();
    let (number, rest) = input.split_once(char::is_whitespace)?;
    let number = number.trim_start_matches('#').parse::<u64>().ok()?;
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    // The report keywords match the whole argument only; anything else stays
    // an inline body or a `~/.orangu/comments/` template filename.
    if rest.eq_ignore_ascii_case(COMMENT_AUTO_REVIEW_KEYWORD) {
        return Some((number, CommentBody::AutoReview));
    }
    if rest.eq_ignore_ascii_case(COMMENT_REVIEW_KEYWORD) {
        return Some((number, CommentBody::Review));
    }
    if rest.starts_with('"') || rest.starts_with('\'') {
        let body = strip_matching_quotes(rest);
        if body.is_empty() {
            return None;
        }
        Some((number, CommentBody::Inline(Cow::Borrowed(body))))
    } else {
        Some((number, CommentBody::File(Cow::Borrowed(rest))))
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
    "Usage: /issue <reviewer|assignee|label> <number> <value>. Use /help to see available commands."
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
    "Usage: /comment <number> \"<comment>\", /comment <number> <file>, or /comment <number> with [auto] review. Use /help to see available commands."
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

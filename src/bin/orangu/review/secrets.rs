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

//! Deterministic detection and redaction of credentials, and of text that
//! tries to steer a model reading it.
//!
//! Everything here is pattern matching on local text: no request leaves the
//! machine, so the checks work the same in an air-gapped environment.

use regex::Regex;
use std::sync::OnceLock;

/// How serious a detected problem is. `High` problems reject a reviewed file
/// on their own; `Medium` ones are reported for the user to confirm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Medium,
    High,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Medium => "medium",
            Severity::High => "high",
        }
    }
}

/// One credential found in a line of text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretMatch {
    /// What kind of credential it is, e.g. `AWS access key ID`.
    pub kind: &'static str,
    /// The matched value, redacted.
    pub redacted: String,
    pub severity: Severity,
}

/// A provider-specific credential pattern: its name, the regex, and the
/// capture group holding the secret value (0 for the whole match).
struct Pattern {
    kind: &'static str,
    regex: Regex,
    group: usize,
    severity: Severity,
}

fn patterns() -> &'static [Pattern] {
    static PATTERNS: OnceLock<Vec<Pattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let pattern = |kind, source: &str, group, severity| Pattern {
            kind,
            regex: Regex::new(source).expect("secret pattern"),
            group,
            severity,
        };
        vec![
            pattern(
                "AWS access key ID",
                r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b",
                0,
                Severity::High,
            ),
            pattern(
                "GitHub token",
                r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b",
                0,
                Severity::High,
            ),
            pattern(
                "GitLab token",
                r"\bglpat-[A-Za-z0-9_\-]{20,}",
                0,
                Severity::High,
            ),
            pattern(
                "Slack token",
                r"\bxox[abposr]-[A-Za-z0-9\-]{10,}",
                0,
                Severity::High,
            ),
            pattern(
                "Slack webhook URL",
                r"https://hooks\.slack\.com/services/[A-Za-z0-9/_\-]{20,}",
                0,
                Severity::High,
            ),
            pattern(
                "Google API key",
                r"\bAIza[0-9A-Za-z_\-]{35}",
                0,
                Severity::High,
            ),
            pattern(
                "Anthropic API key",
                r"\bsk-ant-[A-Za-z0-9_\-]{20,}",
                0,
                Severity::High,
            ),
            pattern(
                "OpenAI API key",
                r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}",
                0,
                Severity::High,
            ),
            pattern(
                "Stripe secret key",
                r"\b[rs]k_live_[0-9A-Za-z]{20,}",
                0,
                Severity::High,
            ),
            pattern(
                "Hugging Face token",
                r"\bhf_[A-Za-z0-9]{34,}\b",
                0,
                Severity::High,
            ),
            pattern(
                "JSON Web Token",
                r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
                0,
                Severity::High,
            ),
            pattern(
                "password in a connection string",
                r#"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@'"`]+:([^\s:/@'"`]+)@"#,
                1,
                Severity::High,
            ),
            pattern(
                "credential assigned to a variable",
                r#"(?i)[a-z0-9_.\-]*(?:secret|token|passwd|password|api[_\-]?key|access[_\-]?key|auth[_\-]?key)[a-z0-9_.\-]*["']?\s*[:=]\s*["']([A-Za-z0-9_\-+=/]{20,})["']"#,
                1,
                Severity::Medium,
            ),
        ]
    })
}

/// The opening line of a PEM private key (`-----BEGIN RSA PRIVATE KEY-----`,
/// `-----BEGIN OPENSSH PRIVATE KEY-----`, `-----BEGIN PGP PRIVATE KEY BLOCK-----`).
fn private_key_begin() -> &'static Regex {
    static BEGIN: OnceLock<Regex> = OnceLock::new();
    BEGIN.get_or_init(|| {
        Regex::new(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----").expect("pem")
    })
}

fn private_key_end() -> &'static Regex {
    static END: OnceLock<Regex> = OnceLock::new();
    END.get_or_init(|| {
        Regex::new(r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----").expect("pem")
    })
}

/// Shannon entropy of `value` in bits per character.
fn entropy(value: &str) -> f64 {
    let mut counts = [0usize; 256];
    for byte in value.bytes() {
        counts[byte as usize] += 1;
    }
    let len = value.len() as f64;
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Whether a matched value is a placeholder, a template reference, or too
/// regular to be a real credential.
fn is_placeholder(kind: &str, value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if value.starts_with(['$', '{', '<', '%', '*'])
        || ["password", "passwd", "pass", "pwd", "secret", "changeme"].contains(&lower.as_str())
        || ["example", "dummy", "placeholder", "xxxx", "redacted"]
            .iter()
            .any(|marker| lower.contains(marker))
    {
        return true;
    }
    if kind == "credential assigned to a variable" {
        let has_digit = value.bytes().any(|b| b.is_ascii_digit());
        let has_alpha = value.bytes().any(|b| b.is_ascii_alphabetic());
        return !(has_digit && has_alpha && entropy(value) >= 4.0);
    }
    false
}

/// `value` with all but its first four characters hidden: `AKIA…[REDACTED]`.
fn redacted(value: &str) -> String {
    let keep: String = value.chars().take(4).collect();
    format!("{keep}…[REDACTED]")
}

/// The credentials in one line of text, in order of appearance. A private-key
/// header counts as one; the key body lines that follow it are not matched by
/// this per-line check (see `redact_secrets`, which hides them).
pub fn scan_line(line: &str) -> Vec<SecretMatch> {
    let mut found: Vec<(usize, SecretMatch)> = Vec::new();
    // Byte ranges already reported, so a value two patterns both match (an
    // API key assigned to `api_key = "..."`) is reported once, by the more
    // specific pattern listed first.
    let mut claimed: Vec<std::ops::Range<usize>> = Vec::new();
    if let Some(m) = private_key_begin().find(line) {
        claimed.push(m.start()..line.len());
        found.push((
            m.start(),
            SecretMatch {
                kind: "private key",
                redacted: "-----BEGIN PRIVATE KEY----- [REDACTED]".to_string(),
                severity: Severity::High,
            },
        ));
    }
    for pattern in patterns() {
        for captures in pattern.regex.captures_iter(line) {
            let Some(value) = captures.get(pattern.group) else {
                continue;
            };
            if is_placeholder(pattern.kind, value.as_str()) {
                continue;
            }
            if claimed
                .iter()
                .any(|range| range.start < value.end() && value.start() < range.end)
            {
                continue;
            }
            claimed.push(value.range());
            found.push((
                value.start(),
                SecretMatch {
                    kind: pattern.kind,
                    redacted: redacted(value.as_str()),
                    severity: pattern.severity,
                },
            ));
        }
    }
    found.sort_by_key(|(start, _)| *start);
    found.into_iter().map(|(_, m)| m).collect()
}

/// `line` with every credential `scan_line` finds replaced by its redacted
/// form.
fn redact_line(line: &str) -> String {
    let mut out = line.to_string();
    if let Some(m) = private_key_begin().find(&out) {
        let end = m.end();
        out.replace_range(end.., " [REDACTED]");
    }
    for pattern in patterns() {
        let replaced = pattern
            .regex
            .replace_all(&out, |captures: &regex::Captures<'_>| {
                let whole = captures.get(0).expect("match").as_str();
                let Some(value) = captures.get(pattern.group) else {
                    return whole.to_string();
                };
                if is_placeholder(pattern.kind, value.as_str()) {
                    return whole.to_string();
                }
                let start = value.start() - captures.get(0).expect("match").start();
                let mut text = whole.to_string();
                text.replace_range(start..start + value.len(), &redacted(value.as_str()));
                text
            })
            .into_owned();
        out = replaced;
    }
    out
}

/// `text` with every detected credential redacted, line for line: the
/// number of lines never changes, so line references into the text stay
/// valid. A PEM private key is hidden from its header through its `END`
/// line; a diff's `+`/`-`/` ` prefix on those lines is kept.
pub fn redact_secrets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_key = false;
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
        }
        if in_key {
            if private_key_end().is_match(line) {
                in_key = false;
                out.push_str(line);
            } else {
                // Keep a diff marker so the line still parses as a diff line.
                if let Some(marker) = line.chars().next().filter(|c| matches!(c, '+' | '-' | ' ')) {
                    out.push(marker);
                }
                out.push_str("[REDACTED]");
            }
            continue;
        }
        if private_key_begin().is_match(line) && !private_key_end().is_match(line) {
            in_key = true;
        }
        out.push_str(&redact_line(line));
    }
    out
}

/// One character that can hide or reorder what a reader sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenCharacter {
    pub codepoint: u32,
    pub description: &'static str,
    pub severity: Severity,
}

/// Classify `ch` as a bidirectional control (Trojan Source,
/// CVE-2021-42574), a Unicode tag character (text a model reads but an
/// editor does not show), or a zero-width character.
fn hidden_character(ch: char) -> Option<(&'static str, Severity)> {
    match ch as u32 {
        0x202A..=0x202E | 0x2066..=0x2069 => {
            Some(("bidirectional control character", Severity::High))
        }
        0xE0000..=0xE007F => Some(("invisible Unicode tag character", Severity::High)),
        0x200B | 0x2060 | 0x180E | 0xFEFF => Some(("zero-width character", Severity::Medium)),
        _ => None,
    }
}

/// The hidden characters in `line`, one entry per distinct kind. A byte-order
/// mark at the very start of a file is legitimate and is skipped when
/// `first_line` is true.
pub fn scan_hidden_characters(line: &str, first_line: bool) -> Vec<HiddenCharacter> {
    let mut found: Vec<HiddenCharacter> = Vec::new();
    for (offset, ch) in line.char_indices() {
        if first_line && offset == 0 && ch == '\u{FEFF}' {
            continue;
        }
        if let Some((description, severity)) = hidden_character(ch)
            && !found.iter().any(|h| h.description == description)
        {
            found.push(HiddenCharacter {
                codepoint: ch as u32,
                description,
                severity,
            });
        }
    }
    found
}

/// Word sequences that try to override a model's instructions or dictate a
/// review verdict. Matched on whole words, case-insensitively.
const INJECTION_PHRASES: [&[&str]; 8] = [
    &["ignore", "previous", "instructions"],
    &["ignore", "all", "previous"],
    &["ignore", "prior", "instructions"],
    &["ignore", "the", "above", "instructions"],
    &["disregard", "previous", "instructions"],
    &["disregard", "all", "previous"],
    &["forget", "your", "instructions"],
    &["verdict", "approve"],
];

/// The first instruction-override phrase in `line`, joined with spaces, if
/// any.
pub fn scan_injection_phrase(line: &str) -> Option<String> {
    let words: Vec<String> = line
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    INJECTION_PHRASES.iter().find_map(|phrase| {
        words
            .windows(phrase.len())
            .any(|window| window.iter().zip(phrase.iter()).all(|(w, p)| w == p))
            .then(|| phrase.join(" "))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixtures are assembled at run time so this file holds no literal that
    // a secret scanner would report.
    fn aws_key() -> String {
        format!("AKIA{}", "Z7QK4M2PLX9TB3WN")
    }

    #[test]
    fn a_provider_key_is_found_and_redacted() {
        let line = format!("let key = \"{}\";", aws_key());
        let found = scan_line(&line);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, "AWS access key ID");
        assert_eq!(found[0].redacted, "AKIA…[REDACTED]");
        assert_eq!(found[0].severity, Severity::High);
        let redacted = redact_secrets(&line);
        assert_eq!(redacted, "let key = \"AKIA…[REDACTED]\";");
    }

    #[test]
    fn a_connection_string_password_is_redacted_but_not_the_user() {
        let line = format!("url = \"postgres://app:{}@db:5432/x\"", "s3cr3tPw");
        let found = scan_line(&line);
        assert_eq!(found[0].kind, "password in a connection string");
        assert_eq!(
            redact_secrets(&line),
            "url = \"postgres://app:s3cr…[REDACTED]@db:5432/x\""
        );
        // A template reference is not a credential.
        assert!(scan_line("postgres://app:${PASSWORD}@db/x").is_empty());
    }

    #[test]
    fn a_generic_assignment_needs_entropy() {
        let strong = format!("api_key = \"{}\"", "q8Zr2LmX0vT7pN4kW1sY9bC3");
        assert_eq!(scan_line(&strong)[0].severity, Severity::Medium);
        assert!(scan_line("api_key = \"aaaaaaaaaaaaaaaaaaaaaaaa\"").is_empty());
        assert!(scan_line("token_path = \"models/tokenizer\"").is_empty());
    }

    #[test]
    fn a_private_key_body_is_hidden_line_for_line() {
        let begin = format!("-----BEGIN RSA {}-----", "PRIVATE KEY");
        let end = format!("-----END RSA {}-----", "PRIVATE KEY");
        let text = format!("+{begin}\n+MIIEowIBAAKCAQEA\n+abcdef\n+{end}\n+after");
        let redacted = redact_secrets(&text);
        assert_eq!(redacted.lines().count(), text.lines().count());
        assert!(!redacted.contains("MIIEow"));
        assert!(redacted.contains("+[REDACTED]"));
        assert!(redacted.ends_with("+after"));
        assert_eq!(scan_line(&begin)[0].kind, "private key");
    }

    #[test]
    fn hidden_characters_are_classified() {
        let trojan = format!("if is_admin {{ /*{}*/ }}", '\u{202E}');
        let found = scan_hidden_characters(&trojan, false);
        assert_eq!(found[0].severity, Severity::High);
        assert_eq!(found[0].codepoint, 0x202E);
        let bom = format!("{}fn main() {{}}", '\u{FEFF}');
        assert!(scan_hidden_characters(&bom, true).is_empty());
        assert_eq!(scan_hidden_characters(&bom, false).len(), 1);
        // A zero-width joiner is part of emoji sequences and is not flagged.
        assert!(scan_hidden_characters("\u{1F469}\u{200D}\u{1F4BB}", false).is_empty());
    }

    #[test]
    fn injection_phrases_match_whole_words() {
        let line = ["// Reviewer:", "Ignore", "previous", "instructions."].join(" ");
        assert_eq!(
            scan_injection_phrase(&line).as_deref(),
            Some("ignore previous instructions")
        );
        let verdict = format!("# {}: {}", "VERDICT", "APPROVE");
        assert_eq!(
            scan_injection_phrase(&verdict).as_deref(),
            Some("verdict approve")
        );
        assert!(scan_injection_phrase("let previous = ignore_instructions();").is_none());
    }
}

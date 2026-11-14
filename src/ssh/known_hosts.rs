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

//! `known_hosts` lines that name hosts with wildcards, such as
//! `*.example.com` or `[10.0.0.?]:2222`. russh matches only exact and hashed
//! names, so these are read here, the way `ssh` reads them: `*` is any run of
//! characters, `?` one character, and a host matching a `!pattern` on the line
//! is not named by it.

use russh::keys::{self, PublicKey};
use std::path::Path;

/// How the keys recorded for a host compare with the key it offers.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Recorded {
    /// A line records the offered key.
    Matches,
    /// No line records it, but the line at this number records another key of
    /// the same type.
    Changed(usize),
    /// No line records a key of this type.
    Absent,
}

/// The keys the wildcard lines of `known_hosts` record for `host` on `port`,
/// with their line numbers. An unreadable file has none.
pub(super) fn wildcard_host_keys(host: &str, port: u16, path: &Path) -> Vec<(usize, PublicKey)> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let name = if port == 22 {
        host.to_ascii_lowercase()
    } else {
        format!("[{}]:{port}", host.to_ascii_lowercase())
    };
    contents
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let mut fields = line.split_whitespace();
            let patterns = fields.next()?;
            // Comments, and `@revoked` or `@cert-authority` lines.
            if patterns.starts_with('#') || patterns.starts_with('@') {
                return None;
            }
            if !patterns.contains(['*', '?']) || !names(patterns, &name) {
                return None;
            }
            let _algorithm = fields.next()?;
            let key = keys::parse_public_key_base64(fields.next()?).ok()?;
            Some((index + 1, key))
        })
        .collect()
}

/// Compares `key` with the `recorded` keys, as `ssh` does: one matching line
/// is enough, and only a key of the same type can have changed.
pub(super) fn compare(key: &PublicKey, recorded: &[(usize, PublicKey)]) -> Recorded {
    if recorded.iter().any(|(_, known)| known == key) {
        return Recorded::Matches;
    }
    recorded
        .iter()
        .find(|(_, known)| known.algorithm() == key.algorithm())
        .map_or(Recorded::Absent, |(line, _)| Recorded::Changed(*line))
}

/// Whether the comma-separated `patterns` of a line name `name`.
fn names(patterns: &str, name: &str) -> bool {
    let mut named = false;
    for pattern in patterns.split(',') {
        let (negated, pattern) = match pattern.strip_prefix('!') {
            Some(pattern) => (true, pattern),
            None => (false, pattern),
        };
        if matches(pattern.to_ascii_lowercase().as_bytes(), name.as_bytes()) {
            if negated {
                return false;
            }
            named = true;
        }
    }
    named
}

/// Whether `text` matches `pattern`, where `*` is any run of characters and `?`
/// exactly one.
fn matches(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0, 0);
    // The last `*` seen, and where in `text` it is tried from next.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        match pattern.get(p) {
            Some(b'*') => {
                star = Some((p, t));
                p += 1;
            }
            Some(&c) if c == b'?' || c == text[t] => {
                p += 1;
                t += 1;
            }
            _ => match star {
                Some((star_p, star_t)) => {
                    p = star_p + 1;
                    t = star_t + 1;
                    star = Some((star_p, star_t + 1));
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|&c| c == b'*')
}

#[cfg(test)]
mod tests {
    use super::*;

    const ED25519_A: &str = "AAAAC3NzaC1lZDI1NTE5AAAAIJdD7y3aLq454yWBdwLWbieU1ebz9/cu7/QEXn9OIeZJ";
    const ED25519_B: &str = "AAAAC3NzaC1lZDI1NTE5AAAAIA6rWI3G1sz07DnfFlrouTcysQlj2P+jpNSOEWD9OJ3X";

    fn key(encoded: &str) -> PublicKey {
        keys::parse_public_key_base64(encoded).unwrap()
    }

    fn known_hosts(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("known_hosts");
        std::fs::write(&path, contents).unwrap();
        (directory, path)
    }

    #[test]
    fn star_and_question_mark_match_like_ssh() {
        assert!(matches(b"*.example.com", b"db.example.com"));
        assert!(matches(b"*.example.com", b"a.b.example.com"));
        assert!(!matches(b"*.example.com", b"example.com"));
        assert!(matches(b"10.0.0.?", b"10.0.0.7"));
        assert!(!matches(b"10.0.0.?", b"10.0.0.17"));
        assert!(matches(b"*", b"anything"));
        assert!(matches(b"db*", b"db"));
        assert!(matches(b"*db*1", b"proddb01"));
        assert!(!matches(b"db", b"db1"));
    }

    #[test]
    fn a_negated_pattern_takes_the_host_off_the_line() {
        assert!(names("*.example.com,!secret.example.com", "db.example.com"));
        assert!(!names(
            "*.example.com,!secret.example.com",
            "secret.example.com"
        ));
        assert!(!names("!secret.example.com", "db.example.com"));
        assert!(names("*.EXAMPLE.com", "db.example.com"));
    }

    #[test]
    fn wildcard_lines_name_hosts_with_their_port() {
        let (_directory, path) = known_hosts(&format!(
            "# a comment\n\
             db.example.com ssh-ed25519 {ED25519_B}\n\
             *.example.com,!secret.example.com ssh-ed25519 {ED25519_A}\n\
             [*.internal]:2222 ssh-ed25519 {ED25519_B}\n\
             @revoked *.example.com ssh-ed25519 {ED25519_B}\n"
        ));
        // The exact line is left to russh.
        assert_eq!(
            wildcard_host_keys("db.example.com", 22, &path),
            vec![(3, key(ED25519_A))]
        );
        assert_eq!(
            wildcard_host_keys("DB.Example.com", 22, &path),
            vec![(3, key(ED25519_A))]
        );
        assert!(wildcard_host_keys("secret.example.com", 22, &path).is_empty());
        assert!(wildcard_host_keys("db.example.com", 2222, &path).is_empty());
        assert_eq!(
            wildcard_host_keys("build.internal", 2222, &path),
            vec![(4, key(ED25519_B))]
        );
        assert!(wildcard_host_keys("build.internal", 22, &path).is_empty());
    }

    #[test]
    fn a_missing_file_has_no_keys() {
        let directory = tempfile::tempdir().unwrap();
        assert!(wildcard_host_keys("db", 22, &directory.path().join("missing")).is_empty());
    }

    #[test]
    fn one_matching_line_is_enough_and_only_the_same_type_can_change() {
        let offered = key(ED25519_A);
        assert_eq!(compare(&offered, &[]), Recorded::Absent);
        assert_eq!(
            compare(&offered, &[(2, key(ED25519_B))]),
            Recorded::Changed(2)
        );
        assert_eq!(
            compare(&offered, &[(2, key(ED25519_B)), (5, key(ED25519_A))]),
            Recorded::Matches
        );
    }
}

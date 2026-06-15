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

use super::git_commit_hashes;
use crate::git::discover_git_root;

/// Tab completion for `/bisect <subcommand> <commit>`: after `/bisect start`,
/// `/bisect good`, `/bisect bad`, or `/bisect skip` with a trailing space,
/// offer commit hashes from the repository.
pub fn bisect_completion_candidates(
    prefix: &str,
    workspace: &Path,
) -> Option<(usize, Vec<String>)> {
    let commit_subcommands = [
        "/bisect start ",
        "/bisect good ",
        "/bisect bad ",
        "/bisect skip ",
    ];
    for cmd in &commit_subcommands {
        if let Some(rest) = prefix.strip_prefix(cmd) {
            // Match candidates against the typed token, trimming any extra
            // spaces after the subcommand. The replacement still starts at the
            // end of the `<subcommand> ` prefix, so those spaces are collapsed
            // when a candidate is accepted — the same convention as
            // `cherry_pick_completion_candidates`.
            let token = rest.trim_start();
            let candidates = discover_git_root(workspace)
                .map(|root| git_commit_hashes(&root, token))
                .unwrap_or_default();
            return Some((cmd.len(), candidates));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::bisect_completion_candidates;
    use tempfile::tempdir;

    #[test]
    fn returns_none_for_non_commit_prefixes() {
        let dir = tempdir().expect("tempdir");
        // Subcommands that take no commit argument, or a different command,
        // produce no candidates.
        assert!(bisect_completion_candidates("/bisect ", dir.path()).is_none());
        assert!(bisect_completion_candidates("/bisect reset", dir.path()).is_none());
        assert!(bisect_completion_candidates("/branch good ", dir.path()).is_none());
    }

    #[test]
    fn offsets_to_the_end_of_the_subcommand_prefix() {
        let dir = tempdir().expect("tempdir");
        for cmd in [
            "/bisect start ",
            "/bisect good ",
            "/bisect bad ",
            "/bisect skip ",
        ] {
            let (start, _candidates) =
                bisect_completion_candidates(cmd, dir.path()).expect("a candidates tuple");
            assert_eq!(start, cmd.len(), "offset should point past '{cmd}'");
        }
        // Extra spaces and a partial token still anchor at the subcommand end.
        let (start, _candidates) =
            bisect_completion_candidates("/bisect good   ab", dir.path()).expect("some");
        assert_eq!(start, "/bisect good ".len());
    }
}

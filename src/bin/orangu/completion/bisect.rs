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
            let token = rest.trim_start();
            let candidates = discover_git_root(workspace)
                .map(|root| git_commit_hashes(&root, token))
                .unwrap_or_default();
            return Some((cmd.len(), candidates));
        }
    }
    None
}

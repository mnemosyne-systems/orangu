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

//! The shared, global home for every per-workspace cache orangu keeps.
//!
//! Each workspace gets one directory, keyed by a hash of its canonical path, so
//! caches never clutter the workspace tree (and can never accidentally get
//! committed):
//!
//! ```text
//! ~/.orangu/workspace/<sha256(canonical path)>/<subsystem>/
//! ```
//!
//! The knowledge graph cache lives at `.../graph/` and the semantic search
//! index at `.../embeddings/`. Both are workspace-scoped, not session-scoped —
//! shared across every session and tab open on that workspace — which is
//! exactly what this shared root reflects.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The cache directory for `subsystem` (e.g. `"graph"`, `"embeddings"`) scoped to
/// `workspace`. Falls back to `<workspace>/.orangu/<subsystem>` only when no home
/// directory resolves.
pub fn workspace_cache_dir(workspace: &Path, subsystem: &str) -> PathBuf {
    let canonical = std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let key = sha256(&canonical.to_string_lossy());
    match home::home_dir() {
        Some(home) => home
            .join(".orangu")
            .join("workspace")
            .join(key)
            .join(subsystem),
        None => workspace.join(".orangu").join(subsystem),
    }
}

fn sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_cache_dir_is_stable_and_scoped_by_subsystem() {
        let dir = tempfile::tempdir().expect("temp dir");
        let graph = workspace_cache_dir(dir.path(), "graph");
        let embeddings = workspace_cache_dir(dir.path(), "embeddings");

        // Same workspace, different subsystem → different leaf directory, same
        // parent (the workspace's hashed directory).
        assert_ne!(graph, embeddings);
        assert_eq!(graph.parent(), embeddings.parent());
        assert_eq!(graph.file_name().unwrap(), "graph");
        assert_eq!(embeddings.file_name().unwrap(), "embeddings");

        // Deterministic: calling again for the same workspace yields the same path.
        assert_eq!(graph, workspace_cache_dir(dir.path(), "graph"));
    }

    #[test]
    fn workspace_cache_dir_differs_for_different_workspaces() {
        let a = tempfile::tempdir().expect("temp dir");
        let b = tempfile::tempdir().expect("temp dir");
        assert_ne!(
            workspace_cache_dir(a.path(), "graph"),
            workspace_cache_dir(b.path(), "graph"),
        );
    }
}

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

//! Shared workspace source scan.
//!
//! The embedding index and the lexical sidecar need the same file set and the
//! same per-file SHA-256 map. This module computes both once per search so the
//! workspace is not walked and hashed twice.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::graph::extract::SupportedLanguage;

/// One readable source file, with its workspace-relative path and content hash.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub rel: String,
    pub hash: String,
}

/// Walk the workspace and hash every supported source file once.
///
/// Files that cannot be read as UTF-8 are skipped, matching the previous
/// per-index behaviour. Discovery honours `.gitignore` and is restricted to the
/// languages the Tree-sitter extractor understands.
pub fn scan_workspace(workspace: &Path) -> Vec<SourceFile> {
    collect_source_files(workspace)
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(workspace)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let content = std::fs::read_to_string(&path).ok()?;
            Some(SourceFile {
                path,
                rel,
                hash: sha256(&content),
            })
        })
        .collect()
}

/// Compute the sha256 hex digest of `content`.
pub fn sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Collect the workspace's source files, honouring `.gitignore`, restricted to
/// languages the Tree-sitter extractor understands.
fn collect_source_files(workspace: &Path) -> Vec<PathBuf> {
    ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build()
        .filter_map(|entry| {
            let path = entry.ok()?.into_path();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension().and_then(|e| e.to_str())?;
            SupportedLanguage::from_extension(ext).map(|_| path)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_and_content_sensitive() {
        assert_eq!(sha256("abc"), sha256("abc"));
        assert_ne!(sha256("abc"), sha256("abcd"));
    }

    #[test]
    fn scan_hashes_supported_files_with_relative_paths() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspace.path().join("src")).unwrap();
        std::fs::write(workspace.path().join("src/lib.rs"), "fn one() {}\n").unwrap();
        std::fs::write(workspace.path().join("notes.txt"), "not source\n").unwrap();

        let scan = scan_workspace(workspace.path());
        assert_eq!(scan.len(), 1);
        assert_eq!(scan[0].rel, "src/lib.rs");
        assert_eq!(scan[0].hash, sha256("fn one() {}\n"));
    }

    #[test]
    fn scan_honours_gitignore() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::create_dir(workspace.path().join(".git")).unwrap();
        std::fs::write(workspace.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(workspace.path().join("kept.rs"), "fn kept() {}\n").unwrap();
        std::fs::write(workspace.path().join("ignored.rs"), "fn ignored() {}\n").unwrap();

        let scan = scan_workspace(workspace.path());
        let files: Vec<_> = scan.iter().map(|file| file.rel.as_str()).collect();
        assert!(files.contains(&"kept.rs"));
        assert!(!files.contains(&"ignored.rs"));
    }
}

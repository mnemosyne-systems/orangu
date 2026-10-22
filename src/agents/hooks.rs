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

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::graph::{
    cache::GraphCache,
    extract::{GraphExtractor, SupportedLanguage},
    store::{GodNodeEntry, GraphStats, GraphStore},
};

// ── Project detection ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectType {
    Rust,
    NodeJs,
    Go,
    Python,
    Cpp,
    C,
    CSharp,
    Ruby,
    Scala,
    Haskell,
    Julia,
    Lua,
    R,
    Zig,
    Swift,
    Dart,
    Erlang,
    Php,
    Ocaml,
    Unknown,
}

impl std::fmt::Display for ProjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectType::Rust => write!(f, "rust"),
            ProjectType::NodeJs => write!(f, "nodejs"),
            ProjectType::Go => write!(f, "go"),
            ProjectType::Python => write!(f, "python"),
            ProjectType::Cpp => write!(f, "c++"),
            ProjectType::C => write!(f, "c"),
            ProjectType::CSharp => write!(f, "c#"),
            ProjectType::Ruby => write!(f, "ruby"),
            ProjectType::Scala => write!(f, "scala"),
            ProjectType::Haskell => write!(f, "haskell"),
            ProjectType::Julia => write!(f, "julia"),
            ProjectType::Lua => write!(f, "lua"),
            ProjectType::R => write!(f, "r"),
            ProjectType::Zig => write!(f, "zig"),
            ProjectType::Swift => write!(f, "swift"),
            ProjectType::Dart => write!(f, "dart"),
            ProjectType::Erlang => write!(f, "erlang"),
            ProjectType::Php => write!(f, "php"),
            ProjectType::Ocaml => write!(f, "ocaml"),
            ProjectType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detects the primary project type from well-known manifest files.
pub fn detect_project_type(workspace: &Path) -> ProjectType {
    if workspace.join("Cargo.toml").exists() {
        ProjectType::Rust
    } else if workspace.join("package.json").exists() {
        ProjectType::NodeJs
    } else if workspace.join("go.mod").exists() {
        ProjectType::Go
    } else if workspace.join("pyproject.toml").exists() || workspace.join("setup.py").exists() {
        ProjectType::Python
    } else if workspace.join("CMakeLists.txt").exists()
        || workspace.join("meson.build").exists()
        || workspace.join("configure.ac").exists()
    {
        // CMake / Meson / Autoconf projects are almost always C or C++.
        // Check for a .cpp/.cc/.cxx file in the root to distinguish C++ from C.
        let is_cpp = std::fs::read_dir(workspace)
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    matches!(
                        e.path().extension().and_then(|x| x.to_str()),
                        Some("cpp" | "cc" | "cxx" | "hpp" | "hxx")
                    )
                })
            })
            .unwrap_or(false);
        if is_cpp {
            ProjectType::Cpp
        } else {
            ProjectType::C
        }
    } else if std::fs::read_dir(workspace)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                matches!(
                    e.path().extension().and_then(|x| x.to_str()),
                    Some("csproj")
                )
            })
        })
        .unwrap_or(false)
        || workspace.join(".sln").exists()
    {
        ProjectType::CSharp
    } else if workspace.join("Gemfile").exists() {
        ProjectType::Ruby
    } else if workspace.join("build.sbt").exists() {
        ProjectType::Scala
    } else if workspace.join("Package.swift").exists() {
        ProjectType::Swift
    } else if workspace.join("pubspec.yaml").exists() {
        ProjectType::Dart
    } else if workspace.join("build.zig").exists() {
        ProjectType::Zig
    } else if workspace.join("stack.yaml").exists()
        || std::fs::read_dir(workspace)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .any(|e| matches!(e.path().extension().and_then(|x| x.to_str()), Some("cabal")))
            })
            .unwrap_or(false)
    {
        ProjectType::Haskell
    } else if workspace.join("dune-project").exists() {
        // OCaml — follow-up enhancement
        ProjectType::Unknown
    } else if workspace.join("Project.toml").exists() {
        ProjectType::Julia
    } else if workspace.join("DESCRIPTION").exists() {
        ProjectType::R
    } else if workspace.join("rebar.config").exists() {
        ProjectType::Erlang
    } else if workspace.join("composer.json").exists() || workspace.join("composer.lock").exists() {
        ProjectType::Php
    } else if workspace.join("dune").exists() || workspace.join("dune-project").exists() {
        ProjectType::Ocaml
    } else {
        ProjectType::Unknown
    }
}

// ── Scan result ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct WorkspaceScanResult {
    pub project_type: ProjectType,
    pub stats: GraphStats,
    pub god_nodes: Vec<GodNodeEntry>,
    pub has_cycles: bool,
    pub store: GraphStore,
    pub warnings: Vec<String>,
    /// Whether the result was loaded from cache (true) or freshly scanned (false).
    pub from_cache: bool,
    /// Hashes of the successfully indexed file versions.
    pub file_hashes: HashMap<String, String>,
}

impl WorkspaceScanResult {
    pub fn summary(&self) -> String {
        let cycle_note = if self.has_cycles {
            " ⚠ circular dependencies detected"
        } else {
            ""
        };
        let cache_note = if self.from_cache { " (cached)" } else { "" };
        let god_list = if self.god_nodes.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = self
                .god_nodes
                .iter()
                .map(|g| format!("{} ({})", g.label, g.kind))
                .collect();
            format!("\n  Key symbols: {}", names.join(", "))
        };
        let warn_note = if self.warnings.is_empty() {
            String::new()
        } else {
            format!("  {} file(s) skipped with warnings\n", self.warnings.len())
        };
        format!(
            "[Knowledge Graph] {} project — {} nodes, {} edges{}{}{}\n{}",
            self.project_type,
            self.stats.node_count,
            self.stats.edge_count,
            cache_note,
            cycle_note,
            god_list,
            warn_note,
        )
    }
}

// ── File discovery ─────────────────────────────────────────────────────────

/// Returns all source files in `workspace` that the extractor can handle.
/// Respects `.gitignore` via the `ignore` crate.
fn collect_source_files(workspace: &Path) -> Vec<PathBuf> {
    ignore::WalkBuilder::new(workspace)
        .hidden(true)
        .git_ignore(true)
        .build()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.into_path();
            if !path.is_file() {
                return None;
            }
            // Check by extension first
            if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && SupportedLanguage::from_extension(ext).is_some()
            {
                return Some(path);
            }
            // Shebang detection for extensionless scripts (e.g. bash/sh)
            if path.extension().is_none()
                && let Ok(content) = std::fs::read_to_string(&path)
            {
                let first_line = content.lines().next().unwrap_or("");
                if first_line.starts_with("#!/bin/bash")
                    || first_line.starts_with("#!/bin/sh")
                    || first_line.starts_with("#!/usr/bin/env bash")
                    || first_line.starts_with("#!/usr/bin/env sh")
                {
                    return Some(path);
                }
            }
            None
        })
        .collect()
}

/// Computes the sha256 hex digest of `content`.
fn sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ── Main session-start hook ────────────────────────────────────────────────

/// Scans the workspace, building (or updating) the Knowledge Graph.
/// Designed to be called from `tokio::task::spawn_blocking`.
///
/// Strategy:
///   1. Try to load the cache from `~/.orangu/workspace/<hash>/graph/cache.json`.
///   2. For each source file, compare its sha256 to the cached hash.
///   3. Re-scan only stale or new files; keep cached nodes/edges for unchanged ones.
///   4. Save updated cache back to disk.
pub fn run_session_start_hook(workspace: &Path) -> WorkspaceScanResult {
    let cache_path = GraphCache::cache_path(workspace);
    scan_workspace(workspace, &cache_path)
}

fn scan_workspace(workspace: &Path, cache_path: &Path) -> WorkspaceScanResult {
    let project_type = detect_project_type(workspace);
    let files = collect_source_files(workspace);
    let current_paths = source_path_keys(workspace, &files);

    // Try loading the cache. If successful, start from the persisted graph.
    let (cache_opt, mut store) = match GraphCache::load(cache_path) {
        Some((cache, store)) => (Some(cache), store),
        None => (None, GraphStore::new()),
    };

    let mut warnings: Vec<String> = Vec::new();
    let mut new_file_hashes = cache_opt
        .as_ref()
        .map(|cache| cache.file_hashes.clone())
        .unwrap_or_default();
    let from_cache = cache_opt.is_some();

    match GraphExtractor::new() {
        Err(e) => {
            warnings.push(format!("Could not initialise graph extractor: {e}"));
        }
        Ok(extractor) => {
            let results: Vec<_> = files
                .par_iter()
                .map(|path| {
                    let path_key = path
                        .strip_prefix(workspace)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(e) => {
                            return (
                                path_key,
                                None,
                                Err(format!("{}: read error — {e}", path.display())),
                            );
                        }
                    };

                    let hash = sha256(&content);

                    if let Some(ref cache) = cache_opt
                        && !cache.is_stale(&path_key, &hash)
                    {
                        return (path_key, Some(hash), Ok(None));
                    }

                    match extractor.extract_from_file(path, &path_key, &content) {
                        Err(e) => (
                            path_key,
                            Some(hash),
                            Err(format!("{}: parse error — {e}", path.display())),
                        ),
                        Ok((nodes, edges)) => (path_key, Some(hash), Ok(Some((nodes, edges)))),
                    }
                })
                .collect();

            let mut all_edges = Vec::new();
            for (path_key, hash_opt, res) in results {
                match res {
                    Err(warn) => warnings.push(warn),
                    Ok(Some((nodes, edges))) => {
                        if let Some(hash) = hash_opt {
                            new_file_hashes.insert(path_key.clone(), hash);
                        }
                        // Evict old nodes for this file before inserting fresh
                        // ones. Without this, renamed/deleted symbols would
                        // persist as phantom nodes in the graph.
                        store.remove_nodes_for_file(&path_key);
                        for node in nodes {
                            store.add_node(node);
                        }
                        all_edges.extend(edges);
                    }
                    Ok(None) => {}
                }
            }
            // Pass 2: Add all edges only after ALL nodes are in the GraphStore
            for edge in all_edges {
                store.add_edge(edge);
            }
        }
    }

    // Purge nodes belonging to files that were in the cache but no longer exist on disk.
    if let Some(ref cache) = cache_opt {
        for cached_key in cache.file_hashes.keys() {
            if !current_paths.contains(cached_key) {
                store.remove_nodes_for_file(cached_key);
                new_file_hashes.remove(cached_key);
            }
        }
    }

    store.rebuild_edges();

    // Persist updated cache.
    GraphCache::save(cache_path, &store, &new_file_hashes);

    let stats = store.stats();
    let god_nodes = store.find_god_nodes(5);
    let has_cycles = store.has_cycles();

    WorkspaceScanResult {
        project_type,
        stats,
        god_nodes,
        has_cycles,
        store,
        warnings,
        from_cache,
        file_hashes: new_file_hashes,
    }
}

// ── Incremental rescan ──────────────────────────────────────────────────────

/// The minimal result of an incremental re-scan: how many files were actually
/// re-parsed and what the graph looks like now.
#[derive(Debug)]
pub struct IncrementalScanResult {
    pub rescanned: usize,
    pub stats: GraphStats,
}

fn source_path_keys(workspace: &Path, files: &[PathBuf]) -> HashSet<String> {
    files
        .iter()
        .map(|path| {
            path.strip_prefix(workspace)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

/// Rescans only source files whose content hash has changed since the last call.
///
/// Designed to be called from `tokio::task::spawn_blocking` on every LLM turn
/// after the WorldState diff is computed. The shared `store` is locked only
/// briefly when writing, so the UI is never blocked.
///
/// `file_hashes` is a mutable map of `relative_path → sha256` that persists
/// across calls so we only re-parse files that have actually changed.
pub fn rescan_changed_files(
    workspace: &Path,
    store: &std::sync::Arc<std::sync::Mutex<Option<GraphStore>>>,
    file_hashes: &mut HashMap<String, String>,
) -> IncrementalScanResult {
    let files = collect_source_files(workspace);
    let current_paths = source_path_keys(workspace, &files);
    let mut rescanned = 0usize;

    match GraphExtractor::new() {
        Err(_) => {}
        Ok(extractor) => {
            let results: Vec<_> = files
                .par_iter()
                .map(|path| {
                    let path_key = path
                        .strip_prefix(workspace)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(_) => return None,
                    };

                    let hash = sha256(&content);

                    if file_hashes.get(&path_key).is_some_and(|h| h == &hash) {
                        return None;
                    }

                    if let Ok((nodes, edges)) =
                        extractor.extract_from_file(path, &path_key, &content)
                    {
                        Some((path_key, hash, nodes, edges))
                    } else {
                        None
                    }
                })
                .collect();

            if let Ok(mut guard) = store.lock()
                && let Some(ref mut s) = *guard
            {
                let mut all_edges = Vec::new();
                let mut graph_changed = false;

                file_hashes.retain(|path_key, _| {
                    if current_paths.contains(path_key) {
                        true
                    } else {
                        s.remove_nodes_for_file(path_key);
                        graph_changed = true;
                        false
                    }
                });

                // Pass 1: Remove stale nodes, add all new nodes
                for res in results.into_iter().flatten() {
                    let (path_key, hash, nodes, edges) = res;
                    file_hashes.insert(path_key.clone(), hash);
                    rescanned += 1;
                    graph_changed = true;

                    s.remove_nodes_for_file(&path_key);
                    for node in nodes {
                        s.add_node(node);
                    }
                    all_edges.extend(edges);
                }

                // Pass 2: Add all edges now that node_map contains all cross-file targets
                for edge in all_edges {
                    s.add_edge(edge);
                }
                if graph_changed {
                    s.rebuild_edges();
                }
            }
        }
    }

    // Return stats from the live store.
    let stats = store
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|s| s.stats()))
        .unwrap_or(GraphStats {
            node_count: 0,
            edge_count: 0,
        });

    IncrementalScanResult { rescanned, stats }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, WorkspaceScanResult) {
        let workspace = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("caller.rs"),
            "fn caller() { callee(); }",
        )
        .unwrap();
        std::fs::write(workspace.path().join("callee.rs"), "fn callee() {}").unwrap();
        let scan = scan_workspace(workspace.path(), &cache.path().join("cache.json"));
        assert_eq!(scan.stats.edge_count, 1);
        (workspace, cache, scan)
    }

    #[test]
    fn startup_hashes_skip_unchanged_files_including_files_without_symbols() {
        let (workspace, cache, _) = fixture();
        std::fs::write(workspace.path().join("empty.rs"), "// no symbols\n").unwrap();
        let scan = scan_workspace(workspace.path(), &cache.path().join("cache.json"));
        let mut hashes = scan.file_hashes;
        assert_eq!(hashes["callee.rs"], sha256("fn callee() {}"));
        assert_eq!(hashes["empty.rs"], sha256("// no symbols\n"));
        let store = Arc::new(Mutex::new(Some(scan.store)));
        assert_eq!(
            rescan_changed_files(workspace.path(), &store, &mut hashes).rescanned,
            0
        );
    }

    #[test]
    fn live_rescan_retains_hashes_and_relationships_from_unchanged_callers() {
        let (workspace, _, scan) = fixture();
        let mut hashes = scan.file_hashes;
        let store = Arc::new(Mutex::new(Some(scan.store)));
        let changed = "fn callee() { let value = 1; }";
        std::fs::write(workspace.path().join("callee.rs"), changed).unwrap();
        let result = rescan_changed_files(workspace.path(), &store, &mut hashes);
        assert_eq!(result.rescanned, 1);
        assert_eq!(result.stats.edge_count, 1);
        assert_eq!(hashes["callee.rs"], sha256(changed));
        let guard = store.lock().unwrap();
        let context = guard.as_ref().unwrap().cross_file_context("callee.rs");
        assert_eq!(context[0].callers[0].node_label, "caller");
        drop(guard);
        assert_eq!(
            rescan_changed_files(workspace.path(), &store, &mut hashes).rescanned,
            0
        );

        // Replacing the caller must remove relationships it no longer contains.
        std::fs::write(workspace.path().join("caller.rs"), "fn caller() {}").unwrap();
        assert_eq!(
            rescan_changed_files(workspace.path(), &store, &mut hashes)
                .stats
                .edge_count,
            0
        );
    }

    #[test]
    fn live_rescan_removes_deleted_files_and_reconnects_restored_targets() {
        let (workspace, _, scan) = fixture();
        let remaining_nodes = scan
            .store
            .all_nodes()
            .iter()
            .filter(|node| node.source_file != "callee.rs")
            .count();
        let mut hashes = scan.file_hashes;
        let store = Arc::new(Mutex::new(Some(scan.store)));
        std::fs::remove_file(workspace.path().join("callee.rs")).unwrap();
        let result = rescan_changed_files(workspace.path(), &store, &mut hashes);
        assert_eq!(result.stats.node_count, remaining_nodes);
        assert_eq!(result.stats.edge_count, 0);
        assert!(!hashes.contains_key("callee.rs"));
        assert!(
            store
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .lookup("callee")
                .is_empty()
        );

        std::fs::write(workspace.path().join("renamed.rs"), "fn callee() {}").unwrap();
        let result = rescan_changed_files(workspace.path(), &store, &mut hashes);
        assert_eq!(result.rescanned, 1);
        assert_eq!(result.stats.edge_count, 1);
        assert!(hashes.contains_key("renamed.rs"));
    }

    #[test]
    fn cached_startup_preserves_incoming_edges_and_unresolved_references() {
        let (workspace, cache, _) = fixture();
        let cache_path = cache.path().join("cache.json");
        std::fs::write(
            workspace.path().join("callee.rs"),
            "fn callee() { let n = 2; }",
        )
        .unwrap();
        let scan = scan_workspace(workspace.path(), &cache_path);
        assert!(scan.from_cache);
        assert_eq!(scan.stats.edge_count, 1);

        std::fs::remove_file(workspace.path().join("callee.rs")).unwrap();
        let scan = scan_workspace(workspace.path(), &cache_path);
        assert_eq!(scan.stats.edge_count, 0);
        assert!(!scan.file_hashes.contains_key("callee.rs"));

        // The cached caller's unresolved reference must survive another restart.
        std::fs::write(workspace.path().join("new.rs"), "fn callee() {}").unwrap();
        let scan = scan_workspace(workspace.path(), &cache_path);
        assert_eq!(scan.stats.edge_count, 1);
    }

    #[test]
    fn live_rescan_rechecks_cross_file_name_ambiguity() {
        let (workspace, _, scan) = fixture();
        let mut hashes = scan.file_hashes;
        let store = Arc::new(Mutex::new(Some(scan.store)));
        std::fs::write(workspace.path().join("other.rs"), "fn callee() {}").unwrap();
        assert_eq!(
            rescan_changed_files(workspace.path(), &store, &mut hashes)
                .stats
                .edge_count,
            0
        );
        std::fs::remove_file(workspace.path().join("other.rs")).unwrap();
        assert_eq!(
            rescan_changed_files(workspace.path(), &store, &mut hashes)
                .stats
                .edge_count,
            1
        );
    }

    #[test]
    fn failed_reads_keep_the_previous_graph_and_hash_for_retry() {
        let (workspace, cache, scan) = fixture();
        std::fs::write(workspace.path().join("callee.rs"), [0xff]).unwrap();
        let mut hashes = scan.file_hashes;
        let store = Arc::new(Mutex::new(Some(scan.store)));
        let result = rescan_changed_files(workspace.path(), &store, &mut hashes);
        assert_eq!(result.rescanned, 0);
        assert_eq!(result.stats.edge_count, 1);
        assert_eq!(hashes["callee.rs"], sha256("fn callee() {}"));
        let scan = scan_workspace(workspace.path(), &cache.path().join("cache.json"));
        assert!(!scan.warnings.is_empty());
        assert_eq!(scan.stats.edge_count, 1);
        assert_eq!(scan.file_hashes["callee.rs"], hashes["callee.rs"]);
    }
}

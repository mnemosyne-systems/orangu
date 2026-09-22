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

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::extract::{Confidence, ExtractedEdge, ExtractedNode};
use super::store::{GraphNode, GraphStore};

const CACHE_VERSION: u32 = 2;

// ── On-disk format ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct CachedEdge {
    source: String,
    target: String,
    relation: String,
    confidence: Confidence,
    source_location: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    /// sha256 hex digest per source file path (relative to workspace root).
    file_hashes: HashMap<String, String>,
    nodes: Vec<GraphNode>,
    edges: Vec<CachedEdge>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Manages loading and saving the knowledge graph cache.
///
/// The cache lives at `~/.orangu/workspace/<hash>/graph/cache.json` — out of the
/// workspace tree, in the same global, per-workspace directory the embeddings
/// index uses (see [`crate::workspace_cache`]) — and stores:
/// - Per-file sha256 hashes for incremental re-scanning
/// - The full serialised graph (nodes + edges)
pub struct GraphCache {
    /// Hashes of files that were part of the most recently loaded/saved cache.
    pub file_hashes: HashMap<String, String>,
}

impl GraphCache {
    /// The on-disk path of the cache for `workspace`:
    /// `~/.orangu/workspace/<sha256(path)>/graph/cache.json`.
    pub fn cache_path(workspace: &Path) -> std::path::PathBuf {
        crate::workspace_cache::workspace_cache_dir(workspace, "graph").join("cache.json")
    }

    /// Loads the cache from `path`. Returns `None` if the file doesn't exist or
    /// is invalid / version-mismatch (triggers a full rescan).
    pub fn load(path: &Path) -> Option<(GraphCache, GraphStore)> {
        let content = std::fs::read_to_string(path).ok()?;
        let cached: CacheFile = serde_json::from_str(&content).ok()?;

        if cached.version != CACHE_VERSION {
            return None;
        }

        // Rebuild the GraphStore from the serialised nodes and edges.
        let mut store = GraphStore::new();

        for node in cached.nodes {
            store.add_node(ExtractedNode {
                id: node.id,
                label: node.label,
                source_file: node.source_file,
                source_location: node.source_location,
                kind: node.kind,
            });
        }

        for edge in cached.edges {
            store.add_edge(ExtractedEdge {
                source: edge.source,
                target: edge.target,
                relation: edge.relation,
                confidence: edge.confidence,
                source_location: edge.source_location,
            });
        }

        Some((
            GraphCache {
                file_hashes: cached.file_hashes,
            },
            store,
        ))
    }

    /// Saves the current `store` and `file_hashes` to `path`, creating any
    /// missing parent directories. Silently ignores write errors to avoid
    /// crashing the session over a non-critical cache failure.
    ///
    /// The file is written through a per-process temporary and renamed into
    /// place, so a reader never sees half a cache and two scanners of the same
    /// workspace — two sessions, or a `graph_lookup` in a subagent that keeps
    /// its own graph — cannot interleave into one corrupt file. The loser of
    /// that race simply overwrites the winner with an equivalent cache.
    pub fn save(path: &Path, store: &GraphStore, file_hashes: &HashMap<String, String>) {
        let edges: Vec<CachedEdge> = store
            .all_edge_data()
            .into_iter()
            .map(|(source, target, edge)| CachedEdge {
                source,
                target,
                relation: edge.relation.clone(),
                confidence: edge.confidence.clone(),
                source_location: edge.source_location.clone(),
            })
            .collect();

        let cache_file = CacheFile {
            version: CACHE_VERSION,
            file_hashes: file_hashes.clone(),
            nodes: store.all_nodes().into_iter().cloned().collect(),
            edges,
        };

        let Ok(json) = serde_json::to_string_pretty(&cache_file) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        if std::fs::write(&temporary, json).is_ok() && std::fs::rename(&temporary, path).is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
    }

    /// Returns `true` if the file at `path` (relative string key) has a
    /// sha256 hash that differs from what is stored in this cache — meaning
    /// the file needs to be re-scanned.
    pub fn is_stale(&self, path_key: &str, current_hash: &str) -> bool {
        match self.file_hashes.get(path_key) {
            Some(cached) => cached != current_hash,
            None => true, // not in cache → new file, must scan
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract::ExtractedNode;

    /// A saved cache loads back, and the rename leaves no temporary behind for
    /// the next scan (or a `du` of `~/.orangu`) to trip over.
    #[test]
    fn a_saved_cache_loads_back_and_leaves_no_temporary() {
        let dir = tempfile::tempdir().expect("cache dir");
        let path = dir.path().join("graph").join("cache.json");
        let mut store = GraphStore::new();
        store.add_node(ExtractedNode {
            id: "cached::symbol".to_string(),
            label: "symbol".to_string(),
            source_file: "src/lib.rs".to_string(),
            source_location: "1".to_string(),
            kind: "function".to_string(),
        });
        let hashes = HashMap::from([("src/lib.rs".to_string(), "abc".to_string())]);

        GraphCache::save(&path, &store, &hashes);

        let (cache, loaded) = GraphCache::load(&path).expect("cache loads back");
        assert_eq!(loaded.stats().node_count, 1);
        assert!(!cache.is_stale("src/lib.rs", "abc"));
        assert!(cache.is_stale("src/lib.rs", "def"));

        let left_behind: Vec<String> = std::fs::read_dir(path.parent().expect("parent"))
            .expect("cache dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(left_behind, vec!["cache.json".to_string()]);
    }
}

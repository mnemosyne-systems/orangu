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

use petgraph::algo::is_cyclic_directed;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use super::extract::{Confidence, ExtractedEdge, ExtractedNode};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub source_file: String,
    pub source_location: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub relation: String,
    pub confidence: Confidence,
    pub source_location: String,
}

/// The central in-memory knowledge graph.
///
/// - Backed by a `petgraph::DiGraph` for O(1) traversal.
/// - A companion `HashMap<id → NodeIndex>` provides O(1) lookup by symbol id.
/// - Deduplication: inserting a node whose id already exists *overwrites* the
///   existing entry (semantic nodes win over structural ones).
#[derive(Debug)]
pub struct GraphStore {
    graph: StableDiGraph<GraphNode, GraphEdge>,
    node_map: HashMap<String, NodeIndex>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    pub fn add_node(&mut self, node: ExtractedNode) -> NodeIndex {
        if let Some(&index) = self.node_map.get(&node.id) {
            // Overwrite to keep the richest semantic context.
            self.graph[index] = GraphNode {
                id: node.id,
                label: node.label,
                source_file: node.source_file,
                source_location: node.source_location,
                kind: node.kind,
            };
            index
        } else {
            let id = node.id.clone();
            let index = self.graph.add_node(GraphNode {
                id: node.id,
                label: node.label,
                source_file: node.source_file,
                source_location: node.source_location,
                kind: node.kind,
            });
            self.node_map.insert(id, index);
            index
        }
    }

    pub fn add_edge(&mut self, edge: ExtractedEdge) {
        let src = self.node_map.get(&edge.source).cloned();
        // `GraphExtractor::extract_from_file` can only recognize a call as
        // "local" — and so emit the callee's real id — when the callee is
        // defined in the *same file* it's scanning; a call to a symbol
        // defined elsewhere gets a synthetic `external::<name>` target
        // instead (its confidence is already `Inferred` for exactly this
        // reason). `resolve_external_target` is the fallback that turns that
        // placeholder into the real cross-file node once the whole graph —
        // every file's nodes — is known, which by the time `add_edge` runs it
        // always is (callers add every node before any edge; see
        // `agents::hooks::run_session_start_hook`'s two-pass structure).
        let tgt = self
            .node_map
            .get(&edge.target)
            .cloned()
            .or_else(|| self.resolve_external_target(&edge.target));

        if let (Some(s), Some(t)) = (src, tgt) {
            // Deduplicate: skip if an edge with the same relation already exists
            // between these two nodes.
            let already_exists = self
                .graph
                .edges_directed(s, petgraph::Direction::Outgoing)
                .any(|e| e.target() == t && e.weight().relation == edge.relation);

            if !already_exists {
                self.graph.add_edge(
                    s,
                    t,
                    GraphEdge {
                        relation: edge.relation,
                        confidence: edge.confidence,
                        source_location: edge.source_location,
                    },
                );
            }
        }
    }

    /// Resolve a synthetic `external::<name>` edge target — the placeholder
    /// `add_edge` receives for a call whose callee isn't defined in the
    /// calling file (see its doc comment) — against every node in the graph
    /// by bare name (`GraphNode::label`, the same bare identifier the
    /// extractor captured at the call site). Resolves only when exactly one
    /// node anywhere carries that label: a cross-file name collision (two
    /// files each defining, say, `new()`) is worse to guess wrong on than to
    /// simply leave the edge dropped, so 0 or 2+ matches both return `None`.
    /// A no-op (`None`) for any `target` that isn't an `external::` id.
    fn resolve_external_target(&self, target: &str) -> Option<NodeIndex> {
        let name = target.strip_prefix("external::")?;
        let mut matches = self
            .graph
            .node_indices()
            .filter(|&idx| self.graph[idx].label == name);
        let only = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        Some(only)
    }

    /// Removes all nodes (and their edges) whose `source_file` matches `file_path`.
    /// Called before re-extracting a file so stale graph data is fully replaced.
    pub fn remove_nodes_for_file(&mut self, file_path: &str) {
        // Collect indices to remove — must not mutate the graph while iterating.
        let to_remove: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&idx| self.graph[idx].source_file == file_path)
            .collect();

        for idx in to_remove {
            let id = self.graph[idx].id.clone();
            self.node_map.remove(&id);
            self.graph.remove_node(idx);
        }
    }

    // ── Analysis ─────────────────────────────────────────────────────────────

    /// Returns the top-`n` nodes sorted by total degree (in + out edges),
    /// often called "God Nodes" — symbols that are deeply central to the graph.
    pub fn find_god_nodes(&self, top_n: usize) -> Vec<GodNodeEntry> {
        let mut scored: Vec<GodNodeEntry> = self
            .graph
            .node_indices()
            .map(|idx| {
                let degree = self.graph.edges(idx).count()
                    + self
                        .graph
                        .edges_directed(idx, petgraph::Direction::Incoming)
                        .count();
                GodNodeEntry {
                    id: self.graph[idx].id.clone(),
                    label: self.graph[idx].label.clone(),
                    kind: self.graph[idx].kind.clone(),
                    degree,
                }
            })
            .collect();

        scored.sort_by_key(|b| std::cmp::Reverse(b.degree));
        scored.truncate(top_n);
        scored
    }

    /// Returns `true` if the graph contains at least one directed cycle.
    /// A cycle in a dependency/call graph signals a circular dependency.
    pub fn has_cycles(&self) -> bool {
        is_cyclic_directed(&self.graph)
    }

    /// Explain one unambiguous symbol with the same evidence kept on the graph
    /// edges. Exact id/label matches win; prefix and substring matches are only
    /// used when no exact match exists. Ties are reported instead of choosing a
    /// random same-named symbol from another file.
    pub fn explain(&self, symbol: &str) -> Result<GraphExplanation, String> {
        let idx = self.resolve_node(symbol)?;
        let (callers, callees) = self.neighbours(idx);
        let mut connections = Vec::with_capacity(callers.len() + callees.len());
        connections.extend(callers.into_iter().map(|edge| GraphConnection {
            direction: ConnectionDirection::Incoming,
            edge,
        }));
        connections.extend(callees.into_iter().map(|edge| GraphConnection {
            direction: ConnectionDirection::Outgoing,
            edge,
        }));
        connections.sort_by(|a, b| {
            self.node_degree_by_id(&b.edge.node_id)
                .cmp(&self.node_degree_by_id(&a.edge.node_id))
                .then_with(|| a.edge.node_label.cmp(&b.edge.node_label))
        });

        Ok(GraphExplanation {
            node: self.graph[idx].clone(),
            community: self.community_for(idx),
            degree: connections.len(),
            connections,
        })
    }

    /// Find the shortest relationship path between two unambiguous symbols.
    /// Traversal follows call/import direction by default; `undirected` is for
    /// architectural discovery where callers and callees are equally useful.
    pub fn shortest_path(
        &self,
        source: &str,
        target: &str,
        undirected: bool,
        max_hops: usize,
    ) -> Result<GraphPath, String> {
        let start = self.resolve_node(source)?;
        let goal = self.resolve_node(target)?;
        if start == goal {
            return Err(format!(
                "\"{source}\" and \"{target}\" resolve to the same node; use a more specific symbol or id"
            ));
        }

        let mut queue = VecDeque::from([start]);
        let mut visited = HashSet::from([start]);
        let mut previous: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        while let Some(current) = queue.pop_front() {
            let neighbours: Vec<NodeIndex> = if undirected {
                self.graph.neighbors_undirected(current).collect()
            } else {
                self.graph
                    .neighbors_directed(current, petgraph::Direction::Outgoing)
                    .collect()
            };
            for next in neighbours {
                if visited.insert(next) {
                    previous.insert(next, current);
                    if next == goal {
                        queue.clear();
                        break;
                    }
                    queue.push_back(next);
                }
            }
        }

        if !visited.contains(&goal) {
            let direction = if undirected { "" } else { "directed " };
            return Err(format!(
                "No {direction}path found between \"{source}\" and \"{target}\".{}",
                if undirected {
                    String::new()
                } else {
                    " Try again with undirected=true to ignore relationship direction.".to_string()
                }
            ));
        }

        let mut nodes = vec![goal];
        let mut cursor = goal;
        while let Some(&parent) = previous.get(&cursor) {
            nodes.push(parent);
            cursor = parent;
        }
        nodes.reverse();
        let hops = nodes.len() - 1;
        if hops > max_hops {
            return Err(format!(
                "Path exceeds max_hops={max_hops} ({hops} hops found)."
            ));
        }

        let mut path_hops = Vec::with_capacity(hops);
        for pair in nodes.windows(2) {
            let from = pair[0];
            let to = pair[1];
            let (edge, forward) = self.path_edge(from, to).ok_or_else(|| {
                "graph traversal selected nodes without a relationship edge".to_string()
            })?;
            path_hops.push(GraphPathHop {
                from: self.graph[from].clone(),
                to: self.graph[to].clone(),
                relation: edge.relation.clone(),
                confidence: edge.confidence.clone(),
                source_location: edge.source_location.clone(),
                forward,
            });
        }

        Ok(GraphPath { path_hops })
    }

    fn resolve_node(&self, symbol: &str) -> Result<NodeIndex, String> {
        let needle = symbol.trim().to_lowercase();
        if needle.is_empty() {
            return Err("symbol must not be empty".to_string());
        }
        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let tiers = [
            indices
                .iter()
                .copied()
                .filter(|&idx| {
                    let node = &self.graph[idx];
                    node.id.eq_ignore_ascii_case(symbol) || node.label.eq_ignore_ascii_case(symbol)
                })
                .collect::<Vec<_>>(),
            indices
                .iter()
                .copied()
                .filter(|&idx| {
                    let node = &self.graph[idx];
                    node.id.to_lowercase().starts_with(&needle)
                        || node.label.to_lowercase().starts_with(&needle)
                })
                .collect::<Vec<_>>(),
            indices
                .iter()
                .copied()
                .filter(|&idx| {
                    let node = &self.graph[idx];
                    node.id.to_lowercase().contains(&needle)
                        || node.label.to_lowercase().contains(&needle)
                })
                .collect::<Vec<_>>(),
        ];
        let matches = tiers
            .into_iter()
            .find(|tier| !tier.is_empty())
            .ok_or_else(|| {
                format!("No node matching \"{symbol}\" found in the Knowledge Graph.")
            })?;
        if matches.len() == 1 {
            return Ok(matches[0]);
        }
        let mut candidates: Vec<String> = matches
            .iter()
            .map(|&idx| {
                let node = &self.graph[idx];
                format!("{} ({})", node.id, node.source_file)
            })
            .collect();
        candidates.sort();
        Err(format!(
            "Ambiguous symbol \"{symbol}\"; use an exact id. Candidates: {}",
            candidates.join(", ")
        ))
    }

    fn path_edge(&self, from: NodeIndex, to: NodeIndex) -> Option<(&GraphEdge, bool)> {
        let mut forward: Vec<&GraphEdge> = self
            .graph
            .edges_directed(from, petgraph::Direction::Outgoing)
            .filter(|edge| edge.target() == to)
            .map(|edge| edge.weight())
            .collect();
        forward.sort_by(|a, b| a.relation.cmp(&b.relation));
        if let Some(edge) = forward.first() {
            return Some((*edge, true));
        }
        let mut backward: Vec<&GraphEdge> = self
            .graph
            .edges_directed(to, petgraph::Direction::Outgoing)
            .filter(|edge| edge.target() == from)
            .map(|edge| edge.weight())
            .collect();
        backward.sort_by(|a, b| a.relation.cmp(&b.relation));
        backward.first().map(|edge| (*edge, false))
    }

    fn node_degree_by_id(&self, id: &str) -> usize {
        self.node_map.get(id).map_or(0, |&idx| {
            self.graph.edges(idx).count()
                + self
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .count()
        })
    }

    /// A deterministic, dependency-free structural partition. The graph is
    /// projected to undirected edges and label propagation is run in stable id
    /// order, so a cached graph gets the same community number across queries.
    fn community_for(&self, target: NodeIndex) -> usize {
        let mut nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
        nodes.sort_by(|a, b| self.graph[*a].id.cmp(&self.graph[*b].id));
        let mut labels: HashMap<NodeIndex, String> = nodes
            .iter()
            .map(|&idx| (idx, self.graph[idx].id.clone()))
            .collect();
        for _ in 0..24 {
            let prior = labels.clone();
            let mut changed = false;
            for &idx in &nodes {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for neighbour in self.graph.neighbors_undirected(idx) {
                    if let Some(label) = prior.get(&neighbour) {
                        *counts.entry(label.as_str()).or_default() += 1;
                    }
                }
                if let Some((winner, _)) = counts.into_iter().max_by(
                    |(left_label, left_count), (right_label, right_count)| {
                        left_count
                            .cmp(right_count)
                            .then_with(|| right_label.cmp(left_label))
                    },
                ) && labels.get(&idx).is_some_and(|current| current != winner)
                {
                    labels.insert(idx, winner.to_string());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut groups: Vec<String> = labels.values().cloned().collect();
        groups.sort();
        groups.dedup();
        let label = labels.get(&target).expect("target graph node has a label");
        groups
            .binary_search(label)
            .expect("label appears in community list")
            + 1
    }

    /// The callers (in-edges) and callees (out-edges) of the node at `idx`, as
    /// `(callers, callees)`. Shared by `lookup` and `cross_file_context`.
    fn neighbours(&self, idx: NodeIndex) -> (Vec<NeighbourEdge>, Vec<NeighbourEdge>) {
        let callers = self
            .graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| NeighbourEdge {
                node_id: self.graph[e.source()].id.clone(),
                node_label: self.graph[e.source()].label.clone(),
                relation: e.weight().relation.clone(),
                confidence: e.weight().confidence.clone(),
            })
            .collect();
        let callees = self
            .graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|e| NeighbourEdge {
                node_id: self.graph[e.target()].id.clone(),
                node_label: self.graph[e.target()].label.clone(),
                relation: e.weight().relation.clone(),
                confidence: e.weight().confidence.clone(),
            })
            .collect();
        (callers, callees)
    }

    /// Looks up all nodes whose `id` or `label` contains `symbol` (case-insensitive).
    /// For each match returns the node itself plus all its in-edges (callers) and
    /// out-edges (callees), formatted for direct use in the `graph_lookup` tool.
    pub fn lookup(&self, symbol: &str) -> Vec<LookupResult> {
        let needle = symbol.to_lowercase();

        let mut results: Vec<LookupResult> = self
            .graph
            .node_indices()
            .filter(|&idx| {
                let n = &self.graph[idx];
                n.id.to_lowercase().contains(&needle) || n.label.to_lowercase().contains(&needle)
            })
            .map(|idx| {
                let (callers, callees) = self.neighbours(idx);
                LookupResult {
                    node: self.graph[idx].clone(),
                    callers,
                    callees,
                    god_rank: None,
                }
            })
            .collect();

        // Sort results so the most highly connected matches appear first.
        results.sort_by(|a, b| {
            let degree_a = a.callers.len() + a.callees.len();
            let degree_b = b.callers.len() + b.callees.len();
            degree_b.cmp(&degree_a)
        });

        results
    }

    /// The nodes defined in `file_path`, each with only the callers/callees
    /// that live in a *different* file — the cross-file relationships a
    /// diff-plus-whole-file review can't see on its own (e.g. a signature
    /// change breaking a caller elsewhere). Nodes with no cross-file
    /// neighbours are omitted; same-file neighbours are dropped since they're
    /// already visible in the file content the review sends. Used by
    /// `/auto_review`'s Deep mode.
    pub fn cross_file_context(&self, file_path: &str) -> Vec<LookupResult> {
        let mut results: Vec<LookupResult> = self
            .graph
            .node_indices()
            .filter(|&idx| self.graph[idx].source_file == file_path)
            .filter_map(|idx| {
                let (callers, callees) = self.neighbours(idx);
                let is_cross_file = |edge: &NeighbourEdge| {
                    self.node_map
                        .get(&edge.node_id)
                        .is_some_and(|&other| self.graph[other].source_file != file_path)
                };
                let callers: Vec<_> = callers.into_iter().filter(is_cross_file).collect();
                let callees: Vec<_> = callees.into_iter().filter(is_cross_file).collect();
                if callers.is_empty() && callees.is_empty() {
                    return None;
                }
                Some(LookupResult {
                    node: self.graph[idx].clone(),
                    callers,
                    callees,
                    god_rank: None,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            let degree_a = a.callers.len() + a.callees.len();
            let degree_b = b.callers.len() + b.callees.len();
            degree_b.cmp(&degree_a)
        });

        results
    }

    /// Pluribus Predictive Group Vectors: Predicts the most strongly coupled files
    /// (subsystems) to `file_path` that the LLM is likely to need next.
    /// Returns a list of predicted file paths, sorted by coupling strength.
    pub fn predictive_group_vectors(&self, file_path: &str) -> Vec<String> {
        let mut file_scores: HashMap<String, usize> = HashMap::new();

        for idx in self.graph.node_indices() {
            if self.graph[idx].source_file == file_path {
                for edge in self
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                {
                    let other_file = &self.graph[edge.source()].source_file;
                    if other_file != file_path {
                        *file_scores.entry(other_file.clone()).or_insert(0) += 1;
                    }
                }
                for edge in self
                    .graph
                    .edges_directed(idx, petgraph::Direction::Outgoing)
                {
                    let other_file = &self.graph[edge.target()].source_file;
                    if other_file != file_path {
                        *file_scores.entry(other_file.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut predictions: Vec<(String, usize)> = file_scores.into_iter().collect();
        predictions.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        predictions.into_iter().map(|(f, _)| f).collect()
    }

    /// Returns all nodes in the graph as a flat list.
    pub fn all_nodes(&self) -> Vec<&GraphNode> {
        self.graph.node_weights().collect()
    }

    /// Returns the total number of nodes and edges.
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
        }
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    /// Returns all edges as `(source_id, target_id, &edge_weight)` tuples.
    /// Used by `GraphCache::save()` for incremental persistence.
    pub fn all_edge_data(&self) -> Vec<(String, String, &GraphEdge)> {
        self.graph
            .edge_references()
            .map(|e| {
                (
                    self.graph[e.source()].id.clone(),
                    self.graph[e.target()].id.clone(),
                    e.weight(),
                )
            })
            .collect()
    }

    /// Serialises the complete graph to a JSON string suitable for persistence.
    pub fn to_json(&self) -> anyhow::Result<String> {
        #[derive(Serialize)]
        struct Export<'a> {
            nodes: Vec<&'a GraphNode>,
            edges: Vec<ExportEdge<'a>>,
        }

        #[derive(Serialize)]
        struct ExportEdge<'a> {
            source: &'a str,
            target: &'a str,
            relation: &'a str,
            confidence: &'a Confidence,
            source_location: &'a str,
        }

        let edges: Vec<ExportEdge> = self
            .graph
            .edge_references()
            .map(|e| ExportEdge {
                source: &self.graph[e.source()].id,
                target: &self.graph[e.target()].id,
                relation: &e.weight().relation,
                confidence: &e.weight().confidence,
                source_location: &e.weight().source_location,
            })
            .collect();

        let export = Export {
            nodes: self.all_nodes(),
            edges,
        };

        Ok(serde_json::to_string_pretty(&export)?)
    }
}

// ── Supporting types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GodNodeEntry {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub degree: usize,
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
}

/// One relationship displayed by [`GraphExplanation`].
#[derive(Debug, Clone)]
pub struct GraphConnection {
    pub direction: ConnectionDirection,
    pub edge: NeighbourEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionDirection {
    Incoming,
    Outgoing,
}

/// A concise, evidence-preserving view of a single graph symbol.
#[derive(Debug, Clone)]
pub struct GraphExplanation {
    pub node: GraphNode,
    /// A deterministic structural cluster id derived from graph connectivity.
    pub community: usize,
    pub degree: usize,
    pub connections: Vec<GraphConnection>,
}

impl GraphExplanation {
    pub fn format(&self) -> String {
        let mut out = format!(
            "Node: {}\nSource: {} {}\nKind: {}\nCommunity: {}\nDegree: {}\n",
            self.node.label,
            self.node.source_file,
            self.node.source_location,
            self.node.kind,
            self.community,
            self.degree,
        );
        if self.connections.is_empty() {
            out.push_str("Connections: none\n");
            return out;
        }
        out.push_str(&format!("Connections ({}):\n", self.connections.len()));
        for connection in self.connections.iter().take(20) {
            let arrow = match connection.direction {
                ConnectionDirection::Incoming => "<--",
                ConnectionDirection::Outgoing => "-->",
            };
            out.push_str(&format!(
                "{arrow} {} [{}] [{:?}]\n",
                connection.edge.node_label, connection.edge.relation, connection.edge.confidence
            ));
        }
        if self.connections.len() > 20 {
            out.push_str(&format!("... and {} more\n", self.connections.len() - 20));
        }
        out
    }
}

/// A single hop in a shortest path. `forward` records whether graph direction
/// agrees with the displayed traversal (false only for an undirected search).
#[derive(Debug, Clone)]
pub struct GraphPathHop {
    pub from: GraphNode,
    pub to: GraphNode,
    pub relation: String,
    pub confidence: Confidence,
    pub source_location: String,
    pub forward: bool,
}

#[derive(Debug, Clone)]
pub struct GraphPath {
    pub path_hops: Vec<GraphPathHop>,
}

impl GraphPath {
    pub fn format(&self) -> String {
        let mut out = format!("Shortest path ({} hops):\n  ", self.path_hops.len());
        if let Some(first) = self.path_hops.first() {
            out.push_str(&first.from.label);
        }
        for hop in &self.path_hops {
            if hop.forward {
                out.push_str(&format!(
                    " --{} [{:?}] @{}:{}--> {}",
                    hop.relation,
                    hop.confidence,
                    hop.from.source_file,
                    hop.source_location,
                    hop.to.label
                ));
            } else {
                out.push_str(&format!(
                    " <--{} [{:?}] @{}:{}-- {}",
                    hop.relation,
                    hop.confidence,
                    hop.to.source_file,
                    hop.source_location,
                    hop.to.label
                ));
            }
        }
        out
    }
}

/// A single edge in a lookup result — either a caller or a callee of the matched node.
#[derive(Debug, Clone)]
pub struct NeighbourEdge {
    pub node_id: String,
    pub node_label: String,
    pub relation: String,
    pub confidence: Confidence,
}

/// The result of a `graph_lookup` query for one matched node.
#[derive(Debug, Clone)]
pub struct LookupResult {
    pub node: GraphNode,
    /// Nodes that have an edge pointing *into* this node (callers / importers).
    pub callers: Vec<NeighbourEdge>,
    /// Nodes that this node has an edge pointing *to* (callees / imports).
    pub callees: Vec<NeighbourEdge>,
    /// Human-readable rank string e.g. "#3 of 142", or None if no edges.
    pub god_rank: Option<String>,
}

impl LookupResult {
    /// Formats the result as a human-readable string the agent can read directly.
    pub fn format(&self) -> String {
        let mut out = format!(
            "[Graph Lookup: \"{}\"]\n{} ({}, {})\n",
            self.node.label, self.node.id, self.node.kind, self.node.source_file,
        );
        if let Some(rank) = &self.god_rank {
            out.push_str(&format!("God Node rank: {}\n", rank));
        }
        if self.callers.is_empty() {
            out.push_str("\nCallers: none\n");
        } else {
            out.push_str("\nCallers (things that call/use this node):\n");
            for c in &self.callers {
                out.push_str(&format!(
                    "  • {}  →  {}  ({:?})\n",
                    c.node_label, c.relation, c.confidence
                ));
            }
        }
        if self.callees.is_empty() {
            out.push_str("\nCallees: none\n");
        } else {
            out.push_str("\nCallees (things this node calls/imports):\n");
            for c in &self.callees {
                out.push_str(&format!(
                    "  • {}  →  {}  ({:?})\n",
                    c.node_label, c.relation, c.confidence
                ));
            }
        }
        out
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract::{Confidence, ExtractedEdge, ExtractedNode};

    fn make_node(id: &str, kind: &str) -> ExtractedNode {
        make_node_in(id, kind, "test.rs")
    }

    fn make_node_in(id: &str, kind: &str, source_file: &str) -> ExtractedNode {
        ExtractedNode {
            id: id.to_string(),
            label: id.to_string(),
            source_file: source_file.to_string(),
            source_location: "L1-L5".to_string(),
            kind: kind.to_string(),
        }
    }

    fn make_edge(src: &str, tgt: &str, relation: &str) -> ExtractedEdge {
        ExtractedEdge {
            source: src.to_string(),
            target: tgt.to_string(),
            relation: relation.to_string(),
            confidence: Confidence::Extracted,
            source_location: "L1".to_string(),
        }
    }

    #[test]
    fn deduplicates_nodes() {
        let mut store = GraphStore::new();
        store.add_node(make_node("a::foo", "fn"));
        store.add_node(make_node("a::foo", "fn")); // duplicate
        assert_eq!(store.stats().node_count, 1);
    }

    #[test]
    fn finds_god_nodes() {
        let mut store = GraphStore::new();
        for name in ["a::hub", "b::foo", "c::bar", "d::baz"] {
            store.add_node(make_node(name, "fn"));
        }
        // hub is called by all others
        store.add_edge(make_edge("b::foo", "a::hub", "calls"));
        store.add_edge(make_edge("c::bar", "a::hub", "calls"));
        store.add_edge(make_edge("d::baz", "a::hub", "calls"));

        let gods = store.find_god_nodes(1);
        assert_eq!(gods[0].id, "a::hub");
    }

    #[test]
    fn detects_cycles() {
        let mut store = GraphStore::new();
        store.add_node(make_node("a::fn1", "fn"));
        store.add_node(make_node("a::fn2", "fn"));
        store.add_edge(make_edge("a::fn1", "a::fn2", "calls"));
        store.add_edge(make_edge("a::fn2", "a::fn1", "calls")); // cycle!
        assert!(store.has_cycles());
    }

    #[test]
    fn explain_reports_evidence_and_rejects_ambiguous_names() {
        let mut store = GraphStore::new();
        store.add_node(make_extracted_node(
            "routing::router",
            "router",
            "routing.rs",
        ));
        store.add_node(make_extracted_node("api::router", "router", "api.rs"));
        store.add_node(make_extracted_node("main::serve", "serve", "main.rs"));
        store.add_edge(make_edge("main::serve", "routing::router", "calls"));

        let explanation = store.explain("routing::router").unwrap();
        assert_eq!(explanation.degree, 1);
        assert_eq!(
            explanation.connections[0].direction,
            ConnectionDirection::Incoming
        );
        assert!(store.explain("router").unwrap_err().contains("Ambiguous"));
    }

    #[test]
    fn shortest_path_is_directed_by_default_and_can_be_undirected() {
        let mut store = GraphStore::new();
        for (id, label) in [
            ("a::start", "start"),
            ("b::middle", "middle"),
            ("c::end", "end"),
        ] {
            store.add_node(make_extracted_node(id, label, "test.rs"));
        }
        store.add_edge(make_edge("a::start", "b::middle", "calls"));
        store.add_edge(make_edge("b::middle", "c::end", "uses"));

        let path = store.shortest_path("start", "end", false, 8).unwrap();
        assert_eq!(path.path_hops.len(), 2);
        assert_eq!(path.path_hops[0].relation, "calls");
        assert!(store.shortest_path("end", "start", false, 8).is_err());
        let reverse = store.shortest_path("end", "start", true, 8).unwrap();
        assert_eq!(reverse.path_hops.len(), 2);
        assert!(!reverse.path_hops[0].forward);
    }

    /// A node as `GraphExtractor::extract_from_file` actually produces one:
    /// `id` is `<file_stem>::<name>` (qualified) but `label` is the bare
    /// symbol name alone — the distinction `resolve_external_target` (and
    /// these tests) depend on. `make_node`/`make_node_in` set `label` equal
    /// to `id`, which doesn't exercise that distinction, so the two tests
    /// below build the node directly instead.
    fn make_extracted_node(id: &str, label: &str, source_file: &str) -> ExtractedNode {
        ExtractedNode {
            id: id.to_string(),
            label: label.to_string(),
            source_file: source_file.to_string(),
            source_location: "L1-L5".to_string(),
            kind: "fn".to_string(),
        }
    }

    #[test]
    fn add_edge_resolves_an_external_target_to_the_real_cross_file_node() {
        // Mirrors what `GraphExtractor::extract_from_file` emits for a call
        // whose callee isn't defined in the calling file: a synthetic
        // `external::<name>` target instead of the real cross-file id.
        let mut store = GraphStore::new();
        store.add_node(make_extracted_node("b::caller", "caller", "b.rs"));
        store.add_node(make_extracted_node("a::changed", "changed", "a.rs"));
        store.add_edge(ExtractedEdge {
            source: "b::caller".to_string(),
            target: "external::changed".to_string(),
            relation: "calls".to_string(),
            confidence: Confidence::Inferred,
            source_location: "L4".to_string(),
        });

        let context = store.cross_file_context("a.rs");
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].callers.len(), 1);
        assert_eq!(context[0].callers[0].node_id, "b::caller");
    }

    #[test]
    fn add_edge_leaves_an_ambiguous_external_target_unresolved() {
        // Two files each define a `changed` symbol: the bare name alone
        // can't say which one the call meant, so the edge is dropped rather
        // than guessed.
        let mut store = GraphStore::new();
        store.add_node(make_extracted_node("b::caller", "caller", "b.rs"));
        store.add_node(make_extracted_node("a::changed", "changed", "a.rs"));
        store.add_node(make_extracted_node("c::changed", "changed", "c.rs"));
        store.add_edge(ExtractedEdge {
            source: "b::caller".to_string(),
            target: "external::changed".to_string(),
            relation: "calls".to_string(),
            confidence: Confidence::Inferred,
            source_location: "L4".to_string(),
        });

        assert!(store.cross_file_context("a.rs").is_empty());
        assert!(store.cross_file_context("c.rs").is_empty());
    }

    #[test]
    fn serialises_to_json() {
        let mut store = GraphStore::new();
        store.add_node(make_node("a::foo", "fn"));
        let json = store.to_json().unwrap();
        assert!(json.contains("a::foo"));
    }

    #[test]
    fn cross_file_context_keeps_only_neighbours_in_other_files() {
        let mut store = GraphStore::new();
        store.add_node(make_node_in("a::changed", "fn", "a.rs"));
        store.add_node(make_node_in("a::same_file_helper", "fn", "a.rs"));
        store.add_node(make_node_in("b::caller", "fn", "b.rs"));
        store.add_node(make_node_in("c::unrelated", "fn", "c.rs"));
        // A same-file edge (dropped: already visible in the file content) and
        // a cross-file edge (kept: the review can't otherwise see it).
        store.add_edge(make_edge("a::changed", "a::same_file_helper", "calls"));
        store.add_edge(make_edge("b::caller", "a::changed", "calls"));

        let context = store.cross_file_context("a.rs");
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].node.id, "a::changed");
        assert_eq!(context[0].callers.len(), 1);
        assert_eq!(context[0].callers[0].node_id, "b::caller");
        assert!(context[0].callees.is_empty());

        // A file with no cross-file neighbours contributes nothing.
        assert!(store.cross_file_context("c.rs").is_empty());
    }

    #[test]
    fn predictive_group_vectors_ranks_files_by_coupling_strength() {
        let mut store = GraphStore::new();
        store.add_node(make_node_in("a::changed1", "fn", "a.rs"));
        store.add_node(make_node_in("a::changed2", "fn", "a.rs"));

        store.add_node(make_node_in("b::coupled1", "fn", "b.rs"));
        store.add_node(make_node_in("b::coupled2", "fn", "b.rs"));
        store.add_node(make_node_in("b::coupled3", "fn", "b.rs"));

        store.add_node(make_node_in("c::weak_coupled1", "fn", "c.rs"));

        // a.rs has 3 edges with b.rs
        store.add_edge(make_edge("a::changed1", "b::coupled1", "calls"));
        store.add_edge(make_edge("a::changed1", "b::coupled2", "calls"));
        store.add_edge(make_edge("b::coupled3", "a::changed2", "calls"));

        // a.rs has 1 edge with c.rs
        store.add_edge(make_edge("c::weak_coupled1", "a::changed1", "calls"));

        let predictions = store.predictive_group_vectors("a.rs");

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], "b.rs"); // strongest coupling (3 edges)
        assert_eq!(predictions[1], "c.rs"); // weaker coupling (1 edge)
    }
}

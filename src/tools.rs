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

use crate::graph::status::{GraphBuildStatus, ScanActivity};
use crate::graph::store::GraphStore;
use crate::llm::{FunctionDefinition, ToolDefinition};
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
};
use tokio::{process::Command, time::Duration};
use walkdir::WalkDir;

/// MCP tools and servers allowed for the rest of a session.
#[derive(Default)]
struct McpGrants {
    tools: std::collections::BTreeSet<String>,
    servers: std::collections::BTreeSet<String>,
}

#[derive(Clone)]
pub struct ToolExecutor {
    workspace: PathBuf,
    http_client: reqwest::Client,
    tool_duration: Arc<Mutex<std::time::Duration>>,
    compression_enabled: bool,
    read_only: bool,
    /// Whether the subagent tools (`explore_repository`, `write_tests`,
    /// `write_docs`) are offered. Off inside a subagent, unless a custom
    /// agent's `Agent` tool lets it nest (see [`Self::nested_under`]).
    subagents: bool,
    /// The only tools offered and run, when set: a custom subagent's
    /// `tools` field (see [`crate::agents::custom`]).
    allowed_tools: Option<Arc<std::collections::HashSet<String>>>,
    /// Tools never offered or run, whatever else allows them: a custom
    /// subagent's `disallowedTools` field.
    denied_tools: Option<Arc<std::collections::HashSet<String>>>,
    /// The custom subagents behind `run_agent`; empty offers no such tool.
    custom_agents: Arc<crate::agents::custom::LiveAgents>,
    /// Where `web_search` searches (`[orangu].web_search`); empty is no web
    /// search. The main agent is offered `research_web` when there are any.
    web_engines: Arc<Vec<crate::web_search::Engine>>,
    /// A browsing subagent's pages: set, it is offered `browse`, and
    /// `web_search` when there are engines.
    browser: Option<Arc<tokio::sync::Mutex<crate::browse::Browser>>>,
    /// This conversation's SSH sessions: set, the main agent is offered
    /// `use_ssh`. `None` when `[orangu].ssh` is `deny`.
    ssh: Option<Arc<crate::ssh::SshSessions>>,
    /// Whether the login and copy tools themselves are offered: only to the
    /// subagent `use_ssh` starts, and to custom agents that name them.
    ssh_tools: bool,
    /// A custom agent's `MEMORY.md`, which `save_memory` rewrites. Outside
    /// the workspace for a user-scope memory, so it has its own tool rather
    /// than the workspace-confined file tools.
    memory_file: Option<PathBuf>,
    /// Every subagent run of this conversation (`/tasks`, `continue_agent`).
    tasks: Arc<crate::subagent_tasks::TaskRegistry>,
    /// Whether subagents may run in the background: only where a client
    /// delivers their reports later.
    background_tasks: bool,
    /// The connection the conversation using this executor is talking to,
    /// as of its latest request ([`crate::session::ChatSession::prompt`] sets
    /// it). Subagents start from it, so they follow `-c` and `/server`.
    active_profile: Arc<Mutex<Option<crate::config::LlmConfiguration>>>,
    /// The configured server sections, for a subagent whose role has a
    /// dedicated one.
    llms: Arc<std::collections::HashMap<String, crate::config::LlmConfiguration>>,
    context_cache: Arc<Mutex<crate::context::ContextCache>>,
    pub compression_metrics: Arc<Mutex<crate::compression::CompressionMetrics>>,
    auto_downsample_lines: usize,
    diff_file_cap: usize,
    pub session_dir: Option<PathBuf>,
    pub compression_store: Arc<crate::compression_cache::CompressionStore>,
    pub tool_counts: Arc<Mutex<std::collections::HashMap<String, usize>>>,
    /// Shared knowledge graph, populated by the startup scan.
    /// `None` while the background scan is still running.
    pub graph_store: Arc<Mutex<Option<GraphStore>>>,
    /// Hashes of the file versions currently represented by this executor's graph.
    pub graph_file_hashes: Arc<Mutex<std::collections::HashMap<String, String>>>,
    semantic_search: Arc<Mutex<Option<crate::embeddings::SemanticSearch>>>,
    /// The startup scan's build status, updated alongside `graph_store` —
    /// see `GraphBuildStatus`. Surfaced as the Graph dot in `/auto_review`'s
    /// status bar.
    pub graph_status: Arc<Mutex<GraphBuildStatus>>,
    /// Whether a workspace scan is running right now — see [`ScanActivity`].
    /// `/graph` waits on this rather than reporting a graph that isn't built
    /// yet, and scans the workspace itself when nothing else is scanning it.
    pub graph_scans: ScanActivity,
    /// Optional workspace-scoped MCP services. Read-only executors never
    /// attach one, so external tools cannot bypass their safety boundary.
    mcp: Option<Arc<crate::mcp::McpManager>>,
    /// MCP services a custom agent defines for itself (`mcpServers`), next
    /// to the shared ones.
    agent_mcp: Option<Arc<crate::mcp::McpManager>>,
    /// Whether nobody can approve a tool call here (a subagent): MCP tools
    /// that ask for approval are then not offered at all.
    mcp_unattended: bool,
    /// MCP tools and servers the user allowed for the rest of the session
    /// at an approval prompt: their calls no longer ask.
    mcp_grants: Arc<Mutex<McpGrants>>,
    /// The files the agent's turns changed, for `/undo` and `/rewind`.
    checkpoints: crate::checkpoints::Checkpoints,
    /// How many subagents deep this executor runs: 0 for the main
    /// conversation.
    depth: usize,
    /// How many levels of subagents may run below the main conversation
    /// (`[orangu].subagent_depth`).
    max_depth: usize,
    /// Whether a subagent call runs in the background unless it says
    /// otherwise (`[orangu].subagent_background`).
    background_default: bool,
    /// The agents `run_agent` may start here, when an agent's
    /// `Agent(a, b)` names them; `None` for any.
    startable_agents: Option<Arc<std::collections::HashSet<String>>>,
    /// The `hooks` of the agent this executor runs, around every call.
    agent_hooks: Option<Arc<crate::agents::agent_hooks::AgentHooks>>,
    /// Skills a subagent can load with `use_skill`.
    skills: Option<Arc<crate::skills::SkillRegistry>>,
    /// The licence generated files are written under — `/license`'s answer
    /// for this session, or [`crate::license::Choice::Auto`] to follow what
    /// the workspace declares.
    ///
    /// Shared rather than owned so `/license` can change it *while* the
    /// executor is in use: the next `create_file` reads it, and nothing has
    /// to be rebuilt for a choice made mid-conversation to take effect.
    pub licence: Arc<Mutex<crate::license::Choice>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ReadFileRequest {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
    start_page: Option<usize>,
    end_page: Option<usize>,
    query: Option<String>,
    /// Optional read mode:
    /// - `"full"` (default) — return the whole file (or the requested line range)
    /// - `"signatures"` — return only public item signatures (pub fn, pub struct,
    ///   pub enum, pub trait, impl blocks, doc comments), stripping function bodies
    /// - `"map"` — return a one-line-per-item structural overview (module-level
    ///   items only, no bodies or doc comments)
    mode: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EditFileRequest {
    path: String,
    old_text: String,
    new_text: String,
    replace_all: Option<bool>,
    /// `true` stages the change with `git add` when the workspace is a
    /// repository. Defaults to off, exactly as the `edits` form (and every
    /// other file tool) does: nothing is staged automatically.
    git: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ListDirectoryRequest {
    path: Option<String>,
    max_depth: Option<usize>,
    /// A glob (`**/*.rs`) over the paths under `path`: the matching files
    /// alone, newest first.
    pattern: Option<String>,
}

/// The most files a `list_directory` pattern lists.
const PATTERN_LIMIT: usize = 1000;

#[derive(Debug, Deserialize, Serialize)]
struct FetchUrlRequest {
    url: String,
    max_chars: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ShellCommandRequest {
    command: String,
    cwd: Option<String>,
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ShellCommandResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

impl ToolExecutor {
    pub fn new(workspace: &Path) -> Self {
        Self::with_config(workspace, true, 300, 20, None)
    }

    pub fn new_read_only(workspace: &Path) -> Self {
        let mut executor = Self::with_config(workspace, true, 300, 20, None);
        executor.read_only = true;
        executor.subagents = false;
        executor
    }

    /// A writable executor for a subagent: every file tool, but no
    /// subagent tools and no MCP services.
    pub fn new_for_subagent(workspace: &Path) -> Self {
        let mut executor = Self::with_config(workspace, true, 300, 20, None);
        executor.subagents = false;
        executor
    }

    pub fn with_config(
        workspace: &Path,
        compression_enabled: bool,
        auto_downsample_lines: usize,
        diff_file_cap: usize,
        session_dir: Option<PathBuf>,
    ) -> Self {
        let checkpoints = crate::checkpoints::Checkpoints::new(
            workspace,
            session_dir.as_ref().map(|dir| dir.join("checkpoints")),
        );
        Self {
            workspace: workspace.to_path_buf(),
            http_client: Client::new(),
            tool_duration: Arc::new(Mutex::new(std::time::Duration::ZERO)),
            compression_enabled,
            read_only: false,
            subagents: true,
            allowed_tools: None,
            denied_tools: None,
            custom_agents: Arc::default(),
            web_engines: Arc::default(),
            browser: None,
            ssh: None,
            ssh_tools: false,
            memory_file: None,
            tasks: Arc::default(),
            background_tasks: false,
            active_profile: Arc::default(),
            llms: Arc::default(),
            context_cache: Arc::new(Mutex::new(crate::context::ContextCache::new())),
            compression_metrics: Arc::new(Mutex::new(
                crate::compression::CompressionMetrics::default(),
            )),
            auto_downsample_lines,
            diff_file_cap,
            compression_store: Arc::new(crate::compression_cache::CompressionStore::new(
                session_dir.clone(),
            )),
            session_dir,
            tool_counts: Arc::new(Mutex::new(std::collections::HashMap::new())),
            graph_store: Arc::new(Mutex::new(None)),
            graph_file_hashes: Arc::new(Mutex::new(std::collections::HashMap::new())),
            semantic_search: Arc::new(Mutex::new(None)),
            graph_status: Arc::new(Mutex::new(GraphBuildStatus::default())),
            graph_scans: ScanActivity::default(),
            mcp: None,
            agent_mcp: None,
            mcp_unattended: false,
            mcp_grants: Arc::default(),
            checkpoints,
            agent_hooks: None,
            skills: None,
            depth: 0,
            max_depth: crate::config::default_subagent_depth(),
            background_default: false,
            startable_agents: None,
            licence: Arc::new(Mutex::new(crate::license::Choice::default())),
        }
    }

    /// Attach already-initialized MCP services for this workspace.
    pub fn with_mcp(mut self, mcp: Arc<crate::mcp::McpManager>) -> Self {
        self.mcp = Some(mcp);
        self
    }

    /// The MCP services this executor shares with its subagents.
    pub fn mcp(&self) -> Option<Arc<crate::mcp::McpManager>> {
        self.mcp.clone()
    }

    /// Gives a subagent its parent's MCP services, less the tools that ask
    /// for approval: nobody is there to approve them.
    pub fn with_subagent_mcp(mut self, mcp: Option<Arc<crate::mcp::McpManager>>) -> Self {
        self.mcp = mcp;
        self.mcp_unattended = true;
        self
    }

    /// Lets a subagent start subagents of its own, one level below `parent`:
    /// the agents in `startable`, or any. They share `parent`'s agents,
    /// server sections, web search and task registry, so `/tasks` lists them.
    pub fn nested_under(
        mut self,
        parent: &ToolExecutor,
        startable: Option<std::collections::HashSet<String>>,
    ) -> Self {
        self.subagents = true;
        self.depth = parent.depth + 1;
        self.max_depth = parent.max_depth;
        self.custom_agents = parent.custom_agents.clone();
        self.llms = parent.llms.clone();
        self.web_engines = parent.web_engines.clone();
        self.tasks = parent.tasks.clone();
        self.startable_agents = startable.map(Arc::new);
        self
    }

    /// Runs `hooks` around every tool call.
    pub fn with_agent_hooks(mut self, hooks: crate::agents::agent_hooks::AgentHooks) -> Self {
        self.agent_hooks = (!hooks.is_empty()).then(|| Arc::new(hooks));
        self
    }

    /// Runs subagents in the background unless a call says `background:
    /// false` (where background tasks are possible at all).
    pub fn with_background_default(mut self, background: bool) -> Self {
        self.background_default = background;
        self
    }

    pub fn background_default(&self) -> bool {
        self.background_default
    }

    /// The `background` argument of a subagent tool, with its default when
    /// that is not `false`.
    fn background_property(&self) -> Value {
        if self.background_default {
            json!({"type": "boolean", "default": true})
        } else {
            json!({"type": "boolean"})
        }
    }

    /// How many levels of subagents may run below the main conversation.
    pub fn with_subagent_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Limits `run_agent` to the agents in `names`.
    pub fn with_startable_agents(mut self, names: std::collections::HashSet<String>) -> Self {
        self.startable_agents = Some(Arc::new(names));
        self
    }

    /// How many subagents deep this executor runs: 0 for the main
    /// conversation.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Whether `run_agent` may start agent `name` here.
    pub fn may_start(&self, name: &str) -> bool {
        self.startable_agents
            .as_ref()
            .is_none_or(|names| names.iter().any(|n| n.eq_ignore_ascii_case(name)))
    }

    /// Offers `use_skill`, which loads one of `skills`, when there are any.
    pub fn with_skills(mut self, skills: Arc<crate::skills::SkillRegistry>) -> Self {
        self.skills = (!skills.all().is_empty()).then_some(skills);
        self
    }

    /// Adds the MCP services a custom agent defines itself.
    pub fn with_agent_mcp(mut self, mcp: Arc<crate::mcp::McpManager>) -> Self {
        self.agent_mcp = Some(mcp);
        self.mcp_unattended = true;
        self
    }

    /// The instructions of the MCP servers whose tools this executor offers.
    pub fn mcp_instructions(&self) -> String {
        let offered: std::collections::HashSet<String> = self
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect();
        self.mcp
            .iter()
            .chain(self.agent_mcp.iter())
            .filter(|mcp| {
                mcp.definitions()
                    .iter()
                    .any(|def| offered.contains(&def.function.name))
            })
            .map(|mcp| mcp.instructions())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// The MCP services that offer `name`, if any.
    fn mcp_for(&self, name: &str) -> Option<&Arc<crate::mcp::McpManager>> {
        self.mcp
            .iter()
            .chain(self.agent_mcp.iter())
            .find(|mcp| mcp.contains(name))
    }

    pub fn configure_semantic_search(
        &self,
        profile: Option<&crate::config::LlmConfiguration>,
        is_coordinator: bool,
        compile_workers: usize,
        budget_tokens: usize,
    ) -> Result<()> {
        let search = match profile {
            Some(profile) => Some(crate::embeddings::SemanticSearch::from_profile(
                profile,
                is_coordinator,
                compile_workers,
                budget_tokens,
            )?),
            None => Some(crate::embeddings::SemanticSearch::lexical_only(
                compile_workers,
                budget_tokens,
            )),
        };
        *self
            .semantic_search
            .lock()
            .map_err(|_| anyhow!("semantic search mutex poisoned"))? = search;
        Ok(())
    }

    /// This session's licence choice, or `Auto` if the lock is poisoned —
    /// a licence header is not worth failing a file write over.
    pub fn licence_choice(&self) -> crate::license::Choice {
        self.licence
            .lock()
            .map(|choice| choice.clone())
            .unwrap_or_default()
    }

    /// Pin the licence generated files are written under.
    pub fn set_licence_choice(&self, choice: crate::license::Choice) {
        if let Ok(mut current) = self.licence.lock() {
            *current = choice;
        }
    }

    /// The three licensing fields of a `create_file` request, as this
    /// session's choice sets them: whether to write a header at all, and —
    /// when the user named them — which licence and whose copyright.
    fn licence_fields(&self) -> (bool, Option<String>, Option<String>) {
        match self.licence_choice() {
            crate::license::Choice::Auto => (true, None, None),
            crate::license::Choice::None => (false, None, None),
            crate::license::Choice::Use { licence, holder } => {
                (true, Some(licence.spdx().to_string()), holder)
            }
        }
    }

    pub fn total_tool_duration(&self) -> std::time::Duration {
        self.tool_duration.lock().map(|d| *d).unwrap_or_default()
    }

    pub fn diff_file_cap(&self) -> usize {
        self.diff_file_cap
    }

    /// The tools offered to the model. Every definition is prefilled with
    /// every request's prompt — on a cold start ~1700 of a one-line
    /// question's ~2040 tokens were these — so
    /// a description says what the model cannot guess from the name and the
    /// parameters, and no more.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![tool(
            "show_file",
            "Read text/PDF. PDFs support page ranges, query keywords, page references and local OCR. \
                 Code mode: signatures/map.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "start_page": {"type": "integer", "minimum": 1},
                    "end_page": {"type": "integer", "minimum": 1},
                    "query": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["full", "signatures", "map"]
                    }
                },
                "required": ["path"]
            }),
        )];

        if !self.read_only {
            defs.push(tool(
                "create_file",
                "Write content; .pdf renders a searchable PDF. Replaces files unless overwrite=false. \
                     git=true stages it. New source files may get a licence header; read back before line edits.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "description": "octal, e.g. \"0644\""},
                        "overwrite": {"type": "boolean"},
                        "parents": {"type": "boolean"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["path"]
                }),
            ));
            defs.push(tool(
                "modify_file",
                "Edit a file: replace `old_text` with `new_text`, or apply `edits`, each \
                     replacing lines `start_line`..`end_line` (1-based, inclusive, of the \
                     file as it is now; no overlaps; `end_line = start_line - 1` inserts) \
                     with `replacement`. git=true stages it.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_line": {"type": "integer"},
                                    "end_line": {"type": "integer"},
                                    "replacement": {"type": "string"}
                                },
                                "required": ["start_line", "end_line"]
                            }
                        },
                        "git": {"type": "boolean"}
                    },
                    "required": ["path"]
                }),
            ));
            defs.push(tool(
                "move_file",
                "Move or rename a file; git=true uses `git mv` when tracked.",
                json!({
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "mode": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                        "parents": {"type": "boolean"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["from", "to"]
                }),
            ));
            defs.push(tool(
                "delete_file",
                "Delete a file (not a directory); git=true uses `git rm` when tracked.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["path"]
                }),
            ));
            defs.push(tool(
                "create_directory",
                "Create a directory; fails if the path exists.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "mode": {"type": "string"},
                        "parents": {"type": "boolean"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["path"]
                }),
            ));
            defs.push(tool(
                "move_directory",
                "Move a directory tree to a path that does not exist yet; git=true uses \
                     `git mv` when tracked.",
                json!({
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "mode": {"type": "string"},
                        "parents": {"type": "boolean"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["from", "to"]
                }),
            ));
            defs.push(tool(
                "delete_directory",
                "Delete an empty directory.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "git": {"type": "boolean"}
                    },
                    "required": ["path"]
                }),
            ));
            defs.push(tool(
                "convert_document",
                "Convert file `source`, or `content` in `format` (md/html/csv/outline/svg/latex), \
                     to `target`, typed by extension (odt/ods/odp/odg/docx/xlsx/pptx/pdf).",
                json!({
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "source": {"type": "string"},
                        "content": {"type": "string"},
                        "format": {"type": "string"},
                        "overwrite": {"type": "boolean"}
                    },
                    "required": ["target"]
                }),
            ));
            if self.subagents {
                defs.extend(
                    crate::subagents::SUBAGENTS
                        .iter()
                        .filter(|spec| {
                            !self.web_engines.is_empty()
                                || spec.tool != crate::subagents::WEB_RESEARCHER.tool
                        })
                        .filter(|spec| {
                            self.ssh.is_some() || spec.tool != crate::subagents::SSH.tool
                        })
                        .map(|spec| spec.definition()),
                );
                defs.push(crate::subagents::document_definition());
            }
        }

        defs.push(tool(
            "list_directory",
            "List files and dirs, or files matching glob `pattern`.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                    "pattern": {"type": "string"}
                }
            }),
        ));
        defs.push(tool(
            "fetch_url",
            "Fetch a URL as readable text.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"}
                },
                "required": ["url"]
            }),
        ));
        defs.push(tool(
            "run_shell_command",
            "Run a shell command in the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"}
                },
                "required": ["command"]
            }),
        ));
        defs.push(tool(
            "expand_context",
            "Retrieve the cached text behind an id in a truncation marker.",
            json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"]
            }),
        ));
        defs.push(tool(
            "graph_lookup",
            "Look up a symbol (name or part of it) in the code graph: its callers and \
             callees.",
            json!({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }),
        ));
        defs.push(tool(
            "graph_explain",
            "Explain one unambiguous workspace symbol from the Knowledge Graph. Returns its source, structural community, degree, and incoming/outgoing relationships with extracted or inferred evidence.",
            json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Exact symbol id or name; ambiguous names return candidate ids rather than guessing"
                    }
                },
                "required": ["symbol"]
            }),
        ));
        defs.push(tool(
            "graph_path",
            "Find the shortest relationship path between two unambiguous workspace symbols. Follows relationship direction by default; set undirected=true for architectural discovery across callers and callees.",
            json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "undirected": {"type": "boolean", "default": false},
                    "max_hops": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8}
                },
                "required": ["source", "target"]
            }),
        ));

        defs.push(tool(
            "semantic_search",
            "Search workspace code by meaning and exact tokens. Uses lexical ranking when no embeddings server is available.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string", "minLength": 1 } },
                "required": ["query"]
            }),
        ));
        for mcp in self.mcp.iter().chain(self.agent_mcp.iter()) {
            defs.extend(
                mcp.definitions().into_iter().filter(|def| {
                    !(self.mcp_unattended && mcp.requires_approval(&def.function.name))
                }),
            );
        }
        // Searching itself is a subagent's job: results and the pages read
        // to check them stay out of the main conversation.
        if self.browser.is_some() {
            defs.push(tool(
                "browse",
                "Open a web page by `url`, or link number `link` of the current page; long \
                     pages come in parts (`part`). Pages read stay cached.",
                json!({
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "link": {"type": "integer", "minimum": 1},
                        "part": {"type": "integer", "minimum": 1}
                    }
                }),
            ));
        }
        if self.browser.is_some() && !self.web_engines.is_empty() {
            defs.push(tool(
                "web_search",
                "Search the web; returns numbered results with title, URL and excerpt. \
                     Open one with browse.",
                json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }),
            ));
        }
        if self.ssh_tools && !self.read_only {
            defs.extend(ssh_definitions());
        }
        let agents = self.custom_agents.current();
        if !self.read_only && self.subagents && !agents.is_empty() {
            defs.push(tool(
                "run_agent",
                &format!(
                    "Have one of {} custom subagents do a self-contained `task`. \
                     agent=\"list\" with task as search words lists matching agents.",
                    agents.len()
                ),
                json!({
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "task": {"type": "string"}
                    },
                    "required": ["agent", "task"]
                }),
            ));
        }
        // Where subagents can run in the background (an interactive
        // conversation), every subagent tool takes `background`, and a
        // finished subagent can be continued.
        if self.background_tasks && self.subagents && !self.read_only {
            for def in &mut defs {
                if self.runs_subagent(&def.function.name)
                    && let Some(properties) = def
                        .function
                        .parameters
                        .get_mut("properties")
                        .and_then(Value::as_object_mut)
                {
                    properties.insert("background".to_string(), self.background_property());
                }
            }
            defs.push(tool(
                "continue_agent",
                "Give finished subagent task `task` (its id, e.g. t3) a follow-up `message`; it \
                     keeps its earlier work. background=true runs a subagent without waiting; \
                     its report arrives later as a message.",
                json!({
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "message": {"type": "string"},
                        "background": self.background_property()
                    },
                    "required": ["task", "message"]
                }),
            ));
        }
        if self.skills.is_some() {
            defs.push(tool(
                "use_skill",
                "Load a skill's instructions by `name`, with optional `arguments`.",
                json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "string"}
                    },
                    "required": ["name"]
                }),
            ));
        }
        if self.memory_file.is_some() {
            defs.push(tool(
                "save_memory",
                "Replace your persistent memory (MEMORY.md) with `content`: the whole new text.",
                json!({
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"]
                }),
            ));
        }
        // Offered only where an allowlist asks for it (a custom agent with
        // `Grep`): everyone else searches through `run_shell_command`.
        if let Some(allowed) = &self.allowed_tools {
            if allowed.contains("search_text") {
                defs.push(tool(
                    "search_text",
                    "Search file contents for a regular expression; respects .gitignore. \
                         Returns path:line: text.",
                    json!({
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "glob": {"type": "string"},
                            "ignore_case": {"type": "boolean"},
                            "max_results": {"type": "integer", "minimum": 1}
                        },
                        "required": ["pattern"]
                    }),
                ));
            }
            defs.retain(|def| crate::agents::custom::lists_tool(allowed, &def.function.name));
        }
        if let Some(denied) = &self.denied_tools {
            defs.retain(|def| !crate::agents::custom::lists_tool(denied, &def.function.name));
        }
        defs
    }

    /// Gives a custom agent its memory: `save_memory` rewrites `file`.
    pub fn with_memory_file(mut self, file: PathBuf) -> Self {
        self.memory_file = Some(file);
        self
    }

    /// Takes the tools in `denied` away, from what is offered and from what
    /// runs, whatever else allows them.
    pub fn with_denied_tools(mut self, denied: std::collections::HashSet<String>) -> Self {
        self.denied_tools = Some(Arc::new(denied));
        self
    }

    /// Restricts this executor to the tools named in `allowed`, both in what
    /// it offers and in what it runs: a custom subagent's `tools` field.
    pub fn with_allowed_tools(mut self, allowed: std::collections::HashSet<String>) -> Self {
        self.allowed_tools = Some(Arc::new(allowed));
        self
    }

    /// The SearXNG instance web searches go to, or `None` for no web search.
    pub fn with_web_search(mut self, engines: Vec<crate::web_search::Engine>) -> Self {
        self.web_engines = Arc::new(engines);
        self
    }

    /// Lets this executor browse: `browse`, and `web_search` with engines.
    /// For a subagent only; its pages stay cached while it runs.
    pub fn with_browsing(mut self) -> Self {
        self.browser = Some(Arc::default());
        self
    }

    /// This conversation's SSH sessions, which offer `use_ssh`.
    pub fn with_ssh(mut self, ssh: Option<Arc<crate::ssh::SshSessions>>) -> Self {
        self.ssh = ssh;
        self
    }

    /// Offers the login and copy tools, on the parent's SSH sessions.
    pub fn with_ssh_tools(mut self, ssh: Option<Arc<crate::ssh::SshSessions>>) -> Self {
        self.ssh_tools = ssh.is_some();
        self.ssh = ssh;
        self
    }

    pub fn ssh(&self) -> Option<Arc<crate::ssh::SshSessions>> {
        self.ssh.clone()
    }

    pub fn web_engines(&self) -> &[crate::web_search::Engine] {
        &self.web_engines
    }

    /// The configured server sections subagents may pick a role's from.
    pub fn with_llms(
        mut self,
        llms: std::collections::HashMap<String, crate::config::LlmConfiguration>,
    ) -> Self {
        self.llms = Arc::new(llms);
        self
    }

    pub fn llms(&self) -> &std::collections::HashMap<String, crate::config::LlmConfiguration> {
        &self.llms
    }

    /// Whether tool `name` starts a subagent: such calls in one turn are
    /// independent of each other, so the chat loop runs them concurrently.
    pub fn runs_subagent(&self, name: &str) -> bool {
        name == crate::subagents::DOCUMENT_TOOL
            || name == "run_agent"
            || name == "continue_agent"
            || crate::subagents::find(name).is_some()
    }

    /// Lets this conversation's subagents run in the background: an
    /// interactive client, which hands their reports to the model later.
    pub fn with_background_tasks(mut self) -> Self {
        self.background_tasks = true;
        self
    }

    pub fn background_tasks(&self) -> bool {
        self.background_tasks
    }

    /// This conversation's subagent tasks.
    pub fn tasks(&self) -> &Arc<crate::subagent_tasks::TaskRegistry> {
        &self.tasks
    }

    /// Records the connection the conversation is using now.
    pub fn set_active_profile(&self, profile: &crate::config::LlmConfiguration) {
        if let Ok(mut active) = self.active_profile.lock() {
            *active = Some(profile.clone());
        }
    }

    /// The connection the conversation used for its latest request, if any.
    pub fn active_profile(&self) -> Option<crate::config::LlmConfiguration> {
        self.active_profile
            .lock()
            .ok()
            .and_then(|active| active.clone())
    }

    /// The custom subagents `run_agent` offers: a fixed set.
    pub fn with_custom_agents(mut self, agents: crate::agents::custom::AgentRegistry) -> Self {
        self.custom_agents = Arc::new(crate::agents::custom::LiveAgents::fixed(agents));
        self
    }

    /// The custom subagents of the workspace, read again when their files
    /// change.
    pub fn with_workspace_agents(mut self) -> Self {
        self.custom_agents = Arc::new(crate::agents::custom::LiveAgents::watch(&self.workspace));
        self
    }

    pub fn custom_agents(&self) -> Arc<crate::agents::custom::AgentRegistry> {
        self.custom_agents.current()
    }

    /// The custom subagents, looked for again now.
    pub fn refresh_custom_agents(&self) -> Arc<crate::agents::custom::AgentRegistry> {
        self.custom_agents.refresh()
    }

    /// The picture at `path` in the workspace — its media type and bytes —
    /// for a caller that can hand a model the picture itself; `None` when
    /// the file is not one ([`picture_type`]).
    pub fn read_picture(&self, path: &str) -> Result<Option<(&'static str, Vec<u8>)>> {
        let bytes = fs::read(self.resolve_workspace_path(path)?)?;
        Ok(picture_type(&bytes).map(|kind| (kind, bytes)))
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Human-readable lifecycle state for configured MCP services.
    pub fn mcp_status(&self) -> String {
        self.mcp
            .as_ref()
            .map(|mcp| mcp.status())
            .unwrap_or_else(|| "No MCP servers configured.".to_string())
    }

    /// Replace the live MCP services after a config change or explicit refresh.
    pub fn refresh_mcp(
        &self,
        configurations: &std::collections::HashMap<String, crate::config::McpServerConfiguration>,
    ) {
        if let Some(mcp) = &self.mcp {
            mcp.refresh_in_background(configurations.clone());
        }
    }

    pub fn requires_mcp_approval(&self, name: &str) -> bool {
        self.mcp_for(name).is_some_and(|mcp| {
            mcp.requires_approval(name)
                && !self.mcp_grants.lock().is_ok_and(|grants| {
                    grants.tools.contains(name)
                        || mcp
                            .route_names(name)
                            .is_some_and(|(server, _)| grants.servers.contains(&server))
                })
        })
    }

    /// The MCP server behind the exposed tool `name`, and the server's own
    /// name for the tool.
    pub fn mcp_route(&self, name: &str) -> Option<(String, String)> {
        self.mcp_for(name).and_then(|mcp| mcp.route_names(name))
    }

    /// Lets every later call of the MCP tool `name` run without asking.
    pub fn grant_mcp_tool(&self, name: &str) {
        if let Ok(mut grants) = self.mcp_grants.lock() {
            grants.tools.insert(name.to_string());
        }
    }

    /// Lets every later call of the MCP server `server`'s tools run without
    /// asking.
    pub fn grant_mcp_server(&self, server: &str) {
        if let Ok(mut grants) = self.mcp_grants.lock() {
            grants.servers.insert(server.to_string());
        }
    }

    /// The tools and the servers granted for this session, each sorted.
    pub fn mcp_grants(&self) -> (Vec<String>, Vec<String>) {
        self.mcp_grants
            .lock()
            .map(|grants| {
                (
                    grants.tools.iter().cloned().collect(),
                    grants.servers.iter().cloned().collect(),
                )
            })
            .unwrap_or_default()
    }

    /// Withdraws the grant for a tool or a server named `name`; `false` when
    /// there was none.
    pub fn revoke_mcp_grant(&self, name: &str) -> bool {
        self.mcp_grants.lock().is_ok_and(|mut grants| {
            let tool = grants.tools.remove(name);
            let server = grants.servers.remove(name);
            tool || server
        })
    }

    /// Withdraws every grant; returns how many there were.
    pub fn clear_mcp_grants(&self) -> usize {
        self.mcp_grants
            .lock()
            .map(|mut grants| {
                let count = grants.tools.len() + grants.servers.len();
                grants.tools.clear();
                grants.servers.clear();
                count
            })
            .unwrap_or_default()
    }

    /// The checkpoints of the files this session's turns changed.
    pub fn checkpoints(&self) -> &crate::checkpoints::Checkpoints {
        &self.checkpoints
    }

    /// Records a subagent's writes in its parent's checkpoints, so `/undo`
    /// covers them too.
    pub fn with_checkpoints(mut self, checkpoints: crate::checkpoints::Checkpoints) -> Self {
        self.checkpoints = checkpoints;
        self
    }

    /// Expose the session-local file context cache for persistence across runs.
    pub fn context_cache(&self) -> &std::sync::Mutex<crate::context::ContextCache> {
        &self.context_cache
    }

    pub async fn execute(&self, name: &str, arguments: &Map<String, Value>) -> Result<String> {
        if self
            .allowed_tools
            .as_ref()
            .is_some_and(|allowed| !crate::agents::custom::lists_tool(allowed, name))
            || self
                .denied_tools
                .as_ref()
                .is_some_and(|denied| crate::agents::custom::lists_tool(denied, name))
        {
            return Err(anyhow!("tool '{name}' is not available to this agent"));
        }
        if let Some(hooks) = &self.agent_hooks
            && let Err(reason) = hooks.before(name, arguments, &self.workspace).await
        {
            return Err(anyhow!("blocked by a PreToolUse hook: {reason}"));
        }
        let start = std::time::Instant::now();
        let result = match name {
            "show_file" => self.show_file(arguments).await,
            "modify_file" => self.mutating(arguments, || self.modify_file(arguments)),
            "create_file" => self.mutating(arguments, || self.create_file(arguments)),
            "move_file" => self.mutating(arguments, || {
                self.file_operation(crate::files::move_, arguments)
            }),
            "delete_file" => self.mutating(arguments, || {
                self.file_operation(crate::files::delete, arguments)
            }),
            "create_directory" => self.mutating(arguments, || {
                self.file_operation(crate::files::create_dir, arguments)
            }),
            "move_directory" => self.mutating(arguments, || {
                self.file_operation(crate::files::move_dir, arguments)
            }),
            "delete_directory" => self.mutating(arguments, || {
                self.file_operation(crate::files::delete_dir, arguments)
            }),
            _ if name == crate::subagents::DOCUMENT_TOOL
                || crate::subagents::find(name).is_some() =>
            {
                if self.read_only {
                    Err(anyhow::anyhow!("tool not available in read-only mode"))
                } else if !self.subagents {
                    Err(anyhow::anyhow!("subagents cannot start subagents"))
                } else if let Some(spec) = crate::subagents::find(name) {
                    crate::subagents::run(spec, self, arguments).await
                } else {
                    crate::subagents::run_document(self, arguments).await
                }
            }
            "convert_document" => self.convert_document(arguments).await,
            "search_text" => self.search_text(arguments),
            "save_memory" => self.save_memory(arguments),
            "web_search" => self.web_search(arguments).await,
            "browse" => self.browse(arguments).await,
            "run_agent" => {
                if self.read_only {
                    Err(anyhow!("tool not available in read-only mode"))
                } else if !self.subagents {
                    Err(anyhow!("subagents cannot start subagents"))
                } else {
                    crate::subagents::run_custom(self, arguments).await
                }
            }
            "continue_agent" => {
                if self.read_only {
                    Err(anyhow!("tool not available in read-only mode"))
                } else if !self.subagents {
                    Err(anyhow!("subagents cannot start subagents"))
                } else {
                    crate::subagents::continue_task(self, arguments).await
                }
            }
            "use_skill" => self.use_skill(arguments),
            "ssh_login" | "ssh_copy" | "ssh_logout" => self.ssh_tool(name, arguments).await,
            "list_directory" => self.list_directory(arguments).await,
            "fetch_url" => self.fetch_url(arguments).await,
            "run_shell_command" => self.run_shell_command(arguments).await,
            "expand_context" => self.expand_context(arguments).await,
            "graph_lookup" => self.graph_lookup(arguments),
            "graph_explain" => self.graph_explain(arguments),
            "graph_path" => self.graph_path(arguments),
            "semantic_search" => self.semantic_search(arguments).await,
            _ if let Some(mcp) = self.mcp_for(name) => {
                if self.mcp_unattended && mcp.requires_approval(name) {
                    Err(anyhow!(
                        "MCP tool '{name}' needs approval, which a subagent cannot ask for"
                    ))
                } else {
                    mcp.execute(name, arguments).await
                }
            }
            _ => Err(anyhow!("unknown tool '{}'", name)),
        };
        if result.is_ok()
            && let Ok(mut counts) = self.tool_counts.lock()
        {
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }
        if let Ok(mut d) = self.tool_duration.lock() {
            *d += start.elapsed();
        }
        match (&self.agent_hooks, result) {
            (Some(hooks), Ok(output)) => {
                match hooks.after(name, arguments, &output, &self.workspace).await {
                    Some(feedback) => Ok(format!("{output}\n\n[PostToolUse hook]\n{feedback}")),
                    None => Ok(output),
                }
            }
            (_, result) => result,
        }
    }

    async fn expand_context(&self, arguments: &Map<String, Value>) -> Result<String> {
        let id = arguments
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'id' argument"))?;
        match self.compression_store.retrieve(id) {
            Ok(content) => Ok(content),
            Err(e) => Err(anyhow!("Error retrieving content: {}", e)),
        }
    }

    /// Blocks until [`Self::graph_store`] holds a graph, so a caller can work
    /// with the Knowledge Graph instead of being told it isn't ready.
    ///
    /// A scan that is already running (the startup scan `orangu` kicks off)
    /// is waited out: it runs on its own thread and publishes the store
    /// itself, so waiting here cannot stall it. Once nothing is scanning the
    /// store is final — still empty means nobody ever scanned this workspace
    /// (a `-p` one-shot starts no background scan), so this scans it here.
    /// The store lock is held across that scan, which keeps two callers from
    /// scanning the same workspace at once: the second blocks, then finds the
    /// graph the first published.
    pub fn ensure_graph(&self) -> Result<()> {
        while self
            .graph_store
            .lock()
            .map_err(|_| anyhow!("graph_store mutex poisoned"))?
            .is_none()
            && self.graph_scans.is_scanning()
        {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let mut guard = self
            .graph_store
            .lock()
            .map_err(|_| anyhow!("graph_store mutex poisoned"))?;
        if guard.is_some() {
            return Ok(());
        }
        {
            let _scan = self.graph_scans.begin();
            let result = crate::agents::hooks::run_session_start_hook(&self.workspace);
            *self
                .graph_file_hashes
                .lock()
                .map_err(|_| anyhow!("graph hashes mutex poisoned"))? = result.file_hashes;
            *guard = Some(result.store);
        }
        drop(guard);
        if let Ok(mut status) = self.graph_status.lock() {
            *status = GraphBuildStatus::Ready;
        }
        Ok(())
    }

    async fn semantic_search(&self, arguments: &Map<String, Value>) -> Result<String> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .filter(|query| !query.trim().is_empty())
            .ok_or_else(|| anyhow!("missing or empty 'query' argument"))?;
        let search = self.semantic_search.lock()
            .map_err(|_| anyhow!("semantic search mutex poisoned"))?
            .clone()
            .ok_or_else(|| anyhow!("Semantic search is unavailable: no embeddings-capable server was detected. Use graph_lookup or run_shell_command with rg."))?;
        // Extraction is synchronous. Run it off the async executor, including
        // the build's async requests, just as the interactive command does.
        let workspace = self.workspace.clone();
        let graph = self.graph_store.clone();
        let query = query.to_owned();
        tokio::task::spawn_blocking(move || {
            use std::sync::atomic::{AtomicBool, AtomicU64};
            let result = tokio::runtime::Handle::current().block_on(search.search(
                &workspace,
                &query,
                &graph,
                &AtomicBool::new(false),
                &AtomicU64::new(0),
                &AtomicU64::new(0),
            ))?;
            Ok(crate::embeddings::format_hits(&query, &result.hits))
        })
        .await?
    }

    fn graph_lookup(&self, arguments: &Map<String, Value>) -> Result<String> {
        let symbol = arguments
            .get("symbol")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'symbol' argument"))?;

        // A lookup waits for the workspace scan rather than answering "not
        // built yet": a moment's wait beats an answer the model has to retry,
        // and a retry it never makes reads as "this symbol does not exist".
        self.ensure_graph()?;

        let guard = self
            .graph_store
            .lock()
            .map_err(|_| anyhow!("graph_store mutex poisoned"))?;

        match &*guard {
            None => Ok(format!(
                "[graph_lookup] No Knowledge Graph could be built for this \
                 workspace.\n(Searched for: \"{}\")",
                symbol
            )),
            Some(store) => {
                let results = store.lookup(symbol);
                if results.is_empty() {
                    Ok(format!(
                        "[graph_lookup] No symbol matching \"{}\" found in the Knowledge Graph.\n\
                         Tip: try a shorter partial name (e.g. \"session\" instead of \"ChatSession\").",
                        symbol
                    ))
                } else {
                    Ok(results
                        .iter()
                        .map(|r| r.format())
                        .collect::<Vec<_>>()
                        .join("\n---\n"))
                }
            }
        }
    }

    fn graph_explain(&self, arguments: &Map<String, Value>) -> Result<String> {
        let symbol = arguments
            .get("symbol")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'symbol' argument"))?;
        self.ensure_graph()?;
        let guard = self
            .graph_store
            .lock()
            .map_err(|_| anyhow!("graph_store mutex poisoned"))?;
        match &*guard {
            None => Ok(
                "[graph_explain] No Knowledge Graph could be built for this workspace.".to_string(),
            ),
            Some(store) => store
                .explain(symbol)
                .map(|explanation| explanation.format())
                .map_err(anyhow::Error::msg),
        }
    }

    fn graph_path(&self, arguments: &Map<String, Value>) -> Result<String> {
        let source = arguments
            .get("source")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'source' argument"))?;
        let target = arguments
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'target' argument"))?;
        let undirected = arguments
            .get("undirected")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let max_hops = arguments
            .get("max_hops")
            .and_then(Value::as_u64)
            .unwrap_or(8);
        if !(1..=64).contains(&max_hops) {
            return Err(anyhow!("'max_hops' must be between 1 and 64"));
        }
        self.ensure_graph()?;
        let guard = self
            .graph_store
            .lock()
            .map_err(|_| anyhow!("graph_store mutex poisoned"))?;
        match &*guard {
            None => Ok(
                "[graph_path] No Knowledge Graph could be built for this workspace.".to_string(),
            ),
            Some(store) => store
                .shortest_path(source, target, undirected, max_hops as usize)
                .map(|path| path.format())
                .map_err(anyhow::Error::msg),
        }
    }

    /// Guard for every tool that changes the workspace: refused outright in
    /// read-only mode (`/review`, the explorer subagent), the same rule the
    /// old `edit_file` had before these tools replaced it.
    fn mutating<T>(
        &self,
        arguments: &Map<String, Value>,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if self.read_only {
            return Err(anyhow!("tool not available in read-only mode"));
        }
        self.checkpoint(arguments);
        operation()
    }

    /// Saves what a file tool called with `arguments` is about to change:
    /// its `path` or `target`, or both ends of a `from`/`to` move.
    fn checkpoint(&self, arguments: &Map<String, Value>) {
        let resolve = |key: &str| {
            arguments
                .get(key)
                .and_then(Value::as_str)
                .and_then(|path| self.resolve_workspace_path(path).ok())
        };
        if let (Some(from), Some(to)) = (resolve("from"), resolve("to")) {
            self.checkpoints.record_move(&from, &to);
        }
        let paths: Vec<PathBuf> = ["path", "target"].into_iter().filter_map(resolve).collect();
        if !paths.is_empty() {
            self.checkpoints.record(&paths);
        }
    }

    /// Run one [`crate::files`] operation: the same function, request shape
    /// and JSON result `orangu-server`'s endpoint of the same name serves,
    /// against this executor's own workspace. Keeping the two on one
    /// implementation is what stops "create a file" meaning two different
    /// things depending on which surface asked.
    fn file_operation<Q, R>(
        &self,
        operation: fn(&Path, Q) -> crate::files::FileResult<R>,
        arguments: &Map<String, Value>,
    ) -> Result<String>
    where
        Q: serde::de::DeserializeOwned,
        R: Serialize,
    {
        let request: Q = serde_json::from_value(Value::Object(arguments.clone()))
            .map_err(|err| anyhow!("{err}"))?;
        let response = operation(&self.workspace, request).map_err(|err| anyhow!("{err}"))?;
        Ok(serde_json::to_string(&response)?)
    }

    /// `web_search`: results for `query` from the first engine that has
    /// any, local SearXNG instances first.
    async fn web_search(&self, arguments: &Map<String, Value>) -> Result<String> {
        if self.web_engines.is_empty() {
            return Err(anyhow!("web search is off ([orangu].web_search)"));
        }
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing or invalid 'query' argument"))?;
        let (engine, results) =
            crate::web_search::search(&crate::web_search::client(), &self.web_engines, query)
                .await?;
        Ok(format!(
            "{}\n(from {engine})\n",
            crate::web_search::format_results(query.trim(), &results).trim_end()
        ))
    }

    /// `ssh_login`, `ssh_copy` and `ssh_logout`.
    async fn ssh_tool(&self, name: &str, arguments: &Map<String, Value>) -> Result<String> {
        let ssh = self
            .ssh
            .as_ref()
            .filter(|_| self.ssh_tools)
            .ok_or_else(|| anyhow!("{name} is only available to the use_ssh subagent"))?;
        if self.read_only {
            return Err(anyhow!("tool not available in read-only mode"));
        }
        let text = |key: &str| {
            arguments
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing or invalid '{key}' argument"))
        };
        match name {
            "ssh_login" => ssh.login(text("host")?).await,
            "ssh_logout" => ssh.logout(text("session")?).await,
            _ => {
                ssh.copy(
                    text("session")?,
                    text("direction")?.parse()?,
                    &self.workspace,
                    text("local")?,
                    text("remote")?,
                    arguments
                        .get("overwrite")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                )
                .await
            }
        }
    }

    /// `browse`: a page by URL or by link number, or another part of the
    /// current one, from the cache when it has been read before.
    async fn browse(&self, arguments: &Map<String, Value>) -> Result<String> {
        let browser = self
            .browser
            .as_ref()
            .ok_or_else(|| anyhow!("browse is only available to web subagents"))?;
        let number = |key: &str| -> Result<Option<usize>> {
            match arguments.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => value
                    .as_u64()
                    .map(|n| Some(n as usize))
                    .ok_or_else(|| anyhow!("'{key}' must be a positive number")),
            }
        };
        let url = arguments.get("url").and_then(Value::as_str);
        let (link, part) = (number("link")?, number("part")?);
        browser
            .lock()
            .await
            .browse(&crate::web_search::client(), url, link, part)
            .await
    }

    /// `save_memory`: replaces the agent's `MEMORY.md`, written to a
    /// temporary file and renamed into place so a reader never sees half.
    fn use_skill(&self, arguments: &Map<String, Value>) -> Result<String> {
        let skills = self
            .skills
            .as_ref()
            .ok_or_else(|| anyhow!("no skills are available here"))?;
        let name = arguments
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid 'name' argument"))?;
        let skill = skills.find(name).ok_or_else(|| {
            let names: Vec<&str> = skills.all().iter().map(|s| s.name.as_str()).collect();
            anyhow!("unknown skill '{name}'; available: {}", names.join(", "))
        })?;
        let arguments = arguments
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(skill.render_activation(arguments))
    }

    fn save_memory(&self, arguments: &Map<String, Value>) -> Result<String> {
        const MAX_BYTES: usize = 64 * 1024;
        let file = self
            .memory_file
            .as_ref()
            .ok_or_else(|| anyhow!("this agent has no memory"))?;
        let content = arguments
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing or invalid 'content' argument"))?;
        if content.len() > MAX_BYTES {
            return Err(anyhow!(
                "memory is limited to {} KB; keep it short",
                MAX_BYTES / 1024
            ));
        }
        let dir = file
            .parent()
            .ok_or_else(|| anyhow!("memory file has no directory"))?;
        fs::create_dir_all(dir)?;
        let mut text = content.trim_end().to_string();
        text.push('\n');
        let temporary = dir.join(".MEMORY.md.tmp");
        fs::write(&temporary, &text)?;
        fs::rename(&temporary, file)?;
        Ok(format!("Saved {} bytes to {}", text.len(), file.display()))
    }

    /// `search_text`: the lines matching a regular expression, below `path`
    /// (the workspace root by default) in files `.gitignore` does not
    /// exclude and `glob` (matched against the workspace-relative path)
    /// selects, as `path:line: text`.
    fn search_text(&self, arguments: &Map<String, Value>) -> Result<String> {
        const DEFAULT_RESULTS: usize = 100;
        const MAX_RESULTS: usize = 500;
        const MAX_LINE: usize = 300;

        let pattern = arguments
            .get("pattern")
            .and_then(Value::as_str)
            .filter(|pattern| !pattern.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid 'pattern' argument"))?;
        let ignore_case = arguments
            .get("ignore_case")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let regex = regex::RegexBuilder::new(pattern)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|err| anyhow!("invalid pattern: {err}"))?;
        let glob = arguments
            .get("glob")
            .and_then(Value::as_str)
            .map(|glob| {
                globset::Glob::new(glob)
                    .map(|glob| glob.compile_matcher())
                    .map_err(|err| anyhow!("invalid glob: {err}"))
            })
            .transpose()?;
        let limit = arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_RESULTS, |n| n as usize)
            .clamp(1, MAX_RESULTS);
        let root = match arguments.get("path").and_then(Value::as_str) {
            Some(path) => self.resolve_workspace_path(path)?,
            None => self.workspace.clone(),
        };

        let mut out = String::new();
        let mut matches = 0;
        'files: for entry in ignore::WalkBuilder::new(&root).build().flatten() {
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let path = entry.path();
            let relative = path.strip_prefix(&self.workspace).unwrap_or(path);
            if glob.as_ref().is_some_and(|glob| !glob.is_match(relative)) {
                continue;
            }
            let Ok(content) = fs::read_to_string(path) else {
                continue; // binary or unreadable
            };
            for (index, line) in content.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                if matches == limit {
                    out.push_str(&format!("[stopped at {limit} matches]\n"));
                    break 'files;
                }
                matches += 1;
                let line = match line.char_indices().nth(MAX_LINE) {
                    Some((cut, _)) => format!("{}…", &line[..cut]),
                    None => line.to_string(),
                };
                out.push_str(&format!(
                    "{}:{}: {}\n",
                    crate::files::to_slash(relative),
                    index + 1,
                    line
                ));
            }
        }
        if matches == 0 {
            out.push_str("no matches\n");
        }
        Ok(out)
    }

    /// `convert_document`: render or convert a LibreOffice document (see
    /// [`crate::office`]). Both paths are confined to the workspace, and the
    /// CPU-bound conversion runs off the async runtime, where a panic in the
    /// converter becomes an error rather than taking the session down.
    async fn convert_document(&self, arguments: &Map<String, Value>) -> Result<String> {
        if self.read_only {
            return Err(anyhow!("tool not available in read-only mode"));
        }
        self.checkpoint(arguments);
        let request: crate::office::ConvertRequest =
            serde_json::from_value(Value::Object(arguments.clone()))
                .map_err(|err| anyhow!("{err}"))?;
        let workspace = self.workspace.clone();
        let response = tokio::task::spawn_blocking(move || {
            crate::office::convert_in_workspace(&workspace, &request)
        })
        .await
        .map_err(|err| anyhow!("document conversion failed: {err}"))??;
        Ok(serde_json::to_string(&response)?)
    }

    /// `create_file`, with this session's licence choice laid over what the
    /// model sent.
    ///
    /// Its own arm rather than [`Self::file_operation`]'s generic one because
    /// the licensing fields are the *caller's* to set and are deliberately
    /// absent from the tool schema — a model must not be able to relicense
    /// somebody's project by writing a field.
    fn create_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        let mut request: crate::files::CreateFileRequest =
            serde_json::from_value(Value::Object(arguments.clone()))
                .map_err(|err| anyhow!("{err}"))?;
        let (license, license_id, license_holder) = self.licence_fields();
        request.license = license;
        request.license_id = license_id;
        request.license_holder = license_holder;
        let response =
            crate::files::create(&self.workspace, request).map_err(|err| anyhow!("{err}"))?;
        Ok(serde_json::to_string(&response)?)
    }

    /// `modify_file` takes either shape: `edits` (the line ranges
    /// `orangu-server`'s own `/v1/modify_file` takes) or the
    /// `old_text`/`new_text` replacement this tool has always accepted,
    /// which stays because it is what a model reaches for when it knows the
    /// text but not the line numbers. `edits` wins if both are given.
    fn modify_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        if arguments.contains_key("edits") {
            return self.file_operation(crate::files::modify, arguments);
        }
        self.replace_text(arguments)
    }

    async fn show_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ReadFileRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let path = self.resolve_workspace_path(&args.path)?;
        let read_path = path.clone();
        let pdf_options = crate::pdf::PdfReadOptions {
            start_page: args.start_page,
            end_page: args.end_page,
            query: args.query.clone(),
        };
        let read = tokio::task::spawn_blocking(move || -> Result<_> {
            let bytes = fs::read(&read_path)?;
            if let Some(kind) = picture_type(&bytes) {
                return Ok(Err((kind, bytes.len())));
            }
            let is_pdf = crate::pdf::is_pdf(&read_path, &bytes);
            let content = if is_pdf {
                crate::pdf::extract_document(&bytes, &pdf_options)?.context_text()?
            } else {
                if pdf_options.start_page.is_some()
                    || pdf_options.end_page.is_some()
                    || pdf_options.query.is_some()
                {
                    return Err(anyhow!(
                        "Page ranges and query are available only for PDF files"
                    ));
                }
                crate::pdf::decode_text(&read_path, &bytes)?
            };
            Ok(Ok((content, is_pdf)))
        })
        .await??;
        let (mut content, is_pdf) = match read {
            Ok(text) => text,
            Err((kind, len)) => {
                return Ok(format!(
                    "{} is a picture ({kind}, {len} bytes), not text: it has no lines to show",
                    args.path
                ));
            }
        };

        redact_secrets(&mut content);

        if self.compression_enabled
            && args.start_line.is_none()
            && args.end_line.is_none()
            && args.start_page.is_none()
            && args.end_page.is_none()
            && args.query.is_none()
            && let Ok(metadata) = fs::metadata(&path)
        {
            let mut cache = self.context_cache.lock().unwrap();
            let fingerprint = cache.fingerprint(&content, &metadata);
            let cache_result = cache.check_file(&path, &content, &fingerprint);
            if let crate::context::CacheResult::Hit { fingerprint } = cache_result {
                return Ok(crate::context::format_cache_stub(
                    &args.path,
                    metadata.len(),
                    &fingerprint,
                ));
            }
            cache.record_read(&path, fingerprint);
        }

        if self.compression_enabled
            && !is_pdf
            && args.mode.is_none()
            && args.start_line.is_none()
            && args.end_line.is_none()
            && self.auto_downsample_lines > 0
            && content.lines().count() > self.auto_downsample_lines
        {
            let mut downsampled = extract_signatures(&content);
            if downsampled != content {
                downsampled.push_str(&format!("\n[Note: This file exceeds {} lines and has been automatically downsampled to 'signatures' mode. Use start_line and end_line bounds to read specific full bodies.]\n", self.auto_downsample_lines));
                return Ok(downsampled);
            }
        }

        if is_pdf {
            return Ok(render_pdf_slice(&content, args.start_line, args.end_line));
        }
        Ok(match args.mode.as_deref() {
            Some("signatures") => extract_signatures(&content),
            Some("map") => extract_map(&content),
            _ => render_file_slice(&content, args.start_line, args.end_line),
        })
    }

    /// `modify_file`'s `old_text`/`new_text` form: compute the new content
    /// here, then write it through [`crate::files::create`] so the write
    /// itself is the same one every other surface performs — same workspace
    /// confinement, same `git add`, same reported shape.
    fn replace_text(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: EditFileRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let path = self.resolve_workspace_path(&args.path)?;
        if crate::pdf::is_pdf_path(&path) {
            return Err(anyhow!(
                "PDFs do not support text replacement; use create_file to save a new PDF from text"
            ));
        }
        let (original, created) = match fs::read_to_string(&path) {
            Ok(content) => (content, false),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => (String::new(), true),
            Err(err) => return Err(err.into()),
        };
        if original.starts_with("%PDF-") {
            return Err(anyhow!(
                "PDFs do not support text replacement; use create_file to save a new PDF from text"
            ));
        }

        let updated = apply_edit(
            &original,
            &args.old_text,
            &args.new_text,
            args.replace_all.unwrap_or(false),
            created,
        )?;

        let (license, license_id, license_holder) = self.licence_fields();
        let response = crate::files::create(
            &self.workspace,
            crate::files::CreateFileRequest {
                path: args.path.clone(),
                content: updated.clone(),
                // A brand-new file gets 0644, matching what this tool has
                // always created; an existing one keeps its own bits.
                // Permission bits are a Unix concept and `files::set_mode`
                // refuses them anywhere else by design, so off Unix this
                // stays `None` — asking for a mode there would fail the
                // whole write rather than just the part that cannot apply.
                mode: (created && cfg!(unix)).then_some(crate::files::Mode::Bits(0o644)),
                overwrite: true,
                parents: true,
                git: args.git.unwrap_or_else(crate::files::git_default),
                // Generated, so a file this brings into existence is
                // licensed. `files::create` is what decides that only a
                // *new* file is — editing one that was already there must
                // not stamp a licence onto it.
                license,
                license_id,
                license_holder,
            },
        )
        .map_err(|err| anyhow!("{err}"))?;

        let replacements = if created || args.old_text.is_empty() {
            0
        } else if args.replace_all.unwrap_or(false) {
            original.matches(args.old_text.as_str()).count()
        } else {
            1
        };
        Ok(json!({
            "path": response.path,
            "created": created,
            "updated": true,
            "replacements": replacements,
            "original_bytes": original.len(),
            "new_bytes": updated.len(),
            "mode": response.mode,
            "git": response.git,
        })
        .to_string())
    }

    async fn list_directory(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ListDirectoryRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let relative = args.path.unwrap_or_else(|| ".".to_string());
        let path = self.resolve_workspace_path(&relative)?;
        if let Some(pattern) = args.pattern {
            let workspace = self.workspace.clone();
            return tokio::task::spawn_blocking(move || {
                matching_files(&workspace, &path, &pattern)
            })
            .await?;
        }
        let max_depth = args.max_depth.unwrap_or(2);

        let entries = WalkDir::new(&path)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| {
                let kind = if entry.file_type().is_dir() {
                    "dir"
                } else {
                    "file"
                };
                let display_path = crate::files::to_slash(
                    entry
                        .path()
                        .strip_prefix(&self.workspace)
                        .unwrap_or(entry.path()),
                );
                format!("{kind}\t{display_path}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(entries)
    }

    async fn fetch_url(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: FetchUrlRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let response = self.http_client.get(&args.url).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!(
                "request failed for {} with status {}",
                args.url,
                status
            ));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let is_pdf = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/pdf")
            || response.url().path().to_ascii_lowercase().ends_with(".pdf");
        let bytes = response.bytes().await?;
        let max_chars = args.max_chars.unwrap_or(20_000);
        if is_pdf || bytes.starts_with(b"%PDF-") {
            let text =
                tokio::task::spawn_blocking(move || crate::pdf::extract_text(&bytes)).await??;
            return Ok(crate::pdf::truncate_context(&text, max_chars));
        }
        let body = String::from_utf8_lossy(&bytes);
        if content_type.contains("html") {
            let rendered = html2text::from_read(body.as_bytes(), 120)?;
            Ok(truncate_text(&rendered, max_chars))
        } else {
            Ok(truncate_text(&body, max_chars))
        }
    }

    async fn run_shell_command(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ShellCommandRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let cwd = match args.cwd {
            Some(path) => self.resolve_workspace_path(&path)?,
            None => self.workspace.clone(),
        };
        let timeout = Duration::from_secs(args.timeout_seconds.unwrap_or(30));

        // `bash -lc` on Unix, PowerShell on Windows — see
        // `crate::shell::command_parts` for why each.
        let (program, shell_args) = crate::shell::command_parts();
        let mut child = Command::new(program);
        child
            .args(shell_args)
            .arg(&args.command)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = tokio::time::timeout(timeout, child.output())
            .await
            .map_err(|_| anyhow!("command timed out after {:?}", timeout))??;

        let stdout_raw = String::from_utf8_lossy(&output.stdout);
        let stderr_raw = String::from_utf8_lossy(&output.stderr);

        let (stdout_compressed, stderr_compressed) = if self.compression_enabled {
            let (compressed_out, out_stats) = crate::compression::compress_shell_output_with_stats(
                &args.command,
                &stdout_raw,
                self.diff_file_cap,
            );
            let (compressed_err, err_stats) = crate::compression::compress_shell_output_with_stats(
                &args.command,
                &stderr_raw,
                self.diff_file_cap,
            );

            let mut compressed_out = compressed_out;
            let mut compressed_err = compressed_err;

            if out_stats.pattern_matched.as_deref() == Some("generic")
                && out_stats.compressed_lines < out_stats.original_lines
            {
                let tmp_dir = self.workspace.join(".orangu/tmp");
                let _ = tokio::fs::create_dir_all(&tmp_dir).await;
                let log_path = tmp_dir.join("cmd_stdout.log");
                if tokio::fs::write(&log_path, stdout_raw.as_bytes())
                    .await
                    .is_ok()
                {
                    compressed_out.push_str(&format!(
                        "\n\n[Raw stdout diverted to {} due to size. Read this file for missing details.]",
                        log_path.display()
                    ));
                }
            }

            if err_stats.pattern_matched.as_deref() == Some("generic")
                && err_stats.compressed_lines < err_stats.original_lines
            {
                let tmp_dir = self.workspace.join(".orangu/tmp");
                let _ = tokio::fs::create_dir_all(&tmp_dir).await;
                let log_path = tmp_dir.join("cmd_stderr.log");
                if tokio::fs::write(&log_path, stderr_raw.as_bytes())
                    .await
                    .is_ok()
                {
                    compressed_err.push_str(&format!(
                        "\n\n[Raw stderr diverted to {} due to size. Read this file for missing details.]",
                        log_path.display()
                    ));
                }
            }

            if let Ok(mut metrics) = self.compression_metrics.lock() {
                metrics.record(&out_stats);
                metrics.record(&err_stats);
            }
            (compressed_out, compressed_err)
        } else {
            (stdout_raw.to_string(), stderr_raw.to_string())
        };

        let result = ShellCommandResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: truncate_text(&stdout_compressed, 20_000),
            stderr: truncate_text(&stderr_compressed, 20_000),
        };
        Ok(serde_json::to_string_pretty(&result)?)
    }

    fn resolve_workspace_path(&self, raw_path: &str) -> Result<PathBuf> {
        resolve_workspace_path(&self.workspace, raw_path)
    }
}

/// The tools of an SSH session. None takes a credential: a login asks the
/// user for what it needs, so no secret passes through the model.
fn ssh_definitions() -> Vec<ToolDefinition> {
    vec![
        tool(
            "ssh_login",
            "Log in to `host`, an ~/.ssh/config alias or [user@]host[:port]; returns the session \
                 name. Credentials are handled without you. Reuses an open session.",
            json!({
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"]
            }),
        ),
        tool(
            "ssh_copy",
            "Copy one file between workspace path `local` and `remote`; relative remote paths \
                 start at the remote home. Refuses to replace a file unless `overwrite`.",
            json!({
                "type": "object",
                "properties": {
                    "session": {"type": "string"},
                    "direction": {"type": "string", "enum": ["upload", "download"]},
                    "local": {"type": "string"},
                    "remote": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["session", "direction", "local", "remote"]
            }),
        ),
        tool(
            "ssh_logout",
            "Close SSH session `session`.",
            json!({
                "type": "object",
                "properties": {"session": {"type": "string"}},
                "required": ["session"]
            }),
        ),
    ]
}

fn tool(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

pub fn apply_edit(
    original: &str,
    old_text: &str,
    new_text: &str,
    replace_all: bool,
    created: bool,
) -> Result<String> {
    if created {
        return Ok(new_text.to_string());
    }

    if old_text.is_empty() {
        return Ok(new_text.to_string());
    }

    let count = original.matches(old_text).count();
    if count == 0 {
        return Err(anyhow!("old_text was not found in the file"));
    }
    // More than one place it could mean: which one is the caller's to say.
    if count > 1 && !replace_all {
        return Err(anyhow!(
            "old_text occurs {count} times in the file: include more of the text around it, \
             or set replace_all"
        ));
    }

    let updated = if replace_all {
        original.replace(old_text, new_text)
    } else {
        original.replacen(old_text, new_text, 1)
    };

    Ok(updated)
}

/// The files under `root` whose path below it matches the glob `pattern`
/// (`*` stays within a directory, `**` crosses them), newest first, as
/// paths under `workspace`; `.git` is never entered. At most
/// [`PATTERN_LIMIT`], with a note of how many more there are.
fn matching_files(workspace: &Path, root: &Path, pattern: &str) -> Result<String> {
    let matcher = globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|err| anyhow!("the pattern: {err}"))?
        .compile_matcher();
    let mut found = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| e.file_name() != ".git")
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let Ok(under) = entry.path().strip_prefix(root) else {
            continue;
        };
        if !matcher.is_match(under) {
            continue;
        }
        let modified = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let shown = crate::files::to_slash(entry.path().strip_prefix(workspace).unwrap_or(under));
        found.push((modified, shown));
    }
    if found.is_empty() {
        return Ok("No files match.".into());
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let total = found.len();
    let mut lines: Vec<String> = found
        .into_iter()
        .take(PATTERN_LIMIT)
        .map(|(_, p)| p)
        .collect();
    if total > PATTERN_LIMIT {
        lines.push(format!("({} more not shown)", total - PATTERN_LIMIT));
    }
    Ok(lines.join("\n"))
}

/// The media type of a picture — PNG, JPEG, GIF or WebP — by its first
/// bytes; `None` for anything else.
pub fn picture_type(bytes: &[u8]) -> Option<&'static str> {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => Some("image/png"),
        [0xff, 0xd8, 0xff, ..] => Some("image/jpeg"),
        [b'G', b'I', b'F', b'8', ..] => Some("image/gif"),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => Some("image/webp"),
        _ => None,
    }
}

fn render_file_slice(content: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    let start = start_line.unwrap_or(1);
    let end = end_line.unwrap_or(usize::MAX);

    content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line_no = index + 1;
            (line_no >= start && line_no <= end).then(|| format!("{line_no}. {line}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_pdf_slice(content: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    let start = start_line.unwrap_or(1);
    let end = end_line.unwrap_or(usize::MAX);
    let lines: Vec<_> = content.lines().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        if lines[index] == "<table>" {
            let close = lines[index..]
                .iter()
                .position(|line| *line == "</table>")
                .map(|offset| index + offset)
                .unwrap_or(lines.len() - 1);
            if index + 1 >= start && close < end {
                out.extend((index..=close).map(|i| format!("{}. {}", i + 1, lines[i])));
            } else if index < end && close + 1 >= start {
                out.push("[An entire table was omitted because the line range cuts through it. Use PDF page ranges to read complete tables.]".into());
            }
            index = close + 1;
        } else {
            let number = index + 1;
            if number >= start && number <= end {
                out.push(format!("{number}. {}", lines[index]));
            }
            index += 1;
        }
    }
    out.join("\n")
}

pub fn resolve_workspace_path(workspace: &Path, raw_path: &str) -> Result<PathBuf> {
    let candidate = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        workspace.join(raw_path)
    };
    let normalized = normalize_path(&candidate);
    let normalized_workspace = normalize_path(workspace);
    if !normalized.starts_with(&normalized_workspace) {
        return Err(anyhow!("path escapes the configured workspace"));
    }

    // The lexical comparison above catches `..`, but an in-workspace symlink
    // can still point outside. Canonicalize the nearest existing ancestor so
    // this also protects paths for files that have not been created yet.
    let canonical_workspace = normalized_workspace
        .canonicalize()
        .map_err(|error| anyhow!("failed to resolve configured workspace: {error}"))?;
    let mut existing = normalized.as_path();
    let anchor = loop {
        if existing.exists() {
            break existing;
        }
        existing = existing
            .parent()
            .ok_or_else(|| anyhow!("path has no existing parent inside the workspace"))?;
    };
    let canonical_anchor = anchor
        .canonicalize()
        .map_err(|error| anyhow!("failed to resolve workspace path: {error}"))?;
    if !canonical_anchor.starts_with(&canonical_workspace) {
        return Err(anyhow!(
            "path escapes the configured workspace through a symlink"
        ));
    }
    Ok(normalized)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(part) => result.push(part),
        }
    }
    result
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated = text.chars().take(max_chars).collect::<String>();
    format!("{truncated}\n\n[truncated]")
}

/// Extract public item signatures from file content, stripping function bodies.
///
/// Keeps: doc comments (`///`, `//!`), `pub fn`, `pub struct`, `pub enum`,
/// `pub trait`, `pub type`, `pub const`, `pub static`, `impl` blocks (header
/// only), `mod` declarations, `use` statements, and attribute lines (`#[…]`).
/// Strips: private items and function bodies (lines between `{` and matching `}`).
///
/// This is a line-based approximation — no full AST — suitable for v1. It
/// works well for idiomatic Rust where bodies are indented and opening braces
/// are on the same line as the signature.
fn extract_signatures(content: &str) -> String {
    let mut result: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut in_body = false;
    let mut last_was_blank = false;

    // Patterns that mark the START of a signature line we want to keep.
    fn is_signature_line(line: &str) -> bool {
        let t = line.trim_start();
        t.starts_with("pub fn ")
            || t.starts_with("pub async fn ")
            || t.starts_with("pub unsafe fn ")
            || t.starts_with("pub struct ")
            || t.starts_with("pub enum ")
            || t.starts_with("pub trait ")
            || t.starts_with("pub type ")
            || t.starts_with("pub const ")
            || t.starts_with("pub static ")
            || t.starts_with("pub mod ")
            || t.starts_with("pub use ")
            || t.starts_with("pub(crate) fn ")
            || t.starts_with("pub(crate) struct ")
            || t.starts_with("pub(crate) enum ")
            || t.starts_with("pub(crate) trait ")
            || t.starts_with("pub(crate) type ")
            || t.starts_with("pub(crate) const ")
            || t.starts_with("pub(crate) mod ")
            || t.starts_with("impl ")
            || t.starts_with("mod ")
            || t.starts_with("use ")
            || t.starts_with("//!")
            || t.starts_with("///")
            || t.starts_with("#[")
            || t.starts_with("#![")
    }

    for line in content.lines() {
        let trimmed = line.trim();

        // Track brace depth to skip bodies.
        let opens = line.chars().filter(|&c| c == '{').count() as i32;
        let closes = line.chars().filter(|&c| c == '}').count() as i32;

        if in_body {
            depth += opens - closes;
            if depth <= 0 {
                in_body = false;
                depth = 0;
                // Emit a closing marker so the reader can see blocks end.
                result.push("    // ...".to_string());
            }
            continue;
        }

        if is_signature_line(line) {
            // Suppress multiple consecutive blank lines before signatures.
            if !trimmed.is_empty() && last_was_blank {
                result.push(String::new());
            }
            result.push(line.to_string());

            // If this line opens a body brace (and doesn't close it on the
            // same line), enter body-skip mode.
            let net = opens - closes;
            if net > 0 {
                in_body = true;
                depth = net;
            }
        }

        last_was_blank = trimmed.is_empty();
    }

    if result.is_empty() {
        // Fallback: file may not be Rust or has no public items — return as-is.
        return content.to_string();
    }

    format!(
        "[signatures mode — bodies stripped]\n\n{}\n",
        result.join("\n")
    )
}

/// Return a one-line-per-item structural map of the file. Even more compact
/// than `signatures` — only top-level item headers, no doc comments, no
/// attribute lines, no `use` statements.
fn extract_map(content: &str) -> String {
    let mut items: Vec<String> = Vec::new();
    let mut depth: i32 = 0;

    fn is_map_item(line: &str) -> bool {
        let t = line.trim_start();
        (t.starts_with("pub ") || t.starts_with("impl ") || t.starts_with("mod "))
            && !t.starts_with("pub use ")
            && !t.starts_with("pub(crate) use ")
    }

    for line in content.lines() {
        let opens = line.chars().filter(|&c| c == '{').count() as i32;
        let closes = line.chars().filter(|&c| c == '}').count() as i32;

        // Only capture top-level items (depth == 0 before this line).
        if depth == 0 && is_map_item(line) {
            // Trim the body if it starts on the same line: keep only up to `{`.
            let display = if let Some(brace_pos) = line.find('{') {
                line[..brace_pos].trim_end().to_string() + " { ... }"
            } else {
                line.trim_end().to_string()
            };
            items.push(display);
        }

        depth = (depth + opens - closes).max(0);
    }

    if items.is_empty() {
        return content.to_string();
    }

    format!(
        "[map mode — top-level items only]\n\n{}\n",
        items.join("\n")
    )
}

fn redact_secrets(content: &mut String) {
    use regex::Regex;
    use std::sync::OnceLock;

    static SECRET_REGEX: OnceLock<Regex> = OnceLock::new();
    let re = SECRET_REGEX.get_or_init(|| {
        Regex::new(r"(?P<prefix>ghp_[a-zA-Z0-9]{36}|sk-ant-[a-zA-Z0-9_-]{30,}|AKIA[0-9A-Z]{16})")
            .unwrap()
    });

    if re.is_match(content) {
        *content = re.replace_all(content, "[REDACTED_SECRET]").to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn an_edit_that_could_mean_two_places_is_refused() {
        let err = apply_edit("a x b x", "x", "y", false, false).unwrap_err();
        assert!(err.to_string().contains("occurs 2 times"), "{err}");
        assert_eq!(
            apply_edit("a x b x", "x", "y", true, false).unwrap(),
            "a y b y"
        );
    }

    #[tokio::test]
    async fn an_edit_reports_its_replacements() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("f.txt"), "x x x").unwrap();
        let executor = ToolExecutor::new(workspace.path());
        let args = json!({"path": "f.txt", "old_text": "x", "new_text": "y", "replace_all": true, "git": false});
        let out: Value =
            serde_json::from_str(&executor.replace_text(args.as_object().unwrap()).unwrap())
                .unwrap();
        assert_eq!(out["replacements"], 3);
    }

    #[tokio::test]
    async fn a_pattern_lists_the_matching_files_newest_first() {
        let workspace = tempfile::tempdir().unwrap();
        let dir = workspace.path();
        fs::create_dir_all(dir.join("src/deep")).unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join("src/old.rs"), "").unwrap();
        fs::write(dir.join(".git/hidden.rs"), "").unwrap();
        fs::write(dir.join("src/notes.md"), "").unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        fs::File::options()
            .write(true)
            .open(dir.join("src/old.rs"))
            .unwrap()
            .set_modified(old)
            .unwrap();
        fs::write(dir.join("src/deep/new.rs"), "").unwrap();
        let executor = ToolExecutor::new(dir);
        let list = |args: Value| {
            let executor = &executor;
            async move {
                executor
                    .list_directory(args.as_object().unwrap())
                    .await
                    .unwrap()
            }
        };
        assert_eq!(
            list(json!({"pattern": "**/*.rs"})).await,
            "src/deep/new.rs\nsrc/old.rs"
        );
        // `*` does not cross a directory; a `path` roots the pattern.
        assert_eq!(
            list(json!({"path": "src", "pattern": "*.rs"})).await,
            "src/old.rs"
        );
        assert_eq!(list(json!({"pattern": "*.txt"})).await, "No files match.");
    }

    #[tokio::test]
    async fn a_picture_is_named_as_one_and_read_as_bytes() {
        let workspace = tempfile::tempdir().unwrap();
        let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 1];
        fs::write(workspace.path().join("shot.png"), png).unwrap();
        fs::write(workspace.path().join("a.txt"), "text").unwrap();
        let executor = ToolExecutor::new(workspace.path());
        let shown = executor
            .show_file(json!({"path": "shot.png"}).as_object().unwrap())
            .await
            .unwrap();
        assert!(
            shown.contains("is a picture (image/png, 10 bytes)"),
            "{shown}"
        );
        let (kind, bytes) = executor.read_picture("shot.png").unwrap().unwrap();
        assert_eq!((kind, bytes.len()), ("image/png", 10));
        assert!(executor.read_picture("a.txt").unwrap().is_none());
        assert!(executor.read_picture("../outside.png").is_err());
    }

    #[test]
    fn apply_single_edit() {
        let updated = apply_edit("hello world", "world", "orangu", false, false).unwrap();
        assert_eq!(updated, "hello orangu");
    }

    #[test]
    fn create_new_file_content() {
        let updated = apply_edit("", "", "new content", false, true).unwrap();
        assert_eq!(updated, "new content");
    }

    #[tokio::test]
    async fn modify_file_creates_missing_file_with_0644() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("sub/new.txt"));
        args.insert("old_text".into(), json!(""));
        args.insert("new_text".into(), json!("hello orangu"));
        executor.replace_text(&args).unwrap();

        let created = workspace.path().join("sub/new.txt");
        assert_eq!(fs::read_to_string(&created).unwrap(), "hello orangu");
        #[cfg(unix)]
        {
            let mode = fs::metadata(&created).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o644);
        }
    }

    #[tokio::test]
    async fn edit_file_modifies_existing_file() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("existing.txt");
        fs::write(&path, "hello world").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("existing.txt"));
        args.insert("old_text".into(), json!("world"));
        args.insert("new_text".into(), json!("orangu"));
        executor.replace_text(&args).unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "hello orangu");
    }

    #[tokio::test]
    async fn read_file_returns_cache_stub_on_repeated_unchanged_full_read() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let first = executor.show_file(&args).await.unwrap();
        assert!(first.contains("1. one"));
        assert!(first.contains("2. two"));

        let second = executor.show_file(&args).await.unwrap();
        assert!(second.starts_with("[cached] README.md is unchanged"));
        assert!(second.contains("start_line/end_line"));
    }

    #[tokio::test]
    async fn pdf_tool_reads_text_and_caches_it_for_queries() {
        let workspace = tempfile::tempdir().unwrap();
        let bytes = crate::pdf::encode_text("Facts", "The project codename is Apricot.").unwrap();
        fs::write(workspace.path().join("facts.PDF"), bytes).unwrap();
        let executor = ToolExecutor::new(workspace.path());
        let mut args = Map::new();
        args.insert("path".into(), json!("facts.PDF"));
        let text = executor.show_file(&args).await.unwrap();
        assert!(text.contains("The project codename is Apricot."));
        assert!(!text.contains("%PDF-"));
        assert!(
            executor
                .show_file(&args)
                .await
                .unwrap()
                .starts_with("[cached]")
        );
        args.insert("start_line".into(), json!(1));
        assert!(
            !executor
                .show_file(&args)
                .await
                .unwrap()
                .starts_with("[cached]")
        );
    }

    #[test]
    fn pdf_limits_never_expose_partial_table_rows() {
        let text = "Header\n<table>\n<caption>Values</caption>\n<tr><td>123</td><td>456</td></tr>\n</table>\nFooter";
        let limited = crate::pdf::truncate_context(text, text.find("456").unwrap());
        assert!(limited.contains("An entire table"));
        assert!(!limited.contains("123"));
        assert!(!limited.contains("<table>"));
        let sliced = render_pdf_slice(text, Some(3), Some(4));
        assert!(sliced.contains("An entire table"));
        assert!(!sliced.contains("123"));
        let complete = render_pdf_slice(text, Some(2), Some(5));
        assert!(complete.contains("123"));
        assert!(complete.contains("</table>"));
        assert!(!complete.contains("omitted"));
    }

    #[tokio::test]
    async fn read_file_returns_full_content_again_after_change() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let _ = executor.show_file(&args).await.unwrap();
        let _ = executor.show_file(&args).await.unwrap();

        fs::write(&path, "one\ntwo\nthree\n").unwrap();

        let changed = executor.show_file(&args).await.unwrap();
        assert!(changed.contains("1. one"));
        assert!(changed.contains("2. two"));
        assert!(changed.contains("3. three"));
        assert!(!changed.starts_with("[cached]"));
    }

    #[tokio::test]
    // Unix-only: the fixture is a `#!/usr/bin/env bash` script made
    // executable with `chmod`, which Windows has no equivalent for.
    // The output compression under test is platform-independent; only
    // this way of standing up a fake `cargo` is not.
    #[cfg(unix)]
    async fn run_shell_command_compresses_cargo_test_success_noise() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 3 tests\\ntest a ... ok\\ntest b ... ok\\ntest c ... ok\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("... (3 tests passed)"));
        assert!(!rendered.contains("test a ... ok"));
        assert!(!rendered.contains("test b ... ok"));
        assert!(!rendered.contains("test c ... ok"));
    }

    #[tokio::test]
    // Unix-only: the fixture is a `#!/usr/bin/env bash` script made
    // executable with `chmod`, which Windows has no equivalent for.
    // The output compression under test is platform-independent; only
    // this way of standing up a fake `cargo` is not.
    #[cfg(unix)]
    async fn run_shell_command_keeps_cargo_test_failures_visible() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 3 tests\\ntest a ... ok\\ntest b ... FAILED\\nfailures:\\n---- test b stdout ----\\npanicked at boom\\ntest result: FAILED. 1 passed; 1 failed\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("test b ... FAILED"));
        assert!(rendered.contains("failures:"));
        assert!(rendered.contains("panicked at boom"));
        assert!(!rendered.contains("test a ... ok"));
    }

    #[tokio::test]
    async fn read_file_without_compression_returns_full_content_on_repeat() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::with_config(workspace.path(), false, 300, 20, None);

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let first = executor.show_file(&args).await.unwrap();
        let second = executor.show_file(&args).await.unwrap();

        assert_eq!(first, second);
        assert!(!second.starts_with("[cached]"));
    }

    #[tokio::test]
    // Unix-only: the fixture is a `#!/usr/bin/env bash` script made
    // executable with `chmod`, which Windows has no equivalent for.
    // The output compression under test is platform-independent; only
    // this way of standing up a fake `cargo` is not.
    #[cfg(unix)]
    async fn run_shell_command_without_compression_keeps_raw_output() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 2 tests\\ntest a ... ok\\ntest b ... ok\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::with_config(workspace.path(), false, 300, 20, None);

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("test a ... ok"));
        assert!(rendered.contains("test b ... ok"));
        assert!(!rendered.contains("tests passed"));
    }

    #[test]
    fn rejects_path_escape() {
        let workspace = PathBuf::from("/tmp/workspace");
        let err = resolve_workspace_path(&workspace, "../outside").unwrap_err();
        assert!(err.to_string().contains("escapes"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let workspace = tempfile::tempdir().expect("workspace");
        let outside = tempfile::tempdir().expect("outside");
        std::os::unix::fs::symlink(outside.path(), workspace.path().join("escape"))
            .expect("symlink");

        let err = resolve_workspace_path(workspace.path(), "escape/new.txt")
            .expect_err("symlink must not leave workspace");
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    #[test]
    fn signatures_mode_strips_bodies_keeps_pub_fn() {
        let src = "
pub fn hello() {
    println!(\"hi\");
}

fn private_fn() {
    // private body
}

/// A doc comment.
pub struct Foo {
    pub x: i32,
}
";
        let result = extract_signatures(src);
        assert!(
            result.contains("[signatures mode"),
            "should have mode header"
        );
        assert!(result.contains("pub fn hello()"), "should keep pub fn");
        assert!(!result.contains("println!"), "should strip body");
        assert!(!result.contains("private_fn"), "should skip private fn");
        assert!(
            result.contains("/// A doc comment."),
            "should keep doc comment"
        );
        assert!(result.contains("pub struct Foo"), "should keep pub struct");
    }

    #[test]
    fn signatures_mode_fallback_for_no_pub_items() {
        let src = "fn private() {}";
        let result = extract_signatures(src);
        // Falls back to returning content as-is when nothing matches.
        assert_eq!(result, src);
    }

    #[test]
    fn map_mode_only_top_level_items() {
        let src = "
pub struct Foo { x: i32 }

impl Foo {
    pub fn method(&self) {}
}

pub fn standalone() {}
";
        let result = extract_map(src);
        assert!(result.contains("[map mode"), "should have mode header");
        assert!(result.contains("pub struct Foo"), "top-level struct");
        assert!(result.contains("impl Foo"), "impl block header");
        assert!(result.contains("pub fn standalone()"), "top-level fn");
        // method() is inside impl, should NOT appear as separate item
        assert!(
            !result.contains("pub fn method"),
            "nested fn should not appear"
        );
    }

    #[test]
    fn grep_context_is_compact_under_limit() {
        use crate::compression::prepare_llm_grep_context;
        let output = "src/foo.rs:10:    pub fn foo() {}\nsrc/bar.rs:20:    pub fn bar() {}";
        let ctx = prepare_llm_grep_context("fn foo", output, true, None);
        // Under 40 matches — all should appear, no omission note.
        assert!(ctx.content.contains("src/foo.rs"));
        assert!(ctx.note.is_none());
    }

    #[test]
    fn grep_context_truncates_over_limit() {
        use crate::compression::prepare_llm_grep_context;
        // Generate 50 fake match lines.
        let output: String = (0..50)
            .map(|i| format!("src/x.rs:{i}: fn item_{i}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");
        let ctx = prepare_llm_grep_context("item", &output, true, None);
        assert!(ctx.note.is_some(), "should have truncation note");
        let note = ctx.note.unwrap();
        assert!(note.contains("50 matches found"), "should mention counts");
        assert!(note.contains("first 40"), "should mention counts");
    }
}

#[cfg(test)]
mod file_lifecycle_tool_tests {
    use super::*;
    use serde_json::json;

    fn args(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// Every file-lifecycle endpoint is a tool of the same name, taking the
    /// same fields.
    #[test]
    fn the_endpoints_are_all_offered_as_tools() {
        let workspace = tempfile::tempdir().unwrap();
        let names: Vec<String> = ToolExecutor::new(workspace.path())
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect();

        for name in [
            "create_file",
            "modify_file",
            "move_file",
            "delete_file",
            "show_file",
            "create_directory",
            "move_directory",
            "delete_directory",
            "semantic_search",
        ] {
            assert!(names.contains(&name.to_string()), "missing tool: {name}");
        }
        // Migrated away from, not kept alongside.
        assert!(!names.contains(&"read_file".to_string()));
        assert!(!names.contains(&"edit_file".to_string()));
    }

    /// Every request's prompt carries the definitions, so they stay small:
    /// 6982 characters of JSON (~1700 tokens) once, trimmed to under 4700
    /// until the graph tools, under 6000 with semantic search included, under
    /// 6500 with the `write_tests` and `write_docs` subagents, under 7300 with
    /// `convert_document` and the `write_document` subagents.
    #[test]
    fn the_tool_definitions_stay_small() {
        let workspace = tempfile::tempdir().unwrap();
        let json =
            serde_json::to_string(&ToolExecutor::new(workspace.path()).definitions()).unwrap();
        assert!(json.len() < 7300, "{} characters", json.len());
    }

    /// `/license`'s answer reaches the file the model writes — the whole
    /// point of the choice living on the executor. Set mid-session, without
    /// rebuilding anything, and the *next* `create_file` uses it.
    #[tokio::test]
    async fn the_session_licence_choice_reaches_create_file() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("Cargo.toml"),
            "[package]\nlicense = \"MIT\"\nauthors = [\"Jane Roe\"]\n",
        )
        .unwrap();
        let executor = ToolExecutor::new(workspace.path());
        let call = |name: &str| args(&[("path", json!(name)), ("content", json!("x = 1\n"))]);

        // Auto: the workspace's own licence.
        executor
            .execute("create_file", &call("a.py"))
            .await
            .unwrap();
        let a = std::fs::read_to_string(workspace.path().join("a.py")).unwrap();
        assert!(a.starts_with("# MIT License"), "{a}");

        // Chosen: that licence and that holder, over the workspace's.
        executor.set_licence_choice(crate::license::Choice::Use {
            licence: crate::license::Licence::Agpl3OrLater,
            holder: Some("Acme Ltd".to_string()),
        });
        executor
            .execute("create_file", &call("b.py"))
            .await
            .unwrap();
        let b = std::fs::read_to_string(workspace.path().join("b.py")).unwrap();
        assert!(b.contains("GNU Affero General Public License"), "{b}");
        assert!(b.contains("Acme Ltd"), "{b}");

        // Off: no header at all.
        executor.set_licence_choice(crate::license::Choice::None);
        executor
            .execute("create_file", &call("c.py"))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(workspace.path().join("c.py")).unwrap(),
            "x = 1\n"
        );
    }

    /// The model must not be able to relicense somebody's project by writing
    /// a field: the licensing fields are the caller's and are not in the
    /// schema, so anything the model sends for them is overwritten.
    #[tokio::test]
    async fn a_model_cannot_choose_the_licence_itself() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(
            workspace.path().join("Cargo.toml"),
            "[package]\nlicense = \"MIT\"\nauthors = [\"Jane Roe\"]\n",
        )
        .unwrap();
        let executor = ToolExecutor::new(workspace.path());
        executor
            .execute(
                "create_file",
                &args(&[
                    ("path", json!("a.py")),
                    ("content", json!("x = 1\n")),
                    ("license_id", json!("AGPL-3.0-or-later")),
                    ("license_holder", json!("Someone Else")),
                ]),
            )
            .await
            .unwrap();
        let written = std::fs::read_to_string(workspace.path().join("a.py")).unwrap();
        assert!(written.starts_with("# MIT License"), "{written}");
        assert!(!written.contains("Someone Else"), "{written}");
    }

    /// A read-only executor (`/review`, the explorer subagent) offers the
    /// read but none of the mutating operations, and refuses them if called
    /// anyway.
    #[tokio::test]
    async fn read_only_mode_offers_show_file_but_no_mutations() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new_read_only(workspace.path());
        let names: Vec<String> = executor
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect();

        assert!(names.contains(&"show_file".to_string()));
        for name in ["create_file", "modify_file", "delete_file", "move_file"] {
            assert!(!names.contains(&name.to_string()), "offered: {name}");
        }

        let err = executor
            .execute(
                "create_file",
                &args(&[("path", json!("a.txt")), ("content", json!("x\n"))]),
            )
            .await
            .expect_err("read-only");
        assert!(err.to_string().contains("read-only"), "{err}");
    }

    fn tool_names(executor: &ToolExecutor) -> Vec<String> {
        executor
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect()
    }

    /// The main agent is offered every subagent; a read-only executor none.
    #[test]
    fn subagent_tools_are_offered_only_to_the_main_writable_executor() {
        let workspace = tempfile::tempdir().unwrap();
        // With web search engines, so `research_web` is offered too.
        let main = tool_names(
            &ToolExecutor::new(workspace.path())
                .with_web_search(crate::web_search::DEFAULT_ENGINES.to_vec())
                .with_ssh(crate::ssh::SshSessions::for_mode(
                    crate::config::SshMode::Prompt,
                    None,
                )),
        );
        let read_only = tool_names(&ToolExecutor::new_read_only(workspace.path()));
        for spec in crate::subagents::SUBAGENTS {
            assert!(
                main.contains(&spec.tool.to_string()),
                "missing: {}",
                spec.tool
            );
            assert!(
                !read_only.contains(&spec.tool.to_string()),
                "offered: {}",
                spec.tool
            );
        }
        let document = crate::subagents::DOCUMENT_TOOL.to_string();
        assert!(main.contains(&document));
        assert!(!read_only.contains(&document));
        assert!(main.contains(&"convert_document".to_string()));
        assert!(!read_only.contains(&"convert_document".to_string()));
    }

    fn custom_agents(files: &[(&str, &str)]) -> crate::agents::custom::AgentRegistry {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            fs::write(dir.path().join(name), content).unwrap();
        }
        crate::agents::custom::AgentRegistry::from_directories(&[(
            dir.path().to_path_buf(),
            crate::agents::custom::AgentSource::User,
        )])
    }

    /// A one-request HTTP server answering with `body` as JSON, standing in
    /// for a SearXNG instance; returns its base URL.
    async fn fake_searxng(body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 4096];
            let _ = socket.read(&mut request).await;
            let _ = socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        });
        base
    }

    /// The main agent gets `research_web` only with engines, and never
    /// `web_search` or `browse` itself; a browsing subagent gets both.
    #[tokio::test]
    async fn web_tools_are_offered_by_engines_and_role() {
        let workspace = tempfile::tempdir().unwrap();
        let engines = crate::web_search::DEFAULT_ENGINES.to_vec();
        let names = |executor: ToolExecutor| tool_names(&executor);

        let without = names(ToolExecutor::new(workspace.path()));
        for tool in ["research_web", "web_search", "browse"] {
            assert!(!without.contains(&tool.to_string()), "{tool}");
        }

        let main = names(ToolExecutor::new(workspace.path()).with_web_search(engines.clone()));
        assert!(main.contains(&"research_web".to_string()));
        assert!(!main.contains(&"web_search".to_string()));
        assert!(!main.contains(&"browse".to_string()));

        let researcher = names(
            ToolExecutor::new_read_only(workspace.path())
                .with_web_search(engines.clone())
                .with_browsing(),
        );
        for tool in ["web_search", "browse", "fetch_url"] {
            assert!(researcher.contains(&tool.to_string()), "{tool}");
        }
        assert!(!researcher.contains(&"research_web".to_string()));
        assert!(!researcher.contains(&"create_file".to_string()));

        // Web search off: still browsing, but no searching.
        let off = names(ToolExecutor::new_read_only(workspace.path()).with_browsing());
        assert!(off.contains(&"browse".to_string()));
        assert!(!off.contains(&"web_search".to_string()));

        // An allowlist without them (a custom agent with neither `WebSearch`
        // nor `WebFetch`) hides them.
        let allowed = std::iter::once("show_file".to_string()).collect();
        let limited = ToolExecutor::new_for_subagent(workspace.path())
            .with_web_search(engines)
            .with_browsing()
            .with_allowed_tools(allowed);
        assert_eq!(tool_names(&limited), ["show_file"]);
    }

    #[tokio::test]
    async fn web_search_returns_numbered_results_and_names_the_engine() {
        let workspace = tempfile::tempdir().unwrap();
        let base = fake_searxng(
            r#"{"results":[{"url":"https://docs.example/a","title":"A","content":"About A."}]}"#,
        )
        .await;
        let engines = vec![crate::web_search::Engine::Searxng(base.clone())];
        let executor = ToolExecutor::new_read_only(workspace.path())
            .with_web_search(engines)
            .with_browsing();
        let out = executor
            .execute("web_search", &args(&[("query", json!(" what is A "))]))
            .await
            .unwrap();
        assert_eq!(
            out,
            format!(
                "Search results for \"what is A\":\n\n[1] A\nhttps://docs.example/a\nAbout A.\n(from {base})\n"
            )
        );

        let err = ToolExecutor::new_read_only(workspace.path())
            .with_browsing()
            .execute("web_search", &args(&[("query", json!("x"))]))
            .await
            .expect_err("search off");
        assert!(err.to_string().contains("[orangu].web_search"), "{err}");

        let err = ToolExecutor::new(workspace.path())
            .execute("browse", &args(&[("url", json!("https://x.example/"))]))
            .await
            .expect_err("not a browsing executor");
        assert!(err.to_string().contains("web subagents"), "{err}");
        let err = ToolExecutor::new_read_only(workspace.path())
            .with_browsing()
            .execute("browse", &args(&[("link", json!("two"))]))
            .await
            .expect_err("not a number");
        assert!(err.to_string().contains("'link'"), "{err}");
    }

    /// Only an interactive conversation's executor offers `background` on
    /// the subagent tools and `continue_agent`; elsewhere nothing changes.
    #[test]
    fn background_tools_are_offered_only_where_reports_can_arrive() {
        let workspace = tempfile::tempdir().unwrap();
        let plain = ToolExecutor::new(workspace.path()).definitions();
        assert!(!plain.iter().any(|d| d.function.name == "continue_agent"));
        assert!(plain.iter().all(|d| {
            d.function.parameters["properties"]
                .get("background")
                .is_none()
        }));

        let interactive = ToolExecutor::new(workspace.path()).with_background_tasks();
        let defs = interactive.definitions();
        assert!(defs.iter().any(|d| d.function.name == "continue_agent"));
        for def in &defs {
            let has = def.function.parameters["properties"]
                .get("background")
                .is_some();
            assert_eq!(
                has,
                interactive.runs_subagent(&def.function.name),
                "{}",
                def.function.name
            );
        }
        // A subagent's own executor never gets them.
        let nested = ToolExecutor::new_for_subagent(workspace.path()).with_background_tasks();
        assert!(!tool_names(&nested).contains(&"continue_agent".to_string()));
    }

    /// `save_memory` exists only with a memory file, writes it whole (even
    /// outside the workspace: a user-scope memory), and refuses an essay.
    #[tokio::test]
    async fn save_memory_rewrites_the_agents_memory_file() {
        let workspace = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        assert!(
            !tool_names(&ToolExecutor::new_for_subagent(workspace.path()))
                .contains(&"save_memory".to_string())
        );

        let file = home.path().join("agent-memory/keeper/MEMORY.md");
        let executor =
            ToolExecutor::new_for_subagent(workspace.path()).with_memory_file(file.clone());
        assert!(tool_names(&executor).contains(&"save_memory".to_string()));
        executor
            .execute(
                "save_memory",
                &args(&[("content", json!("First note.  \n\n"))]),
            )
            .await
            .unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "First note.\n");
        executor
            .execute("save_memory", &args(&[("content", json!("Replaced."))]))
            .await
            .unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "Replaced.\n");
        assert!(!file.with_file_name(".MEMORY.md.tmp").exists());

        let err = executor
            .execute(
                "save_memory",
                &args(&[("content", json!("x".repeat(70_000)))]),
            )
            .await
            .expect_err("too long");
        assert!(err.to_string().contains("limited"), "{err}");
        assert_eq!(fs::read_to_string(&file).unwrap(), "Replaced.\n");
    }

    /// `disallowedTools` is a denylist: those tools are neither offered nor
    /// run, whatever else allows them.
    #[tokio::test]
    async fn a_denylist_takes_tools_away_from_what_is_offered_and_run() {
        let workspace = tempfile::tempdir().unwrap();
        let denied: std::collections::HashSet<String> = ["create_file", "run_shell_command"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let executor = ToolExecutor::new_for_subagent(workspace.path()).with_denied_tools(denied);
        let names = tool_names(&executor);
        assert!(names.contains(&"show_file".to_string()));
        assert!(names.contains(&"modify_file".to_string()));
        assert!(!names.contains(&"create_file".to_string()));
        assert!(!names.contains(&"run_shell_command".to_string()));
        let err = executor
            .execute(
                "create_file",
                &args(&[("path", json!("b.txt")), ("content", json!("x"))]),
            )
            .await
            .expect_err("denied");
        assert!(
            err.to_string().contains("not available to this agent"),
            "{err}"
        );
        assert!(!workspace.path().join("b.txt").exists());
    }

    /// A custom agent's `tools` field is an allowlist: only those tools are
    /// offered, and anything else is refused.
    #[tokio::test]
    async fn an_allowlist_limits_what_is_offered_and_run() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("a.txt"), "alpha\nbeta\n").unwrap();
        let allowed: std::collections::HashSet<String> = ["show_file", "search_text"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let executor = ToolExecutor::new_for_subagent(workspace.path()).with_allowed_tools(allowed);
        let mut names = tool_names(&executor);
        names.sort();
        assert_eq!(names, ["search_text", "show_file"]);

        let err = executor
            .execute(
                "create_file",
                &args(&[("path", json!("b.txt")), ("content", json!("x"))]),
            )
            .await
            .expect_err("not allowed");
        assert!(
            err.to_string().contains("not available to this agent"),
            "{err}"
        );
        assert!(!workspace.path().join("b.txt").exists());

        // search_text is not offered without an allowlist that names it.
        assert!(
            !tool_names(&ToolExecutor::new(workspace.path())).contains(&"search_text".to_string())
        );
    }

    #[tokio::test]
    async fn search_text_finds_lines_and_respects_gitignore_and_glob() {
        let workspace = tempfile::tempdir().unwrap();
        // `ignore` honours .gitignore only inside a repository.
        fs::create_dir(workspace.path().join(".git")).unwrap();
        fs::write(workspace.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::create_dir(workspace.path().join("src")).unwrap();
        fs::write(
            workspace.path().join("src/a.rs"),
            "fn main() {}\nfn Helper() {}\n",
        )
        .unwrap();
        fs::write(workspace.path().join("notes.md"), "fn in prose\n").unwrap();
        fs::write(workspace.path().join("ignored.rs"), "fn hidden() {}\n").unwrap();
        let allowed = std::iter::once("search_text".to_string()).collect();
        let executor = ToolExecutor::new_for_subagent(workspace.path()).with_allowed_tools(allowed);

        let out = executor
            .execute("search_text", &args(&[("pattern", json!(r"^fn \w+\("))]))
            .await
            .unwrap();
        assert!(out.contains("src/a.rs:1: fn main() {}"), "{out}");
        assert!(out.contains("src/a.rs:2: fn Helper() {}"), "{out}");
        assert!(!out.contains("ignored.rs"), "{out}");

        let out = executor
            .execute(
                "search_text",
                &args(&[
                    ("pattern", json!("FN")),
                    ("ignore_case", json!(true)),
                    ("glob", json!("*.md")),
                ]),
            )
            .await
            .unwrap();
        assert_eq!(out, "notes.md:1: fn in prose\n");

        let out = executor
            .execute(
                "search_text",
                &args(&[("pattern", json!("fn")), ("max_results", json!(1))]),
            )
            .await
            .unwrap();
        assert!(out.ends_with("[stopped at 1 matches]\n"), "{out}");

        let out = executor
            .execute("search_text", &args(&[("pattern", json!("zzz"))]))
            .await
            .unwrap();
        assert_eq!(out, "no matches\n");

        assert!(
            executor
                .execute("search_text", &args(&[("pattern", json!("("))]))
                .await
                .is_err()
        );
        assert!(
            executor
                .execute(
                    "search_text",
                    &args(&[("pattern", json!("x")), ("path", json!("../"))])
                )
                .await
                .is_err()
        );
    }

    /// `run_agent` exists only when there are custom agents, lists them by
    /// search, and names the closest ones for an unknown agent, all without
    /// a model.
    #[tokio::test]
    async fn run_agent_is_offered_with_custom_agents_and_lists_them() {
        let workspace = tempfile::tempdir().unwrap();
        assert!(
            !tool_names(&ToolExecutor::new(workspace.path())).contains(&"run_agent".to_string())
        );

        let agents = custom_agents(&[
            (
                "api.md",
                "---\nname: api-designer\ndescription: Designs REST APIs.\ntools: Read, Write\n---\nDesign.\n",
            ),
            (
                "rev.md",
                "---\nname: code-reviewer\ndescription: Reviews code quality.\ntools: Read, Grep\n---\nReview.\n",
            ),
        ]);
        let executor = ToolExecutor::new(workspace.path()).with_custom_agents(agents.clone());
        assert!(tool_names(&executor).contains(&"run_agent".to_string()));
        let read_only =
            ToolExecutor::new_read_only(workspace.path()).with_custom_agents(agents.clone());
        assert!(!tool_names(&read_only).contains(&"run_agent".to_string()));

        let listing = executor
            .execute(
                "run_agent",
                &args(&[("agent", json!("list")), ("task", json!("review"))]),
            )
            .await
            .unwrap();
        assert_eq!(
            listing,
            "- code-reviewer (read-only): Reviews code quality."
        );

        let err = executor
            .execute(
                "run_agent",
                &args(&[("agent", json!("api-architect")), ("task", json!("design"))]),
            )
            .await
            .expect_err("unknown");
        let err = err.to_string();
        assert!(err.contains("unknown agent 'api-architect'"), "{err}");
        assert!(
            err.contains("- api-designer (writes): Designs REST APIs."),
            "{err}"
        );

        // A subagent cannot start a custom agent.
        let nested = ToolExecutor::new_for_subagent(workspace.path()).with_custom_agents(agents);
        assert!(!tool_names(&nested).contains(&"run_agent".to_string()));
        let err = nested
            .execute(
                "run_agent",
                &args(&[("agent", json!("api-designer")), ("task", json!("x"))]),
            )
            .await
            .expect_err("nested");
        assert!(err.to_string().contains("cannot start subagents"), "{err}");
    }

    /// `convert_document` renders text into a LibreOffice document and that
    /// document into a PDF, inside the workspace only.
    #[tokio::test]
    async fn convert_document_renders_and_converts_within_the_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new_for_subagent(workspace.path());

        let response = executor
            .execute(
                "convert_document",
                &args(&[
                    ("target", json!("sheet.ods")),
                    ("content", json!("Item,Cost\nA,2\nB,3\nTotal,=SUM(B2:B3)\n")),
                    ("format", json!("csv")),
                ]),
            )
            .await
            .expect("render");
        let response: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["target"], "sheet.ods");
        assert_eq!(response["format"], "ods");

        executor
            .execute(
                "convert_document",
                &args(&[
                    ("source", json!("sheet.ods")),
                    ("target", json!("sheet.pdf")),
                ]),
            )
            .await
            .expect("to pdf");
        assert!(
            fs::read(workspace.path().join("sheet.pdf"))
                .unwrap()
                .starts_with(b"%PDF-")
        );

        let err = executor
            .execute(
                "convert_document",
                &args(&[
                    ("source", json!("sheet.ods")),
                    ("target", json!("../out.pdf")),
                ]),
            )
            .await
            .expect_err("escape");
        assert!(err.to_string().contains("escapes"), "{err}");

        let err = ToolExecutor::new_read_only(workspace.path())
            .execute(
                "convert_document",
                &args(&[("source", json!("sheet.ods")), ("target", json!("x.pdf"))]),
            )
            .await
            .expect_err("read-only");
        assert!(err.to_string().contains("read-only"), "{err}");
    }

    /// A subagent's executor can change the workspace but cannot start
    /// another subagent unless it is set up to nest.
    #[tokio::test]
    async fn subagent_executor_writes_but_cannot_delegate() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new_for_subagent(workspace.path());
        let names = tool_names(&executor);

        assert!(names.contains(&"create_file".to_string()));
        assert!(names.contains(&"modify_file".to_string()));
        for spec in crate::subagents::SUBAGENTS {
            assert!(
                !names.contains(&spec.tool.to_string()),
                "offered: {}",
                spec.tool
            );
            let err = executor
                .execute(spec.tool, &args(&[(spec.argument, json!("anything"))]))
                .await
                .expect_err("nested subagent");
            assert!(err.to_string().contains("cannot start subagents"), "{err}");
        }
        assert!(!names.contains(&crate::subagents::DOCUMENT_TOOL.to_string()));
        assert!(names.contains(&"convert_document".to_string()));
        let err = executor
            .execute(
                crate::subagents::DOCUMENT_TOOL,
                &args(&[("type", json!("text")), ("task", json!("anything"))]),
            )
            .await
            .expect_err("nested subagent");
        assert!(err.to_string().contains("cannot start subagents"), "{err}");

        executor
            .execute(
                "create_file",
                &args(&[
                    ("path", json!("a.txt")),
                    ("content", json!("x\n")),
                    ("git", json!(false)),
                ]),
            )
            .await
            .expect("subagent may write");
        assert!(workspace.path().join("a.txt").exists());
    }

    /// The whole life cycle through the tool interface, on the same library
    /// functions `orangu-server` serves.
    #[tokio::test]
    async fn undo_puts_back_what_the_file_tools_wrote() {
        let workspace = tempfile::tempdir().unwrap();
        let session = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("keep.txt"), "original\n").unwrap();
        let executor = ToolExecutor::with_config(
            workspace.path(),
            false,
            300,
            20,
            Some(session.path().to_path_buf()),
        );
        executor.checkpoints().begin_turn("rework the files");
        executor
            .execute(
                "modify_file",
                &args(&[
                    ("path", json!("keep.txt")),
                    ("old_text", json!("original")),
                    ("new_text", json!("changed")),
                ]),
            )
            .await
            .expect("modify_file");
        executor
            .execute(
                "create_file",
                &args(&[
                    ("path", json!("made/new.txt")),
                    ("content", json!("new\n")),
                    ("parents", json!(true)),
                ]),
            )
            .await
            .expect("create_file");
        executor
            .execute(
                "move_file",
                &args(&[("from", json!("keep.txt")), ("to", json!("moved.txt"))]),
            )
            .await
            .expect("move_file");
        executor.checkpoints().finish_turn();

        let restored = executor.checkpoints().undo(false).expect("undo");
        assert_eq!(
            fs::read_to_string(workspace.path().join("keep.txt")).unwrap(),
            "original\n"
        );
        assert!(!workspace.path().join("moved.txt").exists());
        assert!(!workspace.path().join("made").exists());
        assert_eq!(restored.written, vec!["keep.txt"]);
        assert_eq!(restored.removed, vec!["made/new.txt", "moved.txt"]);
    }

    #[tokio::test]
    async fn a_file_can_be_created_shown_moved_and_deleted_through_the_tools() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new(workspace.path());

        // `mode` is optional and Unix-only (`files::set_mode` refuses it
        // elsewhere), so it is left out off Unix — the life cycle this test
        // covers is the same either way.
        let mut create_args = vec![
            ("path", json!("src/a.txt")),
            ("content", json!("one\ntwo\n")),
            ("parents", json!(true)),
        ];
        if cfg!(unix) {
            create_args.push(("mode", json!("0640")));
        }
        executor
            .execute("create_file", &args(&create_args))
            .await
            .expect("create_file");
        assert_eq!(
            fs::read_to_string(workspace.path().join("src/a.txt")).unwrap(),
            "one\ntwo\n"
        );

        let shown = executor
            .execute("show_file", &args(&[("path", json!("src/a.txt"))]))
            .await
            .expect("show_file");
        assert!(shown.contains("two"), "{shown}");

        executor
            .execute(
                "modify_file",
                &args(&[
                    ("path", json!("src/a.txt")),
                    (
                        "edits",
                        json!([{"start_line": 1, "end_line": 1, "replacement": "ONE\n"}]),
                    ),
                ]),
            )
            .await
            .expect("modify_file with edits");
        assert_eq!(
            fs::read_to_string(workspace.path().join("src/a.txt")).unwrap(),
            "ONE\ntwo\n"
        );

        executor
            .execute(
                "modify_file",
                &args(&[
                    ("path", json!("src/a.txt")),
                    ("old_text", json!("two")),
                    ("new_text", json!("TWO")),
                ]),
            )
            .await
            .expect("modify_file with old_text");
        assert_eq!(
            fs::read_to_string(workspace.path().join("src/a.txt")).unwrap(),
            "ONE\nTWO\n"
        );

        executor
            .execute(
                "move_file",
                &args(&[("from", json!("src/a.txt")), ("to", json!("src/b.txt"))]),
            )
            .await
            .expect("move_file");
        assert!(workspace.path().join("src/b.txt").exists());

        executor
            .execute("delete_file", &args(&[("path", json!("src/b.txt"))]))
            .await
            .expect("delete_file");
        assert!(!workspace.path().join("src/b.txt").exists());

        executor
            .execute("create_directory", &args(&[("path", json!("src/deep"))]))
            .await
            .expect("create_directory");
        executor
            .execute(
                "move_directory",
                &args(&[("from", json!("src/deep")), ("to", json!("src/deeper"))]),
            )
            .await
            .expect("move_directory");
        executor
            .execute("delete_directory", &args(&[("path", json!("src/deeper"))]))
            .await
            .expect("delete_directory");
        assert!(!workspace.path().join("src/deeper").exists());
    }

    /// The tools are confined to the workspace exactly as the endpoints are.
    #[tokio::test]
    async fn tools_refuse_paths_outside_the_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "secret\n").unwrap();
        let executor = ToolExecutor::new(workspace.path());
        let escape = outside.path().join("secret.txt").display().to_string();

        for (tool, arguments) in [
            (
                "create_file",
                args(&[("path", json!("../escaped.txt")), ("content", json!("x"))]),
            ),
            ("show_file", args(&[("path", json!(escape.clone()))])),
            ("delete_file", args(&[("path", json!(escape.clone()))])),
            (
                "move_file",
                args(&[("from", json!(escape.clone())), ("to", json!("here.txt"))]),
            ),
            ("create_directory", args(&[("path", json!("../escaped"))])),
        ] {
            let err = executor
                .execute(tool, &arguments)
                .await
                .expect_err("outside the workspace");
            assert!(
                err.to_string().contains("workspace"),
                "{tool}: unexpected error: {err}"
            );
        }
        assert_eq!(
            fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "secret\n"
        );
    }
}

#[cfg(test)]
mod graph_tool_tests {
    use super::*;
    use crate::graph::extract::ExtractedNode;
    use crate::graph::store::GraphStore;
    use serde_json::json;

    fn lookup(tools: &ToolExecutor, symbol: &str) -> String {
        let arguments: Map<String, Value> = [("symbol".to_string(), json!(symbol))]
            .into_iter()
            .collect();
        tools.graph_lookup(&arguments).expect("graph_lookup")
    }

    /// Removes the graph cache a scan of `workspace` leaves under
    /// `~/.orangu/workspace/`, which outlives the temporary directory it was
    /// keyed by.
    fn forget_scan(workspace: &Path) {
        let cache = crate::workspace_cache::workspace_cache_dir(workspace, "graph");
        let _ = fs::remove_dir_all(&cache);
        // And the per-workspace directory holding it, if it now holds nothing.
        if let Some(parent) = cache.parent() {
            let _ = fs::remove_dir(parent);
        }
    }

    fn workspace_with_a_symbol() -> tempfile::TempDir {
        let workspace = tempfile::tempdir().expect("workspace");
        fs::create_dir_all(workspace.path().join("src")).expect("src dir");
        fs::write(
            workspace.path().join("src/lib.rs"),
            "pub fn scanned_symbol() {}\n",
        )
        .expect("src file");
        workspace
    }

    /// A lookup that lands while the startup scan is still running waits for
    /// that scan and answers from its graph, rather than reporting a graph
    /// that isn't built — an answer the model reads as "no such symbol".
    #[test]
    fn graph_lookup_waits_for_a_scan_that_is_still_running() {
        let workspace = workspace_with_a_symbol();
        let tools = ToolExecutor::new(workspace.path());

        // A scan that takes its time, and publishes a symbol that scanning
        // the workspace would not turn up.
        let scan = tools.graph_scans.begin();
        let scan_store = tools.graph_store.clone();
        let scanner = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(250));
            let mut store = GraphStore::new();
            store.add_node(ExtractedNode {
                id: "published_symbol".to_string(),
                label: "published_symbol".to_string(),
                source_file: "src/lib.rs".to_string(),
                source_location: "1".to_string(),
                kind: "function".to_string(),
            });
            *scan_store.lock().expect("store") = Some(store);
            // Published before the guard drops, so a waiter that sees no scan
            // running can take the store as final.
            drop(scan);
        });

        let started = std::time::Instant::now();
        let answer = lookup(&tools, "published_symbol");
        let waited = started.elapsed();
        scanner.join().expect("scanner");

        assert!(
            waited >= std::time::Duration::from_millis(200),
            "expected the lookup to wait for the scan, answered after {waited:?}"
        );
        assert!(
            answer.contains("published_symbol") && !answer.contains("No Knowledge Graph"),
            "expected the scan's own graph, got {answer:?}"
        );
    }

    /// Nothing is scanning this workspace — a `-p` one-shot starts no
    /// background scan — so the lookup scans it itself and answers from the
    /// graph it just built.
    #[test]
    fn graph_lookup_scans_the_workspace_when_nothing_else_is_building_it() {
        let workspace = workspace_with_a_symbol();
        let tools = ToolExecutor::new(workspace.path());
        assert!(tools.graph_store.lock().expect("store").is_none());

        let answer = lookup(&tools, "scanned_symbol");
        forget_scan(workspace.path());

        assert!(
            answer.contains("scanned_symbol"),
            "expected the symbol it scanned, got {answer:?}"
        );
        // The graph it built stays live for the next lookup, and it no longer
        // counts as a scan in flight.
        assert!(tools.graph_store.lock().expect("store").is_some());
        assert!(!tools.graph_scans.is_scanning());
        assert!(
            tools
                .graph_file_hashes
                .lock()
                .unwrap()
                .values()
                .all(|hash| hash.len() == 64)
        );
    }

    /// A symbol the graph does not hold is still reported as missing — the
    /// wait is not a licence to answer "not built yet" in disguise.
    #[test]
    fn a_missing_symbol_is_reported_as_missing() {
        let workspace = workspace_with_a_symbol();
        let tools = ToolExecutor::new(workspace.path());

        let answer = lookup(&tools, "no_such_symbol_anywhere");
        forget_scan(workspace.path());

        assert!(
            answer.contains("No symbol matching"),
            "expected a not-found answer, got {answer:?}"
        );
    }
}

#[cfg(test)]
mod semantic_search_tests {
    use super::*;
    use axum::{Json, Router, extract::State, routing::post};

    type Requests = Arc<Mutex<Vec<Value>>>;

    async fn embed(State(requests): State<Requests>, Json(request): Json<Value>) -> Json<Value> {
        let data: Vec<_> = request["input"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, _)| json!({"index": index, "embedding": [1.0, 0.0]}))
            .collect();
        requests.lock().unwrap().push(request);
        Json(json!({"data": data}))
    }

    struct CacheCleanup(PathBuf);
    impl Drop for CacheCleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
            if let Some(parent) = self.0.parent() {
                let _ = fs::remove_dir(parent);
            }
        }
    }

    #[tokio::test]
    async fn semantic_search_works_without_an_embeddings_server() {
        let workspace = tempfile::tempdir().unwrap();
        let _cache = CacheCleanup(crate::embeddings::EmbeddingIndex::cache_dir(
            workspace.path(),
        ));
        fs::write(
            workspace.path().join("retry.rs"),
            "pub fn retry_backoff() {}\n",
        )
        .unwrap();
        let tools = ToolExecutor::new_read_only(workspace.path());
        tools
            .configure_semantic_search(None, false, 1, 16384)
            .unwrap();
        let answer = tools
            .execute(
                "semantic_search",
                json!({"query": "retry_backoff"}).as_object().unwrap(),
            )
            .await
            .unwrap();
        assert!(
            answer.contains("retry_backoff") && answer.contains("retry.rs:1"),
            "{answer}"
        );
    }

    #[tokio::test]
    async fn semantic_search_reports_unavailable_and_rejects_invalid_queries() {
        let workspace = tempfile::tempdir().unwrap();
        let tools = ToolExecutor::new_read_only(workspace.path());
        for arguments in [json!({}), json!({"query": "  "}), json!({"query": 42})] {
            let err = tools
                .execute("semantic_search", arguments.as_object().unwrap())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("query"));
        }
        let err = tools
            .execute(
                "semantic_search",
                json!({"query": "retry"}).as_object().unwrap(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no embeddings-capable server"));
    }

    #[tokio::test]
    async fn semantic_search_refreshes_edits_and_deletions_and_reuses_unchanged_embeddings() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Requests::default();
        let app = Router::new()
            .route("/v1/embeddings", post(embed))
            .with_state(requests.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let workspace = tempfile::tempdir().unwrap();
        let _cache = CacheCleanup(crate::embeddings::EmbeddingIndex::cache_dir(
            workspace.path(),
        ));
        let file = workspace.path().join("retry.rs");
        fs::write(&file, "pub fn retry_backoff() {}\n").unwrap();
        let tools = ToolExecutor::new_read_only(workspace.path());
        let profile = crate::config::LlmConfiguration {
            endpoint,
            model: "chat-model".into(),
            role: "all".into(),
            api_key: None,
            request_timeout_seconds: 5,
            max_tool_rounds: 1,
            review_max_tokens: 0,
            code_max_tokens: 0,
            system_prompt: String::new(),
            model_verbosity: None,
            review_confidence_threshold: 0,
            security_confidence_threshold: 0,
            review_sanitize: Default::default(),
            review_sanitize_timeout: 1_800,
            review_sanitize_alloc_failures: 8,
        };
        tools
            .configure_semantic_search(Some(&profile), true, 1, 16384)
            .unwrap();
        let arguments = json!({"query": "retry delay"});
        let answer = tools
            .clone()
            .execute("semantic_search", arguments.as_object().unwrap())
            .await
            .unwrap();
        assert!(
            answer.contains("retry_backoff") && answer.contains("retry.rs:1"),
            "{answer}"
        );
        assert_eq!(requests.lock().unwrap().len(), 2);

        tools
            .execute("semantic_search", arguments.as_object().unwrap())
            .await
            .unwrap();
        {
            let requests = requests.lock().unwrap();
            assert_eq!(
                requests.len(),
                3,
                "unchanged code should only embed the query"
            );
            assert_eq!(requests[2]["input"], json!(["retry delay"]));
            assert!(
                requests
                    .iter()
                    .all(|request| request["model"] == "embeddings")
            );
        }

        fs::write(&file, "pub fn exponential_delay() {}\n").unwrap();
        let answer = tools
            .execute("semantic_search", arguments.as_object().unwrap())
            .await
            .unwrap();
        assert!(answer.contains("exponential_delay"), "{answer}");
        assert!(!answer.contains("retry_backoff"), "{answer}");

        fs::remove_file(&file).unwrap();
        let answer = tools
            .execute("semantic_search", arguments.as_object().unwrap())
            .await
            .unwrap();
        assert!(answer.contains("No semantic matches"), "{answer}");
        assert_eq!(
            requests.lock().unwrap().len(),
            5,
            "an empty index should not embed the query"
        );

        tools
            .configure_semantic_search(None, false, 1, 16384)
            .unwrap();
        // With no embeddings server, search degrades to lexical-only rather
        // than failing outright.
        let answer = tools
            .execute("semantic_search", arguments.as_object().unwrap())
            .await
            .unwrap();
        assert!(answer.contains("No semantic matches"), "{answer}");
        server.abort();
    }
}

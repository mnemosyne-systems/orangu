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

//! Custom subagents defined in Markdown files, in the format Claude Code
//! uses: YAML frontmatter with `name`, `description`, and optionally `tools`
//! and `model`, followed by the agent's system prompt.
//!
//! ```markdown
//! ---
//! name: api-designer
//! description: Use this agent when designing new APIs.
//! tools: Read, Write, Edit, Bash, Glob, Grep
//! model: sonnet
//! ---
//!
//! You are a senior API designer...
//! ```
//!
//! Files are found recursively under `~/.claude/agents`, `~/.orangu/agents`,
//! `<workspace>/.claude/agents` and `<workspace>/.orangu/agents`, so agents
//! installed for Claude Code work as they are, and a whole collection can be
//! cloned into one of them. A Markdown file without a `name` and a
//! `description` (a README, say) is not an agent and is skipped, as are
//! hidden directories and `commands`, `tools`, `skills` and `hooks`
//! directories, where a collection keeps its other extensions. No agent may
//! be called `list`. A project
//! agent overrides a user agent of the same name.
//!
//! Claude Code's tool names map onto orangu's tools (`WebSearch` onto
//! `web_search`, when `[orangu].web_search` is set); MCP tools keep their
//! names (`mcp__<server>__<tool>`, or `mcp__<server>` for a whole server);
//! the rest (`Task`) are reported as unsupported and left out. `mcpServers`
//! names configured servers or defines Streamable HTTP ones for the agent.
//! `model` maps onto a role: `haiku` asks for the small `explorer` model,
//! everything else for `code`. `disallowedTools` takes tools away before `tools` applies;
//! `maxTurns` is the budget of model requests; `memory` gives the agent a
//! `MEMORY.md` that outlives the task; `omitClaudeMd: true` leaves the
//! project's `AGENTS.md` rules out of its prompt. orangu also reads two
//! fields of its own: `role` (any of `all`, `code`, `review`, `explorer`)
//! and `max_tool_rounds`, which wins over `maxTurns`.

use serde::Deserialize;
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Model requests a custom agent may make, unless its file says otherwise.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 15;

/// The most model requests a file may ask for.
pub const MAX_TOOL_ROUNDS: usize = 50;

/// How deep below an agents directory files are looked for: deep enough for
/// a cloned collection (`<repo>/categories/<category>/<agent>.md`).
const MAX_DEPTH: usize = 5;

/// Directories inside an agents tree that hold Claude Code's other kinds of
/// extension, whose Markdown files look like agents but are not: a cloned
/// collection ships its slash commands next to its agents.
const NOT_AGENT_DIRS: &[&str] = &["commands", "tools", "skills", "hooks"];

/// The name `run_agent` uses to list agents, so no agent can have it.
const RESERVED_NAME: &str = "list";

/// The roles a file's `role` field may name.
const ROLES: &[&str] = &["all", "code", "review", "explorer"];

/// Every orangu tool a custom agent can be given, by its own name.
const ORANGU_TOOLS: &[&str] = &[
    "show_file",
    "expand_context",
    "list_directory",
    "search_text",
    "create_file",
    "create_directory",
    "modify_file",
    "move_file",
    "delete_file",
    "move_directory",
    "delete_directory",
    "convert_document",
    "run_shell_command",
    "fetch_url",
    "web_search",
    "browse",
    "use_skill",
    "ssh_login",
    "ssh_copy",
    "ssh_logout",
];

/// The SSH tools, which an agent gets only by naming them in `tools`.
pub const SSH_TOOLS: &[&str] = &["ssh_login", "ssh_copy", "ssh_logout"];

/// The tools that start subagents, which `Agent` (Claude Code's `Task`)
/// stands for.
const SUBAGENT_TOOLS: &[&str] = &[
    "run_agent",
    "explore_repository",
    "write_tests",
    "write_docs",
    "write_document",
    "research_web",
];

/// The agents `Agent(a, b)` (or `Task(a, b)`) names, or `None` when `name`
/// is not of that form.
fn agent_list(name: &str) -> Option<Vec<String>> {
    let inner = name
        .strip_prefix("Agent(")
        .or_else(|| name.strip_prefix("Task("))?
        .strip_suffix(')')?;
    Some(
        inner
            .split(',')
            .map(str::trim)
            .filter(|agent| !agent.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// The Claude Code tool names that stand for orangu tool `tool`.
pub fn claude_code_names(tool: &str) -> Vec<&'static str> {
    [
        "Read",
        "Glob",
        "LS",
        "Grep",
        "Write",
        "Edit",
        "MultiEdit",
        "Bash",
        "WebFetch",
        "WebSearch",
        "Skill",
        "Agent",
    ]
    .into_iter()
    .filter(|name| map_tool(name).is_some_and(|tools| tools.contains(&tool)))
    .collect()
}

/// The orangu tools a Claude Code tool name stands for, or `None` when
/// orangu has no equivalent.
fn map_tool(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "Read" => &["show_file", "expand_context"],
        "Glob" | "LS" => &["list_directory"],
        "Grep" => &["search_text"],
        "Write" => &["create_file", "create_directory", "convert_document"],
        // Claude Code changes files only through Write, Edit and Bash: moving
        // counts as editing, and deleting is done with the shell.
        "Edit" | "MultiEdit" => &["modify_file", "move_file", "move_directory"],
        "Bash" => &["run_shell_command", "delete_file", "delete_directory"],
        "WebFetch" => &["fetch_url", "browse"],
        "WebSearch" => &["web_search", "browse"],
        "Skill" => &["use_skill"],
        "Agent" | "Task" => SUBAGENT_TOOLS,
        _ => {
            return ORANGU_TOOLS
                .iter()
                .position(|tool| *tool == name)
                .map(|index| std::slice::from_ref(&ORANGU_TOOLS[index]));
        }
    })
}

/// The most of `MEMORY.md` an agent's prompt carries: its first lines, up
/// to a size, as Claude Code does.
const MEMORY_PROMPT_LINES: usize = 200;
const MEMORY_PROMPT_BYTES: usize = 25 * 1024;

/// Where an agent with `memory` keeps its notes between runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// `~/.orangu/agent-memory/<agent>/`: every project.
    User,
    /// `<workspace>/.orangu/agent-memory/<agent>/`: this project, meant to
    /// be committed with it.
    Project,
    /// `<workspace>/.orangu/agent-memory-local/<agent>/`: this project, kept
    /// out of version control.
    Local,
}

impl MemoryScope {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Self::User),
            "project" => Some(Self::Project),
            "local" => Some(Self::Local),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Local => "local",
        }
    }
}

impl CustomAgent {
    /// The agent's memory directory, if it has `memory`.
    pub fn memory_dir(&self, workspace: &Path) -> Option<PathBuf> {
        let base = match self.memory? {
            MemoryScope::User => home::home_dir()?.join(".orangu").join("agent-memory"),
            MemoryScope::Project => workspace.join(".orangu").join("agent-memory"),
            MemoryScope::Local => workspace.join(".orangu").join("agent-memory-local"),
        };
        // A plugin agent's `:` is no file name character everywhere.
        Some(base.join(self.name.replace(':', "-")))
    }
}

/// The `MEMORY.md` in `dir`, as much of it as an agent's prompt carries; an
/// empty string when there is none yet.
pub fn memory_for_prompt(dir: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(dir.join("MEMORY.md")) else {
        return String::new();
    };
    let mut kept = String::new();
    for line in text.lines().take(MEMORY_PROMPT_LINES) {
        if kept.len() + line.len() + 1 > MEMORY_PROMPT_BYTES {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    kept
}

/// What an agent with memory is told, after its own prompt.
pub fn memory_instructions(agent: &CustomAgent, dir: &Path) -> String {
    let scope = agent.memory.map_or("", MemoryScope::name);
    format!(
        "\n\n## Memory\n\nYou have a persistent memory ({scope} scope, {}) that outlives this task: \
         notes for your own future runs. Its current content is below; it is empty on your first \
         run. When you learn something worth knowing next time (where things are, conventions, \
         decisions, pitfalls), call save_memory with the whole new content of the memory, kept \
         short and organised. Never store secrets.\n\n<memory>\n{}</memory>",
        dir.join("MEMORY.md").display(),
        memory_for_prompt(dir)
    )
}

/// One entry of an agent's `mcpServers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentMcpServer {
    /// A server by name: one configured as `[mcp.<name>]`.
    Reference(String),
    /// A Streamable HTTP server the agent defines itself, connected for its
    /// runs only.
    Http { name: String, url: String },
    /// A server orangu cannot connect to (`stdio`, `sse`), with why.
    Unsupported { name: String, reason: String },
}

impl AgentMcpServer {
    pub fn name(&self) -> &str {
        match self {
            Self::Reference(name) | Self::Http { name, .. } | Self::Unsupported { name, .. } => {
                name
            }
        }
    }

    /// One server from its name and inline definition.
    fn inline(name: &str, definition: &serde_yaml::Value) -> Self {
        let name = name.trim().to_string();
        let field = |key: &str| {
            definition
                .get(key)
                .and_then(serde_yaml::Value::as_str)
                .map(|value| value.trim().to_string())
        };
        let kind = field("type").map(|kind| kind.to_ascii_lowercase());
        match (kind.as_deref(), field("url")) {
            (
                None | Some("http") | Some("streamable-http") | Some("streamable_http"),
                Some(url),
            ) if !url.is_empty() => Self::Http { name, url },
            (Some(kind), _) if kind != "http" => Self::Unsupported {
                name,
                reason: format!("{kind} servers are not supported, only HTTP"),
            },
            _ if field("command").is_some() => Self::Unsupported {
                name,
                reason: "stdio servers are not supported, only HTTP".to_string(),
            },
            _ => Self::Unsupported {
                name,
                reason: "no url".to_string(),
            },
        }
    }
}

/// The servers an `mcpServers` value lists: a list of names and
/// `name: {definition}` maps, or one map of them.
fn mcp_servers(value: &serde_yaml::Value) -> Vec<AgentMcpServer> {
    fn from_map(map: &serde_yaml::Mapping, servers: &mut Vec<AgentMcpServer>) {
        for (name, definition) in map {
            if let Some(name) = name.as_str() {
                servers.push(AgentMcpServer::inline(name, definition));
            }
        }
    }
    let mut servers = Vec::new();
    match value {
        serde_yaml::Value::Sequence(items) => {
            for item in items {
                match item {
                    serde_yaml::Value::String(name) if !name.trim().is_empty() => {
                        servers.push(AgentMcpServer::Reference(name.trim().to_string()));
                    }
                    serde_yaml::Value::Mapping(map) => from_map(map, &mut servers),
                    _ => {}
                }
            }
        }
        serde_yaml::Value::Mapping(map) => from_map(map, &mut servers),
        serde_yaml::Value::String(text) => servers.extend(
            text.split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(|name| AgentMcpServer::Reference(name.to_string())),
        ),
        _ => {}
    }
    servers
}

/// Whether `set` lets tool `name` through: by its own name, or, for an MCP
/// tool (`mcp__<server>__<tool>`), by its server's (`mcp__<server>`), the
/// way Claude Code names a whole server.
pub fn lists_tool(set: &HashSet<String>, name: &str) -> bool {
    set.contains(name)
        || name
            .strip_prefix("mcp__")
            .and_then(|rest| rest.split_once("__"))
            .is_some_and(|(server, _)| set.contains(&format!("mcp__{server}")))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSource {
    User,
    Project,
    /// Defined for one run with `--agents`.
    Session,
    /// From a plugin: named `<plugin>:<agent>`.
    Plugin,
}

impl std::fmt::Display for AgentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AgentSource::User => "user",
            AgentSource::Project => "project",
            AgentSource::Session => "session",
            AgentSource::Plugin => "plugin",
        })
    }
}

#[derive(Debug, Clone)]
pub struct CustomAgent {
    pub name: String,
    pub description: String,
    /// The agent's system prompt: the file's body.
    pub prompt: String,
    /// The `tools` field as written, or `None` for every tool.
    pub tools: Option<Vec<String>>,
    /// The `disallowedTools` field as written: taken away first.
    pub disallowed_tools: Vec<String>,
    pub model: Option<String>,
    role: Option<String>,
    max_tool_rounds: Option<usize>,
    pub path: PathBuf,
    pub source: AgentSource,
    /// `omitClaudeMd: true`: the agent runs without the project's
    /// `AGENTS.md` rules.
    pub omit_project_rules: bool,
    /// `memory`: where the agent keeps notes between runs, if anywhere.
    pub memory: Option<MemoryScope>,
    /// `mcpServers`: the MCP servers the agent names or defines.
    pub mcp_servers: Vec<AgentMcpServer>,
    /// `skills`: skills whose instructions start in the agent's prompt.
    pub skills: Vec<String>,
    /// `permissionMode`: `plan` makes the agent read-only; orangu's
    /// subagents never ask, so the other modes change nothing.
    pub permission_mode: Option<String>,
    /// `hooks`: commands run around the agent's tool calls.
    pub hooks: crate::agents::agent_hooks::AgentHooks,
    /// `background: true`: the agent always runs in the background.
    pub background: bool,
    /// `effort`, as written: orangu sends no effort setting.
    pub effort: Option<String>,
    /// `color`, for listings.
    pub color: Option<String>,
    /// `initialPrompt`: the first turn of a conversation run as the agent
    /// (`--agent`).
    pub initial_prompt: Option<String>,
    /// `isolation: worktree`: the agent works in a temporary git worktree.
    pub isolated: bool,
}

/// The orangu tools a list of tool names stands for. MCP tools keep their names (`mcp__<server>__<tool>`, or `mcp__<server>`
/// for all of a server's), which are orangu's names for them too.
fn mapped(names: &[String]) -> HashSet<String> {
    let mut tools: HashSet<String> = names
        .iter()
        .filter_map(|name| map_tool(name))
        .flatten()
        .map(|tool| tool.to_string())
        .collect();
    tools.extend(names.iter().filter(|name| is_mcp_name(name)).cloned());
    // `Agent(a, b)`: the built-in subagents it names, and `run_agent` for
    // the custom ones.
    for agents in names.iter().filter_map(|name| agent_list(name)) {
        for agent in agents {
            match SUBAGENT_TOOLS.iter().find(|tool| **tool == agent) {
                Some(tool) => tools.insert(tool.to_string()),
                None => tools.insert("run_agent".to_string()),
            };
        }
    }
    tools
}

/// Whether `name` names an MCP server or tool.
fn is_mcp_name(name: &str) -> bool {
    name.strip_prefix("mcp__")
        .is_some_and(|rest| !rest.is_empty())
}

/// Tools that only read: an agent with nothing else cannot change the
/// workspace.
fn reads_only(tool: &str) -> bool {
    matches!(
        tool,
        "show_file"
            | "expand_context"
            | "list_directory"
            | "search_text"
            | "fetch_url"
            | "web_search"
            | "browse"
            | "use_skill"
    )
}

impl CustomAgent {
    /// The orangu tools this agent may use, or `None` for all of them (less
    /// [`Self::denied_tools`]). `disallowedTools` applies before `tools`.
    pub fn allowed_tools(&self) -> Option<HashSet<String>> {
        let denied = self.denied_tools();
        let tools = self.tools.as_ref()?;
        // The servers it names in `mcpServers` come with the agent.
        let servers = self
            .mcp_servers
            .iter()
            .map(|server| format!("mcp__{}", server.name()));
        Some(
            mapped(tools)
                .into_iter()
                .chain(servers)
                .filter(|tool| !denied.contains(tool))
                .collect(),
        )
    }

    /// The orangu tools `disallowedTools` takes away, and with
    /// `permissionMode: plan` every tool that changes the workspace.
    pub fn denied_tools(&self) -> HashSet<String> {
        let mut denied = mapped(&self.disallowed_tools);
        if self.plans() {
            denied.extend(
                ORANGU_TOOLS
                    .iter()
                    .filter(|tool| !reads_only(tool))
                    .map(|tool| tool.to_string()),
            );
        }
        denied
    }

    /// `permissionMode: plan`: the agent looks but does not change.
    pub fn plans(&self) -> bool {
        self.permission_mode
            .as_deref()
            .is_some_and(|mode| mode.eq_ignore_ascii_case("plan"))
    }

    /// Whether the agent may use orangu tool `tool`.
    pub fn may_use(&self, tool: &str) -> bool {
        // These reach other machines, so an agent has to name them: having no
        // `tools` field, which allows everything else, does not grant them.
        if SSH_TOOLS.contains(&tool) && self.tools.is_none() {
            return false;
        }
        !lists_tool(&self.denied_tools(), tool)
            && self
                .allowed_tools()
                .is_none_or(|allowed| lists_tool(&allowed, tool))
    }

    /// Whether the agent may start subagents, and which: `None` when it may
    /// not (its `tools` do not list `Agent` or `Task`, or `disallowedTools`
    /// does), `Some(None)` for any, `Some(Some(names))` for those
    /// `Agent(a, b)` names. Without a `tools` field an agent does not nest:
    /// asking for it is explicit.
    pub fn starts_agents(&self) -> Option<Option<HashSet<String>>> {
        if self
            .disallowed_tools
            .iter()
            .any(|tool| tool == "Agent" || tool == "Task")
        {
            return None;
        }
        let tools = self.tools.as_ref()?;
        if tools.iter().any(|tool| tool == "Agent" || tool == "Task") {
            return Some(None);
        }
        let named: HashSet<String> = tools
            .iter()
            .filter_map(|tool| agent_list(tool))
            .flatten()
            .collect();
        (!named.is_empty()).then_some(Some(named))
    }
    /// The tools in `tools` that orangu cannot give it.
    pub fn unsupported_tools(&self) -> Vec<&str> {
        self.tools
            .iter()
            .flatten()
            .map(String::as_str)
            .filter(|tool| {
                map_tool(tool).is_none() && !is_mcp_name(tool) && agent_list(tool).is_none()
            })
            .collect()
    }

    /// Whether the agent may change the workspace.
    pub fn writable(&self) -> bool {
        ORANGU_TOOLS
            .iter()
            .any(|tool| !reads_only(tool) && self.may_use(tool))
    }

    /// The role that picks the agent's model.
    pub fn role(&self) -> &str {
        if let Some(role) = &self.role {
            return role;
        }
        match self.model.as_deref() {
            Some(model) if model.eq_ignore_ascii_case("haiku") => "explorer",
            _ => "code",
        }
    }

    pub fn max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
            .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS)
            .clamp(1, MAX_TOOL_ROUNDS)
    }
}

/// The `tools` field: Claude Code writes a comma-separated string, but a
/// YAML list means the same.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolList {
    Text(String),
    List(Vec<String>),
}

/// `text` split at its commas, except those inside parentheses: `Read,
/// Agent(worker, researcher)` is two tools.
fn split_outside_parentheses(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    parts.push(current);
    parts
}

impl ToolList {
    fn names(self) -> Vec<String> {
        let names: Vec<String> = match self {
            ToolList::Text(text) => split_outside_parentheses(&text),
            ToolList::List(list) => list,
        };
        names
            .into_iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    tools: Option<ToolList>,
    model: Option<String>,
    role: Option<String>,
    max_tool_rounds: Option<usize>,
    #[serde(rename = "maxTurns")]
    max_turns: Option<usize>,
    #[serde(rename = "disallowedTools")]
    disallowed_tools: Option<ToolList>,
    #[serde(rename = "omitClaudeMd")]
    omit_claude_md: Option<bool>,
    memory: Option<String>,
    #[serde(rename = "mcpServers")]
    mcp_servers: Option<serde_yaml::Value>,
    skills: Option<ToolList>,
    #[serde(rename = "permissionMode")]
    permission_mode: Option<String>,
    hooks: Option<serde_yaml::Value>,
    background: Option<bool>,
    effort: Option<serde_yaml::Value>,
    color: Option<String>,
    #[serde(rename = "initialPrompt")]
    initial_prompt: Option<String>,
    isolation: Option<String>,
}

/// The frontmatter and body of `content`, if it opens with `---`.
fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let content = content.trim_start_matches('\u{feff}');
    let rest = content
        .strip_prefix("---\r\n")
        .or_else(|| content.strip_prefix("---\n"))?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

/// The frontmatter as plain `key: value` lines, for a file whose YAML does
/// not parse (an unquoted description containing `: `, for one). Every value
/// is taken as a string, which is all a description needs.
fn lenient_frontmatter(text: &str) -> Frontmatter {
    let mut frontmatter = Frontmatter {
        name: None,
        description: None,
        tools: None,
        model: None,
        role: None,
        max_tool_rounds: None,
        omit_claude_md: None,
        memory: None,
        max_turns: None,
        disallowed_tools: None,
        mcp_servers: None,
        skills: None,
        permission_mode: None,
        hooks: None,
        background: None,
        effort: None,
        color: None,
        initial_prompt: None,
        isolation: None,
    };
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        match key.trim() {
            "name" => frontmatter.name = Some(value),
            "description" => frontmatter.description = Some(value),
            "tools" => frontmatter.tools = Some(ToolList::Text(value)),
            "model" => frontmatter.model = Some(value),
            "role" => frontmatter.role = Some(value),
            "max_tool_rounds" => frontmatter.max_tool_rounds = value.parse().ok(),
            "omitClaudeMd" => frontmatter.omit_claude_md = value.parse().ok(),
            "memory" => frontmatter.memory = Some(value),
            "maxTurns" => frontmatter.max_turns = value.parse().ok(),
            "disallowedTools" => frontmatter.disallowed_tools = Some(ToolList::Text(value)),
            "skills" => frontmatter.skills = Some(ToolList::Text(value)),
            "permissionMode" => frontmatter.permission_mode = Some(value),
            "background" => frontmatter.background = value.parse().ok(),
            "effort" => frontmatter.effort = Some(serde_yaml::Value::String(value)),
            "color" => frontmatter.color = Some(value),
            "initialPrompt" => frontmatter.initial_prompt = Some(value),
            "isolation" => frontmatter.isolation = Some(value),
            _ => {}
        }
    }
    frontmatter
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.eq_ignore_ascii_case(RESERVED_NAME)
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// The agent `content` defines, or `None` when it is not an agent file.
pub fn parse_agent(content: &str, path: &Path, source: AgentSource) -> Option<CustomAgent> {
    let (frontmatter, body) = split_frontmatter(content)?;
    let frontmatter = serde_yaml::from_str::<Frontmatter>(frontmatter)
        .unwrap_or_else(|_| lenient_frontmatter(frontmatter));
    from_frontmatter(frontmatter, body, path, source)
}

/// The agent `frontmatter` and its prompt `body` define, if they are one.
fn from_frontmatter(
    frontmatter: Frontmatter,
    body: &str,
    path: &Path,
    source: AgentSource,
) -> Option<CustomAgent> {
    let name = frontmatter.name?.trim().to_string();
    let description = frontmatter.description?.trim().to_string();
    let prompt = body.trim().to_string();
    if !valid_name(&name) || description.is_empty() || prompt.is_empty() {
        return None;
    }
    let role = frontmatter
        .role
        .map(|role| role.trim().to_ascii_lowercase())
        .filter(|role| ROLES.contains(&role.as_str()));
    Some(CustomAgent {
        name,
        description,
        prompt,
        tools: frontmatter.tools.map(ToolList::names),
        disallowed_tools: frontmatter
            .disallowed_tools
            .map(ToolList::names)
            .unwrap_or_default(),
        model: frontmatter.model.map(|model| model.trim().to_string()),
        role,
        max_tool_rounds: frontmatter.max_tool_rounds.or(frontmatter.max_turns),
        omit_project_rules: frontmatter.omit_claude_md.unwrap_or(false),
        memory: frontmatter.memory.as_deref().and_then(MemoryScope::parse),
        mcp_servers: frontmatter
            .mcp_servers
            .as_ref()
            .map(mcp_servers)
            .unwrap_or_default(),
        skills: frontmatter.skills.map(ToolList::names).unwrap_or_default(),
        permission_mode: frontmatter
            .permission_mode
            .map(|mode| mode.trim().to_string())
            .filter(|mode| !mode.is_empty()),
        hooks: frontmatter
            .hooks
            .as_ref()
            .map(crate::agents::agent_hooks::AgentHooks::parse)
            .unwrap_or_default(),
        background: frontmatter.background.unwrap_or(false),
        effort: frontmatter.effort.map(|effort| match effort {
            serde_yaml::Value::String(text) => text.trim().to_string(),
            other => serde_yaml::to_string(&other)
                .unwrap_or_default()
                .trim()
                .to_string(),
        }),
        color: frontmatter.color.map(|color| color.trim().to_string()),
        initial_prompt: frontmatter
            .initial_prompt
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty()),
        isolated: frontmatter
            .isolation
            .is_some_and(|isolation| isolation.trim().eq_ignore_ascii_case("worktree")),
        path: path.to_path_buf(),
        source,
    })
}

/// Agents defined as JSON (`--agents`), as Claude Code takes them: an object
/// from agent name to its fields, the frontmatter's names, with the system
/// prompt as `prompt`.
///
/// ```json
/// {"reviewer": {"description": "Reviews code.", "prompt": "You review.", "tools": ["Read", "Grep"]}}
/// ```
pub fn parse_session_agents(json: &str) -> anyhow::Result<Vec<CustomAgent>> {
    use anyhow::Context;
    let value: serde_json::Value = serde_json::from_str(json).context("--agents is not JSON")?;
    let object = value
        .as_object()
        .context("--agents must be an object from agent name to its definition")?;
    let mut agents = Vec::new();
    for (name, definition) in object {
        let mut fields = definition
            .as_object()
            .with_context(|| format!("agent '{name}' must be an object"))?
            .clone();
        fields.insert("name".into(), serde_json::Value::String(name.clone()));
        let prompt = fields
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let frontmatter: Frontmatter = serde_json::from_value(serde_json::Value::Object(fields))
            .with_context(|| format!("agent '{name}'"))?;
        let agent = from_frontmatter(
            frontmatter,
            &prompt,
            Path::new("--agents"),
            AgentSource::Session,
        )
        .with_context(|| {
            format!("agent '{name}' needs a valid name, a description and a prompt")
        })?;
        agents.push(agent);
    }
    Ok(agents)
}

/// The names of the custom agents of `workspace`, for completing a mention:
/// read once per workspace and kept current like [`LiveAgents`].
pub fn agent_names(workspace: &Path) -> Vec<String> {
    static WATCHED: std::sync::Mutex<Vec<(PathBuf, std::sync::Arc<LiveAgents>)>> =
        std::sync::Mutex::new(Vec::new());
    let live = {
        let Ok(mut watched) = WATCHED.lock() else {
            return Vec::new();
        };
        match watched.iter().find(|(path, _)| path == workspace) {
            Some((_, live)) => live.clone(),
            None => {
                let live = std::sync::Arc::new(LiveAgents::watch(workspace));
                watched.push((workspace.to_path_buf(), live.clone()));
                live
            }
        }
    };
    live.current()
        .all()
        .iter()
        .map(|agent| agent.name.clone())
        .collect()
}

/// The agents `--agents` defined, and the agent `--agent` runs the main
/// conversation as: set once, at startup, for every tab.
static SESSION_AGENTS: std::sync::OnceLock<Vec<CustomAgent>> = std::sync::OnceLock::new();
static MAIN_AGENT: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Adds `agents` to every workspace's, ahead of agent files of the same name.
pub fn set_session_agents(agents: Vec<CustomAgent>) {
    let _ = SESSION_AGENTS.set(agents);
}

pub fn session_agents() -> &'static [CustomAgent] {
    SESSION_AGENTS.get().map_or(&[], Vec::as_slice)
}

/// Runs the main conversation as agent `name`.
pub fn set_main_agent(name: String) {
    let _ = MAIN_AGENT.set(name);
}

pub fn main_agent() -> Option<&'static str> {
    MAIN_AGENT.get().map(String::as_str)
}
/// The Markdown files under an agents directory that may define agents, in
/// path order.
fn agent_files(dir: &Path) -> impl Iterator<Item = walkdir::DirEntry> {
    WalkDir::new(dir)
        .min_depth(1)
        .max_depth(MAX_DEPTH)
        .follow_links(true)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !name.starts_with('.')
                && !(entry.file_type().is_dir() && NOT_AGENT_DIRS.contains(&name.as_ref()))
        })
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_file()
                && entry.path().extension().and_then(|e| e.to_str()) == Some("md")
        })
}

/// What changes when an agent file is added, removed or edited: every
/// agent file's path, size and modification time, hashed.
fn fingerprint(dirs: &[(PathBuf, AgentSource)]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (dir, _) in dirs {
        for entry in agent_files(dir) {
            entry.path().hash(&mut hasher);
            if let Ok(metadata) = entry.metadata() {
                metadata.len().hash(&mut hasher);
                metadata.modified().ok().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// How often, at most, [`LiveAgents`] looks at the agent directories.
const RELOAD_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// The custom agents of a workspace, kept current: when they are asked for,
/// and the agent directories have changed since they were read (checked at
/// most every [`RELOAD_CHECK_INTERVAL`]), they are read again. An agent file
/// added, edited or removed applies to the next request without a restart.
#[derive(Debug, Default)]
pub struct LiveAgents {
    /// The directories watched; none for a fixed set.
    dirs: Vec<(PathBuf, AgentSource)>,
    /// The workspace whose directories they are: looked up again at each
    /// check, since installing a plugin adds one.
    workspace: Option<PathBuf>,
    /// Agents that win over the files': the session's (`--agents`).
    extra: Vec<CustomAgent>,
    state: std::sync::Mutex<LiveState>,
}

#[derive(Debug, Default)]
struct LiveState {
    agents: std::sync::Arc<AgentRegistry>,
    fingerprint: u64,
    checked: Option<std::time::Instant>,
}

impl LiveAgents {
    /// The agents of `workspace`, reloaded when their files change.
    /// The session's own agents (`--agents`) win over files of the same name.
    pub fn watch(workspace: &Path) -> Self {
        Self {
            workspace: Some(workspace.to_path_buf()),
            ..Self::watch_directories(
                AgentRegistry::directories(workspace),
                session_agents().to_vec(),
            )
        }
    }

    pub fn watch_directories(dirs: Vec<(PathBuf, AgentSource)>, extra: Vec<CustomAgent>) -> Self {
        let state = LiveState {
            agents: std::sync::Arc::new(AgentRegistry::from_directories(&dirs).with(&extra)),
            fingerprint: fingerprint(&dirs),
            checked: Some(std::time::Instant::now()),
        };
        Self {
            dirs,
            workspace: None,
            extra,
            state: std::sync::Mutex::new(state),
        }
    }

    /// A set that never changes.
    pub fn fixed(agents: AgentRegistry) -> Self {
        Self {
            dirs: Vec::new(),
            workspace: None,
            extra: Vec::new(),
            state: std::sync::Mutex::new(LiveState {
                agents: std::sync::Arc::new(agents),
                ..LiveState::default()
            }),
        }
    }

    /// The agents as they are now.
    pub fn current(&self) -> std::sync::Arc<AgentRegistry> {
        let Ok(mut state) = self.state.lock() else {
            return std::sync::Arc::default();
        };
        if !self.dirs.is_empty()
            && state
                .checked
                .is_none_or(|checked| checked.elapsed() >= RELOAD_CHECK_INTERVAL)
        {
            state.checked = Some(std::time::Instant::now());
            let dirs = match &self.workspace {
                Some(workspace) => AgentRegistry::directories(workspace),
                None => self.dirs.clone(),
            };
            let now = fingerprint(&dirs);
            if now != state.fingerprint {
                state.fingerprint = now;
                state.agents =
                    std::sync::Arc::new(AgentRegistry::from_directories(&dirs).with(&self.extra));
            }
        }
        state.agents.clone()
    }

    /// Looks again now, however recently it last did: for `/agents`.
    pub fn refresh(&self) -> std::sync::Arc<AgentRegistry> {
        if let Ok(mut state) = self.state.lock() {
            state.checked = None;
        }
        self.current()
    }
}

/// The `agents` directories of the installed plugins, for `workspace`: the
/// plugins Claude Code installed (`~/.claude/plugins/installed_plugins.json`,
/// less those its settings disable) and orangu's own,
/// `~/.orangu/plugins/<plugin>/` and `<workspace>/.orangu/plugins/<plugin>/`.
fn plugin_agent_directories(workspace: &Path) -> Vec<(PathBuf, AgentSource)> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = home::home_dir() {
        roots.extend(claude_code_plugins(&home, workspace));
        roots.extend(plugin_roots(&home.join(".orangu").join("plugins")));
    }
    roots.extend(plugin_roots(&workspace.join(".orangu").join("plugins")));
    roots
        .into_iter()
        .map(|root| (root.join("agents"), AgentSource::Plugin))
        .collect()
}

/// Every directory in `dir`: one plugin each.
fn plugin_roots(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut roots: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    roots.sort();
    roots
}

/// Where Claude Code installed the plugins that apply to `workspace`: every
/// user-scope plugin, and project-scope ones of this project, unless
/// `enabledPlugins` (user or project settings) turns them off.
fn claude_code_plugins(home: &Path, workspace: &Path) -> Vec<PathBuf> {
    let read = |path: PathBuf| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    };
    let Some(installed) = read(home.join(".claude/plugins/installed_plugins.json")) else {
        return Vec::new();
    };
    let settings: Vec<serde_json::Value> = [
        home.join(".claude/settings.json"),
        workspace.join(".claude/settings.json"),
        workspace.join(".claude/settings.local.json"),
    ]
    .into_iter()
    .filter_map(read)
    .collect();
    // The last settings file that names the plugin decides.
    let enabled = |key: &str| {
        settings
            .iter()
            .filter_map(|s| s.get("enabledPlugins")?.get(key)?.as_bool())
            .next_back()
            .unwrap_or(true)
    };
    let mut roots = Vec::new();
    let Some(plugins) = installed.get("plugins").and_then(|p| p.as_object()) else {
        return roots;
    };
    for (key, installs) in plugins {
        if !enabled(key) {
            continue;
        }
        for install in installs.as_array().into_iter().flatten() {
            let scope = install
                .get("scope")
                .and_then(|s| s.as_str())
                .unwrap_or("user");
            let for_here = scope == "user"
                || install
                    .get("projectPath")
                    .and_then(|p| p.as_str())
                    .is_some_and(|p| Path::new(p) == workspace);
            if let Some(path) = install.get("installPath").and_then(|p| p.as_str())
                && for_here
            {
                roots.push(PathBuf::from(path));
            }
        }
    }
    roots
}

/// The name of the plugin whose `agents` directory is `dir`: its
/// `.claude-plugin/plugin.json` `name`, or its directory's, skipping a
/// version directory (`<plugin>/1.0.0/`) as Claude Code installs them.
fn plugin_name(dir: &Path) -> Option<String> {
    let root = dir.parent()?;
    let manifest = std::fs::read_to_string(root.join(".claude-plugin/plugin.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
    if let Some(name) = manifest
        .as_ref()
        .and_then(|m| m.get("name")?.as_str().map(str::to_string))
    {
        return Some(name);
    }
    let own = root.file_name()?.to_string_lossy().to_string();
    if own.starts_with(|c: char| c.is_ascii_digit()) {
        return root
            .parent()?
            .file_name()
            .map(|name| name.to_string_lossy().to_string());
    }
    Some(own)
}

/// `agent` as a plugin in `dir` provides it: named `<plugin>:<agent>`, and,
/// as in Claude Code, without servers of its own.
fn in_plugin(mut agent: CustomAgent, dir: &Path, source: AgentSource) -> CustomAgent {
    if source == AgentSource::Plugin {
        let plugin = plugin_name(dir).unwrap_or_else(|| "plugin".to_string());
        agent.name = format!("{plugin}:{}", agent.name);
        agent.mcp_servers.clear();
        // Nor, as in Claude Code, hooks or a permission mode.
        agent.hooks = crate::agents::agent_hooks::AgentHooks::default();
        agent.permission_mode = None;
    }
    agent
}
/// Every custom agent available in a workspace.
#[derive(Debug, Clone, Default)]
pub struct AgentRegistry {
    agents: Vec<CustomAgent>,
}

impl AgentRegistry {
    /// The directories agents are read from, lowest precedence first.
    pub fn directories(workspace: &Path) -> Vec<(PathBuf, AgentSource)> {
        let mut dirs = plugin_agent_directories(workspace);
        if let Some(home) = home::home_dir() {
            for client in [".claude", ".orangu"] {
                dirs.push((home.join(client).join("agents"), AgentSource::User));
            }
        }
        for client in [".claude", ".orangu"] {
            dirs.push((workspace.join(client).join("agents"), AgentSource::Project));
        }
        dirs
    }

    pub fn discover(workspace: &Path) -> Self {
        Self::from_directories(&Self::directories(workspace))
    }

    /// Agents from `dirs`; for one name, the last directory wins, and within
    /// a directory the first file in path order.
    pub fn from_directories(dirs: &[(PathBuf, AgentSource)]) -> Self {
        let mut agents: Vec<CustomAgent> = Vec::new();
        for (dir, source) in dirs {
            let mut found: Vec<CustomAgent> = Vec::new();
            for entry in agent_files(dir) {
                let path = entry.path();
                let Ok(content) = std::fs::read_to_string(path) else {
                    continue;
                };
                if let Some(agent) =
                    parse_agent(&content, path, *source).map(|agent| in_plugin(agent, dir, *source))
                    && !found.iter().any(|a| a.name == agent.name)
                {
                    found.push(agent);
                }
            }
            for agent in found {
                agents.retain(|existing| existing.name != agent.name);
                agents.push(agent);
            }
        }
        agents.sort_by(|a, b| a.name.cmp(&b.name));
        Self { agents }
    }

    /// This set with `agents` added, each replacing any of the same name.
    pub fn with(mut self, agents: &[CustomAgent]) -> Self {
        for agent in agents {
            self.agents.retain(|existing| existing.name != agent.name);
            self.agents.push(agent.clone());
        }
        self.agents.sort_by(|a, b| a.name.cmp(&b.name));
        self
    }

    pub fn all(&self) -> &[CustomAgent] {
        &self.agents
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// A plugin agent may be named without its plugin while no other agent
    /// has that name.
    pub fn find(&self, name: &str) -> Option<&CustomAgent> {
        let name = name.trim();
        if let Some(agent) = self
            .agents
            .iter()
            .find(|agent| agent.name.eq_ignore_ascii_case(name))
        {
            return Some(agent);
        }
        let mut scoped = self.agents.iter().filter(|agent| {
            agent
                .name
                .split_once(':')
                .is_some_and(|(_, bare)| bare.eq_ignore_ascii_case(name))
        });
        match (scoped.next(), scoped.next()) {
            (Some(agent), None) => Some(agent),
            _ => None,
        }
    }

    /// Up to `limit` agents matching `query`, best first: a word found in the
    /// name counts three times one found in the description. An empty query
    /// lists agents in name order.
    pub fn search(&self, query: &str, limit: usize) -> Vec<&CustomAgent> {
        let words: BTreeSet<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.len() > 1)
            .map(str::to_lowercase)
            .collect();
        if words.is_empty() {
            return self.agents.iter().take(limit).collect();
        }
        let mut scored: Vec<(usize, &CustomAgent)> = self
            .agents
            .iter()
            .filter_map(|agent| {
                let name = agent.name.to_lowercase();
                let description = agent.description.to_lowercase();
                let score: usize = words
                    .iter()
                    .map(|word| {
                        3 * usize::from(name.contains(word.as_str()))
                            + usize::from(description.contains(word.as_str()))
                    })
                    .sum();
                (score > 0).then_some((score, agent))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, agent)| agent)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const API_DESIGNER: &str = "---\nname: api-designer\ndescription: \"Use this agent when designing new APIs, creating API specifications.\"\ntools: Read, Write, Edit, Bash, Glob, Grep\nmodel: sonnet\n---\n\nYou are a senior API designer.\n";

    fn agent(content: &str) -> CustomAgent {
        parse_agent(content, Path::new("a.md"), AgentSource::User).expect("agent")
    }

    #[test]
    fn parses_a_claude_code_agent_file() {
        let agent = agent(API_DESIGNER);
        assert_eq!(agent.name, "api-designer");
        assert!(agent.description.starts_with("Use this agent"));
        assert_eq!(agent.prompt, "You are a senior API designer.");
        assert_eq!(
            agent.tools.as_deref().unwrap(),
            ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        );
        assert_eq!(agent.role(), "code");
        assert!(agent.writable());
        assert_eq!(agent.max_tool_rounds(), DEFAULT_MAX_TOOL_ROUNDS);
        let tools = agent.allowed_tools().unwrap();
        for tool in [
            "show_file",
            "create_file",
            "modify_file",
            "run_shell_command",
            "list_directory",
            "search_text",
        ] {
            assert!(tools.contains(tool), "{tool}");
        }
        assert!(tools.contains("delete_file"));
        assert!(agent.unsupported_tools().is_empty());
    }

    #[test]
    fn a_read_only_agent_is_not_writable_and_unknown_tools_are_reported() {
        let agent = agent(
            "---\nname: researcher\ndescription: Researches.\ntools: Read, Grep, Glob, WebFetch, WebSearch, NotebookEdit, mcp__x__y\nmodel: haiku\n---\nResearch.\n",
        );
        assert!(!agent.writable());
        assert_eq!(agent.role(), "explorer");
        assert_eq!(agent.unsupported_tools(), ["NotebookEdit"]);
        let tools = agent.allowed_tools().unwrap();
        assert!(tools.contains("mcp__x__y"));
        assert!(tools.contains("fetch_url"));
        assert!(tools.contains("web_search"));
        assert!(!tools.contains("run_shell_command"));
    }

    #[test]
    fn mcp_names_pass_through_and_name_whole_servers() {
        let agent = agent(
            "---\nname: a\ndescription: A.\ntools: Read, mcp__github, mcp__db__query\ndisallowedTools: mcp__github__delete_repo\n---\nA.\n",
        );
        assert!(agent.may_use("mcp__github__list_issues"));
        assert!(!agent.may_use("mcp__github__delete_repo"));
        assert!(agent.may_use("mcp__db__query"));
        assert!(!agent.may_use("mcp__db__drop"));
        assert!(!agent.may_use("mcp__other__x"));
        let set: HashSet<String> = ["mcp__github".to_string()].into();
        assert!(lists_tool(&set, "mcp__github__x"));
        assert!(!lists_tool(&set, "mcp__githubx__y"));
    }

    #[test]
    fn mcp_servers_are_named_or_defined() {
        let named = agent(
            "---\nname: a\ndescription: A.\ntools: Read\nmcpServers:\n  - github\n  - docs:\n      type: http\n      url: http://localhost:9/mcp\n  - playwright:\n      type: stdio\n      command: npx\n  - events:\n      type: sse\n      url: http://localhost:9/sse\n---\nA.\n",
        );
        assert_eq!(
            named.mcp_servers,
            [
                AgentMcpServer::Reference("github".into()),
                AgentMcpServer::Http {
                    name: "docs".into(),
                    url: "http://localhost:9/mcp".into()
                },
                AgentMcpServer::Unsupported {
                    name: "playwright".into(),
                    reason: "stdio servers are not supported, only HTTP".into()
                },
                AgentMcpServer::Unsupported {
                    name: "events".into(),
                    reason: "sse servers are not supported, only HTTP".into()
                },
            ]
        );
        // The servers it names come with it, whatever `tools` lists.
        assert!(named.may_use("mcp__github__x"));
        assert!(named.may_use("mcp__docs__search"));
        let map = agent(
            "---\nname: a\ndescription: A.\nmcpServers:\n  docs:\n    url: http://h/mcp\n---\nA.\n",
        );
        assert_eq!(map.mcp_servers.len(), 1);
        assert!(matches!(&map.mcp_servers[0], AgentMcpServer::Http { name, .. } if name == "docs"));
    }

    #[test]
    fn skills_are_listed_and_skill_maps_to_use_skill() {
        let named = agent(
            "---\nname: a\ndescription: A.\ntools: Read, Skill\nskills:\n  - release-notes\n  - changelog\n---\nA.\n",
        );
        assert_eq!(named.skills, ["release-notes", "changelog"]);
        assert!(named.may_use("use_skill"));
        assert!(!named.writable());
        let inline = agent("---\nname: a\ndescription: A.\nskills: x, y\n---\nA.\n");
        assert_eq!(inline.skills, ["x", "y"]);
    }

    #[test]
    fn session_agents_come_from_json_and_win_over_files() {
        let session = parse_session_agents(
            r#"{"reviewer": {"description": "Reviews code.", "prompt": "You review.", "tools": ["Read", "Grep"], "maxTurns": 4, "mcpServers": ["github"]}}"#,
        )
        .unwrap();
        assert_eq!(session.len(), 1);
        let reviewer = &session[0];
        assert_eq!(reviewer.name, "reviewer");
        assert_eq!(reviewer.prompt, "You review.");
        assert_eq!(reviewer.source, AgentSource::Session);
        assert_eq!(reviewer.max_tool_rounds(), 4);
        assert!(!reviewer.writable());
        assert_eq!(
            reviewer.mcp_servers,
            [AgentMcpServer::Reference("github".into())]
        );

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("r.md"),
            "---\nname: reviewer\ndescription: From a file.\n---\nR.\n",
        )
        .unwrap();
        let live = LiveAgents::watch_directories(
            vec![(dir.path().to_path_buf(), AgentSource::Project)],
            session,
        );
        assert_eq!(live.current().all()[0].description, "Reviews code.");

        for bad in [
            "[1]",
            r#"{"a": 1}"#,
            r#"{"a": {"description": "No prompt."}}"#,
            r#"{"list": {"description": "D.", "prompt": "P."}}"#,
        ] {
            assert!(parse_session_agents(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn live_agents_pick_up_added_edited_and_removed_files() {
        let dir = tempfile::tempdir().unwrap();
        let live = LiveAgents::watch_directories(
            vec![(dir.path().to_path_buf(), AgentSource::Project)],
            Vec::new(),
        );
        assert!(live.current().is_empty());
        let file = dir.path().join("a.md");
        std::fs::write(&file, "---\nname: a\ndescription: First.\n---\nA.\n").unwrap();
        // Within the check interval nothing is looked at; a refresh looks.
        assert!(live.current().is_empty());
        assert_eq!(live.refresh().all()[0].description, "First.");
        std::fs::write(
            &file,
            "---\nname: a\ndescription: Second, longer.\n---\nA.\n",
        )
        .unwrap();
        assert_eq!(live.refresh().all()[0].description, "Second, longer.");
        std::fs::remove_file(&file).unwrap();
        assert!(live.refresh().is_empty());
        // A fixed set never looks.
        let fixed = LiveAgents::fixed(AgentRegistry::default());
        assert!(fixed.refresh().is_empty());
    }

    #[test]
    fn plugin_agents_are_scoped_by_their_plugin() {
        let home = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let agent = "---\nname: scanner\ndescription: Scans.\nmcpServers:\n  - x\nmemory: user\n---\nScan.\n";
        // Installed by Claude Code, in its versioned cache layout.
        let cached = home
            .path()
            .join(".claude/plugins/cache/market/security/1.0.0");
        std::fs::create_dir_all(cached.join("agents")).unwrap();
        std::fs::write(cached.join("agents/scanner.md"), agent).unwrap();
        // Installed but disabled.
        let disabled = home.path().join(".claude/plugins/cache/market/off/2.0.0");
        std::fs::create_dir_all(disabled.join("agents")).unwrap();
        std::fs::write(disabled.join("agents/scanner.md"), agent).unwrap();
        std::fs::write(
            home.path().join(".claude/plugins/installed_plugins.json"),
            serde_json::json!({"version": 2, "plugins": {
                "security@market": [{"scope": "user", "installPath": cached}],
                "off@market": [{"scope": "user", "installPath": disabled}],
                "elsewhere@market": [{"scope": "project", "projectPath": "/other", "installPath": "/nowhere"}]
            }})
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            home.path().join(".claude/settings.json"),
            r#"{"enabledPlugins": {"off@market": false}}"#,
        )
        .unwrap();
        assert_eq!(
            claude_code_plugins(home.path(), workspace.path()),
            std::slice::from_ref(&cached)
        );

        // orangu's own, named by its manifest.
        let own = workspace.path().join(".orangu/plugins/dir-name");
        std::fs::create_dir_all(own.join("agents")).unwrap();
        std::fs::create_dir_all(own.join(".claude-plugin")).unwrap();
        std::fs::write(
            own.join(".claude-plugin/plugin.json"),
            r#"{"name": "reviews"}"#,
        )
        .unwrap();
        std::fs::write(own.join("agents/scanner.md"), agent).unwrap();

        let registry = AgentRegistry::from_directories(&[
            (cached.join("agents"), AgentSource::Plugin),
            (own.join("agents"), AgentSource::Plugin),
        ]);
        let names: Vec<&str> = registry.all().iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["reviews:scanner", "security:scanner"]);
        let scanner = registry.find("security:scanner").unwrap();
        assert_eq!(scanner.source, AgentSource::Plugin);
        assert!(scanner.mcp_servers.is_empty());
        assert!(
            scanner
                .memory_dir(workspace.path())
                .unwrap()
                .ends_with("security-scanner")
        );
        // A bare name finds a plugin agent only while it is the only one.
        assert!(registry.find("scanner").is_none());
        let one = AgentRegistry::from_directories(&[(cached.join("agents"), AgentSource::Plugin)]);
        assert_eq!(one.find("scanner").unwrap().name, "security:scanner");
    }

    #[test]
    fn agent_in_tools_lets_an_agent_start_agents() {
        let any = agent("---\nname: a\ndescription: A.\ntools: Read, Task\n---\nA.\n");
        assert_eq!(any.starts_agents(), Some(None));
        assert!(any.may_use("run_agent") && any.may_use("write_tests"));
        let named = agent(
            "---\nname: a\ndescription: A.\ntools: Read, Agent(worker, explore_repository)\n---\nA.\n",
        );
        assert_eq!(
            named.tools.as_deref().unwrap(),
            ["Read", "Agent(worker, explore_repository)"]
        );
        assert_eq!(
            named.starts_agents(),
            Some(Some(
                ["worker".to_string(), "explore_repository".to_string()].into()
            ))
        );
        assert!(named.may_use("run_agent") && named.may_use("explore_repository"));
        assert!(!named.may_use("write_tests"));
        assert!(named.unsupported_tools().is_empty());
        // No `tools` field: everything but nesting, which is asked for.
        let plain = agent("---\nname: a\ndescription: A.\n---\nA.\n");
        assert_eq!(plain.starts_agents(), None);
        let denied =
            agent("---\nname: a\ndescription: A.\ntools: Agent\ndisallowedTools: Agent\n---\nA.\n");
        assert_eq!(denied.starts_agents(), None);
    }

    #[test]
    fn the_remaining_claude_code_fields_are_read() {
        let read = agent(
            "---\nname: a\ndescription: A.\npermissionMode: plan\nbackground: true\neffort: high\ncolor: blue\ninitialPrompt: Start with the README.\nisolation: worktree\nhooks:\n  PreToolUse:\n    - matcher: Bash\n      hooks:\n        - command: ./check.sh\n---\nA.\n",
        );
        assert!(read.plans());
        assert!(!read.writable(), "plan mode changes nothing");
        assert!(read.may_use("show_file") && !read.may_use("modify_file"));
        assert!(read.background);
        assert_eq!(read.effort.as_deref(), Some("high"));
        assert_eq!(read.color.as_deref(), Some("blue"));
        assert_eq!(
            read.initial_prompt.as_deref(),
            Some("Start with the README.")
        );
        assert!(!read.hooks.is_empty());
        let lenient =
            agent("---\nname: a\ndescription: Bad: yaml\nbackground: true\ncolor: red\n---\nA.\n");
        assert!(lenient.background);
        assert_eq!(lenient.color.as_deref(), Some("red"));

        // A plugin's agent gets neither hooks nor a permission mode.
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path().join("p/agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(
            agents.join("a.md"),
            "---\nname: a\ndescription: A.\npermissionMode: plan\nhooks:\n  PreToolUse:\n    - hooks:\n        - command: x\n---\nA.\n",
        )
        .unwrap();
        let registry = AgentRegistry::from_directories(&[(agents, AgentSource::Plugin)]);
        let plugin = registry.find("p:a").unwrap();
        assert!(plugin.hooks.is_empty() && !plugin.plans());
    }

    #[test]
    fn no_tools_field_means_every_tool() {
        let agent = agent("---\nname: all\ndescription: Everything.\n---\nDo it.\n");
        assert!(agent.allowed_tools().is_none());
        assert!(agent.writable());
    }

    #[test]
    fn tools_may_be_a_yaml_list_and_orangu_names() {
        let agent = agent(
            "---\nname: lister\ndescription: Lists.\ntools:\n  - Glob\n  - convert_document\n---\nList.\n",
        );
        let tools = agent.allowed_tools().unwrap();
        assert!(tools.contains("list_directory"));
        assert!(tools.contains("convert_document"));
        assert!(agent.writable());
    }

    #[test]
    fn max_turns_sets_the_budget_unless_max_tool_rounds_does() {
        let turns = agent("---\nname: a\ndescription: A.\nmaxTurns: 7\n---\nA.\n");
        assert_eq!(turns.max_tool_rounds(), 7);
        let both =
            agent("---\nname: a\ndescription: A.\nmaxTurns: 7\nmax_tool_rounds: 9\n---\nA.\n");
        assert_eq!(both.max_tool_rounds(), 9);
        let lenient = agent("---\nname: a\ndescription: Bad: yaml\nmaxTurns: 3\n---\nA.\n");
        assert_eq!(lenient.max_tool_rounds(), 3);
    }

    #[test]
    fn disallowed_tools_are_taken_away_first() {
        // With `tools`: denied ones leave the allowlist.
        let agent_ = agent(
            "---\nname: a\ndescription: A.\ntools: Read, Write, Bash\ndisallowedTools: Bash\n---\nA.\n",
        );
        let allowed = agent_.allowed_tools().unwrap();
        assert!(allowed.contains("create_file"));
        assert!(!allowed.contains("run_shell_command"));
        assert!(agent_.writable());

        // Without `tools`: everything but the denied ones.
        let reviewer = agent(
            "---\nname: r\ndescription: R.\ndisallowedTools:\n  - Write\n  - Edit\n  - Bash\n---\nR.\n",
        );
        assert!(reviewer.allowed_tools().is_none());
        assert!(reviewer.denied_tools().contains("modify_file"));
        assert!(reviewer.may_use("show_file"));
        assert!(!reviewer.may_use("create_file"));
        assert!(!reviewer.may_use("run_shell_command"));
        // Moving and deleting go with Edit and Bash, so nothing that writes
        // is left.
        assert!(!reviewer.may_use("delete_file"));
        assert!(!reviewer.writable());

        let lenient = agent(
            "---\nname: a\ndescription: Bad: yaml\ntools: Read, Bash\ndisallowedTools: Bash\n---\nA.\n",
        );
        assert!(!lenient.writable());
    }

    #[test]
    fn memory_scopes_name_their_directories() {
        let workspace = Path::new("/work/project");
        let scoped = |scope: &str| {
            agent(&format!(
                "---\nname: notes-keeper\ndescription: N.\nmemory: {scope}\n---\nN.\n"
            ))
        };
        assert_eq!(scoped("project").memory, Some(MemoryScope::Project));
        assert_eq!(
            scoped("project").memory_dir(workspace).unwrap(),
            workspace.join(".orangu/agent-memory/notes-keeper")
        );
        assert_eq!(
            scoped("Local").memory_dir(workspace).unwrap(),
            workspace.join(".orangu/agent-memory-local/notes-keeper")
        );
        assert!(
            scoped("user")
                .memory_dir(workspace)
                .unwrap()
                .ends_with(".orangu/agent-memory/notes-keeper")
        );
        assert_eq!(scoped("everywhere").memory, None);
        assert!(
            agent("---\nname: n\ndescription: N.\n---\nN.\n")
                .memory_dir(workspace)
                .is_none()
        );
    }

    /// A name becomes a directory, so it must not climb out of one.
    #[test]
    fn names_starting_with_a_dot_are_rejected() {
        for name in ["..", ".", ".hidden"] {
            let content = format!("---\nname: {name}\ndescription: d\n---\nBody\n");
            assert!(
                parse_agent(&content, Path::new("x.md"), AgentSource::User).is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn the_prompt_carries_the_start_of_the_memory() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(memory_for_prompt(dir.path()), "");
        let long: String = (0..300).map(|i| format!("line {i}\n")).collect();
        std::fs::write(dir.path().join("MEMORY.md"), &long).unwrap();
        let kept = memory_for_prompt(dir.path());
        assert_eq!(kept.lines().count(), MEMORY_PROMPT_LINES);
        assert!(kept.starts_with("line 0\n"));

        let wide = "x".repeat(1000) + "\n";
        std::fs::write(dir.path().join("MEMORY.md"), wide.repeat(100)).unwrap();
        assert!(memory_for_prompt(dir.path()).len() <= MEMORY_PROMPT_BYTES);

        std::fs::write(dir.path().join("MEMORY.md"), "Tests live in tests/.\n").unwrap();
        let keeper = agent("---\nname: k\ndescription: K.\nmemory: project\n---\nK.\n");
        let text = memory_instructions(&keeper, dir.path());
        assert!(text.contains("## Memory"), "{text}");
        assert!(text.contains("project scope"), "{text}");
        assert!(text.contains("save_memory"), "{text}");
        assert!(
            text.contains("<memory>\nTests live in tests/.\n</memory>"),
            "{text}"
        );
    }

    #[test]
    fn omit_claude_md_leaves_out_the_project_rules() {
        assert!(!agent("---\nname: a\ndescription: A.\n---\nA.\n").omit_project_rules);
        assert!(
            agent("---\nname: a\ndescription: A.\nomitClaudeMd: true\n---\nA.\n")
                .omit_project_rules
        );
        // Through the line-by-line fallback too.
        assert!(
            agent("---\nname: a\ndescription: Bad: yaml\nomitClaudeMd: true\n---\nA.\n")
                .omit_project_rules
        );
    }

    #[test]
    fn orangu_fields_override_the_model_and_bound_the_rounds() {
        let agent = agent(
            "---\nname: r\ndescription: R.\nmodel: haiku\nrole: Review\nmax_tool_rounds: 500\n---\nR.\n",
        );
        assert_eq!(agent.role(), "review");
        assert_eq!(agent.max_tool_rounds(), MAX_TOOL_ROUNDS);
        let unknown = parse_agent(
            "---\nname: r\ndescription: R.\nrole: wizard\n---\nR.\n",
            Path::new("r.md"),
            AgentSource::User,
        )
        .unwrap();
        assert_eq!(unknown.role(), "code");
    }

    #[test]
    fn invalid_yaml_falls_back_to_plain_lines() {
        let agent = agent(
            "---\nname: terragrunt-expert\ndescription: Expert in Terragrunt: DRY configs, stacks\ntools: Read, Bash\n---\nBody.\n",
        );
        assert_eq!(
            agent.description,
            "Expert in Terragrunt: DRY configs, stacks"
        );
        assert_eq!(agent.tools.as_deref().unwrap(), ["Read", "Bash"]);
    }

    #[test]
    fn non_agent_markdown_is_skipped() {
        for content in [
            "# README\n\nNo frontmatter.\n",
            "---\ndescription: A command, no name.\n---\nBody\n",
            "---\nname: x\n---\nNo description.\n",
            "---\nname: bad name!\ndescription: d\n---\nBody\n",
            "---\nname: empty\ndescription: d\n---\n\n",
            "---\nname: open\ndescription: never closed\n",
            "---\nname: list\ndescription: reserved\n---\nBody\n",
        ] {
            assert!(
                parse_agent(content, Path::new("x.md"), AgentSource::User).is_none(),
                "{content}"
            );
        }
    }

    #[test]
    fn discovers_a_cloned_collection_and_projects_override_users() {
        let user = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let category = user.path().join("collection/categories/01-core");
        std::fs::create_dir_all(&category).unwrap();
        std::fs::write(category.join("api-designer.md"), API_DESIGNER).unwrap();
        std::fs::write(category.join("README.md"), "# Core\n").unwrap();
        std::fs::write(
            user.path().join("collection/CLAUDE.md"),
            "# CLAUDE.md\nguidance\n",
        )
        .unwrap();
        // A collection's slash commands look like agents but are not.
        let commands = user.path().join("collection/tools/catalog");
        std::fs::create_dir_all(&commands).unwrap();
        std::fs::write(
            commands.join("search.md"),
            "---\nname: search\ndescription: Search the catalog.\n---\nSearch.\n",
        )
        .unwrap();
        // Hidden directories (a clone's `.git`) are not searched.
        std::fs::create_dir_all(user.path().join(".git")).unwrap();
        std::fs::write(
            user.path().join(".git/hidden.md"),
            "---\nname: hidden\ndescription: d\n---\nb\n",
        )
        .unwrap();
        std::fs::write(
            user.path().join("reviewer.md"),
            "---\nname: reviewer\ndescription: Reviews code.\ntools: Read\n---\nUser reviewer.\n",
        )
        .unwrap();
        std::fs::write(
            project.path().join("reviewer.md"),
            "---\nname: reviewer\ndescription: Reviews code.\ntools: Read\n---\nProject reviewer.\n",
        )
        .unwrap();

        let registry = AgentRegistry::from_directories(&[
            (user.path().to_path_buf(), AgentSource::User),
            (project.path().to_path_buf(), AgentSource::Project),
            (PathBuf::from("/nonexistent/agents"), AgentSource::Project),
        ]);
        let names: Vec<&str> = registry.all().iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["api-designer", "reviewer"]);
        let reviewer = registry.find("Reviewer").unwrap();
        assert_eq!(reviewer.prompt, "Project reviewer.");
        assert_eq!(reviewer.source, AgentSource::Project);
    }

    #[test]
    fn search_ranks_name_matches_first() {
        let registry = AgentRegistry {
            agents: vec![
                agent("---\nname: api-designer\ndescription: Designs APIs.\n---\nb\n"),
                agent(
                    "---\nname: backend-developer\ndescription: Builds APIs and services.\n---\nb\n",
                ),
                agent("---\nname: ui-designer\ndescription: Designs interfaces.\n---\nb\n"),
            ],
        };
        let names = |query: &str| -> Vec<String> {
            registry
                .search(query, 10)
                .into_iter()
                .map(|a| a.name.clone())
                .collect()
        };
        assert_eq!(names("api"), ["api-designer", "backend-developer"]);
        assert_eq!(names("designer"), ["api-designer", "ui-designer"]);
        assert!(names("kubernetes").is_empty());
        assert_eq!(registry.search("", 2).len(), 2);
    }
}

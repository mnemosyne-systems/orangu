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

//! The tools `/v1/messages` runs itself rather than handing back to the
//! client: `web_search_*`, `web_fetch_*`, `code_execution_*`, `advisor_*`,
//! and the tools of the remote MCP servers a request names in
//! `mcp_servers`.
//!
//! [`ServerTools`] is built per request from its `tools` and `mcp_servers`.
//! It declares the tools to the model, recognises a call to one, runs it,
//! and answers with the Anthropic result block for the client and the text
//! the model reads on its next turn.

use base64::Engine as _;
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, JsonObject},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use super::containers::Container;
use crate::engine::tool_calls::ParsedToolCall;

/// How long one search, fetch or MCP call may take.
const CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// How many results a search returns.
const SEARCH_RESULTS: usize = 8;

/// The longest URL `web_fetch` takes, as Anthropic's own limit.
const MAX_URL: usize = 250;

/// What a fetched page is cut to when the request sets no
/// `max_content_tokens` — about 25,000 tokens of text.
const DEFAULT_FETCH_CHARS: usize = 100_000;

/// A server tool's per-request limits, from its declaration.
#[derive(Default)]
struct Limits {
    name: String,
    max_uses: Option<usize>,
    uses: usize,
    allowed_domains: Vec<String>,
    blocked_domains: Vec<String>,
}

impl Limits {
    fn from_declaration(tool: &Value, default_name: &str) -> Self {
        let domains = |key: &str| -> Vec<String> {
            tool.get(key)
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(Value::as_str)
                        .map(|d| d.trim_start_matches("*.").to_ascii_lowercase())
                        .collect()
                })
                .unwrap_or_default()
        };
        Self {
            name: tool
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(default_name)
                .to_string(),
            max_uses: tool
                .get("max_uses")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            uses: 0,
            allowed_domains: domains("allowed_domains"),
            blocked_domains: domains("blocked_domains"),
        }
    }

    /// Counts one use, or says the declaration's `max_uses` is spent.
    fn take_use(&mut self) -> bool {
        if self.max_uses.is_some_and(|max| self.uses >= max) {
            return false;
        }
        self.uses += 1;
        true
    }

    /// Whether `url`'s host passes `allowed_domains` and `blocked_domains`:
    /// a domain names itself and every subdomain of it.
    fn permits(&self, url: &reqwest::Url) -> bool {
        let Some(host) = url.host_str().map(str::to_ascii_lowercase) else {
            return false;
        };
        let under = |domain: &String| host == *domain || host.ends_with(&format!(".{domain}"));
        (self.allowed_domains.is_empty() || self.allowed_domains.iter().any(under))
            && !self.blocked_domains.iter().any(under)
    }
}

struct McpServer {
    name: String,
    service: RunningService<RoleClient, ()>,
}

/// Which server tool a name is, for the call the model made.
#[derive(Clone, PartialEq, Debug)]
pub(super) enum Kind {
    Search,
    Fetch,
    /// `bash_code_execution`: a shell command in the container.
    Bash,
    /// `text_editor_code_execution`: view, create or edit a file in it.
    Editor,
    /// `code_execution` as `code_execution_20250522` has it: Python code.
    Python,
    /// `advisor`: a consultation of the advisor model, which the reply's
    /// driver runs — it is a model request, not a call this runs.
    Advisor,
    Mcp {
        server: usize,
        tool: String,
    },
}

/// What the code execution tool is declared as.
#[derive(Clone, Copy, PartialEq)]
enum Code {
    /// `code_execution_20250522`: one `code_execution` tool running Python.
    Python,
    /// Every later version: `bash_code_execution` and
    /// `text_editor_code_execution`.
    BashAndEditor,
}

/// What running a server tool produced.
pub(super) struct Ran {
    /// The result block, given the id of the `server_tool_use` (or
    /// `mcp_tool_use`) block it answers.
    pub block: Value,
    /// What the model is told the tool returned.
    pub text: String,
}

/// The advisor a request declares.
pub(super) struct Advisor {
    limits: Limits,
    pub model: String,
    pub max_tokens: Option<u64>,
}

pub(super) struct ServerTools {
    search: Option<Limits>,
    advisor: Option<Advisor>,
    fetch: Option<(Limits, usize)>,
    code: Option<Code>,
    /// The container code runs in, once the request has one.
    container: Option<Container>,
    backend: String,
    mcp: Vec<McpServer>,
    /// MCP tools by the name the model calls them, with their declarations.
    mcp_tools: Vec<(String, Kind, Value)>,
    /// URLs `web_fetch` may fetch: every one that has appeared in the
    /// conversation, its search results and fetched pages included. A
    /// model can fetch what it has been shown, not an address it composed —
    /// and [`fetch_http`](Self::fetch_http) reaches public addresses only.
    seen_urls: HashSet<String>,
    pub(super) search_requests: usize,
    pub(super) fetch_requests: usize,
    /// The client tools the model's Python code may call — see
    /// [`super::programmatic`]. Not empty, the model is given a Python
    /// `code_execution` tool whose code calls them.
    programmatic: Vec<Value>,
    http: reqwest::Client,
    /// The client `web_fetch` fetches with, behind
    /// [`crate::http::public_net`]'s guard.
    fetch_http: reqwest::Client,
}

/// The `http(s)://…` URLs in `text`, trailing punctuation trimmed.
fn urls_in(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| {
        c.is_whitespace() || matches!(c, '<' | '>' | '"' | '\'' | '(' | ')' | '[' | ']')
    })
    .filter(|word| word.starts_with("http://") || word.starts_with("https://"))
    .map(|word| {
        word.trim_end_matches(['.', ',', ';', ':', '!', '?'])
            .to_string()
    })
}

impl ServerTools {
    /// The server tools a request declares, its MCP servers connected and
    /// their tools listed. `conversation` is the request's text, for the
    /// URLs `web_fetch` may fetch.
    pub(super) async fn from_request(
        tools: &[Value],
        mcp_servers: &[Value],
        conversation: &str,
        backend: &str,
        sandbox: &str,
    ) -> Result<Self, String> {
        let mut search = None;
        let mut fetch = None;
        let mut code = None;
        let mut advisor = None;
        let mut toolsets = Vec::new();
        for tool in tools {
            let kind = tool.get("type").and_then(Value::as_str).unwrap_or_default();
            if kind.starts_with("web_search_") {
                if backend == "off" {
                    return Err("web_search needs a search backend: set \
                                [orangu-server].web_search to a SearXNG instance's URL"
                        .to_string());
                }
                search = Some(Limits::from_declaration(tool, "web_search"));
            } else if kind.starts_with("web_fetch_") {
                let chars = tool
                    .get("max_content_tokens")
                    .and_then(Value::as_u64)
                    .map_or(DEFAULT_FETCH_CHARS, |tokens| tokens as usize * 4);
                fetch = Some((Limits::from_declaration(tool, "web_fetch"), chars));
            } else if kind.starts_with("code_execution_") {
                if sandbox == "off" {
                    return Err("code_execution needs a sandbox: set \
                                [orangu-server].code_execution = bwrap"
                        .to_string());
                }
                code = Some(if kind == "code_execution_20250522" {
                    Code::Python
                } else {
                    Code::BashAndEditor
                });
            } else if kind.starts_with("advisor_") {
                let model = tool
                    .get("model")
                    .and_then(Value::as_str)
                    .ok_or("the advisor tool names its `model`")?;
                advisor = Some(Advisor {
                    limits: Limits::from_declaration(tool, "advisor"),
                    model: model.to_string(),
                    max_tokens: tool.get("max_tokens").and_then(Value::as_u64),
                });
            } else if kind == "mcp_toolset" {
                toolsets.push(tool);
            }
        }
        let http = reqwest::Client::builder()
            .timeout(CALL_TIMEOUT)
            .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|err| err.to_string())?;
        let fetch_http = crate::http::public_net::guarded(reqwest::Client::builder())
            .timeout(CALL_TIMEOUT)
            .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|err| err.to_string())?;
        let mut this = Self {
            search,
            advisor,
            fetch,
            code,
            container: None,
            backend: backend.to_string(),
            mcp: Vec::new(),
            mcp_tools: Vec::new(),
            seen_urls: urls_in(conversation).collect(),
            search_requests: 0,
            fetch_requests: 0,
            http,
            fetch_http,
            programmatic: Vec::new(),
        };
        this.connect_mcp(mcp_servers, &toolsets).await?;
        Ok(this)
    }

    /// Connects every server in `mcp_servers` and keeps the tools it may
    /// use: with `mcp_toolset` entries, those a toolset names enable; without
    /// any, each server's own `tool_configuration` decides.
    async fn connect_mcp(&mut self, servers: &[Value], toolsets: &[&Value]) -> Result<(), String> {
        let mut used = HashSet::new();
        for server in servers {
            let name = server
                .get("name")
                .and_then(Value::as_str)
                .ok_or("an mcp_servers entry needs a `name`")?
                .to_string();
            let url = server
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("MCP server '{name}' needs a `url`"))?;
            let toolset = toolsets
                .iter()
                .find(|t| t.get("mcp_server_name").and_then(Value::as_str) == Some(name.as_str()));
            if !toolsets.is_empty() && toolset.is_none() {
                continue;
            }
            let configuration = server.get("tool_configuration");
            if configuration
                .and_then(|c| c.get("enabled"))
                .and_then(Value::as_bool)
                == Some(false)
            {
                continue;
            }
            let allowed: Option<Vec<String>> = configuration
                .and_then(|c| c.get("allowed_tools"))
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                });
            let mut config = StreamableHttpClientTransportConfig::with_uri(url.to_string());
            if let Some(token) = server.get("authorization_token").and_then(Value::as_str) {
                config = config.auth_header(token.to_string());
            }
            let transport = StreamableHttpClientTransport::from_config(config);
            let service = tokio::time::timeout(CALL_TIMEOUT, ().serve(transport))
                .await
                .map_err(|_| format!("MCP server '{name}' did not answer in time"))?
                .map_err(|err| format!("MCP server '{name}': {err}"))?;
            let listed = tokio::time::timeout(CALL_TIMEOUT, service.list_all_tools())
                .await
                .map_err(|_| format!("MCP server '{name}' did not list its tools in time"))?
                .map_err(|err| format!("MCP server '{name}': listing tools: {err}"))?;
            let index = self.mcp.len();
            for tool in listed {
                let tool_name = tool.name.to_string();
                if allowed.as_ref().is_some_and(|a| !a.contains(&tool_name)) {
                    continue;
                }
                if let Some(toolset) = toolset {
                    let enabled_by_default = toolset
                        .pointer("/default_config/enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let enabled = toolset
                        .get("configs")
                        .and_then(|c| c.get(&tool_name))
                        .and_then(|c| c.get("enabled"))
                        .and_then(Value::as_bool)
                        .unwrap_or(enabled_by_default);
                    if !enabled {
                        continue;
                    }
                }
                // The first server to offer a name keeps it.
                if !used.insert(tool_name.clone()) {
                    continue;
                }
                let mut function = json!({
                    "name": tool_name,
                    "parameters": Value::Object((*tool.input_schema).clone()),
                });
                if let Some(description) = &tool.description {
                    function["description"] = json!(description);
                }
                self.mcp_tools.push((
                    tool_name.clone(),
                    Kind::Mcp {
                        server: index,
                        tool: tool_name,
                    },
                    json!({"type": "function", "function": function}),
                ));
            }
            self.mcp.push(McpServer { name, service });
        }
        Ok(())
    }

    /// Whether the request declares the code execution tool.
    pub(super) fn runs_code(&self) -> bool {
        self.code.is_some()
    }

    /// Gives the code execution tool the container it runs in.
    pub(super) fn attach(&mut self, container: Container) {
        self.container = Some(container);
    }

    /// The server tools' declarations, in the OpenAI shape the chat
    /// template is handed.
    pub(super) fn definitions(&self) -> Vec<Value> {
        let mut out = Vec::new();
        if let Some(search) = &self.search {
            out.push(json!({"type": "function", "function": {
                "name": search.name,
                "description": "Search the web. Returns the title, URL and an excerpt of each result.",
                "parameters": {"type": "object", "required": ["query"], "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                }},
            }}));
        }
        if let Some((fetch, _)) = &self.fetch {
            out.push(json!({"type": "function", "function": {
                "name": fetch.name,
                "description": "Fetch a web page or PDF and return its text. Only URLs that have \
                                appeared in the conversation can be fetched.",
                "parameters": {"type": "object", "required": ["url"], "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                }},
            }}));
        }
        let container = "a sandboxed Linux container with Python 3 and no network access. \
                         Its working directory is /workspace, whose files persist from one call \
                         to the next; files written there are handed back to the user.";
        match self.code {
            Some(Code::Python) => out.push(json!({"type": "function", "function": {
                "name": "code_execution",
                "description": format!("Run Python code in {container} Returns its output."),
                "parameters": {"type": "object", "required": ["code"], "properties": {
                    "code": {"type": "string", "description": "The Python code to run."},
                }},
            }})),
            Some(Code::BashAndEditor) => {
                out.push(json!({"type": "function", "function": {
                    "name": "bash_code_execution",
                    "description": format!("Run a bash command in {container} \
                                            Returns its stdout, stderr and exit code."),
                    "parameters": {"type": "object", "required": ["command"], "properties": {
                        "command": {"type": "string", "description": "The command to run."},
                    }},
                }}));
                out.push(json!({"type": "function", "function": {
                    "name": "text_editor_code_execution",
                    "description": "View, create or edit a file in the code execution \
                                    container. `view` shows a file (or lists a directory), \
                                    `create` writes `file_text` to a file, `str_replace` \
                                    replaces `old_str`, which must occur exactly once, with \
                                    `new_str`.",
                    "parameters": {"type": "object", "required": ["command", "path"], "properties": {
                        "command": {"type": "string", "enum": ["view", "create", "str_replace"]},
                        "path": {"type": "string", "description": "A path under /workspace."},
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "view_range": {"type": "array", "items": {"type": "integer"},
                                       "description": "First and last line to show; -1 is the end."},
                    }},
                }}));
            }
            None => {}
        }
        if self.code == Some(Code::BashAndEditor) && !self.programmatic.is_empty() {
            out.push(json!({"type": "function", "function": {
                "name": "code_execution",
                "description": super::programmatic::description(&self.programmatic),
                "parameters": {"type": "object", "required": ["code"], "properties": {
                    "code": {"type": "string", "description": "The Python code to run."},
                }},
            }}));
        }
        if let Some(advisor) = &self.advisor {
            out.push(super::advisor::definition(&advisor.limits.name));
        }
        out.extend(
            self.mcp_tools
                .iter()
                .map(|(_, _, definition)| definition.clone()),
        );
        out
    }

    /// The advisor, for one more consultation: `None` once its `max_uses`
    /// is spent.
    pub(super) fn take_advisor(&mut self) -> Option<&Advisor> {
        let advisor = self.advisor.as_mut()?;
        advisor.limits.take_use().then_some(&*advisor)
    }

    /// Makes `tools` callable from the model's Python code.
    pub(super) fn call_from_code(&mut self, tools: Vec<Value>) {
        self.programmatic = tools;
    }

    /// The names of the tools the model's code may call.
    pub(super) fn code_callable(&self) -> Vec<String> {
        self.programmatic
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_string))
            .collect()
    }

    /// The container code runs in, once one is open.
    pub(super) fn container(&self) -> Option<&Container> {
        self.container.as_ref()
    }

    pub(super) fn kind(&self, name: &str) -> Option<Kind> {
        if self.search.as_ref().is_some_and(|s| s.name == name) {
            return Some(Kind::Search);
        }
        if self.fetch.as_ref().is_some_and(|(f, _)| f.name == name) {
            return Some(Kind::Fetch);
        }
        if self.advisor.as_ref().is_some_and(|a| a.limits.name == name) {
            return Some(Kind::Advisor);
        }
        match (self.code, name) {
            (Some(Code::Python), "code_execution") => return Some(Kind::Python),
            (Some(_), "code_execution") if !self.programmatic.is_empty() => {
                return Some(Kind::Python);
            }
            (Some(Code::BashAndEditor), "bash_code_execution") => return Some(Kind::Bash),
            (Some(Code::BashAndEditor), "text_editor_code_execution") => {
                return Some(Kind::Editor);
            }
            _ => {}
        }
        self.mcp_tools
            .iter()
            .find(|(exposed, _, _)| exposed == name)
            .map(|(_, kind, _)| kind.clone())
    }

    /// The block announcing a server tool call: `server_tool_use` for the
    /// web tools, `mcp_tool_use` (naming its server) for an MCP tool.
    pub(super) fn use_block(&self, kind: &Kind, id: &str, call: &ParsedToolCall) -> Value {
        let input: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
        match kind {
            Kind::Mcp { server, tool } => json!({
                "type": "mcp_tool_use", "id": id, "name": tool,
                "server_name": self.mcp[*server].name, "input": input,
            }),
            _ => json!({"type": "server_tool_use", "id": id, "name": call.name, "input": input}),
        }
    }

    pub(super) async fn run(&mut self, kind: &Kind, id: &str, call: &ParsedToolCall) -> Ran {
        let input: Map<String, Value> = serde_json::from_str(&call.arguments).unwrap_or_default();
        match kind {
            Kind::Search => self.search(id, &input).await,
            Kind::Fetch => self.fetch(id, &input).await,
            Kind::Bash | Kind::Python => self.execute(kind, id, &input).await,
            Kind::Editor => self.edit(id, &input).await,
            Kind::Advisor => {
                let (block, text) = super::advisor::result_block(id, &Err("unavailable"));
                Ran { block, text }
            }
            Kind::Mcp { server, tool } => self.call_mcp(*server, tool, id, input).await,
        }
    }

    /// Runs a bash command, or Python code, in the container.
    async fn execute(&mut self, kind: &Kind, id: &str, input: &Map<String, Value>) -> Ran {
        let (result, error, output, field) = match kind {
            Kind::Python => (
                "code_execution_tool_result",
                "code_execution_tool_result_error",
                "code_execution_output",
                "code",
            ),
            _ => (
                "bash_code_execution_tool_result",
                "bash_code_execution_tool_result_error",
                "bash_code_execution_output",
                "command",
            ),
        };
        let failed = |code: &str, why: &str| Ran {
            block: json!({"type": result, "tool_use_id": id,
                          "content": {"type": error, "error_code": code}}),
            text: format!("Error: the code did not run ({code}){why}"),
        };
        let Some(container) = self.container.as_ref() else {
            return failed("unavailable", ".");
        };
        let Some(source) = input.get(field).and_then(Value::as_str) else {
            return failed("invalid_tool_input", &format!(": `{field}` is required."));
        };
        let before = container.snapshot().await;
        let ran = match kind {
            Kind::Python => container.run(&["python3", "-"], Some(source)).await,
            _ => container.run(&["/bin/bash", "-c", source], None).await,
        };
        let ran = match ran {
            Ok(ran) => ran,
            Err(err) => return failed("unavailable", &format!(": {err}")),
        };
        if ran.timed_out {
            return failed(
                "execution_time_exceeded",
                ": it ran past its time limit and was stopped.",
            );
        }
        let written = container.outputs(&before).await;
        let mut text = run_text(&ran.stdout, &ran.stderr, ran.code);
        if !written.is_empty() {
            let paths: Vec<&str> = written.iter().map(|(_, path)| path.as_str()).collect();
            text.push_str(&format!("\nfiles written: {}", paths.join(", ")));
        }
        let kind = if *kind == Kind::Python {
            "code_execution_result"
        } else {
            "bash_code_execution_result"
        };
        Ran {
            block: json!({"type": result, "tool_use_id": id, "content": {
                "type": kind,
                "stdout": ran.stdout,
                "stderr": ran.stderr,
                "return_code": ran.code,
                "content": written
                    .iter()
                    .map(|(file_id, _)| json!({"type": output, "file_id": file_id}))
                    .collect::<Vec<_>>(),
            }}),
            text,
        }
    }

    async fn edit(&mut self, id: &str, input: &Map<String, Value>) -> Ran {
        let failed = |code: &str, message: &str| Ran {
            block: json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": id,
                          "content": {"type": "text_editor_code_execution_tool_result_error",
                                      "error_code": code, "error_message": message}}),
            text: format!("Error: {message}"),
        };
        let Some(container) = self.container.as_ref() else {
            return failed("unavailable", "the container is not available");
        };
        match container.edit(input).await {
            Ok((content, text)) => Ran {
                block: json!({"type": "text_editor_code_execution_tool_result",
                              "tool_use_id": id, "content": content}),
                text,
            },
            Err(err) => failed(err.code, &err.message),
        }
    }

    async fn search(&mut self, id: &str, input: &Map<String, Value>) -> Ran {
        let failed = |code: &str| Ran {
            block: json!({"type": "web_search_tool_result", "tool_use_id": id,
                          "content": {"type": "web_search_tool_result_error", "error_code": code}}),
            text: format!("Error: the search failed ({code})."),
        };
        let Some(limits) = self.search.as_mut() else {
            return failed("unavailable");
        };
        if !limits.take_use() {
            return failed("max_uses_exceeded");
        }
        let Some(query) = input
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.trim().is_empty())
        else {
            return failed("invalid_input");
        };
        if query.len() > 500 {
            return failed("query_too_long");
        }
        self.search_requests += 1;
        let found = self.search_searxng(query).await;
        let Ok(found) = found else {
            return failed("unavailable");
        };
        let limits = self.search.as_ref().expect("checked above");
        let results: Vec<(String, String, String)> = found
            .into_iter()
            .filter(|(url, _, _)| reqwest::Url::parse(url).is_ok_and(|u| limits.permits(&u)))
            .take(SEARCH_RESULTS)
            .collect();
        let mut text = format!("Search results for \"{query}\":\n");
        let mut blocks = Vec::new();
        for (n, (url, title, snippet)) in results.iter().enumerate() {
            text.push_str(&format!("\n[{}] {title}\n{url}\n{snippet}\n", n + 1));
            self.seen_urls.insert(url.clone());
            blocks.push(json!({
                "type": "web_search_result", "url": url, "title": title,
                // The excerpt, so a client sending this block back gives
                // the model the same text it read.
                "encrypted_content": base64::engine::general_purpose::STANDARD.encode(snippet),
                "page_age": null,
            }));
        }
        if results.is_empty() {
            text.push_str("\nNo results.\n");
        }
        Ran {
            block: json!({"type": "web_search_tool_result", "tool_use_id": id, "content": blocks}),
            text,
        }
    }

    /// A SearXNG instance's JSON results, as (URL, title, excerpt): the
    /// same search the terminal's `web_search` tool runs.
    async fn search_searxng(&self, query: &str) -> Result<Vec<(String, String, String)>, String> {
        orangu::web_search::searxng(&self.http, &self.backend, query)
            .await
            .map(|results| {
                results
                    .into_iter()
                    .map(|result| (result.url, result.title, result.snippet))
                    .collect()
            })
            .map_err(|err| err.to_string())
    }

    async fn fetch(&mut self, id: &str, input: &Map<String, Value>) -> Ran {
        let failed = |code: &str| Ran {
            block: json!({"type": "web_fetch_tool_result", "tool_use_id": id,
                          "content": {"type": "web_fetch_tool_result_error", "error_code": code}}),
            text: format!("Error: the fetch failed ({code})."),
        };
        let Some((limits, max_chars)) = self.fetch.as_mut() else {
            return failed("unavailable");
        };
        let max_chars = *max_chars;
        if !limits.take_use() {
            return failed("max_uses_exceeded");
        }
        let Some(raw) = input.get("url").and_then(Value::as_str) else {
            return failed("invalid_input");
        };
        if raw.len() > MAX_URL {
            return failed("url_too_long");
        }
        let Ok(url) = reqwest::Url::parse(raw) else {
            return failed("invalid_input");
        };
        if crate::http::public_net::check(&url).is_err()
            || !limits.permits(&url)
            || !(self.seen_urls.contains(raw) || self.seen_urls.contains(url.as_str()))
        {
            return failed("url_not_allowed");
        }
        self.fetch_requests += 1;
        let Ok(response) = self.fetch_http.get(url.clone()).send().await else {
            return failed("url_not_accessible");
        };
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return failed("too_many_requests");
        }
        if !response.status().is_success() {
            return failed("url_not_accessible");
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let Ok(bytes) = response.bytes().await else {
            return failed("url_not_accessible");
        };
        let is_pdf = content_type.starts_with("application/pdf") || bytes.starts_with(b"%PDF-");
        let (text, title) = if is_pdf {
            match tokio::task::spawn_blocking(move || orangu::pdf::extract_text(&bytes)).await {
                Ok(Ok(text)) => (text, None),
                _ => return failed("unsupported_content_type"),
            }
        } else if content_type.contains("html") || content_type.is_empty() {
            let title = page_title(&String::from_utf8_lossy(&bytes));
            match html2text::from_read(&bytes[..], 120) {
                Ok(text) => (text, title),
                Err(_) => return failed("unsupported_content_type"),
            }
        } else if content_type.starts_with("text/")
            || content_type.contains("json")
            || content_type.contains("xml")
        {
            (String::from_utf8_lossy(&bytes).into_owned(), None)
        } else {
            return failed("unsupported_content_type");
        };
        let text: String = text.chars().take(max_chars).collect();
        self.seen_urls.extend(urls_in(&text));
        Ran {
            block: json!({"type": "web_fetch_tool_result", "tool_use_id": id, "content": {
                "type": "web_fetch_result", "url": url.as_str(),
                "content": {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": text},
                    "title": title,
                },
                "retrieved_at": super::rfc3339(super::unix_now()),
            }}),
            text: format!("Content of {url}:\n\n{text}"),
        }
    }

    async fn call_mcp(
        &mut self,
        server: usize,
        tool: &str,
        id: &str,
        input: Map<String, Value>,
    ) -> Ran {
        let server = &self.mcp[server];
        let params =
            CallToolRequestParams::new(tool.to_string()).with_arguments(input as JsonObject);
        let outcome = tokio::time::timeout(CALL_TIMEOUT, server.service.call_tool(params)).await;
        let (is_error, text) = match outcome {
            Err(_) => (true, format!("MCP tool '{tool}' timed out")),
            Ok(Err(err)) => (true, format!("MCP tool '{tool}' failed: {err}")),
            Ok(Ok(result)) => {
                let value = serde_json::to_value(&result).unwrap_or_default();
                let text = value
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|part| match part.get("text").and_then(Value::as_str) {
                                Some(text) => text.to_string(),
                                None => part.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                (result.is_error.unwrap_or(false), text)
            }
        };
        Ran {
            block: json!({"type": "mcp_tool_result", "tool_use_id": id, "is_error": is_error,
                          "content": [{"type": "text", "text": text}]}),
            text: if is_error {
                format!("Error: {text}")
            } else {
                text
            },
        }
    }
}

/// A page's `<title>`, when it has one.
fn page_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open_end = start + lower[start..].find('>')? + 1;
    let close = open_end + lower[open_end..].find("</title>")?;
    let title = html2text::config::plain_no_decorate()
        .string_from_read(&html.as_bytes()[open_end..close], 1000)
        .ok()?;
    let title = title.trim().to_string();
    (!title.is_empty()).then_some(title)
}

/// The text a server tool result block stands for, when a client sends one
/// back in a later request — what the model read when it was produced.
pub(super) fn result_text(block: &Value) -> String {
    let content = block.get("content");
    match block.get("type").and_then(Value::as_str) {
        Some("web_search_tool_result") => match content.and_then(Value::as_array) {
            Some(results) => results
                .iter()
                .enumerate()
                .map(|(n, r)| {
                    let snippet = r
                        .get("encrypted_content")
                        .and_then(Value::as_str)
                        .and_then(|e| base64::engine::general_purpose::STANDARD.decode(e).ok())
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default();
                    format!(
                        "[{}] {}\n{}\n{snippet}\n",
                        n + 1,
                        r.get("title").and_then(Value::as_str).unwrap_or_default(),
                        r.get("url").and_then(Value::as_str).unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
            None => "Error: the search failed.".to_string(),
        },
        Some("web_fetch_tool_result") => content
            .and_then(|c| c.pointer("/content/source/data"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| "Error: the fetch failed.".to_string()),
        Some("bash_code_execution_tool_result" | "code_execution_tool_result") => {
            let field = |key: &str| {
                content
                    .and_then(|c| c.get(key))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            };
            match content
                .and_then(|c| c.get("return_code"))
                .and_then(Value::as_i64)
            {
                Some(code) => run_text(field("stdout"), field("stderr"), code as i32),
                None => format!("Error: the code did not run ({}).", field("error_code")),
            }
        }
        Some("text_editor_code_execution_tool_result") => {
            let field = |key: &str| {
                content
                    .and_then(|c| c.get(key))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            };
            match content.and_then(|c| c.get("type")).and_then(Value::as_str) {
                Some("text_editor_code_execution_view_result") => field("content").to_string(),
                Some("text_editor_code_execution_create_result") => "The file was written.".into(),
                Some("text_editor_code_execution_str_replace_result") => {
                    let lines = content
                        .and_then(|c| c.get("lines"))
                        .and_then(Value::as_array)
                        .map(|l| l.iter().filter_map(Value::as_str).collect::<Vec<_>>())
                        .unwrap_or_default();
                    format!("The file was edited:\n{}", lines.join("\n"))
                }
                _ => format!("Error: {}", field("error_message")),
            }
        }
        Some("advisor_tool_result") => super::advisor::result_text(content),
        _ => content.map(super::text_of).unwrap_or_default(),
    }
}

/// What the model reads of a command's run.
pub(super) fn run_text(stdout: &str, stderr: &str, code: i32) -> String {
    let mut text = format!("exit code: {code}");
    if !stdout.is_empty() {
        text.push_str(&format!("\nstdout:\n{stdout}"));
    }
    if !stderr.is_empty() {
        text.push_str(&format!("\nstderr:\n{stderr}"));
    }
    text
}

/// Whether a content block is a server tool's result, which a client sends
/// back inside the assistant turn that called the tool.
pub(super) fn is_result(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_tool_result"
            | "web_fetch_tool_result"
            | "mcp_tool_result"
            | "code_execution_tool_result"
            | "bash_code_execution_tool_result"
            | "text_editor_code_execution_tool_result"
            | "advisor_tool_result"
    )
}

/// Every text a request carries, for the URLs in it.
pub(super) fn conversation_text(
    messages: &[super::AnthropicMessage],
    system: Option<&Value>,
) -> String {
    let mut text = system.map(|s| s.to_string()).unwrap_or_default();
    for message in messages {
        text.push('\n');
        text.push_str(&message.content.to_string());
    }
    text
}

/// Whether a request's `tools` declare any server tool.
pub(super) fn declares_any(tools: &[Value]) -> bool {
    tools.iter().any(|t| {
        t.get("type").and_then(Value::as_str).is_some_and(|k| {
            k.starts_with("web_search_")
                || k.starts_with("web_fetch_")
                || k.starts_with("code_execution_")
                || k.starts_with("advisor_")
                || k == "mcp_toolset"
        })
    })
}

/// The `usage.server_tool_use` counts, when a server tool ran.
pub(super) fn usage(tools: &ServerTools) -> Option<Value> {
    let counts: HashMap<&str, usize> = [
        ("web_search_requests", tools.search_requests),
        ("web_fetch_requests", tools.fetch_requests),
    ]
    .into_iter()
    .collect();
    (tools.search_requests + tools.fetch_requests > 0).then(|| json!(counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_name_themselves_and_their_subdomains() {
        let limits = Limits {
            allowed_domains: vec!["example.com".into()],
            blocked_domains: vec!["private.example.com".into()],
            ..Default::default()
        };
        let url = |s: &str| reqwest::Url::parse(s).unwrap();
        assert!(limits.permits(&url("https://example.com/a")));
        assert!(limits.permits(&url("https://docs.example.com/a")));
        assert!(!limits.permits(&url("https://private.example.com/a")));
        assert!(!limits.permits(&url("https://notexample.com/a")));
    }

    #[test]
    fn max_uses_is_counted_per_request() {
        let mut limits = Limits {
            max_uses: Some(2),
            ..Default::default()
        };
        assert!(limits.take_use());
        assert!(limits.take_use());
        assert!(!limits.take_use());
    }

    #[test]
    fn urls_are_found_in_prose_without_their_punctuation() {
        let found: Vec<String> =
            urls_in("See https://a.example/x, or (http://b.example/y). Not ftp://c.").collect();
        assert_eq!(found, ["https://a.example/x", "http://b.example/y"]);
    }

    #[tokio::test]
    async fn web_search_without_a_backend_is_refused_up_front() {
        let tools = [json!({"type": "web_search_20250305", "name": "web_search"})];
        let refused = ServerTools::from_request(&tools, &[], "", "off", "off").await;
        assert!(
            refused
                .err()
                .is_some_and(|e| e.contains("[orangu-server].web_search"))
        );
        let configured =
            ServerTools::from_request(&tools, &[], "", "http://127.0.0.1:1", "off").await;
        assert!(configured.is_ok_and(|t| t.kind("web_search") == Some(Kind::Search)));
    }

    #[tokio::test]
    async fn fetch_refuses_a_url_the_conversation_never_showed() {
        let tools = [json!({"type": "web_fetch_20250910", "name": "web_fetch"})];
        let mut server =
            ServerTools::from_request(&tools, &[], "read https://example.com/ok", "off", "off")
                .await
                .unwrap();
        let call = |url: &str| ParsedToolCall {
            name: "web_fetch".into(),
            arguments: json!({"url": url}).to_string(),
        };
        let ran = server
            .run(&Kind::Fetch, "srvtoolu_1", &call("http://127.0.0.1:22/"))
            .await;
        assert_eq!(ran.block["content"]["error_code"], "url_not_allowed");
        assert_eq!(server.fetch_requests, 0);
    }

    #[test]
    fn a_returned_search_result_reads_back_as_what_the_model_saw() {
        let block = json!({"type": "web_search_tool_result", "tool_use_id": "s", "content": [{
            "type": "web_search_result", "url": "https://x.example", "title": "X",
            "encrypted_content": base64::engine::general_purpose::STANDARD.encode("the excerpt"),
        }]});
        let text = result_text(&block);
        assert!(
            text.contains("https://x.example") && text.contains("the excerpt"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn code_execution_without_a_sandbox_is_refused_up_front() {
        let tools = [json!({"type": "code_execution_20250825", "name": "code_execution"})];
        let refused = ServerTools::from_request(&tools, &[], "", "off", "off").await;
        assert!(
            refused
                .err()
                .unwrap()
                .contains("[orangu-server].code_execution")
        );
        let declared = ServerTools::from_request(&tools, &[], "", "off", "bwrap")
            .await
            .ok()
            .unwrap();
        let names: Vec<Value> = declared
            .definitions()
            .iter()
            .map(|d| d["function"]["name"].clone())
            .collect();
        assert_eq!(
            names,
            [
                json!("bash_code_execution"),
                json!("text_editor_code_execution")
            ]
        );
        assert_eq!(declared.kind("bash_code_execution"), Some(Kind::Bash));
        assert_eq!(declared.kind("code_execution"), None);
    }

    #[test]
    fn a_returned_code_result_reads_back_as_what_the_model_saw() {
        let ran = json!({"type": "bash_code_execution_tool_result", "tool_use_id": "s", "content": {
            "type": "bash_code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0,
            "content": [],
        }});
        assert_eq!(result_text(&ran), "exit code: 0\nstdout:\n42\n");
        let viewed = json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": "s",
            "content": {"type": "text_editor_code_execution_view_result", "file_type": "text",
                        "content": "print(1)"}});
        assert_eq!(result_text(&viewed), "print(1)");
        assert!(is_result("bash_code_execution_tool_result"));
        assert!(!is_result("tool_result"));
    }
}

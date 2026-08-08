// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Streamable HTTP Model Context Protocol client support.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Arc, RwLock},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, JsonObject},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::{
    config::{McpApprovalMode, McpServerConfiguration},
    llm::{FunctionDefinition, ToolDefinition},
};

const MAX_TOOL_NAME: usize = 64;
const MAX_RESULT_CHARS: usize = 20_000;

struct McpServer {
    service: RunningService<RoleClient, ()>,
    timeout: Duration,
    instructions: String,
}

#[derive(Clone)]
struct Route {
    server: Arc<McpServer>,
    server_name: String,
    tool_name: String,
    approval_required: bool,
}

/// The live MCP services and their translated function definitions for one
/// workspace. It is shared by clones of a [`ToolExecutor`](crate::tools::ToolExecutor).
pub struct McpManager {
    configurations: RwLock<HashMap<String, McpServerConfiguration>>,
    refresh_lock: Mutex<()>,
    state: RwLock<McpState>,
}

struct McpState {
    definitions: Vec<ToolDefinition>,
    routes: HashMap<String, Route>,
    status: Vec<String>,
    instructions: String,
}

impl McpManager {
    /// Connect to every configured MCP server. A single failure is returned as
    /// a warning instead of preventing the workspace from opening.
    pub async fn connect_all(
        configurations: &HashMap<String, McpServerConfiguration>,
        _workspace: &Path,
    ) -> Result<(Arc<Self>, Vec<String>)> {
        let (state, warnings) = build_state(configurations).await?;
        Ok((
            Arc::new(Self {
                configurations: RwLock::new(configurations.clone()),
                refresh_lock: Mutex::new(()),
                state: RwLock::new(state),
            }),
            warnings,
        ))
    }

    /// Re-read the configured servers and atomically replace the exposed tool
    /// set. Dropping the previous state closes its HTTP services.
    pub async fn refresh(
        &self,
        configurations: &HashMap<String, McpServerConfiguration>,
    ) -> Result<Vec<String>> {
        let _refresh = self.refresh_lock.lock().await;
        let (state, warnings) = build_state(configurations).await?;
        *self
            .configurations
            .write()
            .map_err(|_| anyhow!("MCP configuration lock poisoned"))? = configurations.clone();
        *self
            .state
            .write()
            .map_err(|_| anyhow!("MCP state lock poisoned"))? = state;
        Ok(warnings)
    }

    /// Start a non-blocking refresh for the synchronous slash-command path.
    pub fn refresh_in_background(
        self: &Arc<Self>,
        configurations: HashMap<String, McpServerConfiguration>,
    ) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = manager.refresh(&configurations).await
                && let Ok(mut state) = manager.state.write()
            {
                state.status.push(format!("× refresh failed: {error:#}"));
            }
        });
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.state
            .read()
            .map(|state| state.definitions.clone())
            .unwrap_or_default()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.state
            .read()
            .is_ok_and(|state| state.routes.contains_key(name))
    }

    pub fn requires_approval(&self, name: &str) -> bool {
        self.state
            .read()
            .ok()
            .and_then(|state| state.routes.get(name).cloned())
            .is_some_and(|route| route.approval_required)
    }

    pub fn status(&self) -> String {
        self.state
            .read()
            .map(|state| {
                if state.status.is_empty() {
                    "No MCP servers configured.".to_string()
                } else {
                    state.status.join("\n")
                }
            })
            .unwrap_or_else(|_| "MCP status unavailable.".to_string())
    }

    pub fn instructions(&self) -> String {
        self.state
            .read()
            .map(|state| state.instructions.clone())
            .unwrap_or_default()
    }

    pub async fn execute(
        &self,
        exposed_name: &str,
        arguments: &Map<String, Value>,
    ) -> Result<String> {
        let route = self.route(exposed_name)?;
        let result = match call_route(&route, arguments).await {
            Ok(result) => result,
            Err(original_error) => {
                self.reconnect().await.with_context(|| {
                    format!(
                        "MCP server '{}' failed while running '{}': {original_error}; reconnect failed",
                        route.server_name, route.tool_name
                    )
                })?;
                let route = self.route(exposed_name)?;
                call_route(&route, arguments).await.with_context(|| {
                    format!(
                        "MCP server '{}' did not recover while running '{}'",
                        route.server_name, route.tool_name
                    )
                })?
            }
        };
        let rendered = render_result(&result)?;
        if result.is_error.unwrap_or(false) {
            return Err(anyhow!("MCP tool error: {rendered}"));
        }
        Ok(rendered)
    }

    fn route(&self, exposed_name: &str) -> Result<Route> {
        self.state
            .read()
            .map_err(|_| anyhow!("MCP state lock poisoned"))?
            .routes
            .get(exposed_name)
            .cloned()
            .ok_or_else(|| anyhow!("unknown MCP tool '{exposed_name}'"))
    }

    async fn reconnect(&self) -> Result<()> {
        let configurations = self
            .configurations
            .read()
            .map_err(|_| anyhow!("MCP configuration lock poisoned"))?
            .clone();
        self.refresh(&configurations).await.map(|_| ())
    }
}

async fn build_state(
    configurations: &HashMap<String, McpServerConfiguration>,
) -> Result<(McpState, Vec<String>)> {
    let mut names = configurations.keys().cloned().collect::<Vec<_>>();
    names.sort();

    let mut definitions = Vec::new();
    let mut routes = HashMap::new();
    let mut warnings = Vec::new();
    let mut status = Vec::new();
    let mut instructions = Vec::new();
    let mut used_names = HashSet::new();

    for server_name in names {
        let configuration = &configurations[&server_name];
        if !configuration.enabled {
            status.push(format!("○ {server_name}: disabled by configuration"));
            continue;
        }
        if configuration.approval_mode == McpApprovalMode::Deny {
            status.push(format!("○ {server_name}: denied by approval policy"));
            continue;
        }
        match connect_server(configuration).await {
            Ok((server, tools)) => {
                if !server.instructions.is_empty() {
                    instructions.push(format!(
                        "# MCP server instructions: {server_name}\n{}",
                        server.instructions
                    ));
                }
                let tool_count = tools.len();
                let server = Arc::new(server);
                for tool in tools {
                    let original_name = tool.name.to_string();
                    if (!configuration.enabled_tools.is_empty()
                        && !configuration.enabled_tools.contains(&original_name))
                        || configuration.disabled_tools.contains(&original_name)
                    {
                        continue;
                    }
                    let exposed_name = exposed_tool_name(&server_name, &original_name);
                    if !used_names.insert(exposed_name.clone()) {
                        warnings.push(format!(
                                "MCP server '{server_name}' tool '{original_name}' was disabled because its exposed name '{exposed_name}' conflicts with another MCP tool"
                            ));
                        continue;
                    }
                    let description = tool
                        .description
                        .as_deref()
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("MCP tool {original_name} from {server_name}"));
                    definitions.push(ToolDefinition {
                        tool_type: "function".to_string(),
                        function: FunctionDefinition {
                            name: exposed_name.clone(),
                            description: format!("[MCP: {server_name}] {description}"),
                            parameters: Value::Object((*tool.input_schema).clone()),
                        },
                    });
                    routes.insert(
                        exposed_name,
                        Route {
                            server: server.clone(),
                            server_name: server_name.clone(),
                            tool_name: original_name,
                            approval_required: match configuration.approval_mode {
                                McpApprovalMode::Prompt => true,
                                McpApprovalMode::Writes => {
                                    tool.annotations
                                        .as_ref()
                                        .and_then(|annotations| annotations.read_only_hint)
                                        != Some(true)
                                }
                                McpApprovalMode::Auto | McpApprovalMode::Deny => false,
                            },
                        },
                    );
                }
                status.push(format!("● {server_name}: connected ({tool_count} tools)"));
            }
            Err(error) => {
                if configuration.required {
                    return Err(anyhow!(
                        "required MCP server '{server_name}' failed: {error:#}"
                    ));
                }
                warnings.push(format!("MCP server '{server_name}' disabled: {error:#}"));
                status.push(format!("× {server_name}: {error}"));
            }
        }
    }
    definitions.sort_by(|left, right| left.function.name.cmp(&right.function.name));
    Ok((
        McpState {
            definitions,
            routes,
            status,
            instructions: instructions.join("\n\n"),
        },
        warnings,
    ))
}

async fn call_route(
    route: &Route,
    arguments: &Map<String, Value>,
) -> Result<rmcp::model::CallToolResult> {
    let params = CallToolRequestParams::new(route.tool_name.clone())
        .with_arguments(arguments.clone() as JsonObject);
    tokio::time::timeout(route.server.timeout, route.server.service.call_tool(params))
        .await
        .map_err(|_| {
            anyhow!(
                "MCP server '{}' tool '{}' timed out after {}s",
                route.server_name,
                route.tool_name,
                route.server.timeout.as_secs()
            )
        })?
        .map_err(|error| {
            anyhow!(
                "MCP server '{}' tool '{}' failed: {error}",
                route.server_name,
                route.tool_name
            )
        })
}

async fn connect_server(
    configuration: &McpServerConfiguration,
) -> Result<(McpServer, Vec<rmcp::model::Tool>)> {
    let startup_timeout = Duration::from_secs(configuration.startup_timeout_seconds);
    let service = connect_http_server(configuration, startup_timeout).await?;
    let tools = tokio::time::timeout(startup_timeout, service.list_all_tools())
        .await
        .map_err(|_| {
            anyhow!(
                "tool discovery timed out after {}s",
                startup_timeout.as_secs()
            )
        })?
        .map_err(|error| anyhow!("tool discovery failed: {error}"))?;
    let instructions = service
        .peer_info()
        .and_then(|info| info.instructions.clone())
        .unwrap_or_default();
    Ok((
        McpServer {
            service,
            timeout: Duration::from_secs(configuration.tool_timeout_seconds),
            instructions,
        },
        tools,
    ))
}

async fn connect_http_server(
    configuration: &McpServerConfiguration,
    startup_timeout: Duration,
) -> Result<RunningService<RoleClient, ()>> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(configuration.endpoint.clone()),
    );
    tokio::time::timeout(startup_timeout, ().serve(transport))
        .await
        .map_err(|_| {
            anyhow!(
                "initialization timed out after {}s",
                startup_timeout.as_secs()
            )
        })?
        .map_err(|error| anyhow!("initialization failed: {error}"))
}

fn exposed_tool_name(server: &str, original: &str) -> String {
    let sanitized = sanitize_name(original);
    let candidate = format!("mcp__{server}__{sanitized}");
    if candidate.len() <= MAX_TOOL_NAME && sanitized == original {
        return candidate;
    }
    let mut hasher = Sha256::new();
    hasher.update(server.as_bytes());
    hasher.update([0]);
    hasher.update(original.as_bytes());
    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let prefix = candidate.chars().take(52).collect::<String>();
    format!("{prefix}__{}", &hash[..10])
}

fn sanitize_name(name: &str) -> String {
    let mut output = String::new();
    let mut previous_replaced = false;
    for character in name.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
            output.push(character);
            previous_replaced = false;
        } else if !previous_replaced {
            output.push('_');
            previous_replaced = true;
        }
    }
    output
}

fn render_result(result: &rmcp::model::CallToolResult) -> Result<String> {
    let rendered = serde_json::to_string_pretty(result)?;
    Ok(truncate(&rendered, MAX_RESULT_CHARS))
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    format!(
        "{}\n\n[truncated]",
        text.chars().take(limit).collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaced_names_are_stable_and_safe() {
        assert_eq!(
            exposed_tool_name("files", "read_file"),
            "mcp__files__read_file"
        );
        let name = exposed_tool_name("files", "read.file");
        assert!(name.starts_with("mcp__files__read_file__"));
        assert!(name.len() <= MAX_TOOL_NAME);
    }
}

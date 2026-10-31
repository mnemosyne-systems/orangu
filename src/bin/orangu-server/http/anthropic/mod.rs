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

//! Anthropic-compatible endpoints: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`.
//!
//! A Messages request is translated into the same [`ChatMessage`] list and
//! OpenAI-shaped tool array `/v1/chat/completions` renders, so both front
//! doors put the same prompt in front of the model; only the wire shapes on
//! either side differ. `anthropic-version` and `anthropic-beta` are accepted
//! and not read: there is one behaviour to version.

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, sse::Event},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod advisor;
pub mod agents;
pub mod batches;
mod citations;
mod containers;
pub mod deployments;
mod egress;
pub mod files;
pub mod memory;
mod programmatic;
mod server_tools;
pub mod sessions;
pub mod skills;
mod toolset;
pub mod vaults;
pub mod webhooks;

use super::AppState;
use super::openai::{
    DEFAULT_MAX_TOKENS, RenderOptions, RenderedChat, RequestSampling, default_cache_prompt,
    reject_unknown_slot, render_chat, request_role, sampling_for, unix_now, with_server_timing,
};
use crate::engine::chat_template::{ChatMessage, Reasoning};
use crate::engine::generate::{FinishReason, GenerateRequest, GenerateStats, StreamEvent};
use crate::engine::tool_calls;

/// The largest `/v1/messages` body accepted, as Anthropic's own limit.
pub const REQUEST_BODY_LIMIT: usize = 32 * 1024 * 1024;

#[derive(Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(flatten)]
    prompt: PromptFields,
    /// Required by Anthropic; optional here, defaulting the way
    /// `/v1/chat/completions` does.
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    /// The sampler fields Anthropic's schema has no name for, under the names
    /// `/v1/chat/completions` uses.
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
    #[serde(default)]
    id_slot: Option<usize>,
}

/// The fields that decide the prompt — all `POST /v1/messages/count_tokens`
/// takes, and the part of a `POST /v1/messages` request both share.
#[derive(Deserialize)]
pub struct PromptFields {
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    /// A string, or an array of `text` blocks.
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    /// `{"type": "enabled" | "adaptive" | "disabled", "budget_tokens": N}`.
    #[serde(default)]
    thinking: Option<Value>,
    /// `{"effort": "low" | "medium" | "high" | "max", "format": {"type":
    /// "json_schema", "schema": …}}`.
    #[serde(default)]
    output_config: Option<Value>,
    /// `output_config.format` under its earlier name.
    #[serde(default)]
    output_format: Option<Value>,
    /// Remote MCP servers whose tools this request may call:
    /// `[{"type": "url", "url": …, "name": …, "authorization_token": …}]`.
    #[serde(default)]
    mcp_servers: Vec<Value>,
    /// The code execution container to run in: its id, or `{"id": …,
    /// "skills": [{"type": "custom", "skill_id": …, "version": …}]}`.
    #[serde(default)]
    container: Option<Value>,
}

#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    /// A string, or an array of content blocks.
    #[serde(default)]
    content: Value,
}

/// Anthropic's error envelope, `{"type": "error", "error": {...}}`, with the
/// `type` its SDKs map to an exception class.
pub(crate) fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = match status.as_u16() {
        400 | 409 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    (
        status,
        Json(json!({
            "type": "error",
            "error": {"type": kind, "message": message.into()},
        })),
    )
        .into_response()
}

/// Rewraps a plain-text rejection from a shared helper in Anthropic's error
/// envelope, keeping its status and headers (`Retry-After` on a full queue).
async fn rewrap(response: Response) -> Response {
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 64 * 1024)
        .await
        .unwrap_or_default();
    let message = String::from_utf8_lossy(&bytes).trim().to_string();
    let mut wrapped = error_response(parts.status, message);
    for (name, value) in &parts.headers {
        if name != axum::http::header::CONTENT_TYPE && name != axum::http::header::CONTENT_LENGTH {
            wrapped.headers_mut().insert(name, value.clone());
        }
    }
    wrapped
}

/// A body that did not parse is a `400`, as Anthropic answers it; one over
/// the size limit keeps its `413`.
fn json_rejection(rejection: JsonRejection) -> Response {
    let status = match rejection.status() {
        StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
        _ => StatusCode::BAD_REQUEST,
    };
    error_response(status, rejection.body_text())
}

/// The text of a `system` field or a `tool_result`'s `content`: a string, or
/// the `text` blocks of an array joined with newlines.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A `document` block's text: a `text` source's data, a `content` source's
/// text blocks, or the text a base64 PDF carries, under its `title` when it
/// has one. `None` for a source with no text to give.
fn document_text(block: &Value) -> Option<String> {
    let source = block.get("source")?;
    let text = match source.get("type").and_then(Value::as_str)? {
        "text" => source.get("data")?.as_str()?.to_string(),
        "content" => text_of(source.get("content")?),
        "base64" if source.get("media_type").and_then(Value::as_str) == Some("application/pdf") => {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(source.get("data")?.as_str()?)
                .ok()?;
            orangu::pdf::extract_text(&bytes).ok()?
        }
        _ => return None,
    };
    Some(match block.get("title").and_then(Value::as_str) {
        Some(title) => format!("{title}\n\n{text}"),
        None => text,
    })
}

/// An `image` block's picture as the URL `ChatMessage::images` carries: a
/// `data:` URL for a base64 source, the URL itself for a `url` source.
fn image_url(block: &Value) -> Option<String> {
    let source = block.get("source")?;
    match source.get("type").and_then(Value::as_str)? {
        "base64" => Some(format!(
            "data:{};base64,{}",
            source.get("media_type").and_then(Value::as_str)?,
            source.get("data").and_then(Value::as_str)?
        )),
        "url" => source
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

/// The Messages conversation as the chat template's message list.
///
/// `system` becomes a leading system message. A `tool_use` block becomes an
/// entry in the assistant turn's OpenAI-shaped `tool_calls`, and a
/// `tool_result` block becomes a `tool` message of its own, named after the
/// call it answers — ahead of any text in the same user turn, since the
/// result is what that text follows. `thinking` blocks are dropped: a
/// template is handed the answers of earlier turns, not their reasoning.
fn to_chat_messages(
    system: Option<&Value>,
    messages: &[AnthropicMessage],
) -> (Vec<ChatMessage>, Option<String>) {
    let mut out = Vec::new();
    if let Some(system) = system {
        let text = text_of(system);
        if !text.is_empty() {
            out.push(ChatMessage::text("system", &text));
        }
    }
    let mut call_names: HashMap<String, String> = HashMap::new();
    for message in messages {
        let blocks = match &message.content {
            Value::String(s) => {
                out.push(ChatMessage::text(&message.role, s));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => {
                out.push(ChatMessage::text(&message.role, ""));
                continue;
            }
        };
        let mut turn = ChatMessage::text(&message.role, "");
        let mut calls = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        if !turn.content.is_empty() {
                            turn.content.push('\n');
                        }
                        turn.content.push_str(text);
                    }
                }
                Some("document") => {
                    if let Some(text) = document_text(block) {
                        if !turn.content.is_empty() {
                            turn.content.push('\n');
                        }
                        turn.content.push_str(&text);
                    }
                }
                Some("image") => {
                    if let Some(url) = image_url(block) {
                        turn.push_image(url);
                    }
                }
                Some("tool_use" | "server_tool_use" | "mcp_tool_use") => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    call_names.insert(id.to_string(), name.to_string());
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": input.to_string()},
                    }));
                }
                // A server tool's result sits inside the assistant turn that
                // called it: the turn so far, the result, and then the rest
                // of the turn as a turn of its own.
                Some(kind) if server_tools::is_result(kind) => {
                    let id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if !calls.is_empty() {
                        turn.tool_calls = Some(Value::Array(std::mem::take(&mut calls)));
                    }
                    let finished =
                        std::mem::replace(&mut turn, ChatMessage::text(&message.role, ""));
                    if !finished.content.is_empty() || finished.tool_calls.is_some() {
                        out.push(finished);
                    }
                    out.push(ChatMessage {
                        role: "tool".to_string(),
                        content: server_tools::result_text(block),
                        tool_call_id: Some(id.to_string()),
                        name: call_names.get(id).cloned(),
                        ..Default::default()
                    });
                }
                Some("tool_result") => {
                    let id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let mut content = block.get("content").map(text_of).unwrap_or_default();
                    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                        content = format!("Error: {content}");
                    }
                    let mut result = ChatMessage {
                        role: "tool".to_string(),
                        content,
                        tool_call_id: Some(id.to_string()),
                        name: call_names.get(id).cloned(),
                        ..Default::default()
                    };
                    // A result's pictures follow its text.
                    let parts = block.get("content").and_then(Value::as_array);
                    for part in parts.into_iter().flatten() {
                        if part.get("type").and_then(Value::as_str) == Some("image")
                            && let Some(url) = image_url(part)
                        {
                            result.push_image(url);
                        }
                    }
                    out.push(result);
                }
                _ => {}
            }
        }
        if !calls.is_empty() {
            turn.tool_calls = Some(Value::Array(calls));
        }
        // A user turn that carried only tool results has nothing left to say.
        if !turn.content.is_empty() || !turn.images.is_empty() || turn.tool_calls.is_some() {
            out.push(turn);
        }
    }
    // A conversation that ends on assistant text is a reply to continue, not
    // a turn to answer. Anthropic refuses one ending in whitespace; it is
    // trimmed here instead.
    let ends_on_assistant_text = messages.last().is_some_and(|m| m.role == "assistant")
        && out
            .last()
            .is_some_and(|m| m.role == "assistant" && m.tool_calls.is_none());
    let continue_from = ends_on_assistant_text
        .then(|| out.pop())
        .flatten()
        .map(|m| m.content.trim_end().to_string());
    (out, continue_from)
}

/// The input schema of an Anthropic-defined client tool, by the prefix of
/// its versioned `type`. The client runs these tools; the model only has to
/// be told how to call them.
fn client_tool_schema(kind: &str) -> Option<(&'static str, Value)> {
    let string = json!({"type": "string"});
    let integer = json!({"type": "integer"});
    let path_commands = |commands: &[&str]| json!({"type": "string", "enum": commands});
    if kind.starts_with("bash_") {
        return Some((
            "Run a shell command in a persistent bash session.",
            json!({"type": "object", "properties": {
                "command": string, "restart": {"type": "boolean"},
            }}),
        ));
    }
    if kind.starts_with("text_editor_") {
        return Some((
            "View, create and edit files.",
            json!({"type": "object", "required": ["command", "path"], "properties": {
                "command": path_commands(&["view", "create", "str_replace", "insert", "undo_edit"]),
                "path": string, "file_text": string, "old_str": string, "new_str": string,
                "insert_line": integer, "insert_text": string,
                "view_range": {"type": "array", "items": integer},
            }}),
        ));
    }
    if kind.starts_with("memory_") {
        return Some((
            "Read and write files in the /memories directory, which persists between conversations.",
            json!({"type": "object", "required": ["command"], "properties": {
                "command": path_commands(&["view", "create", "str_replace", "insert", "delete", "rename"]),
                "path": string, "file_text": string, "old_str": string, "new_str": string,
                "insert_line": integer, "insert_text": string,
                "old_path": string, "new_path": string,
                "view_range": {"type": "array", "items": integer},
            }}),
        ));
    }
    if kind.starts_with("computer_") {
        return Some((
            "Operate the computer's screen, mouse and keyboard.",
            json!({"type": "object", "required": ["action"], "properties": {
                "action": string, "coordinate": {"type": "array", "items": integer},
                "text": string, "scroll_direction": string, "scroll_amount": integer,
                "duration": {"type": "number"}, "key": string,
            }}),
        ));
    }
    None
}

/// Anthropic's tool array in the OpenAI shape the chat template is handed.
/// Custom tools carry their own schema; Anthropic-defined client tools
/// (`bash_…`, `text_editor_…`, `memory_…`, `computer_…`) get theirs from
/// [`client_tool_schema`]. Server tools (`web_search_…`, `code_execution_…`
/// and the rest) are left out: `server_tools` declares the ones it runs.
/// `tool_choice: {"type": "none"}` drops them all.
fn to_openai_tools(tools: Option<&[Value]>, tool_choice: Option<&Value>) -> Option<Value> {
    if tool_choice
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("none")
    {
        return None;
    }
    let tools: Vec<Value> = tools?
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let (description, parameters) = match t.get("type").and_then(Value::as_str) {
                None | Some("custom") => (
                    t.get("description").cloned(),
                    t.get("input_schema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                ),
                Some(kind) => {
                    let (description, schema) = client_tool_schema(kind)?;
                    (Some(json!(description)), schema)
                }
            };
            let mut function = json!({"name": name, "parameters": parameters});
            if let Some(description) = description {
                function["description"] = description;
            }
            Some(json!({"type": "function", "function": function}))
        })
        .collect();
    (!tools.is_empty()).then_some(Value::Array(tools))
}

/// The name a probe call is rendered under, distinct enough never to occur
/// in a template's own text.
const PROBE_TOOL: &str = "orangu_probe_tool";

/// What a reply must begin with to be a tool call — of `name`, or of any
/// tool when `name` is `None` — in this model's own call syntax.
///
/// Found by rendering one call through the chat template and reading back
/// where it starts: the opener (`tool_calls`' formats), and for a named
/// tool everything up to the end of the name. `None` when the template
/// renders no call this server recognises, and the choice is then the
/// model's.
fn tool_call_prefix(state: &AppState, name: Option<&str>) -> Option<String> {
    let template = state.engine.chat_template.as_ref()?;
    let probe_tools = json!([{"type": "function", "function": {
        "name": PROBE_TOOL, "parameters": {"type": "object", "properties": {}},
    }}]);
    let mut call = ChatMessage::text("assistant", "");
    call.tool_calls = Some(json!([{
        "id": "probe", "type": "function",
        "function": {"name": PROBE_TOOL, "arguments": "{}"},
    }]));
    let rendered = template
        .render_with_tools(
            &[ChatMessage::text("user", "probe"), call],
            false,
            "",
            "",
            Default::default(),
            Some(&probe_tools),
        )
        .ok()?;
    // The call is the probe's last mention — the tool list and a template's
    // own instructions, which can spell the opener too, come before it.
    let call_at = rendered.rfind(PROBE_TOOL)?;
    let (at, opener) = tool_calls::last_opener(&rendered[..call_at])?;
    match name {
        None => Some(opener.to_string()),
        Some(name) => {
            let end = at + rendered[at..].find(PROBE_TOOL)? + PROBE_TOOL.len();
            Some(rendered[at..end].replace(PROBE_TOOL, name))
        }
    }
}

/// `thinking.budget_tokens` of a request that turns thinking on: the most
/// tokens each turn of the reply may reason for.
fn thinking_budget(fields: &PromptFields) -> Option<usize> {
    let thinking = fields.thinking.as_ref()?;
    if thinking.get("type").and_then(Value::as_str) != Some("enabled") {
        return None;
    }
    thinking
        .get("budget_tokens")
        .and_then(Value::as_u64)
        .map(|budget| budget.max(1) as usize)
}

/// The reasoning a request's `thinking` and `output_config.effort` ask for,
/// and whether to close the reasoning channel — `None` when they ask for the
/// role's own.
///
/// `disabled` turns reasoning off the way the review role does; `enabled`
/// and `adaptive` turn it on. Effort names pass to the template as its
/// `reasoning_effort`, `max` as `high`; a `budget_tokens` without an
/// effort picks one by size.
fn reasoning_for<'a>(
    state: &'a AppState,
    role: crate::config::Role,
    fields: &'a PromptFields,
) -> Option<(Reasoning<'a>, bool)> {
    let thinking = fields.thinking.as_ref();
    let kind = thinking.and_then(|t| t.get("type")).and_then(Value::as_str);
    let effort = fields
        .output_config
        .as_ref()
        .and_then(|c| c.get("effort"))
        .and_then(Value::as_str)
        .map(|e| match e {
            "max" | "xhigh" => "high",
            other => other,
        })
        .or_else(|| {
            let budget = thinking?.get("budget_tokens")?.as_u64()?;
            Some(match budget {
                ..4096 => "low",
                4096..16384 => "medium",
                _ => "high",
            })
        });
    if kind.is_none() && effort.is_none() {
        return None;
    }
    let mut reasoning = state.engine.reasoning_as(role);
    let mut suppress = role.suppresses_reasoning();
    match kind {
        Some("disabled") => {
            reasoning.enabled = Some(false);
            suppress = true;
        }
        Some("enabled" | "adaptive") => {
            reasoning.enabled = Some(true);
            suppress = false;
        }
        _ => {}
    }
    if let Some(effort) = effort {
        reasoning.effort = Some(effort);
        reasoning.effort_is_default = false;
    }
    Some((reasoning, suppress))
}

/// Holds back generated text that might be the start of a stop sequence,
/// and reports the first one that is completed.
struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
        }
    }

    /// Takes the next piece of text. Returns what is safe to emit and, once
    /// a stop sequence has been written, which one — the text before it is
    /// the last there is.
    fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_string(), None);
        }
        self.held.push_str(text);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = first {
            let stop = stop.clone();
            let emit = self.held[..at].to_string();
            self.held.clear();
            return (emit, Some(stop));
        }
        // Keep the longest tail of `held` that some stop sequence starts with.
        let keep = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &self.held[i..];
                self.stops.iter().any(|s| s.starts_with(tail))
            })
            .unwrap_or(self.held.len());
        let emit = self.held[..keep].to_string();
        self.held.drain(..keep);
        (emit, None)
    }

    /// What is still held back when generation ends: it never became a stop
    /// sequence, so it belongs to the answer.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn stop_reason(finish: FinishReason, saw_calls: bool, stop: Option<&str>) -> &'static str {
    if stop.is_some() {
        "stop_sequence"
    } else if saw_calls {
        "tool_use"
    } else {
        match finish {
            FinishReason::Stop => "end_turn",
            FinishReason::Length => "max_tokens",
        }
    }
}

/// A message id unique within this process, and the stem its tool-use ids
/// share — two turns started in the same second must not hand a client the
/// same `tool_use` id.
fn next_message_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "msg_{:x}{:06x}",
        unix_now(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn tool_use_block(message_id: &str, index: usize, call: &tool_calls::ParsedToolCall) -> Value {
    let input: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
    json!({
        "type": "tool_use",
        "id": format!("toolu_{}_{index}", &message_id["msg_".len()..]),
        "name": call.name,
        "input": input,
    })
}

/// Unix seconds as Anthropic writes a timestamp.
pub(crate) fn rfc3339(seconds: u64) -> String {
    chrono::DateTime::from_timestamp(seconds as i64, 0)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn with_request_id(mut response: Response, id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(id) {
        response.headers_mut().insert("request-id", value);
    }
    response
}

/// How many times one reply may run server tools and go on before it
/// stops with `pause_turn`, for the client to send it back and continue.
const MAX_SERVER_TURNS: usize = 10;

/// A request's conversation made ready for the template: its messages, the
/// tools it declares — the client's and the server's — and how its reply is
/// shaped.
struct Prompt {
    role: crate::config::Role,
    fields: PromptFields,
    chat: Vec<ChatMessage>,
    /// An assistant turn the first reply continues — see `RenderOptions`.
    continue_from: Option<String>,
    tools: Option<Value>,
    server: server_tools::ServerTools,
    /// The tool-call opener the first reply is made to begin with, for a
    /// `tool_choice` that requires a call.
    prefix: Option<String>,
    /// `disable_parallel_tool_use`: the reply makes at most one client call.
    single_call: bool,
    /// The schema the reply is held to, from `output_config.format`.
    json_schema: Option<Arc<crate::engine::json_schema::Schema>>,
    /// The response's `container`, when the request runs code.
    container: Option<Value>,
    /// The client's tools, by name: a call to any other name that is not a
    /// server tool's names no tool at all.
    client_tools: Vec<String>,
    /// Every tool the model is shown, by name.
    tool_names: Vec<String>,
    /// The numbered passages of cited documents and search results.
    citations: Option<Arc<citations::Sources>>,
    /// The `input_schema` of each client tool declared `strict: true`, by
    /// name: a call to one is held to it.
    strict: HashMap<String, Value>,
    /// What running code did with the client's results this request
    /// carries, when it answers calls the code made.
    resumed: Option<Resumed>,
}

/// Paused code, given the client's results.
enum Resumed {
    /// It called more tools: the reply is these calls.
    Paused(Vec<Value>),
    /// It ended: the reply opens with this result, and the model goes on.
    Finished(Value),
}

/// The text of a `tool_result` answering a call from code: a string, or
/// text blocks.
fn code_result_text(block: &Value) -> Result<String, String> {
    match &block["content"] {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) if parts.iter().all(|p| p["type"] == "text") => Ok(parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n")),
        Value::Null => Ok(String::new()),
        _ => Err("a tool_result answering a call from code holds only text".into()),
    }
}

/// When the conversation ends on the client's results for calls running
/// code made, hands them to that code and runs it on: either it calls
/// more tools, or it ends — and then its result is put into the
/// conversation after the call that ran it.
async fn resume_code(messages: &mut [AnthropicMessage]) -> Result<Option<Resumed>, Box<Response>> {
    let bad_request = |message: String| Box::new(error_response(StatusCode::BAD_REQUEST, message));
    let [.., asked, answered] = messages else {
        return Ok(None);
    };
    if asked.role != "assistant" || answered.role != "user" {
        return Ok(None);
    }
    let calls: Vec<&Value> = asked
        .content
        .as_array()
        .into_iter()
        .flatten()
        .filter(|b| programmatic::is_programmatic(b))
        .collect();
    let Some(caller) = calls
        .first()
        .and_then(|c| c["caller"]["tool_id"].as_str())
        .map(str::to_string)
    else {
        return Ok(None);
    };
    let blocks = answered.content.as_array().cloned().unwrap_or_default();
    if blocks.iter().any(|b| b["type"] != "tool_result") {
        return Err(bad_request(
            "a message answering calls from running code holds only their tool_result blocks"
                .into(),
        ));
    }
    let mut results = Vec::new();
    for block in &blocks {
        let id = block["tool_use_id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        results.push((id, code_result_text(block).map_err(bad_request)?));
    }
    let Some(mut run) = programmatic::take(&caller) else {
        return Err(bad_request(format!(
            "the code that made these calls ({caller}) is no longer running: its results came \
             too late, or the server restarted"
        )));
    };
    match run.resume(results).await {
        programmatic::Step::Calls(more) => {
            let blocks = run.blocks(&caller, more);
            programmatic::pause(&caller, run);
            Ok(Some(Resumed::Paused(blocks)))
        }
        step => {
            let (block, _) = programmatic::result_block(&caller, &step);
            for message in messages.iter_mut().rev() {
                let Some(content) = message.content.as_array_mut() else {
                    continue;
                };
                if let Some(at) = content.iter().position(|b| b["id"] == caller.as_str()) {
                    content.insert(at + 1, block.clone());
                    break;
                }
            }
            Ok(Some(Resumed::Finished(block)))
        }
    }
}

/// Leaves out the calls running code made and the client's results for
/// them: the model sees the code and what it printed, not the calls.
fn strip_programmatic(messages: &mut Vec<AnthropicMessage>) {
    let mut ids = std::collections::HashSet::new();
    for message in messages.iter() {
        for block in message.content.as_array().into_iter().flatten() {
            if programmatic::is_programmatic(block)
                && let Some(id) = block["id"].as_str()
            {
                ids.insert(id.to_string());
            }
        }
    }
    if ids.is_empty() {
        return;
    }
    for message in messages.iter_mut() {
        if let Some(content) = message.content.as_array_mut() {
            content.retain(|b| {
                !programmatic::is_programmatic(b)
                    && !(b["type"] == "tool_result"
                        && b["tool_use_id"].as_str().is_some_and(|id| ids.contains(id)))
            });
        }
    }
    messages.retain(|m| m.content.as_array().is_none_or(|c| !c.is_empty()));
}

/// The schema `output_config.format` (or `output_format`) names, compiled;
/// `None` when the request asks for no format.
fn output_schema(
    fields: &PromptFields,
) -> Result<Option<(Arc<crate::engine::json_schema::Schema>, Value)>, String> {
    let format = fields
        .output_config
        .as_ref()
        .and_then(|c| c.get("format"))
        .or(fields.output_format.as_ref());
    let Some(format) = format else {
        return Ok(None);
    };
    if format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return Err("output format type must be \"json_schema\"".into());
    }
    let schema = format
        .get("schema")
        .ok_or("a json_schema output format needs a `schema`")?;
    let compiled = crate::engine::json_schema::compile(schema)
        .map_err(|err| format!("output format schema: {err}"))?;
    Ok(Some((Arc::new(compiled), schema.clone())))
}

/// Adds `text` to the conversation's system message, or starts one.
fn append_system(chat: &mut Vec<ChatMessage>, text: &str) {
    match chat.first_mut().filter(|m| m.role == "system") {
        Some(system) => {
            system.content.push_str("\n\n");
            system.content.push_str(text);
        }
        None => chat.insert(0, ChatMessage::text("system", text)),
    }
}

/// `container`'s id and skills: a bare string is an id.
fn container_request(container: Option<&Value>) -> Result<(Option<String>, Vec<Value>), String> {
    match container {
        None | Some(Value::Null) => Ok((None, Vec::new())),
        Some(Value::String(id)) => Ok((Some(id.clone()), Vec::new())),
        Some(Value::Object(fields)) => Ok((
            fields.get("id").and_then(Value::as_str).map(str::to_string),
            fields
                .get("skills")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        )),
        Some(_) => Err("`container` must be a container id or an object".into()),
    }
}

/// The file ids of the conversation's `container_upload` blocks.
fn container_uploads(messages: &[AnthropicMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| m.content.as_array())
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("container_upload"))
        .filter_map(|b| b.get("file_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Each `container_upload` block as the text telling the model where its
/// file is: `paths` by file id.
fn place_uploads(messages: &mut [AnthropicMessage], paths: &HashMap<String, String>) {
    for message in messages {
        let Some(blocks) = message.content.as_array_mut() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("container_upload") {
                continue;
            }
            let id = block
                .get("file_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(path) = paths.get(id) {
                *block = json!({"type": "text",
                                "text": format!("[A file is in the code execution container at {path}.]")});
            }
        }
    }
}

/// Shared by both endpoints: the generation-capable checks, stored files
/// read in, the server tools connected, and the conversation converted.
/// `open_container` gives a request that runs code its container — what
/// counting tokens has no need of.
async fn prepare(
    state: &AppState,
    headers: &HeaderMap,
    mut fields: PromptFields,
    open_container: bool,
) -> Result<Prompt, Box<Response>> {
    if !state.engine.role.allows_generation()
        || state.engine.image.is_some()
        || state.engine.video.is_some()
        || state.engine.audio.is_some()
    {
        return Err(Box::new(error_response(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "this server is running in --{} mode; /v1/messages needs a text-generation model",
                state.engine.role.label()
            ),
        )));
    }
    files::resolve_sources(state, &mut fields.messages).await?;
    files::resolve_document_urls(&mut fields.messages).await?;
    let resumed = resume_code(&mut fields.messages).await?;
    strip_programmatic(&mut fields.messages);
    let citations = citations::number(&mut fields.messages);
    let role = request_role(state, headers);
    let bad_request = |message: String| Box::new(error_response(StatusCode::BAD_REQUEST, message));
    let declared = fields.tools.as_deref().unwrap_or_default();
    let choice = fields.tool_choice.as_ref();
    let withheld = choice.and_then(|c| c.get("type")).and_then(Value::as_str) == Some("none");
    let mut server =
        if withheld || (!server_tools::declares_any(declared) && fields.mcp_servers.is_empty()) {
            server_tools::ServerTools::from_request(&[], &[], "", "off", "off").await
        } else {
            server_tools::ServerTools::from_request(
                declared,
                &fields.mcp_servers,
                &server_tools::conversation_text(&fields.messages, fields.system.as_ref()),
                &state.web_search,
                &state.code_execution,
            )
            .await
        }
        .map_err(|message| Box::new(error_response(StatusCode::BAD_REQUEST, message)))?;
    let advises = declared.iter().any(|t| {
        t.get("type")
            .and_then(Value::as_str)
            .is_some_and(|k| k.starts_with("advisor_"))
    });
    if !advises && carries_advice(&fields.messages) {
        return Err(bad_request(
            "the conversation holds advisor_tool_result blocks: keep the advisor tool in `tools`"
                .into(),
        ));
    }
    let from_code: Vec<Value> = declared
        .iter()
        .filter(|t| programmatic::callable(t))
        .cloned()
        .collect();
    if !from_code.is_empty() {
        let versioned = declared.iter().any(|t| {
            t.get("type")
                .and_then(Value::as_str)
                .is_some_and(|k| k.starts_with("code_execution_") && k != "code_execution_20250522")
        });
        if !versioned {
            return Err(bad_request(
                "a tool whose allowed_callers name code execution needs the \
                 code_execution_20260120 tool (or later) in the request"
                    .into(),
            ));
        }
        if let Some(named) = choice.and_then(|c| c.get("name")).and_then(Value::as_str)
            && declared
                .iter()
                .any(|t| t["name"] == named && !programmatic::direct(t))
        {
            return Err(bad_request(format!(
                "tool_choice names `{named}`, which only code may call: add \"direct\" to its \
                 allowed_callers"
            )));
        }
        server.call_from_code(from_code);
    }
    let (container_id, requested_skills) =
        container_request(fields.container.as_ref()).map_err(bad_request)?;
    let uploads = container_uploads(&fields.messages);
    if !server.runs_code() && (!requested_skills.is_empty() || !uploads.is_empty()) {
        return Err(bad_request(
            "skills and container_upload blocks need the code_execution tool".into(),
        ));
    }
    let mounted = skills::resolve(&state.api_data, &requested_skills).map_err(bad_request)?;
    let mut container_json = None;
    let mut upload_paths = HashMap::new();
    if server.runs_code() && open_container {
        let mut container = containers::Container::open(&state.api_data, container_id.as_deref())
            .await
            .map_err(|message| {
                let status = if container_id.is_some() {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                Box::new(error_response(status, message))
            })?;
        container.mount_skills(
            mounted
                .iter()
                .map(|m| (m.name.clone(), m.dir.clone()))
                .collect(),
        );
        for id in &uploads {
            let path = container.upload(id).await.map_err(bad_request)?;
            upload_paths.insert(id.clone(), path);
        }
        container_json = Some(json!({
            "id": container.id,
            "expires_at": rfc3339(container.expires_at()),
            "skills": mounted
                .iter()
                .map(|m| json!({"type": "custom", "skill_id": m.skill_id, "version": m.version}))
                .collect::<Vec<_>>(),
        }));
        server.attach(container);
    } else {
        let store = files::FileStore::new(state);
        for id in &uploads {
            let name = store.get(id).map(|m| m.filename).unwrap_or_default();
            upload_paths.insert(id.clone(), format!("/workspace/{name}"));
        }
    }
    place_uploads(&mut fields.messages, &upload_paths);
    let (mut chat, continue_from) = to_chat_messages(fields.system.as_ref(), &fields.messages);
    if citations.is_some() {
        append_system(&mut chat, citations::INSTRUCTION);
    }
    if !mounted.is_empty() {
        let listing: Vec<String> = mounted
            .iter()
            .map(|m| format!("- {}: {}", m.name, m.description))
            .collect();
        append_system(
            &mut chat,
            &format!(
                "These skills are in the code execution container, each a folder under \
                 /skills. A skill is not a tool and cannot be called: to use one, first \
                 read its instructions by running `cat /skills/<name>/SKILL.md` with the \
                 bash tool, then follow them.\n{}",
                listing.join("\n")
            ),
        );
    }
    // A tool only code may call is the model's through `code_execution`.
    let direct: Vec<Value> = fields
        .tools
        .iter()
        .flatten()
        .filter(|t| programmatic::direct(t))
        .cloned()
        .collect();
    let mut tools: Vec<Value> = to_openai_tools(Some(&direct), choice)
        .and_then(|t| t.as_array().cloned())
        .unwrap_or_default();
    let name_of = |t: &Value| {
        t.pointer("/function/name")
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let client_tools: Vec<String> = tools.iter().filter_map(name_of).collect();
    let strict: HashMap<String, Value> = declared
        .iter()
        .filter(|t| t.get("strict").and_then(Value::as_bool) == Some(true))
        .filter_map(|t| {
            Some((
                t.get("name")?.as_str()?.to_string(),
                t.get("input_schema").cloned().unwrap_or_else(|| json!({})),
            ))
        })
        .collect();
    if !withheld {
        tools.extend(server.definitions());
    }
    let tool_names: Vec<String> = tools.iter().filter_map(name_of).collect();
    let tools = (!tools.is_empty()).then_some(Value::Array(tools));
    // A continued reply has already begun, so it cannot also be made to
    // begin with a call.
    let prefix = match (&tools, &continue_from) {
        (Some(_), None) => match choice.and_then(|c| c.get("type")).and_then(Value::as_str) {
            Some("any") => tool_call_prefix(state, None),
            Some("tool") => choice
                .and_then(|c| c.get("name"))
                .and_then(Value::as_str)
                .and_then(|name| tool_call_prefix(state, Some(name))),
            _ => None,
        },
        _ => None,
    };
    let single_call = choice
        .and_then(|c| c.get("disable_parallel_tool_use"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let json_schema = match output_schema(&fields).map_err(bad_request)? {
        None => None,
        Some(_) if citations.is_some() => {
            return Err(bad_request(
                "an output format cannot be combined with citations".into(),
            ));
        }
        // A reply held to a document cannot also be a tool call, nor carry
        // on text that may not be the start of one.
        Some(_) if tools.is_some() => {
            return Err(bad_request(
                "an output format cannot be combined with tools".into(),
            ));
        }
        Some(_) if continue_from.is_some() => {
            return Err(bad_request(
                "an output format cannot be combined with a continued assistant turn".into(),
            ));
        }
        Some((compiled, schema)) => {
            // The constraint decides what can be written; the schema in the
            // prompt is what tells the model what each field is for.
            let instruction = format!(
                "Respond with only a JSON document that matches this JSON Schema:\n{}",
                serde_json::to_string_pretty(&schema).unwrap_or_default()
            );
            append_system(&mut chat, &instruction);
            Some(compiled)
        }
    };
    Ok(Prompt {
        role,
        fields,
        chat,
        continue_from,
        tools,
        server,
        prefix,
        single_call,
        json_schema,
        container: container_json,
        client_tools,
        tool_names,
        citations,
        strict,
        resumed,
    })
}

/// Whether a conversation holds an advisor's results.
fn carries_advice(messages: &[AnthropicMessage]) -> bool {
    messages.iter().any(|m| {
        m.content.as_array().is_some_and(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("advisor_tool_result"))
        })
    })
}

/// The prompt for the reply's next turn: the first carries the continued
/// text and the forced opener, a later one neither.
async fn render_turn(
    state: &AppState,
    prompt: &Prompt,
    first: bool,
) -> Result<RenderedChat, Box<Response>> {
    let options = RenderOptions {
        reasoning: reasoning_for(state, prompt.role, &prompt.fields),
        continue_from: if first {
            prompt.continue_from.as_deref()
        } else {
            None
        },
        suffix: if first {
            prompt.prefix.as_deref()
        } else {
            None
        },
    };
    match render_chat(
        state,
        prompt.role,
        &prompt.chat,
        prompt.tools.as_ref(),
        options,
    )
    .await
    {
        Ok(rendered) => Ok(rendered),
        Err(rejection) => Err(Box::new(rewrap(*rejection).await)),
    }
}

pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Result<Json<PromptFields>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(req) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    let rendered = match prepare(&state, &headers, req, false).await {
        Ok(prompt) => render_turn(&state, &prompt, true).await,
        Err(rejection) => Err(rejection),
    };
    match rendered {
        Ok(rendered) => Json(json!({"input_tokens": rendered.tokens.len()})).into_response(),
        Err(rejection) => *rejection,
    }
}

/// A `/v1/messages` request ready to run.
struct Prepared {
    id: String,
    model: String,
    prompt: Prompt,
    first: RenderedChat,
    sampling: crate::engine::sampling::SamplingParams,
    max_tokens: usize,
    stop_sequences: Vec<String>,
    cache_prompt: bool,
    id_slot: Option<usize>,
    /// A streamed reply: a client tool's call is opened as soon as its name
    /// is written and its input sent as it is written.
    stream_input: bool,
}

async fn start(
    state: &AppState,
    headers: &HeaderMap,
    req: MessagesRequest,
) -> Result<Prepared, Box<Response>> {
    if let Some(rejection) = reject_unknown_slot(state, req.id_slot) {
        return Err(Box::new(rewrap(rejection).await));
    }
    let prompt = prepare(state, headers, req.prompt, true).await?;
    let first = render_turn(state, &prompt, true).await?;
    if let Some(rejection) = crate::http::reject_oversized_context(first.tokens.len()) {
        return Err(Box::new(rewrap(rejection).await));
    }
    let sampling = sampling_for(
        prompt.role,
        RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
        },
    );
    Ok(Prepared {
        id: next_message_id(),
        // The model the caller named, echoed back as Anthropic does, so a
        // client checking it against its own request finds what it sent.
        model: req.model.unwrap_or_else(|| state.model_label.clone()),
        prompt,
        first,
        sampling,
        max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        stop_sequences: req.stop_sequences,
        cache_prompt: req.cache_prompt,
        id_slot: req.id_slot,
        stream_input: req.stream,
    })
}

/// A piece of the reply as it is produced: reasoning and answer text, a
/// piece at a time, and whole blocks — tool calls and server tool results.
/// A client tool's call being streamed is opened, its input sent as JSON
/// text, and closed with the whole block.
enum Piece {
    Thinking(String),
    Text(String),
    Block(Value),
    CallOpen(Value),
    CallInput(String),
    CallClose(Value),
    /// A citation of the text just before it.
    Citation(Value),
}

/// The client tool call whose input is being streamed.
struct OpenCall {
    name: String,
    /// The opener its span starts with.
    opener: &'static str,
    stream: tool_calls::ArgumentStream,
}

/// How a reply ended.
struct Outcome {
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: Value,
    /// The last turn's figures, for `Server-Timing`.
    stats: Option<GenerateStats>,
}

/// The ids a reply hands out, and how many of each.
#[derive(Default)]
struct Calls {
    client: usize,
    server: usize,
    /// Client calls whose streamed block was opened and that never became
    /// a call: they have ids, and are not calls the reply made.
    abandoned: usize,
}

impl Calls {
    /// Whether the reply made a client call.
    fn made(&self) -> bool {
        self.client > self.abandoned
    }
}

/// What one turn of a reply has produced so far.
#[derive(Default)]
struct Turn {
    /// The turn's prose, as the model wrote it.
    text: String,
    /// The server tools it called, to run once it is over.
    server_calls: Vec<(server_tools::Kind, String, tool_calls::ParsedToolCall)>,
    /// Its calls the model is answered about instead of the client: one
    /// naming no tool, or (with the reason) one whose input a `strict`
    /// tool's schema refuses.
    unknown: Vec<(tool_calls::ParsedToolCall, Option<String>)>,
    /// The client call whose input is being streamed.
    open_call: Option<OpenCall>,
}

/// Generated text, split into prose and finished calls, sent to `sink`:
/// a client tool's call as `tool_use`, a server tool's announced and kept
/// in the turn's `server_calls` to run once the turn is over.
fn absorb(
    split: tool_calls::Split,
    written: &str,
    prepared: &Prepared,
    calls: &mut Calls,
    turn: &mut Turn,
    sink: &mut (dyn FnMut(Piece) + Send),
) {
    let mut split = split;
    if let Some(open) = turn.open_call.take() {
        let finished = split
            .calls
            .first()
            .is_some_and(|call| call.name == open.name)
            .then(|| split.calls.remove(0));
        let block = match finished {
            Some(call) => {
                let span = written.strip_prefix(open.opener).unwrap_or(written);
                let (rest, agreed) = open.stream.finish(span, &call.arguments);
                if !agreed {
                    eprintln!(
                        "orangu-server: [messages] the streamed input of a `{}` call differs from its whole parse",
                        call.name
                    );
                }
                sink(Piece::CallInput(rest));
                tool_use_block(&prepared.id, calls.client, &call)
            }
            None => {
                calls.abandoned += 1;
                sink(Piece::CallInput(open.stream.abandon()));
                json!({"type": "tool_use", "name": open.name, "input": {},
                       "id": format!("toolu_{}_{}", &prepared.id["msg_".len()..], calls.client)})
            }
        };
        sink(Piece::CallClose(block));
        calls.client += 1;
    }
    if !split.content.is_empty() {
        turn.text.push_str(&split.content);
        sink(Piece::Text(split.content));
    }
    let stem = &prepared.id["msg_".len()..];
    for call in split.calls {
        match prepared.prompt.server.kind(&call.name) {
            Some(kind) => {
                let tag = match kind {
                    server_tools::Kind::Mcp { .. } => "mcptoolu",
                    _ => "srvtoolu",
                };
                let id = format!("{tag}_{stem}_{}", calls.server);
                calls.server += 1;
                sink(Piece::Block(
                    prepared.prompt.server.use_block(&kind, &id, &call),
                ));
                turn.server_calls.push((kind, id, call));
            }
            // A call naming no tool at all is answered here, not handed to
            // a client that never declared it.
            None if !prepared.prompt.client_tools.contains(&call.name) => {
                turn.unknown.push((call, None))
            }
            None if let Some(why) = strict_refusal(prepared, &call) => {
                turn.unknown.push((call, Some(why)))
            }
            None => {
                if prepared.prompt.single_call && calls.client > 0 {
                    continue;
                }
                sink(Piece::Block(tool_use_block(
                    &prepared.id,
                    calls.client,
                    &call,
                )));
                calls.client += 1;
            }
        }
    }
}

/// Why a call to a `strict` tool is refused — its input is not what the
/// tool's `input_schema` describes — as the error the model is answered
/// with; `None` for a call to any other tool, or one that matches.
fn strict_refusal(prepared: &Prepared, call: &tool_calls::ParsedToolCall) -> Option<String> {
    let schema = prepared.prompt.strict.get(&call.name)?;
    let why = match serde_json::from_str::<Value>(&call.arguments) {
        Ok(input) => crate::engine::json_schema::validate(schema, &input).err()?,
        Err(_) => "the input is not a JSON object".to_string(),
    };
    Some(format!(
        "Error: the input of this `{}` call does not match the tool's schema ({why}). Call it \
         again with input that matches.",
        call.name
    ))
}

/// Runs Python code that may call the client's tools — `None` when the
/// request has none for it to call, and the code runs as usual. `Ok` is
/// the code's result block and what the model is told; `Err` the calls it
/// made, which pause it until the client's results come.
async fn start_code(
    prepared: &Prepared,
    id: &str,
    call: &tool_calls::ParsedToolCall,
) -> Option<Result<(Value, String), Vec<Value>>> {
    let names = prepared.prompt.server.code_callable();
    if names.is_empty() {
        return None;
    }
    let container = prepared.prompt.server.container()?;
    let code = serde_json::from_str::<Value>(&call.arguments)
        .ok()
        .and_then(|input| input["code"].as_str().map(str::to_string))
        .unwrap_or_default();
    let (mut run, step) = match programmatic::Run::start(container, &code, &names).await {
        Ok(started) => started,
        Err(why) => {
            return Some(Ok(programmatic::result_block(
                id,
                &programmatic::Step::Failed("unavailable", why),
            )));
        }
    };
    Some(match step {
        programmatic::Step::Calls(calls) => {
            let blocks = run.blocks(id, calls);
            programmatic::pause(id, run);
            Err(blocks)
        }
        step => Ok(programmatic::result_block(id, &step)),
    })
}

/// A client tool's call as it is written, in a streamed reply: once a call
/// names a declared client tool, the text before it is sent, the call's
/// block opened, and its input sent as it grows.
fn stream_call(
    prepared: &Prepared,
    pending: &mut String,
    calls: &mut Calls,
    turn: &mut Turn,
    sink: &mut (dyn FnMut(Piece) + Send),
) {
    if turn.open_call.is_none() {
        let Some((at, opener)) = tool_calls::first_opener(pending) else {
            return;
        };
        let Some(name) = tool_calls::call_name(opener, &pending[at + opener.len()..]) else {
            return;
        };
        if !prepared.prompt.client_tools.contains(&name)
            || prepared.prompt.strict.contains_key(&name)
            || prepared.prompt.server.kind(&name).is_some()
            || (prepared.prompt.single_call && calls.client > 0)
        {
            return;
        }
        let before: String = pending.drain(..at).collect();
        absorb(
            tool_calls::split(&before),
            &before,
            prepared,
            calls,
            turn,
            sink,
        );
        sink(Piece::CallOpen(json!({
            "type": "tool_use",
            "id": format!("toolu_{}_{}", &prepared.id["msg_".len()..], calls.client),
            "name": name,
            "input": {},
        })));
        turn.open_call = Some(OpenCall {
            name,
            opener,
            stream: tool_calls::ArgumentStream::new(opener),
        });
    }
    if let Some(open) = &mut turn.open_call
        && let Some(span) = pending.strip_prefix(open.opener)
    {
        let input = open.stream.advance(span);
        if !input.is_empty() {
            sink(Piece::CallInput(input));
        }
    }
}

/// [`drive_turns`], with the passage numbers of a reply that cites taken
/// out of its text as citations.
async fn drive(
    state: &AppState,
    prepared: &mut Prepared,
    sink: &mut (dyn FnMut(Piece) + Send),
) -> Result<Outcome, Box<Response>> {
    let Some(sources) = prepared.prompt.citations.clone() else {
        return drive_turns(state, prepared, sink).await;
    };
    let mut citer = citations::Citer::new(sources);
    let outcome = {
        let mut cited = |piece: Piece| match piece {
            Piece::Text(text) => citer.push(&text, sink),
            other => {
                citer.flush(sink);
                sink(other);
            }
        };
        drive_turns(state, prepared, &mut cited).await
    };
    citer.flush(sink);
    outcome
}

/// Runs a reply to its end, turn by turn: each turn generates; a turn that
/// called server tools has them run, their results added to the
/// conversation, and the next turn generated from there. Everything the
/// reply holds goes to `sink` as it is produced.
async fn drive_turns(
    state: &AppState,
    prepared: &mut Prepared,
    sink: &mut (dyn FnMut(Piece) + Send),
) -> Result<Outcome, Box<Response>> {
    let mut stops = StopMatcher::new(&prepared.stop_sequences);
    let mut calls = Calls::default();
    let (mut input, mut cached, mut output) = (0usize, 0usize, 0usize);
    let mut rendered = std::mem::replace(
        &mut prepared.first,
        RenderedChat {
            tokens: Vec::new(),
            cache_point: None,
        },
    );
    match prepared.prompt.resumed.take() {
        Some(Resumed::Paused(blocks)) => {
            for block in blocks {
                sink(Piece::Block(block));
            }
            return Ok(Outcome {
                stop_reason: "tool_use",
                stop_sequence: None,
                usage: json!({"input_tokens": 0, "output_tokens": 0,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
                stats: None,
            });
        }
        Some(Resumed::Finished(block)) => sink(Piece::Block(block)),
        None => {}
    }
    let mut turn = 0usize;
    loop {
        let first = turn == 0;
        let prompt_len = rendered.tokens.len();
        let mut rx = state
            .engine
            .generate(GenerateRequest {
                prompt_tokens: rendered.tokens,
                sampling: prepared.sampling.clone(),
                // The reply's cap, not each turn's.
                max_tokens: prepared.max_tokens.saturating_sub(output).max(1),
                stop_token_ids: state.engine.tokenizer.stop_token_ids(),
                json_output: false,
                json_schema: prepared.prompt.json_schema.clone(),
                cache_prompt: prepared.cache_prompt,
                id_slot: prepared.id_slot,
                timings_per_token: false,
                role: Some(prepared.prompt.role),
                cache_point: rendered.cache_point,
                background: false,
                reasoning_budget: thinking_budget(&prepared.prompt.fields),
            })
            .await;
        // A forced call's opener is read as if the model had written it.
        let mut pending = if first {
            prepared.prompt.prefix.clone().unwrap_or_default()
        } else {
            String::new()
        };
        let mut this_turn = Turn::default();
        let mut hit = None;
        let mut finish = FinishReason::Stop;
        let mut stats = None;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Reasoning(text) => sink(Piece::Thinking(text)),
                StreamEvent::Token(text) => {
                    let (safe, stop) = stops.push(&text);
                    pending.push_str(&safe);
                    if stop.is_some() {
                        hit = stop;
                        break;
                    }
                    if tool_calls::may_be_partial(&pending) {
                        if prepared.stream_input {
                            stream_call(prepared, &mut pending, &mut calls, &mut this_turn, sink);
                        }
                        continue;
                    }
                    let written = std::mem::take(&mut pending);
                    let split = tool_calls::split(&written);
                    absorb(split, &written, prepared, &mut calls, &mut this_turn, sink);
                    // A reply limited to one call ends at it.
                    if prepared.prompt.single_call && calls.client > 0 {
                        break;
                    }
                }
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                StreamEvent::Done {
                    finish_reason,
                    stats: done,
                } => {
                    finish = finish_reason;
                    stats = Some(done);
                    break;
                }
                StreamEvent::Overloaded => {
                    return Err(Box::new(rewrap(crate::http::overloaded_response()).await));
                }
                StreamEvent::Error(err) => {
                    return Err(Box::new(error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        err,
                    )));
                }
            }
        }
        // Dropping the receiver is what stops a generation cut short by a
        // stop sequence or a single call.
        drop(rx);
        pending.push_str(&stops.finish());
        // A call the model ended the reply on is a call, closer or not.
        if hit.is_none() && finish == FinishReason::Stop {
            pending = tool_calls::close_trailing(&pending);
        }
        if !(prepared.prompt.single_call && calls.client > 0) || this_turn.open_call.is_some() {
            let split = tool_calls::split(&pending);
            absorb(split, &pending, prepared, &mut calls, &mut this_turn, sink);
        }
        let Turn {
            text: turn_text,
            server_calls,
            unknown,
            ..
        } = this_turn;
        match &stats {
            Some(stats) => {
                input += stats.prompt_tokens.saturating_sub(stats.cached_tokens);
                cached += stats.cached_tokens;
                output += stats.generated_tokens;
            }
            // Cut short: the engine's own count never arrived, so the prompt
            // is what was sent.
            None => input += prompt_len,
        }
        let ended = |prepared: &Prepared, stop_reason: &'static str, stop_sequence| {
            let mut usage = json!({
                "input_tokens": input,
                "output_tokens": output,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cached,
            });
            if let Some(server) = server_tools::usage(&prepared.prompt.server) {
                usage["server_tool_use"] = server;
            }
            Outcome {
                stop_reason,
                stop_sequence,
                usage,
                stats: stats.clone(),
            }
        };
        if hit.is_some() {
            return Ok(ended(prepared, "stop_sequence", hit));
        }
        if server_calls.is_empty() && (unknown.is_empty() || calls.made()) {
            return Ok(ended(
                prepared,
                stop_reason(finish, calls.made(), None),
                None,
            ));
        }
        let mut results = Vec::new();
        for (kind, id, call) in &server_calls {
            if *kind == server_tools::Kind::Advisor {
                let advice = match prepared.prompt.server.take_advisor() {
                    Some(declared) => {
                        let (model, max_tokens) = (declared.model.clone(), declared.max_tokens);
                        let transcript =
                            advisor::transcript_of_chat(&prepared.prompt.chat, &turn_text);
                        advisor::consult(state, &model, transcript, max_tokens).await
                    }
                    None => Err("max_uses_exceeded"),
                };
                let (block, text) = advisor::result_block(id, &advice);
                sink(Piece::Block(block));
                results.push((id.clone(), call.name.clone(), text));
                continue;
            }
            if *kind == server_tools::Kind::Python
                && let Some(started) = start_code(prepared, id, call).await
            {
                match started {
                    Err(calls) => {
                        for block in calls {
                            sink(Piece::Block(block));
                        }
                        return Ok(ended(prepared, "tool_use", None));
                    }
                    Ok((block, text)) => {
                        sink(Piece::Block(block));
                        results.push((id.clone(), call.name.clone(), text));
                        continue;
                    }
                }
            }
            let ran = prepared.prompt.server.run(kind, id, call).await;
            sink(Piece::Block(ran.block));
            results.push((id.clone(), call.name.clone(), ran.text));
        }
        let stem = &prepared.id["msg_".len()..];
        let unknown: Vec<(String, tool_calls::ParsedToolCall)> = unknown
            .into_iter()
            .map(|(call, refusal)| {
                let id = format!("call_{stem}_{}", calls.server);
                calls.server += 1;
                let names = &prepared.prompt.tool_names;
                let text = if let Some(refusal) = refusal {
                    refusal
                } else if names.is_empty() {
                    format!(
                        "Error: there is no tool named `{}`, and no tools.",
                        call.name
                    )
                } else {
                    format!(
                        "Error: there is no tool named `{}`. The tools are: {}.",
                        call.name,
                        names.join(", ")
                    )
                };
                results.push((id.clone(), call.name.clone(), text));
                (id, call)
            })
            .collect();
        if calls.made() {
            return Ok(ended(prepared, "tool_use", None));
        }
        if finish == FinishReason::Length || output >= prepared.max_tokens {
            return Ok(ended(prepared, "max_tokens", None));
        }
        turn += 1;
        if turn >= MAX_SERVER_TURNS {
            return Ok(ended(prepared, "pause_turn", None));
        }
        // The turn, with its calls and their results, joins the
        // conversation the next turn is generated from.
        let mut content = if first {
            prepared.prompt.continue_from.take().unwrap_or_default()
        } else {
            String::new()
        };
        content.push_str(&turn_text);
        let mut assistant = ChatMessage::text("assistant", &content);
        assistant.tool_calls = Some(Value::Array(
            server_calls
                .iter()
                .map(|(_, id, call)| (id, call))
                .chain(unknown.iter().map(|(id, call)| (id, call)))
                .map(|(id, call)| {
                    json!({"id": id, "type": "function",
                           "function": {"name": call.name, "arguments": call.arguments}})
                })
                .collect(),
        ));
        prepared.prompt.chat.push(assistant);
        for (id, name, text) in results {
            prepared.prompt.chat.push(ChatMessage {
                role: "tool".to_string(),
                content: text,
                tool_call_id: Some(id),
                name: Some(name),
                ..Default::default()
            });
        }
        rendered = render_turn(state, &prepared.prompt, false).await?;
        if let Some(rejection) = crate::http::reject_oversized_context(rendered.tokens.len()) {
            return Err(Box::new(rewrap(rejection).await));
        }
    }
}

/// A whole reply's pieces as its content blocks: neighbouring reasoning and
/// text pieces each joined into one block.
fn assemble(pieces: Vec<Piece>, tokenizer: &crate::engine::tokenizer::Tokenizer) -> Vec<Value> {
    let mut blocks = Vec::new();
    let mut thinking = String::new();
    let mut text = String::new();
    let flush = |blocks: &mut Vec<Value>, thinking: &mut String, text: &mut String| {
        if !thinking.is_empty() {
            blocks.push(json!({
                "type": "thinking",
                "thinking": tokenizer.clean_up_tokenization_spaces(&std::mem::take(thinking)),
                "signature": "",
            }));
        }
        if !text.is_empty() {
            blocks.push(json!({
                "type": "text",
                "text": tokenizer.clean_up_tokenization_spaces(&std::mem::take(text)),
            }));
        }
    };
    for piece in pieces {
        match piece {
            Piece::Thinking(t) => {
                if !text.is_empty() {
                    flush(&mut blocks, &mut String::new(), &mut text);
                }
                thinking.push_str(&t);
            }
            Piece::Text(t) => {
                if !thinking.is_empty() {
                    flush(&mut blocks, &mut thinking, &mut String::new());
                }
                text.push_str(&t);
            }
            Piece::Block(block) | Piece::CallClose(block) => {
                flush(&mut blocks, &mut thinking, &mut text);
                blocks.push(block);
            }
            Piece::CallOpen(_) | Piece::CallInput(_) => {}
            Piece::Citation(citation) => {
                if !thinking.is_empty() {
                    flush(&mut blocks, &mut thinking, &mut String::new());
                }
                flush(&mut blocks, &mut String::new(), &mut text);
                let last_is_text = blocks
                    .last()
                    .is_some_and(|b| b.get("type").and_then(Value::as_str) == Some("text"));
                if !last_is_text {
                    blocks.push(json!({"type": "text", "text": ""}));
                }
                if let Some(last) = blocks.last_mut() {
                    match last.get_mut("citations").and_then(Value::as_array_mut) {
                        Some(list) => list.push(citation),
                        None => last["citations"] = json!([citation]),
                    }
                }
            }
        }
    }
    flush(&mut blocks, &mut thinking, &mut text);
    blocks
}

fn message_json(
    id: &str,
    model: &str,
    content: Vec<Value>,
    outcome: &Outcome,
    container: Option<&Value>,
) -> Value {
    let mut message = json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": outcome.stop_reason,
        "stop_sequence": outcome.stop_sequence,
        "usage": outcome.usage,
    });
    if let Some(container) = container {
        message["container"] = container.clone();
    }
    message
}

/// A non-streamed reply, whole: the `message` object, and how it ended.
async fn collect(state: &AppState, prepared: Prepared) -> Result<(Value, Outcome), Box<Response>> {
    collect_watched(state, prepared, &mut |_| {}).await
}

/// [`collect`], with each piece of the reply shown to `watch` as it is
/// produced.
async fn collect_watched(
    state: &AppState,
    mut prepared: Prepared,
    watch: &mut (dyn FnMut(&Piece) + Send),
) -> Result<(Value, Outcome), Box<Response>> {
    let mut pieces = Vec::new();
    let outcome = drive(state, &mut prepared, &mut |piece| {
        watch(&piece);
        pieces.push(piece);
    })
    .await?;
    let content = assemble(pieces, &state.engine.tokenizer);
    Ok((
        message_json(
            &prepared.id,
            &prepared.model,
            content,
            &outcome,
            prepared.prompt.container.as_ref(),
        ),
        outcome,
    ))
}

/// One non-streamed `/v1/messages` request, start to finish — what a
/// batch runs for each of its requests.
pub(crate) async fn run_one(
    state: &AppState,
    headers: &HeaderMap,
    req: MessagesRequest,
) -> Result<Value, Box<Response>> {
    let prepared = start(state, headers, req).await?;
    collect(state, prepared).await.map(|(message, _)| message)
}

/// [`run_one`], with each piece of the reply shown to `watch` as it is
/// produced.
async fn run_one_watched(
    state: &AppState,
    headers: &HeaderMap,
    req: MessagesRequest,
    watch: &mut (dyn FnMut(&Piece) + Send),
) -> Result<Value, Box<Response>> {
    let prepared = start(state, headers, req).await?;
    collect_watched(state, prepared, watch)
        .await
        .map(|(message, _)| message)
}

/// What the streaming side receives from the task driving the reply.
enum Driven {
    Piece(Piece),
    Done(Outcome),
    Failed(Value),
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(req) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    let stream = req.stream;
    let prepared = match start(&state, &headers, req).await {
        Ok(prepared) => prepared,
        Err(rejection) => return *rejection,
    };
    let id = prepared.id.clone();
    if !stream {
        return match collect(&state, prepared).await {
            Ok((message, outcome)) => with_request_id(
                with_server_timing(Json(message).into_response(), outcome.stats.as_ref()),
                &id,
            ),
            Err(rejection) => *rejection,
        };
    }

    let model = prepared.model.clone();
    let prompt_tokens = prepared.first.tokens.len();
    let container = prepared.prompt.container.clone();
    // The reply is driven on a task of its own and handed over a piece at a
    // time; a client that goes away drops the receiver, which ends the task
    // at its next piece — and the generation with it.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let driver_state = state.clone();
    tokio::spawn(async move {
        let mut prepared = prepared;
        let pieces = tx.clone();
        let outcome = drive(&driver_state, &mut prepared, &mut |piece| {
            let _ = pieces.send(Driven::Piece(piece));
        })
        .await;
        let _ = tx.send(match outcome {
            Ok(outcome) => Driven::Done(outcome),
            Err(response) => Driven::Failed(error_of(*response).await),
        });
    });
    let request_id = id.clone();
    let stream = async_stream::stream! {
        let sse = |(name, data): (&'static str, Value)| {
            Ok::<_, std::convert::Infallible>(Event::default().event(name).data(data.to_string()))
        };
        yield sse(("message_start", json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {
                    "input_tokens": prompt_tokens, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                },
            },
        })));
        yield sse(("ping", json!({"type": "ping"})));
        let mut blocks = Blocks::new();
        while let Some(driven) = rx.recv().await {
            match driven {
                Driven::Piece(Piece::Thinking(text)) => {
                    for e in blocks.reasoning(&text) {
                        yield sse(e);
                    }
                }
                Driven::Piece(Piece::Text(text)) => {
                    for e in blocks.text(&text) {
                        yield sse(e);
                    }
                }
                Driven::Piece(Piece::Block(block)) => {
                    for e in blocks.block(block) {
                        yield sse(e);
                    }
                }
                Driven::Piece(Piece::CallOpen(block)) => {
                    for e in blocks.start(Open::Call, block) {
                        yield sse(e);
                    }
                }
                Driven::Piece(Piece::CallInput(json)) => {
                    yield sse(blocks.delta(json!({"type": "input_json_delta", "partial_json": json})));
                }
                Driven::Piece(Piece::Citation(citation)) => {
                    for e in blocks.citation(citation) {
                        yield sse(e);
                    }
                }
                Driven::Piece(Piece::CallClose(_)) => {
                    for e in blocks.close() {
                        yield sse(e);
                    }
                }
                Driven::Done(outcome) => {
                    for e in blocks.close() {
                        yield sse(e);
                    }
                    let mut delta = json!({
                        "stop_reason": outcome.stop_reason,
                        "stop_sequence": outcome.stop_sequence,
                    });
                    if let Some(container) = &container {
                        delta["container"] = container.clone();
                    }
                    yield sse(("message_delta", json!({
                        "type": "message_delta",
                        "delta": delta,
                        "usage": outcome.usage,
                    })));
                    yield sse(("message_stop", json!({"type": "message_stop"})));
                    break;
                }
                Driven::Failed(error) => {
                    yield sse(("error", error));
                    break;
                }
            }
        }
    };
    with_request_id(
        axum::response::sse::Sse::new(stream).into_response(),
        &request_id,
    )
}

/// An error `Response` as Anthropic's error object — the body this module's
/// errors already carry, or one made from a plain-text body.
pub(crate) async fn error_of(response: Response) -> Value {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap_or_default();
    serde_json::from_slice(&bytes).unwrap_or_else(|_| {
        json!({"type": "error", "error": {
            "type": "api_error",
            "message": format!("{status}: {}", String::from_utf8_lossy(&bytes)),
        }})
    })
}

#[derive(PartialEq)]
enum Open {
    Nothing,
    Thinking,
    Text,
    Call,
}

/// The content blocks of a streamed message, as the SSE events that write
/// them. Thinking and text are blocks opened as their first piece arrives
/// and closed when anything else does; a client tool call being streamed is
/// opened by name and closed when its input is whole; any other call or a
/// server tool result is a whole block of its own.
struct Blocks {
    open: Open,
    index: usize,
    /// The open text block has been cited: more text starts a new one.
    cited: bool,
}

type SseEvent = (&'static str, Value);

impl Blocks {
    fn new() -> Self {
        Self {
            open: Open::Nothing,
            index: 0,
            cited: false,
        }
    }

    fn close(&mut self) -> Vec<SseEvent> {
        if self.open == Open::Nothing {
            return Vec::new();
        }
        self.open = Open::Nothing;
        self.index += 1;
        vec![self.stop(self.index - 1)]
    }

    fn stop(&self, index: usize) -> SseEvent {
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        )
    }

    fn start(&mut self, kind: Open, block: Value) -> Vec<SseEvent> {
        let mut events = self.close();
        self.cited = false;
        events.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        ));
        self.open = kind;
        events
    }

    fn delta(&self, delta: Value) -> SseEvent {
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
    }

    fn reasoning(&mut self, text: &str) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if self.open != Open::Thinking {
            events = self.start(
                Open::Thinking,
                json!({"type": "thinking", "thinking": "", "signature": ""}),
            );
        }
        events.push(self.delta(json!({"type": "thinking_delta", "thinking": text})));
        events
    }

    fn text(&mut self, text: &str) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if text.is_empty() {
            return events;
        }
        if self.open != Open::Text || self.cited {
            events = self.start(Open::Text, json!({"type": "text", "text": ""}));
        }
        events.push(self.delta(json!({"type": "text_delta", "text": text})));
        events
    }

    /// A citation of the open text block, or of an empty one when none is.
    fn citation(&mut self, citation: Value) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if self.open != Open::Text {
            events = self.start(
                Open::Text,
                json!({"type": "text", "text": "", "citations": []}),
            );
        }
        self.cited = true;
        events.push(self.delta(json!({"type": "citations_delta", "citation": citation})));
        events
    }

    /// A whole block: a call not streamed as it was written sends its input
    /// as one `input_json_delta`; a result arrives whole in its start event.
    fn block(&mut self, mut block: Value) -> Vec<SseEvent> {
        let mut events = self.close();
        let is_call = matches!(
            block.get("type").and_then(Value::as_str),
            Some("tool_use" | "server_tool_use" | "mcp_tool_use")
        );
        let input = is_call.then(|| std::mem::replace(&mut block["input"], json!({})));
        events.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        ));
        if let Some(input) = input {
            events.push(
                self.delta(json!({"type": "input_json_delta", "partial_json": input.to_string()})),
            );
        }
        events.push(self.stop(self.index));
        self.index += 1;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages(value: Value) -> Vec<AnthropicMessage> {
        serde_json::from_value(value).unwrap()
    }

    /// The calls code made, and their results, never reach the model's
    /// conversation; the code and its output do.
    #[test]
    fn calls_from_code_are_left_out_of_the_conversation() {
        let mut m = messages(json!([
            {"role": "user", "content": "Sum the sales."},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "code_execution",
                 "input": {"code": "print(await q({}))"}},
                {"type": "tool_use", "id": "toolu_9", "name": "q", "input": {},
                 "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_9", "content": "42"}]},
        ]));
        strip_programmatic(&mut m);
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].content.as_array().unwrap().len(), 1);
        assert_eq!(m[1].content[0]["type"], "server_tool_use");
    }

    #[test]
    fn system_and_string_content_become_plain_messages() {
        let (chat, _) = to_chat_messages(
            Some(&json!([{"type": "text", "text": "Be terse."}])),
            &messages(json!([{"role": "user", "content": "Hi"}])),
        );
        assert_eq!(chat.len(), 2);
        assert_eq!(
            (chat[0].role.as_str(), chat[0].content.as_str()),
            ("system", "Be terse.")
        );
        assert_eq!(
            (chat[1].role.as_str(), chat[1].content.as_str()),
            ("user", "Hi")
        );
    }

    #[test]
    fn a_tool_round_trip_becomes_tool_calls_and_a_named_tool_message() {
        let (chat, _) = to_chat_messages(
            None,
            &messages(json!([
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hidden", "signature": ""},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "4C"}]},
                    {"type": "text", "text": "And tomorrow?"},
                ]},
            ])),
        );
        let roles: Vec<&str> = chat.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "user"]);
        assert_eq!(chat[1].content, "Checking.");
        let call = &chat[1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call["id"], "toolu_1");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], r#"{"city":"Oslo"}"#);
        assert_eq!(chat[2].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(chat[2].name.as_deref(), Some("get_weather"));
        assert_eq!(chat[2].content, "4C");
        assert_eq!(chat[3].content, "And tomorrow?");
    }

    #[test]
    fn an_image_block_becomes_a_data_url() {
        let (chat, _) = to_chat_messages(
            None,
            &messages(json!([{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "text", "text": "What is this?"},
            ]}])),
        );
        assert_eq!(chat[0].images, ["data:image/png;base64,AAAA"]);
        assert_eq!(chat[0].content, "What is this?");
        assert_eq!(chat[0].image_at, [0]);
    }

    #[test]
    fn an_image_keeps_its_place_between_texts() {
        let (chat, _) = to_chat_messages(
            None,
            &messages(json!([{"role": "user", "content": [
                {"type": "text", "text": "Before"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                {"type": "text", "text": "After"},
            ]}])),
        );
        assert_eq!(chat[0].content, "Before\nAfter");
        assert_eq!(chat[0].image_at, ["Before".len()]);
        assert_eq!(chat[0].images, ["https://example.com/a.png"]);
    }

    #[test]
    fn a_tool_results_pictures_follow_its_text() {
        let (chat, _) = to_chat_messages(
            None,
            &messages(json!([
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                        {"type": "text", "text": "Here it is."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    ]},
                ]},
            ])),
        );
        let result = chat.iter().find(|m| m.role == "tool").unwrap();
        assert_eq!(result.content, "Here it is.");
        assert_eq!(result.images, ["data:image/png;base64,AAAA"]);
        assert_eq!(result.image_at, ["Here it is.".len()]);
    }

    #[test]
    fn tools_are_reshaped_and_server_tools_dropped() {
        let tools = vec![
            json!({"name": "get_weather", "description": "Weather", "input_schema": {"type": "object"}}),
            json!({"type": "web_search_20250305", "name": "web_search"}),
        ];
        let shaped = to_openai_tools(Some(&tools), None).unwrap();
        assert_eq!(shaped.as_array().unwrap().len(), 1);
        assert_eq!(shaped[0]["type"], "function");
        assert_eq!(shaped[0]["function"]["name"], "get_weather");
        assert_eq!(shaped[0]["function"]["parameters"]["type"], "object");
        assert!(to_openai_tools(Some(&tools), Some(&json!({"type": "none"}))).is_none());
    }

    #[test]
    fn a_stop_sequence_split_across_tokens_is_caught_and_not_emitted() {
        let mut stops = StopMatcher::new(&["\nEND".to_string()]);
        let mut out = String::new();
        let mut hit = None;
        for piece in ["Hello", " world\n", "EN", "D trailing"] {
            let (emit, stop) = stops.push(piece);
            out.push_str(&emit);
            if stop.is_some() {
                hit = stop;
                break;
            }
        }
        assert_eq!(out, "Hello world");
        assert_eq!(hit.as_deref(), Some("\nEND"));
    }

    #[test]
    fn held_text_that_never_becomes_a_stop_sequence_is_returned() {
        let mut stops = StopMatcher::new(&["STOP".to_string()]);
        let (emit, stop) = stops.push("go ST");
        assert_eq!((emit.as_str(), stop), ("go ", None));
        assert_eq!(stops.finish(), "ST");
        let mut none = StopMatcher::new(&[]);
        assert_eq!(none.push("anything").0, "anything");
    }

    #[test]
    fn streamed_blocks_open_close_and_number_in_order() {
        let mut blocks = Blocks::new();
        let mut events = blocks.reasoning("hmm");
        events.extend(blocks.reasoning(" ok"));
        events.extend(blocks.text("Answer"));
        events.extend(blocks.close());
        let names: Vec<&str> = events.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
        assert_eq!(events[0].1["content_block"]["type"], "thinking");
        assert_eq!(events[3].1["index"], 0);
        assert_eq!(events[4].1["index"], 1);
        assert_eq!(events[5].1["delta"]["text"], "Answer");
        assert!(blocks.close().is_empty());
    }

    #[test]
    fn stop_reasons_follow_anthropic_names() {
        assert_eq!(stop_reason(FinishReason::Stop, false, None), "end_turn");
        assert_eq!(stop_reason(FinishReason::Length, false, None), "max_tokens");
        assert_eq!(stop_reason(FinishReason::Stop, true, None), "tool_use");
        assert_eq!(
            stop_reason(FinishReason::Stop, true, Some("x")),
            "stop_sequence"
        );
    }

    #[test]
    fn a_rejection_is_rewrapped_with_its_status_and_retry_after() {
        let response = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(rewrap(crate::http::overloaded_response()));
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[axum::http::header::RETRY_AFTER], "1");
    }

    #[test]
    fn a_conversation_ending_on_assistant_text_is_continued() {
        let (chat, continue_from) = to_chat_messages(
            None,
            &messages(json!([
                {"role": "user", "content": "Name a colour."},
                {"role": "assistant", "content": "The colour is "},
            ])),
        );
        assert_eq!(chat.len(), 1);
        assert_eq!(continue_from.as_deref(), Some("The colour is"));
        // An assistant turn that called a tool is finished, not continued.
        let (chat, continue_from) = to_chat_messages(
            None,
            &messages(json!([
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t", "name": "w", "input": {}},
                ]},
            ])),
        );
        assert_eq!(chat.len(), 2);
        assert!(continue_from.is_none());
    }

    #[test]
    fn a_pdf_document_is_read_as_its_text_under_its_title() {
        use base64::Engine as _;
        let pdf = orangu::pdf::encode_text("t", "Quarterly revenue rose.").unwrap();
        let block = json!({"type": "document", "title": "Report", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64::engine::general_purpose::STANDARD.encode(pdf),
        }});
        let text = document_text(&block).unwrap();
        assert!(text.starts_with("Report\n\n"), "{text}");
        assert!(text.contains("Quarterly revenue rose."), "{text}");
        let plain = json!({"type": "document", "source": {"type": "text", "data": "hello"}});
        assert_eq!(document_text(&plain).as_deref(), Some("hello"));
    }

    #[test]
    fn anthropic_client_tools_get_their_schemas() {
        let tools = vec![
            json!({"type": "bash_20250124", "name": "bash"}),
            json!({"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}),
            json!({"type": "memory_20250818", "name": "memory"}),
            json!({"type": "code_execution_20250825", "name": "code_execution"}),
        ];
        let shaped = to_openai_tools(Some(&tools), None).unwrap();
        let names: Vec<&str> = shaped
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["bash", "str_replace_based_edit_tool", "memory"]);
        assert_eq!(
            shaped[0]["function"]["parameters"]["properties"]["command"]["type"],
            "string"
        );
    }
}

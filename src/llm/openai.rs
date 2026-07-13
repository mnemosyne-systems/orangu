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

use super::{
    ChatMessage, LlmResponse, StreamMetrics, StreamPromptProgress, ToolCall, ToolCallFunction,
    ToolDefinition,
};
use crate::config::LlmConfiguration;
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Clone)]
pub struct OpenAiClient {
    http_client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    /// Response-token cap sent as `max_tokens`; `None` leaves the server's
    /// default in place.
    max_tokens: Option<u32>,
    /// `orangu-server` `id_slot` to pin this request to, from
    /// [`crate::llm::SlotRegistry`]; `None` lets the server assign any idle
    /// slot (today's behavior).
    id_slot: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings_per_token: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_progress: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id_slot: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiStreamResponse {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    timings: Option<OpenAiTimings>,
    #[serde(default)]
    prompt_progress: Option<OpenAiPromptProgress>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    tool_type: Option<String>,
    #[serde(default)]
    function: Option<OpenAiToolCallFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiTimings {
    #[serde(default)]
    prompt_per_second: Option<f64>,
    #[serde(default)]
    predicted_per_second: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenAiPromptProgress {
    total: i32,
    cache: i32,
    processed: i32,
    time_ms: i64,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: Option<String>,
    tool_type: Option<String>,
    name: String,
    arguments: String,
}

impl OpenAiClient {
    pub fn from_profile(profile: &LlmConfiguration) -> Result<Self> {
        Ok(Self {
            http_client: Client::builder()
                .timeout(Duration::from_secs(profile.request_timeout_seconds))
                .build()?,
            endpoint: normalize_openai_endpoint(&profile.endpoint),
            model: profile.model.clone(),
            api_key: profile.api_key.clone(),
            // Normal chat/tool responses are capped by the configured
            // `code_max_tokens` (0 = no cap).
            max_tokens: (profile.code_max_tokens > 0).then_some(profile.code_max_tokens),
            id_slot: None,
        })
    }

    /// Cap every chat response from this client at `max_tokens` tokens,
    /// replacing any cap taken from the profile. `0` means no cap — a zero is
    /// never sent to the server, which would request an empty response.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = (max_tokens > 0).then_some(max_tokens);
        self
    }

    /// Pin every request from this client to a specific `orangu-server`
    /// `id_slot` (see [`crate::llm::SlotRegistry`]).
    pub fn with_id_slot(mut self, id_slot: Option<u32>) -> Self {
        self.id_slot = id_slot;
        self
    }

    pub async fn chat<F, G>(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        mut on_text_delta: F,
        mut on_stream_metrics: G,
    ) -> Result<LlmResponse>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
    {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let request = OpenAiChatRequest {
            model: &self.model,
            messages,
            stream: true,
            tools: if tools.is_empty() { None } else { Some(tools) },
            max_tokens: self.max_tokens,
            // orangu-server is the only backend and always speaks the
            // OpenAI-compatible API, so its native streaming metrics and
            // slot pinning are always requested.
            timings_per_token: Some(true),
            return_progress: Some(true),
            id_slot: self.id_slot,
        };

        let mut builder = self.http_client.post(&url).json(&request);
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }

        let mut resp = builder.send().await.map_err(|e| {
            anyhow!(
                "failed to send chat request to {} using model {}: {}",
                self.endpoint,
                self.model,
                e
            )
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "chat completion failed (status {}): {}",
                status,
                body
            ));
        }

        let mut pending_lines = Vec::new();
        let mut line_buffer = String::new();
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        while let Some(chunk) = resp.chunk().await? {
            line_buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_index) = line_buffer.find('\n') {
                let mut line = line_buffer.drain(..=newline_index).collect::<String>();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }

                if line.is_empty() {
                    if process_stream_event(
                        &pending_lines,
                        &mut content,
                        &mut tool_calls,
                        &mut on_text_delta,
                        &mut on_stream_metrics,
                    )? {
                        return finalize_stream_response(content, tool_calls);
                    }
                    pending_lines.clear();
                    continue;
                }

                if let Some(payload) = line.strip_prefix("data:") {
                    pending_lines.push(payload.trim_start().to_string());
                }
            }
        }

        if !pending_lines.is_empty() {
            let _ = process_stream_event(
                &pending_lines,
                &mut content,
                &mut tool_calls,
                &mut on_text_delta,
                &mut on_stream_metrics,
            )?;
        }

        finalize_stream_response(content, tool_calls)
    }
}

fn normalize_openai_endpoint(endpoint: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    endpoint.strip_suffix("/v1").unwrap_or(endpoint).to_string()
}

pub fn normalized_openai_endpoint(endpoint: &str) -> String {
    normalize_openai_endpoint(endpoint)
}

fn process_stream_event<F, G>(
    pending_lines: &[String],
    content: &mut String,
    tool_calls: &mut Vec<PartialToolCall>,
    on_text_delta: &mut F,
    on_stream_metrics: &mut G,
) -> Result<bool>
where
    F: FnMut(&str),
    G: FnMut(StreamMetrics),
{
    if pending_lines.is_empty() {
        return Ok(false);
    }

    let payload = pending_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(true);
    }

    let response: OpenAiStreamResponse = serde_json::from_str(&payload)?;
    let metrics = stream_metrics_from_response(&response);
    if !metrics.is_empty() {
        on_stream_metrics(metrics);
    }
    for choice in response.choices {
        if let Some(text) = choice.delta.content {
            content.push_str(&text);
            on_text_delta(&text);
        }
        if let Some(deltas) = choice.delta.tool_calls {
            apply_tool_call_deltas(tool_calls, deltas);
        }
        let _ = choice.finish_reason;
    }

    Ok(false)
}

fn stream_metrics_from_response(response: &OpenAiStreamResponse) -> StreamMetrics {
    let mut metrics = StreamMetrics::default();
    if let Some(timings) = &response.timings {
        metrics.prompt_per_second = timings.prompt_per_second;
        metrics.predicted_per_second = timings.predicted_per_second;
    }
    if let Some(progress) = &response.prompt_progress {
        metrics.prompt_progress = Some(StreamPromptProgress {
            total: progress.total,
            cache: progress.cache,
            processed: progress.processed,
            time_ms: progress.time_ms,
        });
    }
    metrics
}

fn apply_tool_call_deltas(tool_calls: &mut Vec<PartialToolCall>, deltas: Vec<OpenAiToolCallDelta>) {
    for delta in deltas {
        if tool_calls.len() <= delta.index {
            tool_calls.resize_with(delta.index + 1, PartialToolCall::default);
        }
        let entry = &mut tool_calls[delta.index];
        if let Some(id) = delta.id {
            entry.id = Some(id);
        }
        if let Some(tool_type) = delta.tool_type {
            entry.tool_type = Some(tool_type);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                entry.name.push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                entry.arguments.push_str(&arguments);
            }
        }
    }
}

fn finalize_stream_response(
    content: String,
    tool_calls: Vec<PartialToolCall>,
) -> Result<LlmResponse> {
    if !tool_calls.is_empty() {
        let mapped_calls = tool_calls
            .into_iter()
            .map(|tc| {
                let arguments = parse_tool_arguments(&tc.arguments)?;
                let name = tc.name;
                Ok(ToolCall {
                    id: tc.id.unwrap_or_else(|| format!("call_{name}")),
                    tool_type: tc.tool_type.unwrap_or_else(|| "function".to_string()),
                    function: ToolCallFunction { name, arguments },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LlmResponse::ToolCalls(mapped_calls))
    } else {
        Ok(LlmResponse::Text(content))
    }
}

fn parse_tool_arguments(arguments: &str) -> Result<HashMap<String, serde_json::Value>> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(HashMap::new());
    }
    Ok(serde_json::from_str(trimmed)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_appends_text_deltas() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut rendered = String::new();
        let mut metrics = Vec::new();

        let done = process_stream_event(
            &[r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string()],
            &mut content,
            &mut tool_calls,
            &mut |text| rendered.push_str(text),
            &mut |update| metrics.push(update),
        )
        .expect("stream event");

        assert!(!done);
        assert_eq!(content, "Hello");
        assert_eq!(rendered, "Hello");
        assert!(tool_calls.is_empty());
        assert!(metrics.is_empty());
    }

    #[test]
    fn stream_event_assembles_tool_call_deltas() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = Vec::new();

        process_stream_event(
            &[r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{"}}]}}]}"#.to_string()],
            &mut content,
            &mut tool_calls,
            &mut |_| {},
            &mut |update| metrics.push(update),
        )
        .expect("first tool delta");
        process_stream_event(
            &[r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"path\":\"README.md\"}"}}]}}]}"#.to_string()],
            &mut content,
            &mut tool_calls,
            &mut |_| {},
            &mut |update| metrics.push(update),
        )
        .expect("second tool delta");

        let response = finalize_stream_response(content, tool_calls).expect("finalize");
        match response {
            LlmResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "read_file");
                assert_eq!(
                    calls[0].function.arguments.get("path"),
                    Some(&serde_json::Value::String("README.md".to_string()))
                );
            }
            _ => panic!("expected tool calls"),
        }
        assert!(metrics.is_empty());
    }

    #[test]
    fn fenced_json_tool_call_text_remains_text() {
        let content =
            "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}\n```";

        let response = finalize_stream_response(content.to_string(), Vec::new()).expect("finalize");

        assert!(matches!(response, LlmResponse::Text(text) if text == content));
    }

    #[test]
    fn stream_event_extracts_llama_cpp_metrics() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = Vec::new();

        process_stream_event(
            &[r#"{"choices":[{"delta":{"role":"assistant","content":null}}],"timings":{"prompt_per_second":32.3,"predicted_per_second":52.9},"prompt_progress":{"total":100,"cache":20,"processed":60,"time_ms":2000}}"#.to_string()],
            &mut content,
            &mut tool_calls,
            &mut |_| {},
            &mut |update| metrics.push(update),
        )
        .expect("metrics event");

        assert_eq!(
            metrics,
            vec![StreamMetrics {
                prompt_progress: Some(StreamPromptProgress {
                    total: 100,
                    cache: 20,
                    processed: 60,
                    time_ms: 2000,
                }),
                prompt_per_second: Some(32.3),
                predicted_per_second: Some(52.9),
            }]
        );
    }

    #[test]
    fn max_tokens_follows_the_profile_and_zero_means_no_cap() {
        let mut profile = LlmConfiguration {
            endpoint: "http://localhost:8100".to_string(),
            model: "model".to_string(),
            role: "all".to_string(),
            api_key: None,
            request_timeout_seconds: 5,
            max_tool_rounds: 10,
            review_max_tokens: 512,
            review_confidence_threshold: 80,
            code_max_tokens: 0,
            system_prompt: "".to_string(),
            model_verbosity: None,
        };

        // `code_max_tokens = 0` leaves chat responses uncapped; a non-zero
        // value caps them.
        let client = OpenAiClient::from_profile(&profile).expect("client");
        assert_eq!(client.max_tokens, None);
        profile.code_max_tokens = 256;
        let capped = OpenAiClient::from_profile(&profile).expect("client");
        assert_eq!(capped.max_tokens, Some(256));

        // `with_max_tokens` replaces the profile cap; zero clears it rather
        // than requesting an empty response.
        assert_eq!(capped.clone().with_max_tokens(512).max_tokens, Some(512));
        assert_eq!(capped.with_max_tokens(0).max_tokens, None);
    }

    #[test]
    fn chat_request_serializes_a_set_max_tokens() {
        let request = OpenAiChatRequest {
            model: "model",
            messages: &[],
            stream: true,
            tools: None,
            max_tokens: Some(512),
            timings_per_token: None,
            return_progress: None,
            id_slot: None,
        };

        let encoded = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(
            encoded.get("max_tokens"),
            Some(&serde_json::Value::Number(512.into()))
        );
    }

    #[test]
    fn id_slot_serializes_only_when_set() {
        let unset = OpenAiChatRequest {
            model: "model",
            messages: &[],
            stream: true,
            tools: None,
            max_tokens: None,
            timings_per_token: None,
            return_progress: None,
            id_slot: None,
        };
        let encoded = serde_json::to_value(&unset).expect("serialize request");
        assert_eq!(encoded.get("id_slot"), None);

        let pinned = OpenAiChatRequest {
            id_slot: Some(2),
            ..unset
        };
        let encoded = serde_json::to_value(&pinned).expect("serialize request");
        assert_eq!(
            encoded.get("id_slot"),
            Some(&serde_json::Value::Number(2.into()))
        );
    }

    #[test]
    fn with_id_slot_records_the_requested_slot() {
        let profile = LlmConfiguration {
            endpoint: "http://localhost:8100".to_string(),
            model: "model".to_string(),
            role: "all".to_string(),
            api_key: None,
            request_timeout_seconds: 5,
            max_tool_rounds: 10,
            review_max_tokens: 512,
            review_confidence_threshold: 80,
            code_max_tokens: 0,
            system_prompt: "".to_string(),
            model_verbosity: None,
        };
        let client = OpenAiClient::from_profile(&profile)
            .expect("client")
            .with_id_slot(Some(3));
        assert_eq!(client.id_slot, Some(3));
    }

    #[test]
    fn llama_cpp_requests_native_stream_metrics() {
        let request = OpenAiChatRequest {
            model: "model",
            messages: &[],
            stream: true,
            tools: None,
            max_tokens: None,
            timings_per_token: Some(true),
            return_progress: Some(true),
            id_slot: None,
        };

        let encoded = serde_json::to_value(&request).expect("serialize request");
        // An unset response cap is omitted from the request entirely.
        assert_eq!(encoded.get("max_tokens"), None);
        assert_eq!(
            encoded.get("timings_per_token"),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            encoded.get("return_progress"),
            Some(&serde_json::Value::Bool(true))
        );
    }
}

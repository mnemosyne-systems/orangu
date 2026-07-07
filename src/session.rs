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

use crate::{
    config::LlmConfiguration,
    llm::{ChatMessage, LlmResponse, OpenAiClient, StreamMetrics},
    tools::ToolExecutor,
};
use anyhow::{Result, anyhow};

pub struct ChatSession {
    messages: Vec<ChatMessage>,
    /// Cached LLM client, reused across prompts so the underlying HTTP
    /// connection pool survives between requests. Rebuilt only when the
    /// profile fields that shape the client change.
    client: Option<(ClientKey, OpenAiClient)>,

    pub model_verbosity_override: Option<String>,
}

/// The subset of [`LlmConfiguration`] that the [`OpenAiClient`] is built from.
/// Two profiles producing the same key yield an interchangeable client.
#[derive(PartialEq, Eq)]
struct ClientKey {
    provider: String,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    request_timeout_seconds: u64,
    code_max_tokens: u32,
}

impl ClientKey {
    fn from_profile(profile: &LlmConfiguration) -> Self {
        Self {
            provider: profile.provider.clone(),
            endpoint: profile.endpoint.clone(),
            model: profile.model.clone(),
            api_key: profile.api_key.clone(),
            request_timeout_seconds: profile.request_timeout_seconds,
            code_max_tokens: profile.code_max_tokens,
        }
    }
}

impl ChatSession {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
            client: None,

            model_verbosity_override: None,
        }
    }

    pub fn set_system_prompt(&mut self, prompt: &str) {
        let has_user_turns = self.messages.iter().any(|m| m.role == "user");
        if has_user_turns {
            self.messages
                .push(ChatMessage::user(&format!("[System Update]\n{}", prompt)));
        } else {
            match self.messages.first_mut() {
                Some(message) if message.role == "system" => {
                    message.content = prompt.to_string();
                }
                _ => self.messages.insert(0, ChatMessage::system(prompt)),
            }
        }
    }

    pub fn clear(&mut self, system_prompt: &str) {
        self.messages.clear();
        self.messages.push(ChatMessage::system(system_prompt));
    }

    pub fn push_user(&mut self, content: &str) {
        self.messages.push(ChatMessage::user(content));
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn restore(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages;
    }

    pub fn checkpoint(&self) -> usize {
        self.messages.len()
    }

    pub fn rollback(&mut self, checkpoint: usize) {
        self.messages.truncate(checkpoint);
    }

    pub fn compact_transcript(&mut self) {
        let mut user_turns = 0;
        for msg in self.messages.iter_mut().rev() {
            if msg.role == "user" {
                user_turns += 1;
            } else if msg.role == "tool" && user_turns > 3 && msg.content.len() > 500 {
                msg.content = "[Tool output evicted to save tokens]".to_string();
            }
        }
    }

    /// One-shot prompt with no tool definitions and a capped response length:
    /// a single chat round. The model cannot start tool-call rounds and cannot
    /// generate unbounded output — for requests whose prompt is self-contained
    /// (the content to work on is inline), such as `/auto_review`.
    ///
    /// `on_text_delta` and `on_stream_metrics` are forwarded to the streaming
    /// client, which fires them as the response arrives — they drive the live
    /// status display. The complete text is also returned at the end.
    /// A `max_response_tokens` of `0` disables the cap.
    pub async fn prompt_without_tools<F, G>(
        &mut self,
        user_input: &str,
        profile: &LlmConfiguration,
        max_response_tokens: u32,
        mut on_text_delta: F,
        mut on_stream_metrics: G,
    ) -> Result<String>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
    {
        // Built per call rather than cached: the cached client is keyed for
        // the tool-enabled flow and carries that flow's response cap, not
        // this request's.
        let client = OpenAiClient::from_profile(profile)?.with_max_tokens(max_response_tokens);
        let checkpoint = self.checkpoint();
        self.messages.push(ChatMessage::user(user_input));
        match client
            .chat(
                &self.messages,
                &[],
                &mut on_text_delta,
                &mut on_stream_metrics,
            )
            .await
        {
            Ok(LlmResponse::Text(text)) => {
                self.messages.push(ChatMessage::assistant(&text));
                Ok(text)
            }
            Ok(LlmResponse::ToolCalls(_)) => {
                self.rollback(checkpoint);
                Err(anyhow!("the model attempted a tool call without tools"))
            }
            Err(err) => {
                self.rollback(checkpoint);
                Err(err)
            }
        }
    }

    pub async fn prompt<F, G, H>(
        &mut self,
        user_input: &str,
        profile: &LlmConfiguration,
        tools: &ToolExecutor,
        mut on_text_delta: F,
        mut on_stream_metrics: G,
        mut on_tool_running: H,
    ) -> Result<String>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
        H: FnMut(bool),
    {
        self.compact_transcript();

        let key = ClientKey::from_profile(profile);
        if self
            .client
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            self.client = Some((key, OpenAiClient::from_profile(profile)?));
        }
        // Cheap clone: shares the underlying reqwest connection pool.
        let client = self
            .client
            .as_ref()
            .expect("client populated above")
            .1
            .clone();
        let tool_definitions = tools.definitions();
        let checkpoint = self.checkpoint();
        self.messages.push(ChatMessage::user(user_input));

        for _ in 0..profile.max_tool_rounds {
            match client
                .chat(
                    &self.messages,
                    &tool_definitions,
                    &mut on_text_delta,
                    &mut on_stream_metrics,
                )
                .await
            {
                Ok(response) => match response {
                    LlmResponse::Text(text) => {
                        self.messages.push(ChatMessage::assistant(&text));
                        return Ok(text);
                    }
                    LlmResponse::ToolCalls(tool_calls) => {
                        self.messages
                            .push(ChatMessage::assistant_tool_calls(tool_calls.clone()));

                        on_tool_running(true);
                        for tool_call in tool_calls {
                            let rendered = match tools
                                .execute(
                                    &tool_call.function.name,
                                    &tool_call.function.arguments.into_iter().collect(),
                                )
                                .await
                            {
                                Ok(result) => result,
                                Err(err) => format!("error: {err:#}"),
                            };

                            self.messages
                                .push(ChatMessage::tool_result(&tool_call.id, &rendered));
                        }
                        on_tool_running(false);
                    }
                },
                Err(err) => {
                    self.rollback(checkpoint);
                    return Err(err);
                }
            }
        }

        self.rollback(checkpoint);
        Err(anyhow!(
            "model exceeded the configured max_tool_rounds ({})",
            profile.max_tool_rounds
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::ChatSession;
    use crate::config::LlmConfiguration;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn test_profile(endpoint: &str) -> LlmConfiguration {
        LlmConfiguration {
            provider: "llama.cpp".to_string(),
            endpoint: endpoint.to_string(),
            model: "test-model".to_string(),
            role: "all".to_string(),
            api_key: None,
            request_timeout_seconds: 5,
            max_tool_rounds: 10,
            review_max_tokens: 512,
            review_confidence_threshold: 80,
            code_max_tokens: 0,
            system_prompt: "".to_string(),
            model_verbosity: None,
        }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Serve exactly one HTTP request on `listener`, answering with `sse_body`
    /// as a chat-completion event stream, and return the request body that the
    /// client sent.
    fn serve_one_chat_response(
        listener: TcpListener,
        sse_body: &'static str,
    ) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            let header_end = loop {
                let read = stream.read(&mut buffer).expect("read request");
                request.extend_from_slice(&buffer[..read]);
                if let Some(position) = find_subsequence(&request, b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(0);
            while request.len() < header_end + content_length {
                let read = stream.read(&mut buffer).expect("read body");
                request.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                sse_body.len(),
                sse_body,
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            String::from_utf8_lossy(&request[header_end..]).to_string()
        })
    }

    #[tokio::test]
    async fn prompt_without_tools_returns_text_and_caps_the_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"content\":\"VERDICT: APPROVE\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        let mut deltas = Vec::new();
        let response = session
            .prompt_without_tools(
                "review this",
                &test_profile(&endpoint),
                512,
                |delta| deltas.push(delta.to_string()),
                |_| {},
            )
            .await
            .expect("text response");
        assert_eq!(response, "VERDICT: APPROVE");
        // The streamed deltas reach the caller's callback as they arrive.
        assert_eq!(deltas.concat(), "VERDICT: APPROVE");

        // The exchange is recorded like a normal prompt.
        let roles: Vec<&str> = session
            .messages()
            .iter()
            .map(|message| message.role.as_str())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant"]);

        // The request carries the response cap and no tool definitions.
        let body = server.join().expect("server thread");
        assert!(body.contains("\"max_tokens\":512"), "request body: {body}");
        assert!(!body.contains("\"tools\""), "request body: {body}");
    }

    #[tokio::test]
    async fn prompt_without_tools_with_zero_cap_omits_max_tokens() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        session
            .prompt_without_tools("review this", &test_profile(&endpoint), 0, |_| {}, |_| {})
            .await
            .expect("text response");

        // A zero cap means no cap: the request carries no max_tokens at all.
        let body = server.join().expect("server thread");
        assert!(!body.contains("max_tokens"), "request body: {body}");
    }

    #[tokio::test]
    async fn prompt_without_tools_rolls_back_when_the_model_calls_a_tool() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call0\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        let error = session
            .prompt_without_tools("review this", &test_profile(&endpoint), 512, |_| {}, |_| {})
            .await
            .expect_err("tool calls are rejected");
        assert!(
            error.to_string().contains("tool call"),
            "unexpected error: {error:#}"
        );
        // The failed exchange is rolled back; only the system prompt remains.
        assert_eq!(session.messages().len(), 1);
        let _ = server.join();
    }

    #[tokio::test]
    async fn prompt_without_tools_rolls_back_on_request_errors() {
        // The server accepts the connection and closes it without sending a
        // response, which fails the request deterministically on every
        // platform (no reliance on a freed port staying unbound).
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept connection");
            drop(stream);
        });

        let mut session = ChatSession::new("system");
        let result = session
            .prompt_without_tools("review this", &test_profile(&endpoint), 512, |_| {}, |_| {})
            .await;
        assert!(result.is_err());
        // The failed exchange is rolled back; only the system prompt remains.
        assert_eq!(session.messages().len(), 1);
        let _ = server.join();
    }
}

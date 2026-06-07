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
}

impl ClientKey {
    fn from_profile(profile: &LlmConfiguration) -> Self {
        Self {
            provider: profile.provider.clone(),
            endpoint: profile.endpoint.clone(),
            model: profile.model.clone(),
            api_key: profile.api_key.clone(),
            request_timeout_seconds: profile.request_timeout_seconds,
        }
    }
}

impl ChatSession {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
            client: None,
        }
    }

    pub fn set_system_prompt(&mut self, prompt: &str) {
        match self.messages.first_mut() {
            Some(message) if message.role == "system" => {
                message.content = prompt.to_string();
            }
            _ => self.messages.insert(0, ChatMessage::system(prompt)),
        }
    }

    pub fn clear(&mut self, system_prompt: &str) {
        self.messages.clear();
        self.messages.push(ChatMessage::system(system_prompt));
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

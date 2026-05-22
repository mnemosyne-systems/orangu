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
}

impl ChatSession {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
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

    pub async fn prompt<F, G>(
        &mut self,
        user_input: &str,
        profile: &LlmConfiguration,
        tools: &ToolExecutor,
        mut on_text_delta: F,
        mut on_stream_metrics: G,
    ) -> Result<String>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
    {
        let client = OpenAiClient::from_profile(profile)?;
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

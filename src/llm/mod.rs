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

pub mod openai;
pub mod slots;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub use openai::{OpenAiClient, normalized_openai_endpoint};
pub use slots::{SaveRestoreOutcome, SlotRegistry};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamMetrics {
    pub prompt_progress: Option<StreamPromptProgress>,
    pub prompt_per_second: Option<f64>,
    pub predicted_per_second: Option<f64>,
}

impl StreamMetrics {
    pub fn is_empty(&self) -> bool {
        self.prompt_progress.is_none()
            && self.prompt_per_second.is_none()
            && self.predicted_per_second.is_none()
    }

    pub fn merge(&mut self, update: Self) {
        if let Some(progress) = update.prompt_progress {
            self.prompt_progress = Some(progress);
        }
        if let Some(prompt_per_second) = update.prompt_per_second {
            self.prompt_per_second = Some(prompt_per_second);
        }
        if let Some(predicted_per_second) = update.predicted_per_second {
            self.predicted_per_second = Some(predicted_per_second);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StreamPromptProgress {
    pub total: i32,
    pub cache: i32,
    pub processed: i32,
    pub time_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(
        serialize_with = "serialize_tool_call_arguments",
        deserialize_with = "deserialize_tool_call_arguments"
    )]
    pub arguments: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub enum LlmResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

fn serialize_tool_call_arguments<S>(
    arguments: &HashMap<String, Value>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let object: Map<String, Value> = arguments.clone().into_iter().collect();
    serializer.serialize_str(
        &serde_json::to_string(&Value::Object(object)).map_err(serde::ser::Error::custom)?,
    )
}

fn deserialize_tool_call_arguments<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(text) => serde_json::from_str(&text).map_err(serde::de::Error::custom),
        Value::Object(object) => Ok(object.into_iter().collect()),
        Value::Null => Ok(HashMap::new()),
        other => Err(serde::de::Error::custom(format!(
            "unsupported tool call arguments payload: {other}"
        ))),
    }
}

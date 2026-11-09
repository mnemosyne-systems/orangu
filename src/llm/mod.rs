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

pub mod coordinator;
pub mod openai;
pub mod slots;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub use coordinator::{probe_coordinator, probe_coordinator_roles};
pub use openai::{OpenAiClient, normalized_openai_endpoint};
pub use slots::{SaveRestoreOutcome, SlotRegistry, probe_context_size};

/// What every surface says when a turn ended with
/// [`StreamMetrics::truncated`].
///
/// One wording in one place because two clients say it: `-p` prints it under
/// the timings and the terminal interface pushes it into the transcript after
/// the answer. A cut-off answer is the same event on both, and a user who
/// learns to recognise the sentence on one should not meet a different one on
/// the other.
pub const TRUNCATED_NOTICE: &str = "answer cut off at the server's response-length cap \
     — an unfinished tool call was not run; raise [orangu].code_max_tokens";

/// The phrase `orangu-server` puts in the `400` it answers a request whose
/// context is longer than its device can hold with, and that a client matches
/// to recognise one.
///
/// A shared constant rather than each side spelling its own, because this is a
/// contract between them: the server promises the phrase appears, the client
/// promises to look for exactly it, and a rewording that broke the pairing
/// would break it *silently* — the review would go on reporting oversized
/// files as failed requests, which is what it did before this existed.
///
/// Matched on the message rather than the status code because a `400` on its
/// own says nothing about *what* to shorten: a client that shrank its prompt
/// on any bad request would also shrink it on a malformed one.
pub const CONTEXT_TOO_LONG_MARKER: &str = "tokens of KV cache";

/// Whether a failed request failed because its context was too long for the
/// server's device — the one failure a caller can do something about by
/// sending less. See [`CONTEXT_TOO_LONG_MARKER`].
pub fn is_context_too_long(error: &str) -> bool {
    error.contains(CONTEXT_TOO_LONG_MARKER)
}

/// The phrase `orangu-server` refuses an `id_slot` it has no slot for with.
///
/// Paired with the client the same way [`CONTEXT_TOO_LONG_MARKER`] is, and for
/// a failure that is likewise the caller's to fix: it asked for a slot that
/// does not exist and should ask for one that does.
pub const SLOT_OUT_OF_RANGE_MARKER: &str = "out of range (server has";

/// Whether a failed request failed because the `id_slot` it pinned does not
/// exist on the server that answered.
///
/// This is *not* only a client bug. Behind an `orangu-coordinator` one
/// endpoint fronts several `orangu-server` processes with different slot
/// counts — an `embeddings` role defaults to more slots than a chat one — and
/// `GET /props` carries no model, so it can only ever report whichever backend
/// happened to be active when it was asked. A slot count learned from one
/// backend and used against another is wrong through no fault of either.
pub fn is_slot_out_of_range(error: &str) -> bool {
    error.contains(SLOT_OUT_OF_RANGE_MARKER)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamMetrics {
    pub prompt_progress: Option<StreamPromptProgress>,
    pub prompt_per_second: Option<f64>,
    pub predicted_per_second: Option<f64>,
    /// The server stopped because the answer reached its response-length cap
    /// (`finish_reason: "length"`), not because the model was done.
    ///
    /// Not a rate, and here anyway: it is the one thing about how a response
    /// ended that a caller cannot see from the response itself. A cut-off
    /// answer looks exactly like a finished one, and when what it cut off was
    /// a tool call, the call never closes, so it is not recognised as a call,
    /// nothing runs, and the half-written file arrives as prose. That is a
    /// silent wrong answer; every surface that reports a turn should be able
    /// to say it happened.
    pub truncated: bool,
    /// What one finished response cost, from its final chunk: OpenAI's
    /// `usage`, or llama.cpp's `timings` counts when there is no `usage`.
    pub usage: Option<TokenUsage>,
}

/// Tokens one response read and wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt: u64,
    pub completion: u64,
}

impl StreamMetrics {
    pub fn is_empty(&self) -> bool {
        self.prompt_progress.is_none()
            && self.usage.is_none()
            && self.prompt_per_second.is_none()
            && self.predicted_per_second.is_none()
            && !self.truncated
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
        if let Some(usage) = update.usage {
            self.usage = Some(usage);
        }
        // Sticky: one truncated response stays truncated however many further
        // updates arrive after it.
        self.truncated |= update.truncated;
    }
}

/// Prompt tokens this process's chat requests carried, and how many of them
/// the server answered from its prefix cache — summed over every response's
/// final chunk. A conversation whose prompt stops extending the previous one
/// (a reordered tool list, a rewritten history) shows as a falling share.
static PROMPT_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHED_PROMPT_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(prompt tokens, of which cached)` since the process started.
pub fn prompt_cache_totals() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        PROMPT_TOKENS.load(Relaxed),
        CACHED_PROMPT_TOKENS.load(Relaxed),
    )
}

/// Adds one finished response's prompt to [`prompt_cache_totals`].
pub(crate) fn note_prompt_cache(progress: &StreamPromptProgress) {
    use std::sync::atomic::Ordering::Relaxed;
    PROMPT_TOKENS.fetch_add(progress.total.max(0) as u64, Relaxed);
    CACHED_PROMPT_TOKENS.fetch_add(
        progress.cache.clamp(0, progress.total.max(0)) as u64,
        Relaxed,
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairing that makes an oversized prompt recoverable: the server's
    /// own refusal has to be one the client recognises. Two separate spellings
    /// would compile, pass their own tests, and quietly turn every oversized
    /// file in a review back into a failed request.
    #[test]
    fn the_servers_context_refusal_is_the_one_clients_match() {
        let refusal = format!(
            "prompt (41843 tokens) plus one generated token needs 41844 {CONTEXT_TOO_LONG_MARKER}, \
             more than the 16384 this server has room for on its device"
        );
        assert!(is_context_too_long(&refusal));
        // Wrapped in the client's own error context, which is how a caller
        // actually meets it.
        assert!(is_context_too_long(&format!(
            "chat completion failed (status 400 Bad Request): {refusal}"
        )));
        // And not every failure: a caller that shortened its prompt on any
        // error would shorten it on a broken server too.
        assert!(!is_context_too_long(
            "chat completion failed (status 500 Internal Server Error): model not loaded"
        ));
    }

    /// The other half of the same contract, for the other refusal a caller can
    /// act on. Both spellings live in `orangu-server`'s two slot-checking
    /// paths, so this is what keeps the three in step.
    #[test]
    fn the_servers_slot_refusal_is_the_one_clients_match() {
        let refusal = format!("id_slot 1 {SLOT_OUT_OF_RANGE_MARKER} 1 slots)");
        assert!(is_slot_out_of_range(&refusal));
        assert!(is_slot_out_of_range(&format!(
            "chat completion failed (status 400 Bad Request): {refusal}"
        )));
        assert!(!is_slot_out_of_range(
            "chat completion failed (status 400 Bad Request): prompt is too long"
        ));
    }
}

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

//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`,
//! `/v1/completions`, `/v1/embeddings`.

use axum::{
    Json,
    extract::{FromRequest, Request, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::AppState;
use super::chat_choice::{CallRules, ChoiceStream, split_whole};
use super::native::finish_reason_str;
use crate::engine::chat_template::{ChatMessage, Reasoning};
use crate::engine::generate::{GenerateRequest, GenerateStats, StreamEvent};
use crate::engine::image::ImageEvent;
use crate::engine::loader::PoolingType;
use crate::engine::sampling::{SamplingParams, TokenLogprob};
use crate::engine::text_stop::StopRules;
use crate::engine::tool_calls;
use base64::Engine as _;
use tokio::sync::mpsc::UnboundedReceiver;

/// How long an answer may get when the request does not say — the
/// `max_tokens` default for `/v1/chat/completions`, clamped afterwards to
/// what is left of the model's context window.
///
/// It has to be *this* large because the endpoint's real workload is an agent
/// writing a file: the file's content travels inside a `<|tool_call>` span,
/// and a span cut off by the cap never closes, so `engine::tool_calls`
/// correctly refuses to read it as a call and it lands in the answer as prose
/// instead. The tool is never run and nothing reaches disk. 512 — the default
/// this replaces — is about forty lines of code, and a request that hit it
/// wrote no file at all; 4096 was still short of a page of HTML with its game
/// loop in it. This is a long source file with room for the model to think
/// first.
///
/// A cap has to exist all the same, and the reason is **time**, not memory:
/// `LayerCache::new_strided` only *reserves* its buffers, so an oversized
/// capacity costs address space rather than RSS (see its own comment). What an
/// unbounded default would cost is a runaway generation holding the slot every
/// other request queues behind until the whole context window is full — on the
/// order of an hour at a local decode rate, against a couple of minutes here.
///
/// Reaching this is no longer silent either way: the response carries
/// `finish_reason: "length"`, and the client says so — see
/// `llm::StreamMetrics::truncated`.
pub(crate) const DEFAULT_MAX_TOKENS: usize = 8192;

/// OpenAI's `usage` object.
pub(crate) fn usage_json(stats: &GenerateStats) -> serde_json::Value {
    let mut usage = json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        // OpenAI's own field for the part of the prompt served from cache;
        // `prompt_progress` below carries the same number in llama.cpp's shape.
        "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
    });
    if stats.dropped_tokens > 0 {
        usage["context_dropped_tokens"] = stats.dropped_tokens.into();
    }
    usage
}

/// llama.cpp's `timings` object, field for field, so a client (this
/// project's own `llm::openai`, `orangu-bench`, or anything written against
/// llama-server) reads prompt- and decode-rate the same way from either
/// server. Rates come from [`GenerateStats`], which is also what the
/// per-request console log prints — one source of truth for "how fast was
/// that", rather than a wall-clock guess at the far end of an HTTP stream.
///
/// On a top-level node of a `[workers]` tree it also carries `workers`: the
/// request's time at each node, `[{"node": ..., "ms": ...}]` in pipeline
/// order — this node's own layers, then each worker's round trip, its
/// subtree included. Absent without a tree.
pub(crate) fn timings_json(stats: &GenerateStats) -> serde_json::Value {
    let prompt_ms = stats.prompt_time.as_secs_f64() * 1000.0;
    let predicted_ms = stats.generate_time.as_secs_f64() * 1000.0;
    let mut timings = json!({
        "prompt_n": stats.prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": prompt_ms / (stats.prompt_tokens.max(1) as f64),
        "prompt_per_second": stats.prompt_tokens_per_second(),
        "predicted_n": stats.generated_tokens,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": predicted_ms / (stats.generated_tokens.max(1) as f64),
        "predicted_per_second": stats.generate_tokens_per_second(),
    });
    if stats.dropped_tokens > 0 {
        timings["context_dropped_tokens"] = stats.dropped_tokens.into();
    }
    if !stats.workers.is_empty() {
        timings["workers"] = stats
            .workers
            .iter()
            .map(|(node, t)| json!({"node": node, "ms": t.as_secs_f64() * 1000.0}))
            .collect();
    }
    timings
}

/// The `Server-Timing` header of a whole (not streamed) answer: `prompt`
/// and `generate`, and in a `[workers]` tree one `node<i>` per part of the
/// pipeline, named in its `desc`. A streamed answer's headers leave before
/// any of this is known; its last event carries the same in `timings`.
pub(crate) fn server_timing(stats: &GenerateStats) -> axum::http::HeaderValue {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let mut parts = vec![
        format!("prompt;dur={:.1}", ms(stats.prompt_time)),
        format!("generate;dur={:.1}", ms(stats.generate_time)),
    ];
    for (i, (node, t)) in stats.workers.iter().enumerate() {
        // A quoted string may hold any visible character but `"` and `\`,
        // which a node's name never needs.
        let desc: String = node
            .chars()
            .filter(|c| c.is_ascii_graphic() && *c != '"' && *c != '\\')
            .collect();
        parts.push(format!("node{i};desc=\"{desc}\";dur={:.1}", ms(*t)));
    }
    axum::http::HeaderValue::from_str(&parts.join(", "))
        .unwrap_or_else(|_| axum::http::HeaderValue::from_static("total;dur=0"))
}

/// `response` with a [`server_timing`] header, when there are stats for it.
pub(crate) fn with_server_timing(
    mut response: axum::response::Response,
    stats: Option<&GenerateStats>,
) -> axum::response::Response {
    if let Some(stats) = stats {
        response
            .headers_mut()
            .insert("server-timing", server_timing(stats));
    }
    response
}

/// llama.cpp's `prompt_progress` object. llama-server emits it repeatedly
/// *during* prefill; this server has no mid-prefill progress event, so it is
/// sent once, with the finished request's totals — enough to tell a cache hit
/// from real prompt processing, which is what it is read for.
pub(crate) fn prompt_progress_json(stats: &GenerateStats) -> serde_json::Value {
    json!({
        "total": stats.prompt_tokens,
        "cache": stats.cached_tokens,
        "processed": stats.prefilled_tokens(),
        "time_ms": stats.prompt_time.as_millis() as i64,
    })
}

/// The served model in both OpenAI's shape (`object`, `created`,
/// `owned_by`) and Anthropic's (`type`, `display_name`, `created_at`); each
/// SDK reads its own fields and ignores the other's.
fn model_json(state: &AppState) -> serde_json::Value {
    let created = unix_now();
    // What one request may hold: the model's context, or less when the
    // device has room for less.
    let context = crate::engine::generate::kv_device_token_ceiling()
        .map_or(state.engine.model.config().n_ctx_train, |ceiling| {
            ceiling.min(state.engine.model.config().n_ctx_train)
        });
    let created_at = chrono::DateTime::from_timestamp(created as i64, 0)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    json!({
        "id": state.model_label,
        "object": "model",
        "created": created,
        "owned_by": "orangu-server",
        "type": "model",
        "display_name": state.model_label,
        "created_at": created_at,
        "max_input_tokens": context,
        "max_tokens": context,
        "capabilities": {
            "batch": {"supported": true},
            "citations": {"supported": true},
            "image_input": {"supported": state.engine.vision.is_some()},
            "pdf_input": {"supported": true},
            "structured_outputs": {"supported": true},
            "thinking": {"supported": reasons_apart(&state.engine.tokenizer)},
        },
    })
}

/// Whether this vocabulary marks reasoning off from the answer, so a
/// reasoning model's thinking reaches a client apart from its reply.
fn reasons_apart(tokenizer: &crate::engine::tokenizer::Tokenizer) -> bool {
    tokenizer.think_framing().is_some()
        || tokenizer.channel_framing().is_some()
        || tokenizer.message_framing().is_some()
        || tokenizer.content_kinds().is_some()
}

/// `GET /v1/models`, with Anthropic's pagination fields beside OpenAI's
/// `object: "list"` — there is one model, so there is never a next page.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [model_json(&state)],
        "has_more": false,
        "first_id": state.model_label,
        "last_id": state.model_label,
    }))
}

/// `GET /v1/models/{id}`: the served model when `id` names it.
pub async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if id != state.model_label {
        return super::anthropic::error_response(
            StatusCode::NOT_FOUND,
            format!(
                "model '{id}' is not served here; this server serves '{}'",
                state.model_label
            ),
        );
    }
    Json(model_json(&state)).into_response()
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Fields of OpenAI's schema that change nothing about what this server
/// generates, on any endpoint: who is asking, what to call it, and how the
/// platform behind OpenAI's own API should treat it. Everything else a request
/// names is a field the request's struct has, or it is refused.
const NO_EFFECT_FIELDS: &[&str] = &[
    "model",
    "user",
    "metadata",
    "store",
    "service_tier",
    "stream_options",
    "prompt_cache_key",
    "safety_identifier",
];

/// The most choices one request may ask for, as in OpenAI's schema.
const MAX_CHOICES: usize = 128;

/// The most alternatives per token `logprobs` may list, as in OpenAI's schema.
const MAX_TOP_LOGPROBS: usize = 20;

/// A request body of an OpenAI endpoint, whose struct refuses what it does not
/// have: see [`OpenAiBody`].
pub(crate) trait OpenAiRequest: serde::de::DeserializeOwned {
    const ENDPOINT: &'static str;

    /// Whether this field and value can be dropped as changing nothing.
    fn no_effect(key: &str, _value: &serde_json::Value) -> bool {
        NO_EFFECT_FIELDS.contains(&key)
    }
}

/// OpenAI's shape for a request that is refused.
pub(crate) fn invalid_request(message: impl Into<String>) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {
            "message": message.into(),
            "type": "invalid_request_error",
            "param": null,
            "code": null,
        }})),
    )
        .into_response()
}

/// `T` read from a JSON body, with anything it cannot honour answered with a
/// `400` naming the field. A field left out of the schema and dropped looks to
/// the client exactly like a knob that does nothing, and a client that sends
/// `stop` and gets it ignored has output it did not ask for.
fn parse_body<T: OpenAiRequest>(bytes: &[u8]) -> Result<T, String> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|err| format!("{}: the body is not JSON: {err}", T::ENDPOINT))?;
    let Some(fields) = value.as_object_mut() else {
        return Err(format!("{}: the body is not a JSON object", T::ENDPOINT));
    };
    fields.retain(|key, value| !T::no_effect(key, value));
    serde_json::from_value(value).map_err(|err| {
        let refused = if err.to_string().starts_with("unknown field") {
            " A field this server does not honour is refused rather than ignored."
        } else {
            ""
        };
        format!("{}: {err}.{refused}", T::ENDPOINT)
    })
}

/// The extractor for [`parse_body`].
pub struct OpenAiBody<T>(pub(crate) T);

impl<S: Send + Sync, T: OpenAiRequest> FromRequest<S> for OpenAiBody<T> {
    type Rejection = axum::response::Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = axum::body::Bytes::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;
        parse_body::<T>(&bytes)
            .map(OpenAiBody)
            .map_err(invalid_request)
    }
}

/// OpenAI's `stop`: one string or several.
#[derive(Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub(crate) enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    /// The strings of an optional `stop`, the empty ones left out.
    pub(crate) fn strings(field: Option<StopField>) -> Vec<String> {
        let strings = match field {
            None => Vec::new(),
            Some(StopField::One(one)) => vec![one],
            Some(StopField::Many(many)) => many,
        };
        strings.into_iter().filter(|s| !s.is_empty()).collect()
    }
}

/// A `presence_penalty` or `frequency_penalty` in OpenAI's range.
fn check_penalty(name: &str, value: Option<f32>) -> Result<(), String> {
    match value {
        Some(v) if !(-2.0..=2.0).contains(&v) => {
            Err(format!("{name} must be between -2 and 2, not {v}"))
        }
        _ => Ok(()),
    }
}

/// `logit_bias`, a map from token id to a bias, as this vocabulary's ids.
pub(crate) fn parse_logit_bias(
    bias: Option<&std::collections::HashMap<String, f32>>,
    vocab_size: usize,
) -> Result<Vec<(u32, f32)>, String> {
    let Some(bias) = bias else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::with_capacity(bias.len());
    for (id, &value) in bias {
        let token: u32 = id
            .parse()
            .map_err(|_| format!("logit_bias: `{id}` is not a token id"))?;
        if token as usize >= vocab_size {
            return Err(format!(
                "logit_bias: token {token} is past this model's {vocab_size} tokens"
            ));
        }
        if !(-100.0..=100.0).contains(&value) {
            return Err(format!(
                "logit_bias: {value} for token {token} is not between -100 and 100"
            ));
        }
        parsed.push((token, value));
    }
    parsed.sort_by_key(|(token, _)| *token);
    Ok(parsed)
}

/// `n`, checked: how many choices to generate.
fn choices_of(n: Option<usize>, id_slot: Option<usize>) -> Result<usize, String> {
    let n = n.unwrap_or(1);
    if !(1..=MAX_CHOICES).contains(&n) {
        return Err(format!("n must be between 1 and {MAX_CHOICES}, not {n}"));
    }
    if n > 1 && id_slot.is_some() {
        return Err("n > 1 runs on several slots, so it cannot be pinned with id_slot".into());
    }
    Ok(n)
}

/// A sampler for choice `index`: the same request with a seed of its own, so
/// that sampled choices differ and each one is still reproducible.
fn seeded(sampling: &SamplingParams, index: usize) -> SamplingParams {
    SamplingParams {
        seed: sampling.seed.wrapping_add(index as u64),
        ..sampling.clone()
    }
}

/// What a request cost across its choices: the prompt once, the replies summed.
fn total_usage(stats: &[GenerateStats]) -> serde_json::Value {
    let Some(first) = stats.first() else {
        return json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    };
    let completion: usize = stats.iter().map(|s| s.generated_tokens).sum();
    let mut usage = json!({
        "prompt_tokens": first.prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": first.prompt_tokens + completion,
        "prompt_tokens_details": {"cached_tokens": first.cached_tokens},
    });
    let dropped = stats.iter().map(|s| s.dropped_tokens).max().unwrap_or(0);
    if dropped > 0 {
        usage["context_dropped_tokens"] = dropped.into();
    }
    usage
}

/// One choice's whole output.
#[derive(Default)]
struct Collected {
    text: String,
    reasoning: String,
    logprobs: Vec<TokenLogprob>,
    finish: Option<crate::engine::generate::FinishReason>,
    stats: Option<GenerateStats>,
}

/// Reads one generation to its end. With `merge_reasoning`, reasoning stays in
/// the text: a raw completion has one field to put it in.
async fn collect(
    rx: &mut UnboundedReceiver<StreamEvent>,
    merge_reasoning: bool,
) -> Result<Collected, Box<axum::response::Response>> {
    let mut got = Collected::default();
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(text) => got.text.push_str(&text),
            StreamEvent::Reasoning(text) if merge_reasoning => got.text.push_str(&text),
            StreamEvent::Reasoning(text) => got.reasoning.push_str(&text),
            StreamEvent::Logprob(record) => got.logprobs.push(record),
            // Progress and per-token timings only mean something to a reader
            // watching the stream; a whole response already carries the final
            // `timings`/`usage`.
            StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
            StreamEvent::Done {
                finish_reason,
                stats,
            } => {
                got.finish = Some(finish_reason);
                got.stats = Some(stats);
                break;
            }
            StreamEvent::Overloaded => return Err(Box::new(crate::http::overloaded_response())),
            StreamEvent::NoRoom(message) => {
                return Err(Box::new(crate::http::no_room_response(message)));
            }
            StreamEvent::Error(err) => {
                return Err(Box::new(
                    (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
                ));
            }
        }
    }
    Ok(got)
}

/// The next event of any live generation, and whose. Reads from a different one
/// first each time, so one fast choice cannot starve the others. `None` once
/// all are over. A generation is over at its `Done` or when its channel closes.
async fn next_event(
    rxs: &mut [Option<UnboundedReceiver<StreamEvent>>],
    turn: &mut usize,
) -> Option<(usize, StreamEvent)> {
    let found = std::future::poll_fn(|cx| {
        let len = rxs.len();
        let mut live = false;
        for offset in 0..len {
            let index = (*turn + offset) % len;
            let Some(rx) = rxs[index].as_mut() else {
                continue;
            };
            match rx.poll_recv(cx) {
                std::task::Poll::Ready(Some(event)) => {
                    *turn = (index + 1) % len;
                    return std::task::Poll::Ready(Some((index, event)));
                }
                std::task::Poll::Ready(None) => rxs[index] = None,
                std::task::Poll::Pending => live = true,
            }
        }
        if live {
            std::task::Poll::Pending
        } else {
            std::task::Poll::Ready(None)
        }
    })
    .await;
    if let Some((index, StreamEvent::Done { .. })) = &found {
        rxs[*index] = None;
    }
    found
}

fn sse(value: &serde_json::Value) -> Result<axum::response::sse::Event, std::convert::Infallible> {
    Ok(axum::response::sse::Event::default().data(value.to_string()))
}

/// A token as text and bytes: a token can end half way through a character.
fn token_entry(bytes: &[Vec<u8>], id: u32) -> (String, &[u8]) {
    let raw = bytes.get(id as usize).map_or(&[][..], Vec::as_slice);
    (String::from_utf8_lossy(raw).into_owned(), raw)
}

/// One entry of a chat response's `logprobs.content`.
fn chat_logprob(bytes: &[Vec<u8>], record: &TokenLogprob) -> serde_json::Value {
    let (token, raw) = token_entry(bytes, record.id);
    let top: Vec<_> = record
        .top
        .iter()
        .map(|&(id, logprob)| {
            let (token, raw) = token_entry(bytes, id);
            json!({"token": token, "logprob": logprob, "bytes": raw})
        })
        .collect();
    json!({"token": token, "logprob": record.logprob, "bytes": raw, "top_logprobs": top})
}

/// A legacy completion's `logprobs`, parallel arrays. `offset` is where these
/// tokens' text starts in the choice's text, and moves past them.
fn legacy_logprobs(
    bytes: &[Vec<u8>],
    records: &[TokenLogprob],
    offset: &mut usize,
) -> serde_json::Value {
    let mut tokens = Vec::with_capacity(records.len());
    let mut offsets = Vec::with_capacity(records.len());
    let mut top = Vec::with_capacity(records.len());
    for record in records {
        let (token, raw) = token_entry(bytes, record.id);
        offsets.push(*offset);
        *offset += raw.len();
        tokens.push(token);
        top.push(
            record
                .top
                .iter()
                .map(|&(id, logprob)| (token_entry(bytes, id).0, json!(logprob)))
                .collect::<serde_json::Map<_, _>>(),
        );
    }
    json!({
        "tokens": tokens,
        "token_logprobs": records.iter().map(|r| r.logprob).collect::<Vec<_>>(),
        "top_logprobs": top,
        "text_offset": offsets,
    })
}

/// `tool_choice`, read.
#[derive(Clone, Debug, PartialEq)]
enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

/// The names in a request's `tools`.
fn declared_tools(tools: Option<&serde_json::Value>) -> Vec<&str> {
    tools
        .and_then(|t| t.as_array())
        .into_iter()
        .flatten()
        .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
        .collect()
}

/// Reads `tool_choice`, which only makes sense against the tools declared.
fn parse_tool_choice(
    choice: Option<&serde_json::Value>,
    tools: Option<&serde_json::Value>,
) -> Result<ToolChoice, String> {
    let declared = declared_tools(tools);
    let parsed = match choice {
        None | Some(serde_json::Value::Null) => ToolChoice::Auto,
        Some(serde_json::Value::String(mode)) => match mode.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(format!(
                    "tool_choice `{other}` is not one of auto, none, required, or a named function"
                ));
            }
        },
        Some(named) => match (
            named.get("type").and_then(|t| t.as_str()),
            named.pointer("/function/name").and_then(|n| n.as_str()),
        ) {
            (Some("function"), Some(name)) => ToolChoice::Named(name.to_string()),
            _ => {
                return Err(format!(
                    "tool_choice {named} is not one of auto, none, required, or                      {{\"type\": \"function\", \"function\": {{\"name\": …}}}}"
                ));
            }
        },
    };
    match &parsed {
        ToolChoice::Required if declared.is_empty() => {
            Err("tool_choice `required` needs at least one tool in `tools`".into())
        }
        ToolChoice::Named(name) if !declared.contains(&name.as_str()) => Err(format!(
            "tool_choice names `{name}`, which is not in `tools`"
        )),
        // With one tool, requiring a call is naming it.
        ToolChoice::Required if declared.len() == 1 => {
            Ok(ToolChoice::Named(declared[0].to_string()))
        }
        _ => Ok(parsed),
    }
}

impl OpenAiRequest for ChatCompletionRequest {
    const ENDPOINT: &'static str = "/v1/chat/completions";

    fn no_effect(key: &str, _value: &serde_json::Value) -> bool {
        // Which outputs to produce follows from the model served here.
        NO_EFFECT_FIELDS.contains(&key) || key == "modalities"
    }
}

impl OpenAiRequest for CompletionsRequest {
    const ENDPOINT: &'static str = "/v1/completions";

    fn no_effect(key: &str, value: &serde_json::Value) -> bool {
        NO_EFFECT_FIELDS.contains(&key)
            // The last chunk carries the timings whether or not this asks.
            || key == "timings_per_token"
            // The two legacy fields at the values that ask for nothing extra.
            || (key == "echo" && *value == json!(false))
            || (key == "best_of" && *value == json!(1))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// The rest of the sampler, under the same names `/completion` uses.
    ///
    /// These are not OpenAI's fields: the sampler has knobs OpenAI's schema
    /// has no word for.
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// OpenAI's flat penalties: over the same window as `repeat_penalty`, from
    /// -2 to 2. See `SamplingParams`.
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// Token id (as a string, JSON having no other key) to a bias from -100 to
    /// 100, added to its logit before anything else looks at it.
    #[serde(default)]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Text that ends the reply when written. Not sent; the stop may span
    /// tokens, and text is held back until a match is ruled out.
    #[serde(default)]
    stop: Option<StopField>,
    /// Report each token's log probability, with `top_logprobs` alternatives.
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    /// How many choices to generate, each on a slot of its own.
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// OpenAI's newer name for `max_tokens`.
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    /// How hard a reasoning model should think, passed to its template.
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    /// OpenAI's audio output options (`{"voice": ..., "format": ...}`),
    /// read on a speech model only.
    #[serde(default)]
    audio: Option<serde_json::Value>,
    /// llama.cpp's field name and default: reuse an already-computed KV cache
    /// for whatever prefix of this prompt one exists for. `false` forces a
    /// full prefill.
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
    /// Pin this request to one specific slot (llama.cpp's field name), so a
    /// conversation returns to the slot holding its own KV cache instead of
    /// reprefilling on a stranger's. See `engine::generate::GenerateRequest`.
    #[serde(default)]
    id_slot: Option<usize>,
    /// llama.cpp's field: attach a `timings` object to every streamed chunk,
    /// not only the last one. A client showing a live decode rate has no
    /// other honest source for it.
    #[serde(default)]
    timings_per_token: bool,
    /// llama.cpp's field: emit `prompt_progress` chunks *during* prefill, so
    /// a client can show how far a long prompt has got instead of an
    /// indefinite spinner.
    #[serde(default)]
    return_progress: bool,
    /// OpenAI's structured-output request. `{"type": "json_object"}`
    /// constrains generation so that only tokens keeping the output a valid
    /// JSON prefix can be sampled, and so that generation cannot stop until
    /// the document is complete — see `engine::constraint`.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// GBNF the reply is held to, in place of a `response_format`.
    #[serde(default)]
    grammar: Option<String>,
    /// OpenAI's tool array, handed to the chat template as `tools`. Kept as
    /// raw JSON: what a tool declaration must contain is the template's
    /// business, and every field this server invented an opinion about would
    /// be one a model's own template could no longer see.
    #[serde(default)]
    tools: Option<serde_json::Value>,
    /// Whether the reply may, must, or must not call a tool: `auto`, `none`,
    /// `required`, or `{"type": "function", "function": {"name": …}}`.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    /// `false`: at most one call, and the reply ends with it.
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
}

/// Where `head`'s tokens end in `tokens`, if they are a strict prefix of them
/// — a rendering's text being a prefix of another's does not make its tokens
/// one, since a merge can cross the join.
pub(crate) fn cache_point_of(head: &[u32], tokens: &[u32]) -> Option<usize> {
    (!head.is_empty() && head.len() < tokens.len() && tokens.starts_with(head))
        .then_some(head.len())
}

pub(crate) fn default_cache_prompt() -> bool {
    true
}

/// A request's `grammar` field, parsed. A `400` names what is wrong with it.
pub(crate) fn request_grammar(
    grammar: Option<&str>,
) -> Result<Option<Arc<crate::engine::grammar::Grammar>>, Box<axum::response::Response>> {
    grammar
        .map(|text| {
            crate::engine::grammar::Grammar::parse(text)
                .map_err(|error| Box::new(invalid_request(format!("grammar: {error}"))))
        })
        .transpose()
}

/// Whether `response_format` asks for JSON.
///
/// OpenAI's field is an object, `{"type": "json_object"}`. `json_schema`
/// asks for JSON too, and its schema is then held to by [`response_schema`].
///
/// Anything else — including `{"type": "text"}`, the default — is
/// unconstrained.
fn wants_json(response_format: Option<&serde_json::Value>) -> bool {
    response_format
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str())
        .is_some_and(|t| matches!(t, "json_object" | "json_schema"))
}

/// The schema `response_format: {"type": "json_schema", "json_schema":
/// {"schema": …}}` names, compiled; `None` for any other format. A schema
/// that cannot be followed is a `400` naming the reason, never a request
/// served unconstrained.
fn response_schema(
    response_format: Option<&serde_json::Value>,
) -> Result<Option<Arc<crate::engine::json_schema::Schema>>, Box<axum::response::Response>> {
    let Some(format) =
        response_format.filter(|f| f.get("type").and_then(|t| t.as_str()) == Some("json_schema"))
    else {
        return Ok(None);
    };
    let Some(schema) = format.pointer("/json_schema/schema") else {
        return Err(Box::new(
            (
                StatusCode::BAD_REQUEST,
                "response_format json_schema needs json_schema.schema\n",
            )
                .into_response(),
        ));
    };
    crate::engine::json_schema::compile(schema)
        .map(|schema| Some(Arc::new(schema)))
        .map_err(|err| {
            Box::new(
                (
                    StatusCode::BAD_REQUEST,
                    format!("response_format json_schema: {err}\n"),
                )
                    .into_response(),
            )
        })
}

/// `Role::Review`'s reasoning-suppression approximation: real llama-server
/// (`--reasoning-budget 0`) truncates a reasoning model's thinking phase by
/// pre-filling an *empty, already-closed* thinking block right after the
/// rendered prompt, so generation resumes immediately past it rather than
/// entering one at all — this is the same mechanism, without llama.cpp's
/// own reasoning-parsing machinery behind it. `<think>`/`</think>` is a
/// near-universal convention (DeepSeek-R1, QwQ, Qwen3, GLM, and this
/// project's own real-model testing) but not a guaranteed one — a model
/// using a different tag, or no explicit tag at all, won't be affected by
/// this (the `enable_thinking: false` template kwarg, passed separately,
/// is the other half of this approximation and *does* generalize to any
/// template that checks for it).
pub(crate) const EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

/// [`EMPTY_THINK_BLOCK`] for MiniMax's `<mm:think>` spelling. Its chat
/// template does not write an empty pair for a turn without reasoning: it
/// writes the closing marker alone, and its system prompt tells the model
/// that "when thinking is disabled, begin your response directly after the
/// `</mm:think>` prefix".
pub(crate) const MINIMAX_EMPTY_THINK_BLOCK: &str = "</mm:think>";

/// Appends [`EMPTY_THINK_BLOCK`] to a rendered prompt when — and only
/// when — that approximation is the right tool for this model.
///
/// It is not always. A `<|start|>…<|message|>`-framed model
/// (`Tokenizer::message_framing`, `muse-glimmer`) ends its generation
/// prompt *mid-header*, at `<|start|>assistant`, waiting for the model to
/// write its own recipient. Appending anything there lands inside the
/// header rather than after it, and the model then continues from a
/// malformed turn: the observed result was a reply that never wrote
/// `<|message|>` at all, which `engine::generate`'s own header filter then
/// withheld in full — an empty answer.
///
/// The same holds for a model that types each message body with a control
/// token (`Tokenizer::content_kinds`, `inkling`): its generation prompt
/// ends at `<|message_model|>`, and a `<think>` block appended there is
/// body text arriving before the marker that says what kind of body this
/// is.
///
/// Those models need no approximation anyway. Their reasoning is a whole
/// separate message — addressed `to=self`, or opened with
/// `<|content_thinking|>` — and a suppressing role drops it exactly (see
/// `MessageHeader`) rather than trying to talk the model out of producing
/// one.
///
/// **And it is only a marker where the vocabulary has one**
/// (`Tokenizer::think_framing`). Appended to a model that tokenizes
/// `<think>` as `<`, `think`, `>` — three ordinary tokens — this is not a
/// closed reasoning block, it is a sentence in the prompt, and a model reads
/// a sentence as an instruction. Measured on `gemma-4-E2B` at temperature 0,
/// same prompt, the only difference being this block:
///
/// | | the answer opens |
/// | :-- | :-- |
/// | without | `This is a classic example of **undefined behavior** in C…` |
/// | with | `Here's a thinking process for reviewing the code snippet…` |
///
/// So the approximation aimed at removing reasoning was *causing* it, in the
/// one role that exists to be rid of it — and in prose, where nothing
/// downstream can tell it from the answer and hide it. A vocabulary without
/// the markers needs no suppression in the prompt: it has no reasoning
/// channel to close, and `MessageHeader` still hides a marked block if one
/// somehow appears.
pub(crate) fn append_reasoning_suppression(
    prompt: &mut String,
    role: crate::config::Role,
    tokenizer: &crate::engine::tokenizer::Tokenizer,
) {
    if role.suppresses_reasoning() {
        append_empty_think_block(prompt, tokenizer);
    }
}

/// [`EMPTY_THINK_BLOCK`], on the models [`append_reasoning_suppression`]
/// says it is the right tool for, whatever the role.
pub(crate) fn append_empty_think_block(
    prompt: &mut String,
    tokenizer: &crate::engine::tokenizer::Tokenizer,
) {
    if tokenizer.message_framing().is_none()
        && tokenizer.content_kinds().is_none()
        && let Some((open, _)) = tokenizer.think_framing()
    {
        prompt.push_str(if tokenizer.token_text(open) == Some("<mm:think>") {
            MINIMAX_EMPTY_THINK_BLOCK
        } else {
            EMPTY_THINK_BLOCK
        });
    }
}

/// Reject an `id_slot` this server has no slot for, rather than silently
/// ignoring it and quietly reprefilling every turn — the exact failure the
/// field was added to end. Same shape `POST /slots/{id}` already answers with.
pub(crate) fn reject_unknown_slot(
    state: &AppState,
    id_slot: Option<usize>,
) -> Option<axum::response::Response> {
    let index = id_slot?;
    (index >= state.engine.slots.total()).then(|| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "id_slot {index} {} {} slots)\n",
                orangu::llm::SLOT_OUT_OF_RANGE_MARKER,
                state.engine.slots.total()
            ),
        )
            .into_response()
    })
}

/// The header an `orangu-coordinator` names a request's role in.
///
/// One `orangu-server` process has one role, and for a long time that meant
/// one process per role — so a coordinator whose `code` and `review` profiles
/// name the *same model file* stopped and restarted the server, reloading
/// gigabytes of identical weights and throwing away every cached prefix, to
/// change two sampling defaults and whether reasoning is suppressed. Those are
/// per-request facts. This is how they arrive per request.
pub(crate) const ROLE_HEADER: &str = "x-orangu-role";

/// The role this request is served under: the header's, when a coordinator
/// set one, and this process's own otherwise.
///
/// **Only ever narrows within what this process can do.** A header cannot turn
/// an `--embedding` server into one that answers chat: which endpoints work is
/// a property of the model that was loaded, not of the request, and an
/// unrecognised or unusable value falls back to the process role rather than
/// being reported — a request must not fail because a proxy in front of it
/// labelled it in a spelling this build does not know.
pub(crate) fn request_role(state: &AppState, headers: &HeaderMap) -> crate::config::Role {
    // An embedding server has no generation to tune, and a picture or video
    // server's role is the model's own (`Role::fixed_by_model`): neither is
    // a choice a header can make.
    if !state.engine.role.allows_generation() || state.engine.role.fixed_by_model() {
        return state.engine.role;
    }
    headers
        .get(ROLE_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| crate::config::Role::parse(value).ok())
        .filter(|role| role.allows_generation() && !role.fixed_by_model())
        .unwrap_or(state.engine.role)
}

/// A conversation rendered through the model's chat template: the prompt's
/// tokens, and where a model whose state cannot be rewound may keep its
/// cache — see `GenerateRequest::cache_point`.
pub(crate) struct RenderedChat {
    pub(crate) tokens: Vec<u32>,
    pub(crate) cache_point: Option<usize>,
}

/// How [`render_chat`] departs from rendering a conversation for the role's
/// next turn. The default departs from nothing.
#[derive(Default)]
pub(crate) struct RenderOptions<'a> {
    /// The reasoning settings to render with instead of the role's, and
    /// whether to close the reasoning channel before the answer — see
    /// [`append_empty_think_block`].
    pub(crate) reasoning: Option<(Reasoning<'a>, bool)>,
    /// An assistant turn the conversation ends in the middle of: the reply
    /// continues this text instead of starting a turn of its own.
    pub(crate) continue_from: Option<&'a str>,
    /// Text the reply is made to begin with, written after the generation
    /// prompt — a tool-call opener when a call is required.
    pub(crate) suffix: Option<&'a str>,
}

/// Stands in for a continued turn's text while the template renders it, so
/// where the text lands can be found whatever the template does around it.
const CONTINUE_SENTINEL: &str = "\u{1}orangu-continue\u{1}";

/// Renders `messages` (and `tools`, OpenAI's shape) into prompt tokens the
/// way every chat endpoint does, so `/v1/chat/completions` and `/v1/messages`
/// cannot drift apart on what a conversation becomes.
pub(crate) async fn render_chat(
    state: &AppState,
    role: crate::config::Role,
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    options: RenderOptions<'_>,
) -> Result<RenderedChat, Box<axum::response::Response>> {
    let Some(template) = state.engine.chat_template.clone() else {
        return Err(Box::new(
            (
                StatusCode::NOT_IMPLEMENTED,
                "model has no tokenizer.chat_template; use /v1/completions instead",
            )
                .into_response(),
        ));
    };
    let (bos, eos) = (
        state
            .engine
            .tokenizer
            .bos_token
            .and_then(|id| state.engine.tokenizer.token_text(id))
            .unwrap_or(""),
        state
            .engine
            .tokenizer
            .eos_token
            .and_then(|id| state.engine.tokenizer.token_text(id))
            .unwrap_or(""),
    );
    let (reasoning, suppress) = options
        .reasoning
        .unwrap_or((state.engine.reasoning_as(role), role.suppresses_reasoning()));
    let bad_request =
        |err: anyhow::Error| Box::new((StatusCode::BAD_REQUEST, err.to_string()).into_response());
    // An empty `tools: []` is treated as no tools: a template gating on
    // `{%- if tools -%}` would agree, and passing it through only invites a
    // template to emit an empty declaration block.
    let tools = tools.filter(|t| !matches!(t.as_array(), Some(a) if a.is_empty()));
    // A conversation's pictures, each held in its text's place by a
    // sentinel that is swapped for its tokens once the template has run.
    let pictures: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.images.iter().map(String::as_str))
        .collect();
    let vision = match (&state.engine.vision, pictures.is_empty()) {
        (_, true) => None,
        (Some(vision), false) => Some(vision.clone()),
        (None, false) => {
            return Err(Box::new(
                (
                    StatusCode::BAD_REQUEST,
                    "this model reads no pictures: it has no vision projector (an mmproj GGUF \
                     beside it)",
                )
                    .into_response(),
            ));
        }
    };
    let with_pictures;
    let messages = if vision.is_some() {
        with_pictures = picture_sentinels(messages);
        &with_pictures[..]
    } else {
        messages
    };
    let mut prompt = match options.continue_from {
        // The turn is rendered as finished, with a sentinel for its text,
        // and cut where the text would have ended: everything the template
        // writes to close a turn comes after that point.
        Some(text) => {
            let mut messages = messages.to_vec();
            messages.push(ChatMessage::text("assistant", CONTINUE_SENTINEL));
            let rendered = template
                .render_with_tools(&messages, false, bos, eos, reasoning, tools)
                .map_err(bad_request)?;
            let Some(at) = rendered.rfind(CONTINUE_SENTINEL) else {
                return Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        "this model's chat template does not render an assistant turn's \
                         text, so the turn cannot be continued",
                    )
                        .into_response(),
                ));
            };
            let mut prompt = rendered[..at].to_string();
            prompt.push_str(text);
            prompt
        }
        None => {
            let mut prompt = template
                .render_with_tools(messages, true, bos, eos, reasoning, tools)
                .map_err(bad_request)?;
            if suppress {
                append_empty_think_block(&mut prompt, &state.engine.tokenizer);
            }
            prompt
        }
    };
    // Rendered a second time, without the generation prompt, for a model
    // whose state cannot be rewound: the end of that rendering is where the
    // next turn's prompt still agrees with this one — see
    // `GenerateRequest::cache_point`.
    let head = (state.engine.recurrent && options.continue_from.is_none() && vision.is_none())
        .then(|| {
            template
                .render_with_tools(messages, false, bos, eos, reasoning, tools)
                .ok()
        })
        .flatten()
        .filter(|head| prompt.starts_with(head.as_str()) && head.len() < prompt.len());
    if let Some(suffix) = options.suffix {
        prompt.push_str(suffix);
    }
    let tokens = match vision {
        Some(vision) => {
            let pictures: Vec<String> = pictures.iter().map(|p| p.to_string()).collect();
            let tokenizer = state.engine.tokenizer.clone();
            tokio::task::spawn_blocking(move || {
                encode_with_pictures(&tokenizer, &vision, &prompt, &pictures)
            })
            .await
            .map_err(|err| bad_request(anyhow::anyhow!("reading the pictures: {err}")))?
            .map_err(bad_request)?
        }
        None => crate::http::encode_off_runtime(&state.engine.tokenizer, prompt, false).await,
    };
    let cache_point = match head {
        Some(head) => {
            let head = crate::http::encode_off_runtime(&state.engine.tokenizer, head, false).await;
            cache_point_of(&head, &tokens)
        }
        None => None,
    };
    Ok(RenderedChat {
        tokens,
        cache_point,
    })
}

/// Opens and closes the sentinel standing for picture `k` in a prompt.
const PICTURE_OPEN: &str = "\u{1}orangu-picture-";
const PICTURE_CLOSE: char = '\u{1}';

/// The sentinel standing for picture `k` until [`encode_with_pictures`]
/// swaps it for the picture's tokens.
pub(crate) fn picture_sentinel(k: usize) -> String {
    format!("{PICTURE_OPEN}{k}{PICTURE_CLOSE}")
}

/// `messages` with a sentinel for each picture where it stood in its text,
/// numbered in the conversation's order.
fn picture_sentinels(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut k = 0;
    messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            let mut content = String::with_capacity(m.content.len());
            let mut from = 0;
            for &at in &m.image_at {
                let at = at.min(m.content.len());
                content.push_str(&m.content[from..at]);
                content.push_str(&picture_sentinel(k));
                k += 1;
                from = at;
            }
            content.push_str(&m.content[from..]);
            m.content = content;
            m
        })
        .collect()
}

/// The picture at `url`, fetched: at most [`super::images::MAX_IMAGE_BYTES`],
/// through [`super::public_net`]'s guard, and refused when the server does
/// not call it an image.
fn fetch_picture(url: &str) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let parsed = reqwest::Url::parse(url)?;
    super::public_net::check(&parsed).map_err(anyhow::Error::msg)?;
    let response = super::public_net::guarded_blocking(reqwest::blocking::Client::builder())
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")))
        .build()?
        .get(parsed)
        .send()?
        .error_for_status()?;
    let kind = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    anyhow::ensure!(
        kind.starts_with("image/"),
        "{url} is not a picture (Content-Type: {kind})"
    );
    let limit = super::images::MAX_IMAGE_BYTES;
    let mut bytes = Vec::new();
    response.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= limit,
        "{url} is larger than the {} MB this server takes",
        limit / (1024 * 1024)
    );
    Ok(bytes)
}

/// `prompt`'s tokens with each picture sentinel replaced by that picture's
/// tokens — `pictures[k]` a data URL or an `http(s)` URL. The text between
/// sentinels is encoded piece by piece.
pub(crate) fn encode_with_pictures(
    tokenizer: &crate::engine::tokenizer::Tokenizer,
    vision: &crate::engine::vision::Vision,
    prompt: &str,
    pictures: &[String],
) -> anyhow::Result<Vec<u32>> {
    let mut tokens = Vec::new();
    let mut rest = prompt;
    while let Some(start) = rest.find(PICTURE_OPEN) {
        tokens.extend(tokenizer.encode(&rest[..start], false));
        let after = &rest[start + PICTURE_OPEN.len()..];
        let end = after
            .find(PICTURE_CLOSE)
            .ok_or_else(|| anyhow::anyhow!("a picture's place in the prompt is malformed"))?;
        let k: usize = after[..end]
            .parse()
            .map_err(|_| anyhow::anyhow!("a picture's place in the prompt is malformed"))?;
        let payload = pictures
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("the prompt names a picture that was not sent"))?;
        let bytes = if payload.starts_with("http://") || payload.starts_with("https://") {
            fetch_picture(payload).map_err(|err| anyhow::anyhow!("picture {}: {err:#}", k + 1))?
        } else {
            super::images::decode_image_payload(payload)
                .map_err(|err| anyhow::anyhow!("picture {}: {err}", k + 1))?
                .1
        };
        tokens.extend(
            vision
                .tokens(&bytes)
                .map_err(|err| anyhow::anyhow!("picture {}: {err:#}", k + 1))?,
        );
        rest = &after[end + PICTURE_CLOSE.len_utf8()..];
    }
    tokens.extend(tokenizer.encode(rest, false));
    Ok(tokens)
}

/// What a request's reasoning settings render as, when it set any.
fn reasoning_of<'a>(
    state: &'a AppState,
    role: crate::config::Role,
    effort: Option<&str>,
) -> Result<Option<(Reasoning<'a>, bool)>, String> {
    let Some(effort) = effort else {
        return Ok(None);
    };
    let mut reasoning = state.engine.reasoning_as(role);
    let mut suppress = role.suppresses_reasoning();
    let named = match effort {
        "none" => {
            reasoning.enabled = Some(false);
            suppress = true;
            None
        }
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" => Some("high"),
        other => {
            return Err(format!(
                "reasoning_effort `{other}` is not one of none, minimal, low, medium, high, xhigh"
            ));
        }
    };
    if let Some(named) = named {
        reasoning.effort = Some(named);
        reasoning.effort_is_default = false;
    }
    Ok(Some((reasoning, suppress)))
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OpenAiBody(req): OpenAiBody<ChatCompletionRequest>,
) -> axum::response::Response {
    let role = request_role(&state, &headers);
    if !state.engine.role.allows_generation() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "this server is running in --{} mode; generation endpoints are disabled",
                state.engine.role.label()
            ),
        )
            .into_response();
    }
    // A model that answers with a picture, a clip or speech generates no
    // tokens for these to apply to.
    if state.engine.image.is_some() || state.engine.video.is_some() || state.engine.audio.is_some()
    {
        let needs_text = [
            ("n", req.n.is_some_and(|n| n != 1)),
            ("logprobs", req.logprobs == Some(true)),
            (
                "stop",
                !StopField::strings(match &req.stop {
                    Some(StopField::One(one)) => Some(StopField::One(one.clone())),
                    Some(StopField::Many(many)) => Some(StopField::Many(many.clone())),
                    None => None,
                })
                .is_empty(),
            ),
            ("logit_bias", req.logit_bias.is_some()),
            ("tool_choice", req.tool_choice.is_some()),
        ];
        if let Some((field, _)) = needs_text.iter().find(|(_, set)| *set) {
            return invalid_request(format!(
                "`{field}` applies to generated text, and this model does not generate text"
            ));
        }
    }
    // An image model answers a chat turn with a picture — see `image_chat`.
    if state.engine.image.is_some() {
        return image_chat(state, req).await;
    }
    // A video model answers with a clip — see `video_chat`.
    if state.engine.video.is_some() {
        return video_chat(state, req).await;
    }
    // A speech model answers with the turn spoken — see `audio_chat`.
    if state.engine.audio.is_some() {
        return audio_chat(state, req).await;
    }
    if let Some(rejection) = reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
    let n = match choices_of(req.n, req.id_slot) {
        Ok(n) => n,
        Err(message) => return invalid_request(message),
    };
    let logprobs = match (req.logprobs.unwrap_or(false), req.top_logprobs) {
        (_, Some(top)) if top > MAX_TOP_LOGPROBS => {
            return invalid_request(format!(
                "top_logprobs must be at most {MAX_TOP_LOGPROBS}, not {top}"
            ));
        }
        (false, Some(top)) if top > 0 => {
            return invalid_request("top_logprobs needs logprobs: true");
        }
        (true, top) => Some(top.unwrap_or(0)),
        (false, _) => None,
    };
    let bias = parse_logit_bias(req.logit_bias.as_ref(), state.engine.tokenizer.vocab_size());
    let bias = match bias.and_then(|bias| {
        check_penalty("presence_penalty", req.presence_penalty)?;
        check_penalty("frequency_penalty", req.frequency_penalty)?;
        Ok(bias)
    }) {
        Ok(bias) => bias,
        Err(message) => return invalid_request(message),
    };
    let tool_choice = match parse_tool_choice(req.tool_choice.as_ref(), req.tools.as_ref()) {
        Ok(choice) => choice,
        Err(message) => return invalid_request(message),
    };
    let reasoning = match reasoning_of(&state, role, req.reasoning_effort.as_deref()) {
        Ok(reasoning) => reasoning,
        Err(message) => return invalid_request(message),
    };
    let withheld = tool_choice == ToolChoice::None;
    let tools = req.tools.as_ref().filter(|_| !withheld);
    let requires_call = matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_));
    if requires_call && wants_json(req.response_format.as_ref()) {
        return invalid_request(
            "a reply that must call a tool cannot also be held to a response_format",
        );
    }
    if req.grammar.is_some() && (requires_call || wants_json(req.response_format.as_ref())) {
        return invalid_request(
            "a grammar cannot be combined with a required tool call or a response_format",
        );
    }
    let grammar = match request_grammar(req.grammar.as_deref()) {
        Ok(grammar) => grammar,
        Err(rejection) => return *rejection,
    };
    // A call is required by making the reply begin with one. A template whose
    // call nothing here can read gives no way to do that, and answering as if
    // the requirement had been met would be worse than saying so.
    let prefix = match &tool_choice {
        ToolChoice::Required => super::anthropic::tool_call_prefix(&state, None),
        ToolChoice::Named(name) => super::anthropic::tool_call_prefix(&state, Some(name)),
        _ => None,
    };
    // Which tool is the model's to choose, among the declared ones.
    let reply_starts = match &tool_choice {
        ToolChoice::Required => {
            let names: Vec<String> = declared_tools(tools)
                .into_iter()
                .map(str::to_string)
                .collect();
            super::anthropic::declared_call_starts(&state, &names)
        }
        _ => Vec::new(),
    };
    if requires_call && prefix.is_none() {
        return invalid_request(
            "this model's chat template renders no tool call this server can read, so a call \
             cannot be required of it",
        );
    }
    let learned = tools
        .filter(|t| !matches!(t.as_array(), Some(a) if a.is_empty()))
        .and_then(|_| state.engine.chat_template.as_ref())
        .and_then(|template| template.learned_calls());
    let single = req.parallel_tool_calls == Some(false);
    let rules = CallRules {
        withheld,
        single,
        learned: learned.clone(),
        prefix: prefix.clone(),
    };
    // What holds any call this reply makes to the tools it declared. Not beside
    // a grammar or a document, which already say what the reply is.
    let calls = tools
        .filter(|_| grammar.is_none() && !wants_json(req.response_format.as_ref()))
        .zip(state.engine.chat_template.as_ref())
        .and_then(|(tools, template)| {
            let start = match &tool_choice {
                ToolChoice::Required => crate::engine::tool_learn::CallStart::AtName,
                ToolChoice::Named(name) => crate::engine::tool_learn::CallStart::AfterName(name),
                _ => crate::engine::tool_learn::CallStart::Free,
            };
            crate::engine::tool_learn::CallGrammar::build(template, tools, start)
        })
        .map(Arc::new);
    let RenderedChat {
        tokens,
        cache_point,
    } = match render_chat(
        &state,
        role,
        &req.messages,
        tools,
        RenderOptions {
            reasoning,
            suffix: prefix.as_deref(),
            ..Default::default()
        },
    )
    .await
    {
        Ok(rendered) => rendered,
        Err(rejection) => return *rejection,
    };

    let sampling = sampling_for(
        role,
        RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            logit_bias: bias,
        },
    );
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let json_schema = match response_schema(req.response_format.as_ref()) {
        Ok(schema) => schema,
        Err(rejection) => return *rejection,
    };
    if let Some(rejection) = crate::http::reject_oversized_context(&state, tokens.len()) {
        return rejection;
    }
    let stop_token_ids = state.engine.tokenizer.stop_token_ids();
    let created = unix_now();
    let model = state.model_label.clone();
    let mut stop = StopRules {
        strings: StopField::strings(req.stop),
        infill: None,
    };
    // A reply limited to one call ends at the first one's closer: the rest of
    // it would be generated for nobody. What the cut leaves is read as a call
    // the reply ended on.
    if single && !withheld {
        stop.strings
            .extend(tool_calls::closers().map(str::to_string));
        if let Some(close) = learned
            .as_deref()
            .map(|l| l.closer())
            .filter(|c| !c.is_empty())
        {
            stop.strings.push(close.to_string());
        }
    }
    let call_markers = learned.as_deref().map_or(Vec::new(), |l| l.markers());
    let json_output = wants_json(req.response_format.as_ref());

    let mut rxs = Vec::with_capacity(n);
    for index in 0..n {
        rxs.push(
            state
                .engine
                .generate(GenerateRequest {
                    prompt_tokens: tokens.clone(),
                    sampling: seeded(&sampling, index),
                    max_tokens,
                    stop_token_ids: stop_token_ids.clone(),
                    json_output,
                    json_schema: json_schema.clone(),
                    reply_starts: reply_starts.clone(),
                    grammar: grammar.clone(),
                    calls: calls.clone(),
                    n_keep: None,
                    context_shift: None,
                    dropped_tokens: 0,
                    cache_prompt: req.cache_prompt,
                    id_slot: req.id_slot,
                    timings_per_token: req.stream && req.timings_per_token,
                    role: Some(role),
                    cache_point,
                    background: false,
                    reasoning_budget: None,
                    stop: stop.clone(),
                    logprobs,
                    call_markers: call_markers.clone(),
                })
                .await,
        );
    }
    let token_bytes = logprobs.map(|_| state.engine.tokenizer.token_bytes_shared());

    if !req.stream {
        let tokenizer = &state.engine.tokenizer;
        let mut choices = Vec::with_capacity(n);
        let mut all_stats = Vec::with_capacity(n);
        for (index, rx) in rxs.iter_mut().enumerate() {
            let got = match collect(rx, false).await {
                Ok(got) => got,
                Err(rejection) => return *rejection,
            };
            let finish = got
                .finish
                .unwrap_or(crate::engine::generate::FinishReason::Stop);
            let split = split_whole(&got.text, &rules, finish, |text| {
                tokenizer.clean_up_tokenization_spaces(text)
            });
            // The whole answer is in hand, so this is where it gets tidied: the
            // blank line a model leaves between its prose and a call it made is
            // layout around a span that is no longer there.
            let visible = if split.has_calls() {
                split.content.trim()
            } else {
                &split.content
            };
            let mut message = json!({"role": "assistant", "content": visible});
            // `reasoning_content` is where DeepSeek put a reasoning model's
            // thinking and where llama-server puts it under
            // `--reasoning-format deepseek`; a client that does not know the
            // field ignores it and sees the answer alone, which is the point.
            // Absent rather than empty when there was no thinking, so "this
            // model does not reason" and "it reasoned about nothing" stay
            // distinguishable.
            if !got.reasoning.is_empty() {
                message["reasoning_content"] =
                    json!(tokenizer.clean_up_tokenization_spaces(&got.reasoning));
            }
            let mut finish_reason = finish_reason_str(finish);
            if split.has_calls() {
                message["tool_calls"] =
                    super::chat_choice::whole_calls(&split.calls, created, index, 0);
                // OpenAI's contract: a turn that called tools finishes for that
                // reason, whatever the sampler stopped on. A client keying off
                // `"stop"` would treat the calls as a finished answer.
                finish_reason = "tool_calls";
            }
            let mut choice = json!({
                "index": index,
                "message": message,
                "finish_reason": finish_reason,
            });
            if let Some(bytes) = &token_bytes {
                choice["logprobs"] = json!({
                    "content": got.logprobs.iter().map(|r| chat_logprob(bytes, r)).collect::<Vec<_>>(),
                    "refusal": null,
                });
            }
            choices.push(choice);
            all_stats.extend(got.stats);
        }
        let response = Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": choices,
            "usage": total_usage(&all_stats),
            "timings": all_stats.first().map(timings_json),
        }))
        .into_response();
        return with_server_timing(response, all_stats.first());
    }

    let return_progress = req.return_progress;
    let stream = async_stream::stream! {
        let id = format!("chatcmpl-{created}");
        let chunk = |index: usize, delta: serde_json::Value, finish: Option<&str>| json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
        });
        let mut rxs: Vec<_> = rxs.into_iter().map(Some).collect();
        let mut turn = 0usize;
        let mut all_stats: Vec<GenerateStats> = Vec::with_capacity(n);
        // Tool-call syntax has to be recognised before it is forwarded, and a
        // delimiter arrives a few tokens at a time; a `ChoiceStream` holds back
        // text that might still turn into a call and nothing else, so an
        // ordinary answer is not delayed at all.
        let mut choices: Vec<ChoiceStream> = (0..n)
            .map(|index| ChoiceStream::new(rules.clone(), created, index))
            .collect();
        for (index, choice) in choices.iter_mut().enumerate() {
            for delta in choice.start() {
                yield sse(&chunk(index, delta, None));
            }
        }
        while let Some((index, event)) = next_event(&mut rxs, &mut turn).await {
            match event {
                StreamEvent::PromptProgress { total, cached, processed, elapsed } => {
                    if return_progress && index == 0 {
                        // llama.cpp's shape, so a client reads mid-prefill
                        // progress from either server the same way.
                        let mut progress = chunk(index, json!({}), None);
                        progress["prompt_progress"] = json!({
                            "total": total,
                            "cache": cached,
                            "processed": processed,
                            "time_ms": elapsed.as_millis() as i64,
                        });
                        yield sse(&progress);
                    }
                }
                StreamEvent::Timings(stats) => {
                    // A chunk of its own rather than riding along with the
                    // next content delta: content can be held back for a
                    // while by the tool-call splitter above, and a decode
                    // rate that arrives in bursts is worse than none.
                    let mut timed = chunk(index, json!({}), None);
                    timed["timings"] = timings_json(&stats);
                    yield sse(&timed);
                }
                // Reasoning streams straight through in its own delta field,
                // bypassing the tool-call splitter entirely: a call is
                // something the model addresses to the caller, never part of
                // a body it addressed to itself, and holding thinking back
                // to see whether it turns into one would delay the only
                // thing there is to show during a long think.
                StreamEvent::Reasoning(text) => {
                    yield sse(&chunk(index, json!({"reasoning_content": text}), None));
                }
                StreamEvent::Token(text) => {
                    for delta in choices[index].push(&text) {
                        yield sse(&chunk(index, delta, None));
                    }
                }
                StreamEvent::Logprob(record) => {
                    if let Some(bytes) = &token_bytes {
                        let mut entry = chunk(index, json!({}), None);
                        entry["choices"][0]["logprobs"] =
                            json!({"content": [chat_logprob(bytes, &record)], "refusal": null});
                        yield sse(&entry);
                    }
                }
                StreamEvent::Done { finish_reason, stats } => {
                    // Whatever is still held back never became a call. It is
                    // the model's output and belongs in the answer.
                    let (deltas, reason) = choices[index].finish(finish_reason);
                    for delta in deltas {
                        yield sse(&chunk(index, delta, None));
                    }
                    let mut last = chunk(index, json!({}), Some(reason));
                    // The final chunk carries what the request cost, in both
                    // OpenAI's shape (`usage`) and llama.cpp's (`timings`,
                    // `prompt_progress`) — a streaming client otherwise has
                    // only its own wall clock, which cannot separate prefill
                    // from decode or a cache hit from real work. Several
                    // choices share one figure, sent once they are all over.
                    if n == 1 {
                        last["usage"] = usage_json(&stats);
                        last["timings"] = timings_json(&stats);
                        last["prompt_progress"] = prompt_progress_json(&stats);
                    }
                    all_stats.push(stats);
                    yield sse(&last);
                    if all_stats.len() == n {
                        break;
                    }
                }
                StreamEvent::Overloaded => {
                    yield sse(&json!({"error": crate::http::OVERLOADED_MESSAGE}));
                    return;
                }
                StreamEvent::Error(err) | StreamEvent::NoRoom(err) => {
                    yield sse(&json!({"error": err}));
                    return;
                }
            }
        }
        if n > 1 {
            yield sse(&json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [],
                "usage": total_usage(&all_stats),
                "timings": all_stats.first().map(timings_json),
            }));
        }
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    axum::response::sse::Sse::new(stream).into_response()
}

/// `/v1/chat/completions` on a `qwen_image` model: the last user message is
/// the prompt, its last attached picture (an `image_url` content part) is
/// what the picture starts from, and the assistant's answer is the picture
/// as a markdown image over a `data:` URL — which any chat client that
/// renders markdown shows inline, and which is also handed back raw under
/// `message.images` for one that wants the bytes.
///
/// Every other knob (size, steps, guidance, format) is the server's
/// `[orangu-server].image_*` default; `seed` is honoured. A client that
/// wants to choose those per request has `/v1/images/generations`.
///
/// Streamed, the chunks carry `image_progress` (`step`, `steps`,
/// `eta_seconds`) with an empty delta until the picture is done, so a
/// client that shows nothing but content still ends up with the picture and
/// one that reads progress can draw a bar.
async fn image_chat(state: Arc<AppState>, req: ChatCompletionRequest) -> axum::response::Response {
    let pipeline = match super::images::pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    let Some(turn) = req.messages.iter().rev().find(|m| m.role == "user") else {
        return (StatusCode::BAD_REQUEST, "no user message to draw from").into_response();
    };
    let init = match turn.images.last() {
        Some(payload) => match super::images::decode_image_payload(payload) {
            Ok(init) => Some(init),
            Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
        },
        None => None,
    };
    let request = match super::images::build_request(
        &pipeline.defaults(),
        pipeline.size_unit(),
        super::images::ImageParams {
            prompt: &turn.content,
            negative_prompt: None,
            size: None,
            output_format: None,
            seed: req.seed,
            steps: None,
            cfg_scale: None,
            init,
            strength: None,
        },
    ) {
        Ok(request) => request,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let prompt = request.prompt.clone();
    let created = unix_now();
    let model = state.model_label.clone();
    let (mut rx, cancel) = pipeline.spawn(request);
    let image_json = |image: &crate::engine::image::GeneratedImage| {
        json!({
            "b64_json": base64::engine::general_purpose::STANDARD.encode(&image.bytes),
            "output_format": image.format.extension(),
            "mime": image.format.mime(),
            "size": format!("{}x{}", image.width, image.height),
            "seed": image.seed,
            "steps": image.steps,
            "generation_ms": image.elapsed.as_millis() as u64,
        })
    };
    let zero_usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});

    if !req.stream {
        let _cancel = cancel;
        let image = loop {
            match rx.recv().await {
                Some(ImageEvent::Progress(_)) => {}
                Some(ImageEvent::Done(image)) => break image,
                Some(ImageEvent::Error(err)) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
                None => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "generation ended early")
                        .into_response();
                }
            }
        };
        return Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": super::images::markdown_image(&image, &prompt),
                    "images": [image_json(&image)],
                },
                "finish_reason": "stop",
            }],
            "usage": zero_usage,
        }))
        .into_response();
    }

    let stream = async_stream::stream! {
        let _cancel = cancel;
        let id = format!("chatcmpl-{created}");
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                ImageEvent::Progress(p) => {
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                        "image_progress": {
                            "step": p.step, "steps": p.steps,
                            "seconds_per_step": p.seconds_per_step,
                            "eta_seconds": p.eta_seconds(),
                        },
                    });
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                ImageEvent::Done(image) => {
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {
                            "role": "assistant",
                            "content": super::images::markdown_image(&image, &prompt),
                            "images": [image_json(&image)],
                        }, "finish_reason": "stop"}],
                        "usage": zero_usage,
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                ImageEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default().data(json!({"error": err}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

/// A chat turn on a video model: the last user message is the prompt, and
/// the answer is the clip — run as a `/v1/videos` job, so the answer links
/// to `/v1/videos/{id}/content` (markdown a chat client can follow) and
/// carries the job as `videos`. Streaming sends `video_progress` chunks
/// with an empty delta while it runs.
async fn video_chat(state: Arc<AppState>, req: ChatCompletionRequest) -> axum::response::Response {
    let pipeline = match super::videos::pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    let Some(turn) = req.messages.iter().rev().find(|m| m.role == "user") else {
        return (StatusCode::BAD_REQUEST, "no user message to film").into_response();
    };
    let request = match super::videos::build_request(
        &pipeline.defaults(),
        super::videos::VideoParams {
            prompt: &turn.content,
            seed: req.seed,
            ..Default::default()
        },
    ) {
        Ok(request) => request,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let prompt = request.prompt.replace(['[', ']', '\n'], " ");
    let created = unix_now();
    let model = state.model_label.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let id = super::videos::start_job(&pipeline, &model, request, Some(tx));
    let content = format!("[{prompt}](/v1/videos/{id}/content)");
    let zero_usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    let job = |id: &str| -> serde_json::Value { super::videos::job_json(id).unwrap_or_default() };

    if !req.stream {
        loop {
            match rx.recv().await {
                Some(crate::engine::video::VideoEvent::Progress(_)) => {}
                Some(crate::engine::video::VideoEvent::Done(_)) => break,
                Some(crate::engine::video::VideoEvent::Error(err)) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
                None => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "generation ended early")
                        .into_response();
                }
            }
        }
        return Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "videos": [job(&id)]},
                "finish_reason": "stop",
            }],
            "usage": zero_usage,
        }))
        .into_response();
    }

    let stream = async_stream::stream! {
        let chat_id = format!("chatcmpl-{created}");
        while let Some(event) = rx.recv().await {
            match event {
                crate::engine::video::VideoEvent::Progress(p) => {
                    let chunk = json!({
                        "id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                        "video_progress": {
                            "id": id, "step": p.step, "steps": p.steps, "phase": p.phase,
                            "eta_seconds": p.eta_seconds,
                        },
                    });
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(chunk.to_string()));
                }
                crate::engine::video::VideoEvent::Done(_) => {
                    let chunk = json!({
                        "id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {
                            "role": "assistant", "content": content, "videos": [job(&id)],
                        }, "finish_reason": "stop"}],
                        "usage": zero_usage,
                    });
                    yield Ok(axum::response::sse::Event::default().data(chunk.to_string()));
                    yield Ok(axum::response::sse::Event::default().data("[DONE]"));
                    break;
                }
                crate::engine::video::VideoEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default().data(json!({"error": err}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

/// A chat turn on a speech model: the last user message spoken, in the
/// voice and language the server defaults to (or the request's `audio`
/// options: `voice`, `format`), answered as OpenAI's audio output — the
/// text as the transcript and the clip, base64, in `message.audio`.
async fn audio_chat(state: Arc<AppState>, req: ChatCompletionRequest) -> axum::response::Response {
    let pipeline = match super::audio::pipeline_or_reject(&state) {
        Ok(pipeline) => pipeline,
        Err(response) => return *response,
    };
    let Some(turn) = req.messages.iter().rev().find(|m| m.role == "user") else {
        return (StatusCode::BAD_REQUEST, "no user message to speak").into_response();
    };
    let defaults = pipeline.defaults();
    let options = req.audio.as_ref();
    let voice = options.and_then(|a| a.get("voice")).and_then(|v| match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("id").and_then(|i| i.as_str()).map(str::to_string),
        _ => None,
    });
    let format = match super::audio::parse_format(
        options
            .and_then(|a| a.get("format"))
            .and_then(|f| f.as_str()),
        defaults.format,
    ) {
        Ok(format) => format,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let request = match super::audio::build_request(
        &defaults,
        super::audio::SpeechParams {
            text: &turn.content,
            voice: voice.as_deref(),
            temperature: req.temperature,
            top_k: req.top_k,
            seed: req.seed,
            ..Default::default()
        },
    ) {
        Ok(request) => request,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let created = unix_now();
    let model = state.model_label.clone();
    let started = std::time::Instant::now();
    let speech = match pipeline.speak(request).await {
        Ok(speech) => speech,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
    };
    super::audio::log_speech(&speech, started.elapsed(), " (chat)");
    use base64::Engine as _;
    let data = base64::engine::general_purpose::STANDARD
        .encode(format.encode(&speech.samples, speech.sample_rate));
    let transcript = turn.content.trim().to_string();
    let audio = json!({
        "id": format!("audio-{created}-{}", speech.seed),
        "data": data,
        "format": format.label(),
        "transcript": transcript,
        "expires_at": created + 3600,
        "voice": speech.voice,
        "seconds": speech.seconds(),
    });
    let zero_usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    let message = json!({"role": "assistant", "content": null, "audio": audio});
    if !req.stream {
        return Json(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": zero_usage,
        }))
        .into_response();
    }
    let chunk = json!({
        "id": format!("chatcmpl-{created}"), "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": message, "finish_reason": "stop"}],
        "usage": zero_usage,
    });
    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(
            axum::response::sse::Event::default().data(chunk.to_string()),
        );
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    axum::response::sse::Sse::new(stream).into_response()
}

/// The sampler fields an HTTP request may override, gathered so the two
/// OpenAI-compatible endpoints apply them through one place.
///
/// Every field is `Option`: `None` keeps the role's default rather than
/// meaning "zero". A `temperature` of `0.0` is greedy and a `repeat_penalty`
/// of `1.0` is off, so the difference between "unset" and "set to the
/// neutral value" is real and cannot be collapsed.
#[derive(Default)]
pub(crate) struct RequestSampling {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repeat_penalty: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) logit_bias: Vec<(u32, f32)>,
}

/// The role's defaults with a request's overrides applied.
///
/// One function rather than a copy per endpoint: the two of them had drifted
/// — chat honoured `temperature`/`top_p`/`seed` and completions honoured
/// `temperature` alone — and nothing about either endpoint made that
/// deliberate. A field added to one and forgotten in the other is not a
/// visible bug, because the ignored value simply has no effect.
pub(crate) fn sampling_for(role: crate::config::Role, req: RequestSampling) -> SamplingParams {
    let mut sampling = SamplingParams::default_for_role(role);
    if let Some(v) = req.temperature {
        sampling.temperature = v;
    }
    if let Some(v) = req.top_p {
        sampling.top_p = v;
    }
    if let Some(v) = req.top_k {
        sampling.top_k = v;
    }
    if let Some(v) = req.min_p {
        sampling.min_p = v;
    }
    if let Some(v) = req.repeat_penalty {
        sampling.repeat_penalty = v;
    }
    if let Some(v) = req.seed {
        sampling.seed = v;
    }
    if let Some(v) = req.presence_penalty {
        sampling.presence_penalty = v;
    }
    if let Some(v) = req.frequency_penalty {
        sampling.frequency_penalty = v;
    }
    sampling.logit_bias = req.logit_bias;
    sampling
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionsRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// The native endpoint's name for `max_tokens`, which a client written
    /// against both sends beside it.
    #[serde(default)]
    n_predict: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    /// The rest of the sampler — see [`ChatCompletionRequest::top_k`].
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// See [`ChatCompletionRequest::presence_penalty`].
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// See [`ChatCompletionRequest::stop`].
    #[serde(default)]
    stop: Option<StopField>,
    /// This schema's `logprobs` is the number of alternatives per token.
    #[serde(default)]
    logprobs: Option<usize>,
    /// See [`ChatCompletionRequest::n`].
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// Not part of OpenAI's completions schema: the sampler and the constraint
    /// are the same engine here, so it does what it does on chat.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// GBNF the reply is held to, in place of a `response_format`.
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    stream: bool,
    /// Keep generating to `max_tokens` even if the model emits EOS (llama.cpp's
    /// field name). Used by benchmarks (`orangu-bench --depths`, `llama-bench
    /// -d`) to time a fixed number of decode steps at a given context depth
    /// regardless of what the model would otherwise stop on.
    #[serde(default)]
    ignore_eos: bool,
    /// See [`ChatCompletionRequest::cache_prompt`]. `orangu-bench --pp` sets
    /// this `false` so each timed run prefills for real.
    #[serde(default = "default_cache_prompt")]
    cache_prompt: bool,
    /// See [`ChatCompletionRequest::id_slot`].
    #[serde(default)]
    id_slot: Option<usize>,
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OpenAiBody(req): OpenAiBody<CompletionsRequest>,
) -> axum::response::Response {
    let role = request_role(&state, &headers);
    if let Some(rejection) = super::audio::reject_text_endpoint(&state, "/v1/completions") {
        return rejection;
    }
    if !state.engine.role.allows_generation() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "this server is running in --{} mode; generation endpoints are disabled",
                state.engine.role.label()
            ),
        )
            .into_response();
    }
    if let Some(rejection) = reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
    let n = match choices_of(req.n, req.id_slot) {
        Ok(n) => n,
        Err(message) => return invalid_request(message),
    };
    if let Some(top) = req.logprobs.filter(|&top| top > MAX_TOP_LOGPROBS) {
        return invalid_request(format!(
            "logprobs must be at most {MAX_TOP_LOGPROBS}, not {top}"
        ));
    }
    let bias = parse_logit_bias(req.logit_bias.as_ref(), state.engine.tokenizer.vocab_size());
    let bias = match bias.and_then(|bias| {
        check_penalty("presence_penalty", req.presence_penalty)?;
        check_penalty("frequency_penalty", req.frequency_penalty)?;
        Ok(bias)
    }) {
        Ok(bias) => bias,
        Err(message) => return invalid_request(message),
    };
    // The file's own answer, not `true`: a model trained without a
    // BOS gets a prompt one token longer than the same text is
    // anywhere else, and only the continuation shows it.
    let tokens = crate::http::encode_off_runtime(
        &state.engine.tokenizer,
        req.prompt.clone(),
        state.engine.tokenizer.wants_bos(),
    )
    .await;
    let sampling = sampling_for(
        role,
        RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            logit_bias: bias,
        },
    );
    let max_tokens = req.max_tokens.or(req.n_predict).unwrap_or(256);
    if req.grammar.is_some() && wants_json(req.response_format.as_ref()) {
        return invalid_request("a grammar cannot be combined with a response_format");
    }
    let grammar = match request_grammar(req.grammar.as_deref()) {
        Ok(grammar) => grammar,
        Err(rejection) => return *rejection,
    };
    let json_schema = match response_schema(req.response_format.as_ref()) {
        Ok(schema) => schema,
        Err(rejection) => return *rejection,
    };
    if let Some(rejection) = crate::http::reject_oversized_context(&state, tokens.len()) {
        return rejection;
    }
    // `ignore_eos` drops the EOS stop token so generation runs the full
    // `max_tokens` — the "measure decode, not content" contract benchmarks need.
    let stop_token_ids: Vec<u32> = if req.ignore_eos {
        Vec::new()
    } else {
        state.engine.tokenizer.stop_token_ids()
    };
    let created = unix_now();
    let model = state.model_label.clone();
    let stop = StopRules {
        strings: StopField::strings(req.stop),
        infill: None,
    };
    let json_output = wants_json(req.response_format.as_ref());

    let mut rxs = Vec::with_capacity(n);
    for index in 0..n {
        rxs.push(
            state
                .engine
                .generate(GenerateRequest {
                    prompt_tokens: tokens.clone(),
                    sampling: seeded(&sampling, index),
                    max_tokens,
                    stop_token_ids: stop_token_ids.clone(),
                    json_output,
                    json_schema: json_schema.clone(),
                    reply_starts: Vec::new(),
                    grammar: grammar.clone(),
                    calls: None,
                    n_keep: None,
                    context_shift: None,
                    dropped_tokens: 0,
                    cache_prompt: req.cache_prompt,
                    id_slot: req.id_slot,
                    timings_per_token: false,
                    role: Some(role),
                    cache_point: None,
                    background: false,
                    reasoning_budget: None,
                    stop: stop.clone(),
                    logprobs: req.logprobs,
                    call_markers: Vec::new(),
                })
                .await,
        );
    }
    let token_bytes = req
        .logprobs
        .map(|_| state.engine.tokenizer.token_bytes_shared());

    if req.stream {
        let stream = async_stream::stream! {
            let id = format!("cmpl-{created}");
            let mut rxs: Vec<_> = rxs.into_iter().map(Some).collect();
            let mut turn = 0usize;
            let mut offsets = vec![0usize; n];
            let mut all_stats: Vec<GenerateStats> = Vec::with_capacity(n);
            while let Some((index, event)) = next_event(&mut rxs, &mut turn).await {
                match event {
                    StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                    // A raw completion has one field to put text in and no
                    // message shape to split it across, so reasoning stays
                    // in it — a caller that prefilled `<think>` here asked
                    // for exactly that text back.
                    StreamEvent::Token(text) | StreamEvent::Reasoning(text) => {
                        yield sse(&json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": model,
                            "choices": [{"index": index, "text": text, "finish_reason": null}],
                        }));
                    }
                    StreamEvent::Logprob(record) => {
                        if let Some(bytes) = &token_bytes {
                            let logprobs = legacy_logprobs(
                                bytes,
                                std::slice::from_ref(&record),
                                &mut offsets[index],
                            );
                            yield sse(&json!({
                                "id": id, "object": "text_completion", "created": created,
                                "model": model,
                                "choices": [{"index": index, "text": "", "logprobs": logprobs,
                                             "finish_reason": null}],
                            }));
                        }
                    }
                    StreamEvent::Done { finish_reason, stats } => {
                        // A final chunk before `[DONE]`, carrying the same
                        // cost figures the chat endpoint reports — this is
                        // the endpoint `orangu-bench` measures through, so
                        // it is where prefill numbers have to come from.
                        let mut last = json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": model,
                            "choices": [{"index": index, "text": "",
                                         "finish_reason": finish_reason_str(finish_reason)}],
                        });
                        if n == 1 {
                            last["usage"] = usage_json(&stats);
                            last["timings"] = timings_json(&stats);
                            last["prompt_progress"] = prompt_progress_json(&stats);
                        }
                        all_stats.push(stats);
                        yield sse(&last);
                        if all_stats.len() == n {
                            break;
                        }
                    }
                    StreamEvent::Overloaded => {
                        yield sse(&json!({"error": crate::http::OVERLOADED_MESSAGE}));
                        return;
                    }
                    StreamEvent::Error(err) | StreamEvent::NoRoom(err) => {
                        yield sse(&json!({"error": err}));
                        return;
                    }
                }
            }
            if n > 1 {
                yield sse(&json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [],
                    "usage": total_usage(&all_stats),
                    "timings": all_stats.first().map(timings_json),
                }));
            }
            yield Ok(axum::response::sse::Event::default().data("[DONE]"));
        };
        return axum::response::sse::Sse::new(stream).into_response();
    }

    let mut choices = Vec::with_capacity(n);
    let mut all_stats = Vec::with_capacity(n);
    for (index, rx) in rxs.iter_mut().enumerate() {
        // See the streaming arm above: one field, no message shape.
        let got = match collect(rx, true).await {
            Ok(got) => got,
            Err(rejection) => return *rejection,
        };
        let text = state
            .engine
            .tokenizer
            .clean_up_tokenization_spaces(&got.text);
        let mut choice = json!({
            "index": index,
            "text": text,
            "finish_reason": finish_reason_str(
                got.finish.unwrap_or(crate::engine::generate::FinishReason::Stop),
            ),
        });
        if let Some(bytes) = &token_bytes {
            choice["logprobs"] = legacy_logprobs(bytes, &got.logprobs, &mut 0);
        }
        choices.push(choice);
        all_stats.extend(got.stats);
    }
    let response = Json(json!({
        "id": format!("cmpl-{created}"),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": total_usage(&all_stats),
        "timings": all_stats.first().map(timings_json),
    }))
    .into_response();
    with_server_timing(response, all_stats.first())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    input: EmbeddingsInput,
}

#[derive(Serialize)]
struct EmbeddingDatum {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> axum::response::Response {
    if let Some(rejection) = super::audio::reject_text_endpoint(&state, "/v1/embeddings") {
        return rejection;
    }
    let inputs = match req.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    let mut data = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0usize;
    for (index, text) in inputs.into_iter().enumerate() {
        match pooled_embedding(&state, &text).await {
            Ok(pooled) => {
                prompt_tokens += pooled.prompt_tokens;
                data.push(EmbeddingDatum {
                    object: "embedding",
                    embedding: pooled.embedding,
                    index,
                });
            }
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
        }
    }
    Json(json!({
        "object": "list",
        "data": data,
        "model": state.model_label,
        // OpenAI's embeddings `usage`, which has these two fields and not the
        // `completion_tokens` of `usage_json` above — there is no completion.
        // Summed across a batched `input`, as OpenAI's is.
        //
        // Not cosmetic: the token count is not recoverable from the response
        // otherwise, so anything measuring embedding throughput had to
        // tokenize the input itself with a second tool to learn what it just
        // paid for. `doc/perf/embed_bench.sh` shelled out to `llama-tokenize`
        // for exactly this, and `orangu-bench --embed` reads this field
        // instead — from either server, since llama-server sends it too.
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }))
    .into_response()
}

/// One embedding and the prompt length that produced it.
///
/// The token count is carried out rather than recomputed by the caller
/// because this is where it is known exactly — `encode_for_embedding` applies
/// the model's own embedding-specific framing, so a caller re-tokenizing the
/// same string could easily disagree with the forward pass that was actually
/// run. It is what `/v1/embeddings` reports as `usage.prompt_tokens`.
pub(crate) struct PooledEmbedding {
    pub embedding: Vec<f32>,
    pub prompt_tokens: usize,
}

/// Pools a model's per-token final hidden states per its own `<arch>.
/// pooling_type` ([`PoolingType`] — `Mean`, e.g. `gemma-embedding`, or
/// `Last`, e.g. `qwen3vl`-embedding models; every other value falls back
/// to `Mean`, see that type's own doc comment), runs the model's own
/// [`ModelForward::post_pool_projection`] (identity for most
/// architectures; `gemma-embedding`'s `dense_2`/`dense_3` sentence-
/// transformers adapters for that one), then L2-normalizes the result.
pub(crate) async fn pooled_embedding(
    state: &Arc<AppState>,
    text: &str,
) -> Result<PooledEmbedding, String> {
    let tokens = state.engine.tokenizer.encode_for_embedding(text);
    let prompt_tokens = tokens.len();
    let model = state.engine.serving().model;
    let n_embd = model.config().n_embd;
    let pooling_type = model.config().pooling_type;
    let hidden = tokio::task::spawn_blocking({
        let model = model.clone();
        move || crate::engine::generate::embedding_hidden_states(model.as_ref(), &tokens)
    })
    .await
    .map_err(|err| err.to_string())?
    .map_err(|err| err.to_string())?;
    let n_tokens = (hidden.len() / n_embd).max(1);

    let pooled = match pooling_type {
        PoolingType::Last => hidden[(n_tokens - 1) * n_embd..].to_vec(),
        PoolingType::Mean => {
            let mut pooled = vec![0f32; n_embd];
            for row in hidden.chunks(n_embd) {
                for (p, v) in pooled.iter_mut().zip(row.iter()) {
                    *p += v;
                }
            }
            for v in pooled.iter_mut() {
                *v /= n_tokens as f32;
            }
            pooled
        }
    };

    let mut pooled = tokio::task::spawn_blocking(move || model.post_pool_projection(pooled))
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())?;

    let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in pooled.iter_mut() {
            *v /= norm;
        }
    }
    Ok(PooledEmbedding {
        embedding: pooled,
        prompt_tokens,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn pictures_are_held_in_place_by_numbered_sentinels() {
        let mut first = ChatMessage::text("user", "");
        first.push_image("data:image/png;base64,AAAA".into());
        first.content.push_str("Compare");
        let mut second = ChatMessage::text("user", "with");
        second.push_image("data:image/png;base64,BBBB".into());
        let marked = picture_sentinels(&[first, second]);
        assert_eq!(
            marked[0].content,
            format!("{PICTURE_OPEN}0{PICTURE_CLOSE}Compare")
        );
        assert_eq!(
            marked[1].content,
            format!("with{PICTURE_OPEN}1{PICTURE_CLOSE}")
        );
    }
    use super::*;
    use std::time::Duration;

    fn stats() -> GenerateStats {
        GenerateStats {
            prompt_tokens: 200,
            cached_tokens: 50,
            dropped_tokens: 0,
            prompt_time: Duration::from_millis(4000),
            generated_tokens: 30,
            generate_time: Duration::from_millis(1000),
            workers: Vec::new(),
        }
    }

    /// Every sampler field a request can name must actually reach the
    /// sampler, on **both** OpenAI-compatible endpoints.
    ///
    /// Deserializing real JSON rather than constructing the structs is the
    /// point: a field that exists on the struct but is spelled differently on
    /// the wire fails here and nowhere else.
    #[test]
    fn every_sampler_field_a_request_names_reaches_the_sampler() {
        let sampler = serde_json::json!({
            "temperature": 0.25,
            "top_p": 0.6,
            "top_k": 7,
            "min_p": 0.125,
            "repeat_penalty": 1.5,
            "seed": 99,
        });
        let role = crate::config::Role::All;

        let with = |key: &str, value: serde_json::Value| {
            let mut body = sampler.clone();
            body[key] = value;
            body
        };
        let chat: ChatCompletionRequest =
            serde_json::from_value(with("messages", json!([]))).expect("chat request parses");
        let completions: CompletionsRequest = serde_json::from_value(with("prompt", json!("hi")))
            .expect("completions request parses");

        for (label, got) in [
            (
                "chat",
                sampling_for(
                    role,
                    RequestSampling {
                        temperature: chat.temperature,
                        top_p: chat.top_p,
                        top_k: chat.top_k,
                        min_p: chat.min_p,
                        repeat_penalty: chat.repeat_penalty,
                        seed: chat.seed,
                        ..Default::default()
                    },
                ),
            ),
            (
                "completions",
                sampling_for(
                    role,
                    RequestSampling {
                        temperature: completions.temperature,
                        top_p: completions.top_p,
                        top_k: completions.top_k,
                        min_p: completions.min_p,
                        repeat_penalty: completions.repeat_penalty,
                        seed: completions.seed,
                        ..Default::default()
                    },
                ),
            ),
        ] {
            assert_eq!(got.temperature, 0.25, "{label}: temperature");
            assert_eq!(got.top_p, 0.6, "{label}: top_p");
            assert_eq!(got.top_k, 7, "{label}: top_k");
            assert_eq!(got.min_p, 0.125, "{label}: min_p");
            assert_eq!(got.repeat_penalty, 1.5, "{label}: repeat_penalty");
            assert_eq!(got.seed, 99, "{label}: seed");
        }
    }

    /// An unset field keeps the role's default rather than collapsing to
    /// zero — and the default penalty is **off**.
    ///
    /// `1.0` is the neutral value, not the absent one, so "unset" and "set to
    /// neutral" must both leave the penalty off while remaining distinct for
    /// every other field. The default itself is pinned here because it is a
    /// behavioural promise: a penalty applied per token id falls hardest on
    /// the newline, and turning it on by default corrupts generated code.
    #[test]
    fn an_unset_field_keeps_the_role_default_and_the_penalty_is_off() {
        let empty: CompletionsRequest =
            serde_json::from_value(serde_json::json!({"prompt": "hi"})).expect("parses");
        let got = sampling_for(
            crate::config::Role::All,
            RequestSampling {
                temperature: empty.temperature,
                top_p: empty.top_p,
                top_k: empty.top_k,
                min_p: empty.min_p,
                repeat_penalty: empty.repeat_penalty,
                seed: empty.seed,
                ..Default::default()
            },
        );
        let default = SamplingParams::default_for_role(crate::config::Role::All);
        assert_eq!(got.temperature, default.temperature);
        assert_eq!(got.top_k, default.top_k);
        assert_eq!(
            got.repeat_penalty, 1.0,
            "the repetition penalty must default to off"
        );
    }

    fn refusal<T: OpenAiRequest>(body: serde_json::Value) -> String {
        match parse_body::<T>(body.to_string().as_bytes()) {
            Ok(_) => panic!("{body} was accepted"),
            Err(message) => message,
        }
    }

    /// A field this server cannot honour is answered, by name. A client that
    /// sends one and hears nothing has output it did not ask for.
    #[test]
    fn an_unknown_field_is_refused_by_name() {
        for (field, value) in [("functions", json!([])), ("mirostat", json!(2))] {
            let message = refusal::<ChatCompletionRequest>(json!({"messages": [], field: value}));
            assert!(message.contains(field), "{message}");
            assert!(message.contains("/v1/chat/completions"), "{message}");
            assert!(message.contains("refused rather than ignored"), "{message}");
        }
        let message = refusal::<CompletionsRequest>(json!({"prompt": "a", "typical_p": 0.9}));
        assert!(message.contains("typical_p"), "{message}");
    }

    /// The other half of the same rule: what every SDK sends and changes
    /// nothing here must not cost its user a `400`.
    #[test]
    fn what_changes_nothing_is_accepted() {
        let fields = json!({
            "model": "m", "user": "u", "metadata": {"a": "b"}, "store": false,
            "service_tier": "auto", "stream_options": {"include_usage": true},
            "prompt_cache_key": "k", "safety_identifier": "s",
        });
        let with = |mut extra: serde_json::Value| {
            for (key, value) in fields.as_object().unwrap() {
                extra[key] = value.clone();
            }
            extra.to_string()
        };
        parse_body::<ChatCompletionRequest>(
            with(json!({"messages": [], "modalities": ["text"]})).as_bytes(),
        )
        .unwrap_or_else(|err| panic!("{err}"));
        parse_body::<CompletionsRequest>(
            with(json!({"prompt": "a", "echo": false, "best_of": 1, "timings_per_token": true}))
                .as_bytes(),
        )
        .unwrap_or_else(|err| panic!("{err}"));
    }

    /// `echo` and `best_of` at values that do ask for something are refused:
    /// accepting those would be ignoring them.
    #[test]
    fn legacy_fields_that_ask_for_something_are_refused() {
        for body in [
            json!({"prompt": "a", "echo": true}),
            json!({"prompt": "a", "best_of": 3}),
            json!({"prompt": "a", "suffix": "b"}),
        ] {
            refusal::<CompletionsRequest>(body);
        }
    }

    /// Every request this project's own clients send has to keep working.
    #[test]
    fn what_this_projects_clients_send_is_accepted() {
        parse_body::<CompletionsRequest>(
            json!({
                "prompt": "p", "max_tokens": 1, "n_predict": 1, "temperature": 0, "stream": true,
                "cache_prompt": false, "ignore_eos": true, "timings_per_token": true, "model": "m",
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap_or_else(|err| panic!("orangu-bench: {err}"));
        parse_body::<ChatCompletionRequest>(
            json!({
                "model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true,
                "tools": [], "max_tokens": 8, "timings_per_token": true, "return_progress": true,
                "id_slot": 0, "temperature": 0, "cache_prompt": true,
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap_or_else(|err| panic!("orangu: {err}"));
    }

    #[test]
    fn a_body_that_is_not_an_object_is_refused() {
        assert!(
            parse_body::<CompletionsRequest>(b"not json").is_err_and(|e| e.contains("not JSON"))
        );
        assert!(
            parse_body::<CompletionsRequest>(b"[1]")
                .is_err_and(|e| e.contains("not a JSON object"))
        );
    }

    #[test]
    fn stop_is_one_string_or_several_and_empty_ones_do_nothing() {
        let stop = |body: serde_json::Value| {
            let req: ChatCompletionRequest =
                parse_body(body.to_string().as_bytes()).unwrap_or_else(|err| panic!("{err}"));
            StopField::strings(req.stop)
        };
        assert_eq!(stop(json!({"stop": "a"})), ["a"]);
        assert_eq!(stop(json!({"stop": ["a", "", "b"]})), ["a", "b"]);
        assert!(stop(json!({"stop": null})).is_empty());
        assert!(stop(json!({})).is_empty());
        assert!(stop(json!({"stop": ""})).is_empty());
    }

    /// The new penalties, the bias and the stop reach the engine on both.
    #[test]
    fn the_new_sampler_fields_parse_on_both_endpoints() {
        let body = json!({
            "prompt": "a", "messages": [],
            "presence_penalty": 0.5, "frequency_penalty": -0.5,
            "logit_bias": {"7": -100, "3": 2.5}, "stop": ["x"], "n": 2,
        });
        let chat: ChatCompletionRequest = serde_json::from_value({
            let mut b = body.clone();
            b.as_object_mut().unwrap().remove("prompt");
            b
        })
        .unwrap();
        let completion: CompletionsRequest = serde_json::from_value({
            let mut b = body;
            b.as_object_mut().unwrap().remove("messages");
            b
        })
        .unwrap();
        for (bias, presence, frequency) in [
            (
                &chat.logit_bias,
                chat.presence_penalty,
                chat.frequency_penalty,
            ),
            (
                &completion.logit_bias,
                completion.presence_penalty,
                completion.frequency_penalty,
            ),
        ] {
            assert_eq!(
                parse_logit_bias(bias.as_ref(), 10).unwrap(),
                [(3, 2.5), (7, -100.0)]
            );
            let got = sampling_for(
                crate::config::Role::All,
                RequestSampling {
                    presence_penalty: presence,
                    frequency_penalty: frequency,
                    logit_bias: parse_logit_bias(bias.as_ref(), 10).unwrap(),
                    ..Default::default()
                },
            );
            assert_eq!(got.presence_penalty, 0.5);
            assert_eq!(got.frequency_penalty, -0.5);
            assert_eq!(got.logit_bias.len(), 2);
        }
    }

    #[test]
    fn logit_bias_and_penalties_out_of_range_are_refused() {
        let bias = |entries: serde_json::Value| {
            let map: std::collections::HashMap<String, f32> =
                serde_json::from_value(entries).unwrap();
            parse_logit_bias(Some(&map), 10)
        };
        assert!(
            bias(json!({"word": 1}))
                .unwrap_err()
                .contains("not a token id")
        );
        assert!(
            bias(json!({"10": 1}))
                .unwrap_err()
                .contains("past this model")
        );
        assert!(
            bias(json!({"1": 101}))
                .unwrap_err()
                .contains("between -100 and 100")
        );
        assert!(parse_logit_bias(None, 10).unwrap().is_empty());
        assert!(check_penalty("presence_penalty", Some(2.5)).is_err());
        assert!(check_penalty("presence_penalty", Some(-2.0)).is_ok());
        assert!(check_penalty("presence_penalty", None).is_ok());
    }

    #[test]
    fn the_number_of_choices_is_checked() {
        assert_eq!(choices_of(None, Some(0)), Ok(1));
        assert_eq!(choices_of(Some(3), None), Ok(3));
        assert!(choices_of(Some(0), None).is_err());
        assert!(choices_of(Some(MAX_CHOICES + 1), None).is_err());
        assert!(
            choices_of(Some(2), Some(0))
                .unwrap_err()
                .contains("id_slot")
        );
    }

    /// Sampled choices must differ, and a seed must still reproduce them.
    #[test]
    fn each_choice_has_a_seed_of_its_own() {
        let sampling = SamplingParams {
            seed: 41,
            ..Default::default()
        };
        assert_eq!(seeded(&sampling, 0).seed, 41);
        assert_eq!(seeded(&sampling, 3).seed, 44);
        let top = SamplingParams {
            seed: u64::MAX,
            ..Default::default()
        };
        assert_eq!(seeded(&top, 1).seed, 0);
    }

    /// One prompt, however many replies: `prompt_tokens` is not multiplied.
    #[test]
    fn usage_counts_the_prompt_once_and_sums_the_replies() {
        assert_eq!(total_usage(&[stats()]), usage_json(&stats()));
        let two = total_usage(&[stats(), stats()]);
        assert_eq!(two["prompt_tokens"], 200);
        assert_eq!(two["completion_tokens"], 60);
        assert_eq!(two["total_tokens"], 260);
        assert_eq!(total_usage(&[])["total_tokens"], 0);
    }

    fn tools() -> serde_json::Value {
        json!([{"type": "function", "function": {"name": "read", "parameters": {}}}])
    }

    #[test]
    fn tool_choice_reads_every_form_openai_defines() {
        let t = tools();
        let parse = |choice: serde_json::Value| parse_tool_choice(Some(&choice), Some(&t));
        assert_eq!(parse_tool_choice(None, None), Ok(ToolChoice::Auto));
        assert_eq!(parse(json!(null)), Ok(ToolChoice::Auto));
        assert_eq!(parse(json!("auto")), Ok(ToolChoice::Auto));
        assert_eq!(parse(json!("none")), Ok(ToolChoice::None));
        // With one tool there is only one call to require.
        assert_eq!(
            parse(json!("required")),
            Ok(ToolChoice::Named("read".to_string()))
        );
        let two = json!([
            {"type": "function", "function": {"name": "read"}},
            {"type": "function", "function": {"name": "write"}},
        ]);
        assert_eq!(
            parse_tool_choice(Some(&json!("required")), Some(&two)),
            Ok(ToolChoice::Required)
        );
        assert_eq!(
            parse(json!({"type": "function", "function": {"name": "read"}})),
            Ok(ToolChoice::Named("read".to_string()))
        );
    }

    /// A requirement that cannot be met is refused, not quietly dropped.
    #[test]
    fn a_tool_choice_that_cannot_be_met_is_refused() {
        let t = tools();
        let parse = |choice: serde_json::Value| parse_tool_choice(Some(&choice), Some(&t));
        assert!(parse(json!("always")).unwrap_err().contains("`always`"));
        assert!(
            parse(json!({"type": "function", "function": {"name": "write"}}))
                .unwrap_err()
                .contains("`write`")
        );
        assert!(parse(json!({"type": "allowed_tools"})).is_err());
        assert!(
            parse_tool_choice(Some(&json!("required")), None)
                .unwrap_err()
                .contains("at least one tool")
        );
        // Saying none needs nothing to say it about.
        assert_eq!(
            parse_tool_choice(Some(&json!("none")), None),
            Ok(ToolChoice::None)
        );
    }

    fn bytes() -> Vec<Vec<u8>> {
        vec![b"a".to_vec(), b"bc".to_vec(), vec![0xC3]]
    }

    fn record(id: u32) -> TokenLogprob {
        TokenLogprob {
            id,
            logprob: -0.5,
            top: vec![(1, -0.25), (0, -0.5)],
        }
    }

    #[test]
    fn a_chat_logprob_is_in_openais_shape() {
        let entry = chat_logprob(&bytes(), &record(0));
        assert_eq!(entry["token"], "a");
        assert_eq!(entry["logprob"], -0.5);
        assert_eq!(entry["bytes"], json!([97]));
        assert_eq!(entry["top_logprobs"][0]["token"], "bc");
        assert_eq!(entry["top_logprobs"][0]["bytes"], json!([98, 99]));
    }

    /// A token ending half way through a character: `bytes` is what lets a
    /// client put the character back together.
    #[test]
    fn a_token_that_ends_mid_character_keeps_its_bytes() {
        let entry = chat_logprob(&bytes(), &record(2));
        assert_eq!(entry["bytes"], json!([0xC3]));
        assert_eq!(chat_logprob(&bytes(), &record(99))["bytes"], json!([]));
    }

    #[test]
    fn legacy_logprobs_are_parallel_arrays_with_running_offsets() {
        let mut offset = 0;
        let got = legacy_logprobs(&bytes(), &[record(1), record(0)], &mut offset);
        assert_eq!(got["tokens"], json!(["bc", "a"]));
        assert_eq!(got["token_logprobs"], json!([-0.5, -0.5]));
        assert_eq!(got["text_offset"], json!([0, 2]));
        assert_eq!(got["top_logprobs"][0]["bc"], -0.25);
        assert_eq!(offset, 3);
    }

    /// Choices are read as they come, none starves, and the merge ends only
    /// when every one has.
    #[tokio::test]
    async fn the_merge_takes_turns_and_ends_with_the_last_choice() {
        let (a_tx, a_rx) = tokio::sync::mpsc::unbounded_channel();
        let (b_tx, b_rx) = tokio::sync::mpsc::unbounded_channel();
        for tx in [&a_tx, &b_tx] {
            for text in ["1", "2"] {
                tx.send(StreamEvent::Token(text.to_string())).unwrap();
            }
        }
        let mut rxs = vec![Some(a_rx), Some(b_rx)];
        let mut turn = 0;
        let mut order = Vec::new();
        for _ in 0..4 {
            let (index, _) = next_event(&mut rxs, &mut turn).await.unwrap();
            order.push(index);
        }
        assert_eq!(order, [0, 1, 0, 1]);
        drop(a_tx);
        b_tx.send(StreamEvent::Done {
            finish_reason: crate::engine::generate::FinishReason::Stop,
            stats: stats(),
        })
        .unwrap();
        let (index, _) = next_event(&mut rxs, &mut turn).await.unwrap();
        assert_eq!(index, 1);
        assert!(rxs[1].is_none(), "a choice is over at its Done");
        assert!(next_event(&mut rxs, &mut turn).await.is_none());
    }

    #[test]
    fn usage_reports_totals_and_the_cached_share_of_the_prompt() {
        let usage = usage_json(&stats());
        assert_eq!(usage["prompt_tokens"], 200);
        assert_eq!(usage["completion_tokens"], 30);
        assert_eq!(usage["total_tokens"], 230);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 50);
    }

    #[test]
    fn timings_report_prompt_and_decode_rates_in_llama_cpp_field_names() {
        let timings = timings_json(&stats());
        assert_eq!(timings["prompt_n"], 200);
        assert_eq!(timings["prompt_ms"], 4000.0);
        // 200 tokens in 4s, 30 tokens in 1s.
        assert_eq!(timings["prompt_per_second"], 50.0);
        assert_eq!(timings["predicted_per_second"], 30.0);
        assert_eq!(timings["prompt_per_token_ms"], 20.0);
    }

    /// A prompt served entirely from cache must not read as instant prefill:
    /// `processed` is what actually went through a forward pass.
    #[test]
    fn prompt_progress_separates_cached_tokens_from_processed_ones() {
        let progress = prompt_progress_json(&stats());
        assert_eq!(progress["total"], 200);
        assert_eq!(progress["cache"], 50);
        assert_eq!(progress["processed"], 150);
        assert_eq!(progress["time_ms"], 4000);
    }

    /// `Server-Timing` names the prompt and the generation, and in a tree
    /// each node of the pipeline, in order; `timings` carries the nodes too.
    #[test]
    fn server_timing_names_every_node() {
        let alone = stats();
        assert_eq!(
            server_timing(&alone).to_str().unwrap(),
            "prompt;dur=4000.0, generate;dur=1000.0"
        );
        assert!(timings_json(&alone).get("workers").is_none());

        let mut tree = stats();
        tree.workers = vec![
            ("top:8400".to_string(), Duration::from_micros(12_340)),
            ("10.0.0.2:8400".to_string(), Duration::from_millis(31)),
        ];
        assert_eq!(
            server_timing(&tree).to_str().unwrap(),
            "prompt;dur=4000.0, generate;dur=1000.0, node0;desc=\"top:8400\";dur=12.3, \
             node1;desc=\"10.0.0.2:8400\";dur=31.0"
        );
        let workers = &timings_json(&tree)["workers"];
        assert_eq!(workers[1]["node"], "10.0.0.2:8400");
        assert_eq!(workers[1]["ms"], 31.0);
    }

    /// Rate helpers divide by the token count; an empty generation (a prompt
    /// that stopped immediately) must not produce a division by zero.
    #[test]
    fn timings_survive_a_request_that_generated_nothing() {
        let empty = GenerateStats {
            prompt_tokens: 0,
            cached_tokens: 0,
            dropped_tokens: 0,
            prompt_time: Duration::ZERO,
            generated_tokens: 0,
            generate_time: Duration::ZERO,
            workers: Vec::new(),
        };
        let timings = timings_json(&empty);
        assert_eq!(timings["prompt_per_token_ms"], 0.0);
        assert_eq!(timings["predicted_per_token_ms"], 0.0);
    }
}

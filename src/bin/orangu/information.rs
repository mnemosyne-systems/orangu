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

//! `/information` gathers as much information as possible about the active
//! server: every OpenAI-compatible endpoint orangu itself talks to
//! (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`), whatever
//! llama.cpp-native endpoints (`/health`, `/props`, `/slots`, `/metrics`) it
//! exposes, and whether it is actually an orangu-coordinator proxy
//! (`/v1/coordinator`) rather than llama.cpp or a generic OpenAI-compatible
//! server. Every capability is probed independently, so a plain
//! OpenAI-compatible server (which lacks the llama.cpp-native endpoints and
//! `/v1/coordinator`) still gets a full report — those rows just come back
//! unavailable rather than failing the whole command.
//!
//! `/v1/chat/completions` is the one endpoint where "probe" can mean a real
//! request rather than a side-effect-free GET: on a local llama.cpp server a
//! one-token generation costs nothing worth avoiding, so it is actually sent
//! there; on any other (potentially hosted, potentially billed) provider its
//! availability is inferred from `/v1/models` instead.

use crate::*;
use serde_json::Value;
use std::time::Duration;

/// How long to wait for any single probe request before giving up on it.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// One row of the `/information` report: an API surface and whether the
/// connected server currently exposes it.
pub(crate) struct Capability {
    /// `OpenAI` for the standard chat/embeddings API, `llama.cpp` for the
    /// server's native (non-OpenAI) endpoints.
    api: &'static str,
    endpoint: &'static str,
    available: bool,
    details: String,
}

/// Probe every known capability on `profile`'s server and return one
/// [`Capability`] per endpoint, in report order. `is_embeddings_server` is
/// whether this server was the one detected at startup as serving
/// `/v1/embeddings` (see `embeddings_server` in `main.rs`); it is reused
/// rather than re-probed so `/information` never triggers a real embedding
/// request (and its cost, on a hosted API) just to populate a status report.
pub(crate) async fn gather_server_information(
    profile: &LlmConfiguration,
    is_embeddings_server: bool,
) -> Vec<Capability> {
    let endpoint = normalized_openai_endpoint(&profile.endpoint);
    let api_key = profile.api_key.as_deref();
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
        .unwrap_or_default();
    let is_llama_cpp = profile.provider.eq_ignore_ascii_case("llama.cpp");

    let models = probe_models(&client, &endpoint, api_key).await;
    // A real generation costs nothing extra on a local llama.cpp server, so
    // /v1/chat/completions is actually probed there (capped at one token);
    // on any other provider — a hosted API, potentially billed per request —
    // its availability is inferred from /v1/models instead.
    let chat_completions = if is_llama_cpp {
        probe_chat_completions(&client, &endpoint, api_key, &profile.model).await
    } else {
        chat_completions_capability(&models, &profile.model)
    };
    let embeddings = embeddings_capability(is_embeddings_server);
    let coordinator = simplify_unavailable(
        probe_get(
            &client,
            &endpoint,
            "orangu-coordinator",
            "/v1/coordinator",
            api_key,
            summarize_coordinator,
        )
        .await,
    );

    vec![
        coordinator,
        models,
        chat_completions,
        embeddings,
        probe_get(
            &client,
            &endpoint,
            "llama.cpp",
            "/health",
            api_key,
            summarize_health,
        )
        .await,
        probe_get(
            &client,
            &endpoint,
            "llama.cpp",
            "/props",
            api_key,
            summarize_props,
        )
        .await,
        simplify_unavailable(
            probe_get(
                &client,
                &endpoint,
                "llama.cpp",
                "/slots",
                api_key,
                summarize_ok,
            )
            .await,
        ),
        simplify_unavailable(
            probe_get(
                &client,
                &endpoint,
                "llama.cpp",
                "/metrics",
                api_key,
                summarize_reachable,
            )
            .await,
        ),
    ]
}

/// `GET /v1/models`, reusing the same request builder `/model` completion and
/// startup detection use, so the model list is always consistent.
async fn probe_models(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> Capability {
    let response = crate::models::models_request(client, endpoint, api_key)
        .send()
        .await;
    match response {
        Ok(response) if response.status().is_success() => {
            let ids = response
                .json::<crate::models::ModelsResponse>()
                .await
                .map(|models| models.model_ids())
                .unwrap_or_default();
            let details = models_details(&ids);
            Capability {
                api: "OpenAI",
                endpoint: "/v1/models",
                available: true,
                details,
            }
        }
        Ok(response) => Capability {
            api: "OpenAI",
            endpoint: "/v1/models",
            available: false,
            details: http_status_details(response.status()),
        },
        Err(_) => Capability {
            api: "OpenAI",
            endpoint: "/v1/models",
            available: false,
            details: "not reachable".to_string(),
        },
    }
}

/// A minimal, non-streaming chat-completion request body: one user message
/// and a one-token response cap, just enough to confirm the endpoint actually
/// generates rather than merely accepting the request.
#[derive(Serialize)]
struct ChatCompletionsProbeRequest<'a> {
    model: &'a str,
    messages: [ChatCompletionsProbeMessage<'a>; 1],
    max_tokens: u32,
    stream: bool,
}

#[derive(Serialize)]
struct ChatCompletionsProbeMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// `POST /v1/chat/completions` with a one-token cap, used only on a local
/// llama.cpp server (see [`gather_server_information`]) where a real
/// generation costs nothing worth avoiding — every other endpoint this module
/// probes is a side-effect-free GET.
async fn probe_chat_completions(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
    model: &str,
) -> Capability {
    let body = ChatCompletionsProbeRequest {
        model,
        messages: [ChatCompletionsProbeMessage {
            role: "user",
            content: "hi",
        }],
        max_tokens: 1,
        stream: false,
    };
    let mut request = client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&body);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => Capability {
            api: "OpenAI",
            endpoint: "/v1/chat/completions",
            available: true,
            details: "Ok".to_string(),
        },
        Ok(response) => Capability {
            api: "OpenAI",
            endpoint: "/v1/chat/completions",
            available: false,
            details: http_status_details(response.status()),
        },
        Err(_) => Capability {
            api: "OpenAI",
            endpoint: "/v1/chat/completions",
            available: false,
            details: "not reachable".to_string(),
        },
    }
}

/// `GET <path>`, applying `summarize` to the parsed JSON body on success. Used
/// for every llama.cpp-native endpoint: they are all side-effect-free GETs, so
/// probing them costs nothing beyond the request itself.
async fn probe_get(
    client: &reqwest::Client,
    endpoint: &str,
    api: &'static str,
    path: &'static str,
    api_key: Option<&str>,
    summarize: fn(&Value) -> String,
) -> Capability {
    let mut request = client.get(format!("{endpoint}{path}"));
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => {
            let details = response
                .json::<Value>()
                .await
                .map(|value| summarize(&value))
                .unwrap_or_else(|_| "reachable".to_string());
            Capability {
                api,
                endpoint: path,
                available: true,
                details,
            }
        }
        Ok(response) => Capability {
            api,
            endpoint: path,
            available: false,
            details: http_status_details(response.status()),
        },
        Err(_) => Capability {
            api,
            endpoint: path,
            available: false,
            details: "not reachable".to_string(),
        },
    }
}

/// Render the advertised model ids for the `/v1/models` details column: a
/// single model is shown bare (no count prefix needed when there is only one
/// choice), and multiple models are simply comma-separated.
fn models_details(ids: &[String]) -> String {
    if ids.is_empty() {
        "reachable, but advertised no models".to_string()
    } else {
        ids.join(", ")
    }
}

/// Collapse an unavailable capability's details down to a flat "Not
/// available", dropping the HTTP-status reasoning `http_status_details`
/// would otherwise give — for rows (`/slots`) where that extra detail isn't
/// worth showing. Available capabilities pass through unchanged.
fn simplify_unavailable(capability: Capability) -> Capability {
    if capability.available {
        capability
    } else {
        Capability {
            details: "Not available".to_string(),
            ..capability
        }
    }
}

/// Explain a non-success HTTP status for a probed endpoint: `501` is how
/// llama.cpp reports an endpoint disabled at server startup (`--no-slots`,
/// missing `--metrics`, …), `404` means the server never exposed it, and
/// anything else is shown as-is.
fn http_status_details(status: reqwest::StatusCode) -> String {
    match status.as_u16() {
        501 => "disabled by the server (missing startup flag)".to_string(),
        404 => "not supported by this server".to_string(),
        code => format!("HTTP {code}"),
    }
}

/// Append `label=value` to `parts` when `value` is present; used for the
/// numeric `/props` fields below.
fn push_field<T: std::fmt::Display>(parts: &mut Vec<String>, label: &str, value: Option<T>) {
    if let Some(value) = value {
        parts.push(format!("{label}={value}"));
    }
}

/// Append `label=value` to `parts` when `value` is a non-empty string; used
/// for the string `/props` fields below.
fn push_str(parts: &mut Vec<String>, label: &str, value: Option<&str>) {
    if let Some(value) = value
        && !value.is_empty()
    {
        parts.push(format!("{label}={value}"));
    }
}

/// Pull as much as possible out of a llama.cpp `/props` response — the
/// closest thing llama.cpp exposes over HTTP to "how was the server
/// started": the context size (`--ctx-size`), the parallel slot count
/// (`--parallel`), the sampling defaults (`--temp`, `--top-k`, `--top-p`),
/// the loaded model's path and tokenizer boundary tokens, and the build
/// version. The schema is not guaranteed to be stable across llama.cpp
/// versions, so every field is read defensively and simply omitted when
/// absent rather than treated as an error. (llama.cpp does not expose
/// hardware-only flags such as thread count, GPU layer count, or batch size
/// through any HTTP endpoint, so those never appear here.)
fn summarize_props(value: &Value) -> String {
    let mut parts = Vec::new();
    push_field(
        &mut parts,
        "n_ctx",
        value
            .pointer("/default_generation_settings/n_ctx")
            .and_then(Value::as_i64),
    );
    push_field(
        &mut parts,
        "n_predict",
        value
            .pointer("/default_generation_settings/n_predict")
            .and_then(Value::as_i64),
    );
    push_field(
        &mut parts,
        "total_slots",
        value.get("total_slots").and_then(Value::as_i64),
    );
    push_field(
        &mut parts,
        "temperature",
        value
            .pointer("/default_generation_settings/params/temperature")
            .and_then(Value::as_f64),
    );
    push_field(
        &mut parts,
        "top_k",
        value
            .pointer("/default_generation_settings/params/top_k")
            .and_then(Value::as_i64),
    );
    push_field(
        &mut parts,
        "top_p",
        value
            .pointer("/default_generation_settings/params/top_p")
            .and_then(Value::as_f64),
    );
    push_str(
        &mut parts,
        "model_path",
        value.get("model_path").and_then(Value::as_str),
    );
    push_str(
        &mut parts,
        "bos_token",
        value.get("bos_token").and_then(Value::as_str),
    );
    push_str(
        &mut parts,
        "eos_token",
        value.get("eos_token").and_then(Value::as_str),
    );
    push_str(
        &mut parts,
        "build",
        value.get("build_info").and_then(Value::as_str),
    );
    let has_chat_template = value
        .get("chat_template")
        .and_then(Value::as_str)
        .is_some_and(|template| !template.is_empty());
    parts.push(format!(
        "chat_template={}",
        if has_chat_template { "yes" } else { "no" }
    ));
    parts.join(" ")
}

/// Summarize `/v1/coordinator`: show the orangu-coordinator version when the
/// field is present, confirming the connected server actually is an
/// orangu-coordinator proxy rather than llama.cpp or a generic
/// OpenAI-compatible server.
fn summarize_coordinator(value: &Value) -> String {
    match value.get("version").and_then(Value::as_str) {
        Some(version) if !version.is_empty() => format!("orangu-coordinator v{version}"),
        _ => "reachable".to_string(),
    }
}

/// Summarize `/health`: show the `status` field llama.cpp reports (`ok`,
/// `loading model`, `error`, …), capitalized, when present, otherwise just
/// note it responded.
fn summarize_health(value: &Value) -> String {
    match value.get("status").and_then(Value::as_str) {
        Some(status) if !status.is_empty() => capitalize_first(status),
        _ => "reachable".to_string(),
    }
}

/// Capitalize the first character of `text`, leaving the rest untouched
/// (`"ok"` → `"Ok"`, `"loading model"` → `"Loading model"`).
fn capitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// `/metrics`'s JSON body isn't worth summarizing field by field: a 2xx
/// response is all the report needs to say.
fn summarize_reachable(_value: &Value) -> String {
    "reachable".to_string()
}

/// `/slots`'s JSON body isn't worth summarizing field by field either: a 2xx
/// response is all the report needs to say.
fn summarize_ok(_value: &Value) -> String {
    "Ok".to_string()
}

/// Fallback used for `/v1/chat/completions` on a non-llama.cpp (potentially
/// hosted, potentially billed) provider, where [`probe_chat_completions`] is
/// not run: its availability is inferred from `/v1/models` instead, since any
/// server that speaks the OpenAI protocol well enough to list models is
/// expected to serve chat completions too.
fn chat_completions_capability(models: &Capability, model_id: &str) -> Capability {
    Capability {
        api: "OpenAI",
        endpoint: "/v1/chat/completions",
        available: models.available,
        details: if models.available {
            format!(
                "not probed directly (would trigger a real generation); used for every request to {model_id}"
            )
        } else {
            "server unreachable via /v1/models (see that row above)".to_string()
        },
    }
}

/// `/v1/embeddings` is never actively probed (see [`gather_server_information`]);
/// its row instead reflects the embeddings-capable server orangu already
/// detected at startup for `/search`.
fn embeddings_capability(is_embeddings_server: bool) -> Capability {
    Capability {
        api: "OpenAI",
        endpoint: "/v1/embeddings",
        available: is_embeddings_server,
        details: if is_embeddings_server {
            "detected at startup; used by /search".to_string()
        } else {
            "Not available".to_string()
        },
    }
}

/// Render the gathered capabilities as two aligned tables, styled like
/// `/session`'s listing: a small two-row table naming the server and model,
/// then a header row followed by one row per capability with a green/red
/// status dot, each column sized to its widest value.
pub(crate) fn format_information_table(
    server_name: &str,
    model_id: &str,
    capabilities: &[Capability],
) -> String {
    let w_field = "Server".len().max("Model".len());
    let w_value = server_name.chars().count().max(model_id.chars().count());

    let col_width = |header: &str, value: &dyn Fn(&Capability) -> &str| {
        capabilities
            .iter()
            .map(|c| value(c).chars().count())
            .chain(std::iter::once(header.chars().count()))
            .max()
            .unwrap_or(0)
    };
    let w_api = col_width("API", &|c| c.api);
    let w_endpoint = col_width("ENDPOINT", &|c| c.endpoint);

    let mut lines = vec![
        format!("{:<w_field$}  {:<w_value$}", "Server", server_name),
        format!("{:<w_field$}  {:<w_value$}", "Model", model_id),
        String::new(),
        format!(
            "{:<6}  {:<w_api$}  {:<w_endpoint$}  DETAILS",
            "STATUS", "API", "ENDPOINT"
        ),
    ];
    for capability in capabilities {
        let dot = if capability.available {
            FEEDBACK_OK
        } else {
            FEEDBACK_ERR
        };
        lines.push(format!(
            "{dot}       {:<w_api$}  {:<w_endpoint$}  {}",
            capability.api, capability.endpoint, capability.details
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_details_explains_common_codes() {
        assert_eq!(
            http_status_details(reqwest::StatusCode::NOT_IMPLEMENTED),
            "disabled by the server (missing startup flag)"
        );
        assert_eq!(
            http_status_details(reqwest::StatusCode::NOT_FOUND),
            "not supported by this server"
        );
        assert_eq!(
            http_status_details(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            "HTTP 500"
        );
    }

    #[test]
    fn summarize_props_reads_known_fields_and_skips_missing_ones() {
        let value = serde_json::json!({
            "default_generation_settings": {
                "n_ctx": 32768,
                "n_predict": -1,
                "params": {"temperature": 0.8, "top_k": 40, "top_p": 0.95},
            },
            "total_slots": 4,
            "model_path": "/models/gemma.gguf",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "build_info": "b4200-abc1234",
            "chat_template": "{{ messages }}",
        });
        assert_eq!(
            summarize_props(&value),
            "n_ctx=32768 n_predict=-1 total_slots=4 temperature=0.8 top_k=40 top_p=0.95 \
             model_path=/models/gemma.gguf bos_token=<s> eos_token=</s> build=b4200-abc1234 \
             chat_template=yes"
        );

        let empty = serde_json::json!({});
        assert_eq!(summarize_props(&empty), "chat_template=no");
    }

    #[test]
    fn models_details_drops_the_count_prefix_for_a_single_model() {
        assert_eq!(models_details(&["gemma".to_string()]), "gemma");
    }

    #[test]
    fn models_details_comma_separates_multiple_models() {
        assert_eq!(
            models_details(&["gemma".to_string(), "llama".to_string()]),
            "gemma, llama"
        );
    }

    #[test]
    fn models_details_reports_an_empty_list() {
        assert_eq!(models_details(&[]), "reachable, but advertised no models");
    }

    #[test]
    fn summarize_health_reads_status_field() {
        let value = serde_json::json!({"status": "ok"});
        assert_eq!(summarize_health(&value), "Ok");
        let loading = serde_json::json!({"status": "loading model"});
        assert_eq!(summarize_health(&loading), "Loading model");
        assert_eq!(summarize_health(&serde_json::json!({})), "reachable");
    }

    #[test]
    fn summarize_coordinator_reads_version_field() {
        let value = serde_json::json!({"orangu_coordinator": true, "version": "0.11.0"});
        assert_eq!(summarize_coordinator(&value), "orangu-coordinator v0.11.0");
        assert_eq!(summarize_coordinator(&serde_json::json!({})), "reachable");
    }

    #[test]
    fn summarize_ok_ignores_the_response_body() {
        assert_eq!(summarize_ok(&serde_json::json!({"slots": []})), "Ok");
        assert_eq!(summarize_ok(&serde_json::json!({})), "Ok");
    }

    #[test]
    fn capitalize_first_only_touches_the_first_character() {
        assert_eq!(capitalize_first("ok"), "Ok");
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn simplify_unavailable_flattens_only_unavailable_capabilities() {
        let available = Capability {
            api: "llama.cpp",
            endpoint: "/slots",
            available: true,
            details: "reachable".to_string(),
        };
        let capability = simplify_unavailable(available);
        assert!(capability.available);
        assert_eq!(capability.details, "reachable");

        let unavailable = Capability {
            api: "llama.cpp",
            endpoint: "/slots",
            available: false,
            details: "disabled by the server (missing startup flag)".to_string(),
        };
        let capability = simplify_unavailable(unavailable);
        assert!(!capability.available);
        assert_eq!(capability.details, "Not available");
    }

    #[test]
    fn chat_completions_probe_request_caps_the_response_at_one_token() {
        // The probe must stay a minimal, non-streaming, single-token request —
        // this is a real generation on a local llama.cpp server, so it should
        // never balloon into something that takes real time or output.
        let body = ChatCompletionsProbeRequest {
            model: "gemma",
            messages: [ChatCompletionsProbeMessage {
                role: "user",
                content: "hi",
            }],
            max_tokens: 1,
            stream: false,
        };
        let encoded = serde_json::to_value(&body).expect("serialize probe body");
        assert_eq!(encoded["model"], "gemma");
        assert_eq!(encoded["max_tokens"], 1);
        assert_eq!(encoded["stream"], false);
        assert_eq!(encoded["messages"][0]["role"], "user");
    }

    #[test]
    fn chat_completions_capability_follows_models_reachability() {
        let reachable = Capability {
            api: "OpenAI",
            endpoint: "/v1/models",
            available: true,
            details: String::new(),
        };
        let capability = chat_completions_capability(&reachable, "gemma");
        assert!(capability.available);
        assert!(capability.details.contains("gemma"));

        let unreachable = Capability {
            api: "OpenAI",
            endpoint: "/v1/models",
            available: false,
            details: String::new(),
        };
        let capability = chat_completions_capability(&unreachable, "gemma");
        assert!(!capability.available);
    }

    #[test]
    fn embeddings_capability_reflects_detected_server() {
        let detected = embeddings_capability(true);
        assert!(detected.available);
        let not_detected = embeddings_capability(false);
        assert!(!not_detected.available);
    }

    #[test]
    fn format_information_table_lists_every_capability_aligned() {
        let capabilities = vec![
            Capability {
                api: "OpenAI",
                endpoint: "/v1/models",
                available: true,
                details: "gemma".to_string(),
            },
            Capability {
                api: "llama.cpp",
                endpoint: "/metrics",
                available: false,
                details: "disabled by the server (missing startup flag)".to_string(),
            },
        ];
        let table = format_information_table("main-server", "gemma", &capabilities);
        assert!(table.contains("Server  main-server\nModel   gemma      "));
        assert!(table.contains("STATUS  API        ENDPOINT    DETAILS"));
        assert!(table.contains(&format!("{FEEDBACK_OK}       OpenAI     /v1/models  gemma")));
        assert!(table.contains(&format!(
            "{FEEDBACK_ERR}       llama.cpp  /metrics    disabled by the server (missing startup flag)"
        )));
    }
}

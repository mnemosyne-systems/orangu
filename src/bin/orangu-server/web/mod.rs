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

//! The web UI: a small chat front end (vanilla HTML/CSS/JS, embedded into
//! the binary — no build step) bound to its own `web` port alongside the
//! API's own `port`, sharing the same [`Engine`] so a chat message never
//! makes an HTTP hop to reach it. Chat sessions persist to
//! `~/.orangu/server/sessions/<uuid>.json` (`web::sessions`); each
//! assistant message is rendered from markdown to syntax-highlighted HTML
//! server-side (`web::render`), reusing `markdown`/`syntect` — the same
//! crates `orangu`'s own TUI uses for its terminal rendering.

pub mod render;
pub mod sessions;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::generate::{Engine, GenerateRequest, StreamEvent};
use crate::engine::sampling::SamplingParams;
use sessions::{Session, SessionMessage};

const INDEX_HTML: &str = include_str!("assets/index.html");
const APP_CSS: &str = include_str!("assets/app.css");
const APP_JS: &str = include_str!("assets/app.js");

/// Response-length cap for a web-UI turn — generous for a chat reply
/// without risking one runaway request pinning a slot indefinitely.
const MAX_TOKENS: usize = 1024;

pub struct WebState {
    pub engine: Arc<Engine>,
    pub model_label: String,
    pub version: &'static str,
}

pub fn build_router(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/app.css", get(app_css))
        .route("/static/app.js", get(app_js))
        .route("/api/sessions", post(create_session).get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}/messages", post(send_message))
        .with_state(state)
}

async fn index(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let html = INDEX_HTML
        .replace("{{VERSION}}", state.version)
        .replace("{{MODEL}}", &html_escape(&state.model_label))
        .replace("{{YEAR}}", &current_year().to_string());
    Html(html)
}

/// The current UTC calendar year, for the footer's copyright-style link —
/// computed from the Unix clock rather than pulling in a full date/time
/// crate for one integer.
fn current_year() -> i64 {
    let mut days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        / 86400;
    let mut year = 1970i64;
    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in_year = if is_leap { 366 } else { 365 };
        if days < days_in_year {
            return year;
        }
        days -= days_in_year;
        year += 1;
    }
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

async fn app_css() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "text/css; charset=utf-8")],
        APP_CSS,
    )
}

async fn app_js() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "application/javascript; charset=utf-8")],
        APP_JS,
    )
}

async fn create_session() -> impl IntoResponse {
    match sessions::create_session() {
        Ok(session) => Json(session).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

async fn list_sessions() -> impl IntoResponse {
    match sessions::list_sessions() {
        Ok(list) => Json(list).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

#[derive(Serialize)]
struct SessionMessageView {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    html: Option<String>,
}

#[derive(Serialize)]
struct SessionView {
    id: String,
    created_at: u64,
    updated_at: u64,
    title: String,
    messages: Vec<SessionMessageView>,
}

async fn get_session(Path(id): Path<String>) -> impl IntoResponse {
    match sessions::load_session(&id) {
        Ok(session) => Json(SessionView {
            id: session.id,
            created_at: session.created_at,
            updated_at: session.updated_at,
            title: session.title,
            messages: session
                .messages
                .into_iter()
                .map(|m| {
                    let html = (m.role == "assistant")
                        .then(|| render::render_markdown_to_html(&m.content));
                    SessionMessageView {
                        role: m.role,
                        content: m.content,
                        html,
                    }
                })
                .collect(),
        })
        .into_response(),
        Err(err) => (StatusCode::NOT_FOUND, err.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct SendMessageRequest {
    content: String,
}

async fn send_message(
    State(state): State<Arc<WebState>>,
    Path(id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> axum::response::Response {
    let mut session = match sessions::load_session(&id) {
        Ok(session) => session,
        Err(err) => return (StatusCode::NOT_FOUND, err.to_string()).into_response(),
    };
    let Some(template_source) = state.engine.chat_template_source.clone() else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "model has no tokenizer.chat_template; the web UI needs one",
        )
            .into_response();
    };

    let prompt = match render_prompt(&state, &template_source, &session, &req.content) {
        Ok(prompt) => prompt,
        // {err:#} (anyhow's "alternate" chain format) rather than {err} —
        // the latter only prints the outermost context, losing exactly the
        // detail (which template line, which variable) that makes a
        // template-rendering error diagnosable at all.
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{err:#}")).into_response(),
    };
    let tokens = state.engine.tokenizer.encode(&prompt, false);
    let stop_token_ids = state.engine.tokenizer.eos_token.into_iter().collect();

    let mut rx = state
        .engine
        .generate(GenerateRequest {
            prompt_tokens: tokens,
            sampling: SamplingParams::default(),
            max_tokens: MAX_TOKENS,
            stop_token_ids,
        })
        .await;

    let user_message = req.content;
    let stream = async_stream::stream! {
        let mut full = String::new();
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                StreamEvent::Token(text) => {
                    full.push_str(&text);
                    yield Ok::<_, Infallible>(
                        axum::response::sse::Event::default()
                            .data(json!({"type": "token", "text": text}).to_string()),
                    );
                }
                StreamEvent::Done { .. } => {
                    let full = state.engine.tokenizer.clean_up_tokenization_spaces(&full);
                    let html = render::render_markdown_to_html(&full);
                    if let Err(err) = sessions::append_turn(&mut session, &user_message, &full) {
                        yield Ok(axum::response::sse::Event::default()
                            .data(json!({"type": "error", "message": err.to_string()}).to_string()));
                        break;
                    }
                    yield Ok(axum::response::sse::Event::default()
                        .data(json!({"type": "done", "html": html}).to_string()));
                    break;
                }
                StreamEvent::Error(err) => {
                    yield Ok(axum::response::sse::Event::default()
                        .data(json!({"type": "error", "message": err}).to_string()));
                    break;
                }
            }
        }
    };
    axum::response::sse::Sse::new(stream).into_response()
}

fn render_prompt(
    state: &WebState,
    template_source: &str,
    session: &Session,
    new_message: &str,
) -> anyhow::Result<String> {
    let mut messages: Vec<ChatMessage> = session
        .messages
        .iter()
        .map(|m: &SessionMessage| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: new_message.to_string(),
    });

    let bos = state
        .engine
        .tokenizer
        .bos_token
        .and_then(|id| state.engine.tokenizer.token_text(id))
        .unwrap_or("");
    let eos = state
        .engine
        .tokenizer
        .eos_token
        .and_then(|id| state.engine.tokenizer.token_text(id))
        .unwrap_or("");

    ChatTemplate::new(template_source.to_string()).render(&messages, true, bos, eos, None)
}

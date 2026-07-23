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

pub mod attachments;
pub mod render;
pub mod sessions;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::hash_map::DefaultHasher,
    convert::Infallible,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::generate::{Engine, FinishReason, GenerateRequest, StreamEvent};
use crate::engine::sampling::SamplingParams;
use sessions::{Attachment, Session, SessionMessage};

const INDEX_HTML: &str = include_str!("assets/index.html");
const APP_CSS: &str = include_str!("assets/app.css");
const APP_JS: &str = include_str!("assets/app.js");

/// KaTeX (MIT, `assets/katex/LICENSE`) — vendored rather than pulled from a
/// CDN, matching this whole web UI's "no build step, no network
/// dependency" shape (see this module's own doc comment): the server has
/// to keep rendering chat math correctly on a machine with no internet
/// access at all. `web::render` emits `<span class="katex-source"
/// data-tex="...">`/`<div class="katex-source katex-display" ...>`
/// placeholders for `$...$`/`$$...$$` math; `app.js` finds them after each
/// render and calls `katex.render` client-side. Only the `.woff2` font
/// variant is bundled (universal in any browser capable of running this
/// page at all) — `katex.min.css`'s own `@font-face` rules still list
/// `.woff`/`.ttf` fallbacks, which simply 404 through [`katex_font`] on a
/// browser that never asks for them.
const KATEX_JS: &str = include_str!("assets/katex/katex.min.js");
const KATEX_CSS: &str = include_str!("assets/katex/katex.min.css");
const KATEX_FONTS: &[(&str, &[u8])] = &[
    (
        "KaTeX_AMS-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_AMS-Regular.woff2"),
    ),
    (
        "KaTeX_Caligraphic-Bold.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Caligraphic-Bold.woff2"),
    ),
    (
        "KaTeX_Caligraphic-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Caligraphic-Regular.woff2"),
    ),
    (
        "KaTeX_Fraktur-Bold.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Fraktur-Bold.woff2"),
    ),
    (
        "KaTeX_Fraktur-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Fraktur-Regular.woff2"),
    ),
    (
        "KaTeX_Main-Bold.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Main-Bold.woff2"),
    ),
    (
        "KaTeX_Main-BoldItalic.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Main-BoldItalic.woff2"),
    ),
    (
        "KaTeX_Main-Italic.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Main-Italic.woff2"),
    ),
    (
        "KaTeX_Main-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Main-Regular.woff2"),
    ),
    (
        "KaTeX_Math-BoldItalic.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Math-BoldItalic.woff2"),
    ),
    (
        "KaTeX_Math-Italic.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Math-Italic.woff2"),
    ),
    (
        "KaTeX_SansSerif-Bold.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_SansSerif-Bold.woff2"),
    ),
    (
        "KaTeX_SansSerif-Italic.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_SansSerif-Italic.woff2"),
    ),
    (
        "KaTeX_SansSerif-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_SansSerif-Regular.woff2"),
    ),
    (
        "KaTeX_Script-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Script-Regular.woff2"),
    ),
    (
        "KaTeX_Size1-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Size1-Regular.woff2"),
    ),
    (
        "KaTeX_Size2-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Size2-Regular.woff2"),
    ),
    (
        "KaTeX_Size3-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Size3-Regular.woff2"),
    ),
    (
        "KaTeX_Size4-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Size4-Regular.woff2"),
    ),
    (
        "KaTeX_Typewriter-Regular.woff2",
        include_bytes!("assets/katex/fonts/KaTeX_Typewriter-Regular.woff2"),
    ),
];

/// Response-length cap for a web-UI turn — generous for a chat reply
/// (a full worked example, e.g. a from-scratch data-structure
/// implementation, easily runs past 1024 tokens) without risking one
/// runaway request pinning a slot indefinitely. The engine additionally
/// clamps this to what's left of the model's context window, so raising
/// it here never risks overrunning `n_ctx_train`.
const MAX_TOKENS: usize = 8192;

pub struct WebState {
    pub engine: Arc<Engine>,
    pub model_label: String,
    /// Echoed into `GET /api/system-report`'s debug report (`app.js`'s
    /// error-bubble Save button) alongside `model_label`/`version` — the
    /// same detail `serve`'s own startup banner prints, not otherwise
    /// available to the web UI at all.
    pub architecture: String,
    pub backend_label: String,
    /// The root directory this server operates in (`-w`/`--workspace`, or
    /// the current working directory) — echoed into the same debug report as
    /// `architecture`/`backend_label`, so a saved report says which tree the
    /// server was rooted at.
    pub workspace: PathBuf,
    pub version: &'static str,
}

pub fn build_router(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/app.css", get(app_css))
        .route("/static/app.js", get(app_js))
        .route("/static/katex/katex.min.css", get(katex_css))
        .route("/static/katex/katex.min.js", get(katex_js))
        .route("/static/katex/fonts/{name}", get(katex_font))
        .route("/api/asset-version", get(asset_version_handler))
        .route("/api/system-report", get(system_report))
        .route("/api/sessions", post(create_session).get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}/messages", post(send_message))
        // Attachments ride along as base64 in the message JSON, so the
        // default 2 MB body cap is far too small — allow room for a handful
        // of documents (base64 inflates bytes by ~4/3).
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

async fn index(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let html = INDEX_HTML
        .replace("{{VERSION}}", state.version)
        .replace("{{MODEL}}", &html_escape(&state.model_label))
        .replace("{{YEAR}}", &current_year().to_string())
        .replace("{{ASSET_VERSION}}", asset_version());
    Html(html)
}

/// A stable fingerprint of the embedded web assets — same input, same
/// hash, across every request in this process and across separate
/// processes built from identical sources; changes only when
/// `index.html`/`app.css`/`app.js` actually change. The client compares
/// this against the version it was served at load time (`web::index`) to
/// notice a newer binary is now running behind it (see `/api/asset-version`
/// and the Reload button in `app.js`).
fn asset_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let mut hasher = DefaultHasher::new();
        INDEX_HTML.hash(&mut hasher);
        APP_CSS.hash(&mut hasher);
        APP_JS.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    })
}

async fn asset_version_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Cache-Control", "no-cache")],
        Json(json!({ "version": asset_version() })),
    )
}

/// The model/backend identity plus a fresh hardware snapshot (`orangu-
/// server system`'s own report, reused verbatim via `orangu::hardware::
/// format_report` rather than duplicated) — the "what machine, what
/// model" half of the web UI's error-bubble debug report (`app.js`'s Save
/// button); the conversation and error-detail halves are assembled
/// client-side, from data the browser already has. Detected fresh on
/// every call (not cached at startup) since the parts that actually
/// change over a long-running process's lifetime — VRAM/RAM currently in
/// use — are exactly the parts most useful to know at the moment a
/// request just failed, not at server startup.
async fn system_report(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let cpu = orangu::hardware::detect_cpu();
    let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
    let mut report = format!(
        "orangu-server {}\nModel        {}\nArchitecture {}\nBackend      {}\nWorkspace    {}\n\n",
        state.version,
        state.model_label,
        state.architecture,
        state.backend_label,
        state.workspace.display(),
    );
    report.push_str(&orangu::hardware::format_report(&cpu, &gpus));
    (
        StatusCode::OK,
        [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Cache-Control", "no-store"),
        ],
        report,
    )
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
        [
            ("Content-Type", "text/css; charset=utf-8"),
            ("Cache-Control", "no-cache"),
        ],
        APP_CSS,
    )
}

async fn app_js() -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            ("Content-Type", "application/javascript; charset=utf-8"),
            ("Cache-Control", "no-cache"),
        ],
        APP_JS,
    )
}

// KaTeX is a vendored, version-pinned third-party asset (see `KATEX_JS`'s
// own doc comment) rather than something this project edits — unlike
// `app_css`/`app_js` above, it's cached aggressively (`immutable`, a full
// year) instead of `no-cache`, since it only ever changes when a human
// bumps the vendored copy in a new `orangu-server` build, not between
// requests to the same one.
async fn katex_css() -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            ("Content-Type", "text/css; charset=utf-8"),
            ("Cache-Control", "public, max-age=31536000, immutable"),
        ],
        KATEX_CSS,
    )
}

async fn katex_js() -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            ("Content-Type", "application/javascript; charset=utf-8"),
            ("Cache-Control", "public, max-age=31536000, immutable"),
        ],
        KATEX_JS,
    )
}

/// Serves one embedded KaTeX font by exact filename match against
/// [`KATEX_FONTS`] — an allowlist lookup, not a filesystem read, so an
/// unexpected `name` (typo, path-traversal attempt) can only ever produce
/// a 404, never touch disk.
async fn katex_font(Path(name): Path<String>) -> impl IntoResponse {
    match KATEX_FONTS.iter().find(|(font_name, _)| *font_name == name) {
        Some((_, bytes)) => (
            StatusCode::OK,
            [
                ("Content-Type", "font/woff2"),
                ("Cache-Control", "public, max-age=31536000, immutable"),
            ],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
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
struct AttachmentView {
    name: String,
    mime: String,
    size: u64,
}

#[derive(Serialize)]
struct SessionMessageView {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_ms: Option<u64>,
    // Metadata only — the extracted text stays server-side; the UI just
    // needs to render a chip per file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    attachments: Vec<AttachmentView>,
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
                        generation_ms: m.generation_ms,
                        attachments: m
                            .attachments
                            .into_iter()
                            .map(|a| AttachmentView {
                                name: a.name,
                                mime: a.mime,
                                size: a.size,
                            })
                            .collect(),
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
    #[serde(default)]
    attachments: Vec<attachments::IncomingAttachment>,
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

    // Decode + extract text from every attachment up front, so the prompt
    // and the persisted turn share exactly the same view of them.
    let mut extracted: Vec<Attachment> = Vec::with_capacity(req.attachments.len());
    for incoming in &req.attachments {
        match attachments::extract(incoming) {
            Ok(att) => extracted.push(att),
            Err(err) => return (StatusCode::BAD_REQUEST, format!("{err:#}")).into_response(),
        }
    }

    if req.content.trim().is_empty() && extracted.is_empty() {
        return (StatusCode::BAD_REQUEST, "message is empty").into_response();
    }

    let prompt = match render_prompt(&state, &template_source, &session, &req.content, &extracted) {
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
    let user_attachments = extracted;
    let stream = async_stream::stream! {
        let mut full = String::new();
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                StreamEvent::Token(text) => {
                    full.push_str(&text);
                    let html = render::render_markdown_to_html(&full);
                    yield Ok::<_, Infallible>(
                        axum::response::sse::Event::default()
                            .data(json!({"type": "token", "html": html}).to_string()),
                    );
                }
                StreamEvent::Done { finish_reason, stats } => {
                    let full = state.engine.tokenizer.clean_up_tokenization_spaces(&full);
                    let html = render::render_markdown_to_html(&full);
                    let generation_ms = stats.generate_time.as_millis() as u64;
                    if let Err(err) = sessions::append_turn(&mut session, &user_message, user_attachments, &full, Some(generation_ms)) {
                        yield Ok(axum::response::sse::Event::default()
                            .data(json!({"type": "error", "message": err.to_string()}).to_string()));
                        break;
                    }
                    let truncated = finish_reason == FinishReason::Length;
                    yield Ok(axum::response::sse::Event::default()
                        .data(json!({"type": "done", "html": html, "content": full, "truncated": truncated, "generation_ms": generation_ms}).to_string()));
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
    new_attachments: &[Attachment],
) -> anyhow::Result<String> {
    // Each message's stored attachments (with their extracted text) are
    // folded back into its content here rather than being persisted inline,
    // so the transcript stays clean while the model still sees the document
    // — on this turn and on every follow-up turn that replays the history.
    let mut messages: Vec<ChatMessage> = session
        .messages
        .iter()
        .map(|m: &SessionMessage| ChatMessage {
            role: m.role.clone(),
            content: attachments::compose_content(&m.content, &m.attachments),
        })
        .collect();
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: attachments::compose_content(new_message, new_attachments),
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

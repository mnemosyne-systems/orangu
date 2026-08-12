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
//! crates `orangu`'s own TUI uses for its terminal rendering — with
//! ```` ```mermaid ```` and ```` ```plantuml ```` blocks drawn locally by
//! `web::mermaid` and `web::plantuml`.

pub mod attachments;
pub mod mcp;
pub mod mermaid;
pub mod models;
pub mod plantuml;
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
    /// The model as named to a human — `MODEL:QUANT` (see `serve`) — for the
    /// topbar and the debug report. Deliberately *not* the API's model id:
    /// nothing the web UI sends back is keyed by it, both uses here are
    /// display only.
    pub model_display: String,
    /// Echoed into `GET /api/system-report`'s debug report (`app.js`'s
    /// error-bubble Save button) alongside `model_display`/`version` — the
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
    /// The `.gguf` this server loaded, and the directory the model manager
    /// lists — see `web::models`, and `Prepared`'s own fields of the same
    /// names for why the manager needs each.
    pub model_path: PathBuf,
    pub models_dir: PathBuf,
    /// The model manager's one background-download slot, and the cached
    /// models-directory scan its listing is served from.
    pub jobs: Arc<models::ModelJobs>,
    pub catalog: Arc<models::ModelCatalog>,
    /// What the model manager's **Load** button needs to replace this
    /// process with one serving a different model — see `crate::reexec`.
    /// `None` when `[orangu-server].reexec` is off or the platform has no
    /// `execve`, which is how the button knows to disable itself instead of
    /// offering something that would only refuse.
    pub handover: Option<Arc<crate::reexec::Handover>>,
    /// `[web].delete`: whether the model manager may delete models. When
    /// false the panel draws no Delete button at all — unlike **Load**,
    /// which is drawn disabled with a tooltip, because there is nothing
    /// conditional here to explain: this server simply doesn't do that.
    ///
    /// Models only. History's own delete controls are unconditional — see
    /// [`delete_session`]/[`clear_sessions`].
    pub can_delete: bool,
    /// Whether the served model is embedded in this executable (see
    /// `crate::bundle`). It has no row in the model manager's listing and no
    /// Delete button: a bundled model cannot be removed from a running
    /// server — only the whole binary can be. `models::CurrentView` reports
    /// it so the panel can say that rather than leave an unexplained gap.
    pub bundled: bool,
    /// The model a handover has been accepted for, once one has — see
    /// `models::select`. Only ever goes from empty to set: this process is
    /// about to be replaced, so there is nothing to reset it back to.
    pub loading: std::sync::Mutex<Option<String>>,
    /// Configured MCP profiles are displayed read-only; applying config edits
    /// requires a server restart.
    pub mcp_servers: Vec<crate::config::McpConfiguration>,
}

impl WebState {
    /// Claims this process's one handover for `label`, or `false` when
    /// something already claimed it. One per process: there is only one
    /// process to replace, and two racing `execve`s have no useful meaning.
    pub fn arm_handover(&self, label: &str) -> bool {
        let mut loading = self.loading.lock().unwrap();
        if loading.is_some() {
            return false;
        }
        *loading = Some(label.to_string());
        true
    }

    /// The model this process is about to be replaced by, if any.
    pub fn loading_model(&self) -> Option<String> {
        self.loading.lock().unwrap().clone()
    }
}

pub fn build_router(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/app.css", get(app_css))
        .route("/static/app.js", get(app_js))
        .route("/static/katex/katex.min.css", get(katex_css))
        .route("/static/katex/katex.min.js", get(katex_js))
        .route("/static/katex/fonts/{name}", get(katex_font))
        .route("/api/diagrams/{key}/{name}", get(diagram_asset))
        .route("/api/asset-version", get(asset_version_handler))
        .route("/api/system-report", get(system_report))
        // `delete` on both: one row's cross, and History's **Clear all**
        // footer. Unconditional — unlike the model manager's own Delete
        // (`[web].delete`), which owns files on disk that nothing else put
        // there. A chat session is this console's own scratch data, and
        // being unable to tidy up your own transcripts is not a deployment
        // posture anyone asked for.
        .route(
            "/api/sessions",
            post(create_session)
                .get(list_sessions)
                .delete(clear_sessions),
        )
        .route(
            "/api/sessions/{id}",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/{id}/messages", post(send_message))
        .merge(mcp::router())
        // The model manager: list, metadata, download, delete — on the same
        // port as the chat UI.
        .merge(models::router())
        // Attachments ride along as base64 in the message JSON, so the
        // default 2 MB body cap is far too small — allow room for a handful
        // of documents (base64 inflates bytes by ~4/3).
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

/// Delivers a PlantUML render from the in-memory renderer cache.  Keeping
/// diagram bytes off the streamed message payload is particularly important
/// for PNGs, whose base64 representation grows every JSON event by about a
/// third.
async fn diagram_asset(Path((key, name)): Path<(String, String)>) -> impl IntoResponse {
    match plantuml::asset(&key, &name) {
        Some((content_type, bytes)) => (
            [
                ("content-type", content_type),
                ("cache-control", "private, max-age=3600"),
            ],
            bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn index(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let html = INDEX_HTML
        .replace("{{VERSION}}", state.version)
        .replace("{{MODEL}}", &html_escape(&state.model_display))
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
    let os = orangu::os::detect();
    let cpu = orangu::hardware::detect_cpu();
    let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
    let mut report = format!(
        "orangu-server {}\nModel        {}\nArchitecture {}\nBackend      {}\nWorkspace    {}\n\n",
        state.version,
        state.model_display,
        state.architecture,
        state.backend_label,
        state.workspace.display(),
    );
    report.push_str(&orangu::hardware::format_report(&os, &cpu, &gpus));
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

/// Removes one chat session, directory and all — History's per-row cross.
/// A missing or malformed id is a 404 rather than a 500: the row it came
/// from is stale either way, and the browser's answer to both is the same
/// (re-list, and the row is gone).
async fn delete_session(Path(id): Path<String>) -> impl IntoResponse {
    match sessions::delete_session_dir(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => (StatusCode::NOT_FOUND, format!("{err:#}")).into_response(),
    }
}

/// Removes every chat session — History's **Clear all** footer. Reports how
/// many went, so the browser can say so; see
/// [`sessions::delete_all_sessions`] for why the caller's own current
/// session is not spared.
async fn clear_sessions() -> impl IntoResponse {
    match sessions::delete_all_sessions() {
        Ok(removed) => Json(json!({ "removed": removed })).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response(),
    }
}

#[derive(Serialize)]
struct AttachmentView {
    name: String,
    mime: String,
    size: u64,
    /// What the server managed to read out of the file — exactly the text
    /// handed to the model, so the panel shows what was actually sent rather
    /// than a re-derived approximation. `None` for a format nothing could be
    /// extracted from (an unrecognised or binary type), which is the signal
    /// the UI uses to leave out the expand control entirely: there is
    /// nothing behind it.
    ///
    /// Already bounded by `attachments::MAX_TEXT_CHARS`, which marks its own
    /// truncation inline, so this adds no undisclosed cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    /// Mermaid diagrams found inside the file, already drawn. An attachment
    /// is otherwise invisible to the reader — its text goes to the model and
    /// the UI shows only a chip — so a diagram someone attached would
    /// otherwise be the one thing they can't see in their own message.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    diagrams: Vec<DiagramView>,
    /// Set when the file held more diagrams than [`mermaid::MAX_PER_ATTACHMENT`],
    /// so the UI can say so rather than quietly showing a prefix.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    diagrams_capped: bool,
}

/// One drawn diagram as the browser needs it: the two theme variants (see
/// [`mermaid`] for why there are two) and the source it came from.
#[derive(Serialize)]
struct DiagramView {
    kind: &'static str,
    light: String,
    dark: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    light_png: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dark_png: Option<String>,
    source: String,
    /// Natural size for the `<img>` — see [`mermaid::Diagram`] for why an
    /// unsized diagram reflows the transcript when it decodes.
    width: f64,
    height: f64,
}

/// Builds one attachment's view: what was read out of it, plus every
/// diagram drawn from it.
///
/// Called on session load and once per send, never per token — attachment
/// text doesn't change while a reply streams.
fn attachment_view(attachment: sessions::Attachment) -> AttachmentView {
    let mut diagrams = Vec::new();
    let mut diagrams_found = 0;
    if let Some(text) = attachment.text.as_deref() {
        let found = mermaid::find_in_text(text);
        diagrams_found += found.len();
        diagrams.extend(found.into_iter().map(|found| {
            (
                found.offset,
                DiagramView {
                    kind: "mermaid",
                    light: found.diagram.light.clone(),
                    dark: found.diagram.dark.clone(),
                    light_png: None,
                    dark_png: None,
                    width: found.diagram.width,
                    height: found.diagram.height,
                    source: found.source,
                },
            )
        }));
        let found = plantuml::find_in_text(text);
        diagrams_found += found.len();
        diagrams.extend(found.into_iter().map(|found| {
            (
                found.offset,
                DiagramView {
                    kind: "plantuml",
                    light: found.diagram.light.clone(),
                    dark: found.diagram.dark.clone(),
                    light_png: Some(found.diagram.light_png.clone()),
                    dark_png: Some(found.diagram.dark_png.clone()),
                    width: found.diagram.width,
                    height: found.diagram.height,
                    source: found.source,
                },
            )
        }));
    }
    // Each parser walks only its own family, so sort the combined result by
    // fence position before applying the shared attachment cap.
    diagrams.sort_by_key(|(offset, _)| *offset);
    diagrams.truncate(mermaid::MAX_PER_ATTACHMENT);
    AttachmentView {
        name: attachment.name,
        mime: attachment.mime,
        size: attachment.size,
        text: attachment.text,
        diagrams_capped: diagrams_found > mermaid::MAX_PER_ATTACHMENT,
        diagrams: diagrams.into_iter().map(|(_, diagram)| diagram).collect(),
    }
}

#[derive(Serialize)]
struct SessionMessageView {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_ms: Option<u64>,
    // Name/type/size for the chip, plus any diagrams drawn out of the file.
    // The extracted text itself stays server-side.
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
                        attachments: m.attachments.into_iter().map(attachment_view).collect(),
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
    let stop_token_ids = state.engine.tokenizer.stop_token_ids();

    let mut rx = state
        .engine
        .generate(GenerateRequest {
            prompt_tokens: tokens,
            sampling: SamplingParams::default(),
            max_tokens: MAX_TOKENS,
            stop_token_ids,
            cache_prompt: true,
            // The web console keeps one conversation per session but has no
            // notion of a slot; any free one is right.
            id_slot: None,
            timings_per_token: false,
        })
        .await;

    let user_message = req.content;
    let user_attachments = extracted;
    // What this turn's uploads turned into — extracted text and any drawn
    // diagrams — built once here rather than per token. Sent before the
    // first token so the user's own message can show it while the reply is
    // still generating; on a later visit `get_session` rebuilds the same
    // views, so a reload isn't what makes it appear. The browser only ever
    // had the raw bytes, so this is its first sight of what was read.
    let attachment_views: Vec<AttachmentView> = user_attachments
        .iter()
        .cloned()
        .map(attachment_view)
        .collect();

    let stream = async_stream::stream! {
        if !attachment_views.is_empty() {
            yield Ok::<_, Infallible>(
                axum::response::sse::Event::default()
                    .data(json!({"type": "attachments", "attachments": attachment_views}).to_string()),
            );
        }
        let mut full = String::new();
        loop {
            let Some(event) = rx.recv().await else { break };
            match event {
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
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
        .map(|m: &SessionMessage| {
            ChatMessage::text(
                &m.role,
                &attachments::compose_content(&m.content, &m.attachments),
            )
        })
        .collect();
    messages.push(ChatMessage::text(
        "user",
        &attachments::compose_content(new_message, new_attachments),
    ));

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

    // Reasoning is governed by the server's *role* (a CLI choice on
    // `orangu-server`), not by the web console — so this mirrors
    // `http::openai::chat_completions` exactly: pass the role's
    // `enable_thinking` to the template, and for a reasoning-suppressing
    // role (`Role::Review`) also pre-fill an empty, already-closed think
    // block for templates that don't honor the flag. A reasoning-enabled
    // role still thinks; the raw `<think>`/`</think>` framing tokens never
    // reach the stream regardless (the generate loop drops special tokens),
    // so a suppressing role gives a direct answer and a thinking role shows
    // its reasoning prose without the literal tags.
    let mut prompt = ChatTemplate::new(template_source.to_string()).render(
        &messages,
        true,
        bos,
        eos,
        state.engine.role.enable_thinking(),
    )?;
    crate::http::openai::append_reasoning_suppression(
        &mut prompt,
        state.engine.role,
        &state.engine.tokenizer,
    );
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn upload(name: &str, mime: &str, body: &str) -> AttachmentView {
        let incoming = attachments::IncomingAttachment {
            name: name.to_string(),
            mime: mime.to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(body.as_bytes()),
        };
        attachment_view(attachments::extract(&incoming).expect("extracts"))
    }

    #[test]
    fn a_mermaid_file_upload_arrives_as_a_drawn_diagram() {
        // The whole path an attachment takes: browser bytes -> base64
        // decode -> text extraction -> diagram detection -> rendered view.
        // A `.mmd` has no fence around it, so this only works because the
        // header gate recognises a bare diagram.
        let view = upload(
            "architecture.mmd",
            "",
            "flowchart TD\n    A[Start] --> B[Done]\n",
        );
        assert_eq!(view.diagrams.len(), 1);
        assert!(
            view.diagrams[0]
                .light
                .starts_with("data:image/svg+xml;base64,")
        );
        assert!(view.diagrams[0].source.contains("flowchart TD"));
        assert!(!view.diagrams_capped);
    }

    #[test]
    fn a_markdown_upload_yields_the_diagrams_inside_it() {
        let view = upload(
            "design.md",
            "text/markdown",
            "# Design\n\nProse.\n\n```mermaid\nsequenceDiagram\n    A->>B: hi\n```\n",
        );
        assert_eq!(view.diagrams.len(), 1);
        assert!(view.diagrams[0].source.contains("sequenceDiagram"));
    }

    #[test]
    fn a_plantuml_upload_includes_svg_png_and_kind() {
        let view = upload(
            "login.puml",
            "",
            "@startuml\nAlice -> Bob: Login\nBob --> Alice: OK\n@enduml\n",
        );
        assert_eq!(view.diagrams.len(), 1);
        let diagram = &view.diagrams[0];
        assert_eq!(diagram.kind, "plantuml");
        assert!(diagram.light.starts_with("/api/diagrams/"));
        assert!(
            diagram
                .light_png
                .as_deref()
                .is_some_and(|png| png.ends_with("/light.png"))
        );
    }

    #[test]
    fn attachment_diagrams_keep_their_source_order_across_families() {
        let view = upload(
            "mixed.md",
            "text/markdown",
            "```plantuml\n@startuml\nA -> B: first\n@enduml\n```\n\n```mermaid\nflowchart LR\n  C --> D\n```\n\n```puml\n@startuml\nE -> F: third\n@enduml\n```\n",
        );
        assert_eq!(view.diagrams.len(), 3);
        assert_eq!(view.diagrams[0].kind, "plantuml");
        assert_eq!(view.diagrams[1].kind, "mermaid");
        assert_eq!(view.diagrams[2].kind, "plantuml");
    }

    #[test]
    fn an_ordinary_upload_yields_no_diagrams() {
        // The common case, and the one a false positive would ruin: this
        // must stay an ordinary chip with nothing drawn under it.
        let view = upload("main.rs", "", "fn main() {\n    println!(\"hi\");\n}\n");
        assert!(view.diagrams.is_empty());
        assert_eq!(view.name, "main.rs");
    }

    #[test]
    fn a_binary_upload_offers_nothing_to_expand() {
        // `text: None` and no diagrams is what tells the UI to render a
        // plain chip with no expand control — there would be nothing behind
        // it. A format we can't read must not advertise otherwise.
        let incoming = attachments::IncomingAttachment {
            name: "blob.bin".into(),
            mime: "application/octet-stream".into(),
            data: base64::engine::general_purpose::STANDARD.encode([0u8, 1, 2, 255]),
        };
        let view = attachment_view(attachments::extract(&incoming).unwrap());
        assert!(view.text.is_none());
        assert!(view.diagrams.is_empty());
    }

    #[test]
    fn a_readable_upload_carries_its_text_for_the_expand_panel() {
        // The counterpart: anything we could read must send back exactly
        // what the model got, so the panel shows what was actually sent.
        let view = upload("notes.txt", "text/plain", "hello there");
        assert_eq!(view.text.as_deref(), Some("hello there"));
    }
}

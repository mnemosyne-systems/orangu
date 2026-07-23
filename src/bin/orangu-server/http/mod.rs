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

//! Router assembly and shared state. Endpoint handlers live in
//! `http::openai` (OpenAI-compatible) and `http::native`
//! (llama.cpp-native); shutdown is handled here since it's neither.

pub mod native;
pub mod openai;

use crate::engine::generate::Engine;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::mpsc;

pub struct AppState {
    pub engine: Arc<Engine>,
    /// What `general.name`/the resolved model spec reports as the model's
    /// "id" in `/v1/models` and `/props` — not necessarily a real file path,
    /// so a client can display it directly.
    pub model_label: String,
    /// The root directory this server operates in (`-w`/`--workspace`, or
    /// the current working directory). Reported by `/props` so a client can
    /// see which tree it is talking to.
    pub workspace: PathBuf,
    pub started_at: Instant,
    pub shutdown_tx: mpsc::Sender<()>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(native::health))
        .route("/props", get(native::props))
        .route("/slots", get(native::slots))
        .route("/slots/{id_slot}", post(native::slot_action))
        .route("/metrics", get(native::metrics))
        .route("/tokenize", post(native::tokenize))
        .route("/detokenize", post(native::detokenize))
        .route("/completion", post(native::completion))
        .route("/embedding", post(native::embedding))
        .route("/apply-template", post(native::apply_template))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/shutdown", post(shutdown))
        .with_state(state)
}

/// Loopback-only, like `orangu-coordinator`'s own shutdown endpoint — a
/// server bound to a non-loopback `host` must not let an arbitrary network
/// peer kill it with an unauthenticated POST.
async fn shutdown(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !addr.ip().is_loopback() {
        return (
            StatusCode::FORBIDDEN,
            "shutdown is only available from localhost\n",
        );
    }
    let _ = state.shutdown_tx.send(()).await;
    (StatusCode::OK, "shutting down\n")
}

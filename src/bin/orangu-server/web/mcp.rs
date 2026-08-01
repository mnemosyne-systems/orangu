// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Read-only MCP inventory for the web console. Configuration belongs to the
//! server process, so edits take effect only after a restart.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use serde::Serialize;
use std::sync::Arc;

use super::WebState;

pub fn router() -> Router<Arc<WebState>> {
    Router::new()
        .route("/api/mcps", get(list))
        .route("/api/mcps/{name}", get(show))
}

#[derive(Serialize)]
struct McpView {
    name: String,
    endpoint: String,
    enabled: bool,
    approval_mode: String,
}

fn view(mcp: &crate::config::McpConfiguration) -> McpView {
    McpView {
        name: mcp.name.clone(),
        endpoint: mcp.endpoint.clone(),
        enabled: mcp.enabled,
        approval_mode: mcp.approval_mode.clone(),
    }
}

async fn list(State(state): State<Arc<WebState>>) -> Json<Vec<McpView>> {
    Json(state.mcp_servers.iter().map(view).collect())
}

async fn show(State(state): State<Arc<WebState>>, Path(name): Path<String>) -> impl IntoResponse {
    match state.mcp_servers.iter().find(|mcp| mcp.name == name) {
        Some(mcp) => Json(view(mcp)).into_response(),
        None => (StatusCode::NOT_FOUND, "MCP server not found").into_response(),
    }
}

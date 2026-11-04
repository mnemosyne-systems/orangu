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

//! Helpers shared by tests: a stand-in for a model server.

use std::io::{Read, Write};
use std::net::TcpListener;

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// A chat-completion stream whose only content is `text`, finished.
pub(crate) fn text_stream(text: &str) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
    )
}

/// Answers each chat request with the next of `bodies` (SSE) and any
/// other request (a slot probe) with 404, until every body is served;
/// returns the chat request bodies.
pub(crate) fn serve_chat_responses(
    listener: TcpListener,
    bodies: Vec<String>,
) -> std::thread::JoinHandle<Vec<String>> {
    serve_gated_chat_responses(listener, bodies, None)
}

/// Like [`serve_chat_responses`], but each chat answer waits for a message
/// on `gate` first, so a test can act while the request is in flight.
pub(crate) fn serve_gated_chat_responses(
    listener: TcpListener,
    bodies: Vec<String>,
    gate: Option<std::sync::mpsc::Receiver<()>>,
) -> std::thread::JoinHandle<Vec<String>> {
    std::thread::spawn(move || {
        let mut served = Vec::new();
        let mut bodies = bodies.into_iter();
        let mut next = bodies.next();
        while let Some(body) = &next {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            let header_end = loop {
                let read = stream.read(&mut buffer).expect("read request");
                if read == 0 {
                    break None;
                }
                request.extend_from_slice(&buffer[..read]);
                if let Some(position) = find_subsequence(&request, b"\r\n\r\n") {
                    break Some(position + 4);
                }
            };
            let Some(header_end) = header_end else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(0);
            while request.len() < header_end + content_length {
                let read = stream.read(&mut buffer).expect("read body");
                request.extend_from_slice(&buffer[..read]);
            }
            if !headers.contains("chat/completions") {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                );
                continue;
            }
            if let Some(gate) = &gate {
                let _ = gate.recv();
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            served.push(String::from_utf8_lossy(&request[header_end..]).to_string());
            next = bodies.next();
        }
        served
    })
}

/// Starts a Streamable HTTP MCP server offering `tools` (name, read-only)
/// whose calls answer `called <name>`; returns its endpoint.
pub(crate) async fn serve_mcp(tools: Vec<(&'static str, bool)>) -> String {
    use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
    use serde_json::{Value, json};

    let handler = move |Json(request): Json<Value>| {
        let tools = tools.clone();
        async move {
            let Some(id) = request.get("id").cloned() else {
                return StatusCode::ACCEPTED.into_response();
            };
            let result = match request["method"].as_str().unwrap_or_default() {
                "initialize" => json!({
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                    "instructions": "Use the fake tools."
                }),
                "tools/list" => json!({
                    "tools": tools.iter().map(|(name, read_only)| json!({
                        "name": name,
                        "description": format!("The {name} tool."),
                        "inputSchema": {"type": "object"},
                        "annotations": {"readOnlyHint": read_only}
                    })).collect::<Vec<_>>()
                }),
                "tools/call" => json!({
                    "content": [{
                        "type": "text",
                        "text": format!("called {}", request["params"]["name"].as_str().unwrap_or_default())
                    }],
                    "isError": false
                }),
                _ => {
                    return Json(json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": {"code": -32601, "message": "method not found"}
                    }))
                    .into_response();
                }
            };
            Json(json!({"jsonrpc": "2.0", "id": id, "result": result})).into_response()
        }
    };
    let app = Router::new().route(
        "/mcp",
        post(handler)
            .get(|| async { StatusCode::METHOD_NOT_ALLOWED })
            .delete(|| async { StatusCode::OK }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind MCP server");
    let address = listener.local_addr().expect("MCP server address");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{address}/mcp")
}

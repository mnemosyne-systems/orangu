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

//! Local semantic code search.
//!
//! Embeds the workspace's symbols offline through the server that serves the
//! `embeddings` role, persists the vectors under
//! `~/.orangu/workspace/<hash>/embeddings/` (incrementally, like the knowledge
//! graph cache), and answers `/search` queries by cosine similarity fused with
//! the knowledge graph's call edges.
//!
//! Semantic search enables itself when the embeddings endpoint responds at
//! startup; otherwise the whole subsystem stays dormant and retrieval falls back
//! to the knowledge graph and `/grep`.

pub mod chunk;
pub mod client;
pub mod index;

use crate::config::LlmConfiguration;
use crate::llm::normalized_openai_endpoint;

pub use client::EmbeddingClient;
pub use index::{EmbeddedChunk, EmbeddingIndex, SearchHit};

/// Probe a server's `/v1/embeddings` endpoint with a trivial request. `Ok(())`
/// means it responded successfully and semantic `/search` can be enabled;
/// `Err(reason)` gives a one-line, human-readable cause (connection refused,
/// timed out, an HTTP error status, or an unparseable response) so a failed
/// probe is diagnosable instead of a silent "not detected".
///
/// Uses `profile.request_timeout_seconds` (the same tolerance already
/// configured for ordinary chat requests) rather than a short fixed timeout:
/// the first request often triggers a cold model load — Ollama, for
/// instance, unloads idle models and reloading one takes a few seconds, and
/// an orangu-coordinator fronting this server may need to stop whatever
/// model is currently active and start this profile's from scratch, which
/// can take much longer than a few seconds for a large model. A server that
/// is down refuses the connection immediately, so this only waits when the
/// server is actually up and warming up.
pub async fn probe_endpoint(profile: &LlmConfiguration) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            profile.request_timeout_seconds,
        ))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    let url = format!(
        "{}/v1/embeddings",
        normalized_openai_endpoint(&profile.endpoint)
    );
    let body = serde_json::json!({ "model": profile.model, "input": "ping" });
    let mut request = client.post(&url).json(&body);
    if let Some(api_key) = &profile.api_key {
        request = request.bearer_auth(api_key);
    }
    let resp = request
        .send()
        .await
        .map_err(|e| format!("could not reach {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        return Err(format!("{url} responded with status {status}: {snippet}"));
    }
    Ok(())
}

/// Render search hits as a human-readable block for the output window, with each
/// result anchored to its `file:line` so it is clickable.
pub fn format_hits(query: &str, hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return format!("No semantic matches for \"{query}\".");
    }
    let mut out = format!("Semantic matches for \"{query}\":\n");
    for (i, hit) in hits.iter().enumerate() {
        let via = match &hit.expanded_from {
            Some(sym) => format!("  (via {sym})"),
            None => String::new(),
        };
        out.push_str(&format!(
            "{:>2}. {} — [{}:{}]  {:.3}{}\n",
            i + 1,
            hit.symbol,
            hit.file,
            hit.start_line,
            hit.score,
            via,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn test_profile(endpoint: String, request_timeout_seconds: u64) -> LlmConfiguration {
        LlmConfiguration {
            provider: "openai".to_string(),
            endpoint,
            model: "test-model".to_string(),
            role: "embeddings".to_string(),
            api_key: None,
            request_timeout_seconds,
            max_tool_rounds: 1,
            review_max_tokens: 0,
            code_max_tokens: 0,
            system_prompt: String::new(),
            model_verbosity: None,
            review_confidence_threshold: 0,
        }
    }

    /// Accepts one connection, waits `delay` before answering with a
    /// minimal successful JSON body, then closes — standing in for a slow
    /// cold model load (e.g. an orangu-coordinator swapping the active
    /// model) without needing a real llama.cpp or coordinator process.
    async fn spawn_slow_responder(delay: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;
            tokio::time::sleep(delay).await;
            let body = br#"{"data":[{"embedding":[0.0]}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.write_all(body).await;
            let _ = stream.shutdown().await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn probe_endpoint_times_out_when_the_configured_timeout_is_too_short() {
        let endpoint = spawn_slow_responder(Duration::from_secs(2)).await;
        let profile = test_profile(endpoint, 1);
        let err = probe_endpoint(&profile).await.unwrap_err();
        assert!(err.contains("could not reach"), "{err}");
    }

    #[tokio::test]
    async fn probe_endpoint_honors_the_configured_timeout_for_a_slow_cold_load() {
        // Regression test: this timeout used to be hardcoded to 20s
        // regardless of the user's own configured tolerance, which is far
        // too short for an orangu-coordinator that needs to stop one model
        // and cold-load another before it can answer.
        let endpoint = spawn_slow_responder(Duration::from_secs(2)).await;
        let profile = test_profile(endpoint, 5);
        probe_endpoint(&profile)
            .await
            .expect("should tolerate the slow response");
    }
}

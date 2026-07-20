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

//! A minimal client for an OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Semantic `/search` embeds code chunks and queries with the server that serves
//! the `embeddings` role. The endpoint is the OpenAI embeddings shape, which
//! orangu-server also serves, so the same `LlmConfiguration` profile that drives chat
//! works here unchanged.

use std::time::Duration;

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::LlmConfiguration;
use crate::llm::normalized_openai_endpoint;

/// Embeds text through an OpenAI-compatible `/v1/embeddings` endpoint.
/// Cloning is cheap — the underlying HTTP client is reference-counted — so
/// concurrent embedding tasks each hold their own handle.
#[derive(Clone)]
pub struct EmbeddingClient {
    http_client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

impl EmbeddingClient {
    /// Build a client from the server profile serving the `embeddings` role.
    pub fn from_profile(profile: &LlmConfiguration) -> Result<Self> {
        Ok(Self {
            http_client: Client::builder()
                .timeout(Duration::from_secs(profile.request_timeout_seconds))
                .build()?,
            endpoint: normalized_openai_endpoint(&profile.endpoint),
            model: profile.model.clone(),
            api_key: profile.api_key.clone(),
        })
    }

    /// Embed a batch of inputs, returning one vector per input in the same
    /// order. An empty input slice returns an empty vector without a request.
    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/embeddings", self.endpoint);
        let request = EmbeddingRequest {
            model: &self.model,
            input: inputs,
        };

        let mut builder = self.http_client.post(&url).json(&request);
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }

        let resp = builder.send().await.map_err(|e| {
            anyhow!(
                "failed to send embeddings request to {} using model {}: {}",
                self.endpoint,
                self.model,
                e
            )
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "embeddings request failed (status {status}): {body}"
            ));
        }

        let parsed: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("failed to parse embeddings response: {e}"))?;

        // The server may return rows out of order; sort by the `index` field so
        // each embedding lines up with its input.
        let mut rows = parsed.data;
        rows.sort_by_key(|d| d.index);

        if rows.len() != inputs.len() {
            return Err(anyhow!(
                "embeddings response returned {} rows for {} inputs",
                rows.len(),
                inputs.len()
            ));
        }

        Ok(rows.into_iter().map(|d| d.embedding).collect())
    }

    /// Embed a single input, returning its vector.
    pub async fn embed_one(&self, input: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed(std::slice::from_ref(&input.to_string())).await?;
        vectors
            .pop()
            .ok_or_else(|| anyhow!("embeddings response was empty"))
    }
}

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

//! Coordinates llama.cpp's per-server `id_slot` assignment across the
//! sessions sharing one endpoint within a single orangu process.
//!
//! llama.cpp's `/v1/chat/completions` accepts a top-level `id_slot` field to
//! pin a request to a specific slot (default: any idle slot), and `GET
//! /props` reports how many slots (`-np`) the server was started with. Only
//! llama.cpp servers understand either of these; on any other
//! OpenAI-compatible server the `/props` probe simply fails once and every
//! session on that endpoint falls back to today's behavior (no `id_slot`
//! sent, the server picks).
//!
//! [`SlotRegistry`] is deliberately process-lifetime, not persisted: which
//! numeric slot a session lands on this run is arbitrary (round-robin), and
//! does not need to match a previous run — restoring a saved slot's KV cache
//! into a differently-numbered slot is supported by llama.cpp itself (see the
//! `save_slot`/`restore_slot` methods added alongside the KV-cache-paging
//! feature).

use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cheap to [`Clone`] — an `Arc` handle around shared, mutex-guarded
/// per-endpoint slot state.
#[derive(Clone, Default)]
pub struct SlotRegistry {
    inner: Arc<Mutex<HashMap<String, EndpointSlots>>>,
}

#[derive(Default)]
struct EndpointSlots {
    /// `None` until the first `/props` probe completes (successfully or not).
    total_slots: Option<u32>,
    /// Set once a `/props` probe fails or reports no slots — every later call
    /// for this endpoint short-circuits without another request.
    unsupported: bool,
    /// Round-robin cursor, wrapped mod `total_slots` on each assignment.
    next_slot: u32,
    /// Set once a save or restore call fails for this endpoint (e.g. the
    /// server wasn't started with `--slot-save-path`) — every later call
    /// short-circuits without another request.
    save_restore_unsupported: bool,
    /// Whether the one-time "KV cache persistence unavailable" notice has
    /// already been shown for this endpoint.
    save_restore_notified: bool,
}

/// The result of a [`SlotRegistry::save_slot`]/[`SlotRegistry::restore_slot`]
/// call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveRestoreOutcome {
    Ok,
    /// The server doesn't support slot save/restore (or this endpoint has
    /// already failed once and further attempts are short-circuited).
    Unsupported,
}

impl SlotRegistry {
    /// Resolve (probing `/props` at most once per endpoint) and hand out the
    /// next slot index for `endpoint`, round-robin. `None` means the probe
    /// failed (non-llama.cpp server, or `--props` disabled) or the server
    /// reported zero slots — callers should treat that as "don't pin, let the
    /// server pick," which is today's behavior.
    pub async fn assign_slot(
        &self,
        client: &Client,
        endpoint: &str,
        api_key: Option<&str>,
    ) -> Option<u32> {
        let total_slots = self.total_slots(client, endpoint, api_key).await?;
        if total_slots == 0 {
            return None;
        }
        let mut registry = self.inner.lock().expect("slot registry lock poisoned");
        let entry = registry.entry(endpoint.to_string()).or_default();
        let slot = entry.next_slot % total_slots;
        entry.next_slot = entry.next_slot.wrapping_add(1);
        Some(slot)
    }

    async fn total_slots(
        &self,
        client: &Client,
        endpoint: &str,
        api_key: Option<&str>,
    ) -> Option<u32> {
        {
            let registry = self.inner.lock().expect("slot registry lock poisoned");
            if let Some(entry) = registry.get(endpoint) {
                if entry.unsupported {
                    return None;
                }
                if let Some(total) = entry.total_slots {
                    return Some(total);
                }
            }
        }
        // Not yet probed for this endpoint. The lock is dropped before this
        // await; a concurrent caller may probe the same endpoint at the same
        // time, which is harmless (an idempotent GET), just redundant.
        let probed = probe_total_slots(client, endpoint, api_key).await;
        let mut registry = self.inner.lock().expect("slot registry lock poisoned");
        let entry = registry.entry(endpoint.to_string()).or_default();
        match probed {
            Some(total) => {
                entry.total_slots = Some(total);
                Some(total)
            }
            None => {
                entry.unsupported = true;
                None
            }
        }
    }

    /// `POST {endpoint}/slots/{id_slot}?action=save` — persists `id_slot`'s
    /// KV cache to `filename` under the server's `--slot-save-path`
    /// directory. A no-op (returns `Unsupported` without a request) once this
    /// endpoint has failed once.
    pub async fn save_slot(
        &self,
        client: &Client,
        endpoint: &str,
        api_key: Option<&str>,
        id_slot: u32,
        filename: &str,
    ) -> SaveRestoreOutcome {
        self.save_or_restore(client, endpoint, api_key, id_slot, filename, "save")
            .await
    }

    /// `POST {endpoint}/slots/{id_slot}?action=restore` — loads a
    /// previously-saved `filename` back into `id_slot` (which may differ from
    /// the slot it was saved from). Same short-circuiting as `save_slot`.
    pub async fn restore_slot(
        &self,
        client: &Client,
        endpoint: &str,
        api_key: Option<&str>,
        id_slot: u32,
        filename: &str,
    ) -> SaveRestoreOutcome {
        self.save_or_restore(client, endpoint, api_key, id_slot, filename, "restore")
            .await
    }

    async fn save_or_restore(
        &self,
        client: &Client,
        endpoint: &str,
        api_key: Option<&str>,
        id_slot: u32,
        filename: &str,
        action: &str,
    ) -> SaveRestoreOutcome {
        if self.is_save_restore_unsupported(endpoint) {
            return SaveRestoreOutcome::Unsupported;
        }
        let url = format!("{endpoint}/slots/{id_slot}?action={action}");
        let mut builder = client
            .post(url)
            .json(&serde_json::json!({ "filename": filename }));
        if let Some(api_key) = api_key {
            builder = builder.bearer_auth(api_key);
        }
        let succeeded = matches!(
            builder.send().await,
            Ok(response) if response.status().is_success()
        );
        if succeeded {
            SaveRestoreOutcome::Ok
        } else {
            self.mark_save_restore_unsupported(endpoint);
            SaveRestoreOutcome::Unsupported
        }
    }

    fn is_save_restore_unsupported(&self, endpoint: &str) -> bool {
        let registry = self.inner.lock().expect("slot registry lock poisoned");
        registry
            .get(endpoint)
            .is_some_and(|entry| entry.save_restore_unsupported)
    }

    fn mark_save_restore_unsupported(&self, endpoint: &str) {
        let mut registry = self.inner.lock().expect("slot registry lock poisoned");
        registry
            .entry(endpoint.to_string())
            .or_default()
            .save_restore_unsupported = true;
    }

    /// Returns `true` exactly once per endpoint — the first time this
    /// endpoint is found not to support slot save/restore — so callers can
    /// show a one-time notice instead of repeating it on every tab switch.
    pub fn notify_save_restore_unsupported(&self, endpoint: &str) -> bool {
        let mut registry = self.inner.lock().expect("slot registry lock poisoned");
        let entry = registry.entry(endpoint.to_string()).or_default();
        if entry.save_restore_notified {
            false
        } else {
            entry.save_restore_notified = true;
            true
        }
    }
}

async fn probe_total_slots(client: &Client, endpoint: &str, api_key: Option<&str>) -> Option<u32> {
    let mut builder = client.get(format!("{endpoint}/props"));
    if let Some(api_key) = api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    body.get("total_slots")?.as_u64().map(|n| n as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn assign_slot_round_robins_and_wraps_at_total_slots() {
        // `/props` is probed at most once per endpoint (assignments after the
        // first are served from the cached `total_slots`), so the stub server
        // only needs to answer a single request.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            serve_props_response(&mut stream, 2);
        });
        let endpoint = format!("http://{addr}");
        let client = Client::new();
        let registry = SlotRegistry::default();

        let slots: Vec<Option<u32>> = futures_sequential(&registry, &client, &endpoint, 4).await;
        assert_eq!(slots, vec![Some(0), Some(1), Some(0), Some(1)]);
    }

    #[tokio::test]
    async fn assign_slot_returns_none_when_props_is_unavailable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            use std::io::Write;
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
        });
        let endpoint = format!("http://{addr}");
        let client = Client::new();
        let registry = SlotRegistry::default();

        assert_eq!(registry.assign_slot(&client, &endpoint, None).await, None);
        // Second call must not probe again (the listener above only answers
        // one request) — it should short-circuit on the cached `unsupported`.
        assert_eq!(registry.assign_slot(&client, &endpoint, None).await, None);
    }

    #[tokio::test]
    async fn save_slot_succeeds_and_restore_slot_can_target_a_different_slot() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept");
                serve_ok_response(&mut stream, r#"{"n_saved":84}"#);
            }
        });
        let endpoint = format!("http://{addr}");
        let client = Client::new();
        let registry = SlotRegistry::default();

        // Save from slot 1, restore into slot 0 — llama.cpp allows the
        // restore target to differ from the save-time slot.
        assert_eq!(
            registry
                .save_slot(&client, &endpoint, None, 1, "session.bin")
                .await,
            SaveRestoreOutcome::Ok
        );
        assert_eq!(
            registry
                .restore_slot(&client, &endpoint, None, 0, "session.bin")
                .await,
            SaveRestoreOutcome::Ok
        );
    }

    #[tokio::test]
    async fn save_slot_failure_is_remembered_and_notified_once() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            // Only one request is ever expected: the second `save_slot` call
            // below must short-circuit on the cached `unsupported` flag.
            let (mut stream, _) = listener.accept().expect("accept");
            use std::io::Write;
            let _ = stream.write_all(b"HTTP/1.1 501 Not Implemented\r\ncontent-length: 0\r\n\r\n");
        });
        let endpoint = format!("http://{addr}");
        let client = Client::new();
        let registry = SlotRegistry::default();

        assert_eq!(
            registry
                .save_slot(&client, &endpoint, None, 0, "session.bin")
                .await,
            SaveRestoreOutcome::Unsupported
        );
        assert_eq!(
            registry
                .restore_slot(&client, &endpoint, None, 0, "session.bin")
                .await,
            SaveRestoreOutcome::Unsupported
        );

        // The one-time notice fires exactly once across both save and
        // restore having discovered the same unsupported endpoint.
        assert!(registry.notify_save_restore_unsupported(&endpoint));
        assert!(!registry.notify_save_restore_unsupported(&endpoint));
    }

    async fn futures_sequential(
        registry: &SlotRegistry,
        client: &Client,
        endpoint: &str,
        count: usize,
    ) -> Vec<Option<u32>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(registry.assign_slot(client, endpoint, None).await);
        }
        out
    }

    fn serve_props_response(stream: &mut std::net::TcpStream, total_slots: u32) {
        serve_ok_response(stream, &format!(r#"{{"total_slots":{total_slots}}}"#));
    }

    fn serve_ok_response(stream: &mut std::net::TcpStream, body: &str) {
        use std::io::{Read, Write};
        let mut buffer = [0u8; 1024];
        let _ = stream.read(&mut buffer);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}

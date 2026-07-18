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

//! Chat session persistence: one directory per session, at
//! `~/.orangu/server/sessions/<uuid>/chat.json` — a directory rather than a
//! flat `<uuid>.json` file so a session can grow more per-session files
//! later (attachments, a session-scoped cache, ...) without another
//! layout migration, the same "one identifier, one directory" shape
//! `engine::backend::vulkan`'s persistent pipeline cache uses for its own
//! per-adapter directory. A session id is always a UUID v4 minted by
//! [`create_session`] — [`load_session`] parses whatever a caller (an HTTP
//! path segment) hands it back through [`uuid::Uuid`] before ever building
//! a filesystem path from it, so a malformed or path-traversal-shaped id
//! is rejected rather than reaching `fs::read`.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Session {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    /// Derived from the first user message once there is one; empty (shown
    /// as "New chat" by the UI) until then.
    pub title: String,
    pub messages: Vec<SessionMessage>,
}

#[derive(Serialize, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn sessions_dir() -> Result<PathBuf> {
    let dir = home::home_dir()
        .context("failed to resolve home directory")?
        .join(".orangu/server/sessions");
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir)
}

/// This session's own directory, `sessions_dir()/<id>/` — not the
/// `chat.json` file itself, so callers that need to create it first
/// (`save_session`) don't have to re-derive the parent from
/// [`session_chat_path`].
fn session_dir(id: &Uuid) -> Result<PathBuf> {
    Ok(sessions_dir()?.join(id.to_string()))
}

fn session_chat_path(id: &Uuid) -> Result<PathBuf> {
    Ok(session_dir(id)?.join("chat.json"))
}

pub fn create_session() -> Result<Session> {
    let now = unix_now();
    let session = Session {
        id: Uuid::new_v4().to_string(),
        created_at: now,
        updated_at: now,
        title: String::new(),
        messages: Vec::new(),
    };
    save_session(&session)?;
    Ok(session)
}

pub fn save_session(session: &Session) -> Result<()> {
    let id = Uuid::parse_str(&session.id).context("session id is not a valid UUID")?;
    let dir = session_dir(&id)?;
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join("chat.json");
    let json = serde_json::to_string_pretty(session).context("serializing session")?;
    fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))
}

/// Loads a session by id. `id` is parsed as a UUID before touching the
/// filesystem — an invalid id (including anything path-traversal-shaped)
/// is rejected here, never reaching `session_chat_path`/`fs::read_to_string`.
pub fn load_session(id: &str) -> Result<Session> {
    let uuid = Uuid::parse_str(id).map_err(|_| anyhow!("'{id}' is not a valid session id"))?;
    let path = session_chat_path(&uuid)?;
    let contents =
        fs::read_to_string(&path).with_context(|| format!("session '{id}' was not found"))?;
    serde_json::from_str(&contents).with_context(|| format!("session '{id}' is corrupt"))
}

pub fn list_sessions() -> Result<Vec<SessionSummary>> {
    let dir = sessions_dir()?;
    let mut summaries = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let Ok(entry) = entry else { continue };
        if !entry.path().is_dir() {
            continue;
        }
        let Ok(contents) = fs::read_to_string(entry.path().join("chat.json")) else {
            continue;
        };
        let Ok(session) = serde_json::from_str::<Session>(&contents) else {
            continue;
        };
        // A session with no messages was created (e.g. by New Chat, or on
        // first page load) but never actually used — not worth surfacing
        // in History.
        if session.messages.is_empty() {
            continue;
        }
        summaries.push(SessionSummary {
            id: session.id,
            title: session.title,
            created_at: session.created_at,
            updated_at: session.updated_at,
        });
    }
    summaries.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(summaries)
}

/// Appends `user_message`/`assistant_message` to `session`, deriving its
/// title from the first user message if it doesn't have one yet, and saves
/// it to disk.
pub fn append_turn(
    session: &mut Session,
    user_message: &str,
    assistant_message: &str,
) -> Result<()> {
    if session.title.is_empty() {
        session.title = derive_title(user_message);
    }
    session.messages.push(SessionMessage {
        role: "user".to_string(),
        content: user_message.to_string(),
    });
    session.messages.push(SessionMessage {
        role: "assistant".to_string(),
        content: assistant_message.to_string(),
    });
    session.updated_at = unix_now();
    save_session(session)
}

fn derive_title(first_message: &str) -> String {
    const MAX_LEN: usize = 60;
    let trimmed = first_message.trim();
    let title: String = trimmed.chars().take(MAX_LEN).collect();
    if trimmed.chars().count() > MAX_LEN {
        format!("{title}…")
    } else {
        title
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Tests share HOME via env var overrides, and must not run concurrently.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_temp_home<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let original = std::env::var_os("HOME");
        unsafe {
            std::env::set_var("HOME", dir.path());
        }
        let result = f();
        unsafe {
            match &original {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        result
    }

    #[test]
    fn create_then_load_round_trips() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            let loaded = load_session(&session.id).unwrap();
            assert_eq!(loaded.id, session.id);
            assert!(loaded.messages.is_empty());
        });
    }

    #[test]
    fn append_turn_sets_title_from_first_message_only() {
        with_temp_home(|| {
            let mut session = create_session().unwrap();
            append_turn(&mut session, "What is Rust?", "A systems language.").unwrap();
            assert_eq!(session.title, "What is Rust?");
            append_turn(&mut session, "And Go?", "Also a systems-ish language.").unwrap();
            assert_eq!(session.title, "What is Rust?");
            assert_eq!(session.messages.len(), 4);
        });
    }

    #[test]
    fn list_sessions_sorts_by_most_recently_updated() {
        with_temp_home(|| {
            let mut a = create_session().unwrap();
            let mut b = create_session().unwrap();
            a.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
            b.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
            a.updated_at = 100;
            b.updated_at = 200;
            save_session(&a).unwrap();
            save_session(&b).unwrap();

            let summaries = list_sessions().unwrap();
            assert_eq!(summaries.len(), 2);
            assert_eq!(summaries[0].id, b.id);
            assert_eq!(summaries[1].id, a.id);
        });
    }

    #[test]
    fn list_sessions_excludes_sessions_with_no_messages() {
        with_temp_home(|| {
            let empty = create_session().unwrap();
            let mut used = create_session().unwrap();
            used.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
            save_session(&empty).unwrap();
            save_session(&used).unwrap();

            let summaries = list_sessions().unwrap();
            assert_eq!(summaries.len(), 1);
            assert_eq!(summaries[0].id, used.id);
        });
    }

    #[test]
    fn load_session_rejects_path_traversal_ids() {
        with_temp_home(|| {
            let err = load_session("../../../etc/passwd").unwrap_err();
            assert!(err.to_string().contains("not a valid session id"));
        });
    }

    #[test]
    fn derive_title_truncates_long_first_messages() {
        let long = "x".repeat(100);
        let title = derive_title(&long);
        assert_eq!(title.chars().count(), 61); // 60 + ellipsis
        assert!(title.ends_with('…'));
    }
}

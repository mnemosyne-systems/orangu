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
//!
//! Each session directory also gets a `session.json` — see
//! [`SessionActivity`]/[`mark_active`]/[`is_active`] — recording which
//! (still-running or not) `orangu-server` process most recently touched it,
//! so `orangu-server prune` (`crate::prune`, a separate CLI invocation from
//! whatever server process actually owns a session) can tell a session a
//! live server is still using apart from an old, abandoned one.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use sysinfo::{Pid, ProcessesToUpdate, System};
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

fn session_activity_path(id: &Uuid) -> Result<PathBuf> {
    Ok(session_dir(id)?.join("session.json"))
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
    fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    // Best-effort — see `mark_active`'s own doc comment. `create_session`
    // and `append_turn` both funnel through here, so this is the one call
    // site that needs to refresh the activity marker.
    let _ = mark_active(&id);
    Ok(())
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SessionActivity {
    pid: u32,
    /// This process's own start time (`sysinfo::Process::start_time()`,
    /// seconds since the Unix epoch), stored alongside `pid` so a later,
    /// unrelated process the OS happens to reuse this pid for isn't
    /// mistaken for the original one — [`is_active`] treats a start-time
    /// mismatch the same as the pid no longer existing at all: not active.
    started_at: u64,
    updated_at: u64,
}

/// This process's own current `(pid, start_time)`, queried fresh — not
/// cached — since it's only ever called right before a filesystem write
/// (`mark_active`) or a one-off liveness check (`is_active`/
/// `sweep_empty_sessions`), never in a hot per-token loop.
fn process_start_time(pid: u32) -> Option<u64> {
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid)]), true);
    system.process(Pid::from_u32(pid)).map(|p| p.start_time())
}

/// Records that this process (its pid and start time) is the one currently
/// using `id` — called from [`save_session`] so both creating a session and
/// appending a turn to one refresh it. Best-effort: a failure here doesn't
/// fail the session save itself, since `chat.json` (already written by the
/// time this runs) is the data that actually matters; a session that never
/// gets an activity marker (or whose marker write failed) just reads as
/// "not active" to [`is_active`], the same as one from a version of
/// `orangu-server` predating this file.
fn mark_active(id: &Uuid) -> Result<()> {
    let pid = std::process::id();
    let activity = SessionActivity {
        pid,
        started_at: process_start_time(pid).unwrap_or(0),
        updated_at: unix_now(),
    };
    let path = session_activity_path(id)?;
    let json = serde_json::to_string_pretty(&activity).context("serializing session activity")?;
    fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))
}

/// Whether `id` is currently owned by a live `orangu-server` process — read
/// by `orangu-server prune` (`crate::prune`), a separate CLI invocation
/// from whatever process actually wrote the session's `session.json`, to
/// decide whether it's still in use before deleting it. `false` for a
/// session with no `session.json` (created by a build predating this, or
/// never actually saved), whose recorded pid isn't running at all, or whose
/// running process's start time doesn't match what was recorded (the pid
/// was reused by an unrelated process since) — every one of those reads as
/// "not active," never as an error, since an unreadable/missing marker is
/// exactly what an old, safely-prunable session looks like.
pub fn is_active(id: &str) -> bool {
    let Ok(uuid) = Uuid::parse_str(id) else {
        return false;
    };
    let Ok(path) = session_activity_path(&uuid) else {
        return false;
    };
    let Ok(contents) = fs::read_to_string(&path) else {
        return false;
    };
    let Ok(activity) = serde_json::from_str::<SessionActivity>(&contents) else {
        return false;
    };
    process_start_time(activity.pid) == Some(activity.started_at)
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

/// One row of `orangu-server prune`'s listing — unlike [`SessionSummary`]/
/// [`list_sessions`] (the web UI's History list), this includes
/// zero-message sessions too (only the ones [`sweep_empty_sessions`]
/// couldn't remove because they're active — see `list_sessions_for_prune`'s
/// own doc comment), since `prune` needs to show *why* one wasn't already
/// swept, not hide it.
#[derive(Clone, Debug)]
pub struct PruneEntry {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    pub message_count: usize,
    pub active: bool,
}

/// Every session directory under `sessions_dir()`, most-recently-updated
/// first, for `orangu-server prune`. Call [`sweep_empty_sessions`] first
/// (`crate::prune::run` always does) — by the time this runs, the only
/// zero-message entries left are ones [`is_active`] protected, so this
/// doesn't need its own "skip if empty" filter the way [`list_sessions`]
/// does. A session directory whose `chat.json` is missing or fails to
/// parse (and that `sweep_empty_sessions` also couldn't remove, for the
/// same reason: it's active) is still listed — as untitled, zero messages,
/// dated from the directory's own filesystem metadata — rather than
/// silently hidden, since `prune` is specifically the tool for surfacing
/// and cleaning up exactly this kind of leftover.
pub fn list_sessions_for_prune() -> Result<Vec<PruneEntry>> {
    let dir = sessions_dir()?;
    let mut entries = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(id) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if Uuid::parse_str(id).is_err() {
            continue;
        }
        let active = is_active(id);
        let parsed = fs::read_to_string(path.join("chat.json"))
            .ok()
            .and_then(|contents| serde_json::from_str::<Session>(&contents).ok());
        match parsed {
            Some(session) => entries.push(PruneEntry {
                id: session.id,
                title: session.title,
                updated_at: session.updated_at,
                message_count: session.messages.len(),
                active,
            }),
            None => {
                let fallback_time = fs::metadata(&path)
                    .and_then(|m| m.modified().or_else(|_| m.created()))
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                entries.push(PruneEntry {
                    id: id.to_string(),
                    title: String::new(),
                    updated_at: fallback_time,
                    message_count: 0,
                    active,
                });
            }
        }
    }
    entries.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(entries)
}

/// Deletes every non-active session directory whose `chat.json` is empty
/// (zero messages — a "New Chat" click that was never actually used),
/// missing, or unparseable (a leftover from an interrupted write). Called
/// unconditionally at the start of every `orangu-server prune` invocation
/// (`crate::prune::run`), regardless of its own argument, so routine
/// `prune`/`prune all`/`prune <id>` calls also compact away this junk as a
/// side effect rather than needing a separate command for it. Returns how
/// many were removed.
pub fn sweep_empty_sessions() -> Result<usize> {
    let dir = sessions_dir()?;
    let mut removed = 0;
    for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(id) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if Uuid::parse_str(id).is_err() {
            continue;
        }
        if is_active(id) {
            continue;
        }
        let is_empty = match fs::read_to_string(path.join("chat.json")) {
            Ok(contents) => serde_json::from_str::<Session>(&contents)
                .map(|s| s.messages.is_empty())
                .unwrap_or(true),
            Err(_) => true,
        };
        if is_empty && fs::remove_dir_all(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Deletes a session's entire directory (`chat.json`, `session.json`, and
/// anything else under it) — used by `orangu-server prune` for an explicit
/// identifier or `all`. Rejects a malformed id the same way
/// [`load_session`] does, before ever building a filesystem path from it.
pub fn delete_session_dir(id: &str) -> Result<()> {
    let uuid = Uuid::parse_str(id).map_err(|_| anyhow!("'{id}' is not a valid session id"))?;
    let dir = session_dir(&uuid)?;
    fs::remove_dir_all(&dir).with_context(|| format!("failed to delete {}", dir.display()))
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

    #[test]
    fn save_session_marks_it_active_for_the_current_process() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            // `create_session` -> `save_session` -> `mark_active`, recording
            // this test process's own (very much alive) pid.
            assert!(is_active(&session.id));
        });
    }

    #[test]
    fn is_active_is_false_with_no_session_json() {
        with_temp_home(|| {
            // A session directory with a chat.json but no session.json —
            // what a pre-activity-tracking `orangu-server` build would have
            // left behind.
            let id = Uuid::new_v4();
            let session = Session {
                id: id.to_string(),
                created_at: 0,
                updated_at: 0,
                title: String::new(),
                messages: Vec::new(),
            };
            let dir = session_dir(&id).unwrap();
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("chat.json"),
                serde_json::to_string(&session).unwrap(),
            )
            .unwrap();
            assert!(!is_active(&session.id));
        });
    }

    #[test]
    fn is_active_is_false_for_a_pid_that_is_not_running() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            let id = Uuid::parse_str(&session.id).unwrap();
            // Overwrite the just-written (genuinely active) marker with an
            // implausible pid — simulating a session left over by a server
            // process that has since exited.
            let stale = SessionActivity {
                pid: u32::MAX,
                started_at: 0,
                updated_at: 0,
            };
            fs::write(
                session_activity_path(&id).unwrap(),
                serde_json::to_string(&stale).unwrap(),
            )
            .unwrap();
            assert!(!is_active(&session.id));
        });
    }

    #[test]
    fn is_active_is_false_when_the_pid_was_reused_by_a_different_process() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            let id = Uuid::parse_str(&session.id).unwrap();
            // Same (genuinely running) pid as the real marker, but a
            // start_time that can't possibly match this process's actual
            // one — simulating the OS having reused this pid for an
            // unrelated process since the marker was written.
            let mismatched = SessionActivity {
                pid: std::process::id(),
                started_at: u64::MAX,
                updated_at: 0,
            };
            fs::write(
                session_activity_path(&id).unwrap(),
                serde_json::to_string(&mismatched).unwrap(),
            )
            .unwrap();
            assert!(!is_active(&session.id));
        });
    }

    #[test]
    fn sweep_empty_sessions_removes_only_non_active_empty_ones() {
        with_temp_home(|| {
            let empty_inactive = create_session().unwrap();
            let mut used_inactive = create_session().unwrap();
            used_inactive.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
            save_session(&used_inactive).unwrap();
            let empty_active = create_session().unwrap();

            // Simulate `empty_inactive`/`used_inactive` belonging to a
            // server that has since exited, leaving `empty_active` as the
            // only one still owned by a live process (this test process).
            for s in [&empty_inactive, &used_inactive] {
                let id = Uuid::parse_str(&s.id).unwrap();
                let stale = SessionActivity {
                    pid: u32::MAX,
                    started_at: 0,
                    updated_at: 0,
                };
                fs::write(
                    session_activity_path(&id).unwrap(),
                    serde_json::to_string(&stale).unwrap(),
                )
                .unwrap();
            }

            let removed = sweep_empty_sessions().unwrap();
            assert_eq!(removed, 1);
            assert!(load_session(&empty_inactive.id).is_err());
            assert!(load_session(&used_inactive.id).is_ok());
            assert!(load_session(&empty_active.id).is_ok());
        });
    }

    #[test]
    fn list_sessions_for_prune_includes_empty_sessions() {
        with_temp_home(|| {
            let empty = create_session().unwrap();
            let mut used = create_session().unwrap();
            used.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
            save_session(&used).unwrap();

            let entries = list_sessions_for_prune().unwrap();
            assert_eq!(entries.len(), 2);
            assert!(entries.iter().any(|e| e.id == empty.id && e.active));
            assert!(entries.iter().any(|e| e.id == used.id && e.active));
        });
    }

    #[test]
    fn delete_session_dir_removes_it_and_rejects_bad_ids() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            delete_session_dir(&session.id).unwrap();
            assert!(load_session(&session.id).is_err());

            let err = delete_session_dir("not-a-uuid").unwrap_err();
            assert!(err.to_string().contains("not a valid session id"));
        });
    }
}

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
    /// Wall-clock time the engine spent generating this message, in
    /// milliseconds — `None` for user messages and for assistant messages
    /// persisted before this field existed. `#[serde(default)]` so old
    /// `chat.json` files on disk (written before this field existed) still
    /// deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_ms: Option<u64>,
    /// What answering took, for an assistant message the model wrote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<AnswerStats>,
    /// Files the user attached to this message. Kept (with their extracted
    /// text) so a later turn still has the document in context, and shown
    /// as chips under the message in the web UI. `#[serde(default)]` for
    /// backward compatibility with `chat.json` written before attachments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
    /// The model's chain of thought for this message, kept apart from the
    /// answer it led to. Empty for a user message, for a model that does
    /// not reason, and for assistant messages persisted before this field
    /// existed — `#[serde(default)]` so those still deserialize.
    ///
    /// Stored so a reloaded transcript shows what the live one showed, and
    /// **not** replayed into the next turn's prompt: a chat template
    /// rebuilds a previous turn from its answer alone (Qwen3.x's writes an
    /// empty `<think></think>` for it), and feeding thinking back would
    /// grow every following prompt by the largest thing in the reply.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
}

/// One user-uploaded file: its identity, plus the text pulled out of it by
/// `web::attachments` (`None` when the format carries no extractable text).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Attachment {
    pub name: String,
    pub mime: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Page index reused to select context for follow-up questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf: Option<orangu::pdf::ExtractedPdf>,
    /// For a picture (PNG, JPEG, SVG): the file it was kept as under the
    /// session's `files/` directory, so the transcript can show it again
    /// and a `qwen_image` model can start from it on this turn. The bytes
    /// themselves stay out of `chat.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// The picture's bytes between upload and being stored — never
    /// serialized; [`store_file`] moves them to disk and fills `image`.
    #[serde(skip)]
    pub image_bytes: Option<Vec<u8>>,
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
    /// Every version of the last turn — its question and its answer — once
    /// it has been edited or regenerated; empty until then, and emptied
    /// again by the next new turn. The last two `messages` are always
    /// `turn_versions[active_version]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turn_versions: Vec<TurnVersion>,
    #[serde(default)]
    pub active_version: usize,
    /// Set by [`take_last_turn`]: the next [`append_turn`] adds a version of
    /// the last turn rather than a new turn.
    #[serde(skip)]
    pub replacing_last_turn: bool,
    /// This chat's own system prompt and sampling settings.
    #[serde(default, skip_serializing_if = "ChatSettings::is_default")]
    pub settings: ChatSettings,
    /// The user gave `title`, so it is no longer derived from the first
    /// question.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub title_given: bool,
}

/// What one answer took.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnswerStats {
    /// Tokens of the conversation read for it.
    pub prompt_tokens: usize,
    /// Of those, the ones a cache already held.
    pub cached_tokens: usize,
    /// Tokens written, thinking included.
    pub output_tokens: usize,
    /// Milliseconds spent reading the prompt: the wait for the first token.
    pub prompt_ms: u64,
}

/// A chat's own generation settings. An empty system prompt sends none, and
/// each unset value takes the server's default.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ChatSettings {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub system_prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

impl ChatSettings {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Why these settings cannot be used, when they cannot: `limit` is the
    /// most tokens one answer may have.
    pub fn check(&self, limit: usize) -> Result<(), String> {
        let within = |value: Option<f64>, low: f64, high: f64, name: &str| match value {
            Some(v) if !(low..=high).contains(&v) => {
                Err(format!("{name} must be between {low} and {high}"))
            }
            _ => Ok(()),
        };
        within(self.temperature, 0.0, 2.0, "temperature")?;
        within(self.top_p, 0.0, 1.0, "top_p")?;
        within(self.min_p, 0.0, 1.0, "min_p")?;
        match self.max_tokens {
            Some(n) if n == 0 || n > limit => {
                Err(format!("max_tokens must be between 1 and {limit}"))
            }
            _ => Ok(()),
        }
    }
}

/// Stores `settings` as the chat `id`'s own.
pub fn set_settings(id: &str, settings: ChatSettings) -> Result<Session> {
    let mut session = load_session(id)?;
    session.settings = settings;
    save_session(&session)?;
    Ok(session)
}

/// One version of the last turn.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TurnVersion {
    pub user: SessionMessage,
    pub assistant: SessionMessage,
}

#[derive(Serialize, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    /// Where a search matched the chat's messages, rather than its title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
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
        turn_versions: Vec::new(),
        active_version: 0,
        replacing_last_turn: false,
        settings: ChatSettings::default(),
        title_given: false,
    };
    save_session(&session)?;
    Ok(session)
}

/// Takes the last question and its answer off `session`, so the next
/// [`append_turn`] adds another version of that turn: an edited question or
/// a regenerated answer. Returns the question taken off. Nothing is saved:
/// a turn that never completes leaves the session on disk as it was.
pub fn take_last_turn(session: &mut Session) -> Result<SessionMessage> {
    let count = session.messages.len();
    let answered = count >= 2
        && session.messages[count - 2].role == "user"
        && session.messages[count - 1].role == "assistant";
    if !answered {
        return Err(anyhow!(
            "the conversation has no answered question to change"
        ));
    }
    let assistant = session.messages.pop().expect("checked above");
    let user = session.messages.pop().expect("checked above");
    if session.turn_versions.is_empty() {
        session.turn_versions.push(TurnVersion {
            user: user.clone(),
            assistant,
        });
        session.active_version = 0;
    }
    // A first question rewritten gets its title from the new wording,
    // unless the user gave the chat one.
    if session.messages.is_empty() && !session.title_given {
        session.title.clear();
    }
    session.replacing_last_turn = true;
    Ok(user)
}

/// Shows version `index` of the last turn, and saves the session.
pub fn show_version(session: &mut Session, index: usize) -> Result<()> {
    let version = session
        .turn_versions
        .get(index)
        .cloned()
        .ok_or_else(|| anyhow!("the last turn has no version {}", index + 1))?;
    let count = session.messages.len();
    if count < 2 {
        return Err(anyhow!("the conversation has no last turn"));
    }
    session.messages.truncate(count - 2);
    session.messages.push(version.user);
    session.messages.push(version.assistant);
    session.active_version = index;
    session.updated_at = unix_now();
    save_session(session)
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
    let mut system = System::new().ok()?;
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
/// Keeps a file beside a session's transcript — a picture the user
/// attached, or one the model made — and returns the name it was stored
/// under, for [`file_path`] and the `/api/sessions/{id}/files/{name}`
/// route. The name is minted here (a counter and the extension), never
/// taken from the upload, so it is always a plain file name.
pub fn store_file(session_id: &str, extension: &str, bytes: &[u8]) -> Result<String> {
    let uuid = Uuid::parse_str(session_id)
        .map_err(|_| anyhow!("'{session_id}' is not a valid session id"))?;
    let dir = session_dir(&uuid)?.join("files");
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let extension: String = extension
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();
    let count = fs::read_dir(&dir)
        .map(|entries| entries.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    // Counter plus a time stamp: two files stored in one second after a
    // deletion cannot collide, and the name still sorts in order.
    let name = format!("{}-{count}.{extension}", unix_now());
    let path = dir.join(&name);
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(name)
}

/// The on-disk path of a file [`store_file`] kept, or an error for a name
/// that is not one it could have minted — the only names this accepts are
/// `<digits>-<digits>.<alphanumerics>`, so a path segment from a URL can
/// never reach outside the session's `files/` directory.
pub fn file_path(session_id: &str, name: &str) -> Result<PathBuf> {
    let uuid = Uuid::parse_str(session_id)
        .map_err(|_| anyhow!("'{session_id}' is not a valid session id"))?;
    let valid = name.split_once('.').is_some_and(|(stem, ext)| {
        !ext.is_empty()
            && ext.chars().all(|c| c.is_ascii_alphanumeric())
            && stem.split_once('-').is_some_and(|(a, b)| {
                !a.is_empty()
                    && !b.is_empty()
                    && a.chars().all(|c| c.is_ascii_digit())
                    && b.chars().all(|c| c.is_ascii_digit())
            })
    });
    if !valid {
        anyhow::bail!("'{name}' is not a session file name");
    }
    Ok(session_dir(&uuid)?.join("files").join(name))
}

/// The URL the web console loads a stored file from.
pub fn file_url(session_id: &str, name: &str) -> String {
    format!("/api/sessions/{session_id}/files/{name}")
}

pub fn load_session(id: &str) -> Result<Session> {
    let uuid = Uuid::parse_str(id).map_err(|_| anyhow!("'{id}' is not a valid session id"))?;
    let path = session_chat_path(&uuid)?;
    let contents =
        fs::read_to_string(&path).with_context(|| format!("session '{id}' was not found"))?;
    serde_json::from_str(&contents).with_context(|| format!("session '{id}' is corrupt"))
}

#[cfg(test)]
pub fn list_sessions() -> Result<Vec<SessionSummary>> {
    search_sessions("")
}

/// The chats whose title or messages contain `query` (ignoring case), as
/// [`list_sessions`] lists them; an empty query lists them all.
pub fn search_sessions(query: &str) -> Result<Vec<SessionSummary>> {
    let needle = query.trim().to_lowercase();
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
        let mut snippet = None;
        if !needle.is_empty() && !session.title.to_lowercase().contains(&needle) {
            snippet = session
                .messages
                .iter()
                .find_map(|message| snippet_of(&message.content, &needle));
            if snippet.is_none() {
                continue;
            }
        }
        summaries.push(SessionSummary {
            id: session.id,
            title: session.title,
            created_at: session.created_at,
            updated_at: session.updated_at,
            snippet,
        });
    }
    summaries.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(summaries)
}

/// A few words either side of where `needle` (lowercase) is in `text`.
fn snippet_of(text: &str, needle: &str) -> Option<String> {
    const AROUND: usize = 40;
    let lower = text.to_lowercase();
    let at = lower.find(needle)?;
    // Lowercasing can change byte lengths; map back by characters.
    let start_chars = lower[..at].chars().count();
    let chars: Vec<char> = text.chars().collect();
    let from = start_chars.saturating_sub(AROUND);
    let to = (start_chars + needle.chars().count() + AROUND).min(chars.len());
    let mut words: Vec<String> = chars[from..to]
        .iter()
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect();
    // Whole words only: drop the ones the window cut through.
    if from > 0 && words.len() > 1 {
        words.remove(0);
    }
    if to < chars.len() && words.len() > 1 {
        words.pop();
    }
    let mut snippet = words.join(" ");
    if from > 0 {
        snippet.insert(0, '…');
    }
    if to < chars.len() {
        snippet.push('…');
    }
    Some(snippet)
}

/// Gives the chat `id` the title `title`.
pub fn rename_session(id: &str, title: &str) -> Result<Session> {
    let title: String = title.split_whitespace().collect::<Vec<_>>().join(" ");
    if title.is_empty() {
        anyhow::bail!("a title cannot be empty");
    }
    let mut session = load_session(id)?;
    session.title = title.chars().take(200).collect();
    session.title_given = true;
    save_session(&session)?;
    Ok(session)
}

/// A chat as a Markdown document: its title, then each message under the
/// name of who wrote it.
pub fn export_markdown(session: &Session) -> String {
    let title = if session.title.is_empty() {
        "Chat"
    } else {
        &session.title
    };
    let mut out = format!("# {title}\n");
    for message in &session.messages {
        let who = if message.role == "assistant" {
            "orangu"
        } else {
            "You"
        };
        out.push_str(&format!("\n## {who}\n\n{}\n", message.content.trim_end()));
        let names: Vec<&str> = message
            .attachments
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        if !names.is_empty() {
            out.push_str(&format!("\n*Attached: {}*\n", names.join(", ")));
        }
    }
    out
}

/// A chat with the files kept beside it, as one JSON document that
/// [`import_session`] reads back.
#[derive(Serialize, Deserialize)]
pub struct SessionExport {
    pub format: String,
    pub version: u32,
    pub session: Session,
    /// Each kept file's bytes, base64, by name.
    #[serde(default)]
    pub files: std::collections::BTreeMap<String, String>,
}

/// What [`SessionExport::format`] says.
pub const EXPORT_FORMAT: &str = "orangu-chat";

/// The chat `id` and its files as a [`SessionExport`].
pub fn export_session(id: &str) -> Result<SessionExport> {
    use base64::Engine as _;
    let session = load_session(id)?;
    let mut files = std::collections::BTreeMap::new();
    let dir = session_dir(&Uuid::parse_str(id)?)?.join("files");
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if file_path(id, &name).is_ok()
                && let Ok(bytes) = fs::read(entry.path())
            {
                files.insert(
                    name,
                    base64::engine::general_purpose::STANDARD.encode(bytes),
                );
            }
        }
    }
    Ok(SessionExport {
        format: EXPORT_FORMAT.to_string(),
        version: 1,
        session,
        files,
    })
}

/// A new chat made from an exported one, with its files; links to them in
/// the messages are pointed at the new chat.
pub fn import_session(export: SessionExport) -> Result<Session> {
    use base64::Engine as _;
    if export.format != EXPORT_FORMAT {
        anyhow::bail!("not an exported orangu chat");
    }
    let mut session = export.session;
    let old_id = session.id.clone();
    let new_id = Uuid::new_v4().to_string();
    let rewrite = |text: &str| {
        text.replace(
            &format!("/api/sessions/{old_id}/files/"),
            &format!("/api/sessions/{new_id}/files/"),
        )
    };
    for message in &mut session.messages {
        message.content = rewrite(&message.content);
    }
    for version in &mut session.turn_versions {
        version.user.content = rewrite(&version.user.content);
        version.assistant.content = rewrite(&version.assistant.content);
    }
    session.id = new_id.clone();
    session.replacing_last_turn = false;
    if session.active_version >= session.turn_versions.len() {
        session.active_version = 0;
    }
    save_session(&session)?;
    for (name, data) in export.files {
        let path = file_path(&new_id, &name)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .with_context(|| format!("file {name} is not base64"))?;
        fs::create_dir_all(path.parent().expect("files/ has a parent"))?;
        fs::write(&path, bytes)?;
    }
    Ok(session)
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

/// Deletes every session directory under `sessions_dir()` — the web
/// console's **Clear all**. Best-effort per entry: a directory that can't
/// be removed (a permission problem, a file another process still holds
/// open on Windows) is skipped rather than abandoning the rest, since the
/// button's promise is "clear the history," and stopping at the first
/// obstacle would leave a list that's neither cleared nor intact. Returns
/// how many were removed.
///
/// Unlike [`sweep_empty_sessions`] this does *not* spare an active session:
/// the console's own current chat is one of the rows being cleared, and
/// leaving exactly the one the user is looking at behind is not what the
/// button says. The browser starts a fresh session immediately afterwards,
/// so nothing goes on writing into a directory that just went away.
pub fn delete_all_sessions() -> Result<usize> {
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
        // Only UUID-named directories, the same guard every other path
        // here applies — anything else under `sessions/` was not put there
        // by this code and is not this button's to delete.
        if Uuid::parse_str(id).is_err() {
            continue;
        }
        if fs::remove_dir_all(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Appends `user_message`/`assistant_message` to `session`, deriving its
/// title from the first user message if it doesn't have one yet, and saves
/// it to disk. `generation_ms` is however long the engine took to produce
/// `assistant_message` (`GenerateStats::generate_time`), shown as a light
/// footer under the message in the web UI.
pub fn append_turn(
    session: &mut Session,
    user_message: &str,
    user_attachments: Vec<Attachment>,
    assistant_message: &str,
    assistant_reasoning: &str,
    generation_ms: Option<u64>,
    stats: Option<AnswerStats>,
) -> Result<()> {
    if session.title.is_empty() {
        // A file-only message (no typed text) still deserves a title —
        // fall back to the first attachment's name.
        let title_seed = if user_message.trim().is_empty() {
            user_attachments
                .first()
                .map(|a| a.name.as_str())
                .unwrap_or(user_message)
        } else {
            user_message
        };
        session.title = derive_title(title_seed);
    }
    session.messages.push(SessionMessage {
        role: "user".to_string(),
        content: user_message.to_string(),
        generation_ms: None,
        stats: None,
        attachments: user_attachments,
        reasoning: String::new(),
    });
    session.messages.push(SessionMessage {
        role: "assistant".to_string(),
        content: assistant_message.to_string(),
        generation_ms,
        stats,
        attachments: Vec::new(),
        reasoning: assistant_reasoning.to_string(),
    });
    if std::mem::take(&mut session.replacing_last_turn) {
        let count = session.messages.len();
        session.turn_versions.push(TurnVersion {
            user: session.messages[count - 2].clone(),
            assistant: session.messages[count - 1].clone(),
        });
        session.active_version = session.turn_versions.len() - 1;
    } else {
        // A new turn keeps the version shown of the one before and drops
        // the others: only the last turn has versions.
        session.turn_versions.clear();
        session.active_version = 0;
    }
    session.updated_at = unix_now();
    // Settings changed, or a title given, while the answer was being
    // written are kept.
    if let Ok(on_disk) = load_session(&session.id) {
        session.settings = on_disk.settings;
        if on_disk.title_given {
            session.title = on_disk.title;
            session.title_given = true;
        }
    }
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

    // Tests share the home directory via env var overrides, and must not run
    // concurrently.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The variables `home::home_dir` actually consults, which is not the
    /// same set everywhere: `HOME` on Unix, `USERPROFILE` on Windows. Setting
    /// only `HOME` left every one of these tests pointed at the real home
    /// directory on Windows — so they saw each other's sessions and, worse,
    /// wrote into the user's own `~/.orangu` while doing it.
    #[cfg(unix)]
    const HOME_VARS: &[&str] = &["HOME"];
    #[cfg(windows)]
    const HOME_VARS: &[&str] = &["USERPROFILE", "HOME"];

    fn with_temp_home<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let original: Vec<_> = HOME_VARS
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();
        unsafe {
            for name in HOME_VARS {
                std::env::set_var(name, dir.path());
            }
        }
        let result = f();
        unsafe {
            for (name, value) in &original {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
        result
    }

    /// Guards the fix above: whatever this platform resolves a home directory
    /// from, `with_temp_home` has to actually redirect it. If it does not,
    /// every test here silently shares one directory and starts asserting
    /// against leftovers from its neighbours.
    #[test]
    fn with_temp_home_actually_redirects_the_home_directory() {
        with_temp_home(|| {
            // Compared against the variable this platform's lookup is
            // supposed to follow, entirely inside the lock — reading the
            // "real" home outside it would race with whichever neighbouring
            // test currently has the override installed.
            let overridden = std::env::var_os(HOME_VARS[0]).expect("home variable set");
            assert_eq!(
                home::home_dir().expect("a home directory"),
                PathBuf::from(overridden),
                "home_dir does not follow {} on this platform, so these \
                 tests would share one directory",
                HOME_VARS[0]
            );
        });
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
            append_turn(
                &mut session,
                "What is Rust?",
                Vec::new(),
                "A systems language.",
                "",
                Some(123),
                None,
            )
            .unwrap();
            assert_eq!(session.title, "What is Rust?");
            append_turn(
                &mut session,
                "And Go?",
                Vec::new(),
                "Also a systems-ish language.",
                "",
                Some(456),
                None,
            )
            .unwrap();
            assert_eq!(session.title, "What is Rust?");
            assert_eq!(session.messages.len(), 4);
        });
    }

    #[test]
    fn chat_settings_are_kept_with_the_chat() {
        with_temp_home(|| {
            let mut session = create_session().unwrap();
            let settings = ChatSettings {
                system_prompt: "Answer in French.".into(),
                temperature: Some(0.2),
                max_tokens: Some(512),
                ..ChatSettings::default()
            };
            set_settings(&session.id, settings.clone()).unwrap();
            // A turn that was answering while the settings changed keeps them.
            let stats = AnswerStats {
                prompt_tokens: 30,
                cached_tokens: 10,
                output_tokens: 4,
                prompt_ms: 12,
            };
            append_turn(
                &mut session,
                "Hi?",
                Vec::new(),
                "Salut.",
                "",
                Some(5),
                Some(stats),
            )
            .unwrap();
            let reloaded = load_session(&session.id).unwrap();
            assert_eq!(reloaded.settings, settings);
            assert_eq!(reloaded.messages[1].stats, Some(stats));
            assert_eq!(reloaded.messages[0].stats, None);
            assert_eq!(reloaded.messages.len(), 2);

            // Unset settings are not written at all.
            set_settings(&session.id, ChatSettings::default()).unwrap();
            let chat = fs::read_to_string(
                session_chat_path(&Uuid::parse_str(&session.id).unwrap()).unwrap(),
            )
            .unwrap();
            assert!(!chat.contains("settings"), "{chat}");
        });
    }

    #[test]
    fn history_search_rename_and_export_round_trip() {
        with_temp_home(|| {
            let mut session = create_session().unwrap();
            append_turn(
                &mut session,
                "Tell me about otters",
                Vec::new(),
                "Otters hold hands while they sleep in kelp forests.",
                "",
                None,
                None,
            )
            .unwrap();
            let mut other = create_session().unwrap();
            append_turn(&mut other, "Weather?", Vec::new(), "Rain.", "", None, None).unwrap();

            // By title, and by message text with a snippet.
            let by_title = search_sessions("OTTERS").unwrap();
            assert_eq!(by_title.len(), 1);
            assert_eq!(by_title[0].snippet, None);
            let by_text = search_sessions("kelp").unwrap();
            assert_eq!(by_text.len(), 1);
            assert!(
                by_text[0]
                    .snippet
                    .as_deref()
                    .unwrap()
                    .contains("kelp forests")
            );
            assert_eq!(search_sessions("").unwrap().len(), 2);
            assert!(search_sessions("zebra").unwrap().is_empty());

            rename_session(&session.id, "  Otter   notes ").unwrap();
            assert!(rename_session(&session.id, "   ").is_err());
            // An answer finishing after the rename keeps the new title.
            append_turn(&mut session, "More?", Vec::new(), "Yes.", "", None, None).unwrap();
            assert_eq!(load_session(&session.id).unwrap().title, "Otter notes");

            let markdown = export_markdown(&load_session(&session.id).unwrap());
            assert!(markdown.starts_with(
                "# Otter notes\n\n## You\n\nTell me about otters\n\n## orangu\n\nOtters hold hands"
            ));

            let picture = store_file(&session.id, "png", b"png bytes").unwrap();
            let mut stored = load_session(&session.id).unwrap();
            stored.messages[1].content =
                format!("![p](/api/sessions/{}/files/{picture})", session.id);
            save_session(&stored).unwrap();
            let export = export_session(&session.id).unwrap();
            let json = serde_json::to_string(&export).unwrap();
            let imported = import_session(serde_json::from_str(&json).unwrap()).unwrap();
            assert_ne!(imported.id, session.id);
            assert_eq!(imported.title, "Otter notes");
            assert_eq!(imported.messages.len(), 4);
            assert_eq!(
                imported.messages[1].content,
                format!("![p](/api/sessions/{}/files/{picture})", imported.id)
            );
            assert_eq!(
                fs::read(file_path(&imported.id, &picture).unwrap()).unwrap(),
                b"png bytes"
            );

            let mut wrong = export_session(&session.id).unwrap();
            wrong.format = "something".into();
            assert!(import_session(wrong).is_err());
        });
    }

    #[test]
    fn chat_settings_out_of_range_are_refused() {
        let ok = ChatSettings {
            temperature: Some(2.0),
            top_p: Some(1.0),
            min_p: Some(0.0),
            max_tokens: Some(100),
            ..ChatSettings::default()
        };
        assert_eq!(ok.check(100), Ok(()));
        let hot = ChatSettings {
            temperature: Some(2.5),
            ..ChatSettings::default()
        };
        assert_eq!(
            hot.check(100),
            Err("temperature must be between 0 and 2".to_string())
        );
        let long = ChatSettings {
            max_tokens: Some(101),
            ..ChatSettings::default()
        };
        assert_eq!(
            long.check(100),
            Err("max_tokens must be between 1 and 100".to_string())
        );
        let none = ChatSettings {
            max_tokens: Some(0),
            ..ChatSettings::default()
        };
        assert!(none.check(100).is_err());
    }

    #[test]
    fn edits_and_regenerations_are_versions_of_the_last_turn() {
        with_temp_home(|| {
            let mut session = create_session().unwrap();
            append_turn(&mut session, "First?", Vec::new(), "One.", "", None, None).unwrap();
            append_turn(&mut session, "Second?", Vec::new(), "Two.", "", None, None).unwrap();
            assert!(session.turn_versions.is_empty());

            // Regenerate: the same question, a new answer.
            let question = take_last_turn(&mut session).unwrap();
            assert_eq!(question.content, "Second?");
            assert_eq!(session.messages.len(), 2);
            append_turn(
                &mut session,
                &question.content,
                Vec::new(),
                "Deux.",
                "",
                None,
                None,
            )
            .unwrap();
            // Edit: a new question.
            take_last_turn(&mut session).unwrap();
            append_turn(
                &mut session,
                "Second, again?",
                Vec::new(),
                "Zwei.",
                "",
                None,
                None,
            )
            .unwrap();

            let saved = load_session(&session.id).unwrap();
            assert_eq!(saved.turn_versions.len(), 3);
            assert_eq!(saved.active_version, 2);
            assert_eq!(saved.messages.len(), 4);
            assert_eq!(saved.messages[3].content, "Zwei.");

            let mut saved = saved;
            show_version(&mut saved, 0).unwrap();
            let shown = load_session(&session.id).unwrap();
            assert_eq!(shown.messages[2].content, "Second?");
            assert_eq!(shown.messages[3].content, "Two.");
            assert_eq!(shown.active_version, 0);
            assert!(show_version(&mut saved, 3).is_err());

            // The next new turn keeps the version shown and drops the rest.
            let mut shown = shown;
            append_turn(&mut shown, "Third?", Vec::new(), "Three.", "", None, None).unwrap();
            assert!(shown.turn_versions.is_empty());
            assert_eq!(shown.messages[3].content, "Two.");
        });
    }

    #[test]
    fn only_an_answered_question_can_be_changed() {
        with_temp_home(|| {
            let mut session = create_session().unwrap();
            assert!(take_last_turn(&mut session).is_err());
            append_turn(&mut session, "Only?", Vec::new(), "Yes.", "", None, None).unwrap();
            take_last_turn(&mut session).unwrap();
            assert!(
                session.title.is_empty(),
                "a rewritten first question retitles"
            );
            append_turn(
                &mut session,
                "Rewritten?",
                Vec::new(),
                "Yes.",
                "",
                None,
                None,
            )
            .unwrap();
            assert_eq!(session.title, "Rewritten?");
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
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
            });
            b.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
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
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
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
                turn_versions: Vec::new(),
                active_version: 0,
                replacing_last_turn: false,
                settings: ChatSettings::default(),
                title_given: false,
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
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
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
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
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

    #[test]
    fn delete_all_sessions_removes_every_one_including_active() {
        with_temp_home(|| {
            // Both are active — `create_session` marked them for this very
            // process. Clear all must still take them: the console's own
            // current chat is one of the rows being cleared.
            let empty = create_session().unwrap();
            let mut used = create_session().unwrap();
            used.messages.push(SessionMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                generation_ms: None,
                stats: None,
                attachments: Vec::new(),
                reasoning: String::new(),
            });
            save_session(&used).unwrap();

            assert_eq!(delete_all_sessions().unwrap(), 2);
            assert!(load_session(&empty.id).is_err());
            assert!(load_session(&used.id).is_err());
            assert!(list_sessions().unwrap().is_empty());
        });
    }

    #[test]
    fn delete_all_sessions_leaves_non_session_entries_alone() {
        with_temp_home(|| {
            let session = create_session().unwrap();
            // Anything not named like a session id was not put here by this
            // code, and is not Clear all's to remove.
            let stray = sessions_dir().unwrap().join("notes");
            fs::create_dir_all(&stray).unwrap();

            assert_eq!(delete_all_sessions().unwrap(), 1);
            assert!(load_session(&session.id).is_err());
            assert!(stray.is_dir());
        });
    }
}

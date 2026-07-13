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

use crate::*;

pub(crate) const SESSIONS_DIRECTORY: &str = ".orangu/sessions";
pub(crate) const SESSION_SETTINGS_FILE: &str = "settings";
pub(crate) const SESSION_THEME_FILE: &str = "theme";

/// Load the server and model pinned to this session, if any.
/// Returns `(server, model)` — either or both may be `None`.
pub(crate) fn load_session_settings(session_dir: &Path) -> (Option<String>, Option<String>) {
    let path = session_dir.join(SESSION_SETTINGS_FILE);
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return (None, None);
    };
    let mut sections = match orangu::config::parse_ini_sections(&contents) {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    let client = match sections.remove(orangu::config::CLIENT_SECTION) {
        Some(s) => s,
        None => return (None, None),
    };
    let server = client
        .get("server")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let model = client
        .get("model")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    (server, model)
}

/// Persist the active server and model into the session's `settings` file so
/// they are restored when the session is resumed.  Either value may be omitted
/// to leave it unset (which means the global default applies on next resume).
pub(crate) fn save_session_settings(session_dir: &Path, server: Option<&str>, model: Option<&str>) {
    let mut body = format!("[{}]\n", orangu::config::CLIENT_SECTION);
    if let Some(s) = server {
        body.push_str(&format!("server = {s}\n"));
    }
    if let Some(m) = model {
        body.push_str(&format!("model = {m}\n"));
    }
    let _ = std::fs::write(session_dir.join(SESSION_SETTINGS_FILE), body);
}

/// Load the theme override pinned to this session, if any. Stored separately
/// from `settings` because it is a pure UI concern and the maintainer wants a
/// dedicated sidecar under `~/.orangu/sessions/<id>/theme`.
pub(crate) fn load_session_theme(session_dir: &Path) -> Option<String> {
    let path = session_dir.join(SESSION_THEME_FILE);
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Persist or clear this session's theme override.
pub(crate) fn save_session_theme(session_dir: &Path, theme: Option<&str>) {
    let path = session_dir.join(SESSION_THEME_FILE);
    match theme.map(str::trim).filter(|theme| !theme.is_empty()) {
        Some(theme) => {
            let _ = std::fs::write(path, format!("{theme}\n"));
        }
        None => {
            let _ = std::fs::remove_file(path);
        }
    }
}
/// Scratch directory used by `/restart` to stage a runnable copy of the binary
/// when the original on-disk path has been replaced (e.g. rebuilt while
/// running). Cleared on every startup.
pub(crate) const RESTART_DIRECTORY: &str = ".orangu/last";

pub(crate) fn session_dir_path(session_id: &str) -> Result<PathBuf> {
    let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(SESSIONS_DIRECTORY).join(session_id))
}

/// The `~/.orangu/last` scratch directory used across a `/restart` handoff.
pub(crate) fn restart_dir_path() -> Option<PathBuf> {
    Some(home::home_dir()?.join(RESTART_DIRECTORY))
}

/// Remove the `/restart` scratch directory. Errors are ignored: a missing or
/// unremovable directory must never block startup.
pub(crate) fn clear_restart_dir() {
    if let Some(dir) = restart_dir_path() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// File recording the workspace directories open in the last run, one per line.
/// Written on exit and read by `orangu -a|--all` to reopen those tabs.
pub(crate) const OPEN_WORKSPACES_FILE: &str = ".orangu/workspaces";

fn open_workspaces_path() -> Option<PathBuf> {
    Some(home::home_dir()?.join(OPEN_WORKSPACES_FILE))
}

/// Record the open workspace directories and session IDs, one per line, so
/// `orangu -a` can reopen them next time.  Branch information is stored in
/// each session's metadata file instead.  Errors are ignored: failing to save
/// the layout must never block exit.
pub(crate) fn save_open_workspaces(workspaces: &[(PathBuf, String)]) {
    let Some(path) = open_workspaces_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, format_workspace_entries(workspaces));
}

/// Format is `<path>\t<session-id>\n` per entry.  Every entry ends with a
/// newline so the file is a well-formed POSIX text file.
fn format_workspace_entries(workspaces: &[(PathBuf, String)]) -> String {
    use std::fmt::Write as FmtWrite;
    let mut body = String::new();
    for (workspace, session_id) in workspaces {
        writeln!(body, "{}\t{}", workspace.display(), session_id).ok();
    }
    body
}

/// The workspace directories and session IDs saved by the last run that still
/// exist, in tab-bar order.  Read by `orangu -a|--all`; missing directories
/// are skipped so a deleted project does not get recreated.
///
/// Accepts two formats for backward compatibility with pre-session-ID files:
/// - `<path>\t<session-id>` (current format)
/// - `<path>` (path-only, session auto-resumed on open)
pub(crate) fn load_open_workspaces() -> Vec<(PathBuf, Option<String>)> {
    let Some(path) = open_workspaces_path() else {
        return Vec::new();
    };
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    parse_workspace_entries(&contents)
        .into_iter()
        .filter(|(workspace, _)| workspace.is_dir())
        .collect()
}

fn parse_workspace_entries(contents: &str) -> Vec<(PathBuf, Option<String>)> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.splitn(2, '\t');
            let path = PathBuf::from(parts.next().unwrap_or(""));
            let session_id = parts.next().filter(|s| !s.is_empty()).map(str::to_string);
            (path, session_id)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_workspace_entries_writes_two_fields_with_trailing_newline() {
        let entries = vec![
            (PathBuf::from("/home/user/project"), "uuid-1".to_string()),
            (PathBuf::from("/home/user/other"), "uuid-2".to_string()),
        ];
        let body = format_workspace_entries(&entries);
        assert_eq!(
            body,
            "/home/user/project\tuuid-1\n/home/user/other\tuuid-2\n"
        );
    }

    #[test]
    fn format_workspace_entries_empty_produces_empty_string() {
        assert_eq!(format_workspace_entries(&[]), "");
    }

    #[test]
    fn parse_workspace_entries_reads_two_field_format() {
        let body = "/home/user/project\tuuid-1\n/home/user/other\tuuid-2\n";
        let entries = parse_workspace_entries(body);
        assert_eq!(
            entries,
            [
                (
                    PathBuf::from("/home/user/project"),
                    Some("uuid-1".to_string())
                ),
                (
                    PathBuf::from("/home/user/other"),
                    Some("uuid-2".to_string())
                ),
            ]
        );
    }

    #[test]
    fn parse_workspace_entries_accepts_old_path_only_format() {
        let body = "/a\n/b\n";
        let entries = parse_workspace_entries(body);
        assert_eq!(
            entries,
            [(PathBuf::from("/a"), None), (PathBuf::from("/b"), None),]
        );
    }

    #[test]
    fn parse_workspace_entries_skips_blank_lines() {
        let body = "/a\tuuid-1\n\n  \n/b\tuuid-2\n";
        let entries = parse_workspace_entries(body);
        assert_eq!(
            entries,
            [
                (PathBuf::from("/a"), Some("uuid-1".to_string())),
                (PathBuf::from("/b"), Some("uuid-2".to_string())),
            ]
        );
    }

    #[test]
    fn update_session_metadata_branch_persists_and_load_session_branch_retrieves() {
        use tempfile::tempdir;
        let dir = tempdir().expect("temp dir");
        let meta_path = dir.path().join("metadata");

        // Write a minimal metadata file so update/load have something to work with.
        let initial = SessionMetadata {
            started_at: 1_000_000,
            last_updated_at: 1_000_000,
            workspace: "/some/workspace".to_string(),
            branch: String::new(),
        };
        save_session_metadata(&meta_path, &initial).expect("save initial");

        // A session ID that resolves to our temp directory does not exist in the
        // standard location, so load_session_branch can only be tested via the
        // lower-level update/load pair here.
        update_session_metadata_branch(&meta_path, Some("feature/test-branch"))
            .expect("update branch");

        let loaded = load_session_metadata(&meta_path)
            .expect("read ok")
            .expect("metadata present");
        assert_eq!(loaded.branch, "feature/test-branch");
        // last_updated_at must have been bumped.
        assert!(loaded.last_updated_at >= initial.last_updated_at);

        // Setting branch to None leaves the existing value unchanged.
        update_session_metadata_branch(&meta_path, None).expect("no-op update");
        let after_noop = load_session_metadata(&meta_path)
            .expect("read ok")
            .expect("metadata present");
        assert_eq!(after_noop.branch, "feature/test-branch");

        // An empty string branch via Some("") should still write an empty branch.
        update_session_metadata_branch(&meta_path, Some("")).expect("clear branch");
        let after_clear = load_session_metadata(&meta_path)
            .expect("read ok")
            .expect("metadata present");
        assert_eq!(after_clear.branch, "");
    }

    #[test]
    fn format_and_parse_round_trip() {
        let original = vec![
            (PathBuf::from("/workspace/alpha"), "aaa-111".to_string()),
            (PathBuf::from("/workspace/beta"), "bbb-222".to_string()),
        ];
        let serialized = format_workspace_entries(&original);
        let parsed = parse_workspace_entries(&serialized);
        let expected: Vec<(PathBuf, Option<String>)> =
            original.into_iter().map(|(p, id)| (p, Some(id))).collect();
        assert_eq!(parsed, expected);
    }
}

/// Resolve the path to exec when restarting.
///
/// `std::env::current_exe()` reads `/proc/self/exe`, which keeps pointing at the
/// original inode. When the binary is rebuilt while running, that inode is
/// unlinked and the path is reported with a trailing ` (deleted)` marker, so
/// exec'ing it fails with `ENOENT`. We first retry the real on-disk path with
/// the marker stripped — after a rebuild that path holds the fresh binary, which
/// is exactly what a restart should pick up. If that path is gone entirely
/// (e.g. the build directory was cleaned), we fall back to copying the still-open
/// running binary into `~/.orangu/last` and exec'ing that copy, which always
/// succeeds even though it relaunches the previous build.
pub(crate) fn restart_executable_path() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;

    // Fast path: the binary on disk is intact.
    if exe.exists() {
        return Ok(exe);
    }

    // The reported path may carry the kernel's " (deleted)" suffix; the real
    // path without it usually holds the rebuilt binary.
    let display = exe.to_string_lossy();
    if let Some(stripped) = display.strip_suffix(" (deleted)") {
        let real = PathBuf::from(stripped);
        if real.exists() {
            return Ok(real);
        }
    }

    // Last resort: stage a copy of the running binary somewhere stable and exec
    // that. `/proc/self/exe` can still be read while the inode is open, even
    // after the original path is gone.
    let dir = restart_dir_path().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let staged = dir.join("orangu");
    std::fs::copy("/proc/self/exe", &staged)
        .with_context(|| format!("failed to stage restart binary at {}", staged.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("failed to mark {} executable", staged.display()))?;
    }
    Ok(staged)
}

pub(crate) fn load_session_messages(path: &Path) -> Result<Vec<ChatMessage>> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("failed to parse session messages {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read session messages {}", path.display()))
        }
    }
}

pub(crate) fn save_session_messages(path: &Path, messages: &[ChatMessage]) -> Result<()> {
    let json = serde_json::to_string(messages).context("failed to serialize session messages")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write session messages {}", path.display()))
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SessionMetadata {
    pub(crate) started_at: u64,
    pub(crate) last_updated_at: u64,
    pub(crate) workspace: String,
    #[serde(default)]
    pub(crate) branch: String,
}

pub(crate) fn current_unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn format_unix_timestamp(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}{m:02}{d:02}{hour:02}{min:02}")
}

/// Like [`format_unix_timestamp`] but spelled out as `YYYY-MM-DD HH:MM` for
/// human-facing output such as the `orangu -l|--list` table.
pub(crate) fn format_unix_timestamp_human(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02} {hour:02}:{min:02}")
}

pub(crate) fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    let mut year = 1970u32;
    loop {
        let in_year: u64 = if is_leap_year(year) { 366 } else { 365 };
        if days < in_year {
            break;
        }
        days -= in_year;
        year += 1;
    }
    let months: [u64; 12] = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u32;
    for dim in months {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    (year, month, days as u32 + 1)
}

pub(crate) fn is_leap_year(y: u32) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

/// Sidecar recording which server/model a session's saved orangu-server slot (see
/// `orangu::llm::SlotRegistry::save_slot`) was captured under, so a later
/// restore attempt can be gated on both matching — restoring a KV cache saved
/// under a different model is unsafe (tensor shapes are model-specific), and
/// the server API doesn't protect against it.
pub(crate) const SESSION_SLOT_FILE: &str = "slot.json";

#[derive(Serialize, Deserialize)]
pub(crate) struct SessionSlotInfo {
    pub(crate) server_endpoint: String,
    pub(crate) model: String,
    pub(crate) saved_at: u64,
}

pub(crate) fn save_session_slot_info(path: &Path, info: &SessionSlotInfo) -> Result<()> {
    let json = serde_json::to_string(info).context("failed to serialize session slot info")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write session slot info {}", path.display()))
}

pub(crate) fn load_session_slot_info(path: &Path) -> Result<Option<SessionSlotInfo>> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("failed to parse session slot info {}", path.display()))
            .map(Some),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read session slot info {}", path.display()))
        }
    }
}

pub(crate) fn save_session_metadata(path: &Path, metadata: &SessionMetadata) -> Result<()> {
    let json = serde_json::to_string(metadata).context("failed to serialize session metadata")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write session metadata {}", path.display()))
}

pub(crate) fn load_session_metadata(path: &Path) -> Result<Option<SessionMetadata>> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("failed to parse session metadata {}", path.display()))
            .map(Some),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read session metadata {}", path.display()))
        }
    }
}

/// Update both `last_updated_at` and `branch` in the session metadata file.
/// When `branch` is `None` the branch field is left as-is.
pub(crate) fn update_session_metadata_branch(path: &Path, branch: Option<&str>) -> Result<()> {
    if let Ok(Some(mut meta)) = load_session_metadata(path) {
        meta.last_updated_at = current_unix_timestamp();
        if let Some(b) = branch {
            meta.branch = b.to_string();
        }
        save_session_metadata(path, &meta)?;
    }
    Ok(())
}

/// Read the `branch` field from `~/.orangu/sessions/<session_id>/metadata`.
/// Returns `None` when the session doesn't exist, the file can't be read, or
/// the branch field is empty (the session was started outside a git repo).
pub(crate) fn load_session_branch(session_id: &str) -> Option<String> {
    let path = session_dir_path(session_id).ok()?;
    let meta = load_session_metadata(&path.join("metadata"))
        .ok()
        .flatten()?;
    if meta.branch.is_empty() {
        None
    } else {
        Some(meta.branch)
    }
}

pub(crate) fn find_session_for_workspace_branch(workspace: &str, branch: &str) -> Option<String> {
    let sessions_dir = home::home_dir()?.join(SESSIONS_DIRECTORY);
    if !sessions_dir.exists() {
        return None;
    }
    let mut candidates: Vec<(String, u64)> = Vec::new();
    for entry in std::fs::read_dir(&sessions_dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(uuid) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        let Some(meta) = load_session_metadata(&path.join("metadata")).ok().flatten() else {
            continue;
        };
        if meta.workspace != workspace {
            continue;
        }
        if meta.branch != branch {
            continue;
        }
        let has_messages = path
            .join("messages")
            .metadata()
            .map(|m| m.len() > 2)
            .unwrap_or(false);
        if !has_messages {
            continue;
        }
        candidates.push((uuid, meta.last_updated_at));
    }
    if candidates.len() == 1 {
        Some(candidates.remove(0).0)
    } else {
        None
    }
}

pub(crate) fn is_ephemeral_branch(branch: &str) -> bool {
    matches!(branch, "" | "main" | "master")
}

pub(crate) fn delete_session_dir(session_dir: &Path) {
    let _ = std::fs::remove_dir_all(session_dir);
}

/// UUIDs of sessions whose recorded workspace path contains `filter`. Used to
/// decide whether a `/session <workspace>` argument uniquely identifies a
/// session (switch to it) or matches several (list them).
pub(crate) fn sessions_matching_workspace(filter: &str) -> Result<Vec<String>> {
    let sessions_dir = {
        let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
        home.join(SESSIONS_DIRECTORY)
    };
    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }
    let mut matches: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&sessions_dir).with_context(|| {
        format!(
            "failed to read sessions directory {}",
            sessions_dir.display()
        )
    })? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(uuid) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        let Some(meta) = load_session_metadata(&path.join("metadata")).ok().flatten() else {
            continue;
        };
        if meta.workspace.contains(filter) {
            matches.push(uuid);
        }
    }
    Ok(matches)
}

pub(crate) fn list_sessions_output(
    workspace_filter: Option<&str>,
    active_session: &str,
) -> Result<String> {
    let sessions_dir = {
        let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
        home.join(SESSIONS_DIRECTORY)
    };

    if !sessions_dir.exists() {
        return Ok("No sessions found.".to_string());
    }

    let mut entries: Vec<(String, Option<SessionMetadata>, usize)> = Vec::new();

    for entry in std::fs::read_dir(&sessions_dir).with_context(|| {
        format!(
            "failed to read sessions directory {}",
            sessions_dir.display()
        )
    })? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let uuid = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        let meta = load_session_metadata(&path.join("metadata")).ok().flatten();
        if let Some(filter) = workspace_filter
            && !meta
                .as_ref()
                .map(|m| m.workspace.contains(filter))
                .unwrap_or(false)
        {
            continue;
        }
        let cmd_count = load_history(&path.join("history"))
            .unwrap_or_default()
            .len();
        entries.push((uuid, meta, cmd_count));
    }

    if entries.is_empty() {
        return Ok("No sessions found.".to_string());
    }

    entries.sort_by_key(|e| std::cmp::Reverse(e.1.as_ref().map(|m| m.started_at).unwrap_or(0)));

    // Build every cell first so column widths can be sized to the widest value
    // across all sessions, keeping the columns aligned regardless of content.
    struct Row {
        is_active: bool,
        uuid: String,
        started: String,
        last: String,
        cmds: String,
        branch: String,
        workspace: String,
    }
    let rows: Vec<Row> = entries
        .iter()
        .map(|(uuid, meta, cmd_count)| Row {
            is_active: *uuid == active_session,
            uuid: uuid.clone(),
            started: meta
                .as_ref()
                .map(|m| format_unix_timestamp(m.started_at))
                .unwrap_or_else(|| "-".to_string()),
            last: meta
                .as_ref()
                .map(|m| format_unix_timestamp(m.last_updated_at))
                .unwrap_or_else(|| "-".to_string()),
            cmds: cmd_count.to_string(),
            branch: meta
                .as_ref()
                .filter(|m| !m.branch.is_empty())
                .map(|m| m.branch.clone())
                .unwrap_or_else(|| "-".to_string()),
            workspace: meta
                .as_ref()
                .filter(|m| !m.workspace.is_empty())
                .map(|m| m.workspace.clone())
                .unwrap_or_else(|| "-".to_string()),
        })
        .collect();

    let col_width = |header: &str, value: &dyn Fn(&Row) -> &str| {
        rows.iter()
            .map(|r| value(r).chars().count())
            .chain(std::iter::once(header.chars().count()))
            .max()
            .unwrap_or(0)
    };
    let w_uuid = col_width("UUID", &|r| &r.uuid);
    let w_started = col_width("STARTED", &|r| &r.started);
    let w_last = col_width("LAST", &|r| &r.last);
    let w_cmds = col_width("CMDS", &|r| &r.cmds);
    let w_branch = col_width("BRANCH", &|r| &r.branch);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{:<6}  {:<w_uuid$}  {:<w_started$}  {:<w_last$}  {:>w_cmds$}  {:<w_branch$}  {}",
        "ACTIVE", "UUID", "STARTED", "LAST", "CMDS", "BRANCH", "WORKSPACE"
    ));
    for row in &rows {
        // The dot is a single visible glyph; pad manually to the 6-char ACTIVE
        // header width so the surrounding ANSI codes don't skew alignment.
        let dot = if row.is_active {
            FEEDBACK_OK
        } else {
            FEEDBACK_ERR
        };
        lines.push(format!(
            "{dot}       {:<w_uuid$}  {:<w_started$}  {:<w_last$}  {:>w_cmds$}  {:<w_branch$}  {}",
            row.uuid, row.started, row.last, row.cmds, row.branch, row.workspace
        ));
    }
    Ok(lines.join("\n"))
}

/// Render every stored session as a plain `SESSION  WORKSPACE  BRANCH  DATE`
/// table, columns sized to the widest value, for `orangu -l|--list`. DATE is the
/// session's last-updated timestamp. Sessions are listed newest-first by start
/// time. Always ends with a trailing newline.
pub(crate) fn list_all_sessions_output() -> Result<String> {
    let sessions_dir = {
        let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
        home.join(SESSIONS_DIRECTORY)
    };

    if !sessions_dir.exists() {
        return Ok("No sessions found.\n".to_string());
    }

    let mut entries: Vec<(String, Option<SessionMetadata>)> = Vec::new();
    for entry in std::fs::read_dir(&sessions_dir).with_context(|| {
        format!(
            "failed to read sessions directory {}",
            sessions_dir.display()
        )
    })? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(uuid) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        let meta = load_session_metadata(&path.join("metadata")).ok().flatten();
        entries.push((uuid, meta));
    }

    if entries.is_empty() {
        return Ok("No sessions found.\n".to_string());
    }

    entries.sort_by_key(|e| std::cmp::Reverse(e.1.as_ref().map(|m| m.started_at).unwrap_or(0)));

    struct Row {
        session: String,
        workspace: String,
        branch: String,
        date: String,
    }
    let rows: Vec<Row> = entries
        .iter()
        .map(|(uuid, meta)| Row {
            session: uuid.clone(),
            workspace: meta
                .as_ref()
                .filter(|m| !m.workspace.is_empty())
                .map(|m| m.workspace.clone())
                .unwrap_or_else(|| "-".to_string()),
            branch: meta
                .as_ref()
                .filter(|m| !m.branch.is_empty())
                .map(|m| m.branch.clone())
                .unwrap_or_else(|| "-".to_string()),
            date: meta
                .as_ref()
                .map(|m| format_unix_timestamp_human(m.last_updated_at))
                .unwrap_or_else(|| "-".to_string()),
        })
        .collect();

    let col_width = |header: &str, value: &dyn Fn(&Row) -> &str| {
        rows.iter()
            .map(|r| value(r).chars().count())
            .chain(std::iter::once(header.chars().count()))
            .max()
            .unwrap_or(0)
    };
    let w_session = col_width("SESSION", &|r| &r.session);
    let w_workspace = col_width("WORKSPACE", &|r| &r.workspace);
    let w_branch = col_width("BRANCH", &|r| &r.branch);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{:<w_session$}  {:<w_workspace$}  {:<w_branch$}  {}",
        "SESSION", "WORKSPACE", "BRANCH", "DATE"
    ));
    for row in &rows {
        lines.push(format!(
            "{:<w_session$}  {:<w_workspace$}  {:<w_branch$}  {}",
            row.session, row.workspace, row.branch, row.date
        ));
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

pub(crate) fn prune_sessions_output(target: &PruneTarget, active_session: &str) -> Result<String> {
    let sessions_dir = {
        let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
        home.join(SESSIONS_DIRECTORY)
    };

    if !sessions_dir.exists() {
        return Ok("No sessions found.".to_string());
    }

    let now = current_unix_timestamp();

    let mut removed: Vec<String> = Vec::new();
    let mut skipped_active = false;

    for entry in std::fs::read_dir(&sessions_dir).with_context(|| {
        format!(
            "failed to read sessions directory {}",
            sessions_dir.display()
        )
    })? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let uuid = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        let should_remove = match target {
            PruneTarget::All => true,
            PruneTarget::Uuid(id) => uuid == *id,
            PruneTarget::Workspace(filter) => {
                let meta = load_session_metadata(&path.join("metadata")).ok().flatten();
                meta.map(|m| m.workspace.contains(filter.as_str()))
                    .unwrap_or(false)
            }
            PruneTarget::OlderThan(days) => {
                let threshold = days * 86400;
                let meta = load_session_metadata(&path.join("metadata")).ok().flatten();
                meta.map(|m| now.saturating_sub(m.last_updated_at) >= threshold)
                    .unwrap_or(false)
            }
        };

        if !should_remove {
            continue;
        }

        if uuid == active_session {
            skipped_active = true;
            continue;
        }

        std::fs::remove_dir_all(&path)
            .with_context(|| format!("failed to remove session directory {}", path.display()))?;
        removed.push(uuid);
    }

    if removed.is_empty() && !skipped_active {
        return Ok("No matching sessions found.".to_string());
    }

    let mut lines: Vec<String> = Vec::new();
    for uuid in &removed {
        lines.push(format!("Removed: {uuid}"));
    }
    if skipped_active {
        lines.push(format!("Skipped active session: {active_session}"));
    }
    Ok(lines.join("\n"))
}

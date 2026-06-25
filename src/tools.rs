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

use crate::llm::{FunctionDefinition, ToolDefinition};
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
};
use tokio::{process::Command, time::Duration};
use walkdir::WalkDir;

#[derive(Clone)]
pub struct ToolExecutor {
    workspace: PathBuf,
    http_client: reqwest::Client,
    tool_duration: Arc<Mutex<std::time::Duration>>,
    compression_enabled: bool,
    read_only: bool,
    context_cache: Arc<Mutex<crate::context::ContextCache>>,
    pub compression_metrics: Arc<Mutex<crate::compression::CompressionMetrics>>,
    auto_downsample_lines: usize,
    diff_file_cap: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct ReadFileRequest {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
    /// Optional read mode:
    /// - `"full"` (default) — return the whole file (or the requested line range)
    /// - `"signatures"` — return only public item signatures (pub fn, pub struct,
    ///   pub enum, pub trait, impl blocks, doc comments), stripping function bodies
    /// - `"map"` — return a one-line-per-item structural overview (module-level
    ///   items only, no bodies or doc comments)
    mode: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EditFileRequest {
    path: String,
    old_text: String,
    new_text: String,
    replace_all: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ListDirectoryRequest {
    path: Option<String>,
    max_depth: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FetchUrlRequest {
    url: String,
    max_chars: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ShellCommandRequest {
    command: String,
    cwd: Option<String>,
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ShellCommandResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

impl ToolExecutor {
    pub fn new(workspace: &Path) -> Self {
        Self::with_config(workspace, true, 300, 20)
    }

    pub fn new_read_only(workspace: &Path) -> Self {
        let mut executor = Self::with_config(workspace, true, 300, 20);
        executor.read_only = true;
        executor
    }

    pub fn with_config(
        workspace: &Path,
        compression_enabled: bool,
        auto_downsample_lines: usize,
        diff_file_cap: usize,
    ) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            http_client: Client::new(),
            tool_duration: Arc::new(Mutex::new(std::time::Duration::ZERO)),
            compression_enabled,
            read_only: false,
            context_cache: Arc::new(Mutex::new(crate::context::ContextCache::new())),
            compression_metrics: Arc::new(Mutex::new(
                crate::compression::CompressionMetrics::default(),
            )),
            auto_downsample_lines,
            diff_file_cap,
        }
    }

    pub fn total_tool_duration(&self) -> std::time::Duration {
        self.tool_duration.lock().map(|d| *d).unwrap_or_default()
    }

    pub fn diff_file_cap(&self) -> usize {
        self.diff_file_cap
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![tool(
            "read_file",
            "Read a text file from disk, optionally returning only a line range. \
                 When investigating unfamiliar code, leave `start_line` and `end_line` empty \
                 to read the entire file. Use `mode` to get structural overviews.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "mode": {
                        "type": "string",
                        "enum": ["full", "signatures", "map"]
                    }
                },
                "required": ["path"]
            }),
        )];

        if !self.read_only {
            defs.push(tool(
                "edit_file",
                "Edit a file on disk in the current workspace by replacing old_text with new_text. If the file does not exist it is created (mode 0644) with new_text as its contents.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean"}
                    },
                    "required": ["path", "old_text", "new_text"]
                }),
            ));
            defs.push(tool(
                "explore_repository",
                "Spin up an independent explorer subagent to find relevant files and line ranges. Use this for broad searches so you don't pollute your own context. It returns a <final_answer> block with citations.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
            ));
        }

        defs.push(tool(
            "list_directory",
            "List files and directories under the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"}
                }
            }),
        ));
        defs.push(tool(
            "fetch_url",
            "Fetch an external URL and return readable text content.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"}
                },
                "required": ["url"]
            }),
        ));
        defs.push(tool(
            "run_shell_command",
            "Run a shell command inside the workspace. Recognized high-volume output may be compressed before truncation to preserve the most useful lines.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"}
                },
                "required": ["command"]
            }),
        ));

        defs
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Expose the session-local file context cache for persistence across runs.
    pub fn context_cache(&self) -> &std::sync::Mutex<crate::context::ContextCache> {
        &self.context_cache
    }

    pub async fn execute(&self, name: &str, arguments: &Map<String, Value>) -> Result<String> {
        let start = std::time::Instant::now();
        let result = match name {
            "read_file" => self.read_file(arguments).await,
            "edit_file" => {
                if self.read_only {
                    Err(anyhow::anyhow!("tool not available in read-only mode"))
                } else {
                    self.edit_file(arguments).await
                }
            }
            "explore_repository" => {
                if self.read_only {
                    Err(anyhow::anyhow!("tool not available in read-only mode"))
                } else {
                    crate::explorer::run_explorer_subagent(&self.workspace, arguments).await
                }
            }
            "list_directory" => self.list_directory(arguments).await,
            "fetch_url" => self.fetch_url(arguments).await,
            "run_shell_command" => self.run_shell_command(arguments).await,
            _ => Err(anyhow!("unknown tool '{}'", name)),
        };
        if let Ok(mut d) = self.tool_duration.lock() {
            *d += start.elapsed();
        }
        result
    }

    async fn read_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ReadFileRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let path = self.resolve_workspace_path(&args.path)?;
        let mut content = fs::read_to_string(&path)?;

        redact_secrets(&mut content);

        if self.compression_enabled
            && args.start_line.is_none()
            && args.end_line.is_none()
            && let Ok(metadata) = fs::metadata(&path)
        {
            let mut cache = self.context_cache.lock().unwrap();
            let fingerprint = cache.fingerprint(&content, &metadata);
            let cache_result = cache.check_file(&path, &content, &fingerprint);
            if let crate::context::CacheResult::Hit { fingerprint } = cache_result {
                return Ok(crate::context::format_cache_stub(
                    &args.path,
                    metadata.len(),
                    &fingerprint,
                ));
            }
            cache.record_read(&path, fingerprint);
        }

        if self.compression_enabled
            && args.mode.is_none()
            && args.start_line.is_none()
            && args.end_line.is_none()
            && self.auto_downsample_lines > 0
            && content.lines().count() > self.auto_downsample_lines
        {
            let mut downsampled = extract_signatures(&content);
            if downsampled != content {
                downsampled.push_str(&format!("\n[Note: This file exceeds {} lines and has been automatically downsampled to 'signatures' mode. Use start_line and end_line bounds to read specific full bodies.]\n", self.auto_downsample_lines));
                return Ok(downsampled);
            }
        }

        Ok(match args.mode.as_deref() {
            Some("signatures") => extract_signatures(&content),
            Some("map") => extract_map(&content),
            _ => render_file_slice(&content, args.start_line, args.end_line),
        })
    }

    async fn edit_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: EditFileRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let path = self.resolve_workspace_path(&args.path)?;
        let (original, created) = match fs::read_to_string(&path) {
            Ok(content) => (content, false),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => (String::new(), true),
            Err(err) => return Err(err.into()),
        };

        let updated = apply_edit(
            &original,
            &args.old_text,
            &args.new_text,
            args.replace_all.unwrap_or(false),
            created,
        )?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &updated)?;
        if created {
            #[cfg(unix)]
            fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
        }

        Ok(json!({
            "path": path,
            "created": created,
            "updated": true,
            "original_bytes": original.len(),
            "new_bytes": updated.len()
        })
        .to_string())
    }

    async fn list_directory(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ListDirectoryRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let relative = args.path.unwrap_or_else(|| ".".to_string());
        let path = self.resolve_workspace_path(&relative)?;
        let max_depth = args.max_depth.unwrap_or(2);

        let entries = WalkDir::new(&path)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| {
                let kind = if entry.file_type().is_dir() {
                    "dir"
                } else {
                    "file"
                };
                let display_path = entry
                    .path()
                    .strip_prefix(&self.workspace)
                    .unwrap_or(entry.path())
                    .display()
                    .to_string();
                format!("{kind}\t{display_path}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(entries)
    }

    async fn fetch_url(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: FetchUrlRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let response = self.http_client.get(&args.url).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!(
                "request failed for {} with status {}",
                args.url,
                status
            ));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response.text().await?;
        let max_chars = args.max_chars.unwrap_or(20_000);
        if content_type.contains("html") {
            let rendered = html2text::from_read(body.as_bytes(), 120)?;
            Ok(truncate_text(&rendered, max_chars))
        } else {
            Ok(truncate_text(&body, max_chars))
        }
    }

    async fn run_shell_command(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: ShellCommandRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let cwd = match args.cwd {
            Some(path) => self.resolve_workspace_path(&path)?,
            None => self.workspace.clone(),
        };
        let timeout = Duration::from_secs(args.timeout_seconds.unwrap_or(30));

        let mut child = Command::new("bash");
        child
            .arg("-lc")
            .arg(&args.command)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = tokio::time::timeout(timeout, child.output())
            .await
            .map_err(|_| anyhow!("command timed out after {:?}", timeout))??;

        let stdout_raw = String::from_utf8_lossy(&output.stdout);
        let stderr_raw = String::from_utf8_lossy(&output.stderr);

        let (stdout_compressed, stderr_compressed) = if self.compression_enabled {
            let (compressed_out, out_stats) = crate::compression::compress_shell_output_with_stats(
                &args.command,
                &stdout_raw,
                self.diff_file_cap,
            );
            let (compressed_err, err_stats) = crate::compression::compress_shell_output_with_stats(
                &args.command,
                &stderr_raw,
                self.diff_file_cap,
            );

            let mut compressed_out = compressed_out;
            let mut compressed_err = compressed_err;

            if out_stats.pattern_matched.as_deref() == Some("generic")
                && out_stats.compressed_lines < out_stats.original_lines
            {
                let tmp_dir = self.workspace.join(".orangu/tmp");
                let _ = tokio::fs::create_dir_all(&tmp_dir).await;
                let log_path = tmp_dir.join("cmd_stdout.log");
                if tokio::fs::write(&log_path, stdout_raw.as_bytes())
                    .await
                    .is_ok()
                {
                    compressed_out.push_str(&format!(
                        "\n\n[Raw stdout diverted to {} due to size. Read this file for missing details.]",
                        log_path.display()
                    ));
                }
            }

            if err_stats.pattern_matched.as_deref() == Some("generic")
                && err_stats.compressed_lines < err_stats.original_lines
            {
                let tmp_dir = self.workspace.join(".orangu/tmp");
                let _ = tokio::fs::create_dir_all(&tmp_dir).await;
                let log_path = tmp_dir.join("cmd_stderr.log");
                if tokio::fs::write(&log_path, stderr_raw.as_bytes())
                    .await
                    .is_ok()
                {
                    compressed_err.push_str(&format!(
                        "\n\n[Raw stderr diverted to {} due to size. Read this file for missing details.]",
                        log_path.display()
                    ));
                }
            }

            if let Ok(mut metrics) = self.compression_metrics.lock() {
                metrics.record(&out_stats);
                metrics.record(&err_stats);
            }
            (compressed_out, compressed_err)
        } else {
            (stdout_raw.to_string(), stderr_raw.to_string())
        };

        let result = ShellCommandResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: truncate_text(&stdout_compressed, 20_000),
            stderr: truncate_text(&stderr_compressed, 20_000),
        };
        Ok(serde_json::to_string_pretty(&result)?)
    }

    fn resolve_workspace_path(&self, raw_path: &str) -> Result<PathBuf> {
        resolve_workspace_path(&self.workspace, raw_path)
    }
}

fn tool(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

pub fn apply_edit(
    original: &str,
    old_text: &str,
    new_text: &str,
    replace_all: bool,
    created: bool,
) -> Result<String> {
    if created {
        return Ok(new_text.to_string());
    }

    if old_text.is_empty() {
        return Ok(new_text.to_string());
    }

    if !original.contains(old_text) {
        return Err(anyhow!("old_text was not found in the file"));
    }

    let updated = if replace_all {
        original.replace(old_text, new_text)
    } else {
        original.replacen(old_text, new_text, 1)
    };

    Ok(updated)
}

fn render_file_slice(content: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    let start = start_line.unwrap_or(1);
    let end = end_line.unwrap_or(usize::MAX);

    content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line_no = index + 1;
            (line_no >= start && line_no <= end).then(|| format!("{line_no}. {line}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn resolve_workspace_path(workspace: &Path, raw_path: &str) -> Result<PathBuf> {
    let candidate = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        workspace.join(raw_path)
    };
    let normalized = normalize_path(&candidate);
    let normalized_workspace = normalize_path(workspace);
    if !normalized.starts_with(&normalized_workspace) {
        return Err(anyhow!("path escapes the configured workspace"));
    }
    Ok(normalized)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(part) => result.push(part),
        }
    }
    result
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated = text.chars().take(max_chars).collect::<String>();
    format!("{truncated}\n\n[truncated]")
}

/// Extract public item signatures from file content, stripping function bodies.
///
/// Keeps: doc comments (`///`, `//!`), `pub fn`, `pub struct`, `pub enum`,
/// `pub trait`, `pub type`, `pub const`, `pub static`, `impl` blocks (header
/// only), `mod` declarations, `use` statements, and attribute lines (`#[…]`).
/// Strips: private items and function bodies (lines between `{` and matching `}`).
///
/// This is a line-based approximation — no full AST — suitable for v1. It
/// works well for idiomatic Rust where bodies are indented and opening braces
/// are on the same line as the signature.
fn extract_signatures(content: &str) -> String {
    let mut result: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut in_body = false;
    let mut last_was_blank = false;

    // Patterns that mark the START of a signature line we want to keep.
    fn is_signature_line(line: &str) -> bool {
        let t = line.trim_start();
        t.starts_with("pub fn ")
            || t.starts_with("pub async fn ")
            || t.starts_with("pub unsafe fn ")
            || t.starts_with("pub struct ")
            || t.starts_with("pub enum ")
            || t.starts_with("pub trait ")
            || t.starts_with("pub type ")
            || t.starts_with("pub const ")
            || t.starts_with("pub static ")
            || t.starts_with("pub mod ")
            || t.starts_with("pub use ")
            || t.starts_with("pub(crate) fn ")
            || t.starts_with("pub(crate) struct ")
            || t.starts_with("pub(crate) enum ")
            || t.starts_with("pub(crate) trait ")
            || t.starts_with("pub(crate) type ")
            || t.starts_with("pub(crate) const ")
            || t.starts_with("pub(crate) mod ")
            || t.starts_with("impl ")
            || t.starts_with("mod ")
            || t.starts_with("use ")
            || t.starts_with("//!")
            || t.starts_with("///")
            || t.starts_with("#[")
            || t.starts_with("#![")
    }

    for line in content.lines() {
        let trimmed = line.trim();

        // Track brace depth to skip bodies.
        let opens = line.chars().filter(|&c| c == '{').count() as i32;
        let closes = line.chars().filter(|&c| c == '}').count() as i32;

        if in_body {
            depth += opens - closes;
            if depth <= 0 {
                in_body = false;
                depth = 0;
                // Emit a closing marker so the reader can see blocks end.
                result.push("    // ...".to_string());
            }
            continue;
        }

        if is_signature_line(line) {
            // Suppress multiple consecutive blank lines before signatures.
            if !trimmed.is_empty() && last_was_blank {
                result.push(String::new());
            }
            result.push(line.to_string());

            // If this line opens a body brace (and doesn't close it on the
            // same line), enter body-skip mode.
            let net = opens - closes;
            if net > 0 {
                in_body = true;
                depth = net;
            }
        }

        last_was_blank = trimmed.is_empty();
    }

    if result.is_empty() {
        // Fallback: file may not be Rust or has no public items — return as-is.
        return content.to_string();
    }

    format!(
        "[signatures mode — bodies stripped]\n\n{}\n",
        result.join("\n")
    )
}

/// Return a one-line-per-item structural map of the file. Even more compact
/// than `signatures` — only top-level item headers, no doc comments, no
/// attribute lines, no `use` statements.
fn extract_map(content: &str) -> String {
    let mut items: Vec<String> = Vec::new();
    let mut depth: i32 = 0;

    fn is_map_item(line: &str) -> bool {
        let t = line.trim_start();
        (t.starts_with("pub ") || t.starts_with("impl ") || t.starts_with("mod "))
            && !t.starts_with("pub use ")
            && !t.starts_with("pub(crate) use ")
    }

    for line in content.lines() {
        let opens = line.chars().filter(|&c| c == '{').count() as i32;
        let closes = line.chars().filter(|&c| c == '}').count() as i32;

        // Only capture top-level items (depth == 0 before this line).
        if depth == 0 && is_map_item(line) {
            // Trim the body if it starts on the same line: keep only up to `{`.
            let display = if let Some(brace_pos) = line.find('{') {
                line[..brace_pos].trim_end().to_string() + " { ... }"
            } else {
                line.trim_end().to_string()
            };
            items.push(display);
        }

        depth = (depth + opens - closes).max(0);
    }

    if items.is_empty() {
        return content.to_string();
    }

    format!(
        "[map mode — top-level items only]\n\n{}\n",
        items.join("\n")
    )
}

fn redact_secrets(content: &mut String) {
    use regex::Regex;
    use std::sync::OnceLock;

    static SECRET_REGEX: OnceLock<Regex> = OnceLock::new();
    let re = SECRET_REGEX.get_or_init(|| {
        Regex::new(r"(?P<prefix>ghp_[a-zA-Z0-9]{36}|sk-ant-[a-zA-Z0-9_-]{30,}|AKIA[0-9A-Z]{16})")
            .unwrap()
    });

    if re.is_match(content) {
        *content = re.replace_all(content, "[REDACTED_SECRET]").to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_single_edit() {
        let updated = apply_edit("hello world", "world", "orangu", false, false).unwrap();
        assert_eq!(updated, "hello orangu");
    }

    #[test]
    fn create_new_file_content() {
        let updated = apply_edit("", "", "new content", false, true).unwrap();
        assert_eq!(updated, "new content");
    }

    #[tokio::test]
    async fn edit_file_creates_missing_file_with_0644() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("sub/new.txt"));
        args.insert("old_text".into(), json!(""));
        args.insert("new_text".into(), json!("hello orangu"));
        executor.edit_file(&args).await.unwrap();

        let created = workspace.path().join("sub/new.txt");
        assert_eq!(fs::read_to_string(&created).unwrap(), "hello orangu");
        #[cfg(unix)]
        {
            let mode = fs::metadata(&created).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o644);
        }
    }

    #[tokio::test]
    async fn edit_file_modifies_existing_file() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("existing.txt");
        fs::write(&path, "hello world").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("existing.txt"));
        args.insert("old_text".into(), json!("world"));
        args.insert("new_text".into(), json!("orangu"));
        executor.edit_file(&args).await.unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "hello orangu");
    }

    #[tokio::test]
    async fn read_file_returns_cache_stub_on_repeated_unchanged_full_read() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let first = executor.read_file(&args).await.unwrap();
        assert!(first.contains("1. one"));
        assert!(first.contains("2. two"));

        let second = executor.read_file(&args).await.unwrap();
        assert!(second.starts_with("[cached] README.md is unchanged"));
        assert!(second.contains("start_line/end_line"));
    }

    #[tokio::test]
    async fn read_file_returns_full_content_again_after_change() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let _ = executor.read_file(&args).await.unwrap();
        let _ = executor.read_file(&args).await.unwrap();

        fs::write(&path, "one\ntwo\nthree\n").unwrap();

        let changed = executor.read_file(&args).await.unwrap();
        assert!(changed.contains("1. one"));
        assert!(changed.contains("2. two"));
        assert!(changed.contains("3. three"));
        assert!(!changed.starts_with("[cached]"));
    }

    #[tokio::test]
    async fn run_shell_command_compresses_cargo_test_success_noise() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 3 tests\\ntest a ... ok\\ntest b ... ok\\ntest c ... ok\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("... (3 tests passed)"));
        assert!(!rendered.contains("test a ... ok"));
        assert!(!rendered.contains("test b ... ok"));
        assert!(!rendered.contains("test c ... ok"));
    }

    #[tokio::test]
    async fn run_shell_command_keeps_cargo_test_failures_visible() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 3 tests\\ntest a ... ok\\ntest b ... FAILED\\nfailures:\\n---- test b stdout ----\\npanicked at boom\\ntest result: FAILED. 1 passed; 1 failed\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::new(workspace.path());

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("test b ... FAILED"));
        assert!(rendered.contains("failures:"));
        assert!(rendered.contains("panicked at boom"));
        assert!(!rendered.contains("test a ... ok"));
    }

    #[tokio::test]
    async fn read_file_without_compression_returns_full_content_on_repeat() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("README.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        let executor = ToolExecutor::with_config(workspace.path(), false, 300, 20);

        let mut args = Map::new();
        args.insert("path".into(), json!("README.md"));

        let first = executor.read_file(&args).await.unwrap();
        let second = executor.read_file(&args).await.unwrap();

        assert_eq!(first, second);
        assert!(!second.starts_with("[cached]"));
    }

    #[tokio::test]
    async fn run_shell_command_without_compression_keeps_raw_output() {
        let workspace = tempfile::tempdir().unwrap();
        let cargo = workspace.path().join("cargo");
        fs::write(
            &cargo,
            "#!/usr/bin/env bash\nprintf 'running 2 tests\\ntest a ... ok\\ntest b ... ok\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executor = ToolExecutor::with_config(workspace.path(), false, 300, 20);

        let mut args = Map::new();
        args.insert("command".into(), json!("./cargo test"));

        let rendered = executor.run_shell_command(&args).await.unwrap();
        assert!(rendered.contains("test a ... ok"));
        assert!(rendered.contains("test b ... ok"));
        assert!(!rendered.contains("tests passed"));
    }

    #[test]
    fn rejects_path_escape() {
        let workspace = PathBuf::from("/tmp/workspace");
        let err = resolve_workspace_path(&workspace, "../outside").unwrap_err();
        assert!(err.to_string().contains("escapes"));
    }

    #[test]
    fn signatures_mode_strips_bodies_keeps_pub_fn() {
        let src = "
pub fn hello() {
    println!(\"hi\");
}

fn private_fn() {
    // private body
}

/// A doc comment.
pub struct Foo {
    pub x: i32,
}
";
        let result = extract_signatures(src);
        assert!(
            result.contains("[signatures mode"),
            "should have mode header"
        );
        assert!(result.contains("pub fn hello()"), "should keep pub fn");
        assert!(!result.contains("println!"), "should strip body");
        assert!(!result.contains("private_fn"), "should skip private fn");
        assert!(
            result.contains("/// A doc comment."),
            "should keep doc comment"
        );
        assert!(result.contains("pub struct Foo"), "should keep pub struct");
    }

    #[test]
    fn signatures_mode_fallback_for_no_pub_items() {
        let src = "fn private() {}";
        let result = extract_signatures(src);
        // Falls back to returning content as-is when nothing matches.
        assert_eq!(result, src);
    }

    #[test]
    fn map_mode_only_top_level_items() {
        let src = "
pub struct Foo { x: i32 }

impl Foo {
    pub fn method(&self) {}
}

pub fn standalone() {}
";
        let result = extract_map(src);
        assert!(result.contains("[map mode"), "should have mode header");
        assert!(result.contains("pub struct Foo"), "top-level struct");
        assert!(result.contains("impl Foo"), "impl block header");
        assert!(result.contains("pub fn standalone()"), "top-level fn");
        // method() is inside impl, should NOT appear as separate item
        assert!(
            !result.contains("pub fn method"),
            "nested fn should not appear"
        );
    }

    #[test]
    fn grep_context_is_compact_under_limit() {
        use crate::compression::prepare_llm_grep_context;
        let output = "src/foo.rs:10:    pub fn foo() {}\nsrc/bar.rs:20:    pub fn bar() {}";
        let ctx = prepare_llm_grep_context("fn foo", output, true);
        // Under 40 matches — all should appear, no omission note.
        assert!(ctx.content.contains("src/foo.rs"));
        assert!(ctx.note.is_none());
    }

    #[test]
    fn grep_context_truncates_over_limit() {
        use crate::compression::prepare_llm_grep_context;
        // Generate 50 fake match lines.
        let output: String = (0..50)
            .map(|i| format!("src/x.rs:{i}: fn item_{i}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");
        let ctx = prepare_llm_grep_context("item", &output, true);
        assert!(ctx.note.is_some(), "should have truncation note");
        let note = ctx.note.unwrap();
        assert!(note.contains("50 matches found"), "should mention counts");
        assert!(note.contains("first 40"), "should mention counts");
    }
}

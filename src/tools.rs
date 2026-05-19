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
    http_client: Client,
    tool_duration: Arc<Mutex<std::time::Duration>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ReadFileRequest {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EditFileRequest {
    path: String,
    old_text: String,
    new_text: String,
    replace_all: Option<bool>,
    create_if_missing: Option<bool>,
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
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            http_client: Client::new(),
            tool_duration: Arc::new(Mutex::new(std::time::Duration::ZERO)),
        }
    }

    pub fn total_tool_duration(&self) -> std::time::Duration {
        self.tool_duration.lock().map(|d| *d).unwrap_or_default()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            tool(
                "read_file",
                "Read a text file from disk, optionally returning only a line range.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"}
                    },
                    "required": ["path"]
                }),
            ),
            tool(
                "edit_file",
                "Edit a file on disk in the current workspace by replacing old_text with new_text.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "create_if_missing": {"type": "boolean"}
                    },
                    "required": ["path", "old_text", "new_text"]
                }),
            ),
            tool(
                "list_directory",
                "List files and directories under the workspace.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_depth": {"type": "integer"}
                    }
                }),
            ),
            tool(
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
            ),
            tool(
                "run_shell_command",
                "Run a shell command inside the workspace.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "integer"}
                    },
                    "required": ["command"]
                }),
            ),
        ]
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub async fn execute(&self, name: &str, arguments: &Map<String, Value>) -> Result<String> {
        let start = std::time::Instant::now();
        let result = match name {
            "read_file" => self.read_file(arguments).await,
            "edit_file" => self.edit_file(arguments).await,
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
        let content = fs::read_to_string(&path)?;
        Ok(render_file_slice(&content, args.start_line, args.end_line))
    }

    async fn edit_file(&self, arguments: &Map<String, Value>) -> Result<String> {
        let args: EditFileRequest = serde_json::from_value(Value::Object(arguments.clone()))?;
        let path = self.resolve_workspace_path(&args.path)?;
        let create_if_missing = args.create_if_missing.unwrap_or(false);
        let original = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound && create_if_missing => {
                String::new()
            }
            Err(err) => return Err(err.into()),
        };

        let updated = apply_edit(
            &original,
            &args.old_text,
            &args.new_text,
            args.replace_all.unwrap_or(false),
            create_if_missing,
        )?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &updated)?;

        Ok(json!({
            "path": path,
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

        let result = ShellCommandResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: truncate_text(&String::from_utf8_lossy(&output.stdout), 20_000),
            stderr: truncate_text(&String::from_utf8_lossy(&output.stderr), 20_000),
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
    create_if_missing: bool,
) -> Result<String> {
    if original.is_empty() && create_if_missing {
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

    #[test]
    fn rejects_path_escape() {
        let workspace = PathBuf::from("/tmp/workspace");
        let err = resolve_workspace_path(&workspace, "../outside").unwrap_err();
        assert!(err.to_string().contains("escapes"));
    }
}

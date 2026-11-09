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

//! The approval panel an MCP tool call that asks for approval opens under
//! the live answer: what the call is, what it would change, and the
//! choices.

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::path::Path;

/// Lines of arguments or diff shown before `e` expands them.
const FOLDED_LINES: usize = 12;
/// Lines of diff kept at all: past this the panel says how many more.
const MAX_DIFF_LINES: usize = 400;

/// What the user chose for one call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalChoice {
    Once,
    /// Allow this tool for the rest of the session.
    Tool,
    /// Allow every tool of this server for the rest of the session.
    Server,
    Deny(Option<String>),
}

impl ApprovalChoice {
    /// The transcript's record of the choice.
    pub fn record(&self, tool: &str, server: Option<&str>) -> String {
        match self {
            ApprovalChoice::Once => format!("[MCP {tool}: allowed once]"),
            ApprovalChoice::Tool => format!("[MCP {tool}: allowed for this session]"),
            ApprovalChoice::Server => format!(
                "[MCP {tool}: every tool of {} allowed for this session]",
                server.unwrap_or("its server")
            ),
            ApprovalChoice::Deny(None) => format!("[MCP {tool}: denied]"),
            ApprovalChoice::Deny(Some(reason)) => format!("[MCP {tool}: denied — {reason}]"),
        }
    }
}

/// An MCP call waiting for the user.
#[derive(Clone)]
pub struct McpApprovalRequest {
    pub decision: std::sync::mpsc::Sender<ApprovalChoice>,
    /// The tool's exposed name, as the model called it.
    pub tool: String,
    pub server: Option<String>,
    /// The arguments as pretty-printed JSON.
    pub arguments: String,
    /// What the call would change in a workspace file, as a unified diff.
    pub diff: Option<String>,
    /// Whether the arguments and the diff show in full.
    pub expanded: bool,
    /// The reason being typed after `r`.
    pub reason: Option<String>,
}

/// What a key did to the panel.
#[derive(Debug, PartialEq, Eq)]
pub enum ApprovalKey {
    /// Not the panel's key: it goes on to the input line.
    Ignored,
    /// The panel changed: redraw it.
    Changed,
    Decided(ApprovalChoice),
}

impl McpApprovalRequest {
    pub fn new(
        decision: std::sync::mpsc::Sender<ApprovalChoice>,
        call: &orangu::llm::ToolCall,
        server: Option<String>,
        workspace: &Path,
    ) -> Self {
        let arguments = serde_json::to_string_pretty(&call.function.arguments)
            .unwrap_or_else(|_| "{}".to_string());
        Self {
            decision,
            tool: call.function.name.clone(),
            server,
            arguments,
            diff: proposed_diff(workspace, &call.function.arguments),
            expanded: false,
            reason: None,
        }
    }

    /// What changes the panel's drawing, to tell when to redraw it.
    pub fn view(&self) -> (bool, Option<String>) {
        (self.expanded, self.reason.clone())
    }

    /// The panel as Markdown, rendered under the live answer.
    pub fn panel(&self) -> String {
        let mut text = format!("**MCP tool call needs approval:** `{}`", self.tool);
        if let Some(server) = &self.server {
            text.push_str(&format!(" on server `{server}`"));
        }
        text.push_str("\n\n");
        let mut folded = false;
        for (language, body) in std::iter::once(("json", self.arguments.as_str()))
            .chain(self.diff.as_deref().map(|diff| ("diff", diff)))
        {
            let lines: Vec<&str> = body.lines().collect();
            let shown = if self.expanded {
                lines.len()
            } else {
                lines.len().min(FOLDED_LINES)
            };
            text.push_str(&format!(
                "```{language}\n{}\n```\n",
                lines[..shown].join("\n")
            ));
            if shown < lines.len() {
                folded = true;
                text.push_str(&format!("… {} more lines\n", lines.len() - shown));
            }
            text.push('\n');
        }
        match &self.reason {
            Some(reason) => text.push_str(&format!(
                "Reason for the model: {reason}▏\n\n`Enter` deny with this reason · `Esc` back"
            )),
            None => {
                text.push_str(
                    "`y` allow once · `t` always this tool · `s` always this server · `n` deny · `r` deny with a reason",
                );
                if folded || self.expanded {
                    text.push_str(if self.expanded {
                        " · `e` fold"
                    } else {
                        " · `e` show all"
                    });
                }
            }
        }
        text
    }

    /// The panel's answer to a key.
    pub fn key(&mut self, event: &Event) -> ApprovalKey {
        let Event::Key(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press | KeyEventKind::Repeat,
            ..
        }) = event
        else {
            return ApprovalKey::Ignored;
        };
        if modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) {
            return ApprovalKey::Ignored;
        }
        if let Some(reason) = &mut self.reason {
            return match code {
                KeyCode::Enter => {
                    let reason = reason.trim().to_string();
                    ApprovalKey::Decided(ApprovalChoice::Deny(
                        (!reason.is_empty()).then_some(reason),
                    ))
                }
                KeyCode::Esc => {
                    self.reason = None;
                    ApprovalKey::Changed
                }
                KeyCode::Backspace => {
                    reason.pop();
                    ApprovalKey::Changed
                }
                KeyCode::Char(ch) => {
                    reason.push(*ch);
                    ApprovalKey::Changed
                }
                _ => ApprovalKey::Changed,
            };
        }
        match code {
            KeyCode::Char('y' | 'Y') => ApprovalKey::Decided(ApprovalChoice::Once),
            KeyCode::Char('t' | 'T') => ApprovalKey::Decided(ApprovalChoice::Tool),
            KeyCode::Char('s' | 'S') if self.server.is_some() => {
                ApprovalKey::Decided(ApprovalChoice::Server)
            }
            KeyCode::Char('n' | 'N') => ApprovalKey::Decided(ApprovalChoice::Deny(None)),
            KeyCode::Char('r' | 'R') => {
                self.reason = Some(String::new());
                ApprovalKey::Changed
            }
            KeyCode::Char('e' | 'E') => {
                self.expanded = !self.expanded;
                ApprovalKey::Changed
            }
            _ => ApprovalKey::Ignored,
        }
    }
}

/// What a call that writes a workspace file would change, from the common
/// argument shapes: a path with the whole new content, a path with an old
/// and a new text, or a path with a list of such edits.
fn proposed_diff(
    workspace: &Path,
    arguments: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<String> {
    let text = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| arguments.get(*key).and_then(serde_json::Value::as_str))
    };
    let path = text(&["path", "file_path", "filePath", "file", "filename"])?;
    let target = orangu::tools::resolve_workspace_path(workspace, path).ok()?;
    let current = std::fs::read_to_string(&target).unwrap_or_default();

    let proposed = if let Some(content) = text(&["content", "contents", "new_content", "file_text"])
    {
        content.to_string()
    } else {
        let mut edits = Vec::new();
        if let (Some(old), Some(new)) = (
            text(&["old_string", "old_str", "oldText", "old_text"]),
            text(&["new_string", "new_str", "newText", "new_text"]),
        ) {
            edits.push((old.to_string(), new.to_string()));
        }
        if let Some(list) = arguments.get("edits").and_then(serde_json::Value::as_array) {
            for edit in list {
                let field = |keys: &[&str]| {
                    keys.iter()
                        .find_map(|key| edit.get(*key).and_then(serde_json::Value::as_str))
                };
                if let (Some(old), Some(new)) = (
                    field(&["oldText", "old_text", "old_string"]),
                    field(&["newText", "new_text", "new_string"]),
                ) {
                    edits.push((old.to_string(), new.to_string()));
                }
            }
        }
        if edits.is_empty() {
            return None;
        }
        let mut proposed = current.clone();
        for (old, new) in edits {
            proposed = proposed.replacen(&old, &new, 1);
        }
        proposed
    };
    unified_diff(path, &current, &proposed)
}

/// `git diff --no-index` of two texts, labelled with the workspace path.
fn unified_diff(path: &str, current: &str, proposed: &str) -> Option<String> {
    if current == proposed {
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "orangu-approval-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos())
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let (before, after) = (dir.join("a"), dir.join("b"));
    let written = std::fs::write(&before, current).and(std::fs::write(&after, proposed));
    let output = written.ok().and_then(|()| {
        std::process::Command::new("git")
            .args(["diff", "--no-index", "--no-color", "--"])
            .arg(&before)
            .arg(&after)
            .output()
            .ok()
    });
    let _ = std::fs::remove_dir_all(&dir);
    let output = String::from_utf8_lossy(&output?.stdout).into_owned();
    // The hunks only, under the workspace path instead of the scratch files.
    let mut lines = vec![format!("--- a/{path}"), format!("+++ b/{path}")];
    lines.extend(
        output
            .lines()
            .skip_while(|line| !line.starts_with("@@"))
            .map(str::to_string),
    );
    if lines.len() <= 2 {
        return None;
    }
    let total = lines.len();
    lines.truncate(MAX_DIFF_LINES);
    if total > MAX_DIFF_LINES {
        lines.push(format!("… {} more diff lines", total - MAX_DIFF_LINES));
    }
    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(arguments: serde_json::Value) -> orangu::llm::ToolCall {
        orangu::llm::ToolCall {
            id: "call".to_string(),
            tool_type: "function".to_string(),
            function: orangu::llm::ToolCallFunction {
                name: "fs__write_file".to_string(),
                arguments: serde_json::from_value(arguments).unwrap(),
            },
        }
    }

    fn press(request: &mut McpApprovalRequest, code: KeyCode) -> ApprovalKey {
        request.key(&Event::Key(KeyEvent::new(code, KeyModifiers::NONE)))
    }

    fn request(arguments: serde_json::Value, workspace: &Path) -> McpApprovalRequest {
        let (sender, _) = std::sync::mpsc::channel();
        McpApprovalRequest::new(sender, &call(arguments), Some("fs".to_string()), workspace)
    }

    #[test]
    fn each_choice_has_its_key() {
        let workspace = tempfile::tempdir().unwrap();
        let mut panel = request(json!({"query": "x"}), workspace.path());
        assert_eq!(
            press(&mut panel, KeyCode::Char('y')),
            ApprovalKey::Decided(ApprovalChoice::Once)
        );
        assert_eq!(
            press(&mut panel, KeyCode::Char('t')),
            ApprovalKey::Decided(ApprovalChoice::Tool)
        );
        assert_eq!(
            press(&mut panel, KeyCode::Char('s')),
            ApprovalKey::Decided(ApprovalChoice::Server)
        );
        assert_eq!(
            press(&mut panel, KeyCode::Char('n')),
            ApprovalKey::Decided(ApprovalChoice::Deny(None))
        );
        assert_eq!(press(&mut panel, KeyCode::Char('x')), ApprovalKey::Ignored);
    }

    #[test]
    fn r_takes_a_reason_for_the_model() {
        let workspace = tempfile::tempdir().unwrap();
        let mut panel = request(json!({"query": "x"}), workspace.path());
        assert_eq!(press(&mut panel, KeyCode::Char('r')), ApprovalKey::Changed);
        for ch in "use git".chars() {
            press(&mut panel, KeyCode::Char(ch));
        }
        assert!(panel.panel().contains("Reason for the model: use git"));
        assert_eq!(
            press(&mut panel, KeyCode::Enter),
            ApprovalKey::Decided(ApprovalChoice::Deny(Some("use git".to_string())))
        );

        // Esc goes back to the choices; an empty reason is a plain denial.
        press(&mut panel, KeyCode::Esc);
        assert_eq!(panel.reason, None);
        press(&mut panel, KeyCode::Char('r'));
        assert_eq!(
            press(&mut panel, KeyCode::Enter),
            ApprovalKey::Decided(ApprovalChoice::Deny(None))
        );
    }

    #[test]
    fn long_arguments_fold_until_e() {
        let workspace = tempfile::tempdir().unwrap();
        let many: serde_json::Map<String, serde_json::Value> =
            (0..30).map(|i| (format!("key{i:02}"), json!(i))).collect();
        let mut panel = request(serde_json::Value::Object(many), workspace.path());
        assert!(panel.panel().contains("more lines"));
        assert!(panel.panel().contains("`e` show all"));
        press(&mut panel, KeyCode::Char('e'));
        assert!(!panel.panel().contains("more lines"));
        assert!(panel.panel().contains("key29"));
    }

    #[test]
    fn a_write_shows_its_diff_against_the_file() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("notes.txt"), "one\ntwo\n").unwrap();

        let whole = request(
            json!({"path": "notes.txt", "content": "one\nthree\n"}),
            workspace.path(),
        );
        let diff = whole.diff.expect("a diff");
        assert!(
            diff.starts_with("--- a/notes.txt\n+++ b/notes.txt\n@@"),
            "{diff}"
        );
        assert!(
            diff.contains("\n-two") && diff.contains("\n+three"),
            "{diff}"
        );

        let edit = request(
            json!({"path": "notes.txt", "edits": [{"oldText": "one", "newText": "uno"}]}),
            workspace.path(),
        );
        assert!(edit.diff.expect("a diff").contains("+uno"));

        let new_file = request(
            json!({"path": "new.txt", "content": "hello\n"}),
            workspace.path(),
        );
        assert!(new_file.diff.expect("a diff").contains("+hello"));

        let read = request(json!({"path": "notes.txt"}), workspace.path());
        assert_eq!(read.diff, None, "a call that writes nothing has no diff");
        assert!(read.panel().contains("on server `fs`"));
    }
}

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

//! Hooks an agent file defines for its own runs, in Claude Code's shape:
//!
//! ```yaml
//! hooks:
//!   PreToolUse:
//!     - matcher: "Bash"
//!       hooks:
//!         - type: command
//!           command: "./scripts/validate-command.sh"
//!   PostToolUse:
//!     - matcher: "Edit|Write"
//!       hooks:
//!         - type: command
//!           command: "./scripts/run-linter.sh"
//! ```
//!
//! A command runs in the workspace through the shell, with the call as JSON
//! on its standard input (`hook_event_name`, `tool_name`, `tool_input`, and
//! for `PostToolUse` `tool_response`). Exit code 2 from a `PreToolUse` hook
//! blocks the call, and its standard error is what the agent is told; from a
//! `PostToolUse` hook, its standard error is added to the tool's result.
//! Any other exit status lets the call go on. `matcher` is a regular
//! expression for the whole tool name, Claude Code's (`Bash`, `Edit`) or
//! orangu's (`run_shell_command`); without one, or `*`, every tool matches.

use regex::Regex;
use serde_json::{Map, Value, json};
use std::path::Path;
use std::time::Duration;

/// How long one hook command may run.
const HOOK_TIMEOUT: Duration = Duration::from_secs(60);

/// The exit status with which a hook blocks a call or speaks to the agent.
const BLOCKING_EXIT: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    PreToolUse,
    PostToolUse,
}

impl Event {
    fn name(self) -> &'static str {
        match self {
            Event::PreToolUse => "PreToolUse",
            Event::PostToolUse => "PostToolUse",
        }
    }
}

#[derive(Debug, Clone)]
struct Rule {
    event: Event,
    /// `None` matches every tool.
    matcher: Option<Regex>,
    commands: Vec<String>,
}

/// The `hooks` of one agent file.
#[derive(Debug, Clone, Default)]
pub struct AgentHooks {
    rules: Vec<Rule>,
    /// The events and hook types it names that orangu does not run.
    pub unsupported: Vec<String>,
}

impl AgentHooks {
    /// The hooks a frontmatter `hooks` value defines.
    pub fn parse(value: &serde_yaml::Value) -> Self {
        let mut hooks = Self::default();
        let Some(events) = value.as_mapping() else {
            return hooks;
        };
        for (event, groups) in events {
            let Some(event_name) = event.as_str() else {
                continue;
            };
            let event = match event_name {
                "PreToolUse" => Event::PreToolUse,
                "PostToolUse" => Event::PostToolUse,
                other => {
                    hooks.unsupported.push(other.to_string());
                    continue;
                }
            };
            for group in groups.as_sequence().into_iter().flatten() {
                let matcher = group
                    .get("matcher")
                    .and_then(serde_yaml::Value::as_str)
                    .map(str::trim)
                    .filter(|m| !m.is_empty() && *m != "*");
                let matcher = match matcher {
                    Some(pattern) => match Regex::new(&format!("^(?:{pattern})$")) {
                        Ok(regex) => Some(regex),
                        Err(_) => {
                            hooks
                                .unsupported
                                .push(format!("{event_name} matcher '{pattern}'"));
                            continue;
                        }
                    },
                    None => None,
                };
                let mut commands = Vec::new();
                for hook in group
                    .get("hooks")
                    .and_then(serde_yaml::Value::as_sequence)
                    .into_iter()
                    .flatten()
                {
                    let kind = hook
                        .get("type")
                        .and_then(serde_yaml::Value::as_str)
                        .unwrap_or("command");
                    match (
                        kind,
                        hook.get("command").and_then(serde_yaml::Value::as_str),
                    ) {
                        ("command", Some(command)) if !command.trim().is_empty() => {
                            commands.push(command.to_string());
                        }
                        _ => hooks.unsupported.push(format!("{event_name} {kind} hook")),
                    }
                }
                if !commands.is_empty() {
                    hooks.rules.push(Rule {
                        event,
                        matcher,
                        commands,
                    });
                }
            }
        }
        hooks
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The commands for `event` on tool `tool`, which goes by `names`
    /// (orangu's, and the Claude Code names that map to it).
    fn commands(&self, event: Event, names: &[&str]) -> Vec<&str> {
        self.rules
            .iter()
            .filter(|rule| rule.event == event)
            .filter(|rule| {
                rule.matcher
                    .as_ref()
                    .is_none_or(|matcher| names.iter().any(|name| matcher.is_match(name)))
            })
            .flat_map(|rule| rule.commands.iter().map(String::as_str))
            .collect()
    }

    /// Runs the `PreToolUse` hooks for a call of `tool`: `Err` with the
    /// reason when one blocks it.
    pub async fn before(
        &self,
        tool: &str,
        input: &Map<String, Value>,
        workspace: &Path,
    ) -> Result<(), String> {
        let names = names_of(tool);
        let payload = json!({
            "hook_event_name": Event::PreToolUse.name(),
            "tool_name": names[0],
            "tool_input": input,
            "cwd": workspace,
        });
        for command in self.commands(Event::PreToolUse, &names) {
            if let Some(reason) = run(command, &payload, workspace).await {
                return Err(reason);
            }
        }
        Ok(())
    }

    /// Runs the `PostToolUse` hooks for a call of `tool` that returned
    /// `output`: what they have to tell the agent, if anything.
    pub async fn after(
        &self,
        tool: &str,
        input: &Map<String, Value>,
        output: &str,
        workspace: &Path,
    ) -> Option<String> {
        let names = names_of(tool);
        let payload = json!({
            "hook_event_name": Event::PostToolUse.name(),
            "tool_name": names[0],
            "tool_input": input,
            "tool_response": output,
            "cwd": workspace,
        });
        let mut said = Vec::new();
        for command in self.commands(Event::PostToolUse, &names) {
            if let Some(feedback) = run(command, &payload, workspace).await {
                said.push(feedback);
            }
        }
        (!said.is_empty()).then(|| said.join("\n"))
    }
}

/// The names a hook may know orangu tool `tool` by: Claude Code's first
/// (what `tool_name` carries), then orangu's own.
fn names_of(tool: &str) -> Vec<&str> {
    let mut names = crate::agents::custom::claude_code_names(tool);
    names.push(tool);
    names
}

/// Runs hook `command` with `payload` on its standard input; its standard
/// error when it exits with [`BLOCKING_EXIT`], otherwise `None`. A hook that
/// cannot start, fails otherwise or runs too long does not stop the call.
async fn run(command: &str, payload: &Value, workspace: &Path) -> Option<String> {
    use tokio::io::AsyncWriteExt;
    let (program, shell_args) = crate::shell::command_parts();
    let mut child = tokio::process::Command::new(program)
        .args(shell_args)
        .arg(command)
        .current_dir(workspace)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(payload.to_string().as_bytes()).await;
    }
    let output = tokio::time::timeout(HOOK_TIMEOUT, child.wait_with_output())
        .await
        .ok()?
        .ok()?;
    (output.status.code() == Some(BLOCKING_EXIT)).then(|| {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("hook `{command}` said no")
        } else {
            stderr
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hooks(yaml: &str) -> AgentHooks {
        AgentHooks::parse(&serde_yaml::from_str(yaml).unwrap())
    }

    #[test]
    fn hooks_are_read_in_claude_codes_shape() {
        let parsed = hooks(
            "PreToolUse:\n  - matcher: Bash\n    hooks:\n      - type: command\n        command: ./check.sh\nPostToolUse:\n  - hooks:\n      - command: ./lint.sh\n      - type: prompt\n        prompt: judge\nStop:\n  - hooks:\n      - command: ./done.sh\n",
        );
        assert_eq!(parsed.rules.len(), 2);
        assert_eq!(
            parsed.commands(Event::PreToolUse, &names_of("run_shell_command")),
            ["./check.sh"]
        );
        assert!(
            parsed
                .commands(Event::PreToolUse, &names_of("show_file"))
                .is_empty()
        );
        assert_eq!(
            parsed.commands(Event::PostToolUse, &names_of("show_file")),
            ["./lint.sh"]
        );
        assert_eq!(parsed.unsupported, ["PostToolUse prompt hook", "Stop"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_pre_hook_blocks_with_exit_2_and_a_post_hook_speaks_up() {
        use crate::tools::ToolExecutor;

        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("a.txt"), "hello\n").unwrap();
        let tools = ToolExecutor::new_for_subagent(workspace.path()).with_agent_hooks(hooks(
            "PreToolUse:\n  - matcher: LS|Glob\n    hooks:\n      - command: cat > seen.json; echo no listing >&2; exit 2\n  - matcher: Read\n    hooks:\n      - command: exit 1\nPostToolUse:\n  - matcher: show_file\n    hooks:\n      - command: echo looked at it >&2; exit 2\n",
        ));

        let blocked = tools
            .execute("list_directory", &Map::new())
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(blocked, "blocked by a PreToolUse hook: no listing");
        let seen: Value = serde_json::from_str(
            &std::fs::read_to_string(workspace.path().join("seen.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(seen["hook_event_name"], "PreToolUse");
        assert_eq!(seen["tool_name"], "Glob");

        // Exit 1 does not block; the post hook's stderr joins the result.
        let mut arguments = Map::new();
        arguments.insert("path".into(), json!("a.txt"));
        let shown = tools.execute("show_file", &arguments).await.unwrap();
        assert!(shown.contains("hello"), "{shown}");
        assert!(
            shown.ends_with("[PostToolUse hook]\nlooked at it"),
            "{shown}"
        );
    }
}

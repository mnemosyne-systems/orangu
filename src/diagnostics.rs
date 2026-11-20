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

//! Compiler diagnostics after edits: once a round of tool calls has changed
//! files, the workspace's own checker runs and its errors are shown to the
//! model with that round's results.

use regex::Regex;
use serde::Serialize;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeSet, HashSet},
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::Duration,
};

/// The most diagnostics one report lists.
const MAX_SHOWN: usize = 10;
const MAX_MESSAGE_CHARS: usize = 240;
/// Lines of output shown when a checker fails without a parseable error.
const FAILURE_TAIL_LINES: usize = 12;

/// The tools whose successful calls change files in the workspace.
const EDITING_TOOLS: &[&str] = &["modify_file", "create_file", "move_file", "delete_file"];

/// `[orangu].diagnostics`, `diagnostics_command` and `diagnostics_timeout`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DiagnosticsConfig {
    pub enabled: bool,
    /// Replaces the detected checkers; its output is read as
    /// `file:line[:column]: message` lines.
    pub command: Option<String>,
    pub timeout_seconds: u64,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: None,
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

pub fn default_timeout_seconds() -> u64 {
    120
}

/// One conversation's diagnostics: the files edited since the last check
/// and the last report, shared by every clone of its executor.
#[derive(Debug)]
pub struct Diagnostics {
    config: DiagnosticsConfig,
    workspace: PathBuf,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    edited: BTreeSet<String>,
    last_report: Option<String>,
}

impl Diagnostics {
    pub fn new(config: DiagnosticsConfig, workspace: &Path) -> Self {
        Self {
            config,
            workspace: workspace.to_path_buf(),
            state: Mutex::default(),
        }
    }

    pub fn config(&self) -> &DiagnosticsConfig {
        &self.config
    }

    /// Notes the files a successful `tool` call changed.
    pub fn record(&self, tool: &str, arguments: &Map<String, Value>) {
        if !EDITING_TOOLS.contains(&tool) {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        for key in ["path", "from", "to"] {
            if let Some(path) = arguments.get(key).and_then(Value::as_str) {
                state.edited.insert(relative_path(&self.workspace, path));
            }
        }
    }

    /// Checks the workspace if files were edited since the last check, and
    /// returns what the model should be told; `None` when nothing ran.
    pub async fn check_edits(&self) -> Option<String> {
        let edited = {
            let mut state = self.state.lock().ok()?;
            std::mem::take(&mut state.edited)
        };
        if edited.is_empty() {
            return None;
        }
        let checks = plan(&self.workspace, &self.config, &edited);
        if checks.is_empty() {
            return None;
        }
        let timeout = Duration::from_secs(self.config.timeout_seconds.max(1));
        let mut sections = Vec::new();
        let mut any_errors = false;
        for check in &checks {
            let outcome = run(check, &self.workspace, timeout).await;
            any_errors |= matches!(&outcome, Outcome::Errors(found) if !found.is_empty())
                || matches!(outcome, Outcome::Failed(_));
            if let Some(section) = render(check, outcome, &edited, timeout) {
                sections.push(section);
            }
        }
        if sections.is_empty() {
            return None;
        }
        let body = sections.join("\n");
        let mut state = self.state.lock().ok()?;
        let unchanged = state.last_report.as_deref() == Some(body.as_str());
        state.last_report = Some(body.clone());
        Some(if unchanged && any_errors {
            "[diagnostics] Unchanged since the previous check: the same errors remain.".to_string()
        } else {
            format!("[diagnostics]\n{body}")
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Check {
    Cargo,
    GoVet,
    Tsc(PathBuf),
    Python(Vec<String>),
    Custom(String),
}

impl Check {
    fn label(&self) -> String {
        match self {
            Self::Cargo => "cargo check".to_string(),
            Self::GoVet => "go vet".to_string(),
            Self::Tsc(_) => "tsc".to_string(),
            Self::Python(_) => "python syntax check".to_string(),
            Self::Custom(command) => command.clone(),
        }
    }
}

/// The checks that cover `edited`: the configured command alone when there
/// is one, otherwise one per detected toolchain the edits touch.
fn plan(workspace: &Path, config: &DiagnosticsConfig, edited: &BTreeSet<String>) -> Vec<Check> {
    if let Some(command) = &config.command {
        return vec![Check::Custom(command.clone())];
    }
    let touches = |extensions: &[&str], names: &[&str]| {
        edited.iter().any(|path| {
            let name = path.rsplit('/').next().unwrap_or(path);
            names.contains(&name)
                || Path::new(path)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| extensions.contains(&ext))
        })
    };
    let mut checks = Vec::new();
    if workspace.join("Cargo.toml").is_file() && touches(&["rs"], &["Cargo.toml"]) {
        checks.push(Check::Cargo);
    }
    if workspace.join("go.mod").is_file() && touches(&["go"], &["go.mod"]) {
        checks.push(Check::GoVet);
    }
    if workspace.join("tsconfig.json").is_file()
        && touches(&["ts", "tsx", "mts", "cts"], &["tsconfig.json"])
        && let Some(tsc) = local_tsc(workspace)
    {
        checks.push(Check::Tsc(tsc));
    }
    let python: Vec<String> = edited
        .iter()
        .filter(|path| path.ends_with(".py") && workspace.join(path).is_file())
        .cloned()
        .collect();
    if !python.is_empty() {
        checks.push(Check::Python(python));
    }
    checks
}

fn local_tsc(workspace: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "tsc.cmd" } else { "tsc" };
    let path = workspace.join("node_modules").join(".bin").join(name);
    path.is_file().then_some(path)
}

/// Compiles each file without writing bytecode, one line per syntax error.
const PYTHON_CHECK: &str = "import sys\n\
for f in sys.argv[1:]:\n\
    try:\n\
        compile(open(f, encoding='utf-8').read(), f, 'exec')\n\
    except SyntaxError as e:\n\
        print(f'{f}:{e.lineno or 1}:{e.offset or 1}: {type(e).__name__}: {e.msg}')\n\
    except Exception as e:\n\
        print(f'{f}:1:1: {type(e).__name__}: {e}')\n";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Diagnostic {
    file: String,
    line: Option<u32>,
    column: Option<u32>,
    message: String,
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Errors(Vec<Diagnostic>),
    /// It failed without an error it could point at: its output's tail.
    Failed(String),
    TimedOut,
    /// It could not start (the toolchain is not installed).
    Unavailable,
}

async fn run(check: &Check, workspace: &Path, timeout: Duration) -> Outcome {
    let mut command = match check {
        Check::Cargo => {
            let mut command = tokio::process::Command::new("cargo");
            command.args(["check", "--all-targets", "--message-format=json", "--quiet"]);
            command.env("CARGO_TERM_COLOR", "never");
            command
        }
        Check::GoVet => {
            let mut command = tokio::process::Command::new("go");
            command.args(["vet", "./..."]);
            command
        }
        Check::Tsc(tsc) => {
            let mut command = tokio::process::Command::new(tsc);
            command.args(["--noEmit", "--pretty", "false"]);
            command
        }
        Check::Python(files) => {
            let python = if cfg!(windows) { "python" } else { "python3" };
            let mut command = tokio::process::Command::new(python);
            command.args(["-I", "-c", PYTHON_CHECK]).args(files);
            command
        }
        Check::Custom(line) => {
            let (program, shell_args) = crate::shell::command_parts();
            let mut command = tokio::process::Command::new(program);
            command.args(shell_args).arg(line);
            command
        }
    };
    command
        .current_dir(workspace)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let Ok(child) = command.spawn() else {
        return Outcome::Unavailable;
    };
    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(_)) => return Outcome::Unavailable,
        Err(_) => return Outcome::TimedOut,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let found = match check {
        Check::Cargo => parse_cargo(&stdout),
        _ => parse_lines(&format!("{stdout}\n{stderr}")),
    };
    let found = dedup(
        found
            .into_iter()
            .map(|diagnostic| Diagnostic {
                file: relative_path(workspace, &diagnostic.file),
                ..diagnostic
            })
            .collect(),
    );
    if found.is_empty() && !output.status.success() && !matches!(check, Check::Python(_)) {
        return Outcome::Failed(tail(&format!("{stderr}\n{stdout}")));
    }
    Outcome::Errors(found)
}

/// The errors in `cargo --message-format=json` output.
fn parse_cargo(stdout: &str) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        let level = message.get("level").and_then(Value::as_str).unwrap_or("");
        if !level.starts_with("error") {
            continue;
        }
        let text = message.get("message").and_then(Value::as_str).unwrap_or("");
        if text.starts_with("aborting due to") {
            continue;
        }
        let code = message
            .get("code")
            .and_then(|code| code.get("code"))
            .and_then(Value::as_str);
        let primary = message
            .get("spans")
            .and_then(Value::as_array)
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|span| span.get("is_primary").and_then(Value::as_bool) == Some(true))
            });
        let label = primary
            .and_then(|span| span.get("label"))
            .and_then(Value::as_str)
            .filter(|label| !label.is_empty());
        let mut rendered = match code {
            Some(code) => format!("error[{code}]: {text}"),
            None => format!("error: {text}"),
        };
        if let Some(label) = label {
            rendered.push_str(" — ");
            rendered.push_str(label);
        }
        let number = |key: &str| {
            primary
                .and_then(|span| span.get(key))
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
        };
        found.push(Diagnostic {
            file: primary
                .and_then(|span| span.get("file_name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            line: number("line_start"),
            column: number("column_start"),
            message: rendered,
        });
    }
    found
}

static LOCATED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:vet: )?((?:[A-Za-z]:)?[^:()\s][^:()]*?):(\d+)(?::(\d+))?:\s*(.+)$")
        .expect("valid pattern")
});

static TSC_LOCATED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(.+?)\((\d+),(\d+)\):\s*(.+)$").expect("valid pattern"));

/// `file:line[:column]: message` and `file(line,column): message` lines,
/// warnings and notes left out.
fn parse_lines(output: &str) -> Vec<Diagnostic> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim_end();
            let captures = TSC_LOCATED
                .captures(line)
                .or_else(|| LOCATED.captures(line))?;
            let message = captures.get(4)?.as_str().trim().to_string();
            let lower = message.to_ascii_lowercase();
            if lower.starts_with("warning") || lower.starts_with("note") {
                return None;
            }
            Some(Diagnostic {
                file: captures.get(1)?.as_str().to_string(),
                line: captures.get(2).and_then(|n| n.as_str().parse().ok()),
                column: captures.get(3).and_then(|n| n.as_str().parse().ok()),
                message,
            })
        })
        .collect()
}

fn dedup(found: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let mut seen = HashSet::new();
    found
        .into_iter()
        .filter(|d| seen.insert((d.file.clone(), d.line, d.column, d.message.clone())))
        .collect()
}

fn tail(output: &str) -> String {
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    let start = lines.len().saturating_sub(FAILURE_TAIL_LINES);
    lines[start..]
        .iter()
        .map(|line| truncate(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(text: &str) -> String {
    match text.char_indices().nth(MAX_MESSAGE_CHARS) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_string(),
    }
}

/// One check's section of the report, errors in edited files first.
fn render(
    check: &Check,
    outcome: Outcome,
    edited: &BTreeSet<String>,
    timeout: Duration,
) -> Option<String> {
    let label = check.label();
    match outcome {
        Outcome::Unavailable => None,
        Outcome::TimedOut => Some(format!(
            "{label}: did not finish within {}s, so these edits are unchecked.",
            timeout.as_secs()
        )),
        Outcome::Failed(output) => Some(format!("{label} failed:\n{output}")),
        Outcome::Errors(found) if found.is_empty() => Some(format!("{label}: no errors.")),
        Outcome::Errors(mut found) => {
            found.sort_by_key(|d| !edited.contains(&d.file));
            let count = found.len();
            let mut lines = vec![format!(
                "{label}: {count} error{}",
                if count == 1 { "" } else { "s" }
            )];
            lines.extend(found.iter().take(MAX_SHOWN).map(format_diagnostic));
            if count > MAX_SHOWN {
                lines.push(format!("… and {} more", count - MAX_SHOWN));
            }
            Some(lines.join("\n"))
        }
    }
}

fn format_diagnostic(diagnostic: &Diagnostic) -> String {
    let location = match (diagnostic.file.as_str(), diagnostic.line, diagnostic.column) {
        ("", _, _) => String::new(),
        (file, Some(line), Some(column)) => format!("{file}:{line}:{column}: "),
        (file, Some(line), None) => format!("{file}:{line}: "),
        (file, None, _) => format!("{file}: "),
    };
    truncate(&format!("{location}{}", diagnostic.message))
}

/// `path` relative to `workspace`, with `/` separators.
fn relative_path(workspace: &Path, path: &str) -> String {
    let as_path = Path::new(path);
    let relative = as_path
        .strip_prefix(workspace)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| as_path.to_path_buf());
    let text = relative.to_string_lossy().replace('\\', "/");
    text.trim_start_matches("./").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn edited(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn plans_only_the_checkers_the_edits_touch() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("Cargo.toml"), "").unwrap();
        std::fs::write(workspace.path().join("go.mod"), "").unwrap();
        std::fs::write(workspace.path().join("tool.py"), "").unwrap();
        let config = DiagnosticsConfig::default();

        assert!(plan(workspace.path(), &config, &edited(&["README.md"])).is_empty());
        assert_eq!(
            plan(workspace.path(), &config, &edited(&["src/main.rs"])),
            [Check::Cargo]
        );
        assert_eq!(
            plan(
                workspace.path(),
                &config,
                &edited(&["cmd/x.go", "tool.py", "gone.py"])
            ),
            [Check::GoVet, Check::Python(vec!["tool.py".to_string()])]
        );
        let custom = DiagnosticsConfig {
            command: Some("make check".to_string()),
            ..config
        };
        assert_eq!(
            plan(workspace.path(), &custom, &edited(&["README.md"])),
            [Check::Custom("make check".to_string())]
        );
    }

    #[test]
    fn reads_cargo_errors_and_skips_warnings_and_the_summary() {
        let error = json!({
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "message": "mismatched types",
                "code": {"code": "E0308"},
                "spans": [
                    {"file_name": "src/other.rs", "line_start": 1, "column_start": 1, "is_primary": false},
                    {"file_name": "src/lib.rs", "line_start": 12, "column_start": 5, "is_primary": true, "label": "expected `u32`, found `&str`"}
                ]
            }
        });
        let warning = json!({
            "reason": "compiler-message",
            "message": {"level": "warning", "message": "unused variable", "spans": []}
        });
        let summary = json!({
            "reason": "compiler-message",
            "message": {"level": "error", "message": "aborting due to 1 previous error", "spans": []}
        });
        let stdout = format!(
            "{error}\n{warning}\n{summary}\n{}\nnot json\n",
            json!({"reason": "build-finished", "success": false})
        );
        assert_eq!(
            parse_cargo(&stdout),
            [Diagnostic {
                file: "src/lib.rs".to_string(),
                line: Some(12),
                column: Some(5),
                message: "error[E0308]: mismatched types — expected `u32`, found `&str`"
                    .to_string(),
            }]
        );
    }

    #[test]
    fn reads_located_lines_from_go_gcc_tsc_and_python() {
        let output = "# example.com/app\n\
            vet: ./main.go:5:2: undefined: foo\n\
            src/a.c:3:1: error: expected ';'\n\
            src/a.c:4:1: warning: unused\n\
            src/app.ts(7,3): error TS2322: Type 'string' is not assignable\n\
            C:\\work\\b.py:2:9: SyntaxError: invalid syntax\n\
            plain text line\n";
        let found = parse_lines(output);
        let summary: Vec<(String, Option<u32>, Option<u32>)> = found
            .iter()
            .map(|d| (d.file.clone(), d.line, d.column))
            .collect();
        assert_eq!(
            summary,
            [
                ("./main.go".to_string(), Some(5), Some(2)),
                ("src/a.c".to_string(), Some(3), Some(1)),
                ("src/app.ts".to_string(), Some(7), Some(3)),
                ("C:\\work\\b.py".to_string(), Some(2), Some(9)),
            ]
        );
        assert_eq!(found[0].message, "undefined: foo");
    }

    #[test]
    fn report_lists_edited_files_first_and_caps_the_rest() {
        let mut found: Vec<Diagnostic> = (0..12)
            .map(|n| Diagnostic {
                file: format!("src/f{n}.rs"),
                line: Some(n + 1),
                column: None,
                message: "error: broken".to_string(),
            })
            .collect();
        found.push(Diagnostic {
            file: "src/edited.rs".to_string(),
            line: Some(3),
            column: Some(4),
            message: "error: here".to_string(),
        });
        let report = render(
            &Check::Cargo,
            Outcome::Errors(found),
            &edited(&["src/edited.rs"]),
            Duration::from_secs(1),
        )
        .unwrap();
        let lines: Vec<&str> = report.lines().collect();
        assert_eq!(lines[0], "cargo check: 13 errors");
        assert_eq!(lines[1], "src/edited.rs:3:4: error: here");
        assert_eq!(lines.len(), 1 + MAX_SHOWN + 1);
        assert_eq!(lines.last().copied(), Some("… and 3 more"));
    }

    #[test]
    fn records_edited_paths_relative_to_the_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let diagnostics = Diagnostics::new(DiagnosticsConfig::default(), workspace.path());
        let absolute = workspace.path().join("src").join("lib.rs");
        let mut arguments = Map::new();
        arguments.insert("path".into(), json!(absolute.to_string_lossy()));
        diagnostics.record("modify_file", &arguments);
        let mut moved = Map::new();
        moved.insert("from".into(), json!("./a.rs"));
        moved.insert("to".into(), json!("b.rs"));
        diagnostics.record("move_file", &moved);
        diagnostics.record("show_file", &moved);
        let recorded = diagnostics.state.lock().unwrap().edited.clone();
        assert_eq!(recorded, edited(&["a.rs", "b.rs", "src/lib.rs"]));
    }

    #[tokio::test]
    async fn a_custom_command_reports_errors_then_an_unchanged_note() {
        let workspace = tempfile::tempdir().unwrap();
        let config = DiagnosticsConfig {
            command: Some("echo 'src/x.rs:1:2: error: boom'".to_string()),
            ..DiagnosticsConfig::default()
        };
        let diagnostics = Diagnostics::new(config, workspace.path());
        let mut arguments = Map::new();
        arguments.insert("path".into(), json!("src/x.rs"));

        assert_eq!(diagnostics.check_edits().await, None);

        diagnostics.record("modify_file", &arguments);
        let first = diagnostics.check_edits().await.unwrap();
        assert!(first.contains("1 error"), "{first}");
        assert!(first.contains("src/x.rs:1:2: error: boom"), "{first}");

        diagnostics.record("modify_file", &arguments);
        let second = diagnostics.check_edits().await.unwrap();
        assert!(second.contains("Unchanged"), "{second}");
    }
}

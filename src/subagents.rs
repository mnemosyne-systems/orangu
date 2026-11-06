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

//! In-process subagents the main agent reaches as ordinary tools.
//!
//! Every subagent follows one pattern: a fresh [`ChatSession`] with its own
//! system prompt (the parent's conversation is never shared, only the task
//! crosses over), a restricted [`ToolExecutor`], a model chosen by role, a
//! hard `max_tool_rounds` budget, and a reply wrapped in
//! `<final_answer>…</final_answer>` so the parent receives only that block.
//!
//! Subagents get no subagent tools of their own (the explorer runs on a
//! read-only executor, the writers on [`ToolExecutor::new_for_subagent`]),
//! except a custom agent whose `tools` list `Agent`: it may start subagents
//! ([`ToolExecutor::nested_under`]), down to `[orangu].subagent_depth` levels
//! below the main conversation (three by default).
//!
//! The [`DOCUMENT_WRITERS`] are one subagent per LibreOffice document type,
//! all behind the single [`DOCUMENT_TOOL`], chosen by its `type` argument.

use crate::agents::custom::CustomAgent;
use crate::config::{LlmConfiguration, default_client_config_path, load_client_configuration};
use crate::llm::{FunctionDefinition, ToolDefinition};
use crate::session::ChatSession;
use crate::subagent_tasks::{Resumable, TaskRegistry, TaskStatus};
use crate::tools::ToolExecutor;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// What one subagent is: the tool the parent sees and how its run is bounded.
pub struct SubagentSpec {
    /// The tool name offered to the parent model.
    pub tool: &'static str,
    /// The tool description offered to the parent model.
    pub description: &'static str,
    /// The single required string argument that carries the task.
    pub argument: &'static str,
    /// The role asked of the coordinator (or matched against `role =` in
    /// orangu.conf) to pick the model that runs this subagent.
    pub role: &'static str,
    /// Model requests the subagent may make, whatever the parent's setting.
    pub max_tool_rounds: usize,
    /// Whether the subagent may change the workspace.
    pub writable: bool,
    /// Whether the subagent may search the web (`[orangu].web_search`).
    pub web_search: bool,
    /// Whether the subagent follows the `AGENTS.md` rules the main agent does.
    /// Off for the explorer, which searches and never changes anything, and
    /// for the SSH subagent, which only copies files.
    pub project_rules: bool,
    pub system_prompt: &'static str,
    /// Opens the user message; the task follows inside `<task>` tags.
    pub instruction: &'static str,
}

const FINAL_ANSWER_REMINDER: &str = "CRITICAL: You MUST wrap your final conclusion in a `<final_answer>...</final_answer>` block. If you omit these tags, the system will not be able to parse your response.";

pub const EXPLORER: SubagentSpec = SubagentSpec {
    tool: "explore_repository",
    description: "Have a subagent search the repository broadly, keeping your context \
                  small; returns relevant files and line ranges.",
    argument: "query",
    role: "explorer",
    max_tool_rounds: 6,
    writable: false,
    web_search: false,
    project_rules: false,
    system_prompt: crate::explorer::EXPLORER_SYSTEM_PROMPT,
    instruction: "Please explore the codebase to answer the following query:",
};

pub const TEST_WRITER: SubagentSpec = SubagentSpec {
    tool: "write_tests",
    description: "Subagent writes and runs tests for a self-contained `task`.",
    argument: "task",
    role: "code",
    max_tool_rounds: 12,
    writable: true,
    web_search: false,
    project_rules: true,
    system_prompt: TEST_WRITER_SYSTEM_PROMPT,
    instruction: "Write tests for the following task:",
};

pub const DOC_WRITER: SubagentSpec = SubagentSpec {
    tool: "write_docs",
    description: "Subagent writes documentation for a self-contained `task`.",
    argument: "task",
    role: "code",
    max_tool_rounds: 8,
    writable: true,
    web_search: false,
    project_rules: true,
    system_prompt: DOC_WRITER_SYSTEM_PROMPT,
    instruction: "Write documentation for the following task:",
};

/// Offered only when `[orangu].web_search` names a SearXNG instance.
pub const WEB_RESEARCHER: SubagentSpec = SubagentSpec {
    tool: "research_web",
    description: "Subagent searches the web and reads pages to answer a self-contained \
                  `question`; returns the answer with source URLs.",
    argument: "question",
    role: "code",
    max_tool_rounds: 10,
    writable: false,
    web_search: true,
    project_rules: true,
    system_prompt: WEB_RESEARCHER_SYSTEM_PROMPT,
    instruction: "Research the following question on the web:",
};

/// Offered unless `[orangu].ssh` is `deny`.
pub const SSH: SubagentSpec = SubagentSpec {
    tool: "use_ssh",
    description: "Subagent logs in to a host over SSH, copies files to or from the workspace, or \
                  logs out, for a self-contained `task`. Never put credentials in it.",
    argument: "task",
    role: "code",
    max_tool_rounds: 24,
    writable: true,
    web_search: false,
    project_rules: false,
    system_prompt: SSH_SYSTEM_PROMPT,
    instruction: "Do the following over SSH:",
};

/// The tools the SSH subagent runs with, and nothing else: with a shell it
/// could reach any machine without the checks `ssh_login` makes.
const SSH_SUBAGENT_TOOLS: &[&str] = &["ssh_login", "ssh_copy", "ssh_logout", "list_directory"];

/// Every subagent, in the order their tools are offered.
pub const SUBAGENTS: &[&SubagentSpec] =
    &[&EXPLORER, &TEST_WRITER, &DOC_WRITER, &WEB_RESEARCHER, &SSH];

const SSH_SYSTEM_PROMPT: &str = r#"You log in to other machines over SSH and copy files between them and the workspace. You cannot run commands on them.

Rules:
- ssh_login needs only the host. It takes care of credentials itself and may ask the user. If a task contains a password or passphrase, do not use it or repeat it.
- ssh_login reuses an open session, so call it first in every task. Use the session name it returns.
- ssh_copy copies one file per call. Copy several files with several calls; find local ones with list_directory.
- Replace an existing file only if the task says to: pass overwrite only then. If a copy is refused because the file exists, report that.
- Stay logged in when you finish, unless the task says to log out.
- A refused host key or login is a decision or a safeguard. Report the exact reason; do not try another host, user or path to get around it.

End with a `<final_answer>` block saying what was copied, from where to where, the sizes, the session left open if any, and anything not done."#;

const WEB_RESEARCHER_SYSTEM_PROMPT: &str = r#"
You are a research specialist. Your only goal is to answer the question with facts, looking locally before going to the web.

Guidelines:
- Local first. If the workspace can answer the question (its code, README, docs, vendored dependencies, lock files), read it with show_file, list_directory or run_shell_command (rg, grep) before searching the web, and stop there if it does. You cannot change the workspace.
- Search with web_search. Start with a precise query; if the results are thin or off-target, rephrase: different terms, the exact error message in quotes, the project or product name, a version number, or the current year for recent topics.
- Read pages with browse before relying on them: search excerpts are short and can be out of date. Open a result with browse url=…, follow a link on the page you are reading with browse link=N (the numbers listed under the text), and read further into a long page with browse part=N. Pages you have read stay cached, so going back to one costs nothing.
- Prefer primary sources: official documentation, specifications, release notes, source repositories, and the vendor's own pages. From a documentation page, follow its links to the exact section rather than searching again.
- Check important claims against a second independent source. When sources disagree, say so and say which you trust and why.
- Note dates: say when the information is from, and flag anything that may have changed since.
- Never invent a URL, a version, a number or a quotation. If you could not find something, say so plainly.

## Required Output

End your response with a `<final_answer>` block: a direct answer first, then the key details, then a `Sources:` list with the URL of every page the answer relies on.

<example>
<final_answer>
tokio 1.40 is the first release with `JoinSet::join_all` (September 2024).

- It waits for every task in the set and returns their outputs in completion order.
- Earlier versions need a loop over `join_next()`.

Sources:
- https://docs.rs/tokio/1.40.0/tokio/task/struct.JoinSet.html
- https://github.com/tokio-rs/tokio/releases/tag/tokio-1.40.0
</final_answer>
</example>
"#;

/// The one tool behind which every [`DocumentWriter`] sits, chosen by `type`:
/// one definition in every request instead of one per document type.
pub const DOCUMENT_TOOL: &str = "write_document";

/// A subagent that writes one LibreOffice document type with
/// `convert_document` (see [`crate::office`]).
pub struct DocumentWriter {
    /// The `type` argument of [`DOCUMENT_TOOL`] that selects this writer.
    pub kind: &'static str,
    pub spec: SubagentSpec,
}

/// The system prompt of a document writer: what is common to every type,
/// around how the type is written (`rendered` through `convert_document`, or
/// written as a plain `file`) and `$format`, its own source format and
/// target.
macro_rules! document_prompt {
    (rendered, $format:literal) => {
        document_prompt!(
            @ r#"- Write the whole document in ONE convert_document call: pass `content` (the document's source text), `format` (its source format) and `target` (the output path). Do not create the source as a separate file.
- If the task asks for a PDF as well, call convert_document again with the same `content` and `format` and the same path ending in `.pdf` as `target`. Render it from the source text rather than from the document you wrote: that keeps everything the source says.
- If a conversion fails, fix the source text and try again.
"#,
            $format
        )
    };
    (file, $format:literal) => {
        document_prompt!(
            @ r#"- Write the document with ONE create_file call. To revise it, use modify_file.
- If the task asks for a PDF as well, call convert_document with the file as `source` and the same path ending in `.pdf` as `target`.
- If a conversion fails, fix the file and try again.
"#,
            $format
        )
    };
    (@ $how:literal, $format:literal) => {
        concat!(
            r#"
You are a document specialist working inside a software repository.
Your only goal is to produce the document the task asks for.

Guidelines:
- If the task refers to files, code or data in the workspace, read them first (show_file, list_directory, run_shell_command with rg or grep), and never invent facts, numbers or names.
- If the task names no path, choose a short descriptive file name in the workspace root.
"#,
            $how,
            r#"- Never commit, push or touch version control beyond what the tools do themselves.
"#,
            $format,
            r#"
## Required Output

End your response with a `<final_answer>` block listing every file you wrote and a one-line summary of its contents.

<example>
<final_answer>
Wrote: quarterly-report.odt (4 sections: summary, revenue table, risks, next steps)
Wrote: quarterly-report.pdf
</final_answer>
</example>
"#
        )
    };
}

const fn document_writer(
    kind: &'static str,
    max_tool_rounds: usize,
    system_prompt: &'static str,
    instruction: &'static str,
) -> DocumentWriter {
    DocumentWriter {
        kind,
        spec: SubagentSpec {
            tool: DOCUMENT_TOOL,
            description: "",
            argument: "task",
            role: "code",
            max_tool_rounds,
            writable: true,
            web_search: false,
            project_rules: true,
            system_prompt,
            instruction,
        },
    }
}

/// One subagent per document type: the LibreOffice ones, rendered through
/// `convert_document`, and Markdown and HTML, written as plain files.
pub const DOCUMENT_WRITERS: &[DocumentWriter] = &[
    document_writer(
        "text",
        8,
        document_prompt!(
            rendered,
            r#"
## Text documents (LibreOffice Writer)

- Source format: `md` (Markdown: headings, paragraphs, emphasis, lists, tables, links, code blocks, images by workspace-relative path). Use `html` only when Markdown cannot express the layout.
- Target: `.odt` unless the task asks for Word (`.docx`).
"#
        ),
        "Write the following text document:",
    ),
    document_writer(
        "spreadsheet",
        8,
        document_prompt!(
            rendered,
            r#"
## Spreadsheets (LibreOffice Calc)

- Source format: `csv`. The first row holds the column headers. A cell starting with `=` is a formula with A1 references, for example `=SUM(B2:B9)` (row 1 is the header row). Quote any field containing a comma, a quote or a newline, doubling quotes inside it.
- Plain numbers stay numbers: no thousands separators or currency symbols in numeric cells.
- One sheet per document. If the task needs several tables, write one document per table.
- Target: `.ods` unless the task asks for Excel (`.xlsx`).
"#
        ),
        "Write the following spreadsheet:",
    ),
    document_writer(
        "presentation",
        8,
        document_prompt!(
            rendered,
            r#"
## Presentations (LibreOffice Impress)

- Source format: `outline`. A line `# Title` starts a new slide with that title (exactly one `#`). A line `- point` is a bullet on the current slide. Any other non-empty line is a text box. Blank lines are ignored.
- Keep slides short: a title and three to six bullets of at most about ten words each.
- Target: `.odp` unless the task asks for PowerPoint (`.pptx`).
"#
        ),
        "Write the following presentation:",
    ),
    document_writer(
        "drawing",
        8,
        document_prompt!(
            rendered,
            r#"
## Drawings (LibreOffice Draw)

- Source format: `svg`: one `<svg xmlns="http://www.w3.org/2000/svg">` element with explicit `width`, `height` and `viewBox`, built from basic shapes (`rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `path`) and `text`. No scripts, external references or CSS files.
- Lay shapes out on a grid so nothing overlaps unintentionally, and keep text inside its shapes.
- Target: `.odg`.
"#
        ),
        "Draw the following:",
    ),
    document_writer(
        "markdown",
        8,
        document_prompt!(
            file,
            r#"
## Markdown documents

- Write CommonMark with GitHub tables: ATX headings (`#`), one blank line between blocks, fenced code blocks with a language, `-` for bullets, relative links and image paths.
- One `#` title, then `##` sections. Keep lines readable; do not hard-wrap tables or code.
- Target: `.md`.
"#
        ),
        "Write the following Markdown document:",
    ),
    document_writer(
        "html",
        8,
        document_prompt!(
            file,
            r#"
## HTML documents

- Write one self-contained HTML5 file: `<!doctype html>`, `<html lang>`, `<head>` with `<meta charset="utf-8">`, a viewport meta tag and a `<title>`, then semantic `<body>` content (`header`, `main`, `section`, `h1`-`h3`, `p`, `ul`, `table`, `figure`).
- Put styling in one `<style>` element in the head. No scripts, external stylesheets, fonts or remote images unless the task asks for them.
- Target: `.html`.
"#
        ),
        "Write the following HTML document:",
    ),
];

/// The document writer for `kind`, if there is one.
pub fn find_document_writer(kind: &str) -> Option<&'static DocumentWriter> {
    DOCUMENT_WRITERS.iter().find(|writer| writer.kind == kind)
}

/// The [`DOCUMENT_TOOL`] definition offered to the main agent.
pub fn document_definition() -> ToolDefinition {
    let kinds: Vec<&str> = DOCUMENT_WRITERS.iter().map(|writer| writer.kind).collect();
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: DOCUMENT_TOOL.to_string(),
            description:
                "Subagent writes a document (odt/ods/odp/odg/md/html; docx/xlsx/pptx or PDF \
                          on request) for a self-contained `task`."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": kinds},
                    "task": {"type": "string"}
                },
                "required": ["type", "task"]
            }),
        },
    }
}

/// Runs the document writer `arguments["type"]` names.
pub fn run_document<'a>(
    parent: &'a ToolExecutor,
    arguments: &'a serde_json::Map<String, Value>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    let writer = arguments
        .get("type")
        .and_then(|v| v.as_str())
        .context("missing or invalid 'type' argument")
        .and_then(|kind| {
            find_document_writer(kind).with_context(|| {
                let kinds: Vec<&str> = DOCUMENT_WRITERS.iter().map(|w| w.kind).collect();
                format!(
                    "unknown document type '{kind}'; use one of {}",
                    kinds.join(", ")
                )
            })
        });
    match writer {
        Ok(writer) => run(&writer.spec, parent, arguments),
        Err(err) => Box::pin(async move { Err(err) }),
    }
}

const TEST_WRITER_SYSTEM_PROMPT: &str = r#"
You are a test-writing specialist working inside a software repository.
Your only goal is to add or extend automated tests for the code named in the task.

Guidelines:
- First find the code under test and the project's existing tests (show_file, list_directory, run_shell_command with rg or grep). Follow their framework, layout, naming and style exactly.
- Test behaviour through public interfaces. Cover the normal case, edge cases and error paths the task mentions.
- Change only test code. If a test exposes a bug in the code under test, do not fix it: report it.
- Run the tests you wrote with the project's usual command (for example `cargo test <name>`) and fix your tests until they compile and pass, unless they fail because of a real bug.
- Never commit, push or touch version control beyond what the file tools do themselves.

## Required Output

End your response with a `<final_answer>` block listing every file you created or changed, the command you ran, its outcome, and any bug you found.

<example>
<final_answer>
Changed: src/parser.rs:410-472 (3 tests for empty input, nested groups, unterminated string)
Command: cargo test parser::tests
Result: 3 passed
Bugs: none
</final_answer>
</example>
"#;

const DOC_WRITER_SYSTEM_PROMPT: &str = r#"
You are a documentation specialist working inside a software repository.
Your only goal is to write or update the documentation the task asks for.

Guidelines:
- Read the code being documented before writing about it; never describe behaviour you have not confirmed in the source.
- Find where the project already documents similar things (doc comments, README, a manual directory) and follow its structure, tone and formatting.
- Change only documentation and comments. Never change code behaviour.
- Be concise and concrete: what it does, how to use it, its options and defaults, and a short example when it helps.
- Never commit, push or touch version control beyond what the file tools do themselves.

## Required Output

End your response with a `<final_answer>` block listing every file you created or changed and what you documented in each.

<example>
<final_answer>
Changed: doc/manual/en/30-tools.md:40-62 (new "Subagents" section)
Changed: src/subagents.rs:1-20 (module doc comment)
</final_answer>
</example>
"#;

/// The subagent behind tool `name`, if it is one.
pub fn find(name: &str) -> Option<&'static SubagentSpec> {
    SUBAGENTS.iter().copied().find(|spec| spec.tool == name)
}

impl SubagentSpec {
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.tool.to_string(),
                description: self.description.to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        self.argument: {"type": "string"}
                    },
                    "required": [self.argument]
                }),
            },
        }
    }
}

/// Resolves which server a subagent asking for `role` should use.
///
/// Once a coordinator is confirmed (`is_coordinator`), it alone owns every
/// model/role decision: a local `role = …` section in orangu.conf is never
/// consulted. The active connection is reused as-is with `.model` set to
/// the role, and the coordinator resolves that to whatever real model backs
/// it (falling back to `all` if it has none).
///
/// Otherwise a section with that `role` is used (the first by name), falling
/// back to the connection the conversation is using.
pub fn target_profile(link: &SessionLink, is_coordinator: bool, role: &str) -> LlmConfiguration {
    if is_coordinator {
        let mut profile = link.active.clone();
        profile.model = role.to_string();
        return profile;
    }
    let mut dedicated: Vec<&String> = link
        .llms
        .iter()
        .filter(|(_, profile)| profile.role == role)
        .map(|(name, _)| name)
        .collect();
    dedicated.sort();
    dedicated
        .first()
        .and_then(|name| link.llms.get(*name))
        .cloned()
        .unwrap_or_else(|| link.active.clone())
}

/// The `<final_answer>…</final_answer>` block of `response`, tags included.
pub fn final_answer(response: &str) -> Option<&str> {
    let start = response.find("<final_answer>")?;
    let end = response[start..].find("</final_answer>")?;
    Some(&response[start..start + end + "</final_answer>".len()])
}

fn root_listing(workspace: &Path) -> String {
    let mut names: Vec<String> = std::fs::read_dir(workspace)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names.join("\n")
}

/// The user message that hands `task` to the subagent described by `spec`.
fn task_message(spec: &SubagentSpec, task: &str) -> String {
    format!(
        "{}\n\n<task>\n{}\n</task>\n\n{}",
        spec.instruction, task, FINAL_ANSWER_REMINDER
    )
}

/// Runs the subagent `spec` on `arguments` in `parent`'s workspace, on the
/// connection `parent`'s conversation is using, and returns its final
/// answer for the parent.
pub fn run<'a>(
    spec: &'static SubagentSpec,
    parent: &'a ToolExecutor,
    arguments: &'a serde_json::Map<String, Value>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let task = arguments
            .get(spec.argument)
            .and_then(|v| v.as_str())
            .filter(|task| !task.trim().is_empty())
            .with_context(|| format!("missing or invalid '{}' argument", spec.argument))?;
        if spec.tool == SSH.tool && parent.ssh().is_none() {
            anyhow::bail!("SSH is off ([orangu].ssh)");
        }
        let (tools, additions) = spec_tools(spec, parent);
        let system_prompt = format!("{}{additions}", spec.system_prompt);
        launch(
            Launch {
                recipe: recipe_of(spec),
                history: Vec::new(),
                worktree: None,
                name: spec.tool.to_string(),
                role: spec.role.to_string(),
                max_tool_rounds: spec.max_tool_rounds,
                system_prompt,
                message: task_message(spec, task),
                tools,
                writable: spec.writable,
                web_search: spec.web_search,
                project_rules: spec.project_rules,
                task: task.to_string(),
                background: wants_background(parent, arguments),
            },
            SessionLink::of(parent)?,
            parent,
        )
        .await
    })
}

/// How many agents `run_agent` lists for one search.
const AGENT_LIST_LIMIT: usize = 10;

/// What a custom agent is told about reporting back, after its own prompt:
/// files written for Claude Code expect to talk to a person.
const CUSTOM_AGENT_REPORTING: &str = "\n\n## Reporting back\n\nYou are running as a subagent: your final reply goes to the agent that delegated the task, not to a person, and nobody can answer questions in between. End it with a `<final_answer>` block holding everything that agent needs: what you found or changed, with file paths and line ranges, and anything left undone.";

/// One line per agent, for the parent to choose from.
fn agent_listing(agents: &[&CustomAgent]) -> String {
    agents
        .iter()
        .map(|agent| {
            let access = if agent.writable() {
                "writes"
            } else {
                "read-only"
            };
            format!("- {} ({access}): {}", agent.name, agent.description)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `run_agent` tool: runs the custom agent `arguments["agent"]` names on
/// `arguments["task"]`, or with `agent: "list"` searches the agents.
pub fn run_custom<'a>(
    parent: &'a ToolExecutor,
    arguments: &'a serde_json::Map<String, Value>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let workspace = parent.workspace();
        let agents = parent.custom_agents();
        let name = arguments
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .context("missing or invalid 'agent' argument")?;
        let task = arguments
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if name.eq_ignore_ascii_case("list") {
            let mut found = agents.search(task, AGENT_LIST_LIMIT);
            found.retain(|agent| parent.may_start(&agent.name));
            return Ok(if found.is_empty() {
                format!("No agent matches '{task}'.")
            } else {
                agent_listing(&found)
            });
        }
        let Some(agent) = agents.find(name) else {
            let near = agents.search(&format!("{name} {task}"), AGENT_LIST_LIMIT);
            let mut message = format!("unknown agent '{name}'");
            if !near.is_empty() {
                message.push_str("; closest matches:\n");
                message.push_str(&agent_listing(&near));
            }
            anyhow::bail!(message);
        };
        if task.trim().is_empty() {
            anyhow::bail!("missing or invalid 'task' argument");
        }
        if !parent.may_start(&agent.name) {
            anyhow::bail!("this agent may not start '{}'", agent.name);
        }

        let mut system_prompt = format!("{}{CUSTOM_AGENT_REPORTING}", agent.prompt);
        // `isolation: worktree`: its tools work in a checkout of its own.
        let worktree = if agent.isolated {
            Some(crate::subagent_worktree::Worktree::create(
                workspace,
                &agent.name,
            )?)
        } else {
            None
        };
        let (tools, additions) = custom_tools(agent, parent, worktree.as_ref()).await;
        system_prompt.push_str(&additions);
        launch(
            Launch {
                recipe: crate::subagent_tasks::Recipe::Custom(agent.name.clone()),
                history: Vec::new(),
                worktree,
                name: agent.name.clone(),
                role: agent.role().to_string(),
                max_tool_rounds: agent.max_tool_rounds(),
                system_prompt,
                message: format!("<task>\n{task}\n</task>\n\n{FINAL_ANSWER_REMINDER}"),
                tools,
                writable: agent.writable(),
                web_search: agent.may_use("web_search") || agent.may_use("browse"),
                project_rules: !agent.omit_project_rules,
                task: task.to_string(),
                background: wants_background(parent, arguments) || agent.background,
            },
            SessionLink::of(parent)?,
            parent,
        )
        .await
    })
}

/// The tools of built-in subagent `spec`, and what its prompt adds about
/// them. Writers share the parent's MCP services and (but for document
/// writers) the workspace's skills; read-only executors attach neither.
fn spec_tools(spec: &SubagentSpec, parent: &ToolExecutor) -> (ToolExecutor, String) {
    let workspace = parent.workspace();
    if spec.tool == SSH.tool {
        let tools = ToolExecutor::new_for_subagent(workspace)
            .with_ssh_tools(parent.ssh())
            .with_allowed_tools(SSH_SUBAGENT_TOOLS.iter().map(ToString::to_string).collect());
        return (tools, String::new());
    }
    let tools = if spec.writable {
        ToolExecutor::new_for_subagent(workspace).with_subagent_mcp(parent.mcp())
    } else {
        ToolExecutor::new_read_only(workspace)
    };
    let (tools, skills) = if spec.writable && spec.tool != DOCUMENT_TOOL {
        attach_skills(tools, &[])
    } else {
        (tools, String::new())
    };
    let additions = format!("{skills}{}", mcp_instructions(&tools));
    (tools, additions)
}

/// The tools of custom agent `agent`, working in `worktree` when it has
/// one, and what its prompt adds about them.
async fn custom_tools(
    agent: &CustomAgent,
    parent: &ToolExecutor,
    worktree: Option<&crate::subagent_worktree::Worktree>,
) -> (ToolExecutor, String) {
    let workspace = parent.workspace();
    let mut prompt = String::new();
    let root = worktree.map_or_else(|| workspace.to_path_buf(), |worktree| worktree.workspace());
    let (mut tools, restrictions) =
        restrict_to_agent(agent, ToolExecutor::new_for_subagent(&root), workspace);
    if worktree.is_some() {
        prompt.push_str(&format!(
            "\n\nYou work in an isolated git worktree at {} on a branch of your own: \
             what you change stays out of the main workspace until someone merges it.",
            root.display()
        ));
    }
    prompt.push_str(&restrictions);
    if crate::agents::custom::SSH_TOOLS
        .iter()
        .any(|tool| agent.may_use(tool))
    {
        tools = tools.with_ssh_tools(parent.ssh());
    }
    // `Agent` in its `tools`: it may start subagents itself, as deep as
    // `[orangu].subagent_depth` levels below the main conversation.
    if let Some(startable) = agent.starts_agents() {
        if parent.depth() + 1 < parent.max_depth() {
            tools = tools.nested_under(parent, startable);
        } else {
            prompt
                .push_str("\n\nYou cannot start subagents here: this is as deep as subagents go.");
        }
    }
    let shared = parent.mcp();
    let (own, unavailable) = connect_agent_mcp(agent, shared.as_deref(), workspace).await;
    tools = tools.with_subagent_mcp(shared);
    if let Some(own) = own {
        tools = tools.with_agent_mcp(own);
    }
    if !unavailable.is_empty() {
        prompt.push_str(&format!(
            "\n\nThese MCP servers are not available here; work without them: {}.",
            unavailable.join("; ")
        ));
    }
    let (tools, skills) = attach_skills(tools, &agent.skills);
    prompt.push_str(&skills);
    prompt.push_str(&mcp_instructions(&tools));
    (tools, prompt)
}

/// A fork's tools: a writer's, with the parent's MCP services and the
/// workspace's skills.
fn fork_tools(parent: &ToolExecutor) -> ToolExecutor {
    let tools = ToolExecutor::new_for_subagent(parent.workspace()).with_subagent_mcp(parent.mcp());
    attach_skills(tools, &[]).0
}
/// Gives `tools` the workspace's skills (read at each launch, so new ones
/// apply) and returns what the subagent's prompt says about them: the
/// skills it starts with (`preload`), in full, and the index `use_skill`
/// loads from, when its tools include it.
fn attach_skills(tools: ToolExecutor, preload: &[String]) -> (ToolExecutor, String) {
    let skills = crate::skills::SkillRegistry::discover(tools.workspace());
    let mut prompt = String::new();
    let mut missing = Vec::new();
    for name in preload {
        match skills.find(name) {
            Some(skill) => {
                prompt.push_str("\n\n");
                prompt.push_str(&skill.render_activation(""));
            }
            None => missing.push(name.as_str()),
        }
    }
    if !missing.is_empty() {
        prompt.push_str(&format!(
            "\n\nThese skills are not installed; work without them: {}.",
            missing.join(", ")
        ));
    }
    let tools = tools.with_skills(Arc::new(skills.clone()));
    if tools
        .definitions()
        .iter()
        .any(|def| def.function.name == "use_skill")
    {
        prompt.push_str("\n\n");
        prompt.push_str(&skills.subagent_index());
    }
    (tools, prompt)
}

/// Limits `tools` to what `agent` may use (`tools`, `disallowedTools`) and
/// gives it the agent's memory; returns what the agent's prompt says about
/// both.
fn restrict_to_agent(
    agent: &CustomAgent,
    mut tools: ToolExecutor,
    workspace: &Path,
) -> (ToolExecutor, String) {
    let mut prompt = String::new();
    let unsupported = agent.unsupported_tools();
    if !unsupported.is_empty() {
        prompt.push_str(&format!(
            "\n\nThese tools are not available here; work without them: {}.",
            unsupported.join(", ")
        ));
    }
    // The workspace's, also for an agent working in a worktree.
    let memory_dir = agent.memory_dir(workspace);
    if let Some(dir) = &memory_dir {
        prompt.push_str(&crate::agents::custom::memory_instructions(agent, dir));
        tools = tools.with_memory_file(dir.join("MEMORY.md"));
    }
    if let Some(mut allowed) = agent.allowed_tools() {
        // Memory comes with its own tool, whatever `tools` lists.
        if memory_dir.is_some() {
            allowed.insert("save_memory".to_string());
        }
        tools = tools.with_allowed_tools(allowed);
    }
    let denied = agent.denied_tools();
    if !denied.is_empty() {
        tools = tools.with_denied_tools(denied);
    }
    if agent.plans() {
        prompt.push_str("\n\nYou are planning: look, but change nothing.");
    }
    (tools.with_agent_hooks(agent.hooks.clone()), prompt)
}

/// The model requests a fork may make.
pub const FORK_MAX_TOOL_ROUNDS: usize = 15;

/// Forks the conversation (`/fork`): a subagent that starts from
/// `messages`, everything said so far, on the parent's model and with its
/// tools (MCP and skills included, but no subagents), does `task` in the
/// background and reports like any background task. The parent
/// conversation goes on untouched.
pub fn fork(parent: &ToolExecutor, messages: &[crate::llm::ChatMessage], task: &str) -> Result<()> {
    let task = task.trim();
    if task.is_empty() {
        anyhow::bail!("say what the fork should do: /fork <task>");
    }
    if !parent.background_tasks() {
        anyhow::bail!("forks run in the background, which only an interactive session has");
    }
    if messages.is_empty() {
        anyhow::bail!("there is no conversation to fork yet");
    }
    let link = SessionLink::of(parent)?;
    let tools = fork_tools(parent);
    let launch_parent = parent.clone();
    let forked = Launch {
        recipe: crate::subagent_tasks::Recipe::Fork,
        worktree: None,
        history: messages.to_vec(),
        name: "fork".to_string(),
        role: "code".to_string(),
        max_tool_rounds: FORK_MAX_TOOL_ROUNDS,
        system_prompt: String::new(),
        message: format!(
            "<task>\n{task}\n</task>\n\nYou are a fork of this conversation: work on the task above \
             on your own, with the tools you have (you cannot start subagents), while the \
             conversation goes on without you.\n\n{FINAL_ANSWER_REMINDER}"
        ),
        tools,
        writable: true,
        web_search: true,
        project_rules: false,
        task: task.to_string(),
        background: true,
    };
    tokio::spawn(async move {
        // In the background from the start: what comes back here is only
        // that it started, and its report arrives as a notification.
        let _ = launch(forked, link, &launch_parent).await;
    });
    Ok(())
}
/// The first turn of a conversation run as agent `name`: its
/// `initialPrompt`.
pub fn initial_prompt(tools: &ToolExecutor, name: &str) -> Option<String> {
    tools.custom_agents().find(name)?.initial_prompt.clone()
}

/// Runs the main conversation as custom agent `name` (`--agent`): limits
/// `tools` as the agent's fields say and returns what the main system
/// prompt adds: the agent's own prompt, its memory and its skills. The
/// session keeps its connection, whatever the agent's `model` says.
pub fn as_main_agent(tools: ToolExecutor, name: &str) -> Result<(ToolExecutor, String)> {
    let agents = tools.custom_agents();
    let Some(agent) = agents.find(name) else {
        let near = agents.search(name, AGENT_LIST_LIMIT);
        let mut message = format!("--agent: unknown agent '{name}'");
        if !near.is_empty() {
            message.push_str("; closest matches:\n");
            message.push_str(&agent_listing(&near));
        } else {
            message.push_str("; `/agents` lists the installed ones");
        }
        anyhow::bail!(message);
    };
    let mut prompt = format!(
        "\n\n## Agent: {}\n\nThis session runs as the agent '{}'. Its instructions follow and \
         take precedence over the general ones above where they differ.\n\n{}",
        agent.name, agent.name, agent.prompt
    );
    let workspace = tools.workspace().to_path_buf();
    let (mut tools, restrictions) = restrict_to_agent(agent, tools, &workspace);
    prompt.push_str(&restrictions);
    if let Some(Some(names)) = agent.starts_agents() {
        tools = tools.with_startable_agents(names);
    }
    let skills = crate::skills::SkillRegistry::discover(tools.workspace());
    for skill in &agent.skills {
        match skills.find(skill) {
            Some(skill) => {
                prompt.push_str("\n\n");
                prompt.push_str(&skill.render_activation(""));
            }
            None => prompt.push_str(&format!("\n\nThe skill '{skill}' is not installed.")),
        }
    }
    Ok((tools, prompt))
}

/// What marks an agent mention in a prompt: `@agent-<name>`, as in Claude
/// Code.
const MENTION: &str = "@agent-";

/// The agent names `prompt` mentions (`@agent-<name>`), in order, each
/// once. A mention starts a word, so an e-mail address is not one.
fn agent_mentions(prompt: &str) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for (start, _) in prompt.match_indices(MENTION) {
        let starts_word = prompt[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_' && c != '.');
        let rest = &prompt[start + MENTION.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')))
            .unwrap_or(rest.len());
        let name = rest[..end].trim_end_matches(['.', ':']);
        if starts_word && !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// `prompt` with what its agent mentions (`@agent-<name>`) ask for: a note,
/// for each agent `tools` can start, to delegate the request to it. A
/// mention of an agent that is not there is left as it is.
pub fn with_agent_mentions(prompt: &str, tools: &ToolExecutor) -> String {
    let mentions = agent_mentions(prompt);
    if mentions.is_empty() {
        return prompt.to_string();
    }
    let offered: std::collections::HashSet<String> = tools
        .definitions()
        .into_iter()
        .map(|def| def.function.name)
        .collect();
    let agents = tools.custom_agents();
    let mut notes = Vec::new();
    for name in mentions {
        let built_in = SUBAGENTS
            .iter()
            .map(|spec| spec.tool)
            .chain([DOCUMENT_TOOL])
            .find(|tool| *tool == name);
        if let Some(tool) = built_in.filter(|tool| offered.contains(*tool)) {
            notes.push(format!("call {tool}"));
        } else if offered.contains("run_agent")
            && let Some(agent) = agents.find(name)
        {
            notes.push(format!("call run_agent with agent \"{}\"", agent.name));
        }
    }
    if notes.is_empty() {
        return prompt.to_string();
    }
    format!(
        "{prompt}\n\n(The user asked for this to be delegated: {}, and give it the whole \
         request as its task.)",
        notes.join("; ")
    )
}
/// What a server's MCP instructions add to a subagent's prompt, for the
/// services `tools` offers it.
fn mcp_instructions(tools: &ToolExecutor) -> String {
    let instructions = tools.mcp_instructions();
    if instructions.is_empty() {
        String::new()
    } else {
        format!("\n\n{instructions}")
    }
}

/// Connects the servers `agent` defines in `mcpServers` that `shared` does
/// not already provide, and says which of its servers it cannot have.
async fn connect_agent_mcp(
    agent: &CustomAgent,
    shared: Option<&crate::mcp::McpManager>,
    workspace: &std::path::Path,
) -> (Option<Arc<crate::mcp::McpManager>>, Vec<String>) {
    use crate::agents::custom::AgentMcpServer;
    let shared_tools: Vec<String> = shared
        .map(|mcp| {
            mcp.definitions()
                .into_iter()
                .map(|def| def.function.name)
                .collect()
        })
        .unwrap_or_default();
    let configured = |name: &str| {
        let prefix = format!("mcp__{name}__");
        shared_tools.iter().any(|tool| tool.starts_with(&prefix))
    };
    let mut configurations = std::collections::HashMap::new();
    let mut unavailable = Vec::new();
    for server in &agent.mcp_servers {
        match server {
            AgentMcpServer::Reference(name) if !configured(name) => {
                unavailable.push(format!("{name} (not configured)"));
            }
            AgentMcpServer::Reference(_) => {}
            // A configured server of the same name wins.
            AgentMcpServer::Http { name, .. } if configured(name) => {}
            AgentMcpServer::Http { name, .. } if !crate::config::valid_mcp_name(name) => {
                unavailable.push(format!("{name} (invalid name)"));
            }
            AgentMcpServer::Http { name, url } => {
                configurations.insert(
                    name.clone(),
                    crate::config::McpServerConfiguration {
                        endpoint: url.clone(),
                        startup_timeout_seconds: crate::config::default_mcp_timeout(),
                        tool_timeout_seconds: crate::config::default_mcp_timeout(),
                        enabled: true,
                        required: false,
                        enabled_tools: Vec::new(),
                        disabled_tools: Vec::new(),
                        approval_mode: crate::config::McpApprovalMode::Auto,
                    },
                );
            }
            AgentMcpServer::Unsupported { name, reason } => {
                unavailable.push(format!("{name} ({reason})"));
            }
        }
    }
    if configurations.is_empty() {
        return (None, unavailable);
    }
    match crate::mcp::McpManager::connect_all(&configurations, workspace).await {
        Ok((mcp, warnings)) => {
            unavailable.extend(warnings);
            (Some(mcp), unavailable)
        }
        Err(error) => {
            unavailable.push(format!("{error:#}"));
            (None, unavailable)
        }
    }
}
/// One subagent run, whatever defined the subagent: a built-in
/// [`SubagentSpec`] or a [`CustomAgent`] file.
struct Launch {
    /// How its tools were built, to build them again after a restart.
    recipe: crate::subagent_tasks::Recipe,
    /// The worktree its tools work in (`isolation: worktree`).
    worktree: Option<crate::subagent_worktree::Worktree>,
    /// The conversation it starts from, system prompt included: a fork's
    /// copy of its parent's. Empty for a fresh one.
    history: Vec<crate::llm::ChatMessage>,
    /// Names the subagent in errors.
    name: String,
    role: String,
    max_tool_rounds: usize,
    system_prompt: String,
    /// The user message carrying the task.
    message: String,
    tools: ToolExecutor,
    writable: bool,
    /// Gives the executor the configured web search backend.
    web_search: bool,
    /// Gives the subagent the `AGENTS.md` rules the main agent follows.
    project_rules: bool,
    /// The task, for `/tasks`.
    task: String,
    /// Asked to run without the parent waiting (honoured where the parent
    /// allows background tasks).
    background: bool,
}

/// What a subagent starts from: the connection the main conversation is
/// using, the configured server sections, and where web searches go.
pub struct SessionLink {
    pub active: LlmConfiguration,
    pub llms: HashMap<String, LlmConfiguration>,
    pub web_engines: Vec<crate::web_search::Engine>,
}

impl SessionLink {
    /// As `parent`'s conversation left it: the connection of its latest
    /// request, so `-c` and `/server` carry over. An executor no
    /// conversation has used yet falls back to the configuration file.
    pub fn of(parent: &ToolExecutor) -> Result<Self> {
        if let Some(active) = parent.active_profile() {
            return Ok(Self {
                active,
                llms: parent.llms().clone(),
                web_engines: parent.web_engines().to_vec(),
            });
        }
        let config_path = default_client_config_path().context("no config path")?;
        let config = load_client_configuration(&config_path)
            .context("failed to load LLM configuration for subagent")?;
        let active = config
            .llms
            .get(&config.default_server)
            .cloned()
            .context("missing default server in config")?;
        Ok(Self {
            active,
            llms: config.llms,
            web_engines: config.web_search,
        })
    }
}

/// What a subagent that ran out of model requests is asked, without tools.
const PARTIAL_ANSWER_REQUEST: &str = "You have used every step you were given, so stop working now. Without calling any tool, report back: what you found or changed so far (with file paths and line ranges), and exactly what is left to do. Wrap the report in a `<final_answer>...</final_answer>` block.";

/// The most a partial answer may run to.
const PARTIAL_ANSWER_TOKENS: u32 = 2_048;

/// A partial answer as the parent receives it: marked, so it is not taken
/// for finished work.
fn partial_answer(answer: &str, rounds: usize, writable: bool) -> String {
    let report = final_answer(answer).unwrap_or(answer.trim());
    let mut marked =
        format!("[Partial: the subagent used all {rounds} of its model requests before finishing.");
    if writable {
        marked.push_str(" Files it changed remain in the workspace.");
    }
    marked.push_str("]\n\n");
    marked.push_str(report);
    marked
}

/// The system prompt a subagent runs with: its own, then the project's
/// `AGENTS.md` rules when it takes them, then the workspace root listing.
fn compose_system_prompt(base: &str, project_rules: bool, workspace: &Path) -> String {
    let rules = if project_rules {
        crate::config::load_agents_instructions(workspace)
    } else {
        String::new()
    };
    format!(
        "{base}{rules}\n\nThe directory listing of the workspace root is:\n{}\n",
        root_listing(workspace)
    )
}

async fn launch(mut launch: Launch, link: SessionLink, parent: &ToolExecutor) -> Result<String> {
    let workspace = parent.workspace();
    let config = profile_for(
        &link,
        &launch.role,
        !launch.history.is_empty(),
        launch.max_tool_rounds,
    )
    .await;

    let mut system_prompt = launch.system_prompt.clone();
    if launch.web_search {
        let engines = link.web_engines.clone();
        if engines.is_empty() {
            if launch.name == WEB_RESEARCHER.tool {
                anyhow::bail!("web search is off ([orangu].web_search)");
            }
            system_prompt.push_str("\n\nWeb search is not available here; work without it.");
        }
        launch.tools = launch.tools.with_web_search(engines).with_browsing();
    }
    let system_prompt = compose_system_prompt(&system_prompt, launch.project_rules, workspace);
    let model = config.model.clone();
    let mut session = ChatSession::new(&system_prompt).keep_turn_when_rounds_run_out();
    if !launch.history.is_empty() {
        session.restore(std::mem::take(&mut launch.history));
    }
    let resumable = Arc::new(tokio::sync::Mutex::new(Resumable {
        session,
        tools: launch.tools,
        profile: config,
        writable: launch.writable,
        max_tool_rounds: launch.max_tool_rounds,
        worktree: launch.worktree,
        recipe: launch.recipe,
        role: launch.role.clone(),
        web_search: launch.web_search,
    }));
    let tasks = parent.tasks().clone();
    let background = launch.background && parent.background_tasks();
    let id = tasks.start(&launch.name, &launch.task, &model, background);
    tasks.set_resumable(&id, resumable.clone());
    dispatch(
        parent,
        tasks,
        Turn {
            id,
            name: launch.name,
            resumable,
            message: launch.message,
            background,
        },
    )
    .await
}

/// The connection a subagent with `role` uses: the section for its role
/// (or the coordinator's role), or, for a fork (`inherit`), the parent's
/// own, since it carries on the parent's conversation.
async fn profile_for(
    link: &SessionLink,
    role: &str,
    inherit: bool,
    max_tool_rounds: usize,
) -> LlmConfiguration {
    let mut config = if inherit {
        link.active.clone()
    } else {
        let is_coordinator = crate::llm::probe_coordinator(
            &reqwest::Client::new(),
            &link.active.endpoint,
            link.active.api_key.as_deref(),
        )
        .await
        .is_some();
        target_profile(link, is_coordinator, role)
    };
    config.max_tool_rounds = max_tool_rounds;
    config
}

/// A task restored from disk, made ready to continue: its tools built
/// again from its recipe, on today's configuration, with its conversation.
async fn rebuild(
    parent: &ToolExecutor,
    saved: crate::subagent_tasks::SavedResume,
) -> Result<Resumable> {
    use crate::subagent_tasks::Recipe;
    let link = SessionLink::of(parent)?;
    let mut tools = match &saved.recipe {
        Recipe::BuiltIn(tool) => {
            spec_tools(
                find(tool).with_context(|| format!("no subagent '{tool}'"))?,
                parent,
            )
            .0
        }
        Recipe::Document(kind) => {
            let writer = find_document_writer(kind)
                .with_context(|| format!("no document writer for '{kind}'"))?;
            spec_tools(&writer.spec, parent).0
        }
        Recipe::Custom(name) => {
            let agents = parent.custom_agents();
            let agent = agents
                .find(name)
                .with_context(|| format!("the agent '{name}' is no longer installed"))?;
            custom_tools(agent, parent, saved.worktree.as_ref()).await.0
        }
        Recipe::Fork => fork_tools(parent),
    };
    if saved.web_search && !link.web_engines.is_empty() {
        tools = tools
            .with_web_search(link.web_engines.clone())
            .with_browsing();
    }
    let profile = profile_for(
        &link,
        &saved.role,
        saved.recipe == Recipe::Fork,
        saved.max_tool_rounds,
    )
    .await;
    let mut session = ChatSession::new("").keep_turn_when_rounds_run_out();
    session.restore(saved.transcript);
    Ok(Resumable {
        session,
        tools,
        profile,
        writable: saved.writable,
        max_tool_rounds: saved.max_tool_rounds,
        worktree: saved.worktree,
        recipe: saved.recipe,
        role: saved.role,
        web_search: saved.web_search,
    })
}
/// How a built-in subagent's tools are built: by its tool, or for a document
/// writer by its type.
fn recipe_of(spec: &SubagentSpec) -> crate::subagent_tasks::Recipe {
    use crate::subagent_tasks::Recipe;
    match DOCUMENT_WRITERS
        .iter()
        .find(|writer| writer.spec.system_prompt == spec.system_prompt)
    {
        Some(writer) if spec.tool == DOCUMENT_TOOL => Recipe::Document(writer.kind.to_string()),
        _ => Recipe::BuiltIn(spec.tool.to_string()),
    }
}

/// Whether a subagent tool call runs in the background: as it asks, or
/// as `[orangu].subagent_background` says when it does not.
fn wants_background(parent: &ToolExecutor, arguments: &serde_json::Map<String, Value>) -> bool {
    arguments
        .get("background")
        .and_then(Value::as_bool)
        .unwrap_or(parent.background_default())
}

/// Marks a foreground task stopped if the future running it is dropped
/// before it ends; forgotten once it has.
struct InterruptOnDrop<'a> {
    tasks: &'a TaskRegistry,
    id: &'a str,
    /// The task running it, aborted with the turn.
    abort: Option<tokio::task::AbortHandle>,
}

impl Drop for InterruptOnDrop<'_> {
    fn drop(&mut self) {
        self.tasks.interrupt(self.id);
        if let Some(abort) = &self.abort {
            abort.abort();
        }
    }
}

/// One turn of a subagent task: its first, or a continuation.
struct Turn {
    id: String,
    name: String,
    resumable: Arc<tokio::sync::Mutex<Resumable>>,
    message: String,
    background: bool,
}

/// Runs `turn` and returns its report, or starts it in the background and
/// says so; the report then reaches the registry's notifications.
async fn dispatch(parent: &ToolExecutor, tasks: Arc<TaskRegistry>, turn: Turn) -> Result<String> {
    let Turn {
        id,
        name,
        resumable,
        message,
        background,
    } = turn;
    if background {
        let running = tasks.clone();
        let task_id = id.clone();
        let handle = tokio::spawn(async move {
            let (status, report) = match take_turn(&resumable, &message, &running, &task_id).await {
                Ok(ended) => ended,
                Err(err) => (TaskStatus::Failed, format!("error: {err:#}")),
            };
            running.finish(&task_id, status, &report);
        });
        tasks.set_abort(&id, handle.abort_handle());
        return Ok(format!(
            "Started subagent task {id} ({name}) in the background. Its report will arrive as a \
             separate message when it finishes; carry on with other work meanwhile."
        ));
    }
    // In a task of its own, so the user can move it to the background
    // (Ctrl+B) and the turn waiting here go on; it records how it ended.
    let running = tasks.clone();
    let task_id = id.clone();
    let mut handle = tokio::spawn(async move {
        let ended = take_turn(&resumable, &message, &running, &task_id).await;
        match &ended {
            Ok((status, report)) => running.finish(&task_id, *status, report),
            Err(err) => running.finish(&task_id, TaskStatus::Failed, &format!("{err:#}")),
        }
        ended
    });
    // Should the turn waiting here be cancelled (Esc), this future is
    // dropped mid-way: the guard stops the task rather than letting it run
    // on unseen.
    let guard = InterruptOnDrop {
        tasks: &tasks,
        id: &id,
        abort: Some(handle.abort_handle()),
    };
    // Only the main conversation's subagents can move: a report needs a
    // conversation to arrive in.
    let joined = if parent.background_tasks() {
        let signal = tasks.detachable(&id);
        tokio::select! {
            joined = &mut handle => joined,
            () = signal.notified() => {
                if tasks.move_to_background(&id, handle.abort_handle()) {
                    std::mem::forget(guard);
                    return Ok(format!(
                        "The user moved subagent task {id} ({name}) to the background. Its report \
                         will arrive as a separate message when it finishes; carry on with other \
                         work meanwhile."
                    ));
                }
                (&mut handle).await
            }
        }
    } else {
        (&mut handle).await
    };
    std::mem::forget(guard);
    let ended =
        joined.map_err(|err| anyhow::anyhow!("subagent task {id} did not finish: {err}"))?;
    match ended {
        Ok((_, report)) => Ok(if parent.background_tasks() {
            format!("{report}\n\n(subagent task {id}; continue_agent can follow it up)")
        } else {
            report
        }),
        Err(err) => Err(err),
    }
}

/// Sends `message` to the subagent in `resumable` and returns how the turn
/// ended and the report for the parent.
/// With a worktree, the turn runs in it (put back if a turn before removed
/// it), and after the turn it is removed if unchanged or reported.
async fn take_turn(
    resumable: &tokio::sync::Mutex<Resumable>,
    message: &str,
    tasks: &Arc<TaskRegistry>,
    id: &str,
) -> Result<(TaskStatus, String)> {
    let worktree = resumable.lock().await.worktree.clone();
    if let Some(worktree) = &worktree {
        worktree.reopen()?;
    }
    let ended = take_turn_in_place(resumable, message, tasks, id).await;
    let Some(worktree) = worktree else {
        return ended;
    };
    match (ended, worktree.settle()) {
        (Ok((status, report)), Ok(Some(note))) => Ok((status, format!("{report}\n\n{note}"))),
        (Ok(ended), Ok(None)) => Ok(ended),
        (Ok((status, report)), Err(err)) => Ok((
            status,
            format!("{report}\n\n[Its worktree could not be checked: {err:#}]"),
        )),
        (Err(err), Ok(Some(note))) => Err(err.context(note)),
        (Err(err), _) => Err(err),
    }
}

async fn take_turn_in_place(
    resumable: &tokio::sync::Mutex<Resumable>,
    message: &str,
    tasks: &Arc<TaskRegistry>,
    id: &str,
) -> Result<(TaskStatus, String)> {
    let mut guard = resumable.lock().await;
    let subagent = &mut *guard;
    let counter = tasks.clone();
    let counted = id.to_string();
    let metered = tasks.clone();
    let metered_id = id.to_string();
    let streamed = tasks.clone();
    let streamed_id = id.to_string();

    // Silent: nothing streams to the screen, only into what `/tasks` shows
    // (its recent output, the tool it is on, tokens), and MCP calls that need approval (never offered to a
    // subagent executor anyway) are denied rather than prompting.
    let response = subagent
        .session
        .prompt(
            message,
            &subagent.profile,
            &subagent.tools,
            move |text: &str| streamed.add_output(&streamed_id, text),
            move |metrics: crate::llm::StreamMetrics| {
                if let Some(usage) = metrics.usage {
                    metered.add_usage(&metered_id, usage);
                }
            },
            |_| {},
            move |call: &crate::llm::ToolCall| {
                counter.count_tool_call(&counted, &call.function.name)
            },
            |_| false,
        )
        .await;

    match response {
        Ok(response) => Ok((
            TaskStatus::Completed,
            match final_answer(&response) {
                Some(answer) => answer.to_string(),
                None => format!(
                    "[Subagent failed to format output in <final_answer> tags. Raw response follows:]\n\n{response}"
                ),
            },
        )),
        // Out of rounds: what it did is still in its transcript, so ask for
        // an answer from that rather than losing the work.
        Err(error)
            if error
                .downcast_ref::<crate::session::RoundsExhausted>()
                .is_some() =>
        {
            match subagent
                .session
                .prompt_without_tools(
                    PARTIAL_ANSWER_REQUEST,
                    &subagent.profile,
                    PARTIAL_ANSWER_TOKENS,
                    |_| {},
                    |metrics: crate::llm::StreamMetrics| {
                        if let Some(usage) = metrics.usage {
                            tasks.add_usage(id, usage);
                        }
                    },
                )
                .await
            {
                Ok(answer) => Ok((
                    TaskStatus::Partial,
                    partial_answer(&answer, subagent.max_tool_rounds, subagent.writable),
                )),
                Err(_) => Err(error),
            }
        }
        Err(error) if subagent.writable => Err(error.context(
            "subagent stopped before finishing; files it changed remain in the workspace, check \
             `git status`",
        )),
        Err(error) => Err(error),
    }
}

/// The `continue_agent` tool: a follow-up `message` to finished task `task`,
/// which picks up its own conversation where it ended.
pub fn continue_task<'a>(
    parent: &'a ToolExecutor,
    arguments: &'a serde_json::Map<String, Value>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let id = arguments
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .context("missing or invalid 'task' argument")?;
        let message = arguments
            .get("message")
            .and_then(|v| v.as_str())
            .filter(|message| !message.trim().is_empty())
            .context("missing or invalid 'message' argument")?;
        let background = wants_background(parent, arguments) && parent.background_tasks();
        let tasks = parent.tasks().clone();
        // One restored from disk is built again first.
        let resumable = match (tasks.resumable(id), tasks.restorable(id)) {
            (Some(resumable), _) => Some(resumable),
            (None, Some(saved)) => {
                let resumable = Arc::new(tokio::sync::Mutex::new(rebuild(parent, saved).await?));
                tasks.set_resumable(id, resumable.clone());
                Some(resumable)
            }
            (None, None) => None,
        };
        let (Some(resumable), Some(task)) = (resumable, tasks.get(id)) else {
            anyhow::bail!("there is no subagent task {id} to continue");
        };
        if !tasks.restart(id, message, background) {
            anyhow::bail!("subagent task {id} is still running; wait for its report");
        }
        dispatch(
            parent,
            tasks,
            Turn {
                id: id.to_string(),
                name: task.agent,
                resumable,
                message: format!("{message}\n\n{FINAL_ANSWER_REMINDER}"),
                background,
            },
        )
        .await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientAppConfiguration;
    use std::io::Write;

    fn config(text: &str) -> ClientAppConfiguration {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{text}").unwrap();
        load_client_configuration(file.path()).unwrap()
    }

    /// The link a session connected to section `active` of `text` has.
    fn link(text: &str, active: &str) -> SessionLink {
        let config = config(text);
        SessionLink {
            active: config.llms[active].clone(),
            llms: config.llms,
            web_engines: Vec::new(),
        }
    }

    const TWO_SERVERS: &str = "[orangu]\nserver = main\n\n[main]\nendpoint = http://localhost:9000/v1\nmodel = all\nrole = all\n\n[other]\nendpoint = http://localhost:9200/v1\nmodel = org/other\nrole = all\n\n[explorer-server]\nendpoint = http://localhost:9111/v1\nmodel = org/qwen\nrole = explorer\n";

    #[test]
    fn target_profile_under_a_coordinator_ignores_local_role_sections() {
        // Once a coordinator is confirmed, it alone owns model/role
        // decisions: even a dedicated `role = explorer` section (with its
        // own distinct endpoint/model) must be ignored in favor of the
        // active connection with `.model` forced to the role.
        let link = link(TWO_SERVERS, "main");
        let profile = target_profile(&link, true, "explorer");
        assert_eq!(profile.endpoint, "http://localhost:9000/v1");
        assert_eq!(profile.model, "explorer");

        let profile = target_profile(&link, true, "code");
        assert_eq!(profile.endpoint, "http://localhost:9000/v1");
        assert_eq!(profile.model, "code");
    }

    #[test]
    fn target_profile_without_a_coordinator_uses_the_dedicated_role_section() {
        let profile = target_profile(&link(TWO_SERVERS, "main"), false, "explorer");
        assert_eq!(profile.endpoint, "http://localhost:9111/v1");
        assert_eq!(profile.model, "org/qwen");
    }

    /// Without a dedicated section, a subagent follows the connection the
    /// conversation is using (`-c`, `/server`), not the file's default.
    #[test]
    fn target_profile_without_a_section_follows_the_active_connection() {
        let switched = link(TWO_SERVERS, "other");
        for role in ["code", "review"] {
            let profile = target_profile(&switched, false, role);
            assert_eq!(profile.endpoint, "http://localhost:9200/v1");
            assert_eq!(profile.model, "org/other");
        }
        let profile = target_profile(&switched, true, "code");
        assert_eq!(profile.endpoint, "http://localhost:9200/v1");
        assert_eq!(profile.model, "code");
    }

    /// The link comes from the executor the conversation used: its latest
    /// connection, its sections and its web search engines.
    #[test]
    fn session_link_comes_from_the_parent_executor() {
        let config = config(TWO_SERVERS);
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path())
            .with_llms(config.llms.clone())
            .with_web_search(crate::web_search::DEFAULT_ENGINES.to_vec());
        // A clone, as the chat loop holds one: the record is shared.
        parent.clone().set_active_profile(&config.llms["other"]);
        let link = SessionLink::of(&parent).unwrap();
        assert_eq!(link.active.endpoint, "http://localhost:9200/v1");
        assert_eq!(link.llms.len(), 3);
        assert_eq!(link.web_engines, crate::web_search::DEFAULT_ENGINES);
    }

    #[test]
    fn subagents_take_the_project_rules_unless_they_opt_out() {
        let workspace = tempfile::tempdir().unwrap();
        fs_write(
            &workspace.path().join("AGENTS.md"),
            "Always run cargo fmt.\n",
        );
        let with = compose_system_prompt("You write tests.", true, workspace.path());
        assert!(with.starts_with("You write tests."), "{with}");
        assert!(with.contains("Always run cargo fmt."), "{with}");
        assert!(with.contains("The directory listing of the workspace root is:\nAGENTS.md\n"));
        let without = compose_system_prompt("You search.", false, workspace.path());
        assert!(!without.contains("Always run cargo fmt."), "{without}");

        const { assert!(!EXPLORER.project_rules) };
        for spec in [&TEST_WRITER, &DOC_WRITER, &WEB_RESEARCHER] {
            assert!(spec.project_rules, "{}", spec.tool);
        }
        assert!(DOCUMENT_WRITERS.iter().all(|w| w.spec.project_rules));
    }

    fn fs_write(path: &Path, text: &str) {
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn find_resolves_every_subagent_tool_and_nothing_else() {
        for spec in SUBAGENTS {
            assert_eq!(find(spec.tool).map(|s| s.tool), Some(spec.tool));
        }
        assert!(find("show_file").is_none());
    }

    const _: () = assert!(!EXPLORER.writable && TEST_WRITER.writable && DOC_WRITER.writable);

    #[test]
    fn definition_requires_the_task_argument() {
        let definition = TEST_WRITER.definition();
        assert_eq!(definition.function.name, "write_tests");
        assert_eq!(definition.function.parameters["required"], json!(["task"]));
        assert_eq!(
            definition.function.parameters["properties"]["task"]["type"],
            "string"
        );
    }

    #[test]
    fn a_partial_answer_is_marked_and_says_files_remain() {
        let answer = partial_answer(
            "Notes.\n<final_answer>\nDid A; B is left.\n</final_answer>",
            12,
            true,
        );
        assert_eq!(
            answer,
            "[Partial: the subagent used all 12 of its model requests before finishing. \
             Files it changed remain in the workspace.]\n\n<final_answer>\nDid A; B is left.\n</final_answer>"
        );
        let read_only = partial_answer("  Found X.  ", 6, false);
        assert_eq!(
            read_only,
            "[Partial: the subagent used all 6 of its model requests before finishing.]\n\nFound X."
        );
    }

    #[test]
    fn final_answer_extracts_the_tagged_block() {
        let response = "Found it.\n\n<final_answer>\nsrc/a.rs:1-2\n</final_answer>\ntrailing";
        assert_eq!(
            final_answer(response),
            Some("<final_answer>\nsrc/a.rs:1-2\n</final_answer>")
        );
        assert_eq!(final_answer("no tags here"), None);
        assert_eq!(final_answer("<final_answer> unterminated"), None);
    }

    #[test]
    fn task_message_wraps_the_task_and_demands_a_final_answer() {
        let message = task_message(&DOC_WRITER, "Document the parser.");
        assert!(message.starts_with(DOC_WRITER.instruction));
        assert!(message.contains("<task>\nDocument the parser.\n</task>"));
        assert!(message.contains("<final_answer>"));
    }

    #[test]
    fn there_is_one_writable_document_writer_per_type() {
        let kinds: Vec<&str> = DOCUMENT_WRITERS.iter().map(|w| w.kind).collect();
        assert_eq!(
            kinds,
            [
                "text",
                "spreadsheet",
                "presentation",
                "drawing",
                "markdown",
                "html"
            ]
        );
        for writer in DOCUMENT_WRITERS {
            assert!(writer.spec.writable);
            assert_eq!(writer.spec.tool, DOCUMENT_TOOL);
            assert_eq!(writer.spec.argument, "task");
            assert!(writer.spec.system_prompt.contains("convert_document"));
            assert!(writer.spec.system_prompt.contains("<final_answer>"));
            assert_eq!(
                find_document_writer(writer.kind).map(|w| w.kind),
                Some(writer.kind)
            );
        }
        // Each prompt names how it writes, its source format and target.
        for (kind, how, format, target) in [
            ("text", "ONE convert_document call", "`md`", "`.odt`"),
            (
                "spreadsheet",
                "ONE convert_document call",
                "`csv`",
                "`.ods`",
            ),
            (
                "presentation",
                "ONE convert_document call",
                "`outline`",
                "`.odp`",
            ),
            ("drawing", "ONE convert_document call", "`svg`", "`.odg`"),
            ("markdown", "ONE create_file call", "CommonMark", "`.md`"),
            ("html", "ONE create_file call", "HTML5", "`.html`"),
        ] {
            let prompt = find_document_writer(kind).unwrap().spec.system_prompt;
            assert!(prompt.contains(how), "{kind}");
            assert!(prompt.contains(format), "{kind}");
            assert!(prompt.contains(target), "{kind}");
        }
        assert!(find("write_document").is_none());
    }

    #[test]
    fn document_definition_offers_every_type() {
        let definition = document_definition();
        assert_eq!(definition.function.name, DOCUMENT_TOOL);
        assert_eq!(
            definition.function.parameters["properties"]["type"]["enum"],
            json!([
                "text",
                "spreadsheet",
                "presentation",
                "drawing",
                "markdown",
                "html"
            ])
        );
        assert_eq!(
            definition.function.parameters["required"],
            json!(["type", "task"])
        );
    }

    #[tokio::test]
    async fn run_document_rejects_an_unknown_type() {
        let workspace = tempfile::tempdir().unwrap();
        let mut arguments = serde_json::Map::new();
        arguments.insert("type".into(), json!("database"));
        arguments.insert("task".into(), json!("x"));
        let error = run_document(&ToolExecutor::new(workspace.path()), &arguments)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("unknown document type"),
            "{error}"
        );
        assert!(error.to_string().contains("spreadsheet"), "{error}");
    }

    /// The whole cycle against a stand-in model server: a custom agent runs
    /// in the background, its report becomes a notification, and
    /// `continue_agent` takes it further with its earlier work still in its
    /// conversation.
    #[tokio::test]
    async fn a_background_task_reports_later_and_can_be_continued() {
        let agents_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            agents_dir.path().join("helper.md"),
            "---\nname: helper\ndescription: Helps.\ntools: Read\n---\nYou help.\n",
        )
        .unwrap();
        let agents = crate::agents::custom::AgentRegistry::from_directories(&[(
            agents_dir.path().to_path_buf(),
            crate::agents::custom::AgentSource::User,
        )]);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![
                crate::test_support::text_stream("<final_answer>first report</final_answer>"),
                crate::test_support::text_stream("<final_answer>second report</final_answer>"),
            ],
        );
        let profile = config(&format!(
            "[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"
        ))
        .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path())
            .with_custom_agents(agents)
            .with_background_tasks();
        parent.set_active_profile(&profile);
        let call = |pairs: &[(&str, Value)]| -> serde_json::Map<String, Value> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect()
        };

        let started = parent
            .execute(
                "run_agent",
                &call(&[
                    ("agent", json!("helper")),
                    ("task", json!("look into the parser")),
                    ("background", json!(true)),
                ]),
            )
            .await
            .unwrap();
        assert!(
            started.starts_with("Started subagent task t1 (helper) in the background."),
            "{started}"
        );

        let tasks = parent.tasks().clone();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !tasks.has_notifications() {
            assert!(
                std::time::Instant::now() < deadline,
                "no report: {:?}",
                tasks.list()
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let notifications = tasks.take_notifications();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].id, "t1");
        assert_eq!(notifications[0].status, TaskStatus::Completed);
        assert!(notifications[0].report.contains("first report"));

        let followed = parent
            .execute(
                "continue_agent",
                &call(&[("task", json!("t1")), ("message", json!("now the lexer"))]),
            )
            .await
            .unwrap();
        assert!(followed.contains("second report"), "{followed}");
        assert!(followed.ends_with("(subagent task t1; continue_agent can follow it up)"));
        let task = tasks.get("t1").unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(!task.background);
        assert!(
            !tasks.has_notifications(),
            "a foreground turn does not notify"
        );

        // The follow-up carried the first exchange: the same conversation.
        let bodies = server.join().unwrap();
        assert_eq!(bodies.len(), 2);
        assert!(bodies[1].contains("look into the parser"), "{}", bodies[1]);
        assert!(bodies[1].contains("first report"), "{}", bodies[1]);
        assert!(bodies[1].contains("now the lexer"), "{}", bodies[1]);

        let err = parent
            .execute(
                "continue_agent",
                &call(&[("task", json!("t9")), ("message", json!("x"))]),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no subagent task t9"), "{err}");
    }

    /// Cancelling the turn that waits for a foreground subagent (Esc) leaves
    /// its task stopped, not running forever, and not to be continued.
    #[tokio::test]
    async fn a_cancelled_foreground_task_ends_stopped() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        // Answers the coordinator probe with 404; never answers a chat.
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buffer = [0u8; 4096];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let head = String::from_utf8_lossy(&buffer[..read]).to_string();
                if head.contains("chat/completions") {
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_secs(30));
                        drop(stream);
                    });
                } else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    );
                }
            }
        });
        let profile = config(&format!(
            "[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"
        ))
        .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path()).with_background_tasks();
        parent.set_active_profile(&profile);
        let mut arguments = serde_json::Map::new();
        arguments.insert("task".into(), json!("cover the parser"));

        let tasks = parent.tasks().clone();
        let mut turn = Box::pin(parent.execute("write_tests", &arguments));
        // Let it register and reach the model, then cancel the turn.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            tokio::select! {
                _ = &mut turn => panic!("the model never answers"),
                _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
            }
            if !tasks.list().is_empty() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "task never started");
        }
        drop(turn);

        let task = tasks.get("t1").unwrap();
        assert_eq!(task.status, TaskStatus::Stopped);
        assert!(!task.resumable);
        assert!(task.result.unwrap().contains("cancelled"));
    }

    /// Without a client to deliver reports later (one-shot runs, workflow
    /// loops), `background` is ignored: the subagent runs in the foreground.
    #[tokio::test]
    async fn background_is_ignored_where_reports_cannot_be_delivered() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![crate::test_support::text_stream(
                "<final_answer>tests written</final_answer>",
            )],
        );
        let profile = config(&format!(
            "[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"
        ))
        .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path());
        parent.set_active_profile(&profile);
        let mut arguments = serde_json::Map::new();
        arguments.insert("task".into(), json!("cover the parser"));
        arguments.insert("background".into(), json!(true));
        let report = parent.execute("write_tests", &arguments).await.unwrap();
        assert_eq!(report, "<final_answer>tests written</final_answer>");
        assert!(!parent.tasks().list()[0].background);
        let _ = server.join();
    }

    #[tokio::test]
    async fn run_rejects_a_missing_task_before_contacting_any_server() {
        let workspace = tempfile::tempdir().unwrap();
        let arguments = serde_json::Map::new();
        let error = run(
            &TEST_WRITER,
            &ToolExecutor::new(workspace.path()),
            &arguments,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("'task'"));
    }
}

#[cfg(test)]
mod mcp_tests {
    use super::*;
    use crate::config::{McpApprovalMode, McpServerConfiguration};
    use crate::test_support::serve_mcp;

    fn server(endpoint: String, approval_mode: McpApprovalMode) -> McpServerConfiguration {
        McpServerConfiguration {
            endpoint,
            startup_timeout_seconds: 5,
            tool_timeout_seconds: 5,
            enabled: true,
            required: true,
            enabled_tools: Vec::new(),
            disabled_tools: Vec::new(),
            approval_mode,
        }
    }

    fn names(tools: &ToolExecutor) -> Vec<String> {
        tools
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .filter(|name| name.starts_with("mcp__"))
            .collect()
    }

    #[tokio::test]
    async fn subagents_share_mcp_without_the_tools_that_need_approval() {
        let endpoint = serve_mcp(vec![("lookup", true), ("drop_table", false)]).await;
        let configurations =
            HashMap::from([("db".to_string(), server(endpoint, McpApprovalMode::Writes))]);
        let workspace = tempfile::tempdir().unwrap();
        let (mcp, warnings) =
            crate::mcp::McpManager::connect_all(&configurations, workspace.path())
                .await
                .unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let parent = ToolExecutor::new(workspace.path()).with_mcp(mcp);
        let mut parent_tools = names(&parent);
        parent_tools.sort();
        assert_eq!(parent_tools, ["mcp__db__drop_table", "mcp__db__lookup"]);

        let child =
            ToolExecutor::new_for_subagent(workspace.path()).with_subagent_mcp(parent.mcp());
        assert_eq!(names(&child), ["mcp__db__lookup"]);
        assert!(
            child
                .execute("mcp__db__lookup", &Default::default())
                .await
                .unwrap()
                .contains("called lookup")
        );
        let refused = child
            .execute("mcp__db__drop_table", &Default::default())
            .await
            .unwrap_err();
        assert!(refused.to_string().contains("needs approval"), "{refused}");
        assert!(child.mcp_instructions().contains("Use the fake tools."));

        // An allowlist naming another server leaves none of these.
        let limited = ToolExecutor::new_for_subagent(workspace.path())
            .with_subagent_mcp(parent.mcp())
            .with_allowed_tools(["mcp__other".to_string()].into());
        assert!(names(&limited).is_empty());
        assert!(limited.mcp_instructions().is_empty());
    }

    #[tokio::test]
    async fn an_agent_connects_the_servers_it_defines() {
        let endpoint = serve_mcp(vec![("search", false)]).await;
        let workspace = tempfile::tempdir().unwrap();
        let agent = crate::agents::custom::parse_agent(
            &format!(
                "---\nname: a\ndescription: A.\ntools: Read\nmcpServers:\n  - docs:\n      url: {endpoint}\n  - github\n  - local:\n      command: npx\n---\nA.\n"
            ),
            Path::new("a.md"),
            crate::agents::custom::AgentSource::User,
        )
        .unwrap();
        let (own, unavailable) = connect_agent_mcp(&agent, None, workspace.path()).await;
        assert_eq!(
            unavailable,
            [
                "github (not configured)",
                "local (stdio servers are not supported, only HTTP)"
            ]
        );
        let tools = ToolExecutor::new_for_subagent(workspace.path())
            .with_subagent_mcp(None)
            .with_agent_mcp(own.expect("the agent's server"))
            .with_allowed_tools(agent.allowed_tools().unwrap());
        // Its own server is unattended but trusted: nothing asks first.
        assert_eq!(names(&tools), ["mcp__docs__search"]);
        assert!(
            tools
                .execute("mcp__docs__search", &Default::default())
                .await
                .unwrap()
                .contains("called search")
        );
    }
}

#[cfg(test)]
mod skill_tests {
    use super::*;

    fn workspace_with_skill() -> tempfile::TempDir {
        let workspace = tempfile::tempdir().unwrap();
        let dir = workspace.path().join(".orangu/skills/release-notes");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\ndescription: Write release notes.\n---\nGroup changes by area. $ARGUMENTS\n",
        )
        .unwrap();
        workspace
    }

    #[tokio::test]
    async fn skills_are_preloaded_or_loaded_with_use_skill() {
        let workspace = workspace_with_skill();
        let (tools, prompt) = attach_skills(
            ToolExecutor::new_for_subagent(workspace.path()),
            &["release-notes".to_string(), "missing-skill".to_string()],
        );
        assert!(
            prompt.contains("<skill_content name=\"release-notes\">"),
            "{prompt}"
        );
        assert!(prompt.contains("not installed; work without them: missing-skill."));
        assert!(
            prompt.contains("- release-notes: Write release notes."),
            "{prompt}"
        );
        let loaded = tools
            .execute(
                "use_skill",
                serde_json::json!({"name": "release-notes", "arguments": "Be brief."})
                    .as_object()
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            loaded.contains("Group changes by area. Be brief."),
            "{loaded}"
        );
        let unknown = tools
            .execute(
                "use_skill",
                serde_json::json!({"name": "nope"}).as_object().unwrap(),
            )
            .await
            .unwrap_err();
        assert!(unknown.to_string().contains("release-notes"), "{unknown}");
    }

    #[test]
    fn an_allowlist_without_skill_gets_no_index_but_keeps_preloaded_skills() {
        let workspace = workspace_with_skill();
        let tools = ToolExecutor::new_for_subagent(workspace.path())
            .with_allowed_tools(["show_file".to_string()].into());
        let (tools, prompt) = attach_skills(tools, &["release-notes".to_string()]);
        assert!(prompt.contains("<skill_content name=\"release-notes\">"));
        assert!(!prompt.contains("<skills>"), "{prompt}");
        assert!(
            !tools
                .definitions()
                .iter()
                .any(|def| def.function.name == "use_skill")
        );
    }
}

#[cfg(test)]
mod main_agent_tests {
    use super::*;
    use crate::agents::custom::{AgentRegistry, AgentSource};

    #[test]
    fn the_main_conversation_runs_as_an_agent() {
        let workspace = tempfile::tempdir().unwrap();
        let agents = workspace.path().join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(
            agents.join("reviewer.md"),
            "---\nname: reviewer\ndescription: Reviews.\ntools: Read, Grep\nskills: absent\n---\nReview carefully.\n",
        )
        .unwrap();
        let tools = ToolExecutor::new(workspace.path()).with_custom_agents(
            AgentRegistry::from_directories(&[(agents, AgentSource::Project)]),
        );

        let (tools, prompt) = as_main_agent(tools, "reviewer").unwrap();
        assert!(prompt.contains("## Agent: reviewer"), "{prompt}");
        assert!(prompt.contains("Review carefully."));
        assert!(prompt.contains("The skill 'absent' is not installed."));
        let mut offered: Vec<String> = tools
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect();
        offered.sort();
        assert_eq!(offered, ["expand_context", "search_text", "show_file"]);

        let Err(unknown) = as_main_agent(tools, "reviewr") else {
            panic!("an unknown agent is refused");
        };
        let unknown = unknown.to_string();
        assert!(unknown.contains("unknown agent 'reviewr'"), "{unknown}");
        assert!(unknown.contains("`/agents` lists"), "{unknown}");
    }
}

#[cfg(test)]
mod mention_tests {
    use super::*;

    #[test]
    fn mentions_start_a_word_and_are_named_once() {
        assert_eq!(
            agent_mentions("@agent-reviewer look, then @agent-api-designer. And @agent-reviewer."),
            ["reviewer", "api-designer"]
        );
        assert!(agent_mentions("mail me@agent-x.com").is_empty());
        assert!(agent_mentions("just @agent- nothing").is_empty());
        assert_eq!(agent_mentions("(@agent-write_tests)"), ["write_tests"]);
        assert_eq!(
            agent_mentions("@agent-security:scan-researcher: look"),
            ["security:scan-researcher"]
        );
    }

    /// SSH reaches other machines: an agent with no `tools` field, which
    /// allows everything else, has to ask for it by name.
    #[tokio::test]
    async fn a_custom_agent_gets_the_ssh_tools_only_by_naming_them() {
        let workspace = tempfile::tempdir().unwrap();
        let directory = workspace.path().join(".orangu/agents");
        std::fs::create_dir_all(&directory).unwrap();
        for (file, tools) in [
            ("anything.md", ""),
            ("named.md", "tools: Read, ssh_login, ssh_copy\n"),
            ("planner.md", "tools: ssh_login\npermissionMode: plan\n"),
        ] {
            let name = file.trim_end_matches(".md");
            std::fs::write(
                directory.join(file),
                format!("---\nname: {name}\ndescription: x\n{tools}---\nGo.\n"),
            )
            .unwrap();
        }
        let parent = ToolExecutor::new(workspace.path())
            .with_workspace_agents()
            .with_ssh(crate::ssh::SshSessions::for_mode(
                crate::config::SshMode::Prompt,
                None,
            ));
        let agents = parent.custom_agents();
        let offered = async |name: &str| -> Vec<String> {
            let agent = agents.find(name).unwrap();
            let (tools, _) = custom_tools(agent, &parent, None).await;
            tools
                .definitions()
                .into_iter()
                .map(|def| def.function.name)
                .filter(|name| name.contains("ssh"))
                .collect()
        };
        assert!(offered("anything").await.is_empty());
        assert_eq!(offered("named").await, ["ssh_login", "ssh_copy"]);
        assert!(offered("planner").await.is_empty());
    }

    /// The tools of `use_ssh`, and of a task rebuilt after a restart.
    #[test]
    fn the_ssh_subagent_has_no_shell_and_no_skills() {
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path()).with_ssh(
            crate::ssh::SshSessions::for_mode(crate::config::SshMode::Prompt, None),
        );
        let (tools, additions) = spec_tools(&SSH, &parent);
        let mut names: Vec<String> = tools
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["list_directory", "ssh_copy", "ssh_login", "ssh_logout"]
        );
        assert!(additions.is_empty(), "{additions}");
    }

    #[tokio::test]
    async fn use_ssh_says_so_when_ssh_is_off() {
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path());
        let mut arguments = serde_json::Map::new();
        arguments.insert("task".into(), json!("copy a"));
        let error = parent.execute("use_ssh", &arguments).await.unwrap_err();
        assert!(error.to_string().contains("[orangu].ssh"), "{error}");
    }

    #[test]
    fn a_mention_asks_the_model_to_delegate() {
        let workspace = tempfile::tempdir().unwrap();
        let agents = workspace.path().join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(
            agents.join("r.md"),
            "---\nname: reviewer\ndescription: Reviews.\n---\nReview.\n",
        )
        .unwrap();
        let tools = ToolExecutor::new(workspace.path()).with_custom_agents(
            crate::agents::custom::AgentRegistry::from_directories(&[(
                agents,
                crate::agents::custom::AgentSource::Project,
            )]),
        );
        let prompt = with_agent_mentions(
            "@agent-reviewer and @agent-explore_repository: check src/ (@agent-nobody)",
            &tools,
        );
        assert!(
            prompt.ends_with(
                "(The user asked for this to be delegated: call run_agent with agent \"reviewer\"; \
                 call explore_repository, and give it the whole request as its task.)"
            ),
            "{prompt}"
        );
        assert_eq!(with_agent_mentions("no mention", &tools), "no mention");
        // A read-only executor starts no agents, so nothing is asked.
        let read_only = ToolExecutor::new_read_only(workspace.path());
        assert_eq!(
            with_agent_mentions("@agent-reviewer", &read_only),
            "@agent-reviewer"
        );
    }
}

#[cfg(test)]
mod nesting_tests {
    use super::*;
    use crate::agents::custom::{AgentRegistry, AgentSource};

    fn tool_call(name: &str, arguments: Value) -> String {
        let delta = json!({"tool_calls": [{
            "index": 0, "id": "call0", "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()}
        }]});
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{"delta": delta, "finish_reason": "tool_calls"}]})
        )
    }

    fn agents(dir: &std::path::Path) -> AgentRegistry {
        for (file, text) in [
            (
                "lead.md",
                "---\nname: lead\ndescription: Leads.\ntools: Read, Agent(worker)\n---\nLead.\n",
            ),
            (
                "worker.md",
                "---\nname: worker\ndescription: Works.\ntools: Read\n---\nWork.\n",
            ),
            (
                "other.md",
                "---\nname: other\ndescription: Other.\ntools: Read\n---\nOther.\n",
            ),
        ] {
            std::fs::write(dir.join(file), text).unwrap();
        }
        AgentRegistry::from_directories(&[(dir.to_path_buf(), AgentSource::Project)])
    }

    #[tokio::test]
    async fn an_agent_with_agent_in_its_tools_starts_the_agents_it_names() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![
                tool_call(
                    "run_agent",
                    json!({"agent": "worker", "task": "count the files"}),
                ),
                crate::test_support::text_stream("<final_answer>3 files</final_answer>"),
                crate::test_support::text_stream("<final_answer>the worker says 3</final_answer>"),
            ],
        );
        let conf = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            conf.path(),
            format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"),
        )
        .unwrap();
        let profile = crate::config::load_client_configuration(conf.path())
            .unwrap()
            .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path()).with_custom_agents(agents(dir.path()));
        parent.set_active_profile(&profile);

        let mut arguments = serde_json::Map::new();
        arguments.insert("agent".into(), json!("lead"));
        arguments.insert("task".into(), json!("ask the worker"));
        let report = parent.execute("run_agent", &arguments).await.unwrap();
        assert!(report.contains("the worker says 3"), "{report}");

        let requests = server.join().unwrap();
        // The lead is offered run_agent (for its one agent), no built-ins.
        assert!(requests[0].contains("\"run_agent\""), "{}", requests[0]);
        assert!(!requests[0].contains("\"write_tests\""));
        // The worker's tools do not list Agent: it starts nothing.
        assert!(!requests[1].contains("\"run_agent\""), "{}", requests[1]);
        // Both runs are tasks of the session.
        let listed: Vec<String> = parent.tasks().list().into_iter().map(|t| t.agent).collect();
        assert_eq!(listed, ["lead", "worker"]);
    }

    #[tokio::test]
    async fn a_nested_agent_starts_only_the_agents_it_names_and_only_so_deep() {
        let workspace = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let main = ToolExecutor::new(workspace.path()).with_custom_agents(agents(dir.path()));
        let lead = ToolExecutor::new_for_subagent(workspace.path())
            .nested_under(&main, Some(["worker".to_string()].into()));
        assert_eq!(lead.depth(), 1);
        let mut arguments = serde_json::Map::new();
        arguments.insert("agent".into(), json!("other"));
        arguments.insert("task".into(), json!("anything"));
        let refused = lead.execute("run_agent", &arguments).await.unwrap_err();
        assert!(
            refused.to_string().contains("may not start 'other'"),
            "{refused}"
        );
        arguments.insert("agent".into(), json!("list"));
        arguments.insert("task".into(), json!(""));
        let listed = lead.execute("run_agent", &arguments).await.unwrap();
        assert!(
            listed.contains("worker") && !listed.contains("other"),
            "{listed}"
        );

        let second = ToolExecutor::new_for_subagent(workspace.path()).nested_under(&lead, None);
        let third = ToolExecutor::new_for_subagent(workspace.path()).nested_under(&second, None);
        assert_eq!(third.depth(), 3);
        assert_eq!(third.max_depth(), 3);
        assert!(
            third.depth() + 1 > third.max_depth(),
            "a fourth level is never set up"
        );
    }
}

#[cfg(test)]
mod fork_tests {
    use super::*;
    use crate::llm::ChatMessage;

    #[tokio::test]
    async fn a_fork_starts_from_the_whole_conversation_and_reports_later() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![crate::test_support::text_stream(
                "<final_answer>the other branch has the same bug</final_answer>",
            )],
        );
        let conf = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            conf.path(),
            format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"),
        )
        .unwrap();
        let profile = crate::config::load_client_configuration(conf.path())
            .unwrap()
            .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let messages = vec![
            ChatMessage::system("You are the main agent."),
            ChatMessage::user("The parser drops trailing commas."),
            ChatMessage::assistant("Fixed on this branch."),
        ];

        // Forks need a conversation, a task and an interactive session.
        let plain = ToolExecutor::new(workspace.path());
        plain.set_active_profile(&profile);
        assert!(fork(&plain, &messages, "look").is_err());
        let parent = plain.with_background_tasks();
        assert!(fork(&parent, &messages, " ").is_err());
        assert!(fork(&parent, &[], "look").is_err());

        fork(&parent, &messages, "check the other branch").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let notification = loop {
            if let Some(notification) = parent.tasks().take_notifications().pop() {
                break notification;
            }
            assert!(std::time::Instant::now() < deadline, "no report");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        };
        assert_eq!(notification.agent, "fork");
        assert!(
            notification.report.contains("same bug"),
            "{}",
            notification.report
        );

        let request = server.join().unwrap().remove(0);
        assert!(request.contains("You are the main agent."), "{request}");
        assert!(request.contains("The parser drops trailing commas."));
        assert!(request.contains("check the other branch"));
        // Its tools are a writer's, with no subagents.
        assert!(request.contains("\"modify_file\"") && !request.contains("\"write_tests\""));
    }
}

#[cfg(test)]
mod worktree_tests {
    use super::*;
    use crate::agents::custom::{AgentRegistry, AgentSource};

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}");
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    #[tokio::test]
    async fn an_isolated_agent_changes_its_worktree_not_the_workspace() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let call = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call0", "type": "function",
                "function": {"name": "create_file", "arguments": json!({"path": "notes.txt", "content": "isolated\n"}).to_string()}
            }]}, "finish_reason": "tool_calls"}]})
        );
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![
                call,
                crate::test_support::text_stream("<final_answer>wrote notes.txt</final_answer>"),
                crate::test_support::text_stream("<final_answer>nothing to do</final_answer>"),
            ],
        );
        let conf = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            conf.path(),
            format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"),
        )
        .unwrap();
        let profile = crate::config::load_client_configuration(conf.path())
            .unwrap()
            .llms["main"]
            .clone();

        let workspace = tempfile::tempdir().unwrap();
        git(workspace.path(), &["init", "--quiet"]);
        std::fs::write(workspace.path().join("README"), "r\n").unwrap();
        git(workspace.path(), &["add", "."]);
        git(
            workspace.path(),
            &[
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "commit",
                "--quiet",
                "-m",
                "init",
            ],
        );
        let agents = tempfile::tempdir().unwrap();
        std::fs::write(
            agents.path().join("w.md"),
            "---\nname: writer\ndescription: Writes.\nisolation: worktree\n---\nWrite.\n",
        )
        .unwrap();
        let parent = ToolExecutor::new(workspace.path()).with_custom_agents(
            AgentRegistry::from_directories(&[(agents.path().to_path_buf(), AgentSource::Project)]),
        );
        parent.set_active_profile(&profile);
        let mut arguments = serde_json::Map::new();
        arguments.insert("agent".into(), json!("writer"));
        arguments.insert("task".into(), json!("write notes"));

        let report = parent.execute("run_agent", &arguments).await.unwrap();
        assert!(report.contains("wrote notes.txt"), "{report}");
        assert!(report.contains("isolated worktree"), "{report}");
        assert!(!workspace.path().join("notes.txt").exists());
        assert!(git(workspace.path(), &["status", "--porcelain"]).is_empty());
        let worktrees = git(workspace.path(), &["worktree", "list", "--porcelain"]);
        let kept = worktrees
            .lines()
            .filter_map(|line| line.strip_prefix("worktree "))
            .find(|path| path.contains("orangu-worktrees"))
            .expect("the changed worktree stays");
        assert_eq!(
            std::fs::read_to_string(Path::new(kept).join("notes.txt")).unwrap(),
            "isolated\n"
        );

        // A run that changes nothing leaves nothing behind, and says so.
        let report = parent.execute("run_agent", &arguments).await.unwrap();
        assert!(report.contains("changed no files"), "{report}");
        let worktrees = git(workspace.path(), &["worktree", "list"]);
        assert_eq!(
            worktrees.matches("orangu-worktrees").count(),
            1,
            "{worktrees}"
        );
        let _ = server.join();
    }
}

#[cfg(test)]
mod detach_tests {
    use super::*;

    #[tokio::test]
    async fn ctrl_b_moves_a_foreground_subagent_to_the_background() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let (release, gate) = std::sync::mpsc::channel();
        let server = crate::test_support::serve_gated_chat_responses(
            listener,
            vec![crate::test_support::text_stream(
                "<final_answer>tests written</final_answer>",
            )],
            Some(gate),
        );
        let conf = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            conf.path(),
            format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"),
        )
        .unwrap();
        let profile = crate::config::load_client_configuration(conf.path())
            .unwrap()
            .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let parent = ToolExecutor::new(workspace.path()).with_background_tasks();
        parent.set_active_profile(&profile);
        let tasks = parent.tasks().clone();

        let waiting = parent.clone();
        let turn = tokio::spawn(async move {
            let mut arguments = serde_json::Map::new();
            arguments.insert("task".into(), json!("cover the parser"));
            waiting.execute("write_tests", &arguments).await
        });
        // Once it runs (in the foreground), the user presses Ctrl+B.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while tasks.detach_foreground() == 0 {
            assert!(std::time::Instant::now() < deadline, "never ran");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let answer = turn.await.unwrap().unwrap();
        assert!(answer.contains("moved subagent task t1"), "{answer}");
        assert!(tasks.get("t1").unwrap().background);
        assert_eq!(tasks.get("t1").unwrap().status, TaskStatus::Running);

        // It goes on, and its report arrives as a notification.
        release.send(()).unwrap();
        let notification = loop {
            if let Some(notification) = tasks.take_notifications().pop() {
                break notification;
            }
            assert!(std::time::Instant::now() < deadline, "no report");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert!(notification.report.contains("tests written"));
        let _ = server.join();
    }

    #[test]
    fn the_background_default_applies_when_a_call_does_not_say() {
        let workspace = tempfile::tempdir().unwrap();
        let mut asks = serde_json::Map::new();
        let plain = ToolExecutor::new(workspace.path()).with_background_tasks();
        let by_default = plain.clone().with_background_default(true);
        assert!(!wants_background(&plain, &asks));
        assert!(wants_background(&by_default, &asks));
        asks.insert("background".into(), json!(false));
        assert!(!wants_background(&by_default, &asks));
        // The schema tells the model.
        let schema = serde_json::to_string(&by_default.definitions()).unwrap();
        assert!(
            schema.contains(r#""background":{"type":"boolean","default":true}"#),
            "{schema}"
        );
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    #[tokio::test]
    async fn a_task_survives_a_restart_and_can_be_continued() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = crate::test_support::serve_chat_responses(
            listener,
            vec![
                crate::test_support::text_stream("<final_answer>3 tests added</final_answer>"),
                crate::test_support::text_stream("<final_answer>2 more added</final_answer>"),
            ],
        );
        let conf = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            conf.path(),
            format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n"),
        )
        .unwrap();
        let profile = crate::config::load_client_configuration(conf.path())
            .unwrap()
            .llms["main"]
            .clone();
        let workspace = tempfile::tempdir().unwrap();
        let session = tempfile::tempdir().unwrap();
        let file = session.path().join("subagent-tasks.json");

        {
            let parent = ToolExecutor::new(workspace.path()).with_background_tasks();
            parent.set_active_profile(&profile);
            parent.tasks().persist_to(file.clone());
            let mut arguments = serde_json::Map::new();
            arguments.insert("task".into(), json!("cover the parser"));
            parent.execute("write_tests", &arguments).await.unwrap();
        }
        assert!(file.exists());

        // A new run of the client: a new executor, the same session.
        let parent = ToolExecutor::new(workspace.path()).with_background_tasks();
        parent.set_active_profile(&profile);
        parent.tasks().persist_to(file.clone());
        let restored = parent.tasks().get("t1").expect("the task is back");
        assert_eq!(restored.status, TaskStatus::Completed);
        assert!(restored.resumable);
        assert!(restored.result.unwrap().contains("3 tests added"));

        let mut arguments = serde_json::Map::new();
        arguments.insert("task".into(), json!("t1"));
        arguments.insert("message".into(), json!("now the lexer"));
        let report = parent.execute("continue_agent", &arguments).await.unwrap();
        assert!(report.contains("2 more added"), "{report}");
        let requests = server.join().unwrap();
        // The continuation carried the earlier exchange.
        assert!(requests[1].contains("cover the parser"), "{}", requests[1]);
        assert!(requests[1].contains("3 tests added"));
        assert!(requests[1].contains("now the lexer"));
        // New tasks carry on the numbering.
        assert_eq!(parent.tasks().start("x", "y", "m", false), "t2");
    }
}

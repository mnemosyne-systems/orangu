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

//! `orangu -p "<prompt>"`: send one prompt to the configured server, stream the
//! answer to stdout and exit — no terminal UI, no session on disk. The prompt is
//! assembled exactly as the interactive client assembles it (system prompt,
//! skills index, `AGENTS.md`, tool definitions), so what it measures is what the
//! prompt area does.
//!
//! What the prompt area handles itself, `-p` handles itself too: the input is
//! first offered to the same parser and dispatcher the interactive client uses,
//! so a slash command (`-p "/export pr"`), its natural-language form
//! (`-p "show git status"`), and a skill invocation (`-p "/code-review auth"`,
//! which expands to the skill's prompt and *is* sent to the model) all behave
//! the way they do at the prompt. Only text that resolves to no local command
//! reaches the server. Commands that exist purely to change the running
//! session, and those that need the terminal interface, are refused by name
//! rather than reported as a silent success — see [`session_only_reason`] and
//! the interactive arm of [`run_outcome`].

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};

use crate::commands::{
    CommandContext, CommandOutcome, CommandState, ExportTarget, LocalCommand,
    current_terminal_width, parse_local_command, system_prompt,
};
use crate::dispatch::{handle_command, run_duplicates_scan};
use crate::export;
use crate::git::{Forge, ReviewReports, fetch_pull_request_details};
use crate::models::{detect_embeddings_server, is_active_connection_a_coordinator};
use crate::stats::UsageStats;
use orangu::{
    config::ClientAppConfiguration,
    llm::{StreamMetrics, normalized_openai_endpoint},
    session::ChatSession,
    skills::SkillRegistry,
    tools::ToolExecutor,
};

/// Everything the run loop already resolved before it knew this was a one-shot:
/// the loaded config, where it came from, and the workspace the prompt runs
/// against.
pub(crate) struct OneshotContext {
    pub(crate) config: ClientAppConfiguration,
    pub(crate) config_path: PathBuf,
    pub(crate) workspace: PathBuf,
    /// `-q`: print nothing at all, and let the exit code carry the result.
    pub(crate) quiet: bool,
}

/// What the `-p` text turned out to be.
#[derive(Debug)]
enum Resolution {
    /// orangu handled it locally; nothing goes to the model.
    Handled,
    /// Text for the model — the prompt as typed, or a skill's expansion of it.
    Prompt(String),
}

/// Where a one-shot's own output goes — the terminal, or under `-q` nowhere at
/// all. Failures never travel through here: they go up as `Err` for `main` to
/// print on stderr, so `-q` silences a success without ever hiding a failure.
#[derive(Clone, Copy)]
struct Console {
    quiet: bool,
}

impl Console {
    /// A finished block of command output, with a single trailing newline —
    /// the output window's line handling has no equivalent here.
    fn block(self, text: &str) {
        let text = text.trim_end_matches('\n');
        if self.quiet || text.is_empty() {
            return;
        }
        println!("{text}");
    }

    /// One line of a streaming command, printed as it is produced.
    fn line(self, text: &str) {
        if self.quiet {
            return;
        }
        println!("{}", text.trim_end_matches('\n'));
        let _ = std::io::stdout().flush();
    }

    /// A piece of the model's answer, as it streams in.
    fn delta(self, text: &str) {
        if self.quiet {
            return;
        }
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    /// A diagnostic — what the run is talking to, which tools it called, what
    /// it cost. On stderr, so it stays out of piped output even without `-q`.
    fn note(self, text: &str) {
        if self.quiet {
            return;
        }
        eprintln!("{text}");
    }
}

pub(crate) async fn run_prompt(prompt: &str, context: OneshotContext) -> Result<()> {
    let OneshotContext {
        config,
        config_path,
        workspace,
        quiet,
    } = context;
    let console = Console { quiet };

    let server = config.default_server.clone();
    let profile = config
        .llms
        .get(&server)
        .ok_or_else(|| anyhow!("missing configured server {server}"))?;

    let (mcp, mcp_warnings) =
        orangu::mcp::McpManager::connect_all(&config.mcp_servers, &workspace).await?;
    for warning in mcp_warnings {
        console.note(&format!("Warning: {warning}"));
    }
    let tools = orangu::tools::ToolExecutor::with_config(
        &workspace,
        config.compression,
        config.auto_downsample_lines,
        config.diff_file_cap,
        None,
    )
    .with_mcp(mcp.clone());
    let skills = orangu::skills::SkillRegistry::discover(&workspace);

    // Anything orangu answers on its own is answered here, before a single byte
    // goes to the server.
    let prompt = match run_local_command(
        prompt,
        LocalRun {
            config: &config,
            config_path: &config_path,
            workspace: &workspace,
            server: &server,
            tools: &tools,
            skills: &skills,
            console,
        },
    )
    .await?
    {
        Resolution::Handled => return Ok(()),
        Resolution::Prompt(text) => text,
    };
    let prompt = prompt.as_str();

    // The same system prompt the interactive client builds for a tab.
    let mut system = system_prompt(profile, None).to_string();
    let index = skills.system_prompt_index();
    if !index.is_empty() {
        system.push_str("\n\n");
        system.push_str(&index);
    }
    system.push_str(&orangu::config::load_agents_instructions(&workspace));
    if !mcp.instructions().is_empty() {
        system.push_str("\n\n");
        system.push_str(&mcp.instructions());
    }

    let definitions = tools.definitions();
    let tools_bytes = serde_json::to_string(&definitions).map_or(0, |json| json.len());
    console.note(&format!(
        "server {server} ({}) model {} — workspace {}",
        profile.endpoint,
        profile.model,
        workspace.display()
    ));
    console.note(&format!(
        "sending {} chars of system prompt and {} tool definitions ({tools_bytes} chars)",
        system.chars().count(),
        definitions.len(),
    ));

    let mut session = ChatSession::new(&system);
    let metrics = Arc::new(Mutex::new(StreamMetrics::default()));
    let collected = Arc::clone(&metrics);
    let started = Instant::now();
    let mut first_delta: Option<std::time::Duration> = None;

    let result = session
        .prompt(
            prompt,
            profile,
            &tools,
            |delta| {
                if first_delta.is_none() {
                    first_delta = Some(started.elapsed());
                }
                console.delta(delta);
            },
            move |update| {
                if let Ok(mut state) = collected.lock() {
                    state.merge(update);
                }
            },
            |_running| {},
            |tool_call| {
                console.note(&format!("[tool] {}", tool_call.function.name));
            },
            |_| false,
        )
        .await;

    console.delta("\n");
    let elapsed = started.elapsed();
    let metrics = metrics.lock().ok().map(|state| state.clone());
    report_timings(console, elapsed, first_delta, metrics.as_ref());

    result.map(|_| ())
}

/// What the local-command path needs that the run loop would normally hold in
/// its own locals.
struct LocalRun<'a> {
    config: &'a ClientAppConfiguration,
    config_path: &'a Path,
    workspace: &'a Path,
    /// The configured server section the one-shot runs against.
    server: &'a str,
    tools: &'a ToolExecutor,
    skills: &'a SkillRegistry,
    console: Console,
}

/// Offer the `-p` text to the interactive dispatcher and, when it is a command,
/// run it here. Returns the text to send to the model when it is not.
async fn run_local_command(input: &str, run: LocalRun<'_>) -> Result<Resolution> {
    let profile = run
        .config
        .llms
        .get(run.server)
        .ok_or_else(|| anyhow!("missing configured server {}", run.server))?;

    if let Some(command) = parse_local_command(input)
        && let Some(reason) = session_only_reason(&command)
    {
        return Err(anyhow!("'{}' {reason}", input.trim()));
    }

    // `/search` is the one command that needs a resolved embeddings connection,
    // so the probes it costs stay off the path of every other one.
    let endpoint = normalized_openai_endpoint(&profile.endpoint);
    let (is_coordinator, embeddings_server) =
        if matches!(parse_local_command(input), Some(LocalCommand::Search(_))) {
            let http_client = reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()?;
            let is_coordinator = is_active_connection_a_coordinator(
                &http_client,
                run.config,
                run.server,
                Some(&endpoint),
            )
            .await;
            let embeddings = detect_embeddings_server(run.config, run.server, is_coordinator).await;
            (is_coordinator, embeddings)
        } else {
            (false, String::new())
        };

    let mut active_model = run.server.to_string();
    let mut active_model_id = profile.model.clone();
    let mut current_endpoint = Some(endpoint);
    // A command's LLM context (the `/diff` output a follow-up question would
    // build on, say) has nowhere to go in a one-shot: there is no next turn.
    let mut session = ChatSession::new("");
    let mut detect_model = false;
    let forge = Forge::from_platform(&run.config.platform);
    let session_dir = absent_session_dir();

    let outcome = handle_command(
        input,
        CommandState {
            active_model: &mut active_model,
            active_model_id: &mut active_model_id,
            current_endpoint: &mut current_endpoint,
            session: &mut session,
            detect_model: &mut detect_model,
        },
        CommandContext {
            startup_model: run.server,
            startup_endpoint: &profile.endpoint,
            llms: &run.config.llms,
            tools: run.tools,
            workspace: run.workspace,
            session_dir: &session_dir,
            embeddings_server: &embeddings_server,
            is_coordinator,
            usage_stats: &UsageStats::new(),
            // Empty means "not detected": `/model` reports the configured id
            // rather than a server's list, which a one-shot never fetches.
            available_models: &[],
            virtual_width: current_terminal_width(),
            auto_rebase: run.config.auto_rebase,
            auto_squash: run.config.auto_squash,
            compile_workers: run.config.compile_workers,
            compression: run.config.compression,
            terminal: &run.config.terminal,
            forge,
            review_reports: ReviewReports::default(),
            skills: run.skills,
            semantic_budget_tokens: run.config.semantic_budget_tokens,
            config_path: run.config_path,
        },
    )?;

    run_outcome(
        outcome,
        input,
        run.workspace,
        &active_model_id,
        forge,
        run.console,
    )
    .await
}

/// `-p` keeps no session on disk, so there is no directory for the per-session
/// settings a few commands persist. Those commands are refused before dispatch
/// (see [`session_only_reason`]); this path only satisfies [`CommandContext`],
/// and deliberately names a directory that is never created — a write to it
/// fails, and every such write is already ignored.
fn absent_session_dir() -> PathBuf {
    std::env::temp_dir().join("orangu-oneshot-no-session")
}

/// Commands whose whole effect is on the running session — the connection, the
/// active server or model, the theme, the verbosity. A one-shot exits the
/// moment the command is done, so running them would change nothing anyone
/// could see; say that instead of reporting a silent success.
fn session_only_reason(command: &LocalCommand<'_>) -> Option<&'static str> {
    match command {
        LocalCommand::Disconnect | LocalCommand::Reload => {
            Some("only changes the running session, which -p does not keep")
        }
        LocalCommand::SetModelId(_) | LocalCommand::SetServer(_) => Some(
            "only changes the running session, which -p does not keep; \
             select the server in orangu.conf, or pass --config",
        ),
        // A bare `/theme` lists the themes instead of setting one, so it is
        // left to run.
        LocalCommand::SetTheme(name) if !name.trim().is_empty() => {
            Some("only changes the running session's theme; -p renders no interface")
        }
        LocalCommand::SetVerbosity(_) => {
            Some("only changes the running session's system prompt, which -p does not keep")
        }
        _ => None,
    }
}

/// Carry out what the dispatcher decided, printing to stdout instead of to the
/// output window.
async fn run_outcome(
    outcome: CommandOutcome,
    input: &str,
    workspace: &Path,
    model_id: &str,
    forge: Forge,
    console: Console,
) -> Result<Resolution> {
    match outcome {
        CommandOutcome::Unhandled => Ok(Resolution::Prompt(input.to_string())),
        CommandOutcome::SkillInvoked { prompt, .. } => Ok(Resolution::Prompt(prompt)),
        CommandOutcome::Quiet => Ok(Resolution::Handled),
        CommandOutcome::Output(text)
        | CommandOutcome::MarkdownOutput(text)
        | CommandOutcome::OutputWithLlmContext { display: text, .. }
        | CommandOutcome::WideOutputWithLlmContext { display: text, .. } => {
            console.block(&text);
            Ok(Resolution::Handled)
        }
        // A failed command exits non-zero, so `orangu -p` composes in a script
        // the way any other command-line tool does.
        CommandOutcome::OutputError(message) => Err(anyhow!("{}", message.trim_end())),
        CommandOutcome::Blocking(work) => {
            console.block(&work()?);
            Ok(Resolution::Handled)
        }
        CommandOutcome::Streaming(work, _control) => {
            // Off-thread so a long `/shell` or `/build` prints as it goes
            // rather than all at once at the end. Cancellation is the terminal's
            // job here (Ctrl-C ends the process), so the control is unused.
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<String>();
            let handle = tokio::task::spawn_blocking(move || work(sender));
            while let Some(line) = receiver.recv().await {
                console.line(&line);
            }
            handle.await??;
            Ok(Resolution::Handled)
        }
        CommandOutcome::Duplicates(threshold) => {
            console.block(&run_duplicates_scan(workspace, threshold)?.to_console());
            Ok(Resolution::Handled)
        }
        CommandOutcome::Export(target) => {
            let path = export_path(target, workspace, model_id, forge)?;
            console.block(&format!("Exported to {}", path.display()));
            Ok(Resolution::Handled)
        }
        CommandOutcome::Review(_)
        | CommandOutcome::AutoReview(_)
        | CommandOutcome::Manual
        | CommandOutcome::Cleared
        | CommandOutcome::Quit
        | CommandOutcome::Restart
        | CommandOutcome::SwitchSession(_)
        | CommandOutcome::SwitchWorkspace(_)
        | CommandOutcome::SwitchWorkspaceTab(_)
        | CommandOutcome::ChangeWorkspace(_)
        | CommandOutcome::OpenWorkspaceTab(_)
        | CommandOutcome::CloseWorkspaceTab
        | CommandOutcome::PendingList
        | CommandOutcome::PendingDelete(_) => Err(anyhow!(
            "'{}' needs the terminal interface; run orangu without -p",
            input.trim()
        )),
    }
}

/// Write the requested export and return the file it landed in. The targets a
/// one-shot cannot serve are the ones that export something the session
/// accumulated: the console window, and a review report.
fn export_path(
    target: ExportTarget,
    workspace: &Path,
    model_id: &str,
    forge: Forge,
) -> Result<PathBuf> {
    match target {
        ExportTarget::Pr => {
            let prs = fetch_pull_request_details(workspace, forge)?;
            export::export_pr(workspace, &prs, model_id)
        }
        ExportTarget::Statistics(total) => export::export_statistics(workspace, model_id, total),
        ExportTarget::Duplicates => {
            let report = run_duplicates_scan(workspace, orangu::duplicates::DEFAULT_THRESHOLD)?;
            export::export_duplicates(workspace, &report, model_id)
        }
        ExportTarget::Console => Err(anyhow!(
            "There is no console output to export with -p; /export console \
             writes the interactive output window"
        )),
        ExportTarget::Review | ExportTarget::AutoReview => Err(anyhow!(
            "There is no review to export with -p; /review and /auto_review \
             need the terminal interface"
        )),
    }
}

/// The line that makes a slow answer diagnosable: how long the server spent on
/// the prompt (and how much of it came from its KV cache) against how long it
/// spent generating.
fn report_timings(
    console: Console,
    elapsed: std::time::Duration,
    first_delta: Option<std::time::Duration>,
    metrics: Option<&StreamMetrics>,
) {
    let mut parts = vec![format!("total {:.1}s", elapsed.as_secs_f64())];
    if let Some(first) = first_delta {
        parts.push(format!("first token {:.1}s", first.as_secs_f64()));
    }
    if let Some(progress) = metrics.and_then(|m| m.prompt_progress.as_ref()) {
        parts.push(format!(
            "prompt {} tokens ({} cached) in {:.1}s",
            progress.total,
            progress.cache,
            progress.time_ms as f64 / 1000.0
        ));
    }
    if let Some(rate) = metrics.and_then(|m| m.prompt_per_second) {
        parts.push(format!("prefill {rate:.1} t/s"));
    }
    if let Some(rate) = metrics.and_then(|m| m.predicted_per_second) {
        parts.push(format!("decode {rate:.1} t/s"));
    }
    console.note(&format!("[{}]", parts.join(" | ")));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal decision as `-p` makes it: parse the typed line first, so a
    /// change in what a line parses to shows up here.
    fn reason_for(input: &str) -> Option<&'static str> {
        session_only_reason(&parse_local_command(input).expect("a local command"))
    }

    #[test]
    fn session_only_commands_are_refused_by_name() {
        for input in [
            "/disconnect",
            "/reload",
            "/model gemma",
            "/server llama",
            "/theme dark",
            "/verbosity terse",
            "/verbosity",
        ] {
            assert!(reason_for(input).is_some(), "{input} should be refused");
        }
    }

    #[test]
    fn commands_that_do_something_outside_the_session_run() {
        // A bare `/theme` lists the available themes instead of setting one,
        // and `/model`/`/server` without an argument only report.
        for input in [
            "/theme",
            "/model",
            "/server",
            "/status",
            "/export pr",
            "/help",
            "/diff",
            "/usage",
        ] {
            assert!(reason_for(input).is_none(), "{input} should run");
        }
    }

    #[tokio::test]
    async fn text_the_dispatcher_does_not_claim_goes_to_the_model() {
        let resolution = run_outcome(
            CommandOutcome::Unhandled,
            "explain the prefill path",
            Path::new("."),
            "gemma",
            Forge::GitHub,
            Console { quiet: false },
        )
        .await
        .expect("unhandled input resolves");

        match resolution {
            Resolution::Prompt(prompt) => assert_eq!(prompt, "explain the prefill path"),
            Resolution::Handled => panic!("expected the prompt to reach the model"),
        }
    }

    #[tokio::test]
    async fn a_skill_invocation_sends_its_expansion() {
        let resolution = run_outcome(
            CommandOutcome::SkillInvoked {
                name: "code-review".to_string(),
                prompt: "Review focus: auth".to_string(),
            },
            "/code-review auth",
            Path::new("."),
            "gemma",
            Forge::GitHub,
            Console { quiet: false },
        )
        .await
        .expect("a skill resolves");

        match resolution {
            Resolution::Prompt(prompt) => assert_eq!(prompt, "Review focus: auth"),
            Resolution::Handled => panic!("expected the skill prompt to reach the model"),
        }
    }

    #[tokio::test]
    async fn an_interactive_outcome_is_refused_rather_than_dropped() {
        let error = run_outcome(
            CommandOutcome::Manual,
            "/manual",
            Path::new("."),
            "gemma",
            Forge::GitHub,
            Console { quiet: false },
        )
        .await
        .expect_err("the manual needs a terminal");

        assert!(
            error.to_string().contains("needs the terminal interface"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_failed_command_is_an_error_not_a_printed_line() {
        let error = run_outcome(
            CommandOutcome::OutputError("Unknown command '/bogus'.".to_string()),
            "/bogus",
            Path::new("."),
            "gemma",
            Forge::GitHub,
            Console { quiet: false },
        )
        .await
        .expect_err("a failed command exits non-zero");

        assert!(error.to_string().contains("/bogus"), "{error}");
    }

    #[test]
    fn exports_of_session_state_say_why_they_cannot_run() {
        for target in [
            ExportTarget::Console,
            ExportTarget::Review,
            ExportTarget::AutoReview,
        ] {
            let error = export_path(target, Path::new("."), "gemma", Forge::GitHub)
                .expect_err("nothing to export");
            assert!(error.to_string().contains("-p"), "{error}");
        }
    }
}

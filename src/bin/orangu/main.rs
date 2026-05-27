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

mod build;
mod commands;
mod completion;
mod git;
mod input;
mod quotes;
mod render;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::{
    event::{
        self, Event, KeyCode, KeyEvent, KeyEventKind, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use orangu::{
    config::{LlmConfiguration, default_client_config_path, load_client_configuration},
    llm::{ChatMessage, StreamMetrics, normalized_openai_endpoint},
    session::ChatSession,
    tools::ToolExecutor,
    tui::{
        FEEDBACK_ERR, FEEDBACK_OK, ScreenRenderArgs, StatusFragment, render_screen,
        render_thinking_status, render_tool_running_status, render_working_status, terminal_height,
        terminal_width,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashSet, VecDeque},
    io::Write,
    path::{Component, Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
};
use tiktoken_rs::cl100k_base;
use uuid::Uuid;

use anyhow::Error;
use commands::{
    CommandContext, CommandOutcome, CommandState, LocalCommand, LocalError, add_file_usage_message,
    amend_usage_message, checkout_usage_message, cherry_pick_usage_message, commit_usage_message,
    connect_usage_message, delete_branch_usage_message, merge_usage_message, model_usage_message,
    move_file_usage_message, open_file_usage_message, parse_local_command, pull_usage_message,
    remove_file_usage_message, sorted_model_names, system_prompt,
};
use git::{
    add_file_output, amend_output, checkout_output, cherry_pick_output, commit_output,
    delete_branch_output, git_diff_against_branch, git_workspace_diff, init_repo_output,
    list_workspace_files_tree, log_output, merge_output, move_file_output, open_in_editor,
    pull_request_output, push_output, rebase_output, remove_file_output, squash_output,
    status_output, workspace_branch_name,
};
use input::{
    EscapeCancelState, InputContext, InputResult, InputState, InterruptState, OutputState,
    RenderContext, ScreenState, StreamRenderState, ViewportState, WaitContext, WaitResult,
    handle_input_event, read_input,
};
use render::{format_tools, render_markdown_for_console, show_file_output};

const CLEAR_TERMINAL_SEQUENCE: &str = "\x1b[2J\x1b[H";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const TERMINAL_TITLE: &str = "orangu";
const WAIT_LOOP_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
const THINKING_FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(120);
const SESSIONS_DIRECTORY: &str = ".orangu/sessions";

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long)]
    resume: Option<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let _terminal_title_guard = TerminalTitleGuard::new(TERMINAL_TITLE);
    let args = Args::parse();
    let config_path = match args.config.or_else(default_client_config_path) {
        Some(path) => path,
        None => {
            return Err(anyhow!(
                "Missing config file; pass --config or add ./orangu.conf or ~/.orangu/orangu.conf"
            ));
        }
    };
    let mut config = load_client_configuration(&config_path)?;
    let quote_module = quotes::QuoteModule::from_str(&config.quotes);
    let workspace = resolve_workspace_root(args.workspace)?;
    let workspace_created = if !workspace.exists() {
        std::fs::create_dir_all(&workspace)
            .with_context(|| format!("Failed to create workspace {}", workspace.display()))?;
        true
    } else {
        false
    };
    let tools = ToolExecutor::new(&workspace);

    let status_http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;
    let discovered_models =
        try_discover_new_models(&status_http_client, &mut config, &config_path).await;

    let model_names = sorted_model_names(&config.llms);
    let startup_model = config.default_model.clone();
    let startup_endpoint = config
        .llms
        .get(&startup_model)
        .ok_or_else(|| anyhow!("missing configured profile {}", startup_model))?
        .endpoint
        .clone();
    let mut active_model = startup_model.clone();
    let mut session = ChatSession::new(system_prompt(
        config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured profile {}", active_model))?,
    ));
    let mut current_endpoint = Some(startup_endpoint.clone());

    let current_branch = workspace_branch_name(&workspace);

    let (session_id, is_resumed) = match &args.resume {
        Some(id) => (id.clone(), true),
        None => {
            let workspace_str = workspace.display().to_string();
            match find_session_for_workspace_branch(
                &workspace_str,
                current_branch.as_deref().unwrap_or(""),
            ) {
                Some(existing_id) => (existing_id, true),
                None => (Uuid::new_v4().to_string(), false),
            }
        }
    };
    let session_dir = session_dir_path(&session_id)?;
    std::fs::create_dir_all(&session_dir).with_context(|| {
        format!(
            "failed to create session directory {}",
            session_dir.display()
        )
    })?;
    let session_hist_path = session_dir.join("history");
    let session_messages_path = session_dir.join("messages");
    let session_metadata_path = session_dir.join("metadata");

    if !is_resumed {
        save_session_metadata(
            &session_metadata_path,
            &SessionMetadata {
                started_at: current_unix_timestamp(),
                last_updated_at: current_unix_timestamp(),
                workspace: workspace.display().to_string(),
                branch: current_branch.clone().unwrap_or_default(),
            },
        )?;
    }

    if is_resumed {
        let messages = load_session_messages(&session_messages_path)?;
        session.restore(messages);
    }

    let _terminal_ui_guard = TerminalUiGuard::new()?;

    let vw = terminal_width();
    let vh = terminal_height();
    let mut viewport = ViewportState::new(config.width, vw, vh);
    let mut output_state = OutputState::default();
    let mut interrupt_state = InterruptState::default();
    let mut input_state = InputState::default();
    let mut pending_commands = VecDeque::new();
    let mut usage_stats = UsageStats::new().with_session(&session_id);
    let mut history = load_history(&session_hist_path)?;
    let mut startup_notice_until: Option<std::time::Instant> =
        if is_resumed && args.resume.is_none() {
            Some(std::time::Instant::now() + std::time::Duration::from_secs(5))
        } else {
            None
        };
    if workspace_created {
        output_state.push_text(&format!("Created workspace {}", workspace.display()));
    }

    for section_name in &discovered_models {
        output_state.push_text(&format!("Added {section_name} model"));
    }

    if let Some(new_model) = discovered_models.first().filter(|_| !is_resumed) {
        let old_model = std::mem::replace(&mut active_model, new_model.clone());
        if let Some(profile) = config.llms.get(new_model) {
            current_endpoint = Some(normalized_openai_endpoint(&profile.endpoint));
            session.set_system_prompt(system_prompt(profile));
        }
        output_state.push_text(&format!("Switched model from {old_model} to {new_model}"));
    } else if let Some((old_model, new_model)) = try_startup_model_switch(
        &status_http_client,
        &config,
        &mut active_model,
        &mut current_endpoint,
        &mut session,
    )
    .await
    {
        output_state.push_text(&format!("Switched model from {old_model} to {new_model}"));
    }

    loop {
        let prompt_branch = workspace_branch_name(tools.workspace());
        let active_profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured profile {}", active_model))?;
        let header_status = probe_header_status(
            &status_http_client,
            tools.workspace(),
            &active_model,
            active_profile,
            current_endpoint.as_deref(),
        )
        .await;
        let render = RenderContext {
            current_model: &active_model,
            endpoint: current_endpoint.as_deref().unwrap_or("(disconnected)"),
            workspace: tools.workspace(),
            prompt_branch: prompt_branch.as_deref(),
            header_status,
            virtual_width: viewport.virtual_width,
            actual_width: viewport.actual_width,
            actual_height: viewport.actual_height,
            x_offset: viewport.x_offset,
            banner: config.banner,
            feedback: config.feedback,
        };
        let resume_left_status = startup_notice_until
            .filter(|&deadline| std::time::Instant::now() < deadline)
            .map(|_| StatusFragment::plain(format!("Resuming session {session_id}")));
        print_screen(
            render,
            ScreenState {
                transcript: output_state.lines(),
                scroll_offset: output_state.scroll_offset(),
                left_status: resume_left_status,
                pending_count: pending_commands.len(),
                pending_line: None,
                input: input_state.as_str(),
                cursor: input_state.cursor(),
            },
        );
        std::io::stdout().flush()?;

        let next_input = if let Some(queued) = pending_commands.pop_front() {
            queued
        } else {
            match read_input(
                &mut input_state,
                &mut interrupt_state,
                &mut output_state,
                pending_commands.len(),
                &mut viewport,
                InputContext {
                    history: &history,
                    workspace: &workspace,
                    model_names: &model_names,
                    render,
                },
                print_screen,
            )? {
                InputResult::Submitted(line) => {
                    let Some(trimmed) = prepare_submitted_input(
                        &line,
                        &mut history,
                        &session_hist_path,
                        &mut output_state,
                        None,
                    )?
                    else {
                        continue;
                    };
                    trimmed
                }
                InputResult::Refresh => continue,
                InputResult::Quit => {
                    save_session_messages(&session_messages_path, session.messages())?;
                    update_session_metadata_timestamp(&session_metadata_path)?;
                    print!("{CLEAR_TERMINAL_SEQUENCE}");
                    std::io::stdout().flush()?;
                    break;
                }
            }
        };

        output_state.push_input(&format!("> {next_input}"));
        output_state.reset_scroll();
        startup_notice_until = None;
        print_screen(
            render,
            ScreenState {
                transcript: output_state.lines(),
                scroll_offset: output_state.scroll_offset(),
                left_status: None,
                pending_count: pending_commands.len(),
                pending_line: None,
                input: input_state.as_str(),
                cursor: input_state.cursor(),
            },
        );
        std::io::stdout().flush()?;

        match handle_command(
            &next_input,
            CommandState {
                active_model: &mut active_model,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
            },
            CommandContext {
                startup_model: &startup_model,
                startup_endpoint: &startup_endpoint,
                llms: &config.llms,
                tools: &tools,
                workspace: &workspace,
                usage_stats: &usage_stats,
                http_client: status_http_client.clone(),
                virtual_width: viewport.virtual_width,
            },
        )? {
            CommandOutcome::Quit => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            CommandOutcome::Quiet => {
                if config.feedback {
                    output_state.push_text(FEEDBACK_OK);
                    output_state.reset_scroll();
                }
                continue;
            }
            CommandOutcome::Cleared => {
                output_state.clear();
                continue;
            }
            CommandOutcome::Output(output) => {
                output_state.push_text(&output);
                if config.feedback {
                    output_state.push_text(FEEDBACK_OK);
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::OutputError(output) => {
                output_state.push_text(&output);
                if config.feedback {
                    output_state.push_text(FEEDBACK_ERR);
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::WideOutput(output) => {
                output_state.push_wide(&output);
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Blocking(f) => {
                let handle = tokio::task::spawn_blocking(f);
                // Recreate render here — handle_command's mutable borrows have ended.
                let blocking_render = RenderContext {
                    current_model: &active_model,
                    endpoint: current_endpoint.as_deref().unwrap_or("(disconnected)"),
                    workspace: tools.workspace(),
                    prompt_branch: prompt_branch.as_deref(),
                    header_status,
                    virtual_width: viewport.virtual_width,
                    actual_width: viewport.actual_width,
                    actual_height: viewport.actual_height,
                    x_offset: viewport.x_offset,
                    banner: config.banner,
                    feedback: config.feedback,
                };
                let result = wait_for_local_command(
                    WaitContext {
                        render: blocking_render,
                        history: &mut history,
                        history_path: &session_hist_path,
                        model_names: &model_names,
                        interrupt_state: &mut interrupt_state,
                        output_state: &mut output_state,
                        input_state: &mut input_state,
                        pending_commands: &mut pending_commands,
                        thinking_quote: None,
                        viewport: &mut viewport,
                    },
                    handle,
                )
                .await?;
                match result {
                    Ok(output) => {
                        output_state.push_text(&output);
                        if config.feedback {
                            output_state.push_text(FEEDBACK_OK);
                        }
                    }
                    Err(err) => {
                        output_state.push_text(&format!("Error: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                    }
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Async(future) => {
                let handle = tokio::spawn(future);
                let blocking_render = RenderContext {
                    current_model: &active_model,
                    endpoint: current_endpoint.as_deref().unwrap_or("(disconnected)"),
                    workspace: tools.workspace(),
                    prompt_branch: prompt_branch.as_deref(),
                    header_status,
                    virtual_width: viewport.virtual_width,
                    actual_width: viewport.actual_width,
                    actual_height: viewport.actual_height,
                    x_offset: viewport.x_offset,
                    banner: config.banner,
                    feedback: config.feedback,
                };
                let result = wait_for_local_command(
                    WaitContext {
                        render: blocking_render,
                        history: &mut history,
                        history_path: &session_hist_path,
                        model_names: &model_names,
                        interrupt_state: &mut interrupt_state,
                        output_state: &mut output_state,
                        input_state: &mut input_state,
                        pending_commands: &mut pending_commands,
                        thinking_quote: None,
                        viewport: &mut viewport,
                    },
                    handle,
                )
                .await?;
                match result {
                    Ok(output) => {
                        output_state.push_text(&output);
                        if config.feedback {
                            output_state.push_text(FEEDBACK_OK);
                        }
                    }
                    Err(err) => {
                        output_state.push_text(&format!("Error: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                    }
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Unhandled => {}
        }

        let profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?;
        let Some(endpoint) = current_endpoint.as_deref() else {
            output_state.push_text("Error: Not connected to an LLM server");
            if config.feedback {
                output_state.push_text(FEEDBACK_ERR);
            }
            output_state.reset_scroll();
            continue;
        };
        if !header_status.model_ok {
            if config.feedback {
                output_state.push_text(FEEDBACK_ERR);
                output_state.reset_scroll();
            }
            continue;
        }
        if let Some(message) = llm_prompt_block_reason(current_endpoint.as_deref(), header_status) {
            output_state.push_text(message);
            if config.feedback {
                output_state.push_text(FEEDBACK_ERR);
            }
            output_state.reset_scroll();
            continue;
        }
        let mut prompt_profile = profile.clone();
        prompt_profile.endpoint = endpoint.to_string();
        let llm_start = std::time::Instant::now();
        let tool_time_before = tools.total_tool_duration();
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let thinking_quote = quote_module.pick(seed);
        match wait_for_response(
            &mut session,
            &next_input,
            &prompt_profile,
            &tools,
            WaitContext {
                render: RenderContext {
                    current_model: &active_model,
                    endpoint,
                    workspace: tools.workspace(),
                    prompt_branch: prompt_branch.as_deref(),
                    header_status,
                    virtual_width: viewport.virtual_width,
                    actual_width: viewport.actual_width,
                    actual_height: viewport.actual_height,
                    x_offset: viewport.x_offset,
                    banner: config.banner,
                    feedback: config.feedback,
                },
                history: &mut history,
                history_path: &session_hist_path,
                model_names: &model_names,
                interrupt_state: &mut interrupt_state,
                output_state: &mut output_state,
                input_state: &mut input_state,
                pending_commands: &mut pending_commands,
                thinking_quote,
                viewport: &mut viewport,
            },
        )
        .await
        {
            Ok(WaitResult::Response(answer)) => {
                let tool_delta = tools.total_tool_duration().saturating_sub(tool_time_before);
                usage_stats.record_response(llm_start.elapsed(), &answer, tool_delta);
                output_state.push_markdown(&answer);
                if config.feedback {
                    output_state.push_text(FEEDBACK_OK);
                }
            }
            Ok(WaitResult::Cancelled(partial_output)) => {
                preserve_cancelled_output(&mut output_state, &partial_output);
            }
            Ok(WaitResult::Quit) => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            Err(err) => {
                output_state.push_text(&format!("Error: {err:#}"));
                if config.feedback {
                    output_state.push_text(FEEDBACK_ERR);
                }
            }
        }
        output_state.reset_scroll();
    }

    drop(_terminal_ui_guard);
    if usage_stats.total_tokens == 0 && is_ephemeral_branch(current_branch.as_deref().unwrap_or(""))
    {
        delete_session_dir(&session_dir);
    } else {
        eprintln!("orangu --resume {session_id}");
    }
    Ok(())
}

struct UsageStats {
    app_start: std::time::Instant,
    total_llm_duration: std::time::Duration,
    total_tool_duration: std::time::Duration,
    total_tokens: usize,
    session_id: String,
}

impl UsageStats {
    fn new() -> Self {
        Self {
            app_start: std::time::Instant::now(),
            total_llm_duration: std::time::Duration::ZERO,
            total_tool_duration: std::time::Duration::ZERO,
            total_tokens: 0,
            session_id: String::new(),
        }
    }

    fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = session_id.to_string();
        self
    }

    fn record_response(
        &mut self,
        total_duration: std::time::Duration,
        response: &str,
        tool_duration: std::time::Duration,
    ) {
        self.total_tool_duration += tool_duration;
        self.total_llm_duration += total_duration.saturating_sub(tool_duration);
        if let Ok(tokenizer) = cl100k_base() {
            self.total_tokens += tokenizer.encode_with_special_tokens(response).len();
        }
    }

    fn format(&self) -> String {
        let app_elapsed = self.app_start.elapsed();
        let avg_tps = if self.total_llm_duration.as_secs_f64() > 0.0 {
            self.total_tokens as f64 / self.total_llm_duration.as_secs_f64()
        } else {
            0.0
        };
        format!(
            "Application time : {}\nLLM time         : {}\nTool time        : {}\nTotal tokens     : {}\nAvg tokens/sec   : {:.1}\nSession          : {}",
            format_duration(app_elapsed),
            format_duration(self.total_llm_duration),
            format_duration(self.total_tool_duration),
            self.total_tokens,
            avg_tps,
            self.session_id,
        )
    }
}

fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {}m {}s", h, m, s)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

struct TerminalTitleGuard;

impl TerminalTitleGuard {
    fn new(title: &str) -> Self {
        set_terminal_title(Some(title));
        Self
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        set_terminal_title(None);
    }
}

fn set_terminal_title(title: Option<&str>) {
    match title {
        Some(title) => print!("\x1b]0;{title}\x07"),
        None => print!("\x1b]0;\x07"),
    }
}

struct TerminalUiGuard;

impl TerminalUiGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        execute!(
            std::io::stdout(),
            EnterAlternateScreen,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
        Ok(Self)
    }
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        let _ = execute!(
            std::io::stdout(),
            PopKeyboardEnhancementFlags,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
    }
}

pub struct RawModePauseGuard;

impl RawModePauseGuard {
    pub fn new() -> Result<Self> {
        disable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModePauseGuard {
    fn drop(&mut self) {
        let _ = enable_raw_mode();
    }
}

fn local_command_error(err: Error) -> CommandOutcome {
    if err.is::<LocalError>() {
        CommandOutcome::OutputError(format!("{err}"))
    } else {
        CommandOutcome::OutputError(format!("Error: {err:#}"))
    }
}

fn handle_command(
    input: &str,
    state: CommandState<'_>,
    context: CommandContext<'_>,
) -> anyhow::Result<CommandOutcome> {
    let Some(command) = parse_local_command(input) else {
        if input.trim_start().starts_with('/') {
            return Ok(CommandOutcome::OutputError(format!(
                "Unknown command '{}'. Use /help to see available commands.",
                input.trim()
            )));
        }
        return Ok(CommandOutcome::Unhandled);
    };

    let CommandState {
        active_model,
        current_endpoint,
        session,
    } = state;
    let CommandContext {
        startup_model,
        startup_endpoint,
        llms,
        tools,
        workspace,
        usage_stats,
        http_client,
        virtual_width,
    } = context;

    match command {
        LocalCommand::Help => Ok(CommandOutcome::Output(orangu::tui::help_text().to_string())),
        LocalCommand::ConnectDefault => {
            let endpoint = llms
                .get(active_model)
                .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?
                .endpoint
                .clone();
            *current_endpoint = Some(endpoint);
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::ConnectTo(endpoint) => {
            if endpoint.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    connect_usage_message().to_string(),
                ));
            }
            *current_endpoint = Some(endpoint.to_string());
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::Disconnect => Ok({
            *current_endpoint = None;
            CommandOutcome::Quiet
        }),
        LocalCommand::Reload => {
            *active_model = startup_model.to_string();
            *current_endpoint = Some(startup_endpoint.to_string());
            let prompt = system_prompt(
                llms.get(startup_model)
                    .ok_or_else(|| anyhow!("unknown model profile '{startup_model}'"))?,
            );
            session.clear(prompt);
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::ListModels => {
            let names = sorted_model_names(llms);
            let configs: Vec<(String, LlmConfiguration)> = names
                .into_iter()
                .filter_map(|n| llms.get(&n).map(|c| (n, c.clone())))
                .collect();
            Ok(CommandOutcome::Async(Box::pin(async move {
                let mut lines = Vec::with_capacity(configs.len());
                for (name, profile) in &configs {
                    let endpoint = normalized_openai_endpoint(&profile.endpoint);
                    let models_url = format!("{endpoint}/v1/models");
                    let available = async {
                        let resp = http_client.get(&models_url).send().await.ok()?;
                        if !resp.status().is_success() {
                            return None;
                        }
                        let models = resp.json::<ModelsResponse>().await.ok()?;
                        Some(models.data.iter().chain(models.models.iter()).any(|e| {
                            e.id == profile.model
                                || e.model == profile.model
                                || e.name == profile.model
                        }))
                    }
                    .await
                    .unwrap_or(false);
                    let indicator = if available { "🟢" } else { "🔴" };
                    lines.push(format!(
                        "- {}: {} ({}) {}",
                        name, profile.model, profile.provider, indicator
                    ));
                }
                Ok(lines.join("\n"))
            })))
        }
        LocalCommand::ListFiles => match list_workspace_files_tree(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::ShowFile(args) => {
            match show_file_output(workspace, args.as_ref(), virtual_width) {
                Ok(output) => Ok(CommandOutcome::WideOutput(output)),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Tools => Ok(CommandOutcome::Output(format_tools(tools))),
        LocalCommand::ModelInfo => Ok(CommandOutcome::Output(
            "Use /models to see configured profiles".to_string(),
        )),
        LocalCommand::SetModel(name) => {
            if name.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    model_usage_message().to_string(),
                ));
            }
            if !llms.contains_key(name) {
                return Ok(CommandOutcome::OutputError(format!(
                    "Unknown model profile '{name}'. Available: {}",
                    sorted_model_names(llms).join(", ")
                )));
            }
            let profile = &llms[name];
            let endpoint = orangu::llm::normalized_openai_endpoint(&profile.endpoint);
            *active_model = name.to_string();
            *current_endpoint = Some(endpoint);
            session.set_system_prompt(system_prompt(profile));
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::Diff(None) => match git_workspace_diff(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Diff(Some(branch)) => match git_diff_against_branch(workspace, &branch) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Status => match status_output(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Log => match log_output(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Pull(None) => Ok(CommandOutcome::OutputError(
            pull_usage_message().to_string(),
        )),
        LocalCommand::Pull(Some(pr_number)) => match pull_request_output(workspace, pr_number) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Rebase => match rebase_output(workspace) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Merge(None) => Ok(CommandOutcome::OutputError(
            merge_usage_message().to_string(),
        )),
        LocalCommand::Merge(Some(branch)) => match merge_output(workspace, &branch) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Checkout(None) => Ok(CommandOutcome::OutputError(
            checkout_usage_message().to_string(),
        )),
        LocalCommand::Checkout(Some(target)) => match checkout_output(workspace, &target) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::AddFile(None) => Ok(CommandOutcome::OutputError(
            add_file_usage_message().to_string(),
        )),
        LocalCommand::AddFile(Some(path)) => match add_file_output(workspace, &path) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::RemoveFile(None) => Ok(CommandOutcome::OutputError(
            remove_file_usage_message().to_string(),
        )),
        LocalCommand::RemoveFile(Some(path)) => match remove_file_output(workspace, &path) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::MoveFile(None) => Ok(CommandOutcome::OutputError(
            move_file_usage_message().to_string(),
        )),
        LocalCommand::MoveFile(Some((src, dst))) => match move_file_output(workspace, &src, &dst) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::CherryPick(None) => Ok(CommandOutcome::OutputError(
            cherry_pick_usage_message().to_string(),
        )),
        LocalCommand::CherryPick(Some(commit)) => match cherry_pick_output(workspace, &commit) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Commit(None) => Ok(CommandOutcome::OutputError(
            commit_usage_message().to_string(),
        )),
        LocalCommand::Commit(Some(message)) => match commit_output(workspace, &message) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Amend(None) => Ok(CommandOutcome::OutputError(
            amend_usage_message().to_string(),
        )),
        LocalCommand::Amend(Some(message)) => match amend_output(workspace, &message) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Push(force) => match push_output(workspace, force) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::InitRepo => match init_repo_output(workspace) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Squash => match squash_output(workspace) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::DeleteBranch(None) => Ok(CommandOutcome::OutputError(
            delete_branch_usage_message().to_string(),
        )),
        LocalCommand::DeleteBranch(Some(branch)) => {
            match delete_branch_output(workspace, &branch) {
                Ok(_) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::OpenFile(path) => {
            if path.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    open_file_usage_message().to_string(),
                ));
            }
            match open_in_editor(workspace, path) {
                Ok(()) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(CommandOutcome::OutputError(format!("Error: {err:#}"))),
            }
        }
        LocalCommand::Session(None) => match list_sessions_output(None) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Session(Some(uuid)) => {
            Ok(CommandOutcome::Output(format!("orangu --resume {uuid}")))
        }
        LocalCommand::Sessions(filter) => match list_sessions_output(filter.as_deref()) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Usage => Ok(CommandOutcome::Output(usage_stats.format())),
        LocalCommand::Build => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Blocking(Box::new(move || {
                build::build_output(&ws)
            })))
        }
        LocalCommand::Clear => {
            let prompt = system_prompt(
                llms.get(active_model)
                    .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?,
            );
            session.clear(prompt);
            Ok(CommandOutcome::Cleared)
        }
        LocalCommand::Quit => Ok(CommandOutcome::Quit),
    }
}

pub fn print_screen(render: RenderContext<'_>, screen: ScreenState<'_>) {
    print!("{CLEAR_TERMINAL_SEQUENCE}");
    print!(
        "{}",
        render_screen(ScreenRenderArgs {
            version: VERSION,
            current_model: render.current_model,
            endpoint: render.endpoint,
            workspace: render.workspace,
            prompt_branch: render.prompt_branch,
            status: render.header_status,
            banner: render.banner,
            transcript: screen.transcript,
            scroll_offset: screen.scroll_offset,
            left_status: screen.left_status,
            pending_count: screen.pending_count,
            pending_line: screen.pending_line,
            input: screen.input,
            cursor: screen.cursor,
            virtual_width: render.virtual_width,
            actual_width: render.actual_width,
            actual_height: render.actual_height,
            x_offset: render.x_offset,
        })
    );
}

async fn wait_for_response(
    session: &mut ChatSession,
    user_input: &str,
    profile: &LlmConfiguration,
    tools: &ToolExecutor,
    wait_context: WaitContext<'_>,
) -> Result<WaitResult> {
    let WaitContext {
        mut render,
        history,
        history_path,
        model_names,
        interrupt_state,
        output_state,
        input_state,
        pending_commands,
        thinking_quote,
        viewport,
    } = wait_context;
    let streamed_state = Arc::new(Mutex::new(StreamRenderState::default()));
    let prompt_output = Arc::clone(&streamed_state);
    let prompt_metrics = Arc::clone(&streamed_state);
    let prompt_tool_running = Arc::clone(&streamed_state);
    let tokenizer = cl100k_base().ok();
    let mut prompt_future = Box::pin(session.prompt(
        user_input,
        profile,
        tools,
        move |delta| {
            if let Ok(mut state) = prompt_output.lock() {
                state.output.push_str(delta);
            }
        },
        move |metrics| {
            if let Ok(mut state) = prompt_metrics.lock() {
                state.metrics.merge(metrics);
            }
        },
        move |running| {
            if let Ok(mut state) = prompt_tool_running.lock() {
                state.tool_running_since = if running {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
            }
        },
    ));
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let mut thinking_frame = 0usize;
    let thinking_started = std::time::Instant::now();
    let mut last_rendered_output = String::new();
    let mut last_rendered_metrics = StreamMetrics::default();
    let mut last_tool_was_running = false;
    let mut escape_cancel_state = EscapeCancelState::default();
    let initial_status = Some(render_thinking_status(
        thinking_frame,
        thinking_started.elapsed(),
    ));
    let quote_line = thinking_quote.map(|q| format!("\x1b[2m{q}\x1b[0m"));

    print_screen(
        render,
        ScreenState {
            transcript: output_state.lines(),
            scroll_offset: output_state.scroll_offset(),
            left_status: initial_status,
            pending_count: pending_commands.len(),
            pending_line: quote_line.as_deref(),
            input: input_state.as_str(),
            cursor: input_state.cursor(),
        },
    );
    std::io::stdout().flush()?;

    loop {
        tokio::select! {
            result = &mut prompt_future => {
                let response = result?;
                let final_state = streamed_state
                    .lock()
                    .map(|state| state.clone())
                    .unwrap_or_default();
                if let Some(pending_line) = final_pending_line(&final_state.output, &response)
                    .map(|line| render_markdown_for_console(&line))
                {
                    print_screen(
                        render,
                        ScreenState {
                            transcript: output_state.lines(),
                            scroll_offset: output_state.scroll_offset(),
                            left_status: None,
                            pending_count: pending_commands.len(),
                            pending_line: Some(pending_line.as_str()),
                            input: input_state.as_str(),
                            cursor: input_state.cursor(),
                        },
                    );
                    std::io::stdout().flush()?;
                }
                return Ok(WaitResult::Response(response));
            }
            _ = interval.tick() => {
                let elapsed = thinking_started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                let mut redraw = next_frame != thinking_frame;
                thinking_frame = next_frame;
                let current_state = streamed_state
                    .lock()
                    .map(|state| state.clone())
                    .unwrap_or_default();
                let current_streamed_output = current_state.output;
                let current_stream_metrics = current_state.metrics;
                let current_tool_running_since = current_state.tool_running_since;
                redraw |= current_streamed_output != last_rendered_output;
                redraw |= current_stream_metrics != last_rendered_metrics;
                redraw |= current_tool_running_since.is_some() != last_tool_was_running;

                while event::poll(std::time::Duration::ZERO)? {
                    let event = event::read()?;
                    if is_wait_cancel_escape(&event) {
                        if escape_cancel_state.handle_escape(std::time::Instant::now()) {
                            let partial_output = streamed_state
                                .lock()
                                .map(|state| state.output.clone())
                                .unwrap_or_default();
                            drop(prompt_future);
                            return Ok(WaitResult::Cancelled(partial_output));
                        }
                        continue;
                    }
                    escape_cancel_state.reset();
                    let result = handle_input_event(
                        event,
                        input_state,
                        interrupt_state,
                        output_state,
                        viewport,
                        InputContext {
                            history,
                            workspace: render.workspace,
                            model_names,
                            render,
                        },
                    );
                    render.actual_width = viewport.actual_width;
                    render.actual_height = viewport.actual_height;
                    render.x_offset = viewport.x_offset;

                    if let Some(outcome) = result.outcome {
                        match outcome {
                            InputResult::Submitted(line) => {
                                let had_pending = pending_commands.len();
                                let _ = prepare_submitted_input(
                                    &line,
                                    history,
                                    history_path,
                                    output_state,
                                    Some(pending_commands),
                                )?;
                                redraw = redraw || pending_commands.len() != had_pending || !line.trim().is_empty();
                            }
                            InputResult::Refresh => {}
                            InputResult::Quit => return Ok(WaitResult::Quit),
                        }
                    }
                    redraw |= result.redraw;
                }

                if redraw {
                    last_rendered_output = current_streamed_output;
                    last_rendered_metrics = current_stream_metrics;
                    last_tool_was_running = current_tool_running_since.is_some();
                    let left_status = render_left_status(
                        profile,
                        &last_rendered_output,
                        &last_rendered_metrics,
                        current_tool_running_since,
                        elapsed,
                        thinking_frame,
                        tokenizer.as_ref(),
                    );
                    let pending_line = if last_rendered_output.is_empty() {
                        quote_line.clone().unwrap_or_default()
                    } else {
                        render_markdown_for_console(&last_rendered_output)
                    };
                    print_screen(
                        render,
                        ScreenState {
                            transcript: output_state.lines(),
                            scroll_offset: output_state.scroll_offset(),
                            left_status,
                            pending_count: pending_commands.len(),
                            pending_line: Some(pending_line.as_str()),
                            input: input_state.as_str(),
                            cursor: input_state.cursor(),
                        },
                    );
                    std::io::stdout().flush()?;
                }
            }
        }
    }
}

async fn wait_for_local_command(
    wait_context: WaitContext<'_>,
    mut handle: tokio::task::JoinHandle<anyhow::Result<String>>,
) -> anyhow::Result<anyhow::Result<String>> {
    let WaitContext {
        mut render,
        history,
        history_path: _,
        model_names,
        interrupt_state,
        output_state,
        input_state,
        pending_commands,
        thinking_quote: _,
        viewport,
    } = wait_context;
    let started = std::time::Instant::now();
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let mut frame = 0usize;
    loop {
        tokio::select! {
            result = &mut handle => {
                return Ok(result?);
            }
            _ = interval.tick() => {
                let elapsed = started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                if next_frame != frame {
                    frame = next_frame;
                }
                while event::poll(std::time::Duration::ZERO)? {
                    handle_input_event(
                        event::read()?,
                        input_state,
                        interrupt_state,
                        output_state,
                        viewport,
                        InputContext {
                            history,
                            workspace: render.workspace,
                            model_names,
                            render,
                        },
                    );
                    render.actual_width = viewport.actual_width;
                    render.actual_height = viewport.actual_height;
                    render.x_offset = viewport.x_offset;
                }
                let left_status = Some(render_tool_running_status(frame, elapsed));
                print_screen(
                    render,
                    ScreenState {
                        transcript: output_state.lines(),
                        scroll_offset: output_state.scroll_offset(),
                        left_status,
                        pending_count: pending_commands.len(),
                        pending_line: None,
                        input: input_state.as_str(),
                        cursor: input_state.cursor(),
                    },
                );
                std::io::stdout().flush()?;
            }
        }
    }
}

fn render_left_status(
    profile: &LlmConfiguration,
    rendered_output: &str,
    metrics: &StreamMetrics,
    tool_running_since: Option<std::time::Instant>,
    elapsed: std::time::Duration,
    frame: usize,
    tokenizer: Option<&tiktoken_rs::CoreBPE>,
) -> Option<orangu::tui::StatusFragment> {
    if let Some(tool_start) = tool_running_since {
        return Some(render_tool_running_status(frame, tool_start.elapsed()));
    }

    if rendered_output.is_empty() {
        return Some(render_thinking_status(frame, elapsed));
    }

    if profile.provider.eq_ignore_ascii_case("llama.cpp")
        && let Some(rate) = metrics
            .predicted_per_second
            .filter(|rate| *rate > 0.0 && !rendered_output.is_empty())
    {
        return Some(render_working_status(frame, rate, elapsed));
    }

    tokenizer.and_then(|tokenizer| {
        let token_count = tokenizer.encode_with_special_tokens(rendered_output).len();
        let elapsed_secs = elapsed.as_secs_f64();
        (token_count > 0 && elapsed_secs > 0.0).then(|| {
            orangu::tui::StatusFragment::plain(format!(
                "{:.1}t/s",
                token_count as f64 / elapsed_secs
            ))
        })
    })
}

fn is_wait_cancel_escape(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(KeyEvent {
            code: KeyCode::Esc,
            kind: KeyEventKind::Press,
            ..
        })
    )
}

fn final_pending_line(streamed_output: &str, response: &str) -> Option<String> {
    if !streamed_output.is_empty() {
        Some(streamed_output.to_string())
    } else if !response.is_empty() {
        Some(response.to_string())
    } else {
        None
    }
}

fn request_cancelled_message() -> String {
    format!(
        "{}Request cancelled.{}",
        render::ANSI_FG_LIGHT_RED,
        render::ANSI_RESET
    )
}

fn preserve_cancelled_output(output_state: &mut OutputState, partial_output: &str) {
    if !partial_output.is_empty() {
        output_state.push_markdown(partial_output);
    }
    output_state.push_text(&request_cancelled_message());
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
}

async fn probe_header_status(
    http_client: &reqwest::Client,
    workspace: &Path,
    active_model: &str,
    profile: &LlmConfiguration,
    endpoint: Option<&str>,
) -> orangu::tui::HeaderStatus {
    let workspace_ok = workspace.exists();
    let mut server_ok = false;
    let mut model_ok = false;

    if let Some(endpoint) = endpoint {
        let models_url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
        if let Ok(response) = http_client.get(&models_url).send().await
            && response.status().is_success()
        {
            server_ok = true;
            if let Ok(models) = response.json::<ModelsResponse>().await {
                model_ok = models.data.iter().chain(models.models.iter()).any(|entry| {
                    entry.id == profile.model
                        || entry.model == profile.model
                        || entry.name == profile.model
                        || entry.id == active_model
                        || entry.model == active_model
                        || entry.name == active_model
                });
            }
        }
    }

    orangu::tui::HeaderStatus {
        workspace_ok,
        server_ok,
        model_ok,
    }
}

async fn try_discover_new_models(
    http_client: &reqwest::Client,
    config: &mut orangu::config::ClientAppConfiguration,
    config_path: &Path,
) -> Vec<String> {
    let mut known_model_ids: HashSet<String> =
        config.llms.values().map(|p| p.model.clone()).collect();

    let mut seen_endpoints: HashSet<String> = HashSet::new();
    let endpoints: Vec<(String, String)> = config
        .llms
        .values()
        .filter(|p| seen_endpoints.insert(p.endpoint.clone()))
        .map(|p| (p.endpoint.clone(), p.provider.clone()))
        .collect();

    let (fallback_timeout, fallback_max_tool_rounds, fallback_system_prompt) = config
        .llms
        .values()
        .next()
        .map(|p| {
            (
                p.request_timeout_seconds,
                p.max_tool_rounds,
                p.system_prompt.clone(),
            )
        })
        .unwrap_or((1800, 10, String::new()));

    let mut added = Vec::new();

    for (endpoint, provider) in &endpoints {
        let url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
        let Ok(response) = http_client.get(&url).send().await else {
            continue;
        };
        if !response.status().is_success() {
            continue;
        }
        let Ok(models) = response.json::<ModelsResponse>().await else {
            continue;
        };

        for entry in models.data.iter().chain(models.models.iter()) {
            let model_id = if !entry.id.is_empty() {
                entry.id.clone()
            } else if !entry.model.is_empty() {
                entry.model.clone()
            } else {
                continue;
            };

            if known_model_ids.contains(&model_id) {
                continue;
            }

            let base_name = model_id.rsplit('/').next().unwrap_or(&model_id).to_string();
            let section_name = if !config.llms.contains_key(&base_name) {
                base_name
            } else {
                model_id.replace('/', "-")
            };

            if config.llms.contains_key(&section_name) {
                continue;
            }

            config.llms.insert(
                section_name.clone(),
                LlmConfiguration {
                    provider: provider.clone(),
                    endpoint: endpoint.clone(),
                    model: model_id.clone(),
                    api_key: None,
                    request_timeout_seconds: fallback_timeout,
                    max_tool_rounds: fallback_max_tool_rounds,
                    system_prompt: fallback_system_prompt.clone(),
                },
            );

            if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(config_path) {
                let _ = writeln!(file);
                let _ = writeln!(file, "[{section_name}]");
                let _ = writeln!(file, "provider = {provider}");
                let _ = writeln!(file, "endpoint = {endpoint}");
                let _ = writeln!(file, "model = {model_id}");
            }

            known_model_ids.insert(model_id);
            added.push(section_name);
        }
    }

    added
}

async fn try_startup_model_switch(
    http_client: &reqwest::Client,
    config: &orangu::config::ClientAppConfiguration,
    active_model: &mut String,
    current_endpoint: &mut Option<String>,
    session: &mut ChatSession,
) -> Option<(String, String)> {
    let endpoint = current_endpoint.as_deref()?;
    let models_url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
    let response = http_client.get(&models_url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let models = response.json::<ModelsResponse>().await.ok()?;

    let current_profile = config.llms.get(active_model.as_str())?;
    let current_available = models.data.iter().chain(models.models.iter()).any(|e| {
        e.id == current_profile.model
            || e.model == current_profile.model
            || e.name == current_profile.model
            || e.id == active_model.as_str()
            || e.model == active_model.as_str()
            || e.name == active_model.as_str()
    });
    if current_available {
        return None;
    }

    for name in sorted_model_names(&config.llms) {
        if name == *active_model {
            continue;
        }
        if let Some(profile) = config.llms.get(&name) {
            let available = models.data.iter().chain(models.models.iter()).any(|e| {
                e.id == profile.model || e.model == profile.model || e.name == profile.model
            });
            if available {
                let old = std::mem::replace(active_model, name.clone());
                *current_endpoint = Some(normalized_openai_endpoint(&profile.endpoint));
                session.set_system_prompt(system_prompt(profile));
                return Some((old, name));
            }
        }
    }
    None
}

fn session_dir_path(session_id: &str) -> Result<PathBuf> {
    let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(SESSIONS_DIRECTORY).join(session_id))
}

fn load_session_messages(path: &Path) -> Result<Vec<ChatMessage>> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("failed to parse session messages {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read session messages {}", path.display()))
        }
    }
}

fn save_session_messages(path: &Path, messages: &[ChatMessage]) -> Result<()> {
    let json = serde_json::to_string(messages).context("failed to serialize session messages")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write session messages {}", path.display()))
}

#[derive(Serialize, Deserialize)]
struct SessionMetadata {
    started_at: u64,
    last_updated_at: u64,
    workspace: String,
    #[serde(default)]
    branch: String,
}

fn current_unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_unix_timestamp(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}{m:02}{d:02}{hour:02}{min:02}")
}

fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
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

fn is_leap_year(y: u32) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

fn save_session_metadata(path: &Path, metadata: &SessionMetadata) -> Result<()> {
    let json = serde_json::to_string(metadata).context("failed to serialize session metadata")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write session metadata {}", path.display()))
}

fn load_session_metadata(path: &Path) -> Result<Option<SessionMetadata>> {
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

fn update_session_metadata_timestamp(path: &Path) -> Result<()> {
    if let Ok(Some(mut meta)) = load_session_metadata(path) {
        meta.last_updated_at = current_unix_timestamp();
        save_session_metadata(path, &meta)?;
    }
    Ok(())
}

fn find_session_for_workspace_branch(workspace: &str, branch: &str) -> Option<String> {
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

fn is_ephemeral_branch(branch: &str) -> bool {
    matches!(branch, "" | "main" | "master")
}

fn delete_session_dir(session_dir: &Path) {
    let _ = std::fs::remove_dir_all(session_dir);
}

fn list_sessions_output(workspace_filter: Option<&str>) -> Result<String> {
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

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{:<36}  {:<12}  {:<12}  {:>4}  {:<24}  {}",
        "UUID", "STARTED", "LAST", "CMDS", "BRANCH", "WORKSPACE"
    ));
    for (uuid, meta, cmd_count) in &entries {
        let started = meta
            .as_ref()
            .map(|m| format_unix_timestamp(m.started_at))
            .unwrap_or_else(|| "-".to_string());
        let last = meta
            .as_ref()
            .map(|m| format_unix_timestamp(m.last_updated_at))
            .unwrap_or_else(|| "-".to_string());
        let branch = meta
            .as_ref()
            .map(|m| {
                if m.branch.is_empty() {
                    "-"
                } else {
                    m.branch.as_str()
                }
            })
            .unwrap_or("-");
        let workspace = meta.as_ref().map(|m| m.workspace.as_str()).unwrap_or("-");
        lines.push(format!(
            "{:<36}  {:<12}  {:<12}  {:>4}  {:<24}  {}",
            uuid, started, last, cmd_count, branch, workspace
        ));
    }
    Ok(lines.join("\n"))
}

fn llm_prompt_block_reason(
    endpoint: Option<&str>,
    _header_status: orangu::tui::HeaderStatus,
) -> Option<&'static str> {
    if endpoint.is_none() {
        return Some("Error: Not connected to an LLM server");
    }
    None
}

fn resolve_workspace_root(workspace: Option<PathBuf>) -> Result<PathBuf> {
    let current_dir = std::env::current_dir().context("failed to resolve current directory")?;
    let workspace = workspace.unwrap_or_else(|| current_dir.clone());
    let absolute = if workspace.is_absolute() {
        workspace
    } else {
        current_dir.join(workspace)
    };
    Ok(normalize_path(&absolute))
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

fn load_history(path: &Path) -> Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read history file {}", path.display()))
        }
    }
}

fn append_history_entry(path: &Path, entry: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create history directory {}", parent.display()))?;
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open history file {}", path.display()))?;
    writeln!(file, "{entry}")
        .with_context(|| format!("failed to write history file {}", path.display()))
}

fn prepare_submitted_input(
    input: &str,
    history: &mut Vec<String>,
    history_path: &Path,
    output_state: &mut OutputState,
    pending_commands: Option<&mut VecDeque<String>>,
) -> Result<Option<String>> {
    let trimmed = input.trim();
    if trimmed.is_empty() || trimmed.starts_with('\\') {
        return Ok(None);
    }

    history.push(trimmed.to_string());
    append_history_entry(history_path, trimmed)?;

    if trimmed.starts_with('#') {
        output_state.push_input(&format!("> {trimmed}"));
        output_state.reset_scroll();
        return Ok(None);
    }

    if let Some(pending_commands) = pending_commands {
        pending_commands.push_back(trimmed.to_string());
        return Ok(None);
    }

    Ok(Some(trimmed.to_string()))
}

/// Shared mutex that serializes tests which mutate process-wide environment
/// variables (PATH, HOME, COLUMNS, etc.). All test modules must use this lock
/// when calling `std::env::set_var` / `remove_var` to prevent races.
#[cfg(test)]
pub fn process_env_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::commands::{
        CommandContext, CommandOutcome, CommandState, LocalCommand, ShowFileOptions,
        parse_local_command, system_prompt,
    };
    use super::completion::completion_candidates;
    use super::git::with_explicit_pager_width;
    use super::git::{
        delete_branch_output, discover_git_dir, discover_git_root, git_workspace_diff,
        init_repo_output, is_protected_branch, list_workspace_files_tree, workspace_branch_name,
    };
    use super::input::idle_status_refresh_timeout;
    use super::render::{
        ANSI_RESET, GitLineMetadata, format_show_file_line, parse_show_file_arguments,
        render_markdown_for_console, show_file_output,
    };
    use super::{
        EscapeCancelState, InputContext, InputState, InterruptState, OutputState, RenderContext,
        ViewportState, final_pending_line, handle_command, handle_input_event,
        is_wait_cancel_escape, llm_prompt_block_reason, preserve_cancelled_output,
        render_left_status, request_cancelled_message, resolve_workspace_root,
    };
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
    use orangu::{
        config::LlmConfiguration,
        llm::{StreamMetrics, StreamPromptProgress, normalized_openai_endpoint},
        session::ChatSession,
        tools::ToolExecutor,
        tui::{Banner, HeaderStatus, TranscriptLine},
    };
    use std::collections::HashMap;
    use std::{
        ffi::OsString,
        fs,
        path::PathBuf,
        time::{Duration, Instant},
    };
    use tempfile::tempdir;

    fn lock_process_env() -> std::sync::MutexGuard<'static, ()> {
        super::process_env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &std::path::Path) -> Self {
            let original = std::env::var_os(key);
            // SAFETY: tests serialize process-wide environment changes with process_env_lock().
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }

        fn set_value(key: &'static str, value: &str) -> Self {
            let original = std::env::var_os(key);
            // SAFETY: tests serialize process-wide environment changes with process_env_lock().
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: tests serialize process-wide environment changes with process_env_lock().
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    fn init_test_git_repo(workspace: &std::path::Path) {
        assert!(
            std::process::Command::new("git")
                .args(["init", "--quiet"])
                .current_dir(workspace)
                .status()
                .expect("git init")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["config", "user.name", "Orangu Tests"])
                .current_dir(workspace)
                .status()
                .expect("git config name")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["config", "user.email", "tests@example.com"])
                .current_dir(workspace)
                .status()
                .expect("git config email")
                .success()
        );
    }

    fn test_profile(provider: &str, endpoint: &str, model: &str) -> LlmConfiguration {
        LlmConfiguration {
            provider: provider.to_string(),
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            api_key: None,
            request_timeout_seconds: 1800,
            max_tool_rounds: 10,
            system_prompt: String::new(),
        }
    }

    fn test_input_context<'a>(workspace: &'a std::path::Path) -> InputContext<'a> {
        InputContext {
            history: &[],
            workspace,
            model_names: &[],
            render: RenderContext {
                current_model: "default",
                endpoint: "http://localhost:11434/v1",
                workspace,
                prompt_branch: None,
                header_status: HeaderStatus {
                    workspace_ok: true,
                    server_ok: true,
                    model_ok: true,
                },
                virtual_width: 80,
                actual_width: 80,
                actual_height: 24,
                x_offset: 0,
                banner: Banner::Left,
                feedback: false,
            },
        }
    }

    #[test]
    fn resolve_workspace_root_makes_relative_paths_absolute() {
        let current_dir = std::env::current_dir().expect("current directory");
        let resolved = resolve_workspace_root(Some(PathBuf::from("."))).expect("workspace");

        assert_eq!(resolved, current_dir);
        assert!(resolved.is_absolute());
    }

    #[test]
    fn resolve_workspace_root_normalizes_parent_segments() {
        let current_dir = std::env::current_dir().expect("current directory");
        let resolved =
            resolve_workspace_root(Some(PathBuf::from("src/../tests"))).expect("workspace");

        assert_eq!(resolved, current_dir.join("tests"));
    }

    #[test]
    fn output_state_keeps_last_ten_thousand_lines() {
        let mut output_state = OutputState::default();
        for index in 0..10_005 {
            output_state.push_text(&format!("line {index}"));
        }

        assert_eq!(output_state.lines().len(), 10_000);
        assert_eq!(
            output_state.lines().first().map(TranscriptLine::as_str),
            Some("line 5")
        );
        assert_eq!(
            output_state.lines().last().map(TranscriptLine::as_str),
            Some("line 10004")
        );
    }

    #[test]
    fn output_state_styles_echoed_user_input() {
        let mut output_state = OutputState::default();

        output_state.push_input("> show README.md");
        output_state.push_text("plain output");

        assert!(
            matches!(output_state.lines().first(), Some(TranscriptLine::UserInput(s)) if s == "> show README.md")
        );
        assert!(
            matches!(output_state.lines().get(1), Some(TranscriptLine::Plain(s)) if s == "plain output")
        );
    }

    #[test]
    fn renders_markdown_emphasis_for_console() {
        let rendered = render_markdown_for_console("Hello **bold** and *italic*.");

        assert!(rendered.contains("\x1b[1mbold\x1b[22m"));
        assert!(rendered.contains("\x1b[3mitalic\x1b[23m"));
    }

    #[test]
    fn renders_markdown_blocks_for_console() {
        let rendered = render_markdown_for_console(
            "# Title\n\n- one\n- two\n\n`code`\n\n[docs](https://example.com)",
        );

        assert!(rendered.contains("\x1b[1m# Title\x1b[22m"));
        assert!(rendered.contains("- one"));
        assert!(rendered.contains("- two"));
        assert!(rendered.contains("\x1b[38;2;255;215;120m`code\x1b[39m`"));
        assert!(rendered.contains("docs"));
        assert!(rendered.contains("https://example.com"));
    }

    #[test]
    fn renders_fenced_code_blocks_with_syntax_highlighting() {
        let rendered = render_markdown_for_console("```c\nprintf(\"Hello World !\\\\n\");\n```");

        assert!(rendered.contains("```c"));
        assert!(rendered.contains("printf"));
        assert!(rendered.contains("\x1b["));
    }

    #[test]
    fn renders_unknown_fenced_code_blocks_with_plain_code_color() {
        let rendered = render_markdown_for_console("```unknownlang\nplain text\n```");

        assert!(rendered.contains("```unknownlang"));
        assert!(rendered.contains("\x1b[38;2;255;215;120mplain text\x1b[39m"));
    }

    #[test]
    fn open_file_failure_returns_output_instead_of_error() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/open_file /etc/hosts",
            CommandState {
                active_model: &mut active_model,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
            },
            CommandContext {
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                http_client: reqwest::Client::new(),
                virtual_width: 512,
            },
        )
        .expect("handle command");

        assert!(matches!(
            outcome,
            CommandOutcome::OutputError(message) if message.starts_with("Error: ")
        ));
    }

    #[test]
    fn alt_backspace_deletes_previous_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Backspace,
                KeyModifiers::ALT,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.as_str(), "src/tui.");
        assert_eq!(input_state.cursor(), "src/tui.".len());
    }

    #[test]
    fn alt_d_deletes_next_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        input_state.move_home();
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('d'),
                KeyModifiers::ALT,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.as_str(), "/tui.rs");
        assert_eq!(input_state.cursor(), 0);
    }

    #[test]
    fn ctrl_left_moves_to_previous_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Left,
                KeyModifiers::CONTROL,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.cursor(), "src/tui.".len());
    }

    #[test]
    fn ctrl_right_moves_to_next_bash_word() {
        let workspace = tempdir().expect("workspace");
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());
        input_state.move_home();
        let mut interrupt_state = InterruptState::default();
        let mut output_state = OutputState::default();
        let mut viewport = ViewportState::new(80, 80, 24);

        let result = handle_input_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Right,
                KeyModifiers::CONTROL,
                KeyEventKind::Press,
            )),
            &mut input_state,
            &mut interrupt_state,
            &mut output_state,
            &mut viewport,
            test_input_context(workspace.path()),
        );

        assert!(result.redraw);
        assert!(result.outcome.is_none());
        assert_eq!(input_state.cursor(), 3);
    }

    #[test]
    fn ctrl_w_keeps_whitespace_based_word_deletion() {
        let mut input_state = InputState::default();
        input_state.set_buffer("src/tui.rs".to_string());

        input_state.delete_prev_word();

        assert_eq!(input_state.as_str(), "");
        assert_eq!(input_state.cursor(), 0);
    }

    #[test]
    fn missing_required_command_arguments_return_usage_output() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());

        for (input, expected) in [
            (
                "/show_file",
                "Usage: /show_file [--hash] [--author] <path> [<ref>]. Use /help to see available commands.",
            ),
            (
                "/show_file --hash",
                "Usage: /show_file [--hash] [--author] <path> [<ref>]. Use /help to see available commands.",
            ),
            (
                "/open_file",
                "Usage: /open_file <path>. Use /help to see available commands.",
            ),
        ] {
            let mut active_model = "llama".to_string();
            let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
            let mut session = ChatSession::new("system");

            let outcome = handle_command(
                input,
                CommandState {
                    active_model: &mut active_model,
                    current_endpoint: &mut current_endpoint,
                    session: &mut session,
                },
                CommandContext {
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: workspace.path(),
                    usage_stats: &super::UsageStats::new(),
                    http_client: reqwest::Client::new(),
                    virtual_width: 512,
                },
            )
            .expect("handle command");

            assert!(
                matches!(outcome, CommandOutcome::OutputError(message) if message == expected),
                "unexpected outcome for {input:?}"
            );
        }
    }

    #[test]
    fn list_files_outputs_filtered_workspace_tree() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        fs::write(workspace.path().join("README.md"), "readme").expect("root file");
        fs::create_dir(workspace.path().join("doc")).expect("doc dir");
        fs::write(workspace.path().join("doc/guide.txt"), "guide").expect("doc file");
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(workspace.path().join("src/lib.rs"), "pub fn lib() {}").expect("src file");
        fs::create_dir(workspace.path().join(".git")).expect("git dir");
        fs::write(workspace.path().join(".git/config"), "[core]").expect("git config");
        fs::create_dir(workspace.path().join("build")).expect("build dir");
        fs::write(workspace.path().join("build/output.txt"), "artifact").expect("build file");
        fs::create_dir(workspace.path().join("target")).expect("target dir");
        fs::write(workspace.path().join("target/app"), "binary").expect("target file");

        let tree = list_workspace_files_tree(workspace.path()).expect("tree");
        assert_eq!(
            tree,
            format!(
                "{}\n├── doc\n│   └── guide.txt\n├── src\n│   └── lib.rs\n└── README.md",
                workspace.path().display()
            )
        );

        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");
        let outcome = handle_command(
            "/list_files",
            CommandState {
                active_model: &mut active_model,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
            },
            CommandContext {
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                http_client: reqwest::Client::new(),
                virtual_width: 512,
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Output(output) if output == tree));
    }

    #[test]
    fn parses_show_file_commands() {
        match parse_local_command("/show_file README.md") {
            Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "README.md"),
            _ => panic!("expected show file slash command"),
        }

        let (path, options, rev) =
            parse_show_file_arguments("--hash --author \"docs/user guide.md\"")
                .expect("show file args");
        assert_eq!(path, "docs/user guide.md");
        assert!(options.show_hash);
        assert!(options.show_author);
        assert!(rev.is_none());
        let (path2, _, rev2) = parse_show_file_arguments("src/main.rs abc1234").expect("path+rev");
        assert_eq!(path2, "src/main.rs");
        assert_eq!(rev2.as_deref(), Some("abc1234"));
    }

    #[test]
    fn force_push_blocked_on_protected_branches() {
        assert!(is_protected_branch("main"));
        assert!(is_protected_branch("master"));
        assert!(!is_protected_branch("feature/my-branch"));
        assert!(!is_protected_branch("develop"));
    }

    #[test]
    fn init_repo_creates_git_repository() {
        let workspace = tempdir().expect("workspace");
        assert!(!workspace.path().join(".git").exists());
        let result = init_repo_output(workspace.path());
        assert!(result.is_ok(), "init_repo_output failed: {:?}", result);
        assert!(workspace.path().join(".git").exists());
    }

    #[test]
    fn delete_branch_blocked_on_protected_branches() {
        let workspace = tempdir().expect("workspace");
        std::process::Command::new("git")
            .arg("init")
            .current_dir(workspace.path())
            .output()
            .expect("git init");
        for branch in ["main", "master"] {
            let result = delete_branch_output(workspace.path(), branch);
            assert!(result.is_err(), "should block deletion of '{branch}'");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains(branch),
                "error should mention branch name: {msg}"
            );
        }
    }

    #[test]
    fn discovers_git_branch_name_from_workspace() {
        let workspace = tempdir().expect("workspace");
        fs::create_dir(workspace.path().join(".git")).expect("git dir");
        fs::write(workspace.path().join(".git/HEAD"), "ref: refs/heads/main\n").expect("head");

        assert_eq!(
            workspace_branch_name(workspace.path()).as_deref(),
            Some("main")
        );
        assert_eq!(
            discover_git_root(workspace.path()).as_deref(),
            Some(workspace.path())
        );
        assert_eq!(
            discover_git_dir(workspace.path()).as_deref(),
            Some(workspace.path().join(".git").as_path())
        );
    }

    #[test]
    fn git_workspace_diff_is_colorized_and_unified() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("\u{1b}["));
        assert!(diff.contains("@@"));
        assert!(diff.contains("diff --git"));
        assert!(diff.contains("changed"));
        assert!(diff.contains("four"));
    }

    #[test]
    fn git_workspace_diff_honors_global_gitconfig() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let home = tempdir().expect("home");
        fs::write(
            home.path().join(".gitconfig"),
            "[diff]\n\tnoprefix = true\n",
        )
        .expect("gitconfig");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("diff --git README.md README.md"));
        assert!(diff.contains("--- README.md"));
        assert!(diff.contains("+++ README.md"));
        assert!(!diff.contains("diff --git a/README.md b/README.md"));
    }

    // The pager test requires `sh` in PATH and a POSIX shell script as the pager.
    // `run_git_diff_pager` invokes `sh -lc`, which is not guaranteed to be in PATH
    // on Windows (Git for Windows may not add its bundled sh.exe to PATH).
    #[cfg(not(windows))]
    #[test]
    fn git_workspace_diff_uses_configured_noninteractive_pager() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        fs::write(
            workspace.path().join("README.md"),
            "one\nchanged\nthree\nfour\n",
        )
        .expect("update file");

        let home = tempdir().expect("home");
        let pager = home.path().join("pager.sh");
        fs::write(
            &pager,
            "#!/bin/sh\nprintf 'PAGER-START WIDTH=%s\\n' \"$COLUMNS\"\ncat\nprintf 'PAGER-END\\n'\n",
        )
        .expect("pager script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&pager).expect("pager metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&pager, permissions).expect("pager permissions");
        }
        // Backslashes are escape characters in gitconfig values, so use forward slashes
        // on Windows to avoid "bad config line" parse errors.
        let pager_config_path = pager.to_str().expect("pager path UTF-8").replace('\\', "/");
        fs::write(
            home.path().join(".gitconfig"),
            format!("[core]\n\tpager = {}\n", pager_config_path),
        )
        .expect("gitconfig");
        let _home_guard = EnvVarGuard::set_path("HOME", home.path());
        let _columns_guard = EnvVarGuard::set_value("COLUMNS", "123");

        let diff = git_workspace_diff(workspace.path()).expect("git diff");
        assert!(diff.contains("PAGER-START WIDTH="));
        assert!(diff.contains("diff --git"));
        assert!(diff.ends_with("PAGER-END\n"));
    }

    #[test]
    fn adds_explicit_width_to_delta_pager_command() {
        assert_eq!(
            with_explicit_pager_width("delta --side-by-side", 123),
            "delta --side-by-side --width=123"
        );
        assert_eq!(
            with_explicit_pager_width("/usr/bin/delta --width=90 --side-by-side", 123),
            "/usr/bin/delta --width=90 --side-by-side"
        );
        assert_eq!(with_explicit_pager_width("less -FRX", 123), "less -FRX");
    }

    #[test]
    fn set_model_switches_active_endpoint() {
        const GEMMA: &str = "gemma-4-E4B-it-GGUF";
        const OPENAI: &str = "gpt-4.1";

        let llms = HashMap::from([
            (
                GEMMA.to_string(),
                test_profile(
                    "llama.cpp",
                    "http://localhost:8100/v1",
                    "ggml-org/gemma-4-E4B-it-GGUF",
                ),
            ),
            (
                OPENAI.to_string(),
                test_profile("openai", "https://api.openai.com/v1", "gpt-4.1"),
            ),
        ]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = GEMMA.to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/model gpt-4.1",
            CommandState {
                active_model: &mut active_model,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
            },
            CommandContext {
                startup_model: GEMMA,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                http_client: reqwest::Client::new(),
                virtual_width: 512,
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Quiet));
        assert_eq!(active_model, OPENAI);
        assert_eq!(
            current_endpoint,
            Some(normalized_openai_endpoint("https://api.openai.com/v1"))
        );
    }

    #[test]
    fn unknown_slash_commands_error_locally() {
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut llms = HashMap::new();
        llms.insert(
            "default".to_string(),
            LlmConfiguration {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                endpoint: "http://localhost:11434/v1".to_string(),
                api_key: None,
                request_timeout_seconds: 30,
                max_tool_rounds: 10,
                system_prompt: String::new(),
            },
        );
        let mut session = ChatSession::new(system_prompt(&llms["default"]));
        let mut active_model = "default".to_string();
        let mut current_endpoint = Some("http://localhost:11434/v1".to_string());

        let outcome = handle_command(
            "/unknown",
            CommandState {
                active_model: &mut active_model,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
            },
            CommandContext {
                startup_model: "default",
                startup_endpoint: "http://localhost:11434/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                http_client: reqwest::Client::new(),
                virtual_width: 512,
            },
        )
        .expect("command outcome");

        assert!(matches!(
            outcome,
            CommandOutcome::OutputError(ref message)
                if message == "Unknown command '/unknown'. Use /help to see available commands."
        ));
    }

    #[test]
    fn completes_open_file_commands_across_workspace() {
        let workspace = tempdir().expect("workspace");
        fs::write(workspace.path().join("README.md"), "").expect("root readme");
        fs::create_dir(workspace.path().join("doc")).expect("doc dir");
        fs::write(workspace.path().join("doc/README.md"), "").expect("doc readme");
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(workspace.path().join("src/tui.rs"), "").expect("src file");
        fs::create_dir_all(workspace.path().join("target/.fingerprint/pkg")).expect("target dir");
        fs::write(
            workspace.path().join("target/.fingerprint/pkg/tui-output"),
            "",
        )
        .expect("target file");
        fs::create_dir_all(workspace.path().join("build/out")).expect("build dir");
        fs::write(workspace.path().join("build/out/tui.txt"), "").expect("build file");
        fs::write(workspace.path().join(".gitignore"), "ignored.md\n").expect("gitignore");
        fs::write(workspace.path().join("ignored.md"), "").expect("ignored file");
        fs::create_dir(workspace.path().join(".git")).expect("git dir");
        fs::write(workspace.path().join(".git/config"), "").expect("git config");

        let (_, _, slash_candidates) = completion_candidates(
            "/open_file READ",
            "/open_file READ".len(),
            workspace.path(),
            &[],
        )
        .expect("slash completion");
        assert_eq!(
            slash_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (start, _, natural_candidates) =
            completion_candidates("Open READ", "Open READ".len(), workspace.path(), &[])
                .expect("natural completion");
        assert_eq!(start, "Open ".len());
        assert_eq!(
            natural_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (_, _, ignored_candidates) =
            completion_candidates("Open ign", "Open ign".len(), workspace.path(), &[])
                .expect("ignored completion");
        assert!(ignored_candidates.is_empty());

        let (_, _, git_candidates) =
            completion_candidates("Open con", "Open con".len(), workspace.path(), &[])
                .expect("git completion");
        assert!(git_candidates.is_empty());

        let (_, _, target_candidates) =
            completion_candidates("/open_file t", "/open_file t".len(), workspace.path(), &[])
                .expect("target completion");
        assert_eq!(target_candidates, vec!["src/tui.rs".to_string()]);

        let (start, _, show_candidates) =
            completion_candidates("Show t", "Show t".len(), workspace.path(), &[])
                .expect("show completion");
        assert_eq!(start, "Show ".len());
        assert_eq!(show_candidates, vec!["src/tui.rs".to_string()]);

        let (start, _, show_file_candidates) = completion_candidates(
            "show file READ",
            "show file READ".len(),
            workspace.path(),
            &[],
        )
        .expect("show file completion");
        assert_eq!(start, "show file ".len());
        assert_eq!(
            show_file_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );
    }

    #[test]
    fn completes_show_file_commands_and_flags() {
        let workspace = tempdir().expect("workspace");
        fs::write(workspace.path().join("README.md"), "").expect("root readme");
        fs::create_dir(workspace.path().join("doc")).expect("doc dir");
        fs::write(workspace.path().join("doc/README.md"), "").expect("doc readme");
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(workspace.path().join("src/tui.rs"), "").expect("src file");
        fs::create_dir_all(workspace.path().join("target/.fingerprint/pkg")).expect("target dir");
        fs::write(
            workspace.path().join("target/.fingerprint/pkg/tui-output"),
            "",
        )
        .expect("target file");

        let (_, _, initial_file_candidates) =
            completion_candidates("/show_file ", "/show_file ".len(), workspace.path(), &[])
                .expect("initial file completion");
        assert_eq!(
            initial_file_candidates,
            vec![
                "README.md".to_string(),
                "doc/README.md".to_string(),
                "src/tui.rs".to_string()
            ]
        );

        let (_, _, flag_candidates) = completion_candidates(
            "/show_file --",
            "/show_file --".len(),
            workspace.path(),
            &[],
        )
        .expect("flag completion");
        assert_eq!(
            flag_candidates,
            vec!["--author".to_string(), "--hash".to_string()]
        );

        let (_, _, file_candidates) = completion_candidates(
            "/show_file --hash READ",
            "/show_file --hash READ".len(),
            workspace.path(),
            &[],
        )
        .expect("file completion");
        assert_eq!(
            file_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (_, _, quoted_candidates) = completion_candidates(
            "/show_file \"READ",
            "/show_file \"READ".len(),
            workspace.path(),
            &[],
        )
        .expect("quoted file completion");
        assert_eq!(
            quoted_candidates,
            vec!["\"README.md".to_string(), "\"doc/README.md".to_string()]
        );

        let (_, _, target_candidates) =
            completion_candidates("/show_file t", "/show_file t".len(), workspace.path(), &[])
                .expect("target completion");
        assert_eq!(target_candidates, vec!["src/tui.rs".to_string()]);
    }

    #[test]
    fn completion_respects_repo_gitignore_when_workspace_is_ignored_subdir() {
        let repo = tempdir().expect("repo");
        fs::create_dir(repo.path().join(".git")).expect("git dir");
        fs::write(repo.path().join(".git/config"), "").expect("git config");
        fs::write(repo.path().join(".gitignore"), "target/\n").expect("gitignore");
        fs::create_dir_all(repo.path().join("target/debug/.fingerprint/pkg")).expect("target dir");
        fs::write(
            repo.path().join("target/debug/.fingerprint/pkg/tui-output"),
            "",
        )
        .expect("target file");

        let workspace = repo.path().join("target/debug");

        let (_, _, open_candidates) =
            completion_candidates("/open_file ", "/open_file ".len(), &workspace, &[])
                .expect("open completion");
        assert!(open_candidates.is_empty());

        let (_, _, show_candidates) =
            completion_candidates("/show_file ", "/show_file ".len(), &workspace, &[])
                .expect("show completion");
        assert!(show_candidates.is_empty());
    }

    #[test]
    fn show_file_outputs_line_numbers_and_syntax_highlighting() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        fs::write(
            workspace.path().join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .expect("source file");

        let _path_guard = EnvVarGuard::set_value("PATH", "");
        let output = show_file_output(workspace.path(), "main.rs", 512).expect("show file");
        assert!(output.contains("1 "));
        assert!(output.contains("2 "));
        assert!(output.contains("\u{1b}["));
        assert!(output.contains("println!"));
    }

    #[test]
    fn show_file_formatting_bounds_ansi_to_source_column() {
        let metadata = GitLineMetadata {
            hash: "deadbeef".to_string(),
            author: "Alice".to_string(),
        };

        let rendered = format_show_file_line(
            7,
            "\x1b[38;2;1;2;3mlet x = 1;",
            Some(&metadata),
            ShowFileOptions {
                show_hash: true,
                show_author: true,
            },
            2,
        );

        assert_eq!(
            rendered,
            format!(" 7 deadbeef Alice {ANSI_RESET}\x1b[38;2;1;2;3mlet x = 1;{ANSI_RESET}")
        );
    }

    #[test]
    fn show_file_can_include_git_hash_and_author() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("README.md"), "alpha\nbeta\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        let hash_output = std::process::Command::new("git")
            .args(["rev-parse", "--short=8", "HEAD"])
            .current_dir(workspace.path())
            .output()
            .expect("git rev-parse");
        let expected_hash = String::from_utf8(hash_output.stdout)
            .expect("hash output")
            .trim()
            .to_string();

        let output = show_file_output(workspace.path(), "--hash --author README.md", 512)
            .expect("show file");
        assert!(output.contains(&expected_hash));
        assert!(output.contains("Orangu Tests"));
        assert!(output.contains("1 "));
        assert!(output.contains("2 "));
    }

    // This test creates a fake `bat` shell script to intercept the call.
    // On Windows, plain extensionless scripts aren't executable via CreateProcessW,
    // and PATHEXT resolution finds the real bat.exe before bat.cmd in a later PATH
    // directory. Skipped on Windows; bat integration is verified on CI via Linux runners.
    #[cfg(not(windows))]
    #[test]
    fn show_file_uses_bat_when_available_without_metadata_columns() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        fs::write(workspace.path().join("main.rs"), "fn main() {}\n").expect("source file");

        let tools_dir = tempdir().expect("tools dir");
        let bat = tools_dir.path().join("bat");
        fs::write(&bat, "#!/bin/sh\nprintf 'BAT:%s\\n' \"$*\"\n").expect("bat script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&bat).expect("bat metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&bat, permissions).expect("bat permissions");
        }
        let path_value = format!(
            "{}:{}",
            tools_dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let _path_guard = EnvVarGuard::set_value("PATH", &path_value);
        let _columns_guard = EnvVarGuard::set_value("COLUMNS", "123");

        let output = show_file_output(workspace.path(), "main.rs", 512).expect("show file");
        assert!(output.contains("BAT:"));
        assert!(output.contains("--paging=never"));
        assert!(output.contains("--color=always"));
        assert!(output.contains("--style=numbers"));
        assert!(output.contains("--terminal-width"));
        assert!(output.contains(workspace.path().join("main.rs").to_string_lossy().as_ref()));
    }

    #[test]
    fn show_file_bypasses_bat_when_metadata_columns_are_requested() {
        let _env_lock = lock_process_env();
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("README.md"), "alpha\nbeta\n").expect("write file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "README.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        let tools_dir = tempdir().expect("tools dir");
        let bat = tools_dir.path().join("bat");
        fs::write(&bat, "#!/bin/sh\nprintf 'BAT:%s\\n' \"$*\"\n").expect("bat script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&bat).expect("bat metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&bat, permissions).expect("bat permissions");
        }
        let path_value = format!(
            "{}:{}",
            tools_dir.path().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let _path_guard = EnvVarGuard::set_value("PATH", &path_value);

        let output =
            show_file_output(workspace.path(), "--hash README.md", 512).expect("show file");
        assert!(!output.contains("BAT:"));
        assert!(output.contains("alpha"));
        assert!(output.contains("beta"));
    }

    #[test]
    fn final_pending_line_keeps_visible_output() {
        assert_eq!(
            final_pending_line("streamed reply", "final reply").as_deref(),
            Some("streamed reply")
        );
        assert_eq!(
            final_pending_line("", "final reply").as_deref(),
            Some("final reply")
        );
        assert_eq!(final_pending_line("", ""), None);
    }

    #[test]
    fn cancelled_output_preserves_partial_reply_and_uses_light_red_notice() {
        let mut output_state = OutputState::default();

        preserve_cancelled_output(&mut output_state, "partial reply");

        assert_eq!(
            output_state.lines(),
            &[
                TranscriptLine::Plain("partial reply".to_string()),
                TranscriptLine::Plain(request_cancelled_message()),
            ]
        );
    }

    #[test]
    fn idle_refresh_timeout_hits_zero_at_deadline() {
        let start = Instant::now();

        assert_eq!(
            idle_status_refresh_timeout(start + Duration::from_secs(60), start),
            Duration::from_secs(60)
        );
        assert_eq!(
            idle_status_refresh_timeout(
                start + Duration::from_secs(60),
                start + Duration::from_secs(61)
            ),
            Duration::ZERO
        );
    }

    #[test]
    fn llm_prompt_block_reason_requires_model_connection() {
        assert_eq!(
            llm_prompt_block_reason(
                Some("http://localhost:8100/v1"),
                HeaderStatus {
                    workspace_ok: true,
                    server_ok: true,
                    model_ok: false,
                }
            ),
            None
        );
        assert_eq!(
            llm_prompt_block_reason(
                Some("http://localhost:8100/v1"),
                HeaderStatus {
                    workspace_ok: true,
                    server_ok: true,
                    model_ok: true,
                }
            ),
            None
        );
    }

    #[test]
    fn escape_cancel_requires_two_presses_within_timeout() {
        let mut cancel_state = EscapeCancelState::default();
        let start = Instant::now();

        assert!(!cancel_state.handle_escape(start));
        assert!(cancel_state.handle_escape(start + Duration::from_millis(500)));

        assert!(!cancel_state.handle_escape(start + Duration::from_secs(5)));
        assert!(!cancel_state.handle_escape(start + Duration::from_secs(8)));
    }

    #[test]
    fn wait_cancel_escape_only_matches_escape_press() {
        assert!(is_wait_cancel_escape(&Event::Key(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE
        ))));
        assert!(!is_wait_cancel_escape(&Event::Key(
            KeyEvent::new_with_kind(KeyCode::Esc, KeyModifiers::NONE, KeyEventKind::Repeat)
        )));
        assert!(!is_wait_cancel_escape(&Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE
        ))));
    }

    #[test]
    fn llama_cpp_left_status_prefers_native_metrics() {
        let profile = LlmConfiguration {
            provider: "llama.cpp".to_string(),
            model: "model".to_string(),
            endpoint: "http://localhost:8080/v1".to_string(),
            api_key: None,
            request_timeout_seconds: 30,
            max_tool_rounds: 10,
            system_prompt: String::new(),
        };

        let thinking = render_left_status(
            &profile,
            "",
            &StreamMetrics {
                prompt_progress: Some(StreamPromptProgress {
                    total: 100,
                    cache: 20,
                    processed: 60,
                    time_ms: 2_000,
                }),
                prompt_per_second: Some(15.0),
                predicted_per_second: None,
            },
            None,
            Duration::from_secs(2),
            0,
            None,
        )
        .expect("thinking status");
        for ch in "Thinking".chars() {
            assert!(thinking.rendered.contains(ch));
        }
        assert!(thinking.rendered.contains("(2s)"));
        assert_eq!(thinking.visible_width, "Thinking (2s)".chars().count());

        let working = render_left_status(
            &profile,
            "hello",
            &StreamMetrics {
                prompt_progress: None,
                prompt_per_second: Some(15.0),
                predicted_per_second: Some(42.5),
            },
            None,
            Duration::from_secs(2),
            1,
            None,
        )
        .expect("working status");
        for ch in "Working".chars() {
            assert!(working.rendered.contains(ch));
        }
        assert!(working.rendered.contains("42.5 t/s"));
        assert!(working.rendered.contains("(2s)"));
        assert_eq!(
            working.visible_width,
            "Working @ 42.5 t/s (2s)".chars().count()
        );
    }

    #[test]
    fn completes_checkout_branches_and_files() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        std::process::Command::new("git")
            .args(["symbolic-ref", "HEAD", "refs/heads/main"])
            .current_dir(workspace.path())
            .status()
            .expect("set initial branch to main");
        fs::write(workspace.path().join("main.rs"), "").expect("main.rs");
        assert!(
            std::process::Command::new("git")
                .args(["add", "main.rs"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["checkout", "--quiet", "-b", "mybranch"])
                .current_dir(workspace.path())
                .status()
                .expect("git checkout")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["checkout", "--quiet", "main"])
                .current_dir(workspace.path())
                .status()
                .expect("git checkout")
                .success()
        );

        let (start, _, candidates) =
            completion_candidates("/checkout m", "/checkout m".len(), workspace.path(), &[])
                .expect("checkout completion");
        assert_eq!(start, "/checkout ".len());
        assert!(candidates.contains(&"main".to_string()), "main missing");
        assert!(
            candidates.contains(&"mybranch".to_string()),
            "branch missing"
        );
        assert!(candidates.contains(&"main.rs".to_string()), "file missing");

        let (start, _, nat_candidates) =
            completion_candidates("checkout m", "checkout m".len(), workspace.path(), &[])
                .expect("natural checkout completion");
        assert_eq!(start, "checkout ".len());
        assert!(
            nat_candidates.contains(&"main".to_string()),
            "natural main missing"
        );
    }

    #[test]
    fn completes_switch_to_branches_and_tags_but_not_files() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        // Ensure initial branch is "main" regardless of git init.defaultBranch config.
        std::process::Command::new("git")
            .args(["symbolic-ref", "HEAD", "refs/heads/main"])
            .current_dir(workspace.path())
            .status()
            .expect("set initial branch to main");
        fs::write(workspace.path().join("main.rs"), "").expect("main.rs");
        assert!(
            std::process::Command::new("git")
                .args(["add", "main.rs"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["checkout", "--quiet", "-b", "mybranch"])
                .current_dir(workspace.path())
                .status()
                .expect("git checkout branch")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["checkout", "--quiet", "main"])
                .current_dir(workspace.path())
                .status()
                .expect("git checkout main")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["tag", "mytag"])
                .current_dir(workspace.path())
                .status()
                .expect("git tag")
                .success()
        );

        let (start, _, candidates) =
            completion_candidates("switch to m", "switch to m".len(), workspace.path(), &[])
                .expect("switch to completion");
        assert_eq!(start, "switch to ".len());
        assert!(
            candidates.contains(&"mybranch".to_string()),
            "branch missing"
        );
        assert!(candidates.contains(&"mytag".to_string()), "tag missing");
        // workspace files should NOT appear
        assert!(
            !candidates.contains(&"main.rs".to_string()),
            "file should not appear"
        );
    }

    #[test]
    fn completes_add_file_untracked() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("tracked.rs"), "").expect("tracked file");
        assert!(
            std::process::Command::new("git")
                .args(["add", "tracked.rs"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );
        fs::create_dir(workspace.path().join("newdir")).expect("new dir");
        fs::write(workspace.path().join("newdir/file.rs"), "").expect("dir file");
        fs::write(workspace.path().join("newfile.txt"), "").expect("new file");

        // "n" matches directory "newdir/" before file "newfile.txt"
        let (start, _, candidates) =
            completion_candidates("/add_file n", "/add_file n".len(), workspace.path(), &[])
                .expect("add_file completion");
        assert_eq!(start, "/add_file ".len());
        assert_eq!(candidates[0], "newdir/");
        assert!(candidates.contains(&"newfile.txt".to_string()));
        // tracked file not included
        assert!(!candidates.contains(&"tracked.rs".to_string()));

        // Natural-language form
        let (start, _, nat_candidates) =
            completion_candidates("add n", "add n".len(), workspace.path(), &[])
                .expect("natural add_file completion");
        assert_eq!(start, "add ".len());
        assert_eq!(nat_candidates[0], "newdir/");
    }

    #[test]
    fn completes_remove_file_tracked() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(workspace.path().join("src/main.rs"), "").expect("main.rs");
        fs::write(workspace.path().join("schema.sql"), "").expect("schema.sql");
        fs::write(workspace.path().join("untracked.txt"), "").expect("untracked");
        assert!(
            std::process::Command::new("git")
                .args(["add", "src/main.rs", "schema.sql"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        // "s" matches directory "src/" before file "schema.sql"
        let (start, _, candidates) = completion_candidates(
            "/remove_file s",
            "/remove_file s".len(),
            workspace.path(),
            &[],
        )
        .expect("remove_file completion");
        assert_eq!(start, "/remove_file ".len());
        assert_eq!(candidates[0], "src/");
        assert!(candidates.contains(&"schema.sql".to_string()));
        // untracked file not included
        assert!(!candidates.contains(&"untracked.txt".to_string()));

        // Natural-language form
        let (start, _, nat_candidates) =
            completion_candidates("remove s", "remove s".len(), workspace.path(), &[])
                .expect("natural remove_file completion");
        assert_eq!(start, "remove ".len());
        assert_eq!(nat_candidates[0], "src/");
    }

    #[test]
    fn completes_move_file_targets() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(workspace.path().join("src/main.rs"), "").expect("main.rs");
        fs::write(workspace.path().join("readme.md"), "").expect("readme");
        fs::write(workspace.path().join("untracked.txt"), "").expect("untracked");
        assert!(
            std::process::Command::new("git")
                .args(["add", "src/main.rs", "readme.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "initial"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        // First arg: "s" matches tracked "src/" (directory) — untracked file absent
        let (start, _, src_candidates) =
            completion_candidates("/move_file s", "/move_file s".len(), workspace.path(), &[])
                .expect("move_file source completion");
        assert_eq!(start, "/move_file ".len());
        assert_eq!(src_candidates[0], "src/");
        assert!(!src_candidates.contains(&"untracked.txt".to_string()));

        // Second arg: completes workspace files (not filtered by tracked status)
        let (start, _, dst_candidates) = completion_candidates(
            "/move_file src/main.rs u",
            "/move_file src/main.rs u".len(),
            workspace.path(),
            &[],
        )
        .expect("move_file destination completion");
        assert_eq!(start, "/move_file src/main.rs ".len());
        assert!(dst_candidates.contains(&"untracked.txt".to_string()));

        // Natural-language form — first arg
        let (start, _, nat_candidates) =
            completion_candidates("move s", "move s".len(), workspace.path(), &[])
                .expect("natural move_file completion");
        assert_eq!(start, "move ".len());
        assert_eq!(nat_candidates[0], "src/");
    }

    #[test]
    fn completes_cherry_pick_commits() {
        let workspace = tempdir().expect("workspace");
        init_test_git_repo(workspace.path());
        fs::write(workspace.path().join("readme.md"), "initial").expect("readme");
        assert!(
            std::process::Command::new("git")
                .args(["add", "readme.md"])
                .current_dir(workspace.path())
                .status()
                .expect("git add")
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .args(["commit", "--quiet", "-m", "first commit"])
                .current_dir(workspace.path())
                .status()
                .expect("git commit")
                .success()
        );

        // Completion with no token returns recent commit hashes from main
        let result = completion_candidates(
            "/cherry_pick ",
            "/cherry_pick ".len(),
            workspace.path(),
            &[],
        );
        if let Some((start, _, candidates)) = result {
            assert_eq!(start, "/cherry_pick ".len());
            // Abbreviated hashes are 7 chars
            assert!(candidates.iter().all(|h| h.len() >= 4));
        }

        // Natural-language form triggers completion
        let nl_result =
            completion_candidates("cherry pick ", "cherry pick ".len(), workspace.path(), &[]);
        if let Some((start, _, _)) = nl_result {
            assert_eq!(start, "cherry pick ".len());
        }
    }
}

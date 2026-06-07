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
mod init;
mod input;
mod quotes;
mod render;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::{
    event::{
        self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
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
        FEEDBACK_ERR, FEEDBACK_OK, ReviewCommentEditor, ReviewEntry, ReviewFeedbackView,
        ReviewScreenArgs, ReviewStatus, ScreenRenderArgs, StatusFragment, render_review_screen,
        render_screen, render_thinking_status, render_tool_running_status, render_working_status,
        review_pane_body_height, terminal_height, terminal_width,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    io::Write,
    path::{Component, Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
};
use tiktoken_rs::cl100k_base;
use uuid::Uuid;

use anyhow::Error;
use commands::ReviewLaunch;
use commands::{
    BranchSubcommand, CommandContext, CommandOutcome, CommandState, LocalCommand, LocalError,
    StashSubcommand, add_file_usage_message, amend_usage_message, cherry_pick_usage_message,
    close_usage_message, comment_usage_message, commit_usage_message, connect_usage_message,
    grep_usage_message, merge_usage_message, model_usage_message, move_file_usage_message,
    open_file_usage_message, parse_local_command, pull_usage_message, remove_file_usage_message,
    restore_usage_message, server_usage_message, sorted_model_names, system_prompt,
};
use git::{
    Forge, add_file_output, amend_output, branch_create_output, branch_delete_output,
    branch_list_all_output, branch_list_output, branch_rename_output, cherry_pick_output,
    close_output, collect_review_diff, comment_output, commit_output, create_pull_request_output,
    discover_git_root, git_checkout, git_diff_against_branch, git_workspace_diff, grep_output,
    init_repo_output, list_workspace_files_tree, log_output, merge_output, move_file_output,
    open_in_editor, pull_request_output, push_output, rebase_output, remove_file_output,
    restore_output, squash_output, stash_drop_output, stash_list_output, stash_output,
    stash_pop_output, status_output, sync_default_branch, workspace_branch_name,
};
use input::{
    EscapeCancelState, IDLE_STATUS_REFRESH_INTERVAL, InputContext, InputResult, InputState,
    InterruptState, OutputState, RenderContext, ScreenState, StreamRenderState, ViewportState,
    WaitContext, WaitResult, handle_input_event, read_input,
};
use render::{format_tools, render_markdown_for_console, show_file_output};

const CLEAR_TERMINAL_SEQUENCE: &str = "\x1b[2J\x1b[H";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const TERMINAL_TITLE: &str = "orangu";
const WAIT_LOOP_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
const THINKING_FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(120);
const SESSIONS_DIRECTORY: &str = ".orangu/sessions";
/// Scratch directory used by `/restart` to stage a runnable copy of the binary
/// when the original on-disk path has been replaced (e.g. rebuilt while
/// running). Cleared on every startup.
const RESTART_DIRECTORY: &str = ".orangu/last";

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    config: Option<PathBuf>,
    #[arg(short, long)]
    workspace: Option<PathBuf>,
    #[arg(short, long)]
    resume: Option<String>,
    /// Interactively create ~/.orangu/orangu.conf and exit.
    #[arg(short, long)]
    init: bool,
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
    if args.init {
        return init::run_init().await;
    }
    let config_path = match args.config.or_else(default_client_config_path) {
        Some(path) => path,
        None => {
            return Err(anyhow!(
                "Missing config file; pass --config or add ./orangu.conf or ~/.orangu/orangu.conf"
            ));
        }
    };
    let config = load_client_configuration(&config_path)?;
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

    // Remove any binary staged by a previous `/restart`; it is only needed
    // across the exec handoff and must not accumulate.
    clear_restart_dir();

    let status_http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;

    // Server section names, used for `/server` completion.
    let server_names = sorted_model_names(&config.llms);
    let startup_model = config.default_server.clone();
    let startup_profile = config
        .llms
        .get(&startup_model)
        .ok_or_else(|| anyhow!("missing configured server {}", startup_model))?;
    let startup_endpoint = startup_profile.endpoint.clone();
    // The wire model id sent to the active server, initialised from the active
    // server's resolved model and changed at runtime by `/model`.
    let mut active_model_id = startup_profile.model.clone();
    let mut active_model = startup_model.clone();
    let mut session = ChatSession::new(system_prompt(
        config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured server {}", active_model))?,
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
    let mut restart_requested = false;
    // When set, the post-loop exec resumes this session instead of the current
    // one — used by `/session <UUID>` to switch sessions in place.
    let mut resume_override: Option<String> = None;
    let mut startup_notice_until: Option<std::time::Instant> =
        if is_resumed && args.resume.is_none() {
            Some(std::time::Instant::now() + std::time::Duration::from_secs(5))
        } else {
            None
        };
    if workspace_created {
        output_state.push_text(&format!("Created workspace {}", workspace.display()));
    }

    // If the active server isn't serving the configured model at startup, switch
    // to a model it does advertise.
    if let Some(active_profile) = config.llms.get(&active_model)
        && let Some((old_model, new_model)) = try_startup_model_switch(
            &status_http_client,
            active_profile,
            &mut active_model_id,
            current_endpoint.as_deref(),
        )
        .await
    {
        output_state.push_text(&format!("Switched model from {old_model} to {new_model}"));
    }

    // In a Git repository, fast-forward the local default branch to origin on
    // startup. Run it in the background so it never blocks the UI; its progress
    // and result are shown on the left of the status bar.
    let forge = Forge::from_platform(&config.platform);
    let mut sync_handle = discover_git_root(tools.workspace()).map(|_| {
        let sync_workspace = tools.workspace().to_path_buf();
        tokio::task::spawn_blocking(move || sync_default_branch(&sync_workspace, forge))
    });
    let mut sync_notice: Option<(String, std::time::Instant)> = None;

    loop {
        let prompt_branch = workspace_branch_name(tools.workspace());
        let active_profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured server {}", active_model))?;
        let (header_status, server_models) = probe_header_status(
            &status_http_client,
            tools.workspace(),
            &active_model_id,
            active_profile,
            current_endpoint.as_deref(),
        )
        .await;
        // Models advertised by the selected server, used for `/model` completion.
        let available_models = server_models;
        let render = RenderContext {
            current_model: &active_model_id,
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
        // Collect the startup branch-sync result once its task finishes.
        if sync_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = sync_handle.take()
        {
            let notice = match handle.await {
                Ok(Ok(Some(message))) => Some(message),
                Ok(Err(err)) => Some(format!("Sync failed: {err}")),
                _ => None,
            };
            if let Some(message) = notice {
                sync_notice = Some((
                    message,
                    std::time::Instant::now() + std::time::Duration::from_secs(5),
                ));
            }
        }

        let resume_left_status = startup_notice_until
            .filter(|&deadline| std::time::Instant::now() < deadline)
            .map(|_| StatusFragment::plain(format!("Resuming session {session_id}")));
        // The branch sync takes priority on the left of the status bar while it
        // runs and for a few seconds after it completes.
        let left_status = if sync_handle.is_some() {
            Some(StatusFragment::plain("Syncing with origin…".to_string()))
        } else {
            sync_notice
                .as_ref()
                .filter(|(_, deadline)| std::time::Instant::now() < *deadline)
                .map(|(message, _)| StatusFragment::plain(message.clone()))
                .or(resume_left_status)
        };
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
                ghost_index: input_state.ghost_index,
            },
        );
        std::io::stdout().flush()?;

        // While the startup sync is running or its result is still on the status
        // bar, refresh more often so the status clears promptly when it is done.
        let sync_active = sync_handle.is_some()
            || sync_notice
                .as_ref()
                .is_some_and(|(_, deadline)| std::time::Instant::now() < *deadline);
        let max_idle = if sync_active {
            std::time::Duration::from_millis(500)
        } else {
            IDLE_STATUS_REFRESH_INTERVAL
        };

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
                    server_names: &server_names,
                    available_models: &available_models,
                    render,
                },
                print_screen,
                max_idle,
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
                ghost_index: input_state.ghost_index,
            },
        );
        std::io::stdout().flush()?;

        let mut detect_model = false;
        let command_outcome = handle_command(
            &next_input,
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut detect_model,
            },
            CommandContext {
                startup_model: &startup_model,
                startup_endpoint: &startup_endpoint,
                llms: &config.llms,
                tools: &tools,
                workspace: &workspace,
                usage_stats: &usage_stats,
                available_models: &available_models,
                virtual_width: viewport.virtual_width,
                auto_rebase: config.auto_rebase,
                auto_squash: config.auto_squash,
                terminal: &config.terminal,
                forge,
            },
        )?;
        // When `/server` (or `/reload`) selects a server, auto-detect an
        // available model on it — exactly like the startup model switch. This
        // runs even when re-selecting the server we are already on.
        if detect_model
            && let Some(profile) = config.llms.get(&active_model)
            && let Some((old_model, new_model)) = try_startup_model_switch(
                &status_http_client,
                profile,
                &mut active_model_id,
                current_endpoint.as_deref(),
            )
            .await
        {
            output_state.push_text(&format!("Switched model from {old_model} to {new_model}"));
        }
        match command_outcome {
            CommandOutcome::Quit => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            CommandOutcome::Restart => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                break;
            }
            CommandOutcome::SwitchSession(target) => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                resume_override = Some(target);
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
                    current_model: &active_model_id,
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
                        server_names: &server_names,
                        available_models: &available_models,
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
            CommandOutcome::Streaming(f) => {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                let handle = tokio::task::spawn_blocking(move || f(tx));
                // Recreate render here — handle_command's mutable borrows have ended.
                let blocking_render = RenderContext {
                    current_model: &active_model_id,
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
                let result = wait_for_streaming_command(
                    WaitContext {
                        render: blocking_render,
                        history: &mut history,
                        history_path: &session_hist_path,
                        server_names: &server_names,
                        available_models: &available_models,
                        interrupt_state: &mut interrupt_state,
                        output_state: &mut output_state,
                        input_state: &mut input_state,
                        pending_commands: &mut pending_commands,
                        thinking_quote: None,
                        viewport: &mut viewport,
                    },
                    handle,
                    &mut rx,
                )
                .await?;
                match result {
                    Ok(()) => {
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
            CommandOutcome::Review(launch) => {
                let mut review = ReviewState::new(launch);
                loop {
                    let chrome = ReviewChrome {
                        current_model: &active_model_id,
                        prompt_branch: prompt_branch.as_deref(),
                        pending_count: pending_commands.len(),
                    };
                    match run_review_mode(&mut review, &mut viewport, &mut input_state, chrome)? {
                        ReviewSignal::Exit => break,
                        ReviewSignal::OpenFile { path } => {
                            if let Err(err) = open_in_editor(&workspace, &path, &config.terminal) {
                                review.feedback = Some(FeedbackWindow {
                                    title: format!("Open: {path}"),
                                    question: None,
                                    lines: vec![format!("Error: {err:#}")],
                                    scroll: 0,
                                    x_offset: 0,
                                });
                            }
                        }
                        ReviewSignal::RequestReview {
                            path,
                            patch,
                            request,
                        } => {
                            // Keep the typed request in the input window until the
                            // review succeeds, so a cancel or error can be retried.
                            let title = format!("Review: {path}");
                            // A typed request is echoed in the popup; a plain
                            // Alt+o (empty input) has no question to echo.
                            let question = (!request.trim().is_empty()).then(|| request.clone());
                            let Some(endpoint) = current_endpoint.as_deref() else {
                                review.feedback = Some(FeedbackWindow {
                                    title,
                                    question,
                                    lines: vec![
                                        "Error: Not connected to an LLM server".to_string(),
                                    ],
                                    scroll: 0,
                                    x_offset: 0,
                                });
                                continue;
                            };
                            let Some(profile) = config.llms.get(&active_model) else {
                                review.feedback = Some(FeedbackWindow {
                                    title,
                                    question,
                                    lines: vec![format!("Error: unknown server '{active_model}'")],
                                    scroll: 0,
                                    x_offset: 0,
                                });
                                continue;
                            };
                            let mut prompt_profile = profile.clone();
                            prompt_profile.endpoint = endpoint.to_string();
                            prompt_profile.model = active_model_id.clone();
                            let prompt = build_review_prompt(&path, &request, &patch);
                            let llm_start = std::time::Instant::now();
                            let tool_before = tools.total_tool_duration();
                            let result = run_review_request(
                                &mut session,
                                &prompt,
                                &prompt_profile,
                                &tools,
                                &review,
                                &input_state,
                                &mut viewport,
                                chrome,
                            )
                            .await?;
                            let lines = match result {
                                ReviewRequestOutcome::Exit => break,
                                ReviewRequestOutcome::Cancelled => continue,
                                ReviewRequestOutcome::Completed(Ok(text)) => {
                                    let tool_delta =
                                        tools.total_tool_duration().saturating_sub(tool_before);
                                    usage_stats.record_response(
                                        llm_start.elapsed(),
                                        &text,
                                        tool_delta,
                                    );
                                    // The request succeeded; clear the input window.
                                    input_state.clear();
                                    render::render_markdown_for_console(&text)
                                        .lines()
                                        .map(str::to_string)
                                        .collect()
                                }
                                ReviewRequestOutcome::Completed(Err(err)) => {
                                    vec![format!("Error: {err:#}")]
                                }
                            };
                            review.feedback = Some(FeedbackWindow {
                                title,
                                question,
                                lines,
                                scroll: 0,
                                x_offset: 0,
                            });
                        }
                    }
                }
                // On exit, print the per-file status and comments to the output
                // window, and copy the comments to the system clipboard.
                let (lines, clipboard) =
                    review_exit_output(&review.files, &review.comments, &review.general_notes);
                for line in &lines {
                    output_state.push_text(line);
                }
                if let Some(text) = clipboard
                    && let Err(err) = copy_to_clipboard(&text)
                {
                    output_state.push_text(&format!(
                        "Could not copy review comments to the clipboard: {err}"
                    ));
                }
                // The modal view overwrote the screen; the next loop iteration
                // redraws the normal interface from the top.
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Unhandled => {}
        }

        let profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("unknown server '{active_model}'"))?;
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
        prompt_profile.model = active_model_id.clone();
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
                    current_model: &active_model_id,
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
                server_names: &server_names,
                available_models: &available_models,
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
                let tool_delta = tools.total_tool_duration().saturating_sub(tool_time_before);
                usage_stats.record_response(llm_start.elapsed(), &partial_output, tool_delta);
                preserve_cancelled_output(&mut output_state, &partial_output);
            }
            Ok(WaitResult::Failed { partial, error }) => {
                let tool_delta = tools.total_tool_duration().saturating_sub(tool_time_before);
                usage_stats.record_response(llm_start.elapsed(), &partial, tool_delta);
                output_state.push_text(&format!("Error: {error:#}"));
                if config.feedback {
                    output_state.push_text(FEEDBACK_ERR);
                }
            }
            Ok(WaitResult::Quit) => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            Err(err) => {
                let tool_delta = tools.total_tool_duration().saturating_sub(tool_time_before);
                usage_stats.record_elapsed(llm_start.elapsed(), tool_delta);
                output_state.push_text(&format!("Error: {err:#}"));
                if config.feedback {
                    output_state.push_text(FEEDBACK_ERR);
                }
            }
        }
        output_state.reset_scroll();
    }

    drop(_terminal_ui_guard);

    if restart_requested {
        let resume_target = resume_override.as_deref().unwrap_or(&session_id);
        // When switching to a different session, follow it to the workspace it
        // was started in so the resumed client (and its banner) reflect that
        // session's project rather than the current directory.
        let resume_workspace = match &resume_override {
            Some(uuid) => session_dir_path(uuid)
                .ok()
                .and_then(|dir| load_session_metadata(&dir.join("metadata")).ok().flatten())
                .map(|meta| meta.workspace)
                .filter(|ws| !ws.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| workspace.clone()),
            None => workspace.clone(),
        };
        let exe = restart_executable_path()?;
        let mut command = std::process::Command::new(&exe);
        command
            .arg("--config")
            .arg(&config_path)
            .arg("--workspace")
            .arg(&resume_workspace)
            .arg("--resume")
            .arg(resume_target);
        // Replace the current process image so the restarted client keeps the
        // controlling terminal as the foreground process. Spawning a child and
        // exiting instead would hand the terminal back to the launching shell,
        // leaving the new process in the background where terminal I/O fails
        // with EIO. The PID is preserved, mirroring how `exec $SHELL` restarts.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            return Err(command.exec().into());
        }
        #[cfg(not(unix))]
        {
            let status = command.status()?;
            std::process::exit(status.code().unwrap_or(0));
        }
    }

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

    /// Record the time spent on a turn, splitting it into tool time and LLM
    /// time. Called for every outcome — success, cancellation, and failure —
    /// so the LLM time before a failure or cancellation is still counted.
    fn record_elapsed(
        &mut self,
        total_duration: std::time::Duration,
        tool_duration: std::time::Duration,
    ) {
        self.total_tool_duration += tool_duration;
        self.total_llm_duration += total_duration.saturating_sub(tool_duration);
    }

    fn record_response(
        &mut self,
        total_duration: std::time::Duration,
        response: &str,
        tool_duration: std::time::Duration,
    ) {
        self.record_elapsed(total_duration, tool_duration);
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
            "Application time : {}\nLLM time         : {}\nTool time        : {}\nTotal tokens     : {}\nAvg tokens/sec   : {:.1}\nSession          : {}\nPID              : {}",
            format_duration(app_elapsed),
            format_duration(self.total_llm_duration),
            format_duration(self.total_tool_duration),
            self.total_tokens,
            avg_tps,
            self.session_id,
            std::process::id(),
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
        active_model_id,
        current_endpoint,
        session,
        detect_model,
    } = state;
    let CommandContext {
        startup_model,
        startup_endpoint,
        llms,
        tools,
        workspace,
        usage_stats,
        available_models,
        virtual_width,
        auto_rebase,
        auto_squash,
        terminal,
        forge,
    } = context;

    match command {
        LocalCommand::Help => Ok(CommandOutcome::Output(orangu::tui::help_text().to_string())),
        LocalCommand::ConnectDefault => {
            let endpoint = llms
                .get(active_model)
                .ok_or_else(|| anyhow!("unknown server '{active_model}'"))?
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
            let profile = llms
                .get(startup_model)
                .ok_or_else(|| anyhow!("unknown server '{startup_model}'"))?;
            *active_model_id = profile.model.clone();
            session.clear(system_prompt(profile));
            *detect_model = true;
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::Restart => Ok(CommandOutcome::Restart),
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
        LocalCommand::ModelInfo => {
            // The active model is marked active (green dot); every other model
            // the server advertises is listed as inactive (red dot).
            let mut lines = vec![format!("{FEEDBACK_OK} {active_model_id}")];
            for model in available_models {
                if model != active_model_id {
                    lines.push(format!("{FEEDBACK_ERR} {model}"));
                }
            }
            Ok(CommandOutcome::Output(lines.join("\n")))
        }
        LocalCommand::SetModelId(name) => {
            if name.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    model_usage_message().to_string(),
                ));
            }
            *active_model_id = name.to_string();
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::ServerInfo => {
            // The active server is marked active (green dot); every other
            // configured server is listed as inactive (red dot).
            let lines: Vec<String> = sorted_model_names(llms)
                .into_iter()
                .map(|name| {
                    if name == *active_model {
                        format!("{FEEDBACK_OK} {name}")
                    } else {
                        format!("{FEEDBACK_ERR} {name}")
                    }
                })
                .collect();
            Ok(CommandOutcome::Output(lines.join("\n")))
        }
        LocalCommand::SetServer(name) => {
            if name.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    server_usage_message().to_string(),
                ));
            }
            if !llms.contains_key(name) {
                return Ok(CommandOutcome::OutputError(format!(
                    "Unknown server '{name}'. Available: {}",
                    sorted_model_names(llms).join(", ")
                )));
            }
            let profile = &llms[name];
            let endpoint = orangu::llm::normalized_openai_endpoint(&profile.endpoint);
            *active_model = name.to_string();
            *active_model_id = profile.model.clone();
            *current_endpoint = Some(endpoint);
            session.set_system_prompt(system_prompt(profile));
            // Re-run the startup-style model detection against the selected
            // server, even when it is the server we were already on.
            *detect_model = true;
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
        LocalCommand::Review => match collect_review_diff(workspace) {
            Ok(review) if review.files.is_empty() => Ok(CommandOutcome::Output(format!(
                "No changes to review against {}.",
                review.base_label
            ))),
            Ok(review) => {
                let files = review
                    .files
                    .into_iter()
                    .map(|file| ReviewEntry {
                        path: file.path,
                        status: ReviewStatus::Unreviewed,
                        diff_lines: file.lines,
                        patch: file.patch,
                    })
                    .collect();
                Ok(CommandOutcome::Review(ReviewLaunch { files }))
            }
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Status => match status_output(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Grep(None) => Ok(CommandOutcome::OutputError(
            grep_usage_message().to_string(),
        )),
        LocalCommand::Grep(Some(pattern)) => match grep_output(workspace, &pattern) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Log(count) => match log_output(workspace, count) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Pull(None) => Ok(CommandOutcome::OutputError(
            pull_usage_message().to_string(),
        )),
        LocalCommand::Pull(Some(pr_number)) => {
            match pull_request_output(workspace, pr_number, forge) {
                Ok(Some(advice)) => Ok(CommandOutcome::Output(advice)),
                Ok(None) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Comment(None) => Ok(CommandOutcome::OutputError(
            comment_usage_message().to_string(),
        )),
        LocalCommand::Comment(Some((issue_number, body))) => {
            match comment_output(workspace, issue_number, &body, forge) {
                Ok(_) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Close(None) => Ok(CommandOutcome::OutputError(
            close_usage_message().to_string(),
        )),
        LocalCommand::Close(Some(target)) => match close_output(workspace, &target, forge) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::CreatePullRequest => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Blocking(Box::new(move || {
                create_pull_request_output(&ws, auto_rebase, auto_squash, forge)
            })))
        }
        LocalCommand::Rebase => match rebase_output(workspace, forge) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Merge(None) => Ok(CommandOutcome::OutputError(
            merge_usage_message().to_string(),
        )),
        LocalCommand::Merge(Some(branch)) => match merge_output(workspace, &branch, forge) {
            Ok(_) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Branch(sub) => match sub {
            BranchSubcommand::List => match branch_list_output(workspace) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            },
            BranchSubcommand::ListAll => match branch_list_all_output(workspace) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            },
            BranchSubcommand::Switch(name) => {
                let root = match discover_git_root(workspace) {
                    Some(r) => r,
                    None => {
                        return Ok(local_command_error(anyhow::anyhow!(
                            "branch is only available inside a Git repository"
                        )));
                    }
                };
                match git_checkout(&root, &name) {
                    Ok(_) => Ok(CommandOutcome::Quiet),
                    Err(err) => Ok(local_command_error(err)),
                }
            }
            BranchSubcommand::Create(name) => match branch_create_output(workspace, &name) {
                Ok(_) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            },
            BranchSubcommand::Rename(name) => match branch_rename_output(workspace, &name) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            },
            BranchSubcommand::Delete(name) => match branch_delete_output(workspace, &name) {
                Ok(_) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            },
        },
        LocalCommand::Restore(None) => Ok(CommandOutcome::OutputError(
            restore_usage_message().to_string(),
        )),
        LocalCommand::Restore(Some(arg)) => {
            let staged = arg.starts_with("--staged ");
            let path = if staged {
                arg.split_once(' ')
                    .map(|x| x.1)
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                arg.to_string()
            };
            match restore_output(workspace, &path, staged) {
                Ok(_) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(local_command_error(err)),
            }
        }
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
            Ok(Some(advice)) => Ok(CommandOutcome::Output(advice)),
            Ok(None) => Ok(CommandOutcome::Quiet),
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
        LocalCommand::Stash(sub) => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Blocking(Box::new(move || match sub {
                StashSubcommand::Push => stash_output(&ws),
                StashSubcommand::Pop => stash_pop_output(&ws),
                StashSubcommand::List => stash_list_output(&ws),
                StashSubcommand::Drop => stash_drop_output(&ws),
            })))
        }
        LocalCommand::OpenFile(path) => {
            if path.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    open_file_usage_message().to_string(),
                ));
            }
            match open_in_editor(workspace, path, terminal) {
                Ok(()) => Ok(CommandOutcome::Quiet),
                Err(err) => Ok(CommandOutcome::OutputError(format!("Error: {err:#}"))),
            }
        }
        LocalCommand::Session(None) => match list_sessions_output(None, &usage_stats.session_id) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Session(Some(uuid)) => {
            if uuid == usage_stats.session_id {
                return Ok(CommandOutcome::Output(format!("Already in session {uuid}")));
            }
            match session_dir_path(&uuid) {
                Ok(path) if path.is_dir() => Ok(CommandOutcome::SwitchSession(uuid.into_owned())),
                Ok(_) => Ok(CommandOutcome::OutputError(format!(
                    "Error: no session found with UUID '{uuid}'"
                ))),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Sessions(filter) => {
            match list_sessions_output(filter.as_deref(), &usage_stats.session_id) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Usage => Ok(CommandOutcome::Output(usage_stats.format())),
        LocalCommand::Build => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Streaming(Box::new(move |sink| {
                build::build_output(&ws, &sink)
            })))
        }
        LocalCommand::Clear => {
            let prompt = system_prompt(
                llms.get(active_model)
                    .ok_or_else(|| anyhow!("unknown server '{active_model}'"))?,
            );
            session.clear(prompt);
            Ok(CommandOutcome::Cleared)
        }
        LocalCommand::Quit => Ok(CommandOutcome::Quit),
    }
}

pub fn print_screen(render: RenderContext<'_>, screen: ScreenState<'_>) {
    // Only hint a completion while the cursor sits at the end of what was typed.
    // Slash commands take priority over natural-language bindings; for the latter,
    // `ghost_index` selects which candidate to preview (cycled with Shift+Tab).
    let ghost = if screen.cursor == screen.input.len() {
        completion::command_ghost_suffix(screen.input)
            .or_else(|| {
                completion::natural_language_ghost_suffix_at(screen.input, screen.ghost_index)
            })
            .unwrap_or("")
    } else {
        ""
    };
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
            ghost,
            virtual_width: render.virtual_width,
            actual_width: render.actual_width,
            actual_height: render.actual_height,
            x_offset: render.x_offset,
        })
    );
}

/// The Alt+o feedback popup contents.
struct FeedbackWindow {
    title: String,
    /// The asked request, echoed below the title; `None` for a plain review.
    question: Option<String>,
    lines: Vec<String>,
    scroll: usize,
    x_offset: usize,
}

/// A review comment kept against a specific diff line of a file.
#[derive(Clone)]
struct ReviewComment {
    file: String,
    /// Diff-line index within the file (0-based).
    line: usize,
    text: String,
}

/// Interactive state for `/review` mode.
struct ReviewState {
    files: Vec<ReviewEntry>,
    selected: usize,
    /// Index of the highlighted line within the selected file's diff (moved
    /// with Up/Down).
    line: usize,
    /// Index of the first line shown in the left pane, within the selected
    /// file's diff.
    scroll: usize,
    /// Horizontal pan offset for the left pane.
    x_offset: usize,
    /// When set, the LLM feedback popup is open over the panes.
    feedback: Option<FeedbackWindow>,
    /// Comments recorded against diff lines, keyed by (file, line).
    comments: Vec<ReviewComment>,
    /// General notes entered in the input window as `# <note>`.
    general_notes: Vec<String>,
    /// When set, the inline comment editor is open for the highlighted line.
    comment_editor: Option<InputState>,
}

/// Why `run_review_mode` returned control to the caller.
enum ReviewSignal {
    /// Leave review mode.
    Exit,
    /// Run an LLM review of the selected file using the typed request.
    RequestReview {
        path: String,
        patch: String,
        request: String,
    },
    /// Open the selected file in the configured editor.
    OpenFile { path: String },
}

/// Static rendering pieces for the review prompt frame.
#[derive(Clone, Copy)]
struct ReviewChrome<'a> {
    current_model: &'a str,
    prompt_branch: Option<&'a str>,
    pending_count: usize,
}

impl ReviewState {
    fn new(launch: ReviewLaunch) -> Self {
        Self {
            files: launch.files,
            selected: 0,
            line: 0,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        }
    }

    /// Record a `# <note>` typed in the input window as a general note.
    fn add_general_note(&mut self, text: &str) {
        let body = general_comment_body(text);
        if !body.is_empty() {
            self.general_notes.push(body);
        }
    }

    fn selected_lines(&self) -> &[String] {
        self.files
            .get(self.selected)
            .map(|file| file.diff_lines.as_slice())
            .unwrap_or(&[])
    }

    fn selected_path(&self) -> Option<&str> {
        self.files.get(self.selected).map(|file| file.path.as_str())
    }

    /// The existing comment text for the highlighted line, if any.
    fn comment_for_selected_line(&self) -> Option<&str> {
        let path = self.selected_path()?;
        self.comments
            .iter()
            .find(|comment| comment.file == path && comment.line == self.line)
            .map(|comment| comment.text.as_str())
    }

    /// Diff-line indices of the selected file that carry a comment.
    fn commented_lines(&self) -> Vec<usize> {
        let Some(path) = self.selected_path() else {
            return Vec::new();
        };
        self.comments
            .iter()
            .filter(|comment| comment.file == path)
            .map(|comment| comment.line)
            .collect()
    }

    /// Open the inline comment editor for the highlighted line, pre-filled with
    /// any existing comment, and scroll so the editor box fits below the line.
    fn open_comment_editor(&mut self, body_height: usize) {
        let existing = self.comment_for_selected_line().unwrap_or("").to_string();
        let mut input = InputState::default();
        input.set_buffer(existing);
        self.comment_editor = Some(input);

        // Keep the highlighted line high enough that the box fits beneath it.
        let room = body_height.saturating_sub(orangu::tui::REVIEW_COMMENT_BOX_HEIGHT + 1);
        if self.line.saturating_sub(self.scroll) > room {
            self.scroll = self.line.saturating_sub(room);
        }
        if self.scroll > self.line {
            self.scroll = self.line;
        }
    }

    /// Save the editor's text as the comment for the highlighted line (an empty
    /// comment removes any existing one) and close the editor.
    fn commit_comment(&mut self) {
        let Some(editor) = self.comment_editor.take() else {
            return;
        };
        let Some(path) = self.selected_path().map(str::to_string) else {
            return;
        };
        let line = self.line;
        let text = editor.as_str().trim().to_string();
        self.comments
            .retain(|comment| !(comment.file == path && comment.line == line));
        if !text.is_empty() {
            self.comments.push(ReviewComment {
                file: path,
                line,
                text,
            });
        }
    }

    /// Clamp scroll/pan offsets for whichever view is active.
    fn clamp(&mut self, body_height: usize, left_width: usize, full_width: usize) {
        if let Some(feedback) = &mut self.feedback {
            // A pinned question line costs one row of review text.
            let review_rows = body_height.saturating_sub(usize::from(feedback.question.is_some()));
            let max_scroll = feedback.lines.len().saturating_sub(review_rows);
            feedback.scroll = feedback.scroll.min(max_scroll);
            let content_width = feedback
                .lines
                .iter()
                .map(|line| orangu::tui::visible_line_width(line))
                .max()
                .unwrap_or(0);
            feedback.x_offset = feedback
                .x_offset
                .min(content_width.saturating_sub(full_width));
        } else {
            self.line = self.line.min(self.selected_lines().len().saturating_sub(1));
            let max_scroll = self.selected_lines().len().saturating_sub(body_height);
            self.scroll = self.scroll.min(max_scroll);
            let content_width = self
                .selected_lines()
                .iter()
                .map(|line| orangu::tui::visible_line_width(line))
                .max()
                .unwrap_or(0);
            self.x_offset = self.x_offset.min(content_width.saturating_sub(left_width));
        }
    }

    /// Move the highlighted line up, scrolling the pane to keep it visible.
    fn cursor_up(&mut self) {
        self.line = self.line.saturating_sub(1);
        if self.line < self.scroll {
            self.scroll = self.line;
        }
    }

    /// Move the highlighted line down, scrolling the pane to keep it visible.
    fn cursor_down(&mut self, body_height: usize) {
        let last = self.selected_lines().len().saturating_sub(1);
        self.line = (self.line + 1).min(last);
        if body_height > 0 && self.line >= self.scroll + body_height {
            self.scroll = self.line + 1 - body_height;
        }
    }

    fn select_next(&mut self) {
        if self.selected + 1 < self.files.len() {
            self.selected += 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.line = 0;
            self.scroll = 0;
            self.x_offset = 0;
        }
    }

    fn set_status(&mut self, status: ReviewStatus) {
        if let Some(file) = self.files.get_mut(self.selected) {
            file.status = status;
        }
    }
}

fn build_review_prompt(path: &str, request: &str, patch: &str) -> String {
    let request = request.trim();
    let instruction = if request.is_empty() {
        format!(
            "Please review the following changes to `{path}` and give concise, actionable feedback."
        )
    } else {
        format!("Please review the following changes to `{path}`. {request}")
    };
    format!("{instruction}\n\n```diff\n{patch}\n```")
}

/// Copy `text` to the system clipboard.
fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new().context("failed to access the clipboard")?;
    clipboard
        .set_text(text.to_string())
        .context("failed to write to the clipboard")?;
    Ok(())
}

/// Format the recorded review comments as `<file>:<line>: <comment>` lines,
/// ordered by file then line. Line numbers are shown 1-based.
fn format_review_comments(comments: &[ReviewComment]) -> Vec<String> {
    let mut ordered: Vec<&ReviewComment> = comments.iter().collect();
    ordered.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
    ordered
        .into_iter()
        .map(|comment| format!("{}:{}: {}", comment.file, comment.line + 1, comment.text))
        .collect()
}

/// The human-readable status label for a file in the exit summary.
fn review_status_label(status: ReviewStatus) -> &'static str {
    match status {
        ReviewStatus::Approved => "Approved",
        ReviewStatus::Rejected => "Rejected",
        ReviewStatus::Unreviewed => "No review",
    }
}

/// A colored dot shown after the status label: green/red/white.
fn review_status_dot(status: ReviewStatus) -> &'static str {
    match status {
        ReviewStatus::Approved => "\x1b[38;2;80;200;120m●\x1b[0m",
        ReviewStatus::Rejected => "\x1b[38;2;220;80;80m●\x1b[0m",
        ReviewStatus::Unreviewed => "\x1b[38;2;230;230;230m●\x1b[0m",
    }
}

/// The body of a `# <note>` general comment, with the leading `#` removed.
fn general_comment_body(text: &str) -> String {
    let trimmed = text.trim_start();
    trimmed
        .strip_prefix('#')
        .unwrap_or(trimmed)
        .trim_start()
        .to_string()
}

/// Build the review exit summary: the lines to print to the output window, and
/// the text (if any) to copy to the clipboard. When every file is approved and
/// there are no comments, the summary is just "Patch approved". Otherwise it is
/// each file's status, the line comments, then the general notes, then a final
/// "Patch rejected" verdict. The line comments and general notes are copied to
/// the clipboard (never the per-file status or the verdict).
fn review_exit_output(
    files: &[ReviewEntry],
    comments: &[ReviewComment],
    general_notes: &[String],
) -> (Vec<String>, Option<String>) {
    let line_lines = format_review_comments(comments);

    // Both kinds of comment are copied (line comments first, then notes).
    let mut clip: Vec<String> = line_lines.clone();
    clip.extend(general_notes.iter().cloned());
    let clipboard = (!clip.is_empty()).then(|| clip.join("\n"));

    let all_approved = !files.is_empty()
        && files
            .iter()
            .all(|file| file.status == ReviewStatus::Approved);
    if all_approved && clip.is_empty() {
        return (vec!["\x1b[1mPatch approved\x1b[0m".to_string()], None);
    }

    let mut lines: Vec<String> = files
        .iter()
        .map(|file| {
            format!(
                "{}: {} {}",
                file.path,
                review_status_label(file.status),
                review_status_dot(file.status),
            )
        })
        .collect();
    lines.extend(line_lines);
    lines.extend(general_notes.iter().cloned());
    lines.push("\x1b[1mPatch rejected\x1b[0m".to_string());
    (lines, clipboard)
}

fn print_review_screen(
    state: &ReviewState,
    input_state: &InputState,
    viewport: &ViewportState,
    chrome: ReviewChrome<'_>,
    left_status: Option<StatusFragment>,
) {
    let feedback = state.feedback.as_ref().map(|feedback| ReviewFeedbackView {
        title: &feedback.title,
        question: feedback.question.as_deref(),
        lines: &feedback.lines,
        scroll: feedback.scroll,
        x_offset: feedback.x_offset,
    });
    let comment_editor = state
        .comment_editor
        .as_ref()
        .map(|editor| ReviewCommentEditor {
            text: editor.as_str(),
            cursor: editor.cursor(),
        });
    let commented_lines = state.commented_lines();
    print!("{CLEAR_TERMINAL_SEQUENCE}");
    print!(
        "{}",
        render_review_screen(ReviewScreenArgs {
            files: &state.files,
            selected: state.selected,
            line: state.line,
            scroll: state.scroll,
            x_offset: state.x_offset,
            feedback,
            comment_editor,
            commented_lines: &commented_lines,
            current_model: chrome.current_model,
            prompt_branch: chrome.prompt_branch,
            input: input_state.as_str(),
            cursor: input_state.cursor(),
            left_status,
            pending_count: chrome.pending_count,
            actual_width: viewport.actual_width,
            actual_height: viewport.actual_height,
        })
    );
}

/// Run the review event loop until the user exits or asks for an LLM review.
fn run_review_mode(
    state: &mut ReviewState,
    viewport: &mut ViewportState,
    input_state: &mut InputState,
    chrome: ReviewChrome<'_>,
) -> Result<ReviewSignal> {
    let mut escape_cancel = EscapeCancelState::default();
    loop {
        let body_height = review_pane_body_height(
            viewport.actual_height,
            input_state.as_str(),
            chrome.prompt_branch,
            viewport.actual_width,
        );
        let right_width = orangu::tui::review_right_width(&state.files, viewport.actual_width);
        let left_width = viewport.actual_width.saturating_sub(right_width + 1).max(1);
        state.clamp(body_height, left_width, viewport.actual_width);
        print_review_screen(state, input_state, viewport, chrome, None);
        std::io::stdout().flush()?;

        let (code, modifiers) = match event::read()? {
            Event::Resize(width, height) => {
                viewport.on_resize(usize::from(width), usize::from(height));
                continue;
            }
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind,
                ..
            }) if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat => (code, modifiers),
            _ => continue,
        };

        let alt =
            modifiers.contains(KeyModifiers::ALT) && !modifiers.contains(KeyModifiers::CONTROL);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);

        // While the feedback popup is open it is modal: scroll it, or close it.
        if let Some(feedback) = &mut state.feedback {
            escape_cancel.reset();
            match code {
                KeyCode::Char('x') | KeyCode::Esc => state.feedback = None,
                KeyCode::Up => feedback.scroll = feedback.scroll.saturating_sub(1),
                KeyCode::Down => feedback.scroll = feedback.scroll.saturating_add(1),
                KeyCode::Left => feedback.x_offset = feedback.x_offset.saturating_sub(1),
                KeyCode::Right => feedback.x_offset = feedback.x_offset.saturating_add(1),
                KeyCode::PageUp => feedback.scroll = feedback.scroll.saturating_sub(body_height),
                KeyCode::PageDown => feedback.scroll = feedback.scroll.saturating_add(body_height),
                _ => {}
            }
            continue;
        }

        // While the inline comment editor is open it is modal: type the comment,
        // Enter saves it, Esc discards it.
        if state.comment_editor.is_some() {
            escape_cancel.reset();
            match (code, alt, ctrl) {
                (KeyCode::Enter, _, _) => state.commit_comment(),
                (KeyCode::Esc, _, _) => state.comment_editor = None,
                (KeyCode::Backspace, true, _) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .delete_backward_readline_word();
                }
                (KeyCode::Backspace, _, _) => state.comment_editor.as_mut().unwrap().backspace(),
                (KeyCode::Delete, _, _) => state.comment_editor.as_mut().unwrap().delete(),
                (KeyCode::Left, _, true) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .move_backward_readline_word();
                }
                (KeyCode::Right, _, true) => {
                    state
                        .comment_editor
                        .as_mut()
                        .unwrap()
                        .move_forward_readline_word();
                }
                (KeyCode::Left, _, _) => state.comment_editor.as_mut().unwrap().move_left(),
                (KeyCode::Right, _, _) => state.comment_editor.as_mut().unwrap().move_right(),
                (KeyCode::Home, _, _) => state.comment_editor.as_mut().unwrap().move_home(),
                (KeyCode::End, _, _) => state.comment_editor.as_mut().unwrap().move_end(),
                (KeyCode::Char(ch), false, false) => {
                    state.comment_editor.as_mut().unwrap().insert_char(ch);
                }
                _ => {}
            }
            continue;
        }

        // A second Esc within the timeout leaves review mode; the first arms it.
        if code == KeyCode::Esc {
            if escape_cancel.handle_escape(std::time::Instant::now()) {
                return Ok(ReviewSignal::Exit);
            }
            continue;
        }
        escape_cancel.reset();

        match (code, alt, ctrl) {
            (KeyCode::Char('x'), true, _) => return Ok(ReviewSignal::Exit),
            (KeyCode::Char('j'), true, _) => state.select_next(),
            (KeyCode::Char('k'), true, _) => state.select_prev(),
            (KeyCode::Char('a'), true, _) => state.set_status(ReviewStatus::Approved),
            (KeyCode::Char('r'), true, _) => state.set_status(ReviewStatus::Rejected),
            (KeyCode::Char('c'), true, _) => state.open_comment_editor(body_height),
            (KeyCode::Char('e'), true, _) => {
                if let Some(file) = state.files.get(state.selected) {
                    return Ok(ReviewSignal::OpenFile {
                        path: file.path.clone(),
                    });
                }
            }
            (KeyCode::Char('o'), true, _) | (KeyCode::Enter, _, _) => {
                if input_state.as_str().trim_start().starts_with('#') {
                    // A `# <note>` in the input window is a general note, not an
                    // LLM request.
                    state.add_general_note(input_state.as_str());
                    input_state.clear();
                } else if let Some(file) = state.files.get(state.selected) {
                    return Ok(ReviewSignal::RequestReview {
                        path: file.path.clone(),
                        patch: file.patch.clone(),
                        request: input_state.as_str().to_string(),
                    });
                }
            }
            // Left-pane scrolling (Alt+arrows / PageUp/Down), mirroring the
            // main output window.
            (KeyCode::Up, true, _) => state.scroll = state.scroll.saturating_sub(1),
            (KeyCode::Down, true, _) => state.scroll = state.scroll.saturating_add(1),
            (KeyCode::Left, true, _) => state.x_offset = state.x_offset.saturating_sub(1),
            (KeyCode::Right, true, _) => state.x_offset = state.x_offset.saturating_add(1),
            (KeyCode::PageUp, _, _) => state.scroll = state.scroll.saturating_sub(body_height),
            (KeyCode::PageDown, _, _) => state.scroll = state.scroll.saturating_add(body_height),
            // Move the highlighted line through the diff, view following.
            (KeyCode::Up, false, _) => state.cursor_up(),
            (KeyCode::Down, false, _) => state.cursor_down(body_height),
            // Input window editing.
            (KeyCode::Backspace, true, _) => input_state.delete_backward_readline_word(),
            (KeyCode::Backspace, _, _) => input_state.backspace(),
            (KeyCode::Delete, _, _) => input_state.delete(),
            (KeyCode::Left, _, true) => input_state.move_backward_readline_word(),
            (KeyCode::Right, _, true) => input_state.move_forward_readline_word(),
            (KeyCode::Left, _, _) => input_state.move_left(),
            (KeyCode::Right, _, _) => input_state.move_right(),
            (KeyCode::Home, _, _) | (KeyCode::Char('a'), _, true) => input_state.move_home(),
            (KeyCode::End, _, _) | (KeyCode::Char('e'), _, true) => input_state.move_end(),
            (KeyCode::Char('k'), _, true) => input_state.kill_to_end(),
            (KeyCode::Char('u'), _, true) => input_state.kill_to_start(),
            (KeyCode::Char('w'), _, true) => input_state.delete_prev_word(),
            (KeyCode::Char(ch), false, false) => input_state.insert_char(ch),
            _ => {}
        }
    }
}

/// Result of an Alt+o review request.
enum ReviewRequestOutcome {
    /// The model responded (`Ok`) or the request errored (`Err`); either way the
    /// outcome is shown in the feedback popup.
    Completed(Result<String>),
    /// The user pressed Esc twice — abort and return to the panes.
    Cancelled,
    /// The user pressed Alt+x — leave review mode entirely.
    Exit,
}

/// Ask the LLM to review the selected file, rendering the review screen with a
/// thinking indicator until the response arrives. The exchange is recorded in
/// the session so it can be followed up after leaving review mode. While the
/// model works, `Esc` `Esc` cancels the request and `Alt+x` exits review mode;
/// either way the pending exchange is rolled back out of the session.
#[allow(clippy::too_many_arguments)]
async fn run_review_request(
    session: &mut ChatSession,
    prompt: &str,
    profile: &LlmConfiguration,
    tools: &ToolExecutor,
    state: &ReviewState,
    input_state: &InputState,
    viewport: &mut ViewportState,
    chrome: ReviewChrome<'_>,
) -> Result<ReviewRequestOutcome> {
    let checkpoint = session.checkpoint();
    let mut future = Box::pin(session.prompt(prompt, profile, tools, |_| {}, |_| {}, |_| {}));
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let started = std::time::Instant::now();
    let mut escape_cancel = EscapeCancelState::default();

    loop {
        tokio::select! {
            result = &mut future => return Ok(ReviewRequestOutcome::Completed(result)),
            _ = interval.tick() => {
                while event::poll(std::time::Duration::ZERO)? {
                    let (code, modifiers) = match event::read()? {
                        Event::Resize(width, height) => {
                            viewport.on_resize(usize::from(width), usize::from(height));
                            continue;
                        }
                        Event::Key(KeyEvent { code, modifiers, kind, .. })
                            if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat =>
                        {
                            (code, modifiers)
                        }
                        _ => continue,
                    };
                    let alt = modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL);
                    if code == KeyCode::Char('x') && alt {
                        drop(future);
                        session.rollback(checkpoint);
                        return Ok(ReviewRequestOutcome::Exit);
                    }
                    if code == KeyCode::Esc {
                        if escape_cancel.handle_escape(std::time::Instant::now()) {
                            drop(future);
                            session.rollback(checkpoint);
                            return Ok(ReviewRequestOutcome::Cancelled);
                        }
                    } else {
                        escape_cancel.reset();
                    }
                }
                let frame = (started.elapsed().as_millis()
                    / THINKING_FRAME_INTERVAL.as_millis().max(1)) as usize;
                let status = render_thinking_status(frame, started.elapsed());
                print_review_screen(state, input_state, viewport, chrome, Some(status));
                std::io::stdout().flush()?;
            }
        }
    }
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
        server_names,
        available_models,
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
            ghost_index: input_state.ghost_index,
        },
    );
    std::io::stdout().flush()?;

    loop {
        tokio::select! {
            result = &mut prompt_future => {
                let response = match result {
                    Ok(response) => response,
                    Err(error) => {
                        let partial = streamed_state
                            .lock()
                            .map(|state| state.output.clone())
                            .unwrap_or_default();
                        return Ok(WaitResult::Failed { partial, error });
                    }
                };
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
            ghost_index: input_state.ghost_index,
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
                            server_names,
                            available_models,
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
            ghost_index: input_state.ghost_index,
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
        server_names,
        available_models,
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
                            server_names,
                            available_models,
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
            ghost_index: input_state.ghost_index,
                    },
                );
                std::io::stdout().flush()?;
            }
        }
    }
}

/// Wait for a streaming command, draining its line channel into the output
/// window as lines arrive so the build log appears live instead of all at once.
async fn wait_for_streaming_command(
    wait_context: WaitContext<'_>,
    mut handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) -> anyhow::Result<anyhow::Result<()>> {
    let WaitContext {
        mut render,
        history,
        history_path: _,
        server_names,
        available_models,
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
                // Drain any lines still buffered before the task finished.
                while let Ok(line) = rx.try_recv() {
                    output_state.push_text(&line);
                }
                return Ok(result?);
            }
            _ = interval.tick() => {
                let elapsed = started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                if next_frame != frame {
                    frame = next_frame;
                }
                while let Ok(line) = rx.try_recv() {
                    output_state.push_text(&line);
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
                            server_names,
                            available_models,
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
            ghost_index: input_state.ghost_index,
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

/// Build a GET request to a server's `/v1/models` endpoint, attaching the
/// optional bearer token. OpenAI-compatible servers — including a llama.cpp
/// server started with `--api-key` — require `Authorization: Bearer <key>` on
/// every `/v1/*` endpoint, not just chat completions.
fn models_request(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    let url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
    let request = http_client.get(url);
    match api_key {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// Probe the active server and return its header status together with the list
/// of model ids it advertises (used for `/model` completion). `model_ok` is set
/// when the active wire model id is among the advertised models.
async fn probe_header_status(
    http_client: &reqwest::Client,
    workspace: &Path,
    active_model_id: &str,
    profile: &LlmConfiguration,
    endpoint: Option<&str>,
) -> (orangu::tui::HeaderStatus, Vec<String>) {
    let workspace_ok = workspace.exists();
    let mut server_ok = false;
    let mut model_ok = false;
    let mut available_models = Vec::new();

    if let Some(endpoint) = endpoint
        && let Ok(response) = models_request(http_client, endpoint, profile.api_key.as_deref())
            .send()
            .await
        && response.status().is_success()
    {
        server_ok = true;
        if let Ok(models) = response.json::<ModelsResponse>().await {
            for entry in models.data.iter().chain(models.models.iter()) {
                let id = if !entry.id.is_empty() {
                    &entry.id
                } else if !entry.model.is_empty() {
                    &entry.model
                } else if !entry.name.is_empty() {
                    &entry.name
                } else {
                    continue;
                };
                if id == active_model_id
                    || entry.model == active_model_id
                    || entry.name == active_model_id
                {
                    model_ok = true;
                }
                available_models.push(id.clone());
            }
        }
    }

    (
        orangu::tui::HeaderStatus {
            workspace_ok,
            server_ok,
            model_ok,
        },
        available_models,
    )
}

/// If the active server is not serving the configured model at startup, switch
/// to a model the server actually advertises. Returns `(old, new)` model ids
/// when a switch happened. The server (endpoint, provider, system prompt) is
/// unchanged — only the wire model id moves.
async fn try_startup_model_switch(
    http_client: &reqwest::Client,
    profile: &LlmConfiguration,
    active_model_id: &mut String,
    endpoint: Option<&str>,
) -> Option<(String, String)> {
    let endpoint = endpoint?;
    let response = models_request(http_client, endpoint, profile.api_key.as_deref())
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let models = response.json::<ModelsResponse>().await.ok()?;

    let available: Vec<String> = models
        .data
        .iter()
        .chain(models.models.iter())
        .filter_map(|entry| {
            if !entry.id.is_empty() {
                Some(entry.id.clone())
            } else if !entry.model.is_empty() {
                Some(entry.model.clone())
            } else if !entry.name.is_empty() {
                Some(entry.name.clone())
            } else {
                None
            }
        })
        .collect();

    // The server already serves the configured model — nothing to switch.
    if available.iter().any(|model| model == active_model_id) {
        return None;
    }

    // Otherwise move to the first model the server actually offers.
    let new_model = available.into_iter().next()?;
    let old = std::mem::replace(active_model_id, new_model.clone());
    Some((old, new_model))
}

fn session_dir_path(session_id: &str) -> Result<PathBuf> {
    let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(SESSIONS_DIRECTORY).join(session_id))
}

/// The `~/.orangu/last` scratch directory used across a `/restart` handoff.
fn restart_dir_path() -> Option<PathBuf> {
    Some(home::home_dir()?.join(RESTART_DIRECTORY))
}

/// Remove the `/restart` scratch directory. Errors are ignored: a missing or
/// unremovable directory must never block startup.
fn clear_restart_dir() {
    if let Some(dir) = restart_dir_path() {
        let _ = std::fs::remove_dir_all(&dir);
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
fn restart_executable_path() -> Result<PathBuf> {
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

fn list_sessions_output(workspace_filter: Option<&str>, active_session: &str) -> Result<String> {
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
    use super::completion::{
        command_ghost_suffix, completion_candidates, natural_language_ghost_candidates,
        natural_language_ghost_suffix_at,
    };
    use super::git::with_explicit_pager_width;
    use super::git::{
        branch_delete_output, discover_git_dir, discover_git_root, git_workspace_diff,
        init_repo_output, is_protected_branch, list_workspace_files_tree, workspace_branch_name,
    };
    use super::input::{apply_completion, cycle_ghost_suggestion, idle_status_refresh_timeout};
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
            server_names: &[],
            available_models: &[],
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
    fn models_request_attaches_optional_bearer_token() {
        let client = reqwest::Client::new();

        let with_key = super::models_request(&client, "http://localhost:8100/v1", Some("secret"))
            .build()
            .expect("build request");
        assert_eq!(with_key.url().as_str(), "http://localhost:8100/v1/models");
        assert_eq!(
            with_key
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer secret")
        );

        let without_key = super::models_request(&client, "http://localhost:8100/v1", None)
            .build()
            .expect("build request");
        assert!(
            without_key
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .is_none()
        );
    }

    #[test]
    fn record_elapsed_counts_llm_time_without_tokens() {
        use std::time::Duration;

        let mut stats = super::UsageStats::new();
        // A failed or cancelled turn: time is spent but no response is recorded.
        stats.record_elapsed(Duration::from_secs(5), Duration::from_secs(2));

        assert_eq!(stats.total_llm_duration, Duration::from_secs(3));
        assert_eq!(stats.total_tool_duration, Duration::from_secs(2));
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn record_response_counts_llm_time_and_tokens() {
        use std::time::Duration;

        let mut stats = super::UsageStats::new();
        stats.record_response(
            Duration::from_secs(4),
            "hello world",
            Duration::from_secs(1),
        );

        assert_eq!(stats.total_llm_duration, Duration::from_secs(3));
        assert_eq!(stats.total_tool_duration, Duration::from_secs(1));
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn review_state_navigation_shows_only_selected_file_diff() {
        use super::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![
                ReviewEntry {
                    path: "a.txt".to_string(),
                    status: ReviewStatus::Unreviewed,
                    diff_lines: (0..30).map(|i| format!("a {i}")).collect(),
                    patch: String::new(),
                },
                ReviewEntry {
                    path: "b.txt".to_string(),
                    status: ReviewStatus::Unreviewed,
                    diff_lines: (0..8).map(|i| format!("b {i}")).collect(),
                    patch: String::new(),
                },
            ],
            selected: 0,
            line: 0,
            scroll: 7,
            x_offset: 5,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // The left pane reflects the selected file's own diff.
        assert_eq!(state.selected_lines().len(), 30);

        // Moving to the next file shows it from the top.
        state.select_next();
        assert_eq!(state.selected, 1);
        assert_eq!(state.scroll, 0, "scroll resets on file change");
        assert_eq!(state.x_offset, 0, "horizontal pan resets on file change");
        assert_eq!(state.selected_lines().len(), 8);

        // Cannot move past the last file.
        state.select_next();
        assert_eq!(state.selected, 1);

        // Scroll is clamped to the selected file's diff length minus the body.
        state.scroll = 999;
        state.clamp(5, 20, 40);
        assert_eq!(state.scroll, 8 - 5);

        // Marking sets status on the selected file only.
        state.set_status(ReviewStatus::Approved);
        assert_eq!(state.files[1].status, ReviewStatus::Approved);
        assert_eq!(state.files[0].status, ReviewStatus::Unreviewed);

        state.select_prev();
        assert_eq!(state.selected, 0);
        assert_eq!(state.scroll, 0);
        assert_eq!(state.line, 0, "line cursor resets on file change");
    }

    #[test]
    fn review_cursor_moves_and_scrolls_to_follow() {
        use super::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: (0..20).map(|i| format!("a {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            line: 0,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Down past the visible body height scrolls the pane to follow.
        let body = 5;
        for _ in 0..6 {
            state.cursor_down(body);
        }
        assert_eq!(state.line, 6);
        assert_eq!(state.scroll, 6 + 1 - body, "view follows the cursor down");

        // Back up above the top scrolls the pane back up.
        for _ in 0..5 {
            state.cursor_up();
        }
        assert_eq!(state.line, 1);
        assert_eq!(state.scroll, 1, "view follows the cursor up");

        // The cursor cannot move past the last line.
        for _ in 0..100 {
            state.cursor_down(body);
        }
        assert_eq!(state.line, 19);
    }

    #[test]
    fn review_comments_are_recorded_per_file_and_line() {
        use super::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Unreviewed,
            diff_lines: (0..10).map(|i| format!("x {i}")).collect(),
            patch: String::new(),
        };
        let mut state = ReviewState {
            files: vec![entry("a.txt"), entry("b.txt")],
            selected: 0,
            line: 3,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Open the editor (pre-filled empty), type, and commit.
        state.open_comment_editor(20);
        assert!(state.comment_editor.is_some());
        for ch in "looks off".chars() {
            state.comment_editor.as_mut().unwrap().insert_char(ch);
        }
        state.commit_comment();
        assert!(state.comment_editor.is_none());
        assert_eq!(state.comments.len(), 1);
        assert_eq!(state.comments[0].file, "a.txt");
        assert_eq!(state.comments[0].line, 3);
        assert_eq!(state.comments[0].text, "looks off");
        assert_eq!(state.commented_lines(), vec![3]);

        // Re-opening pre-fills the existing comment; editing replaces it.
        state.open_comment_editor(20);
        assert_eq!(state.comment_editor.as_ref().unwrap().as_str(), "looks off");
        state.commit_comment();
        assert_eq!(state.comments.len(), 1, "no duplicate for the same line");

        // An empty comment removes it.
        state.open_comment_editor(20);
        state.comment_editor.as_mut().unwrap().kill_to_start();
        state.commit_comment();
        assert!(state.comments.is_empty());
        assert!(state.commented_lines().is_empty());

        // Comments are scoped to the selected file.
        state.open_comment_editor(20);
        for ch in "note".chars() {
            state.comment_editor.as_mut().unwrap().insert_char(ch);
        }
        state.commit_comment();
        state.select_next();
        assert_eq!(state.selected, 1);
        assert!(
            state.commented_lines().is_empty(),
            "b.txt has no comments yet"
        );
    }

    #[test]
    fn alt_c_on_commented_line_opens_editor_prefilled_in_the_box() {
        use super::ReviewState;
        use orangu::tui::{
            ReviewCommentEditor, ReviewEntry, ReviewScreenArgs, ReviewStatus, render_review_screen,
        };

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: (0..10).map(|i| format!("x {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            line: 2,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Record a comment on the highlighted line, then re-open with Alt+c.
        state.open_comment_editor(12);
        for ch in "needs a guard".chars() {
            state.comment_editor.as_mut().unwrap().insert_char(ch);
        }
        state.commit_comment();
        state.open_comment_editor(12);

        // The editor holds the existing comment, and it renders inside the box.
        assert_eq!(
            state.comment_editor.as_ref().unwrap().as_str(),
            "needs a guard"
        );
        let editor = state
            .comment_editor
            .as_ref()
            .map(|input| ReviewCommentEditor {
                text: input.as_str(),
                cursor: input.cursor(),
            });
        let commented = state.commented_lines();
        let rendered = render_review_screen(ReviewScreenArgs {
            files: &state.files,
            selected: state.selected,
            line: state.line,
            scroll: state.scroll,
            x_offset: state.x_offset,
            feedback: None,
            comment_editor: editor,
            commented_lines: &commented,
            current_model: "model",
            prompt_branch: Some("main"),
            input: "",
            cursor: 0,
            left_status: None,
            pending_count: 0,
            actual_width: 60,
            actual_height: 16,
        });
        assert!(
            rendered.contains("needs a guard"),
            "existing comment not loaded into the box"
        );
    }

    #[test]
    fn format_review_comments_sorts_and_uses_one_based_lines() {
        use super::{ReviewComment, format_review_comments};

        let comments = vec![
            ReviewComment {
                file: "src/b.rs".to_string(),
                line: 4,
                text: "second file".to_string(),
            },
            ReviewComment {
                file: "src/a.rs".to_string(),
                line: 9,
                text: "later line".to_string(),
            },
            ReviewComment {
                file: "src/a.rs".to_string(),
                line: 0,
                text: "first line".to_string(),
            },
        ];

        assert_eq!(
            format_review_comments(&comments),
            vec![
                "src/a.rs:1: first line".to_string(),
                "src/a.rs:10: later line".to_string(),
                "src/b.rs:5: second file".to_string(),
            ]
        );
    }

    #[test]
    fn review_exit_output_summarizes_statuses_and_comments() {
        use super::{ReviewComment, review_exit_output, review_status_dot};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str, status| ReviewEntry {
            path: path.to_string(),
            status,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let status_line = |path: &str, label: &str, status| {
            format!("{path}: {label} {}", review_status_dot(status))
        };

        // All approved + no comments => just "Patch approved", nothing copied.
        let files = vec![
            entry("a.txt", ReviewStatus::Approved),
            entry("b.txt", ReviewStatus::Approved),
        ];
        let (lines, clipboard) = review_exit_output(&files, &[], &[]);
        assert_eq!(lines, vec!["\x1b[1mPatch approved\x1b[0m".to_string()]);
        assert!(clipboard.is_none());

        // Mixed statuses: per-file status lines, then comments; only comments
        // are copied.
        let files = vec![
            entry("a.txt", ReviewStatus::Approved),
            entry("b.txt", ReviewStatus::Rejected),
            entry("c.txt", ReviewStatus::Unreviewed),
        ];
        let comments = vec![ReviewComment {
            file: "b.txt".to_string(),
            line: 2,
            text: "fix this".to_string(),
        }];
        let (lines, clipboard) = review_exit_output(&files, &comments, &[]);
        assert_eq!(
            lines,
            vec![
                status_line("a.txt", "Approved", ReviewStatus::Approved),
                status_line("b.txt", "Rejected", ReviewStatus::Rejected),
                status_line("c.txt", "No review", ReviewStatus::Unreviewed),
                "b.txt:3: fix this".to_string(),
                "\x1b[1mPatch rejected\x1b[0m".to_string(),
            ]
        );
        assert_eq!(clipboard.as_deref(), Some("b.txt:3: fix this"));

        // Approved but with a comment => not the "Patch approved" shortcut.
        let files = vec![entry("a.txt", ReviewStatus::Approved)];
        let comments = vec![ReviewComment {
            file: "a.txt".to_string(),
            line: 0,
            text: "nit".to_string(),
        }];
        let (lines, clipboard) = review_exit_output(&files, &comments, &[]);
        assert_eq!(
            lines,
            vec![
                status_line("a.txt", "Approved", ReviewStatus::Approved),
                "a.txt:1: nit".to_string(),
                "\x1b[1mPatch rejected\x1b[0m".to_string(),
            ]
        );
        assert_eq!(clipboard.as_deref(), Some("a.txt:1: nit"));
    }

    #[test]
    fn review_exit_output_appends_general_notes() {
        use super::{ReviewComment, review_exit_output, review_status_dot};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let files = vec![ReviewEntry {
            path: "a.txt".to_string(),
            status: ReviewStatus::Approved,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        }];
        let comments = vec![ReviewComment {
            file: "a.txt".to_string(),
            line: 4,
            text: "tighten this".to_string(),
        }];
        // General notes come from the input window with the '#' already stripped.
        let general_notes = vec!["overall solid, ship after nits".to_string()];

        let (lines, clipboard) = review_exit_output(&files, &comments, &general_notes);
        // Line comments first, then the general note, then the verdict.
        assert_eq!(
            lines,
            vec![
                format!(
                    "a.txt: Approved {}",
                    review_status_dot(ReviewStatus::Approved)
                ),
                "a.txt:5: tighten this".to_string(),
                "overall solid, ship after nits".to_string(),
                "\x1b[1mPatch rejected\x1b[0m".to_string(),
            ]
        );
        // Both the line comment and the general note are copied.
        assert_eq!(
            clipboard.as_deref(),
            Some("a.txt:5: tighten this\noverall solid, ship after nits")
        );
    }

    #[test]
    fn add_general_note_strips_hash_and_keeps_line_comments_separate() {
        use super::ReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = ReviewState {
            files: vec![ReviewEntry {
                path: "a.txt".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: (0..5).map(|i| format!("x {i}")).collect(),
                patch: String::new(),
            }],
            selected: 0,
            line: 2,
            scroll: 0,
            x_offset: 0,
            feedback: None,
            comments: Vec::new(),
            general_notes: Vec::new(),
            comment_editor: None,
        };

        // Input-window "# ..." is stored as a general note with the '#' removed.
        state.add_general_note("# please add a test");
        state.add_general_note("#no space");
        // Whitespace-only / bare '#' notes are ignored.
        state.add_general_note("#   ");

        assert_eq!(
            state.general_notes,
            vec!["please add a test".to_string(), "no space".to_string()]
        );
        // General notes do not become line comments.
        assert!(state.comments.is_empty());
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
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/open_file /etc/hosts",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
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
            let mut active_model_id = "gemma".to_string();
            let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
            let mut session = ChatSession::new("system");

            let outcome = handle_command(
                input,
                CommandState {
                    active_model: &mut active_model,
                    active_model_id: &mut active_model_id,
                    current_endpoint: &mut current_endpoint,
                    session: &mut session,
                    detect_model: &mut false,
                },
                CommandContext {
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: workspace.path(),
                    usage_stats: &super::UsageStats::new(),
                    available_models: &[],
                    virtual_width: 512,
                    auto_rebase: false,
                    auto_squash: false,
                    terminal: "",
                    forge: crate::git::Forge::GitHub,
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
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");
        let outcome = handle_command(
            "/list_files",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
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
            let result = branch_delete_output(workspace.path(), branch);
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
    // `run_git_diff_pager` invokes `sh -c`, which is not guaranteed to be in PATH
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
    fn set_server_switches_active_endpoint() {
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
        let mut active_model_id = GEMMA.to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");
        let mut detect_model = false;

        let outcome = handle_command(
            "/server gpt-4.1",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut detect_model,
            },
            CommandContext {
                startup_model: GEMMA,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Quiet));
        assert_eq!(active_model, OPENAI);
        // Switching server resets the wire model id to the server's model.
        assert_eq!(active_model_id, "gpt-4.1");
        assert_eq!(
            current_endpoint,
            Some(normalized_openai_endpoint("https://api.openai.com/v1"))
        );
        // Selecting a server requests model auto-detection against it.
        assert!(detect_model);
    }

    #[test]
    fn set_model_changes_wire_model_only() {
        const GEMMA: &str = "gemma-4-E4B-it-GGUF";

        let llms = HashMap::from([(
            GEMMA.to_string(),
            test_profile(
                "llama.cpp",
                "http://localhost:8100/v1",
                "ggml-org/gemma-4-E4B-it-GGUF",
            ),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = GEMMA.to_string();
        let mut active_model_id = "ggml-org/gemma-4-E4B-it-GGUF".to_string();
        let endpoint = normalized_openai_endpoint("http://localhost:8100/v1");
        let mut current_endpoint = Some(endpoint.clone());
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/model some-other-model",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: GEMMA,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Quiet));
        // The wire model id changes; the server and endpoint stay put.
        assert_eq!(active_model_id, "some-other-model");
        assert_eq!(active_model, GEMMA);
        assert_eq!(current_endpoint, Some(endpoint));
    }

    #[test]
    fn model_info_marks_active_green_and_others_red() {
        const SERVER: &str = "local";

        let llms = HashMap::from([(
            SERVER.to_string(),
            test_profile("llama.cpp", "http://localhost:8100/v1", "model-a"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = SERVER.to_string();
        let mut active_model_id = "model-a".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");
        let available = vec![
            "model-a".to_string(),
            "model-b".to_string(),
            "model-c".to_string(),
        ];

        let outcome = handle_command(
            "/model",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: SERVER,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &available,
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
            },
        )
        .expect("handle command");

        match outcome {
            CommandOutcome::Output(text) => {
                let ok = super::FEEDBACK_OK;
                let err = super::FEEDBACK_ERR;
                assert_eq!(text, format!("{ok} model-a\n{err} model-b\n{err} model-c"));
            }
            _ => panic!("expected output from /model"),
        }
    }

    #[test]
    fn server_info_marks_active_green_and_others_red() {
        let llms = HashMap::from([
            (
                "alpha".to_string(),
                test_profile("llama.cpp", "http://localhost:8100/v1", "model-a"),
            ),
            (
                "bravo".to_string(),
                test_profile("llama.cpp", "http://localhost:8200/v1", "model-b"),
            ),
            (
                "charlie".to_string(),
                test_profile("llama.cpp", "http://localhost:8300/v1", "model-c"),
            ),
        ]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "bravo".to_string();
        let mut active_model_id = "model-b".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8200/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/server",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: "bravo",
                startup_endpoint: "http://localhost:8200/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
            },
        )
        .expect("handle command");

        match outcome {
            CommandOutcome::Output(text) => {
                let ok = super::FEEDBACK_OK;
                let err = super::FEEDBACK_ERR;
                // Servers are listed in sorted order; only the active one is green.
                assert_eq!(text, format!("{err} alpha\n{ok} bravo\n{err} charlie"));
            }
            _ => panic!("expected output from /server"),
        }
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
        let mut active_model_id = "default".to_string();
        let mut current_endpoint = Some("http://localhost:11434/v1".to_string());

        let outcome = handle_command(
            "/unknown",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                startup_model: "default",
                startup_endpoint: "http://localhost:11434/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
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
            &[],
        )
        .expect("slash completion");
        assert_eq!(
            slash_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (start, _, natural_candidates) =
            completion_candidates("Open READ", "Open READ".len(), workspace.path(), &[], &[])
                .expect("natural completion");
        assert_eq!(start, "Open ".len());
        assert_eq!(
            natural_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (_, _, ignored_candidates) =
            completion_candidates("Open ign", "Open ign".len(), workspace.path(), &[], &[])
                .expect("ignored completion");
        assert!(ignored_candidates.is_empty());

        let (_, _, git_candidates) =
            completion_candidates("Open con", "Open con".len(), workspace.path(), &[], &[])
                .expect("git completion");
        assert!(git_candidates.is_empty());

        let (_, _, target_candidates) = completion_candidates(
            "/open_file t",
            "/open_file t".len(),
            workspace.path(),
            &[],
            &[],
        )
        .expect("target completion");
        assert_eq!(target_candidates, vec!["src/tui.rs".to_string()]);

        let (start, _, show_candidates) =
            completion_candidates("Show t", "Show t".len(), workspace.path(), &[], &[])
                .expect("show completion");
        assert_eq!(start, "Show ".len());
        assert_eq!(show_candidates, vec!["src/tui.rs".to_string()]);

        let (start, _, show_file_candidates) = completion_candidates(
            "show file READ",
            "show file READ".len(),
            workspace.path(),
            &[],
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
    fn suggests_ghost_suffix_for_partial_slash_commands() {
        // A unique prefix completes to the rest of the command.
        assert_eq!(command_ghost_suffix("/q"), Some("uit"));
        assert_eq!(command_ghost_suffix("/qui"), Some("t"));

        // The first matching command wins, so the hint narrows as letters arrive.
        assert_eq!(command_ghost_suffix("/"), Some("help"));

        // A fully typed command and unmatched prefixes have nothing to suggest.
        assert_eq!(command_ghost_suffix("/quit"), None);
        assert_eq!(command_ghost_suffix("/zzz"), None);

        // Once an argument is being typed (whitespace) the name hint stops.
        assert_eq!(command_ghost_suffix("/quit "), None);
        assert_eq!(command_ghost_suffix("not a command"), None);
    }

    #[test]
    fn suggests_ghost_suffix_for_partial_natural_language_bindings() {
        // The rendered hint is cycle position 0.
        let ghost = |input| natural_language_ghost_suffix_at(input, 0);

        // A partial verb completes to the rest of the binding.
        assert_eq!(ghost("discon"), Some("nect"));
        assert_eq!(ghost("rebas"), Some("e"));

        // Argument-taking prefixes complete through their trailing space.
        assert_eq!(ghost("diff a"), Some("gainst "));
        assert_eq!(ghost("connect t"), Some("o "));

        // Matching is case-insensitive; the suggested suffix is canonical.
        assert_eq!(ghost("DIF"), Some("f"));

        // A complete binding has nothing left to hint, even when a longer
        // binding shares its prefix (e.g. "connect" vs "connect to ").
        assert_eq!(ghost("commit"), None);
        assert_eq!(ghost("merge"), None);
        assert_eq!(ghost("connect"), None);
        assert_eq!(ghost("diff"), None);

        // Still hinted while the binding is incomplete.
        assert_eq!(ghost("c"), Some("onnect"));

        // Empty input, slash input, and unknown prefixes suggest nothing.
        assert_eq!(ghost(""), None);
        assert_eq!(ghost("/q"), None);
        assert_eq!(ghost("xyzzy"), None);
    }

    #[test]
    fn shift_tab_cycles_through_natural_language_candidates() {
        // "c" matches several bindings; cycling walks them in priority order and
        // wraps back to the first. Bindings differing only by trailing whitespace
        // (e.g. "checkout " vs "checkout") collapse to one entry.
        let candidates = natural_language_ghost_candidates("c");
        assert!(
            candidates.len() > 1,
            "expected multiple candidates for \"c\", got {candidates:?}"
        );
        assert_eq!(
            natural_language_ghost_suffix_at("c", 0),
            Some(candidates[0])
        );
        assert_eq!(
            natural_language_ghost_suffix_at("c", 1),
            Some(candidates[1])
        );
        // Index wraps around the candidate list.
        assert_eq!(
            natural_language_ghost_suffix_at("c", candidates.len()),
            Some(candidates[0])
        );

        // The whole list completes "c" to distinct, real commands.
        for suffix in candidates {
            let completed = format!("c{suffix}");
            assert!(
                parse_local_command(completed.trim()).is_some()
                    || parse_local_command(&format!("{completed}1")).is_some()
                    || parse_local_command(&format!("{completed}1 2")).is_some(),
                "cycled candidate {completed:?} does not parse"
            );
        }
    }

    #[test]
    fn tab_accepts_natural_language_ghost_suggestion() {
        let workspace = tempdir().expect("workspace");

        // Tab fills in the ghosted binding when no structured candidate applies.
        let mut input_state = InputState::default();
        input_state.set_buffer("ad".to_string());
        apply_completion(&mut input_state, workspace.path(), &[], &[]);
        assert_eq!(input_state.as_str(), "add comment on ");
        assert_eq!(input_state.cursor(), "add comment on ".len());

        // A fully typed binding has no ghost, so Tab leaves it untouched.
        let mut input_state = InputState::default();
        input_state.set_buffer("commit".to_string());
        apply_completion(&mut input_state, workspace.path(), &[], &[]);
        assert_eq!(input_state.as_str(), "commit");

        // The binding ghost wins over a same-prefixed filename: typing "c" with
        // a "contrib/" directory present completes to "connect", not "contrib/".
        let repo = tempdir().expect("repo");
        std::fs::create_dir(repo.path().join("contrib")).expect("contrib dir");
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        apply_completion(&mut input_state, repo.path(), &[], &[]);
        assert_eq!(input_state.as_str(), "connect");

        // Shift+Tab advances the preview; Tab then accepts the shown candidate.
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        let second = format!("c{}", natural_language_ghost_candidates("c")[1]);
        cycle_ghost_suggestion(&mut input_state);
        assert_eq!(input_state.ghost_index, 1);
        apply_completion(&mut input_state, workspace.path(), &[], &[]);
        assert_eq!(input_state.as_str(), second);

        // Editing the line resets the cycle back to the first candidate.
        let mut input_state = InputState::default();
        input_state.set_buffer("c".to_string());
        cycle_ghost_suggestion(&mut input_state);
        input_state.insert_char('o');
        assert_eq!(input_state.ghost_index, 0);
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

        let (_, _, initial_file_candidates) = completion_candidates(
            "/show_file ",
            "/show_file ".len(),
            workspace.path(),
            &[],
            &[],
        )
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
            &[],
        )
        .expect("quoted file completion");
        assert_eq!(
            quoted_candidates,
            vec!["\"README.md".to_string(), "\"doc/README.md".to_string()]
        );

        let (_, _, target_candidates) = completion_candidates(
            "/show_file t",
            "/show_file t".len(),
            workspace.path(),
            &[],
            &[],
        )
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
            completion_candidates("/open_file ", "/open_file ".len(), &workspace, &[], &[])
                .expect("open completion");
        assert!(open_candidates.is_empty());

        let (_, _, show_candidates) =
            completion_candidates("/show_file ", "/show_file ".len(), &workspace, &[], &[])
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

        let (start, _, candidates) = completion_candidates(
            "/checkout m",
            "/checkout m".len(),
            workspace.path(),
            &[],
            &[],
        )
        .expect("checkout completion");
        assert_eq!(start, "/checkout ".len());
        assert!(candidates.contains(&"main".to_string()), "main missing");
        assert!(
            candidates.contains(&"mybranch".to_string()),
            "branch missing"
        );
        assert!(candidates.contains(&"main.rs".to_string()), "file missing");

        let (start, _, nat_candidates) =
            completion_candidates("checkout m", "checkout m".len(), workspace.path(), &[], &[])
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

        let (start, _, candidates) = completion_candidates(
            "switch to m",
            "switch to m".len(),
            workspace.path(),
            &[],
            &[],
        )
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
        let (start, _, candidates) = completion_candidates(
            "/add_file n",
            "/add_file n".len(),
            workspace.path(),
            &[],
            &[],
        )
        .expect("add_file completion");
        assert_eq!(start, "/add_file ".len());
        assert_eq!(candidates[0], "newdir/");
        assert!(candidates.contains(&"newfile.txt".to_string()));
        // tracked file not included
        assert!(!candidates.contains(&"tracked.rs".to_string()));

        // Natural-language form
        let (start, _, nat_candidates) =
            completion_candidates("add n", "add n".len(), workspace.path(), &[], &[])
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
            completion_candidates("remove s", "remove s".len(), workspace.path(), &[], &[])
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
        let (start, _, src_candidates) = completion_candidates(
            "/move_file s",
            "/move_file s".len(),
            workspace.path(),
            &[],
            &[],
        )
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
            &[],
        )
        .expect("move_file destination completion");
        assert_eq!(start, "/move_file src/main.rs ".len());
        assert!(dst_candidates.contains(&"untracked.txt".to_string()));

        // Natural-language form — first arg
        let (start, _, nat_candidates) =
            completion_candidates("move s", "move s".len(), workspace.path(), &[], &[])
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
            &[],
        );
        if let Some((start, _, candidates)) = result {
            assert_eq!(start, "/cherry_pick ".len());
            // Abbreviated hashes are 7 chars
            assert!(candidates.iter().all(|h| h.len() >= 4));
        }

        // Natural-language form triggers completion
        let nl_result = completion_candidates(
            "cherry pick ",
            "cherry pick ".len(),
            workspace.path(),
            &[],
            &[],
        );
        if let Some((start, _, _)) = nl_result {
            assert_eq!(start, "cherry pick ".len());
        }
    }
}

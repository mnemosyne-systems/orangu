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
mod dispatch;
mod export;
mod git;
mod init;
mod input;
mod manual;
mod models;
mod quotes;
mod render;
mod review;
mod session_store;
mod shell;
mod stats;
mod terminal;
mod wait;

#[cfg(test)]
mod test_support;

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
        AutoReviewRejectView, AutoReviewScreenArgs, FEEDBACK_ERR, FEEDBACK_OK, ReviewCommentEditor,
        ReviewEntry, ReviewFeedbackView, ReviewScreenArgs, ReviewStatus, ScreenRenderArgs,
        StatusFragment, auto_review_pane_body_height, render_auto_review_screen,
        render_review_screen, render_screen, render_thinking_status, render_tool_running_status,
        render_working_status, review_pane_body_height, terminal_height, terminal_width,
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
    BisectSubcommand, BranchSubcommand, CommandContext, CommandOutcome, CommandState, ExportTarget,
    LocalCommand, LocalError, PruneTarget, StashSubcommand, add_file_usage_message,
    amend_usage_message, cherry_pick_usage_message, close_usage_message, comment_usage_message,
    commit_usage_message, get_comments_usage_message, grep_usage_message, merge_usage_message,
    model_usage_message, move_file_usage_message, open_file_usage_message, parse_local_command,
    prune_usage_message, pull_usage_message, remove_file_usage_message, restore_usage_message,
    server_usage_message, sorted_model_names, system_prompt,
};
use dispatch::*;
use git::{
    Forge, add_file_output, amend_output, bisect_bad_output, bisect_good_output, bisect_log_output,
    bisect_reset_output, bisect_skip_output, bisect_start_output, bisect_status_output,
    branch_create_output, branch_delete_output, branch_list_all_output, branch_list_output,
    branch_rename_output, cherry_pick_output, close_output, collect_review_diff, comment_output,
    commit_output, create_pull_request_output, discover_git_root, fetch_active_pull_requests,
    get_comments_output, git_checkout, git_diff_against_branch, git_workspace_diff, grep_output,
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
use models::*;
use render::{format_tools, render_markdown_for_console, show_file_output};
use review::*;
use session_store::*;
use stats::*;
use terminal::*;
use wait::*;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    /// Print the shell completion script for the detected shell and exit.
    ///
    /// Detects the current shell from $SHELL. Pipe into your shell's eval or
    /// drop the output into the appropriate completions directory:
    ///
    ///   bash: eval "$(orangu -s)"
    ///   zsh:  orangu -s > ~/.zsh/completions/_orangu
    ///   fish: orangu -s > ~/.config/fish/completions/orangu.fish
    #[arg(short = 's', long = "shell-completions")]
    shell_completions: bool,
}

fn print_shell_completions() -> Result<()> {
    let shell = std::env::var("SHELL").unwrap_or_default();
    let script = if shell.ends_with("/bash") || shell == "bash" {
        shell::BASH
    } else if shell.ends_with("/zsh") || shell == "zsh" {
        shell::ZSH
    } else if shell.ends_with("/fish") || shell == "fish" {
        shell::FISH
    } else {
        return Err(anyhow!(
            "could not detect shell from $SHELL ({shell:?}).\n\
             Supported shells: bash, zsh, fish.\n\
             \n\
             Usage:\n\
             \x20 bash: eval \"$(orangu -s)\"\n\
             \x20 zsh:  orangu -s > ~/.zsh/completions/_orangu\n\
             \x20 fish: orangu -s > ~/.config/fish/completions/orangu.fish"
        ));
    };
    print!("{script}");
    Ok(())
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
    if args.shell_completions {
        return print_shell_completions();
    }
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
    // The last `/review` summary and `/auto_review` report (Markdown), kept
    // so `/comment <number> with [auto] review` can post them.
    let mut last_review_report: Option<String> = None;
    let mut last_auto_review_report: Option<String> = None;
    // When set, the post-loop exec resumes this session instead of the current
    // one — used by `/session <UUID>` to switch sessions in place.
    let mut resume_override: Option<String> = None;
    // When set, the post-loop exec switches to this workspace directory (without
    // a resume target) — used by `/session <path>` to open a new workspace.
    let mut workspace_override: Option<PathBuf> = None;
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

    // Fetch the open pull/merge requests once at startup, off the UI thread, and
    // keep them in memory. Caching them here means `/pull` completion can offer
    // numbers without shelling out to `gh`/`glab` on every keystroke.
    let mut pr_handle = discover_git_root(tools.workspace()).map(|_| {
        let pr_workspace = tools.workspace().to_path_buf();
        tokio::task::spawn_blocking(move || fetch_active_pull_requests(&pr_workspace, forge))
    });

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
            server_names: &server_names,
            available_models: &available_models,
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

        // Collect the startup pull-request fetch once it finishes, caching the
        // open requests in memory for later use (e.g. `/pull` completion).
        if pr_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = pr_handle.take()
        {
            match handle.await {
                Ok(Ok(requests)) => completion::set_active_pull_requests(&requests),
                Ok(Err(err)) => {
                    output_state.push_text(&format!("Could not load open pull requests: {err}"))
                }
                Err(_) => {}
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
                review_reports: git::ReviewReports {
                    review: last_review_report.as_deref(),
                    auto_review: last_auto_review_report.as_deref(),
                },
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
            CommandOutcome::SwitchWorkspace(dir) => {
                save_session_messages(&session_messages_path, session.messages())?;
                update_session_metadata_timestamp(&session_metadata_path)?;
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                workspace_override = Some(dir);
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
            CommandOutcome::PendingList => {
                output_state.push_text(&format_pending_list(&pending_commands));
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::PendingDelete(index) => {
                apply_pending_delete(index, &mut pending_commands, &mut output_state);
                output_state.reset_scroll();
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
                    server_names: &server_names,
                    available_models: &available_models,
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
                    server_names: &server_names,
                    available_models: &available_models,
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
                // On exit, print the category-grouped report to the output
                // window and copy its raw Markdown to the system clipboard; the
                // Markdown is also kept for `/comment <n> with review` and
                // `/export review`.
                let (lines, markdown) =
                    review_exit_output(&review.files, &review.comments, &review.general_notes);
                last_review_report = Some(markdown.clone());
                completion::set_available_review_reports(
                    last_review_report.is_some(),
                    last_auto_review_report.is_some(),
                );
                for line in &lines {
                    output_state.push_text(line);
                }
                if let Err(err) = copy_to_clipboard(&markdown) {
                    output_state.push_text(&format!(
                        "Could not copy the review report to the clipboard: {err}"
                    ));
                }
                // The modal view overwrote the screen; the next loop iteration
                // redraws the normal interface from the top.
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::AutoReview(launch) => {
                // Auto review needs a connected server before opening the view.
                let Some(endpoint) = current_endpoint.clone() else {
                    output_state.push_text("Error: Not connected to an LLM server");
                    output_state.reset_scroll();
                    continue;
                };
                let Some(profile) = config.llms.get(&active_model) else {
                    output_state.push_text(&format!("Error: unknown server '{active_model}'"));
                    output_state.reset_scroll();
                    continue;
                };
                let mut prompt_profile = profile.clone();
                prompt_profile.endpoint = endpoint;
                prompt_profile.model = active_model_id.clone();
                let chrome = ReviewChrome {
                    current_model: &active_model_id,
                    prompt_branch: prompt_branch.as_deref(),
                    pending_count: pending_commands.len(),
                };

                let state = run_auto_review_mode(
                    launch,
                    &prompt_profile,
                    &mut usage_stats,
                    &mut viewport,
                    chrome,
                    &workspace,
                    &config.terminal,
                )
                .await?;

                // On exit, print the rendered report to the output window and
                // copy its raw Markdown to the system clipboard; the Markdown
                // is also kept for `/comment <n> with auto review`.
                let (lines, clipboard) = auto_review_exit_output(&state);
                last_auto_review_report = Some(clipboard.clone());
                completion::set_available_review_reports(
                    last_review_report.is_some(),
                    last_auto_review_report.is_some(),
                );
                for line in &lines {
                    output_state.push_text(line);
                }
                if let Err(err) = copy_to_clipboard(&clipboard) {
                    output_state.push_text(&format!(
                        "Could not copy the auto review report to the clipboard: {err}"
                    ));
                }
                // The modal view overwrote the screen; the next loop iteration
                // redraws the normal interface from the top.
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Export(target) => {
                let result = match target {
                    ExportTarget::Console => {
                        export::export_console(&workspace, output_state.lines(), &active_model_id)
                    }
                    ExportTarget::Review => {
                        match last_review_report
                            .as_deref()
                            .or(last_auto_review_report.as_deref())
                        {
                            Some(report) => {
                                export::export_review(&workspace, report, &active_model_id)
                            }
                            None => Err(anyhow!(
                                "No review to export; run /review or /auto_review first"
                            )),
                        }
                    }
                };
                match result {
                    Ok(path) => {
                        output_state.push_text(&format!("Exported to {}", path.display()));
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
            CommandOutcome::Manual => {
                manual::run_manual_mode(
                    &mut viewport,
                    manual::ManualChrome {
                        current_model: &active_model_id,
                        prompt_branch: prompt_branch.as_deref(),
                        pending_count: pending_commands.len(),
                    },
                )?;
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
                    server_names: &server_names,
                    available_models: &available_models,
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
        let exe = restart_executable_path()?;
        let mut command = std::process::Command::new(&exe);
        command.arg("--config").arg(&config_path);
        if let Some(new_workspace) = &workspace_override {
            // Opening a different workspace: pass no --resume, so startup either
            // auto-resumes an existing session for that workspace/branch or
            // starts a fresh one there.
            command.arg("--workspace").arg(new_workspace);
        } else {
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
            command
                .arg("--workspace")
                .arg(&resume_workspace)
                .arg("--resume")
                .arg(resume_target);
        }
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

/// Resolve a `/session <path>` argument to an existing workspace directory,
/// expanding a leading `~`/`~/` and normalizing the result. Relative paths are
/// taken against the current directory. Returns `None` when the argument does
/// not point at a real directory.
fn resolve_existing_dir_arg(arg: &str) -> Option<PathBuf> {
    let expanded = if arg == "~" {
        home::home_dir()?
    } else if let Some(rest) = arg.strip_prefix("~/") {
        home::home_dir()?.join(rest)
    } else {
        PathBuf::from(arg)
    };
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir().ok()?.join(expanded)
    };
    let normalized = normalize_path(&absolute);
    normalized.is_dir().then_some(normalized)
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

    use super::{llm_prompt_block_reason, resolve_workspace_root};

    use orangu::tui::HeaderStatus;

    use std::path::PathBuf;

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
}

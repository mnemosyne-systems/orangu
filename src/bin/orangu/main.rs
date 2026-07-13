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

mod activity_log;
mod build;
mod commands;
mod completion;
mod dispatch;
mod export;
mod git;
mod information;
mod init;
mod input;
mod manual;
mod models;
mod quotes;
mod render;
mod review;
mod schedule;
mod session_store;
mod shell;
mod shell_command;
mod slash_command;
mod stats;
mod terminal;
mod wait;
mod workspace_tab;

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
    agents::hooks::{rescan_changed_files, run_session_start_hook},
    config::{
        ClientAppConfiguration, LlmConfiguration, default_client_config_path,
        load_client_configuration,
    },
    llm::{ChatMessage, StreamMetrics, normalized_openai_endpoint},
    session::ChatSession,
    tools::ToolExecutor,
    tui::{
        AutoReviewDiffView, AutoReviewFileMode, AutoReviewRejectView, AutoReviewScreenArgs,
        FEEDBACK_ERR, FEEDBACK_OK, ReviewCommentEditor, ReviewEntry, ReviewFeedbackView,
        ReviewScreenArgs, ReviewStatus, ScreenRenderArgs, StatusFragment, TabStatus,
        WorkspaceTabsView, auto_review_pane_body_height, render_thinking_status,
        render_tool_running_status, render_working_status, review_pane_body_height,
        terminal_height, terminal_width,
    },
    workspaces::{normalize_path, resolve_workspace_root},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
};
use tiktoken_rs::cl100k_base;
use uuid::Uuid;

use anyhow::Error;
use commands::ReviewLaunch;
use commands::{
    AutoReviewTarget, BisectSubcommand, BranchSubcommand, CommandContext, CommandOutcome,
    CommandState, ExportTarget, LocalCommand, LocalError, PruneTarget, StashSubcommand,
    amend_usage_message, cherry_pick_usage_message, close_usage_message, comment_usage_message,
    commit_usage_message, create_file_usage_message, delete_file_usage_message,
    get_comments_usage_message, grep_usage_message, issue_usage_message, merge_usage_message,
    model_usage_message, move_file_usage_message, open_file_usage_message, parse_local_command,
    prune_usage_message, pull_usage_message, restore_usage_message, server_usage_message,
    sorted_model_names, system_prompt,
};
use dispatch::*;
use git::{
    Forge, amend_output, bisect_bad_output, bisect_good_output, bisect_log_output,
    bisect_reset_output, bisect_skip_output, bisect_start_output, bisect_status_output,
    branch_create_output, branch_delete_output, branch_list_all_output, branch_list_output,
    branch_rename_output, cherry_pick_output, close_output, collect_review_diff, comment_output,
    commit_output, create_pull_request_output, discover_git_root, fetch_active_pull_requests,
    fetch_issue_metadata, fetch_output, fetch_pull_request_details, get_comments_output,
    git_checkout, git_diff_against_branch, git_workspace_diff, grep_output, init_repo_output,
    issue_field_output, list_workspace_files_tree, log_output, merge_output, open_in_editor,
    pull_request_output, push_output, rebase_output, restore_output, show_output, squash_output,
    stash_drop_output, stash_list_output, stash_output, stash_pop_output, status_output,
    sync_default_branch, workspace_branch_name,
};
use input::{
    EscapeCancelState, IDLE_STATUS_REFRESH_INTERVAL, InputContext, InputResult, InputState,
    InterruptState, OutputState, PendingResponse, RenderContext, ScreenState, StreamRenderState,
    ViewportState, WaitContext, WaitResult, read_input,
};
use models::*;
use render::{format_tools, render_markdown_for_console, show_file_output};
use review::*;
use session_store::*;
use stats::*;
use terminal::*;
use wait::*;
use workspace_tab::{
    TabAction, TabServerConfig, WorkspaceRing, WorkspaceTab, restore_session_slot,
    save_session_slot, spawn_slot_save,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Override the configured TUI theme. Pass a shipped/user theme name, or a
    /// path to a `.theme` file (typically under `~/.orangu/themes/`).
    #[arg(long)]
    theme: Option<String>,
    #[arg(short, long)]
    workspace: Option<PathBuf>,
    #[arg(short, long)]
    resume: Option<String>,
    /// Reopen the workspace tabs that were open at the end of the last run
    /// (saved in ~/.orangu/workspaces).
    #[arg(short = 'a', long = "all")]
    all: bool,
    /// List all stored sessions as a table (SESSION, WORKSPACE, BRANCH) and exit.
    #[arg(short = 'l', long = "list")]
    list: bool,
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

fn main() -> ExitCode {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let result = runtime.block_on(run());

    // Background `spawn_blocking` work — the startup knowledge-graph scan,
    // git sync/PR/issue-metadata fetches — may still be running when `run()`
    // returns. Dropping the runtime normally blocks until every such task
    // finishes, which on a large or slow-to-scan workspace can make quitting
    // hang for a long time. Give it a brief grace period, then exit anyway;
    // the OS reclaims the abandoned thread.
    runtime.shutdown_timeout(std::time::Duration::from_millis(500));

    match result {
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
    if args.list {
        print!("{}", list_all_sessions_output()?);
        return Ok(());
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
    let cli_theme_override = args.theme.clone();
    orangu::tui::theme::Theme::set_auto_theme_names(
        &config.auto_dark_theme,
        &config.auto_light_theme,
    );
    let requested_theme = cli_theme_override.as_deref().unwrap_or(&config.theme);
    if let Some(cli_theme) = cli_theme_override.as_deref() {
        orangu::tui::theme::Theme::apply_cli_override(cli_theme)
            .with_context(|| format!("failed to apply --theme {}", cli_theme))?;
    } else if let Err(err) = orangu::tui::theme::Theme::apply_named(requested_theme) {
        eprintln!(
            "warning: failed to apply configured theme '{}': {err}",
            requested_theme
        );
    }

    let quote_module = quotes::QuoteModule::from_str(&config.quotes);
    // Remove any binary staged by a previous `/restart`; it is only needed
    // across the exec handoff and must not accumulate.
    clear_restart_dir();

    let status_http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;
    // Shared orangu-server `id_slot` assignment across every tab's interactive
    // session on this server (see `orangu::llm::SlotRegistry`).
    let slot_registry = orangu::llm::SlotRegistry::default();

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
    let mut current_endpoint = Some(startup_endpoint.clone());

    // The system prompt is shared across workspace tabs (per-workspace model
    // comes later), so it is resolved once and reused when opening tabs.
    let system_prompt = system_prompt(
        config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured server {}", active_model))?,
        None,
    )
    .to_string();

    // Open the first workspace tab (tab 1). Each tab is its own session; the
    // ring holds the others as the user opens more with `/workspace`/Alt+Insert.
    //
    // When -a is given without an explicit --workspace/--resume, the first
    // entry in ~/.orangu/workspaces is opened with its exact session ID so
    // the -a loop can reliably skip it by ID and never creates a duplicate tab.
    let saved_workspaces = if args.all {
        load_open_workspaces()
    } else {
        vec![]
    };
    let (initial_workspace, initial_session) =
        if args.all && args.workspace.is_none() && args.resume.is_none() {
            if let Some((ws, sess)) = saved_workspaces.first() {
                (ws.clone(), sess.clone())
            } else {
                (resolve_workspace_root(None)?, None)
            }
        } else {
            (resolve_workspace_root(args.workspace)?, None)
        };
    let mut initial_tab = WorkspaceTab::open(
        initial_workspace,
        initial_session.as_deref().or(args.resume.as_deref()),
        initial_session.is_none() && args.resume.is_none(),
        &system_prompt,
        config.compression,
        config.auto_downsample_lines,
        config.diff_file_cap,
        TabServerConfig {
            server: active_model.clone(),
            model_id: active_model_id.clone(),
            endpoint: current_endpoint.clone(),
            llms: &config.llms,
            slots: slot_registry.clone(),
        },
    )?;
    // Load the initial tab's branch from session metadata so that Alt+./, and
    // checkout_tab_branch! can restore the right branch on next tab switch.
    if let Some(branch) = load_session_branch(&initial_tab.session_id) {
        initial_tab.current_branch = Some(branch);
    }
    // Restore this tab's orangu-server slot from disk, if one was saved for the
    // same server+model in a previous run (see `SlotRegistry`) — skips
    // re-prefilling a prefix orangu-server already has cached, on both an
    // explicit `--resume` and an automatic workspace+branch resume. A no-op
    // for a brand-new session (nothing was ever saved for it).
    if let Some(profile) = config.llms.get(&initial_tab.active_model)
        && let Some(notice) = restore_session_slot(
            &mut initial_tab.session,
            &initial_tab.session_slot_path,
            &initial_tab.session_id,
            initial_tab.current_endpoint.as_deref(),
            &initial_tab.active_model_id,
            &slot_registry,
            &status_http_client,
            profile,
        )
        .await
    {
        initial_tab.output_state.push_text(&notice);
    }
    let mut ring = WorkspaceRing::new();
    // If -a/--all, reopen the workspace tabs that were open at the end of the
    // last run. Skip paths that equal the initial workspace (already open) and
    // silently ignore tabs whose directories have since been deleted.
    if args.all {
        for (saved_workspace, saved_session) in &saved_workspaces {
            let is_initial = match saved_session {
                Some(id) => *id == initial_tab.session_id,
                None => *saved_workspace == initial_tab.workspace,
            };
            if !is_initial
                && let Ok(mut tab) = WorkspaceTab::open(
                    saved_workspace.clone(),
                    saved_session.as_deref(),
                    saved_session.is_none(),
                    &system_prompt,
                    config.compression,
                    config.auto_downsample_lines,
                    config.diff_file_cap,
                    TabServerConfig {
                        server: active_model.clone(),
                        model_id: active_model_id.clone(),
                        endpoint: current_endpoint.clone(),
                        llms: &config.llms,
                        slots: slot_registry.clone(),
                    },
                )
            {
                if let Some(branch) = load_session_branch(&tab.session_id) {
                    tab.current_branch = Some(branch);
                }
                ring.park(tab);
            }
        }
    }

    let mut _terminal_ui_guard = TerminalUiGuard::new(config.mouse)?;

    let vw = terminal_width();
    let vh = terminal_height();
    let mut viewport = ViewportState::new(config.width, vw, vh);
    let mut interrupt_state = InterruptState::default();
    let mut input_state = InputState::default();
    let mut restart_requested = false;
    // When set, the post-loop exec resumes this session instead of the current
    // one — used by `/session <UUID>` to switch sessions in place.
    let mut resume_override: Option<String> = None;
    // When set, the post-loop exec switches to this workspace directory (without
    // a resume target) — used by `/session <path>` to open a new workspace.
    let mut workspace_override: Option<PathBuf> = None;
    // A pending workspace tab switch, applied at the top of the next iteration
    // so it never races the active tab's borrows in the render context.
    let mut pending_tab_action: Option<TabAction> = None;

    // The active tab's live state lives in these locals, exactly as the single
    // workspace did before tabs. A tab switch parks them in the ring and unpacks
    // the target tab back into them (see `apply_tab_action!`).
    let WorkspaceTab {
        mut workspace,
        mut tools,
        mut skills,
        mut session,
        mut output_state,
        mut pending_commands,
        mut usage_stats,
        mut history,
        mut session_id,
        mut session_dir,
        mut session_hist_path,
        mut session_messages_path,
        mut session_metadata_path,
        mut session_theme_override,
        mut session_slot_path,
        mut current_branch,
        mut last_review_report,
        mut last_auto_review_report,
        mut last_review_appendix,
        mut last_auto_review_appendix,
        mut last_review_was_auto,
        mut last_duplicates_report,
        mut startup_notice_until,
        mut pending_response,
        mut last_git_diff_hash,
        active_model: initial_active_model,
        active_model_id: initial_active_model_id,
        current_endpoint: initial_current_endpoint,
    } = initial_tab;
    active_model = initial_active_model;
    active_model_id = initial_active_model_id;
    current_endpoint = initial_current_endpoint;
    let apply_session_theme = |session_theme_override: &Option<String>| {
        if cli_theme_override.is_some() {
            return;
        }
        let theme_name = session_theme_override.as_deref().unwrap_or(&config.theme);
        if let Err(err) = orangu::tui::Theme::apply_named(theme_name) {
            eprintln!(
                "warning: failed to apply session theme '{}': {err}",
                theme_name
            );
        }
    };
    apply_session_theme(&session_theme_override);

    // Resolves server connectivity (with auto-failover to another configured
    // server if the default doesn't respond), model pinning, and semantic
    // `/search`'s embeddings capability — all in the background, so the very
    // first render never blocks on it. Until this completes, the header
    // shows "Server"/"Model" as pending (a white dot); the main loop below
    // applies the result the moment it's ready and prints a notice if a
    // different server was chosen. `embeddings_server` starts empty (search
    // unavailable) until then, matching its existing "empty means
    // unavailable" convention.
    let mut startup_resolution_handle: Option<tokio::task::JoinHandle<StartupResolution>> =
        Some(tokio::spawn(resolve_startup_state(
            status_http_client.clone(),
            config.clone(),
            server_names.clone(),
        )));
    let mut embeddings_server = String::new();

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

    // Likewise fetch the repository's reviewers, assignees, and labels once, off
    // the UI thread, so `/issue` value completion has them without a per-keystroke
    // `gh`/`glab` call.
    let mut issue_meta_handle = discover_git_root(tools.workspace()).map(|_| {
        let meta_workspace = tools.workspace().to_path_buf();
        tokio::task::spawn_blocking(move || fetch_issue_metadata(&meta_workspace, forge))
    });

    // Scan the workspace and build the initial Knowledge Graph in the background.
    // Runs on a blocking thread so the TUI is never stalled. The handle is
    // collected in the event loop below, exactly like `sync_handle` / `pr_handle`.
    //
    // `kg_store` is shared between `tools` (reads it for `graph_lookup`) and
    // the event loop below (writes into it once the scan finishes). `kg_status`
    // is the same pairing for the build-status signal `/auto_review`'s status
    // bar reads (`GraphBuildStatus`): `Building` until this block sets it.
    let kg_store = tools.graph_store.clone(); // Arc pointer shared with ToolExecutor
    let kg_status = tools.graph_status.clone();
    let scan_workspace = tools.workspace().to_path_buf();
    let mut kg_scan_handle: Option<tokio::task::JoinHandle<_>> =
        Some(tokio::task::spawn_blocking(move || {
            run_session_start_hook(&scan_workspace)
        }));

    // Per-file sha256 hashes used by the incremental KG rescan.
    // Populated once the initial scan completes, then updated incrementally.
    let mut kg_file_hashes: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    // Handle for an in-flight incremental rescan (spawned per LLM turn when files change).
    let mut kg_rescan_handle: Option<
        tokio::task::JoinHandle<orangu::agents::hooks::IncrementalScanResult>,
    > = None;

    // Workspace tab switching. `current_tab!()` packs the active tab's locals
    // into a `WorkspaceTab` (moving them out); `load_tab!(t)` unpacks one back
    // into them; `apply_tab_action!` parks the active tab in the ring and makes
    // another active. The macros close over the run-loop locals by definition
    // site, so the loop body keeps using those locals unchanged.
    macro_rules! current_tab {
        () => {{
            let session_theme_override = load_session_theme(&session_dir);
            WorkspaceTab {
                workspace,
                tools,
                skills,
                session,
                output_state,
                pending_commands,
                usage_stats,
                history,
                session_id,
                session_dir,
                session_hist_path,
                session_messages_path,
                session_metadata_path,
                session_theme_override,
                session_slot_path,
                current_branch,
                last_review_report,
                last_auto_review_report,
                last_review_appendix,
                last_auto_review_appendix,
                last_review_was_auto,
                last_duplicates_report,
                startup_notice_until,
                pending_response,
                last_git_diff_hash,
                active_model: active_model.clone(),
                active_model_id: active_model_id.clone(),
                current_endpoint: current_endpoint.clone(),
            }
        }};
    }
    macro_rules! load_tab {
        ($tab:expr) => {{
            let tab = $tab;
            workspace = tab.workspace;
            tools = tab.tools;
            skills = tab.skills;
            session = tab.session;
            output_state = tab.output_state;
            pending_commands = tab.pending_commands;
            usage_stats = tab.usage_stats;
            history = tab.history;
            session_id = tab.session_id;
            session_dir = tab.session_dir;
            session_hist_path = tab.session_hist_path;
            session_messages_path = tab.session_messages_path;
            session_metadata_path = tab.session_metadata_path;
            session_theme_override = tab.session_theme_override;
            apply_session_theme(&session_theme_override);
            session_slot_path = tab.session_slot_path;
            current_branch = tab.current_branch;
            last_review_report = tab.last_review_report;
            last_auto_review_report = tab.last_auto_review_report;
            last_review_appendix = tab.last_review_appendix;
            last_auto_review_appendix = tab.last_auto_review_appendix;
            last_review_was_auto = tab.last_review_was_auto;
            last_duplicates_report = tab.last_duplicates_report;
            startup_notice_until = tab.startup_notice_until;
            pending_response = tab.pending_response;
            last_git_diff_hash = tab.last_git_diff_hash;
            active_model = tab.active_model;
            active_model_id = tab.active_model_id;
            current_endpoint = tab.current_endpoint;
        }};
    }
    // Fire-and-forget save of the about-to-be-parked tab's orangu-server slot (see
    // `SlotRegistry`), so switching tabs doesn't block on a network
    // round-trip. MUST be called before current_tab!()/ring.rotate/open, i.e.
    // while `session` etc. still refer to the tab being left.
    macro_rules! park_active_slot {
        () => {
            spawn_slot_save(
                session.assigned_slot(),
                &session_slot_path,
                &session_id,
                current_endpoint.as_deref(),
                &active_model_id,
                &slot_registry,
                &status_http_client,
            );
        };
    }
    // Restore the newly active tab's orangu-server slot from disk, if one was
    // saved for the same server+model — skips re-prefilling a prefix
    // orangu-server already has cached. Awaited (not fire-and-forget): the user
    // has to look at the tab before typing anyway, and the failure mode is
    // parity with today's full-reprocess behavior.
    //
    // MUST be called after load_tab! so `session`/`active_model`/
    // `current_endpoint` all refer to the incoming (newly active) tab.
    macro_rules! restore_active_slot {
        () => {
            if let Some(profile) = config.llms.get(&active_model)
                && let Some(notice) = restore_session_slot(
                    &mut session,
                    &session_slot_path,
                    &session_id,
                    current_endpoint.as_deref(),
                    &active_model_id,
                    &slot_registry,
                    &status_http_client,
                    profile,
                )
                .await
            {
                output_state.push_text(&notice);
            }
        };
    }
    macro_rules! apply_tab_action {
        ($action:expr) => {{
            match $action {
                TabAction::Next => {
                    current_branch = workspace_branch_name(tools.workspace());
                    park_active_slot!();
                    let target = ring.rotate(current_tab!(), 1);
                    load_tab!(target);
                    restore_active_slot!();
                    checkout_tab_branch!();
                }
                TabAction::Previous => {
                    current_branch = workspace_branch_name(tools.workspace());
                    park_active_slot!();
                    let target = ring.rotate(current_tab!(), -1);
                    load_tab!(target);
                    restore_active_slot!();
                    checkout_tab_branch!();
                }
                TabAction::SwitchTo(index) => {
                    if index >= ring.total() {
                        output_state.push_text(&format!("No workspace {} is open.", index + 1));
                    } else {
                        current_branch = workspace_branch_name(tools.workspace());
                        park_active_slot!();
                        let target = ring.switch_to(current_tab!(), index);
                        load_tab!(target);
                        restore_active_slot!();
                        checkout_tab_branch!();
                    }
                }

                TabAction::New => {
                    // A fresh tab on the active workspace's directory, with its
                    // own new session; the user re-points it with `/workspace`.
                    match WorkspaceTab::open(
                        workspace.clone(),
                        None,
                        false,
                        &system_prompt,
                        config.compression,
                        config.auto_downsample_lines,
                        config.diff_file_cap,
                        TabServerConfig {
                            server: active_model.clone(),
                            model_id: active_model_id.clone(),
                            endpoint: current_endpoint.clone(),
                            llms: &config.llms,
                            slots: slot_registry.clone(),
                        },
                    ) {
                        Ok(new_tab) => {
                            // No restore afterward: this is a brand-new
                            // session with nothing saved to restore.
                            park_active_slot!();
                            let target = ring.open(current_tab!(), new_tab);
                            load_tab!(target);
                        }
                        Err(err) => output_state.push_text(&format!("Error: {err:#}")),
                    }
                }
                TabAction::Close => {
                    if ring.total() == 1 {
                        output_state.push_text("Only /quit exits orangu.");
                    } else {
                        let parked = current_tab!();
                        let _ = parked.save(&slot_registry, &status_http_client).await;
                        // `total > 1` guarantees a neighbour to focus.
                        let target = ring.close(parked).expect("more than one tab is open");
                        load_tab!(target);
                        checkout_tab_branch!();
                    }
                }
            }
            output_state.reset_scroll();
        }};
    }
    // After loading a new tab (Next/Previous/SwitchTo/Close), switch the git
    // repo to the branch that was active when that tab was last used.  Skips
    // workspaces without a git root or when the branch is already correct.
    //
    // MUST be called after load_tab! so that `workspace`, `tools`, and
    // `current_branch` all refer to the incoming (newly active) tab.
    macro_rules! checkout_tab_branch {
        () => {
            if let Some(branch) = &current_branch {
                if discover_git_root(&workspace).is_some()
                    && workspace_branch_name(&workspace).as_deref() != Some(branch.as_str())
                {
                    if let Err(err) = git_checkout(&workspace, branch) {
                        output_state.push_text(&format!("branch switch: {err:#}"));
                    }
                }
            }
        };
    }
    // Restart the per-workspace background tasks (default-branch sync, open PRs,
    // issue metadata) after the active workspace changes. Called after load_tab!
    // so `tools.workspace()` already points at the new workspace.
    macro_rules! restart_sync_tasks {
        () => {{
            sync_notice = None;
            sync_handle = discover_git_root(tools.workspace()).map(|_| {
                let w = tools.workspace().to_path_buf();
                tokio::task::spawn_blocking(move || sync_default_branch(&w, forge))
            });
            pr_handle = discover_git_root(tools.workspace()).map(|_| {
                let w = tools.workspace().to_path_buf();
                tokio::task::spawn_blocking(move || fetch_active_pull_requests(&w, forge))
            });
            issue_meta_handle = discover_git_root(tools.workspace()).map(|_| {
                let w = tools.workspace().to_path_buf();
                tokio::task::spawn_blocking(move || fetch_issue_metadata(&w, forge))
            });
        }};
    }
    // Persist every open tab on exit so each one stays resumable — the active
    // tab from its locals, plus every parked tab in the ring. Also record the
    // open workspace paths so `orangu -a` can restore the layout next time.
    macro_rules! save_all_tabs {
        () => {{
            save_session_messages(&session_messages_path, session.messages())?;
            let active_branch = workspace_branch_name(tools.workspace());
            update_session_metadata_branch(&session_metadata_path, active_branch.as_deref())?;
            save_session_slot(
                session.assigned_slot(),
                &session_slot_path,
                &session_id,
                current_endpoint.as_deref(),
                &active_model_id,
                &slot_registry,
                &status_http_client,
            )
            .await;
            for tab in ring.parked() {
                let _ = tab.save(&slot_registry, &status_http_client).await;
            }
            {
                let active_pos = ring.active_pos();
                let all_workspaces: Vec<(PathBuf, String)> = (0..ring.total())
                    .map(|pos| {
                        if pos == active_pos {
                            (workspace.clone(), session_id.clone())
                        } else {
                            let idx = if pos < active_pos { pos } else { pos - 1 };
                            let tab = &ring.parked()[idx];
                            (tab.workspace.clone(), tab.session_id.clone())
                        }
                    })
                    .collect();
                save_open_workspaces(&all_workspaces);
            }
        }};
    }

    // `/schedule`: minutes already over when orangu starts never fire, so the
    // marker begins at the current minute.
    let mut schedule_last_minute = session_store::current_unix_timestamp() / 60;

    loop {
        // Apply a pending tab switch before anything borrows the active tab's
        // locals this iteration (the render context below borrows `tools`).
        if let Some(action) = pending_tab_action.take() {
            apply_tab_action!(action);
        }

        let tab_bar = (ring.total() > 1).then(|| WorkspaceTabsView {
            count: ring.total(),
            active: ring.active_pos(),
            placement: config.workspaces,
        });
        // Per-tab colored dots: only computed when feedback is on and multiple
        // tabs are open. The wait loop recomputes these from live handle state
        // on every render tick so parked-tab dots update without a tab switch.
        let tab_statuses: Vec<TabStatus> = if config.feedback && ring.total() > 1 {
            let active_pos = ring.active_pos();
            (0..ring.total())
                .map(|pos| {
                    if pos == active_pos {
                        TabStatus::Valid
                    } else {
                        let others_idx = if pos < active_pos { pos } else { pos - 1 };
                        ring.parked()[others_idx].dot_status()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // Pre-compute the variant used during any wait (local command, streaming
        // or LLM response): the active tab's dot becomes white-blinking.
        let mut tab_statuses_working = tab_statuses.clone();
        if let Some(s) = tab_statuses_working.get_mut(ring.active_pos()) {
            *s = TabStatus::Working;
        }
        let prompt_branch = workspace_branch_name(tools.workspace());

        // Apply the background startup resolution's result the moment it's
        // ready (server auto-failover, model pinning, embeddings detection —
        // see `resolve_startup_state`), announcing any switch it made.
        if let Some(handle) = startup_resolution_handle.take() {
            if handle.is_finished() {
                if let Ok(resolution) = handle.await {
                    if resolution.switched_server {
                        output_state
                            .push_text(&format!("Switched to server: {}", resolution.server));
                    }
                    if let Some((old_model, new_model)) = &resolution.model_switched {
                        output_state
                            .push_text(&format!("Switched model from {old_model} to {new_model}"));
                    }
                    active_model = resolution.server;
                    active_model_id = resolution.active_model_id;
                    current_endpoint = config
                        .llms
                        .get(&active_model)
                        .map(|profile| profile.endpoint.clone());
                    embeddings_server = resolution.embeddings_server;
                }
                // Left as `None`: the normal per-iteration probe below now
                // applies, reflecting whatever was just settled on.
            } else {
                startup_resolution_handle = Some(handle);
            }
        }
        // While that resolution is still pending, show "Server"/"Model" as
        // pending (white) instead of blocking this render on the same
        // network check the background task is already doing.
        let (mut header_status, available_models) = if startup_resolution_handle.is_some() {
            (
                orangu::tui::HeaderStatus {
                    workspace_ok: tools.workspace().exists(),
                    server_ok: orangu::tui::ConnStatus::Pending,
                    model_ok: orangu::tui::ConnStatus::Pending,
                    is_coordinator: false,
                },
                Vec::new(),
            )
        } else {
            let active_profile = config
                .llms
                .get(&active_model)
                .ok_or_else(|| anyhow!("missing configured server {}", active_model))?;
            probe_header_status(
                &status_http_client,
                tools.workspace(),
                &active_model_id,
                active_profile,
                current_endpoint.as_deref(),
            )
            .await
        };
        // The idle status refresh also re-checks the model: if the server is up
        // but no longer serves the model we are pinned to (e.g. it swapped the
        // loaded model while we sat idle), switch to one it advertises so the
        // header banner reflects the change instead of showing a stale model
        // with a red indicator.
        if let Some(new_model) = idle_model_switch_target(header_status, &available_models) {
            let new_model = new_model.to_string();
            let old_model = std::mem::replace(&mut active_model_id, new_model.clone());
            output_state.push_text(&format!("Switched model from {old_model} to {new_model}"));
            header_status.model_ok = orangu::tui::ConnStatus::Ok;
        }
        let endpoint = current_endpoint.as_deref().unwrap_or("(disconnected)");
        let render = RenderContext {
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
            drop_down: config.drop_down,
            feedback: config.feedback,
            server_names: &server_names,
            available_models: &available_models,
            skills: &skills,
            tab_bar,
            tab_statuses: &tab_statuses,
        };

        // If the active tab has a background streaming task, re-attach to it
        // so the user sees live output and stats, then continue to the next
        // loop iteration (which will re-run the normal prompt-read path).
        if let Some(pr) = pending_response.take() {
            let pr_llm_start = pr.llm_start;
            let pr_tool_time_before = pr.tool_time_before;
            let mut deferred_tab_during_wait: Option<TabAction> = None;
            let pr_result = wait_for_pending_response(
                &mut session,
                pr,
                WaitContext {
                    render: RenderContext {
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
                        drop_down: config.drop_down,
                        feedback: config.feedback,
                        server_names: &server_names,
                        available_models: &available_models,
                        skills: &skills,
                        tab_bar,
                        tab_statuses: &tab_statuses_working,
                    },
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
                    skills: &skills,
                    deferred_tab: &mut deferred_tab_during_wait,
                    parked_tabs: ring.parked(),
                },
                &mut |r, s| _terminal_ui_guard.print_screen(r, s),
            )
            .await;
            match pr_result {
                Ok(WaitResult::Response(answer)) => {
                    let tool_delta = tools
                        .total_tool_duration()
                        .saturating_sub(pr_tool_time_before);
                    usage_stats.record_response(pr_llm_start.elapsed(), &answer, tool_delta);
                    output_state.push_markdown(&answer);
                    if config.feedback {
                        output_state.push_text(FEEDBACK_OK);
                    }
                }
                Ok(WaitResult::Cancelled(partial_output)) => {
                    let tool_delta = tools
                        .total_tool_duration()
                        .saturating_sub(pr_tool_time_before);
                    usage_stats.record_response(
                        pr_llm_start.elapsed(),
                        &partial_output,
                        tool_delta,
                    );
                    preserve_cancelled_output(&mut output_state, &partial_output);
                }
                Ok(WaitResult::Failed { partial, error }) => {
                    let tool_delta = tools
                        .total_tool_duration()
                        .saturating_sub(pr_tool_time_before);
                    usage_stats.record_response(pr_llm_start.elapsed(), &partial, tool_delta);
                    output_state.push_text(&format!("Error: {error:#}"));
                    if config.feedback {
                        output_state.push_text(FEEDBACK_ERR);
                    }
                }
                Ok(WaitResult::BackgroundStreaming(new_pr)) => {
                    pending_response = Some(new_pr);
                }
                Ok(WaitResult::Quit) => {
                    save_all_tabs!();
                    print!("{CLEAR_TERMINAL_SEQUENCE}");
                    std::io::stdout().flush()?;
                    break;
                }
                Err(err) => {
                    let tool_delta = tools
                        .total_tool_duration()
                        .saturating_sub(pr_tool_time_before);
                    usage_stats.record_elapsed(pr_llm_start.elapsed(), tool_delta);
                    output_state.push_text(&format!("Error: {err:#}"));
                    if config.feedback {
                        output_state.push_text(FEEDBACK_ERR);
                    }
                }
            }
            pending_tab_action = deferred_tab_during_wait;
            output_state.reset_scroll();
            continue;
        }

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

        // Collect the startup `/issue` metadata fetch once it finishes, caching
        // the reviewers/assignees/labels for `/issue` completion.
        if issue_meta_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = issue_meta_handle.take()
            && let Ok(metadata) = handle.await
        {
            completion::set_issue_metadata(metadata);
        }

        // Collect the Knowledge Graph scan result once it finishes and push a
        // short summary to the output pane (mirrors the sync-notice pattern).
        if kg_scan_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = kg_scan_handle.take()
        {
            match handle.await {
                Ok(result) => {
                    // Seed the per-file hash table from the initial full scan.
                    kg_file_hashes = result
                        .store
                        .all_nodes()
                        .iter()
                        .map(|n| (n.source_file.clone(), String::new()))
                        .collect();
                    // Write the finished store into the shared Arc so the
                    // `graph_lookup` tool becomes live for the agent.
                    if let Ok(mut guard) = kg_store.lock() {
                        *guard = Some(result.store);
                    }
                    if let Ok(mut status) = kg_status.lock() {
                        *status = orangu::graph::status::GraphBuildStatus::Ready;
                    }
                }
                Err(err) => {
                    output_state.push_text(&format!("Knowledge graph scan failed: {err:#}"));
                    if let Ok(mut status) = kg_status.lock() {
                        *status = orangu::graph::status::GraphBuildStatus::Failed;
                    }
                }
            }
        }

        // Collect finished incremental rescan handles (silent — no UI noise for
        // routine re-parses; the updated store is already live via the Arc).
        if kg_rescan_handle.as_ref().is_some_and(|h| h.is_finished()) {
            kg_rescan_handle = None;
        }

        let resume_left_status = startup_notice_until
            .filter(|&deadline| std::time::Instant::now() < deadline)
            .map(|_| StatusFragment::plain(format!("Resuming session {session_id}")));
        // The branch sync takes priority on the left of the status bar while it
        // runs and for a few seconds after it completes.
        let mut print_screen = |r: RenderContext<'_>, s: ScreenState<'_>| {
            _terminal_ui_guard.print_screen(r, s);
        };

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
                transcript_epoch: output_state.render_epoch(),
                scroll_offset: output_state.scroll_offset(),
                left_status,
                pending_count: pending_commands.len(),
                pending_lines: &[],
                input: input_state.as_str(),
                cursor: input_state.cursor(),
                ghost_index: input_state.ghost_index,
                dropdown: input_state.dropdown.as_ref(),
            },
        );
        std::io::stdout().flush()?;

        // While the startup sync is running or its result is still on the status
        // bar, refresh more often so the status clears promptly when it is done.
        let sync_active = sync_handle.is_some()
            || sync_notice
                .as_ref()
                .is_some_and(|(_, deadline)| std::time::Instant::now() < *deadline);
        let max_idle = if startup_resolution_handle.is_some() {
            // Poll quickly so the pending (white) "Server"/"Model" dots turn
            // green/red the instant the background resolution finishes,
            // rather than sitting idle for a normal refresh cycle.
            std::time::Duration::from_millis(50)
        } else if sync_active {
            std::time::Duration::from_millis(500)
        } else {
            IDLE_STATUS_REFRESH_INTERVAL
        };

        // Enqueue any cron jobs whose minute has passed since the last check;
        // they drain through the same queue as commands typed mid-request.
        schedule::enqueue_due_jobs(&mut schedule_last_minute, &mut pending_commands);

        // Set when this iteration's command came from the scheduler: its `&&`
        // chain id, used to run unattended and to drop the chain's remainder
        // if the command fails.
        let mut current_chain: Option<u64> = None;
        let next_input = if let Some(queued) = pending_commands.pop_front() {
            current_chain = queued.chain;
            queued.text
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
                    skills: &skills,
                },
                &mut print_screen,
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
                InputResult::WorkspacePrevious => {
                    pending_tab_action = Some(TabAction::Previous);
                    continue;
                }
                InputResult::WorkspaceNext => {
                    pending_tab_action = Some(TabAction::Next);
                    continue;
                }
                InputResult::WorkspaceNew => {
                    pending_tab_action = Some(TabAction::New);
                    continue;
                }
                InputResult::WorkspaceClose => {
                    pending_tab_action = Some(TabAction::Close);
                    continue;
                }
                InputResult::Quit => {
                    save_all_tabs!();
                    print!("{CLEAR_TERMINAL_SEQUENCE}");
                    std::io::stdout().flush()?;
                    break;
                }
            }
        };

        startup_notice_until = None;
        print_screen(
            render,
            ScreenState {
                transcript: output_state.lines(),
                transcript_epoch: output_state.render_epoch(),
                scroll_offset: output_state.scroll_offset(),
                left_status: None,
                pending_count: pending_commands.len(),
                pending_lines: &[],
                input: input_state.as_str(),
                cursor: input_state.cursor(),
                ghost_index: input_state.ghost_index,
                dropdown: input_state.dropdown.as_ref(),
            },
        );
        std::io::stdout().flush()?;

        let mut detect_model = false;
        let mut prompt_input = next_input.clone();
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
                session_dir: &session_dir,
                embeddings_server: &embeddings_server,
                is_coordinator: header_status.is_coordinator,
                usage_stats: &usage_stats,
                available_models: &available_models,
                virtual_width: viewport.virtual_width,
                auto_rebase: config.auto_rebase,
                auto_squash: config.auto_squash,
                compile_workers: config.compile_workers,
                compression: config.compression,
                terminal: &config.terminal,
                forge,
                review_reports: git::ReviewReports {
                    review: last_review_report.as_deref(),
                    auto_review: last_auto_review_report.as_deref(),
                },
                skills: &skills,
                semantic_budget_tokens: config.semantic_budget_tokens,
                config_path: &config_path,
            },
        )?;
        // When `/server` (or `/reload`) selects a server, auto-detect an
        // available model on it — exactly like the startup model switch. This
        // runs even when re-selecting the server we are already on. Under a
        // confirmed coordinator, it alone owns every model/role decision, so
        // neither this nor the embeddings re-detection below applies.
        if detect_model {
            let is_coordinator = is_active_connection_a_coordinator(
                &status_http_client,
                &config,
                &active_model,
                current_endpoint.as_deref(),
            )
            .await;
            if !is_coordinator
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
            // Re-detect embeddings capability against whichever connection is
            // now active: switching to (or away from) a coordinator
            // mid-session changes what's actually embeddings-capable, and the
            // startup-only detection would otherwise leave /search
            // permanently disabled (or stuck pointed at a server no longer in
            // use) for the rest of the session.
            embeddings_server =
                detect_embeddings_server(&config, &active_model, is_coordinator).await;
        }
        match command_outcome {
            CommandOutcome::Quit => {
                save_all_tabs!();
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            CommandOutcome::Restart => {
                save_all_tabs!();
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                break;
            }
            CommandOutcome::SwitchSession(target) => {
                save_all_tabs!();
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                resume_override = Some(target);
                break;
            }
            CommandOutcome::SwitchWorkspace(dir) => {
                save_all_tabs!();
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                restart_requested = true;
                workspace_override = Some(dir);
                break;
            }
            CommandOutcome::SwitchWorkspaceTab(index) => {
                pending_tab_action = Some(TabAction::SwitchTo(index));
                continue;
            }
            CommandOutcome::OpenWorkspaceTab(dir) => {
                // If the directory is already open in another tab, focus it rather
                // than creating a duplicate.
                if let Some(index) = ring.position_of(&workspace, &dir) {
                    pending_tab_action = Some(TabAction::SwitchTo(index));
                    continue;
                }
                if let Err(err) = save_session_messages(&session_messages_path, session.messages())
                {
                    output_state.push_text(&format!("Failed to save session: {err:#}"));
                    if config.feedback {
                        output_state.push_text(FEEDBACK_ERR);
                    }
                    output_state.reset_scroll();
                }
                if let Err(err) = update_session_metadata_branch(
                    &session_metadata_path,
                    workspace_branch_name(tools.workspace()).as_deref(),
                ) {
                    output_state.push_text(&format!("Failed to save session metadata: {err:#}"));
                    if config.feedback {
                        output_state.push_text(FEEDBACK_ERR);
                    }
                    output_state.reset_scroll();
                }
                match WorkspaceTab::open(
                    dir,
                    None,
                    true,
                    &system_prompt,
                    config.compression,
                    config.auto_downsample_lines,
                    config.diff_file_cap,
                    TabServerConfig {
                        server: active_model.clone(),
                        model_id: active_model_id.clone(),
                        endpoint: current_endpoint.clone(),
                        llms: &config.llms,
                        slots: slot_registry.clone(),
                    },
                ) {
                    Ok(new_tab) => {
                        let target = ring.open(current_tab!(), new_tab);
                        load_tab!(target);
                        restart_sync_tasks!();
                        output_state.reset_scroll();
                    }
                    Err(err) => {
                        output_state.push_text(&format!("Error: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                        output_state.reset_scroll();
                    }
                }
                continue;
            }
            CommandOutcome::CloseWorkspaceTab => {
                pending_tab_action = Some(TabAction::Close);
                continue;
            }
            CommandOutcome::ChangeWorkspace(dir) => {
                if let Some(index) = ring.position_of(&workspace, &dir) {
                    pending_tab_action = Some(TabAction::SwitchTo(index));
                } else {
                    if let Err(err) =
                        save_session_messages(&session_messages_path, session.messages())
                    {
                        output_state.push_text(&format!("Failed to save session: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                        output_state.reset_scroll();
                    }
                    if let Err(err) = update_session_metadata_branch(
                        &session_metadata_path,
                        workspace_branch_name(tools.workspace()).as_deref(),
                    ) {
                        output_state
                            .push_text(&format!("Failed to save session metadata: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                        output_state.reset_scroll();
                    }
                    match WorkspaceTab::open(
                        dir,
                        None,
                        true,
                        &system_prompt,
                        config.compression,
                        config.auto_downsample_lines,
                        config.diff_file_cap,
                        TabServerConfig {
                            server: active_model.clone(),
                            model_id: active_model_id.clone(),
                            endpoint: current_endpoint.clone(),
                            llms: &config.llms,
                            slots: slot_registry.clone(),
                        },
                    ) {
                        Ok(new_tab) => {
                            load_tab!(new_tab);
                            restart_sync_tasks!();
                            output_state.reset_scroll();
                        }
                        Err(err) => {
                            output_state.push_text(&format!("Error: {err:#}"));
                            if config.feedback {
                                output_state.push_text(FEEDBACK_ERR);
                            }
                            output_state.reset_scroll();
                        }
                    }
                }
                continue;
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
            CommandOutcome::MarkdownOutput(output) => {
                output_state.push_markdown(&output);
                if config.feedback {
                    output_state.push_text(FEEDBACK_OK);
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::OutputWithLlmContext {
                display,
                llm_context,
            } => {
                output_state.push_text(&display);
                if config.feedback {
                    output_state.push_text(FEEDBACK_OK);
                }
                output_state.reset_scroll();
                if current_endpoint.is_some() {
                    session.push_user(&llm_context);
                }
                continue;
            }
            CommandOutcome::OutputError(output) => {
                output_state.push_text(&output);
                if config.feedback {
                    output_state.push_text(FEEDBACK_ERR);
                }
                // A failed scheduled command drops the rest of its `&&`
                // chain, like a shell would.
                if let Some(chain) = current_chain {
                    let dropped = wait::drop_chain_remainder(&mut pending_commands, chain);
                    if dropped > 0 {
                        output_state.push_text(&format!(
                            "Scheduled chain stopped: skipped {dropped} follow-up command{}.",
                            if dropped == 1 { "" } else { "s" }
                        ));
                    }
                }
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::WideOutputWithLlmContext {
                display,
                llm_context,
            } => {
                output_state.push_wide(&display);
                output_state.reset_scroll();
                if current_endpoint.is_some() {
                    session.push_user(&llm_context);
                }
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
                    drop_down: config.drop_down,
                    feedback: config.feedback,
                    server_names: &server_names,
                    available_models: &available_models,
                    skills: &skills,
                    tab_bar,
                    tab_statuses: &tab_statuses_working,
                };
                let mut deferred_tab_during_wait: Option<TabAction> = None;
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
                        skills: &skills,
                        deferred_tab: &mut deferred_tab_during_wait,
                        parked_tabs: ring.parked(),
                    },
                    handle,
                    &mut |r, s| _terminal_ui_guard.print_screen(r, s),
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
                pending_tab_action = deferred_tab_during_wait;
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Duplicates(threshold) => {
                // Run the scan off the UI thread (with a spinner), like Blocking,
                // but keep the typed report: show its console form and cache its
                // Markdown so a later /export duplicates reuses it (no re-scan).
                let root = workspace.to_path_buf();
                let handle =
                    tokio::task::spawn_blocking(move || run_duplicates_scan(&root, threshold));
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
                    drop_down: config.drop_down,
                    feedback: config.feedback,
                    server_names: &server_names,
                    available_models: &available_models,
                    skills: &skills,
                    tab_bar,
                    tab_statuses: &tab_statuses_working,
                };
                let mut deferred_tab_during_wait: Option<TabAction> = None;
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
                        skills: &skills,
                        deferred_tab: &mut deferred_tab_during_wait,
                        parked_tabs: ring.parked(),
                    },
                    handle,
                    &mut |r, s| _terminal_ui_guard.print_screen(r, s),
                )
                .await?;
                match result {
                    Ok(report) => {
                        output_state.push_text(&report.to_console());
                        last_duplicates_report = Some(report);
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
                pending_tab_action = deferred_tab_during_wait;
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Streaming(f, control) => {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                let handle = tokio::task::spawn_blocking(move || f(tx));
                // Kept so the feedback dot can be red when the command was
                // cancelled (`control` itself is moved into the wait loop).
                let cancel_flag = control.as_ref().map(|c| c.cancel.clone());
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
                    drop_down: config.drop_down,
                    feedback: config.feedback,
                    server_names: &server_names,
                    available_models: &available_models,
                    skills: &skills,
                    tab_bar,
                    tab_statuses: &tab_statuses_working,
                };
                let mut deferred_tab_during_wait: Option<TabAction> = None;
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
                        skills: &skills,
                        deferred_tab: &mut deferred_tab_during_wait,
                        parked_tabs: ring.parked(),
                    },
                    handle,
                    &mut rx,
                    control,
                    &mut |r, s| _terminal_ui_guard.print_screen(r, s),
                )
                .await?;
                match result {
                    Ok(()) => {
                        // A cancelled command shows the red dot, not the green one.
                        let cancelled = cancel_flag
                            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                            .unwrap_or(false);
                        if config.feedback {
                            output_state.push_text(if cancelled {
                                FEEDBACK_ERR
                            } else {
                                FEEDBACK_OK
                            });
                        }
                    }
                    Err(err) => {
                        output_state.push_text(&format!("Error: {err:#}"));
                        if config.feedback {
                            output_state.push_text(FEEDBACK_ERR);
                        }
                    }
                }
                pending_tab_action = deferred_tab_during_wait;
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Review(launch) => {
                if header_status.is_coordinator
                    && let Some(endpoint) = current_endpoint.as_deref()
                    && let Some(profile) = review_prompt_profile(
                        &config,
                        &active_model,
                        &active_model_id,
                        endpoint,
                        header_status.is_coordinator,
                    )
                {
                    spawn_coordinator_activation_hint(
                        status_http_client.clone(),
                        profile.endpoint.clone(),
                        profile.model.clone(),
                        profile.api_key.clone(),
                    );
                }
                let mut review = ReviewState::new(launch);
                loop {
                    let chrome = ReviewChrome {
                        current_model: orangu::tui::display_model_name(
                            header_status.is_coordinator,
                            &active_model_id,
                        ),
                        prompt_branch: prompt_branch.as_deref(),
                        pending_count: pending_commands.len(),
                        skills: &skills,
                    };
                    match run_review_mode(
                        &mut review,
                        &mut viewport,
                        &mut input_state,
                        chrome,
                        &workspace,
                        &server_names,
                        &available_models,
                        &mut |args: orangu::tui::ReviewScreenArgs<'_>| {
                            if let Err(err) = _terminal_ui_guard.terminal.draw(|f| {
                                orangu::tui::review_native::draw_review_screen(f, args);
                            }) {
                                eprintln!("failed to draw review screen: {err}");
                            }
                        },
                    )? {
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
                            let Some(prompt_profile) = review_prompt_profile(
                                &config,
                                &active_model,
                                &active_model_id,
                                endpoint,
                                header_status.is_coordinator,
                            ) else {
                                review.feedback = Some(FeedbackWindow {
                                    title,
                                    question,
                                    lines: vec![format!("Error: unknown server '{active_model}'")],
                                    scroll: 0,
                                    x_offset: 0,
                                });
                                continue;
                            };
                            let (prompt, stats) = build_review_prompt_with_stats(
                                &path,
                                &request,
                                &patch,
                                config.compression,
                                config.diff_file_cap,
                                Some(&*tools.compression_store),
                            );
                            if let Ok(mut metrics) = tools.compression_metrics.lock() {
                                metrics.record(&stats);
                            }
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
                                &mut |args: orangu::tui::ReviewScreenArgs<'_>| {
                                    if let Err(err) = _terminal_ui_guard.terminal.draw(|f| {
                                        orangu::tui::review_native::draw_review_screen(f, args);
                                    }) {
                                        eprintln!("failed to draw review screen: {err}");
                                    }
                                },
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
                let (lines, markdown) = review_exit_output(
                    &review.files,
                    &review.comments,
                    &review.general_notes,
                    orangu::tui::display_model_name(header_status.is_coordinator, &active_model_id),
                );
                last_review_report = Some(markdown.clone());
                // Capture the per-comment source code for the export's appendix
                // (the `/show_file` view around each comment).
                last_review_appendix = Some(review::review_export_appendix(
                    &review.files,
                    &review.comments,
                    &review.general_notes,
                    &workspace,
                ));
                last_review_was_auto = false;
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
                let Some(prompt_profile) = review_prompt_profile(
                    &config,
                    &active_model,
                    &active_model_id,
                    &endpoint,
                    header_status.is_coordinator,
                ) else {
                    output_state.push_text(&format!("Error: unknown server '{active_model}'"));
                    output_state.reset_scroll();
                    continue;
                };
                // Hint the coordinator to start swapping to the review role
                // now, in parallel with the prestart screen / per-file setup
                // run_auto_review_mode is about to do, instead of only
                // starting the swap once the first real request fires.
                if header_status.is_coordinator {
                    spawn_coordinator_activation_hint(
                        status_http_client.clone(),
                        prompt_profile.endpoint.clone(),
                        prompt_profile.model.clone(),
                        prompt_profile.api_key.clone(),
                    );
                }
                let chrome = ReviewChrome {
                    current_model: orangu::tui::display_model_name(
                        header_status.is_coordinator,
                        &active_model_id,
                    ),
                    prompt_branch: prompt_branch.as_deref(),
                    pending_count: pending_commands.len(),
                    skills: &skills,
                };

                let state = run_auto_review_mode(
                    launch,
                    &prompt_profile,
                    &mut usage_stats,
                    &mut viewport,
                    chrome,
                    &workspace,
                    &config.terminal,
                    config.feedback,
                    config.compression,
                    config.diff_file_cap,
                    tools.compression_metrics.clone(),
                    tools.compression_store.clone(),
                    &skills,
                    slot_registry.clone(),
                    tools.graph_store.clone(),
                    tools.graph_status.clone(),
                    // A scheduled auto review runs unattended: it starts at
                    // once and returns as soon as the run finishes, so a
                    // chained `export auto review` can pick up the report.
                    current_chain.is_some(),
                    &mut |args: orangu::tui::AutoReviewScreenArgs<'_>| {
                        if let Err(err) = _terminal_ui_guard.terminal.draw(|f| {
                            orangu::tui::auto_review_native::draw_auto_review_screen(f, args);
                        }) {
                            eprintln!("failed to draw auto-review screen: {err}");
                        }
                    },
                )
                .await?;

                // On exit, print the rendered report to the output window and
                // copy its raw Markdown to the system clipboard; the Markdown
                // is also kept for `/comment <n> with auto review`.
                let (lines, clipboard) = auto_review_exit_output(&state);
                last_auto_review_report = Some(clipboard.clone());
                // Capture the per-finding source code for the export's appendix
                // (the `/show_file` view around each finding), reflecting the
                // post-browse report.
                last_auto_review_appendix = Some(state.export_appendix(&workspace));
                last_review_was_auto = true;
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
                pending_tab_action = state.pending_tab;
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::Export(target) => {
                let result = match target {
                    ExportTarget::Console => {
                        export::export_console(&workspace, output_state.lines(), &active_model_id)
                    }
                    ExportTarget::Review => {
                        // Export whichever review ran most recently, falling
                        // back to the other when only one has run, each with its
                        // own source appendix.
                        let auto_appendix = last_auto_review_appendix.as_deref().unwrap_or(&[]);
                        let review_appendix = last_review_appendix.as_deref().unwrap_or(&[]);
                        let auto = last_auto_review_report.as_deref();
                        let interactive = last_review_report.as_deref();
                        let chosen = if last_review_was_auto {
                            auto.map(|report| (report, auto_appendix))
                                .or(interactive.map(|report| (report, review_appendix)))
                        } else {
                            interactive
                                .map(|report| (report, review_appendix))
                                .or(auto.map(|report| (report, auto_appendix)))
                        };
                        match chosen {
                            Some((report, appendix)) => export::export_review(
                                &workspace,
                                report,
                                &active_model_id,
                                appendix,
                            ),
                            None => Err(anyhow!(
                                "No review to export; run /review or /auto_review first"
                            )),
                        }
                    }
                    ExportTarget::AutoReview => match last_auto_review_report.as_deref() {
                        Some(report) => export::export_review(
                            &workspace,
                            report,
                            &active_model_id,
                            last_auto_review_appendix.as_deref().unwrap_or(&[]),
                        ),
                        None => Err(anyhow!("No auto review to export; run /auto_review first")),
                    },
                    ExportTarget::Pr => fetch_pull_request_details(&workspace, forge)
                        .and_then(|prs| export::export_pr(&workspace, &prs, &active_model_id)),
                    ExportTarget::Statistics(total) => {
                        export::export_statistics(&workspace, &active_model_id, total)
                    }
                    ExportTarget::Duplicates => match &last_duplicates_report {
                        // Reuse the report cached by the most recent /duplicates
                        // run — no second scan.
                        Some(report) => {
                            export::export_duplicates(&workspace, report, &active_model_id)
                        }
                        // None yet this session: scan once at the default
                        // threshold so the export still works without a prior
                        // /duplicates, then cache it for any later export.
                        None => {
                            run_duplicates_scan(&workspace, orangu::duplicates::DEFAULT_THRESHOLD)
                                .and_then(|report| {
                                    let path = export::export_duplicates(
                                        &workspace,
                                        &report,
                                        &active_model_id,
                                    );
                                    last_duplicates_report = Some(report);
                                    path
                                })
                        }
                    },
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
                    &mut _terminal_ui_guard,
                )?;
                // The modal view overwrote the screen; the next loop iteration
                // redraws the normal interface from the top.
                output_state.reset_scroll();
                continue;
            }
            CommandOutcome::SkillInvoked { name, prompt } => {
                usage_stats.record_skill(&name);
                prompt_input = prompt;
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
        if header_status.model_ok != orangu::tui::ConnStatus::Ok {
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
        // Under a confirmed coordinator, it alone owns model/role decisions:
        // plain chat always names the conventional `code` role — orangu is
        // fundamentally a coding assistant, so ordinary chat is the `code`
        // role in spirit, while `all` is reserved as the coordinator's
        // required universal fallback. If no dedicated `code` profile is
        // configured, the coordinator falls back to `all` on its own; either
        // way this ignores whatever wire model id orangu.conf (or a prior
        // /model switch) happens to carry.
        let mut prompt_profile = profile.clone();
        prompt_profile.endpoint = endpoint.to_string();
        prompt_profile.model = if header_status.is_coordinator {
            "code".to_string()
        } else {
            active_model_id.clone()
        };

        if let Some((hash, diff)) = orangu::context::world_state::get_current_workspace_diff(
            &workspace,
            config.diff_file_cap,
        )
        .await
            && hash != last_git_diff_hash
        {
            let fragment =
                orangu::context::fragments::ContextualFragment::new("world_state_changes", &diff);
            prompt_input = format!("{}\n\n{}", fragment.render(), prompt_input);
            last_git_diff_hash = hash;

            // Files changed — kick off an incremental KG rescan in the background
            // only if no rescan is already in flight and the initial scan is done.
            if kg_rescan_handle.is_none() && kg_scan_handle.is_none() {
                let rescan_store = kg_store.clone();
                let rescan_workspace = workspace.clone();
                let mut hashes_snapshot = kg_file_hashes.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    rescan_changed_files(&rescan_workspace, &rescan_store, &mut hashes_snapshot)
                });
                kg_rescan_handle = Some(handle);
            }
        }

        let llm_start = std::time::Instant::now();
        let tool_time_before = tools.total_tool_duration();
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let thinking_quote = quote_module.pick(seed);
        let mut deferred_tab_during_wait: Option<TabAction> = None;
        match wait_for_response(
            &mut session,
            &prompt_input,
            &prompt_profile,
            &tools,
            llm_start,
            tool_time_before,
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
                    drop_down: config.drop_down,
                    feedback: config.feedback,
                    server_names: &server_names,
                    available_models: &available_models,
                    skills: &skills,
                    tab_bar,
                    tab_statuses: &tab_statuses_working,
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
                skills: &skills,
                deferred_tab: &mut deferred_tab_during_wait,
                parked_tabs: ring.parked(),
            },
            &mut |r, s| _terminal_ui_guard.print_screen(r, s),
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
                save_all_tabs!();
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            Ok(WaitResult::BackgroundStreaming(pr)) => {
                pending_response = Some(pr);
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
        pending_tab_action = deferred_tab_during_wait;
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

    // Finish every open tab: print one resume line per tab (with its workspace
    // and branch), or clean up its session if it was ephemeral.
    let active_tab = current_tab!();
    active_tab.finish();
    for tab in ring.parked() {
        tab.finish();
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
    pending_commands: Option<&mut VecDeque<crate::wait::PendingCommand>>,
) -> Result<Option<String>> {
    let trimmed = input.trim();
    if trimmed.is_empty() || trimmed.starts_with('\\') {
        return Ok(None);
    }

    history.push(trimmed.to_string());
    append_history_entry(history_path, trimmed)?;

    output_state.push_input(&format!("> {trimmed}"));
    output_state.reset_scroll();

    if trimmed.starts_with('#') {
        return Ok(None);
    }

    if let Some(pending_commands) = pending_commands {
        pending_commands.push_back(crate::wait::PendingCommand::typed(trimmed));
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

    use super::llm_prompt_block_reason;

    use orangu::tui::{ConnStatus, HeaderStatus};

    #[test]
    fn llm_prompt_block_reason_requires_model_connection() {
        assert_eq!(
            llm_prompt_block_reason(
                Some("http://localhost:8100/v1"),
                HeaderStatus {
                    workspace_ok: true,
                    server_ok: ConnStatus::Ok,
                    model_ok: ConnStatus::Failed,
                    is_coordinator: false,
                }
            ),
            None
        );
        assert_eq!(
            llm_prompt_block_reason(
                Some("http://localhost:8100/v1"),
                HeaderStatus {
                    workspace_ok: true,
                    server_ok: ConnStatus::Ok,
                    model_ok: ConnStatus::Ok,
                    is_coordinator: false,
                }
            ),
            None
        );
    }
}

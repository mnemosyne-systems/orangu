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

use crate::commands::build_workspace_system_prompt;
use crate::*;

pub(crate) fn local_command_error(err: Error) -> CommandOutcome {
    if err.is::<LocalError>() {
        CommandOutcome::OutputError(format!("{err}"))
    } else {
        CommandOutcome::OutputError(format!("Error: {err:#}"))
    }
}

/// `/create_file <path> [with <mode>] [containing <text>]`.
///
/// A path that already exists is overwritten — the same on every surface
/// (`orangu::files::overwrite_default`), so the endpoint, the tool and this
/// command agree. The file is written and staged with `git add`; nothing is
/// committed.
fn create_file_command(
    workspace: &std::path::Path,
    args: crate::commands::CreateFileArgs<'_>,
) -> Result<CommandOutcome> {
    file_command(
        orangu::files::create,
        workspace,
        orangu::files::CreateFileRequest {
            path: args.path.to_string(),
            content: args
                .content
                .map(|content| {
                    let mut content = content.to_string();
                    if !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content
                })
                .unwrap_or_default(),
            mode: args
                .mode
                .map(|mode| orangu::files::Mode::Text(mode.to_string())),
            overwrite: orangu::files::overwrite_default(),
            parents: true,
            git: orangu::files::git_default(),
        },
    )
}

/// Run one `orangu::files` operation for a typed command, reporting what it
/// did in one line. These are the same functions the `create_file`/
/// `modify_file`/… tools call and `orangu-server` serves over HTTP, so a
/// typed `/create_file`, a model's tool call, and an API request all land on
/// identical behaviour — workspace-confined, and staged with `git add`/`git
/// mv`/`git rm` in a repository.
fn file_command<Q, R: FileOperationSummary>(
    operation: fn(&std::path::Path, Q) -> orangu::files::FileResult<R>,
    workspace: &std::path::Path,
    request: Q,
) -> Result<CommandOutcome> {
    match operation(workspace, request) {
        Ok(response) => Ok(CommandOutcome::Output(response.summary())),
        Err(err) => Ok(CommandOutcome::OutputError(format!("Error: {err}"))),
    }
}

/// The one line a file command prints on success — what happened, and
/// whether it reached the Git index.
trait FileOperationSummary {
    fn summary(&self) -> String;
}

/// `Staged.` / `Not staged (ignored).` / nothing at all outside a
/// repository — the Git half of a command's one-line report.
fn git_suffix(git: &Option<orangu::git_index::GitOutcome>) -> String {
    match git {
        Some(outcome) if outcome.staged => " (staged)".to_string(),
        Some(outcome) => match (&outcome.skipped, &outcome.error) {
            (Some(reason), _) => format!(" (not staged: {reason})"),
            (None, Some(error)) => format!(" (not staged: {error})"),
            _ => String::new(),
        },
        None => String::new(),
    }
}

impl FileOperationSummary for orangu::files::CreateFileResponse {
    fn summary(&self) -> String {
        let mode = self.mode.as_deref().unwrap_or("-");
        format!("Created {} ({mode}){}", self.path, git_suffix(&self.git))
    }
}

impl FileOperationSummary for orangu::files::DeleteFileResponse {
    fn summary(&self) -> String {
        format!("Deleted {}{}", self.path, git_suffix(&self.git))
    }
}

impl FileOperationSummary for orangu::files::MoveFileResponse {
    fn summary(&self) -> String {
        format!(
            "Moved {} -> {}{}",
            self.from,
            self.to,
            git_suffix(&self.git)
        )
    }
}

impl FileOperationSummary for orangu::files::CreateDirectoryResponse {
    fn summary(&self) -> String {
        let mode = self.mode.as_deref().unwrap_or("-");
        format!("Created directory {} ({mode})", self.path)
    }
}

impl FileOperationSummary for orangu::files::MoveDirectoryResponse {
    fn summary(&self) -> String {
        format!(
            "Moved directory {} -> {}{}",
            self.from,
            self.to,
            git_suffix(&self.git)
        )
    }
}

impl FileOperationSummary for orangu::files::DeleteDirectoryResponse {
    fn summary(&self) -> String {
        format!("Deleted directory {}", self.path)
    }
}

/// Refuse the review when the branch is behind main/master: it would run
/// against stale code, so point at `/rebase` instead. Returns the error outcome
/// to surface, or `None` when the branch is up to date.
fn behind_default_branch_guard(workspace: &Path) -> Option<CommandOutcome> {
    match git::behind_default_branch(workspace) {
        Ok((0, _)) => None,
        Ok((behind, base_ref)) => Some(CommandOutcome::OutputError(format!(
            "The branch is {behind} commit{} behind {base_ref}; run /rebase before reviewing.",
            if behind == 1 { "" } else { "s" }
        ))),
        Err(err) => Some(local_command_error(err)),
    }
}

/// Run a duplicate-code scan over `workspace`, choosing the mode from the Git
/// state: on a branch other than the default (main/master), compare only the
/// functions the branch adds or changes against the whole project; otherwise (on
/// the default branch, or outside a repository) compare the whole project
/// against itself.
pub(crate) fn run_duplicates_scan(
    workspace: &Path,
    threshold: f64,
) -> anyhow::Result<orangu::duplicates::DuplicatesReport> {
    match git::branch_added_lines(workspace) {
        Some(changes) => {
            let mut regions = orangu::duplicates::ChangedRegions::new();
            for (path, ranges) in &changes.files {
                for &(start, end) in ranges {
                    regions.add_range(path.clone(), start, end);
                }
            }
            orangu::duplicates::scan_duplicates_in_patch(
                workspace,
                threshold,
                &regions,
                &changes.base,
            )
        }
        None => orangu::duplicates::scan_duplicates(workspace, threshold),
    }
}

/// Collect the launch data shared by `/review` and `/auto_review`, wrapped in
/// the caller's `CommandOutcome` variant. A review only starts on an
/// up-to-date branch: when the branch is behind main/master the review would
/// run against stale code, so the command refuses and points at `/rebase`.
pub(crate) fn review_outcome(
    workspace: &Path,
    launch_outcome: impl FnOnce(ReviewLaunch) -> CommandOutcome,
) -> CommandOutcome {
    if let Some(refusal) = behind_default_branch_guard(workspace) {
        return refusal;
    }
    match collect_review_diff(workspace) {
        Ok(review) if review.files.is_empty() => CommandOutcome::Output(format!(
            "No changes to review against {}.",
            review.base_label
        )),
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
            launch_outcome(ReviewLaunch {
                files,
                immediate: false,
                deep: false,
            })
        }
        Err(err) => local_command_error(err),
    }
}

/// Launch an `/auto_review` of a single file. On main/master the whole file is
/// reviewed (a full read of its current content); on any other branch only the
/// file's changes against the default branch are reviewed. This mirrors what
/// Tab completion offers for `/auto_review <file>` (every tracked file on
/// main/master, only the changed files on a branch). The report style is
/// identical to a whole-branch run — there is just one file in the checklist.
pub(crate) fn auto_review_file_outcome(
    workspace: &Path,
    file: &str,
    immediate: bool,
    deep: bool,
) -> CommandOutcome {
    let Some(repo_root) = git::discover_git_root(workspace) else {
        return CommandOutcome::OutputError(
            "auto review is only available inside a Git repository".to_string(),
        );
    };
    let on_protected = match git::git_current_branch(&repo_root) {
        Ok(branch) => git::is_protected_branch(&branch),
        Err(err) => return local_command_error(err),
    };

    let entry = if on_protected {
        match full_file_review_entry(workspace, &repo_root, file) {
            Ok(entry) => entry,
            Err(err) => return local_command_error(err),
        }
    } else {
        // On a branch only the file's changes against main/master are reviewed,
        // and only a file that actually changed can be reviewed — the same
        // branch-must-be-rebased guard as a whole-branch run applies.
        if let Some(refusal) = behind_default_branch_guard(workspace) {
            return refusal;
        }
        match collect_review_diff(workspace) {
            Ok(review) => {
                let base_label = review.base_label.clone();
                match review
                    .files
                    .into_iter()
                    .find(|f| review_path_matches(&f.path, file))
                {
                    Some(f) => ReviewEntry {
                        path: f.path,
                        status: ReviewStatus::Unreviewed,
                        diff_lines: f.lines,
                        patch: f.patch,
                    },
                    None => {
                        return CommandOutcome::OutputError(format!(
                            "'{file}' has no changes against {base_label}."
                        ));
                    }
                }
            }
            Err(err) => return local_command_error(err),
        }
    };

    CommandOutcome::AutoReview(ReviewLaunch {
        files: vec![entry],
        immediate,
        deep,
    })
}

/// Launch an `/auto_review all` of every file in the project: a full read of
/// each Git-tracked file's current content (the same treatment
/// `auto_review_file_outcome` gives a single file on main/master), regardless
/// of the current branch — this reviews what is actually on disk, not a diff,
/// so being behind the default branch does not matter. Untracked and
/// gitignored files are never included, since `git ls-files` is the source of
/// the file list.
pub(crate) fn auto_review_all_outcome(
    workspace: &Path,
    immediate: bool,
    deep: bool,
) -> CommandOutcome {
    let Some(repo_root) = git::discover_git_root(workspace) else {
        return CommandOutcome::OutputError(
            "auto review is only available inside a Git repository".to_string(),
        );
    };

    let mut paths = git::git_tracked_files(workspace);
    paths.sort();
    if paths.is_empty() {
        return CommandOutcome::Output("No files to review.".to_string());
    }

    // A file that cannot be read as text (a binary asset, for instance) is
    // silently left out rather than aborting the whole run — with hundreds of
    // tracked files one unreadable one should not sink the batch, and binary
    // assets get no useful review anyway.
    let files: Vec<ReviewEntry> = paths
        .into_iter()
        .filter_map(|path| full_file_review_entry(workspace, &repo_root, &path).ok())
        .collect();
    if files.is_empty() {
        return CommandOutcome::Output("No files to review.".to_string());
    }

    CommandOutcome::AutoReview(ReviewLaunch {
        files,
        immediate,
        deep,
    })
}

/// Whether a changed file's repo-relative `path` matches the user's `arg`: the
/// exact path (what Tab completion fills in), or a trailing path / basename
/// match so a hand-typed bare name like `tui.rs` still resolves.
fn review_path_matches(path: &str, arg: &str) -> bool {
    path == arg
        || path.ends_with(&format!("/{arg}"))
        || Path::new(path).file_name().and_then(|name| name.to_str()) == Some(arg)
}

/// Read a file's current content and present it as an all-added unified diff so
/// a single-file `/auto_review` on main/master reviews the whole file with the
/// same per-category machinery used for a change: with every line marked added,
/// the category prompts treat the entire file as in scope.
fn full_file_review_entry(
    workspace: &Path,
    repo_root: &Path,
    file: &str,
) -> anyhow::Result<ReviewEntry> {
    let candidate = Path::new(file);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        workspace.join(candidate)
    };
    if !absolute.is_file() {
        return Err(LocalError::Usage(format!("No such file '{file}' to review.")).into());
    }
    let content = std::fs::read_to_string(&absolute)
        .map_err(|err| anyhow!("failed to read {file}: {err}"))?;
    // Repo-relative path for the report and prompt headers.
    let rel = absolute
        .strip_prefix(repo_root)
        .unwrap_or(candidate)
        .to_string_lossy()
        .replace('\\', "/");

    let line_count = content.lines().count().max(1);
    let mut patch = format!(
        "diff --git a/{rel} b/{rel}\nnew file mode 100644\n--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{line_count} @@\n"
    );
    let mut diff_lines = Vec::new();
    for line in content.lines() {
        patch.push('+');
        patch.push_str(line);
        patch.push('\n');
        diff_lines.push(format!("+{line}"));
    }

    Ok(ReviewEntry {
        path: rel,
        status: ReviewStatus::Unreviewed,
        diff_lines,
        patch,
    })
}

pub(crate) fn handle_command(
    input: &str,
    state: CommandState<'_>,
    context: CommandContext<'_>,
) -> anyhow::Result<CommandOutcome> {
    let Some(command) = parse_local_command(input) else {
        if let Some(cmd) = input.trim_start().strip_prefix('/') {
            let (name, arguments) = match cmd.split_once(char::is_whitespace) {
                Some((n, a)) => (n, a.trim()),
                None => (cmd, ""),
            };
            if let Some(skill) = context.skills.find(name) {
                return Ok(CommandOutcome::SkillInvoked {
                    name: name.to_string(),
                    prompt: skill.render_activation(arguments),
                });
            }
        }
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
        session_dir,
        embeddings_server,
        is_coordinator,
        usage_stats,
        available_models,
        virtual_width,
        auto_rebase,
        auto_squash,
        compile_workers,
        compression,
        terminal,
        forge,
        review_reports,
        skills,
        semantic_budget_tokens,
        config_path,
    } = context;

    match command {
        LocalCommand::Help => Ok(CommandOutcome::Output(orangu::tui::help_text().to_string())),
        LocalCommand::Skills => {
            let mut out = String::from("Available skills:\n");
            if skills.all().is_empty() {
                out.push_str("  (None discovered)\n");
            } else {
                for skill in skills.all() {
                    out.push_str(&format!(
                        "  /{:<14} {} [{}]\n",
                        skill.name,
                        skill.description.trim(),
                        skill.source
                    ));
                }
            }
            Ok(CommandOutcome::Output(out))
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
            session.clear(&build_workspace_system_prompt(
                profile, skills, workspace, None,
            ));
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
                Ok(output) => {
                    let (context, stats) = orangu::compression::prepare_llm_file_context_with_stats(
                        args.as_ref(),
                        &output,
                        compression,
                        Some(&*tools.compression_store),
                    );
                    if let Ok(mut metrics) = tools.compression_metrics.lock() {
                        metrics.record(&stats);
                    }
                    let mut llm_msg = String::new();
                    llm_msg.push_str(&format!(
                        "The user executed `/show_file {}`. Output:\n\n",
                        args.as_ref()
                    ));
                    if let Some(note) = context.note {
                        llm_msg.push_str(&note);
                        llm_msg.push_str("\n\n");
                    }
                    llm_msg.push_str("```\n");
                    llm_msg.push_str(&context.content);
                    llm_msg.push_str("\n```");

                    Ok(CommandOutcome::WideOutputWithLlmContext {
                        display: output,
                        llm_context: llm_msg,
                    })
                }
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
            if !available_models.is_empty() && !available_models.iter().any(|m| m == name) {
                return Ok(CommandOutcome::OutputError(format!(
                    "Unknown model '{name}'. Available: {}",
                    available_models.join(", ")
                )));
            }
            *active_model_id = name.to_string();
            save_session_settings(session_dir, Some(active_model), Some(active_model_id));
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::Information => {
            let profile = llms[active_model].clone();
            let server_name = active_model.clone();
            let model_id = active_model_id.clone();
            let is_embeddings_server = active_model.as_str() == embeddings_server;
            let graph_status = crate::information::graph_status_label(
                tools
                    .graph_status
                    .lock()
                    .map(|status| *status)
                    .unwrap_or_default(),
            );
            Ok(CommandOutcome::Blocking(Box::new(move || {
                let capabilities = tokio::runtime::Handle::current().block_on(async {
                    crate::information::gather_server_information(&profile, is_embeddings_server)
                        .await
                });
                Ok(crate::information::format_information_table(
                    &server_name,
                    &model_id,
                    graph_status,
                    &capabilities,
                ))
            })))
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
            session.set_system_prompt(&build_workspace_system_prompt(
                profile, skills, workspace, None,
            ));
            save_session_settings(session_dir, Some(active_model), Some(active_model_id));
            // Re-run the startup-style model detection against the selected
            // server, even when it is the server we were already on.
            *detect_model = true;
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::SetTheme(name) => {
            if name.is_empty() {
                return Ok(CommandOutcome::OutputError(format!(
                    "Usage: /theme <name> (available: {})",
                    orangu::tui::Theme::available_session_theme_summary()
                )));
            }
            if matches!(name.trim(), "default" | "global") {
                save_session_theme(session_dir, None);
                let configured_theme = orangu::config::load_client_configuration(config_path)
                    .map(|config| config.theme)
                    .unwrap_or_else(|_| "auto".to_string());
                match orangu::tui::Theme::apply_named(&configured_theme) {
                    Ok(_) => {
                        return Ok(CommandOutcome::Output(format!(
                            "Theme reset to config default ({})",
                            configured_theme
                        )));
                    }
                    Err(err) => return Ok(CommandOutcome::OutputError(err.to_string())),
                }
            }
            match orangu::tui::Theme::apply_named(name) {
                Ok(canonical_name) => {
                    save_session_theme(session_dir, Some(&canonical_name));
                    Ok(CommandOutcome::Output(format!(
                        "Theme set to {}",
                        canonical_name
                    )))
                }
                Err(err) => Ok(CommandOutcome::OutputError(err.to_string())),
            }
        }
        LocalCommand::SetVerbosity(verbosity) => {
            let profile = &llms[active_model];
            let v = if verbosity.is_empty() || verbosity == "default" {
                None
            } else {
                Some(verbosity)
            };
            session.set_system_prompt(&build_workspace_system_prompt(
                profile, skills, workspace, v,
            ));
            Ok(CommandOutcome::Quiet)
        }
        LocalCommand::Diff(None) => match git_workspace_diff(workspace) {
            Ok(output) => {
                let (context, stats) = orangu::compression::prepare_llm_diff_context_with_stats(
                    &output,
                    compression,
                    tools.diff_file_cap(),
                    Some(&*tools.compression_store),
                );
                if let Ok(mut metrics) = tools.compression_metrics.lock() {
                    metrics.record(&stats);
                }
                let mut llm_msg = String::new();
                llm_msg.push_str("The user executed `/diff`. Output:\n\n");
                if let Some(note) = context.note {
                    llm_msg.push_str(&note);
                    llm_msg.push_str("\n\n");
                }
                llm_msg.push_str("```diff\n");
                llm_msg.push_str(&context.content);
                llm_msg.push_str("\n```");
                Ok(CommandOutcome::OutputWithLlmContext {
                    display: output,
                    llm_context: llm_msg,
                })
            }
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Diff(Some(branch)) => match git_diff_against_branch(workspace, &branch) {
            Ok(output) => {
                let (context, stats) = orangu::compression::prepare_llm_diff_context_with_stats(
                    &output,
                    compression,
                    tools.diff_file_cap(),
                    Some(&*tools.compression_store),
                );
                if let Ok(mut metrics) = tools.compression_metrics.lock() {
                    metrics.record(&stats);
                }
                let mut llm_msg = String::new();
                llm_msg.push_str(&format!(
                    "The user executed `/diff {}`. Output:\n\n",
                    branch
                ));
                if let Some(note) = context.note {
                    llm_msg.push_str(&note);
                    llm_msg.push_str("\n\n");
                }
                llm_msg.push_str("```diff\n");
                llm_msg.push_str(&context.content);
                llm_msg.push_str("\n```");
                Ok(CommandOutcome::OutputWithLlmContext {
                    display: output,
                    llm_context: llm_msg,
                })
            }
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Review => Ok(review_outcome(workspace, CommandOutcome::Review)),
        LocalCommand::AutoReview(AutoReviewTarget::Branch, immediate, deep) => {
            Ok(review_outcome(workspace, |mut launch| {
                launch.immediate = immediate;
                launch.deep = deep;
                CommandOutcome::AutoReview(launch)
            }))
        }
        LocalCommand::AutoReview(AutoReviewTarget::File(file), immediate, deep) => Ok(
            auto_review_file_outcome(workspace, file.trim(), immediate, deep),
        ),
        LocalCommand::AutoReview(AutoReviewTarget::All, immediate, deep) => {
            Ok(auto_review_all_outcome(workspace, immediate, deep))
        }
        LocalCommand::Duplicates(threshold) => Ok(CommandOutcome::Duplicates(
            threshold.unwrap_or(orangu::duplicates::DEFAULT_THRESHOLD),
        )),
        LocalCommand::Status => match status_output(workspace) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Grep(None) => Ok(CommandOutcome::OutputError(
            grep_usage_message().to_string(),
        )),
        LocalCommand::Grep(Some(pattern)) => match grep_output(workspace, &pattern) {
            Ok(output) => {
                let (context, stats) = orangu::compression::prepare_llm_grep_context_with_stats(
                    &pattern,
                    &output,
                    compression,
                    Some(&*tools.compression_store),
                );
                if let Ok(mut metrics) = tools.compression_metrics.lock() {
                    metrics.record(&stats);
                }
                let mut llm_msg = String::new();
                llm_msg.push_str(&format!(
                    "The user executed `/grep {}`. Output:\n\n",
                    pattern
                ));
                if let Some(note) = context.note {
                    llm_msg.push_str(&note);
                    llm_msg.push_str("\n\n");
                }
                llm_msg.push_str(&context.content);
                Ok(CommandOutcome::OutputWithLlmContext {
                    display: output,
                    llm_context: llm_msg,
                })
            }
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Search(None) => Ok(CommandOutcome::OutputError(
            "Usage: /search <query>".to_string(),
        )),
        LocalCommand::Search(Some(query)) => {
            // Semantic search is enabled only when an embeddings-capable endpoint
            // was detected at startup; `embeddings_server` holds that server's
            // name, or is empty when none responded.
            if embeddings_server.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    "Semantic /search is unavailable: no embeddings-capable server was detected \
                     at startup. Give a server section `role = embeddings` (or load an embedding \
                     model on your default server) and restart. Meanwhile, use /grep or the \
                     knowledge graph."
                        .to_string(),
                ));
            }
            let Some(profile) = llms.get(embeddings_server) else {
                return Ok(CommandOutcome::OutputError(format!(
                    "The embeddings server '{embeddings_server}' is not a configured server \
                     section."
                )));
            };
            // Under a confirmed coordinator, it alone owns model/role
            // decisions: force `.model = "embeddings"` on the active
            // connection rather than sending whatever it's normally
            // configured with (e.g. "all"), which — since a coordinator
            // matches an exact model/role name before it ever considers the
            // request path — could otherwise route this to the wrong
            // backend entirely.
            let mut embeddings_profile = profile.clone();
            if is_coordinator {
                embeddings_profile.model = "embeddings".to_string();
            }
            let client =
                match orangu::embeddings::EmbeddingClient::from_profile(&embeddings_profile) {
                    Ok(client) => client,
                    Err(err) => {
                        return Ok(CommandOutcome::OutputError(format!(
                            "Could not initialise embeddings client: {err:#}"
                        )));
                    }
                };
            let workspace = workspace.to_path_buf();
            let query = query.into_owned();
            let graph = tools.graph_store.clone();
            // Cancel (double-Esc) and progress are shared with the run loop,
            // which sets the cancel flag and renders the progress percentage and
            // ETA in the status bar.
            let control = crate::commands::StreamControl::new();
            let task_cancel = control.cancel.clone();
            let task_progress = control.progress.clone();
            let task_eta = control.eta.clone();
            Ok(CommandOutcome::Streaming(
                Box::new(move |sink| {
                    // How many hybrid results to surface.
                    const TOP_K: usize = 10;
                    use orangu::tui::format_status_duration;
                    use std::sync::atomic::Ordering;
                    let started = std::time::Instant::now();
                    // The embedding build + query embedding are async; drive them
                    // to completion here, off the UI thread. build_or_update
                    // publishes progress (0–50% analysing, 50–100% embedding) and
                    // a total-time estimate to the shared atomics, appends the
                    // cache incrementally, and checks the cancel flag between
                    // files so a double-Esc stops it promptly.
                    let index = tokio::runtime::Handle::current().block_on(async {
                        orangu::embeddings::EmbeddingIndex::build_or_update(
                            &workspace,
                            &client,
                            &task_cancel,
                            compile_workers,
                            &task_progress,
                            &task_eta,
                        )
                        .await
                    })?;

                    if task_cancel.load(Ordering::Relaxed) {
                        let _ = sink.send("Request cancelled.".to_string());
                        return Ok(());
                    }
                    if index.is_empty() {
                        let _ = sink.send(
                            "The embedding index is empty — no supported source files found to \
                             search."
                                .to_string(),
                        );
                        return Ok(());
                    }
                    let query_vector = tokio::runtime::Handle::current()
                        .block_on(async { client.embed_one(&query).await })?;
                    // Hold the knowledge-graph lock only for the brief,
                    // synchronous ranking step, never across the embedding build.
                    let hits = {
                        let guard = graph
                            .lock()
                            .map_err(|_| anyhow!("graph_store mutex poisoned"))?;
                        index.search(&query_vector, guard.as_ref(), TOP_K, semantic_budget_tokens)
                    };
                    let _ = sink.send(format!(
                        "Searched {} symbols in {}.",
                        index.len(),
                        format_status_duration(started.elapsed())
                    ));
                    let _ = sink.send(orangu::embeddings::format_hits(&query, &hits));
                    Ok(())
                }),
                Some(control),
            ))
        }
        LocalCommand::Log(count) => match log_output(workspace, count) {
            Ok(output) => Ok(CommandOutcome::Output(output)),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Show(commit) => match show_output(workspace, commit.as_deref()) {
            Ok(output) => {
                let (context, stats) = orangu::compression::prepare_llm_diff_context_with_stats(
                    &output,
                    compression,
                    tools.diff_file_cap(),
                    Some(&*tools.compression_store),
                );
                if let Ok(mut metrics) = tools.compression_metrics.lock() {
                    metrics.record(&stats);
                }
                let mut llm_msg = String::new();
                match commit.as_deref() {
                    Some(commit) => llm_msg.push_str(&format!(
                        "The user executed `/show {}`. Output:\n\n",
                        commit
                    )),
                    None => llm_msg.push_str("The user executed `/show`. Output:\n\n"),
                }
                if let Some(note) = context.note {
                    llm_msg.push_str(&note);
                    llm_msg.push_str("\n\n");
                }
                llm_msg.push_str("```diff\n");
                llm_msg.push_str(&context.content);
                llm_msg.push_str("\n```");
                Ok(CommandOutcome::OutputWithLlmContext {
                    display: output,
                    llm_context: llm_msg,
                })
            }
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::Fetch(remote) => match fetch_output(workspace, remote.as_deref(), forge) {
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
            match comment_output(workspace, issue_number, &body, review_reports, forge) {
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
        LocalCommand::Issue(None) => Ok(CommandOutcome::OutputError(
            issue_usage_message().to_string(),
        )),
        LocalCommand::Issue(Some(action)) => match issue_field_output(workspace, &action, forge) {
            Ok(()) => Ok(CommandOutcome::Quiet),
            Err(err) => Ok(local_command_error(err)),
        },
        LocalCommand::GetComments(None) => Ok(CommandOutcome::OutputError(
            get_comments_usage_message().to_string(),
        )),
        LocalCommand::GetComments(Some(target)) => {
            match get_comments_output(workspace, &target, forge) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::CreatePullRequest => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Blocking(Box::new(move || {
                create_pull_request_output(&ws, auto_rebase, auto_squash, forge)
            })))
        }
        LocalCommand::Rebase(target) => match rebase_output(workspace, target.as_deref(), forge) {
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
        LocalCommand::CreateFile(None) => Ok(CommandOutcome::OutputError(
            create_file_usage_message().to_string(),
        )),
        LocalCommand::CreateFile(Some(args)) => create_file_command(workspace, args),
        LocalCommand::DeleteFile(None) => Ok(CommandOutcome::OutputError(
            delete_file_usage_message().to_string(),
        )),
        LocalCommand::DeleteFile(Some(path)) => file_command(
            orangu::files::delete,
            workspace,
            orangu::files::DeleteFileRequest {
                path: path.to_string(),
                git: orangu::files::git_default(),
            },
        ),
        LocalCommand::MoveFile(None) => Ok(CommandOutcome::OutputError(
            move_file_usage_message().to_string(),
        )),
        LocalCommand::MoveFile(Some((src, dst))) => file_command(
            orangu::files::move_,
            workspace,
            orangu::files::MoveFileRequest {
                from: src.to_string(),
                to: dst.to_string(),
                mode: None,
                overwrite: false,
                parents: true,
                git: orangu::files::git_default(),
            },
        ),
        LocalCommand::CreateDirectory(None) => Ok(CommandOutcome::OutputError(
            "Usage: /create_directory <path> [with <mode>]".to_string(),
        )),
        LocalCommand::CreateDirectory(Some((path, mode))) => file_command(
            orangu::files::create_dir,
            workspace,
            orangu::files::CreateDirectoryRequest {
                path: path.to_string(),
                mode: mode.map(|mode| orangu::files::Mode::Text(mode.to_string())),
                parents: true,
                git: orangu::files::git_default(),
            },
        ),
        LocalCommand::MoveDirectory(None) => Ok(CommandOutcome::OutputError(
            "Usage: /move_directory <from> <to>".to_string(),
        )),
        LocalCommand::MoveDirectory(Some((src, dst))) => file_command(
            orangu::files::move_dir,
            workspace,
            orangu::files::MoveDirectoryRequest {
                from: src.to_string(),
                to: dst.to_string(),
                mode: None,
                parents: true,
                git: orangu::files::git_default(),
            },
        ),
        LocalCommand::DeleteDirectory(None) => Ok(CommandOutcome::OutputError(
            "Usage: /delete_directory <path>".to_string(),
        )),
        LocalCommand::DeleteDirectory(Some(path)) => file_command(
            orangu::files::delete_dir,
            workspace,
            orangu::files::DeleteDirectoryRequest {
                path: path.to_string(),
                git: orangu::files::git_default(),
            },
        ),
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
        LocalCommand::Bisect(sub) => {
            // BisectSubcommand<'a> borrows from the input string; materialise to
            // owned data so the closure satisfies the 'static bound on Blocking.
            enum BisectOp {
                Start(Option<String>),
                Good(Option<String>),
                Bad(Option<String>),
                Skip(Option<String>),
                Reset,
                Log,
                Status,
            }
            let op = match sub {
                BisectSubcommand::Start(a) => BisectOp::Start(a.map(|c| c.into_owned())),
                BisectSubcommand::Good(c) => BisectOp::Good(c.map(|c| c.into_owned())),
                BisectSubcommand::Bad(c) => BisectOp::Bad(c.map(|c| c.into_owned())),
                BisectSubcommand::Skip(c) => BisectOp::Skip(c.map(|c| c.into_owned())),
                BisectSubcommand::Reset => BisectOp::Reset,
                BisectSubcommand::Log => BisectOp::Log,
                BisectSubcommand::Status => BisectOp::Status,
            };
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Blocking(Box::new(move || match op {
                BisectOp::Start(a) => bisect_start_output(&ws, a.as_deref()),
                BisectOp::Good(c) => bisect_good_output(&ws, c.as_deref()),
                BisectOp::Bad(c) => bisect_bad_output(&ws, c.as_deref()),
                BisectOp::Skip(c) => bisect_skip_output(&ws, c.as_deref()),
                BisectOp::Reset => bisect_reset_output(&ws),
                BisectOp::Log => bisect_log_output(&ws),
                BisectOp::Status => bisect_status_output(&ws),
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
        LocalCommand::Session(Some(arg)) => {
            if arg == usage_stats.session_id {
                return Ok(CommandOutcome::Output(format!("Already in session {arg}")));
            }
            // A bare name (no path separators) matching an existing session
            // directory is a session UUID: switch to it.
            let is_session_id = !arg.contains('/')
                && !arg.contains('\\')
                && matches!(session_dir_path(&arg), Ok(path) if path.is_dir());
            if is_session_id {
                return Ok(CommandOutcome::SwitchSession(arg.into_owned()));
            }
            // Otherwise treat the argument as a workspace.
            let matches = sessions_matching_workspace(arg.as_ref())?;
            match matches.as_slice() {
                // A workspace that uniquely identifies one session switches to it.
                [uuid] => {
                    if *uuid == usage_stats.session_id {
                        return Ok(CommandOutcome::Output(format!("Already in session {uuid}")));
                    }
                    return Ok(CommandOutcome::SwitchSession(uuid.clone()));
                }
                // No session uses this workspace yet: if the argument resolves to
                // a real directory on disk (with `~` expanded), open it as a new
                // workspace; otherwise fall through to the empty listing.
                [] => {
                    if let Some(dir) = resolve_existing_dir_arg(arg.as_ref()) {
                        return Ok(CommandOutcome::SwitchWorkspace(dir));
                    }
                }
                // Several sessions share the workspace: list them so the user can
                // pick a UUID.
                _ => {}
            }
            match list_sessions_output(Some(arg.as_ref()), &usage_stats.session_id) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::Workspace(None) => Ok(CommandOutcome::Output(format!(
            "Active workspace: {}",
            workspace.display()
        ))),
        LocalCommand::Workspace(Some(arg)) => {
            let arg = arg.trim();
            // A bare integer is a tab number ("number is the tab, everything
            // else is a directory"); switch to that tab.
            if let Ok(number) = arg.parse::<usize>() {
                if number == 0 {
                    return Ok(CommandOutcome::OutputError(
                        "Workspace numbers start at 1.".to_string(),
                    ));
                }
                return Ok(CommandOutcome::SwitchWorkspaceTab(number - 1));
            }
            // Otherwise the argument is a directory: switch the current tab's
            // workspace to it in-place, or switch to an existing tab if open.
            match resolve_existing_dir_arg(arg) {
                Some(dir) => Ok(CommandOutcome::ChangeWorkspace(dir)),
                None => Ok(CommandOutcome::OutputError(format!(
                    "No such directory: {arg}"
                ))),
            }
        }
        LocalCommand::Prune(None) => Ok(CommandOutcome::OutputError(
            prune_usage_message().to_string(),
        )),
        LocalCommand::Prune(Some(target)) => {
            match prune_sessions_output(&target, &usage_stats.session_id) {
                Ok(output) => Ok(CommandOutcome::Output(output)),
                Err(err) => Ok(local_command_error(err)),
            }
        }
        LocalCommand::CreateWorkspace(dir) => {
            let dir = dir.trim();
            if dir.is_empty() {
                return Ok(CommandOutcome::OutputError(
                    "Usage: /create_workspace <directory>".to_string(),
                ));
            }
            match resolve_existing_dir_arg(dir) {
                Some(path) => Ok(CommandOutcome::OpenWorkspaceTab(path)),
                None => Ok(CommandOutcome::OutputError(format!(
                    "No such directory: {dir}"
                ))),
            }
        }
        LocalCommand::DeleteWorkspace => Ok(CommandOutcome::CloseWorkspaceTab),
        LocalCommand::Export(target) => Ok(CommandOutcome::Export(target)),
        LocalCommand::Manual => Ok(CommandOutcome::Manual),
        LocalCommand::Usage => Ok(CommandOutcome::Output(usage_stats.format(tools))),
        LocalCommand::Statistics(total) => Ok(CommandOutcome::Output(
            crate::activity_log::format_report(workspace, total),
        )),
        LocalCommand::Schedule => Ok(CommandOutcome::Output(
            crate::schedule::format_schedule_list(),
        )),
        LocalCommand::Build(request) => {
            let ws = workspace.to_path_buf();
            Ok(CommandOutcome::Streaming(
                Box::new(move |sink| {
                    build::build_output(
                        &ws,
                        request.profile,
                        request.target.as_deref(),
                        compile_workers,
                        &sink,
                    )
                }),
                None,
            ))
        }
        LocalCommand::Shell(None) => Ok(CommandOutcome::OutputError(
            "Usage: /shell <command>".to_string(),
        )),
        LocalCommand::Shell(Some(command)) => {
            let ws = workspace.to_path_buf();
            let command = command.into_owned();
            Ok(CommandOutcome::Streaming(
                Box::new(move |sink| shell_command::shell_output(&ws, &command, &sink)),
                None,
            ))
        }
        LocalCommand::Clear => {
            let prompt = build_workspace_system_prompt(
                llms.get(active_model)
                    .ok_or_else(|| anyhow!("unknown server '{active_model}'"))?,
                skills,
                workspace,
                Some("terse"),
            );
            session.clear(&prompt);
            Ok(CommandOutcome::Cleared)
        }
        LocalCommand::Quit => Ok(CommandOutcome::Quit),
        LocalCommand::PendingList => Ok(CommandOutcome::PendingList),
        LocalCommand::PendingDelete(None) => Ok(CommandOutcome::Output(
            "Usage: /pending delete <number>. Use /pending to list.".to_string(),
        )),
        LocalCommand::PendingDelete(Some(index)) => Ok(CommandOutcome::PendingDelete(index)),
        LocalCommand::Graph => {
            let guard = tools
                .graph_store
                .lock()
                .map_err(|_| anyhow::anyhow!("graph store mutex poisoned"))?;
            match &*guard {
                None => Ok(CommandOutcome::OutputError(
                    "Knowledge Graph is still being built — please wait a moment and try again."
                        .to_string(),
                )),
                Some(store) => {
                    let repo_name = crate::export::repository_display(workspace);
                    let repo_name = crate::export::sanitize(&repo_name);
                    let branch_name = crate::export::branch_display(workspace);
                    let branch_name = crate::export::sanitize(&branch_name);
                    match orangu::graph::html::write_html(
                        store,
                        workspace,
                        &repo_name,
                        &branch_name,
                    ) {
                        Ok(path) => {
                            let stats = store.stats();
                            let file_url = format!("file://{}", path.display());
                            let path_display =
                                path.file_name().unwrap_or_default().to_string_lossy();
                            Ok(CommandOutcome::MarkdownOutput(format!(
                                "Knowledge Graph written to: [{path_display}]({file_url}) ({} nodes / {} edges)",
                                stats.node_count, stats.edge_count
                            )))
                        }
                        Err(err) => Ok(CommandOutcome::OutputError(format!(
                            "Failed to generate graph: {err:#}"
                        ))),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use std::collections::HashMap;
    use std::fs;
    use tempfile::tempdir;

    /// The typed commands run the same `orangu::files` operations the tools
    /// and the server's endpoints do, and report what reached the index.
    #[test]
    fn file_commands_run_the_shared_operations_and_report_staging() {
        let workspace = tempdir().expect("workspace");
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "T"],
        ] {
            std::process::Command::new("git")
                .arg("-C")
                .arg(workspace.path())
                .args(args)
                .output()
                .expect("git setup");
        }
        let create = |path: &str, content: Option<&str>| {
            create_file_command(
                workspace.path(),
                crate::commands::CreateFileArgs {
                    path: std::borrow::Cow::Borrowed(path),
                    mode: Some(std::borrow::Cow::Borrowed("0644")),
                    content: content.map(std::borrow::Cow::Borrowed),
                },
            )
            .expect("create")
        };

        match create("notes.md", Some("first")) {
            CommandOutcome::Output(message) => {
                assert!(message.starts_with("Created notes.md (0644)"), "{message}");
                assert!(message.contains("(staged)"), "{message}");
            }
            _ => panic!("expected an output line"),
        }
        assert_eq!(
            fs::read_to_string(workspace.path().join("notes.md")).unwrap(),
            "first\n"
        );

        // A path that already exists is overwritten: the typed command is an
        // override, unlike the endpoint and the tool.
        create("notes.md", Some("second"));
        assert_eq!(
            fs::read_to_string(workspace.path().join("notes.md")).unwrap(),
            "second\n"
        );

        // Outside the workspace is refused here exactly as it is everywhere
        // else, and reports the failure rather than the change.
        match file_command(
            orangu::files::create,
            workspace.path(),
            orangu::files::CreateFileRequest {
                path: "../escaped.md".to_string(),
                content: String::new(),
                mode: None,
                overwrite: false,
                parents: true,
                git: orangu::files::git_default(),
            },
        )
        .expect("call")
        {
            CommandOutcome::OutputError(message) => {
                assert!(message.contains("workspace"), "{message}");
            }
            _ => panic!("expected an error outcome"),
        }
    }

    #[test]
    fn review_path_matches_accepts_exact_paths_and_bare_names() {
        // The full repo-relative path (what Tab completion fills in) matches...
        assert!(review_path_matches("src/tui.rs", "src/tui.rs"));
        // ...as does a bare basename or a trailing path segment.
        assert!(review_path_matches("src/tui.rs", "tui.rs"));
        assert!(review_path_matches("a/b/tui.rs", "b/tui.rs"));
        // A different file does not match.
        assert!(!review_path_matches("src/tui.rs", "main.rs"));
        assert!(!review_path_matches("src/tui.rs", "ui.rs"));
    }

    #[test]
    fn auto_review_file_on_main_reviews_the_whole_file() {
        let _env_lock = crate::process_env_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home = crate::git::EnvVarGuard::set_path("HOME", home.path());
        crate::git::init_git_for_test(workspace.path());
        crate::git::git_run(workspace.path(), &["checkout", "-B", "main"]);
        fs::create_dir(workspace.path().join("src")).expect("src dir");
        fs::write(
            workspace.path().join("src/tui.rs"),
            "fn main() {}\nlet x = 1;\n",
        )
        .expect("file");
        crate::git::git_run(workspace.path(), &["add", "."]);
        crate::git::git_run(workspace.path(), &["commit", "-m", "base"]);

        // On main the whole file is reviewed: a one-file launch whose patch is
        // the file content as an all-added diff.
        match auto_review_file_outcome(workspace.path(), "src/tui.rs", false, false) {
            CommandOutcome::AutoReview(launch) => {
                assert_eq!(launch.files.len(), 1);
                let entry = &launch.files[0];
                assert_eq!(entry.path, "src/tui.rs");
                assert!(
                    entry
                        .patch
                        .starts_with("diff --git a/src/tui.rs b/src/tui.rs")
                );
                assert!(entry.patch.contains("+fn main() {}"));
                assert!(entry.patch.contains("+let x = 1;"));
            }
            _ => panic!("expected an AutoReview outcome for a file on main"),
        }

        // An unknown file is refused.
        assert!(matches!(
            auto_review_file_outcome(workspace.path(), "src/missing.rs", false, false),
            CommandOutcome::OutputError(_)
        ));
    }

    #[test]
    fn auto_review_file_on_branch_reviews_only_the_change() {
        let _env_lock = crate::process_env_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home = crate::git::EnvVarGuard::set_path("HOME", home.path());
        crate::git::init_git_for_test(workspace.path());
        crate::git::git_run(workspace.path(), &["checkout", "-B", "main"]);
        fs::write(workspace.path().join("README.md"), "one\ntwo\n").expect("file");
        crate::git::git_run(workspace.path(), &["add", "."]);
        crate::git::git_run(workspace.path(), &["commit", "-m", "base"]);

        crate::git::git_run(workspace.path(), &["checkout", "-b", "feature/x"]);
        fs::write(workspace.path().join("README.md"), "one\ntwo\nthree\n").expect("edit");
        crate::git::git_run(workspace.path(), &["commit", "-am", "edit"]);

        // The changed file is reviewed against the merge base: the patch shows
        // only the added line, not the whole file.
        match auto_review_file_outcome(workspace.path(), "README.md", false, false) {
            CommandOutcome::AutoReview(launch) => {
                assert_eq!(launch.files.len(), 1);
                let patch = &launch.files[0].patch;
                assert!(patch.contains("+three"), "{patch:?}");
                assert!(!patch.contains("+one"), "{patch:?}");
            }
            _ => panic!("expected an AutoReview outcome for a changed file on a branch"),
        }

        // A file with no changes on the branch is refused.
        fs::write(workspace.path().join("other.txt"), "x\n").expect("other");
        assert!(matches!(
            auto_review_file_outcome(workspace.path(), "other.txt", false, false),
            CommandOutcome::OutputError(_)
        ));
    }

    #[test]
    fn auto_review_all_reviews_every_tracked_file_only() {
        let _env_lock = crate::process_env_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let workspace = tempdir().expect("workspace");
        let home = tempdir().expect("home");
        let _home = crate::git::EnvVarGuard::set_path("HOME", home.path());
        crate::git::init_git_for_test(workspace.path());
        crate::git::git_run(workspace.path(), &["checkout", "-B", "main"]);
        fs::write(workspace.path().join("README.md"), "one\ntwo\n").expect("file");
        fs::write(workspace.path().join(".gitignore"), "ignored.txt\n").expect("gitignore");
        crate::git::git_run(workspace.path(), &["add", "."]);
        crate::git::git_run(workspace.path(), &["commit", "-m", "base"]);

        // Neither an untracked file nor one excluded by .gitignore is committed
        // yet, so `all` must leave both out — only what `git` tracks is reviewed.
        fs::write(workspace.path().join("untracked.rs"), "fn x() {}\n").expect("untracked");
        fs::write(workspace.path().join("ignored.txt"), "secret\n").expect("ignored");

        match auto_review_all_outcome(workspace.path(), false, false) {
            CommandOutcome::AutoReview(launch) => {
                let paths: Vec<&str> = launch.files.iter().map(|f| f.path.as_str()).collect();
                assert!(paths.contains(&"README.md"), "{paths:?}");
                assert!(paths.contains(&".gitignore"), "{paths:?}");
                assert!(!paths.contains(&"untracked.rs"), "{paths:?}");
                assert!(!paths.contains(&"ignored.txt"), "{paths:?}");

                // Each tracked file is reviewed whole, as an all-added diff.
                let readme = launch
                    .files
                    .iter()
                    .find(|f| f.path == "README.md")
                    .expect("README.md entry");
                assert!(readme.patch.contains("+one"), "{:?}", readme.patch);
                assert!(readme.patch.contains("+two"), "{:?}", readme.patch);
            }
            _ => panic!("expected an AutoReview outcome for `all`"),
        }
    }

    #[test]
    fn open_file_failure_returns_output_instead_of_error() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(
            outcome,
            CommandOutcome::OutputError(message) if message.starts_with("Error: ")
        ));
    }

    #[test]
    fn explicit_skill_invocation_overrides_the_prompt() {
        let workspace = tempdir().expect("workspace");
        let skill_dir = workspace.path().join(".agents/skills/code-review");
        std::fs::create_dir_all(skill_dir.join("references")).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: Review code\n---\nReview focus: $ARGUMENTS\n",
        )
        .expect("skill");
        std::fs::write(skill_dir.join("references/checklist.md"), "checklist").expect("resource");
        let skills = orangu::skills::SkillRegistry::discover(workspace.path());
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
        )]);
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/code-review auth",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &skills,
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        match outcome {
            CommandOutcome::SkillInvoked { name, prompt } => {
                assert_eq!(name, "code-review");
                assert!(prompt.contains("<skill_content name=\"code-review\">"));
                assert!(prompt.contains("Review focus: auth"));
                assert!(prompt.contains("<file>references/checklist.md</file>"));
            }
            _ => panic!("expected an overridden prompt"),
        }
    }

    #[test]
    fn clear_keeps_skill_catalog_in_the_system_prompt() {
        let workspace = tempdir().expect("workspace");
        let skill_dir = workspace.path().join(".agents/skills/code-review");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: Review code\n---\nReview focus: $ARGUMENTS\n",
        )
        .expect("skill");
        let skills = orangu::skills::SkillRegistry::discover(workspace.path());
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
        )]);
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new(&build_workspace_system_prompt(
            &llms["llama"],
            &skills,
            Path::new(""),
            None,
        ));

        let outcome = handle_command(
            "/clear",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &skills,
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Cleared));
        let system_message = session.messages().first().expect("system message");
        assert!(system_message.content.contains("<available_skills>"));
        assert!(system_message.content.contains("<name>code-review</name>"));
    }

    #[test]
    fn workspace_command_switches_by_number_and_path() {
        let other = tempdir().expect("other workspace");
        // Bind the TempDir so the directory survives for the whole test; only
        // its normalized path is needed for the comparisons below.
        let here_dir = tempdir().expect("workspace");
        let here = crate::normalize_path(here_dir.path());

        // Run `input` with `here` as the active workspace.
        let run = |input: &str| -> CommandOutcome {
            let llms = HashMap::from([(
                "llama".to_string(),
                test_profile("http://localhost:8100/v1", "gemma"),
            )]);
            let tools = ToolExecutor::new(&here);
            let mut active_model = "llama".to_string();
            let mut active_model_id = "gemma".to_string();
            let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
            let mut session = ChatSession::new("system");
            handle_command(
                input,
                CommandState {
                    active_model: &mut active_model,
                    active_model_id: &mut active_model_id,
                    current_endpoint: &mut current_endpoint,
                    session: &mut session,
                    detect_model: &mut false,
                },
                CommandContext {
                    skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: &here,
                    session_dir: &here,
                    embeddings_server: "",
                    is_coordinator: false,
                    usage_stats: &super::UsageStats::new(),
                    available_models: &[],
                    virtual_width: 512,
                    auto_rebase: false,
                    auto_squash: false,
                    compile_workers: 1,
                    compression: false,
                    terminal: "",
                    forge: crate::git::Forge::GitHub,
                    semantic_budget_tokens: 16384,
                    config_path: &here,
                    review_reports: crate::git::ReviewReports::default(),
                },
            )
            .expect("handle command")
        };

        // No argument reports the active workspace.
        match run("/workspace") {
            CommandOutcome::Output(message) => assert!(message.contains("Active workspace")),
            _ => panic!("expected the active workspace to be reported"),
        }

        // A number switches to that tab (0-based index); the loop resolves
        // whether the tab exists. Zero is rejected up front.
        assert!(matches!(
            run("/workspace 1"),
            CommandOutcome::SwitchWorkspaceTab(0)
        ));
        assert!(matches!(
            run("/workspace 5"),
            CommandOutcome::SwitchWorkspaceTab(4)
        ));
        assert!(matches!(
            run("/workspace 0"),
            CommandOutcome::OutputError(_)
        ));

        // An existing directory switches the current tab's workspace in-place.
        match run(&format!("/workspace {}", other.path().display())) {
            CommandOutcome::ChangeWorkspace(dir) => {
                assert_eq!(dir, crate::normalize_path(other.path()));
            }
            _ => panic!("expected a workspace change for a directory"),
        }

        // A path that is not a directory is rejected.
        match run("/workspace /no/such/orangu/dir") {
            CommandOutcome::OutputError(message) => assert!(message.contains("No such directory")),
            _ => panic!("expected an error for a missing directory"),
        }
    }

    #[test]
    fn create_workspace_and_delete_workspace_commands() {
        let other = tempdir().expect("other workspace");
        let here_dir = tempdir().expect("workspace");
        let here = crate::normalize_path(here_dir.path());

        let run = |input: &str| -> CommandOutcome {
            let llms = HashMap::from([(
                "llama".to_string(),
                test_profile("http://localhost:8100/v1", "gemma"),
            )]);
            let tools = ToolExecutor::new(&here);
            let mut active_model = "llama".to_string();
            let mut active_model_id = "gemma".to_string();
            let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
            let mut session = ChatSession::new("system");
            handle_command(
                input,
                CommandState {
                    active_model: &mut active_model,
                    active_model_id: &mut active_model_id,
                    current_endpoint: &mut current_endpoint,
                    session: &mut session,
                    detect_model: &mut false,
                },
                CommandContext {
                    skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: &here,
                    session_dir: &here,
                    embeddings_server: "",
                    is_coordinator: false,
                    usage_stats: &super::UsageStats::new(),
                    available_models: &[],
                    virtual_width: 512,
                    auto_rebase: false,
                    auto_squash: false,
                    compile_workers: 1,
                    compression: false,
                    terminal: "",
                    forge: crate::git::Forge::GitHub,
                    semantic_budget_tokens: 16384,
                    config_path: &here,
                    review_reports: crate::git::ReviewReports::default(),
                },
            )
            .expect("handle command")
        };

        // An existing directory opens a new workspace tab.
        match run(&format!("/create_workspace {}", other.path().display())) {
            CommandOutcome::OpenWorkspaceTab(dir) => {
                assert_eq!(dir, crate::normalize_path(other.path()));
            }
            _ => panic!("expected OpenWorkspaceTab for an existing directory"),
        }

        // Natural-language form works too.
        match run(&format!("create workspace {}", other.path().display())) {
            CommandOutcome::OpenWorkspaceTab(dir) => {
                assert_eq!(dir, crate::normalize_path(other.path()));
            }
            _ => panic!("expected OpenWorkspaceTab for natural-language create"),
        }

        // A non-existent directory is rejected.
        assert!(matches!(
            run("/create_workspace /no/such/orangu/dir"),
            CommandOutcome::OutputError(_)
        ));

        // Bare `/create_workspace` (no directory) shows a usage error.
        assert!(matches!(
            run("/create_workspace"),
            CommandOutcome::OutputError(_)
        ));

        // `/delete_workspace` closes the current tab.
        assert!(matches!(
            run("/delete_workspace"),
            CommandOutcome::CloseWorkspaceTab
        ));

        // Natural-language form closes the current tab.
        assert!(matches!(
            run("delete workspace"),
            CommandOutcome::CloseWorkspaceTab
        ));

        // `/delete <branch>` still routes to branch deletion, not workspace close.
        assert!(matches!(
            run("/delete some-feature-branch"),
            CommandOutcome::Output(_)
                | CommandOutcome::OutputError(_)
                | CommandOutcome::Blocking(_)
        ));
    }

    #[test]
    fn missing_required_command_arguments_return_usage_output() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
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
                    skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: workspace.path(),
                    session_dir: workspace.path(),
                    embeddings_server: "",
                    is_coordinator: false,
                    usage_stats: &super::UsageStats::new(),
                    available_models: &[],
                    virtual_width: 512,
                    auto_rebase: false,
                    auto_squash: false,
                    compile_workers: 1,
                    compression: false,
                    terminal: "",
                    forge: crate::git::Forge::GitHub,
                    semantic_budget_tokens: 16384,
                    config_path: workspace.path(),
                    review_reports: crate::git::ReviewReports::default(),
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
            test_profile("http://localhost:8100/v1", "gemma"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::Output(output) if output == tree));
    }

    #[test]
    fn set_server_switches_active_endpoint() {
        const GEMMA: &str = "gemma-4-E4B-it-GGUF";
        const OPENAI: &str = "gpt-4.1";

        let llms = HashMap::from([
            (
                GEMMA.to_string(),
                test_profile("http://localhost:8100/v1", "ggml-org/gemma-4-E4B-it-GGUF"),
            ),
            (
                OPENAI.to_string(),
                test_profile("https://api.openai.com/v1", "gpt-4.1"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: GEMMA,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
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
            test_profile("http://localhost:8100/v1", "ggml-org/gemma-4-E4B-it-GGUF"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: GEMMA,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
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
    fn set_model_rejects_unknown_model_name() {
        const SERVER: &str = "local";

        let llms = HashMap::from([(
            SERVER.to_string(),
            test_profile("http://localhost:8100/v1", "model-a"),
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
            "/model unicorn-v99",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: SERVER,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &available,
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(
            matches!(&outcome, CommandOutcome::OutputError(msg) if msg.contains("Unknown model")),
            "expected OutputError with 'Unknown model'"
        );
        // The active model ID should remain unchanged.
        assert_eq!(active_model_id, "model-a");
    }

    #[test]
    fn model_info_marks_active_green_and_others_red() {
        const SERVER: &str = "local";

        let llms = HashMap::from([(
            SERVER.to_string(),
            test_profile("http://localhost:8100/v1", "model-a"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: SERVER,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &available,
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
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
                test_profile("http://localhost:8100/v1", "model-a"),
            ),
            (
                "bravo".to_string(),
                test_profile("http://localhost:8200/v1", "model-b"),
            ),
            (
                "charlie".to_string(),
                test_profile("http://localhost:8300/v1", "model-c"),
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "bravo",
                startup_endpoint: "http://localhost:8200/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
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
    fn information_runs_off_the_ui_thread() {
        const SERVER: &str = "local";

        let llms = HashMap::from([(
            SERVER.to_string(),
            test_profile("http://localhost:8100/v1", "model-a"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = SERVER.to_string();
        let mut active_model_id = "model-a".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/information",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: SERVER,
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        // The probes hit the network, so /information runs like /duplicates and
        // /search: off the UI thread, via `CommandOutcome::Blocking`, rather than
        // producing its report synchronously.
        assert!(matches!(outcome, CommandOutcome::Blocking(_)));
    }

    #[test]
    fn unknown_slash_commands_error_locally() {
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut llms = HashMap::new();
        llms.insert(
            "default".to_string(),
            LlmConfiguration {
                model: "gpt-4.1".to_string(),
                endpoint: "http://localhost:11434/v1".to_string(),
                role: "all".to_string(),
                api_key: None,
                request_timeout_seconds: 30,
                max_tool_rounds: 10,
                review_max_tokens: 512,
                code_max_tokens: 0,
                system_prompt: String::new(),
                model_verbosity: None,
                review_confidence_threshold: 80,
            },
        );
        let mut session = ChatSession::new(&system_prompt(&llms["default"], None));
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
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "default",
                startup_endpoint: "http://localhost:11434/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
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
    fn pending_list_returns_pending_list_outcome() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/pending",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::PendingList));
    }

    #[test]
    fn pending_delete_with_index_returns_pending_delete_outcome() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/pending delete 3",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(outcome, CommandOutcome::PendingDelete(3)));
    }

    #[test]
    fn pending_delete_without_index_returns_usage_output() {
        let llms = HashMap::from([(
            "llama".to_string(),
            test_profile("http://localhost:8100/v1", "gemma"),
        )]);
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut active_model = "llama".to_string();
        let mut active_model_id = "gemma".to_string();
        let mut current_endpoint = Some(normalized_openai_endpoint("http://localhost:8100/v1"));
        let mut session = ChatSession::new("system");

        let outcome = handle_command(
            "/pending delete",
            CommandState {
                active_model: &mut active_model,
                active_model_id: &mut active_model_id,
                current_endpoint: &mut current_endpoint,
                session: &mut session,
                detect_model: &mut false,
            },
            CommandContext {
                skills: &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
                startup_model: "llama",
                startup_endpoint: "http://localhost:8100/v1",
                llms: &llms,
                tools: &tools,
                workspace: workspace.path(),
                session_dir: workspace.path(),
                embeddings_server: "",
                is_coordinator: false,
                usage_stats: &super::UsageStats::new(),
                available_models: &[],
                virtual_width: 512,
                auto_rebase: false,
                auto_squash: false,
                compile_workers: 1,
                compression: false,
                terminal: "",
                forge: crate::git::Forge::GitHub,
                semantic_budget_tokens: 16384,
                config_path: workspace.path(),
                review_reports: crate::git::ReviewReports::default(),
            },
        )
        .expect("handle command");

        assert!(matches!(
            outcome,
            CommandOutcome::Output(ref msg) if msg.contains("Usage")
        ));
    }
}

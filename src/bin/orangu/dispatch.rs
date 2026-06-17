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

use crate::*;

pub(crate) fn local_command_error(err: Error) -> CommandOutcome {
    if err.is::<LocalError>() {
        CommandOutcome::OutputError(format!("{err}"))
    } else {
        CommandOutcome::OutputError(format!("Error: {err:#}"))
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
            launch_outcome(ReviewLaunch { files })
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
pub(crate) fn auto_review_file_outcome(workspace: &Path, file: &str) -> CommandOutcome {
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

    CommandOutcome::AutoReview(ReviewLaunch { files: vec![entry] })
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
        review_reports,
    } = context;

    match command {
        LocalCommand::Help => Ok(CommandOutcome::Output(orangu::tui::help_text().to_string())),
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
        LocalCommand::Review => Ok(review_outcome(workspace, CommandOutcome::Review)),
        LocalCommand::AutoReview(None) => Ok(review_outcome(workspace, CommandOutcome::AutoReview)),
        LocalCommand::AutoReview(Some(file)) => {
            Ok(auto_review_file_outcome(workspace, file.trim()))
        }
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
            // else is a directory"). Until tabs share one process, the active
            // workspace is the only one open, so it is tab 1; any other number
            // has no tab to switch to yet.
            if let Ok(number) = arg.parse::<usize>() {
                if number == 0 {
                    return Ok(CommandOutcome::OutputError(
                        "Workspace numbers start at 1.".to_string(),
                    ));
                }
                if number == 1 {
                    return Ok(CommandOutcome::Output("Already in workspace 1".to_string()));
                }
                return Ok(CommandOutcome::OutputError(format!(
                    "No workspace {number} is open."
                )));
            }
            // Otherwise the argument is a directory: open it, or report it is
            // already the active workspace. Re-exec into a real directory the
            // same way `/session <path>` opens a new workspace.
            match resolve_existing_dir_arg(arg) {
                Some(dir) if dir == workspace => Ok(CommandOutcome::Output(format!(
                    "Already in workspace {}",
                    dir.display()
                ))),
                Some(dir) => Ok(CommandOutcome::SwitchWorkspace(dir)),
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
        LocalCommand::Export(target) => Ok(CommandOutcome::Export(target)),
        LocalCommand::Manual => Ok(CommandOutcome::Manual),
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
        LocalCommand::PendingList => Ok(CommandOutcome::PendingList),
        LocalCommand::PendingDelete(None) => Ok(CommandOutcome::Output(
            "Usage: /pending delete <number>. Use /pending to list.".to_string(),
        )),
        LocalCommand::PendingDelete(Some(index)) => Ok(CommandOutcome::PendingDelete(index)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use std::collections::HashMap;
    use std::fs;
    use tempfile::tempdir;

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
        match auto_review_file_outcome(workspace.path(), "src/tui.rs") {
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
            auto_review_file_outcome(workspace.path(), "src/missing.rs"),
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
        match auto_review_file_outcome(workspace.path(), "README.md") {
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
            auto_review_file_outcome(workspace.path(), "other.txt"),
            CommandOutcome::OutputError(_)
        ));
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
                test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
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
                    startup_model: "llama",
                    startup_endpoint: "http://localhost:8100/v1",
                    llms: &llms,
                    tools: &tools,
                    workspace: &here,
                    usage_stats: &super::UsageStats::new(),
                    available_models: &[],
                    virtual_width: 512,
                    auto_rebase: false,
                    auto_squash: false,
                    terminal: "",
                    forge: crate::git::Forge::GitHub,
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

        // Tab 1 is the active workspace; 0 and any higher number have no tab.
        assert!(matches!(run("/workspace 1"), CommandOutcome::Output(_)));
        assert!(matches!(
            run("/workspace 0"),
            CommandOutcome::OutputError(_)
        ));
        assert!(matches!(
            run("/workspace 2"),
            CommandOutcome::OutputError(_)
        ));

        // The active workspace's own path is a no-op switch.
        match run(&format!("/workspace {}", here.display())) {
            CommandOutcome::Output(message) => assert!(message.contains("Already in workspace")),
            _ => panic!("expected a no-op for the active workspace path"),
        }

        // A different existing directory re-execs into it.
        match run(&format!("/workspace {}", other.path().display())) {
            CommandOutcome::SwitchWorkspace(dir) => {
                assert_eq!(dir, crate::normalize_path(other.path()));
            }
            _ => panic!("expected a workspace switch for a different directory"),
        }

        // A path that is not a directory is rejected.
        match run("/workspace /no/such/orangu/dir") {
            CommandOutcome::OutputError(message) => assert!(message.contains("No such directory")),
            _ => panic!("expected an error for a missing directory"),
        }
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
                review_max_tokens: 512,
                code_max_tokens: 0,
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
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
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
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
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
            test_profile("llama.cpp", "http://localhost:8100/v1", "gemma"),
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

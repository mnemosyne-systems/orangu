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

use std::path::Path;

use crate::git::{
    discover_git_root, git_branch_names, git_local_branch_names, is_protected_branch,
};

mod bisect;
mod files;
mod ghost;
mod git_refs;
mod issue;
mod prune;
mod pull;
mod session;

pub use bisect::*;
pub use files::*;
pub use ghost::*;
pub use git_refs::*;
pub use issue::*;
pub use prune::*;
pub use pull::*;
pub(crate) use session::*;

use crate::slash_command::SlashCommand;
use strum::IntoEnumIterator;

pub fn completion_candidates(
    input: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
    skills: &orangu::skills::SkillRegistry,
) -> Option<(usize, usize, Vec<String>)> {
    let cursor = cursor.min(input.len());
    let prefix = &input[..cursor];

    if let Some(result) =
        structured_completion_candidates(prefix, cursor, workspace, server_names, available_models)
    {
        return Some(result);
    }

    if prefix.starts_with('/') {
        return Some((
            0,
            cursor,
            SlashCommand::iter()
                .map(|cmd| cmd.command())
                .filter(|command| command.starts_with(prefix))
                .chain(skills_for_prefix(prefix, skills))
                .collect(),
        ));
    }

    let start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    let token = &prefix[start..];
    Some((start, cursor, file_completion_candidates(token, workspace)))
}

fn skills_for_prefix<'a>(
    prefix: &'a str,
    skills: &'a orangu::skills::SkillRegistry,
) -> impl Iterator<Item = String> + 'a {
    skills
        .all()
        .iter()
        .map(|skill| format!("/{}", skill.name))
        .filter(move |command| {
            command.starts_with(prefix) && !SlashCommand::iter().any(|c| c.command() == *command)
        })
}

/// The grey inline "ghost" suffix previewing the first structured completion
/// candidate at the cursor, e.g. `switch to branch m` -> `ain` (completing the
/// `main` branch) or `/server lo` -> `cal`. This complements the command and
/// natural-language ghosts so branch, tag, file, model, and server arguments get
/// the same inline hint that Tab fills in.
///
/// Limited to the structured completions (those tied to a recognised command);
/// the generic last-word file completion that also fires for ordinary prose is
/// deliberately excluded so plain prompts are not littered with hints. Returns
/// `None` unless the cursor sits at the end of the input and the first candidate
/// extends the typed token rather than rewriting it.
pub fn completion_ghost_suffix(
    input: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
    _skills: &orangu::skills::SkillRegistry,
) -> Option<String> {
    if cursor != input.len() {
        return None;
    }
    let (start, end, candidates) =
        structured_completion_candidates(input, cursor, workspace, server_names, available_models)?;
    let candidate = candidates.into_iter().next()?;
    let typed = input.get(start..end)?;
    candidate
        .strip_prefix(typed)
        .filter(|rest| !rest.is_empty())
        .map(str::to_string)
}

/// Tab/ghost completion for the `/export <target>` argument — and its
/// natural-language `export <target>` form — as `(token_start, candidates)`:
/// the target words `console`, `review`, and `auto review` that extend what is
/// typed. The whole argument (after `export `/`/export `) is matched as one
/// prefix so the multi-word `auto review` option completes from `a`/`auto`.
/// Returns `None` when `prefix` is not an export argument.
fn export_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    let (token_start, value) = prefix
        .strip_prefix("/export ")
        .map(|value| ("/export ".len(), value))
        .or_else(|| {
            prefix
                .strip_prefix("export ")
                .map(|value| ("export ".len(), value))
        })?;
    let lower = value.to_ascii_lowercase();
    let candidates = crate::commands::EXPORT_TARGETS
        .iter()
        .filter(|target| target.starts_with(lower.as_str()))
        .map(|target| (*target).to_string())
        .collect();
    Some((token_start, candidates))
}

/// Tab/ghost completion for `/theme <name>` — and its natural-language `theme
/// <name>`, `use theme <name>`, `switch theme to <name>`, and `set theme to
/// <name>` forms — as `(token_start, candidates)`: available session themes
/// (`default`, `auto`, built-ins, and user theme files).
fn theme_completion_candidates(prefix: &str) -> Option<(usize, Vec<String>)> {
    let (token_start, value) = prefix
        .strip_prefix("/theme ")
        .map(|value| ("/theme ".len(), value))
        .or_else(|| {
            prefix
                .strip_prefix("theme ")
                .map(|value| ("theme ".len(), value))
        })
        .or_else(|| {
            prefix
                .strip_prefix("use theme ")
                .map(|value| ("use theme ".len(), value))
        })
        .or_else(|| {
            prefix
                .strip_prefix("switch theme to ")
                .map(|value| ("switch theme to ".len(), value))
        })
        .or_else(|| {
            prefix
                .strip_prefix("set theme to ")
                .map(|value| ("set theme to ".len(), value))
        })?;
    let lower = value.to_ascii_lowercase();
    let themes = orangu::tui::Theme::available_session_theme_names();
    let candidates = themes
        .into_iter()
        .filter(|t| t.to_ascii_lowercase().starts_with(&lower))
        .collect();
    Some((token_start, candidates))
}

/// Tab/ghost completion for the `/build [debug|release] [<target>]` argument
/// as `(token_start, candidates)`: the profile keywords plus the workspace's
/// discovered build targets (cargo binary names, Makefile rule names — see
/// `build::completion_targets`). Only the token under the cursor is
/// completed, so `/build release do` offers `docs` without re-offering the
/// profile. Returns `None` when `prefix` is not a `/build` argument.
fn build_completion_candidates(prefix: &str, workspace: &Path) -> Option<(usize, Vec<String>)> {
    let value = prefix.strip_prefix("/build ")?;
    // Complete the token under the cursor — the one after the last space.
    let token_start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    let token = &prefix[token_start..];

    let mut candidates: Vec<String> = Vec::new();
    // The profile keywords are only offered while typing the first argument
    // token; once a profile (or target) is already given, only targets remain.
    if !value.trim_end().contains(char::is_whitespace) {
        candidates.extend(["debug".to_string(), "release".to_string()]);
    }
    candidates.extend(crate::build::completion_targets(workspace));
    let lower = token.to_ascii_lowercase();
    candidates.retain(|candidate| candidate.to_ascii_lowercase().starts_with(&lower));
    Some((token_start, candidates))
}

/// The completion candidates tied to a recognised command (branch, tag, file,
/// model, server, session, ... arguments). Returns `None` when the input is not
/// one of those forms, leaving [`completion_candidates`] to fall back to the
/// slash-command list or generic file completion. Split out so the inline ghost
/// hint can reuse exactly these structured candidates without the prose-noisy
/// fallback.
fn structured_completion_candidates(
    prefix: &str,
    cursor: usize,
    workspace: &Path,
    server_names: &[String],
    available_models: &[String],
) -> Option<(usize, usize, Vec<String>)> {
    if let Some((start, candidates)) = show_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = shell_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = auto_review_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, path_prefix)) = open_file_completion_prefix(prefix) {
        return Some((
            start,
            cursor,
            open_file_completion_candidates(path_prefix, workspace),
        ));
    }

    if let Some((start, path_prefix)) = natural_show_file_completion_prefix(prefix) {
        return Some((
            start,
            cursor,
            open_file_completion_candidates(path_prefix, workspace),
        ));
    }

    if let Some(model_prefix) = prefix.strip_prefix("/model ") {
        return Some((
            7,
            cursor,
            available_models
                .iter()
                .filter(|model| model.starts_with(model_prefix))
                .cloned()
                .collect(),
        ));
    }

    if let Some(server_prefix) = prefix.strip_prefix("/server ") {
        return Some((
            8,
            cursor,
            server_names
                .iter()
                .filter(|server| server.starts_with(server_prefix))
                .cloned()
                .collect(),
        ));
    }

    if let Some((start, candidates)) = theme_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, token)) = pull_completion_prefix(prefix) {
        return Some((start, cursor, pull_number_candidates(token)));
    }

    if let Some((start, candidates)) = comment_file_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = issue_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = export_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = build_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    // `/create_workspace <dir>`: mirror the same completion logic as `/workspace
    // <dir>` — previously-seen workspace paths first, then filesystem directory
    // completion for paths starting with `~`, `/`, or `.`.
    if let Some(arg_prefix) = prefix.strip_prefix("/create_workspace ") {
        let mut candidates: Vec<String> = session_workspaces_newest_first()
            .into_iter()
            .filter(|w| w.starts_with(arg_prefix))
            .collect();
        if candidates.is_empty() {
            candidates = session_path_completion_candidates(arg_prefix);
        }
        return Some(("/create_workspace ".len(), cursor, candidates));
    }

    if let Some((start, candidates)) = close_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = get_comments_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = push_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = stash_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = restore_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    // `/branch -…` flag forms first, so they are not swallowed by the switch-form
    // branch completion below.
    if let Some((start, candidates)) = branch_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = checkout_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = create_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = delete_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = move_file_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = cherry_pick_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = show_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = fetch_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = rebase_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, candidates)) = bisect_completion_candidates(prefix, workspace) {
        return Some((start, cursor, candidates));
    }

    if let Some((start, branch_prefix)) = diff_completion_prefix(prefix) {
        let branches = discover_git_root(workspace)
            .map(|root| {
                let local = git_local_branch_names(&root);
                let all = git_branch_names(&root);
                let local_set: std::collections::HashSet<&str> =
                    local.iter().map(String::as_str).collect();
                let remote_only: Vec<String> = all
                    .into_iter()
                    .filter(|b| !local_set.contains(b.as_str()))
                    .collect();
                local.into_iter().chain(remote_only).collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|b| b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some((start, branch_prefix)) = merge_completion_prefix(prefix) {
        let branches = discover_git_root(workspace)
            .map(|root| {
                let local = git_local_branch_names(&root);
                let all = git_branch_names(&root);
                let local_set: std::collections::HashSet<&str> =
                    local.iter().map(String::as_str).collect();
                let remote_only: Vec<String> = all
                    .into_iter()
                    .filter(|b| !local_set.contains(b.as_str()))
                    .collect();
                local.into_iter().chain(remote_only).collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|b| b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some((start, branch_prefix)) = delete_branch_completion_prefix(prefix) {
        let branches: Vec<String> = discover_git_root(workspace)
            .map(|root| git_local_branch_names(&root))
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !is_protected_branch(b) && b.starts_with(branch_prefix))
            .collect();
        return Some((start, cursor, branches));
    }

    if let Some(arg_prefix) = prefix.strip_prefix("/pending ") {
        // `/pending <list|delete>`: the two subcommands. `delete` takes a
        // free-form index after it, so `/pending delete 2` narrows to nothing.
        let candidates = ["list", "delete"]
            .into_iter()
            .filter(|sub| sub.starts_with(arg_prefix))
            .map(str::to_string)
            .collect();
        return Some(("/pending ".len(), cursor, candidates));
    }

    if let Some(arg_prefix) = prefix.strip_prefix("/session ") {
        // `/session <arg>` takes either a session UUID (to switch) or a workspace
        // path (to filter the listing), so offer both: UUIDs first, newest-first,
        // then the unique workspace paths.
        let mut candidates: Vec<String> = session_uuids_newest_first()
            .into_iter()
            .filter(|u| u.starts_with(arg_prefix))
            .collect();
        candidates.extend(
            session_workspaces_newest_first()
                .into_iter()
                .filter(|w| w.starts_with(arg_prefix)),
        );
        // When the argument matches no session UUID or known workspace, fall back
        // to filesystem directory completion so a brand-new workspace can be
        // navigated to (e.g. `/session ~/Pr<TAB>/po<TAB>`).
        if candidates.is_empty() {
            candidates = session_path_completion_candidates(arg_prefix);
        }
        return Some(("/session ".len(), cursor, candidates));
    }

    if let Some(arg_prefix) = prefix.strip_prefix("/workspace ") {
        // `/workspace <arg>` takes a tab number or a directory. A bare number is
        // typed directly, so completion only helps with the directory form:
        // offer the workspaces seen in past sessions first, then fall back to
        // filesystem directory completion so a brand-new directory can be
        // navigated to (e.g. `/workspace ~/Pr<TAB>/po<TAB>`).
        let mut candidates: Vec<String> = session_workspaces_newest_first()
            .into_iter()
            .filter(|w| w.starts_with(arg_prefix))
            .collect();
        if candidates.is_empty() {
            candidates = session_path_completion_candidates(arg_prefix);
        }
        return Some(("/workspace ".len(), cursor, candidates));
    }

    if let Some((start, candidates)) = prune_completion_candidates(prefix) {
        return Some((start, cursor, candidates));
    }

    None
}

pub fn slash_command_dropdown_candidates(
    prefix: &str,
    skills: &orangu::skills::SkillRegistry,
) -> Vec<(String, String)> {
    let mut candidates = Vec::new();

    for cmd in SlashCommand::iter() {
        let cmd_str = cmd.command();
        if cmd_str.starts_with(prefix) {
            candidates.push((cmd_str, cmd.description().to_string()));
        }
    }

    for skill in skills.all() {
        let cmd = format!("/{}", skill.name);
        if cmd.starts_with(prefix) && !SlashCommand::iter().any(|c| c.command() == cmd) {
            let desc = skill.description.clone();
            candidates.push((cmd, desc));
        }
    }

    candidates
}

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_review_command_completes_from_the_command_list() {
        // `/auto_review` is registered in the completion list...
        assert!(SlashCommand::iter().any(|c| c.command() == "/auto_review"));
        assert!(SlashCommand::iter().any(|c| c.command() == "/skills"));
        // ...the inline ghost hint completes it once the prefix is unambiguous...
        assert_eq!(
            command_ghost_suffix(
                "/auto",
                &orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))
            ),
            Some("_review".to_string())
        );
        // ...and the natural-language ghost knows the alias too.
        assert!(natural_language_ghost_candidates("auto re").contains(&"view"));
    }

    #[test]
    fn auto_review_completes_the_immediate_keyword() {
        let workspace = tempfile::tempdir().expect("workspace");
        // Typing a prefix of `immediate` offers it as a candidate.
        let (_, candidates) =
            auto_review_completion_candidates("/auto_review imm", workspace.path())
                .expect("auto-review argument");
        assert!(
            candidates.iter().any(|c| c == "immediate"),
            "{candidates:?}"
        );
    }

    #[test]
    fn auto_review_completes_the_all_keyword() {
        let workspace = tempfile::tempdir().expect("workspace");
        // Typing a prefix of `all` offers it as a candidate.
        let (_, candidates) =
            auto_review_completion_candidates("/auto_review al", workspace.path())
                .expect("auto-review argument");
        assert!(candidates.iter().any(|c| c == "all"), "{candidates:?}");
    }

    #[test]
    fn auto_review_completes_the_deep_keyword() {
        let workspace = tempfile::tempdir().expect("workspace");
        // Typing a prefix of `deep` offers it as a candidate.
        let (_, candidates) =
            auto_review_completion_candidates("/auto_review de", workspace.path())
                .expect("auto-review argument");
        assert!(candidates.iter().any(|c| c == "deep"), "{candidates:?}");
    }

    #[test]
    fn theme_completion_offers_default_and_builtins() {
        let workspace = tempfile::tempdir().expect("workspace");
        let skills = orangu::skills::SkillRegistry::discover(std::path::Path::new("/"));
        for prefix in [
            "/theme ",
            "theme ",
            "use theme ",
            "switch theme to ",
            "set theme to ",
        ] {
            let (start, _, candidates) =
                completion_candidates(prefix, prefix.len(), workspace.path(), &[], &[], &skills)
                    .expect("theme candidates");
            assert_eq!(start, prefix.len());
            for expected in ["default", "classic", "auto"] {
                assert!(
                    candidates.iter().any(|candidate| candidate == expected),
                    "missing {expected} for {prefix:?}: {candidates:?}"
                );
            }
        }

        // The inline ghost suffix previews matching theme name
        assert_eq!(
            completion_ghost_suffix("/theme c", 8, workspace.path(), &[], &[], &skills),
            Some("lassic".to_string())
        );
        assert_eq!(
            completion_ghost_suffix("use theme c", 11, workspace.path(), &[], &[], &skills),
            Some("lassic".to_string())
        );
    }

    #[test]
    fn build_completes_profiles_and_workspace_targets() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("Makefile"),
            "all: docs\n\ttrue\ndocs:\n\ttrue\ninstall:\n\ttrue\n",
        )
        .expect("makefile");

        // The bare argument offers the profiles, then the discovered rules.
        let (start, all) =
            build_completion_candidates("/build ", workspace.path()).expect("build argument");
        assert_eq!(start, "/build ".len());
        assert_eq!(all, vec!["debug", "release", "all", "docs", "install"]);

        // Typing narrows to whatever still matches, profile or target.
        assert_eq!(
            build_completion_candidates("/build d", workspace.path())
                .expect("argument")
                .1,
            vec!["debug".to_string(), "docs".to_string()]
        );

        // After a first token, only targets are offered, and only the token
        // under the cursor completes.
        let (start, second) =
            build_completion_candidates("/build release do", workspace.path()).expect("argument");
        assert_eq!(start, "/build release ".len());
        assert_eq!(second, vec!["docs".to_string()]);

        // Not a /build argument at all.
        assert!(build_completion_candidates("/builder ", workspace.path()).is_none());
    }

    #[test]
    fn export_completes_console_review_auto_review_and_duplicates() {
        // The bare argument offers all six targets, in order.
        let (start, all) = export_completion_candidates("/export ").expect("export argument");
        assert_eq!(start, "/export ".len());
        assert_eq!(
            all,
            vec![
                "console",
                "review",
                "auto review",
                "duplicates",
                "pr",
                "statistics"
            ]
        );

        // Typing narrows; `auto review` completes from `auto` (multi-word).
        assert_eq!(
            export_completion_candidates("/export re")
                .expect("argument")
                .1,
            vec!["review".to_string()]
        );
        assert_eq!(
            export_completion_candidates("/export d")
                .expect("argument")
                .1,
            vec!["duplicates".to_string()]
        );
        assert_eq!(
            export_completion_candidates("/export p")
                .expect("argument")
                .1,
            vec!["pr".to_string()]
        );
        assert_eq!(
            export_completion_candidates("/export auto")
                .expect("argument")
                .1,
            vec!["auto review".to_string()]
        );

        // The inline ghost extends the typed token the same way.
        let skills = orangu::skills::SkillRegistry::discover(std::path::Path::new("/"));
        assert_eq!(
            completion_ghost_suffix(
                "/export auto",
                12,
                std::path::Path::new("/"),
                &[],
                &[],
                &skills
            ),
            Some(" review".to_string())
        );

        // The natural-language `export ` form completes the same targets — and
        // `export a` completes straight to `auto review`.
        let (start, all) = export_completion_candidates("export ").expect("natural argument");
        assert_eq!(start, "export ".len());
        assert_eq!(
            all,
            vec![
                "console",
                "review",
                "auto review",
                "duplicates",
                "pr",
                "statistics"
            ]
        );
        assert_eq!(
            export_completion_candidates("export a")
                .expect("argument")
                .1,
            vec!["auto review".to_string()]
        );
        assert_eq!(
            completion_ghost_suffix("export a", 8, std::path::Path::new("/"), &[], &[], &skills),
            Some("uto review".to_string())
        );

        // Not an export argument.
        assert!(export_completion_candidates("/export").is_none());
        assert!(export_completion_candidates("/exports x").is_none());
        assert!(export_completion_candidates("export").is_none());
        assert!(export_completion_candidates("exports x").is_none());
    }

    #[test]
    fn prune_dash_ghosts_the_first_flag() {
        // Typing `/prune -` previews `--workspace` inline (Tab fills it in), and
        // narrows to the matching flag as more is typed. The structured ghost
        // reuses `prune_completion_candidates`, so the natural-language and
        // workspace/uuid forms get the same inline hint.
        let skills = orangu::skills::SkillRegistry::discover(std::path::Path::new("/"));
        assert_eq!(
            completion_ghost_suffix(
                "/prune -",
                "/prune -".len(),
                std::path::Path::new("/"),
                &[],
                &[],
                &skills
            ),
            Some("-workspace".to_string())
        );
        assert_eq!(
            completion_ghost_suffix(
                "/prune --o",
                "/prune --o".len(),
                std::path::Path::new("/"),
                &[],
                &[],
                &skills
            ),
            Some("lder-than".to_string())
        );
    }

    #[test]
    fn push_stash_pending_complete_their_subcommands() {
        let workspace = std::path::Path::new("/");
        let skills = orangu::skills::SkillRegistry::discover(workspace);
        let candidates = |input: &str| {
            completion_candidates(input, input.len(), workspace, &[], &[], &skills)
                .expect("completion")
                .2
        };

        // `/push` flags (including the bare `force` keyword the parser accepts).
        assert_eq!(candidates("/push "), vec!["--force", "-f", "force"]);
        assert_eq!(candidates("/push f"), vec!["force".to_string()]);

        // `/stash` subcommands.
        assert_eq!(candidates("/stash "), vec!["pop", "list", "drop", "push"]);
        assert_eq!(candidates("/stash p"), vec!["pop", "push"]);

        // `/pending` subcommands.
        assert_eq!(candidates("/pending "), vec!["list", "delete"]);
        assert_eq!(candidates("/pending d"), vec!["delete".to_string()]);

        // The inline ghost previews the first one (`/stash p` -> `op`).
        assert_eq!(
            completion_ghost_suffix("/stash p", "/stash p".len(), workspace, &[], &[], &skills),
            Some("op".to_string())
        );
    }

    #[test]
    fn shell_completes_workspace_files_one_segment_at_a_time() {
        // `/shell ./te<Tab>` -> `/shell ./test/` -> typing `c` and `Tab` again ->
        // `/shell ./test/check.sh`, exactly like a real shell's filename
        // completion: each Tab only fills in the current path segment.
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::create_dir(workspace.path().join("test")).expect("test dir");
        std::fs::write(workspace.path().join("test").join("check.sh"), "").expect("check.sh");
        let skills = orangu::skills::SkillRegistry::discover(workspace.path());

        let (start, _, candidates) =
            completion_candidates("/shell ./te", 11, workspace.path(), &[], &[], &skills)
                .expect("shell completion");
        assert_eq!(start, "/shell ".len());
        assert_eq!(candidates, vec!["./test/".to_string()]);

        let (start, _, candidates) =
            completion_candidates("/shell ./test/c", 15, workspace.path(), &[], &[], &skills)
                .expect("shell completion");
        assert_eq!(start, "/shell ".len());
        assert_eq!(candidates, vec!["./test/check.sh".to_string()]);

        // The inline ghost previews the same completion.
        assert_eq!(
            completion_ghost_suffix(
                "/shell ./te",
                "/shell ./te".len(),
                workspace.path(),
                &[],
                &[],
                &skills
            ),
            Some("st/".to_string())
        );

        // An earlier argument in the command line is left untouched; only the
        // last token completes.
        let (start, _, candidates) =
            completion_candidates("/shell cat ./te", 15, workspace.path(), &[], &[], &skills)
                .expect("shell completion");
        assert_eq!(start, "/shell cat ".len());
        assert_eq!(candidates, vec!["./test/".to_string()]);
    }

    #[test]
    fn slash_completion_includes_discovered_skills() {
        let workspace = tempfile::tempdir().expect("workspace");
        let skill_dir = workspace.path().join(".agents/skills/deploy");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: Deploy the app\n---\nDeploy it\n",
        )
        .expect("skill");
        let skills = orangu::skills::SkillRegistry::discover(workspace.path());

        let (_, _, candidates) =
            completion_candidates("/dep", 4, workspace.path(), &[], &[], &skills)
                .expect("slash completion");
        assert!(candidates.contains(&"/deploy".to_string()));
    }

    #[test]
    fn every_completion_command_parses_as_a_slash_command() {
        // The completion list is maintained by hand; every entry must stay a
        // real command, so a typo or a renamed command fails here instead of
        // silently completing to something the parser rejects.
        for command in crate::slash_command::SlashCommand::iter() {
            let cmd_str = command.command();
            assert!(
                crate::commands::parse_slash_command(&cmd_str).is_some(),
                "completion entry {cmd_str:?} does not parse as a slash command"
            );
        }
    }
}

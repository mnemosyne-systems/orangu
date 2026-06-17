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

use super::*;
use crate::test_support::*;
use std::fs;
use tempfile::tempdir;

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
fn completes_workspace_directories() {
    // `/workspace <path>` offers directories to open. A forward slash is
    // appended to the typed portion so the path is recognised on every
    // platform, matching how the session path completion is exercised.
    let root = tempdir().expect("tempdir");
    let base = root.path();
    fs::create_dir(base.join("proj-alpha")).expect("dir");
    fs::create_dir(base.join("proj-beta")).expect("dir");
    fs::write(base.join("proj-notes.txt"), "").expect("file");

    let input = format!("/workspace {}/proj-", base.display());
    let (start, _, candidates) =
        completion_candidates(&input, input.len(), base, &[], &[]).expect("workspace completion");
    assert_eq!(start, "/workspace ".len());
    assert!(candidates.contains(&format!("{}/proj-alpha", base.display())));
    assert!(candidates.contains(&format!("{}/proj-beta", base.display())));
    // Only directories are offered; the plain file is skipped.
    assert!(
        !candidates.iter().any(|c| c.ends_with("proj-notes.txt")),
        "files must not be offered: {candidates:?}"
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
fn completes_fetch_remotes_with_origin_first() {
    let workspace = tempdir().expect("workspace");
    init_test_git_repo(workspace.path());
    // Add remotes out of alphabetical order; `origin` must still be offered
    // first (the default), with the rest following alphabetically.
    for (name, url) in [
        ("upstream", "https://example.com/upstream.git"),
        ("origin", "https://example.com/origin.git"),
        ("fork", "https://example.com/fork.git"),
    ] {
        assert!(
            std::process::Command::new("git")
                .args(["remote", "add", name, url])
                .current_dir(workspace.path())
                .status()
                .expect("git remote add")
                .success()
        );
    }

    // Empty argument offers every remote, origin first.
    let (start, _, candidates) =
        completion_candidates("/fetch ", "/fetch ".len(), workspace.path(), &[], &[])
            .expect("fetch completion");
    assert_eq!(start, "/fetch ".len());
    assert_eq!(candidates, vec!["origin", "fork", "upstream"]);

    // The grey ghost previews the default (origin) when nothing is typed.
    assert_eq!(
        completion_ghost_suffix("/fetch ", "/fetch ".len(), workspace.path(), &[], &[]).as_deref(),
        Some("origin")
    );

    // Typing narrows; `/fetch u` -> `upstream`.
    let (start, _, narrowed) =
        completion_candidates("/fetch u", "/fetch u".len(), workspace.path(), &[], &[])
            .expect("fetch completion");
    assert_eq!(start, "/fetch ".len());
    assert_eq!(narrowed, vec!["upstream"]);
    assert_eq!(
        completion_ghost_suffix("/fetch u", "/fetch u".len(), workspace.path(), &[], &[])
            .as_deref(),
        Some("pstream")
    );

    // The natural-language form completes the same way.
    let (start, _, nat_candidates) =
        completion_candidates("fetch f", "fetch f".len(), workspace.path(), &[], &[])
            .expect("natural fetch completion");
    assert_eq!(start, "fetch ".len());
    assert_eq!(nat_candidates, vec!["fork"]);
}

#[test]
fn completes_rebase_targets_local_then_remotes_then_remote_branches() {
    let workspace = tempdir().expect("workspace");
    init_test_git_repo(workspace.path());
    std::process::Command::new("git")
        .args(["symbolic-ref", "HEAD", "refs/heads/main"])
        .current_dir(workspace.path())
        .status()
        .expect("set initial branch to main");
    fs::write(workspace.path().join("main.rs"), "").expect("main.rs");
    for args in [
        &["add", "main.rs"][..],
        &["commit", "--quiet", "-m", "initial"][..],
        &["checkout", "--quiet", "-b", "feature"][..],
    ] {
        assert!(
            std::process::Command::new("git")
                .args(args)
                .current_dir(workspace.path())
                .status()
                .expect("git")
                .success()
        );
    }
    // A remote with a tracking branch, created without touching the network by
    // pointing the remote at the repo itself and fetching it.
    assert!(
        std::process::Command::new("git")
            .args(["remote", "add", "origin", "."])
            .current_dir(workspace.path())
            .status()
            .expect("git remote add")
            .success()
    );
    assert!(
        std::process::Command::new("git")
            .args(["fetch", "--quiet", "origin"])
            .current_dir(workspace.path())
            .status()
            .expect("git fetch")
            .success()
    );

    let (start, _, candidates) =
        completion_candidates("/rebase ", "/rebase ".len(), workspace.path(), &[], &[])
            .expect("rebase completion");
    assert_eq!(start, "/rebase ".len());
    // Local branches come first, then the remote, then remote-tracking branches.
    let main_local = candidates.iter().position(|c| c == "main");
    let origin_remote = candidates.iter().position(|c| c == "origin");
    let origin_main = candidates.iter().position(|c| c == "origin/main");
    assert!(main_local.is_some(), "local branch missing: {candidates:?}");
    assert!(origin_remote.is_some(), "remote missing: {candidates:?}");
    assert!(
        origin_main.is_some(),
        "remote-tracking branch missing: {candidates:?}"
    );
    assert!(
        main_local < origin_remote && origin_remote < origin_main,
        "wrong order: {candidates:?}"
    );

    // The grey ghost previews the first local branch when nothing is typed.
    assert_eq!(
        completion_ghost_suffix("/rebase ", "/rebase ".len(), workspace.path(), &[], &[])
            .as_deref(),
        Some("feature")
    );

    // `origin/` narrows to remote-tracking branches; the natural-language form
    // resolves the same candidates.
    let (start, _, narrowed) = completion_candidates(
        "rebase origin/",
        "rebase origin/".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("natural rebase completion");
    assert_eq!(start, "rebase ".len());
    assert!(
        narrowed.iter().all(|c| c.starts_with("origin/")),
        "{narrowed:?}"
    );
    assert!(
        narrowed.contains(&"origin/main".to_string()),
        "{narrowed:?}"
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

    // The longer `switch to branch ` phrasing must complete branches too,
    // keeping `m` (not `branch m`) as the token being completed.
    let (start, _, branch_candidates) = completion_candidates(
        "switch to branch m",
        "switch to branch m".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("switch to branch completion");
    assert_eq!(start, "switch to branch ".len());
    assert!(
        branch_candidates.contains(&"main".to_string()),
        "main missing"
    );
    assert!(
        branch_candidates.contains(&"mybranch".to_string()),
        "mybranch missing"
    );
}

#[test]
fn ghost_previews_first_structured_completion() {
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

    // The first matching branch is previewed as the trailing ghost suffix.
    let input = "switch to branch m";
    assert_eq!(
        completion_ghost_suffix(input, input.len(), workspace.path(), &[], &[]).as_deref(),
        Some("ain")
    );

    // `/server` argument completion previews the first server name.
    let servers = vec!["local".to_string(), "remote".to_string()];
    assert_eq!(
        completion_ghost_suffix(
            "/server lo",
            "/server lo".len(),
            workspace.path(),
            &servers,
            &[]
        )
        .as_deref(),
        Some("cal")
    );

    // Ordinary prose gets no ghost even when its last word prefixes a file,
    // so plain prompts stay clean.
    assert_eq!(
        completion_ghost_suffix(
            "tell me about main",
            "tell me about main".len(),
            workspace.path(),
            &[],
            &[]
        ),
        None
    );

    // No ghost when the cursor is not at the end of the input.
    assert_eq!(
        completion_ghost_suffix(input, 0, workspace.path(), &[], &[]),
        None
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

#[test]
fn auto_review_completes_files_by_name_per_branch() {
    let _env_lock = crate::process_env_lock()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let workspace = tempdir().expect("workspace");
    let home = tempdir().expect("home");
    let _home = crate::git::EnvVarGuard::set_path("HOME", home.path());
    crate::git::init_git_for_test(workspace.path());

    // A `main` branch with two committed files.
    crate::git::git_run(workspace.path(), &["checkout", "-B", "main"]);
    fs::create_dir(workspace.path().join("src")).expect("src dir");
    fs::write(workspace.path().join("src/tui.rs"), "fn main() {}\n").expect("tui");
    fs::write(workspace.path().join("README.md"), "# readme\n").expect("readme");
    crate::git::git_run(workspace.path(), &["add", "."]);
    crate::git::git_run(workspace.path(), &["commit", "-m", "base"]);

    // On main/master every tracked file is offered, completed by name not
    // location: `t` resolves to `src/tui.rs`.
    let (start, _, candidates) = completion_candidates(
        "/auto_review t",
        "/auto_review t".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("auto_review completion");
    assert_eq!(start, "/auto_review ".len());
    assert_eq!(candidates, vec!["src/tui.rs".to_string()]);

    // The natural-language form completes the same way, case-insensitively.
    let (start, _, candidates) = completion_candidates(
        "Auto review t",
        "Auto review t".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("auto review completion");
    assert_eq!(start, "Auto review ".len());
    assert_eq!(candidates, vec!["src/tui.rs".to_string()]);

    // A feature branch that changes only README.md.
    crate::git::git_run(workspace.path(), &["checkout", "-b", "feature/x"]);
    fs::write(workspace.path().join("README.md"), "# readme\nmore\n").expect("readme edit");
    crate::git::git_run(workspace.path(), &["commit", "-am", "edit readme"]);

    // On the branch only the changed file is a candidate; the empty token lists
    // exactly it...
    let (_, _, candidates) = completion_candidates(
        "/auto_review ",
        "/auto_review ".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("auto_review completion");
    assert_eq!(candidates, vec!["README.md".to_string()]);

    // ...and `src/tui.rs`, unchanged on this branch, is not offered.
    let (_, _, candidates) = completion_candidates(
        "/auto_review t",
        "/auto_review t".len(),
        workspace.path(),
        &[],
        &[],
    )
    .expect("auto_review completion");
    assert!(candidates.is_empty(), "{candidates:?}");
}

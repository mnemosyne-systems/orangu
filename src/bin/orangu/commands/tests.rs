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

#[test]
fn leaves_regular_prompts_unhandled() {
    assert!(parse_local_command("help me understand this code").is_none());
    assert!(parse_local_command("show me the files in the workspace").is_none());
}

#[test]
fn parses_open_file_commands() {
    match parse_local_command("/open_file README.md") {
        Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
        _ => panic!("expected open file slash command"),
    }
    match parse_local_command("Open README.md") {
        Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
        _ => panic!("expected open file natural language command"),
    }
    match parse_local_command("open \"docs/user guide.md\"") {
        Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "docs/user guide.md"),
        _ => panic!("expected quoted natural language open file command"),
    }
}

#[test]
fn parses_show_file_natural_language_commands() {
    match parse_local_command("show README.md") {
        Some(LocalCommand::ShowFile(path)) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected natural language show file command"),
    }
    match parse_local_command("show file \"docs/user guide.md\"") {
        Some(LocalCommand::ShowFile(path)) => assert_eq!(path.as_ref(), "docs/user guide.md"),
        _ => panic!("expected quoted natural language show file command"),
    }
    match parse_local_command("show src/tui.rs with hash") {
        Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "--hash src/tui.rs"),
        _ => panic!("expected natural language show file hash command"),
    }
    match parse_local_command("show src/tui.rs with author") {
        Some(LocalCommand::ShowFile(args)) => {
            assert_eq!(args.as_ref(), "--author src/tui.rs")
        }
        _ => panic!("expected natural language show file author command"),
    }
    match parse_local_command("show file \"docs/user guide.md\" with hash and author") {
        Some(LocalCommand::ShowFile(args)) => {
            assert_eq!(args.as_ref(), "--hash --author \"docs/user guide.md\"")
        }
        _ => panic!("expected natural language show file metadata command"),
    }
}

#[test]
fn parses_show_file_commands() {
    match parse_local_command("/show_file README.md") {
        Some(LocalCommand::ShowFile(args)) => assert_eq!(args.as_ref(), "README.md"),
        _ => panic!("expected show file slash command"),
    }

    let (path, options, rev) =
        super::super::render::parse_show_file_arguments("--hash --author \"docs/user guide.md\"")
            .expect("show file args");
    assert_eq!(path, "docs/user guide.md");
    assert!(options.show_hash);
    assert!(options.show_author);
    assert!(rev.is_none());
}

#[test]
fn parses_list_files_commands() {
    assert!(matches!(
        parse_local_command("/list_files"),
        Some(LocalCommand::ListFiles)
    ));
    assert!(matches!(
        parse_local_command("list files"),
        Some(LocalCommand::ListFiles)
    ));
    assert!(matches!(
        parse_local_command("show workspace files"),
        Some(LocalCommand::ListFiles)
    ));
}

#[test]
fn parses_manual_command_and_aliases() {
    for input in ["/manual", "manual", "show manual", "open manual"] {
        assert!(
            matches!(parse_local_command(input), Some(LocalCommand::Manual)),
            "expected {input:?} to parse as Manual"
        );
    }
}

#[test]
fn parses_natural_language_command_aliases() {
    assert!(matches!(
        parse_local_command("show commands"),
        Some(LocalCommand::Help)
    ));
    assert!(matches!(
        parse_local_command("diff"),
        Some(LocalCommand::Diff(None))
    ));
    assert!(matches!(
        parse_local_command("list models"),
        Some(LocalCommand::ModelInfo)
    ));
    assert!(matches!(
        parse_local_command("show tools"),
        Some(LocalCommand::Tools)
    ));
    assert!(matches!(
        parse_local_command("disconnect"),
        Some(LocalCommand::Disconnect)
    ));
    assert!(matches!(
        parse_local_command("reset conversation"),
        Some(LocalCommand::Clear)
    ));
    assert!(matches!(
        parse_local_command("exit"),
        Some(LocalCommand::Quit)
    ));
}

#[test]
fn binding_phrases_all_parse() {
    // Every listed phrase must be a real binding so the ghost completion
    // never suggests something the parser would reject. Argument-taking
    // prefixes only parse once an argument follows (some, like `git mv `,
    // need two), so accept the bare phrase or one with trailing tokens.
    for phrase in NATURAL_LANGUAGE_BINDINGS {
        let parses = parse_local_command(phrase.trim()).is_some()
            || parse_local_command(&format!("{phrase}1")).is_some()
            || parse_local_command(&format!("{phrase}1 2")).is_some();
        assert!(parses, "natural-language binding {phrase:?} does not parse");
    }
}

#[test]
fn parse_export_target_handles_buffers_and_rejects_unknown() {
    // Empty defaults to the console; both buffers parse; case is ignored;
    // surrounding whitespace is trimmed; anything else is rejected.
    assert!(matches!(
        parse_export_target(""),
        Some(ExportTarget::Console)
    ));
    assert!(matches!(
        parse_export_target("console"),
        Some(ExportTarget::Console)
    ));
    assert!(matches!(
        parse_export_target("review"),
        Some(ExportTarget::Review)
    ));
    assert!(matches!(
        parse_export_target("  Review "),
        Some(ExportTarget::Review)
    ));
    assert!(matches!(
        parse_export_target("CONSOLE"),
        Some(ExportTarget::Console)
    ));
    assert!(parse_export_target("bogus").is_none());
}

#[test]
fn parses_export_commands() {
    // The bare command and an explicit "console" both default to the console.
    assert!(matches!(
        parse_local_command("/export"),
        Some(LocalCommand::Export(ExportTarget::Console))
    ));
    assert!(matches!(
        parse_local_command("/export console"),
        Some(LocalCommand::Export(ExportTarget::Console))
    ));
    assert!(matches!(
        parse_local_command("/export review"),
        Some(LocalCommand::Export(ExportTarget::Review))
    ));
    // An unknown buffer is not an export command.
    assert!(parse_local_command("/export bogus").is_none());

    // Natural-language forms.
    assert!(matches!(
        parse_local_command("export"),
        Some(LocalCommand::Export(ExportTarget::Console))
    ));
    assert!(matches!(
        parse_local_command("export console"),
        Some(LocalCommand::Export(ExportTarget::Console))
    ));
    assert!(matches!(
        parse_local_command("export review"),
        Some(LocalCommand::Export(ExportTarget::Review))
    ));
}

#[test]
fn parses_natural_language_commands_with_arguments() {
    match parse_local_command("switch model to local") {
        Some(LocalCommand::SetModelId(name)) => assert_eq!(name, "local"),
        _ => panic!("expected set model command"),
    }
    match parse_local_command("switch server to main") {
        Some(LocalCommand::SetServer(name)) => assert_eq!(name, "main"),
        _ => panic!("expected set server command"),
    }
    match parse_local_command("/server main") {
        Some(LocalCommand::SetServer(name)) => assert_eq!(name, "main"),
        _ => panic!("expected set server command"),
    }
}

#[test]
fn parses_pull_request_commands() {
    assert!(matches!(
        parse_local_command("/pull 58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
    assert!(matches!(
        parse_local_command("/pull"),
        Some(LocalCommand::Pull(None))
    ));
    assert!(matches!(
        parse_local_command("/pull notanumber"),
        Some(LocalCommand::Pull(None))
    ));
    assert!(matches!(
        parse_local_command("pull 58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
    assert!(matches!(
        parse_local_command("Pull 58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
    assert!(matches!(
        parse_local_command("pull pr 58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
    assert!(matches!(
        parse_local_command("pull request 58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
    assert!(matches!(
        parse_local_command("pull #58"),
        Some(LocalCommand::Pull(Some(58)))
    ));
}

#[test]
fn parses_comment_commands() {
    assert!(matches!(
        parse_local_command("/comment 51 \"My comment\""),
        Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
    ));
    assert!(matches!(
        parse_local_command("/comment 51 My comment"),
        Some(LocalCommand::Comment(Some((51, CommentBody::File(ref name))))) if name == "My comment"
    ));
    assert!(matches!(
        parse_local_command("/comment #51 \"My comment\""),
        Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
    ));
    assert!(matches!(
        parse_local_command("Add comment on 51 \"My comment\""),
        Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
    ));
    assert!(matches!(
        parse_local_command("comment on 51 \"My comment\""),
        Some(LocalCommand::Comment(Some((51, CommentBody::Inline(ref body))))) if body == "My comment"
    ));
    assert!(matches!(
        parse_local_command("/comment 51 merged.md"),
        Some(LocalCommand::Comment(Some((51, CommentBody::File(ref name))))) if name == "merged.md"
    ));
    assert!(matches!(
        parse_local_command("/comment"),
        Some(LocalCommand::Comment(None))
    ));
    assert!(matches!(
        parse_local_command("/comment 51"),
        Some(LocalCommand::Comment(None))
    ));
    assert!(matches!(
        parse_local_command("/comment 51 \"\""),
        Some(LocalCommand::Comment(None))
    ));
    assert!(matches!(
        parse_local_command("/comment notanumber \"My comment\""),
        Some(LocalCommand::Comment(None))
    ));
}

#[test]
fn parses_comment_report_keywords() {
    // `with review` / `with auto review` post the last report; the match
    // is case-insensitive and covers the natural-language forms too.
    assert!(matches!(
        parse_local_command("/comment 48 with review"),
        Some(LocalCommand::Comment(Some((48, CommentBody::Review))))
    ));
    assert!(matches!(
        parse_local_command("/comment 48 with auto review"),
        Some(LocalCommand::Comment(Some((48, CommentBody::AutoReview))))
    ));
    assert!(matches!(
        parse_local_command("comment on 48 With Review"),
        Some(LocalCommand::Comment(Some((48, CommentBody::Review))))
    ));
    assert!(matches!(
        parse_local_command("Add comment on 48 with auto review"),
        Some(LocalCommand::Comment(Some((48, CommentBody::AutoReview))))
    ));
    // Only the exact phrase is a keyword: anything else stays a template
    // filename, so templates starting with `w` keep working.
    assert!(matches!(
        parse_local_command("/comment 48 with-review.md"),
        Some(LocalCommand::Comment(Some((48, CommentBody::File(ref name))))) if name == "with-review.md"
    ));
    assert!(matches!(
        parse_local_command("/comment 48 weekly.md"),
        Some(LocalCommand::Comment(Some((48, CommentBody::File(ref name))))) if name == "weekly.md"
    ));
}

#[test]
fn parses_close_commands() {
    assert!(matches!(
        parse_local_command("/close -i 69"),
        Some(LocalCommand::Close(Some(CloseTarget::Issue(69))))
    ));
    assert!(matches!(
        parse_local_command("/close -p 42"),
        Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
    ));
    assert!(matches!(
        parse_local_command("close issue 69"),
        Some(LocalCommand::Close(Some(CloseTarget::Issue(69))))
    ));
    assert!(matches!(
        parse_local_command("close pr 42"),
        Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
    ));
    assert!(matches!(
        parse_local_command("close pull request 42"),
        Some(LocalCommand::Close(Some(CloseTarget::PullRequest(42))))
    ));
    assert!(matches!(
        parse_local_command("/close"),
        Some(LocalCommand::Close(None))
    ));
    assert!(matches!(
        parse_local_command("/close -i"),
        Some(LocalCommand::Close(None))
    ));
    assert!(matches!(
        parse_local_command("/close -p notanumber"),
        Some(LocalCommand::Close(None))
    ));
}

#[test]
fn parses_get_comments_commands() {
    assert!(matches!(
        parse_local_command("/get_comments -i 69"),
        Some(LocalCommand::GetComments(Some(GetCommentsTarget::Issue(
            69
        ))))
    ));
    assert!(matches!(
        parse_local_command("/get_comments -p 42"),
        Some(LocalCommand::GetComments(Some(
            GetCommentsTarget::PullRequest(42)
        )))
    ));
    assert!(matches!(
        parse_local_command("get comments for issue 69"),
        Some(LocalCommand::GetComments(Some(GetCommentsTarget::Issue(
            69
        ))))
    ));
    assert!(matches!(
        parse_local_command("get comments for pull request 42"),
        Some(LocalCommand::GetComments(Some(
            GetCommentsTarget::PullRequest(42)
        )))
    ));
    assert!(matches!(
        parse_local_command("/get_comments"),
        Some(LocalCommand::GetComments(None))
    ));
    assert!(matches!(
        parse_local_command("/get_comments -i"),
        Some(LocalCommand::GetComments(None))
    ));
    assert!(matches!(
        parse_local_command("/get_comments -p notanumber"),
        Some(LocalCommand::GetComments(None))
    ));
}

#[test]
fn parses_review_commands() {
    for input in [
        "/review",
        "review",
        "Review",
        "review changes",
        "code review",
    ] {
        assert!(
            matches!(parse_local_command(input), Some(LocalCommand::Review)),
            "expected {input:?} to parse as Review"
        );
    }
}

#[test]
fn parses_auto_review_commands() {
    for input in ["/auto_review", "auto review", "Auto Review"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::AutoReview(None))
            ),
            "expected {input:?} to parse as a whole-branch AutoReview"
        );
    }

    // The slash command and its natural-language form both carry the file
    // argument for a single-file review.
    for input in ["/auto_review src/tui.rs", "auto review src/tui.rs"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::AutoReview(Some(file))) if file == "src/tui.rs"
            ),
            "expected {input:?} to carry the path"
        );
    }
}

#[test]
fn parses_status_commands() {
    assert!(matches!(
        parse_local_command("/status"),
        Some(LocalCommand::Status)
    ));
    assert!(matches!(
        parse_local_command("status"),
        Some(LocalCommand::Status)
    ));
    assert!(matches!(
        parse_local_command("Status"),
        Some(LocalCommand::Status)
    ));
    assert!(matches!(
        parse_local_command("show status"),
        Some(LocalCommand::Status)
    ));
    assert!(matches!(
        parse_local_command("git status"),
        Some(LocalCommand::Status)
    ));
}

#[test]
fn parses_log_commands() {
    assert!(matches!(
        parse_local_command("/log"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("log"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("Log"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("show log"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("git log"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("git lg"),
        Some(LocalCommand::Log(None))
    ));
    assert!(matches!(
        parse_local_command("/log 5"),
        Some(LocalCommand::Log(Some(5)))
    ));
    assert!(matches!(
        parse_local_command("log 10"),
        Some(LocalCommand::Log(Some(10)))
    ));
    assert!(matches!(
        parse_local_command("show log 3"),
        Some(LocalCommand::Log(Some(3)))
    ));
    assert!(matches!(
        parse_local_command("git lg 7"),
        Some(LocalCommand::Log(Some(7)))
    ));
}

#[test]
fn parses_rebase_commands() {
    assert!(matches!(
        parse_local_command("/rebase"),
        Some(LocalCommand::Rebase)
    ));
    assert!(matches!(
        parse_local_command("rebase"),
        Some(LocalCommand::Rebase)
    ));
    assert!(matches!(
        parse_local_command("Rebase"),
        Some(LocalCommand::Rebase)
    ));
    assert!(matches!(
        parse_local_command("git rebase"),
        Some(LocalCommand::Rebase)
    ));
}

#[test]
fn parses_merge_commands() {
    assert!(matches!(
        parse_local_command("/merge"),
        Some(LocalCommand::Merge(None))
    ));
    assert!(matches!(
        parse_local_command("/merge "),
        Some(LocalCommand::Merge(None))
    ));
    assert!(matches!(
        parse_local_command("merge"),
        Some(LocalCommand::Merge(None))
    ));
    assert!(matches!(
        parse_local_command("Merge"),
        Some(LocalCommand::Merge(None))
    ));
    match parse_local_command("/merge feature/foo") {
        Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
        _ => panic!("expected merge with branch"),
    }
    match parse_local_command("merge feature/foo") {
        Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
        _ => panic!("expected natural merge with branch"),
    }
    match parse_local_command("Merge feature/foo") {
        Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
        _ => panic!("expected case-insensitive merge with branch"),
    }
    match parse_local_command("git merge feature/foo") {
        Some(LocalCommand::Merge(Some(branch))) => assert_eq!(branch.as_ref(), "feature/foo"),
        _ => panic!("expected git merge natural language with branch"),
    }
}

#[test]
fn parses_branch_commands() {
    assert!(matches!(
        parse_local_command("/branch"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
    assert!(matches!(
        parse_local_command("branch"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
    assert!(matches!(
        parse_local_command("list branches"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
    assert!(matches!(
        parse_local_command("checkout"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
    assert!(matches!(
        parse_local_command("/branch -a"),
        Some(LocalCommand::Branch(BranchSubcommand::ListAll))
    ));
    assert!(matches!(
        parse_local_command("list all branches"),
        Some(LocalCommand::Branch(BranchSubcommand::ListAll))
    ));
    match parse_local_command("/branch feature/foo") {
        Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
            assert_eq!(target.as_ref(), "feature/foo")
        }
        _ => panic!("expected branch switch"),
    }
    match parse_local_command("/checkout feature/foo") {
        Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
            assert_eq!(target.as_ref(), "feature/foo")
        }
        _ => panic!("expected checkout alias switch"),
    }
    match parse_local_command("checkout feature/foo") {
        Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
            assert_eq!(target.as_ref(), "feature/foo")
        }
        _ => panic!("expected natural checkout switch"),
    }
    match parse_local_command("switch to main") {
        Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
            assert_eq!(target.as_ref(), "main")
        }
        _ => panic!("expected switch to main"),
    }
    match parse_local_command("switch to main branch") {
        Some(LocalCommand::Branch(BranchSubcommand::Switch(target))) => {
            assert_eq!(target.as_ref(), "main")
        }
        _ => panic!("expected switch to main branch -> main"),
    }
    match parse_local_command("/branch -b feature/new") {
        Some(LocalCommand::Branch(BranchSubcommand::Create(name))) => {
            assert_eq!(name.as_ref(), "feature/new")
        }
        _ => panic!("expected branch create"),
    }
    match parse_local_command("create branch feature/new") {
        Some(LocalCommand::Branch(BranchSubcommand::Create(name))) => {
            assert_eq!(name.as_ref(), "feature/new")
        }
        _ => panic!("expected NL branch create"),
    }
    match parse_local_command("/branch -m new-name") {
        Some(LocalCommand::Branch(BranchSubcommand::Rename(name))) => {
            assert_eq!(name.as_ref(), "new-name")
        }
        _ => panic!("expected branch rename"),
    }
    match parse_local_command("/branch -d feature/old") {
        Some(LocalCommand::Branch(BranchSubcommand::Delete(name))) => {
            assert_eq!(name.as_ref(), "feature/old")
        }
        _ => panic!("expected branch delete"),
    }
}

#[test]
fn parses_add_file_commands() {
    assert!(matches!(
        parse_local_command("/add_file"),
        Some(LocalCommand::AddFile(None))
    ));
    assert!(matches!(
        parse_local_command("/add_file "),
        Some(LocalCommand::AddFile(None))
    ));
    assert!(matches!(
        parse_local_command("add"),
        Some(LocalCommand::AddFile(None))
    ));
    assert!(matches!(
        parse_local_command("Add"),
        Some(LocalCommand::AddFile(None))
    ));
    match parse_local_command("/add_file README.md") {
        Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected add_file with path"),
    }
    match parse_local_command("add README.md") {
        Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected natural add with path"),
    }
    match parse_local_command("Add src/") {
        Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "src/"),
        _ => panic!("expected case-insensitive add with directory"),
    }
    match parse_local_command("add file README.md") {
        Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected add file prefix"),
    }
    match parse_local_command("git add README.md") {
        Some(LocalCommand::AddFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected git add natural language"),
    }
}

#[test]
fn parses_remove_file_commands() {
    assert!(matches!(
        parse_local_command("/remove_file"),
        Some(LocalCommand::RemoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("/remove_file "),
        Some(LocalCommand::RemoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("remove"),
        Some(LocalCommand::RemoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("Remove"),
        Some(LocalCommand::RemoveFile(None))
    ));
    match parse_local_command("/remove_file README.md") {
        Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected remove_file with path"),
    }
    match parse_local_command("remove README.md") {
        Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected natural remove with path"),
    }
    match parse_local_command("Remove src/") {
        Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "src/"),
        _ => panic!("expected case-insensitive remove with directory"),
    }
    match parse_local_command("remove file README.md") {
        Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected remove file prefix"),
    }
    match parse_local_command("git rm README.md") {
        Some(LocalCommand::RemoveFile(Some(path))) => assert_eq!(path.as_ref(), "README.md"),
        _ => panic!("expected git rm natural language"),
    }
}

#[test]
fn parses_move_file_commands() {
    assert!(matches!(
        parse_local_command("/move_file"),
        Some(LocalCommand::MoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("/move_file "),
        Some(LocalCommand::MoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("/move_file onlyone"),
        Some(LocalCommand::MoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("move"),
        Some(LocalCommand::MoveFile(None))
    ));
    assert!(matches!(
        parse_local_command("Move"),
        Some(LocalCommand::MoveFile(None))
    ));
    match parse_local_command("/move_file old.rs new.rs") {
        Some(LocalCommand::MoveFile(Some((src, dst)))) => {
            assert_eq!(src.as_ref(), "old.rs");
            assert_eq!(dst.as_ref(), "new.rs");
        }
        _ => panic!("expected move_file with source and destination"),
    }
    match parse_local_command("move old.rs new.rs") {
        Some(LocalCommand::MoveFile(Some((src, dst)))) => {
            assert_eq!(src.as_ref(), "old.rs");
            assert_eq!(dst.as_ref(), "new.rs");
        }
        _ => panic!("expected natural move with source and destination"),
    }
    match parse_local_command("move file old.rs new.rs") {
        Some(LocalCommand::MoveFile(Some((src, dst)))) => {
            assert_eq!(src.as_ref(), "old.rs");
            assert_eq!(dst.as_ref(), "new.rs");
        }
        _ => panic!("expected move file prefix"),
    }
    match parse_local_command("git mv old.rs new.rs") {
        Some(LocalCommand::MoveFile(Some((src, dst)))) => {
            assert_eq!(src.as_ref(), "old.rs");
            assert_eq!(dst.as_ref(), "new.rs");
        }
        _ => panic!("expected git mv natural language"),
    }
}

#[test]
fn parses_cherry_pick_commands() {
    assert!(matches!(
        parse_local_command("/cherry_pick"),
        Some(LocalCommand::CherryPick(None))
    ));
    match parse_local_command("/cherry_pick abc1234") {
        Some(LocalCommand::CherryPick(Some(commit))) => {
            assert_eq!(commit.as_ref(), "abc1234");
        }
        _ => panic!("expected cherry_pick with commit"),
    }
    match parse_local_command("cherry pick abc1234") {
        Some(LocalCommand::CherryPick(Some(commit))) => {
            assert_eq!(commit.as_ref(), "abc1234");
        }
        _ => panic!("expected natural cherry pick with commit"),
    }
    match parse_local_command("cherry-pick abc1234") {
        Some(LocalCommand::CherryPick(Some(commit))) => {
            assert_eq!(commit.as_ref(), "abc1234");
        }
        _ => panic!("expected cherry-pick with commit"),
    }
    match parse_local_command("git cherry-pick abc1234") {
        Some(LocalCommand::CherryPick(Some(commit))) => {
            assert_eq!(commit.as_ref(), "abc1234");
        }
        _ => panic!("expected git cherry-pick with commit"),
    }
    assert!(matches!(
        parse_local_command("cherry pick"),
        Some(LocalCommand::CherryPick(None))
    ));
    assert!(matches!(
        parse_local_command("cherry-pick"),
        Some(LocalCommand::CherryPick(None))
    ));
}

#[test]
fn parses_commit_commands() {
    assert!(matches!(
        parse_local_command("/commit"),
        Some(LocalCommand::Commit(None))
    ));
    assert!(matches!(
        parse_local_command("commit"),
        Some(LocalCommand::Commit(None))
    ));
    match parse_local_command("/commit [#42] My feature") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected commit with plain message"),
    }
    match parse_local_command("/commit \"[#42] My feature\"") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected commit with double-quoted message"),
    }
    match parse_local_command("Commit \"[#42] My feature\"") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected natural commit with quoted message"),
    }
    match parse_local_command("commit [#42] My feature") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected natural commit without quotes"),
    }
    match parse_local_command("git commit -a -m \"[#42] My feature\"") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected git commit -a -m with quoted message"),
    }
    match parse_local_command("git commit -m fixed") {
        Some(LocalCommand::Commit(Some(msg))) => {
            assert_eq!(msg.as_ref(), "fixed");
        }
        _ => panic!("expected git commit -m form"),
    }
}

#[test]
fn parses_amend_commands() {
    assert!(matches!(
        parse_local_command("/amend"),
        Some(LocalCommand::Amend(None))
    ));
    assert!(matches!(
        parse_local_command("amend"),
        Some(LocalCommand::Amend(None))
    ));
    assert!(matches!(
        parse_local_command("git amend"),
        Some(LocalCommand::Amend(None))
    ));
    assert!(matches!(
        parse_local_command("git commit --amend"),
        Some(LocalCommand::Amend(None))
    ));
    match parse_local_command("/amend [#42] My feature") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected amend with plain message"),
    }
    match parse_local_command("/amend \"[#42] My feature\"") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected amend with double-quoted message"),
    }
    match parse_local_command("amend \"[#42] My feature\"") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected natural amend with quoted message"),
    }
    match parse_local_command("amend message \"[#42] My feature\"") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected amend message form"),
    }
    match parse_local_command("git commit --amend -m \"[#42] My feature\"") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected git commit --amend -m form"),
    }
    match parse_local_command("git amend \"[#42] My feature\"") {
        Some(LocalCommand::Amend(Some(msg))) => {
            assert_eq!(msg.as_ref(), "[#42] My feature");
        }
        _ => panic!("expected git amend form"),
    }
}

#[test]
fn parses_push_commands() {
    assert!(matches!(
        parse_local_command("/push"),
        Some(LocalCommand::Push(false))
    ));
    assert!(matches!(
        parse_local_command("/push --force"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("/push -f"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("/push force"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("push"),
        Some(LocalCommand::Push(false))
    ));
    assert!(matches!(
        parse_local_command("Push"),
        Some(LocalCommand::Push(false))
    ));
    assert!(matches!(
        parse_local_command("git push"),
        Some(LocalCommand::Push(false))
    ));
    assert!(matches!(
        parse_local_command("force push"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("push force"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("push --force"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("git push --force"),
        Some(LocalCommand::Push(true))
    ));
    assert!(matches!(
        parse_local_command("git push origin --force"),
        Some(LocalCommand::Push(true))
    ));
}

#[test]
fn parses_init_repo_commands() {
    assert!(matches!(
        parse_local_command("/init_repo"),
        Some(LocalCommand::InitRepo)
    ));
    assert!(matches!(
        parse_local_command("init"),
        Some(LocalCommand::InitRepo)
    ));
    assert!(matches!(
        parse_local_command("Init"),
        Some(LocalCommand::InitRepo)
    ));
    assert!(matches!(
        parse_local_command("init repo"),
        Some(LocalCommand::InitRepo)
    ));
    assert!(matches!(
        parse_local_command("Init Repo"),
        Some(LocalCommand::InitRepo)
    ));
    assert!(matches!(
        parse_local_command("git init"),
        Some(LocalCommand::InitRepo)
    ));
}

#[test]
fn parses_delete_branch_commands() {
    assert!(matches!(
        parse_local_command("/branch -d feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("delete feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("Delete feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("delete branch feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("Delete Branch feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("git branch -D feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
    assert!(matches!(
        parse_local_command("delete branch"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
    assert!(matches!(
        parse_local_command("delete"),
        Some(LocalCommand::Branch(BranchSubcommand::List))
    ));
}

#[test]
fn parses_squash_commands() {
    assert!(matches!(
        parse_local_command("/squash"),
        Some(LocalCommand::Squash)
    ));
    assert!(matches!(
        parse_local_command("squash"),
        Some(LocalCommand::Squash)
    ));
    assert!(matches!(
        parse_local_command("Squash"),
        Some(LocalCommand::Squash)
    ));
    assert!(matches!(
        parse_local_command("squash branch"),
        Some(LocalCommand::Squash)
    ));
    assert!(matches!(
        parse_local_command("squash commits"),
        Some(LocalCommand::Squash)
    ));
    assert!(matches!(
        parse_local_command("git squash"),
        Some(LocalCommand::Squash)
    ));
}

#[test]
fn splits_editor_command_and_flags() {
    assert_eq!(
        shell_words("code --wait").expect("editor command"),
        vec!["code".to_string(), "--wait".to_string()]
    );
    assert_eq!(
        shell_words("\"/tmp/my editor\" --flag").expect("quoted editor command"),
        vec!["/tmp/my editor".to_string(), "--flag".to_string()]
    );
}

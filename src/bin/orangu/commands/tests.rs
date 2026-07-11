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
fn parses_schedule_commands() {
    assert!(matches!(
        parse_local_command("/schedule"),
        Some(LocalCommand::Schedule)
    ));
    assert!(matches!(
        parse_local_command("schedule"),
        Some(LocalCommand::Schedule)
    ));
    // `show schedule` is the schedule listing, not a request to show a file
    // named "schedule" (which the `show <file>` natural form would otherwise
    // claim first).
    assert!(matches!(
        parse_local_command("show schedule"),
        Some(LocalCommand::Schedule)
    ));
}

#[test]
fn show_forms_resolve_to_their_commands_not_files() {
    // Like `show manual` and `show pending`, these must resolve before the
    // `show <file>` natural form claims them as file names.
    assert!(matches!(
        parse_local_command("show usage"),
        Some(LocalCommand::Usage)
    ));
    assert!(matches!(
        parse_local_command("show statistics"),
        Some(LocalCommand::Statistics(false))
    ));
    // A real file target still goes to show-file.
    assert!(matches!(
        parse_local_command("show README.md"),
        Some(LocalCommand::ShowFile(_))
    ));
}

#[test]
fn parses_prune_commands() {
    // The natural "older than" form maps the day count to `OlderThan`, matching
    // the `/prune --older-than <days>` slash flag — previously the bare number
    // was misread as a session UUID.
    assert!(matches!(
        parse_local_command("prune sessions older than 7"),
        Some(LocalCommand::Prune(Some(PruneTarget::OlderThan(7))))
    ));
    assert!(matches!(
        parse_local_command("/prune --older-than 7"),
        Some(LocalCommand::Prune(Some(PruneTarget::OlderThan(7))))
    ));
    // A non-numeric "older than" argument is not a prune command (left to be
    // handled as a prompt) rather than silently pruning a bogus UUID.
    assert!(parse_local_command("prune sessions older than soon").is_none());

    // The other forms still parse as before.
    match parse_local_command("prune session abc-123") {
        Some(LocalCommand::Prune(Some(PruneTarget::Uuid(uuid)))) => assert_eq!(uuid, "abc-123"),
        _ => panic!("expected uuid prune"),
    }
    match parse_local_command("prune sessions in ~/project") {
        Some(LocalCommand::Prune(Some(PruneTarget::Workspace(path)))) => {
            assert_eq!(path, "~/project")
        }
        _ => panic!("expected workspace prune"),
    }
    assert!(matches!(
        parse_local_command("prune all"),
        Some(LocalCommand::Prune(Some(PruneTarget::All)))
    ));
    assert!(matches!(
        parse_local_command("prune"),
        Some(LocalCommand::Prune(None))
    ));
}

#[test]
fn parses_build_commands() {
    use crate::build::BuildProfile;

    let build_of = |input: &str| match parse_local_command(input) {
        Some(LocalCommand::Build(request)) => Some(request),
        _ => None,
    };

    // Bare forms, slash and natural, default to release with no target.
    for input in ["/build", "build", "build project", "run build"] {
        let request = build_of(input).unwrap_or_else(|| panic!("no build for {input:?}"));
        assert_eq!(request.profile, BuildProfile::Release, "for {input:?}");
        assert_eq!(request.target, None, "for {input:?}");
    }

    // An explicit profile, slash or natural, in either order.
    for input in ["/build debug", "build debug", "debug build"] {
        let request = build_of(input).unwrap_or_else(|| panic!("no build for {input:?}"));
        assert_eq!(request.profile, BuildProfile::Debug, "for {input:?}");
    }
    for input in ["/build release", "build release", "release build"] {
        let request = build_of(input).unwrap_or_else(|| panic!("no build for {input:?}"));
        assert_eq!(request.profile, BuildProfile::Release, "for {input:?}");
    }

    // Case-insensitive, and surrounding whitespace on the slash argument is
    // trimmed.
    assert_eq!(
        build_of("/build DEBUG").unwrap().profile,
        BuildProfile::Debug
    );
    assert_eq!(
        build_of("/build  release  ").unwrap().profile,
        BuildProfile::Release
    );

    // A non-profile token is the build target — with or without a profile,
    // in either order.
    let request = build_of("/build docs").unwrap();
    assert_eq!(request.profile, BuildProfile::Release);
    assert_eq!(request.target.as_deref(), Some("docs"));
    let request = build_of("/build debug orangu-gguf").unwrap();
    assert_eq!(request.profile, BuildProfile::Debug);
    assert_eq!(request.target.as_deref(), Some("orangu-gguf"));
    let request = build_of("/build install release").unwrap();
    assert_eq!(request.profile, BuildProfile::Release);
    assert_eq!(request.target.as_deref(), Some("install"));

    // Two profiles or two targets are rejected rather than one silently
    // dropped.
    assert!(parse_local_command("/build debug release").is_none());
    assert!(parse_local_command("/build docs install").is_none());
}

#[test]
fn parses_shell_commands() {
    // A bare `/shell` has no command line, which is a usage error at dispatch
    // rather than an unrecognized command.
    assert!(matches!(
        parse_local_command("/shell"),
        Some(LocalCommand::Shell(None))
    ));
    assert!(matches!(
        parse_local_command("/shell   "),
        Some(LocalCommand::Shell(None))
    ));

    // The whole remainder is kept as one command line, including internal
    // whitespace and flags — only the outer whitespace is trimmed.
    match parse_local_command("/shell ls -la ./src") {
        Some(LocalCommand::Shell(Some(command))) => assert_eq!(command, "ls -la ./src"),
        _ => panic!("expected a shell command"),
    }
    match parse_local_command("  /shell   echo hi  ") {
        Some(LocalCommand::Shell(Some(command))) => assert_eq!(command, "echo hi"),
        _ => panic!("expected a shell command"),
    }
}

#[test]
fn parses_workspace_commands() {
    // Bare forms, slash and natural, list/report the active workspace.
    assert!(matches!(
        parse_local_command("/workspace"),
        Some(LocalCommand::Workspace(None))
    ));
    assert!(matches!(
        parse_local_command("workspace"),
        Some(LocalCommand::Workspace(None))
    ));
    assert!(matches!(
        parse_local_command("switch workspace"),
        Some(LocalCommand::Workspace(None))
    ));

    // Number form (the tab to switch to).
    match parse_local_command("/workspace 2") {
        Some(LocalCommand::Workspace(Some(arg))) => assert_eq!(arg.as_ref(), "2"),
        _ => panic!("expected /workspace 2 to parse with its argument"),
    }
    match parse_local_command("workspace 1") {
        Some(LocalCommand::Workspace(Some(arg))) => assert_eq!(arg.as_ref(), "1"),
        _ => panic!("expected natural `workspace 1` to parse with its argument"),
    }

    // Path form (a directory to open).
    match parse_local_command("/workspace ~/project") {
        Some(LocalCommand::Workspace(Some(arg))) => assert_eq!(arg.as_ref(), "~/project"),
        _ => panic!("expected /workspace <path> to parse with its argument"),
    }
    match parse_local_command("switch workspace ~/project") {
        Some(LocalCommand::Workspace(Some(arg))) => assert_eq!(arg.as_ref(), "~/project"),
        _ => panic!("expected natural `switch workspace <path>` to parse"),
    }
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
fn parse_open_command_target_recognizes_the_review_open_forms() {
    // The `/review` and `/auto_review` input windows accept the same open/edit
    // forms the main prompt does, opening any project file in `$EDITOR`.
    assert_eq!(
        parse_open_command_target("/open_file src/main.rs"),
        Some("src/main.rs")
    );
    assert_eq!(
        parse_open_command_target("open src/main.rs"),
        Some("src/main.rs")
    );
    // `open file <x>` yields the path, not `file <x>`; quotes are stripped.
    assert_eq!(
        parse_open_command_target("open file README.md"),
        Some("README.md")
    );
    assert_eq!(
        parse_open_command_target("edit src/lib.rs"),
        Some("src/lib.rs")
    );
    assert_eq!(
        parse_open_command_target("open \"docs/user guide.md\""),
        Some("docs/user guide.md")
    );
    // Matching is case-insensitive on the verb.
    assert_eq!(
        parse_open_command_target("OPEN src/main.rs"),
        Some("src/main.rs")
    );

    // Anything that is not an open/edit form — a review request, a note, or a
    // bare verb with no path — is left for the LLM (returns `None`).
    assert_eq!(parse_open_command_target("focus on error handling"), None);
    assert_eq!(parse_open_command_target("# please add a test"), None);
    assert_eq!(parse_open_command_target("open"), None);
    assert_eq!(parse_open_command_target("open   "), None);
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
fn parses_information_command_and_aliases() {
    for input in [
        "/information",
        "information",
        "show information",
        "server information",
        "llm information",
    ] {
        assert!(
            matches!(parse_local_command(input), Some(LocalCommand::Information)),
            "expected {input:?} to parse as Information"
        );
    }
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
    assert!(matches!(
        parse_export_target("duplicates"),
        Some(ExportTarget::Duplicates)
    ));
    assert!(matches!(
        parse_export_target("  Duplicates "),
        Some(ExportTarget::Duplicates)
    ));
    for arg in [
        "pr",
        "PR",
        "pull requests",
        "pull_requests",
        "pull-requests",
    ] {
        assert!(
            matches!(parse_export_target(arg), Some(ExportTarget::Pr)),
            "{arg:?}"
        );
    }
    // The auto-review buffer is selected by `auto review` (and its punctuation
    // variants), case-insensitively.
    for arg in ["auto review", "Auto Review", "auto_review", "auto-review"] {
        assert!(
            matches!(parse_export_target(arg), Some(ExportTarget::AutoReview)),
            "{arg:?}"
        );
    }
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
    assert!(matches!(
        parse_local_command("/export auto review"),
        Some(LocalCommand::Export(ExportTarget::AutoReview))
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
    assert!(matches!(
        parse_local_command("export auto review"),
        Some(LocalCommand::Export(ExportTarget::AutoReview))
    ));
    assert!(matches!(
        parse_local_command("/export duplicates"),
        Some(LocalCommand::Export(ExportTarget::Duplicates))
    ));
    assert!(matches!(
        parse_local_command("export duplicates"),
        Some(LocalCommand::Export(ExportTarget::Duplicates))
    ));
    assert!(matches!(
        parse_local_command("/export pr"),
        Some(LocalCommand::Export(ExportTarget::Pr))
    ));
    assert!(matches!(
        parse_local_command("export pr"),
        Some(LocalCommand::Export(ExportTarget::Pr))
    ));
}

#[test]
fn parses_duplicates_commands() {
    // The bare command uses the default threshold.
    assert!(matches!(
        parse_local_command("/duplicates"),
        Some(LocalCommand::Duplicates(None))
    ));
    // Natural-language forms.
    for input in ["duplicates", "find duplicates", "find duplicate code"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::Duplicates(None))
            ),
            "{input:?}"
        );
    }
    // A percentage argument is read as a 0.0–1.0 fraction; a trailing percent
    // sign and a bare fraction are both accepted.
    let threshold_of = |input| match parse_local_command(input) {
        Some(LocalCommand::Duplicates(threshold)) => threshold,
        _ => panic!("expected a /duplicates command for {input:?}"),
    };
    assert!((threshold_of("/duplicates 80").unwrap() - 0.80).abs() < 1e-9);
    assert!((threshold_of("/duplicates 90%").unwrap() - 0.90).abs() < 1e-9);
    assert!((threshold_of("/duplicates 0.5").unwrap() - 0.50).abs() < 1e-9);
    // An unparseable argument falls back to the default (None).
    assert!(matches!(
        parse_local_command("/duplicates lots"),
        Some(LocalCommand::Duplicates(None))
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
fn parses_show_commands() {
    // Bare forms default to HEAD (`None`).
    for input in ["/show", "git show", "show commit"] {
        assert!(
            matches!(parse_local_command(input), Some(LocalCommand::Show(None))),
            "{input:?} should parse to /show HEAD"
        );
    }
    // A commit argument is carried through, trimmed.
    for input in ["/show abc123", "git show abc123", "show commit abc123"] {
        match parse_local_command(input) {
            Some(LocalCommand::Show(Some(commit))) => assert_eq!(commit, "abc123"),
            _ => panic!("{input:?} expected /show abc123"),
        }
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
fn parses_fetch_commands() {
    // Bare command (and natural-language aliases) fetch the default remote.
    assert!(matches!(
        parse_local_command("/fetch"),
        Some(LocalCommand::Fetch(None))
    ));
    assert!(matches!(
        parse_local_command("/fetch "),
        Some(LocalCommand::Fetch(None))
    ));
    assert!(matches!(
        parse_local_command("fetch"),
        Some(LocalCommand::Fetch(None))
    ));
    assert!(matches!(
        parse_local_command("git fetch"),
        Some(LocalCommand::Fetch(None))
    ));
    // A remote argument is captured verbatim, slash and natural forms alike.
    assert!(matches!(
        parse_local_command("/fetch upstream"),
        Some(LocalCommand::Fetch(Some(ref remote))) if remote == "upstream"
    ));
    assert!(matches!(
        parse_local_command("fetch upstream"),
        Some(LocalCommand::Fetch(Some(ref remote))) if remote == "upstream"
    ));
    assert!(matches!(
        parse_local_command("git fetch upstream"),
        Some(LocalCommand::Fetch(Some(ref remote))) if remote == "upstream"
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
fn parses_issue_commands() {
    use crate::commands::IssueField;

    match parse_local_command("/issue reviewer 114 jesperpedersen") {
        Some(LocalCommand::Issue(Some(action))) => {
            assert_eq!(action.field, IssueField::Reviewer);
            assert_eq!(action.number, 114);
            assert_eq!(action.value, "jesperpedersen");
        }
        other => panic!("expected a reviewer action, got {:?}", other.is_some()),
    }

    // The field is case-insensitive and a leading `#` on the number is allowed.
    match parse_local_command("/issue Assignee #5 bob") {
        Some(LocalCommand::Issue(Some(action))) => {
            assert_eq!(action.field, IssueField::Assignee);
            assert_eq!(action.number, 5);
            assert_eq!(action.value, "bob");
        }
        _ => panic!("expected an assignee action"),
    }

    // A label value may carry spaces — it is the rest of the line.
    match parse_local_command("/issue label 7 needs triage") {
        Some(LocalCommand::Issue(Some(action))) => {
            assert_eq!(action.field, IssueField::Label);
            assert_eq!(action.number, 7);
            assert_eq!(action.value, "needs triage");
        }
        _ => panic!("expected a label action"),
    }

    // Missing pieces, an unknown field, or a non-numeric number are usage errors.
    for bad in [
        "/issue",
        "/issue reviewer",
        "/issue reviewer 114",
        "/issue bogus 1 x",
        "/issue reviewer notanumber x",
    ] {
        assert!(
            matches!(parse_local_command(bad), Some(LocalCommand::Issue(None))),
            "expected a usage error for {bad:?}"
        );
    }
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
                Some(LocalCommand::AutoReview(
                    AutoReviewTarget::Branch,
                    false,
                    false
                ))
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
                Some(LocalCommand::AutoReview(AutoReviewTarget::File(file), false, false)) if file == "src/tui.rs"
            ),
            "expected {input:?} to carry the path"
        );
    }

    // The `immediate` keyword starts the run at once — alone (whole branch) or
    // alongside a file, in either order.
    assert!(matches!(
        parse_local_command("/auto_review immediate"),
        Some(LocalCommand::AutoReview(
            AutoReviewTarget::Branch,
            true,
            false
        ))
    ));
    assert!(matches!(
        parse_local_command("auto review immediate"),
        Some(LocalCommand::AutoReview(
            AutoReviewTarget::Branch,
            true,
            false
        ))
    ));
    assert!(matches!(
        parse_local_command("/auto_review src/tui.rs immediate"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::File(file), true, false)) if file == "src/tui.rs"
    ));
    assert!(matches!(
        parse_local_command("/auto_review immediate src/tui.rs"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::File(file), true, false)) if file == "src/tui.rs"
    ));

    // The `all` keyword requests every project file — alone, with `immediate`
    // in either order, and its natural-language form.
    for input in ["/auto_review all", "auto review all", "/auto_review ALL"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::AutoReview(
                    AutoReviewTarget::All,
                    false,
                    false
                ))
            ),
            "expected {input:?} to parse as an All AutoReview"
        );
    }
    assert!(matches!(
        parse_local_command("/auto_review all immediate"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::All, true, false))
    ));
    assert!(matches!(
        parse_local_command("/auto_review immediate all"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::All, true, false))
    ));
    // `all` wins over a file argument if both are somehow given.
    assert!(matches!(
        parse_local_command("/auto_review src/tui.rs all"),
        Some(LocalCommand::AutoReview(
            AutoReviewTarget::All,
            false,
            false
        ))
    ));

    // The `deep` keyword starts every file in Deep mode — alone (whole
    // branch), with a file, with `all`, and its natural-language form —
    // mirroring `immediate` and `all` exactly, in any order.
    for input in ["/auto_review deep", "auto review deep", "/auto_review DEEP"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::AutoReview(
                    AutoReviewTarget::Branch,
                    false,
                    true
                ))
            ),
            "expected {input:?} to parse as a Deep whole-branch AutoReview"
        );
    }
    assert!(matches!(
        parse_local_command("/auto_review deep src/tui.rs"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::File(file), false, true)) if file == "src/tui.rs"
    ));
    assert!(matches!(
        parse_local_command("/auto_review src/tui.rs deep"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::File(file), false, true)) if file == "src/tui.rs"
    ));
    assert!(matches!(
        parse_local_command("/auto_review deep all"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::All, false, true))
    ));
    assert!(matches!(
        parse_local_command("/auto_review all deep"),
        Some(LocalCommand::AutoReview(AutoReviewTarget::All, false, true))
    ));
    // `deep`, `immediate`, and `all` combine freely in any order — the
    // longest accepted form.
    for input in [
        "/auto_review deep all immediate",
        "/auto_review all immediate deep",
        "/auto_review immediate deep all",
        "auto review deep all immediate",
    ] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::AutoReview(AutoReviewTarget::All, true, true))
            ),
            "expected {input:?} to parse as a Deep, immediate, All AutoReview"
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
    // Bare command (and natural-language aliases) rebase onto the default branch.
    assert!(matches!(
        parse_local_command("/rebase"),
        Some(LocalCommand::Rebase(None))
    ));
    assert!(matches!(
        parse_local_command("/rebase "),
        Some(LocalCommand::Rebase(None))
    ));
    assert!(matches!(
        parse_local_command("rebase"),
        Some(LocalCommand::Rebase(None))
    ));
    assert!(matches!(
        parse_local_command("Rebase"),
        Some(LocalCommand::Rebase(None))
    ));
    assert!(matches!(
        parse_local_command("git rebase"),
        Some(LocalCommand::Rebase(None))
    ));
    // An explicit target is captured verbatim across slash and natural forms,
    // including remote and remote-tracking-branch targets.
    assert!(matches!(
        parse_local_command("/rebase develop"),
        Some(LocalCommand::Rebase(Some(ref target))) if target == "develop"
    ));
    assert!(matches!(
        parse_local_command("rebase develop"),
        Some(LocalCommand::Rebase(Some(ref target))) if target == "develop"
    ));
    assert!(matches!(
        parse_local_command("git rebase upstream"),
        Some(LocalCommand::Rebase(Some(ref target))) if target == "upstream"
    ));
    assert!(matches!(
        parse_local_command("/rebase origin/main"),
        Some(LocalCommand::Rebase(Some(ref target))) if target == "origin/main"
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

#[test]
fn parses_pending_commands() {
    assert!(matches!(
        parse_local_command("/pending"),
        Some(LocalCommand::PendingList)
    ));
    assert!(matches!(
        parse_local_command("/pending list"),
        Some(LocalCommand::PendingList)
    ));
    assert!(matches!(
        parse_local_command("pending"),
        Some(LocalCommand::PendingList)
    ));
    assert!(matches!(
        parse_local_command("list pending"),
        Some(LocalCommand::PendingList)
    ));
    assert!(matches!(
        parse_local_command("show pending"),
        Some(LocalCommand::PendingList)
    ));
    assert!(matches!(
        parse_local_command("/pending delete"),
        Some(LocalCommand::PendingDelete(None))
    ));
    match parse_local_command("/pending delete 2") {
        Some(LocalCommand::PendingDelete(Some(2))) => {}
        _ => panic!("expected pending delete 2"),
    }
    match parse_local_command("/pending delete 1") {
        Some(LocalCommand::PendingDelete(Some(1))) => {}
        _ => panic!("expected pending delete 1"),
    }
}

/// `/create_workspace <dir>` and `/delete_workspace` parse into the correct
/// variants and do not shadow branch deletion for any other argument.
#[test]
fn parses_create_and_delete_workspace_commands() {
    // Bare /create_workspace (no argument yet) triggers the CreateWorkspace ghost.
    assert!(matches!(
        parse_local_command("/create_workspace"),
        Some(LocalCommand::CreateWorkspace(ref dir)) if dir.is_empty()
    ));

    // /create_workspace with a path.
    match parse_local_command("/create_workspace ~/project") {
        Some(LocalCommand::CreateWorkspace(dir)) => assert_eq!(dir.as_ref(), "~/project"),
        _ => panic!("expected /create_workspace ~/project"),
    }
    match parse_local_command("/create_workspace /abs/path") {
        Some(LocalCommand::CreateWorkspace(dir)) => assert_eq!(dir.as_ref(), "/abs/path"),
        _ => panic!("expected /create_workspace /abs/path"),
    }

    // /delete_workspace closes the current tab.
    assert!(matches!(
        parse_local_command("/delete_workspace"),
        Some(LocalCommand::DeleteWorkspace)
    ));

    // /delete <branch> still routes to branch deletion, never workspace close.
    assert!(matches!(
        parse_local_command("/delete feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));

    // Natural-language: `create workspace <dir>`.
    match parse_local_command("create workspace ~/project") {
        Some(LocalCommand::CreateWorkspace(dir)) => assert_eq!(dir.as_ref(), "~/project"),
        _ => panic!("expected natural 'create workspace ~/project'"),
    }
    match parse_local_command("CREATE WORKSPACE /abs/path") {
        Some(LocalCommand::CreateWorkspace(dir)) => assert_eq!(dir.as_ref(), "/abs/path"),
        _ => panic!("expected case-insensitive natural create workspace"),
    }

    // Natural-language: `delete workspace`.
    for input in ["delete workspace", "Delete Workspace", "DELETE WORKSPACE"] {
        assert!(
            matches!(
                parse_local_command(input),
                Some(LocalCommand::DeleteWorkspace)
            ),
            "expected {input:?} to parse as DeleteWorkspace"
        );
    }

    // `delete <branch>` must still work in the natural-language path.
    assert!(matches!(
        parse_local_command("delete feature/foo"),
        Some(LocalCommand::Branch(BranchSubcommand::Delete(_)))
    ));
}

/// `/bisect` and its natural-language aliases parse into the right
/// [`BisectSubcommand`], covering the optional commit/rev arguments, the
/// case-insensitive and whitespace-tolerant slash forms, and the fall-through
/// to `Status` for a bare or unrecognised subcommand.
#[test]
fn parses_bisect_commands() {
    // bare /bisect and explicit status subcommand both map to Status
    assert!(matches!(
        parse_local_command("/bisect"),
        Some(LocalCommand::Bisect(BisectSubcommand::Status))
    ));
    assert!(matches!(
        parse_local_command("/bisect status"),
        Some(LocalCommand::Bisect(BisectSubcommand::Status))
    ));
    // subcommands without arguments
    assert!(matches!(
        parse_local_command("/bisect start"),
        Some(LocalCommand::Bisect(BisectSubcommand::Start(None)))
    ));
    assert!(matches!(
        parse_local_command("/bisect good"),
        Some(LocalCommand::Bisect(BisectSubcommand::Good(None)))
    ));
    assert!(matches!(
        parse_local_command("/bisect bad"),
        Some(LocalCommand::Bisect(BisectSubcommand::Bad(None)))
    ));
    assert!(matches!(
        parse_local_command("/bisect skip"),
        Some(LocalCommand::Bisect(BisectSubcommand::Skip(None)))
    ));
    assert!(matches!(
        parse_local_command("/bisect reset"),
        Some(LocalCommand::Bisect(BisectSubcommand::Reset))
    ));
    assert!(matches!(
        parse_local_command("/bisect log"),
        Some(LocalCommand::Bisect(BisectSubcommand::Log))
    ));
    // subcommands with an explicit commit argument
    match parse_local_command("/bisect good abc123") {
        Some(LocalCommand::Bisect(BisectSubcommand::Good(Some(c)))) if c == "abc123" => {}
        _ => panic!("expected /bisect good abc123"),
    }
    match parse_local_command("/bisect bad deadbeef") {
        Some(LocalCommand::Bisect(BisectSubcommand::Bad(Some(c)))) if c == "deadbeef" => {}
        _ => panic!("expected /bisect bad deadbeef"),
    }
    match parse_local_command("/bisect skip abc123") {
        Some(LocalCommand::Bisect(BisectSubcommand::Skip(Some(c)))) if c == "abc123" => {}
        _ => panic!("expected /bisect skip abc123"),
    }
    // /bisect start accepts optional bad/good rev-range args
    match parse_local_command("/bisect start v1.0 HEAD") {
        Some(LocalCommand::Bisect(BisectSubcommand::Start(Some(a)))) if a == "v1.0 HEAD" => {}
        _ => panic!("expected /bisect start v1.0 HEAD"),
    }
    // subcommand matching is case-insensitive and tolerates extra whitespace
    match parse_local_command("/bisect GOOD abc123") {
        Some(LocalCommand::Bisect(BisectSubcommand::Good(Some(c)))) if c == "abc123" => {}
        _ => panic!("expected case-insensitive /bisect GOOD abc123"),
    }
    match parse_local_command("/bisect skip   abc123") {
        Some(LocalCommand::Bisect(BisectSubcommand::Skip(Some(c)))) if c == "abc123" => {}
        _ => panic!("expected /bisect skip to ignore extra spaces"),
    }
    // a longer word that merely starts with a known verb is not mistaken for it
    assert!(matches!(
        parse_local_command("/bisect starts"),
        Some(LocalCommand::Bisect(BisectSubcommand::Status))
    ));
    // natural-language forms
    assert!(matches!(
        parse_local_command("bisect start"),
        Some(LocalCommand::Bisect(BisectSubcommand::Start(None)))
    ));
    assert!(matches!(
        parse_local_command("start bisect"),
        Some(LocalCommand::Bisect(BisectSubcommand::Start(None)))
    ));
    assert!(matches!(
        parse_local_command("mark good"),
        Some(LocalCommand::Bisect(BisectSubcommand::Good(None)))
    ));
    assert!(matches!(
        parse_local_command("mark bad"),
        Some(LocalCommand::Bisect(BisectSubcommand::Bad(None)))
    ));
    assert!(matches!(
        parse_local_command("skip commit"),
        Some(LocalCommand::Bisect(BisectSubcommand::Skip(None)))
    ));
    assert!(matches!(
        parse_local_command("bisect reset"),
        Some(LocalCommand::Bisect(BisectSubcommand::Reset))
    ));
    assert!(matches!(
        parse_local_command("reset bisect"),
        Some(LocalCommand::Bisect(BisectSubcommand::Reset))
    ));
    assert!(matches!(
        parse_local_command("bisect log"),
        Some(LocalCommand::Bisect(BisectSubcommand::Log))
    ));
    assert!(matches!(
        parse_local_command("bisect"),
        Some(LocalCommand::Bisect(BisectSubcommand::Status))
    ));
    assert!(matches!(
        parse_local_command("git bisect"),
        Some(LocalCommand::Bisect(BisectSubcommand::Status))
    ));
}

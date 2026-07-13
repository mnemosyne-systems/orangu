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
use std::borrow::Cow;

/// Parse the argument string of `/auto_review` into `(target, immediate,
/// deep)`: the `immediate` keyword (case-insensitive) requests an at-once
/// start, the `all` keyword (case-insensitive) requests a review of every
/// project file, the `deep` keyword (case-insensitive) requests every file
/// start in Deep mode, and otherwise the first remaining token, if any, is
/// the single-file target. So `immediate`, `all`, `deep`, `src/main.rs`,
/// `src/main.rs immediate`, and `deep all immediate` are all accepted, in any
/// order; `all` wins over a file argument if both are somehow given.
pub(crate) fn parse_auto_review_args(args: &str) -> (AutoReviewTarget<'_>, bool, bool) {
    let mut file = None;
    let mut immediate = false;
    let mut all = false;
    let mut deep = false;
    for token in args.split_whitespace() {
        if token.eq_ignore_ascii_case(AUTO_REVIEW_IMMEDIATE) {
            immediate = true;
        } else if token.eq_ignore_ascii_case(AUTO_REVIEW_ALL) {
            all = true;
        } else if token.eq_ignore_ascii_case(AUTO_REVIEW_DEEP) {
            deep = true;
        } else if file.is_none() {
            file = Some(token);
        }
    }
    let target = if all {
        AutoReviewTarget::All
    } else if let Some(file) = file {
        AutoReviewTarget::File(Cow::Borrowed(file))
    } else {
        AutoReviewTarget::Branch
    };
    (target, immediate, deep)
}

pub fn parse_slash_command(input: &str) -> Option<LocalCommand<'_>> {
    match input {
        "/help" => Some(LocalCommand::Help),
        "/disconnect" => Some(LocalCommand::Disconnect),
        "/reload" => Some(LocalCommand::Reload),
        "/restart" => Some(LocalCommand::Restart),
        "/model" => Some(LocalCommand::ModelInfo),
        "/server" => Some(LocalCommand::ServerInfo),
        "/theme" => Some(LocalCommand::SetTheme("")),
        "/information" => Some(LocalCommand::Information),

        "/verbosity" => Some(LocalCommand::SetVerbosity("")),
        "/tools" => Some(LocalCommand::Tools),
        "/session" => Some(LocalCommand::Session(None)),
        "/workspace" => Some(LocalCommand::Workspace(None)),
        "/list_files" => Some(LocalCommand::ListFiles),
        "/open_file" => Some(LocalCommand::OpenFile("")),
        "/show_file" => Some(LocalCommand::ShowFile(Cow::Borrowed(""))),
        "/build" => Some(LocalCommand::Build(crate::build::BuildRequest::default())),
        "/shell" => Some(LocalCommand::Shell(None)),
        "/create_file" => Some(LocalCommand::CreateFile(None)),
        "/create_directory" => Some(LocalCommand::CreateDirectory(None)),
        "/move_directory" => Some(LocalCommand::MoveDirectory(None)),
        "/delete_directory" => Some(LocalCommand::DeleteDirectory(None)),
        "/amend" => Some(LocalCommand::Amend(None)),
        "/branch" => Some(LocalCommand::Branch(BranchSubcommand::List)),
        "/cherry_pick" => Some(LocalCommand::CherryPick(None)),
        "/commit" => Some(LocalCommand::Commit(None)),
        "/restore" => Some(LocalCommand::Restore(None)),
        "/diff" => Some(LocalCommand::Diff(None)),
        "/grep" => Some(LocalCommand::Grep(None)),
        "/search" => Some(LocalCommand::Search(None)),
        "/init_repo" => Some(LocalCommand::InitRepo),
        "/log" => Some(LocalCommand::Log(None)),
        "/show" => Some(LocalCommand::Show(None)),
        "/fetch" => Some(LocalCommand::Fetch(None)),
        "/merge" => Some(LocalCommand::Merge(None)),
        "/move_file" => Some(LocalCommand::MoveFile(None)),
        "/pull" => Some(LocalCommand::Pull(None)),
        "/comment" => Some(LocalCommand::Comment(None)),
        "/close" => Some(LocalCommand::Close(None)),
        "/issue" => Some(LocalCommand::Issue(None)),
        "/get_comments" => Some(LocalCommand::GetComments(None)),
        "/prune" => Some(LocalCommand::Prune(None)),
        "/pull_request" => Some(LocalCommand::CreatePullRequest),
        "/review" => Some(LocalCommand::Review),
        "/auto_review" => Some(LocalCommand::AutoReview(
            AutoReviewTarget::Branch,
            false,
            false,
        )),
        "/duplicates" => Some(LocalCommand::Duplicates(None)),
        "/export" => Some(LocalCommand::Export(ExportTarget::Console)),
        "/push" => Some(LocalCommand::Push(false)),
        "/rebase" => Some(LocalCommand::Rebase(None)),
        "/delete_file" => Some(LocalCommand::DeleteFile(None)),
        "/squash" => Some(LocalCommand::Squash),
        "/stash" => Some(LocalCommand::Stash(StashSubcommand::Push)),
        "/bisect" => Some(LocalCommand::Bisect(BisectSubcommand::Status)),
        "/status" => Some(LocalCommand::Status),
        "/create_workspace" => Some(LocalCommand::CreateWorkspace(Cow::Borrowed(""))),
        "/delete_workspace" => Some(LocalCommand::DeleteWorkspace),
        "/manual" => Some(LocalCommand::Manual),
        "/usage" => Some(LocalCommand::Usage),
        "/statistics" => Some(LocalCommand::Statistics(false)),
        "/schedule" => Some(LocalCommand::Schedule),
        "/clear" => Some(LocalCommand::Clear),
        "/quit" => Some(LocalCommand::Quit),
        "/pending" => Some(LocalCommand::PendingList),
        "/skills" => Some(LocalCommand::Skills),
        "/graph" => Some(LocalCommand::Graph),
        _ => {
            if let Some(args) = input.strip_prefix("/session ") {
                let uuid = args.trim();
                return Some(LocalCommand::Session(if uuid.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(uuid))
                }));
            }
            if let Some(args) = input.strip_prefix("/workspace ") {
                let arg = args.trim();
                return Some(LocalCommand::Workspace(if arg.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(arg))
                }));
            }
            if let Some(args) = input.strip_prefix("/diff ") {
                let branch = args.trim();
                return Some(LocalCommand::Diff(if branch.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(branch))
                }));
            }
            if let Some(args) = input.strip_prefix("/grep ") {
                let pattern = args.trim();
                return Some(LocalCommand::Grep(if pattern.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(pattern))
                }));
            }
            if let Some(args) = input.strip_prefix("/search ") {
                let query = args.trim();
                return Some(LocalCommand::Search(if query.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(query))
                }));
            }
            if let Some(name) = input.strip_prefix("/model ") {
                return Some(LocalCommand::SetModelId(name.trim()));
            }
            if let Some(name) = input.strip_prefix("/server ") {
                return Some(LocalCommand::SetServer(name.trim()));
            }
            if let Some(name) = input.strip_prefix("/theme ") {
                return Some(LocalCommand::SetTheme(name.trim()));
            }

            if let Some(level) = input.strip_prefix("/verbosity ") {
                return Some(LocalCommand::SetVerbosity(level.trim()));
            }
            if let Some(args) = input.strip_prefix("/show_file ") {
                return Some(LocalCommand::ShowFile(Cow::Borrowed(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/auto_review ") {
                let (target, immediate, deep) = parse_auto_review_args(args.trim());
                return Some(LocalCommand::AutoReview(target, immediate, deep));
            }
            if let Some(args) = input.strip_prefix("/export ") {
                return parse_export_target(args.trim()).map(LocalCommand::Export);
            }
            if let Some(args) = input.strip_prefix("/statistics ") {
                return match args.trim().to_ascii_lowercase().as_str() {
                    "total" => Some(LocalCommand::Statistics(true)),
                    _ => None,
                };
            }
            if let Some(args) = input.strip_prefix("/build ") {
                return crate::build::BuildRequest::parse(args).map(LocalCommand::Build);
            }
            if let Some(args) = input.strip_prefix("/shell ") {
                let command = args.trim();
                return Some(LocalCommand::Shell(if command.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(command))
                }));
            }
            if let Some(args) = input.strip_prefix("/duplicates ") {
                return Some(LocalCommand::Duplicates(parse_similarity_threshold(
                    args.trim(),
                )));
            }
            if let Some(args) = input.strip_prefix("/log ") {
                return Some(LocalCommand::Log(args.trim().parse::<u64>().ok()));
            }
            if let Some(args) = input.strip_prefix("/show ") {
                let commit = args.trim();
                return Some(LocalCommand::Show(if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                }));
            }
            if let Some(args) = input.strip_prefix("/fetch ") {
                let remote = args.trim();
                return Some(LocalCommand::Fetch(if remote.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(remote))
                }));
            }
            if let Some(args) = input.strip_prefix("/pull ") {
                return Some(LocalCommand::Pull(args.trim().parse::<u64>().ok()));
            }
            if let Some(args) = input.strip_prefix("/comment ") {
                return Some(LocalCommand::Comment(parse_comment_args(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/close ") {
                return Some(LocalCommand::Close(parse_close_args(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/issue ") {
                return Some(LocalCommand::Issue(parse_issue_args(args)));
            }
            if let Some(args) = input.strip_prefix("/get_comments ") {
                return Some(LocalCommand::GetComments(parse_get_comments_args(
                    args.trim(),
                )));
            }
            if let Some(args) = input.strip_prefix("/prune ") {
                return Some(LocalCommand::Prune(parse_prune_args(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/merge ") {
                let branch = args.trim();
                return Some(LocalCommand::Merge(if branch.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(branch))
                }));
            }
            if let Some(args) = input.strip_prefix("/rebase ") {
                let target = args.trim();
                return Some(LocalCommand::Rebase(if target.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(target))
                }));
            }
            if let Some(args) = input.strip_prefix("/checkout ") {
                let target = args.trim();
                if !target.is_empty() {
                    return Some(LocalCommand::Branch(BranchSubcommand::Switch(
                        Cow::Borrowed(target),
                    )));
                }
            }
            if let Some(args) = input.strip_prefix("/create_file ") {
                return Some(LocalCommand::CreateFile(parse_create_file_args(args)));
            }
            if let Some(args) = input.strip_prefix("/delete_file ") {
                let path = args.trim();
                return Some(LocalCommand::DeleteFile(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
            }
            if let Some(args) = input.strip_prefix("/create_directory ") {
                return Some(LocalCommand::CreateDirectory(parse_path_with_mode(args)));
            }
            if let Some(args) = input.strip_prefix("/delete_directory ") {
                let path = args.trim();
                return Some(LocalCommand::DeleteDirectory(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
            }
            if let Some(args) = input.strip_prefix("/move_directory ") {
                let args = args.trim();
                return Some(match shell_words(args) {
                    Ok(words) if words.len() >= 2 => LocalCommand::MoveDirectory(Some((
                        Cow::Owned(words[0].clone()),
                        Cow::Owned(words[1].clone()),
                    ))),
                    _ => LocalCommand::MoveDirectory(None),
                });
            }
            if let Some(args) = input.strip_prefix("/move_file ") {
                let args = args.trim();
                return Some(match shell_words(args) {
                    Ok(words) if words.len() >= 2 => LocalCommand::MoveFile(Some((
                        Cow::Owned(words[0].clone()),
                        Cow::Owned(words[1].clone()),
                    ))),
                    _ => LocalCommand::MoveFile(None),
                });
            }
            if let Some(args) = input.strip_prefix("/cherry_pick ") {
                let commit = args.trim();
                return Some(LocalCommand::CherryPick(if commit.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(commit))
                }));
            }
            if let Some(args) = input.strip_prefix("/commit ") {
                let message = strip_matching_quotes(args.trim());
                return Some(LocalCommand::Commit(if message.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(message))
                }));
            }
            if let Some(args) = input.strip_prefix("/amend ") {
                let message = strip_matching_quotes(args.trim());
                return Some(LocalCommand::Amend(if message.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(message))
                }));
            }
            if let Some(flag) = input.strip_prefix("/push ") {
                let flag = flag.trim();
                if flag == "--force" || flag == "-f" || flag.eq_ignore_ascii_case("force") {
                    return Some(LocalCommand::Push(true));
                }
            }
            if let Some(sub) = input.strip_prefix("/branch ") {
                let sub = sub.trim();
                return Some(LocalCommand::Branch(match sub {
                    "-a" | "--all" => BranchSubcommand::ListAll,
                    _ if sub.starts_with("-b ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Create(Cow::Borrowed(name))
                        }
                    }
                    _ if sub.starts_with("-m ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Rename(Cow::Borrowed(name))
                        }
                    }
                    _ if sub.starts_with("-d ") => {
                        let name = sub[3..].trim();
                        if name.is_empty() {
                            BranchSubcommand::List
                        } else {
                            BranchSubcommand::Delete(Cow::Borrowed(name))
                        }
                    }
                    _ if !sub.is_empty() => BranchSubcommand::Switch(Cow::Borrowed(sub)),
                    _ => BranchSubcommand::List,
                }));
            }
            if let Some(path) = input.strip_prefix("/restore ") {
                let path = path.trim();
                let staged = path.starts_with("--staged ") || path.starts_with("-S ");
                let file = if staged {
                    path.split_once(' ').map(|x| x.1).unwrap_or("").trim()
                } else {
                    path
                };
                return Some(LocalCommand::Restore(if file.is_empty() {
                    None
                } else {
                    Some(Cow::Owned(format!(
                        "{}{}",
                        if staged { "--staged " } else { "" },
                        file
                    )))
                }));
            }
            if let Some(sub) = input.strip_prefix("/stash ") {
                return Some(LocalCommand::Stash(match sub.trim() {
                    "pop" => StashSubcommand::Pop,
                    "list" => StashSubcommand::List,
                    "drop" => StashSubcommand::Drop,
                    _ => StashSubcommand::Push,
                }));
            }
            if let Some(sub) = input.strip_prefix("/bisect ") {
                return Some(LocalCommand::Bisect(parse_bisect_subcommand(sub)));
            }
            if let Some(dir) = input.strip_prefix("/create_workspace ") {
                return Some(LocalCommand::CreateWorkspace(Cow::Borrowed(dir.trim())));
            }
            if input.strip_prefix("/delete_workspace ").is_some() {
                return Some(LocalCommand::DeleteWorkspace);
            }
            if let Some(args) = input.strip_prefix("/delete ") {
                let branch = args.trim();
                if !branch.is_empty() {
                    return Some(LocalCommand::Branch(BranchSubcommand::Delete(
                        Cow::Borrowed(branch),
                    )));
                }
            }
            if let Some(args) = input.strip_prefix("/open_file ")
                && args.trim().is_empty()
            {
                return Some(LocalCommand::OpenFile(""));
            }
            if let Some(args) = input.strip_prefix("/pending ") {
                let args = args.trim();
                return Some(if args.eq_ignore_ascii_case("list") || args.is_empty() {
                    LocalCommand::PendingList
                } else if let Some(rest) = strip_ascii_prefix(args, "delete") {
                    LocalCommand::PendingDelete(rest.trim().parse::<usize>().ok())
                } else {
                    LocalCommand::PendingList
                });
            }
            parse_open_file_target(input, "/open_file ").map(LocalCommand::OpenFile)
        }
    }
}

/// Parse the text after `/bisect ` into a [`BisectSubcommand`].
///
/// The first whitespace-delimited word selects the subcommand
/// (case-insensitively); any text after it is the optional commit/rev argument
/// for `start`, `good`, `bad`, and `skip`. Matching the whole word — rather than
/// a bare prefix — keeps inputs like `started` from being read as `start`. An
/// empty or unrecognised verb maps to [`BisectSubcommand::Status`], mirroring a
/// bare `/bisect`.
pub(super) fn parse_bisect_subcommand(sub: &str) -> BisectSubcommand<'_> {
    let sub = sub.trim();
    let (verb, rest) = match sub.split_once(char::is_whitespace) {
        Some((verb, rest)) => (verb, rest.trim()),
        None => (sub, ""),
    };
    let arg = if rest.is_empty() {
        None
    } else {
        Some(Cow::Borrowed(rest))
    };
    match verb.to_ascii_lowercase().as_str() {
        "start" => BisectSubcommand::Start(arg),
        "good" => BisectSubcommand::Good(arg),
        "bad" => BisectSubcommand::Bad(arg),
        "skip" => BisectSubcommand::Skip(arg),
        "reset" => BisectSubcommand::Reset,
        "log" => BisectSubcommand::Log,
        _ => BisectSubcommand::Status,
    }
}

/// `<path> [with <mode>]` — the shape both `/create_file` and
/// `/create_directory` take, and what the natural-language forms ("create
/// myfile.txt with 0644") parse down to. The mode is kept as typed; it is
/// `orangu::files` that validates it, so one octal parser covers every
/// surface.
pub(crate) fn parse_path_with_mode(args: &str) -> Option<(Cow<'_, str>, Option<Cow<'_, str>>)> {
    let parsed = parse_create_file_args(args)?;
    Some((parsed.path, parsed.mode))
}

/// `<path> [with <mode>] [containing <text>]` — what `/create_file` and its
/// natural-language forms ("create myfile.txt with 0644 containing hello")
/// take. `containing` is everything to the end of the line, so content needs
/// no quoting; the mode is kept as typed, since it is `orangu::files` that
/// validates it, one octal parser for every surface.
pub(crate) fn parse_create_file_args(args: &str) -> Option<CreateFileArgs<'_>> {
    let args = args.trim();
    if args.is_empty() {
        return None;
    }
    let (head, content) = match args.split_once(" containing ") {
        Some((head, content)) => (head.trim(), Some(content)),
        None => (args, None),
    };
    let (path, mode) = match head.split_once(" with ") {
        Some((path, mode)) => (path.trim(), Some(mode.trim())),
        None => (head, None),
    };
    if path.is_empty() {
        return None;
    }
    Some(CreateFileArgs {
        path: Cow::Borrowed(path),
        mode: mode.filter(|mode| !mode.is_empty()).map(Cow::Borrowed),
        content: content
            .map(str::trim_end)
            .filter(|content| !content.is_empty())
            .map(Cow::Borrowed),
    })
}

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

pub fn parse_slash_command(input: &str) -> Option<LocalCommand<'_>> {
    match input {
        "/help" => Some(LocalCommand::Help),
        "/disconnect" => Some(LocalCommand::Disconnect),
        "/reload" => Some(LocalCommand::Reload),
        "/restart" => Some(LocalCommand::Restart),
        "/tools" => Some(LocalCommand::Tools),
        "/model" => Some(LocalCommand::ModelInfo),
        "/server" => Some(LocalCommand::ServerInfo),
        "/session" => Some(LocalCommand::Session(None)),
        "/workspace" => Some(LocalCommand::Workspace(None)),
        "/list_files" => Some(LocalCommand::ListFiles),
        "/open_file" => Some(LocalCommand::OpenFile("")),
        "/show_file" => Some(LocalCommand::ShowFile(Cow::Borrowed(""))),
        "/build" => Some(LocalCommand::Build),
        "/add_file" => Some(LocalCommand::AddFile(None)),
        "/amend" => Some(LocalCommand::Amend(None)),
        "/branch" => Some(LocalCommand::Branch(BranchSubcommand::List)),
        "/cherry_pick" => Some(LocalCommand::CherryPick(None)),
        "/commit" => Some(LocalCommand::Commit(None)),
        "/restore" => Some(LocalCommand::Restore(None)),
        "/diff" => Some(LocalCommand::Diff(None)),
        "/grep" => Some(LocalCommand::Grep(None)),
        "/init_repo" => Some(LocalCommand::InitRepo),
        "/log" => Some(LocalCommand::Log(None)),
        "/fetch" => Some(LocalCommand::Fetch(None)),
        "/merge" => Some(LocalCommand::Merge(None)),
        "/move_file" => Some(LocalCommand::MoveFile(None)),
        "/pull" => Some(LocalCommand::Pull(None)),
        "/comment" => Some(LocalCommand::Comment(None)),
        "/close" => Some(LocalCommand::Close(None)),
        "/get_comments" => Some(LocalCommand::GetComments(None)),
        "/prune" => Some(LocalCommand::Prune(None)),
        "/pull_request" => Some(LocalCommand::CreatePullRequest),
        "/review" => Some(LocalCommand::Review),
        "/auto_review" => Some(LocalCommand::AutoReview(None)),
        "/export" => Some(LocalCommand::Export(ExportTarget::Console)),
        "/push" => Some(LocalCommand::Push(false)),
        "/rebase" => Some(LocalCommand::Rebase(None)),
        "/remove_file" => Some(LocalCommand::RemoveFile(None)),
        "/squash" => Some(LocalCommand::Squash),
        "/stash" => Some(LocalCommand::Stash(StashSubcommand::Push)),
        "/bisect" => Some(LocalCommand::Bisect(BisectSubcommand::Status)),
        "/status" => Some(LocalCommand::Status),
        "/manual" => Some(LocalCommand::Manual),
        "/usage" => Some(LocalCommand::Usage),
        "/clear" => Some(LocalCommand::Clear),
        "/quit" => Some(LocalCommand::Quit),
        "/pending" => Some(LocalCommand::PendingList),
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
            if let Some(name) = input.strip_prefix("/model ") {
                return Some(LocalCommand::SetModelId(name.trim()));
            }
            if let Some(name) = input.strip_prefix("/server ") {
                return Some(LocalCommand::SetServer(name.trim()));
            }
            if let Some(args) = input.strip_prefix("/show_file ") {
                return Some(LocalCommand::ShowFile(Cow::Borrowed(args.trim())));
            }
            if let Some(args) = input.strip_prefix("/auto_review ") {
                let file = args.trim();
                return Some(LocalCommand::AutoReview(if file.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(file))
                }));
            }
            if let Some(args) = input.strip_prefix("/export ") {
                return parse_export_target(args.trim()).map(LocalCommand::Export);
            }
            if let Some(args) = input.strip_prefix("/log ") {
                return Some(LocalCommand::Log(args.trim().parse::<u64>().ok()));
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
            if let Some(args) = input.strip_prefix("/add_file ") {
                let path = args.trim();
                return Some(LocalCommand::AddFile(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
            }
            if let Some(args) = input.strip_prefix("/remove_file ") {
                let path = args.trim();
                return Some(LocalCommand::RemoveFile(if path.is_empty() {
                    None
                } else {
                    Some(Cow::Borrowed(path))
                }));
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

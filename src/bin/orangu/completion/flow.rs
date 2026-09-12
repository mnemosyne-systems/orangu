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

//! The merge flow: the fixed sequence of commands that takes a pull request
//! from checkout to merged, and which drives the inline ghost hint and the Tab
//! completions so the next step is always the first thing offered.
//!
//! ```text
//! pull <number>            -> remembers the request and the branch it checked out
//! build | review | auto review   (optional, while still on the branch)
//! switch to main|master    -> the base branch the request merges into
//! merge <branch>
//! push
//! delete <branch>
//! comment on <number> merged.md
//! close pr <number>         (only after a rebase on the branch)
//! close issue <number>      (only after a rebase, and only when the request's
//!                            title names an issue)
//! ```
//!
//! The optional steps are offered but never advance anything: the flow is
//! remembered across them, so the next thing suggested after a build or a
//! review is still the step the request is actually waiting on.
//!
//! A `rebase` run on the pulled branch rewrites its commits, so the forge no
//! longer recognises the merge as the request's and leaves it open. The flow
//! then adds the two closing steps: the request itself, and the issue it
//! refers to (`[#45]`, `#45`, `issue 45`) in the subject of one of its
//! commits or in its title, when it names one.
//!
//! [`start`] is called when `pull` checks a request out; each following step is
//! confirmed by the `note_*` hooks in `dispatch`, which only fire once the
//! command they observe actually succeeded. The state lives in a process-wide
//! lock because completion runs on every keystroke, far from the dispatcher.
//! When the last step lands, the request is dropped from the cached open
//! requests and a refetch is requested (see [`take_refresh_request`]), so the
//! empty prompt does not offer to pull a request that has just been merged.

use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::commands::CloseTarget;
use crate::git::{
    discover_git_root, git_branch_subjects, git_find_base_ref, is_protected_branch,
    workspace_branch_name,
};

/// The comment template the flow posts once the branch is merged and deleted,
/// read from `~/.orangu/comments/` by the `comment` command.
const MERGED_COMMENT_FILE: &str = "merged.md";

/// How many `pull <number>` suggestions the empty prompt offers before the flow
/// starts. The ghost cycles through these with Shift+Tab, so the list is kept
/// short; `pull <TAB>` still completes against every open request.
const PULL_SUGGESTION_LIMIT: usize = 5;

/// What you might reasonably do to a request you have just checked out, before
/// taking it to the base branch: build it, review it, have the model review it.
/// Offered alongside the flow while it sits on the pulled branch, but never as
/// the first suggestion and never advancing anything — running one leaves the
/// next flow step exactly where it was. Spelled as the natural-language
/// bindings, like every other step.
const OPTIONAL_STEPS: &[&str] = &["build", "review", "auto review"];

/// One step of the merge flow, in the order they are performed. The ordering is
/// derived so a step can only ever move forward.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MergeFlowStep {
    /// Leave the pull request's branch for the base branch it merges into.
    Switch,
    /// Merge the request's branch into the base branch.
    Merge,
    /// Publish the merge.
    Push,
    /// Drop the now-merged branch.
    Delete,
    /// Tell the request it was merged.
    Comment,
    /// Close the request the forge could not tie to the merge, because the
    /// branch was rebased.
    ClosePullRequest,
    /// Close the issue the request's title refers to, for the same reason.
    CloseIssue,
}

impl MergeFlowStep {
    /// Every step of the flow, in order.
    const ALL: [MergeFlowStep; 7] = {
        use MergeFlowStep::*;
        [
            Switch,
            Merge,
            Push,
            Delete,
            Comment,
            ClosePullRequest,
            CloseIssue,
        ]
    };

    /// Every step from this one to the end of the flow, in order.
    fn remaining(self) -> &'static [MergeFlowStep] {
        let from = Self::ALL.iter().position(|step| *step == self).unwrap_or(0);
        &Self::ALL[from..]
    }

    /// The steps that come after this one, in order.
    fn following(self) -> &'static [MergeFlowStep] {
        &self.remaining()[1..]
    }
}

/// A pull request being merged: the request itself, the branch `pull` checked
/// out for it, the base branch it merges into, how far the flow has come, and
/// the workspace it all belongs to — the suggestions are only offered back in
/// that workspace, so another tab's prompt is not hinted at branches that only
/// exist here. `rebased` records that the branch was rebased under the flow,
/// which is what puts the closing steps on offer; `issue` is the issue the
/// request's commits or title name, if any.
#[derive(Clone, Debug)]
struct MergeFlow {
    pr: u64,
    branch: String,
    base: String,
    step: MergeFlowStep,
    workspace: PathBuf,
    rebased: bool,
    issue: Option<u64>,
}

impl MergeFlow {
    /// The command that performs `step` for this request, spelled the way the
    /// natural-language parser accepts it — or `None` when the step does not
    /// apply to this request: the closing steps exist only once the branch
    /// has been rebased, and the issue one only when the title names an issue.
    fn command(&self, step: MergeFlowStep) -> Option<String> {
        Some(match step {
            MergeFlowStep::Switch => format!("switch to {}", self.base),
            MergeFlowStep::Merge => format!("merge {}", self.branch),
            MergeFlowStep::Push => "push".to_string(),
            MergeFlowStep::Delete => format!("delete {}", self.branch),
            MergeFlowStep::Comment => format!("comment on {} {MERGED_COMMENT_FILE}", self.pr),
            MergeFlowStep::ClosePullRequest => {
                if !self.rebased {
                    return None;
                }
                format!("close pr {}", self.pr)
            }
            MergeFlowStep::CloseIssue => {
                if !self.rebased {
                    return None;
                }
                format!("close issue {}", self.issue?)
            }
        })
    }

    /// The step this request still has to take after `step`, skipping the
    /// ones that do not apply to it; `None` when `step` is its last.
    fn next_after(&self, step: MergeFlowStep) -> Option<MergeFlowStep> {
        step.following()
            .iter()
            .copied()
            .find(|next| self.command(*next).is_some())
    }

    /// The commands to offer, the step the flow is waiting on first — that one
    /// is the ghost, so the flow is what an untouched prompt always proposes.
    ///
    /// While the request is still checked out, [`OPTIONAL_STEPS`] follow it:
    /// they belong to this moment (the branch is under your hands) rather than
    /// to any later one, so they are cycled to before the steps that come after
    /// the switch. Once the flow has left the branch they are dropped.
    fn remaining_commands(&self) -> Vec<String> {
        let mut steps = self
            .step
            .remaining()
            .iter()
            .filter_map(|step| self.command(*step));
        let mut commands: Vec<String> = steps.next().into_iter().collect();
        if self.step == MergeFlowStep::Switch {
            commands.extend(OPTIONAL_STEPS.iter().map(|step| (*step).to_string()));
        }
        commands.extend(steps);
        commands
    }

    /// The argument values this flow prefers when completing a command
    /// argument, best first: the one the current step needs, then the rest.
    /// Used to reorder branch, number, and file candidates so the flow's own
    /// value is the one Tab picks.
    fn preferred_arguments(&self) -> Vec<String> {
        let pr = self.pr.to_string();
        let merged = MERGED_COMMENT_FILE.to_string();
        let mut preferred = match self.step {
            MergeFlowStep::Switch => vec![self.base.clone(), self.branch.clone()],
            MergeFlowStep::Merge | MergeFlowStep::Push | MergeFlowStep::Delete => {
                vec![self.branch.clone(), self.base.clone()]
            }
            MergeFlowStep::Comment => vec![
                pr.clone(),
                merged.clone(),
                self.branch.clone(),
                self.base.clone(),
            ],
            MergeFlowStep::ClosePullRequest => vec![pr.clone()],
            MergeFlowStep::CloseIssue => self.issue.iter().map(u64::to_string).collect(),
        };
        // The values no step is waiting for still beat an unrelated candidate.
        for value in [pr, merged] {
            if !preferred.contains(&value) {
                preferred.push(value);
            }
        }
        preferred
    }
}

/// The issue a commit subject or pull request title says it addresses: the
/// first `#<number>` in it, or the number following the word `issue`
/// (`issue 45`, `issue #45`), however the reference is bracketed or
/// punctuated — `[#45] Fix the prompt` included. `None` when it names no
/// issue.
pub(crate) fn issue_in_subject(subject: &str) -> Option<u64> {
    fn bare(word: &str) -> &str {
        word.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '#')
    }
    fn number(word: &str) -> Option<u64> {
        bare(word).trim_start_matches('#').parse().ok()
    }
    let words: Vec<&str> = subject.split_whitespace().map(bare).collect();
    for (index, word) in words.iter().enumerate() {
        if word.starts_with('#')
            && let Some(issue) = number(word)
        {
            return Some(issue);
        }
        if word.eq_ignore_ascii_case("issue")
            && let Some(issue) = words.get(index + 1).copied().and_then(number)
        {
            return Some(issue);
        }
    }
    None
}

/// The merge flow in progress, or `None` when no pull request has been checked
/// out this session. Read on every keystroke by the completion code, written
/// only by the `dispatch` hooks below.
static FLOW: RwLock<Option<MergeFlow>> = RwLock::new(None);

/// A copy of the flow in progress. A poisoned lock is recovered rather than
/// panicking — a missing hint is not worth taking the prompt down for.
fn snapshot() -> Option<MergeFlow> {
    FLOW.read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// The flow in progress, but only when `workspace` is the one it was started
/// in. Everything the prompt reads goes through here.
fn snapshot_for(workspace: &Path) -> Option<MergeFlow> {
    snapshot().filter(|flow| flow.workspace == workspace)
}

fn store(flow: Option<MergeFlow>) {
    *FLOW
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = flow;
}

/// Set once a flow has run its course: the open requests cached at startup
/// are stale from then on, and the main loop fetches them again (see
/// [`take_refresh_request`]).
static REFRESH_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Whether a merge flow has finished since the last call, in which case the
/// cached open pull requests need fetching again. Clears the request, so each
/// finished flow triggers one refetch.
pub fn take_refresh_request() -> bool {
    REFRESH_REQUESTED.swap(false, Ordering::AcqRel)
}

/// `step` was performed for the flow `matches` accepts: the flow moves on to
/// the next step that applies to the request, or — when there is none — ends,
/// with the landed request dropped from the cached open requests and a refetch
/// requested. Steps only ever move forward, so repeating a command does not
/// rewind the flow.
fn complete(step: MergeFlowStep, matches: impl FnOnce(&MergeFlow) -> bool) {
    let mut guard = FLOW
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(flow) = guard.as_mut() else { return };
    if !matches(flow) {
        return;
    }
    match flow.next_after(step) {
        Some(next) => {
            if flow.step < next {
                flow.step = next;
            }
        }
        None => {
            let pr = flow.pr;
            *guard = None;
            super::remove_active_pull_request(pr);
            REFRESH_REQUESTED.store(true, Ordering::Release);
        }
    }
}

/// Begin the flow for pull request `pr`, which has just been checked out into
/// the workspace's current branch. The base branch and the issue the request
/// refers to are resolved once, here, rather than on every keystroke: the
/// issue is read from the subjects of the commits the branch adds on top of
/// the base (newest first), then from the request's cached title.
///
/// Nothing is remembered when the branch cannot be determined (detached HEAD)
/// or when the checkout left the workspace on `main`/`master`, since there
/// would be no branch to merge and delete.
pub fn start(pr: u64, workspace: &Path) {
    let Some(branch) = workspace_branch_name(workspace) else {
        store(None);
        return;
    };
    if is_protected_branch(&branch) {
        store(None);
        return;
    }
    let base_ref = discover_git_root(workspace)
        .and_then(|root| git_find_base_ref(&root).ok().map(|base| (root, base)));
    let issue = base_ref
        .as_ref()
        .and_then(|(root, base)| {
            git_branch_subjects(root, base)
                .iter()
                .find_map(|subject| issue_in_subject(subject))
        })
        .or_else(|| {
            super::pull_request_title(pr)
                .as_deref()
                .and_then(issue_in_subject)
        });
    store(Some(MergeFlow {
        pr,
        branch,
        base: base_ref.map_or_else(|| "main".to_string(), |(_, base)| base_branch(&base)),
        step: MergeFlowStep::Switch,
        workspace: workspace.to_path_buf(),
        rebased: false,
        issue,
    }));
}

/// The local name of the branch behind `base_ref` — `main` for `origin/main`.
fn base_branch(base_ref: &str) -> String {
    base_ref
        .strip_prefix("origin/")
        .unwrap_or(base_ref)
        .to_string()
}

/// A branch was checked out. Reaching the base branch advances the flow;
/// leaving for an unrelated branch ends it, so the prompt stops suggesting
/// steps for a request the user has moved on from.
pub fn note_checkout(branch: &str) {
    let Some(flow) = snapshot() else { return };
    if branch == flow.base {
        complete(MergeFlowStep::Switch, |_| true);
    } else if branch != flow.branch {
        store(None);
    }
}

/// The branch checked out in `workspace` was rebased. When it is the flow's
/// branch, the forge will no longer tie the merge to the request, so the
/// closing steps join the flow.
pub fn note_rebase(workspace: &Path) {
    let Some(branch) = workspace_branch_name(workspace) else {
        return;
    };
    let mut guard = FLOW
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(flow) = guard.as_mut()
        && flow.workspace == workspace
        && flow.branch == branch
    {
        flow.rebased = true;
    }
}

/// The flow's branch was merged.
pub fn note_merge(branch: &str) {
    complete(MergeFlowStep::Merge, |flow| branch == flow.branch);
}

/// The merge was pushed. Only counts once the merge itself is done, so a push
/// made earlier in the flow does not skip a step.
pub fn note_push() {
    complete(MergeFlowStep::Push, |flow| flow.step == MergeFlowStep::Push);
}

/// The flow's branch was deleted.
pub fn note_delete(branch: &str) {
    complete(MergeFlowStep::Delete, |flow| branch == flow.branch);
}

/// The flow's pull request was commented on: the last step — unless the
/// branch was rebased, in which case the closing steps remain.
pub fn note_comment(pr: u64) {
    complete(MergeFlowStep::Comment, |flow| flow.pr == pr);
}

/// An issue or pull request was closed. Closing the flow's request moves on
/// to its issue, when the title names one, and closing that issue ends the
/// flow.
pub fn note_close(target: &CloseTarget) {
    match *target {
        CloseTarget::PullRequest(pr) => {
            complete(MergeFlowStep::ClosePullRequest, |flow| flow.pr == pr);
        }
        CloseTarget::Issue(issue) => {
            complete(MergeFlowStep::CloseIssue, |flow| flow.issue == Some(issue));
        }
    }
}

/// The flow commands `input` could still grow into, the next step first: the
/// remaining steps of the pull request being merged, or — before one is
/// checked out — `pull <number>` for the open requests fetched at startup.
///
/// Only the committer prompt (`/committer`) is walked through the flow; the
/// developer prompt is left alone, though the steps taken there are still
/// tracked so switching over picks up where the branch actually is.
///
/// Matching is ASCII case-insensitive from the start of the line, mirroring the
/// natural-language ghost, and a command already typed in full is dropped since
/// there is nothing left to suggest. Empty input matches everything, which is
/// what puts the next step on the empty prompt.
pub fn flow_candidates(input: &str, workspace: &Path) -> Vec<String> {
    if crate::mode::current() != crate::mode::PromptMode::Committer {
        return Vec::new();
    }
    let commands = match snapshot_for(workspace) {
        Some(flow) => flow.remaining_commands(),
        None => super::pull_number_candidates("")
            .into_iter()
            .take(PULL_SUGGESTION_LIMIT)
            .map(|number| format!("pull {number}"))
            .collect(),
    };
    commands
        .into_iter()
        .filter(|command| {
            command.len() > input.len()
                && command.as_bytes()[..input.len()].eq_ignore_ascii_case(input.as_bytes())
        })
        .collect()
}

/// Reorder command-argument candidates so the values the flow needs come first
/// (the merged branch for `merge `/`delete `, the base for `switch to `, the
/// request number and `merged.md` for `comment on `). The rest keep their
/// order, and nothing is added or dropped: this only decides what Tab and the
/// ghost reach for first. A no-op on the developer prompt, and while no flow is
/// in progress.
pub fn hoist_preferred_arguments(candidates: &mut [String], workspace: &Path) {
    if crate::mode::current() != crate::mode::PromptMode::Committer {
        return;
    }
    let Some(flow) = snapshot_for(workspace) else {
        return;
    };
    let preferred = flow.preferred_arguments();
    candidates.sort_by_key(|candidate| {
        preferred
            .iter()
            .position(|value| value == candidate)
            .unwrap_or(usize::MAX)
    });
}

/// Test-only access to the process-wide flow state, shared with the completion
/// tests that need a merge in progress to have something to suggest. Taking
/// [`crate::test_support::exclusive_prompt_state`] first is what keeps those
/// tests from tripping over each other.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Forget any flow in progress, and any refetch a finished one asked for.
    pub(crate) fn reset() {
        store(None);
        take_refresh_request();
    }

    /// Put a flow for pull request `pr` in progress at `step`, as if it had
    /// been pulled into `workspace` from branch `branch` onto base `main`.
    pub(crate) fn begin(pr: u64, branch: &str, step: MergeFlowStep, workspace: &Path) {
        begin_rebased(pr, branch, step, workspace, false, None);
    }

    /// [`begin`], with the branch already rebased under the flow (`rebased`)
    /// and the request's title naming `issue`.
    pub(crate) fn begin_rebased(
        pr: u64,
        branch: &str,
        step: MergeFlowStep,
        workspace: &Path,
        rebased: bool,
        issue: Option<u64>,
    ) {
        store(Some(MergeFlow {
            pr,
            branch: branch.to_string(),
            base: "main".to_string(),
            step,
            workspace: workspace.to_path_buf(),
            rebased,
            issue,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::set_active_pull_requests;
    use crate::git::PullRequest;
    use crate::test_support::exclusive_committer_prompt as exclusive;

    /// The workspace the tests' flow belongs to.
    const WORKSPACE: &str = "/tmp/orangu-flow-tests";

    fn workspace() -> &'static Path {
        Path::new(WORKSPACE)
    }

    fn begin(step: MergeFlowStep) {
        test_support::begin(231, "pr-231", step, workspace());
    }

    fn candidates(input: &str) -> Vec<String> {
        flow_candidates(input, workspace())
    }

    #[test]
    fn flow_offers_the_next_step_first_on_the_empty_prompt() {
        let _guard = exclusive();
        begin(MergeFlowStep::Switch);
        // The step the request is waiting on leads; what you might do to the
        // branch you just pulled follows; the later steps come after those.
        assert_eq!(
            candidates(""),
            vec![
                "switch to main",
                "build",
                "review",
                "auto review",
                "merge pr-231",
                "push",
                "delete pr-231",
                "comment on 231 merged.md",
            ]
        );
        // A part-typed line narrows to the steps that can still complete it,
        // and a step typed in full has nothing left to offer.
        assert_eq!(candidates("m"), vec!["merge pr-231"]);
        assert_eq!(candidates("MER"), vec!["merge pr-231"]);
        assert_eq!(candidates("merge pr-231"), Vec::<String>::new());
    }

    #[test]
    fn the_optional_steps_are_offered_only_while_the_branch_is_checked_out() {
        let _guard = exclusive();
        begin(MergeFlowStep::Switch);
        assert_eq!(candidates("b"), vec!["build"]);
        assert_eq!(candidates("auto"), vec!["auto review"]);

        // Running one changes nothing: the flow is remembered, so the next
        // suggestion is still the step the request is waiting on.
        for _ in 0..3 {
            assert_eq!(candidates("")[0], "switch to main");
        }

        // Once the flow has left the branch they are no longer on offer — the
        // branch they would act on is not the checked-out one any more.
        note_checkout("main");
        assert_eq!(
            candidates(""),
            vec![
                "merge pr-231",
                "push",
                "delete pr-231",
                "comment on 231 merged.md",
            ]
        );
        assert!(candidates("b").is_empty());
    }

    #[test]
    fn each_command_advances_the_flow_one_step() {
        let _guard = exclusive();
        begin(MergeFlowStep::Switch);
        note_checkout("main");
        assert_eq!(candidates("")[0], "merge pr-231");
        note_merge("pr-231");
        assert_eq!(candidates("")[0], "push");
        note_push();
        assert_eq!(candidates("")[0], "delete pr-231");
        note_delete("pr-231");
        assert_eq!(candidates("")[0], "comment on 231 merged.md");
        // Commenting on the request finishes the flow.
        note_comment(231);
        assert!(snapshot().is_none());
    }

    #[test]
    fn a_rebase_on_the_branch_adds_the_closing_steps() {
        let _guard = exclusive();
        let repo = tempfile::tempdir().expect("repo");
        crate::git::init_git_for_test(repo.path());
        std::fs::write(repo.path().join("README.md"), "orangu\n").expect("readme");
        crate::git::git_run(repo.path(), &["add", "README.md"]);
        crate::git::git_run(repo.path(), &["commit", "--quiet", "-m", "initial"]);
        crate::git::git_run(repo.path(), &["branch", "-M", "main"]);
        // The request's commit names the issue the way commit subjects do;
        // its title does not.
        crate::git::git_run(repo.path(), &["checkout", "--quiet", "-b", "pr-231"]);
        std::fs::write(repo.path().join("prompt.rs"), "").expect("prompt.rs");
        crate::git::git_run(repo.path(), &["add", "prompt.rs"]);
        crate::git::git_run(
            repo.path(),
            &["commit", "--quiet", "-m", "[#58] Leave room for the prompt"],
        );
        set_active_pull_requests(&[PullRequest {
            number: 231,
            title: "Leave room for the prompt".to_string(),
        }]);
        start(231, repo.path());
        let candidates = |input: &str| flow_candidates(input, repo.path());

        // Untouched, the flow ends with the comment: the forge closes the
        // request itself when the branch is merged as it was pulled.
        assert_eq!(candidates("").last().unwrap(), "comment on 231 merged.md");

        // A rebase in some other workspace is not this branch's rebase.
        note_rebase(workspace());
        assert_eq!(candidates("").last().unwrap(), "comment on 231 merged.md");

        // Rebased, the forge will not tie the merge to the request, so the
        // flow closes the request — and the issue its commit names — itself.
        note_rebase(repo.path());
        assert_eq!(
            candidates(""),
            vec![
                "switch to main",
                "build",
                "review",
                "auto review",
                "merge pr-231",
                "push",
                "delete pr-231",
                "comment on 231 merged.md",
                "close pr 231",
                "close issue 58",
            ]
        );
        note_checkout("main");
        note_merge("pr-231");
        note_push();
        note_delete("pr-231");
        note_comment(231);
        assert_eq!(candidates(""), vec!["close pr 231", "close issue 58"]);
        // Closing some other issue or request is not this flow's step.
        note_close(&CloseTarget::Issue(231));
        note_close(&CloseTarget::PullRequest(58));
        assert_eq!(candidates("")[0], "close pr 231");
        note_close(&CloseTarget::PullRequest(231));
        assert_eq!(candidates(""), vec!["close issue 58"]);
        assert!(!take_refresh_request(), "the flow is not over yet");
        note_close(&CloseTarget::Issue(58));
        assert!(snapshot().is_none());
        assert!(take_refresh_request());

        // A request whose commits name no issue falls back to its title...
        crate::git::git_run(
            repo.path(),
            &["checkout", "--quiet", "-b", "pr-232", "main"],
        );
        std::fs::write(repo.path().join("other.rs"), "").expect("other.rs");
        crate::git::git_run(repo.path(), &["add", "other.rs"]);
        crate::git::git_run(repo.path(), &["commit", "--quiet", "-m", "Fix the prompt"]);
        set_active_pull_requests(&[PullRequest {
            number: 232,
            title: "Fix the prompt (#45)".to_string(),
        }]);
        start(232, repo.path());
        note_rebase(repo.path());
        assert_eq!(candidates("").last().unwrap(), "close issue 45");

        // ...and with neither naming one, only the request is left to close.
        // A rebase after the branch has been left is not the branch's.
        set_active_pull_requests(&[PullRequest {
            number: 232,
            title: "Fix the prompt".to_string(),
        }]);
        start(232, repo.path());
        crate::git::git_run(repo.path(), &["checkout", "--quiet", "main"]);
        note_checkout("main");
        note_rebase(repo.path());
        assert_eq!(
            candidates(""),
            vec![
                "merge pr-232",
                "push",
                "delete pr-232",
                "comment on 232 merged.md"
            ]
        );
        crate::git::git_run(repo.path(), &["checkout", "--quiet", "pr-232"]);
        note_rebase(repo.path());
        assert_eq!(candidates("").last().unwrap(), "close pr 232");
        note_comment(232);
        assert_eq!(candidates(""), vec!["close pr 232"]);
        note_close(&CloseTarget::PullRequest(232));
        assert!(snapshot().is_none());
    }

    #[test]
    fn a_finished_flow_drops_the_request_and_asks_for_a_refetch() {
        let _guard = exclusive();
        set_active_pull_requests(&[
            PullRequest {
                number: 231,
                title: "A".to_string(),
            },
            PullRequest {
                number: 12,
                title: "B".to_string(),
            },
        ]);
        assert!(!take_refresh_request());
        begin(MergeFlowStep::Comment);
        note_comment(231);
        // The landed request is gone at once — the prompt must not offer to
        // pull it again while the refetch is still on its way...
        assert_eq!(candidates(""), vec!["pull 12"]);
        // ...and the refetch is asked for exactly once.
        assert!(take_refresh_request());
        assert!(!take_refresh_request());

        // Abandoning a flow is not finishing it: nothing is refetched.
        begin(MergeFlowStep::Switch);
        note_checkout("some-other-work");
        assert!(snapshot().is_none());
        assert!(!take_refresh_request());
    }

    #[test]
    fn the_issue_is_read_from_the_subject() {
        for (title, issue) in [
            ("[#45] Fix the prompt", Some(45)),
            ("Fix the prompt (#45)", Some(45)),
            ("Fixes #45: the prompt", Some(45)),
            ("#45", Some(45)),
            ("Resolve issue 45", Some(45)),
            ("Resolve Issue #45.", Some(45)),
            ("Resolve [issue 45]", Some(45)),
            ("Fix #12 and #45", Some(12)),
            ("Fix the prompt", None),
            ("Fix the issue", None),
            ("Fix the issue with #tags", None),
            ("Fix issue #", None),
            ("", None),
        ] {
            assert_eq!(issue_in_subject(title), issue, "{title:?}");
        }
    }

    #[test]
    fn steps_only_move_forward_and_only_for_the_flows_own_branch() {
        let _guard = exclusive();
        // A push before the merge is not this flow's push; a merge of some
        // other branch is not this flow's merge.
        begin(MergeFlowStep::Merge);
        note_push();
        assert_eq!(candidates("")[0], "merge pr-231");
        note_merge("some-other-branch");
        assert_eq!(candidates("")[0], "merge pr-231");
        note_merge("pr-231");
        assert_eq!(candidates("")[0], "push");
        // Switching back to the base does not rewind a flow past it.
        note_checkout("main");
        assert_eq!(candidates("")[0], "push");
    }

    #[test]
    fn leaving_for_an_unrelated_branch_ends_the_flow() {
        let _guard = exclusive();
        begin(MergeFlowStep::Switch);
        note_checkout("pr-231");
        assert!(snapshot().is_some(), "staying on the branch keeps the flow");
        note_checkout("some-other-work");
        assert!(snapshot().is_none());
        assert!(candidates("switch").is_empty());
    }

    #[test]
    fn the_developer_prompt_is_left_alone_but_still_tracked() {
        use crate::mode::{self, PromptMode};

        let _guard = exclusive();
        begin(MergeFlowStep::Merge);
        assert_eq!(candidates("")[0], "merge pr-231");

        // `/developer` takes the flow off the prompt entirely...
        mode::set(PromptMode::Developer);
        assert!(candidates("").is_empty());
        let mut branches = vec!["main".to_string(), "pr-231".to_string()];
        hoist_preferred_arguments(&mut branches, workspace());
        assert_eq!(branches, vec!["main", "pr-231"]);

        // ...but the steps taken there still count, so `/committer` picks up
        // where the branch actually is rather than at the start.
        note_merge("pr-231");
        mode::set(PromptMode::Committer);
        assert_eq!(candidates("")[0], "push");
    }

    #[test]
    fn another_workspace_is_not_hinted_at_this_ones_branches() {
        let _guard = exclusive();
        begin(MergeFlowStep::Merge);
        assert_eq!(candidates("")[0], "merge pr-231");
        assert!(flow_candidates("", Path::new("/tmp/orangu-other")).is_empty());
    }

    #[test]
    fn pull_remembers_the_request_and_the_branch_it_checked_out() {
        let _guard = exclusive();
        let repo = tempfile::tempdir().expect("repo");
        crate::git::init_git_for_test(repo.path());
        std::fs::write(repo.path().join("README.md"), "orangu\n").expect("readme");
        crate::git::git_run(repo.path(), &["add", "README.md"]);
        crate::git::git_run(repo.path(), &["commit", "--quiet", "-m", "initial"]);
        crate::git::git_run(repo.path(), &["branch", "-M", "main"]);
        crate::git::git_run(repo.path(), &["checkout", "--quiet", "-b", "pr-231"]);

        start(231, repo.path());
        assert_eq!(
            flow_candidates("", repo.path()),
            vec![
                "switch to main",
                "build",
                "review",
                "auto review",
                "merge pr-231",
                "push",
                "delete pr-231",
                "comment on 231 merged.md",
            ]
        );

        // Pulling while already on the base branch leaves nothing to merge, so
        // no flow is remembered.
        crate::git::git_run(repo.path(), &["checkout", "--quiet", "main"]);
        start(231, repo.path());
        assert!(snapshot().is_none());
    }

    #[test]
    fn without_a_flow_the_open_pull_requests_are_offered() {
        let _guard = exclusive();
        set_active_pull_requests(&[
            PullRequest {
                number: 231,
                title: "A".to_string(),
            },
            PullRequest {
                number: 12,
                title: "B".to_string(),
            },
        ]);
        assert_eq!(candidates(""), vec!["pull 12", "pull 231"]);
        assert_eq!(candidates("pull 2"), vec!["pull 231"]);
        set_active_pull_requests(&[]);
        assert!(candidates("").is_empty());
    }

    #[test]
    fn argument_candidates_lead_with_the_value_the_step_needs() {
        let _guard = exclusive();
        begin(MergeFlowStep::Merge);
        // The branch being merged wins over every other branch offered...
        let mut branches = vec![
            "main".to_string(),
            "other".to_string(),
            "pr-231".to_string(),
        ];
        hoist_preferred_arguments(&mut branches, workspace());
        assert_eq!(branches, vec!["pr-231", "main", "other"]);

        // ...while the base branch leads for the checkout step.
        begin(MergeFlowStep::Switch);
        let mut branches = vec![
            "other".to_string(),
            "pr-231".to_string(),
            "main".to_string(),
        ];
        hoist_preferred_arguments(&mut branches, workspace());
        assert_eq!(branches, vec!["main", "pr-231", "other"]);

        // The comment step reaches for the request number and the template.
        begin(MergeFlowStep::Comment);
        let mut numbers = vec!["12".to_string(), "231".to_string()];
        hoist_preferred_arguments(&mut numbers, workspace());
        assert_eq!(numbers, vec!["231", "12"]);
        let mut files = vec!["review.md".to_string(), "merged.md".to_string()];
        hoist_preferred_arguments(&mut files, workspace());
        assert_eq!(files, vec!["merged.md", "review.md"]);

        // The closing steps reach for the request, then the issue.
        test_support::begin_rebased(
            231,
            "pr-231",
            MergeFlowStep::ClosePullRequest,
            workspace(),
            true,
            Some(45),
        );
        let mut numbers = vec!["12".to_string(), "231".to_string()];
        hoist_preferred_arguments(&mut numbers, workspace());
        assert_eq!(numbers, vec!["231", "12"]);
        test_support::begin_rebased(
            231,
            "pr-231",
            MergeFlowStep::CloseIssue,
            workspace(),
            true,
            Some(45),
        );
        let mut numbers = vec!["231".to_string(), "45".to_string()];
        hoist_preferred_arguments(&mut numbers, workspace());
        assert_eq!(numbers, vec!["45", "231"]);

        // Nothing is reordered while no flow is in progress.
        store(None);
        let mut branches = vec!["other".to_string(), "main".to_string()];
        hoist_preferred_arguments(&mut branches, workspace());
        assert_eq!(branches, vec!["other", "main"]);
    }
}

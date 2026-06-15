\newpage

# Git tools

The Git tools wrap common Git, GitHub, and GitLab operations as local slash commands. They all require a Git repository in the workspace and, unless noted, run plain Git under the hood. A few commands integrate with the `gh` (GitHub) or `glab` (GitLab) CLI when it is installed — and a small number require it.

Several commands surface a combined hint when the current branch is behind its base or carries more than one commit: `branch is N commits behind <base>; run /rebase` and/or `N commits ahead of <base>; run /squash`. The base branch is detected in the order `origin/main`, `origin/master`, `main`, then `master`.

As with the core tools, every command below also accepts the natural-language aliases listed in its **Examples**.

\newpage

## /status

Shows the working-tree status with color highlighting.

It runs `git status --branch --short`. Added files and untracked entries are shown in green, deleted entries in red, and modified entries in the default terminal color; the branch line is shown in a muted color. `gh` has no equivalent, so this always uses plain Git.

### Examples

```text
/status
```

Natural-language forms:

```text
status
show status
git status
```

\newpage

## /log

Shows the commit log.

If a `lg` alias is found in `~/.gitconfig` it runs `git lg`, otherwise it falls back to `git log --graph --oneline --decorate`. Pass an optional number to limit the output to the latest that many commits. Below the log it appends a highlighted summary of the working tree — either `● Working tree clean` or, when there are pending changes, `● N change(s)` broken down into uncommitted (tracked) and untracked counts. See the optional tools chapter for the recommended `git lg` alias setup.

### Examples

Show the full log:

```text
/log
```

Show only the latest five commits:

```text
/log 5
```

Natural-language forms:

```text
log
show log
git log
git lg
```

\newpage

## /diff

Shows a color unified diff.

Without a branch it shows unstaged changes; with a branch it runs `git diff <branch>...HEAD` to show the commits on the current branch that are not yet in the specified branch. Inside a Git repository it applies any configured non-interactive Git pager such as `delta`. Tab completion after `/diff ` (or the natural-language form `diff against `) offers local and remote branch names.

### Examples

Show unstaged changes:

```text
/diff
```

Show what the current branch adds over `main`:

```text
/diff main
```

Natural-language forms — unstaged changes:

```text
diff
show diff
git diff
```

Natural-language forms — against a branch:

```text
diff against main
show diff against main
git diff main
```

\newpage

## /grep

Searches the workspace for a pattern with `git grep`.

It searches all tracked files and requires a Git repository. Output is piped through the configured non-interactive pager (`pager.grep`, then `core.pager`) when one is set — if `delta` is configured it will colorize and format the results. Exit code 1 (no matches) is handled gracefully. `gh` has no equivalent, so it always uses plain Git.

### Examples

```text
/grep TODO
/grep "fn main"
```

Natural-language forms:

```text
grep TODO
find TODO
git grep TODO
```

\newpage

## /add_file

Stages a file or directory with `git add`.

Tab completion after `/add_file ` offers untracked directories first, then untracked files; already-tracked content is excluded.

### Examples

```text
/add_file README.md
/add_file src/
```

Natural-language forms:

```text
add README.md
add file src/
git add README.md
```

\newpage

## /remove_file

Removes a file or directory from Git tracking with `git rm` (using `-r` for directories).

Tab completion after `/remove_file ` offers tracked directories first, then tracked files; untracked content is excluded.

### Examples

```text
/remove_file old.rs
/remove_file legacy/
```

Natural-language forms:

```text
remove old.rs
remove file legacy/
git rm old.rs
```

\newpage

## /move_file

Renames or moves a tracked file with `git mv`.

```text
/move_file <source> <destination>
```

Tab completion for the first argument offers tracked directories first, then tracked files; the second argument completes from all workspace paths.

### Examples

```text
/move_file old.rs new.rs
```

Natural-language forms:

```text
move old.rs new.rs
move file old.rs new.rs
git mv old.rs new.rs
```

\newpage

## /restore

Discards working-tree changes to a file with `git restore <file>`, or unstages it with `--staged`.

```text
/restore [--staged] <file>
```

With `--staged` it runs `git restore --staged <file>` to unstage the file without changing its contents. `gh` has no equivalent, so it always uses plain Git.

### Examples

Discard local changes to a file:

```text
/restore src/main.rs
```

Unstage a file:

```text
/restore --staged src/main.rs
```

Natural-language form:

```text
restore src/main.rs
```

\newpage

## /commit

Commits all tracked changes with `git commit -a -m <message>`.

`gh` has no equivalent, so it always uses plain Git. The message may be bare or quoted; quote it when it contains spaces or shell-significant characters such as the `[#42]` issue prefix.

### Examples

```text
/commit Fix the parser
/commit "[#42] Add the new feature"
```

Natural-language forms:

```text
commit Fix the bug
commit "[#42] My feature"
git commit -m "Fix the bug"
```

\newpage

## /amend

Rewrites the last commit message with `git commit --amend -m <message>`.

The message is mandatory and may be bare or quoted. `gh` has no equivalent, so it always uses plain Git.

### Examples

```text
/amend Fix the parser
/amend "[#42] Add the new feature"
```

Natural-language forms:

```text
amend Fix the bug
amend "[#42] My feature"
amend message "[#42] My feature"
git amend "Fix the bug"
git commit --amend -m "Fix the bug"
```

\newpage

## /squash

Squashes all commits on the current branch into a single commit, reusing the oldest commit's message.

The branch is compared against `origin/main`, `origin/master`, `main`, or `master`, tried in that order. At least two commits are required, and squashing on `main` or `master` is blocked. `gh` has no equivalent, so it always uses plain Git.

### Examples

```text
/squash
```

Natural-language forms:

```text
squash
squash branch
squash commits
git squash
```

\newpage

## /stash

Saves and restores uncommitted changes on the stash stack.

- `/stash` runs `git stash push` to save all uncommitted changes, both staged and unstaged.
- `/stash pop` restores and removes the most recent stash entry with `git stash pop`.
- `/stash list` shows all stash entries with their index and description.
- `/stash drop` discards the most recent stash entry with `git stash drop`.

`gh` has no stash equivalent, so all four operations always use plain Git. Running `/stash` with a clean working tree produces an error from Git.

### Examples

```text
/stash
/stash pop
/stash list
/stash drop
```

Natural-language forms:

```text
stash
pop stash
list stashes
drop stash
```

\newpage

## /bisect

Runs a binary-search session to find the commit that introduced a bug.

- `/bisect` or `/bisect status` — shows the current bisect state; reports "No bisect session in progress" when none is active.
- `/bisect start [<commit>]` — begins a new bisect session, optionally marking `<commit>` as the first known-bad revision.
- `/bisect good [<commit>]` — marks the current (or specified) commit as good.
- `/bisect bad [<commit>]` — marks the current (or specified) commit as bad.
- `/bisect skip [<commit>]` — tells Git to skip the current (or specified) commit when it cannot be tested.
- `/bisect reset` — ends the bisect session and returns `HEAD` to its original position.
- `/bisect log` — prints the log of all good/bad markings made so far.

All subcommands run plain `git bisect <sub>`; there is no `gh` equivalent.

### Examples

```text
/bisect start
/bisect bad
/bisect good abc1234
/bisect skip
/bisect log
/bisect reset
```

Natural-language forms:

```text
bisect start
start bisect
bisect good
mark good
bisect bad
mark bad
bisect skip
skip commit
bisect reset
reset bisect
bisect log
bisect
```

\newpage

## /branch

Lists, switches, creates, renames, or deletes branches.

```text
/branch [<name> | -b <name> | -m <name> | -d <name> | -a]
```

- No arguments — lists local branches (`git branch`).
- `-a` — lists all branches including remote.
- `<name>` — switches to an existing branch.
- `-b <name>` — creates and switches to a new branch (`git checkout -b`).
- `-m <new-name>` — renames the current branch (`git branch -m`).
- `-d <name>` — deletes the branch (`git branch -D`), blocked on `main` and `master`.

Tab completion after `/branch ` offers local branch names. `gh` has no equivalent, so all operations always use plain Git.

### Examples

```text
/branch
/branch -a
/branch feature/login
/branch -b feature/login
/branch -m feature/auth
/branch -d feature/old
```

Natural-language forms:

```text
branch
list branches
list all branches
checkout main
switch to main
create branch feature/x
rename to new-name
delete branch feature/old
```

\newpage

## /merge

Merges a branch into the current branch.

If `gh` is installed it uses `gh pr merge --merge`; otherwise it uses `git merge`.

### Examples

```text
/merge feature/login
```

Natural-language forms:

```text
merge feature/login
git merge feature/login
```

\newpage

## /rebase

Rebases the current branch against the repository default branch.

If `gh` is installed it queries the repository default branch; otherwise it probes `origin/main` then `origin/master`.

### Examples

```text
/rebase
```

Natural-language forms:

```text
rebase
git rebase
```

\newpage

## /cherry_pick

Cherry-picks a commit onto the current branch with `git cherry-pick`.

`gh` has no equivalent, so it always uses plain Git. Tab completion offers abbreviated commit hashes from the default branch (`origin/main`, `origin/master`, `main`, or `master`, tried in that order).

### Examples

```text
/cherry_pick abc1234
```

Natural-language forms:

```text
cherry pick abc1234
cherry-pick abc1234
git cherry-pick abc1234
```

\newpage

## /init_repo

Initializes a Git repository in the workspace with `git init`.

It works both inside and outside an existing Git repository — reinitializing an existing repo is safe. `gh` has no equivalent, so it always uses plain Git.

### Examples

```text
/init_repo
```

Natural-language forms:

```text
init
init repo
git init
```

\newpage

## /push

Pushes the current branch to `origin` with `git push origin <branch>`.

`gh` has no equivalent, so it always uses plain Git. `--force` (or `-f`, or `force`) runs `git push -f origin <branch>` but is blocked on `main` and `master` to prevent accidental history rewrites. After a successful push it compares the branch against the base branch and, when the branch is behind the base or has more than one commit ahead, reports that the branch needs a `/rebase` and/or `/squash`.

### Examples

```text
/push
/push --force
```

Natural-language forms:

```text
push
git push origin
force push
push --force
```

\newpage

## /pull

Checks out a GitHub pull request on a dedicated branch.

```text
/pull <number>
```

If `gh` is installed it uses `gh pr checkout`; otherwise it fetches the pull request directly from `origin`. After the checkout it compares the branch against the base branch and, when the branch is behind the base or has more than one commit ahead, reports that the pull request needs a `/rebase` and/or `/squash`.

### Examples

```text
/pull 58
```

Natural-language forms:

```text
pull 58
pull pr 58
pull request 58
pull #58
```

\newpage

## /pull_request

Creates a pull request for the current branch. Requires the `gh` CLI.

Before creating the pull request it runs several pre-flight checks: it blocks on `main` and `master`, requires at least one commit ahead of the base branch, and blocks when the branch is behind the base and/or has more than one commit ahead — reporting the combined `/rebase` and/or `/squash` hint. When all checks pass it pushes the branch with `--set-upstream origin` and calls `gh pr create` with the title and body derived from the single commit message.

The checks can be bypassed by setting `auto_rebase = on` or `auto_squash = on` in the `[orangu]` configuration section, which triggers the corresponding fix automatically before continuing.

### Examples

```text
/pull_request
```

Natural-language forms:

```text
pull request
create pull request
open pull request
new pull request
create pr
open pr
new pr
```

\newpage

## /comment

Adds a comment to a GitHub issue or GitLab issue. Requires the `gh` or `glab` CLI.

```text
/comment <number> "<comment>"
/comment <number> <file>
/comment <number> with review
/comment <number> with auto review
```

It runs `gh issue comment <number> --body <body>` (or the GitLab equivalent). Without the CLI installed it reports an error, since there is no plain Git equivalent. When the third argument is a quoted string it is used as the comment body directly. When it is a bare word it is treated as a filename relative to `~/.orangu/comments/` and the file contents become the body — Tab completion after `/comment <number> ` (without a leading `"`) lists files in that directory.

### Submitting a review as the comment

`with review` posts the last `/review` summary of this session as the comment body, and `with auto review` posts the last `/auto_review` report — the same Markdown that is copied to the clipboard on exit, ready for an issue or pull request. The keywords are matched case-insensitively against the whole argument, so a template file whose name merely starts with `w` (or even `with`) is still treated as a filename; only the exact phrases are keywords. When no matching review has been run yet, the command reports an error pointing at `/review` or `/auto_review`.

Tab completion (and the inline grey ghost) after `/comment <number> ` offers the template files from `~/.orangu/comments/` first — an existing template keeps its priority — followed by the report keywords. Each keyword is only offered once the matching review has actually been run in the session: before any `/review`, `with review` is ignored by completion (and likewise `with auto review` before any `/auto_review`), so the hints never suggest a report that does not exist. Typing `with ` narrows the hint to the available keywords.

### Examples

Inline comment body:

```text
/comment 51 "Thanks, merged."
```

Comment body from a Markdown file in `~/.orangu/comments/`:

```text
/comment 51 merged.md
```

The last review or auto review report as the body:

```text
/comment 48 with review
/comment 48 with auto review
```

Natural-language forms:

```text
add comment on 51 "My comment"
add comment to 51 "My comment"
comment on 51 "My comment"
comment on 48 with review
comment on 48 with auto review
```

\newpage

## /close

Closes a GitHub/GitLab issue or pull request. Requires the `gh` or `glab` CLI.

```text
/close -i <number>
/close -p <number>
```

- `-i <number>` runs `gh issue close <number>` (or `glab issue close <number>`) to close an issue.
- `-p <number>` runs `gh pr close <number>` (or `glab mr close <number>`) to close a pull request or merge request.

### Examples

```text
/close -i 51
/close -p 58
```

Natural-language forms:

```text
close issue 51
close pr 58
```

\newpage

## /get_comments

Lists the comments on a GitHub/GitLab issue or pull request. Requires the `gh` or `glab` CLI.

```text
/get_comments -i <number>
/get_comments -p <number>
```

- `-i <number>` runs `gh api repos/{owner}/{repo}/issues/<number>/comments` (or `glab api projects/:id/issues/<number>/notes`) to list the comments on an issue.
- `-p <number>` lists the comments on a pull request or merge request. On GitHub a pull request keeps its conversation comments and its inline review comments on separate endpoints, so both `gh api repos/{owner}/{repo}/issues/<number>/comments` and `gh api repos/{owner}/{repo}/pulls/<number>/comments` are fetched and merged chronologically. On GitLab it runs `glab api projects/:id/merge_requests/<number>/notes`, which already mixes discussion and inline diff notes.

Each comment is shown as a block: a grey `● <date> <author>` header line, then the body indented two spaces. On GitLab, system notes (label changes, assignments, and so on) are skipped.

```text
● 2026-06-01 12:30:45 alice
  Looks good!

● 2026-06-02 08:00:00 bob
  Merged.
```

### Examples

```text
/get_comments -i 51
/get_comments -p 58
```

Natural-language forms (after `get comments for ` the inline ghost hint offers `issue` and `pull request`; Tab accepts, Shift+Tab cycles):

```text
get comments for issue 51
get comments for pull request 58
```

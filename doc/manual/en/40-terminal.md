\newpage

# Terminal interface

`orangu` is an interactive terminal client with a persistent header and a prompt area anchored to the bottom of the terminal.

## Header

The top banner displays:

- Current version
- Workspace status
- Server status
- Model status
- `/help` reminder

While no request is active, server and model status are rechecked once per minute.

## Prompt area

The prompt area stays at the bottom of the terminal window.

- Long input wraps upward
- Submitted input moves directly into the output area
- The banner and prompt stay fixed while the output window scrolls independently
- Markdown in assistant output is rendered with terminal styling when possible, including emphasis, strong text, lists, headings, links, and code
- Fenced code blocks with a language tag such as ```c use syntax highlighting in the terminal when the language is supported by the bundled highlighter

## Waiting state

While the model is generating a response, the left side of the footer shows a rolling:

```text
Thinking (2s)
```

status indicator.

You can keep typing and submitting commands while a response is pending. Submitted commands are queued and executed in order after the active response finishes.

When a profile uses `provider = llama.cpp`, the footer starts with `Thinking (<CLOCK>)` and switches to llama.cpp's native generation throughput once tokens are streaming, for example `Working @ 42.5 t/s (2s)`.

Press `Esc` twice within 2 seconds during the waiting state to cancel the active request without exiting the client. Queued commands are preserved.

## Sessions

Each run of `orangu` creates or resumes a session identified by a UUID.

### Automatic resume

On startup, `orangu` checks whether a session already exists for the current workspace path and Git branch. If exactly one matching session with conversation history is found, it is resumed automatically. The status bar lower-left shows:

```text
Resuming session 550e8400-e29b-41d4-a716-446655440000
```

for five seconds or until the first command is run. No `--resume` flag is needed for normal branch-based workflows.

If more than one session matches the current workspace and branch, a fresh session is started instead. Use `--resume <uuid>` to target a specific session explicitly.

### Manual resume

To resume a specific session regardless of workspace or branch, pass `--resume <uuid>` when starting:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

This restores the previous conversation context and per-session readline history.

### Session cleanup on exit

When you exit, the resume command is printed:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

Sessions that had no LLM interaction (zero tokens generated) and are on `main`, `master`, or a workspace with no Git repository are deleted automatically on exit. No resume command is printed for deleted sessions. Sessions on feature branches are always kept even when empty, so that returning to the branch triggers auto-resume correctly.

### Session storage

Session data is stored under `~/.orangu/sessions/<uuid>/`:

```text
history    per-session command history (readline)
messages   full conversation turn history (JSON array of role/content objects)
metadata   session metadata (JSON)
```

The `messages` file preserves the complete conversation so that resuming restores the exact context the model had when the session was last active.

The `metadata` file records when the session was created, last used, which workspace it belongs to, and which Git branch was active:

```json
{
  "started_at": 1748000000,
  "last_updated_at": 1748003600,
  "workspace": "/home/user/myproject",
  "branch": "feature/my-pr"
}
```

`branch` is an empty string for sessions started outside a Git repository or in a detached HEAD state. Timestamps are Unix seconds (UTC).

### Listing and switching sessions

Use `/sessions` to list all sessions:

```text
UUID                                  STARTED       LAST          CMDS  BRANCH                WORKSPACE
550e8400-e29b-41d4-a716-446655440000  202605220910  202605221143    42  feature/my-pr         /home/user/myproject
a1b2c3d4-e5f6-7890-abcd-ef1234567890  202605210830  202605210831     3  -                     /home/user/other
```

Pass an optional workspace filter to narrow results:

```text
/sessions myproject
```

Use `/session <uuid>` to print the resume command for a specific session:

```text
/session 550e8400-e29b-41d4-a716-446655440000
```

This outputs:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

Tab completion after `/session ` (with a trailing space) cycles through session UUIDs, newest first.

## History and navigation

Command history is stored per session in:

```text
~/.orangu/sessions/<uuid>/history
```

Use:

- `<ARROW_UP>` to move backward in history
- `<ARROW_DOWN>` to move forward in history

## Local commands

All slash commands are handled locally. They are not sent to the model.

| Command | Description |
| :-- | :-- |
| `/help` | Show available commands |
| `/connect [url]` | Connect to the configured server, or a specific server |
| `/disconnect` | Disconnect from the current server |
| `/reload` | Restore the configured model and server |
| `/tools` | List tools |
| `/model [name]` | Switch to the configured model, or a specific model |
| `/models` | List models |
| `/session [uuid]` | Print the resume command for a specific session; Tab completion cycles UUIDs newest-first |
| `/sessions [workspace]` | List all sessions, optionally filtered by workspace path |
| `/list_files` | List workspace files as a tree |
| `/open_file <path>` | Open a workspace file in $EDITOR |
| `/show_file [--hash] [--author] <path> [<ref>]` | Show a file; optional ref shows that commit via git show |
| `/build` | Build the workspace project (Rust, C, or Java) |
| `/add_file <path>` | Stage a file or directory with git add |
| `/amend <message>` | Rewrite the last commit message |
| `/checkout <branch\|file>` | Switch branch or restore a file |
| `/cherry_pick <commit>` | Cherry-pick a commit onto the current branch |
| `/comment <number> "<comment>"` | Add a comment to a GitHub issue with gh issue comment |
| `/commit <message>` | Commit all tracked changes with git commit -a -m |
| `/delete <branch>` | Delete a local branch |
| `/diff [branch]` | Show a color unified diff; without a branch shows unstaged changes, with a branch shows changes since diverging from it |
| `/init_repo` | Initialize a Git repository in the workspace |
| `/log` | Show commit log (uses `git lg` alias if configured) |
| `/merge <branch>` | Merge a branch into the current branch |
| `/move_file <source> <destination>` | Rename or move a tracked file with git mv |
| `/pull <number>` | Check out a GitHub pull request on a dedicated branch |
| `/pull_request` | Create a pull request for the current branch |
| `/push [--force]` | Push the current branch to origin |
| `/rebase` | Rebase the current branch against master/main |
| `/remove_file <path>` | Remove a file or directory from Git tracking |
| `/squash` | Squash all branch commits into one using the first commit message |
| `/status` | Show working tree status with color highlighting |
| `/usage` | Show usage statistics for this session |
| `/clear` | Clear the current conversation |
| `/quit` | Exit the client |

Local commands continue to work even when the model is unavailable.

Free-form prompts are blocked when the server or model status in the header is red.

## Command notes

- `/tools` lists the model-facing workspace tools described in the tools chapter
- `/open_file <path>` is workspace-scoped; paths outside the workspace are rejected
- `/show_file [--hash] [--author] <path> [<ref>]` is workspace-scoped; without a ref, the current workspace file is shown — when `bat` is installed it is used for the plain view, otherwise the built-in syntax-highlighted renderer is used; when a ref (commit hash, branch, or tag) is given, the file content at that ref is retrieved via `git show <ref>:<path>` and rendered with the built-in renderer; `--hash` and `--author` add per-line blame columns sourced from `git blame`, using the same ref when one is provided; Tab completion for the first positional argument offers workspace file paths recursively; Tab completion for the second positional argument cycles through that file's commit history (abbreviated hashes from `git log --follow`)
- `/build` detects the project type from the workspace root and runs the appropriate toolchain: for Rust (`Cargo.toml`) it runs `cargo fmt`, `cargo clippy`, `cargo build`, and `cargo test`; for C (`CMakeLists.txt`) it runs `clang-format.sh` (if present), creates a `build/` directory if needed, runs `cmake ..` on the first build, then `make`; for Java (`pom.xml`) it installs frontend dependencies with `npm ci` when outdated, runs `npm run fix` and `npm run check` for the frontend (if `src/frontend/` exists), then `mvn package`; each step is reported individually and the pipeline stops on first failure
- `/diff` uses `git diff` inside Git repositories and applies configured non-interactive Git pagers such as `delta`; outside Git repositories it keeps the existing non-Git behavior; `/diff <branch>` runs `git diff <branch>...HEAD` to show commits on the current branch not yet in the specified branch; Tab completion after `/diff ` or natural-language forms such as `diff against <branch>` offers local and remote branch names
- `/status` requires a Git repository and runs `git status --branch --short`; `gh` has no equivalent so it always uses plain Git; added files and untracked entries are shown in green, deleted entries in red, and modified entries in the default terminal color; the branch line is shown in a muted color
- `/log` requires a Git repository; if a `lg` alias is found in `~/.gitconfig` it runs `git lg`, otherwise it falls back to `git log --graph --oneline --decorate`; see the optional tools chapter for the recommended `git lg` alias setup
- `/pull <number>` requires a Git repository; if `gh` is installed it uses `gh pr checkout`, otherwise it fetches the pull request directly from `origin`
- `/pull_request` requires a Git repository and the `gh` CLI; it runs several pre-flight checks before creating the pull request: it blocks on `main` and `master`, requires at least one commit ahead of the base branch, blocks if the branch is behind the base (suggesting `/rebase`), and blocks if there is more than one commit ahead (suggesting `/squash`); when all checks pass it pushes the branch with `--set-upstream origin` and calls `gh pr create` with the title and body derived from the single commit message; the checks can be bypassed by setting `auto_rebase = on` or `auto_squash = on` in the `[orangu]` config section, which triggers the corresponding fix automatically before continuing
- `/rebase` requires a Git repository; if `gh` is installed it queries the repository default branch, otherwise it probes `origin/main` then `origin/master`
- `/merge <branch>` requires a Git repository; if `gh` is installed it uses `gh pr merge --merge`, otherwise it uses `git merge`
- `/checkout <branch|file>` requires a Git repository and runs `git checkout`; Tab completion offers branch names first, then workspace file paths
- `/add_file <path>` requires a Git repository and runs `git add`; Tab completion offers untracked directories first, then untracked files
- `/remove_file <path>` requires a Git repository and runs `git rm` (with `-r` for directories); Tab completion offers tracked directories first, then tracked files
- `/move_file <source> <destination>` requires a Git repository and runs `git mv`; Tab completion for the first argument offers tracked directories first, then tracked files; Tab completion for the second argument offers workspace paths
- `/cherry_pick <commit>` requires a Git repository and runs `git cherry-pick`; `gh` has no equivalent so it always uses plain Git; Tab completion offers abbreviated commit hashes from the default branch (`origin/main`, `origin/master`, `main`, or `master`)
- `/comment <number> "<comment>"` requires a Git repository and the `gh` CLI, and runs `gh issue comment <number> --body <comment>` to add a comment to a GitHub issue; without `gh` installed it reports an error since there is no plain Git equivalent; the comment text may be bare (`/comment 51 My comment`) or quoted (`/comment 51 "My comment"`); the natural-language form `add comment on 51 "My comment"` is also handled
- `/commit <message>` requires a Git repository and runs `git commit -a -m <message>`; `gh` has no equivalent so it always uses plain Git; the message may be bare (`/commit Fix the bug`) or quoted (`/commit "[#42] My feature"`)
- `/amend <message>` requires a Git repository and runs `git commit --amend -m <message>`; `gh` has no equivalent so it always uses plain Git; the message is mandatory and may be bare (`/amend Fix the bug`) or quoted (`/amend "[#42] My feature"`)
- `/push [--force]` requires a Git repository and runs `git push origin <branch>` using the current branch name; `gh` has no equivalent so it always uses plain Git; `--force` (or `-f` or `force`) runs `git push -f origin <branch>` but is blocked on `main` and `master` to prevent accidental history rewrites
- `/init_repo` runs `git init` in the workspace directory; works both inside and outside an existing Git repository (reinitializing an existing repo is safe); `gh` has no equivalent so it always uses plain Git
- `/squash` requires a Git repository; squashes all commits on the current branch (relative to `origin/main`, `origin/master`, `main`, or `master`, tried in that order) into a single commit using the oldest commit's message; `gh` has no equivalent so it always uses plain Git; squashing on `main` or `master` is blocked; requires at least two commits on the branch
- `/delete <branch>` requires a Git repository and runs `git branch -D`; `gh` has no equivalent so it always uses plain Git; deleting `main` or `master` is blocked; Tab completion offers local branch names excluding `main` and `master`
- `/sessions [workspace]` lists all sessions found under `~/.orangu/sessions/`; output is one line per session with aligned columns: UUID, start date, last-updated date, command count, branch, and workspace path; sessions are sorted by creation time, most-recent first; an optional workspace argument filters the list to sessions whose workspace path contains the given string; the branch column shows `-` for sessions with no recorded branch
- `/session [uuid]` prints the `orangu --resume <uuid>` command for the given session; Tab completion after `/session ` (with a trailing space) cycles through all session UUIDs, newest first; with no argument it lists all sessions (same as `/sessions`)
- `/usage` shows session statistics: total application time, total time spent waiting for LLM responses, total tokens generated (counted with the bundled tokenizer), and average tokens per second
- `/list_files` is a local convenience command and is separate from the model-facing `list_directory` tool
- `/reload` also clears the current conversation history in memory
- `/quit` exits immediately, while `Ctrl+C` uses a two-step confirmation; on exit the full resume command is printed unless the session had no LLM interaction and was on `main`, `master`, or outside a Git repository — in that case the session directory is deleted silently
- Unknown slash commands are handled locally and produce an error message that points back to `/help`

## Natural-language command aliases

Local commands can also be entered in plain language. Examples:

- `open README.md`
- `show README.md`
- `list models`
- `list files`
- `show tools`
- `show help`
- `switch model to <name>`
- `pull 58` or `pull request 58` or `pull #58`
- `add comment on 51 "My comment"` or `comment on 51 "My comment"`
- `log` or `show log` or `git log` or `git lg`
- `status` or `show status` or `git status`
- `rebase` or `git rebase`
- `merge feature/foo` or `git merge feature/foo`
- `checkout main` or `checkout README.md` or `git checkout main`
- `switch to main` or `switch to feature/foo` or `switch to main branch`
- `add README.md` or `add file src/` or `git add README.md`
- `remove README.md` or `remove file src/` or `git rm README.md`
- `move old.rs new.rs` or `move file old.rs new.rs` or `git mv old.rs new.rs`
- `cherry pick abc1234` or `cherry-pick abc1234` or `git cherry-pick abc1234`
- `commit "[#42] My feature"` or `commit Fix the bug` or `git commit -m "Fix the bug"`
- `amend "[#42] My feature"` or `amend Fix the bug` or `git amend "[#42] My feature"` or `git commit --amend -m "Fix the bug"`
- `pull request` or `create pull request` or `open pull request` or `create pr` or `open pr`
- `push` or `git push` or `git push origin`
- `force push` or `push force` or `push --force`
- `init` or `init repo` or `git init`
- `delete feature/foo` or `delete branch feature/foo` or `git branch -D feature/foo`
- `usage` or `show usage`
- `session` or `switch session`

Natural-language forms are recognized only for the built-in local command phrases. Ordinary prompts continue to go to the model.

## Comments and ignored input

- If the first non-whitespace character is `#`, the line is treated as a local comment, shown in the transcript, and not sent to the LLM
- If the first non-whitespace character is `\`, the line is ignored

## Shortcuts and keys

### Prompt editing

- `Ctrl+A` or `Home` moves the cursor to the start of the input line
- `Ctrl+E` or `End` moves the cursor to the end of the input line
- `Left` moves the cursor one character left
- `Right` moves the cursor one character right
- `Ctrl+Left` moves one word left using bash-style word boundaries
- `Ctrl+Right` moves one word right using bash-style word boundaries
- `Backspace` deletes the character to the left of the cursor
- `Delete` deletes the character under the cursor
- `Ctrl+D` behaves like `Delete`; when the input is empty it exits the client immediately
- `Ctrl+K` deletes from the cursor to the end of the line
- `Ctrl+U` deletes from the start of the line to the cursor
- `Ctrl+W` deletes from the cursor to the previous whitespace
- `Alt+Backspace` deletes backward using bash-style word boundaries
- `Alt+D` deletes forward using bash-style word boundaries
- Pasted text is inserted at the current cursor position

### History and completion

- `<ARROW_UP>` moves backward through command history
- `<ARROW_DOWN>` moves forward through command history
- History navigation preserves the current unfinished line as a draft and restores it when you move back out of history

### Tab completion

`Tab` uses context-sensitive completion. The first `Tab` inserts the first match. Repeated `Tab` presses cycle through the remaining matches for the same completion range.

Completion cycling is reset as soon as you edit the line, move the cursor, paste text, or otherwise change the input.

The completion modes are checked in order:

1. If the line starts with `/checkout `, or with the natural-language prefixes `checkout ` or `git checkout `, complete branch names first (from `git branch --all`), then workspace file paths. Branch names always appear before file names in the candidate list. If the line starts with the natural-language prefix `switch to `, complete branch names and tag names (from `git tag`), sorted together; workspace file paths are excluded.
2. If the line starts with `/add_file `, or with the natural-language prefixes `add `, `add file `, or `git add `, complete untracked directories first (from `git ls-files --others --directory`), then untracked files. Already-tracked content is excluded.
3. If the line starts with `/remove_file `, or with the natural-language prefixes `remove `, `remove file `, or `git rm `, complete tracked directories first (from `git ls-files`), then tracked files. Untracked content is excluded.
4. If the line starts with `/move_file `, or with the natural-language prefixes `move `, `move file `, or `git mv `, complete the first argument from tracked directories and files; complete the second argument from all workspace paths.
5. If the line starts with `/cherry_pick `, or with the natural-language prefixes `cherry pick `, `cherry-pick `, or `git cherry-pick `, complete abbreviated commit hashes from the default branch (`origin/main`, `origin/master`, `main`, or `master`, tried in that order).
6. If the line starts with `/merge `, or with the natural-language prefixes `merge ` or `git merge `, complete local branch names first (from `git branch`), then remote-only branch names (from `git branch --all`).
7. If the line starts with `/delete `, or with the natural-language prefixes `delete `, `delete branch `, or `git branch -D `, complete local branch names (from `git branch`) excluding `main` and `master`.
8. If the line starts with `/session ` (with a trailing space), complete session UUIDs sorted newest-first by last-modified time.
9. If the line starts with `/model `, complete configured model profile names.
10. If the line starts with `/open_file ` or `/show_file `, complete workspace file paths recursively for the first positional argument. `/show_file` also completes `--hash` and `--author`. When a file path is already present, the next Tab press cycles through that file's commit history (abbreviated hashes from `git log --follow`).
11. If the line starts with the natural-language prefixes `open `, `open file `, `edit `, or `edit file `, complete workspace file paths recursively.
12. If the line starts with `/`, complete built-in slash commands such as `/help`, `/models`, `/list_files`, `/show_file`, `/tools`, and `/quit`.
13. Otherwise, complete filesystem entries from the current token relative to the workspace, using the token before the cursor.

Path-completion details:

- General filesystem completion lists entries from the matching directory level and appends `/` to directories
- `/open_file`, `/show_file`, and the natural-language open/edit forms search recursively through the workspace
- Recursive file completion matches either the full relative path or, when no `/` is present in the token, the file name
- Quoted file completion is supported for `/open_file "..."`, `/show_file "..."`, and `open "..."`; the inserted completion keeps the opening quote
- Completion skips `.git`, `build`, and `target` content
- Completion also skips paths ignored by the workspace `.gitignore`

### Output scrolling

- `Shift+PageUp` scrolls backward through the output window by a full page
- `Shift+PageDown` scrolls forward through the output window by a full page
- `Alt+Up` scrolls backward one line at a time
- `Alt+Down` scrolls forward one line at a time
- The output scrollback buffer keeps the most recent 10,000 lines
- Scrolling is limited to the output window; it does not replace the header or prompt area

### Horizontal panning

`orangu` maintains a virtual canvas that can be wider than the visible terminal. Source files shown with `/show_file` may contain lines longer than the terminal width; those lines are laid out on the full virtual canvas and can be panned horizontally without reflowing.

- `Alt+Right` pans the output window right (reveals content that extends past the right edge)
- `Alt+Left` pans the output window left (back toward the start of the line)

The header, status bar, and input window always occupy the full visible terminal width and are not affected by panning.

The virtual canvas width is set by the `width` key in the `[orangu]` config section (default `512`). When the terminal is resized to a width larger than the configured virtual width, the virtual width grows to match so that content is never clipped unexpectedly. The virtual width never shrinks below its initial value during a session.

LLM and tool output is always wrapped or clipped to the visible terminal width and does not pan. Only `/show_file` output — where source lines must stay intact — uses the full virtual canvas.

### Waiting and exit control

- `Esc` twice within 2 seconds cancels the active request without exiting and keeps queued commands
- `Ctrl+C` once arms quit mode, shows a warning in the transcript, and clears the current input line
- `Ctrl+C` again within 2 seconds exits the client
- `Enter` submits the current input line

## Footer behavior

- The left side of the footer shows `Thinking (<CLOCK>)` while waiting for a response to start, and `Working @ X.Y t/s (<CLOCK>)` while tokens are streaming
- The center side of the footer shows `Pending: X` to show how many queued commands are waiting
- The right side of the footer shows the model name used

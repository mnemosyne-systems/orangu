\newpage

# Core tools

The core tools are the local slash commands that drive the client itself — discovering commands, switching server and model, managing sessions, browsing the workspace, building the project, and reviewing changes. They are handled locally and are never sent to the model, so they keep working even when the server or model status in the header is red.

Every command below also accepts the natural-language aliases listed in its **Examples**; the recognized phrases are matched only for these built-in commands, and any other input is sent to the model as an ordinary prompt.

\newpage

## /help

Shows the list of available commands with a one-line description of each, grouped the same way as in this manual.

It is the fastest way to rediscover a command name or its arguments without leaving the prompt, and it works regardless of server or model state.

### Examples

```text
/help
```

Natural-language forms:

```text
help
show help
show commands
show available commands
```

\newpage

## /skills

Lists the discovered Agent Skills.

Each line shows the slash command name, the skill description, and whether the
skill came from the project or the user scope. Skills are discovered from
`~/.orangu/skills/`, `~/.agents/skills/`, `<workspace>/.orangu/skills/`, and
`<workspace>/.agents/skills/`, with project skills overriding user skills that
share the same name.

Invoke a listed skill directly with `/skill-name`. When a skill is invoked,
orangu injects the skill instructions into the next model request in a
structured wrapper that also identifies the skill directory and any bundled
resource files.

### Examples

```text
/skills
```

Invoke the bundled debugging skill:

```text
/debugging reproduce the failing request path and identify the root cause
```

\newpage

## /tools

Lists the model-facing workspace tools — `read_file`, `edit_file`, `list_directory`, `fetch_url`, and `run_shell_command` — that the active model may call. These are the tools described in the Tools chapter and are distinct from the local slash commands documented here.

`/tools` is purely informational: it shows what the model is able to do in the current workspace, not what you can type at the prompt. In particular it is separate from the local `/list_files` convenience command.

### Examples

```text
/tools
```

Natural-language forms:

```text
tools
show tools
list tools
show local tools
```

\newpage

## /model

Selects the model used for requests on the active server.

With no argument it lists the selected server's models, with the active model shown in green and the others in red. With a name it switches to that model. Pressing `Tab` after `/model ` cycles through the models the server reports.

### Examples

List the available models:

```text
/model
```

Switch to a specific model:

```text
/model llama3.1:8b
```

Natural-language forms — list the models:

```text
models
list models
show models
show available models
```

Natural-language forms — show the current model:

```text
model
show model
current model
what model am i using
```

Natural-language forms — switch model:

```text
use model llama3.1:8b
switch model to llama3.1:8b
set model to llama3.1:8b
select model llama3.1:8b
```

\newpage

## /server

Selects the server `orangu` talks to.

With no argument it lists the configured servers — each `[section]` in the configuration file identified as a server — with the active one in green and the others in red. With a name it switches to that server and re-detects an available model on it, so the model status in the header is refreshed automatically. Pressing `Tab` after `/server ` cycles through the configured server names.

### Examples

List the configured servers:

```text
/server
```

Switch to a named server:

```text
/server local-llama
```

Natural-language forms — switch server:

```text
use server local-llama
switch server to local-llama
set server to local-llama
select server local-llama
```

\newpage

## /disconnect

Disconnects from the current server.

After disconnecting, the server status in the header turns red and free-form prompts are blocked, but every local command in this manual continues to work. Use `/server` or `/reload` to reconnect.

### Examples

```text
/disconnect
```

Natural-language form:

```text
disconnect
```

\newpage

## /reload

Restores the configured model and server from the configuration file, undoing any `/model` or `/server` switches made during the session.

`/reload` also clears the in-memory conversation history, so it doubles as a way to start a clean exchange while returning to the configured defaults.

### Examples

```text
/reload
```

Natural-language forms:

```text
reload
reload configuration
reset session
```

\newpage

## /restart

Restarts `orangu` in place, resuming the same workspace and session.

The current session is saved first, then the running process is replaced via `exec` with the same binary, passing `--workspace`, `--config`, and `--resume <session-id>` so the new process picks up exactly where the old one left off. This is the recommended way to pick up a freshly built binary or an edited configuration file without losing conversation context.

When the binary was rebuilt while running, the original on-disk path may be reported as deleted; `/restart` resolves the real path so it relaunches the freshly built binary. Only if that path is gone entirely does it fall back to staging a runnable copy under `~/.orangu/last`, a scratch directory that is cleared on every startup.

### Examples

```text
/restart
```

Natural-language forms:

```text
restart
restart orangu
```

\newpage

## /prune

Deletes session directories from `~/.orangu/sessions/`. The active session is never deleted regardless of strategy.

With a UUID it deletes that single session. With `all` it deletes every session except the active one. With `--workspace <path>` (or `-w <path>`) it deletes all sessions whose recorded workspace path contains `<path>`. With `--older-than <days>` (or `-o <days>`) it deletes all sessions whose `last_updated_at` timestamp is more than `<days>` days in the past.

Tab completion after `/prune ` cycles through session UUIDs newest-first.

### Examples

Delete a single session by UUID:

```text
/prune 550e8400-e29b-41d4-a716-446655440000
```

Delete all sessions except the active one:

```text
/prune all
```

Delete all sessions for a workspace:

```text
/prune -w myproject
```

Delete sessions not used in the last 30 days:

```text
/prune -o 30
```

Natural-language forms:

```text
prune session 550e8400-e29b-41d4-a716-446655440000
prune all
prune sessions in myproject
prune sessions older than 30
```

\newpage

## /session

Lists, switches, and opens sessions. See the Sessions section of the Terminal interface chapter for how sessions are stored, auto-resumed, and cleaned up.

With no argument it lists all sessions found under `~/.orangu/sessions/`, one line per session with aligned columns: UUID, start date, last-updated date, command count, branch, and workspace path. Sessions are sorted by creation time, most-recent first, and the branch column shows `-` for sessions with no recorded branch:

```text
UUID                                  STARTED       LAST          CMDS  BRANCH                WORKSPACE
550e8400-e29b-41d4-a716-446655440000  202605220910  202605221143    42  feature/my-pr         /home/user/myproject
a1b2c3d4-e5f6-7890-abcd-ef1234567890  202605210830  202605210831     3  -                     /home/user/other
```

The argument is resolved in order:

- When it names an existing session directory (a UUID), it switches to that session in place; the current session is saved first.
- Otherwise it is treated as a workspace match: if exactly one session's workspace path contains the string it switches to that session; if several do it lists only those sessions.
- If none match but the argument resolves to a real directory on disk (a leading `~`/`~/` is expanded), it opens that directory as a new workspace, auto-resuming any existing session for it or starting a fresh one.

Tab completion after `/session ` (with a trailing space) cycles through all session UUIDs newest first, then the distinct workspace paths recorded across sessions; when the typed text matches neither, it falls back to filesystem directory completion so a new workspace can be navigated to one segment at a time (`/session ~/co<Tab>/pr<Tab>`).

### Examples

List every session:

```text
/session
```

Narrow to sessions whose workspace path contains a string (switches straight to it when exactly one matches):

```text
/session myproject
```

Switch to a specific session by UUID:

```text
/session 550e8400-e29b-41d4-a716-446655440000
```

Open a directory that no session uses yet as a new workspace:

```text
/session ~/PostgreSQL/pgagroal/official
```

Natural-language forms:

```text
session
sessions
switch session
list sessions
show sessions
```

\newpage

## /list_files

Lists the workspace files as a tree.

This is a local convenience command for quickly orienting yourself in the workspace; it is separate from the model-facing `list_directory` tool that the model uses to explore files on its own.

### Examples

```text
/list_files
```

Natural-language forms:

```text
list files
show files
list workspace files
show workspace files
```

\newpage

## /open_file

Opens a workspace file in your `$EDITOR`.

The command is workspace-scoped — paths outside the workspace are rejected. It launches `$EDITOR` on the file in a separate window so `orangu` stays usable, and it never waits for the editor to close.

Tab completion after `/open_file ` searches the workspace recursively for file paths, and quoted paths such as `/open_file "..."` are supported.

The same `/open_file <path>` (and `open <path>`) form also works inside the `/review` and `/auto_review` split views, where it opens any project file — not only the changed ones — in your editor, with the same whole-workspace `Tab` completion. In `/review` it is available the whole time; in `/auto_review` it works once the run has finished. See the `/review` and `/auto_review` tools for the details.

### Examples

```text
/open_file README.md
/open_file src/main.rs
```

Natural-language forms:

```text
open README.md
open file README.md
edit src/main.rs
edit file src/main.rs
```

\newpage

## /show_file

Shows the contents of a workspace file, optionally at a specific Git ref and optionally with per-line blame columns.

```text
/show_file [--hash] [--author] <path> [<ref>]
```

The command is workspace-scoped. Without a ref, the current workspace file is shown — when `bat` is installed it is used for the plain view, otherwise the built-in syntax-highlighted renderer is used. When a ref (commit hash, branch, or tag) is given, the file content at that ref is retrieved via `git show <ref>:<path>` and rendered with the built-in renderer.

`--hash` and `--author` add per-line blame columns sourced from `git blame`, using the same ref when one is provided.

Tab completion for the first positional argument offers workspace file paths recursively; once a file path is present, the next `Tab` press cycles through that file's commit history (abbreviated hashes from `git log --follow`). `--hash` and `--author` are completed as well.

Because source lines may be wider than the terminal, `/show_file` output is laid out on the full virtual canvas and can be panned horizontally with `Alt+Left`/`Alt+Right` without reflowing.

### Examples

Show the current file:

```text
/show_file src/main.rs
```

Show the file with blame columns:

```text
/show_file --hash --author src/main.rs
```

Show the file as it was at an earlier commit:

```text
/show_file src/main.rs HEAD~3
```

Natural-language forms:

```text
show README.md
show file README.md
```

\newpage

## /build

Builds the workspace project, detecting the toolchain from the workspace root.

Each step is reported individually and the pipeline stops on the first failure:

- **Rust** (`Cargo.toml`) — runs `cargo fmt`, `cargo clippy`, `cargo build`, and `cargo test`.
- **C** (`CMakeLists.txt`) — runs `clang-format.sh` (if present), creates a `build/` directory if needed, runs `cmake ..` on the first build, then `make`.
- **Java** (`pom.xml`) — installs frontend dependencies with `npm ci` when outdated, runs `npm run fix` and `npm run check` for the frontend (if `src/frontend/` exists), then `mvn package`.

### Examples

```text
/build
```

Natural-language forms:

```text
build
build project
run build
```

\newpage

## /review

`/review` opens a full-screen, two-pane view for reviewing the changes on the current branch before sharing them, with the status bar and input window kept at the bottom so you can ask the model for help. It is available inside a Git repository and has no `gh` dependency.

Enter it with the `/review` command, or the natural-language forms `review`, `review changes`, `code review`, or `review branch`.

### What is reviewed

The review shows everything the current branch adds on top of the default branch:

- Committed changes on the branch, and
- Local uncommitted changes (staged and unstaged) in the working tree.

The comparison is made against the merge base with the default branch. The default branch is detected in the usual order: `origin/main`, `origin/master`, `main`, then `master`. If the working tree has no changes against that base, `/review` reports that there is nothing to review and does not open the view.

The branch must be **up to date (rebased)** against the default branch: when commits have landed on main/master that the branch has not incorporated, the review would run against stale code, so `/review` refuses to start and points at `/rebase` instead. The check uses the locally known base ref; nothing is fetched.

### Layout

Above the bottom prompt frame (the status bar and input window, exactly as on the normal screen), the view is split into two panes separated by a single straight vertical line.

- **Left pane** — the diff of the **selected file only**, rendered through the same pipeline as the `/diff` command, including the configured non-interactive git pager (such as `delta`) when one is set. It is the larger pane and scrolls independently.
- **Right pane** — the checklist of changed files, one per row, shown with their full repository-relative paths. The right pane is kept as narrow as possible: just wide enough for the longest path (capped on very narrow terminals so the diff stays usable).

Each file row begins with a review-status box:

- `[ ]` — not yet reviewed
- a green dot in the box — approved
- a red dot in the box — rejected

The currently selected file is highlighted in the right pane. Selecting a different file replaces the left pane with that file's diff, shown from the top.

```
 diff --git a/src/main.rs b/src/main.rs |Files (3)
 @@ -1,4 +1,5 @@                        |[ ] README.md
 +fn new() {}                           |[*] src/main.rs  <- selected
  (only the selected file's diff,       |[*] src/git.rs
   scrollable)                          |
                                        |
```

### Asking the model to review a file

The input window at the bottom takes a request for the model about the selected file — for example `focus on error handling` or `is this thread-safe?`. Press `Alt+o` (or `Enter`) to send that request together with the selected file's diff to the LLM. While the model works, the status bar shows the usual thinking indicator over the panes.

While the model works you are not stuck: press `Esc` twice to cancel the request and return to the diff, or `Alt+x` to leave review mode entirely. A cancelled request is rolled back out of the session.

When the response arrives it opens in a **feedback window** over the panes. If you typed a question, it is echoed at the top of the window — styled like a submitted prompt in the main output — above the model's review. A plain `Alt+o` (empty input) just asks for a plain review of the file, with no question echoed. Press `x` (or `Esc`) to close the window and return to the diff.

The request and the model's reply are added to your chat session, so after leaving review mode you can keep discussing it with full context.

### Opening a file

Press `Alt+e` to open the currently selected file in your `$EDITOR` — the same way as the `/open_file` command. Terminal editors open in a new window (the configured `terminal` command, or an auto-detected emulator) and GUI editors open their own window; either way the editor is detached, leaving review mode on screen so you can keep working through the diff. If the file cannot be opened, the error is shown in a feedback window.

To open **any file in the project — not only the changed files** — type `/open_file <path>` (or just `open <path>`) into the input window and press `Enter`. `Tab` completes the path against every file in the workspace, exactly like `/open_file` at the main prompt: typing a bare name matches files anywhere in the tree, and the grey ghost previews the first match. This open form is available the whole time you are in `/review`. Anything you type that is **not** an `open <path>` (or `edit <path>`) line is still sent to the model as a review request, so an ordinary request such as `focus on error handling` is unaffected.

### Commenting on a line

Move the highlighted line to the place you want to comment on and press `Alt+c`. A small comment window opens **inline, just below that line** in the left pane. Its first row is a single-line **category selector** — the same categories as `/auto_review`: **Overall**, **Code**, **Security**, **Memory**, **Performance**, **Test Suite**, and **Documentation** — and below it the comment text. The focus starts on the text, so you can type right away (it wraps and the five-line window scrolls if the comment is long). Press `Tab` to switch the focus between the category selector and the comment text; while the selector has the focus, `Up`/`Down` move through the categories. Press `Enter` to save the comment or `Esc` to discard it.

Each comment defaults to the **Overall** category and is recorded against the file, that diff line, and the chosen category; lines with a comment are flagged with an amber dot at the right edge. Pressing `Alt+c` on a line that already has a comment re-opens it for editing — pre-filled with both its category and text — and saving an empty comment removes it.

You can also add a **general note** about the patch: type `# <note>` in the input window and press `Enter` (or `Alt+o`). Instead of being sent to the model, it is recorded as a general note (the `#` is dropped). Anything not starting with `#` is still treated as an LLM request.

When you leave review mode (`Alt+x`), a **category-grouped report** is written to the output window, laid out like the `/auto_review` report. Each category — **Overall**, **Code**, **Security**, **Memory**, **Performance**, **Test Suite**, and **Documentation** — is a heading, under which its line comments appear as a bullet list (`<file>:<line>: <comment>`, ordered by file then line, with 1-based line numbers); a category with no comments reads `No issues found`. The general `# <note>` notes are whole-patch commentary, so they lead the **Overall** category. The report closes with a **Conclusion**: the bold verdict — `Patch approved` when every file is approved, otherwise `Patch rejected` — followed by any `Rejected:` or `Not reviewed:` files (approved files are implied by the verdict).

The whole report's Markdown is copied to the system clipboard on exit. If the clipboard cannot be reached (for example on a headless machine), a short note is shown instead and the output-window report is unaffected.

The same Markdown is also kept for the rest of the session, so `/comment <number> with review` can post it on a GitHub/GitLab issue (see the `/comment` tool in the Git tools chapter) and `/export review` can write it to a PDF (see the `/export` tool).

### Key bindings

| Key | Action |
| --- | --- |
| `Alt+j` | Select the next file (shows its diff in the left pane) |
| `Alt+k` | Select the previous file (shows its diff in the left pane) |
| `Alt+a` | Mark the selected file approved (green dot) |
| `Alt+r` | Mark the selected file rejected (red dot) |
| `Alt+c` | Comment on the highlighted line — pick a category (`Tab` to focus it, `Up`/`Down` to move), `Enter` saves, `Esc` discards |
| `Alt+e` | Open the selected file in your configured editor |
| `/open_file <path>` / `open <path>` + `Enter` | Open any project file in your configured editor (`Tab` completes every workspace file) |
| `Alt+o` / `Enter` | Ask the model to review the selected file using the typed request (when the line is not an `open <path>`) |
| `Esc` `Esc` | Cancel an in-progress review request (while the model is thinking) |
| `Alt+x` or `Esc` `Esc` | Exit review mode and return to the prompt |

When the feedback window is open it is modal: `x` or `Esc` closes it, and `Up`/`Down`, `PageUp`/`PageDown`, and `Left`/`Right` scroll and pan it.

Otherwise you can type into the input window normally, and move through the selected file's diff:

- `Up` / `Down` move a highlighted line cursor through the diff; the pane scrolls to keep it in view
- `Alt+Up` / `Alt+Down` scroll the diff one line at a time without moving the cursor
- `PageUp` / `PageDown` scroll the diff by a full page
- `Alt+Left` / `Alt+Right` pan horizontally for long lines
- `Tab` completes an `/open_file <path>` / `open <path>` line against the workspace files (`Shift+Tab` cycles the ghost preview)

The review status marks and line comments are kept for the duration of the review session and are not persisted after exit.

### Examples

```text
/review
```

Natural-language forms:

```text
review
review changes
code review
review branch
```
\newpage

## /auto_review

`/auto_review` runs an LLM-driven review of the changes on the current branch, in a full-screen, two-pane view modeled on `/review`. The model reviews the changes overall and each file by itself, sorts what it finds into the **Overall**, **Code**, **Security**, **Memory**, **Performance**, **Test Suite**, and **Documentation** categories, and marks each file approved or rejected. It is available inside a Git repository and requires a connected LLM server.

Enter it with the `/auto_review` command, or the natural-language form `auto review`. Give it a file — `/auto_review <file>` — to review just that one file instead of the whole branch (see *Reviewing a single file* below).

### What is reviewed

The same change set as `/review`: everything the current branch adds on top of the default branch — committed changes on the branch plus local uncommitted changes — measured against the merge base with the default branch (`origin/main`, `origin/master`, `main`, then `master`). If there is nothing to review, `/auto_review` reports that and does not open the view.

Like `/review`, the branch must be **up to date (rebased)** against the default branch before the auto review starts: when the branch is behind main/master, the command refuses with `The branch is N commits behind <base>; run /rebase before reviewing.` — reviewing against stale code would waste the run and could approve changes that conflict with the newer base.

### Reviewing a single file

`/auto_review <file>` reviews one file rather than the whole branch. The view, the categories, and the report are exactly the same as a whole-branch run — there is just one file in the checklist. What gets reviewed depends on the branch you are on:

- **On `main`/`master`** the whole file is reviewed — a full read of its current content, every line in scope — not a diff. This is the way to have the model review a file that is not part of any in-progress change.
- **On any other branch** only the file's **changes** against the default branch are reviewed, exactly as in a whole-branch run (the same rebased-branch guard applies).

The natural-language form takes a file too: `auto review <file>` is equivalent to `/auto_review <file>`. The file argument is resolved by **Tab completion** in either form, and it completes on the file's **name, not its location** — typing `t` and pressing Tab offers `src/tui.rs`. The candidate list matches what will be reviewed: on `main`/`master` it is every tracked file (files ignored by `.gitignore` are excluded); on any other branch it is only the files that differ from the default branch. Selecting a candidate fills in its full repository-relative path; a hand-typed bare name (e.g. `tui.rs`) is resolved too. On a branch, a file with no changes against the default branch is refused with `'<file>' has no changes against <base>.`

### Layout

The view opens with the tool header row at the top and under it the two panes, exactly like `/review`; the **status area** is the first row of the left pane, so the file checklist on the right keeps its full height. The input window stays empty — auto review takes no typed request.

- **Header row** — the tool title (`Auto review: <branch>`) and the key help, with the `Files (n)` header of the right pane.
- **Status area** — a highlighted bar across the left pane, just below the header, showing what is being worked on: the file (with its position in the file list), the category, the overall progress across all of the run's requests, the total time spent on the run so far, and the estimated time still to go, e.g. `File: src/main.rs (2/5)  Category: Security  Progress: 8/26 (30%)  Time: 1m12s  Estimated: 2m48s`. Both times use the same shortest form as the Thinking/Working timers (`5s`, `1m5s`, `1h2m3s`). The **estimate** is the average time per completed request so far extrapolated over the requests still to run; it is recomputed after each request finishes and counts down between them. It appears once the first request completes and drops away when the run ends — after the run the bar shows `Done` (or `Cancelled`) with the time frozen at the run's total.
- **Left pane** — below the status area, the **report**, rendered from Markdown with the syntax markers consumed: one bold heading per category (Overall, Code, Security, Memory, Performance, Test Suite, Documentation), each listing the findings collected so far as a bullet list with the file names in bold, ending with the **Conclusion**. A category that has produced nothing yet shows `(pending)` while the run is in progress, and `No issues found` once it is done. The pane scrolls and pans independently.
- **Right pane** — the checklist of changed files, one per row, as in `/review`. The file currently being reviewed is highlighted and its status box blinks a white dot until its review resolves to green or red. Once the run ends (or the whole-change pass starts) the highlight is cleared — nothing is being reviewed anymore; `Alt+j`/`Alt+k` bring it back to move through the list while browsing.

While the run is in progress the header row offers the run keys (`Esc Esc Cancel  Alt+x Exit`); once the run has ended it switches to the browse keys (`Alt+j/k Switch file  Alt+a Approve  Alt+r Reject  Alt+e Open  ↑/↓ Item  PgUp/PgDn Category  - Remove  Alt+x Exit`).

```
 Auto review: feature/x ...                          |Files (3)
 File: src/main.rs (2/3) ... Time: 45s Estimated: 1m |[*] README.md
 Overall                                |[o] src/main.rs  <- reviewing (blinks)
   (pending)                            |[ ] src/git.rs
 Code                                   |
   - src/main.rs:42: unwrap may panic   |
 Security                               |
   (pending)                            |
```

### How the review runs

The files are reviewed one at a time, in diff order. Each file's name and extension enable the categories that are scanned, so the run spends its requests only where a review can act:

- A file on the **skip list** is approved at once, with **no requests**. These are files whose diff a review cannot act on: generated dependency **lock files** — by extension `.lock` (`Cargo.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, `composer.lock`, `flake.lock`, …) and by name `package-lock.json`, `npm-shrinkwrap.json`, `pnpm-lock.yaml`, `go.sum`, `go.work.sum` — and **binary assets**: images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.ico`, `.svg`, `.webp`, `.tiff`, `.tif`), fonts (`.otf`, `.ttf`, `.woff`, `.woff2`, `.eot`), and `.pdf`, `.p12`, `.jks`, `.keystore`.
- A file detected as **documentation** (`.md`, `.markdown`, `.mkd`, `.mdown`, `.mdx`, `.rst`, `.adoc`, `.asciidoc`, `.txt`, `.text`, `.org`, `.tex`, `.texi`, `.texinfo`, `.pod`, `.rdoc`) skips the code-related checks and is reviewed only for the **Documentation** category — a single request per file.
- Every other file — the fallback — is scanned for all six per-file categories: Code, Security, Memory, Performance, Test Suite, then Documentation. Build and metadata files take this full review too: extensionless ones (`Makefile`, `Dockerfile`) fall through to it automatically, and the few that carry a documentation-looking extension (`CMakeLists.txt`, `requirements.txt`) are pulled back into it by name so a `.txt` extension does not demote them to documentation only.

For each enabled category, one focused request is sent to the LLM asking for a verdict plus findings for that category only, with the file's diff attached. The review is explicitly scoped to **the changes made** — the added, removed, and modified lines, and how they fit into the surrounding context — not to pre-existing content the change does not touch, and each category is capped at five short findings. Each finding is prefixed with its location as `<file>:<line>:` — the affected **line number**, or range as `<start>-<end>`, in the new version of the file (the right side of the diff) — so the report points at where each issue lives, e.g. `src/main.rs:42: unwrap may panic`. The status area names the file and category being worked on and counts the overall progress — the total reflects only the enabled categories — while the status bar shows the usual thinking indicator.

As each category review arrives, its findings are appended to the matching section in the left pane, each prefixed with its location (`file:line`) in bold — so the report fills in category by category. When all of a file's categories have run, the file is automatically marked in the right pane — a **green dot** when every category passed, a **red dot** when any category rejected. Without an explicit verdict, a category passes only when its review found nothing. If a request fails — or its response carries neither a verdict nor findings (for example, truncated by the response cap) — the file keeps its white (unreviewed) box and the problem is noted under Overall; such a response never passes silently as a clean review.

After the last file, a final pass reviews the change as a whole: the per-file verdicts and findings are summarized by the model into a few bullet points — how the changes fit together, readiness, risk, and common themes — under **Overall**.

The report ends with the **Conclusion**, derived from the file statuses rather than from the model: the verdict — `orangu approves this patch` when every file is approved, or `orangu rejects this patch` when any file was rejected or not reviewed — stands alone in bold rather than as a list item; the affected files then follow as a bullet list in bold, grouped by their status (`Rejected: **file**`, `Not reviewed: **file**`). A closing **`Generated by: orangu <version> (<model>)`** line credits the orangu version and the reviewing model, e.g. `Generated by: **orangu 0.7.0** (gemma)`.

Each request runs in its own scratch exchange, **without tool definitions** and with a **capped response length** (`[orangu].review_max_tokens`, default `512`; `0` disables the cap) — a review can neither wander off into tool calls nor generate unbounded output, which keeps single requests fast and bounded even on slow local models. For deeper reviews with a thinking model, raise the cap (e.g. `2048`) so the thinking tokens do not eat the answer; the *Response-token caps* part of the Configuration chapter covers the trade-offs in depth. The diff leads each prompt, so a file's category requests share their prefix and llama.cpp's prompt cache can reuse the processed diff across them. The reviews are independent of each other and nothing is added to your chat session.

While the model works, the status bar shows `Thinking (...)` until the first token arrives and then the live generation rate (`Working @ X.Y t/s (...)` on llama.cpp), so a stalled server and a slowly generating model are easy to tell apart.

A full-branch auto review can take a while, so when **`feedback` is on** (see the Configuration chapter) orangu surfaces its progress outside the window too: while the run is in progress the **terminal title** reads `orangu ●` with the white dot blinking once a second, so a backgrounded or unfocused terminal still shows that a review is running. When the run **finishes**, orangu rings the **terminal bell** — the standard desktop notification sound (or a visual flash, depending on your terminal) — and drops the title back to a plain `orangu`. A run that is cancelled (`Esc Esc`) or exited (`Alt+x`) before it finishes does not ring. With `feedback` off, neither the title nor the bell is touched.

### Browsing and overriding the report

Once the run has ended (done or cancelled), the report stays on screen and you can override the model's verdicts file by file. `Alt+j`/`Alt+k` move the highlight through the file list — from no highlight, `Alt+j` starts at the first file and `Alt+k` at the last.

You can also work through the report **item by item** in the left pane. `Up`/`Down` move a highlight between the individual report items — the findings and the Conclusion entries, never the category headings — scrolling the pane as needed to keep the highlighted item in view (from no highlight, `Down` starts at the first item and `Up` at the last). Moving the highlight also points the file list on the right at the item's file, so `Alt+a`/`Alt+r` act on it.

To skip across a long report **category by category**, use `PageDown`/`PageUp`: they jump the highlight straight to the first item of the next or previous category that **has findings**, scrolling that category's heading to the top so the whole section comes into view. Empty categories (those reading `No issues found`) are skipped, and the Conclusion entries count as the final category. From no highlight, `PageDown` lands on the first category with findings and `PageUp` on the last; once the highlight is already in the last (or first) such category the key is a no-op, so it never jumps backward past the report. Use `Up`/`Down` to walk the individual findings within a category.

- **`-` — remove the highlighted item.** A **finding** is dropped from its category; if that was the **last** finding recorded against its file, the file is approved (its dot turns green) and it drops out of the Conclusion. Removing a **Conclusion** item approves the whole file it stands for, clearing every finding recorded against it across the report — the same as approving that file. So you can approve the patch outright by removing all the flagged items: once nothing is left, every file is approved and the verdict reads `orangu approves this patch`.
- **`Alt+a` — approve the highlighted file.** Its dot turns green and **every finding recorded against it is removed from the report** — the model's findings and your own rejection comments alike — so an approved file no longer appears in any category, in the exit report, or on the clipboard. The Conclusion follows the file statuses, so approving the last rejected file flips the verdict to `orangu approves this patch`.
- **`Alt+r` — reject the highlighted file.** A reject window opens over the panes with a **category selector** (Overall, Code, Security, Memory, Performance, Test Suite, Documentation) and a **multi-line Markdown comment editor**. `Tab` moves the focus between the two; in the selector `Up`/`Down` pick the category (`Enter` moves on to the editor), and in the editor `Enter` inserts a newline while `Up`/`Down`, `Home`/`End`, and the usual editing keys move and edit. Press `Alt+Enter` to save — the file's dot turns red and the comment is appended to the chosen category, prefixed with the file path in bold — or `Esc` to discard the window. Saving with an empty comment still rejects the file without adding a finding. `Alt+r` can be repeated on the same file; each saved comment is kept.
- **`Alt+e` — open the highlighted file** in your `$EDITOR`, exactly like `Alt+e` in `/review`: terminal editors open in a new window and GUI editors open their own, leaving the report on screen.
- **`/open_file <path>` / `open <path>` + `Enter` — open any project file**, not only the changed ones. Once the run is done the input window at the bottom accepts an open command: type `/open_file <path>` (or `open <path>`), with `Tab` completing every workspace file just like `/open_file` at the main prompt, and press `Enter` to open it in your `$EDITOR`. This works **only after the run has finished** — during the run the input window stays empty. While the input is empty, `-` still removes the highlighted item; a `-` typed into a path is left for editing.

Rejection comments become part of the report: they are rendered in the matching category of the left pane (and the exit report), and land on the clipboard as Markdown bullets — a multi-line comment keeps its lines inside one bullet, indented under the first line.

### Cancelling and exiting

Press `Esc` `Esc` to **cancel** the auto review: the in-flight request is dropped, the run stops, and the report collected so far stays on screen for browsing. Press `Alt+x` (or `Esc` `Esc` again once the run is no longer in progress) to **exit**.

On exit the report — every category with its findings (or `No issues found`), the **Conclusion** and its patch verdict, and the closing `Generated by:` line — is rendered into the output window exactly like the left pane (bold headings and file names, no Markdown syntax markers), while its raw Markdown is copied to the system clipboard: there each category is a `##` heading and its findings a bullet list with the file names in `**bold**`, ready to paste into an issue or pull request. The per-file statuses are not listed separately: the rejected and not-reviewed files appear inside the Conclusion (`Rejected: **file**`, `Not reviewed: **file**`). If the clipboard cannot be reached (for example on a headless machine), a short note is shown instead.

The Markdown report is also kept for the rest of the session, so `/comment <number> with auto review` can post it on a GitHub/GitLab issue (see the `/comment` tool in the Git tools chapter).

### Key bindings

| Key | Action |
| --- | --- |
| `Esc` `Esc` | Cancel the auto review (the collected report stays open) |
| `Alt+x` | Exit auto review mode; the report is copied to the clipboard |
| `Esc` `Esc` (after the run) | Exit auto review mode, like `Alt+x` |
| `Alt+j` / `Alt+k` | Move the highlight through the file list (after the run) |
| `Alt+a` | Approve the highlighted file and drop its findings from the report |
| `Alt+r` | Reject the highlighted file: pick a category, write a Markdown comment |
| `Alt+e` | Open the highlighted file in your configured editor |
| `/open_file <path>` / `open <path>` + `Enter` | Open any project file in your editor (after the run; `Tab` completes every workspace file) |
| `Up` / `Down` | Move the item highlight through the report's findings and Conclusion entries (after the run) |
| `PageUp` / `PageDown` | Jump the item highlight to the previous / next category that has findings (after the run) |
| `-` | Remove the highlighted item; clearing a file's last item approves it (after the run, while the input window is empty) |
| `Alt+Left` / `Alt+Right` | Pan the report horizontally for long lines |

The report can be scrolled while the run is still in progress; `Alt+a`, `Alt+r`, `Alt+e`, and the `/open_file` input act once the run has ended.

When the reject window is open it is modal:

| Key | Action |
| --- | --- |
| `Tab` | Move the focus between the category selector and the comment editor |
| `Up` / `Down` | Pick the category (selector) / move the cursor (editor) |
| `Enter` | Move on to the editor (selector) / insert a newline (editor) |
| `Alt+Enter` | Save: mark the file rejected and add the comment to the category |
| `Esc` | Discard the window without saving |

### Examples

```text
/auto_review
```

Review a single file (Tab-completes on the file name):

```text
/auto_review src/tui.rs
```

Natural-language form (a whole-branch review, or a single file):

```text
auto review
auto review src/tui.rs
```

\newpage

## /export

`/export` writes a buffer to a PDF file in the root of the workspace, so a session's output or a review can be saved and shared outside the terminal.

It takes one optional argument selecting what to export:

- `/export` or `/export console` — the **console output window**: everything currently in the main output window (prompts, command output, and model responses), with the terminal ANSI styling removed and the lines printed verbatim.
- `/export review` — the **review buffer**: the Markdown of the last `/review` (or, if none, the last `/auto_review`) report from this session. If no review has been run yet, the command reports that there is nothing to export.

The file is saved in the workspace root as `{repository}-{branch}-console.pdf` or `{repository}-{branch}-review.pdf`, where `{repository}` is the Git repository (or workspace) directory name and `{branch}` is the current branch (`nobranch` when not on one); both are sanitized for use in a filename, so a branch such as `feature/x` becomes `feature-x`. An existing file with the same name is overwritten. On success the saved path is printed to the output window.

Every page carries a **header band** centered on `{repository}-{branch}` and a **footer band** centered on `orangu {version} ({model})` (the active model), both in white on the orangu brand colour to match the terminal banner; in the footer the word `orangu` links to the project site.

The PDF keeps the Markdown formatting as much as a self-contained file can. Text is set in **Red Hat Text**, embedded into the binary (SIL Open Font License), so no system fonts are needed; if for any reason the embedded font cannot be loaded the export falls back to the closest built-in face, Helvetica. Headings use the orangu brand colour. Long lines wrap to the page width using the font's real glyph metrics — prose on word boundaries, code lines hard at the margin — and the content flows across as many pages as needed.

The **console** export preserves the output window line for line (terminal colours removed).

The **review** export is organized for reading and sharing:

- **Page 1 — summary.** A table of the repository, branch, and generation date/time, followed by one row per category with its number of entries (findings), and then an overall **Approved** (green) or **Rejected** (red) status banner read from the report's conclusion.
- **Page 2 — table of contents.** Each category with the page it starts on.
- **Page 3 onward — the report.** Each category (`Overall`, `Code`, `Security`, `Memory`, `Performance`, `Test Suite`, `Documentation`, then `Conclusion`) starts on its own page, rendered from the report's Markdown with brand-coloured headings, **bold** and *italic* emphasis, ordered and unordered (including nested) lists, fenced code blocks, block quotes, and tables. So `Overall` opens on page 3.

### Examples

Export the output window (the default):

```text
/export
/export console
```

Export the last review report:

```text
/export review
```

Natural-language forms:

```text
export
export console
export review
```

\newpage

# Terminal interface

`orangu` is an interactive terminal client with a persistent header and a prompt area anchored to the bottom of the terminal.

The individual commands you can type at the prompt are documented in the Core tools, Git tools, and Usage tools chapters. This chapter covers the terminal itself: startup, the header and prompt, sessions, input editing, completion, and scrolling.

## Startup

When started inside a Git repository, **orangu** fast-forwards the local default branch (`main`/`master`) to `origin` so it is in sync with upstream. If you are on the default branch it fast-forwards your working tree (`git pull --ff-only`); on any other branch it fast-forwards the local default ref in place (`git fetch origin <branch>:<branch>`) without touching your current branch or working tree. It never creates a merge commit or rebases.

The sync runs in the background so it never delays startup. Its progress and result appear on the left of the status bar — `Syncing with origin…` while it runs, then `Synced <branch> with origin` (or `Sync failed: …`) for a few seconds. It is skipped silently when there is no `origin` remote, and a diverged branch or an unreachable `origin` is reported only on the status bar; startup continues normally regardless.

### Command-line options

| Short | Long          | Description                                                   |
| ----- | ------------- | ------------------------------------------------------------- |
| `-c`  | `--config`    | Path to the configuration file (`orangu.conf`).               |
| `-w`  | `--workspace` | Workspace root the local tools operate on. Defaults to `.`.   |
| `-r`  | `--resume`    | Resume a specific session by UUID.                            |
| `-a`  | `--all`       | Reopen the workspace tabs that were open at the end of the last run. |
|       | `--theme`     | Apply a theme for this run. Built-in names and paths to `.theme` files are accepted. |
| `-l`  | `--list`      | List all stored sessions as a `SESSION WORKSPACE BRANCH DATE` table and exit. |
| `-i`  | `--init`      | Interactively create `~/.orangu/orangu.conf` and exit (see the Configuration chapter). |
| `-s`  | `--shell-completions` | Print the shell completion script for the detected shell (`$SHELL`; bash, zsh, or fish) and exit. |

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

All slash commands are handled locally and are never sent to the model, so they continue to work even when the model is unavailable. Free-form prompts, by contrast, are blocked when the server or model status in the header is red.

## Waiting state

While the model is generating a response, the left side of the footer shows a rolling:

```text
Thinking (2s)
```

status indicator.

You can keep typing and submitting commands while a response is pending. Submitted commands are queued and executed in order after the active response finishes.

The footer starts with `Thinking (<CLOCK>)` and switches to `orangu-server`'s native generation throughput once tokens are streaming, for example `Working @ 42.5 t/s (2s)`.

Press `Esc` twice within 2 seconds during the waiting state to cancel the active request without exiting the client. Queued commands are preserved.

## Sessions

Each run of `orangu` creates or resumes a session identified by a UUID. The `/session` command (see the Core tools chapter) lists and switches between them; this section describes how they are stored and resumed.

### Automatic resume

On startup, `orangu` checks whether a session already exists for the current workspace path and Git branch. If exactly one matching session with conversation history is found, it is resumed automatically. The status bar lower-left shows:

```text
Resuming session 550e8400-e29b-41d4-a716-446655440000
```

for five seconds or until the first command is run. No `--resume` flag is needed for normal branch-based workflows.

If more than one session matches the current workspace and branch, a fresh session is started instead. Use `--resume <uuid>` to target a specific session explicitly.

### Manual resume

To resume a specific session regardless of workspace or branch, pass `--resume <uuid>` (short form `-r`) when starting:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

This restores the previous conversation context and per-session readline history.

### Listing sessions

To see every stored session without starting a run, pass `--list` (short form `-l`):

```text
orangu --list
```

This prints a table of all sessions, newest first, with the columns sized to the
widest value in each:

```text
SESSION                               WORKSPACE           BRANCH         DATE
550e8400-e29b-41d4-a716-446655440000  /home/user/project  main           2026-06-26 11:04
6ba7b810-9dad-11d1-80b4-00c04fd430c8  /home/user/other    feature/login  2026-06-26 03:27
```

The `DATE` column is the session's last-updated timestamp (`YYYY-MM-DD HH:MM`).

`orangu` then exits. To list and switch between sessions from inside a running
session, use the `/session` command (see the Core tools chapter).

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

## History and navigation

Command history is stored per session in:

```text
~/.orangu/sessions/<uuid>/history
```

Use:

- `<ARROW_UP>` to move backward in history
- `<ARROW_DOWN>` to move forward in history

## Natural-language command aliases

Local commands can also be entered in plain language — for example `open README.md`, `show status`, `create pull request`, or `switch model to <name>`. The phrases recognized for each command are listed under its **Examples** in the Core tools, Git tools, and Usage tools chapters.

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

### Inline command hints (ghost text)

As you type, a grey inline hint previews the command your input is growing into, drawn just after the cursor. It covers both slash commands and the natural-language bindings:

- Typing `/q` shows `/q``uit`, with `uit` greyed; typing `c` shows `c``urrent model`.
- Press `Tab` to accept the hint, filling in the rest of the command (for an argument-taking form such as `diff against `, the cursor lands after the trailing space, ready for the argument).
- When several commands share your prefix (for example `c` matches `current model`, `code review`, `checkout`, `commit`, and more), `Shift+Tab` cycles the hint through them in priority order, wrapping back to the first. `Tab` then accepts whichever candidate is currently shown.
- The hint only appears while the cursor is at the end of the line, and disappears once your input already spells a complete command (so `status` and `diff` show no hint, even though `diff against ` shares the latter's prefix).
- Editing the line, moving the cursor, or pasting resets the `Shift+Tab` cycle back to the first candidate.

The natural-language hint takes priority over generic filename completion, so `c` + `Tab` completes to `current model` rather than a same-prefixed file such as `contrib/`. Slash-command and argument completion (branches, files, commit hashes, and so on) continue to use the cycling `Tab` behavior described next.

### Tab completion

`Tab` uses context-sensitive completion. The first `Tab` inserts the first match. Repeated `Tab` presses cycle through the remaining matches for the same completion range.

Completion cycling is reset as soon as you edit the line, move the cursor, paste text, or otherwise change the input.

The completion modes are checked in order:

1. If the line starts with `/branch `, `/checkout `, or with the natural-language prefixes `checkout ` or `git checkout `, complete branch names first (from `git branch --all`), then workspace file paths. Branch names always appear before file names in the candidate list. If the line starts with the natural-language prefix `switch to `, complete branch names and tag names (from `git tag`), sorted together; workspace file paths are excluded.
2. If the line starts with `/create_file `, or with the natural-language prefixes `create `, `create file `, `add `, `add file `, or `git add `, complete untracked directories first (from `git ls-files --others --directory`), then untracked files. Already-tracked content is excluded.
3. If the line starts with `/delete_file `, or with the natural-language prefixes `delete file `, `remove `, `remove file `, or `git rm `, complete tracked directories first (from `git ls-files`), then tracked files. Untracked content is excluded.
4. If the line starts with `/move_file `, or with the natural-language prefixes `move `, `move file `, or `git mv `, complete the first argument from tracked directories and files; complete the second argument from all workspace paths.
5. If the line starts with `/cherry_pick `, or with the natural-language prefixes `cherry pick `, `cherry-pick `, or `git cherry-pick `, complete abbreviated commit hashes from the default branch (`origin/main`, `origin/master`, `main`, or `master`, tried in that order).
6. If the line starts with `/fetch `, or with the natural-language prefixes `fetch ` or `git fetch `, complete the configured remotes (from `git remote`), with `origin` floated to the front so the default is offered first and previewed as the inline ghost.
7. If the line starts with `/rebase `, or with the natural-language prefixes `rebase ` or `git rebase `, complete the rebase target in priority order: local branch names first (from `git branch`), then the configured remotes (from `git remote`, `origin` floated to the front), then the remote-tracking branches (from `git branch --all`, e.g. `origin/main`). The first local branch is previewed as the inline ghost.
8. If the line starts with `/merge `, or with the natural-language prefixes `merge ` or `git merge `, complete local branch names first (from `git branch`), then remote-only branch names (from `git branch --all`).
9. If the line starts with `/branch -d `, or with the natural-language prefixes `delete `, `delete branch `, or `git branch -D `, complete local branch names (from `git branch`) excluding `main` and `master`.
10. If the line starts with `/session ` (with a trailing space), complete session UUIDs sorted newest-first by last-modified time.
11. If the line starts with `/model `, complete the models available on the selected server, cycling through them. If the line starts with `/server `, complete the names of all INI sections identified as servers; selecting one switches the active server.
12. If the line starts with `/open_file ` or `/show_file `, complete workspace file paths recursively for the first positional argument. `/show_file` also completes `--hash` and `--author`. When a file path is already present, the next Tab press cycles through that file's commit history (abbreviated hashes from `git log --follow`).
13. If the line starts with the natural-language prefixes `open `, `open file `, `edit `, or `edit file `, complete workspace file paths recursively.
14. If the line starts with `/`, complete built-in slash commands and any
    discovered Agent Skills. Examples include `/help`, `/skills`, `/model`,
    `/server`, `/list_files`, `/show_file`, `/tools`, `/quit`, and a discovered
    skill such as `/debugging`. When the `drop_down` option is enabled in
    `orangu.conf` (the default), this completion is also visualized as an interactive
    dropdown menu that intercepts the Up/Down arrow keys.
15. Otherwise, complete filesystem entries from the current token relative to the workspace, using the token before the cursor.

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

## Built-in manual

Type `/manual` at the prompt (or the natural-language forms `manual`, `show manual`, or `open manual`) to open this manual inside the client. The manual text is embedded into the binary at compile time, so no external files are read — the manual is always available, even offline.

The viewer uses the same full-screen, two-pane layout as `/review`, with the status bar and an (inactive) input window kept at the bottom:

- **Left pane** — the text of the selected section, rendered with the same Markdown styling as model output in the console (bold, italics, headings, lists, links, and tables). Fenced code blocks are shown syntax-highlighted according to their language tag, without the ``` fence lines, and links are shown as their underlined labels only. It is the larger pane and scrolls independently.
- **Right pane** — the table of contents, one entry per section. The sections follow the page breaks of the printed manual (one entry per `\newpage`-delimited page): chapter entries are flush left and their sections are indented beneath them. The pane is kept as narrow as possible while still fitting the longest entry. The selected entry is highlighted, and selecting a different entry replaces the left pane with that section's text, shown from the top.

```
 # Core tools                           |Contents (56)
                                        |Introduction
 The core tools are the local slash     |Quickstart
 commands that drive the client ...     |Core tools    <- selected
 (only the selected section's text,     |  /help
  scrollable)                           |  /server
```

### Searching

Press `Alt+S` to open a search window at the top of the text pane. Type the text to find and press `Enter` to jump to its next occurrence: the search is case-insensitive, scans forward from the highlighted line through the **entire manual** — continuing past the end of the current section into the following ones and wrapping around to the beginning — and highlights the matching line. Press `Enter` again to jump to the next instance. If the text is not found anywhere, `No match for '<text>'` is shown on the status bar.

Press `Esc` to close the search window; the highlighted line stays on the last match, so you can keep reading from there.

### Key bindings

| Key | Action |
| --- | --- |
| `Alt+J` | Select the next section (shows its text in the left pane) |
| `Alt+K` | Select the previous section |
| `Alt+S` | Open the search window (`Enter` next match, `Esc` close) |
| `Up` / `Down` | Move the highlighted line through the text, view following |
| `Alt+Up` / `Alt+Down` | Scroll the text one line at a time |
| `PageUp` / `PageDown` | Scroll the text by a full page |
| `Left` / `Right` | Pan long lines horizontally |
| `Alt+X` or `Esc` `Esc` | Leave the manual and return to the prompt |

## Footer behavior

- The left side of the footer shows `Thinking (<CLOCK>)` while waiting for a response to start, and `Working @ X.Y t/s (<CLOCK>)` while tokens are streaming
- The center side of the footer shows `Pending: X` to show how many queued commands are waiting
- The right side of the footer shows the model name used

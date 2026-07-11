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

## /information

Reports everything `orangu` can learn about the active server: every OpenAI-compatible endpoint orangu itself talks to (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`), plus whatever llama.cpp-native endpoints it exposes. It is handled entirely locally, needs no arguments, and never sends anything to the model.

`/information` probes each capability independently — a plain OpenAI-compatible server (which has no llama.cpp-native endpoints) still gets a full report, just with those rows marked unavailable rather than the whole command failing. The result is a table, one row per capability, with a green dot for a capability that is available and enabled and a red dot for one that is not (shown below as `x`, since this static example cannot render color):

Most probes are side-effect-free `GET` requests, so they run unconditionally. `/v1/chat/completions` is the one exception — it only accepts real generation requests — so it is only actually sent on a local llama.cpp server; on a hosted API it is not sent, to avoid a needless (potentially billed) request (see the `/v1/chat/completions` row below).

```text
Server  main-server
Model   ggml-org/gemma-4-E4B-it-GGUF
Graph   Complete

STATUS  API        ENDPOINT              DETAILS
●       OpenAI     /v1/models            ggml-org/gemma-4-E4B-it-GGUF
●       OpenAI     /v1/chat/completions  Ok
x       OpenAI     /v1/embeddings        Not available
●       llama.cpp  /health               Ok
●       llama.cpp  /props                n_ctx=32768 n_predict=-1 total_slots=1 temperature=0.8 top_k=40 top_p=0.95 model_path=/models/gemma.gguf bos_token=<bos> eos_token=<eos> build=b4200-abc1234 chat_template=yes
x       llama.cpp  /slots                Not available
x       llama.cpp  /metrics              Not available
```

The header table's third line, **`Graph`**, is not a probed capability — it is the local workspace's background knowledge-graph scan status (the same one Deep `/auto_review` and `graph_lookup` depend on): `Building` while the scan is still running, `Complete` once it has finished, `None` if the scan task itself failed.

Each row:

- **`/v1/models`** — the standard OpenAI model listing, the same request `/model` and startup detection use; the details column lists every advertised model id, comma-separated (a single advertised model is shown bare, with no count prefix).
- **`/v1/chat/completions`** — the endpoint every prompt and tool round-trip actually goes through. On a local llama.cpp server (`provider = llama.cpp`) a real generation costs nothing worth avoiding, so `/information` actually sends one: a minimal, non-streaming request capped at a single response token (`ggml-org/gemma-4-E4B-it-GGUF`'s row above shows this). On any other provider — potentially a hosted, billed API — it is not sent a real request; its availability is instead inferred from `/v1/models`, since any server that speaks the OpenAI protocol well enough to list models is expected to serve chat completions too.
- **`/v1/embeddings`** — likewise never sent a real request; this row reflects whether the active server is the one `orangu` already detected at startup as embeddings-capable (the same detection `/search` relies on — see the `role = embeddings` configuration option).
- **`/health`** — a llama.cpp health check; the details column shows the server's reported `status` (`Ok`, `Loading model`, …, capitalized) when present.
- **`/props`** — the closest thing llama.cpp exposes over HTTP to *how the server was started*: the context size (`--ctx-size`), the default max response length (`--n-predict`), the parallel slot count (`--parallel`), the sampling defaults (`--temp`, `--top-k`, `--top-p`), the loaded model's path and tokenizer boundary tokens, the build version, and whether a chat template is configured. The details column reads back whichever of these the server's response actually included (the schema is not identical across llama.cpp versions, so fields it omits are simply left out rather than reported as an error). llama.cpp does not expose hardware-only startup flags — thread count, GPU layer count, batch size — through any HTTP endpoint, so `/information` cannot report those; they only ever appear in the server's own startup log.
- **`/slots`** and **`/metrics`** — llama.cpp diagnostics endpoints that a server operator can disable at startup; a red dot here usually means the corresponding llama.cpp flag (`--no-slots`, `--metrics`) was not passed, not that something is broken. Their JSON body isn't worth summarizing field by field, so a reachable `/slots` simply reads `Ok` and a reachable `/metrics` reads `reachable`; either one, when unavailable, reads a flat `Not available` regardless of the underlying reason (disabled, not implemented, unreachable, …).

A llama.cpp-native endpoint outside `/slots`/`/metrics` that is unreachable (connection refused, timeout) or answers with an unexpected status is reported the same way as one the server never implemented — `/information` only distinguishes "the server told us it's disabled" (`HTTP 501`) from "the server doesn't know this path" (`HTTP 404`) from any other response, shown as its raw HTTP status.

### Examples

```text
/information
```

Natural-language forms:

```text
information
show information
server information
llm information
```

\newpage

## /statistics

Reports persistent activity: how active this project has been over time. Where `/usage` reports the current session's totals and disappears when orangu exits, `/statistics` reads back a small log that survives restarts, and folds in the workspace's own `git log` — so a repository with commit history has something to report from the very first run, not just after you've used orangu in it for a while.

Every completed turn — success, cancellation, or failure — appends one record to `~/.orangu/workspace/<hash>/stats/activity.json` (the same shared cache root the knowledge graph and semantic search index use, keyed by a hash of the workspace's canonical path — see `/search`): the day it happened, the session it belongs to, and that turn's token count, LLM time, and tool time. Bare `/statistics` reads the current workspace's log and merges in its `git log`; `/statistics total` merges every workspace's turn log into one aggregate instead (commit history is left out of `total`, since it would mean reading unrelated repositories that don't share one `git log`).

The report has two parts:

- **Total** — all-time figures, split into a **Repository Activity** table (total commits, days active, and your current/longest streak of consecutive active days) and a **Token Usage** table (sessions, turns, tokens, LLM and tool time), then a **Heatmap** and an **Authors** breakdown (from `git log`, most commits first) of each author's commit count and lines added/removed — similar to GitHub's contributors graph.
- One section per calendar **year** with any activity, newest first: that year's **Yearly total** (tokens and commits), then a **Monthly** breakdown (also newest first).

The heatmap is a GitHub-style grid: the last 20 weeks, one Monday-to-Sunday column per week and one row per weekday (Monday on top, each row labelled with its initial), shaded from blank (no activity) through four increasing quartiles of your busiest recorded day. A day with orangu usage shades by its token count; a day with only a commit (no orangu usage that day) still gets the lightest tint rather than staying blank:

```text
Total

Repository Activity
Commits          : 512
Days active      : 340
Current streak   : 4 days
Longest streak   : 9 days

Token Usage
Sessions         : 42
Turns            : 358
Tokens           : 1284091
LLM time         : 6h 12m 0s
Tool time        : 1h 47m 0s

Heatmap

M ░░▒▒▓▓██████████████████████████████████░░
T ▒▒▓▓██████████████████████████████████░░░░
W   ░░▒▒▓▓████████████████████████████████░░
T ░░▒▒▓▓██████████████████████████████████░░
F ▒▒▓▓██████████████████████████████████░░░░
S   ░░▒▒▓▓████████████████████████████████░░
S ░░▒▒▓▓██████████████████████████████████░░

Authors
Jesper Pedersen           310 commits    +45210   -12043
Tejas Bhati               202 commits    +28114    -9021

2026

Yearly total
Tokens           : 442080
Commits          : 302

Monthly
2026-02                 : 232036 tokens, 182 commits
2026-01                 : 210044 tokens, 120 commits
```

A current streak counts today if you've already used orangu today, or ends at yesterday if you haven't yet — so a streak in progress isn't reported as broken partway through the day. With no activity recorded yet (a fresh install outside a Git repository, or a corrupted log), `/statistics` reports that plainly rather than an error or a blank table.

See [`/export statistics`](#export) to save the same report — plus a two-layer table of contents, per-year and per-month heatmaps, Authors breakdowns, and bar charts, and a per-author appendix — to a PDF.

### Examples

```text
/statistics
/statistics total
```

Natural-language forms:

```text
statistics
show statistics
activity
statistics total
show statistics total
activity total
```

\newpage

## /schedule

Runs commands on a schedule, cron style. Jobs live in `~/.orangu/schedule`, one per line in classic crontab form — five time fields, then the command to run:

```text
# hourly pull request report
0 * * * * /export pr
30 6 * * 1-5 /statistics
```

The command is anything you could type at the prompt — a slash command, a natural-language form, or a free-form prompt for the model. While orangu is running, a job whose minute arrives is queued exactly like a command typed while a request is in flight: it waits its turn behind whatever is running, echoes into the output window as `> /export pr`, and executes in the **active workspace tab**. If orangu was busy (or idle in the background) when the minute passed, the job still runs as soon as the loop comes back around — every minute boundary since the last check is considered, so nothing is silently skipped. Minutes that passed before orangu started never fire, and orangu must be running for anything to run at all: this is a scheduler inside the client, not a system daemon.

`&&` chains commands, shell style — each part runs after the one before it, and a part that fails drops the rest of its chain (the output window notes how many follow-ups were skipped):

```text
0 6 * * * auto review immediate && export auto review
```

Scheduled commands run **unattended**. `/auto_review` normally opens interactive phases — a pre-start screen waiting for Alt+s, and the finished report kept on screen until Alt+x — but when launched by the scheduler it starts at once and returns as soon as the run completes, so a chained `export auto review` can pick up the report with nobody at the keyboard. The report still lands in the output window (and the clipboard) exactly as if the run had been watched.

The five fields are minute (0-59), hour (0-23), day of month (1-31), month (1-12), and day of week (0-7, both 0 and 7 meaning Sunday), each supporting `*`, lists (`1,15`), ranges (`1-5`), and steps (`*/10`, `8-18/2`). Fields are numeric — no `JAN`/`MON` names. When both day-of-month and day-of-week are restricted the job runs when either matches, like classic cron. **Times are UTC**, since orangu carries no timezone database. Blank lines and `#` comments are skipped. The file is re-read every minute, so edits apply without a restart.

Bare `/schedule` lists the jobs with the next time each will run, and points out any lines that didn't parse instead of silently ignoring them:

```text
Scheduled jobs (~/.orangu/schedule, times UTC):
/export pr                               next: 2026-07-09 14:00
/statistics                              next: 2026-07-10 06:30
```

With no schedule file (or an empty one) it says how to create one.

### Examples

```text
/schedule
```

Natural-language forms:

```text
schedule
show schedule
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

Builds the workspace project, detecting the toolchain from the workspace root. Takes an optional profile, `debug` or `release` (the default is `release`), and an optional **build target** — a single word, in either order (`/build debug docs` and `/build docs debug` mean the same). Giving two profiles or two targets is rejected rather than one being silently dropped.

Each step is reported individually and the pipeline stops on the first failure. The profile is mapped onto each toolchain's own notion of a build profile, and the target onto its own notion of a target:

- **Rust** (`Cargo.toml`) — runs `cargo fmt`, `cargo clippy`, then `cargo build` and `cargo test`, adding `--release` for the release profile (omitted for debug) and `--jobs <compile_workers>` when it is nonzero. A target names one **binary**: both the build and its tests are scoped to it with `--bin <target>`, leaving the rest of the workspace untouched.
- **C/C++, CMake** (`CMakeLists.txt`) — runs `clang-format.sh` (if present), then configures and builds in a single reused `build/` directory, then `make`, adding `-j <compile_workers>` when it is nonzero. The first build runs `cmake .. -DCMAKE_BUILD_TYPE=Debug` (or `Release`); later builds skip straight to `make` when the directory is already configured for the requested profile, and reconfigure with the new `-DCMAKE_BUILD_TYPE` (which `cmake` applies in place) when it isn't — so switching between `/build debug` and `/build release` reconfigures the same directory rather than building each profile separately. A target is a rule in the generated Makefile (`make <target>` — CMake emits one per `add_executable`/`add_library` target, plus `install` and friends).
- **C/C++, Autotools** (`configure`, checked when there is no `CMakeLists.txt`; takes priority over Meson when a project has both, e.g. PostgreSQL mid-migration) — runs `clang-format.sh` (if present), then builds in place, like a plain `./configure && make`. If the workspace root already has a `config.status`/`GNUmakefile` (an existing configuration, from a previous `/build` or a manual run), it runs `make distclean` first — autotools has no separate build-type flag, and an out-of-tree VPATH build does not mix safely with an in-tree one, so any previous configuration is wiped rather than built alongside. Then it runs `sh ./configure CFLAGS=... CXXFLAGS=...` (`-g -O0` for debug, `-O2` for release) and `make`, adding `-j <compile_workers>` when it is nonzero. A target is a **Makefile rule** (`make <target>`).
- **C/C++, Meson** (`meson.build`, checked when there is no `CMakeLists.txt` or `configure`) — runs `clang-format.sh` (if present), cleans up a stale in-tree Autotools configuration the same way as above if one is found, then builds in a single reused `build/` directory (Meson refuses to build in place). The first build runs `meson setup build --buildtype=debug|release`; later builds run `meson configure build --buildtype=...` instead — a cheap no-op if the profile is unchanged, and a correct incremental rebuild if it isn't — then `meson compile -C build`, adding `-j <compile_workers>` when it is nonzero. A target is one of Meson's own compile targets (`meson compile <target>`).
- **Java** (`pom.xml`) — installs frontend dependencies with `npm ci` when outdated, runs `npm run fix` and `npm run check` for the frontend (if `src/frontend/` exists), then `mvn package` for debug, or `mvn -P release package` for release (this assumes the project defines a Maven profile named `release` in its `pom.xml`; Maven has no built-in debug/release axis). A target is a Maven lifecycle phase or goal, replacing the default `package` (`/build verify` runs `mvn verify`).
- **Python** (`pyproject.toml`, `setup.py`, or `setup.cfg`, checked in that order) — runs `pip install -e .`. There is no separate debug/release artifact, so the profile is accepted but has no effect — and no target concept either, so a requested target is rejected with an error rather than silently ignored.
- **Go** (`go.mod`, checked when none of the above are found) — runs `go build ./...`, adding `-p <compile_workers>` when it is nonzero. Go has no separate debug/release artifact either; debug instead adds `-gcflags="all=-N -l"`, disabling optimizations and inlining (mirroring the C backends' `-O0`) so a debugger such as delve can step through unoptimized code. Release passes no extra flags. A target is a package path (e.g. `./cmd/server`), replacing the default whole-module `./...`.
- **Plain Makefile** (`GNUmakefile`, `makefile`, or `Makefile`, in GNU make's own lookup order — the **last resort**, only when none of the managed build systems above claim the workspace, since CMake, Autotools, and Meson all generate or ship their own Makefiles) — runs `clang-format.sh` (if present), then `make`, adding `-j <compile_workers>` when it is nonzero. A target is a Makefile rule (`/build docs` runs `make docs`). Plain make has no universal debug/release convention, so the profile is accepted but not mapped to anything.

The target **Tab-completes** (and shows the inline ghost hint): after `/build `, Tab offers `debug`, `release`, and the workspace's discovered targets — cargo binary names (from `[[bin]]` entries and `src/bin/`) for a Cargo project, Makefile rule names for a plain-Makefile one. Discovery is best-effort: an unlisted target can still be typed by hand.

`<compile_workers>` is the `[orangu].compile_workers` value. It defaults to `0`, which means unused: no job flag is passed at all, and each toolchain falls back to its own default (Cargo's own automatic parallelism, a bare serial `make`, `meson compile`'s ninja default, `go build`'s own `-p` default). Setting it above `0` passes an explicit job count to every backend above — Rust, the C/C++ backends, plain make, and Go. Java and Python are unaffected: Maven and `pip` have no directly equivalent per-compile job flag.

### Examples

```text
/build
/build debug
/build release
/build docs
/build release orangu-gguf
```

Natural-language forms:

```text
build
build project
run build
build debug
debug build
build release
release build
```

\newpage

## /shell

Runs a shell command in the workspace, streaming its output to the output window line by line as it is produced — long-running commands (a test suite, a script that tails a log) show output live rather than all at once when they finish.

```text
/shell <command>
```

The command line runs through `bash -lc`, exactly like the model-facing `run_shell_command` tool (see the Tools chapter), so it supports everything a login shell does: pipes, redirects, globs, and any executable on `$PATH`, not a fixed allow-list. It runs with the workspace as its current directory. `/shell` requires a command — a bare `/shell` reports usage instead of doing nothing quietly.

The command exits non-zero the same way it would at a real terminal — `/shell` reports the failure but the output already streamed stays in the output window.

Tab completion after `/shell ` completes the token being typed against files in the workspace, one path segment at a time, exactly like a real shell: `/shell ./te` offers `./test/`, and once inside that directory `/shell ./test/c` offers `./test/check.sh`. Only the last word of the command line completes this way — the program name and any earlier arguments are left alone. The inline ghost hint previews the same completion.

### Examples

```text
/shell ls -la
/shell cargo test
/shell ./test/check.sh
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

Enter it with the `/auto_review` command, or the natural-language form `auto review`. Give it a file — `/auto_review <file>` — to review just that one file instead of the whole branch (see *Reviewing a single file* below), or the `all` keyword — `/auto_review all` — to review every file in the project instead (see *Reviewing every file* below). The view opens in a **pre-start phase** that waits for you to begin the run (see *Starting the run* below); add the `immediate` keyword — `/auto_review immediate` (or `/auto_review <file> immediate`, or `/auto_review all immediate`) — to skip it and start at once. Add the `deep` keyword — `/auto_review deep` — to start every file in **Deep** mode instead of Normal, the same Deep the Alt+m pre-start cycle offers per file (see *Normal, Deep, and Ignore* below), applied to the whole launch at once. The three keywords and a file argument combine freely, in any order — the longest accepted form is `/auto_review deep all immediate` (or, natural-language, `auto review deep all immediate`).

### What is reviewed

The same change set as `/review`: everything the current branch adds on top of the default branch — committed changes on the branch plus local uncommitted changes — measured against the merge base with the default branch (`origin/main`, `origin/master`, `main`, then `master`). If there is nothing to review, `/auto_review` reports that and does not open the view.

Like `/review`, the branch must be **up to date (rebased)** against the default branch before the auto review starts: when the branch is behind main/master, the command refuses with `The branch is N commits behind <base>; run /rebase before reviewing.` — reviewing against stale code would waste the run and could approve changes that conflict with the newer base.

### Reviewing a single file

`/auto_review <file>` reviews one file rather than the whole branch. The view, the categories, and the report are exactly the same as a whole-branch run — there is just one file in the checklist. What gets reviewed depends on the branch you are on:

- **On `main`/`master`** the whole file is reviewed — a full read of its current content, every line in scope — not a diff. This is the way to have the model review a file that is not part of any in-progress change.
- **On any other branch** only the file's **changes** against the default branch are reviewed, exactly as in a whole-branch run (the same rebased-branch guard applies).

The natural-language form takes a file too: `auto review <file>` is equivalent to `/auto_review <file>`. The file argument is resolved by **Tab completion** in either form, and it completes on the file's **name, not its location** — typing `t` and pressing Tab offers `src/tui.rs`. The `immediate`, `all`, and `deep` keywords Tab-complete (and ghost) the same way — typing `imm` offers `immediate`, typing `al` offers `all`, and typing `de` offers `deep` — and any of them may be combined with a file, and with each other, in any order. The candidate list matches what will be reviewed: on `main`/`master` it is every tracked file (files ignored by `.gitignore` are excluded); on any other branch it is only the files that differ from the default branch. Selecting a candidate fills in its full repository-relative path; a hand-typed bare name (e.g. `tui.rs`) is resolved too. On a branch, a file with no changes against the default branch is refused with `'<file>' has no changes against <base>.`

### Reviewing every file

`/auto_review all` reviews every file `git` tracks in the project instead of the branch diff or a single file — the view, the categories, and the report are exactly the same, just with every tracked file in the checklist. Each file gets the same treatment as a single-file review on `main`/`master`: a full read of its current content, every line in scope, regardless of which branch you are on — `all` reviews what is actually on disk, not a diff, so a branch behind the default branch is reviewed anyway (the rebased-branch guard does not apply here). Untracked files and anything excluded by `.gitignore` are left out — the file list comes straight from `git ls-files`. If `all` is combined with a file argument, `all` wins and the file is ignored; combine it with `immediate` — `/auto_review all immediate` — to skip the pre-start phase and begin reviewing the whole project at once. The natural-language form takes it too: `auto review all`.

### Layout

The view opens with the tool header row at the top and under it the two panes, exactly like `/review`; the **status area** is the first row of the left pane, so the file checklist on the right keeps its full height. The input window stays empty — auto review takes no typed request.

- **Header row** — the tool title (`Auto review: <branch>`) and the key help, with the `Files (n)` header of the right pane.
- **Status area** — a highlighted bar across the left pane, just below the header, showing what is being worked on: the file (with its position in the file list), the category, the overall progress across all of the run's requests, the total time spent on the run so far, and the estimated time still to go, e.g. `File: src/main.rs (2/5)  Category: Security  Progress: 8/26 (30%)  Time: 1m12s  Estimated: 2m48s`. Both times use the same shortest form as the Thinking/Working timers (`5s`, `1m5s`, `1h2m3s`). The **estimate** is the average time per completed request so far extrapolated over the requests still to run; it is recomputed after each request finishes and counts down between them. It appears once the first request completes and drops away when the run ends — after the run the bar shows `Done` (or `Cancelled`) with the time frozen at the run's total.
- **Left pane** — below the status area, the **report**, rendered from Markdown with the syntax markers consumed: one bold heading per category (Overall, Code, Security, Memory, Performance, Test Suite, Documentation), each listing the findings collected so far as a bullet list with the file names in bold, ending with the **Conclusion**. A category that has produced nothing yet shows `(Press Alt+s)` before the run starts, `(pending)` while it is in progress, and `No issues found` once it is done. The pane scrolls and pans independently.
- **Right pane** — the checklist of changed files, one per row, as in `/review`. Before the run starts, each file's box shows its mode as a dot — **white** for Normal, **purple** for Deep, **blue** for Ignore — so all three read the same way at a glance; see *Normal, Deep, and Ignore* below. Once the run starts, the file currently being reviewed is highlighted and its status box blinks a white dot regardless of mode, until its review resolves to green or red. Once the run ends (or the whole-change pass starts) the highlight is cleared — nothing is being reviewed anymore; `Alt+j`/`Alt+k` bring it back to move through the list while browsing.

The header row offers different keys in each phase: **before the run starts** the pre-start keys (`Alt+s Start  Alt+j/k Switch file  Alt+m Mode  Alt+e Diff  Esc Esc Cancel  Alt+x Exit`); **while the run is in progress** the run keys (`Esc Esc Cancel  Alt+x Exit`); once the run has **ended** the browse keys (`Alt+j/k Switch file  Alt+a Approve  Alt+r Reject  Alt+e Open  ↑/↓ Item  Enter Diff  PgUp/PgDn Category  - Remove  Alt+x Exit`).

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

### Starting the run

Unless you passed `immediate`, the view opens in a **pre-start phase**: the panes are drawn but no requests are sent yet, every category reads `(Press Alt+s)`, and the status area shows how many files will be reviewed. Press **`Alt+s`** to begin; from then on the run behaves exactly as described below.

Before starting, you can prepare the run:

- **`Alt+j`/`Alt+k`** move the highlight through the file list (like `/review`).
- **`Alt+m`** (Mode) cycles the highlighted file through three modes, in order — **Normal → Deep → Ignore → Normal** — described in full below.
- **`Alt+e`** (Diff) opens the highlighted file's diff in `$EDITOR`, like `/diff` for one file — orangu writes the file's unified diff to a temporary file and opens it in a separate window, so you can read what changed before deciding which mode to give it.

`Alt+x` (or a double `Esc`) leaves without reviewing. The mode can only be changed in this phase; once the run has started, `Alt+s`, `Alt+j`/`Alt+k`, and `Alt+m` drop from the header.

### Normal, Deep, and Ignore

Every file starts in **Normal** — `/auto_review`'s default review, described in full in *How the review runs* below — unless the `deep` keyword was given at launch (`/auto_review deep`, `/auto_review deep all`, …), in which case every file starts in Deep instead. Either way, `Alt+m` advances the highlighted file through the cycle **Normal → Deep → Ignore → Normal** during the pre-start phase, so individual files can still be adjusted (or excluded) before the run begins. Only **Ignore** changes *what* gets reviewed; **Deep** reviews the file through the same categories as Normal, just with three additional passes:

- **No diff compression.** Normal respects the configured diff compression (`[orangu].compression`, on by default), which can shorten a very large diff before it reaches the model. Deep always sends the **full, untruncated diff** — at the cost of more tokens on large changes.
- **Cross-file graph context.** Deep folds in the callers and callees of the file's changed symbols that live in *other* files — pulled from the workspace's background knowledge-graph scan, the same one `/graph` and the `graph_lookup` tool use — right after the diff in each category prompt, so the model can see how a changed function is used elsewhere, not just the file in front of it. This section is only present when the graph actually has cross-file neighbours recorded for that file; see *Knowledge graph status* below for why it can come up empty.
- **Verify pass on rejects.** If a Deep file ends with at least one rejected category, one more request re-lists every finding recorded against it and asks the model to reconsider each one now that every category has had its say. A finding it withdraws is dropped from the report; if that clears every finding, the file is approved after all instead of staying rejected on a false positive.

**Ignore** shows a **blue dot** and is **skipped from the run entirely** — it gets no requests, and when the run starts it is **automatically approved**: its dot turns **green** and it counts as approved toward the verdict, so it never appears in the Conclusion's rejected / not-reviewed listing. This lets you exclude files you do not want reviewed at all (vendored code, generated output, an unrelated change) before the run begins.

A Deep file's dot is **purple** before the run starts; once reviewed it shows the normal green/red status like any other file — Deep does not change how a file's final status is decided, only how thoroughly it gets there.

#### Knowledge graph status

Deep's cross-file context depends on orangu's background knowledge-graph scan of the workspace. While a Deep review is running, the status bar shows a **`Graph: ●`** indicator alongside `Pending: N`: white while the scan is still building, green once it has finished, red if the scan task itself failed. (`/information` reports the same thing as a `Graph` row worded `Building`, `Complete`, or `None`.) A file reviewed before the graph has finished — or before it has picked up very recent edits, since the graph only rescans changed files on its own schedule rather than synchronously before a review — simply gets no cross-file context for that request; this is not an error.

### How the review runs

The files are reviewed one at a time, in diff order. Each file's name and extension enable the categories that are scanned, so the run spends its requests only where a review can act:

- A file on the **skip list** is approved at once, with **no requests**. These are files whose diff a review cannot act on: generated dependency **lock files** — by extension `.lock` (`Cargo.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, `composer.lock`, `flake.lock`, …) and by name `package-lock.json`, `npm-shrinkwrap.json`, `pnpm-lock.yaml`, `go.sum`, `go.work.sum` — and **binary assets**: images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.ico`, `.svg`, `.webp`, `.tiff`, `.tif`), fonts (`.otf`, `.ttf`, `.woff`, `.woff2`, `.eot`), and `.pdf`, `.p12`, `.jks`, `.keystore`.
- A file detected as **documentation** (`.md`, `.markdown`, `.mkd`, `.mdown`, `.mdx`, `.rst`, `.adoc`, `.asciidoc`, `.txt`, `.text`, `.org`, `.tex`, `.texi`, `.texinfo`, `.pod`, `.rdoc`) skips the code-related checks and is reviewed only for the **Documentation** category — a single request per file.
- Every other file — the fallback — is scanned for all six per-file categories: Code, Security, Memory, Performance, Test Suite, then Documentation. Build and metadata files take this full review too: extensionless ones (`Makefile`, `Dockerfile`) fall through to it automatically, and the few that carry a documentation-looking extension (`CMakeLists.txt`, `requirements.txt`) are pulled back into it by name so a `.txt` extension does not demote them to documentation only.

For each enabled category, one focused request is sent to the LLM asking for a verdict plus findings for that category only, with the file's diff attached. The review is explicitly scoped to **the changes made** — the added, removed, and modified lines, and how they fit into the surrounding context — not to pre-existing content the change does not touch, and each category is capped at five short findings. Each finding is prefixed with its location as `<file>:<line>:` — the affected **line number**, or range as `<start>-<end>`, in the new version of the file (the right side of the diff) — so the report points at where each issue lives, e.g. `src/main.rs:42: unwrap may panic`. The status area names the file and category being worked on and counts the overall progress — the total reflects only the enabled categories — while the status bar shows the usual thinking indicator.

As each category review arrives, its findings are appended to the matching section in the left pane, each prefixed with its location (`file:line`) in bold — so the report fills in category by category. When all of a file's categories have run, the file is automatically marked in the right pane — a **green dot** when every category passed, a **red dot** when any category rejected. Without an explicit verdict, a category passes only when its review found nothing. If a request fails — or its response carries neither a verdict nor findings (for example, truncated by the response cap) — the file keeps its white (unreviewed) box and the problem is noted under Overall; such a response never passes silently as a clean review.

After the last file, a final pass reviews the change as a whole: the per-file verdicts and findings are summarized by the model into a few bullet points — how the changes fit together, readiness, risk, and common themes — under **Overall**.

The report ends with the **Conclusion**, derived from the file statuses rather than from the model: the verdict — `orangu approves this patch` when every file is approved, or `orangu rejects this patch` when any file was rejected or not reviewed — stands alone in bold rather than as a list item; the affected files then follow as a bullet list in bold, grouped by their status (`Rejected: **file**`, `Not reviewed: **file**`). A closing **`Generated by: orangu <version> (<model>)`** line credits the orangu version and the reviewing model, e.g. `Generated by: **orangu 0.7.0** (gemma)`.

Each request runs in its own scratch exchange, **without tool definitions** and with a **capped response length** (`[orangu].review_max_tokens`, default `512`; `0` disables the cap) — a review can neither wander off into tool calls nor generate unbounded output, which keeps single requests fast and bounded even on slow local models. For deeper reviews with a thinking model, raise the cap (e.g. `2048`) so the thinking tokens do not eat the answer; the *Response-token caps* part of the Configuration chapter covers the trade-offs in depth. Every category prompt carries the whole file plus its diff (not just the diff), and — on a llama.cpp server — a file's category requests are all pinned to the same `id_slot`, so its file and diff are prompt-processed once and reused from the server's KV cache across every category instead of being recomputed each time. The reviews are independent of each other and nothing is added to your chat session.

While the model works, the status bar shows `Thinking (...)` until the first token arrives and then the live generation rate (`Working @ X.Y t/s (...)` on llama.cpp), so a stalled server and a slowly generating model are easy to tell apart.

A full-branch auto review can take a while, so when **`feedback` is on** (see the Configuration chapter) orangu surfaces its progress outside the window too: while the run is in progress the **terminal title** reads `orangu ●` (`orangu ◆` while the file currently being reviewed is Deep — a window title cannot carry color, so Deep is shown by shape instead of the purple used in the pane) with the dot blinking once a second, so a backgrounded or unfocused terminal still shows that a review is running. When the run **finishes**, orangu rings the **terminal bell** — the standard desktop notification sound (or a visual flash, depending on your terminal) — and drops the title back to a plain `orangu`. A run that is cancelled (`Esc Esc`) or exited (`Alt+x`) before it finishes does not ring. With `feedback` off, neither the title nor the bell is touched.

### Browsing and overriding the report

Once the run has ended (done or cancelled), the report stays on screen and you can override the model's verdicts file by file. `Alt+j`/`Alt+k` move the highlight through the file list — from no highlight, `Alt+j` starts at the first file and `Alt+k` at the last.

You can also work through the report **item by item** in the left pane. `Up`/`Down` move a highlight between the individual report items — the findings and the Conclusion entries, never the category headings — scrolling the pane as needed to keep the highlighted item in view (from no highlight, `Down` starts at the first item and `Up` at the last). Moving the highlight also points the file list on the right at the item's file, so `Alt+a`/`Alt+r` act on it.

To skip across a long report **category by category**, use `PageDown`/`PageUp`: they jump the highlight straight to the first item of the next or previous category that **has findings**, scrolling that category's heading to the top so the whole section comes into view. Empty categories (those reading `No issues found`) are skipped, and the Conclusion entries count as the final category. From no highlight, `PageDown` lands on the first category with findings and `PageUp` on the last; once the highlight is already in the last (or first) such category the key is a no-op, so it never jumps backward past the report. Use `Up`/`Down` to walk the individual findings within a category.

- **`-` — remove the highlighted item.** A **finding** is dropped from its category; if that was the **last** finding recorded against its file, the file is approved (its dot turns green) and it drops out of the Conclusion. Removing a **Conclusion** item approves the whole file it stands for, clearing every finding recorded against it across the report — the same as approving that file. So you can approve the patch outright by removing all the flagged items: once nothing is left, every file is approved and the verdict reads `orangu approves this patch`.
- **`Alt+a` — approve the highlighted file.** Its dot turns green and **every finding recorded against it is removed from the report** — the model's findings and your own rejection comments alike — so an approved file no longer appears in any category, in the exit report, or on the clipboard. The Conclusion follows the file statuses, so approving the last rejected file flips the verdict to `orangu approves this patch`.
- **`Alt+r` — reject the highlighted file.** A reject window opens over the panes with a **category selector** (Overall, Code, Security, Memory, Performance, Test Suite, Documentation) and a **multi-line Markdown comment editor**. `Tab` moves the focus between the two; in the selector `Up`/`Down` pick the category (`Enter` moves on to the editor), and in the editor `Enter` inserts a newline while `Up`/`Down`, `Home`/`End`, and the usual editing keys move and edit. Press `Alt+Enter` to save — the file's dot turns red and the comment is appended to the chosen category, prefixed with the file path in bold — or `Esc` to discard the window. Saving with an empty comment still rejects the file without adding a finding. `Alt+r` can be repeated on the same file; each saved comment is kept.
- **`Alt+e` — open the highlighted file** in your `$EDITOR`, exactly like `Alt+e` in `/review`: terminal editors open in a new window and GUI editors open their own, leaving the report on screen.
- **`Enter` — show the selected file's diff.** With the input window empty, pressing `Enter` opens a **diff popup** over the panes showing the colorized `/diff` of the highlighted file. When that file is **not part of the diff** (for example a whole-file `/auto_review <file>` review of an unchanged file), it falls back to the `/show_file` tool, showing the file's syntax-highlighted code around the highlighted finding's line — **3 lines before and after**. `Up`/`Down` (and `PageUp`/`PageDown`) scroll it, `Alt+Left`/`Alt+Right` pan it for long lines, and `Esc` closes it. When the input window instead holds an `open <path>` line, `Enter` submits that instead (see below).
- **`/open_file <path>` / `open <path>` + `Enter` — open any project file**, not only the changed ones. Once the run is done the input window at the bottom accepts an open command: type `/open_file <path>` (or `open <path>`), with `Tab` completing every workspace file just like `/open_file` at the main prompt, and press `Enter` to open it in your `$EDITOR`. This works **only after the run has finished** — during the run the input window stays empty. While the input is empty, `Enter` opens the diff popup and `-` still removes the highlighted item; a `-` typed into a path is left for editing.

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
| `Enter` | Show the selected file's `/diff` (or its `/show_file` code, ±3 lines around the finding, when it is not part of the diff) in a scrollable popup (after the run, while the input window is empty) |
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

When the diff popup is open it is modal:

| Key | Action |
| --- | --- |
| `Up` / `Down` | Scroll the diff |
| `PageUp` / `PageDown` | Scroll the diff by a page |
| `Alt+Left` / `Alt+Right` | Pan the diff horizontally for long lines |
| `Esc` | Close the popup, returning to the report |

### Examples

```text
/auto_review
```

Review a single file (Tab-completes on the file name):

```text
/auto_review src/tui.rs
```

Review every file in the project, starting at once:

```text
/auto_review all immediate
```

Natural-language form (a whole-branch review, a single file, or every file):

```text
auto review
auto review src/tui.rs
auto review all
```

\newpage

## /duplicates

`/duplicates` scans the workspace for **duplicated code** across many languages and reports the function pairs that are structurally similar, so you can review them and decide whether they should be unified. It is handled locally and never sent to the model, and needs no Git repository.

Run it with the `/duplicates` command, or the natural-language forms `find duplicates` or `find duplicate code`.

### Supported languages

A file is analysed when its extension matches one of the built-in languages:

| Language | Extensions | Language | Extensions |
|---|---|---|---|
| Rust | `.rs` | Scala | `.scala`, `.sc` |
| C | `.c`, `.h` | OCaml | `.ml`, `.mli` |
| C++ | `.cc`, `.cpp`, `.cxx`, `.c++`, `.hpp`, `.hh`, `.hxx` | Haskell | `.hs` |
| C# | `.cs` | Julia | `.jl` |
| Go | `.go` | Lua | `.lua` |
| Java | `.java` | R | `.r` |
| Python | `.py`, `.pyi` | Zig | `.zig` |
| JavaScript | `.js`, `.mjs`, `.cjs`, `.jsx` | Swift | `.swift` |
| TypeScript | `.ts`, `.mts`, `.cts` | Dart | `.dart` |
| TSX | `.tsx` | Erlang | `.erl`, `.hrl` |
| Ruby | `.rb` | PHP | `.php` |
| Bash | `.sh`, `.bash` | | |

Functions are only ever compared **within the same language**, never across two — even where two languages share a file family (a `.ts` is never compared against a `.tsx`, nor a `.c` against a `.cpp`). Each language is a self-contained entry in orangu's grammar registry, so support for a new one is a small, isolated addition.

### How it works

Every matching file in the workspace is walked (honouring `.gitignore`) and parsed into an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) with [tree-sitter](https://tree-sitter.github.io/tree-sitter/). Each function or method definition — including nested ones — is reduced to the multiset of **bigrams** (adjacent pairs) of its AST node kinds, visited in pre-order. Because only the node *kinds* are compared, two functions that differ only in their names, variables, or literal values still match: the comparison is about *shape*, not text.

Every pair of functions **in the same language** is then scored with the **Sørensen–Dice coefficient** over those bigram multisets:

```text
similarity = 2 × (shared bigrams) / (bigrams in A + bigrams in B)
```

The result runs from 0% (nothing in common) to 100% (identical structure). Functions in different languages are never compared against each other. Very small functions (fewer than 20 AST nodes) are skipped, since trivial one-liners are structurally interchangeable and would only add noise.

Pairs scoring at or above the **threshold** are reported, sorted most-similar first. The default threshold is **80%**; pass a different one as the argument — a percentage such as `90` or `90%`, or a fraction such as `0.9` (an unrecognised argument falls back to the default).

The report is a **starting point for human review, not a verdict**: a high score says two functions have the same shape, which is a strong hint they may be duplicated — but you should open each pair and confirm before refactoring.

### On a branch: only what the branch adds

When the workspace is a Git repository checked out on a branch **other than the default** (`main`/`master`), `/duplicates` narrows the analysis to the change the branch introduces. It diffs the branch against its merge base with the default branch (`origin/main`, `origin/master`, `main`, then `master`, in that order — the same base `/review` uses), takes the **functions the branch adds or changes** (any function whose lines overlap an added line), and compares **each of those against the whole project**.

So on a branch the report answers *"does my branch duplicate code that already exists?"* — every reported pair has the new or changed function first. On the default branch (or outside a Git repository) the whole project is compared against itself as usual. If the branch **adds nothing** — for example it has been rebased onto the base with no commits of its own and a clean working tree — there is nothing branch-specific to analyse, so the whole-project report runs instead. The summary line and the PDF's first page state which mode ran, the base it compared against, and how many new/changed functions were analysed.

### Output

The output window prints a summary (files scanned, functions analysed, the threshold, and the number of candidate pairs) followed by each pair: its similarity percentage, the two function names, and for each function its `path:start–end` location. To save the same report as a PDF, use `/export duplicates` (see the `/export` tool).

### Examples

Scan with the default 80% threshold:

```text
/duplicates
```

Only report pairs that are at least 90% similar:

```text
/duplicates 90
```

Natural-language forms:

```text
find duplicates
find duplicate code
```

## /export

`/export` writes a buffer to a PDF file in the root of the workspace, so a session's output or a review can be saved and shared outside the terminal.

It takes one optional argument selecting what to export:

- `/export` or `/export console` — the **console output window**: everything currently in the main output window (prompts, command output, and model responses), with the terminal ANSI styling removed and the lines printed verbatim.
- `/export review` — the **review buffer**: the Markdown of the last `/review` (or, if none, the last `/auto_review`) report from this session. If no review has been run yet, the command reports that there is nothing to export.
- `/export auto review` — the **auto-review buffer** specifically: the Markdown of the last `/auto_review` report. If no auto review has been run yet, the command reports that there is nothing to export.
- `/export duplicates` — a **duplicate-code report**: the report from the most recent `/duplicates` run in this tab, rendered to a PDF. The report is cached when `/duplicates` runs, so the export reuses it directly — including that run's threshold (run `/duplicates 0.8` and `/export duplicates` writes an 80% report) — without scanning the workspace a second time. If `/duplicates` has not been run this session, the export scans once at the default 80% threshold and caches the result, so it still works with no prior command.
- `/export pr` — a **pull request report**: every open pull/merge request in the repository, fetched from the forge (`gh`/`glab`) at export time, one page per pull request with as much detail as the forge returns.
- `/export statistics` (or `/export statistics total`) — the **persistent activity history**: the same Total-then-year report `/statistics` (or `/statistics total`) prints in the console — totals, streaks, heatmap, and by-author commit breakdown, then a yearly/monthly breakdown per year — plus a token-usage bar chart, rendered to a PDF.

The argument **Tab-completes** (and shows the inline ghost hint): pressing Tab after `/export` offers `console`, `review`, `auto review`, `duplicates`, `pr`, and `statistics`, and the multi-word `auto review` completes from as little as `a` (so `export a` → `export auto review`). The natural-language `export <target>` form (without the leading slash) completes the same way.

The file is saved in the workspace root as `{repository}-{branch}-console.pdf`, `{repository}-{branch}-review.pdf`, or `{repository}-{branch}-duplicates.pdf`, where `{repository}` is the Git repository (or workspace) directory name and `{branch}` is the current branch (`nobranch` when not on one); both are sanitized for use in a filename, so a branch such as `feature/x` becomes `feature-x`. The `pr` and `statistics` exports are saved as `{repository}-pr.pdf` and `{repository}-statistics.pdf` instead — no branch, since those reports cover the whole repository, not one branch. An existing file with the same name is overwritten. On success the saved path is printed to the output window.

Every page carries a **header band** centered on `{repository}-{branch}` (`{repository}-statistics` for the statistics export, which covers the whole repository rather than one branch) and a **footer band** centered on `orangu {version} ({model})` (the active model), both in white on the orangu brand colour to match the terminal banner; in the footer the word `orangu` links to the project site.

The PDF keeps the Markdown formatting as much as a self-contained file can. Text is set in **Red Hat Text**, embedded into the binary (SIL Open Font License), so no system fonts are needed; if for any reason the embedded font cannot be loaded the export falls back to the closest built-in face, Helvetica. Headings use the orangu brand colour. Long lines wrap to the page width using the font's real glyph metrics — prose on word boundaries, code lines hard at the margin — and the content flows across as many pages as needed.

The **console** export preserves the output window line for line (terminal colours removed).

The **duplicates** export is organized like the review export:

- **Page 1 — summary.** A table of the repository, branch, generation date/time, the similarity threshold the scan used, and the file, function, and candidate-pair counts.
- **Page 2 — table of contents.** Each similarity chapter with the page it starts on; the entries are **clickable links** that jump to their chapter.
- **Page 3 onward — the chapters.** The candidate pairs are grouped by their similarity percentage; each `{n}% similar` chapter starts on its own page and lists its pairs, every pair showing the two function names and each function's location as `path:start–end`. When the repository's `origin` remote is on **GitHub or GitLab**, each location is a **link to that file on the forge with the lines highlighted** (at the current branch, or the default branch when detached); on any other host the location is plain text. (A report with no pairs is the summary page followed by a short note instead.)

The **review** export is organized for reading and sharing:

- **Page 1 — summary.** A table of the repository, branch, and generation date/time, followed by one row per category with its number of entries (findings), and then an overall **Approved** (green) or **Rejected** (red) status banner read from the report's conclusion.
- **Page 2 — table of contents.** Each category with the page it starts on, then a final **Appendix** entry.
- **Page 3 onward — the report.** Each category (`Overall`, `Code`, `Security`, `Memory`, `Performance`, `Test Suite`, `Documentation`, then `Conclusion`) starts on its own page, rendered from the report's Markdown with brand-coloured headings, **bold** and *italic* emphasis, ordered and unordered (including nested) lists, fenced code blocks, block quotes, and tables. So `Overall` opens on page 3.
- **Appendix.** Both the `/review` and `/auto_review` exports add a **source appendix** following the categories on its own page: grouped by category, each finding (or comment) is listed with the **source code around its line** — the `/show_file` view, 3 lines before and after, with line numbers. Only the finding's **recorded line(s)** are **syntax-highlighted and drawn in bold**; the surrounding context lines are shown plain, so the line the finding points at stands out. (For `/review`, the comment's diff position is mapped to its real source line so the appendix matches the file.)

The **pr** export is organized like the review and duplicates exports:

- **Page 1 — pull request status.** A single table: the repository name, generation date/time, the open pull requests broken down by status — **Open** (the total), **Ready** (not a draft), and **Conflicts** (reported as having a merge conflict) — then **Oldest** and **Newest**, each a clickable link to that pull request followed by its creation date. With no open pull requests both read `N/A`; with exactly one, **Oldest** is left empty rather than repeating the same entry as **Newest**.
- **Page 2 — table of contents.** One entry per open pull request, `#N Title`, **clickable links** that jump to their page. Each entry ends with a status icon: a **green checkmark** when the pull request is neither a draft nor conflicting, otherwise a **red "X"**.
- **Page 3 onward — one page per pull request** (more when it changes many files or has a long last comment). The title **links to the pull/merge request's home page on the forge**, followed by a table (spanning the full page width) of author, a **Link** row with the pull request's full URL (also clickable), created/updated dates, the branch, draft status, merge-conflict status, comment count, assignees, reviewers, and labels — whatever the forge returned. The **Draft** and **Conflicts** values are shown in **bold** when they are `Yes`, so an unfinished or blocked pull request catches the eye. Each **reviewer** is shown as their name followed by a status icon rather than the review state spelled out: a **green checkmark** for an approval, a **red "X"** for a change request (or, on GitLab, a still-outstanding review request — its merge-request list does not carry per-reviewer approval state at all), and a **"?"** for anything else that isn't a clear verdict (a comment, a still-pending GitHub review request, or a dismissed review). Below the table, the **changed files**: one line per file, its **full path** followed by its added-line count in **green** and removed-line count in **red** (GitLab's merge-request list carries no diff, so this section is empty there). Finally a **Last comment** table (also full width): a header spanning both columns, then one row with the comment's **author** (left) and its **text** (right, word-wrapped and truncated if very long); a pull request with no comments — or, on GitLab, whose comment bodies the list endpoint does not carry — shows `N/A` in both columns. (A repository with no open pull requests is the status page followed by a short note instead.)

The **statistics** export has the most pages of the six, since most of the work is in the PDF rather than the console:

- **Page 1 — Total.** A **Repository Activity** table (total commits, days active, current streak, longest streak) and a **Token Usage** table (total sessions, turns, tokens, LLM and tool time) — the same figures `/statistics` prints in the console.
- **Table of contents.** Two layers: one top-level entry per section — Activity, Authors, each calendar year, and Author Details — with each year's months nested beneath it, so a specific month ("June, 2026") is one click away. Every entry is a **clickable link** that jumps to its page, in the same style as `/export pr` and `/export review`'s tables of contents; the contents flow across as many pages as the history needs.
- **Activity.** The last 20 weeks of activity as a grid of filled squares, one Monday-to-Sunday column per week and one row per weekday (Monday on top, each row labelled with its initial), shaded from a light tint through the full orangu brand colour by quartile of your busiest recorded day — the PDF counterpart of the console's block-character heatmap. A commit-only day (no orangu usage) still gets the lightest tint rather than staying blank.
- **Authors.** A borderless table, one row per `git log` author, most commits first: the name, then the commit count, lines added (green), and lines removed (red), each right-aligned in its own column so the `+`/`-` figures line up down the page — the same colouring `/export pr` uses for a pull request's changed files. Spans as many pages as it needs. Omitted for `statistics total` and for a workspace with no commit history.
- **One section per calendar year with activity, newest first.** The year's **Yearly Total** table (tokens and commits), its own **Activity** heatmap spanning the whole year (cells shrink to fit all ~53 weeks on the page), a **Monthly Token Usage** bar chart of that year's months, and that year's own **Authors** breakdown.
- **One page per month, newest first,** headed by its name ("July, 2026"): the month's tokens and commits, its own **Activity** heatmap, and its own **Authors** breakdown.
- **Author Details — one page per commit author.** Their total commits and lines added/removed, then a **Yearly Commits** table breaking down their commit count by calendar year. Omitted for `statistics total` and for a workspace with no commit history.

`statistics total` renders the Total page, table of contents, and year/month sections aggregated across every workspace's turn log instead of just the current one (with the Total heading noting it covers "all workspaces"), but without the Authors breakdowns or appendix — `total` has no one workspace's `git log` to read.

### Examples

Export the output window (the default):

```text
/export
/export console
```

Export the last review report (or the auto-review report specifically):

```text
/export review
/export auto review
```

Export a fresh duplicate-code report:

```text
/export duplicates
```

Export a pull request report:

```text
/export pr
```

Export the persistent activity history:

```text
/export statistics
/export statistics total
```

Natural-language forms:

```text
export
export console
export review
export auto review
export duplicates
export pr
export statistics
export statistics total
```

\newpage

## /graph

Generates an interactive, standalone HTML visualization of the workspace's Knowledge Graph.

Orangu incrementally extracts a semantic graph of the codebase in the background using Tree-sitter. This command takes that internal graph and writes it out to `<repository>-<branch>-graph.html` (for example, `orangu-knowledge-graph-graph.html` when in the `orangu` repo on the `knowledge-graph` branch) using the `vis-network` JavaScript library. 

The generated HTML file is fully self-contained (all data and scripts are embedded). You can open it directly in your web browser (e.g. `file:///path/to/project/orangu-main-graph.html`).

Inside the visualization you can:
- **Search**: Use the sidebar to quickly locate specific functions, classes, or types.
- **Inspect**: Click on any node (function or class) to see its full location and a list of all callers (Incoming) and callees (Outgoing).
- **Filter**: Toggle specific files on and off in the legend to unclutter the graph.
- **Jump**: Click on any related node in the info panel to instantly jump to it and see its connections.

Nodes are automatically clustered and color-coded based on the file they belong to, helping you easily visualize architectural boundaries and module dependencies.

### Examples

```text
/graph
```

Natural-language forms:

```text
graph
show graph
generate graph
visualize codebase
```

\newpage

## /search

Semantic code search. Where `/grep` matches text and the Knowledge Graph matches
symbol names and structure, `/search` matches *meaning*: a query like
`where is rate-limiting handled?` can surface a `throttle_requests` function whose
name shares no words with the query.

```text
/search <query>
```

The first time it runs, orangu embeds every extractable symbol in the workspace
and stores the vectors under `~/.orangu/workspace/<hash>/embeddings/`, keyed by a
hash of the workspace path so the cache is shared across sessions without
cluttering the workspace tree — the same per-workspace cache root the Knowledge
Graph uses (at `.../graph/` alongside it). The directory holds `chunks.json` (one
embedded chunk per line), a small `meta.json` (version and per-file hashes), and
a `processed.log` recording each file's path and the time it was embedded. Every
subsequent search re-embeds only the files whose contents changed and drops
chunks for files that no longer exist — the same incremental sha256 approach the
Knowledge Graph cache uses — so the cache is always consistent with the current
workspace and stays cheap to keep current. Everything is computed locally through
the server's OpenAI-compatible `/v1/embeddings` endpoint — nothing leaves the
machine.

That first indexing pass can take a while on a large codebase with a local
embedding model. It runs in two phases, each half of the progress bar. The first
half (0–50%) does all the local work up front: it parses every file into chunks
in parallel — across `compile_workers` threads, or every CPU thread when it is
`0` — logs each to `processed.log`, and writes the full `meta.json` when done, so
the on-disk state lists every file before anything is uploaded. The second half
(50–100%) uploads: it embeds the parsed chunks, keeping several requests in flight
at once so the embedding server — the real bottleneck — stays busy rather than
idling between round-trips (a llama-server started with `-np N` embeds them in
parallel). Each request stays within a conservative token budget — chunks are
grouped so a request comfortably fits under a stock llama.cpp server's default
physical batch size (`-b`/`--batch-size 512`), so `/search` works out of the box
without needing that flag raised. The status bar shows the percentage and an
estimate of the time remaining that counts down (`Working (57%) (4m10s, ~3m
left)`); on completion it reports the total (`Searched 2000 symbols in 3m12s.`).

Vectors are appended to `chunks.json` as each file finishes — the existing
vectors are never rewritten — so persistence stays cheap no matter how large the
index grows, and the work on disk builds up as the pass proceeds, surviving even
a hard kill. The pass can also be stopped with a double-`Esc`, which aborts the
in-flight requests at once; either way the
files embedded so far are kept, and the next `/search` resumes from where it left
off rather than starting over.

Ranking is hybrid. The query is embedded and scored against every chunk by cosine
similarity; the strongest matches are then expanded along the Knowledge Graph's
call edges — a semantic seed followed by structural expansion — so a relevant
function pulls in its callers and callees. Each result is anchored to its
`file:line` for quick navigation, and results expanded from a semantic hit are
marked `(via <symbol>)`.

Semantic search enables itself automatically. At startup orangu probes the
`/v1/embeddings` endpoint of the server that serves the `embeddings` role — a
server with `role = embeddings`, or the default `all` server — and turns `/search`
on when it responds. To dedicate a server to embeddings, give its section
`role = embeddings` (see the Configuration chapter). When no server serves
embeddings, `/search` explains how to enable it and existing retrieval (`/grep`,
the Knowledge Graph) is unaffected.

### Examples

```text
/search where is the retry backoff computed
/search session persistence
```

Natural-language forms:

```text
search for where the config is parsed
search rate limiting
semantic search token budget
```

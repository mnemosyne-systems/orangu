\newpage

# Quick start

This chapter gets **orangu** running against a local OpenAI-compatible server such as **llama.cpp** with the sample configuration in `doc/etc/orangu.conf`.

## Install orangu

The quickest way to install the latest release binary is the one-liner installer.

**Linux / macOS:**

```sh
curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.sh | sh
```

**Windows** (Command Prompt):

```cmd
curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.cmd -o install.cmd && install.cmd
```

**Windows** (PowerShell alternative):

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.cmd -OutFile install.cmd; .\install.cmd
```

The script installs to `~/.local/bin` (Linux/macOS) or `%USERPROFILE%\.local\bin` (Windows) and warns if the directory is not in your `PATH`. See [BUILDING.md](../../BUILDING.md) for instructions on building from source.

## Start llama.cpp

Run `llama-server` with your preferred model, for example:

```sh
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF \
             --port 8100 \
             --ctx-size 65536 \
             -sm layer \
             -t 4 \
             --webui-mcp-proxy \
             --fit on
```

**orangu** expects an OpenAI-compatible endpoint, such as:

```text
http://localhost:8100/v1
```

## Create a configuration

The fastest path is the interactive wizard, which auto-detects a model from the
server and writes `~/.orangu/orangu.conf`:

```sh
orangu --init
```

See [Configuration](20-configuration.md) for the details of the wizard.

Or copy the sample:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Default configuration lookup order is:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

## Run the client

```sh
orangu
```

Or:

```sh
orangu --config ./orangu.conf
```

Then start with:

```text
/help
/server
/disconnect
/reload
/tools
/model
/session <UUID>
/list_files
/open_file README.md
/show_file README.md
/build
/add_file README.md
/auto_review
/amend "[#42] My feature"
/branch main
/branch -b feature/new
/branch -m new-name
/branch -d feature/old
/cherry_pick abc1234
/comment 51 "My comment"
/close -i 51
/get_comments -i 51
/commit "[#42] My feature"
/restore README.md
/diff
/init_repo
/log
/log 5
/merge feature/foo
/move_file old.rs new.rs
/pull 42
/push
/push --force
/rebase
/remove_file README.md
/review
/squash
/status
/usage
/clear
/quit
```

Most of these are thin wrappers around the matching `git`/`gh` commands. Two open full-screen views instead: `/review` walks you through the branch's diff file by file for a manual review, and `/auto_review` has the connected model review the branch's changes by itself — per file and per category (Code, Security, Memory, Performance, Test Suite, Documentation) — lets you override its verdicts afterwards (approve a file, or reject it with your own categorized comments), and copies the resulting report to the clipboard on exit. `/auto_review <file>` (Tab-completes on the file name) reviews a single file — the whole file on main/master, or just its changes on a branch. Both are described in detail in the Core tools chapter.

By default the tools operate on the current directory. Use `--workspace /path/to/project` to point **orangu** at another tree.

The startup flags have short forms: `-c` for `--config`, `-w` for `--workspace`, `-r` for `--resume`, and `-i` for `--init`.

**orangu** automatically resumes an existing session when you return to the same workspace and Git branch. When a previous session is found, the status bar shows:

```text
Resuming session 550e8400-e29b-41d4-a716-446655440000
```

for five seconds or until the first command is run.

On exit, the resume command is printed so you can return to the session from a different branch or machine:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

Sessions that had no LLM interaction on `main`, `master`, or outside a Git repository are deleted automatically on exit. Feature branch sessions are always kept.

Use `/session` to list all sessions and their branches. Use `/session <uuid>` (Tab completion cycles UUIDs and workspace paths) to switch to a specific session; passing a workspace switches straight to it when it matches exactly one session, otherwise it lists the matches. Passing a directory path that no session uses yet opens it as a new workspace — Tab falls back to filesystem completion (with `~` expansion) so you can navigate there, e.g. `/session ~/Po<Tab>/pga<Tab>/of<Tab>`.

Lines whose first non-whitespace character is `#` stay local and are not sent to the model. Lines whose first non-whitespace character is `\` are ignored.

\newpage

# Getting Started

This chapter gets **orangu** running against a local **orangu-server**, from nothing installed to a working prompt. After installing, it is five commands in this order:

```sh
orangu-server -i
orangu-server download unsloth/gemma-4-E2B-it-GGUF
orangu-server --all unsloth/gemma-4-E2B-it-GGUF
orangu -i
orangu
```

The order matters: steps 2 and 3 both read the models directory from the configuration step 1 writes, and step 4 asks step 3's running server which model it is serving.

## Install orangu

The quickest way to install the latest release is the one-liner installer. It installs the whole stack — `orangu`, `orangu-coordinator`, `orangu-server`, and the benchmarking tool `orangu-bench`.

**Linux / macOS:**

```sh
curl -fsSL https://mnemosyne-systems.github.io/orangu/install.sh | sh
```

**Windows** (Command Prompt):

```cmd
curl -fsSL https://mnemosyne-systems.github.io/orangu/install.cmd -o install.cmd && install.cmd
```

**Windows** (PowerShell alternative):

```powershell
Invoke-WebRequest -Uri https://mnemosyne-systems.github.io/orangu/install.cmd -OutFile install.cmd; .\install.cmd
```

The script installs to `~/.local/bin` (Linux/macOS) or `%USERPROFILE%\.local\bin` (Windows) and warns if the directory is not in your `PATH`. See [BUILDING.md](../../BUILDING.md) for instructions on building from source.

## Configure orangu-server

`orangu-server -i` (long form `--init`) writes `~/.orangu/orangu-server.conf`. Run it before anything else: downloading and serving both read the models directory from that file, and with no configuration present they stop with `Missing config file; pass --config or add ./orangu-server.conf or ~/.orangu/orangu-server.conf (see --init)`.

```sh
orangu-server -i
```

Each prompt shows its default in brackets; Enter accepts it.

| Prompt | Default | Notes |
| --- | --- | --- |
| `models` | `~/.cache/huggingface/hub` | Where GGUF files live. Tab-completes paths, and is created if it does not exist. Defaulting to the Hugging Face hub cache means models already fetched by `huggingface-cli` or `llama.cpp` are found without moving anything. |
| `model` | *(none)* | Which model to serve when the command line names none. Nothing is installed yet at this point, so press Enter; the prompt is skipped altogether when exactly one model is already present. |
| `role` | `all` | One of `all`, `code`, `review`, `explorer`, `embedding`. |
| `host` | `all` | `all` for every interface, or a literal address such as `127.0.0.1`. Tab-completes the machine's own interfaces. |
| `port` | `8100` | The HTTP API port. |
| `Add web console` | `Y` | `n` writes no `[web]` section and serves no console. Accepting asks four more: the console's `host` (defaulting to the API's), `port` (`8101`), `reexec` (may it load a different model), and `delete` (may it delete models). |

The wizard prints the file before writing it and asks `Write this configuration? [Y/n]`; anything but Enter/`y`/`yes` aborts with nothing written. Only non-defaults are written, so a minimal run yields:

```ini
[orangu-server]
models = /home/you/.cache/huggingface/hub
host = all
port = 8100

[web]
host = all
port = 8101
```

`-i` always targets `~/.orangu/orangu-server.conf` and ignores `-c`/`--config`. See the *Inference server* chapter for the full configuration reference.

## Download a model

```sh
orangu-server download unsloth/gemma-4-E2B-it-GGUF
```

The argument is a Hugging Face repo, `<user>/<model>[:quant]`. With no `:quant`, `Q4_K_M` is preferred, then `Q8_0`, then the first GGUF file in the repo; append one to pin it, e.g. `unsloth/gemma-4-E2B-it-GGUF:Q8_0`. Every shard of a multi-part model is fetched, plus the matching `mmproj-*` sidecar when the repo has one.

Files land under the `models` directory from step 1 in Hugging Face hub-cache layout. Free space is checked up front, an interrupted transfer resumes where it stopped, and progress is redrawn in place — one line per file plus a total — ending with:

```text
Downloaded to /home/you/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/<commit>/gemma-4-E2B-it-Q4_K_M.gguf
```

Gated or private repos need `HF_TOKEN` set in the environment. `orangu-server list` shows what is installed, `delete` removes a model, and `refresh` re-fetches it at a newer revision; see the *Inference server* chapter.

## Start orangu-server

```sh
orangu-server --all unsloth/gemma-4-E2B-it-GGUF
```

The model is the same spec that was downloaded, and now resolves to the local copy — nothing is fetched a second time. A local `.gguf` path or an `NR`/`MODEL` label from `orangu-server list` works just as well, and omitting the argument prints the installed models and asks which one to serve.

`--all` picks the *role*, not a set of models: it is the general-purpose one, and the default. The alternatives are `--code`, `--review` (suppresses reasoning), `--explorer` (broader, more varied output), and `--embedding` (embeddings only, with the chat and completion endpoints disabled). They are mutually exclusive.

The server binds the host and port from step 1 and serves an OpenAI-compatible endpoint, which is what **orangu** connects to and what step 4 asks for:

```text
http://localhost:8100/v1
```

`-d`/`--daemon` detaches it from the terminal; that needs `[orangu-server].model` set in the configuration, because there is no terminal left to prompt on. See the *Inference server* chapter for GPU backend selection and the rest of the operational detail.

To move a working server to another machine without repeating steps 1–3, bundle it and its model into one executable:

```sh
orangu-server bundle unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --all -y
```

Copy the resulting `orangu-server-bundle-<arch>` over, `chmod +x` it, and run it: no
models directory, no download, and no `orangu-server.conf` — the API comes up
on `127.0.0.1:8100` and the web console on `127.0.0.1:8200`. See *Bundling* in
the *Inference server* chapter.

## Configure orangu

With the server from step 3 running:

```sh
orangu -i
```

The first prompt is `LLM URL`: enter `http://localhost:8100/v1` (a bare `http://localhost:8100` is accepted too). The wizard queries that server's `/v1/models` and offers the first model it advertises as the default for the `Model` prompt — if the server cannot be reached it says so and asks for the name manually, which is why it is worth starting the server first.

It then walks every remaining option showing its default, so a full run is a row of Enters, and omits from the file anything left at its default. It finishes by reporting which optional tools it detected (`git lg`, `delta`, `bat`, `gh`, `glab`), printing the configuration, and asking `Write this configuration? (Yes/No) [Yes]`. On confirmation it writes `~/.orangu/orangu.conf` — ignoring `-c`/`--config`, like the server wizard — and installs any bundled skills into `~/.orangu/skills/` that are not already present:

```ini
[orangu]
server = main-server
model = unsloth/gemma-4-E2B-it-GGUF

[main-server]
endpoint = http://localhost:8100/v1
```

See [Configuration](20-configuration.md) for every key the wizard can set.

Or copy the sample instead:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Default configuration lookup order is:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

## Run orangu

`~/.orangu/orangu.conf` is one of the lookup locations, so there is nothing left to pass:

```sh
orangu
```

Or point at a specific file:

```sh
orangu --config ./orangu.conf
```

Then start with:

```text
/help
/skills
/server
/disconnect
/reload
/tools
/model
/session <UUID>
/list_files
/open_file README.md
/show_file README.md
/debugging reproduce the failing request path
/build
/shell ls -la
/create_file README.md
/auto_review
/amend "[#42] My feature"
/branch main
/branch -b feature/new
/branch -m new-name
/branch -d feature/old
/cherry_pick abc1234
/comment 51 "My comment"
/close -i 51
/issue reviewer 114 jesperpedersen
/get_comments -i 51
/commit "[#42] My feature"
/restore README.md
/diff
/init_repo
/log
/log 5
/show
/show aafd1cb
/merge feature/foo
/move_file old.rs new.rs
/pull 42
/push
/push --force
/rebase
/delete_file README.md
/review
/squash
/status
/usage
/clear
/quit
```

Most of these are thin wrappers around the matching `git`/`gh` commands. Two open full-screen views instead: `/review` walks you through the branch's diff file by file for a manual review, and `/auto_review` has the connected model review the branch's changes by itself — per file and per category (Code, Security, Memory, Performance, Test Suite, Documentation) — lets you override its verdicts afterwards (approve a file, or reject it with your own categorized comments), and copies the resulting report to the clipboard on exit. `/auto_review <file>` (Tab-completes on the file name) reviews a single file — the whole file on main/master, or just its changes on a branch — and `/auto_review all` reviews every Git-tracked file in the project. Both are described in detail in the Core tools chapter.

orangu also supports Agent Skills: reusable directories containing a
`SKILL.md`. Skills are discovered from `~/.orangu/skills/`,
`~/.agents/skills/`, `<workspace>/.orangu/skills/`, and
`<workspace>/.agents/skills/`. Use `/skills` to list them. Invoke one directly
with `/skill-name`, for example:

```text
/debugging reproduce the failing request path and identify the root cause
```

## Review your first branch

The review workflow is orangu's standout feature, so it is worth trying right away. From a feature branch with some changes (committed or just edited in the working tree):

```text
review
```

This opens the interactive reviewer: a two-pane view with your changed files on the right and the selected file's diff on the left. Use `Alt+j`/`Alt+k` to move between files, `Alt+a`/`Alt+r` to approve or reject one, and `Alt+c` to leave a categorized comment on a line. Type a question such as `is this thread-safe?` and press `Enter` to ask the model about the selected file. Press `Alt+x` to leave; the report is copied to your clipboard.

To have the model do the work, run:

```text
auto review
```

orangu reviews the whole change and each file across the Overall, Code, Security, Memory, Performance, Test Suite, and Documentation categories, marks every file with a green or red dot, and ends with an `orangu approves/rejects this patch` verdict. When the run finishes you can override any verdict and remove findings before the report lands on the clipboard.

> The branch must be rebased up to date first — if it is behind, orangu points you at `/rebase`. If you review with a *thinking* model and the answers look truncated, raise `review_max_tokens` in `[orangu]` (e.g. `2048`); see the Configuration chapter.

Share the result without leaving the terminal:

```text
export review              # write the report to a PDF in the workspace root
comment on 42 with auto review  # post it on GitHub/GitLab issue/PR #42
```

By default the tools operate on the current directory. Use `--workspace /path/to/project` to point **orangu** at another tree.

Every startup flag has a short form: `-c` for `--config`, `-t` for `--theme`, `-w` for `--workspace`, `-r` for `--resume`, `-a` for `--all` (reopen the tabs from the previous run), `-l` for `--list` (print every stored session as a table and exit), `-i` for `--init`, `-p` for `--prompt` (run one prompt or command and exit), `-q` for `--quiet` (print nothing on success), and `-s` for `--shell-completions`. `-h`/`--help` and `-V`/`--version` are available on all four binaries. `orangu --help` lists them all.

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

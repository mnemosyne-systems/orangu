# orangu

**orangu** is a local workspace-aware tool-driven coding environment for **OpenAI-compatible** servers - especially **[llama.cpp](https://github.com/ggml-org/llama.cpp)**.

**orangu** **does not** require an Internet connection after **[llama.cpp](https://github.com/ggml-org/llama.cpp)** and models have been downloaded.

**orangu** is named after the [Orangutan](https://en.wikipedia.org/wiki/Orangutan) - the smartest ape.

![orangu terminal interface](doc/images/orangu-terminal.png)

## Table of Contents

- [Why orangu?](#why-orangu)
- [Features](#features)
  - [Code review and auto review](#code-review-and-auto-review)
- [orangu vs. a cloud coding assistant](#orangu-vs-a-cloud-coding-assistant)
- [Installation](#installation)
  - [Install dependencies on Fedora](#install-dependencies-on-fedora)
  - [Release build](#release-build)
  - [Debug build](#debug-build)
- [Configuration and first run](#configuration-and-first-run)
- [Documentation](#documentation)
- [Tested platforms](#tested-platforms)
- [Sponsors](#sponsors)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

## Why orangu?

orangu is the lean, private, Git-centric coding companion for the terminal — built for developers who run their own models and want a tightly integrated review workflow without sending a single line of code to the cloud.

- **100% local and private** — zero telemetry; after the model is downloaded no Internet connection is needed, so it runs happily in privacy-sensitive or air-gapped environments. Your code and conversations stay on your machine.
- **Built-in code review** — an interactive two-pane reviewer (`/review`) *and* a category-by-category LLM auto-reviewer (`/auto_review`). This review story is orangu's standout feature; few tools its size match it.
- **A single fast native binary** — written entirely in Rust, with quick startup, no runtime to install, no garbage-collector pauses, and a small download.
- **The whole Git loop lives in the prompt** — branch, commit, rebase, squash, cherry-pick, stash, bisect, push, and GitHub/GitLab pull requests, comments, and issues, all without leaving the terminal.
- **Tuned for llama.cpp** — live tokens/second in the footer, and an interactive `--init` wizard that auto-detects the model your server is serving.
- **Agent Skills & Memory** — discovers reusable `SKILL.md` skills and merges cross-session memory and instructions from global (`~/.orangu/AGENTS.md`) and workspace-level (`./AGENTS.md`) files directly into the LLM context.
- **Natural to drive** — dozens of slash commands, each with plain-English aliases (`review`, `auto review`, `commit "..."`, `merge feature/foo`, `pull 58`).

## Features

**Fully local and private.** orangu talks to any OpenAI-compatible server — and is tuned for **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — so once the server and models are downloaded, nothing leaves your machine and no Internet connection is required. Your code is never sent to a third-party cloud.

**Code review built in.** orangu's standout feature is a pair of in-terminal review workflows — an interactive reviewer and a fully automated, LLM-driven one — covered in [Code review and auto review](#code-review-and-auto-review) below.

**Workspace-aware tooling.** Local tools read, edit, list, search (`/grep`), and fetch files, and run shell commands — all scoped to your workspace. A full set of Git commands (`/status`, `/diff`, `/log`, `/show`, `/commit`, `/amend`, `/squash`, `/rebase`, `/merge`, `/cherry_pick`, `/branch`, `/stash`, `/bisect`, `/push`, `/pull`, …) and forge integration (`/pull_request`, `/comment`, `/close`, `/get_comments` on GitHub and GitLab) keep the whole change-and-review loop in one place.

**Advanced Context Compression Engine.** orangu protects the LLM's context window and minimizes latency using a state-of-the-art compression pipeline. Features include AST-aware file downsampling, an intelligent Git diff engine, session fingerprinting, secret redaction, and automatic transcript compaction. See the [Compression](doc/manual/en/75-compression.md) manual for details.

**Duplicate-code detection.** `/duplicates` parses every function in the workspace — across more than 20 languages (Rust, C/C++, C#, Go, Java, Python, JavaScript/TypeScript, Ruby, PHP, and more) — into a tree-sitter AST and scores each same-language pair with the Sørensen–Dice coefficient over their AST node bigrams, so functions that share a *shape* — even with different names and values — surface as similarity-ranked candidates for you to review. Save the report to a PDF with `/export duplicates`. See the [`/duplicates`](doc/manual/en/41-core_tools.md) tool.

**Multiple workspaces as tabs.** Open several projects at once in one orangu instead of one instance per project. Each workspace is a tab with its own session, scrollback, pending queue, and command history; switch with `Alt+,`/`Alt+.` or the `/workspace` command, open and close tabs with `/create_workspace <dir>` / `/delete_workspace` (or `Alt+Insert`/`Alt+Delete`), and reopen the last set of tabs at startup with `-a`/`--all`. See the [Workspaces](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/31-workspaces.md) chapter.

**Agent Skills & Workspace Memory.** orangu discovers skills from four locations: `~/.orangu/skills/`, `~/.agents/skills/`, `<workspace>/.orangu/skills/`, and `<workspace>/.agents/skills/`. At startup it discloses only each skill's name, description, and `SKILL.md` location to the model. Additionally, `orangu` automatically scans for `AGENTS.md` files in your home directory and workspace root, injecting persistent project instructions and long-term memory into every chat and review session.

**Knowledge Graph & Codebase Visualization.** orangu incrementally parses your entire codebase offline using Tree-sitter, mapping every function, class, and call into a dependency graph. The AI uses this graph to instantly pinpoint the most central code symbols for any query without flooding its context window. You can also type `/graph` to instantly generate an interactive HTML visualization of the codebase architecture that opens in any browser.

**Semantic code search.** `/search <query>` retrieves code by *meaning*, not just text — a query like `where is rate-limiting handled?` surfaces a `throttle_requests` function whose name shares no words with the query. orangu embeds every symbol offline through the server's OpenAI-compatible `/v1/embeddings` endpoint, persists the vectors under `~/.orangu/workspace/<hash>/embeddings/` (re-embedding only changed files and dropping deleted ones on every search, like the knowledge graph cache), and ranks results by a hybrid of cosine similarity and the knowledge graph's call edges — a semantic seed followed by structural expansion. It enables itself automatically: tag a server `role = embeddings` (or let the default `all` server serve it), and `/search` turns on when that endpoint responds at startup — otherwise it stays dormant and retrieval falls back to `/grep` and the knowledge graph. See the [`/search`](doc/manual/en/41-core_tools.md) tool.

**Comfortable terminal experience.**

- Persistent terminal UI with workspace, server, and model status in the header, refreshed every minute while idle
- Shell-style prompt editing, history, scrolling, and context-sensitive Tab completion, with grey inline command hints (Tab accepts, Shift+Tab cycles between matches)
- Natural-language aliases for nearly every command — e.g. `review`, `auto review`, `open README.md`, `list models`, `pull 58`, `commit "[#42] My feature"`, `rebase`, `merge feature/foo`, `get comments for issue 51`, `export review`
- Streaming responses with live footer status such as `Thinking (...)` and llama.cpp-native `Working @ X.Y t/s (...)`
- Queued local commands while a response is in flight, plus double-`Esc` request cancellation
- Markdown rendering in the console (bold, italic, headings, lists, links, code) with syntax highlighting for fenced code blocks

**Share what you produce.** Export the output window, the last review report, or a duplicate-code report to a PDF in the workspace root (`/export console`, `/export review`, `/export duplicates`), or post a review straight onto an issue or pull request with `/comment <number> with review` / `with auto review`.

**Works offline, end to end.** Even the built-in user manual (`/manual`) — a two-pane viewer with full-text search (`Alt+S`) — is embedded in the binary at compile time, so the docs are there with no network.

### Code review and auto review

orangu turns the terminal into a code-review workstation for the changes on your current branch (committed plus local uncommitted work), measured against the merge base with the default branch. Both reviewers require the branch to be rebased up to date, so you never review against stale code.

<!-- TODO: add a screenshot or asciinema GIF of the /auto_review two-pane view (status bar + category report + file dots) here, e.g.:
![orangu auto review](doc/images/orangu-auto-review.png)
A captured image sells this feature far better than prose. -->


**`/review` — interactive review.** A full-screen, two-pane view (file checklist on the right, the selected file's diff on the left, your prompt at the bottom) for reading a branch before you share it. You can:

- Mark each file approved (green) or rejected (red)
- Comment on any diff line under a chosen category (Overall, Code, Security, Memory, Performance, Test Suite, Documentation), plus whole-patch notes
- Ask the connected model about the selected file on demand (`focus on error handling`, `is this thread-safe?`) — the exchange joins your chat session for follow-up
- Open any workspace file in your `$EDITOR` without leaving the view

On exit it writes a category-grouped report with an approve/reject **Conclusion**, copies the Markdown to the clipboard, and keeps it for `/export review` and `/comment ... with review`. No `gh`/`glab` needed.

**`/auto_review` — LLM-driven review.** The model reviews the whole change and each file on its own, sorting findings into the same seven categories and marking every file approved or rejected — then summarizes the change as a whole under **Overall** and renders a final **Conclusion** verdict (`orangu approves/rejects this patch`). It is smart about effort:

- File type decides what's scanned — lock files and binary assets are auto-approved with no requests, documentation is reviewed only for the Documentation category, and source files get the full set of checks
- Per-file review depth (`Alt+m` cycles **Normal → Deep → Ignore**, or launch every file straight into Deep with the `deep` keyword): Normal is the default pass; **Deep** never truncates the diff, pulls in cross-file context from the workspace's knowledge graph (who else calls the changed code), and re-checks any rejected findings before accepting them; **Ignore** skips the file entirely and auto-approves it
- Uses a **Rigorous Review Rubric** combined with **Confidence Scoring** (0-100) to automatically filter out false positives, hallucinations, and pedantic nitpicks. It only flags high-confidence bugs that meaningfully impact correctness, security, or performance.
- A live status bar shows the current file, category, overall progress, elapsed time, and an updating time estimate; the terminal title blinks and the bell rings on completion (when `feedback` is on)
- Each finding is pinned to its `file:line`, and requests are length-capped and tool-free so reviews stay fast and bounded even on slow local models
- After the run you can browse the report, override any verdict (approve/reject with your own comment), and remove findings before exporting

Run `/auto_review <file>` to review a single file (the whole file on `main`/`master`, or just its changes on a branch), or `/auto_review all` to review every Git-tracked file in the project. `immediate` and `deep` combine with either form, or the bare command, in any order — add `immediate` to skip the pre-start phase and `deep` to start every file in Deep mode, e.g. `/auto_review all immediate`, `/auto_review deep <file>`, or `/auto_review deep all immediate` (Tab-completed and ghosted like the file argument). Like `/review`, the report is copied to the clipboard and reusable with `/export review` and `/comment ... with auto review`.

> **Tip:** You can control the chatty nature of local models using the `model_verbosity` (`terse`, `normal`, `verbose`) and `reasoning_effort` options in your `orangu.conf`. 
> The per-request length cap is `review_max_tokens` (default `512`; `0` disables it). If you review with a model that *thinks* before answering, raise it (e.g. `2048`) so the reasoning tokens don't crowd out the answer — see the [Configuration](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/20-configuration.md) chapter (*Response-token caps*). Set `feedback = on` to get the blinking terminal title and completion bell during a run.

See [Core tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/41-core_tools.md) in the manual for the full reference and key bindings.

## orangu vs. a cloud coding assistant

orangu makes a deliberate trade: a focused, offline-first, Git-centric terminal experience instead of a broad cloud platform. If you run your own models and care about privacy, that trade is the whole point.

| | **orangu** | **Typical cloud coding assistant** |
| --- | --- | --- |
| **Where your code goes** | Stays on your machine — zero telemetry | Sent to a third-party provider |
| **Offline use** | First-class; only the initial model download needs a network | Generally requires connectivity |
| **Models** | Any OpenAI-compatible server — tuned for local llama.cpp (Ollama, LM Studio, …) | Vendor-hosted models, usually behind API keys |
| **Cost** | Free to run against models you host | Per-token / subscription billing |
| **Footprint** | One native Rust binary, fast start, no runtime | Editor/cloud service + account |
| **Code review** | Built-in interactive **and** LLM auto review in the terminal | Usually delegated to the hosting platform |
| **Git workflow** | Full Git + GitHub/GitLab loop from the prompt | Varies; often browser-based |
| **Privacy posture** | Suited to regulated / air-gapped environments | Depends on the provider's data policy |

orangu trades breadth and extensibility for simplicity, predictability, and a small attack surface. If you need a multi-front-end platform with a large plugin/MCP ecosystem, a cloud-first agent will fit better — orangu is the lean, private alternative for local models.

## Installation

### One-liner install (Linux, macOS, Windows)

**Linux / macOS** (requires `curl` or `wget`, and `tar`):

```sh
curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.sh | sh
```

**Windows** (requires PowerShell, included with Windows 10 and later):

```cmd
curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.cmd -o install.cmd && install.cmd
```

Both scripts download the latest release binary, install it to `~/.local/bin` (Linux/macOS) or `%USERPROFILE%\.local\bin` (Windows), and warn if the directory is not in your `PATH`.

**Custom install directory:** set `INSTALL_DIR` before running the script:

```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.sh | INSTALL_DIR=/usr/local/bin sh
```

```cmd
:: Windows
set "INSTALL_DIR=C:\Tools" && install.cmd
```

**Shell completions:** after installing, run `orangu -s` to print the completion script for your shell:

```sh
# bash
orangu -s >> ~/.bashrc && source ~/.bashrc

# zsh
orangu -s >> ~/.zshrc && source ~/.zshrc

# fish
orangu -s | source
```

On Windows, add `Invoke-Expression (orangu -s)` to your PowerShell `$PROFILE`.

### Build from source

#### Install dependencies

**Fedora / RHEL:**

```sh
dnf install -y git rust cargo
```

**Debian / Ubuntu:**

```sh
apt-get install -y git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**macOS:**

```sh
brew install rust
```

#### Release build

The following commands build an optimized release binary:

```sh
git clone https://github.com/mnemosyne-systems/orangu.git
cd orangu
cargo build --release
```

The binary will be available at:

```text
target/release/orangu
```

To install it system-wide:

```sh
sudo install -Dm755 target/release/orangu /usr/local/bin/orangu
```

#### Debug build

The following commands build a debug binary:

```sh
git clone https://github.com/mnemosyne-systems/orangu.git
cd orangu
cargo build
```

The binary will be available at:

```text
target/debug/orangu
```

## Configuration and first run

The quickest way to get a working configuration is the interactive wizard:

```sh
orangu --init
```

It asks for the **LLM URL**, auto-detects a model the server advertises (and
pre-fills it as the **Model**), then walks every option showing its default.
Anything left at its default is omitted from the file, and the result is shown
for confirmation before being written to `~/.orangu/orangu.conf` (creating the
directory if needed, and overwriting any existing file). The provider is
assumed to be [llama.cpp](https://github.com/ggml-org/llama.cpp). The wizard
also installs bundled skills into `~/.orangu/skills/` when they are not
already present; currently this includes `debugging`.

Alternatively, start from the sample configuration:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Default configuration lookup order:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

Run the client:

```sh
orangu --config ./orangu.conf
```

Or run it directly from the build tree:

```sh
./target/release/orangu --config ./orangu.conf
```

By default, local tools operate on the current working directory. Use `--workspace /path/to/project` (`-w`) to point **orangu** at another tree.

The startup flags also have short forms: `-c` for `--config`, `-w` for `--workspace`, `-r` for `--resume`, `-a` for `--all` (reopen the last run's workspace tabs), `-l` for `--list` (print every stored session as a table and exit), `-i` for `--init`, and `-s` for `--shell-completions`.

Shell completion scripts (bash, zsh, fish) for these flags live in [`contrib/shell/`](contrib/shell/README.md).

Useful first commands:

```text
/help
/skills
/tools
/list_files
/open_file README.md
/show_file README.md
/debugging reproduce the failing request path and identify the root cause
/amend "[#42] My feature"
/cherry_pick abc1234
/commit "[#42] My feature"
/delete feature/foo
/log
/log 5
/show
/show aafd1cb
/squash
/status
/graph
```

## Documentation

- [Latest manual](https://github.com/mnemosyne-systems/orangu/tree/main/doc/manual/en)
- [Getting Started](https://github.com/mnemosyne-systems/orangu/blob/main/doc/GETTING_STARTED.md)
- [orangu-coordinator](https://github.com/mnemosyne-systems/orangu/blob/main/doc/COORDINATOR.md) — auto-start/stop llama.cpp for machines that only run one local model at a time
- [Quick start](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/03-quickstart.md)
- [Configuration](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/20-configuration.md)
- [Workspaces](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/31-workspaces.md)
- [Skills](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/32-skills.md)
- [Terminal interface](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/40-terminal.md)
- [Core tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/41-core_tools.md)
- [Git tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/42-git_tools.md)
- [Usage tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/43-usage_tools.md)
- [Tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/30-tools.md)

## Tested platforms

- [Fedora](https://getfedora.org/) 44

## Sponsors

- [mnemosyne systems](https://www.mnemosyne-systems.ai/)

## Contributing

Contributions to **orangu** are managed on [GitHub](https://github.com/mnemosyne-systems/orangu/):

- [Ask a question](https://github.com/mnemosyne-systems/orangu/discussions)
- [Raise an issue](https://github.com/mnemosyne-systems/orangu/issues)
- [Feature request](https://github.com/mnemosyne-systems/orangu/issues)
- [Code submission](https://github.com/mnemosyne-systems/orangu/pulls)

Contributions are most welcome.

Please consult the [Code of Conduct](https://github.com/mnemosyne-systems/orangu/blob/main/CODE_OF_CONDUCT.md) before contributing.

## Community

- GitHub: [mnemosyne-systems/orangu](https://github.com/mnemosyne-systems/orangu)
- Discussions: [GitHub Discussions](https://github.com/mnemosyne-systems/orangu/discussions)

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

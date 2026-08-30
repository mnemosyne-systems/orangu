# Optional tools

None of these are required. When one is installed **and** configured, orangu
uses it by itself — `orangu -i` reports each as `No`, `Yes (Used)` or
`Yes (Not used)`.

| Command | What it does |
| --- | --- |
| `git lg` | A graph-formatted `/log`. Used when the `lg` alias is in `~/.gitconfig`: `git config --global alias.lg "log --color --graph --abbrev-commit"`. |
| `delta` | A syntax-highlighted, side-by-side `/diff`. Used when it is your Git diff pager (`pager.diff`, then `core.pager`). |
| `bat` | A syntax-highlighted `/show_file`. Used as soon as it is installed. |
| `gh` | `/pull_request`, `/comment`, `/close` and `/issue` need it; `/pull`, `/merge` and `/rebase` improve with it. Used when `[orangu].platform = github`. |
| `glab` | The same, for GitLab. Used when `[orangu].platform = gitlab`. |

## MCP servers

Tools from an already-running Streamable HTTP MCP service, offered to the model
beside the built-in ones. orangu connects; it does not launch them.

| Command | What it does |
| --- | --- |
| `[mcp.<name>]` | A configured Streamable HTTP service; URL, filters and approval are in `orangu.conf`. |
| `/mcp` `/mcp refresh` | Show and reconnect; `/mcp add` / `remove` manage one. `/tools` lists `mcp__<server>__<tool>`. |

## Workflow files

| Command | What it does |
| --- | --- |
| `orangu --workflow run.yaml --dry-run` / `--workflow run.yaml` | Validate every job without executing; then run the jobs in order. |
| `orangu --workflow run.yaml` `status` `pause` `resume` `clear` | Run a workflow, or inspect/pause/resume/clear its saved job state. |

## The rest of the stack

| Command | What it does |
| --- | --- |
| `orangu-coordinator` | An HTTP proxy that swaps `orangu-server` models on demand — one GPU, a different model per role. |
| `orangu-server bundle <model>` | One executable carrying the server and its model: copy it over and run it. |
| `orangu-bench` | Benchmark a model or a server — throughput, latency, and quality. |
| `orangu-server list` | What is installed; `delete` removes one, `refresh` re-fetches it, `-d` runs detached. |
| Web console | `orangu-server -i` offers it on port 8101: models, load and requests, in a browser. |
| GPU backends | Vulkan, Metal, CUDA, ROCm, OpenCL — or plain CPU. Pure Rust either way. |

## When something looks off

| Command | What it does |
| --- | --- |
| `/information` `/server` | What the server is, and what it can actually do. |
| `/tools` | The tools the model is offered. A model whose template has no tool support will say it has none. |
| `/disconnect` `/reload` `/prune` | Drop the connection; restore the configured server and model; clear out old sessions. |

## Where to read more

| Command | What it does |
| --- | --- |
| `/manual` | The full manual, offline, with search. `orangu-en.pdf` is the same, in the release archive. |
| `mnemosyne-systems.github.io/orangu` | Documentation, releases and the installers. |
| `github.com/mnemosyne-systems/orangu` | Source, issues and discussions. GPL v3, with commercial support at `mnemosyne-systems.ai`. |

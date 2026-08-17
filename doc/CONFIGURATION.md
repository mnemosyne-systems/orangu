# Configuration

`orangu` uses an INI configuration file.

Default lookup order:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

This file is the complete key reference for the client. `orangu-server` and
`orangu-coordinator` read their own files, documented in
[SERVER.md](SERVER.md) and [COORDINATOR.md](COORDINATOR.md).

## Main section

The client section is named `[orangu]`. It selects the default server and
holds client-wide settings.

```ini
[orangu]
server = orangu-server
model = ggml-org/gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10
review_max_tokens = 512
code_max_tokens = 0
compression = on
theme = classic
```

### Server selection

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. Defaults to `1800` |

### Limits and budgets

| Key | Required | Description |
| :-- | :-- | :-- |
| `max_tool_rounds` | No | Maximum tool-calling turns per prompt before the client aborts it. Defaults to `10` |
| `review_max_tokens` | No | Response-token cap for each `/auto_review` request. Defaults to `512`; `0` disables the cap. Raise it (e.g. `2048`) when the review model thinks before answering |
| `code_max_tokens` | No | Response-token cap for normal chat and tool responses. Defaults to `0` (no cap) |
| `review_confidence_threshold` | No | Minimum confidence score (0–100) for `/auto_review` findings; findings below it are silently dropped. Defaults to `80`. Set to `0` to disable filtering |
| `semantic_budget_tokens` | No | Token budget for the code chunks `/search` injects into a turn. Hits are added in rank order until the next one would exceed it, so the cap bounds what semantic search costs in context rather than the number of results. Defaults to `16384`; the top hit is always kept |
| `world_state_max_bytes` | No | Ceiling in bytes on the `world_state_changes` fragment prepended to a turn when the working tree has changed. Defaults to `8192`; `0` disables the cap. The fragment is prefilled by the server, so its size is response latency, not just context |
| `compile_workers` | No | Parallel job count `/build` passes to toolchains that support one (e.g. `make -j`, `meson compile -j`, `cargo --jobs`). Defaults to `0`, meaning unused: no job flag is passed and each toolchain falls back to its own default |

### Compression

| Key | Required | Description |
| :-- | :-- | :-- |
| `compression` | No | Enable orangu's built-in compression layer: context deduplication, file-read stubbing, and shell-output compression (handles `cargo`, `ls`, `grep`/`rg`, `npm`/`yarn`/`pip`, and diff truncations). Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_downsample_lines` | No | Line count above which an unbounded file read returns signatures instead of the whole file, with a note saying so. Defaults to `300`; `0` reads every file in full. Applies only while `compression` is on, and never to a read that asked for a `mode` or a line range |
| `diff_file_cap` | No | Maximum number of files kept when a `git diff` is compressed. Defaults to `20` |

### Prompt

| Key | Required | Description |
| :-- | :-- | :-- |
| `system_prompt` | No | Override the base system prompt sent to the model. When empty (the default) orangu uses its built-in coding-assistant prompt. The discovered Agent Skills index is appended to whichever prompt is in effect |

### Interface

| Key | Required | Description |
| :-- | :-- | :-- |
| `theme` | No | Global default UI theme. Defaults to `classic`. Built-ins are `classic`, `modern_dark`, `modern_light`, `oranguday`, `tokyonight`, and `rosepine-moon`; `random` draws one of the available themes at each launch. User themes are loaded from `~/.orangu/themes/*.theme` |
| `banner` | No | Horizontal placement of the header banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `width` | No | Virtual terminal width for the output canvas. Source lines from `/show_file` are laid out at this width and can be panned horizontally. Defaults to `512` |
| `word_wrap` | No | Wrap long lines in the main TUI, `/show_file`, `/review`, and `/auto_review` windows. Defaults to `off`; set it to `on` to wrap at the visible width. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `drop_down` | No | Enable the autocomplete dropdown for slash commands. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `mouse` | No | Enable mouse capture, so the TUI handles scroll and double-click. Defaults to `on`; hold **Shift** while clicking or dragging for native text selection. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `workspaces` | No | Placement of the workspace tabs. Defaults to `top`. Options: `top`, `bottom`, `left`, `right` |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `sun_tzu_mandarin`, `sun_tzu_english`, `attila_the_hun`, `all` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure, blink an `orangu ●` progress title and ring the terminal bell when a `/auto_review` finishes. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `terminal` | No | Launch command used to open `$EDITOR` for terminal editors in a new window for `/open_file` (for example `xterm -e` or `kitty`). When unset, a terminal emulator is auto-detected |

### Git and code hosting

| Key | Required | Description |
| :-- | :-- | :-- |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |

## Server sections

Each server section is a valid value for `[orangu].server` and carries the host
information for that model. `[orangu-server]` is the name written by `orangu -i`.

```ini
[orangu-server]
role = all
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | `orangu-server` URL (its OpenAI-compatible API) |
| `model` | No | Model identifier sent to the server. Overrides the general `[orangu].model` when set |
| `role` | No | A specific role this server fulfills. Valid roles are: `all` (default), `code`, `review`, `explorer`, and `embeddings`. If a specific subsystem needs a server and one is tagged with its role, it will use that server instead of the default. `embeddings` designates the server that embeds code for semantic `/search`; an `all` server also serves it, and search auto-enables when that endpoint responds at startup. Ignored behind a confirmed [orangu-coordinator](COORDINATOR.md) — it alone decides which model backs each role, so a single server section is enough there. |
| `api_key` | No | API key sent as `Authorization: Bearer <key>` on every request to the server (chat completions and model listing). Required when `orangu-server` is started with `--api-key` |
| `model_verbosity` | No | How chatty this server's model should be. Defaults to `normal`. Options: `terse`, `normal`, `verbose`. It is a per-server key: writing it in `[orangu]` has no effect |

At least one of `[orangu].model` or a server's own `model` must be set, so every
server resolves to a non-empty model.

Each server section must resolve to a **unique** (`endpoint`, `model`) pair —
a server represents one host serving one model, and `/model` cycles the
models that host offers. `http://x` and `http://x/v1` are treated as the same
endpoint. Two sections *may* share an `endpoint` as long as their `model`
differs, e.g. several roles proxied through one
[orangu-coordinator](COORDINATOR.md) address. The `api_key` is attached to every `/v1/*` request, so the
`/v1/models` health probe also works against API-key-protected servers.

Use `/server` to switch between the configured servers at runtime; Tab
completion lists every server section.

The canonical example file is `doc/etc/orangu.conf`.

## MCP servers

`orangu` connects only to already-running Streamable HTTP MCP servers; it never
starts a command or manages a child process. Their tools are namespaced as
`mcp__<server>__<tool>`. `/mcp` shows the connected servers, while `/tools`
shows the model-facing tools. Configure a server in `[mcp.<name>]`, or set
`mcp = on` in an existing section.

```ini
[mcp.weather]
endpoint = http://localhost:9000/mcp
timeout = 30
approval_mode = writes
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | Streamable HTTP URL of the running service, normally ending in `/mcp` |
| `mcp` | Only for the shortcut form | Set to `on` in a section that is not named `mcp.<name>` to read that section as an MCP service too. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `timeout` | No | Seconds allowed for initialization, tool discovery, and tool calls alike. Defaults to `30`, and supplies the default for the two keys below |
| `startup_timeout` | No | Seconds for connection and tool discovery alone. Defaults to `timeout`. Must be greater than zero |
| `tool_timeout` | No | Seconds for a single tool call. Defaults to `timeout`. Must be greater than zero |
| `enabled` | No | Whether the service is used at all. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `required` | No | Make a failed connection abort workspace startup instead of disabling the service with a warning. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `enabled_tools` | No | Comma-separated allowlist of tool names. Empty (the default) offers every discovered tool |
| `disabled_tools` | No | Comma-separated denylist of tool names. The denylist wins over `enabled_tools` |
| `approval_mode` | No | How tool calls are confirmed: `auto` runs them directly, `prompt` asks for each call, `writes` asks unless the tool declares `readOnlyHint`, and `deny` disables the service. Defaults to `auto`. Noninteractive and review runs deny whatever needs asking |

The name after `mcp.` is the service name used in the `mcp__<server>__<tool>`
prefix, and it accepts ASCII letters, digits, `_` and `-`. Configuring the same
name twice — once as `[mcp.<name>]` and once through the `mcp = on` shortcut —
is a startup error.

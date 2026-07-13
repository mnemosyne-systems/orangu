# Configuration

`orangu` uses an INI configuration file.

Default lookup order:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

## Main section

The client section is named `[orangu]`. It selects the default server and
holds client-wide settings. Each server is described in its own section,
named by the value of `server`.

```ini
[orangu]
server = main-server
model = ggml-org/gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10
review_max_tokens = 512
code_max_tokens = 0
compression = on
theme = classic
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. Defaults to `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns per prompt. Defaults to `10` |
| `review_max_tokens` | No | Response-token cap for each `/auto_review` request. Defaults to `512`; `0` disables the cap. Raise it (e.g. `2048`) when the review model thinks before answering |
| `code_max_tokens` | No | Response-token cap for normal chat and tool responses. Defaults to `0` (no cap) |
| `compile_workers` | No | Parallel job count `/build` passes to toolchains that support one (e.g. `make -j`, `meson compile -j`, `cargo --jobs`). Defaults to `0`, meaning unused: no job flag is passed and each toolchain falls back to its own default |
| `compression` | No | Enable orangu's built-in compression layer. This provides context deduplication, file read stubbing, and advanced shell output compression (handles `cargo`, `ls`, `grep`/`rg`, `npm`/`yarn`/`pip`, and diff truncations). Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `sun_tzu_mandarin`, `sun_tzu_english`, `attila_the_hun`, `all` |
| `width` | No | Virtual terminal width for the output canvas. Source lines from `/show_file` are laid out at this width and can be panned horizontally. Defaults to `512` |
| `banner` | No | Horizontal placement of the header banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `theme` | No | Global default UI theme. Defaults to `classic`. Built-ins are `classic`, `oranguday`, `tokyonight`, `rosepine-moon`, and `auto`; user themes are loaded from `~/.orangu/themes/*.theme` |
| `auto_dark_theme` | No | Concrete theme used when `theme = auto` detects a dark terminal. Defaults to `classic` |
| `auto_light_theme` | No | Concrete theme used when `theme = auto` detects a light terminal. Defaults to `oranguday` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure, blink an `orangu ●` progress title and ring the terminal bell when a `/auto_review` finishes. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |

## Server sections

Each server is declared in its own section. The section name (for example
`[main-server]`) is what `[orangu].server` points to, and the section carries
the host information for that server.

```ini
[main-server]
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

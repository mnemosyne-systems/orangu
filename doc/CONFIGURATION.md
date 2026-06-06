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
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. Defaults to `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns per prompt. Defaults to `10` |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `all` |
| `width` | No | Virtual terminal width for the output canvas. Source lines from `/show_file` are laid out at this width and can be panned horizontally. Defaults to `512` |
| `banner` | No | Horizontal placement of the header banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |

## Server sections

Each server is declared in its own section. The section name (for example
`[main-server]`) is what `[orangu].server` points to, and the section carries
the host information for that server.

```ini
[main-server]
provider = llama.cpp
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `provider` | Yes | `llama.cpp` or `openai` |
| `endpoint` | Yes | OpenAI-compatible server URL |
| `model` | No | Model identifier sent to the server. Overrides the general `[orangu].model` when set |
| `api_key` | No | API key sent as `Authorization: Bearer <key>` on every request to the server (chat completions and model listing). Required when a llama.cpp server is started with `--api-key`, or for any authenticated OpenAI-compatible endpoint |

At least one of `[orangu].model` or a server's own `model` must be set, so every
server resolves to a non-empty model.

Each server section must use a **unique** `endpoint` — a server represents one
host, and `/model` cycles the models that host offers. `http://x` and
`http://x/v1` are treated as the same endpoint. The `api_key` is attached to
every `/v1/*` request, so the `/v1/models` health probe also works against
API-key-protected servers.

Use `/server` to switch between the configured servers at runtime; Tab
completion lists every server section.

The canonical example file is `doc/etc/orangu.conf`.

\newpage

# Configuration

`orangu` uses an INI configuration file.

## `[orangu]`

The main section selects the default server and client-wide limits. The
`server` key names the server section that holds the host information:

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
| `timeout` | No | Request timeout in seconds. The default is `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns before the client aborts the prompt |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `all` |
| `width` | No | Virtual terminal width in characters. Controls the layout canvas for `/show_file` output. Defaults to `512` |
| `banner` | No | Horizontal placement of the banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `terminal` | No | Launch command used to open `$EDITOR` for terminal editors in a new window for `/open_file` (for example `xterm -e` or `kitty`). When unset, a terminal emulator is auto-detected |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |

## Server sections

Each server is a named section. The section name is what `[orangu].server`
points to, and it carries the host information for that server:

```ini
[main-server]
provider = llama.cpp
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `provider` | Yes | `llama.cpp` or `openai` |
| `endpoint` | Yes | OpenAI-compatible API URL |
| `model` | No | Model identifier used in chat completion requests. Overrides the general `[orangu].model` when set |
| `api_key` | No | API key sent as `Authorization: Bearer <key>` on every request to the server. Required when a llama.cpp server runs with `--api-key`, or for any authenticated OpenAI-compatible endpoint |

- At least one of `[orangu].model` or a server's own `model` must be set, so every server resolves to a non-empty model
- The endpoint may be configured either with or without `/v1`
- The client normalizes the endpoint internally before calling `/v1/chat/completions`
- Set `api_key` when the server requires authentication, for example a llama.cpp server started with `llama-server --api-key <key>`. The key is sent as a bearer token on every request, including the `/v1/models` probe
- Each server section must use a unique `endpoint`; `http://x` and `http://x/v1` are treated as the same host
- Use `/server` to switch between the configured servers at runtime; Tab completion lists every server section
- Set `feedback = on` in `[orangu]` to show a green or red dot in the output window after each command completes

## Sample file

The distributed sample lives at:

```text
doc/etc/orangu.conf
```

It ships with llama.cpp-style servers and a 30-minute timeout suitable for local tool-calling workloads.

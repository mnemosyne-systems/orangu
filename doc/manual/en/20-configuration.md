\newpage

# Configuration

`orangu` uses an INI configuration file.

## Interactive setup (`--init`)

Run `orangu --init` (short form `-i`) to generate the configuration
interactively instead of editing the file by hand:

```sh
orangu --init
```

The wizard:

1. Asks for the **LLM URL** (the server `endpoint`).
2. Queries the server's `/v1/models` endpoint and pre-fills the first
   advertised model as the **Model** value; if no model can be detected, you
   enter one manually.
3. Walks every `[orangu]` and server option, showing its default in
   `[brackets]`. Press Enter to keep the default. Boolean options accept
   `Yes`/`Y`/`No`/`N` (case-insensitive).
4. Reports which [optional external tools](#optional-external-tools) it
   detects (`git lg`, `delta`, `bat`, `gh`, and `glab`). Each is shown as `No`
   when the tool is absent, `Yes (Used)` when it is installed and configured to
   be used, or `Yes (Not used)` when it is installed but not yet wired up — for
   example `delta` installed but not set as your Git diff pager. See
   [Optional external tools](#optional-external-tools) for how each one is
   activated.
5. Installs bundled skills into `~/.orangu/skills/` when they are not already
   present. At the moment this includes `debugging`.
6. Shows the resulting configuration and asks for confirmation before writing.

The provider is assumed to be `llama.cpp`. Only values that differ from their
default are written, so the generated file stays minimal. It is written to
`~/.orangu/orangu.conf`, creating `~/.orangu/` if needed and overwriting any
existing file. Bundled skills are written to `~/.orangu/skills/<skill>/SKILL.md`
and are left untouched when the file already exists.

## Agent Skills

orangu supports Agent Skills: directories containing a `SKILL.md` file with
YAML frontmatter and markdown instructions. Skills are discovered from four
locations:

1. `~/.orangu/skills/`
2. `~/.agents/skills/`
3. `<workspace>/.orangu/skills/`
4. `<workspace>/.agents/skills/`

Project skills override user skills with the same name. The `/skills` command
lists the discovered skills. A skill can be invoked explicitly with
`/skill-name`; for example, the bundled `debugging` skill is typically
available after `--init`:

```text
/debugging reproduce the failing request path and identify the root cause
```

## `[orangu]`

The main section selects the default server and client-wide limits. The
`server` key names the server section that holds the host information:

```ini
[orangu]
server = main-server
model = ggml-org/gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10
review_max_tokens = 512
code_max_tokens = 0
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. The default is `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns before the client aborts the prompt |
| `review_max_tokens` | No | Response-token cap for each `/auto_review` request. Defaults to `512`; `0` disables the cap. Raise it (e.g. `2048`) when the review model thinks before answering |
| `code_max_tokens` | No | Response-token cap for normal chat and tool responses. Defaults to `0` (no cap) |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `sun_tzu_mandarin`, `sun_tzu_english`, `attila_the_hun`, `all` |
| `width` | No | Virtual terminal width in characters. Controls the layout canvas for `/show_file` output. Defaults to `512` |
| `banner` | No | Horizontal placement of the banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `workspaces` | No | Placement of the workspace tabs. Defaults to `top`. Options: `top`, `bottom`, `left`, `right`. See the Workspaces chapter |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure, blink an `orangu ●` progress title and ring the terminal bell when a `/auto_review` finishes. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `terminal` | No | Launch command used to open `$EDITOR` for terminal editors in a new window for `/open_file` (for example `xterm -e` or `kitty`). When unset, a terminal emulator is auto-detected |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |

### Response-token caps

`review_max_tokens` and `code_max_tokens` bound how long a model response may
get. Each is sent to the server as the `max_tokens` field of the chat
completion request, so the server stops generating when the cap is reached —
the request does not fail, the response is simply cut off at that point. A
value of `0` disables the cap entirely: no `max_tokens` field is sent and the
server's own default applies. Like `timeout` and `max_tool_rounds`, both keys
are client-wide and apply to every configured server.

The two caps cover the two kinds of request the client makes:

- **`review_max_tokens`** applies to every `/auto_review` request — the
  per-file category reviews and the final whole-change pass. The default of
  `512` fits the requested format (a verdict plus at most five one-line
  findings) comfortably, and exists so a review can never generate unbounded
  output: a runaway or endlessly deliberating model is cut off rather than
  stalling the run.
- **`code_max_tokens`** applies to the normal conversation — prompts typed at
  the input window, including tool-calling turns. It defaults to `0` (no cap)
  because coding answers are open-ended: explanations, diffs, and file
  contents can legitimately be long. Set it only when a model tends to ramble
  or you want a hard latency bound per response.

**Reasoning ("thinking") models need a larger review cap.** A model's hidden
thinking tokens count against `max_tokens`, so with the default `512` a model
that deliberates at length can be cut off before it emits its verdict. Such a
truncated review is handled safely — a response with no verdict and no
findings is recorded under **Overall** as a failed category review and the
file keeps its white (unreviewed) box, so a truncation can never silently
approve a file — but the review is wasted. When reviewing with thinking
enabled, raise the cap so the answer survives the thinking:

```ini
[orangu]
review_max_tokens = 2048
```

Conversely, for the fastest reviews disable thinking on the server (for
llama.cpp: `--reasoning-budget 0` together with
`--chat-template-kwargs '{"enable_thinking": false}'`) and keep the default
`512` — the cap then almost never binds and only guards against runaways.

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
- Set `feedback = on` in `[orangu]` to show a green or red dot in the output window after each command completes, and to blink an `orangu ●` title and ring the terminal bell while/when a `/auto_review` runs and finishes

## Sample file

The distributed sample lives at:

```text
doc/etc/orangu.conf
```

It ships with llama.cpp-style servers and a 30-minute timeout suitable for local tool-calling workloads.

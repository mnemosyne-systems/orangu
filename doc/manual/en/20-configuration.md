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

The server is `orangu-server`. Only values that differ from their
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

See the Skills chapter for how to write instruction-only skills, skills with
helper files, and skills that compile helper code.

## Startup connectivity

The terminal UI renders immediately on launch — it never waits on a network
check first. The header's `Server`/`Model` rows show a white dot while
connectivity is still being resolved in the background, turning green or red
within a moment.

If the default server (`[orangu] server = ...`) doesn't respond, orangu
automatically tries the other configured server sections in turn and
switches to the first one that does, printing "Switched to server: `<name>`"
to the output window. If none of them respond either, it stays on the
default (which then shows red, as usual). This happens once, at startup;
`/server` still switches on demand exactly as before.

## Per-session server and model

Each workspace tab keeps its own active server, model, and endpoint. A `/server`
or `/model` command in one tab does not affect any other tab, and switching tabs
restores the server and model that were active there.

These choices are persisted automatically. When you run `/server` or `/model`,
the selected values are written to the session's settings file:

```
~/.orangu/sessions/<UUID>/settings
```

The next time that session is resumed — whether in a new run or after a tab
switch away and back — orangu restores the server and model from there. No
manual config files are needed.

\newpage

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
theme = classic
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. The default is `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns before the client aborts the prompt |
| `review_max_tokens` | No | Response-token cap for each `/auto_review` request. Defaults to `512`; `0` disables the cap. Raise it (e.g. `2048`) when the review model thinks before answering |
| `code_max_tokens` | No | Response-token cap for normal chat and tool responses. Defaults to `0` (no cap) |
| `compile_workers` | No | Parallel job count `/build` passes to toolchains that support one (e.g. `make -j`, `meson compile -j`, `cargo --jobs`). Defaults to `0`, meaning unused: no job flag is passed and each toolchain falls back to its own default |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `sun_tzu_mandarin`, `sun_tzu_english`, `attila_the_hun`, `all` |
| `width` | No | Virtual terminal width in characters. Controls the layout canvas for `/show_file` output. Defaults to `512` |
| `banner` | No | Horizontal placement of the banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `theme` | No | Global default UI theme. Defaults to `classic`. Built-ins are `classic`, `oranguday`, `tokyonight`, `rosepine-moon`, and `auto`; user themes are loaded from `~/.orangu/themes/*.theme` |
| `auto_dark_theme` | No | Concrete theme used when `theme = auto` detects a dark terminal. Defaults to `classic` |
| `auto_light_theme` | No | Concrete theme used when `theme = auto` detects a light terminal. Defaults to `oranguday` |
| `drop_down` | No | Enable the autocomplete dropdown for slash commands. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `mouse` | No | Enable mouse capture in the terminal. When `true` (the default), the TUI handles mouse scroll and double-click. Hold **Shift** while clicking/dragging to do native text selection and copy. Set to `false` to disable all mouse handling |
| `workspaces` | No | Placement of the workspace tabs. Defaults to `top`. Options: `top`, `bottom`, `left`, `right`. See the Workspaces chapter |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure, blink an `orangu ●` progress title and ring the terminal bell when a `/auto_review` finishes. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `terminal` | No | Launch command used to open `$EDITOR` for terminal editors in a new window for `/open_file` (for example `xterm -e` or `kitty`). When unset, a terminal emulator is auto-detected |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |
| `system_prompt` | No | Override the base system prompt sent to the model. When empty (the default) orangu uses its built-in coding-assistant prompt. The discovered Agent Skills index is appended to whichever prompt is in effect |
| `model_verbosity` | No | Set the model's chattiness. Defaults to `normal`. Options: `terse`, `normal`, `verbose` |
| `review_confidence_threshold` | No | Minimum confidence score (0–100) for `/auto_review` findings; findings below this threshold are silently dropped. Defaults to `80`. Set to `0` to disable filtering |

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

Conversely, for the fastest reviews disable thinking on the server
(`orangu-server --reasoning-budget 0` together with
`--chat-template-kwargs '{"enable_thinking": false}'`) and keep the default
`512` — the cap then almost never binds and only guards against runaways.

### Themes

The global default theme is configured in `[orangu]`:

```ini
[orangu]
theme = classic
```

Built-in themes are shipped inside the binary: `classic`, `oranguday`, `tokyonight`, and `rosepine-moon`. The `auto` selector follows the detected terminal appearance, using `auto_dark_theme` and `auto_light_theme` internally; on a dark terminal it normally looks the same as `classic`.

Custom themes live in:

```text
~/.orangu/themes/<name>.theme
```

You can switch the current session with `/theme <name>`. That writes a session override to:

```text
~/.orangu/sessions/<UUID>/theme
```

Use `/theme default` (or `/theme global`) to remove the session override and return to the global `[orangu].theme`. The command completes built-in themes and user theme files. The startup option `--theme <name-or-path>` applies a theme only for that process and takes precedence over the global and session settings.

## Server sections

Each server is a named section. The section name is what `[orangu].server`
points to, and it carries the host information for that server:

```ini
[main-server]
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | `orangu-server` URL (its OpenAI-compatible API) |
| `model` | No | Model identifier used in chat completion requests. Overrides the general `[orangu].model` when set |
| `api_key` | No | API key sent as `Authorization: Bearer <key>` on every request to the server. Required when `orangu-server` runs with `--api-key` |
| `role` | No | A specific role this server fulfills. Valid roles are: `all` (default), `code`, `review`, `explorer`, and `embeddings`. If a specific subsystem needs a server and one is tagged with its role, it will use that server instead of the default. `embeddings` designates the server that embeds code for semantic `/search`; an `all` server also serves it, and search auto-enables when that endpoint responds at startup. Ignored behind a confirmed orangu-coordinator — it alone decides which model backs each role, so a single server section is enough there |

- At least one of `[orangu].model` or a server's own `model` must be set, so every server resolves to a non-empty model
- The endpoint may be configured either with or without `/v1`
- The client normalizes the endpoint internally before calling `/v1/chat/completions`
- Set `api_key` when the server requires authentication, for example `orangu-server --api-key <key>`. The key is sent as a bearer token on every request, including the `/v1/models` probe
- Each server section must use a unique `endpoint`; `http://x` and `http://x/v1` are treated as the same host
- Use `/server` to switch between the configured servers at runtime; Tab completion lists every server section
- Set `feedback = on` in `[orangu]` to show a green or red dot in the output window after each command completes, and to blink an `orangu ●` title and ring the terminal bell while/when a `/auto_review` runs and finishes

## Sample file

The distributed sample lives at:

```text
doc/etc/orangu.conf
```

It ships with `orangu-server` sections and a 30-minute timeout suitable for local tool-calling workloads.

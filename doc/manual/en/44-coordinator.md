\newpage

# Coordinator

`orangu-coordinator` is a small companion HTTP proxy for people who run local
models but only have the resources to keep **one** `orangu-server` process
resident at a time.

Instead of hand-starting `orangu-server` yourself before every
`orangu` session — and picking exactly one role to work in for that session —
point `orangu.conf` at the coordinator instead. It starts and stops
`orangu-server` on demand, swapping to whichever model a request actually
needs, so `/review`, the explorer subagent, semantic `/search`, and ordinary
chat can each use a different model without you ever running more than one
`orangu-server` at once.

This is purely optional. If you have enough VRAM to keep every role's model
loaded simultaneously, plain `orangu.conf` with one server section per role
(see the Configuration chapter) works exactly as before and needs no
coordinator.

## Why use it

Without a coordinator, using a different model per role means either running
several `orangu-server` processes side by side (one per port) — which most
single-GPU setups can't afford — or manually stopping and restarting
`orangu-server` yourself every time you switch tasks.

`orangu-coordinator` automates that: by default it owns one `orangu-server`
child process, and swaps it out for a different model the moment a request
needs one, entirely transparently to orangu. A machine with room for more —
a second device, or the CPU beside the card — can keep several resident
(`resident`), and moving between those swaps nothing.

The trade-off is latency, not capability: swapping pays the cost of a fresh
model load, so this suits a single-GPU/single-model machine well, but isn't
the right choice if you want every role to stay warm and instantly
responsive at the same time.

## Quick start

Generate a configuration interactively:

```sh
orangu-coordinator --init
```

It behaves exactly like `orangu-server --init`. It walks every
`[orangu-coordinator]` setting (including the shared `models` directory
every profile's `orangu-server` uses), then asks for a model, host, and port
role by role — `all` is mandatory, `code`/`review`/`explorer`/`embeddings`
are optional (leave the model prompt blank to skip one). It's written to
`~/.orangu/orangu-coordinator.conf` — tersely: only `host`/`port` and each
profile's `model` are always present, every other answer left at its default
is simply omitted.

Every prompt that has something to offer offers it inline, in grey, as you
type — press Right Arrow (or TAB, which also lists every candidate) to
accept:

- `models` completes real filesystem paths, and defaults to the Hugging Face
  cache (`~/.cache/huggingface/hub`) that `orangu-server` itself scans. A
  directory that doesn't exist yet is created.
- Every `host` prompt — the coordinator's own and each role's — completes
  over `all` (the default: every network interface), its `*` alias, and
  every address your machine's interfaces actually have, each labelled with
  the interface it belongs to.
- Every role's `model` prompt (once `models` is set) completes over every
  installed model, offering both the `NR` shorthand and the
  `MODEL:QUANT` label `orangu-server list` prints — `QUANT` included, so a
  repo you have in several quantizations resolves to the exact one you
  picked rather than whichever came first. An `NR` is only a shorthand for
  typing: it's written out as that row's `MODEL:QUANT` label, since a
  profile's `model` is read back for as long as the file lives and is also
  the id clients match against. A models directory holding exactly one model
  is taken for the mandatory `all` role without asking.

Nothing offered is required — typing anything else (a path, an undownloaded
Hugging Face spec) works too.

A minimal hand-written configuration looks like this:

```ini
[orangu-coordinator]
host = 127.0.0.1
port = 9000
models = /srv/models
startup_timeout = 180

[main]
model = ggml-org/gemma-4-E4B-it-GGUF

[explorer]
model = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
```

Neither profile sets `host`/`port` here — both default to `all:8100` (the
same default `orangu-server` itself uses: `all` means every network
interface, and the coordinator reaches the profile it started over
loopback). Sharing one address across profiles is fine even though they're
identical: profiles on one address are never running at the same time (see
`resident`), and swapping always fully stops the old one before starting
the new one, so
there's never a real conflict over the port. Before it stops the old one, the
coordinator lets every response still streaming from it finish (for up to two
minutes), so a swap requested by one client never cuts off another client's
answer. Set a profile's `host` to
`127.0.0.1` if you want its `orangu-server` reachable from this machine
only.

| Key | Section | Required | Description |
| :-- | :-- | :-- | :-- |
| `host` | `[orangu-coordinator]` | No | Host the proxy listens on: `all` (every interface), its `*` alias, or a literal address such as `127.0.0.1`. Defaults to `all` |
| `port` | `[orangu-coordinator]` | No | Port the proxy listens on. Defaults to `9000` |
| `models` | `[orangu-coordinator]` | Yes | Models directory forwarded to every profile's own `orangu-server` (`[orangu-server].models`) — one shared directory across every profile |
| `startup_timeout` | `[orangu-coordinator]` | No | Seconds to wait for a newly started `orangu-server` to answer `GET /v1/models` before giving up. Defaults to `180` |
| `max_body_bytes` | `[orangu-coordinator]` | No | Request/response body size cap in bytes. Defaults to `67108864` (64 MiB) |
| `idle_timeout` | `[orangu-coordinator]` | No | Seconds of inactivity before automatically unloading the active model to free system resources (RAM/VRAM). Disabled by default. |
| `resident` | `[orangu-coordinator]` | No | How many profiles' `orangu-server`s may run at once. Defaults to `1`: every swap stops the running one. Above one, see Keeping several models resident |
| `shutdown_token` | `[orangu-coordinator]` | No | Shared secret that enables the `GET /v1/coordinator/shutdown` endpoint. The caller must pass `?token=<value>` and connect from localhost. Disabled by default when absent. |
| `log_type` | `[orangu-coordinator]` | No | Where the coordinator's output goes: `console` (the default — exactly what it printed before the key existed) or `file`, which appends every line to `log_path` instead, stamped with a time and a level. Forwarded to every profile's `orangu-server`, which then logs into the same file — without the once-a-second progress line a request rewrites on a terminal, which a file has no use for |
| `log_path` | `[orangu-coordinator]` | No | The file `log_type = file` writes to. Defaults to `orangu-coordinator.log` in the directory the coordinator was started from; a leading `~` is expanded. Ignored under `log_type = console` |
| `role` | profile | No | Same roles as `orangu.conf`: `all` (default), `code`, `review`, `explorer`, `embeddings`. A section named after a role (`[code]`) is that role and needs no `role` key; set it only on a section with another name (`[qwen]`), and a value contradicting a role-named section is rejected. At least one profile must resolve to `all` — it's the fallback profile. Maps to `orangu-server`'s own `--all`/`--code`/`--review`/`--explorer`/`--embedding` flag |
| `model` | profile | Yes | A model spec — local `.gguf` path, `NR`/`MODEL` label, or `<user>/<model>[:quant]` Hugging Face repo — the same shape `orangu-server`'s own positional `MODEL` argument accepts |
| `host` | profile | No | Host this profile's `orangu-server` listens on — written straight into its generated config, so it takes the same `all`/`*`/address spellings. Defaults to `all` |
| `port` | profile | No | Port this profile's `orangu-server` listens on. Defaults to `8100` — the same default `orangu-server` itself uses |
| `backend` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].backend` (`auto`/`cpu`/`vulkan`/`metal`/`dx12`/`cuda`/`opencl`/`rocm`/`npu`) when set |
| `device` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].device` when set — which of the machine's devices it runs on |
| `slots` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].slots` when set |
| `context` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].context` when set — the tokens of context the server keeps room for on its device. The check before a swap counts this context's KV cache |
| `web` | profile | No | Forwarded to this profile's `orangu-server` as `[web].port` when set, exposing that profile's own web console while it's active |

Run it with:

```sh
orangu-coordinator --config ./orangu-coordinator.conf
```

Like `orangu.conf`, the config file defaults to `./orangu-coordinator.conf`,
then `~/.orangu/orangu-coordinator.conf`, so `--config` can usually be
omitted once it's in place.

## Flags

- `-q`/`--quiet` suppresses the startup banner, profile list, and shutdown
  message — useful when running it under a supervisor that captures stdout.
  Errors (a bad config, a port already in use, ...) still go to stderr
  regardless. It is about the console: a log file (`log_type = file`) is
  written in full whatever the flag says.
- `-d`/`--daemon` detaches from the terminal and runs in the background
  (Unix-only). With the console as its log there is nothing left to print
  to, so a daemon logs nothing at all — set `log_type = file` to have it log
  to a file, which is what that key is for. The config is loaded, the log
  file opened and the listen address bound *before* detaching, so a bad
  config, an unwritable `log_path` or a port already in use is still
  reported to your terminal rather than failing silently. There is no PID
  file: find the process with `pgrep -f orangu-coordinator` and stop it
  with `kill -INT <pid>` for the same graceful shutdown `Ctrl+C` triggers
  in the foreground.
- `-s`/`--shell-completions` prints a bash/zsh/fish/PowerShell completion script for
  the shell detected from `$SHELL` and exits — the same switch every orangu
  binary has (see the Shell completions chapter). It covers every flag
  here, with `-c`/`--config` completing files:

  ```sh
  # bash — add to ~/.bashrc:
  eval "$(orangu-coordinator -s)"
  # zsh — write once to your fpath directory:
  orangu-coordinator -s > ~/.zsh/completions/_orangu-coordinator
  # fish — add to ~/.config/fish/config.fish:
  orangu-coordinator -s | source
  # PowerShell — add to $PROFILE:
  orangu-coordinator -s | Out-String | Invoke-Expression
  ```
- `-h`/`--help` and `-V`/`--version` are also available, as on every orangu
  binary.

Running in the foreground (not `--daemon`) sets the terminal window/tab
title to `orangu-coordinator` for the life of the process, same as `orangu`
itself.

`orangu-coordinator` spawns a sibling `orangu-server` next to its own
executable by default, falling back to `orangu-server` resolved via `PATH`
if no sibling exists. Set `ORANGU_COORDINATOR_SERVER_BIN` to point it at a
specific `orangu-server` executable instead.

## Pointing orangu.conf at it

Once orangu confirms an endpoint is a coordinator, it alone decides which
model backs every role — a single, ordinary server section is enough:

```ini
[orangu]
server = main-server

[main-server]
endpoint = http://localhost:9000/
```

No `role = explorer`/`review`/`embeddings` sections are needed: `/review`,
`/auto_review`, the explorer subagent, and semantic `/search` all reuse this
same connection automatically, and the coordinator routes each to whatever
real model actually backs that role. `/model` and `/server` keep working
exactly as before.

## How it decides which model to use

Every request orangu sends already says what it's for — `/review` and
`/auto_review` ask for the `review` role, the explorer subagent asks for
`explorer`, semantic `/search` asks for `embeddings`, and ordinary chat and the
`write_tests`/`write_docs` subagents ask for `code` (or `all` if you haven't
configured a dedicated `code` profile).
The coordinator reads that and starts (or keeps running) whichever
`orangu-server` actually backs the requested role, stopping first whatever
it cannot run beside (with the default `resident = 1`, anything else that is
running). You never pick a model yourself when a coordinator is in charge —
that's the whole point.

### Keeping several models resident

With `resident` above one, a profile that has a place of its own starts
beside the ones already running instead of replacing them, and going back
to any of them restarts nothing. A profile has a place of its own when it
listens on its own `port` and runs on the CPU (`backend = cpu`) or on a
`device` no running profile names. Two profiles that name no device both
use the default one, and two servers on one card would each size their
memory as if they had it alone, so those still swap. When the set is full,
the least recently used profile is the one stopped. A server stopped either
way finishes what it is streaming and saves its slots first, exactly as a
swap does.

Before a profile starts — whether it replaces a running one or starts beside
them — the coordinator asks `orangu-server plan --deep --json` whether its
model can run here: its shards are present, this build knows its
architecture, and what it needs fits. What it needs is the weights every
token touches plus, when the profile sets `context`, that context's KV cache;
what the machine has is its memory plus its dedicated cards' memory; and the
models that stay running beside it count against that too. A start that does
not fit is refused with what it would take, and every running profile keeps
serving:

```text
'big' cannot run on this machine: it needs 40.2 GB, and beside 'chat' (30.1 GB)
that is more than this machine's 66.8 GB — 'chat' keeps serving
```

Only a definite answer refuses: a plan that cannot be had in time, or a
server too old to report what a model needs, lets the start go ahead.

```ini
[orangu-coordinator]
models = ~/.cache/huggingface/hub
resident = 2

[main]
model = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
device = 5500M

[explorer]
model = unsloth/Qwen3-1.7B-GGUF
backend = cpu
port = 8101
```

If a role you're using has no dedicated profile in
`orangu-coordinator.conf`, the coordinator falls back to the `all` profile
instead, so nothing errors out; it just means that role doesn't get its own
specialized model.

## File operations through the coordinator

`orangu-server`'s file-lifecycle endpoints (`/v1/create_file`,
`/v1/modify_file`, `/v1/move_file`, `/v1/delete_file`, `/v1/show_file`,
`/v1/create_directory`, `/v1/move_directory`, `/v1/delete_directory`) work
through the coordinator like everything else, because the coordinator is an
HTTP proxy: it forwards the request untouched and streams the reply back.
Their fields are documented in the *HTTP endpoints* chapter.

The coordinator has no workspace of its own and no `-w`/`--workspace` flag —
the file operation happens in the workspace of the `orangu-server` that
receives it. For a backend the coordinator started, that is the directory
the coordinator itself was started in, since the child inherits it. So start
the coordinator in the tree you want operated on.

These requests carry no `model`, so they resolve like any model-less
request: the currently active profile, then `all`. If nothing is running,
the first one pays the same cold start as any other request.

## Things to know while using it

- A request that needs a different model while another client's answer is
  still streaming waits for that answer to finish before the swap starts.
- The first request after a swap waits for the new model to finish loading —
  swapping to a role you haven't used yet in this session pays a real
  cold-load cost, same as starting `orangu-server` fresh.
- A swap saves the outgoing model's conversations to disk first, and the
  model's next start loads them back, so a conversation picks up where it
  was instead of reading its whole history again. Saving a long conversation
  adds a few seconds to the swap away from it (a 4 800-token one on a 3B
  model is a 1.1 GB file).
- Once connected to a coordinator, orangu shows "Automatic" for the model
  everywhere in the UI (the header banner, the status line, `/review` and
  `/auto_review`) instead of a specific model id, since the coordinator — not
  you — decides which model is actually loaded at any moment.
- If you use semantic `/search`, give your server section's `timeout` in
  `orangu.conf` enough headroom for a full cold load, not just a quick
  health check — otherwise `/search` may report itself unavailable simply
  because the very first detection attempt gave up too early.
- **Dynamic Hot-Reloading**: The coordinator watches `orangu-coordinator.conf` and hot-reloads changes automatically (polled every ~5 seconds) without needing a restart.
- **Fallback Routing**: If a requested profile fails to load (e.g. out of memory), the coordinator automatically falls back to starting the `all`-role profile rather than failing the request entirely.
- **Swap progress**: a streamed request (`"stream": true`) that has to wait
  more than a second for a swap gets its response straight away, opening
  with a progress line every two seconds — SSE comment lines such as
  `: orangu-coordinator: making 'code' active: loading org/model (12s) — <the
  server's last line of output>`, which every OpenAI-compatible client
  ignores — and then the model's answer. A swap that fails after that point
  arrives as an error event in the stream rather than a `502`. Any client can
  also read the same progress from `GET /v1/coordinator` under `activating`.
- **Crash recovery**: a profile whose `orangu-server` has stopped — it lost
  its GPU device to a driver reset (see the Server chapter), was OOM-killed,
  or was killed by hand — is simply started again on the next request. A
  request that was in flight when it stopped is sent again once, after that
  restart, rather than failed: nothing had been written back to you yet, so
  the request was still whole. What you see is a slow answer, not an error.
  A second failure is reported rather than retried.
- On shutdown (`Ctrl+C` or the internal `GET /v1/coordinator/shutdown` API), the coordinator gracefully stops whatever `orangu-server`
  process is currently active — or still starting up — so nothing is left
  running in the background.

See the Developer information chapter for the exact routing algorithm, the
`/v1/coordinator` protocol, and how model swapping works internally.

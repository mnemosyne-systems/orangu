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

`orangu-coordinator` automates that: it owns exactly one `orangu-server` child
process, and swaps it out for a different model the moment a request needs
one, entirely transparently to orangu.

The trade-off is latency, not capability: swapping pays the cost of a fresh
model load, so this suits a single-GPU/single-model machine well, but isn't
the right choice if you want every role to stay warm and instantly
responsive at the same time.

## Quick start

Generate a configuration interactively:

```sh
orangu-coordinator --init
```

This walks every `[orangu-coordinator]` setting (including the shared
`models` directory every profile's `orangu-server` uses), then asks for a
model, host, and port role by role — `all` is mandatory, `code`/`review`/
`explorer`/`embeddings` are optional (leave the model prompt blank to skip
one). It's written to `~/.orangu/orangu-coordinator.conf` — tersely: only
`host`/`port` and each profile's `model` are always present, every other
answer left at its default is simply omitted.

Both the `models` prompt and every role's `model` prompt offer inline
ghost-text suggestions and TAB completion: `models` completes real
filesystem paths, and each `model` prompt (once `models` is set) completes
over every installed model's user-facing label — the same label
`orangu-server list` prints in its `MODEL` column, not a raw filename or
its `NR` shorthand (a coordinator profile's `model` is written once and
read back indefinitely, so only the stable label is offered). Nothing
offered is required — typing anything else (a path, an undownloaded
Hugging Face spec) works too.

A minimal hand-written configuration looks like this:

```ini
[orangu-coordinator]
host = 127.0.0.1
port = 9000
models = /srv/models
startup_timeout = 180

[main]
role = all
model = ggml-org/gemma-4-E4B-it-GGUF

[explorer]
role = explorer
model = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
```

Neither profile sets `host`/`port` here — both default to `127.0.0.1:8100`,
which is fine even though they're identical: only one profile's
`orangu-server` is ever active at a time, and swapping always fully stops
the old one before starting the new one, so there's never a real conflict
over the port.

| Key | Section | Required | Description |
| :-- | :-- | :-- | :-- |
| `host` | `[orangu-coordinator]` | No | Host the proxy listens on. Defaults to `127.0.0.1` |
| `port` | `[orangu-coordinator]` | No | Port the proxy listens on. Defaults to `9000` |
| `models` | `[orangu-coordinator]` | Yes | Models directory forwarded to every profile's own `orangu-server` (`[orangu-server].models`) — one shared directory across every profile |
| `startup_timeout` | `[orangu-coordinator]` | No | Seconds to wait for a newly started `orangu-server` to answer `GET /v1/models` before giving up. Defaults to `180` |
| `max_body_bytes` | `[orangu-coordinator]` | No | Request/response body size cap in bytes. Defaults to `67108864` (64 MiB) |
| `idle_timeout` | `[orangu-coordinator]` | No | Seconds of inactivity before automatically unloading the active model to free system resources (RAM/VRAM). Disabled by default. |
| `shutdown_token` | `[orangu-coordinator]` | No | Shared secret that enables the `GET /v1/coordinator/shutdown` endpoint. The caller must pass `?token=<value>` and connect from localhost. Disabled by default when absent. |
| `role` | profile | No | Same roles as `orangu.conf`: `all` (default), `code`, `review`, `explorer`, `embeddings`. At least one profile must resolve to `all` — it's the fallback profile. Maps to `orangu-server`'s own `--all`/`--code`/`--review`/`--explorer`/`--embedding` flag |
| `model` | profile | Yes | A model spec — local `.gguf` path, `NR`/`MODEL` label, or `<user>/<model>[:quant]` Hugging Face repo — the same shape `orangu-server`'s own positional `MODEL` argument accepts |
| `host` | profile | No | Host this profile's `orangu-server` listens on. Defaults to `127.0.0.1` |
| `port` | profile | No | Port this profile's `orangu-server` listens on. Defaults to `8100` — the same default `orangu-server` itself uses |
| `backend` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].backend` (`auto`/`cpu`/`vulkan`/`cuda`/`opencl`/`rocm`) when set |
| `slots` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].slots` when set |
| `web` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].web` when set, exposing that profile's own web UI while it's active |

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
  regardless.
- `-d`/`--daemon` detaches from the terminal and runs in the background
  (Unix-only). It always implies `--quiet`. The config is loaded and the
  listen address is bound *before* detaching, so a bad config or a port
  already in use is still reported to your terminal rather than failing
  silently. There is no PID file: find the process with `pgrep -f
  orangu-coordinator` and stop it with `kill -INT <pid>` for the same
  graceful shutdown `Ctrl+C` triggers in the foreground.

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
`explorer`, semantic `/search` asks for `embeddings`, and ordinary chat asks
for `code` (or `all` if you haven't configured a dedicated `code` profile).
The coordinator reads that and starts (or keeps running) whichever
`orangu-server` actually backs the requested role, stopping anything else
that happens to be running first. You never pick a model yourself when a
coordinator is in charge — that's the whole point.

If a role you're using has no dedicated profile in
`orangu-coordinator.conf`, the coordinator falls back to the `all` profile
instead, so nothing errors out; it just means that role doesn't get its own
specialized model.

## Things to know while using it

- The first request after a swap waits for the new model to finish loading —
  swapping to a role you haven't used yet in this session pays a real
  cold-load cost, same as starting `orangu-server` fresh.
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
- On shutdown (`Ctrl+C` or the internal `GET /v1/coordinator/shutdown` API), the coordinator gracefully stops whatever `orangu-server`
  process is currently active — or still starting up — so nothing is left
  running in the background.

See the Developer information chapter for the exact routing algorithm, the
`/v1/coordinator` protocol, and how model swapping works internally.

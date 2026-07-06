\newpage

# Coordinator

`orangu-coordinator` is a small companion HTTP proxy for people who run local
models but only have the resources to keep **one** llama.cpp process resident
at a time.

Instead of hand-starting `llama-server` yourself before every
`orangu` session — and picking exactly one role to work in for that session —
point `orangu.conf` at the coordinator instead. It starts and stops
`llama-server` on demand, swapping to whichever model a request actually
needs, so `/review`, the explorer subagent, semantic `/search`, and ordinary
chat can each use a different model without you ever running more than one
`llama-server` at once.

This is purely optional. If you have enough VRAM to keep every role's model
loaded simultaneously, plain `orangu.conf` with one server section per role
(see the Configuration chapter) works exactly as before and needs no
coordinator.

## Why use it

Without a coordinator, using a different model per role means either running
several `llama-server` processes side by side (one per port) — which most
single-GPU setups can't afford — or manually stopping and restarting
`llama-server` yourself every time you switch tasks.

`orangu-coordinator` automates that: it owns exactly one `llama-server` child process, and swaps
it out for a different model the moment a request needs one, entirely
transparently to orangu.

The trade-off is latency, not capability: swapping pays the cost of a fresh
model load, so this suits a single-GPU/single-model machine well, but isn't
the right choice if you want every role to stay warm and instantly
responsive at the same time.

## Quick start

Generate a configuration interactively:

```sh
orangu-coordinator --init
```

This walks every `[orangu-coordinator]` setting, then asks for a `llamacpp`
command role by role — `all` is mandatory, `code`/`review`/`explorer`/
`embeddings` are optional (leave the prompt blank to skip one). Each command
is validated before being accepted, so the wizard can't write a config that
fails to load. It's written to `~/.orangu/orangu-coordinator.conf`.

A minimal hand-written configuration looks like this:

```ini
[orangu-coordinator]
host = 127.0.0.1
port = 9000
startup_timeout = 180

[main]
role = all
llamacpp = llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --host localhost --port 8100 --ctx-size 32768 -fa on --jinja

[explorer]
role = explorer
llamacpp = llama-server -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF --host localhost --port 8200 --ctx-size 65536
```

| Key | Section | Required | Description |
| :-- | :-- | :-- | :-- |
| `host` | `[orangu-coordinator]` | No | Host the proxy listens on. Defaults to `127.0.0.1` |
| `port` | `[orangu-coordinator]` | No | Port the proxy listens on. Defaults to `9000` |
| `startup_timeout` | `[orangu-coordinator]` | No | Seconds to wait for a newly started llama.cpp to answer `/v1/models` before giving up. Defaults to `180` |
| `max_body_bytes` | `[orangu-coordinator]` | No | Request/response body size cap in bytes. Defaults to `67108864` (64 MiB) |
| `role` | profile | No | Same roles as `orangu.conf`: `all` (default), `code`, `review`, `explorer`, `embeddings`. At least one profile must resolve to `all` — it's the fallback profile |
| `llamacpp` | profile | Yes | Full shell-style command line used to start llama.cpp for this profile, e.g. `llama-server -hf org/Model-GGUF --host localhost --port 8100 --ctx-size 32768`. There is no separate `model`, `host`, or `port` key — they're all read straight off this command line (`-hf`/`--hf-repo`/`-m`/`--model` for the model, `--host`/`--port` for where the coordinator proxies to). Leading `KEY=VALUE` tokens (e.g. `LLAMA_CACHE=/models llama-server ...`) are recognized and set as environment variables on the spawned process, and a leading `~`/`~/...` in any argument or value (e.g. `--slot-save-path ~/.orangu/llama-slots`) is expanded to the home directory — both are shell conveniences this command line would otherwise lose, since it's run directly rather than through a real shell |
| `api_key` | profile | No | Sent as `Authorization: Bearer <key>` on the coordinator's own requests to this profile's llama.cpp, if `llamacpp` starts it with `--api-key` |

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

## Pointing orangu.conf at it

Once orangu confirms an endpoint is a coordinator, it alone decides which
model backs every role — a single, ordinary server section is enough:

```ini
[orangu]
server = main-server

[main-server]
provider = llama.cpp
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
`llama-server` actually backs the requested role, stopping anything else
that happens to be running first. You never pick a model yourself when a
coordinator is in charge — that's the whole point.

If a role you're using has no dedicated profile in
`orangu-coordinator.conf`, the coordinator falls back to the `all` profile
instead, so nothing errors out; it just means that role doesn't get its own
specialized model.

## Things to know while using it

- The first request after a swap waits for the new model to finish loading —
  swapping to a role you haven't used yet in this session pays a real
  cold-load cost, same as starting `llama-server` fresh.
- Once connected to a coordinator, orangu shows "Automatic" for the model
  everywhere in the UI (the header banner, the status line, `/review` and
  `/auto_review`) instead of a specific model id, since the coordinator — not
  you — decides which model is actually loaded at any moment.
- If you use semantic `/search`, give your server section's `timeout` in
  `orangu.conf` enough headroom for a full cold load, not just a quick
  health check — otherwise `/search` may report itself unavailable simply
  because the very first detection attempt gave up too early.
- On shutdown (`Ctrl+C`), the coordinator stops whatever `llama-server`
  process is currently active — or still starting up — so nothing is left
  running in the background.

See the Developer information chapter for the exact routing algorithm, the
`/v1/coordinator` protocol, and how model swapping works internally.

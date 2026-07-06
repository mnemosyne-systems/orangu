# orangu-coordinator

`orangu-coordinator` is a small HTTP proxy for people who run local models but
only have the resources to keep **one** llama.cpp process resident at a time.
Instead of hand-starting `llama-server` yourself before every `orangu`
session, point `orangu.conf` at the coordinator; it starts and stops
`llama-server` on demand, swapping to whichever model a request actually
needs.

## How it works

`orangu.conf` can already tag each server section with a `role` (`all`,
`code`, `review`, `explorer`, `embeddings`) so different subsystems use a
different model. Normally that means running one llama.cpp process per role,
each on its own port. `orangu-coordinator` collapses those into a single
proxy address, and once orangu confirms it's talking to one, it stops relying
on `orangu.conf`'s own `role`/`model` tags for this at all: `/review`,
`/auto_review`, the explorer subagent, and embeddings detection each just
send the conventional role name (`review`, `explorer`, `embeddings`) as the
request's `model` field, and plain chat sends `code` (falling back to `all`
if no dedicated `code` profile is configured) — see [Pointing orangu.conf at
it](#pointing-oranguconf-at-it) below. Every request's JSON `model` field
tells the coordinator which model is wanted; it:

1. Looks at the incoming request's `model` field, if it has one, and
   matches it against a profile — first against the real model id in that
   profile's `llamacpp` command line (its `-hf`/`--hf-repo` or `-m`/`--model`
   argument; there's no separate `model` key to keep in sync), then against
   the profile's `role` name directly (so `orangu.conf` can just set `model
   = explorer` and never need to know the real backend model id at all).
2. If a `model` field was given but matched nothing configured, looks at
   what *kind* of request it is: `/v1/embeddings` implies the `embeddings`
   role regardless of what `model` named — a stale or unconfigured `model`
   field must not send an embeddings request to whatever chat model happens
   to be loaded. Failing that too, falls straight through to the `all`-role
   profile: an explicit-but-unmatched request for something specific must
   not silently inherit whatever unrelated role a prior request left active.
3. If no `model` field was given at all (bodyless requests like `/health`,
   `/props`, `/v1/models`), falls back to whichever profile is *currently
   active* — this is what makes those report on what's actually running
   rather than forcing a swap — and only falls back further to the
   `all`-role profile when nothing is active yet.
4. If a *different* profile's llama.cpp is currently running, stops it.
5. Starts the requested profile's llama.cpp (via its `llamacpp` command) if
   it isn't already running, and waits for `/v1/models` to answer.
6. Forwards the original request unchanged and streams the response back.

Only one `llama-server` process is ever alive under the coordinator. Swapping
pays the cost of a fresh model load, so this suits a single-GPU/single-model
machine, not a setup where every role should stay warm simultaneously.

On startup, the coordinator eagerly activates the `all`-role profile in the
background — it doesn't wait for a first request to start loading the
default model. This runs concurrently with the listener coming up, so `GET
/v1/coordinator` still answers instantly even while that model is loading; a
request for a different role that arrives before it finishes simply queues
behind the same startup sequence.

### Supported endpoints

Every OpenAI-compatible path orangu talks to, and every llama.cpp-native one,
is supported: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`,
`/health`, `/props`, `/slots`, `/metrics`, plus the coordinator's own
`/v1/coordinator`. All but the last are **pass-through**: the coordinator
picks a target profile, then forwards the request's method, path+query,
headers (minus hop-by-hop ones like `Connection`/`Host`/`Content-Length`),
and body to that profile's llama.cpp origin *exactly as received*, and
streams the response back byte-for-byte — it never inspects or rewrites the
actual request/response content, only decides which backend gets it.

Only one endpoint has a fixed role baked in; the rest are resolved
dynamically per the routing order in [How it works](#how-it-works):

| Endpoint | Path-implied role | How it actually resolves |
| :-- | :-- | :-- |
| `POST /v1/embeddings` | **`embeddings`** (fixed) | Always the `embeddings` profile, regardless of what `model` did or didn't say |
| `POST /v1/chat/completions` | *(none)* | Ambiguous by path alone — any role could be a chat request — so it's resolved purely by `model` (real model id or role name); an unmatched `model` falls straight to `all`, never to whatever's currently active |
| `GET /v1/models` | *(none)* | No `model` field is ever sent with it → currently active profile, then `all` |
| `GET /health` | *(none)* | Same as above |
| `GET /props` | *(none)* | Same as above |
| `GET /slots` | *(none)* | Same as above |
| `GET /metrics` | *(none)* | Same as above |
| `GET /v1/coordinator` | *(none — special)* | **Not pass-through.** Answered directly by the coordinator itself; see below |
| `POST /v1/coordinator/activate` | Whatever `model` names | **Not pass-through.** A pre-warming hint, answered directly; see below |

The "currently active" fallback for the model-less rows matters in practice:
without it, something like `/information` probing a server's `/health` would
itself force a swap back to `all`, killing whatever role a real request had
just switched to.

### Self-identification: `GET /v1/coordinator`

Neither llama.cpp nor a generic OpenAI-compatible server exposes this path, so
orangu (or any other client) can probe it to tell the three apart:

```sh
curl http://localhost:9000/v1/coordinator
```

```json
{
  "orangu_coordinator": true,
  "version": "0.11.0",
  "models": {
    "all": "bartowski/gemma-4-12B-it-GGUF",
    "code": "bartowski/gemma-4-12B-it-GGUF",
    "review": "bartowski/gemma-4-12B-it-GGUF",
    "explorer": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "embeddings": "bartowski/gemma-4-12B-it-GGUF"
  }
}
```

`models` reports the model each conventional role (`all`, `code`, `review`,
`explorer`, `embeddings`) currently resolves to — a role with no profile of
its own falls back to the `all`-role default's model, same as routing does.
This lets a caller see what `model` to put in `orangu.conf` for a given role
without needing its own copy of `orangu-coordinator.conf`.

It is answered directly by the coordinator itself — never proxied, and never
triggers starting a profile's llama.cpp — so it works even before any model
has been requested. orangu's `/information` command probes it as part of its
usual capability report.

### Pre-warming: `POST /v1/coordinator/activate`

A hint a caller can send *before* the request that actually needs a model,
so the coordinator can start swapping to it in parallel with whatever local
work (computing a diff, waiting on a user, ...) happens first, instead of
only starting the swap once the real request arrives:

```sh
curl -X POST http://localhost:9000/v1/coordinator/activate \
  -H 'Content-Type: application/json' -d '{"model": "review"}'
# {"activating":"review"}    (202 Accepted)
```

`model` is matched exactly like ordinary routing — a real model id or a role
name. The swap itself runs detached in the background and is not waited on:
the endpoint returns the instant it's kicked off, so it can never block on a
slow cold load, and the swap survives the caller disconnecting or not
reading the response at all. There is nothing to poll — the real request
that follows will simply find the model already active (or wait on the same
in-progress swap) exactly as it always does; if the hint's swap fails for
any reason, that real request retries it from scratch.

The one way this differs from ordinary routing: an unmatched `model` is a
`404`, not a silent fallback to `all` or "currently active" — those
fallbacks exist so a request that must be answered somehow always is, but an
explicit "activate X" call has no such obligation, and silently activating
the wrong thing would be worse than saying so.

orangu sends this hint at the start of `/review` and `/auto_review` (only
when it has already detected it's talking to a coordinator), naming the
`review` role, so cold-load latency is hidden behind diff collection and the
auto-review prestart screen instead of stalling the first review request.

## orangu-coordinator.conf

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
| `llamacpp` | profile | Yes | Full shell-style command line used to start llama.cpp for this profile, e.g. `llama-server -hf org/Model-GGUF --host localhost --port 8100 --ctx-size 32768`. Parsed like a shell command line (quoting works, e.g. `--chat-template-kwargs '{"enable_thinking": false}'`), but run directly — no shell is invoked, so stopping the profile always stops the real `llama-server` process, not just a wrapper. Leading `KEY=VALUE` tokens (the shell convention for setting one-off variables before a command, e.g. `LLAMA_CACHE=/models llama-server ...`) are recognized explicitly and set as environment variables on the spawned process, since there's no real shell to interpret them. A leading `~` or `~/...` in any argument or `KEY=VALUE` value (e.g. `--slot-save-path ~/.orangu/llama-slots`) is expanded to the home directory for the same reason — a real shell would otherwise have done this automatically (`~otheruser` forms are not supported). There is no separate `model`, `host`, or `port` key: they're all read straight off this command line, so there's exactly one place each is named — `-hf`/`--hf-repo` or `-m`/`--model` for the model (required — profiles *may* share a model, e.g. the same model configured once per role; `resolve_entry` breaks any resulting tie by profile name), `--host`/`--port` for where the coordinator proxies to (default to `127.0.0.1:8080`, same as llama.cpp itself, when omitted) |
| `api_key` | profile | No | Sent as `Authorization: Bearer <key>` on the coordinator's own requests to this profile's llama.cpp, if `llamacpp` starts it with `--api-key` |

Default lookup order for the config file, same as `orangu.conf`:

1. `./orangu-coordinator.conf`
2. `~/.orangu/orangu-coordinator.conf`

Run it with:

```sh
orangu-coordinator --config ./orangu-coordinator.conf
```

Running in the foreground (not `--daemon`) sets the terminal window/tab
title to `orangu-coordinator` for the life of the process, restoring it on
exit — same as `orangu` itself. This happens regardless of `--quiet`, since
it's a terminal escape sequence rather than console output.

Pass `-q`/`--quiet` to suppress the startup banner, profile list, and
shutdown message — useful when running it under a supervisor that captures
stdout. Errors (a bad config, a port already in use, ...) still go to
stderr regardless.

Pass `-d`/`--daemon` to detach from the terminal and run in the background
(Unix-only). It always implies `--quiet` — once detached there is no
terminal left to print to. The config is loaded and the listen address is
bound *before* detaching, so a bad config or a port already in use is still
reported to your terminal, with a non-zero exit code, rather than failing
silently in the background. There is no PID file: find the process with
`pgrep -f orangu-coordinator` (or similar) and stop it with `kill -INT
<pid>` for the same graceful shutdown `Ctrl+C` triggers in the foreground.

### Interactive setup

```sh
orangu-coordinator --init
```

Mirrors `orangu --init`: it walks every `[orangu-coordinator]` key showing its
default, then asks for a `llamacpp` command role by role — `all` is
mandatory, `code`/`review`/`explorer`/`embeddings` are skipped by leaving the
prompt blank. Each `llamacpp` command is validated the same way the loader
would (a parseable command line naming a model via `-hf`/`--hf-repo` or
`-m`/`--model`) before it's accepted, so the wizard can't write a config that
fails to load. It shows the resulting file and asks for confirmation before
writing `~/.orangu/orangu-coordinator.conf` (creating the directory if
needed, and overwriting any existing file).

## Pointing orangu.conf at it

Once orangu confirms an endpoint is a coordinator (the same `GET
/v1/coordinator` check the header status probe already does), it alone owns
every model/role decision — orangu.conf's own `role`/`model` machinery is
never consulted for anything. That means a single, ordinary server section
is enough:

```ini
[orangu]
server = main-server

[main-server]
provider = llama.cpp
endpoint = http://localhost:9000/v1
model = all
```

No `role = explorer`/`review`/`embeddings` sections are needed — `/review`,
`/auto_review`, the explorer subagent, and semantic `/search`'s embeddings
detection all reuse this same connection and each send the conventional role
name (`review`, `explorer`, `embeddings`) as `model` on their own requests,
regardless of what `model` this section names. Plain chat itself sends
`code` — orangu is fundamentally a coding assistant, so ordinary chat is the
`code` role in spirit, while `all` is reserved as the coordinator's required
universal fallback. The coordinator resolves each of those to whatever real
model actually backs it (falling back to `all` if it has none configured) —
see [How it works](#how-it-works). Renaming or swapping a model in
`orangu-coordinator.conf` never requires touching `orangu.conf` at all.

`/model` and `/server` keep working exactly as before; orangu never needs to
know a coordinator is involved, or what model any role actually loads.

(If you'd rather not rely on this and use a coordinator purely as a process
manager behind what still looks like several distinct servers, the old
pattern — one `orangu.conf` section per role, each pointed at the
coordinator's shared endpoint with `model` set to either the role name or
the real backend model id — still works exactly as documented before, but
only takes effect when talking to something that *isn't* confirmed to be a
coordinator. Behind a confirmed coordinator, those sections' own `role` and
`model` are ignored in favor of the behavior above.)

## Notes

- The coordinator does not manage remote/already-running llama.cpp instances;
  `llamacpp` is always a command it spawns and owns the lifecycle of.
- The first request after a swap waits for the new model to finish loading;
  size `startup_timeout` generously for large models.
- Once orangu confirms it's talking to a coordinator, it shows "Automatic"
  for the model everywhere in the UI (the header banner, the status line,
  `/review`/`/auto_review`) instead of a wire model id — since the
  coordinator, not orangu, decides which model is actually loaded, that id
  isn't a meaningful "what's running" answer. For the same reason, orangu's
  own startup/`/server`/`/reload` model auto-detection (which otherwise
  switches to whatever a server advertises, printing "Switched model from X
  to Y") is skipped entirely behind a confirmed coordinator.
- Forwarded requests have no fixed timeout of their own — generation can
  legitimately take as long as it takes, and the coordinator never cuts a
  response off partway through. Only the internal `/v1/models` health-check
  probe used while starting a profile has a short (5s) timeout, so a
  genuinely stuck/unreachable process is still detected and retried quickly
  without affecting real requests.
- Behind a confirmed coordinator, embeddings detection sends a real `POST
  /v1/embeddings` naming the `embeddings` role on the active connection,
  which — if a different role is currently active — makes the coordinator
  stop it and cold-load the embeddings model before answering, same as any
  other request. Give the active section's `timeout` (in `orangu.conf`)
  enough headroom for a full cold load, not just a quick health check, or
  semantic `/search` will be reported unavailable simply because detection
  gave up too early. This runs at startup and again whenever `/server` or
  `/reload` selects a new connection, so switching to (or away from) a
  coordinator mid-session re-evaluates search availability rather than
  leaving it fixed at whatever was true at launch.
- If a profile's llama.cpp crashes or exits before answering `/v1/models`
  (a bad flag, an out-of-memory kill, a GGML assertion, ...), the error
  includes the last 20 lines of its stdout/stderr, so the actual reason is
  visible alongside the exit status instead of just a bare signal number.
  Unless `--quiet`, that same output is also echoed live to the
  coordinator's own console as it's produced. The same diagnostic is printed
  (unless `--quiet`) if a profile crashes *after* becoming active — including
  mid-request, which a client only sees as a broken connection — the next
  time anything asks for it: the coordinator notices the process has died,
  logs its last captured output and exit status to its own console, and
  restarts it before serving that request.
- On shutdown (Ctrl+C), the coordinator stops whatever llama.cpp process is
  currently active — or still starting up, mid health-check — so nothing is
  left running in the background.

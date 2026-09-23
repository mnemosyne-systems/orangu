# orangu-coordinator

`orangu-coordinator` is a small HTTP proxy for people who run local models but
only have the resources to keep **one** `orangu-server` process resident at a
time. Instead of hand-starting `orangu-server` yourself before every `orangu`
session, point `orangu.conf` at the coordinator; it starts and stops
`orangu-server` on demand, swapping to whichever model a request actually
needs.

## How it works

`orangu.conf` can already tag each server section with a `role` (`all`,
`code`, `review`, `explorer`, `embeddings`) so different subsystems use a
different model. Normally that means running one `orangu-server` process per
role, each on its own port. `orangu-coordinator` collapses those into a
single proxy address, and once orangu confirms it's talking to one, it stops
relying on `orangu.conf`'s own `role`/`model` tags for this at all: `/review`,
`/auto_review`, the explorer subagent, and embeddings detection each just
send the conventional role name (`review`, `explorer`, `embeddings`) as the
request's `model` field, and plain chat sends `code` (falling back to `all`
if no dedicated `code` profile is configured) — see [Pointing orangu.conf at
it](#pointing-oranguconf-at-it) below. Every request's JSON `model` field
tells the coordinator which model is wanted; it:

1. Looks at the incoming request's `model` field, if it has one, and
   matches it against a profile — first against that profile's own `model`
   key, then against the profile's `role` name directly (so `orangu.conf`
   can just set `model = explorer` and never need to know the real backend
   model id at all).
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
4. If a *different* profile's `orangu-server` is currently running, stops it
   — **unless that process already serves what this profile asks for.** Two
   profiles that name the same model, listener, backend, slots and web port
   and differ only in `role` are one process: a role decides sampling
   defaults and whether reasoning is suppressed, and both of those travel
   with the request instead (see [Roles share a
   process](#roles-share-a-process)).
5. Starts the requested profile's `orangu-server` (with that profile's own
   role flag — `--all`/`--code`/`--review`/`--explorer`/`--embedding` — and
   model) if it isn't already running, and waits for `GET /v1/models` to
   answer.
6. Forwards the original request unchanged — plus an `x-orangu-role` header
   naming the resolved profile's role — and streams the response back.

Only one `orangu-server` process is ever alive under the coordinator.
Swapping pays the cost of a fresh model load, so this suits a
single-GPU/single-model machine, not a setup where every role should stay
warm simultaneously.

On startup, the coordinator eagerly activates the `all`-role profile in the
background — it doesn't wait for a first request to start loading the
default model. This runs concurrently with the listener coming up, so `GET
/v1/coordinator` still answers instantly even while that model is loading; a
request for a different role that arrives before it finishes simply queues
behind the same startup sequence.

### Supported endpoints

Every OpenAI-compatible path orangu talks to, and every one of
`orangu-server`'s own native endpoints, is supported: `/v1/models`,
`/v1/chat/completions`, `/v1/embeddings`, `/health`, `/props`, `/slots`,
`/metrics`, the file-lifecycle endpoints (`/v1/create_file`,
`/v1/modify_file`, `/v1/move_file`, `/v1/delete_file`, `/v1/show_file`,
`/v1/create_directory`, `/v1/move_directory`, `/v1/delete_directory`), plus
the coordinator's own `/v1/coordinator`. All but the last
are **pass-through**: the coordinator picks a target profile, then forwards
the request's method, path+query, headers (minus hop-by-hop ones like
`Connection`/`Host`/`Content-Length`), and body to that profile's
`orangu-server` origin *exactly as received*, and streams the response back
byte-for-byte — it never inspects or rewrites the actual request/response
content, only decides which backend gets it.

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
| `POST /v1/create_file`, `/v1/modify_file`, `/v1/move_file`, `/v1/delete_file`, `/v1/show_file`, `/v1/create_directory`, `/v1/move_directory`, `/v1/delete_directory` | *(none)* | Same as above — pass-through like everything else. See **File-lifecycle endpoints** below |
| `GET /v1/coordinator` | *(none — special)* | **Not pass-through.** Answered directly by the coordinator itself; see below |
| `POST /v1/coordinator/activate` | Whatever `model` names | **Not pass-through.** A pre-warming hint, answered directly; see below |
| `GET /v1/coordinator/shutdown` | *(none — special)* | **Not pass-through.** Cleanly terminates the proxy process and unloads the active model. Disabled unless `shutdown_token` is configured; requires `?token=<secret>` and a loopback source IP |

The "currently active" fallback for the model-less rows matters in practice:
without it, something like `/information` probing a server's `/health` would
itself force a swap back to `all`, killing whatever role a real request had
just switched to.

### File-lifecycle endpoints

`orangu-server`'s eight file endpoints (see its own documentation, and the
manual's **File-lifecycle API** section) pass straight through the
coordinator like any other request: it picks a profile, forwards the method,
path, headers and body untouched, and streams the reply back.

Two consequences worth being explicit about, since the coordinator is an
HTTP proxy and nothing more:

- **The workspace is the backend's.** The coordinator has no workspace of its
  own and no `-w`/`--workspace` flag. A file operation happens in the
  workspace of whichever `orangu-server` the request reached — which, for a
  server the coordinator started, is the directory the coordinator itself was
  started in (the child inherits it), unless that profile's own configuration
  says otherwise. Point the coordinator at the tree you want operated on.
- **They resolve like any model-less request**, so they go to the currently
  active profile, then `all`. If nothing is running yet, the first file
  request starts a backend — the same cold start any other request pays.

Everything else about them is unchanged by the hop: workspace confinement,
Git staging (`git add`/`git mv`/`git rm`, never a commit), the response
shapes, and the error codes are all the server's, byte-for-byte.

### Self-identification: `GET /v1/coordinator`

Neither `orangu-server` nor a generic OpenAI-compatible server exposes this
path, so orangu (or any other client) can probe it to tell the three apart:

```sh
curl http://localhost:9000/v1/coordinator
```

```json
{
  "orangu_coordinator": true,
  "version": "0.12.0",
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
triggers starting a profile's `orangu-server` — so it works even before any
model has been requested. orangu's `/information` command probes it as part
of its usual capability report.

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
models = /srv/models
startup_timeout = 180

[main]
model = ggml-org/gemma-4-E4B-it-GGUF

[explorer]
model = unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
backend = vulkan
slots = 4
```

Neither profile above sets `host`/`port` — both fall back to `all:8100`
(the same default `orangu-server` itself applies), which is fine (see
below); set them explicitly only if a profile needs something different.

| Key | Section | Required | Description |
| :-- | :-- | :-- | :-- |
| `host` | `[orangu-coordinator]` | No | Host the proxy listens on: `all` (every network interface), its `*` alias, or a literal interface address such as `127.0.0.1`. Defaults to `all`, the same default and the same spellings `orangu-server`'s own `host` uses |
| `port` | `[orangu-coordinator]` | No | Port the proxy listens on. Defaults to `9000` |
| `models` | `[orangu-coordinator]` | Yes | Models directory forwarded to every profile's own `orangu-server` (its `[orangu-server].models` key) — one shared directory across every profile, same as pointing plain `orangu-server` at one `models` directory. Supports a leading `~`/`~/...` |
| `startup_timeout` | `[orangu-coordinator]` | No | Seconds to wait for a newly started `orangu-server` to answer `GET /v1/models` before giving up. Defaults to `180` |
| `max_body_bytes` | `[orangu-coordinator]` | No | Request/response body size cap in bytes. Defaults to `67108864` (64 MiB) |
| `idle_timeout` | `[orangu-coordinator]` | No | Seconds of inactivity before automatically unloading the active model to free system resources (RAM/VRAM). Disabled by default. |
| `shutdown_token` | `[orangu-coordinator]` | No | Shared secret that enables the `GET /v1/coordinator/shutdown` endpoint. The caller must pass `?token=<value>` and connect from localhost. Disabled by default when absent. |
| `log_type` | `[orangu-coordinator]` | No | Where the coordinator's output goes: `console` (the default — exactly what it printed before the key existed) or `file`, which appends every line to `log_path` instead, with a timestamp and level in front. See [Logging to a file](#logging-to-a-file) |
| `log_path` | `[orangu-coordinator]` | No | The file `log_type = file` writes to. Defaults to `orangu-coordinator.log` in the directory the coordinator was started from; a leading `~` is expanded. Ignored under `log_type = console` |
| `role` | profile | No | Same roles as `orangu.conf`: `all` (default), `code`, `review`, `explorer`, `embeddings`. A section named after a role (`[code]`) is that role and needs no `role` key; set it only on a section with another name (`[qwen]`), and a value contradicting a role-named section is rejected. At least one profile must resolve to `all` — it's the fallback profile. Maps to `orangu-server`'s own `--all`/`--code`/`--review`/`--explorer`/`--embedding` flag; anything else is rejected at load time |
| `model` | profile | Yes | A model spec in the same shape `orangu-server`'s own positional `MODEL` argument accepts: a local `.gguf` path, an `NR`/`MODEL` label already under the shared `models` directory, or a `<user>/<model>[:quant]` Hugging Face repo (fetched on first start if not already cached). This is the model id a client request's `model` field matches against — profiles *may* share one, e.g. the same model configured once per role; `resolve_entry` breaks any resulting tie by profile name |
| `host` | profile | No | Host this profile's `orangu-server` listens on, written verbatim into its generated config so it takes the same `all`/`*`/address spellings. Defaults to `all`. The coordinator reaches a wildcard-bound profile over loopback |
| `port` | profile | No | Port this profile's `orangu-server` listens on. Defaults to `8100` — the same default `orangu-server` itself uses |
| `backend` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].backend` (`auto`/`cpu`/`vulkan`/`metal`/`cuda`/`opencl`/`rocm`/`npu`) when set. Defaults to `orangu-server`'s own default (`auto`) |
| `slots` | profile | No | Forwarded to this profile's `orangu-server` as `[orangu-server].slots` when set. Defaults to `orangu-server`'s own role-based default |
| `web` | profile | No | Forwarded to this profile's `orangu-server` as `[web].port` when set, exposing that profile's own web console on the given port while it's active. Off by default |

Each profile's own `orangu-server` is started with a small, coordinator-
generated config file (`~/.orangu/coordinator/servers/<profile-name>.conf`,
overwritten on every start) carrying its `models`/`host`/`port`, whichever
of `backend`/`slots`/`web` were set, and — under `log_type = file` — the
coordinator's own `log_type`/`log_path` — inspect that file directly to see
exactly what a given profile's `orangu-server` last ran with.

Every profile defaulting to `all:8100` — the same address, not a distinct
one per role — is intentional and safe: at most one profile's
`orangu-server` is ever alive under the coordinator (its whole invariant),
and swapping to a different profile always fully stops whichever one is
currently running before starting the new one, so the new process never
races the old one for the same port. Give a profile its own explicit
`host`/`port` only if you want its `orangu-server` somewhere specific — a
`host` of `127.0.0.1` to keep it off the network entirely, or a distinct
`port` to reach it directly, bypassing the coordinator.

Default lookup order for the config file, same as `orangu.conf`:

1. `./orangu-coordinator.conf`
2. `~/.orangu/orangu-coordinator.conf`

Run it with:

```sh
orangu-coordinator --config ./orangu-coordinator.conf
```

`orangu-coordinator` spawns a sibling `orangu-server` next to its own
executable by default (the common case: both binaries come from the same
build), falling back to `orangu-server` resolved via `PATH` if no sibling
exists. Set `ORANGU_COORDINATOR_SERVER_BIN` to point it at a specific
`orangu-server` executable instead.

Running in the foreground (not `--daemon`) sets the terminal window/tab
title to `orangu-coordinator` for the life of the process, restoring it on
exit — same as `orangu` itself. This happens regardless of `--quiet`, since
it's a terminal escape sequence rather than console output.

Pass `-q`/`--quiet` to suppress the startup banner, profile list, and
shutdown message — useful when running it under a supervisor that captures
stdout. Errors (a bad config, a port already in use, ...) still go to
stderr regardless. It is about the console: a log file (`log_type = file`)
is written in full whatever the flag says.

Pass `-d`/`--daemon` to detach from the terminal and run in the background
(Unix-only). With the console as its log there is nothing left to print to,
so a daemon logs nothing at all — set `log_type = file` to have it log to a
file instead, which is the case that setting exists for (see [Logging to a
file](#logging-to-a-file)). The config is loaded, the log file opened and
the listen address bound *before* detaching, so a bad config, an unwritable
`log_path` or a port already in use is still reported to your terminal,
with a non-zero exit code, rather than failing silently in the background.
There is no PID file: find the process with `pgrep -f orangu-coordinator`
(or similar) and stop it with `kill -INT <pid>` for the same graceful
shutdown `Ctrl+C` triggers in the foreground.

### Logging to a file

```ini
[orangu-coordinator]
models = /srv/models
log_type = file
log_path = /var/log/orangu/coordinator.log
```

`log_type = file` sends everything the coordinator would have printed —
the startup banner and profile list, `reloaded configuration ...`, the
profile swaps, idle unloads and crash reports, the shutdown line, and the
output of each profile's `orangu-server` — to `log_path` instead of the
terminal, appending to the file if it exists and creating it (and its
directory) if it doesn't. `log_path` defaults to `orangu-coordinator.log`
in the directory the coordinator was started from, so `log_type = file` on
its own is enough. Each line is stamped:

```text
2026-09-16 23:36:29 INFO  orangu-coordinator 1.4.0 listening on 127.0.0.1:9000
2026-09-16 23:36:29 INFO    main: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
2026-09-16 23:36:41 INFO  Model      unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (gemma4 arch, Vulkan, 30 layers, 32768 ctx)
2026-09-16 23:36:41 INFO  Mode       all
2026-09-16 23:37:02 INFO  orangu-server: [slot 0] prompt 41 tokens in 0.31s (132.26 tok/s), generated 64 tokens in 3.10s (20.65 tok/s)
```

The `Model`/`Mode` and `[slot 0]` lines are the `orangu-server`'s own: the
coordinator forwards its `log_type`/`log_path` into the config it generates
for every profile, so a profile's server appends to the same file rather
than printing into the pipe the coordinator reads — with one difference from
its console output, which is that the progress line a request rewrites once
a second on a terminal (`\r`, no newline) is not written; a file gets each
request's completed line and nothing in between. The keys are read once, at
startup: a reload that changes them takes effect on the next start.

`log_type = console` (or no `log_type` at all) prints to the terminal.

Pass `-s`/`--shell-completions` to print a bash/zsh/fish/PowerShell completion script
for the shell detected from `$SHELL` and exit — the same switch every orangu
binary has. It covers every flag above, with `-c`/`--config` completing
files:

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

### Interactive setup

```sh
orangu-coordinator --init
```

Behaves the same way `orangu-server --init` does: it walks every
`[orangu-coordinator]` key showing its default (including the `models`
directory, which defaults to the Hugging Face cache
`~/.cache/huggingface/hub` and is created if it isn't there yet, and
`log_type`, whose `log_path` is only asked for on `file`), then asks
for a model, host, and port role by role — `all` is mandatory, `code`/
`review`/`explorer`/`embeddings` are skipped by leaving the model prompt
blank. It shows the resulting file and asks for confirmation before writing
`~/.orangu/orangu-coordinator.conf` (creating the directory if needed, and
overwriting any existing file).

The written file is kept terse: only `host`/`port` (in
`[orangu-coordinator]`) and each profile's `model` are always present —
every other answer left at its default is simply omitted, since the loader
already falls back to the exact same value on its own. The one exception is
a `file` log's `log_path`, written even at its default: that default is
`orangu-coordinator.log` in whatever directory the coordinator is started
from, and the file the wizard showed is the file the config should keep
naming.

Every prompt with something to offer shows it inline as grey ghost text
while you type, and completes it on TAB (which also lists every candidate),
exactly as `orangu-server --init` does:

- `models` completes real filesystem paths.
- `log_type` completes over `console` and `file`, ghosting `console`. On
  `file`, `log_path` completes real filesystem paths as you type and ghosts
  its default — `orangu-coordinator.log` in the current directory — on the
  empty line.
- Every `host` prompt — `[orangu-coordinator]`'s own and each profile's —
  completes over `all`, its `*` alias, and every address this machine's
  interfaces actually have, each annotated with the interface it belongs to
  in the TAB list. An empty line ghosts `all`.
- Each `model` prompt (once `models` is set) completes over every installed
  model: both its `NR` shorthand and its `MODEL:QUANT` label — the same
  pairing `orangu-server list` prints (e.g.
  `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M`), not the raw on-disk filename. The
  quantization is part of the offered label on purpose: a repo present in
  several quantizations otherwise lists one name once per row and resolves
  to whichever came first. Only the labels are ghosted; an `NR` is a
  shorthand to type, not a model to preview.
- An `NR` answer is written into the file as that row's own `MODEL:QUANT`
  label. A coordinator profile's `model` is read back for as long as the
  file lives *and* is the literal string clients match against, so a
  scan-order-dependent digit must never be persisted — it would silently
  start naming a different model as soon as the `models` directory changes.
- A `models` directory holding exactly one model is taken for the mandatory
  `all` role without asking, echoed as `model/all: <label>`.

No prompt requires typing an offered value — a local path or a
not-yet-downloaded `<user>/<model>[:quant]` spec is equally valid.


## The server belongs to the coordinator

At most one `orangu-server` is alive under a coordinator, and its lifetime is
the coordinator's to end. Three things enforce that, because each covers a
case the others cannot:

- **`Ctrl+C`, `SIGTERM`, or `POST /shutdown`** run the shutdown path, which
  stops the child and waits for it to actually exit.
- **A `SIGKILL`** runs nothing at all, so the child asks the kernel to signal
  it when its parent goes away (`PR_SET_PDEATHSIG`, Linux). A server started
  by hand is unaffected — this is set per child, between fork and exec.
- **A server already serving this profile is adopted**, not competed with.
  Before starting one, the coordinator asks `/props` what is on the address:
  when the `model` is exactly the spec the profile names and the `role` is one
  that can serve it, that server *is* what starting one would produce, so it is
  used as-is — no second process, no reload, no wait.

  ```
  adopting the orangu-server already serving 'all' at http://127.0.0.1:8100 (process 716)
  ```

  The comparison is deliberately exact. Adopting the wrong server would be
  worse than starting a doomed one, because it is silent: requests answered by
  another model with nothing saying so. A profile naming a local path or a
  catalogue number will not match the label a server reports and simply starts
  its own.

  An adopted server is stopped like any other when a swap needs the port. A
  server that must not be touched belongs on its own `port`, where no swap will
  ask for it.

- **An `orangu-server` serving something else** has the address taken from
  it. That is the same act as swapping away from an adopted server, and who
  started the incumbent changes nothing about it — the address belongs to the
  profile by configuration:

  ```
  taking http://127.0.0.1:8100 for 'all': process 22103 is serving
  ggml-org/embeddinggemma-300M-GGUF in embedding mode, which this profile does
  not ask for
  ```

  Only against a pid the kernel agrees is an `orangu-server` (`/proc/<pid>/comm`
  on Linux). The pid comes from a `/health` answer — a network fact — and it is
  about to be signalled; a server reporting someone else's pid would otherwise
  have the coordinator stop an unrelated process for it.

- **Anything else already listening** on the profile's port is *not* touched,
  and is reported rather than trusted. The readiness probe is `GET /health`,
  which names the process answering, and a pid that is not the child just
  spawned is reported:

  ```
  'review' cannot serve http://127.0.0.1:8100: process 4159676 is already
  listening there, and it is not the orangu-server just started (process
  4159810). Stop it — an orangu-server or orangu-coordinator left running from
  earlier is the usual cause — or give this profile its own `port`.
  ```

A leftover server answers at the address instantly, so a probe that only
asked whether anything answers would proxy every request to a process the
coordinator did not start, serving whatever model *that* one holds. Checking
the pid is what rules this out.

## Roles share a process

A common configuration gives `all`, `code`, `review` and `explorer` the same
model file, differing only in `role`. Three of them share a process; **`review`
does not** — a review is run by an `orangu-server --review`, so the `Mode` row
on its banner says what is serving it.

`ensure_active` keeps the running process whenever the
requested profile differs from it in nothing but `role`, and the proxy sends
the resolved role along with each request as `x-orangu-role`.
`orangu-server` reads that header for the two things a role actually decides
— the sampling defaults it starts from, and whether reasoning is suppressed
— and ignores it for anything it cannot change: a header cannot make an
`--embedding` server answer chat, because which endpoints work is a property
of the model that was loaded.

Anything else — a different model, port, backend, `slots` or `web` — is a
different process and swaps. So is `review`, so that a review always runs
on a server started in that role. So is `embeddings`, in either direction,
however identical the rest of the profile: an `--embedding` server refuses the
generation endpoints outright, so sharing a process with a chat role would
answer chat with `501` rather than swapping to something that can serve it.

The visible effect, on four same-model profiles:

| | before | after |
| :-- | --: | --: |
| a request that switches role | ~11 s | ~0.4 s |
| model loads across seven such requests | 5 | 1 |
| cached prompt tokens surviving a switch | none | all of them |

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

- The coordinator does not manage remote/already-running `orangu-server`
  instances; each profile's `orangu-server` is always a process it spawns
  and owns the lifecycle of itself.
- **Dynamic Hot-Reloading**: The coordinator watches `orangu-coordinator.conf` and hot-reloads changes automatically (polled every ~5 seconds) without needing a restart. Any change to the active profile's settings — not just its model — restarts its `orangu-server`.
- **Fallback Routing**: If a requested profile fails to load (e.g. out of memory), the coordinator automatically falls back to starting the `all`-role profile rather than failing the request entirely.
- **Crash recovery and one retry**: `ensure_active` restarts a profile whose
  child process it finds dead, so a stopped `orangu-server` costs the next
  request a cold load rather than an error. That check can't close the
  window where the child is alive when checked and gone microseconds later
  — which is exactly what `orangu-server` exiting on a lost GPU device does
  under a request — so a forwarded request that fails to *reach* the child
  is retried exactly once. It's safe because nothing has been streamed back
  to the caller at that point; a second failure is reported as a `502`
  rather than retried again.
- **The retry asks whether the profile answers, not whether it is running**:
  `ensure_reachable` (not `ensure_active`) precedes it, because
  `Child::try_wait` lags. A `SIGKILL`ed child is gone immediately but is not
  *reported* as exited until tokio's `SIGCHLD` handling has run, so a retry
  that consults it milliseconds after the connection failed is told the dead
  process is fine and sends the request straight back into the same closed
  port — measured against a real profile, not theorized. `ensure_reachable`
  instead probes `GET /v1/models` with the same short health-check timeout:
  an answer means the failure was transient and nothing is restarted (a
  restart would throw away a working process and every other request on it);
  no answer means the child is gone whatever `try_wait` believes, and it is
  replaced before the request is re-sent.
- **A lost GPU device is a recognized exit, not a crash**: `orangu-server`
  exits with `75` (`EX_TEMPFAIL`) when a driver reset destroys its GPU
  device, after telling the caller so in one sentence. The coordinator names
  the same status (`process::SERVER_EXIT_DEVICE_LOST`) and reports the
  restart as the recovery it is — `'main' exited after losing its GPU device
  (a driver reset); restarting it on a fresh device` — rather than as an
  unexplained crash with an output tail attached.
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
  response off partway through. Only the internal `GET /v1/models`
  health-check probe used while starting a profile has a short (5s)
  timeout, so a genuinely stuck/unreachable process is still detected and
  retried quickly without affecting real requests.
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
- If a profile's `orangu-server` crashes or exits before answering
  `GET /v1/models` (a bad model spec, an out-of-memory kill, ...), the
  error includes the last 20 lines of its stdout/stderr, so the actual
  reason is visible alongside the exit status instead of just a bare signal
  number. Unless `--quiet`, that same output is also echoed live to the
  coordinator's own console as it's produced — or into its log file, under
  `log_type = file`, where a profile's `orangu-server` writes its own lines
  directly and only what it prints around them (its `error:` line, most
  usefully) arrives through the coordinator. The same diagnostic is printed
  (unless `--quiet`) if a profile crashes *after* becoming active — including
  mid-request, which a client only sees as a broken connection — the next
  time anything asks for it: the coordinator notices the process has died,
  logs its last captured output and exit status to its own console, and
  restarts it before serving that request.
- On shutdown (Ctrl+C), the coordinator stops whatever `orangu-server`
  process is currently active — or still starting up, mid health-check —
  so nothing is left running in the background.

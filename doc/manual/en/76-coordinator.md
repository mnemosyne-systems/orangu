\newpage

## Coordinator internals

`orangu-coordinator` (`src/bin/orangu-coordinator/`) is a second binary in
the same Cargo package as `orangu`, built around a single invariant: **at
most one `orangu-server` child process is alive at any time.**

Everything else — the HTTP proxy, the self-identification endpoint, the pre-warming
hint, client-side integration in `orangu` itself — exists to make swapping
that one process safe and (mostly) transparent.

### Architecture

`process::Coordinator` (`src/bin/orangu-coordinator/process.rs`) owns:

- `config: CoordinatorConfiguration` — the parsed `orangu-coordinator.conf`,
  keyed by profile name, each a `CoordinatorLlmEntry { name, role, model,
  host, port, backend, slots, web }` — every field an explicit, individually-
  parsed config key (`config.rs`'s `parse_llm_profiles`); `role` is
  validated against `role_server_flag` at load time, since it has to map to
  one of `orangu-server`'s own `--all`/`--code`/`--review`/`--explorer`/
  `--embedding` flags for the profile to be startable at all.
- `active: Mutex<Option<ActiveProcess>>` — the currently running process, if
  any, plus its rolling output tail (`OutputTail = Arc<Mutex<VecDeque<String>>>`,
  capped at 20 lines) kept for the process's whole lifetime, not just while
  starting.
- `current_pid: AtomicU32` — set the instant a process is spawned, *before*
  its health check even begins, and cleared once it's known to have
  stopped. `active` is held locked for an entire start sequence (which can
  take up to `startup_timeout`); `shutdown()` must not have to wait on that
  same lock just to kill a still-starting process, so the PID is tracked
  separately and only ever locked briefly.
- `http_client: reqwest::Client` — shared by both the health-check probe and
  request forwarding, with **no default timeout**. This client is used to
  proxy real, potentially long-running generation requests; only the
  internal `GET /v1/models` health-check applies its own short, explicit
  per-request timeout (`HEALTH_CHECK_TIMEOUT`, 5s). Giving the whole client
  a default timeout instead was a real bug: any generation slower than it
  got its connection torn down mid-stream, surfacing to the caller as a bare
  "unexpected EOF during chunk size line" rather than a clear error.
- `server_binary_override: Option<PathBuf>` — which `orangu-server`
  executable to spawn, read once by `main` from
  `ORANGU_COORDINATOR_SERVER_BIN` at startup (never re-read per spawn, and
  never a bare env lookup inside `start` itself — this also lets tests
  inject a stand-in executable deterministically without racing parallel
  test threads over a shared process-wide environment variable). `None`
  falls back to `Coordinator::resolve_server_binary`'s sibling-binary/`PATH`
  search.

### File-lifecycle endpoints

Nothing in the coordinator handles these: `orangu-server` mounts them
(`orangu::files_http::router`, over the shared `orangu::files`
implementation), and the coordinator forwards them through `proxy::proxy`
like any other path. `implied_role_for_path` returns `None` for them — only
`/v1/embeddings` has a role baked into its path — so they resolve to the
currently active profile, then `all`.

Deliberately not handled locally, even though the operations are available
to the coordinator as a library: it has no workspace to act in. Giving it
one would mean two workspaces in one deployment (the coordinator's and each
backend's) and a client unable to tell which one a request reached. The
proxy stays a proxy.

### `GET /v1/coordinator`

A fixed, side-effect-free identity marker (`proxy::coordinator_info`),
answered directly and never proxied — it must work even before any profile
has been activated. Neither `orangu-server` nor a generic OpenAI-compatible
server exposes this path, so a client can probe it to tell the three apart:

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

`models` reports what `CoordinatorConfiguration::models_by_role` resolves
each conventional role to — a role with no profile of its own falls back to
the `all`-role default's model. `orangu`'s side of this lives in
`src/llm/coordinator.rs`: `probe_coordinator` does the GET and parses the
body, returning `Option<Vec<String>>` (`Some` only when
`orangu_coordinator` is `true`). It's shared library code (not
bin-specific) because both the `orangu` binary's per-cycle header-status
probe and library code with no `HeaderStatus` of its own (the explorer
subagent, which reloads its own config from disk) need it.

**Asked once, not every cycle.** The client's status refresh runs on a timer —
every 60 s idle, every 500 ms while the startup git sync is in flight — and
this path exists on nothing but a coordinator, so probing it each cycle is a
404 each cycle against a plain `orangu-server` (and against a hosted
OpenAI-compatible endpoint). `models::probe_server_reachability` remembers what
each endpoint answered: a known-plain one skips the probe entirely and lets
`GET /v1/models` serve as both the model list and the health check, halving the
requests a poll costs. Over one short session that is 4 coordinator probes down
to 1, and the saving grows with session length.

The memo is dropped the moment an endpoint stops answering, so a server
restarted as a coordinator — or a coordinator that simply was not up yet when
the client started — is identified afresh rather than assumed to be whatever
was there before. A *confirmed* coordinator is still probed every cycle, since
for it this endpoint is the health check and `/v1/models` is deliberately never
called (see above).

`/server` drops it too, in both its forms: listing the servers re-opens the
question for the active endpoint, and `/server <name>` for the one it selects —
including when that is the server already in use, matching the model
re-detection that same command triggers. Listing or selecting a server is the
moment a user is asking about the connection rather than working through it,
and the moment the answer is most likely to have just changed. The memo is
there to keep an unchanging answer from being re-asked on a timer, not to
outlast a direct question about it; `/server` is how you say "look again".

### `POST /v1/coordinator/activate`

A pre-warming hint (`proxy::activate`) a caller can send *before* the
request that actually needs a model. `model` is matched exactly like
ordinary routing (see below). Critically, the swap is **spawned detached
and not awaited** — the handler resolves the target entry, `tokio::spawn`s
`ensure_active` on it, and returns `202 Accepted` immediately. This must
work this way: the whole point is to start the swap ahead of the real
request, so it has to survive the caller disconnecting early or never
reading the response, and it must never itself block on a slow cold load.

Unlike ordinary routing, an unmatched `model` here is a `404`, not a silent
fallback to `all` or "currently active": those fallbacks exist so a request
that must be answered *somehow* always is, but an explicit "activate X" call
has no such obligation — silently activating the wrong thing would be worse
than saying so (`Coordinator::match_hint`, which does only the model-id/role
match, with none of `select_entry`'s further fallback steps).

`orangu` fires this hint from `spawn_coordinator_activation_hint`
(`src/bin/orangu/models.rs`) at the start of `/review`/`/auto_review`,
naming the `review` role, so cold-load latency is hidden behind local work
(diff collection, the auto-review prestart screen) instead of stalling the
first real request. It's fire-and-forget from `orangu`'s side too: any
failure is ignored, since the real request that follows triggers the same
swap on its own if the hint didn't get there first.

### Request routing: `select_entry`

Every other endpoint goes through `proxy::proxy`, which extracts the JSON
body's `model` field (`extract_model_field`) and the path-implied role
(`implied_role_for_path` — currently only `/v1/embeddings` implies
`embeddings`), then calls `Coordinator::resolve_entry`, which delegates to
the pure, unit-tested `select_entry` (`process.rs`):

1. If a `model` field was given, match it against a profile's real model id
   first, then its `role` name (`match_hint`). Ties (more than one profile
   sharing a model or role) resolve to the lexicographically smallest
   profile name, so the choice is deterministic rather than depending on
   `HashMap` iteration order.
2. If a `model` field was given but matched nothing, try the path-implied
   role; failing that, go **straight to the `all` default** — not to
   "currently active". An explicit-but-unmatched hint is a deliberate
   request for something specific; silently substituting whatever unrelated
   role a prior request left running would be surprising and
   non-deterministic.
3. If *no* `model` field was given at all (bodyless requests: `/health`,
   `/props`, `/v1/models`, `/slots`, `/metrics`), fall back to whichever
   profile is currently active — this is what lets those report on what's
   actually running rather than forcing a swap — and only fall back further
   to `all` when nothing is active yet.

Once an entry is chosen, `ensure_active` (also `process.rs`) does the actual
lifecycle work: if it's already the active entry and its process is still
alive (checked via `try_wait`), reuse it; otherwise stop whatever's running
(clearing `current_pid` *before* reaping, so a concurrent `shutdown` never
targets a stale, possibly-recycled PID) and `start` the requested entry —
write that profile's own generated `orangu-server.conf`
(`~/.orangu/coordinator/servers/<name>.conf`, carrying `models`/`host`/
`port` and whichever of `backend`/`slots`/`web` were set), resolve which
`orangu-server` executable to run (`resolve_server_binary`), spawn it with
`--config <that path> <role flag> <model>`, piped stdout/stderr captured
into the rolling tail, record `current_pid` immediately (before the
possibly-long health check), then poll `GET /v1/models` on the new origin
every 500ms until it succeeds or `startup_timeout` elapses. A profile's
`host` travels into that generated config verbatim, so it takes exactly the
spellings `orangu-server`'s own `host` does — including the default, `all`.
`config.rs` resolves that wildcard in the two directions it has to:
`resolve_bind_host` (`all`/`*` → `0.0.0.0`) for the coordinator's own
listener, and `resolve_connect_host` (`all`/`*` → `127.0.0.1`) for
`CoordinatorLlmEntry::origin`, since a wildcard says where a process
listens, not an address anything can dial. Only *then* does
`proxy` forward the original request — headers (minus hop-by-hop ones),
method, and body unchanged — and stream the response back.

Unlike a shell-command-line design, `start`
never parses a shell command line at all: every argument it passes is
either a config value validated at load time or a path it generated itself,
so there's no argv-scraping, no leading-`KEY=VALUE`-environment-assignment
convention, and no manual `~` expansion left to reproduce — `orangu-server`
being a sibling process built from the same source, not an
independently-installed external tool, is what makes this simplification
possible.

### Recovering a profile that stopped

`ensure_active` restarts a child it finds dead (`try_wait`), which covers a
profile that stopped between requests. The proxy handles the case that check
cannot see: a request that fails to *reach* the child — the window where it
was alive when checked and gone by the time the request arrived, which is
what `orangu-server` exiting on a lost GPU device (its own
`device_lost::EXIT_CODE`, `75`) does under a request. Nothing has been
written back to the caller at that point, so the request is still whole and
is sent again exactly once.

The retry goes through `ensure_reachable`, deliberately not `ensure_active`.
`Child::try_wait` lags a `SIGKILL`ed (or just-exited) child by however long
tokio's `SIGCHLD` handling takes to run, and a retry that consults it
milliseconds after a connection failure is told the dead process is fine —
so the retry lands in the same closed port and the caller gets the `502` the
retry existed to prevent. That is a measured failure of the first version of
this code, not a hypothetical. `ensure_reachable` asks the question that
cannot lag — a `GET /v1/models` probe under `HEALTH_CHECK_TIMEOUT` — and
restarts only when it goes unanswered, so a transient blip doesn't cost a
working process and its other in-flight requests.

### Crash diagnostics

If a profile's `orangu-server` exits before answering `GET /v1/models`, or
dies later while actively serving requests (mid-generation, discovered
lazily the next time `ensure_active` reuses that entry and finds `try_wait`
returning `Some`), the coordinator logs a warning (unless `--quiet`) with
the exit status and the last 20 captured lines of stdout/stderr, then
restarts it before serving whatever triggered the check. A crash mid-stream
still surfaces to the *client* as a broken connection (an already-started
200 response can't be retroactively turned into an error), but the
coordinator's own console now always has the actual reason logged.

### Client-side integration (`orangu`)

Everything above assumes `orangu` cooperates by sending the right `model`
field per request type, and by not treating the coordinator like an
ordinary server. The relevant pieces, all gated on a per-connection
`is_active_connection_a_coordinator`/`header_status.is_coordinator` check
(`src/bin/orangu/models.rs`, `src/bin/orangu/main.rs`):

- **Role overrides.** `review_prompt_profile` (review/auto-review, via the
  shared `coordinator_role_profile` helper), `explorer_target_profile`
  (`src/explorer.rs`), `detect_embeddings_server`, and the plain-chat path
  in `main.rs` all skip `orangu.conf`'s own `role`/`model` resolution
  entirely once a coordinator is confirmed, forcing `.model` to the
  conventional role name (`review`, `explorer`, `embeddings`, `code`) on
  whichever connection is already active.
- **Startup/`/server` auto-detection is skipped.** `try_startup_model_switch`
  (which otherwise probes `/v1/models` and reassigns the pinned model id to
  whatever a server advertises) never runs behind a confirmed coordinator —
  there's nothing meaningful to detect when the coordinator alone decides.
- **Display.** `orangu::tui::display_model_name(is_coordinator, model_id)`
  returns `"Automatic"` instead of the wire model id; `render_screen`
  computes this once and reuses it for both the header banner and the
  status line, and the two `ReviewChrome` construction sites in `main.rs`
  apply it directly since the review screens carry no `HeaderStatus` of
  their own.
- **Embeddings re-detection.** Because role selection no longer depends on
  a fixed `orangu.conf` section, `/server`/`/reload` re-runs
  `detect_embeddings_server` against whichever connection just became
  active, so switching to (or away from) a coordinator mid-session
  re-evaluates semantic `/search` availability instead of leaving it fixed
  at whatever was true at launch.

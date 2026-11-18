\newpage

# Developer information

## Main components

- `src/bin/orangu.rs` - terminal loop, command handling, history, connection state, and waiting state
- `src/bin/orangu/manual.rs` - built-in manual viewer (`/manual`); embeds the `doc/manual/en` chapters at compile time, so a new chapter file must be added to its `MANUAL_SOURCES` list
- `src/config.rs` - INI parsing and normalization
- `src/logging.rs` - where `orangu-server` and `orangu-coordinator` log (`log_type`/`log_path`): the `log` facade with `fern` behind it — bare lines on the console exactly as `println!` put them, stamped lines in a file
- `src/llm/openai.rs` - OpenAI-compatible client for `orangu-server`
- `src/session.rs` - tool-calling conversation flow
- `src/tools.rs` - workspace-scoped local tool execution
- `src/tui.rs` - header, prompt frame, and status rendering

## Prompt Construction & KV Caching

Orangu is optimized for local LLMs (served by `orangu-server`), which reuse a KV cache by matching an exact **token prefix**. When developing features that touch `ChatSession`, the rule that matters is:

> Editing a message costs a re-prefill of that message **and everything after it**. Appending costs nothing.

Two consequences that are easy to get wrong:

1. **An in-place edit is not free just because it is not the first message.** `compact_transcript` replaces old tool outputs with a stub, and the prefix *before* the edit does survive — but everything after it is processed again. Worse, the stub is *shorter* than what it replaced, so the next turn's divergence point lands even earlier. Compacting eagerly, one message per turn, measured 3.5× more prefill than never compacting at all. Compaction therefore runs only when a pass can reclaim at least half the transcript, so the cost is paid once per doubling rather than once per turn. See the Compression chapter.

2. **Not every edit is worth avoiding.** `set_system_prompt` rewrites `messages[0]`, which does discard the whole conversation's cache. It used to append the new prompt as a `user` message prefixed `[System Update]` to dodge that, and the dodge was worse than the disease: the model got its own instructions in the user's voice while the real system message still held the superseded prompt. The rewrite is affordable because of *when* it runs — only `/server` (which changes endpoint, so that cache is cold anyway) and `/verbosity` (one explicit command, paid once). An unchanged prompt is left untouched.

The general shape: prefer appending; when you must edit, edit rarely and be able to say what the edit buys.

## Development workflow

```sh
cargo fmt
cargo test
```

### Testing a workers channel

`tests/workers_channel.rs` starts several real `orangu-server` processes
on one machine — each on the CPU, with its own API port and
`[workers].port`, finding the others through `ORANGU_WORKERS_PEERS` — and
checks what clients get while a node is killed (`SIGKILL`), frozen
(`SIGSTOP`), or cut off by a proxy between two nodes. The scenarios:

- a model every node holds, with no speed measured (`shares = memory`), is
  served where it is asked: nobody is claimed, and `/v1/workers` says
  `divide.last.verdict` is `here`;
- with `ORANGU_WORKERS_DIVIDE=always`, a model every node holds is divided
  over all three, and a worker's own client is sent on to the lead
  (`x-orangu-routed-to`);
- `POST /v1/workers/sync-bench` between two free nodes answers both formats'
  timings and the prediction, prints them, and leaves both nodes free;
- a model no node holds alone (`ORANGU_WORKERS_BUDGET_MIB`) is split over
  three nodes, a worker's own client is sent on to the node that claimed
  it, and the workers are let go after `release`;
- a worker killed, frozen or cut off mid-answer: the answer runs to its
  last token on the nodes left;
- two nodes that get a request at the same moment form one tree: one leads,
  the other sends its request there, and both answer;
- a leader frozen between requests is let go by its workers through the
  channel, within 40 seconds, and the nodes left answer again;
- two prompts sharing a long prefix on a split model: the second reuses the
  prefix from the lead's pool, its workers' rows kept on the workers;
- a prefix in the lead's pool survives `release`: saved on every node as the
  workers are let go, it is put back when the lead claims them again, and a
  second prompt sharing it reports it as cached;
- a channel with two models, two nodes on the first and one on the second:
  each node's `/v1/models` lists the other model, a request naming a model
  goes to a node serving it, a name no node serves stays where it was sent,
  and the node with the second model is never claimed. It runs only when
  `ORANGU_HARNESS_MODEL2` names the second model, in the same models
  directory;
- a node with no copy of the model, `download = range` and no way to the
  Hub fetches the model from the other two (which divide every model), runs
  its share of a divided answer, and the answer is the one the lead gives alone, word for word, at
  temperature 0. `ORANGU_HARNESS_RANGE_BACKEND` sets that node's `backend`
  (`cpu` unless set); `vulkan` checks that a device which ran the model's
  layers before they were fetched runs them from the fetched bytes;
- a prompt run on one worker that holds the whole model (every node
  divides), its rows then handed to every node of the tree, answers word for word what the same
  prompt answers through the tree, at temperature 0, and the lead's
  `/v1/workers` names that worker under `prefill.last`.

`ORANGU_WORKERS_PREFILL` overrides where the top-level node of a tree runs a
prompt, for such a test: `force` runs every prompt, whatever its length, on
one node of the plan that holds the whole model — the one predicted fastest,
else the first worker that holds it, else the top-level node — and a node's
name (`host:port`, as plans print it) runs every prompt on that node when it
holds the whole model; `0` (or `off`) keeps every prompt on the tree. Unset,
or `auto`, decides by the measured numbers (*Workers*, **Where a long prompt
runs**). A prompt given to a node that fails still goes through the tree.

`ORANGU_WORKERS_DIVIDE=always` makes a node divide every model over the free
nodes of its channel, whatever its size and speed, as a tree of nodes that
each hold the model would never be otherwise: the harness's nodes share by
memory, measure no speed, and serve a model they hold alone. Unset (or any
other value) decides by the measured numbers (*Workers*, **Alone or
divided**).

### Timing the sums of a split layer

`POST /v1/workers/sync-bench` measures, before anything is built, what two
nodes that split every layer's matrices between them would spend summing
their halves: twice a layer — after attention and after the feed-forward —
each node sends the other a hidden-size vector and waits for the other's.
The node it is sent to connects to a free node of its channel as a parent
does (over TLS when the channel uses it, proving the `secret`), echoes a
hidden-size payload off it `2 × layers` times a simulated token, in `f32`
and then in `f16`, after one token untimed, and lets it go. It runs between
requests only:

```sh
curl -s -X POST alpha:8100/v1/workers/sync-bench \
  -H 'Content-Type: application/json' \
  -d '{"peer": "beta:8400", "tokens": 50, "layers": 35, "hidden": 1536}'
```

`peer` is the other node's `[workers]` name, address or node id, as
`/v1/workers` lists it under `found`; `tokens` defaults to 50, `layers` and
`hidden` to this node's model. The API key, if any, is needed as for any
other endpoint. The answer:

- `f32` and `f16`: `payload_bytes`, `per_token_ms` (a simulated token's
  round trips added up: `median`, `p90`), `round_trip_ms` (`median`,
  `p90`) and `mb_per_s`, the payload there and back over the time it took;
- `link`: the round trip and bandwidth the handshake timed, as `/v1/workers`
  shows a worker's; `peer`: its name, address, `tls`, whether it has the
  same model, and its measured `decode_gb_per_s`;
- `predicted`: `alone_ms`, this node's decode step alone (every layer and
  the head at its measured speed, and its fixed cost); `tp2_compute_ms`,
  half of that on each node at the slower node's speed — the peer's, when it
  measured one, else this node's — with the fixed cost once; `sync_ms`, the
  per-token median in the format hidden states travel in (`f32` with
  `activations = f32`, `f16` otherwise, named in `sync_from`);
  `tp2_total_ms`, the two added; and `pays`, whether that is less than
  alone. `null` on a node that measured no speed (`shares = memory`).

It answers `409` while this node leads a tree or works for one — a running
tree is never touched — while another benchmark runs here, or when the peer
is not free; `404` for a peer the channel does not have, or a node with no
`[workers]` section; `400` for sizes out of bounds (`tokens` at most 1000,
`layers` 512, `hidden` 65536, and at most 200 000 round trips a format);
`502` when the peer cannot be reached or drops the connection. The peer is
claimed while it runs: its API keeps serving, and no other node can claim
it. The echo is answered on the peer's connection thread, so the time is
the transport's and the network's, without the thread a worker starts for
each forward.

They need a model, so they are ignored by default:

```sh
ORANGU_HARNESS_MODELS=/path/to/models \
ORANGU_HARNESS_MODEL=bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M \
  cargo test --profile release-with-debug --test workers_channel -- --ignored --test-threads 1
```

`ORANGU_HARNESS_MODELS` defaults to `models` in
`~/.orangu/orangu-server.conf`. A failed scenario prints the end of every
node's log.

## Documentation workflow

The manual and the cheat sheet are built by **orangu itself**: the same
printpdf engine that writes the PDFs `/export` produces draws both documents,
so they carry the reports' bands, brand colour and embedded Red Hat Text
without a second toolchain. There is no LaTeX in the project, and no template
to download. Pandoc is needed only for the HTML manual.

1. Download dependencies

```sh
    dnf install pandoc
```

2. Build

```sh
./doc/build.sh
```

which will produce a HTML and PDF manual, and the PDF cheat sheet. Pass
`manual` or `cheatsheet` to build just one of them.

The sources are `doc/manual/en` (one file per chapter) and
`doc/cheatsheet/en` (one file per page); both are drawn by
`src/bin/orangu/docs.rs`.

## Orangu files

- The default config lookup path is `~/.orangu/orangu.conf`
- Command history is stored in `~/.orangu/orangu.history`

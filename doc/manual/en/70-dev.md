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

- a busy node routes a request to an idle one (`x-orangu-routed-to`);
- a model no node holds alone (`ORANGU_WORKERS_BUDGET_MIB`) is split over
  three nodes, a worker's own client is sent on to the node that claimed
  it, and the workers are let go after `release`;
- a worker killed, frozen or cut off mid-answer: the answer runs to its
  last token on the nodes left;
- two nodes that get a request at the same moment form one tree: one leads,
  the other sends its request there, and both answer;
- a leader frozen between requests is let go by its workers through the
  channel, within 40 seconds, and the nodes left answer again;
- of two free nodes the model needs one: the other stands by, and takes the
  worker's place when it is killed mid-answer;
- two prompts sharing a long prefix on a split model: the second reuses the
  prefix from the lead's pool, its workers' rows kept on the workers;
- a prefix in the lead's pool survives `release`: saved on every node as the
  workers are let go, it is put back when the lead claims them again, and a
  second prompt sharing it reports it as cached.

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

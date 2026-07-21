\newpage

# Inference server

`orangu-server` loads a GGUF model and serves an OpenAI-compatible HTTP
API — both the OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`) and its own
native ones (`/health`, `/props`, `/slots`, `/metrics`, `/completion`,
`/tokenize`, `/detokenize`, `/embedding`, `/apply-template`).

`orangu-server` *is* the inference engine: GGUF loading, tokenization, the
transformer forward pass, sampling, and request scheduling are implemented
directly in Rust, with no dependency on llama.cpp/ggml's own compiled code.
`orangu-coordinator` (see the Coordinator chapter) sits in front of it,
starting and stopping an `orangu-server` process on demand for machines
that only have the resources to keep one model resident at a time — this
chapter covers `orangu-server` itself.

It's also the machine's GGUF inventory tool — the `system`/`suggest`/
`list`/`show`/`download`/`delete` subcommands (below) answer the questions
that matter when *getting*, *choosing*, and *cleaning up* a model, before
or after serving. Those six read (or write) GGUF files directly off disk
and query the local machine, no model loaded and no HTTP listener bound;
`download` talks to the Hugging Face Hub to fetch a model, and `list` talks
to it too — before printing its table, to check whether a newer commit
exists for each Hugging Face-backed model already on disk (see **`list` and
`show`** below). If the Hub is unreachable, `list` still prints the table;
it just skips the check silently rather than failing the command.

## Quick start

```sh
orangu-server unsloth/gemma-4-E2B-it-GGUF
```

The model argument is resolved the same way `show`/`download` resolve one:
an existing local `.gguf` path, an `NR`/`MODEL` label already under the
configured `models` directory (see `orangu-server list`), or a
`<user>/<model>[:quant]` Hugging Face repo — fetched into `models` first if
it isn't already cached there. No separate download step is needed.

Leave it off entirely and `orangu-server` lists every `.gguf` model under
the configured `models` directory and prompts for one by `NR`, then —
unless `--all`/`--code`/`--review`/`--explorer`/`--embedding` was passed —
prompts for a role too (see below), TAB-completing over the five valid
names (dropdown-style: an empty `TAB` press lists all five) and defaulting
to `all` on an empty entry:

```sh
orangu-server
```

```
NR  MODEL                                    QUANT   SIZE
 1  Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M    Q5_0    468.64 MiB
 2  unsloth/gemma-4-E2B-it-GGUF:Q4_K_M        Q5_K    2.89 GiB

Select a model (NR): 2
role [all]: 
```

On startup, `orangu-server` prints the same CPU/GPU report `system` does,
followed by the model/UI/API summary:

```
CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  ...

GPU
  [0] AMD Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]
      ...

Model  unsloth/gemma-4-E2B-it-GGUF (llama arch, CPU/AVX2, 26 layers, 8192 ctx)
UI     disabled
API    http://127.0.0.1:8100
```

The model line's second field names the backend the forward pass actually
ran on: `CPU`/`CPU/AVX2`, or `Vulkan/<adapter name>`, `CUDA/<device name>`,
`OpenCL/<device name>`, `ROCm/<device name>` when the matching GPU backend
was used (see **GPU backend** below).

Every completed request logs a throughput line, orangu-server-style:

```
orangu-server: [slot 0] prompt 42 tokens in 0.18s (233.33 tok/s), generated 128 tokens in 4.31s (29.70 tok/s)
```

## GGUF inventory

Six subcommands cover getting, choosing, and cleaning up a model, all
sharing the same `orangu-server.conf` and its `models` directory (see
**Configuration** below).

**`download`** fetches a model from Hugging Face into the configured
`models` directory, laid out **exactly** the way llama.cpp's own
`-hf`/`--hf-repo` downloads into —
`models--<user>--<model>/{blobs,refs,snapshots}`, content-addressed blobs
with a relative symlink per file — so `list`/`show` already read what this
writes, and llama.cpp itself recognizes it as already downloaded rather
than fetching it again:

```sh
orangu-server download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
orangu-server download ggml-org/embeddinggemma-300M-GGUF   # no :quant -> prefers Q4_K_M, then Q8_0
```

```
Downloading Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf: 47% [1/1]
```

If the repository also ships a multimodal projector (`mmproj-*.gguf`,
needed for vision/audio input), it's fetched alongside the model too — the
same best-matching one orangu-server's own `-hf` would auto-fetch on first
launch anyway, so `LLAMA_CACHE=<models>` already has it ready offline
instead of needing a live fetch the first time a vision-capable model is
launched. A multi-part model's every shard (and a bundled `mmproj`)
downloads concurrently rather than one at a time; an interrupted download
resumes from where it left off next time. Set `HF_TOKEN` in the environment
for a private or gated repository.

**`system`** detects the machine's CPU and GPU(s) — the same report printed
at the top of every attached `orangu-server` startup (see **Quick start**
above):

```sh
orangu-server system
```

```
CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  Vendor           : AuthenticAMD
  Architecture     : x86_64
  Physical cores   : 8
  Logical cores    : 16
  Frequency        : 4.29 GHz
  Memory total     : 62.19 GiB
  Memory available : 36.19 GiB

GPU
  [0] AMD Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]
      Memory type  : Dedicated
      VRAM total   : 3.98 GiB
      VRAM used    : 3.71 GiB
      Driver       : amdgpu
```

GPU detection has no single cross-platform API, so it layers several
best-effort sources: `nvidia-smi` for NVIDIA (Linux and Windows), Linux's
`/sys/class/drm` for everything else on Linux (AMD, Intel, and any other
PCI display device), and native OS tools (`system_profiler`/PowerShell's
`Win32_VideoController`) on macOS and Windows. `Memory type` tells apart a
genuine dedicated card from an integrated GPU/APU sharing the CPU's system
RAM — a `Shared` GPU's `VRAM total` is always reported as the machine's
total system RAM regardless of what its own platform query said, since
that's the real ceiling on how much it can actually draw on.

**`suggest`** estimates a GGUF model *size* (parameter count, not a
specific model yet) likely to run comfortably on this machine, printed as a
table — one row per context length, one column per quantization — sized
against two budgets: dedicated GPU VRAM alone (its table is skipped
entirely on a machine with no dedicated GPU at all, rather than printing a
useless 0 B budget of nothing but `-`), and every GPU's memory combined:

```sh
orangu-server suggest
```

```
Suggested model size (Dedicated)
  Estimated budget : 3.98 GiB

  Context  Suggestion (Q2_K)  Suggestion (Q4_K_M)  Suggestion (Q8_0)
  -------  -----------------  -------------------  -----------------
  1K       ~9B parameters     ~4B parameters       ~3B parameters
  ...
```

The memory-estimation formula mirrors [Sam McLeod's GGUF VRAM
Estimator](https://smcleod.net/vram-estimator/): model weight bytes scale
as parameters × bits-per-weight ÷ 8, KV cache bytes scale with context
length × layers × hidden size, plus a small fixed runtime overhead.

**`list`** recursively scans the configured `models` directory for `.gguf`
files and prints one row per model (a multi-shard model collapses into a
single row, with `SIZE` summed across shards):

```sh
orangu-server list
```

```
NR  MODEL                                                QUANT  SIZE
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M     Q4_K   17.28 GiB
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M   Q4_K   270.14 GiB
 3  ggml-org/gemma-4-12B-it-GGUF:Q4_K_M                  Q4_K   7.14 GiB
```

`NR` numbers models in the printed order, starting from 1 — a shorthand for
`show` so you don't have to retype a long `MODEL` string. When a file was
downloaded by `-hf`/`--hf-repo`, `MODEL` is exactly the string to hand back
to `-hf`: `<user>/<model>[:quant]`. A multimodal projector ("mmproj")
sidecar file doesn't count as its own model — it's meant to be loaded
*alongside* a base model, not to stand in as one.

**`show`** prints a GGUF file's full metadata — every key/value pair in the
file, not just the well-known keys. Omit the argument entirely to pick one
interactively (`list`'s own table, then an `NR` prompt):

```sh
orangu-server show 3                                     # NR from `list`
orangu-server show unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M   # MODEL from `list`
orangu-server show Qwen3-Coder-30B-A3B-Instruct.gguf      # bare name under `models`
orangu-server show ./relative/or/absolute/path.gguf
orangu-server show 3 --tensors   # also list every tensor's shape/type/offset
orangu-server show 3 --full      # print full arrays instead of a preview
orangu-server show               # no argument: list, then pick an NR interactively
```

Array-valued metadata (e.g. `tokenizer.ggml.tokens`, which routinely holds
well over 100,000 entries) is truncated to a short preview by default —
`--full` disables that. Tensor data itself is never read, only the header,
metadata, and tensor-info table, so `list`/`show` stay fast even against
multi-gigabyte model files.

**`delete`** removes a model from disk, resolving its argument the same
way `show` does (or, omitted, the same interactive `list` + `NR` prompt
bare `orangu-server` uses to pick a model to serve — here picking one to
remove instead), and always against every shard the model is made of, so a
multi-shard model is deleted atomically rather than leaving orphans behind:

```sh
orangu-server delete 3                                     # NR from `list`
orangu-server delete unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M   # MODEL from `list`
orangu-server delete                                        # no argument: interactive
```

```
Delete 'unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M' (4 files, 17.28 GiB) from /home/you/models? [y/N]: y
Deleted 'unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M' (4 files, 17.28 GiB)
```

Asks for confirmation first (`[y/N]`, defaulting to **No**) unless
`-y`/`--yes` is given. When a file lives under a Hugging Face hub cache,
its target blob is reclaimed too — but only when no other snapshot left in
that repo still references it — and any now-empty `snapshots/<rev>/` or
`models--<user>--<model>/` directory left behind is cleaned up, never
anything above the configured `models` directory itself.

## Configuration

`orangu-server.conf`:

```ini
[orangu-server]
models = ~/models
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
host = 127.0.0.1
port = 8100
slots = 1
web = 8101
backend = auto
role = all
```

- `models` — the base directory a model spec resolves into: what `list`/
  `show` scan (recursively) for `.gguf` files, `download` fetches into, and
  the serving path resolves the CLI's positional `model` argument against.
  Required by every subcommand except `system` and `suggest` (pure hardware
  inventory, no models directory involved) and a `show` given a direct
  path.
- `model` — a model spec, the same shape as the CLI's positional argument
  (a local `.gguf` path, an `NR`/`MODEL` label, or a `<user>/<model>
  [:quant]` Hugging Face repo). **Only consulted in `--daemon` mode** — a
  normal, attached-terminal run still takes its model from the CLI argument,
  or prompts interactively if none is given, exactly as before; `model`
  in the config is otherwise ignored. `-i`/`--init` prompts for it with
  TAB-completion over the models already installed under `models`.
- `host`/`port` — the bind address, printed on startup.
- `slots` — how many requests generate concurrently, each with its own KV
  cache (default `1`). Raise it to serve overlapping requests without
  queuing behind each other.
- `web` — port for the built-in web UI (see below), bound alongside `port`
  rather than instead of it. `0` (the default) disables it — no second
  listener is bound.
- `backend` — `auto` (the default), `cpu`, `vulkan`, `cuda`, `opencl`, or
  `rocm`. `auto` tries every GPU backend compiled into this build, in order
  (Vulkan, CUDA, OpenCL, then ROCm if built with the `rocm` feature),
  falling back to the CPU backend silently if none is found; naming a
  backend explicitly fails to start instead of falling back, for when GPU
  inference was asked for specifically. See **GPU backend** below.
- `role` — `all` (the default), `code`, `review`, `explorer`, or
  `embedding`. See **Roles** below. **Only consulted in `--daemon`
  mode** — same as `model`, and for the same reason: an attached-terminal
  run always takes its role from the CLI flag if one was given, or (when
  no model was given on the CLI either) the interactive `role [all]: `
  prompt right after model selection, or `all` otherwise — never from this
  key; `role` in the config is otherwise ignored. In `--daemon` mode, an
  explicit CLI role flag still overrides it.

`-c`/`--config` picks a config file explicitly; without it, `./orangu-server.conf`
then `~/.orangu/orangu-server.conf` are tried, in that order — the same
order every subcommand above resolves it in too, not just serving.
`-i`/`--init` writes `~/.orangu/orangu-server.conf` interactively — it also
prompts for `role` (TAB-completing over the five valid names, defaulting to
`all`), right after `model`, and only writes the `role =` line when a
non-default value was chosen. `-d`/`--daemon` detaches
from the terminal and runs in the background (Unix-only) — it requires
`model` to be set in the config, since there's no attached terminal left to
pass a CLI argument to or prompt on; the config and model are resolved, and
both listeners bound, *before* detaching, so a bad config or a port already
in use is still reported to the invoking terminal rather than silently lost.
`-h`/`--help` and `-V`/`--version` are also available. `-s`/
`--shell-completions` prints a bash/zsh/fish completion script for the
shell detected from `$SHELL` — covering every flag above, the six
subcommand names, and the positional `model` argument plus `show`'s and
`delete`'s own arguments, the latter three completed by shelling out to
`orangu-server list` itself.

## Roles

`--all`/`--code`/`--review`/`--explorer`/`--embedding` (mutually exclusive;
`--all` is the default) hint at which of `orangu-server`'s own features
matter for a given deployment. These mirror `orangu`'s conventional
deployment roles (`all`/`code`/`review`/`explorer`/`embeddings`), but a
single `orangu-server` process serves whatever model it's given rather than
picking one — so unlike a real `orangu-server` process per role, this only
adjusts the handful of things that are actually role-specific in an engine
that doesn't have `orangu-server`'s `--fit`/`--tools`/`--webui-mcp-proxy`/
`-sm`/`--cache-reuse`/`-ctk`/`-ctv` equivalents at all:

- **Default slot count**, when the config doesn't set `slots` explicitly.
  `embedding` defaults to `8` (embedding requests are typically short,
  cheap, and bursty compared to open-ended generation); every other role
  keeps the previous flat default of `1`.
- **Default sampling parameters**, when a request doesn't specify its own
  `temperature`/`top_p`/`top_k`/`min_p`. `explorer` defaults to
  `temperature=0.7, top_p=0.8, top_k=20, min_p=0` (broader, more varied
  output); every other role keeps the engine's existing defaults
  (`temperature=0.8, top_k=40, top_p=0.95, min_p=0.05`).
- **Whether the generation endpoints are served at all.** `embedding`
  disables `/v1/chat/completions`, `/v1/completions`, and `/completion` —
  a clear `501` instead of silently running text generation against a
  model that isn't meant for it. Every other role leaves them on
  (`/v1/embeddings`/`/embedding` stay available regardless of role too —
  they just work if the loaded model supports it).
- **Reasoning suppression, `review` only.** Approximates real llama-
  server's `--reasoning-budget 0 --reasoning off`: `/v1/chat/completions`
  (and `/apply-template`, so it shows the same thing that will actually be
  sent) passes `enable_thinking: false` into the chat template — the
  kwarg convention several reasoning-capable models' own templates check
  (Qwen3's among them) to skip whatever preamble tells the model to think
  first — *and* appends an empty, already-closed `<think>\n\n</think>\n\n`
  block right after the rendered prompt, so generation resumes immediately
  past any thinking phase rather than entering one. `<think>`/`</think>`
  is a near-universal convention (DeepSeek-R1, QwQ, Qwen3, GLM) but not a
  guaranteed one — a model using a different tag, or none at all, won't be
  affected by the prefill half of this.

`code` behaves identically to `all` today — no `orangu-server` feature is
`code`-specific yet beyond what `all` already provides.

The role in effect is, in order: whichever CLI flag was passed; or, if none
was and this is an attached run with no model given on the command line
either, whatever's typed at the interactive `role [all]: ` prompt; or, in
`--daemon` mode only (no attached terminal to prompt on), the config
file's own `role` key; or, failing all three, `all`.

## GPU backend

`orangu-server` can run the forward pass on a GPU as well as on the CPU.
Four GPU backends are available, chosen via `backend` in the config (or
`auto`, the default — see **Configuration** above for the fallback order):

- **Vulkan** (`backend = vulkan`) — the most mature and heavily tuned of
  the four. Weight tensors are uploaded once and cached on the GPU for the
  model's lifetime rather than re-uploaded per request, and a decode
  step's matrix multiplications, attention, RoPE, and normalization are
  fused together into as few GPU submissions as practical, cutting the
  amount of CPU/GPU round-tripping a naive implementation would otherwise
  pay for on every generated token. Reaches AMD GPUs through Mesa's RADV
  driver with no AMD-specific code needed, and reaches NVIDIA/Intel GPUs
  the same way, wherever a working Vulkan driver is installed — no Vulkan
  SDK is needed to *build* `orangu-server`, only a Vulkan driver to *run*
  it on a GPU. Verified end-to-end against real AMD hardware. Still
  meaningfully behind llama.cpp's own tuned Vulkan backend on the same
  model and hardware — a real, ongoing, and openly tracked performance
  gap, not a hidden one.
- **CUDA** (`backend = cuda`, NVIDIA GPUs), **OpenCL** (`backend = opencl`,
  any OpenCL-capable GPU), and **ROCm** (`backend = rocm`, AMD GPUs via
  HIP) — each real and working, cross-checked in automated tests against
  the CPU backend's own output, but scoped more narrowly than Vulkan: a
  straightforward dequantizing matmul kernel without Vulkan's fused,
  GPU-resident optimizations. None of the three has been run against real
  NVIDIA/OpenCL/ROCm hardware during development, so treat them as
  functional but less proven than the Vulkan path until verified on your
  own hardware. ROCm additionally requires building with the `rocm`
  Cargo feature, since it's off by default in a plain build.

Naming a `backend` explicitly fails to start rather than silently falling
back to the CPU, for when GPU inference was asked for specifically.
Startup prints which backend actually ran the model (see **Quick start**
above).

## Web UI

Set `web` in the config (or at the `web` prompt in `--init`) and visit
`http://<host>:<web>/` for a small built-in chat UI:
an input box, a scrolling transcript, a **New Chat** button, and a
**History** button that lists previous chat sessions — sessions with no
messages in them are left out, so History only ever shows conversations
that actually happened. It's a plain server-rendered HTML/CSS/JS page (no
build step, no WASM) served by the same binary — a chat turn calls
straight into the model in process, never making an HTTP hop to the
API's own `port`.

Each assistant reply is rendered from markdown to HTML server-side,
including syntax-highlighted fenced code blocks.

While a reply is streaming in, the **Send** button becomes a **Stop** (✕)
button; clicking it cancels the request. Whatever text had already
streamed in stays on screen, marked as stopped, but since the turn never
reached completion it isn't saved — a stopped reply won't reappear if you
reload or revisit it from **History**.

Chat sessions persist as one directory per session at
`~/.orangu/server/sessions/<uuid>/chat.json`, so **History** survives a
restart.

## Session management

```sh
orangu-server prune            # list sessions, pick one (or 'all') interactively
orangu-server prune all        # delete every non-active session
orangu-server prune <uuid>     # a specific session, by NR or full id
```

`prune` deletes chat sessions from `~/.orangu/server/sessions/`. Needs no
config file and loads no model. Every invocation, regardless of its own
argument, first removes any non-active session with an empty chat history
(a **New Chat** click that was never sent to) **and** any persisted slot
KV-cache file (`~/.orangu/server/<fingerprint>/slots/`, written by the
`?action=save` endpoint) untouched for over 30 days, reporting the space
reclaimed. Those slot files are a pure reprefill-avoidance cache, so an
over-eager sweep only ever costs a one-time prefill; age is used rather than
session-liveness because a slot file is named by the *client's* session id,
which the server can't cross-reference. With no argument, it lists
the rest as a numbered table, newest first, and prompts for an `NR` or
`all`; `all` deletes every remaining session except **active** ones —
sessions a currently-running `orangu-server` is still using, checked live
against the process table each time `prune` runs, not a snapshot from
startup. Naming an active session explicitly refuses rather than deleting
it. `-y`/`--yes` skips the confirmation prompt, the same flag `delete` uses.

## Shutting it down

Three equivalent ways: `Ctrl+C`, `SIGINT` (`kill -INT <pid>`), or
`POST /v1/shutdown` (loopback-only — refused from a non-localhost peer, the
same safety rule `orangu-coordinator`'s own shutdown endpoint uses). Both
the API and (if enabled) the web UI listener stop together.

## Endpoint reference

| Endpoint | |
| :-- | :-- |
| `GET /v1/models` | |
| `POST /v1/chat/completions` | streaming (SSE) and non-streaming; requires the model to have a `tokenizer.chat_template`; disabled under `--embedding` |
| `POST /v1/completions` | legacy OpenAI completion, no chat template needed; disabled under `--embedding` |
| `POST /v1/embeddings` | pooled (mean or last-token, per the model's own `pooling_type`) and L2-normalized |
| `GET /health` | |
| `GET /props` | model + server metadata |
| `GET /slots` | per-slot busy/prompt/generated-token state |
| `GET /metrics` | Prometheus text |
| `POST /completion` | native, streaming; disabled under `--embedding` |
| `POST /tokenize` / `POST /detokenize` | |
| `POST /embedding` | native embeddings |
| `POST /apply-template` | renders the chat template without generating |
| `POST /v1/shutdown` | not part of the standard API — orangu-server's own |

The built-in **Web UI** (above) is served on its own `web` port, separate
from the API's `port`, and exposes a small `/api/...` surface of its own —
used only by that page's own JavaScript, not part of the OpenAI-
compatible API above, and only reachable at all when `web` is configured:

| Endpoint | |
| :-- | :-- |
| `GET /api/asset-version` | the served page's own asset fingerprint — powers the Reload prompt shown when a newer build is running behind an already-open tab |
| `GET /api/system-report` | plain-text hardware report (`system`'s own output) plus model/backend identity — what an error bubble's **Save** button bundles into its downloadable debug report, alongside the visible conversation |
| `POST /api/sessions` | creates a new, empty chat session, returning its id |
| `GET /api/sessions` | lists every non-empty session, newest-updated first |
| `GET /api/sessions/{id}` | one session's full message history, each assistant reply already rendered to HTML |
| `POST /api/sessions/{id}/messages` | sends one chat turn against that session; streaming (SSE) reply, the same shape `/v1/chat/completions`' own stream uses |

## Scope

Text-in/text-out GGUF chat, completion, and embedding models, for four
architecture families: Llama-style (`general.architecture` one of `llama`,
`qwen2`, `qwen3`, `mistral`, and `qwen3vl` — Qwen3-VL's text backbone,
*text-only* input), Gemma4 (`gemma`/`gemma2`/`gemma3`/`gemma4`, plus the
bidirectional-attention, embeddings-only `gemma-embedding`), Qwen3.5/3.6-MoE
(`qwen35moe`), and Qwen3.5 dense (`qwen35` — the same hybrid full-attention/
gated-DeltaNet layer shape as `qwen35moe`, plain SwiGLU FFN instead of MoE
routing) — using `F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/`Q5_0`/`Q4_K`/`Q5_K`/`Q6_K`
tensors. Weight matrices and embedding tables are read lazily from the
memory-mapped file (dequantized one row at a time, on demand) rather than
eagerly resident, so even large models fit in modest RAM.

Not yet built, and out of scope for now: multimodal input, `/infill`,
`/rerank`, LoRA hot-swap, and slot save/restore.

See the Developer information chapter for how the GPU backends, request
scheduler, model forward passes, and GGUF inventory tooling work
internally.

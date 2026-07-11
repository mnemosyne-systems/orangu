\newpage

# Inference server

`orangu-server` loads a GGUF model and serves a llama.cpp-compatible HTTP
API — both the OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`) and llama.cpp's own
native ones (`/health`, `/props`, `/slots`, `/metrics`, `/completion`,
`/tokenize`, `/detokenize`, `/embedding`, `/apply-template`).

Unlike `orangu-coordinator` (which starts and proxies to an external
`llama-server` process), `orangu-server` *is* the inference engine: GGUF
loading, tokenization, the transformer forward pass, sampling, and request
scheduling are implemented directly in Rust, with no dependency on
llama.cpp/ggml's own compiled code.

## Quick start

```sh
orangu-server unsloth/gemma-4-E2B-it-GGUF
```

The model argument is resolved the same way `orangu-gguf show`/`download`
resolve one: an existing local `.gguf` path, an `NR`/`MODEL` label already
under the configured `models` directory, or a `<user>/<model>[:quant]`
Hugging Face repo — fetched into `models` first if it isn't already cached
there. No separate download step is needed.

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

On startup:

```
Model  unsloth/gemma-4-E2B-it-GGUF (llama arch, CPU/AVX2, 26 layers, 8192 ctx)
UI     disabled
API    http://127.0.0.1:8100
```

The second field names the backend the forward pass actually ran on:
`CPU`/`CPU/AVX2`, or `Vulkan/<adapter name>`, `CUDA/<device name>`,
`OpenCL/<device name>`, `ROCm/<device name>` when the matching GPU backend
was used (see **GPU backend** below).

Every completed request logs a throughput line, llama-server-style:

```
orangu-server: [slot 0] prompt 42 tokens in 0.18s (233.33 tok/s), generated 128 tokens in 4.31s (29.70 tok/s)
```

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

- `models` — the base directory a model spec resolves (and downloads) into.
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
then `~/.orangu/orangu-server.conf` are tried, in that order. `-i`/`--init`
writes `~/.orangu/orangu-server.conf` interactively — it also prompts for
`role` (TAB-completing over the five valid names, defaulting to `all`),
right after `model`, and only writes the `role =` line when a non-default
value was chosen. `-d`/`--daemon` detaches
from the terminal and runs in the background (Unix-only) — it requires
`model` to be set in the config, since there's no attached terminal left to
pass a CLI argument to or prompt on; the config and model are resolved, and
both listeners bound, *before* detaching, so a bad config or a port already
in use is still reported to the invoking terminal rather than silently lost.
`-h`/`--help` and `-V`/`--version` are also available. `-s`/
`--shell-completions` prints a bash/zsh/fish completion script for the
shell detected from `$SHELL` — covering every flag above plus the
positional `model` argument, completed by shelling out to `orangu-gguf list`.

## Roles

`--all`/`--code`/`--review`/`--explorer`/`--embedding` (mutually exclusive;
`--all` is the default) hint at which of `orangu-server`'s own features
matter for a given deployment. These mirror `orangu`'s conventional
deployment roles (see the GGUF inventory chapter's role wizard), but a
single `orangu-server` process serves whatever model it's given rather than
picking one — so unlike a real `llama-server` process per role, this only
adjusts the handful of things that are actually role-specific in an engine
that doesn't have `llama-server`'s `--fit`/`--tools`/`--webui-mcp-proxy`/
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
**History** button that lists previous chat sessions. It's a plain
server-rendered HTML/CSS/JS page (no build step, no WASM) served by the
same binary — a chat turn calls straight into the model in process, never
making an HTTP hop to the API's own `port`.

Each assistant reply is rendered from markdown to HTML server-side,
including syntax-highlighted fenced code blocks.

Chat sessions persist as one directory per session at
`~/.orangu/server/sessions/<uuid>/chat.json`, so **History** survives a
restart.

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
| `POST /completion` | llama.cpp-native, streaming; disabled under `--embedding` |
| `POST /tokenize` / `POST /detokenize` | |
| `POST /embedding` | llama.cpp-native embeddings |
| `POST /apply-template` | renders the chat template without generating |
| `POST /v1/shutdown` | not a llama.cpp endpoint — orangu-server's own |

## Scope

Text-in/text-out GGUF chat, completion, and embedding models, for three
architecture families: Llama-style (`general.architecture` one of `llama`,
`qwen2`, `qwen3`, `mistral`, and `qwen3vl` — Qwen3-VL's text backbone,
*text-only* input), Gemma4 (`gemma`/`gemma2`/`gemma3`/`gemma4`, plus the
bidirectional-attention, embeddings-only `gemma-embedding`), and
Qwen3.5/3.6-MoE (`qwen35moe`) — using `F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/
`Q5_0`/`Q4_K`/`Q5_K`/`Q6_K` tensors. Weight matrices and embedding tables
are read lazily from the memory-mapped file (dequantized one row at a
time, on demand) rather than eagerly resident, so even large models fit in
modest RAM.

Not yet built, and out of scope for now: multimodal input, `/infill`,
`/rerank`, LoRA hot-swap, and slot save/restore.

See the Developer information chapter for how the GPU backends, request
scheduler, and model forward passes work internally.

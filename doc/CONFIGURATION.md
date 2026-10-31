# Configuration

`orangu`, `orangu-server` and `orangu-coordinator` each read their own INI
file. This file is the complete key reference for all three, section by
section.

| Program | File | Lookup order |
| :-- | :-- | :-- |
| `orangu` | `orangu.conf` | `./orangu.conf`, then `~/.orangu/orangu.conf` |
| `orangu-server` | `orangu-server.conf` | `-c`/`--config`, then `./orangu-server.conf`, then `~/.orangu/orangu-server.conf` |
| `orangu-coordinator` | `orangu-coordinator.conf` | `-c`/`--config`, then `./orangu-coordinator.conf`, then `~/.orangu/orangu-coordinator.conf` |

Each program's `-i`/`--init` writes its file under `~/.orangu/`.
`orangu-server` switches accept `yes`/`no`, `true`/`false`, `on`/`off` and
`1`/`0`; the client's are listed per key. [SERVER.md](SERVER.md) and
[COORDINATOR.md](COORDINATOR.md) describe the servers' keys at more length.

## orangu

### `[orangu]`

The client section is named `[orangu]`. It selects the default server and
holds client-wide settings.

```ini
[orangu]
server = orangu-server
model = ggml-org/gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10
review_max_tokens = 512
code_max_tokens = 0
compression = on
theme = classic
```

#### Server selection

| Key | Required | Description |
| :-- | :-- | :-- |
| `server` | Yes, if multiple servers exist | Name of the default server section |
| `model` | No | General default model name. Used unless the selected server defines its own `model`, which takes precedence |
| `timeout` | No | Request timeout in seconds. Defaults to `1800` |

#### Limits and budgets

| Key | Required | Description |
| :-- | :-- | :-- |
| `max_tool_rounds` | No | Maximum tool-calling turns per prompt before the client aborts it. Defaults to `10` |
| `review_max_tokens` | No | Response-token cap for each `/auto_review` request. Defaults to `512`; `0` disables the cap. Raise it (e.g. `2048`) when the review model thinks before answering |
| `code_max_tokens` | No | Response-token cap for normal chat and tool responses. Defaults to `0` (no cap) |
| `review_confidence_threshold` | No | Minimum confidence score (0–100) for `/auto_review` findings; findings below it are silently dropped. Defaults to `80`. Set to `0` to disable filtering |
| `semantic_budget_tokens` | No | Token budget for the code chunks `/search` injects into a turn. Hits are added in rank order until the next one would exceed it, so the cap bounds what semantic search costs in context rather than the number of results. Defaults to `16384`; the top hit is always kept |
| `world_state_max_bytes` | No | Ceiling in bytes on the `world_state_changes` fragment prepended to a turn when the working tree has changed. Defaults to `8192`; `0` disables the cap. The fragment is prefilled by the server, so its size is response latency, not just context |
| `compile_workers` | No | Parallel job count `/build` passes to toolchains that support one (e.g. `make -j`, `meson compile -j`, `cargo --jobs`). Defaults to `0`, meaning unused: no job flag is passed and each toolchain falls back to its own default |

#### Compression

| Key | Required | Description |
| :-- | :-- | :-- |
| `compression` | No | Enable orangu's built-in compression layer: context deduplication, file-read stubbing, and shell-output compression (handles `cargo`, `ls`, `grep`/`rg`, `npm`/`yarn`/`pip`, and diff truncations). Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_downsample_lines` | No | Line count above which an unbounded file read returns signatures instead of the whole file, with a note saying so. Defaults to `300`; `0` reads every file in full. Applies only while `compression` is on, and never to a read that asked for a `mode` or a line range |
| `diff_file_cap` | No | Maximum number of files kept when a `git diff` is compressed. Defaults to `20` |

#### Prompt

| Key | Required | Description |
| :-- | :-- | :-- |
| `system_prompt` | No | Override the base system prompt sent to the model. When empty (the default) orangu uses its built-in coding-assistant prompt. The discovered Agent Skills index is appended to whichever prompt is in effect |

#### Interface

| Key | Required | Description |
| :-- | :-- | :-- |
| `theme` | No | Global default UI theme. Defaults to `classic`. Built-ins are `classic`, `modern_dark`, `modern_light`, `oranguday`, `tokyonight`, and `rosepine-moon`; `random` draws one of the available themes at each launch. User themes are loaded from `~/.orangu/themes/*.theme` |
| `banner` | No | Horizontal placement of the header banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `width` | No | Virtual terminal width for the output canvas. Source lines from `/show_file` are laid out at this width and can be panned horizontally. Defaults to `512` |
| `word_wrap` | No | Wrap long lines in the main TUI, `/show_file`, `/review`, and `/auto_review` windows. Defaults to `off`; set it to `on` to wrap at the visible width. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `drop_down` | No | Enable the autocomplete dropdown for slash commands. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `mouse` | No | Enable mouse capture, so the TUI handles the wheel, drag-to-select (copied to the clipboard on release), and double-click. Defaults to `on`; hold **Shift** while clicking or dragging for the terminal's own selection. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `workspaces` | No | Placement of the workspace tabs. Defaults to `top`. Options: `top`, `bottom`, `left`, `right` |
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `sun_tzu_mandarin`, `sun_tzu_english`, `attila_the_hun`, `all` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure, blink an `orangu ●` progress title and ring the terminal bell when a `/auto_review` finishes. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `prime` | No | When a TUI tab opens a fresh session, send its opening — the system prompt and tool definitions every turn starts with, ~1900 tokens on `gemma-4-E2B` — to the server as a one-token request in the background, so the server has it prefilled by the time the first prompt is typed. Measured on the CIX P1: the first turn of a fresh server goes from a full prefill to the cached one. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `terminal` | No | Launch command used to open `$EDITOR` for terminal editors in a new window for `/open_file` (for example `xterm -e` or `kitty`). When unset, a terminal emulator is auto-detected |

#### Git and code hosting

| Key | Required | Description |
| :-- | :-- | :-- |
| `platform` | No | Code-hosting platform driven for `/pull`, `/pull_request`, `/merge`, and `/comment`. Defaults to `github` (uses the `gh` CLI). Options: `github`, `gitlab` (uses the `glab` CLI) |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |

### Server sections

Each server section is a valid value for `[orangu].server` and carries the host
information for that model. `[orangu-server]` is the name written by `orangu -i`.

```ini
[orangu-server]
role = all
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | `orangu-server` URL (its OpenAI-compatible API) |
| `model` | No | Model identifier sent to the server. Overrides the general `[orangu].model` when set |
| `role` | No | A specific role this server fulfills. Valid roles are: `all` (default), `code`, `review`, `explorer`, and `embeddings`. If a specific subsystem needs a server and one is tagged with its role, it will use that server instead of the default. `embeddings` designates the server that embeds code for semantic `/search`; an `all` server also serves it, and search auto-enables when that endpoint responds at startup. Ignored behind a confirmed [orangu-coordinator](COORDINATOR.md) — it alone decides which model backs each role, so a single server section is enough there. |
| `api_key` | No | API key sent as `Authorization: Bearer <key>` on every request to the server (chat completions and model listing). Required when `orangu-server` is started with `--api-key` |
| `model_verbosity` | No | How chatty this server's model should be. Defaults to `normal`. Options: `terse`, `normal`, `verbose`. It is a per-server key: writing it in `[orangu]` has no effect |

At least one of `[orangu].model` or a server's own `model` must be set, so every
server resolves to a non-empty model.

Each server section must resolve to a **unique** (`endpoint`, `model`) pair —
a server represents one host serving one model, and `/model` cycles the
models that host offers. `http://x` and `http://x/v1` are treated as the same
endpoint. Two sections *may* share an `endpoint` as long as their `model`
differs, e.g. several roles proxied through one
[orangu-coordinator](COORDINATOR.md) address. The `api_key` is attached to every `/v1/*` request, so the
`/v1/models` health probe also works against API-key-protected servers.

Use `/server` to switch between the configured servers at runtime; Tab
completion lists every server section.

The canonical example file is `doc/etc/orangu.conf`.

### MCP servers

`orangu` connects only to already-running Streamable HTTP MCP servers; it never
starts a command or manages a child process. Their tools are namespaced as
`mcp__<server>__<tool>`. `/mcp` shows the connected servers, while `/tools`
shows the model-facing tools. Configure a server in `[mcp.<name>]`, or set
`mcp = on` in an existing section.

```ini
[mcp.weather]
endpoint = http://localhost:9000/mcp
timeout = 30
approval_mode = writes
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | Streamable HTTP URL of the running service, normally ending in `/mcp` |
| `mcp` | Only for the shortcut form | Set to `on` in a section that is not named `mcp.<name>` to read that section as an MCP service too. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `timeout` | No | Seconds allowed for initialization, tool discovery, and tool calls alike. Defaults to `30`, and supplies the default for the two keys below |
| `startup_timeout` | No | Seconds for connection and tool discovery alone. Defaults to `timeout`. Must be greater than zero |
| `tool_timeout` | No | Seconds for a single tool call. Defaults to `timeout`. Must be greater than zero |
| `enabled` | No | Whether the service is used at all. Defaults to `on`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `required` | No | Make a failed connection abort workspace startup instead of disabling the service with a warning. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `enabled_tools` | No | Comma-separated allowlist of tool names. Empty (the default) offers every discovered tool |
| `disabled_tools` | No | Comma-separated denylist of tool names. The denylist wins over `enabled_tools` |
| `approval_mode` | No | How tool calls are confirmed: `auto` (or `approve`) runs them directly, `prompt` asks for each call, `writes` asks unless the tool declares `readOnlyHint`, and `deny` disables the service. Defaults to `auto`. Noninteractive and review runs deny whatever needs asking |

The name after `mcp.` is the service name used in the `mcp__<server>__<tool>`
prefix, and it accepts ASCII letters, digits, `_` and `-`. Configuring the same
name twice — once as `[mcp.<name>]` and once through the `mcp = on` shortcut —
is a startup error.


## orangu-server

```ini
[orangu-server]
models = ~/.cache/huggingface/hub
host = all
port = 8100

[web]
port = 8200

[prometheus]
port = 8300

[workers]
port = 8400
```

### `[orangu-server]`

Only `models` is required.

#### Serving

| Key | Required | Description |
| :-- | :-- | :-- |
| `models` | Yes | Base directory model specs resolve against, and that downloads land in. A leading `~` is expanded |
| `model` | No | Model to serve when none is given on the command line: a local path, an `NR`/`MODEL` label, or a `<user>/<model>[:quant]` Hugging Face repo. Required for `--daemon` |
| `host` | No | Bind address for the API. Defaults to `all` (every interface; `*` is an alias); a literal address such as `127.0.0.1` restricts it |
| `port` | No | HTTP API port. Defaults to `8100` |
| `role` | No | Role the server is tuned for: `all`, `code`, `review`, `explorer`, `embedding`, `image`, `video`, or `audio`. Read only under `--daemon`; otherwise the command-line flag (or `all`) decides. An image model is always served as `image`, a video model as `video`, a speech model as `audio` |
| `slots` | No | Concurrent requests, each with its own KV cache. Defaults to `8` for `embedding` and `1` for every other role; must be at least `1` |
| `queue_limit` | No | Requests allowed to wait for a slot before the server answers `503`. Defaults to `0`, unbounded |
| `context` | No | Context, in tokens, one request must be able to hold on the device. Unset, every layer goes to the card when it fits and the context is what is left; set, layers move to the host until the card has room for this much KV cache. Must be at least `1` |
| `api_key` | No | Bearer token every API request must carry. Unset leaves the server open. `ORANGU_API_KEY` takes precedence |
| `tls_cert` | No | PEM certificate for serving HTTPS. Needs `tls_key`; set both or neither |
| `tls_key` | No | PEM private key for serving HTTPS. Needs `tls_cert` |
| `reasoning_effort` | No | Passed to the chat template as `reasoning_effort`. Unset by default, so the template uses its own default. Valid levels depend on the template (`low`/`medium`/`high`, or Qwen3.x's `low`/`medium`/`xhigh`) |
| `prefix_warmup` | No | Remember the prompt prefixes requests reuse, per model, and prefill them again in the background after a restart. Defaults to `on`. `ORANGU_PREFIX_WARMUP` overrides it |
| `draft_model` | No | A smaller model whose proposed tokens the served model verifies (speculative decoding). Unset turns speculation off |
| `draft_tokens` | No | Tokens the draft proposes per verification. Defaults to `4`; must be at least `1` |
| `web` | No | Pre-section spelling of `[web].port`, honored only when the file has no `[web]` section. Defaults to `0`, no console |

#### Hardware and memory

| Key | Required | Description |
| :-- | :-- | :-- |
| `backend` | No | Compute backend: `auto` (the default), `cpu`, `vulkan`, `metal`, `dx12`, `cuda`, `opencl`, `rocm`, or `npu` |
| `device` | No | Which device: an index, part of its name, or `auto` (the default). An unknown device is reported at startup with the list of devices that exist |
| `device_split` | No | Spread one model across several devices: `off` (the default), `auto`, `all`, or a ratio list such as `3,1` |
| `threads` | No | CPU worker threads. Unset uses one per logical core; must be at least `1` |
| `kv_cache` | No | Storage of the KV cache's device mirror: `f16` (the default), `q8_0`, or `f32` |
| `read_size` | No | Size, in KiB, an explicit read of a model file is widened to. Defaults to `8192` (8 MiB); must be a multiple of `4`, and `4` disables widening |
| `prefill_backend` | No | Where multi-token prompt passes run: `auto` (the default: the backend that is faster on a prompt-shaped GEMM timed at load), `device` (where decode runs), or `cpu` (prompts on the CPU, decode on the device). `ORANGU_HYBRID_PREFILL_CPU=1` is `cpu` |
| `prompt_weights` | No | How the weights a prompt multiplies on the CPU are held: `auto` (the default: copies in the format the cores' matrix instructions use, kept when they measure at least 10% faster at load and fit in half the free memory), `copy` (the copies, unmeasured), or `file` (the file's own weights, no copy) |
| `mlp_unroll` | No | Force the GPU's block-unroll decode kernels on or off. Unset (the default) decides at startup by checking each quantized type against the CPU |
| `npu_precompile` | No | Whether this model may use the NPU at all. Defaults to `on`; `off` neither compiles nor loads blocks, including ones already cached |
| `npu_cache_gb` | No | GiB of compiled NPU blocks one model may use. Defaults to a quarter of system memory, between `1` and `16`. `0` stops compiling while still loading cached blocks |

#### Image models

These apply to a `qwen_image` or `qwen_image_2_1` model. The `image_size`,
`image_steps`, `image_cfg_scale`, `image_negative_prompt`, `image_strength`
and `image_format` keys are the defaults for a request that does not set
its own.

| Key | Required | Description |
| :-- | :-- | :-- |
| `text_encoder` | No | The text encoder GGUF (`qwen2vl` for `qwen_image`, `qwen3vl` for `qwen_image_2_1`). Defaults to the largest one found |
| `vae` | No | The model's VAE (`.safetensors`). Defaults to the first one found |
| `vision` | No | The vision projector (`mmproj-*.gguf`) a `qwen_image_2_1` model reads attached pictures with. Defaults to the one beside the text encoder; `none` draws over attached pictures instead |
| `image_lora` | No | Adapter applied to the picture transformer: `auto` (the default: the Lightning adapter under `models`), `none` (the base model), or the path of a `.safetensors` adapter |
| `image_lora_merge` | No | Fold the adapter into the weights at startup instead of applying it in `f32` on every pass. Defaults to `yes` |
| `image_weights` | No | How a `qwen_image_2_1` transformer's linears are held: `auto` (the default: per-row `int8` when the transformer runs on the CPU and total memory is at least three times the 7 GB copy), `int8`, or `file` (the file's K-quants, no copy) |
| `vae_precision` | No | The VAE's convolutions: `int8` (the default, as `Q6_K` on the `int8` kernel) or `f32` (as stored, exact) |
| `image_cache` | No | Whether a step may reuse the last transformer pass instead of running one: `easy` (the default, threshold `0.08`), `easy:<threshold>`, or `off` |
| `image_reference_size` | No | Area an edit reads its reference picture at: `source` (the default: the output's area, but never more than the attached picture's own), `output` (the output's area), or a `WIDTHxHEIGHT` cap |
| `image_size` | No | Picture size, `WIDTHxHEIGHT` (or one number for a square), both multiples of 16. Defaults to `1024x1024` |
| `image_steps` | No | Denoising steps. Defaults to `50`, or the adapter's own count under one |
| `image_cfg_scale` | No | Classifier-free guidance scale; `1` turns guidance off. Defaults to `4`, or `1` under an adapter |
| `image_negative_prompt` | No | What a picture is pushed away from. Defaults to a single space |
| `image_strength` | No | How much of the schedule an attached picture goes through, `0` to `1`. Defaults to `0.6` |
| `image_format` | No | Format a picture is returned in: `png` (the default), `jpeg`, `gif`, `webp`, or `svg` |

#### Video models

These apply to an `ltxv` model (LTX-2.5). `text_encoder` and `vae` name its
text encoder and video VAE, and `vae_precision` applies to that VAE too.
The `video_*` keys are the defaults for a request that does not set its
own.

| Key | Required | Description |
| :-- | :-- | :-- |
| `text_encoder` | No | LTX's Gemma 4 text encoder with its projection heads (a `gemma4` GGUF). Defaults to the largest one found |
| `vae` | No | LTX's convolutional video VAE (`.safetensors`). Defaults to the first one found |
| `audio_vae` | No | LTX's audio VAE and vocoder (`.safetensors`). Defaults to the first one found; `none` makes every clip silent |
| `upscaler` | No | LTX's ×2 latent upscaler (`.safetensors`) for the two-stage recipe. Defaults to the first one found; `none` draws every clip in one stage at its full size |
| `video_size` | No | Clip size, `WIDTHxHEIGHT` (or one number for a square), both multiples of 32, and of 64 in two stages. Defaults to `768x448` |
| `video_seconds` | No | Clip length in seconds, snapped to 8k + 1 frames. Defaults to `5` (121 frames at 24 fps). Not together with `video_frames` |
| `video_frames` | No | Clip length in frames, snapped to 8k + 1. Defaults to `121` |
| `video_fps` | No | Frames per second, a whole number from `1` to `60`. Defaults to `24` |
| `video_audio` | No | Put the sound in the file, when the audio VAE is there. Defaults to `yes` |
| `video_two_stage` | No | The two-stage recipe (half size, upscale, refine), when the upscaler is there. Defaults to `yes` |
| `video_quality` | No | The AV1 encoder's quantizer, `0` (best) to `255` (smallest file). Defaults to `80` |

#### Speech models

These apply to a `qwen3tts` model (Qwen3-TTS). `codec` and `voices` name
its speech codec and voice packs. The `audio_*` keys are the defaults for
a request that does not set its own.

| Key | Required | Description |
| :-- | :-- | :-- |
| `codec` | No | The speech codec GGUF (`qwen3tts_tokenizer`), absolute or relative to `models`. Defaults to the largest one found |
| `voices` | No | A voice pack file (`qwen3tts.voicepack`), or a directory searched for them. Defaults to every pack under `models` made for the model's width |
| `audio_voice` | No | Voice a request gets when it names none. Defaults to the first one loaded |
| `audio_language` | No | `auto` (the default) or a language the model names: `chinese`, `english`, `french`, `german`, `italian`, `japanese`, `korean`, `portuguese`, `russian`, `spanish` (or `zh`, `en`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru`, `es`) |
| `audio_format` | No | Format the speech is returned in: `wav` (the default), `flac`, or `pcm` |
| `audio_temperature` | No | Sampling temperature for the first codebook, `0` (greedy) to `2`. Defaults to `0.9` |
| `audio_top_k` | No | Top-k for the first codebook, `1` or more. Defaults to `50` |
| `audio_repetition_penalty` | No | Repetition penalty for the first codebook, `1` to `2`. Defaults to `1.05` |
| `audio_max_seconds` | No | Longest clip, in seconds, above `0` and at most `600`. Defaults to `120` |

#### Logging

| Key | Required | Description |
| :-- | :-- | :-- |
| `log_type` | No | `console` (the default) or `file`, which appends timestamped output, without the progress line, to `log_path`. A `--daemon` needs `file` to log anything |
| `log_path` | No | File `log_type = file` writes to. Defaults to `orangu-server.log` in the start directory; `~` is expanded and a missing directory is created |

### `[web]`

The built-in web console. Having the section at all is what enables it;
without it no second listener is bound.

```ini
[web]
host = 127.0.0.1
port = 8200
reexec = yes
delete = yes
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `port` | No | Port the console listens on, alongside `[orangu-server].port`. Defaults to `8200`; `0` disables the console |
| `host` | No | Address the console binds. Defaults to `[orangu-server].host`; set it to keep the console somewhere the API isn't, e.g. `127.0.0.1` with the API on `all` |
| `reexec` | No | Whether the console's model manager may load a different model, which restarts the server on it. Defaults to `yes`. Treated as `no` on platforms other than Unix |
| `delete` | No | Whether the console's model manager may delete models from disk. Defaults to `yes`. Chat sessions can always be deleted |

### `[prometheus]`

A dedicated, unauthenticated Prometheus `/metrics` listener. Having the
section at all is what enables it.

| Key | Required | Description |
| :-- | :-- | :-- |
| `port` | No | Port the metrics listener binds. Defaults to `8300` |
| `host` | No | Address it binds. Defaults to `[orangu-server].host`; an explicit value stands even under `--host` |

### `[workers]`

Spreads one model's layers over several `orangu-server` nodes. A node with
this section listens for a parent; a node whose `workers` list is not empty
hands layers to those workers.

| Key | Required | Description |
| :-- | :-- | :-- |
| `host` | No | Address this node binds for its parent. Defaults to `[orangu-server].host` |
| `port` | No | Port this node binds for its parent. Defaults to `8400` |
| `workers` | No | Comma-separated `host:port` list of workers, in layer order (`[v6-address]:port` for IPv6). Defaults to empty |
| `standby` | No | Spare workers in the same form, given layers only when a worker is lost. Defaults to empty |
| `secret` | No | Shared secret parent and worker prove to each other when they connect. Unset authenticates nobody; use the same value on every node |
| `activations` | No | Encoding of hidden states between nodes: `f16` (the default), `f32` (exact), or `q8_0` |
| `timeout` | No | Seconds one forward may take on a worker's subtree before the worker counts as lost. Defaults to `60` |
| `connect_timeout` | No | Seconds reaching a worker may take. Defaults to `10` |
| `local_layers` | No | Layers this node runs itself: `auto` (the default) or a count; `0` hands every layer to the workers |
| `download` | No | `full` (the default) or `range`, which fetches only the layers a parent assigns. `range` is only for a node with no workers of its own |
| `shares` | No | What layers are divided by: `decode` (the default) or `prompt` (each node's measured speed, as far as its memory holds), or `memory` (memory alone) |
| `decode` | No | Where a top-level node decodes once a prompt is through the tree: `auto` (the default: alone when that measures faster and it holds the whole model), `tree`, or `top` |
| `head` | No | Which node applies the output head while the tree decodes: `auto` (the default), `top`, or `last` |
| `offload` | No | `auto` (the default: use the workers only when the tree measures at least 10% faster) or `always` |
| `tls_cert` | No | Certificate the worker listener serves TLS with. Needs `tls_key` |
| `tls_key` | No | Private key for `tls_cert`. Needs `tls_cert` |
| `tls_ca` | No | Certificates trusted when dialing workers, which is then over TLS. Defaults to `tls_cert` |

### MCP sections

Every other section is an MCP server shown in the web console's
**Settings › MCP** pane, named after the section. The server neither
connects to nor calls them.

| Key | Required | Description |
| :-- | :-- | :-- |
| `endpoint` | Yes | URL of the MCP service |
| `enabled` | No | Whether the console reports it as enabled. Defaults to `yes` |
| `approval_mode` | No | Approval policy recorded for it: `auto`, `prompt`, `writes` (the default), or `deny` |

The canonical example file is `doc/etc/orangu-server.conf`.

## orangu-coordinator

```ini
[orangu-coordinator]
models = ~/.cache/huggingface/hub
port = 9000

[code]
model = unsloth/gemma-4-E4B-it-GGUF
port = 8100
```

### `[orangu-coordinator]`

| Key | Required | Description |
| :-- | :-- | :-- |
| `models` | Yes | Models directory passed to every profile's `orangu-server` as its `models`. A leading `~` is expanded |
| `host` | No | Address the proxy listens on: `all` (the default; `*` is an alias) or a literal address |
| `port` | No | Port the proxy listens on. Defaults to `9000` |
| `startup_timeout` | No | Seconds to wait for a newly started `orangu-server` to answer `GET /v1/models`. Defaults to `180` |
| `max_body_bytes` | No | Request and response body size cap, in bytes. Defaults to `67108864` (64 MiB) |
| `idle_timeout` | No | Seconds without requests before the active model is unloaded. Unset (the default) never unloads |
| `shutdown_token` | No | Secret that enables `GET /v1/coordinator/shutdown`; the caller passes `?token=<value>` from a loopback address. Unset disables the endpoint |
| `log_type` | No | `console` (the default) or `file`, which appends timestamped lines to `log_path` |
| `log_path` | No | File `log_type = file` writes to. Defaults to `orangu-coordinator.log` in the start directory; `~` is expanded |

### Profile sections

Every other section is a profile: one `orangu-server` the coordinator
starts and proxies to. At least one profile must have the role `all`.

| Key | Required | Description |
| :-- | :-- | :-- |
| `model` | Yes | Model spec, in the same forms as `orangu-server`'s `model`. A request's `model` field is matched against it |
| `role` | No | `all` (the default), `code`, `review`, `explorer`, or `embeddings`. A section named after a role (`[code]`) is that role; a `role` key that contradicts it is an error |
| `host` | No | Address the profile's `orangu-server` listens on. Defaults to `all`; the coordinator reaches a wildcard-bound profile over loopback |
| `port` | No | Port the profile's `orangu-server` listens on. Defaults to `8100` |
| `backend` | No | Passed to the profile's `orangu-server` as `backend`. Defaults to that server's own default, `auto` |
| `slots` | No | Passed to the profile's `orangu-server` as `slots`. Defaults to that server's own default for the role |
| `web` | No | Passed to the profile's `orangu-server` as `[web].port`, giving that profile a web console while it is active. Off by default |

The canonical example file is `doc/etc/orangu-coordinator.conf`.

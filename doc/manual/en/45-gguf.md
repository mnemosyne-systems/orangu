\newpage

# GGUF inventory

`orangu-gguf` is a small standalone companion tool for local LLM inference.
It answers the questions neither `orangu` nor `orangu-coordinator` need to
at runtime, but that matter when *getting* and *choosing* a model to run:
fetching one from Hugging Face in the first place, what hardware is
available to run a model on, roughly how large a model that hardware can
run comfortably, what models are actually on disk, and — given a role and a
model — what `llama-server` command line actually fits this machine. It
starts no `llama-server` process of its own — it reads GGUF files directly
off disk, queries the local machine, and (only for `download`) talks to the
Hugging Face Hub.

## `download`: fetching a model from Hugging Face

```sh
orangu-gguf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
orangu-gguf download ggml-org/embeddinggemma-300M-GGUF   # no :quant -> prefers Q4_K_M, then Q8_0
```

```
Downloading Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf: 47% [1/1]
```

Downloads into the configured `models` directory, laid out **exactly** the
way llama.cpp's own `-hf`/`--hf-repo` downloads into —
`models--<user>--<model>/{blobs,refs,snapshots}`, content-addressed blobs
with a relative symlink per file — so `list`/`show`/the role wizard already
read what this writes, and llama.cpp itself recognizes it as already
downloaded rather than fetching it again. This mirrors llama.cpp's own
downloader directly rather than reinventing the format, including which
files count as "the model" (a bundled `mmproj`/`imatrix`/`mtp-` sidecar
never does) and the same `Q4_K_M` then `Q8_0` default preference when no
`:quant` is given.

If the repository also ships a multimodal projector (`mmproj-*.gguf`,
needed for vision/audio input), it's fetched alongside the model too —
picking the same best-matching one llama-server's own `-hf` would auto-fetch
on first launch anyway, so `LLAMA_CACHE=<models>` already has it ready
offline instead of needing a live fetch the first time a vision-capable
model is launched:

```
Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf: 100% [1/2]
Downloading mmproj-BF16.gguf: 100% [2/2]
```

A multi-part model's every shard downloads together, in order. Progress
prints as a percentage per file; an interrupted download resumes from where
it left off next time, and a file already fully present is skipped rather
than re-fetched. Set `HF_TOKEN` in the environment for a private or gated
repository.

## Role wizard: a tuned llama-server command

Running `orangu-gguf` with no subcommand launches an interactive wizard:
pick one of the five conventional roles `orangu.conf`/`orangu-coordinator.conf`
already use (`all`, `code`, `review`, `explorer`, `embeddings`, by number or
name), then pick a model — by `NR` or `MODEL`, same as `show` — and it
prints a `llama-server` command line tuned for that combination, using the
detected hardware and the model's own GGUF metadata.

```sh
orangu-gguf
```

```
Roles
  1  all
  2  code
  3  review
  4  explorer
  5  embeddings

Select a role [1-5 or name]: review

NR  MODEL                                     QUANT  SIZE
 1  bartowski/gemma-4-12B-it-GGUF:Q4_K_M      Q4_K   7.14 GiB
 ...

Select a model [NR or MODEL]: 1

Recommended command for role 'review':

  LLAMA_CACHE=/home/you/models llama-server -hf bartowski/gemma-4-12B-it-GGUF:Q4_K_M --port 8100 --ctx-size 131072 -np 1 -fa on -sm layer -t 8 --webui-mcp-proxy --fit on --tools all -b 2048 -ub 2048 --cache-reuse 256 --slot-save-path ~/.orangu/llama-slots --reasoning-budget 0 --reasoning off -ctk q8_0 -ctv q8_0
```

The command is prefixed with `LLAMA_CACHE=<models>` — llama.cpp's own
highest-priority override for where `-hf` looks for (and downloads into)
its Hugging Face hub cache. Pointing it at the configured `models`
directory is what makes `-hf` find a model `download` already fetched
there, instead of falling back to llama.cpp's own default
`~/.cache/huggingface/hub`.

The rest of the flags for each role aren't a hardware-derived heuristic —
they're the hand-tuned, verified examples from the OpenAI platform chapter,
this manual's own canonical reference for running llama.cpp well *with
orangu specifically*: KV cache reuse
(`--cache-reuse`), on-disk slot persistence (`--slot-save-path
~/.orangu/llama-slots`, so a tab's conversation survives being parked,
closed, or the whole client restarting without a full re-prefill), and
`--fit on` to fit device memory automatically rather than a separately
estimated `-ngl`. Only two things are substituted per machine/model:
`--ctx-size` (each role's usual value, capped down to the model's own
reported maximum, `<architecture>.context_length`, when that's smaller)
and `-t` (the detected physical core count, for the roles whose example
sets it). The model reference is `-hf <user>/<model>[:quant]` when it came
from a Hugging Face hub cache (the same string `list` shows), or `-m
<path>` for a plain local file.

The `embeddings` role's `--pooling` is read directly from the model's own
`<architecture>.pooling_type` metadata rather than a fixed guess — pooling
is genuinely model-specific, and a hard-coded default can be wrong even for
the model it was written for (embeddinggemma-300M's own metadata reports
`mean`). It falls back to `--pooling mean`, with a note, only when a model
has no usable `pooling_type` metadata at all.

If the model's own chat template looks like it supports llama-server's
`--reasoning-preserve` flag (useful for keeping a model's prior reasoning
in context across turns, rather than dropping it after each response), it's
added to the command — an informed guess based on the template text, not a
certainty. Skipped for `review`, which turns reasoning off entirely
(`--reasoning off`), so there'd be nothing to preserve.

## `system`: CPU and GPU inventory

```sh
orangu-gguf system
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

  [1] AMD Renoir [Radeon Vega Series / Radeon Vega Mobile Series]
      Memory type  : Shared
      VRAM total   : 62.19 GiB
      VRAM used    : 432.22 MiB
      Driver       : amdgpu
```

GPU detection has no single cross-platform API, so it layers several
best-effort sources and reports whatever they find: `nvidia-smi` for
NVIDIA (Linux and Windows), `/sys/class/drm` for AMD/Intel/other devices on
Linux, `system_profiler` on macOS, and PowerShell's `Win32_VideoController`
on Windows. A card no source recognizes simply doesn't show up.

`Memory type` tells apart a genuine dedicated card from an integrated
GPU/APU sharing the CPU's system RAM — the two behave very differently for
offloading model layers, since a dedicated card's VRAM is a hard capacity
limit while shared memory instead competes with everything else running on
the machine. NVIDIA is always `Dedicated` (no consumer NVIDIA GPU is
anything else); on Linux, AMD/Intel/other devices are told apart via a
kernel driver detail confirmed against real dual-GPU hardware (see the
Developer information chapter); macOS and Windows use their own
platform-specific signals, with Windows falling back to `Unknown` for AMD
cards its driver naming can't reliably distinguish.

A `Shared` GPU's `VRAM total` is always the machine's total system RAM,
regardless of what its own platform query reported — the Renoir APU above
genuinely has only a 512 MiB BIOS-reserved carve-out according to `amdgpu`,
but system RAM is the real ceiling on how much it can actually draw on, and
the only figure worth showing as its total.

## `suggest`: a hardware-based model-size suggestion

```sh
orangu-gguf suggest
```

```
CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  ...

GPU
  [0] AMD Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]
      Memory type  : Dedicated
      VRAM total   : 3.98 GiB
      ...

Suggested model size (Dedicated)
  Estimated budget : 3.98 GiB

  Context  Suggestion (Q2_K)  Suggestion (Q4_K_M)  Suggestion (Q8_0)
  -------  -----------------  -------------------  -----------------
  1K       ~9B parameters     ~4B parameters       ~3B parameters
  2K       ~9B parameters     ~4B parameters       ~3B parameters
  4K       ~9B parameters     ~4B parameters       ~3B parameters
  8K       ~8B parameters     ~4B parameters       ~2B parameters
  16K      ~4B parameters     ~4B parameters       ~2B parameters
  32K      ~4B parameters     ~2B parameters       ~1B parameters
  64K      ~2B parameters     ~1B parameters       -
  128K     -                  -                    -
  256K     -                  -                    -

Suggested model size (Shared)
  Estimated budget : 66.17 GiB

  Context  Suggestion (Q2_K)  Suggestion (Q4_K_M)  Suggestion (Q8_0)
  -------  -----------------  -------------------  -----------------
  1K       ~120B parameters   ~110B parameters     ~65B parameters
  2K       ~120B parameters   ~110B parameters     ~65B parameters
  4K       ~120B parameters   ~110B parameters     ~34B parameters
  8K       ~120B parameters   ~110B parameters     ~34B parameters
  16K      ~120B parameters   ~70B parameters      ~34B parameters
  32K      ~120B parameters   ~70B parameters      ~34B parameters
  64K      ~120B parameters   ~70B parameters      ~34B parameters
  128K     ~70B parameters    ~34B parameters      ~32B parameters
  256K     ~34B parameters    ~30B parameters      ~14B parameters
```

Prints the same CPU/GPU report `system` does, then estimates how large a
model (in parameters) is likely to run comfortably — as a table, one row per
context length (1K to 256K tokens) and one column per quantization (`Q2_K`,
`Q4_K_M` — this project's own default — and `Q8_0`), not a specific model
recommendation yet, just a size class to aim `download` at once you know
what to look for. Two such tables are printed: one sized against dedicated
GPU VRAM alone (everything fits in real VRAM, no spillover), and one against
every GPU's memory combined — dedicated and shared alike, a
shared/integrated GPU's already being the system's total RAM — falling back
to the CPU's own total RAM when there's no GPU at all.

The memory-estimation formula mirrors [Sam McLeod's GGUF VRAM
Estimator](https://smcleod.net/vram-estimator/) and the general shape of
[erans/selfhostllm](https://github.com/erans/selfhostllm)'s calculator:
model weight bytes scale as parameters × bits-per-weight, KV cache bytes
scale with context length × layers × hidden size, plus a small fixed
runtime overhead.

## `list`: what's on disk

```sh
orangu-gguf list
```

```
NR  MODEL                                                QUANT  SIZE
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M     Q4_K   17.28 GiB
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M   Q4_K   270.14 GiB
 3  ggml-org/gemma-4-12B-it-GGUF:Q4_K_M                  Q4_K   7.14 GiB
```

Recursively scans the configured `models` directory for `.gguf` files (a
file is used as-is even when it's reached through a symlink — the layout
Hugging Face's own hub cache uses to name a file under `blobs/`). A model
split into multiple shard files (`name-00001-of-00004.gguf`,
`name-00002-of-00004.gguf`, ...) is collapsed into a single `MODEL` row,
with `SIZE` summed across every shard — `list` reports models, not files.
Only unique models are counted and listed: two directories referencing the
exact same underlying bytes (e.g. two Hugging Face snapshot revisions whose
ref moved without the file's content changing) collapse to one entry, and a
bundled multimodal projector ("mmproj") file doesn't count as a model of
its own, since it accompanies a base model rather than standing in for
one — downloading 4 models still shows 4, even if one of them ships a
bundled `mmproj-*.gguf` alongside its main checkpoint. You can still `show`
an mmproj file by its path (a bare filename only resolves when the file
sits directly in the `models` root, not nested inside a cache's
per-revision subfolders).

`NR` numbers models in the printed order, starting from 1 — a shorthand for
`show` (below) so you don't have to retype or paste a long `MODEL` string.
It's recomputed fresh on every run from whatever's currently on disk, so it
only stays stable between one `list`/`show` and the next as long as the
models directory's contents haven't changed.

When a file was downloaded by `-hf`/`--hf-repo` (llama.cpp stores those in
the standard Hugging Face hub cache, `models--<user>--<model>/...`), `MODEL`
is exactly the string to hand back to `-hf`: `<user>/<model>[:quant]` — e.g.
`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` above can be pasted
straight into `llama-server -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M`.
A file outside a Hugging Face hub cache directory (no repo to recommend)
falls back to a shard-stripped filename on its own.

`QUANT` is a separate, coarser best-effort label: the `ggml_type` accounting
for the most tensor *elements* overall, combined across every shard. It
can't distinguish e.g. `Q4_K_S` from `Q4_K_M` — both use the `Q4_K` ggml
type for most tensors — which is exactly why `MODEL`'s `:quant` tag (read
from the filename, not the tensor types) is the one to actually use with
`-hf`. A file that fails to parse (truncated download, not actually a GGUF
file) is still listed, with its error in place of `QUANT`/`SIZE` — one bad
file doesn't abort the scan.

## `show`: a model's full metadata

```sh
orangu-gguf show 3                                     # NR from `list`
orangu-gguf show unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M   # MODEL from `list`
orangu-gguf show Qwen3-Coder-30B-A3B-Instruct.gguf      # bare name under `models`
orangu-gguf show ./relative/or/absolute/path.gguf
orangu-gguf show 3 --tensors   # also list every tensor's shape/type/offset
orangu-gguf show 3 --full      # print full arrays instead of a preview
```

Prints every metadata key/value pair in the file — the full [GGUF
specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)'s
key-value section, not just the well-known keys. The argument is resolved,
in order: as a direct or relative file path; as a bare filename under the
configured `models` directory; as an `NR` from `list`'s first column; as a
`MODEL` name from its second. For a model split into shards, `show` reads
the first shard — GGUF metadata for a multi-part model lives there in full.

Array-valued metadata (e.g. `tokenizer.ggml.tokens`, which routinely holds
well over 100,000 entries) is truncated to a short preview by default —
`--full` disables that. Tensor data itself is never read, only the header,
metadata, and tensor-info table — `list`/`show` stay fast even against
multi-gigabyte model files.

## orangu-gguf.conf

```ini
[orangu-gguf]
models = ~/models
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `models` | Yes | Directory `list`/`show` scan (recursively) for `.gguf` files, and `download` fetches into. A leading `~`/`~/` is expanded to the home directory |

`models` is only needed by `list`, `download`, and a bare-filename/NR/MODEL
`show`; `system` and a `show` given a direct path need no config at all.

Default lookup order for the config file, same as `orangu.conf`:

1. `./orangu-gguf.conf`
2. `~/.orangu/orangu-gguf.conf`

Generate one interactively with `orangu-gguf --init`: it prompts for the
`models` directory, shows the resulting file, and asks for confirmation
before writing `~/.orangu/orangu-gguf.conf`. The prompt defaults to Hugging
Face's own cache location — `~/.cache/huggingface/hub` on Linux/macOS,
`%USERPROFILE%\.cache\huggingface\hub` on Windows — so pressing Enter alone
points `orangu-gguf` at whatever's likely already there, and TAB-completes
the path you're typing against the filesystem, the same as a shell would.

## Shell completions

```sh
orangu-gguf -s
```

Detects the current shell from `$SHELL` (bash, zsh, or fish) and prints its
completion script — `eval "$(orangu-gguf -s)"` in bash, `orangu-gguf -s |
source` in fish, or write zsh's output once to a file on your `fpath`.
`show`'s argument completes with every current `NR` and `MODEL` from
`list`.

See the Developer information chapter for how GGUF parsing, hardware
detection, and shard grouping work internally.

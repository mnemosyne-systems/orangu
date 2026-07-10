# orangu-gguf

`orangu-gguf` is a small standalone companion tool for local LLM inference. It
answers the questions neither `orangu` nor `orangu-coordinator` need to at
runtime, but that matter when *getting* and *choosing* a model to run:

- Fetching a model from Hugging Face in the first place (`download`).
- What hardware is available to run a model on (`system`)?
- Roughly how large a model that hardware can run comfortably (`suggest`)?
- What models are actually on disk, and what's in them (`list`, `show`)?
- Given a role and a model, what `llama-server` command line actually fits
  this machine (the role wizard, below)?

It starts no `llama-server` process of its own — it reads GGUF files directly
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
downloaded rather than fetching it again. This isn't a reimplementation
guessing at the format: it mirrors llama.cpp's own `common/download.cpp`/
`common/hf-cache.cpp` directly, including which files count as "the model"
(a bundled `mmproj`/`imatrix`/`mtp-` sidecar never does) and the same
`Q4_K_M` then `Q8_0` default preference when no `:quant` is given.

If the repository also ships a multimodal projector (`mmproj-*.gguf`,
needed for vision/audio input), it's fetched alongside the model too —
picking the same best-matching one llama-server's own `-hf` would auto-fetch
on first launch anyway (closest quantization bit-depth to the model's own,
preferring one in the same directory):

```
Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf: 100% [1/2]
Downloading mmproj-BF16.gguf: 100% [2/2]
```

A multi-part model's every shard downloads together, in order. Progress
prints as a percentage per file; an interrupted download resumes from where
it left off next time, and a file already fully present (matching the
repository's own reported size) is skipped rather than re-fetched. Set
`HF_TOKEN` in the environment for a private or gated repository.

Not supported (out of scope for a first version): downloading a `--mtp`
companion file alongside the model, `preset.ini`-based repos, and Docker
registry sources.

## Role wizard: a tuned llama-server command

Running `orangu-gguf` with no subcommand launches an interactive wizard:
pick one of the five conventional roles `orangu.conf`/`orangu-coordinator.conf`
already use (`all`, `code`, `review`, `explorer`, `embeddings`), then pick a
model — by `NR` or `MODEL`, same as `show` — and it prints a `llama-server`
command line tuned for that combination.

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
directory is what makes `-hf` find a model `orangu-gguf download` already
fetched there, instead of falling back to llama.cpp's own default
`~/.cache/huggingface/hub`.

The rest of the flags for each role aren't a hardware-derived heuristic —
they're the hand-tuned, verified examples from the manual's [OpenAI
platform chapter](manual/en/73-openai.md), which is the project's own
canonical reference for running llama.cpp well *with orangu specifically*
(KV cache reuse via `--cache-reuse`, on-disk slot persistence via
`--slot-save-path ~/.orangu/llama-slots`, `--fit on` to fit device memory
automatically instead of a separately estimated `-ngl`, and so on). Only two
things are substituted per machine/model: `--ctx-size` (each role's usual
value, capped down to the model's own reported maximum,
`<architecture>.context_length`, when that's smaller) and `-t` (the
detected physical core count, for the roles whose example sets it). `-hf`
is used when the model came from a Hugging Face hub cache (the same label
`list` shows), falling back to `-m <path>` for a plain local file.

The `embeddings` role's `--pooling` is read directly from the model's own
`<architecture>.pooling_type` metadata (mapped through llama.cpp's
`enum llama_pooling_type`: `none`/`mean`/`cls`/`last`/`rank`) rather than a
fixed guess — pooling is genuinely model-specific, and a hard-coded default
can be wrong even for the model it was written for (embeddinggemma-300M's
own metadata reports `mean`, which is what its `-hf` command now
recommends). Falls back to `--pooling mean` with a note, the most broadly
applicable choice for sentence-embedding models, only when a model has no
usable `pooling_type` metadata at all.

If the model's own chat template (`tokenizer.chat_template`) looks like it
supports llama-server's `--reasoning-preserve` flag, it's added to the
command — this is a cheap textual check (does the template reference one of
the three Jinja variables llama.cpp's own capability probe conditions on:
`preserve_thinking`, `clear_thinking`, `truncate_history_thinking`?), not a
guarantee, since the real check involves actually executing the template.
Skipped for `review`, whose own flags already set `--reasoning off` —
nothing would be left to preserve.

If that manual chapter's examples ever change, `orangu-gguf`'s output
should change with them — they're meant to stay identical.

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

CPU statistics (model, vendor, architecture, physical/logical core counts,
frequency, total/available system RAM) come from [`sysinfo`](https://docs.rs/sysinfo).

GPU detection has no single cross-platform API, so it layers several
best-effort sources and reports whatever they find — a card no source
recognizes simply doesn't show up:

- **NVIDIA** (Linux and Windows): `nvidia-smi`'s CSV query mode, installed
  alongside any NVIDIA driver. Always reported as `Dedicated` — no consumer
  NVIDIA GPU is anything but a discrete card.
- **AMD, Intel, and other PCI display devices on Linux**: `/sys/class/drm`,
  the kernel interface every Linux GPU driver exposes. VRAM total/used comes
  from `amdgpu`'s `mem_info_vram_total`/`mem_info_vram_used` sysfs attributes
  when present; the device's marketing name is looked up in the system's
  `pci.ids` database (the `hwdata` package on Fedora/RHEL, `pciutils`
  elsewhere) when installed, falling back to a raw `vendor:device` id
  otherwise. `Memory type` is `Dedicated` when `amdgpu` also exposes
  `mem_info_vram_vendor` (the VRAM chip manufacturer — only present for a
  real dedicated memory pool, not an APU's carve-out of system RAM) and
  `Shared` otherwise — verified directly against a machine with both a
  discrete AMD card and an integrated AMD APU.
- **macOS**: `system_profiler SPDisplaysDataType -json`. `Memory type` comes
  from which of its own `spdisplays_vram` (dedicated) / `spdisplays_vram_shared`
  (Apple Silicon unified memory, or an older integrated Mac) keys is present.
- **Windows**: PowerShell's `Win32_VideoController` WMI class. Its
  `AdapterRAM` field is a well-known 32-bit value that can misreport VRAM on
  cards with more than ~4 GiB; it's still the best zero-dependency source
  available. `Win32_VideoController` has no dedicated/shared field of its
  own, so `Memory type` is guessed from the adapter name: NVIDIA is always
  `Dedicated`, Intel is `Shared` unless the name says `Arc` (its rare
  discrete line), and AMD is reported `Unknown` — its driver names an APU's
  integrated GPU and a discrete Radeon card too similarly to guess from the
  name alone.

A `Shared` GPU's `VRAM total` is always the machine's total system RAM,
regardless of what (if anything) its own platform query reported — the
Renoir APU above genuinely has only a 512 MiB BIOS-reserved carve-out
according to `amdgpu`, but system RAM (62.19 GiB) is the real ceiling on how
much it can actually draw on, and the only figure worth showing as its
total.

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
`Q4_K_M` — the same default `download` and the role wizard already assume —
and `Q8_0`). Not a specific model recommendation yet — just a size class to
aim `download` at.

Two such tables are printed, sized against two different budgets:

- **Dedicated**: the sum of every **dedicated** GPU's VRAM alone (multiple
  dedicated cards add up, matching `-sm layer`'s multi-GPU tensor split) —
  everything fits in real VRAM, no spillover.
- **Shared**: the sum of *every* GPU's own reported total, dedicated and
  shared alike (a shared/integrated GPU's is already the system's total RAM,
  per the note above) — the more permissive figure, representing every
  device `--fit on` could spread layers across at once. Falls back to the
  CPU's own total RAM when there's no GPU detected at all.

The memory-estimation formula mirrors [Sam McLeod's GGUF VRAM
Estimator](https://smcleod.net/vram-estimator/) (read directly from its
published source, not guessed) and the general shape of
[erans/selfhostllm](https://github.com/erans/selfhostllm)'s calculator:
model weight bytes scale as parameters × bits-per-weight ÷ 8, KV cache bytes
scale with context length × layers × hidden size, plus a small fixed runtime
overhead. Since there's no real GGUF file to read yet, hidden size and layer
count are themselves estimated from the parameter count via the standard
transformer parameter-count approximation (params ≈ 12 × layers ×
hidden_size²).

## `list` and `show`: reading GGUF files

```sh
orangu-gguf list
```

```
NR  MODEL                                                QUANT  SIZE
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M     Q4_K   17.28 GiB
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M   Q4_K   270.14 GiB
 3  ggml-org/gemma-4-12B-it-GGUF:Q4_K_M                  Q4_K   7.14 GiB
```

`NR` numbers models in the printed order (alphabetically by `MODEL`), starting
from 1 — a shorthand for `show` (below) so you don't have to retype or paste a
long `MODEL` string. It's recomputed fresh on every run from whatever's
currently on disk, so it only stays stable between one `list`/`show` and the
next as long as the models directory's contents haven't changed.

Recursively scans the configured `models` directory for `.gguf` files (a file
is used as-is even when it's reached through a symlink — the layout Hugging
Face's own hub cache uses to name a file under `blobs/`). A model split into
multiple shard files (`name-00001-of-00004.gguf`, `name-00002-of-00004.gguf`,
...) is collapsed into a single `MODEL` row, with `SIZE` summed across every
shard — `list` reports models, not files. Only unique models are counted and
listed:

- **A duplicated download counts once.** If two directories reference the
  exact same underlying bytes — most often two Hugging Face snapshot
  revisions of one repo whose ref moved without the file's content
  changing, so the cache reuses (symlinks to) the already-downloaded blob
  rather than fetching it again — resolving each candidate to its real,
  symlink-free path collapses those back down to a single entry.
- **Multimodal projector ("mmproj") sidecar files don't count as their own
  model.** A vision/audio "mmproj" file is meant to be loaded *alongside* a
  base model's own checkpoint (llama.cpp's `--mmproj` flag), not to stand
  in as a model of its own — so if you download 4 models and one of them
  ships a bundled `mmproj-*.gguf`, `list` still reports 4, not 5. Identified
  the same way llama.cpp's own `clip.cpp` loader does: `general.architecture`
  is `"clip"`. You can still `show` an mmproj file by its path (a bare
  filename only resolves when the file sits directly in the `models` root,
  not nested inside a cache's per-revision subfolders) — it just isn't
  counted or given its own `NR`/`MODEL` entry.

When a file was downloaded by `-hf`/`--hf-repo` (llama.cpp stores those in
the standard Hugging Face hub cache, `models--<user>--<model>/...`), `MODEL`
is exactly the string to hand back to `-hf`: `<user>/<model>[:quant]` — e.g.
`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` above can be pasted
straight into `llama-server -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M`.
The `:quant` tag is extracted from the filename the same way llama.cpp's own
`-hf` resolver does (`common/download.cpp`'s `get_gguf_split_info`): the
trailing run of letters/digits/underscores after the last `-` or `.` in the
name, once any shard suffix is stripped. A file outside a Hugging Face hub
cache directory (no repo to recommend) falls back to that same shard-stripped
filename on its own.

`QUANT` is a separate, coarser best-effort label: the `ggml_type` accounting
for the most tensor *elements* overall, combined across every shard (not
just the most tensors — a model has far more small `F32` bias/norm tensors
than large weight matrices, but those matrices hold nearly all the
parameters). It can't distinguish e.g. `Q4_K_S` from `Q4_K_M` — both use the
`Q4_K` ggml type for most tensors — which is exactly why `MODEL`'s `:quant`
tag (read from the filename, not the tensor types) is the one to actually use
with `-hf`. A file that fails to parse (truncated download, not actually a
GGUF file) is still listed, with its error in place of `QUANT`/`SIZE` — one
bad file doesn't abort the scan.

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
key-value section, not just the well-known keys. The argument is resolved, in
order: as a direct or relative file path; as a bare filename under the
configured `models` directory; as an `NR` from `list`'s first column; as a
`MODEL` name from its second. For a model split into shards, `show` reads the
first shard — GGUF metadata for a multi-part model lives there in full.

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

`models` is only needed by `list`, `download`, and a bare-filename `show`;
`system` and a `show` given a direct path need no config at all.

Default lookup order for the config file, same as `orangu.conf`:

1. `./orangu-gguf.conf`
2. `~/.orangu/orangu-gguf.conf`

### Interactive setup

```sh
orangu-gguf --init
```

Prompts for the `models` directory, shows the resulting file, and asks for
confirmation before writing `~/.orangu/orangu-gguf.conf` (creating the
directory if needed, and overwriting any existing file). Defaults to Hugging
Face's own cache location — `~/.cache/huggingface/hub` on Linux/macOS,
`%USERPROFILE%\.cache\huggingface\hub` on Windows, the same directory
llama.cpp's own `-hf` falls back to — so pressing Enter without typing
anything points `orangu-gguf` at whatever's likely already there. TAB
completes the path you're typing against the filesystem, the same as a
shell would.

## Shell completions

```sh
orangu-gguf -s
```

Detects the current shell from `$SHELL` (bash, zsh, or fish) and prints its
completion script:

```sh
# bash — add to ~/.bashrc:
eval "$(orangu-gguf -s)"
# zsh — write once to your fpath directory:
orangu-gguf -s > ~/.zsh/completions/_orangu-gguf
# fish — add to ~/.config/fish/config.fish:
orangu-gguf -s | source
```

`show`'s argument completes with every current `NR` and `MODEL` from `list` —
the completion script calls `orangu-gguf list` itself to read them, the same
way `orangu`'s own shell completions read `~/.orangu/sessions` directly
rather than needing any extra plumbing in the binary.

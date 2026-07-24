# orangu-server

`orangu-server` loads a GGUF model and serves a OpenAI-compatible HTTP
API — both the OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`) and its own
native ones (`/health`, `/props`, `/slots`, `/metrics`, `/completion`,
`/tokenize`, `/detokenize`, `/embedding`, `/apply-template`).

`orangu-server` *is* the inference engine: GGUF loading, tokenization, the
transformer forward pass, sampling, and request scheduling are implemented
directly in Rust, with no dependency on llama.cpp/ggml's own compiled code.
`orangu-coordinator` (see [COORDINATOR.md](COORDINATOR.md)) sits in front of
it, starting and stopping an `orangu-server` process on demand for machines
that only have the resources to keep one model resident at a time — this
document covers `orangu-server` itself.

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
the configured `models` directory (the same table `list` prints) and
prompts for one by `NR`, then — unless `--all`/`--code`/`--review`/
`--explorer`/`--embedding` was passed — prompts for a [role](#roles) too,
TAB-completing over the five valid names (dropdown-style: an empty `TAB`
press lists all five) and defaulting to `all` on an empty entry:

```sh
orangu-server
```

```
NR  MODEL                                   QUANT  SIZE        SUPPORTED
 1  Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M  Q5_0   468.64 MiB  Yes (qwen2)
 2  unsloth/gemma-4-E2B-it-GGUF:Q4_K_M      Q5_K   2.89 GiB    Yes (gemma4)

Select a model (NR): 2
role [all]: 
```

On startup, `orangu-server` prints the same CPU/GPU report `system` does
(so a startup log alone is enough to see what hardware the process actually
has to work with), followed by the model/UI/API/workspace summary:

```
CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  ...

GPU
  [0] AMD Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]
      ...

Model      unsloth/gemma-4-E2B-it-GGUF (llama arch, CPU/AVX2, 26 layers, 8192 ctx)
UI         disabled
API        http://127.0.0.1:8100
Workspace  /home/user/src/orangu
```

The model line's second field names the backend the forward pass actually
ran on: `CPU`/`CPU/AVX2`, or `Vulkan/<adapter name>`, `CUDA/<device name>`,
`OpenCL/<device name>`, `ROCm/<device name>` when the matching GPU backend
was used (see **GPU backend** below). The workspace line is the directory
tree this server operates in (see **Workspace** below).

Every completed request logs a throughput line, orangu-server-style:

```
orangu-server: [slot 0] prompt 42 tokens in 0.18s (233.33 tok/s), generated 128 tokens in 4.31s (29.70 tok/s)
```

## GGUF inventory

Six subcommands cover getting, choosing, and cleaning up a model, all
sharing the same `orangu-server.conf` and its `models` directory (see
**Configuration** below):

### `download`: fetching a model from Hugging Face

```sh
orangu-server download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
orangu-server download ggml-org/embeddinggemma-300M-GGUF   # no :quant -> prefers Q4_K_M, then Q8_0
```

```
Downloading Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf: 47% [1/1]
```

Downloads into the configured `models` directory, laid out **exactly** the
way llama.cpp's own `-hf`/`--hf-repo` downloads into —
`models--<user>--<model>/{blobs,refs,snapshots}`, content-addressed blobs
with a relative symlink per file — so `list`/`show` already read what this
writes, and llama.cpp itself recognizes it as already downloaded rather
than fetching it again. This isn't a reimplementation guessing at the
format: it mirrors llama.cpp's own `common/download.cpp`/
`common/hf-cache.cpp` directly, including which files count as "the model"
(a bundled `mmproj`/`imatrix`/`mtp-` sidecar never does) and the same
`Q4_K_M` then `Q8_0` default preference when no `:quant` is given.

If the repository also ships a multimodal projector (`mmproj-*.gguf`,
needed for vision/audio input), it's fetched alongside the model too —
picking the same best-matching one orangu-server's own `-hf` would auto-fetch
on first launch anyway (closest quantization bit-depth to the model's own,
preferring one in the same directory):

```
Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf: 47% [1/2]
Downloading mmproj-BF16.gguf: 100% [2/2]
```

A multi-part model's every shard (and a bundled `mmproj`) downloads
concurrently rather than one at a time, each printing its own progress line
in place until all are done — a smaller sidecar file like `mmproj-BF16.gguf`
above typically finishes well before the main model. An interrupted download
resumes from where it left off next time, and a file already fully present
(matching the repository's own reported size) is skipped rather than
re-fetched. Set `HF_TOKEN` in the environment for a private or gated
repository.

Not supported (out of scope for a first version): downloading a `--mtp`
companion file alongside the model, `preset.ini`-based repos, and Docker
registry sources.

### `system`: CPU and GPU inventory

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

  [1] AMD Renoir [Radeon Vega Series / Radeon Vega Mobile Series]
      Memory type  : Shared
      VRAM total   : 62.19 GiB
      VRAM used    : 432.22 MiB
      Driver       : amdgpu
```

This is the same report printed at the top of every attached (non-daemon)
`orangu-server` startup (see **Quick start** above) — `system` is that
report on its own, with no model involved.

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

### `suggest`: a hardware-based model-size suggestion

```sh
orangu-server suggest
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

Suggested model size (Combined)
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
`Q4_K_M` — the same default `download` already assumes — and `Q8_0`). Not a
specific model recommendation yet — just a size class to aim `download` at.

Two such tables are printed, sized against two different budgets:

- **Dedicated**: the sum of every **dedicated** GPU's VRAM alone (multiple
  dedicated cards add up) — everything fits in real VRAM, no spillover.
  Skipped entirely when the machine has no dedicated GPU at all — a 0 B
  budget would only print a useless table of `-` in every cell.
- **Combined**: the sum of *every* GPU's own reported total, dedicated and
  shared alike (a shared/integrated GPU's is already the system's total RAM,
  per the note above) — the more permissive figure, representing every
  device this server could spread layers across at once. Falls back to the
  CPU's own total RAM when there's no GPU detected at all. Inherently
  optimistic: the shared part of the pool is the same RAM the OS and
  everything else on the machine live in, so treat it as a hardware
  ceiling, not a promise — with dedicated VRAM added on top it can even
  exceed the machine's total RAM.

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

### `list` and `show`: reading GGUF files

```sh
orangu-server list
```

```
NR  MODEL                                               QUANT  SIZE        SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M    Q4_K   17.28 GiB   Yes (qwen3)
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M  Q4_K   270.14 GiB  Yes (qwen3)
 3  ggml-org/gemma-4-12B-it-GGUF:Q4_K_M                 Q4_K   7.14 GiB    Yes (gemma4)
 4  unsloth/GLM-5.2-GGUF:Q4_K_M                         Q4_K   433.83 GiB  No (glm-dsa)
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
straight into `orangu-server -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M`
(or `orangu-server unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` — see
**Quick start** above). The `:quant` tag is extracted from the filename the
same way llama.cpp's own `-hf` resolver does (`common/download.cpp`'s
`get_gguf_split_info`): the trailing run of letters/digits/underscores after
the last `-` or `.` in the name, once any shard suffix is stripped. A file
outside a Hugging Face hub cache directory (no repo to recommend) falls back
to that same shard-stripped filename on its own.

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

`SUPPORTED` says whether this build can actually load the model —
`Yes (<arch>)` or `No (<arch>)`, judged from the file's header alone
(`general.architecture` plus the tensor directory — cheap: no tensor data).
It's stricter than just recognising the architecture *string*: a model whose
architecture is known can still carry tensors the loader rejects when it goes
to build the model, and reporting `Yes` for one would promise a load that
then fails. The case today is a **gemma checkpoint with MoE expert tensors**
(`ffn_gate_inp` present) — `orangu-server`'s gemma path is dense-only and
bails on it — which is reported `No (gemma4-moe)` (a refined `<arch>-moe`
label) rather than a bare `No (gemma4)`, so the column names the real
blocker. A `No` row is printed *greyed*, not hidden: you can still select
it, but loading fails with a clear "not yet supported" error — the column
just surfaces that before you commit. The greying is emitted only to a
terminal; piped or redirected output stays plain text, so the shell
completions that read `list` by column (below) are unaffected by it.

For every row whose `MODEL` names a Hugging Face repo, `list` also checks
that repo's commit — the `snapshots/<commit>/` directory it's cached
under — against the Hub's current `main` commit (the same `GET
/api/models/<repo>/refs` lookup `download` itself uses to resolve `main`),
in parallel across every distinct repo on the row list. A row whose local
commit is behind gets a trailing `(Refresh)` marker, after `SIZE`:

```
NR  MODEL                                             QUANT  SIZE       SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M  Q4_K   17.28 GiB  Yes (qwen3)  (Refresh)
 2  ggml-org/gemma-4-12B-it-GGUF:Q4_K_M               Q4_K   7.14 GiB   Yes (gemma4)
```

Re-running `orangu-server download` on that repo fetches the newer commit.
The check needs the Hub to be reachable; if it isn't (no network, a
timeout, `HF_TOKEN` rejected, ...), `list` still prints the table — the
lookup for that repo is simply skipped, silently, rather than failing the
command or leaving a stale marker. A model outside the Hugging Face hub
cache layout has no repo to check and never gets a marker.

```sh
orangu-server show 3                                     # NR from `list`
orangu-server show unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M   # MODEL from `list`
orangu-server show Qwen3-Coder-30B-A3B-Instruct.gguf      # bare name under `models`
orangu-server show ./relative/or/absolute/path.gguf
orangu-server show 3 --tensors   # also list every tensor's shape/type/offset
orangu-server show 3 --full      # print full arrays instead of a preview
```

Prints every metadata key/value pair in the file — the full [GGUF
specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)'s
key-value section, not just the well-known keys. The argument is resolved, in
order: as a direct or relative file path; as a bare filename under the
configured `models` directory; as an `NR` from `list`'s first column; as a
`MODEL` name from its second. For a model split into shards, `show` reads the
first shard — GGUF metadata for a multi-part model lives there in full.

Omit the argument entirely to pick one interactively — `list`'s own table is
printed, then `show` prompts for an `NR` (same as `delete` with no argument):

```sh
orangu-server show
```

Array-valued metadata (e.g. `tokenizer.ggml.tokens`, which routinely holds
well over 100,000 entries) is truncated to a short preview by default —
`--full` disables that. Tensor data itself is never read, only the header,
metadata, and tensor-info table — `list`/`show` stay fast even against
multi-gigabyte model files.

### `delete`: removing a model from disk

```sh
orangu-server delete 3                                     # NR from `list`
orangu-server delete unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M   # MODEL from `list`
orangu-server delete Qwen3-Coder-30B-A3B-Instruct.gguf      # bare name under `models`
orangu-server delete                                        # no argument: prints `list`'s table and prompts for an NR
```

```
Delete 'unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M' (4 files, 17.28 GiB) from /home/you/models? [y/N]: y
Deleted 'unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M' (4 files, 17.28 GiB)
```

Resolves its argument exactly the way `show` does — direct/relative/
absolute path, bare filename under `models`, `NR`, or `MODEL` — but always
against every shard the model is made of, not just the first: a
multi-shard model (`name-00001-of-00004.gguf`, ...) is deleted atomically,
even when only one shard's own path was named. Omit the argument entirely
and `delete` prints the same table `list` does and prompts for an `NR`,
the same interaction bare `orangu-server` (no subcommand at all) uses to
pick a model to *serve* — here picking one to remove instead.

Asks for confirmation before deleting anything (`[y/N]`, defaulting to
**No** on an empty entry or a closed/non-interactive stdin) — `-y`/`--yes`
skips the prompt, for scripted use.

When a file lives under a Hugging Face hub cache (`models--<user>--<model>/
snapshots/<rev>/...`, the layout `download` itself writes), its target blob
under that repo's own `blobs/` directory is deleted too, reclaiming the
actual disk space — but only when no other snapshot left in that same repo
still points at it (a repo's ref can move without a file's content
changing, in which case the cache reuses rather than re-fetches the blob;
`delete` won't leave a still-needed one dangling). Empty `snapshots/<rev>/`
and `models--<user>--<model>/` directories left behind by the last shard
removed from them are cleaned up too — never anything above the configured
`models` directory itself.

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
  the serving path resolves the CLI's positional `model` argument against. A
  leading `~`/`~/` is expanded to the home directory. Required by every
  subcommand except `system` and `suggest` (pure hardware inventory, no
  models directory involved) and a `show` given a direct path.
- `model` — a model spec, the same shape as the CLI's positional argument
  (a local `.gguf` path, an `NR`/`MODEL` label, or a `<user>/<model>
  [:quant]` Hugging Face repo). **Only consulted in `--daemon` mode** — a
  normal, attached-terminal run still takes its model from the CLI argument,
  or prompts interactively if none is given, exactly as before; `model`
  in the config is otherwise ignored. `-i`/`--init` prompts for it with
  TAB-completion over the models already installed under `models` — every
  `NR` and every `MODEL` label, in the same order `list` prints them.
- `host`/`port` — the bind address, printed on startup.
- `slots` — how many requests generate concurrently, each with its own KV
  cache (default `1`). Raise it to serve overlapping requests without
  queuing behind each other.
- `web` — port for the built-in web UI (see below), bound alongside `port`
  rather than instead of it. `0` (the default) disables it — no second
  listener is bound.
- `backend` — `auto` (the default), `cpu`, `vulkan`, `cuda`, `opencl`, or
  `rocm`. `auto` tries every GPU backend compiled into this build, in order
  (Vulkan, CUDA, OpenCL, then ROCm if built with `--features rocm`),
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
  explicit CLI role flag still overrides it, same as `model` doesn't get
  that override (there's no CLI model argument to override *with* once
  daemonized, but a role flag can still be passed alongside `--daemon`).

Default lookup order for the config file: `-c`/`--config` picks one
explicitly; without it, `./orangu-server.conf` then
`~/.orangu/orangu-server.conf` are tried, in that order — the same order
every subcommand above resolves it in too, not just serving.

`-i`/`--init` writes `~/.orangu/orangu-server.conf` interactively: prompts
for `models` (TAB-completing the typed path against the filesystem, the
same as a shell would; defaults to Hugging Face's own cache location —
`~/.cache/huggingface/hub` on Linux/macOS,
`%USERPROFILE%\.cache\huggingface\hub` on Windows, the same directory
llama.cpp's own `-hf` falls back to — so pressing Enter without typing
anything points `orangu-server` at whatever's likely already there), then
`model` and `role` (TAB-completing the five valid names, defaulting to
`all`), then `host`/`port`/`web`, shows the resulting file, and asks for
confirmation before writing (creating the directory if needed, and
overwriting any existing file). Only writes the `role =` line when a
non-default value was chosen.

`-d`/`--daemon` detaches from the terminal and runs in the background
(Unix-only) — it requires `model` to be set in the config, since there's no
attached terminal left to pass a CLI argument to or prompt on; the config
and model are resolved, and both listeners bound, *before* detaching, so a
bad config or a port already in use is still reported to the invoking
terminal rather than silently lost. `-h`/`--help` and `-V`/`--version` are
also available.

`-s`/`--shell-completions` prints a bash/zsh/fish completion script for the
shell detected from `$SHELL`:

```sh
# bash — add to ~/.bashrc:
eval "$(orangu-server -s)"
# zsh — write once to your fpath directory:
orangu-server -s > ~/.zsh/completions/_orangu-server
# fish — add to ~/.config/fish/config.fish:
orangu-server -s | source
```

Covers every flag above, the six subcommand names, and the positional
`model` argument plus `show`'s and `delete`'s own arguments — the latter
three completed by shelling back out to `orangu-server list` itself and
reading its first two columns (`NR`/`MODEL`), the same way `orangu`'s own
shell completions read
`~/.orangu/sessions` directly rather than needing any extra plumbing in the
binary. `-w`/`--workspace` completes directories (only), and `-c`/`--config`
any file, in all three shells.

## Workspace

`-w`/`--workspace` sets the root directory `orangu-server` operates in —
the same concept, spelled the same way, as `orangu`'s own `-w`/`--workspace`
(see the Workspaces chapter of the manual):

```sh
orangu-server -w ~/src/orangu unsloth/gemma-4-E2B-it-GGUF
orangu-server --workspace ~/src/orangu unsloth/gemma-4-E2B-it-GGUF
```

It is a run-time parameter only — there is no `orangu-server.conf` key for
it. Without the argument the current working directory is used. Either way
the path is made absolute against the directory the server was started in
and normalized (`.` and `..` segments folded away, symlinks left alone),
then checked to be an existing directory — a typo fails at startup, while
there's still a terminal to report it on, rather than at first use. With
`--daemon` this all happens *before* detaching, so a relative path still
means what it meant in the launching shell.

The resolved path is printed on the startup banner, reported as
`workspace` by `GET /props`, and included in the web UI's saved debug
report. It is the root every workspace-scoped feature operates in: the
file-lifecycle API (the five `*_file` and three `*_directory` endpoints —
see **Endpoint reference** below) refuses any path that resolves outside it,
and the features built on top of it later will do the same.

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
  they just work if the loaded model supports `forward_hidden_states`).
- **Reasoning suppression, `review` only.** Approximates real llama-
  server's `--reasoning-budget 0 --reasoning off` without its reasoning-
  parsing machinery: `/v1/chat/completions` (and `/apply-template`, so it
  shows the same thing that will actually be sent) passes `enable_thinking:
  false` into the chat template — the kwarg convention several reasoning-
  capable models' own templates check (Qwen3's among them) to skip
  whatever preamble tells the model to think first — *and* appends an
  empty, already-closed `<think>\n\n</think>\n\n` block right after the
  rendered prompt, so generation resumes immediately past any thinking
  phase rather than entering one. `<think>`/`</think>` is a near-universal
  convention (DeepSeek-R1, QwQ, Qwen3, GLM) but not a guaranteed one — a
  model using a different tag, or none at all, won't be affected by the
  prefill half of this (the `enable_thinking` kwarg still applies, for
  whatever templates check it).

`code` behaves identically to `all` today — no `orangu-server` feature is
`code`-specific yet beyond what `all` already provides.

The role in effect is, in order: whichever CLI flag was passed; or, if none
was and this is an attached run with no model given on the command line
either, whatever's typed at the interactive `role [all]: ` prompt (TAB-
completes, defaults to `all` — see **Quick start** above); or, in
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
  own hardware. ROCm additionally requires building with `cargo build
  --features rocm` — see [BUILDING.md](BUILDING.md) — since it's off by
  default in a plain build.

Naming a `backend` explicitly fails to start rather than silently falling
back to the CPU, for when GPU inference was asked for specifically.
Startup prints which backend actually ran the model (see **Quick start**
above).

## Web UI

Set `web` in the config (or at the `web` prompt in `--init`) and visit
`http://<host>:<web>/` for a small built-in chat UI:
an input box, a scrolling transcript, a **New Chat** button, and a
**History** button that lists previous chat sessions — sessions with no
messages in them (e.g. one just started with **New Chat** but never sent
to) are left out, so History only ever shows conversations that actually
happened. It's a plain server-rendered HTML/CSS/JS page (no build step,
no WASM) served by the same binary — a chat turn calls straight into the
model's `Engine` in process, never making an HTTP hop to the API's own
`port`.

Each assistant reply is rendered from markdown to HTML server-side —
including syntax-highlighted fenced code blocks — reusing the same
`markdown`/`syntect` crates `orangu`'s own terminal UI uses for its
rendering, just pointed at HTML instead of ANSI.

While a reply is streaming in, the **Send** button becomes a **Stop** (×)
button; clicking it cancels the request. This closes the connection the
reply was streaming over, which the engine notices the next time it goes
to send a token and stops generating right there. Whatever text had
already streamed in stays on screen, marked as stopped — but since the
turn never reached completion, it isn't written to the session file, so a
stopped reply (and the message that triggered it) won't reappear if you
reload or revisit it from **History**.

Chat sessions persist as one directory per session at
`~/.orangu/server/sessions/<uuid>/chat.json`, so **History** survives a
restart. A directory (not a flat `<uuid>.json` file) so a session can grow
more per-session files later without another layout migration — see
**Session management** below for cleaning old ones up.

## Session management

Every session directory also gets a `session.json`, alongside its
`chat.json`, recording which `orangu-server` process most recently touched
it — written whenever a session is created or a turn is appended to it, read
by `orangu-server prune` (below) to tell a session a server is still using
apart from an old, abandoned one. This is internal bookkeeping, not
something to edit or rely on the shape of directly.

### `prune`: deleting old chat sessions

```sh
orangu-server prune            # list sessions, pick one (or 'all') interactively
orangu-server prune all        # delete every non-active session
orangu-server prune 3          # NR from prune's own listing
orangu-server prune <uuid>     # a specific session id
orangu-server prune all -y     # skip the confirmation prompt
```

Needs no config file and loads no model — like `system`/`suggest`, it's a
pure filesystem operation against a fixed path
(`~/.orangu/server/sessions/`).

Every invocation, regardless of its own argument, first removes any
**non-active** session whose `chat.json` is empty (a **New Chat** click that
was never sent to, or a leftover from an interrupted write) — routine use of
`prune` in any form also compacts away this junk as a side effect:

```
Removed 2 empty sessions.
```

What's left is then handled by the argument:

- **No argument**: prints every remaining session as a numbered table,
  newest-updated first, and prompts for an `NR` or `all`:

  ```
  NR  ID                                    TITLE                MESSAGES  UPDATED
   1  153ed918-1cde-4ac3-aa3e-fc8eb9d2c462  What is Rust?                4  2m ago  (active)
   2  f082af10-39c9-465c-b2b1-92e4682bb689  Explain sliding windows      6  1d ago

  Prune (NR or 'all', empty to cancel):
  ```

- **`all`**: deletes every remaining session **except** active ones,
  printing which were skipped and asking for confirmation (`-y`/`--yes`
  skips it, for scripted use — the same flag `delete` uses).
- **An `NR`** (from `prune`'s own listing above) **or a full session id**:
  prunes that one session.

A session is **active** when its `session.json` names a process that's
still running — checked by pid *and* start time, so a pid the OS has since
reused for an unrelated process doesn't count. This is re-checked live
against the current process table every time `prune` runs, in a separate
CLI invocation from whatever server process actually owns the session — not
a snapshot taken once at some earlier point — so a session started long
after some other still-running server's own startup is still correctly
protected, and one whose server has since exited becomes prunable the
moment that happens, not after some delay. Naming an active session
explicitly refuses rather than deleting it:

```
Session '153ed918-1cde-4ac3-aa3e-fc8eb9d2c462' is active (in use by a running orangu-server) — not pruned.
```

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
| `POST /v1/create_file` | file lifecycle: write a new file, with optional permissions |
| `POST /v1/modify_file` | file lifecycle: replace named line ranges, returning a diff |
| `POST /v1/move_file` | file lifecycle: rename a file, optionally re-setting permissions |
| `POST /v1/delete_file` | file lifecycle: delete a file |
| `POST /v1/show_file` | file lifecycle: return a file's entire content |
| `POST /v1/create_directory` | file lifecycle: create one directory, with optional permissions |
| `POST /v1/move_directory` | file lifecycle: move an entire directory tree |
| `POST /v1/delete_directory` | file lifecycle: delete an empty directory |
| `POST /v1/shutdown` | not part of the standard API — orangu-server's own |

Those eight are orangu-server's own JSON API for the whole life cycle of a
file and the directories it lives in, and are confined to the workspace (see
**Workspace** above): a path outside it is refused before anything is
touched. `delete_directory` only removes an empty directory, and nothing in
the API deletes a tree.

When the workspace is a **Git repository**, they are Git operations: a file
is created, modified, moved and deleted with `git add`, `git mv` and
`git rm`, so the change is staged, and each reply reports what reached the
index (including the forge, when `gh` or `glab` is installed). **Nothing is
ever committed** — that stays the user's own decision. A request can pass
`"git": false` for a plain filesystem change. They are documented field by field in the Inference
server internals chapter of the manual (`doc/manual/en/78-server.md`), under
**File-lifecycle API**.

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
`mmap`ped file (dequantized one row at a time, on demand) rather than
eagerly resident, so even large models fit in modest RAM. Runs on CPU or,
via `backend = vulkan`/`cuda`/`opencl`/`rocm`/`auto`
(see **GPU backend** above), a Vulkan/CUDA/OpenCL/ROCm-capable GPU —
Vulkan is the only one of the four with real fused/GPU-resident
optimizations beyond a basic matmul kernel, verified against real AMD
hardware; the other three are real but smaller-scoped and unverified on
real hardware.

Not yet built, and out of scope for now: multimodal input, `/infill`,
`/rerank`, LoRA hot-swap, and slot save/restore.

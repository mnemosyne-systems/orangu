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
`list`/`show`/`download`/`delete`/`refresh` subcommands (below) answer the
questions that matter when *getting*, *choosing*, and *cleaning up* a model,
before or after serving. Those seven read (or write) GGUF files directly off
disk and query the local machine, no model loaded and no HTTP listener bound;
`download` and `refresh` talk to the Hugging Face Hub to fetch a model, and
`list` talks to it too — before printing its table, to check whether a newer commit
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
NR  MODEL                            QUANT   SIZE        SUPPORTED
 1  Qwen/Qwen2.5-0.5B-Instruct-GGUF  Q4_K_M  468.64 MiB  Yes (qwen2)
 2  unsloth/gemma-4-E2B-it-GGUF      Q4_K_M  2.89 GiB    Yes (gemma4)

Select a model (NR): 2
role [all]: 
```

When the directory holds exactly one model there is nothing to choose
between, so the `NR` prompt is skipped — the table is still printed (it
names the model and whether this build supports it), and the run goes
straight on to the role prompt:

```
NR  MODEL                            QUANT   SIZE        SUPPORTED
 1  Qwen/Qwen2.5-0.5B-Instruct-GGUF  Q4_K_M  468.64 MiB  Yes (qwen2)

model: Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
role [all]: 
```

On startup, `orangu-server` prints the same OS/CPU/GPU report `system`
does (so a startup log alone is enough to see what machine, and what OS,
the process actually has to work with), followed by the model/UI/API/
workspace summary:

```
OS
  Name             : Fedora Linux
  ...

CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  ...

GPU
  [0] AMD Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]
      ...

Model      unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (llama arch, CPU/AVX2, 26 layers, 8192 ctx)
UI         disabled
API        http://0.0.0.0:8100
Workspace  /home/user/src/orangu
```

The model line names the model as `MODEL:QUANT` — the quantization the
resolved file is actually stored at, the same value `list`'s `QUANT` column
shows, appended unless the model was named with a `:tag` of its own already.
Its second field names the backend the forward pass actually
ran on: `CPU`/`CPU/AVX2`, or `Vulkan/<adapter name>`, `Metal/<device name>`,
`CUDA/<device name>`,
`OpenCL/<device name>`, `ROCm/<device name>` when the matching GPU backend
was used (see **GPU backend** below). The workspace line is the directory
tree this server operates in (see **Workspace** below).

Every completed request logs a throughput line, orangu-server-style:

```
orangu-server: [slot 0] prompt 42 tokens in 0.18s (233.33 tok/s), generated 128 tokens in 4.31s (29.70 tok/s)
```

## GGUF inventory

Seven subcommands cover getting, choosing, keeping current, and cleaning up
a model, all sharing the same `orangu-server.conf` and its `models`
directory (see **Configuration** below):

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

### `system`: OS, CPU and GPU inventory

```sh
orangu-server system
```

```
OS
  Name             : Fedora Linux
  Version          : 44
  Kernel           : 7.1.3-200.fc44.x86_64
  Distribution     : fedora
  Machine          : Micro-Star International Co., Ltd. Bravo 15 A4DDR
  Hostname         : orangu
  Uptime           : 20d 07h
  Load average     : 4.80, 4.09, 3.25
  Swap total       : 16.00 GiB
  Swap used        : 13.23 GiB
  Huge pages       : madvise
  Page size        : 4.00 KiB
  Open files       : 1048576 (max 1048576)
  Models           : /home/orangu/models
  Models used      : 42.31 GiB
  Models free      : 118.92 GiB
  Built for        : x86_64-linux-gnu

CPU
  Model            : AMD Ryzen 7 4800H with Radeon Graphics
  Vendor           : AuthenticAMD
  Architecture     : x86_64
  Physical cores   : 8
  Logical cores    : 16
  Frequency        : 4.29 GHz
  Memory total     : 62.19 GiB
  Memory available : 36.19 GiB
  SSE4.2           : Yes
  AVX2             : Yes
  AVX512           : No

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

The `OS` section comes first because it frames how the two below it should
be read: a 16 KiB page size or a swapped-out machine says more about what a
model will do there than its core count does.

Every field is best-effort and read through a Rust API — [`sysinfo`](https://docs.rs/sysinfo)
for the portable ones, `libc` for the POSIX ones, plain `procfs`/`sysfs`
file reads on Linux. Nothing here shells out to `uname`, `sysctl` or
PowerShell. A field the running platform can't answer gets no line at all
rather than a line saying `unknown`:

| Field | What it is | Linux | macOS | Windows |
|---|---|---|---|---|
| `Name`, `Version` | Distribution or OS name and release | ✅ | ✅ | ✅ |
| `Full name` | The verbose name, printed only when it adds something the two above don't — the macOS codename (`MacOS 15.1 Sequoia`) or the Windows edition (`Windows 11 Pro`) | — | ✅ | ✅ |
| `Kernel` | Kernel release; the build number on Windows | ✅ | ✅ | ✅ |
| `Distribution` | `/etc/os-release`'s `ID`, the spelling other tooling keys on. Omitted where it would only repeat `Name` | ✅ | — | — |
| `Machine` | The physical machine's vendor and product, from SMBIOS/DMI (`/sys/class/dmi/id`) or `hw.model` | ✅ | ✅ | — |
| `Hostname` | | ✅ | ✅ | ✅ |
| `Uptime` | Time since boot, as its two largest units | ✅ | ✅ | ✅ |
| `Load average` | 1/5/15-minute run-queue averages | ✅ | ✅ | — |
| `Swap total`, `Swap used` | Omitted entirely on a machine with no swap configured | ✅ | ✅ | ✅ |
| `Huge pages` | The transparent-hugepage policy (`always`/`madvise`/`never`), which decides whether a mapped model's weights get 2 MiB pages or 4 KiB ones | ✅ | — | — |
| `Page size` | `sysconf(_SC_PAGESIZE)` — 4 KiB on most machines, 16 KiB on Apple Silicon | ✅ | ✅ | — |
| `Open files` | `RLIMIT_NOFILE`, soft and hard. A server holding a listener, every accepted connection and one mapping per model shard runs into this one first | ✅ | ✅ | — |
| `Models` | The configured `[orangu-server].models` directory. Printed only when a config file is found — `system` itself needs none | ✅ | ✅ | ✅ |
| `Models used` | Bytes on disk under that directory: everything it holds, not only the `.gguf` files `list` shows, with a blob shared by several snapshot revisions counted once | ✅ | ✅ | ✅ |
| `Models free` | Room left on the filesystem holding it — what the *next* `download` has to fit into. Excludes the root-only reserve, so it's space this user can actually use | ✅ | ✅ | — |
| `Built for` | The target this binary was *built* for, which isn't always the machine it runs on: an `x86_64` build on an `aarch64` Mac is running under Rosetta | ✅ | ✅ | ✅ |

CPU statistics (model, vendor, architecture, physical/logical core counts,
frequency, total/available system RAM) come from `sysinfo` too; the SIMD
feature lines are run-time CPUID checks, so a binary built on one machine
reports accurately on whatever machine it actually runs on.

GPU detection has no single cross-platform API, so it layers several
best-effort sources and reports whatever they find — a card no source
recognizes simply doesn't show up, and a machine where none of them finds
anything gets no `GPU` section at all (the CPU inventory is the whole
report) rather than a heading over a "none detected" line:

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

Suggested model size (Total)
  Estimated budget : 62.19 GiB

  Context  Suggestion (Q2_K)  Suggestion (Q4_K_M)  Suggestion (Q8_0)
  -------  -----------------  -------------------  -----------------
  1K       ~120B parameters   ~70B parameters      ~34B parameters
  2K       ~120B parameters   ~70B parameters      ~34B parameters
  4K       ~120B parameters   ~70B parameters      ~34B parameters
  8K       ~120B parameters   ~70B parameters      ~34B parameters
  16K      ~120B parameters   ~70B parameters      ~34B parameters
  32K      ~120B parameters   ~70B parameters      ~34B parameters
  64K      ~120B parameters   ~70B parameters      ~34B parameters
  128K     ~70B parameters    ~34B parameters      ~30B parameters
  256K     ~34B parameters    ~27B parameters      ~14B parameters
```

Prints the same OS/CPU/GPU report `system` does, then estimates how large a
model (in parameters) is likely to run comfortably — as a table, one row per
context length (1K to 256K tokens) and one column per quantization (`Q2_K`,
`Q4_K_M` — the same default `download` already assumes — and `Q8_0`). Not a
specific model recommendation yet — just a size class to aim `download` at.

Two such tables are printed, sized against two different budgets:

- **Dedicated**: the largest **dedicated** GPU's VRAM alone — everything
  fits in real VRAM, no spillover. Skipped entirely when the machine has no
  dedicated GPU at all — a 0 B budget would only print a useless table of
  `-` in every cell. The largest card, not every card added up: a model is
  loaded onto one device and nothing splits its tensors across two, so two
  24 GiB cards run the models one 24 GiB card runs.
- **Total**: the **largest single memory pool** on the machine — the biggest
  eligible GPU's own total, or the CPU's total RAM, whichever is larger
  (a shared/integrated GPU's total is already the system's RAM, per the note
  above). Dedicated VRAM is one of the candidates, so this budget is never
  smaller than the one above it: the `Dedicated` table is the *fast* subset
  of this one, not a different machine.

  The largest pool, deliberately **not** the sum of every pool. A model runs
  on exactly one backend, and there is no partial-offload split of layers
  between a GPU and the CPU, so no single run can ever draw on a discrete
  card's VRAM *and* system RAM at once — adding them would put a budget on
  the table that nothing on the machine can reach (on the laptop above, a
  3.98 GiB card beside 62.19 GiB of RAM would suggest a ~110B model that
  neither pool can hold). Still a hardware ceiling rather than a promise:
  the RAM in it is the same RAM the OS and everything else live in.

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
NR  MODEL                                        QUANT   SIZE        SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF    Q4_K_M  17.28 GiB   Yes (qwen3)
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF  Q4_K_M  270.14 GiB  Yes (qwen3)
 3  ggml-org/gemma-4-12B-it-GGUF                 Q4_K_M  7.14 GiB    Yes (gemma4)
 4  unsloth/GLM-5.2-GGUF                         Q4_K_M  433.83 GiB  No (glm-dsa)
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
is the repo id to hand back to `-hf`: `<user>/<model>` — e.g.
`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` above can be pasted straight into
`orangu-server unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` (see **Quick start**
above). A file outside a Hugging Face hub cache directory (no repo to
recommend) falls back to the shard-stripped filename on its own.

The `:quant` tag is left off, since `QUANT` shows it in its own column right
beside it. Two quantizations of the same repo therefore print the same
`MODEL` and are told apart by their `QUANT` cells:

```
NR  MODEL              QUANT   SIZE        SUPPORTED
 1  acme/Test-3B-GGUF  Q4_K_M  468.64 MiB  Yes (qwen2)
 2  acme/Test-3B-GGUF  Q8_0    4.97 GiB    Yes (gemma4)
```

Both spellings resolve against what's already on disk — `acme/Test-3B-GGUF`
and `acme/Test-3B-GGUF:Q8_0` alike — so a `model =` value or a script written
against an older listing keeps working. For a repo with several
quantizations on disk, the tagged form (or the row's `NR`) is what names one
of them in particular; a bare repo id takes the first row it matches, the one
`NR` 1 above names. `delete` always spells out the quantization it is about to
remove in its confirmation line (`Delete 'acme/Test-3B-GGUF' (Q8_0, 1 file,
4.97 GiB) ...`), so an ambiguous argument can't quietly take the wrong one.
The tagged form is also how you ask for a specific quantization that *isn't*
downloaded yet, since a bare repo id lets the downloader pick (`Q4_K_M`, then
`Q8_0`).

The tag itself is extracted from the filename the same way llama.cpp's own
`-hf` resolver does (`common/download.cpp`'s `get_gguf_split_info`): the
trailing run of letters/digits/underscores after the last `-` or `.` in the
name, once any shard suffix is stripped.

`QUANT` is the quantization scheme the model itself names — the same
filename tag `MODEL`'s `:quant` suffix comes from, so the two always agree.
It is only shown when that tag really reads as a quantization (`Q4_K_M`,
`IQ2_XXS`, `TQ1_0`, `F16`, ...); a name whose trailing token is something
else (`gemma-4-E2B-it`, `TinyLlama-1.1B`) falls back to a coarser
best-effort label instead: the `ggml_type` accounting for the most tensor
*elements* overall, combined across every shard (not just the most tensors —
a model has far more small `F32` bias/norm tensors than large weight
matrices, but those matrices hold nearly all the parameters).

The fallback is genuinely coarser, which is why the name wins whenever there
is one. A mixed scheme is *defined* by storing part of the model at a heavier
type than its name — a real `Q4_K_M` model can have `Q5_K` or `Q6_K` as its
single most common type — so reporting the dominant type would contradict the
`MODEL` label sitting right next to it. It also can't tell `Q4_K_S` from
`Q4_K_M` at all, since both store most tensors as `Q4_K`.

A file that fails to parse (truncated download, not actually a
GGUF file) is still listed, with its error in place of `QUANT`/`SIZE` — one
bad file doesn't abort the scan.

`SUPPORTED` says whether this build can actually load the model —
`Yes (<arch>)` or `No (<arch>)`, judged from the file's header alone
(`general.architecture` plus the tensor directory — cheap: no tensor data).
It's stricter than just recognising the architecture *string*: a model whose
architecture is known can still carry tensors the loader rejects when it goes
to build the model, and reporting `Yes` for one would promise a load that
then fails. No recognised architecture hits that today — gemma MoE
checkpoints (`gemma-4-26B-A4B`, `ffn_gate_inp` present, fused or separate
gate/up experts with optional per-expert `.scale` companions) load via the
gemma routed-expert path, so they report `Yes (gemma4)`. A `No`
row is printed *greyed*, not hidden: you can still select
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
NR  MODEL                                      QUANT   SIZE       SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  Q4_K_M  17.28 GiB  Yes (qwen3) (Refresh)
 2  ggml-org/gemma-4-12B-it-GGUF               Q4_K_M  7.14 GiB   Yes (gemma4)
```

`orangu-server refresh` (below) is what acts on that marker — it deletes the
local copy and downloads the newer commit in its place.

The check needs the Hub to be reachable; if it isn't (no network, a
timeout, `HF_TOKEN` rejected, ...), `list` still prints the table — the
lookup for that repo is simply skipped, silently, rather than failing the
command or leaving a stale marker. A model outside the Hugging Face hub
cache layout has no repo to check and never gets a marker.

```sh
orangu-server show 3                                     # NR from `list`
orangu-server show unsloth/Qwen3-Coder-Next-GGUF          # MODEL from `list`
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
orangu-server delete unsloth/Qwen3-Coder-Next-GGUF          # MODEL from `list`
orangu-server delete Qwen3-Coder-30B-A3B-Instruct.gguf      # bare name under `models`
orangu-server delete                                        # no argument: prints `list`'s table and prompts for an NR
```

```
Delete 'unsloth/Qwen3-Coder-Next-GGUF' (Q4_K_M, 4 files, 17.28 GiB) from /home/you/models? [y/N]: y
Deleted 'unsloth/Qwen3-Coder-Next-GGUF' (Q4_K_M, 4 files, 17.28 GiB)
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

### `refresh`: downloading a model again

```sh
orangu-server refresh 3                                        # NR from `list`
orangu-server refresh unsloth/Qwen3-Coder-Next-GGUF             # MODEL from `list`
orangu-server refresh bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K # one quantization of a repo with several on disk
orangu-server refresh                                           # no argument: prints `list`'s table and prompts for an NR
```

```
Refresh 'unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M' (4 files, 17.28 GiB)? The local copy is deleted first, then downloaded again. [y/N]: y
Deleted 'unsloth/Qwen3-Coder-Next-GGUF' (4 files, 17.28 GiB)
Downloaded Qwen3-Coder-30B-A3B-Instruct-Q4_K_M-00001-of-00004.gguf: 100% [4/4]
Downloaded to /home/you/models/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/<newcommit>/...
```

What `list`'s `(Refresh)` marker asks for: the repo has moved on since the
model was fetched, so the local copy is deleted and the same
`<user>/<model>:<quant>` spec downloaded again — `delete` and `download` in
one step, at the newer commit.

The local copy goes first, and the download follows. The point of a refresh
is that the repo's files have changed, so the new revision is a full second
copy on disk rather than a cheap blob-sharing snapshot: deleting first means
a 17 GiB model needs 17 GiB free to refresh, not 34. The trade is that an
interrupted download leaves the model missing rather than stale — which is
what the confirmation line says, and what re-running `refresh` (or
`download`, which resumes from the `.part` file left behind) recovers from.

The argument resolves the way `delete`'s does — `NR`, `MODEL`, bare name, or
path, always against every shard of the model — with one deliberate
difference. Two quantizations of one repo share a `MODEL` cell (`QUANT` is
what tells them apart), and where `delete` takes the first match and spells
out in its confirmation line which one that was, `refresh` refuses:

```
$ orangu-server refresh bartowski/Llama-3.2-1B-Instruct-GGUF
error: 'bartowski/Llama-3.2-1B-Instruct-GGUF' names 2 models on disk (Q4_K_M, Q6_K); name the quantization too — 'bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M' — or use an NR from 'orangu-server list'
```

It has to: `refresh` deletes what it then downloads, so silently picking a
row would refresh the wrong quantization *and* leave the one you meant
untouched. Name it as `<repo>:<quant>`, or use the row's `NR`.

With no argument at all, `refresh` prints the same table `list` does — and
greys every row that is *already* at its repo's latest commit, the inverse of
what `list` greys. Only the `NR`s worth refreshing (the `(Refresh)` rows)
stand out:

```
NR  MODEL                                      QUANT   SIZE       SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  Q4_K_M  17.28 GiB  Yes (qwen3) (Refresh)
 2  ggml-org/gemma-4-12B-it-GGUF               Q4_K_M  7.14 GiB   Yes (gemma4)     <- greyed

Select a model to refresh (NR):
```

When nothing is behind (or the Hub is unreachable, in which case nothing is
*known* to be behind and no row is greyed), it says so before the prompt. As
with the marker itself, a model outside the Hugging Face hub cache layout has
no repo to refresh from; naming one is an error, raised before anything is
deleted.

Confirmation, and `-y`/`--yes` to skip it, work exactly as in `delete`.

## Configuration

`orangu-server.conf`:

```ini
[orangu-server]
models = ~/models
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
host = all
port = 8100
slots = 1
backend = auto
role = all

[web]
port = 8101
reexec = yes
```

- `models` — the base directory a model spec resolves into: what `list`/
  `show` scan (recursively) for `.gguf` files, `download` fetches into, and
  the serving path resolves the CLI's positional `model` argument against. A
  leading `~`/`~/` is expanded to the home directory. Required by every
  subcommand except `system` and `suggest` (pure hardware inventory, no
  models directory involved) and a `show` given a direct path. `-i`/`--init`
  prompts for it with TAB-completion over real filesystem paths and an
  inline grey ghost suggestion of the directory being typed — the same
  completer drives both, so what the grey text previews and what TAB fills
  in can never disagree.
- `model` — a model spec, the same shape as the CLI's positional argument
  (a local `.gguf` path, an `NR`/`MODEL` label, or a `<user>/<model>
  [:quant]` Hugging Face repo). **Required by `--daemon`**, which has no
  terminal to prompt on. An attached run still takes its model from the CLI
  argument when one is given; when none is, the interactive picker
  **pre-selects this one** — its `NR` is shown as the prompt's default and
  ghosted on the empty line, so a config that already names a model is one
  Enter away rather than a row to find. Type a different `NR` (or a label,
  or a path) to override it. A model the config names but that isn't
  installed has no row to point at; its spec is offered as written instead,
  and Enter fetches it exactly as `orangu-server <spec>` would. `-i`/`--init` prompts for it with
  TAB-completion over the models already installed under `models` — every
  `NR` and every `MODEL:QUANT`, in the same order `list` prints them. The
  quantization is part of the offered name (`unsloth/gemma-4-E2B-it-GGUF:Q4_K_M`)
  rather than a separate column, because a repo with several quantizations
  on disk prints the same bare `MODEL` on every one of their rows: offering
  that would list one name once per quantization, and write a `model =`
  value resolving to whichever came first instead of the one picked. The
  labels also drive an inline grey ghost suggestion: the prompt opens
  already previewing the first model listed, and narrows to whatever the
  typed prefix matches (an `NR` is completed but never ghosted — it's a
  shorthand to type, not a name to preview). A
  `models` directory holding exactly one model is not asked about at all:
  that model is taken (and echoed), since there is nothing to choose
  between.
- `host`/`port` — the bind address, printed on startup. `host` defaults to
  `all` (`*` is accepted as an alias for it), which binds every network
  interface on the machine — the API and the web UI are then reachable from
  anywhere that can route to it, not just from this machine. Give a literal
  address instead to narrow that down: `127.0.0.1` keeps the server on the
  loopback interface only, and any other address of a local interface binds
  just that one.
- `slots` — how many requests generate concurrently, each with its own KV
  cache (default `1`). Raise it to serve overlapping requests without
  queuing behind each other.
- `backend` — `auto` (the default), `cpu`, `vulkan`, `metal`, `cuda`,
  `opencl`, or
  `rocm`. `auto` tries every GPU backend compiled into this build, in order
  (Vulkan, CUDA, OpenCL, then ROCm if built with `--features rocm`),
  falling back to the CPU backend silently if none is found. **On macOS the
  order starts with Metal**, which is the only GPU API Apple ships — Vulkan
  is still tried behind it, for a Mac running MoltenVK. Naming a
  backend explicitly fails to start instead of falling back, for when GPU
  inference was asked for specifically. See **GPU backend** below.
- `role` — `all` (the default), `code`, `review`, `explorer`, or
  `embedding`. See **Roles** below. Resolved in this order: an explicit CLI
  flag (`--all`/`--code`/`--review`/`--explorer`/`--embedding`) wins
  everywhere and skips the prompt entirely; failing that, `--daemon` takes
  this key directly, having no terminal to prompt on; and an attached run
  with no flag uses it to **pre-select the interactive `role` prompt** —
  ghosted on the empty line, TAB-completing over the five names, and
  overridable by typing another. That prompt only appears when no model was
  given on the CLI either; an attached run that names a model and no role
  flag is `all`, as before.

### The `[web]` section

The built-in web console (see **Web UI** below) is configured in its own
section, and **having that section at all is what enables it**. A config
with no `[web]` binds no second listener; `-i`/`--init` asks
`Add web console` and then `host`, `port`, `reexec` and `delete`, or writes
no section at all.

```ini
[web]
host = 127.0.0.1
port = 8101
reexec = yes
delete = yes
```

- `port` — where the console listens, bound alongside `[orangu-server].port`
  rather than instead of it. Defaults to `8101` when the section is present
  but says nothing.
- `host` — the address it binds, prompted for with the same interface
  completion and ghost suggestion `[orangu-server].host` gets, and defaulting
  to whatever that was just answered. **When the key is absent it falls back
  to `[orangu-server].host`**, so an ordinary config names one host and both
  listeners use it; answering differently is how the two get separated — an
  API on `all` for the machines that consume it, with the console kept on
  `127.0.0.1`.
- `reexec` — whether the console's model manager may load a different model
  (default `yes`; `no`/`true`/`false`/`on`/`off`/`1`/`0` are all accepted).
  Loading one restarts this process on it, so a deployment that needs the
  server it started to stay the server it started — behind a supervisor, or
  where one specific model is the point of the process — sets `no`, and every
  row's **Load** button is gone. Removed rather than disabled, for the same
  reason `delete` below removes its own: a control that can never do anything
  on this server explains less than its absence does. Non-Unix platforms have
  no `execve` and behave as though it were `no`. See **Loading a different
  model** below.
- `delete` — whether the console's model manager may delete models (default
  `yes`, same spellings). Set `no` and every row's **Delete** button is
  gone, and the endpoint behind it refuses. Its own key rather than riding
  on `reexec` because the two are genuinely separate wishes: deleting a
  model is the one irreversible thing the console can do, and a deployment
  may well want a model switch allowed while the models directory stays
  read-only. It governs **models only** — History's own delete controls are
  unconditional, since a chat session is the console's own scratch data
  rather than a file on disk something else put there.

`web = <port>` under `[orangu-server]` is what this replaced, and still
works: a configuration written against it goes on serving the console on
that port, with `host` and `reexec` at their defaults. A `[web]` section
takes precedence over it wherever both appear.

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
anything points `orangu-server` at whatever's likely already there, and a
directory that doesn't exist yet is created, parents included), then
`model` and `role` (TAB-completing the five valid names, defaulting to
`all`), then `host` (TAB-completing — and previewing as an inline grey
ghost — `all`, `*`, and every address this machine's network interfaces
actually have, each listed with the interface it belongs to), then
`port`/`web`, shows the resulting file, and asks for
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

Covers every flag above, the subcommand names, and the positional
`model` argument plus `show`'s, `delete`'s and `refresh`'s own arguments —
those four completed by shelling back out to `orangu-server list` itself and
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
Five GPU backends are available, chosen via `backend` in the config (or
`auto`, the default — see **Configuration** above for the fallback order):

- **Vulkan** (`backend = vulkan`) — the most mature and heavily tuned of
  the five. Weight tensors are uploaded once and cached on the GPU for the
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
- **Metal** (`backend = metal`, Apple GPUs; the default on macOS) — the
  Vulkan backend's engine and its kernels, running on Apple hardware. Not
  a separate, smaller implementation: the compute shaders, the cached
  GPU-resident weights, the fused decode and prefill submissions, split-k
  attention and GPU sampling are all written against portable `wgpu` and
  WGSL, and this backend simply brings that same code up on a Metal device
  instead of a Vulkan one. So everything listed for Vulkan above is live
  here too, and both get every future optimization at the same time. macOS
  ships no Vulkan driver, which is why `auto` prefers Metal there and why
  a Mac previously fell all the way back to the CPU backend. Verified on
  each push by CI's macOS runner: the same per-quantization-type
  cross-checks against the CPU backend that gate the Vulkan path, plus a
  whole-model prefill and a batched decode on a real GGUF.
- **CUDA** (`backend = cuda`, NVIDIA GPUs), **OpenCL** (`backend = opencl`,
  any OpenCL-capable GPU), and **ROCm** (`backend = rocm`, AMD GPUs via
  HIP) — each real and working, cross-checked in automated tests against
  the CPU backend's own output, but scoped more narrowly than Vulkan and
  Metal: a
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

Add a `[web]` section to the config (or answer `Add web console` in
`--init`) and visit `http://<host>:<port>/` for a small built-in chat UI:
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

History can also clean them up itself. Each row carries a **cross** on its
right that deletes that one chat, and the dropdown's footer carries a small
**Clear all** that deletes every one of them; both ask for confirmation
first. Deleting removes the session's whole directory, exactly as `prune`
does. If what went was the chat currently on screen, the console starts a
fresh empty one in its place rather than leaving a transcript up that no
longer exists anywhere; the dropdown stays open either way, so several can
be cleared in a row.

Neither control is gated by `[web].delete` — that switch is about **models**,
files on disk that something else put there. A chat session is the console's
own scratch data, and a console that can't clear its own transcripts isn't a
deployment posture worth configuring for.

## Model management

The topbar's **Models** button opens a panel showing the models directory as
`orangu-server list` prints it — the same numbered table, column for column:

| | |
| :-- | :-- |
| `NR` | the row number, the same one `list` gives the same model |
| `MODEL` | what to pass to `show`/`delete`/`refresh` on the command line |
| `QUANT` | the quantization the file is stored at, `-` when it says nothing |
| `SIZE` | summed across every shard |
| `SUPPORTED` | e.g. `Yes (llama)`, `No (glm-dsa)`, `No (llama, TQ1_0)` |

Those strings come from the same code that prints them in the terminal, so
the two tables cannot end up saying different things about the same file. A
model this build cannot load is greyed, exactly as `list` greys it, and a
file whose header wouldn't parse shows its `error:` in place of the last
three columns — again as `list` does. The row this server actually loaded is
tinted and marked **loaded**; a row whose Hugging Face repo has a newer
revision is marked **Refresh**, `list`'s own marker (`orangu-server refresh`
is what acts on it).

Above the table sits the loaded model with its architecture, backend, layer
count, context length, role and slot count, and the models directory with how
much of its filesystem is used and free.

Two icon buttons per row — hover either for what it does:

| Icon | Tooltip | |
| :-- | :-- | :-- |
| play triangle | **Load ...** | serve this model instead — see below. Absent entirely when `[web].reexec` is off |
| document | **Show ...** | this file's full GGUF metadata — `orangu-server show` |
| waste basket | **Delete ...** | remove every shard — `orangu-server delete`. Absent entirely when `[web].delete` is off |

The loaded model's row shows a check mark where its **Load** button would
be (and is named **loaded** beside its own name, which is what says so when
there are no Load buttons at all). **Delete** is disabled on it: its weights are memory-mapped by
the running engine, so removing the file would leave this process reading
something that no longer has a name. It asks for confirmation naming the
model and its size, and reclaims the Hugging Face hub-cache blobs too when
nothing else still references them.

**Show** opens a scrolling pane with the file's metadata, and has two toggles
of its own — **Include tensors** (`show --tensors`: every tensor's name,
shape, type and offset) and **Expand truncated arrays** (`show --full`: every
element, including a 100,000-entry vocabulary) — plus a **Save** button that
downloads what is on screen as a text file.

Above the table, a text box takes a `user/model:QUANT` Hugging Face repo and
downloads it. Without `:QUANT` it prefers `Q4_K_M` then `Q8_0`, exactly as
`orangu-server download` does. The download runs in the background — closing
the panel, or the browser tab, does not stop it — and reports its progress
per file, with an overall percentage and ETA, in the panel: the same numbers
`download`'s own terminal progress board draws, as data rather than as
in-place-updating text. One download runs at a time; starting a second while
one is in flight is refused rather than queued, since two fetches into the
same directory would compete for the same disk and the same free-space
check. An interrupted one resumes from its `.part` file the next time it is
asked for.

**Rescan** (the circular arrow in the panel header) re-reads the models
directory. The panel does not re-read it on its own: opening every GGUF
header under a directory holding a few dozen models takes seconds, and
nothing there changes by itself. A delete or a finished download refreshes
the listing automatically; **Rescan** is for a `.gguf` that arrived some
other way. The **Refresh** markers come from one Hugging Face request per
distinct repo, made when the panel opens and when **Rescan** is pressed,
never on the poll — an unreachable Hub marks nothing, since "unknown" is not
"behind".

### Loading a different model

**Load** serves a different model without you going back to the terminal.
It does that by **restarting the server on it** — the process replaces
itself (`execve`) with a new one started on the chosen model, rather than
swapping the model inside the running process. That is deliberate: the new
model is loaded by exactly the same code that loads one at startup, so there
is no second load path that could behave differently from a normal start.

Three things survive the restart, which is what makes it a handover rather
than a stop and start:

- **The listening sockets.** Both are kept open across the restart and
  picked back up by the new process, so neither port is ever unbound.
  Nothing can take the port in between, and a client connecting during the
  load simply waits rather than getting "connection refused". Measured on a
  400-request probe across a live handover: every request answered, except
  the single one already in flight at the moment of the switch.
- **The process id.** `execve` replaces the program but keeps the pid, so
  systemd, a `--daemon` launcher, or a shell job goes on tracking the same
  process. A `--daemon` server stays detached.
- **Everything on disk.** Chat history, downloaded models, saved slot
  KV-caches.

What does not survive is anything held only in memory. Requests in flight
are cut off, which is why **Load** refuses while any slot is still
generating — finish or stop the reply, then load. (A request that has
arrived but has not yet been given a slot can still be caught by the switch;
the window is small and the client simply retries.)

The server keeps everything about itself that was not the model: the same
**role**, workspace, host, ports, backend and slot count it was started
with, whether they came from the command line, the config file, or an
interactive prompt. Only the model changes.

Before switching, the console checks what it can while the current model is
still working — that the file resolves, and that its header names an
architecture and quantization this build can read (the same judgement the
`SUPPORTED` column reports). Some failures can only be found by actually
loading: a GPU backend with no kernel for one of the model's tensor types,
or a model too large for the machine. If that happens the server restarts
once more on the model it was serving before, and the console says so
rather than leaving you with a dead port.

The switch is not written anywhere: restart the server and it comes back on
whatever the command line or `model` in `orangu-server.conf` names. To make
a choice permanent, set `model` in the config.

Set `reexec = no` in the `[web]` section to turn this off — the **Load**
buttons are then gone entirely, and the endpoint behind them refuses. It is
also unavailable on non-Unix platforms, which have no `execve`.

The whole panel is served on the `[web]` port, which is unauthenticated — like
the rest of the web UI, and like the file-lifecycle API on the API port, it
assumes a trusted network. A server reachable from an untrusted one should
not have `web` enabled at all.

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
compatible API above, and only reachable at all when a `[web]` section is
configured:

| Endpoint | |
| :-- | :-- |
| `GET /api/asset-version` | the served page's own asset fingerprint — powers the Reload prompt shown when a newer build is running behind an already-open tab |
| `GET /api/system-report` | plain-text hardware report (`system`'s own output) plus model/backend identity — what an error bubble's **Save** button bundles into its downloadable debug report, alongside the visible conversation |
| `POST /api/sessions` | creates a new, empty chat session, returning its id |
| `GET /api/sessions` | lists every non-empty session, newest-updated first |
| `GET /api/sessions/{id}` | one session's full message history, each assistant reply already rendered to HTML |
| `POST /api/sessions/{id}/messages` | sends one chat turn against that session; streaming (SSE) reply, the same shape `/v1/chat/completions`' own stream uses |
| `DELETE /api/sessions/{id}` | deletes one chat session, directory and all — History's per-row cross |
| `DELETE /api/sessions` | deletes every chat session — History's **Clear all** footer |
| `GET /api/models` | the models directory as the manager panel draws it: `list`'s own table, which row is loaded, disk use, and any download in flight. Serves a cached scan; `?rescan=true` re-reads the directory |
| `GET /api/models/updates` | which rows are behind their Hugging Face repo — `list`'s `(Refresh)` marker, one Hub request per distinct repo |
| `GET /api/models/metadata?model=…` | a model's full GGUF metadata as plain text — `show`'s own output. `&tensors=true` and `&full=true` are `show --tensors`/`--full` |
| `POST /api/models/select` | restarts the server on a different model, keeping both listening sockets and the pid; answers `202` before it acts, since `execve` leaves nothing to answer from |
| `POST /api/models/download` | starts a Hugging Face download in the background, returning at once |
| `DELETE /api/models` | deletes a model, refusing the one currently loaded |
| `DELETE /api/models/job` | clears a finished download's result |

The three that name a model take `{"model": "..."}` — an `NR`, a `MODEL`
label, a bare filename, or a path, exactly as the matching subcommand's own
argument does. The panel sends the `NR`, since that is the only spelling
that names one row exactly (a repo with several quantizations on disk prints
the same bare `MODEL` on each of their rows), and for a load or a delete it
also sends the `path` that row showed. Given both, the server checks they
still agree before acting: an `NR` is a *position*, and a download finishing
while a confirmation dialog is open re-sorts the listing underneath it.

## Scope

Text-in/text-out GGUF chat, completion, and embedding models, for six
architecture families: Llama-style (`general.architecture` one of `llama`,
`qwen2`, `qwen3`, `mistral`, and `qwen3vl` — Qwen3-VL's text backbone,
*text-only* input), Gemma4 (`gemma`/`gemma2`/`gemma3`/`gemma4`, dense **and**
the `gemma-4-26B-A4B` routed-expert MoE — a dense shared MLP plus softmax
top-k experts per MoE layer — plus the bidirectional-attention,
embeddings-only `gemma-embedding`), Qwen3.5/3.6-MoE (`qwen35moe`),
Qwen3.5 dense (`qwen35` — the same hybrid full-attention/gated-DeltaNet layer
shape as `qwen35moe`, plain SwiGLU FFN instead of MoE routing), and Phi-3
(`phi3`, covering Phi-3 and Phi-4-mini — Llama-style attention and SwiGLU,
but with the query/key/value projections fused into one `attn_qkv` tensor,
the FFN gate and up projections fused into one `ffn_up` tensor, and LongRoPE
frequency factors on a partially-rotated head), and Mistral 3 (`mistral3`,
e.g. Ministral-3 — `llama`'s block shape plus YaRN RoPE scaling, a head
width read from `attention.key_length` rather than derived from
`n_embd / n_head`, and an attention temperature scale) — using
`F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/`Q5_0`/`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K` and the
`IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_NL`/`IQ4_XS` tensors. Weight matrices and embedding tables are read lazily from the
`mmap`ped file (dequantized one row at a time, on demand) rather than
eagerly resident, so even large models fit in modest RAM. A model split
across several files (`<name>-00001-of-000NN.gguf` …) is loaded from every
shard — the shard count comes from the `split.count` metadata key, and each
shard is mapped separately. Runs on CPU or,
via `backend = vulkan`/`metal`/`cuda`/`opencl`/`rocm`/`auto`
(see **GPU backend** above), a Vulkan/Metal/CUDA/OpenCL/ROCm-capable GPU —
Vulkan and Metal are the same engine and are the only ones with real
fused/GPU-resident
optimizations beyond a basic matmul kernel, verified against real AMD
and Apple hardware respectively; the other three are real but
smaller-scoped and unverified on real hardware.

A quantization label names the file's *dominant* type, not its only one. A
K-quant block is 256 elements wide, so every tensor it covers needs a row
length divisible by 256; where a model's rows aren't, upstream's quantizer
substitutes a narrower type row by row. `unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q2_K`
is the common case — its `embedding_length` is 896, which is 28 blocks of 32
but not a multiple of 256, so the file that download produces is mostly
`IQ4_NL` and `Q5_0`, with `Q3_K` only on the 4864-wide `ffn_down` rows and
`Q8_0` on the embedding table. Every one of those types is read, so the model
loads and runs; what the label predicts is the size, not a single tensor type.

Type coverage differs by backend. Only `cpu` reads every type listed above.
`vulkan` and `metal` — the same kernels — cover all of them except `IQ1_S`,
`IQ1_M`, and `IQ2_XXS`; `cuda`,
`opencl`, and `rocm` cover the float types, the legacy quants,
`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`, and `IQ4_NL`. What's missing in each case
is the `IQ*` types that index a lattice codebook the backend has no uploaded
buffer for. A model carrying a type the selected backend lacks is refused at
startup, naming each missing type, rather than failing partway through the
first request.

Six further types load that upstream cannot read at all: `Q4_0_4_4`,
`Q4_0_4_8`, `Q4_0_8_8`, and the `IQ4_NL_4_4`/`_4_8`/`_8_8` equivalents.
ggml retired those ids and `llama.cpp` refuses such a file outright
("TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking"). They are
ARM-SIMD *pre-repacked* `Q4_0`/`IQ4_NL`: the packing interleaves 4 or 8
rows, and for the `Q4_0` family also flips a bit per nibble. That is a
lossless permutation, so orangu undoes it once when the model opens and
serves the result as ordinary `Q4_0`/`IQ4_NL`. Quality is identical to a
plain build of the same weights — bit-identical, not merely close — and no
GPU backend needs a kernel for any of them. One consequence worth knowing:
those tensors are held in memory rather than read from the mapped file,
because interleaving rows leaves no row with a contiguous range to be lazy
about.

Not yet built, and out of scope for now: multimodal input, `/infill`,
`/rerank`, LoRA hot-swap, and slot save/restore.

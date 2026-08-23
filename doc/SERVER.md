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

One further subcommand is neither serving nor inventory: `bundle` writes a
single executable carrying both this server and a model, which then runs
with no models directory and no configuration file at all. See **`bundle`**
below.

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
API key    No
TLS        No
Workspace  /home/user/src/orangu
Frequency  Powersave
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

`API key` and `TLS` are the two deployment gates, each simply `Yes` or `No`,
reported on every start rather than only when something is missing — a row
that always has a value is one you can check, where a warning that appears
conditionally is one you learn to expect the absence of. Read them against
the address on the line above: two `No`s beside a loopback bind are the
default and are fine, and the same two beside `0.0.0.0` mean the machine is
serving an inference engine to the network unauthenticated and in the clear.
`api_key` and `tls_cert`/`tls_key` under **Configuration** below are the
settings that answer them.

`Frequency` is the CPU's scaling governor: it decides whether a core holds
its clock through the bursty CPU work between GPU submissions, so
`Performance` is what makes a throughput number comparable and anything else
is worth seeing before reading one. Change it with `sudo cpupower
frequency-set -g performance`; the server cannot, the file being root-owned
`sysfs`. On a machine with no `cpufreq` at all the row is absent rather than
guessed.

An AMD GPU has the same kind of setting and it is **not** on the banner:
`power_dpm_force_performance_level`, which at its default `auto` lets the
core clock idle down between submissions — and decode submits in short
bursts with gaps, which is exactly the pattern that setting reads as idle.
Check it per card and pin it before measuring anything:

```sh
cat /sys/class/drm/card1/device/power_dpm_force_performance_level
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

`auto` and `low` let the clock drop; `high`, `manual` and the `profile_*`
levels hold it up. Card numbering is the kernel's, not this server's — a
machine with a discrete card and an integrated one has both, and only the
card actually serving the model matters (the GPU listing above the banner
names the one that was taken). The setting does not survive a reboot.

The server does not change it — the file is root-owned — and no longer
warns about it either: it used to print one `Note` per card on every start,
which on a machine with a discrete card and an integrated one is two lines
saying the same thing, every time, whether or not the card in question was
the one serving the model.

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
orangu-server download unsloth/Kimi-K3-GGUF -y             # skip the cannot-run confirmation
```

Every download **plans the repo against this machine first** — the same
report `plan` gives for a local model, from each shard's GGUF header
fetched over HTTP rather than from the model. Only the headers are
transferred and the connection is then dropped, so planning a 1.3 TiB repo
costs seconds:

```
Download   unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M · 4f2c9ab · 18.30 GiB
Model      qwen3 · 1 shard · 17.04 GiB on disk
Dense      3.12 GiB — attention, norms, embeddings, shared experts. Must be resident.
Experts    13.92 GiB — 128 per layer x 48 layers, 111.40 MiB each. Can stream.
Per token  891.20 MiB of experts (8 of 128 per layer, 48 layers)
This box   24.41 GiB RAM available, 3.98 GiB VRAM (AMD Radeon RX 5500M)
Verdict    fits entirely in RAM (7.37 GiB to spare); nothing needs to stream
Device     3.12 GiB of weights on a 3.98 GiB GPU — fits, 878.08 MiB spare
```

Only a model whose **dense** part does not fit stops to confirm — that one
cannot run at any speed. A model whose *experts* do not fit never prompts:
those stream, which is what the streaming path is for. `-y`/`--yes` skips
the confirmation. Planning is advisory and never a gate: an unreachable
Hub, a private repo, or an unparseable header prints one line and the
download proceeds, since it is about to report the real problem itself.

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

POWER
  Source           : Mains (battery 98%)
  k10temp Tctl     : 70.8 °C
  acpitz_0 temp1   : 58.0 °C
  amdgpu junction  : 52.0 °C (critical 100.0 °C)

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

The `POWER` section answers two environmental questions. **Source** is where
the machine is drawing from: on battery, platform power management drops the
CPU governor and the GPU clock, so a throughput figure measured there is not
a figure about this machine. No battery reads `Mains`; a platform that will
not say reads `Unknown`, never a blank. Under it are the three warmest
sensors, hottest first, with the critical threshold where the platform
declares one. Temperatures come from `sysinfo`; the power source does not
(it has no battery API) — `sysfs` on Linux, `pmset` on macOS,
`GetSystemPowerStatus` on Windows. The whole section is omitted where
neither is known, which is normal in a container.

At startup the same data drives the `Note` lines: one when running on
battery, and one when a sensor is already within a tenth of a critical
threshold the platform declared. Those two are all that remain — they are
*conditions*, and neither has a command as a fix. The machine *settings*
that once printed beside them are elsewhere now: the CPU governor is the
banner's `Frequency` row, and AMD GPU power levels are documented under
**Quick start**, beside it, rather than reprinted per card on every start.

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
  512K     -                  -                    -
  1M       -                  -                    -

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
  512K     ~14B parameters    ~9B parameters       ~4B parameters
  1M       ~4B parameters     ~3B parameters       ~1B parameters
```

Prints the same OS/CPU/GPU report `system` does, then estimates how large a
model (in parameters) is likely to run comfortably — as a table, one row per
context length (1K to 1M tokens) and one column per quantization (`Q2_K`,
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

Because every figure here is an estimate from a parameter count, the report
closes by pointing at the two commands that do not have to estimate:
`download` reads a repo's real tensor tables before fetching it, and `plan`
reads a local model's. A size class is where model selection starts, not
where it ends.

### `list` and `show`: reading GGUF files

```sh
orangu-server list
```

```
NR  MODEL                                        QUANT   SIZE        LAST_USED        SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF    Q4_K_M  17.28 GiB   2026-08-24 15:42  Yes (qwen3)
 2  unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF  Q4_K_M  270.14 GiB  Never             Yes (qwen3)
 3  ggml-org/gemma-4-12B-it-GGUF                 Q4_K_M  7.14 GiB    2026-08-21 09:16  Yes (gemma4)
 4  unsloth/GLM-5.2-GGUF                         Q4_K_M  433.83 GiB  Never             No (glm-dsa)
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
NR  MODEL              QUANT   SIZE        LAST_USED        SUPPORTED
 1  acme/Test-3B-GGUF  Q4_K_M  468.64 MiB  2026-08-24 15:42  Yes (qwen2)
 2  acme/Test-3B-GGUF  Q8_0    4.97 GiB    Never             Yes (gemma4)
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

`LAST_USED` is the local date and time at which that model last completed
server startup. Resolving it for `show`, `plan`, or shell completion does not
count; neither does a startup that fails while building the backend or binding
its listener. `Never` means the model has not been successfully served since
this tracking was introduced. Downloads and successful starts are recorded in
the versioned JSON file `~/.orangu/models`, including the model name, canonical
path, download time, and last-use time. A model copied into the directory by
hand gets a registry entry on its first successful start, and deleting a model
removes its entry.

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
NR  MODEL                                      QUANT   SIZE       LAST_USED        SUPPORTED
 1  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  Q4_K_M  17.28 GiB  2026-08-24 15:42  Yes (qwen3) (Refresh)
 2  ggml-org/gemma-4-12B-it-GGUF               Q4_K_M  7.14 GiB   Never             Yes (gemma4)
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

## `bundle`: the server and a model as one file

```sh
orangu-server bundle unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --all -y
```

```
Model      unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (2.89 GiB)
Role       all
Binary     /usr/local/bin/orangu-server (57.10 MiB, x86_64)
Output     ./orangu-server-bundle-x86_64 (2.95 GiB)
Wrote ./orangu-server-bundle-x86_64 (2.95 GiB)
```

`bundle` writes a **new executable** carrying both this server and the model
it should serve. Running it needs nothing else — no models directory, no
download, and no `orangu-server.conf`:

```sh
chmod +x orangu-server-bundle-x86_64
./orangu-server-bundle-x86_64
```

```
Model      unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (gemma4 arch, CPU/AVX2, 30 layers, 32768 ctx)
Bundled    2.89 GiB embedded in /home/you/orangu-server-bundle-x86_64
UI         http://127.0.0.1:8200
API        http://127.0.0.1:8100
API key    No
TLS        No
Workspace  /home/you
Frequency  Performance
```

That is the whole point: one file to copy to a machine, and a working
OpenAI-compatible server on it. The model is *inside* the binary — not
downloaded on first run, not extracted to a cache directory, not referenced
from one — so the binary is as large as the model, and copying it copies
everything.

### Choosing the model and the role

Both come from the command line, or from the same prompts an ordinary
interactive `orangu-server` start uses:

```sh
orangu-server bundle                                    # prompts for model, then role
orangu-server bundle 3 --code                           # NR from `list`, coding role
orangu-server bundle ./my-model.gguf -o ./my-server     # a local file, a chosen output name
orangu-server bundle unsloth/gemma-4-E2B-it-GGUF:Q4_K_M # a repo, fetched first if not cached
```

With no model argument, `bundle` prints the same table `list` does and
prompts for one, with `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` — the project's
default — ghosted as the answer an empty line takes. Unlike the serving
picker, an empty (or missing) `models` directory is not an error here:
nothing has to be installed to bundle, since the answer is a spec and a spec
that names a Hugging Face repo is fetched.

The role prompt follows, exactly as at startup, unless
`--all`/`--code`/`--review`/`--explorer`/`--embedding` was passed. Those work
both after the subcommand (`bundle <model> --code`, which reads most
naturally) and before it (`--code bundle <model>`). `-y`/`--yes` skips the
role prompt as well as the confirmation, taking `all`.

The role travels *with* the bundle: a `--code` bundle comes up in the coding
role wherever it's run, with no flag needed.

| flag | |
|---|---|
| `-o`, `--output` | where to write. Default `./orangu-server-bundle-<arch>` (plus `.exe` for a Windows binary) — never the running binary's own name, so a `bundle` run in a directory holding one can't overwrite it. |
| `--binary` | the executable to bundle *into*, instead of this one. For bundling a build for another platform, which can't be run here to bundle itself. |
| `--host`, `--port`, `--web` | the address the bundle listens on by default, baked in. `--host` takes `all`/`*` or a literal IP; `--web 0` builds a bundle with no web console. |
| `-y`, `--yes` | skip the confirmation, and take the default role rather than asking. |

### Baking in the address

`bundle` takes `--host`, `--port` and `--web` too, and records them *in* the
bundle — exactly as it records the role. A bundle is started without a config
file, so where it listens has to be decidable when it is built, not only when
it is run:

```sh
orangu-server bundle <model> --all --host all -y   # LAN-reachable wherever it lands
orangu-server --host all bundle <model> --all -y   # the same, before the subcommand
orangu-server bundle <model> --all --web 0 -y      # API only, no web console
```

```
Model      unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (2.89 GiB)
Role       all
Listen     API all:8100, console all:8200
```

The `Listen` line spells out in full what the bundle will be reachable on,
defaults included. Without any of these flags a bundle keeps the built-in
`127.0.0.1:8100` and `127.0.0.1:8200`.

The address is checked at build time rather than left for the target machine's
`bind` to reject: `--host` accepts `all`, `*`, or a literal IP address, and a
hostname or a typo is an error while there is still somebody to tell. Whatever
is baked in is a *default*, not a lock — the same `--host`/`--port`/`--web`
flags at run time still override it, and so does a config file.

### The architecture is in the name

The default output is `orangu-server-bundle-<arch>` — `orangu-server-bundle-x86_64`,
`orangu-server-bundle-aarch64`, `orangu-server-bundle-x86_64.exe`. A bundle is
a file that gets copied around, and its one hard requirement is a machine that
can run it, so a directory holding bundles for three platforms has to stay
readable; three files called `orangu-server-bundle` would not be. It also stops
a cross-bundling run from writing over the bundle it made a moment ago for a
different target.

The architecture is read out of the **binary being bundled**, not taken from
the machine doing the bundling — ELF's `e_machine`, Mach-O's `cputype` (a
universal binary is named `universal`), or PE's `Machine`, which also decides
the `.exe` suffix. That is what makes `--binary` work: cross-bundling an
`aarch64` build on an `x86_64` host produces `orangu-server-bundle-aarch64`,
not a mislabelled `x86_64`. `bundle` prints the detected architecture on its
`Binary` line so a wrong reading is visible before anything is written. An
executable format it doesn't recognize falls back to this machine's own
architecture rather than refusing to bundle — the name is a label, not
something anything resolves against.

### What a bundled server does differently

Only what it has to:

- **No config file is required.** With none found, a bundled server uses the
  address it was bundled with — `127.0.0.1:8100` for the API and
  `127.0.0.1:8200` for the web console unless `bundle` was given
  `--host`/`--port`/`--web` — plus the Hugging Face hub cache as its `models`
  directory, and the role the bundle was built with. Loopback rather than the usual `all`: a binary somebody
  downloaded and ran should not put itself on every interface of a network
  it knows nothing about. An `orangu-server.conf` that *is* found is used in
  full, exactly as for any other server — including `host = all` to opt back
  in.
- **It serves its own model without asking.** There is nothing to choose
  between, so no model prompt and no role prompt appear. Naming a model on
  the command line still overrides it (`./orangu-server-bundle-x86_64 ./other.gguf`),
  and `--daemon` works with no `[orangu-server].model` key, since the bundle
  answers the question that key exists for.
- **The embedded model can't be deleted.** It isn't a file in the models
  directory, so it has no row in the web console's model manager and no
  Delete button anywhere; the console marks the header `bundled` instead of
  leaving an unexplained gap. Removing it means removing the binary.
- **Loading a different model still works.** The console's **Load** button
  re-executes the bundle with another model, as always; if that model fails
  to load, the fallback comes back to the embedded one rather than going to
  the network for it.

Everything else — endpoints, roles, backends, the web console, sessions — is
the same server, because it *is* the same binary with bytes after it.

### Overriding the address and ports

`--host`, `--port` and `--web` override whatever the config file (or, for a
bundle, the built-in defaults) resolved to:

```sh
./orangu-server-bundle-x86_64 --host all              # every interface, not just loopback
./orangu-server-bundle-x86_64 --host 0.0.0.0          # the same thing, spelled out
./orangu-server-bundle-x86_64 --port 9100 --web 9300  # both listeners moved
./orangu-server-bundle-x86_64 --web 0                 # web console off
```

`--host` takes `all` (or `*`) for every network interface, or a literal
address — the same values `[orangu-server].host` accepts. It is how a bundle
that was not built with an address of its own gets exposed to the network for
one run, without writing a config file for it; to make that a bundle's
*default*, pass the same flags to `bundle` itself (see **Baking in the
address** above).

It moves the **web console with the API**, since the two share an address
unless something says otherwise. The one exception is a config file that set
`[web].host` explicitly: that address stands, so an API deliberately separated
from the console cannot be exposed in a way that quietly exposes the console
too. Use `--web 0` to turn the console off entirely when only the API should
be reachable.

Where to listen is the setting that is routinely per-*run* rather than
per-machine — a second server alongside one already on 8100, a port a firewall
happens to allow, a bundle that should be reachable from the LAN for one
afternoon — and a bundle may have no config file to edit. All three flags apply
to an ordinary `orangu-server` too.

### How it works

The bundle is this binary's program image, byte for byte, with the model's
`.gguf` bytes appended after it and a manifest and 32-byte footer after
those:

```text
[ program image      ]  unchanged — the OS loader reads this and stops
[ padding to 4 KiB   ]
[ shard 1 .gguf      ]  and further shards for a split model
[ manifest (JSON)    ]  model, quantization, role, where each shard landed
[ manifest offset+len, magic ]
```

Appending to an executable leaves it runnable: the loader reads the program
headers at the front and never looks past what they describe. At startup the
server seeks to the last 32 bytes of its own file, and either finds the
magic — in which case the model is memory-mapped straight out of the
executable, with no copy and no unpacking — or doesn't, in which case it is
an ordinary `orangu-server` and behaves exactly as before. Shards start on a
4 KiB boundary so a bundled model's tensor data is aligned exactly as it
would be in a file of its own.

Because the manifest records where the program image ends, a bundle can be
bundled again: `./orangu-server-bundle-x86_64 bundle <other-model>` replaces the
model rather than stacking a second one behind the first.

On macOS the copied program image is re-signed ad-hoc (`codesign --force
--sign -`) *before* the model is appended to it — `codesign` writes the new
signature at the end of the image it is given, so signing afterwards would
write it straight over the payload. Signing first leaves the model outside
the signed range, where the kernel never looks. If `codesign` isn't
available, `bundle` says so and names the command to run — without it macOS
kills the bundle on sight.

Releases ship the ordinary `orangu-server`, which includes `bundle`; the
bundles themselves are built locally, from whichever model suits the
machine.

## Configuration

`orangu-server.conf`:

```ini
[orangu-server]
models = ~/models
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
host = all
port = 8100
slots = 1
api_key = a-long-random-string
tls_cert = ~/certs/cert.pem
tls_key = ~/certs/key.pem
queue_limit = 0
draft_model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
backend = auto
kv_cache = f16
read_size = 8192
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
  terminal to prompt on — unless the binary is a bundle, which carries its
  own model and so needs no key to name one. A `--daemon` run that *is* given
  a positional model argument uses that instead of this key. An attached run
  still takes its model from the CLI argument when one is given; when none is, the interactive picker
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
  just that one. `--host` on the command line overrides this for one run —
  `--host all` exposes a server that a config (or a bundle's own defaults)
  keeps on loopback, without editing anything. It moves `[web].host` with it
  unless that key was set explicitly, so an API and a console a config
  deliberately separated stay separated. `-p`/`--port` overrides `port` the
  same way, and `--web` overrides `[web].port` (`--web 0` turns the console
  off) — useful for a second server alongside one already on 8100, and for a
  bundle with no config file to edit.
- `slots` — how many requests generate concurrently, each with its own KV
  cache (default `1`). Raise it to serve overlapping requests without
  queuing behind each other.
- `tls_cert` / `tls_key` — PEM paths for serving HTTPS. Both or neither; one
  alone is a startup error, because the alternative is serving in the clear
  while looking configured. PKCS#8, PKCS#1 and SEC1 keys all load. Terminating
  in a reverse proxy stays valid — this exists so one binary on one machine is
  not forced into one.
- `api_key` — bearer token every request must carry (`Authorization: Bearer
  <key>`). Unset by default: the server is open, which suits the loopback bind
  it also defaults to and does not suit a widened `host`. `ORANGU_API_KEY`
  overrides the file so the secret need not be written down. Only `/health` is
  exempt; everything else answers `401` with `WWW-Authenticate: Bearer`.
- `device` — *which* card when `backend` finds more than one: `auto` (the
  default), an index as printed at startup, or any part of the device's name.
- `device_split` — spread one model's layers across several devices; `off` by
  default. See **Splitting a model across devices**.
- `threads` — CPU worker threads; rayon's own choice by default.
- `queue_limit` — how many requests may wait for a slot before the server
  refuses with `503` + `Retry-After` (default `0`, unbounded). `id_slot`-pinned
  requests bypass it. Depth and limit are exported on `/metrics` as
  `orangu_server_queue_depth` / `orangu_server_queue_limit`.
- `draft_model` / `draft_tokens` — speculative decoding: a second, smaller
  model guesses `draft_tokens` tokens (default 4) and the served model verifies
  them in one forward, keeping the longest prefix it would have produced
  itself. The output is unchanged; only the time taken differs. The pair must
  share a vocabulary and both must have a multi-position forward (`gemma4`,
  `deepseek4`, `glm-dsa`, `muse-glimmer`), both checked at startup. Greedy,
  unconstrained requests only. Whether it pays depends on the hardware — see
  **Speculative decoding** in the manual for a measurement where it does not.
- `kv_cache` — how the GPU-side KV mirror is stored: `f16` (the default),
  `q8_0`, or `f32`. `q8_0` is ~44% smaller than `f16` and cuts attention's
  read bandwidth at long context (−32% attention GPU time at ~295 tokens,
  growing with the cache), buying context and concurrent `slots` out of the
  same VRAM — and is the only **lossy** setting here, which is why it is
  opt-in. Vulkan-family backends only. `ORANGU_KV_CACHE` overrides it.
- `read_size` — the smallest explicit read of a model file, **in KiB**;
  default `8192` (8 MiB), `4` disables widening, and it must be a positive
  multiple of `4` because a read has to be a whole number of pages. Widening
  measured **+36% decode tok/s** on a cold mixture-of-experts model and cut the
  run-to-run spread from 2.25–8.65 tok/s to 7.89–8.44: large sequential reads
  hold the device in its fast regime. Warm, the setting is within noise. Storage throughput is closer to a step than a slope:
  below a device-specific request size every read pays a full round trip and a
  small read costs about what a large one does; above it the block layer
  splits the request into several commands and issues them together. Measured
  on one USB-bridged SSD the step sits at 512 KiB — 15–28 MB/s at or below it,
  206–214 MB/s from 1 MiB up. Where it falls is a property of the controller,
  the bus and any bridge in front of them, which is why this is configurable.
  A span smaller than `read_size` is widened outward to it and the wanted
  bytes taken from the middle; a larger span is read as itself. Only the
  explicit read routes (`ORANGU_EXPERT_READ=pread|direct`) consult it — the
  default memory mapping leaves request size to the kernel's readahead, so on
  a default deployment this key changes nothing.
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

  A model whose format makes reasoning a *separate message* rather than a
  tagged span — `muse-glimmer`, which addresses one message `to=self` and
  the next `to=user`, and `inkling`, which opens one with
  `<|content_thinking|>` and the next with `<|content_text|>` — is handled
  exactly rather than approximated: the reasoning message is dropped from
  the reply, and no `<think>` block is prefilled (that prefill would land
  inside the message header the first format leaves open, or ahead of the
  marker that types the body in the second, and the reply came back empty
  when it did).

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

Every code block carries a file name and a **download** button at its
lower right — the console's standard place for a download, the same as the
save control under a finished answer — so a single block can be saved on
its own rather than selected by hand out of a transcript. The name comes
from the fence
(```` ```rust src/main.rs ````, ```` ```rust:src/main.rs ````,
```` ```rust title="src/main.rs" ````), or from a first line that is a
comment holding nothing but a name (`// src/lib.rs`), or failing both from
a generated `orangu-snippet-<n>.<ext>` numbered by position and extended
from the fence's language.

Every code block is shown with the MIT licence as a header, written as a
comment in that block's own language (`//`, `#`, `--`, `;;`, `%`, `REM`, or a delimited
`<!-- -->` / `/* */` block), below any shebang or XML declaration — so the
download saves what is on screen. The reply as the model wrote it is
untouched; the licence is added when the message is rendered. The text is a
string compiled into the binary — not a file beside it — with `<YEAR>`
filled in at render time. A file whose comment syntax isn't known (`.json`, `.csv`, an
untagged fence's `.txt`) saves without a header rather than with a guessed
one. The same header, from the same module (`orangu::license`), is what
`orangu`'s `create_file` tool writes onto a file it generates. See the
manual's **Web UI** chapter for the full rules.

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

The topbar's **Models** button opens a panel showing the models directory from
the same scan as `orangu-server list`, with its core numbered inventory fields:

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

Three answers apply to every row rather than to any one of them: `401` when
`api_key` is set and the request carries no valid bearer token (`GET /health`
and `GET /ready` excepted); `503` with `Retry-After` when `queue_limit` is
reached; and `https` on the same port when `tls_cert`/`tls_key` are set.

| Endpoint | |
| :-- | :-- |
| `GET /v1/models` | |
| `POST /v1/chat/completions` | streaming (SSE) and non-streaming; requires the model to have a `tokenizer.chat_template`; disabled under `--embedding`. Accepts `response_format` — `{"type": "json_object"}` constrains sampling so only tokens keeping the output a valid JSON object can be chosen, and withholds end-of-sequence until the document is complete. `json_schema` is treated the same way: valid JSON, not the schema's shape |
| `POST /v1/completions` | legacy OpenAI completion, no chat template needed; disabled under `--embedding` |
| `POST /v1/embeddings` | pooled (mean or last-token, per the model's own `pooling_type`) and L2-normalized |
| `GET /health` | liveness — stays `200` while the server is busy |
| `GET /ready` | readiness — `503` (with a reason) when the admission queue is full or the GPU device was lost. Open without an `api_key`, like `/health` |
| `GET /props` | model + server metadata |
| `GET /slots` | per-slot busy/prompt/generated-token state |
| `GET /metrics` | Prometheus text: slot and queue gauges; latency histograms (`queue_wait`, `time_to_first_token`, `inter_token`, `request`); counters for requests by outcome and for prompt/cached/generated tokens |
| `GET /moe-stats` | mixture-of-experts counters since the previous call, **and reset** — expert visits, the per-layer-call union, rows and bytes dequantized, plus the process's fault and RSS figures. Drain once before a workload and again after to measure exactly that window. Dense models report `layer_calls: 0` |
| `orangu-server plan <model> [--deep]` | (a subcommand, not an endpoint) Reports what a model would need to run here **without loading it** — dense vs routed-expert bytes, experts streamed per token, and a verdict. Reads only the GGUF tensor tables, so a 434 GiB 11-shard model takes well under a second. `--deep` also checks every shard is present and the architecture supported |
| `GET /gpu-timings` | per-stage GPU timings for the last decode step, when `ORANGU_GPU_TIMESTAMPS=1` asked for them |
| `POST /slots/{id_slot}` | `?action=save\|restore` — persist or reload that slot's KV cache |
| `GET /model-cache` | how many of the model's bytes are in the page cache right now. `resident_bytes` is `null` where the platform cannot measure it — never `0`, which would make "unknowable" read as "cold" |
| `POST /model-cache/drop` | evict the model from the page cache so the next request reads it from disk; loopback-only. Reports residency before and after rather than a success flag, because a partial drop is the realistic failure and looks identical from outside |

`/moe-stats` also carries a `store` block describing where routed experts'
weights came from. Three environment variables govern it, all off by default:

**Boolean flags read `0`, `false`, `no`, `off` and the empty string as OFF**, and anything else as on. They used to be presence-checked, so `FLAG=0` switched the feature *on* and a sweep of `0,1` measured it against itself. The variables that carry a value or a path — `ORANGU_NORM_WG`, `ORANGU_COOP_GEOM`, `ORANGU_DUMP_SHADERS`, `ORANGU_EXPERT_USAGE`, `ORANGU_PREFIX_CACHE_DIR` — are parsed or presence-checked instead, as their descriptions say.

| variable | effect |
| :-- | :-- |
| `ORANGU_EXPERT_CACHE_GB` | Size of an in-process expert weight cache, in GiB. Unset or `0` keeps the incumbent behaviour — weights read straight from the `mmap`, placement left to the OS page cache. Worth setting only when the model does **not** fit in RAM: below that the page cache already holds every expert and a cache can only duplicate memory. Even above it, measurement has not yet found a budget where the cache beats the page cache — it competes with it for the same RAM, and the page cache gets the rest of the machine |
| `ORANGU_EXPERT_READ` | Where the expert cache's copies are read from: `mmap` (default — a memcpy from the page cache), `pread` (an explicit read of the shard, still cached), or `direct` (`O_DIRECT`, bypassing the page cache). **`direct` is dramatically slower on a model the page cache can hold** — measured 54x on a 26B MoE — because it converts every memcpy into a disk read. It is for models far larger than RAM |
| `ORANGU_EXPERT_CACHE_POLICY` | Replacement rule: `lfru` (frequency-first with an admission margin), `lfu` (the same scoring, admitting every miss), or the default `lru`. Which one wins depends on the regime, so the default is not a recommendation: on a model that fits in RAM under a small budget, LRU measured 12x better than LFRU; on one whose experts genuinely do not fit, LRU gets **zero** hits at a tight budget while both frequency rules score, because a routing pass is a scan and recency evicts every expert before it comes round again. The rule actually in force is reported back in `/moe-stats` |
| `ORANGU_PREFIX_CACHE_DIR` | Directory for a durable snapshot of the prefix-cache pool, so a conversation survives a restart instead of re-prefilling. Needs `ORANGU_PREFIX_CACHE` as well. A snapshot carries the model's fingerprint and is refused for any other model — a KV cache from elsewhere would match on token ids and answer from the wrong state. Sized per entry as a whole KV cache (~330 KB per position on a 26B MoE), so it is opt-in |
| `ORANGU_EXPERT_BUDGET` | Cap on distinct experts evaluated per layer at decode. **Changes what the model computes** — the only setting here that does — and is off by default. Never applied to prefill, and never leaves a position with nothing routed |
| `ORANGU_ROUTE_AHEAD` / `ORANGU_PREFETCH_K` | Measure how predictable the next layer's routing is, and prefetch the top `k` of that prediction. `k` above 2 wastes more than 20% of what it fetches on the model this was measured on |
| `ORANGU_EXPERT_RESIDENCY` | `1` asks the kernel (`mincore`) whether each expert's bytes were in RAM at the moment they were wanted. Off by default because it is one syscall per expert per layer; with it off, acquisitions report as `unmeasured` rather than as misses |
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

Text-in/text-out GGUF chat, completion, and embedding models, for twelve
architecture families: Llama-style (`general.architecture` one of `llama`,
`qwen2`, `qwen3`, `mistral`, and `qwen3vl` — Qwen3-VL's text backbone,
*text-only* input), Gemma4 (`gemma`/`gemma2`/`gemma3`/`gemma4`, dense **and**
the `gemma-4-26B-A4B` routed-expert MoE — a dense shared MLP plus softmax
top-k experts per MoE layer — plus the bidirectional-attention,
embeddings-only `gemma-embedding`), Qwen3.5/3.6-MoE (`qwen35moe`, e.g.
`unsloth/Qwen3.6-35B-A3B-GGUF`), Qwen3.5-family dense (`qwen35`, e.g.
`unsloth/Qwen3.8-27B-GGUF` — the same hybrid full-attention/gated-DeltaNet
layer shape as `qwen35moe`, plain SwiGLU FFN instead of MoE routing),
Qwen3-Next (`qwen3next`), DeepSeek-V4 (`deepseek4`, e.g.
`unsloth/DeepSeek-V4-Flash-0731-GGUF` — four parallel residual streams mixed
per token, one shared key/value vector serving every query head, compressed
attention blocks on top of a sliding window, and hash-routed experts),
GLM-5 (`glm-dsa`, e.g. `unsloth/GLM-5.2-GGUF` — absorbed multi-head latent
attention over a compressed key/value cache, with a lightning indexer
choosing which positions each layer attends), Kimi-K3 (`kimi-k3`, e.g.
`unsloth/Kimi-K3-GGUF` — three-in-four delta-net layers alternating with
latent attention, cross-layer residuals, and experts running in a latent
space), and
Phi-3
(`phi3`, covering Phi-3 and Phi-4-mini — Llama-style attention and SwiGLU,
but with the query/key/value projections fused into one `attn_qkv` tensor,
the FFN gate and up projections fused into one `ffn_up` tensor, and LongRoPE
frequency factors on a partially-rotated head), and Mistral 3 (`mistral3`,
e.g. Ministral-3 — `llama`'s block shape plus YaRN RoPE scaling, a head
width read from `attention.key_length` rather than derived from
`n_embd / n_head`, and an attention temperature scale), and Muse-Glimmer
(`muse-glimmer`, e.g. `unsloth/Muse-Glimmer-30B-GGUF` — a dense GQA block
with a norm on both sides of each sub-layer, per-head query/key norms, a
sigmoid gate on the attention output, three rotated sliding-window layers
to every unrotated full-attention one, and both a logit scale and final
logit softcapping on the output), and Inkling (`inkling`, e.g.
`unsloth/Inkling-Small-GGUF` — a mixture-of-experts decoder that rotates
nothing at all: position arrives through a learned per-head
relative-position bias and a causal short convolution on the key/value
projections and on each sub-layer's output, layers alternate
sliding-window and full attention, and the routed experts share their
weight normalization with two always-on shared ones), and Nemotron-H
(`nemotron_h_moe`, e.g.
`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` — a hybrid whose
blocks are a *single* sub-layer each rather than the usual
attention-plus-FFN pair: a selective state-space mixer, an unrotated
attention, or a squared-ReLU mixture-of-experts FFN), and Ling 3.0
(`bailingmoe3`, e.g. `bartowski/Ling-3.0-tiny-GGUF` — three-in-four Kimi
Delta Attention layers alternating with gated, *rotated* absorbed latent
attention, over sigmoid-routed experts whose selection is group-limited:
the experts form `expert_group_count` groups and only the best
`expert_group_used_count` of them may serve a token) — using
`F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/`Q5_0`/`MXFP4`/`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K` and the
`IQ1_S`/`IQ1_M`/`IQ1_XS`/`IQ1_XXS`/`IQ1_XXXS`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_NL`/`IQ4_XS` tensors. Weight matrices and embedding tables are read lazily from the
`mmap`ped file (dequantized one row at a time, on demand) rather than
eagerly resident, so even large models fit in modest RAM. A model split
across several files (`<name>-00001-of-000NN.gguf` …) is loaded from every
shard — the shard count comes from the `split.count` metadata key, and each
shard is mapped separately. Runs
on CPU or,
via `backend = vulkan`/`metal`/`cuda`/`opencl`/`rocm`/`auto`
(see **GPU backend** above), a Vulkan/Metal/CUDA/OpenCL/ROCm-capable GPU —
Vulkan and Metal are the same engine and are the only ones with real
fused/GPU-resident
optimizations beyond a basic matmul kernel, verified against real AMD
and Apple hardware respectively; the other three are real but
smaller-scoped and unverified on real hardware.

Kimi-K3 (`kimi-k3`) runs on the CPU path only. Three layers in every four
are Kimi Delta Attention — a gated delta-net whose per-token state is a
matrix rather than a growing key/value list, so those layers cost nothing
per token of context — and every fourth is absorbed multi-head latent
attention like `glm-dsa`'s, minus the RoPE (this model rotates nothing; the
`rope.dimension_count` key only names how the cached key splits) and plus a
sigmoid gate on the attention output. Four further mechanisms have no
counterpart elsewhere here. **Cross-layer residual attention**: every
`attn_res.block_size`th layer banks its raw input and the residual stream
restarts from that layer's attention output, with each half-layer re-mixing
the stream against every banked checkpoint by a softmax over per-checkpoint
scores. **Latent MoE**: the routed experts run at `expert_latent_length`
rather than at `n_embd`, so the FFN input is projected down, run, normed and
projected back up — while the *router* still scores the full-width input.
**The situ activation** replaces SwiGLU throughout: a soft-clipped SiLU on
the gate branch, and the same soft clip on the up branch when
`activation.situ_linear_beta` is positive. **A full-rank KDA gate**, where
Kimi-Linear factors the same gate into two matrices. The delta-net state is
what dominates memory: a fixed `kda.head_dim`-squared matrix per head per
recurrent layer, about 440 MiB per sequence for Kimi-K3, allocated up front
and independent of context length. The multimodal projector these repos ship
alongside the text weights (`mmproj-*.gguf`) is not used; multimodal input is
out of scope for every architecture here.

GLM with DeepSeek sparse attention (`glm-dsa`) runs on the CPU path only.
Its block shape is an ordinary pre-norm transformer, and its FFN is the same
routed-experts-plus-shared-expert MoE as `qwen35moe` (dense for the first
`leading_dense_block_count` layers); what is different is the attention.
Keys and values are stored *compressed*: one `attention.kv_lora_rank`-wide
vector per token plus a shared rotary part serves every head, so even
GLM-5.2's 79 layers keep a small cache. Rather than decompressing that back into
per-head keys, the query is pushed through the key-decompression matrix
(`attn_k_b`) so it can be dotted against the compressed vector directly, and
the attention output is pushed back through `attn_v_b` afterwards — which is
also why the cache is K-only, the value being the leading part of the same
row. On top of that, a lightning indexer (a small 32-head attention with its
own per-token key cache) scores every earlier position and the real
attention attends only the `attention.indexer.top_k` best; only some layers
score, the rest reusing the previous scoring layer's choice
(`attention.indexer.types`, defaulted from the reference config when the
file omits it, as GLM-5.2's quants do). Below `indexer.top_k` positions the
selection cannot change the answer — every visible position is chosen — so
the scoring pass is skipped there. The multi-token-prediction block these
files carry (`blk.78` in GLM-5.2) is a draft head and is not run.

DeepSeek-V4 (`deepseek4`) runs on the CPU path only, and differs from every
other family here in four ways at once: the residual stream is
`hyper_connection.count` parallel streams rather than one, mixed down and
back out per half-layer by weights the model predicts per token (the
out-mix is made doubly stochastic by a Sinkhorn normalization);
`attention.head_count_kv` is 1 and the value *is* the key, so all 64 query
heads attend one shared vector per token, whose trailing RoPE dimensions are
rotated back out of the attention output again; `attention.compress_ratios`
gives each layer a sliding window plus either whole 128-token compressed
blocks or 4-token blocks chosen by the model's own lightning indexer, both
pooled by a per-dimension softmax over their members; and the first
`hash_layer_count` layers pick their experts from an integer
`ffn_gate_tid2eid` table indexed by token id rather than by score. Its
compressed blocks live in the same positional KV cache as its per-token
keys — one row per block — so context rollback, prefix reuse, and slot
persistence cover all of it. That cache is wide: on top of the shared
512-wide key each layer keeps its compressor's per-token value/score rows,
which for `DeepSeek-V4-Flash-0731` works out to roughly half a megabyte per
token of context across all 43 layers, allocated up front for the
prompt-plus-`max_tokens` budget of each request.

Muse-Glimmer (`muse-glimmer`) runs on the CPU path only. It is a dense
grouped-query decoder, and most of what it adds to the ordinary block it
borrows from families already here: a norm on both sides of each
sub-layer (`attn_norm`/`post_attention_norm`, `ffn_norm`/`post_ffw_norm`)
as Gemma has; per-head query and key norms; a sliding window of 2048 on
three layers in every four, the fourth attending the whole prefix
(`attention.sliding_window_pattern`); and a sigmoid gate on the attention
output as Kimi-K3 and Qwen3.5 carry — here its own `attn_gate` tensor,
projected from the same normed layer input as query/key/value and
multiplied into the attention output before the output projection. What
has no counterpart elsewhere here is that **the rotation runs on the
sliding-window layers only**: the full-attention quarter rotates nothing,
and nothing in the file says so. Its output logits are scaled
(`logit_scale`) and then soft-capped (`final_logit_softcapping`), and its
token embeddings are normalized on the way in. The multimodal projector
shipped beside the text weights (`mmproj-*.gguf`) is not used, as for
every architecture here.

This model's prompt format is worth knowing about, because an assistant
turn is several *messages* rather than one. The chat template ends the
generation prompt at `<|start|>assistant` and leaves the model to write its
own recipient — `to=self` for a reasoning message, `to=user` for the
answer, `to=<tool>` for a tool call — before the `<|message|>` that starts
the text. The markers are control tokens and are filtered out like any
others; the recipient is ordinary text, and it is dropped too, so a reply
never begins ` to=user`.

Which messages you see is the server's **role**, the same switch that
governs reasoning everywhere else. A reasoning-suppressing role
(`--review`) shows the message addressed to you and nothing else; every
other role shows the reasoning first, then a blank line, then the answer —
the same treatment a `<think>`-style model's reasoning already gets here.
Note that this model reasons on every turn: its template writes
`Reasoning strength: high` into the system block itself and does not read
the `enable_thinking` flag, so `--review` is what turns the reasoning off
in the reply, not in the generation.

Not implemented: reporting reasoning separately as `reasoning_content`
rather than inline, and parsing this model's XML-shaped tool calls (a
`to=<tool>` message reaches you as its literal markup).

Inkling (`inkling`, e.g. `unsloth/Inkling-Small-GGUF`) runs on the CPU path
only, and is the first architecture here that **rotates nothing** — no
layer applies a rotary embedding. Position reaches attention two other
ways. The first is a learned relative-position bias: each layer projects
its input to a small per-head vector and mixes it against a per-layer bank
into one additive term per query/key *distance*, so a key further back than
the bank is wide contributes no bias at all and a short bank still serves a
long prefix. The second is a causal depthwise short convolution — four of
them per layer, of the width `inkling.shortconv_kernel` gives: on the raw
key and value projections, and on the output of each sub-layer before its
residual add. Each carries the previous few inputs forward, which is state
that outlives a decode step, so it lives in the same per-sequence recurrent
slot Qwen3.5's linear-attention layers use. That state has no per-position
history to roll back, so the opt-in prompt-lookup speculative decoding is
not available for this model, exactly as for the other recurrent families
here.

The rest is assembled from parts already present: an alternating
sliding-window/full-attention pattern read per layer, per-head query and
key norms, `dense_block_count` leading dense layers, and sigmoid-routed
experts with a selection bias. Two things differ from every other
mixture-of-experts model here. The router emits one logit per routed expert
**plus** one per shared expert, and the selected routed weights are
normalized together with the shared ones rather than among themselves.
And the full-attention layers multiply every score by a factor that grows
with the context (`inkling.log_scaling_n_floor`,
`inkling.log_scaling_alpha`), so a long conversation attends differently
from a short one; below the floor that factor is exactly 1, which is why a
short prompt cannot tell whether it is implemented at all.

Its vocabulary is padded — `inkling.unpadded_vocab_size` names how many of
its rows are real tokens — and the padding rows are masked out of the
logits, since one of them can otherwise win an argmax and decode to
nothing. The audio and image inputs the model was trained for are out of
scope, as multimodal input is for every architecture here: the
`mmproj-*.gguf` shipped beside the text weights is a separate model this
server does not load, and the audio embedding table is not part of the text
GGUF at all.

This model's prompt format types each message *body* with a control token:
`<|content_thinking|>` opens the model's reasoning, `<|content_text|>` the
answer, and `<|end_message|>` closes either. The markers are filtered out
of the reply like any other control token, and which bodies you see is the
server's **role**, the same switch that governs reasoning everywhere else.
A reasoning-suppressing role (`--review`) shows the answer alone; every
other role shows the reasoning, a blank line, then the answer. The model
reasons on every turn regardless — its template writes a thinking-effort
line into the system block and never reads the `enable_thinking` flag — so
`--review` turns the reasoning off in the reply, not in the generation.

Not implemented for this model: reporting reasoning separately as
`reasoning_content` rather than inline, and parsing its JSON tool
invocations back into `tool_calls` (a `<|content_invoke_tool_json|>` body
reaches you as its literal JSON).

Nemotron-H (`nemotron_h_moe` and the dense `nemotron_h`, e.g.
`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` and
`bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF`) runs on the CPU
path only, and breaks the assumption every other architecture here shares:
a block is **not** an attention sub-layer plus an FFN sub-layer. It is one
or the other — or neither. Each block holds exactly one mixer under one
norm and one residual, and the file's own per-layer metadata says which:
where `feed_forward_length` is nonzero the block is a mixture-of-experts
FFN, where it is zero and `attention.head_count_kv` is nonzero the block is
self-attention, and where both are zero the block is a selective
state-space mixer. On the 30B-A3B model that is 23 state-space blocks, 23
expert blocks and just **6 attention blocks** across 52 layers, so the
great majority of the sequence mixing is recurrent and the key/value cache
covers a sixth of the depth. A long conversation therefore costs far less
cache here than its context length suggests.

Position enters this model **only** through the recurrence. There is no
rotary embedding on any layer — the attention blocks are unrotated, and the
`rope.dimension_count` and `rope.freq_base` the file still carries are
vestigial. What carries order instead is the state-space block: a causal
convolution `ssm.conv_kernel` taps wide over the projected input, then a
per-head recurrence whose state decays by a learned, input-dependent
timestep. That state is `ssm.inner_size / ssm.time_step_rank` by
`ssm.state_size` per head — rectangular, unlike the square accumulator the
gated-DeltaNet families here carry — and it is fixed, independent of how
long the conversation gets. Like every recurrent family here it has no
per-position history to roll back, so the opt-in prompt-lookup speculative
decoding is not available for this model.

Its expert layers differ from every other mixture-of-experts model here in
one respect worth naming: the FFN has **no gate projection**. Both the
routed experts and the single shared expert are squared ReLU —
`down(relu(up(x))^2)` — a two-matrix FFN rather than the three-matrix
SwiGLU everything else uses, and the shared branch is added at full
strength rather than being folded into the routing weights. The routing
itself is the familiar one: sigmoid probabilities, a correction bias that
steers the selection only, top-k, then normalization and a scale.

The file also carries a trailing multi-token-prediction block — an extra
`block_count` entry holding a self-contained draft head that predicts two
tokens ahead. Nothing in the trunk reads it, and this server has no
second-model speculative path to use it with, so it is left on disk. `plan`
reports it on its own `Draft head` line rather than folding it into either
of the two figures that decide whether a model is usable: it is neither
weight that must be resident nor weight that can stream, because it is never
read at all.

That shape is not unique to this model. `glm-dsa` and the whole Qwen 3.5
family do the same, and `unsloth/Qwen3.8-27B-GGUF` is the plainest example:
`block_count` is 65, `blk.64` is the draft head, and the trunk is the 64
blocks before it. All of them are handled the same way — the head is
identified from `nextn_predict_layers` and never loaded.

Not implemented for this model: embeddings requests, and reporting its
reasoning separately as `reasoning_content`. It reasons inline before
answering, with no marker tokens around the reasoning, so a
reasoning-suppressing role cannot separate the two.

Ling 3.0 (`bailingmoe3`, e.g. `bartowski/Ling-3.0-tiny-GGUF` and
`bartowski/Ling-3.0-flash-GGUF`) runs on the CPU path only. Its trunk is a
hybrid, and the file says so per layer: `attention.head_count_kv` is an
*array*, and a `0` entry marks a recurrent Kimi Delta Attention layer while
a nonzero one marks a full-attention layer. Three of the first for every one
of the second, so on the 24-layer tiny model six layers carry a key/value
cache and eighteen do not — a long conversation costs a quarter of the
cache its context length suggests. Like every recurrent family here it has
no per-position history to roll back, so the opt-in prompt-lookup
speculative decoding is not available for this model.

The delta-net layers are the same Kimi Delta Attention `unsloth/Kimi-K3-GGUF`
uses, and they share one implementation with it: a short causal convolution
`ssm.conv_kernel` taps wide over each of the query, key and value
projections, then a delta rule whose state decays **per dimension** rather
than by one scalar per head, then a gated per-head norm. What is specific
here is the *safe gate* (`kda.safe_gate`): the log-decay is
`kda.gate_lower_bound * sigmoid(..)` rather than an unbounded
`-exp(A_log) * softplus(..)`, so the per-dimension decay lives strictly
between `e^lower_bound` and 1 and cannot reach 0 and erase the state.

The full-attention layers are multi-head latent attention in its absorbed
form — one compressed vector per token stands in for both key and value,
and each head's query is pushed through that head's key decompression up
front, so the cache never has to be expanded. Two things separate them from
Kimi-K3's: they **do** rotate (NORM-paired RoPE over
`rope.dimension_count` of each query head's tail and of the shared key
half), and their sigmoid output gate is one scalar per head rather than one
per value dimension.

Its experts are where the new shared machinery went. The routing is
DeepSeek-V3's — sigmoid probabilities, an `exp_probs_b` bias that steers the
selection but never the weights, renormalization, then
`expert_weights_scale` — plus **group-limited selection**, which no
architecture here had before: the experts are cut into
`expert_group_count` contiguous groups (8 on both released models), each
group is scored by the sum of its *two* best members, only the best
`expert_group_used_count` groups (4) survive, and the top-k then runs over
those alone. A strong expert in a weak group is therefore not selected —
which is the point, and which a router that quietly ignored the grouping
would get wrong while still producing fluent text.

Tool calling works in the model's own format: it writes a call as
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`,
which this server already parsed — with the wrinkle that this vocabulary
spells all six of those delimiters as *tokens* rather than as text, so they
have to be exempted from the suppression that hides every other structural
token. Without that exemption the call reaches the parser as loose prose
and quietly becomes chat rather than an invocation.

Reasoning is inline, as it is for every `<think>`-tagged model here: the
tags themselves are vocabulary tokens and are hidden, but the reasoning
between them arrives as part of the answer rather than as a separate
`reasoning_content` field. A reasoning-suppressing role (`--review`) does
stop it at the source — the model's own template closes the block
immediately when `enable_thinking` is false, so nothing is generated to
hide.

Not implemented for this model: embeddings requests, and the trailing
multi-token-prediction head `Ling-3.0-flash` carries inside its
`block_count`, which is trimmed exactly as every other draft head here is.

`orangu-server list` also recognizes `dflash` draft GGUFs such as the
DeepSeek-V4-Flash DSpark sidecar. A draft carries no token embeddings and no
output projection — it reads the target model's hidden states (the layers
`dflash.target_layers` names) and drafts through the target's own embedding
table and LM head — so there is no standalone model in the file to serve.
Selecting one therefore serves the *paired target model* from the same
Hugging Face repo, downloading it first if the models directory does not
have it yet; the startup banner names the model actually being served.
Running a draft as an actual draft would
need a second-model speculative path, which this server does not have: its
speculative decoding drafts by prompt lookup against the served model
itself.

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
`vulkan` and `metal` — the same kernels — cover all of them except the three
`IQ1_*` types below `IQ1_S` (`IQ1_XS`, `IQ1_XXS`, `IQ1_XXXS`); `cuda`,
`opencl`, and `rocm` cover the float types, the legacy quants,
`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`, and `IQ4_NL`. What's missing in each case
is the `IQ*` types that index a lattice codebook the backend has no uploaded
buffer for. A model carrying a type the selected backend lacks is refused at
startup, naming each missing type, rather than failing partway through the
first request.

`IQ1_S`, `IQ1_M` and `IQ2_XXS` were in that missing list until recently, and
what they cost was whole models rather than speed: a "dynamic" 2-bit build
such as `unsloth/Qwen3.8-27B-GGUF:IQ2_XXS` is 96 `IQ1_M` and 48 `IQ2_XXS`
tensors, so the startup check refused the GPU for the entire file. All three
now have kernels, at the price of a codebook buffer that grew from about 15
KiB to about 33 KiB.

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

Three further types are narrower still, and come from *outside* the type
enum ggml maintains: `IQ1_XS`, `IQ1_XXS` and `IQ1_XXXS`, at 1.4375, 1.3125
and 1.1875 bits per weight. They are how a "dynamic" 1-bit release of a
very large mixture-of-experts model gets under its size target — the expert
stacks of `unsloth/Qwen3.8-2.4T-A95B-GGUF:Q1_0` are stored as `IQ1_XXXS`,
38 bytes per 256 weights. Each narrows exactly one field of `IQ1_S`, its
codebook index, from 11 bits to 10, 9 and 8, selecting from a 1024-, 512-
or 256-point subset of the same 2048-point lattice `IQ1_S` itself indexes.
Every other field keeps its `IQ1_S` meaning — the `f16` super-block scale,
the 3-bit sub-block scale, the `±0.125` delta — so orangu reads all three
through the same code path, bit-for-bit against the reference
implementation on both random blocks and real model tensors. Their ids are
64, 65 and 66, deliberately above the 42..63 range left free for ggml to
grow into, so a build without them rejects such a file rather than
misreading it; `list` prints anything in that gap as `reserved(N)` for the
same reason.

Not yet built, and out of scope for now: multimodal input, `/infill`,
`/rerank`, LoRA hot-swap, and slot save/restore.

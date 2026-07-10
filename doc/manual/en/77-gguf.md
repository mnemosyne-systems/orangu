\newpage

## GGUF inventory internals

`orangu-gguf` (`src/bin/orangu-gguf/`) is a third binary in the same Cargo
package as `orangu` and `orangu-coordinator`. Unlike those two, it is
entirely offline and stateless between runs: every invocation re-detects
hardware and re-scans the models directory from scratch, so there is no
cache, config-reload, or background process to reason about.

### Module layout

- `main.rs` — clap `Args`/`Commands` (`system`, `suggest`, `list`, `show`,
  `download`), dispatch, and `format_show`/`format_bytes` (the latter shared
  by `system`'s RAM/VRAM figures and `list`'s file sizes).
- `gguf.rs` — the GGUF binary-format reader (`GgufFile::open`).
- `models.rs` — recursive directory scan, shard grouping, and the Hugging
  Face repo-id/quant-tag reconstruction behind `list`'s `MODEL` column.
- `download.rs` — `download`: fetches a model from the Hugging Face Hub
  into the `models` directory in the same on-disk format `models.rs` scans;
  see below.
- `system.rs` — CPU (via `sysinfo`) and GPU (layered platform-specific
  probes) detection.
- `suggest.rs` — `suggest`: a hardware-based model-size estimate built on
  top of `system.rs`'s own detection; see below.
- `config.rs`, `init.rs` — `orangu-gguf.conf` loading and the `--init`
  wizard, following the same shape as `orangu-coordinator`'s. `init.rs`'s
  `models` prompt uses `rustyline` (already a project dependency; see
  `src/tui/helper.rs`) for filesystem TAB completion rather than the plain
  `prompt.rs` helpers below.
- `prompt.rs` — the `prompt`/`prompt_bool` stdin helpers, shared by `init.rs`
  (for everything but the `models` prompt) and `roles.rs` (factored out once
  a second interactive flow needed them).
- `roles.rs` — the interactive role wizard launched by a bare `orangu-gguf`
  invocation (no subcommand): see below.
- `shell.rs` — hand-written bash/zsh/fish completion scripts.

### GGUF parsing (`gguf.rs`)

`GgufFile::read` implements the header, metadata key-value, and tensor-info
sections of the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
directly against a `BufReader`, without ever reading the tensor-data section
itself — a `Reader<R>` wrapper tracks `bytes_read` as it goes, so
`GgufFile::data_offset` (where tensor data would begin, aligned up to
`general.alignment`, default 32) is computed for free without seeking into
it. This is what keeps `list`/`show` fast against multi-gigabyte model
files: parsing a file's full metadata and tensor-info table costs only a
few KB of reads regardless of the file's total size.

Only little-endian GGUF is read — the spec itself notes there is currently
no reliable way to detect a big-endian file, and none exist in practice.
GGUFv1 (32-bit tensor/metadata counts, long deprecated upstream) is
rejected with a clear error rather than silently misread.

Two circuit breakers (`MAX_STRING_BYTES` = 100 MiB, `MAX_ARRAY_ELEMENTS` =
200M) guard string and array length prefixes: a corrupt or truncated
download could otherwise claim an enormous length and force a huge
allocation attempt before a single byte of it is verified to exist in the
file.

`GgufValue::display(preview_limit)` renders a value for `show`; arrays
longer than `preview_limit` print a truncated preview (`... (N more)`)
rather than every element, since metadata arrays like
`tokenizer.ggml.tokens` routinely hold well over 100,000 entries — `--full`
passes `usize::MAX` to disable this.

`ggml_type_name` maps the `ggml_type` enum (ids 0–41, per
[`ggml.h`](https://github.com/ggml-org/ggml/blob/master/include/ggml.h)) to
its canonical name; ids the format has since retired (e.g. `Q4_0_4_4`,
whose numeric slot is never reused) print as `reserved(N)`, and anything
beyond the table (a type added after this was written) as `unknown(N)`.

### Quantization: element counts, not tensor counts (`type_element_totals`)

`GgufFile::type_element_totals` sums each tensor's element count
(`dims.iter().product()`) by `ggml_type`, rather than counting tensors. A
model has far more small `F32` bias/norm tensors than large weight
matrices, but those matrices hold nearly all the parameters — a
per-tensor-count majority would misreport a heavily quantized model as
`F32`. This is a coarser signal than the true filename-derived quant tag
(next section): it can't distinguish `Q4_K_S` from `Q4_K_M`, since both use
the `Q4_K` ggml type for most tensors, differing only in which few tensors
(e.g. the output projection) get upgraded to a higher-precision type.

### Shard grouping and the Hugging Face repo id (`models.rs`)

`scan_models_dir` walks the configured directory with
`walkdir::WalkDir::new(dir).follow_links(true)`. This is not optional:
Hugging Face's own hub cache — the layout llama.cpp's `-hf`/`--hf-repo`
itself downloads into — names every file under `snapshots/<rev>/` as a
symlink into `blobs/`. Without `follow_links`, `entry.file_type().is_file()`
reports the symlink itself (never `true`), and every such model is silently
skipped rather than listed.

Two further filters run in `scan_models_dir` itself, before any shard
grouping, so only unique models are ever counted or listed:

- **Duplicate-file collapsing.** All matching paths are collected and
  sorted first, then each is resolved with `std::fs::canonicalize` (which
  follows symlinks to their real target) into a `seen_targets: HashSet`;
  a path whose canonical target was already seen is skipped. This matters
  because the Hugging Face hub cache can reference the exact same blob from
  more than one `snapshots/<rev>/` directory — when a repo's ref moves but a
  file's content doesn't change, the cache creates a new snapshot folder
  that symlinks to the already-downloaded blob rather than re-fetching it,
  so without this step a single physical download could count twice.
- **Multimodal projector ("mmproj") exclusion.** After a file parses
  successfully, `GgufFile::is_clip_projector` is checked
  (`general.architecture == "clip"`, identified the same way llama.cpp's
  own `clip.cpp` loader does) and, if true, the file is skipped entirely —
  it's excluded before it ever reaches `ModelSummary`/`group_models`. An
  mmproj sidecar accompanies a base model rather than standing in as one
  (llama.cpp loads it via `--mmproj`, separately from the base checkpoint),
  so it shouldn't inflate the count of "models" a directory holds. This
  exclusion only affects `list`'s counting/grouping — `resolve_model_path`'s
  direct-path and bare-filename lookups (the first thing `show` tries) are
  untouched, so an mmproj file can still be `show`n by its path (the
  bare-filename branch, `models_dir.join(requested)`, only resolves a file
  sitting directly in the `models` root, not one nested under a cache's
  `snapshots/<rev>/`).

`group_models` collapses a multi-part model's shard files
(`name-00001-of-00004.gguf`, ...) into one `ModelGroup`, keyed by (parent
directory, shard-suffix-stripped file stem) — so two files that merely
share a name in different directories (e.g. two Hugging Face snapshot
revisions of the same release) stay separate rows, while genuine shards of
one model merge, with `size_bytes` summed and `type_totals` combined across
every shard before picking one dominant type (a single shard's own tensors
are only part of the whole model).

`shard_group_label` and `hf_tag_from_label` deliberately mirror llama.cpp's
own resolver in `common/download.cpp` byte-for-byte, rather than
reinventing the convention:

- The shard suffix regex, `-\d{5}-of-\d{5}$`, matches
  `get_gguf_split_info`'s `re_split`.
- The quant-tag regex, `[-.]([A-Z0-9_]+)$` in llama.cpp's `re_tag`, is
  reimplemented as `hf_tag_from_label`: the trailing run of
  alphanumeric/underscore characters after the *last* `-` or `.` in the
  (shard-stripped) name, uppercased. This is why `MODEL`'s `:quant` suffix
  can say `Q4_K_M` where the coarser `QUANT` column can only say `Q4_K` —
  the tag comes from the filename llama.cpp itself would match against, not
  from the tensor types.

`hf_repo_id_from_path` recovers `<user>/<model>` by walking a file's
ancestor directories for one matching `models--<user>--<model>` (checking
every ancestor, not just the immediate parent, since real files sit under
`snapshots/<rev>/`, sometimes with a further per-quant subfolder). This
directory-naming convention — `folder_name = "models--" + repo_id.replace("/",
"--")` — is Hugging Face's own, confirmed directly against llama.cpp's
README ("models downloaded with `-hf` are now stored in the standard
Hugging Face cache directory"). A file outside that layout has no
`repo_id` to recover, so `group_models` falls back to the bare
shard-stripped label.

`resolve_show_target` resolves whatever `show` was given, checking the
fast, scan-free path first: `resolve_model_path` (a direct/relative/
absolute path, or a bare name under `models`) is tried before falling back
to a full `scan_models_dir` + `group_models` for an `NR` or `MODEL` lookup —
so the common case of `show /path/to/file.gguf` never pays the cost of
scanning the whole directory. `ModelGroup::representative_path` (the first
shard by sorted path order, which is also the one carrying full GGUF
metadata under the standard shard-naming convention) is what `show` actually
opens for a multi-shard model.

### Downloading from Hugging Face (`download.rs`)

`download::run_download` implements `orangu-gguf download <user>/<model>[:quant]`
by directly mirroring llama.cpp's own `common/download.cpp` and
`common/hf-cache.cpp` — read from that source rather than reimplemented
from a guess at the Hugging Face API, since the whole point is producing a
cache llama.cpp itself recognizes as already downloaded.

**Resolving the commit.** `resolve_commit` calls
`GET /api/models/<repo>/refs`, which returns `{"branches": [{"name", "targetCommit"}, ...]}`;
the branch named `main` wins, falling back to the first one listed. A repo
that doesn't exist can return `401` rather than `404` when unauthenticated
(Hugging Face doesn't distinguish "doesn't exist" from "exists but is
private" for a caller without access) — `resolve_commit` reports this as
"repository not found ... if it's private or gated, set HF_TOKEN" when no
token was supplied, or "authentication failed ... check HF_TOKEN" when one
was (a `401` with a token in hand means the token itself was rejected, not
that the repo is missing).

**Listing files.** `list_repo_files` calls
`GET /api/models/<repo>/tree/<commit>?recursive=true`, returning every file
with its `path`, and either a top-level `oid` (the git blob sha1, for small
files) or an `lfs.oid` (the LFS object's sha256, for anything large enough
to be stored as LFS — every real GGUF file). `RepoFile::oid` takes whichever
is present; it doubles as the blob's filename in the cache, so two
snapshots referencing byte-identical content share one on-disk copy exactly
like the real Hugging Face cache does.

**Choosing what to download.** `select_files_to_download` mirrors
`find_best_model` + `get_split_files`:

- `is_model_gguf` excludes `mmproj`/`imatrix`/`mtp-` files from counting as
  "the model" — the same exclusion `gguf_filename_is_model` applies
  upstream, and the same one `models::scan_models_dir` applies when
  *reading* a cache back (see the shard-grouping section above).
- With an explicit `:quant`, `find_by_tag` looks for it as a substring
  immediately followed by `.` or `-` anywhere in a candidate's path (so
  `"Q4_K_M"` matches both `model-Q4_K_M.gguf` and
  `model-Q4_K_M-00001-of-00004.gguf`) — the same non-anchored rule
  llama.cpp's own resolver uses, deliberately different from
  `models::hf_tag_from_label`'s anchored *extraction* of an unknown tag
  from a filename, since here the tag is already known and being searched
  for. A file only matches as a **primary** if it's shard 1 (or unsharded);
  a later shard never stands in for the whole model on its own.
- Without a `:quant`, `DEFAULT_TAG_PREFERENCE` (`["Q4_K_M", "Q8_0"]`, in
  that order — llama.cpp's own default) is tried before falling back to
  the first model file found at all.
- Once a primary file is chosen, `shard_info` (the same
  `-NNNNN-of-NNNNN` suffix regex `models::shard_group_label` strips,
  here also extracting the index and total) finds every sibling sharing
  its prefix and total count, so a multi-part model downloads whole.

**Choosing a multimodal projector, if any.** After the primary model file is
picked, `find_best_mmproj` (calling the generic `find_best_sibling` with
`keyword = "mmproj"`) directly mirrors llama.cpp's own `find_best_sibling`/
`find_best_mmproj`: among every `.gguf` path containing `mmproj`, it prefers
the one sharing the deepest directory prefix with the primary file's own
path (rejecting any candidate whose directory list isn't a prefix of the
model's), then — among ties at that depth — the one whose quantization bit
depth (`extract_quant_bits`, reading the first run of digits in the
filename's trailing tag, e.g. `Q4_K_M` -> `4`, `BF16`/`F16` -> `16`, `F32`
-> `32`) is numerically closest to the primary file's own. This is the same
file llama-server's own `-hf` auto-fetches the first time a vision-capable
model is launched with an image-related flag (verified against a real
repo, `unsloth/Qwen3.6-35B-A3B-GGUF`, which offers three top-level mmproj
variants — `BF16`/`F16`/`F32` — alongside a `Q4_K_M` primary; both this
code and a live `llama-server -hf ...:Q4_K_M --image-min-tokens 1024` run
independently picked `mmproj-BF16.gguf`), so fetching it up front here means
`LLAMA_CACHE=<models>` already has it ready offline. If found, it's appended
to the file list `run_download` fetches, alongside whatever shards the
primary model itself has.

**Fetching bytes.** `download_with_resume` streams the response body to a
`<blob>.part` file, resuming from wherever that file left off via an HTTP
`Range` request if one already exists from an interrupted attempt (falling
back to a full restart if the server doesn't honor it, signaled by a `200`
instead of the expected `206`). Progress is a plain percentage against the
tree API's own reported file size — not the response's `Content-Length`,
which would only cover the *remaining* bytes on a resumed request — plus a
trailing `[index/total]` (`run_download` enumerates `selected`, 1-based, to
pass this position through), so a multi-file download (a sharded model, or
the mmproj sidecar above) reads as progress through the whole batch rather
than restarting a bare percentage at 0% with no indication of how many
files remain. A blob already present on disk with a matching size is
skipped entirely rather than re-verified byte-for-byte (cheap and good
enough; matches the practicality bar the rest of this tool holds to
elsewhere, e.g. the element-count quantization guess) — its skip message
gets the same `[index/total]` suffix for consistency.

**Placing the file.** `link_or_copy` computes the same relative symlink
target the real Hugging Face cache uses (`../` once per path component
between `snapshots/<commit>/` and the file, plus two more to reach the
repo root, then into `blobs/<oid>`) rather than an absolute path, so the
whole `models` directory stays portable if moved. Falls back to a plain
copy if symlinks aren't available at all (e.g. Windows without developer
mode enabled) — mirroring `hf_cache::finalize_file`'s own degraded-mode
fallback.

**Not implemented**, out of scope for a first version: `--mtp` companion
downloads (also a `find_best_sibling` call upstream, with
`keyword = "mtp-"`), `preset.ini`-based repos (a repo-root manifest naming
one specific file to fetch regardless of tag matching), and Docker registry
sources.

### CPU/GPU detection (`system.rs`)

CPU statistics (brand, vendor, architecture, physical/logical core counts,
peak frequency, total/available RAM) come from
[`sysinfo`](https://docs.rs/sysinfo), used with only its `system` feature
(no `disk`/`network`/`component`/`user`) to keep the dependency footprint
minimal.

GPU detection has no single cross-platform API, so `detect_gpus` layers
several best-effort, independent sources and concatenates whatever each
finds — a card no source recognizes simply doesn't appear, rather than the
whole command failing:

1. **NVIDIA** (`detect_nvidia_gpus`, Linux and Windows): shells out to
   `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`, the one
   interface guaranteed to exist wherever an NVIDIA driver is installed. A
   missing binary or non-zero exit returns an empty list, not an error —
   "no NVIDIA GPU" is the expected common case. `memory_kind` is always
   `MemoryKind::Dedicated` — no consumer NVIDIA GPU is anything else.
2. **AMD/Intel/other, Linux only** (`detect_linux_sysfs_gpus`): enumerates
   `/sys/class/drm/card*/device`, the kernel interface every Linux GPU
   driver exposes. NVIDIA vendor ids (`0x10de`) are skipped here — already
   reported by `nvidia-smi` above, and `mem_info_vram_total` is an
   amdgpu-specific sysfs attribute this path can't get for NVIDIA anyway.
   VRAM total/used come from `mem_info_vram_total`/`mem_info_vram_used`
   when present (AMD only; Intel iGPUs report no separate VRAM, being
   shared system memory). The device's marketing name is looked up in the
   system's `pci.ids` database (`load_pci_ids`, checking
   `/usr/share/hwdata/pci.ids` first — the `hwdata` package's path on
   Fedora/RHEL — then the `pciutils` paths used elsewhere), the same file
   `lspci` itself reads; if it isn't installed, the raw `vendor:device` PCI
   ids are shown instead of a name, rather than failing.
3. **macOS** (`detect_macos_gpus`): `system_profiler SPDisplaysDataType
   -json`, parsed with `serde_json` (already a workspace dependency).
4. **Windows** (`detect_windows_gpus`): PowerShell's `Win32_VideoController`
   WMI class via `Get-CimInstance | ConvertTo-Json`. A single result comes
   back as a bare JSON object rather than a one-element array, which the
   parser normalizes explicitly. `AdapterRAM` is a well-known 32-bit field
   that misreports (often as 0 or wrapped) for cards with more than ~4 GiB
   of VRAM; it's still the best zero-dependency source available on
   Windows, so a `0` reading is treated as "unknown" rather than shown
   literally.

### Dedicated vs. shared memory (`MemoryKind`)

Every `GpuInfo` carries a `memory_kind: MemoryKind` (`Dedicated` / `Shared` /
`Unknown`), derived by a different signal per platform — there is no single
cross-platform API for this either:

- **Linux** (`linux_memory_kind`): whether `amdgpu` exposes
  `mem_info_vram_vendor` (the VRAM chip manufacturer, e.g.
  `samsung`/`hynix`) for the device. This was verified directly against
  real hardware carrying both a discrete card and an integrated APU on the
  same machine (a Ryzen laptop's Navi 14 dGPU and Renoir iGPU): the
  discrete card has this file, the integrated one — which still reports a
  `mem_info_vram_total` for its BIOS-reserved carve-out of system RAM —
  does not, since there's no separate memory chip to name. A device with no
  `mem_info_vram_*` attributes at all (Intel's `i915` driver, almost always
  integrated) also defaults to `Shared`; a rare discrete Intel Arc card
  would be misclassified here, since its local-memory sysfs interface
  isn't read.
- **macOS** (`macos_memory_kind`): `system_profiler`'s own two keys already
  say which kind of memory this is — `spdisplays_vram` names a real
  dedicated-VRAM figure, while `spdisplays_vram_shared` marks Apple
  Silicon's unified-memory architecture or an older integrated Mac.
- **Windows** (`windows_memory_kind`): `Win32_VideoController` has no
  dedicated/shared field of its own (that lives in DXGI's
  `DXGI_ADAPTER_DESC`, unreachable from a WMI/PowerShell query without a
  real helper binary), so this guesses from the adapter name string
  instead: NVIDIA is always `Dedicated`, Intel is `Shared` unless the name
  says `Arc`, and AMD is left `Unknown` outright — its driver names an
  APU's integrated GPU and a discrete Radeon card too similarly (e.g. plain
  "AMD Radeon(TM) Graphics" for either) to guess reliably from the name
  alone.

`MemoryKind::Unknown` is only ever constructed on macOS/Windows, whose
detection functions are `cfg`'d out on other build targets — hence the
variant carries a blanket `#[allow(dead_code)]` rather than one scoped per
target.

### Shared memory's total is system RAM, not the raw query result

`detect_gpus(total_memory_bytes)` takes the system's total RAM —
`CpuInfo::total_memory_bytes`, computed once by the caller so this doesn't
pay for a second `sysinfo` query — and, after concatenating every
platform's GPUs, runs `apply_shared_memory_total` over the result: any
`GpuInfo` with `memory_kind == MemoryKind::Shared` has its
`vram_total_bytes` overwritten with `total_memory_bytes`, unconditionally.

This matters because a shared GPU's own reported figure (where one exists
at all) drastically understates what it can actually use: `amdgpu` reports
an APU's tiny BIOS-reserved carve-out via `mem_info_vram_total` (as little
as a few hundred MiB — 512 MiB on the Renoir APU this was verified
against), and Intel/Windows sources often report nothing at all. System RAM
is the real ceiling on how much such a GPU can draw on, so it's the only
figure worth showing as its total; `vram_used_bytes` is left untouched
(whatever the platform reported, or `None`), since "how much of the shared
pool is currently claimed as graphics memory" is a real and distinct
figure from the override, unlike the total.

### Hardware-based model-size suggestion (`suggest.rs`)

`main.rs`'s `Commands::Suggest` arm calls the same `system::detect_cpu`/
`detect_gpus` pair `Commands::System` does, then passes the result to
`suggest::format_suggestion`, which appends two size-suggestion tables after
`system::format_report`'s own CPU/GPU listing (via the shared
`push_suggestion_block` helper). There is no separate detection path —
`suggest` is purely a second interpretation of the same hardware inventory
`system` already knows how to gather.

**The memory-estimation formula.** `estimate_total_vram_bytes` mirrors [Sam
McLeod's GGUF VRAM Estimator](https://smcleod.net/vram-estimator/)'s own
`calculateMemoryBreakdown` function (read directly from its published
`vram-calculator.min.js`, not guessed) and the general shape of
[erans/selfhostllm](https://github.com/erans/selfhostllm)'s calculator:

- Model weight bytes: `params × bits_per_weight ÷ 8`, plus a fixed 500 MiB
  runtime/CUDA-context overhead (`RUNTIME_OVERHEAD_BYTES`, matching
  smcleod's own `CUDA_SIZE` constant exactly).
- KV cache bytes: `context_size × 2 (K and V) × layers × hidden_size ×
  (kv_cache_bits ÷ 8)`, plus a smaller "compute buffer" term for attention
  scratch space, `context_size × hidden_size × 3 × (bits_per_weight ÷ 8)`.

Since `suggest` runs before any model is chosen, there's no real GGUF file
to read `hidden_size`/`layers` from (unlike `roles.rs`'s
`architecture_metadata_u64`, which reads them from an actual model).
`estimate_hidden_dims` instead estimates both from the parameter count
alone, via the standard transformer parameter-count approximation params ≈
12 × layers × hidden_size² (solved for `hidden_size = sqrt(params / 12)`,
then `layers` from that) — the exact same fallback smcleod's own calculator
uses when it has no real GGUF metadata to read either.

`DEFAULT_BITS_PER_WEIGHT` (4.83, Q4_K_M) and `KV_CACHE_BITS` (8, Q8_0) match
this project's own established defaults (`download.rs`'s
`DEFAULT_TAG_PREFERENCE`, and the role wizard's `-ctk q8_0 -ctv q8_0`)
rather than assuming full FP16 throughout.

**A table, not a single guess.** Actual context usage varies far too much to
guess well from hardware alone, and bits-per-weight depends on which
quantization tag you end up downloading — so instead of picking one of each,
`push_suggestion_block` prints a row per context length in `CONTEXT_LADDER`
(1K up to the role wizard's own largest default, 262144) and a column per
quantization in `QUANT_LADDER` (`Q2_K` at 3.00 bits/weight, `Q4_K_M` at
`DEFAULT_BITS_PER_WEIGHT`, and `Q8_0` at 8.5 — all three bits-per-weight
figures read from smcleod's own table, the same source as the formula
itself). Each cell is independently computed by `suggest_param_count`, so
the suggested size correctly shrinks along a row as quantization gets
heavier, and down a column as context grows.

**Picking a size.** `suggest_param_count` walks `PARAM_LADDER_BILLIONS` — a
curated list of common open-weight parameter counts, largest first — and
returns the first whose `estimate_total_vram_bytes` result (at that cell's
context length and bits-per-weight) fits within the budget, or `None` if
even the smallest rung (1B) doesn't (rendered as `-`).

**Two budgets, two tables.** `format_suggestion` computes two separate
budgets and prints a labeled `push_suggestion_block` for each, `"Suggested
model size (Dedicated)"` and `"Suggested model size (Shared)"`:

- `dedicated_vram_budget_bytes` sums every `Dedicated` GPU's
  `vram_total_bytes` alone (multiple dedicated cards add up, matching the
  role wizard's `-sm layer` multi-GPU tensor split) — `0` when there's no
  dedicated GPU at all, which `suggest_param_count` then correctly reports
  as nothing on the ladder fitting.
- `combined_gpu_budget_bytes` sums every GPU's `vram_total_bytes`,
  `Dedicated` and `Shared` alike (a `Shared` GPU's is already the system RAM
  total via `apply_shared_memory_total`, described above) — the more
  permissive figure, representing every device `--fit on` could spread
  layers across at once. Falls back to the CPU's own `total_memory_bytes`
  when that sum is `0` (no GPU detected at all, or only `Unknown`-kind
  ones).

`Unknown`-kind GPUs are never counted in either budget — there's no safe
way to know whether their reported figure is a hard VRAM ceiling or already
system RAM, and double-counting or miscounting either way would make both
suggestions worse than not counting them at all.

### The role wizard (`roles.rs`)

`main.rs` launches `roles::run_wizard` whenever `orangu-gguf` is invoked with
no subcommand (checked after `--init`/`-s`, both of which still take
priority): it scans the models directory once (`models::scan_models_dir` +
`group_models`), prompts for a role, then a model, then prints a tuned
`llama-server` command line. Model resolution deliberately doesn't reuse
`models::resolve_show_target` — that function re-scans the directory from
scratch, and the wizard already has the one scan's `groups` in hand; its own
`find_group` matches an `NR` or `MODEL` label against that in-memory slice
instead, avoiding a second, redundant full scan/parse pass.

`ROLE_PROFILES` is a fixed table of five `RoleProfile`s (`name`,
`default_ctx_size`) — `resolve_role` matches a 1-based index into it or a
case-insensitive name.

`build_extra_args` is deliberately **not** a hardware heuristic: it's a
`match` on the role name returning that role's extra arguments verbatim
from `doc/manual/en/73-openai.md`'s own per-role example (the OpenAI
platform chapter — the project's canonical reference for running
llama.cpp well *with orangu specifically*), substituting `ctx_size` and
`threads` for every role and, for `embeddings` only, `pooling` (see
below). Each of the five arms is locked in by its own test
(`build_extra_args_matches_the_manuals_*_role_example`) asserting the exact
joined string against that chapter's example — if the chapter's examples
ever change, these tests should fail and prompt updating both together.
This replaced an earlier version that instead estimated `-ngl` from
detected VRAM against the model's own tensor layer count; that approach
was dropped once every role in the manual's own reference turned out to
rely on `--fit on` (which fits unset parameters — including GPU layers —
to actual device memory automatically) rather than a separately guessed
`-ngl`, making the hand-derived estimate both redundant and a needless
source of drift from the project's own documented, verified configuration.

`build_command` assembles the full line: a leading `LLAMA_CACHE=<models_dir>`
(the `models_dir` parameter is the loaded configuration's `models` path),
`llama-server`, the model
reference (`model_reference`: `-hf <label>` when `label` contains a `/` —
a Hugging Face hub cache label always does, and no bare GGUF filename ever
does; see the shard-grouping section above — else `-m <path>` for a plain
local file), a bare `--port 8100` (every role example in that chapter uses
it), then `build_extra_args`'s output. The `LLAMA_CACHE` prefix is
llama.cpp's own highest-priority override for where `-hf` reads/writes its
Hugging Face hub cache; pointing it at the same directory `download.rs`
populates and `models::scan_models_dir` reads is what makes the printed
command find a model already fetched by `orangu-gguf download`, instead of
falling back to llama.cpp's own default `~/.cache/huggingface/hub`.
`ctx_size` is the role's
`default_ctx_size` clamped to the model's own `<arch>.context_length`
metadata when that's smaller (`architecture_metadata_u64`, which reads
`general.architecture` and looks up `<that>.<suffix>` — the same
per-architecture-namespaced hyperparameter convention the GGUF spec uses
for `block_count` too); `threads` is the detected physical core count
(`cpu.physical_cores.unwrap_or(cpu.logical_cores)`).

For the `embeddings` role, `pooling` is read from the model's own
`<arch>.pooling_type` metadata and mapped through `pooling_type_name`
(llama.cpp's `enum llama_pooling_type` from `include/llama.h`: `0`=`none`,
`1`=`mean`, `2`=`cls`, `3`=`last`, `4`=`rank`), falling back to `"mean"`
with a note only when a model has no usable value there. This *replaced* an
earlier version that hard-coded `--pooling last` (this chapter's own
`73-openai.md` example previously did too) and only used the model's
metadata as a cross-check note when it disagreed — which it always did for
the one real GGUF this was tested against: `embeddinggemma-300M` reports
`pooling_type=1` (`mean`) in its own metadata, not `last`. Once confirmed
that `mean` is the one that's actually correct for that model, both the
code and `73-openai.md`'s example were corrected together, and pooling is
now genuinely read from metadata rather than asserted — a hard-coded
default can be wrong even for the one model it was written for, which is
exactly what happened here.

**Detecting `--reasoning-preserve` support.** `supports_reasoning_preserve`
appends `--reasoning-preserve` to the command when the model's own
`tokenizer.chat_template` string metadata looks like it supports
llama-server's `--reasoning-preserve` flag (which keeps a model's prior
reasoning trace in context across turns instead of dropping it).
llama.cpp's own detection (`jinja::caps_get` in `common/jinja/caps.cpp`,
surfaced as a `SRV_INF` log line in `tools/server/server-context.cpp`)
actually *executes* the chat template against a synthetic conversation
carrying a reasoning trace and checks whether that trace survives the
rendered output — reproducing that exactly would mean embedding a
Jinja-compatible template engine (llama.cpp's own
`common/jinja/{lexer,parser,runtime}.cpp`), well out of proportion for a
tool that otherwise only ever reads GGUF metadata directly. That probe's
result only ever depends on whether the template references one of three
Jinja variables it sets beforehand (`caps_apply_preserve_reasoning`):
`preserve_thinking`, `clear_thinking`, `truncate_history_thinking`. A
template referencing none of them categorically can't honor the flag
(nothing in it branches on that state), and one that does is a strong,
though not certain, signal that it does — `supports_reasoning_preserve`
checks the raw template text for these three substrings rather than
executing it.

`build_command` skips adding it when the role's own `build_extra_args`
output already contains `--reasoning off` — currently only `review`, whose
canonical `73-openai.md` example turns reasoning off entirely, so there'd
be nothing left to preserve and the two flags together would read as
contradictory.

### Shell completions (`shell.rs`)

Mirrors `orangu`'s own `-s`/`--shell-completions` exactly (`src/bin/orangu/
shell.rs`, `print_shell_completions` in `main.rs`): hand-written bash/zsh/
fish scripts embedded as `&str` constants, selected by inspecting `$SHELL`,
rather than clap-generated completions. `show`'s NR/MODEL argument
completion works the same way `orangu`'s own scripts complete session
UUIDs — the shell function shells back out to `orangu-gguf list` itself
(`2>/dev/null`, so a missing config yields no candidates rather than an
error) and reads its first two columns with `awk`. This keeps the
completion logic entirely in the shell script, depending on nothing but
`orangu-gguf` itself being on `$PATH` — no dynamic-completion protocol or
extra binary flag is needed.

An earlier version of this explored `clap_complete`'s `unstable-dynamic`
feature for this instead; it was backed out in favor of the approach above
once `orangu`'s own precedent was found, since introducing a genuinely
unstable (semver-exempt) dependency wasn't warranted when a small,
self-contained shell script does the same job with zero new dependencies.

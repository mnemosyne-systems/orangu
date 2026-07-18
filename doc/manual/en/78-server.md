\newpage

## Inference server internals

`orangu-server` (`src/bin/orangu-server/`) is a third binary in the same
Cargo package as `orangu` and `orangu-coordinator`. Besides serving a GGUF
model, it's also the machine's GGUF inventory tool (`system`/`suggest`/
`list`/`show`/`download`/`delete`) — stateless between runs for those six:
every invocation re-detects hardware and re-scans the models directory from
scratch, so there is no cache, config-reload, or background process to
reason about for them. `system`/`suggest`/`show`/`delete` stay entirely
offline; `download` always talks to the Hub, and `list` does too, before
printing its table, to check each Hugging Face-backed model for a newer
commit (see `latest_commits` below) — swallowing the lookup silently rather
than failing when the Hub can't be reached. It does real tensor computation
itself for serving — GGUF loading, dequantization, the transformer forward
pass, sampling, and request scheduling are implemented in Rust with no
dependency on llama.cpp/ggml's own compiled code.

### Module layout

- `main.rs` — CLI parsing (serving plus the `system`/`suggest`/`list`/
  `show`/`download`/`delete` subcommands), model-spec resolution, GPU
  backend selection (`select_backend`), `format_show`/
  `DEFAULT_ARRAY_PREVIEW` (for `show`), `select_model_for_deletion`/
  `confirm` (for `delete`), and process wiring (Ctrl+C/`SIGINT`/
  `--daemon`).
- `config.rs`, `init.rs` — `orangu-server.conf` loading and the `--init`
  wizard.
- `suggest.rs` — `suggest`: a hardware-based model-size estimate built on
  top of `orangu::hardware`'s own detection; see below.
- `shell.rs` — hand-written bash/zsh/fish completion scripts.
- `engine/loader.rs` — memory-maps a GGUF file, reads `<arch>.*`
  hyperparameters, resolves tensor byte ranges.
- `engine/quant.rs` — dequantization for every supported `ggml_type`.
- `engine/tensor.rs` — the handful of numeric ops (matmul, RMSNorm,
  softmax, RoPE, SwiGLU/GEGLU) a forward pass needs, on plain `f32`
  slices — not a general ND-array library.
- `engine/arch/{mod,llama,gemma,qwen35moe}.rs` — one `ModelForward`
  implementor per architecture family.
- `engine/backend/{mod,cpu,vulkan,vulkan_shaders,cuda,opencl,rocm}.rs` —
  the `Backend` trait and its five implementors; see below.
- `engine/tokenizer.rs` — a from-scratch BPE tokenizer.
- `engine/chat_template.rs` — renders `tokenizer.chat_template` via
  `minijinja`.
- `engine/sampling.rs` — repetition penalty, temperature/top-k/top-p/min-p.
- `engine/kv_cache.rs` — per-sequence KV cache buffers.
- `engine/scheduler.rs`, `engine/generate.rs`, `engine/batch.rs` — the
  multi-slot request scheduler and continuous-batching machinery.
- `http/{mod,openai,native}.rs` — the HTTP surface.
- `web/{mod,render,sessions}.rs` — the built-in chat UI.

The GGUF-inventory subcommands lean on library modules shared with the rest
of the workspace rather than binary-local ones: `orangu::gguf` (the GGUF
binary-format reader), `orangu::model_spec` (directory scan, shard
grouping, and the Hugging Face repo-id/quant-tag reconstruction behind
`list`'s `MODEL` column), `orangu::model_download` (`download`'s fetch
logic), and `orangu::hardware` (CPU/GPU detection). Living in `src/`
alongside `orangu`'s and `orangu-coordinator`'s own shared code, rather
than nested under `src/bin/orangu-server/`, is what let `orangu-server`
absorb these subcommands from the now-removed `orangu-gguf` binary without
duplicating any of this logic — `orangu-server`'s `main.rs` was already
calling straight into `orangu::model_spec::resolve_or_fetch_model` for its
own positional `model` argument, so `list`/`show`/`download` calling the
same modules directly was additive, not a rewrite.

### GGUF parsing (`orangu::gguf`)

`GgufFile::read` implements the header, metadata key-value, and tensor-info
sections of the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
directly against a `BufReader`, without ever reading the tensor-data section
itself — a `Reader<R>` wrapper tracks `bytes_read` as it goes, so
`GgufFile::data_offset` (where tensor data would begin, aligned up to
`general.alignment`, default 32) is computed for free without seeking into
it. This is what keeps `list`/`show` fast against multi-gigabyte model
files: parsing a file's full metadata and tensor-info table costs only a
few KB of reads regardless of the file's total size. `engine::loader`
(above) is a separate, `mmap`-based reader over the same format, built for
loading tensor *data* rather than just metadata.

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

### Shard grouping and the Hugging Face repo id (`orangu::model_spec`)

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
opens for a multi-shard model. `resolve_or_fetch_model` builds on top of
`resolve_show_target` for the serving path's own positional `model`
argument: try resolving locally first, and only reach for
`orangu::model_download::download_model` when nothing local matched — the
same fallback `main.rs`'s `prepare` and `select_model_interactively` share.

### Deleting a model (`orangu::model_spec`)

`resolve_delete_target` resolves `delete`'s argument to a full
`ModelGroup`, not just `resolve_show_target`'s single representative
path — `delete_model` needs every shard to remove a multi-shard model
atomically, so this always scans and groups first rather than reusing
`resolve_show_target`'s scan-free fast path for a plain file argument
(that fast path only ever returns one file, with no way to tell whether it
belongs to a larger group). Resolution order otherwise matches
`resolve_show_target`: a direct/bare path first — returning that file's
whole group when `group_models` placed it in one, or a synthetic
one-`ModelGroup`-of-one-path when it didn't (an mmproj sidecar, which
`group_models` deliberately excludes from every real group but `delete`
should still be able to name directly) — then an `NR`, then a `MODEL`
label.

`delete_model` removes every path in the resolved group, and, for each one
that turns out to be a Hugging Face hub-cache symlink
(`models--<user>--<model>/snapshots/<rev>/<file>`, resolved with
`std::fs::canonicalize` *before* the symlink itself is unlinked), also
removes its target blob under that same repo's `blobs/` — but only when
`blob_still_referenced` finds no other symlink left under
`<repo>/snapshots/` still pointing at it. This matters for the same reason
`scan_models_dir`'s own duplicate-file collapsing does: a repo's ref can
move without a file's content changing, so the cache reuses (symlinks to)
an already-downloaded blob from a second snapshot revision rather than
re-fetching it — and `scan_models_dir` only ever lists the first,
sorted-earliest occurrence of that shared content, so the second
snapshot's symlink is never part of any group `delete` was asked to
remove. Scoping the reference check to just `<repo>/snapshots/` (not the
whole `models` directory) is both cheap and correct: blobs are already
nested per-repo (`models--<user>--<model>/blobs/`), so cross-repo sharing
can't happen by construction — no walk of the full, potentially huge
`models` directory is ever needed just to delete one model.

`remove_empty_ancestors` walks up from a path's parent directory, removing
it (and its own parent, and so on) as long as it's empty, stopping the
moment one isn't or at `models_dir` itself (which is never removed,
whatever's left inside it). `delete_model` calls it twice per shard — once
from the removed symlink's own `snapshots/<rev>/` chain, and, when a blob
was also reclaimed, once more from that blob's sibling `blobs/` chain,
since the two aren't nested inside each other and either could be the one
left holding the repo directory open. Together, deleting a repo's last
shard collapses the now-empty `snapshots/<rev>/` and `blobs/`, and, once
both are gone, `models--<user>--<model>/` itself, rather than leaving a
hollowed-out shell of empty directories behind.

`main.rs`'s `Command::Delete` arm always confirms before calling
`delete_model` (`confirm`, a plain stdin Yes/No reader defaulting to *No*
on an empty entry or closed stdin — the same fail-safe default a
destructive filesystem action should have) unless `--yes` was passed, and
resolves an omitted argument through `select_model_for_deletion`: the same
`format_list` table `list` prints, followed by an `NR` prompt — the
delete-time counterpart of `main.rs`'s own `select_model_interactively`
(used to pick a model to *serve*), returning a full `ModelGroup` rather
than just a path/label pair since that's what `delete_model` needs.

### Downloading from Hugging Face (`orangu::model_download`)

`download_model` implements `orangu-server download <user>/<model>[:quant]`
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
  upstream, and the same one `orangu::model_spec::scan_models_dir` applies
  when *reading* a cache back (see the shard-grouping section above).
- With an explicit `:quant`, `find_by_tag` looks for it as a substring
  immediately followed by `.` or `-` anywhere in a candidate's path (so
  `"Q4_K_M"` matches both `model-Q4_K_M.gguf` and
  `model-Q4_K_M-00001-of-00004.gguf`) — the same non-anchored rule
  llama.cpp's own resolver uses, deliberately different from
  `orangu::model_spec::hf_tag_from_label`'s anchored *extraction* of an
  unknown tag from a filename, since here the tag is already known and
  being searched for. A file only matches as a **primary** if it's shard 1
  (or unsharded); a later shard never stands in for the whole model on its
  own.
- Without a `:quant`, `DEFAULT_TAG_PREFERENCE` (`["Q4_K_M", "Q8_0"]`, in
  that order — llama.cpp's own default) is tried before falling back to
  the first model file found at all.
- Once a primary file is chosen, `shard_info` (the same
  `-NNNNN-of-NNNNN` suffix regex `orangu::model_spec::shard_group_label`
  strips, here also extracting the index and total) finds every sibling
  sharing its prefix and total count, so a multi-part model downloads
  whole.

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
to the file list `download_model` fetches, alongside whatever shards the
primary model itself has.

**Fetching bytes, concurrently.** `download_model` first walks `selected`
sequentially just to decide what needs fetching at all — a blob already
present on disk with a matching size is skipped entirely rather than
re-verified byte-for-byte (cheap and good enough; matches the practicality
bar the rest of this tool holds to elsewhere, e.g. the element-count
quantization guess), printed immediately with an `[index/total]` suffix.
Everything left becomes a `DownloadTask` (label, URL, blob path, size, and
that same `(index, total)` position), and `download_all` hands the whole
batch to rayon's `par_iter().try_for_each` — bounded by rayon's global
thread pool rather than one OS thread per file, so a model with dozens of
shards doesn't open dozens of simultaneous connections. This means a sharded
model's shards, and a bundled mmproj sidecar, download at the same time
instead of one at a time; `download_model` only does the symlink-placement
pass (`link_or_copy`, below) after every download has finished.

Each parallel task's own `download_with_resume` streams its response body to
a `<blob>.part` file, resuming from wherever that file left off via an HTTP
`Range` request if one already exists from an interrupted attempt (falling
back to a full restart if the server doesn't honor it, signaled by a `200`
instead of the expected `206`). Progress is a plain percentage against the
tree API's own reported file size — not the response's `Content-Length`,
which would only cover the *remaining* bytes on a resumed request. Since
several tasks report progress at once, each writes into its own line of a
`ProgressBoard` shared behind a single `Mutex` (one mutex around the whole
board, not one per line, so a "set this line, then redraw every line" update
is atomic and two threads' redraws can't interleave); `ProgressBoard::update`
redraws in place with `\x1b[{n}A` (cursor up `n` lines) followed by
`\x1b[2K` (clear line) per row, so every in-flight file's percentage stays
visible at once until all are done, at which point its line switches from
`Downloading` to a final `Downloaded <label>: 100% [index/total]` — kept at
100% rather than dropped, so every line stays in the same
`<verb> <label>: <percent>% [index/total]` shape whether still in flight or
finished. If a task fails, the others still
run to completion rather than being cancelled (each writes its own `.part`
file, so a later retry only re-fetches whatever actually failed);
`download_all` surfaces the first error once every task has finished.

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

### Checking for updates (`list`'s `(Refresh)` marker)

`list` doesn't just read local disk state — it also asks the Hub whether a
newer commit exists for every model it found under a Hugging Face hub-cache
directory. Two pieces make this work:

- **The local commit.** `orangu::model_spec::hf_local_commit_from_path`
  recovers the sha a `ModelGroup` is cached at by walking its
  representative file's ancestors for the `snapshots` directory and taking
  the child folder's name directly below it — the same
  `snapshots/<commit>/...` layout `download_model` itself creates and
  `hf_repo_id_from_path` (above) already walks to recover the repo id.
  Stored on `ModelGroup` as `hf_repo`/`local_commit`, alongside `label`.
- **The remote commit.** `orangu::model_download::latest_commits` takes
  every *distinct* `hf_repo` id `list` found (deduped, so a repo with
  several `:quant` rows is still only queried once even when those rows
  were cached at different commits) and, in parallel via `rayon`'s
  `par_iter`, calls the very same `resolve_commit` `download` uses to
  resolve `main` (`GET /api/models/<repo>/refs`) — not a separate code
  path, so a repo `list` says is stale is guaranteed to actually update if
  `download`ed again. Its own short-lived `reqwest::Client` (via
  `build_client`'s optional timeout parameter) carries a 5-second timeout
  (`download`'s own client passes `None`, since a multi-gigabyte transfer
  legitimately takes longer), and every per-repo failure — unreachable
  Hub, DNS failure, rate limit, a repo gone private — is discarded with
  `.ok()` rather than propagated: `list` must still print its table when
  offline, just with no `(Refresh)` markers, rather than fail the whole
  command over one lookup (or over having no network at all). An empty
  repo list short-circuits before even building a client.

`main.rs`'s `Command::List` arm wires the two together: `group_models` runs
first, its groups' distinct `hf_repo` ids feed `latest_commits`, which
returns a `repo -> commit` map — not a "these repos are stale" set — and
`format_groups` (the renderer `format_list` itself now delegates to)
compares each row's *own* `local_commit` against that map when deciding
whether to append `  (Refresh)` after `SIZE`. Comparing per row rather than
per repo matters: a repo can have two `ModelGroup` rows cached at different
commits (e.g. `:Q4_K_M` downloaded weeks ago, `:Q8_0` downloaded today), and
only the one actually behind should be marked — a `HashSet` of "stale
repos" would incorrectly mark both just because they share a repo id. The
marker sits deliberately *after* `SIZE` rather than folded into `MODEL`, so
the shell completion scripts (above), which only ever read `list`'s first
two whitespace-separated columns, stay unaffected by a row growing a
trailing marker.

### CPU/GPU detection (`orangu::hardware`)

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

`main.rs`'s `Command::Suggest` arm calls the same `orangu::hardware::
detect_cpu`/`detect_gpus` pair `Command::System` does, then passes the
result to `suggest::format_suggestion`, which appends two size-suggestion
tables after `orangu::hardware::format_report`'s own CPU/GPU listing (via
the shared `push_suggestion_block` helper). There is no separate detection
path — `suggest` is purely a second interpretation of the same hardware
inventory `system` already knows how to gather (and the same report
printed at the top of every attached `orangu-server` startup — see the
Inference server chapter's Quick start section).

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
to read `hidden_size`/`layers` from. `estimate_hidden_dims` instead
estimates both from the parameter count alone. The standard transformer
parameter-count approximation (params ≈ 12 × layers × hidden_size²) is one
equation with two unknowns, so the split is underdetermined; it's resolved
by putting everything into the hidden size (`hidden_size = sqrt(params /
12)`), which makes `layers` work out to exactly 1 by construction. The
KV-cache estimate built on it therefore scales as context × √params — which
tracks modern GQA-era models well (their per-layer KV width shrinks as
depth grows, so total KV grows sublinearly in parameters), and matches the
fallback smcleod's own calculator uses when it has no real GGUF metadata to
read either.

`DEFAULT_BITS_PER_WEIGHT` (4.83, Q4_K_M) and `KV_CACHE_BITS` (8, Q8_0) match
this project's own established defaults (`orangu::model_download`'s
`DEFAULT_TAG_PREFERENCE`, and the same Q8_0 KV-cache quantization
`engine::kv_cache` itself stores) rather than assuming full FP16 throughout.

**A table, not a single guess.** Actual context usage varies far too much to
guess well from hardware alone, and bits-per-weight depends on which
quantization tag you end up downloading — so instead of picking one of each,
`push_suggestion_block` prints a row per context length in `CONTEXT_LADDER`
(1K up to a generous long-context ceiling, 262144) and a column per
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

**Two budgets, (up to) two tables.** `format_suggestion` computes two
separate budgets and prints a labeled `push_suggestion_block` for each,
`"Suggested model size (Dedicated)"` and `"Suggested model size
(Combined)"`. Both sum each eligible GPU's own `vram_total_bytes` —
deliberately *not* reduced by `vram_used_bytes`, since `suggest` estimates
the hardware's own capability (this file's module doc — "likely to run
comfortably on this machine", picked before any model is chosen), not how
much happens to be free at the exact moment it runs; whatever else is
transiently using VRAM (a compositor, a browser, an already-running
`llama-server`) shouldn't shrink a hardware-based estimate:

- `dedicated_vram_budget_bytes` sums every GPU `is_dedicated_for_budget`
  accepts (multiple dedicated cards add up) — `0` when there's none at
  all. The `Dedicated` block itself is skipped in that case
  (`gpus.iter().any(is_dedicated_for_budget)` gates the call to
  `push_suggestion_block`), rather than printing a `0 B` budget and a
  table where `suggest_param_count` correctly, but uselessly, reports
  nothing on the ladder fitting for every single cell.
- `combined_gpu_budget_bytes` sums every GPU `is_combined_budget_eligible`
  accepts (a `Shared` GPU's `vram_total_bytes` is already the system RAM
  total via `apply_shared_memory_total`, described above) — the more
  permissive figure, representing every device this server could spread
  layers across at once. Falls back to the CPU's own `total_memory_bytes`
  when that sum is `0` (no GPU detected at all). Always printed, even
  when it just reduces to system RAM alone — unlike `Dedicated`, this
  budget is never literally `0` on a real machine.

**`Unknown`-kind GPUs: a Windows-specific path.** On Linux/macOS,
`is_dedicated_for_budget`/`is_combined_budget_eligible` only ever see
`Dedicated`/`Shared` GPUs — `MemoryKind` is already reliably known there (see
above), so both functions have a plain, `cfg`-free body for those targets.
Windows is different: `windows_memory_kind` classifies *any* AMD adapter
`Unknown`, discrete Radeon and integrated APU alike, since that distinction
only exists in DXGI's `DXGI_ADAPTER_DESC` — unreachable from the WMI query
`detect_windows_gpus` uses. Rather than counting every `Unknown` GPU
(overcounts an APU's tiny carve-out as if it were a hard VRAM ceiling) or
none (undercounts a real discrete Radeon card), the `#[cfg(target_os =
"windows")]` variants of both functions trust an `Unknown` GPU's own
`vram_total_bytes` only above `WINDOWS_UNKNOWN_DEDICATED_THRESHOLD_BYTES`
(1 GiB — comfortably above a typical integrated carve-out, comfortably below
any real discrete card). Below the threshold it's treated like a `Shared`
GPU: excluded from both budgets, since its real ceiling is system RAM, which
`combined_gpu_budget_bytes`'s own `total_memory_bytes` fallback already
supplies once nothing else in the sum counts it.

### Shell completions (`shell.rs`)

Mirrors `orangu`'s own `-s`/`--shell-completions` (`src/bin/orangu/
shell.rs`, `print_shell_completions` in `main.rs`): hand-written bash/zsh/
fish scripts embedded as `&str` constants, selected by inspecting `$SHELL`,
rather than clap-generated completions. The positional `model` argument,
and `show`'s and `delete`'s own arguments, complete the same way `orangu`'s
own scripts complete session UUIDs — the shell function shells back out to
`orangu-server list` itself (`2>/dev/null`, so a missing config yields no
candidates rather than an error) and reads its first two columns with
`awk`. This keeps the completion logic entirely in the shell script,
depending on nothing but `orangu-server` itself being on `$PATH` — no
dynamic-completion protocol or extra binary flag is needed. The bash and
fish scripts also list the six subcommand names as literal completion
candidates alongside the dynamic model list at the first argument position;
the zsh script achieves the same with `_alternative` combining a `_values`
list (subcommand names) and a `compadd`-based function (model candidates)
for that position.

An earlier version of this explored `clap_complete`'s `unstable-dynamic`
feature for this instead; it was backed out in favor of the approach above
once `orangu`'s own precedent was found, since introducing a genuinely
unstable (semver-exempt) dependency wasn't warranted when a small,
self-contained shell script does the same job with zero new dependencies.

### GGUF loading and dequantization

`engine::loader` memory-maps the file and reads hyperparameters using the
same `<arch>.*` key names llama.cpp itself reads (confirmed directly
against `llama.cpp/src/llama-arch.cpp`'s `LLM_KV_*` table). Weight tensors
are **not** eagerly dequantized into RAM — each row is read straight from
the `mmap` and dequantized on demand, so even a large model's memory
footprint stays close to its file size.

`engine::quant`'s dequantization struct layouts and algorithms are taken
directly from ggml's own `ggml-common.h`/`ggml-quants.c`
(`dequantize_row_*`), not reimplemented from a description, so the CPU
path is bit-for-bit compatible with what llama.cpp itself reads. Supported
types: `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`
— any other `ggml_type` fails to load with a clear "not yet supported"
error rather than misreading it.

### Model forward passes

One `ModelForward` implementor per architecture family (`engine::arch::
mod`), so adding a family is additive rather than a rewrite:

- `llama.rs` — grouped-query attention, RoPE, RMSNorm, SwiGLU: the shape
  shared by `llama`/`qwen2`/`qwen3`/`mistral`/`qwen3vl` GGUFs (tensor names
  confirmed against `llama.cpp/src/llama-arch.cpp`'s `LLM_TENSOR_NAMES`
  table for `LLM_ARCH_LLAMA`).
- `gemma.rs` — targets `gemma4` (confirmed against upstream `llama.cpp`'s
  `src/models/gemma4.cpp`), with `gemma`/`gemma2`/`gemma3` as subsets of
  its hyperparameter set: soft-capping, sliding-window attention,
  per-layer embeddings (PLE), and GEGLU.
- `qwen35moe.rs` — Qwen3.5/3.6-MoE (confirmed against upstream
  `src/models/qwen35moe.cpp`/`delta-net-base.cpp`): a genuinely different
  shape, with mixture-of-experts FFN routing.

### Request scheduling and continuous batching

`engine::scheduler`'s `SlotPool` bounds how many requests generate
concurrently (`slots` in the config) and tracks each one's progress for
`/slots`. Each slot's prefill+decode loop (`engine::generate::run`) runs on
its own blocking-pool thread against its own KV cache — real concurrency,
bounded fairly by slot count, but not a single fused multi-sequence GEMM by
default.

`engine::batch::BatchCoordinator` is an opt-in alternative for that last
part: when `slots > 1` and the `ORANGU_BATCH_DECODE` environment variable
is set, concurrently-decoding requests within a short window are collected
and handed to `ModelForward::forward_batch_decode` as one call, fusing
every sequence's QKV/`wo`/FFN/PLE/`lm_head` matmuls into a single backend
call each (attention, RoPE, and the KV-cache write stay per-sequence, since
each sequence has its own cache and position). Correctness-verified
against independent per-sequence `forward` calls, but **off by default**:
under concurrent load (4 requests, 100 tokens each, `slots=4`) it measured
around 60% *slower* than the unbatched path — the generic `Backend::matmul`/
`matmul_batch` interface reads results back to the CPU between steps,
reintroducing per-layer round trips the Vulkan backend's own fused decode
path (below) was specifically built to eliminate, and that cost outweighs
the weight-bandwidth savings batching provides at this scale on the
hardware this was measured on. Left available behind the flag rather than
removed, since a genuinely GPU-resident batched-and-fused pipeline could
plausibly flip this positive on different hardware or at higher
concurrency.

### GPU backend architecture

`engine::backend::Backend` (`backend/mod.rs`) is the trait every backend
implements — `matmul`/`matmul_batch` plus a downcast hook (`as_vulkan`) the
model forward pass uses to reach `VulkanBackend`'s much larger fused
surface when it's the active backend. Five implementors exist:
`CpuBackend` (scalar with runtime AVX2 dispatch via `engine::tensor::dot`,
parallelized across output rows with `rayon`; always available, and the
fallback when no GPU backend is found), `VulkanBackend`, `CudaBackend`,
`OpenClBackend`, and `RocmBackend`.

`main.rs`'s `select_backend` implements the `backend = auto` cascade:
Vulkan, then CUDA, then OpenCL, then ROCm (if built with the `rocm`
feature), falling back to `CpuBackend` if none of them initialize. An
explicit `backend = <name>` instead calls that one backend's `try_init`
directly and fails to start if it returns `None`, rather than falling back
— useful when GPU inference was asked for specifically and a silent
CPU fallback would be the wrong failure mode.

### The Vulkan backend

`VulkanBackend` (`engine::backend::vulkan`, via `wgpu`'s Vulkan backend —
`ash` dlopens the system Vulkan loader at runtime, so no Vulkan SDK is
needed to build, only a driver to run against a GPU) is the mature,
hardware-verified backend. Each supported `ggml_type` gets two WGSL
compute pipelines sharing the same per-type dequantization math
(`dequant_element` in `vulkan_shaders.rs`, a line-for-line port of
`engine::quant`'s dequant algorithm restated in WGSL), dispatched
differently by `n_tokens`:

- **Small `n_tokens`** (decode's `n_tokens == 1`, the dominant case for
  interactive generation): `MAIN_REDUCE_SUFFIX` dispatches one workgroup
  per `(output row group, token)` pair — `REDUCE_N_ROWS` (4) output rows
  computed per workgroup, reusing each activation read across all four and
  combining partial sums via a tree reduction, with adjacent threads
  reading adjacent elements of the same row for memory coalescing.
- **Large `n_tokens`** (`>= 64`, e.g. a long prompt's prefill): a
  cooperative/tiled dispatch, one workgroup per output row, that
  dequantizes each weight block once per workgroup into shared memory and
  shares it across up to 64 tokens instead of redoing that dequant per
  token.

A weight tensor is uploaded once (still quantized) and cached on the GPU
for the model's lifetime. For Gemma-family models, `VulkanBackend::
fused_attention` chains QKV projection, Q/K-norm, RoPE, the KV-cache
write, and the attention kernel itself into one GPU submission;
`fused_post_attention` similarly chains the residual add, RMSNorm, and
GEGLU; `record_fused_layer`/`fused_layer` fold a whole layer (attention +
FFN) into one command encoder; and `GemmaModel::forward` chains every
layer plus `output_norm`/`lm_head` into one shared encoder per decode
step. Together these dropped GPU submissions per decode token from roughly
107 to 2 on this project's own benchmark hardware (an AMD RX 5500M running
`gemma-4-E2B`), taking real end-to-end decode throughput from ~1.4 tok/s
to the ~7–9 tok/s range depending on which of the opt-in kernels below are
also enabled — still meaningfully behind llama.cpp's own tuned Vulkan
backend (~36 tok/s on the same hardware/model), a gap now attributed to
kernel quality (`f32` math throughout, no subgroup reductions, no
flash-attention) rather than round-trip overhead, since round trips have
already been mostly eliminated.

Several further optimizations exist behind environment-variable opt-ins,
each correctness-verified against `CpuBackend` but left off by default
because a real, same-session A/B measurement didn't clearly justify making
it the default:

- **`ORANGU_KV_F16=1`** — stores the KV cache as `f16` instead of `f32`,
  halving its memory bandwidth at the cost of an extra per-layer cast
  dispatch. Measured ~2–3% *slower* at the context lengths this project
  can test end-to-end (KV traffic isn't the bottleneck at that scale); a
  much longer context could plausibly flip this positive.
- **`ORANGU_PACKED_DOT=1`** — dequantizes `Q4_K` weight elements in pairs
  and accumulates the dot product as `vec2<f16>` instead of two scalar
  `f32` multiplies. The first genuine kernel-quality win found: a real,
  reproducible ~19% throughput gain. Off by default because the win is
  only measured on one GPU generation (RDNA1), and packed-`f16` throughput
  isn't necessarily a universal hardware advantage.
- **`ORANGU_WIDE_LOAD=1`** — binds the weight buffer as `array<vec4<u32>>`
  (16-byte reads) instead of `array<u32>` (byte-wise reads), so a `Q4_K`/
  `Q5_K` block's whole header can load in one 16-byte read instead of
  several byte reads. Bit-for-bit correctness-verified for all 9 quant
  types; measured a real, reproducible ~11–13% throughput gain on `Q4_K`.
  Combining it with `ORANGU_PACKED_DOT` was tried and measured a
  *regression* relative to either alone, so the two are not meant to be
  combined.
- **`ORANGU_TILED_PREFILL=1`** — a `16×64`-output-tile GEMM for prefill
  (`n_tokens >= 64`) that reuses activations across output rows, unlike
  the default cooperative kernel (one workgroup per output row, each
  re-reading the whole activation matrix independently). Correctness-
  verified but unmeasured end-to-end: long prompts on this project's own
  dev hardware reliably trigger GPU driver hangs (a pre-existing hardware
  limit that affects the unchanged default kernel too, not something this
  change causes), which ruled out a trustworthy A/B.
- **`ORANGU_GPU_SAMPLE=1`** — runs greedy (temperature-0) argmax sampling
  on the GPU in the same submission as the forward pass, avoiding a full
  `[n_vocab]` logits readback. Correctness-verified, but measured ~5–10%
  *slower* — a single-workgroup reduction over a large vocabulary
  apparently costs more GPU time than the PCIe readback and CPU-side
  argmax it replaces. A wider, multi-workgroup reduction could plausibly
  flip this positive but hasn't been attempted.

Shader compilation is cached to disk across restarts
(`~/.orangu/server/<adapter-key>/cache.bin`, keyed by a vendor/device-
derived string so a cache built for one GPU is never handed to another) —
a startup-time optimization only, with no effect on decode/prefill
throughput once running.

### CUDA, OpenCL, and ROCm backends

`engine::backend::cuda::CudaBackend`, `engine::backend::opencl::
OpenClBackend`, and `engine::backend::rocm::RocmBackend` each implement the
same `Backend` trait, at a deliberately smaller scope than Vulkan: one
dequantizing matmul kernel per `ggml_type`, a direct port of
`vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` reduction strategy restated per
kernel language (CUDA-C, OpenCL-C, HIP-C), cross-checked against
`CpuBackend` the same way `VulkanBackend`'s own tests are. Deliberately
**not** ported: `VulkanBackend`'s cooperative/tiled dispatch, GPU-resident
attention/RoPE/norm fusion, fused whole-layer submissions, GPU-side argmax
sampling, and the disk pipeline cache — none of the three has been run
against real hardware during development (no NVIDIA GPU, no ROCm install,
no OpenCL ICD on the project's dev machine), so correctness rests on the
kernel math matching `engine::quant`'s already-verified dequant code
line-for-line, plus the same CPU cross-check test pattern `vulkan.rs`
uses (which, like those tests, skips gracefully rather than fails when no
matching device is found).

`cudarc` and the resolved `opencl3` version both dlopen their vendor
library (`libcuda.so`/`libnvrtc.so`, `libOpenCL.so`) at runtime and return
a real error if it can't be found, so `cuda`/`opencl` are always compiled
in — nothing extra is needed to *build* `orangu-server`. `cubecl-hip-sys`
(ROCm's underlying bindings) is different: it directly links
`-lamdhip64 -lhiprtc` at *build* time whenever its build script finds a
ROCm install, which would break a plain build on a machine without ROCm —
so `rocm` sits behind its own Cargo feature, off by default (see
[BUILDING.md](../../BUILDING.md)).

`cudarc` has one notable wrinkle: unlike every other fallible step here, it
`panic!`s (rather than returning a `Result`) the first time a driver/NVRTC
call is made and no `libcuda.so` is found. `CudaBackend::try_init` runs
`try_init_inner` under `std::panic::catch_unwind` (with the panic hook
silenced for the call) specifically so a non-NVIDIA machine gets the same
graceful `None`/CPU-fallback outcome every other missing-backend path
already has, not a crashed server.

### Correctness testing

`VulkanBackend`'s dequant math (each quant type, bit-for-bit against the
CPU backend, across both dispatch paths), fused post-attention chain
(including a dedicated test that calls it twice for one layer with
different inputs each time, to catch cache-reuse bugs specifically), and
fused attention (including GQA head-grouping, sliding-window attention,
proportional RoPE, and Gemma4's cross-layer KV-donor case — two different
layers sharing one KV cache) are covered by cross-check tests in
`engine::backend::vulkan::tests`, run on real Vulkan hardware whenever
it's present and skipped otherwise. The CUDA/OpenCL/ROCm backends follow
the same skip-if-no-device pattern.

### HTTP layer and web UI

`http::mod` assembles the router and shared `AppState` (model, scheduler
handle, config, start time); `http::openai` and `http::native` hold the
OpenAI-compatible and llama.cpp-native handlers respectively; `/v1/shutdown`
lives in `http::mod` itself since it's neither. Ctrl+C, `SIGINT`, and
`POST /v1/shutdown` all converge on the same shutdown path via
`tokio::select!`, mirroring `orangu-coordinator`'s own pattern.

`web::mod` serves a small server-rendered chat UI (vanilla HTML/CSS/JS, no
build step) on its own `web` port, sharing the same in-process `Engine` as
the API so a chat turn never makes an HTTP hop. `web::render` renders
markdown to HTML (including syntax-highlighted code blocks) with the same
`markdown`/`syntect` crates `orangu`'s terminal UI uses. `web::sessions`
persists each chat as `~/.orangu/server/sessions/<uuid>/chat.json`.

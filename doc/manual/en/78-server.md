\newpage

## Inference server internals

`orangu-server` (`src/bin/orangu-server/`) is a third binary in the same
Cargo package as `orangu` and `orangu-coordinator`. Besides serving a GGUF
model, it's also the machine's GGUF inventory tool (`system`/`suggest`/
`list`/`show`/`download`/`delete`/`refresh`) — stateless between runs for
those seven: every invocation re-detects hardware and re-scans the models
directory from scratch, so there is no cache, config-reload, or background
process to reason about for them. `system`/`suggest`/`show`/`delete` stay
entirely offline; `download` and `refresh` always talk to the Hub, and
`list` does too, before printing its table, to check each Hugging
Face-backed model for a newer commit (see `latest_commits` below) — swallowing the lookup silently rather
than failing when the Hub can't be reached. It does real tensor computation
itself for serving — GGUF loading, dequantization, the transformer forward
pass, sampling, and request scheduling are implemented in Rust with no
dependency on any C or C++ inference library.

### Module layout

- `main.rs` — CLI parsing (serving plus the `system`/`suggest`/`list`/
  `show`/`download`/`delete`/`refresh`/`bundle`/`prune` subcommands),
  model-spec resolution (`ModelSource`, which is either a `.gguf` on disk or
  a byte range inside this executable), GPU backend selection
  (`select_backend`), `format_show`/
  `DEFAULT_ARRAY_PREVIEW` (for `show`), `select_model_for_deletion`/
  `confirm` (for `delete`, and reused by `prune`/`refresh`), the two
  helpers every table-printing subcommand shares (`check_for_updates`, one
  Hub lookup per distinct repo; `dimming`, which downgrades a `Dimming`
  mode to `Dimming::Off` off a terminal), workspace-root
  resolution (`resolve_workspace`, over the shared
  `orangu::workspaces::resolve_workspace_root` that `orangu`'s own
  `-w`/`--workspace` uses), and process wiring
  (Ctrl+C/`SIGINT`/`--daemon`).
- `bundle.rs` — `bundle`: writing an executable that carries both this
  server and a model, and finding one at startup (`embedded`). See below.
- `reexec.rs` — replacing this process with one serving a different model,
  for the web console's **Load** button: descriptor hand-over, `argv`
  reconstruction, the header pre-check, and the one-shot fallback. See
  below.
- `refresh.rs` — `refresh`'s own CLI logic: the confirmation, the
  delete-then-download ordering, and the interactive picker that greys
  every already-current row; see below.
- `prune.rs` — `prune`'s own CLI logic (listing, `NR`/id resolution, the
  `all`/interactive/explicit-identifier flows), built on `web::sessions`'s
  activity tracking; see below.
- `config.rs`, `init.rs` — `orangu-server.conf` loading and the `--init`
  wizard. The web console has its own `[web]` section, whose *presence* is
  what enables it; `[web].host` falls back to `[orangu-server].host`, and
  the pre-section `[orangu-server].web` key is still honored when no `[web]`
  section is there to take precedence over it.
- `suggest.rs` — `suggest`: a hardware-based model-size estimate built on
  top of `orangu::hardware`'s own detection; see below.
- `shell.rs` — hand-written bash/zsh/fish completion scripts.
- `engine/loader.rs` — memory-maps a GGUF file, reads `<arch>.*`
  hyperparameters, resolves tensor byte ranges.
- `engine/quant.rs` — dequantization for every supported `ggml_type`.
- `engine/vecdot.rs` — the fused `int8` CPU kernels: a dot product taken
  against the still-quantized weight bytes, skipping the dequantize
  entirely. See below.
- `engine/iq_grids.rs` — the `IQ*` codebooks (lattice-point tables, the
  sign table, and `KVALUES_IQ4NL`) that `quant`, `vecdot` and the Vulkan
  shaders' uploaded grid buffer all read from one copy.
- `engine/tensor.rs` — the handful of numeric ops (matmul, RMSNorm,
  softmax, RoPE, SwiGLU/GEGLU) a forward pass needs, on plain `f32`
  slices — not a general ND-array library.
- `engine/arch/{mod,llama,gemma,phi,mistral,muse,inkling,nemotron,qwen35moe,qwen35,qwen3next,deepseek4,glm,kimi3,dflash}.rs` — one
  `ModelForward` implementor per architecture family.
- `engine/backend/{mod,cpu,vulkan,vulkan_shaders,metal,cuda,opencl,rocm}.rs`
  — the `Backend` trait and its six implementors; see below.
- `engine/backend/device.rs` — *which* device a backend runs on: the
  ranking policy, the `[orangu-server].device` override, and the startup
  inventory. Shared by all five GPU backends, and deliberately free of any
  one API's types so it is testable without a GPU.
- `engine/footprint.rs` — what *this model* costs on that device: weights
  split device/host, headroom, and how much context the headroom buys.
  Distinct from `engine/plan.rs`, which answers the same family of question
  about a GGUF nobody has opened yet. The two now weigh the same quantities
  — `Plan::device_bytes` applies `is_cpu_only_tensor`, the backend's own
  rule, and excludes the draft block exactly as `resident_tensor_sizes`
  does — so a plan's `Device` line and the startup footprint agree on a
  model they have both seen. They used to be incomparable: a plan spoke
  only of system RAM, so it could report "fits in RAM with 23.9 GiB to
  spare" for a model the footprint would then call 17.3 GiB too large for
  the card.
- `engine/placement.rs` — which device runs which layer when a model is
  spread across several: `SplitMode`, and the pure apportionment that turns
  capacities into contiguous layer runs.
- `engine/backend/multi.rs` — `MultiDeviceBackend`, the `Backend` that
  routes each matmul to the device holding its weights.
- `engine/expert_tier.rs` — which routed experts a *device* could hold, and
  what holding them would be worth. Policy and projection only; nothing
  executes on a device expert tier yet.
- `engine/tokenizer.rs` — a from-scratch BPE tokenizer.
- `engine/chat_template.rs` — renders `tokenizer.chat_template` via
  `minijinja`, plus the three compatibility shims real templates need:
  `minijinja_contrib::pycompat` for Python dict/str methods (`.get()`,
  `.strip()`), and a source rewrite that parenthesizes a keyword
  argument whose value is a bare inline conditional
  (`f(k=a if b else c)` → `f(k=(a if b else c))`). The latter is a
  minijinja parser gap that Jinja2 and llama.cpp's `minja` both lack, and
  it bites at *compile* time — one such argument in a branch that never
  runs still fails every request. `muse-glimmer`'s template has one. The
  third is the `tojson` filter itself: this build's `minijinja` has
  none (its JSON filters sit behind a feature this project does not
  enable), so the filter is registered here with `transformers`'
  semantics — `json.dumps` with `ensure_ascii` defaulting to false,
  compact separators, and none of Jinja2's HTML escaping — including the
  `ensure_ascii`/`indent`/`separators` keyword arguments templates pass
  it. `Inkling-Small`'s writes every tool declaration through one, and
  without the filter its tool-carrying chat requests fail at render time
  with a 500.
- `engine/sampling.rs` — repetition penalty, temperature/top-k/top-p/min-p.
- `engine/kv_cache.rs` — per-sequence KV cache buffers.
- `engine/scheduler.rs`, `engine/generate.rs`, `engine/batch.rs` — the
  multi-slot request scheduler and continuous-batching machinery.
  `generate.rs` also owns `MessageHeader`, which decides what a
  reply in one of the two *multi-message* assistant formats actually
  shows. For the `<|start|>…<|message|>` framing: the header text
  between the two markers is framing (`muse-glimmer` has the model write
  its own ` to=self` / ` to=user` recipient there) and is dropped like the
  markers around it, and a message addressed to `to=self` is reasoning, so
  a reasoning-suppressing role drops its whole body while every other role
  shows it followed by a blank line. The second format asks the same
  question with tokens: `inkling` writes no header and instead opens each
  body with a marker naming its *kind*
  (`<|content_thinking|>`/`<|content_text|>`, read off
  `Tokenizer::content_kinds`), and the same rule applies to it. Inert for
  a vocabulary with neither, which is every other model here.
- `http/{mod,openai,native}.rs` — the HTTP surface.
- `web/{mod,render,sessions,models,attachments}.rs` — the built-in chat UI.
  `sessions.rs` also owns the `session.json` activity marker
  (`mark_active`/`is_active`) and the prune-facing listing/sweep
  (`list_sessions_for_prune`/`sweep_empty_sessions`/`delete_session_dir`)
  `prune.rs` calls into. `models.rs` is the model manager's own HTTP
  surface; see below.

The GGUF-inventory subcommands lean on library modules shared with the rest
of the workspace rather than binary-local ones: `orangu::gguf` (the GGUF
binary-format reader), `orangu::model_spec` (directory scan, shard
grouping, and the Hugging Face repo-id/quant-tag reconstruction behind
`list`'s `MODEL` column), `orangu::model_download` (`download`'s fetch
logic), `orangu::os` (OS detection) and `orangu::hardware` (CPU/GPU
detection). Living in `src/`
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
Hugging Face's own hub cache — the layout an `-hf`/`--hf-repo` download
produces — names every file under `snapshots/<rev>/` as a
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
  (`general.architecture == "clip"`, identified the same way the reference
  `clip.cpp` loader does) and, if true, the file is skipped entirely —
  it's excluded before it ever reaches `ModelSummary`/`group_models`. An
  mmproj sidecar accompanies a base model rather than standing in as one
  (it is loaded via `--mmproj`, separately from the base checkpoint),
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

`shard_group_label` and `hf_tag_from_label` deliberately mirror the reference
implementation's resolver in `common/download.cpp` byte-for-byte, rather than
reinventing the convention:

- The shard suffix regex, `-\d{5}-of-\d{5}$`, matches
  `get_gguf_split_info`'s `re_split`.
- The quant-tag regex, `[-.]([A-Z0-9_]+)$` in the reference `re_tag`, is
  reimplemented as `hf_tag_from_label`: the trailing run of
  alphanumeric/underscore characters after the *last* `-` or `.` in the
  (shard-stripped) name, uppercased. It comes from the filename the reference
  resolver would match against, not from the tensor types, so it can say
  `Q4_K_M` where the ggml-type accounting can only say `Q4_K`. `QUANT` uses
  the same tag, narrowed by `quant_tag_from_label`/`is_quant_tag` to tags
  that really name a quantization (`hf_tag_from_label` alone would offer
  `IT` for `gemma-4-E2B-it`, fine to try as an `-hf` tag but not something
  to print as a quantization), and falls back to the dominant `ggml_type`
  only for a file whose name says nothing.

`group_models` then drops that `:TAG` back off the `MODEL` label whenever
`QUANT` already shows the same string — repeating it only widens the table's
widest column. Two quantizations of one repo consequently share a `MODEL`
cell, so `MODEL` is no longer a unique key: `ModelGroup::matches_label`
accepts the printed label *and* the reconstructed `<repo>:<quant>` form
(which keeps every spelling ever printed resolving locally, instead of
falling through to `resolve_or_fetch_model`'s download path), and a bare
request takes the first matching row. The final `sort_by` is stable, so rows
sharing a label keep their `(parent directory, file stem)` order and both
`NR` and first-match resolution stay put between runs. `delete` prints the
group's `quantization` in its confirmation line for the same reason — the
label alone can't say which of two rows is about to go.

`hf_repo_id_from_path` recovers `<user>/<model>` by walking a file's
ancestor directories for one matching `models--<user>--<model>` (checking
every ancestor, not just the immediate parent, since real files sit under
`snapshots/<rev>/`, sometimes with a further per-quant subfolder). This
directory-naming convention — `folder_name = "models--" + repo_id.replace("/",
"--")` — is Hugging Face's own, and is where a `-hf` download is documented to
land ("models downloaded with `-hf` are now stored in the standard Hugging Face
cache directory"). A file outside that layout has no
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

### Refreshing a model (`refresh.rs`)

`refresh` is `delete` followed by `download` of the same spec — the command
that acts on a `(Refresh)` marker. It is its own module rather than another
`main.rs` arm because three of its decisions are specific to it:

**Delete first, then download.** The reverse order would be safer against an
interrupted transfer, but a refresh exists precisely because the repo's files
changed, so the new revision is a full second copy on disk rather than the
blob-sharing symlink a re-download of the *same* commit produces. Downloading
first would mean a 17 GiB model needing 34 GiB free to refresh. The cost —
an interrupted download leaving the model missing rather than stale — is
recovered by re-running `refresh` or `download`, which resumes from the
`.part` file `download_attempt` left behind, and is stated up front in the
confirmation line rather than discovered afterwards.

**An ambiguous `MODEL` name is an error.**
`model_spec::resolve_refresh_target` mirrors `resolve_delete_target` (path,
then `NR`, then `MODEL` label) with exactly one difference: where `delete`
takes the first of several rows sharing a `MODEL` cell and names the
quantization it picked in its confirmation line, `refresh` refuses and lists
the quantizations on disk. It has to — `refresh` deletes what it then
downloads, so a first-match would refresh the wrong quantization *and* leave
the one the user meant untouched, which no confirmation line can undo. The
error only suggests the `<repo>:<quant>` spelling when naming a quantization
would actually disambiguate (two rows of the same repo *and* the same quant,
two snapshots deep, can only be told apart by `NR`). A path that resolves to
no group at all is a companion mmproj sidecar — `delete` synthesizes a
one-file group for it, but `download` only ever fetches one *alongside* its
base model, so `refresh` points at that model instead.

**The download spec comes off the filename, not `QUANT`.**
`ModelGroup::download_spec` rebuilds `<user>/<model>[:tag]` from the
representative path's own `hf_tag_from_label` tag rather than from
`quantization`: the `QUANT` column falls back to the dominant ggml type for
a file whose name carries no tag, and that type names no file in the repo,
so `select_files_to_download` would reject it as an unknown quant instead of
re-fetching the model that is actually on disk. A group with no `hf_repo` at
all has no spec, and `run` bails on it *before* deleting anything — a
hand-copied `.gguf` has no repo to come back from.

With no argument, `select_model_to_refresh` prints the same `format_groups`
table `list` does, with `Dimming::UpToDate` and the same
`check_for_updates` lookup, so the un-greyed rows are exactly the
`(Refresh)` ones — then prompts for an `NR`, the counterpart of
`select_model_for_deletion`. An unreachable Hub greys nothing (no row is
*known* to be behind), which reads correctly: with no update information, no
row is a better pick than another.

### Downloading from Hugging Face (`orangu::model_download`)

`download_model` implements `orangu-server download <user>/<model>[:quant]`
by directly mirroring the reference implementation's `common/download.cpp` and
`common/hf-cache.cpp` — read from that source rather than reimplemented
from a guess at the Hugging Face API, since the whole point is producing a
cache other GGUF tools recognize as already downloaded.

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
  the reference resolver uses, deliberately different from
  `orangu::model_spec::hf_tag_from_label`'s anchored *extraction* of an
  unknown tag from a filename, since here the tag is already known and
  being searched for. A file only matches as a **primary** if it's shard 1
  (or unsharded); a later shard never stands in for the whole model on its
  own.
- Without a `:quant`, `DEFAULT_TAG_PREFERENCE` (`["Q4_K_M", "Q8_0"]`, in
  that order — the ecosystem's default) is tried before falling back to
  the first model file found at all.
- Once a primary file is chosen, `shard_info` (the same
  `-NNNNN-of-NNNNN` suffix regex `orangu::model_spec::shard_group_label`
  strips, here also extracting the index and total) finds every sibling
  sharing its prefix and total count, so a multi-part model downloads
  whole.

**Choosing a multimodal projector, if any.** After the primary model file is
picked, `find_best_mmproj` (calling the generic `find_best_sibling` with
`keyword = "mmproj"`) directly mirrors the reference `find_best_sibling`/
`find_best_mmproj`: among every `.gguf` path containing `mmproj`, it prefers
the one sharing the deepest directory prefix with the primary file's own
path (rejecting any candidate whose directory list isn't a prefix of the
model's), then — among ties at that depth — the one whose quantization bit
depth (`extract_quant_bits`, reading the first run of digits in the
filename's trailing tag, e.g. `Q4_K_M` -> `4`, `BF16`/`F16` -> `16`, `F32`
-> `32`) is numerically closest to the primary file's own. This is the same
file orangu-server's own `-hf` auto-fetches the first time a vision-capable
model is launched with an image-related flag (verified against a real
repo, `unsloth/Qwen3.6-35B-A3B-GGUF`, which offers three top-level mmproj
variants — `BF16`/`F16`/`F32` — alongside a `Q4_K_M` primary; both this
code and a live `orangu-server -hf ...:Q4_K_M --image-min-tokens 1024` run
independently picked `mmproj-BF16.gguf`), so fetching it up front here means
`LLAMA_CACHE=<models>` already has it ready offline. If found, it's appended
to the file list `download_model` fetches, alongside whatever shards the
primary model itself has.

**Planning the model before fetching it (`RemoteModel`).** `resolve_commit`
and `list_repo_files` answer everything `engine::plan` needs *except* the
tensor tables, and those are not behind the download either — a GGUF file
puts its header at the front, so the tables are the first few hundred
kilobytes of each shard. `resolve_remote_model` performs exactly the two
Hub calls above and stops there, returning a `RemoteModel` that names the
commit and the selected shards with nothing fetched;
`RemoteModel::headers` then streams each shard from
`/<repo>/resolve/<commit>/<path>` into `GgufFile::read_from` and yields the
parsed header.

Two properties make this cost the header rather than the file. The GGUF
parser is strictly sequential and stops at the end of the tensor-info
table — pinned by `read_from_stops_at_the_end_of_the_tensor_table`, which
asserts the reader's final position rather than merely that the parse
succeeded, because a parser that read to EOF would return the same
`GgufFile` and only the unread bytes reveal the difference. And dropping a
`reqwest` response cancels the rest of its transfer, so returning from
`RemoteModel::header` closes the connection. Planning a 1.3 TiB repo
therefore transfers a few megabytes.

`headers` is lazy, so a consumer that fails on shard 1 never pays for the
remaining ten, and it yields `Result` rather than swallowing failures: a
header that will not parse is precisely the thing worth knowing before a
multi-hour download. `RemoteModel` deliberately excludes the `mmproj`
sidecar that `download_model` fetches — it is a separate CLIP-architecture
model, and its tensors would be counted as this model's weights by anything
reading the tables.

`main.rs`'s `plan_before_download` is the only caller. It hands
`RemoteModel::headers` to `engine::plan::analyze_shards` — the same
classifier `plan` uses on local files, which cannot tell where the tables
came from — prints the report, and consults `Plan::dense_fits_in` to decide
whether to confirm. It confirms on the dense part alone, never on the
experts: experts stream, so a model that overflows on them is slow rather
than broken, and prompting there would cry wolf on the workload the expert
path exists for. Every failure in this whole path reports one line and
returns "go ahead", since `download_model` is about to attempt the same
repo and is the better place for the real error to surface. It sits outside
`download_model_reporting` on purpose: that function also serves the web
console's model manager and model-spec resolution ahead of serving, neither
of which has a terminal to confirm on.

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

`main.rs`'s `Command::List` arm wires the two together through
`check_for_updates` (which does the dedupe-by-repo and is shared with
`refresh`, so both commands agree on what's stale): `group_models` runs
first, its groups' distinct `hf_repo` ids feed `latest_commits`, which
returns a `repo -> commit` map — not a "these repos are stale" set — and
`format_groups` (the renderer `format_list` itself now delegates to) asks
`ModelGroup::is_behind` to compare each row's *own* `local_commit` against
that map when deciding
whether to append ` (Refresh)` after `SIZE`. Comparing per row rather than
per repo matters: a repo can have two `ModelGroup` rows cached at different
commits (e.g. `:Q4_K_M` downloaded weeks ago, `:Q8_0` downloaded today), and
only the one actually behind should be marked — a `HashSet` of "stale
repos" would incorrectly mark both just because they share a repo id. The
marker sits deliberately *after* `SIZE` rather than folded into `MODEL`, so
the shell completion scripts (above), which only ever read `list`'s first
two whitespace-separated columns, stay unaffected by a row growing a
trailing marker.

### The `SUPPORTED` column

`list` prints a `SUPPORTED` column reading `Yes (<arch>)` or `No (<arch>)`
per row — so a user sees which models this build can actually load *before*
selecting one, rather than only discovering it can't once it's loaded.
`model_spec::format_groups` renders the column (and `format_list`'s
signature carries the `support`/`dim` parameters through), but the lib
deliberately doesn't decide *what* is supported: that judgement lives in
`orangu-server`, in `engine::loader::model_load_support`. So `main.rs`'s
`model_support` opens each group's representative file (header only — no
tensor data, the same cheap read `show` does), calls `model_load_support`,
and stores the result as one `model_spec::ModelSupport { architecture,
supported, unsupported_quant }` per group before handing the slice to
`format_groups`. Every *shard* of a group is inspected, not just its
representative file: a split model's later shards carry their own tensor
directory and can use a quantization shard 1 never does. An empty
slice omits the column entirely, which is what `format_list` (lib-side
tests) and any caller without the loader pass.

`model_load_support` is deliberately allowed to be *stricter* than
`resolve_arch_family` (whose family tables are the single source of truth
for the architecture *string*): a model whose architecture is recognised can
still carry tensors this build cannot read, so a bare `resolve_arch_family`
"yes" would promise a load that then fails partway through. It therefore
also checks every tensor's `ggml_type` against `quant::supports_type` and
reports the first unreadable one, which `ModelSupport::cell` renders as
`No (llama, TQ1_0)` — distinct from `No (glm-dsa)`, because only the former
is fixed by fetching a different quantization of the same model. Note this
is *not* the same question as the arch module's own tensor expectations:
gemma MoE checkpoints (`gemma-4-26B-A4B`, `blk.{i}.ffn_gate_inp.weight`
present) load via `arch::gemma`'s routed-expert path and report
`Yes (gemma4)`.

`SUPPORTED` answers "can this build read the file", which is not quite the
same question as "will it run on the backend you selected". Every GPU
backend covers fewer `ggml_type`s than `engine::quant` does, so a row can
read `Yes` and still be refused at startup by
`engine::backend::unsupported_tensor_types` — see the CUDA/OpenCL/ROCm
section for the coverage each backend has. The column deliberately does not
fold that in: it is rendered before a backend is chosen, and the same file
that one backend refuses runs on `cpu`.

A `No` row is *greyed* (dim ANSI SGR), not hidden: a user can still pick it
and will hit the same clear "not yet supported" error `prepare` gives for
any other unsupported model — the greying just deprioritizes it visually.

*Which* rows are greyed is `format_groups`' `dim` parameter, a
`model_spec::Dimming` rather than the boolean it started as, because
`refresh` wants the same table deprioritizing a different set of rows:
`Dimming::Unsupported` greys what this build can't load (`list`, `show`,
`delete`, the serve-time picker), `Dimming::UpToDate` greys what isn't
behind its repo (`refresh`), and `Dimming::Off` emits no escapes at all.
Every call site passes its mode through `main.rs`'s `dimming` helper, which
returns `Dimming::Off` unless `std::io::stdout().is_terminal()`. Piped or
redirected output — including what the shell completion scripts parse with
`awk '{print $1; print $2}'` — therefore stays escape-free, so an ANSI
prefix can never corrupt the `NR`/`MODEL` columns those scripts read. One
renderer serves every table this binary prints, so they all carry the
column consistently.

### OS detection (`orangu::os`)

`orangu::os::detect` gathers the `OS` section `orangu::hardware::
format_report` prints first, and `orangu::os::format_section` formats it —
the report's other two sections stay in `orangu::hardware`, so neither
module has to know how the other's fields are gathered. `OsInfo` is a flat
struct of `Option` fields precisely because platform coverage is uneven:
`format_section` skips any field that came back `None`, which is what lets
one formatter serve three platforms without a `cfg` in it.

Nothing here runs a subprocess. Every field comes from a Rust API:

- **Portable** (`sysinfo`, all three platforms): name, version, long
  version, kernel version, distribution id, hostname, uptime, swap
  total/used, and — everywhere but Windows, where `sysinfo` documents it as
  not working — load average.
- **POSIX** (`libc`, Linux and macOS): `sysconf(_SC_PAGESIZE)` for the page
  size and `getrlimit(RLIMIT_NOFILE)` for the open-file limit. The `as u64`
  casts on `rlim_cur`/`rlim_max` carry an
  `#[allow(clippy::unnecessary_cast)]`: `rlim_t` is `u64` on Linux and
  macOS, where clippy sees a no-op, but it's `i64` on the BSDs and
  `c_ulong` on some 32-bit targets that `#[cfg(unix)]` also covers.
- **Linux** (plain file reads): `/sys/class/dmi/id/sys_vendor` and
  `product_name` for `Machine` (world-readable, unlike the serial/UUID
  attributes beside them, which are root-only and not read), and
  `/sys/kernel/mm/transparent_hugepage/enabled` for the hugepage policy
  (`parse_selected_option` returns the bracketed entry out of
  `always [madvise] never`).
- **macOS** (`libc::sysctlbyname`, wrapped in `sysctl_string`): `hw.model`
  for `Machine`. The wrapper makes the usual two calls — a null buffer to
  learn the length, then one of that length — and trims the trailing NUL.

`is_redundant_long_version` decides whether the `Full name` line is worth
printing. `sysinfo`'s long version is `Linux (Fedora Linux 44)` on Linux —
`Name` and `Version` rearranged, plus the kernel name and punctuation —
but `MacOS 15.1 Sequoia` and `Windows 11 Pro` on the other two, where the
codename and edition appear nowhere else. Comparing word by word (ignoring
punctuation, and treating `linux` as already-known) drops the first and
keeps the other two, without a `cfg` deciding it per platform.

### CPU/GPU detection (`orangu::hardware`)

CPU statistics (brand, vendor, architecture, physical/logical core counts,
peak frequency, total/available RAM) come from
[`sysinfo`](https://docs.rs/sysinfo), used with its `system` and `component`
features only (no `disk`/`network`/`user`) to keep the dependency footprint
small — the same dependency `orangu::os` uses above.

**Power and thermals (`detect_power`).** `component` is the second feature,
and it is there for temperatures: `sysinfo::Components` reads sensors on
every platform this targets, which is worth a dependency rather than
writing three `hwmon`/SMC/WMI readers by hand. Sensors reporting `None`, or
exactly `0.0`, are dropped — the latter is a channel that is not wired up
rather than a component at freezing, and an integrated GPU's memory channel
does report it. What survives is sorted warmest first, because the only
sensor anybody acts on is the one nearest its limit.

The power *source* is not from `sysinfo`, which has no battery or AC-line
API at any version, so each platform is read natively:

- **Linux** walks `/sys/class/power_supply`, the interface every desktop
  battery indicator reads.
- **macOS** parses `pmset -g batt`, following `detect_macos_gpus`'s existing
  precedent of shelling out to the platform's own tool.
- **Windows** declares `GetSystemPowerStatus` and calls it directly. This is
  the one place that does *not* go through PowerShell, deliberately: the
  rest of this module needs WMI, whereas this is a single `kernel32` call
  filling a six-field struct, and spawning a shell to learn one byte would
  put several hundred milliseconds on the startup path. The ABI has been
  fixed since Windows 95, so a crate would carry nothing for us.

`classify_power_source` is split out from the Linux walk and compiled into
the tests on every platform, because it holds the one real trap: a laptop
plugged in with a full battery reports its battery status as **`Not
charging`**, not `Charging`. Deciding from the battery alone therefore means
reading a double negative that is easy to invert, and inverting it would
tell every desk-bound laptop it was running down a battery. Asking the AC
line first removes the question — an adapter reporting itself online is
authoritative whatever the battery says. A machine with neither an online
adapter nor a draining battery is a desktop, a server, or a container, and
all three answer `Mains`: every caller is really asking "is my power about
to run out".

`power_advisories` is kept separate from `performance_advisories` because
the two are different kinds of finding. Those are settings, Linux-only, and
each ships the `sudo` line that fixes it. These are conditions, hold on
every platform, and neither has a command as an answer — one is answered by
a cable and the other by airflow. The thermal one fires only against a
threshold the platform itself declared and only within
`THERMAL_ADVISORY_FRACTION` of it, which means a hot sensor that declares no
threshold is reported in the `POWER` section and never warned about. That is
deliberate: silicon runs hot under load, `Tctl` on AMD parts is a control
offset rather than a junction temperature, and a fixed limit invented here
would fire on machines that are working perfectly.

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

`main.rs`'s `Command::Suggest` arm calls the same `orangu::os::detect` and
`orangu::hardware::detect_cpu`/`detect_gpus` trio `Command::System` does,
then passes the result to `suggest::format_suggestion`, which appends two
size-suggestion tables after `orangu::hardware::format_report`'s own
OS/CPU/GPU listing (via the shared `push_suggestion_block` helper). There
is no separate detection path — `suggest` is purely a second interpretation
of the same inventory `system` already knows how to gather (and the same report
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

**Where the estimate ends.** `format_suggestion` closes with `NEXT_STEP`, a
fixed paragraph naming `download` and `plan` as the two commands that do
not have to estimate. It is there because every figure above it is derived
from a parameter count and the approximation described above — at this
point no model has been chosen, so there is no file to read, and
`engine::plan` (which reads real tensor tables) has nothing to read *from*.
`suggest` therefore cannot be made exact; what it can do is hand the user
the command that is. `download` plans the repo's real tables before
fetching it and `plan` does the same for a local model, so the size class
`suggest` produces gets checked against reality before any bandwidth is
spent on it.

**Two budgets, (up to) two tables.** `format_suggestion` computes two
separate budgets and prints a labeled `push_suggestion_block` for each,
`"Suggested model size (Dedicated)"` and `"Suggested model size (Total)"`.
Both read each eligible GPU's own `vram_total_bytes` —
deliberately *not* reduced by `vram_used_bytes`, since `suggest` estimates
the hardware's own capability (this file's module doc — "likely to run
comfortably on this machine", picked before any model is chosen), not how
much happens to be free at the exact moment it runs; whatever else is
transiently using VRAM (a compositor, a browser, an already-running
`orangu-server`) shouldn't shrink a hardware-based estimate:

- `dedicated_vram_budget_bytes` takes the maximum over every GPU
  `is_dedicated_for_budget` accepts — `0` when there's none at all. The
  maximum rather than the sum for the same reason `total_budget_bytes`
  uses one: a model is loaded onto a single device, and nothing here
  splits its tensors across two, so two 24 GiB cards are a 24 GiB budget.
  The `Dedicated` block itself is skipped in the `0` case
  (`gpus.iter().any(is_dedicated_for_budget)` gates the call to
  `push_suggestion_block`), rather than printing a `0 B` budget and a
  table where `suggest_param_count` correctly, but uselessly, reports
  nothing on the ladder fitting for every single cell.
- `total_budget_bytes` takes the **maximum** of every GPU
  `is_total_budget_eligible` accepts (a `Shared` GPU's `vram_total_bytes`
  is already the system RAM total via `apply_shared_memory_total`,
  described above) and the CPU's own `total_memory_bytes` — the largest
  single pool a run could use. System RAM being a candidate rather than a
  fallback is what covers a machine with no GPU at all, and it makes this
  budget a superset of `dedicated_vram_budget_bytes`: the `Dedicated`
  table is the fast subset of this one. Always printed, even when it just
  reduces to system RAM alone — unlike `Dedicated`, this budget is never
  literally `0` on a real machine.

  The maximum, **not** the sum: `select_backend` picks one backend for the
  whole model and there is no partial-offload split of layers between a
  GPU and the CPU, so no single run can draw on a discrete card's VRAM
  *and* system RAM together. Summing them printed budgets nothing could
  reach — on a 3.98 GiB card beside 62.19 GiB of RAM, a 66.17 GiB sum
  suggested a ~110B model needing 62.69 GiB, which neither pool holds.

**`Unknown`-kind GPUs: a Windows-specific path.** On Linux/macOS,
`is_dedicated_for_budget`/`is_total_budget_eligible` only ever see
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
`total_budget_bytes` already considers on its own.

### Shell completions (`shell.rs`)

Mirrors `orangu`'s own `-s`/`--shell-completions` (`src/bin/orangu/
shell.rs`, `print_shell_completions` in `main.rs`): hand-written bash/zsh/
fish scripts embedded as `&str` constants, selected by inspecting `$SHELL`,
rather than clap-generated completions. The positional `model` argument,
and `show`'s, `delete`'s and `refresh`'s own arguments, complete the same way `orangu`'s
own scripts complete session UUIDs — the shell function shells back out to
`orangu-server list` itself (`2>/dev/null`, so a missing config yields no
candidates rather than an error) and reads its first two columns with
`awk`. This keeps the completion logic entirely in the shell script,
depending on nothing but `orangu-server` itself being on `$PATH` — no
dynamic-completion protocol or extra binary flag is needed. The bash and
fish scripts also list the subcommand names as literal completion
candidates alongside the dynamic model list at the first argument position;
the zsh script achieves the same with `_alternative` combining a `_values`
list (subcommand names) and a `compadd`-based function (model candidates)
for that position.

An earlier version of this explored `clap_complete`'s `unstable-dynamic`
feature for this instead; it was backed out in favor of the approach above
once `orangu`'s own precedent was found, since introducing a genuinely
unstable (semver-exempt) dependency wasn't warranted when a small,
self-contained shell script does the same job with zero new dependencies.

`prune`'s own argument completes differently from
`model`/`show`/`delete`/`refresh` above: directly against `~/.orangu/server/sessions/*` (each entry a UUID
directory) plus the literal `all`, with no process invocation at all — this
time genuinely the same trick `orangu`'s own `-r`/`--resume` completion
uses (`_orangu_sessions`/`__orangu_sessions` in `src/bin/orangu/shell.rs`),
not just the same general shape. Shelling out to `orangu-server prune`
itself the way model completion shells out to `list` isn't an option here:
`prune` with no argument prints its table and then reads a selection from
stdin, so piping its output into a completion function would risk the
completion hanging on that prompt — `list` never reads stdin, which is
exactly why it's safe to use as a completion source and `prune` isn't.

### GGUF loading and dequantization

`engine::loader` memory-maps the file and reads hyperparameters using the
same `<arch>.*` key names every GGUF loader reads (confirmed directly
against the reference `llama-arch.cpp`'s `LLM_KV_*` table). Weight tensors
are **not** eagerly dequantized into RAM — each row is read straight from
the `mmap` and dequantized on demand, so even a large model's memory
footprint stays close to its file size.

`engine::quant`'s dequantization struct layouts and algorithms are taken
directly from ggml's own `ggml-common.h`/`ggml-quants.c`
(`dequantize_row_*`), not reimplemented from a description, so the CPU
path is bit-for-bit compatible with what other GGUF loaders read. Supported
types: the floats (`F32`, `F16`, `BF16`), the legacy quants (`Q4_0`,
`Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`), the whole K-quant family (`Q2_K` through
`Q6_K`), and the `IQ*` codebook quants (`IQ1_S`, `IQ1_M`, `IQ1_XS`,
`IQ1_XXS`, `IQ1_XXXS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL`, `IQ4_XS`) — any other `ggml_type` fails to load with a clear
"not yet supported" error rather than misreading it.

`IQ4_NL` is worth calling out because it turns up in files whose *name*
promises a pure K-quant. It is the one `IQ*` type that blocks at 32
elements rather than 256, and a K-quant needs 256 to divide the row, so
upstream's `llama_tensor_get_type` substitutes it per tensor wherever a
row is too narrow — every 896-wide row of a `Qwen2.5-0.5B`, for instance.
Any code that reads "`IQ*`" as "`QK_K`-blocked" gets its stride wrong for
this one type alone; `quant::block_layout`, `vecdot::supports`, and every
backend's block-size table each place it with the 32-element quants
deliberately.

The six repacked layouts — `Q4_0_4_4`/`_4_8`/`_8_8` (ggml ids 31-33) and
`IQ4_NL_4_4`/`_4_8`/`_8_8` (36-38) — are handled differently from every
other type, in `engine::loader` rather than `engine::quant`'s per-row
dequantizers. They are ARM-SIMD pre-repacked `Q4_0`/`IQ4_NL`: ggml retired
the ids and never shipped a `to_float` for any of them (only `gemv`/`gemm`
kernels), which is why upstream refuses such a file. The packing is
lossless though — `repack_{q4_0,iq4_nl}_to_*_bl` plus the `make_block_*`
functions interleave 4 or 8 *rows* into shared records — so
`quant::deinterleave_repack` inverts it and `LoadedModel::open` rewrites
every such tensor to its plain base type before anything reads one. Past
the loader these ids do not exist, so the CPU fused kernel and all four GPU
backends serve them with no new code.

`IQ1_XS`/`IQ1_XXS`/`IQ1_XXXS` (ids 64-66) are the other three that sit
outside ggml's own enum, and the reason `gguf::GGML_TYPE_NAMES` now runs
past its end. They are `IQ1_S` with a narrower codebook index — 10, 9 and 8
bits instead of 11, into 1024-, 512- and 256-point subsets of the very same
2048-point lattice — reaching 1.4375, 1.3125 and 1.1875 bits per weight, and
a "dynamic" 1-bit release of a trillion-parameter mixture-of-experts model
stores its expert stacks as the last of them. Because only the index
changes, `dequantize_iq1_xs`/`_xxs`/`_xxxs` are the `IQ1_S` dequantizer with
a different index reassembly, ending in the same `push_iq1_grid` call, and
`iq1_narrow_sub_scale` is shared by the two whose scale and sign live in a
nibble array rather than in the index byte itself. Nothing downstream needed
a change: the expert stacks these appear in are read on the CPU whatever the
backend, so `SUPPORTED_TYPES` for the GPU backends stays as it was.

The ids matter as much as the layouts. 42-63 are left free for ggml to grow
into, which is what makes 64+ safe for a quantizer to claim: a build that
does not know these types rejects the file instead of reading it at the
wrong stride. `GGML_TYPE_NAMES` therefore spells the gap out as `None`,
printed as `reserved(N)`, so `list` distinguishes "an id ggml has not used
yet" from `unknown(N)`, "a type newer than this build".

`testdata/ggml-dequant-reference.bin` covers 20 types rather than 16 for
this: the three new ones plus `IQ1_S`, whose presence is what says the
harness covers this family rather than agreeing vacuously. Regenerating it
leaves the 16 that were already there byte-identical, but needs a ggml
build that implements the narrow types — that fixture is local rather than
checked in, so on a checkout without it the cross-check skips and
`dequantizing_every_type_is_unchanged`'s checksums are what hold the three
dequantizers still.

`quant::repack_layout` is the whole specification, as
`(base type, rows, run, xor)`. Two entries in it do not follow from the
type names and would corrupt every weight if guessed:

- `*_4_8` interleaves **4** rows in 8-byte runs, not 8 rows. The digits are
  (rows) x (run), and `block_q4_0x4`/`block_iq4_nlx4` back both the `4_4`
  and `4_8` variants.
- The `Q4_0` family XORs each byte with `0x88`, letting the ARM kernels
  read a nibble as a signed offset without a subtract. The `IQ4_NL` family
  does **not** — its nibble indexes `KVALUES_IQ4NL`, so flipping the high
  bit would select a different level for every weight rather than adjusting
  a sign.

`IQ4_NL_4_8` is the one layout with no executable upstream definition:
`make_block_iq4_nlx4`'s 8-byte branch is commented out and marked "this
branch seems wrong". Its entry is the straightforward generalization — the
`x4` index math with 8-byte runs — and no released file is known to carry
the id.

Two properties follow from the row interleaving. The conversion is eager,
not lazy: a row's blocks are strided across its 4- or 8-row group, so there
is no row-shaped slice of the mapped file to defer — `TensorLocation` holds
an owned buffer for these tensors instead of an `Mmap` slice (both behind
the `TensorBytes` trait object, so readers are unchanged). And
`quant::dequantize` refuses these ids outright rather than decoding a
"row", since bytes for one row are not a thing that exists in this layout.

Correctness is pinned by `testdata/ggml-dequant-reference.bin`: random
blocks of every quantized type paired with the `f32`s ggml's own
`ggml_get_type_traits(t)->to_float` produced from them, compared
bit-for-bit. Regenerate it with `testdata/ggml-dequant-reference.c` (see
`quant.rs`'s `read_ggml_reference` for the command) when adding a type —
appending to that file's type list leaves every existing entry
byte-identical.

### The fused `int8` CPU path (`engine::vecdot`)

The obvious way to multiply a quantized weight matrix on the CPU is to
dequantize each row to `f32` and take an `f32` dot product. `engine::vecdot`
does neither: it quantizes the *activations* to `int8` once (32 elements per
shared `f32` scale), then dots them against the weight bytes while those are
still quantized, using integer SIMD. That removes the dequantize, removes
the per-row allocation, and replaces scalar `f32` multiplies with 16- or
32-wide `int8` ones (NEON `vmull_s8`/`sdot` on aarch64; AVX-512 VNNI, AVX2
or SSE4.1 on x86-64, all chosen by runtime feature detection).

Five types have a fused kernel — `Q8_0`, `Q5_0`, `IQ4_NL`, `Q4_K`, `Q6_K`.
Anything else returns `false` from `vecdot::supports` and the caller keeps
the ordinary dequantize path, so the module is strictly additive.

`supports` takes the row length as well as the type, and that second check
is load-bearing rather than defensive. A GGUF row is only guaranteed to be
a whole number of blocks, and the block sizes differ: `Q8_0`, `Q5_0` and
`IQ4_NL` need `32 | in_dim`, while `Q4_K` and `Q6_K` need `256 | in_dim`.
Real small models mix both within one file — `Qwen2.5-0.5B` is 896 wide and
`SmolLM2-360M` 960, each a whole number of 32-element blocks but neither a
multiple of 256, so their attention weights take the 32-block kernels while
only the `256`-divisible `ffn_down` reaches a K-quant one.

Every supported type reduces to the same shape, which is what lets one dot
loop serve all five:

```text
weight[i] = scale[i / GROUP] * q[i]  -  min[i / GROUP]
```

`min` is zero for the symmetric types (`Q8_0`, `Q5_0`, `Q6_K` and `IQ4_NL`
all fold their bias — or, for `IQ4_NL`, their codebook level — straight into
the signed `int8` weight, so no correction term survives); only `Q4_K` is
genuinely asymmetric. `GROUP` is 16 because `Q6_K` carries one scale per 16
weights; every other type repeats its scale across the two halves of its
32-element block, which the dot loop exploits.

There are two entry points, because decode and prefill want different
things. `dot_row` (GEMV) walks a row's blocks once for a single token.
`unpack_row` + `dot_unpacked_multi` (GEMM) unpack a row into plain `int8`
plus per-group scale metadata **once per matmul rather than once per
(row, token)**, which is what stops prefill from re-unpacking the same row
for every token in the batch.

Quantizing activations to `int8` is lossy, exactly as it is anywhere else —
that is the accepted tradeoff of this kernel family, not an oversight. The
tests check every kernel against `quant::dequantize` plus an exact `f32`
dot, to within the error `int8` activation quantization can introduce, and
separately require the two entry points to agree with each other far more
tightly than either agrees with the reference (they quantize identically,
so only summation order differs).

### Model forward passes

One `ModelForward` implementor per architecture family (`engine::arch::
mod`), so adding a family is additive rather than a rewrite:

- `llama.rs` — grouped-query attention, RoPE, RMSNorm, SwiGLU: the shape
  shared by `llama`/`qwen2`/`qwen3`/`mistral`/`qwen3vl` GGUFs (tensor names
  confirmed against the reference `llama-arch.cpp`'s `LLM_TENSOR_NAMES`
  table for `LLM_ARCH_LLAMA`).

  These architectures share a block shape but **do not share a RoPE
  pairing**, and the module selects it per architecture
  (`rope_layout_for`), mirroring upstream's `llama_model_rope_type`:
  `llama` and `mistral` rotate *consecutive* elements
  (`LLAMA_ROPE_TYPE_NORM`, pairs `2p`/`2p+1`), while `qwen2`, `qwen3` and
  `qwen3vl` rotate elements offset by half the rotary width
  (`LLAMA_ROPE_TYPE_NEOX`). Using one pairing for all of them is silently
  wrong rather than an error — position 0 is the identity under both and
  small positions rotate by small angles, so a short prompt still reads
  fine while a longer one collapses into repetition.

  A Llama-3.1/3.2 checkpoint additionally ships `rope_freqs.weight`, the
  per-pair frequency divisor its `"llama3"` RoPE scaling is baked into at
  conversion time; upstream applies it whenever present
  (`get_rope_factors`' first branch), and so does this module. It only
  moves the lowest frequencies, so it matters at long context rather than
  in a short prompt.
- `gemma.rs` — targets `gemma4` (confirmed against the reference
  `src/models/gemma4.cpp`), with `gemma`/`gemma2`/`gemma3` as subsets of
  its hyperparameter set: soft-capping, sliding-window attention,
  per-layer embeddings (PLE), and GEGLU.
- `qwen_hybrid.rs` — **not an architecture**: the hybrid full-attention /
  gated-DeltaNet trunk the three Qwen 3.5-family modules below all run,
  mirroring the way `qwen35.cpp`, `qwen35moe.cpp` and `qwen3next.cpp`
  upstream all call one `llm_build_delta_net_base` and differ only in
  `build_layer_ffn`. It owns the hyper-parameters, the layer-kind mask, both
  layer loaders, both forward halves, and the KV-cache layout, and is generic
  over one trait — `HybridFfn` — which is the only thing the three differ in.
  Two variations it absorbs so the architecture modules need not: recurrent
  QKV+gate as either the split `attn_qkv`/`attn_gate` or one fused `ssm_in`,
  and beta/alpha as either the split `ssm_beta`/`ssm_alpha` or one packed
  `ssm_ba` (`ssm_beta_alpha` in older conversions) whose rows interleave the
  two per K/V group.

  Its `trunk_layer_count` is `block_count` *less* `nextn_predict_layers`, as
  in `glm.rs` and `nemotron.rs`: releases that ship a multi-token-prediction
  head count it as the last `block_count` entry rather than putting it past
  the end, and it carries no attention tensors, so reading it as a trunk
  layer fails on `blk.N.attn_qkv.weight` rather than running the wrong thing.
  That trim used to be in `qwen35moe.rs` only, which is exactly the drift a
  shared trunk removes — `unsloth/Qwen3.8-27B-GGUF` is a *dense* `qwen35`
  file with `nextn_predict_layers = 1` and could not be loaded at all.
- `qwen35moe.rs` — Qwen3.5/3.6-MoE (confirmed against upstream
  `src/models/qwen35moe.cpp`/`delta-net-base.cpp`), e.g.
  `unsloth/Qwen3.6-35B-A3B-GGUF`: `qwen_hybrid::Trunk` with
  `qwen_hybrid::MoeFfn` — softmax top-k routing (renormalized) over routed
  experts plus a separately-`sigmoid`-gated shared expert.
- `qwen35.rs` — Qwen3.5-family dense (confirmed against upstream
  `src/models/qwen35.cpp`), e.g. `unsloth/Ornith-1.0-9B-GGUF` and
  `unsloth/Qwen3.8-27B-GGUF`: the same trunk with `qwen_hybrid::DenseFfn`, a
  plain SwiGLU FFN in place of MoE routing. That FFN is `arch::swiglu_ffn`,
  shared with `llama.rs` — which is what serves `qwen2` and `qwen3` — since
  `LLM_FFN_SILU`/`LLM_FFN_PAR` is one computation whatever the block around
  it looks like.
- `qwen3next.rs` — Qwen3-Next (confirmed against upstream `src/models/
  qwen3next.cpp`), e.g. `unsloth/Qwen3-Coder-Next-GGUF`: the same trunk and
  the same `MoeFfn` as `qwen35moe.rs`. It differs only in which recurrent
  tensor-naming variants its files use, which the trunk's loader already
  takes, so this module carries no forward pass of its own.
- `deepseek4.rs` — DeepSeek-V4 (`general.architecture = "deepseek4"`),
  e.g. `unsloth/DeepSeek-V4-Flash-0731-GGUF:IQ1_M`, confirmed against
  upstream `src/models/deepseek4.cpp` and the block planner in
  `src/llama-kv-cache-dsv4.cpp`. Four things are unlike anything else here:
  **hyper-connections** (the residual stream is `hyper_connection.count`
  parallel streams; each half-layer collapses them to one vector and
  expands the result back out, with per-token weights the layer predicts
  from the streams themselves and a Sinkhorn-normalized stream-combination
  matrix); **a single shared key/value vector per token**
  (`head_count_kv = 1`, and the value *is* the key, so the attention output
  carries the keys' rotations and is de-RoPEd at the query's position
  before the grouped low-rank output projection); **compressed attention**
  (`attention.compress_ratios` per layer: `0` is the plain
  `attention.sliding_window`, `128` adds one pooled key per completed
  128-token block, `4` adds one per completed *overlapping* 8-token window
  of which only the `attention.indexer.top_k` best-scoring are attended,
  scored by the lightning indexer's own narrower compressed cache); and
  **hash-routed experts** on the first `hash_layer_count` layers, whose
  expert selection is the `I32` `ffn_gate_tid2eid` table indexed by token
  id rather than anything the router scores. Upstream's Hadamard key
  rotation is deliberately not carried over — it is orthonormal and
  self-inverse and is applied to query and key alike, so it changes no dot
  product; it exists there for quantized KV caches. Compressed blocks live
  in `KvCache::layers` like any other key, in slots whose rows each stand
  for `ratio` tokens (`LayerCache::stride`), which is what keeps rollback,
  prefix reuse and slot persistence exact for them.
- `glm.rs` — GLM with DeepSeek sparse attention (`general.architecture =
  "glm-dsa"`), e.g. `unsloth/GLM-5.2-GGUF`, confirmed against upstream
  `src/models/glm-dsa.cpp` and the DSA `build_attn` overload in
  `src/llama-graph.cpp`. The block shape is an ordinary pre-norm
  transformer and the FFN is `qwen35moe`'s routed-plus-shared MoE (dense
  for the first `leading_dense_block_count` layers); the attention is what
  earns it a module. **MLA, absorbed**: one `kv_lora_rank`-wide compressed
  vector per token plus a shared rotary part is the whole layer's K and V
  for every head, and instead of decompressing it, the query goes through
  the per-head key-decompression matrix (`attn_k_b`) and the output comes
  back through `attn_v_b`. The cache is therefore K-only — the value is the
  leading `kv_lora_rank` of the same row, which is what
  `arch::attend`'s `value_dim` parameter exists for. **The lightning
  indexer**: a 32-head, 128-wide attention with its own per-token key cache
  scores every earlier position, and the layer attends only the
  `indexer.top_k` best; `attention.indexer.types` says which layers score
  and which reuse the last scoring layer's choice, defaulted from the
  reference config for the GLM-5.2 quants that omit the key. The scoring
  pass is skipped below `indexer.top_k` positions, where it provably cannot
  change the mask. Note the two normalizations that differ from everything
  around them: the indexer key is **LayerNorm**ed (mean-centred, with a
  bias) rather than RMS-normed, and the attention scale follows
  `key_length_mla`, not the wider absorbed query/key width. The
  multi-token-prediction block (`blk.78`) is not run, as in `deepseek4.rs`.
- `kimi3.rs` — Kimi-K3 (`general.architecture = "kimi-k3"`), e.g.
  `unsloth/Kimi-K3-GGUF`, transcribed from `src/models/kimi-k3.cpp` in
  upstream's Kimi-K3 pull request (ggml-org/llama.cpp#26185), the
  `LLM_FFN_SITU` arm it adds to `build_moe_ffn`, and the delta-net
  recurrence in `src/models/delta-net-base.cpp` that the released tree
  already carries for Kimi-Linear. `attention.head_count_kv` is a per-layer
  *array* here: `0` marks a **KDA** layer (three in four), anything else a
  **MLA** layer. The KDA layer is `qwen35moe.rs`'s delta rule with two
  differences — a short causal convolution over each of Q/K/V before it
  (the three kernels are concatenated into one depthwise pass, so
  `RecurrentLayerState::conv_step` is reused unchanged), and a decay that is
  **per key dimension** rather than one scalar per head, from
  `kda.gate_lower_bound * sigmoid(exp(A_log) * (f + dt_bias))`. The MLA
  layer is `glm.rs`'s absorbed form with the RoPE removed (nothing is
  rotated) and a sigmoid gate on the output. Beyond those: **cross-layer
  residual attention** (`res_mix`/checkpoint banking — note the scores use
  RMS-normalized values while the weighted sum uses the raw ones), **latent
  MoE** (routed experts at `expert_latent_length`, router scoring the
  full-width input), and the **situ** activation. The delta-net state is
  the memory cost of this architecture: `kda.head_dim` squared per head per
  recurrent layer, fixed and independent of context.
- `nemotron.rs` — Nemotron-H (`general.architecture = "nemotron_h_moe"`),
  e.g. `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`. The one
  family here whose **block is a single sub-layer**, not an
  attention/FFN pair — which is the thing most likely to be got wrong by
  analogy, because a loader written to the usual shape finds every tensor
  it looks for on *some* layer and simply builds the wrong graph. Each
  block is one mixer under one `attn_norm` and one residual add, and the
  per-layer metadata arrays pick which: `feed_forward_length[i] != 0` is a
  MoE block, else `attention.head_count_kv[i] != 0` is attention, else a
  state-space block. Both arrays must be read — a zero
  `feed_forward_length` alone covers the recurrent *and* the attention
  blocks. On the 30B-A3B model that is 23/6/23 across 52 trunk layers, so
  `new_kv_cache` allocates six positional slots and 23 recurrent ones.

  **The state-space block.** One `ssm_in` projection fans out to
  `[z | x | B | C | dt]` — the gate, the convolved streams, and the
  per-head timestep, in that order. `x`/`B`/`C` go through the causal
  depthwise convolution (`RecurrentLayerState::conv_step`, the primitive
  `qwen35moe.rs` already had, plus a bias this architecture has and the
  delta-net families do not) and a SiLU; then per head, `dt` is biased and
  softplussed once and used twice — to scale this token's contribution and,
  through `ssm_a`, to decay the state. `ssm_a` is stored already negated,
  so the decay is `exp(step * a)` with no sign flip of its own. `ssm_d` is
  a per-head skip on the block's own *convolved* input, added to the
  recurrence output before the gate. Then the gate is applied **and only
  then** the grouped norm — `ssm_norm` is `n_group` consecutive
  `ssm.inner_size / n_group`-wide weight vectors, each normalizing its own
  slice. Reversing those two (norm, then gate — which is what
  `qwen3next.rs` does) loads, runs, and answers fluently while being wrong.

  The state is **rectangular**: `[ssm.inner_size / ssm.time_step_rank,
  ssm.state_size]` per head, `[64, 128]` on this model. That is what
  `kv_cache::RecurrentSpec::ssm` exists for — the delta-net families all
  take `::delta_net`, whose per-head state is square because it accumulates
  an outer product of two same-width vectors. `RecurrentLayerState` carries
  the two axes separately for exactly this reason.

  **The attention block** is plain causal GQA with nothing on it: no
  rotation (this architecture rotates nothing anywhere — the
  `rope.dimension_count` and `rope.freq_base` in the file are vestigial and
  applying them is silently wrong), no QK-norm, no biases, no sliding
  window. `head_count_kv` is read per layer even though every attention
  block on this model agrees, since it is the same array that classified
  the block.

  **The MoE block** has **no gate projection** on either branch: routed
  experts and the one shared expert are both `down(relu(up(x))^2)`, two
  matrices rather than SwiGLU's three, and there is no `ffn_gate_exps` /
  `ffn_gate_shexp` tensor to read. The shared branch is added unweighted.
  Routing is `arch::mod`'s `ExpertRouting` unchanged — sigmoid
  probabilities, `exp_probs_b` steering the selection only, top-k,
  normalize, scale — and evaluation is the shared
  `evaluate_routed_experts` / `project_expert` host path.

  The router and the shared expert's up projection read the same normalized
  input and neither depends on the other, so `router_and_shared_up` issues
  them as one `matmul_batch` — one input upload and one backend round trip
  per MoE block rather than two. It splits them only when the shared expert
  is a quantization the selected backend has no kernel for, which is allowed:
  `engine::backend::is_cpu_only_tensor` exempts the shared-expert matrices
  from the startup device-capability check so that a low-bit file still gets
  a GPU for the rest of the model. Every shared-expert matmul therefore goes
  through `arch::mod`'s `matmul_host_fallback`; handing such a tensor
  straight to the backend is not a slow path but a panic.

  The trailing multi-token-prediction block (`nextn_predict_layers`, one
  extra `block_count` entry with its own `nextn.*` tensors, attention and
  MoE) is not loaded, as in `glm.rs` and `deepseek4.rs`. `engine::plan`
  knows about it too — `trunk_block_count` bounds which blocks it charges to
  the resident and streamable totals, so a draft block's experts are not
  counted as weight anything reads. That applied to all three of these
  architectures, not just this one.

  The recurrence is sequential by construction, but the projections around
  it are not, and that distinction is most of the throughput: `ssm_in`, the
  MoE router, the shared expert and `ssm_out` each run once for the whole
  batch, with only the per-token mixing between them walking the prompt.
  Writing the recurrence output into one buffer and projecting it once,
  rather than projecting each token as it is produced, is worth two orders
  of magnitude on prefill: the per-token form issues a backend round trip
  per token per block, and the round trip, not the arithmetic, is the cost.
- `dflash.rs` — DeepSeek draft sidecars (`general.architecture =
  "dflash"`), e.g. the `dspark-` file in
  `unsloth/DeepSeek-V4-Flash-0731-GGUF`. There is no standalone model in
  these files: no `token_embd`, no `output`, and the graph reads the target
  model's hidden states (`dflash.target_layers`). `main::
  auto_pair_dflash_target` therefore resolves the paired target from the
  same Hugging Face repo and serves *that* — for the DeepSeek-V4-Flash
  sidecar, the `deepseek4` model above. This module is only the error for a
  draft handed over directly with no repo to pair it from.
- `mistral.rs` — `mistral3` (Mistral 3 / Ministral-3), confirmed against
  upstream `src/models/mistral3.cpp`. The block shape is `llama.rs`'s node
  for node; what earns it a module is four hyperparameters that are
  wrong-by-default rather than absent: a head width read from
  `attention.key_length` (Ministral-3-3B declares 128 where
  `n_embd / n_head` would give 96, and its `attn_q.weight` really is
  `[3072, 4096]`), YaRN RoPE scaling, `NORM` rope pairing, and a
  Llama-4-style attention temperature scale — the last of which is exactly
  `1.0` below the trained context, so a short prompt cannot tell whether it
  is implemented at all.

- `muse.rs` — `muse-glimmer` (Muse-Glimmer), e.g.
  `unsloth/Muse-Glimmer-30B-GGUF`, confirmed against upstream
  `src/models/muse-glimmer.cpp`. A dense GQA decoder: `llama.rs`'s block
  plus four things this engine already had, each from a different family,
  and one it did not. Borrowed: gemma's **sandwich norms**
  (`attn_norm`/`post_attention_norm` and `ffn_norm`/`post_ffw_norm`, all
  four on every layer — the presence of `ffn_norm` is what distinguishes
  this from `qwen35.rs`'s two-norm block), gemma/qwen3's **per-head
  QK-norm**, the alternating **sliding-window pattern** (a *scalar*
  `attention.sliding_window_pattern` of 4 fed to upstream's
  `set_swa_pattern`, so `il % 4 < 3` is windowed at 2048 and every fourth
  layer attends the whole prefix), and kimi3/qwen35's **sigmoid output
  gate on attention** — here its own `attn_gate` tensor rather than folded
  into the query projection, projected from the same normed input as
  Q/K/V and multiplied into attention's output before `attn_output`. New:
  **RoPE runs on the sliding-window layers only** (`use_rope =
  hparams.is_swa(il)`), so the full-attention quarter is NoPE and caches
  its keys unrotated. Nothing in the file says so.

  Four further details are each invisible from the tensor directory and
  each produce a model that loads and runs while answering "The capital of
  France is" with a newline: the token embeddings take a **weightless**
  RMSNorm on the way in (gemma scales by `sqrt(n_embd)` instead, and the
  llama family does nothing); the two *post*-norms use a hardcoded
  `1e-8` epsilon rather than the file's `attention.layer_norm_rms_epsilon`
  (`1e-5`); the FFN is SwiGLU, not the GEGLU the gemma-shaped block would
  suggest; and the rotation is `NORM`-paired — `muse-glimmer` sits in
  upstream's `LLAMA_ROPE_TYPE_NORM` arm with `llama`/`mistral`, not with
  the gemma and Qwen families it otherwise resembles. Nothing extra is
  folded in at run time for the query-key scale factor: upstream notes it
  was baked into `attn_q_norm`'s weights at conversion, and `attn_k_norm`
  is a vector of ones.

  This module is CPU-orchestrated, like `qwen35.rs`/`glm.rs`/`kimi3.rs` —
  matmuls still dispatch to the `Backend` and attention still goes
  wherever `engine::attention` decides, but neither whole-half fused
  Vulkan prefill chain is taken, because neither can express this layer:
  `fused_attention_prefill` rotates unconditionally, and
  `fused_post_attention_prefill` takes one epsilon for all three of its
  norms where this needs two. Teaching those two chains a per-layer NoPE
  flag and a second epsilon is the obvious way to put this family back on
  the fused path. Its `mmproj-*.gguf` is a separate `clip` model and is
  not loaded, as for every other family here.

  The vocabulary is worth its own note: `tokenizer.ggml.pre = "llama4"`
  reads like the llama3 pre-tokenizer family and is not one — upstream
  routes it to `LLAMA_VOCAB_PRE_TYPE_GPT4O`, which splits letter runs on
  case boundaries and leaves `ignore_merges` clear. See
  `engine/tokenizer.rs`'s `GPT4O_PRE_TYPES`.

- `inkling.rs` — `inkling` (Inkling), e.g. `unsloth/Inkling-Small-GGUF`.
  The one family here with **no rotation anywhere**, which is also the
  thing most likely to be got wrong by analogy: there is no
  `rope.dimension_count`, no `rope.freq_base`, and nothing in the tensor
  directory that would look wrong if a rotation were applied anyway.
  Position enters twice instead.

  **A learned relative-position bias.** `attn_r` (`[n_embd, n_head *
  d_rel]`) gives each token a `d_rel`-wide coefficient vector per head;
  `attn_rel_proj` (`[d_rel, rel_extent]`) is the per-layer bank they mix,
  producing one additive logit per query/key *distance*. Distances past
  the bank's width take no bias, which is how a 1024-wide bank serves a
  million-token context. The bank is **narrower on the sliding-window
  layers** (`inkling.rel_extent_swa`) than on the full-attention ones
  (`inkling.rel_extent`), and that is used as the load-time cross-check on
  `attention.sliding_window_pattern`: each layer's own `attn_rel_proj`
  shape has to agree with the kind of layer the pattern says it is, so a
  pattern read the wrong way round is a load error instead of a model that
  answers badly.

  **Four causal depthwise short convolutions per layer**
  (`shortconv_k`/`_v`/`_attn`/`_mlp`, width `inkling.shortconv_kernel`),
  each `conv(x) + x` — the residual is inside the operator, and a layer's
  own residual add is a *second*, separate add. The key/value ones run on
  the **raw projections**, before the per-head norm and before anything is
  cached, so the cache holds already-convolved keys. Their rolling windows
  outlive a decode step, so they live in `KvCache::recurrent` and reuse
  `RecurrentLayerState::conv_step` — the primitive `qwen35moe.rs`'s
  linear-attention layers already had. Four slots per layer in a fixed
  order; they are all the same shape and the same type, so swapping two is
  neither a compile error nor a crash.

  Three more details each produce a model that loads, runs, and answers
  fluently while being wrong. The attention scale is `1 / head_dim`, not
  `1 / sqrt(head_dim)` — the per-head query/key RMSNorm accounts for the
  other factor. The routed experts' weights are normalized **together
  with** the shared experts' (the router emits `n_expert +
  n_expert_shared` logits, and the trailing ones gate the shared branch),
  where every other MoE family here normalizes the routed weights among
  themselves. And the full-attention layers multiply every score by
  `1 + log_scaling_alpha * ln((pos + 1) / log_scaling_n_floor)` once the
  position passes the floor — exactly `1.0` below it, so a short prompt
  cannot tell whether it is implemented.

  The tail divides the final hidden state by `inkling.logit_scale_denom`
  before the output projection, and masks the vocabulary's padding rows
  (`inkling.unpadded_vocab_size` real rows out of `vocab_size`) to `-inf`
  rather than truncating the vector, so a caller indexing logits by token
  id still can. Left unmasked, a padding row wins the argmax and decodes
  to nothing.

  CPU-orchestrated like `glm.rs`/`kimi3.rs`: the matmuls (including every
  expert projection, through `arch::mod`'s shared `project_expert` /
  `evaluate_routed_experts` / expert-budget machinery) still dispatch to
  the `Backend`, but the attention loop itself is the host's — the
  relative bias is a per-`(token, head)` additive term and the length
  scaling a per-query multiply, and no attention kernel here takes either.
  `forward_all_logits` is deliberately not implemented, so the opt-in
  prompt-lookup speculative decoder refuses this family rather than
  silently mismatching a convolution window it cannot roll back.

  Its vocabulary is the o200k one: `tokenizer.ggml.pre = "inkling"` is
  routed to the same `GPT4O_PRE_TYPES` arm `muse-glimmer` uses, and the
  `mmproj-*.gguf` shipped beside it is a separate `clip` model that is not
  loaded.

- `phi.rs` — `phi3`, covering both Phi-3 and Phi-4-mini (e.g.
  `unsloth/Phi-4-mini-instruct-GGUF`), confirmed against upstream
  `src/models/phi3.cpp` and the ggml kernels it calls. Llama-shaped
  attention and SwiGLU, but with four details that each silently corrupt
  output if guessed: query/key/value fused into one `attn_qkv.weight`
  sliced Q-then-K-then-V; the FFN gate and up projections fused into one
  `[n_embd, 2*n_ff]` `ffn_up.weight` whose **first** half is the activated
  one; partial NEOX RoPE carrying LongRoPE `rope_factors_long`/
  `rope_factors_short` divisors; and a `rope.scaling.attn_factor`
  magnitude scale on cos/sin. Sliding-window attention is deliberately
  *not* implemented, matching upstream, which disables it for this
  architecture even when the GGUF declares a window.

  Which LongRoPE factor tensor applies is a property of the serving
  context, not of a request: upstream picks `long` when the context
  exceeds `rope.scaling.original_context_length` and `short` otherwise,
  once, so every key already in the KV cache was rotated the same way as
  the query reading it. This server has no separate context knob —
  `engine::generate` caps a sequence at the model's own `n_ctx_train` — so
  `n_ctx_train` is what the comparison uses. For Phi-4-mini (131072
  trained, 4096 original) that selects the long factors, matching
  a server started with `-c 0`; note the common CLI default of `-c 4096` selects
  the short ones instead, so a logit-level comparison has to pass a
  matching `-c`.

### Constrained decoding (`engine::constraint`, `engine::sampling::Constraint`)

`JsonPrefix` is a byte-level recogniser for **prefixes** of valid JSON. At
every point it answers "could this still become valid JSON", which is what a
mask needs, and separately whether what it has *is already* a complete document
(`is_complete`), which is what a stop condition needs. Both are required and
they are different questions: `{"a":` satisfies the first and not the second.

It is a hand-written state machine rather than a parser generator because the
state has to be **cheap to clone** — testing a candidate token is "clone, feed
its bytes, see if it survived", and that happens per candidate per step. The
state is a small enum, a stack of `{`/`[` frames, and two flags.

Two details are easy to get wrong and are pinned by tests, having both been
wrong first:

- **A trailing comma is not an empty container.** `[1,]` and `{"a":1,}` arrive
  at "a value/key is expected" exactly as `[]` and `{}` do, so the state after
  `[` is a distinct `ValueOrClose` from the state after `,`. Collapsing them
  accepts both trailing commas.
- **Object keys and values are both strings.** A closing quote means "expect
  `:`" after a key and "expect `,` or `}`" after a value, so the machine tracks
  which one it opened. Without that, `{"a" "b"}` reads as two values in a row.

`Constraint` joins the grammar to a vocabulary. It holds the tokenizer's
`token_bytes` table — built once, on the first constrained request, and shared
by `Arc`, so a deployment that never constrains anything never pays for it. The
table comes from the same `append_token_bytes` that `Tokenizer::decode` uses,
because a constraint reasoning about text the model is not actually emitting
would be worse than none.

**Where the mask is applied, and what it costs.** On the greedy path the plain
argmax is tested first and returned if it is legal, which it almost always is —
one grammar probe, nothing sorted. Only when the model's preferred token is
illegal is the vocabulary ordered and walked. On the sampled path the mask is
applied after `top_k` and *before* the softmax, so probabilities are
renormalized over the allowed set; if every one of the top `k` is rejected the
field is refilled from the whole vocabulary rather than failing.

**Three things a constrained request gives up**, all because they choose tokens
without consulting the mask: the device-side argmax fast path, the batch
coordinator's greedy sampling, and prompt-lookup speculation. Each is gated on
`Sampler::is_constrained`. A speculative draft accepted on "this is what greedy
would have produced" would sail straight past the grammar.

**Stopping.** End-of-sequence ids are masked until the document is complete,
and once it *is* complete they are the only ids left. The second half is not
symmetry for its own sake: measured without it, a model that had just written
`{}` went on emitting blank lines to `max_tokens`, because whitespace after a
finished document is legal JSON and far more probable to that model than
end-of-sequence.

**Whitespace before the document is deliberately still legal.** Banning
whitespace-only tokens was tried, and it made the output worse rather than
better: with no room to hesitate the model opens `{` on the first step and,
having committed with nothing planned, immediately closes it — `{}` where
allowing it to pause produced `{"Name": "John Doe", "Age": 32}`. The cost is
that a model which refuses to emit JSON at all stalls on whitespace until
`max_tokens`. A constraint can make invalid output unreachable; it cannot make
a model cooperate, and pretending otherwise would trade a visible failure for a
worthless document.

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
plausibly flip this positive on different hardware — but **not at higher
concurrency**, which has now been measured and does not flip it. See below.


**The fused path has been measured at real batch sizes, and it loses.**
`ORANGU_BATCH_DECODE=1` routes concurrent decode steps through
`engine::batch::BatchCoordinator`, which collects whatever arrives inside a
window (`ORANGU_BATCH_WAIT_MS`, 4 ms by default) and runs them as one
`forward_batch_decode`. At 32 concurrent streams:

| window | mean batch | aggregate tok/s |
| ---: | ---: | ---: |
| 4 ms | 2.21 | 65.02 |
| 25 ms | 13.86 | 45.47 |
| 100 ms | 22.55 | 44.67 |
| 400 ms | 26.98 | 42.75 |

The first row is why `/moe-stats` reports `batch.mean_batch`: an earlier
comparison at the default window concluded "batching is 3% slower" when the
mean batch size was 2, so it had measured a rendezvous rather than a batch. At
a mean of 27 the fused path is 34% *slower* than not batching.

Unbatched decode reads ~78 GiB/s of weights here, at the card's ceiling;
batched at mean 27 reads ~1.9 GiB/s and still delivers fewer tokens. Fusing
does remove the weight-bandwidth bottleneck, and what remains — the
per-sequence attention, RoPE and KV write the fused path leaves per-sequence,
plus the window every sequence pays on every token — costs more than it saves.

A rendezvous can only build a batch by making sequences wait, and the waiting
loses at every size tried, so there is no window setting that wins. Closing
this gap means a scheduler that batches whatever is ready on each step without
waiting for stragglers, not a larger constant.

### Serving over TLS (`tls.rs`)

Built in rather than delegated to a reverse proxy, because "one static binary,
nothing to install" is the property this project trades other things for, and
"put nginx in front of it" spends exactly that property in the deployments the
binary exists for — air-gapped, sovereign, one machine, no package manager.
Terminating in front stays valid and is what most fleets will do; it just is
not the only way to reach the server safely over a network.

It cost glue and no cryptography: `rustls` and `tokio-rustls` were already in
the tree for `reqwest`'s HTTPS, so what was added is an acceptor and a PEM
reader — the latter through `rustls_pki_types`' own `PemObject`, already
present as a `rustls` dependency, rather than `rustls-pemfile`, which is the
historical spelling and is now flagged unmaintained.

`TlsListener` implements `axum::serve::Listener` — accept a connection, report
the local address — so it drops into the existing `axum::serve` call and keeps
one serving path, one shutdown `select!`, and the `ConnectInfo<SocketAddr>` the
loopback-only routes rely on. A hand-rolled accept loop feeding `hyper` would
have duplicated all of that to add one wrapper.

**`accept` cannot fail, and that shapes the error handling.** A refused
handshake is not a server error: on any network-reachable port it happens
constantly — scanners, plain HTTP sent to an HTTPS port, clients with no shared
cipher — so the loop drops those and continues. Giving up would turn background
noise into an outage; logging each one would flood the log with it.

**`ConnectInfo` needed an idiom.** `axum` implements
`Connected<IncomingStream<'_, L>>` for `SocketAddr` only for its own
`TcpListener`, plus a blanket impl for anything wrapped in `TapIo`. Writing the
impl here is not allowed — both types are foreign and a local type appearing
only as a type parameter does not satisfy the orphan rule — so
`TlsListener::with_connect_info` wraps in a `TapIo` that does nothing and earns
the blanket impl. Worth the indirection because the alternative was dropping
`ConnectInfo` to make it compile, which would have quietly widened what
`/model-cache/drop` accepts.

Both keys or neither: `tls_cert` alone is a configuration error, because the
alternative is a server that starts in the clear while the config looks like it
does not. Certificate loading happens before anything is served, so a bad file
is a startup failure naming it.

**A note at startup when the server is exposed without either gate.** Binding
off-loopback is deliberate; serving an inference engine there unauthenticated
and in the clear usually is not. The default bind is loopback precisely so that
cannot happen by accident, so the only way to reach that line is to have
widened `host` — at which point the omission is worth naming rather than
leaving to be discovered from outside.

### Authentication (`http::require_api_key`)

A `route_layer` over the whole API router. With no `[orangu-server].api_key`
(or `ORANGU_API_KEY`) configured it passes everything through, which is the
behaviour before it existed and the right default for the loopback address the
server also defaults to.

`Authorization: Bearer <key>` rather than a scheme of this project's own,
because the orangu client **already sent one** — `orangu::llm`'s `bearer_auth`,
from the client's own `api_key` config — and every OpenAI-shaped client sends
one. The server was the only half of this that was missing, so the work was
checking a header that was already arriving, not designing a protocol.

The comparison is constant-time. A `==` on secrets returns at the first
differing byte, so its duration reports how many leading bytes were right —
enough, over many attempts, to recover a key one byte at a time. The cost of
avoiding that is a few nanoseconds on a path about to run a forward pass.

`OPEN_PATHS` is `/health` and nothing else. `/health` names no model, reports
no load, and returns the same bytes to everyone, so requiring a secret for it
buys nothing and costs what every deployment needs: a probe that works from a
load balancer with no credentials.

**`/v1/models` is deliberately closed, and the coordinator was changed rather
than the exemption widened.** Its `ensure_reachable` and startup health check
both probed `/v1/models` and required `is_success()` — so the first server
given an `api_key` would have answered `401`, been read as "stopped
answering", and been restarted on every request, in a loop. Both now accept
**any** HTTP response as proof of life, which is what they were actually
asking: a `401` proves a process is there just as well as a `200`. Conflating
reachability with authorization is the bug; exempting the path would only have
hidden it.

Proxied requests need nothing: `proxy` forwards every header verbatim, so a
client's `Authorization` reaches the backend unchanged.

### Admission: a bounded, first-come-first-served queue

`SlotPool::try_acquire` takes a **ticket** before it waits, and only the holder
of the head ticket may claim a slot. Without that the wait is a scramble: a
release wakes one sleeper, which then rescans, and a request arriving in that
gap can call `try_take_any` first and take the slot it was woken for. Under
steady load that is unbounded waiting for whoever is unlucky, with nothing in
the system that would ever report it.

A global semaphore would be the obvious fix and does not work here, fair though
`tokio`'s is: `acquire_slot` bypasses admission by design — a pinned request is
waiting for one specific slot's warm cache, not competing for one — so a global
permit count drifts out of step with the slots actually free. That is the same
failure this pool's own comment records having removed a global semaphore for
once already.

**The ticket's `Drop` is the load-bearing part.** A waiter can vanish at any
moment: the client hangs up, the request times out, the runtime cancels the
task. A ticket that outlived its holder would leave the head pointing at a
number nobody will ever claim, stalling every request behind it for the life of
the process. Releasing on drop makes cancellation safe by construction rather
than by remembering, and
`an_abandoned_waiter_does_not_stall_the_queue` fails — by hanging — without it.

`[orangu-server].queue_limit` bounds the depth. Past it `try_acquire` returns
`None`, which becomes a `StreamEvent::Overloaded` and then `503` with
`Retry-After`. It is its own event rather than an `Error` because the two mean
different things to a caller: an error is a request that cannot be served, and
this is one that could be served later. `/metrics` exports
`orangu_server_queue_depth` and `orangu_server_queue_limit`, because a bound
nobody can see is half a feature.

**Client disconnects already free their slot, and did before any of this.**
Both the streaming and non-streaming paths consume the same channel, so a
dropped receiver makes the next `tx.send` fail and generation stops at the next
token. Measured: a non-streaming client killed three seconds into a 2000-token
request left the slot `busy: false` one second later, 125 tokens in, and the
next request was served immediately. No cancellation plumbing was needed.

**What this does not do** is stop one long generation from occupying a slot
while short requests wait. That needs preemption, which is a scheduler
question rather than an admission one — see the batching section above for why
the scheduler is the real gap.

### The host-resident bound on prefix reuse

`LayerCache::len` counts every position the cache logically holds. The host
buffers do not always hold all of them: the fused GPU decode path writes a
token's key and value straight into the device mirror and calls
`advance_gpu_only`, which moves `len` without pushing anything host-side. After
N decode steps `len` is N rows ahead of `k`/`v`.

Everything that reads the host side — cross-request prefix reuse, per-slot
retained caches, slot save, CPU attention — therefore has to bound itself by
`KvCache::host_committed_len`, not by `committed_len`. Bounding by the wrong
one is not a subtle inaccuracy: it indexes off the end of the buffer.

That was a live crash, on the commonest usage there is — the same slot, a
growing conversation, a GPU backend. The second turn panicked with `range end
index 31488 out of range for slice of length 30592`: 246 rows claimed against
239 held, the difference being exactly the tokens the first turn generated.
`advance_gpu_only`'s own doc comment had named the hazard in advance ("*If
prompt-prefix reuse (slot save/restore) is ever built, this becomes unsafe*"),
and before the host buffers were sized to their contents it read zeros instead
of panicking, which is worse.

Two details of the bound are load-bearing. It is **derived** from the buffer
length rather than tracked in a second field, so it cannot drift from what it
describes. And it takes the shortest layer that holds *anything*, skipping
permanently-empty ones: `engine::arch::gemma`'s cross-layer KV donor is empty
by design, and a plain minimum would read it as "no prefix is reusable" and
switch reuse off for the whole architecture.

What this costs is reuse of the *generated tail*, which the next turn
re-prefills. Measured on a four-turn conversation: 695 of 923 tokens reused,
the shortfall being the eight tokens the previous turn generated.

`engine::slot_store` also stopped locking its cells with `lock().unwrap()`.
One panic under that lock poisoned it, and every later request on every slot
then panicked on the `unwrap` rather than being served — a process that stayed
alive and answered `500` to everything. It is a cache: a poisoned entry is at
worst stale, the caller already handles absence, so it recovers the guard and
reads as a miss.

### Latency histograms and readiness (`engine::metrics`)

`/metrics` was three gauges and a limit. Gauges answer "what is happening right
now", which is the wrong question for latency — the useful questions are all
about the tail, and a mean is dominated by whichever requests happened to be
long. Four Prometheus histograms answer them instead: queue wait, time to first
token, inter-token, and request duration.

**Bucket counts are per-bucket and made cumulative only at render.** A
cumulative *write* would mean touching every bucket at or above the observed
value on every observation, and the inter-token histogram is observed once per
generated token. The format requires cumulative buckets, so the conversion has
to happen somewhere; doing it on the read side costs a scrape and not a decode
step.

**Sums are microseconds in a `u64`, not a float.** `AtomicU64` is lock-free
everywhere this runs, while a float sum accumulated by compare-and-swap would
be slower *and* non-deterministic in its last digits across runs — which makes
two measurements of the same workload disagree for no reason.

**Two bucket sets, because one cannot span both.** A request is milliseconds to
a minute; a decode step is milliseconds to a second. Putting inter-token gaps
on the request bounds lands almost every observation in one bucket, which is a
histogram that reports nothing.

**Time to first token counts from arrival**, so it carries the queue wait and
the prefill together — what an interactive caller actually waits through. It is
the first token *produced*, not the first the client sees: a chat format's
structural prefix is filtered out of the stream (`MessageHeader`), and charging
the template's shape to the server's latency would make two models of the same
speed report different numbers.

**Outcomes are recorded by a `Drop` guard, not at each exit.** `run` has five
places that report an error and return, one that returns on a client
disconnect, and a `catch_unwind` above it that turns a panic into a reply. A
counter with a path that forgets to increment it is worse than no counter,
because the total silently stops matching the request count. `OutcomeGuard`
defaults to `error` and is told otherwise on the paths that know better, so a
failure path added later is counted correctly without anyone noticing it needed
to be. A refusal is counted through its own entry point rather than as a
zero-duration request: folding a handful of microseconds into the latency
histogram would drag every quantile down exactly when the server is under the
load those quantiles are being watched for.

**Verified by arithmetic rather than by looking plausible.** One 24-token
request produced one time-to-first-token observation and twenty-three
inter-token observations, and the two sums added to the request duration
exactly (1.724549 + 1.157920 = 2.882525 s). Parts that add up to the whole is
the check worth having, because every individual number here looks reasonable
whether or not it is right.

**`/ready` is not `/health`, and the split is the point.** `/health` asks "is
this process alive" — a supervisor's question, and one that must stay `200`
while the server is merely busy, because restarting a loaded server is the
worst possible response to load. `/ready` asks "would a request sent now be
served" — a load balancer's question, where busy is exactly the case worth
reporting. It is `503` when the admission queue is already at `queue_limit`
(the request would be refused anyway, so saying so saves a round trip) or when
the GPU device has been lost. The decision is a free function taking
`(device_lost, queued, limit)` rather than reading `AppState`, so the rule can
be tested at all — the alternative is standing up a loaded model to assert five
comparisons, which is why rules like this usually go untested.

`/ready` joins `/health` in `OPEN_PATHS`, which is a deliberate widening: it
does disclose load, where `/health` discloses nothing. That is the one fact the
probe exists to report, it is bounded (depths and slot counts, no model name
and no request content), and anyone able to send a request learns the same
thing from the `503` they would get instead.

**What this immediately found, and the care the finding needed.**
`outcome="cancelled"` sat at zero under a test that should have produced one,
which looked like the disconnect abort not working at all. It was not that. The
abort rode on *sending a token* — `tx.send(…).is_err()` — so it only ran for a
token that produced visible text, and the model under test rendered one
character for forty tokens. Almost every iteration skipped the check entirely.

That is a real hole rather than a quirk of one model: a chat format's
structural markers are suppressed, and under `--review` a whole reasoning body
is, so a request could generate to `max_tokens` — holding the slot every other
request is queued for — for a client that hung up at the first token.

The check is now `tx.is_closed()`, asked every token whether or not there was
anything to send. Measured on the model that leaked: a 2000-token request whose
client sent a TCP reset at four seconds used to run to completion and finish as
`length`; it now stops within about four seconds and finishes as `cancelled`.
On a model with ordinary visible output the behaviour was already correct, and
still is — which is why the first reading of the zero counter, that disconnect
abort did not work at all, was itself wrong.

### Draft-model speculative decoding

`[orangu-server].draft_model` puts a second, smaller model in front of the
served one: it proposes `draft_tokens` continuations, the target verifies all
of them in a single multi-position forward, and the longest prefix the target
would itself have produced is kept. Everything after that prefix is thrown
away, so the emitted text is byte-for-byte what greedy decoding alone would
have emitted — a drafter can only change how fast an answer arrives.

**The verification machinery already existed** for prompt-lookup drafting, and
the two now differ only in where the candidate tokens come from. That is the
whole abstraction: one enum with a `draft` and a `commit`, and the source with
no model behind it simply has nothing to commit. Their cost profiles are
opposite, which is why both are kept — prompt lookup produces nothing unless
the context repeats itself but its misses are free, while a draft model always
produces a draft and always pays a forward pass per token for it.

**Both models need `forward_all_logits`, and the reason is not the obvious
one.** The target needs it because verification *is* a multi-position forward.
The draft needs it because it is the only entry point that keeps the KV rows on
the host: a single-token `forward` takes the fused GPU decode path, which
writes key and value straight into the device mirror and leaves the host rows
unpopulated. `KvCache::advance_gpu_only`'s own doc comment had predicted the
consequence in as many words — "a resumed cache could need this position's real
data for a later multi-token prefill's CPU attention path, which would silently
read zeros instead" — and a draft cache is exactly that: rolled back to the
accepted prefix and re-read on every step. Running the draft through `forward`
produced both failures the warning describes, one loud (a read one row past the
end of the host buffer) and one silent (garbage where zeros were read). Both
models are probed with a one-token forward at startup rather than checked
against a list of architectures, because the method is defaulted and there is
nothing to ask a type about.

**The draft's cache follows the target's exactly.** After verification the
target truncates to `start_pos + 1 + accepted`, and the draft truncates to the
same count; when everything was accepted the draft is left one token short and
the next step's catch-up fills it in. Getting this wrong has no visible symptom
— the target re-derives every token either way — beyond an acceptance rate
quietly falling to zero, which is why `commit` is asserted directly rather than
through output.

**Two guards worth naming.** The draft is capped by the room left in the KV
cache, because a verification forward appends the whole draft at once and the
GPU mirror is sized to the request's capacity — a bound only reachable in the
last few tokens of a full context, which is why it survived unnoticed on the
prompt-lookup path. And the pair's vocabularies are compared *by token string*,
not by size: two tokenizers of the same size whose contents diverge is exactly
what a same-family model at a different scale can be, and speculation compares
token ids.

**Measured, and it lost.** On a 4 GiB card serving a target that overflows it
(`gemma-4-12B-it:Q4_K_M`, 1.43 tok/s unassisted), a `gemma-4-E4B` draft
achieved a *better* acceptance rate than prompt lookup — 2.15 accepted per
verification against 1.67 — and still ran at 1.02 tok/s where prompt lookup ran
at 3.01. Drafting deeper made it worse in exactly the way that identifies the
cost: eight tokens per step raised acceptance to 2.56 and dropped throughput to
0.67, so the cost is per drafted token, not per step. Each draft forward is a
pass through a second set of weights competing for device memory the target
already overflows. The condition for a draft model to pay is therefore not
about prediction quality at all — it is that the draft must be small enough not
to disturb the target's residency.

*Not built:* the DFlash sidecar format's own draft-and-verify loop. A DFlash
draft is not a standalone model — it reads the target's hidden states at named
layers (`dflash.target_layers`) and fuses them through its own `fc` — so
running one needs `ModelForward` to expose intermediate hidden states, which
would be an interface change landing on every architecture at once for one
sidecar format. `engine::arch::dflash` still resolves such a file to its paired
target and serves that. Sampled (non-greedy) acceptance is also not built: it
needs the draft's own probability distribution and a rejection-and-resample
rule, which is a different algorithm from "did the draft match the argmax",
not a relaxation of it.

### Durable slot persistence (`engine::slot_store`)

`orangu-server` implements the `POST /slots/{id_slot}?action=save|restore`
endpoints — its equivalent of the `--slot-save-path` prompt-cache
save/restore, and the receiving end of the orangu client's per-session slot
persistence (`orangu::llm::SlotRegistry`, driven from tab park/activate).

The important structural difference from that design is that an orangu-server
slot is a *concurrency permit* (`SlotPool`), not a long-lived owner of one KV
cache. A completed request's cache otherwise survives only inside the in-RAM
`engine::prefix_cache` pool (opt-in, bounded, cross-slot). `engine::slot_store`
adds the durable layer: each slot retains the `(tokens, KvCache)` of the last
request that ran on it, and

- `save` serializes that snapshot to
  `~/.orangu/server/<fingerprint>/slots/<filename>` — only the committed KV
  positions (not the whole allocated context window), written atomically
  (temp file + rename);
- `restore` loads it back into the slot's retained cache, so that slot's
  *next* request reuses the prefix through the same `KvCache::copy_prefix_from`
  path (and the same `CachedPrefill::reusable_prefix_len` committed-length and
  recurrent-state rules) as the prefix pool, instead of re-prefilling.

Prefix reuse from the cross-request pool **moves** the source's buffers rather
than copying them. `PrefixCache::take_best_match` removes the entry it returns,
so the request owns it outright and nothing else can still be reading it;
`KvCache::adopt_prefix` trims the source to the reused length and takes its
layer buffers, keeping this request's own capacity, `kv_dim` and stride. On a
2001-token conversational prefix that is **0.26 ms against 18.1 ms** for the
copy it replaced, and it removes the transient doubling of the prefix's
resident footprint at the moment the source was about to be dropped. The host
buffers grow with the committed length instead of being pre-filled to the
context ceiling, which is what makes the move possible.

`engine::slot_store` still copies, and the asymmetry is deliberate: it retains
its snapshot for the same slot's *next* request, so it does not own the source
and cannot move out of it. Both paths go on sharing
`CachedPrefill::reusable_prefix_len`, so the matching rules stay
single-sourced even though the transfer differs.

A committed length is a **token** count, and on a block-compressed slot that is
not its row count — one row stands for `stride` tokens. `KvCache::committed_len`
converts through the stride, and `reusable_prefix_len` calls it rather than
recomputing the maximum inline, which is what it used to do. Both readings gave
the same answer on every architecture built so far, because `deepseek4` is the
only one that strides and it always carries ordinary per-token slots that are
longer — a coincidence of the current models rather than a rule, and exactly
the sort that a change to how rows are allocated would remove silently.

This is what survives the cases the RAM pool cannot: eviction under cache
pressure, a server restart, and — most relevant behind `orangu-coordinator` —
a model swap that tears the server down. The client saves a tab's slot
*before* the coordinator activates the next tab's model, so the save reaches
the still-active server and lands on disk; the later restore repopulates a
freshly (re)started server.

`<fingerprint>` is a SHA-256 of the architecture, the model label, and the KV
structure tag (layer count, per-layer `kv_dim` **and stride**, recurrent
specs). The stride was missing until it was audited for: two caches differing
only in how many token positions one row stands for laid their rows out
differently and shared a signature, which the model label hashed alongside
happened to mask — a signature that is right only because something else is
also checked is not one worth relying on. A snapshot
saved for one model therefore resolves to a different directory than any other
model's, and every file also carries the fingerprint internally — a mismatched
or corrupt file is treated as "nothing to restore" (`n_restored: 0`, a normal
prefill next request), never a hard error, so a stale sidecar never trips the
client's fallback notice. Client-supplied filenames are validated to a single
safe path component before touching the filesystem.

The feature is **on by default**; set `ORANGU_NO_SLOT_SAVE` to disable it (it
also stays off when `$HOME` can't be resolved). While off, the endpoints report
"not supported" exactly as a server started without `--slot-save-path`
does — which the orangu client already degrades against, falling back to a full
reprefill. The opt-out exists for the same reason `ORANGU_PREFIX_CACHE` is
itself opt-in — a bug in prefix reuse would produce a silently *wrong*
generation, not merely a slow one — but persistence is only ever exercised when
a client explicitly saves or restores a slot, so it stays dormant unless used.

Requests routed through `orangu-coordinator` carry the session's `model` in
the slots request body (the orangu client adds it), so the coordinator proxies
each save/restore to that model's backing server rather than its default
profile; a direct orangu-server or plain OpenAI-compatible server ignores the
extra field.

Saved files accumulate one directory per distinct model. `orangu-server prune`
sweeps slot files untouched for over 30 days on every run
(`slot_store::sweep_stale_slot_files`), alongside its empty-session sweep — see
[Session management](46-server.md).

### GPU backend architecture

`engine::backend::Backend` (`backend/mod.rs`) is the trait every backend
implements — `matmul`/`matmul_batch` plus a downcast hook (`as_wgpu`) the
model forward pass uses to reach `VulkanBackend`'s much larger fused
surface when it's the active backend. Six implementors exist:
`CpuBackend` (scalar with runtime AVX2 dispatch via `engine::tensor::dot`,
parallelized across output rows with `rayon`; always available, and the
fallback when no GPU backend is found), `VulkanBackend`, `MetalBackend`,
`CudaBackend`, `OpenClBackend`, and `RocmBackend`.

The hook is named for `wgpu` rather than for Vulkan because two backends
answer `Some` to it: `VulkanBackend` and `MetalBackend`, which is that
same engine on another `wgpu` API (see **The Metal backend** below). Every
fused path reached through it is therefore live on Apple GPUs too.

`main.rs`'s `select_backend` implements the `backend = auto` cascade:
Vulkan, then CUDA, then OpenCL, then ROCm (if built with the `rocm`
feature), falling back to `CpuBackend` if none of them initialize. On
Apple targets the cascade starts with Metal instead — not a preference
but a cost: macOS ships no Vulkan driver, so leading with Vulkan there is
four retry rounds of guaranteed failure before reaching the API the
machine actually has, and `MetalBackend` gives up nothing, being the same
kernels. On Windows, DX12 sits *behind* Vulkan and ahead of CUDA/OpenCL —
also the same `wgpu` engine and the same WGSL (via `naga`'s HLSL output),
so it reaches every fused path the matmul-only backends do not, but behind
Vulkan because that is the API this engine was tuned on. An
explicit `backend = <name>` instead brings up that one backend and fails
to start if it can't, rather than falling back
— useful when GPU inference was asked for specifically and a silent
CPU fallback would be the wrong failure mode.

### Device selection

`backend` picks the API; `engine::backend::device` picks the device within
it. Every GPU backend exposes the same two entry points — `devices() ->
Vec<DeviceCandidate>` (enumeration only, no device creation, so it is safe
and cheap on a machine with no driver) and `try_init_index(index)` /
`try_init_selected(&[index])` — and
`select_backend` drives both through the one shared policy, resolving
`--device` first, then `ORANGU_DEVICE`, then `[orangu-server].device`
(`requested_device`).

**Selection returns a set, not a device.** `device::select_all` answers
with every candidate the request admits, best first: under `auto` that is
the whole ranked hardware list, and under an index or a name it is exactly
one — which is what makes a named device *exclusive* rather than merely
preferred. The head runs the model; the tail is carried into
`VulkanBackend::device_selection` purely so the startup inventory and
`/props` can report it. The distinction is load-bearing before any
placement pass exists: "orangu chose this card out of three" and "orangu
was told to use this card and nothing else" are different runs, and only
the second one stays correct when a second card appears in the machine.

`DeviceRole` is what the inventory reports per device — `InUse`, `Idle`
(selected but not running the model), or `Excluded` with a reason. An idle
device is named as idle rather than as "available": a second card sitting
unused beside a slow first one is a question worth answering on the same
screen as the number that prompted it.

The enumeration itself goes through the same short retry as bring-up
(`devices_with_retry`). A driver whose previous context has just been torn
down can briefly report *no* adapters, and `request_adapter` used to hide
that inside the retried call by both choosing and creating in one step —
splitting the two apart would otherwise have turned a restart race into
"no device was found" on a machine that has one.

The policy ranks by class, then by size: discrete > unclassified >
virtual > integrated, with software rasterizers never selected
automatically (orangu's own `CpuBackend` is faster than llvmpipe, and a
software adapter reporting itself as a GPU run is worse than useless).
Class beats size deliberately — an iGPU reports the machine's whole system
RAM as its memory and would otherwise win on a laptop. Unknown size ranks
last *within* a class but never demotes a device out of it: "unknown" is
not "zero", which is `llama.cpp`'s own rule for its `--fit` accounting.

What this replaced was `request_adapter(PowerPreference::HighPerformance)`
— a *hint*, answered by the loader, and routinely answered with the
integrated GPU on a machine that also has a card. `llama-server` has the
same trap and the same fix (`--device Vulkan1`).

Three properties are worth keeping if this code is touched:

- **The inventory is printed unconditionally**, on every start, marking
  the device in use. It is what makes a measurement attributable, and a
  diagnostic that has to be enabled before the run it describes is not
  one. It also goes into `/props` next to the tuning report.
- **A named device that isn't there is an error listing the ones that
  are** — never a fall-back. This holds under `backend = auto` too: a
  backend can only be chosen by satisfying the request, and a request no
  backend in the cascade could satisfy stops the server at the end of the
  chain instead of dropping to the CPU. An A/B between two cards is
  worthless if one run quietly measured the other card.
- **An ambiguous name is rejected**, not resolved by rank. Two identical
  cards is precisely the case where picking one silently destroys the
  comparison being attempted.

`VRAM` comes from `vulkan_replay::adapter_device_local_bytes`, which
reaches through `Adapter::as_hal` to `vkGetPhysicalDeviceMemoryProperties`
— `wgpu` has no memory query on any backend. It reports heap *size*, not
`VK_EXT_memory_budget`'s free figure: ranking wants a property of the card,
and a card that happens to be driving a compositor must not be demoted
below an iGPU for it. On non-Vulkan APIs it answers `None`, which the
policy already handles.

Each backend's `try_init()` (no index) still exists but is now `#[cfg(test)]`:
tests want a device and don't care which, while the server proper always
goes through enumerate → select → report → `try_init_selected`.

### The KV mirror grows with the sequence

`LayerCache::sync_gpu` allocates the GPU-side mirror for the rows in use,
doubling from a 256-row floor and capped at the layer's capacity. It used to
allocate the whole capacity — which is `prompt + max_tokens` — on first use, so
a request that asked for a large budget reserved it in VRAM whether or not it
generated anything. On a 3.98 GiB card, the same two-token prompt and one-word
answer took 2191 MiB at `max_tokens = 64` and **3727 MiB** at 32768; it is now
flat at ~2185 MiB across both.

Host buffers never had this problem and were not changed: a large zeroed `Vec`
is `mmap`ed, and the kernel commits pages only as they are written — two
gigabytes of `vec![0.0f32; ..]` adds no resident memory at all. Device memory
is not overcommitted, which is why the same over-reservation that costs nothing
on the host costs the whole card on the GPU.

Growth copies the rows already on the device across with
`copy_buffer_to_buffer` rather than re-uploading them, so a doubling costs
device-local bandwidth instead of putting the whole cache back over the bus;
`synced_len` carries over unchanged. Cached attention bind groups name the old
buffer, so they are dropped on growth and rebuilt through the caller's existing
cache-miss path.

**The mirror is sized to `len + 1`, and that is load-bearing.** The fused
decode path binds these buffers and then writes the current token's key and
value at row `len`, *before* the host-side `push` that commits it. Sizing to
exactly `len` puts that write one row past the end of the k region — and
because k and v are two sub-ranges of a single buffer, it lands on row 0 of v
rather than outside the allocation, so no validation fires and the corruption
is silent. Reserving the full capacity hid this indefinitely; it appeared one
growth step after demand-sizing landed, as an overrun on the grow-copy.

### Device footprint

`engine::footprint::DeviceFootprint` measures the loaded model against the
chosen device, at startup, and prints it under the inventory.

- **Weights** come from `LoadedModel::resident_tensor_sizes` split by
  `engine::backend::device_resident_split`, which reuses the same
  `is_cpu_only_tensor` predicate the startup type check uses. That is what
  keeps a MoE model's routed experts — which have no GPU path at all —
  from being charged against VRAM: a 20.6 GiB Qwen3.6-35B-A3B reports
  2.26 GiB on device and 18.35 GiB in host memory.
- `resident_tensor_sizes` rather than `tensor_sizes`, and the difference is
  the trailing multi-token-prediction block. Its tensors are mapped like any
  other and the forward pass never reaches one, so counting them inflated
  every device figure by the draft head: `Qwen3.8-27B-Q6_K` reported
  `weights 21.30 GiB on device` where `plan` — which had excluded the draft
  head from the start — said 21.0 GiB, and the two now agree at 20.97 GiB.
  `device_resident_split`'s own doc comment justified the loose bound with
  "every layer of a served model is reached by the first token", which is
  true of every layer except this one. `tensor_sizes` is unchanged and still
  reports what the *file* holds; memory questions ask the other one.
- **KV** comes from `KvCache::gpu_mirror_bytes`, called on a
  `new_kv_cache(1)` probe. The per-layer `kv_dim`/`stride` is fixed model
  geometry, so a one-token cache — which `main` already builds for the
  slot-persistence fingerprint — carries the whole shape, and the sizing
  then scales it to a context far too large to allocate. Recurrent layers
  are excluded: they have no GPU mirror.
- `kv_cache::gpu_layer_bytes` sits directly above `GpuLayerCache::new` and
  has to agree with it; `kv_mirror_bytes_agree_with_the_allocation` pins
  the arithmetic to literals rather than restating the formula, so a
  mistake in the original can't be copied into the test that guards it.

**It reports, it does not refuse.** Weights reach the device lazily and are
never evicted, the KV cache is sized per request rather than at the context
limit, and the arenas grow to the widest prefill — so "does it fit" is not
decidable at startup, while headroom and what that headroom buys are. A
model whose weights exceed VRAM gets a warning naming the shortfall and
keeps running: the driver pages, which is slow rather than broken, and
refusing would convert working configurations into failures. Resist adding
a verdict here.

### Splitting a model across devices

`engine::placement` decides which device holds which layer;
`engine::backend::multi::MultiDeviceBackend` makes it happen. The whole
design rests on one observation: **a `QuantMatrix` can carry its own
device**.

`LoadedModel::matrix` is the single place every architecture obtains a
weight, and it knows the tensor's name — so it stamps each matrix with the
device its `blk.<n>.` layer was placed on (`LoadedModel::layer_device`, set
by `main` between loading the weights and building the model, because
building the model is what calls `matrix`). `MultiDeviceBackend::matmul`
then reads `w.device()` and forwards. **Not one line of any forward pass
changes**, and there are eleven of them.

The cross-device transfer falls out of the same shape rather than being
written: `Backend::matmul` already takes host `&[f32]` and returns host
`Vec<f32>`, so a layer on device 0 ends with its output in host memory and
the next layer's first matmul uploads it to device 1. There is no
peer-to-peer path and no residual to shuttle by hand.

**Two hooks, and the line between them is layer scope.**

`Backend::as_wgpu_on(device)` is for work scoped to one layer: fused
attention, the fused post-attention/FFN chain, the device-side KV mirror.
Each takes host input, returns host output, and touches only that layer's
weights and cache, so it runs happily on whichever card the layer is on.
`device` is always read off a weight the call is about
(`QuantMatrix::device`) rather than tracked separately — one map, living on
the weights, so it cannot disagree with where `matmul` sends the same
layer's operands.

`Backend::as_wgpu()` still answers `None` on a split, and now means
something narrower: work that *spans* layers. The whole-step decode
submission (~37 submissions down to 1), GPU sampling, the logits readback.
Those assume one device holds the whole chain.
`multi::tests::a_split_model_never_exposes_a_wgpu_backend` and
`per_layer_work_asks_the_layer_s_own_device` hold both halves.

**What that `None` costs the reporting, and what is done about it.** Three
things hang off `as_wgpu()` that have nothing to do with running the model:
the kernel/tuning report, the device footprint, and `/gpu-timings`. A split
therefore used to lose all three at once, silently — `/props` carried a
placement plan in the `gpu` slot with no tuning fields, the footprint was
never measured, and the timings endpoint answered `enabled: false` in the same
words it uses for "you did not switch timestamps on".

Two of the three are now answered per device instead of not at all:

- **Footprint.** `SplitReport` carries each device's capacity *and* its
  `KvStorage` out of `apply_device_split`, captured while the concrete
  backends are still in hand, and `DeviceFootprint::for_split_device` builds
  one footprint per device: the weights the plan placed there
  (`weights_per_device`, which asks `LoadedModel::device_for_tensor` rather
  than re-deriving the placement) and the KV mirror for *that device's own
  layers* (`KvCache::gpu_mirror_bytes_where`). Per layer, not per layer
  *count*: `kv_dim` and stride vary down a model's depth, so a device holding
  a quarter of the layers can hold half the cache. Host-resident weights
  (routed experts) are charged to no device and stated once at the top of the
  plan JSON — charging them per card would count them once per card.
- **Timings** stay single-device (a query set belongs to one device), but the
  endpoint now names the reason: `unavailable: "split"` against
  `"no_wgpu_backend"`, read off the `split` flag in the tuning report so there
  is one source of truth for "was this run split".

`orangu-bench` renders both, and the last of the three — the kernel report —
is printed as explicitly absent rather than omitted.

The KV mirror is safe by construction: a layer's device never changes,
`sync_gpu` is only ever called from inside a `VulkanBackend` (so the mirror
lands on that backend's own device), and `LayerCache::copy_prefix_from`
drops the mirror, so no buffer survives into a cache reused elsewhere.

Measured on the dev machine, release build, a 0.5B model split 3:1 over two
GPUs: **11.9 tok/s** with per-layer fusion off, **14.9** with it on,
against **27.8** unsplit. Per-layer fusion is worth about a quarter.
`ORANGU_NO_SPLIT_FUSION=1` is what makes that A/B possible from one binary
— measuring it by building a second binary is how a stale copy ends up
being the thing timed.

**The whole-layer decode chain, per device run.** A split used to lose
`record_fused_layer` for the *whole* model, not just at boundaries: the
recorder is reached through `as_wgpu()`, which answers `None`. Every layer
therefore fell to the step-by-step path, which round-trips through host
memory between individual ops.

`LlamaModel::record_split_decode` restores it. `record_decode_run` takes a
**layer range**, a host input vector, and whether to append the tail; a
single-device model is one run over every layer with the tail — the same
code that ran before it took a range — and a split model is one run per
device, with the hidden state crossing to host in between
(`VulkanBackend::submit_and_read_at`). The vocab projection runs where
`output_weight` is, which is device 0, so a model whose last layers are
elsewhere pays one more hand-off.

Two measurements decided that design, and the first killed the obvious
alternative:

- **Submission count is not a cost here.** On one device,
  `ORANGU_DECODE_CHUNKS=24` (one submission per layer) measured *faster*
  than the default single submission — 47.6 against 41.3 tok/s — because
  early chunks execute while the CPU records later ones. So there is no
  reason to keep one encoder alive across a device switch, which is the
  hard part of that design and is now simply not needed.
- **It was the kernel, not the second device.** Before the fix, moving a
  *single* layer to the iGPU dropped decode from 32.2 to 13.9 tok/s while
  moving twelve dropped it only to 16.1 — a shape no "second card is
  slower" explanation produces.

Result, one batch, 0.5B model, 3:1 split:

| | decode |
| :-- | --: |
| unsplit | 41.8 tok/s |
| split, all split GPU work on | **21.2 tok/s** |
| split, all off (`ORANGU_NO_SPLIT_FUSION=1`) | 12.1 tok/s |

**+75%.** And the diagnostic that exposed the bug is monotonic again: one
iGPU layer now costs 26.4 tok/s against 41.4 unsplit, twelve cost 19.1 —
consistent with a slower second device and a per-boundary hand-off, which
is what is left and is inherent.

**llama, phi and mistral** all have it, each verified live on a split with
byte-identical output to the same run with the paths disabled:

| family | split, on | split, off |
| :-- | --: | --: |
| llama/qwen2 (0.5B Q4_K_M) | 21.2 tok/s | 12.1 |
| phi (Phi-4-mini Q4_K_M) | 14.0 tok/s | 9.5 |
| mistral (Ministral-3B IQ3_XXS) | 5.0 tok/s | 2.9 |

Consistently +47% to +75%. gemma is verified for correctness on a split
(`gemma-4-12B` across both cards) rather than for speed — 36 of its 48
layers land on a 4 GiB card, so that run is dominated by driver paging and
says nothing about the chain. The one piece shared across the three is
`arch::decode_device_runs` — the grouping of consecutive layers by device,
and the rule that any layer without a GPU behind it declines the whole
chain (which is how a CPU overflow tier opts out). It is shared precisely
because its failure mode is silent: a mis-grouped run would record a
layer's fused chain against the wrong card's weights.

**gemma has it too**, with one exclusion. Its recorder takes an encoder
from the caller — the cross-sequence batched path shares one across
sequences — so the range and `with_tail` slot in without disturbing that,
and `record_split_decode` brings its own encoder per run.

The exclusion is **per-layer embeddings**. `record_one_sequence_decode`
projects the token embedding once into a `[n_layer, per_layer]` `ple_buf`
that every layer reads a slice of, and that buffer belongs to one device.
Worse, a later run's `x` is a mid-model hidden state, not the token
embedding, so recomputing it per run needs the original vector threaded
through as well. A model with PLE therefore declines the chain and takes
the step-by-step path; that is gemma-3n (`gemma-4-E2B` and relatives).
Dense gemma-4 has `per_layer == 0` and is unaffected. `has_ple` also
requires `layers.start == 0`, so the guard is in the code as well as in the
caller.

(Absolute numbers move between measurement batches with GPU clock state.
Only compare within one batch.)

Points worth preserving if this is extended:

- **Contiguous runs, never interleaved.** Crossings per token equal the
  number of boundaries, and interleaving would make it one per layer.
- **Shares follow reported memory**, which on an iGPU is the whole of
  system RAM — so `all` will over-weight the slower device on a
  dGPU+iGPU box. Explicit ratios exist for that; inventing a correction
  factor would be a guess.
- **One API per split.** Every device comes from one backend's own
  enumeration, so a model can never be spread across two vendors' kernels,
  whose different accumulation orders would make output depend on which
  layers landed where.
- **Non-`wgpu` backends refuse** rather than silently running on one
  device. `CudaBackend`/`OpenClBackend`/`RocmBackend` are matmul-only and
  unverified against real hardware; an untested multi-device path there
  would be worse than none.
- **Device loss policy is unchanged**: any device's loss is `device_lost::
  fail`'s exit 75 and a coordinator restart. A partial-device server is not
  a state worth supporting.

### The CPU as a device

`SplitMode::Cpu` appends `CpuBackend` to the device set as the last entry
and lets `placement` place layers on it. Nothing else was needed:
`MultiDeviceBackend` holds `Arc<dyn Backend>`, and the CPU is one.

Three things make it a *fill* (`placement::fill_in_order`) rather than a
share, and they are the parts worth keeping:

- **The host's budget is system RAM**, so a proportional share would hand
  it most of the model. It gets only what the devices could not hold.
- **The first device is charged for the non-layer tensors** — token
  embeddings, output norm, `lm_head` — because `device_for_tensor` puts
  them there. A live Kimi-K3 fill put 4.96 GiB of weights on a card
  budgeted for 3.20 GiB before that subtraction existed.
- **A plan that puts every layer on the host is still returned**, not
  discarded as "not a split": the embeddings stay on device 0 either way,
  and dropping the plan would hand the whole model back to the GPU, which
  is the paging this mode exists to avoid, reached by asking for the
  opposite. A test covers exactly that.

`WEIGHTS_SHARE_OF_DEVICE` (0.8) is the only invented constant in the
device work. It cannot be computed: the KV geometry needs a built model,
and the model cannot be built until placement is decided, because building
it is what stamps each tensor's device. Explicit ratios are the escape
hatch, and the footprint report says afterwards what the choice left.

`project_expert` reads the quantized bytes **directly**, through the same
`engine::vecdot` integer-dot kernels `CpuBackend` uses for every dense
matmul: `dot_row`/`dot_k_row` when one token is routed to an expert, and
`unpack_row` once plus `dot_unpacked_multi` when several are. Only the types
`vecdot` has no unpacking for fall back to dequantizing a row to `f32` and
dotting it there.

That was worth finding. A routed expert's rows are read once and thrown
away, so the `f32` row was pure overhead — materialized, dotted, discarded —
and on a `nemotron_h_moe` decode profile `quant::dequantize_into` under this
one function was **60% of all CPU time in the process**. It is the same
shape for every mixture-of-experts family here, since they all share this
path. Note what it also does to the arithmetic: an expert now quantizes its
activations exactly as the dense path always has, so an expert's numerics
are *consistent with* the rest of its layer rather than more precise than
it.

`configure_cpu_threads` sizes rayon's *global* pool once, before anything
parallel runs — `CpuBackend`'s matmul, `project_expert`, and the
per-expert fan-out all share it, so the knob belongs there rather than on
any one of them. Unset leaves rayon's own default, so a config that says
nothing keeps the behaviour it had. `0` is rejected: rayon reads
`num_threads(0)` as "the default", which would make a typo silently mean
the opposite of what it looks like.

NUMA and P-core/E-core awareness stay out of scope. That is a decision
rather than an oversight: rayon's pool is unaware of core topology, and
pinning workers to a socket or to P-cores only pays once there is a
measurement saying the default placement is the bottleneck.

### Device expert tiers — the seam, and why it stops there

Routed experts are host-resident (`is_cpu_only_tensor`), and a hot subset
lives in owned RAM under `engine::expert_store`'s budget, with an LRU/LFRU
policy and a learned-heat sidecar that survives a restart. A *device* tier
would be the same idea in VRAM. Two pieces of it exist:

- `ExpertQuantMatrix::expert_matrix(e)` — one expert as an ordinary
  `QuantMatrix`, zero-copy. This is the piece that makes a device expert
  tier a **dispatch** problem rather than a kernel one: every GPU backend
  already has a kernel for every quant type an expert is stored in,
  `Backend::matmul` takes a `QuantMatrix`, and `MultiDeviceBackend` already
  routes one by the device stamped on it. Unused outside its tests, and
  marked `#[allow(dead_code)]` rather than deleted, because the two tests
  over it (byte-identical against `row()`, distinct cache keys per expert)
  are what make the first dispatch a small change.
- `engine::expert_tier` — the placement policy: whole experts, hottest
  first, fastest device first, all three borrowed from colibri. Its
  `coverage()` is the number the whole decision turns on, and its
  `projection()` is printed at startup for a MoE model on a GPU.

**The dispatch now exists**, behind `ORANGU_GPU_EXPERTS=1` and off by
default. `arch::gpu_project_expert` views one expert as a `QuantMatrix`
(`ExpertQuantMatrix::expert_matrix`) and hands it to `Backend::matmul` —
**no new kernel**, which is what the seam was for: every GPU backend
already has one for every quantization an expert is stored in. It declines
to the host path when the backend has no GPU or no kernel for that type
(the `IQ*` types large MoE models often ship in are exactly the gap).

It is a **measurement knob before it is a feature**, and it does no
residency management at all: every expert it touches lands in the weight
arena, which never evicts. Point it at a device that cannot hold them and
the number it produces is driver paging.

What it exists to answer is R2's own premise — whether a GPU expert matmul
beats `engine::vecdot`'s tuned AVX2/rayon path *at all*, given that this
dispatch is a blocking submit-and-readback per (expert, projection, layer).

**Measured, and batching is what decides it.** Qwen3.6-35B-A3B with its
dense part on the iGPU, page cache warm, same prompt throughout:

| routed experts on | decode |
| :-- | --: |
| CPU (`engine::vecdot`, AVX2 + rayon) | 2.12–2.14 tok/s |
| GPU, one dispatch per expert | 1.39–1.42 tok/s |
| GPU, batched across experts | 2.67–3.71, settling ~3.3 tok/s |

One dispatch per expert *loses* to the host by 1.5×; batching them —
`arch::evaluate_routed_experts_batched`, one `matmul_batch` per (layer,
projection) rather than one blocking submit-and-readback per expert — turns
that into a 1.55× win. `matmul_batch` requires a uniform token count, so
ops bucket by `n_tokens`; at decode that is one bucket.

Three traps in measuring this, all hit while doing so:

- **Both sides must be warm.** Cold, the same run is 0.09 tok/s prefill and
  0.22 decode. That number is storage, not matmul.
- **The number rises across a run** as experts land in the weight arena and
  stop being read from the page cache. Take the settled value, not the
  first.
- **A different prompt routes a different expert set** and gives a
  different number. Compare only within one prompt.

**The tier is bounded.** `main::plan_expert_tier` chooses the resident set
up front from `expert_tier::plan` — half the device's free memory after the
dense weights — and stamps it per expert
(`ExpertQuantMatrix::is_device_resident`). Non-resident experts stay on the
host path through `engine::expert_store` as always, so the batch never
pulls an expert into an arena that cannot evict it. On the model above that
is 15978 of 30720 experts in a fixed 9.40 GiB, measuring 2.68–2.74 tok/s:
some of the unbounded win given back, because 48% of routed experts fall
back, in exchange for a tier that does not grow until the device is full.

Five MoE architectures take the batched path — `qwen35moe`, `qwen3next`,
`glm`, `deepseek4`, `kimi3` — each supplying its own activation closure.
gemma's MoE does not: it projects a *fused* `gate_up` tensor by row range
rather than separate gate/up tensors, so it needs a variant of the helper.

The resident set is filled from the routing profile
(`ORANGU_EXPERT_USAGE`, `expert_store::learned_heat`) when one exists and
by size otherwise; the startup line says which, because that choice is most
of what the tier is worth — colibri measured the same tier 3–5× apart
depending on it. Every number above is the *by size* floor.

The batched path bypasses `engine::expert_store`'s residency tier, which is
correct for weights in VRAM and wrong for the host path, so non-resident
experts keep `project_expert`.

**What is still not built is the tier**, and the projection is why. On this project's dev machine a 20.6 GiB MoE model on a
4 GiB card leaves room for 5% of the experts — a tier that cannot pay for
itself whatever the kernels look like. Three things would have to be true
before it should be:

1. **Enough coverage to matter**, measured on the target machine, and
   **`ORANGU_GPU_EXPERTS=1` beating the host path** on it. If the naive
   dispatch loses badly there is still hope in step 2; if it loses by an
   order of magnitude, there is not.
2. **Batched dispatch.** `project_expert` is called per expert per
   projection per layer, and the knob above issues one blocking
   submit-and-readback for each. The shape that works is one
   `matmul_batch` per (layer, projection) over every routed expert —
   `MatmulOp` already carries a per-op `x`, so the operands fit, but
   `evaluate_routed_experts` would have to be restructured into
   gather-then-batch **without disturbing its bit-identical accumulation
   order**, across six architectures.
3. **A bounded residency.** `VulkanBackend::weight_buffer`'s arena never
   evicts, so experts reaching it on demand grow without limit. Placement
   has to choose the resident set up front, from the profile — which is
   exactly what `expert_tier::plan` returns, and which
   `ORANGU_GPU_EXPERTS` deliberately skips.

The honest prior is not favourable: colibri's own finding is that a GPU
expert tier "earns its VRAM only when the CPU is the weak link", and
orangu's host expert path is tuned AVX2 over rayon.

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
step. Together these collapse the number of GPU submissions per decode
token from one per matmul/op down to a small constant (as low as one for a
fully-fused Gemma decode step), removing the per-submission submit/poll/
readback latency that otherwise dominates a many-layer forward pass. With
round trips largely eliminated, the remaining cost is per-kernel compute
and weight-memory bandwidth, which the alternative decode kernels below
target.

#### Vulkan backend environment variables

The Vulkan backend reads these environment variables at startup to select
between alternative compute kernels. Each is read once when the backend
initializes; changing one takes effect on the next server start. All are
correctness-verified against `CpuBackend`.

**Boolean flags read `0`, `false`, `no`, `off` and the empty string as OFF**,
and anything else as on — `engine::env::flag_on`, which every one of them goes
through. That matters because these knobs exist to be swept: they were
previously read for *presence*, so `FLAG=0` switched the feature **on** and a
sweep of `0,1` ran it on both arms, reporting the difference between a thing
and itself. The handful of variables that carry a *value* rather than a
boolean — `ORANGU_NORM_WG`, `ORANGU_COOP_GEOM`, `ORANGU_DUMP_SHADERS`,
`ORANGU_EXPERT_USAGE`, `ORANGU_PREFIX_CACHE_DIR` — are still presence-checked
or parsed, and are noted as such where they appear.

| Variable | Default | Effect |
| :-- | :-- | :-- |
| `ORANGU_PREFILL_BATCH` | `512` (integer, not a presence flag) | **Ceiling** on how many prompt tokens go into one forward pass; the actual width is chosen per chunk by `ORANGU_PREFILL_CHUNK_MS` below and never exceeds this. `0` disables chunking entirely — the whole prompt in one submission, which is what prefill did unconditionally before this existed and which loses the device on any long prompt. Measured on a 4 GiB RX 5500M holding 2.5 GiB of weights, with a 17.5k-token prompt — no chunking: device lost after 21s; `2048`: device lost after 3m54s; `512`: completed in 4m13s, with peak VRAM identical (3.67 GiB) in all three. Not a throughput tax: at 8k tokens on the same card `512` prefilled at 115.4 tok/s against 105.5 unchunked, since a smaller working set pages less. |
| `ORANGU_PREFILL_CHUNK_MS` | `3000` (integer) | Wall-clock target for one prefill submission. A chunk is timed and the next is scaled by the rate just measured, because cost per token climbs with context: a fixed 512-token chunk measured 2.3 s at position 512 and **10.1 s at position 6 656**, past the ~10 s `amdgpu` allows before it resets the device (see *Losing the GPU device*). A token-count limit alone therefore stops protecting anything on a long prompt. With this, a 48 000-token prompt that used to reset the device at position 7 680 completes with a slowest submission of 3.3 s. Lower it if resets still happen; the default leaves room for the estimate to lag a rate that only rises. |
| `ORANGU_NO_MLP_UNROLL` | unset (block-unroll **on**) | Set to **disable** the block-unroll reduce kernel for K-quant (`Q4_K`/`Q5_K`/`Q6_K`) decode and fall back to the scalar per-element reduce kernel. The block-unroll iterates whole super-blocks, loading each block header once and issuing several weight/activation loads before the dependent dot; it is the default decode path. |
| `ORANGU_NO_BLOCK_HOISTED` | unset (block-hoisted **on**) | Set to **disable** the block-hoisted decode kernel and fall back to the scalar per-element `reduce` for every type that has one. The block-unroll above is fast and covers only `Q4_K`/`Q5_K`/`Q6_K` (it hardcodes their 256-element super-block as a fixed 4x64 geometry); the element-wise `reduce` covers everything and re-decodes each block's header once *per element*, which for a 32-element block is 32 times over. The block-hoisted kernel is the third algorithm: several adjacent lanes share one block and take contiguous byte slices of it, so the header is amortized while the coalesced access shape is kept. It covers every remaining quantized type — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` — and is the control for any A/B of it. Term-for-term identical arithmetic to the element-wise path but a different summation order, so output is not bit-identical and a greedy argmax can flip on a near-tie. Ask a running server which kernel a type resolves to: `/props` -> `gpu.kernels.decode`. |
| `ORANGU_NO_DUAL_NIBBLE` | unset (dual **on** for `Q4_K` and `Q6_K` decode) | Set to **disable** the dual decode kernels for both `Q4_K` and `Q6_K` and fall back to the two-wave block-unroll. Each two-wave kernel splits a 64-thread workgroup into two halves that re-read shared weight bytes — `Q4_K` streams every qs byte twice (once per nibble half), `Q6_K` re-reads every `qh` byte (once per `w_lo` half). The dual kernels use a 32-thread (single-subgroup) workgroup that loads each such byte once, cutting decode GPU-execution time (~22% for `Q4_K`, a further ~4–8% for `Q6_K`) with identical greedy output. They reorder the per-lane float adds, so they cross-check against the CPU backend within a tolerance rather than bit-for-bit. No effect on `Q4_K` when `ORANGU_PACKED_DOT=1`, or on either when `ORANGU_NO_MLP_UNROLL=1`. |
| `ORANGU_NO_Q6K_DUAL` | unset (`Q6_K` dual **on**) | Set to **disable** only the `Q6_K` dual kernel (reverting `Q6_K` tensors — e.g. `ffn_down` — to the two-wave block-unroll) while leaving the `Q4_K` dual kernel on. For A/B isolation of the `Q6_K` kernel; `ORANGU_NO_DUAL_NIBBLE=1` disables both. |
| `ORANGU_Q4K_CONTIG` | unset (off) | Set to select an alternative `Q4_K` decode kernel with a contiguous thread→element mapping: each lane loads a `u32` of four consecutive qs bytes (all used) plus two `vec4<f32>` activations, ~3× fewer VMEM load instructions than the default dual kernel. Correctness-verified (byte-identical greedy output) but measured **no faster** on this hardware — the `Q4_K` matmul is memory-latency-bound rather than load-issue-bound — so it is off by default. Kept for other GPUs where issue rate may bind. |
| `ORANGU_REDUCE_N_ROWS` | `2` (integer, not a presence flag) | Output rows one decode matmul-vec workgroup computes (reusing each activation element across all of them). Lower values launch more, smaller workgroups — more independent wavefronts in flight per compute unit to hide VRAM latency — at the cost of re-reading each activation in more workgroups. Clamped `1..=16`. Was `4` historically; re-swept to `2` after the dual-nibble kernel and chunked submission made GPU occupancy (not CPU submission cost) the decode critical path. Applies to every K-quant reduce/block-unroll kernel and its dispatch-count math together. |
| `ORANGU_NORM_WG` | `128` (must be `64`, `128`, or `256`) | Workgroup size (thread count) of the default tree-reduce RMSNorm and RMSNorm+residual-add kernels. These run one `dispatch_workgroups(1,1,1)` workgroup over the whole `n_embd` row, so they are occupancy-starved; more threads shorten each thread's grid-stride loop and light up more of one work-group processor's SIMDs. Raised from `64` to `128` after measurement (halving to `32` had previously *doubled* the time, so these are compute/load-bound, not launch-bound): decode GPU-execution time dropped ~11% (35.4 → 31.5 ms/token) on real `E2B`/`RX 5500M` with byte-identical output; `256` was no better than `128` (deeper reduction tree, more barriers). Only affects the default (non-subgroup) norm path. |
| `ORANGU_BUSY_POLL` | unset (off) | Spin-poll the decode-path GPU readback instead of blocking on it. The blocking wait parks the decode thread while the GPU runs (~30 ms/token), so its CPU core can drop clock or be migrated off — leaving the next token's recording/submission to start on a cold core. Spinning keeps that core at its boost clock and returns within microseconds of the GPU finishing rather than after a scheduler wake-up. Measured **+8% decode throughput** (27.4 → 29.6 tok/s) on real `E2B`/`RX 5500M`, byte-identical output. Trades one busy-spun core (power) for latency — recommended for a dedicated single-stream inference server, less so under many concurrent slots or on a shared machine. |
| `ORANGU_ATTN_SPLIT_K` | `8` (must be a power of two, `1..=32`) | Split-k factor for decode attention: how many workgroups each query head's KV-position range is split across (`n_head × k_num` phase-1 workgroups, merged by a phase-2 pass). Each head attends a KV range that *grows with context*, so more splits expose more parallelism the longer a generation runs, at the cost of more phase-2 merge overhead when the range is short. Raised from `4` to `8` after re-sweeping in the full decode chain at real context lengths (the earlier sweep was on the isolated dispatch at short context): on real `E2B`/`RX 5500M`, `k_num=8` cut per-token GPU time at ~245 tokens of context from 35.6 to 32.4 ms while staying neutral at short context, and had the best end-to-end throughput of `{4,8,16}`; `16` wins only once context is very long. Byte-identical output across values. Workloads dominated by very long contexts may prefer `16`. Pure runtime uniform — no shader rebuild. |
| `ORANGU_PACKED_DOT` | unset (off) | Dequantizes `Q4_K` weight elements in pairs and accumulates the dot product as `vec2<f16>` instead of two scalar `f32` multiplies. Requires an adapter with WGSL `f16` support. When set together with the block-unroll, selects the combined unroll+packed `Q4_K` decode kernel. |
| `ORANGU_WIDE_LOAD` | unset (off) | Binds the weight buffer as `array<vec4<u32>>` (16-byte reads) instead of `array<u32>` (byte-wise reads), consolidating each `Q4_K`/`Q5_K` block header into one 16-byte read. Covers all supported quant types. |
| `ORANGU_NO_KV_F16` | unset (`f16` **on** when the adapter supports it) | `1` **disables** storing the per-request KV-cache GPU mirror as `f16` and fall back to `f32`. `f16` (the default on an adapter with WGSL `f16` support) halves KV-read memory traffic per attention dispatch, with a per-write cast, and matches the ecosystem's default KV cache type. |
| `ORANGU_KV_Q8_0` | unset (off) | `1` stores the per-request KV-cache GPU mirror as `q8_0` (8-bit block-quantized) instead of `f16`, dequantized inline in the attention shader. Halves KV-read bytes again vs `f16`, directly cutting attention's cost at long context — measured **−32%** attention GPU time at ~295 tokens on real `E2B`/`RX 5500M` (5.87 → 3.99 ms), the saving growing with context, at a slight cost at short context (per-write quantize overhead). Takes precedence over `f16`. **Lossy** (unlike `f16`), so off by default; the recommended lever for long-context / long-generation workloads, where per-token decode slows as the KV cache grows. Superseded by `[orangu-server].kv_cache`, which names every value rather than one; kept because existing repro lines and sweeps use it. |
| `ORANGU_KV_CACHE` | unset (the config file's `kv_cache`) | `f16`, `q8_0` or `f32` — the same three values `[orangu-server].kv_cache` takes, overriding it for one run. The variable a sweep should use: it is the only one that can name all three arms, so a sweep does not have to express "not the other one". |
| `ORANGU_NO_TILED_PREFILL` | unset (tiled prefill **on**) | Set to **disable** the `16×64`-output-tile GEMM for prefill (`n_tokens >= 64`) and fall back to the plain cooperative kernel (one workgroup per output row, looping over the whole prompt internally) — measured on real hardware to drive real requests into GPU-driver hangs at ordinary prompt lengths (~170-450 tokens) and, even where both complete, ~10x slower. Not recommended; kept for A/B comparison. |
| `ORANGU_COOP_MIN_TOKENS` | `64` (integer, `>= 1`) | The token count at or above which a matmul takes the weight-amortizing tiled-GEMM path instead of the reduce family. The reduce kernels dispatch `n_row_groups × n_tokens` workgroups, so **each token re-streams the whole weight matrix** (their cost grows with `n_tokens`); the tiled kernel loads a `16×in_dim` weight tile into shared memory once and reuses it across a `64`-wide token tile (cost ~flat in `n_tokens`). Below the threshold the per-token re-stream is cheaper because it keeps far more wavefronts in flight to hide this GPU's weight-load latency. Lowering it to route the "missing middle" (K ≈ 2…63, e.g. speculative or small-chunk prefill) onto the tiled path was measured **slower on `RX 5500M`** (this matmul is latency-bound, not bandwidth-bound — see `SERVER_ROADMAP.md` Step 13), so the default keeps that split at `64`; exposed as a knob for GPUs where the crossover sits lower and as the A/B harness for a future small-K kernel. `1` forces every matmul tiled; a value past the longest batch keeps everything on the reduce path. |
| `ORANGU_NO_GPU_SAMPLE` | unset (GPU sampling **on**) | Set to **disable** running greedy (temperature-0) argmax sampling with repeat penalty on the GPU in the same submission as the forward pass (reading back one token id instead of the full `[n_vocab]` logits vector) and fall back to a CPU-side readback + sample. |
| `ORANGU_DECODE_CHUNKS` | `7` (integer, not a presence flag) | How many `queue.submit()` calls one decode step's layer loop is split across. `1` records the whole token and submits once (the historical behaviour); `> 1` submits the first `chunks - 1` groups of layers as soon as they are recorded, so the GPU starts executing them while the CPU is still recording and validating the later ones — overlapping the CPU-side submission cost with GPU execution instead of serialising it. Clamped to `1..=n_layers`. On real `E2B`/`RX 5500M` this raised decode throughput from 14.4 tok/s (`1`) to 18.8 tok/s (`7`, the default), with byte-identical output; `35` (one submit per layer) reaches 19.4 tok/s but adds per-submission overhead for a marginal gain. |
| `ORANGU_BATCH_DECODE` | unset (off) | Fuses the matmul steps of concurrent requests that submit a decode step within a short window into one batched call (attention/RoPE/KV-write stay per-sequence). Only takes effect when `slots > 1`. |
| `ORANGU_BATCH_WAIT_MS` | `4` | How long a fused-decode collection window stays open. Only meaningful with `ORANGU_BATCH_DECODE`. Raising it is the only way batches actually form — at 32 concurrent streams the 4 ms default gives a mean batch of 2.2 and 400 ms gives 27 — but aggregate throughput falls monotonically as it rises (65 → 42.75 tok/s), because every sequence pays the wait on every token. The mean batch size achieved is reported as `batch.mean_batch` on `/moe-stats`; a batching measurement cannot be read without it. |
| `ORANGU_PREFILL_FUSED_ATTN` | unset (**off**) | Set to **enable** running a prefill layer's whole pre-attention half as one submission — the Q/K/V projections, the per-head Q/K norms, RoPE, V's weightless norm, the KV-cache write for the whole batch, and attention itself — and fall back to the step-by-step path (projections read back to the CPU, norms and RoPE on the CPU, a per-token cache push, then a separate attention dispatch). The fused path keeps everything between the projections and attention in GPU memory, so nothing but attention's output and the K/V rows for the host mirror crosses the bus. Correctness-verified against the step-by-step path, but currently **slower**: re-measured against correct output it loses ~10% at a 158-token prompt and is a wash at 1120. Off until the transfers it removes pay for the work it adds. Automatically declined (with the same fallback) under `ORANGU_Q4K_MMVQ` and for a non-causal batch longer than one submission chunk. |
| `ORANGU_PREFILL_ATTN` | unset (off) | Set to run the **standalone** prefill attention dispatch where the fused path above declines, instead of the CPU attention loop. Unlike the fused path this pays a Q upload and an attention-output readback per layer, which the CPU loop does not, so it wins only at long prompts. A measurement aid rather than a recommended setting. |
| `ORANGU_NO_PREFILL_GQA` | unset (GQA sharing **on**) | Set to **disable** sharing each KV head's reads across the query heads that use it in the prefill attention kernel, falling back to one workgroup per `(head, query)`. Only affects models whose `n_head` exceeds `n_head_kv`. |
| `ORANGU_GQA_HEADS` | unset (chosen by register budget) | Pins how many query heads of one KV group a prefill attention workgroup owns. Must divide the group size. Sharing a KV read across more heads and keeping enough waves resident to hide that read pull in opposite directions; the default picks from inside the measured band. A tuning knob for other GPUs. |
| `ORANGU_GPU_TRACE` | unset (off) | Logs the number of GPU submissions per decode step to stdout — a diagnostic for round-trip counting, no effect on the computation. |
| `ORANGU_DUMP_SHADERS` | unset (a directory path) | Set to a directory to write the generated WGSL of the decode-path kernels (Q4_K/Q6_K matmul-vec, RMSNorm, split-attention) into it as `.wgsl` files at startup, then continue normally. A profiling aid: pair it with the driver's own `RADV_DEBUG=shaders,shaderstats` (which dumps the compiled ACO ISA and register/occupancy stats for this GPU), or hand the WGSL to an offline analyzer such as Radeon GPU Analyzer on an RDNA3+ machine. No effect on the computation. |
| `ORANGU_SPECULATIVE` | unset (off) | Enables prompt-lookup speculative decoding: each step drafts the next few tokens by matching recent output against an earlier point in the context and verifies the whole draft in one forward, so the weights stream once for several tokens. Greedy-only (the output is identical to non-speculative greedy decoding); ignored for non-greedy sampling and for multi-slot batched decode. **Currently slower on this GPU** — see `SERVER_ROADMAP.md` Step 12 — because the multi-token verify runs the CPU-orchestrated forward; kept for hardware/paths where a resident multi-position forward makes it a win. Off by default. |
| `ORANGU_SPEC_NGRAM` | `2` | With `ORANGU_SPECULATIVE`, how many trailing tokens must match an earlier point in the context to trigger a draft. Lower drafts more often (more speculative work, more misses); higher drafts only on a longer exact echo. |
| `ORANGU_SPEC_DRAFT` | `4` | How many tokens to draft (and verify in one forward) once a match is found. The ceiling on tokens a single accepted step can produce. Also overrides `[orangu-server].draft_tokens` for a run using a draft *model*, so a sweep of drafting depth reads the same whichever drafter is in play. |
| `ORANGU_GPU_TIMESTAMPS` | unset (off) | Logs a per-decode-step GPU timing breakdown to stderr — the per-layer-embedding (PLE) projection, the sum/average/slowest across all model layers, and the output-norm-plus-`lm_head` tail, in milliseconds. Also logs a `[gpu-op-breakdown]` line splitting each token into **qkv-side** (Q/K/V matmuls + norm/RoPE + KV write), **attention** (split-k), and **ffn-side** (wo/gate/up/down matmuls + their norms + GELU/mul + PLE + copies) — so matmul vs attention vs overhead is measured, not estimated. Requires an adapter with `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS`; a diagnostic, no effect on the computation. **Unavailable on a split model**: a query set belongs to one device and a split resolves none, so `GET /gpu-timings` answers `{"enabled": false, "timings": null, "unavailable": "split"}` — the reason is named rather than left as an empty result, since a client that reports nothing when it receives nothing makes a split run look like one whose GPU stages cost nothing. `--flamegraph` still profiles the CPU side. |

Shader compilation is cached to disk across restarts
(`~/.orangu/server/<adapter-key>/cache.bin`, keyed by a vendor/device-
derived string so a cache built for one GPU is never handed to another) —
a startup-time optimization only, with no effect on decode/prefill
throughput once running.

### The Metal backend

`engine::backend::metal::MetalBackend` is the section above, on Apple
hardware. It is not a reimplementation of anything: nothing in
`VulkanBackend` is Vulkan-specific. Every compute pipeline, every kernel
in `vulkan_shaders`, the weight/op/uniform arenas, the fused decode and
prefill submissions, split-k attention and GPU sampling are written
against portable `wgpu` and WGSL, and only ever ran on Vulkan because
`VulkanBackend::try_init` asked `wgpu` for a Vulkan adapter and nothing
else.

So `try_init` now delegates to `try_init_backends(wgpu::Backends)`, and
`MetalBackend` calls that with `METAL` and wraps the result. `naga`
translates the same WGSL to MSL rather than SPIR-V. `MetalBackend::
as_wgpu` returns the inner engine, so every fused submission
`engine::arch` reaches for through that hook runs here too — the point of
the newtype is to keep the two distinct at the type level while sharing
all of the code.

Bring-up is entirely feature-negotiated in `try_init_backends`, so the
Metal module has no adapter logic of its own. On Apple silicon
`SHADER_F16` (so the `f16` KV cache is on), `SUBGROUP` (so the
cooperative-reduction attention kernel is on) and `TIMESTAMP_QUERY` (so
`ORANGU_GPU_TIMESTAMPS` works) are all present. `PIPELINE_CACHE` is not:
`wgpu::util::pipeline_cache_key` returns `None` for anything but Vulkan,
so no on-disk pipeline cache is built and cold start recompiles the
kernels, which Metal's own shader cache largely absorbs.

One kernel needs a different *form* on Metal, and it is a correctness
difference rather than a tuning one. The tiled prefill GEMM is the only
kernel that fills shared memory by having many threads each write a single
dynamically-indexed **component** of a shared `vec4` (`store_w`/`store_x`
into `tile_w`/`tile_x`; four different threads write the four components of
each vector). On Vulkan/RADV that lowers to a 4-byte store, which is what
lets the read side take four values per load. On Metal it does not: a probe
kernel on CI's Apple Paravirtual device read back
`[1, 0, 0, 0, 5, 0, 0, 0, …]` — exactly one surviving component per vector,
the signature of a read-modify-write of the whole 16 bytes with the four
writing threads clobbering one another. Every tiled-path cross-check
disagreed with the CPU backend there, including plain `f32` with no
dequantization in play, while every scalar-shared-memory kernel passed.

`vulkan_shaders::coop_vec4_tiles` therefore enables `vec4` tiles only where
they are known-correct, and everything else gets a scalar tile: four loads
where the `vec4` form takes one, same layout, same arithmetic, same results.
An unrecognized backend defaults to the safe form on purpose. Whole-`vec4`
stores would also have worked and are the faster-looking fix, but the tiled
kernel cannot use them without changing its tile layout — `store_w`'s four
consecutive slots are four consecutive *rows* at one `k`, while the fill
deliberately gives one thread `RUN` consecutive `k` of one row so a
quantized block's scale and min hoist once per run.
`tests::coop_tiles_are_vec4_only_where_component_stores_work` asserts the
invariant, and its whole-`vec4` twin is the control that pins the fault to
the component store rather than to shared memory or the barrier.

Two further things genuinely are Vulkan-only, and both check the API rather
than assuming:

- `ORANGU_Q4K_GLSL`, a glslc-compiled Q4_K GEMV handed to the driver
  through shader passthrough. `PASSTHROUGH_SHADERS` *is* advertised by
  wgpu's Metal backend, where it means **MSL** passthrough — so gating on
  the feature bit alone would have handed a Metal driver a SPIR-V blob.
  `try_init_backends` gates it on `wgpu::Backend::Vulkan` instead.
- The raw-Vulkan replay path (`ORANGU_REPLAY`, `vulkan_replay`), which
  reaches through `as_hal` for an `ash::Device`. It was already compiled
  out on Apple targets; setting the variable on macOS is a no-op, not a
  failure.

Correctness rests on the same tests as the Vulkan path, not on a separate
suite: `vulkan::shared_test_backend` asks for whichever `wgpu` API the
platform has, so the per-`ggml_type` cross-checks against `CpuBackend` —
ordinary tests, not `#[ignore]`d ones — run against a real Metal device on
CI's macOS runner. Two `#[ignore]`d real-model tests
(`gemma4_predicts_paris_after_capital_of_france_metal`,
`forward_batch_decode_matches_independent_forward_calls_metal`) add the
whole-model claim on top, and CI fails rather than passes if either skips
for want of an adapter.

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

Their `ggml_type` coverage is a subset of what `engine::quant` reads on the
CPU, and each backend's `SUPPORTED_TYPES` is the authority: the float types,
the legacy quants, `Q2_K` through `Q6_K`, and `IQ4_NL`. The remaining `IQ*`
types are absent because each indexes a lattice codebook that would need its
own uploaded buffer — `VulkanBackend` has one (`vulkan_shaders`'s
`IQ_GRID_PRELUDE`, bound at `@binding(4)`) and these three do not. `IQ4_NL`
is the exception that fits: a 16-entry level table, small enough to inline
into the kernel source, and the reason a `Q2_K` download of a model whose
rows aren't 256-divisible (see the Scope section of the server chapter) is
runnable here at all.

`VulkanBackend`'s own coverage is a subset too — it has no shader for the
three narrow `IQ1_*` types (ids 64-66). Every GPU backend therefore overrides
`Backend::supports_type`, and `engine::backend::unsupported_tensor_types`
walks every shard's tensor directory once at startup so a gap is reported as
an error naming each missing type. Before that check existed, the gap
surfaced as a panic from inside `matmul` partway through the first request.

That list used to include `IQ1_S`, `IQ1_M` and `IQ2_XXS`, and dropping them
from it is what makes a "dynamic" 2-bit release runnable on a GPU at all:
`unsloth/Qwen3.8-27B-GGUF:IQ2_XXS` is 96 `IQ1_M` tensors and 48 `IQ2_XXS`
ones, so the startup check refused the device for the whole model rather than
for those tensors. `iq_grid_words` now uploads `IQ2XXS_GRID` and `IQ1S_GRID`
alongside the four it already carried, taking the codebook buffer from ~15
KiB to ~33 KiB — `IQ1S_GRID` is 2048 eight-byte lattice points, 16 KiB by
itself. The `iq1*` pair needed one thing the others did not: their codebook
values are **signed** and they carry no sign field at all, the per-group
freedom being whether a `±0.125` delta is added or subtracted, so
`IQ_GRID_PRELUDE` gained `iq_grid8_signed` beside `iq_grid8`. Reading an
`iq1*` grid byte as unsigned is well formed and produces plausible output,
which is why `matmul_matches_cpu_backend_for_iq1_m` exists rather than being
folded into a generic sweep.

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

### Losing the GPU device (`device_lost.rs`)

Every `wgpu` readback in `VulkanBackend` funnels its failure paths through
one place, `crate::device_lost::fail`, which records the loss, writes the
real detail to the server's log, arms a `75`/`EX_TEMPFAIL` exit two seconds
out, and panics to unwind the request that was in flight. Three things it
replaced are worth naming, because each was a separate way the old code
made a driver reset worse than it had to be:

- **`map_async` callbacks called `.expect()` in place.** `wgpu` runs that
  callback from inside `poll` — or, on the failure path a lost device takes,
  synchronously from inside `map_async` itself — while `wgpu-core` holds its
  own locks, so the unwind went straight through them. The callbacks now
  only record two bits (`MapWait`), and the waiter reports from its own
  frame, with a stack that names which readback was in flight.
- **A failed poll, a failed map, and the `READBACK_WAIT_TIMEOUT` deadline
  each panicked with their own wording.** All three are the same event seen
  from three angles; all three now report it as one.
- **Two of the callbacks discarded their result entirely** (the striped
  matmul and the batch readback's `|_| {}`), so a device that died partway
  through a set of buffers was noticed only later, by whatever read the
  unmapped memory next. Each set now shares one `MapWait` and is checked
  once after its single poll.

One funnel is not enough on its own, though, because `wgpu` does not always
hand a lost device back as an `Err` at all: `Device::poll` routes it through
`handle_error_fatal`, which **panics from inside `wgpu`** (`Error in
Device::poll: Validation Error / Caused by: Parent device is lost`), so the
`Result` the engine checks never arrives. Every `wgpu` call made after the
device dies ends that way. `panic_capture`'s hook therefore reads every
panic's message and calls `device_lost::note_panic`, which marks the loss and
arms the same exit without panicking again — process-wide, so a `wgpu` panic
on a thread nothing catches (a rayon worker) still takes the process down
cleanly instead of leaving it up with a dead GPU. The message match is on the
condition (`device is lost`, `DeviceLost`), not on any one call's name.

`engine::generate`'s `catch_unwind` recognizes the loss (`device_lost::
is_lost`) and swaps the panic's captured detail — meaningless to a caller,
since the backtrace describes the driver rather than their request — for
`device_lost::CLIENT_MESSAGE` (`panic_report`). Requests that arrive in the
window before the process exits get that same sentence without being
started at all.

The exit code is the contract with `orangu-coordinator`, which names the
same number (`process::SERVER_EXIT_DEVICE_LOST`) so it can report the
restart as the recovery it is instead of an unexplained crash. Nothing
about the mechanism is Vulkan-specific: `MetalBackend` is the same engine,
so a lost Metal device takes the identical path.

### Correctness testing

`VulkanBackend`'s dequant math (each quant type, bit-for-bit against the
CPU backend, across both dispatch paths), fused post-attention chain
(including a dedicated test that calls it twice for one layer with
different inputs each time, to catch cache-reuse bugs specifically), and
fused attention (including GQA head-grouping, sliding-window attention,
proportional RoPE, and Gemma4's cross-layer KV-donor case — two different
layers sharing one KV cache) are covered by cross-check tests in
`engine::backend::vulkan::tests`, run on real hardware whenever
it's present and skipped otherwise. Those same tests are the Metal
backend's tests: the device they run against comes from
`shared_test_backend`, which asks for whichever `wgpu` API the platform
has, so they are a Vulkan cross-check on Linux and a Metal one on macOS.
The CUDA/OpenCL/ROCm backends follow
the same skip-if-no-device pattern.

A second set of tests runs a full forward pass against a real downloaded
model and is marked `#[ignore]` so the normal suite doesn't require one.
These read the model path from an environment variable, and each panics
with a clear message if its variable is unset when the test is run
(`cargo test -- --ignored`):

| Variable | Used by | Points to |
| :-- | :-- | :-- |
| `ORANGU_TEST_MODEL` | Gemma/qwen35moe/qwen35 real-model forward-pass tests | A local `.gguf` chat model file |
| `ORANGU_TEST_EMBEDDING_MODEL` | embedding-model tests | A local `.gguf` embedding model file |
| `ORANGU_TEST_QWEN3VL_MODEL` | qwen3vl tokenizer/embedding tests | A local qwen3vl `.gguf` file |
| `ORANGU_TEST_LLAMA_MODEL` | `llama`-architecture forward-pass test | A local Llama-3.x Instruct `.gguf` file |
| `ORANGU_TEST_MISTRAL_MODEL` | `mistral3` forward-pass test | A local Ministral-3 `.gguf` file |
| `ORANGU_TEST_PHI_MODEL` | phi3 real-model forward-pass test | A local Phi-3/Phi-4-mini `.gguf` file |

### HTTP layer and web UI

`http::mod` assembles the router and shared `AppState` (model, scheduler
handle, config, workspace root, start time); `http::openai` and
`http::native` hold the OpenAI-compatible and native handlers respectively;
`http::files` holds the file-lifecycle API (see the next section);
`/v1/shutdown` lives in `http::mod` itself since it's neither. Ctrl+C,
`SIGINT`, and `POST /v1/shutdown` all converge on the same shutdown path via
`tokio::select!`, mirroring `orangu-coordinator`'s own pattern.

`web::mod` serves a small server-rendered chat UI (vanilla HTML/CSS/JS, no
build step) on its own `[web].port`, sharing the same in-process `Engine` as
the API so a chat turn never makes an HTTP hop. `web::render` renders
markdown to HTML (including syntax-highlighted code blocks) with the same
`markdown`/`syntect` crates `orangu`'s terminal UI uses. `web::mermaid`
draws ```` ```mermaid ```` blocks (below). `web::sessions`
persists each chat as `~/.orangu/server/sessions/<uuid>/chat.json`.
`web::models` is the model manager (below).

### Mermaid diagrams (`web::mermaid`)

Diagrams are rendered by `merman`, a headless Rust implementation of
Mermaid that parses, lays out, and emits SVG without Node, Puppeteer, or a
JavaScript runtime — the same requirement that made KaTeX a vendored asset
rather than a CDN link. It is the opposite arrangement to the math path:
`$...$` ships raw TeX to the browser because no server-side TeX engine
exists in Rust, whereas a diagram is finished server-side and the client
only ever receives a picture.

Three decisions carry the design, each forced by something measured rather
than assumed:

**The SVG is embedded as an `<img>` data URI, never inlined.** merman
strips scripts and event-handler attributes from diagram labels, but it
does not escape a literal `</svg>` inside one: a label of `A["</svg>…"]`
emits that tag raw. Inlined, the HTML parser reads it as the real end tag,
closes the diagram early, and lets the rest of the label escape into the
transcript. An `<img>` makes the SVG a separate document — scripts inert,
`id`s unable to collide with the page or with a second diagram on it — and
keeps the same escape-everything stance `web::render` takes for every
other node kind. Base64 rather than percent-encoding, because an SVG is
full of `#`, `<`, `"` and `&`, and one missed escape silently truncates
the image.

**Labels are SVG `<text>`, not `<foreignObject>`.** Mermaid's usual HTML
labels get no layout engine inside an `<img>` document and would render as
nothing at all; `HostThemeOutput::resvg_safe_editor` is what converts
them, and is therefore not optional. A test asserts no `foreignObject`
survives, since dropping that setting fails silently as blank labels.

**Each diagram is rendered twice, once per theme.** An `<img>` cannot
inherit the page's CSS variables, so the palette is baked in and `app.css`
shows whichever of the two finished pictures matches the current theme.
The theme roles map onto the console's own custom properties, so a diagram
is styled like the transcript around it rather than arriving in Mermaid's
stock lavender.

##### Sizing, alignment, and getting the original out

A message is `max-width: 50%` of the transcript and diagrams are large —
the ER diagram used to shake this out measures 2734×3571 — so the picture
is scaled to fit with `max-width: 100%`. Rendering at natural size instead
was tried and is wrong at this width: it puts most of the diagram behind a
scrollbar inside a half-width bubble. The viewBox dimensions still go on
the `<img>` as `width`/`height`, not to force full size but so the browser
reserves the correct aspect ratio before the image decodes rather than
reflowing the transcript when it lands.

Scaling means the displayed picture is well below full resolution, so each
diagram carries a download control — a plain anchor onto the same `data:`
URI the `<img>` already holds, so saving needs no JavaScript and no round
trip and the file is exactly what is displayed. One per theme, toggled by
the same rules that pick the image, so the saved SVG matches the screen
instead of always being the light variant. A test decodes the `href` and
asserts it is both the image's own URI and a complete SVG document, since
a broken link here fails silently.

Alignment: the `<img>` is `display: block`. As an inline element it sat in
a line box, subject to inline alignment and carrying the baseline's
descender gap beneath it, which pulled diagrams off the left edge; the
theme rules therefore switch between `block` and `none`, never `inline`.
`.mermaid-diagram`'s `margin: 0.6em 0` also clears the UA stylesheet's
`margin-inline: 40px` on `<figure>`, which would indent every diagram.

Two properties of the streaming path shape the rest. The transcript is
re-rendered from scratch on **every token**, and a diagram costs roughly
2 ms to lay out — so completed diagrams are cached by source hash
(failures too, or a mislabelled block would be retried once per token),
and `render::unterminated_fence_start` withholds the one block the
document ends inside of.

That second guard matters more than it looks. A half-written diagram
usually still parses — `flowchart TD` plus one edge is valid Mermaid — so
without it the reader would watch a diagram redraw, reflow and jump on
every token until the fence closed, each throwaway state costing a full
layout the cache can never hit. The guard therefore keys on the fence, not
on whether the source happens to parse.

A source that doesn't parse falls back to the ordinary highlighted code
block. This is a common path, not an edge case — models emit near-miss
Mermaid regularly — and it is why merman was chosen over
`mermaid-rs-renderer`, the other pure-Rust candidate: the latter answers
malformed input with a 16×16 blank SVG rather than an error, in strict
mode as well as lenient, leaving no way to tell a diagram from a failure
and putting an empty frame where the source should be.

#### Detecting a diagram without a tag

A ```` ```mermaid ```` tag is not always there — models emit diagrams into
bare fences, and an attached `.mmd` file has no fence at all — so
`mermaid::looks_like_diagram` decides from the content.

It cannot simply ask merman. Mermaid's parsers are extremely permissive,
and merman inherits that faithfully: handed the sentence `graph is a data
structure of nodes and edges`, it detects a flowchart and **renders** one,
with `is` as a node. `classDiagram is what you want` and a log line
reading `info: build succeeded` behave the same way. Measured over a
corpus of realistic non-diagram blocks, merman's own detector produced
false positives on three of seventeen — and each would have turned
someone's prose or logs into a nonsense picture.

The gate is therefore a table of the diagram headers with the tokens each
may be followed by (`flowchart` takes a direction, `pie` takes `title` or
`showData`, most stand alone). The first meaningful line — after front
matter and `%%` comments, both legal above a header — must be a bare
header and nothing else. That admits all 24 header forms tested, including
`pie title A Very Long Descriptive Title`, and rejects all seventeen
non-diagrams including the three merman renders. Successful rendering is
still required on top.

The gate applies only where there is no explicit tag. A block tagged
`bash` or `json` is left alone even when its contents would parse: the tag
is the author saying what they wrote, and overriding it is exactly how a
shell transcript ends up drawn as a flowchart.

#### Diagrams in attachments

`mermaid::find_in_text` runs the same detection over an attachment's
extracted text, handling both a file that *is* a diagram (no fence — the
case the header gate exists for) and a document that *contains* them
(found by parsing as markdown, so fence lengths and info strings follow
the same CommonMark rules as the transcript). Capped at
`MAX_PER_ATTACHMENT`, with the cap reported to the reader rather than
silently truncating.

The results ride on `AttachmentView`, which `get_session` builds on load
and `send_message` emits as an `attachments` SSE event before the first
token — so a diagram is on screen while the reply is still generating, and
a reload is not what makes it appear. Cost is bounded by construction:
attachment text doesn't change while a reply streams, so this runs once
per send and once per load, never per token.

The view also carries the extracted text, and the browser turns a chip
into a collapsed disclosure holding it plus the diagrams. `text: None` —
a binary or otherwise unreadable format — is what tells the client to
render a bare chip with no expand control, so one is never offered with
nothing behind it. The text is what the model received verbatim, already
bounded by `attachments::MAX_TEXT_CHARS` which marks its own truncation
inline, so displaying it adds no undisclosed cap.

This closes a real gap rather than adding a flourish. An attachment is
otherwise invisible to its sender — the text goes to the model and the UI
shows only a chip with the file's name and size — so a diagram someone
attached was the one part of their own message they could not see.

#### Fencing an attachment into the prompt

`compose_content` inlines an attachment's text into the prompt as a fenced
block, and the fence has to outgrow the body: `attachments::fence_width`
returns one more backtick than the longest run inside it.

Three backticks is only safe for a body containing no fences of its own,
and the documents most worth attaching do contain them. CommonMark ends a
fenced block at the first fence *at least as long* as the opening one, so
a 3-backtick wrapper around a Markdown file holding a ```` ```mermaid ````
block was closed by that file's own closing fence — the rest of the
document escaped the block, and the wrapper's real closing fence went on
to open a new, unterminated one. Measured on a real 200-line file, the
model received four fence transitions where there should have been two,
with the document split into fragments. Asked to render the diagram in it,
the model described it instead. With the fence widened the same file
parses back out of the composed prompt as exactly one code block with its
`mermaid` fence intact.

Whether the model then re-emits the diagram is still the model's call —
nothing here can force that — but it is now working from an intact
document rather than a scrambled one.

#### Putting the attachment's diagram in the answer

Measured across four sessions against the same file, the answer to
"Please, render this" contained a Mermaid fence **zero** times. The replies
open with "Here is the rendered content" and then describe the diagram in
prose — a correct explanation, and no picture. Only an explicit follow-up
("You have a Mermaid diagram") produced a fence. Fixing the prompt fencing
above did not change this; it is how models answer that request.

`appendAttachedDiagramsToAnswer` in `app.js` therefore appends the turn's
attachment diagrams below the answer. Two rules keep it honest:

* It fires **only when the answer contains no `.mermaid-diagram` of its
  own**, so a model that does emit Mermaid is never second-guessed or
  duplicated.
* Each figure carries a `From <file>` caption, so a picture drawn from the
  attachment never reads as one the model produced.

Nothing is written into the message. The persisted content stays exactly
what the model generated, which is also what Save-as-Markdown exports and
what the next turn's context replays — this is a presentation-layer
addition, not a rewrite of model output. It runs on the `done` event
(every token reassigns `innerHTML`, which would wipe an earlier append)
and again on session load, so a reloaded answer carries the same picture a
live one did.

### PlantUML diagrams (`web::plantuml`)

`web::plantuml` is a clean-room, offline implementation of the commonly
generated PlantUML UML families. It parses guarded `@startuml` documents,
lays them out directly, emits theme-specific SVG, and rasterizes the same SVG
to PNG with `resvg`. It never starts Java or Graphviz, downloads a jar, expands
an include, or contacts a PlantUML server.

The parser deliberately accepts a smaller language than PlantUML itself:
sequence, class/object/interface, component/deployment/use-case/state, and
modern activity diagrams. Unsupported structural input returns `None`, so
`web::render` leaves the original fenced block visible instead of presenting
an incomplete diagram as if it were authoritative. Presentation-only syntax
such as common `skinparam` blocks may be accepted when ignoring it cannot
change topology. Source size, item count, output dimensions, and raster pixel
count are capped before expensive work.

Each successful render produces four assets: light and dark SVG plus light and
dark PNG. The streamed HTML contains short `/api/diagrams/<sha256>/<asset>`
URLs rather than repeating base64 PNG data on every token. A 256-entry,
128-MiB LRU holds owned diagrams and cached failures; eviction drops the `Arc`
and its SVG and PNG buffers instead of leaking them for the rest of the server
process.
Attachment discovery shares the same renderer, recognises `plantuml`, `puml`,
and `pu` fences case-insensitively, preserves mixed Mermaid/PlantUML source
order, and applies the common per-attachment cap.

### Bundling a model into the binary (`bundle.rs`)

`orangu-server bundle` writes a new executable: this binary's program image,
byte for byte, then the model's `.gguf` bytes, then a JSON manifest and a
fixed 32-byte footer.

```text
[ program image                       ]  base_len bytes, byte-identical
[ padding to a 4 KiB boundary         ]
[ shard 1 .gguf                       ]
[ padding, shard 2 .gguf, ...         ]  only a split model has these
[ manifest (JSON)                     ]
[ manifest_offset: u64                ]  ─┐
[ manifest_len:    u64                ]   ├ the footer
[ MAGIC:           16 bytes           ]  ─┘
```

The obvious alternative, `include_bytes!`, would put a multi-gigabyte array
through `rustc` on every build, tie one binary to one model at compile time,
and make a bundle something only whoever can build the project could produce.
Appending instead makes the bundle a *file operation* on a finished binary,
so anyone with one can make a bundle in seconds — and, because the manifest
records `base_len`, a bundle can be bundled again, replacing its model rather
than stacking a second one behind the first.

The footer being fixed-size and last is what makes "is this a bundle?" a
seek and a 32-byte read at startup regardless of payload size — cheap enough
that `bundle::embedded()` runs unconditionally on every start. A file whose
footer matches but whose manifest doesn't parse is reported on `stderr` and
then treated as unbundled: a corrupt bundle should not take away the one
thing that might still work, `orangu-server <model>` against a real file.

Alignment is why each shard starts on a 4 KiB boundary. The mapping is of the
executable, not of a `.gguf`, and a model should not read differently for
having been carried in one — a page-aligned start gives every tensor the same
alignment relative to a page that it has in a file of its own.

Reading it back needed two small generalizations rather than a second load
path:

- `orangu::gguf::GgufFile::open_at(path, offset)` parses the GGUF structure
  that begins `offset` bytes into a file. `data_offset` stays relative to the
  segment, so the caller that knows where the segment starts is the one that
  adds `offset` back.
- `engine::loader::LoadedModel` now resolves shards as `ShardSource { path,
  offset }` instead of bare paths, and a tensor's `start` is
  `shard.offset + shard_gguf.data_offset + tensor.offset`. Every on-disk
  model is the `offset == 0` case, so nothing else changed:
  `LoadedModel::open_bundled` differs from `open` only in that every shard
  names the same file at a different offset.

At startup `main::prepare` picks a `ModelSource` — `File` or `Embedded` —
and everything past it is written against the result. A bundled binary with
no config file uses `config::bundled_configuration`: `127.0.0.1:8100`,
`127.0.0.1:8200`, the Hugging Face hub cache as `models`, and the bundle's
own role. Loopback rather than `all` because a bundle is a binary somebody
downloaded and ran, not a deployment somebody configured; `--host all` is how
that gets opted out of for one run.

`bundle` records `--host`/`--port`/`--web` in the manifest as
`config::BundledListen`, which `bundled_configuration` then layers over those
defaults. Every field is `Option` and `#[serde(default, skip_serializing_if)]`,
so a bundle written before they existed parses unchanged and keeps exactly the
behaviour it had — the manifest is a format other builds read, and adding a key
to it must never be a reason an older bundle stops starting. The value is a
default, not a lock: a run-time flag and a config file both still win over it.
`bundle` validates the host itself (`all`/`*` or a literal `IpAddr`) rather
than leaving it to the target machine's `bind`, since the machine that would
report the failure is not the machine that could fix it.

`--host` is also why `ServerConfiguration` carries `web_host_explicit`. The
console follows the API's address unless something says otherwise, so `--host`
has to move it too — but a config that *deliberately* separated them (an API on
the network, the console on loopback) must keep them separated, or exposing the
API would be a way to expose the console by accident. The two addresses being
equal cannot answer that question, so whether the key was written is recorded
rather than inferred.

The bundle's default output name, `orangu-server-bundle-<arch>`, comes from
`bundle::detect_target`, which reads the architecture out of the **binary being
bundled** — ELF `e_machine`, Mach-O `cputype` (a fat binary with more than one
slice is `universal`), PE `Machine`, which also decides the `.exe` suffix.
Reading the header rather than using `std::env::consts::ARCH` is what makes
`--binary` honest: cross-bundling an `aarch64` build on an `x86_64` host has to
produce a file named for the machine that can run it. An unrecognized format
falls back to the host's own architecture rather than failing — the name is a
label, and nothing resolves against it.

Two interactions are worth naming. A handover
(`reexec.rs`, below) that fails and falls back must not name the embedded
model by its label — that is a Hugging Face repo id, and the fallback would
go to the network for a model already inside the file it is falling back
into — so `bundle::EMBEDDED_SPEC` (`"bundled"`) is a reserved spec meaning
"the model in this binary", and that is what travels in
`FALLBACK_MODEL_VAR`. And the model manager's listing has no row for the
embedded model, since it isn't a file in the models directory: nothing is
marked loaded and no Delete button exists for it, so `CurrentView.bundled`
tells the panel to say `bundled` rather than leave an unexplained gap.

On macOS the copied program image is re-signed ad-hoc (`codesign --force
--sign -`), and the ordering is the point: `codesign` writes the new
signature at the end of the image it is pointed at, so it runs *before* a
single payload byte follows, leaving the model outside the signed range where
the kernel never looks. Signing can change the image's length, so the length
the image actually ended up with — not the `base_len` asked for — is what the
manifest records and what a re-bundle truncates back to. ELF and PE images
need none of this. Failure is noisy but not fatal: the file is written and
correct, `codesign` is the one step that depends on the developer tools being
installed, and the manifest is read back off disk afterwards, which is also
what would catch a `codesign` that rewrote more than it was asked to.

### Loading a different model (`reexec.rs`)

`POST /api/models/select` does not swap the model inside the process. It
`execve`s this binary again with the new model in `argv`, which means the
model is loaded by `main::prepare` — the code that already runs at startup —
and there is never a second load path to keep in step with it. Three
properties turn that from a restart into a handover:

**The listening sockets survive.** Rust opens every socket `SOCK_CLOEXEC`,
so `Handover::exec` clears `FD_CLOEXEC` on both listeners before the exec and
names their descriptors to the new image in `ORANGU_INHERIT_FDS`
(`api:<fd>[,web:<fd>]`). `prepare` calls `reexec::adopt_or_bind` instead of
`TcpListener::bind`: given a descriptor it verifies still open (`F_GETFD`,
so a number recycled after a failed handover can't be adopted by mistake) it
takes it, otherwise it binds. The port is therefore never released — a client
connecting mid-load is queued in the listen backlog rather than refused. A
400-request probe across a live handover saw every request answered but the
single one already in flight.

**The process identity survives.** `execve` keeps the pid, so a supervisor
goes on watching the same process, and a `--daemon` server inherits its own
already-detached session. `Handover::argv` therefore deliberately omits
`--daemon`: passing it again would fork a second time and orphan the pid
being watched.

`argv` is rebuilt from what this process *resolved*, not from what it was
given: the workspace as an absolute path (a `--daemon` process has since
moved to `/`) and the role as an explicit flag (it may have been answered at
an interactive prompt). `--config` is passed only if it was passed to this
process, so a server that found its config by the default search makes the
new image repeat that search rather than pinning a path it never chose;
`--host`/`--port`/`--web` follow the same rule (`reexec::Listen`). The address
is normally moot on a handover — both listeners are inherited, so nothing is
bound — but it matters in the one case the adoption check exists for, a
descriptor that didn't survive, where the new image binds instead and must bind
where this server has been answering.

**A failed load falls back.** `FALLBACK_MODEL_VAR` carries the previous
model spec; if the new image's `prepare` fails, `main` execs once more with
it and *without* the variable, which is what bounds the retry to one. This
matters because the pre-check cannot be exhaustive: `reexec::precheck` reads
the header and applies the same judgement as the `SUPPORTED` column
(architecture resolvable, every tensor type decodable), but a GPU backend
with no kernel for one of those types, or a model too large for the machine,
can only be found by loading it. `prepare` binds its listeners *after*
loading the model precisely so that case leaves the inherited descriptors
untouched for the fallback to hand on again.

Both environment variables are read once, at the very top of `main`, by
`reexec::take_inherited`, which also removes them — the same
only-thread-that-exists-yet window that makes `main`'s own
`set_var("RUST_BACKTRACE", ...)` sound. Nothing afterwards reads the
environment for them, so a stale value can't reach a child process or a
second handover.

`select` answers `202` and arms the handover on a 300 ms timer, because
`execve` leaves no "after" to answer from. The timer is best-effort UX, not
correctness: a client whose connection is reset instead of receiving the
`202` is looking at the same event, and its next poll lands on the new image
either way. `WebState::arm_handover` allows one per process — there is only
one process to replace.

`[web].reexec` (default `true`) and `reexec::supported()` (`cfg!(unix)`)
gate the whole thing. When either is false `serve` builds no `Handover` and
`GET /api/models` reports `can_load: false`.

### Model manager (`web::models`)

Served on the **web port**: `orangu-server list` as the view, plus `show`,
`download` and `delete` as the things that can be done from it. Each endpoint
calls the same shared code the matching subcommand does — `orangu::model_spec`
for the scan, grouping and delete, `crate::format_show` for the metadata
dump, `orangu::model_download` for the fetch — rather than a second
implementation that could drift from it.

That extends to the table itself. `ModelView` is one row of `list`, column
for column, and carries the **strings the CLI would print**, not the raw
numbers: `quant` already fell back to `-`, `size` has been through
`format_bytes`, and `supported` is `ModelSupport::cell` verbatim (`Yes
(llama)`, `No (llama, TQ1_0)`) — which is why that method is `pub`. The
client only decides layout. A row this build can't load is greyed, an
unreadable file's `error:` replaces its last three cells, and a repo behind
its Hub revision is marked, all exactly as `format_groups` does the same
three things.

Two things shape the module beyond that:

**The download runs detached.** It takes minutes to hours, so `POST` starts a
`Job` on a blocking thread and returns `202` immediately; the panel polls
`GET /api/models` for its progress. `ModelJobs` holds one job slot — a second
`POST` while one runs is refused with the name of the one holding it, since
two fetches into one models directory would compete for the same disk and the
same free-space check. A *finished* job doesn't hold the slot but stays
readable, so a completed download's result survives a page refresh;
`DELETE /api/models/job` clears it. There is no cancel: the worker is
detached precisely so a closed browser tab doesn't abandon a download
part-way.

Progress comes from `orangu::model_download::DownloadProgress`, a sink the
existing `ProgressBoard` publishes into. Passing one also turns *all* of the
board's printing off, interactive and logged alike: a running server has no
terminal to draw an in-place-updating block on, and the per-file log lines
would go somewhere nobody is watching.

**The listing is cached, not re-scanned per request.** `GET /api/models`
costs a `scan_models_dir` plus a `model_support` pass, which between them open
every GGUF header under the directory *and* every shard of every group —
seconds on a directory holding a few dozen models. The panel polls once a
second for download progress, and that progress is in memory, so
`ModelCatalog` serves a cached scan and only rebuilds on `?rescan=true` (the
panel opening, its **Rescan** button) or after `invalidate()` (a delete, a
finished download). Which row is `loaded` is *not* cached — that is about this
process, not about the directory, so it is decided per request against
`WebState::model_path`.

`[web].delete` (default `true`) gates removal the same way, reported as
`can_delete`.

**It gates models only.** History's own `DELETE /api/sessions/{id}` and
`DELETE /api/sessions` (its per-row cross and **Clear all** footer) are
unconditional, and `GET /api/sessions` carries no capability flag for the
page to check. The two are not the same kind of thing: a model is a file on
disk that a download, or a human with `scp`, put there, and a deployment can
reasonably want that directory read-only while still allowing a model
switch. A chat session is this console's own scratch data, written by the
page that is now asking to delete it — a console unable to clear its own
transcripts is not a posture worth a config key.

`Clear all` does not spare an active session the way `sweep_empty_sessions`
does. The console's own current chat is one of the rows being cleared, and
leaving behind exactly the one the user is looking at is not what the button
says; the browser starts a fresh session immediately afterwards, so nothing
goes on writing into a directory that just went away. Both delete paths also
replace the on-screen transcript when what went was the session it belongs
to — otherwise the next message would `POST` against an id that no longer
resolves. And both stop generation first when a reply is still streaming
into the session being removed: `save_session` on the stream's `done` event
recreates the directory, so the chat would otherwise reappear seconds after
being deleted.

Both switches **remove** their button rather than disabling it. A disabled
control with a tooltip is the right shape for something conditional — a
handover already in flight, a model this build can't load — where the same
button works a moment later or on the row below. A capability the config has
switched off is not a condition of any row; it is what this server does, and
a column of permanently dead buttons explains less than their absence. When
`can_load` is false the loaded row's check mark goes too: which model is
serving is already on the row, as the `loaded` badge beside its name.

Three things are refused rather than attempted:

- **Loading a model while a slot is generating.** The exec would cut the
  reply off mid-stream. Not airtight and cannot be — a request that has
  arrived but not yet acquired its slot isn't `busy` yet — but closing that
  window would mean a barrier between accepting requests and running them,
  for a button a person presses.
- **Deleting the loaded model.** Its weights are mapped by the running
  engine; removing the file leaves this process reading something with no
  name and the next request generating from whatever the kernel still has
  cached. `WebState` carries `model_path` for exactly this check (and to mark
  the row).
- **Deleting a row number that no longer means what the caller saw.** The
  panel sends the `path` its listing showed alongside the `NR`, and
  `ModelRequest::check_still_matches` compares them before anything is
  removed — an `NR` is a position, and a download finishing while a
  confirmation dialog is open re-sorts the listing underneath it.

These endpoints are neither authenticated nor loopback-restricted, matching
the rest of the `web` port and the file-lifecycle API on the API port: the
whole server assumes a trusted network.

### File-lifecycle API (`http::files`)

Served on the **API port**, alongside the OpenAI-compatible and native
endpoints, eight dedicated endpoints cover the whole life cycle of a file,
plus the directories it lives in:

| Endpoint | |
| :-- | :-- |
| `POST /v1/create_file` | write a new file, with optional permissions |
| `POST /v1/modify_file` | replace named line ranges, returning a diff |
| `POST /v1/move_file` | rename a file, optionally re-setting permissions |
| `POST /v1/delete_file` | delete a file |
| `POST /v1/show_file` | return a file's entire content |
| `POST /v1/create_directory` | create one directory, with optional permissions |
| `POST /v1/move_directory` | move an entire directory tree |
| `POST /v1/delete_directory` | delete an empty directory |

Every one is `POST` with a JSON body and a JSON reply, including
`show_file` — one request shape across the whole API is worth more than
matching HTTP verbs to intent for a single read.

Nothing here is recursive except `move_directory`, which moves a tree
because a rename inherently does. Everything else touches exactly one file
or one directory, so a mistyped path costs one entry.

**In a Git repository, these are Git operations** — a file is created,
modified, moved and deleted with `git add`, `git mv` and `git rm`, so the
change is staged rather than only written to disk. **Nothing is ever
committed**; see **Git integration** below.

The implementation lives in `orangu::files`, shared with `orangu`'s own
local tools and typed commands of the same names (`create_file`,
`modify_file`, `/delete_file`, "create myfile.txt with 0644", …), so a tool
call, a typed command and an API request are the same operation with the
same fields, defaults and errors. This chapter is where those fields are
documented for all three.

**Everything is confined to the workspace.** Each path in a request is
resolved against the server's workspace root (`-w`/`--workspace`, default
the current working directory — see the Workspace section of the Inference
server chapter) and refused if it lands outside it. A path may be given
relative to the workspace (`src/main.rs`) or as an absolute path that is
itself inside it; anything else — a `..` that climbs out, an absolute path
elsewhere on the machine, or a symlink inside the tree pointing out of it —
is a `403 outside_workspace` before any file is touched. Two checks back
that up: the lexical one (`orangu::tools::resolve_workspace_path`, the same
resolution `orangu`'s own file tools use, which folds `..` away before
comparing) and a physical one that canonicalizes the nearest *existing*
ancestor of the target — the nearest existing one, so it works for
`create_file`, whose target does not exist yet by definition.

Paths come back in replies relative to the workspace, in the same shape a
client sent them, never as the server's absolute layout.

Three types recur across the endpoints below:

| Type | |
| :-- | :-- |
| *path* | a string, either relative to the workspace (`src/main.rs`) or an absolute path inside it. Never empty |
| *mode* | in a **request**: an octal string (`"0644"`, `"644"`, `"0o644"`) or the number `chmod` takes (`420`); at most `0o7777`. In a **response**: always the four-digit octal string (`"0644"`), or `null` on a non-Unix platform |
| *git* | the object described under **Git integration** below, or `null` when the workspace is not a repository or the request passed `"git": false` |

Unknown fields in a request body are rejected by neither serde nor these
handlers — they are ignored. A missing required field, a wrong type, or
malformed JSON is a `400 bad_request` carrying serde's own message.

#### `POST /v1/create_file`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | file to write |
| `content` | optional, default `""` | the file's full content |
| `mode` | optional | permission bits, as an octal string (`"0644"`) or the number `chmod` takes (`420`) |
| `overwrite` | optional, default `true` | replace the file if it already exists; `false` for create-if-absent |
| `parents` | optional, default `false` | create missing parent directories |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/create_file \
  -H 'Content-Type: application/json' \
  -d '{"path": "src/hello.py", "content": "print(1)\n", "mode": "0640", "parents": true}'
```

```json
{"path":"src/hello.py","bytes_written":9,"mode":"0640","overwritten":false,
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":true,
        "command":"git add src/hello.py","skipped":null,"error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the file written, relative to the workspace |
| `bytes_written` | integer | byte length of `content` as written |
| `mode` | *mode* | the file's permission bits after the write |
| `overwritten` | boolean | `true` when an existing file was replaced (only possible with `overwrite`) |
| `git` | *git* | what Git did |

An existing path is **overwritten** — creating a file that is already there
is an override, and the same is true of `orangu`'s own `create_file` tool
and its typed `/create_file`, which share this implementation. Pass
`"overwrite": false` for create-if-absent, which turns an existing path into
a `409 already_exists`. Without `parents`, a missing parent directory is a
`404 not_found` rather than a quietly-created tree. `mode` is parsed and
validated *before* anything is written, so a bad mode never leaves a file
behind with the wrong permissions. Leaving `mode` out lets the process
umask decide, exactly as an ordinary `create` would.

#### `POST /v1/modify_file`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | file to edit |
| `edits` | required, non-empty | the changes, each naming the lines it replaces |
| `edits[].start_line` | required | first line replaced, 1-based |
| `edits[].end_line` | required | last line replaced, inclusive |
| `edits[].replacement` | optional, default `""` | the lines to put in their place |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

Every range refers to the file **as it was read**, not to the numbering
left behind by an earlier edit in the same request — edits are applied
last-first internally so a caller never has to re-number around its own
changes. Ranges must not overlap, and must address real lines; the one
exception is an insert at `start_line = <line count> + 1`, which appends.

- `end_line = start_line - 1` inserts before `start_line` without replacing
  anything.
- `"replacement": ""` deletes the range.
- The file's trailing-newline state is preserved — a file that ended
  without a newline still does afterwards.

```sh
curl -s -X POST http://127.0.0.1:8100/v1/modify_file \
  -H 'Content-Type: application/json' \
  -d '{"path": "a.txt",
       "edits": [{"start_line": 2, "end_line": 2, "replacement": "TWO\n"},
                 {"start_line": 4, "end_line": 3, "replacement": "four\n"}]}'
```

```json
{"path":"a.txt","lines_before":3,"lines_after":4,"edits_applied":2,
 "diff":"--- a/a.txt\n+++ b/a.txt\n@@ -2,1 +2,1 @@\n-two\n+TWO\n@@ -3,0 +4,1 @@\n+four\n",
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":true,
        "command":"git add a.txt","skipped":null,"error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the file edited |
| `lines_before` | integer | line count before the edits |
| `lines_after` | integer | line count after them |
| `edits_applied` | integer | how many entries of `edits` were applied — always all of them, since any invalid range rejects the whole request |
| `diff` | string | a zero-context unified diff of exactly what changed (see below) |
| `git` | *git* | what Git did |

The `diff` is a **zero-context unified diff** — what `diff -U0` prints. No
diff algorithm is involved: the caller said exactly which lines it was
replacing, so each edit is one exact hunk, and adjacent edits never end up
with two hunks fighting over the same context lines. The `+++` side's line
numbers carry the running length change from the hunks before them, the
same way real unified diff output does.

A file that isn't valid UTF-8 has no line structure to edit, so it is a
`400 not_utf8` rather than a mangled write.

#### `POST /v1/move_file`

| Field | | |
| :-- | :-- | :-- |
| `from` | required | file to move |
| `to` | required | its new path |
| `mode` | optional | permission bits to set at the destination; unset keeps what the file already had |
| `overwrite` | optional, default `false` | replace the destination if it exists |
| `parents` | optional, default `false` | create missing parent directories of the destination |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/move_file \
  -H 'Content-Type: application/json' \
  -d '{"from": "a.txt", "to": "docs/b.txt", "mode": "0600", "parents": true}'
```

```json
{"from":"a.txt","to":"docs/b.txt","mode":"0600","overwritten":false,
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":true,
        "command":"git mv a.txt docs/b.txt","skipped":null,"error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `from` | *path* | where the file was |
| `to` | *path* | where it now is |
| `mode` | *mode* | its permission bits at the destination |
| `overwritten` | boolean | `true` when an existing destination was replaced |
| `git` | *git* | what Git did |

Both paths are workspace-checked, so a move can neither read from nor write
to anything outside the tree.

#### `POST /v1/delete_file`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | file to delete |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/delete_file \
  -H 'Content-Type: application/json' -d '{"path": "src/hello.py"}'
```

```json
{"path":"src/hello.py","deleted":true,
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":true,
        "command":"git rm -f src/hello.py","skipped":null,"error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the file deleted |
| `deleted` | boolean | always `true` — a failure is an error response, not `false` |
| `git` | *git* | what Git did |

Only regular files: a directory is a `400 not_a_file`. This API is a
*file's* life cycle, and a recursive delete behind one JSON field is a much
bigger gun than anything else here hands out.

#### `POST /v1/show_file`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | file to read |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/show_file \
  -H 'Content-Type: application/json' -d '{"path": "a.txt"}'
```

```json
{"path":"a.txt","content":"one\nTWO\nthree\nfour\n","bytes":19,"lines":4,"mode":"0644"}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the file read |
| `content` | string | the whole file, verbatim |
| `bytes` | integer | its byte length |
| `lines` | integer | its line count — a trailing newline does not add an empty last line |
| `mode` | *mode* | its current permission bits |

The only endpoint that changes nothing, so it has no `git` field and takes
no `git` flag. A file that isn't valid UTF-8 has no JSON representation
here, so it is a `400 not_utf8` rather than a lossy conversion.

#### `POST /v1/create_directory`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | directory to create |
| `mode` | optional | permission bits, as an octal string (`"0755"`) or the number `chmod` takes (`493`) |
| `parents` | optional, default `false` | create missing parent directories too |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/create_directory \
  -H 'Content-Type: application/json' \
  -d '{"path": "src/engine/backend", "mode": "0750", "parents": true}'
```

```json
{"path":"src/engine/backend","mode":"0750",
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":false,
        "command":null,"skipped":"nothing_to_stage","error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the directory created |
| `mode` | *mode* | its permission bits |
| `git` | *git* | always `skipped: "nothing_to_stage"` in a repository — Git tracks no directories |

`mode` applies to the directory named by `path`; parents created along the
way keep the umask's own permissions, the same way `mkdir -p -m` behaves.
Leaving `mode` out lets the umask decide for all of them, exactly as an
ordinary `mkdir` would. Like `create_file`, the mode is parsed and validated
before anything is created.

An existing path — file or directory — is a `409 already_exists`. There is
deliberately no `overwrite` counterpart: replacing a directory that is
already there would mean deleting whatever it holds, which is precisely
what `delete_directory` refuses to do.

#### `POST /v1/move_directory`

| Field | | |
| :-- | :-- | :-- |
| `from` | required | directory to move |
| `to` | required | its new path |
| `mode` | optional | permission bits to set on the moved directory; unset keeps what it had |
| `parents` | optional, default `false` | create missing parent directories of the destination |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/move_directory \
  -H 'Content-Type: application/json' \
  -d '{"from": "src", "to": "lib/src", "parents": true}'
```

```json
{"from":"src","to":"lib/src","mode":"0755",
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":true,
        "command":"git mv src lib/src","skipped":null,"error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `from` | *path* | where the directory was |
| `to` | *path* | where it now is |
| `mode` | *mode* | its permission bits at the destination |
| `git` | *git* | one `git mv` covering every tracked file in the subtree — or `skipped: "untracked"` when the directory holds nothing Git tracks |

The whole subtree moves — everything under `from` comes along — in a single
`rename`, so it is atomic, and a move that would cross filesystems fails
outright (`EXDEV`, reported as `io_error`) rather than half-copying a tree.
`mode` applies to the moved directory itself, never to anything inside it.

The destination must not exist (`409 already_exists`): there is no
`overwrite` here, for the same reason `create_directory` has none. Moving a
directory into itself (`{"from": "src", "to": "src/nested"}`) is a
`400 bad_request` rather than the kernel's bare "Invalid argument", and the
workspace root itself cannot be moved.

#### `POST /v1/delete_directory`

| Field | | |
| :-- | :-- | :-- |
| `path` | required | directory to delete |
| `git` | optional, default `true` | perform the change with its Git command (`git add`/`git mv`/`git rm`) when the workspace is a repository; `false` for a plain filesystem change |

```sh
curl -s -X POST http://127.0.0.1:8100/v1/delete_directory \
  -H 'Content-Type: application/json' -d '{"path": "src/engine/backend"}'
```

```json
{"path":"src/engine/backend","deleted":true,
 "git":{"repo_root":"/home/user/src/demo","forge":"github","staged":false,
        "command":null,"skipped":"nothing_to_stage","error":null}}
```

| Response field | Type | |
| :-- | :-- | :-- |
| `path` | *path* | the directory deleted |
| `deleted` | boolean | always `true` — a failure is an error response, not `false` |
| `git` | *git* | always `skipped: "nothing_to_stage"` in a repository — an empty directory holds nothing Git tracks |

**The directory has to be empty.** Anything still in it — files or
subdirectories — is a `409 not_empty`, and nothing is removed. Emptiness is
checked explicitly rather than left to `remove_dir`'s own errno, so the
refusal is one stable code on every platform. A path that isn't a directory
is a `400 not_a_directory`, and the workspace root itself cannot be deleted:
every later request resolves against it.

There is no recursive form. Deleting a tree is the caller's to do, one
`delete_file`/`delete_directory` at a time, which keeps the blast radius of
a single mistyped path to a single directory.

#### Git integration (`git.rs`)

When the workspace sits inside a Git repository, every endpoint above
performs its change **with the matching Git command**, so the result is
staged rather than merely written:

| Endpoint | Git command |
| :-- | :-- |
| `create_file`, `modify_file` | `git add <path>` — after the write, so the staged content is what is now on disk |
| `move_file`, `move_directory` | `git mv <from> <to>` — Git performs the move itself, so the index records a **rename** rather than a delete plus an add |
| `delete_file` | `git rm -f <path>` — Git deletes the file and stages the deletion in one step |
| `create_directory`, `delete_directory` | none — Git tracks files, not directories |

**Nothing is ever committed.** Every operation stops at the index; what to
commit, when, and with what message is the user's decision, and this API
gives no way to make it for them. `git rm` is forced (`-f`) because the
endpoint's contract is that the file goes away — without it Git refuses
whenever the working copy differs from the index, which is exactly when a
deletion is most likely to be wanted. `git mv` is forced only when the
request itself passed `"overwrite": true`.

Each reply carries a `git` object saying what happened, or `null` when the
workspace isn't a repository:

```json
{"from":"a.txt","to":"sub/b.txt","mode":"0644","overwritten":false,
 "git":{"repo_root":"/home/user/src/orangu","forge":"github","staged":true,
        "command":"git mv a.txt sub/b.txt","skipped":null,"error":null}}
```

| Field | Type | |
| :-- | :-- | :-- |
| `repo_root` | string | absolute path of the repository the workspace resolved to |
| `forge` | string or `null` | `"github"`/`"gitlab"`, and only when that forge's CLI (`gh`/`glab`) is installed |
| `staged` | boolean | whether the change reached the index |
| `command` | string or `null` | the Git command that ran, verbatim; `null` when none was run |
| `skipped` | string or `null` | why nothing was staged: `"untracked"`, `"ignored"`, or `"nothing_to_stage"` |
| `error` | string or `null` | Git's own stderr, when its command failed |

Exactly one of `staged: true`, `skipped`, or `error` describes the outcome:
a staged change has both others `null`, a skip carries no `error`, and a
failure carries no `skipped`.

Three cases are skipped rather than treated as failures:

- **`untracked`** — Git has no record of the path, so there is nothing for
  `git mv`/`git rm` to rewrite; the move or delete is a plain filesystem
  operation and the file stays untracked.
- **`ignored`** — `.gitignore` covers the path. `git add` refuses an ignored
  path outright, so writing into e.g. `build/` succeeds and simply isn't
  staged.
- **`nothing_to_stage`** — the directory endpoints. Git tracks no
  directories of its own; a new one becomes visible to Git with the first
  file created inside it.

Where the Git command *performs* the change (`git mv`, `git rm`), a failure
means nothing happened, and the endpoint returns an `io_error`. Where it
only stages an already-written change (`git add`), the file operation has
already succeeded, so the reply is a normal `200` with `staged: false` and
Git's message in `git.error` — the response tells the truth about what
happened rather than implying the write was rolled back.

To bypass Git entirely for one request, pass `"git": false`:

```sh
curl -s -X POST http://127.0.0.1:8100/v1/delete_file \
  -H 'Content-Type: application/json' \
  -d '{"path": "scratch.txt", "git": false}'
```

The file is removed from disk and the index is left alone. Outside a
repository this is what every request does anyway, and `git` comes back
`null`.

`gh`/`glab` are detected (by `origin`'s URL, and only when the matching CLI
is on `PATH`) and reported as `forge`, so a client knows which platform it
is working against. Neither CLI can touch the index — there is no `gh add`
— so the staging itself always runs through plain `git`.

#### Errors

Every failure — including a malformed request body — comes back with the
same shape and a stable `code` a client can branch on, rather than message
text:

```json
{"error":{"code":"outside_workspace","message":"\"../secret.txt\": path escapes the configured workspace"}}
```

The body is always a single `error` object and nothing else:

| Field | Type | |
| :-- | :-- | :-- |
| `error.code` | string | one of the stable codes below |
| `error.message` | string | a human-readable explanation, naming the path it concerns. Wording is not part of the contract — branch on `code` |

| `code` | HTTP | |
| :-- | :-- | :-- |
| `outside_workspace` | 403 | the path resolves outside the workspace root |
| `not_found` | 404 | no such file, or a missing parent directory without `parents` |
| `already_exists` | 409 | the target exists: `create_file` with `"overwrite": false`, `move_file` without `overwrite`, or `create_directory`/`move_directory`, which have no overwrite at all |
| `not_a_file` | 400 | the path exists but isn't a regular file |
| `not_a_directory` | 400 | a directory endpoint was given a path that isn't a directory |
| `not_empty` | 409 | `delete_directory` was given a directory that still has something in it |
| `bad_request` | 400 | unparsable body, empty path, bad mode, an invalid/overlapping line range, a move into itself, or an attempt on the workspace root |
| `not_utf8` | 400 | the file isn't valid UTF-8 |
| `io_error` | 500 | the filesystem refused the operation |

#### Permissions on non-Unix platforms

Permission bits are a Unix concept. Elsewhere `mode` is reported as `null`
in every reply, and a request that tries to *set* one is refused with
`bad_request` rather than silently ignored.

### Session activity tracking and `prune` (`web::sessions`, `prune.rs`)

`save_session` (called by both `create_session` and `append_turn`, so both
creating a session and appending a turn to one trigger it) writes a second
file alongside `chat.json`: `session.json`, recording this process's own pid
and — critically — its `sysinfo::Process::start_time()`. Recording pid alone
would be enough as long as the writing process stays alive, but not once it
exits: the OS is free to hand that same pid number to an unrelated later
process, and without a way to tell the two apart, `is_active` would read the
old session as still active forever. `start_time` is what closes that gap —
a different process at the same pid almost never has the same start time
down to the second, so a mismatch (or the pid not running at all) both read
as "not active," never as an error. `mark_active`'s own write is
best-effort: a failure doesn't fail the session save itself, since
`chat.json` — already written by the time `mark_active` runs — is the data
that actually matters; a session that never got a marker (or whose marker
write failed) just reads as not active, the same as one from a build
predating this.

`is_active` is read from an entirely separate process: `orangu-server
prune` (`prune.rs`), a plain CLI invocation with no connection to whatever
server process actually owns a session. That separation is the whole point
— it's what makes "keep track of which sessions are active" correct even
for a session created long after some other still-running server's own
startup: `is_active` re-queries the live process table every time `prune`
runs, rather than consulting anything cached or computed once earlier, so
the answer is always current relative to *this* invocation, not relative to
whenever the server happened to start.

`prune` itself needs no config file and loads no model — a pure filesystem
operation against a fixed path, the same shape as `system`/`suggest`.
Every invocation first calls `sweep_empty_sessions` (deletes every
non-active session whose `chat.json` is empty, missing, or fails to parse —
the last two read as "empty" too, so an interrupted-write leftover doesn't
linger forever uncleaned), then lists what's left via
`list_sessions_for_prune` (unlike `list_sessions`, the web UI's History
source, this includes zero-message sessions too — only ones `is_active`
protected from the sweep, which `prune` needs to show, not hide) and hands
off to one of three flows: no argument (prints the table, prompts for an
`NR` or `all`), `all` (deletes every remaining non-active session,
`partition`-ing active from inactive first), or a specific `NR`/id
(resolved against the same listing). `main.rs`'s `confirm` — the same
Yes/No stdin reader `delete` uses — is reused here rather than duplicated
(`pub(crate)` in `main.rs`); `prune`'s own relative-time formatter
(`format_relative`, "2h ago") is hand-rolled rather than pulling in a
date/time dependency, the same reasoning `web::current_year` already used
for the copyright year.

# Developers

This project is a local coding-environment client built around a direct OpenAI-compatible chat loop.

## Main components

- `src/bin/orangu.rs` - terminal loop, commands, history, prompt rendering, and waiting state
- `src/config.rs` - INI parsing and normalization
- `src/llm/openai.rs` - OpenAI-compatible client for `orangu-server`
- `src/session.rs` - tool-calling conversation flow
- `src/tools.rs` - local workspace tools for reading, editing, listing, fetching, and shell commands
- `src/tui.rs` - banner and prompt frame rendering

The model builder is a separate program too: `src/bin/orangu-gguf/` pretrains
a model from a manifest of repositories and writes it as GGUF, or rewrites an
existing model at another weight format. Its internals — the parameter
layout, the hand-written backward pass, the tokenizer's one hard constraint,
and the quantization mixture rules — are in
[`doc/manual/en/81-gguf.md`](manual/en/81-gguf.md).

The inference engine is a separate program with its own shape, and it has its
own map: **[`doc/ENGINE.md`](ENGINE.md)** — where the architecture and backend
modules sit, what a new one has to implement, and the handful of rules that
are invisible from inside the module they bite. Read that before adding a
model architecture or a device backend; `doc/SERVER.md` covers running the
server rather than extending it.

## Development workflow

```sh
cargo fmt
cargo test
```

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests.

`format`, `lint` and `audit` run on `ubuntu-latest` only — rustfmt, clippy
and the advisory database give the same verdict on every platform, so
running them three times would only spend runner minutes. `test` runs the
full suite on `ubuntu-latest`, `macos-latest` and `windows-latest`, because
whether the code compiles and behaves identically is the one thing that is
genuinely per-platform.

The Windows job is the one that catches Unix-only code, and a test counts as
code: `#[cfg(unix)]` belongs on any test that spawns a shell script, sets a
file mode, creates a symlink, or signals a PID, not only on the production
code that does those things. `orangu-coordinator`'s tests reach for
`process::fake_server_script` — itself `#[cfg(all(test, unix))]` — whenever
they need a stand-in `orangu-server` to spawn, so the gate travels with the
helper. Anything a gated test alone uses (a fixture, an import) needs the
same gate, or Windows builds it and warns that it is unused. A local
`cargo test` cannot see any of this; only the Windows job can.

### No model in CI

CI runs no model. It builds, lints, audits and runs the ordinary `cargo
test` suite, and nothing more — no weights are downloaded, and none are kept
in the Actions cache. A gate that depends on a multi-gigabyte file living in
a 7-day cache is a gate that fails for reasons unrelated to the change under
test, which is exactly what it did.

The `#[ignore]`d real-model tests are therefore a local step. Point
`ORANGU_TEST_MODEL` at a chat GGUF and name the test:

```
ORANGU_TEST_MODEL=/path/to/model.gguf cargo test --release --bin orangu-server \
  engine::arch::gemma::real_model_tests::gemma4_predicts_paris_after_capital_of_france \
  -- --ignored --exact --nocapture
```

`--release` is not optional: a CPU forward pass of a 5B model in a debug
build turns seconds into minutes. `-- --ignored` as a whole is not useful
either — the ignored tests want different models (`ORANGU_TEST_MODEL`,
`ORANGU_TEST_MOE_MODEL`, `ORANGU_TEST_PLKV_MODEL`, a qwen or embedding
GGUF), a Vulkan device, or are `_scratch_` benchmarks. Name the ones you
want. `orangu-server download` fetches a model if you need one; it resolves
its target against `[orangu-server].models`.

### Reference fixtures

The `#[ignore]`d embedding tests compare against vectors captured from real
llama.cpp, in `src/bin/orangu-server/engine/arch/testdata/`.

**These are generated, not committed.** They are gitignored, and are read at
run time (`engine::arch::read_reference_fixture`) rather than through
`include_str!`, so a checkout without them still compiles and each affected
test simply skips. Generate one when you want to run the test that uses it.
Always capture from llama.cpp, never from orangu's own output — the whole
point is an independent implementation to cross-check against.

Each test's doc comment records the exact llama.cpp build, model, invocation
and input its vector came from. For `embeddinggemma_reference.csv`:

```sh
llama-server -m /path/to/embeddinggemma-300M-Q8_0.gguf \
    --embedding --pooling mean --ctx-size 2048 --port 18080

# Confirm the tokenization still matches the ids the test feeds directly —
# [2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 1].
curl -s localhost:18080/tokenize \
    -d '{"content":"The quick brown fox jumps over the lazy dog","add_special":true}'

curl -s localhost:18080/embedding \
    -d '{"content":"The quick brown fox jumps over the lazy dog"}' \
  | python3 -c 'import json,sys; v=json.load(sys.stdin)[0]["embedding"][0]; \
      print(",".join(repr(x) for x in v))' \
  > src/bin/orangu-server/engine/arch/testdata/embeddinggemma_reference.csv
```

Then run the test against it:

```sh
ORANGU_TEST_EMBEDDING_MODEL=/path/to/embeddinggemma-300M-Q8_0.gguf \
    cargo test --release --bin orangu-server \
    gemma_embedding_matches_real_llama_cpp -- --ignored
```

`qwen3vl_embedding_reference.csv` follows the same pattern but needs a
`Qwen3-VL-Embedding-8B` GGUF and `--pooling last`; see
`qwen3vl_embedding_matches_real_llama_cpp`'s doc comment in
`engine/arch/llama.rs`.

The same arrangement covers a second, unrelated fixture:
`src/bin/orangu-server/engine/testdata/ggml-dequant-reference.bin`, which
pins `engine::quant`'s dequantizers. It holds random quantized blocks of
every type paired with the `f32`s **ggml itself** produced from them, via
`ggml_get_type_traits(t)->to_float` — the same entry point llama.cpp reads
a weight tensor through — so the comparison is bit-for-bit rather than
approximate. Its generator, `ggml-dequant-reference.c`, sits beside it and
is gitignored too; building it needs `libggml-base` installed:

```sh
cc -O2 -o genfix \
    src/bin/orangu-server/engine/testdata/ggml-dequant-reference.c \
    -I/usr/local/include -L/usr/local/lib64 -lggml-base
LD_LIBRARY_PATH=/usr/local/lib64 ./genfix \
    src/bin/orangu-server/engine/testdata/ggml-dequant-reference.bin
```

Adding a quantization type means appending it to the `types[]` list in that
generator and regenerating. Appending is safe: each type is seeded from its
own `ggml_type` id and written as an independent record, so every existing
entry stays byte-identical and only the header count changes — diff the old
and new files to confirm before trusting a regeneration. Two tests read the
fixture, and `dequantize_matches_ggml_for_every_quantized_type` asserts the
type count, so bump that number in the same change.

The list ends with `IQ1_S` and the three types below it
(`IQ1_XS`/`IQ1_XXS`/`IQ1_XXXS`, ids 64-66), which a stock `libggml-base`
does not define — those ids sit above the range ggml reserves for itself, so
only a build carrying them can produce the fixture. Point `-I`/`-L` at such
a build to regenerate, or drop the last four entries and the count to 16 to
regenerate against a stock one. Either way the checked-in guard on those
three dequantizers is `dequantizing_every_type_is_unchanged`'s checksums,
which need no fixture at all.

CI never generates these: the tests that use them need embedding models that
are not cached there, so they skip, exactly as an unset `ORANGU_TEST_*_MODEL`
already makes them skip. The dequant fixture skips the same way when absent
(`read_ggml_reference` returns `None` with a note on stderr).

## Documentation workflow

The manual sources live under `doc/manual/en` (one file per chapter), the cheat
sheet under `doc/cheatsheet/en` (one file per page). One script builds both:

```sh
./doc/build.sh
```

Pass `manual` or `cheatsheet` to build just one. Both PDFs are drawn by
`src/bin/orangu/docs.rs` through the hidden `--build-manual` /
`--build-cheatsheet` flags, on the printpdf engine in `src/bin/orangu/export.rs`
that also writes the `/export` reports — so the documents and the reports share
their branding by construction, and the project carries no LaTeX. Pandoc is
still needed for the HTML manual. The cheat sheet is four pages and stays four:
the build fails when a page's boxes no longer fit.

## Notes

- The client is workspace-scoped by default and uses the current directory unless `--workspace` is supplied. `orangu-server` takes the same `-w`/`--workspace` for the root it operates in, with the same default.
- Command history is stored in `~/.orangu/orangu.history`.
- Local `orangu-server` deployments may take significant time to answer tool-calling prompts, so the default timeout is 30 minutes.

## Basic git guide

Here are some links that will help you

* [How to Squash Commits in Git](https://www.git-tower.com/learn/git/faq/git-squash)
* [ProGit book](https://github.com/progit/progit2/releases)

### Start by forking the repository

This is done by the "Fork" button on GitHub.

### Clone your repository locally

This is done by

```sh
git clone git@github.com:<username>/orangu.git
```

### Add upstream

Do

```sh
cd orangu
git remote add upstream https://github.com/mnemosyne-systems/orangu.git
```

### Do a work branch

```sh
git checkout -b mywork main
```

### Make the changes

Remember to verify the compile and execution of the code

### AUTHORS

Remember to add your name to the following files,

```
AUTHORS
doc/manual/en/97-acknowledgement.md
```

in your first pull request

### Multiple commits

If you have multiple commits on your branch then squash them

``` sh
git rebase -i HEAD~2
```

for example. It is `p` for the first one, then `s` for the rest

### Rebase

Always rebase

``` sh
git fetch upstream
git rebase -i upstream/main
```

### Force push

When you are done with your changes force push your branch

``` sh
git push -f origin mywork
```

and then create a pull requests for it

### Repeat

Based on feedback keep making changes, squashing, rebasing and force pushing

### PTAL

When you are working on a change put it into Draft mode, so we know that you are not
happy with it yet.

Please, send a PTAL to the Committer that were assigned to you once you think that
your change is complete. And, of course, take it out of Draft mode.

### Undo

Normally you can reset to an earlier commit using `git reset <commit hash> --hard`.
But if you accidentally squashed two or more commits, and you want to undo that,
you need to know where to reset to, and the commit seems to have lost after you rebased.

But they are not actually lost - using `git reflog`, you can find every commit the HEAD pointer
has ever pointed to. Find the commit you want to reset to, and do `git reset --hard`.

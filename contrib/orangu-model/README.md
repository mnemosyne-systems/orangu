# orangu-model

A model built from scratch, in four stages: a smoke test, a training run
that writes **BF16**, and the two quantizations — **Q6_K** and
**Q4_K_M** — that come off it.

Everything here drives `orangu-gguf`, which does the whole pipeline
itself: clone the corpus, train a tokenizer on it, pack it into tokens,
pretrain from random weights, and write the GGUF that `orangu-server`
loads. No Python, no training framework, nothing to install beside the
binary. The manual chapter *Building a model* is the reference; this
directory is the ready-to-run version of it.

## The corpus

`corpus.json` is the training list: this project's own family, the large
codebases it is written against, and the top of GitHub in the three
languages they are written in.

| Project | Licence |
|:---|:---|
| [pgagroal](https://github.com/pgagroal/pgagroal), [pgmoneta](https://github.com/pgmoneta/pgmoneta), [pgexporter](https://github.com/pgexporter/pgexporter), [pgvictoria](https://github.com/pgvictoria/pgvictoria) | BSD-3-Clause |
| [pgmoneta_mcp](https://github.com/pgmoneta/pgmoneta_mcp), [orangu](https://github.com/mnemosyne-systems/orangu) | **GPL-3.0-or-later** |
| [pgopr](https://github.com/pgopr/pgopr), [billetsys](https://github.com/mnemosyne-systems/billetsys) | **EPL-2.0** |
| [linux](https://github.com/torvalds/linux) | **GPL-2.0-only** |
| [postgres](https://github.com/postgres/postgres) | PostgreSQL |
| [quarkus](https://github.com/quarkusio/quarkus), [wildfly](https://github.com/wildfly/wildfly) | Apache-2.0 |

Beside them are the **most-starred open source projects on GitHub in C,
Rust and Java** — twenty of each, less the handful named below. Sixty-six
repositories in all, and the file itself is the list.

A source does not have to be a Git repository. `orangu-gguf` looks at each
`url` and does the right thing with it: clone a remote or a local
repository, read a plain directory of files where it sits, or unpack a
`.tar.gz`/`.tar.bz2`/`.zip` (downloading it first if it is remote). Files
inside a source may be `.gz` or `.bz2`; they are read through.

All sixty-six are open source, so all sixty-six train: the gate is OSI
approval (<https://opensource.org/license>), not permissiveness.
**Nineteen of them are copyleft**, five of those AGPL-3.0, and every one of
those licences is written into the finished model's metadata.

The gate is also why a few of the most-starred projects in those languages
are *missing*: `hello-algo` is CC BY-NC-SA (non-commercial),
`LeetCodeAnimation` has no licence file, `advanced-java` is CC BY-SA,
`curl` is under its own `curl` licence, and `wrk` is under a *modified*
Apache 2.0. None of those is an OSI-approved SPDX identifier, so adding
them means saying so deliberately with `allow_any_license`.

That is a record, not a refusal. What a corpus licence decides is not whether
a model can be trained but what the trained weights may be published under.
`corpus.json` answers that question for this build: **the model is
Apache-2.0**, written into the finished file as `general.license`.

A source whose licence is *not* an OSI-approved identifier is left out of
the corpus and named in the output — excluded rather than fatal, since one
odd entry should not stop a run, and named rather than dropped, since a
corpus quietly smaller than the manifest says is a run nobody can
reproduce. `"allow_any_license": true` in the manifest trains on it anyway.

The corpus is recorded either way: every repository URL and every licence
it was taken under is written into the finished model, beside the model's
own. A manifest that omits `license` writes no such key rather than
inventing one.

### English

Code alone teaches a model to write code and nothing else — not to follow
an instruction, not to explain what it wrote. `corpus.json` adds prose:

```json
"wikipedia": { "language": "en", "max_bytes": 8589934592 }
```

That is 8 GiB of English Wikipedia article text, taken from Wikimedia's
search-index dumps — the one form of the dumps that is already plain prose
rather than wikitext, so no markup, no infoboxes, no image embeds and no
parser to be wrong. It streams into `~/.orangu/gguf/corpus/wikipedia/` a
shard at a time and stops at the cap; a shard already there is never
fetched twice, so an interrupted download resumes. Lower `max_bytes` for a
faster first run, or drop the key entirely for code only.

Wikipedia text is CC BY-SA 4.0 — share-alike, and recorded in the model's
provenance with the corpus's other licences.

`corpus-smoke.json` is the four smallest permissive projects plus 20 MB of
Simple English Wikipedia, about 40 MB in total. It exists only so the smoke
test can fetch in seconds while still covering both kinds of source.

## The stages

| Script | What it does | How long |
|:---|:---|:---|
| `00-smoke.sh` | The whole pipeline on a 20 MB corpus at the `smoke` size, then both quantizations | ~30 minutes |
| `10-bf16.sh` | The training run: random weights to a BF16 GGUF | days to weeks |
| `20-q6_k.sh` | BF16 to Q6_K | seconds to minutes |
| `30-q4_k_m.sh` | BF16 to Q4_K_M | seconds to minutes |
| `install.sh` | Puts the finished files where `orangu-server` looks for them | seconds |
| `run-all.sh` | The four build stages, in order | as above |

```sh
./00-smoke.sh       # prove the pipeline works here first
./10-bf16.sh        # the real run, entirely per corpus.json
./20-q6_k.sh
./30-q4_k_m.sh
./install.sh        # and make the server able to see them
```

Both quantizations read the **BF16** file, never each other. Rounding an
already-quantized model a second time is worse than rounding the original
once, and `orangu-gguf` refuses to do it rather than quietly producing the
worse file.

### Installing

`./install.sh` reads the models directory out of `orangu-server.conf` — the
same file and the same key the server reads, so there is nothing to keep in
step — and copies each `*.gguf` into it in Hugging Face's hub cache layout:

```
<models>/models--mnemosynesystems--orangu-smoke-GGUF/
    refs/main                                    the revision
    blobs/<sha256>                               the file, named by contents
    snapshots/<rev>/orangu-code-smoke-BF16.gguf  a symlink to the blob
```

That layout is what turns a file into a *model*. `orangu-server list` finds
any `.gguf` anywhere under the models directory, but it can only show one as
`<org>/<name>:<QUANT>` — the form `--model` and `download` accept — when
there is a `models--<org>--<name>` directory on its path:

```
NR  MODEL                               QUANT   SIZE       SUPPORTED
 1  mnemosynesystems/orangu-smoke-GGUF  BF16    14.77 MiB  Yes (qwen3)
 2  mnemosynesystems/orangu-smoke-GGUF  Q4_K_M   4.99 MiB  Yes (qwen3)
 3  mnemosynesystems/orangu-smoke-GGUF  Q6_K     6.21 MiB  Yes (qwen3)
```

```sh
orangu-server --model mnemosynesystems/orangu-smoke-GGUF:Q4_K_M
```

One repository per *training size*, with every quantization of that size
inside it, so a `smoke` build and a `2b` build never land in the same one.
`ORANGU_MODEL_ORG` and `ORANGU_MODEL_NAME` change the two halves of the
name; `./install.sh a.gguf b.gguf` installs only what you name.

Re-running it is safe. The revision is derived from the file contents, so
installing an unchanged model twice changes nothing, and installing a
retrained one replaces the previous revision and drops the files nothing
points at any more. A repository directory this script did not create — a
downloaded copy of the same one — makes it stop rather than rewrite it.

### Run the smoke test first

It costs about half an hour, and it is the only thing standing between a typo in the
settings and finding out about it a week into the training run. Its output
is meant to be gibberish — 200 steps on 20 MB of source teaches a model
nothing. What it proves is that every stage runs on this machine and that
`orangu-server` loads what came out:

```sh
orangu-server ./smoke/orangu-code-smoke-BF16.gguf --port 8100
```

## Settings

**They are all in the manifests.** `corpus.json` and `corpus-smoke.json`
carry the size, the context length, the weight format, the vocabulary, the
schedule and the corpus; the scripts carry none of it. To change a run,
edit the manifest — the *Building a model* chapter documents every field
and its default.

```json
{
  "training_size": "0.5b",
  "context_size": "32k",
  "epochs": 3,
  "repositories": [ ... ]
}
```

Anything passed to a script is handed straight to `orangu-gguf`, so a
one-off override still works without editing anything:

```sh
./10-bf16.sh -ts 0.5b        # try the smaller size before committing
./20-q6_k.sh path/to/other-BF16.gguf
```

There is nothing else to set. The scripts take no environment variables at
all: to run one stage, invoke that stage; to skip one, do not invoke it.
`run-all.sh` is only the four of them in order. The binary is found on the
PATH, or in `target/release/` when this is a checkout.

Where the files land follows from the manifest: the model is written to
`<name>-<training_size>-<format>.gguf` in the current directory, and the
corpus, tokenizer, packed tokens and checkpoints to
`~/.orangu/gguf/<manifest name>/`. Both are settable in the manifest
(`output`, `work_dir`) if you want them elsewhere. The smoke stage keeps
everything in `smoke/` so it never collides with the real run.

## What it costs

- **Disk.** The corpus is tens of gigabytes — sixty-six shallow clones,
  several of which (the kernel, `elasticsearch`, `ghidra`, `rust`, `guava`,
  `PowerToys`) are a gigabyte or more of source on their own — and English
  Wikipedia adds whatever `max_bytes` says (8 GiB by default). The first
  run is also the slow one: `jobs` clones at a time, over the network. A
  `2b` checkpoint is another ~32 GB, and it is rewritten every 200 steps.
- **Memory.** Training holds four numbers per parameter: the weight, its
  gradient, and the optimizer's two moments. That is ~32 GB for `2b`
  before activations, and ~8 GB for `0.5b`.
- **Time.** Everything runs on the CPU in 32-bit floating point. `smoke` is
  minutes. `0.5b` is days. `2b` is weeks.
- **Re-running a stage.** A finished run leaves a checkpoint, and the next
  run replaces it and says so. Only an *unfinished* one stops a run, so
  `00-smoke.sh` can be run as often as you like without cleaning up first.
- **Interruption.** `10-bf16.sh` checkpoints every 200 steps; set
  `"resume": true` in `corpus.json` and run it again to continue. Nothing
  else needs redoing: the clone, the tokenizer and the packed tokens are
  all reused as they stand.

### A word on the token budget

A model is only as good as the data it saw, and the rule of thumb is at
least 20 tokens of training data per parameter — ~40B tokens for `2b`.
This corpus is several billion tokens with Wikipedia in it — the sixty-six
repositories are most of that, and Wikipedia the rest. That makes it a real
training set for `smoke` and `0.5b`, and still short of what `2b` wants
from a single pass; raise `epochs`, or widen `corpus.json` further, before
reading much into a `2b` result. `orangu-gguf` prints the exact token count after packing, so
the first run tells you where you actually stand.

\newpage

# Building a model

`orangu-gguf` builds a model file. It has two jobs, and they are the two
ends of one pipeline:

- **Train one from nothing.** Given a JSON manifest of
  permissively-licensed repositories, it clones them, trains a tokenizer on
  what it finds, packs the text into tokens, pretrains a transformer from
  random weights, and writes the result as a GGUF file that
  `orangu-server` serves directly.
- **Convert one you already have.** Given an existing full-precision model
  — a GGUF file, or a Hugging Face checkpoint directory — it writes it out
  at the weight format you ask for: `BF16`, `Q6_K`, `Q4_K_M`, or any of
  the dozen in *Weight formats* below.

There is nothing to install beside the binary, and no conversion step
between the two: what training produces is already the file the server
loads.

## Quick start

Rewrite a model you already have at 4-bit:

```sh
orangu-gguf -m ./my-model-BF16.gguf -q q4_k_m
```

Train a small one, end to end, from three repositories:

```sh
cat > corpus.json <<'JSON'
{
  "name": "my-model",
  "training_size": "smoke",
  "context_size": "8k",
  "repositories": [
    { "url": "https://github.com/BurntSushi/ripgrep",  "license": "MIT" },
    { "url": "https://github.com/sharkdp/fd",          "license": "Apache-2.0" },
    { "url": "https://github.com/rust-lang/rustlings", "license": "MIT" }
  ]
}
JSON

orangu-gguf corpus.json
```

The manifest is the whole command: the size, the context length, the weight
format, the corpus and the schedule all live in it, and everything it does
not mention takes a documented default. `smoke` trains in about a quarter
of an hour and exists to prove the whole pipeline works on your machine
before you commit a week to it; `tiny` is a smaller shape again, for when
the question is only whether the tool runs at all. Then serve what came
out:

```sh
orangu-server ./my-model-smoke-BF16.gguf
```

## The manifest

The manifest is the whole build. Every setting of a run lives in it, the
training material lives under `repositories`, and nothing is passed on the
command line that the file cannot say — which is what makes a build
reproducible. The file *is* the description of the model, and re-running it
months later against a newer binary produces the same thing.

Reproducible means bit-for-bit, and it does not depend on the machine's
parallel width: the same manifest and the same `seed` write the same file
on one core or on thirty-two.

The smallest useful manifest is the material and nothing else, because
every other field has a default:

```json
{
  "repositories": [
    { "url": "https://github.com/owner/repo", "license": "MIT" }
  ]
}
```

That trains a 2B model at a 256k context and writes it as BF16. The full
form says everything out loud:

```json
{
  "name": "orangu-code",
  "license": "Apache-2.0",
  "description": "A small code model trained on permissive Rust and C.",

  "architecture": "qwen4exp",
  "training_size": "1b",
  "context_size": "256k",
  "quantization": "bf16",
  "vocab_size": 32768,

  "sequence_length": 2048,
  "batch": 4,
  "epochs": 1,
  "seed": 1,
  "checkpoint_every": 200,

  "repositories": [
    { "url": "https://github.com/owner/repo", "license": "MIT" },
    { "url": "https://github.com/other/repo", "license": "ISC", "branch": "stable" }
  ]
}
```

An unknown key is an error rather than being ignored. A misspelled
`traning_size` that silently trained the default size for a week is a worse
outcome than a parse failure.

### Identity

| Field | Default | Meaning |
|:---|:---|:---|
| `name` | `orangu` | The model's name, and the stem of the file written |
| `license` | *(none)* | The licence the **model** is published under. Absent writes no such key rather than inventing one |
| `description` | *(none)* | Free text, written into the file |

### The model

| Field | Default | Meaning |
|:---|:---|:---|
| `architecture` | `qwen4exp` | `qwen4exp` or `qwen3`; see [Sizes](#sizes) |
| `training_size` | `1b` | One of the architecture's sizes: `tiny`, `smoke`, `1b`, and for `qwen3` also `0.5b` and `2b` |
| `context_size` | `256k` | The context length the model declares. Accepts `8192`, `8k`, `1M` |
| `quantization` | `bf16` | The weight format written |
| `calibration` | *(none)* | A text the trained model reads before it is written, measuring the importance matrix the weights are fitted to; required by the formats below three bits |
| `vocab_size` | `32768` | Tokens in a newly trained vocabulary |
| `chat_template` | *(ChatML)* | The Jinja2 template written as `tokenizer.chat_template`; `""` writes none |

### Training

| Field | Default | Meaning |
|:---|:---|:---|
| `sequence_length` | `2048` | Tokens per training sequence |
| `batch` | `4` | Sequences per optimizer step |
| `steps` | *(from `epochs`)* | Optimizer steps to run |
| `epochs` | `1` | Passes over the corpus, when `steps` is absent |
| `learning_rate` | *(per size)* | Peak learning rate |
| `seed` | `1` | Weight initialization and batch sampling |
| `log_every` | `60` | Seconds between progress lines; `0` prints only the last step |
| `eval_every` | `200` | Steps between validation passes; `0` disables |
| `checkpoint_every` | `200` | Steps between checkpoints; `0` disables |
| `resume` | `false` | Continue from the checkpoint in the work directory |
| `export_only` | `false` | Write the model from the checkpoint without training further |

`log_every` is in **seconds**, unlike `eval_every` and `checkpoint_every`.
A step is seconds at `tiny` and minutes at `2b`, so a count of steps is a
different amount of output at every size — and it is worst at the size that
runs for days, which is the one somebody is watching to see that it is
still alive. A run prints a line a minute:

```
step 47/200  loss 7.6696  lr 9.23e-4  |g| 1.687  2.4k tok/s  elapsed 0d:0h:0m:20s  eta 0d:0h:1m:5s
```

The interval is wall-clock and holds *inside* a step as well as between
them — at `0.5b` and `2b` one step is minutes, and a line that waited for a
step boundary would not arrive on any schedule you could predict. Those
lines name the step in flight and carry the last finished step's numbers:

```
step 1/411527  elapsed 0d:0h:4m:0s
```

The estimate is against **tokens**, not against finished steps. Tokens are
counted as each sequence lands, so it moves during a step as well as at the
end of one — the difference between an estimate and a number that stands
still for four minutes and then jumps — and it is against the run's own
average rate, so a slow patch shows up as the estimate lengthening rather
than as a rate the next line contradicts.

Whatever the interval, the last step always prints, and its rate is the
run's rather than the last minute's. Validation and checkpoints stay on
step counts: those are work the run does, not a report on it.

### The corpus

| Field | Default | Meaning |
|:---|:---|:---|
| `repositories` | *(required)* | The training material — see below |
| `wikipedia` | *(absent)* | English (or another language's) Wikipedia, as prose to train on beside the code |
| `jobs` | `4` | Repositories cloned at once |
| `max_file_size` | `1048576` | Largest corpus file read, in bytes |
| `tokenizer_sample` | `268435456` | Bytes of corpus text the tokenizer is trained on |
| `offline` | `false` | Use the corpus already in the work directory; clone nothing |
| `rebuild` | `false` | Retrain the tokenizer and repack, discarding what is there |
| `allow_any_license` | `false` | Train on sources whose licence is not an OSI-approved one, instead of excluding them |

### Paths

| Field | Default | Meaning |
|:---|:---|:---|
| `work_dir` | `~/.orangu/gguf/<manifest>` | Corpus, tokenizer, packed tokens and checkpoints |
| `output` | `<name>-<size>-<format>.gguf` | Where to write the model |

### The training material

Each entry of `repositories` names a repository and the licence it is
under:

| Field | Meaning |
|:---|:---|
| `url` | The source — a Git remote, a local repository, a directory of files, or an archive |
| `license` | The SPDX identifier the source is under. Required |
| `branch` | A branch or tag to clone instead of the default. Git sources only |

**A source is not always a repository**, and `orangu-gguf` works out which
it is by looking rather than assuming:

| `url` | What happens |
|:---|:---|
| `https://…`, `git@…` | `git clone --depth 1` into the work directory |
| a local path containing `.git` | the same clone, which Git makes nearly free by hard-linking |
| a local directory | **read where it is** — a corpus directory can be tens of gigabytes, and copying it to look at it would be absurd |
| `*.tar.gz`, `*.tgz`, `*.tar.bz2`, `*.tar`, `*.zip` | unpacked into the work directory, downloading it first if it is remote |

Files inside a source may be compressed: a `.gz` or `.bz2` is read
through, and what decides whether it is training text is the name
*underneath* — `main.rs.gz` is Rust, `logo.png.gz` is still a picture. An
archive entry that tries to climb out of its directory is dropped rather
than unpacked.

**The licence is checked, not decorated.** A source is trained on if its
declared licence is one of the OSI-approved ones
(<https://opensource.org/license>, by SPDX identifier, deprecated
spellings like `GPL-3.0` included). Copyleft is in that set: what a corpus
licence decides is not whether a model can be trained but what the trained
weights may then be published under, and that is the manifest author's
call, not this tool's. Every licence in the corpus is written into the
finished model's metadata, so the question can be answered from the file
itself.

Anything that is *not* an OSI-approved identifier — an unrecognised name, a
source-available licence that is not open source, or a compound expression
like `MIT OR Apache-2.0`, where which licence you are relying on is a
choice this tool will not make for you — is **excluded from the corpus**
and named in the run's output. Excluded rather than fatal, because one odd
entry in a list of hundreds should not stop a run; named rather than
dropped, because a corpus quietly smaller than the manifest says is a
training run nobody can reproduce. Excluding *everything* is an error.
Setting `"allow_any_license": true` trains on them anyway; it sits in the
same file as the declarations it overrides, which is where a decision like
that belongs.

What is checked is the licence the manifest *declares*. `orangu-gguf` does
not read each repository's own `LICENSE` file to confirm the claim; making
that claim correctly is the manifest author's job, which is exactly why the
field is mandatory rather than inferred.

The list of repositories and the set of licences they were taken under are
written into the finished model file, so the weights carry their own
provenance rather than a note beside them.

### English, and where it comes from

A corpus of source code teaches a model to write code and nothing else —
not to follow an instruction, not to explain what it wrote, not to write a
sentence. Adding `wikipedia` to the manifest fixes that:

```json
"wikipedia": { "language": "en", "max_bytes": 8589934592 }
```

| Field | Default | Meaning |
|:---|:---|:---|
| `language` | `en` | The wiki's language code: `en`, `de`, `simple` |
| `max_bytes` | `8589934592` (8 GiB) | How much *extracted article text* to take |
| `date` | *(newest)* | A dump date (`YYYYMMDD`) to pin |

**Which dump, and why it matters.** Wikimedia publishes the same articles
several ways, and only one of them is already what a language model should
read:

| Dump | What is in it |
|:---|:---|
| `mediawiki_content_current` | Raw wikitext: templates, infoboxes, tables, `[[File:…\|thumb\|300px\|…]]` embeds. Needs a wikitext parser, and a hand-written one leaks citation markup into the corpus |
| `mediawiki_content_history` | The same wikitext, but *every revision of every page* — 3,374 files against 19, and the same article dozens of times over with small edits |
| **`cirrus_search_index`** | **The article text with all of that already resolved — plain running prose, line-delimited JSON, sharded, updated weekly** |

The third is what `orangu-gguf` reads, because it is the only one that
needs no parser to be wrong in. Only namespace-0 pages are taken, and only
the article `text`; the dump's `auxiliary_text` — where image captions,
table cells and navbox links live — is deliberately not read. Measured over
a thousand consecutive articles of the real dump, that leaves one mention
of an image file name (inside a citation URL) and no markup at all.

Text is written into `<work_dir>/wikipedia/` a shard at a time, and a shard
already there is not fetched again — so an interrupted download resumes,
and a second run costs nothing. The stream stops at `max_bytes`, so the cap
is what you download, not what you filter afterwards. Roughly: a gigabyte
of compressed shard yields most of a gigabyte of prose.

Wikipedia text is CC BY-SA 4.0. That is not a software licence and never
goes through the repository gate, but it is written into the model's
provenance alongside the corpus's other licences, because it bears on the
same question they do.

### A ready-made corpus, and the scripts that build it

`contrib/orangu-model/` is the whole thing ready to run: two manifests,
and a script per stage from the smoke test through BF16 to both
quantizations, plus one that installs the result. Its own `README.md` covers the settings and what each stage
costs; this is the summary.

`contrib/orangu-model/corpus.json` is the training list — this project's
own family, the large codebases it is written against, and the top of
GitHub in the three languages they are written in:

| Project | Licence |
|:---|:---|
| pgagroal, pgmoneta, pgexporter, pgvictoria | BSD-3-Clause |
| pgmoneta_mcp, orangu | **GPL-3.0-or-later** |
| pgopr, billetsys | **EPL-2.0** |
| linux | **GPL-2.0-only** |
| postgres | PostgreSQL |
| quarkus, wildfly | Apache-2.0 |

Beside them are the most-starred open source projects on GitHub in each of
**C**, **Rust** and **Java** — 54 more, from `scrcpy`, `git`, `FFmpeg` and
`php-src` through `rust`, `deno`, `ripgrep` and `alacritty` to
`spring-boot`, `guava`, `ghidra` and `dubbo`. Sixty-six in all.

Every one of them is open source, so every one trains. Nineteen are
copyleft — five of those AGPL — which bears on the licence of the finished
weights, and `corpus.json` answers that: the model is `Apache-2.0`, which
is what the file's `general.license` says.

A few of the most-starred projects in those languages are *not* in the
list, and each for the same reason — the gate is an OSI-approved SPDX
identifier, and they do not have one. `hello-algo` is CC BY-NC-SA
(non-commercial), `LeetCodeAnimation` carries no licence at all,
`advanced-java` is CC BY-SA, `curl` is under its own `curl` licence, and
`wrk` is under a *modified* Apache 2.0. Adding any of them means saying so
deliberately with `allow_any_license`.

The stages, in order:

```sh
cd contrib/orangu-model
./00-smoke.sh       # the whole pipeline, on 20 MB, in about twenty minutes
./10-bf16.sh        # the real run, entirely per corpus.json
./20-q6_k.sh
./30-q4_k_m.sh
./install.sh        # into the models directory, so the server can see them
```

The scripts hold no settings of their own — everything is in the two
manifests, so changing the run means editing `corpus.json` rather than
hunting through shell. `install.sh` is the exception to "everything is in
the manifest", and only because the setting it needs is not a property of
the model: it reads the models directory out of `orangu-server.conf`, the
same file and the same key the server reads, and lays each file out as
`models--<org>--<model>-<size>-GGUF/snapshots/<rev>/` so that
`orangu-server list` shows it as `mnemosynesystems/orangu-smoke-GGUF:Q4_K_M`
rather than as a bare filename — which is also the form `--model` takes. Its
own `README.md` has the details. `00-smoke.sh` uses `corpus-smoke.json` — the four
smallest permissive projects, about 20 MB — so it clones in seconds and
keeps its output in `smoke/`. Its result is meant to be gibberish; what it
proves is that every stage runs on this machine, which is worth knowing
before committing days to the next one. Both quantization stages read the
**BF16** file rather than each other, and find it in the current directory
unless you name one.

Expect the real corpus to take a while and a fair amount of disk: the Linux
kernel alone is over a gigabyte of source at `--depth 1`, and the whole set
is several. It is also worth being honest about its scale — it is a corpus
of hundreds of millions of tokens, which is a real training set for
`smoke` and `0.5b`, and short of the budget the `2b` size wants from a
single pass (see **Sizes** below).

## The command line

There is almost none of it, and that is the point. Training reads the
manifest:

```sh
orangu-gguf corpus.json
```

Converting an existing model has no manifest to read, so it takes what it
needs as arguments:

```sh
orangu-gguf -m ./my-model-BF16.gguf -q q4_k_m
```

| Option | What it does |
|:---|:---|
| `-m`, `--model` | Convert this file, listed model, checkpoint directory or Hugging Face repository instead of training |
| `-ts`, `--training-size` | Override the manifest's size for this run |
| `-q`, `--quantization` | Override the weight format written |
| `-cs`, `--context-size` | Override the declared context length |
| `-o`, `--output` | Override where the model is written |
| `--imatrix FILE` | Fit the quantization to an importance matrix: a file, or a Hugging Face repository that publishes one |
| `--calibration FILE` | Measure an importance matrix over this text and fit the quantization to it; with `-q auto`, also the text its KL is measured over |
| `--imatrix-out FILE` | With `--calibration`, write the measured matrix here and stop |
| `--max-size SIZE` | With `-q auto`, the most bytes of weights the file may take |
| `--max-kl NATS` | With `-q auto`, the most KL from the source over the calibration text, in nats |
| `--tensor-type PATTERN=TYPE` | Write tensors whose name matches `PATTERN` as `TYPE` (repeatable) |
| `--list-quantizations` | Print the weight formats and exit |
| `--dump MODEL` | Print a model file's metadata and tensor table |
| `--hash MODEL` | Print a SHA-256 of every tensor's stored bytes |
| `--compare A B` | Compare two model files; exits non-zero when they differ |
| `--split-max-size SIZE`, `--split-max-tensors N` | Write the model as shards; see [Shards and metadata](#shards-and-metadata) |
| `--keep-split` | Write a shard set cut where the source set is cut |
| `--merge FIRST` | Join a shard set into one file |
| `--set`, `--delete`, `--override-kv` | Change metadata, in place or in the file written |
| `--pure` | Quantize every tensor that may be to the format's base type |
| `--allow-requantize` | Accept a quantized source |
| `--worker` | Join the `[workers]` channel as a gguf node; see [Several machines](#several-machines) |
| `--channel` | List the channel's nodes, servers and gguf nodes |
| `--config FILE` | Read `[workers]` from this file instead of `orangu-gguf.conf` |
| `--perplexity TEXT` | Print the perplexity of the `-m` model over the text, divided over the channel's gguf nodes when there are any |
| `-s`, `--shell-completions` | Print the shell completion script and exit |

The four overrides exist for the one-off — trying `smoke` before committing
to `1b`, or writing a second quantization of a model you already have.
Anything you want to keep belongs in the manifest, where the next run will
still find it. `-ts` and `-cs` are two letters behind a single dash, which
most tools read as two separate short flags; they are accepted here anyway,
in both the `-ts 1b` and `-ts=1b` spellings, alongside the long forms.

`-s`/`--shell-completions` prints a bash/zsh/fish/PowerShell completion script for the
shell detected from `$SHELL` — the same switch every orangu binary has (see
the Shell completions chapter). It completes the manifest with `.json`
files, `-m` with `.gguf` files, `-q` with the weight formats (by shelling
back out to `orangu-gguf --list-quantizations`), `-o`, `--imatrix` and `--flamegraph`
with files, and offers `-ts`/`-cs` alongside their long forms:

```sh
# bash — add to ~/.bashrc:
eval "$(orangu-gguf -s)"
# zsh — write once to your fpath directory:
orangu-gguf -s > ~/.zsh/completions/_orangu-gguf
# fish — add to ~/.config/fish/config.fish:
orangu-gguf -s | source
# PowerShell — add to $PROFILE:
orangu-gguf -s | Out-String | Invoke-Expression
```

## Converting a Hugging Face checkpoint

A Hugging Face checkpoint — `config.json`, `tokenizer.json`,
`tokenizer_config.json` and the `*.safetensors` weights, one file or a
shard set with its `model.safetensors.index.json` — converts the same way a
GGUF file does. Name the repository:

```sh
orangu-gguf -m Qwen/Qwen3-0.6B -q bf16
```

It is looked for in the models directory `orangu-server` serves from
(`[orangu-server].models` in `orangu-server.conf`) and downloaded there when
it is not, in the same layout `orangu-server download` uses, so a second
conversion of the same model fetches nothing. A local checkpoint directory
works too: `orangu-gguf -m ./Qwen3-0.6B -q bf16`.

The file is written to `<models>/orangu-converted/<name>-<format>.gguf`,
where `orangu-server list` shows it as `<name>-<format>` and the server
serves it by that name; `-o` writes it elsewhere.

`-m` takes, in this order: a model file; a model `orangu-server list`
shows, by `MODEL` name or `NR`; a checkpoint directory; a Hugging Face
repository.

Architectures converted:

| Checkpoint | Written as |
|:---|:---|
| `LlamaForCausalLM`, `MistralForCausalLM` | `llama` |
| `Qwen2ForCausalLM` | `qwen2` |
| `Qwen3ForCausalLM` | `qwen3` |
| `Qwen3MoeForCausalLM` | `qwen3moe` |
| `Phi3ForCausalLM` | `phi3` |
| `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | `gemma4` |
| `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM` | `qwen35` |
| `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` | `qwen35moe` |

Any other is refused by name. So is a checkpoint tensor the
converter has no place for, rather than being left out of a file that
would then load and answer wrong.

The exceptions are a vision tower and a multi-token prediction head: each
is a file of its own, so the conversion leaves it out.

The tokenizer is BPE: byte-level (GPT-2 style), or with byte fallback
(`▁` for a space, `<0xNN>` tokens) as Gemma writes it. A byte-level
tokenizer's pre-tokenizer — how text is cut into
words before merging — is recognised from its definition in
`tokenizer.json`, and one that is not recognised is an error rather than a
guess, since a wrong guess tokenizes every prompt slightly differently from
how the model was trained. The special tokens, whether BOS and EOS are
added, and the chat template come from `tokenizer.json`'s post-processor,
then `tokenizer_config.json`, then `config.json`, in that order of
authority. A token `tokenizer_config.json` adds that `tokenizer.json` lacks
is written too. `add_bos_token` and `add_eos_token` are always written: with
nothing in the checkpoint asking for them, the tokenizer adds neither, and
the file says so. A turn that ends on a token other than EOS — `<|end|>`,
`<turn|>` — is written as `tokenizer.ggml.eot_token_id`, from
`tokenizer_config.json`'s `eot_token` or the first stop token in
`generation_config.json`, so the server stops at the end of the turn. The
sampling defaults in `generation_config.json` are written as
`general.sampling.*`.

The licence in the model card's front matter (`README.md`) is written as
`general.license` when the card is there.

`--compare` is how a conversion is checked. Two files of the same model at
the same weight format hold the same bytes in every tensor; where the
formats differ, the largest difference between the values read back is
printed instead, as a share of the largest value — the format's rounding
when the weights are the same, and far more when they are not. Every
metadata key that differs, or is on one side only, is named, and for a list
the first entry that differs. Both sides are files, or models by the name
`orangu-server list` shows:

```sh
orangu-gguf --compare SmolLM2-360M-Instruct-F16 bartowski/SmolLM2-360M-Instruct-GGUF:F16
```

## Sizes

Each architecture has its own sizes; all share one vocabulary, so a
tokenizer trained once carries across them. The parameter counts are for the
default 32,768-token vocabulary; a smaller one takes two tensors down with
it, which is most of what `tiny` is.

Every hidden and feed-forward width is a multiple of **256**, and that is a
requirement rather than a preference: 256 is the K-quants' super-block, and
a row length it does not divide cannot be a K-quant at all. The
feed-forward *down* projection is the one that decides it, because its row
length is the feed-forward width rather than the hidden one.

### `qwen4exp`, the default

A hybrid of attention and the gated delta rule, three delta-rule blocks to
one attention block, with routed experts in every feed-forward:

- **Hyper-connections.** The residual is several streams wide; each
  sub-layer reads a learned mix of them and writes back into each with a
  learned weight.
- **Gated attention** with partial rotary positions, and a **sparse-attention
  indexer** on every attention block that picks which earlier blocks of
  positions a query reads once the context passes its budget.
- **The gated delta rule**: a short causal convolution, then a recurrent
  state per value head, with a learned decay and step size.
- **Routed experts** behind a softmax router, with a shared expert beside
  them, trained with a load-balancing term.
- **An n-gram hash embedding** on block 1: each token and its predecessors
  hashed into a table, gated into the streams.

| Size | Hidden | Streams | Blocks | Heads | KV heads | Experts (used) | Expert width | Parameters |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tiny` | 256 | 2 | 4 | 4 | 2 | 8 (2) | 256 | ~28M |
| `smoke` | 768 | 2 | 8 | 6 | 2 | 16 (4) | 512 | ~0.25B |
| `1b` | 1024 | 4 | 16 | 8 | 2 | 32 (4) | 512 | ~1.07B |

A token passes through only the experts it is routed to, so the parameters
it uses are a fraction of the count: about a third at `1b`.

### `qwen3`

A dense decoder: grouped-query attention with rotary positions, RMSNorm, a
SwiGLU feed-forward, and a per-head norm on the queries and keys.

| Size | Hidden | Feed-forward | Blocks | Heads | KV heads | Parameters |
|:---|---:|---:|---:|---:|---:|---:|
| `tiny` | 256 | 768 | 4 | 4 | 4 | ~20M |
| `smoke` | 768 | 2048 | 8 | 6 | 2 | ~100M |
| `0.5b` | 1280 | 3584 | 24 | 10 | 5 | ~0.53B |
| `1b` | 1536 | 5632 | 28 | 12 | 6 | ~1.0B |
| `2b` | 2048 | 8192 | 30 | 16 | 8 | ~2.0B |

### Which size for what

The two small ones have different jobs. `tiny` answers "does this machine
run the tool at all", and it answers it in a couple of minutes. `smoke`
answers "does this machine run the tool *properly*", because the things that
go wrong quietly need a model shaped like a real one to go wrong in: it has
fewer key/value heads than query heads, so grouped-query attention is
actually grouped; three query heads to a key/value head, so the group is not
a power of two and an index that works by alignment stops working; eight
blocks; and matrices larger than a last-level cache, which is what makes a
measurement taken on it mean anything at the sizes below. At `qwen4exp` it
also routes each token to 4 of 16 experts, and its indexer selects at the
sequence lengths it trains on.

**These are real training runs, and they cost what training costs.**
Everything runs on the CPU in 32-bit floating point. `tiny` is minutes and
`smoke` is about a quarter of an hour;
the three real sizes are days to weeks of continuous compute on a corpus
large enough to be worth it, and they need memory to match — training holds
four numbers per parameter (the weight, its gradient, and the optimizer's
two moments), so a 1B-parameter run needs roughly 16 GB before activations.
`checkpoint_every` and `resume` exist because a run that long will be
interrupted.

A model is only as good as the token budget it saw. Below roughly 20 tokens
of training data per parameter, a run is not worth evaluating; the sizes
above earn their keep well past that.

## Weight formats

| `-q` | Bits per weight | Use it when |
|:---|:---|:---|
| `bf16` | 16 | The reference file. Quantize *from* this. |
| `f16` | 16 | Same size as `bf16`, less exponent range. |
| `f32` | 32 | Twice the size, no more accurate in practice. |
| `q6_k` | ~6.6 | Indistinguishable from `bf16` on anything measurable. |
| `q5_k_m` | ~5.7 | Half the size of `bf16`, nothing visibly lost. |
| `q5_k_s` | ~5.6 | As above, without the reinforced tensors. |
| `q4_k_m` | ~4.8 | The size and quality sweet spot. |
| `q4_k_s` | ~4.7 | A little smaller than `q4_k_m`, a little weaker. |
| `iq4_nl` | ~4.5 | Four bits on rows a K-quant's 256-wide block will not divide. |
| `iq4_xs` | ~4.3 | The smallest four-bit file. |
| `q3_k_l` | ~4.3 | Three bits with the most reinforcement. |
| `q3_k_m` | ~4.0 | Three bits, usable; the model is now measurably worse. |
| `q3_k_s` | ~3.6 | Three bits throughout. |
| `q2_k` | ~3.2 | The smallest K-quant. Expect to notice. |
| `iq3_m` | ~3.7 | The best three-bit lattice file. |
| `iq3_s` | ~3.5 | Three-bit lattice throughout. |
| `iq3_xs` | ~3.3 | Between `iq3_s` and `iq3_xxs`. |
| `iq3_xxs` | ~3.1 | The smallest three-bit file. Needs `--imatrix`. |
| `q2_k_s` | ~2.9 | Two bits with less reinforcement. Needs `--imatrix`. |
| `iq2_m` | ~2.7 | The smallest file most models still answer well from. Needs `--imatrix`. |
| `iq2_s` | ~2.5 | Needs `--imatrix`. |
| `iq2_xs` | ~2.3 | Needs `--imatrix`. |
| `iq2_xxs` | ~2.1 | Needs `--imatrix`. |
| `iq1_m` | ~1.8 | For models too large to fit any other way. Needs `--imatrix`. |
| `iq1_s` | ~1.6 | The smallest file; a much weaker model. Needs `--imatrix`. |
| `auto` | ~4.5 | A type per tensor, chosen for this machine's speed within `q4_k_m`'s quality and size (below). Not in `--list-quantizations`. |

`--list-quantizations` prints the same table with a line on each. Bits per
weight are approximate and for a whole file, mixture included — which is why
`q3_k_l` and `iq4_xs` land so close together, and why every K-quant reads
higher than its own block would suggest. The exact figure depends on how
much of a model is vocabulary.

### What the suffix means

A `Q4_K_M` file is not a file of `Q4_K` tensors. The `_S`, `_M` and `_L` are
*mixtures*: which tensors are carried above the file's base format, and by
how much. They are the standard spellings — a model published as `Q4_K_M`
anywhere means the same mixture this writes, tensor for tensor — and the
tensors that earn the extra bits are always the same ones:

- **The vocabulary projection** (`output.weight`) is six bits in every
  quantized file (five in the `IQ1`, `IQ2` and `IQ3_XXS` files), eight when
  its rows are not a multiple of 256. It is the last thing before the
  softmax, where an error becomes a wrong token directly. A model without
  one reads its token embedding as the output, and that embedding is
  carried the same way.
- **The value and down projections** are reinforced on the outer blocks,
  and on every third block in between. An error there lands on every token
  that attends through them. A mixture's experts are placed by the block
  they sit in. Blocks count in number order, and a quantized file lists its
  tensors that way, block by block, whatever order its source had.
- **Norms are never quantized**, in any format, and neither are a
  mixture's router and the other small tensors that steer the rest; they
  keep the type they had in the source.

`_S` reinforces least, `_M` more, `_L` most; `Q6_K`, `IQ4_NL` and `IQ4_XS`
have no variants because there is nothing above them worth spending on.
Grouped-query attention shifts the rules slightly: it makes the value
projection small, so carrying it high costs almost nothing, and the mixtures
spend there whenever the model has four or more query heads per key/value
head. A model with eight experts carries its key and value projections at
eight bits.

A row a K-quant's 256-wide block does not divide falls back to the 32-wide
type of about the same precision: `Q2_K` and `Q3_K` to `Q4_0`, `Q4_K` to
`Q5_0`, `Q5_K` to `Q5_1`, `Q6_K` to `Q8_0`, `IQ4_XS` and the lattice types
to `IQ4_NL`; and to `f16` when not even 32 divides it. Each one is reported
as a `note:` while the file is written. `orangu-server show --tensors`
prints what a finished file actually contains.

### Fitting to an importance matrix

```sh
orangu-gguf -m HuggingFaceTB/SmolLM2-360M-Instruct -q q4_k_m \
    --imatrix bartowski/SmolLM2-360M-Instruct-GGUF
```

An *importance matrix* records, for every input column of every weight
matrix, the mean squared activation that reached it while the model read a
calibration text. With one, each block is fitted to keep the errors that
matter least: a column the model leans on gets a closer fit, at the expense
of columns it barely reads. The file is the same size and the same format;
it is the choice of levels that changes. Publishers who make quantized
files with one usually publish it beside them, and `--imatrix` takes either
that file or the repository's name. A named repository's matrix is found in
the models directory, or downloaded into it the first time.

Both matrix formats are read: the GGUF one (`*imatrix*.gguf`) and the older
binary `.imatrix`. A mixture's experts each have their own importances,
and each expert is fitted to its own.

Every K-quant (`Q2_K` to `Q6_K`), every lattice type, both 4-bit codebook types (`IQ4_NL`,
`IQ4_XS`) and the 32-wide fallbacks except `Q8_0` use the matrix; `bf16`,
`f16` and `f32` ignore it. With a matrix, an `IQ4_NL` or `IQ4_XS` file
leaves its first blocks' down projections at four bits instead of
reinforcing them, as the fit already protects them. Tensors the matrix has
no entry for, the token embedding typically among them, are written as
they would be without one. A matrix measured on a different model is
refused, by the first tensor whose shape does not match. The line
`importance` in the output says how many tensors were fitted, and the
written file records which matrix it was (`quantize.imatrix.file`, with the
calibration text's name, entry and chunk counts beside it).

### Measuring an importance matrix

```sh
orangu-gguf -m Qwen/Qwen3-0.6B -q iq2_m --calibration calib.txt
orangu-gguf -m 3 --calibration calib.txt --imatrix-out model.imatrix.gguf
```

`--calibration` measures the matrix instead of taking a published one: the
model reads the text, and every weight matrix of every block records the
squared activations that reached each of its input columns. The text is
cut into chunks of 512 tokens, each read from the start with a fresh
context; a leftover shorter than a chunk is not read. Any text of the kind
the model will be asked to handle works; a few hundred kilobytes, a few
hundred chunks, is typical. For a mixture each expert records only the
tokens routed to it. Each chunk starts with the model's BOS token when the
model takes one.

The model runs inside `orangu-gguf`, on the CPU, with the same forward pass
`orangu-server` serves it with; it needs no GPU and takes the memory the
model takes. It ends by saying how many of the blocks' weight matrices
were reached: one that is never multiplied (a convolution, say) has no
entry. A GGUF source is measured as it is;
a checkpoint is first written as BF16 to `<models>/orangu-converted/`, and
that file is measured. The matrix is kept there too, as
`<model>.<text>.imatrix.gguf`, and reused while it is newer than both the
model and the text. With `--imatrix-out FILE` the matrix is written there
instead and nothing is converted: it can then be given to any number of
conversions with `--imatrix`. It is in the GGUF importance-matrix format
(`<tensor>.in_sum2`, `<tensor>.counts`, `imatrix.datasets`,
`imatrix.chunk_count`, `imatrix.chunk_size`), which other quantizers read
too. `--calibration` and `--imatrix` are alternatives, except with `-q auto`,
where the matrix comes from `--imatrix` and the text is read for the KL.

A manifest takes the same text as `"calibration"`. The trained model is
then written as BF16 into the work directory, measured there
(`calibration.imatrix.gguf`), and written at the manifest's `quantization`
fitted to that matrix. A format below three bits without a `calibration`
is refused before training starts, not after.

### The lattice I-quants

The lattice I-quants — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S` — store each group of four or eight weights as one index
into a fixed codebook of points, with the signs kept apart. Below about
three bits that search only lands anywhere useful when it is told which
weights matter, so `IQ1_*`, `IQ2_*`, `IQ3_XXS` and `Q2_K_S` need an
importance matrix (`--imatrix`); without one they are refused, naming the
first tensor that needs it, rather than written to look fine and answer
badly. The token embedding and the output projection are never held to it.
`IQ3_S`, `IQ3_XS` and `IQ3_M` can be written without one and are better
with one.

### Choosing for this machine: `-q auto`

```sh
orangu-gguf -m ./Qwen3-0.6B-BF16.gguf -q auto --calibration calib.txt
orangu-gguf -m ./model-BF16.gguf -q auto --imatrix model.imatrix.gguf --calibration calib.txt
```

The standard mixtures spend bits by rule, the same on every machine. `-q
auto` spends them by measurement, on the machine that will run the file:
each tensor gets its own type, chosen so the file decodes and reads prompts
as fast as this machine can run it, while its KL divergence from the source
stays within the standard `q4_k_m`'s and its size within the standard
file's. The standard file is the one made from the same source with the
same importance matrix. A file is written for one machine; on another its
choices are still valid, just not chosen for it.

It needs a calibration text: the KL is measured over it. With
`--calibration` alone the importance matrix is measured over the same text
(as for any `-q`); with `--imatrix` as well, the matrix is that file and the
text is only read for the KL. The KL is measured on the end of the text,
two chunks of 512 tokens, on the CPU, the way an importance matrix is.

What is measured:

- **What each type costs each tensor in error.** Every tensor is encoded
  at every candidate type and read back, and the difference is weighed by
  the importance matrix, so it counts where the model reads. A large tensor
  is measured on a sample of its rows.
- **What a unit of error costs in KL, per role.** The source is read over
  the text, then the standard mixture, held in memory; then, one role at a
  time (every block's value projection, the output projection, …), that
  role is lowered about a quarter in size and read again. The KL it adds,
  per unit of error it adds, is the role's weight. It differs between
  models: an output projection tied to the token embedding, or one
  architecture's value projection, can cost several times what the same
  error costs in another.
- **What each type costs in time**, at the model's own shapes, on the device
  the model will decode on (the card the server would choose, else the
  cores): one token's product and one prompt chunk's, through the same
  kernels the server runs, on each kernel family a card offers, since the
  server chooses one for the whole file. When the standard file is larger
  than the card can hold beside its KV pages, a mixture's routed experts
  are timed on the cores instead, where the server keeps them, and do not
  count against the card's room. The answers are kept for a week, per
  device, driver and build, so the next file for this machine starts from
  them.

Types are chosen a role at a time: every block's value projection gets one
type, every block's down projection one, and so on. A role whose blocks
are of different types can decode slower than either type would alone.
Every role starts at its fastest type, and the change that buys the most
predicted KL for the least added time is made, one role at a time, until
the prediction is back within the standard file's. Neither the
file's size nor the weights the card holds may grow past the standard's,
nor past the card's memory less room for its KV pages, and neither a
decoded token nor a prompt token may take longer than the standard file's.
A card picks its decode kernels and its prompt kernels apart, each for the
whole file, and each is priced on the family it would pick. A smaller type is
not always a faster one — the cards' fastest kernels are for `Q4_K`, `Q5_K`
and `Q6_K` — so the types below those are considered only when
`--max-size` asks for a file smaller than the standard one.

The choice is then held in memory and its KL measured over the text. If it
is over the standard's, it is chosen again under a budget tightened by the
miss, up to four rounds in all. When no round comes within it, the standard
`q4_k_m` is what is written, and the output says so; the file never misses
its KL without saying so.

| Option | What it does |
|:---|:---|
| `--max-size SIZE` | The most bytes of weights the file may take (`2.2G`, `2300M`); smaller than the standard file's brings in the three-bit and smaller types |
| `--max-kl NATS` | The most KL from the source over the text, instead of the standard file's |
| `--tensor-type PATTERN=TYPE` | Write every tensor whose name matches the regular expression as `TYPE`; repeatable, the last match wins. Works with every `-q`, and `-q auto` keeps those types |

The output shows each round and ends with the chosen types:

```
  the standard Q4_K_M reads at KL 0.0612 over 1024 tokens (56.3 s)
  measured 8 roles' weights in 108.0 s
  round 1: predicted KL 0.0611, measured 0.0713 against 0.0612
  round 2: predicted KL 0.0523, measured 0.0661 against 0.0612
  round 3: predicted KL 0.0486, measured 0.0594 against 0.0612
chosen for AMD Radeon RX 5500M (RADV NAVI14) [radv (Mesa 26.2.7)]
      Q4_K    82 tensors       204.4 MiB
      Q5_K   115 tensors       140.9 MiB
  against the standard Q4_K_M: decode -7.5% and prompts -39.0% predicted, size -7.3%, KL 0.0612 -> 0.0594 measured
```

Time is predicted, not measured: serve the file and compare it with
`orangu-bench` to see it. The file records `general.file_type` as the
standard name of the type most of its weights are in, and the name it is
written to ends in `-auto`. The model is held in memory twice while the
choice is made — the source, and the encodings being measured.

### What is not written, and why

The format defines more types than this writes; one group is left out
deliberately, and one family of names is not offered.

The **round-number quantizations** — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
`Q8_0` — are the generation before the K-quants, and none of them is a file
type here. `Q4_K_M` and `Q5_K_M` are about the size of the 4- and 5-bit ones
and more accurate; `Q6_K` is a quarter smaller than `Q8_0` and nearly as
accurate. Asking for one names its replacement rather than failing as
unknown. They appear inside K-quant files as the fallback for rows a
256-wide block does not divide, and `Q8_0` is among the types `-q auto`
chooses from.

You may also see `_XL` or `UD-` on models published elsewhere. They are not
standard suffixes: they belong to one vendor's dynamic quantizations, whose
per-tensor choices are their own, so they are not offered here.

Converting works from a full-precision file. Quantizing an
already-quantized model would round twice and produce something worse than
quantizing the original once, so it is refused with a message saying which
file to start from instead, unless `--allow-requantize` is given ([Shards and
metadata](#shards-and-metadata)).

## Shards and metadata

A model too large for one file is a **shard set**:
`model-00001-of-00003.gguf`, `model-00002-of-00003.gguf`,
`model-00003-of-00003.gguf`. The first shard carries every metadata key;
each shard says which it is and how many there are. Wherever this tool
reads a model, the first shard stands for the whole set, and
`orangu-server` serves a set the same way.

| Option | What it does |
|:---|:---|
| `--split-max-size SIZE` | Write the model as shards of at most `SIZE` each (`300M`, `4G`), headers included |
| `--split-max-tensors N` | Write the model as shards of at most `N` tensors each |
| `--keep-split` | Write a shard set cut where the source set is cut |
| `--merge FIRST` | Join a shard set into one file |
| `--set KEY=TYPE:VALUE` | Set a metadata key (repeatable) |
| `--delete KEY` | Delete a metadata key (repeatable) |
| `--override-kv KEY=TYPE:VALUE` | Write this metadata key into the output (repeatable) |
| `--pure` | Quantize every tensor that may be to the format's base type |
| `--allow-requantize` | Accept a quantized source, rounding its tensors a second time |

With a format (`-q`), the split options cut the file being written. Without
one, nothing is converted: every tensor is copied as stored, so splitting a
file and merging it back gives the original byte for byte.

```sh
# A 4 GB limit per file, converting as it goes:
orangu-gguf -m model-BF16.gguf -q q4_k_m --split-max-size 4G
# A file cut into shards as it is, and joined again:
orangu-gguf -m model-Q4_K_M.gguf --split-max-size 2G
orangu-gguf --merge model-Q4_K_M-00001-of-00003.gguf
# A shard set quantized into a set cut the same way:
orangu-gguf -m model-BF16-00001-of-00004.gguf -q q6_k --keep-split
```

A single tensor larger than the limit has a shard to itself. A converted
Hugging Face checkpoint is written whole and then cut, so it needs room for
both for a moment.

**Metadata.** `TYPE` is one of `u8`, `i8`, `u16`, `i16`, `u32`, `i32`,
`u64`, `i64`, `f32`, `f64`, `bool` and `str`, or `int` and `float`, which
take the type the key already has: `--set llama.context_length=int:4096`
keeps it the `u32` it was. With no format given, `--set` and `--delete`
change the file itself: in place when the new header fills the same space as
the old (the header is padded, so changing a number usually does);
otherwise the file is rewritten beside itself and renamed over it, its
tensors copied unchanged. On a shard set they change the first shard. With
`-o`, the original is left alone and the changed copy written there. With a
format, `--set`, `--delete` and `--override-kv` all apply to the file being
written. The `split.*` keys belong to the shard set and cannot be set.

**`--pure`** skips the mixture: no tensor is carried above the format's base
type, and a tensor whose rows the base type's block does not divide still
falls back to the narrower type, as always.

**`--allow-requantize`.** A quantized source is refused unless this is
given, because each tensor is then rounded twice and the result is worse
than quantizing the full-precision original once. With it, the tool warns,
naming the source's types, and goes ahead.

## Several machines

`orangu-gguf` joins the same `[workers]` channel as `orangu-server` (see
*Workers*). A channel then holds two kinds of node: **servers**, which serve
models, and **gguf nodes**, which take a share of building a model file.
Both answer the same probe, so each kind finds the other.

The settings live in `orangu-gguf.conf` — `./orangu-gguf.conf`, then
`~/.orangu/orangu-gguf.conf`, or the file `--config` names — in a
`[workers]` section spelled as the server's:

```ini
[workers]
name = lab
secret = a-long-shared-secret
```

| Key | Default | Meaning |
|:---|:---|:---|
| `name` | *(none)* | The channel. Without it `orangu-gguf` joins none |
| `secret` | *(none)* | The channel's shared secret, as the servers have it |
| `host` | `0.0.0.0` | The address `--worker` listens on |
| `port` | `8401` | The port `--worker` listens on, and gguf nodes are looked for on |
| `server_port` | `8400` | The port the channel's servers listen on |
| `network` | this machine's networks | The IPv4 networks to look on, `10.0.0.0/24,…` |
| `peers` | *(none)* | `host:port` entries asked as well, as `ORANGU_WORKERS_PEERS` is |
| `scan` | `on` | `off` asks only `peers`; with peers and no `network`, it is off |
| `tls_cert`, `tls_key`, `tls_ca` | *(none)* | TLS between nodes, as the servers use it |

The model is not in this file and the file is not in the manifest: which
machines do the work is not a property of the model, and the same manifest
builds the same file on any set of them.

**`--worker`** runs this machine as a gguf node until it is stopped. Its
port is 8401 so it can run beside a server on 8400.

**`--channel`** lists the channel's nodes:

```text
$ orangu-gguf --channel
channel 'lab': 3 nodes

kind    name                         address                state        load  does
server  10.0.0.5:8400                10.0.0.5:8400          free           3%  serves unsloth/Qwen3-8B-GGUF Q4_K_M
gguf    10.0.0.6:8401                10.0.0.6:8401          free           0%  imatrix, perplexity, quantize, train
gguf    10.0.0.7:8401                10.0.0.7:8401          working      100%  imatrix, perplexity, quantize, train
```

A gguf node is claimed by the rules a server's workers are: a top about to
claim asks each free node for a lease first; a free node promises itself
for ten seconds to the lowest id that asked, so two tops starting together
do not split the nodes between them; then the top proves the secret and the
node proves it back. A claimed node works for that top until the top lets
go, and answers probes as `working` meanwhile. A node more than 80% busy is
not claimed. A server never claims a gguf node as one of its workers, and a
gguf top never takes over a server.

**What is divided.** Measuring an importance matrix (`--calibration`, and a
manifest's `calibration`) and `--perplexity` run in chunks of 512 tokens
that do not depend on each other. With a channel configured, `orangu-gguf`
claims the free gguf nodes, runs chunks itself, and hands the next chunk to
whichever node is free. Each chunk is measured from nothing and the chunks
are added up in chunk order, alone or divided, so one, two or three nodes
write the same matrix byte for byte and print the same perplexity. A node
that is lost has its chunk run again elsewhere. Every node reads its own
copy of the model: the same file, at the same path or by name under its
models directory, checked by size and a hash of its first 16 MiB. The
nodes should be the same kind of CPU: a different one can round a sum
differently. Each node prints how many chunks it ran and how much it sent.

**Quantizing** a GGUF file (`-m FILE -q TYPE`) divides tensors by rows.
Every row encodes on its own, so a tensor's rows are cut into pieces,
encoded on several nodes, and written in row order: the file is the same,
byte for byte, as one machine writes. Each node reads the rows from its own
copy of the source, found as above, and sends back only the encoded bytes;
the top sends the piece's importance values with it.

Whether a tensor is divided is decided per type, tensor by tensor, and
printed with the numbers it was made from whenever it changes. Each link's
round trip and bandwidth are timed when the run starts.

- *Alone* is the tensor's size over how fast this machine encoded that type
  by itself, first timed on a few rows.
- *Divided*, until the type has been divided past its first tensor, takes
  each node to be as fast as this one, behind its link's measured latency
  and bandwidth and the time it takes to read the tensor. After that it is
  the measured pace of the tensors of that type that were divided, from
  the first piece sent to the last one back. The first divided tensor of a
  type is not counted: the nodes build the type's tables during it.
- A tensor is divided when that is at least 10% sooner. Tensors too small
  to give every node four pieces are encoded alone.

At the end, the run prints how many tensors of each type were divided and
how many pieces each node encoded.

**Training** divides each step's sequences. At the start of every step the
top sends each claimed node the model's configuration and parameters; each
node then takes the step's sequences one at a time, and the top takes as
many at once as it runs side by side alone, so a faster node runs more of
them. A sequence's gradient is computed from zero
and sent back whole, and the step's gradient is the sum of the sequences'
gradients in sequence order, alone or divided. On nodes with the same kind
of CPU and with every product on the CPU, one, two or three nodes therefore
write the same checkpoints and the same model. A node that is lost has its
sequence run on the top, and the run continues without it. Each step moves
the parameters to every node and one gradient back per sequence a node
runs. When no sequence is left to start, the top also runs the ones the
nodes are still at; a node whose sequence the top finishes first is
released with a `train: … released` line, so a node whose link or
processor costs more than it gives leaves the run after one step. At the
end the run prints how many sequences each node ran.

`--perplexity TEXT` with `-m MODEL` prints the perplexity of the model over
the text: windows of 512 tokens, each scored from its middle on, on the
CPU.

## What comes out

A GGUF file of the manifest's architecture, `qwen4exp` or `qwen3`, which
`orangu-server` serves. A `qwen4exp` file carries the hash constants of its
n-gram embedding among its keys, so the server hashes exactly as training
did.

The tokenizer is a byte-level BPE vocabulary trained on your corpus and
carried inside the file. Every byte has a token, so there is no input it
cannot represent. `<|endoftext|>` separates documents during training and
is the model's stop token; `<|im_start|>` and `<|im_end|>` are in the
vocabulary from the start, and the file declares the ChatML template that
uses them.

That template is about the *file*, not the model. A chat endpoint has no
way to turn a list of messages into a prompt without one, and refuses the
request rather than guessing — so a model with no template cannot be opened
in the web console or any other chat client at all, which makes a perfectly
good file look broken. Writing the template makes the file serveable
everywhere. It does not make the model an instruct model: a pretrained base
model has never been taught what a conversation is, and its answers will
read like the corpus continuing rather than like a reply. Instruction
tuning is the step that earns those tokens, and it is not what this tool
does. Set `"chat_template": ""` for a file that carries none, and
`"chat_template": "<jinja>"` for one of your own.

`--context-size` sets the context length the file declares and scales the
rotary base to match it. It is not the same thing as the sequence length
trained on (`sequence_length`, 2048 by default): a longer declared
context costs nothing at training time and lets the model be *used* further
out than it was trained, with the quality falling off the further past the
training window you go.

## The work directory

Every stage lands on disk, so a run that stops can be picked up:

```
~/.orangu/gguf/<manifest>/
  corpus/            one directory per cloned repository
  wikipedia/         article text, one shard per file
  tokenizer.json     the trained vocabulary and merge table
  tokens.bin         the packed token stream
  checkpoint.bin     weights, optimizer state, and the step reached
```

Re-running the same command reuses each of these. A checkpoint left by a run
that reached its last step is replaced, and the run says so; a checkpoint
from an *interrupted* run stops the command instead, because silently
discarding days of compute is the worse mistake. Set `"resume": true` to
continue from that one, or `"rebuild": true` to throw the tokenizer and the
packed tokens away and start from nothing. `"offline": true` skips cloning
entirely, which is what to use once the corpus is on disk and you would
rather not talk to the network again.

Deleting a stage is a supported way to redo it: remove `tokens.bin` to
repack, remove `tokenizer.json` to retrain the vocabulary. Removing
`corpus/` or `wikipedia/` fetches that source again — and removing one
shard out of `wikipedia/` fetches just that shard.

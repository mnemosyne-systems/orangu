\newpage

# Building a model

`orangu-gguf` builds a model file. It has two jobs, and they are the two
ends of one pipeline:

- **Train one from nothing.** Given a JSON manifest of
  permissively-licensed repositories, it clones them, trains a tokenizer on
  what it finds, packs the text into tokens, pretrains a transformer from
  random weights, and writes the result as a GGUF file that
  `orangu-server` serves directly.
- **Convert one you already have.** Given an existing full-precision model,
  it writes it out again at a smaller weight format — `Q6_K`, `Q4_K_M`,
  or any of the dozen in *Weight formats* below.

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
not mention takes a documented default. `smoke` finishes in minutes and
exists to prove the whole pipeline works on your machine before you commit
a week to it. Then serve what came out:

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

  "training_size": "2b",
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
| `training_size` | `2b` | `smoke`, `0.5b`, `1b` or `2b` |
| `context_size` | `256k` | The context length the model declares. Accepts `8192`, `8k`, `1M` |
| `quantization` | `bf16` | The weight format written |
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
A step is seconds at `smoke` and minutes at `2b`, so a count of steps is a
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
quantizations. Its own `README.md` covers the settings and what each stage
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

The four stages, in order:

```sh
cd contrib/orangu-model
./00-smoke.sh       # the whole pipeline, on 20 MB, in minutes
./10-bf16.sh        # the real run, entirely per corpus.json
./20-q6_k.sh
./30-q4_k_m.sh
```

The scripts hold no settings of their own — everything is in the two
manifests, so changing the run means editing `corpus.json` rather than
hunting through shell. `00-smoke.sh` uses `corpus-smoke.json` — the four
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
| `-m`, `--model` | Convert this file instead of training |
| `-ts`, `--training-size` | Override the manifest's size for this run |
| `-q`, `--quantization` | Override the weight format written |
| `-cs`, `--context-size` | Override the declared context length |
| `-o`, `--output` | Override where the model is written |
| `--list-quantizations` | Print the weight formats and exit |

The four overrides exist for the one-off — trying `0.5b` before committing
to `2b`, or writing a second quantization of a model you already have.
Anything you want to keep belongs in the manifest, where the next run will
still find it. `-ts` and `-cs` are two letters behind a single dash, which
most tools read as two separate short flags; they are accepted here anyway,
in both the `-ts 1b` and `-ts=1b` spellings, alongside the long forms.

## Sizes

Four sizes, sharing one vocabulary, so a tokenizer trained once carries
across all of them. The parameter counts are for the default 32,768-token
vocabulary; a smaller one takes two tensors down with it, which is most of
what `smoke` is.

Every hidden and feed-forward width is a multiple of **256**, and that is a
requirement rather than a preference: 256 is the K-quants' super-block, and
a row length it does not divide cannot be a K-quant at all. The
feed-forward *down* projection is the one that decides it, because its row
length is the feed-forward width rather than the hidden one.

| Size | Hidden | Feed-forward | Blocks | Heads | KV heads | Parameters |
|:---|---:|---:|---:|---:|---:|---:|
| `smoke` | 256 | 768 | 4 | 4 | 4 | ~20M |
| `0.5b` | 1280 | 3584 | 24 | 10 | 5 | ~0.53B |
| `1b` | 1536 | 5632 | 28 | 12 | 6 | ~1.0B |
| `2b` | 2048 | 8192 | 30 | 16 | 8 | ~2.0B |

**These are real training runs, and they cost what training costs.**
Everything runs on the CPU in 32-bit floating point. `smoke` is minutes;
the three real sizes are days to weeks of continuous compute on a corpus
large enough to be worth it, and they need memory to match — training holds
four numbers per parameter (the weight, its gradient, and the optimizer's
two moments), so a `2b` run needs roughly 32 GB before activations.
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
| `q2_k` | ~3.2 | The smallest file worth writing. Expect to notice. |

`--list-quantizations` prints the same table with a line on each. Bits per
weight are approximate and for a whole file, mixture included — which is why
`q3_k_l` and `iq4_xs` land so close together, and why every K-quant reads
higher than its own block would suggest. The exact figure depends on how
much of a model is vocabulary.

### What the suffix means

A `Q4_K_M` file is not a file of `Q4_K` tensors. The `_S`, `_M` and `_L` are
*mixtures*: which tensors are carried above the file's base format, and by
how much. They are the standard spellings — a model published as `Q4_K_M`
anywhere means the same mixture this writes — and the tensors that earn the
extra bits are always the same ones:

- **The vocabulary projection** (`output.weight`) is six bits in every
  quantized file. It is the last thing before the softmax, where an error
  becomes a wrong token directly.
- **The value and down projections** are reinforced on the outer blocks,
  and on every third block in between. An error there lands on every token
  that attends through them.
- **Norms are never quantized**, in any format. They are the divisor of
  every activation and a rounding error's worth of file size.

`_S` reinforces least, `_M` more, `_L` most; `Q6_K`, `IQ4_NL` and `IQ4_XS`
have no variants because there is nothing above them worth spending on.
Grouped-query attention shifts the rules slightly: it makes the value
projection small, so carrying it high costs almost nothing, and the mixtures
spend there whenever the model has four or more query heads per key/value
head. `orangu-server show --tensors` prints what a finished file actually
contains.

### What is not written, and why

The format defines more types than this writes. Two groups are left out
deliberately.

The **round-number quantizations** — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
`Q8_0` — are the generation before the K-quants. A K-quant beats them at
every size, and offering both mostly offers a way to pick the worse one.
`Q6_K` is `Q8_0`'s size and more accurate. Asking for one says so rather
than failing as unknown.

The **very low bit I-quants** — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`,
`IQ2_S`, `IQ3_XXS`, `IQ3_S`, and `Q2_K_S` — are a search against a fixed
codebook of lattice points. Below about three bits that search only lands
anywhere useful when it is told which weights matter, which means an
*importance matrix*: a measurement of how much each weight moves the output,
taken by running the finished model over calibration text. This tool does
not measure one yet, and a file written without it looks fine and answers
badly. They are refused with that reason, not written badly. `IQ4_NL` and
`IQ4_XS` are in the same family but wide enough not to need it, so they are
written.

You may also see `_XL` on models published elsewhere. It is not a standard
suffix: it belongs to one vendor's dynamic quantizations, which are derived
from an importance matrix, and it is not written here for the same reason.

Converting works from a full-precision file only. Quantizing an
already-quantized model would round twice and produce something worse than
quantizing the original once, so it is refused with a message saying which
file to start from instead.

## What comes out

A GGUF file with a dense decoder architecture: grouped-query attention with
rotary positions, RMSNorm, a SwiGLU feed-forward, and a per-head norm on
the queries and keys. `orangu-server` serves it on its fully-supported
dense path, and so does anything else that reads the format.

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

\newpage

# Model building internals

This chapter is the developer's view of `orangu-gguf`
(`src/bin/orangu-gguf/`) — how the pipeline is put together, which
decisions are load-bearing, and what the tests are actually proving. The
user-facing chapter is *Building a model*.

## The modules

| Module | What it owns |
|:---|:---|
| `main.rs` | The command line, and the two pipelines (train, convert). |
| `manifest.rs` | The JSON manifest: every setting of a run, its default, and the licence policy. |
| `corpus.rs` | Cloning, and the walk that decides what counts as training text. |
| `wikipedia.rs` | Prose: which Wikimedia dump, and streaming article text out of it. |
| `vocab.rs` | Byte-level BPE: training the merges, and encoding with them. |
| `pack.rs` | Corpus text to a token stream on disk, and reading windows back. |
| `model.rs` | The architecture: sizes, parameter layout, forward, backward. |
| `train.rs` | AdamW, the schedule, the loop, and the checkpoint. |
| `quant.rs` | Float encodings, block quantizations, and the per-tensor type rules. |
| `write.rs` | The GGUF writer. |

Each stage's output is a file in the work directory, and each stage reads
the previous stage's file rather than a value passed to it. That is what
makes the pipeline restartable at any point, and it is why the stages have
no shared state to speak of.

## The manifest is the interface

`Manifest` in `manifest.rs` is not a list of repositories with a few extras
bolted on — it is the whole description of a run, and every field but
`repositories` carries a `serde` default. That has two consequences worth
keeping:

- **The defaults live in exactly one place.** Not in `clap`, not in the
  scripts, not in the manual. A `default_*` function per field is verbose
  and it is the reason `a_manifest_of_material_alone_takes_every_default`
  can assert the whole set at once.
- **`deny_unknown_fields` is load-bearing.** A misspelled key that parsed
  and was ignored would train the default for a week and report success.

The command line keeps four overrides and the convert mode, and `build`
applies those overrides *onto* the manifest before anything reads it — so
there is one description of the run downstream, not a manifest plus a
parallel set of arguments that could disagree with it.

## Two kinds of corpus root

`corpus::Root` carries a per-directory file-size cap, and the reason is a
bug worth remembering. The cap — skip anything over a megabyte — is a
heuristic about *repositories*, where a large file is a minified bundle or
a generated table. Applied to the prose this tool downloads itself it threw
away every Wikipedia shard, silently: the run reported "1 too large" and
trained on a corpus a twentieth the size it should have been.

Cloned repositories get `Root::repository` and the cap; the prose
directory gets `Root::generated` and no cap. The prose also lives *beside*
`corpus/` rather than inside it, so the two roots never overlap and the
offline path (which walks `corpus/` whole) cannot pick prose up under the
wrong rules.

## The architecture, and why this one

`ARCHITECTURE` is `qwen3`: a dense block of grouped-query attention with
rotary positions, RMSNorm, and a SwiGLU feed-forward, plus a per-head
RMSNorm on the queries and keys before the rotation.

It is not the plainest option, and both of its departures from the plainest
option are the reason it was chosen:

- **The query and key norms** hold attention logits in range through the
  early steps of a from-scratch run, which is exactly when a model trained
  without them diverges.
- **Grouped-query attention** is what makes the declared context
  affordable. The KV cache is proportional to the number of *key/value*
  heads, so eight instead of sixteen halves it.

It is also on the inference side's fully-supported dense path, which means
a file this tool writes is served without a conversion step and without a
new code path to keep working. Note that the rotation pairing follows from
the architecture name: everything but the original `llama` name is read
back with the NeoX pairing (index `i` against `i + dim/2`), and `rope` in
`model.rs` applies exactly that.

## Forward and backward

There is no autodiff framework here. Each operation's backward is written
beside its forward in `model.rs`, and the whole network is one flat `f32`
buffer (`Layout`) rather than a tree of tensors — so the gradient, and both
of AdamW's moments, are arrays of exactly the same length, and none of them
needs to know the architecture.

**Activations are recomputed, not stored.** The forward pass keeps only
each block's *input*, one `[tokens, hidden]` buffer per block; the backward
pass re-runs `layer_forward` on that input to get the intermediates it
needs. The cost is one extra forward pass. What it buys is an activation
footprint that does not grow with the feed-forward width — with the width
at four times the hidden size and three matrices through it, storing them
would dominate everything else.

The attention backward goes further and recomputes the probabilities too,
because the alternative is a `[heads, tokens, tokens]` array that grows
with the *square* of the sequence length and would be the largest thing in
the process.

### The test that matters

`gradients_match_finite_differences` takes the largest-magnitude gradient
entry in *every* parameter tensor and compares it against a central
difference of the loss. It is the only thing standing between a wrong
derivative and a training run that looks healthy and learns nothing — a
wrong backward does not crash, it just converges to something worse, over
days.

Two details of that test are deliberate:

- The step is `1e-3`. A central difference has two error terms pulling in
  opposite directions: truncation, which falls with the square of the step,
  and the floating-point noise of subtracting two nearly equal losses,
  which grows as the step shrinks. At `1e-2` the truncation term alone put
  the embedding gradient 5% out.
- The tolerance is absolute *plus* relative. The norms' gradients are
  genuinely tiny, and for those the noise floor — `f32::EPSILON * loss /
  step` — is the whole budget.

`attention_is_causal` is the other structural check: truncating the input
before an edited position must leave the loss bit-identical, which it can
only do if no earlier position ever saw a later one.

## The tokenizer's one hard constraint

The vocabulary is trained with the *same* pre-tokenizer split that a reader
applies. `SPLIT_PATTERN` in `vocab.rs` is the generic pattern, and the file
declares `tokenizer.ggml.pre = "gpt-2"`, which is what the reader resolves
to that pattern.

Getting this wrong does not fail loudly. It produces merges that the
encoder can never reproduce, so every prompt tokenizes into sequences the
model was not trained on, and the model appears merely bad. If the split
pattern on either side ever changes, they change together or not at all.

`Splitter` applies that pattern by hand rather than by DFA, because it is
one fixed expression run over every byte of the corpus twice. It is a
scanner for exactly the pattern's four classes and five alternatives — and
it is only a scanner where it can be sure: `\p{L}` outside ASCII is a
Unicode table, not a range, so any match that could involve a non-ASCII
byte is handed back to the pattern itself and the scan resumes after it.
The two must agree byte for byte, and three tests say so, one of them over
whatever corpus is on the machine.

Encoding merges by rescanning for the lowest-ranked pair — quadratic, and
the fastest thing there is at the length of a word. Past `RESCAN_MAX` bytes
it switches to a heap over a linked list, which is the same algorithm the
engine's tokenizer uses. Both paths must produce the same ids; a test runs
the same input through each.

The merge loop keeps pair counts incrementally with a lazily-validated
heap: a pair's count changes as other merges consume it, so the heap holds
stale entries by design and a popped entry is checked against the current
count before it is used. When a word is rewritten, its *whole* pair
multiset is subtracted and the new one added — doing it wholesale rather
than per-occurrence is what keeps a repeated symbol (`aaa` merging `aa`)
correct without a special case.

## The matmul is load-bound, not bandwidth-bound

Worth knowing before optimizing it, because the two want opposite things.

The kernel issues two loads for every fused multiply-add — a vector of
weights and a vector of activations — and the core retires two loads and
two FMAs a cycle. The loads are the ceiling; half the arithmetic capacity
sits idle. So `tile` holds each weight vector across four token rows, which
makes it ten loads for eight FMAs instead of sixteen, with two accumulators
per row to keep enough dependency chains in flight. Four rows is measured:
eight rows has a better ratio still and falls off a cliff on wider shapes,
where it runs out of registers and chains together.

The evidence that it is not bandwidth is in `bf16_versus_f32_matmul`: the
weight matrix in `bf16` is half the bytes and takes the same time, and an
8 MB weight matrix runs at the same rate as a 256 KB one. Narrowing
anything to save memory traffic is solving a problem this kernel does not
have.

## Reductions run in a fixed order

A float sum is not associative, so a reduction whose grouping depends on
which thread stole what gives an answer that depends on it too. Two places
in the training loop reduce across the parallel split — the gradient norm,
and the weight gradient of every RMSNorm — and both sum partials by index
rather than through `par_iter().sum()` or `fold`/`reduce`, whose leaves are
whatever the work-stealing left behind.

This is not fastidiousness. The gradient norm scales every gradient through
the clip, so a difference in its last bits is a difference in the weights on
the next step, and compounding by the one after: the same seed used to give
a different model on the same machine. The norm backward's groups are a
fixed 32 rows for the same reason — a thread-count dependent split would
make the answer depend on the thread count.

The test that this holds is external: train ten steps and hash the written
GGUF, at one thread and at sixteen.

## The mixture rules

`Q4_K_M` is not a file of `Q4_K` tensors, and `plan_tensor` in `quant.rs`
is where that mixture is decided. Taking `Q4_K_M` as the example:

| Tensor | Type |
|:---|:---|
| any 1-D tensor (every norm) | `F32`, in every file type |
| `output.weight` | `Q6_K`, in every quantized file type |
| `token_embd.weight` | the file's base type |
| `attn_v` / `ffn_down` in an outer block, or every third one between | `Q6_K` |
| everything else | `Q4_K` |

"Outer block, or every third one between" is `use_more_bits`: the first
eighth, the last eighth, and every third block in the middle. Spending the
extra bits there rather than spreading them evenly is what the M is; `_S`
spends less and `_L` more, and `mixture` holds one arm per file type
saying exactly where.

`plan_tensor` takes a `Model { layers, gqa }` rather than a layer count
alone, because two of the rules key off grouped-query attention: when there
are four or more query heads per key/value head the value projection is
small enough that carrying it above the base type is nearly free, and the
mixtures take that trade.

A block quantization needs its block to divide the row length, and a row
that does not divide 256 cannot be a K-quant at all. `fit` demotes it —
four bits and below to `IQ4_NL`, the one 32-wide type here; five and six to
`F16`, because dropping them to four would give up the precision they were
chosen for — and records what it demoted from, so the run *reports* the
fallback rather than silently writing a different file than the name
promises.

### The types that are refused

`Ftype::parse` distinguishes three kinds of no. A round-number
quantization (`Q8_0` and its family) is refused because a K-quant of the
same size is better. `IQ1_*`, `IQ2_*`, `IQ3_XXS`, `IQ3_S` and `Q2_K_S` are
refused because at that width the codebook search needs an importance
matrix to land anywhere useful, and nothing here measures one — writing
them anyway produces a file that loads, serves, and answers badly, which
is worse than not writing it. Everything else is refused as unknown, with
the list of what is not.

Quantizing an already-quantized file is refused rather than supported.
Rounding twice is worse than rounding once from the original, and a tool
that quietly does the worse thing is a trap.

## Finding out where the time goes

Two instruments, and they answer different questions. Using the wrong one
is how a stage that costs a quarter of a training run stays invisible.

### `--flamegraph`: which code the CPUs are in

```sh
orangu-gguf manifest.json --flamegraph run.svg
```

Samples this process for the whole build — corpus, tokenizer, packing,
training and the write — and renders the flamegraph, the folded stacks
beside it, and a PNG with `--flamegraph-png`. There is no shell pipeline
and no external script: `perf record` is the only outside program, and
everything after it is `orangu::profiling`, which `orangu-bench` uses for
the same job on a running server.

A manifest with no steps left to run therefore profiles corpus
preparation, and one with steps profiles training — which is how the two
are told apart without a flag that could disagree with the manifest.

The thread pool is warmed before sampling starts, deliberately. `perf
record -p` attaches to the threads that exist at that instant and never
picks up ones created later, and rayon builds its workers on first use —
so on a run whose corpus is already packed, a recorder started any earlier
would sample the main thread and nothing else, and still produce a
perfectly confident-looking flamegraph of a program doing one thing at a
time.

### `ORANGU_GGUF_STAGES`: where the seconds go

```sh
ORANGU_GGUF_STAGES=1 orangu-gguf manifest.json
```

prints a table after the run: seconds in each stage of a training step,
its share of the run, and how many times it was entered. Each stage is
timed on the calling thread, around its parallel region rather than inside
it, so what it reports is elapsed time — and the total comes back within a
few percent of the run.

**This is the one that finds a narrow stage.** A profiler counts where the
CPUs are, so a stage running on one thread while fifteen sit idle costs a
sixteenth of the samples it costs in seconds. Both of the largest wins in
this tool's history were invisible to the flamegraph and obvious here: a
kernel at 9.6% of samples and 24.7% of the clock, and a serial loop at
half a percent of samples and 5.4% of the clock. Read the two together —
the profiler says which code, the table says whether it is slow because it
is *slow* or slow because it is *narrow*.

## The writer's invariant

Every tensor's data offset follows from the types and shapes alone, so the
whole header can be written before any weight is encoded — which is what
makes writing a file larger than memory ordinary rather than special. The
write loop asserts, per tensor, that the bytes written so far match the
offset the header recorded. A header that disagrees with the data by one
padding byte produces a file that reads back as plausible noise, and that
is a bad thing to find out about later.

## The checkpoint

`ORANGUCK`, a version, the step, the config, then the parameters and
AdamW's two moments. It is written to a temporary file and renamed into
place, so an interrupted write cannot replace a complete checkpoint with
half of one.

Loading refuses a checkpoint whose config differs from the run's in any
field. Resuming into a mismatched architecture would produce weights that
are numerically fine and completely meaningless.

## Extending it

- **A new size**: add a `Size` to `SIZES` in `model.rs`. The hidden width
  must be the head count times the head width, and the head count must
  divide by the key/value head count — `the_named_sizes_are_the_sizes_they_claim`
  checks both. `tiny` and `smoke` are not two names for the same idea:
  `tiny` is the shape the unit tests use, and `smoke` is the shape a
  change is *measured* on, which is why it has grouped attention and
  matrices past a last-level cache.
- **A new weight format**: add an `Ftype` variant, its `file_type` number,
  its per-tensor rule in `plan_tensor`, a block encoder, and its fallback
  in `fit`. The round-trip test's tolerance should be set just above the
  error the encoder actually achieves, not loosely enough to hide a
  regression.
- **A new architecture**: `Layout::new` and `layer_forward`/`layer_backward`
  are the three places that would need to agree, plus the metadata keys in
  `model_metadata`. The gradient test covers whatever `Layout` lists, so a
  new tensor is checked the moment it exists.

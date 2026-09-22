\newpage

## Benchmarking throughput (`orangu-bench`)

`orangu-bench` (`src/bin/orangu-bench/`) is a **developer tool** — a fourth
binary in the same Cargo package as `orangu`, `orangu-coordinator`, and
`orangu-server`. It is not part of the served product and has no bearing on
running a model in production; it exists to answer one question during
performance work: *how fast does token generation (decode) run, how does that
rate change as the context grows, and how fast is prompt processing
(prefill)?*

Rather than embedding an inference engine, it points at a **running
OpenAI-compatible server** over HTTP and measures the tokens per second it
streams back. Any server that speaks `POST /v1/completions` with SSE streaming
is measured through the *same* path by the *same* tool — which is the only way
to get a genuinely apples-to-apples comparison between two engines, since an
in-process benchmark on one side and an ad-hoc `curl` on the other are not
comparable.

### What it measures

For each run, `orangu-bench` sends one streaming completion and reports
steady-state decode throughput — the standard token-generation (`tg`)
quantity, with prompt processing (prefill) and time-to-first-token *excluded*.
Time to first token is printed separately (`ttft_ms`) so prefill cost is still
visible.

The rate comes from the server's own `predicted_n` / `predicted_ms`, which
already exclude prefill. It is **not** derived by counting streamed chunks,
because that under-reports: a generated token whose text is empty — a special
token the server filters out of the stream, a partial UTF-8 or BPE
continuation that decodes to nothing on its own — costs a full forward pass and
arrives as no visible text, while the elapsed window still spans it. Measured
on `gemma-4-E2B-it:Q4_K_M` at depth 1024, 128 generated tokens streamed as 79
visible ones, and counting chunks read 27.6 tok/s where the server had done
44.5 — a 38% under-read that looked exactly like a decode cliff. Counting
chunks, `(tokens - 1) / decode_seconds`, remains the fallback for a server that
reports no `timings` block at all; the `n_tok` column shows which count was
used.

**Profile decode at depth 0.** `--flamegraph` brackets the measured window, and
that window includes each repetition's prefill. At depth 1024 with `--gen 256`
prefill is 6.4 s against 5.8 s of decode, so half the profile is prefill and it
reads like one. `--depths 0 --gen 512` puts 0.5 s of prefill against 12 s of
decode.

To see how decode scales with context, it sweeps **depths**: each depth pads
the prompt with filler so generation begins at roughly that many tokens of
context. A flat curve across depths means decode is
context-insensitive; a curve that falls with depth means attention or KV
traffic is growing per token.

#### Ranges

Anywhere a list of points is accepted — `--depths`, `--pp`, `--pg`,
`--pp-continue`, `--embed`, `--image`, `--streams` — an item may be a range
instead of a
number, in the three forms the wider ecosystem's benchmarks use, so a sweep can
be copied between them:

| form | means | example |
| :-- | :-- | :-- |
| `first-last*mult` | multiply until past `last` | `128-2048*2` → 128, 256, 512, 1024, 2048 |
| `first-last+step` | step until past `last` | `0-2048+512` → 0, 512, 1024, 1536, 2048 |
| `first-last` | every value, stepping by one | `1-8` → 1, 2, 3, 4, 5, 6, 7, 8 |

Ranges and numbers mix: `--depths 0,128-512*2,3072`. The end is a bound, not a
member — `128-3000*2` stops at 2048.

A range is expanded **before the first request**, so a mistyped one costs
nothing: `2048-128` ("ends before it starts"), `128-2048*1` ("would never reach
its end") and `128-4096` ("expands to more than 256 points — did a doubling
sweep lose its `*2`?") are all refused with their reason rather than run. That
last cap is deliberate: `128-4096` is legal and means 3969 measurements, which
looks exactly like a doubling sweep missing its multiplier.

#### Letting the card cool (`--delay`)

`--delay <seconds>` waits between measured points — between them only, never
before the first or after the last. On a laptop card a sweep that heats through
its own run reports a falling curve that looks exactly like the effect these
sweeps are run to find. The clock and DPM state printed in the header make the
cause visible after the fact; this is what avoids it.

#### Measuring the sampled path (`--temperature`)

`--temperature <T>` sets the sampling temperature of the timed decode; the
default `0` is greedy. A server takes a different path for a sampled step —
the device hands back the sampler's candidates instead of one token, and the
host draws from them — and a client that sends no temperature gets the
server's own default, which is sampled. So a decode number for the path such
a client actually runs is taken with `--temperature 0.8` (or whatever the
server's default is), not with the greedy default.

> The depth padding is approximate — it appends `~depth` filler words
> (≈ one BPE token each) rather than exact tokens, because the tool has no
> tokenizer and talks only HTTP. It is close enough to compare *slopes*
> between two engines or two builds; it is not an exact context length.

### Usage

Start the server you want to measure, then run the tool against its base URL.

```sh
# orangu-server (default port 8100): sweep decode rate across context depths
orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024,2048,3072 --gen 128

# another OpenAI-compatible server on port 8300, identical harness
orangu-bench --url http://127.0.0.1:8300 --depths 0,512,1024,2048,3072 --gen 128
```

Typical output (one row per depth):

```
orangu-bench → http://127.0.0.1:8100
   depth |   gen | ttft_ms |    n_tok |     best |        mean ± sd
-------------------------------------------------------------------
       0 |   128 |     140 |      128 |    31.20 |    31.05 ±  0.12
    1024 |   128 |     520 |      128 |    24.90 |    24.70 ±  0.18
    2048 |   128 |     980 |      128 |    20.10 |    19.95 ±  0.20
```

### The web console (`--web`)

Everything below is also available from a browser. `--web` serves a console —
the same vanilla HTML/CSS/JS, embedded in the binary with no build step, that
`orangu-server`'s own web UI is built from, and deliberately the same
palette — where a run is **defined in a form** and its result comes back as a
**summary table**, a **flamegraph** and a **chart**:

```sh
orangu-bench --web                        # http://127.0.0.1:8300
orangu-bench --web --port 9000
orangu-bench --web --host all             # reachable from another machine
orangu-bench --web --host 192.168.1.10    # or one specific interface
```

```text
orangu-bench 1.2.0
Console  http://127.0.0.1:8300
Runs     /home/you/.orangu/orangu-bench/runs
```

The banner is one aligned table: labels padded to a common width, and a value
too long for the terminal continued under its own column rather than back at
the left margin. A value is only ever broken at a space, so an address or a
path stays in one piece — an address split across two lines cannot be copied,
and copying it is what the line is for. Bound off loopback there is a third
row, `Note`, saying so.

Pick the measurement (decode, prefill, combined, continuation prefill, curve,
concurrency, embeddings, decode CPU), give it the server URL and its sweep
points, and press **Run benchmark**. The console then shows, live:

- the tool's own output as it measures, one row per point as the row is taken,
  with anything it wrote to stderr — a retried send, a profile that
  undercounted — kept visibly apart from the measurements;
- a **summary table**: best, mean and ± sd per point, above the model, backend,
  server pid/uptime and GPU clock state that were live *while measuring*;
- the **flamegraph**, in the page and interactive (click a frame to zoom, the
  title to reset, Ctrl-F to highlight), with **both `.svg` and `.png`** to
  download, and the sample count, window, cores-busy and GPU-wait share beside
  it — a flamegraph is normalised to its own total, so only those numbers say
  how much time there was to divide;
- the **chart**, likewise as `.svg` and `.png`;
- a **Report** button beside the summary table, which builds the whole run as
  one PDF (see *The run as a document* below) and saves it. It is built on the
  click rather than by every run: most runs are never sent anywhere, and a
  document written every time is a directory full of PDFs nobody opened.
  Everything it needs is already archived, so it can be built at any time —
  including from a run measured weeks ago;
- the run's `bundle.json`, which is what all of the above was read back from.

Every image and every text pane carries a **save control at its lower right**
— the same dimmed footer with the same icon `orangu-server`'s console puts
under an answer and under a rendered diagram. What lands on disk is the
byte-identical file the run wrote, named after the run (`orangu-bench-<id>-
flamegraph.png`), so two arms of an A/B saved from two tabs do not both arrive
as `flamegraph.png`.

A run's directory holds what cannot be regenerated from it: the bundle, the
chart and flamegraph (SVG, their PNG twins, and the collapsed `.folded` a
profile can be re-rendered or compared from), the log, and the console's own
`run.json`. The PDF is built on demand and the comparison's working copies are
deleted once the comparison is made.

Every run gets a directory under `~/.orangu/orangu-bench/runs/<id>/`, so a
result outlives the console process that produced it: the **Past runs** button
lists them newest first, and opening one re-loads its table, its artifacts and
the form that produced it — which is how a second A/B arm gets defined, one
field away from the first.

The topbar's **New** and the history panel's **Clear all** are
`orangu-server`'s console's own, doing the same thing to runs that they do to
chats there. **New** empties the result pane and lets go of the run it was
showing — deliberately leaving the form alone, since the next run is almost
always the last one with one field changed, which is exactly what an A/B arm
is. It is remembered, so a reload after it comes back to an empty pane rather
than reopening the newest run; a row's own **×** does the same for one run.

**Clear all** deletes every kept run. A run that is still measuring is **not**
one of them: it is kept, named in the reply, and the console follows it —
tidying a list of finished results should never be the thing that ends a
twenty-minute sweep, and Cancel is one button away and says what it does.

#### Scaling tests: the sweeps this project actually ran

Under **Measurement** sits **Scaling test**, a second drop-down whose entries
are named for the range they cover — "Prefill" plus "128 to 3072". Choosing one
fills in the points, the token count and the repetitions, and locks them, so
the sweep about to run is on screen rather than implied. The default is
**None**, which leaves every field free: a one-off measurement, or an A/B
against one hand-picked depth, is not a scaling test.

The ranges are not round numbers. Each one is the sweep this repository used to
find what it found, so a run lands beside an existing baseline rather than
beside nothing:

| Measurement | Scaling test | Where it comes from |
| :-- | :-- | :-- |
| Decode | `0 to 2048` | the tracked series in `perf-history.tsv`; `PERF-GAP.md`'s standard harness, best of 3 |
| Prefill | `128 to 3072` | the tracked `pp` series — every recorded prefill row came from these lengths |
| Combined | `128 to 3072` | the tracked prefill lengths with the tracked decode length, timed as one turn |
| Continuation prefill | `10 to 130 added` | `PERF-GAP.md` increment 7, the sweep that found the 2× cooperative-GEMM cliff between 50 and 66 tokens |
| Decode curve | `0 to 3072 in one pass` | the curve invocation this manual documents below |
| Concurrency | `1 to 8 streams` | `PERF-GAP.md` item 7 — 99% engine occupancy at two streams against a generic path stuck at 66% with eight |
| Embeddings | `64 to 256` | `embeddinggemma-300M`'s own sweep, 15 reps |
| Decode CPU | `0 to 1024` | the depths that separated a claimed +58% CPU-per-token growth from the real +8.8% |

#### Comparing against an earlier run

The question a benchmark is usually run to answer is not "how fast is this?"
but "is this faster than what I had?" — so every finished run offers a
**Compare** panel listing every earlier run this console has kept. Pick one and
the console runs `orangu-bench --read-bundle old,new` against the two archived
bundles and shows what it prints:

- **what differed** between the two configurations — model, backend, kernels,
  host, GPU clocks — because a throughput comparison is only readable once you
  know whether the two runs were the same experiment;
- **where the GPU time went** per decode stage, when both runs recorded it;
- **what each measured**, point by point, with the percentage against the older
  run;
- **one chart holding both**, saveable as SVG and PNG like every other image;
- **`compare.pdf`** — the same comparison as a document, which is the artifact
  most likely to be attached to a pull request.

The two bundles are copied into the newer run's directory as `old-<id>.json`
and `new-<id>.json` before being compared, and the copies' series labels are
tagged `old ·` / `new ·`. Both halves matter: every run's bundle is called
`bundle.json`, so the table's two columns would otherwise both read "bundle",
and two runs of the same server carry the same series label, so the chart would
draw them as one line. The runs' own bundles are not touched.

Comparing is allowed while a benchmark is running, unlike starting a second
run: it reads two files that were written when their runs ended and talks to no
server, so it cannot disturb a measurement in flight.

Two things it does on purpose:

- **One run at a time.** A second run pressed while one is going is refused
  rather than queued or allowed: two benchmarks sharing a server measure each
  other's interference.
- **It runs `orangu-bench`, it does not reimplement it.** Each run re-executes
  this same binary with the flags the form describes — the first line of the
  log is that command line, quoted so it can be pasted straight into a
  terminal. A console with its own copy of the harness could disagree with the
  command line about the same workload, and a benchmark tool that gives two
  answers for one question is worse than none.

The console is **unauthenticated**, like `orangu-server`'s, and assumes a
trusted network. It differs in its default: loopback, where that console
defaults to every interface — because this one starts processes rather than
answering requests. `--host all` (or `*`, or a literal interface address)
opens it up for the case that wants it, which is real and common in
performance work: the machine with the GPU is rarely the machine you are
sitting at. Bound anywhere but loopback it says so at startup, because from
then on anyone who can route to the port can start runs on that machine.

`--sweep` is the one mode with no place in the UI at all: a sweep's whole
input is a shell command to start a server with, and this console never takes
one.

`--per-rep` adds a line under each decode row with every repetition's
rate in the order it ran. The best and the mean each hide a first
repetition well under the rest — a server still settling after its
load, a card whose clock has not ramped — and a row whose repetitions
read `2.1 / 10.8 / 10.9` is a different finding from one that reads
`7.9 / 7.9 / 8.0` with the same mean.

### Prefill mode (`--pp`) — prompt processing, not decode

`--pp` sweeps *prompt lengths* and reports **prompt-processing** throughput —
the standard `pp` test to the default's `tg`. Each run sends a prompt of
roughly the requested length and generates a single token, so what is timed is
prefill and nothing else.

```sh
orangu-bench --url http://127.0.0.1:8100 --pp 128,512,1024,2048
```

```
orangu-bench → http://127.0.0.1:8100
  model    unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
  backend  Vulkan/AMD Radeon RX 5500M (RADV NAVI14) (Vulkan)
  gpu      card1 sclk 1700Mhz (auto)
      pp |   n_tok |  cached | prompt_ms |     best |        mean ± sd
----------------------------------------------------------------------
     256 |     288 |       0 |    6021.7 |    47.83 |    44.73 ±  3.10
    1024 |    1120 |       0 |   24037.8 |    46.59 |    44.35 ±  2.24
```

Two columns exist to keep the number honest:

- **`n_tok`** is the prompt length the *server* reported (`timings.prompt_n`),
  not the length that was asked for. The `--pp` value is only a target, since
  the tool has no tokenizer — the rate is computed from what was actually
  processed.
- **`cached`** is how much of the prompt came from the server's prefix cache
  and so never went through a forward pass. A cached prompt "prefills"
  instantly and would otherwise look like a spectacular result. Every run
  sends `cache_prompt: false`, which any server implementing a prefix cache
  honours, so this column should read `0`; if it ever climbs toward `n_tok`,
  the server ignored the flag and the row is measuring a cache lookup rather
  than prefill.

A server that reports no `timings` at all (an older `orangu-server`, or one
built without them) gets a row marked `no server timings (ttft only)`, carrying wall-clock time-to-first-token instead. That figure includes
queueing and the first decode step, so it is not comparable with the rest — the
row says so rather than quietly printing a smaller number.

### Combined mode (`--pg`) — the whole turn, timed as one thing

Every other mode here splits the turn on purpose: `--pp` times prefill with a
single token generated, the default sweep times generation with prefill
excluded. That is what a *diagnosis* needs — the two halves have different
bottlenecks and move for different reasons.

`--pg` answers the other question, the one a user actually experiences: how
long the whole turn takes. It sends one request that prefills a prompt of
roughly the given length and generates `--gen` tokens, times it from before the
send to the last streamed token, and reports `(prompt + generated) / total`.

```sh
orangu-bench --url http://127.0.0.1:8100 --pg 128,512,1024 --gen 128
```

```text
      pp |   n_tok |     gen | prompt_ms |  total_ms |     best |   mean ± sd(n-1)
---------------------------------------------------------------------------------
      64 |      71 |      40 |      12.5 |      63.3 |  1754.89 |  1749.80 ±  7.20
     128 |     146 |      40 |      12.5 |      63.6 |  2951.57 |  2938.98 ± 17.81
```

This figure **cannot be reconstructed** from a `--pp` run and a decode run:
neither carries the queueing and the hand-off between prefill and the first
token, which is exactly the part a combined number includes. It is also the
figure most third-party comparisons quote.

`prompt_ms` is the server's own prefill time, so the split inside the turn
stays visible; `n_tok` is what the server said it processed, and the rate is
computed from that rather than from the requested length. `cache_prompt: false`
as everywhere else — a cached prompt would make the prefill half vanish and
quietly turn this into a decode measurement.

The swept axis is the **prompt**; the generated length comes from `--gen`, so
one invocation is one generation length. That keeps `n` meaning one thing per
mode in the history file, where these rows are recorded under mode `pg`.

### Continuation-prefill mode (`--pp-continue`) — the narrow-batch regime

`--pp` cannot reach small batch widths. It sends whole prompts with
`cache_prompt: false`, and a whole prompt carrying a chat template is hundreds
of tokens before any user text, so every `--pp` row exercises a *wide* batch.
Real multi-turn chat does the opposite: the prefix cache supplies everything but
the newest message, and the server prefills a handful of tokens. That is a
different point on the batch-width curve, and it is the one that batch-width
thresholds actually govern.

`--pp-continue` takes comma-separated **added** token counts. For each, it
primes a base prompt (`--pp-continue-base`, default 512 tokens), then sends
base + extension and reports the rate over the tokens the server says it
actually processed:

```sh
orangu-bench --url http://127.0.0.1:8100 --pp-continue 8,16,32,64,128
```

```
   added |   n_tok |  cached | processed | prompt_ms |     best |        mean ± sd
----------------------------------------------------------------------------------
       8 |     582 |     572 |        10 |     275.9 |    36.24 |    34.74 ±  1.37
      16 |     590 |     572 |        18 |     410.6 |    43.84 |    41.89 ±  1.38
      32 |     606 |     572 |        34 |     616.1 |    55.32 |    54.94 ±  0.44
      64 |     638 |     572 |        66 |     794.9 |    83.03 |    80.70 ±  1.67
     128 |     702 |     572 |       130 |    1213.8 |   107.10 |   105.47 ±  1.20
```

This mode is the one place the tool sends `cache_prompt: true`, because a cache
hit is the entire point. Two things keep that from becoming a lie:

- **`processed`** (= `n_tok` − `cached`) is what the rate is computed from.
  Dividing by `n_tok` would credit the cached prefix with work nobody did and
  inflate every row by the cache ratio.
- **Each rep's extension differs from the last in its *first* token.** A prefix
  cache matches on the longest common token prefix, so two reps that extended
  the base with the same words would let the cache swallow the extension too,
  and the run would time a lookup instead of a prefill.

A row whose `cached` column reads `0` is **dropped with a note** rather than
recorded: no cache hit means the whole prompt was prefilled, which is an
ordinary wide `--pp` row wearing a continuation's label, and averaging the two
together would be meaningless.

Because the token counts are approximate (the tool has no tokenizer), `added`
is a target and `processed` is the truth — history rows are recorded against
`processed`.

### Decode CPU per token (`--decode-cpu`)

Reports the **server's own CPU time per generated token**, read from
`/proc/<pid>/stat`, with prefill excluded. Useful when a change is expected to
remove CPU work rather than wall-clock time — a throughput number can be too
noisy to show a few percent, while CPU seconds are counted by the kernel.

```sh
orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024 --gen 192 --decode-cpu
```

```
   depth |   n_tok | prefilled |  cpu_ms/token |    tok/s
----------------------------------------------------------
       0 |      33 |         1 | 14.427 ± 0.052 |    22.06
     512 |     572 |         1 | 15.313 ± 0.208 |    21.51
    1024 |    1118 |         1 | 15.703 ± 0.286 |    21.46
```

**Why each depth is measured twice.** The obvious implementation — read the CPU
counter around a whole `--depths N` run and divide by the tokens generated — is
wrong, because that run *prefills N tokens first*. Prefilling 1024 tokens costs
several CPU-seconds, and charging them to the generated tokens makes decode CPU
appear to grow 58% with depth when the real figure is under 9%. The growth is
the prefill's, and it scales with depth because the prefill does.

So the first request pays the prefill and leaves it in the prefix cache, the
counter is read after it, and the second request's prefill is a cache hit. The
**`prefilled` column is the check**: it must read `1` (the one token the cache
deliberately re-processes). If it ever shows the whole prompt, the cache did not
hit and the row is measuring prefill again.

This needs the server to be a local process, since it reads `/proc` —
`--flamegraph-pid` overrides the pid if the tool cannot work it out.

Rows are recorded under a third history mode, `cpu`, and charted on their own
panel: the unit is milliseconds and **lower is better**, so putting them on a
tok/s axis would read as a collapse.

Throughput does not need this treatment — the depth sweep already times from the
first streamed token to the last, so prefill is outside its window either way.

### Embedding mode (`--embed`) — the only mode an embedding model answers

An embedding-only server serves `/v1/embeddings` and answers
`/v1/completions` with **HTTP 501**, so the decode sweep and `--pp` both fail
on one outright — warmup included. `--embed` sweeps prompt lengths against
`/v1/embeddings` and reports forward-pass throughput, which is what `--pp`
reports for a generative model: an embedding pass is prompt processing without
the decode that follows it.

```sh
orangu-server embeddinggemma-300M-Q8_0.gguf --embedding &
orangu-bench --embed 64,128,256 --reps 15
orangu-bench --url http://127.0.0.1:8300 --embed 64,128,256 --reps 15  # the reference
```

```
   embed |   n_tok |   wall_ms |     best |        mean ± sd
------------------------------------------------------------
      64 |      81 |    1163.3 |    70.49 |    68.33 ±  1.64
     128 |     159 |    2401.3 |    66.21 |    63.99 ±  1.50
     256 |     289 |    5003.7 |    59.47 |    57.95 ±  1.89
```

`n_tok` is the count from the response's `usage.prompt_tokens` — the length the
forward pass actually ran, not the requested target — which `orangu-server`
and every other conformant server reports. A server that omits it gets a row
marked `no usage.prompt_tokens (latency only)`: an embedding response carries
no other clue to its token count, so there is no rate to print and the row is
not recorded.

Unlike `--pp`, the time is **wall-clock for the whole request**: neither server
attaches a `timings` object to `/v1/embeddings`, so there is nothing to prefer
over the clock. HTTP and the JSON encoding of an `n_embd`-long float array are
therefore inside the number — small beside a forward pass, and identical on
both engines, since the same client sends both.

### Image mode (`--image`) — where a picture's minutes go

A picture is three models in a row — the prompt through the text encoder,
the latent through the diffusion transformer once per step (twice under
guidance), and the latent through the VAE — and a user who typed *Create an
image of a cat* into the console and waited sees one number: the wait.
`--image` takes the same request through the same endpoint
(`POST /v1/images/generations`, streamed) and splits the wait three ways
from the server's own phase timings (the `timings` object on the completed
event: `encode_ms`, `steps_ms`, `decode_ms`), so a slow picture is a slow
*phase* — and, with `--flamegraph`, a slow phase is a hot function.

```sh
orangu-server unsloth/Qwen-Image-2512-GGUF:Q4_K_M &     # comes up in the image role
orangu-bench --image 256 --reps 1 --flamegraph cat.svg
orangu-bench --image 256,512 --image-steps 4 --image-cfg 1 --reps 3
```

```
    image | tokens | steps | encode_s |   step_s | decode_s |  total_s |     best |   mean ± sd(n-1)
--------------------------------------------------------------------------------------------------------
 256x256  |    256 |   1x2 |      6.3 |     43.7 |      6.6 |     56.6 |     11.7 |     11.7 ±     —
          | transformer: mlp 57%  qkv 26%  out 8%  attention 6%  modulation 2%  other 1%
 512x512  |   1024 |   1x2 |     14.9 |    185.8 |     22.0 |    222.7 |     11.0 |     11.0 ±     —
          | transformer: mlp 51%  qkv 23%  attention 17%  out 8%  modulation 1%  other 1%

  rate: image-token passes per second over the denoising loop (latent tokens x transformer passes / step_s x steps);
  encode_s is the prompts through the text encoder, decode_s the VAE and the file
  profile  cat.svg (795026 samples over 96s — 8.27 cores busy, 8.04 per /proc)
           0.00 gpu-wait  0.01 pool-idle  8.26 working (cores)
           top self-time frames:
            80.4%  orangu_server::engine::vecdot::dot_k_pair
             6.6%  orangu_server::engine::vecdot::dot_row_f32
             2.3%  <&orangu_server::engine::image::transformer::joint_attention::{c
             1.8%  <orangu_server::engine::backend::cpu::CpuBackend>::finish_into
             1.7%  orangu_server::engine::vecdot::unpack_k_row
```

That run — the 20-billion-parameter transformer on twelve ARM cores, no
GPU — is the answer to "why did *Create an image of a cat* never come
back". The rate is flat at 11–12 token passes a second from 256 to 1024
tokens, so the transformer is compute-bound in its linears already at the
smallest size: `mlp` + `qkv` + `out` are nine tenths of every pass, and the
profile puts 80% of the samples in the Q4_K dot kernel under them, with 8.3
of twelve cores busy. At the console's defaults — 1024 pixels, fifty steps,
guidance on — that is 4096 tokens × 100 passes = 409,600 token passes, or
close to ten hours at this rate; the text encoder and VAE together are
under a tenth of the wall time and grow slower than the loop. What moves
the number is the kernel and the cores it runs on (a GPU backend, or the
four idle cores), not the pipeline around it; what moves the *wait* is
`image_size`, `image_steps` and `image_cfg_scale = 1`, in that order.

`tokens` is the transformer's sequence: one latent token per 16×16 pixels
(the VAE's 8× compression times the transformer's 2×2 patch), so a 256-pixel
square is 256 tokens and a 1024-pixel one is 4096. `steps` says how many
times the transformer ran over them, `x2` when the server ran guidance (two
passes per step, one per prompt). `encode_s` is the prompts through the text
encoder — both of them under guidance — `step_s` one denoising step,
`decode_s` the VAE and the file format, and `total_s` the request from send
to `[DONE]`: what the console's user waited.

The recorded rate — `best`/`mean`, what `--history` and the chart keep —
is **latent-token passes per second over the denoising loop**: `tokens ×
passes / (step_s × steps)`. It is to a picture what prompt-processing tok/s
is to a chat turn: how fast the transformer chews through positions,
independent of how many the request asked for, so a 256-pixel run at two
steps and a 1024-pixel run at fifty sit on one chart.

**Two steps, not fifty.** Every step is the same work, so two of them say
what fifty would at a fortieth of the wait — and profiling the fiftieth step
finds nothing the second did not. The size is what to vary: attention is
quadratic in the token count and the linears are linear in it, and the two
regimes trade places somewhere between 256 and 1024 pixels. Guidance is
left to the server's default (Qwen-Image's own, on) unless `--image-cfg` says
otherwise, because that is how the console's request arrives; `--image-cfg
1` halves the passes and is the honest number for a server configured with
`image_cfg_scale = 1`. The seed is fixed, so every repetition draws the same
noise; the picture is not the point.

**The warmup is a 16-pixel picture** — one latent token, one step, no
guidance: the smallest request the endpoint accepts, which still creates
the pipeline's own threads before `perf` attaches (see *Why `--flamegraph`
needs the warmup*) and still reads every transformer weight once, so the
first measured picture pays no page-cache cost. A text completion would not
do either: it exercises the text encoder, which is the one model of the
three a picture spends the least time in.

**The line under each row is the transformer by operation class** — the
server's `timings.stages`, as shares of the passes' own total, largest
first: `modulation` (the `img_mod`/`txt_mod` linears, `[3072 → 18432]`
weights read for a *single* token, sixty blocks a pass — pure weight
traffic), `qkv` (the six attention input projections), `attention` (head
norms, RoPE and the joint attention itself), `out` (the two output
projections), `mlp` (the four feed-forward linears and their GELU) and
`other` (embeddings, norms, modulation arithmetic, residual adds, the final
projection). This split is the server's own account, not the profile's: the
matmuls run on rayon's workers, whose stacks bottom out in the pool rather
than in the layer that asked, so in the flamegraph every linear of every
block lands on the one `dot_k_pair` frame and the caller is unknowable
from the stacks alone. The two are read together — the flamegraph says
*which kernel*, the stage line says *which layer*.

`first_progress_s` (in `--json` output) is the request-to-first-progress
time seen from this side — the encode phase plus the first step, a
cross-check on the server's own split that needs no server cooperation. A
server older than this mode, whose completed event carries no `timings`, is
refused rather than guessed at.

### A sweep can contaminate its own later rows

One measured caveat that applies to `--pp` and `--pp-continue` alike: the rows
in a single invocation are **not** independent. They run against one long-lived
server process, and what that process has already been asked to do can change
what a later row measures.

The reproducible case: a 130-token prefill measured 360 tok/s on a fresh server
and on a server that had only seen wide widths, but **94 tok/s** on a server
that had just been swept across eight narrow widths — the identical request,
four runs out of four. The cause is not established (an arena-pressure
hypothesis was implemented and measured neutral, see `PERF-GAP.md`).

What to do about it:

- **Comparisons within one sweep order are still valid.** If both arms of an
  A/B sweep the same widths in the same order, whatever the effect is applies
  to both, and the ratio survives. This is why the A/Bs in `PERF-GAP.md` pin
  the width list across arms.
- **A single absolute number is not.** If a specific width's rate is the
  result, measure that width on a fresh server rather than reading it off the
  end of a long sweep.
- **Restart between arms**, which the harness scripts do — and if an arm can
  drift (thermals, clocks, a cold first run), interleave the arms rather than
  running all of one and then all of the other.

### Position inside a `--sweep` is a confound, and a large one

`--sweep` runs its values in the order you give them, and each value's
repetitions run together. The **later slot is systematically worse** on
hardware that warms or a drive that tires — measured here at **2.96% slower
with three to six times the variance**, pooled over seven `pp512` sweeps.

That is bigger than most changes worth measuring, so a two-value sweep written
`VAR=a,b` charges the whole positional penalty to `b`:

| what the sweep said | what it was |
| :-- | :-- |
| +4.8%, −9.8%, +4.0%, −8.8% across four same-order runs | a **+0.11%** change |
| −3.1% | a **+3.9%** change |

**Write each value more than once, in both orders** — `VAR=a,b,b,a` and
`VAR=b,a,a,b` — so every value holds every position, and pool the runs.
And score on **mean ± sd, not best-of**: the extra variance in the late slot
inflates a best-of statistic, which is how the +0.11% change above also
produced a confident-looking +4.9% "win".

### Storage mode (`--storage-probe`) — what request size is worth

`[orangu-server].read_size` exists because read throughput is a function of
request size, and the shape of that function is a property of the drive, the
bus and the bridge rather than something derivable from `sysfs`. This measures
it:

```sh
orangu-bench --storage-probe 4,64,128,256,512,1024,2048,4096,8192 \
  --storage-file <a model shard> --storage-span 384 --storage-ramp 16 \
  --history storage-history.tsv --chart storage-curve.svg \
  --chart-y-label "MB/s (log)"
```

Sizes are KiB. `--storage-file` names the file to read; without it the probe
asks the server at `--url` for its largest shard, which is usually what you
want and needs a server running. `--storage-span` is how much to read at each
size, per pass, and `--storage-ramp` how much to read and discard first.

Reads bypass the page cache (`O_DIRECT` on Linux, `F_NOCACHE` on macOS),
so it is neither consulted nor populated: the probe measures the device
rather than memcpy, and it does not evict a model a later benchmark is about
to want. On other platforms the probe declines rather than measure the cache.

**Every size is measured twice, ascending then descending**, and the table
reports both passes with their spread. This is not redundancy. A drive that
degrades under load gives its later sizes a worse reading, so a
one-directional sweep can invent a threshold that is not there — which is
exactly what happened to this project's own storage table, where an apparent
8× cliff at 512 KiB turned out to be the drive changing state partway through
the measurement. When the two passes disagree by more than a quarter the
probe says so and names the size:

```text
  request |     MB/s |       up |     down |  spread
      4 K |     27.4 |     46.6 |      8.2 |  140.0%
   8192 K |    305.9 |    305.6 |    306.2 |    0.2%
  the 4 K point differs by 140% between the ascending and descending passes —
  this drive changed state during the probe, so read the two columns, not the mean
```

Read that as: 8 MiB reads hold ~306 MB/s whether measured first or last, while
4 KiB reads collapse from 46.6 to 8.2 — **small requests put this drive into a
slow state and large ones never do.** That is the argument for a large
`read_size`, and it is a stronger one than "clear a threshold".

### Capped runs (`--cap`) — the regime a large model actually meets

A model smaller than RAM is served from the page cache, so a benchmark of it
measures the CPU. Streaming behaviour only appears once the working set
exceeds what the kernel will hold, and reproducing that on a large host
otherwise means finding a large enough model.

`--cap` runs each `--sweep` server inside a memory-capped transient cgroup:

```sh
orangu-bench --cap 4G --drop-model-cache \
  --sweep 'ORANGU_CHUNK_POLICY=flat,adaptive' \
  --sweep-cmd "orangu-server -c bench.conf <model>"
```

It uses `systemd-run --user --scope`, so it needs no root and the scope is
cleaned up when the server exits. Swap is pinned off — a cap the kernel can
satisfy by swapping measures the swap device instead of the model's read path.

Combined with the `io` line (below) this reports what a streamed model costs
per token:

```text
  io       read 19.74 GiB from disk  ·  1263.6 MiB/token  ·  20688 major faults
```

1,263.6 MiB per token against a 1.23 GiB model is the whole model, every
token — the signature of a working set that does not fit.

### The `io` line — bytes, which do not drift

Every run reports what it pulled through the block layer, from the server's
own `/proc/self/io` over the measured window, and records it as
`io_mb_per_token` and `io_majflt` in `--history`.

This is the most trustworthy number a run produces on storage-bound work.
Rates on real hardware drift — thermals, clocks, a drive tiring — and this
manual's own sections say so repeatedly. **Bytes moved do not.** Two runs of
the same prompt against the same model read the same number, so a change that
moves bytes has moved something real, whatever the stopwatch says that
afternoon.

A warm run reads nothing and says `(fully cached)` rather than showing a bare
zero, because "read nothing" and "could not measure" must not look alike.

### Measuring across cards, and across models

`orangu-bench` is an HTTP client, so it never chooses a device: the server
does, with `--device` / `[orangu-server].device` (an index, part of a name, or
`auto`) and `--device-split` for spreading one model's layers across several.
What the benchmark side owns is making the answers *comparable* afterwards —
recording which card produced each row, and restarting the server between
points so one invocation can cover several.

`ORANGU_DEVICE` is read ahead of the config file precisely so a sweep can walk
a machine's cards without editing anything between runs:

```sh
# the same model, every card in the machine, one invocation
orangu-bench --sweep ORANGU_DEVICE=0,1 \
             --sweep-cmd "orangu-server --model gemma-4-E2B-it-Q4_K_M.gguf" \
             --depths 0,512,1024 --history perf-history.tsv --chart cards.svg
```

Each point starts its own server, waits for it to answer, proves through
`/props` that the process answering is the one it started, measures it, and
stops it — see the warnings in *A sweep can contaminate its own later rows*
for why the restart matters, and `--sweep`'s own refusal to measure a server it
did not launch for why the identity check does. The summary table is followed
by a line naming the device behind each point, because `0` and `1` are the
indices asked for and not the cards that answered:

```text
  ORANGU_DEVICE=0          Vulkan/Radeon Graphics
  ORANGU_DEVICE=1          Vulkan/Radeon RX 5500M
```

That line appears only when the points really did run on different devices, so
an ordinary tuning sweep's output is unchanged.

**Models sweep the same way.** `--sweep-cmd` runs through a shell with the
swept variable exported, so the variable does not have to be one the engine
reads — it can be one the command line reads:

```sh
orangu-bench --sweep 'MODEL=/models/a.gguf;/models/b.gguf' \
             --sweep-cmd 'orangu-server --model "$MODEL"' --pp 512
```

**Values that contain commas need `;`.** A split ratio is written `3,1`, which
is also how the value list is separated: `--sweep ORANGU_DEVICE_SPLIT=off,all,3,1`
would be four points, two of them halves of a ratio, and because `3` and `1`
are themselves legal one-entry ratios nothing would complain. A spec containing
a semicolon is therefore split on semicolons and never on commas:

```sh
orangu-bench --sweep 'ORANGU_DEVICE_SPLIT=off;all;3,1' --sweep-cmd "orangu-server ..." --depths 0
```

A spec with no semicolon splits on commas exactly as before. The one case a
parser can still catch is caught: a comma-separated sweep of
`ORANGU_DEVICE_SPLIT` containing a bare number is refused, with the semicolon
form in the message, rather than measured.

#### Two engines through one harness

The comparison this tool exists for — orangu against another engine, on the
same model, through the same requests — is a `--sweep` whose variable the
*command* reads rather than the engine:

```sh
cat > serve.sh <<'END'
#!/bin/sh
case "$ENGINE" in
  orangu)    exec orangu-server "$MODEL" --port 8100 --web 0 --device 5500M ;;
  reference) exec other-server -m "$MODEL" --port 8100 --device Vulkan1 ;;
esac
END
orangu-bench --sweep ENGINE=orangu,reference --sweep-cmd ./serve.sh \
             --sweep-env MODEL=/models/gemma-4-E2B-it-Q4_K_M.gguf \
             --label gemma-4-E2B-it-Q4_K_M --depths 0,1024,4096 --delay 15 \
             --history gemma-4-E2B-it-Q4_K_M.tsv --chart gemma-4-E2B-it-Q4_K_M.svg
```

Every guarantee the sweep makes for a tuning knob then holds for the engine
comparison: each point is a freshly started server on a port that was free,
proved to be the process this invocation spawned, measured, killed and reaped,
with the port waited back to free and **every card's memory in use waited back
to what it was before the first point** (within a quarter of a gigabyte, for a
desktop's own drift) before the next engine starts. Neither engine ever shares
the machine with the other, or with a leftover of itself. The `exec` in the
script is what makes the supervised pid the server's own; see *`--sweep`*.

A server that reports no pid in `/props` is identified the way the profiler
identifies one: the kernel is asked which process holds the listening socket,
and that pid must be the child this invocation spawned. A third-party server
therefore passes the identity check without being taught anything — and a
leftover of either engine on the port still fails it, which is the point.

The comparison table at the end reads *against the first value*, so list
orangu first and the percentages say how far behind or ahead it is. The
history rows carry the label `<label> · ENGINE=<value>`, which is the shape
`--table` turns into one row per model with one column per engine.

A prefill row is keyed on the prompt length the *server* reported, and two
engines' chat templates tokenize one `--pp 64` prompt to counts a token or
two apart (78 against 79, on the Prism fork against orangu). The table
therefore clusters counts within two tokens of each other — three percent,
for a long prompt whose template differences scale with it — into one row
named by the range (`78–79`), so the ratio column has both engines to read.
Two lengths a sweep actually asked for are never that close.

A decode sweep warms each fresh server on its own workload — one generation
of `--gen` tokens at the deepest requested context — before anything is
recorded. An eight-token warmup was measured leaving the first timed
repetition at 17 tok/s against 72 for the next, which the best-of statistic
hides and the mean ± sd does not.

#### Every card against every model

`--sweep` is deliberately **one axis**. A cartesian product costs the product of
two model loads, and its result is not a table anyone can read — so the matrix
is a shell loop around the sweep, writing into **one** history file:

```sh
for model in /models/llama-1b.gguf /models/gemma-12b.gguf; do
  orangu-bench --url http://127.0.0.1:8100 \
    --sweep ORANGU_DEVICE=0,1 \
    --sweep-cmd "orangu-server $model --port 8100 --web 0" \
    --depths 0,512,2048 --gen 128 --reps 3 \
    --label "$(basename "$model" .gguf)" \
    --history matrix.tsv
done
orangu-bench --chart-only --history matrix.tsv --chart matrix.svg
```

The three things that vary land in three different fields, which is what makes
the file readable rather than a pile of colliding rows:

- **the model** in `--label`, which inside a sweep *prefixes* each point rather
  than replacing it — so the series above come out as
  `llama-1b · ORANGU_DEVICE=0`. The loop is what varies the model, and a label
  is the only field it can reach.
- **the swept variable** in the point name, which the sweep writes itself.
- **the device** in the `device` column, which the run fills in from the server
  it measured.

That asymmetry is why the loop goes over *models* and the sweep over *devices*
rather than the other way round: the device needs no naming and the model does.

The chart is drawn at the end, once, from the accumulated file — every run's
rows are already in it, so nothing has to be re-measured to redraw. Each series
comes out as `llama-1b · Vulkan/Radeon RX 5500M`, and the four combinations sit
on one pair of axes.

Two practical notes for a matrix that takes hours. Give each model its own
`--sweep-cmd` port only if you are running loops concurrently — otherwise let
them share one, since each sweep frees the port before it returns. And check
the `build` line in each block: a matrix is long enough that a rebuild can land
in the middle of it, which is exactly how two points end up measuring two
different binaries (see *Confirm you are measuring the build you think you
are*).

A split model reports its **placement plan** in the header where a single
device reports its kernel selection, since there is no one device to describe:

```text
  api      vulkan · split across 2 devices · 1 hand-off/token
  device 0  Radeon RX 5500M · 2 layers · 0.27 GiB of 4.00 GiB · 3.73 GiB free · KV room ~918k tok
  device 1  Radeon Graphics · 14 layers · 0.47 GiB of 21.06 GiB · 20.59 GiB free · KV room ~724k tok
  kernels  not reported on a split — the tuning report describes one device, and this run has none
  timings  no GPU stage breakdown on a split — query sets belong to one device; --flamegraph still profiles the CPU side
```

**KV room is per device and is not proportional to the layer count.** In the
run above, device 1 holds seven times the layers and five times the free memory
of device 0 and has *less* room for KV, because its fourteen layers cost seven
times as much per token. That figure comes from the server measuring each
device's own layers (`kv_dim` and stride vary down a model's depth), not from
scaling a total — which is why it cannot be worked out from the layer counts by
hand.

A device the plan overfilled says so, on its own line:

```text
  device 0  Radeon RX 5500M · 36 layers · 5.49 GiB of 4.00 GiB · 0.00 GiB free · KV room ~0k tok
  device 0  OVER by 1.49 GiB — the driver will page this device's weights on every token; give it a smaller share of the split
```

This is the single most useful thing the header can tell you before trusting a
split's throughput, because paging weights on every token produces a rate that
looks like a slow engine rather than a placement to change.

Two things that block is honest about, because they change what a split run's
numbers can be used for. The **kernel and flag report is unavailable**, so a
split cannot be A/B'd against a single card for kernel selection — only for
placement. And **`--flamegraph` is the only profiler that still works**: a GPU
timestamp query set belongs to one device, so `ORANGU_GPU_TIMESTAMPS` resolves
none of them on a split. The server says so rather than answering an empty
result — `GET /gpu-timings` returns `{"enabled": false, "timings": null,
"unavailable": "split"}` — and the header line above repeats it, because a
diagnostic that silently does nothing invites the reading that what it measures
cost nothing.

#### What a split costs, and how to measure it

A split's loss has two terms, and they behave completely differently: a **fixed**
cost paid the moment the model is split at all, and a **marginal** cost for each
layer that ends up on the slower device. Separating them is one sweep — move the
boundary and watch the slope:

```sh
orangu-bench --sweep 'ORANGU_DEVICE_SPLIT=off;15,1;14,2;12,4;8,8' \
             --sweep-cmd "orangu-server model.gguf --port 8100 --web 0" \
             --depths 0 --gen 32 --reps 4
```

`off` is the baseline, and every other point has **exactly one boundary** — the
plan hands out layers in contiguous runs — so anything the points share is
fixed and anything that grows with the layer count is marginal. Convert the
rates to milliseconds per token before reading it; a slope in tok/s is not a
slope in cost. On this project's dev machine (RX 5500M + Renoir iGPU,
Llama-3.2-1B-Q4_K_M, 16 layers):

| placement | decode | ms/token |
| :-- | --: | --: |
| `off` — 16 layers on the discrete card | 60.0 tok/s | 16.7 |
| `15,1` | 39.4 | 25.4 |
| `14,2` | 36.0 | 27.8 |
| `12,4` | 36.3 | 27.5 |
| `8,8` | 27.4 | 36.5 |

The slope is about **1.7–2.4 ms per layer moved**, and extrapolating back to
zero layers moved leaves about **6–7 ms** that the split costs before any layer
has changed device — roughly 40% of an entire unsplit token. So **moving one
single layer costs 34%**, and moving seven more costs only about as much again.
If you take one thing from this: a split's price is mostly paid at the moment
you split, so a model that fits on one card belongs on one card.

That fixed term is not the hand-off across the bus. The hidden state for this
model is 2048 floats — 8 KiB — which is microseconds on any link; what the
fixed term actually is has not been established. Note also that it cannot be
attributed *per boundary* on a two-device machine, because every split there has
exactly one: separating "the cost of a boundary" from "the cost of splitting"
needs a third device, and this rig has two.

Two more things worth knowing before trusting such a sweep:

- **Use `--reps 4` or more.** At two repetitions this same sweep put `14,2`
  ahead of `15,1`, which is impossible — the ordering came back at four. The
  unsplit point is the noisiest of all (±4.8 tok/s here) because it is the
  fastest, so scheduling jitter is a larger share of its per-token time.
- **`ORANGU_NO_SPLIT_FUSION=1` is the A/B for the split's GPU paths**, from one
  binary and two runs — the way to check they are live rather than assume it:

  ```sh
  orangu-bench --sweep 'ORANGU_NO_SPLIT_FUSION=;1' \
               --sweep-env 'ORANGU_DEVICE_SPLIT=12,4' \
               --sweep-cmd "orangu-server model.gguf --port 8100 --web 0" \
               --depths 0 --gen 32 --reps 2
  ```

  The empty first value means *unset*, so the two points are "fusion on" against
  "fusion off". Here that was 34.8 against 20.8 tok/s — per-layer fusion is
  worth **+68%** on a split, which is most of what makes one usable at all.
  `--sweep-env` is what holds the placement identical across both.

### What was measured

Every run opens with the server's model, backend and **build** (from
`GET /props`) and, on Linux with an AMD card, each GPU's current core clock and
DPM mode read from `/sys/class/drm/card*/device/`.

```text
  model    unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
  backend  Vulkan/AMD Radeon RX 5500M (RADV NAVI14) (Vulkan)
  build    1.2.0 (52c04435f-dirty)
  server   pid 48219 up 12s
```

The `build` line is `version` plus the git commit the server was compiled
from, `-dirty` when tracked files differed from that commit. A version alone
cannot separate two builds during performance work — every build between two
releases carries the same one — and a run whose engine is identified only by a
hand-typed `--label` is a run whose provenance depends on someone having
remembered. The bundle archives both fields, so `--read-bundle` names a build
change in **what differed** before it shows what moved. A server that reports
neither (an older `orangu-server`, or another engine) simply gets no such line. This is not decoration: a card left at
`power_dpm_force_performance_level = auto` can idle its clock down between
requests, which moves throughput by more than the difference most benchmarks
are run to detect. A rate recorded without the device and clock state beside it
cannot be compared against one recorded later. Under `--json` the same
information is the first object emitted, tagged `"type": "env"`, so a stored
result carries its own provenance.

### Options

`orangu-bench --help`:

```text
Measure throughput of an OpenAI-compatible server

Usage: orangu-bench [OPTIONS]

Options:
      --url <URL>                      Base URL of the server [default: http://127.0.0.1:8100]
      --depths <LIST>                  Comma-separated context depths to sweep. Ranges too: `0-2048+512` [default: 0]
      --pp <LIST>                      Prefill mode: prompt lengths to sweep, reporting prompt-processing rate
      --pp-continue <LIST>             Continuation-prefill mode: comma-separated *added* token counts to sweep
      --pg <LIST>                      Combined mode: prompt lengths prefilled and generated in one request
      --decode-cpu                     Report the server's CPU time per generated token, with prefill excluded
      --streams <LIST>                 Concurrency mode: comma-separated stream counts; reports AGGREGATE tok/s
      --pp-continue-base <N>           Prompt length (tokens) to prime the prefix cache with for `--pp-continue` [default: 512]
      --embed <LIST>                   Embedding mode: prompt lengths to sweep against /v1/embeddings
      --image <LIST>                   Image mode: square picture sizes (pixels, multiples of 16) to sweep against /v1/images/generations
      --image-steps <N>                Denoising steps per picture for `--image` [default: 2]
      --image-cfg <SCALE>              Guidance scale for `--image`; 1 runs the prompt alone. Default: the server's
      --image-prompt <TEXT>            The prompt every `--image` picture is drawn from [default: "Create an image of a cat"]
      --gen <N>                        Number of tokens to generate per timed run [default: 128]
      --curve <N>                      Curve mode: one generation of this many tokens, bucketed by context; 0 disables [default: 0]
      --bucket <N>                     Bucket width (in context tokens) for `--curve` [default: 256]
      --reps <N>                       Repetitions per depth; the reported rate is the best run with mean±sd [default: 3]
      --drop-model-cache               Evict the model from the page cache before every repetition
      --no-warmup                      Skip the initial warmup run
      --per-rep                        Also print every repetition's rate, in order, after each row
      --timeout <SECONDS>              Per-request timeout in seconds [default: 600]
      --model <ID>                     Model id to request
      --json                           Emit machine-readable JSON
      --history <PATH>                 Append each measured point to this tab-separated history file
      --label <NAME>                   Series name recorded in the history file (defaults to the server's model); prefixes each `--sweep` point
      --chart <PATH>                   Render the history file to this SVG after measuring
      --chart-only                     Only render the chart and table from an existing history file; measure nothing
      --table <PATH>                   Write the history file as a markdown comparison table to this path (`-` for stdout)
      --storage-probe <LIST>           Storage mode: comma-separated read request sizes in KiB to sweep
      --storage-file <PATH>            File the storage probe reads. Defaults to the server's largest shard
      --storage-span <MIB>             MiB to read at each request size, per pass [default: 256]
      --storage-ramp <MIB>             MiB read and discarded before timing starts, at each size [default: 32]
      --cap <SIZE>                     Run each `--sweep` server under this memory cap (e.g. `4G`)
      --chart-png                      Also render a PNG beside the chart SVG
      --chart-scale <MIN:MAX>          Pin the chart's tok/s axis to `MIN:MAX` so a pair of charts compare
      --chart-y-label <TEXT>           Label for the chart's y-axis [default: "tok/s (log)"]
      --chart-x-label <TEXT>           Label for the chart's x-axis
      --chart-panels <LIST>            Draw only these modes' panels (e.g. `pp,tg`); default: every mode in the file
      --flamegraph <PATH>              Record a CPU flamegraph of the server over the measured window
      --flamegraph-pid <PID>           Process to profile (default: the server's own, else the URL port's owner)
      --flamegraph-freq <HZ>           Sampling frequency in Hz for `--flamegraph` [default: 999]
      --flamegraph-call-graph <MODE>   Call-graph mode for `--flamegraph`: `auto`, `fp` or `dwarf` [default: auto]
      --flamegraph-png                 Also render a PNG beside the flamegraph SVG
      --flamegraph-layers <DIR>        Profile every running orangu, orangu-coordinator and orangu-server for `--flamegraph-duration` while you drive the workload; one flamegraph per process in DIR. Measures nothing itself.
      --flamegraph-duration <SECONDS>  Seconds to keep sampling under `--flamegraph-layers` [default: 60]
      --compare-profiles <LIST>        Compare already-collapsed `.folded` profiles side by side; measure nothing
      --bundle <PATH>                  Write the whole run — measurements, configuration, host — to one JSON file
      --read-bundle <LIST>             Read bundles and report them side by side; measure nothing
      --sweep <SPEC>                   Sweep one tuning variable: `VAR=v1,v2,...` (`;` if a value has a comma); needs `--sweep-cmd`
      --sweep-cmd <CMD>                Shell command that starts the server, run once per `--sweep` value
      --sweep-env <K=V>                Environment held constant across every `--sweep` point (repeatable)
      --sweep-start-timeout <SECONDS>  Seconds to wait for a swept server to come up [default: 300]
      --render-profile <PATH>          Re-render an already-collapsed `.folded` profile to SVG; measure nothing
      --report <PATH>                  Write the run — provenance, measurements, chart, flamegraph — to one PDF
      --web                            Serve the web console instead of measuring
      --host <HOST>                    Address the web console binds: "all" (or "*") for every interface [default: 127.0.0.1]
      --port <PORT>                    Port the web console listens on [default: 8300]
      --delay <SECONDS>                Seconds to wait between measured points, for a card that heats up [default: 0]
      --temperature <T>                Sampling temperature for the timed decode; `0` is greedy [default: 0]
  -s, --shell-completions              Print the shell completion script for the detected shell and exit
  -h, --help                           Print help
  -V, --version                        Print version
```

Notes: `--url` is the server base URL (the tool appends `/v1/completions`);
`--depths` is comma-separated and accepts ranges (e.g. `0,512,1024,2048` or
`128-2048*2`, see *Ranges* above); `--reps` reports the
best (fastest) run with mean ± standard deviation alongside; warmup (one short
generation) is on unless `--no-warmup`; `--json` emits one JSON object per depth
instead of the table.

`-s`/`--shell-completions` prints a bash/zsh/fish/PowerShell completion script for the
shell detected from `$SHELL` and exits — the same switch every orangu binary
has (see the Shell completions chapter). It offers every flag above; the
path-taking ones (`--history`, `--chart`, `--storage-file`, `--flamegraph`,
`--compare-profiles`, `--bundle`, `--read-bundle`, `--render-profile`,
`--report`) complete files, `--flamegraph-call-graph` its three modes and
`--host` the usual bind addresses:

```sh
# bash — add to ~/.bashrc:
eval "$(orangu-bench -s)"
# zsh — write once to your fpath directory:
orangu-bench -s > ~/.zsh/completions/_orangu-bench
# fish — add to ~/.config/fish/config.fish:
orangu-bench -s | source
# PowerShell — add to $PROFILE:
orangu-bench -s | Out-String | Invoke-Expression
```

#### Mixture-of-experts models, and models that do not fit

On a MoE model whose weights are larger than RAM, a rate is not enough to score
a change. One that moves fewer expert bytes and one that moves the same bytes
faster report the same tok/s, and once the weights stream from disk neither is
visible over the I/O at all. So against an `orangu-server` that reports them,
each measured `--pp` and decode point carries three mechanism figures beside its
rate, printed as a `moe` line and recorded in the history file and the bundle:

| history mode | what it is |
| :-- | :-- |
| `moe_union` | how many times the average expert's weights were read inside one layer call, against reading each distinct expert once. `1.0` is the floor. The server also reports `selection_ratio` — the routing's own redundancy, which is the ceiling on what grouping by expert can save and which no implementation moves |
| `moe_mb` | expert weight megabytes dequantized per token per MoE layer |
| `moe_majflt` | major faults per repetition — the only honest signal that weights came off the disk rather than out of the page cache |

They are exact counts, not repeated measurements: `best` and `mean` are the same
number and the `sd` columns are empty, and each is drawn on its own chart panel
for that reason. A dense model writes none of them, rather than writing zeros
that would read as "no redundancy".

The run header also reports how much of the model was in the page cache when it
started (`GET /model-cache`). That is not context — on a model that does not
fit, a cold run and a warm run of the same build are two different experiments,
and their rates must not be compared. `--drop-model-cache` evicts the model
before **every repetition**, so a `--reps 3` run measures three cold ones rather
than one cold and two warm; without it, whatever the previous run left behind is
what is being measured.

> `--drop-model-cache` is silent against a server without the endpoint, which
> would produce warm numbers under a cold heading. The residency figure in the
> header is what catches that, which is why it is printed and archived rather
> than merely available.

#### Which standard deviation

The `±` printed beside the mean is the **sample** standard deviation — the sum
of squares divided by `n - 1` — which is the standard estimator and the one
every other benchmark reports. The column says so: `mean ± sd(n-1)`. A run of
one repetition has none, and the row reads `—` rather than `0.00`, which would
claim a spread was measured and found to be zero.

The history file and the bundle carry **both**. Their `sd` column is the
*population* estimator (divided by `n`), which is what it has meant since the
file was created — a column in an append-only record is not redefined, so a new
one, `sd_sample`, was added beside it. Rows written before that column existed
read back with it empty, not as parse failures.

The difference is not cosmetic: at the default three repetitions the sample
figure is `sqrt(3/2)` — 22% — larger than the population one, which is bigger
than most of the differences this tool is run to detect. Quoting one against the
other would be a systematic error, which is exactly why both are recorded and
the printed one is the comparable one.

The **headline is still the best run**, not the mean, in the table, the report
and the history file. A best is always the flattering statistic, so the PDF
report says so on the page and the console's table heads its column `± sd (n-1)`
for the same reason.

### The `stages` lines — where a forward pass's *elapsed* time went

A CPU profile counts cycles, and that is the wrong unit for a forward pass. A
stage that runs alone on one core and a stage that spreads the same arithmetic
over sixteen look alike in a flamegraph, while the first costs sixteen times
the wall clock. Worse, a stage that spends its time parked in a GPU driver
waiting for a submission barely appears at all — the thread is not running, so
nothing samples it — even though the token is waiting on exactly that.

Start the server with `ORANGU_DECODE_STAGES=1` and each measured point gains a
breakdown in the same units as the rate it explains:

```text
  stages   217.6 ms per forward pass over 256 passes
           recurrent.project     31.13 ms/pass   14.3%  (30.0 calls/pass)
           recurrent.delta       54.47 ms/pass   25.0%  (75.5 calls/pass)
           recurrent.out         29.72 ms/pass   13.7%  (37.7 calls/pass)
           attn                   0.94 ms/pass    0.4%  (10.0 calls/pass)
           ffn.router            13.49 ms/pass    6.2%  (80.0 calls/pass)
           ffn.routed            51.32 ms/pass   23.6%  (40.0 calls/pass)
           ffn.shared            40.42 ms/pass   18.6%  (40.0 calls/pass)
           head                  10.17 ms/pass    4.7%  (1.0 calls/pass)
           other                  0.00 ms/pass    0.0%
```

The counters are drained on read, exactly like `moe` and `gpu-timings`: the
tool reads once before the workload to discard the warmup and once after, so
the window is the measured one and nothing else.

Read it with three rules:

- **Percentages are against `forward`, never against each other.** `forward`
  is the parent and every other row is inside it.
- **`other` is `forward` less what was attributed.** `forward`, `attn`,
  `ffn.router` and `ffn.routed` are timed inside code every architecture goes
  through, so they are reported for any model; the finer rows are timed at the
  one call site that can name them, which is per-architecture. On a model whose
  architecture has not been given those, `other` is most of the pass — and
  saying so is the point, because a breakdown that quietly omitted the
  unattributed remainder would look complete when it was not.
- **`ffn.routed` and `ffn.shared` overlap** when the FFN runs its two branches
  concurrently, so their sum can exceed the elapsed time they shared. That is
  what overlapping them was for; `other` clamps at zero rather than going
  negative when it happens, and the size of the overlap is the two rows' sum
  less what the pass actually spent there.

A row's `calls/pass` is the other half of the reading. A stage entered thirty
times a pass is a per-layer cost and scales with depth; one entered once is
not. Where the stage is a device call, `calls/pass` is also a submission count,
and `ORANGU_GPU_TRACE=1` reports the total per decode step to check it against.

### Profiling what was measured (`--flamegraph`)

A rate says *how* slow something is and never *where* the time went. Both
questions get asked together during performance work, and they used to be
answered by two different procedures: this tool for the number, a hand-assembled
`perf record` pipeline for the profile. The profile then routinely covered a
different workload than the number it was supposed to explain — a different
prompt length, the warmup included, or a window that opened before the server
was busy.

`--flamegraph FILE.svg` records a CPU profile of the server **over exactly the
measured window**: sampling starts after warmup and stops when the last
repetition finishes, so the flamegraph and the tok/s printed above it describe
the same seconds of the same process. Any other server goes through the same
path, so two engines' profiles are as comparable as their two rates.

Collapsing and rendering are done **in this binary**, so `perf` is the only
external program involved. The rendered SVG is interactive on its own: click a
frame to zoom into it, click the title to reset, Ctrl-F to highlight every frame
matching a substring and report what share of samples matched.

```sh
# orangu-server, decode: one profile of the 512-token generation that was timed
orangu-bench --url http://127.0.0.1:8100 --depths 0 --gen 512 --reps 2 \
             --flamegraph perf/gemma4-e2b-orangu-decode.svg --flamegraph-png

# the reference engine on :8300, the same workload through the same harness
orangu-bench --url http://127.0.0.1:8300 --depths 0 --gen 512 --reps 2 \
             --flamegraph perf/gemma4-e2b-reference-decode.svg --flamegraph-png
```

Three files come out of one `--flamegraph out.svg`:

| file | what it is |
| :-- | :-- |
| `out.svg` | the flamegraph, interactive (click to zoom, Ctrl-F to search) |
| `out.folded` | the collapsed stacks — a text file that diffs, and re-renders without re-running |
| `out.meta.json` | pid, sampling frequency, duration, samples, cores busy |
| `out.png` | a raster copy, with `--flamegraph-png`, for documents that cannot embed an SVG |

The transient `perf.data` is removed once collapsed: it is the largest artifact
by an order of magnitude and nothing downstream reads it. The `.folded` file is
the durable one.

Because it is durable, the SVG can be rebuilt from it at any time without
measuring again:

```sh
orangu-bench --render-profile out.folded
orangu-bench --render-profile out.folded --flamegraph new.svg
orangu-bench --render-profile out.folded --flamegraph-png
```

`--render-profile` measures nothing. It writes beside its input unless
`--flamegraph` names an output, and combining it with `--flamegraph-png` adds
the PNG a run could not produce at the time because no rasterizer was
installed.

Beside the files the tool prints what the stacks say, so the common case needs
no SVG viewer at all:

```text
  profile  perf/gemma4-e2b-orangu-decode.svg (24140 samples over 25s)
           perf/gemma4-e2b-orangu-decode.folded
           kernel        33.5%
           app/other     22.5%
           kernel:gpu    16.3%
           libc/alloc     9.4%
           radv/vulkan    9.1%
           wgpu           9.1%
           top self-time frames:
             5.0%  __memset_avx2_unaligned_erms
             1.8%  amdgpu_vm_bo_update_[k]
             …
```

Attribution is by **leaf frame** — a stack's whole count is charged to the
function that was actually executing, which is what a flamegraph's plateau
widths show. Kernel-mode frames carry a `_[k]` mark, taken from the **object
file** `perf` reports rather than guessed from the symbol name, so `amdgpu_*`
reached through an ioctl is never confused with the userspace `radv_*` that runs
on the calling thread — and so `read_hpet`, an ordinary-looking symbol that
lives in the kernel, is not filed as application code.
Everything below that line is a heuristic over symbol names, and anything it
cannot name stays visible as `app/other` rather than being dropped — which is
what the leaf table beside it is for: a residual you can read is a claim you can
check.

### Profiling the whole chain (`--flamegraph-layers`)

`--flamegraph` profiles the process answering `--url` while this tool drives
its own workload. That answers "where does the *server* spend its CPU on a
benchmark". It cannot answer the question a slow `/auto_review` raises, which
is where a request's time goes across **three** processes — the `orangu` that
renders it, the `orangu-coordinator` that routes it, and the `orangu-server`
that computes it — under the real client's real traffic.

`--flamegraph-layers DIR` does that. It measures nothing itself: it samples the
machine for `--flamegraph-duration` seconds while *you* drive the workload in
a real `orangu`, then writes one flamegraph per orangu-family process that ran
during the window, named by layer and pid — `orangu-1311063.svg`,
`orangu-coordinator-1310966.svg`, `orangu-server-1311286.svg` — with
`--flamegraph-png` giving each its PNG.

```
$ orangu-bench --flamegraph-layers perf/review --flamegraph-duration 90 --flamegraph-png
profiling every orangu, orangu-coordinator and orangu-server for 90s — drive the workload now
    (in another terminal: orangu, then /auto_review src/admin.c)

=== orangu (pid 1311063)
  profile  perf/review/orangu-1311063.svg (2178 samples over 110s — 0.02 cores busy)
           …
=== orangu-server (pid 1311286)
  profile  perf/review/orangu-server-1311286.svg (40683 samples over 110s — 0.37 cores busy)
           …

   layer                  pid   samples   cores   gpu-wait  pool-idle
   orangu              1311063      2178    0.02       0.0%       1.2%
   orangu-coordinator  1310966         1    0.00       0.0%     100.0%
   orangu-server       1310985         3    0.00       0.0%      33.3%
   orangu-server       1311286     40683    0.37       0.0%      25.1%
```

Two things about how it samples are the reason it exists as its own mode:

- **System-wide, not attached per pid.** `perf record -p` sees only the threads
  that exist at the instant it attaches (the warmup rule below). A coordinator
  replaces its `orangu-server` on every model swap, and an `/auto_review`
  begins with one — so the process worth profiling did not exist when
  sampling started. `perf record -a` sees every thread from the moment it
  runs, and the split into processes is done afterwards from each sample's
  own `comm` and `pid`. It needs `kernel.perf_event_paranoid <= 0` (or
  `CAP_PERFMON`), one step more than `-p` does.
- **Every process that ran gets a graph.** The table above shows two servers
  because the review swapped one for another mid-window; a run with several
  clients lists each. A process that forked and exec'd something else
  (`orangu` running `git`) is a few samples under its parent's name; those are
  written but kept out of the report.

The final table is the one to read first: it is the only place the layers sit
beside each other in the same units. In the run above it says what the
individual graphs then explain — the client is 0.02 cores of terminal
redrawing, the coordinator is one sample in a hundred seconds, and the server
is 0.37 cores because it is waiting on the GPU, which is where the tokens per
second are decided.

### Why `--flamegraph` needs the warmup

`--flamegraph` is **refused** together with `--no-warmup`, and the reason is
worth knowing because it applies to profiling any server this way.

`perf record -p PID` attaches to the threads that exist at the moment it
attaches. It does not pick up threads created afterwards. A server builds its
compute threads lazily, on its first request — and this tool deliberately starts
the profiler *after* the warmup and *before* the timed window, so that the
profile covers the measurement and nothing else. With no warmup, the profiler
attaches while the server is idle and never sees a single compute thread.

The failure is silent and plausible. It does not error; it produces a
well-formed flamegraph, entirely of the HTTP thread doing TCP reads and futex
waits, with a confident `cores busy` under it. Measured here: **0.02 cores
reported against 0.44 actually used.**

So there are two guards:

- The flag combination is rejected outright, with that explanation.
- Every profile reads the target's CPU time from `/proc/<pid>/stat` at both ends
  of its own window and compares. If the samples account for less than half of
  the CPU the kernel says was used, the run prints:

```text
  WARNING: the profile accounts for 0.02 cores but /proc says the
           process used 0.44. `perf record -p` does not follow threads
           created after it attached — run a warmup before profiling so the
           workload's threads already exist. This profile is not trustworthy.
```

The `/proc` figure is also stored in the `.meta.json` sidecar as
`cores_from_proc`, so a saved profile carries the evidence for whether to trust
itself. The kernel's accounting cannot miss a thread, which is exactly what
makes it the right thing to check sampling against.

### Comparing two profiles (`--compare-profiles`)

A flamegraph is normalised to its own total, so two of them side by side cannot
be read against each other: the wider plateau belongs to whichever engine
happened to be sampled more. `--compare-profiles` reads the `.folded` files back
and puts them in one table — the `--chart-only` of profiling, no server and no
re-run:

```sh
orangu-bench --compare-profiles perf/gemma4-e2b-orangu-decode.folded,perf/gemma4-e2b-reference-decode.folded
```

```text
bucket         | gemma-orangu-decode | gemma-reference-decode
------------------------------------------------------------
app/other      |               22.5% |              66.2%
kernel         |               33.5% |              10.4%
kernel:gpu     |               16.3% |               3.9%
ggml           |                0.0% |              14.5%
radv/vulkan    |                9.1% |               4.4%
libc/alloc     |                9.4% |               0.6%
wgpu           |                9.1% |               0.0%
samples        |                3288 |               1659
gpu-wait       |                6.7% |              60.0%
pool-idle      |               32.2% |               1.3%
working        |               61.2% |              38.8%
cores busy     |                0.47 |               0.42
— working      |                0.29 |               0.16
```

**`cores busy` is the row that makes the rest of the table mean anything.** It
is `samples / (freq × seconds)`: the mean number of the server's threads that
were on a CPU. Every other row is a share of *that* engine's own CPU time, so
without it "33.5% here against 10.4% there" compares two different totals. It is
read from the `.meta.json` written beside the profile; a `.folded` carried off
the machine alone shows `?`.

**`gpu-wait` and `pool-idle` are why an occupancy number was needed at all.**
Two engines can wait for the GPU in different ways — one spinning on
`_mm_pause`, `orangu-server` blocking — and a blocked thread produces no samples
while a spinning one produces them at full rate. Read naively, that makes the
engine wasting a whole core look busy with useful work and the one yielding it
look idle. Both are therefore detected from the **stack**, not the leaf, and
subtracted: `— working` is the cores actually spent on the model.

`pool-idle` is kept separate from `gpu-wait` because they have different
remedies. Time under `wait_until_out_of_work` is a thread pool waking more
workers than there is work for; time under a fence is the device owing an
answer. Merging them would charge a threading problem to the GPU.

**Requirements.** `perf` — and, for the SVG, nothing else.

- `perf`, with `kernel.perf_event_paranoid` low enough to attach to a process
  you own (`-1` on this project's rig). It is the one piece that cannot be
  replaced: it reads the kernel's perf events.
- Collapsing and rendering are **in-process** (`src/bin/orangu-bench/
  flamegraph.rs`). There is no dependency on `stackcollapse-perf.pl`,
  `stackcollapse-recursive.pl` or `flamegraph.pl`, and nothing to install or
  point at. The rendered SVG is self-contained: no external stylesheet, script,
  font or image.
- `rsvg-convert`, **only** for the optional `--flamegraph-png`. Missing, the SVG
  is still written and the run still succeeds.
- **Nothing. The ordinary build is profiled.** `--flamegraph-call-graph`
  defaults to `auto`: the benchmark asks the server whether it kept frame
  pointers (`GET /props`, `frame_pointers`) and unwinds with `fp` when it did
  and `dwarf` when it did not. So `target/release-with-debug/orangu-server` —
  the binary a sweep measures — is also the binary a profile reads, and the
  rate and the graph describe the same thing.

  The two modes differ in how the stack is recovered, not in what they show.
  `fp` walks the frame-pointer chain and is cheaper, but needs a binary built
  with `-C force-frame-pointers=yes`; without them the chain is lost for most
  samples in the hot leaf, which renders as a flamegraph of a process doing
  nothing. `dwarf` copies a slice of each sampled stack and unwinds it with
  the tables every build carries, which costs more per sample and can truncate
  a very deep stack, but asks nothing of the build — including a server from a
  distribution package, where no build convention can help.

  Frame pointers are not on by default because they are not free: measured on
  this project's own hardware, the same source built with them is **9% slower
  at a small model's decode and 5% at a prefill**. They are also not a profile
  setting — stable cargo has no per-profile `rustflags` — so a build that
  wants them says so, and then `auto` picks `fp` for it:
  ```sh
  RUSTFLAGS="-C force-frame-pointers=yes -C target-feature=+avx2,+fma,+sse4.2" \
    cargo build --profile release-with-debug --bin orangu-server
  ```
  `--flamegraph-call-graph fp` or `dwarf` overrides the question entirely, and
  `dwarf,<bytes>` passes a stack-dump size through to `perf`.

`--flamegraph-pid` is only needed when neither route to the pid works. The tool
asks the server for its own pid first (`orangu-server` reports one), and
otherwise asks the operating system which process owns the port under test —
both of which identify the process that *answered these requests*, rather than
one that merely has a matching name.

### Can the engine fill the device? (`--streams`)

Sweeps the number of **concurrent** decode streams and reports the aggregate
tok/s across them — what a server's capacity actually is, which no single-stream
number shows.

```sh
orangu-bench --url http://127.0.0.1:8100 --streams 1,2,4,8 --gen 160
```

```
 streams |    aggregate |  per-stream |   tokens
------------------------------------------------
       1 |  37.76 ± 0.00 |       37.76 |      160
       2 |  40.21 ± 0.00 |       20.10 |      320
       8 |  45.02 ± 0.00 |        5.63 |     1280
  gpu busy card1 engine 98%  memory 30% (mean while measuring)
```

Read it together with the `gpu busy` line, because there are two very different
reasons the aggregate can stop rising:

- **aggregate flat, engine at ~100%** — the device is full. This is the good
  case; per-stream falling as 1/n is just fair sharing of a saturated GPU.
- **aggregate flat, engine well under 100%** — the engine cannot be filled. The
  work is serialising somewhere the extra requests cannot bypass, and there is
  capacity being left on the table.

Measured on this rig, the two decode paths in this engine land on opposite sides
of that: one reaches 98–99% engine occupancy with two streams, the other never
passes 66% with eight. Same device.

Each stream gets a distinct prompt so no two share a prefix-cache entry and get a
free ride.

### Was it memory-bound? (`engine` and `memory` busy)

On Linux with an AMD card, every run also reports the mean occupancy of the two
resources a kernel can run out of, sampled over the measured window from
`amdgpu`'s `gpu_busy_percent` and `mem_busy_percent`:

```text
  gpu peak card1 1700Mhz  card2 1600Mhz (while measuring)
  gpu busy card1 engine 64%  memory 21% (mean while measuring)
```

This is printed beside the rate because a rate that looks low means completely
different things at 20% memory occupancy and at 90%:

- **memory high, engine lower** — bandwidth-bound. The kernel is moving as many
  bytes as the card will give it, and the only lever is moving fewer.
- **both moderate** — *stalled*. Neither resource is exhausted; the work is
  waiting on latency, on a dependency chain, or on dispatches too small to fill
  the machine. Adding bytes or arithmetic will not show up.
- **engine high, memory low** — compute-bound.

For reference on the rig these docs were written on: a trivial coalesced
streaming kernel at 204 GB/s reads `memory 84%`; `orangu-server`'s decode reads
`engine 62% memory 20%`, and its prefill `engine 85% memory 14%`. Neither phase
is bandwidth-bound, which took a separate experiment to establish every time
before this line existed.

Counters come from `/sys/class/drm/<card>/device/`; the line is omitted on
hardware or kernels that do not publish them, rather than printing a zero.

### The clock the run actually reached

The header's `gpu … sclk` line is read once, before the workload starts, and on
an idle laptop dGPU that is its sleep state: `card1 sclk 0Mhz (high)` is what a
*correctly* pinned card looks like between requests, and it reads exactly like a
misconfigured one.

So every run also samples each card's core clock **while measuring** and prints
the peak underneath the results:

```text
  gpu peak card1 1700Mhz  card2 0Mhz (while measuring)
```

That is the line to check. A card that stayed parked at 300 MHz through a whole
sweep produces entirely plausible numbers, and this is the only place that says
so.

### Tracking throughput over time (`--history`, `--chart`)

A rate on its own says nothing. The two things it needs to be read against —
*the other engine* and *last month's build* — are both outside any single
invocation, so `--history` appends each measured point to a tab-separated file
that accumulates across runs, and `--chart` draws the chart from **that file**
rather than from the run that produced it. A run measuring only orangu still
redraws the reference engine's line beside it.

```sh
# orangu, recorded as its own series
orangu-bench --url http://127.0.0.1:8100 --label "orangu $(git rev-parse --short HEAD)" \
             --pp 128,512,1024,2048 --history perf-history.tsv --chart perf-history.svg

# the reference, same harness, same file
orangu-bench --url http://127.0.0.1:8300 --label "reference b10104" \
             --depths 0,512,1024,2048 --history perf-history.tsv --chart perf-history.svg

# redraw after hand-editing the file — no server needed
orangu-bench --chart-only --history perf-history.tsv --chart perf-history.svg
```

The file is plain TSV with a `#` header, so it diffs in review and is
hand-editable; blank lines, comments and unparseable rows are skipped rather
than fatal. Each row is `date`, `label`, `mode` (`pp`, `tg`, `curve`, `cpu` or
`pg`, `embed` — each drawn as its own chart panel, since an embedding pass and a
generative model's prefill are not the same measurement, and two of the five
are not even in tokens/second), `n`, the best / mean / both standard
deviations of the run's repetitions (see *Which standard deviation* above), and
the `device` that produced the row:

```text
#date	label	mode	n	best	mean	sd	sd_sample	device
2026-07-25	orangu af7c767	pp	1120	81.75	81.40	0.26	0.32	Vulkan/Radeon RX 5500M
2026-07-25	reference b10104	pp	1120	1061.66	1049.40	8.84	10.83	Vulkan/Radeon RX 5500M
```

Nothing is ever rewritten — a row is a measurement that was taken, and a later
run that disagrees is another row, not a correction. Columns are only ever
appended, and a shorter row is read as "that column was not recorded" rather
than as a broken line, so a file written by an older build keeps working: rows
with seven, eight and nine columns can sit in one file and all three parse.

`device` is filled in from the server's own backend label, shortened to fit a
legend — `Vulkan/Radeon RX 5500M`, `CPU/AVX2`, or
`Vulkan/Radeon RX 5500M (split 2)` for a model spread across cards. It is
recorded because with `--device` on the server the same model measured on two
cards produces rows that are identical in every other field: same date, same
label, same mode, same `n`. Without it the file cannot tell the two cards
apart, and the chart draws one series with two points at every x — a comparison
silently rendered as a trend. The bundle (`--bundle`) keeps the unshortened
name, the driver string and, on a split, the whole placement plan.

`--label` is the series identity, so it must stay stable across runs for a line
to be drawn; it defaults to the server's model id, which distinguishes two
models but *not* two builds of orangu, so pass it explicitly when A/B-ing
builds. Two devices need no such care: when a file holds rows from more than
one named device, the chart appends the device to each series name by itself —
`gemma-4-E2B · Vulkan/Radeon RX 5500M`. When every row names the same device,
or none names one at all, nothing is appended and the chart is exactly what it
was before the column existed. Prompt-processing rows are keyed by the token count the server actually
reported rather than the requested length, and a row without server timings is
printed but not recorded — a time-to-first-token is a different measurement and
does not belong on the same line as a prefill rate.

The chart is a standalone SVG with no external references: **two charts** —
prompt processing and token generation — each plotting **tokens/second against
context length**, with one line per engine.

Two charts rather than a grid of small multiples because the question the file
is kept to answer is how throughput behaves *as context grows*: whether a curve
is flat or falling away, and how far apart two engines stay along it. That is a
shape, and a shape needs the workload on an axis rather than spread across
facets. Prefill and decode stay separate because they are different
measurements that happen to share a unit — putting them in one frame would
need a second y-axis, which makes unrelated quantities look comparable.

The y-axis is **logarithmic**. The engines on it differ by an order of
magnitude, so on a linear axis the slower one collapses onto the baseline and
its own shape — the thing being tracked — becomes unreadable. On a log axis a
constant ratio is a constant vertical distance, which is how "N× behind" should
read; the bounds are `1/2/5 × 10^k` so every gridline is a round number.

Every mode in the file gets a panel, with two exceptions. The storage panels
(bytes read from disk per token, major faults) are left out when no row of
the file read a byte from disk — a fully cached model makes them two plots of
zero, and beside a two-engine comparison they read as a difference between
the engines when only one of them reports the figure at all. And
`--chart-panels pp,tg` keeps exactly the panels named, for the same
comparison: the mechanism rows a mixture-of-experts run records (expert
re-reads, MB dequantized) are one engine's instrumentation, and a chart meant
to put two engines side by side wants the panels both appear in.

Only the **newest measurement date** in the file is drawn, and the subtitle says
which — `showing 2026-07-26 (18 of 50 rows)`. The file keeps every run; that is
what it is for. But a chart of "how does throughput behave as context grows" is
answered by the build that exists, and overlaying superseded runs on top of it
only crowds the lines it is being read for. To look at history, read the file,
or point `--chart` at a filtered copy of it. Repeated measurements of one series
at one context on one date collapse to their best, for the same reason a single
run reports its best repetition. Series colours are assigned in first-seen order and never
recycled; past the last slot a label is dropped from the chart rather than
given a colour another series already owns.

### The history file as a table (`--table`)

The chart is the view for one question — throughput against context. A
document comparing two engines across many models needs the other one: every
model on its own row, one column per engine, and the ratio. `--table FILE.md`
writes exactly that from the history file, as markdown, so the table in a
report is produced by the tool that took the measurements rather than retyped
from terminal transcripts:

```sh
# after a run, beside the chart
orangu-bench … --history perf.tsv --chart perf.svg --table perf.md
# from the file alone, no server needed
orangu-bench --chart-only --history perf.tsv --table -
```

One table per mode, in the order the modes appear in the file; each cell is
the best repetition with the mean ± sample sd beside it. Rows come from the
label convention the sweep writes — `<label> · VAR=value` is one row per
`<label>` and one column per `value` — and with exactly two columns a last
column gives their ratio, first over second, so `ENGINE=orangu,reference`
reads as "above 1, orangu ahead":

```text
**tg** (tok/s)

| model | depth | orangu | reference | orangu / reference |
| :-- | --: | --: | --: | --: |
| gemma-4-E2B-it-Q4_K_M | 0 | 70.8 (70.6 ± 0.2) | 68.0 (67.9 ± 0.1) | 1.04× |
```

Labels without that shape render as one row each under a single `best`
column, so an ordinary run's history still tabulates. As with the chart, only
the latest measurement of each point is shown: the file keeps every run, the
table answers what the build that exists does now.

### The run as a document (`--report`)

A rate travels badly. It goes into a pull request, an issue, or a mail to
someone with different hardware — and the parts that make it *checkable* are
the parts a terminal cannot carry: the chart it sits on and the profile that
explains it. `--report FILE.pdf` writes all of it as one file:

```sh
orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024 --gen 128 \
             --chart run.svg --flamegraph run-profile.svg \
             --report run.pdf
```

The document holds, in order:

- **What produced it** — the URL, model, backend, the server's **build**
  (version and commit, see *What was measured* above), its pid and uptime, the
  GPU kernels it selected, each card's clock and DPM mode, the host, the
  workload spelled out, **when it was measured** (to the second, in UTC) and
  this tool's own build. A date alone cannot separate two runs taken on the
  same afternoon, which is when an A/B is usually taken — so the bundle records
  the instant, and a report rebuilt from it reads that instant back.
- **What it measured** — best, mean and ± sd per point, with the unit column,
  under a line saying that the headline is the *best* of the repetitions.
  A card measured at `power_dpm_force_performance_level = auto` adds a caution
  here, because that alone makes the numbers incomparable with a pinned run.
- **Throughput** — the chart, embedded.
- **Where the time went** — the profile's samples, window, cores busy and what
  it was waiting on, its self-time buckets, and the flamegraph, embedded.

The same document can be rebuilt later from what the run archived:

```sh
orangu-bench --read-bundle run/bundle.json --report run.pdf
```

A single `--read-bundle` with `--report` writes a *run* report rather than a
comparison — one bundle is not a comparison — and looks for the pictures beside
the bundle, which is exactly how a run's directory is laid out. That is what
the console's **Report** button runs.

A PDF rather than a markdown table for exactly one reason: **it folds the PNGs
in**. The chart and the flamegraph are written as SVG (the canonical artifact)
and rasterized beside them; the report embeds the raster, rendering it on
demand, so `--report` needs neither `--chart-png` nor `--flamegraph-png` to
produce a complete document. On a machine with no `rsvg-convert` the report is
still written, with a line where each image would have been — the measurements
are the deliverable and are never withheld because a picture could not be made.

### Curve mode (`--curve`) — decode scaling without prefill

The depth sweep pads the *prompt* to reach a context depth, which means a large,
slow, VRAM-heavy prefill on orangu (its multi-hundred-token prefill is
CPU-orchestrated). `--curve N` avoids that entirely: it does **one** generation
of `N` tokens, timestamps each streamed token, and reports the instantaneous
decode rate per `--bucket`-token context window. That is the cleanest way to see
decode-vs-context scaling, and it works identically against any
OpenAI-compatible server.

```sh
orangu-bench --curve 3072 --bucket 256   # decode rate at ctx 0, 256, 512, …, 2816
```

```text
orangu-bench → http://127.0.0.1:8100 (curve: 3072 tokens, bucket 256)
     ctx |    tok/s
------------------
       0 |    29.29
     256 |    24.10
     512 |    23.47
     ...
```

Context position is approximated by the generated-token index (the prompt is
short). `--json` emits `{"ctx":…,"tok_per_s":…,"tokens":…}` per bucket.

Each bucket is also recorded to `--history` under its own **`curve`** mode and
drawn on its own chart panel. That makes a decode-vs-context before/after
affordable on a slow build: the depth sweep reaches a context by *padding the
prompt*, so every row costs a full prefill at that depth **per repetition**,
which on a model whose prefill is itself the thing under investigation is both
slow and a measurement of the wrong phase. One generation gives the whole curve
for one prefill.

It is a separate panel from `tg` rather than extra points on it, even though
both are decode rates against context in the same unit. A `tg` row is
best-of-N; a curve bucket is a **single-pass instantaneous rate**, so `best`
equals `mean` and `sd` is `0` by construction. Merging them would present two
different statistics as one series — and since the chart reduces by `best` per
`(label, n)`, the noisier single sample would win wherever the two overlapped.

### Confirm you are measuring the build you think you are

The header line

```
  server   pid 2623608 up 2s
```

is there to be checked, not skipped. When A/B-ing two builds, the **pid must
change** between the two runs and `up` must be a few seconds, not minutes.

The failure it catches is quiet and convincing. Stop the old server by process
name (`pkill -x orangu-server`) and it will miss a build you copied to another
filename — `pkill` matches the *binary's* name, not the path. The replacement
server then fails to bind the port, exits, and `orangu-bench` measures the
server that is still running: the old one. Both runs report the same numbers,
which reads as a clean "this change did nothing" result rather than as a broken
measurement. Keeping both builds named `orangu-server` in separate directories
avoids it; checking the pid proves it.

A server that does not report these fields gets no such line — there, check
the process yourself.

### Interpreting a comparison

Run the same sweep against both servers and compare **the shape of the curve**,
not just the top-of-context point. Two builds (or two engines) that start at a
similar short-context rate but diverge as depth grows differ in how their
attention / KV path scales, not in their per-token matmul — which is the
distinction that matters when deciding what to optimize. The overall
performance investigation this tool supports lives in
`doc/SERVER_ROADMAP.md`.

### Measuring kernel occupancy (register pressure), not just throughput

`orangu-bench` measures end-to-end **throughput**, which on a laptop dGPU is at
the mercy of the GPU's power state — if the core clock isn't pinned at its
maximum, two runs minutes apart are not comparable (check
`cat /sys/class/drm/card1/device/pp_dpm_sclk` and confirm the `*` is on the top
frequency). When the question is instead *why* a compute kernel is slow — its
register (VGPR) count and occupancy — there is a **clock-independent** measure:
the RADV driver's compile-time shader statistics.

```sh
# Print per-pipeline VGPR/SGPR/occupancy as RADV compiles each kernel.
# Run it through the cross-check TEST, not the server: the test builds the
# GPU backend and compiles the pipelines, and is immune to model load and to
# the occasional flaky long-lived server startup. `,nocache` forces a fresh
# compile every run (RADV otherwise serves the stats-less disk cache).
RADV_DEBUG=shaderstats,nocache \
  cargo test --bin orangu-server matmul_matches_cpu_backend_for_q4_k -- --nocapture
```

To attribute a stats block to a specific kernel, capture with the kernel's env
flag on and off (e.g. `ORANGU_Q4K_LIGHT=1` vs unset) and diff the `VGPRs:` /
`Code size:` blocks — the one that appears only with the flag on is that kernel.
`ORANGU_DUMP_SHADERS=<dir>` additionally writes each kernel's generated WGSL to
`<dir>` for inspection. This is the harness the `doc/SERVER_ROADMAP.md` Step 16
work used to settle a kernel's occupancy without trusting a throttled
throughput number.

### Requirements and caveats

- Use `temperature 0` semantics: the tool always sends `temperature: 0` so runs
  are deterministic and comparable.
- It sends both `max_tokens` (OpenAI) and `n_predict` (the widely-used native
  spelling) so a server honors whichever it recognizes.
- Force the GPU to a stable clock state before benchmarking, or the numbers
  reflect the governor, not the code (see `orangu-server`'s startup power-state
  advisory).
- The tool disables prompt caching (`cache_prompt: false`) so each run
  re-establishes its context rather than reusing a cached prefix.

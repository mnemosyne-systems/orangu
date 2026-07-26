\newpage

## Benchmarking throughput (`orangu-bench`)

`orangu-bench` (`src/bin/orangu-bench/`) is a **developer tool** — a fourth
binary in the same Cargo package as `orangu`, `orangu-coordinator`, and
`orangu-server`. It is not part of the served product and has no bearing on
running a model in production; it exists to answer one question during
performance work: *how fast does token generation (decode) run, how does that
rate change as the context grows, and how fast is prompt processing
(prefill)?*

It is the HTTP-client analogue of `llama.cpp`'s `llama-bench -n` (its `tg`,
token-generation, test). Rather than embedding an inference engine, it points
at a **running OpenAI-compatible server** over HTTP and measures the tokens
per second it streams back. Because both `orangu-server` and `llama-server`
speak `POST /v1/completions` with SSE streaming, the *same* tool measures both
through the *same* path — the only way to get a genuinely apples-to-apples
comparison (in-process `llama-bench` numbers and an ad-hoc `curl` of orangu are
not comparable).

### What it measures

For each run, `orangu-bench` sends one streaming completion and times the
window **from the first streamed token to the last**. Prompt processing
(prefill) and time-to-first-token are therefore *excluded* from the reported
rate — the number is steady-state decode throughput, `(tokens - 1) /
decode_seconds`, exactly the quantity `llama-bench`'s `tg` reports. Time to
first token is printed separately (`ttft_ms`) so prefill cost is still visible.

To see how decode scales with context, it sweeps **depths**: each depth pads
the prompt with filler so generation begins at roughly that many tokens of
context, mirroring `llama-bench -d`. A flat curve across depths means decode is
context-insensitive; a curve that falls with depth means attention or KV
traffic is growing per token.

> The depth padding is approximate — it appends `~depth` filler words
> (≈ one BPE token each) rather than exact tokens, because the tool has no
> tokenizer and talks only HTTP. It is close enough to compare *slopes*
> between two engines or two builds; it is not an exact context length.

### Usage

Start the server you want to measure, then run the tool against its base URL.

```sh
# orangu-server (default port 8100): sweep decode rate across context depths
orangu-bench --url http://127.0.0.1:8100 --depths 0,512,1024,2048,3072 --gen 128

# llama-server on port 8300, identical harness (uses the OpenAI-compat endpoint)
llama-server -m model.gguf -ngl 99 --port 8300 -c 4096
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

### Prefill mode (`--pp`) — prompt processing, not decode

`--pp` sweeps *prompt lengths* and reports **prompt-processing** throughput —
`llama-bench`'s `pp` test to the default's `tg`. Each run sends a prompt of
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
  sends `cache_prompt: false`, which both `orangu-server` and `llama-server`
  honour, so this column should read `0`; if it ever climbs toward `n_tok`,
  the server ignored the flag and the row is measuring a cache lookup rather
  than prefill.

A server that reports no `timings` at all (an older `orangu-server`, or a
llama-server built without them) gets a row marked `no server timings (ttft
only)`, carrying wall-clock time-to-first-token instead. That figure includes
queueing and the first decode step, so it is not comparable with the rest — the
row says so rather than quietly printing a smaller number.

### What was measured

Every run opens with the server's model and backend (from `GET /props`) and,
on Linux with an AMD card, each GPU's current core clock and DPM mode read from
`/sys/class/drm/card*/device/`. This is not decoration: a card left at
`power_dpm_force_performance_level = auto` can idle its clock down between
requests, which moves throughput by more than the difference most benchmarks
are run to detect. A rate recorded without the device and clock state beside it
cannot be compared against one recorded later. Under `--json` the same
information is the first object emitted, tagged `"type": "env"`, so a stored
result carries its own provenance.

### Options

`orangu-bench --help`:

```text
Usage: orangu-bench [OPTIONS]

Options:
      --url <URL>          Base URL of the server [default: http://127.0.0.1:8100]
      --depths <DEPTHS>    Comma-separated context depths to sweep [default: 0]
      --pp <PP>            Prefill mode: comma-separated prompt lengths to sweep, reporting prompt-processing throughput
      --gen <N_GEN>        Number of tokens to generate per timed run [default: 128]
      --curve <CURVE>      Curve mode: ONE generation of this many tokens, decode rate bucketed by context [default: 0]
      --bucket <BUCKET>    Bucket width (in context tokens) for --curve [default: 256]
      --reps <REPS>        Repetitions per depth; the reported rate is the best run with mean±sd [default: 3]
      --no-warmup          Skip the initial warmup run
      --timeout <TIMEOUT>  Per-request timeout in seconds [default: 600]
      --model <MODEL>      Model id to request
      --json               Emit machine-readable JSON
      --history <HISTORY>  Append each measured point to this tab-separated history file
      --label <LABEL>      Series name recorded in the history file
      --chart <CHART>      Render the history file to this SVG after measuring
      --chart-only         Only render the chart from an existing history file; measure nothing
  -h, --help               Print help
  -V, --version            Print version
```

Notes: `--url` is the server base URL (the tool appends `/v1/completions`);
`--depths` is comma-separated (e.g. `0,512,1024,2048`); `--reps` reports the
best (fastest) run with mean ± standard deviation alongside; warmup (one short
generation) is on unless `--no-warmup`; `--json` emits one JSON object per depth
instead of the table.

### Tracking throughput over time (`--history`, `--chart`)

A rate on its own says nothing. The two things it needs to be read against —
*the other engine* and *last month's build* — are both outside any single
invocation, so `--history` appends each measured point to a tab-separated file
that accumulates across runs, and `--chart` draws the chart from **that file**
rather than from the run that produced it. A run measuring only orangu still
redraws llama.cpp's line beside it.

```sh
# orangu, recorded as its own series
orangu-bench --url http://127.0.0.1:8100 --label "orangu $(git rev-parse --short HEAD)" \
             --pp 128,512,1024,2048 --history perf-history.tsv --chart perf-history.svg

# the reference, same harness, same file
orangu-bench --url http://127.0.0.1:8300 --label "llama.cpp b10104" \
             --depths 0,512,1024,2048 --history perf-history.tsv --chart perf-history.svg

# redraw after hand-editing the file — no server needed
orangu-bench --chart-only --history perf-history.tsv --chart perf-history.svg
```

The file is plain TSV with a `#` header, so it diffs in review and is
hand-editable; blank lines, comments and unparseable rows are skipped rather
than fatal. Each row is `date`, `label`, `mode` (`pp` or `tg`), `n`, and the
best / mean / sd of the run's repetitions:

```text
#date	label	mode	n	best	mean	sd
2026-07-25	orangu af7c767	pp	1120	81.75	81.40	0.26
2026-07-25	llama.cpp b10104	pp	1120	1061.66	1049.40	8.84
```

Nothing is ever rewritten — a row is a measurement that was taken, and a later
run that disagrees is another row, not a correction.

`--label` is the series identity, so it must stay stable across runs for a line
to be drawn; it defaults to the server's model id, which distinguishes two
models but *not* two builds of orangu, so pass it explicitly when A/B-ing
builds. Prompt-processing rows are keyed by the token count the server actually
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

### Curve mode (`--curve`) — decode scaling without prefill

The depth sweep pads the *prompt* to reach a context depth, which means a large,
slow, VRAM-heavy prefill on orangu (its multi-hundred-token prefill is
CPU-orchestrated). `--curve N` avoids that entirely: it does **one** generation
of `N` tokens, timestamps each streamed token, and reports the instantaneous
decode rate per `--bucket`-token context window. That is the cleanest way to see
decode-vs-context scaling, and it works identically against orangu-server and
llama-server.

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

`llama-server` does not report these fields, so the line is omitted for it —
there, check the process yourself.

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
- It sends both `max_tokens` (OpenAI) and `n_predict` (llama.cpp native) so a
  server honors whichever it recognizes.
- Force the GPU to a stable clock state before benchmarking, or the numbers
  reflect the governor, not the code (see `orangu-server`'s startup power-state
  advisory).
- The tool disables prompt caching (`cache_prompt: false`) so each run
  re-establishes its context rather than reusing a cached prefix.

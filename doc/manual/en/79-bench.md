\newpage

## Benchmarking decode throughput (`orangu-bench`)

`orangu-bench` (`src/bin/orangu-bench/`) is a **developer tool** — a fourth
binary in the same Cargo package as `orangu`, `orangu-coordinator`, and
`orangu-server`. It is not part of the served product and has no bearing on
running a model in production; it exists to answer one question during
performance work: *how fast does token generation (decode) run, and how does
that rate change as the context grows?*

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
   depth |   gen | ttft_ms | tok/s(best) |    n_tok |     tok/s(mean±sd)
--------------------------------------------------------------------------
       0 |   128 |     140 |       31.20 |      128 |    31.05 ±  0.12
    1024 |   128 |     520 |       24.90 |      128 |    24.70 ±  0.18
    2048 |   128 |     980 |       20.10 |      128 |    19.95 ±  0.20
```

### Options

- `--url <base>` — server base URL (default `http://127.0.0.1:8100`). The tool
  appends `/v1/completions`.
- `--depths a,b,c` — comma-separated context depths to sweep (default `0`).
- `--gen <n>` — tokens to generate per timed run (default `128`).
- `--reps <n>` — repetitions per depth (default `3`); the reported rate is the
  best (fastest) run, with mean ± standard deviation alongside.
- `--no-warmup` — skip the initial short generation that loads/JITs before
  timing (warmup is on by default).
- `--timeout <secs>` — per-request timeout (default `600`).
- `--model <id>` — model id to request; most single-model servers ignore it.
- `--json` — emit one JSON object per depth instead of the table, for scripting.

### Interpreting a comparison

Run the same sweep against both servers and compare **the shape of the curve**,
not just the top-of-context point. Two builds (or two engines) that start at a
similar short-context rate but diverge as depth grows differ in how their
attention / KV path scales, not in their per-token matmul — which is the
distinction that matters when deciding what to optimize. The overall
performance investigation this tool supports lives in
`doc/SERVER_ROADMAP.md`.

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

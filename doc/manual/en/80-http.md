\newpage

# HTTP endpoints

Everything in the orangu stack talks over HTTP, and this chapter is the
reference for every path any of it serves or calls. The *Client* half is what
`orangu` itself sends and what a program written against these servers has to
get right; the *Server* half is the endpoint-by-endpoint contract.

Four listeners exist, and they are separate sockets with separate rules:

| Listener | Default | Configured by | What is on it |
| :-- | :-- | :-- | :-- |
| `orangu-server` API | `127.0.0.1:8100` | `[orangu-server].host`/`port` | OpenAI-compatible, native, diagnostic and file-lifecycle endpoints |
| `orangu-server` web console | `127.0.0.1:8101` | `[web].host`/`port` | the chat page, its `/api/…` surface, and the model manager |
| `orangu-coordinator` | `all:9000` | `[orangu-coordinator].host`/`port` | two coordinator endpoints, a shutdown endpoint, and a proxy for everything else |
| `orangu-bench` console | `127.0.0.1:8300` | `--host`/`--port` | the benchmark console and its `/api/…` surface |

Only the first is an API in the sense of something to build against. The two
consoles serve their own page's JavaScript and nothing else promises to stay
stable; the coordinator is a front door for the first.

## Client

### Where to point a client

A server section in `orangu.conf` names an `endpoint`, and it may be written
with or without the `/v1` suffix — `http://localhost:8100` and
`http://localhost:8100/v1` are the same host, normalized internally before any
path is appended. Each server section must use a unique endpoint after that
normalization.

Point a client at the **coordinator's** address instead and nothing about the
requests changes: it forwards them to whichever `orangu-server` its routing
picks. See **Coordinator** below.

### Authentication

When `[orangu-server].api_key` (or `ORANGU_API_KEY`) is set, every endpoint
except `GET /health` and `GET /ready` requires:

```
Authorization: Bearer <key>
```

and answers `401` with `WWW-Authenticate: Bearer` without it. The `orangu`
client sends its own `api_key` this way on every request, including the
`/v1/models` probe, and so does any OpenAI-shaped client.

The **web console** port has no key of its own and is not loopback-restricted
— it assumes a trusted network. A server reachable from an untrusted one
should not have a `[web]` section at all.

### What `orangu` calls, and when

| Endpoint | When the client sends it |
| :-- | :-- |
| `GET /v1/coordinator` | once per endpoint to tell a coordinator from a plain server, then every status cycle against a confirmed coordinator |
| `GET /v1/models` | startup model detection, `/model` completion, `/server`, and as the health check against a plain server |
| `POST /v1/chat/completions` | every prompt and every tool round trip |
| `POST /v1/embeddings` | `/search`, when a server has the `embeddings` role (or the default `all` role) |
| `GET /props` | once per endpoint, to learn the slot count for pinning |
| `POST /slots/{id}?action=save\|restore` | to keep a workspace tab's KV cache across restarts |
| `GET /health`, `/slots`, `/metrics` | the `/information` report only |

`/information` is the command that probes all of these and prints one row per
capability — see the *Core tools* chapter. A plain OpenAI-compatible server
answers only the `/v1/…` rows, and the rest are reported unavailable rather
than failing the command.

**The coordinator probe is asked once, not every cycle.** The client's status
refresh runs on a timer — every 60 s idle, every 500 ms while the startup git
sync is in flight — and `/v1/coordinator` exists on nothing but a coordinator,
so probing it each cycle is a `404` each cycle against a plain server. The
client remembers what each endpoint answered: a known-plain one skips the
probe entirely and lets `GET /v1/models` serve as both the model list and the
health check, halving the requests a poll costs. A *confirmed* coordinator is
still probed every cycle, since for it this endpoint is the health check and
`/v1/models` is deliberately never called.

The memo is dropped the moment an endpoint stops answering, so a server
restarted as a coordinator — or a coordinator that simply was not up yet when
the client started — is identified afresh rather than assumed to be whatever
was there before. `/server` drops it too, in both its forms: listing the
servers re-opens the question for the active endpoint, and `/server <name>`
for the one it selects, including when that is the server already in use.
Listing or selecting a server is the moment a user is asking about the
connection rather than working through it, and the moment the answer is most
likely to have just changed. The memo keeps an unchanging answer from being
re-asked on a timer; it is not meant to outlast a direct question about it.

### Slot pinning from a client

Each workspace tab probes `GET /props` once per endpoint, takes a slot
round-robin from `total_slots`, and pins every request in that tab to it with
`id_slot`, so tabs stop evicting each other's KV cache. One-shot requests
(`orangu -p`) do not pin — there is no later turn to keep a cache warm for.
See `id_slot` under `POST /v1/chat/completions` for what that buys and what it
costs.

### Pre-warming through the coordinator

`orangu` fires `POST /v1/coordinator/activate` at the start of `/review` and
`/auto_review`, naming the `review` role, so a cold load happens behind local
work (diff collection, the auto-review prestart screen) instead of stalling
the first real request. It is fire-and-forget: any failure is ignored, since
the real request that follows triggers the same swap on its own if the hint
did not get there first.

### The file-lifecycle API from a client

The eight file endpoints, the model's `create_file`/`modify_file`/… tools, and
the typed `/create_file`, `/delete_file`, "create myfile.txt with 0644" style
commands are **one implementation** with the same fields, defaults and errors.
Whatever you learn from one surface holds for the other two; the fields are
documented once, under **File-lifecycle API** below.

That router is library code rather than something the server binary owns, so a
host that wants the same eight endpoints in its own axum service can mount
them: `orangu::files_http::router::<S>()`, where `S` implements
`orangu::files_http::WorkspaceState` and names the workspace root every
request is resolved against and confined to. `orangu-server` mounts it beside
its inference endpoints; `orangu-coordinator` forwards to it.

## Server

### Rules that apply to every endpoint

Three things apply to the whole API port rather than to any one path, and are
easy to miss looking down a table:

- **`401`** — the bearer-token rule above.
- **`503`** — when `[orangu-server].queue_limit` is set and that many requests
  are already waiting for a slot, a generating endpoint answers `503` with
  `Retry-After: 1` instead of joining the queue. `503` rather than `500`
  deliberately: this request could be served later, and the distinction is the
  whole point of bounding the queue. Requests pinned with `id_slot` bypass the
  limit — they are waiting for one specific slot's warm cache, not competing
  for admission.
- **`https`** — when `tls_cert`/`tls_key` are set, every endpoint is served
  over TLS on the same port; there is no plaintext listener alongside.

And one that applies to the generating endpoints: an `embedding`-role server
answers `POST /v1/chat/completions`, `/v1/completions` and `/completion` with
**`501`** naming the mode, rather than pretending to generate.

### Endpoints at a glance

| Endpoint | |
| :-- | :-- |
| `GET /v1/models` | the one model this process serves |
| `POST /v1/chat/completions` | streaming (SSE) and non-streaming; `tools`/`tool_calls` and `response_format`; `cache_prompt`/`id_slot`/`timings_per_token`/`return_progress`; needs a chat template; disabled under `--embedding` |
| `POST /v1/completions` | legacy completion, no chat template needed; `cache_prompt`/`id_slot`/`response_format`/`ignore_eos`; disabled under `--embedding` |
| `POST /v1/embeddings` | pooled and L2-normalized; carries OpenAI's `usage` |
| `GET /health` | liveness: is this process up. Stays `200` while the server is merely busy |
| `GET /ready` | readiness: would a request sent now be served |
| `GET /props` | model and server metadata: backend, devices, build, slot count, workspace |
| `GET /slots` | per-slot busy/prompt/generated-token state |
| `POST /slots/{id_slot}` | `?action=save\|restore` — persist or reload that slot's KV cache |
| `GET /metrics` | Prometheus text: gauges, latency histograms, outcome and token counters |
| `POST /completion` | native, streaming; `cache_prompt`/`id_slot`; disabled under `--embedding` |
| `POST /embedding` | native embeddings |
| `POST /tokenize`, `POST /detokenize` | text to token ids and back |
| `POST /apply-template` | renders the chat template without generating |
| `POST /v1/create_file` … `/v1/delete_directory` | the eight file-lifecycle endpoints |
| `GET /moe-stats` | mixture-of-experts counters since the previous call, and reset |
| `GET /gpu-timings` | per-stage GPU timings since the previous call, and reset |
| `GET /model-cache` | how much of the model is in the page cache right now |
| `POST /model-cache/drop` | evict the model from the page cache; loopback-only |
| `POST /v1/shutdown` | stop the server; loopback-only |

### OpenAI-compatible endpoints

#### `GET /v1/models`

The model this process serves, in OpenAI's listing shape. One entry, always —
`orangu-server` loads exactly one model, and switching to another restarts the
process (see **Loading a different model** in the *Inference server* chapter).

```json
{"object":"list","data":[{"id":"ggml-org/gemma-4-E4B-it-GGUF","object":"model",
 "created":1755600000,"owned_by":"orangu-server"}]}
```

#### `POST /v1/chat/completions`

The endpoint every prompt and tool round trip actually goes through. It needs
the model to carry a `tokenizer.chat_template`; without one it answers `501`
and points at `/v1/completions`.

| Field | Default | |
| :-- | :-- | :-- |
| `messages` | `[]` | the conversation, in OpenAI's shape |
| `stream` | `false` | SSE instead of one JSON body |
| `max_tokens` | `8192` | response-length cap, clamped to what is left of the context window; reaching it reports `finish_reason: "length"` |
| `temperature`, `top_p` | role's own | OpenAI's sampler fields |
| `top_k`, `min_p`, `repeat_penalty`, `seed` | role's own | the rest of the sampler |
| `cache_prompt` | `true` | may this request reuse an already-computed KV prefix |
| `id_slot` | — | pin to one slot |
| `timings_per_token` | `false` | attach `timings` to every streamed chunk |
| `return_progress` | `false` | emit `prompt_progress` chunks during prefill |
| `response_format` | — | `{"type": "json_object"}` constrains output to JSON |
| `tools` | — | OpenAI's tool array, handed to the chat template |

`top_k`, `min_p`, `repeat_penalty` and `seed` are **not** OpenAI's fields, and
they are accepted anyway: the sampler has knobs OpenAI's schema has no word
for, and an unknown key is dropped silently rather than rejected. A caller
sending `repeat_penalty` and seeing no change would conclude the penalty does
nothing, when in fact the value never arrived. Every sampler field is
optional in the strict sense — leaving one out keeps the role's default, which
is not the same as setting it to zero, since `temperature: 0.0` is greedy and
`repeat_penalty: 1.0` is off.

A non-streaming reply is `chat.completion`:

```json
{"id":"chatcmpl-1755600000","object":"chat.completion","created":1755600000,
 "model":"…","choices":[{"index":0,
   "message":{"role":"assistant","content":"…"},"finish_reason":"stop"}],
 "usage":{…},"timings":{…}}
```

A streaming reply is a series of `chat.completion.chunk` events, each carrying
a `delta`, ending with a chunk that carries `finish_reason` (plus `usage`,
`timings` and `prompt_progress`) and then `[DONE]`.

##### `reasoning_content`

A reasoning model's chain of thought is reported **apart from the answer**,
never inside `content`. Whole, it is `message.reasoning_content`; streaming,
it arrives as `delta.reasoning_content` chunks, which come before the
`delta.content` ones the answer arrives in. The field is absent — not empty
— when the model wrote no thinking, so "does not reason" and "reasoned about
nothing" stay distinguishable, and a client that does not know the field
ignores it and sees the answer alone.

This covers every format that *names* a reasoning body: a
`<think>`…`</think>` span, a message addressed `to=self`, a body opened with
`<|content_thinking|>`. Thinking never passes through the tool-call splitter
— a call is something the model addresses to the caller, not part of a body
it addressed to itself — so `tool_calls` are parsed from the answer only.

##### `cache_prompt`

`/v1/chat/completions`, `/v1/completions` and `/completion` all accept it, and
it defaults to `true`. It controls whether a request may **reuse** an
already-computed KV cache for whatever prefix of its prompt one exists for —
the cross-slot prefix pool, or a slot's own retained cache. Leaving it at the
default is what makes a growing conversation cheap: only the new suffix is
processed.

Set it `false` to force the whole prompt through a real forward pass. That is
what a prefill measurement needs, since a cached prompt is reported as
processing thousands of tokens per second while doing almost nothing —
`usage.prompt_tokens_details.cached_tokens` and `prompt_progress.cache` show
exactly how much was skipped. The flag governs only what a request *reads*:
the resulting cache is still stored for later requests either way.

##### `id_slot`

`/v1/chat/completions`, `/v1/completions` and `/completion` accept `id_slot`,
pinning a request to one specific slot instead of letting it take whichever is
free. An unknown slot number is a `400`, not a silent fallback.

What it buys is **cache affinity**. A slot retains the `(tokens, KV cache)` of
the last request that ran on it, so a conversation that returns to its own
slot continues from a warm prefix and prefills only the new turn. Landing on a
neighbour instead finds another conversation's cache there and reprefills the
whole prompt — and since an idle server hands out the *lowest* free slot, two
alternating conversations otherwise both land on slot 0 and evict each other
every turn.

Two conversations interleaved on a two-slot server, three turns each
(`gemma-4-E2B-it:Q4_K_M`, ~430-token prompts):

| | tokens actually prefilled | prefill time |
| --- | ---: | ---: |
| without `id_slot` | 2 567 | 13.4 s |
| with `id_slot` | **889** | **5.0 s** |

Steady-state per turn is where it shows: 2.0 s of prefill becomes 0.25 s,
because the whole previous turn is served from the slot's own cache
(`cached_tokens` 417 of 433, rather than 7).

A pinned request **waits** for its slot rather than being bounced to a free
one — that is the point, and it is a trade the caller has already chosen.
Waiting costs no one else any concurrency: a queued request holds nothing.

##### `timings_per_token` and `return_progress`

Both apply to a **streaming** request, and both exist for the same reason: the
longest part of a turn is the part a client otherwise knows nothing about.

`return_progress: true` emits a `prompt_progress` chunk after every prefill
chunk (`ORANGU_PREFILL_BATCH` tokens, or the width the backend and the model's
residency choose when that is unset) rather than only at the
end. `processed` counts cached tokens as already done, so a mostly-cached
prompt does not appear to start from zero. On a 2712-token prompt that is six
updates across a 12.8-second prefill:

```
 512/2712   2725 ms      2048/2712   9573 ms
1024/2712   4996 ms      2560/2712  11904 ms
1536/2712   7274 ms      2712/2712  12845 ms
```

`timings_per_token: true` attaches a `timings` object to every generated
token, not just the last chunk, so a client can display a live decode rate
measured by the server. The first token is deliberately skipped: it was
sampled from the prefill's own logits, so a rate computed there is a division
by a few microseconds and comes out in the tens of thousands of tokens per
second.

Each arrives as its own chunk with an empty `delta`, which is what lets them
keep flowing while content is briefly held back by the tool-call splitter.

The `orangu` client requests both. They are what its status line shows: a
`n/total tok` prefill bar while the prompt is processing, then the server's
`predicted_per_second` while the answer streams.

##### Structured output (`response_format`)

`POST /v1/chat/completions` and `/v1/completions` accept OpenAI's
`response_format`. With `{"type": "json_object"}` the server does not ask the
model for JSON — it makes anything else **unsampleable**:

```sh
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Give me a person record with a name and an age."}],
  "response_format": {"type": "json_object"},
  "temperature": 0
}'
```

Unconstrained, the same request returns `Here is a person record with a name
and an age:` followed by prose. Constrained, it returns
`{"Name": "John Doe", "Age": 32}`.

At every step the sampler is offered only tokens that keep the output a prefix
of some valid JSON object, and the end-of-sequence token is withheld until the
document is **complete** — so a model cannot stop at `{"a":` and leave a
caller parsing a fragment. Once the document is complete, stopping is the only
move left, which is what keeps a model from trailing blank lines to
`max_tokens` after it has finished.

`json_object` means an **object**, matching the field's name: a bare string is
technically valid JSON, and a caller asking for a record and receiving
`"Name: Sophia, Age: 32"` gets something that parses and then fails at
`result["name"]`, which is worse than failing outright. `json_schema` is
accepted and treated the same way — output is constrained to valid JSON, but
**not** to the schema's shape. Constraining to a schema is a larger job and is
not implemented; the type is honoured rather than ignored so that a caller
asking for it is not silently given free text.

Two limits worth knowing. A constrained request cannot use the GPU's argmax
fast path or prompt-lookup speculation, because both pick tokens without
consulting the constraint — so constrained decoding is somewhat slower than
unconstrained. And a constraint makes invalid output unreachable; it cannot
make a model cooperate. Told *"reply with the single word hello, do not use
JSON"*, the model spends its budget on whitespace and returns no document at
all — the response is then not valid JSON, and `finish_reason` is `length`.
Ask for JSON in the prompt as well as the parameter.

##### Tool calling

`/v1/chat/completions` accepts OpenAI's `tools` array and answers with
OpenAI's `tool_calls`. Nothing about the tools themselves is interpreted here:
the array is handed to the model's own `tokenizer.chat_template` as the
`tools` variable, which is what every tool-capable template gates its
declaration block on (`{%- if tools -%}`). A model whose template has no tool
support simply ignores it. An empty `tools: []` counts as no tools.

Messages carry the other half of the conversation:

| Field | On | Meaning |
| :-- | :-- | :-- |
| `tool_calls` | `assistant` | the calls that turn made, passed to the template verbatim |
| `tool_call_id` | `tool` | which call this message answers |
| `name` | `tool` | the function's name; some templates use it directly, others resolve it from `tool_call_id` |

All three are required for a **multi-turn** tool conversation. Without them
the transcript replayed on turn N+1 shows an assistant message with empty
content and no record of any call, and the model calls the same tool again.

**Reading the model's answer back.** There is no standard for how a model
*writes* a call — its template teaches it one, and the forms differ far more
than the OpenAI shape they all become. Six delimiter-anchored forms are
recognised:

| Family | Form |
| :-- | :-- |
| gemma-4 | `<\|tool_call>call:NAME{key:value,…}<tool_call\|>` (the markers are special tokens; `call:` is optional) |
| Qwen / Hermes | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| GLM, Ling 3.0 | `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>` (Ling spells all six delimiters as special tokens) |
| Nemotron | `<tool_call><function=NAME><parameter=k>v</parameter>…</function></tool_call>` |
| Mistral | `[TOOL_CALLS][{"name": …, "arguments": {…}}]` |
| Muse-Glimmer, DeepSeek-V4 | an `<…invoke name="NAME">` block of `<…parameter name="k">v</…>` elements, in each model's own tag namespace |

gemma-4's string values are delimited by its own `<|"|>` token, not by a
plain `"`, and are taken **verbatim** between the two — the template escapes
nothing writing them, so nothing is unescaped reading them. That is what lets
a `create_file` carry a source file containing quotes, braces and backslashes;
a value the model delimits with a plain `"` instead is still accepted, with
JSON-style escapes, but there a quote inside the value ends it early.

Note the three that share `<tool_call>`: the delimiters are the same and the
bodies are not, so the body's own leading structure decides which it is. A
value is read as JSON where it parses as JSON and as a plain string otherwise,
so `3` stays a number and a sentence stays a sentence; where a format marks a
parameter `string="true"`, that wins, and a version like `1.20` is not quietly
turned into `1.2`.

Only these delimiters count. A bare JSON object that merely *looks* like a
call is left as ordinary content — an answer that explains an API must not be
mistaken for a request to invoke one, and a model asked to *write* a tool call
has to be able to. A span that opens and never closes, or whose body matches
none of the forms, is also left as content rather than silently dropped.

A turn that produced calls reports `finish_reason: "tool_calls"` and carries
them in `choices[0].message.tool_calls` (non-streaming) or in a
`delta.tool_calls` chunk (streaming). `function.arguments` is a JSON
**string**, as OpenAI specifies. Streaming emits each call complete in one
delta rather than character by character, since a call is only recognised once
it is fully written.

#### `POST /v1/completions`

The legacy completion endpoint: a raw `prompt` string, no chat template
involved, which makes it the one endpoint that works against a model with no
`tokenizer.chat_template` at all. It is also what `orangu-bench` measures
through.

| Field | Default | |
| :-- | :-- | :-- |
| `prompt` | *required* | the raw prompt, sent to the tokenizer as-is |
| `max_tokens` | `256` | response-length cap |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed` | role's own | the sampler, as above |
| `stream` | `false` | SSE instead of one JSON body |
| `ignore_eos` | `false` | keep generating to `max_tokens` even if the model emits EOS |
| `cache_prompt` | `true` | as above |
| `id_slot` | — | as above |
| `response_format` | — | as above |

`ignore_eos` exists for benchmarks: timing a fixed number of decode steps at a
given context depth needs the step count to be the same whatever the model
would otherwise have stopped on.

The reply is `text_completion`, with the generated text in
`choices[0].text` — one JSON body, or a stream of chunks ending with a final
chunk carrying `usage`/`timings`/`prompt_progress` and then `[DONE]`.

#### `POST /v1/embeddings`

```sh
curl -s localhost:8100/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["first text", "second text"]}'
```

`input` is a string or an array of strings. Each vector is pooled — mean or
last-token, per the model's own `pooling_type` — and L2-normalized, and comes
back in OpenAI's `data` list with its `index`. The reply carries OpenAI's
embeddings `usage`, which has `prompt_tokens` and `total_tokens` and no
`completion_tokens`, summed across a batched `input`. That count is not
recoverable from the response otherwise, so without it anything measuring
embedding throughput would have to tokenize the input a second time with
another tool to learn what it just paid for.

This is the endpoint `/search` embeds code through, and the one a server
tagged `role = embeddings` exists to serve.

### What a request cost

Every generation endpoint reports what the request cost, so a client never has
to infer it from its own wall clock — which cannot separate prompt processing
from generation, nor a cache hit from real work:

- **`usage`** (OpenAI's shape) — `prompt_tokens`, `completion_tokens`,
  `total_tokens`, and `prompt_tokens_details.cached_tokens` for the part of
  the prompt served from the prefix cache.
- **`timings`** (the ecosystem's shape, field for field) — `prompt_n`,
  `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
  `predicted_per_second` and their per-token equivalents. These are the same
  figures the per-request console log prints.
- **`prompt_progress`** (likewise) — `total`, `cache`, `processed`,
  `time_ms`, reported once per prefill chunk while the prompt is still being
  processed (see `return_progress` above).

On a streaming response they ride on the final chunk (the one carrying
`finish_reason`), immediately before `[DONE]`; on a non-streaming response
they are top-level fields. `orangu-bench --pp` reads them to report prefill
throughput, and the orangu client reads them for its status-line rates.

### Native endpoints

These are `orangu-server`'s own, in the shapes the wider ecosystem's tooling
already reads. They are close enough for `/information`, `orangu-bench` and
`curl` inspection; they are not a byte-for-byte schema match with anything.

#### `GET /health`

```json
{"status":"ok"}
```

Liveness, and nothing more: is this process up. It is what a supervisor uses
to decide whether to restart the server, and it must stay `200` while the
server is merely busy, because restarting a loaded server under load is the
worst possible response to load. Reachable without an `api_key` — a liveness
probe that needs a credential is a liveness probe that fails before
credentials are distributed, and this endpoint names no model and discloses
nothing.

#### `GET /ready`

```json
{"status": "queue full", "queue_depth": 2, "queue_limit": 2, "slots_busy": 1, "slots_total": 1}
```

Readiness: would a request sent now be served. That is what a load balancer
needs, and it is a different question from `/health` — here "busy" is exactly
the case worth reporting.

Two things make it `503`, and the body names which, because a probe that only
flips a status code leaves an operator with the alert and none of the reason:

- **`"queue full"`** — the admission queue is at `queue_limit`. A new unpinned
  request would be refused with `503` anyway, so saying so up front lets a
  balancer send it somewhere that can take it instead of spending a round trip
  to find out. With no `queue_limit` set the queue never refuses, so the
  server is never unready for this reason however deep the queue gets.
- **`"device lost"`** — the GPU device was lost and this process is on its way
  out. It has not stopped being *alive*, and a supervisor restarting it is
  precisely the intended recovery, but nothing should be routed to it
  meanwhile. Checked first: a lost device makes every answer wrong, including
  a cheerful one about an empty queue.

`200` with `"ok"` otherwise. Reachable without an `api_key`, like `/health` —
a readiness probe that needed a credential would fail closed exactly when a
balancer most needs an answer. It is a deliberate widening of what an
unauthenticated caller can see: `/ready` does disclose load, where `/health`
discloses nothing. That one fact is the price of being routable.

#### `GET /props`

Model and server metadata — the closest thing the server exposes over HTTP to
*how it was started*:

| Field | |
| :-- | :-- |
| `model`, `architecture` | which model, and which architecture family reads it |
| `backend`, `gpu` | the backend and device it runs on, every other device that backend saw, `gpu.footprint` for what this model costs on it, and `gpu.kernels` for which kernel each quantization resolved to. `null` on a backend with no kernel selection to report |
| `version`, `commit` | which build answered |
| `n_ctx`, `n_vocab`, `n_embd` | context length, vocabulary size, embedding width |
| `total_slots` | how many concurrent requests, and the range `id_slot` may name |
| `chat_template` | the template source, or `null` when the model carries none |
| `workspace` | the root the file-lifecycle endpoints resolve against |
| `uptime_seconds`, `pid` | which process answered, and for how long it has been up |

`version` dates the release; `commit` is the only field that tells two builds
of one version apart, which during performance work is every pair of builds
that matters. Paired with `pid` and `uptime_seconds` it is what lets a
benchmark prove it is talking to the build it just launched rather than one
left over from a previous run.

Hardware-only startup flags — thread count, GPU layer count, batch size — are
not exposed here or anywhere else over HTTP; they appear only in the server's
own startup log.

#### `GET /slots`

One entry per slot: `id`, `busy`, `prompt_tokens`, `generated_tokens`. This is
the live view behind `/information`'s `/slots` row and the arithmetic
`/metrics` reports as gauges.

#### `POST /slots/{id_slot}?action=save|restore`

Persists a slot's KV cache to, or restores it from,
`~/.orangu/server/<fingerprint>/slots/<filename>`. The body is a JSON object
with one field:

```sh
curl -s -X POST 'localhost:8100/slots/0?action=save' \
  -H 'Content-Type: application/json' -d '{"filename": "tab-a.bin"}'
```

```json
{"id_slot":0,"filename":"tab-a.bin","n_saved":433}
```

`restore` answers the same shape with `n_restored`. A missing or
model-incompatible saved file is **not** an error: `restore` succeeds with
`n_restored: 0`, so a stale sidecar never trips a client's "persistence
unavailable" notice and simply costs a full reprefill.

`501` when durable slot persistence is off (`ORANGU_NO_SLOT_SAVE`, or no
resolvable home directory); `400` for an out-of-range `id_slot`, an empty or
missing `filename`, or an `action` that is neither `save` nor `restore`.
Persisted files are swept by `orangu-server prune` once untouched for 30 days
— they are a pure reprefill-avoidance cache, so an over-eager sweep only ever
costs a one-time prefill.

#### `GET /metrics`

Prometheus text (`text/plain; version=0.0.4`). Four gauges describe the
scheduler right now:

| gauge | |
| :-- | :-- |
| `orangu_server_slots_total` | configured concurrent request slots |
| `orangu_server_slots_busy` | slots currently generating |
| `orangu_server_queue_depth` | requests waiting for a slot |
| `orangu_server_queue_limit` | waiting requests allowed before refusing; `0` is unbounded |

Beyond them it carries four latency **histograms**, which is what makes a
latency question answerable at all: a mean is dominated by whichever requests
happened to be long, and the useful questions are about the tail.

| metric | what it measures |
| :-- | :-- |
| `orangu_server_queue_wait_seconds` | arrival to holding a slot |
| `orangu_server_time_to_first_token_seconds` | arrival to the first generated token |
| `orangu_server_inter_token_seconds` | the gap between consecutive tokens, one observation *per token* |
| `orangu_server_request_seconds` | arrival to the last token |

Each is a standard histogram (`_bucket{le="…"}`, `_sum`, `_count`), so a
quantile is a query rather than a setting:

```
histogram_quantile(0.95, rate(orangu_server_time_to_first_token_seconds_bucket[5m]))
```

**Queue wait is the one to reach for first.** "Slow" and "overloaded" look
identical from outside and have opposite fixes: if queue wait is near zero the
requests themselves are expensive (bigger model than the device holds, longer
prompts), and if it is not, the server is short of slots.

Counters carry the totals a rate is taken from:

| counter | notes |
| :-- | :-- |
| `orangu_server_requests_total{outcome="…"}` | `stop`, `length`, `cancelled`, `overloaded`, `error` — every label is exported from the start, at zero |
| `orangu_server_prompt_tokens_total` | prompt tokens accepted, cached or not |
| `orangu_server_cached_prompt_tokens_total` | the part served from a reused KV prefix; the two together are the cache-hit rate |
| `orangu_server_generated_tokens_total` | tokens generated |

`outcome="overloaded"` counts requests refused by `queue_limit` and is the one
to alert on. `outcome="length"` is not an error — it counts answers truncated
by `max_tokens` — but a rate that climbs usually means clients are asking for
less room than the model wants. `outcome="cancelled"` counts clients that
disconnected mid-generation; the server stops within a token of noticing and
frees the slot, so a rate here is not a fault, but one that climbs usually
means a client-side timeout set below what this server can deliver.

#### `POST /completion`

The native generation endpoint. Same engine, a smaller surface, and its own
field names:

| Field | Default | |
| :-- | :-- | :-- |
| `prompt` | *required* | the raw prompt |
| `n_predict` | `256` | response-length cap |
| `stream` | `false` | SSE instead of one JSON body |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed` | role's own | the sampler |
| `cache_prompt` | `true` | as on `/v1/chat/completions` |
| `id_slot` | — | as on `/v1/chat/completions` |

Non-streaming, the reply is `{"content": "…", "stop": true, "timings": {…}}`.
This endpoint has one text field and no message shape to split a reasoning
model's thinking out into, so — unlike `/v1/chat/completions` — thinking
stays in `content`, which is what a caller who prefilled `<think>` here
asked for.
Streaming, each event is `{"content": "…", "stop": false}` and the last is
`{"content": "", "stop": true, "finish_reason": …, "timings": {…},
"prompt_progress": {…}}`. There is no `response_format` here — structured
output is a `/v1/…` feature.

#### `POST /embedding`

`{"content": "…"}` in, `{"embedding": [...]}` out. The vector is pooled and
normalized exactly as `/v1/embeddings` does it; the token count is dropped,
which is what `/v1/embeddings`' `usage` is for.

#### `POST /tokenize` and `POST /detokenize`

```sh
curl -s localhost:8100/tokenize -H 'Content-Type: application/json' \
  -d '{"content": "hello", "add_special": true}'
```

`tokenize` takes `content` and an optional `add_special` (default `false`, for
the model's own BOS-style framing) and answers `{"tokens": [...]}`.
`detokenize` takes `{"tokens": [...]}` and answers `{"content": "…"}`, with
the model's tokenization-space cleanup applied — so a round trip gives back
readable text rather than the tokenizer's internal spelling.

#### `POST /apply-template`

`{"messages": [...]}` in, `{"prompt": "…"}` out: exactly the string
`/v1/chat/completions` would build from those messages and hand to the model,
including a `review`-role server's reasoning-suppression prefill. Nothing is
generated. `501` when the model carries no `tokenizer.chat_template`.

This is the endpoint to reach for when an answer looks like the model was
asked something other than what you sent.

#### `POST /v1/shutdown`

Stops the server — the API listener and, if enabled, the web console listener
together. **Loopback-only**: a `403` from a non-localhost peer, because a
server bound to a non-loopback `host` must not let an arbitrary network peer
kill it. `Ctrl+C`, `SIGINT` and this endpoint all converge on the same
shutdown path.

### Diagnostic endpoints

Four endpoints exist for measurement rather than for serving. Three of them
are **drain-on-read**: they report what accumulated since the previous call
and reset, so a client brackets a window by reading once before the workload
(discarding whatever the warmup left), running it, and reading again.

#### `GET /gpu-timings`

The per-stage GPU timestamp breakdown for the decode steps in the window.

```json
{"enabled": false, "timings": null, "unavailable": "split"}
```

`steps: 0` means no timestamped decode step happened — either
`ORANGU_GPU_TIMESTAMPS=1` is not set, or this adapter has no timestamp query.
Reported as zero *steps* rather than zero milliseconds, so "not measured"
cannot be read as "took no time".

`unavailable` extends that rule one step further out. A **split** model has no
single device to ask — a timestamp query set belongs to one device and a split
resolves none of them — so the reason is named (`"split"`, or
`"no_wgpu_backend"`) rather than left as an empty result. A client that
reports nothing when it receives nothing makes a split run look like one whose
GPU stages cost nothing.

#### `GET /moe-stats`

Mixture-of-experts counters since the previous call, and reset: expert visits,
bytes dequantized, `stats.union_ratio` for the redundancy in the per-token
expert loop, the expert store's `hit_rate`, `route_ahead.accuracy`,
and `process.major_faults_window`.

This is what makes work on models too large for RAM scoreable: throughput
alone cannot separate a change that moves fewer bytes from one that moves the
same bytes faster, and on a model whose weights stream from disk it cannot see
either one over the I/O. `major_faults_window` is the only honest signal that
the weights came from the disk rather than the page cache.

`stats.layer_calls: 0` means no MoE layer ran in the window — a dense model,
or an empty window; `store.hit_rate` and `route_ahead.accuracy` are `null`
unless `ORANGU_EXPERT_RESIDENCY=1` and `ORANGU_ROUTE_AHEAD=1` respectively
asked for them, reporting "not measured" rather than a made-up zero.
`process` is `null` off Linux.

#### `GET /model-cache`

How much of the model's weights are in RAM right now — `model_bytes`,
`resident_bytes`, and a per-`shards` breakdown. Not drain-on-read: this is a
state, not a window. A benchmark records it beside its numbers so a later
reader can tell a cold run from a warm one instead of assuming; on a model
larger than memory those are different experiments and their rates are not
comparable.

`resident_bytes` is `null` where the platform cannot measure it, **never
zero** — "nothing is cached" and "this machine cannot say" would otherwise be
the same answer, and only one of them means the run was cold.

#### `POST /model-cache/drop`

Evicts the model's weights from the page cache, so the next request reads them
from disk. **Loopback-only**, like `/v1/shutdown`: it makes the server
dramatically slower on purpose, which is not something an arbitrary network
peer should be able to do to it.

The reply reports residency **before and after** rather than a success flag,
because a partial drop is the realistic failure and it reads exactly like a
successful one from the outside.

### File-lifecycle API

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

Through `orangu-coordinator` they work exactly the same way — it forwards
them untouched — but the workspace they act in is the backend
`orangu-server`'s, not the coordinator's, which has none. See **Coordinator**
below.

One implementation serves all three surfaces — these endpoints, `orangu`'s
own local tools, and its typed commands of the same names (`create_file`,
`modify_file`, `/delete_file`, "create myfile.txt with 0644", …) — so a tool
call, a typed command and an API request are the same operation with the
same fields, defaults and errors. What follows documents the fields for all
three.

**Everything is confined to the workspace.** Each path in a request is
resolved against the server's workspace root (`-w`/`--workspace`, default
the current working directory — see **Workspace** in the *Inference server*
chapter) and refused if it lands outside it. A path may be given
relative to the workspace (`src/main.rs`) or as an absolute path that is
itself inside it; anything else — a `..` that climbs out, an absolute path
elsewhere on the machine, or a symlink inside the tree pointing out of it —
is a `403 outside_workspace` before any file is touched. Two checks back
that up: a lexical one that folds `..` away before comparing — the same
resolution `orangu`'s own file tools use — and a physical one that
canonicalizes the nearest *existing* ancestor of the target. The nearest
existing one, so it works for `create_file`, whose target does not exist yet
by definition.

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

#### Git integration

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


### Web console

The built-in web console is served on its own `[web].port`, separate from the
API's `port`, and exists only when the config has a `[web]` section. Its
`/api/…` surface is used by that page's own JavaScript; it is **not** part of
the OpenAI-compatible API above, carries no `api_key`, and is not
loopback-restricted.

The page itself and its assets:

| Endpoint | |
| :-- | :-- |
| `GET /` | the console page |
| `GET /static/app.css`, `GET /static/app.js` | its stylesheet and script |
| `GET /static/katex/…` | the vendored math renderer and its fonts, served from an allowlist rather than from disk |
| `GET /api/diagrams/{key}/{name}` | one rendered diagram, kept off the streamed message payload — a PNG's base64 form would grow every event by about a third |
| `GET /api/asset-version` | the served page's own asset fingerprint, which powers the Reload prompt shown when a newer build is running behind an already-open tab |
| `GET /api/system-report` | plain-text hardware report plus model/backend identity — what an error bubble's **Save** button bundles into its downloadable debug report, alongside the visible conversation. Detected fresh on every call, since the parts that change over a long run (VRAM and RAM in use) are exactly the parts worth knowing at the moment a request just failed |

Chat sessions:

| Endpoint | |
| :-- | :-- |
| `POST /api/sessions` | creates a new, empty chat session, returning its id |
| `GET /api/sessions` | lists every non-empty session, newest-updated first |
| `GET /api/sessions/{id}` | one session's full message history, each assistant reply already rendered to HTML |
| `POST /api/sessions/{id}/messages` | sends one chat turn against that session; SSE reply |
| `DELETE /api/sessions/{id}` | deletes one chat session, directory and all — History's per-row cross |
| `DELETE /api/sessions` | deletes every chat session — History's **Clear all** footer |

`POST /api/sessions/{id}/messages` takes `{"content": "…", "attachments":
[…]}`; attachments ride along as base64 in the JSON, which is why this port
raises its body cap to 64 MB. Its SSE events are typed rather than
OpenAI-shaped, because the browser is being handed rendered HTML rather than
raw tokens: an `attachments` event first (what each upload turned into, sent
before the first token so the user's own message can show it while the reply
generates), then a `token` event per token carrying the whole answer
re-rendered to HTML, then one `done` event with the final HTML, the raw
`content`, `truncated`, and `generation_ms`. Errors arrive as an `error`
event. An empty message with no attachments is a `400`, and a model with no
chat template makes the whole endpoint a `501`.

The model manager, on the same port:

| Endpoint | |
| :-- | :-- |
| `GET /api/models` | the models directory as the manager panel draws it: `list`'s own table, which row is loaded, disk use, and any download in flight. Serves a cached scan; `?rescan=true` re-reads the directory |
| `GET /api/models/updates` | which rows are behind their Hugging Face repo — `list`'s `(Refresh)` marker, one Hub request per distinct repo |
| `GET /api/models/metadata?model=…` | a model's full GGUF metadata as plain text — `show`'s own output. `&tensors=true` and `&full=true` are `show --tensors`/`--full` |
| `POST /api/models/select` | restarts the server on a different model, keeping both listening sockets and the pid; answers `202` before it acts, since there is nothing left to answer from afterwards |
| `POST /api/models/download` | starts a Hugging Face download in the background, returning at once |
| `DELETE /api/models` | deletes a model, refusing the one currently loaded |
| `DELETE /api/models/job` | clears a finished download's result |

The scan behind `GET /api/models` opens the GGUF header of every model — and
every shard of every model — under the directory, which is seconds of disk
work on a directory holding a few dozen multi-shard models. So it is cached
and rebuilt explicitly: when the panel opens, after anything that changes the
directory, and whenever `?rescan=true` asks, which is also the answer for a
`.gguf` copied in by hand. The panel's once-a-second poll for download
progress does not rescan, because that progress lives in memory.

`GET /api/models/updates` is its own endpoint rather than part of the listing
precisely because it is a network round trip per repo: the panel opens
instantly on local state and marks rows when this answers — or never, on a
machine with no internet, which is the same "unknown, so not behind" the CLI
table already treats it as.

A download runs **detached, not inside a request**: it takes minutes to hours,
far past any browser's patience, so `POST /api/models/download` takes
`{"repo": "<user>/<model>[:quant]"}`, starts the job and returns. Only one
runs at a time — two concurrent fetches into one models directory would
compete for the same disk and the same free-space check — and a second `POST`
while one is running is a `409` naming what is already downloading, rather
than a silent wait.

The three endpoints that name a model take `{"model": "…"}` — an `NR`, a
`MODEL` label, a bare filename, or a path, exactly as the matching
subcommand's own argument does. The panel sends the `NR`, since that is the
only spelling that names one row exactly (a repo with several quantizations on
disk prints the same bare `MODEL` on each of their rows), and for a load or a
delete it also sends the `path` that row showed. Given both, the server checks
they still agree before acting: an `NR` is a *position*, and a download
finishing while a confirmation dialog is open re-sorts the listing underneath
it. `select` is a `403` when `[web].reexec` is `no`, and `DELETE /api/models`
a `403` when `[web].delete` is `no`.

Finally, a read-only MCP inventory, since configuration belongs to the server
process and edits only take effect on restart:

| Endpoint | |
| :-- | :-- |
| `GET /api/mcps` | every configured MCP section: `name`, `endpoint`, `enabled`, `approval_mode` |
| `GET /api/mcps/{name}` | one of them, or `404` |

### Coordinator

`orangu-coordinator` serves three endpoints of its own and **proxies
everything else** to whichever profile's `orangu-server` its routing selects.
There is no separate API surface to learn: point an OpenAI-compatible client
at the coordinator's address and every endpoint in this chapter works, because
the request is forwarded untouched and the reply streamed back.

#### `GET /v1/coordinator`

A fixed, side-effect-free identity marker, answered directly and never proxied
— it must work even before any profile has been activated. Neither
`orangu-server` nor a generic OpenAI-compatible server exposes this path, so a
client can probe it to tell the three apart:

```json
{
  "orangu_coordinator": true,
  "version": "0.12.0",
  "models": {
    "all": "bartowski/gemma-4-12B-it-GGUF",
    "code": "bartowski/gemma-4-12B-it-GGUF",
    "review": "bartowski/gemma-4-12B-it-GGUF",
    "explorer": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "embeddings": "bartowski/gemma-4-12B-it-GGUF"
  }
}
```

`models` reports the model each conventional role currently resolves to, so a
caller can see what `model` to send for a given role without needing its own
copy of `orangu-coordinator.conf`. A role with no profile of its own falls
back to the `all`-role default's model, exactly as routing does. See **What
`orangu` calls, and when** above for how often a client should ask.

#### `POST /v1/coordinator/activate`

A pre-warming hint a caller can send *before* the request that actually needs
a model:

```sh
curl -s -X POST localhost:9000/v1/coordinator/activate \
  -H 'Content-Type: application/json' -d '{"model": "review"}'
```

`model` is a real model id or a role name, matched exactly the way ordinary
routing matches one. Answered directly, never proxied.

The swap is **spawned detached and not awaited**: the handler resolves the
target profile, starts the swap, and returns `202 Accepted` immediately. That
is the whole point — the hint is sent ahead of the real request, so it has to
survive the caller disconnecting early or never reading the response, and it
must never itself block on a slow cold load. A caller that does want to fail
loudly should send its real request instead, which does wait.

Unlike ordinary routing, an unmatched `model` here is a **`404`**, not a
silent fallback to `all` or to whatever is currently active. Those fallbacks
exist so a request that must be answered *somehow* always is; an explicit
"activate X" call has no such obligation, and silently activating the wrong
thing would be worse than saying so.

#### `GET /v1/coordinator/shutdown?token=…`

Stops the coordinator, and with it whatever `orangu-server` is active or still
starting up. Disabled unless `[orangu-coordinator].shutdown_token` is set — a
`404` naming the missing key rather than a silent refusal. With it set, the
caller must present a matching `?token=` (`403` otherwise) **and** connect from
localhost (`403` otherwise, even with a valid token).

#### Everything else: the proxy

Any other path is forwarded. Three things decide where:

1. The request body's top-level `model` field, when the body is a JSON object
   carrying one.
2. The path, for the one endpoint that names a distinct capability rather than
   a model: a path ending in `/v1/embeddings` implies the `embeddings` role.
   Matched by suffix, so a request arriving through a mount point or a reverse
   proxy prefix still resolves the same way.
3. Otherwise — a bodyless request such as `/health`, `/props`, `/v1/models`,
   `/slots`, `/metrics`, or a file-lifecycle call, none of which name a model
   — the currently active profile, then the `all` profile.

Hop-by-hop headers (`connection`, `keep-alive`, `transfer-encoding`,
`upgrade`, `host`, `content-length`, and the `proxy-*` and `te`/`trailers`
pair) are dropped in both directions; everything else is carried over
unchanged, including `Authorization`. Bodies are capped by
`[orangu-coordinator].max_body_bytes`, 64 MiB by default.

A profile that cannot be started falls back to the `all` profile, and only a
failure of *both* is a `502`. A backend that was alive when the coordinator
checked and gone by the time the request reached it gets the request sent
again, exactly once, after the profile is brought back up — nothing had been
written back to the caller yet, so the request was still whole. A second
failure is a `502` rather than another retry.

### Benchmark console

`orangu-bench --web` serves a console on `127.0.0.1:8300` by default. Like the
server's own console, its `/api/…` surface exists for that page and is not an
API to build against.

| Endpoint | |
| :-- | :-- |
| `GET /` , `GET /static/app.css`, `GET /static/app.js` | the console page and its assets |
| `GET /api/defaults` | what the form starts from, plus whether the profiler and its PNG renderer are actually available on this machine — checked here so the UI can say so beside the checkbox rather than failing twenty minutes into a sweep |
| `GET /api/runs` | every run this console has kept |
| `POST /api/runs` | starts one; the fields are validated into flags, and nothing reaches a shell |
| `DELETE /api/runs` | **Clear all**. A run still measuring is kept and named in the reply rather than killed |
| `GET /api/runs/{id}` | one run: its status and its captured log, stdout and stderr kept apart |
| `DELETE /api/runs/{id}` | deletes one run |
| `POST /api/runs/{id}/cancel` | stops the run in flight |
| `POST /api/runs/{id}/compare` | compares two runs from their archived bundles; allowed while a benchmark is running, since it reads two files and talks to no server |
| `POST /api/runs/{id}/report` | builds the PDF report from the archived bundle, on the one click that wants it |
| `GET /api/runs/{id}/artifacts/{name}` | one of the run's own files, by name from a fixed allowlist |

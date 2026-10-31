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
| `orangu-server` web console | `127.0.0.1:8200` | `[web].host`/`port` | the chat page, its `/api/…` surface, and the model manager |
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

or, equally,

```
x-api-key: <key>
```

and answers `401` with `WWW-Authenticate: Bearer` without one. The `orangu`
client sends its own `api_key` as a bearer token on every request, including
the `/v1/models` probe, and so does any OpenAI-shaped client; Anthropic SDKs
send `x-api-key`. A refused `/v1/messages*`, `/v1/files*`, `/v1/skills*`,
`/v1/agents*`, `/v1/environments*`, `/v1/sessions*`, `/v1/memory_stores*` or
`/v1/vaults*` request gets the
`401` in the Anthropic error shape (see **Anthropic-compatible endpoints**).

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
**`501`** naming the mode, rather than pretending to generate. A speech
(`qwen3tts`) server answers `/v1/completions`, `/completion`,
`/v1/embeddings` and `/v1/messages` with `501` the same way.

### Endpoints at a glance

| Endpoint | |
| :-- | :-- |
| `GET /v1/models` | the one model this process serves, readable by OpenAI and Anthropic clients |
| `GET /v1/models/{id}` | that model, when `id` names it; `404` otherwise |
| `POST /v1/chat/completions` | streaming (SSE) and non-streaming; `tools`/`tool_calls` and `response_format` (`json_object`, `json_schema`); `cache_prompt`/`id_slot`/`timings_per_token`/`return_progress`; needs a chat template; disabled under `--embedding` |
| `POST /v1/messages` | Anthropic's Messages API: streaming (SSE) and non-streaming; content blocks and PDF documents, `tools`/`tool_choice`/`tool_use`, continued replies, `stop_sequences`, `thinking`/`effort`; needs a chat template |
| `POST /v1/messages/count_tokens` | the prompt token count a `/v1/messages` request would have |
| `POST /v1/messages/batches`, `GET /v1/messages/batches` | Anthropic's Message Batches API: submit many `/v1/messages` requests at once, or list the batches |
| `GET /v1/messages/batches/{id}`, `DELETE /v1/messages/batches/{id}` | one batch's status and counts; delete an ended one |
| `POST /v1/messages/batches/{id}/cancel` | stop a batch; requests not yet started end `canceled` |
| `GET /v1/messages/batches/{id}/results` | an ended batch's results, one JSON line per request |
| `POST /v1/files`, `GET /v1/files` | Anthropic's Files API: upload a file (multipart, up to 500 MiB), or list them |
| `GET /v1/files/{id}`, `DELETE /v1/files/{id}` | one file's metadata; delete it |
| `GET /v1/files/{id}/content` | the file's bytes |
| `/v1/memory_stores…` | memory stores, their memories and versions (see **Memory stores**) |
| `/v1/vaults…` | vaults and their credentials (see **Vaults**) |
| `POST /v1/skills`, `GET /v1/skills` | Anthropic's Skills API: upload a skill (multipart, a folder with a `SKILL.md`), or list them |
| `GET /v1/skills/{id}`, `DELETE /v1/skills/{id}` | one skill; delete one with no versions left |
| `POST /v1/skills/{id}/versions`, `GET /v1/skills/{id}/versions` | upload a new version of a skill, or list its versions |
| `GET /v1/skills/{id}/versions/{version}`, `DELETE /v1/skills/{id}/versions/{version}` | one version; delete it |
| `POST /v1/agents`, `GET /v1/agents` | Managed Agents: create an agent, or list them |
| `GET /v1/agents/{id}`, `POST /v1/agents/{id}`, `POST /v1/agents/{id}/archive`, `GET /v1/agents/{id}/versions` | one agent (at a `version`); update it, archive it, list its versions |
| `POST /v1/environments`, `GET /v1/environments` | create an environment, or list them |
| `GET`, `POST`, `DELETE /v1/environments/{id}`, `POST /v1/environments/{id}/archive` | one environment; update, delete or archive it |
| `POST /v1/sessions`, `GET /v1/sessions` | start a session of an agent in an environment, or list them |
| `GET`, `POST`, `DELETE /v1/sessions/{id}`, `POST /v1/sessions/{id}/archive` | one session; update, delete or archive it |
| `POST /v1/sessions/{id}/events`, `GET /v1/sessions/{id}/events` | send a session events (a message, a tool result, an interrupt), or list its events |
| `GET /v1/sessions/{id}/events/stream` | a session's events as they happen, as server-sent events |
| `POST /v1/completions` | legacy completion, no chat template needed; `cache_prompt`/`id_slot`/`response_format`/`ignore_eos`; disabled under `--embedding` |
| `POST /v1/embeddings` | pooled and L2-normalized; carries OpenAI's `usage` |
| `POST /v1/images/generations` | OpenAI's Images API, on a `qwen_image` or `qwen_image_2_1` model; `501` on a language model |
| `POST /v1/videos`, `GET /v1/videos` | OpenAI's Videos API, on an `ltxv` model: start a clip as a job, or list the jobs; `501` on any other model |
| `GET /v1/videos/{id}`, `DELETE /v1/videos/{id}` | one job's status and progress; cancel or drop it |
| `GET /v1/videos/{id}/content` | the finished clip as `video/mp4`, with `Range` |
| `POST /v1/audio/speech` | OpenAI's speech API, on a `qwen3tts` model: the text spoken, as WAV, FLAC or PCM; `501` on any other model |
| `GET /v1/audio/voices` | the voices, languages and formats a `qwen3tts` model offers |
| `GET /health` | liveness: is this process up. Stays `200` while the server is merely busy |
| `GET /ready` | readiness: would a request sent now be served |
| `GET /v1/workers` | this node's place in its `[workers]` tree: role, layers, plan, workers |
| `GET /version` | which build this is: version, commit, compiler, profile, target |
| `GET /props` | model and server metadata: backend, devices, build, slot count, workspace |
| `GET /slots` | per-slot busy/prompt/generated-token state |
| `POST /slots/{id_slot}` | `?action=save\|restore` — persist or reload that slot's KV cache |
| `GET /metrics` | Prometheus text: gauges, latency histograms, outcome and token counters |
| `POST /completion` | native, streaming; `cache_prompt`/`id_slot`; disabled under `--embedding` |
| `POST /embedding` | native embeddings |
| `POST /tokenize`, `POST /detokenize` | text to token ids and back |
| `POST /logits` | teacher-forced next-token log-probabilities after every prefix of a text |
| `POST /apply-template` | renders the chat template without generating |
| `POST /v1/create_file` … `/v1/delete_directory` | the eight file-lifecycle endpoints |
| `GET /moe-stats` | mixture-of-experts counters since the previous call, and reset |
| `GET /gpu-timings` | per-stage GPU timings since the previous call, and reset |
| `GET /model-cache` | how much of the model is in the page cache right now |
| `POST /model-cache/drop` | evict the model from the page cache; loopback-only |
| `POST /v1/shutdown` | stop the server; loopback-only |

### OpenAI-compatible endpoints

#### `GET /v1/models`

The model this process serves. One entry, always — `orangu-server` loads
exactly one model, and switching to another restarts the process (see
**Loading a different model** in the *Inference server* chapter).

The entry carries both OpenAI's fields (`object`, `created`, `owned_by`) and
Anthropic's (`type`, `display_name`, `created_at`), and the listing both
`object: "list"` and Anthropic's `has_more`/`first_id`/`last_id`, so either
SDK's model listing reads it.

```json
{"object":"list","data":[{"id":"ggml-org/gemma-4-E4B-it-GGUF","object":"model",
 "created":1755600000,"owned_by":"orangu-server","type":"model",
 "display_name":"ggml-org/gemma-4-E4B-it-GGUF","created_at":"2025-08-19T10:40:00Z"}],
 "has_more":false,"first_id":"ggml-org/gemma-4-E4B-it-GGUF","last_id":"ggml-org/gemma-4-E4B-it-GGUF"}
```

Each entry also carries Anthropic's `max_input_tokens` and `max_tokens` — the
model's context, or the device's room when that is smaller — and
`capabilities`: `batch`, `citations`, `pdf_input` and `structured_outputs`
as supported, `thinking` as this model has it (whether the vocabulary marks
reasoning off from the answer), and `image_input` as supported when the
model has a vision tower (see **Pictures**).

`GET /v1/models/{id}` returns the same entry when `id` is the served model's
id, and `404` otherwise.

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
`<think>`…`</think>` span, gemma-4's `thought` channel, a message addressed
`to=self`, a body opened with `<|content_thinking|>`. Thinking never passes through the tool-call splitter
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
`result["name"]`, which is worse than failing outright.

With `{"type": "json_schema", "json_schema": {"schema": {…}}}` the output is
held to the **schema**, not just to JSON: a required property cannot be left
out, an undeclared one cannot be added, an `integer` cannot be written as
`36.5`, and an `enum` value cannot be anything else. What a schema may use:

- `type` — one, or a list such as `["string", "null"]` — with `properties`,
  `required` and `additionalProperties` for an object, and `items`,
  `minItems` and `maxItems` for an array.
- `enum`, `const`, `anyOf`, `oneOf`, a one-schema `allOf`, and `$ref` into
  the same schema (`#/$defs/…`, `#/definitions/…`, `#`), recursion included.
- `true`, `{}` or a schema with no `type`: any JSON value.

An object with `properties` is written with its properties **in declared
order**, each optional one either present or skipped, and no others. An
object without `properties` takes any keys, valued as
`additionalProperties` says (`false` allows only `{}`).

Annotations and value checks — `description`, `title`, `format`, `pattern`,
`minLength`, `minimum`, `uniqueItems` and the like — are accepted and not
enforced. Keywords that change the shape in ways the constraint cannot
follow — `not`, `if`/`then`/`else`, `patternProperties`, a many-schema
`allOf`, `prefixItems` and the like — make the request a `400` naming the
keyword, as do a `$ref` outside the schema and a `required` name that
`properties` lacks. The schema is not added to the prompt here: describe
what you want in the messages too, or the model writes the right keys while
pausing in whitespace before each one.

Two limits worth knowing. A constrained request cannot use the GPU's argmax
fast path, which picks tokens without consulting the constraint, so each step
reads its logits back to the host. Speculation still runs: each drafted token
is checked against what the constrained sampler picks there. And a constraint makes invalid output unreachable; it cannot
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

#### `POST /v1/images/generations`

Only on a server whose model is a `qwen_image` or `qwen_image_2_1`
diffusion transformer (see
the *Image generation* section of the server chapter); a language model
answers `501`.

```sh
curl -s localhost:8100/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a lighthouse at dusk, oil painting", "size": "768x512", "steps": 20}' \
  | jq -r '.data[0].b64_json' | base64 -d > lighthouse.png
```

| field | default | |
| :-- | :-- | :-- |
| `prompt` | *required* | what to draw |
| `negative_prompt` | `image_negative_prompt` | what to draw away from; read only when `cfg_scale > 1` |
| `size` | `image_size` | `WIDTHxHEIGHT`, both multiples of 16; `auto` is the default, or the attached `image`'s own proportions when there is one |
| `n` | `1` | pictures to make, `1` to `10`, in turn; a `seed` seeds the set, each one `seed + i` |
| `output_format` | `image_format` (`png`) | `png`, `jpeg`, `gif`, `webp` (lossless) or `svg` (an SVG document carrying the picture as PNG); an attached picture defaults to its own format |
| `response_format` | `b64_json` | the only value — there are no files to return a `url` to |
| `seed` | drawn | reproduces a picture on the same build; the reply reports the one used |
| `steps` | `image_steps` | denoising steps |
| `cfg_scale` | `image_cfg_scale` | classifier-free guidance; `1` runs the prompt alone, half the work |
| `image` | — | a picture to start from — a `data:image/…;base64,…` URL or bare base64 of a PNG, JPEG, GIF, WebP or SVG. Resized to `size`. On a `qwen_image_2_1` server with a vision projector (`/props`' `image.vision`), the picture is **edited** by the prompt instead: read by the text encoder and kept wherever the prompt does not ask for a change |
| `strength` | `image_strength` | with `image`: how much of the schedule to run, `0` (unchanged) to `1` (the picture's content is ignored). Not read for an edit |
| `stream` | `false` | server-sent events instead of one body |

The reply is OpenAI's: `{"created", "model", "data": [{"b64_json", …}]}`,
with each picture also carrying `output_format`, `size`, `seed`, `steps`,
`generation_ms` and `timings` — `{"encode_ms", "steps_ms", "decode_ms"}`,
where the picture's time went: the prompts through the text encoder, the
denoising loop, and the VAE plus the file format — and `timings.stages`,
the loop's transformer passes by operation class in milliseconds
(`passes`, `modulation_ms`, `qkv_ms`, `attention_ms`, `out_ms`, `mlp_ms`,
`other_ms`). It is what `orangu-bench --image` reads; the server logs the
same split per picture. Streamed, the
events are `image_generation.progress` —
`step`, `steps`, `seconds_per_step`, `eta_seconds` — once per step, then one
`image_generation.completed` with the same fields the body form has, then
`[DONE]`. The first progress event has `step` **0**: it is the server's
announcement before any work, its `seconds_per_step` and `eta_seconds`
the estimate from a rate model — the linears' cost per token and
attention's, which grows with the token count, learned from each finished
picture's own stage timings and seeded at startup by a calibration
matmul; within about 20% across a fourfold change of size, closer at the
size last measured. Its `eta_seconds` also counts the encode ahead of
the first step — for an edit, the reference through the vision tower, the
VAE and the prefix, and every step attending to the reference's tokens as
well as the picture's. A client can show the wait the moment it
sends; the real steps follow from `1`. Closing the stream cancels the
work at the next step. `GET /props` carries the same figures under `image`
as `token_passes_per_second` (at the defaults' size) and
`estimated_default_seconds` (the whole picture: encode, steps, decode),
and the rate model itself — see **`GET /props`** below.

`POST /v1/chat/completions` on the same server draws from the last user
message: its text is the prompt, and the last `image_url` part of an
array-shaped `content` (`{"type": "image_url", "image_url": {"url":
"data:image/png;base64,…"}}`) is the picture to start from. Every knob is
the server's `image_*` default except `seed`. The assistant's `content` is
the picture as a markdown image over a `data:` URL, which a markdown-rendering
chat client shows inline; `message.images` carries the same picture as
`b64_json` with `mime`, `size`, `seed`, `steps` and `generation_ms` for a
client that wants the bytes. Streamed, the chunks carry an empty `delta` and
an `image_progress` object (`step`, `steps`, `eta_seconds`) until the final
chunk delivers the content with `finish_reason: "stop"`.

#### `POST /v1/videos`

Only on a server whose model is an `ltxv` video model (see the *Video
generation* section of the server chapter); any other model answers
`501`. A clip is minutes of work, so this is a job: the request answers
at once with the job, and the clip is fetched when it is done.

```sh
curl -s localhost:8100/v1/videos -H 'Content-Type: application/json' \
  -d '{"prompt": "A paper boat drifts down a rain-filled gutter", "seconds": 3, "size": "512x320"}'
```

| field | default | |
| :-- | :-- | :-- |
| `prompt` | *required* | what to film, the sound included |
| `seconds` | `video_seconds` | the length, a number or a string (`"4"`) as OpenAI sends it; more than 0 and at most 60. Snapped to 8k + 1 frames at `fps` |
| `frames` | `video_frames` | the length in frames instead, snapped the same way. Give `seconds` or `frames`, not both |
| `size` | `video_size` | `WIDTHxHEIGHT`, both multiples of 32, and of 64 when the clip is drawn in two stages; `auto` is the default |
| `fps` | `video_fps` | frames per second, 1 to 60 |
| `seed` | drawn | reproduces a clip on the same build; the job reports the one used |
| `audio` | `video_audio` | whether the sound goes into the file; without the audio VAE a clip is silent either way |
| `two_stage` | `video_two_stage` | the distilled two-stage recipe; without the upscaler a clip is drawn in one stage either way |
| `quality` | `video_quality` | the AV1 quantizer, `0` (best) to `255` (smallest file) |
| `stream` | `false` | server-sent events until the clip is done, instead of answering at once |

A bad field — an empty prompt, a size that is not a multiple of the unit,
both `seconds` and `frames`, a value out of range — is a `400` with the
reason as plain text, before any work.

The answer is the job, a `video` object:

```json
{"id":"video_3f1c9a0d5e7b2468","object":"video","model":"unsloth/LTX-2.5-GGUF:Q4_K_M",
 "status":"queued","progress":0,"phase":null,"eta_seconds":null,
 "created_at":1760000000,"completed_at":null,
 "prompt":"A paper boat drifts down a rain-filled gutter","size":"512x320",
 "seconds":"3.04","frames":73,"fps":24,"seed":null,"audio":null,
 "bytes":null,"generation_ms":null,"timings":null,"error":null}
```

| field | |
| :-- | :-- |
| `id` | `video_` and sixteen hex digits |
| `status` | `queued` until the work starts (one clip runs at a time, so a second job waits), `in_progress`, then `completed` or `failed` |
| `progress` | `0` to `100`: the transformer steps done across both stages; `100` only when `completed` |
| `phase` | what is running: `encoding` (the prompt), `stage 1`, `upscaling`, `stage 2`, `decoding`, `sound`, `encoding the file` |
| `eta_seconds` | this server's estimate of the seconds still ahead, from its measured rate |
| `created_at`, `completed_at` | Unix seconds; `completed_at` is set on `completed` and on `failed` |
| `prompt`, `size`, `fps` | as asked, with the defaults filled in |
| `seconds` | the length as a string, OpenAI's way, from the frames actually drawn |
| `frames` | the frames actually drawn, 8k + 1 |
| `seed` | the one used, once the clip is done; the one asked for before that |
| `audio` | once done: whether the file has a sound track |
| `bytes` | once done: the MP4's size |
| `generation_ms` | once done: the whole run |
| `timings` | once done: where the time went — `encode_ms`, `stage_1_ms`, `upscale_ms`, `stage_2_ms`, `decode_ms`, `audio_ms`, `mux_ms` |
| `error` | on `failed`: `{"message": …}` |

Jobs live in memory: the last sixteen are kept (the oldest finished one
is dropped first), and a restart forgets them all.

**Streamed** (`stream: true`), the connection stays open and each event
is a JSON object with a `type`:

- `video.created` — `{"type", "video"}`, the job as above, first.
- `video.progress` — `{"type", "id", "step", "steps", "phase",
  "eta_seconds"}`, at every phase and every step. The first has `step`
  **0**: the announcement before any work, its `eta_seconds` the estimate
  of the whole clip.
- `video.completed` — `{"type", "video"}`, the finished job, then
  `[DONE]`.
- `video.failed` — `{"type", "id", "error": {"message"}}`.

Leaving the stream does not stop the job: it stays in the table, and
`DELETE /v1/videos/{id}` is how a client cancels it.

#### `GET /v1/videos` and `GET /v1/videos/{id}`

`GET /v1/videos` lists the jobs kept, newest first, as
`{"object": "list", "data": [ … ]}`. `GET /v1/videos/{id}` is one job, the
object above; poll it for `progress`, `phase` and `eta_seconds`. An
unknown id is a `404` with `{"error": {"message": "no video '…'"}}`.

#### `GET /v1/videos/{id}/content`

The finished clip: an MP4 of AV1 video and a FLAC sound track, as
`video/mp4`. Its index comes before the media, so a player starts before
the whole file has arrived, and a single `Range: bytes=a-b` (or `a-`, or
`-n`) is answered with `206` and that part, so a `<video>` element can
seek. A job that is not `completed` answers `409` with
`{"error": {"message": "video '…' is in_progress"}}`; an unknown id `404`.

```sh
curl -s -o boat.mp4 localhost:8100/v1/videos/video_3f1c9a0d5e7b2468/content
```

#### `DELETE /v1/videos/{id}`

Cancels a running job, which stops at its next transformer block, or
drops a finished one, and answers
`{"id", "object": "video.deleted", "deleted": true}`. An unknown id is a
`404`.

`POST /v1/chat/completions` on a video server films the last user
message: its text is the prompt, and every knob is the server's `video_*`
default except `seed`. It runs as a `/v1/videos` job, so the clip is also
at `/v1/videos/{id}`. The assistant's `content` is a markdown link to the
clip, `[prompt](/v1/videos/{id}/content)`, which a chat client can
follow, and `message.videos` carries the job object. Without `stream`
the reply comes when the clip is done. Streamed, the chunks carry an
empty `delta` and a `video_progress` object (`id`, `step`, `steps`,
`phase`, `eta_seconds`) until the final chunk delivers the content with
`finish_reason: "stop"`. A failed clip is a `500` with the reason, or an
`{"error"}` event when streamed.

#### `POST /v1/audio/speech`

OpenAI's speech API. Only on a server whose model is a `qwen3tts` speech
model (see the *Speech generation* section of the server chapter); any
other model answers `501`. The answer is the clip itself, sent whole
once it is made:

```sh
curl -s localhost:8100/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello from orangu.", "voice": "1.7b-designed"}' -o hello.wav
```

| field | default | |
| :-- | :-- | :-- |
| `input` | _required_ | the text to speak |
| `model` | — | ignored: the server speaks with the model it serves |
| `voice` | `audio_voice` | a voice's name, or `{"id": name}`; `GET /v1/audio/voices` lists them |
| `response_format` | `audio_format` | `wav`, `flac` or `pcm`. `mp3`, `opus` and `aac` are refused with `400` naming the supported ones |
| `speed` | `1.0` | only `1.0`: any other value is a `400` |
| `stream_format` | `audio` | only `audio`; `sse` is a `400`, the clip being sent whole |
| `instructions` | — | refused with `400`: a Base model cannot be steered by a description |
| `language` | `audio_language` | `auto`, or a language the model names (`english`, `german`, …, or their ISO codes) |
| `temperature` | `audio_temperature` | codebook 0's temperature, `0` (greedy) to `2` |
| `top_k` | `audio_top_k` | codebook 0's top-k, `1` or more |
| `repetition_penalty` | `audio_repetition_penalty` | codebook 0's repetition penalty, `1` to `2` |
| `max_seconds` | `audio_max_seconds` | the longest clip, above `0` and at most `600` |
| `seed` | random | the sampler's seed, for a clip that can be made again |

The first seven are OpenAI's, the rest this server's. A bad field — an
empty `input`, a value out of range, an unknown voice or language — is a
`400` with the reason as plain text, before any work; an unknown voice or
language names the valid ones.

The body is the audio, with a `Content-Type` of `audio/wav`,
`audio/flac` or `audio/pcm`. `pcm` is bare 16-bit little-endian mono
samples at 24 kHz, as OpenAI's is. Five headers describe the clip:

| header | |
| :-- | :-- |
| `x-orangu-voice` | the voice that spoke |
| `x-orangu-seed` | the seed it was made with |
| `x-orangu-frames` | the codec frames spoken, 12.5 a second |
| `x-orangu-sample-rate` | `24000` |
| `x-orangu-finished` | `true` when the model ended the clip, `false` when the length limit cut it |

#### `GET /v1/audio/voices`

The voices a speech server offers, the languages and the formats:

```json
{"object":"list",
 "data":[{"id":"1.7b-designed","object":"voice","description":"…"}],
 "default":"1.7b-designed",
 "languages":["auto","chinese","english","french","german","italian",
              "japanese","korean","portuguese","russian","spanish"],
 "formats":["wav","flac","pcm"]}
```

`default` is the voice a request that names none gets. `501` on any
other model.

`POST /v1/chat/completions` on a speech server speaks the last user
message and answers in OpenAI's audio-output shape: `choices[0].message`
is `{"role": "assistant", "content": null, "audio": {…}}`, and `audio`
holds `id`, `data` (the clip, base64), `format`, `transcript` (the text
spoken), `expires_at`, `voice` and `seconds`. The request's `audio`
object (`{"voice", "format"}`), `temperature`, `top_k` and `seed` are
honoured; every other knob is the server's `audio_*` default. With
`stream: true` the answer is one chunk carrying the same message as its
`delta`, then `[DONE]`.

`/v1/completions`, `/completion` and `/v1/embeddings` answer `501` on a
speech model: there is no text to generate or embed, and the message says
to use `/v1/audio/speech`, or `/v1/chat/completions` for a spoken reply.
`/v1/messages` answers `501` too, naming the `--audio` mode.

#### Citations

A `document` or `search_result` block with `citations: {"enabled": true}`
is shown to the model in numbered passages, and the model is asked to write
the numbers it draws on after its claims. The server takes the numbers out
of the reply again: the text before each one becomes a text block whose
`citations` hold the passages it named, located as Anthropic locates them:

| Source | Passage | Citation |
|---|---|---|
| text document | a sentence | `char_location` (`start_char_index`, `end_char_index`) |
| PDF document | a sentence of a page | `page_location` (`start_page_number`, `end_page_number`) |
| `content` document | a content block | `content_block_location` (`start_block_index`, `end_block_index`) |
| search result | a content block | `search_result_location` (`search_result_index`, `source`, `title`) |

Each citation carries `cited_text`, and a document's carries its
`document_index` (counting every `document` block of the request, in order)
and `document_title`. Streamed, a citation is a `citations_delta` on the
text block it belongs to. A number that names no passage stays in the text
as written. How well the model cites depends on the model: the passages
are offered, not enforced. Citations cannot be combined with an output
format (`400`).

#### Pictures

A gemma-4 model reads pictures when its vision tower — the
`mmproj-*.gguf` `orangu-server download` fetches beside it — is loaded:
`[orangu-server].vision` names one, or the server finds the one beside the
model whose projection is the model's width (see the server chapter). The
server says which at startup (`vision … (up to N tokens a picture)`).

A picture reaches the model from:

- `/v1/chat/completions` — an `image_url` part, a `data:` URL or an `http(s)`
  URL;
- `/v1/messages` — an `image` block, a `base64`, `url` or `file` source, in
  a turn or in a `tool_result`;
- a Managed Agents session's `user.message` `image` blocks, and its `read`
  tool on a picture file;
- the web console's attached pictures.

PNG, JPEG, GIF, WebP and SVG are read; an `http(s)` URL is fetched by the
server, which takes an `image/*` reply of up to 50 MB. Like every URL the
server fetches for a request, it must reach a public address (see
`fetch_private` in the server chapter). A picture is read at
its own proportions, each side a multiple of 48 pixels, in 70 to
`[orangu-server].vision_tokens` (default 280) prompt tokens — a larger
picture at a smaller size. Its tokens count in `usage` and in
`count_tokens` like text, and a picture seen again (the next turn of a
conversation) comes from the prompt cache: it is read through the tower
once while it stays in the server's store of recent pictures.

A picture sent to a model without a vision tower is a `400`, as is one
that cannot be decoded. The model's entry in `/v1/models` reports
`image_input` accordingly.

### Anthropic-compatible endpoints

An Anthropic SDK works against `orangu-server` by pointing its base URL at the
server (`base_url="http://127.0.0.1:8100"`) and setting its API key to the
server's `api_key`, or to anything when none is set. `anthropic-version` and
`anthropic-beta` are accepted and not read.

A `/v1/messages` conversation is rendered through the same chat template as
`/v1/chat/completions`, so the model sees the same prompt from either
endpoint. Errors use Anthropic's envelope —
`{"type":"error","error":{"type":"invalid_request_error","message":"…"}}` —
with `authentication_error` (`401`), `not_found_error` (`404`),
`overloaded_error` (`503`, with `Retry-After`) and `api_error` (`500`, `501`)
for the other statuses. Every response carries a `request-id` header, the
message's `id`.

#### `POST /v1/messages`

| Field | Default | |
| :-- | :-- | :-- |
| `model` | the served model | echoed back in the response; any value is served by the one model this process has |
| `messages` | `[]` | `user`/`assistant` turns; `content` a string or a list of blocks |
| `system` | none | a string, or a list of `text` blocks |
| `max_tokens` | `8192` | cap on the answer |
| `stream` | `false` | server-sent events |
| `stop_sequences` | `[]` | strings that end the answer; the matched one is not included |
| `temperature`, `top_p`, `top_k` | the role's | sampler |
| `min_p`, `repeat_penalty`, `seed` | the role's | sampler fields under `/v1/chat/completions`'s names |
| `tools` | none | custom tools (`name`, `description`, `input_schema`), Anthropic-defined client tools, and the server tools `web_search_*`, `web_fetch_*`, `code_execution_*`, `advisor_*` and `mcp_toolset` (see below) |
| `container` | a new one | for code execution: a container's id, or `{"id": …, "skills": [{"type": "custom", "skill_id": …, "version": …}]}` (see **Code execution** below) |
| `mcp_servers` | none | remote MCP servers whose tools the server calls itself: `[{"type": "url", "url": …, "name": …, "authorization_token": …}]` |
| `tool_choice` | `auto` | `auto`, `any` (the reply is a tool call), `tool` with `name` (a call of that tool), `none` (the tools are withheld); `disable_parallel_tool_use: true` ends the reply at its first call |
| `thinking` | the role's | `{"type": "disabled"}` turns reasoning off; `enabled` and `adaptive` turn it on; with `enabled`, `budget_tokens` caps each turn's reasoning — once the model has reasoned that many tokens, the token that ends its reasoning is the next one it writes (for a model that marks reasoning with `<think>` or a channel) — and picks an effort when `output_config` names none (under 4096 `low`, under 16384 `medium`, else `high`) |
| `output_config` | none | `effort`: `"low"`, `"medium"`, `"high"` or `"max"`, passed to the template as its reasoning effort (`max` as `high`); `format`: `{"type": "json_schema", "schema": {…}}` holds the reply to the schema (see **Structured output** below) |
| `output_format` | none | `output_config.format` under its earlier name |
| `cache_control` | — | accepted on any block and at the top level; the prefix cache is automatic, so it changes nothing |
| `cache_prompt`, `id_slot` | `true`, none | as on `/v1/chat/completions` |

A request body may be up to 32 MiB; a larger one is a `413`
`request_too_large`.

Content blocks read from `messages`:

- `text` — the turn's text; several are joined with newlines.
- `image` — a `base64`, `url` or `file` source, read by the model where it
  stands among the turn's text (see **Pictures**). An `image` inside a
  `tool_result`'s content follows the result's text.
- `document` — the text of a `text` source, a `content` source's `text`
  blocks, or a base64 PDF's embedded text (scanned pages carry none), under
  the document's `title` when it has one. A `url` source is fetched by the
  server (a PDF or text, up to 32 MB) and read the same way. With
  `citations: {"enabled": true}` its passages are numbered (see
  **Citations**).
- `search_result` — its `title`, `source` and text `content`, at the top
  level of a turn or inside a `tool_result`.
- `tool_use` (assistant) — becomes one of the turn's tool calls.
- `tool_result` (user) — becomes the result of the call its `tool_use_id`
  names; `is_error: true` prefixes its content with `Error:`.
- `container_upload` (user) — copies a stored file into the code execution
  container (see **Code execution** below); the model is told where it is.
- `thinking` and `redacted_thinking` — skipped: earlier turns are rendered
  without their reasoning.

Anthropic-defined client tools — `bash_*`, `text_editor_*`, `memory_*` and
`computer_*` — are declared to the model with their standard input schemas
under the `name` the request gives them; the client runs them, as with any
custom tool. Anthropic's server tools other than those below (tool search
and the rest) are left out.

A call the model makes to a tool the request does not have — a client
tool it never declared, or a name it made up — is not handed to the client:
the model is told there is no such tool, and which tools there are, and the
reply goes on.

**Server tools.** `web_search_*`, `web_fetch_*`, `code_execution_*`,
`advisor_*` and the tools of the `mcp_servers` are run by the server itself. A reply that calls
one gets a `server_tool_use` block (`mcp_tool_use` for an MCP tool, naming
its `server_name`), then the result — `web_search_tool_result`,
`web_fetch_tool_result`, `bash_code_execution_tool_result`,
`text_editor_code_execution_tool_result`, `code_execution_tool_result`,
`advisor_tool_result` or `mcp_tool_result` — and then the model goes on
from the result, all within the one reply. A reply that also calls a
client tool stops there with `tool_use`. After ten rounds of server tools a
reply stops with `pause_turn`; sending it back as the last assistant turn
continues it. Server tool blocks sent back in a conversation are read as the
calls and results they record. `usage.server_tool_use` counts
`web_search_requests` and `web_fetch_requests`.

- `web_search` searches the SearXNG instance `[orangu-server].web_search`
  names, and returns up to eight results — `url`, `title`, and the excerpt
  the model read as `encrypted_content` (base64 text, not encrypted). With no
  instance configured, a request declaring `web_search` is a `400`.
  `max_uses`, `allowed_domains` and `blocked_domains` apply; a domain covers
  its subdomains.
- `web_fetch` fetches a page or PDF and returns its text as a document. It
  fetches only URLs that have appeared in the conversation — in a message, a
  search result, or a page fetched before — so the model cannot be steered to
  addresses it was never shown, and only public addresses are reached
  (see `fetch_private` in the server chapter). `max_uses`, `allowed_domains`,
  `blocked_domains` and `max_content_tokens` apply. Errors come back as
  `web_fetch_tool_result_error` with Anthropic's `error_code`s (`url_not_allowed`,
  `url_not_accessible`, `unsupported_content_type`, `max_uses_exceeded`, …).
- MCP servers are connected for each request over streamable HTTP, with
  `authorization_token` as a bearer token. Without `mcp_toolset` entries,
  every server's tools are available, narrowed by its
  `tool_configuration` (`enabled`, `allowed_tools`). With them, only the
  servers a toolset names are used, and its `default_config` and per-tool
  `configs` decide which tools are enabled. When two servers offer the same
  tool name, the first keeps it.
- `code_execution` runs code in a sandboxed container — see **Code
  execution** below.
- `advisor` (`{"type": "advisor_20260301", "name": "advisor", "model": …}`)
  is a second request to the served model, which reads the conversation so
  far — its start and its most recent part when it is long — and answers
  with guidance for the model to go on from. The call takes no input. Its
  result's `content` is `advisor_result` (`text`, and `stop_reason`:
  `end_turn`, or `max_tokens` when the advice was cut off, which the model is
  told), or `advisor_tool_result_error` with an `error_code`
  (`max_uses_exceeded`, `prompt_too_long`, `overloaded`, `unavailable`).
  `model` is required and recorded; `max_uses` and `max_tokens` (2048 when
  not given) apply. A conversation holding `advisor_tool_result` blocks
  needs the advisor tool in `tools`, or it is a `400`.

**Code execution.** With `[orangu-server].code_execution = bwrap`, a
`code_execution_*` tool gives the model a Linux container to run code in;
with it `off` (the default), a request declaring one is a `400`.
`code_execution_20250522` is one `code_execution` tool taking Python `code`,
answered with `code_execution_tool_result`. Every later version is two
tools: `bash_code_execution`, taking a shell `command` and answered with
`bash_code_execution_tool_result` (`stdout`, `stderr`, `return_code`), and
`text_editor_code_execution`, taking `view`, `create` or `str_replace` on a
`path` and answered with `text_editor_code_execution_tool_result`
(`text_editor_code_execution_view_result`, `…_create_result` or
`…_str_replace_result`, or `…_tool_result_error` with an `error_code` such
as `file_not_found`). A command that fails to start comes back as an
`unavailable` error, and one running past two minutes is stopped and comes
back as `execution_time_exceeded`. Tools may be called from the model's
code (see **Tools called from code**). Python state does not carry from one
code execution call to the next: files do.

A container is a directory under `[orangu-server].api_data`; its
`/workspace` keeps its files from one command, and one request, to the
next. The response's `container` gives its `id` and `expires_at` — 30 days
after its last use — and a request that passes that id as `container`
works in it again; an expired or unknown one is a `404`. Each command runs
in a fresh bubblewrap sandbox as the server's user, in its own user, PID,
network, IPC and UTS namespaces: no network, `/usr` and the few `/etc` files
the dynamic loader and user lookup need read-only, a private `/tmp`, and
nothing else of the host — not the home directory. A command may use up to
5 GiB of address space and write files of up to 1 GiB; its stdout and
stderr are each cut at 100,000 bytes. The text editor runs in the server on
the same paths, and refuses one that leads outside `/workspace`, links
included.

The files a command writes in `/workspace` — new or changed, up to 20 of
100 MiB or less, hidden ones and `__pycache__` left out — are stored in the
Files API, and each one's `file_id` is listed in the result's `content` as
a `bash_code_execution_output` (or `code_execution_output`) block, for
`GET /v1/files/{id}/content` to download. A `container_upload` block,
`{"type": "container_upload", "file_id": "file_…"}`, copies a stored file
into `/workspace` under its file name (`name-1.ext` and so on when the name
is taken) once per container.

`container.skills` lists up to eight skills from the Skills API (see
**Skills** below), each at its `version` (`latest` when it names none).
Each one's folder is mounted read-only at `/skills/<name>`, and the system
prompt lists each skill's name and description, telling the model to read
its `SKILL.md` before using it. The response's `container.skills` gives the
version each resolved to. Skills, and `container_upload` blocks, need the
`code_execution` tool; a `type: "anthropic"` skill (Anthropic's own `pptx`,
`xlsx` and the rest) is a `400`.

**Structured output.** With `output_config.format` (or `output_format`) the
reply is one `text` block holding a JSON document the schema describes —
held there the way `response_format: json_schema` holds
`/v1/chat/completions` (see **Structured output (`response_format`)**, for
what a schema may use). The schema is also added to the system prompt, so
the model knows what each field is for. It cannot be combined with `tools`
or a continued assistant turn; either is a `400`. A schema the constraint
cannot follow is a `400` naming why.

**Tools called from code.** A client tool whose `allowed_callers` lists
`code_execution_20260120` (or `code_execution_20260521`), in a request that
declares one of those code execution tools, can be called by the model's
Python code: the model gets a `code_execution` tool for Python beside the
bash and editor ones, and in its code each such tool is an async function
taking one dict of arguments and returning the result's text — `rows =
json.loads(await query_sales({"region": "north"}))`, several at once with
`asyncio.gather`. A tool without `"direct"` in its `allowed_callers` is
offered to the model only that way.

When the code calls tools, the reply stops with `stop_reason: "tool_use"`:
the `server_tool_use` block of the code, then a `tool_use` block for each
call, whose `caller` is `{"type": "code_execution_20260120", "tool_id":
<the server_tool_use id>}` (calls made together arrive together). The code
waits, paused. The next request — the conversation with that reply, then a
user message of only `tool_result` blocks, each a string or text blocks —
hands it the results, and the reply is either the code's next calls alone
or, once it ends, its `code_execution_tool_result` followed by the model
going on from what it printed. The model never sees the calls or their
results, only the code and its output. A paused run waits about four and
a half minutes and is lost on a restart; results for one no longer waiting
are a `400`, as is a message holding anything besides the `tool_result`
blocks, or a `tool_choice` naming a tool only code may call.

**Strict tools.** A client tool declared `strict: true` has every call's
input checked against its `input_schema` once the call is written — `type`,
`properties`, `required`, `additionalProperties`, `items`, `minItems`,
`maxItems`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`, `$ref`, `minLength`,
`maxLength`, `pattern`, `minimum` and `maximum`. A call that does not match
never reaches the client: the model is told what is wrong with it and
writes the call again, within the reply's turn limit (a reply that runs out
of turns stops with `pause_turn`). A strict tool's input is not streamed as
it is written; it arrives once it has been checked.

**Continuing a reply.** A conversation that ends on an assistant turn holding
text (and no `tool_use`) is continued: the reply carries on from that text,
which the response does not repeat. Trailing whitespace on it is trimmed.

**Forced tool calls.** `any` and `tool` begin the reply with the model's own
tool-call syntax, found by rendering a sample call through the chat template.
A template that renders no call this server recognises leaves the choice to
the model. A continued reply is never forced.

The response is a `message`:

```json
{"id":"msg_6ae28c2a000005","type":"message","role":"assistant",
 "model":"gemma","content":[{"type":"text","text":"The capital of France is Paris."}],
 "stop_reason":"end_turn","stop_sequence":null,
 "usage":{"input_tokens":27,"output_tokens":7,
          "cache_creation_input_tokens":0,"cache_read_input_tokens":0}}
```

- `content` — a `thinking` block when the model reasoned (its `signature` is
  empty), a `text` block for the answer, and one `tool_use` block per call,
  with `input` as an object.
- `stop_reason` — `end_turn`, `max_tokens`, `stop_sequence` (with the matched
  string in `stop_sequence`), `tool_use` or `pause_turn`.
- `container` — `id`, `expires_at` and `skills`, when the request declares
  the code execution tool.
- `usage` — `input_tokens` is the part of the prompt that was processed, and
  `cache_read_input_tokens` the part served from the prefix cache; together
  they are the whole prompt. `cache_creation_input_tokens` is always `0`:
  caching is automatic and costs nothing to write.

Streamed, the events are `message_start`, `ping`, then for each block
`content_block_start`, its deltas (`thinking_delta`, `text_delta` or
`input_json_delta`) and `content_block_stop`, then `message_delta` with
`stop_reason`, the `container` when there is one, and the final `usage`,
then `message_stop`. A failure mid-stream is an `error` event.

A call to one of the client's tools is streamed as it is written: its
`tool_use` block starts once the tool's name is written, and its input
follows as `input_json_delta`s, which join into the same input the finished
call has. How finely depends on the model's call format:

- gemma-4's: each argument once written, and a long string such as a file's
  content a few characters at a time.
- JSON calls (`<tool_call>{…}` as Qwen2.5 and Hermes write them,
  `[TOOL_CALLS]`): the arguments object's own text as it is written, up to
  anything that is not JSON (a raw line break inside a string, say), after
  which the rest waits for the whole call.
- Tag formats (`<function=…>`/`<parameter=…>`, `<arg_key>`, invoke tags):
  each parameter once it is whole, since its value's type (string or JSON)
  is known only then.

`eager_input_streaming` is accepted and changes nothing. Server tool calls
send their input as one `input_json_delta`. If the model stops in the
middle of a call (at `max_tokens`, or writing something that turns out not
to be a call), the block is closed with the input written up to there,
the text stays in the reply as written, and the call does not count: the
reply stops with `max_tokens` or `end_turn`, not `tool_use`. A reply that
ends on a whole call whose closing marker the model never wrote counts
that call, streamed or not.

#### `POST /v1/messages/count_tokens`

Takes `messages`, `system`, `tools`, `tool_choice`, `thinking`,
`output_config` and `container` as `/v1/messages` does, renders them the
same way, and answers `{"input_tokens": N}` without generating or opening a
container.

#### Files

`POST /v1/files` takes a `multipart/form-data` body with the file in a field
named `file`, up to 500 MiB, and answers with its metadata:

```json
{"id":"file_17dbcfb6b1424048b4221e031616e562","type":"file","filename":"memo.pdf",
 "mime_type":"application/pdf","size_bytes":569,"created_at":"2026-10-29T01:13:20Z",
 "downloadable":true}
```

The `mime_type` is the one the upload declares, or else what the bytes are,
or else what the file name suggests. Files are stored under
`[orangu-server].api_data` and kept until deleted.

`GET /v1/files` lists them newest first, `limit` at a time (default 20, at
most 1000), paging with `after_id` and `before_id` and answering with
`data`, `has_more`, `first_id` and `last_id`; `scope_id` lists only the
files a session stored from its outputs. `GET /v1/files/{id}` is one
file's metadata, `GET /v1/files/{id}/content` its bytes, and `DELETE
/v1/files/{id}` answers `{"id": …, "type": "file_deleted"}`.

A `document` or `image` block in `/v1/messages` can name a stored file
instead of carrying it: `"source": {"type": "file", "file_id": "file_…"}`.
A PDF is read as its embedded text, a picture as a picture, and any other
file as UTF-8 text; a file that is none of these, or an unknown `file_id`,
is a `400`. This applies inside `tool_result` content too, and in batched
requests.

#### Skills

`POST /v1/skills` takes a `multipart/form-data` body: the skill's files,
each in a `files` field (`files[]` too) whose file name is its path, and
optionally a `display_name`. The files make one folder, with a `SKILL.md` at
its root — `my-skill/SKILL.md`, `my-skill/scripts/run.py`, … — or are that
folder's contents with no folder named, up to 30 MiB in all. `SKILL.md`
opens with front matter giving the skill's `name` (lowercase letters,
digits and hyphens, at most 64; the folder's name when it gives none) and
`description` (at most 1024 characters):

```markdown
---
name: csv-report
description: Turns a CSV file into a summary report. Use when asked to report on a CSV.
---
# CSV report
Run `python3 /skills/csv-report/report.py <file>` and …
```

The answer is the skill, its first version made from the upload:

```json
{"id":"skill_85ea05a10ee747ec8e6c4144ce3f3c9d","type":"skill",
 "display_name":"CSV report","latest_version_id":"1793261310635549",
 "source":{"type":"custom"},"created_at":"2026-10-29T03:28:30Z",
 "updated_at":"2026-10-29T03:28:30Z"}
```

`display_name` is the `name` when the upload gives none. `POST
/v1/skills/{id}/versions` takes another upload of the same skill — its
`SKILL.md` has to give the same `name` — and answers with the version
(`id`, `type: "skill_version"`, `skill_id`, `name`, `description`,
`created_at`). A version's `id` is when it was made, in Unix microseconds.

`GET /v1/skills` lists skills newest first, and `GET
/v1/skills/{id}/versions` a skill's versions newest first, `limit` at a time
(default 20, at most 1000): the answer has `data`, `has_more` and
`next_page`, which passed back as `page` gives the next page. Every skill
here is `custom`; `source=anthropic` lists none. `DELETE
/v1/skills/{id}/versions/{version}` removes a version, and `DELETE
/v1/skills/{id}` a skill whose versions are all removed — with versions
left it is a `400`. Skills are stored under `[orangu-server].api_data`.

#### Message Batches

`POST /v1/messages/batches` takes `{"requests": [{"custom_id": …, "params":
{…}}, …]}` — up to 100,000 requests and 256 MiB — where each `params` is a
`/v1/messages` request body without `stream`. Every request is checked
before the batch is accepted: a malformed one, a streamed one, or a
`custom_id` that is not 1–64 letters, digits, `_` or `-`, or is used twice,
refuses the whole batch with a `400`.

The answer is the batch:

```json
{"id":"msgbatch_07d2c158d2c347e9a8e02e790f8c0656","type":"message_batch",
 "processing_status":"in_progress",
 "request_counts":{"processing":7,"succeeded":0,"errored":0,"canceled":0,"expired":0},
 "created_at":"2026-10-29T01:13:30Z","expires_at":"2026-10-30T01:13:30Z",
 "ended_at":null,"cancel_initiated_at":null,"archived_at":null,"results_url":null}
```

The server runs a batch's requests in the background, as many at once as it
has slots, through the same path as a non-streamed `/v1/messages` request.
They share the queue with interactive requests; one the full queue turns
away is retried rather than failed. A batch is stored under
`[orangu-server].api_data`, and one still running when the server stops is
resumed when it starts again, without repeating a finished request. A
request not finished 24 hours after the batch was created ends `expired`.

`GET /v1/messages/batches/{id}` reports progress in `request_counts`, and
`processing_status` moves from `in_progress` to `ended` (through
`canceling` when cancelled). `POST /v1/messages/batches/{id}/cancel` lets the
requests already running finish and ends the rest `canceled`. `GET
/v1/messages/batches` lists batches newest first, paged as `GET /v1/files`
is.

Once a batch has ended, `results_url` names `GET
/v1/messages/batches/{id}/results`: one JSON line per request, in the order
they finished —

```json
{"custom_id":"q1","result":{"type":"succeeded","message":{"id":"msg_…","type":"message",…}}}
{"custom_id":"bad","result":{"type":"errored","error":{"type":"error","error":{"type":"invalid_request_error","message":"…"}}}}
```

— where a `canceled` or `expired` result carries nothing else. Before then it
is a `400`. `DELETE /v1/messages/batches/{id}` removes an ended batch and its
results; a running one has to be cancelled first.

#### Managed Agents

The Managed Agents endpoints run an agent loop on the server: an **agent**
is a stored configuration (model, system prompt, tools, MCP servers,
skills), an **environment** says what its tools may reach, and a
**session** runs the agent in an environment, in a code execution
container of its own. A client sends events to a session and reads the
events it produces. Anthropic's SDKs reach them as `client.beta.agents`,
`client.beta.environments` and `client.beta.sessions`; the
`anthropic-beta: managed-agents-2026-04-01` header they send is accepted
and not required.

`POST /v1/agents` takes `name` and `model` (an id, or `{"id": …, "speed":
…, "effort": …}`), and optionally `description`, `system`, `tools`,
`mcp_servers` (`[{"type": "url", "name": …, "url": …}]`), `skills`
(`[{"type": "custom", "skill_id": …, "version": …}]`, from the Skills API)
and `metadata`. `tools` lists:

- `{"type": "agent_toolset_20260401"}` — the built-in tools `bash`, `read`,
  `write`, `edit`, `glob`, `grep`, `web_fetch` and `web_search`. A
  `default_config` (`enabled`, `permission_policy`) applies to all of them,
  and a `configs` entry (`{"name": "bash", "enabled": false}`) to one; the
  web tools' entries also take `allowed_domains` or `blocked_domains`. The
  answer shows every tool's resolved config.
- `{"type": "mcp_toolset", "mcp_server_name": …}` — the tools of one of the
  agent's MCP servers, with the same `default_config` and `configs`.
- `{"type": "custom", "name": …, "description": …, "input_schema": …}` — a
  tool the client runs.

The answer is the agent at `version` 1. `POST /v1/agents/{id}` changes the
fields it names — `metadata` pairs merge, a `null` value removes one — and
adds a version when anything changed; a `version` in the body must be the
current one, or it is a `409`. `GET /v1/agents/{id}?version=N` is one
version, and `GET /v1/agents/{id}/versions` lists them newest first. `POST
/v1/agents/{id}/archive` makes an agent read-only for good: no new session
can use it, and the list leaves it out unless `include_archived=true`.

`POST /v1/environments` takes `name`, and optionally `description`,
`metadata` and `config`: `{"type": "cloud", "networking": …}`. Networking is
what the session's sandbox may reach:

- `{"type": "unrestricted"}` — the server's own network.
- `{"type": "limited", "allowed_hosts": […], "allow_mcp_servers": …}` — no
  network at all in the sandbox. `allowed_hosts` limits the web tools to
  those hosts (a bare name is that host, `*.example.com` its subdomains),
  and the web tools are not offered when it lists none. A session of an
  agent whose MCP servers it does not allow is a `400`. This is the
  networking of an environment that gives none.

`packages` lists nothing to install — the sandbox runs the server's own
programs, so a package to install is a `400` — and `type: "self_hosted"` is
a `400`. `DELETE /v1/environments/{id}` answers `{"id": …, "type":
"environment_deleted"}`; `POST /v1/environments/{id}/archive` keeps it,
read-only, for the sessions that use it.

`POST /v1/sessions` takes `agent` — its id (the latest version), `{"type":
"agent", "id": …, "version": …}`, or `{"type": "agent_with_overrides",
"id": …, "model": …, "system": …, "tools": …, "mcp_servers": …, "skills":
…}` — and `environment_id`, and optionally `title`, `metadata`,
`resources` and `initial_events` (up to 50 `user.message` events, sent as
the session starts). A `{"type": "file", "file_id": …, "mount_path": …}`
resource copies a Files API file to `/mnt/session/uploads/<mount_path>`
(`/mnt/session/uploads/<file_id>` without one), read-only, and the agent is
told where it is. What the agent writes under `/mnt/session/outputs` is
stored in the Files API each time a turn ends — a changed file replaces its
earlier copy — and `GET /v1/files?scope_id=<session id>` lists those files,
each with `scope: {"id": <session id>, "type": "session"}`. A
`{"type": "memory_store", "memory_store_id": …, "access": …, "instructions":
…}` resource (up to 8; see **Memory stores**) mounts the store at
`/mnt/memory/<name>`, the store's name in lower case with anything but
letters, digits, `-`, `_` and `.` made `-`; `access` is `read_write` (the
default) or `read_only`, and the system prompt lists each mount with its
access, the store's `description` and the resource's `instructions`. The
agent's skills are copied to `/workspace/skills/<name>`. A session of an
agent with `bash` or `grep` on needs `[orangu-server].code_execution =
bwrap`. `vault_ids` lists the vaults whose credentials the session uses
(see **Vaults**); `budget` is a `400`.

A session is `idle` until it has work, `running` while the agent loop runs,
and `idle` again when the loop stops. `POST /v1/sessions/{id}/events` takes
`{"events": […]}` and answers with them as recorded (`{"data": […]}`); a
bad one refuses them all:

- `user.message` — `content` blocks (`text`, `image`, `document`) or a
  string. Sent while the loop runs, it joins the conversation at the next
  model request.
- `user.interrupt` — stops the loop: a model request or tool call in
  progress ends, and the session goes idle.
- `user.custom_tool_result` — `custom_tool_use_id` (the id of the
  `agent.custom_tool_use` event), `content`, `is_error`.
- `user.tool_confirmation` — `tool_use_id` (the id of an `agent.tool_use`
  or `agent.mcp_tool_use` event that asked), `result` `allow` or `deny`,
  and a `deny_message` the agent is given on a deny.
- `system.message` — text `content` added to the system prompt from then
  on.
- `user.define_outcome` — `description` (the task), `rubric` (`{"type":
  "text", "content": …}` or `{"type": "file", "file_id": …}`, the criteria
  the work is graded by) and `max_iterations` (1–20, default 3); it is also
  accepted in `initial_events`. It is echoed with its `outcome_id`, and the
  agent starts on it: the task and the rubric are its message. One outcome
  is worked on at a time; another before the first has ended is a `400`.

The loop sends the conversation through the path a non-streamed
`/v1/messages` request takes, the agent's tools declared to the model, and
records what happens as events: `session.status_running`;
`span.model_request_start` and `span.model_request_end` (with
`model_usage`) around each model request; `agent.message` for its text and
`agent.thinking` when it reasons; `agent.tool_use` and `agent.tool_result`
for a built-in tool, `agent.mcp_tool_use` and `agent.mcp_tool_result` for an
MCP tool, `agent.custom_tool_use` for a custom one; `session.error` when a
model request fails; and `session.status_idle` with a `stop_reason` —
`end_turn`, `requires_action` (with the `event_ids` waiting on the client)
or `retries_exhausted` after an error. The loop stops after 50 model
requests and waits for the next message.

When the agent's turn ends while an outcome is active, a grader — a model
request of its own, which sees the task, the rubric, the agent's last
messages, the files in `/mnt/session/outputs` (their text, up to 80 000
characters in all) and a listing of `/workspace` — judges the work, and
the events are `span.outcome_evaluation_start` and
`span.outcome_evaluation_ongoing` (`outcome_id`, `iteration`, from 0), then
`span.outcome_evaluation_end` with `result`, `explanation` and the grader's
`usage`. `satisfied` and `failed` (a rubric the task cannot meet) end the
outcome and the session goes idle; `needs_revision` hands the agent the
explanation and it goes on, the session still running; on the last
iteration it is `max_iterations_reached` instead, and the agent makes one
final revision that is not graded. A `user.interrupt` while an outcome is
active ends it as `interrupted` (with an empty
`outcome_evaluation_start_id` when grading had not begun). Each ended
grading is in the session's `outcome_evaluations` (`outcome_id`,
`description`, `iteration`, `result`, `explanation`, `completed_at`).

Each built-in and MCP call carries `evaluated_permission` and the
`evaluation` behind it, from its `permission_policy`:

- `always_allow` (the built-in tools' default) runs the call.
- `always_ask` (the MCP tools' default) waits for a
  `user.tool_confirmation`.
- `auto` runs a call that touches only the container — `bash`, `read`,
  `write`, `edit`, `glob`, `grep` — and asks about one that reaches outside
  it: the web tools and MCP tools.

A custom tool call always waits for its `user.custom_tool_result`. While a
call waits, a `user.message` is a `400`.

The built-in tools work in the container, whose working directory is
`/workspace` and whose files last as long as the session. `read`, `write`,
`edit` and `glob` are the workspace tools `orangu` gives a model
(`show_file`, `create_file`, `modify_file`, and `list_directory` with a
`pattern`), confined to `/workspace` and the session's mounts, and refusing
to write in a read-only one — `edit`'s `old_string`
must occur once, unless `replace_all`; `glob` lists the matching files,
newest first. `read` of a picture (PNG, JPEG, GIF or WebP), on a model
with a vision tower (see **Pictures**), returns the picture itself as an
`image` block, for the model to look at. `bash` and `grep` run in the container's
sandbox, as code execution commands do (see **Code execution**), with the
environment's networking. `web_fetch` and `web_search` are the server tools
of the same names, and need `[orangu-server].web_search` set for a search.

`GET /v1/sessions/{id}/events` lists a session's events oldest first
(`order=desc` for newest first; `types[]` to pick some), `limit` at a time
(default 1000) — `data` and `next_page`, which passed back as `page` gives
the next page. `GET /v1/sessions/{id}/events/stream` sends each event as it
is recorded, as a server-sent event named by its `type`; it starts with the
next event, so open it before sending and list the events for what came
before.

With `event_deltas[]=agent.message`, the stream also previews the agent's
text while the model writes it: an `event_start` (`{"type": "event_start",
"event": {"type": "agent.message", "id": …}}`), then `event_delta`s
(`{"type": "event_delta", "event_id": …, "delta": {"type":
"content_delta", "index": 0, "content": {"type": "text", "text": …}}}`),
then the buffered `agent.message` with the same `id`, whose text the deltas
are a prefix of. `event_deltas[]=agent.thinking` sends an `event_start` when
the model begins to reason, and its `agent.thinking` takes the same id.
Previews go only to the connection that asked for them, are never listed
among the events, and are not replayed: a connection opened mid-request sees
none for it. Any other value, or more than 100, is a `400`.

`GET /v1/sessions` lists sessions newest first (filters `agent_id`,
`agent_version`, `statuses[]`, `include_archived`; `order=asc`), paged as
the events are. A session shows the agent it runs, its `resources`, its
`usage` (tokens, and `active_seconds`) and `stats`. `POST
/v1/sessions/{id}` changes `title`, `metadata`, and — while it is idle — the
agent's `tools` and `mcp_servers` for this session only. `POST
/v1/sessions/{id}/archive` stops it and makes it read-only, and `DELETE
/v1/sessions/{id}` stops it and removes it with its container. Agents,
environments and sessions are stored under `[orangu-server].api_data`; a
session's loop stopped by a server restart leaves it idle, and the next
event starts it again.

#### Multiagent sessions

An agent's `multiagent` block makes it a coordinator: `{"type":
"multiagent_20261001", "subagents": {"type": "enabled", "predefined_agents":
[…], "inline_agents": {"type": "enabled" | "disabled"}}, "workflows": …,
"advisor": …}`, or the older `{"type": "coordinator", "agents": […]}`. A
roster entry is an agent id (its latest version), `{"type": "agent", "id",
"version"}`, or `{"type": "self"}` — the coordinator itself, delegating no
further, which responses show as a reference to the agent's own id and
version; at most 20, and an agent with a `multiagent` block of its own
cannot be one (a `400`). `inline_agents` enabled (the default) lets the
coordinator also define agents of its own, which run its model and tools;
disabled, the list needs at least one agent (a `400`). An update that
sends a `multiagent_20261001` block to an agent that has one is merged into
it, level by level: a key left out keeps its value, a `null` one takes its
default, a `predefined_agents` list replaces the stored one, and a member
whose `type` changes is replaced whole. Workflows and the advisor are
described below.

In a session of a coordinator, its thread — the primary thread, whose
events are the session's — has two more tools. `list_agents` shows the
roster and the threads already started; `send_to_agent` (`message`, and
`agent_name` for a new thread, `session_thread_id` for a follow-up, or
`inline_agent` with a `name` and `system`) gives an agent a task in a
thread of its own and returns at once. A thread runs in parallel with the
others, in the session's container, with its own conversation, agent,
tools and event stream; it sees none of the coordinator's conversation.
When its turn ends, its last reply is its report: an
`agent.thread_message_sent` on its stream, an
`agent.thread_message_received` (`from_session_thread_id`,
`from_agent_name`, `content`) on the session's, and a message the
coordinator goes on from. A session holds at most 25 child threads,
archived ones aside.

The session's stream shows `session.thread_created`,
`session.thread_status_running`, `session.thread_status_idle` (with its
`stop_reason`) and `session.thread_status_terminated` for each child
(`session_thread_id`, `agent_name`), the coordinator's
`agent.thread_message_sent` for each task, and — with `session_thread_id`
— every call of a child's that waits on the client, answered on the
session's own events endpoint like any other (the answer reaches the
thread that asked). The session is `running` while any thread runs, and its
`session.status_idle` comes once all are idle, with `requires_action` and
the waiting calls' `event_ids` when any thread waits. A `user.interrupt`
stops every thread, or only the one its `session_thread_id` names.

`GET /v1/sessions/{id}/threads` lists the threads, the primary first
(`parent_thread_id` null), each with `agent`, `status` (`running`, `idle`
or `terminated`), `usage` and `parent_thread_id`; `GET …/threads/{thread}`
reads one, `POST …/threads/{thread}/archive` terminates an idle child (a
running one is a `400`: interrupt it first), and `GET
…/threads/{thread}/events` and `…/threads/{thread}/stream` are the thread's
own events, listed and streamed as the session's are (`event_deltas[]`
included). A child's tokens count in its own `usage` and the session's.

**Advisor.** `"advisor": {"type": "enabled", "model": …}` in a
`multiagent_20261001` block — or one `{"type": "advisor", "model": …}`
entry among a coordinator's `agents` — gives the session's primary thread
an `advisor` tool, with or without a roster. An update that leaves
`advisor` out keeps it; `{"type": "disabled"}` removes it. While it is
enabled, `anthropic.advisor` is a reserved name: listing an agent with that
name is a `400`. Each consultation is a thread of its own, named
`anthropic.advisor`, whose `agent` is `{"type": "advisor", "model": …}`: the
session's stream shows `session.thread_created`,
`session.thread_status_running`, the advice as an
`agent.thread_message_received` (`from_agent_name` `anthropic.advisor`),
then the thread's `session.thread_status_idle` and
`session.thread_status_terminated`; no `agent.tool_use` is emitted for it.
The advisor reads the primary thread's conversation and answers on the
served model. A consultation that fails, or that a `user.interrupt` naming
its thread abandons, leaves the agent going on with a notice that there is
no advice; an interrupt of the whole session stops it with everything
else. Advisor threads do not count toward the 25 child threads, and are
invisible to `list_agents` and `send_to_agent`; their tokens count in
their own `usage` and the session's.

**Dynamic workflows.** `"workflows": {"type": "enabled"}` — the default in a
`multiagent_20261001` block — lets the session's agent start workflow runs:
a plan, which the agent writes as Python, that runs many agents in phases,
in the background, while the agent works on or ends its turn. Its primary
thread gets two tools for it, whose calls are on no stream:
`ant__start_workflow_run` (a `name`, `description`, `phases` and
`timeout_minutes` — 1 to 1440, 1440 when left out — and the plan's `code`)
and `ant__stop_workflow_run`. Custom tool names starting with `ant__` are
refused (a `400`). The plan runs in the session's container, with
`await agent(prompt, name=…, system=…, use=…)`, which runs one agent in a
thread of its own and returns its final message (raising `AgentFailed`
when the agent fails), and `with phase(name):`, which marks a phase. An
agent the plan defines runs the session agent's model, tools, MCP servers
and skills, with the `system` the plan gives it; its thread's `agent` has
`"type": "inline"`. `use=` names one of `workflows.predefined_agents`
instead, which runs with its own configuration; with `inline_agents`
disabled, every agent is one of those. Up to 64 of a run's agents work at
once, and a run makes at most 1000 threads. A start whose code is that of
a run still open is refused, with that run's id. When the plan ends, the agent
is sent a notice with what the plan printed and takes a turn on it.

The primary thread's stream shows each run: `workflow_run.created`
(`workflow_run_id`, `name`, `description`, and `phases`, each with its
`id`, `name` and `description`), `workflow_run.status_running`,
`workflow_run.phase_started` (`workflow_run_phase_id`) and
`workflow_run.phase_ended` (with `phase_started_id`), and last
`workflow_run.status_ended`, whose `result` is `completed`, `stopped` (the
agent stopped the run, or the session was archived) or `error`, with an
`error` of type `program_error` (the plan's code failed),
`thread_limit_error`, `timeout_error` or `unknown_error` (the server
stopped while the run was open). A `workflow_run.error` with the same
`error` comes before an `error` result, and alone, with a null
`workflow_run_id`, for a start refused: `program_error` for code that does
not compile, `max_workflow_runs_error` with 10 runs open. A run's threads
carry its `workflow_run_id`, as do their `session.thread_created`; their
status events are on the session's stream, their messages only on their
own. They are archived when they report and when the run ends, do not
count toward the 25 child threads, and are invisible to `list_agents`; an
interrupt does not stop them, and archiving one while its run is open is a
`400` whose `error.details.error_code` is `workflow_run_open`, as is a
change to the session's agent. The session is `running` while a run is
open, and says `session.status_idle` only with `requires_action`, until
the runs have ended and the agent's turn on their notices ends. Archiving
the session ends its runs as `stopped`; deleting it ends them.

#### Vaults

A vault holds credentials a session uses without seeing them: `POST
/v1/vaults` (`display_name`, `metadata`), `GET` to list them
(`include_archived`), `GET`/`POST`/`DELETE /v1/vaults/{id}`, and `POST
…/archive`, which also archives its credentials. A session names vaults in
`vault_ids` when it is created.

A vault's credentials are `POST /v1/vaults/{id}/credentials` (`auth`,
`display_name`, `metadata`), listed, read, changed, deleted and archived
under it — at most 20 active ones a vault, unique by MCP server URL or
variable name (a duplicate is a `409`). The secret fields (`token`,
`access_token`, `refresh_token`, `client_secret`, `secret_value`) are
written, never shown: they are kept in a file of their own under
`[orangu-server].api_data` that only the server's user can read, and
archiving deletes them. A credential's URL, variable name, token endpoint
and client id do not change; archive it and create another.

- `static_bearer` (`mcp_server_url`, `token`) and `mcp_oauth`
  (`mcp_server_url`, `access_token`, `expires_at`, and `refresh`:
  `refresh_token`, `client_id`, `token_endpoint`, `token_endpoint_auth` of
  type `none`, `client_secret_basic` or `client_secret_post`, `scope`,
  `resource`) are the bearer token of the session's MCP server at that URL
  — matched with the scheme and host in lower case, a default port and a
  trailing slash left out. An `mcp_oauth` token about to run out is
  refreshed through its token endpoint first. `POST
  …/credentials/{id}/mcp_oauth_validate` refreshes one when it can and asks
  its server to `initialize` with the token: `status` is `valid`, `invalid`
  (a 401 or 403) or `unknown`, with the `refresh` and `mcp_probe` HTTP
  responses.
- `environment_variable` (`secret_name`, `secret_value`, `networking` —
  `{"type": "limited", "allowed_hosts": […]}`, `*.example.com` covering
  subdomains, or `{"type": "unrestricted"}` — and `injection_location`
  `{header, body}`, both on when left out) is a variable whose value in the
  sandbox is a placeholder. In an environment with `unrestricted`
  networking, the sandbox's `HTTPS_PROXY` and `HTTP_PROXY` name a proxy of
  the session's, and its certificate bundle (`SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`,
  `GIT_SSL_CAINFO`, `PIP_CERT`) trusts that proxy's own certificate
  authority. A request to an allowed host is opened there, its
  placeholders replaced with the secret in the headers and body the
  credential allows, and sent on over TLS of its own; a request to any
  other host passes through untouched, placeholder and all. With `limited`
  networking the sandbox has no network, and the placeholder goes nowhere.
  The secret reaches the host it was meant for — what that host sends
  back, the agent sees.

#### Memory stores

A memory store is a named collection of small text documents (memories,
each at most 100 KB, addressed by an absolute `path` such as `/notes/a.md`)
that outlives the sessions it is attached to. Every change to a memory adds
an immutable version (`memver_…`) recording the `operation` (`created`,
`modified` or `deleted`), the path and content after it, and who made it:
`created_by` is `{"type": "session_actor", "session_id": …}` for an agent,
`{"type": "api_actor", …}` for an API call.

- `POST /v1/memory_stores` (`name`, `description` — written for the model,
  which is shown it — and `metadata`), `GET` to list them
  (`include_archived`, `created_at_gte`/`_lte`), and `GET`/`POST`/`DELETE
  /v1/memory_stores/{id}`. `POST /v1/memory_stores/{id}/archive` makes it
  read-only: its memories can be read and deleted, not changed, and no new
  session can attach it.
- `POST /v1/memory_stores/{id}/memories` (`path`, `content`) creates one;
  a path already taken is a `409` of type `memory_path_conflict_error`
  naming the `conflicting_memory_id`. `GET …/memories` lists them in path
  order (`path_prefix`; `depth` folds deeper ones into `memory_prefix`
  entries, one per directory; `view=full` includes `content`). `GET`,
  `POST` (or `PATCH`) and `DELETE …/memories/{memory_id}` read, change
  (`content`, `path` or both; a `precondition` `{"type":
  "content_sha256", "content_sha256": …}` that no longer matches is a
  `409` of type `memory_precondition_failed_error`) and delete one
  (`expected_content_sha256` likewise).
- `GET …/memory_versions` lists the versions newest first (`memory_id`,
  `operation`, `session_id`, `created_at_gte`/`_lte`; `view=full`), `GET
  …/memory_versions/{id}` reads one, and `POST …/redact` clears a past
  version's `content`, `content_sha256`, `content_size_bytes` and `path`,
  keeping who made it and when; the current version cannot be redacted.

An attached store's memories are written into the session's container as
files when the session is created. After each `bash`, `write` or `edit`
call the agent makes, what changed under a `read_write` mount is recorded
in the store: a new file creates a memory, a changed one modifies it, a
removed one deletes it. Files over 100 KB, files that are not text, and
names starting with `.` are left out. Changes made to the store by others
while the session runs are seen by later sessions, not by that one.

#### Scheduled deployments

A deployment runs an agent on a cron schedule, each firing a new session.
`POST /v1/deployments` takes `name`, `description`, `agent` (an id or
`{"type": "agent", "id", "version"}`, pinned to its latest version when
none is named), `environment_id`, `vault_ids`, `resources`, `metadata`,
`initial_events` — at least one `user.message` or `user.define_outcome`,
with `system.message` events allowed too, which apply before the first
message — and `schedule`: `{"type": "cron", "expression", "timezone"}`, a
five-field POSIX expression (minute, hour, day of month, month, day of
week; `*`, ranges, steps, lists, month and day names; day of week 7 is
Sunday; a day matching either the day of month or the day of week fires
when both are restricted) in an IANA time zone, `UTC` when left out. A
schedule that never fires is a `400`, as is a `budget`. At most 1000
deployments, archived ones aside.

Times are wall-clock times in the zone: one a change of offset skips does
not fire, and one it repeats fires twice. The scheduler checks at the start
of every minute, so a run starts within a second or so of its time.
`schedule.upcoming_runs_at` lists the next five fire times (also while
paused; none once archived), and `schedule.last_run_at` when the last
scheduled run started.

`GET /v1/deployments` lists them (`agent_id`, `status`,
`include_archived`); `GET`/`POST`/`DELETE /v1/deployments/{id}` read,
change — any field `create` takes; a new schedule starts from now — and
delete one. `POST …/pause` stops its scheduled runs (`status` `paused`,
`paused_reason` `{"type": "manual"}`); `POST …/unpause` resumes from the
next fire time, without making up for those missed; `POST …/archive` ends
it for good. `POST …/run` makes a session now — while paused too — and
returns the run.

Every attempt is a run: `GET /v1/deployment_runs` (`deployment_id`,
`has_error`) and `GET /v1/deployment_runs/{id}`, each with
`trigger_context` (`{"type": "schedule", "scheduled_at"}` or `{"type":
"manual"}`), the pinned `agent`, and the `session_id` it made or an
`error`: `environment_not_found_error`, `environment_archived_error`,
`vault_not_found_error`, `vault_archived_error`, `agent_archived_error` or
`session_creation_rejected_error`, with a `message`. A scheduled run that
fails on its environment or a vault pauses the deployment, with
`paused_reason` `{"type": "error", "error": {"type": …}}`; other failures
leave it running. Archiving an agent archives its deployments, and one
whose agent is gone is archived when it next fires. The session a run
makes has the deployment's `name` as its `title`, and no client is
attached: calls that wait for approval wait until one answers them.

#### Webhooks

The server POSTs a thin event to each subscribed endpoint when a resource
changes state: `{"type": "event", "id": "whe_…", "created_at", "data":
{"type", "id", "organization_id": "local", "workspace_id": "default"}}`,
`data.id` the resource's id. Each delivery carries `webhook-id` (the event's
`id`), `webhook-timestamp` and `webhook-signature` — `v1,` and the base64
HMAC-SHA256 of `{id}.{timestamp}.{body}`, keyed with the endpoint's secret —
so the SDKs' `beta.webhooks.unwrap()` verifies it with that secret.

Endpoints are kept by the server: `POST /v1/webhook_endpoints` (`url`, an
http or https URL; `event_types`; `description`) returns the endpoint with
its `secret` (`whsec_` and 32 random bytes in base64) — shown this once,
and kept in a file only the server's user can read. `GET` lists them, `GET`,
`POST` and `DELETE /v1/webhook_endpoints/{id}` read, change (`url`,
`event_types`, `description`, and `enabled`) and delete one, and `POST
…/rotate_secret` replaces the secret, returning the new one. An endpoint's
URL is reached wherever it points, the server's own network included.

A delivery that gets no `2xx` is tried up to three times in all, 5 to 120
seconds apart; then the event is dropped. A redirect (`3xx`) disables the
endpoint at once, as does a day of every delivery failing; `disabled_reason`
says which, and `enabled: true` turns it back on. Events are not stored, and
not delivered in order: fetch the resource.

The event types: `session.status_scheduled`, `session.status_run_started`,
`session.status_idled`, `session.status_terminated` (on archive),
`session.thread_created`, `session.thread_idled` and
`session.thread_terminated` (child threads), `session.outcome_evaluation_ended`,
`session.updated`, `session.deleted`; `agent.created`, `agent.updated` (a
new version), `agent.archived`; `environment.created`,
`environment.updated` (a change), `environment.archived`,
`environment.deleted`; `vault.created`, `vault.archived`, `vault.deleted`,
`vault_credential.created`, `vault_credential.archived`,
`vault_credential.deleted`, `vault_credential.refresh_failed`;
`memory_store.created`, `memory_store.archived`, `memory_store.deleted`;
`deployment.created`, `deployment.updated`, `deployment.paused`,
`deployment.unpaused`, `deployment.archived`, `deployment.deleted`; and,
for scheduled runs only, `deployment_run.started`,
`deployment_run.succeeded` and `deployment_run.failed`.
`session.status_rescheduled` and `agent.deleted` may be subscribed to as
well; the server has no rescheduling or agent deletion, so they never come.

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
  figures the per-request console log prints. On a top-level node of a
  `[workers]` tree, `timings.workers` also lists the request's time at each
  node of the pipeline, in order — `[{"node": "top:8400", "ms": 200.5},
  {"node": "10.0.0.2:8400", "ms": 168.5}]` — where a worker's time is its
  round trip, its own workers included.
- **`prompt_progress`** (likewise) — `total`, `cache`, `processed`,
  `time_ms`, reported once per prefill chunk while the prompt is still being
  processed (see `return_progress` above).

On a streaming response they ride on the final chunk (the one carrying
`finish_reason`), immediately before `[DONE]`; on a non-streaming response
they are top-level fields, and the `Server-Timing` header carries the same
times — `prompt;dur=…, generate;dur=…`, then one `node<i>;desc="…";dur=…`
per node in a tree — for tools that read timings from headers. `orangu-bench --pp` reads them to report prefill
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

Four things make it `503`, and the body names which, because a probe that only
flips a status code leaves an operator with the alert and none of the reason:

- **`"queue full"`** — the admission queue is at `queue_limit`. A new unpinned
  request would be refused with `503` anyway, so saying so up front lets a
  balancer send it somewhere that can take it instead of spending a round trip
  to find out. With `queue_limit = 0` the queue never refuses, so the
  server is never unready for this reason however deep the queue gets.
- **`"device lost"`** — the GPU device was lost and this process is on its way
  out. It has not stopped being *alive*, and a supervisor restarting it is
  precisely the intended recovery, but nothing should be routed to it
  meanwhile. Checked first: a lost device makes every answer wrong, including
  a cheerful one about an empty queue.
- **`"serving a parent orangu-server"`** — this node is a worker in a
  `[workers]` tree, and its own API is off while it works for its parent.
- **`"a worker was lost"`** — a worker this node's plan relies on stopped
  answering, and the plan has not been repaired yet. The next request
  repairs it, and so does the worker coming back.

`200` with `"ok"` otherwise. Reachable without an `api_key`, like `/health` —
a readiness probe that needed a credential would fail closed exactly when a
balancer most needs an answer. It is a deliberate widening of what an
unauthenticated caller can see: `/ready` does disclose load, where `/health`
discloses nothing. That one fact is the price of being routable.

#### `GET /version`

Exactly which build is answering. The version alone is every build between two
releases; the commit is what tells them apart:

```json
{
  "name": "orangu",
  "version": "2.0.0",
  "commit": "fd48326ff",
  "build": "2.0.0 (fd48326ff)",
  "rustc": "1.91.0",
  "profile": "release",
  "target": "x86_64-unknown-linux-gnu",
  "frame_pointers": false
}
```

`commit` is the short hash the binary was built from, with `-dirty` appended
when tracked files differed from it, and `unknown` when there was no git to ask
— a source tarball. A packager who knows the commit sets `ORANGU_BUILD_COMMIT`
at build time. `build` is the same string every binary prints for `--version`.
Nothing here depends on the model, so it answers the same while one is loading.
It needs the `api_key` like every endpoint but `/health` and `/ready`.

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
| `image` | `null` for a language model; on a `qwen_image` model, the picture pipeline — see below |
| `video` | `null` but on an `ltxv` model: the video pipeline — see below |
| `audio` | `null` but on a `qwen3tts` model: the speech pipeline — see below |

`version` dates the release; `commit` is the only field that tells two builds
of one version apart, which during performance work is every pair of builds
that matters. Paired with `pid` and `uptime_seconds` it is what lets a
benchmark prove it is talking to the build it just launched rather than one
left over from a previous run.

Hardware-only startup flags — thread count, GPU layer count, batch size — are
not exposed here or anywhere else over HTTP; they appear only in the server's
own startup log.

On a picture model, `image` is what a client needs to know how a
picture will come out, and how long it will take, before asking for one:

| `image.` | |
| :-- | :-- |
| `architecture` | `qwen_image` or `qwen_image_2_1` |
| `size_unit` | the pixels each side of a picture must be a multiple of: `16`, or `32` for Qwen-Image 2.1 |
| `turbo` | `true` for Qwen-Image 2.1's step-distilled Turbo checkpoint, sampled on its eight-step schedule |
| `text_encoder`, `vae` | the companion files the pipeline was loaded with |
| `vision` | the vision projector a `qwen_image_2_1` pipeline edits attached pictures with, or `null` — an attached picture is then a starting point |
| `lora` | the adapter in the weights — `path`, and `steps`, the count a Lightning file's name says it was distilled for — or `null` for the base model |
| `n_layer`, `n_head`, `head_dim`, `dim` | the transformer's shape |
| `defaults` | what a request gets when it does not say: `size`, `steps`, `cfg_scale`, `negative_prompt`, `strength`, `format` — as they stand now |
| `configured` | the same six as the server came up, from the configuration |
| `token_passes_per_second`, `estimated_default_seconds` | the last measured rate at the defaults' size, and what a picture at the defaults costs at it — `null` before anything was measured |
| `rate` | the model behind both: `encode` (seconds per picture), `linear_per_token_pass` and `attention_per_token_pass` (seconds per latent-token pass, attention's measured at `attention_tokens` tokens and growing with the count), `decode_per_pixel`, `edit_encode_per_token` (an edit's encode per reference latent token, `null` until an edit ran), and `step_share` (the share of steps that run the transformer — the step cache, `image_cache`, reuses the rest; `1` with it off). Seconds for any settings = `encode + steps × step_share × n × passes × (linear + attention × n / attention_tokens) + decode_per_pixel × pixels`, `n` the latent tokens (`width/16 × height/16`) and `passes` 2 under guidance, 1 without; an edit attends over `n + r` keys in place of `n`, `r` the reference's tokens. `null` before anything was measured |

On a video model, `video` is the same for a clip:

| `video.` | |
| :-- | :-- |
| `architecture` | `ltxv` |
| `size_unit` | the pixels each side of a clip must be a multiple of: `64` when the defaults draw in two stages, `32` otherwise |
| `text_encoder`, `vae` | the text encoder and the video VAE the pipeline was loaded with |
| `audio_vae`, `upscaler` | the audio VAE and the latent upscaler, or `null` — clips are then silent, or drawn in one stage |
| `n_layer`, `dim`, `head_dim`, `audio_dim`, `audio_head_dim` | the transformer's shape: its blocks, and each stream's width and head width |
| `defaults` | what a request gets when it does not say: `size`, `frames`, `fps`, `seconds`, `audio`, `two_stage`, `quality` — as they stand now |
| `configured` | the same, as the server came up, from the configuration |
| `estimated_default_seconds` | what a clip at the defaults costs at the current rate |
| `rate` | the model behind it, seeded by the startup calibration and replaced by each finished clip's measurements: `encode` (seconds for the prompt), `per_step`, `per_token` and `per_token_sq` (a step's seconds), `decode_per_pixel` and `other_per_pixel` (per output pixel per frame: the video VAE; the upscaler and the file), `audio_per_second`. Seconds for any settings = `encode + Σ steps × (per_step + per_token × n + per_token_sq × n²) + pixels × frames × (decode_per_pixel + other_per_pixel) + audio_per_second × seconds of sound`, summed over the stages' steps (eight, then three), `n` the stage's video tokens: latent frames × `height/32` × `width/32`, a quarter of that in stage 1 of two |

On a speech model, `audio` is the same for a clip of speech:

| `audio.` | |
| :-- | :-- |
| `architecture` | `qwen3tts` |
| `codec` | the speech codec the pipeline was loaded with |
| `voice_packs` | the voice pack files loaded |
| `voices` | the voices they hold, each `{name, description}` |
| `languages` | `auto` and the languages the model names |
| `formats` | `wav`, `flac`, `pcm` |
| `sample_rate`, `frame_rate` | `24000` samples and `12.5` codec frames a second |
| `talker_layers`, `predictor_layers` | the talker's and the code predictor's depth |
| `defaults` | what a request gets when it does not say: `voice`, `language`, `format`, `temperature`, `top_k`, `repetition_penalty`, `max_seconds` — as they stand |
| `configured` | the same seven, as the server came up, from the configuration |

#### `POST /props`

Changes the picture defaults — the ones every request that does not say
otherwise gets, and so what a chat turn draws. The body is `{"image":
{…}}` with any of `size` (`WIDTHxHEIGHT`, multiples of 16), `steps`,
`cfg_scale` (`1` turns guidance off), `negative_prompt`, `strength`
(0–1), `format` (`png`, `jpeg`, `gif`, `webp`, `svg`); keys left out keep
their value, and `"reset": true` puts every one
back to the configuration first. The answer is `GET /props`' document, so
the caller sees what it set and what it now costs:

```sh
curl -s -X POST localhost:8100/props -H 'Content-Type: application/json' \
  -d '{"image": {"size": "512x512", "steps": 4}}'
```

`400` names the value the config loader would have refused too (a size
not in multiples of 16, zero steps, a guidance below 0, a strength outside
0–1) and changes nothing; `501` on a language model. The web console's
**Settings › Image** pane is this endpoint, on the console's own port as
`/api/props`; the change holds until the server restarts, when the
configuration's values are back.

On a video model the body is `{"video": {…}}` with any of `size`
(`WIDTHxHEIGHT`, multiples of 32, or of 64 in two stages), `seconds` or
`frames` (not both), `fps` (1–60), `audio`, `two_stage` and `quality`
(0–255), and `"reset": true` as above:

```sh
curl -s -X POST localhost:8100/props -H 'Content-Type: application/json' \
  -d '{"video": {"size": "512x320", "seconds": 3}}'
```

The checks are the ones `POST /v1/videos` makes, and a value that fails
one is a `400` that changes nothing; `{"video": …}` on any other model is
a `501`. The console's **Settings › Video** pane is this endpoint, and the
change holds until the server restarts.

On a speech model the body is `{"audio": {…}}` with any of `voice`,
`language`, `format` (`wav`, `flac`, `pcm`), `temperature` (0–2), `top_k`
(1 or more), `repetition_penalty` (1–2) and `max_seconds` (above 0, at
most 600), and `"reset": true` as above:

```sh
curl -s -X POST localhost:8100/props -H 'Content-Type: application/json' \
  -d '{"audio": {"voice": "1.7b-designed", "format": "flac"}}'
```

A value that fails the checks `POST /v1/audio/speech` makes refuses the
whole change with a `400`; `{"audio": …}` on any other model is a `501`.
The console's **Settings › Audio** pane is this endpoint, and the change
holds until the server restarts.

A top-level `[workers]` node also takes `{"workers": {"enabled": false}}`:
it serves alone — its model's own paths, with the prefix pool, paged KV
and MTP head a tree turns off — and lets its workers go, which serve their
own APIs again; `true` plans the tree again. `GET /props` has `workers`
(`enabled`, `switchable`, `plan`; `null` without `[workers]`). The switch
waits for no request: `409` while one runs or waits, on a worker or a node
without workers, and on one with only part of the model. The slots' kept
conversations are dropped by it. `orangu-bench --workers compare` is built
on this.

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

Prometheus text (`text/plain; version=0.0.4`). Five gauges describe the
scheduler right now:

| gauge | |
| :-- | :-- |
| `orangu_server_slots_total` | configured concurrent request slots |
| `orangu_server_slots_busy` | slots currently generating |
| `orangu_server_slots_idle` | slots free to take a request |
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

**About the process itself.** Related figures are folded into a few metrics
that carry a label, rather than one metric each. All of them are gauges.

| metric | notes |
| :-- | :-- |
| `orangu_server_host_info{name,version,commit,rustc,profile,target,os,os_version,kernel,arch}` | constant `1`; the build and the host are in the labels. `commit` and `rustc` read `unknown` when the build could not resolve them |
| `orangu_server_uptime{start_time}` | the value is seconds since the server started; the `start_time` label is when, as a Unix timestamp |
| `orangu_server_cpu{stat}` | `process_seconds` is CPU time this process has used so far (cumulative, so `rate()` gives cores busy); `load1`, `load5`, `load15` are the host's load averages (not reported on Windows); `logical_cores` is what the process may use |
| `orangu_server_http_requests{stat}` | `in_flight` is requests being handled right now on the API and web console listeners, and a streamed answer counts until its last byte is sent; scrapes of the dedicated metrics port are not counted, though a scrape through the API port counts itself; `total`, `client_errors` (4xx) and `server_errors` (5xx) count responses since start |
| `process_resident_memory_bytes`, `process_virtual_memory_bytes` | this process's memory |
| `process_open_fds` | open file descriptors (Linux only) |
| `orangu_server_host_memory_total_bytes`, `orangu_server_host_memory_available_bytes` | the host's physical memory |

**About a tree of workers.** Present when the server has a `[workers]`
section; the per-worker families carry a `worker` label, the address as
`[workers].workers` lists it.

| metric | notes |
| :-- | :-- |
| `orangu_server_workers_assigned` | gauge: `1` while this node works for a parent (its own API is off) |
| `orangu_server_workers_plans_total` | times this node planned its layers over its workers |
| `orangu_server_workers_recoveries_total` | times a lost worker made this node plan again mid-request |
| `orangu_server_worker_up{worker}` | gauge: whether the connection to the worker is up |
| `orangu_server_worker_forward_seconds{worker}` | histogram: one forward's round trip through the worker's subtree, network included |
| `orangu_server_worker_sent_bytes_total{worker}`, `orangu_server_worker_received_bytes_total{worker}` | bytes each way |
| `orangu_server_worker_failures_total{worker}` | requests to the worker that failed: unreachable, closed, or timed out |

**A second, unauthenticated copy of this endpoint.** Adding a `[prometheus]`
section (or passing `--metrics`) binds this exact response on a port of
its own, with no `api_key` check even when one is configured for
everything above — see **Monitoring: `/metrics` and `/ready`** in the
*Running the server* chapter. No section (absent) disables it; there is no
difference in content between the two, only in where each is reachable from.
That includes the build and host labels above, so on the dedicated port they
are visible to anyone who can reach it.
That port also answers `GET /` with a small static HTML page linking to
`/metrics`; every other path is a `404`.

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

#### `POST /logits`

```sh
curl -s localhost:8100/logits -H 'Content-Type: application/json' \
  -d '{"content": "fn main() {", "rows": 1, "top": 8}'
```

Runs a fixed token sequence through the model and answers, after every
prefix of it, the `top` most likely next tokens with their log-probabilities
over the whole vocabulary. Nothing is sampled: two servers given the same
`tokens` see the same input at every position, so their answers differ only
in arithmetic. `orangu-bench --kl` compares two such answers.

| Field | Default | Meaning |
|:--|:--|:--|
| `content` | — | text, tokenised with the model's special tokens |
| `tokens` | — | the exact token ids to run; wins over `content` |
| `rows` | 1 | positions a pass: 1 runs each as a decode step, 2–64 in multi-position passes, the path speculative verification takes |
| `top` | 32 | candidates per position, 1–256 |

The answer is `{"tokens": [...], "rows": N, "positions": [[[id, logprob], …], …], "seconds": S}`,
one entry per token. The run takes a request slot and waits for one like any
other request; its cache is its own and is not kept.

#### `POST /apply-template`

`{"messages": [...], "tools": [...]}` in, `{"prompt": "…"}` out: exactly the
string `/v1/chat/completions` would build from those messages and tool
definitions (`tools` is optional, and an empty list is none) and hand to the
model, including a `review`-role server's reasoning-suppression prefill.
`/tokenize` on the result counts what a request's tools cost in tokens. Nothing is
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
| `GET /api/sessions/{id}/files/{name}` | a file kept beside the session: a picture attached to a message (PNG, JPEG, GIF, WebP, SVG), one a `qwen_image` model made, a clip an `ltxv` model made (`video/mp4`, with `Range`), or speech a `qwen3tts` model made (`audio/wav` or `audio/flac`, with `Range`) |
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

A PNG, JPEG, GIF, WebP or SVG attachment is kept as a picture rather than read as text:
it is stored beside the session and comes back in the `attachments` event
with an `image_url` the page shows as a thumbnail. A model with a vision
tower (see **Pictures**) reads the picture where it was attached, on that
turn and every later one; a model without one is told a picture it cannot
see was attached. On a `qwen_image` server
the turn is a picture request — the text is the prompt, the last attached
picture is what it starts from — and the events are `progress` (`step`,
`steps`, `eta_seconds`) once per step, then the same `done` event as a text
reply, whose `html` holds the picture and whose `content` is a markdown
image over the session file's URL.

On an `ltxv` server the turn is a clip request — the text is the prompt,
and every knob is the server's `video_*` default — and the events are
`progress` (`step`, `steps`, `phase`, `eta_seconds`) at every phase and
step, then the same `done` event with `"video": true`. The clip is kept
beside the session as an `.mp4`; `content` is a markdown image over its
URL followed by a caption line (size, frames, frame rate, length, sound,
seed), and `html` renders that link as a `<video>` player with a save
control. The file route answers an `.mp4` as `video/mp4` with
`Accept-Ranges: bytes`, and a single `Range` with `206` and that part, so
the player can seek. Closing the stream stops the clip at its next
transformer block. A clip made here is not a `/v1/videos` job and is not
listed there.

On a `qwen3tts` server the turn is speech — the text is spoken, and every
knob is the server's `audio_*` default — and the events are `progress`
(`phase`: reading the text, speaking with the seconds so far, decoding),
then the same `done` event with `"audio": true`. The clip is kept beside
the session as a `.wav`, or a `.flac` when the default format is `flac`;
`content` is a markdown image over its URL followed by a caption line
(length, voice, language, seed, and "— cut at the length limit" when it
was cut), and `html` renders that link as an `<audio>` player with a
download button. The file route answers `audio/wav` or `audio/flac`, with
`Range`. Closing the stream stops the clip at its next frame.

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
it. `select` is a `403` when `[web].reexec` is `no`, `DELETE /api/models`
a `403` when `[web].delete` is `no`, and `POST /api/models/download` a `403`
when `[web].download` is `no`. `GET /api/models` reports the three as
`can_load`, `can_delete` and `can_download`, so the panel can leave out a
control that would only be refused.

Finally, the MCP inventory — the configuration file's MCP sections, which
orangu clients read and this server only keeps:

| Endpoint | |
| :-- | :-- |
| `GET /api/mcps` | `servers` — every configured MCP section: `name`, `endpoint`, `enabled`, `approval_mode` — and `path`, the configuration file (`null` for a bundled binary with no file, when the list is read-only) |
| `GET /api/mcps/{name}` | one of them, or `404` |
| `PUT /api/mcps` | the whole list, as an array of the same objects (`enabled` defaults to `true`, `approval_mode` to `writes`): every entry is checked first (`400` names the fault — a name that is empty, `orangu-server`, `web` or holds `[ ] = # ;`, an empty endpoint, an approval mode other than `auto`/`prompt`/`writes`/`deny`, a duplicate), then sections the list no longer names are removed and new or changed ones rewritten in place, the rest of the file untouched; the answer is `GET`'s document read back from the file with `restart_required: true` — a server started from the file is what reads it. `501` without a file |

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
  "commit": "fd48326ff",
  "build": "0.12.0 (fd48326ff)",
  "models": {
    "all": "bartowski/gemma-4-12B-it-GGUF",
    "code": "bartowski/gemma-4-12B-it-GGUF",
    "review": "bartowski/gemma-4-12B-it-GGUF",
    "explorer": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "embeddings": "bartowski/gemma-4-12B-it-GGUF"
  }
}
```

`version`, `commit` and `build` are the coordinator's own build, as on a
server's `GET /version` — `/version` itself is proxied through to the active
backend, so it answers for the server behind the coordinator.

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

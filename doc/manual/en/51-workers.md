\newpage

# Workers: one model over several machines

A model too large for one machine can still be served by several. With a
`[workers]` section, `orangu-server` nodes form a tree: the node clients
talk to runs some of the model's layers and hands the rest to its
workers, and a worker can hand part of its share to workers of its own.
For every prompt chunk and every generated token, the hidden states travel
down the tree and back, and the top-level node samples, checks stop
conditions and streams the reply, exactly as a single server does.

This chapter is the practical side: setting up a tree, securing it, and
watching it. The *Inference server* chapter's **The `[workers]` section**
has the reference for every key.

## When it helps

Splitting a model is about **room, not speed**. Every layer still runs once
per token, one machine after another, and each hand-over adds a network
round trip, so a model that fits on one machine is fastest there. A tree
is for a model that does not fit, or for machines that are each too small
to work together. Keep them close: on a LAN a round trip is a fraction of a
millisecond, and across the internet it is tens of milliseconds for every
token.

Each node holds in memory only the layers it runs. Measured with
`Llama-3.2-3B-Instruct-Q8_0` (3.4 GB) over three nodes of one machine:

| | after startup | after a request |
| :-- | --: | --: |
| one server alone | 3.5 GiB | 3.5 GiB |
| top (layers 0–8, and the embedding and head) | 178 MiB | 1.5 GiB |
| middle (layers 9–17) | 188 MiB | 1.1 GiB |
| leaf (layers 18–27) | 186 MiB | 1.2 GiB |

Every node needs the model: its own copy of the file, or, for a worker,
the parts it runs (below). Llama-family models (`llama`, `qwen2`, `qwen3`,
`qwen3moe`, `mistral`, `qwen2vl`, `qwen3vl`, `granite`), Phi-3 (`phi3`),
Qwen3.5 dense (`qwen35`, such as Ternary-Bonsai-2-27B), the Qwen3.5 mixture
(`qwen35moe`, such as Qwen3.6-35B-A3B), Qwen3-Next (`qwen3next`, such as
Qwen3-Coder-Next) and Gemma 4 can be split so far.

A worker does not need the whole model. On a machine without it, set

```ini
[workers]
download = range
```

and start it on the model as usual (`orangu-server --daemon` with `model`
set). It fetches only what building the model reads, and the layers a
parent assigns when it does — a third of a model and a little more in a
tree of three. It serves only as a worker: its own API stays off.

The small Gemma 4 models limit where they can be cut. Their last layers
reuse an earlier layer's KV cache, so they stay on the node that has that
layer. On `gemma-4-E2B` that keeps layers 13–34 together, so the other
nodes share the first 13:

```text
workers plan: alpha:8400 0..13, beta:8400 13..35
```

The 12B, the 31B and the 26B-A4B mixture share no KV and can be cut
anywhere. Every E2B, E4B and 12B quantization, the 31B and the 26B-A4B were
checked: split, each answers exactly what the unsplit model does.

## A tree of three machines

Three machines, `alpha` (clients talk to it), `beta` and `gamma`, each with
the same model in its models directory. `alpha` hands layers to `beta`, and
`beta` to `gamma`:

```text
   clients ──▶  alpha  ──▶  beta  ──▶  gamma
               layers       layers     layers
                0..9        9..18      18..28
```

On `gamma`, the leaf:

```ini
[workers]
host = all
port = 8400
secret = a-long-random-string
```

On `beta`, in the middle:

```ini
[workers]
host = all
port = 8400
workers = gamma:8400
secret = a-long-random-string
```

On `alpha`, the top:

```ini
[workers]
host = all
port = 8400
workers = beta:8400
secret = a-long-random-string
```

Start each one with the model, leaves first:
`orangu-server unsloth/Qwen3-32B-GGUF:Q4_K_M`. The top-level node logs
its plan:

```text
workers plan: alpha:8400 0..9, beta:8400 9..18, gamma:8400 18..28
```

A tree need not be deep. The same three machines as one top-level node
with two workers of its own:

```ini
# on beta and on gamma
[workers]
host = all
port = 8400
secret = a-long-random-string

# on alpha
[workers]
host = all
port = 8400
workers = beta:8400, gamma:8400
secret = a-long-random-string
```

`alpha` runs its share and hands the rest to `beta` and then `gamma`, in
the order listed; a prompt's parts are pipelined through both, so the two
workers work at the same time. On one board this shape read a 2041-token
prompt as fast as the chain above.

To check a tree once it is up:

- `curl alpha:8100/v1/workers` shows the plan, what its shares
  followed, each worker's connection and its link's round trip and
  bandwidth (`link`), and every node's processors, measured speed and
  what a forward costs it whatever its layers (`setup`, and `setups`
  under each worker); `curl beta:8100/ready` answers `503 serving a parent
  orangu-server`.
- A request's log line on `alpha` says where its time went:
  `workers: prefill 946 tokens in 8 forwards (alpha:8400 …, beta:8400 …)`,
  and a whole answer carries the same in `timings.workers` and a
  `Server-Timing` header.
- `orangu-bench --url http://alpha:8100` measures through the tree:
  `--pp 2048` a long prompt's prefill, `--gen 64` decode, `--streams 3`
  several requests at once. `--workers compare` runs the same options
  with `alpha` alone and then through the tree, side by side (`POST /props`
  switches it between the two).

Machines of different kinds make a tree too — the model check compares
the file, not the hardware — but their arithmetic rounds differently, so
a greedy answer can take a different near-tie from one server's, and a
`f32` tree is exact only against itself, not against another machine.

Machines of one kind round alike: the top-level node measures whether
prompts go through `bf16` or `int8` copies of the weights (`prompt_weights`)
and hands that choice to every worker with its layers, and each node copies
only the layers it runs. A worker whose cores lack the instruction keeps the
file's weights and rounds as those do. To compare a tree with one server
exactly, set `activations = f32` on the top-level node.

Shares follow speed, as far as memory holds. At start each node times the
model's first layers on the processor its layers run on — a decode step
and a 128-token prompt chunk — and reports that, with its CPU, GPUs and
NPU, to its parent. A parent gives every node (a worker counting with its
whole subtree) a share of the layers' bytes in proportion to its speed,
`shares = decode` by default or `shares = prompt` for a tree that mostly
reads long prompts. A node whose share would pass its budget keeps what
the budget holds and the rest goes to the others the same way. Each
node's budget for weights is 90% of its GPU memory, or 80% of its RAM
without one; the top-level node's is what its embedding and output head
leave. On one board with the top-level node on its Mali GPU and two
workers on its cores (Llama-3.2-3B Q4_K_M):

| `shares` | layers (GPU · CPU · CPU) | prompt, 512 tokens | decode |
|---|---|---|---|
| `memory` | 10 · 9 · 9 | 87.3 tok/s | 4.97 tok/s |
| `decode` | 15 · 6 · 7 | 64.1 tok/s | 6.28 tok/s |
| `prompt` | 7 · 12 · 9 | 110.9 tok/s | 4.51 tok/s |

With `shares = memory`, or when a node could not time itself (a
`download = range` worker has not got its layers yet), shares follow the
budgets alone. The layers are cut by their own sizes, not their count. A
share bigger than its node's budget is logged as a warning naming the
node:

```text
the workers are short of memory for this model — gamma:8400 gets 21.3 GiB
for 12.4 GiB; the layers over budget will be paged in from disk as they run
```

`local_layers = 0` makes a node a pure coordinator that runs no layer
itself.

A top-level node that holds the whole model and decodes faster than its
tree takes each sequence back once its prompt is through (`decode = auto`):
at the first decode step every worker sends its layers' rows back — and,
for a hybrid model (`qwen35`, `qwen35moe`, `qwen3next`), each recurrent
layer's state, a fixed size whatever the prompt's length — and the node
decodes alone on the model's own paths while the workers free theirs.
The prompt keeps the tree's speed and the answer the node's. On the board
above: decode 15.2 tok/s with the tree's 58.5 tok/s prompts, where
decoding through the tree gave 6.2. `decode = tree` keeps every step on the
tree, `decode = top` hands over whenever the node holds the model, and
`/v1/workers` says which applies. With `auto` the measured speeds decide
at first; once requests have decoded through the tree, the step they took
does, weighed again between requests, and the node runs the whole model
once before its first sequence decodes alone.

A node with workers does not have to use them. When it holds the whole
model and its measured speeds say the tree would not take at most 90% of
its time for a request — a 128-token prompt chunk at the tree's slowest
stage and an answer as long, its steps through every stage — it serves
alone and gives its workers nothing (`offload = auto`, the default;
`always` uses them regardless). The whole model alone is timed at start
for this, on the paths it would run alone; and once requests have run
through the tree, what they took goes before the prediction: between
requests, a tree that turns out not to pay gives way to the node alone. `/v1/workers` then says why
under `not_worth_offloading`, and every later plan weighs it again. On a
LAN of this board and two RK3588s with gemma-4-E2B, the node chose to
serve alone: 261 tok/s on a long prompt, where its tree gave 28. With
Qwen3.6-35B-A3B the prediction chose the tree; its first requests decoded
at 2.8 tok/s, where the node alone gives 4.5, and the node then served
alone.

When sequences decode through the tree, the output head — a matrix as wide
as the vocabulary, read once a token — can run on the node with the final
layer instead (`head = last`, or `auto` when that node measured the faster
decode): it sends logits back rather than its layers' output. That spares a
weak top-level node the model's largest read.

Once requests have run, a lead plans by the speed each node really showed —
its layers over its time a token — and plans again between requests when a
node runs more than 20% off what it was planned at; `/v1/workers` shows
both (`observed_gb_per_s`, `planned_gb_per_s`).

The decisions weigh the links too. A parent times each worker's link when
it connects — its round trip and bandwidth — so a fast machine on a slow
network is not mistaken for a fast part of the tree: handing a sequence
back to decode alone moves every prompt position's keys and values over
it, and moving the head moves a row of logits a token.

A node with a `[workers]` section reads and uploads a layer's weights the
first time the layer runs, not at startup. It skips:

- the probe that times decode on the CPU against the GPU;
- the host preload;
- the NPU's precompile of every block;
- the automatic split of a model too big for its GPU;
- the prefix warm-up.

It still chooses its GPU kernels as any server does. A choice an earlier
start kept in the probe cache is taken as it is; one that was not kept is
timed, which reads the whole model once. A node that fetches its model in
ranges (`download = range`) skips the timing, since reading the model is a
download there. When a new plan moves layers away from a node, it lets go
of their pages.

While `beta` and `gamma` work for their parent, their own APIs are off:
everything but `/health`, `/ready`, `/metrics` and `/v1/workers` answers
`503`. When the parent goes away, they serve their own clients again.

## Checks a parent makes

A worker is left out, with a warning naming it and the reason, when it:

- cannot be reached within `connect_timeout`;
- does not know the `secret`. The secret is never sent: each side proves
  it with an HMAC over fresh random values;
- has a **different file of the same model spec**. The parent compares,
  by content, the tensor types and shapes and samples of the weights of
  the layers it would run, so another release of a spec is refused
  whatever the file is called;
- already works for another parent, or would close a loop in the configs.

The tree then forms without it.

A worker assigned **another model** than the one it serves switches to the
lead's: it answers that it is switching, restarts itself on that model —
found under its `models`, or downloaded — and the lead takes it back at its
next plan. It keeps the model it had when the switch fails, or when its
`[web].reexec` is `no`, and is then left out. A lead that loads a new model
takes its workers with it the same way.

The lead finds out at the handshake: every worker says which model it
serves when it connects. One with another model that cannot switch — its
`[web].reexec` is `no` — is left out before any layers are planned, and the
lead says so once, naming both models:

```text
worker left out: beta:8400 serves bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M, not unsloth/gemma-4-E2B-it-GGUF:Q4_K_M, and cannot switch to it ([web].reexec is off there)
```

A node serving a picture model does not use its workers: its text encoder
is not split.

## Encryption

`secret` decides who may take part. To keep what travels between nodes —
prompts, as hidden states, and every generated token — private as well,
give the worker listeners a certificate:

```ini
[workers]
tls_cert = /etc/orangu/workers.pem
tls_key = /etc/orangu/workers.key
```

A node dials its workers with TLS when it trusts a certificate for them:
`tls_ca`, or its own `tls_cert` when that is not set. So the simplest
setup is **one certificate shared by every node**, naming every node's
address:

```sh
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -days 365 -subj /CN=orangu-workers \
  -addext "subjectAltName=DNS:alpha,DNS:beta,DNS:gamma,IP:192.168.1.10" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -keyout workers.key -out workers.pem
```

The certificate has to name the address its parent dials, as a DNS name
or an IP address. It also has to be a server certificate
(`CA:FALSE`): a certificate that is its own authority is refused as a
server's.

## Watching a tree

- `GET /v1/workers` on any node: its role, the layers it runs, the plan
  below it, each configured worker's connection, and any worker the plan
  lost.
- **Settings › Workers** in the web console: the same, as tables.
- `GET /ready`: `serving a parent orangu-server` on a worker;
  `a worker was lost` on a node whose plan lost one and has not repaired
  it yet.
- `GET /metrics` adds, per worker, the round trip of a forward
  (`orangu_server_worker_forward_seconds`), bytes each way, failures and
  whether it is up; per node, how often it planned and how often a lost
  worker made it plan again mid-request.
- When a request finishes, the top-level node logs where its time went:

  ```text
  workers: prefill 11 tokens in 1 forward (alpha:8400 12 ms, beta:8400 31 ms);
           decode 70 tokens in 70 forwards (alpha:8400 90 ms, beta:8400 610 ms)
  ```

  A worker's time is its round trip: the network, and its whole subtree.
  The client gets the same, prefill and decode together, as
  `timings.workers` in the answer and, on a whole answer, a
  `Server-Timing` header:

  ```text
  server-timing: prompt;dur=201.5, generate;dur=202.9,
    node0;desc="alpha:8400";dur=200.5, node1;desc="beta:8400";dur=168.5
  ```

## When a machine goes away

A request in flight is not lost with a worker. The top-level node plans
again over the workers that still answer, rebuilds the conversation on the
new plan by running its tokens through again, and carries on. Tokens
already sent stay sent. The rebuilt cache comes from one prefill rather
than token by token, so a greedy continuation can pick a different
near-tie from the one an uninterrupted run would have.

A spare machine makes that cheaper. List it as a standby on the node
above the worker it should cover:

```ini
[workers]
workers = alpha:8400, beta:8400
standby = spare:8400
```

The standby connects like a worker and waits. When `alpha` or `beta` is
lost mid-request, the standby takes its layers, and each conversation is
rebuilt there from the inputs the node sent into those layers, which it
keeps while a standby is configured. The other workers keep what they
have, the request goes on, and `/v1/workers` shows the standby standing in.
A worker that does not answer when the node plans is filled in for the
same way. When the lost worker is back, the node takes it back once it
is quiet, and the standby goes back to serving its own API.

A worker that comes back is taken back by the node above it within about
30 seconds, once no request has run for two seconds, never in the middle
of one. A conversation that continues after that is replayed through the
new plan once, and reuses its rows from then on.

## Machines that clock down

A node works in bursts: its layers of a token, then nothing while the
nodes after it run theirs. A CPU frequency governor that sees cores busy
half the time clocks them down, and every layer then runs slower. On the
development board, a two-node tree held to separate cores decoded at 163 ms
a token that way, against 99 ms for one server.

On a machine that serves only as a node, set the CPU governor to
`performance`. Without root, `ORANGU_WORKERS_CLOCK=1` holds the node's cores
at full clock while a request runs, with one idle-priority spinner per
core. The spinners yield to any real work, and stop about 300 ms after the
node's last forward. With them, the same tree decoded at 103 ms a token.

**Do not turn it on for nodes that share a machine's cores.** Each node's
spinners then take the cores another node computes on: three nodes on one
board, each using every core, decoded at 237 ms a token with it and 118 ms
without. Nodes like that keep each other's cores busy anyway.

## Limits

- A slot's next turn reuses its conversation's rows on the workers, and a
  new conversation copies a prompt prefix another slot's conversation has
  in common, on every worker. The prefix cache, paged KV, MTP heads and
  slot save/restore are off on a top-level node with workers.
- One request runs through the machines one after another, so a single
  request is not faster than on one machine. What a tree adds is
  concurrency: with several `slots`, each machine works on a different
  request at the same time. And a long prompt is pipelined: while one
  machine works on a chunk of it, the one before it already works on the
  next chunk. The top cuts the prompt into 128-token parts for this
  (`ORANGU_WORKERS_CHUNK`). `ORANGU_WORKERS_PIPELINE=0` turns the
  pipelining off.
  On one board, a pipelined tree of three 4-thread nodes prefilled a
  2041-token prompt 2.2× faster than without pipelining.
  One decode step of every slot goes as a single message per worker
  (`ORANGU_DECODE_BATCH=0` sends them one by one).
- On a GPU, a node runs its layers of a decode step as one recorded
  submission, as a whole model does: a Llama-3.2-3B step split in three
  took 67.8 ms against 65.5 ms whole. A prompt chunk still runs layer by
  layer, and so does a Gemma 4 mixture's decode.
- Each node keeps the whole model file on disk. Fetching only a node's
  own layers is still to come.
- A GPU keeps the layers it has uploaded until the process ends. A new plan
  frees their host memory, but not their device memory.
- Before a parent claims it, a node serves its own clients with the whole
  model, which reads every layer in. The next plan lets go of those
  pages.

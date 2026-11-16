\newpage

# Workers: one model over several machines

Several machines on one network can serve as one. Give each
`orangu-server` the same `[workers].name` — a **channel** — and they find
each other on their own, then share the work two ways:

- A request goes to the **least busy machine that holds the model**. Each
  node reports how busy it is, from `0.0` (idle) to `1.0` (fully busy),
  and a busy node hands a request to a less busy one.
- A model **too large for any one machine** is split over several. The
  node a request reaches claims the free nodes of its channel as its
  workers: it runs some of the model's layers and hands the rest to them.
  For every prompt chunk and every generated token, the hidden states
  travel to each worker and back, and the node that claimed them — the
  top-level node — samples, checks stop conditions and streams the reply,
  exactly as a single server does.

This chapter is the practical side: setting up a channel, securing it, and
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

## A channel of three machines

Three machines, `alpha`, `beta` and `gamma`, on one network, each with the
same model in its models directory. On every one of them:

```ini
[workers]
host = all
port = 8400
name = lab
secret = a-long-random-string
```

Every node of a channel listens on the same `port`. Start each one with the
model: `orangu-server unsloth/Qwen3-32B-GGUF:Q4_K_M`. Every minute a node
scans its IPv4 networks — every address of each, or the /22 around its own
address on a wider network — on that port, and asks whoever answers whether
it is on `lab`. The name itself never crosses the network, only a hash of
it, and a node on another channel does not answer. Each node logs what it
found:

```text
workers: channel 'lab' has beta:8400 (192.168.1.11:8400, idle), gamma:8400 (192.168.1.12:8400, idle)
```

Between scans it asks the nodes it found how busy they are every two
seconds.

**Every node of a channel needs the same model.** A probe's answer names
the model a node serves — its layout and quantization, and samples of every
layer's weights — and a node works only with nodes of its own model: a
request is routed only to a node whose model matches in both, and only a
node with the same layout is claimed as a worker, its weights compared
layer by layer when it is. A node with another model stays on the channel
unused, and the log and `/v1/workers` (`same_model`) say so:

```text
workers: channel 'lab' has delta:8400 (192.168.1.13:8400, idle, another model: not used)
```

**When each machine holds the model**, clients can talk to any of them. A
node keeps a request unless another node that holds the model is at least
0.1 less busy, and then sends it there and streams the answer back. A
request is routed once at most. A node's load is the larger of its slots in
use and how busy its processors are — the CPU, and a GPU where the driver
reports its busy time — so a machine busy with something else counts as
busy too. The requests routed are the ones that run the model:
`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/completion`,
`/infill`, `/embedding` and `/v1/embeddings`. The API key, if any, is
checked first and passed on, so every node of a channel needs the same.

**When no machine holds the model**, the first node to get a request claims
every free node it found whose load is at most 0.8, plans the model's layers
over itself and them, and serves the request:

```text
   clients ──▶  alpha  ──▶  beta
               layers  ──▶  gamma
                0..9        9..18, 18..28
```

```text
workers: claiming beta:8400 (192.168.1.11:8400), gamma:8400 (192.168.1.12:8400) on channel 'lab'
workers plan: alpha:8400 0..9, beta:8400 9..18, gamma:8400 18..28
```

A node that holds the model never claims workers: it routes instead. A tree
is one level deep: a node that leads workers of its own is not claimed by
another, and a worker never claims any. Two nodes that get a request at the
same moment can split the free nodes between them. When `release` seconds
(`300` by default) pass with no request running on the top-level node, it
lets its workers go, and they are free for the next node with a request.

`alpha` runs its share and hands the rest to `beta` and then `gamma`, in
the order of their names; a prompt's parts are pipelined through both, so
the two workers work at the same time.

To check a tree once it is up:

- `curl alpha:8100/v1/workers` shows the channel and the nodes it found,
  each with its load, the plan, what its shares
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
file's weights and rounds as those do. To compare a tree with one server,
set `activations = f32` on the top-level node and `ORANGU_SPECULATIVE=0` on
both, since a verify pass computes its rows as a prompt does and runs only
while it pays. On Llama-3.2-3B Q4_K_M that matched exactly for prompts up to
62 tokens, but not at 112 or more.

Shares follow speed, as far as memory holds, and load: each node's speed
counts at `1 − load`, never under 5% of it, so a busy machine gets fewer
layers. At start each node times the
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

When sequences decode through the tree, the output head — a matrix as wide
as the vocabulary, read once a token — can run on the node with the final
layer instead (`head = last`, or `auto` when that node measured the faster
decode): it sends logits back rather than its layers' output. That spares a
weak top-level node the model's largest read.

Once requests have run, a lead plans by the speed each node really showed —
its layers over its time a token — and plans again between requests when a
node runs more than 20% off what it was planned at; `/v1/workers` shows
both (`observed_gb_per_s`, `planned_gb_per_s`). A new plan has to save more
than 5% of a decode step or of the slowest stage. Otherwise the plan stays, with
its conversations, because a worker given other layers has to build and warm
them, which takes a request behind it about a minute on a board with a large
model. Workers of about the same speed stay in the order they are in, and start
in the order of their names, so restarting a node does not move layers.

The decisions weigh the links too. A parent times each worker's link when
it connects — its round trip and bandwidth — so a fast machine on a slow
network is not mistaken for a fast part of the tree: moving the head
moves a row of logits a token over it.

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

While `beta` and `gamma` work for `alpha`, a request that runs the model
and reaches one of them is sent on to `alpha`; everything else but
`/health`, `/ready`, `/metrics` and `/v1/workers` answers `503`. When
`alpha` lets them go, or goes away, they serve their own clients again.

## Checks a parent makes

A worker is left out, with a warning naming it and the reason, when it:

- cannot be reached within `connect_timeout`;
- does not know the `secret`. The secret is never sent: each side proves
  it with an HMAC over fresh random values;
- has a **different file of the same model spec**. The parent compares,
  by content, the tensor types and shapes and samples of the weights of
  the layers it would run, so another release of a spec is refused
  whatever the file is called;
- already works for another node, or leads workers of its own.

The tree then forms without it. A node found on the channel only becomes a
candidate: being found grants nothing, and the handshake still asks for the
`secret`.

Without a `secret`, anyone who reaches a worker's port can take part — and a
channel name is no secret, since any node that asks with it is answered — so
set one on any network you do not fully trust. During the handshake a peer can only
send frames of up to 1 MiB, and is closed if it sends a larger one.

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

A node dials its workers, and probes its channel, with TLS when it trusts
a certificate for them: `tls_ca`, or its own `tls_cert` when that is not
set. So the simplest setup is **one certificate shared by every node**.
A channel dials nodes by IP address, so the certificate names every node's
address:

```sh
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -days 365 -subj /CN=orangu-workers \
  -addext "subjectAltName=IP:192.168.1.10,IP:192.168.1.11,IP:192.168.1.12" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -keyout workers.key -out workers.pem
```

The certificate has to name the address its parent dials, as an IP
address. It also has to be a server certificate
(`CA:FALSE`): a certificate that is its own authority is refused as a
server's.

## Watching a tree

- `GET /v1/workers` on any node: its channel, the nodes it found with
  their state and load, its own load, its role, the layers it runs, the
  plan below it, each worker's connection and failures in a row, how long
  those leave it out, and any worker the plan lost.
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

A machine that loses power or its network closes nothing, so while a request
is out a node pings the worker below it. A worker that says nothing for 15
seconds counts as lost. A worker that is only slow goes on answering, and has
until `timeout` for a decode step, and longer for a prompt's forward.

A request in flight is not lost with a worker. The top-level node plans
again over the workers that still answer, rebuilds the conversation on the
new plan by running its tokens through again, and carries on. Tokens
already sent stay sent. The rebuilt cache comes from one prefill rather
than token by token, so a greedy continuation can pick a different
near-tie from the one an uninterrupted run would have. This is not the
tree's doing: a prompt's kernels and a decode step's quantize activations
differently, so one server does the same when it reuses a retained prompt.
On Llama-3.2-3B Q4_K_M the two differ by up to 0.9 in log-probability among
a token's 20 likeliest alternatives. A rebuilt conversation holds what one
prefill of its tokens holds on a single server.

A worker that comes back is taken back by the node above it within about
30 seconds, once no request has run for two seconds, never in the middle
of one. A conversation that continues after that is replayed through the
new plan once, and reuses its rows from then on.

A worker does the same for its parent. A parent pings every 5 seconds while
idle and more often with a request out, so a worker that hears nothing for 60
seconds takes it for gone: it lets go of its layers and sessions, serves its own
API again and can be claimed by another node. A parent that was only stalled that long finds the worker
lost, plans again, and takes it back as usual.

A worker that keeps failing is left out for a while, so it does not cost every
request a new plan and a replay. Two failures in a row leave it out for 2
minutes, and each further failure doubles that, up to 30 minutes. A request
served start to finish without a replay forgives the workers it used, and so
does an hour without a failure. A node below the top sees no request finish, so
only the hour forgives there.

A failure is a connection lost with a request out, or a worker that takes the
connection and fails the handshake. A worker that cannot be reached, or one
that refuses (it serves another parent, or its secret differs), is no failure,
and a machine that restarts comes back as usual. While a worker is left out,
the node plans without it: the others take its layers, and any over their budgets are paged in from disk. A
worker you have repaired stays out until its wait ends, or until you restart
the node above it. `/v1/workers` shows each worker's `failures_in_a_row`
and `left_out_seconds`.

## When a prompt does not fit

A node's KV cache grows as a prompt runs, so a prompt too long for a worker's
memory would run until the kernel killed that worker. Before a prompt starts,
the top-level node asks every node for room for all of it, plus one answer
token. Each node compares what its own layers need with its free memory, less
5% and less what it has promised prompts still running.

If a node has too little free, it first lets go of conversations nobody has used
for 30 seconds, oldest first; one taken up later is replayed. If that is not
enough, the client gets a `503` naming the node and how much it is short:

```text
OutOfMemory: the prompt's KV cache needs 272.70 MiB here, with 137.40 MiB free
and 0 B promised to other sequences: send a shorter prompt, or give this node
more memory (at beta:8400)
```

Nothing has run by then, no worker counts as failed, and the plan stands.
`/v1/workers` shows `kv_refusals` for the tree and `kv_promised_bytes` for this
node's own layers (a worker's promises do not show on its parent), and
`/metrics` shows `orangu_server_workers_kv_refusals_total`.

Free memory takes in page cache, as the kernel can take it back, and a cgroup v2
limit where there is one, so this holds in a container. Some limits:

- `max_tokens` is a cap and not a promise, so a long answer can still run a node
  out of memory.
- Only a worker that lists `room` under `features` in `/v1/workers` is asked. An
  older one is sent the prompt unasked, as before.
- A node with a GPU counts its device copy against host memory. A card with its
  own memory is held to host memory too, so it can refuse early, and the card's
  own room is not checked on a worker.

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
  (`ORANGU_WORKERS_CHUNK`). A node keeps two parts per stage below it and
  holds the next one back until there is room (`ORANGU_WORKERS_WINDOW`; `0`
  is no bound). `ORANGU_WORKERS_PIPELINE=0` turns the pipelining off.
  On one board, a pipelined tree of three 4-thread nodes prefilled a
  2041-token prompt 2.2× faster than without pipelining.
  One decode step of every slot goes as a single message per worker
  (`ORANGU_DECODE_BATCH=0` sends them one by one).
- On a GPU, a node runs its layers of a decode step as one recorded
  submission, as a whole model does: a Llama-3.2-3B step split in three
  took 67.8 ms against 65.5 ms whole. A prompt chunk still runs layer by
  layer, and so does a Gemma 4 mixture's decode.
- A node with `download = range` fetches just its layers, and so only ever
  works for another node: it never routes, leads or serves alone. Every
  other node keeps the whole model file on disk. A model split in shards is
  refused.
- A channel is found over IPv4 only, on networks this machine has an
  address on. Nodes behind a router are not found.
- A GPU keeps the layers it has uploaded until the process ends. A new plan
  frees their host memory, but not their device memory.
- Before another node claims it, a node serves its own clients with the
  whole model, which reads every layer in. The next plan lets go of those
  pages.

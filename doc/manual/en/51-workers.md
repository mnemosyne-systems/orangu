\newpage

# Workers: one model over several machines

Several machines on one network can serve as one. Give each
`orangu-server` the same `[workers].name` — a **channel** — and they find
each other on their own, then share the work two ways:

- A request goes to the machine that serves it **fastest**. A node that
  holds the model alone serves it alone, or sends it to another node that
  holds it and decodes it faster for how busy it is. Each node reports how
  busy it is, from `0.0` (idle) to `1.0` (fully busy).
- The model is **divided over the free machines** when no machine holds it
  alone, or when the speeds the machines measured predict that divided it
  decodes faster than the fastest machine alone (see **Alone or
  divided**). The node a request reaches claims the free nodes of its
  channel as its workers: it runs some of the model's layers and hands the
  rest to them, each node's share sized by its speed and memory.
  For every prompt chunk and every generated token, the hidden states
  travel to each worker and back, and the node that claimed them — the
  top-level node — samples, checks stop conditions and streams the reply,
  exactly as a single server does. A request reaching a node that works
  for another is sent on to that node.
- A channel can serve **several models**. Nodes work together only on the
  same model; a request whose `model` field names another model that a node
  of the channel serves is sent to that node.

This chapter is the practical side: setting up a channel, securing it, and
watching it. The *Inference server* chapter's **The `[workers]` section**
has the reference for every key.

## When it helps

Splitting a model is about **room, not speed**. Every layer still runs once
per token, one machine after another, and each hand-over adds a network
round trip, so a model that fits on one machine is fastest there. A tree
is for a model that does not fit, or for machines that are each too small
to work together. The nodes decide this themselves from what they measure
(see **Alone or divided**). Keep them close: on a LAN a round trip is a
fraction of a millisecond, and across the internet it is tens of
milliseconds for every token.

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

It fetches from the other nodes of its channel before Hugging Face, so a
channel with no internet needs the model on one node's disk only. Every
node that holds its whole model in a single file serves byte ranges of
it on its `[workers]` port, to any node that proves the channel's
`secret` (over TLS when the channel uses it), and says so when probed. A
fetching node asks, in order:

1. the node that claimed it, for the layers that node assigns;
2. the other nodes whose probe names its model (`label` and quantization,
   with the whole file), least busy first;
3. Hugging Face.

At start, with a node of the channel serving its model, it fetches the
header from that node and needs no Hugging Face at all. With none yet,
and Hugging Face out of reach, it looks again every few seconds for ten
minutes before it gives up.

Nothing it is sent runs unchecked. The header must have the layout the
serving node's probe named; every layer's vectors, and samples of every
layer's weights, must hash to the probe's content hash before the model is
built; and each range a parent assigns must match the parent's own weights
before it runs. Bytes that fail are fetched again from the next source, and
the source that sent them is not asked again. Every source must also serve
the header already in place, byte for byte, before any of its weights are
taken.

A node serves ranges at full speed while it runs no request. While it
serves requests, or leads a tree with requests in flight, all of its
fetching nodes together get 8 MiB a second, so a download does not take the
link and the disk from the requests. While it waits for a worker to take the
layers it assigned, it serves at full speed: its requests are waiting for
that worker. A node that fetches in part serves
nothing itself.

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
model: `orangu-server unsloth/Qwen3-32B-GGUF:Q4_K_M`. At start and every
`scan` seconds (10 minutes by default) a node scans its IPv4 networks — every address of each, or the /22 around its own
address on a wider network — on that port, and asks whoever answers whether
it is on `lab`. The name itself never crosses the network, only a hash of
it keyed by the `secret`, and a node on another channel, or with another
secret, does not answer. Each node logs what it found:

```text
workers: channel 'lab' has beta:8400 (192.168.1.11:8400, idle), gamma:8400 (192.168.1.12:8400, idle)
```

Between scans it asks the nodes it found how busy they are every two
seconds. A node that starts later is found sooner than the next scan: the
nodes its own first scan probes probe it back within seconds.

A machine on two networks — Wi-Fi and a cable, say — answers on both, and
is still one node: it is reached at the address that answered faster, and
the other is kept as its alternate (`alternates` in `/v1/workers`, with
each probe's time as `probe_ms`). It moves to another address only when
that one answers at least twice as fast, or when its own stops answering;
a worker whose link went down is then claimed again at the address that
answers.

**Nodes work together only on the same model.** A probe's answer names
the model a node serves — its layout and quantization, and samples of every
layer's weights — and a node works only with nodes of its own model: a
request is routed only to a node whose model matches in both, and only a
node with the same layout is claimed as a worker, its weights compared
layer by layer when it is. A node with another model is never claimed, and
the log and `/v1/workers` (`same_model`) say so; it serves its own model,
and the requests that name it (see **Several models on one channel**):

```text
workers: channel 'lab' has delta:8400 (192.168.1.13:8400, idle, another model: not used)
```

**When each machine holds the model**, clients can talk to any of them. A
node keeps a request unless another node that holds the model, and is free,
serves it faster: each node's time a token for its model alone, from the
decode speed it measured, over what its load leaves free (`1 − load`, never
under 5%). The other node must come out at most 0.9 of this one, and then
the request is sent there and the answer streamed back. So a slow board
that gets a request sends it to an idle laptop that decodes the model
faster, and keeps it only when the laptop is so busy that it would be
slower. A
node that measured no speed, or finds another that holds the model and did
not, goes by load alone: it keeps a request unless another such node is at
least 0.1 less busy. A request is routed once at most. A node's load is the
larger of its slots in use and how busy its processors are — the CPU, and a
GPU where the driver reports its busy time — so a machine busy with
something else counts as busy too. The requests routed are the ones that run the model:
`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/completion`,
`/infill`, `/embedding` and `/v1/embeddings`. The API key, if any, is
checked first and passed on, so every node of a channel needs the same.
A chat turn in the web console is routed the same way: its prompt goes to
the other node as tokens, and the answer streams back into the console
of the node it was typed on. A turn with pictures attached is served
where it was typed.

**When no machine holds the model**, or dividing it is predicted faster than
any machine alone, the first node to get a request claims every free node it
found whose load is at most 0.8, plans the model's layers over itself and
them, and serves the request:

```text
   clients ──▶  alpha  ──▶  beta
               layers  ──▶  gamma
                0..9        9..18, 18..28
```

```text
workers: claiming beta:8400 (192.168.1.11:8400), gamma:8400 (192.168.1.12:8400) on channel 'lab'
workers plan: alpha:8400 0..9, beta:8400 9..18, gamma:8400 18..28
```

A node that holds the model claims workers only when dividing pays. A tree
is one level deep: a node that leads workers of its own is not claimed by
another, and a worker never claims any. When `release` seconds
(`300` by default) pass with no request running on the top-level node, it
lets its workers go, and they are free for the next node with a request.

A channel forms one tree for its model:

- **A node that already leads one serves the others.** A node that cannot
  hold the model, and finds no free node that can, sends its request to a
  node that leads a tree of the model and is not fully busy, instead of
  claiming what is left. A node that holds the model sends it there only
  when that tree is predicted at most 0.9 of its own time alone.
- **Two claims at once have one winner.** Before it claims, a node asks
  each free node to promise itself to it for 10 seconds. A free node is
  promised to the lowest node id that asks in that time, the asking nodes
  included, so two nodes that get a request at the same moment agree on
  the winner; the other sends its request there:

  ```text
  workers: another node claims channel 'lab' at the same time; this request goes to it (3f0c9a17d2b46e81)
  ```

  A node that already leads one of them wins, whatever its id.
- **The smaller of two trees gives way.** If two trees of the model form
  anyway, the leader with fewer workers (or as many and the higher id)
  lets its workers go as soon as it has no request running, rather than
  after `release`; its next request goes to the larger tree.

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

The output head — a matrix as wide as the vocabulary, read once a token —
can run on the node with the final layer instead (`head = last`, or `auto` when that node measured the faster
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
- the prefix warm-up: a top-level node's prefix cache fills from the
  requests it serves.

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

## Alone or divided

A tree runs every layer once a token, one node after another, and each
hand-over adds a round trip, so it decodes faster than a node alone only
when its fast nodes cannot hold the model by themselves — a GPU with room
for half of it beside a board that holds all of it. The node a request
reaches weighs the two before it claims anyone, from what each node
measured at start (`setup` in `/v1/workers`):

- **Alone**: every layer and the output head at the node's decode speed
  (`decode_gb_per_s`), and what a step costs it whatever its layers
  (`decode_fixed_ms`). Every node says its own in its answer to a probe, as
  `step_ms`; a node that leads a tree says the tree's.
- **Divided**: a tree over this node and the free nodes, cut as a plan cuts
  it — by decode speed, up to each node's memory, in the order that runs
  soonest — then each node's layers at its speed and its fixed cost, the
  head on this node, and for every worker a round trip and the hidden state
  each way. A free node's round trip is its probe's time, halved, since the
  probe dials too; a worker this node already holds counts with its timed
  `link`. Once requests have run through a tree, the speed each node showed
  there counts instead, and for a tree of the same nodes, the time its
  decode steps took.

Each side is weighed by load, as shares are: a node's time over `1 − load`,
never under 5%. The request is divided only when the tree comes out at most
0.9 of the fastest node alone; otherwise it goes to that node. A tree
another node leads takes the request on the same terms, by its `step_ms`.

What cannot be priced is not divided over:

- a node that holds the model with no measured speed, or beside another
  that holds it and has none — with `shares = memory` no node times
  itself: the request is served alone, by load as above;
- a tree with a free node that measured nothing, or does not send its
  speed: served alone.

A model no node holds alone is always divided: the request goes to a node
that leads a tree of it, else over the free nodes.

A node that leads a tree of a model it holds alone keeps checking: when the
decode steps of 32 tokens through the tree took longer than it predicts for
itself alone, it lets its workers go between requests and serves alone, and
the measured tree then keeps it from dividing again. While it leads, it
takes a free node found later only when that is predicted to make the step
at least 10% faster, or when its workers lack the room for the model.

The log says how requests are served whenever that changes:

```text
workers: requests are served here alone: predicted 13.2 ms a token on this node alone, 94.1 ms divided
workers: requests are sent to laptop:8400: predicted 13.2 ms a token on laptop:8400 alone, 94.1 ms divided
```

and `/v1/workers` has `divide`: `alone_ms`, this node's time a token alone;
`step_ms`, what its probe answer says; `observed_tree`, the nodes of the
last tree, its step in milliseconds and the tokens it was seen over; and
`last`, the last decision — `verdict` (`here`, `routed` or `divided`), `to`,
`reason`, `alone_ms` (the best node alone, by its load) and `tree_ms`. Each
node under `found` shows the `decode_gb_per_s` and `step_ms` its probe
said.

## Several models on one channel

One channel can hold nodes with different models. Each node's answer to a
probe carries its model's name as its `/v1/models` lists it, and its
quantization; `/v1/workers` shows them under each found node as
`model_name` and `model_quant`. A node learns what another serves only from
that answer: a node that does not send a name is never chosen by one.

Each model forms its own tree. A node divides its model over the free nodes
with the same model only, and a node with another model is never claimed.

A request's `model` field picks the model. When a request that runs the
model names one that this node does not serve and another node of the
channel does, it is sent there and its answer streamed back, with
`x-orangu-routed-to` naming that node. A name matches a node's label in any
case, with or without its `:QUANT` tag: `unsloth/gemma-4-E2B-it-GGUF`,
`unsloth/gemma-4-e2b-it-gguf` and `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` all
name a node serving that label at `Q4_K_M`. Of the nodes that serve it, the
request goes to the least busy one leading a tree of it that is not fully
busy, else the least busy free one, else the least busy of them. A node
working for another, or one with only part of a model, is never chosen. The
node it reaches serves it as it serves any request: it may route it on to a
node of its model that holds it, or divide its model over the free nodes of
its model.

A request that names this node's own model, names none, or names a model no
node of the channel serves is served as it would be without the field, so a
client that sends any name, such as one of another provider's models, keeps
working. A request is routed once at most.

`GET /v1/models` lists this node's model first, then each other model the
channel serves, once per name and quantization. Each of those has
`owned_by: "orangu-server channel"`, its `quant`, and `served_by`, the
`[workers]` names of the nodes serving it; its `id` is the name a request
uses, with the quantization's tag. Only a node that holds a whole model and
serves its own clients is listed, as the last scan found it.

```text
$ curl -s alpha:8100/v1/models | jq -r '.data[] | "\(.id)  \(.owned_by)"'
unsloth/gemma-4-E2B-it-GGUF            orangu-server
unsloth/Qwen3-8B-GGUF:Q4_K_M           orangu-server channel
```

A chat turn in the web console names no model: it is served by the
console's own model, routed as described above.

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

Without a `secret`, anyone who reaches a worker's port can take part, and
anyone who knows or guesses the channel name is answered with a node's
state, load, API address and model. With one, the secret keys the hash a
probe carries, so a node answers only probes from nodes that share it; set
one on any network you do not fully trust. During the handshake a peer can only
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
  their state and load, its own load, its role, on a worker its `parent`
  (id, name, whether it is suspect, probes it missed in a row and seconds
  since its connection last carried anything), the layers it runs, the
  plan below it, each worker's connection and failures in a row, how long
  those leave it out, the extensions it advertised under `features`
  (`fork`, `head`, `echo`, `switch`, `room`, `metrics`, `retain`,
  `prefixes`, `rows`), and any
  worker the plan lost. Each worker's `sessions` shows what its handshake
  reported — its `slots`, `n_ctx` and `pool_reserve` — with the
  `session_limit` it holds for this node and the `retained_sessions` this
  node's prefix cache keeps there. A top-level node adds `prefix_cache`:
  the entries its prefix cache holds, its `hits` and `misses`, and
  `tokens_held`. A worker adds `retained_sessions`: how many sessions it keeps for its
  parent's prefix cache, and their bytes. Each entry of `workers` also
  carries the worker's `name` as plans print it, its `link` as timed after
  the handshake (`rtt_ms`, the round trip, and `mb_per_s`, the bandwidth
  each way) and `tls`, whether the link is encrypted.
- **Settings › Workers** in the web console: the same, as a graph above
  the tables. The graph draws this node, the parent it works for, each
  worker it claimed with the nodes below that worker, its standbys, and
  under *Elsewhere on the channel* every other node the scan found:

  | On the graph | Shows |
  |---|---|
  | Node colour | its state: idle, leading workers, working for another node, worker only, or another model (never claimed) |
  | Node size | its load: a busier node is drawn larger |
  | Under a node | its name, the layers the plan gives it, and its load |
  | Outlined node | this node |
  | Line to a worker | a claimed link, labelled with its round trip, bandwidth and `TLS` or `plain TCP`; dashed when the worker is unreachable, dotted for a standby |
  | Ring around a node | the node the last console turn ran on: this one, or the node of the channel it was routed to (the address an API answer names in `x-orangu-routed-to`) |
  | Time above a node | the last console turn's `timings.workers` time there; the lines it crossed are highlighted |

  Hovering a node lists the same facts. The caption under the graph names
  where the last console turn ran and every stage time, including any
  for a node the graph does not draw. The overlay follows each console
  turn; *Refresh* reads the channel again. Below the graph, the tables
  list the channel with this node's load, every node found with its
  address and alternates, state, load and whether it serves the same
  model, then the plan, the workers and their hardware.
- `GET /ready`: `serving a parent orangu-server` on a worker, or
  `serving a parent orangu-server that does not answer` while its parent is
  suspect; `a worker was lost` on a node whose plan lost one and has not repaired
  it yet.
- `GET /metrics` adds, per worker, the round trip of a forward
  (`orangu_server_worker_forward_seconds`), bytes each way, failures and
  whether it is up; per node, how often it planned and how often a lost
  worker made it plan again mid-request. A top-level node adds its prefix
  cache's `orangu_server_workers_prefix_cache_hits_total`,
  `orangu_server_workers_prefix_cache_misses_total` and
  `orangu_server_workers_prefix_cache_entries`.
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

**A free node takes a lost worker's place.** Every free node takes a share
when the tree forms, so none waits in reserve. When a worker is lost, the
plan made after the loss gives its layers to a free node of the channel when
there is one — a node that joined since, or was too busy to claim — rather
than spreading them over the nodes left. The log says so (`claiming … to
stand in for a lost worker`), and `/v1/workers` lists it under `standby`,
with the worker it stands in for.

With no free node, a request in flight is still not lost with a worker. The
top-level node plans again over the workers that still answer, rebuilds the conversation on the
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

A worker does the same for its parent. A parent whose process ends closes
its connections, and the worker lets go at once. One whose machine loses power
or its network closes nothing; the worker then takes it for gone on the first
of two signs:

- **The channel stops finding it.** A worker on a channel asks the nodes there
  how busy they are every 2 seconds, its parent included. A parent that leaves
  3 of those in a row unanswered, while nothing has come over its connection
  for 15 seconds, is gone: about 15 seconds after it went.
- **Its connection stays silent.** A parent pings every 5 seconds while idle
  and more often with a request out, so a worker that hears nothing for 60
  seconds takes it for gone. This is the only sign for a parent the channel
  never found, such as one that names the worker in `[workers].workers`.

Either way the worker lets go of its layers and sessions, serves its own API
again and can be claimed by another node. A parent that was only stalled that
long finds the worker lost, plans again, and takes it back as usual.

A parent that misses one of those probes, or whose API a request sent on to it
could not reach, is *suspect*: the worker asks the channel about it again at
once, `/v1/workers` shows it under `parent` with `"suspect": true`, and
`/ready` says `serving a parent orangu-server that does not answer`. The next
answer from the parent clears it.

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

## Where a long prompt runs

A request served by one node alone runs its prompt there, like any server.
In a tree, a prompt goes through the tree part by part, every node running
its own layers. When a node of the plan holds the whole model — memory for every
layer, `holds_model` in `/v1/workers` — and reads prompts much faster than
the tree, a long prompt runs there instead: that node runs every layer of
the prompt alone, each node of the tree is then given the rows of its own
layers, and the answer is decoded through the tree as usual. The node is the
top-level node itself or one of the workers directly below it.

The top-level node decides for every prompt, from what was measured:

- **Through the tree**: each node's prompt speed and fixed cost (`setup`:
  `prompt_gb_per_s` and `prompt_fixed_ms`), the hidden state over each link,
  and the parts overlapping once the pipeline is full. Once a prompt of at
  least 256 tokens has run through the current plan, the speed it ran at
  counts instead.
- **On one node**: every layer at that node's prompt speed, then the rows of
  every layer it does not run sent to the node that does, at each link's
  bandwidth (`link`), and taken in there. Once a prompt has run there, the
  speed the whole hand-over ran at counts instead.

The prompt goes to the node only when that comes out at three quarters of
the tree's time or less; otherwise, and whenever a side cannot be priced, it
stays on the tree:

- prompts under 256 tokens;
- a node of the plan that has no measured prompt speed (with `shares =
  memory` no node times itself), or a link that was not timed;
- a worker that does not list `rows` under `features`;
- a model that keeps a recurrent state — Qwen3.5 and Qwen3-Next: a state is
  of every position it saw, and stays where it was computed.

A prompt that starts like a cached conversation reuses its rows on the tree
and runs only what is new. The node that runs a prompt alone holds no rows
of that conversation, so it runs all of it from the start, and every node's
rows of it are replaced; whether it goes there is decided by the length of
the new part.

The log line says where a prompt ran:

```text
workers: a prompt of 2765 tokens ran on gamma:8400 alone, every node given its
rows, in 4330 ms, 910 ms of it the rows (predicted 4.1 s there, 52.8 s through
the tree)
```

and `/v1/workers` on the top-level node has `prefill`: how many prompts ran
on one node (`on_node`), through the tree (`on_tree`) and on the tree after a
hand-over failed (`fell_back`), the speeds observed (`observed_tokens_per_s`,
under `tree` and each node's name), and the last prompt (`last`): where it
ran, its tokens, how long it took and how much of that the rows took
(`rows_ms`), the two predictions and the reason.

A hand-over that fails — the node goes away, a node has no room for the
rows, or refuses them — releases what it opened on every node, and the
prompt goes through the tree; the answer is the tree's. Every node checks the
rows it is given as it checks a saved prefix: the model, the layers, the
shape of its cache, the context and the positions.

The rows are computed in the parts the tree cuts a prompt into and travel
in `f32`, so on machines of one kind they are the rows the tree computes but
for the hidden state between nodes, which the tree sends in `activations`.
A node of another kind — a GPU beside boards that compute on their cores —
rounds as it does. A node runs the layers outside its share from the file's
weights, not from `prompt_weights` copies. While it hands the rows over, the
node holds every layer's rows of the prompt.

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

## The prefix cache on a tree

A top-level node with workers keeps its last four finished conversations
in a prefix cache, as a single server does with `ORANGU_PREFIX_CACHE=1`. It
is on by default there; `ORANGU_PREFIX_CACHE=0` turns it off. A new prompt
that starts like one of them takes that conversation's rows over and
computes only what is new, whichever slot it runs in.

Each entry keeps a session of its own on every worker: a copy of the
conversation's rows, made as the request ends, while the slot keeps the
original for its next turn. A worker keeps one such session per slot of its
own `slots`, up to four, beside the two sessions each of the top-level
node's slots takes, and says so in its handshake, together with its own
`slots` and context; a node in the middle of a tree reports the fewest of
itself and the workers below it. The top-level node holds no more entries
than the fewest any worker reported, and sends a worker no copy it did not
say it keeps. A worker that does not list `retain` under `features`, or
does not report its configuration, keeps none, and the cache stays empty.

A worker that lists `retain` keeps an entry's session until the top-level
node drops the entry, past its own idle and memory eviction.

The top-level node decides what goes. When a worker has no room for a
prompt, the node drops its oldest entry, which frees the entry's session on
every worker, and asks again, until the cache is empty; only then is the
prompt refused. When the workers cannot make a copy, the conversation is not
added to the cache, and its slot still keeps it.

A worker that decodes on its GPU keeps the reply's rows of some layers — a
sliding window's — on the device alone, and copies only what it holds on
the host. The copy then takes the conversation's positions every node wrote
with its prompts, and the entry keeps those tokens: a shared system prompt
and the turns before the last reply, which is what a new conversation
shares. The slot's own next turn reuses the reply as well. Each copy the
workers refuse outright is logged (`no prefix pool copy of …`).

### Saved across a release

When a top-level node lets its workers go (`release`) and when it stops, it
saves its prefix cache on every node of the tree: each worker writes the
rows its own layers hold of each entry to its own disk, and the top-level
node writes its own layers' rows and the entry's tokens beside a
`manifest.json` that records each entry and the split of the tree — every
node's name and layers — the entries were made on. A worker never saves on
its own. The files are under

```
~/.orangu/server/<model header hash>/tree-prefixes/<node name>/
```

on each node, one `<key>.bin` per entry, under a random 128-bit key the
top-level node chooses; the top-level node's own files and its manifest are
in `lead/` there. Each file records the model, the node's layers, the shape
of its cache, the context it was saved for and the positions it holds.

The nodes of a tree differ in memory, processors and load, so a new plan
can cut the same nodes differently every time. When the top-level node
claims the nodes named in its manifest again, it keeps the saved split as
long as the context is the same and every node still holds its saved share:
within the budget it reports now, or with a budget no smaller than when the
entries were saved. The manifest records each node's budget for this. A
node the saved split names is claimed whatever its load, so a busy moment
does not cost the saved entries. Each entry is then opened as a session on
every worker from its file and put back in the cache, and a prompt that
starts like it computes only what is new. When a node of the saved split is
missing, another one takes part, or a node cannot hold its share, the
tree is cut afresh and the saved entries are deleted on every node.

The top-level node knows what every node holds before it sends anything:
a worker lists `prefixes` under `features` and, in its handshake, the keys
it holds with the layers each covers. An entry is only saved when every
worker of the plan lists `prefixes`, and only opened on a worker that listed
its key for the layers it runs now. Each node checks a file field by field
before it uses it; a file that is missing, damaged, or saved for another
model, other layers, another cache shape or another context is refused, and
the entry is dropped on every node.

An entry taken out of the cache by a prompt, or dropped from it, is deleted
on every node; the next save writes what the cache holds then. A node keeps
at most four saved entries, as many as the cache holds, and deletes its
oldest beyond that.

`GET /v1/workers` on the top-level node shows `saved` (entries on disk) and
`loaded` (entries put back at the last claim) under `prefix_cache`, and on
every node `saved_prefixes` with the entries and bytes it holds on its disk.
The log has one line for each save and each load, with the entries and their
tokens.

## Limits

- A slot's next turn reuses its conversation's rows on the workers, and a
  new conversation copies a prompt prefix another slot's conversation has
  in common, on every worker. Paged KV, MTP heads and slot save/restore are
  off on a top-level node with workers.
- One request runs through the machines one after another, so a single
  request is not faster than on one machine — but for a long prompt, which
  can run on one fast node of the tree (see **Where a long prompt runs**).
  What a tree adds is
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
  works for another node: it never routes, leads or serves alone. At least
  one node of the channel, or Hugging Face, has to have the whole model file.
  A model split in shards is refused, and only a single-file model is
  served to other nodes. The tensors outside the layer blocks — the
  embedding, the output head and the final norm — are covered by no
  fingerprint: they are taken as the first source that serves the right
  header sends them.
- A channel is found over IPv4 only, on networks this machine has an
  address on. Nodes behind a router are not found.
- A GPU keeps the layers it has uploaded until the process ends. A new plan
  frees their host memory, but not their device memory.
- Before another node claims it, a node serves its own clients with the
  whole model, which reads every layer in. The next plan lets go of those
  pages.

# The inference engine: a contributor's map

`orangu-server` carries fifteen architecture modules across fourteen
architecture families, and six device backends.
Each one documents itself well; what has been missing is the thing you need
*first* — where they sit, which of them you have to touch, and which rules are
not visible from the module you happen to be reading.

This is an orientation document, not a reference. It says where to look and
what will bite you. The module doc comments say how each piece works, and they
are the authority wherever this and they disagree.

## The shape, in one page

A generating request travels through four layers, and each one has a single
job:

```
http::openai / http::native      the wire format (OpenAI-shaped, or native)
        |
engine::generate::run            the decode loop: sample, stream, stop
        |
engine::arch::<architecture>     the forward pass: this model's graph
        |
engine::backend::<device>        the arithmetic: matmul, on a device
```

Two of those are extension points, and the whole engine is arranged so that
you only ever add to one at a time:

- **`ModelForward`** (`engine::arch`) — a model architecture. Given tokens and
  a KV cache, produce logits.
- **`Backend`** (`engine::backend`) — a device. Given a quantized matrix and
  some activations, produce a product.

Everything else — slots, admission, prefix reuse, sampling, streaming,
metrics — is shared and architecture-agnostic. If you find yourself needing to
change `engine::generate` to add a model, stop and check: that has almost
always meant the architecture was fighting an abstraction rather than using
it.

## Adding an architecture

Three edits and one implementation.

**1. Recognise the name.** `engine::loader` maps a file's
`general.architecture` string onto an `ArchFamily` through a set of constant
lists (`LLAMA_STYLE_ARCHITECTURES`, `GEMMA_ARCHITECTURES`, and so on). A
family is a *forward pass*, not a vendor: `llama`, `qwen2`, `qwen3`, `mistral`
and `qwen3vl` are all one family because they are the same graph.

Adding a name to an existing list is the cheapest possible change and is
usually the right one. Adding a new family is what you do when the graph
differs.

**Do not add a name you have not checked against a real checkpoint.** Beside
those lists sits a second, shorter list — `KNOWN_UNSUPPORTED` — of names that
look like they belong to a supported family and do not: `glm4` beside
`glm-dsa`, `kimi-linear` beside `kimi-k3`, and `phi2`/`phimoe` beside `phi3`.
Each refuses with its own reason. They are there because nothing downstream
would catch the mistake: the loader
validates quantization types, not graphs, so a wrong name loads if the tensors
happen to be named the same and then generates confident nonsense. That is also
why there is no configuration option to map one architecture onto another.

The list shrinks the honest way. `nemotron_h` was on it until its dense FFN
was implemented and checked against a real file; a test asserts nothing can be
both supported and listed, so the entry had to go in the same change.

**2. Construct it.** `build_model` in `main.rs` is a single `match` over
`ArchFamily`. One arm each, and the same function loads both halves of a
speculative pair, so an architecture reachable as the served model is
automatically reachable as a draft.

**3. Implement `ModelForward`.** Four methods are required:

| method | what it must do |
| :-- | :-- |
| `config` | hand back the `ModelConfig` read from the file |
| `new_kv_cache` | build a cache with this architecture's own per-layer shape |
| `forward` | tokens in, last position's logits out, cache extended |
| `forward_hidden_states` | one-shot pass returning final hidden states, for embeddings |

Six more are defaulted, and each is an opt-in capability rather than a hole to
fill. Skipping one costs a feature, never correctness:

| method | overriding it buys | who overrides it today |
| :-- | :-- | :-- |
| `n_trunk_layer` | correct layer count on files carrying a trailing draft block | `deepseek4`, `glm`, `nemotron`, the three Qwen 3.5-family modules |
| `vulkan_backend` | GPU timing and tuning reports reach `/props` and `/gpu-timings` | most GPU-capable modules |
| `forward_maybe_sampling` | device-side argmax, so a greedy decode step never reads back a whole vocabulary | `llama`, `gemma`, `mistral`, `phi` |
| `forward_all_logits` | **multi-position decode** — speculative decoding needs it on *both* halves of a pair | `gemma`, `deepseek4`, `glm`, `muse` |
| `post_pool_projection` | extra adapter layers after embedding pooling | `gemma` |
| `forward_batch_decode` | cross-sequence fused decode | `gemma` |

Start with the four required methods and nothing else. A correct architecture
that gives up every fast path is a finished contribution; the optional methods
are separate, measurable follow-ups.

## Reuse the shared machinery

Almost every graph is assembled from pieces that already exist, and the pieces
carry hard-won behaviour that a local reimplementation will not:

- `engine::attention::attention` — grouped-query attention with the sliding
  window, the causal mask and the KV-cache reads already right. Four modules
  read the cache directly instead, and all four do it because their attention
  genuinely differs (latent attention, sparse indexers, delta-net) — that is
  scope, not preference.
- `engine::arch::rms_norm_rows`, `swiglu_ffn`, `top_k_indices` — the ordinary
  building blocks.
- `engine::arch::route` and `evaluate_routed_experts*` — mixture-of-experts
  routing, expert batching and the expert budget.
- `engine::kv_cache::KvCache` — per-layer dims, strides, recurrent state, the
  device mirror.

Three modules are worth reading before writing a fourth of anything:
`llama.rs` is the plain case, `gemma.rs` the elaborate one, and `qwen_hybrid.rs`
is a *shared trunk* — `qwen35`, `qwen35moe` and `qwen3next` are 80–170 lines
each because they delegate to it. If your new architecture is a variant of an
existing one, that is the pattern to copy.

## The rules that are not obvious

These are the ones that have actually cost debugging time. Every one of them
is invisible from inside the module where it bites.

**A KV cache's `len` is not what the host holds.** The fused GPU decode path
writes a token's key and value straight into the device mirror, so `len` runs
ahead of the host `k`/`v` buffers by one row per decode step. Anything reading
the host side must bound itself by `KvCache::host_committed_len`. Bounding by
`len` indexes off the end of the buffer — a live crash on the commonest
workload there is, a growing conversation on a GPU.

**A layer's `len` is rows, not tokens.** A block-compressed slot has a
`stride`, and one row stands for `stride` tokens. Convert through it; several
helpers exist that already do.

**Some tensors never go to a device.** `is_cpu_only_tensor` names them —
routed and shared expert weights. Shared-expert matmuls must go through
`matmul_host_fallback` rather than `Backend::matmul`, or a GPU backend is
handed a tensor type it has no kernel for.

**Multi-position forwards must keep their KV rows on the host.** That is what
separates `forward_all_logits` from a single-token `forward`, and it is why
speculative decoding requires it on both models rather than only the one being
verified.

**A trailing block is not always a layer.** A file's `block_count` can include
a multi-token-prediction block the forward pass never runs. Override
`n_trunk_layer` and the banner, the footprint and the placement plan all agree.

## Adding a backend

Smaller than it looks: `Backend` has exactly **one** required method,
`matmul`. Everything else — batching, the decode-specific variants, type
support, the `wgpu` downcasts — is defaulted, and the defaults are correct if
slow.

`cpu.rs` is the reference implementation and the one every other backend is
checked against. `vulkan.rs` is the flagship and is very large; it is not the
place to learn the trait from.

`supports_type` is worth overriding early: a backend that claims a quantization
it cannot decode fails inside a matmul partway through the first request,
where the error has no useful context. Declaring the truth makes it a startup
error naming the type.

## Testing conventions

The engine's tests are unusually load-bearing, because almost nothing here
fails loudly — a wrong forward pass produces plausible text.

- **Cross-check against something independent.** A test that shares the
  component under test proves nothing; a fast-because-wrong prefill once
  survived more than a thousand of them. `doc/DEVELOPERS.md` documents the
  reference-fixture convention for capturing vectors from an independent
  implementation.
- **Test at model-shaped dimensions.** Head counts and hidden sizes of 2 and 4
  hide indexing bugs that only appear at real ratios.
- **Prove the test bites.** Break the fix deliberately and watch the test fail.
  Several tests in this tree passed for the wrong reason until someone did
  that, and the ones that survived it say so in their doc comments.
- **Smoke-test with enough tokens.** Short generations miss the chunking, the
  cache growth steps and the window boundaries.

## The module map

Architectures, by what they are rather than by name:

| module | shape |
| :-- | :-- |
| `llama` | plain GQA + RoPE + RMSNorm + SwiGLU — the baseline |
| `mistral`, `phi` | the same family with their own tensor layouts |
| `gemma` | QK-norm, per-layer sliding/full attention, cross-layer KV sharing, per-layer embeddings, GEGLU, logit softcapping, optional MoE |
| `qwen_hybrid` | shared trunk: full attention alternating with gated-DeltaNet |
| `qwen35`, `qwen35moe`, `qwen3next` | thin modules over that trunk — dense, MoE, and a different MoE layout |
| `deepseek4` | multi-head latent attention, block-compressed KV, routed experts |
| `glm` | latent attention with a sparse-attention indexer |
| `kimi3`, `inkling`, `muse` | their own attention or framing variants |
| `nemotron` | one sub-layer per block, rectangular SSM state, gate-less squared-ReLU FFN — routed (`nemotron_h_moe`) or dense (`nemotron_h`) |
| `dflash` | not a servable model — a draft sidecar that resolves to its target |

Backends, and they are not six of a kind:

- `cpu` — the reference implementation, and what every other backend is
  checked against.
- `vulkan` — the flagship, written entirely against portable `wgpu` and WGSL:
  every pipeline, the fused decode and prefill submissions, GPU sampling.
  Large. Not the place to learn the trait from.
- `metal` — the same kernels as `vulkan`, on a different `wgpu` backend
  rather than a reimplementation. `dx12` reaches them the same way.
- `cuda`, `opencl`, `rocm` — separate, smaller `matmul`-focused
  implementations, each in its own kernel language.
- `multi` — not a device at all: the wrapper that routes each layer's weights
  to whichever device holds them, which is what makes `device_split` work
  without a single line of any forward pass changing.

## Where to start

- A new model that is an existing graph: add its name to a family list.
- A new graph: copy the closest module, strip it to the four required methods,
  and get greedy output matching an independent implementation before adding a
  single fast path.
- A new device: implement `matmul` and `supports_type`, and check it against
  `cpu`.

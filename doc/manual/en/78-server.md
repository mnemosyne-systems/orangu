\newpage

## Inference server internals

`orangu-server` (`src/bin/orangu-server/`) is a fourth binary in the same
Cargo package as `orangu`, `orangu-coordinator`, and `orangu-gguf`. Unlike
those three, it does real tensor computation itself — GGUF loading,
dequantization, the transformer forward pass, sampling, and request
scheduling are implemented in Rust with no dependency on llama.cpp/ggml's
own compiled code.

### Module layout

- `main.rs` — CLI parsing, model-spec resolution, GPU backend selection
  (`select_backend`), and process wiring (Ctrl+C/`SIGINT`/`--daemon`).
- `config.rs`, `init.rs` — `orangu-server.conf` loading and the `--init`
  wizard, mirroring `orangu-gguf.conf`'s shape.
- `shell.rs` — hand-written bash/zsh/fish completion scripts.
- `engine/loader.rs` — memory-maps a GGUF file, reads `<arch>.*`
  hyperparameters, resolves tensor byte ranges.
- `engine/quant.rs` — dequantization for every supported `ggml_type`.
- `engine/tensor.rs` — the handful of numeric ops (matmul, RMSNorm,
  softmax, RoPE, SwiGLU/GEGLU) a forward pass needs, on plain `f32`
  slices — not a general ND-array library.
- `engine/arch/{mod,llama,gemma,qwen35moe}.rs` — one `ModelForward`
  implementor per architecture family.
- `engine/backend/{mod,cpu,vulkan,vulkan_shaders,cuda,opencl,rocm}.rs` —
  the `Backend` trait and its five implementors; see below.
- `engine/tokenizer.rs` — a from-scratch BPE tokenizer.
- `engine/chat_template.rs` — renders `tokenizer.chat_template` via
  `minijinja`.
- `engine/sampling.rs` — repetition penalty, temperature/top-k/top-p/min-p.
- `engine/kv_cache.rs` — per-sequence KV cache buffers.
- `engine/scheduler.rs`, `engine/generate.rs`, `engine/batch.rs` — the
  multi-slot request scheduler and continuous-batching machinery.
- `http/{mod,openai,native}.rs` — the HTTP surface.
- `web/{mod,render,sessions}.rs` — the built-in chat UI.

### GGUF loading and dequantization

`engine::loader` memory-maps the file and reads hyperparameters using the
same `<arch>.*` key names llama.cpp itself reads (confirmed directly
against `llama.cpp/src/llama-arch.cpp`'s `LLM_KV_*` table). Weight tensors
are **not** eagerly dequantized into RAM — each row is read straight from
the `mmap` and dequantized on demand, so even a large model's memory
footprint stays close to its file size.

`engine::quant`'s dequantization struct layouts and algorithms are taken
directly from ggml's own `ggml-common.h`/`ggml-quants.c`
(`dequantize_row_*`), not reimplemented from a description, so the CPU
path is bit-for-bit compatible with what llama.cpp itself reads. Supported
types: `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`
— any other `ggml_type` fails to load with a clear "not yet supported"
error rather than misreading it.

### Model forward passes

One `ModelForward` implementor per architecture family (`engine::arch::
mod`), so adding a family is additive rather than a rewrite:

- `llama.rs` — grouped-query attention, RoPE, RMSNorm, SwiGLU: the shape
  shared by `llama`/`qwen2`/`qwen3`/`mistral`/`qwen3vl` GGUFs (tensor names
  confirmed against `llama.cpp/src/llama-arch.cpp`'s `LLM_TENSOR_NAMES`
  table for `LLM_ARCH_LLAMA`).
- `gemma.rs` — targets `gemma4` (confirmed against upstream `llama.cpp`'s
  `src/models/gemma4.cpp`), with `gemma`/`gemma2`/`gemma3` as subsets of
  its hyperparameter set: soft-capping, sliding-window attention,
  per-layer embeddings (PLE), and GEGLU.
- `qwen35moe.rs` — Qwen3.5/3.6-MoE (confirmed against upstream
  `src/models/qwen35moe.cpp`/`delta-net-base.cpp`): a genuinely different
  shape, with mixture-of-experts FFN routing.

### Request scheduling and continuous batching

`engine::scheduler`'s `SlotPool` bounds how many requests generate
concurrently (`slots` in the config) and tracks each one's progress for
`/slots`. Each slot's prefill+decode loop (`engine::generate::run`) runs on
its own blocking-pool thread against its own KV cache — real concurrency,
bounded fairly by slot count, but not a single fused multi-sequence GEMM by
default.

`engine::batch::BatchCoordinator` is an opt-in alternative for that last
part: when `slots > 1` and the `ORANGU_BATCH_DECODE` environment variable
is set, concurrently-decoding requests within a short window are collected
and handed to `ModelForward::forward_batch_decode` as one call, fusing
every sequence's QKV/`wo`/FFN/PLE/`lm_head` matmuls into a single backend
call each (attention, RoPE, and the KV-cache write stay per-sequence, since
each sequence has its own cache and position). Correctness-verified
against independent per-sequence `forward` calls, but **off by default**:
under concurrent load (4 requests, 100 tokens each, `slots=4`) it measured
around 60% *slower* than the unbatched path — the generic `Backend::matmul`/
`matmul_batch` interface reads results back to the CPU between steps,
reintroducing per-layer round trips the Vulkan backend's own fused decode
path (below) was specifically built to eliminate, and that cost outweighs
the weight-bandwidth savings batching provides at this scale on the
hardware this was measured on. Left available behind the flag rather than
removed, since a genuinely GPU-resident batched-and-fused pipeline could
plausibly flip this positive on different hardware or at higher
concurrency.

### GPU backend architecture

`engine::backend::Backend` (`backend/mod.rs`) is the trait every backend
implements — `matmul`/`matmul_batch` plus a downcast hook (`as_vulkan`) the
model forward pass uses to reach `VulkanBackend`'s much larger fused
surface when it's the active backend. Five implementors exist:
`CpuBackend` (scalar with runtime AVX2 dispatch via `engine::tensor::dot`,
parallelized across output rows with `rayon`; always available, and the
fallback when no GPU backend is found), `VulkanBackend`, `CudaBackend`,
`OpenClBackend`, and `RocmBackend`.

`main.rs`'s `select_backend` implements the `backend = auto` cascade:
Vulkan, then CUDA, then OpenCL, then ROCm (if built with the `rocm`
feature), falling back to `CpuBackend` if none of them initialize. An
explicit `backend = <name>` instead calls that one backend's `try_init`
directly and fails to start if it returns `None`, rather than falling back
— useful when GPU inference was asked for specifically and a silent
CPU fallback would be the wrong failure mode.

### The Vulkan backend

`VulkanBackend` (`engine::backend::vulkan`, via `wgpu`'s Vulkan backend —
`ash` dlopens the system Vulkan loader at runtime, so no Vulkan SDK is
needed to build, only a driver to run against a GPU) is the mature,
hardware-verified backend. Each supported `ggml_type` gets two WGSL
compute pipelines sharing the same per-type dequantization math
(`dequant_element` in `vulkan_shaders.rs`, a line-for-line port of
`engine::quant`'s dequant algorithm restated in WGSL), dispatched
differently by `n_tokens`:

- **Small `n_tokens`** (decode's `n_tokens == 1`, the dominant case for
  interactive generation): `MAIN_REDUCE_SUFFIX` dispatches one workgroup
  per `(output row group, token)` pair — `REDUCE_N_ROWS` (4) output rows
  computed per workgroup, reusing each activation read across all four and
  combining partial sums via a tree reduction, with adjacent threads
  reading adjacent elements of the same row for memory coalescing.
- **Large `n_tokens`** (`>= 64`, e.g. a long prompt's prefill): a
  cooperative/tiled dispatch, one workgroup per output row, that
  dequantizes each weight block once per workgroup into shared memory and
  shares it across up to 64 tokens instead of redoing that dequant per
  token.

A weight tensor is uploaded once (still quantized) and cached on the GPU
for the model's lifetime. For Gemma-family models, `VulkanBackend::
fused_attention` chains QKV projection, Q/K-norm, RoPE, the KV-cache
write, and the attention kernel itself into one GPU submission;
`fused_post_attention` similarly chains the residual add, RMSNorm, and
GEGLU; `record_fused_layer`/`fused_layer` fold a whole layer (attention +
FFN) into one command encoder; and `GemmaModel::forward` chains every
layer plus `output_norm`/`lm_head` into one shared encoder per decode
step. Together these dropped GPU submissions per decode token from roughly
107 to 2 on this project's own benchmark hardware (an AMD RX 5500M running
`gemma-4-E2B`), taking real end-to-end decode throughput from ~1.4 tok/s
to the ~7–9 tok/s range depending on which of the opt-in kernels below are
also enabled — still meaningfully behind llama.cpp's own tuned Vulkan
backend (~36 tok/s on the same hardware/model), a gap now attributed to
kernel quality (`f32` math throughout, no subgroup reductions, no
flash-attention) rather than round-trip overhead, since round trips have
already been mostly eliminated.

Several further optimizations exist behind environment-variable opt-ins,
each correctness-verified against `CpuBackend` but left off by default
because a real, same-session A/B measurement didn't clearly justify making
it the default:

- **`ORANGU_KV_F16=1`** — stores the KV cache as `f16` instead of `f32`,
  halving its memory bandwidth at the cost of an extra per-layer cast
  dispatch. Measured ~2–3% *slower* at the context lengths this project
  can test end-to-end (KV traffic isn't the bottleneck at that scale); a
  much longer context could plausibly flip this positive.
- **`ORANGU_PACKED_DOT=1`** — dequantizes `Q4_K` weight elements in pairs
  and accumulates the dot product as `vec2<f16>` instead of two scalar
  `f32` multiplies. The first genuine kernel-quality win found: a real,
  reproducible ~19% throughput gain. Off by default because the win is
  only measured on one GPU generation (RDNA1), and packed-`f16` throughput
  isn't necessarily a universal hardware advantage.
- **`ORANGU_WIDE_LOAD=1`** — binds the weight buffer as `array<vec4<u32>>`
  (16-byte reads) instead of `array<u32>` (byte-wise reads), so a `Q4_K`/
  `Q5_K` block's whole header can load in one 16-byte read instead of
  several byte reads. Bit-for-bit correctness-verified for all 9 quant
  types; measured a real, reproducible ~11–13% throughput gain on `Q4_K`.
  Combining it with `ORANGU_PACKED_DOT` was tried and measured a
  *regression* relative to either alone, so the two are not meant to be
  combined.
- **`ORANGU_TILED_PREFILL=1`** — a `16×64`-output-tile GEMM for prefill
  (`n_tokens >= 64`) that reuses activations across output rows, unlike
  the default cooperative kernel (one workgroup per output row, each
  re-reading the whole activation matrix independently). Correctness-
  verified but unmeasured end-to-end: long prompts on this project's own
  dev hardware reliably trigger GPU driver hangs (a pre-existing hardware
  limit that affects the unchanged default kernel too, not something this
  change causes), which ruled out a trustworthy A/B.
- **`ORANGU_GPU_SAMPLE=1`** — runs greedy (temperature-0) argmax sampling
  on the GPU in the same submission as the forward pass, avoiding a full
  `[n_vocab]` logits readback. Correctness-verified, but measured ~5–10%
  *slower* — a single-workgroup reduction over a large vocabulary
  apparently costs more GPU time than the PCIe readback and CPU-side
  argmax it replaces. A wider, multi-workgroup reduction could plausibly
  flip this positive but hasn't been attempted.

Shader compilation is cached to disk across restarts
(`~/.orangu/server/<adapter-key>/cache.bin`, keyed by a vendor/device-
derived string so a cache built for one GPU is never handed to another) —
a startup-time optimization only, with no effect on decode/prefill
throughput once running.

### CUDA, OpenCL, and ROCm backends

`engine::backend::cuda::CudaBackend`, `engine::backend::opencl::
OpenClBackend`, and `engine::backend::rocm::RocmBackend` each implement the
same `Backend` trait, at a deliberately smaller scope than Vulkan: one
dequantizing matmul kernel per `ggml_type`, a direct port of
`vulkan_shaders`'s `MAIN_REDUCE_SUFFIX` reduction strategy restated per
kernel language (CUDA-C, OpenCL-C, HIP-C), cross-checked against
`CpuBackend` the same way `VulkanBackend`'s own tests are. Deliberately
**not** ported: `VulkanBackend`'s cooperative/tiled dispatch, GPU-resident
attention/RoPE/norm fusion, fused whole-layer submissions, GPU-side argmax
sampling, and the disk pipeline cache — none of the three has been run
against real hardware during development (no NVIDIA GPU, no ROCm install,
no OpenCL ICD on the project's dev machine), so correctness rests on the
kernel math matching `engine::quant`'s already-verified dequant code
line-for-line, plus the same CPU cross-check test pattern `vulkan.rs`
uses (which, like those tests, skips gracefully rather than fails when no
matching device is found).

`cudarc` and the resolved `opencl3` version both dlopen their vendor
library (`libcuda.so`/`libnvrtc.so`, `libOpenCL.so`) at runtime and return
a real error if it can't be found, so `cuda`/`opencl` are always compiled
in — nothing extra is needed to *build* `orangu-server`. `cubecl-hip-sys`
(ROCm's underlying bindings) is different: it directly links
`-lamdhip64 -lhiprtc` at *build* time whenever its build script finds a
ROCm install, which would break a plain build on a machine without ROCm —
so `rocm` sits behind its own Cargo feature, off by default (see
[BUILDING.md](../../BUILDING.md)).

`cudarc` has one notable wrinkle: unlike every other fallible step here, it
`panic!`s (rather than returning a `Result`) the first time a driver/NVRTC
call is made and no `libcuda.so` is found. `CudaBackend::try_init` runs
`try_init_inner` under `std::panic::catch_unwind` (with the panic hook
silenced for the call) specifically so a non-NVIDIA machine gets the same
graceful `None`/CPU-fallback outcome every other missing-backend path
already has, not a crashed server.

### Correctness testing

`VulkanBackend`'s dequant math (each quant type, bit-for-bit against the
CPU backend, across both dispatch paths), fused post-attention chain
(including a dedicated test that calls it twice for one layer with
different inputs each time, to catch cache-reuse bugs specifically), and
fused attention (including GQA head-grouping, sliding-window attention,
proportional RoPE, and Gemma4's cross-layer KV-donor case — two different
layers sharing one KV cache) are covered by cross-check tests in
`engine::backend::vulkan::tests`, run on real Vulkan hardware whenever
it's present and skipped otherwise. The CUDA/OpenCL/ROCm backends follow
the same skip-if-no-device pattern.

### HTTP layer and web UI

`http::mod` assembles the router and shared `AppState` (model, scheduler
handle, config, start time); `http::openai` and `http::native` hold the
OpenAI-compatible and llama.cpp-native handlers respectively; `/v1/shutdown`
lives in `http::mod` itself since it's neither. Ctrl+C, `SIGINT`, and
`POST /v1/shutdown` all converge on the same shutdown path via
`tokio::select!`, mirroring `orangu-coordinator`'s own pattern.

`web::mod` serves a small server-rendered chat UI (vanilla HTML/CSS/JS, no
build step) on its own `web` port, sharing the same in-process `Engine` as
the API so a chat turn never makes an HTTP hop. `web::render` renders
markdown to HTML (including syntax-highlighted code blocks) with the same
`markdown`/`syntect` crates `orangu`'s terminal UI uses. `web::sessions`
persists each chat as `~/.orangu/server/sessions/<uuid>/chat.json`.

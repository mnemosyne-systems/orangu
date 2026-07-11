# Building orangu

`orangu` is a Rust project with four binaries: the interactive client
(`orangu`), an optional HTTP proxy that starts/stops llama.cpp on demand
(`orangu-coordinator`, see [doc/COORDINATOR.md](COORDINATOR.md)), a
standalone CPU/GPU and GGUF file inventory tool (`orangu-gguf`, see
[doc/GGUF.md](GGUF.md)), and a native GGUF inference server
(`orangu-server`, see [doc/SERVER.md](SERVER.md)).

## Prerequisites

- Rust toolchain with `cargo`
- A running llama.cpp server exposing an OpenAI-compatible API (not needed
  if `orangu.conf` points at `orangu-server` instead — see
  [doc/SERVER.md](SERVER.md))

## Build

```sh
cargo build
```

For an optimized build:

```sh
cargo build --release
```

For an optimized build that keeps debug symbols (for profiling a release
build with `perf`/`valgrind`, or debugging one with `gdb`/`lldb` — a plain
`--release` build strips symbols, making stack traces and flame graphs
unreadable):

```sh
cargo build --profile release-with-debug
```

`[profile.release-with-debug]` in `Cargo.toml` inherits every optimization
setting from `release` and adds two things: `debug = 1` (line-table debug
info — enough for a backtrace or flame graph to resolve file/line, without
the full type/variable info `debug = 2`/`true` would add, keeping the
binary smaller and the build faster) and `panic = "unwind"` (already
`release`'s own default, spelled out explicitly here rather than left
implicit, since unwind is what lets a debugger catch a panic mid-unwind
and what gives `RUST_BACKTRACE` a real stack to walk). Same codegen, same
runtime speed as `release` otherwise — no separate profile to keep in sync
as `release` itself changes. The binary lands at
`target/release-with-debug/<name>` (Cargo names the output directory
after the profile, not `release`).

## AVX2 / SSE4.2

`.cargo/config.toml` sets `-C target-feature=+avx2,+fma,+sse4.2` for
x86_64 builds only — scoped via `[target.'cfg(target_arch = "x86_64")']`,
which Cargo resolves against whatever architecture is actually being
*built for* (the `--target` passed, or the host triple for a plain
`cargo build`), so it doesn't affect the aarch64/musl/macOS-arm
cross-compile targets `.github/workflows/release.yml` also builds. Verified
directly, not just by inspection: `rustc --print cfg --target
aarch64-unknown-linux-gnu` reports `target_arch="aarch64"` (so the `cfg`
predicate above is false there), and `cargo build --target
aarch64-unknown-linux-gnu` compiles real dependency crates that have their
own CPU-feature-gated code paths (`cfg-if`, `memchr`, `smallvec`, `log`)
with no `target feature avx2 is not supported` error — it only fails on
this machine not having the aarch64 sysroot installed, unrelated to this
config.

SSE4.2 is listed explicitly as its own mandatory floor even though AVX2
already requires it as a
hardware prerequisite (there's no x86_64 CPU with AVX2 but not SSE4.2) —
it changes nothing about codegen today, but keeps SSE4.2 a hard
requirement even if `+avx2,+fma` is ever dropped on its own.
`orangu-server`'s `engine::tensor` module additionally does its own
*runtime* `is_x86_feature_detected!` dispatch with a scalar fallback for
its hottest loop (`dot`, the matmul/attention inner product), so that path
works either way — but everywhere else, this flag is what lets LLVM
autovectorize the engine's other elementwise loops (RMSNorm, residual
adds, SwiGLU/GEGLU) with AVX2/FMA instructions.

This means a binary built from this repo (including the ones
`release.yml` publishes for `x86_64-unknown-linux-gnu`/`musl` and
`x86_64-pc-windows-msvc`) requires an AVX2+FMA+SSE4.2-capable CPU to run
at all — every x86_64 CPU since ~2013 (Intel Haswell, AMD Excavator/Zen)
qualifies, but older or restricted-CPUID virtualized x86_64 hosts don't.
Delete or edit `.cargo/config.toml` to build a more portable (and slower)
binary instead.

## GPU backends

`orangu-server`'s Vulkan/CUDA/OpenCL GPU backends (`engine::backend::
vulkan`/`cuda`/`opencl`) are always compiled in — a plain `cargo build`
needs nothing beyond what's already covered above, since `wgpu`/`cudarc`/
`opencl3` all dlopen their vendor library at *runtime*, not build time.

The ROCm/HIP backend (`engine::backend::rocm`) is the one exception: it's
behind a `rocm` Cargo feature, off by default, because its underlying
bindings (`cubecl-hip-sys`) link directly against `libamdhip64`/`libhiprtc`
at *build* time whenever a ROCm install is detected — harmless on a
machine that has ROCm, but it would break a plain `cargo build` on any
machine that doesn't (confirmed directly on this project's own dev
machine, which has no ROCm installed). Build with it via:

```sh
cargo build --release --features rocm
```

See `doc/manual/en/78-server.md` (the Developer information chapter's
"CUDA, OpenCL, and ROCm backends" section) for what each of the three
non-Vulkan GPU backends actually implements (a real but smaller-scoped
`matmul`-only kernel, unverified on real hardware — none of CUDA, OpenCL,
or ROCm hardware was available when they were built).

## Test

```sh
cargo test
```

## Manual generation

The project includes a pandoc-based manual layout under `doc/manual/en`.

To build the manual:

```sh
./doc/build_manual.sh
```

The script writes HTML and PDF output to `target/doc/`.

## Example run

```sh
cargo run --bin orangu -- --config ./doc/etc/orangu.conf
```

```sh
cargo run --bin orangu-coordinator -- --config ./doc/etc/orangu-coordinator.conf
```

```sh
cargo run --bin orangu-gguf -- system
cargo run --bin orangu-gguf -- --config ./doc/etc/orangu-gguf.conf list
```

```sh
cargo run --bin orangu-server -- --config ./doc/etc/orangu-server.conf unsloth/gemma-4-E2B-it-GGUF
```

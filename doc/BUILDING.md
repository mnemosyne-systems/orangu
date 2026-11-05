# Building orangu

`orangu` is a Rust project with three binaries: the interactive client
(`orangu`), an optional HTTP proxy that starts/stops `orangu-server` on
demand (`orangu-coordinator`, see [doc/COORDINATOR.md](COORDINATOR.md)), and
a native GGUF inference server that doubles as a standalone CPU/GPU and GGUF
file inventory tool (`orangu-server`, see [doc/SERVER.md](SERVER.md)).

## Prerequisites

- Rust toolchain with `cargo`
- A running `orangu-server` exposing its OpenAI-compatible API (see
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

## Single-file bundles

A release build of `orangu-server` can write a copy of itself with a model
embedded in it — one executable that serves without a models directory or an
`orangu-server.conf`:

```sh
cargo build --release
./target/release/orangu-server bundle unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --all -y
./orangu-server-bundle-x86_64
```

The bundle is a file operation on an already-built binary, not a build step:
nothing is recompiled, and the model is appended rather than linked in, so a
multi-gigabyte model never goes through `rustc`. Bundling for another platform
is the same command with `--binary` naming the cross-compiled executable:

```sh
cargo build --release --target aarch64-unknown-linux-gnu
./target/release/orangu-server bundle <model> --all -y \
    --binary target/aarch64-unknown-linux-gnu/release/orangu-server
# -> ./orangu-server-bundle-aarch64
```

The architecture in the default name is read out of the binary being bundled,
not out of the host, so cross-bundled output names itself correctly without an
`--output` for each target.

Releases ship the ordinary binaries only — a bundle is as large as the model
inside it, so it is built locally from whichever model suits the machine. See
the *Bundling* section of the
[server chapter](manual/en/46-server.md#bundling-the-server-and-a-model-as-one-file).

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

`orangu-server`'s Vulkan/Metal/CUDA/OpenCL GPU backends (`engine::backend::
vulkan`/`metal`/`cuda`/`opencl`) are always compiled in — a plain `cargo
build`
needs nothing beyond what's already covered above, since `wgpu`/`cudarc`/
`opencl3` all dlopen their vendor library at *runtime*, not build time.
Metal needs no build-time setup of its own either: it is the same `wgpu`
engine as the Vulkan backend, asked for a Metal adapter instead, and
`wgpu`'s `metal` feature is already in `Cargo.toml`. It finds a device
only on macOS, so a Linux or Windows build simply never selects it.

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
"CUDA, OpenCL, and ROCm backends" section) for what each of those three
backends actually implements (a real but smaller-scoped
`matmul`-only kernel, unverified on real hardware — none of CUDA, OpenCL,
or ROCm hardware was available when they were built). Metal is not in that
group: it shares the Vulkan backend's kernels outright, so it is at full
parity and is covered by the same tests.

### Newer Vulkan headers on Debian 12 (for a reference `llama.cpp`)

`orangu-server` itself never needs Vulkan headers — `wgpu` carries its own
bindings — but building a *reference* engine beside it does: `llama.cpp`'s
Vulkan backend (and Prism ML's fork of it, the one that runs the
`Ternary-Bonsai-2` files) uses `vk::LayerSettingEXT` and other symbols
that arrived after the `1.3.239` headers Debian 12 ships (`libvulkan-dev
1.3.239.0-1`; `bookworm-backports` has nothing newer), so a plain
`cmake -DGGML_VULKAN=ON` build fails in `ggml-vulkan.cpp`. The driver is
not the problem: the Mali ICD (`/etc/vulkan/icd.d/mali.json`) reports
`1.3.296`, and the `1.3.239` **loader** (`libvulkan1`) talks to it fine.
Only the headers are old, and they are header-only. Two ways to get them:

**Per build**, no root:

```sh
git clone --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers.git
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
      -DVulkan_INCLUDE_DIR=$PWD/../Vulkan-Headers/include \
      -DCMAKE_CXX_FLAGS="-I$PWD/../Vulkan-Headers/include"
```

Match the tag to the driver's `apiVersion` (`vulkaninfo --summary`);
newer headers than the driver are fine too, older than the code being
built is what fails.

**System-wide**, into `/usr/local` — searched before `/usr/include`, so it
takes precedence over the distro's copy without replacing any package:

```sh
git clone --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers.git
cd Vulkan-Headers
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --install build          # /usr/local/include/vulkan/*.h, vulkan.hpp, vk_video/
```

Then `cmake -DGGML_VULKAN=ON` finds them on its own (`find_package(Vulkan)`
looks in `/usr/local/include` first). To go back, `sudo rm -r
/usr/local/include/vulkan /usr/local/include/vk_video
/usr/local/share/cmake/VulkanHeaders`. Do **not** pull `libvulkan-dev`
from Debian 13 instead: that package depends on a `libvulkan1` built
against a newer glibc than bookworm's.

The other pieces, and whether they need updating:

| piece | on this board | needed? |
| :-- | :-- | :-- |
| loader `libvulkan1` | 1.3.239 | no — a loader is forward-compatible with newer drivers and headers; a source build of `KhronosGroup/Vulkan-Loader` is only for a loader *bug* |
| `vulkan-tools` (`vulkaninfo`) | 1.3.239 | no — it reports the driver's version either way |
| `glslc` (`shaderc` 2023.2) | 2023.2 | no for `llama.cpp` today; a newer `KhronosGroup/glslang` + `google/shaderc` source build only if a shader uses a newer GLSL extension than it accepts (the build log names it) |
| Mali ICD + `libmali` | vendor, `1.3.296` | leave alone — it is the device |

The stock `/usr/local/bin/llama-*` on this board is a CPU-only build
(`ldd llama-bench` shows no `libggml-vulkan`); a Vulkan build of the
same tree installs beside it with `cmake --install build` after the
headers above are in place, or runs from its `build/bin` — the fork under
`/mnt/ai/pgmoneta/prism-llama.cpp` does the latter.

## Test

```sh
cargo test
```

## Documentation generation

One script builds both documents into `target/doc/`:

```sh
./doc/build.sh
```

Both PDFs are drawn by **orangu itself** — `src/bin/orangu/docs.rs`, reached
through the hidden `--build-manual` and `--build-cheatsheet` flags — using the
same printpdf engine that writes the reports `/export` produces. The bands, the
brand colour, the embedded Red Hat Text faces and the page geometry all come
from `src/bin/orangu/export.rs`, so a manual page, a cheat-sheet box and a
review report are the same document family by construction. There is no LaTeX
in the project.

* The **manual** from `doc/manual/en`, one file per chapter, as
  `orangu-en.pdf`: a brand cover, a linked table of contents, numbered
  headings, real tables, and the architecture images.
* The **cheat sheet** from `doc/cheatsheet/en`, one file per page, as
  `orangu-cheatsheet-en.pdf`. The build fails if a page's boxes outgrow it.

Pass `manual` or `cheatsheet` to build just one of them.

`cargo` is the only requirement for the PDFs. Pandoc is still used for the
HTML manual (`orangu-en.html`), which a PDF engine cannot produce.

## Example run

```sh
cargo run --bin orangu -- --config ./doc/etc/orangu.conf
```

```sh
cargo run --bin orangu-coordinator -- --config ./doc/etc/orangu-coordinator.conf
```

```sh
cargo run --bin orangu-server -- system
cargo run --bin orangu-server -- --config ./doc/etc/orangu-server.conf list
```

```sh
cargo run --bin orangu-server -- --config ./doc/etc/orangu-server.conf unsloth/gemma-4-E2B-it-GGUF
```

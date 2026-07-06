# Building orangu

`orangu` is a Rust project with two binaries: the interactive client
(`orangu`) and an optional HTTP proxy that starts/stops llama.cpp on demand
(`orangu-coordinator`, see [doc/COORDINATOR.md](COORDINATOR.md)).

## Prerequisites

- Rust toolchain with `cargo`
- A running llama.cpp server exposing an OpenAI-compatible API

## Build

```sh
cargo build
```

For an optimized build:

```sh
cargo build --release
```

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

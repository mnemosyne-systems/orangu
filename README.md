# orangu

**orangu** is a local workspace-aware tool-driven coding environment for **OpenAI-compatible** servers - especially **[llama.cpp](https://github.com/ggml-org/llama.cpp)**.

**orangu** is named after the [Orangutan](https://en.wikipedia.org/wiki/Orangutan) - the smartest ape.

![orangu terminal interface](doc/images/orangu-terminal.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Install dependencies on Fedora](#install-dependencies-on-fedora)
  - [Release build](#release-build)
  - [Debug build](#debug-build)
- [Configuration and first run](#configuration-and-first-run)
- [Documentation](#documentation)
- [Tested platforms](#tested-platforms)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

## Features

- OpenAI-compatible chat completions, with special attention to llama.cpp workflows
- Workspace-aware local tools for reading, editing, listing, fetching, and shell commands
- Persistent terminal UI with workspace, server, and model status in the header
- Shell-style prompt editing, history, scrolling, and context-sensitive Tab completion
- Local commands such as `/help`, `/list_models`, `/list_files`, `/show_file`, `/tools`, `/diff`, and `/open_file`
- Natural-language local aliases such as `open README.md`, `list models`, `list files`, and `show help`
- Streaming responses with live footer status such as `Thinking (...)` and llama.cpp-native `Working @ X.Y t/s (...)`
- Queued local commands while a response is in flight, plus double-`Esc` request cancellation
- Markdown rendering in the console, including bold, italic, headings, lists, links, and code
- Syntax highlighting for fenced code blocks in supported programming languages

## Installation

### Install dependencies on Fedora

Install the tools needed to build and run **orangu** from source:

```sh
dnf install -y git rust cargo gcc
```

### Release build

The following commands build an optimized release binary:

```sh
git clone https://github.com/mnemosyne-systems/orangu.git
cd orangu
cargo build --release
```

The binary will be available at:

```text
target/release/orangu
```

To install it system-wide:

```sh
sudo install -Dm755 target/release/orangu /usr/local/bin/orangu
```

### Debug build

The following commands build a debug binary:

```sh
git clone https://github.com/mnemosyne-systems/orangu.git
cd orangu
cargo build
```

The binary will be available at:

```text
target/debug/orangu
```

## Configuration and first run

Start from the sample configuration:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Default configuration lookup order:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

Run the client:

```sh
orangu --config ./orangu.conf
```

Or run it directly from the build tree:

```sh
./target/release/orangu --config ./orangu.conf
```

By default, local tools operate on the current working directory. Use `--workspace /path/to/project` to point **orangu** at another tree.

Useful first commands:

```text
/help
/list_models
/list_files
/show_file README.md
/tools
/open_file README.md
```

## Documentation

- [Latest manual](https://github.com/mnemosyne-systems/orangu/tree/main/doc/manual/en)
- [Getting Started](https://github.com/mnemosyne-systems/orangu/blob/main/doc/GETTING_STARTED.md)
- [Quick start](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/03-quickstart.md)
- [Configuration](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/20-configuration.md)
- [Terminal interface](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/40-terminal.md)
- [Tools](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/30-tools.md)

## Tested platforms

- [Fedora](https://getfedora.org/) 44

## Contributing

Contributions to **orangu** are managed on [GitHub](https://github.com/mnemosyne-systems/orangu/):

- [Ask a question](https://github.com/mnemosyne-systems/orangu/discussions)
- [Raise an issue](https://github.com/mnemosyne-systems/orangu/issues)
- [Feature request](https://github.com/mnemosyne-systems/orangu/issues)
- [Code submission](https://github.com/mnemosyne-systems/orangu/pulls)

Contributions are most welcome.

Please consult the [Code of Conduct](https://github.com/mnemosyne-systems/orangu/blob/main/CODE_OF_CONDUCT.md) before contributing.

## Community

- GitHub: [mnemosyne-systems/orangu](https://github.com/mnemosyne-systems/orangu)
- Discussions: [GitHub Discussions](https://github.com/mnemosyne-systems/orangu/discussions)

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

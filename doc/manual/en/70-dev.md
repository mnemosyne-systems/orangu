\newpage

# Developer information

## Main components

- `src/bin/orangu.rs` - terminal loop, command handling, history, connection state, and waiting state
- `src/bin/orangu/manual.rs` - built-in manual viewer (`/manual`); embeds the `doc/manual/en` chapters at compile time, so a new chapter file must be added to its `MANUAL_SOURCES` list
- `src/config.rs` - INI parsing and normalization
- `src/llm/openai.rs` - OpenAI-compatible client for llama.cpp-style backends
- `src/session.rs` - tool-calling conversation flow
- `src/tools.rs` - workspace-scoped local tool execution
- `src/tui.rs` - header, prompt frame, and status rendering

## Prompt Construction & KV Caching

Orangu is specifically optimized for local LLMs (like `llama.cpp`) which rely on exact token prefix matching to reuse KV cache and avoid massive prefill latencies. When developing features that touch `ChatSession`, you must preserve this cache:

1. **Append-Only System Updates**: When updating the system prompt mid-session (e.g., changing workspaces or verbosity), `ChatSession::set_system_prompt()` appends the new instructions as a `user` message with a `[System Update]` prefix. It **never** mutates the initial system message (`messages[0]`). Mutating the first message would instantly destroy the prefix cache for the entire conversation.
2. **In-Place Tool Eviction**: When context limits are reached, `compact_transcript` replaces old tool outputs with a tiny stub `[Tool output evicted...]`. Because this mutation happens in-place further down the array, it perfectly preserves the KV cache prefix for all messages that preceded it.

## Development workflow

```sh
cargo fmt
cargo test
```

## Documentation workflow

1. Download dependencies

    ``` sh
    dnf install pandoc texlive-scheme-basic
    ```

2. Download Eisvogel

    Use the command `pandoc --version` to locate the user data directory. On Fedora systems, this directory is typically located at `$HOME/.local/share/pandoc`.

    Download the `Eisvogel` template for `pandoc`, please visit the [pandoc-latex-template](https://github.com/Wandmalfarbe/pandoc-latex-template) repository. For a standard installation, you can follow the steps outlined below.

```sh
    wget https://github.com/Wandmalfarbe/pandoc-latex-template/releases/download/v3.4.0/Eisvogel-3.4.0.tar.gz
    tar -xzf Eisvogel-3.4.0.tar.gz
    mkdir -p $HOME/.local/share/pandoc/templates
    mv Eisvogel-3.4.0/eisvogel.latex $HOME/.local/share/pandoc/templates/
```

3. Add package for LaTeX

    Download the additional packages required for generating PDF and HTML files.

```sh
    dnf install 'tex(footnote.sty)' 'tex(footnotebackref.sty)' 'tex(pagecolor.sty)' 'tex(hardwrap.sty)' 'tex(mdframed.sty)' 'tex(sourcesanspro.sty)' 'tex(ly1enc.def)' 'tex(sourcecodepro.sty)' 'tex(titling.sty)' 'tex(csquotes.sty)' 'tex(zref-abspage.sty)' 'tex(needspace.sty)' 'tex(selnolig.sty)' texlive-collection-latexextra
```

Then

```sh
./doc/build_manual.sh
```

which will produce a HTML and PDF manual.

## Orangu files

- The default config lookup path is `~/.orangu/orangu.conf`
- Command history is stored in `~/.orangu/orangu.history`

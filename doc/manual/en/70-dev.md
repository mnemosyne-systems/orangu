\newpage

# Developer information

## Main components

- `src/bin/orangu.rs` - terminal loop, command handling, history, connection state, and waiting state
- `src/bin/orangu/manual.rs` - built-in manual viewer (`/manual`); embeds the `doc/manual/en` chapters at compile time, so a new chapter file must be added to its `MANUAL_SOURCES` list
- `src/config.rs` - INI parsing and normalization
- `src/llm/openai.rs` - OpenAI-compatible client for `orangu-server`
- `src/session.rs` - tool-calling conversation flow
- `src/tools.rs` - workspace-scoped local tool execution
- `src/tui.rs` - header, prompt frame, and status rendering

## Prompt Construction & KV Caching

Orangu is optimized for local LLMs (served by `orangu-server`), which reuse a KV cache by matching an exact **token prefix**. When developing features that touch `ChatSession`, the rule that matters is:

> Editing a message costs a re-prefill of that message **and everything after it**. Appending costs nothing.

Two consequences that are easy to get wrong:

1. **An in-place edit is not free just because it is not the first message.** `compact_transcript` replaces old tool outputs with a stub, and the prefix *before* the edit does survive — but everything after it is processed again. Worse, the stub is *shorter* than what it replaced, so the next turn's divergence point lands even earlier. Compacting eagerly, one message per turn, measured 3.5× more prefill than never compacting at all. Compaction therefore runs only when a pass can reclaim at least half the transcript, so the cost is paid once per doubling rather than once per turn. See the Compression chapter.

2. **Not every edit is worth avoiding.** `set_system_prompt` rewrites `messages[0]`, which does discard the whole conversation's cache. It used to append the new prompt as a `user` message prefixed `[System Update]` to dodge that, and the dodge was worse than the disease: the model got its own instructions in the user's voice while the real system message still held the superseded prompt. The rewrite is affordable because of *when* it runs — only `/server` (which changes endpoint, so that cache is cold anyway) and `/verbosity` (one explicit command, paid once). An unchanged prompt is left untouched.

The general shape: prefer appending; when you must edit, edit rarely and be able to say what the edit buys.

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

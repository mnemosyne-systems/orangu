\newpage

## Compression

`orangu` has a built-in compression layer for reducing repeated or noisy
tool output before it is sent back to the model.

## Goals

The goals are:

- Avoid resending unchanged file content
- Reduce noisy shell output
- Preserve the lines most useful to the model
- Improve context efficiency without adding an external runtime dependency

## Design

Compression is implemented natively inside orangu.

It currently sits on the model-facing tool boundary: tool output is shortened
or replaced only when orangu can do so deterministically.

This means it currently applies to the tool results sent back to the model,
as well as the context populated by native user commands. For example, `/show_file`,
`/diff`, and `/show` render their output directly to the user in full, but they queue their
content locally. When the LLM is actually called, this queued context is then
compressed and injected into the active session. This ensures we only compress
and transmit data when the model's reasoning is actively requested.

The setting is controlled by:

```ini
[orangu]
compression = on
```

The default is `on`.

Accepted values are: `on`, `true`, `1`, `off`, `false`, `0`.

When it is `off`, orangu returns the raw output.

## Features

Compression currently affects two model-facing tools, as well as the LLM context
populated by native commands.

While native commands like `/show_file`, `/diff`, and `/show` continue to display their full,
uncompressed output directly to the user, they now automatically compress their
payload when injecting it into the model's context. This lets the LLM follow along
with your local investigation without flooding its context window.

### `show_file`

When the model asks for the same whole file again, and the file is unchanged,
orangu may return a cache stub instead of resending the full file content.
This context cache is **per session**, so you do not waste tokens re-reading unmodified files.

This applies only to repeated whole-file reads. Line-range reads still return
the requested lines normally.

Additionally, `show_file` supports **structural read modes** to grab a high-level
understanding of large files without reading their full bodies:
- `signatures`: Extracts only the public interfaces (functions, structs) while stripping private bodies.
- `map`: Extracts top-level item declarations for an overview.

Additionally, orangu employs **AST-Aware Auto-Downsampling**: if the model requests a read of a file exceeding a certain length without specifying bounding lines, orangu will automatically downsample the read into `signatures` mode to prevent the context window from being flooded with massive file bodies. The threshold is controlled by the `auto_downsample_lines` setting in `[orangu]` (default: 300).

### `run_shell_command`

For recognized high-volume commands, orangu compresses noisy output before the
usual output truncation is applied. Command wrappers and prefixes like `time` or
`CARGO_TERM_COLOR=always` are stripped transparently before matching.

Current patterns include:

- **Rust:** `cargo build`, `cargo check`, `cargo test`
- **C / C++:** `make`, `cmake`, `gcc`, `clang`
- **Python:** `pytest`, `python -m unittest`
- **Node:** `npm test`, `jest`, `yarn test`
- **Java:** `mvn test`, `gradle test`
- **Version Control:** `git log`, `git diff`, `git show`
- **System:** Directory listings (`ls`, `find`), Search outputs (`rg`, `grep`)
- **Package Managers:** `npm install`, `yarn`, `pip install`

### Hot-Line Context Extractor & Disk Diversion (Generic Fallback)

For any unrecognised or extremely large generic output (e.g., custom shell scripts or Makefiles), orangu uses a sophisticated "Hot-Line Context Extractor". Instead of blindly truncating the middle of a log, it scans for universal failure markers (`error:`, `Exception`, `Traceback`, `panic:`, etc.) and perfectly preserves those specific lines along with their surrounding context (+/- 3 lines). All surrounding noise is dynamically collapsed. This ensures the LLM never misses a stack trace, regardless of the language.

**Reverse Compression (Context Expansion):** As an absolute safety net, whenever *any* massive blob of text (a giant diff, a massive file read, or a huge shell output) is severely truncated, `orangu` automatically persists the original uncompressed text to a session-scoped disk cache using a SHA-256 hash. It injects a tiny marker into the LLM's prompt (e.g., `[Note: Output truncated. Run expand_context(id="abc1234567")]`). This grants the LLM the ability to dynamically "reverse" the compression and retrieve the exact missing data on demand using the `expand_context` tool.
### Advanced Diff Engine

When processing `git diff` outputs, orangu bypasses raw text truncation and uses a structured AST diff parser:
- **File Capping:** Preserves only the top most changed files. The limit is controlled by the `diff_file_cap` setting in `[orangu]` (default: 20).
- **Context Trimming:** Squeezes unchanged lines surrounding additions and deletions down to exactly 2 lines.
- **Intelligent Hunk Scoring:** Scores diff hunks based on line density and priority keywords (e.g., `error`, `panic`, `secret`), preserving only the most highly-scored blocks.

### Array/Log Anchor Selector

For raw shell outputs, if orangu detects a massive JSON array or highly repetitive log lines (determined by matching line prefixes), it activates an Anchor Selector. This perfectly preserves the first 3 lines and the last 3 lines while dynamically dropping the massive middle section.

### Secret-Aware Filtering

All strings processed by orangu (file reads, shell outputs) undergo a fast regex redaction pass that scrubs hardcoded API keys (e.g., Anthropic, AWS, GitHub tokens), replacing them with `[REDACTED_SECRET]` before the LLM can read them.

### Transcript Compaction (Live Zone)

Orangu continuously grooms the active conversation transcript. Right before sending a prompt to the LLM, it scans backwards and permanently evicts massive tool outputs (like 10,000-line shell errors) that are older than 3 user turns, replacing them with a tiny stub. This guarantees the context window never permanently fills up with dead tool artifacts.

### Metrics & `/usage`

You can monitor the effectiveness of the compression layer by running the `/usage`
command during your session. It displays real-time metrics including total file reads,
cache hits/misses, bytes saved by the context cache, and lines stripped by shell compression.

## Implementation

The current implementation is a v1.

It provides:

- Repeated unchanged whole-file `show_file` suppression
- Per-session file caching
- Structural read modes (`signatures` and `map`)
- Shell output compression for common noisy commands
- Tracking metrics exposed via `/usage`
- Native command integration (e.g., `/show_file`, `/diff`, `/show`) sending compressed context to the LLM
- A single config switch for enabling or disabling the behavior
- Transcript compaction and tool-output eviction (Live Zone)
- Advanced structural diff compression (File Capping, Context Trimming, Hunk Scoring)
- Secret-aware token scrubbing
- Anchor selection for arrays and repetitive logs

The current design intentionally separates user display from model context: native tools show full content to the user while sending compressed summaries to the model upon LLM query.

It does not yet provide:

- Adaptive learning of which compression heuristics fail most often

\newpage

# Tools

`orangu` exposes local workspace tools to the active model.

Tools are sent as OpenAI `tools` on every chat request, and the model asks for
one by answering with `tool_calls`; `orangu` runs it in the workspace, feeds the
result back as a `tool` message, and lets the model continue. `max_tool_rounds`
in `[orangu]` bounds how many such rounds one prompt may take.

Whether this works depends on the **model**, not on orangu: a model whose chat
template has no tool support never sees the declarations and will say it has no
tools available. `orangu-server` passes the array through to the template and
translates the answer back — see **Tool calling** in the Inference server
chapter for which answer formats it recognises.

Configured Streamable HTTP MCP servers can add tools to this list. Their names are
namespaced as `mcp__<server>__<tool>` so they cannot collide with built-in tools
or another server. orangu discovers them once when a workspace opens; a server
that cannot be reached is disabled with a warning while built-in and other MCP tools
remain available. MCP tool results are returned through the same text tool
result channel as built-in tools and are capped at 20,000 characters.

MCP support currently exposes tools only. Resources and prompts are not exposed;
`/mcp refresh` re-discovers tools at runtime.

## Available tools

| Tool | Purpose | Key arguments |
| :-- | :-- | :-- |
| `show_file` | Show a text file or extract PDF text from the workspace | `path`, optional `start_line`, optional `end_line`, optional `mode` |
| `create_file` | Create a file with content, optionally with its permissions | `path`, optional `content`, `mode`, `overwrite`, `parents`, `git` |
| `modify_file` | Modify a file, by text replacement or by line ranges | `path`, either `old_text`/`new_text` (+ `replace_all`) or `edits`, optional `git` |
| `move_file` | Move or rename a file | `from`, `to`, optional `mode`, `overwrite`, `parents`, `git` |
| `delete_file` | Delete a file | `path`, optional `git` |
| `create_directory` | Create one directory | `path`, optional `mode`, `parents` |
| `move_directory` | Move a directory and everything under it | `from`, `to`, optional `mode`, `parents` |
| `delete_directory` | Delete an empty directory | `path` |
| `list_directory` | List files and directories below the workspace, or the files a glob matches | optional `path`, optional `max_depth`, optional `pattern` |
| `fetch_url` | Fetch an external URL and return readable text | `url`, optional `max_chars` |
| `run_shell_command` | Run a shell command inside the workspace | `command`, optional `cwd`, optional `timeout_seconds` |
| `expand_context` | Retrieve previously compressed/truncated output using its hash ID | `id` |
| `semantic_search` | Search workspace code by meaning, refreshing the embedding index first | `query` |

`semantic_search` uses the same implementation and embeddings server as `/search`.
It reuses embeddings for unchanged files and refreshes changed or deleted files
before returning up to ten results with source locations. If no embeddings-capable
server is available, it reports that semantic search is unavailable.

### PDF documents

`show_file` extracts text from PDF files, including `.PDF` paths, so
the model can answer questions about their contents. Use `/show_file report.pdf`
to load a document into the conversation, then ask a question. Line ranges in
the model tool refer to extracted text lines, not PDF page numbers.
`fetch_url` also extracts text from PDF responses, with its usual `max_chars`
limit. Each page is labelled with its original number and extraction source.
Scanned pages use local OCR when its models are installed. Pages with a small
amount of embedded text and an image also try OCR, which handles scans with
digitally added page numbers. Failed pages are marked unavailable while
readable pages remain usable.

Use page ranges or query keywords for larger documents:

```text
/show_file --start-page 21 --end-page 30 report.pdf
/show_file --query "annual revenue" report.pdf
```

The model tool and `/v1/show_file` accept `start_page`, `end_page` and `query`.
Page numbers start at one and the range includes both endpoints. Ordinary
reads inspect the first 20 pages. Queries inspect up to 200 pages and select
up to six pages by keyword overlap, keeping their original page numbers.
This is keyword retrieval, not semantic search. Specify a starting page to
search another part of a long document. Page selection applies to workspace
files; PDF reads at a Git ref use the default range.

The web console stores a bounded PDF page index alongside the attachment.
Each follow-up question selects up to six stored pages again, without
rerunning OCR. Older saved attachments continue to use their stored text.
Uploads inspect up to 200 pages. At most 20 pages per extraction use OCR;
use narrower page ranges in the file tools for additional scanned pages.
Extraction retains up to 200,000 characters, with 30,000 per page, and each
PDF context includes up to 60,000 text characters. Limits and omissions are
reported in the context. PDF input is limited to 64 MiB, in addition to any
upload-size limit configured by the web server.

#### Local OCR setup

OCR uses the existing `ocrs` engine with `hayro` for PDF rendering. Both run
locally in Rust. Install the official models once in `~/.orangu/ocr/`:

```sh
mkdir -p ~/.orangu/ocr
curl -fL https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx -o ~/.orangu/ocr/text-detection.onnx
curl -fL https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx -o ~/.orangu/ocr/text-recognition.onnx
```

On Windows, the default directory is `%USERPROFILE%\.orangu\ocr`. Set
`ORANGU_OCR_MODEL_DIR` before launching the terminal or server to use another
directory containing those two files. Reading a PDF never downloads models
or sends its contents to a service. Missing or invalid models are reported
on affected pages; embedded text extraction continues to work.
Reattach a document after installing models if it was previously uploaded
without them, so its stored page index includes the OCR results.

The supplied OCR recognition model supports Latin script. OCR text is marked
as potentially inaccurate; it has no fabricated numerical confidence score.
Use a release build for OCR; the inference libraries are much slower without
optimization.
Page images render at a target 200 DPI, bounded to 4096 pixels per edge and
12 million pixels per page. See the [ocrs documentation](https://github.com/robertknight/ocrs)
for model and language support.

`create_file` renders its `content` as a searchable PDF when the output path
ends in `.pdf`, with an embedded font, line wrapping and multiple pages as
needed. For example, ask the model to save its answer to `summary.pdf`. The
same behavior is available through `/v1/create_file`. Existing documents are
replaced when `overwrite` is true. This creates a plain text document;
`/export` remains available for formatted session and review reports.

#### Tables in query context

Table extraction uses `pdfni` with positioned native text or OCR characters.
Bordered tables retain rows, columns, empty cells, and merged-cell spans.
Scanned borders are detected with `imageproc`; table extraction needs no
additional models. Native tables work without the OCR models installed.
Query context includes complete HTML tables with row and column spans, and
keeps narrative text outside the tables. The saved web attachment index
retains this structure for follow-up questions.

Borderless tables use text alignment and are explicitly marked as inferred.
Their merged cells cannot be verified. Ambiguous layouts produce a warning;
flattened text should not be used to guess cell associations. OCR can still
misread characters and numbers. Rotated, skewed, broken-border, nested, and
cross-page tables are not guaranteed. Context and storage limits omit entire
tables rather than cutting cells in half, with an explicit omission notice.
URL character limits and model-tool line ranges also omit tables that would
be cut. Use PDF page ranges to retrieve a complete table.

Damaged or password-protected documents may be unreadable.
Extracted text does not preserve the original page layout.
PDF line edits and text replacement are refused; save a new
document with `create_file` instead. Saving does not preserve an existing PDF's
forms, annotations or layout.

The eight file-lifecycle tools are the same operations `orangu-server`
serves over HTTP as `/v1/create_file`, `/v1/modify_file` and so on — one
shared implementation (`orangu::files`), so a tool call, a typed command and
an API request behave identically. Their full field-by-field schemas are in
the *HTTP endpoints* chapter, under **File-lifecycle API**.

They replace the earlier `read_file` and `edit_file`: `read_file` is now
`show_file` with the same arguments, and `edit_file` is now `modify_file`,
which still takes `old_text`/`new_text` and additionally accepts the
server's `edits` line ranges. The typed `/add_file` is obsolete for the same
reason — it was `/create_file` without content (see the Git commands
chapter). An existing path is overwritten on every
surface — tool, typed command and HTTP endpoint alike; pass
`"overwrite": false` for create-if-absent.

**Licence headers.** A file that `create_file` brings into existence is
generated code, and it is written with **the workspace's own licence** at
the top of it, as a comment in that file's own language — `//` for Rust and the C family, `#`
for shells and configuration formats, `--` for SQL and Lua, `;;` for Lisps,
`%` for TeX and Erlang, `REM` for batch files, and a delimited `<!-- -->`,
`/* */` or `(* *)` block for a language with no line comment. A shebang or
an XML declaration keeps the first line; the licence goes directly beneath
it.

The licence and the copyright holder are read from the workspace: the
`license` field of its `Cargo.toml`, `pyproject.toml` or `package.json`, or
failing that its `LICENSE`/`COPYING` file — used *verbatim* when it is a
licence orangu has no header for. A project that says nothing gets MIT.
**`/license` overrides all of it** for the session, including turning headers
off; see the Core tools chapter.

Three things never get one, and they are the point of the rule:

- **A file that already existed.** Rewriting or editing it is a change to
  somebody's project, and a project's licensing is not the agent's to
  decide — an MIT header must never land on top of a GPL source file.
- **A file you spelled out yourself** with the typed `/create_file`
  command. That is you writing a file, not orangu generating one.
- **A format with nowhere to put a comment** — `.json`, `.csv` — which
  would be corrupted rather than licensed.
- **A workspace with no copyright holder to name.** It is taken from the
  manifest's first author, the `LICENSE` file's own copyright line (for MIT
  and the BSDs, whose licence files name the project rather than a
  foundation), or `git config user.name`. With none to be found there is no
  header, because inventing an attribution is worse than omitting one —
  `/license <spdx> <holder>` supplies one.

Because the header is added on top of what was sent, the file on disk has
more lines than the `content` argument did. The tool result reports this as
`"licensed": true`, and a `modify_file` that names line numbers should read
the file back first. `"licensed": false` means no header was added — which,
now that the licence is the project's, is the ordinary answer in a project
that does not declare one. The licence texts are strings compiled into the
binary rather than files read from disk, and they are shared with the web
console's code blocks, which use the same implementation.

**Git.** In a Git repository these tools make their change *with* the Git
command — `create_file`/`modify_file` stage with `git add`, `move_file`
moves with `git mv`, `delete_file` deletes with `git rm` — so work is
staged as it happens. **Nothing is ever committed**; that stays your
decision (`/commit`). Pass `"git": false` on a call for a plain filesystem
change.

## Workspace restrictions

The tools are rooted in the active workspace. By default this is the current directory, unless `orangu` was started with `--workspace /path/to/project`.

Paths that attempt to escape the workspace are rejected.

Absolute paths are allowed only when they still resolve inside the workspace after normalization.

## `show_file`

`show_file` returns text content with line numbers:

```json
{
  "path": "src/main.rs",
  "start_line": 10,
  "end_line": 20,
  "mode": "full"
}
```

Behavior:

- `path` is required
- `start_line` defaults to line 1
- `end_line` defaults to the end of the file
- `mode` defaults to `full`. Valid modes are `full` (read actual content), `signatures` (extract only public interfaces), or `map` (extract top-level item declarations for an overview).
- Each returned line is prefixed as `N. text` (only applies to `full` mode)
- A picture (PNG, JPEG, GIF or WebP, known by its first bytes, whatever its name) is not shown as text: the tool says it is a picture, with its media type and size. A client that can hand a model the picture itself — the Managed Agents `read` tool on a model with a vision tower — reads its bytes instead
- Repeated unchanged whole-file reads in the same conversation may return a cache stub instead of resending the entire file
- The cache stub means the model should reuse the earlier full content already in context; use `start_line` and `end_line` to request a fresh focused excerpt when needed

## `modify_file`

`modify_file` performs a targeted replacement inside a workspace file:

```json
{
  "path": "src/main.rs",
  "old_text": "fn old_name()",
  "new_text": "fn new_name()"
}
```

Optional flags:

- `replace_all` replaces every match; without it `old_text` must occur exactly once

Important details:

- `path`, `old_text`, and `new_text` are required by the tool schema
- If the file does not exist, it is created (mode `0644`) with `new_text` as its contents
- If `old_text` is empty, the file content is replaced with `new_text`
- If `old_text` is not found in an existing file, the tool returns an error
- If `old_text` occurs more than once and `replace_all` is not set, the tool returns an error saying how many times: the model includes more of the surrounding text, or sets `replace_all`
- Successful edits return JSON with `path`, `created`, `updated`, `replacements` (how many matches were replaced; `0` for a created or wholly replaced file), `original_bytes`, `new_bytes`, `mode`, and `git`
- Instead of `old_text`/`new_text`, `edits` may be given — an array of `{start_line, end_line, replacement}` line ranges, exactly as `orangu-server`'s `/v1/modify_file` takes them. `edits` wins if both are supplied
- In a Git repository the change is staged with `git add`; pass `"git": false` to leave the index alone

## `list_directory`

`list_directory` is a workspace-scoped directory listing tool:

```json
{
  "path": "src",
  "max_depth": 3
}
```

Behavior:

- `path` defaults to `.`
- `max_depth` defaults to `2`
- Each result line is formatted as `kind<TAB>path`
- `kind` is either `dir` or `file`
- Paths are shown relative to the workspace when possible

With `pattern`, a glob over the paths below `path`, it lists the matching
files instead — one workspace-relative path per line, newest first:

```json
{
  "path": "src",
  "pattern": "**/*.rs"
}
```

- `*` and `?` stay within one directory; `**` crosses any number of them
- `.git` is never entered, and directories are not listed
- At most 1000 paths, followed by a `(N more not shown)` line when there are more
- `No files match.` when nothing does

## `fetch_url`

`fetch_url` retrieves external documentation or reference material:

```json
{
  "url": "https://example.com/docs",
  "max_chars": 12000
}
```

Behavior:

- `url` is required
- `max_chars` defaults to `20000`
- HTML responses are converted into readable text
- Non-HTML responses are returned as plain text
- Long responses are truncated and end with `[truncated]`

## `expand_context`

`expand_context` retrieves exact session-cached context that `orangu` kept outside the prompt to save tokens. Most truncation markers point directly to the complete original blob; bounded workflows such as `/create_patch` may instead point to a small index whose entries can be expanded selectively.

```json
{
  "id": "abc1234567"
}
```

Behavior:

- `id` is required and must be a 10-character cache ID.
- You will find these IDs injected into your context as markers (e.g. `[Note: Output truncated. Run expand_context(id="abc1234567")]`).
- The tool returns the exact cached node. This is normally the full, uncompressed original output; a bounded index node contains IDs for smaller exact chunks.
- The cache ID is a SHA-256 hash prefix of the content, meaning it is mathematically guaranteed to perfectly match the truncated payload.
- Cache files are strictly scoped to the active session and automatically deleted when the session ends.

## `run_shell_command`

`run_shell_command` executes a Bash command inside the workspace:

```json
{
  "command": "cargo test --quiet",
  "cwd": "crates/core",
  "timeout_seconds": 60
}
```

Behavior:

- `command` is required
- `cwd` defaults to the workspace root
- `timeout_seconds` defaults to `30`
- The command runs through the platform's shell: `bash -lc` on Linux and macOS, `powershell -NoLogo -NonInteractive -Command` on Windows. It is passed verbatim and never translated between them, so anything beyond the simplest command is written for one or the other
- Output is intercepted and compressed before being sent to the LLM to prevent flooding the context window. Native slash commands (like `/diff`, `/log`, `/build`) execute locally and display their full output to the user, but when this output is injected into the model's context, it is compressed and tracked via a cache identifier to ensure it is only transmitted once if unchanged.
- Output is returned as pretty-printed JSON with `exit_code`, `stdout`, and `stderr`
- `stdout` and `stderr` are each truncated to at most 20,000 characters

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
| `explore_repository` | Have a read-only subagent search the repository | `query` |
| `write_tests` | Have a subagent write and run tests | `task` |
| `write_docs` | Have a subagent write documentation | `task` |
| `write_document` | Have a subagent write a LibreOffice, Markdown or HTML document | `type`, `task` |
| `convert_document` | Render text into a document, or convert a document, for example to PDF | `target`, either `source` or `content` + `format`, optional `overwrite` |
| `run_agent` | Have a custom subagent do a task, or list matching ones; offered only when agents are installed | `agent`, `task` |
| `continue_agent` | Give a finished subagent task a follow-up; it keeps its earlier work. Interactive sessions only | `task`, `message`, optional `background` |
| `research_web` | Have a subagent answer a question from the workspace or the web, with sources; not offered when `[orangu].web_search` is `off` | `question` |
| `use_ssh` | Have a subagent log in to a host over SSH, copy files between it and the workspace, or log out; not offered when `[orangu].ssh` is `deny`. See **SSH** | `task` |

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

**Git.** Nothing is staged automatically: a change is a plain filesystem
change unless the call passes `"git": true`. Then, in a Git repository, these
tools make their change *with* the Git command — `create_file`/`modify_file`
stage with `git add`, `move_file` moves with `git mv`, `delete_file` deletes
with `git rm`. **Nothing is ever committed**; that stays your decision
(`/commit`).

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
- With `"git": true` in a Git repository, the change is staged with `git add`; by default the index is left alone

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

## Subagents

`explore_repository`, `write_tests`, `write_docs`, `write_document`,
`research_web`, `use_ssh` and `run_agent` hand one task to a subagent and return only its
answer, so the main conversation's context stays small. Each subagent:

- runs in a fresh conversation with its own system prompt. It never sees the
  main conversation, so the `query` or `task` must be complete in itself
- gets the model for its role: `explorer` for `explore_repository`, `code` for
  the writers. It starts from the connection the conversation is using, so it
  follows `-c` and `/server`. Behind a coordinator that connection is reused
  and the coordinator picks the model; otherwise a section with that `role` is
  used (the first by name), falling back to that connection
- follows the same `AGENTS.md` rules as the main agent (`~/.orangu/AGENTS.md`
  and the workspace's), except `explore_repository`, which only searches, and
  `use_ssh`, which only copies files
- runs at the same time as the other subagents the model starts in the same
  turn, up to four at once; their answers come back in the order the model
  asked for them. Read-only tools the model starts together in one turn also
  run at the same time; a tool that writes, or a subagent beside a read, keeps
  the order the model asked for. Subagents that write share the workspace, so
  give concurrent writers separate files
- has a fixed budget of model requests, whatever `max_tool_rounds` says
- runs silently: no streaming; `/tasks` shows the tool it is on and the
  tokens it has used
- shares the main agent's MCP tools (`write_tests`, `write_docs` and custom
  agents, as their `tools` allow; `explore_repository`, `research_web`,
  `use_ssh` and `write_document` get none), except the tools that ask for approval
  (`approval_mode = prompt`, or `writes` for a tool not marked read-only):
  nobody is there to approve them, so they are not offered
- can load the installed skills with `use_skill` (`write_tests`, `write_docs`
  and custom agents whose `tools` allow `Skill`): its prompt lists them by
  name and description, and `use_skill` returns one's instructions, user
  skills included
- must end with a `<final_answer>` block, which is all the main agent receives.
  Without one, the raw reply is returned with a warning

| Tool | Role | Rounds | Workspace access |
| :-- | :-- | :-- | :-- |
| `explore_repository` | `explorer` | 6 | read-only |
| `write_tests` | `code` | 12 | read and write; changes only tests and runs them |
| `write_docs` | `code` | 8 | read and write; changes only documentation |
| `write_document` | `code` | 8 | read and write; writes one document |
| `research_web` | `code` | 10 | read-only; `web_search` and `browse` |
| `use_ssh` | `code` | 24 | `ssh_login`, `ssh_copy`, `ssh_logout` and `list_directory` only |
| `run_agent` | by its `model` or `role` | 15 by default | as its `tools` allow |

The writers change the same workspace as the main agent. Their answer lists
every file changed (and, for `write_tests`, the command run and its result).
A subagent that runs out of model requests is asked, without tools, for what
it got done and what is left, and the main agent receives that marked
`[Partial: …]` rather than a bare error; whatever a writer already wrote stays
in the workspace.

Subagents are not offered subagent tools, except custom agents that ask for
them (see `Agent` under Custom agents), so delegation goes at most
`[orangu].subagent_depth` levels (three by default) below the main
conversation.
Read-only sessions such as `/review` offer none of them.

### Background subagents and tasks

Every subagent run is a task, numbered `t1`, `t2`, … per conversation and
listed by `/tasks` (see the Core tools chapter). The main agent receives the
task's id with its answer.

In an interactive session, every subagent tool also takes `background`. With
`background: true` the call returns at once and the main agent carries on.
When the subagent finishes, its report arrives as a message of its own: the
client hands it to the model as the next turn, within a second when idle, or
after the turn that is running. A report never runs as a command, whatever it
starts with, and `/pending` lists it as `(subagent report)` while it waits.
While background tasks run, the status line shows `Tasks: N`.
`Ctrl+B` moves the subagents the main conversation is waiting for to the
background, as if they had been started with `background: true`; the turn goes
on and their reports arrive later. Subagents started by other subagents stay
where they are. With `[orangu].subagent_background = true`, subagents run in
the background unless a call says `background: false`.
`/tasks stop <id>` stops a background task; Esc stops the turn, and with it
any subagent the turn is waiting for. A stopped task is marked so, sends no
report, and cannot be continued.

`/fork <task>` starts a background subagent from the whole conversation
instead of a fresh one (see the Core tools chapter).

`continue_agent` gives a finished task a follow-up `message`. The subagent
picks up its own conversation where it ended, with its earlier work, on the
same model and tools, in the foreground or (`background: true`) in the
background. A task that ran out of model requests can be continued too, to
finish what its partial answer says is left.

Finished tasks are kept with the session, conversations included, so after
orangu is restarted and the session resumed (`--resume`), `/tasks` lists them
and `continue_agent` can still follow one up. A restored task's tools are
built again as they were (a custom agent's from its file as it is now, so one
that was removed cannot be continued), on the connection the configuration
names now. A task that was running when orangu stopped is not kept.

One-shot runs (`--prompt`) and workflow loops have no later turn to deliver a
report in, so there subagents always run in the foreground and
`continue_agent` is not offered. Tasks belong to the session that is running:
they are not kept when orangu exits, and a tab's reports wait while another
tab is in front.

### Documents

`write_document` starts one subagent per document `type`. Its `task` names
what to write and, optionally, where and in which format:

| `type` | Written as | Default target | Also on request |
| :-- | :-- | :-- | :-- |
| `text` | Markdown, rendered by `convert_document` | `.odt` (Writer) | `.docx`, `.pdf` |
| `spreadsheet` | CSV, rendered by `convert_document` | `.ods` (Calc) | `.xlsx`, `.pdf` |
| `presentation` | slide outline, rendered by `convert_document` | `.odp` (Impress) | `.pptx`, `.pdf` |
| `drawing` | SVG, rendered by `convert_document` | `.odg` (Draw) | `.pdf` |
| `markdown` | a plain file, by `create_file` | `.md` | `.pdf` |
| `html` | a plain file, by `create_file` | `.html` | `.pdf` |

The LibreOffice writers never leave a source file behind: the source text goes
straight into the document. When a PDF is asked for as well, they render it
from that same source text.

### Web research

`research_web` answers a `question`, looking locally before going to the web.
Its subagent first checks whether the workspace answers it (code, README,
docs, vendored dependencies). Otherwise it searches with `web_search`, reads
the promising pages with `browse`, checks important claims against a second
source, and ends with the answer and a `Sources:` list of the URLs it relied
on. It can read the workspace but not change it. The search results and pages
stay in the subagent's conversation, so only the answer reaches the main one.

**Where searches go.** `[orangu].web_search` is a comma-separated list of
[SearXNG](https://docs.searxng.org/) instances (by base URL) and public
engines (`duckduckgo`, `bing`, `google`). They are tried in order until one
returns results, but every SearXNG instance comes first, whatever order the
list is written in: a self-hosted instance keeps searches local, and the
public engines are only the fallback.

```ini
[orangu]
# Local SearXNG first, then the public engines.
web_search = http://localhost:8888, duckduckgo, bing, google
```

| `web_search` | Searches |
| :-- | :-- |
| unset | `duckduckgo`, then `bing`, then `google` |
| `http://localhost:8888` | only the local SearXNG instance |
| `http://localhost:8888, duckduckgo, bing` | SearXNG, then DuckDuckGo, then Bing |
| `off` | nothing: `research_web` is not offered |

A SearXNG instance must offer JSON results (`json` under `search.formats` in
its `settings.yml`); it is the same backend `orangu-server`'s `web_search`
server tool uses. The public engines are read from their plain result pages:
DuckDuckGo's lite page and Bing's results page.

Every request identifies itself as `orangu/<version>`. An engine that answers
with a challenge, or with a page that needs JavaScript, counts as unavailable
and the next one is tried; orangu never tries to get past an engine's bot
checks. Google currently answers such a client with a page that needs
JavaScript, so in practice it is the last resort that does not answer.
Querying public engines this way is subject to their terms and rate limits;
a SearXNG instance avoids both.

**`web_search`**, with a `query`, returns numbered results with title, URL and
excerpt, and names the engine that answered.

**`browse`** reads the web a page at a time:

| Arguments | Opens |
| :-- | :-- |
| `url` | that page |
| `link` | link number `link` of the current page |
| `part` | part `part` of the current page (with `url` or `link`: of that page) |

A page is rendered to text once and split into parts of about 6,000
characters. Each part ends with the links it mentions, numbered and made
absolute (anchors into the same page are left out), so the next step is
`link: 7` rather than a copied URL. HTML, plain text, JSON and PDF pages can be
read, up to 5 MB. Every page read stays cached for the rest of the subagent's
run (up to 64 pages), so going back to a page or to another part of it is
answered locally, without a request, and is marked `(cached)`.

Only subagents get `web_search` and `browse`: `research_web`, and custom
agents whose `tools` include `WebSearch` or `WebFetch` (or that have no
`tools` field).

### SSH

`use_ssh` hands a task such as "copy dist/app.tar.gz to /srv/app on build" to a
subagent that can log in to other machines, copy files, and log out. It cannot
run commands on them, and it has no shell.

Its tools:

| Tool | Purpose | Arguments |
| :-- | :-- | :-- |
| `ssh_login` | Open a session, or find the one already open | `host`: an alias from `~/.ssh/config`, or `[user@]host[:port]`, with `[address]:port` for IPv6 |
| `ssh_copy` | Copy one file | `session`, `direction` (`upload` or `download`), `local`, `remote`, optional `overwrite` |
| `ssh_logout` | Close a session | `session` |

**Logging in.** No tool takes a credential, so a password or passphrase never
reaches the model, the transcript or the session files. `ssh_login` tries, in
order, until one works:

1. The keys in your SSH agent (`SSH_AUTH_SOCK`), those matching the host's
   `IdentityFile` first, at most four. Not on Windows.
2. The host's `IdentityFile` entries, or `~/.ssh/id_ed25519`, `id_ecdsa` and
   `id_rsa`. An encrypted key asks for its passphrase.
3. A password, if the server offers password login.

A second factor and keyboard-interactive login are not supported: the error
lists what the server accepts.

**What you are asked.** Questions appear in the input area, hidden while you
type a secret, and also while the subagent works in the background. Esc cancels
one, and one left unanswered for two minutes counts as cancelled.

| Question | When |
| :-- | :-- |
| Allow this conversation to log in to `user@host:port`? | Once per host per conversation, before connecting, unless `[orangu].ssh` is `auto` |
| Trust this host key and add it to `known_hosts`? | The host is not in `known_hosts`, or is there with another key type. Shows the fingerprint |
| Passphrase or password | The login needs one |

`[orangu].ssh` sets the first row:

| Value | Meaning |
| :-- | :-- |
| `prompt` | The default. Ask, and refuse where nobody can answer: `orangu -p` and workflow loops |
| `auto` | Do not ask. Questions from the other rows are still asked where someone can answer; elsewhere an unknown host, an encrypted key or a password login fails |
| `deny` | `use_ssh` is not offered |

**Host keys** are always checked, whatever `StrictHostKeyChecking` says. A key
that differs from the one in `known_hosts`, or one marked `@revoked`, is refused
without asking. Host certificates are not supported. `known_hosts` entries may
be hashed, and `UserKnownHostsFile` is honoured; wildcard entries are not
matched, so such a host counts as unknown.

**`~/.ssh/config`.** `Host`, `HostName`, `User`, `Port`, `IdentityFile`,
`UserKnownHostsFile` and settings before the first `Host` are read. `Include`,
`Match` and the system-wide configuration are not, so a host can resolve
differently than with `ssh`; the login result says so when the file uses either.
A host with `ProxyJump` or `ProxyCommand` is refused rather than connected to
directly.

**Copying.**

- `local` is confined to the workspace, like every file tool. Once you have
  allowed a host, copies to it are not asked about, so any workspace file can
  reach it
- A relative `remote`, or one starting with `~/`, is relative to the remote home
- Naming a directory, or ending the path in `/`, keeps the file's name
- A local parent directory is created when missing; a remote one is not
- An existing file is replaced only with `overwrite`
- One file per call. Directories are refused
- A file is written under a temporary name in the same directory, its size is
  checked, and only then does it take its name. A failed or cancelled copy
  leaves the target as it was. Replacing a remote file deletes the old one just
  before the rename
- The file's mode comes along, without setuid, setgid or sticky bits

**Sessions** belong to the conversation and outlive the subagent, so a later
task needs no second login. One closes on `ssh_logout`, after 15 minutes with no
copy, or when orangu exits. Name the host in the next task to log in again.

## Custom agents

orangu reads custom subagents in the format Claude Code uses, so a collection
such as
[awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
works as it is. Each agent is a Markdown file: YAML frontmatter, then the
agent's system prompt.

```markdown
---
name: api-designer
description: Use this agent when designing new APIs or API specifications.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---

You are a senior API designer...
```

Agent files are read recursively, up to five directories deep, from:

| Directory | Scope |
| :-- | :-- |
| `~/.claude/agents` | user |
| `~/.orangu/agents` | user |
| `<workspace>/.claude/agents` | project |
| `<workspace>/.orangu/agents` | project |
| `agents/` of each plugin | plugin |

Plugins are those Claude Code installed (`~/.claude/plugins/installed_plugins.json`:
every user-scope plugin, and project-scope ones of this workspace, less any
that `enabledPlugins` turns off in Claude Code's settings) and every
directory in `~/.orangu/plugins/` and `<workspace>/.orangu/plugins/`. A
plugin's agents are named `<plugin>:<agent>`, the plugin named by its
`.claude-plugin/plugin.json` or its directory; a model may leave the plugin
out while no other agent has the name. As in Claude Code, a plugin's agents
get no `mcpServers` of their own.

A project agent replaces a user agent of the same name. Agents installed for
Claude Code are picked up without changes, and a whole collection can be
cloned (or symlinked) into one of these directories:

```sh
git clone https://github.com/VoltAgent/awesome-claude-code-subagents \
    ~/.orangu/agents/voltagent
```

A Markdown file without both `name` and `description` (a README) is skipped,
as are hidden directories (`.git`) and directories named `commands`, `tools`,
`skills` or `hooks`, where a collection keeps its other kinds of extensions.
No agent may be called `list`. Frontmatter whose YAML does not parse is read
line by line as `key: value`.

**`tools`** is a comma-separated list or a YAML list. It is an allowlist: the
agent is offered, and may run, only these tools. Without `tools`, the agent
gets every tool. **`disallowedTools`**, in the same form, takes tools away
first, so `disallowedTools: Write, Edit, Bash` makes an agent without `tools`
read-only.

| Claude Code tool | orangu tools |
| :-- | :-- |
| `Read` | `show_file`, `expand_context` |
| `Glob`, `LS` | `list_directory` |
| `Grep` | `search_text` |
| `Write` | `create_file`, `create_directory`, `convert_document` |
| `Edit`, `MultiEdit` | `modify_file`, `move_file`, `move_directory` |
| `Bash` | `run_shell_command`, `delete_file`, `delete_directory` |
| `WebFetch` | `fetch_url`, `browse` |
| `WebSearch` | `web_search`, `browse` (no search when `[orangu].web_search` is `off`) |
| `Skill` | `use_skill` |
| `Agent`, `Task` | the subagent tools: `run_agent`, `explore_repository`, `write_tests`, `write_docs`, `write_document`, `research_web` |
| `Agent(a, b)` | the built-in subagents it names, and `run_agent` for only the custom agents it names |

`ssh_login`, `ssh_copy` and `ssh_logout` have no Claude Code name, and an agent
gets them only by listing them in `tools`: an agent without a `tools` field
does not. A login still asks you what it would ask `use_ssh` (see **SSH**).

orangu's own tool names may be listed too, and MCP tools by their names:
`mcp__<server>__<tool>` for one tool, `mcp__<server>` for all of a server's
(in `disallowedTools` too). Anything else, such as `Task`, is unsupported:
the agent runs without it and is told so, as it is about `WebSearch` when
`[orangu].web_search` is `off`.
`search_text` exists only for custom agents: a regular-expression search of
file contents that respects `.gitignore`, with optional `path`, `glob`,
`ignore_case` and `max_results`.

**`model`** picks the role, and so the model: `haiku` asks for `explorer`, the
small fast model, and anything else (`sonnet`, `opus`, `inherit`) for `code`.
Two fields of orangu's own may be added: `role` (`all`, `code`, `review` or
`explorer`) overrides that, and `max_tool_rounds` sets the agent's budget of
model requests (default 15, at most 50).

**`omitClaudeMd: true`** runs the agent without the project's `AGENTS.md`
rules, which every custom agent follows otherwise.

**`maxTurns`** sets the agent's budget of model requests, like
`max_tool_rounds` (which wins when both are set). An agent that uses them
all returns a partial answer, as above.

**Agents that start agents.** An agent whose `tools` list `Agent` (or
Claude Code's older `Task`) may start subagents itself; `Agent(worker,
researcher)` limits it to those, so `run_agent` refuses and does not list
the others. Nesting is asked for: an agent without a `tools` field starts
none, and `disallowedTools: Agent` forbids it. Subagents go at most
`[orangu].subagent_depth` levels below the main conversation (three by
default); an agent at the last level is told it cannot start more. Every run, at any level, is a task of the session in
`/tasks`. The built-in subagents never start others. With `--agent`, the
same `Agent` entries decide which subagent tools the conversation keeps.

**`hooks`** runs commands around the agent's tool calls, in Claude Code's
shape:

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/run-linter.sh"
```

A hook command runs through the shell in the workspace, with the call as
JSON on its standard input: `hook_event_name`, `tool_name` (Claude Code's
name for the tool, such as `Bash` or `Glob`, when there is one), `tool_input`,
`cwd`, and for `PostToolUse` `tool_response`. `matcher` is a regular
expression for the whole tool name, Claude Code's or orangu's; without one,
or `*`, every tool matches. A `PreToolUse` hook that exits with 2 blocks the
call, and the agent is told its standard error; a `PostToolUse` hook that
exits with 2 adds its standard error to the tool's result. Any other exit
status lets the call go on, and a hook runs for at most 60 seconds. orangu
runs `PreToolUse` and `PostToolUse` command hooks; other events and hook
types are listed by `/agents` as not run. The hooks also apply with
`--agent`.

**`permissionMode: plan`** makes the agent read-only: every tool that
changes the workspace is taken away. orangu's subagents never ask for
permission, so the other modes change nothing.

**`background: true`** runs the agent in the background whenever the
conversation can take its report later (an interactive tab).

**`initialPrompt`** is the first turn of a new conversation run as the
agent with `--agent`, as if typed, so it may be a command or a skill.

**`isolation: worktree`** runs the agent in a temporary git worktree: a
checkout of `HEAD` on a branch of its own (`orangu/<agent>-…`), in the
repository's git directory (`.git/orangu-worktrees/`), so what it changes
stays out of the workspace. A workspace in a subdirectory of the repository
is the same subdirectory of the worktree, and a `project` memory stays the
workspace's. After each run a worktree the agent left as it found it is
removed with its branch, and the report says it changed no files, whatever
the agent itself claims. One with commits or changes stays, and the report
says where and how to merge or remove it. `continue_agent` puts a removed
worktree back. The workspace must be a git repository with a commit.
Isolation applies to agents started by `run_agent`, not to `--agent`.

**`color`** is shown by `/agents`. **`effort`** is read but has no effect:
orangu sends no effort setting to the server.

As in Claude Code, a plugin's agents get no `hooks` and no `permissionMode`.

**`skills`** names skills, as a list or comma-separated, whose instructions
are part of the agent's prompt from the start, whatever its `tools` say. A
skill that is not installed is left out, and the agent is told so.

**`mcpServers`** lists MCP servers for the agent, by name or defined inline:

```yaml
mcpServers:
  - github                  # a server configured as [mcp.github]
  - docs:                   # the agent's own, connected for its runs
      type: http
      url: http://localhost:8300/mcp
```

The servers it lists come with the agent even when `tools` does not name
them. An inline server is connected when the agent starts and needs no
approval; a configured server of the same name wins over it. orangu speaks
Streamable HTTP only: a `stdio` or `sse` server, a name that is not
configured, or a server that cannot be reached is left out, and the agent is
told which.

**`memory`** gives the agent notes that outlive the task:

| `memory` | Directory |
| :-- | :-- |
| `user` | `~/.orangu/agent-memory/<agent>/`, shared by every project |
| `project` | `<workspace>/.orangu/agent-memory/<agent>/`, meant to be committed |
| `local` | `<workspace>/.orangu/agent-memory-local/<agent>/`; add it to `.gitignore` |

The start of the directory's `MEMORY.md` (its first 200 lines, at most
25 KB) is part of the agent's prompt, and the agent rewrites it with
`save_memory` (at most 64 KB), a tool it gets whatever its `tools` say. The
agent is told never to store secrets there.

The main agent reaches custom agents through one tool, `run_agent`, offered
only when at least one agent is installed. `agent` names the agent and `task`
is a self-contained task. With `agent: "list"`, `task` is search words instead,
and the best ten matches come back with their descriptions. A whole
collection therefore costs one tool definition per request, not one per agent.
An unknown name is answered with the closest matches. Like every other
subagent, a custom agent starts in a fresh conversation and is asked to end
with a `<final_answer>` block.

Agent files are read again when they change: an agent added, edited or
removed applies within seconds, without restarting orangu or opening a new
tab. A task already running keeps the agent it started with.

**Agents for one run.** `--agents` defines agents as JSON, Claude Code's
format: an object from agent name to its `description`, its system prompt
as `prompt`, and any of the fields above. They are offered in every tab of
the run and win over agent files of the same name.

```sh
orangu --agents '{"reviewer": {"description": "Reviews code.", "prompt": "You review code for bugs.", "tools": ["Read", "Grep"]}}'
```

**The whole conversation as an agent.** `--agent <name>` runs the main
conversation, interactive or `-p`, as an agent: its prompt (with its memory
and `skills`) joins the system prompt, and its `tools` and `disallowedTools`
limit the main agent's tools. A `tools` list without the subagent tools
leaves the conversation without them. The conversation keeps its connection,
whatever the agent's `model` says.

**Asking for an agent.** `@agent-<name>` in a prompt asks for that agent,
a custom one or a built-in subagent (`@agent-explore_repository`): the model
is told to hand the request to it. Tab completes the names. A mention of an
agent that is not installed is left as typed.

`/agents` lists the installed agents (see the Core tools chapter).

## `convert_document`

`convert_document` writes LibreOffice and Office documents without LibreOffice
installed, through the pure-Rust
[libreoffice-pure](https://crates.io/crates/libreoffice-pure) crate. The format
of `target` comes from its extension. The input is either an existing
workspace file, `source`, or text, `content`, in a `format`:

```json
{
  "target": "budget.ods",
  "content": "Item,Cost\nRent,1200\nFood,400\nTotal,=SUM(B2:B3)\n",
  "format": "csv"
}
```

```json
{
  "source": "budget.ods",
  "target": "budget.pdf"
}
```

| `format` | What it is | Targets |
| :-- | :-- | :-- |
| `md`, `html`, `txt` | a text document | `odt`, `docx`, `pdf`, `html`, `md`, `txt` |
| `csv` | one sheet; the first row is the header, a cell starting with `=` is a formula | `ods`, `xlsx`, `odb`, `pdf`, `html`, `md` |
| `outline` | slides: `# Title` starts one, `- point` adds a bullet | `odp`, `pptx`, `pdf`, `html`, `md` |
| `svg` | a drawing | `odg`, `pdf` |
| `latex`, `mathml` | a formula | `odf`, `pdf` |

Existing documents (`odt`, `docx`, `doc`, `ods`, `xlsx`, `xls`, `odp`, `pptx`,
`ppt`, `odg`, `odb`, and PDF as a source) convert by extension; `format`
overrides the source's extension.

Behavior:

- `target` and `source` must stay inside the workspace
- `overwrite` defaults to `true`
- No Git command is run
- HTML comments in Markdown are dropped, so a licence header in a `.md` file
  never shows up in the document
- Markdown rendered straight to PDF supports Mermaid, math and images by
  relative path
- The result is JSON with `target`, `format`, `bytes_written` and
  `overwritten`
- Fidelity is not LibreOffice's own. In particular, reading an existing `.odp`
  loses the text of its bulleted lists, so render a presentation's PDF from
  its outline rather than from the `.odp`

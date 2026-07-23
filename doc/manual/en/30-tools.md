\newpage

# Tools

`orangu` exposes local workspace tools to the active model.

## Available tools

| Tool | Purpose | Key arguments |
| :-- | :-- | :-- |
| `show_file` | Show a text file from the workspace | `path`, optional `start_line`, optional `end_line`, optional `mode` |
| `create_file` | Create a file with content, optionally with its permissions | `path`, optional `content`, `mode`, `overwrite`, `parents`, `git` |
| `modify_file` | Modify a file, by text replacement or by line ranges | `path`, either `old_text`/`new_text` (+ `replace_all`) or `edits`, optional `git` |
| `move_file` | Move or rename a file | `from`, `to`, optional `mode`, `overwrite`, `parents`, `git` |
| `delete_file` | Delete a file | `path`, optional `git` |
| `create_directory` | Create one directory | `path`, optional `mode`, `parents` |
| `move_directory` | Move a directory and everything under it | `from`, `to`, optional `mode`, `parents` |
| `delete_directory` | Delete an empty directory | `path` |
| `list_directory` | List files and directories below the workspace | optional `path`, optional `max_depth` |
| `fetch_url` | Fetch an external URL and return readable text | `url`, optional `max_chars` |
| `run_shell_command` | Run a shell command inside the workspace | `command`, optional `cwd`, optional `timeout_seconds` |
| `expand_context` | Retrieve previously compressed/truncated output using its hash ID | `id` |

The eight file-lifecycle tools are the same operations `orangu-server`
serves over HTTP as `/v1/create_file`, `/v1/modify_file` and so on — one
shared implementation (`orangu::files`), so a tool call, a typed command and
an API request behave identically. Their full field-by-field schemas are in
the Inference server internals chapter, under **File-lifecycle API**.

They replace the earlier `read_file` and `edit_file`: `read_file` is now
`show_file` with the same arguments, and `edit_file` is now `modify_file`,
which still takes `old_text`/`new_text` and additionally accepts the
server's `edits` line ranges. The typed `/add_file` is obsolete for the same
reason — it was `/create_file` without content (see the Git commands
chapter). An existing path is overwritten on every
surface — tool, typed command and HTTP endpoint alike; pass
`"overwrite": false` for create-if-absent.

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

- `replace_all` replaces every match instead of only the first one

Important details:

- `path`, `old_text`, and `new_text` are required by the tool schema
- If the file does not exist, it is created (mode `0644`) with `new_text` as its contents
- If `old_text` is empty, the file content is replaced with `new_text`
- If `old_text` is not found in an existing file, the tool returns an error
- Successful edits return JSON with `path`, `created`, `updated`, `original_bytes`, `new_bytes`, `mode`, and `git`
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

`expand_context` retrieves massive text blobs (like large file diffs or lengthy command outputs) that `orangu` automatically truncated before they reached your context window to save tokens.

```json
{
  "id": "abc1234567"
}
```

Behavior:

- `id` is required and must be a 10-character cache ID.
- You will find these IDs injected into your context as markers (e.g. `[Note: Output truncated. Run expand_context(id="abc1234567")]`).
- The tool returns the full, uncompressed, raw text of the original output.
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
- The command runs through `bash -lc`
- Output is intercepted and compressed before being sent to the LLM to prevent flooding the context window. Native slash commands (like `/diff`, `/log`, `/build`) execute locally and display their full output to the user, but when this output is injected into the model's context, it is compressed and tracked via a cache identifier to ensure it is only transmitted once if unchanged.
- Output is returned as pretty-printed JSON with `exit_code`, `stdout`, and `stderr`
- `stdout` and `stderr` are each truncated to at most 20,000 characters

\newpage

# Tools

`orangu` exposes local workspace tools to the active model.

## Available tools

| Tool | Purpose | Key arguments |
| :-- | :-- | :-- |
| `read_file` | Read a text file from the workspace | `path`, optional `start_line`, optional `end_line` |
| `edit_file` | Edit a workspace file by replacing text | `path`, `old_text`, `new_text`, optional `replace_all`, optional `create_if_missing` |
| `list_directory` | List files and directories below the workspace | optional `path`, optional `max_depth` |
| `fetch_url` | Fetch an external URL and return readable text | `url`, optional `max_chars` |
| `run_shell_command` | Run a shell command inside the workspace | `command`, optional `cwd`, optional `timeout_seconds` |

## Workspace restrictions

The tools are rooted in the active workspace. By default this is the current directory, unless `orangu` was started with `--workspace /path/to/project`.

Paths that attempt to escape the workspace are rejected.

Absolute paths are allowed only when they still resolve inside the workspace after normalization.

## `read_file`

`read_file` returns text content with line numbers:

```json
{
  "path": "src/main.rs",
  "start_line": 10,
  "end_line": 20
}
```

Behavior:

- `path` is required
- `start_line` defaults to line 1
- `end_line` defaults to the end of the file
- Each returned line is prefixed as `N. text`

## `edit_file`

`edit_file` performs a targeted replacement inside a workspace file:

```json
{
  "path": "src/main.rs",
  "old_text": "fn old_name()",
  "new_text": "fn new_name()"
}
```

Optional flags:

- `replace_all` replaces every match instead of only the first one
- `create_if_missing` allows a missing file to be created

Important details:

- `path`, `old_text`, and `new_text` are required by the tool schema
- If `old_text` is empty, the file content is replaced with `new_text`
- If `old_text` is not found, the tool returns an error
- Successful edits return JSON with `path`, `updated`, `original_bytes`, and `new_bytes`

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
- Output is returned as pretty-printed JSON with `exit_code`, `stdout`, and `stderr`
- `stdout` and `stderr` are each truncated to at most 20,000 characters

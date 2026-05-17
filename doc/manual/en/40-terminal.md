\newpage

# Terminal interface

`orangu` is an interactive terminal client with a persistent header and a prompt area anchored to the bottom of the terminal.

## Header

The top banner displays:

- Current version
- Workspace status
- Server status
- Model status
- `/help` reminder

## Prompt area

The prompt area stays at the bottom of the terminal window.

- Long input wraps upward
- Submitted input moves directly into the output area
- The banner and prompt stay fixed while the output window scrolls independently
- Markdown in assistant output is rendered with terminal styling when possible, including emphasis, strong text, lists, headings, links, and code
- Fenced code blocks with a language tag such as ```c use syntax highlighting in the terminal when the language is supported by the bundled highlighter

## Waiting state

While the model is generating a response, the left side of the footer shows a rolling:

```text
Thinking (2s)
```

status indicator.

You can keep typing and submitting commands while a response is pending. Submitted commands are queued and executed in order after the active response finishes.

When a profile uses `provider = llama.cpp`, the footer starts with `Thinking (<CLOCK>)` and switches to llama.cpp's native generation throughput once tokens are streaming, for example `Working @ 42.5 t/s (2s)`.

Press `Esc` twice within 2 seconds during the waiting state to cancel the active request without exiting the client. Queued commands are preserved.

## History and navigation

Command history is stored in:

```text
~/.orangu/orangu.history
```

Use:

- `<ARROW_UP>` to move backward in history
- `<ARROW_DOWN>` to move forward in history

## Local commands

All slash commands are handled locally. They are not sent to the model.

| Command | Description |
| :-- | :-- |
| `/help` | Show available commands |
| `/connect [url]` | Connect to the configured server, or a specific server |
| `/disconnect` | Disconnect from the current server |
| `/reload` | Restore the configured model and server |
| `/list_models` | List models |
| `/list_files` | List workspace files as a tree |
| `/show_file [--hash] [--author] <path>` | Show a file with optional Git metadata |
| `/tools` | List tools |
| `/model [name]` | Switch to the configured model, or a specific model |
| `/diff` | Show a color unified diff against the current branch |
| `/open_file <path>` | Open a workspace file in $EDITOR |
| `/clear` | Clear the current conversation |
| `/quit` | Exit the client |

Local commands continue to work even when the model is unavailable.

Free-form prompts are blocked when the server or model status in the header is red.

## Command notes

- `/tools` lists the model-facing workspace tools described in the tools chapter
- `/open_file <path>` is workspace-scoped; paths outside the workspace are rejected
- `/show_file [--hash] [--author] <path>` is workspace-scoped, syntax highlights the file content, and can prepend Git blame hash and author columns when the workspace is a Git repository
- `/list_files` is a local convenience command and is separate from the model-facing `list_directory` tool
- `/reload` also clears the current conversation history in memory
- `/quit` exits immediately, while `Ctrl+C` uses a two-step confirmation
- Unknown slash commands are handled locally and produce an error message that points back to `/help`

## Natural-language command aliases

Local commands can also be entered in plain language. Examples:

- `open README.md`
- `list models`
- `list files`
- `show tools`
- `show help`
- `switch model to <name>`

Natural-language forms are recognized only for the built-in local command phrases. Ordinary prompts continue to go to the model.

## Comments and ignored input

- If the first non-whitespace character is `#`, the line is treated as a local comment, shown in the transcript, and not sent to the LLM
- If the first non-whitespace character is `\`, the line is ignored

## Shortcuts and keys

### Prompt editing

- `Ctrl+A` or `Home` moves the cursor to the start of the input line
- `Ctrl+E` or `End` moves the cursor to the end of the input line
- `Left` moves the cursor one character left
- `Right` moves the cursor one character right
- `Backspace` deletes the character to the left of the cursor
- `Delete` deletes the character under the cursor
- `Ctrl+D` behaves like `Delete`; when the input is empty it exits the client immediately
- `Ctrl+K` deletes from the cursor to the end of the line
- `Ctrl+U` deletes from the start of the line to the cursor
- `Ctrl+W` deletes the word immediately before the cursor
- Pasted text is inserted at the current cursor position

### History and completion

- `<ARROW_UP>` moves backward through command history
- `<ARROW_DOWN>` moves forward through command history
- History navigation preserves the current unfinished line as a draft and restores it when you move back out of history

### Tab completion

`Tab` uses context-sensitive completion. The first `Tab` inserts the first match. Repeated `Tab` presses cycle through the remaining matches for the same completion range.

Completion cycling is reset as soon as you edit the line, move the cursor, paste text, or otherwise change the input.

The completion modes are:

1. If the line starts with `/`, complete built-in slash commands such as `/help`, `/list_models`, `/list_files`, `/show_file`, `/tools`, and `/quit`.
2. If the line starts with `/model `, complete configured model profile names.
3. If the line starts with `/open_file ` or `/show_file `, complete workspace file paths recursively. `/show_file` also completes `--hash` and `--author`.
4. If the line starts with the natural-language prefixes `open `, `open file `, `edit `, or `edit file `, complete workspace file paths recursively.
5. Otherwise, complete filesystem entries from the current token relative to the workspace, using the token before the cursor.

Path-completion details:

- General filesystem completion lists entries from the matching directory level and appends `/` to directories
- `/open_file`, `/show_file`, and the natural-language open/edit forms search recursively through the workspace
- Recursive file completion matches either the full relative path or, when no `/` is present in the token, the file name
- Quoted file completion is supported for `/open_file "..."`, `/show_file "..."`, and `open "..."`; the inserted completion keeps the opening quote
- Completion skips `.git`, `build`, and `target` content
- Completion also skips paths ignored by the workspace `.gitignore`

### Output scrolling

- `Shift+PageUp` scrolls backward through the output window
- `Shift+PageDown` scrolls forward through the output window
- The output scrollback buffer keeps the most recent 10,000 lines
- Scrolling is limited to the output window; it does not replace the header or prompt area

### Waiting and exit control

- `Esc` twice within 2 seconds cancels the active request without exiting and keeps queued commands
- `Ctrl+C` once arms quit mode, shows a warning in the transcript, and clears the current input line
- `Ctrl+C` again within 2 seconds exits the client
- `Enter` submits the current input line

## Footer behavior

- The left side of the footer shows `Thinking (<CLOCK>)` while waiting for a response to start, and `Working @ X.Y t/s (<CLOCK>)` while tokens are streaming
- The center side of the footer shows `Pending: X` to show how many queued commands are waiting
- The right side of the footer shows the model name used

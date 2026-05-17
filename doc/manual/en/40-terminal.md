\newpage

# Terminal interface

`orangu` is an interactive terminal client with a persistent header and a prompt area anchored to the bottom of the terminal.

## Header

The top banner displays:

- current version
- workspace status
- server status
- model status
- `/help` reminder

## Prompt area

The prompt area stays at the bottom of the terminal window.

- Long input wraps upward
- The model name is right-aligned below the prompt frame
- Submitted input moves directly into the output area
- The banner and prompt stay fixed while the output window scrolls independently

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
| `/help` | Show the local command reference |
| `/connect` | Reconnect to the configured endpoint of the active model profile |
| `/connect <url>` | Set a specific current server target |
| `/disconnect` | Disconnect from the current server target |
| `/reload` | Restore the startup model and configured server target and clear the current conversation |
| `/list_models` | Show configured model profiles |
| `/list_files` | Show a recursive Unicode tree of the workspace, excluding `.git/`, `build/`, and `target/` |
| `/tools` | Show the tool definitions exposed to the model |
| `/model` | Show a reminder to use `/list_models` |
| `/model <name>` | Switch to a configured model profile |
| `/diff` | Show a colorized unified Git diff for the current workspace |
| `/open_file <path>` | Open a workspace file in the editor defined by `$EDITOR` |
| `/clear` | Clear the current conversation and reset the system prompt for the active profile |
| `/quit` | Exit the client |

Local commands continue to work even when the model is unavailable.

Free-form prompts are blocked when the model status in the header is red.

## Command notes

- `/tools` lists the model-facing workspace tools described in the tools chapter
- `/open_file <path>` is workspace-scoped; paths outside the workspace are rejected
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

- `Ctrl+A`
- `Ctrl+E`
- `Ctrl+K`
- `Ctrl+U`
- `Ctrl+W`
- `Home`
- `End`
- `Left`
- `Right`

### History and completion

- `<ARROW_UP>` history backward
- `<ARROW_DOWN>` history forward
- `Tab` completion for slash commands and `/model`
- File completion across the project for paths, including `/open_file <path>` and `open <path>`
- File completion skips `.git` content and paths ignored by the workspace `.gitignore`

### Output scrolling

- `Shift+PageUp` scrolls backward through the output window
- `Shift+PageDown` scrolls forward through the output window
- The output scrollback buffer keeps the most recent 10,000 lines
- Scrolling is limited to the output window; it does not replace the header or prompt area

### Waiting and exit control

- `Esc` twice within 2 seconds cancels the active request without exiting and keeps queued commands
- `Ctrl+C` once arms quit mode
- `Ctrl+C` again within 2 seconds exits the client

## Footer behavior

- The footer centers `Pending: X` to show how many queued commands are waiting
- The left side of the footer shows `Thinking (<CLOCK>)` while waiting for a response to start
- For llama.cpp profiles, the left side switches to `Working @ X.Y t/s (<CLOCK>)` while tokens are streaming

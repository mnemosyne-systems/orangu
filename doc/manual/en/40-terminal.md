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

While the model is generating a response, the output area shows a rolling:

```text
Thinking (2s)
```

placeholder in the position where the reply will appear.

You can keep typing and submitting commands while a response is pending. Submitted commands are queued and executed in order after the active response finishes.

## History and navigation

Command history is stored in:

```text
~/.orangu/orangu.history
```

Use:

- `<ARROW_UP>` to move backward in history
- `<ARROW_DOWN>` to move forward in history

## Connection commands

`orangu` supports runtime server target control:

- `/connect` reconnects to the configured endpoint of the active model profile
- `/connect <url>` sets a specific current server target
- `/disconnect` disconnects from the current server target
- `/reload` restores the startup model and configured server target and clears the current conversation

## Local editor command

Use `/open_file <path>` to launch a workspace file in the editor configured by `$EDITOR`.

## Natural-language command aliases

Local commands can also be entered in plain language. Examples:

- `open README.md`
- `list models`
- `show tools`
- `show help`
- `switch model to <name>`

## Comments and ignored input

- If the first non-whitespace character is `#`, the line is treated as a local comment, shown in the transcript, and not sent to the LLM
- If the first non-whitespace character is `\`, the line is ignored

## Editing keys

The prompt supports standard shell-style editing:

- `Ctrl+A`
- `Ctrl+E`
- `Ctrl+K`
- `Ctrl+U`
- `Ctrl+W`
- `Home`
- `End`
- `Left`
- `Right`
- `Tab` completion for slash commands and `/model`
- File completion across the project for paths, including `/open_file <path>` and `open <path>`
- File completion skips `.git` content and paths ignored by the workspace `.gitignore`
- `Shift+PageUp` scrolls backward through the output window
- `Shift+PageDown` scrolls forward through the output window
- The output scrollback buffer keeps the most recent 10,000 lines
- Scrolling is limited to the output window; it does not replace the header or prompt area
- The footer centers `Pending: X` to show how many queued commands are waiting

Press `Ctrl+C` once to arm quit mode. Press it again within 2 seconds to exit.

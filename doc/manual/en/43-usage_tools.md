\newpage

# Usage tools

The usage tools cover the session-level housekeeping commands: reading the built-in manual, inspecting how much the current session has used, clearing the conversation, and leaving the client. Like the other local commands, they are handled locally and keep working even when the server or model status in the header is red.

\newpage

## /manual

Opens this manual in a full-screen, two-pane viewer: the text of the selected section in the left pane and the table of contents in the right pane. The manual text is embedded into the binary at compile time, so it is always available — no external files are read. Pressing `Alt+S` opens a search window that searches the entire manual (`Enter` jumps to the next match, `Esc` closes it).

The viewer, its layout, and its key bindings are described in the Built-in manual section of the Terminal interface chapter.

It is handled locally and works regardless of server or model state.

### Examples

```text
/manual
```

Natural-language forms:

```text
manual
show manual
open manual
```

\newpage

## /usage

Shows usage statistics for the current session.

The report covers:

- total application time,
- total time spent waiting for LLM responses,
- total tool execution time,
- total tokens generated (counted with the bundled tokenizer),
- average tokens per second,
- context cache statistics (reads, hits, misses, rate, bytes saved), and
- context compression statistics (lines saved, patterns applied).

### Examples

```text
/usage
```

Natural-language forms:

```text
usage
show usage
```

\newpage

## /clear

Clears the current conversation.

The in-memory conversation history is dropped so the next prompt starts a fresh exchange. The session itself is preserved — only the conversation context is cleared. (To also return to the configured model and server, use `/reload`.)

### Examples

```text
/clear
```

Natural-language forms:

```text
clear
clear conversation
reset conversation
```

\newpage

## /quit

Exits the client.

`/quit` exits immediately, whereas `Ctrl+C` uses a two-step confirmation (press it once to arm quit mode, then again within two seconds to exit). On exit the full resume command is printed — for example:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

so you can return to exactly this session later. The resume command is not printed when the session had no LLM interaction (zero tokens generated) and was on `main`, `master`, or outside a Git repository; in that case the session directory is deleted silently. See the Sessions section of the Terminal interface chapter for the full cleanup rules.

### Examples

```text
/quit
```

Natural-language forms:

```text
quit
exit
```

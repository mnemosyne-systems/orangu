\newpage

# Workspaces

A *workspace* is the project directory orangu is open on. It is the directory the model's tools read from and write to, the directory whose Git branch drives the prompt and the Git and review commands, and the directory a session belongs to. By default the workspace is the directory you launched orangu in; pass `--workspace <path>` (or `-w <path>`) at startup to begin somewhere else.

Several workspaces can be open at once as tabs in a single orangu, instead of running one instance per project. Each tab is its own session — its own conversation, scrollback, pending queue and command history — and its own active server and model. A `/server` or `/model` command in one tab does not affect any other tab. The chosen server and model are saved to the session's settings file automatically, so they are restored the next time that session is resumed. Switching tabs returns to exactly where you left that workspace, or resumes the session it last had. A tab bar shows the open tabs with the active one in bold, placed where the `workspaces` setting puts it (see Tab placement); it appears once more than one workspace is open. When `feedback` is on, a colored dot precedes each tab number: green for a valid tab, red when the tab's branch has been deleted or changed underneath it, and a blinking white dot on the active tab while it is processing a response. See the Sessions section of the Terminal interface chapter for how sessions are matched to a workspace and branch.

Pass `-a` or `--all` at startup to reopen the tabs that were open at the end of the previous run. Their workspace directories and auto-resumed sessions are restored behind the initial tab.

You move between tabs with the `/workspace` command or the workspace key bindings, and choose where the tab bar sits with the `workspaces` configuration key — both are described below.

\newpage

## /workspace

With no argument, `/workspace` reports the active workspace directory.

A bare number switches to the open tab with that number. Anything else is treated as a directory path: orangu switches to the tab already open on it, or opens a new tab for it (resuming the matching session there, or starting a fresh one). `Alt+Insert` opens a fresh tab on the current directory immediately; you then re-point it with `/workspace <path>`.

Tab completion offers the workspaces seen in earlier sessions first, then completes filesystem directories, so a directory that has never been opened can still be navigated to a segment at a time:

```text
/workspace ~/pro<Tab>/ora<Tab>
```

It is handled locally and works regardless of server or model state.

### Examples

```text
/workspace
/workspace 1
/workspace ~/projects/orangu
```

Natural-language forms:

```text
workspace
switch workspace
workspace 1
switch workspace ~/projects/orangu
```

\newpage

## /create_workspace and /delete_workspace

`/create_workspace <dir>` opens a new tab on an existing directory — the slash-command equivalent of `Alt+Insert` followed by `/workspace <dir>`. The directory must already exist; orangu resolves it, opens a tab on it (resuming the matching session there, or starting a fresh one), and switches to it. A missing or non-existent directory is reported as an error.

`/delete_workspace` closes the active tab — the slash-command equivalent of `Alt+Delete`. The remaining tabs are renumbered and focus moves to a neighbour. The last open tab is never closed; use `/quit` to leave orangu.

Both are handled locally and work regardless of server or model state.

### Examples

```text
/create_workspace ~/projects/orangu
/delete_workspace
```

Natural-language forms:

```text
create workspace ~/projects/orangu
delete workspace
```

\newpage

## Key bindings

At the prompt, the workspace tabs are driven by these keys:

| Key | Action |
| :-- | :-- |
| `Alt+,` | Switch to the previous tab (to the left), wrapping |
| `Alt+.` | Switch to the next tab (to the right), wrapping |
| `Alt+Insert` | Open a new workspace tab (re-point it with `/workspace`) |
| `Alt+Delete` | Close the active tab |

Closing a tab renumbers the ones after it and moves focus to a neighbour. The last open tab is never closed — only `/quit` ends orangu. The keys work at the prompt and during streaming: pressing one while the model is responding parks the stream in the background and switches immediately. The response keeps running in the original tab; a blinking dot in the tab bar shows it is still active. Switching back re-attaches the live view so you can watch it complete. Press Escape twice to cancel a response whether it is running in the foreground or in the background.

\newpage

## Tab placement

Where the workspace tab bar is shown is set by the `workspaces` key in the `[orangu]` section of the configuration file:

```ini
[orangu]
workspaces = top
```

The options, matched case-insensitively, are:

- `top` (the default) — a horizontal bar across the top of the screen, above the banner.
- `bottom` — a horizontal bar across the bottom, under the input.
- `left` — a vertical column of tab numbers down the left edge, beside the banner and output.
- `right` — the same column down the right edge.

The interactive `--init` wizard prompts for it with Tab completion. Leaving the key out uses `top`; an unrecognised value is rejected when the configuration is loaded. The full list of `[orangu]` keys is in the Configuration chapter.

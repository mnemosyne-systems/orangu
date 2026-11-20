# orangu for VS Code

An extension that brings [orangu](https://github.com/mnemosyne-systems/orangu)
into VS Code (1.95 and later), and into editors built on it such as
VSCodium and Cursor. It does no inference and has no agent of its own: it
runs the `orangu` binary, which talks to `orangu-server` or
`orangu-coordinator` the way `orangu.conf` says. Whatever orangu can do in
the terminal, it can do here.

The *IDE* chapter of the manual
([`doc/manual/en/53-ide.md`](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/53-ide.md))
is the user guide; this file is also about the source.

## What it adds

| Where | What it does |
|:---|:---|
| **orangu** in the activity bar | The orangu panel: a prompt box and the answers, with **Review Changes**, **Open orangu**, **Stop** and **Clear** |
| **orangu: Open orangu** (also in the panel's title bar) | Starts orangu in the integrated terminal, in the workspace folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection (or the cursor's line) to the next prompt in the panel, with its file and line numbers |
| **Right click in the editor > Review This File with orangu** | Runs orangu's `/auto_review` on the file |
| **orangu: Review Changes with orangu** | Runs `/auto_review` on the changes on the branch |
| **Settings > orangu > Path** (`orangu.path`) | The path of the `orangu` binary, when it is not on the `PATH` |

Each prompt in the panel runs `orangu -w <workspace> -p <prompt>`, and the
answer streams in as orangu writes it. Commands work as in `orangu -p`:
`/status`, `/log`, `/diff`, "show git status", skills and so on. A prompt
stands on its own: nothing is remembered from the one before it. For a
conversation, or for `/review`, which needs the terminal, use
**Open orangu**.

The workspace is the workspace folder of the active file, or the first
folder. With only a file open, it is the Git repository the file is in,
or else the file's folder. The extension runs where the workspace is, so
over Remote - SSH, WSL and Dev Containers it uses the `orangu` installed
on the remote side.

## Before you start

- orangu installed (`cargo install --path .` from the orangu repository
  puts it in `~/.cargo/bin`, which the extension also looks in), and
- an `orangu.conf` it can find, in the workspace folder or in
  `~/.orangu/`. Run `orangu --init` once to write one.

## Building

With Node.js 20 or later:

```sh
cd contrib/vscode
npm ci
npm run compile
npm run test:unit
npm run test:vscode
npm run package
```

`npm run test:vscode` downloads VS Code and runs the integration tests
inside it; on Linux without a display, run it under `xvfb-run -a`.
`npm run package` writes `orangu-<version>.vsix`; install it with
**Extensions > ... > Install from VSIX...**, or
`code --install-extension orangu-<version>.vsix`. To try changes, open
`contrib/vscode` in VS Code and press **F5**.

## What is here

| Path | What it is |
|:---|:---|
| `src/extension.ts` | The commands, the terminal, and finding the workspace and the binary for them |
| `src/panel.ts` | The panel: its page, and running a prompt in it |
| `src/transcript.ts` | The panel's state apart from VS Code: prompts, answers, the status line and the selection |
| `src/locator.ts`, `src/workspace.ts` | Finding orangu, and the folder it runs in |
| `src/commandLines.ts`, `src/runner.ts`, `src/ansi.ts` | orangu's two command lines, and running a one-shot with its output streamed and its escapes removed |
| `src/prompts.ts` | The prompts the editor commands send |
| `media/` | The panel's script and styles, and the icons |
| `test/unit` | Tests of everything that does not need VS Code, including runs of a stand-in orangu: streaming, failures, Stop and a missing binary |
| `test/integration` | Tests in a real VS Code: the commands, the view and the setting are registered, and each command end to end against the stand-in |

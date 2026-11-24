\newpage

# IDE

`orangu` can be used from inside an IDE. The plugin is a thin layer over
the `orangu` binary: it does no inference and keeps no agent of its own,
so the server, the model, the tools and the skills are the ones
`orangu.conf` sets up, exactly as in the terminal.

## Visual Studio

The Visual Studio extension works in Visual Studio 2022 (17.14 and later)
and Visual Studio 2026. It is built on the VisualStudio.Extensibility SDK,
so it runs in its own process and cannot slow down or crash Visual Studio.
Its source is in `contrib/visualstudio`.

### Installing

Every release has the extension as `orangu-<version>-visualstudio.vsix`
beside the binary archives. Open it (or run `VSIXInstaller.exe` on it),
and restart Visual Studio.

The extension needs `orangu` itself:

- `orangu.exe` from the Windows archive of the same release, on the `PATH`,
  in `%USERPROFILE%\.cargo\bin` (where `cargo install --path .` puts it), or
  wherever **Tools > Options > orangu** says.
- An `orangu.conf` that `orangu` can find: in the solution folder, or in
  `%USERPROFILE%\.orangu\`. Run `orangu --init` once to write one.

### What it adds

| Where | What it does |
|:---|:---|
| **Tools > Open orangu** | Starts `orangu` in its own window, in the solution folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **View > Other Windows > orangu** (also under **Tools**) | The orangu tool window: a prompt box and the answers |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection to the next prompt in the tool window |
| **Right click in the editor > Review This File with orangu** | Asks for a review of the file in the editor |
| **Tools > Options > orangu** | The path of `orangu.exe` |

### The tool window

Type a prompt and press **Ask**. It runs as

```
orangu -w <workspace> -p "<prompt>"
```

and the answer streams in as `orangu` writes it. The line under the buttons
shows what `orangu` is doing (the server it talks to, the tools it calls)
and, at the end, how the run went. **Stop** ends the run and everything it
started, and **Clear** empties the window.

Anything `orangu -p` takes works here: a question, a request to change
code, a command such as `/status`, `/log` or `/diff`, or a skill. A prompt
stands on its own; nothing is remembered from the one before it. For a
conversation, and for `/review` and `/auto_review`, which need the
terminal, press **Open orangu**.

### Asking about code

**Ask orangu About This Selection** attaches the selection (or, with
nothing selected, the line the caret is on) to the next prompt. The tool
window shows it as `src/Parser.cs:12-14` with a **Remove** button, and
fills the prompt box with *Explain what this code does.* when it is empty.
Change the question and press **Ask**: the model gets the question, the
file and the lines, and the code itself, and can open the rest of the file
with its own tools.

**Review This File with orangu** asks for a review of the file: bugs,
risky code, missing error handling and missing tests, each with its line
number and a suggested fix, and no changes to the file. `orangu` reads
the file from disk, so a file with unsaved changes is reviewed as it was
last saved, which the tool window points out.

### The workspace

`orangu` runs in the solution's folder. With no solution, only a file, it
runs in the Git repository the file is in, or else in the file's folder.
That folder is also where `orangu` looks for `orangu.conf` first.

### Settings

**Tools > Options > orangu** has one setting, **orangu binary**: the path
of `orangu.exe`, or the folder it is in. Left empty, the extension uses the
first `orangu.exe` on the `PATH`, then the one in
`%USERPROFILE%\.cargo\bin`. A path that is set but does not exist is
reported rather than replaced by another binary.

### Building it

`contrib/visualstudio` builds with the dotnet SDK, version 8 or later, on
Windows, Linux or macOS:

```sh
cd contrib/visualstudio
dotnet build
dotnet test
```

The `.vsix` is packed by the Visual Studio SDK's tools, which are Windows
programs: on Windows with Visual Studio installed, `dotnet build` packs it
too, and elsewhere the build compiles the extension and stops there. In
Visual Studio, open `Orangu.VisualStudio.sln`; **F5** starts the extension
in an experimental instance.

## VS Code

The VS Code extension works in VS Code 1.95 and later, and in editors
built on it such as VSCodium and Cursor, on Linux, macOS and Windows. Its
source is in `contrib/vscode`.

### Installing

Every release has the extension as `orangu-<version>-vscode.vsix` beside
the binary archives. Install it with **Extensions > ... > Install from
VSIX...**, or

```sh
code --install-extension orangu-<version>-vscode.vsix
```

The extension needs `orangu` itself:

- `orangu` from the archive of the same release, on the `PATH`, in
  `~/.cargo/bin` (where `cargo install --path .` puts it), or wherever the
  `orangu.path` setting says.
- An `orangu.conf` that `orangu` can find: in the workspace folder, or in
  `~/.orangu/`. Run `orangu --init` once to write one.

The extension runs where the workspace is. Over Remote - SSH, WSL and Dev
Containers it starts the `orangu` installed on the remote side, with the
`orangu.conf` there. It is only started in a trusted workspace, since
`orangu` reads the workspace's `orangu.conf`.

### What it adds

| Where | What it does |
|:---|:---|
| **orangu** in the activity bar | The orangu panel: a prompt box and the answers |
| **orangu: Open orangu** (also in the panel's title bar) | Starts `orangu` in the integrated terminal, in the workspace folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection to the next prompt in the panel |
| **Right click in the editor > Review This File with orangu** | Runs `/auto_review` on the file in the editor |
| **orangu: Review Changes with orangu** (also in the panel) | Runs `/auto_review` on the changes on the branch |
| **Settings > orangu > Path** | The path of `orangu` |

### The panel

Type a prompt and press **Enter** (or **Ask**). **Shift+Enter** starts a
new line. It runs as

```
orangu -w <workspace> -p "<prompt>"
```

and the answer streams in as `orangu` writes it. The line under the buttons
shows what `orangu` is doing (the server it talks to, the tools it calls)
and, at the end, how the run went. **Stop** ends the run and everything it
started, and **Clear** empties the panel.

Anything `orangu -p` takes works here: a question, a request to change
code, a command such as `/status`, `/log` or `/diff`, or a skill. A prompt
stands on its own; nothing is remembered from the one before it. For a
conversation, and for `/review`, which needs the terminal, press
**Open orangu**. Pressing it again shows the same terminal rather than
starting a second `orangu`.

### Asking about code

**Ask orangu About This Selection** attaches the selection (or, with
nothing selected, the line the cursor is on) to the next prompt. The panel
shows it as `src/parser.rs:12-14` with a **Remove** button, and fills the
prompt box with *Explain what this code does.* when it is empty. Change
the question and press **Enter**: the model gets the question, the file
and the lines, and the code itself, and can open the rest of the file with
its own tools.

### Reviews

**Review This File with orangu** runs `orangu -p "/auto_review <file>"`,
the same review as in the terminal, with its report in the panel. On
`main` or `master` it reviews the whole file; on any other branch, the
file's changes against the default branch, so a file the branch does not
change is refused with `orangu`'s own message. `orangu` reads the file from
disk, so a file with unsaved changes is reviewed as it was last saved,
which the panel points out.

The file is passed relative to the workspace. `orangu` reads an argument
with `*`, `?`, `[` or `{` in it as a pattern, so a path such as
`app/[id]/page.tsx` is sent as a pattern that matches only it. A space in
a path is sent as `?`, so in the rare case of two files whose names differ
only there, both are reviewed.

**Review Changes with orangu** runs `orangu -p "/auto_review"` on
everything the branch changes. Both need a Git repository and a branch
that is up to date with the default branch, as in the terminal. A review
that rejects the changes ends as failed, with the verdict on the status
line.

### The workspace

`orangu` runs in the workspace folder of the file in the editor; with
several folders open, each file gets its own. With no file in the editor,
it runs in the first folder, and with only a file open and no folder, in
the Git repository the file is in, or else in the file's folder. That
folder is also where `orangu` looks for `orangu.conf` first.

### Settings

**Settings > orangu > Path** (`orangu.path`) is the path of the `orangu`
binary, or the folder it is in. Left empty, the extension uses the first
`orangu` on the `PATH`, then the one in `~/.cargo/bin`. A path that is set
but does not exist is reported rather than replaced by another binary.

### Building it

`contrib/vscode` builds with Node.js 20 or later:

```sh
cd contrib/vscode
npm ci
npm run compile
npm run test:unit
npm run test:vscode
npm run package
```

`npm run test:vscode` downloads VS Code and runs the integration tests in
it; on Linux without a display, run it under `xvfb-run -a`. `npm run
package` writes the `.vsix`. To try a change, open `contrib/vscode` in VS
Code and press **F5**.

## IntelliJ

The IntelliJ plugin works in every JetBrains IDE from 2024.3 on: IntelliJ
IDEA, RustRover, CLion, PyCharm, GoLand, WebStorm and the others, on Linux,
macOS and Windows. It only uses the base platform and the terminal that
every one of them has. Its source is in `contrib/intellij`.

### Installing

Every release has the plugin as `orangu-<version>-intellij.zip` beside the
binary archives. Install it with **Settings > Plugins**, the gear menu at
the top, **Install Plugin from Disk...**, picking the `.zip` as it is.

The plugin needs `orangu` itself:

- `orangu` from the archive of the same release, on the `PATH`, in
  `~/.cargo/bin` (where `cargo install --path .` puts it), or wherever
  **Settings > Tools > orangu** says.
- An `orangu.conf` that `orangu` can find: in the project folder, or in
  `~/.orangu/`. Run `orangu --init` once to write one.

### What it adds

| Where | What it does |
|:---|:---|
| **orangu** tool window, on the right | A prompt box and the answers |
| **Tools > Open orangu** (also in the tool window) | Starts `orangu` in the IDE's terminal, in the project folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection to the next prompt in the tool window |
| **Right click in the editor > Review This File with orangu** | Runs `/auto_review` on the file in the editor |
| **Tools > Review Changes with orangu** (also in the tool window) | Runs `/auto_review` on the changes on the branch |
| **Settings > Tools > orangu** | The path of `orangu` |

### The tool window

Type a prompt and press **Enter** (or **Ask**). **Shift+Enter** starts a
new line. It runs as

```
orangu -w <workspace> -p "<prompt>"
```

and the answer streams in as `orangu` writes it. The line under the buttons
shows what `orangu` is doing (the server it talks to, the tools it calls)
and, at the end, how the run went. **Stop** ends the run and everything it
started, and **Clear** empties the window.

Anything `orangu -p` takes works here: a question, a request to change
code, a command such as `/status`, `/log` or `/diff`, or a skill. A prompt
stands on its own; nothing is remembered from the one before it. For a
conversation, and for `/review`, which needs the terminal, press
**Open orangu**. `orangu` is the terminal tab's program, so the tab closes
when you quit it. Pressing **Open orangu** again while it runs shows the
same tab rather than starting a second `orangu`.

### Asking about code

**Ask orangu About This Selection** attaches the selection (or, with
nothing selected, the line the caret is on) to the next prompt. The tool
window shows it as `src/parser.rs:12-14` with a **Remove** button, and
fills the prompt box with *Explain what this code does.* when it is empty.
Change the question and press **Enter**: the model gets the question, the
file and the lines, and the code itself, and can open the rest of the file
with its own tools.

### Reviews

**Review This File with orangu** runs `orangu -p "/auto_review <file>"`,
the same review as in the terminal, with its report in the tool window.
On `main` or `master` it reviews the whole file; on any other branch, the
file's changes against the default branch, so a file the branch does not
change is refused with `orangu`'s own message. `orangu` reads the file
from disk, so a file with unsaved changes is reviewed as it was last
saved, which the tool window points out.

The file is passed relative to the workspace, escaped the same way as in
VS Code: a path such as `app/[id]/page.tsx` is sent as a pattern that
matches only it, and a space as `?`.

**Review Changes with orangu** runs `orangu -p "/auto_review"` on
everything the branch changes. Both need a Git repository and a branch
that is up to date with the default branch, as in the terminal. A review
that rejects the changes ends as failed, with the verdict on the status
line.

### The workspace

`orangu` runs in the project's folder, also for a file in one of its
modules. A file in a content root outside the project folder runs in that
root, and a file outside all of them runs in the project's folder. That
folder is also where `orangu` looks for `orangu.conf` first.

### Settings

**Settings > Tools > orangu** has one setting, **orangu binary**: the path
of the `orangu` binary, or the folder it is in. Left empty, the plugin uses
the first `orangu` on the `PATH`, then the one in `~/.cargo/bin`. A path
that is set but does not exist is reported rather than replaced by another
binary. The setting belongs to the machine, so it is not synced with the
IDE's other settings.

### Building it

`contrib/intellij` builds with Gradle (the wrapper is in the folder) and a
JDK 21:

```sh
cd contrib/intellij
./gradlew test
./gradlew buildPlugin
```

The first build downloads the IDE it builds against. `./gradlew test` runs
the plugin's tests inside it, and `./gradlew buildPlugin` writes the `.zip`
to `build/distributions`. `./gradlew runIde` starts an IDE with the plugin
in it, to try a change, and `./gradlew verifyPlugin` checks it against the
current IDE releases.

## Emacs

The Emacs package works in Emacs 28.1 and later, in a terminal or a window,
and needs nothing that is not part of Emacs; `markdown-mode`, when it is
installed, formats the answers in `orangu-chat`. Its source is in
`contrib/emacs`.

### Installing

From a copy of the repository:

```sh
cd contrib/emacs
make install
```

This installs the package into `~/.emacs.d/elpa/orangu`, in place of
any version already there, and `make uninstall` removes it. Restart Emacs, and the commands are there: start with
`M-x orangu`. Every release also has the package as
`orangu-<version>-emacs.tar` beside the binary archives, for
`M-x package-install-file`.

To have every command under a key, and **orangu** in the **Tools** menu and
the context menu, put this in the init file:

```elisp
(global-set-key (kbd "C-c o") 'orangu-command-map)
(orangu-mode 1)
```

The package needs `orangu` itself:

- `orangu` from the archive of the same release, on the `PATH`, in
  `~/.cargo/bin` (where `cargo install --path .` puts it), or wherever
  `orangu-path` says.
- An `orangu.conf` that `orangu` can find: in the project, or in
  `~/.orangu/`. Run `orangu --init` once to write one.

Git is needed to show and to undo what `orangu` changed.

### What it adds

| Command | Key | What it does |
|:---|:---|:---|
| `orangu` | `C-c o o` | The orangu panel: a place for the prompt, and the answers |
| `orangu-implement` | `C-c o i` | Has `orangu` write or change code, and shows the diff |
| `orangu-show-changes` | `C-c o v` | What the last run changed, as a diff |
| `orangu-undo-changes` | `C-c o u` | Undoes what the last run changed |
| `orangu-ask-about-selection` | `C-c o a` | Attaches the region to the next prompt in the panel |
| `orangu-review-file` | `C-c o f` | Runs `/auto_review` on the file |
| `orangu-review-changes` | `C-c o r` | Runs `/auto_review` on the changes on the branch |
| `orangu-chat` | `C-c o c` | A conversation that is kept |
| `orangu-open` | `C-c o t` | Starts the full interface in a terminal inside Emacs |
| `orangu-stop` | `C-c o k` | Stops the run, and everything it started |

### Implementing code

`orangu-implement` asks what to implement and sends it to `orangu`, which
writes the files with its own tools. With a region, the model gets the
file, the lines and the code; without one, it gets the file and the line
point is on; and with a prefix argument (`C-u`), only the instruction.

Before it runs, the unsaved buffers of the workspace are saved, as
`orangu-save-buffers` says. When it is done, buffers whose files changed
are read again. A buffer with unsaved changes of its own is left as it is,
and the panel names it.

In a Git repository the panel lists the files that changed, and the diff
is shown below it. `C-c C-c` in the diff goes to the code, and `C-c C-k`
undoes all of it, after asking. To undo one hunk, use `C-u C-c C-a` on it.
Only what changed during the run is in the diff: changes you had not
committed, and files Git does not track, are part of the state it is
compared with and are put back as they were. The repository itself is only
read. When a file has changed since, undoing it all is refused and nothing
is undone.

The diff is of every prompt that changes files, from the panel and the
conversation as well as from `orangu-implement`, and it is kept until
another one changes something. These are not part of it: files that Git
ignores, untracked files over 1 MiB, every untracked file when there are
more than 5000, and what is in a submodule. What you save yourself while
`orangu` runs is in it.

### The panel

Type a prompt at the end and press **RET**. `C-j` (or **Shift+RET**) starts
a new line, and `M-p` and `M-n` go through the prompts typed before. It
runs as

```
orangu -w <workspace> -p "<prompt>"
```

and the answer streams in as `orangu` writes it. The header line shows what
`orangu` is doing (the server it talks to, the tools it calls) and, at the
end, how the run went. `C-c C-k` ends the run and everything it started,
and `C-c C-l` empties the panel. There is one panel for each workspace.

Anything `orangu -p` takes works here: a question, a request to change
code, a command such as `/status`, `/log` or `/diff`, or a skill. A prompt
stands on its own; nothing is remembered from the one before it.

### Conversations

`orangu-chat` runs `orangu --plain` in a buffer, where what was said is
remembered and an MCP tool that needs approval asks for it. `C-j` starts a
new line. What an answer changed is read again and can be shown and undone
as above.

With `markdown-mode` installed, answers are shown as Markdown: headings,
emphasis, lists and code blocks, with the code highlighted in its own mode
and the markup hidden. The text in the buffer stays as the model wrote it,
so what is copied is Markdown. Without `markdown-mode`, or with
`orangu-chat-markdown` turned off, answers are plain text.

`orangu-open` starts the full interface, for `/review` and everything else
that needs a terminal. It uses `eat` or `vterm` when one of them is
installed, and else `term`, which is part of Emacs; `orangu-terminal`
chooses one. Both commands show what is open already for the workspace
rather than starting a second `orangu`. Emacs has no terminal on Windows,
so `orangu-open` refuses there.

### Asking about code

`orangu-ask-about-selection` attaches the region (or, with none, the line
point is on) to the next prompt. The panel shows it as `src/parser.rs:12-14`
with a **Remove** button, and fills the prompt with *Explain what this code
does.* when it is empty. Change the question and press **RET**: the model
gets the question, the file and the lines, and the code itself, and can
open the rest of the file with its own tools.

### Reviews

`orangu-review-file` runs `orangu -p "/auto_review <file>"` and
`orangu-review-changes` runs `orangu -p "/auto_review"`, with the report in
the panel. They behave as they do in VS Code, and the file is passed the
same way. `orangu` reads the file from disk, so one with unsaved changes is
reviewed as it was last saved, which the panel points out.

### The workspace

`orangu` runs in the project (as `project.el` has it) of the current
buffer, or else in the Git repository it is in, or else in its folder. That
folder is also where `orangu` looks for `orangu.conf` first. A file on
another machine, over TRAMP, has no workspace `orangu` can run in.

### Settings

`M-x customize-group RET orangu` has these:

| Setting | What it is |
|:---|:---|
| `orangu-path` | The path of the `orangu` binary, or the folder it is in. `nil` means the first `orangu` on the `PATH`, then the one in `~/.cargo/bin`. A path that is set but does not exist is reported rather than replaced by another binary. |
| `orangu-save-buffers` | What to do with unsaved buffers before `orangu` runs: ask about each, save them, or leave them |
| `orangu-show-changes` | Whether the diff is shown after `orangu-implement` |
| `orangu-terminal` | The terminal `orangu-open` uses |

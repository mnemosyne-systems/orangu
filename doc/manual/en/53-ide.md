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

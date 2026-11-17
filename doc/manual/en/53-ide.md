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

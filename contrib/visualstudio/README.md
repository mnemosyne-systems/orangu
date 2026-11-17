# orangu for Visual Studio

An extension that brings orangu into Visual Studio 2022 (17.14 and later)
and Visual Studio 2026. It does no inference and has no agent of its own:
it runs the `orangu` binary, which talks to `orangu-server` or
`orangu-coordinator` the way `orangu.conf` says. Whatever orangu can do
in the terminal, it can do here.

The *IDE* chapter of the manual ([`doc/manual/en/53-ide.md`](../../doc/manual/en/53-ide.md))
is the user guide; this file is about the source.

It is built on the VisualStudio.Extensibility SDK, so it runs out of
process and cannot slow down or crash Visual Studio.

## What it adds

| Where | What it does |
|:---|:---|
| **Tools > Open orangu** | Starts orangu in its own window, in the solution folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **View > Other Windows > orangu** (also under **Tools**) | The orangu tool window: a prompt box and the answers |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection (or the caret's line) to the next prompt in the tool window, with its file and line numbers |
| **Right click in the editor > Review This File with orangu** | Asks orangu to review the file, with line numbers and fixes, without changing it |
| **Tools > Options > orangu** | The path of `orangu.exe`, when it is not on the `PATH` |

Each prompt in the tool window runs `orangu -w <workspace> -p <prompt>`,
and the answer streams in as orangu writes it. Commands work as in
`orangu -p`: `/status`, `/log`, `/diff`, "show git status", skills and so on.
A prompt stands on its own: nothing is remembered from the one before it.
For a conversation, or for `/review` and `/auto_review`, which need the
terminal, use **Open orangu**.

The workspace is the solution's folder. With only a file open, it is the
Git repository the file is in, or else the file's folder.

## Before you start

- orangu installed (`cargo install --path .` from this repository puts it
  in `.cargo\bin`, which the extension also looks in), and
- an `orangu.conf` it can find, in the solution folder or in
  `%USERPROFILE%\.orangu\`. Run `orangu --init` once to write one.

## Building

Everything here builds with the dotnet SDK, version 8 or later, on
Windows, Linux or macOS:

```sh
cd contrib/visualstudio
dotnet build
dotnet test
```

That compiles the extension and generates its manifest, and runs the
tests. The `.vsix` itself is packed by the Visual Studio SDK's tools,
which are Windows programs: on Windows with Visual Studio installed,
`dotnet build` packs it as well, and elsewhere the build stops before
packing. You can also open `Orangu.VisualStudio.sln` in Visual Studio
and build it there (F5 starts it in an experimental instance), or use a
Developer PowerShell:

```powershell
msbuild contrib\visualstudio\Orangu.VisualStudio\Orangu.VisualStudio.csproj /restore /p:Configuration=Release
```

The package is
`Orangu.VisualStudio\bin\Release\net8.0-windows8.0\Orangu.VisualStudio.vsix`;
open it to install the extension.

## What is here

| Path | What it is |
|:---|:---|
| `Orangu.Core` | Everything that does not need Visual Studio: finding orangu, its command lines, running a one-shot and streaming its output, and the prompts |
| `Orangu.Core.Tests` | Tests for it, including runs of a stand-in `orangu` script: the streamed answer, failures, Stop, and a missing binary |
| `Orangu.VisualStudio` | The extension: its commands, the tool window and its XAML, and the setting |
| `Orangu.VisualStudio.Tests` | Tests for the tool window's state, and for what Visual Studio only reads at run time: the XAML's resource name and bindings, and the strings the manifest names |

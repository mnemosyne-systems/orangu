# orangu for IntelliJ

A plugin that brings [orangu](https://github.com/mnemosyne-systems/orangu)
into the JetBrains IDEs, 2024.3 and later: IntelliJ IDEA, RustRover, CLion,
PyCharm, GoLand, WebStorm and the others. It does no inference and has no
agent of its own: it runs the `orangu` binary, which talks to
`orangu-server` or `orangu-coordinator` the way `orangu.conf` says.
Whatever orangu can do in the terminal, it can do here.

The *IDE* chapter of the manual ([`doc/manual/en/53-ide.md`](../../doc/manual/en/53-ide.md))
is the user guide; this file is about the source.

It only depends on the base platform and the bundled terminal plugin, so
the same `.zip` installs in every one of those IDEs.

## What it adds

| Where | What it does |
|:---|:---|
| **orangu** tool window, on the right | A prompt box and the answers, with **Stop**, **Clear**, **Review Changes** and **Open orangu** |
| **Tools > Open orangu** | Starts orangu in the IDE's terminal, in the project folder: the full interface, with `/review`, `/auto_review` and a conversation that is kept |
| **Right click in the editor > Ask orangu About This Selection** | Attaches the selection (or the caret's line) to the next prompt in the tool window, with its file and line numbers |
| **Right click in the editor > Review This File with orangu** | Runs orangu's `/auto_review` on the file |
| **Tools > Review Changes with orangu** | Runs `/auto_review` on the changes on the branch |
| **Settings > Tools > orangu** | The path of the `orangu` binary, when it is not on the `PATH` |

Each prompt in the tool window runs `orangu -w <workspace> -p <prompt>`,
and the answer streams in as orangu writes it. Commands work as in
`orangu -p`: `/status`, `/log`, `/diff`, "show git status", skills and so
on. A prompt stands on its own: nothing is remembered from the one before
it. For a conversation, or for `/review`, which needs the terminal, use
**Open orangu**.

The workspace is the project's folder, also for a file in one of its
modules; a file in a content root outside it runs in that root.

## Before you start

- orangu installed (`cargo install --path .` from the orangu repository
  puts it in `~/.cargo/bin`, which the plugin also looks in), and
- an `orangu.conf` it can find, in the project folder or in `~/.orangu/`.
  Run `orangu --init` once to write one.

## Building

With a JDK 21; the Gradle wrapper is here:

```sh
cd contrib/intellij
./gradlew test
./gradlew buildPlugin
```

The first build downloads the IDE the plugin is built against (IntelliJ
IDEA Community 2024.3, the oldest it supports). `./gradlew buildPlugin`
writes `build/distributions/orangu-<version>.zip`; install it with
**Settings > Plugins**, the gear menu, **Install Plugin from Disk...**.
`./gradlew runIde` starts a sandbox IDE with the plugin in it, and
`./gradlew verifyPlugin` checks it against the current IDE releases.
`-PplatformVersion=2026.2.5` builds and tests against another version.

## What is here

| Path | What it is |
|:---|:---|
| `src/main/kotlin/.../OranguService.kt` | The tool window's state and what it does: running a prompt, the reviews, the attached selection |
| `src/main/kotlin/.../OranguPanel.kt`, `OranguToolWindowFactory.kt` | The tool window |
| `src/main/kotlin/.../Actions.kt` | The menu and editor actions |
| `src/main/kotlin/.../OranguTerminal.kt` | Open orangu, in the IDE's terminal |
| `src/main/kotlin/.../OranguSettings.kt`, `OranguConfigurable.kt` | The setting and its page |
| `src/main/kotlin/.../core/` | Everything that does not need the IDE: finding orangu and the workspace, the two command lines, running a one-shot with its output streamed and its escapes removed, the prompts, and the tool window's state |
| `src/main/resources/` | `plugin.xml` and the icons |
| `src/test/` | Tests of the core, runs of a stand-in orangu (streaming, failures, Stop and a missing binary), and the actions and the tool window in a running IDE against the stand-in |

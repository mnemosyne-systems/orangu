\newpage

## Shell completions

Every orangu binary — `orangu`, `orangu-server`, `orangu-coordinator`,
`orangu-bench` and `orangu-gguf` — has shell completion built in, behind the
same switch: `-s`/`--shell-completions` prints a bash, zsh, fish or
PowerShell completion script for the shell it detects, and exits. No
separate package or download is involved; the script comes from the binary
it completes, so it is always the one matching the installed version. Each
binary's script is installed the same way, shown for `orangu` below and
summarised for the others at the end of this chapter.

The shell is read from `$SHELL` — its last path component, so `/bin/bash`,
`bash`, `C:\Program Files\PowerShell\7\pwsh.exe` and `powershell` all name
what they look like. Windows sets no `$SHELL`, so when it names nothing the
binary looks for `$PSModulePath`, which PowerShell (Windows PowerShell and
pwsh alike, on every platform) sets for itself, and prints the PowerShell
script. Git Bash on Windows sets `$SHELL` and gets the bash script. To
override the detection, set `$SHELL` for the one command:
`SHELL=zsh orangu -s`.

Run `orangu -s` to print the completion script for the shell detected from
`$SHELL`, then source it:

| Short | Long                  | Completion              |
| ----- | --------------------- | ----------------------- |
| `-c`  | `--config`            | files                   |
| `-t`  | `--theme`             | theme names and `.theme` files |
| `-w`  | `--workspace`         | directories             |
| `-r`  | `--resume`            | session UUIDs           |
| `-a`  | `--all`               | —                       |
|       | `--developer`         | —                       |
|       | `--committer`         | —                       |
| `-p`  | `--prompt`            | —                       |
| `-q`  | `--quiet`             | —                       |
| `-l`  | `--list`              | —                       |
| `-i`  | `--init`              | —                       |
| `-s`  | `--shell-completions` | —                       |
| `-h`  | `--help`              | —                       |
| `-V`  | `--version`           | —                       |

Completion for `--resume` scans `~/.orangu/sessions/` for the available session
UUIDs and offers them newest first. The in-app `/session` Tab completion offers
the same UUIDs, then the distinct workspace paths recorded across sessions, and
finally — when the typed text matches neither — falls back to filesystem
directory completion (expanding `~`) so a brand-new workspace can be navigated
to.

### bash

Add to `~/.bashrc`:

```sh
eval "$(orangu -s)"
```

Or write once to the `bash-completion` drop-in directory:

```sh
orangu -s > ~/.local/share/bash-completion/completions/orangu
```

### zsh

Add to `~/.zshrc`:

```sh
eval "$(orangu -s)"
```

Or write once to a directory on your `$fpath`:

```sh
mkdir -p ~/.zsh/completions
orangu -s > ~/.zsh/completions/_orangu
```

```sh
# ~/.zshrc
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

### fish

Add to `~/.config/fish/config.fish`:

```sh
orangu -s | source
```

Or write once to the fish completions directory:

```sh
orangu -s > ~/.config/fish/completions/orangu.fish
```

### PowerShell

Add to your profile (`notepad $PROFILE` opens it, creating it if needed):

```powershell
orangu -s | Out-String | Invoke-Expression
```

`Out-String` matters: `Invoke-Expression (orangu -s)` would join the
script's lines with spaces, and its first `#` comment would swallow the rest.

Or write once to a file and dot-source it from `$PROFILE`:

```powershell
orangu -s > "$HOME\orangu.ps1"
# $PROFILE:
. "$HOME\orangu.ps1"
```

The script registers a native argument completer
(`Register-ArgumentCompleter -Native`), so it works in Windows PowerShell
5.1 and in pwsh 7 on every platform. Flags complete with their help text as
the tooltip; where a flag takes a path the script offers nothing of its
own, and PowerShell's usual file completion takes over.

### The other binaries

The same four setups work for every other binary, with its own name in
place of `orangu` — `orangu-server -s > ~/.zsh/completions/_orangu-server`,
`orangu-bench -s | source`, `orangu-gguf -s | Out-String | Invoke-Expression`,
and so on. What each script completes:

| Binary | Completes |
| --- | --- |
| `orangu-server` | every flag, the subcommand names, and the positional `model` argument plus `show`'s, `plan`'s, `delete`'s, `refresh`'s and `bundle`'s own arguments (by shelling back out to `orangu-server list`); `prune`'s session UUIDs; `-w` directories, `-c` files — see the Server chapter |
| `orangu-coordinator` | `-c`/`--config` (files), `-i`/`--init`, `-q`/`--quiet`, `-d`/`--daemon`, `-s`, `-h`, `-V` — see the Coordinator chapter |
| `orangu-bench` | every flag; the path-taking ones (`--history`, `--chart`, `--storage-file`, `--flamegraph`, `--compare-profiles`, `--bundle`, `--read-bundle`, `--render-profile`, `--report`) complete files, `--flamegraph-call-graph` its three modes and `--host` the usual bind addresses — see the Benchmarking chapter |
| `orangu-gguf` | the positional manifest (`.json` files), `-m`/`--model` (`.gguf` files), `-q`/`--quantization` (the weight formats, by shelling back out to `orangu-gguf --list-quantizations`), `-o`/`--output` and `--flamegraph` (files), `--flamegraph-call-graph` (`fp`/`dwarf`), and the two-letter `-ts`/`-cs` alongside their long forms — see the Building a model chapter |

With an unsupported `$SHELL` every binary fails the same way, naming the
shells it does support and the three install lines for its own script.

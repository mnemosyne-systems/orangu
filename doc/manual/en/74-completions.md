\newpage

## Shell completions

**orangu** ships command-line completion scripts under `contrib/shell/`. They
complete the startup flags and their arguments:

| Short | Long          | Completion   |
| ----- | ------------- | ------------ |
| `-c`  | `--config`    | files        |
| `-w`  | `--workspace` | directories  |
| `-r`  | `--resume`    | session UUIDs |
| `-i`  | `--init`      | —            |
| `-h`  | `--help`      | —            |

Completion for `--resume` scans `~/.orangu/sessions/` for the available session
UUIDs and offers them newest first. The in-app `/session` Tab completion offers
the same UUIDs, then the distinct workspace paths recorded across sessions, and
finally — when the typed text matches neither — falls back to filesystem
directory completion (expanding `~`) so a brand-new workspace can be navigated
to.

| Shell | File          |
| ----- | ------------- |
| bash  | `orangu.bash` |
| zsh   | `_orangu`     |
| fish  | `orangu.fish` |

### bash

Source the script from `~/.bashrc`:

```sh
source /path/to/orangu/contrib/shell/orangu.bash
```

Or install it where `bash-completion` looks for per-command scripts:

```sh
install -Dm644 contrib/shell/orangu.bash \
    ~/.local/share/bash-completion/completions/orangu
```

### zsh

Copy the file (it must be named `_orangu`) into a directory on your `$fpath`
and make sure `compinit` runs:

```sh
mkdir -p ~/.zsh/completions
cp contrib/shell/_orangu ~/.zsh/completions/_orangu
```

```sh
# ~/.zshrc
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

### fish

fish loads completions from `~/.config/fish/completions/` automatically:

```sh
cp contrib/shell/orangu.fish ~/.config/fish/completions/orangu.fish
```

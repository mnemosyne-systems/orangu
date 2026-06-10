\newpage

## Shell completions

**orangu** has shell completion built in. Run `orangu -s` to print the
completion script for the shell detected from `$SHELL`, then source it:

| Short | Long                  | Completion    |
| ----- | --------------------- | ------------- |
| `-c`  | `--config`            | files         |
| `-w`  | `--workspace`         | directories   |
| `-r`  | `--resume`            | session UUIDs |
| `-i`  | `--init`              | —             |
| `-s`  | `--shell-completions` | —             |
| `-h`  | `--help`              | —             |

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

# orangu for Emacs

A package that brings [orangu](https://github.com/mnemosyne-systems/orangu)
into Emacs (28.1 and later). It does no inference and has no agent of its
own: it runs the `orangu` binary, which talks to `orangu-server` or
`orangu-coordinator` the way `orangu.conf` says. Whatever orangu can do in
the terminal, it can do here.

The *IDE* chapter of the manual
([`doc/manual/en/53-ide.md`](https://github.com/mnemosyne-systems/orangu/blob/main/doc/manual/en/53-ide.md))
is the user guide; this file is also about the source.

## What it adds

| Command | Key | What it does |
|:---|:---|:---|
| `orangu` | `o` | The orangu panel: the prompts and the answers, and a place to type the next prompt |
| `orangu-implement` | `i` | Asks what to implement and has orangu write the code, with the region or the line point is on; shows what it changed as a diff |
| `orangu-show-changes` | `v` | What the last run changed, as a diff |
| `orangu-undo-changes` | `u` | Undoes what the last run changed |
| `orangu-ask-about-selection` | `a` | Attaches the region (or the line point is on) to the next prompt in the panel |
| `orangu-review-file` | `f` | Runs orangu's `/auto_review` on the file |
| `orangu-review-changes` | `r` | Runs `/auto_review` on the changes on the branch |
| `orangu-chat` | `c` | A conversation that is kept, in a buffer (`orangu --plain`); answers are formatted as Markdown when `markdown-mode` is installed |
| `orangu-open` | `t` | Starts orangu's full interface in a terminal inside Emacs |
| `orangu-stop` | `k` | Stops the run, and everything it started |

After installing, `M-x orangu` and the other commands are there. To have
them under a key and in the menus too, put this in the init file:

```elisp
(global-set-key (kbd "C-c o") 'orangu-command-map)
(orangu-mode 1) ; Tools > orangu, and the context menu
```

`M-x customize-group RET orangu` has the settings.

## Before you start

- orangu installed (`cargo install --path .` from the orangu repository
  puts it in `~/.cargo/bin`, which the package also looks in), and
- an `orangu.conf` it can find, in the project or in `~/.orangu/`. Run
  `orangu --init` once to write one.

## Installing

```sh
make install
```

This installs the package into `~/.emacs.d/elpa/orangu`. Restart Emacs,
and it is there. `make uninstall` removes it, and `make clean` removes what
a build leaves in this directory.

## What is here

| Path | What it is |
|:---|:---|
| `orangu.el` | The commands, the keys and the menus |
| `orangu-panel.el` | The panel, and running a prompt in it |
| `orangu-changes.el` | What a run changed: snapshots, the diff, undoing it, and keeping buffers in step with the files |
| `orangu-terminal.el` | The full interface in a terminal, and the conversation |
| `orangu-run.el` | One run of orangu, with its output streamed and its escapes removed, and stopping it |
| `orangu-core.el` | Finding orangu and its workspace, the command lines, and the prompts |

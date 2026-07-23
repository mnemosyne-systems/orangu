# Developers

This project is a local coding-environment client built around a direct OpenAI-compatible chat loop.

## Main components

- `src/bin/orangu.rs` - terminal loop, commands, history, prompt rendering, and waiting state
- `src/config.rs` - INI parsing and normalization
- `src/llm/openai.rs` - OpenAI-compatible client for `orangu-server`
- `src/session.rs` - tool-calling conversation flow
- `src/tools.rs` - local workspace tools for reading, editing, listing, fetching, and shell commands
- `src/tui.rs` - banner and prompt frame rendering

## Development workflow

```sh
cargo fmt
cargo test
```

## Documentation workflow

The manual sources live under `doc/manual/en`.

```sh
./doc/build_manual.sh
```

## Notes

- The client is workspace-scoped by default and uses the current directory unless `--workspace` is supplied. `orangu-server` takes the same `-w`/`--workspace` for the root it operates in, with the same default.
- Command history is stored in `~/.orangu/orangu.history`.
- Local `orangu-server` deployments may take significant time to answer tool-calling prompts, so the default timeout is 30 minutes.

## Basic git guide

Here are some links that will help you

* [How to Squash Commits in Git](https://www.git-tower.com/learn/git/faq/git-squash)
* [ProGit book](https://github.com/progit/progit2/releases)

### Start by forking the repository

This is done by the "Fork" button on GitHub.

### Clone your repository locally

This is done by

```sh
git clone git@github.com:<username>/orangu.git
```

### Add upstream

Do

```sh
cd orangu
git remote add upstream https://github.com/mnemosyne-systems/orangu.git
```

### Do a work branch

```sh
git checkout -b mywork main
```

### Make the changes

Remember to verify the compile and execution of the code

### AUTHORS

Remember to add your name to the following files,

```
AUTHORS
doc/manual/en/97-acknowledgement.md
```

in your first pull request

### Multiple commits

If you have multiple commits on your branch then squash them

``` sh
git rebase -i HEAD~2
```

for example. It is `p` for the first one, then `s` for the rest

### Rebase

Always rebase

``` sh
git fetch upstream
git rebase -i upstream/main
```

### Force push

When you are done with your changes force push your branch

``` sh
git push -f origin mywork
```

and then create a pull requests for it

### Repeat

Based on feedback keep making changes, squashing, rebasing and force pushing

### PTAL

When you are working on a change put it into Draft mode, so we know that you are not
happy with it yet.

Please, send a PTAL to the Committer that were assigned to you once you think that
your change is complete. And, of course, take it out of Draft mode.

### Undo

Normally you can reset to an earlier commit using `git reset <commit hash> --hard`.
But if you accidentally squashed two or more commits, and you want to undo that,
you need to know where to reset to, and the commit seems to have lost after you rebased.

But they are not actually lost - using `git reflog`, you can find every commit the HEAD pointer
has ever pointed to. Find the commit you want to reset to, and do `git reset --hard`.

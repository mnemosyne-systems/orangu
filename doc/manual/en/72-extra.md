\newpage

## Optional external tools

**orangu** integrates with several optional command-line tools. None of them
are required, but when present they improve the output of the corresponding
commands. The sections below describe each tool and, importantly, **how it is
configured to be used** — installing a tool is not always enough; some must
also be wired into your Git configuration.

The interactive setup wizard (`orangu --init`, short form `-i`) reports the
state of each tool just before writing the configuration:

- `No` — the tool is not installed.
- `Yes (Used)` — the tool is installed **and** configured so orangu will use
  it.
- `Yes (Not used)` — the tool is installed but not configured to be used; for
  example `delta` is on your `PATH` but is not set as your Git diff pager.

What "configured to be used" means for each tool:

| Tool | Used when |
| :-- | :-- |
| `git lg` | the `lg` alias is set in `~/.gitconfig` |
| `delta` | it is installed and resolves as the Git diff pager (`pager.diff`, then `core.pager`) |
| `bat` | it is installed as `bat` or `batcat` (no further configuration needed) |
| `gh` | it is installed and `[orangu].platform` is `github` (the default) |
| `glab` | it is installed and `[orangu].platform` is `gitlab` |

### git lg

`git lg` is a compact, graph-formatted commit log alias for Git. When it is configured in `~/.gitconfig`, **orangu** will use it automatically for `/log` output instead of the plain `git log` fallback.

**Setup**

Add the alias to your global Git configuration:

```sh
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

This adds the following entry to `~/.gitconfig`:

```ini
[alias]
    lg = log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
```

The alias produces a compact, colored, graph-annotated log that shows abbreviated commit hashes in red, branch and tag decorations in yellow, commit subjects, relative timestamps in green, and author names in bold blue.

Once the alias is present, `/log` picks it up automatically — no further configuration is needed.

### delta

[**delta**](https://github.com/dandavison/delta) is an optional pager and syntax-highlighted diff viewer for Git.

If it is installed and configured in your Git setup, **orangu** will use it for `/diff` output inside Git repositories.

**Installation**

Install `delta` using your platform package manager or one of the installation methods described in the upstream project.

On Fedora, for example:

```sh
sudo dnf install git-delta
```

Then configure Git in `~/.gitconfig` to use it. A minimal setup is:

```ini
[core]
  pager = delta

[interactive]
  diffFilter = delta --color-only

[delta]
  navigate = true     # use n and N to move between diff sections
  dark = true         # or light = true, or omit for auto-detection
  side-by-side = true
  line-numbers = true
```

Please refer to the upstream documentation for full installation and configuration details:

<https://github.com/dandavison/delta>

### bat

[**bat**](https://github.com/sharkdp/bat/) is an optional `cat` clone with syntax highlighting and Git integration.

If it is installed, **orangu** will use it for plain `/show_file` output. No
further configuration is required — installing `bat` is enough for it to be
used. On systems where the binary is called `batcat` instead (Debian and
Ubuntu, where the name `bat` was already taken by another package), **orangu**
finds it under that name as well: `bat` is tried first, then `batcat`.

**Installation**

Install `bat` using your platform package manager or one of the installation methods described in the upstream project.

On Fedora, for example:

```sh
sudo dnf install bat
```

On Debian or Ubuntu, where the binary is installed as `batcat`:

```sh
sudo apt install bat
```

Please refer to the upstream documentation for full installation and configuration details:

<https://github.com/sharkdp/bat/>

### gh

[**gh**](https://cli.github.com/) is the official GitHub CLI. It provides commands such as `gh repo clone`, `gh pr create`, and `gh issue list` for interacting with GitHub repositories directly from the terminal.

**orangu** selects the CLI based on the `[orangu].platform` setting: `github` (the default) uses `gh`, and `gitlab` uses `glab` (see [GitLab CLI](#glab) below). The descriptions here apply to `gh`; the GitLab equivalents are described in the next section.

If it is installed, **orangu** will use it for `/pull` to check out pull requests, for `/rebase` to determine the default branch, for `/merge` to merge pull requests, and to detect the default branch for the startup sync (see below). Without it, **orangu** falls back to plain Git for all of these. The `/comment` command requires `gh` and runs `gh issue comment` to add a comment to a GitHub issue, or `gh pr list` and `gh pr comment` for `/comment all` to comment on every open pull request; there is no plain Git fallback for it. The `/pull_request` command also requires `gh` and runs `gh pr create` to open a pull request from the current branch; there is no plain Git fallback for it. The `/issue` command likewise requires `gh` and runs `gh pr edit` / `gh issue edit` to add a reviewer, assignee, or label, or `gh issue create` to open a new issue; there is no plain Git fallback for it.

**Installation**

`gh` is not available in the default Fedora repositories. Add the official GitHub CLI repository first, then install the package:

```sh
curl -fsSL https://cli.github.com/packages/rpm/gh-cli.repo | sudo tee /etc/yum.repos.d/github-cli.repo
sudo dnf install gh
```

After installation, authenticate with your GitHub account:

```sh
gh auth login
```

Please refer to the upstream documentation for full installation and configuration details:

<https://cli.github.com/manual/>

### glab

[**glab**](https://gitlab.com/gitlab-org/cli) is the official GitLab CLI. It provides commands such as `glab mr create`, `glab mr merge`, and `glab issue note` for interacting with GitLab projects directly from the terminal.

Set `platform = gitlab` in the `[orangu]` section to drive `glab` instead of `gh`. The behaviour mirrors the GitHub integration, mapping each command to its GitLab equivalent:

| orangu command | GitLab command |
| :-- | :-- |
| `/pull <number>` | `glab mr checkout <number>` |
| `/pull_request` | `glab mr create --title … --description … --source-branch … --target-branch … --yes` |
| `/merge <branch>` | `glab mr merge <branch> --yes` |
| `/comment <number> "…"` | `glab issue note <number> --message "…"` |
| `/comment all <file>` | `glab mr list`, then `glab mr note <number> --message "…"` for each open merge request |
| `/issue reviewer <n> <user>` | `glab mr update <n> --reviewer <user>` |
| `/issue assignee <n> <user>` | `glab mr update <n> --assignee <user>` / `glab issue update <n> --assignee <user>` |
| `/issue label <n> <label>` | `glab mr update <n> --label <label>` / `glab issue update <n> --label <label>` |
| `/issue create <title> ...` | `glab issue create --title <title> --description <body> [--label ...] [--assignee ...] --yes` |

As with `gh`, `/pull` and `/merge` fall back to plain Git when `glab` is not installed, while `/comment`, `/issue`, and `/pull_request` require it. The default branch used by `/rebase` and the startup sync is detected through Git (`origin/HEAD`, then `main`/`master`) when running against GitLab.

**Credential prompts at startup**

The startup sync and the startup `gh`/`glab` fetches run in the background. If one of them needs a password, a username, or an SSH key passphrase, the prompt is shown in the input area, prefixed with `GitHub` or `GitLab` according to `[orangu].platform`:

```
> GitHub: Enter passphrase for key '/home/me/.ssh/id_ed25519': ******
```

Passwords and passphrases are masked. Press `Enter` to answer, or `Esc` (or `Ctrl+C`) to cancel the prompt, in which case the sync reports that it failed. **orangu** does this by acting as `GIT_ASKPASS` and `SSH_ASKPASS` for those commands. The SSH prompt needs OpenSSH 8.4 or later.

**Installation**

```sh
sudo dnf install glab
```

After installation, authenticate with your GitLab account:

```sh
glab auth login
```

Please refer to the upstream documentation for full installation and configuration details:

<https://gitlab.com/gitlab-org/cli>

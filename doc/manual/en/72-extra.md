\newpage

## Optional external tools

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

Then configure Git to use it. A minimal setup is:

```ini
[core]
    pager = delta
```

Please refer to the upstream documentation for full installation and configuration details:

<https://github.com/dandavison/delta>

### bat

[**bat**](https://github.com/sharkdp/bat/) is an optional `cat` clone with syntax highlighting and Git integration.

If it is installed, **orangu** will use it for plain `/show_file` output.

**Installation**

Install `bat` using your platform package manager or one of the installation methods described in the upstream project.

On Fedora, for example:

```sh
sudo dnf install bat
```

Please refer to the upstream documentation for full installation and configuration details:

<https://github.com/sharkdp/bat/>

### gh

[**gh**](https://cli.github.com/) is the official GitHub CLI. It provides commands such as `gh repo clone`, `gh pr create`, and `gh issue list` for interacting with GitHub repositories directly from the terminal.

If it is installed, **orangu** will use it for `/pull` to check out pull requests, for `/rebase` to determine the default branch, and for `/merge` to merge pull requests. Without it, **orangu** falls back to plain Git for all three operations. The `/comment` command requires `gh` and runs `gh issue comment` to add a comment to a GitHub issue; there is no plain Git fallback for it. The `/pull_request` command also requires `gh` and runs `gh pr create` to open a pull request from the current branch; there is no plain Git fallback for it.

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

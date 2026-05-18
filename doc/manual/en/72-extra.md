\newpage

## Optional external tools

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

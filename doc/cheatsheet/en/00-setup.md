> orangu is a complete, self-contained AI coding stack

# Setup

Install the stack, serve a model, point **orangu** at it. In this order — the
download and the server read the models directory the wizard writes, and
`orangu -i` asks the running server which model it serves.

```
curl -fsSL https://mnemosyne-systems.github.io/orangu/install.sh | sh
```

Installs `orangu`, `orangu-coordinator`, `orangu-server` and `orangu-bench`
into `~/.local/bin`. On Windows: `install.cmd`.

| Command | What it does |
| --- | --- |
| `orangu-server -i` | Configure the server: models directory, role, host, port 8100, web console. |
| `orangu-server download unsloth/gemma-4-E2B-it-GGUF` | Fetch a model. Append `:Q8_0` to pin a quant; `HF_TOKEN` for gated repos. |
| `orangu-server --all unsloth/gemma-4-E2B-it-GGUF` | Serve it on `http://localhost:8100/v1`. Roles: `--all`, `--code`, `--review`, `--explorer`, `--embedding`. |
| `orangu -i` | Configure orangu. Enter that URL; the wizard reads the model off the server. |
| `orangu` | Start, in your project directory. |

## Good to know

| Command | What it does |
| --- | --- |
| `~/.orangu/` | Holds `orangu.conf`, `orangu-server.conf` and `skills/`. A local `./orangu.conf` wins over it. |
| `orangu -w /path/to/project` | Work on another tree. `-r <uuid>` resumes a session, `-l` lists them, `-a` reopens the previous run's tabs, `-p "<prompt>"` runs one prompt and exits. |
| `orangu -s` | Print the shell completions. |
| `/help` `/manual` `/model` | Every command, the full manual offline, and model switching — from the prompt. |
| `/server` `/theme` `/tools` | The server target and its capabilities, the colours, and the tools the model has. |
| `/information` | What the connected server can do — the first thing to check when something looks off. |

## What the wizards write

```
[orangu]
server = main-server
model = unsloth/gemma-4-E2B-it-GGUF

[main-server]
endpoint = http://localhost:8100/v1
```

Only non-defaults are written, so a run of Enters gives a file this short. A few
keys worth knowing in `[orangu]`:

| Command | What it does |
| --- | --- |
| `platform = github` | Or `gitlab`, which switches the forge commands from `gh` to `glab`. |
| `auto_rebase = on` `auto_squash = on` | Let `/pull_request` fix the branch instead of stopping on it. |
| `review_max_tokens = 2048` | Raise it if a thinking model's review answers come back truncated. |

> Nothing leaves the machine: once the model is downloaded, orangu needs no
> Internet connection. Sessions resume automatically per workspace and branch.

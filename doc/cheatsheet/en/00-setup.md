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
| `orangu-server -i` | Models directory, role, host, port 8100, console; `api_key` unless loopback. |
| `orangu-server download unsloth/gemma-4-E2B-it-GGUF` | Reports what a model needs here, then fetches it. `:Q8_0` pins a quant, `HF_TOKEN` opens gated repos, `-y` skips the prompt. |
| `orangu-server plan 3` | That same report for a model already on disk, by `list` NR. Headers only, so it is quick. |
| `orangu-server --all unsloth/gemma-4-E2B-it-GGUF` | Serve it on `http://localhost:8100/v1`. Roles: `--all`, `--code`, `--review`, `--explorer`, `--embedding`. |
| `orangu -i` | Configure orangu. Enter that URL; the wizard reads the model off the server. |
| `orangu` | Start, in your project directory. |

## Good to know

| Command | What it does |
| --- | --- |
| `~/.orangu/` | `orangu.conf`, `orangu-server.conf`, `skills/`; a local `./orangu.conf` wins. |
| `api_key` / `tls_cert` + `tls_key` | Bearer auth (`401` without it) and HTTPS, before exposing a server. `ORANGU_API_KEY` beats the file. |
| `/metrics` `/ready` | Prometheus histograms and counters, no `api_key`. `/ready`: `503` on a full queue. |
| `draft_model` | Speculative decoding: a small model guesses, the served one verifies. Greedy only, and measure it. |
| `orangu -w /path/to/project` | Another tree. `-r` resumes, `-l` lists, `-a` reopens tabs, `-p` runs one prompt. |
| `orangu -s` | Print the shell completions. |
| `/help` `/manual` `/model` | Every command, the full manual offline, and model switching. |
| `/server` `/theme` `/tools` | The server and its capabilities, the colours, the model's tools. |
| `/information` | What the connected server can do — check this first. |

## What the wizards write

```
[orangu]
server = main-server
model = unsloth/gemma-4-E2B-it-GGUF

[main-server]
endpoint = http://localhost:8100/v1
```

Only non-defaults are written, so a run of Enters gives a file this short. Add
`review_max_tokens = 2048` if a thinking model's reviews come back truncated.
Nothing leaves the machine once the model is downloaded, and sessions resume
per workspace and branch.

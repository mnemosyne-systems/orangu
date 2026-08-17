# Getting started

Five commands, in this order: configure the server, download a model, serve it,
configure the client, run the client.

```sh
orangu-server -i
orangu-server download unsloth/gemma-4-E2B-it-GGUF
orangu-server --all unsloth/gemma-4-E2B-it-GGUF
orangu -i
orangu
```

The rest of this chapter walks through each one.

## 1. Create the server configuration

`orangu-server -i` (long form `--init`) writes `~/.orangu/orangu-server.conf`.
Do this first: both the `download` subcommand and serving read the models
directory from that file, and without it they stop with

```text
Missing config file; pass --config or add ./orangu-server.conf or ~/.orangu/orangu-server.conf (see --init)
```

```sh
orangu-server -i
```

The wizard asks, in order, each prompt showing its default in brackets — press
Enter to keep it:

| Prompt | Default | Notes |
| --- | --- | --- |
| `models` | `~/.cache/huggingface/hub` | Where GGUF files live. Tab-completes paths; created if missing. The default is the Hugging Face hub cache, so anything already fetched by `huggingface-cli` or `llama.cpp` is picked up. |
| `model` | *(none)* | The model to serve when none is named on the command line. On a fresh machine nothing is installed yet — press Enter and leave it empty. Skipped entirely when exactly one model is already installed. |
| `role` | `all` | `all`, `code`, `review`, `explorer`, or `embedding`. |
| `host` | `all` | `all` (every interface) or a literal address such as `127.0.0.1`. Tab-completes this machine's interfaces. |
| `port` | `8100` | The HTTP API port. |
| `Add web console` | `Y` | Answering `n` writes no `[web]` section, and no console is served. |

Accepting the web console asks four more: its `host` (defaulting to the address
the API just took), `port` (`8101`), `reexec` (`Y` — may the console load a
different model), and `delete` (`Y` — may the console delete models).

The wizard then prints the file it is about to write and asks
`Write this configuration? [Y/n]`. Anything but Enter/`y`/`yes` aborts without
touching anything. A minimal answer produces:

```ini
[orangu-server]
models = /home/you/.cache/huggingface/hub
host = all
port = 8100

[web]
host = all
port = 8101
```

Note that `-i` always writes `~/.orangu/orangu-server.conf`; it ignores
`-c`/`--config`.

## 2. Download a model

```sh
orangu-server download unsloth/gemma-4-E2B-it-GGUF
```

The argument is a Hugging Face repo, `<user>/<model>[:quant]`. Without a
`:quant` suffix, `Q4_K_M` is preferred, then `Q8_0`, then the first GGUF file in
the repo — so pin one explicitly when it matters:

```sh
orangu-server download unsloth/gemma-4-E2B-it-GGUF:Q8_0
```

Before anything is fetched, the repo is planned against this machine — each
shard's GGUF header is read over the network, which is a few hundred kilobytes
rather than the model, and the result says how much of the model must stay
resident and how much can stream from disk. A model that cannot run here stops
to confirm; `-y` skips that. `orangu-server plan` gives the same report for a
model already on disk.

Every shard of a multi-part model is fetched, along with the matching
`mmproj-*` sidecar when the repo has one. Files land in the `models` directory
from step 1, in Hugging Face hub-cache layout. Free space is checked up front,
an interrupted download resumes, and progress is redrawn in place, one line per
file, closing with:

```text
Downloaded to /home/you/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/<commit>/gemma-4-E2B-it-Q4_K_M.gguf
```

Set `HF_TOKEN` in the environment for gated or private repos. `orangu-server
list` shows everything installed; `orangu-server delete` and `orangu-server
refresh` remove and re-fetch a model.

## 3. Start the server

```sh
orangu-server --all unsloth/gemma-4-E2B-it-GGUF
```

The model argument is the same spec used to download it — it now resolves to
the local copy, so nothing is fetched again. It can equally be a local `.gguf`
path or an `NR`/`MODEL` label from `orangu-server list`. Omit it entirely to
pick from a table of installed models interactively.

`--all` selects the *role*, the general-purpose one. The alternatives are
`--code`, `--review` (suppresses reasoning), `--explorer` (broader, more varied
output), and `--embedding` (embeddings only — chat and completion endpoints are
disabled). They are mutually exclusive, and `all` is the default, so passing
`--all` is a statement of intent rather than a change of behaviour.

The server comes up on the host and port from step 1, serving an
OpenAI-compatible API — this is the URL step 4 asks for:

```text
http://localhost:8100/v1
```

Add `-d`/`--daemon` to detach it from the terminal; that requires
`[orangu-server].model` to be set in the config, since there is no terminal to
prompt on.

As an alternative to steps 1–3, the server and a model can be built into one
self-contained executable — no models directory and no `orangu-server.conf`,
listening on `127.0.0.1:8100` with the web console on `127.0.0.1:8200`:

```sh
orangu-server bundle unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --all -y
./orangu-server-bundle-x86_64
```

## 4. Create the client configuration

With the server from step 3 still running:

```sh
orangu -i
```

The wizard first asks for `LLM URL` — enter the endpoint from step 3,
`http://localhost:8100/v1` (a bare `http://localhost:8100` works too). It then
queries that server's `/v1/models` and offers the first model it advertises as
the default for the `Model` prompt; if the server is unreachable it says so and
asks you to type the model name yourself.

From there it walks every remaining option showing its default — `timeout`,
`max_tool_rounds`, `review_max_tokens`, `code_max_tokens`, `compile_workers`,
`quotes`, `width`, `banner`, `theme`, `workspaces`, `drop_down`, `word_wrap`,
`feedback`, `auto_rebase`, `auto_squash`, `terminal`, `platform`, and
`api_key` — so a full run is a row of Enters. Options left at their default are
omitted from the file rather than written out.

It then reports which optional tools it found (`git lg`, `delta`, `bat`, `gh`,
`glab`), prints the configuration, and asks
`Write this configuration? (Yes/No) [Yes]`. On confirmation it writes
`~/.orangu/orangu.conf` — again ignoring `-c`/`--config` — and installs any
bundled skills into `~/.orangu/skills/` that are not already there:

```ini
[orangu]
server = main-server
model = unsloth/gemma-4-E2B-it-GGUF

[main-server]
endpoint = http://localhost:8100/v1
```

Instead of the wizard you can start from the sample file, adjusting the model
name and endpoint:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

## 5. Run the client

`~/.orangu/orangu.conf` is a default lookup location, so with step 4 done there
is nothing left to pass:

```sh
orangu
```

The configuration is looked up as `./orangu.conf` first, then
`~/.orangu/orangu.conf`. Point at another file explicitly with:

```sh
orangu --config ./orangu.conf
```

Or, with an uninstalled build:

```sh
cargo run --bin orangu -- --config ./orangu.conf
```

## 6. Try a few commands

- `/help`
- `/skills`
- `/server`
- `/disconnect`
- `/reload`
- `/tools`
- `/model`
- `/session`
- `/list_files`
- `/open_file README.md`
- `/show_file README.md`
- `/debugging reproduce the failing request path`
- `/build`
- `/shell ls -la`
- `/create_file README.md`
- `/auto_review`
- `/amend <message>`
- `/branch main`
- `/branch -b feature/new`
- `/branch -m new-name`
- `/branch -d feature/old`
- `/cherry_pick <commit>`
- `/comment 51 "My comment"`
- `/close -i 51`
- `/get_comments -i 51`
- `/commit <message>`
- `/restore README.md`
- `/diff`
- `/fetch`
- `/fetch upstream`
- `/init_repo`
- `/log`
- `/log 5`
- `/show`
- `/show aafd1cb`
- `/merge feature/foo`
- `/move_file old.rs new.rs`
- `/pull 42`
- `/push`
- `/push --force`
- `/rebase`
- `/rebase develop`
- `/rebase origin/main`
- `/delete_file README.md`
- `/review`
- `/squash`
- `/status`
- `/usage`
- `/clear`
- `/quit`

Then try a natural-language request such as:

```text
list files
```

You can also create or copy Agent Skills. orangu discovers skills from:

- `~/.orangu/skills/`
- `~/.agents/skills/`
- `<workspace>/.orangu/skills/`
- `<workspace>/.agents/skills/`

Each skill lives in its own directory and must contain a `SKILL.md` file. List
the discovered skills with:

```text
/skills
```

Invoke one directly with its directory or frontmatter name:

```text
/debugging reproduce the failing request path and identify the root cause
```

Built-in commands also accept natural-language forms, for example:

```text
open README.md
show README.md
list models
list files
pull 42
log
status
rebase
merge feature/foo
checkout main
add README.md
remove README.md
move old.rs new.rs
cherry pick abc1234
commit "[#42] My feature"
amend "[#42] My feature"
push
force push
init repo
squash
delete feature/foo
show help
```

Lines whose first non-whitespace character is `#` stay local and are not sent to the model. Lines whose first non-whitespace character is `\` are ignored.

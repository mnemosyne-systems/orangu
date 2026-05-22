\newpage

# Quick start

This chapter gets **orangu** running against a local OpenAI-compatible server such as **llama.cpp** with the sample configuration in `doc/etc/orangu.conf`.

## Start llama.cpp

Run `llama-server` with your preferred model, for example:

```sh
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF \
             --port 8100 \
             --ctx-size 65536 \
             -sm layer \
             -t 4 \
             --webui-mcp-proxy \
             --fit on
```

**orangu** expects an OpenAI-compatible endpoint, such as:

```text
http://localhost:8100/v1
```

## Create a configuration

Copy the sample:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Default configuration lookup order is:

1. `./orangu.conf`
2. `~/.orangu/orangu.conf`

## Run the client

```sh
orangu
```

Or:

```sh
orangu --config ./orangu.conf
```

Then start with:

```text
/help
/list_models
/list_files
/show_file README.md
/tools
/diff
/status
/log
/pull 42
/rebase
/merge feature/foo
/checkout main
/add_file README.md
/remove_file README.md
/move_file old.rs new.rs
/cherry_pick abc1234
/commit "[#42] My feature"
/amend "[#42] My feature"
/push
/push --force
/init_repo
/squash
/delete feature/foo
```

By default the tools operate on the current directory. Use `--workspace /path/to/project` to point **orangu** at another tree.

Each run creates a session. When you exit, the resume command is printed:

```text
orangu --resume 550e8400-e29b-41d4-a716-446655440000
```

Use `/sessions` to list previous sessions and find a UUID to resume.

Lines whose first non-whitespace character is `#` stay local and are not sent to the model. Lines whose first non-whitespace character is `\` are ignored.

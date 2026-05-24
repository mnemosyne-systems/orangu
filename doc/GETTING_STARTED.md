# Getting started

## 1. Start llama.cpp

Run a local `llama-server` instance with an OpenAI-compatible endpoint.

```sh
llama-server \
  --model /path/to/model.gguf \
  --port 8100 \
  --ctx-size 8192
```

## 2. Create a client configuration

Start from the sample file:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Adjust the model name and endpoint if needed.

## 3. Run the client

```sh
cargo run --bin orangu -- --config ./orangu.conf
```

Or with an installed binary:

```sh
orangu --config ./orangu.conf
```

## 4. Try a few commands

- `/help`
- `/connect`
- `/disconnect`
- `/models`
- `/list_files`
- `/show_file README.md`
- `/tools`
- `/model`
- `/reload`
- `/diff`
- `/status`
- `/log`
- `/pull 42`
- `/rebase`
- `/merge feature/foo`
- `/checkout main`
- `/add_file README.md`
- `/remove_file README.md`
- `/move_file old.rs new.rs`
- `/cherry_pick <commit>`
- `/commit <message>`
- `/amend <message>`
- `/push`
- `/push --force`
- `/init_repo`
- `/squash`
- `/delete feature/foo`
- `/open_file README.md`

Then try a natural-language request such as:

```text
list files
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

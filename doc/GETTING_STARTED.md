# Getting started

## 1. Start orangu-server

Run a local `orangu-server` instance; it serves an OpenAI-compatible endpoint
(API port `8100` by default).

```sh
orangu-server --all /path/to/model.gguf
```

## 2. Create a client configuration

The quickest way is the interactive wizard, which asks for the LLM URL,
auto-detects a model the server advertises, walks every option showing its
default, writes `~/.orangu/orangu.conf` after a confirmation, and installs any
bundled skills into `~/.orangu/skills/` if they are not already present:

```sh
orangu --init
```

Or, with an uninstalled build:

```sh
cargo run --bin orangu -- --init
```

Alternatively, start from the sample file and adjust the model name and
endpoint if needed:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

## 3. Run the client

If you used `orangu --init`, the configuration already lives at
`~/.orangu/orangu.conf` (a default lookup location), so just run:

```sh
orangu
```

Otherwise point the client at your configuration file:

```sh
cargo run --bin orangu -- --config ./orangu.conf
```

Or with an installed binary:

```sh
orangu --config ./orangu.conf
```

## 4. Try a few commands

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

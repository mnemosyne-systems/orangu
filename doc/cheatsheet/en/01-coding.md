# Coding

Say what you want in plain English. The model works through the workspace
tools — show, create, modify, move, delete files, run shell commands, fetch
URLs — and every change is staged with Git as it is made. **Nothing is ever
committed for you.** Typed commands take a leading slash; the natural-language
form (`commit "..."`, `push`) works just as well.

## Start the change

| Command | What it does |
| --- | --- |
| `/branch -b feature/login` | Create and switch. `/branch` lists, `/branch <name>` switches, `-m` renames, `-d` deletes. |
| `/list_files` `/open_file src/main.rs` | List the workspace; open a file in your editor. |
| `/show_file README.md` | Print a file — syntax-highlighted when `bat` is installed. |
| `/create_file` `/move_file old.rs new.rs` `/delete_file` | File lifecycle, staged with `git add`, `git mv` and `git rm` as it happens. |
| `/grep <pattern>` | Search the workspace, without spending a token on it. |

## Let the model work

| Command | What it does |
| --- | --- |
| `add a retry to the HTTP client` | Just ask. Tab completes paths, and `#` starts a line that stays local. |
| `/debugging <what to reproduce>` | Run a skill from `~/.orangu/skills/`. `/skills` lists them. |
| `/graph` `/search <meaning>` | Knowledge graph of the codebase; semantic search by meaning, not by string. |
| `/duplicates` | Structurally similar functions across 20+ languages, entirely local. |
| `/build` `/shell cargo test` | Build with the detected toolchain; run anything inside the workspace. |
| `AGENTS.md` `SKILL.md` | Workspace memory and reusable skills, merged into the model's context. |

## Check and commit

| Command | What it does |
| --- | --- |
| `/status` `/diff` `/log` | Where you are, what changed, what landed. |
| `/restore README.md` | Throw away the changes to one file. |
| `/commit "[#42] Add the feature"` | Commit every tracked change. Quote the message when it has spaces. |
| `/amend "[#42] Better wording"` | Rewrite the last commit message. |
| `/stash` `/cherry_pick abc1234` | Park the working tree; take one commit from elsewhere. |

## Keep going

| Command | What it does |
| --- | --- |
| `/clear` `/copy` `/usage` | Reset the conversation; copy the latest response as Markdown; see what the context window is being spent on. |
| `/session` `/session <uuid or path>` | List sessions; switch to one, or open another project as a new tab. |
| `/schedule` | Run a command on a cron-style schedule, unattended. |
| `#` `\` | A line starting with `#` stays local and is never sent; one starting with `\` is ignored entirely. |
| `Tab` | Completes commands, paths and branch names, previewed as an inline ghost. |
| `/reload` `/restart` | Undo the session's `/model` and `/server` switches; restart orangu in place, same workspace and session. |

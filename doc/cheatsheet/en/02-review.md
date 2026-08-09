# Review

Review the branch before anyone else sees it — yourself, or with the model.
Rebase first: a branch behind its base is sent back to `/rebase`. Both views
copy the report to the clipboard on exit.

| Command | What it does |
| --- | --- |
| `/review` | Interactive: your changed files on the right, the selected file's diff on the left. |
| `/auto_review` | The model reviews the branch and each file, and marks every file green or red. |
| `/auto_review src/tui.rs` `all` | One file, or every Git-tracked file in the project. |
| `is this thread-safe?` | Type a question in either view and press Enter to ask about the selected file. |

## Keys, in both views

| Command | What it does |
| --- | --- |
| `Alt+j` `Alt+k` | Next / previous file. |
| `Alt+a` `Alt+r` | Approve / reject the file — including overriding what the model concluded. |
| `Alt+c` `-` | Comment on the highlighted line (`/review`); drop a finding (`/auto_review`). |
| `Alt+e` `Enter` | Open the file in your editor; show the diff around a finding. |
| `Alt+x` `Esc Esc` | Exit, report to the clipboard; cancel a running request. |

## Share it

| Command | What it does |
| --- | --- |
| `/export review` | A PDF in the workspace root: summary, contents, a page per category, and a source appendix around every finding. |
| `comment on 42 with auto review` | Post the report on issue or pull request #42. |

Categories: Overall, Code, Security, Memory, Performance, Test Suite,
Documentation.

# Merge and push

One rebased commit, pushed, then a pull request — which is exactly what
`/pull_request` checks for before it opens one.

| Command | What it does |
| --- | --- |
| `/rebase` | Onto the default branch, fetched first. `/rebase origin/main` for a specific one. |
| `/squash` | Every commit on the branch into one, keeping the oldest message. |
| `/push` | `git push origin <branch>`. `--force` works, but never on `main` or `master`. |
| `/pull_request` | Push with upstream and open the pull request from the commit message. Needs `gh`. |
| `/pull 42` `/merge feature/login` | Check out someone's pull request; merge a branch. |
| `/close -i 42` `/issue reviewer 114 <user>` | Close an issue; add a reviewer, assignee or label. |

Set `auto_rebase = on` and `auto_squash = on` in `[orangu]` to have the
pre-flight fix itself instead of stopping.

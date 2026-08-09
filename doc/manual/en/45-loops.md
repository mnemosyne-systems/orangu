\newpage

# Code-and-review loops

`orangu loop` is the non-interactive workflow for a bounded sequence of coding
and review iterations. It is intended for a well-scoped change that benefits
from repeated implementation, validation, and independent review without
requiring someone to start every next prompt manually.

It is a CLI subcommand, not a terminal slash command. It needs the same
configuration and workspace as a normal Orangu request:

```text
orangu loop --turns 3 --check 'cargo test' -- 'Fix the parser and keep the public API stable'
```

Use it only when the objective and the checks are clear enough to run
unattended. A loop can change the workspace, but it never commits, pushes, or
opens a pull request by itself.

## What an iteration does

Every iteration has two separate phases:

1. **Work.** A tool-enabled coding session receives the full objective, the
   previous review feedback when there is any, and the current active-time
   budget. It inspects and changes the workspace and can run its own relevant
   validation.
2. **Review.** Orangu runs every configured `--check` command in the workspace,
   then starts a separate reviewer with no tools. The reviewer receives the
   worker report, check output, and the committed-plus-local branch diff used by
   `/review`. It cannot claim to have changed files.

The review response is saved and included in the next work phase. This means a
finding such as a missing test or a regression is an instruction for the next
iteration rather than only a report printed to the screen.

## Starting a loop

Supply one objective and exactly one stopping policy:

```text
orangu loop --turns 3 --check 'cargo test' -- 'Fix the parser'
orangu loop --time 30m --review security -- 'Harden authentication'
orangu loop --until 'All tests pass and no P1 review finding remains' \
  --check 'cargo test' -- 'Finish the migration'
```

`--turns COUNT` stops after that many complete work-and-review iterations.
`--time DURATION` stops before the next iteration once the active time reaches
the duration; accepted units are `s`, `m`, `h`, and `d`. Orangu never stops a
tool call or review response mid-flight. `--until CONDITION` keeps iterating
until the reviewer can establish the condition from its evidence.

`--check COMMAND` may be given more than once. Commands run through the shell
in the workspace after the work phase; their output and exit status are included
in the review even when they fail, so a later iteration can repair the failure.
`--review CRITERION` may also be repeated to extend the default review rubric
of correctness, regressions, tests, and maintainability.

Goal loops use an intentionally strict completion signal. The reviewer must end
with the exact standalone line:

```text
LOOP_COMPLETE: yes
```

Any other text, including an inline mention of the marker, leaves the loop
running. This keeps a goal from ending merely because the reviewer sounds
optimistic.

## Reusable YAML definitions

For a loop that the project runs repeatedly, keep its definition in a YAML file
and pass it with `--file`:

```yaml
version: 1
objective: Harden authentication without changing public APIs.
stop:
  type: goal
  condition: All authentication tests pass and no P1 review finding remains.
review:
  checks:
    - cargo test -p auth
  rubric:
    - security
    - backward compatibility
```

```text
orangu loop --file .orangu/loops/auth-hardening.yaml
```

The file has `version: 1`, an `objective`, one `stop` mapping, and optional
`review` settings. `stop.type` is one of:

| Type | Required field |
| --- | --- |
| `turns` | `count`, a positive integer |
| `time` | `duration`, such as `30m` or `2h` |
| `goal` | `condition`, the reviewer-verifiable completion condition |

`review.checks` and `review.rubric` are lists of strings. A YAML definition is
complete by itself: do not combine `--file` with an objective, stop flag, check,
or review flag.

## Status, pause, and resume

Each workspace has one saved loop state in Orangu's workspace cache, outside
the repository. It stores the objective, policy, iteration count, active time,
and most recent review. That state avoids adding control files to your diff and
makes an interrupted run recoverable:

```text
orangu loop status
orangu loop pause
orangu loop resume
orangu loop clear
```

`status` prints the saved state and last review. `pause` asks a running loop to
stop at its next safe phase boundary; it does not interrupt an active tool call
or reviewer response. `resume` continues a paused or failed loop from the saved
objective and review feedback. `clear` cancels the saved loop, allowing a new
one to replace it. Orangu uses a per-workspace lock so two loops cannot run
against the same checkout at once.

## Operational notes

- `--quiet` suppresses successful loop output, which makes loops suitable for
  scripts; failures still report on stderr and produce a non-zero exit status.
- Review input is bounded before it is sent to the model. The reviewer sees the
  branch-level diff and validation evidence, but very large inputs are marked as
  truncated rather than silently treated as complete evidence.
- The loop's persisted state is control metadata, not a saved model
  conversation. On resume, the workspace and the last review are the durable
  context for the next work session.

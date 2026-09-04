\newpage

# Workflow files

A workflow file describes a set of jobs and the ordered Orangu operations to
run in each workspace. It is a thin YAML layer over the same slash commands,
natural-language prompts, skills, and tools available through `orangu -p`.

Validate the complete file before starting it:

```text
orangu --workflow export-prs.yaml --dry-run
orangu --workflow export-prs.yaml
```

`--workflow --dry-run` loads no model configuration and executes nothing. It checks
every job, workspace, variable, function call, approval, loop definition, and
explicit slash command. `--workflow` performs the same preflight before it
loads configuration or starts the first job.

## Complete file

The following workflow runs `/export pr` in two projects and collects both
PDFs in one explicitly approved directory:

```yaml
orangu:
  version: 1
  role: all
  variables:
    upstream: /home/me/Upstream
  jobs:
    - job: pgagroal
      workspace: /home/me/PostgreSQL/pgagroal
    - job: orangu
      workspace: /home/me/Company/orangu
  functions:
    export_pr:
      - command: /export pr
    move_pdf:
      - command: /shell mv ${job}-pr.pdf ${upstream}/
  main:
    - approved: ${upstream}
    - call: export_pr
    - call: move_pdf
```

The root key is `orangu`. `version` is required and is currently `1`. `jobs`
and `main` must both be non-empty. Unknown keys are errors rather than ignored
configuration.

## Jobs, roles, and conversations

Each `jobs` entry requires a unique `job` name and an existing `workspace`
directory. Jobs run sequentially in declaration order and stop at the first
failure. The complete `main` sequence runs once for every job, with `${job}`
set to that job's name.

`role` may be declared once under `orangu` or overridden by a job. It defaults
to `all`; the other v1 values are `code`, `review`, `explorer`, and
`embeddings`. When the configured endpoint is an Orangu coordinator, the role
is sent to it for routing. Otherwise Orangu selects a configured server with
that role, falling back to the default server.

Command steps that reach the model share one conversation within their job.
The next job starts a separate conversation in its own workspace.

## Variables

Top-level `variables` apply to every job. A job may add variables or override a
top-level value:

```yaml
  variables:
    profile: release
  jobs:
    - job: server
      workspace: server
      variables:
        profile: debug
```

Values may be strings, integers, floating-point numbers, or booleans and are
inserted with `${name}`. Variables may refer to other variables. `${job}` is
reserved for the current job name and cannot be redefined. Undefined,
recursive, malformed, or unterminated references fail validation.

## Functions and execution order

`functions` contains named, reusable lists of steps. A `call` expands a
function at that exact position, so `main` is the single unambiguous execution
order:

```yaml
  functions:
    build:
      - command: /build release
    explain:
      - command: Explain any build warnings and fix actionable ones.
  main:
    - call: build
    - call: explain
```

Functions may call other functions. Unknown calls and direct or indirect
recursion are validation errors. Function and variable names start with a
letter or underscore and contain only letters, digits, and underscores.

## Step types

Version 1 has four explicit step types. A step contains exactly one of them.

### `command`

The value is offered to Orangu's existing dispatcher:

```yaml
  main:
    - command: /status
    - command: /code-review authentication
    - command: Investigate the failing authentication test and fix it.
```

A built-in slash command runs locally, a workspace skill expands into a model
prompt, and other text is a natural-language model prompt. Preflight rejects
unknown slash commands, missing required arguments, and commands that require
the interactive terminal or persistent terminal session.

### `call`

`call` inserts a named function's steps. Expansion happens separately for each
job, after its variables and `${job}` have been resolved.

### `approved`

An external path supplied through a variable requires an earlier `approved`
step. One path or a list of paths may be approved:

```yaml
  main:
    - approved:
        - ${reports}
        - ${archives}
    - command: /shell mv report.pdf ${reports}/
```

Approved paths must already exist. Approval is ordered: placing it after the
command does not authorize that command. It is an explicit workflow-language
opt-in for passing a variable-derived external path; it does not widen the
roots of Orangu's file tools.

File and directory tools reject absolute, `..`, and symlinked paths that leave
the job workspace, and a tool's `cwd` must remain inside it. `/shell` explicitly
invokes the user's platform shell and is not an operating-system sandbox.
Literal shell syntax therefore retains the existing `/shell` semantics.

### `loop`

A loop repeats a tool-enabled work phase followed by an independent,
tool-free review phase:

```yaml
  functions:
    improve_parser:
      - loop:
          objective: Fix ${job}'s parser without changing its public API.
          stop:
            type: goal
            condition: All parser tests pass and review finds no regression.
          review:
            checks:
              - cargo test
            rubric:
              - correctness
              - backward compatibility
  main:
    - call: improve_parser
```

Every iteration gives the worker the objective and the previous review. After
the work phase, Orangu runs the configured checks in the job workspace. The
reviewer receives the worker report, check results, and branch diff, but no
tools. Its response becomes feedback for the next iteration.

Each loop has exactly one stopping policy:

| `stop.type` | Required field | Meaning |
| --- | --- | --- |
| `turns` | `count` | Stop after a positive number of complete iterations. |
| `time` | `duration` | Stop at a safe boundary after active time such as `30m` or `2h`. |
| `goal` | `condition` | Continue until the reviewer verifies the stated condition. |

Durations use `s`, `m`, `h`, or `d`. Orangu never interrupts a tool call or
review response merely because the duration has elapsed. For a goal loop, the
reviewer must end with the exact standalone line `LOOP_COMPLETE: yes`; an
inline mention or optimistic wording does not complete it.

`review.checks` and `review.rubric` are optional lists. Variables are expanded
in the objective, stop value, checks, and rubric. The default rubric covers
correctness, regressions, tests, and maintainability; listed entries add
workflow-specific criteria.

## Managing a workflow

Loop progress is stored in Orangu's workspace cache rather than in the
repository. The workflow file supplies the jobs and their workspaces; append a
lifecycle action after the file to inspect or control the saved loop for every
job:

```text
orangu --workflow /path/to/workflow.yml status
orangu --workflow /path/to/workflow.yml pause
orangu --workflow /path/to/workflow.yml resume
orangu --workflow /path/to/workflow.yml clear
```

`status` reports the objective, policy, iteration count, active time, and most
recent review for each job. `pause` stops an active loop at its next safe phase
boundary. `resume` continues a paused or failed loop with its saved review
feedback. `clear` cancels saved state so another loop may start. A
per-workspace lock prevents two loops from running against the same checkout
simultaneously.

## Operational behavior

- `-q` suppresses successful workflow output. Failures still write to stderr
  and return a non-zero exit status.
- A loop can modify its workspace but does not commit, push, or open a pull
  request unless an explicit workflow command does so.
- Review evidence is bounded before it is sent to the model. Truncation is
  marked rather than silently treated as complete evidence.
- Saved loop state is control metadata, not a persisted model conversation.

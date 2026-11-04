\newpage

# Skills

Agent Skills are reusable instructions stored as `SKILL.md` files. Use them for
workflows that are more specific than the built-in slash commands, such as a
project release checklist, a debugging routine, or house style for migrations.

## Where skills live

orangu discovers skills from:

1. `~/.orangu/skills/`
2. `~/.agents/skills/`
3. `<workspace>/.orangu/skills/`
4. `<workspace>/.agents/skills/`

Each skill lives in its own directory:

```text
.agents/skills/debugging/SKILL.md
```

Project skills override user skills with the same name. Run `/skills` to see
what orangu found.

The model sees every skill's name and description, and loads one whose
description matches the task with the `use_skill` tool, which reads user
skills too, outside the workspace. Subagents can use skills the same way (see
the Tools chapter).

## Minimal skill

A skill needs YAML frontmatter followed by Markdown instructions:

```markdown
---
name: debugging
description: Investigate a bug methodically before proposing a fix
---

# Debugging

First reproduce the issue from the available context. Then identify the most
likely root cause, propose the smallest fix, and suggest validation steps.

Use this extra context when present:

$ARGUMENTS
```

`description` is required. `name` is optional; when it is omitted, the directory
name is used. `$ARGUMENTS` is replaced with the text after the slash command:

```text
/debugging reproduce the failing request path
```

## Skill without code

Most skills should be instruction-only. They work well for review checklists,
release steps, incident triage, documentation style, or project conventions.

Example:

```markdown
---
name: release-check
description: Prepare a release branch for publishing
---

# Release Check

Check the changelog, version number, release notes, tests, and uncommitted
changes. If anything is missing, report it before suggesting a release command.
```

## Skill with helper files

A skill directory may contain extra files such as examples, templates, scripts,
or source code:

```text
.agents/skills/release-check/
  SKILL.md
  templates/release-notes.md
  examples/checklist.md
```

Mention those files in `SKILL.md` and explain when to use them. When a skill is
invoked directly, orangu tells the model which resource files are present and
where the skill directory is.

## Compiled helper code

If a skill needs compiled helper code, keep the source in the skill directory
and write the build command in `SKILL.md`. orangu does not compile skill code
automatically; the model follows the skill instructions and can use
`run_shell_command` when a build is needed.

Rust example:

```text
.agents/skills/log-parser/
  SKILL.md
  helper.rs
  log-parser
```

````markdown
---
name: log-parser
description: Parse service logs with the bundled helper
---

# Log Parser

When log parsing is needed, build the helper first:

```sh
rustc .agents/skills/log-parser/helper.rs -o .agents/skills/log-parser/log-parser
```

Then run it against the user-provided log path:

```sh
.agents/skills/log-parser/log-parser <path>
```
````

C and Java follow the same pattern:

```sh
cc .agents/skills/parser/helper.c -o .agents/skills/parser/parser
javac .agents/skills/parser/Helper.java -d .agents/skills/parser/classes
```

For project skills, prefer storing helper code under `<workspace>/.agents/skills`
so workspace tools can inspect and edit it like normal project files. User
skills under `~/.orangu/skills` or `~/.agents/skills` are still useful for
portable instructions, but project-local helper code is easier for the model to
build, test, and modify.

## Workspace Memory (AGENTS.md)

In addition to explicit `SKILL.md` commands, orangu supports global and workspace-level memory injection using `AGENTS.md` files. 

If you create a file at `~/.orangu/AGENTS.md` or in your project root at `<workspace>/AGENTS.md`, orangu will automatically read the contents and inject them as persistent rules into every chat and code review session. This is the ideal place to document project-wide architectural decisions, language preferences, or rules that should always be active without needing to be explicitly invoked like a skill.

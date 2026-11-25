\newpage

# Knowledge

A knowledge bundle is a directory of Markdown files in the
[Open Knowledge Format](https://github.com/open-knowledge-format/open-knowledge-format)
(OKF, version 0.3). Each file is one concept, such as a table, a metric, an API
or a runbook, with YAML frontmatter that says what it is and how far to trust
it. orangu reads bundles; it does not write them.

Use a bundle for knowledge the model should look things up in rather than carry
in every request: too big for `AGENTS.md`, and with structure and trust
attached.

## Where bundles live

orangu finds bundles in:

1. the paths in `[orangu].knowledge`, a comma-separated list (relative paths are
   from the workspace, `~` is the home directory)
2. `~/.orangu/knowledge/`, where each directory is a bundle
3. `<workspace>/.orangu/knowledge/`, where each directory is a bundle
4. the workspace itself, when its `index.md` declares `okf_version`

```ini
[orangu]
knowledge = docs/knowledge, ~/bundles/acme_retail
```

A bundle is called by its directory's name. Run `/knowledge` to see what orangu
found.

## What the model sees

The model sees each bundle's name, and the `title` and `description` from its
root `index.md`, in the description of the `knowledge` tool. Nothing else is
sent until it asks. The tool has three actions:

- `list` shows a directory's `index.md`, the bundle root when no `path` is
  given. A directory without one gets a list made from its concepts.
- `search` finds concepts whose path, `type`, `title`, `description` or `tags`
  hold every word of `query`, across all bundles unless `bundle` names one.
  Deprecated concepts come last.
- `read` returns the concept at `path` (`metrics/revenue`, with or without
  `.md` or a leading `/`), under a header that says how far to trust it, for
  example:

```text
Concept: [acme_retail] metrics/gross-margin (Metric)
Title: Gross Margin
Trust: human-reviewed, stable, fresh until 2026-12-31
```

A deprecated concept names its `superseded_by` replacement, and an
`Attested Computation` says that orangu has not run or attested it.

## The consumer rules

orangu keeps the spec's consumer rules itself, so they do not depend on the
model:

- The trust tier (`unverified`, `machine-confirmed` or `human-reviewed`) comes
  only from `verified` events at or after `generated.at`. A human review of
  text an agent has since rewritten does not count. Only a `human:` actor gives
  `human-reviewed`.
- A concept is stale when the time is at or past its `stale_after`. A
  `stale_after` that does not parse is reported as such, and never as fresh.
- `status` is `stable` when absent. Any other value is shown as it is.
- `index.md` and `log.md` are never concepts, nor is a `.md` file without a
  `type`, so a `README.md` in a bundle is skipped.
- Unknown keys and types are accepted. A scalar `tags` is one tag.
- Nothing outside the bundle's root is read: not `..`, not a URL, not a hidden
  file, and not a symbolic link that leads out.

## /knowledge

```text
/knowledge
/knowledge stale
/knowledge deprecated
/knowledge unverified
```

With no argument it lists the bundles, where each came from, and how many of
their concepts are stale, deprecated, unverified or drafts. The others list
those concepts. It works with `-p` too, so a CI job can check a bundle:

```sh
orangu -w path/to/repo -p "/knowledge stale"
```

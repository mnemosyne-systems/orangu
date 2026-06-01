\newpage

# Review

`/review` opens a full-screen, two-pane view for reviewing the changes on the current branch before sharing them, with the status bar and input window kept at the bottom so you can ask the model for help. It is available inside a Git repository and has no `gh` dependency.

Enter it with the `/review` command, or the natural-language forms `review`, `review changes`, `code review`, or `review branch`.

## What is reviewed

The review shows everything the current branch adds on top of the default branch:

- Committed changes on the branch, and
- Local uncommitted changes (staged and unstaged) in the working tree.

The comparison is made against the merge base with the default branch. The default branch is detected in the usual order: `origin/main`, `origin/master`, `main`, then `master`. If the working tree has no changes against that base, `/review` reports that there is nothing to review and does not open the view.

## Layout

Above the bottom prompt frame (the status bar and input window, exactly as on the normal screen), the view is split into two panes separated by a single straight vertical line.

- **Left pane** — the diff of the **selected file only**, rendered through the same pipeline as the `/diff` command, including the configured non-interactive git pager (such as `delta`) when one is set. It is the larger pane and scrolls independently.
- **Right pane** — the checklist of changed files, one per row, shown with their full repository-relative paths. The right pane is kept as narrow as possible: just wide enough for the longest path (capped on very narrow terminals so the diff stays usable).

Each file row begins with a review-status box:

- `[ ]` — not yet reviewed
- a green dot in the box — approved
- a red dot in the box — rejected

The currently selected file is highlighted in the right pane. Selecting a different file replaces the left pane with that file's diff, shown from the top.

```
 diff --git a/src/main.rs b/src/main.rs |Files (3)
 @@ -1,4 +1,5 @@                        |[ ] README.md
 +fn new() {}                           |[*] src/main.rs  <- selected
  (only the selected file's diff,       |[*] src/git.rs
   scrollable)                          |
                                        |
```

## Asking the model to review a file

The input window at the bottom takes a request for the model about the selected file — for example `focus on error handling` or `is this thread-safe?`. Press `Alt+o` (or `Enter`) to send that request together with the selected file's diff to the LLM. While the model works, the status bar shows the usual thinking indicator over the panes.

While the model works you are not stuck: press `Esc` twice to cancel the request and return to the diff, or `Alt+x` to leave review mode entirely. A cancelled request is rolled back out of the session.

When the response arrives it opens in a **feedback window** over the panes. If you typed a question, it is echoed at the top of the window — styled like a submitted prompt in the main output — above the model's review. A plain `Alt+o` (empty input) just asks for a plain review of the file, with no question echoed. Press `x` (or `Esc`) to close the window and return to the diff.

The request and the model's reply are added to your chat session, so after leaving review mode you can keep discussing it with full context.

## Commenting on a line

Move the highlighted line to the place you want to comment on and press `Alt+c`. A small comment window opens **inline, just below that line** in the left pane. Type your note (it wraps and the five-line window scrolls if the comment is long), then press `Enter` to save it or `Esc` to discard it.

Each comment is recorded against the file and that diff line; lines with a comment are flagged with an amber dot at the right edge. Pressing `Alt+c` on a line that already has a comment re-opens it for editing, and saving an empty comment removes it.

You can also add a **general note** about the patch: type `# <note>` in the input window and press `Enter` (or `Alt+o`). Instead of being sent to the model, it is recorded as a general note (the `#` is dropped). Anything not starting with `#` is still treated as an LLM request.

When you leave review mode (`Alt+x`), a summary is written to the output window. Each file is listed with its status and a colored dot — `<file>: Approved` with a green dot, `Rejected` with a red dot, or `No review` with a white dot (for unmarked files) — followed by the line comments, one per line, as `<file>:<line>: <comment>` (ordered by file then line, with 1-based line numbers), then the general notes (with the `#` removed). The summary ends with a bold verdict line. If every file is approved and there are no comments or notes the summary is just `Patch approved`; otherwise (any file rejected or unreviewed, or any comment/note) the verdict is `Patch rejected`.

The comments — both the line comments and the general (`#`) notes — are copied to the system clipboard; the per-file statuses and the verdict are not. If the clipboard cannot be reached (for example on a headless machine), a short note is shown instead and the output-window summary is unaffected.

## Key bindings

| Key | Action |
| --- | --- |
| `Alt+j` | Select the next file (shows its diff in the left pane) |
| `Alt+k` | Select the previous file (shows its diff in the left pane) |
| `Alt+a` | Mark the selected file approved (green dot) |
| `Alt+r` | Mark the selected file rejected (red dot) |
| `Alt+c` | Comment on the highlighted line (`Enter` saves, `Esc` discards) |
| `Alt+o` / `Enter` | Ask the model to review the selected file using the typed request |
| `Esc` `Esc` | Cancel an in-progress review request (while the model is thinking) |
| `Alt+x` or `Esc` `Esc` | Exit review mode and return to the prompt |

When the feedback window is open it is modal: `x` or `Esc` closes it, and `Up`/`Down`, `PageUp`/`PageDown`, and `Left`/`Right` scroll and pan it.

Otherwise you can type into the input window normally, and move through the selected file's diff:

- `Up` / `Down` move a highlighted line cursor through the diff; the pane scrolls to keep it in view
- `Alt+Up` / `Alt+Down` scroll the diff one line at a time without moving the cursor
- `PageUp` / `PageDown` scroll the diff by a full page
- `Alt+Left` / `Alt+Right` pan horizontally for long lines

The review status marks and line comments are kept for the duration of the review session and are not persisted after exit.

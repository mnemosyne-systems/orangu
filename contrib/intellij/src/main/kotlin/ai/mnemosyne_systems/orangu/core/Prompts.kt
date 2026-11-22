// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

package ai.mnemosyne_systems.orangu.core

// Paths here are the IDE's: forward slashes on every system, so
// `C:/src/app/main.rs` on Windows.

/** A selection in the editor, attached to the next prompt. Lines count from 1. */
data class EditorSelection(
    val filePath: String,
    val startLine: Int,
    val endLine: Int,
    val text: String,
)

/** What the prompt box is filled with after a selection is attached. */
const val DEFAULT_SELECTION_QUESTION = "Explain what this code does."

/** orangu's review of everything the branch changes. */
const val REVIEW_CHANGES = "/auto_review"

/** A review prompt, or why there is none. */
sealed interface Review {
    data class Prompt(val prompt: String) : Review

    data class Refused(val error: String) : Review
}

private val LANGUAGES = mapOf(
    "c" to "c", "h" to "c",
    "cc" to "cpp", "cpp" to "cpp", "cxx" to "cpp", "hpp" to "cpp",
    "cs" to "csharp",
    "rs" to "rust",
    "py" to "python",
    "js" to "javascript", "mjs" to "javascript", "jsx" to "jsx",
    "ts" to "typescript", "tsx" to "tsx",
    "java" to "java",
    "kt" to "kotlin", "kts" to "kotlin",
    "scala" to "scala",
    "groovy" to "groovy", "gradle" to "groovy",
    "go" to "go",
    "rb" to "ruby",
    "php" to "php",
    "swift" to "swift",
    "dart" to "dart",
    "sql" to "sql",
    "json" to "json",
    "xml" to "xml",
    "html" to "html",
    "css" to "css",
    "yml" to "yaml", "yaml" to "yaml",
    "md" to "markdown",
    "sh" to "bash",
    "ps1" to "powershell",
    "toml" to "toml",
)

/**
 * A question about a selection: the question, then where the code is and
 * the code itself, so the model can also open the file around it.
 */
fun aboutSelection(workspace: String, selection: EditorSelection, question: String): String {
    val asked = question.trim().ifEmpty { DEFAULT_SELECTION_QUESTION }
    val lines = if (selection.startLine == selection.endLine) {
        "line ${selection.startLine}"
    } else {
        "lines ${selection.startLine}-${selection.endLine}"
    }
    val code = selection.text.replace("\r\n", "\n").trimEnd('\n')
    val fence = fenceFor(code)
    return "$asked\n\nThis is $lines of ${displayPath(workspace, selection.filePath)}:\n\n" +
        "$fence${language(selection.filePath)}\n$code\n$fence"
}

/** The short form shown for an attached selection: `src/a.kt:3-9`. */
fun selectionLabel(workspace: String, selection: EditorSelection): String {
    val file = displayPath(workspace, selection.filePath)
    return if (selection.startLine == selection.endLine) {
        "$file:${selection.startLine}"
    } else {
        "$file:${selection.startLine}-${selection.endLine}"
    }
}

/**
 * orangu's own review of one file, `/auto_review <file>`: on main or master
 * the whole file, on a branch the file's changes. orangu takes the file
 * relative to the workspace, so a file outside it has no review.
 */
fun reviewFile(workspace: String, filePath: String): Review {
    val relative = relativeInside(workspace, filePath)
        ?: return Review.Refused("$filePath is outside the workspace $workspace, so orangu cannot review it.")
    return Review.Prompt("$REVIEW_CHANGES ${autoReviewArgument(relative)}")
}

/**
 * A workspace path as one `/auto_review` argument. orangu splits the
 * arguments on whitespace and reads one with `*`, `?`, `[` or `{` in it as
 * a pattern, so a path such as `app/[id]/page.tsx` is given as a pattern
 * that matches only it: each special character in a class of its own. A
 * space cannot be kept, so it becomes `?`, which matches it (and, rarely,
 * another file differing only there). As with a bare name typed in orangu,
 * a file at the top of the workspace is matched by its name, so a file of
 * the same name deeper down is reviewed with it.
 */
fun autoReviewArgument(relative: String): String {
    if (relative.none { it in "*?[{" || it.isWhitespace() }) {
        return relative
    }
    val pattern = StringBuilder()
    for (c in relative) {
        when {
            c in "*?[]{}" -> pattern.append('[').append(c).append(']')
            c.isWhitespace() -> pattern.append('?')
            else -> pattern.append(c)
        }
    }
    return pattern.toString()
}

/** A file as orangu's tools take it: relative to the workspace, or the full path when it is outside it. */
fun displayPath(workspace: String, filePath: String): String = relativeInside(workspace, filePath) ?: filePath

/** The path of `filePath` under `folder`, or null when it is not under it. */
internal fun relativeInside(folder: String, filePath: String): String? {
    val prefix = folder.trimEnd('/') + "/"
    if (!filePath.startsWith(prefix, ignoreCase = isWindowsPath(folder)) || filePath.length == prefix.length) {
        return null
    }
    return filePath.substring(prefix.length)
}

internal fun isWindowsPath(value: String): Boolean =
    (value.length >= 2 && value[0].isLetter() && value[1] == ':') || value.startsWith("//")

private fun language(filePath: String): String =
    LANGUAGES[filePath.substringAfterLast('/').substringAfterLast('.', "").lowercase()].orEmpty()

/** A fence longer than any run of backticks in the code. */
private fun fenceFor(code: String): String {
    var longest = 0
    var run = 0
    for (c in code) {
        run = if (c == '`') run + 1 else 0
        longest = maxOf(longest, run)
    }
    return "`".repeat(maxOf(3, longest + 1))
}

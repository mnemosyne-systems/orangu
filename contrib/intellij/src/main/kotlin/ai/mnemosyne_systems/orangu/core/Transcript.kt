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

import java.util.Locale

/** A prompt as it was asked, or the answer to it. */
data class Entry(val kind: Kind, val text: String) {
    enum class Kind { PROMPT, ANSWER }
}

/** Everything the tool window shows, so a window that is opened again starts from it. */
data class Snapshot(
    val entries: List<Entry>,
    /** What orangu is doing now, or how the last run went. */
    val status: String,
    val busy: Boolean,
    /** The attached selection, as `src/a.kt:3-9`. */
    val selection: String?,
)

/** What changed: the whole state, or just more of the answer. */
sealed interface Change {
    data class Reset(val snapshot: Snapshot) : Change

    data class Answer(val text: String) : Change
}

/**
 * The tool window's state, apart from the IDE: the prompts and their
 * answers, the status line, whether a run is going, and the attached
 * selection. Not thread safe; the plugin keeps it on the UI thread.
 */
class Transcript {
    private val entries = mutableListOf<Entry>()
    private val listeners = mutableListOf<(Change) -> Unit>()
    private var status = ""
    private var busy = false
    private var selection: String? = null

    fun snapshot(): Snapshot = Snapshot(entries.toList(), status, busy, selection)

    val running: Boolean get() = busy

    /** Calls `listener` on every change from now on, until the returned function is called. */
    fun listen(listener: (Change) -> Unit): () -> Unit {
        listeners += listener
        return { listeners -= listener }
    }

    /** A run starts: its prompt is shown, and the status says so. */
    fun begin(prompt: String) {
        entries += Entry(Entry.Kind.PROMPT, prompt)
        busy = true
        status = "Starting orangu…"
        reset()
    }

    /** More of the answer. */
    fun answer(text: String) {
        val last = entries.lastOrNull()
        if (last?.kind == Entry.Kind.ANSWER) {
            entries[entries.size - 1] = last.copy(text = last.text + text)
            emit(Change.Answer(text))
        } else {
            entries += Entry(Entry.Kind.ANSWER, text)
            reset()
        }
    }

    /** A line about what orangu is doing. */
    fun note(line: String) {
        status = line
        reset()
    }

    /** The run ended: the status says how. */
    fun end(result: RunResult, elapsedMs: Long) {
        busy = false
        status = endStatus(result, elapsedMs)
        reset()
    }

    /** Something went wrong before orangu could run. */
    fun fail(message: String) {
        status = message
        reset()
    }

    fun attach(label: String?) {
        selection = label
        reset()
    }

    /** Empties the window; a run that is going keeps going into a new answer. */
    fun clear() {
        entries.clear()
        if (!busy) {
            status = ""
        }
        reset()
    }

    private fun reset() = emit(Change.Reset(snapshot()))

    private fun emit(change: Change) = listeners.toList().forEach { it(change) }
}

/** The status line once a run is over. */
fun endStatus(result: RunResult, elapsedMs: Long): String = when (result.outcome) {
    Outcome.SUCCEEDED -> String.format(Locale.ROOT, "Done in %.1fs", elapsedMs / 1000.0)
    Outcome.FAILED -> "orangu failed: ${result.error ?: "exit code ${result.exitCode}"}"
    Outcome.CANCELLED -> "Stopped"
    Outcome.NOT_STARTED -> "Could not start orangu: ${result.error ?: "unknown error"}"
}

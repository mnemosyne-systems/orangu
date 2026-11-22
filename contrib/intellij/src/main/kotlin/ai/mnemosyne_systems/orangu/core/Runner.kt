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

import java.io.IOException
import java.io.InputStreamReader
import java.nio.CharBuffer
import kotlin.concurrent.thread

/** How a one-shot ended. */
enum class Outcome {
    /** orangu exited with 0. */
    SUCCEEDED,

    /** orangu exited with an error; its stderr says why. */
    FAILED,

    /** The user stopped it. */
    CANCELLED,

    /** orangu could not be started at all. */
    NOT_STARTED,
}

data class RunResult(val outcome: Outcome, val exitCode: Int? = null, val error: String? = null)

/**
 * One run of `orangu -p`, from start to end or until [stop]. Its output is
 * handed over as it comes: the answer as it streams on stdout, and the
 * notes on stderr (the server, the tools it calls, what it cost) line by
 * line. Both are called on the threads that read them.
 */
class OneShotRun(
    private val line: CommandLine,
    private val start: (CommandLine) -> Process,
    private val answer: (String) -> Unit,
    private val note: (String) -> Unit,
) {
    private val lock = Any()
    private var process: Process? = null
    private var stopped = false

    @Volatile
    private var lastNote = ""

    /** Runs orangu to its end, on the calling thread. */
    fun run(): RunResult {
        val process = synchronized(lock) {
            if (stopped) {
                return RunResult(Outcome.CANCELLED)
            }
            val started = try {
                start(line)
            } catch (e: Exception) {
                return RunResult(Outcome.NOT_STARTED, error = "${line.command.first()}: ${e.message ?: e}")
            }
            process = started
            started
        }
        // Nothing is ever typed into a one-shot; a closed stdin makes that
        // plain to anything it runs.
        runCatching { process.outputStream.close() }

        val answers = thread(isDaemon = true, name = "orangu answer") { readAnswer(process) }
        val notes = thread(isDaemon = true, name = "orangu notes") { readNotes(process) }
        val code = process.waitFor()
        // What is left in the pipes comes right after; a pipe that something
        // orangu started still holds open is not waited for.
        answers.join(READ_GRACE_MS)
        notes.join(READ_GRACE_MS)

        return when {
            synchronized(lock) { stopped } -> RunResult(Outcome.CANCELLED)
            code == 0 -> RunResult(Outcome.SUCCEEDED, 0)
            else -> RunResult(Outcome.FAILED, code, lastNote.ifEmpty { "orangu exited with $code" })
        }
    }

    /**
     * Ends the run and everything orangu started: a tool may have started a
     * build or a test run of its own, which should not outlive it.
     */
    fun stop() {
        val process = synchronized(lock) {
            stopped = true
            process
        } ?: return
        val tree = process.descendants().toList() + process.toHandle()
        tree.forEach { it.destroy() }
        thread(isDaemon = true, name = "orangu stop") {
            Thread.sleep(STOP_GRACE_MS)
            tree.filter { it.isAlive }.forEach { it.destroyForcibly() }
        }
    }

    // Read what is there, not whole lines: the answer streams a few words at
    // a time and should show up as it does.
    private fun readAnswer(process: Process) {
        val filter = AnsiFilter()
        val buffer = CharArray(4096)
        InputStreamReader(process.inputStream, Charsets.UTF_8).use { reader ->
            while (true) {
                val count = try {
                    reader.read(buffer)
                } catch (e: IOException) {
                    -1
                }
                if (count < 0) {
                    break
                }
                val text = filter.push(CharBuffer.wrap(buffer, 0, count))
                if (text.isNotEmpty()) {
                    answer(text)
                }
            }
        }
    }

    private fun readNotes(process: Process) {
        val filter = AnsiFilter()
        try {
            process.errorStream.bufferedReader(Charsets.UTF_8).use { reader ->
                while (true) {
                    val text = filter.push(reader.readLine() ?: break).trimEnd()
                    if (text.isNotEmpty()) {
                        lastNote = text
                        note(text)
                    }
                }
            }
        } catch (e: IOException) {
            // The process was killed while its notes were read.
        }
    }

    private companion object {
        /** How long a stopped run gets to end before it is killed outright. */
        const val STOP_GRACE_MS = 2000L

        /** How long the output may still come in after orangu has exited. */
        const val READ_GRACE_MS = 2000L
    }
}

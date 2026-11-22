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

package ai.mnemosyne_systems.orangu

import ai.mnemosyne_systems.orangu.core.CommandLine
import ai.mnemosyne_systems.orangu.core.OneShotRun
import ai.mnemosyne_systems.orangu.core.Outcome
import ai.mnemosyne_systems.orangu.core.RunResult
import ai.mnemosyne_systems.orangu.core.oneShot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Collections

/**
 * One-shots of the stand-in, started the way the plugin starts orangu, so
 * the quoting of the arguments (which matters on Windows) is tested too.
 */
class RunnerTest {
    private val workspace = File(System.getProperty("java.io.tmpdir")).canonicalPath

    private class Ran(val result: RunResult, val answer: String, val pieces: Int, val notes: List<String>)

    private fun run(prompt: String): Ran {
        val answer = StringBuffer()
        var pieces = 0
        val notes = Collections.synchronizedList(mutableListOf<String>())
        val result = OneShotRun(
            StandIn.instead(oneShot("orangu", workspace, prompt)),
            ::startWithIde,
            answer = {
                pieces++
                answer.append(it)
            },
            note = { notes += it },
        ).run()
        return Ran(result, answer.toString(), pieces, notes.toList())
    }

    @Test
    fun `orangu gets -w and -p, runs in the workspace without colour, and stdin is closed`() {
        val prompt = "what is \"this\"?\nsecond line, a \\ and a \\\" and C:\\path\\"
        val ran = run(prompt)
        assertEquals(RunResult(Outcome.SUCCEEDED, 0), ran.result)
        val report = StandIn.parse(ran.answer)
        assertEquals(listOf("-w", workspace, "-p", prompt), report.args)
        assertEquals(File(workspace).canonicalPath, File(report.cwd).canonicalPath)
        assertEquals("1", report.noColor)
        assertTrue(report.stdinClosed)
        // Notes come line by line, without their escape sequences.
        assertEquals(listOf("Using the stand-in server", "Done: 1 request"), ran.notes)
    }

    @Test
    fun `the answer streams in pieces, without escape sequences cut between them`() {
        val ran = run("stream")
        assertEquals(Outcome.SUCCEEDED, ran.result.outcome)
        assertEquals("Hello, world!\n", ran.answer)
        assertTrue("the answer came in ${ran.pieces} piece(s)", ran.pieces > 1)
    }

    @Test
    fun `a failure reports orangu's last note and its exit code`() {
        assertEquals(RunResult(Outcome.FAILED, 3, "error: the stand-in server is not running"), run("fail").result)
    }

    @Test
    fun `Stop ends orangu and what it started`() {
        var child: Long? = null
        lateinit var run: OneShotRun
        run = OneShotRun(
            StandIn.instead(oneShot("orangu", workspace, "sleep")),
            ::startWithIde,
            answer = {},
            note = { note ->
                Regex("child (\\d+)").find(note)?.let {
                    child = it.groupValues[1].toLong()
                    run.stop()
                }
            },
        )
        val started = System.currentTimeMillis()
        assertEquals(RunResult(Outcome.CANCELLED), run.run())
        assertTrue("Stop took too long", System.currentTimeMillis() - started < 15000)
        // The child held the output open; it is gone with its parent.
        val pid = child ?: error("the stand-in never said which child it started")
        val deadline = System.currentTimeMillis() + 5000
        while (ProcessHandle.of(pid).map { it.isAlive }.orElse(false) && System.currentTimeMillis() < deadline) {
            Thread.sleep(50)
        }
        assertFalse("the child $pid is still running", ProcessHandle.of(pid).map { it.isAlive }.orElse(false))
    }

    @Test
    fun `a binary that is not there is not started, with the reason`() {
        val missing = File(workspace, "no-such-orangu-here").path
        val result = OneShotRun(oneShot(missing, workspace, "hi"), ::startWithIde, {}, {}).run()
        assertEquals(Outcome.NOT_STARTED, result.outcome)
        assertTrue(result.error, result.error!!.contains("no-such-orangu-here"))
    }

    @Test
    fun `a run stopped before it starts never starts`() {
        var started = false
        val run = OneShotRun(
            CommandLine(listOf("orangu"), workspace),
            { started = true; error("not to be started") },
            {},
            {},
        )
        run.stop()
        assertEquals(RunResult(Outcome.CANCELLED), run.run())
        assertFalse(started)
    }
}

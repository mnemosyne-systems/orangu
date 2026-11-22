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

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AnsiFilterTest {
    @Test
    fun `plain text passes through, without carriage returns`() {
        assertEquals("one\ntwo\n", AnsiFilter().push("one\r\ntwo\n"))
    }

    @Test
    fun `colours and cursor moves are removed`() {
        assertEquals("red and plain", AnsiFilter().push("\u001b[1;31mred\u001b[0m and \u001b[2Kplain"))
    }

    @Test
    fun `a sequence cut between two pieces is held back until it ends`() {
        val filter = AnsiFilter()
        assertEquals("a", filter.push("a\u001b"))
        assertEquals("", filter.push("[3"))
        assertEquals("b", filter.push("2mb"))
    }

    @Test
    fun `titles and links end at a bell or at ESC backslash`() {
        val filter = AnsiFilter()
        assertEquals("x", filter.push("\u001b]0;title\u0007x"))
        assertEquals("link", filter.push("\u001b]8;;https://example.org\u001b\\link\u001b]8;;\u001b\\"))
    }

    @Test
    fun `a two-character sequence is dropped whole`() {
        assertEquals("abc", AnsiFilter().push("a\u001b7b\u001b8c"))
    }
}

class LocatorTest {
    private class FakeMachine(
        private val files: Set<String>,
        override val pathVariable: String?,
        override val home: String?,
        override val windows: Boolean,
    ) : Machine {
        override fun isFile(candidate: String) = candidate in files
    }

    private fun linux(vararg files: String, path: String = "/usr/local/bin:/usr/bin") =
        FakeMachine(files.toSet(), path, "/home/me", false)

    private fun windows(vararg files: String, path: String = "C:\\Windows;\"C:\\Tools\\orangu\"") =
        FakeMachine(files.toSet(), path, "C:\\Users\\me", true)

    @Test
    fun `the first orangu on the PATH is used`() {
        assertEquals(Located.Found("/usr/bin/orangu"), locate("", linux("/usr/bin/orangu", "/home/me/.cargo/bin/orangu")))
    }

    @Test
    fun `cargo install is found when it is not on the PATH`() {
        assertEquals(Located.Found("/home/me/.cargo/bin/orangu"), locate(null, linux("/home/me/.cargo/bin/orangu")))
        assertEquals(
            Located.Found("C:\\Users\\me\\.cargo\\bin\\orangu.exe"),
            locate("", windows("C:\\Users\\me\\.cargo\\bin\\orangu.exe")),
        )
    }

    @Test
    fun `on Windows the binary is orangu exe, and quoted PATH entries count`() {
        assertEquals(Located.Found("C:\\Tools\\orangu\\orangu.exe"), locate("", windows("C:\\Tools\\orangu\\orangu.exe")))
    }

    @Test
    fun `a PATH entry with a trailing separator still joins cleanly`() {
        assertEquals(Located.Found("/opt/bin/orangu"), locate("", linux("/opt/bin/orangu", path = "/opt/bin/")))
    }

    @Test
    fun `the setting wins, as a file or as the folder it is in`() {
        val machine = linux("/opt/orangu/orangu", "/usr/bin/orangu")
        assertEquals(Located.Found("/opt/orangu/orangu"), locate("/opt/orangu/orangu", machine))
        assertEquals(Located.Found("/opt/orangu/orangu"), locate("  \"/opt/orangu\"  ", machine))
    }

    @Test
    fun `a setting that does not exist is reported, not replaced by the PATH`() {
        val result = locate("/nowhere/orangu", linux("/usr/bin/orangu"))
        assertTrue(result is Located.Missing)
        assertEquals(
            "Settings > Tools > orangu names a file that does not exist: /nowhere/orangu",
            (result as Located.Missing).error,
        )
    }

    @Test
    fun `nothing found says how to fix it`() {
        val result = locate("", linux()) as Located.Missing
        assertEquals(
            "Could not find orangu on the PATH. Install orangu, or set its path in Settings > Tools > orangu.",
            result.error,
        )
    }
}

class CommandLinesTest {
    @Test
    fun `a one-shot is -w and -p, in the workspace, without colour`() {
        val line = oneShot("/usr/bin/orangu", "/work/my project", "what does \"x\" do?\nand y")
        // Arguments are passed as a list, so spaces, quotes and new lines stay as they are.
        assertEquals(listOf("/usr/bin/orangu", "-w", "/work/my project", "-p", "what does \"x\" do?\nand y"), line.command)
        assertEquals("/work/my project", line.workingDirectory)
        assertEquals(mapOf("NO_COLOR" to "1"), line.environment)
    }

    @Test
    fun `the terminal runs orangu itself in the workspace`() {
        val line = interactive("C:\\bin\\orangu.exe", "C:\\work\\app")
        assertEquals(listOf("C:\\bin\\orangu.exe", "-w", "C:\\work\\app"), line.command)
        assertEquals("C:\\work\\app", line.workingDirectory)
    }
}

class PromptsTest {
    private val selection = EditorSelection(
        filePath = "/work/app/src/parser.rs",
        startLine = 12,
        endLine = 14,
        text = "fn parse() {\r\n    todo!()\r\n}\r\n",
    )

    @Test
    fun `a question about a selection carries the file, the lines and the code`() {
        assertEquals(
            "Why is this slow?\n\nThis is lines 12-14 of src/parser.rs:\n\n```rust\nfn parse() {\n    todo!()\n}\n```",
            aboutSelection("/work/app", selection, "  Why is this slow?  "),
        )
    }

    @Test
    fun `an empty question asks what the code does, and one line says line`() {
        val one = selection.copy(endLine = 12, text = "let x = 1;")
        assertEquals(
            "Explain what this code does.\n\nThis is line 12 of src/parser.rs:\n\n```rust\nlet x = 1;\n```",
            aboutSelection("/work/app", one, " "),
        )
    }

    @Test
    fun `the fence is longer than any backticks in the code`() {
        val fenced = selection.copy(filePath = "/work/app/README.md", text = "````\ncode\n````")
        assertTrue(aboutSelection("/work/app", fenced, "q").endsWith("\n`````markdown\n````\ncode\n````\n`````"))
    }

    @Test
    fun `an unknown extension gets a fence without a language`() {
        val plain = selection.copy(filePath = "/work/app/notes.txt", text = "x")
        assertTrue(aboutSelection("/work/app", plain, "q").endsWith("\n```\nx\n```"))
    }

    @Test
    fun `the label is the short path and the lines`() {
        assertEquals("src/parser.rs:12-14", selectionLabel("/work/app", selection))
        assertEquals("src/parser.rs:12", selectionLabel("/work/app", selection.copy(endLine = 12)))
    }

    @Test
    fun `paths are relative, or whole when outside`() {
        assertEquals("a/b.kt", displayPath("/work/app", "/work/app/a/b.kt"))
        assertEquals("/work/other/b.kt", displayPath("/work/app", "/work/other/b.kt"))
        assertEquals("/work/application/b.kt", displayPath("/work/app", "/work/application/b.kt"))
        assertEquals("src/Main.kt", displayPath("C:/work/app", "C:/work/app/src/Main.kt"))
        assertEquals("src/Main.kt", displayPath("C:/work/app", "c:/work/app/src/Main.kt"))
        assertEquals("D:/elsewhere/Main.kt", displayPath("C:/work/app", "D:/elsewhere/Main.kt"))
        assertEquals("etc/hosts", displayPath("/", "/etc/hosts"))
    }

    @Test
    fun `a file review is orangu's auto_review of that file`() {
        assertEquals(Review.Prompt("/auto_review src/parser.rs"), reviewFile("/work/app", "/work/app/src/parser.rs"))
        assertEquals(Review.Prompt("/auto_review src/Main.kt"), reviewFile("C:/work/app", "C:/work/app/src/Main.kt"))
    }

    @Test
    fun `a file outside the workspace has no review`() {
        val review = reviewFile("/work/app", "/tmp/scratch.rs")
        assertTrue(review is Review.Refused)
        assertTrue((review as Review.Refused).error.contains("outside the workspace"))
    }

    @Test
    fun `an ordinary path is one argument as it is`() {
        assertEquals("src/main.rs", autoReviewArgument("src/main.rs"))
        assertEquals("Makefile", autoReviewArgument("Makefile"))
        // `]` and `}` alone do not make orangu read a pattern.
        assertEquals("notes/a]b}.md", autoReviewArgument("notes/a]b}.md"))
    }

    @Test
    fun `glob characters are matched literally, and spaces by a question mark`() {
        assertEquals("app/[[]id[]]/page.tsx", autoReviewArgument("app/[id]/page.tsx"))
        assertEquals("src/what[?].rs", autoReviewArgument("src/what?.rs"))
        assertEquals("src/[{]a,b[}].rs", autoReviewArgument("src/{a,b}.rs"))
        assertEquals("src/star[*].rs", autoReviewArgument("src/star*.rs"))
        assertEquals("docs/my?notes.md", autoReviewArgument("docs/my notes.md"))
    }

    @Test
    fun `a top-level name is escaped the same way`() {
        assertEquals("my?notes.md", autoReviewArgument("my notes.md"))
        assertEquals("[[]x[]].kt", autoReviewArgument("[x].kt"))
    }

    @Test
    fun `a file review argument never splits into two`() {
        for (file in listOf("a b/c d.rs", "tab\there.rs", "[id] x/{y}.kt")) {
            assertEquals(file, 1, autoReviewArgument(file).split(Regex("\\s+")).size)
        }
    }
}

class WorkspaceTest {
    private val nothing: (String) -> Boolean = { false }

    @Test
    fun `a file in the project runs in the project folder, even inside a module`() {
        val roots = listOf("/work/app", "/work/app/server")
        assertEquals("/work/app", chooseWorkspace("/work/app", roots, "/work/app/server/src/Main.kt", nothing))
    }

    @Test
    fun `a file in a content root outside the project runs in that root, the outermost one`() {
        val roots = listOf("/work/app", "/work/lib", "/work/lib/core")
        assertEquals("/work/lib", chooseWorkspace("/work/app", roots, "/work/lib/core/src/a.kt", nothing))
    }

    @Test
    fun `a folder whose name only starts the same is not the file's`() {
        assertEquals("/work/app", chooseWorkspace("/work/app", listOf("/work/application"), "/work/app/a.kt", nothing))
        assertEquals(
            "/work/application",
            chooseWorkspace("/work/app", listOf("/work/application"), "/work/application/a.kt", nothing),
        )
    }

    @Test
    fun `no file, or one outside everything, gives the project folder`() {
        assertEquals("/work/app", chooseWorkspace("/work/app", listOf("/work/lib"), null, nothing))
        assertEquals("/work/app", chooseWorkspace("/work/app", emptyList(), "/tmp/scratch.kt", nothing))
    }

    @Test
    fun `without a project folder, the first content root`() {
        assertEquals("/work/lib", chooseWorkspace(null, listOf("/work/lib", "/work/web"), "/tmp/scratch.kt", nothing))
    }

    @Test
    fun `a file on its own runs in its Git repository`() {
        val exists = { p: String -> p == "/home/me/repo/.git" }
        assertEquals("/home/me/repo", chooseWorkspace(null, emptyList(), "/home/me/repo/src/deep/a.rs", exists))
    }

    @Test
    fun `a Windows file on its own finds its repository too`() {
        val exists = { p: String -> p == "C:/code/repo/.git" }
        assertEquals("C:/code/repo", chooseWorkspace(null, emptyList(), "C:/code/repo/src/a.kt", exists))
        val atTop = { p: String -> p == "C:/.git" }
        assertEquals("C:/", chooseWorkspace(null, emptyList(), "C:/code/a.kt", atTop))
    }

    @Test
    fun `outside any repository, the file's folder`() {
        assertEquals("/tmp/notes", chooseWorkspace(null, emptyList(), "/tmp/notes/a.md", nothing))
    }

    @Test
    fun `nothing open gives nothing`() {
        assertEquals(null, chooseWorkspace(null, emptyList(), null, nothing))
    }
}

class TranscriptTest {
    @Test
    fun `a run shows its prompt, streams its answer and ends with how it went`() {
        val changes = mutableListOf<Change>()
        val transcript = Transcript()
        transcript.listen { changes += it }
        transcript.begin("/status")
        assertTrue(transcript.snapshot().busy)
        transcript.answer("Branch: main")
        transcript.note("Using server local")
        transcript.answer("\nClean")
        transcript.end(RunResult(Outcome.SUCCEEDED, 0), 1234)
        assertEquals(
            Snapshot(
                entries = listOf(Entry(Entry.Kind.PROMPT, "/status"), Entry(Entry.Kind.ANSWER, "Branch: main\nClean")),
                status = "Done in 1.2s",
                busy = false,
                selection = null,
            ),
            transcript.snapshot(),
        )
        // More of the same answer is sent as just the new text.
        assertEquals(listOf(Change.Answer("\nClean")), changes.filterIsInstance<Change.Answer>())
    }

    @Test
    fun `clearing while a run goes on keeps it going into a new answer`() {
        val transcript = Transcript()
        transcript.begin("q")
        transcript.answer("one")
        transcript.clear()
        transcript.answer("two")
        assertEquals(listOf(Entry(Entry.Kind.ANSWER, "two")), transcript.snapshot().entries)
        assertTrue(transcript.snapshot().busy)
    }

    @Test
    fun `clearing when nothing runs empties the status too`() {
        val transcript = Transcript()
        transcript.fail("Could not find orangu")
        transcript.clear()
        assertEquals("", transcript.snapshot().status)
    }

    @Test
    fun `the attached selection is part of the state`() {
        val transcript = Transcript()
        transcript.attach("src/a.kt:3-9")
        assertEquals("src/a.kt:3-9", transcript.snapshot().selection)
        transcript.attach(null)
        assertEquals(null, transcript.snapshot().selection)
    }

    @Test
    fun `a listener that stops listening hears nothing more`() {
        val transcript = Transcript()
        var heard = 0
        val stop = transcript.listen { heard++ }
        transcript.note("one")
        stop()
        transcript.note("two")
        assertEquals(1, heard)
    }

    @Test
    fun `every way a run ends has its own status`() {
        assertEquals("Done in 0.1s", endStatus(RunResult(Outcome.SUCCEEDED, 0), 50))
        assertEquals("orangu failed: no server", endStatus(RunResult(Outcome.FAILED, 2, "no server"), 0))
        assertEquals("orangu failed: exit code 2", endStatus(RunResult(Outcome.FAILED, 2), 0))
        assertEquals("Stopped", endStatus(RunResult(Outcome.CANCELLED), 0))
        assertEquals("Could not start orangu: no such file", endStatus(RunResult(Outcome.NOT_STARTED, error = "no such file"), 0))
    }
}

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
import java.io.File
import java.util.Base64

/** Runs the Java stand-in for orangu in place of the real binary. */
object StandIn {
    private val java = File(System.getProperty("java.home"), "bin/java").path
    // The folder FakeOrangu.class is under. The IDE's class loader gives no
    // code source, so it is worked out from where the class file is.
    private val classes = run {
        val name = FakeOrangu::class.java.name.replace('.', '/') + ".class"
        val file = File(FakeOrangu::class.java.classLoader.getResource(name)!!.toURI()).path
        file.substring(0, file.length - name.length - 1)
    }

    /** The stand-in's own command, which takes orangu's arguments after it. */
    val command = listOf(java, "-cp", classes, FakeOrangu::class.java.name)

    /** `line` with the stand-in in place of the binary. */
    fun instead(line: CommandLine): CommandLine = line.copy(command = command + line.command.drop(1))

    /** What the stand-in reported about how it was run. */
    data class Report(val args: List<String>, val cwd: String, val noColor: String, val stdinClosed: Boolean)

    fun parse(answer: String): Report {
        val fields = answer.lines().filter { it.isNotEmpty() }.associate { it.substringBefore(' ') to it.substringAfter(' ', "") }
        return Report(
            args = fields.getValue("args").split(' ').map { String(Base64.getDecoder().decode(it), Charsets.UTF_8) },
            cwd = fields.getValue("cwd"),
            noColor = fields.getValue("noColor"),
            stdinClosed = fields.getValue("stdinClosed").toBoolean(),
        )
    }
}

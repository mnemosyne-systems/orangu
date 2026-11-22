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

/** A program to run: what to run, where, and with what environment. */
data class CommandLine(
    val command: List<String>,
    val workingDirectory: String,
    val environment: Map<String, String> = emptyMap(),
)

/**
 * `orangu -w WORKSPACE -p PROMPT`: one prompt, the answer streamed on stdout
 * and the notes about what it is doing on stderr. Nothing is asked on stdin.
 */
fun oneShot(binary: String, workspace: String, prompt: String): CommandLine =
    CommandLine(
        command = listOf(binary, "-w", workspace, "-p", prompt),
        workingDirectory = workspace,
        // Plain text: the tool window cannot show colours.
        environment = mapOf("NO_COLOR" to "1"),
    )

/**
 * `orangu -w WORKSPACE` in the IDE's terminal: the full interface, with
 * /review, /auto_review and the session kept.
 */
fun interactive(binary: String, workspace: String): CommandLine =
    CommandLine(command = listOf(binary, "-w", workspace), workingDirectory = workspace)

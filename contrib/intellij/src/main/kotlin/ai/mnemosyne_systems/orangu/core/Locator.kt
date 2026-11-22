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

import java.io.File

/** Where orangu is, or why it could not be found. */
sealed interface Located {
    data class Found(val path: String) : Located

    data class Missing(val error: String) : Located
}

/** Everything the search reads from the machine, so tests can pass their own. */
interface Machine {
    val pathVariable: String?
    val home: String?
    val windows: Boolean

    fun isFile(candidate: String): Boolean
}

/** This machine: the IDE's PATH, home and file system. */
object ThisMachine : Machine {
    override val pathVariable: String? get() = System.getenv("PATH") ?: System.getenv("Path")
    override val home: String? get() = System.getProperty("user.home")
    override val windows: Boolean get() = System.getProperty("os.name").startsWith("Windows")

    override fun isFile(candidate: String): Boolean = File(candidate).isFile
}

/** Where the setting is, for the messages that point at it. */
const val SETTING_NAME = "Settings > Tools > orangu"

/**
 * Finds the orangu binary: the path in the setting, or else `orangu` on the
 * PATH, or else where `cargo install` puts it.
 */
fun locate(configured: String?, machine: Machine): Located {
    val name = if (machine.windows) "orangu.exe" else "orangu"
    val setting = unquote(configured.orEmpty())
    if (setting.isNotEmpty()) {
        // A folder is taken to mean the binary inside it.
        for (candidate in listOf(setting, join(machine, setting, name))) {
            if (machine.isFile(candidate)) {
                return Located.Found(candidate)
            }
        }
        return Located.Missing("$SETTING_NAME names a file that does not exist: $setting")
    }
    for (directory in searchDirectories(machine)) {
        val candidate = join(machine, directory, name)
        if (machine.isFile(candidate)) {
            return Located.Found(candidate)
        }
    }
    return Located.Missing("Could not find $name on the PATH. Install orangu, or set its path in $SETTING_NAME.")
}

private fun searchDirectories(machine: Machine): List<String> {
    val separator = if (machine.windows) ';' else ':'
    val directories = machine.pathVariable.orEmpty()
        .split(separator)
        .map(::unquote)
        .filter { it.isNotEmpty() }
        .toMutableList()
    // The IDE may have been started before ~/.cargo/bin was added to the
    // PATH, or from a desktop launcher that never read the shell's profile,
    // so `cargo install --path .` is looked for directly.
    machine.home?.takeIf { it.isNotEmpty() }?.let {
        directories += join(machine, join(machine, it, ".cargo"), "bin")
    }
    return directories
}

private fun join(machine: Machine, directory: String, name: String): String {
    val separator = if (machine.windows) '\\' else '/'
    val trimmed = directory.trimEnd('/', separator)
    return "$trimmed$separator$name"
}

private fun unquote(value: String): String = value.trim().trim('"')

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

/**
 * The folder orangu runs in, given what the IDE has open: the project's
 * folder when the active file is in it; else the content root the file is
 * in (the outermost, since a root inside another is a module of it); else
 * the project's folder, or the first content root; else, for a file that
 * belongs to none of them, the Git repository around it, or its folder.
 * Null with nothing to go on. Paths use forward slashes.
 */
fun chooseWorkspace(
    projectFolder: String?,
    contentRoots: List<String>,
    activeFile: String?,
    exists: (String) -> Boolean,
): String? {
    if (activeFile != null) {
        val containing = listOfNotNull(projectFolder).filter { relativeInside(it, activeFile) != null } +
            contentRoots.filter { relativeInside(it, activeFile) != null }.sortedBy { it.trimEnd('/').length }
        containing.firstOrNull()?.let { return it }
    }
    if (projectFolder != null) {
        return projectFolder
    }
    contentRoots.firstOrNull()?.let { return it }
    if (activeFile == null) {
        return null
    }
    val folder = parent(activeFile) ?: return null
    var directory = folder
    while (true) {
        // A worktree has a .git file rather than a folder; either counts.
        if (exists(directory.trimEnd('/') + "/.git")) {
            return directory
        }
        directory = parent(directory) ?: return folder
    }
}

/** The folder a path is in, or null at the top: `/a/b` is in `/a`, `/a` in `/`, `C:/a` in `C:/`. */
private fun parent(path: String): String? {
    val trimmed = path.trimEnd('/')
    val slash = trimmed.lastIndexOf('/')
    if (slash < 0) {
        return null
    }
    val folder = trimmed.substring(0, slash)
    return when {
        folder.isEmpty() -> "/"
        folder.length == 2 && folder[1] == ':' -> "$folder/"
        else -> folder
    }
}

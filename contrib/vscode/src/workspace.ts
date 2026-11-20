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

import * as path from 'path';

/**
 * The folder orangu runs in, given what VS Code has open: the workspace
 * folder the active file is in (the innermost, when folders nest); else
 * the first workspace folder; else, for a file opened on its own, the Git
 * repository around it, or its folder. Undefined with nothing open.
 */
export function chooseWorkspace(
    folders: readonly string[],
    activeFile: string | undefined,
    exists: (candidate: string) => boolean,
): string | undefined {
    if (activeFile !== undefined) {
        const paths = pathsFor(activeFile);
        const containing = folders
            .filter((folder) => {
                const relative = paths.relative(folder, activeFile);
                return relative.length > 0 && relative !== '..' && !relative.startsWith(`..${paths.sep}`) && !paths.isAbsolute(relative);
            })
            .sort((a, b) => b.length - a.length);
        if (containing.length > 0) {
            return containing[0];
        }
    }
    if (folders.length > 0) {
        return folders[0];
    }
    if (activeFile === undefined) {
        return undefined;
    }
    const paths = pathsFor(activeFile);
    const folder = paths.dirname(activeFile);
    for (let directory = folder; ; directory = paths.dirname(directory)) {
        // A worktree has a .git file rather than a folder; either counts.
        if (exists(paths.join(directory, '.git'))) {
            return directory;
        }
        if (paths.dirname(directory) === directory) {
            return folder;
        }
    }
}

function pathsFor(file: string): path.PlatformPath {
    return /^[A-Za-z]:[\\/]/.test(file) || file.startsWith('\\\\') ? path.win32 : path.posix;
}

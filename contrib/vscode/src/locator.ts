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

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

/** Where orangu is, or why it could not be found. */
export type LocateResult = { path: string } | { error: string };

/** Everything the search reads from the machine, so tests can pass their own. */
export interface Machine {
    pathVariable: string | undefined;
    home: string | undefined;
    windows: boolean;
    isFile(candidate: string): boolean;
}

/** This machine: the extension host's PATH, home and file system. */
export function thisMachine(): Machine {
    return {
        pathVariable: process.env.PATH ?? process.env.Path,
        home: os.homedir(),
        windows: process.platform === 'win32',
        isFile: (candidate) => {
            try {
                return fs.statSync(candidate).isFile();
            } catch {
                return false;
            }
        },
    };
}

/**
 * Finds the orangu binary: the path in the `orangu.path` setting, or else
 * `orangu` on the PATH, or else where `cargo install` puts it.
 */
export function locate(configured: string | undefined, machine: Machine): LocateResult {
    const join = machine.windows ? path.win32.join : path.posix.join;
    const name = machine.windows ? 'orangu.exe' : 'orangu';
    const setting = unquote(configured ?? '');
    if (setting.length > 0) {
        // A folder is taken to mean the binary inside it.
        for (const candidate of [setting, join(setting, name)]) {
            if (machine.isFile(candidate)) {
                return { path: candidate };
            }
        }
        return { error: `The orangu.path setting names a file that does not exist: ${setting}` };
    }
    for (const directory of searchDirectories(machine)) {
        const candidate = join(directory, name);
        if (machine.isFile(candidate)) {
            return { path: candidate };
        }
    }
    return {
        error: `Could not find ${name} on the PATH. Install orangu, or set its path in the orangu.path setting.`,
    };
}

function searchDirectories(machine: Machine): string[] {
    const separator = machine.windows ? ';' : ':';
    const directories = (machine.pathVariable ?? '')
        .split(separator)
        .map(unquote)
        .filter((directory) => directory.length > 0);
    // VS Code may have been started before ~/.cargo/bin was added to the
    // PATH, so `cargo install --path .` is looked for directly.
    if (machine.home) {
        const join = machine.windows ? path.win32.join : path.posix.join;
        directories.push(join(machine.home, '.cargo', 'bin'));
    }
    return directories;
}

function unquote(value: string): string {
    return value.trim().replace(/^"+|"+$/g, '');
}

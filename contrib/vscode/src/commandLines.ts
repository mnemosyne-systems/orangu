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

/** A process to start: the binary, its arguments, where, and with what environment. */
export interface CommandLine {
    command: string;
    args: string[];
    cwd: string;
    env: NodeJS.ProcessEnv;
}

/**
 * `orangu -w WORKSPACE -p PROMPT`: one prompt, the answer streamed on
 * stdout and the notes about what it is doing on stderr. Nothing is asked
 * on stdin, so it runs without a terminal.
 */
export function oneShot(
    binary: string,
    workspace: string,
    prompt: string,
    environment: NodeJS.ProcessEnv = process.env,
): CommandLine {
    return {
        command: binary,
        args: ['-w', workspace, '-p', prompt],
        cwd: workspace,
        // Plain text: the panel shows no colours.
        env: { ...environment, NO_COLOR: '1' },
    };
}

/** What VS Code needs to start orangu's interface in a terminal of its own. */
export interface TerminalLine {
    name: string;
    shellPath: string;
    shellArgs: string[];
    cwd: string;
}

/**
 * `orangu -w WORKSPACE` in VS Code's integrated terminal: the full
 * interface, with /review, /auto_review and the session kept. The terminal
 * runs orangu itself rather than a shell, so there is nothing to quote.
 */
export function interactive(binary: string, workspace: string, name = 'orangu'): TerminalLine {
    return {
        name,
        shellPath: binary,
        shellArgs: ['-w', workspace],
        cwd: workspace,
    };
}

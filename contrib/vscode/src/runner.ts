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

import { ChildProcess, spawn } from 'child_process';
import { AnsiFilter } from './ansi';
import { CommandLine } from './commandLines';

/** How a one-shot ended. */
export type Outcome =
    /** orangu exited with 0. */
    | 'succeeded'
    /** orangu exited with an error; its stderr says why. */
    | 'failed'
    /** The user stopped it. */
    | 'cancelled'
    /** orangu could not be started at all. */
    | 'notStarted';

export interface RunResult {
    outcome: Outcome;
    exitCode?: number;
    error?: string;
}

/** How long a stopped run gets to end before it is killed outright. */
const STOP_GRACE_MS = 2000;

/**
 * Runs a one-shot to its end, or until `signal` stops it, and hands its
 * output over as it comes: the answer as it streams on stdout, and the notes
 * on stderr (the server, the tools it calls, what it cost) line by line.
 */
export function runOneShot(
    line: CommandLine,
    answer: (text: string) => void,
    note: (line: string) => void,
    signal: AbortSignal,
): Promise<RunResult> {
    return new Promise((resolve) => {
        if (signal.aborted) {
            resolve({ outcome: 'cancelled' });
            return;
        }
        let child: ChildProcess;
        try {
            child = spawn(line.command, line.args, {
                cwd: line.cwd,
                env: line.env,
                stdio: ['pipe', 'pipe', 'pipe'],
                windowsHide: true,
                // Its own process group, so Stop reaches whatever it started.
                detached: process.platform !== 'win32',
            });
        } catch (err) {
            resolve({ outcome: 'notStarted', error: `${line.command}: ${message(err)}` });
            return;
        }

        let settled = false;
        const finish = (result: RunResult) => {
            if (!settled) {
                settled = true;
                signal.removeEventListener('abort', stop);
                resolve(result);
            }
        };

        let lastNote = '';
        const answers = new AnsiFilter();
        const notes = new AnsiFilter();
        let pendingNote = '';
        const emitNote = (raw: string) => {
            const text = notes.push(raw).trimEnd();
            if (text.length > 0) {
                lastNote = text;
                note(text);
            }
        };

        // Read what is there, not whole lines: the answer streams a few
        // words at a time and should show up as it does.
        child.stdout?.setEncoding('utf8');
        child.stdout?.on('data', (piece: string) => {
            const text = answers.push(piece);
            if (text.length > 0) {
                answer(text);
            }
        });
        child.stderr?.setEncoding('utf8');
        child.stderr?.on('data', (piece: string) => {
            const lines = (pendingNote + piece).split('\n');
            pendingNote = lines.pop() ?? '';
            lines.forEach(emitNote);
        });
        // Nothing is ever typed into a one-shot; a closed stdin makes that
        // plain to anything it runs.
        child.stdin?.end();
        child.stdin?.on('error', () => undefined);

        let stopped = false;
        let killTimer: NodeJS.Timeout | undefined;
        const stop = () => {
            stopped = true;
            killTree(child, false);
            killTimer = setTimeout(() => killTree(child, true), STOP_GRACE_MS);
        };
        signal.addEventListener('abort', stop);

        child.on('error', (err) => {
            if (killTimer) {
                clearTimeout(killTimer);
            }
            finish({ outcome: 'notStarted', error: `${line.command}: ${message(err)}` });
        });
        child.on('close', (code) => {
            if (killTimer) {
                clearTimeout(killTimer);
            }
            if (pendingNote.length > 0) {
                emitNote(pendingNote);
                pendingNote = '';
            }
            if (stopped) {
                finish({ outcome: 'cancelled' });
            } else if (code === 0) {
                finish({ outcome: 'succeeded', exitCode: 0 });
            } else {
                finish({
                    outcome: 'failed',
                    exitCode: code ?? undefined,
                    error: lastNote.length > 0 ? lastNote : `orangu exited with ${code ?? 'a signal'}`,
                });
            }
        });
    });
}

/**
 * Ends a process and everything it started. orangu may already have exited
 * while something it started still holds the output open, so the group is
 * signalled either way; one that is gone already is no error.
 */
function killTree(child: ChildProcess, force: boolean): void {
    if (child.pid === undefined) {
        return;
    }
    try {
        if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true }).on('error', () => undefined);
        } else {
            process.kill(-child.pid, force ? 'SIGKILL' : 'SIGTERM');
        }
    } catch {
        // It had already exited, or is exiting.
    }
}

function message(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

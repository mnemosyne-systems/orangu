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

// A stand-in for orangu in the tests. With `-p`, what it does depends on
// the prompt:
//
//   fail   a note on stderr and exit code 3
//   stream the answer in pieces, with escape sequences cut between them
//   sleep  starts a child that sleeps too, and never ends by itself
//   other  reports what it was given: the arguments, the folder, NO_COLOR
//          and whether stdin is closed
//
// Without `-p` (the terminal interface), it says so and waits.

import { spawn } from 'child_process';

const args = process.argv.slice(2);
const p = args.indexOf('-p');
const prompt = p >= 0 ? args[p + 1] : undefined;

function later(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
    if (prompt === undefined) {
        process.stdout.write(`orangu stand-in interface in ${process.cwd()}\n`);
        setInterval(() => undefined, 1 << 30);
        return;
    }
    // A note, in colour, the way a tool run might print one.
    process.stderr.write('\u001b[2mUsing the stand-in server\u001b[0m\n');
    switch (prompt) {
        case 'fail':
            process.stderr.write('error: the stand-in server is not running\n');
            process.exitCode = 3;
            return;
        case 'stream':
            for (const piece of ['Hello', ', \u001b[1', 'mworld\u001b[0m', '!\r\n']) {
                process.stdout.write(piece);
                await later(30);
            }
            return;
        case 'sleep': {
            const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1 << 30)'], { stdio: 'inherit' });
            process.stderr.write(`sleeping, child ${child.pid}\n`);
            setInterval(() => undefined, 1 << 30);
            return;
        }
        default: {
            let stdinClosed = false;
            process.stdin.on('end', () => (stdinClosed = true));
            process.stdin.resume();
            await later(100);
            const report = {
                args,
                cwd: process.cwd(),
                noColor: process.env.NO_COLOR ?? null,
                stdinClosed,
            };
            process.stdout.write(`${JSON.stringify(report)}\n`);
            process.stderr.write('Done: 1 request\n');
            process.exit(0);
        }
    }
}

void main();

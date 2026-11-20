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

import * as assert from 'node:assert/strict';
import * as os from 'node:os';
import * as path from 'node:path';
import { test } from 'node:test';
import { CommandLine, oneShot } from '../../src/commandLines';
import { runOneShot } from '../../src/runner';

const FAKE = path.join(__dirname, '..', 'fixtures', 'fake-orangu.js');

/** The stand-in, run by node, the way the extension runs orangu. */
function standIn(prompt: string): CommandLine {
    const line = oneShot(process.execPath, os.tmpdir(), prompt);
    return { ...line, args: [FAKE, ...line.args] };
}

async function run(line: CommandLine, signal = new AbortController().signal) {
    let answer = '';
    const notes: string[] = [];
    const result = await runOneShot(line, (text) => (answer += text), (note) => notes.push(note), signal);
    return { result, answer, notes };
}

test('orangu gets -w and -p, runs in the workspace without colour, and stdin is closed', async () => {
    const { result, answer, notes } = await run(standIn('what is "this"?\nsecond line'));
    assert.deepEqual(result, { outcome: 'succeeded', exitCode: 0 });
    const report = JSON.parse(answer);
    assert.deepEqual(report.args, ['-w', os.tmpdir(), '-p', 'what is "this"?\nsecond line']);
    assert.equal(path.resolve(report.cwd), path.resolve(os.tmpdir()));
    assert.equal(report.noColor, '1');
    assert.equal(report.stdinClosed, true);
    // Notes come line by line, without their escape sequences.
    assert.deepEqual(notes, ['Using the stand-in server', 'Done: 1 request']);
});

test('the answer streams in pieces, without escape sequences cut between them', async () => {
    let pieces = 0;
    let answer = '';
    const result = await runOneShot(
        standIn('stream'),
        (text) => {
            pieces++;
            answer += text;
        },
        () => undefined,
        new AbortController().signal,
    );
    assert.equal(result.outcome, 'succeeded');
    assert.equal(answer, 'Hello, world!\n');
    assert.ok(pieces > 1, `the answer came in ${pieces} piece(s)`);
});

test('a failure reports orangu’s last note and its exit code', async () => {
    const { result } = await run(standIn('fail'));
    assert.deepEqual(result, {
        outcome: 'failed',
        exitCode: 3,
        error: 'error: the stand-in server is not running',
    });
});

test('Stop ends orangu and what it started', async () => {
    const controller = new AbortController();
    const started = Date.now();
    let childPid: number | undefined;
    const running = runOneShot(
        standIn('sleep'),
        () => undefined,
        (note) => {
            const match = /child (\d+)/.exec(note);
            if (match) {
                childPid = Number(match[1]);
                controller.abort();
            }
        },
        controller.signal,
    );
    const result = await running;
    assert.deepEqual(result, { outcome: 'cancelled' });
    assert.ok(Date.now() - started < 10000, 'Stop took too long');
    assert.ok(childPid !== undefined);
    // The child held the output open; it is gone with its parent.
    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.throws(() => process.kill(childPid!, 0), /ESRCH|EPERM/);
});

test('a binary that is not there is not started, with the reason', async () => {
    const missing = oneShot(path.join(os.tmpdir(), 'no-such-orangu-here'), os.tmpdir(), 'hi');
    const { result } = await run(missing);
    assert.equal(result.outcome, 'notStarted');
    assert.match(result.error ?? '', /no-such-orangu-here.*ENOENT/);
});

test('a run stopped before it starts never starts', async () => {
    const controller = new AbortController();
    controller.abort();
    const { result } = await run(standIn('sleep'), controller.signal);
    assert.deepEqual(result, { outcome: 'cancelled' });
});

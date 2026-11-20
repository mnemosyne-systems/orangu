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
import { test } from 'node:test';
import { Change, endStatus, Transcript } from '../../src/transcript';

test('a run shows its prompt, streams its answer and ends with how it went', () => {
    const changes: Change[] = [];
    const transcript = new Transcript((change) => changes.push(change));
    transcript.begin('/status');
    assert.equal(transcript.snapshot().busy, true);
    transcript.answer('Branch: main');
    transcript.note('Using server local');
    transcript.answer('\nClean');
    transcript.end({ outcome: 'succeeded', exitCode: 0 }, 1234);
    assert.deepEqual(transcript.snapshot(), {
        entries: [
            { kind: 'prompt', text: '/status' },
            { kind: 'answer', text: 'Branch: main\nClean' },
        ],
        status: 'Done in 1.2s',
        busy: false,
        selection: undefined,
    });
    // More of the same answer is sent as just the new text.
    assert.deepEqual(
        changes.filter((change) => change.type === 'answer'),
        [{ type: 'answer', text: '\nClean' }],
    );
});

test('clearing while a run goes on keeps it going into a new answer', () => {
    const transcript = new Transcript();
    transcript.begin('q');
    transcript.answer('one');
    transcript.clear();
    transcript.answer('two');
    assert.deepEqual(transcript.snapshot().entries, [{ kind: 'answer', text: 'two' }]);
    assert.equal(transcript.snapshot().busy, true);
});

test('clearing when nothing runs empties the status too', () => {
    const transcript = new Transcript();
    transcript.fail('Could not find orangu');
    transcript.clear();
    assert.equal(transcript.snapshot().status, '');
});

test('the attached selection is part of the state', () => {
    const transcript = new Transcript();
    transcript.attach('src/a.ts:3-9');
    assert.equal(transcript.snapshot().selection, 'src/a.ts:3-9');
    transcript.attach(undefined);
    assert.equal(transcript.snapshot().selection, undefined);
});

test('every way a run ends has its own status', () => {
    assert.equal(endStatus({ outcome: 'succeeded', exitCode: 0 }, 50), 'Done in 0.1s');
    assert.equal(endStatus({ outcome: 'failed', exitCode: 2, error: 'no server' }, 0), 'orangu failed: no server');
    assert.equal(endStatus({ outcome: 'failed', exitCode: 2 }, 0), 'orangu failed: exit code 2');
    assert.equal(endStatus({ outcome: 'cancelled' }, 0), 'Stopped');
    assert.equal(endStatus({ outcome: 'notStarted', error: 'ENOENT' }, 0), 'Could not start orangu: ENOENT');
});

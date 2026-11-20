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
import { interactive, oneShot } from '../../src/commandLines';

test('a one-shot is -w and -p, in the workspace, without colour', () => {
    const line = oneShot('/usr/bin/orangu', '/work/my project', 'what does "x" do?\nand y', { HOME: '/home/me' });
    assert.equal(line.command, '/usr/bin/orangu');
    // Arguments are passed as a list, so spaces, quotes and new lines stay as they are.
    assert.deepEqual(line.args, ['-w', '/work/my project', '-p', 'what does "x" do?\nand y']);
    assert.equal(line.cwd, '/work/my project');
    assert.deepEqual(line.env, { HOME: '/home/me', NO_COLOR: '1' });
});

test('the terminal runs orangu itself in the workspace', () => {
    assert.deepEqual(interactive('C:\\bin\\orangu.exe', 'C:\\work\\app'), {
        name: 'orangu',
        shellPath: 'C:\\bin\\orangu.exe',
        shellArgs: ['-w', 'C:\\work\\app'],
        cwd: 'C:\\work\\app',
    });
    assert.equal(interactive('/bin/orangu', '/w', 'orangu: w').name, 'orangu: w');
});

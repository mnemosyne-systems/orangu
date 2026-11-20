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
import { locate, Machine } from '../../src/locator';

function linux(files: string[], pathVariable = '/usr/local/bin:/usr/bin'): Machine {
    return { pathVariable, home: '/home/me', windows: false, isFile: (p) => files.includes(p) };
}

function windows(files: string[], pathVariable = 'C:\\Windows;"C:\\Tools\\orangu"'): Machine {
    return { pathVariable, home: 'C:\\Users\\me', windows: true, isFile: (p) => files.includes(p) };
}

test('the first orangu on the PATH is used', () => {
    const machine = linux(['/usr/bin/orangu', '/home/me/.cargo/bin/orangu']);
    assert.deepEqual(locate('', machine), { path: '/usr/bin/orangu' });
});

test('cargo install is found when it is not on the PATH', () => {
    assert.deepEqual(locate(undefined, linux(['/home/me/.cargo/bin/orangu'])), { path: '/home/me/.cargo/bin/orangu' });
    assert.deepEqual(locate('', windows(['C:\\Users\\me\\.cargo\\bin\\orangu.exe'])), {
        path: 'C:\\Users\\me\\.cargo\\bin\\orangu.exe',
    });
});

test('on Windows the binary is orangu.exe, and quoted PATH entries count', () => {
    assert.deepEqual(locate('', windows(['C:\\Tools\\orangu\\orangu.exe'])), { path: 'C:\\Tools\\orangu\\orangu.exe' });
});

test('the setting wins, as a file or as the folder it is in', () => {
    const machine = linux(['/opt/orangu/orangu', '/usr/bin/orangu']);
    assert.deepEqual(locate('/opt/orangu/orangu', machine), { path: '/opt/orangu/orangu' });
    assert.deepEqual(locate('  "/opt/orangu"  ', machine), { path: '/opt/orangu/orangu' });
});

test('a setting that does not exist is reported, not replaced by the PATH', () => {
    const result = locate('/nowhere/orangu', linux(['/usr/bin/orangu']));
    assert.ok('error' in result);
    assert.match(result.error, /orangu\.path.*\/nowhere\/orangu/);
});

test('nothing found says how to fix it', () => {
    const result = locate('', linux([]));
    assert.ok('error' in result);
    assert.match(result.error, /Could not find orangu on the PATH/);
    assert.match(result.error, /orangu\.path/);
});

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
import { chooseWorkspace } from '../../src/workspace';

const nothing = () => false;

test('the folder of the active file, among several', () => {
    const folders = ['/work/api', '/work/web'];
    assert.equal(chooseWorkspace(folders, '/work/web/src/app.ts', nothing), '/work/web');
});

test('nested folders give the innermost', () => {
    const folders = ['/work', '/work/web'];
    assert.equal(chooseWorkspace(folders, '/work/web/src/app.ts', nothing), '/work/web');
    assert.equal(chooseWorkspace(folders, '/work/README.md', nothing), '/work');
});

test('a folder whose name only starts the same is not the file’s', () => {
    assert.equal(chooseWorkspace(['/work/app', '/work/application'], '/work/application/a.ts', nothing), '/work/application');
});

test('no active file, or one outside every folder, gives the first folder', () => {
    assert.equal(chooseWorkspace(['/work/api', '/work/web'], undefined, nothing), '/work/api');
    assert.equal(chooseWorkspace(['/work/api'], '/tmp/scratch.ts', nothing), '/work/api');
});

test('a file opened on its own runs in its Git repository', () => {
    const exists = (p: string) => p === '/home/me/repo/.git';
    assert.equal(chooseWorkspace([], '/home/me/repo/src/deep/a.rs', exists), '/home/me/repo');
});

test('a Windows file opened on its own finds its repository too', () => {
    const exists = (p: string) => p === 'C:\\code\\repo\\.git';
    assert.equal(chooseWorkspace([], 'C:\\code\\repo\\src\\a.cs', exists), 'C:\\code\\repo');
});

test('outside any repository, the file’s folder', () => {
    assert.equal(chooseWorkspace([], '/tmp/notes/a.md', nothing), '/tmp/notes');
});

test('nothing open gives nothing', () => {
    assert.equal(chooseWorkspace([], undefined, nothing), undefined);
});

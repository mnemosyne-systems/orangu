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
import { AnsiFilter } from '../../src/ansi';

test('plain text passes through, without carriage returns', () => {
    assert.equal(new AnsiFilter().push('one\r\ntwo\n'), 'one\ntwo\n');
});

test('colours and cursor moves are removed', () => {
    assert.equal(new AnsiFilter().push('\u001b[1;31mred\u001b[0m and \u001b[2Kplain'), 'red and plain');
});

test('a sequence cut between two pieces is held back until it ends', () => {
    const filter = new AnsiFilter();
    assert.equal(filter.push('a\u001b'), 'a');
    assert.equal(filter.push('[3'), '');
    assert.equal(filter.push('2mb'), 'b');
});

test('titles and links end at a bell or at ESC backslash', () => {
    const filter = new AnsiFilter();
    assert.equal(filter.push('\u001b]0;title\u0007x'), 'x');
    assert.equal(filter.push('\u001b]8;;https://example.org\u001b\\link\u001b]8;;\u001b\\'), 'link');
});

test('a two-character sequence is dropped whole', () => {
    assert.equal(new AnsiFilter().push('a\u001b7b\u001b8c'), 'abc');
});

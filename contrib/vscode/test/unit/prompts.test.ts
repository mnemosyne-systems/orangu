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
import {
    aboutSelection,
    autoReviewArgument,
    displayPath,
    reviewFile,
    selectionLabel,
} from '../../src/prompts';

const selection = {
    filePath: '/work/app/src/parser.rs',
    startLine: 12,
    endLine: 14,
    text: 'fn parse() {\r\n    todo!()\r\n}\r\n',
};

test('a question about a selection carries the file, the lines and the code', () => {
    assert.equal(
        aboutSelection('/work/app', selection, '  Why is this slow?  '),
        'Why is this slow?\n\nThis is lines 12-14 of src/parser.rs:\n\n```rust\nfn parse() {\n    todo!()\n}\n```',
    );
});

test('an empty question asks what the code does, and one line says line', () => {
    const one = { ...selection, endLine: 12, text: 'let x = 1;' };
    assert.equal(
        aboutSelection('/work/app', one, ' '),
        'Explain what this code does.\n\nThis is line 12 of src/parser.rs:\n\n```rust\nlet x = 1;\n```',
    );
});

test('the fence is longer than any backticks in the code', () => {
    const fenced = { ...selection, filePath: '/work/app/README.md', text: '````\ncode\n````' };
    assert.match(aboutSelection('/work/app', fenced, 'q'), /\n`````markdown\n````\ncode\n````\n`````$/);
});

test('an unknown extension gets a fence without a language', () => {
    const plain = { ...selection, filePath: '/work/app/notes.txt', text: 'x' };
    assert.match(aboutSelection('/work/app', plain, 'q'), /\n```\nx\n```$/);
});

test('the label is the short path and the lines', () => {
    assert.equal(selectionLabel('/work/app', selection), 'src/parser.rs:12-14');
    assert.equal(selectionLabel('/work/app', { ...selection, endLine: 12 }), 'src/parser.rs:12');
});

test('paths are relative with forward slashes, or whole when outside', () => {
    assert.equal(displayPath('/work/app', '/work/app/a/b.ts'), 'a/b.ts');
    assert.equal(displayPath('/work/app', '/work/other/b.ts'), '/work/other/b.ts');
    assert.equal(displayPath('/work/app', '/work/application/b.ts'), '/work/application/b.ts');
    assert.equal(displayPath('C:\\work\\app', 'C:\\work\\app\\src\\Main.cs'), 'src/Main.cs');
    assert.equal(displayPath('C:\\work\\app', 'D:\\elsewhere\\Main.cs'), 'D:\\elsewhere\\Main.cs');
});

test('a file review is orangu’s /auto_review of that file', () => {
    assert.deepEqual(reviewFile('/work/app', '/work/app/src/parser.rs'), { prompt: '/auto_review src/parser.rs' });
    assert.deepEqual(reviewFile('C:\\work\\app', 'C:\\work\\app\\src\\Main.cs'), { prompt: '/auto_review src/Main.cs' });
});

test('a file outside the workspace has no review', () => {
    const review = reviewFile('/work/app', '/tmp/scratch.rs');
    assert.ok('error' in review);
    assert.match(review.error, /outside the workspace/);
});

test('an ordinary path is one argument as it is', () => {
    assert.equal(autoReviewArgument('src/main.rs'), 'src/main.rs');
    assert.equal(autoReviewArgument('Makefile'), 'Makefile');
    // `]` and `}` alone do not make orangu read a pattern.
    assert.equal(autoReviewArgument('notes/a]b}.md'), 'notes/a]b}.md');
});

test('glob characters are matched literally, and spaces by ?', () => {
    assert.equal(autoReviewArgument('app/[id]/page.tsx'), 'app/[[]id[]]/page.tsx');
    assert.equal(autoReviewArgument('src/what?.rs'), 'src/what[?].rs');
    assert.equal(autoReviewArgument('src/{a,b}.rs'), 'src/[{]a,b[}].rs');
    assert.equal(autoReviewArgument('src/star*.rs'), 'src/star[*].rs');
    assert.equal(autoReviewArgument('docs/my notes.md'), 'docs/my?notes.md');
});

test('a top-level name is escaped the same way', () => {
    assert.equal(autoReviewArgument('my notes.md'), 'my?notes.md');
    assert.equal(autoReviewArgument('[x].ts'), '[[]x[]].ts');
});

test('a file review argument never splits into two', () => {
    for (const file of ['a b/c d.rs', 'tab\there.rs', '[id] x/{y}.ts']) {
        assert.equal(autoReviewArgument(file).split(/\s+/).length, 1, file);
    }
});

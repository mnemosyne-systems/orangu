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

import * as path from 'path';

/** A selection in the editor, attached to the next prompt. Lines count from 1. */
export interface EditorSelection {
    filePath: string;
    startLine: number;
    endLine: number;
    text: string;
}

/** What the prompt box is filled with after a selection is attached. */
export const DEFAULT_SELECTION_QUESTION = 'Explain what this code does.';

/** orangu's review of everything the branch changes. */
export const REVIEW_CHANGES = '/auto_review';

const LANGUAGES: Record<string, string> = {
    '.c': 'c',
    '.h': 'c',
    '.cc': 'cpp',
    '.cpp': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.cs': 'csharp',
    '.rs': 'rust',
    '.py': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.jsx': 'jsx',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.java': 'java',
    '.kt': 'kotlin',
    '.go': 'go',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.sql': 'sql',
    '.json': 'json',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
    '.yml': 'yaml',
    '.yaml': 'yaml',
    '.md': 'markdown',
    '.sh': 'bash',
    '.ps1': 'powershell',
    '.toml': 'toml',
};

/**
 * A question about a selection: the question, then where the code is and
 * the code itself, so the model can also open the file around it.
 */
export function aboutSelection(workspace: string, selection: EditorSelection, question: string): string {
    const asked = question.trim().length > 0 ? question.trim() : DEFAULT_SELECTION_QUESTION;
    const lines =
        selection.startLine === selection.endLine
            ? `line ${selection.startLine}`
            : `lines ${selection.startLine}-${selection.endLine}`;
    const code = selection.text.replace(/\r\n/g, '\n').replace(/\n+$/, '');
    const fence = fenceFor(code);
    return (
        `${asked}\n\nThis is ${lines} of ${displayPath(workspace, selection.filePath)}:\n\n` +
        `${fence}${language(selection.filePath)}\n${code}\n${fence}`
    );
}

/** The short form shown for an attached selection: `src/a.ts:3-9`. */
export function selectionLabel(workspace: string, selection: EditorSelection): string {
    const file = displayPath(workspace, selection.filePath);
    return selection.startLine === selection.endLine
        ? `${file}:${selection.startLine}`
        : `${file}:${selection.startLine}-${selection.endLine}`;
}

/**
 * orangu's own review of one file, `/auto_review <file>`: on main or master
 * the whole file, on a branch the file's changes. orangu takes the file
 * relative to the workspace, so a file outside it has no review.
 */
export function reviewFile(workspace: string, filePath: string): { prompt: string } | { error: string } {
    const relative = relativeInside(workspace, filePath);
    if (relative === undefined) {
        return { error: `${filePath} is outside the workspace ${workspace}, so orangu cannot review it.` };
    }
    return { prompt: `${REVIEW_CHANGES} ${autoReviewArgument(relative)}` };
}

/**
 * A workspace path as one `/auto_review` argument. orangu splits the
 * arguments on whitespace and reads one with `*`, `?`, `[` or `{` in it as
 * a pattern, so a path such as `app/[id]/page.tsx` is given as a pattern
 * that matches only it: each special character in a class of its own. A
 * space cannot be kept, so it becomes `?`, which matches it (and, rarely,
 * another file differing only there). As with a bare name typed in orangu,
 * a file at the top of the workspace is matched by its name, so a file of
 * the same name deeper down is reviewed with it.
 */
export function autoReviewArgument(relative: string): string {
    if (!/[*?[{\s]/.test(relative)) {
        return relative;
    }
    let pattern = '';
    for (const c of relative) {
        if ('*?[]{}'.includes(c)) {
            pattern += `[${c}]`;
        } else if (/\s/.test(c)) {
            pattern += '?';
        } else {
            pattern += c;
        }
    }
    return pattern;
}

/**
 * A file as orangu's tools take it: relative to the workspace with forward
 * slashes, or the full path when it is outside it.
 */
export function displayPath(workspace: string, filePath: string): string {
    return relativeInside(workspace, filePath) ?? filePath;
}

function relativeInside(workspace: string, filePath: string): string | undefined {
    const paths = isWindowsPath(workspace) || isWindowsPath(filePath) ? path.win32 : path.posix;
    const relative = paths.relative(workspace, filePath);
    if (relative.length === 0 || relative === '..' || relative.startsWith(`..${paths.sep}`) || paths.isAbsolute(relative)) {
        return undefined;
    }
    return relative.split(paths.sep).join('/');
}

function isWindowsPath(value: string): boolean {
    return /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\');
}

function language(filePath: string): string {
    return LANGUAGES[path.extname(filePath).toLowerCase()] ?? '';
}

/** A fence longer than any run of backticks in the code. */
function fenceFor(code: string): string {
    let longest = 0;
    let run = 0;
    for (const c of code) {
        run = c === '`' ? run + 1 : 0;
        longest = Math.max(longest, run);
    }
    return '`'.repeat(Math.max(3, longest + 1));
}

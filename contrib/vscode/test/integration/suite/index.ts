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

// Runs in VS Code itself, with the extension loaded: what VS Code only
// checks at run time (the commands, the view, the setting), and the
// commands end to end against the stand-in orangu.

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import type { OranguApi } from '../../../src/extension';

const EXTENSION_ID = 'mnemosyne-systems.orangu';
const root = process.env.ORANGU_TEST_ROOT ?? path.resolve(__dirname, '..', '..', '..', '..');
const standIn = path.join(root, 'test', 'fixtures', 'orangu');

type Test = [name: string, body: () => Promise<void>];
const tests: Test[] = [];

function test(name: string, body: () => Promise<void>): void {
    tests.push([name, body]);
}

/** The stand-in is a shell script, so the tests that start it need a POSIX system. */
function posixTest(name: string, body: () => Promise<void>): void {
    test(name, async () => {
        if (process.platform === 'win32') {
            console.log(`    (skipped on Windows)`);
            return;
        }
        await body();
    });
}

async function waitFor(what: string, done: () => boolean, timeoutMs = 10000): Promise<void> {
    const start = Date.now();
    while (!done()) {
        if (Date.now() - start > timeoutMs) {
            throw new Error(`timed out waiting for ${what}`);
        }
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
}

async function api(): Promise<OranguApi> {
    const extension = vscode.extensions.getExtension<OranguApi>(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} is not installed`);
    return extension.activate();
}

async function setPath(value: string): Promise<void> {
    await vscode.workspace.getConfiguration('orangu').update('path', value, vscode.ConfigurationTarget.Global);
}

function workspaceFolder(): string {
    const folder = vscode.workspace.workspaceFolders?.[0];
    assert.ok(folder, 'the tests run with a workspace folder open');
    return folder.uri.fsPath;
}

async function openSample(relative: string): Promise<vscode.TextEditor> {
    const document = await vscode.workspace.openTextDocument(path.join(workspaceFolder(), relative));
    return vscode.window.showTextDocument(document);
}

function lastAnswer(panel: OranguApi['panel']): string {
    const entries = panel.transcript.snapshot().entries;
    return entries.filter((entry) => entry.kind === 'answer').pop()?.text ?? '';
}

/** The report the stand-in prints: what it was started with. */
function report(panel: OranguApi['panel']): { args: string[]; cwd: string; noColor: string | null; stdinClosed: boolean } {
    return JSON.parse(lastAnswer(panel));
}

test('the extension activates and hands back its panel', async () => {
    const orangu = await api();
    assert.ok(orangu.panel);
});

test('every command in package.json is registered', async () => {
    await api();
    const manifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
    const declared: string[] = manifest.contributes.commands.map((c: { command: string }) => c.command);
    const registered = new Set(await vscode.commands.getCommands(true));
    for (const command of declared) {
        assert.ok(registered.has(command), `${command} is declared but not registered`);
    }
    assert.ok(declared.length >= 7);
});

test('the orangu.path setting is there, empty by default', async () => {
    const setting = vscode.workspace.getConfiguration('orangu').inspect<string>('path');
    assert.equal(setting?.defaultValue, '');
});

test('the panel opens in its own view', async () => {
    await api();
    await vscode.commands.executeCommand('orangu.showPanel');
    await vscode.commands.executeCommand('orangu.panel.focus');
});

posixTest('a prompt runs orangu -w <workspace> -p <prompt> and streams the answer', async () => {
    const orangu = await api();
    fs.chmodSync(standIn, 0o755);
    await setPath(standIn);
    orangu.panel.clear();
    await orangu.panel.ask('show git status');
    const result = report(orangu.panel);
    assert.deepEqual(result.args, ['-w', workspaceFolder(), '-p', 'show git status']);
    assert.equal(fs.realpathSync(result.cwd), fs.realpathSync(workspaceFolder()));
    assert.equal(result.noColor, '1');
    assert.equal(result.stdinClosed, true);
    const snapshot = orangu.panel.transcript.snapshot();
    assert.equal(snapshot.entries[0].text, 'show git status');
    assert.match(snapshot.status, /^Done in /);
    assert.equal(snapshot.busy, false);
});

posixTest('a selection is attached, then asked about with its file and lines', async () => {
    const orangu = await api();
    await setPath(standIn);
    orangu.panel.clear();
    const editor = await openSample('src/sample.ts');
    // Lines 2 and 3, up to the start of line 4, which does not count.
    editor.selection = new vscode.Selection(1, 0, 3, 0);
    await vscode.commands.executeCommand('orangu.askAboutSelection');
    assert.equal(orangu.panel.transcript.snapshot().selection, 'src/sample.ts:2-3');
    await orangu.panel.ask('');
    const prompt = report(orangu.panel).args[3];
    assert.equal(
        prompt,
        'Explain what this code does.\n\nThis is lines 2-3 of src/sample.ts:\n\n```typescript\n    return a + b;\n}\n```',
    );
    // Sent once, then gone.
    assert.equal(orangu.panel.transcript.snapshot().selection, undefined);
});

posixTest('with nothing selected, the line the cursor is on is attached', async () => {
    const orangu = await api();
    const editor = await openSample('src/sample.ts');
    editor.selection = new vscode.Selection(5, 3, 5, 3);
    await vscode.commands.executeCommand('orangu.askAboutSelection');
    assert.equal(orangu.panel.transcript.snapshot().selection, 'src/sample.ts:6');
    orangu.panel.removeSelection();
    assert.equal(orangu.panel.transcript.snapshot().selection, undefined);
});

posixTest('Review This File runs /auto_review on it, and a path with brackets is matched literally', async () => {
    const orangu = await api();
    await setPath(standIn);
    await openSample('src/sample.ts');
    await vscode.commands.executeCommand('orangu.reviewFile');
    assert.equal(report(orangu.panel).args[3], '/auto_review src/sample.ts');
    await openSample('app/[id]/page.ts');
    await vscode.commands.executeCommand('orangu.reviewFile');
    assert.equal(report(orangu.panel).args[3], '/auto_review app/[[]id[]]/page.ts');
});

posixTest('an unsaved file is reviewed as saved, and the panel says so', async () => {
    const orangu = await api();
    await setPath(standIn);
    const editor = await openSample('src/sample.ts');
    await editor.edit((edit) => edit.insert(new vscode.Position(0, 0), '// changed\n'));
    await vscode.commands.executeCommand('orangu.reviewFile');
    const shown = orangu.panel.transcript.snapshot().entries.filter((e) => e.kind === 'prompt').pop()?.text ?? '';
    assert.match(shown, /unsaved changes/);
    await vscode.commands.executeCommand('workbench.action.files.revert');
});

posixTest('Review Changes runs /auto_review', async () => {
    const orangu = await api();
    await setPath(standIn);
    await vscode.commands.executeCommand('orangu.reviewChanges');
    assert.equal(report(orangu.panel).args[3], '/auto_review');
});

posixTest('a failing run shows orangu’s error', async () => {
    const orangu = await api();
    await setPath(standIn);
    await orangu.panel.ask('fail');
    assert.equal(orangu.panel.transcript.snapshot().status, 'orangu failed: error: the stand-in server is not running');
});

posixTest('Stop ends a run', async () => {
    const orangu = await api();
    await setPath(standIn);
    const running = orangu.panel.ask('sleep');
    await waitFor('the run to start', () => /sleeping/.test(orangu.panel.transcript.snapshot().status));
    await vscode.commands.executeCommand('orangu.stop');
    await running;
    assert.equal(orangu.panel.transcript.snapshot().status, 'Stopped');
    assert.equal(orangu.panel.transcript.snapshot().busy, false);
});

posixTest('Open orangu starts the interface in a terminal, and shows it again rather than a second one', async () => {
    await api();
    await setPath(standIn);
    await vscode.commands.executeCommand('orangu.open');
    await waitFor('the terminal', () => vscode.window.terminals.some((t) => t.name === 'orangu'));
    await vscode.commands.executeCommand('orangu.open');
    const terminals = vscode.window.terminals.filter((t) => t.name === 'orangu');
    assert.equal(terminals.length, 1);
    const options = terminals[0].creationOptions as vscode.TerminalOptions;
    assert.equal(options.shellPath, standIn);
    assert.deepEqual(options.shellArgs, ['-w', workspaceFolder()]);
    terminals[0].dispose();
});

test('a path that does not exist is reported in the panel', async () => {
    const orangu = await api();
    await setPath(path.join(root, 'no-such-orangu'));
    await orangu.panel.ask('hello');
    assert.match(orangu.panel.transcript.snapshot().status, /orangu\.path.*no-such-orangu/);
    await setPath('');
});

export async function run(): Promise<void> {
    let failed = 0;
    for (const [name, body] of tests) {
        try {
            await body();
            console.log(`  ok   ${name}`);
        } catch (err) {
            failed++;
            console.log(`  FAIL ${name}\n${err instanceof Error ? err.stack : String(err)}`);
        }
    }
    console.log(`${tests.length - failed} passed, ${failed} failed`);
    if (failed > 0) {
        throw new Error(`${failed} VS Code test(s) failed`);
    }
}

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

import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { interactive } from './commandLines';
import { locate, thisMachine } from './locator';
import { OranguPanel, Target } from './panel';
import { chooseWorkspace } from './workspace';

/** What the extension hands back from `activate`, for its tests. */
export interface OranguApi {
    panel: OranguPanel;
    target(file?: string): Target;
}

/** The schemes of files orangu can read: on this machine, or on the remote it runs on. */
const FILE_SCHEMES = new Set(['file', 'vscode-remote']);

export function activate(context: vscode.ExtensionContext): OranguApi {
    const panel = new OranguPanel(context.extensionUri, target);
    const terminals = new Map<string, vscode.Terminal>();

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(OranguPanel.viewId, panel),
        vscode.window.onDidCloseTerminal((closed) => {
            for (const [workspace, terminal] of terminals) {
                if (terminal === closed) {
                    terminals.delete(workspace);
                }
            }
        }),
        vscode.commands.registerCommand('orangu.open', () => open(terminals)),
        vscode.commands.registerCommand('orangu.showPanel', () => showPanel()),
        vscode.commands.registerCommand('orangu.askAboutSelection', async () => {
            const editor = editorWithFile();
            if (editor === undefined) {
                return;
            }
            const { document, selection } = editor;
            if (selection.isEmpty) {
                // Nothing selected means the line the cursor is on.
                const line = document.lineAt(selection.active.line);
                panel.attachSelection({
                    filePath: document.uri.fsPath,
                    startLine: line.lineNumber + 1,
                    endLine: line.lineNumber + 1,
                    text: line.text,
                });
            } else {
                // A selection that ends at the start of a line does not take that line.
                const end =
                    selection.end.character === 0 && selection.end.line > selection.start.line
                        ? selection.end.line - 1
                        : selection.end.line;
                panel.attachSelection({
                    filePath: document.uri.fsPath,
                    startLine: selection.start.line + 1,
                    endLine: end + 1,
                    text: document.getText(selection),
                });
            }
            await showPanel();
        }),
        vscode.commands.registerCommand('orangu.reviewFile', async (uri?: vscode.Uri) => {
            const document = uri !== undefined ? await vscode.workspace.openTextDocument(uri) : editorWithFile()?.document;
            if (document === undefined) {
                return;
            }
            await showPanel();
            await panel.reviewFile(document);
        }),
        vscode.commands.registerCommand('orangu.reviewChanges', async () => {
            await showPanel();
            await panel.reviewChanges();
        }),
        vscode.commands.registerCommand('orangu.stop', () => panel.stop()),
        vscode.commands.registerCommand('orangu.clear', () => panel.clear()),
    );

    return { panel, target };
}

export function deactivate(): void {
    // Nothing to do: a run that is going ends with the extension host.
}

/** The binary and the workspace, for the active file or the one given. */
function target(file?: string): Target {
    const configured = vscode.workspace.getConfiguration('orangu').get<string>('path');
    const located = locate(configured, thisMachine());
    if ('error' in located) {
        return located;
    }
    const folders = (vscode.workspace.workspaceFolders ?? [])
        .filter((folder) => FILE_SCHEMES.has(folder.uri.scheme))
        .map((folder) => folder.uri.fsPath);
    const workspace = chooseWorkspace(folders, file ?? activeFile(), fs.existsSync);
    if (workspace === undefined) {
        return { error: 'Open a folder or a file first: orangu runs in its folder.' };
    }
    return { binary: located.path, workspace };
}

/** Starts orangu's full interface in a terminal, or shows the one already open for this workspace. */
function open(terminals: Map<string, vscode.Terminal>): void {
    const resolved = target();
    if ('error' in resolved) {
        void vscode.window.showErrorMessage(resolved.error);
        return;
    }
    const existing = terminals.get(resolved.workspace);
    if (existing !== undefined && existing.exitStatus === undefined) {
        existing.show();
        return;
    }
    const several = (vscode.workspace.workspaceFolders?.length ?? 0) > 1;
    const name = several ? `orangu: ${path.basename(resolved.workspace)}` : 'orangu';
    const terminal = vscode.window.createTerminal(interactive(resolved.binary, resolved.workspace, name));
    terminals.set(resolved.workspace, terminal);
    terminal.show();
}

async function showPanel(): Promise<void> {
    await vscode.commands.executeCommand(`${OranguPanel.viewId}.focus`);
}

function editorWithFile(): vscode.TextEditor | undefined {
    const editor = vscode.window.activeTextEditor;
    if (editor === undefined || !FILE_SCHEMES.has(editor.document.uri.scheme)) {
        void vscode.window.showInformationMessage('orangu works on files saved in a folder; this editor has none.');
        return undefined;
    }
    return editor;
}

function activeFile(): string | undefined {
    const document = vscode.window.activeTextEditor?.document;
    return document !== undefined && FILE_SCHEMES.has(document.uri.scheme) ? document.uri.fsPath : undefined;
}

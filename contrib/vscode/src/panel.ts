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

import * as crypto from 'crypto';
import * as vscode from 'vscode';
import { oneShot } from './commandLines';
import {
    aboutSelection,
    DEFAULT_SELECTION_QUESTION,
    EditorSelection,
    REVIEW_CHANGES,
    reviewFile,
    selectionLabel,
} from './prompts';
import { runOneShot } from './runner';
import { Change, Transcript } from './transcript';

/** Where orangu runs and which binary, or why it cannot. */
export type Target = { binary: string; workspace: string } | { error: string };

/** A message from the panel's page. */
type FromPage =
    | { type: 'ready' }
    | { type: 'ask'; text: string }
    | { type: 'stop' }
    | { type: 'clear' }
    | { type: 'open' }
    | { type: 'reviewChanges' }
    | { type: 'removeSelection' };

/**
 * The orangu panel: a prompt box, and the answers as orangu writes them.
 * Each prompt is one `orangu -w <workspace> -p <prompt>`.
 */
export class OranguPanel implements vscode.WebviewViewProvider {
    static readonly viewId = 'orangu.panel';

    readonly transcript: Transcript;
    private view: vscode.WebviewView | undefined;
    private attached: EditorSelection | undefined;
    private running: AbortController | undefined;
    private pendingFill: string | undefined;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly target: (file?: string) => Target,
    ) {
        this.transcript = new Transcript((change) => this.forward(change));
    }

    resolveWebviewView(view: vscode.WebviewView): void {
        this.view = view;
        const media = vscode.Uri.joinPath(this.extensionUri, 'media');
        view.webview.options = { enableScripts: true, localResourceRoots: [media] };
        view.webview.html = this.html(view.webview, media);
        view.webview.onDidReceiveMessage((message: FromPage) => this.receive(message));
        view.onDidDispose(() => {
            if (this.view === view) {
                this.view = undefined;
            }
        });
    }

    /** Asks orangu; with a selection attached, about that selection. */
    async ask(text: string): Promise<void> {
        const selection = this.attached;
        if (text.trim().length === 0 && selection === undefined) {
            return;
        }
        const target = this.resolve(selection?.filePath);
        if (target === undefined) {
            return;
        }
        let prompt = text.trim();
        let shown = prompt;
        if (selection !== undefined) {
            prompt = aboutSelection(target.workspace, selection, text);
            const question = text.trim().length > 0 ? text.trim() : DEFAULT_SELECTION_QUESTION;
            shown = `${question}\n(${selectionLabel(target.workspace, selection)})`;
            this.removeSelection();
        }
        await this.run(target, prompt, shown);
    }

    /** orangu's review of everything the branch changes. */
    async reviewChanges(): Promise<void> {
        const target = this.resolve();
        if (target !== undefined) {
            await this.run(target, REVIEW_CHANGES, REVIEW_CHANGES);
        }
    }

    /** orangu's review of one file, as it is on disk. */
    async reviewFile(document: vscode.TextDocument): Promise<void> {
        const target = this.resolve(document.uri.fsPath);
        if (target === undefined) {
            return;
        }
        const review = reviewFile(target.workspace, document.uri.fsPath);
        if ('error' in review) {
            this.transcript.fail(review.error);
            void vscode.window.showWarningMessage(review.error);
            return;
        }
        // orangu reads the file from disk and Git, not from the editor.
        const shown = document.isDirty
            ? `${review.prompt}\n(the file has unsaved changes; orangu reviews it as it was last saved)`
            : review.prompt;
        await this.run(target, review.prompt, shown);
    }

    /** Attaches a selection to the next prompt, and puts a question in the box if it is empty. */
    attachSelection(selection: EditorSelection): void {
        const target = this.target(selection.filePath);
        const workspace = 'workspace' in target ? target.workspace : '';
        this.attached = selection;
        this.transcript.attach(selectionLabel(workspace, selection));
        // The page may not be open yet; it gets the question once it is.
        this.pendingFill = DEFAULT_SELECTION_QUESTION;
        this.sendFill();
    }

    removeSelection(): void {
        this.attached = undefined;
        this.transcript.attach(undefined);
    }

    stop(): void {
        this.running?.abort();
    }

    clear(): void {
        this.transcript.clear();
    }

    private async run(target: { binary: string; workspace: string }, prompt: string, shown: string): Promise<void> {
        if (this.running !== undefined) {
            void vscode.window.showInformationMessage('orangu is still answering; stop it first.');
            return;
        }
        const controller = new AbortController();
        this.running = controller;
        void vscode.commands.executeCommand('setContext', 'orangu.running', true);
        const started = Date.now();
        this.transcript.begin(shown);
        try {
            const result = await runOneShot(
                oneShot(target.binary, target.workspace, prompt),
                (text) => this.transcript.answer(text),
                (line) => this.transcript.note(line),
                controller.signal,
            );
            this.transcript.end(result, Date.now() - started);
        } finally {
            this.running = undefined;
            void vscode.commands.executeCommand('setContext', 'orangu.running', false);
        }
    }

    /** The binary and workspace, or the error shown in the panel and as a message. */
    private resolve(file?: string): { binary: string; workspace: string } | undefined {
        const target = this.target(file);
        if ('error' in target) {
            this.transcript.fail(target.error);
            void vscode.window.showErrorMessage(target.error, 'Open Settings').then((choice) => {
                if (choice === 'Open Settings') {
                    void vscode.commands.executeCommand('workbench.action.openSettings', 'orangu.path');
                }
            });
            return undefined;
        }
        return target;
    }

    private receive(message: FromPage): void {
        switch (message.type) {
            case 'ready':
                this.forward({ type: 'reset', snapshot: this.transcript.snapshot() });
                this.sendFill();
                break;
            case 'ask':
                void this.ask(message.text);
                break;
            case 'stop':
                this.stop();
                break;
            case 'clear':
                this.clear();
                break;
            case 'open':
                void vscode.commands.executeCommand('orangu.open');
                break;
            case 'reviewChanges':
                void this.reviewChanges();
                break;
            case 'removeSelection':
                this.removeSelection();
                break;
        }
    }

    private forward(change: Change): void {
        void this.view?.webview.postMessage(change);
    }

    /** Puts the pending question in the prompt box, once the page is there to take it. */
    private sendFill(): void {
        if (this.pendingFill === undefined || this.view === undefined) {
            return;
        }
        const text = this.pendingFill;
        void this.view.webview.postMessage({ type: 'fill', text }).then((delivered) => {
            if (delivered && this.pendingFill === text) {
                this.pendingFill = undefined;
            }
        });
    }

    private html(webview: vscode.Webview, media: vscode.Uri): string {
        const nonce = crypto.randomBytes(16).toString('base64');
        const style = webview.asWebviewUri(vscode.Uri.joinPath(media, 'panel.css'));
        const script = webview.asWebviewUri(vscode.Uri.joinPath(media, 'panel.js'));
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="${style}" rel="stylesheet">
<title>orangu</title>
</head>
<body>
<div id="transcript" aria-live="polite"></div>
<div id="empty">Ask orangu anything it answers in the terminal: a question, a change to make, or a command such as <code>/status</code> or <code>/diff</code>.</div>
<div id="attachment" hidden><span id="label"></span><button id="remove" class="link" title="Remove the selection">Remove</button></div>
<textarea id="prompt" rows="3" placeholder="Ask orangu… (Enter to send, Shift+Enter for a new line)"></textarea>
<div id="buttons">
<button id="ask">Ask</button>
<button id="stop" class="secondary" disabled>Stop</button>
<button id="review" class="secondary" title="/auto_review: orangu's review of the changes on this branch">Review Changes</button>
<button id="open" class="secondary" title="orangu's full interface in a terminal">Open orangu</button>
<button id="clear" class="secondary">Clear</button>
</div>
<div id="status"></div>
<script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
    }
}

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

// The orangu panel's page. The extension keeps the state; this draws it
// and sends back what was typed and pressed.
(function () {
    const vscode = acquireVsCodeApi();
    const transcript = document.getElementById('transcript');
    const empty = document.getElementById('empty');
    const attachment = document.getElementById('attachment');
    const label = document.getElementById('label');
    const prompt = document.getElementById('prompt');
    const status = document.getElementById('status');
    const buttons = {
        ask: document.getElementById('ask'),
        stop: document.getElementById('stop'),
        review: document.getElementById('review'),
        open: document.getElementById('open'),
        clear: document.getElementById('clear'),
        remove: document.getElementById('remove'),
    };
    let busy = false;
    let hasSelection = false;

    function nearBottom() {
        return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 40;
    }

    function scrollDown(follow) {
        if (follow) {
            transcript.scrollTop = transcript.scrollHeight;
        }
    }

    function draw(snapshot) {
        const follow = nearBottom();
        transcript.replaceChildren(
            ...snapshot.entries.map((entry) => {
                const block = document.createElement('div');
                block.className = entry.kind;
                block.textContent = entry.text;
                return block;
            }),
        );
        empty.hidden = snapshot.entries.length > 0;
        status.textContent = snapshot.status;
        busy = snapshot.busy;
        hasSelection = snapshot.selection !== undefined;
        attachment.hidden = !hasSelection;
        label.textContent = snapshot.selection ?? '';
        buttons.ask.disabled = busy;
        buttons.review.disabled = busy;
        buttons.stop.disabled = !busy;
        scrollDown(follow);
    }

    function ask() {
        if (busy || (prompt.value.trim().length === 0 && !hasSelection)) {
            return;
        }
        vscode.postMessage({ type: 'ask', text: prompt.value });
        prompt.value = '';
    }

    window.addEventListener('message', (event) => {
        const message = event.data;
        switch (message.type) {
            case 'reset':
                draw(message.snapshot);
                break;
            case 'answer': {
                const follow = nearBottom();
                const last = transcript.lastElementChild;
                if (last && last.className === 'answer') {
                    last.textContent += message.text;
                }
                scrollDown(follow);
                break;
            }
            case 'fill':
                if (prompt.value.trim().length === 0) {
                    prompt.value = message.text;
                }
                prompt.focus();
                prompt.select();
                break;
        }
    });

    prompt.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            ask();
        }
    });
    buttons.ask.addEventListener('click', ask);
    buttons.stop.addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
    buttons.review.addEventListener('click', () => vscode.postMessage({ type: 'reviewChanges' }));
    buttons.open.addEventListener('click', () => vscode.postMessage({ type: 'open' }));
    buttons.clear.addEventListener('click', () => vscode.postMessage({ type: 'clear' }));
    buttons.remove.addEventListener('click', () => vscode.postMessage({ type: 'removeSelection' }));

    vscode.postMessage({ type: 'ready' });
})();

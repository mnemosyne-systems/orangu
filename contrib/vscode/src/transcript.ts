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

import { RunResult } from './runner';

/** A prompt as it was asked, or the answer to it. */
export interface Entry {
    kind: 'prompt' | 'answer';
    text: string;
}

/** Everything the panel shows, so a panel that is opened again starts from it. */
export interface Snapshot {
    entries: Entry[];
    /** What orangu is doing now, or how the last run went. */
    status: string;
    busy: boolean;
    /** The attached selection, as `src/a.ts:3-9`. */
    selection: string | undefined;
}

/** What changed: the whole state, or just more of the answer. */
export type Change = { type: 'reset'; snapshot: Snapshot } | { type: 'answer'; text: string };

/**
 * The panel's state, apart from VS Code: the prompts and their answers, the
 * status line, whether a run is going, and the attached selection.
 */
export class Transcript {
    private entries: Entry[] = [];
    private status = '';
    private busy = false;
    private selection: string | undefined;

    constructor(private readonly listener: (change: Change) => void = () => undefined) {}

    snapshot(): Snapshot {
        return {
            entries: this.entries.map((entry) => ({ ...entry })),
            status: this.status,
            busy: this.busy,
            selection: this.selection,
        };
    }

    get running(): boolean {
        return this.busy;
    }

    /** A run starts: its prompt is shown, and the status says so. */
    begin(prompt: string): void {
        this.entries.push({ kind: 'prompt', text: prompt });
        this.busy = true;
        this.status = 'Starting orangu…';
        this.reset();
    }

    /** More of the answer. */
    answer(text: string): void {
        const last = this.entries[this.entries.length - 1];
        if (last?.kind === 'answer') {
            last.text += text;
            this.listener({ type: 'answer', text });
        } else {
            this.entries.push({ kind: 'answer', text });
            this.reset();
        }
    }

    /** A line about what orangu is doing. */
    note(line: string): void {
        this.status = line;
        this.reset();
    }

    /** The run ended: the status says how. */
    end(result: RunResult, elapsedMs: number): void {
        this.busy = false;
        this.status = endStatus(result, elapsedMs);
        this.reset();
    }

    /** Something went wrong before orangu could run. */
    fail(message: string): void {
        this.status = message;
        this.reset();
    }

    attach(label: string | undefined): void {
        this.selection = label;
        this.reset();
    }

    /** Empties the panel; a run that is going keeps going into a new answer. */
    clear(): void {
        this.entries = [];
        if (!this.busy) {
            this.status = '';
        }
        this.reset();
    }

    private reset(): void {
        this.listener({ type: 'reset', snapshot: this.snapshot() });
    }
}

/** The status line once a run is over. */
export function endStatus(result: RunResult, elapsedMs: number): string {
    switch (result.outcome) {
        case 'succeeded':
            return `Done in ${(elapsedMs / 1000).toFixed(1)}s`;
        case 'failed':
            return `orangu failed: ${result.error ?? `exit code ${result.exitCode}`}`;
        case 'cancelled':
            return 'Stopped';
        case 'notStarted':
            return `Could not start orangu: ${result.error ?? 'unknown error'}`;
    }
}

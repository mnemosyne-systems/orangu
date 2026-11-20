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

const ESCAPE = '\u001b';
const BELL = '\u0007';

type State = 'text' | 'escape' | 'csi' | 'osc' | 'oscEscape';

/**
 * Removes terminal escape sequences from streamed text. orangu's own output
 * is plain when it is not on a terminal, but a command it runs (git, for
 * one) may still colour its output. The text arrives in pieces, so a
 * sequence cut between two pieces is held back until the rest of it comes.
 */
export class AnsiFilter {
    private state: State = 'text';

    /** The next piece of text, without its escape sequences. */
    push(piece: string): string {
        let text = '';
        for (const c of piece) {
            switch (this.state) {
                case 'text':
                    if (c === ESCAPE) {
                        this.state = 'escape';
                    } else if (c !== '\r') {
                        text += c;
                    }
                    break;
                case 'escape':
                    // ESC [ starts a control sequence, ESC ] an operating
                    // system command (a title, a link); anything else is a
                    // two-character sequence.
                    this.state = c === '[' ? 'csi' : c === ']' ? 'osc' : 'text';
                    break;
                case 'csi':
                    // Parameters and intermediates run up to a final byte
                    // between '@' and '~'.
                    if (c >= '@' && c <= '~') {
                        this.state = 'text';
                    }
                    break;
                case 'osc':
                    if (c === BELL) {
                        this.state = 'text';
                    } else if (c === ESCAPE) {
                        this.state = 'oscEscape';
                    }
                    break;
                case 'oscEscape':
                    // ESC \ ends it; a stray ESC inside it is dropped.
                    this.state = c === '\\' ? 'text' : 'osc';
                    break;
            }
        }
        return text;
    }
}
